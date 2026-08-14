use std::os::windows::process::CommandExt;
use std::path::Path;
use tauri::{AppHandle, Manager};

use crate::utils::{download_file, unzip_file};

lazy_static::lazy_static! {
    pub static ref SERVER_CHILD: std::sync::Mutex<Option<std::process::Child>> = std::sync::Mutex::new(None);
}

fn lock_server_child() -> std::sync::MutexGuard<'static, Option<std::process::Child>> {
    SERVER_CHILD.lock().unwrap_or_else(|e| e.into_inner())
}

/// 终止本进程派生的 server 子进程（先 kill，再 wait 回收）
pub fn kill_spawned_server() {
    if let Some(mut child) = lock_server_child().take() {
        let _ = child.kill();
        let _ = child.wait();
    }
}

/// 兜底：清理所有遗留的 server.exe 进程（孤儿进程），同步等待完成
fn cleanup_orphan_servers() {
    let _ = std::process::Command::new("taskkill")
        .args(["/f", "/im", "server.exe"])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output();
}

#[tauri::command]
pub fn file_exists(path: String) -> bool {
    Path::new(&path).exists()
}

#[tauri::command]
pub fn get_resource_dir(app: AppHandle) -> String {
    app.path().resource_dir()
        .map(|p| p.to_string_lossy().to_string())
        .unwrap_or_default()
}

#[tauri::command]
pub fn get_exe_dir() -> String {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|pp| pp.to_string_lossy().to_string()))
        .unwrap_or_default()
}

#[tauri::command]
pub async fn spawn_server() -> Result<bool, String> {
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|pp| pp.to_path_buf()))
        .ok_or("Failed to get exe dir")?;

    let server_path = exe_dir.join("server.exe");
    if !server_path.exists() {
        return Err(format!("server.exe not found at {}", server_path.display()));
    }

    // 检查端口是否已被占用
    if check_server_running() {
        if let Some(child) = lock_server_child().as_mut() {
            // 本进程管理过 server：校验是否仍存活
            match child.try_wait() {
                Ok(Some(_)) => { /* 已退出，继续启动新实例 */ }
                Ok(None) => return Ok(true), // 存活中，直接复用
                Err(_) => return Ok(true),
            }
            // 已退出的子进程，清空引用后继续启动新的
            *lock_server_child() = None;
        } else {
            // 端口被占用但占用者不是本进程派生的子进程：视为残留（如旧版残留、
            // 手动启动的控制台进程），先清理再重新启动，保证受控。
            cleanup_orphan_servers();
            // 等待端口释放
            for _ in 0..30 {
                std::thread::sleep(std::time::Duration::from_millis(200));
                if !check_server_running() {
                    break;
                }
            }
        }
    }

    // 启动 server 子进程（无窗口，后台运行）
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    const DETACHED_PROCESS: u32 = 0x00000008;

    // 日志文件：server 输出重定向到 srv_out.log / srv_err.log，便于诊断启动失败
    let out_log = std::fs::OpenOptions::new()
        .create(true).append(true).open(exe_dir.join("srv_out.log"));
    let err_log = std::fs::OpenOptions::new()
        .create(true).append(true).open(exe_dir.join("srv_err.log"));

    let mut cmd = std::process::Command::new(&server_path);
    cmd.arg("--workers").arg("auto")
        .current_dir(&exe_dir)  // 设置工作目录，让 server 找到 models
        .stdin(std::process::Stdio::null())
        .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS);

    match (out_log, err_log) {
        (Ok(o), Ok(e)) => {
            cmd.stdout(std::process::Stdio::from(o)).stderr(std::process::Stdio::from(e));
        }
        _ => {
            cmd.stdout(std::process::Stdio::null()).stderr(std::process::Stdio::null());
        }
    }

    let child = cmd.spawn()
        .map_err(|e| format!("Failed to spawn server: {}", e))?;

    // 保存子进程引用
    *lock_server_child() = Some(child);

    // 等待 server 启动并监听端口（最多 8 秒），校验子进程存活
    let mut started = false;
    for _ in 0..40 {
        std::thread::sleep(std::time::Duration::from_millis(200));
        if check_server_running() {
            started = true;
            break;
        }
        if let Some(c) = lock_server_child().as_mut() {
            if let Ok(Some(_)) = c.try_wait() {
                // 子进程已退出（如模型缺失、端口冲突），清理引用
                *lock_server_child() = None;
                return Err("Server exited unexpectedly during startup".to_string());
            }
        }
    }

    if !started {
        // 超时仍未就绪：杀掉并清理
        kill_spawned_server();
        return Err("Server failed to start in 8s".to_string());
    }

    Ok(true)
}

#[tauri::command]
pub fn kill_server() -> Result<(), String> {
    kill_spawned_server();
    cleanup_orphan_servers();
    Ok(())
}

#[tauri::command]
pub fn is_server_running() -> bool {
    check_server_running()
}

fn check_server_running() -> bool {
    use std::net::TcpStream;
    TcpStream::connect("localhost:8023").is_ok()
}

#[tauri::command]
pub async fn download_and_unzip(
    app: AppHandle,
    url: String,
    target_dir: String,
) -> Result<(), String> {
    let temp_dir = std::env::temp_dir().to_string_lossy().to_string();

    let temp_path = download_file(&app, &url, &temp_dir).await?;

    unzip_file(&app, &temp_path, &target_dir).await?;

    if let Err(e) = std::fs::remove_file(&temp_path) {
        return Err(format!("Failed to remove temp file: {}", e));
    }

    Ok(())
}
