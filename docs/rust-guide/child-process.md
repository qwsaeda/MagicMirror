# 子进程生命周期管理 (Child Process Lifecycle Management)

> 本文档记录 MagicMirror 如何派生、托管、清理 server.exe 子进程的完整设计，以及我们在实现过程中踩过的坑和最终方案。
>
> This document covers the complete design of how MagicMirror spawns, manages, and cleans up the `server.exe` child process, including pitfalls we hit and the final solution.

---

## 1. 目标与问题 / Goal & Problem

### 中文

**目标**：MagicMirror 启动时自动派生 `server.exe` 子进程（无控制台窗口、后台运行），关闭时随主进程一起退出；保证主进程始终拥有受控的 server，不留孤儿。

**问题**：早期版本存在三个关键 bug：
1. **弹出控制台窗口** — server.exe 是控制台程序，直接 spawn 会弹黑框
2. **子进程未及时清理** — 关闭 MagicMirror 后 server.exe 仍在运行
3. **残留接管失效** — 旧版残留的 server.exe 占用端口时，MagicMirror 不接管，导致用户看到两个 server

### English

**Goal**: Auto-spawn `server.exe` as a background child process (no console window) on startup, and clean it up when the app exits. Ensure the main process always owns the server with no orphan processes.

**Problems in early versions**:
1. **Console window pops up** — server.exe is a console app, direct spawn shows a black box
2. **Child not cleaned up** — server.exe keeps running after MagicMirror closes
3. **Orphan takeover fails** — stale server.exe occupying port 8023 is not reclaimed

---

## 2. 架构设计 / Architecture

```
用户双击 MagicMirror.exe
        │
        ▼
LaunchPage 挂载
  └─ useServer().launch()
      └─ invoke("spawn_server")  ← 始终调用，不做 is_server_running 短路
            │
            ▼
      Rust spawn_server()
        ├─ server.exe 不存在 → Err
        ├─ 端口 8023 未被占用 → 派生子进程 (CREATE_NO_WINDOW | DETACHED_PROCESS)
        ├─ 已被本进程子进程占用（且存活）→ Ok(true) 复用
        ├─ 被残留进程占用 → taskkill /f /im server.exe 同步清理 → 等待释放 → 派生新实例
        └─ spawn 后轮询端口 ≤8s，期间 try_wait() 检测子进程异常退出 → 失败则清理并报错
        
退出清理（双保险）：
  1. lib.rs RunEvent::Exit → kill_spawned_server()（覆盖所有退出路径，含 WM_CLOSE）
  2. 前端 Mirror 页 Quit 按钮：先 kill_server() 再 exit(0)
```

---

## 3. 关键实现细节 / Key Implementation Details

### 3.1 无窗口后台启动 / Windowless Background Spawn

```rust
// src-tauri/src/commands.rs
use std::os::windows::process::CommandExt;

const CREATE_NO_WINDOW: u32 = 0x08000000;  // Windows: 不创建控制台窗口
const DETACHED_PROCESS: u32 = 0x00000008;  // 脱离控制台父进程

let child = std::process::Command::new(&server_path)
    .arg("--workers").arg("auto")
    .current_dir(&exe_dir)           // 设置工作目录，让 server 找到 models/
    .stdin(std::process::Stdio::null())
    .stdout(out_log)                 // 重定向到日志文件，便于诊断
    .stderr(err_log)
    .creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS)
    .spawn()?;
```

**注意**：`DETACHED_PROCESS` 单独使用即可保证无控制台，但加上 `CREATE_NO_WINDOW` 更稳妥（MSDN 说明两者组合时 CREATE_NO_WINDOW 被忽略，但行为符合预期）。

### 3.2 子进程引用管理 / Child Reference Management

```rust
lazy_static::lazy_static! {
    pub static ref SERVER_CHILD: std::sync::Mutex<Option<std::process::Child>> = 
        std::sync::Mutex::new(None);
}

// Mutex 锁中毒处理：用 unwrap_or_else 而非静默吞掉
fn lock_server_child() -> std::sync::MutexGuard<'static, Option<std::process::Child>> {
    SERVER_CHILD.lock().unwrap_or_else(|e| e.into_inner())
}
```

### 3.3 启动校验 / Startup Verification

```rust
// 等待 server 启动并监听端口（最多 8 秒）
let mut started = false;
for _ in 0..40 {
    std::thread::sleep(std::time::Duration::from_millis(200));
    
    if check_server_running() {
        started = true;
        break;
    }
    
    // 检测子进程是否异常退出（如模型缺失）
    if let Some(c) = lock_server_child().as_mut() {
        if let Ok(Some(_)) = c.try_wait() {
            *lock_server_child() = None;
            return Err("Server exited unexpectedly during startup".to_string());
        }
    }
}

if !started {
    kill_spawned_server();
    return Err("Server failed to start in 8s".to_string());
}
```

### 3.4 残留清理 / Orphan Cleanup

```rust
/// 兜底：清理所有遗留的 server.exe 进程（孤儿进程），同步等待完成
fn cleanup_orphan_servers() {
    // 使用 .output() 而非 .spawn()，确保同步等待完成
    let _ = std::process::Command::new("taskkill")
        .args(["/f", "/im", "server.exe"])
        .creation_flags(0x08000000) // CREATE_NO_WINDOW
        .output();
}
```

---

## 4. 踩过的坑与修复 / Pitfalls & Fixes

### Bug 1: 前端用 is_server_running 短路，绕过接管逻辑

**症状**：端口被旧版残留 server 占用时，MagicMirror 返回"已运行"，不调用 spawn_server，旧 server 继续弹控制台。

**根因**：前端 `launch()` 先检查 `is_server_running()`，为 true 则直接 prepare，不执行 spawn_server 里的残留清理逻辑。

**修复**：前端始终调用 `spawn_server()`，由 Rust 统一处理接管/清理/启动。

```typescript
// src/services/server.ts - 修复后
async launch(): Promise<boolean> {
  // 始终调用 spawn_server，不短路
  const spawned = await invoke<boolean>("spawn_server");
  if (!spawned) return false;
  // ...
}
```

### Bug 2: React unmount cleanup 误杀刚启动的 server

**症状**：从启动页导航到主页时，server 进程立即被杀，换脸失败。

**根因**：`useServer.ts` 的 useEffect cleanup 在组件卸载时调用 `Server.kill()`。LaunchPage 导航到 MirrorPage 时会触发 LaunchPage 卸载，导致刚启动的 server 被误杀。

**修复**：删除 unmount cleanup。退出清理只由 Rust `RunEvent::Exit` 负责。

```typescript
// src/hooks/useServer.ts - 修复后
export function useServer() {
  // ...
  // 删除以下代码：
  // useEffect(() => {
  //   return () => { Server.kill().catch(console.error); };
  // }, []);
  
  return { status, launch, kill };
}
```

### Bug 3: taskkill 异步执行导致竞态

**症状**：清理残留后立即启动新 server，端口仍未释放，spawn 失败。

**根因**：`cleanup_orphan_servers()` 使用 `.spawn()`（fire-and-forget），taskkill 还没执行完就继续启动。

**修复**：改用 `.output()` 同步等待完成。

---

## 5. 验证清单 / Verification Checklist

启动 MagicMirror 后执行：

```powershell
# 检查进程关系
Get-CimInstance Win32_Process -Filter "Name='server.exe'" | Select-Object ProcessId, ParentProcessId

# 确认：
# - server.exe 的 ParentProcessId == MagicMirror.exe PID
# - server.exe MainWindowTitle 为空（无控制台窗口）
# - netstat 8023 处于 LISTENING
```

关闭时执行：

```powershell
# 正常关闭（WM_CLOSE）后检查
netstat -ano | findstr ":8023"
# 应为空（端口已释放）或只有 TIME_WAIT
```

---

## 6. 代码位置 / Code Locations

| 功能 | 文件 | 行号 |
|------|------|------|
| 子进程定义 | `src-tauri/src/commands.rs:7-9` | SERVER_CHILD |
| spawn_server | `src-tauri/src/commands.rs:52-143` | 完整启动逻辑 |
| kill_server | `src-tauri/src/commands.rs:145-150` | 清理逻辑 |
| RunEvent::Exit | `src-tauri/src/lib.rs:26-33` | 退出兜底 |
| GPU 检测 | `src-server/src/main.rs:62-90` | detect_gpu_backend |

---

## 7. FAQ

**Q: 为什么不用 tauri_plugin_shell 的 spawn？**
A: `spawn` 命令需要 shell scope 配置，且无法直接获取 Child 引用。用 std::process::Command 更灵活，能精确控制 creation_flags。

**Q: CREATE_NO_WINDOW 和 DETACHED_PROCESS 有什么区别？**
A: `CREATE_NO_WINDOW` 防止创建控制台窗口；`DETACHED_PROCESS` 让新进程脱离控制台父进程。两者组合时 MSDN 说 CREATE_NO_WINDOW 被忽略，但 DETACHED_PROCESS 单独已足够。

**Q: 为什么不用 Stdio::null() 丢弃输出？**
A: stdout/stderr 重定向到 srv_out.log/srv_err.log，否则 server 启动失败（如模型缺失）时无从诊断。
