mod commands;
mod utils;

use commands::{download_and_unzip, file_exists, get_resource_dir, get_exe_dir, spawn_server, kill_server, is_server_running, kill_spawned_server};

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_os::init())
        .invoke_handler(tauri::generate_handler![
            file_exists,
            download_and_unzip,
            get_resource_dir,
            get_exe_dir,
            spawn_server,
            kill_server,
            is_server_running
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|_app_handle, event| {
        // 任何退出路径（窗口关闭、进程退出等）都终止本进程派生的 server 子进程
        if let tauri::RunEvent::Exit = event {
            kill_spawned_server();
        }
    });
}
