use axum::{
    extract::{Path, State},
    http::StatusCode,
    routing::{delete, get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{error, info};

mod inference;
mod worker;

/// Application state shared across handlers
#[derive(Clone)]
struct AppState {
    /// Task sender for worker pool
    task_sender: Arc<tokio::sync::mpsc::Sender<worker::Task>>,
    /// Whether models are loaded and ready
    prepared: Arc<std::sync::atomic::AtomicBool>,
    /// Number of workers
    worker_count: usize,
}

#[derive(Serialize)]
struct StatusResponse {
    status: &'static str,
    workers: usize,
}

#[derive(Serialize)]
struct PrepareResponse {
    success: bool,
}

#[derive(Serialize)]
struct TaskResponse {
    task_id: String,
    result: String,
}

#[derive(Serialize)]
struct CancelResponse {
    success: bool,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct TaskRequest {
    id: String,
    #[serde(alias = "input_image")]
    input_image: String,
    #[serde(alias = "target_face")]
    target_face: String,
}

/// 检测 GPU 类型，返回推荐的后端
/// Detects available GPU and returns recommended backend
#[cfg(target_os = "windows")]
fn detect_gpu_backend() -> &'static str {
    // 使用 WMIC 查询显卡信息
    let output = std::process::Command::new("wmic")
        .args(["path", "win32_VideoController", "get", "Name"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok());
    
    match output {
        Some(ref info) if info.to_lowercase().contains("nvidia") => {
            tracing::info!("Detected NVIDIA GPU, CUDA backend recommended");
            "cuda"
        }
        Some(ref info) if info.to_lowercase().contains("amd") || info.to_lowercase().contains("radeon") => {
            tracing::info!("Detected AMD GPU, DirectML backend recommended");
            "directml"
        }
        Some(_) | None => {
            tracing::info!("No discrete GPU detected or using integrated graphics, DirectML will be used as fallback");
            "directml"
        }
    }
}

#[cfg(not(target_os = "windows"))]
fn detect_gpu_backend() -> &'static str {
    tracing::info!("Non-Windows platform, using CPU backend by default");
    "cpu"
}

/// Parse command line arguments
fn parse_args() -> (Option<usize>,) {
    let args: Vec<String> = std::env::args().collect();
    
    let mut num_workers: Option<usize> = None;
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--workers" | "-w" => {
                if i + 1 < args.len() {
                    i += 1;
                    let val = &args[i];
                    if val == "auto" {
                        num_workers = None; // Use auto-detection
                    } else if let Ok(n) = val.parse::<usize>() {
                        num_workers = Some(n);
                    } else {
                        eprintln!("Invalid worker count: {}", val);
                        std::process::exit(1);
                    }
                } else {
                    eprintln!("--workers requires a value");
                    std::process::exit(1);
                }
            }
            _ => {}
        }
        i += 1;
    }
    
    (num_workers,)
}

/// Determine models directory from executable location or home directory
fn get_models_dir() -> std::path::PathBuf {
    // Check current working directory first
    if let Ok(cwd) = std::env::current_dir() {
        let cwd_models = cwd.join("models");
        if cwd_models.exists() {
            return cwd_models;
        }
    }

    // Check executable directory
    let exe_dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_default();

    if exe_dir.join("models").exists() {
        return exe_dir.join("models");
    }

    // Check home directory
    dirs::home_dir()
        .map(|h| h.join("MagicMirror").join("models"))
        .unwrap_or(exe_dir.join("models"))
}

async fn status(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    let prepared = state.prepared.load(std::sync::atomic::Ordering::Relaxed);
    Json(StatusResponse {
        status: if prepared { "running" } else { "starting" },
        workers: state.worker_count,
    })
}

async fn prepare(State(state): State<AppState>) -> impl axum::response::IntoResponse {
    // Wait for models to be loaded
    let timeout = tokio::time::timeout(std::time::Duration::from_secs(180), async {
        loop {
            if state.prepared.load(std::sync::atomic::Ordering::Relaxed) {
                return Json(PrepareResponse { success: true });
            }
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
        }
    }).await;
    
    match timeout {
        Ok(response) => response,
        Err(_) => Json(PrepareResponse { success: false }),
    }
}

async fn create_task(
    State(state): State<AppState>,
    Json(payload): Json<TaskRequest>,
) -> Result<impl axum::response::IntoResponse, StatusCode> {
    if payload.id.is_empty()
        || payload.input_image.is_empty()
        || payload.target_face.is_empty()
    {
        return Err(StatusCode::BAD_REQUEST);
    }

    if !state.prepared.load(std::sync::atomic::Ordering::Relaxed) {
        return Err(StatusCode::BAD_REQUEST);
    }

    // Read input images
    info!("Reading input image: {}", payload.input_image);
    let input_image =
        std::fs::read(&payload.input_image).map_err(|e| { error!("Read input failed: {e}"); StatusCode::INTERNAL_SERVER_ERROR })?;
    info!("Reading target image: {}", payload.target_face);
    let target_image =
        std::fs::read(&payload.target_face).map_err(|e| { error!("Read target failed: {e}"); StatusCode::INTERNAL_SERVER_ERROR })?;

    // Create a channel for this task result
    let (tx, mut rx) = tokio::sync::mpsc::channel(1);
    
    let task = worker::Task {
        id: payload.id.clone(),
        input_image,
        target_image,
        target_face_path: payload.target_face.clone(),
        sender: tx,
    };

    // Send task to worker pool
    if state.task_sender.send(task).await.is_err() {
        error!("Failed to send task to worker pool");
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }

    // Wait for result with timeout
    match tokio::time::timeout(std::time::Duration::from_secs(120), rx.recv()).await {
        Ok(Some(result)) => match result.result {
            Ok(output_path) => Ok(Json(TaskResponse {
                task_id: result.id,
                result: output_path,
            })),
            Err(e) => {
                error!("Task failed: {e}");
                Err(StatusCode::INTERNAL_SERVER_ERROR)
            }
        },
        Ok(None) => {
            error!("Worker channel closed");
            Err(StatusCode::INTERNAL_SERVER_ERROR)
        }
        Err(_) => {
            error!("Task timeout");
            Err(StatusCode::GATEWAY_TIMEOUT)
        }
    }
}

async fn cancel_task(Path(_task_id): Path<String>) -> impl axum::response::IntoResponse {
    // 当前实现不支持异步任务取消，返回 405 Method Not Allowed
    StatusCode::METHOD_NOT_ALLOWED
}

async fn root_handler() -> &'static str {
    "MagicMirror"
}

#[tokio::main]
async fn main() {
    // 只记录 ERROR 和 WARN 级别，减少日志输出
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .init();

    // 禁用 ONNX Runtime 的详细日志和警告
    std::env::set_var("ORT_LOGGING_LEVEL", "Error");
    std::env::set_var("OMP_WAIT_POLICY", "PASSIVE");  // 减少 OpenMP 线程干扰

    // GPU 自动检测并设置环境变量
    let gpu_backend = detect_gpu_backend();
    match gpu_backend {
        "cuda" => {
            std::env::set_var("ORT_CUDA_AVAILABLE", "1");
            info!("Using CUDA backend (NVIDIA GPU detected)");
        }
        "directml" => {
            std::env::set_var("ORT_DIRECTML_AVAILABLE", "1");
            info!("Using DirectML backend (AMD/Intel GPU or fallback)");
        }
        _ => {
            info!("Using CPU backend (no GPU detected)");
        }
    }

    // Parse command line arguments
    let (num_workers,) = parse_args();
    
    let models_dir = get_models_dir();
    
    info!("Models directory: {:?}", models_dir);
    info!("Worker config: {:?}", num_workers);
    
    // Create task channel (buffer size = number of pending tasks)
    let (task_tx, task_rx) = tokio::sync::mpsc::channel(10);
    
    // Spawn worker pool in background
    let worker_config = worker::WorkerConfig {
        num_workers,
        models_dir: models_dir.clone(),
    };
    
    // Channel to receive prepared notification from workers
    let (prepared_tx, mut prepared_rx) = tokio::sync::mpsc::channel(1);
    
    let prepared_state = Arc::new(std::sync::atomic::AtomicBool::new(false));
    
    // Clone for spawn closure
    let prepared_for_spawn = prepared_state.clone();
    
    tokio::spawn(async move {
        match worker::spawn_workers(worker_config.clone(), task_rx, prepared_tx).await {
            Ok(_worker_count) => {
                info!("Worker pool started");
            }
            Err(e) => {
                error!("Worker pool error: {e}");
            }
        }
    });
    
    // Wait for prepared notification and update state
    tokio::spawn(async move {
        if prepared_rx.recv().await.is_some() {
            prepared_for_spawn.store(true, std::sync::atomic::Ordering::Relaxed);
            info!("Server is ready");
        }
    });
    
    let state = AppState {
        task_sender: Arc::new(task_tx),
        prepared: prepared_state,
        worker_count: num_workers.unwrap_or(1),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([axum::http::Method::GET, axum::http::Method::POST, axum::http::Method::DELETE])
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(root_handler))
        .route("/status", get(status))
        .route("/prepare", post(prepare))
        .route("/task", post(create_task))
        .route("/task/{task_id}", delete(cancel_task))
        .with_state(state)
        .layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8023));
    info!("Starting MagicMirror server on {{addr}}");

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("Failed to bind to port 8023");
    axum::serve(listener, app)
        .await
        .expect("Failed to start server");
}
