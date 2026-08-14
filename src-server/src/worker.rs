use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{mpsc, Mutex};
use tracing::{error, info};

use crate::inference::TinyFace;

/// Configuration for the worker pool
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// Number of workers (None = auto-detect, Some(n) = use n workers)
    pub num_workers: Option<usize>,
    /// Models directory
    pub models_dir: PathBuf,
}

/// Task to be processed by a worker
pub struct Task {
    pub id: String,
    pub input_image: Vec<u8>,
    pub target_image: Vec<u8>,
    /// Path of the target face image (used for output path)
    pub target_face_path: String,
    pub sender: mpsc::Sender<TaskResult>,
}

/// Result of task processing
pub struct TaskResult {
    pub id: String,
    pub result: Result<String, String>,
}

/// Calculate optimal number of workers based on CPU cores and memory
pub fn calculate_auto_workers() -> usize {
    // CPU cores is the primary limit (compute-intensive task)
    let cpu_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    
    // Memory limit (secondary check)
    let total_memory_gb = sys_info::mem_info()
        .map(|m| m.total as f64 / 1024.0 / 1024.0)
        .unwrap_or(16.0);
    
    // Shared model size (approximate)
    let shared_model_size_mb = 774.0;
    
    // Per-worker overhead during inference
    let worker_peak_overhead_mb = 300.0;
    
    // Reserve 2 GB for safety
    let safety_margin_mb = 2048.0;
    
    // Available memory for workers
    let available_mb = total_memory_gb * 1024.0 - shared_model_size_mb - safety_margin_mb;
    let max_by_memory = (available_mb / worker_peak_overhead_mb) as usize;
    
    // Worker count limited by CPU cores (compute-intensive)
    let calculated = cpu_cores.min(max_by_memory).max(1);
    
    info!("Auto-calculated workers: {} (CPU cores: {}, memory: {:.0}MB available)", 
          calculated, cpu_cores, available_mb);
    
    calculated
}

/// Spawn worker tasks
pub async fn spawn_workers(
    config: WorkerConfig,
    rx: mpsc::Receiver<Task>,
    prepared_tx: tokio::sync::mpsc::Sender<bool>,
) -> anyhow::Result<usize> {
    let num_workers = config.num_workers.unwrap_or_else(calculate_auto_workers);
    
    info!("Starting {} worker(s)...", num_workers);
    
    // Create shared tinyface wrapped in Mutex for interior mutability
    let tinyface = Arc::new(Mutex::new(TinyFace::new()));
    
    // Load models once
    {
        let mut face = tinyface.lock().await;
        face.load_models(&config.models_dir)?;
        face.prepare()?;
        info!("Models loaded successfully");
    }
    
    // Notify main that models are loaded
    let _ = prepared_tx.send(true).await;
    
    // Spawn a single worker that processes tasks sequentially
    tokio::spawn(async move {
        worker_loop(rx, tinyface).await;
    });
    
    Ok(num_workers)
}

async fn worker_loop(
    mut rx: mpsc::Receiver<Task>,
    tinyface: Arc<Mutex<TinyFace>>,
) {
    info!("Worker started");
    
    while let Some(task) = rx.recv().await {
        info!("Processing task: {}", task.id);
        
        // Process task
        let result = process_task(task.id.clone(), task.input_image, task.target_image, &task.target_face_path, &tinyface).await;
        
        // Send result back
        if let Err(e) = task.sender.send(TaskResult {
            id: task.id,
            result,
        }).await {
            error!("Failed to send result: {}", e);
        }
    }
    
    info!("Worker finished");
}

async fn process_task(
    id: String,
    input_image: Vec<u8>,
    target_image: Vec<u8>,
    target_face_path: &str,
    tinyface: &Arc<Mutex<TinyFace>>,
) -> Result<String, String> {
    use image::GenericImageView;
    
    // Lock tinyface for inference
    let mut face = tinyface.lock().await;
    
    // Read images
    info!("Detecting face in input image...");
    let source_boxes = face.detect_faces(&input_image).map_err(|e| format!("Detection failed: {e}"))?;
    if source_boxes.is_empty() {
        return Err("No face detected in input image".to_string());
    }
    
    info!("Detecting face in target image...");
    let target_boxes = face.detect_faces(&target_image).map_err(|e| format!("Detection failed: {e}"))?;
    if target_boxes.is_empty() {
        return Err("No face detected in target image".to_string());
    }
    
    info!("Performing face swap...");
    let swapped = face.swap_face(&input_image, &source_boxes[0], &target_image, &target_boxes[0])
        .map_err(|e| format!("Swap failed: {e}"))?;
    
    drop(face); // Release lock before I/O
    
    // Save output to same directory as target face image (second image)
    // Add timestamp if file already exists to avoid overwriting
    info!("Target face path: {}", target_face_path);
    let target_path = std::path::Path::new(target_face_path);
    let output_dir = target_path.parent().unwrap_or_else(|| std::path::Path::new("."));
    info!("Output directory: {}", output_dir.display());
    let base_name = target_path.file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    info!("Output base name: {}", base_name);
    
    let mut output_path = output_dir.join(format!("{base_name}_output.jpg"));
    if output_path.exists() {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis();
        output_path = output_dir.join(format!("{base_name}_output_{timestamp}.jpg"));
    }
    
    // Decode and save
    let img = image::load_from_memory(&input_image)
        .map_err(|e| format!("Failed to load input: {e}"))?;
    let (width, height) = img.dimensions();
    let rgb_image = image::RgbImage::from_raw(width, height, swapped)
        .ok_or("Failed to create RGB image".to_string())?;
    
    let mut file = std::fs::File::create(&output_path)
        .map_err(|e| format!("Failed to create output: {e}"))?;
    let mut encoder = jpeg_encoder::Encoder::new(&mut file, 95);
    encoder.set_sampling_factor(jpeg_encoder::SamplingFactor::R_4_2_0);
    encoder.encode(rgb_image.as_raw(), width as u16, height as u16, jpeg_encoder::ColorType::Rgb)
        .map_err(|e| format!("Failed to encode: {e}"))?;
    
    info!("Task {} completed: {}", id, output_path.display());
    Ok(output_path.to_string_lossy().to_string())
}
