use std::path::Path;

use ort::session::Session;

use crate::inference::warp::ARCFACE_WARP_TEMPLATE;

/// Create a session with default execution provider
pub fn create_session(model_path: impl AsRef<Path>) -> InferenceResult<Session> {
    let session = ort::session::Session::builder()?
        .commit_from_file(model_path.as_ref())?;
    Ok(session)
}

/// Type alias for inference operation results
pub type InferenceResult<T> = Result<T, InferenceError>;

#[allow(dead_code)]
/// Type alias for embedding vectors
pub type Embedding = Vec<f32>;

#[allow(dead_code)]
/// Type alias for image data (RGB bytes)
pub type ImageBytes = Vec<u8>;

/// Common trait for all ONNX model components
pub trait OnnxModel: std::fmt::Debug {
    /// Load model from file path
    fn load(&mut self, model_path: impl AsRef<Path>) -> InferenceResult<()>;

    /// Verify model is loaded and ready
    fn prepare(&self) -> InferenceResult<()>;

    /// Get underlying session (used internally)
    #[allow(dead_code)]
    fn session(&self) -> Option<&Session>;
}

/// Base implementation for ONNX model components with shared error handling
#[macro_export]
macro_rules! impl_onnx_model {
    ($struct_name:ident) => {
        impl $crate::inference::OnnxModel for $struct_name {
            fn load(&mut self, model_path: impl AsRef<std::path::Path>) -> $crate::inference::InferenceResult<()> {
                let session = $crate::inference::create_session(model_path)?;
                self.session = Some(session);
                Ok(())
            }

            fn prepare(&self) -> $crate::inference::InferenceResult<()> {
                self.session.as_ref().ok_or($crate::inference::InferenceError::NotLoaded)?;
                Ok(())
            }

            fn session(&self) -> Option<&ort::session::Session> {
                self.session.as_ref()
            }
        }
    };
}

#[derive(Debug, thiserror::Error)]
pub enum InferenceError {
    #[error("ONNX error: {0}")]
    Onnx(#[from] ort::Error),
    #[error("Image error: {0}")]
    Image(String),
    #[error("No face detected")]
    NoFace,
    #[error("Model not loaded")]
    NotLoaded,
}

#[derive(Debug, Clone)]
pub struct FaceBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub score: f32,
    #[allow(dead_code)]
    pub landmarks: [[f32; 5]; 2],
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct FaceCrop {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

/// Main orchestrator for face swap pipeline
#[derive(Debug)]
pub struct TinyFace {
    detector: detector::Detector,
    embedder: embedder::Embedder,
    swapper: swapper::Swapper,
    enhancer: Option<enhancer::Enhancer>,
}

impl Default for TinyFace {
    fn default() -> Self {
        Self::new()
    }
}

impl TinyFace {
    pub fn new() -> Self {
        Self {
            detector: detector::Detector::new(),
            embedder: embedder::Embedder::new(),
            swapper: swapper::Swapper::new(),
            enhancer: None,
        }
    }

    /// Load all models from directory
    pub fn load_models(&mut self, models_dir: &std::path::Path) -> InferenceResult<()> {
        self.detector.load(models_dir.join("scrfd_2.5g.onnx"))?;
        self.embedder.load(models_dir.join("arcface_w600k_r50.onnx"))?;
        self.swapper.load(models_dir.join("inswapper_128_fp16.onnx"))?;

        // Load the inswapper weight matrix (last initializer from ONNX model)
        let weight_path = models_dir.join("inswapper_weight.bin");
        if weight_path.exists() {
            self.swapper.load_weight(&weight_path)?;
        }

        let enhancer_path = models_dir.join("gfpgan_1.4.onnx");
        if enhancer_path.exists() {
            self.enhancer = Some(enhancer::Enhancer::new());
            self.enhancer.as_mut().unwrap().load(&enhancer_path)?;
        }

        Ok(())
    }

    /// Prepare all models for inference
    pub fn prepare(&mut self) -> InferenceResult<()> {
        self.detector.prepare()?;
        self.embedder.prepare()?;
        self.swapper.prepare()?;
        if let Some(ref mut e) = self.enhancer {
            e.prepare()?;
        }
        Ok(())
    }

    /// Get embedding for cropped face
    #[allow(dead_code)]
    pub fn get_embedding(&mut self, face_crop: &FaceCrop) -> InferenceResult<Vec<f32>> {
        self.embedder.embed(face_crop)
    }

    /// Perform face swap with full pipeline
    /// Uses face landmarks to align source face, transforms embedding, and runs swapper
    pub fn swap_face(
        &mut self,
        input_image: &[u8],
        source_box: &FaceBox,
        target_image: &[u8],
        target_box: &FaceBox,
    ) -> InferenceResult<Vec<u8>> {
        // Load full images
        let source_img = detector::ImageData::from_bytes(input_image)?;
        let target_img = detector::ImageData::from_bytes(target_image)?;

        // Convert landmarks from FaceBox format [[x0..x4], [y0..y4]] to [[x,y]; 5]
        let source_landmarks = crate::inference::detector::landmarks_to_array(&source_box.landmarks);
        let target_landmarks = crate::inference::detector::landmarks_to_array(&target_box.landmarks);

        // Warp source face using inswapper template (128x128)
        let (warped_source, source_affine) = self.swapper.warp_face(&source_img, &source_landmarks)?;

        // Warp target face using ArcFace template for embedding (112x112)
        // ArcFace expects 112x112 aligned face
        let (warped_target, _) = self.swapper.warp_face_with_template(
            &target_img, &target_landmarks, 
            &ARCFACE_WARP_TEMPLATE, 112
        )?;

        // Convert warped target to FaceCrop for embedder
        let target_crop = FaceCrop {
            data: warped_target.data.clone(),
            width: warped_target.width,
            height: warped_target.height,
            channels: 3,
        };

        // Extract embedding
        let embedding = self.embedder.embed(&target_crop)?;

        // Transform embedding with weight matrix
        let transformed = self.swapper.transform_embedding(&embedding)?;

        // Run swapper
        let swapped = self.swapper.swap(&warped_source, &transformed)?;

        // Paste the swapped face back onto the original image
        let mut result = self.swapper.paste_back(&source_img, &swapped, &source_affine, 128);

        // Enhance face quality with GFPGAN (matching Python's swap_face behavior)
        if let Some(ref mut enhancer) = self.enhancer {
            // Warp the SWAPPED result face to 512x512, run GFPGAN, paste back, then blend
            let swapped_img = detector::ImageData {
                data: result.clone(),
                width: source_img.width,
                height: source_img.height,
                channels: 3,
            };
            let (warped_face, enh_affine) = self.swapper.warp_face_with_template(
                &swapped_img, &source_landmarks,
                &crate::inference::warp::GFPGAN_WARP_TEMPLATE, 512,
            )?;
            let enhanced = enhancer.enhance(&warped_face)?;
            let paste_result = self.swapper.paste_back(
                &swapped_img, &enhanced, &enh_affine, 512,
            );
            // Blend: 0.25 * swapped + 0.75 * enhanced (improved over Python's 0.4/0.6 for better sharpness)
            for i in 0..result.len() {
                let orig = result[i] as f32;
                let paste = paste_result[i] as f32;
                result[i] = (0.25 * orig + 0.75 * paste).clamp(0.0, 255.0) as u8;
            }
        }

        Ok(result)
    }
}

impl TinyFace {
    /// Simple swap: detect face, crop, swap (no landmark alignment)
    /// Kept for backward compatibility
    #[allow(dead_code)]
    pub fn swap_face_simple(
        &mut self,
        source_crop: &FaceCrop,
        target_embedding: &[f32],
    ) -> InferenceResult<Vec<u8>> {
        let source_img = detector::ImageData {
            data: source_crop.data.clone(),
            width: source_crop.width,
            height: source_crop.height,
            channels: source_crop.channels,
        };
        let swapped = self.swapper.swap(&source_img, target_embedding)?;
        if let Some(ref mut enhancer) = self.enhancer {
            let swapped_img = detector::ImageData {
                data: swapped,
                width: source_crop.width,
                height: source_crop.height,
                channels: 3,
            };
            enhancer.enhance(&swapped_img)
        } else {
            Ok(swapped)
        }
    }

    /// Get single face crop from image
    #[allow(dead_code)]
    pub fn get_one_face(&mut self, image: &[u8]) -> InferenceResult<FaceCrop> {
        let boxes = self.detector.detect(image)?;
        if boxes.is_empty() {
            return Err(InferenceError::NoFace);
        }
        self.detector.crop_face(image, &boxes[0])
    }

    /// Detect faces in image, returns face boxes with landmarks
    pub fn detect_faces(&mut self, image: &[u8]) -> InferenceResult<Vec<FaceBox>> {
        self.detector.detect(image)
    }
}

// Re-export submodules
pub mod detector;
pub mod embedder;
pub mod swapper;
pub mod enhancer;
pub mod warp;


