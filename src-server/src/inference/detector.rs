use super::{FaceBox, FaceCrop, InferenceError, InferenceResult};
use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

/// Input size for SCRFD face detection model
pub const SCRFD_INPUT_SIZE: usize = 640;

/// Normalization constants for SCRFD model (matching Python blobFromImage)
/// Python: (val - 127.5) / 128.0, swapRB=True
pub const SCRFD_MEAN: [f32; 3] = [127.5, 127.5, 127.5];
pub const SCRFD_STD: [f32; 3] = [128.0, 128.0, 128.0];

/// Minimum confidence threshold for face detection
pub const SCORE_THRESHOLD: f32 = 0.07;

/// IoU threshold for Non-Maximum Suppression
pub const NMS_IOU_THRESHOLD: f32 = 0.5;

/// SCRFD face detector using ONNX Runtime
#[derive(Debug)]
pub struct Detector {
    session: Option<Session>,
}

impl Default for Detector {
    fn default() -> Self {
        Self::new()
    }
}

impl Detector {
    pub fn new() -> Self {
        Self { session: None }
    }

    /// Detect faces in image bytes
    pub fn detect(&mut self, image_data: &[u8]) -> InferenceResult<Vec<FaceBox>> {
        let img = ImageData::from_bytes(image_data)?;
        self.detect_from_image(&img)
    }

    /// Detect faces from pre-loaded ImageData
    pub fn detect_from_image(&mut self, img: &ImageData) -> InferenceResult<Vec<FaceBox>> {
        let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;

        let (orig_h, orig_w) = (img.height, img.width);
        let (input_tensor, scale, pad_x, pad_y) = preprocess_image(img);

        let outputs = session.run(inputs!["input" => TensorRef::from_array_view(&input_tensor)?])?;

        Ok(decode_scrfd_outputs(outputs, orig_h, orig_w, scale, pad_x, pad_y))
    }

    /// Crop face region from image with padding
    #[allow(dead_code)]
    pub fn crop_face(&self, image_data: &[u8], bbox: &FaceBox) -> InferenceResult<FaceCrop> {
        let img = ImageData::from_bytes(image_data)?;
        let padding = 20usize;

        let x1 = (bbox.x1 as usize).saturating_sub(padding);
        let y1 = (bbox.y1 as usize).saturating_sub(padding);
        let x2 = (bbox.x2 as usize + padding).min(img.width);
        let y2 = (bbox.y2 as usize + padding).min(img.height);

        if x2 <= x1 || y2 <= y1 {
            return Err(InferenceError::NoFace);
        }

        let cropped = img.crop(x1, y1, x2 - x1, y2 - y1)?;

        Ok(FaceCrop {
            data: cropped.to_rgb().data,
            width: x2 - x1,
            height: y2 - y1,
            channels: 3,
        })
    }
}

impl_onnx_model!(Detector);

/// Preprocess image for SCRFD model, matching Python blobFromImage
/// Keeps aspect ratio, pads to 640x640 with fill value (mean)
/// Returns (tensor, scale, pad_x, pad_y)
fn preprocess_image(img: &ImageData) -> (Array4<f32>, f32, f32, f32) {
    let (src_h, src_w) = (img.height as f32, img.width as f32);
    // Compute scale to fit 640x640 while keeping aspect ratio
    let scale = (SCRFD_INPUT_SIZE as f32 / src_w).min(SCRFD_INPUT_SIZE as f32 / src_h);
    let new_w = (src_w * scale).round() as usize;
    let new_h = (src_h * scale).round() as usize;
    // Padding offset (centered)
    let pad_x = ((SCRFD_INPUT_SIZE - new_w) / 2) as f32;
    let pad_y = ((SCRFD_INPUT_SIZE - new_h) / 2) as f32;

    let bgr = img.to_bgr();
    let data = &bgr.data;

    let mut input = Array4::<f32>::zeros((1, 3, SCRFD_INPUT_SIZE, SCRFD_INPUT_SIZE));

    for y in 0..SCRFD_INPUT_SIZE {
        for x in 0..SCRFD_INPUT_SIZE {
            // Map (x,y) from 640x640 canvas back to source image
            let src_x = ((x as f32 - pad_x) / scale) as i32;
            let src_y = ((y as f32 - pad_y) / scale) as i32;

            for c in 0..3 {
                let val = if src_x >= 0 && src_y >= 0 && src_x < img.width as i32 && src_y < img.height as i32 {
                    let pixel_idx = (src_y as usize * img.width + src_x as usize) * 3;
                    if pixel_idx + c < data.len() {
                        data[pixel_idx + c] as f32
                    } else {
                        SCRFD_MEAN[c]
                    }
                } else {
                    SCRFD_MEAN[c]
                };
                input[[0, c, y, x]] = (val - SCRFD_MEAN[c]) / SCRFD_STD[c];
            }
        }
    }

    // Cast scale to f32 for return
    (input, scale, pad_x, pad_y)
}

/// Generate anchors for a given stride (matching TinyFace Python implementation)
/// Python: np.mgrid[:stride_height, :stride_width][::-1] → y from bottom to top
fn generate_anchors(stride: usize, anchor_total: usize) -> Vec<(f32, f32)> {
    let feat_h = SCRFD_INPUT_SIZE / stride;
    let feat_w = SCRFD_INPUT_SIZE / stride;
    let mut anchors = Vec::with_capacity(feat_h * feat_w * anchor_total);
    // Generate anchors in standard top-to-bottom order
    for i in 0..feat_h {
        let cy = i as f32 * stride as f32;
        for j in 0..feat_w {
            let cx = j as f32 * stride as f32;
            for _ in 0..anchor_total {
                anchors.push((cx, cy));
            }
        }
    }
    anchors
}

/// Decode SCRFD multi-scale outputs
/// SCRFD outputs are distances from anchor centers, not absolute coordinates
fn decode_scrfd_outputs(
    outputs: ort::session::SessionOutputs,
    orig_h: usize,
    orig_w: usize,
    scale: f32,
    pad_x: f32,
    pad_y: f32,
) -> Vec<FaceBox> {
    let mut all_boxes: Vec<FaceBox> = Vec::new();

    // Collect all output arrays
    let mut arrays: Vec<ndarray::Array1<f32>> = Vec::new();
    for (_, value) in outputs {
        match value.try_extract_array::<f32>() {
            Ok(arr) => {
                let data: Vec<f32> = arr.iter().copied().collect();
                arrays.push(ndarray::Array1::from(data));
            }
            Err(_) => {}
        }
    }

    if arrays.len() < 9 {
        return all_boxes;
    }

    // SCRFD: strides 8, 16, 32, each with 2 anchors per cell
    let stride_config = [(8usize, 2usize), (16, 2), (32, 2)];
    let scale_indices = [(0, 3, 6), (1, 4, 7), (2, 5, 8)];

    for (scale_idx, &(score_idx, box_idx, kp_idx)) in scale_indices.iter().enumerate() {
        let (stride, anchor_total) = stride_config[scale_idx];
        let stride_f = stride as f32;
        let scores = &arrays[score_idx];
        let boxes = &arrays[box_idx];
        let kps = &arrays[kp_idx];

        let anchors = generate_anchors(stride, anchor_total);
        let num_proposals = anchors.len();

        for i in 0..num_proposals {
            let score = scores[i];
            if score < SCORE_THRESHOLD {
                continue;
            }

            let (anchor_cx, anchor_cy) = anchors[i];

            // SCRFD bbox output: [l, t, r, b] in feature map space (multiply by stride)
            // Each unit in feature map = stride pixels in 640x640 image
            let l = boxes[i * 4] * stride_f;
            let t = boxes[i * 4 + 1] * stride_f;
            let r = boxes[i * 4 + 2] * stride_f;
            let b = boxes[i * 4 + 3] * stride_f;

            // Convert from 640x640 padded grid to original image coordinates
            let x1 = ((anchor_cx - l - pad_x) / scale).clamp(0.0, orig_w as f32 - 1.0);
            let y1 = ((anchor_cy - t - pad_y) / scale).clamp(0.0, orig_h as f32 - 1.0);
            let x2 = ((anchor_cx + r - pad_x) / scale).clamp(0.0, orig_w as f32 - 1.0);
            let y2 = ((anchor_cy + b - pad_y) / scale).clamp(0.0, orig_h as f32 - 1.0);

            // Extract landmarks: offset from anchor center (also in feature map space)
            let mut landmarks = [[0.0_f32; 5]; 2];
            for j in 0..5 {
                let lx = ((anchor_cx + kps[i * 10 + j * 2] * stride_f) - pad_x) / scale;
                let ly = ((anchor_cy + kps[i * 10 + j * 2 + 1] * stride_f) - pad_y) / scale;
                landmarks[0][j] = lx.clamp(0.0, orig_w as f32 - 1.0);
                landmarks[1][j] = ly.clamp(0.0, orig_h as f32 - 1.0);
            }

            all_boxes.push(FaceBox {
                x1,
                y1,
                x2,
                y2,
                score,
                landmarks,
            });
        }
    }

    all_boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    nms(all_boxes, NMS_IOU_THRESHOLD)
}

fn nms(mut boxes: Vec<FaceBox>, iou_threshold: f32) -> Vec<FaceBox> {
    if boxes.is_empty() {
        return boxes;
    }

    boxes.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    let mut suppressed = vec![false; boxes.len()];
    let mut keep = Vec::new();

    for i in 0..boxes.len() {
        if suppressed[i] {
            continue;
        }
        keep.push(boxes[i].clone());

        for j in (i + 1)..boxes.len() {
            if !suppressed[j] && compute_iou(&boxes[i], &boxes[j]) > iou_threshold {
                suppressed[j] = true;
            }
        }
    }
    keep
}

fn compute_iou(a: &FaceBox, b: &FaceBox) -> f32 {
    let x1 = a.x1.max(b.x1);
    let y1 = a.y1.max(b.y1);
    let x2 = a.x2.min(b.x2);
    let y2 = a.y2.min(b.y2);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    let area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Image data wrapper for ONNX inference
#[derive(Debug, Clone)]
pub struct ImageData {
    pub data: Vec<u8>,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("Failed to load image: {0}")]
    LoadError(String),
    #[error("Invalid image format")]
    InvalidFormat,
    #[error("Invalid crop: requested ({width}x{height}) at ({x},{y}) but image is ({img_width}x{img_height})")]
    InvalidCrop {
        x: usize,
        y: usize,
        width: usize,
        height: usize,
        img_width: usize,
        img_height: usize,
    },
}

impl ImageData {
    pub fn from_bytes(bytes: &[u8]) -> InferenceResult<Self> {
        let img = image::load_from_memory(bytes).map_err(|e| ImageError::LoadError(e.to_string()))?;
        Ok(Self::from_dynamic_image(&img))
    }

    fn from_dynamic_image(img: &image::DynamicImage) -> Self {
        let rgba = img.to_rgba8();
        let (width, height) = rgba.dimensions();
        Self {
            data: rgba.into_raw(),
            width: width as usize,
            height: height as usize,
            channels: 4,
        }
    }

    pub fn to_rgb(&self) -> Self {
        if self.channels == 3 {
            return self.clone();
        }
        if self.channels == 4 {
            let rgba_data = &self.data;
            let mut rgb_pixels: Vec<u8> = Vec::with_capacity(self.width * self.height * 3);
            for i in 0..(self.width * self.height) {
                let rgba_idx = i * 4;
                rgb_pixels.push(rgba_data[rgba_idx]);
                rgb_pixels.push(rgba_data[rgba_idx + 1]);
                rgb_pixels.push(rgba_data[rgba_idx + 2]);
            }
            return Self {
                data: rgb_pixels,
                width: self.width,
                height: self.height,
                channels: 3,
            };
        }
        Self {
            data: self.data.clone(),
            width: self.width,
            height: self.height,
            channels: self.channels,
        }
    }

    pub fn to_bgr(&self) -> Self {
        let rgb = self.to_rgb();
        let mut bgr_data = rgb.data.clone();
        for i in 0..(rgb.width * rgb.height) {
            let idx = i * 3;
            bgr_data.swap(idx, idx + 2);
        }
        Self {
            data: bgr_data,
            width: rgb.width,
            height: rgb.height,
            channels: 3,
        }
    }

    pub fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> InferenceResult<Self> {
        if x + width > self.width || y + height > self.height {
            return Err(ImageError::InvalidCrop {
                x,
                y,
                width,
                height,
                img_width: self.width,
                img_height: self.height,
            }.into());
        }
        let rgb = self.to_rgb();
        let img: image::RgbImage =
            image::ImageBuffer::from_raw(rgb.width as u32, rgb.height as u32, rgb.data)
                .ok_or(ImageError::InvalidFormat)?;
        let cropped =
            image::imageops::crop_imm(&img, x as u32, y as u32, width as u32, height as u32)
                .to_image();
        Ok(Self {
            data: cropped.into_raw(),
            width,
            height,
            channels: 3,
        })
    }
}

/// Convert FaceBox landmarks format [[x0..x4], [y0..y4]] to [[x,y]; 5]
pub fn landmarks_to_array(landmarks: &[[f32; 5]; 2]) -> [[f32; 2]; 5] {
    let mut result = [[0.0f32; 2]; 5];
    for i in 0..5 {
        result[i][0] = landmarks[0][i];
        result[i][1] = landmarks[1][i];
    }
    result
}

impl Default for ImageData {
    fn default() -> Self {
        Self {
            data: Vec::new(),
            width: 0,
            height: 0,
            channels: 3,
        }
    }
}

impl From<ImageError> for InferenceError {
    fn from(err: ImageError) -> Self {
        InferenceError::Image(err.to_string())
    }
}
