use super::{FaceCrop, InferenceError, InferenceResult};
use ndarray::Array4;
use ort::{inputs, session::Session, value::TensorRef};

/// Input size for ArcFace embedding model
pub const ARCFACE_INPUT_SIZE: usize = 112;

/// ArcFace embedding dimension
#[allow(dead_code)]
pub const EMBEDDING_DIM: usize = 512;

/// ArcFace model normalization constants (RGB, matching Python)
/// Python: crop / 127.5 - 1 == (val - 127.5) / 127.5
const ARCFACE_MEAN: [f32; 3] = [127.5, 127.5, 127.5];
const ARCFACE_STD: [f32; 3] = [127.5, 127.5, 127.5];

/// ArcFace embedding model
#[derive(Debug)]
pub struct Embedder {
    session: Option<Session>,
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder {
    pub fn new() -> Self {
        Self { session: None }
    }

    /// Extract face embedding from cropped face image
    pub fn embed(&mut self, face: &FaceCrop) -> InferenceResult<Vec<f32>> {
        let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;
        let input_tensor = preprocess_face(face);

        // ArcFace model uses "input" as input name
        let outputs = session
            .run(inputs!["input" => TensorRef::from_array_view(&input_tensor)?])?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::Onnx(ort::Error::new("No outputs")))?;

        let array = output
            .1
            .try_extract_array::<f32>()
            .map_err(InferenceError::Onnx)?;

        let mut embedding: Vec<f32> = array.iter().copied().collect();
        normalize_embedding(&mut embedding);

        Ok(embedding)
    }
}

impl_onnx_model!(Embedder);

/// Preprocess face crop for ArcFace model
fn preprocess_face(face: &FaceCrop) -> Array4<f32> {
    let mut input = Array4::<f32>::zeros((1, 3, ARCFACE_INPUT_SIZE, ARCFACE_INPUT_SIZE));
    let (src_h, src_w) = (face.height, face.width);
    let scale_x = ARCFACE_INPUT_SIZE as f32 / src_w as f32;
    let scale_y = ARCFACE_INPUT_SIZE as f32 / src_h as f32;

    for y in 0..ARCFACE_INPUT_SIZE {
        for x in 0..ARCFACE_INPUT_SIZE {
            let src_x = (x as f32 / scale_x) as usize;
            let src_y = (y as f32 / scale_y) as usize;

            if src_x >= src_w || src_y >= src_h {
                continue;
            }

            let pixel_idx = (src_y * src_w + src_x) * 3;
            for c in 0..3 {
                if pixel_idx + c < face.data.len() {
                    let val = face.data[pixel_idx + c] as f32;
                    input[[0, c, y, x]] = (val - ARCFACE_MEAN[c]) / ARCFACE_STD[c];
                }
            }
        }
    }
    input
}

/// L2 normalize embedding vector
fn normalize_embedding(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in embedding.iter_mut() {
            *v /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_embedding() {
        let mut emb = vec![3.0, 4.0, 0.0];
        normalize_embedding(&mut emb);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_preprocess_face() {
        let face = FaceCrop {
            data: vec![128u8; ARCFACE_INPUT_SIZE * ARCFACE_INPUT_SIZE * 3],
            width: ARCFACE_INPUT_SIZE,
            height: ARCFACE_INPUT_SIZE,
            channels: 3,
        };
        let result = preprocess_face(&face);
        assert_eq!(result.shape(), &[1, 3, 112, 112]);
    }
}
