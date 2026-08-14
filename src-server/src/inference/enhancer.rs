use super::{InferenceError, InferenceResult};
use crate::inference::detector::ImageData;
use ndarray::{Array1, Array4};
use ort::{inputs, session::Session, value::TensorRef};

/// GFPGAN enhancement model input/output size
pub const GFPGAN_SIZE: usize = 512;

/// GFPGAN face enhancement model
#[derive(Debug)]
pub struct Enhancer {
    session: Option<Session>,
    /// Cache whether the model requires a "weight" input
    has_weight: bool,
}

impl Default for Enhancer {
    fn default() -> Self {
        Self::new()
    }
}

impl Enhancer {
    pub fn new() -> Self {
        Self {
            session: None,
            has_weight: false,
        }
    }

    /// Enhance face image quality using GFPGAN
    pub fn enhance(&mut self, image: &ImageData) -> InferenceResult<Vec<u8>> {
        let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;

        let input_tensor = preprocess_image(image)?;

        // Use cached has_weight value instead of checking on every call
        let has_weight = self.has_weight;
        let weight = Array1::<f64>::from_vec(vec![1.0]);

        let outputs = if has_weight {
            session.run(inputs![
                "input" => TensorRef::from_array_view(&input_tensor)?,
                "weight" => TensorRef::from_array_view(&weight)?
            ])?
        } else {
            session.run(inputs!["input" => TensorRef::from_array_view(&input_tensor)?])?
        };

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::Onnx(ort::Error::new("No outputs")))?;

        let array = output
            .1
            .try_extract_array::<f32>()
            .map_err(InferenceError::Onnx)?;

        decode_output(&array)
    }
}

impl_onnx_model!(Enhancer);

/// Preprocess image for GFPGAN enhancement
/// Expects RGB image, resizes to GFPGAN_SIZE x GFPGAN_SIZE
fn preprocess_image(image: &ImageData) -> InferenceResult<Array4<f32>> {
    let rgb = image.to_rgb();
    let (src_width, src_height) = (rgb.width, rgb.height);
    let image = &rgb.data;

    let mut input = Array4::<f32>::zeros((1, 3, GFPGAN_SIZE, GFPGAN_SIZE));

    for y in 0..GFPGAN_SIZE {
        for x in 0..GFPGAN_SIZE {
            let src_x = (x as f32 * src_width as f32 / GFPGAN_SIZE as f32) as usize;
            let src_y = (y as f32 * src_height as f32 / GFPGAN_SIZE as f32) as usize;

            if src_x >= src_width || src_y >= src_height {
                continue;
            }

            let pixel_idx = (src_y * src_width + src_x) * 3;
            for c in 0..3 {
                if pixel_idx + c < image.len() {
                    let val = image[pixel_idx + c] as f32;
                    input[[0, c, y, x]] = val / 127.5 - 1.0;
                }
            }
        }
    }

    Ok(input)
}

/// Decode GFPGAN output to RGB image bytes
fn decode_output(output: &ndarray::ArrayViewD<'_, f32>) -> InferenceResult<Vec<u8>> {
    let shape = output.shape();
    if shape.len() < 4 || shape[1] != 3 {
        return Err(InferenceError::Onnx(ort::Error::new("Invalid output shape")));
    }

    let height = shape[2];
    let width = shape[3];
    let mut image_data = Vec::with_capacity(width * height * 3);

    for y in 0..height {
        for x in 0..width {
            for c in 0..3 {
                let val = output[[0, c, y, x]];
                let clamped = val.clamp(-1.0, 1.0);
                let pixel = ((clamped + 1.0) * 127.5).round() as u8;
                image_data.push(pixel);
            }
        }
    }

    Ok(image_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_output() {
        let data = vec![0.0f32; 1 * 3 * 64 * 64];
        let array = ndarray::Array4::<f32>::from_shape_vec([1, 3, 64, 64], data)
            .expect("Invalid shape")
            .into_dyn();

        let result = decode_output(&array.view().into_dyn()).unwrap();
        assert_eq!(result.len(), 64 * 64 * 3);

        // All values should be 127 (since (0.0 + 1.0) * 127.5 = 127.5 -> 127)
        for &v in &result {
            assert_eq!(v, 127);
        }
    }

    #[test]
    fn test_preprocess_image_square() {
        // Create a 64x64 RGB image
        let image = ImageData {
            data: vec![128u8; 64 * 64 * 3],
            width: 64,
            height: 64,
            channels: 4,
        };
        let result = preprocess_image(&image).unwrap();
        assert_eq!(result.shape(), &[1, 3, GFPGAN_SIZE, GFPGAN_SIZE]);
    }

    #[test]
    fn test_preprocess_image_invalid_size() {
        // Non-RGB data (not divisible by 3)
        let image = ImageData {
            data: vec![128u8; 100],
            width: 10,
            height: 10,
            channels: 4,
        };
        let result = preprocess_image(&image);
        assert!(result.is_ok());
    }
}
