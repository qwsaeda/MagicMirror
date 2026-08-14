use super::detector::ImageData;
use super::warp::{estimate_similarity_transform, invert_affine, transform_point, INSWAPPER_WARP_TEMPLATE};
use super::{InferenceError, InferenceResult};
use ndarray::{Array1, Array2, Array4};
use ort::{inputs, session::Session, value::TensorRef};

/// Input size for inswapper model
pub const SWAPPER_INPUT_SIZE: usize = 128;

/// inswapper face swap model
#[derive(Debug)]
pub struct Swapper {
    session: Option<Session>,
    weight: Vec<f32>,
}

impl Default for Swapper {
    fn default() -> Self {
        Self::new()
    }
}

impl Swapper {
    pub fn new() -> Self {
        Self {
            session: None,
            weight: Vec::new(),
        }
    }

    /// Load the source embedding weight matrix from a binary file (512x512 f32)
    pub fn load_weight(&mut self, weight_path: impl AsRef<std::path::Path>) -> InferenceResult<()> {
        let data = std::fs::read(weight_path.as_ref()).map_err(|e| {
            InferenceError::Image(format!("Failed to read weight file: {e}"))
        })?;
        if data.len() != 512 * 512 * 4 {
            return Err(InferenceError::Image(format!(
                "Invalid weight file size: expected {}, got {}",
                512 * 512 * 4,
                data.len()
            )));
        }
        let mut weight = vec![0.0f32; 512 * 512];
        for i in 0..512 * 512 {
            let bytes = [
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ];
            weight[i] = f32::from_le_bytes(bytes);
        }
        self.weight = weight;
        Ok(())
    }

    /// Transform the source embedding using the weight matrix
    /// embedding_out = embedding @ W / ||embedding||
    ///
    /// Uses ndarray for optimized matrix multiplication (512x512)
    pub fn transform_embedding(&self, embedding: &[f32]) -> InferenceResult<Vec<f32>> {
        if self.weight.is_empty() {
            return Err(InferenceError::Image("Weight matrix not loaded".to_string()));
        }
        if embedding.len() != 512 {
            return Err(InferenceError::Image(format!(
                "Invalid embedding length: expected 512, got {}",
                embedding.len()
            )));
        }

        // Use ndarray for optimized matrix-vector multiplication
        let emb = ndarray::Array1::from_vec(embedding.to_vec());
        let weight = ndarray::Array2::from_shape_vec((512, 512), self.weight.clone())
            .map_err(|e| InferenceError::Image(format!("Weight reshape failed: {e}")))?;
        let result = emb.dot(&weight);

        // L2 normalize the result
        let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-8 {
            return Ok(result.iter().map(|x| x / norm).collect());
        }
        Ok(result.to_vec())
    }

    /// Warp the face from the full image using the 5 landmarks to the given template
    /// Returns (warped_image, affine_matrix)
    pub fn warp_face(
        &self,
        image: &ImageData,
        landmarks: &[[f32; 2]; 5],
    ) -> InferenceResult<(ImageData, [[f32; 3]; 2])> {
        self.warp_face_with_template(image, landmarks, &INSWAPPER_WARP_TEMPLATE, SWAPPER_INPUT_SIZE)
    }

    /// Warp face with custom template and size
    pub fn warp_face_with_template(
        &self,
        image: &ImageData,
        landmarks: &[[f32; 2]; 5],
        template: &[[f32; 2]; 5],
        output_size: usize,
    ) -> InferenceResult<(ImageData, [[f32; 3]; 2])> {
        // Compute the template scaled to output_size
        let scaled_template: [[f32; 2]; 5] = template.map(|p| {
            [p[0] * output_size as f32, p[1] * output_size as f32]
        });

        let affine = estimate_similarity_transform(landmarks, &scaled_template);
        let inv = invert_affine(&affine);

        // Warp the image
        let rgb = image.to_rgb();
        let src_w = rgb.width;
        let src_h = rgb.height;
        let src_data = &rgb.data;

        let mut warped = vec![0u8; output_size * output_size * 3];

        for y in 0..output_size {
            for x in 0..output_size {
                let (sx, sy) = transform_point(&inv, x as f32, y as f32);

                // BORDER_REPLICATE: clamp to edges
                let sx_clamped = sx.clamp(0.0, (src_w - 1) as f32);
                let sy_clamped = sy.clamp(0.0, (src_h - 1) as f32);

                let x0 = sx_clamped as usize;
                let y0 = sy_clamped as usize;
                let x1 = (x0 + 1).min(src_w - 1);
                let y1 = (y0 + 1).min(src_h - 1);

                let fx = sx_clamped - x0 as f32;
                let fy = sy_clamped - y0 as f32;

                for c in 0..3 {
                    let p00 = src_data[(y0 * src_w + x0) * 3 + c] as f32;
                    let p10 = src_data[(y0 * src_w + x1) * 3 + c] as f32;
                    let p01 = src_data[(y1 * src_w + x0) * 3 + c] as f32;
                    let p11 = src_data[(y1 * src_w + x1) * 3 + c] as f32;

                    let top = p00 * (1.0 - fx) + p10 * fx;
                    let bottom = p01 * (1.0 - fx) + p11 * fx;
                    let val = top * (1.0 - fy) + bottom * fy;

                    warped[(y * output_size + x) * 3 + c] = (val.clamp(0.0, 255.0)).round() as u8;
                }
            }
        }

        Ok((
            ImageData {
                data: warped,
                width: output_size,
                height: output_size,
                channels: 3,
            },
            affine,
        ))
    }

    /// Paste the swapped face back onto the original image using mask blending
    /// Matches Python: warpAffine mask + face, blend with (1-mask)*orig + mask*face
    pub fn paste_back(
        &self,
        original: &ImageData,
        swapped: &[u8],
        affine: &[[f32; 3]; 2],
        sw: usize,
    ) -> Vec<u8> {
        let (orig_w, orig_h) = (original.width, original.height);

        // Convert original to RGB float
        let mut result = vec![0.0f32; orig_w * orig_h * 3];
        for i in 0..(orig_w * orig_h) {
            let src_idx = if original.channels == 4 { i * 4 } else { i * 3 };
            for c in 0..3 {
                result[i * 3 + c] = original.data[src_idx + c] as f32;
            }
        }

        // Create the static box mask (128x128) matching Python's create_static_box_mask.
        // face_mask_blur=0.3, face_mask_padding=(0,0,0,0):
        //   blur_amount = int(128 * 0.5 * 0.3) = 19, blur_area = max(19//2,1) = 9
        //   box_mask borders (9px) set to 0, then GaussianBlur sigma = 19*0.25 = 4.75
        let blur_amount = (sw as f32 * 0.5 * 0.3) as usize;
        let blur_area = (blur_amount / 2).max(1);
        let mask_sigma = blur_amount as f32 * 0.25;
        let box_mask = create_box_mask(sw, blur_area, mask_sigma);

        // Warp the swapped face and box mask back using the forward affine (original -> template).
        let mut mask = vec![0.0f32; orig_w * orig_h];
        let mut inv_face = vec![0.0f32; orig_w * orig_h * 3];

        for y in 0..orig_h {
            for x in 0..orig_w {
                let (sx, sy) = transform_point(affine, x as f32, y as f32);
                let sx_int = sx as i32;
                let sy_int = sy as i32;

                if sx_int < 0 || sx_int >= sw as i32 || sy_int < 0 || sy_int >= sw as i32 {
                    continue;
                }

                let x0 = sx_int.max(0).min(sw as i32 - 1) as usize;
                let y0 = sy_int.max(0).min(sw as i32 - 1) as usize;
                let x1 = (x0 + 1).min(sw - 1);
                let y1 = (y0 + 1).min(sw - 1);
                let fx = sx - x0 as f32;
                let fy = sy - y0 as f32;

                // Bilinear sample of the blurred box mask
                let m00 = box_mask[y0 * sw + x0];
                let m10 = box_mask[y0 * sw + x1];
                let m01 = box_mask[y1 * sw + x0];
                let m11 = box_mask[y1 * sw + x1];
                let m_top = m00 * (1.0 - fx) + m10 * fx;
                let m_bottom = m01 * (1.0 - fx) + m11 * fx;
                mask[y * orig_w + x] = m_top * (1.0 - fy) + m_bottom * fy;

                for c in 0..3 {
                    let p00 = swapped[(y0 * sw + x0) * 3 + c] as f32;
                    let p10 = swapped[(y0 * sw + x1) * 3 + c] as f32;
                    let p01 = swapped[(y1 * sw + x0) * 3 + c] as f32;
                    let p11 = swapped[(y1 * sw + x1) * 3 + c] as f32;
                    let top = p00 * (1.0 - fx) + p10 * fx;
                    let bottom = p01 * (1.0 - fx) + p11 * fx;
                    inv_face[(y * orig_w + x) * 3 + c] = top * (1.0 - fy) + bottom * fy;
                }
            }
        }

        // Blend: result = mask * inv_face + (1 - mask) * original
        let mut output = vec![0u8; orig_w * orig_h * 3];
        for i in 0..(orig_w * orig_h) {
            let m = mask[i];
            for c in 0..3 {
                let val = m * inv_face[i * 3 + c] + (1.0 - m) * result[i * 3 + c];
                output[i * 3 + c] = val.clamp(0.0, 255.0).round() as u8;
            }
        }

        output
    }

    /// Perform face swap. Takes the warped source face (128x128 RGB) and transformed embedding.
    pub fn swap(
        &mut self,
        source: &ImageData,
        target_embedding: &[f32],
    ) -> InferenceResult<Vec<u8>> {
        if target_embedding.is_empty() {
            return Err(InferenceError::Onnx(ort::Error::new("Empty embedding")));
        }

        let session = self.session.as_mut().ok_or(InferenceError::NotLoaded)?;

        let (input_tensor, input_lro) = prepare_swap_input(source, target_embedding)?;

        let outputs = session.run(inputs![
            "target" => TensorRef::from_array_view(&input_tensor)?,
            "source" => TensorRef::from_array_view(&input_lro)?,
        ])?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| InferenceError::Onnx(ort::Error::new("No outputs")))?;

        let array = output
            .1
            .try_extract_array::<f32>()
            .map_err(InferenceError::Onnx)?;

        // Decode output to BGR image (output range [0, 1], RGB)
        let shape = array.shape();
        if shape.len() < 4 || shape[0] != 1 || shape[1] != 3 {
            return Err(InferenceError::Onnx(ort::Error::new("Invalid output shape")));
        }

        let height = shape[2];
        let width = shape[3];
        let mut image_data = Vec::with_capacity(width * height * 3);

// Model outputs RGB
                for y in 0..height {
                    for x in 0..width {
                        let r = array[[0, 0, y, x]].clamp(0.0, 1.0) * 255.0;
                        let g = array[[0, 1, y, x]].clamp(0.0, 1.0) * 255.0;
                        let b = array[[0, 2, y, x]].clamp(0.0, 1.0) * 255.0;
                        image_data.push(r.round() as u8);
                        image_data.push(g.round() as u8);
                        image_data.push(b.round() as u8);
                    }
                }

        Ok(image_data)
    }
}

/// Create a 128x128 box mask matching Python's create_static_box_mask.
/// Sets border pixels (blur_area) to 0, then applies Gaussian blur.
fn create_box_mask(size: usize, blur_area: usize, sigma: f32) -> Vec<f32> {
    let mut mask = vec![1.0f32; size * size];
    for y in 0..size {
        for x in 0..size {
            if y < blur_area || y >= size - blur_area || x < blur_area || x >= size - blur_area {
                mask[y * size + x] = 0.0;
            }
        }
    }
    if sigma > 0.0 {
        gaussian_blur_2d(&mut mask, size, size, sigma);
    }
    mask
}

/// Simple 2D Gaussian blur via separable convolution.
fn gaussian_blur_2d(data: &mut [f32], width: usize, height: usize, sigma: f32) {
    let radius = (sigma * 3.0).ceil() as usize;
    let kernel_len = 2 * radius + 1;
    let mut kernel = vec![0.0f32; kernel_len];
    let mut sum = 0.0f32;
    for i in 0..kernel_len {
        let x = i as i32 - radius as i32;
        let val = (-(x as f32 * x as f32) / (2.0 * sigma * sigma)).exp();
        kernel[i] = val;
        sum += val;
    }
    for k in kernel.iter_mut() { *k /= sum; }

    let mut tmp = vec![0.0f32; width * height];

    // Horizontal pass
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for kx in 0..kernel_len {
                let sx = (x as i32 + kx as i32 - radius as i32).clamp(0, width as i32 - 1) as usize;
                acc += data[y * width + sx] * kernel[kx];
            }
            tmp[y * width + x] = acc;
        }
    }

    // Vertical pass
    for y in 0..height {
        for x in 0..width {
            let mut acc = 0.0f32;
            for ky in 0..kernel_len {
                let sy = (y as i32 + ky as i32 - radius as i32).clamp(0, height as i32 - 1) as usize;
                acc += tmp[sy * width + x] * kernel[ky];
            }
            data[y * width + x] = acc;
        }
    }
}

impl_onnx_model!(Swapper);

/// Prepare input tensors for face swap inference
/// source: 128x128 RGB image, embedding: 512-dim
fn prepare_swap_input(
    source: &ImageData,
    target_embedding: &[f32],
) -> InferenceResult<(Array4<f32>, Array2<f32>)> {
    let mut input = Array4::<f32>::zeros((1, 3, SWAPPER_INPUT_SIZE, SWAPPER_INPUT_SIZE));
    let mut input_lro = Array2::<f32>::zeros((1, target_embedding.len()));

    // Preprocess source face: RGB, normalize to [0, 1]
    for y in 0..SWAPPER_INPUT_SIZE {
        for x in 0..SWAPPER_INPUT_SIZE {
            let pixel_idx = (y * SWAPPER_INPUT_SIZE + x) * 3;
            for c in 0..3 {
                if pixel_idx + c < source.data.len() {
                    let val = source.data[pixel_idx + c] as f32;
                    input[[0, c, y, x]] = val / 255.0;
                }
            }
        }
    }

    // Copy embedding
    for (i, &val) in target_embedding.iter().enumerate() {
        input_lro[[0, i]] = val;
    }

    Ok((input, input_lro))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swap_not_loaded() {
        let mut swapper = Swapper::new();
        let face = ImageData {
            data: vec![128u8; SWAPPER_INPUT_SIZE * SWAPPER_INPUT_SIZE * 3],
            width: SWAPPER_INPUT_SIZE,
            height: SWAPPER_INPUT_SIZE,
            channels: 3,
        };
        assert!(swapper.swap(&face, &[0.0f32; 512]).is_err());
    }

    #[test]
    fn test_swap_empty_embedding() {
        let mut swapper = Swapper::new();
        let face = ImageData {
            data: vec![128u8; SWAPPER_INPUT_SIZE * SWAPPER_INPUT_SIZE * 3],
            width: SWAPPER_INPUT_SIZE,
            height: SWAPPER_INPUT_SIZE,
            channels: 3,
        };
        assert!(swapper.swap(&face, &[]).is_err());
    }

    #[test]
    fn test_transform_embedding_requires_weight() {
        let swapper = Swapper::new();
        assert!(swapper.transform_embedding(&[0.0f32; 512]).is_err());
    }

    #[test]
    fn test_warp_face() {
        let swapper = Swapper::new();
        let image = ImageData {
            data: vec![128u8; 200 * 200 * 3],
            width: 200,
            height: 200,
            channels: 3,
        };
        let landmarks = [
            [60.0, 80.0],
            [140.0, 80.0],
            [100.0, 120.0],
            [70.0, 150.0],
            [130.0, 150.0],
        ];
        let warped = swapper.warp_face(&image, &landmarks).unwrap();
        assert_eq!(warped.width, SWAPPER_INPUT_SIZE);
        assert_eq!(warped.height, SWAPPER_INPUT_SIZE);
        assert_eq!(warped.data.len(), SWAPPER_INPUT_SIZE * SWAPPER_INPUT_SIZE * 3);
    }
}