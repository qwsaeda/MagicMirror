use ndarray::Array3;

/// L2 normalize an embedding vector in-place
pub fn l2_normalize(embedding: &mut [f32]) {
    let norm: f32 = embedding.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for v in embedding.iter_mut() {
            *v /= norm;
        }
    }
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(emb1: &[f32], emb2: &[f32]) -> f32 {
    let dot: f32 = emb1.iter().zip(emb2.iter()).map(|(a, b)| a * b).sum();
    let norm1: f32 = emb1.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    let norm2: f32 = emb2.iter().map(|x| x.powi(2)).sum::<f32>().sqrt();
    dot / (norm1 * norm2 + 1e-8)
}

/// Compute Euclidean distance between two embeddings
#[allow(dead_code)]
pub fn compute_distance(emb1: &[f32], emb2: &[f32]) -> f32 {
    emb1.iter()
        .zip(emb2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        .sqrt()
}
