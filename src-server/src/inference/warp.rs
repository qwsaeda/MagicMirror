/// 5-point face warp template for inswapper (scaled to 128x128)
pub const INSWAPPER_WARP_TEMPLATE: [[f32; 2]; 5] = [
    [0.36167656, 0.40387734],
    [0.63696719, 0.40235469],
    [0.50019687, 0.56044219],
    [0.38710391, 0.72160547],
    [0.61507734, 0.72034453],
];

/// ARC face 5-point warp template (scaled to 112x112 for embedder)
pub const ARCFACE_WARP_TEMPLATE: [[f32; 2]; 5] = [
    [0.34191607, 0.46157411],
    [0.65653393, 0.45983393],
    [0.500225, 0.64050536],
    [0.37097589, 0.82469196],
    [0.63151696, 0.82325089],
];

/// GFPGAN 5-point warp template (scaled to 512x512 for face enhancement)
pub const GFPGAN_WARP_TEMPLATE: [[f32; 2]; 5] = [
    [0.37691676, 0.46864664],
    [0.62285697, 0.46912813],
    [0.50123859, 0.61331904],
    [0.39308822, 0.72541100],
    [0.61150205, 0.72490465],
];

/// Estimate a similarity transform (rotation + scale + translation) that maps
/// source points to destination points using least-squares.
/// Matches OpenCV's estimateAffinePartial2D (similarity transform).
/// Returns a 2x3 affine matrix [a b tx; c d ty] where a=d, b=-c (similarity).
pub fn estimate_similarity_transform(
    src: &[[f32; 2]],
    dst: &[[f32; 2]],
) -> [[f32; 3]; 2] {
    let n = src.len();
    if n == 0 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    }

    // Build the normal equations for the similarity transform:
    // For each (sx,sy) -> (dx,dy):
    //   [sx, -sy, 1, 0] [a]   [dx]
    //   [sy,  sx, 0, 1] [b] = [dy]
    //                    [tx]
    //                    [ty]
    // Solve (A^T A) x = A^T b
    let mut ata = [[0.0f64; 4]; 4];
    let mut atb = [0.0f64; 4];
    let nf = n as f64;

    // Accumulate sums
    let mut sum_sx2_sy2 = 0.0f64;
    let mut sum_sx = 0.0f64;
    let mut sum_sy = 0.0f64;
    let mut sum_sx_dx_p_sy_dy = 0.0f64;
    let mut sum_sx_dy_m_sy_dx = 0.0f64;
    let mut sum_dx = 0.0f64;
    let mut sum_dy = 0.0f64;

    for i in 0..n {
        let sx = src[i][0] as f64;
        let sy = src[i][1] as f64;
        let dx = dst[i][0] as f64;
        let dy = dst[i][1] as f64;

        sum_sx2_sy2 += sx * sx + sy * sy;
        sum_sx += sx;
        sum_sy += sy;
        sum_sx_dx_p_sy_dy += sx * dx + sy * dy;
        sum_sx_dy_m_sy_dx += sx * dy - sy * dx;
        sum_dx += dx;
        sum_dy += dy;
    }

    // A^T A = [[sum_sx2_sy2, 0,           sum_sx, sum_sy],
    //          [0,           sum_sx2_sy2, -sum_sy, sum_sx],
    //          [sum_sx,      -sum_sy,     nf,      0     ],
    //          [sum_sy,      sum_sx,      0,       nf    ]]
    // A^T b = [sum_sx_dx_p_sy_dy, sum_sx_dy_m_sy_dx, sum_dx, sum_dy]

    ata[0][0] = sum_sx2_sy2; ata[0][1] = 0.0;         ata[0][2] = sum_sx; ata[0][3] = sum_sy;
    ata[1][0] = 0.0;         ata[1][1] = sum_sx2_sy2;  ata[1][2] = -sum_sy; ata[1][3] = sum_sx;
    ata[2][0] = sum_sx;      ata[2][1] = -sum_sy;      ata[2][2] = nf;     ata[2][3] = 0.0;
    ata[3][0] = sum_sy;      ata[3][1] = sum_sx;       ata[3][2] = 0.0;    ata[3][3] = nf;

    atb[0] = sum_sx_dx_p_sy_dy;
    atb[1] = sum_sx_dy_m_sy_dx;
    atb[2] = sum_dx;
    atb[3] = sum_dy;

    // Solve 4x4 system using Gaussian elimination with partial pivoting
    let mut aug = [[0.0f64; 5]; 4];
    for i in 0..4 {
        for j in 0..4 {
            aug[i][j] = ata[i][j];
        }
        aug[i][4] = atb[i];
    }

    for col in 0..4 {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for row in (col + 1)..4 {
            let val = aug[row][col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }
        if max_val < 1e-12 {
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        }
        // Swap rows
        aug.swap(col, max_row);
        // Eliminate
        let pivot = aug[col][col];
        for row in (col + 1)..4 {
            let factor = aug[row][col] / pivot;
            for j in col..5 {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }

    // Back substitution
    let mut x = [0.0f64; 4];
    for i in (0..4).rev() {
        let mut sum = aug[i][4];
        for j in (i + 1)..4 {
            sum -= aug[i][j] * x[j];
        }
        x[i] = sum / aug[i][i];
    }

    let a = x[0] as f32;
    let b = x[1] as f32;
    let tx = x[2] as f32;
    let ty = x[3] as f32;

    // Similarity transform: [[a, -b, tx], [b, a, ty]]
    [[a, -b, tx], [b, a, ty]]
}

/// Apply affine transform to a single point
pub fn transform_point(m: &[[f32; 3]; 2], x: f32, y: f32) -> (f32, f32) {
    (
        m[0][0] * x + m[0][1] * y + m[0][2],
        m[1][0] * x + m[1][1] * y + m[1][2],
    )
}

/// Invert a 2x3 affine matrix
pub fn invert_affine(m: &[[f32; 3]; 2]) -> [[f32; 3]; 2] {
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    if det.abs() < 1e-8 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    }
    let inv_det = 1.0 / det;
    let a = m[1][1] * inv_det;
    let b = -m[0][1] * inv_det;
    let c = -m[1][0] * inv_det;
    let d = m[0][0] * inv_det;
    let tx = -(a * m[0][2] + b * m[1][2]);
    let ty = -(c * m[0][2] + d * m[1][2]);
    [[a, b, tx], [c, d, ty]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_similarity() {
        let src = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let dst = [[10.0, 10.0], [11.0, 10.0], [10.0, 11.0]];
        let m = estimate_similarity_transform(&src, &dst);
        // Identity + translation (10, 10)
        let (x, y) = transform_point(&m, 0.0, 0.0);
        assert!((x - 10.0).abs() < 0.01 && (y - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_invert_affine() {
        let m = [[2.0, 0.0, 5.0], [0.0, 3.0, 7.0]];
        let inv = invert_affine(&m);
        let (x, y) = transform_point(&inv, 5.0, 7.0);
        assert!((x - 0.0).abs() < 0.01 && (y - 0.0).abs() < 0.01);
    }
}