//! RMSNorm, plain and fused with the gain multiply.
use super::*;

// === rms_norm_f32 (ops.cpp lines 3757-3817) ===
// y[i] = x[i] * rsqrt(mean(x²) + eps)
// where mean(x²) = sum(x[i]²) / n
#[inline]
pub fn rms_norm_f32(n: usize, y: &mut [f32], x: &[f32], eps: f32) {
    debug_assert!(y.len() >= n && x.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { rms_norm_f32_avx2(n, y, x, eps) };
            return;
        }
    }

    // Scalar fallback
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        sum_sq += (x[i] as f64) * (x[i] as f64);
    }
    let mean = (sum_sq / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();

    if y.as_ptr() != x.as_ptr() {
        vec_cpy_f32(n, y, x);
    }
    vec_scale_f32(n, y, scale);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn rms_norm_f32_avx2(n: usize, y: &mut [f32], x: &[f32], eps: f32) {
    use std::arch::x86_64::*;

    // Compute sum of squares with AVX2
    let mut sum_sq = 0.0f64;
    let mut i = 0;
    let np = n & !7;

    if np > 0 {
        let mut sum_vec = _mm256_setzero_ps();
        for i_step in (0..np).step_by(8) {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
            sum_vec = _mm256_fmadd_ps(vx, vx, sum_vec);
        }

        // Horizontal reduction
        let mut res = _mm256_extractf128_ps(sum_vec, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(sum_vec));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        sum_sq = _mm_cvtss_f32(res) as f64;
        i = np;
    }

    // Leftovers
    for j in i..n {
        sum_sq += (x[j] as f64) * (x[j] as f64);
    }

    let mean = (sum_sq / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();

    // Copy and scale
    if y.as_ptr() != x.as_ptr() {
        vec_cpy_f32(n, y, x);
    }
    vec_scale_f32(n, y, scale);
}

// === rms_norm_fused_f32 ===
// Fused RMSNorm + weight multiply: y[i] = x[i] * scale * w[i]
// Avoids materializing intermediate normalized result
#[inline]
pub fn rms_norm_fused_f32(n: usize, y: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    debug_assert!(y.len() >= n && x.len() >= n && w.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { rms_norm_fused_f32_avx2(n, y, x, w, eps) };
            return;
        }
    }

    // Scalar fallback
    let mut sum_sq = 0.0f64;
    for i in 0..n {
        sum_sq += (x[i] as f64) * (x[i] as f64);
    }
    let mean = (sum_sq / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();

    for i in 0..n {
        y[i] = x[i] * scale * w[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn rms_norm_fused_f32_avx2(n: usize, y: &mut [f32], x: &[f32], w: &[f32], eps: f32) {
    use std::arch::x86_64::*;

    // Compute sum of squares with AVX2
    let mut sum_sq = 0.0f64;
    let mut i = 0;
    let np = n & !7;

    if np > 0 {
        let mut sum_vec = _mm256_setzero_ps();
        for i_step in (0..np).step_by(8) {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
            sum_vec = _mm256_fmadd_ps(vx, vx, sum_vec);
        }

        // Horizontal reduction
        let mut res = _mm256_extractf128_ps(sum_vec, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(sum_vec));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        sum_sq = _mm_cvtss_f32(res) as f64;
        i = np;
    }

    // Leftovers
    for j in i..n {
        sum_sq += (x[j] as f64) * (x[j] as f64);
    }

    let mean = (sum_sq / n as f64) as f32;
    let scale = 1.0 / (mean + eps).sqrt();
    let scale_v = _mm256_set1_ps(scale);

    // Fused scale × weight multiply
    let mut i = 0;
    let np = n & !7;
    if np > 0 {
        for i_step in (0..np).step_by(8) {
            let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
            let vw = _mm256_loadu_ps(w.as_ptr().add(i_step));
            let vy = _mm256_mul_ps(_mm256_mul_ps(vx, scale_v), vw);
            _mm256_storeu_ps(y.as_mut_ptr().add(i_step), vy);
        }
        i = np;
    }

    // Leftovers
    for j in i..n {
        y[j] = x[j] * scale * w[j];
    }
}
