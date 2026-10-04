//! Softmax (out-of-place and in-place).
use super::*;

// === vec_soft_max_f32 (vec.cpp lines 531-560, simplified) ===
// Computes softmax: y[i] = exp(x[i] - max) / sum(exp(x[i] - max))
// Returns the sum before scaling
#[inline]
pub fn vec_soft_max_f32(n: usize, y: &mut [f32], x: &[f32], max: f32) -> f64 {
    debug_assert!(y.len() >= n && x.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_soft_max_f32_avx2(n, y, x, max) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            return unsafe { vec_soft_max_f32_neon(n, y, x, max) };
        }
    }

    // Scalar fallback
    let mut sum = 0.0f64;
    for i in 0..n {
        let val = (x[i] - max).exp();
        y[i] = val;
        sum += val as f64;
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_soft_max_f32_avx2(n: usize, y: &mut [f32], x: &[f32], max: f32) -> f64 {
    use std::arch::x86_64::*;

    let mut i = 0;
    let mut sum = 0.0f64;
    let max_v = _mm256_set1_ps(max);

    // Process 8 at a time (vec.cpp lines 542-550)
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let val = vec_exp_f32_avx2(_mm256_sub_ps(
            _mm256_loadu_ps(x.as_ptr().add(i_step)),
            max_v,
        ));
        _mm256_storeu_ps(y.as_mut_ptr().add(i_step), val);

        // Horizontal sum (vec.cpp lines 546-550)
        let val2 = _mm_add_ps(_mm256_extractf128_ps(val, 1), _mm256_castps256_ps128(val));
        let val2 = _mm_add_ps(val2, _mm_movehl_ps(val2, val2));
        let val2 = _mm_add_ss(val2, _mm_movehdup_ps(val2));
        sum += _mm_cvtss_f32(val2) as f64;
        i = i_step + 8;
    }

    // Leftovers
    for j in i..n {
        let val = (x[j] - max).exp();
        y[j] = val;
        sum += val as f64;
    }

    sum
}

/// In-place softmax: y[i] = exp(y[i] - max) / Σ. The SIMD paths load 4/8
/// elements before storing the same range, so operating on one buffer is safe.
/// (Avoids an `&`/`&mut` alias of the same buffer in the caller.)
#[inline]
pub fn vec_soft_max_inplace_f32(n: usize, y: &mut [f32], max: f32) -> f64 {
    debug_assert!(y.len() >= n);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            return unsafe { super::neon::vec_soft_max_f32_inplace(n, y, max) };
        }
    }
    let mut sum = 0.0f64;
    for i in 0..n {
        let val = (y[i] - max).exp();
        y[i] = val;
        sum += val as f64;
    }
    sum
}
