//! SiLU and the fused SwiGLU (`silu(gate) * up`).
#[cfg(target_arch = "x86_64")]
use super::vec_exp_f32_avx2;

// === vec_silu_f32 (vec.cpp lines 380-399, vec.h lines 1255-1262) ===
// SiLU activation: x * sigmoid(x) = x / (1 + exp(-x))
#[inline]
pub fn vec_silu_f32(n: usize, y: &mut [f32], x: &[f32]) {
    debug_assert!(y.len() >= n && x.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { vec_silu_f32_avx2(n, y, x) };
            return;
        }
    }

    // Scalar fallback (vec.h line 1255 formula, inlined)
    for i in 0..n {
        y[i] = x[i] / (1.0 + (-x[i]).exp());
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_silu_f32_avx2(n: usize, y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    // Process 8 at a time (vec.cpp lines 387-389)
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
        // ggml_v_silu: x / (1 + exp(-x)) (vec.h lines 1255-1262)
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let neg_x = _mm256_sub_ps(zero, vx);
        let exp_neg_x = vec_exp_f32_avx2(neg_x);
        let one_plus_exp = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(vx, one_plus_exp);
        _mm256_storeu_ps(y.as_mut_ptr().add(i_step), result);
        i = i_step + 8;
    }

    // Leftovers
    for j in i..n {
        y[j] = x[j] / (1.0 + (-x[j]).exp());
    }
}

// === vec_swiglu_f32 ===
// dst[i] = silu(gate[i]) * up[i]  (llama `ggml_swiglu_split` semantics).
// Single pass: avoids the full-size intermediate a silu-then-mul pair needs,
// and is bit-identical to that pair (same formula, same per-element order).
pub fn vec_swiglu_f32(n: usize, dst: &mut [f32], gate: &[f32], up: &[f32]) {
    debug_assert!(dst.len() >= n && gate.len() >= n && up.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { vec_swiglu_f32_avx2(n, dst, gate, up) };
            return;
        }
    }

    // Scalar fallback (same formula as vec_silu_f32 + vec_mul_f32; the multiply
    // is exact, so composing it here stays bit-identical to the AVX2 mul path
    // on machines that have AVX2 but not FMA).
    for i in 0..n {
        dst[i] = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_swiglu_f32_avx2(n: usize, dst: &mut [f32], gate: &[f32], up: &[f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vg = _mm256_loadu_ps(gate.as_ptr().add(i_step));
        let vu = _mm256_loadu_ps(up.as_ptr().add(i_step));
        let one = _mm256_set1_ps(1.0);
        let zero = _mm256_setzero_ps();
        let neg_g = _mm256_sub_ps(zero, vg);
        let exp_neg_g = vec_exp_f32_avx2(neg_g);
        let silu = _mm256_div_ps(vg, _mm256_add_ps(one, exp_neg_g));
        let result = _mm256_mul_ps(silu, vu);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i_step), result);
        i = i_step + 8;
    }

    for j in i..n {
        dst[j] = (gate[j] / (1.0 + (-gate[j]).exp())) * up[j];
    }
}
