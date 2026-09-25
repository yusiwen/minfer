// SIMD Vector Operations + Core Ops

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum RopeStyle {
    NonInterleaved = 0, // Qwen2: pairs [i, i+half_dim]
    /// LLaMA/Mistral: pairs [2*i, 2*i+1] — kept for llama.cpp parity; no
    /// supported architecture uses it yet.
    #[allow(dead_code)]
    Interleaved = 1, // LLaMA/Mistral: pairs [2*i, 2*i+1]
}

// === vec_dot_f32 (vec.cpp lines 11-137) ===
// Compute dot product of two f32 vectors
// Uses AVX2 FMA when available
#[inline]
pub fn vec_dot_f32(n: usize, x: &[f32], y: &[f32]) -> f32 {
    debug_assert!(x.len() >= n && y.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_f32_avx2(n, x, y) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            return unsafe { vec_dot_f32_neon(n, x, y) };
        }
    }

    // Scalar fallback
    let mut sumf = 0.0f64;
    for i in 0..n {
        sumf += x[i] as f64 * y[i] as f64;
    }
    sumf as f32
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_f32_avx2(n: usize, x: &[f32], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    let mut i = 0;
    let mut sumf = 0.0f32;

    // Process 8 floats at a time with AVX2 (vec.cpp lines 111-117)
    let np = n & !7; // n & ~(GGML_F32_STEP - 1) where GGML_F32_STEP = 8
    if np > 0 {
        // GGML_F32_ARR = 1 for AVX2 on x86_64
        let mut sum = _mm256_setzero_ps();

        for i_step in (0..np).step_by(8) {
            let ax = _mm256_loadu_ps(x.as_ptr().add(i_step));
            let ay = _mm256_loadu_ps(y.as_ptr().add(i_step));
            sum = _mm256_fmadd_ps(ax, ay, sum);
        }

        // Horizontal reduction (vec.cpp lines 43-49 / hsum_float_8)
        let mut res = _mm256_extractf128_ps(sum, 1);
        res = _mm_add_ps(res, _mm256_castps256_ps128(sum));
        res = _mm_add_ps(res, _mm_movehl_ps(res, res));
        res = _mm_add_ss(res, _mm_movehdup_ps(res));
        sumf += _mm_cvtss_f32(res);
        i = np;
    }

    // Leftovers
    for j in i..n {
        sumf += x[j] * y[j];
    }

    sumf
}

// === vec_exp_f32 (vec.h lines 1215-1252) ===
// AVX2 polynomial approximation of exp(x)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_exp_f32_avx2(x: __m256) -> __m256 {
    // Polynomial approximation constants from vec.h lines 1216-1232
    // Converted from C99 hex float literals
    let r = _mm256_set1_ps(f32::from_bits(0x4B400000)); // 0x1.8p23f = 12582912.0
    let z = _mm256_fmadd_ps(x, _mm256_set1_ps(f32::from_bits(0x3FB8AA3B)), r); // 0x1.715476p+0f
    let n = _mm256_sub_ps(z, r);
    let b = _mm256_fnmadd_ps(
        n,
        _mm256_set1_ps(f32::from_bits(0x35BFBE8E)), // 0x1.7f7d1cp-20f
        _mm256_fnmadd_ps(n, _mm256_set1_ps(f32::from_bits(0x3F317200)), x),
    ); // 0x1.62e4p-1f
    let e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
    let k = _mm256_castsi256_ps(_mm256_add_epi32(
        e,
        _mm256_castps_si256(_mm256_set1_ps(1.0f32)),
    ));
    let c = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), n),
        _mm256_set1_ps(126.0f32),
        _CMP_GT_OQ,
    ));
    let u = _mm256_mul_ps(b, b);
    let j = _mm256_fmadd_ps(
        _mm256_fmadd_ps(
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3C072010)),
                b, // 0x1.0e4020p-7f
                _mm256_set1_ps(f32::from_bits(0x3D2B9F17)),
            ), // 0x1.573e2ep-5f
            u,
            _mm256_fmadd_ps(
                _mm256_set1_ps(f32::from_bits(0x3E2AAF33)),
                b, // 0x1.555e66p-3f
                _mm256_set1_ps(f32::from_bits(0x3EFFFEDB)),
            ),
        ), // 0x1.fffdb6p-2f
        u,
        _mm256_mul_ps(_mm256_set1_ps(f32::from_bits(0x3F7FFFF6)), b),
    ); // 0x1.ffffecp-1f

    if _mm256_movemask_ps(_mm256_castsi256_ps(c)) == 0 {
        return _mm256_fmadd_ps(j, k, k);
    }

    let g = _mm256_and_si256(
        _mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
        _mm256_set1_epi32(-2_113_929_216i32),
    );
    let s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000i32)));
    let s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
    let d = _mm256_castps_si256(_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f32), n),
        _mm256_set1_ps(192.0f32),
        _CMP_GT_OQ,
    ));
    _mm256_or_ps(
        _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
        _mm256_andnot_ps(
            _mm256_castsi256_ps(d),
            _mm256_or_ps(
                _mm256_and_ps(
                    _mm256_castsi256_ps(c),
                    _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1),
                ),
                _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)),
            ),
        ),
    )
}

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
            return unsafe { neon_vec::vec_soft_max_f32_inplace(n, y, max) };
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

// === vec_add_f32 (vec.h lines 89-101) ===
// z[i] = x[i] + y[i]
#[inline]
pub fn vec_add_f32(n: usize, z: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert!(z.len() >= n && x.len() >= n && y.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { vec_add_f32_avx2(n, z, x, y) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            unsafe { vec_add_f32_neon(n, z, x, y) };
            return;
        }
    }

    for i in 0..n {
        z[i] = x[i] + y[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_add_f32_avx2(n: usize, z: &mut [f32], x: &[f32], y: &[f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    // Process 8 at a time (vec.h lines 92-97)
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
        let vy = _mm256_loadu_ps(y.as_ptr().add(i_step));
        let vz = _mm256_add_ps(vx, vy);
        _mm256_storeu_ps(z.as_mut_ptr().add(i_step), vz);
        i = i_step + 8;
    }

    for j in i..n {
        z[j] = x[j] + y[j];
    }
}

// === vec_scale_f32 (scalar multiply) ===
// y[i] = y[i] * scale
#[inline]
pub fn vec_scale_f32(n: usize, y: &mut [f32], scale: f32) {
    debug_assert!(y.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { vec_scale_f32_avx2(n, y, scale) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            unsafe { vec_scale_f32_neon(n, y, scale) };
            return;
        }
    }

    for i in 0..n {
        y[i] *= scale;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_scale_f32_avx2(n: usize, y: &mut [f32], scale: f32) {
    use std::arch::x86_64::*;

    let mut i = 0;
    let scale_v = _mm256_set1_ps(scale);
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vy = _mm256_loadu_ps(y.as_ptr().add(i_step));
        _mm256_storeu_ps(y.as_mut_ptr().add(i_step), _mm256_mul_ps(vy, scale_v));
        i = i_step + 8;
    }

    for j in i..n {
        y[j] *= scale;
    }
}

// === vec_mul_f32 ===
// z[i] = x[i] * y[i]
#[inline]
pub fn vec_mul_f32(n: usize, z: &mut [f32], x: &[f32], y: &[f32]) {
    debug_assert!(z.len() >= n && x.len() >= n && y.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { vec_mul_f32_avx2(n, z, x, y) };
            return;
        }
    }

    for i in 0..n {
        z[i] = x[i] * y[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_mul_f32_avx2(n: usize, z: &mut [f32], x: &[f32], y: &[f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
        let vy = _mm256_loadu_ps(y.as_ptr().add(i_step));
        let vz = _mm256_mul_ps(vx, vy);
        _mm256_storeu_ps(z.as_mut_ptr().add(i_step), vz);
        i = i_step + 8;
    }

    for j in i..n {
        z[j] = x[j] * y[j];
    }
}

// === vec_cpy_f32 ===
// y[i] = x[i]
#[inline]
pub fn vec_cpy_f32(n: usize, y: &mut [f32], x: &[f32]) {
    debug_assert!(y.len() >= n && x.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { vec_cpy_f32_avx2(n, y, x) };
            return;
        }
    }

    y[..n].copy_from_slice(&x[..n]);
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_cpy_f32_avx2(n: usize, y: &mut [f32], x: &[f32]) {
    use std::arch::x86_64::*;

    let mut i = 0;
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
        _mm256_storeu_ps(y.as_mut_ptr().add(i_step), vx);
        i = i_step + 8;
    }

    for j in i..n {
        y[j] = x[j];
    }
}

// === vec_muladd_f32 ===
// y[i] += scale * x[i] for i in 0..n (FMA when AVX2 available)
#[inline]
pub fn vec_muladd_f32(n: usize, y: &mut [f32], x: &[f32], scale: f32) {
    debug_assert!(y.len() >= n && x.len() >= n);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { vec_muladd_f32_avx2(n, y, x, scale) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            unsafe { vec_muladd_f32_neon(n, y, x, scale) };
            return;
        }
    }

    for i in 0..n {
        y[i] += scale * x[i];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_muladd_f32_avx2(n: usize, y: &mut [f32], x: &[f32], scale: f32) {
    use std::arch::x86_64::*;

    let s = _mm256_set1_ps(scale);
    let mut i = 0;
    for i_step in (0..n).step_by(8) {
        if i_step + 7 >= n {
            break;
        }
        let vx = _mm256_loadu_ps(x.as_ptr().add(i_step));
        let vy = _mm256_loadu_ps(y.as_ptr().add(i_step));
        _mm256_storeu_ps(y.as_mut_ptr().add(i_step), _mm256_fmadd_ps(s, vx, vy));
        i = i_step + 8;
    }

    for j in i..n {
        y[j] += scale * x[j];
    }
}

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

// === rope_f32 — strict 1:1 translation of ops.cpp lines 5707-5811 ===
// Applies rotary position embeddings in NEOX style (Qwen2, GPT-NeoX)
// === mat_mul_f32 ===
// Simple f32 matrix multiply: C[n][m] = B[n][k] * A[m][k]^T
// (token-major output, matching minfer's [nt][d] activation convention)
// Uses vec_dot_f32 for each token-output pair
pub fn mat_mul_f32(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
    a: &[f32], // [m, k] — weight rows ([out][in])
    b: &[f32], // [n, k] token-major activations ([nt][d])
) {
    debug_assert!(c.len() >= m * n);
    debug_assert!(a.len() >= m * k);
    debug_assert!(b.len() >= k * n);

    // 8a②: this used to write C[row*n + col] = [m, n] — the output was
    // transposed for every nt > 1. Decode (nt == 1) was accidentally correct,
    // which is why no decode-only test ever caught it.
    for col in 0..n {
        let b_row = &b[col * k..(col + 1) * k];
        let c_row = &mut c[col * m..(col + 1) * m];
        for row in 0..m {
            let a_row = &a[row * k..(row + 1) * k];
            c_row[row] = vec_dot_f32(k, a_row, b_row);
        }
    }
}

// ============================================================
// f16 weights × f32 activations (#141)
// ============================================================
//
// F6 made an f16 GGUF *runnable* on the CPU by decoding one weight row at a
// time; #141 makes that path vectorized. Two properties are load-bearing here
// and both are asserted by the tests below:
//
//   1. `dot_f16_f32` is bit-identical to `vec_dot_f32` over the decoded row, on
//      every path (SIMD and scalar). The half→float conversion is exact, and
//      the SIMD loops run the same FMA tree in the same order as `vec_dot_f32`
//      — only the left operand's load differs (a convert instead of a load).
//   2. because of (1), `mat_mul_f16` may decode each weight row **once** and
//      reuse it across tokens without changing a single value. That is what
//      makes prefill read the 2-byte weight stream once per forward instead of
//      once per token; the F6 per-(row, token) shape is kept behind
//      `MINFER_NO_F16_ROWB=1` as the A/B control.

/// Which implementation `dot_f16_f32` runs on this machine.
///
/// This is the **single dispatch authority** — `dot_f16_f32` matches on it, so
/// the unit gate `f16_dot_uses_the_vectorized_path` cannot pass for the wrong
/// reason: it reads the path *and* observes the counter the SIMD entry point
/// bumps. Reverting the vectorization (dispatching to the scalar dot, or making
/// this report `Scalar`) leaves the counter unmoved and fails the gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum F16DotPath {
    #[cfg(target_arch = "x86_64")]
    Avx2,
    #[cfg(target_arch = "aarch64")]
    Neon,
    Scalar,
}

/// The path `dot_f16_f32` takes with the current feature detection and env.
pub fn f16_dot_path() -> F16DotPath {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
            && is_x86_feature_detected!("f16c")
        {
            return F16DotPath::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            return F16DotPath::Neon;
        }
    }
    F16DotPath::Scalar
}

/// Test-only proof-of-execution for the SIMD f16 code. A gate that only asserts
/// `f16_dot_path() != Scalar` is blind to a `dot_f16_f32` body that calls the
/// scalar function anyway — this counter is bumped by the SIMD entry points
/// themselves (both the dot and the row decode), so that mutation is caught.
/// Thread-local so a parallel test harness cannot bump another test's count.
#[cfg(test)]
thread_local! {
    pub(crate) static F16_SIMD_PATH_CALLS: std::cell::Cell<usize> =
        const { std::cell::Cell::new(0) };
}

/// `MINFER_NO_F16_ROWB=1` keeps F6's per-(row, token) loop shape (the A/B
/// control for the row-blocked prefill; the two are bit-identical).
fn no_f16_rowb() -> bool {
    std::env::var("MINFER_NO_F16_ROWB").map_or(false, |v| v == "1")
}

/// Scalar reference for the f16 dot — the correctness oracle the SIMD paths are
/// compared against, and the fallback with no SIMD. f64 accumulation, mirroring
/// `vec_dot_f32`'s scalar fallback so both agree bit for bit on a scalar box.
pub fn dot_f16_f32_scalar(k: usize, w: &[u8], y: &[f32]) -> f32 {
    debug_assert!(w.len() >= k * 2 && y.len() >= k);
    let mut sumf = 0.0f64;
    for i in 0..k {
        let x = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
        sumf += x as f64 * y[i] as f64;
    }
    sumf as f32
}

/// Vectorized f16 weight row × f32 activations (`k` LE half bits in `w`).
///
/// Bit-identical to `vec_dot_f32(k, &decode(w), y)` when both take their SIMD
/// path — the invariant `mat_mul_f16`'s row blocking relies on.
pub fn dot_f16_f32(k: usize, w: &[u8], y: &[f32]) -> f32 {
    debug_assert!(w.len() >= k * 2 && y.len() >= k);
    match f16_dot_path() {
        #[cfg(target_arch = "x86_64")]
        F16DotPath::Avx2 => unsafe { dot_f16_f32_avx2(k, w, y) },
        #[cfg(target_arch = "aarch64")]
        F16DotPath::Neon => unsafe { neon_f16::dot(k, w, y) },
        F16DotPath::Scalar => dot_f16_f32_scalar(k, w, y),
    }
}

/// Decode one f16 weight row into an f32 row (exact: every f16 value is an f32).
fn decode_f16_row(k: usize, w: &[u8], dst: &mut [f32]) {
    debug_assert!(w.len() >= k * 2 && dst.len() >= k);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("f16c") {
            unsafe { decode_f16_row_avx2(k, w, dst) };
            return;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_vec_enabled() {
            unsafe { neon_f16::decode(k, w, dst) };
            return;
        }
    }
    for i in 0..k {
        dst[i] = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma,f16c")]
unsafe fn dot_f16_f32_avx2(k: usize, w: &[u8], y: &[f32]) -> f32 {
    use std::arch::x86_64::*;
    #[cfg(test)]
    F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
    let mut i = 0usize;
    let mut sum = _mm256_setzero_ps();
    let np = k & !7;
    while i < np {
        let h = _mm_loadu_si128(w.as_ptr().add(i * 2) as *const __m128i);
        let f = _mm256_cvtph_ps(h);
        sum = _mm256_fmadd_ps(f, _mm256_loadu_ps(y.as_ptr().add(i)), sum);
        i += 8;
    }
    // Same horizontal reduction as `vec_dot_f32_avx2` (hsum_float_8).
    let mut res = _mm256_extractf128_ps(sum, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(sum));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    let mut sumf = _mm_cvtss_f32(res);
    for j in i..k {
        sumf += crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * j], w[2 * j + 1]])) * y[j];
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,f16c")]
unsafe fn decode_f16_row_avx2(k: usize, w: &[u8], dst: &mut [f32]) {
    use std::arch::x86_64::*;
    #[cfg(test)]
    F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
    let mut i = 0usize;
    let np = k & !7;
    while i < np {
        let h = _mm_loadu_si128(w.as_ptr().add(i * 2) as *const __m128i);
        _mm256_storeu_ps(dst.as_mut_ptr().add(i), _mm256_cvtph_ps(h));
        i += 8;
    }
    for j in i..k {
        dst[j] = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * j], w[2 * j + 1]]));
    }
}

/// NEON half→float conversion + FMA. `FCVTL`/`FCVTN` are baseline ARMv8-A NEON,
/// so no `+fp16` arithmetic feature is required (the FMA is f32).
#[cfg(target_arch = "aarch64")]
mod neon_f16 {
    use std::arch::aarch64::*;

    #[inline]
    unsafe fn convert8(h: uint16x8_t) -> (float32x4_t, float32x4_t) {
        (
            vcvt_f32_f16(vreinterpret_f16_u16(vget_low_u16(h))),
            vcvt_f32_f16(vreinterpret_f16_u16(vget_high_u16(h))),
        )
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn dot(k: usize, w: &[u8], y: &[f32]) -> f32 {
        #[cfg(test)]
        super::F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
        let mut i = 0usize;
        let mut sum = vdupq_n_f32(0.0);
        while i + 8 <= k {
            let h = vld1q_u16(w.as_ptr().add(i * 2) as *const u16);
            let (lo, hi) = convert8(h);
            // Two 4-wide FMAs per 8 elements — the same registers, order and
            // rounding as `vec_dot_f32_neon` over the decoded row.
            sum = vfmaq_f32(sum, lo, vld1q_f32(y.as_ptr().add(i)));
            sum = vfmaq_f32(sum, hi, vld1q_f32(y.as_ptr().add(i + 4)));
            i += 8;
        }
        let mut acc = vaddvq_f32(sum);
        while i < k {
            acc += crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]])) * y[i];
            i += 1;
        }
        acc
    }

    #[target_feature(enable = "neon")]
    pub(super) unsafe fn decode(k: usize, w: &[u8], dst: &mut [f32]) {
        #[cfg(test)]
        super::F16_SIMD_PATH_CALLS.with(|c| c.set(c.get() + 1));
        let mut i = 0usize;
        while i + 8 <= k {
            let h = vld1q_u16(w.as_ptr().add(i * 2) as *const u16);
            let (lo, hi) = convert8(h);
            vst1q_f32(dst.as_mut_ptr().add(i), lo);
            vst1q_f32(dst.as_mut_ptr().add(i + 4), hi);
            i += 8;
        }
        while i < k {
            dst[i] = crate::block::fp16_to_f32(u16::from_le_bytes([w[2 * i], w[2 * i + 1]]));
            i += 1;
        }
    }
}

/// f16 weight ([m, k], row-major, little-endian half bits) × f32 activations.
///
/// The CPU matmul kernels are integer dots over quantized weights, so an f16
/// weight has no integer form; #141 gives it a vectorized float dot instead.
/// Weights are **never** materialized as f32 — only one row at a time is
/// decoded, and only when the row is reused across tokens.
///
/// Decoding a row once and folding every token into it is what lets the row
/// loop go to the shared worker pool (rows are disjoint outputs, so one worker
/// per contiguous row range is race-free by construction); it is also what
/// keeps prefill's DRAM traffic at the weight stream plus the activation
/// matrix. The two loop shapes are bit-identical per output element (property 1
/// above), so neither `n` nor the worker count changes a value.
pub fn mat_mul_f16(
    m: usize,
    n: usize,
    k: usize,
    c: &mut [f32],
    a: &[u8],  // [m, k] f16 weight rows
    b: &[f32], // [n, k] token-major activations
) {
    debug_assert!(c.len() >= m * n);
    debug_assert!(a.len() >= m * k * 2);
    debug_assert!(b.len() >= k * n);
    if n > 1
        && !no_f16_rowb()
        && crate::kernel::cpu_threads() > 1
        && m >= 2
        && m.saturating_mul(k).saturating_mul(n) >= crate::kernel::MIN_PARALLEL_MACS
    {
        let job = F16MatMulJob {
            m,
            n,
            k,
            a: a.as_ptr(),
            b: b.as_ptr(),
            c: c.as_mut_ptr(),
        };
        // SAFETY: `job` outlives the call; `f16_mm_rows` writes only rows it
        // owns (`c[col*m + r]` for its own `r` range), and `a`/`b`/`c` stay
        // borrowed for the duration. Same contract as `kernel`'s own jobs.
        crate::kernel::par_for(m, &job as *const F16MatMulJob as *const (), f16_mm_rows);
        return;
    }
    if n > 1 && !no_f16_rowb() {
        // Row-outer: one decode per weight row, `n` dots against the token rows.
        let mut row = vec![0f32; k];
        for r in 0..m {
            decode_f16_row(k, &a[r * k * 2..(r + 1) * k * 2], &mut row);
            for col in 0..n {
                c[col * m + r] = vec_dot_f32(k, &row, &b[col * k..(col + 1) * k]);
            }
        }
        return;
    }
    // Decode (F6 shape): the weight row is consumed by exactly one token, so a
    // scratch row would be pure overhead.
    for col in 0..n {
        let b_row = &b[col * k..(col + 1) * k];
        let c_row = &mut c[col * m..(col + 1) * m];
        for r in 0..m {
            c_row[r] = dot_f16_f32(k, &a[r * k * 2..(r + 1) * k * 2], b_row);
        }
    }
}

/// One f16 matmul shared by all pool workers (raw pointers, like `kernel`'s
/// `MmJob`: the borrows live on `mat_mul_f16`'s stack and the pool call waits
/// for every worker before they end).
struct F16MatMulJob {
    m: usize,
    n: usize,
    k: usize,
    a: *const u8,
    b: *const f32,
    c: *mut f32,
}

/// Row kernel: rows `[r0, r1)` of the f16 matmul. SAFETY: `a`/`b`/`c` must stay
/// valid for the pool call; each row `r` is written exactly once by its owner
/// (`c[col*m + r]` for `r ∈ [r0, r1)` are disjoint per worker).
unsafe fn f16_mm_rows(ctx: *const (), r0: usize, r1: usize) {
    let j = &*(ctx as *const F16MatMulJob);
    let (m, n, k) = (j.m, j.n, j.k);
    let mut row = vec![0f32; k];
    for r in r0..r1 {
        let w = std::slice::from_raw_parts(j.a.add(r * k * 2), k * 2);
        decode_f16_row(k, w, &mut row);
        for col in 0..n {
            let b_row = std::slice::from_raw_parts(j.b.add(col * k), k);
            *j.c.add(col * m + r) = vec_dot_f32(k, &row, b_row);
        }
    }
}

// ============================================================
// aarch64 NEON fast paths (f32 vector ops used by attention, norms, etc.)
// ============================================================
#[cfg(target_arch = "aarch64")]
mod neon_vec {
    use std::arch::aarch64::*;

    pub(super) fn enabled() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            std::arch::is_aarch64_feature_detected!("neon")
                && !std::env::var("MINFER_NO_NEON").map_or(false, |v| v == "1")
        })
    }

    pub(super) unsafe fn vec_dot_f32(n: usize, x: &[f32], y: &[f32]) -> f32 {
        let mut i = 0;
        let mut sum = vdupq_n_f32(0.0);
        while i + 4 <= n {
            let ax = vld1q_f32(x.as_ptr().add(i));
            let ay = vld1q_f32(y.as_ptr().add(i));
            sum = vfmaq_f32(sum, ax, ay);
            i += 4;
        }
        let mut acc = vaddvq_f32(sum);
        for &xv in &x[i..n] {
            acc += xv * y[i];
            i += 1;
        }
        acc
    }

    pub(super) unsafe fn vec_soft_max_f32(n: usize, y: &mut [f32], x: &[f32], max: f32) -> f64 {
        // fast exp via 2^(x·log2e): Cephes-style polynomial on the fractional
        // part + exponent-bit scaling. Input clamped to [-88, 0] so exp never
        // overflows and the exponent shift stays in range (scalar expf handles
        // -inf → 0 the same way via the clamp).
        let log2e = vdupq_n_f32(1.4426950408889634f32);
        let max_v = vdupq_n_f32(max);
        let zero = vdupq_n_f32(0.0);
        let cl = vdupq_n_f32(-88.0);
        let c0 = vdupq_n_f32(0.001333355814642844);
        let c1 = vdupq_n_f32(0.009618129107628477);
        let c2 = vdupq_n_f32(0.05550410866482158);
        let c3 = vdupq_n_f32(0.2402265069591007);
        let c4 = vdupq_n_f32(0.6931471805599453);
        let c5 = vdupq_n_f32(1.0);
        let exp_bias = vdupq_n_s32(127 << 23);

        let mut i = 0;
        let mut sum = 0.0f64;
        while i + 4 <= n {
            let a = vmaxq_f32(
                vminq_f32(vsubq_f32(vld1q_f32(x.as_ptr().add(i)), max_v), zero),
                cl,
            );
            let nn = vmulq_f32(a, log2e);
            let nf = vrndmq_f32(nn);
            let f = vsubq_f32(nn, nf);
            let mut p = c0;
            p = vfmaq_f32(c1, p, f);
            p = vfmaq_f32(c2, p, f);
            p = vfmaq_f32(c3, p, f);
            p = vfmaq_f32(c4, p, f);
            p = vfmaq_f32(c5, p, f);
            let e = vshlq_n_s32::<23>(vcvtq_s32_f32(nf));
            let ex = vreinterpretq_f32_s32(vaddq_s32(e, exp_bias));
            let val = vmulq_f32(p, ex);
            vst1q_f32(y.as_mut_ptr().add(i), val);
            sum += (vaddvq_f32(val)) as f64;
            i += 4;
        }
        for &xv in &x[i..n] {
            let val = (xv - max).exp();
            y[i] = val;
            sum += val as f64;
            i += 1;
        }
        sum
    }

    pub(super) unsafe fn vec_soft_max_f32_inplace(n: usize, y: &mut [f32], max: f32) -> f64 {
        let log2e = vdupq_n_f32(1.4426950408889634f32);
        let max_v = vdupq_n_f32(max);
        let zero = vdupq_n_f32(0.0);
        let cl = vdupq_n_f32(-88.0);
        let c0 = vdupq_n_f32(0.001333355814642844);
        let c1 = vdupq_n_f32(0.009618129107628477);
        let c2 = vdupq_n_f32(0.05550410866482158);
        let c3 = vdupq_n_f32(0.2402265069591007);
        let c4 = vdupq_n_f32(0.6931471805599453);
        let c5 = vdupq_n_f32(1.0);
        let exp_bias = vdupq_n_s32(127 << 23);

        let mut i = 0;
        let mut sum = 0.0f64;
        while i + 4 <= n {
            let a = vmaxq_f32(
                vminq_f32(vsubq_f32(vld1q_f32(y.as_ptr().add(i)), max_v), zero),
                cl,
            );
            let nn = vmulq_f32(a, log2e);
            let nf = vrndmq_f32(nn);
            let f = vsubq_f32(nn, nf);
            let mut p = c0;
            p = vfmaq_f32(c1, p, f);
            p = vfmaq_f32(c2, p, f);
            p = vfmaq_f32(c3, p, f);
            p = vfmaq_f32(c4, p, f);
            p = vfmaq_f32(c5, p, f);
            let e = vshlq_n_s32::<23>(vcvtq_s32_f32(nf));
            let ex = vreinterpretq_f32_s32(vaddq_s32(e, exp_bias));
            let val = vmulq_f32(p, ex);
            vst1q_f32(y.as_mut_ptr().add(i), val);
            sum += vaddvq_f32(val) as f64;
            i += 4;
        }
        for j in i..n {
            let val = (y[j] - max).exp();
            y[j] = val;
            sum += val as f64;
        }
        sum
    }

    pub(super) unsafe fn vec_add_f32(n: usize, z: &mut [f32], x: &[f32], y: &[f32]) {
        let mut i = 0;
        while i + 4 <= n {
            vst1q_f32(
                z.as_mut_ptr().add(i),
                vaddq_f32(vld1q_f32(x.as_ptr().add(i)), vld1q_f32(y.as_ptr().add(i))),
            );
            i += 4;
        }
        for j in i..n {
            z[j] = x[j] + y[j];
        }
    }

    pub(super) unsafe fn vec_scale_f32(n: usize, y: &mut [f32], scale: f32) {
        let s = vdupq_n_f32(scale);
        let mut i = 0;
        while i + 4 <= n {
            vst1q_f32(
                y.as_mut_ptr().add(i),
                vmulq_f32(vld1q_f32(y.as_ptr().add(i)), s),
            );
            i += 4;
        }
        for j in i..n {
            y[j] *= scale;
        }
    }

    pub(super) unsafe fn vec_muladd_f32(n: usize, y: &mut [f32], x: &[f32], scale: f32) {
        let s = vdupq_n_f32(scale);
        let mut i = 0;
        while i + 4 <= n {
            let vx = vld1q_f32(x.as_ptr().add(i));
            let vy = vld1q_f32(y.as_ptr().add(i));
            vst1q_f32(y.as_mut_ptr().add(i), vfmaq_f32(vy, s, vx));
            i += 4;
        }
        for j in i..n {
            y[j] += scale * x[j];
        }
    }
}

#[cfg(target_arch = "aarch64")]
use neon_vec::enabled as neon_vec_enabled;
#[cfg(target_arch = "aarch64")]
use neon_vec::{
    vec_add_f32 as vec_add_f32_neon, vec_dot_f32 as vec_dot_f32_neon,
    vec_muladd_f32 as vec_muladd_f32_neon, vec_scale_f32 as vec_scale_f32_neon,
    vec_soft_max_f32 as vec_soft_max_f32_neon,
};

#[cfg(test)]
mod tests {
    use super::*;

    /// The fused pass replaces the silu-then-mul pair, so it must agree with it
    /// bit for bit (the unfused execution is the reference for `Op::SwiGLU`).
    #[test]
    fn swiglu_matches_silu_then_mul() {
        let gate: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.37).collect();
        let up: Vec<f32> = (0..37).map(|i| i as f32 * 0.11 - 2.0).collect();
        let n = gate.len();

        let mut want = vec![0f32; n];
        vec_silu_f32(n, &mut want, &gate);
        let silu = want.clone();
        vec_mul_f32(n, &mut want, &silu, &up);

        let mut got = vec![0f32; n];
        vec_swiglu_f32(n, &mut got, &gate, &up);

        assert_eq!(got, want, "fused swiglu must be bit-identical to silu+mul");
    }

    /// Tail handling: `vec_swiglu_f32`'s vector loop stops before the last
    /// partial chunk, so exercise a non-multiple-of-8 length too.
    #[test]
    fn swiglu_scalar_tail() {
        let gate = [-3.0f32, 0.0, 0.5, 7.25];
        let up = [1.5f32, -2.0, 0.0, 4.0];
        let mut got = [0f32; 4];
        vec_swiglu_f32(4, &mut got, &gate, &up);
        for i in 0..4 {
            let want = (gate[i] / (1.0 + (-gate[i]).exp())) * up[i];
            assert_eq!(got[i], want);
        }
    }

    // === #141: f16 weights × f32 activations ===

    fn f16_bytes(vals: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            out.extend_from_slice(&half::f16::from_f32(v).to_bits().to_le_bytes());
        }
        out
    }

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 0.5
    }

    /// The vectorized dot must agree with the scalar oracle. Dimensions include
    /// non-multiples of 8 so the SIMD tail is exercised too.
    #[test]
    fn f16_dot_matches_the_scalar_oracle() {
        let mut seed = 0x141_c0ffee_u64;
        for k in [1usize, 7, 8, 9, 16, 31, 64, 127, 256] {
            let w: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 8.0).collect();
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
            let wb = f16_bytes(&w);
            let got = dot_f16_f32(k, &wb, &x);
            let want = dot_f16_f32_scalar(k, &wb, &x);
            let tol = (want.abs().max(got.abs())).max(1e-6) * 1e-5;
            assert!(
                (got - want).abs() <= tol,
                "k={k}: vectorized {got} != scalar {want}"
            );
        }
    }

    /// Property (1) of the f16 block: `dot_f16_f32` is bit-identical to
    /// `vec_dot_f32` over the decoded row. This is what licenses the row-blocked
    /// prefill loop; if a future SIMD change reorders the accumulation, the
    /// prefill and decode paths would silently stop agreeing and this fails.
    #[test]
    fn f16_dot_is_bit_identical_to_an_f32_dot_over_the_decoded_row() {
        let mut seed = 0x141_d0d0_u64;
        for k in [8usize, 64, 127, 896] {
            let w: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 16.0).collect();
            let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
            let wb = f16_bytes(&w);
            let mut row = vec![0f32; k];
            decode_f16_row(k, &wb, &mut row);
            // The decode itself is exact.
            for i in 0..k {
                assert_eq!(
                    row[i].to_bits(),
                    crate::block::fp16_to_f32(u16::from_le_bytes([wb[2 * i], wb[2 * i + 1]]))
                        .to_bits(),
                    "k={k} i={i}: decoded row is not the exact f16 value"
                );
            }
            assert_eq!(
                dot_f16_f32(k, &wb, &x).to_bits(),
                vec_dot_f32(k, &row, &x).to_bits(),
                "k={k}: f16 dot != f32 dot over the decoded row"
            );
        }
    }

    /// Property (2): the row-blocked multi-token matmul (prefill) and the
    /// per-token form (decode) must produce the same bits — a prefill of one
    /// token is a decode. `n` therefore never changes a value.
    #[test]
    fn f16_matmul_prefill_matches_decode_bitwise() {
        let mut seed = 0x141_beef_u64;
        let (m, k, n) = (13usize, 96usize, 5usize);
        let wf: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 8.0).collect();
        let w = f16_bytes(&wf);
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

        let mut prefill = vec![0f32; m * n];
        mat_mul_f16(m, n, k, &mut prefill, &w, &b);

        for col in 0..n {
            let mut one = vec![0f32; m];
            mat_mul_f16(m, 1, k, &mut one, &w, &b[col * k..(col + 1) * k]);
            for r in 0..m {
                assert_eq!(
                    prefill[col * m + r].to_bits(),
                    one[r].to_bits(),
                    "col={col} row={r}: prefill != decode"
                );
            }
        }
    }

    /// The worker-pool path must produce the same bits as the direct form. A
    /// wrong chunk boundary (a duplicated or skipped row range) shows up as a
    /// mismatch here. Shape is above `MIN_PARALLEL_MACS` so the pool is used
    /// when the box has more than one CPU.
    #[test]
    fn f16_matmul_pool_matches_the_direct_form_bitwise() {
        let mut seed = 0x141_7007_u64;
        let (m, k, n) = (256usize, 512usize, 8usize);
        assert!(m * k * n >= crate::kernel::MIN_PARALLEL_MACS);
        let wf: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 8.0).collect();
        let w = f16_bytes(&wf);
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 3.0).collect();

        let mut got = vec![0f32; m * n];
        mat_mul_f16(m, n, k, &mut got, &w, &b);

        for col in 0..n {
            let b_row = &b[col * k..(col + 1) * k];
            for r in 0..m {
                let want = dot_f16_f32(k, &w[r * k * 2..(r + 1) * k * 2], b_row);
                assert_eq!(
                    got[col * m + r].to_bits(),
                    want.to_bits(),
                    "col={col} row={r}: pooled != direct"
                );
            }
        }
        eprintln!(
            "#141 f16 matmul pool path: {} threads",
            crate::kernel::cpu_threads()
        );
    }

    /// The vectorized path is the one that actually **runs** where the CPU has
    /// it. Two observations, because either alone can pass for the wrong
    /// reason: `f16_dot_path()` names the dispatch branch, and
    /// `F16_SIMD_PATH_CALLS` is bumped by the SIMD entry points themselves. A
    /// revert that dispatches to the scalar dot (or that removes the feature
    /// check) leaves the counter unmoved and fails here.
    #[test]
    fn f16_dot_uses_the_vectorized_path() {
        let path = f16_dot_path();
        let simd = path != F16DotPath::Scalar;

        let mut seed = 0x141_9a7bu64;
        let k = 64usize;
        let wf: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 4.0).collect();
        let w = f16_bytes(&wf);
        let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed) * 2.0).collect();
        let (m, n) = (4usize, 3usize);
        let wm: Vec<f32> = (0..m * k).map(|_| lcg(&mut seed) * 4.0).collect();
        let wmb = f16_bytes(&wm);
        let b: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed) * 2.0).collect();

        let before = F16_SIMD_PATH_CALLS.with(|c| c.get());
        let _ = dot_f16_f32(k, &w, &x);
        let after_dot = F16_SIMD_PATH_CALLS.with(|c| c.get());
        let mut c = vec![0f32; m * n];
        mat_mul_f16(m, n, k, &mut c, &wmb, &b);
        let after_mm = F16_SIMD_PATH_CALLS.with(|c| c.get());

        if simd {
            assert!(
                after_dot > before,
                "f16_dot_path() reports {path:?} but the SIMD dot never ran \
                 ({before} -> {after_dot})"
            );
            // The row-blocked prefill decodes each weight row through its own
            // SIMD path; reverting only that one must fail here too.
            assert!(
                after_mm > after_dot,
                "f16_dot_path() reports {path:?} but the SIMD row decode never ran \
                 ({after_dot} -> {after_mm})"
            );
        } else {
            assert_eq!(
                (after_dot, after_mm),
                (before, before),
                "the scalar path must not run a SIMD entry point"
            );
        }
        eprintln!(
            "#141 f16 dot path: {path:?}, SIMD entry-point calls: dot {before} -> {after_dot}, \
             row decode -> {after_mm}"
        );
    }
}
