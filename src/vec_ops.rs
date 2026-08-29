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
