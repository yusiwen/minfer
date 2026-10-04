//! Element-wise f32 vector ops and the plain f32 matmul: `vec_dot_f32` (the
//! crate's scalar/SIMD dot) plus add / scale / mul / copy / muladd and
//! `mat_mul_f32`.
use super::*;

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
pub(super) unsafe fn vec_exp_f32_avx2(x: __m256) -> __m256 {
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
