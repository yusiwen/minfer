// Phase 3: AVX2 Q4_0 × Q8_0 Dot Product Kernel
// Translated from: llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c lines 701-857
//   + helper functions: bytes_from_nibbles_32, mul_sum_i8_pairs_float, hsum_float_8, etc.
// Strict 1:1 translation — no extra code, no design changes

use crate::block::{self, BlockQ4_0, BlockQ8_0};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute Q4_0 × Q8_0 dot product for one row using AVX2
/// Corresponds to: ggml_vec_dot_q4_0_q8_0 (quants.c lines 701-857)
/// n: number of elements (must be multiple of 32)
/// Returns: dot product as f32
#[inline]
pub fn vec_dot_q4_0_q8_0(n: i32, vx: &[BlockQ4_0], vy: &[BlockQ8_0]) -> f32 {
    const QK: i32 = 32; // QK8_0 = QK4_0 = 32
    let nb = (n / QK) as usize;

    debug_assert!(n % QK == 0);
    debug_assert!(vx.len() >= nb);
    debug_assert!(vy.len() >= nb);

    // AVX2 path — corresponds to #if defined(__AVX2__) block (quants.c lines 718-741)
    #[cfg(target_arch = "x86_64")]
    {
        // Check if AVX2 + FMA is available at runtime
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q4_0_q8_0_avx2(nb, vx, vy) };
        }
    }

    // Scalar fallback — corresponds to loop at quants.c lines 840-854
    vec_dot_q4_0_q8_0_scalar(nb, vx, vy)
}

// === AVX2 implementation ===

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q4_0_q8_0_avx2(nb: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    use core::arch::x86_64::*;

    // Cast to raw byte pointers once, avoids bounds checks on struct field access.
    // BlockQ4_0: [d: fp16(2B) | qs: u8[16]] = 18 bytes
    // BlockQ8_0: [d: fp16(2B) | qs: i8[32]] = 34 bytes
    let xb = x.as_ptr() as *const u8;
    let yb = y.as_ptr() as *const u8;

    let mut acc = _mm256_setzero_ps();

    for ib in 0..nb {
        let xp = xb.add(ib * 18);
        let yp = yb.add(ib * 34);

        // d_x = GGML_CPU_FP16_TO_FP32(x[ib].d)
        let x_d = half::f16::from_bits(u16::from_le_bytes([*xp, *xp.add(1)])).to_f32();
        let y_d = half::f16::from_bits(u16::from_le_bytes([*yp, *yp.add(1)])).to_f32();
        let d = _mm256_set1_ps(x_d * y_d);

        // qx = bytes_from_nibbles_32(x[ib].qs)
        let mut qx = bytes_from_nibbles_32(xp.add(2) as *const i8);
        let off = _mm256_set1_epi8(8);
        qx = _mm256_sub_epi8(qx, off);

        // qy = _mm256_loadu_si256((const __m256i *)y[ib].qs)
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);

        let q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    hsum_float_8(acc)
}

// === Helper: bytes_from_nibbles_32 (quants.c lines 90-96) ===
// Unpack 32 4-bit fields into 32 bytes
// The output vector contains 32 bytes, each one in [ 0 .. 15 ] interval
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn bytes_from_nibbles_32(rsi: *const i8) -> __m256i {
    use core::arch::x86_64::*;

    // const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    let tmp = _mm_loadu_si128(rsi as *const __m128i);

    // const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    // MM256_SET_M128I(a, b) = _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
    let bytes = mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);

    // const __m256i lowMask = _mm256_set1_epi8( 0xF );
    let low_mask = _mm256_set1_epi8(0xF);

    // return _mm256_and_si256(lowMask, bytes);
    _mm256_and_si256(low_mask, bytes)
}

// === Helper: MM256_SET_M128I (quants.c line 26) ===
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    use core::arch::x86_64::*;
    // _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
    _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
}

// === Helper: sum_i16_pairs_float (quants.c lines 99-103) ===
// add int16_t pairwise and return as float vector
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    use core::arch::x86_64::*;

    // const __m256i ones = _mm256_set1_epi16(1);
    let ones = _mm256_set1_epi16(1);

    // const __m256i summed_pairs = _mm256_madd_epi16(ones, x);
    let summed_pairs = _mm256_madd_epi16(ones, x);

    // return _mm256_cvtepi32_ps(summed_pairs);
    _mm256_cvtepi32_ps(summed_pairs)
}

// === Helper: mul_sum_us8_pairs_float (quants.c lines 105-119) ===
// multiply unsigned int8_t, add results pairwise twice and return as float vector
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    use core::arch::x86_64::*;

    // On NUC12 (no AVX512VNNI/AVXVNNI), we take the #else branch:
    // const __m256i dot = _mm256_maddubs_epi16(ax, sy);
    let dot = _mm256_maddubs_epi16(ax, sy);

    // return sum_i16_pairs_float(dot);
    sum_i16_pairs_float(dot)
}

// === Helper: mul_sum_i8_pairs_float (quants.c lines 122-134) ===
// multiply int8_t, add results pairwise twice and return as float vector
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    use core::arch::x86_64::*;

    // On NUC12 (no AVXVNNIINT8), we take the #else branch:
    // const __m256i ax = _mm256_sign_epi8(x, x);
    let ax = _mm256_sign_epi8(x, x);

    // const __m256i sy = _mm256_sign_epi8(y, x);
    let sy = _mm256_sign_epi8(y, x);

    // return mul_sum_us8_pairs_float(ax, sy);
    mul_sum_us8_pairs_float(ax, sy)
}

// === Helper: hsum_float_8 (quants.c lines 43-49) ===
// horizontally add 8 floats
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    use core::arch::x86_64::*;

    // __m128 res = _mm256_extractf128_ps(x, 1);
    let mut res = _mm256_extractf128_ps(x, 1);

    // res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));

    // res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));

    // res = _mm_add_ss(res, _mm_movehdup_ps(res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));

    // return _mm_cvtss_f32(res);
    _mm_cvtss_f32(res)
}

// === Scalar fallback (quants.c lines 840-854) ===

fn vec_dot_q4_0_q8_0_scalar(nb: usize, x: &[BlockQ4_0], y: &[BlockQ8_0]) -> f32 {
    let qk = 32; // QK8_0 = QK4_0
    let mut sumf = 0.0f32;

    for ib in 0..nb {
        let mut sumi0: i32 = 0;
        let mut sumi1: i32 = 0;

        for j in 0..(qk / 2) as usize {
            let v0 = (x[ib].qs[j] & 0x0F) as i32 - 8;
            let v1 = (x[ib].qs[j] >> 4) as i32 - 8;

            sumi0 += v0 * y[ib].qs[j] as i32;
            sumi1 += v1 * y[ib].qs[j + (qk / 2) as usize] as i32;
        }

        let sumi = sumi0 + sumi1;
        sumf += sumi as f32 * block::fp16_to_f32(x[ib].d) * block::fp16_to_f32(y[ib].d);
    }

    sumf
}

// === quantize_row_q8_0 (ggml-quants.c lines 238-261, scalar ref + quants.c lines 302-397 AVX2) ===
/// Quantize f32 row to Q8_0 blocks. Each block holds 32 values.
/// x: input f32 array of length k (must be multiple of 32)
/// Returns: Vec<BlockQ8_0> with k/32 blocks
#[inline]
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<block::BlockQ8_0> {
    let k = x.len();
    debug_assert!(k % 32 == 0);
    let nb = k / 32;
    let mut y = vec![block::BlockQ8_0::default(); nb];

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { quantize_row_q8_0_avx2(x, &mut y, k) };
            return y;
        }
    }

    // Scalar fallback (quantize_row_q8_0_ref, ggml-quants.c lines 238-261)
    quantize_row_q8_0_scalar(x, &mut y, k);
    y
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_row_q8_0_avx2(x: &[f32], y: &mut [block::BlockQ8_0], k: usize) {
    use std::arch::x86_64::*;

    let nb = k / 32;
    for i in 0..nb {
        let off = i * 32;
        // Load 4 AVX vectors of 8 floats each (quants.c lines 312-315)
        let v0 = _mm256_loadu_ps(x.as_ptr().add(off));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(off + 8));
        let v2 = _mm256_loadu_ps(x.as_ptr().add(off + 16));
        let v3 = _mm256_loadu_ps(x.as_ptr().add(off + 24));

        // Compute max(abs(e)) (quants.c lines 319-328)
        let sign_bit = _mm256_set1_ps(-0.0f32);
        let mut max_abs = _mm256_andnot_ps(sign_bit, v0);
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v1));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v2));
        max_abs = _mm256_max_ps(max_abs, _mm256_andnot_ps(sign_bit, v3));

        let mut max4 = _mm_max_ps(_mm256_extractf128_ps(max_abs, 1), _mm256_castps256_ps128(max_abs));
        max4 = _mm_max_ps(max4, _mm_movehl_ps(max4, max4));
        max4 = _mm_max_ss(max4, _mm_movehdup_ps(max4));
        let max_scalar = _mm_cvtss_f32(max4);

        // Quantize (quants.c lines 331-366)
        let d = max_scalar / 127.0f32;
        y[i].d = block::f32_to_fp16(d);
        let id = if max_scalar != 0.0 { 127.0f32 / max_scalar } else { 0.0f32 };
        let mul = _mm256_set1_ps(id);

        let v0 = _mm256_mul_ps(v0, mul);
        let v1 = _mm256_mul_ps(v1, mul);
        let v2 = _mm256_mul_ps(v2, mul);
        let v3 = _mm256_mul_ps(v3, mul);

        let v0 = _mm256_round_ps(v0, _MM_ROUND_NEAREST as i32);
        let v1 = _mm256_round_ps(v1, _MM_ROUND_NEAREST as i32);
        let v2 = _mm256_round_ps(v2, _MM_ROUND_NEAREST as i32);
        let v3 = _mm256_round_ps(v3, _MM_ROUND_NEAREST as i32);

        let i0 = _mm256_cvtps_epi32(v0);
        let i1 = _mm256_cvtps_epi32(v1);
        let i2 = _mm256_cvtps_epi32(v2);
        let i3 = _mm256_cvtps_epi32(v3);

        // pack int32→int16→int8 (quants.c lines 356-367)
        let i0 = _mm256_packs_epi32(i0, i1);
        let i2 = _mm256_packs_epi32(i2, i3);
        let i0 = _mm256_packs_epi16(i0, i2);

        // Fix order (quants.c lines 362-365)
        let perm = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);
        let i0 = _mm256_permutevar8x32_epi32(i0, perm);

        _mm256_storeu_si256(y[i].qs.as_mut_ptr() as *mut __m256i, i0);
    }
}

/// Scalar fallback for quantize_row_q8_0 (quantize_row_q8_0_ref, ggml-quants.c lines 238-261)
fn quantize_row_q8_0_scalar(x: &[f32], y: &mut [block::BlockQ8_0], k: usize) {
    let nb = k / 32;
    for i in 0..nb {
        let mut amax = 0.0f32;
        for j in 0..32 {
            let v = x[i * 32 + j];
            amax = amax.max(v.abs());
        }

        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };

        y[i].d = block::f32_to_fp16(d);
        for j in 0..32 {
            y[i].qs[j] = (x[i * 32 + j] * id).round().clamp(-128.0, 127.0) as i8;
        }
    }
}

/// Quantize multiple rows of f32 to Q8_0 blocks, writing to a pre-allocated buffer.
/// x: [n_tokens * dim] f32, q8: [n_tokens * (dim/32)] output blocks.
pub fn quantize_row_q8_0_buf(x: &[f32], n_tokens: usize, dim: usize, q8: &mut [block::BlockQ8_0]) {
    let nb = dim / 32;
    for t in 0..n_tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let out = &mut q8[t * nb..(t + 1) * nb];
        let k = row.len();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                unsafe { quantize_row_q8_0_avx2(row, out, k) };
                continue;
            }
        }
        quantize_row_q8_0_scalar(row, out, k);
    }
}

/// Quantize multiple rows of f32 to Q8_0 blocks [n_tokens, nb].
/// Allocates and returns a new Vec.
pub fn quantize_row_q8_0_batch(x: &[f32], n_tokens: usize, dim: usize) -> Vec<block::BlockQ8_0> {
    let nb = dim / 32;
    let total = n_tokens * nb;
    let mut y = vec![block::BlockQ8_0::default(); total];
    for t in 0..n_tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let out = &mut y[t * nb..(t + 1) * nb];
        // Use quantize_row_q8_0 which dispatches to AVX2 when available
        // (quants.c lines 238-261: quantize_row_q8_0_ref with AVX2 acceleration)
        let q = quantize_row_q8_0(row);
        out.copy_from_slice(&q);
    }
    y
}

// === ggml_vec_dot_q8_0_q8_0 (quants.c lines 1170-1236) ===
/// Compute Q8_0 × Q8_0 dot product for one row using AVX2
/// n: number of elements (must be multiple of 32)
/// Returns: dot product as f32
#[inline]
pub fn vec_dot_q8_0_q8_0(n: i32, vx: &[BlockQ8_0], vy: &[BlockQ8_0]) -> f32 {
    debug_assert!(n % 32 == 0);
    let nb = (n / 32) as usize;
    debug_assert!(vx.len() >= nb);
    debug_assert!(vy.len() >= nb);

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { vec_dot_q8_0_q8_0_avx2(nb, vx, vy) };
        }
    }

    // Scalar fallback (quants.c lines 1225-1233)
    vec_dot_q8_0_q8_0_scalar(nb, vx, vy)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn vec_dot_q8_0_q8_0_avx2(nb: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    // Cast to raw byte pointers once, avoids bounds checks.
    // BlockQ8_0: [d: fp16(2B) | qs: i8[32]] = 34 bytes
    let xb = x.as_ptr() as *const u8;
    let yb = y.as_ptr() as *const u8;

    let mut acc = _mm256_setzero_ps();

    // Main loop (quants.c lines 1192-1202)
    for ib in 0..nb {
        let xp = xb.add(ib * 34);
        let yp = yb.add(ib * 34);

        // d_x = GGML_CPU_FP16_TO_FP32(x[ib].d)
        let x_d = half::f16::from_bits(u16::from_le_bytes([*xp, *xp.add(1)])).to_f32();
        let y_d = half::f16::from_bits(u16::from_le_bytes([*yp, *yp.add(1)])).to_f32();
        let d = _mm256_set1_ps(x_d * y_d);

        // qx = _mm256_loadu_si256((const __m256i *)x[ib].qs)
        let qx = _mm256_loadu_si256(xp.add(2) as *const __m256i);
        // qy = _mm256_loadu_si256((const __m256i *)y[ib].qs)
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);

        let q = mul_sum_i8_pairs_float(qx, qy);
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    hsum_float_8(acc)
}

/// Scalar fallback for vec_dot_q8_0_q8_0 (quants.c lines 1225-1233)
fn vec_dot_q8_0_q8_0_scalar(nb: usize, x: &[BlockQ8_0], y: &[BlockQ8_0]) -> f32 {
    let qk = 32;
    let mut sumf = 0.0f32;
    for ib in 0..nb {
        let mut sumi: i32 = 0;
        for j in 0..qk {
            sumi += x[ib].qs[j] as i32 * y[ib].qs[j] as i32;
        }
        sumf += sumi as f32 * block::fp16_to_f32(x[ib].d) * block::fp16_to_f32(y[ib].d);
    }
    sumf
}
