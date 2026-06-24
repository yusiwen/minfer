// AVX2 kernels — all &[u8] interface. Directly follows minfer2/src/quant.rs pattern.
use crate::block::{self, BlockQ4_0, BlockQ8_0};

pub const Q4B: usize = 18;
pub const Q8B: usize = 34;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

// ============================================================
// Q4_0 × Q8_0 dot product (raw &[u8] interface, no n_blocks param — slice length-based)
// ============================================================
#[inline]
pub fn dot_q4_0_q8_0(q4: &[u8], q8: &[u8]) -> f32 {
    let nb = q8.len() / Q8B;
    debug_assert!(q4.len() >= nb * Q4B);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q4_0_q8_0_avx2(q4, q8, nb) };
        }
    }
    dot_q4_0_q8_0_scalar(q4, q8, nb)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q4_0_q8_0_avx2(x: &[u8], y: &[u8], nb: usize) -> f32 {
    use core::arch::x86_64::*;
    let xb = x.as_ptr();
    let yb = y.as_ptr();
    let mut acc = _mm256_setzero_ps();
    for ib in 0..nb {
        let xp = xb.add(ib * Q4B);
        let yp = yb.add(ib * Q8B);
        let xd = half::f16::from_bits(u16::from_le_bytes([*xp, *xp.add(1)])).to_f32();
        let yd = half::f16::from_bits(u16::from_le_bytes([*yp, *yp.add(1)])).to_f32();
        let d = _mm256_set1_ps(xd * yd);
        let mut qx = bytes_from_nibbles_32(xp.add(2) as *const i8);
        qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);
        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc);
    }
    hsum_float_8(acc)
}

fn dot_q4_0_q8_0_scalar(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q4B..];
        let yb = &y[ib * Q8B..];
        let dx = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let dy = block::fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let mut si = 0i32;
        for j in 0..16 {
            let v0 = (xb[2 + j] & 0x0F) as i8 - 8;
            let v1 = (xb[2 + j] >> 4) as i8 - 8;
            si += (v0 as i32) * (yb[2 + j] as i8 as i32);
            si += (v1 as i32) * (yb[2 + j + 16] as i8 as i32);
        }
        s += si as f32 * dx * dy;
    }
    s
}

// ============================================================
// Q8_0 × Q8_0 dot product (raw &[u8] interface)
// ============================================================
#[inline]
pub fn dot_q8_0_q8_0(x: &[u8], y: &[u8]) -> f32 {
    let nb = y.len() / Q8B;
    debug_assert!(x.len() >= nb * Q8B);
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return unsafe { dot_q8_0_q8_0_avx2(x, y, nb) };
        }
    }
    dot_q8_0_q8_0_scalar(x, y, nb)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_q8_0_q8_0_avx2(x: &[u8], y: &[u8], nb: usize) -> f32 {
    use core::arch::x86_64::*;
    let xb = x.as_ptr();
    let yb = y.as_ptr();
    let mut acc = _mm256_setzero_ps();
    for ib in 0..nb {
        let xp = xb.add(ib * Q8B);
        let yp = yb.add(ib * Q8B);
        let xd = half::f16::from_bits(u16::from_le_bytes([*xp, *xp.add(1)])).to_f32();
        let yd = half::f16::from_bits(u16::from_le_bytes([*yp, *yp.add(1)])).to_f32();
        let d = _mm256_set1_ps(xd * yd);
        let qx = _mm256_loadu_si256(xp.add(2) as *const __m256i);
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);
        acc = _mm256_fmadd_ps(d, mul_sum_i8_pairs_float(qx, qy), acc);
    }
    hsum_float_8(acc)
}

fn dot_q8_0_q8_0_scalar(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q8B..];
        let yb = &y[ib * Q8B..];
        let dx = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let dy = block::fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let mut si = 0i32;
        for j in 0..32 { si += (xb[2 + j] as i8 as i32) * (yb[2 + j] as i8 as i32); }
        s += si as f32 * dx * dy;
    }
    s
}

// ============================================================
// Legacy struct-based wrappers (for backward compat in vec_ops.rs)
// ============================================================
#[inline]
pub fn vec_dot_q4_0_q8_0(n: i32, vx: &[BlockQ4_0], vy: &[BlockQ8_0]) -> f32 {
    let nb = (n / 32) as usize;
    let vx_b: &[u8] = unsafe { std::mem::transmute(vx) };
    let vy_b: &[u8] = unsafe { std::mem::transmute(vy) };
    dot_q4_0_q8_0(&vx_b[..nb * Q4B], &vy_b[..nb * Q8B])
}

#[inline]
pub fn vec_dot_q8_0_q8_0(n: i32, vx: &[BlockQ8_0], vy: &[BlockQ8_0]) -> f32 {
    let nb = (n / 32) as usize;
    let vx_b: &[u8] = unsafe { std::mem::transmute(vx) };
    let vy_b: &[u8] = unsafe { std::mem::transmute(vy) };
    dot_q8_0_q8_0(&vx_b[..nb * Q8B], &vy_b[..nb * Q8B])
}

// ============================================================
// Quantize f32 → Q8_0 bytes (raw &[u8], no struct types)
// ============================================================
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<u8> {
    let k = x.len();
    debug_assert!(k % 32 == 0);
    let nb = k / 32;
    let mut y = vec![0u8; nb * Q8B];
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { quantize_avx2(x, &mut y, k) };
            return y;
        }
    }
    quantize_scalar(x, &mut y, k);
    y
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn quantize_avx2(x: &[f32], y: &mut [u8], k: usize) {
    use std::arch::x86_64::*;
    let nb = k / 32;
    for i in 0..nb {
        let off = i * 32;
        let v0 = _mm256_loadu_ps(x.as_ptr().add(off));
        let v1 = _mm256_loadu_ps(x.as_ptr().add(off + 8));
        let v2 = _mm256_loadu_ps(x.as_ptr().add(off + 16));
        let v3 = _mm256_loadu_ps(x.as_ptr().add(off + 24));
        let sb = _mm256_set1_ps(-0.0f32);
        let ma = _mm256_max_ps(
            _mm256_max_ps(_mm256_andnot_ps(sb, v0), _mm256_andnot_ps(sb, v1)),
            _mm256_max_ps(_mm256_andnot_ps(sb, v2), _mm256_andnot_ps(sb, v3)),
        );
        let m4 = _mm_max_ps(_mm256_extractf128_ps(ma, 1), _mm256_castps256_ps128(ma));
        let m4 = _mm_max_ps(m4, _mm_movehl_ps(m4, m4));
        let ms = _mm_cvtss_f32(_mm_max_ss(m4, _mm_movehdup_ps(m4)));
        let d = ms / 127.0f32;
        let db = half::f16::from_f32(d).to_bits().to_le_bytes();
        let yo = i * Q8B;
        y[yo] = db[0]; y[yo + 1] = db[1];
        let id = if ms != 0.0 { 127.0f32 / ms } else { 0.0f32 };
        let mul = _mm256_set1_ps(id);
        let i0 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST as i32));
        let i1 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v1, mul), _MM_ROUND_NEAREST as i32));
        let i2 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v2, mul), _MM_ROUND_NEAREST as i32));
        let i3 = _mm256_cvtps_epi32(_mm256_round_ps(_mm256_mul_ps(v3, mul), _MM_ROUND_NEAREST as i32));
        let i0 = _mm256_packs_epi32(i0, i1);
        let i2 = _mm256_packs_epi32(i2, i3);
        let i0 = _mm256_packs_epi16(i0, i2);
        let i0 = _mm256_permutevar8x32_epi32(i0, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
        _mm256_storeu_si256(y.as_mut_ptr().add(yo + 2) as *mut __m256i, i0);
    }
}

fn quantize_scalar(x: &[f32], y: &mut [u8], k: usize) {
    let nb = k / 32;
    for i in 0..nb {
        let mut am = 0.0f32;
        for j in 0..32 { am = am.max(x[i * 32 + j].abs()); }
        let d = am / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let db = half::f16::from_f32(d).to_bits().to_le_bytes();
        let yo = i * Q8B;
        y[yo] = db[0]; y[yo + 1] = db[1];
        for j in 0..32 { y[yo + 2 + j] = (x[i * 32 + j] * id).round().clamp(-128.0, 127.0) as i8 as u8; }
    }
}

/// Quantize multiple rows, write to &mut [u8] buffer.
pub fn quantize_row_q8_0_buf(x: &[f32], nt: usize, dim: usize, buf: &mut [u8]) {
    let nb = dim / 32;
    let rowb = nb * Q8B;
    for t in 0..nt {
        let out = &mut buf[t * rowb..(t + 1) * rowb];
        let q = quantize_row_q8_0(&x[t * dim..(t + 1) * dim]);
        out.copy_from_slice(&q);
    }
}

// ============================================================
// Helper: bytes_from_nibbles_32
// ============================================================
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn bytes_from_nibbles_32(rsi: *const i8) -> __m256i {
    use core::arch::x86_64::*;
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
    _mm256_and_si256(bytes, _mm256_set1_epi8(0xF))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mm256_set_m128i(hi: __m128i, lo: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(lo), hi, 1)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), x))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    _mm_cvtss_f32(_mm_add_ss(x128, _mm_movehdup_ps(x128)))
}
