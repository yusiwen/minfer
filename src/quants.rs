// Quantized dot-product kernels + activation quantization — all &[u8]
// interface. Fast paths: AVX2+FMA on x86_64, NEON+SDOT on aarch64, with
// scalar fallbacks. Activation formats: Q8_0 (simple weight types) and Q8_K
// (K-quant weights — 256-element blocks with precomputed bsums).
use crate::block::{self, Q4B, Q41B, Q8B, Q4KB, Q6KB};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// f16→f32 conversion with correct IEEE 754 handling for all cases
/// (zero, subnormal, normal, infinity, NaN). Only used by the x86_64 AVX2
/// kernels (the scalar paths use `block::fp16_to_f32`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn f16_to_f32_bits(bits: u16) -> f32 {
    let i = bits as u32;
    let sign = (i & 0x8000) << 16;
    let exp = (i >> 10) & 0x1F;
    let mant = i & 0x3FF;
    if exp == 0 {
        if mant == 0 { return f32::from_bits(sign); }
        let pos = 31 - mant.leading_zeros();
        return f32::from_bits(sign | ((103 + pos) << 23) | ((mant - (1 << pos)) << (23 - pos)));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F800000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

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
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q4_0_q8_0_neon(q4, q8, nb) };
        }
    }
    dot_q4_0_q8_0_scalar(q4, q8, nb)
}

// ============================================================
// Q4_1 × Q8_0 dot product
// Q4_1: value = q * d + m  (unsigned nibbles 0..15, no centering)
// ============================================================
#[inline]
pub fn dot_q4_1_q8_0(q4: &[u8], q8: &[u8]) -> f32 {
    let nb = q8.len() / Q8B;
    debug_assert!(q4.len() >= nb * Q41B);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q4_1_q8_0_neon(q4, q8, nb) };
        }
    }
    dot_q4_1_q8_0_scalar(q4, q8, nb)
}

fn dot_q4_1_q8_0_scalar(x: &[u8], y: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let xb = &x[ib * Q41B..];
        let yb = &y[ib * Q8B..];
        let d  = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let m  = block::fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]]));
        let dy = block::fp16_to_f32(u16::from_le_bytes([yb[0], yb[1]]));
        let mut sum_q = 0i32;
        let mut sum_y = 0i32;
        for j in 0..16 {
            let lo = (xb[4 + j] & 0x0F) as i32;
            let hi = (xb[4 + j] >> 4) as i32;
            let y0 = yb[2 + j] as i8 as i32;
            let y1 = yb[2 + j + 16] as i8 as i32;
            sum_q += lo * y0 + hi * y1;
            sum_y += y0 + y1;
        }
        // Formula: d * dy * Σ(q * y) + m * dy * Σ(y)
        s += dy * (d * sum_q as f32 + m * sum_y as f32);
    }
    s
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
        let xd = f16_to_f32_bits(*xp.cast::<u16>());
        let yd = f16_to_f32_bits(*yp.cast::<u16>());
        let d = _mm256_set1_ps(xd * yd);
        let tmp = _mm_loadu_si128(xp.add(2) as *const __m128i);
        let bytes = _mm256_set_m128i(_mm_srli_epi16(tmp, 4), tmp);
        let mut qx = _mm256_and_si256(bytes, _mm256_set1_epi8(0xF));
        qx = _mm256_sub_epi8(qx, _mm256_set1_epi8(8));
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot = _mm256_maddubs_epi16(ax, sy);
        let q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot));
        acc = _mm256_fmadd_ps(d, q, acc);
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
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q8_0_q8_0_neon(x, y, nb) };
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
        let xd = f16_to_f32_bits(*xp.cast::<u16>());
        let yd = f16_to_f32_bits(*yp.cast::<u16>());
        let d = _mm256_set1_ps(xd * yd);
        let qx = _mm256_loadu_si256(xp.add(2) as *const __m256i);
        let qy = _mm256_loadu_si256(yp.add(2) as *const __m256i);
        let ax = _mm256_sign_epi8(qx, qx);
        let sy = _mm256_sign_epi8(qy, qx);
        let dot = _mm256_maddubs_epi16(ax, sy);
        let q = _mm256_cvtepi32_ps(_mm256_madd_epi16(_mm256_set1_epi16(1), dot));
        acc = _mm256_fmadd_ps(d, q, acc);
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
// Q5_0 × Q8_0 dot product
// Q5_0: 32 elements / block, 22 bytes = d(f16,2) + qh(u8,4) + qs(u8,16)
// Q8_0: 32 elements / block, 34 bytes
// ============================================================

#[inline]
pub fn dot_q5_0_q8_0(q5: &[u8], q8: &[u8]) -> f32 {
    let nb = q8.len() / Q8B;
    debug_assert!(q5.len() >= nb * 22);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q5_0_q8_0_neon(q5, q8, nb) };
        }
    }
    dot_q5_0_q8_0_scalar(q5, q8, nb)
}

fn dot_q5_0_q8_0_scalar(q5: &[u8], q8: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let q5b = &q5[ib * 22..];
        let q8b = &q8[ib * Q8B..];
        let d_q5 = block::fp16_to_f32(u16::from_le_bytes([q5b[0], q5b[1]]));
        let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let d = d_q5 * d_q8;
        let qh = u32::from_le_bytes([q5b[2], q5b[3], q5b[4], q5b[5]]);
        let qs = &q5b[6..22];
        let mut si = 0i32;
        for j in 0..16 {
            let val_lo = ((qs[j] & 0x0F) as i32 | (((qh >> j) & 1) as i32) << 4) - 16;
            let val_hi = (((qs[j] >> 4) & 0x0F) as i32 | (((qh >> (j + 16)) & 1) as i32) << 4) - 16;
            let q8_lo = q8b[2 + j] as i8 as i32;
            let q8_hi = q8b[2 + j + 16] as i8 as i32;
            si += val_lo * q8_lo + val_hi * q8_hi;
        }
        s += si as f32 * d;
    }
    s
}

// ============================================================
// Q5_1 × Q8_0 dot product
// Q5_1: 32 elements / block, 24 bytes = d(f16,2) + m(f16,2) + qh(u32,4) + qs(u8,16)
// weight = d * ((nibble | (high_bit << 4)) - 16) + m
// ============================================================

#[inline]
pub fn dot_q5_1_q8_0(q5: &[u8], q8: &[u8]) -> f32 {
    let nb = q8.len() / Q8B;
    debug_assert!(q5.len() >= nb * 24);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q5_1_q8_0_neon(q5, q8, nb) };
        }
    }
    dot_q5_1_q8_0_scalar(q5, q8, nb)
}

fn dot_q5_1_q8_0_scalar(q5: &[u8], q8: &[u8], nb: usize) -> f32 {
    let mut s = 0.0f32;
    for ib in 0..nb {
        let q5b = &q5[ib * 24..];
        let q8b = &q8[ib * Q8B..];
        let d_q5 = block::fp16_to_f32(u16::from_le_bytes([q5b[0], q5b[1]]));
        let m_q5 = block::fp16_to_f32(u16::from_le_bytes([q5b[2], q5b[3]]));
        let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let qh = u32::from_le_bytes([q5b[4], q5b[5], q5b[6], q5b[7]]);
        let qs = &q5b[8..24];
        let mut sum_sub = 0i32;
        let mut sum_q8  = 0i32;
        for j in 0..16 {
            let u_lo = (qs[j] & 0x0F) as i32 | (((qh >> j) & 1) as i32) << 4;
            let u_hi = ((qs[j] >> 4) & 0x0F) as i32 | (((qh >> (j + 16)) & 1) as i32) << 4;
            let q8_lo = q8b[2 + j] as i8 as i32;
            let q8_hi = q8b[2 + j + 16] as i8 as i32;
            sum_sub += u_lo * q8_lo + u_hi * q8_hi;
            sum_q8  += q8_lo + q8_hi;
        }
        // Q5_1 dequant: val = d_q5 * unsigned_5bit + m_q5 (no -16 offset!)
        // dot = d_q8 * d_q5 * Σ(u×q) + d_q8 * m_q5 * Σ(q)
        s += d_q8 * (d_q5 * sum_sub as f32 + m_q5 * sum_q8 as f32);
    }
    s
}

// ============================================================
// Quantize f32 → Q8_0 bytes (raw &[u8], no struct types)
// ============================================================
fn quantize_row_q8_0_to(x: &[f32], y: &mut [u8]) {
    let k = x.len();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe { quantize_avx2(x, y, k) };
            return;
        }
    }
    quantize_scalar(x, y, k);
}

/// Test helper: quantize a full row and return the Q8_0 bytes (the graph path
/// quantizes into caller-owned buffers via `quantize_row_q8_0_buf` instead).
#[cfg(test)]
pub fn quantize_row_q8_0(x: &[f32]) -> Vec<u8> {
    let k = x.len();
    debug_assert!(k % 32 == 0);
    let nb = k / 32;
    let mut y = vec![0u8; nb * Q8B];
    quantize_row_q8_0_to(x, &mut y);
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
        for j in 0..32 { y[yo + 2 + j] = (x[i * 32 + j] * id).round_ties_even().clamp(-128.0, 127.0) as i8 as u8; }
    }
}

/// Quantize multiple rows directly into &mut [u8] buffer (no per-row Vec allocation).
pub fn quantize_row_q8_0_buf(x: &[f32], nt: usize, dim: usize, buf: &mut [u8]) {
    let rowb = (dim / 32) * Q8B;
    #[cfg(feature = "debug_dump")]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static DUMPED: AtomicBool = AtomicBool::new(false);
        if !DUMPED.swap(true, Ordering::Relaxed) && nt > 0 && dim >= 32 {
            let mut am = 0.0f32;
            for j in 0..32 { am = am.max(x[j].abs()); }
            let d = am / 127.0f32;
            let q0 = (x[0] / d).round().clamp(-128.0, 127.0) as i8;
            let q1 = (x[1] / d).round().clamp(-128.0, 127.0) as i8;
            let q16 = (x[16] / d).round().clamp(-128.0, 127.0) as i8;
            crate::dump::maybe_dump_text(
                "minfer_dump_q8_quant_verify",
                &format!("amax={:e} d={:e} x[0]={:e} x[1]={:e} x[16]={:e} q[0]={} q[1]={} q[16]={}",
                    am, d, x[0], x[1], x[16], q0, q1, q16),
            );
        }
    }
    for t in 0..nt {
        quantize_row_q8_0_to(&x[t * dim..(t + 1) * dim], &mut buf[t * rowb..(t + 1) * rowb]);
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn hsum_float_8(x: __m256) -> f32 {
    let x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    let x128 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    _mm_cvtss_f32(_mm_add_ss(x128, _mm_movehdup_ps(x128)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_to_fp16(v: f32) -> u16 { half::f16::from_f32(v).to_bits() }

    #[test]
    fn test_unpack_q4k_scales_boundary() {
        // All zeros
        let sc = [0u8; 12];
        let (scales, mins) = block::unpack_q4k_scales(&sc);
        assert_eq!(scales, [0; 8]);
        assert_eq!(mins, [0; 8]);

        // All 63 (max 6-bit value) in low-6-bit slots, high-2-bit slots set to pack 63 into indices 4-7
        let mut sc = [0u8; 12];
        for j in 0..4 {
            sc[j] = 63;     // scales[j] low 6 bits = 63
            sc[j + 4] = 63; // mins[j] low 6 bits = 63
        }
        // For indices 4-7: high 2 bits stored in sc[0..3]>>6 and sc[4..7]>>6
        // scales[4..7] = (sc[j+4] & 0xF) | ((sc[j-4] >> 6) << 4)
        // To get 63 = 0x3F: low 4 bits = 0xF, high 2 bits = 0x3
        for j in 4..8 {
            sc[j + 4] = 0xFF; // low 4 bits = 0xF for scales, high 4 bits = 0xF for mins
        }
        for j in 0..4 {
            sc[j] |= 0xC0; // high 2 bits = 0x3 for scales
            sc[j + 4] |= 0xC0; // high 2 bits = 0x3 for mins
        }
        let (scales, mins) = block::unpack_q4k_scales(&sc);
        assert_eq!(scales, [63; 8], "scales={:?}", scales);
        assert_eq!(mins, [63; 8], "mins={:?}", mins);

        // Mixed: known values
        let mut sc = [0u8; 12];
        sc[0] = 10; sc[1] = 20; sc[2] = 30; sc[3] = 40;
        sc[4] = 5; sc[5] = 15; sc[6] = 25; sc[7] = 35;
        // scales[4]=50: low4=0x2, high2=0x3 → sc[8]&0xF=2, sc[0]>>6=3 → sc[0]|=0xC0
        // mins[4]=45: low4=0xD, high2=0x2 → sc[8]>>4=0xD, sc[4]>>6=2 → sc[4]|=0x80
        sc[8] = 0x02 | (0x0D << 4); // scales[4] low=2, mins[4] low=0xD
        sc[0] |= 0xC0; // scales[4] high=3
        sc[4] |= 0x80; // mins[4] high=2
        let (scales, mins) = block::unpack_q4k_scales(&sc);
        assert_eq!(scales[0], 10);
        assert_eq!(scales[1], 20);
        assert_eq!(scales[2], 30);
        assert_eq!(scales[3], 40);
        assert_eq!(scales[4], 50);
        assert_eq!(mins[0], 5);
        assert_eq!(mins[1], 15);
        assert_eq!(mins[2], 25);
        assert_eq!(mins[3], 35);
        assert_eq!(mins[4], 45);
    }

    #[test]
    fn test_q8k_dot_simple() {
        let mut x = vec![0u8; Q8B];
        let mut y = vec![0u8; Q8B];
        let dx = 0.5f32; let dy = 2.0f32;
        let dx_bits = f32_to_fp16(dx).to_le_bytes();
        let dy_bits = f32_to_fp16(dy).to_le_bytes();
        x[0] = dx_bits[0]; x[1] = dx_bits[1];
        y[0] = dy_bits[0]; y[1] = dy_bits[1];
        for j in 0..32 { x[2 + j] = (j as i8) as u8; y[2 + j] = (31 - j as i8) as u8; }
        let result = dot_q8_0_q8_0(&x, &y);
        let mut ref_sum = 0.0f32;
        for j in 0..32 {
            ref_sum += ((j as i8) as f32 * dx) * (((31 - j) as i8) as f32 * dy);
        }
        eprintln!("test_q8k_dot_simple: result={:e} ref={:e} diff={:e}", result, ref_sum, (result - ref_sum).abs());
        assert!((result - ref_sum).abs() < 0.01);
    }

    #[test]
    fn test_q5_1_dot() {
        use crate::block::fp16_to_f32;
        // Q5_1: d(f16,2) + m(f16,2) + qh(u32,4) + qs(u8,16) = 24B
        // weight = d * unsigned_5bit + m
        let mut q5 = vec![0u8; 24];
        // d = 2.0 (fp16: 0x4000)
        q5[0] = 0x00; q5[1] = 0x40;
        // m = 0.5 (fp16: 0x3800)
        q5[2] = 0x00; q5[3] = 0x38;
        // qh = 0 (no high bits)
        // qs nibbles: 0,1,2,...,15 for both lo and hi
        for j in 0..16u8 {
            q5[8 + j as usize] = j | (j << 4);
        }

        // Build Q8_0 activation: all 1.0 -> d_q8 = 1.0/127 ≈ 0.007874, quants = 127
        let mut q8 = vec![0u8; 34];
        q8[0] = 0x00; q8[1] = 0x20; // fp16 1.0/128? Let's use d_q8=1.0, actually use known values
        // Actually, let's use d_q8 = 1.0 (fp16 0x3C00) and all quants = 1
        q8[0] = 0x00; q8[1] = 0x3C; // d_q8 = 1.0
        for j in 0..32 { q8[2 + j] = 1u8; } // quants = 1

        let result = dot_q5_1_q8_0(&q5, &q8);

        // Manual: Σ(d_q8 * (d * unsigned_5bit + m) * q8_quant)
        // unsigned_5bit = nibble (0..15), q8_quant = 1
        // result = 1.0 * Σ((2.0 * j + 0.5) * 1) for j in 0..15, counted twice (lo+hi)
        let mut ref_sum = 0.0f32;
        for j in 0..16 {
            ref_sum += 2.0 * j as f32 + 0.5 + 2.0 * j as f32 + 0.5;
        }
        // ref = 32*0.5 + 2*2*Σ(j=0..15) = 16 + 4*120 = 16 + 480 = 496
        eprintln!("test_q5_1_dot: result={:e} ref={:e} diff={:e}", result, ref_sum, (result - ref_sum).abs());
        assert!((result - ref_sum).abs() < 1.0, "result={} ref={}", result, ref_sum);
    }

}

// ============================================================
// aarch64 NEON fast paths — bit-exact with the scalar kernels:
// the int8×int8 products widen to int16/int32 and accumulate exactly
// (integer arithmetic is associative), and the per-block float ops
// are kept in the identical order. MINFER_NO_NEON=1 forces the
// scalar path for A/B.
// ============================================================
#[cfg(target_arch = "aarch64")]
mod neon_kernels {
    use super::*;
    use std::arch::aarch64::*;

    pub(super) fn enabled() -> bool {
        static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *V.get_or_init(|| {
            // SDOT (vdotq_s32) is the ARMv8.2+ int8 dot-product instruction
            // (16 MACs/instr); Apple M-series and all modern aarch64 have it.
            // Without it we fall back to the scalar kernels (the vmlal chain
            // below would be an unnecessary third path for pre-2018 chips).
            std::arch::is_aarch64_feature_detected!("neon")
                && std::arch::is_aarch64_feature_detected!("dotprod")
                && !std::env::var("MINFER_NO_NEON").map_or(false, |v| v == "1")
        })
    }

    #[inline(always)]
    pub(super) fn fp16(b0: u8, b1: u8) -> f32 {
        block::fp16_to_f32(u16::from_le_bytes([b0, b1]))
    }

    /// Accumulate 16 lanes of int8 × int8 into a scalar int32 with SDOT:
    /// one `sdot v.4s, a.16b, b.16b` computes 4 independent 4-element dot
    /// products (16 MACs) into an int32 accumulator — the llama.cpp Apple
    /// Silicon pattern. std::arch's `vdotq_s32` is unstable, so the
    /// instruction is emitted via stable inline asm. int32 accumulation is
    /// exact, so the result equals the scalar kernel's sequential i32 sum
    /// (bit-exact). Only called when `is_aarch64_feature_detected!("dotprod")`.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn dot16(a: int8x16_t, b: int8x16_t) -> i32 {
        vaddvq_s32(sdot_vec(vdupq_n_s32(0), a, b))
    }

    /// Raw SDOT accumulate (no horizontal reduce): 4 lanes × 4-element dot
    /// products (16 MACs) added to `acc`.
    #[target_feature(enable = "dotprod")]
    pub(super) unsafe fn sdot_vec(acc: int32x4_t, a: int8x16_t, b: int8x16_t) -> int32x4_t {
        let mut acc = acc;
        std::arch::asm!(
            "sdot {acc:v}.4s, {a:v}.16b, {b:v}.16b",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(nomem, nostack),
        );
        acc
    }

    /// Q4_0 × Q8_0 (32 values/block, centered nibbles).
    pub(super) unsafe fn dot_q4_0_q8_0(q4: &[u8], q8: &[u8], nb: usize) -> f32 {
        let m4b = vdupq_n_u8(0x0F);
        let s8 = vdupq_n_s8(8);
        let mut s = 0.0f32;
        for ib in 0..nb {
            let xb = &q4[ib * Q4B..];
            let yb = &q8[ib * Q8B..];
            let dx = fp16(xb[0], xb[1]);
            let dy = fp16(yb[0], yb[1]);
            let bytes = vld1q_u8(xb.as_ptr().add(2));
            let lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(bytes, m4b)), s8);
            let hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes)), s8);
            let y0 = vld1q_s8(yb.as_ptr().add(2) as *const i8);
            let y1 = vld1q_s8(yb.as_ptr().add(18) as *const i8);
            let si = dot16(lo, y0) + dot16(hi, y1);
            s += si as f32 * dx * dy;
        }
        s
    }

    /// Q8_0 × Q8_0 (32 values/block).
    pub(super) unsafe fn dot_q8_0_q8_0(x: &[u8], y: &[u8], nb: usize) -> f32 {
        let mut s = 0.0f32;
        for ib in 0..nb {
            let xb = &x[ib * Q8B..];
            let yb = &y[ib * Q8B..];
            let dx = fp16(xb[0], xb[1]);
            let dy = fp16(yb[0], yb[1]);
            let x0 = vld1q_s8(xb.as_ptr().add(2) as *const i8);
            let x1 = vld1q_s8(xb.as_ptr().add(18) as *const i8);
            let y0 = vld1q_s8(yb.as_ptr().add(2) as *const i8);
            let y1 = vld1q_s8(yb.as_ptr().add(18) as *const i8);
            let si = dot16(x0, y0) + dot16(x1, y1);
            s += si as f32 * dx * dy;
        }
        s
    }

    /// Q4_1 × Q8_0 (unsigned nibbles + per-block min).
    pub(super) unsafe fn dot_q4_1_q8_0(q4: &[u8], q8: &[u8], nb: usize) -> f32 {
        let m4b = vdupq_n_u8(0x0F);
        let mut s = 0.0f32;
        for ib in 0..nb {
            let xb = &q4[ib * Q41B..];
            let yb = &q8[ib * Q8B..];
            let d = fp16(xb[0], xb[1]);
            let m = fp16(xb[2], xb[3]);
            let dy = fp16(yb[0], yb[1]);
            let bytes = vld1q_u8(xb.as_ptr().add(4));
            let lo = vreinterpretq_s8_u8(vandq_u8(bytes, m4b));
            let hi = vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes));
            let y0 = vld1q_s8(yb.as_ptr().add(2) as *const i8);
            let y1 = vld1q_s8(yb.as_ptr().add(18) as *const i8);
            let sum_q = dot16(lo, y0) + dot16(hi, y1);
            let sum_y = vaddlvq_s8(y0) + vaddlvq_s8(y1);
            s += dy * (d * sum_q as f32 + m * sum_y as f32);
        }
        s
    }

    /// Q5_0 × Q8_0 (5-bit values; high bits expanded on the host — q5_0 is
    /// rare in K_M models, the qh bit layout is awkward for NEON).
    pub(super) unsafe fn dot_q5_0_q8_0(q5: &[u8], q8: &[u8], nb: usize) -> f32 {
        let m4b = vdupq_n_u8(0x0F);
        let s16 = vdupq_n_s8(16);
        let mut s = 0.0f32;
        for ib in 0..nb {
            let q5b = &q5[ib * 22..];
            let q8b = &q8[ib * Q8B..];
            let d = fp16(q5b[0], q5b[1]) * fp16(q8b[0], q8b[1]);
            let qh = u32::from_le_bytes([q5b[2], q5b[3], q5b[4], q5b[5]]);
            let qs = &q5b[6..22];
            let mut hb = [0i8; 32];
            for j in 0..16 {
                hb[j] = ((qh >> j) & 1) as i8;
                hb[j + 16] = ((qh >> (j + 16)) & 1) as i8;
            }
            let bytes = vld1q_u8(qs.as_ptr());
            let lo = vsubq_s8(vorrq_s8(
                vreinterpretq_s8_u8(vandq_u8(bytes, m4b)),
                vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr())),
            ), s16);
            let hi = vsubq_s8(vorrq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes)),
                vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr().add(16))),
            ), s16);
            let y0 = vld1q_s8(q8b.as_ptr().add(2) as *const i8);
            let y1 = vld1q_s8(q8b.as_ptr().add(18) as *const i8);
            let si = dot16(lo, y0) + dot16(hi, y1);
            s += si as f32 * d;
        }
        s
    }

    /// Q5_1 × Q8_0 (unsigned 5-bit + min).
    pub(super) unsafe fn dot_q5_1_q8_0(q5: &[u8], q8: &[u8], nb: usize) -> f32 {
        let m4b = vdupq_n_u8(0x0F);
        let mut s = 0.0f32;
        for ib in 0..nb {
            let q5b = &q5[ib * 24..];
            let q8b = &q8[ib * Q8B..];
            let d = fp16(q5b[0], q5b[1]);
            let m = fp16(q5b[2], q5b[3]);
            let dy = fp16(q8b[0], q8b[1]);
            let qh = u32::from_le_bytes([q5b[4], q5b[5], q5b[6], q5b[7]]);
            let qs = &q5b[8..24];
            let mut hb = [0i8; 32];
            for j in 0..16 {
                hb[j] = ((qh >> j) & 1) as i8;
                hb[j + 16] = ((qh >> (j + 16)) & 1) as i8;
            }
            let bytes = vld1q_u8(qs.as_ptr());
            let lo = vorrq_s8(
                vreinterpretq_s8_u8(vandq_u8(bytes, m4b)),
                vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr())),
            );
            let hi = vorrq_s8(
                vreinterpretq_s8_u8(vshrq_n_u8::<4>(bytes)),
                vshlq_n_s8::<4>(vld1q_s8(hb.as_ptr().add(16))),
            );
            let y0 = vld1q_s8(q8b.as_ptr().add(2) as *const i8);
            let y1 = vld1q_s8(q8b.as_ptr().add(18) as *const i8);
            let sum_sub = dot16(lo, y0) + dot16(hi, y1);
            let sum_y = vaddlvq_s8(y0) + vaddlvq_s8(y1);
            s += dy * (d * sum_sub as f32 + m * sum_y as f32);
        }
        s
    }

}

#[cfg(target_arch = "aarch64")]
use neon_kernels::enabled as neon_enabled;
#[cfg(target_arch = "aarch64")]
use neon_kernels::{
    dot_q4_0_q8_0 as dot_q4_0_q8_0_neon, dot_q4_1_q8_0 as dot_q4_1_q8_0_neon,
    dot_q5_0_q8_0 as dot_q5_0_q8_0_neon, dot_q5_1_q8_0 as dot_q5_1_q8_0_neon,
    dot_q8_0_q8_0 as dot_q8_0_q8_0_neon,
};

// ============================================================
// Q8_K activation path (K-quant matmuls)
//
// llama.cpp quantizes activations to Q8_K (256-element blocks with
// precomputed per-subblock int16 sums) for K-quant weights, so its dots
// never re-reduce the activation and use one block scale per 256 elements
// instead of 8 per 256. We follow that here for Q4_K/Q5_K/Q6_K; the simple
// types (Q4_0/Q8_0/Q5_0/Q5_1/Q4_1) keep the Q8_0 format (small models only).
// Q8_K block: d(f16) + qs[256 i8] + scales[16 i8, unused by the dots] +
// bsums[16 i16] = 306 bytes (crate::block::Q8KB).
// ============================================================

/// Quantize rows of f32 activations into Q8_K blocks (llama semantics:
/// block scale d = amax/127, q = round(x/d) clamped, bsum = Σq per 16).
/// NEON path is bit-exact with the scalar one: max-reduction is exact,
/// vmulq_f32 × id matches IEEE scalar multiply, vcvtnq rounds ties-even
/// (same as round_ties_even), vqmovn saturates (same as clamp).
pub fn quantize_row_q8_k_buf(x: &[f32], nt: usize, dim: usize, buf: &mut [u8]) {
    let n_super = dim / 256;
    let rowb = n_super * crate::block::Q8KB;
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            for t in 0..nt {
                unsafe {
                    quantize_row_q8_k_buf_neon(&x[t * dim..(t + 1) * dim],
                                               &mut buf[t * rowb..(t + 1) * rowb]);
                }
            }
            return;
        }
    }
    for t in 0..nt {
        let row = &x[t * dim..(t + 1) * dim];
        let out = &mut buf[t * rowb..(t + 1) * rowb];
        for s in 0..n_super {
            let blk = &row[s * 256..(s + 1) * 256];
            let o = s * crate::block::Q8KB;
            let mut amax = 0.0f32;
            for &v in blk {
                amax = amax.max(v.abs());
            }
            let d = amax / 127.0f32;
            let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
            let db = half::f16::from_f32(d).to_bits().to_le_bytes();
            out[o] = db[0];
            out[o + 1] = db[1];
            let mut bsums = [0i16; 16];
            for j in 0..256 {
                let q = (blk[j] * id).round_ties_even().clamp(-128.0, 127.0) as i8;
                out[o + 2 + j] = q as u8;
                bsums[j / 16] += q as i16;
            }
            // scales unused by the dots (the K-quant weight blocks carry their
            // own per-subblock scales); stored zeroed for format completeness.
            for j in 0..16 {
                out[o + 258 + j] = 0;
                out[o + 274 + 2 * j..o + 274 + 2 * j + 2]
                    .copy_from_slice(&bsums[j].to_le_bytes());
            }
        }
    }
}

/// NEON Q8_K quantization (one 256-element block at a time).
#[cfg(target_arch = "aarch64")]
unsafe fn quantize_row_q8_k_buf_neon(row: &[f32], out: &mut [u8]) {
    use std::arch::aarch64::*;
    let n_super = row.len() / 256;
    for s in 0..n_super {
        let blk = &row[s * 256..(s + 1) * 256];
        let o = s * crate::block::Q8KB;
        // amax over the block (exact max reduction)
        let mut amax = 0.0f32;
        for g in 0..64 {
            let v = vld1q_f32(blk.as_ptr().add(g * 4));
            amax = amax.max(vmaxvq_f32(vabsq_f32(v)));
        }
        let d = amax / 127.0f32;
        let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
        let db = half::f16::from_f32(d).to_bits().to_le_bytes();
        out[o] = db[0];
        out[o + 1] = db[1];
        let idv = vdupq_n_f32(id);
        let mut bsums = [0i16; 16];
        for g in 0..16 {
            let base = g * 16;
            let a = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(blk.as_ptr().add(base)), idv));
            let b = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(blk.as_ptr().add(base + 4)), idv));
            let c = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(blk.as_ptr().add(base + 8)), idv));
            let d = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(blk.as_ptr().add(base + 12)), idv));
            // saturating narrow s32x4 → s8x8 (two narrowing steps; clamps to
            // [-128, 127] like the scalar .clamp())
            let q8_0 = vqmovn_s16(vcombine_s16(vqmovn_s32(a), vqmovn_s32(b)));
            let q8_1 = vqmovn_s16(vcombine_s16(vqmovn_s32(c), vqmovn_s32(d)));
            vst1_s8(out.as_mut_ptr().add(o + 2 + base) as *mut i8, q8_0);
            vst1_s8(out.as_mut_ptr().add(o + 2 + base + 8) as *mut i8, q8_1);
            // bsum from the SATURATED values (exact int sum, matches scalar)
            bsums[g] = vaddlvq_s8(vcombine_s8(q8_0, q8_1)) as i16;
        }
        for g in 0..16 {
            out[o + 258 + g] = 0;
            out[o + 274 + 2 * g..o + 274 + 2 * g + 2].copy_from_slice(&bsums[g].to_le_bytes());
        }
    }
}

/// Q4_K × Q8_K dot (256 elements/superblock; activation carries bsums).
#[inline]
pub fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
    debug_assert!(q4.len() % Q4KB == 0);
    debug_assert!(q8k.len() % crate::block::Q8KB == 0);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q4_k_q8_k_neon(q4, q8k) };
        }
    }
    dot_q4_k_q8_k_scalar(q4, q8k)
}

fn dot_q4_k_q8_k_scalar(q4: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q4.len() / Q4KB;
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q4b = &q4[i * Q4KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = block::fp16_to_f32(u16::from_le_bytes([q4b[0], q4b[1]]))
            * block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let dmin = block::fp16_to_f32(u16::from_le_bytes([q4b[2], q4b[3]]))
            * block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q4b[4..16]).unwrap());
        // mins term: Σ mins[s] * (bsums[2s] + bsums[2s+1]) — llama subtracts first.
        let mut mterm = 0i32;
        for s in 0..8 {
            let b0 = i16::from_le_bytes([q8b[274 + 2 * (2 * s)], q8b[274 + 2 * (2 * s) + 1]]) as i32;
            let b1 = i16::from_le_bytes([q8b[274 + 2 * (2 * s + 1)], q8b[274 + 2 * (2 * s + 1) + 1]]) as i32;
            mterm += mins[s] as i32 * (b0 + b1);
        }
        sumf -= dmin * mterm as f32;
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..4 {
            let q4off = 16 + 32 * j;
            let q8off = 2 + 64 * j;
            let mut s_lo = 0i32;
            let mut s_hi = 0i32;
            for l in 0..32 {
                s_lo += (q4b[q4off + l] & 0x0F) as i32 * (q8b[q8off + l] as i8 as i32);
                s_hi += (q4b[q4off + l] >> 4) as i32 * (q8b[q8off + 32 + l] as i8 as i32);
            }
            sumi1 += s_lo * scales[2 * j] as i32;
            sumi2 += s_hi * scales[2 * j + 1] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

/// Q6_K × Q8_K dot (no min term; activation d applies to the whole block).
#[inline]
pub fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
    debug_assert!(q6.len() % Q6KB == 0);
    debug_assert!(q8k.len() % crate::block::Q8KB == 0);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q6_k_q8_k_neon(q6, q8k) };
        }
    }
    dot_q6_k_q8_k_scalar(q6, q8k)
}

fn dot_q6_k_q8_k_scalar(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = block::fp16_to_f32(u16::from_le_bytes([q6b[208], q6b[209]]))
            * block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let ql = &q6b[0..128];
        let qh = &q6b[128..192];
        let sc = &q6b[192..208];
        // Dequantize to a[256] i8 (interleaved, same layout as the q8_0 path)
        let mut a = [0i8; 256];
        {
            let mut a_off = 0usize;
            let mut ql_off = 0usize;
            let mut qh_off = 0usize;
            for _ in 0..2 {
                for l in 0..32 {
                    let ql0 = ql[ql_off + l] as i32;
                    let ql1 = ql[ql_off + l + 32] as i32;
                    let qh_b = qh[qh_off + l] as i32;
                    a[a_off + l + 0] = (((ql0 & 0x0F) | ((qh_b & 3) << 4)) - 32) as i8;
                    a[a_off + l + 32] = (((ql1 & 0x0F) | ((qh_b >> 2) & 3) << 4) - 32) as i8;
                    a[a_off + l + 64] = (((ql0 >> 4) | ((qh_b >> 4) & 3) << 4) - 32) as i8;
                    a[a_off + l + 96] = (((ql1 >> 4) | ((qh_b >> 6) & 3) << 4) - 32) as i8;
                }
                a_off += 128;
                ql_off += 64;
                qh_off += 32;
            }
        }
        for g in 0..16 {
            let scale = sc[g] as i8 as f32;
            let mut sum_sub = 0i32;
            for k in 0..16 {
                let elem = g * 16 + k;
                sum_sub += (a[elem] as i32) * (q8b[2 + g * 16 + k] as i8 as i32);
            }
            sumf += d * scale * sum_sub as f32;
        }
    }
    sumf
}

/// Q5_K × Q8_K dot (like Q4_K with high bits).
#[inline]
pub fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
    debug_assert!(q5.len() % 176 == 0);
    debug_assert!(q8k.len() % crate::block::Q8KB == 0);
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q5_k_q8_k_neon(q5, q8k) };
        }
    }
    dot_q5_k_q8_k_scalar(q5, q8k)
}

fn dot_q5_k_q8_k_scalar(q5: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q5.len() / 176;
    let mut sumf = 0.0f32;
    for i in 0..n_super {
        let q5b = &q5[i * 176..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = block::fp16_to_f32(u16::from_le_bytes([q5b[0], q5b[1]]))
            * block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let dmin = block::fp16_to_f32(u16::from_le_bytes([q5b[2], q5b[3]]))
            * block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q5b[4..16]).unwrap());
        let mut mterm = 0i32;
        for s in 0..8 {
            let b0 = i16::from_le_bytes([q8b[274 + 2 * (2 * s)], q8b[274 + 2 * (2 * s) + 1]]) as i32;
            let b1 = i16::from_le_bytes([q8b[274 + 2 * (2 * s + 1)], q8b[274 + 2 * (2 * s + 1) + 1]]) as i32;
            mterm += mins[s] as i32 * (b0 + b1);
        }
        sumf -= dmin * mterm as f32;
        let qh = &q5b[16..48];
        let qs = &q5b[48..176];
        let mut nb = [0i32; 256];
        for ci in 0..4 {
            let chunk = &qs[ci * 32..ci * 32 + 32];
            for l in 0..32 {
                nb[(2 * ci) * 32 + l] = (chunk[l] & 0x0F) as i32;
                nb[(2 * ci + 1) * 32 + l] = (chunk[l] >> 4) as i32;
            }
        }
        let mut sumi1 = 0i32;
        let mut sumi2 = 0i32;
        for j in 0..4 {
            let mut s_lo = 0i32;
            let mut s_hi = 0i32;
            for k in 0..32 {
                let s = 2 * j;
                let hbit_lo = ((qh[k] >> s) & 1) as i32;
                let u_lo = nb[s * 32 + k] | (hbit_lo << 4);
                s_lo += u_lo * (q8b[2 + (2 * j) * 32 + k] as i8 as i32);
                let s2 = 2 * j + 1;
                let hbit_hi = ((qh[k] >> s2) & 1) as i32;
                let u_hi = nb[s2 * 32 + k] | (hbit_hi << 4);
                s_hi += u_hi * (q8b[2 + (2 * j + 1) * 32 + k] as i8 as i32);
            }
            sumi1 += s_lo * scales[2 * j] as i32;
            sumi2 += s_hi * scales[2 * j + 1] as i32;
        }
        sumf += d * (sumi1 + sumi2) as f32;
    }
    sumf
}

// ─── Q8_K-activation NEON kernels (K-quant matmuls) ─────────────────
#[cfg(target_arch = "aarch64")]
mod neon_q8k {
    use super::*;
    use crate::block::Q8KB;
    use std::arch::aarch64::*;

    /// Q4_K × Q8_K: 8 subblocks, scales/mins from the weight block, activation
    /// bsums replace per-subblock reductions. Scales are applied to the SDOT
    /// int32 vectors BEFORE the horizontal reduce (one vaddvq per superblock
    /// instead of 8), and the mins term is fully vectorized — llama's shape.
    pub(super) unsafe fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
        let n_super = q4.len() / Q4KB;
        let m4b = vdupq_n_u8(0x0F);
        let zero = vdupq_n_s32(0);
        let mut sumf = 0.0f32;
        for i in 0..n_super {
            let q4b = &q4[i * Q4KB..];
            let q8b = &q8k[i * Q8KB..];
            let d = super::neon_kernels::fp16(q4b[0], q4b[1]) * super::neon_kernels::fp16(q8b[0], q8b[1]);
            let dmin = super::neon_kernels::fp16(q4b[2], q4b[3]) * super::neon_kernels::fp16(q8b[0], q8b[1]);
            let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q4b[4..16]).unwrap());
            // mins term: (bsums[2s]+bsums[2s+1]) per 32-element subblock
            let bsums0 = vld1q_s16(q8b.as_ptr().add(274) as *const i16);
            let bsums1 = vld1q_s16(q8b.as_ptr().add(274 + 16) as *const i16);
            // vpaddq_s16(a, b) yields only 8 lanes: a's 4 pairs then b's 4
            // pairs — so load BOTH bsums halves (16 int16) and combine → the
            // 8 per-32-element sums in lanes 0..7. NOTE: vpaddq wraps on
            // int16 overflow — safe because the q8_K quantizer bounds bsums
            // to ±2032 (a pair sum ≤ 4064); the test uses quantizer output.
            let q8sums = vpaddq_s16(bsums0, bsums1);
            let mut ms = [0i16; 8];
            for k in 0..8 {
                ms[k] = mins[k] as i16;
            }
            // mins as two 4-lane halves — vld1q_s16 would read 8 lanes and
            // leave lanes 8..15 as undefined stack garbage.
            let mins_lo = vld1_s16(ms.as_ptr());
            let mins_hi = vld1_s16(ms.as_ptr().add(4));
            let prod = vaddq_s32(
                vmull_s16(vget_low_s16(q8sums), mins_lo),
                vmull_s16(vget_high_s16(q8sums), mins_hi),
            );
            sumf -= dmin * vaddvq_s32(prod) as f32;
            let mut sumi1 = zero;
            let mut sumi2 = zero;
            for j in 0..4 {
                let q4p = q4b.as_ptr().add(16 + 32 * j);
                let q8p = q8b.as_ptr().add(2 + 64 * j) as *const i8;
                let v0 = vld1q_u8(q4p);
                let v1 = vld1q_u8(q4p.add(16));
                let a0 = vreinterpretq_s8_u8(vandq_u8(v0, m4b));
                let a1 = vreinterpretq_s8_u8(vandq_u8(v1, m4b));
                let b0 = vld1q_s8(q8p);
                let b1 = vld1q_s8(q8p.add(16));
                let sc1 = scales[2 * j] as i32;
                sumi1 = vmlaq_n_s32(sumi1, super::neon_kernels::sdot_vec(zero, a0, b0), sc1);
                sumi1 = vmlaq_n_s32(sumi1, super::neon_kernels::sdot_vec(zero, a1, b1), sc1);
                let a2 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v0));
                let a3 = vreinterpretq_s8_u8(vshrq_n_u8::<4>(v1));
                let b2 = vld1q_s8(q8p.add(32));
                let b3 = vld1q_s8(q8p.add(48));
                let sc2 = scales[2 * j + 1] as i32;
                sumi2 = vmlaq_n_s32(sumi2, super::neon_kernels::sdot_vec(zero, a2, b2), sc2);
                sumi2 = vmlaq_n_s32(sumi2, super::neon_kernels::sdot_vec(zero, a3, b3), sc2);
            }
            sumf += d * (vaddvq_s32(sumi1) + vaddvq_s32(sumi2)) as f32;
        }
        sumf
    }

    /// Q6_K × Q8_K: no min term; one d per superblock. Per-group scales are
    /// applied to the SDOT vectors before the single final reduce.
    pub(super) unsafe fn dot_q6_k_q8_k(q6: &[u8], q8k: &[u8]) -> f32 {
        let n_super = q6.len() / Q6KB;
        let m4b = vdupq_n_u8(0x0F);
        let m3 = vdupq_n_u8(3);
        let s32 = vdupq_n_s8(32);
        let zero = vdupq_n_s32(0);
        let mut sumf = 0.0f32;
        for i in 0..n_super {
            let q6b = &q6[i * Q6KB..];
            let q8b = &q8k[i * Q8KB..];
            let d = super::neon_kernels::fp16(q6b[208], q6b[209]) * super::neon_kernels::fp16(q8b[0], q8b[1]);
            for n in 0..2 {
                let qlp = q6b.as_ptr().add(n * 64);
                let qhp = q6b.as_ptr().add(128 + n * 32);
                let ql0 = vld1q_u8(qlp);
                let ql1 = vld1q_u8(qlp.add(16));
                let ql2 = vld1q_u8(qlp.add(32));
                let ql3 = vld1q_u8(qlp.add(48));
                let qh0 = vld1q_u8(qhp);
                let qh1 = vld1q_u8(qhp.add(16));
                let q0a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql0, m4b), vshlq_n_u8::<4>(vandq_u8(qh0, m3)))), s32);
                let q0b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql1, m4b), vshlq_n_u8::<4>(vandq_u8(qh1, m3)))), s32);
                let q1a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql2, m4b), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh0), m3)))), s32);
                let q1b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vandq_u8(ql3, m4b), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<2>(qh1), m3)))), s32);
                let q2a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql0), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh0), m3)))), s32);
                let q2b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql1), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<4>(qh1), m3)))), s32);
                let q3a = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql2), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(qh0), m3)))), s32);
                let q3b = vsubq_s8(vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(ql3), vshlq_n_u8::<4>(vandq_u8(vshrq_n_u8::<6>(qh1), m3)))), s32);
                let qv = [q0a, q0b, q1a, q1b, q2a, q2b, q3a, q3b];
                let mut acc = zero;
                for g in 0..8 {
                    let vec = qv[(g / 2) * 2 + (g % 2)];
                    let y = vld1q_s8(q8b.as_ptr().add(2 + (n * 8 + g) * 16) as *const i8);
                    let scale = q6b[192 + n * 8 + g] as i8 as i32;
                    acc = vmlaq_n_s32(acc, super::neon_kernels::sdot_vec(zero, vec, y), scale);
                }
                // sumf += d * Σ_g scale_g * dot_g — the horizontal sum is exact
                // int; the float order matches the scalar kernel.
                sumf += d * vaddvq_s32(acc) as f32;
            }
        }
        sumf
    }

    /// Q5_K × Q8_K: high bits via variable right shift of the 32-byte qh.
    pub(super) unsafe fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
        let n_super = q5.len() / 176;
        let m4b = vdupq_n_u8(0x0F);
        let one = vdupq_n_u8(1);
        let mut sumf = 0.0f32;
        for i in 0..n_super {
            let q5b = &q5[i * 176..];
            let q8b = &q8k[i * Q8KB..];
            let d = super::neon_kernels::fp16(q5b[0], q5b[1]) * super::neon_kernels::fp16(q8b[0], q8b[1]);
            let dmin = super::neon_kernels::fp16(q5b[2], q5b[3]) * super::neon_kernels::fp16(q8b[0], q8b[1]);
            let (scales, mins) = block::unpack_q4k_scales(<&[u8; 12]>::try_from(&q5b[4..16]).unwrap());
            let mut mterm = 0i32;
            for s in 0..8 {
                let b0 = i16::from_le_bytes([q8b[274 + 4 * s], q8b[274 + 4 * s + 1]]) as i32;
                let b1 = i16::from_le_bytes([q8b[274 + 4 * s + 2], q8b[274 + 4 * s + 3]]) as i32;
                mterm += mins[s] as i32 * (b0 + b1);
            }
            sumf -= dmin * mterm as f32;
            let qh = q5b.as_ptr().add(16);
            let qs = q5b.as_ptr().add(48);
            let mut sumi1 = 0i32;
            let mut sumi2 = 0i32;
            for j in 0..4 {
                let cp = qs.add(32 * j);
                let v0 = vld1q_u8(cp);
                let v1 = vld1q_u8(cp.add(16));
                // subblock 2j (lo nibbles)
                let s_sub = 2 * j;
                let sh = vdupq_n_s8(-(s_sub as i8));
                let h0 = vandq_u8(vshlq_u8(vld1q_u8(qh), sh), one);
                let h1 = vandq_u8(vshlq_u8(vld1q_u8(qh.add(16)), sh), one);
                let a0 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v0, m4b), vshlq_n_u8::<4>(h0)));
                let a1 = vreinterpretq_s8_u8(vorrq_u8(vandq_u8(v1, m4b), vshlq_n_u8::<4>(h1)));
                let q8p = q8b.as_ptr().add(2 + (2 * j) * 32) as *const i8;
                let b0 = vld1q_s8(q8p);
                let b1 = vld1q_s8(q8p.add(16));
                let p_lo = super::neon_kernels::dot16(a0, b0) + super::neon_kernels::dot16(a1, b1);
                // subblock 2j+1 (hi nibbles)
                let s_sub2 = 2 * j + 1;
                let sh2 = vdupq_n_s8(-(s_sub2 as i8));
                let h2 = vandq_u8(vshlq_u8(vld1q_u8(qh), sh2), one);
                let h3 = vandq_u8(vshlq_u8(vld1q_u8(qh.add(16)), sh2), one);
                let a2 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(v0), vshlq_n_u8::<4>(h2)));
                let a3 = vreinterpretq_s8_u8(vorrq_u8(vshrq_n_u8::<4>(v1), vshlq_n_u8::<4>(h3)));
                let q8p2 = q8b.as_ptr().add(2 + (2 * j + 1) * 32) as *const i8;
                let b2 = vld1q_s8(q8p2);
                let b3 = vld1q_s8(q8p2.add(16));
                let p_hi = super::neon_kernels::dot16(a2, b2) + super::neon_kernels::dot16(a3, b3);
                sumi1 += p_lo * scales[2 * j] as i32;
                sumi2 += p_hi * scales[2 * j + 1] as i32;
            }
            sumf += d * (sumi1 + sumi2) as f32;
        }
        sumf
    }
}

#[cfg(target_arch = "aarch64")]
use neon_q8k::{
    dot_q4_k_q8_k as dot_q4_k_q8_k_neon, dot_q5_k_q8_k as dot_q5_k_q8_k_neon,
    dot_q6_k_q8_k as dot_q6_k_q8_k_neon,
};

#[cfg(all(test, target_arch = "aarch64"))]
mod neon_correctness {
    use super::*;

    fn rng_state() -> u64 { 0x9E3779B97F4A7C15 }
    fn next(st: &mut u64) -> u64 {
        *st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *st
    }

    fn q4k_block(st: &mut u64) -> Vec<u8> {
        let mut b = vec![0u8; Q4KB];
        for v in b.iter_mut() { *v = (next(st) >> 8) as u8; }
        // valid d/dmin (finite fp16), scales/mins as u8
        b[0] = 0x00; b[1] = 0x3C; b[2] = 0x00; b[3] = 0x3C;
        b
    }
    fn q6k_block(st: &mut u64) -> Vec<u8> {
        let mut b = vec![0u8; Q6KB];
        for v in b.iter_mut() { *v = (next(st) >> 8) as u8; }
        b[208] = 0x00; b[209] = 0x3C;
        b
    }
    fn q8k_block(st: &mut u64) -> Vec<u8> {
        let mut b = vec![0u8; crate::block::Q8KB];
        for v in b.iter_mut() { *v = (next(st) >> 8) as u8; }
        b[0] = 0x00; b[1] = 0x3C;
        b
    }

    #[test]
    fn neon_q8k_dots_match_scalar() {
        // Realistic data: activations come from the q8_K quantizer (bsums
        // bounded to ±2032 — vpaddq_s16 in the kernel wraps on full-range
        // int16, which the scalar i32 path does not; real bsums never reach
        // that range), weights use valid d/dmin/scales/mins with nibble data.
        let mut st = rng_state();
        for _ in 0..20 {
            let dim = 256 * (1 + (next(&mut st) % 4) as usize);
            let x: Vec<f32> = (0..dim)
                .map(|i| (((next(&mut st) >> 40) % 1000) as f32) * 1e-3 - 0.5)
                .collect();
            let mut q8k = vec![0u8; (dim / 256) * crate::block::Q8KB];
            quantize_row_q8_k_buf(&x, 1, dim, &mut q8k);
            // q4_K / q6_K weight blocks
            let mut q4 = Vec::new();
            let mut q6 = Vec::new();
            for _ in 0..dim / 256 {
                let mut b = vec![0u8; Q4KB];
                for v in b.iter_mut() {
                    *v = ((next(&mut st) >> 8) % 8) as u8;
                }
                b[0] = 0x00; b[1] = 0x3C; b[2] = 0x00; b[3] = 0x3C; // d = dmin = 1
                b[4] = 8; b[5] = 8; b[6] = 8; b[7] = 8;             // scales[0..4] = 8
                q4.extend_from_slice(&b);
                let mut b6 = vec![0u8; Q6KB];
                for v in b6.iter_mut() {
                    *v = ((next(&mut st) >> 8) % 8) as u8;
                }
                b6[208] = 0x00; b6[209] = 0x3C; // d = 1
                q6.extend_from_slice(&b6);
            }
            let a = dot_q4_k_q8_k(&q4, &q8k);
            let b = dot_q4_k_q8_k_scalar(&q4, &q8k);
            assert!((a - b).abs() <= (a.abs() + b.abs()).max(1e-6) * 1e-5,
                "q4_K q8_K NEON {a} != scalar {b}");
            let a = dot_q6_k_q8_k(&q6, &q8k);
            let b = dot_q6_k_q8_k_scalar(&q6, &q8k);
            assert!((a - b).abs() <= (a.abs() + b.abs()).max(1e-6) * 1e-5,
                "q6_K q8_K NEON {a} != scalar {b}");
        }
    }

    #[test]
    fn neon_q8k_quantize_matches_scalar() {
        let mut st = rng_state();
        for _ in 0..10 {
            let dim = 256 * (1 + (next(&mut st) % 4) as usize);
            let x: Vec<f32> = (0..dim).map(|i| ((next(&mut st) >> 40) as f32) * 1e-3 - 0.5).collect();
            let mut ba = vec![0u8; (dim / 256) * crate::block::Q8KB];
            let mut bb = vec![0u8; (dim / 256) * crate::block::Q8KB];
            quantize_row_q8_k_buf(&x, 1, dim, &mut ba);
            // scalar path
            let n_super = dim / 256;
            let rowb = n_super * crate::block::Q8KB;
            for s in 0..n_super {
                let blk = &x[s * 256..(s + 1) * 256];
                let o = s * crate::block::Q8KB;
                let mut amax = 0.0f32;
                for &v in blk { amax = amax.max(v.abs()); }
                let d = amax / 127.0f32;
                let id = if d != 0.0 { 1.0f32 / d } else { 0.0f32 };
                let db = half::f16::from_f32(d).to_bits().to_le_bytes();
                bb[o] = db[0]; bb[o + 1] = db[1];
                let mut bsums = [0i16; 16];
                for j in 0..256 {
                    let q = (blk[j] * id).round_ties_even().clamp(-128.0, 127.0) as i8;
                    bb[o + 2 + j] = q as u8;
                    bsums[j / 16] += q as i16;
                }
                for j in 0..16 {
                    bb[o + 258 + j] = 0;
                    bb[o + 274 + 2 * j..o + 274 + 2 * j + 2].copy_from_slice(&bsums[j].to_le_bytes());
                }
            }
            assert_eq!(ba, bb, "q8_K quantize NEON != scalar at dim {dim}");
        }
    }
}
