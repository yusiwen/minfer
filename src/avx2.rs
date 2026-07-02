// AVX2 kernels — all &[u8] interface. Directly follows minfer2/src/quant.rs pattern.
use crate::block::{self, BlockQ4_0, BlockQ8_0, Q4B, Q41B, Q8B, Q4KB, Q6KB, Q8KB};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// f16→f32 conversion with correct IEEE 754 handling for all cases
/// (zero, subnormal, normal, infinity, NaN).
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
// Q4_K × Q8_0 dot product
// Q4_K: 256 elements / superblock, 8 subblocks × 32 elements, 144 bytes
// Q8_0: 32 elements / block, 34 bytes
// 1 Q4_K superblock needs 8 Q8_0 blocks for the same 256 elements
// ============================================================

#[inline]
pub fn dot_q4_k_q8_0(q4: &[u8], q8: &[u8]) -> f32 {
    debug_assert!(q4.len() % Q4KB == 0);
    dot_q4_k_q8_0_scalar(q4, q8)
}

fn dot_q4_k_q8_0_scalar(q4: &[u8], q8: &[u8]) -> f32 {
    let n_super = q4.len() / Q4KB;
    let mut sum = 0.0f32;

    for i in 0..n_super {
        let q4b  = &q4[i * Q4KB..];
        let q8b  = &q8[i * 8 * Q8B..];

        let d    = block::fp16_to_f32(u16::from_le_bytes([q4b[0], q4b[1]]));
        let dmin = block::fp16_to_f32(u16::from_le_bytes([q4b[2], q4b[3]]));
        let sc_bytes = <&[u8; 12]>::try_from(&q4b[4..16]).unwrap();
        let (scales, mins) = block::unpack_q4k_scales(sc_bytes);

        let qs   = &q4b[16..144];

        // 8 subblocks, each 32 elements
        for s in 0..8 {
            let sc_val = scales[s];
            let mm_val = mins[s];

            let dl = d * sc_val as f32;
            let ml = dmin * mm_val as f32;

            // Q8_0 block for this subblock
            let q8blk = &q8b[s * Q8B..];
            let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8blk[0], q8blk[1]]));
            let q8qs = &q8blk[2..];  // 32 i8 quants

            // 32 nibbles from qs for this subblock (16 bytes)
            // llama.cpp format: byte i low nibble = elem i, byte i high nibble = elem i+16
            let q4_sub = &qs[s * 16..];

            let mut sum_sub = 0i32;
            let mut sum_q8  = 0i32;
            for j in 0..16 {
                let lo = (q4_sub[j] & 0x0F) as i32;
                let hi = (q4_sub[j] >> 4) as i32;
                let q8_lo = q8qs[j] as i8 as i32;
                let q8_hi = q8qs[j + 16] as i8 as i32;
                sum_sub += lo * q8_lo + hi * q8_hi;
                sum_q8  += q8_lo + q8_hi;
            }

            sum += d_q8 * (dl * sum_sub as f32 - ml * sum_q8 as f32);
        }
    }

    sum
}

// ============================================================
// Q6_K × Q8_0 dot product
// Q6_K: 256 elements / superblock, 16 subblocks × 16 elements, 210 bytes
//   ql[128] = low 4 bits of each value (nibbles)
//   qh[64]  = high 2 bits of each value (packed 4 per byte)
//   scales[16] = direct i8 values (no 6-bit unpacking)
//   d = super-block scale
// No dmin/min term.
// ============================================================

#[inline]
pub fn dot_q6_k_q8_0(q6: &[u8], q8: &[u8]) -> f32 {
    debug_assert!(q6.len() % Q6KB == 0);
    dot_q6_k_q8_0_scalar(q6, q8)
}

fn dot_q6_k_q8_0_scalar(q6: &[u8], q8: &[u8]) -> f32 {
    // Translated from: llama.cpp/ggml/src/ggml-cpu/quants.c :: ggml_vec_dot_q6_K_q8_K_generic
    //
    // Step 1: dequantize Q6_K weights to a[256] (interleaved order matching dequantize_row_q6_K)
    // Step 2: element-wise dot with Q8_0 activation, per-subblock scale applied to groups of 16
    let n_super = q6.len() / Q6KB;
    let mut sumf = 0.0f32;

    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8[i * 8 * Q8B..];

        let d = block::fp16_to_f32(u16::from_le_bytes([q6b[208], q6b[209]]));
        let ql = &q6b[0..128];
        let qh = &q6b[128..192];
        let sc = &q6b[192..208];

        // Dequantize 256 weight values into a[256] (interleaved)
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
                    a[a_off + l + 0]  = (((ql0 & 0x0F) | ((qh_b       & 3) << 4)) - 32) as i8;
                    a[a_off + l + 32] = (((ql1 & 0x0F) | ((qh_b >> 2) & 3) << 4) - 32) as i8;
                    a[a_off + l + 64] = (((ql0 >> 4)   | ((qh_b >> 4) & 3) << 4) - 32) as i8;
                    a[a_off + l + 96] = (((ql1 >> 4)   | ((qh_b >> 6) & 3) << 4) - 32) as i8;
                }
                a_off += 128;
                ql_off += 64;
                qh_off += 32;
            }
        }

        // Dot with Q8_0 activation (8 blocks × 32 elements, each with its own fp16 scale)
        // 16 groups of 16 elements each, 2 groups share one Q8_0 block
        for g in 0..16 {
            let scale = sc[g] as i8 as f32;
            let blk = g / 2;           // Q8_0 block index (2 groups per block)
            let blk_off = blk * Q8B;
            let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8b[blk_off], q8b[blk_off + 1]]));
            let q8q = &q8b[blk_off + 2..];
            let mut sum_sub = 0i32;
            for k in 0..16 {
                let elem = g * 16 + k;
                let off = elem % 32;
                sum_sub += (a[elem] as i32) * (q8q[off] as i8 as i32);
            }
            sumf += d * scale * d_q8 * sum_sub as f32;
        }
    }

    sumf
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
        for j in 0..32 { y[yo + 2 + j] = (x[i * 32 + j] * id).round().clamp(-128.0, 127.0) as i8 as u8; }
    }
}

/// Quantize multiple rows directly into &mut [u8] buffer (no per-row Vec allocation).
pub fn quantize_row_q8_0_buf(x: &[f32], nt: usize, dim: usize, buf: &mut [u8]) {
    let rowb = (dim / 32) * Q8B;
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

    fn make_q4k_block(d: f32, dmin: f32, scales: &[u8; 8], mins: &[u8; 8], values: &[u8; 128]) -> Vec<u8> {
        let mut block = vec![0u8; Q4KB];
        let d_bits = f32_to_fp16(d).to_le_bytes();
        let dm_bits = f32_to_fp16(dmin).to_le_bytes();
        block[0] = d_bits[0]; block[1] = d_bits[1];
        block[2] = dm_bits[0]; block[3] = dm_bits[1];
        
        // Pack in llama.cpp get_scale_min_k4 format:
        // Bytes 0-3: scales[0-3] low 6 bits each
        // Bytes 4-7: mins[0-3] low 6 bits each
        // Bytes 8-11: high 2 bits cross-referenced
        let mut raw = [0u8; 12];

        raw[0] = scales[0] & 0x3F;
        raw[1] = scales[1] & 0x3F;
        raw[2] = scales[2] & 0x3F;
        raw[3] = scales[3] & 0x3F;
        raw[4] = mins[0] & 0x3F;
        raw[5] = mins[1] & 0x3F;
        raw[6] = mins[2] & 0x3F;
        raw[7] = mins[3] & 0x3F;
        raw[8]  = ((scales[4] & 0xF) << 0) | ((mins[4] & 0xF) << 4);
        raw[9]  = ((scales[5] & 0xF) << 0) | ((mins[5] & 0xF) << 4);
        raw[10] = ((scales[6] & 0xF) << 0) | ((mins[6] & 0xF) << 4);
        raw[11] = ((scales[7] & 0xF) << 0) | ((mins[7] & 0xF) << 4);
        raw[0] |= (scales[4] & 0x30) << 2;
        raw[1] |= (scales[5] & 0x30) << 2;
        raw[2] |= (scales[6] & 0x30) << 2;
        raw[3] |= (scales[7] & 0x30) << 2;
        raw[4] |= (mins[4] & 0x30) << 2;
        raw[5] |= (mins[5] & 0x30) << 2;
        raw[6] |= (mins[6] & 0x30) << 2;
        raw[7] |= (mins[7] & 0x30) << 2;
        
        block[4..16].copy_from_slice(&raw);
        block[16..144].copy_from_slice(values);
        block
    }

    fn make_q80_block(d: f32, values: &[i8; 32]) -> Vec<u8> {
        let mut block = vec![0u8; Q8B];
        let d_bits = f32_to_fp16(d).to_le_bytes();
        block[0] = d_bits[0]; block[1] = d_bits[1];
        for j in 0..32 { block[2 + j] = values[j] as u8; }
        block
    }

    fn reference_dot(q4: &[u8], q8: &[u8]) -> f32 {
        let n_super = q4.len() / Q4KB;
        let mut sum = 0.0f32;
        for i in 0..n_super {
            let q4b = &q4[i * Q4KB..];
            let q8b = &q8[i * 8 * Q8B..];
            let d = block::fp16_to_f32(u16::from_le_bytes([q4b[0], q4b[1]]));
            let dmin = block::fp16_to_f32(u16::from_le_bytes([q4b[2], q4b[3]]));
            
            let sc = <&[u8; 12]>::try_from(&q4b[4..16]).unwrap();
            let (scales_i, mins_i) = block::unpack_q4k_scales(sc);
            let scales: [f32; 8] = scales_i.map(|v| v as f32);
            let mins: [f32; 8] = mins_i.map(|v| v as f32);
            
            for s in 0..8usize {
                let sv = scales[s];
                let mv = mins[s];
                let dl = d * sv; let ml = dmin * mv;
                let q8blk = &q8b[s * Q8B..];
                let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8blk[0], q8blk[1]]));
                let q4_sub = &q4b[16 + s * 16..];
                for j in 0..16usize {
                    let lo = (q4_sub[j] & 0x0F) as f32;
                    let hi = (q4_sub[j] >> 4) as f32;
                    let y_lo = d_q8 * q8blk[2 + j] as i8 as f32;
                    let y_hi = d_q8 * q8blk[2 + j + 16] as i8 as f32;
                    let w_lo = dl * lo - ml;
                    let w_hi = dl * hi - ml;
                    sum += w_lo * y_lo + w_hi * y_hi;
                }
            }
        }
        sum
    }

    #[test]
    fn test_q4k_dot_simple() {
        let scales = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mins = [0u8; 8];
        let mut values = [0u8; 128];
        for i in 0..128 { values[i] = (i % 16) as u8; }
        let q4k = make_q4k_block(0.1, 0.0, &scales, &mins, &values);
        let mut q8_vals = [0i8; 32];
        for i in 0..32 { q8_vals[i] = (i as i8) - 16; }
        let mut q8d = Vec::new();
        for _ in 0..8 { q8d.extend_from_slice(&make_q80_block(0.05, &q8_vals)); }
        let r = reference_dot(&q4k, &q8d);
        let t = dot_q4_k_q8_0(&q4k, &q8d);
        assert!((r - t).abs() < 0.01, "diff={}", (r - t).abs());
    }

    #[test]
    fn test_q4k_dot_nonzero_mins() {
        let scales = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mins = [1u8, 2, 3, 4, 5, 6, 7, 8];
        let mut values = [0u8; 128];
        for i in 0..128 { values[i] = (i % 16) as u8; }
        let q4k = make_q4k_block(0.1, 0.05, &scales, &mins, &values);
        let mut q8_vals = [0i8; 32];
        for i in 0..32 { q8_vals[i] = (i as i8) - 16; }
        let mut q8d = Vec::new();
        for _ in 0..8 { q8d.extend_from_slice(&make_q80_block(0.05, &q8_vals)); }
        let r = reference_dot(&q4k, &q8d);
        let t = dot_q4_k_q8_0(&q4k, &q8d);
        assert!((r - t).abs() < 0.01, "diff={}", (r - t).abs());
    }

    #[test]
    fn test_q4k_dot_random() {
        let mut rng: u32 = 12345;
        let mut next = || -> u8 { rng = rng.wrapping_mul(1103515245).wrapping_add(12345); (rng >> 16) as u8 };
        let sc: [u8; 8] = std::array::from_fn(|_| next() % 64);
        let mn: [u8; 8] = std::array::from_fn(|_| next() % 64);
        let vl: [u8; 128] = std::array::from_fn(|_| next());
        let q4k = make_q4k_block(0.0123, 0.0045, &sc, &mn, &vl);
        let mut q8d = Vec::new();
        for _ in 0..8 {
            let qv: [i8; 32] = std::array::from_fn(|_| next() as i8);
            q8d.extend_from_slice(&make_q80_block(0.03, &qv));
        }
        let r = reference_dot(&q4k, &q8d);
        let t = dot_q4_k_q8_0(&q4k, &q8d);
        assert!((r - t).abs() < 0.01, "diff={}", (r - t).abs());
    }

    #[test]
    fn test_q4k_dot_multi_superblocks() {
        let mut rng: u32 = 99999;
        let mut next = || -> u8 { rng = rng.wrapping_mul(1103515245).wrapping_add(12345); (rng >> 16) as u8 };
        let mut q4m = Vec::new();
        let mut q8m = Vec::new();
        for _ in 0..3 {
            let sc: [u8; 8] = std::array::from_fn(|_| next() % 64);
            let mn: [u8; 8] = std::array::from_fn(|_| next() % 64);
            let vl: [u8; 128] = std::array::from_fn(|_| next());
            q4m.extend_from_slice(&make_q4k_block(0.05, 0.02, &sc, &mn, &vl));
            for _ in 0..8 {
                let qv: [i8; 32] = std::array::from_fn(|_| next() as i8);
                q8m.extend_from_slice(&make_q80_block(0.04, &qv));
            }
        }
        let r = reference_dot(&q4m, &q8m);
        let t = dot_q4_k_q8_0(&q4m, &q8m);
        assert!((r - t).abs() < 0.01, "diff={}", (r - t).abs());
    }

    fn make_q6k_block(d: f32, scales: &[i8; 16], ql: &[u8; 128], qh: &[u8; 64]) -> Vec<u8> {
        let mut block = vec![0u8; Q6KB];
        block[0..128].copy_from_slice(ql);
        block[128..192].copy_from_slice(qh);
        for i in 0..16 { block[192 + i] = scales[i] as u8; }
        let d_bits = f32_to_fp16(d).to_le_bytes();
        block[208] = d_bits[0]; block[209] = d_bits[1];
        block
    }

    fn reference_dot_q6k(q6: &[u8], q8: &[u8]) -> f32 {
        let n_super = q6.len() / Q6KB;
        let mut sumf = 0.0f32;
        for i in 0..n_super {
            let q6b = &q6[i * Q6KB..];
            let q8b = &q8[i * 8 * Q8B..];
            let d = block::fp16_to_f32(u16::from_le_bytes([q6b[208], q6b[209]]));
            let ql = &q6b[0..128];
            let qh = &q6b[128..192];
            let sc = &q6b[192..208];
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
                        a[a_off + l + 0]  = (((ql0 & 0x0F) | ((qh_b       & 3) << 4)) - 32) as i8;
                        a[a_off + l + 32] = (((ql1 & 0x0F) | ((qh_b >> 2) & 3) << 4) - 32) as i8;
                        a[a_off + l + 64] = (((ql0 >> 4)   | ((qh_b >> 4) & 3) << 4) - 32) as i8;
                        a[a_off + l + 96] = (((ql1 >> 4)   | ((qh_b >> 6) & 3) << 4) - 32) as i8;
                    }
                    a_off += 128; ql_off += 64; qh_off += 32;
                }
            }
            for g in 0..16 {
                let scale = sc[g] as i8 as f32;
                let blk = g / 2;
                let blk_off = blk * Q8B;
                let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8b[blk_off], q8b[blk_off + 1]]));
                let q8q = &q8b[blk_off + 2..];
                let mut sum_sub = 0i32;
                for k in 0..16 {
                    let elem = g * 16 + k;
                    let off = elem % 32;
                    sum_sub += (a[elem] as i32) * (q8q[off] as i8 as i32);
                }
                sumf += d * scale * d_q8 * sum_sub as f32;
            }
        }
        sumf
    }

    #[test]
    fn test_q6k_dot_simple() {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for i in 0..128 { ql[i] = ((i * 7 + 3) % 16) as u8; }
        for i in 0..64 { qh[i] = ((i * 3 + 1) % 4) as u8; }
        let scales: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8];
        let q6k = make_q6k_block(0.1, &scales, &ql, &qh);
        let mut q8_vals = [0i8; 32];
        for i in 0..32 { q8_vals[i] = (i as i8) - 16; }
        let mut q8d = Vec::new();
        for _ in 0..8 { q8d.extend_from_slice(&make_q80_block(0.05, &q8_vals)); }
        let r = reference_dot_q6k(&q6k, &q8d);
        let t = dot_q6_k_q8_0(&q6k, &q8d);
        assert!((r - t).abs() < 0.01, "diff={}", (r - t).abs());
    }

    // Truly independent reference: dequantize to f32, then dot
    fn independent_dot_q4k(q4: &[u8], q8: &[u8]) -> f32 {
        let n_super = q4.len() / Q4KB;
        let mut sum = 0.0f32;
        for i in 0..n_super {
            let q4b = &q4[i * Q4KB..];
            let q8b = &q8[i * 8 * Q8B..];
            let d = block::fp16_to_f32(u16::from_le_bytes([q4b[0], q4b[1]]));
            let dmin = block::fp16_to_f32(u16::from_le_bytes([q4b[2], q4b[3]]));
            let sc = &q4b[4..16];
            
            // Extract scales and mins using get_scale_min_k4 format
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for j in 0..4 {
                scales[j] = (sc[j] & 0x3F) as f32;
                mins[j] = (sc[j + 4] & 0x3F) as f32;
            }
            for j in 4..8 {
                scales[j] = ((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4)) as f32;
                mins[j] = ((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4)) as f32;
            }
            
            // Dequantize 256 weights to f32
            let mut w = [0.0f32; 256];
            for s in 0..8 {
                let qs = &q4b[16 + s * 16..];
                let base = s * 32;
                let dl = d * scales[s];
                let ml = dmin * mins[s];
                for j in 0..16 {
                    w[base + j] = dl * (qs[j] & 0x0F) as f32 - ml;
                    w[base + j + 16] = dl * (qs[j] >> 4) as f32 - ml;
                }
            }
            
            // Dequantize 256 activations to f32
            let mut x = [0.0f32; 256];
            for s in 0..8 {
                let q8blk = &q8b[s * Q8B..];
                let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8blk[0], q8blk[1]]));
                let base = s * 32;
                for j in 0..32 {
                    x[base + j] = d_q8 * q8blk[2 + j] as i8 as f32;
                }
            }
            
            // Compute dot product
            for j in 0..256 {
                sum += w[j] * x[j];
            }
        }
        sum
    }

    #[test]
    fn test_q4k_independent_reference() {
        let mut rng: u32 = 12345;
        let mut next = || -> u8 { rng = rng.wrapping_mul(1103515245).wrapping_add(12345); (rng >> 16) as u8 };
        let sc: [u8; 8] = std::array::from_fn(|_| next() % 64);
        let mn: [u8; 8] = std::array::from_fn(|_| next() % 64);
        let vl: [u8; 128] = std::array::from_fn(|_| next());
        let q4k = make_q4k_block(0.0123, 0.0045, &sc, &mn, &vl);
        let mut q8d = Vec::new();
        for _ in 0..8 {
            let qv: [i8; 32] = std::array::from_fn(|_| next() as i8);
            q8d.extend_from_slice(&make_q80_block(0.03, &qv));
        }
        let indep = independent_dot_q4k(&q4k, &q8d);
        let impl_result = dot_q4_k_q8_0(&q4k, &q8d);
        assert!((indep - impl_result).abs() < 0.01, "Q4K independent ref mismatch: diff={}", (indep - impl_result).abs());
    }

    // Truly independent reference for Q6_K: dequantize to f32, then dot
    fn independent_dot_q6k(q6: &[u8], q8: &[u8]) -> f32 {
        let n_super = q6.len() / Q6KB;
        let mut sum = 0.0f32;
        for i in 0..n_super {
            let q6b = &q6[i * Q6KB..];
            let q8b = &q8[i * 8 * Q8B..];
            let d = block::fp16_to_f32(u16::from_le_bytes([q6b[208], q6b[209]]));
            let ql = &q6b[0..128];
            let qh = &q6b[128..192];
            let sc = &q6b[192..208];
            
            // Dequantize 256 weights to f32 (matching llama.cpp dequantize_row_q6_K)
            let mut w = [0.0f32; 256];
            for n in 0..2 {
                let ql_off = n * 64;
                let qh_off = n * 32;
                let out_off = n * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let si = is + n * 8;
                    let q0 = (((ql[ql_off + l] & 0xF) as i32) | ((((qh[qh_off + l] >> 0) & 3) as i32) << 4)) - 32;
                    let q1 = (((ql[ql_off + l + 32] & 0xF) as i32) | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4)) - 32;
                    let q2 = (((ql[ql_off + l] >> 4) as i32) | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4)) - 32;
                    let q3 = (((ql[ql_off + l + 32] >> 4) as i32) | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4)) - 32;
                    w[out_off + l] = d * (sc[si + 0] as i8 as f32) * q0 as f32;
                    w[out_off + l + 32] = d * (sc[si + 2] as i8 as f32) * q1 as f32;
                    w[out_off + l + 64] = d * (sc[si + 4] as i8 as f32) * q2 as f32;
                    w[out_off + l + 96] = d * (sc[si + 6] as i8 as f32) * q3 as f32;
                }
            }
            
            // Dequantize 256 activations to f32
            let mut x = [0.0f32; 256];
            for s in 0..8 {
                let q8blk = &q8b[s * Q8B..];
                let d_q8 = block::fp16_to_f32(u16::from_le_bytes([q8blk[0], q8blk[1]]));
                let base = s * 32;
                for j in 0..32 {
                    x[base + j] = d_q8 * q8blk[2 + j] as i8 as f32;
                }
            }
            
            // Compute dot product
            for j in 0..256 {
                sum += w[j] * x[j];
            }
        }
        sum
    }

    #[test]
    fn test_q6k_independent_reference() {
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for i in 0..128 { ql[i] = ((i * 7 + 3) % 16) as u8; }
        for i in 0..64 { qh[i] = ((i * 3 + 1) % 4) as u8; }
        let scales: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, -1, -2, -3, -4, -5, -6, -7, -8];
        let q6k = make_q6k_block(0.1, &scales, &ql, &qh);
        let mut q8_vals = [0i8; 32];
        for i in 0..32 { q8_vals[i] = (i as i8) - 16; }
        let mut q8d = Vec::new();
        for _ in 0..8 { q8d.extend_from_slice(&make_q80_block(0.05, &q8_vals)); }
        let indep = independent_dot_q6k(&q6k, &q8d);
        let impl_result = dot_q6_k_q8_0(&q6k, &q8d);
        assert!((indep - impl_result).abs() < 0.01, "Q6K independent ref mismatch: diff={}", (indep - impl_result).abs());
    }

    #[test]
    fn test_q6k_dot_random() {
        let mut rng: u32 = 54321;
        let mut next = || -> u8 { rng = rng.wrapping_mul(1103515245).wrapping_add(12345); (rng >> 16) as u8 };
        let ql: [u8; 128] = std::array::from_fn(|_| next());
        let qh: [u8; 64] = std::array::from_fn(|_| next());
        let sc: [i8; 16] = std::array::from_fn(|_| next() as i8);
        let q6k = make_q6k_block(0.025, &sc, &ql, &qh);
        let mut q8d = Vec::new();
        for _ in 0..8 {
            let qv: [i8; 32] = std::array::from_fn(|_| next() as i8);
            q8d.extend_from_slice(&make_q80_block(0.03, &qv));
        }
        let r = reference_dot_q6k(&q6k, &q8d);
        let t = dot_q6_k_q8_0(&q6k, &q8d);
        assert!((r - t).abs() < 0.01, "diff={}", (r - t).abs());
    }

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
}
