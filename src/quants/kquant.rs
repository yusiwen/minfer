//! K-quant × Q8_K dot products (Q4_K / Q5_K / Q6_K; 256 elements per
//! superblock, the activation carrying precomputed per-subblock bsums).
use super::*;

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

pub(super) fn dot_q4_k_q8_k_scalar(q4: &[u8], q8k: &[u8]) -> f32 {
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
            let b0 =
                i16::from_le_bytes([q8b[274 + 2 * (2 * s)], q8b[274 + 2 * (2 * s) + 1]]) as i32;
            let b1 =
                i16::from_le_bytes([q8b[274 + 2 * (2 * s + 1)], q8b[274 + 2 * (2 * s + 1) + 1]])
                    as i32;
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

pub(super) fn dot_q6_k_q8_k_scalar(q6: &[u8], q8k: &[u8]) -> f32 {
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
            let b0 =
                i16::from_le_bytes([q8b[274 + 2 * (2 * s)], q8b[274 + 2 * (2 * s) + 1]]) as i32;
            let b1 =
                i16::from_le_bytes([q8b[274 + 2 * (2 * s + 1)], q8b[274 + 2 * (2 * s + 1) + 1]])
                    as i32;
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
