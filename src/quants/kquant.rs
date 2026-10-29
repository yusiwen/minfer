//! K-quant × Q8_K dot products (Q4_K / Q5_K / Q6_K; 256 elements per
//! superblock, the activation carrying precomputed per-subblock bsums).
use super::*;

/// Q4_K × Q8_K dot (256 elements/superblock; activation carries bsums).
#[inline]
pub fn dot_q4_k_q8_k(q4: &[u8], q8k: &[u8]) -> f32 {
    debug_assert!(q4.len() % Q4KB == 0);
    debug_assert!(q8k.len() % crate::block::Q8KB == 0);
    #[cfg(target_arch = "x86_64")]
    {
        // Dispatch order: AVX-512/VNNI → AVX2+FMA → scalar. Both x86 gates are
        // runtime and honour MINFER_NO_AVX512 / MINFER_NO_AVX2.
        if avx512_enabled() {
            return unsafe { dot_q4_k_q8_k_avx512(q4, q8k) };
        }
        if avx2_enabled() {
            return unsafe { dot_q4_k_q8_k_avx2(q4, q8k) };
        }
    }
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
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            return unsafe { dot_q6_k_q8_k_avx512(q6, q8k) };
        }
        if avx2_enabled() {
            return unsafe { dot_q6_k_q8_k_avx2(q6, q8k) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q6_k_q8_k_neon(q6, q8k) };
        }
    }
    dot_q6_k_q8_k_scalar(q6, q8k)
}

/// Q6_K × Q8_K — the reference order of ADR-0061: eight i32 lanes per
/// superblock (per-group scales in the lane sums, the -32 offset corrected from
/// the q8_K bsums), one f32 accumulator per lane across the row, and a fixed
/// reduction tree. The SIMD kernels reproduce this bitwise; the order *is* the
/// contract, so do not "simplify" it back into a per-group fold.
pub(super) fn dot_q6_k_q8_k_scalar(q6: &[u8], q8k: &[u8]) -> f32 {
    let n_super = q6.len() / Q6KB;
    let mut acc = [0.0f32; 8];
    for i in 0..n_super {
        let q6b = &q6[i * Q6KB..];
        let q8b = &q8k[i * crate::block::Q8KB..];
        let d = block::fp16_to_f32(u16::from_le_bytes([q6b[208], q6b[209]]))
            * block::fp16_to_f32(u16::from_le_bytes([q8b[0], q8b[1]]));
        let ql = &q6b[0..128];
        let qh = &q6b[128..192];
        let sc = &q6b[192..208];
        // Decode to a[256] as the unsigned 0..63 code (interleaved, the same
        // element order the SIMD kernels see).
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
                    a[a_off + l + 0] = ((ql0 & 0x0F) | ((qh_b & 3) << 4)) as i8;
                    a[a_off + l + 32] = ((ql1 & 0x0F) | ((qh_b >> 2) & 3) << 4) as i8;
                    a[a_off + l + 64] = ((ql0 >> 4) | ((qh_b >> 4) & 3) << 4) as i8;
                    a[a_off + l + 96] = ((ql1 >> 4) | ((qh_b >> 6) & 3) << 4) as i8;
                }
                a_off += 128;
                ql_off += 64;
                qh_off += 32;
            }
        }
        // Step 1: lane j takes the even-group contributions at sub-position j,
        // lane 4+j the odd-group ones (32 consecutive elements per chunk).
        let mut lane = [0i32; 8];
        for c in 0..8 {
            let s_even = sc[2 * c] as i8 as i32;
            let s_odd = sc[2 * c + 1] as i8 as i32;
            for j in 0..4 {
                let mut p_even = 0i32;
                let mut p_odd = 0i32;
                for e in 0..4 {
                    // a[] holds the unsigned code, exactly as the SIMD
                    // kernels see it; the step-2 correction is not applied
                    // twice.
                    let i0 = 32 * c + 4 * j + e;
                    let i1 = i0 + 16;
                    p_even += (a[i0] as i32) * (q8b[2 + i0] as i8 as i32);
                    p_odd += (a[i1] as i32) * (q8b[2 + i1] as i8 as i32);
                }
                lane[j] += s_even * p_even;
                lane[4 + j] += s_odd * p_odd;
            }
        }
        // Step 2: the -32 offset, once per lane.
        for k in 0..8 {
            let b0 = i16::from_le_bytes([q8b[274 + 4 * k], q8b[274 + 4 * k + 1]]) as i32;
            let b1 = i16::from_le_bytes([q8b[274 + 4 * k + 2], q8b[274 + 4 * k + 3]]) as i32;
            let s0 = sc[2 * k] as i8 as i32;
            let s1 = sc[2 * k + 1] as i8 as i32;
            lane[k] -= 32 * (s0 * b0 + s1 * b1);
        }
        // Step 3: one float fold per lane, held across the whole row.
        for k in 0..8 {
            acc[k] += d * (lane[k] as f32);
        }
    }
    // Step 4: the fixed reduction tree.
    ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]))
}

/// Q5_K × Q8_K dot (like Q4_K with high bits).
#[inline]
pub fn dot_q5_k_q8_k(q5: &[u8], q8k: &[u8]) -> f32 {
    debug_assert!(q5.len() % 176 == 0);
    debug_assert!(q8k.len() % crate::block::Q8KB == 0);
    #[cfg(target_arch = "x86_64")]
    {
        if avx512_enabled() {
            return unsafe { dot_q5_k_q8_k_avx512(q5, q8k) };
        }
        if avx2_enabled() {
            return unsafe { dot_q5_k_q8_k_avx2(q5, q8k) };
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if neon_enabled() {
            return unsafe { dot_q5_k_q8_k_neon(q5, q8k) };
        }
    }
    dot_q5_k_q8_k_scalar(q5, q8k)
}

pub(super) fn dot_q5_k_q8_k_scalar(q5: &[u8], q8k: &[u8]) -> f32 {
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
