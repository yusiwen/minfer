//! Q5_0 / Q5_1 × Q8_0 dot products.
//!
//! Q5_0: 32 elements / block, 22 bytes = d(f16,2) + qh(u8,4) + qs(u8,16).
//! Q5_1: 32 elements / block, 24 bytes = d(f16,2) + m(f16,2) + qh(u32,4) + qs(u8,16);
//! weight = d * ((nibble | (high_bit << 4)) - 16) + m.
use super::*;

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
        let mut sum_q8 = 0i32;
        for j in 0..16 {
            let u_lo = (qs[j] & 0x0F) as i32 | (((qh >> j) & 1) as i32) << 4;
            let u_hi = ((qs[j] >> 4) & 0x0F) as i32 | (((qh >> (j + 16)) & 1) as i32) << 4;
            let q8_lo = q8b[2 + j] as i8 as i32;
            let q8_hi = q8b[2 + j + 16] as i8 as i32;
            sum_sub += u_lo * q8_lo + u_hi * q8_hi;
            sum_q8 += q8_lo + q8_hi;
        }
        // Q5_1 dequant: val = d_q5 * unsigned_5bit + m_q5 (no -16 offset!)
        // dot = d_q8 * d_q5 * Σ(u×q) + d_q8 * m_q5 * Σ(q)
        s += d_q8 * (d_q5 * sum_sub as f32 + m_q5 * sum_q8 as f32);
    }
    s
}
