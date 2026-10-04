//! Q4_1 × Q8_0 dot product: value = q * d + m (unsigned nibbles 0..15, no
//! centering).
use super::*;

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
        let d = block::fp16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
        let m = block::fp16_to_f32(u16::from_le_bytes([xb[2], xb[3]]));
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
