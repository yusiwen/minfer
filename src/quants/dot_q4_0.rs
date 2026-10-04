//! Q4_0 × Q8_0 dot product (raw `&[u8]` interface, no `n_blocks` param —
//! slice length-based).
use super::*;

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
