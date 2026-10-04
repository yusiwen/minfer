//! Q8_K activation quantization (256-element blocks, llama.cpp's format for
//! K-quant weights), plus the NEON fast path for it.
// The scalar body is fully qualified; only the NEON branch reaches out, so the
// import is gated the same way (an ungated `use super::*` is unused on x86_64).
#[cfg(target_arch = "aarch64")]
use super::neon::enabled as neon_enabled;

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
                    quantize_row_q8_k_buf_neon(
                        &x[t * dim..(t + 1) * dim],
                        &mut buf[t * rowb..(t + 1) * rowb],
                    );
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
                out[o + 274 + 2 * j..o + 274 + 2 * j + 2].copy_from_slice(&bsums[j].to_le_bytes());
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
