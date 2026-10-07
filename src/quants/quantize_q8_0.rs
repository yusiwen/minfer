//! Q8_0 activation quantization — the weight-activation quantizer and the
//! read/write pair C4's packed KV region shares with the graph
//! (`quantize_row_q8_0_into` / `dequantize_row_q8_0`).
use super::*;

/// f16→f32 conversion with correct IEEE 754 handling for all cases
/// (zero, subnormal, normal, infinity, NaN). Only used by the x86_64 AVX2
/// kernels (the scalar paths use `block::fp16_to_f32`).
#[cfg(target_arch = "x86_64")]
#[inline(always)]
pub(super) fn f16_to_f32_bits(bits: u16) -> f32 {
    let i = bits as u32;
    let sign = (i & 0x8000) << 16;
    let exp = (i >> 10) & 0x1F;
    let mant = i & 0x3FF;
    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        let pos = 31 - mant.leading_zeros();
        return f32::from_bits(sign | ((103 + pos) << 23) | ((mant - (1 << pos)) << (23 - pos)));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F800000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 112) << 23) | (mant << 13))
}

// ============================================================
// Quantize f32 → Q8_0 bytes (raw &[u8], no struct types)
// ============================================================
fn quantize_row_q8_0_to(x: &[f32], y: &mut [u8]) {
    let k = x.len();
    #[cfg(target_arch = "x86_64")]
    {
        if avx2_enabled() {
            unsafe { quantize_avx2(x, y, k) };
            return;
        }
    }
    quantize_scalar(x, y, k);
}

/// Quantize one row into caller-owned Q8_0 bytes (`y.len() == x.len()/32 * Q8B`).
///
/// C4's KV store writes a whole row at a time into a packed region, so it needs the
/// same routine the weight quantizer uses without a per-row allocation.
pub(crate) fn quantize_row_q8_0_into(x: &[f32], y: &mut [u8]) {
    assert_eq!(
        y.len(),
        (x.len() / 32) * Q8B,
        "Q8_0 row buffer must be {} bytes for {} elements",
        (x.len() / 32) * Q8B,
        x.len()
    );
    debug_assert_eq!(x.len() % 32, 0);
    quantize_row_q8_0_to(x, y);
}

/// Dequantize one Q8_0 row (`x.len()` = whole 34-byte blocks) into `out`.
///
/// The exact inverse of [`quantize_row_q8_0_into`] up to the per-block rounding, and
/// the read side of a packed KV region: `out[i] = d_b * q[i]` with `d_b` the block's
/// f16 scale.
pub fn dequantize_row_q8_0(x: &[u8], out: &mut [f32]) {
    let nb = x.len() / Q8B;
    assert!(
        out.len() >= nb * 32,
        "Q8_0 dequant needs {} outputs for {nb} blocks, got {}",
        nb * 32,
        out.len()
    );
    for b in 0..nb {
        let off = b * Q8B;
        let d = block::fp16_to_f32(u16::from_le_bytes([x[off], x[off + 1]]));
        for j in 0..32 {
            out[b * 32 + j] = d * x[off + 2 + j] as i8 as f32;
        }
    }
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

/// Quantize multiple rows directly into &mut [u8] buffer (no per-row Vec allocation).
pub fn quantize_row_q8_0_buf(x: &[f32], nt: usize, dim: usize, buf: &mut [u8]) {
    let rowb = (dim / 32) * Q8B;
    #[cfg(feature = "debug_dump")]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static DUMPED: AtomicBool = AtomicBool::new(false);
        if !DUMPED.swap(true, Ordering::Relaxed) && nt > 0 && dim >= 32 {
            let mut am = 0.0f32;
            for j in 0..32 {
                am = am.max(x[j].abs());
            }
            let d = am / 127.0f32;
            let q0 = (x[0] / d).round().clamp(-128.0, 127.0) as i8;
            let q1 = (x[1] / d).round().clamp(-128.0, 127.0) as i8;
            let q16 = (x[16] / d).round().clamp(-128.0, 127.0) as i8;
            crate::dump::maybe_dump_text(
                "minfer_dump_q8_quant_verify",
                &format!(
                    "amax={:e} d={:e} x[0]={:e} x[1]={:e} x[16]={:e} q[0]={} q[1]={} q[16]={}",
                    am, d, x[0], x[1], x[16], q0, q1, q16
                ),
            );
        }
    }
    for t in 0..nt {
        quantize_row_q8_0_to(
            &x[t * dim..(t + 1) * dim],
            &mut buf[t * rowb..(t + 1) * rowb],
        );
    }
}
