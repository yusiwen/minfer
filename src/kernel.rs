// Compute kernel dispatch layer.
//   CPU (AVX2/scalar) is always available as fallback.
//   MPS (Apple Silicon GPU) is enabled at runtime when Metal is available.

use crate::tensor::{Tensor, TensorType};
use crate::block::{Q4B, Q41B, Q8B, Q4KB, Q6KB};

/// Minimum batch size for GPU dispatch (llama.cpp uses `op_offload_min_batch_size = 32`).
/// Below this threshold, CPU is often faster due to kernel launch overhead.
const GPU_MIN_BATCH: usize = 1;

/// Quantized matmul with f32 activation.
/// GPU path passes f32 directly; CPU path quantizes internally.
pub fn quant_matmul_f32(
    w: &Tensor, x: &[f32], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    #[cfg(target_os = "macos")]
    if nt >= GPU_MIN_BATCH {
        if let Some(mps) = crate::metal::MpsState::get() {
            if mps.has_weight(&w.name) {
                return mps.quant_matmul_f32(w, x, out, od, id, nt);
            }
        }
    }
    #[cfg(feature = "cuda")]
    if nt >= GPU_MIN_BATCH {
        if let Some(cuda) = crate::cuda::CudaState::get() {
            if cuda.has_weight(&w.name) {
                return cuda.quant_matmul_f32(w, x, out, od, id, nt);
            }
        }
    }
    cpu_quant_matmul_f32(w, x, out, od, id, nt)
}

/// Quantize `x` once and run several Q4_0 matmuls that share the same activation.
/// This reduces per-matmul command-buffer and upload overhead.
pub fn quant_matmul_f32_batch(
    mats: &mut [(/*weight*/ &Tensor, /*output*/ &mut [f32], /*od*/ usize)],
    x: &[f32], id: usize, nt: usize,
) {
    #[cfg(target_os = "macos")]
    if nt >= GPU_MIN_BATCH {
        if let Some(mps) = crate::metal::MpsState::get() {
            if mats.iter().all(|(w, _out, _od)| mps.has_weight(&w.name)) {
                return mps.quant_matmul_f32_batch(mats, x, id, nt);
            }
        }
    }
    #[cfg(feature = "cuda")]
    if nt >= GPU_MIN_BATCH {
        if let Some(cuda) = crate::cuda::CudaState::get() {
            if mats.iter().all(|(w, _out, _od)| cuda.has_weight(&w.name)) {
                return cuda.quant_matmul_f32_batch(mats, x, id, nt);
            }
        }
    }
    // CPU fallback: run each matmul independently.
    for mat in mats.iter_mut() {
        cpu_quant_matmul_f32(mat.0, x, mat.1, mat.2, id, nt);
    }
}

/// Q8_0 activation matmul (kept for backward compat, now delegates to f32 path).
pub fn quant_matmul(
    w: &Tensor, x: &[u8], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    cpu_quant_matmul(w, x, out, od, id, nt)
}

/// CPU fallback for f32 activation: quantize → call existing dot product.
pub fn cpu_quant_matmul_f32(
    w: &Tensor, x: &[f32], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    if w.ttype == TensorType::Q5_K {
        return cpu_q5_k_matmul_f32(w, x, out, od, id, nt);
    }
    if w.ttype == TensorType::Q5_0 {
        return cpu_q5_0_matmul_f32(w, x, out, od, id, nt);
    }
    let nbe = id / 32;
    let mut qb = vec![0u8; nt * nbe * Q8B];
    crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut qb);
    cpu_quant_matmul(w, &qb, out, od, id, nt)
}

pub fn cpu_quant_matmul(
    w: &Tensor, x: &[u8], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    match w.ttype {
        TensorType::Q4_0 => {
            let nb = id / 32;
            let ws = nb * Q4B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_0_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        TensorType::Q4_1 => {
            let nb = id / 32;
            let ws = nb * Q41B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_1_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        TensorType::Q4_K => {
            let nk = id / 256;
            let ws = nk * Q4KB;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_k_q8_0(
                        wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
                }
            }
        }
        TensorType::Q6_K => {
            let nk = id / 256;
            let ws = nk * Q6KB;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q6_k_q8_0(
                        wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
                }
            }
        }
        TensorType::Q8_0 => {
            let nb = id / 32;
            let ws = nb * Q8B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q8_0_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in quant_matmul", w.ttype),
    }
}

/// Q5_K × f32 matmul: dequantize Q5_K weights on-the-fly and compute f32 dot product.
/// Block: d(f16,2) + dmin(f16,2) + scales(u8,12) + qh(u8,32) + qs(u8,128) = 176 bytes.
/// Each block dequantizes 256 elements using 5 bits (4 low + 1 high) per weight.
fn cpu_q5_k_matmul_f32(
    w: &Tensor, x: &[f32], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    use crate::block::fp16_to_f32;
    let n_super = id / 256;
    let ws = n_super * 176;
    let wb = w.data();
    for o in 0..od {
        let wrow = &wb[o * ws..(o + 1) * ws];
        for t in 0..nt {
            let a_base = t * id;
            let mut sum = 0.0f32;
            for s in 0..n_super {
                let off = s * 176;
                let d  = fp16_to_f32(u16::from_le_bytes([wrow[off],     wrow[off + 1]]));
                let dm = fp16_to_f32(u16::from_le_bytes([wrow[off + 2], wrow[off + 3]]));
                let sc_arr: &[u8; 12] = wrow[off + 4..off + 16].try_into().unwrap();
                let (sc, mn) = crate::block::unpack_q4k_scales(sc_arr);
                let qh = &wrow[off + 16..off + 48];
                let qs = &wrow[off + 48..off + 176];
                for sub in 0..8 {
                    let dl = d * sc[sub] as f32;
                    let ml = dm * mn[sub] as f32;
                    let qs_sub = &qs[sub * 16..];
                    for j in 0..16 {
                        let h0 = ((qh[sub * 4 + j / 8] >> (j % 8)) & 1) as u8;
                        let h1 = ((qh[sub * 4 + j / 8 + 2] >> (j % 8)) & 1) as u8;
                        let w0 = (qs_sub[j] & 0x0F) as f32 + 16.0 * h0 as f32;
                        let w1 = ((qs_sub[j] >> 4) & 0x0F) as f32 + 16.0 * h1 as f32;
                        let off_a = a_base + s * 256 + sub * 32;
                        sum += dl * w0 * x[off_a + j] - ml * x[off_a + j];
                        sum += dl * w1 * x[off_a + j + 16] - ml * x[off_a + j + 16];
                    }
                }
            }
            out[t * od + o] = sum;
        }
    }
}

/// Q5_0 × f32 matmul: dequantize Q5_0 weights on-the-fly and compute f32 dot product.
/// Block: d(f16,2) + qh(u8,4) + qs(u8,16) = 22 bytes per 32 elements.
/// Dequant: val = d * ((nibble | (high_bit << 4)) - 16).
fn cpu_q5_0_matmul_f32(
    w: &Tensor, x: &[f32], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    use crate::block::fp16_to_f32;
    let nb = id / 32;
    let ws = nb * 22;
    let wb = w.data();
    for o in 0..od {
        let wrow = &wb[o * ws..(o + 1) * ws];
        for t in 0..nt {
            let a_base = t * id;
            let mut sum = 0.0f32;
            for b in 0..nb {
                let off = b * 22;
                let d = fp16_to_f32(u16::from_le_bytes([wrow[off], wrow[off + 1]]));
                let qh = u32::from_le_bytes([wrow[off+2], wrow[off+3], wrow[off+4], wrow[off+5]]);
                let qs = &wrow[off + 6..off + 22];
                for j in 0..16 {
                    let xh0 = ((qh >> j) & 1) as i32;
                    let xh1 = ((qh >> (j + 16)) & 1) as i32;
                    let w0 = (((qs[j] & 0x0F) as i32 | (xh0 << 4)) - 16) as f32 * d;
                    let w1 = ((((qs[j] >> 4) & 0x0F) as i32 | (xh1 << 4)) - 16) as f32 * d;
                    sum += w0 * x[a_base + b * 32 + j];
                    sum += w1 * x[a_base + b * 32 + j + 16];
                }
            }
            out[t * od + o] = sum;
        }
    }
}
