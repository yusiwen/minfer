// Compute kernel dispatch layer.
//   CPU (AVX2/scalar) is always available as fallback.
//   MPS (Apple Silicon GPU) is enabled at runtime when Metal is available.
// Translated from: llama.cpp/ggml/src/ggml-metal (ops dispatch pattern)

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
