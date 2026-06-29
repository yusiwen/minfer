// Compute kernel dispatch layer.
//   CPU (AVX2/scalar) is always available as fallback.
//   MPS (Apple Silicon GPU) is enabled at runtime when Metal is available.
// Translated from: llama.cpp/ggml/src/ggml-metal (ops dispatch pattern)

use crate::tensor::{Tensor, TensorType};

// Block size constants (matching avx2.rs)
pub const Q4B: usize = 18;
pub const Q41B: usize = 20;
pub const Q8B: usize = 34;
pub const Q4KB: usize = 144;
pub const Q6KB: usize = 210;

/// Quantized matmul: dispatch to MPS (Apple Silicon) or CPU (AVX2/scalar).
/// `w`:   weight tensor (Q4_0/Q4_1/Q4_K/Q6_K/Q8_0 quantized)
/// `x`:   Q8_0 quantized activation buffer (nt rows × id/32*Q8B bytes each)
/// `out`: output f32 buffer (nt rows × od)
/// `od`:  output dimension (weight rows)
/// `id`:  input dimension (activation columns)
/// `nt`:  number of tokens (activation rows)
pub fn quant_matmul(
    w: &Tensor, x: &[u8], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    #[cfg(target_os = "macos")]
    if let Some(mps) = crate::metal::MpsState::get() {
        if mps.has_weight(&w.name) {
            return mps.quant_matmul(w, x, out, od, id, nt);
        }
    }
    cpu_quant_matmul(w, x, out, od, id, nt)
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
