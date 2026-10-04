// Compute kernel dispatch layer.
//   CPU (AVX2/NEON/scalar) is always available as fallback.
//   MPS (Apple Silicon GPU) is enabled at runtime when Metal is available.

use crate::block::{Q41B, Q4B, Q4KB, Q6KB, Q8B, Q8KB};
use crate::tensor::{Tensor, TensorType};

mod dispatch;
mod embed;
mod pool;

// `cpu_quant_matmul` (the byte-in/byte-out worker entry) stays `pub` in
// `dispatch`: its one caller is `cpu_quant_matmul_f32` in the same file and no
// `kernel::cpu_quant_matmul` path is named anywhere in the crate, so a
// re-export would be an unused import under `#![deny(warnings)]`.
pub use dispatch::cpu_quant_matmul_f32;
pub use embed::embed_tokens;
pub(crate) use pool::MIN_PARALLEL_MACS;
pub use pool::{cpu_threads, par_for, set_cpu_threads};

#[cfg(test)]
mod tests;
