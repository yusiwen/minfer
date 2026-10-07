// Quantized dot-product kernels + activation quantization — all &[u8]
// interface. Fast paths: AVX2+FMA on x86_64, NEON+SDOT on aarch64, with
// scalar fallbacks. Activation formats: Q8_0 (simple weight types) and Q8_K
// (K-quant weights — 256-element blocks with precomputed bsums).
use crate::block::{self, Q41B, Q4B, Q4KB, Q6KB, Q8B};

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

mod avx2;
#[cfg(target_arch = "x86_64")]
mod avx512;
mod dot_q4_0;
mod dot_q4_1;
mod dot_q5;
mod dot_q8_0;
mod kquant;
mod quantize_q8_0;
mod quantize_q8_k;

#[cfg(target_arch = "aarch64")]
mod neon;

pub use dot_q4_0::dot_q4_0_q8_0;
pub use dot_q4_1::dot_q4_1_q8_0;
pub use dot_q5::{dot_q5_0_q8_0, dot_q5_1_q8_0};
pub use dot_q8_0::dot_q8_0_q8_0;
pub use kquant::{dot_q4_k_q8_k, dot_q5_k_q8_k, dot_q6_k_q8_k};
#[cfg(test)]
pub use quantize_q8_0::quantize_row_q8_0;
pub(crate) use quantize_q8_0::quantize_row_q8_0_into;
pub use quantize_q8_0::{dequantize_row_q8_0, quantize_row_q8_0_buf};
pub use quantize_q8_k::quantize_row_q8_k_buf;

// Internal wiring: the submodule bodies reach these names through `use super::*`,
// exactly as the inline modules did before the split. A name that only a child
// uses is still a use of the import, so the non-test build stays warning-free.
use avx2::quantize_scalar;
#[cfg(target_arch = "x86_64")]
use avx2::{hsum_float_8, quantize_avx2};
#[cfg(target_arch = "x86_64")]
use quantize_q8_0::f16_to_f32_bits;

#[cfg(target_arch = "aarch64")]
use neon::{
    dot_q4_0_q8_0 as dot_q4_0_q8_0_neon, dot_q4_1_q8_0 as dot_q4_1_q8_0_neon,
    dot_q5_0_q8_0 as dot_q5_0_q8_0_neon, dot_q5_1_q8_0 as dot_q5_1_q8_0_neon,
    dot_q8_0_q8_0 as dot_q8_0_q8_0_neon, enabled as neon_enabled,
};
#[cfg(target_arch = "aarch64")]
use neon::{
    dot_q4_k_q8_k as dot_q4_k_q8_k_neon, dot_q5_k_q8_k as dot_q5_k_q8_k_neon,
    dot_q6_k_q8_k as dot_q6_k_q8_k_neon,
};

#[cfg(target_arch = "x86_64")]
use avx2::{
    dot_q4_k_q8_k as dot_q4_k_q8_k_avx2, dot_q5_k_q8_k as dot_q5_k_q8_k_avx2,
    dot_q6_k_q8_k as dot_q6_k_q8_k_avx2, enabled as avx2_enabled,
};
#[cfg(target_arch = "x86_64")]
use avx512::{
    dot_q4_k_q8_k as dot_q4_k_q8_k_avx512, dot_q5_k_q8_k as dot_q5_k_q8_k_avx512,
    dot_q6_k_q8_k as dot_q6_k_q8_k_avx512, enabled as avx512_enabled,
};

#[cfg(all(test, target_arch = "x86_64"))]
use kquant::{dot_q4_k_q8_k_scalar, dot_q5_k_q8_k_scalar, dot_q6_k_q8_k_scalar};
#[cfg(all(test, target_arch = "aarch64"))]
use kquant::{dot_q4_k_q8_k_scalar, dot_q6_k_q8_k_scalar};

#[cfg(test)]
mod tests;

#[cfg(all(test, target_arch = "x86_64"))]
mod avx2_correctness;
#[cfg(all(test, target_arch = "aarch64"))]
mod neon_correctness;
