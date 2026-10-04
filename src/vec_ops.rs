// SIMD Vector Operations + Core Ops

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

mod bf16;
mod f16;
#[cfg(target_arch = "aarch64")]
mod neon;
mod rms_norm;
mod rope;
mod silu;
mod softmax;
mod vec;

pub use bf16::mat_mul_bf16;
pub use f16::mat_mul_f16;
// `dot_f16_f32` / `dot_f16_f32_scalar` / `f16_dot_path` / `F16DotPath` have
// exactly one consumer, `vec_ops::tests` (the f16 gates), so their re-export is
// test-only: a non-test `pub use` of an unused name is what
// `#![deny(warnings)]` rejects.
#[cfg(test)]
pub use f16::{dot_f16_f32, dot_f16_f32_scalar, f16_dot_path, F16DotPath};
pub use rms_norm::{rms_norm_f32, rms_norm_fused_f32};
pub use rope::RopeStyle;
pub use silu::{vec_silu_f32, vec_swiglu_f32};
pub use softmax::{vec_soft_max_f32, vec_soft_max_inplace_f32};
pub use vec::{
    mat_mul_f32, vec_add_f32, vec_cpy_f32, vec_dot_f32, vec_mul_f32, vec_muladd_f32, vec_scale_f32,
};

// Internal wiring: the submodule bodies reach these names through `use super::*`,
// exactly as they did inside the one file before the split.
#[cfg(target_arch = "x86_64")]
use vec::vec_exp_f32_avx2;

#[cfg(target_arch = "aarch64")]
use neon::enabled as neon_vec_enabled;
#[cfg(target_arch = "aarch64")]
use neon::{
    vec_add_f32 as vec_add_f32_neon, vec_dot_f32 as vec_dot_f32_neon,
    vec_muladd_f32 as vec_muladd_f32_neon, vec_scale_f32 as vec_scale_f32_neon,
    vec_soft_max_f32 as vec_soft_max_f32_neon,
};

#[cfg(test)]
use bf16::decode_bf16_row;
#[cfg(test)]
use f16::{decode_f16_row, F16_SIMD_PATH_CALLS};

#[cfg(test)]
mod tests;
