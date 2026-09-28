//! The q4_K `W_dsc` plane's pure admission contract (issue #165).
//!
//! The CUDA loader precomputes a `W_dsc` f32-pair plane for some q4_K weights and
//! uploads it; `mmq_raw_nb_bt` is its only consumer. The rule that decides which
//! weights that is — q4_K **and** a payload that really is the q4_K block layout —
//! lives here rather than in `cuda.rs` for one concrete reason: it is pure, so it is
//! compiled and **run** without the `cuda` feature. CI's CPU job executes these tests
//! (`cargo test --release`); the CUDA job only compile-checks the device modules (a
//! hosted runner has no GPU), so a predicate that lived behind `#[cfg(feature =
//! "cuda")]` would never be *run* in CI at all.
//!
//! The device half — the registry really gains no `__q4dsc` entry and the q4_K case
//! still does — is `graph::cuda_backend::tests::cuda_q4dsc_plane_is_q4k_only`, and the
//! end-to-end half (a real non-q4_K model registers no plane) is the `#[ignore]`d
//! `cuda_real_model_registers_q4dsc_planes_only_for_q4k`.
#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use crate::tensor::TensorType;

/// The exact host byte length one `od x id` q4_K payload must have: `od * (id / 256) *
/// 144`, i.e. one 144-byte q4_K super-block per 256 input elements per row. This is
/// q4_K's bytes-per-element ratio (144/256 = 0.5625) applied to a whole number of
/// super-blocks per row, which is also `CudaState::expand_q4k_dsc`'s own row
/// arithmetic. `None` when the geometry is empty or `id` is not a whole number of
/// super-blocks.
///
/// **Exact equality is the right test, not `>=`.** A `>=` check admits a q8_0 row
/// (1088 B where q4_K needs 576) and the expander then misreads it into the plane —
/// the defect #165 fixes. A type with a ratio *smaller* than q4_K's (a 2-bit K-quant:
/// 84 B per 256) is shorter than this, so it is refused instead of being read past the
/// tensor. A q4_0 payload has exactly q4_K's ratio, so this check cannot tell those two
/// apart — the type gate in [`q4k_dsc_plane_admitted`] is what refuses it.
pub fn q4k_dsc_payload_bytes(od: usize, id: usize) -> Option<usize> {
    if od == 0 || id == 0 || id % 256 != 0 {
        return None;
    }
    Some(od * (id / 256) * crate::block::Q4KB)
}

/// Whether `payload_bytes` is exactly the q4_K payload length for an `od x id` tensor.
/// See [`q4k_dsc_payload_bytes`] for why this is equality and not a lower bound.
pub fn q4k_dsc_payload_ok(payload_bytes: usize, od: usize, id: usize) -> bool {
    q4k_dsc_payload_bytes(od, id) == Some(payload_bytes)
}

/// The one admission rule for the q4_K `W_dsc` f32-pair plane. Both halves are
/// load-bearing:
///
/// * the **type** gate — `TensorType::Q4_K` is the only type the NB-BT kernel
///   (`mmq_raw_nb_bt`, which is also the only consumer of the `q4k_dsc` map)
///   dispatches the dsc template for. Any other admitted type's plane would be built
///   for bytes no kernel ever reads, at device-memory and host-CPU cost per tensor. A
///   q4_0 weight has q4_K's bytes-per-element ratio, so only this gate can refuse it;
/// * the **payload** gate — the bytes must be exactly [`q4k_dsc_payload_bytes`], the
///   defence that survives a future type with a *smaller* ratio joining the loader's
///   `matches!`: it is refused here instead of making `CudaState::expand_q4k_dsc` read
///   past the tensor.
///
/// The qwen2 loader calls this before `CudaState::register_weight_q4k_dsc`, and
/// `register_weight_q4k_dsc` re-checks the payload half itself, so a direct caller
/// cannot bypass it either.
pub fn q4k_dsc_plane_admitted(
    ttype: TensorType,
    payload_bytes: usize,
    od: usize,
    id: usize,
) -> bool {
    ttype == TensorType::Q4_K && q4k_dsc_payload_ok(payload_bytes, od, id)
}

#[cfg(test)]
mod tests;
