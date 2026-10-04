//! `#[cfg(test)] mod tests` for `src/graph/cuda_backend.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::alloc::GraphAllocator;
use crate::graph::backend::{Backend as _, KvProvider};
use crate::graph::builder::GraphBuilder;
use crate::graph::cache::GraphCache;
use crate::graph::scheduler::BackendScheduler;
use crate::graph::DType;
// ─── Phase 7b: per-op dispatch parity ───────────────────────

use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
use crate::tensor::{Tensor, TensorType};

mod attention;
mod attn_window;
mod capture;
mod elementwise;
mod kv;
mod matmul;
mod mmvq;
mod pool;
mod prefill;
mod staging;
mod weights;
/// Init the CUDA singleton; silent-skip the test when no device answers
/// (e.g. CI without a GPU). Run with --nocapture to see skips.
fn device() -> Option<&'static crate::cuda::CudaState> {
    crate::cuda::CudaState::init();
    crate::cuda::CudaState::get()
}
/// Fresh backend on an initialized device (None → skip on no-GPU hosts).
fn pool() -> Option<CudaBackend> {
    device()?;
    CudaBackend::new()
}
fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
    assert_eq!(got.len(), want.len(), "{name}: length mismatch");
    let mut worst = (0.0f32, 0usize);
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let d = (g - w).abs();
        if d > worst.0 {
            worst = (d, i);
        }
    }
    assert!(
        worst.0 <= tol,
        "{name}: max diff {} at {} (got {}, want {})",
        worst.0,
        worst.1,
        got[worst.1],
        want[worst.1]
    );
}
// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `cuda_backend.rs` (bucket B of the dead-code census —
// every test caller already lives in this module's subtree, including the
// `#[cfg(test)]` shim through which `elems` was reached).
// ────────────────────────────────────────────────────────────────────────────

impl CudaBackend {
    /// `None` when CUDA is unavailable (no device, or disabled via
    /// `MINFER_DISABLE_CUDA` — both handled by `CudaState::try_new`).
    ///
    /// The layout defaults to **F32**; a real engine's backend is built by
    /// `GraphAllocator::enable_cuda`, which passes the allocator's stamped format
    /// (see [`Self::with_layout`]).
    ///
    /// Test-only (#239): driven by `cuda_backend::tests::pool::cuda_pool_roundtrip` and
    /// 35 further device gates in this file. `pub(crate)` because `graph::op_matrix`
    /// (a `#[cfg(test)] mod`) builds an f32-layout backend through it — an inherent
    /// impl may live in a child module, so the cross-module caller keeps compiling
    /// while production loses the constructor entirely.
    pub(crate) fn new() -> Option<Self> {
        Self::with_layout(crate::cuda::KV_LAYOUT_F32)
    }

    /// Device tests address pool buffers by id (they always did), so convert to
    /// owning `BufRef`s here — `elems(id)` gives the real length, which the D1
    /// window arithmetic relies on. Views are exercised through the op matrix
    /// (which goes through the allocator), not here.
    ///
    /// Test-only (#239): driven by the device parity gates in this file
    /// (`cuda_elementwise_parity`, `cuda_norm_parity`, `cuda_matmul_parity`, …).
    pub fn exec_ids(
        &mut self,
        node: &CNode,
        in_ids: &[usize],
        out_id: usize,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        let ins: Vec<BufRef> = in_ids
            .iter()
            .map(|&id| BufRef::own(crate::graph::Backend::CUDA, id, self.elems(id)))
            .collect();
        let out = BufRef::own(crate::graph::Backend::CUDA, out_id, self.elems(out_id));
        self.execute_node(node, &ins, out, kv_pair)
    }

    /// Bytes of one pool buffer, as f32 elements.
    ///
    /// Test-only (#239): reached only through `exec_ids` above.
    fn elems(&self, id: usize) -> usize {
        self.pool[id].bytes / 4
    }
}
