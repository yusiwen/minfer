//! `#[cfg(test)] mod tests` for `src/graph/alloc.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::batch::Batch;
use crate::graph::builder::GraphBuilder;
use crate::graph::DType;

mod backend_fence;
mod budget;
mod kv_arena;
mod liveness;
mod staging;
mod views;
fn chain(n_ops: usize) -> ComputeGraph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
    let mut h = x;
    for i in 0..n_ops {
        h = if i % 2 == 0 { b.silu(h) } else { b.add(h, x) };
    }
    b.output(h);
    b.build()
}
// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `alloc.rs` (bucket B of the dead-code census — every
// test caller already lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl GraphAllocator {
    /// Set (or clear, with `None`) a backend's memory budget. `None` restores the
    /// backend's own default.
    ///
    /// Test-only (#239): driven by
    /// `graph::alloc::tests::{the_budget_counts_weights_and_activations_together,
    /// a_graph_that_cannot_fit_is_refused_with_its_numbers}`.
    ///
    /// Note: the production doc claimed "a future offload policy" would use it; the E5 S2
    /// offload policy landed and resolves its budget through `MINFER_GPU_MEM` +
    /// `allocplan::weight_budget` instead, so that clause was stale. Recorded for
    /// [#244](https://github.com/yusiwen/minfer/issues/244); the callers are tests.
    pub fn set_memory_budget(&mut self, backend: Backend, budget: Option<usize>) {
        match budget {
            Some(b) => {
                self.budget.insert(backend, b);
            }
            None => {
                self.budget.remove(&backend);
            }
        }
    }

    /// Number of distinct buffers currently allocated (for tests).
    ///
    /// Test-only (#239): driven by `graph::alloc::tests::{slots,
    /// a_rebuild_inside_one_class_reuses_the_pool,
    /// a_graph_that_cannot_fit_is_refused_with_its_numbers}`.
    pub fn n_cpu_buffers(&self) -> usize {
        self.cpu.pool_len()
    }

    /// Number of distinct buffers actually mapped to nodes (for tests).
    ///
    /// Test-only (#239): driven by `graph::alloc::tests::{kv_regions_two_per_layer,
    /// liveness_reuses_buffers_along_chain, parallel_chains_do_not_share}`.
    pub fn n_mapped_buffers(&self) -> usize {
        let mut s: std::collections::BTreeSet<(Backend, usize)> = std::collections::BTreeSet::new();
        for br in self.node_to_buf.values() {
            s.insert((br.backend, br.id));
        }
        s.len()
    }
}
