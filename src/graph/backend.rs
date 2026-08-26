//! Backend trait (Phase 2).
//!
//! Deviation from the plan's §3.5 sketch: `execute_node` takes `&mut self`
//! (the CPU backend mutates its own pool), and buffer ids are resolved inside
//! the backend's own pool. `read_host`/`write_host` give the allocator host
//! access (CPU: direct slices; GPU backends: staged copies at split
//! boundaries — Phase 3).

use super::ops::{FusedOp, Op};
use super::{CNode, DType};

/// KV-region access: each layer owns two persistent regions (K and V).
/// Backends resolve the sibling buffer (e.g. the V region when executing
/// attention on the K view) from the `kv_pair` argument the scheduler passes
/// to `execute_node`.
pub trait KvProvider {
    /// (k_buf_id, v_buf_id) of a layer's persistent regions on this pool.
    fn kv_pair(&self, layer: usize) -> Option<(usize, usize)>;
}

pub trait Backend: Send + Sync {
    /// Human-readable backend name (diagnostics). Not called by the scheduler
    /// today, but part of the Backend API surface.
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// Op support by (op, dtype). `supports_fused` gates the fusion pass
    /// (Phase 4) so fused IR nodes are only produced when a kernel exists.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool;
    fn supports_fused(&self, fused: &FusedOp) -> bool;

    /// Buffer pool: allocate / release a buffer of `size` f32 elements.
    fn alloc_buffer(&mut self, size: usize) -> usize;
    fn free_buffer(&mut self, id: usize);

    /// Execute one node: inputs and output are ids in this backend's pool.
    /// `kv_pair` is the layer's (k, v) region buffer ids for KV ops
    /// (None for non-KV ops or when the layer has no regions). The output
    /// buffer may alias an input buffer (liveness reuse) — the backend must
    /// handle in-place execution safely.
    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[usize],
        out_buf: usize,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String>;

    /// Host read/write of a pool buffer (for input filling and output
    /// extraction; GPU backends implement these as staged transfers).
    fn read_host(&self, id: usize) -> Option<&[f32]>;
    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String>;

    /// Wait for async work to complete (CPU: no-op; Metal: submit the pending
    /// command buffer). Called between splits and after the last split; only the
    /// Metal path invokes it today, so a CPU-only build never calls it.
    #[allow(dead_code)]
    fn synchronize(&mut self);
}
