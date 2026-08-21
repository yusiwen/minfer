//! Backend trait (Phase 2).
//!
//! Deviation from the plan's §3.5 sketch: `execute_node` takes `&mut self`
//! (the CPU backend mutates its own pool), and buffer ids are resolved inside
//! the backend's own pool. `read_host`/`write_host` give the allocator host
//! access (CPU: direct slices; GPU backends: staged copies at split
//! boundaries — Phase 3).

use super::ops::{FusedOp, Op};
use super::{CNode, DType};

pub trait Backend: Send + Sync {
    fn name(&self) -> &str;

    /// Op support by (op, dtype). `supports_fused` gates the fusion pass
    /// (Phase 4) so fused IR nodes are only produced when a kernel exists.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool;
    fn supports_fused(&self, fused: &FusedOp) -> bool;

    /// Buffer pool: allocate / release a buffer of `size` f32 elements.
    fn alloc_buffer(&mut self, size: usize) -> usize;
    fn free_buffer(&mut self, id: usize);

    /// Execute one node: inputs and output are ids in this backend's pool.
    /// The output buffer may alias an input buffer (liveness reuse) — the
    /// backend must handle in-place execution safely.
    fn execute_node(&mut self, node: &CNode, in_bufs: &[usize], out_buf: usize) -> Result<(), String>;

    /// Host read/write of a pool buffer (for input filling and output
    /// extraction; GPU backends implement these as staged transfers).
    fn read_host(&self, id: usize) -> Option<&[f32]>;
    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String>;

    /// Wait for async work to complete (no-op for CPU).
    fn synchronize(&self);
}
