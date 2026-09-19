//! Backend trait (Phase 2).
//!
//! Deviation from the plan's §3.5 sketch: `execute_node` takes `&mut self`
//! (the CPU backend mutates its own pool), and buffer ids are resolved inside
//! the backend's own pool. `read_host`/`write_host` give the allocator host
//! access (CPU: direct slices; GPU backends: staged copies at split
//! boundaries — Phase 3).

use super::ops::{FusedOp, Op};
use super::{BufRef, CNode, DType};

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

    /// Whether this backend bounds attention from the explicit `attn_span` input
    /// (E1) instead of deriving the causal window from `positions`.
    ///
    /// The default is `false` on purpose: a backend that has not been ported
    /// must never receive a multi-sequence attention node, and
    /// `GraphAllocator::supports` reads this instead of the backend's own
    /// source deciding (Metal is the unported one today — Phase G).
    fn supports_attn_span(&self) -> bool {
        false
    }

    /// Buffer pool: allocate / release a buffer of `size` f32 elements.
    fn alloc_buffer(&mut self, size: usize) -> usize;
    fn free_buffer(&mut self, id: usize);

    /// Allocate a buffer that bypasses the recycle free list. Split-boundary
    /// staging needs this: at execute time the free list holds ids whose
    /// physical contents are still referenced by node_to_buf and get
    /// read/written later in the same execute — recycling one would clobber
    /// in-flight data. Fresh buffers enter the normal free list on
    /// free_buffer (at graph rebuild), where liveness recycling is safe.
    fn alloc_fresh(&mut self, size: usize) -> usize;

    /// Execute one node: inputs and output are [`BufRef`]s into this backend's
    /// pool. A reference carries an element `offset` and `len` (D1 views), so a
    /// backend must apply the offset when it resolves the buffer — an owning
    /// node's reference has `offset == 0` and covers the whole buffer, and a
    /// view's is a window of its parent's.
    ///
    /// `kv_pair` is the layer's (k, v) region buffer *ids* for KV ops (None for
    /// non-KV ops or when the layer has no regions); the persistent KV regions
    /// are never views, so they stay ids.
    ///
    /// The output buffer may alias an input buffer (liveness reuse, in-place
    /// ops, D1 views) — the backend must handle that safely.
    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[BufRef],
        out_buf: BufRef,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String>;

    /// Host read/write of a pool buffer (for input filling and output
    /// extraction; GPU backends implement these as staged transfers).
    fn read_host(&self, id: usize) -> Option<&[f32]>;
    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String>;

    /// Wait for async work to complete (CPU: no-op; Metal: submit the pending
    /// command buffer). Called between splits and after the last split; only the
    /// Metal path invokes it today, so a CPU-only build never calls it.
    ///
    /// A backend that captured a CUDA Graph window for the current split must
    /// close it here (instantiate + launch the captured work once), because
    /// capture records launches without executing them.
    #[allow(dead_code)]
    fn synchronize(&mut self);

    /// Try to replay a previously captured graph for `(uid, range)` on this
    /// backend (Phase 7d, CUDA only). Returns `true` when the replay replaced
    /// the node loop — the scheduler then skips executing this split's nodes.
    ///
    /// Returning `false` may have armed or ENTERED capture mode for a future
    /// replay as a side effect (warmup bookkeeping internal to the backend);
    /// the window stays open until this split's `synchronize`. Implementations
    /// must keep captured pointers stable (pool ids never move memory) and
    /// re-capture when pool generation changed.
    ///
    /// Default: no capture support (CPU/Metal are no-ops).
    /// `nt_hint`: the graph's token count when it has matmul nodes
    /// (`capture_nt_hint()`), `None` otherwise. Backends that support graph
    /// capture (CUDA) use it to gate capture to decode-shaped graphs (8g①).
    ///
    /// CUDA-only surface: the sole caller (the scheduler's replay path) and
    /// the only override (CudaBackend) are both `#[cfg(feature = "cuda")]`,
    /// so the method itself is gated too — non-CUDA builds drop it (and its
    /// dead-code warning) entirely.
    #[cfg(feature = "cuda")]
    fn graph_replay(&mut self, _uid: u64, _range: (usize, usize), _nt_hint: Option<usize>) -> bool {
        false
    }
}
