//! CUDA graph backend (Phase 7).
//!
//! Wraps the [`crate::cuda::CudaState`] singleton in the graph [`Backend`]
//! trait contract (`src/graph/backend.rs`), mirroring `metal_backend.rs` where
//! the mechanics allow: a device buffer pool with a byte-length free list,
//! name → device-pointer weight resolution, sync H2D/D2H host transfers, and
//! per-op kernel dispatch on the shared stream. Design + rollout:
//! `docs/CUDA-BACKEND-PLAN.md`.

use super::backend::Backend;
use super::ops::{FusedOp, NodeMeta, Op};
use super::{CNode, DType};
use crate::vec_ops::RopeStyle;

struct CudaBuf {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
}

pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,
    /// 8b: KV cache element type — f16 (bandwidth-halving, Metal-aligned
    /// auto-select policy) or f32. Set at construction from the process-wide
    /// flag; a `#[cfg(test)]` setter flips it per instance so device tests
    /// can exercise both layouts in one process.
    kv_f16: bool,
    pool: Vec<CudaBuf>,
    free: Vec<usize>,
    /// Bumped on every pool allocation (fresh or free-list reuse). A captured
    /// CUDA Graph (7d) is only valid while the node → device-pointer mapping
    /// is unchanged, and any alloc_buffer() call may change it.
    pool_gen: u64,
    /// Device scratch holding raw-int32 positions decoded from the f32-bits
    /// I32 input buffers (grown on demand; freed in Drop alongside the pool).
    pos_scratch: *mut std::ffi::c_void,
    pos_scratch_bytes: usize,
    /// Captured CUDA Graphs (Phase 7d), keyed by (graph uid, split node
    /// range) and valid only for the pool_gen captured at. Few entries: one
    /// per executed split of each reused graph (decode captures; a one-shot
    /// prefill never passes warmup).
    graph_execs: Vec<CapturedGraph>,
    /// Direct-launch warmup counter per (uid, range): the 3rd consecutive
    /// execution enters capture (llama.cpp warms up twice).
    graph_runs: std::collections::HashMap<(u64, (usize, usize)), u32>,
    /// Open capture window (armed by `graph_replay`, closed by `synchronize`).
    capturing: Option<(u64, (usize, usize))>,
    /// Held process-wide stream lock while `capturing` is open (released when
    /// the window closes; see `CudaState::stream_lock`).
    stream_guard: Option<std::sync::MutexGuard<'static, ()>>,
    /// `MINFER_NO_CUDA_GRAPH=1` (at construction) or a capture failure
    /// (session-wide) force the plain direct-launch path.
    graphs_mode: GraphMode,
    /// 8g②: deliberate prefill-capture opt-in (default off). Repeated
    /// identical-nt prefills (server/slot scenario) may then capture after
    /// the usual 3-run protocol; MINFER_CAPTURE_PREFILL=1 enables it.
    prefill_capture: bool,
}

/// An instantiated CUDA Graph exec with its capture identity.
struct CapturedGraph {
    exec: *mut std::ffi::c_void,
    uid: u64,
    range: (usize, usize),
    pool_gen: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GraphMode {
    /// Capture/replay allowed.
    Enabled,
    /// Forced off (`MINFER_NO_CUDA_GRAPH=1`) or disabled after a failure.
    Disabled,
}

// SAFETY: CudaBuf holds raw device pointers that are only dereferenced by the
// GPU; the backend is only mutated through &mut self (allocator/scheduler).
// Same reasoning as metal_backend.rs's unsafe Send/Sync.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// `None` when CUDA is unavailable (no device, or disabled via
    /// `MINFER_DISABLE_CUDA` — both handled by `CudaState::try_new`).
    /// (Called by GraphAllocator::enable_cuda; bin builds see it as dead until
    /// the Phase 7c model wiring, tests use it meanwhile.)
    #[allow(dead_code)]
    pub fn new() -> Option<Self> {
        let state = crate::cuda::CudaState::get()?;
        let kv_f16 = crate::cuda::kv_cache_is_f16();
        let graphs_mode = if std::env::var("MINFER_NO_CUDA_GRAPH").as_deref() == Ok("1") {
            GraphMode::Disabled
        } else {
            GraphMode::Enabled
        };
        let prefill_capture = std::env::var("MINFER_CAPTURE_PREFILL").as_deref() == Ok("1");
        Some(Self {
            state,
            pool: Vec::new(),
            free: Vec::new(),
            pool_gen: 0,
            pos_scratch: std::ptr::null_mut(),
            pos_scratch_bytes: 0,
            graph_execs: Vec::new(),
            graph_runs: std::collections::HashMap::new(),
            capturing: None,
            stream_guard: None,
            graphs_mode,
            prefill_capture,
            kv_f16,
        })
    }

    /// Pool generation counter (CUDA Graph replay invalidation, Phase 7d).
    #[allow(dead_code)]
    pub fn pool_gen(&self) -> u64 {
        self.pool_gen
    }

    /// Number of captured graphs currently held (test introspection).
    #[cfg(test)]
    fn captured_count(&self) -> usize {
        self.graph_execs.len()
    }

    #[cfg(test)]
    /// 8b: flip the per-instance KV element type (device tests exercise both
    /// layouts in one process; production backends take the global policy
    /// set by the loader at construction).
    pub(crate) fn set_kv_f16_for_test(&mut self, f16: bool) {
        self.kv_f16 = f16;
    }

    /// 8g②: turn on deliberate prefill capture for tests (the pp16/pp300
    /// bit-parity harness is the validation gate).
    pub(crate) fn set_prefill_capture_for_test(&mut self, enabled: bool) {
        self.prefill_capture = enabled;
    }

    pub(crate) fn set_graphs_enabled_for_test(&mut self, enabled: bool) {
        self.graphs_mode = if enabled {
            GraphMode::Enabled
        } else {
            GraphMode::Disabled
        };
        self.graph_execs.clear();
        self.graph_runs.clear();
    }

    /// Phase 7d (CUDA Graph capture/replay), llama.cpp's state machine:
    ///
    /// - executions 1 and 2 of a `(uid, range)` split run direct launches
    ///   (warmup — one-shot graphs like prefill never reach capture);
    /// - the 3rd execution opens a stream-capture window around the node loop
    ///   (this call returns `false`; the window is closed by `synchronize`,
    ///   which instantiates, launches once so the step still produces output,
    ///   and caches the exec);
    /// - subsequent executions launch the captured graph and return `true`
    ///   (the caller skips the node loop). Input staging buffers were
    ///   H2D-filled before the split at stable addresses, so replay reads
    ///   fresh data — the invariant llama.cpp relies on.
    ///
    /// A pool_gen change invalidates the stored exec (pointers may differ).
    /// `MINFER_NO_CUDA_GRAPH=1` or any failure disables graphs for the
    /// backend's lifetime and everything falls back to direct launches.
    fn graph_replay_step(
        &mut self,
        uid: u64,
        range: (usize, usize),
        nt_hint: Option<usize>,
    ) -> bool {
        if self.graphs_mode != GraphMode::Enabled {
            return false;
        }
        // a replay launch into OUR OWN open capture window would be
        // CUDA-invalid — defer to direct execution (Phase 8 review;
        // unreachable today: 7e③ made graphs single-split)
        if self.capturing.is_some() {
            return false;
        }
        let key = (uid, range);
        if let Some(pos) = self
            .graph_execs
            .iter()
            .position(|g| g.uid == uid && g.range == range)
        {
            if self.graph_execs[pos].pool_gen != self.pool_gen {
                // pool churned since capture — pointers may differ, re-capture
                let g = self.graph_execs.remove(pos);
                self.graph_runs.remove(&key);
                self.state.graph_destroy(g.exec);
            } else {
                let exec = self.graph_execs[pos].exec;
                // a plain stream launch — serialized like any other stream op
                let _sg = self.stream_guard();
                if self.state.graph_launch_exec(exec) {
                    return true;
                }
                eprintln!("CUDA: graph replay launch failed; graphs disabled for this session");
                self.graphs_mode = GraphMode::Disabled;
                return false;
            }
        }
        let runs = self.graph_runs.entry(key).or_insert(0);
        *runs += 1;
        // 8g①: capture decode-shaped graphs only (nt_hint None = no matmul
        // in the graph, synthetic tests). A repeated identical-nt prefill
        // (server/slot scenario) must not silently start capturing a
        // ~437-node graph that was never validated for capture. 8g② turns
        // prefill capture into a DELIBERATE opt-in (test setter or
        // MINFER_CAPTURE_PREFILL=1) — gated by the pp16/pp300 bit-parity
        // harness.
        if *runs >= 3
            && self.capturing.is_none()
            && nt_hint.map_or(true, |nt| nt == 1 || self.prefill_capture)
        {
            // Hold the process-wide stream lock across the capture window:
            // any other backend's stream work would otherwise be recorded
            // into this graph (capture is per-stream, not per-thread).
            let guard = self.state.stream_lock().lock().unwrap();
            if self.state.graph_begin_capture() {
                self.capturing = Some(key);
                self.stream_guard = Some(guard);
            } else {
                drop(guard);
                eprintln!("CUDA: stream capture unavailable; graphs disabled for this session");
                self.graphs_mode = GraphMode::Disabled;
            }
        }
        false
    }

    /// Stream-work serialization for backend methods: `None` while THIS
    /// backend holds an open capture window (its own enqueues are the
    /// recorded work); otherwise a held process-wide lock that blocks while
    /// any other backend is capturing.
    fn stream_guard(&self) -> Option<std::sync::MutexGuard<'static, ()>> {
        if self.capturing.is_some() {
            None
        } else {
            Some(self.state.stream_lock().lock().unwrap())
        }
    }

    /// Close an open capture window (instantiate + launch once + cache), or
    /// fall back to a plain synchronize. Called at split boundaries and after
    /// the last split — never inside a capture window.
    fn close_capture_or_sync(&mut self) {
        if let Some(key) = self.capturing.take() {
            // the stream lock stays held (self.stream_guard) until the window
            // is fully closed and the capture launch has been enqueued
            let exec = self.state.graph_end_capture_to_exec();
            let ok = !exec.is_null() && self.state.graph_launch_exec(exec);
            self.stream_guard = None; // release after the last stream op
            if ok {
                self.graph_execs.push(CapturedGraph {
                    exec,
                    uid: key.0,
                    range: key.1,
                    pool_gen: self.pool_gen,
                });
                self.state.sync();
            } else {
                self.state.graph_destroy(exec);
                // The recorded launches never executed — this step's
                // outputs are undefined. End-of-capture failure is
                // effectively unreachable for our graphs (no host syncs,
                // no readbacks, async D2D only inside the window), so log
                // loudly, disable, and surface the broken state to the
                // next caller via a poisoned error on the next replay.
                eprintln!(
                    "CUDA: graph capture end/instantiate/launch failed; \
                     graphs disabled for this session (this step's split did not execute — \
                     rerun with MINFER_NO_CUDA_GRAPH=1)"
                );
                self.graphs_mode = GraphMode::Disabled;
                self.state.sync();
                // NOTE: there is no poisoned-error mechanism — later steps
                // run direct-launch with graphs disabled; this step's outputs
                // were undefined and are consumed as-is. (Phase 8 review:
                // the old comment claimed otherwise.)
            }
            return;
        }
        self.state.sync();
    }

    fn ptr_of(&self, id: usize) -> Result<*mut std::ffi::c_void, String> {
        match self.pool.get(id) {
            Some(b) if !b.ptr.is_null() => Ok(b.ptr),
            Some(_) => Err(format!("cuda: buffer {id} has a null device pointer")),
            None => Err(format!("cuda: unknown buffer id {id}")),
        }
    }

    /// Sync D2H readback of a pool buffer as owned data. This is the alloc.rs
    /// `copy_to_cpu` CUDA arm (the trait's `read_host` cannot return a borrowed
    /// slice for a staged transfer). Explicit `sync()` first: never rely on the
    /// legacy-default-stream's implicit synchronization with blocking streams.
    pub fn copy_to_host(&self, id: usize) -> Option<Vec<f32>> {
        let b = self.pool.get(id)?;
        if b.ptr.is_null() || b.bytes == 0 {
            return None;
        }
        let _sg = self.stream_guard();
        let mut out = vec![0f32; b.bytes / 4];
        self.state.sync();
        let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, b.bytes) };
        // R3-A2: read through the pinned staging buffer (pageable-memcpy
        // bounce removed); MINFER_NO_PINNED_READBACK=1 reverts.
        self.state.copy_from_device_pinned(b.ptr, dst);
        Some(out)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // cudaFree implicitly syncs — serialize against open capture windows.
        let _sg = self.stream_guard();
        // Pools live as long as the backend (inside GraphCache); only real
        // teardown frees device memory. free_buffer() only recycles.
        for b in &self.pool {
            Self::state_free(b.ptr);
        }
        self.pool.clear();
        self.free.clear();
        if !self.pos_scratch.is_null() {
            Self::state_free(self.pos_scratch);
            self.pos_scratch = std::ptr::null_mut();
            self.pos_scratch_bytes = 0;
        }
        for g in &self.graph_execs {
            self.state.graph_destroy(g.exec);
        }
        self.graph_execs.clear();
        self.capturing = None;
    }
}

impl CudaBackend {
    /// End an open capture window WITHOUT launching it (error path, Phase 8
    /// review): the recorded launches never executed, so the split's outputs
    /// are invalid. Disables graph capture for the session.
    fn abort_capture(&mut self, cause: &str) {
        if let Some(key) = self.capturing.take() {
            let exec = self.state.graph_end_capture_to_exec();
            if !exec.is_null() {
                self.state.graph_destroy(exec);
            }
            self.stream_guard = None;
            self.graphs_mode = GraphMode::Disabled;
            eprintln!(
                "CUDA: node error inside capture window (split {key:?}); capture aborted, \
                 graphs disabled for this session: {cause}"
            );
            self.state.sync();
        }
    }

    fn execute_node_inner(
        &mut self,
        node: &CNode,
        in_bufs: &[usize],
        out_buf: usize,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        // one lock acquisition per node; None while this backend itself is
        // capturing (its own enqueues are the recorded work)
        let _sg = self.stream_guard();
        match &node.op {
            // Inputs are host-filled by the allocator; KvcacheLoad is a view
            // of the persistent K region (out_buf IS the region — no kernel).
            Op::Input | Op::KvcacheLoad { .. } => Ok(()),
            // Layout-only nodes: identity copy of the source buffer (same
            // semantics as cpu_backend's View/Reshape/Permute handling).
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                let src = *in_bufs
                    .first()
                    .ok_or_else(|| format!("cuda: {} without source buffer", node.name))?;
                self.copy_d2d(src, out_buf)
            }
            // 7e③: row gather. Embed meta = weight gather + dequantize (the
            // embedding, type dispatched on device); no meta = generic f32
            // gather (the G3 tail reduction). ids are I32-as-f32 bits.
            Op::GetRows => match &node.meta {
                NodeMeta::Embed(m) => {
                    let wptr = self.state.get_weight_ptr(&m.weight_name).ok_or_else(|| {
                        format!(
                            "cuda: {} weight '{}' not registered",
                            node.name, m.weight_name
                        )
                    })?;
                    let n_embd = node.out_shape[0];
                    let nt = node.out_shape[1];
                    self.state.embed_rows_on_gpu(
                        m.weight_ttype,
                        wptr,
                        self.ptr_of(in_bufs[0])?,
                        self.ptr_of(out_buf)?,
                        n_embd,
                        nt,
                        self.state.is_weight_padded(&m.weight_name),
                    )?;
                    Ok(())
                }
                NodeMeta::None => {
                    let n_embd = node.out_shape[0];
                    let nt = node.out_shape[1];
                    self.state.gather_rows_f32_on_gpu(
                        self.ptr_of(in_bufs[0])?,
                        self.ptr_of(in_bufs[1])?,
                        self.ptr_of(out_buf)?,
                        n_embd,
                        nt,
                    );
                    Ok(())
                }
                other => Err(format!("get_rows node with unexpected meta: {other:?}")),
            },

            Op::Add => {
                let n = self.elems(out_buf);
                if self.elems(in_bufs[0]) != n || self.elems(in_bufs[1]) != n {
                    return Err(format!("cuda: {}: add input size mismatch", node.name));
                }
                self.state.add_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(in_bufs[1])?,
                    self.ptr_of(out_buf)?,
                    n,
                );
                Ok(())
            }
            Op::Mul => {
                let n = self.elems(out_buf);
                if self.elems(in_bufs[0]) != n || self.elems(in_bufs[1]) != n {
                    return Err(format!("cuda: {}: mul input size mismatch", node.name));
                }
                self.state.mul_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(in_bufs[1])?,
                    self.ptr_of(out_buf)?,
                    n,
                );
                Ok(())
            }
            // In-place op (alias rule, graph rules §5): stage via D2D copy when
            // the allocator did not alias the input, then run on the output.
            Op::Silu => {
                if in_bufs[0] != out_buf {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                self.state
                    .silu_f32(self.ptr_of(out_buf)?, self.elems(out_buf));
                Ok(())
            }
            Op::SwiGLU => {
                let n = self.elems(out_buf);
                if self.elems(in_bufs[0]) != n || self.elems(in_bufs[1]) != n {
                    return Err(format!("cuda: {}: swiglu input size mismatch", node.name));
                }
                self.state.swiglu_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(in_bufs[1])?,
                    self.ptr_of(out_buf)?,
                    n,
                );
                Ok(())
            }

            Op::RmsNorm { eps } => {
                let wptr = self.norm_weight(node)?;
                let d = node.out_shape[0];
                if d % 4 != 0 || d == 0 || self.elems(out_buf) % d != 0 {
                    return Err(format!(
                        "cuda: {}: rms_norm dim {d} must be a nonzero multiple of 4 (float4 kernel)",
                        node.name
                    ));
                }
                let n = self.elems(out_buf) / d;
                self.state.rms_norm(
                    self.ptr_of(in_bufs[0])?,
                    Some(wptr),
                    self.ptr_of(out_buf)?,
                    d,
                    n,
                    *eps,
                );
                Ok(())
            }
            // Per-head RMSNorm: the flat [nt*nh*hd] buffer is a contiguous
            // [nt*nh, hd] row matrix (t*(nh*hd) + h*hd == (t*nh+h)*hd), so the
            // same rms_norm kernel runs with d = hd (weight shared per head).
            Op::QkNorm { hd, eps, .. } => {
                let wptr = self.norm_weight(node)?;
                let d = *hd;
                if d % 4 != 0 || d == 0 || self.elems(out_buf) % d != 0 {
                    return Err(format!(
                        "cuda: {}: qk_norm head dim {d} must be a nonzero multiple of 4 (float4 kernel)",
                        node.name
                    ));
                }
                let n = self.elems(out_buf) / d;
                self.state.rms_norm(
                    self.ptr_of(in_bufs[0])?,
                    Some(wptr),
                    self.ptr_of(out_buf)?,
                    d,
                    n,
                    *eps,
                );
                Ok(())
            }

            // 7e⑤: decode FFN gate+up fusion (decode nt==1 only): one concat
            // matmul (ffn_gate|ffn_up rows → gate|up in the output buffer),
            // then an in-place offset swiglu folding silu(gate)*up into the
            // gate rows. The following down matmul reads rows 0..nf.
            Op::FusedFFN => {
                let meta = match &node.meta {
                    NodeMeta::FusedFfn(m) => m,
                    other => {
                        return Err(format!("fused_ffn node missing FusedFfnMeta: {other:?}"));
                    }
                };
                let wptr = self.state.get_weight_ptr(&meta.gu_weight).ok_or_else(|| {
                    format!(
                        "cuda: gu weight '{}' not registered ({})",
                        meta.gu_weight, node.name
                    )
                })?;
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "cuda: {}: FusedFFN is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                let od_total = 2 * meta.nf;
                // 1) concat matmul: x × [ffn_gate|ffn_up]
                self.state.matmul_f32_ptr_layout(
                    wptr,
                    meta.weight_ttype,
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(out_buf)?,
                    od_total,
                    meta.in_dim,
                    nt,
                    self.state.is_weight_padded(&meta.gu_weight),
                )?;
                // 2) in-place swiglu: silu(rows 0..nf) × (rows nf..2*nf)
                let n = nt * meta.nf;
                self.state
                    .swiglu_f32_off_on_gpu(self.ptr_of(out_buf)?, n, n);
                Ok(())
            }
            Op::MatMul { transpose_b } => {
                if *transpose_b {
                    return Err(format!(
                        "cuda: {}: transposed matmul not supported",
                        node.name
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::MatMul(m) => m,
                    other => {
                        return Err(format!("matmul node missing MatMulMeta: {other:?}"));
                    }
                };
                let wptr = self
                    .state
                    .get_weight_ptr(&meta.weight_name)
                    .ok_or_else(|| {
                        format!(
                            "cuda: weight '{}' not registered on CUDA ({})",
                            meta.weight_name, node.name
                        )
                    })?;
                let (od, id) = (meta.out_dim, meta.in_dim);
                let nt = node.out_shape[1];
                // quant kernels address whole 32-element blocks; the F32×F32
                // kernel has no such constraint (vec path needs id % 8 == 0
                // and falls back to a scalar kernel otherwise)
                if meta.weight_ttype != crate::tensor::TensorType::F32 && id % 32 != 0 {
                    return Err(format!(
                        "cuda: {}: matmul input dim {id} is not a multiple of the 32-element quant block",
                        node.name
                    ));
                }
                if self.elems(in_bufs[0]) < id * nt || self.elems(out_buf) < od * nt {
                    return Err(format!(
                        "cuda: {}: buffer size mismatch for [{od}x{id}] x nt={nt}",
                        node.name
                    ));
                }
                self.state.matmul_f32_ptr_layout(
                    wptr,
                    meta.weight_ttype,
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(out_buf)?,
                    od,
                    id,
                    nt,
                    self.state.is_weight_padded(&meta.weight_name),
                )?;
                if let Some(bname) = &meta.bias_name {
                    let bptr = self.state.get_weight_ptr(bname).ok_or_else(|| {
                        format!(
                            "cuda: bias '{bname}' not registered on CUDA ({})",
                            node.name
                        )
                    })?;
                    // add_bias_f32's last argument is the ROW COUNT (nt), not
                    // the total element count — the kernel grid maps one block
                    // row per token (a wrong count writes out of bounds).
                    self.state.add_bias_f32(self.ptr_of(out_buf)?, bptr, od, nt);
                }
                Ok(())
            }

            Op::RoPE { style } => {
                if !matches!(style, RopeStyle::NonInterleaved) {
                    return Err(format!(
                        "cuda: rope style {style:?} not supported (kernel is neox/non-interleaved only)"
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::Rope(m) => m,
                    other => return Err(format!("rope node missing RoPEMeta: {other:?}")),
                };
                if meta.hd == 0 || meta.hd % 2 != 0 {
                    return Err(format!(
                        "cuda: {}: rope head dim {} must be even",
                        node.name, meta.hd
                    ));
                }
                if in_bufs[0] != out_buf {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                let nt = node.out_shape[1];
                let pos = self.positions_i32(in_bufs[1])?;
                self.state.rope_f32(
                    self.ptr_of(out_buf)?,
                    meta.n_head,
                    meta.hd,
                    nt,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                );
                Ok(())
            }

            Op::KvcacheStore { layer } => {
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                if out_buf != k_id {
                    return Err(format!(
                        "cuda: kv store output buffer {out_buf} is not the K region {k_id}"
                    ));
                }
                let nkt = node.out_shape[0];
                if nkt == 0 || self.elems(in_bufs[0]) % nkt != 0 {
                    return Err(format!(
                        "cuda: kv store k input {} elems not a multiple of nkt {nkt}",
                        self.elems(in_bufs[0])
                    ));
                }
                let nt = self.elems(in_bufs[0]) / nkt;
                let pos = self.positions_i32(in_bufs[2])?;
                // Note: positions >= n_ctx are not validated here (device-side
                // data); the CPU backend checks them, the GPU backends trust
                // session-level clamping like Metal's store_kv dispatch.
                // 8b: f16 KV stores into the same persistent region viewed as
                // half (2 bytes/elem) — halves attention read bandwidth, same
                // trade-off as Metal's store_kv dispatch.
                let (sk, sv) = (self.ptr_of(in_bufs[0])?, self.ptr_of(in_bufs[1])?);
                let (dk, dv) = (self.ptr_of(k_id)?, self.ptr_of(v_id)?);
                if self.kv_f16 {
                    self.state.store_kv_f16(sk, dk, nkt, nt, pos);
                    self.state.store_kv_f16(sv, dv, nkt, nt, pos);
                } else {
                    self.state.store_kv_f32(sk, dk, nkt, nt, pos);
                    self.state.store_kv_f32(sv, dv, nkt, nt, pos);
                }
                Ok(())
            }

            Op::Attn { .. } => {
                let meta = match &node.meta {
                    NodeMeta::Attn(m) => m,
                    other => return Err(format!("attn node missing AttnMeta: {other:?}")),
                };
                // Same kernel-invariant guards as Metal (docs/GPU_SAFETY.md):
                // the kernel strides KV by nk*hd, uses the query head dim, and
                // keeps hd/4 accumulators in registers (oc[32] → hd ≤ 128).
                if meta.nkt != meta.n_head_kv * meta.hd {
                    return Err(format!(
                        "cuda: attention nkt={} != n_head_kv*hd={} (kernel strides KV by nk*hd)",
                        meta.nkt,
                        meta.n_head_kv * meta.hd
                    ));
                }
                if meta.hd != meta.hd_kv {
                    return Err(format!(
                        "cuda: attention hd={} != hd_kv={} (kernel uses the query head dim)",
                        meta.hd, meta.hd_kv
                    ));
                }
                if meta.hd == 0 || meta.hd > 128 || meta.hd % 4 != 0 {
                    return Err(format!(
                        "cuda: attention head dim {} outside the kernel's supported range (multiple of 4, 1..=128)",
                        meta.hd
                    ));
                }
                if meta.n_head_kv == 0 || meta.n_head % meta.n_head_kv != 0 {
                    return Err(format!(
                        "cuda: attention n_head {} not divisible by n_head_kv {}",
                        meta.n_head, meta.n_head_kv
                    ));
                }
                let (k_id, v_id) = kv_pair
                    .ok_or_else(|| format!("KV regions for layer {} not allocated", meta.layer))?;
                let nt = node.out_shape[1];
                let pos = self.positions_i32(in_bufs[2])?;
                // The causal bound (positions[t]+1) is derived from the device
                // positions inside the kernel — no host scalar crosses here
                // (precondition for CUDA Graph replay, Phase 7d).
                // 8d: decode (nt == 1) uses split-K flash-decoding — the
                // single-warp kernel leaves the GPU idle at nt == 1 (nsys:
                // 48% of the 7B decode step at 2K ctx). Fixed grid +
                // device-side range split keeps CUDA Graph capture valid;
                // the partials scratch is size-stable (nh/hd constants).
                if nt == 1 {
                    self.state.gqa_attn_split(
                        self.ptr_of(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of(out_buf)?,
                        pos,
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        self.kv_f16,
                    );
                    return Ok(());
                }
                // nt > 1 (prefill): single-warp-per-(token, head) kernel —
                // the grid already covers nt × nh blocks.
                // 8b: f16-KV variant reads half K/V (q/o stay f32)
                if self.kv_f16 {
                    self.state.gqa_attn_f16kv(
                        self.ptr_of(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of(out_buf)?,
                        pos,
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        nt,
                    );
                } else {
                    self.state.gqa_attn_f32(
                        self.ptr_of(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of(out_buf)?,
                        pos,
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        nt,
                    );
                }
                Ok(())
            }

            op => Err(format!(
                "cuda: op {op:?} has no kernel (stays on the CPU backend per supports_op)"
            )),
        }
    }

    fn state_free(ptr: *mut std::ffi::c_void) {
        <crate::cuda::CudaState>::cuda_free(ptr);
    }

    fn elems(&self, id: usize) -> usize {
        self.pool[id].bytes / 4
    }

    fn copy_d2d(&self, src: usize, dst: usize) -> Result<(), String> {
        let (s, d) = (self.ptr_of(src)?, self.ptr_of(dst)?);
        let (sb, db) = (self.pool[src].bytes, self.pool[dst].bytes);
        if sb != db {
            return Err(format!(
                "cuda: device copy size mismatch src {sb} vs dst {db} bytes"
            ));
        }
        self.state.copy_device_to_device(s, d, db);
        Ok(())
    }

    /// Decode an I32 input buffer (f32::from_bits bit patterns, alloc.rs
    /// fill_input_i32) into raw int32 on the device. The rope/store/attention
    /// kernels read `const int* positions`; one tiny elementwise pass keeps
    /// the whole path on-device — no host sync, and the pointer stays stable
    /// across steps (a precondition for CUDA Graph replay in Phase 7d).
    fn positions_i32(&mut self, id: usize) -> Result<*mut std::ffi::c_void, String> {
        let src = self.ptr_of(id)?;
        let bytes = self.pool[id].bytes;
        if self.pos_scratch_bytes < bytes {
            if !self.pos_scratch.is_null() {
                Self::state_free(self.pos_scratch);
            }
            let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
            if ptr.is_null() {
                self.pos_scratch_bytes = 0;
                return Err("cuda: positions scratch allocation failed".to_string());
            }
            self.pos_scratch = ptr;
            self.pos_scratch_bytes = bytes;
            // the freed scratch pointer may be embedded in captured graph
            // execs — invalidate them so they re-capture against the new
            // address (Phase 8 review; currently masked because growth only
            // happens on a larger prefill whose allocs churn pool_gen anyway)
            self.pool_gen += 1;
        }
        self.state.bits_to_i32(src, self.pos_scratch, bytes / 4);
        Ok(self.pos_scratch)
    }

    /// Resolve a NormMeta weight by name on the CUDA registry. Unlike Metal
    /// (which silently degrades to a weightless norm when the weight is not on
    /// the backend), a declared-but-missing weight is an invariant violation
    /// here and returns Err (docs/GPU_SAFETY.md).
    fn norm_weight(&self, node: &CNode) -> Result<*mut std::ffi::c_void, String> {
        let name = match &node.meta {
            NodeMeta::Norm(m) => m.weight_name.as_deref(),
            other => {
                return Err(format!(
                    "cuda: {} node missing NormMeta: {other:?}",
                    node.name
                ))
            }
        };
        let Some(name) = name else {
            return Err(format!(
                "cuda: {} has no norm weight (the CUDA rms_norm kernel requires one)",
                node.name
            ));
        };
        self.state.get_weight_ptr(name).ok_or_else(|| {
            format!(
                "cuda: weight '{name}' not registered on CUDA ({})",
                node.name
            )
        })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    /// v1 capability matrix (docs/CUDA-BACKEND-PLAN.md §4.3): the full
    /// per-layer chain runs on CUDA; Embed/GetRows, Scale, Softmax and the
    /// fused decode ops have no kernels and stay on the CPU backend. RoPE is
    /// gated to the neox (non-interleaved) layout — the only style the
    /// supported architectures emit.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        if dtype != DType::F32 {
            return false;
        }
        match op {
            Op::Input
            | Op::Add
            | Op::Mul
            | Op::Silu
            | Op::SwiGLU
            | Op::RmsNorm { .. }
            | Op::QkNorm { .. }
            | Op::MatMul { .. }
            | Op::Attn { .. }
            | Op::KvcacheStore { .. }
            | Op::KvcacheLoad { .. }
            | Op::View { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
            // 7e③: row gather — the embedding (Embed meta, weight dequant by
            // type; weight-type support is enforced by the model-level gate,
            // which only admits F32/Q4_0/Q8_0/Q4_K/Q6_K tok_embd) and the
            // generic f32 tail gather (no meta). Removes the CPU round trips
            // around the prefill's embed and G3 tail reduction.
            // 7e⑤: decode FFN gate+up fusion — concat matmul + in-place
            // offset swiglu (the gu_concat_available / CParams.fuse_ffn
            // gates decide when the node is built).
            | Op::GetRows
            | Op::FusedFFN => true,
            // MatMul ttype gating happens at the model level (weights must all
            // be registered on CUDA — same all-or-nothing rule as Metal).
            Op::RoPE { style } => matches!(style, RopeStyle::NonInterleaved),
            _ => false,
        }
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        matches!(fused, FusedOp::SwiGLU)
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        let _sg = self.stream_guard(); // cudaMalloc syncs the device
        let bytes = size * 4;
        if let Some(pos) = self
            .free
            .iter()
            .position(|&id| self.pool[id].bytes == bytes)
        {
            let id = self.free.remove(pos);
            self.pool_gen += 1;
            return id;
        }
        // On OOM, cuda_malloc logs and returns null; the null buffer fails
        // cleanly (Err) at execute time via ptr_of — do NOT panic here: the
        // backend may be holding the process-wide stream lock, and panicking
        // under a mutex poisons it for every other user.
        let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
        self.pool.push(CudaBuf { ptr, bytes });
        self.pool_gen += 1;
        self.pool.len() - 1
    }

    fn free_buffer(&mut self, id: usize) {
        let _sg = self.stream_guard();
        // Recycle, never cudaFree here: persistent KV regions survive rebuilds
        // and the pool keeps freed device memory for reuse (CPU/Metal alike).
        if !self.free.contains(&id) {
            self.free.push(id);
        }
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // bypass the free list entirely (see Backend::alloc_fresh): the ids in
        // it are still referenced by node_to_buf and physically live during
        // the execute that follows
        let _sg = self.stream_guard(); // cudaMalloc syncs the device
        let bytes = size * 4;
        // On OOM, cuda_malloc logs and returns null; the null buffer fails
        // cleanly (Err) at execute time via ptr_of — do NOT panic here: the
        // backend may be holding the process-wide stream lock, and panicking
        // under a mutex poisons it for every other user.
        let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
        self.pool.push(CudaBuf { ptr, bytes });
        self.pool_gen += 1;
        self.pool.len() - 1
    }

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[usize],
        out_buf: usize,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        match self.execute_node_inner(node, in_bufs, out_buf, kv_pair) {
            Ok(()) => Ok(()),
            Err(e) => {
                // A node error during an open capture window dooms the window:
                // the scheduler propagates before the boundary sync, so nothing
                // would close it — later input fills would be RECORDED into the
                // window and the eventual close would cache a multi-step graph
                // (double KV commit on every replay). Abort the window loudly.
                if self.capturing.is_some() {
                    self.abort_capture(&e);
                }
                Err(e)
            }
        }
    }

    fn read_host(&self, _id: usize) -> Option<&[f32]> {
        // A staged D2H transfer cannot return a borrowed slice (this method
        // takes &self; the host staging buffer would escape its guard). Use
        // `copy_to_host` via alloc.rs's copy_to_cpu CUDA arm instead.
        None
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let _sg = self.stream_guard();
        let bytes = data.len() * 4;
        let dst = self.ptr_of(id)?;
        if self.pool[id].bytes < bytes {
            return Err(format!(
                "cuda: buffer {id} too small: {} < {bytes} bytes",
                self.pool[id].bytes
            ));
        }
        // 7e⑥: pinned-staged async fill (same-stream ordering makes this
        // race-free with the kernels that read the input; the ring syncs
        // only if more than STAGING_SLOTS fills queue up without a sync).
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
        self.state.write_input_async(src, dst);
        Ok(())
    }

    fn synchronize(&mut self) {
        if self.capturing.is_none() {
            let _sg = self.stream_guard();
        }
        self.close_capture_or_sync();
    }

    fn graph_replay(&mut self, uid: u64, range: (usize, usize), nt_hint: Option<usize>) -> bool {
        self.graph_replay_step(uid, range, nt_hint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::backend::{Backend as _, KvProvider};
    use crate::graph::builder::GraphBuilder;
    use crate::graph::cache::GraphCache;
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::DType;

    /// Init the CUDA singleton; silent-skip the test when no device answers
    /// (e.g. CI without a GPU). Run with --nocapture to see skips.
    fn device() -> Option<&'static crate::cuda::CudaState> {
        crate::cuda::CudaState::init();
        crate::cuda::CudaState::get()
    }

    #[test]
    fn cuda_pool_roundtrip() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut cb = CudaBackend::new().expect("backend after device init");
        let id = cb.alloc_buffer(16);
        cb.write_host(id, &[1.5f32; 16]).unwrap();
        assert_eq!(cb.copy_to_host(id).unwrap(), vec![1.5f32; 16]);
        // shorter than the buffer is fine, longer is rejected
        cb.write_host(id, &[2.0f32; 4]).unwrap();
        assert!(cb.write_host(id, &[2.0f32; 32]).is_err());
        // free-list reuse hands back the same id; pool_gen tracked both times
        cb.free_buffer(id);
        let id2 = cb.alloc_buffer(16);
        assert_eq!(id, id2);
        assert_eq!(cb.pool_gen, 2);
    }

    #[test]
    fn cuda_pinned_readback_roundtrip() {
        // 5.6 MB > the 4 MiB initial pinned readback buffer: exercises the
        // grow-on-demand path of copy_from_device_pinned (R3-A2).
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut cb = CudaBackend::new().expect("backend after device init");
        let n = 1_400_000usize; // elements — alloc_buffer takes an element count
        let id = cb.alloc_buffer(n);
        let data: Vec<f32> = (0..n).map(|i| (i % 997) as f32 + 0.5).collect();
        cb.write_host(id, &data).unwrap();
        let got = cb.copy_to_host(id).unwrap();
        assert_eq!(got.len(), n);
        assert_eq!(got, data, "full roundtrip mismatch");
    }

    #[test]
    fn copy_across_cpu_to_cuda_and_back() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut b = crate::graph::builder::GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        alloc.alloc_graph(&g).unwrap();
        let data = [1.0f32, -2.0, 3.0, 4.0];
        alloc.fill_input(&g, "x", &data).unwrap();

        let canon = alloc.node_buffer(x).unwrap();
        // CPU → CUDA staging copy: canonical buffer untouched, cross map holds
        // the device copy (Phase 7c: the old remap-into-node_to_buf semantics
        // broke re-execution of reused graphs — the producing split found its
        // buffer remapped to another backend on the next execute)
        alloc.copy_across(x, crate::graph::Backend::Cuda).unwrap();
        let cross = alloc.cross_buffer(x).expect("cross staging buffer");
        assert_eq!(cross.backend, crate::graph::Backend::Cuda);
        assert_eq!(
            alloc.node_buffer(x).unwrap(),
            canon,
            "canonical buffer must not be remapped"
        );
        assert_eq!(alloc.copy_to_cpu(x).unwrap(), data.to_vec());
        // re-copy (same dst) reuses the same staging buffer id
        alloc.copy_across(x, crate::graph::Backend::Cuda).unwrap();
        assert_eq!(alloc.cross_buffer(x).unwrap().id, cross.id);
        // same-backend copy is a no-op
        alloc.copy_across(x, crate::graph::Backend::CPU).unwrap();
        assert!(alloc.cross_buffer(x).unwrap().backend == crate::graph::Backend::Cuda);
        // rebuild clears staging (buffers freed, map empty)
        alloc.alloc_graph(&g).unwrap();
        assert!(alloc.cross_buffer(x).is_none());
    }

    #[test]
    fn kv_persistent_regions_survive_realloc() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut b = crate::graph::builder::GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
        let k = b.input("k", [16, 1, 1, 1], DType::F32);
        let v = b.input("v", [16, 1, 1, 1], DType::F32);
        let store = b.kvcache_store(0, k, v, pos, 1024);
        let load = b.kvcache_load(0, 16, 1024, 2);
        b.output(load);
        let mut g = b.build();
        g.nodes[store].backend = Some(crate::graph::Backend::Cuda);
        g.nodes[load].backend = Some(crate::graph::Backend::Cuda);

        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        alloc.alloc_graph(&g).unwrap();
        let pair = alloc.kv_pair(0).unwrap();

        // the store node's buffer IS the K region, on the CUDA pool
        let kbuf = alloc.node_buffer(store).unwrap();
        assert_eq!(kbuf.backend, crate::graph::Backend::Cuda);
        assert_eq!(kbuf.id, pair.0);
        {
            let c = alloc.cuda_mut().unwrap();
            c.write_host(kbuf.id, &[7.5f32; 16]).unwrap();
        }

        // rebuild: liveness buffers recycle, KV regions survive unchanged
        alloc.alloc_graph(&g).unwrap();
        assert_eq!(alloc.kv_pair(0).unwrap(), pair);
        let back = alloc.copy_to_cpu(store).unwrap();
        assert_eq!(&back[..16], &[7.5f32; 16]);
    }

    // ─── Phase 7b: per-op dispatch parity ───────────────────────

    use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
    use crate::tensor::{Tensor, TensorType};

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

    #[test]
    fn cuda_elementwise_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let n = 257usize; // odd size exercises the elementwise tail guard
        let mut b = GraphBuilder::new();
        let a = b.input("a", [n, 1, 1, 1], DType::F32);
        let c = b.input("c", [n, 1, 1, 1], DType::F32);
        let add = b.add(a, c);
        let mul = b.mul(add, c);
        let sw = b.swiglu(mul, c); // gate is the RAW pre-activation
        let silu = b.silu(mul);
        b.output(silu);
        b.output(sw);
        let g = b.build();

        let (x, y) = (cb.alloc_buffer(n), cb.alloc_buffer(n));
        let (t1, t2, t3) = (cb.alloc_buffer(n), cb.alloc_buffer(n), cb.alloc_buffer(n));
        let xs: Vec<f32> = (0..n).map(|i| ((i * 37) % 23) as f32 / 4.0 - 2.5).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i * 91) % 17) as f32 / 3.0 - 2.0).collect();
        cb.write_host(x, &xs).unwrap();
        cb.write_host(y, &ys).unwrap();

        cb.execute_node(&g.nodes[add], &[x, y], t1, None).unwrap();
        cb.execute_node(&g.nodes[mul], &[t1, y], t2, None).unwrap();
        // SwiGLU consumes the RAW mul output, so it must run before the
        // in-place Silu overwrites t2 (alias path, graph rules §5).
        cb.execute_node(&g.nodes[sw], &[t2, y], t3, None).unwrap();
        cb.execute_node(&g.nodes[silu], &[t2], t2, None).unwrap();

        // Host reference through the same vec_ops the CPU backend uses.
        let mut r1 = vec![0f32; n];
        crate::vec_ops::vec_add_f32(n, &mut r1, &xs, &ys);
        assert_eq!(cb.copy_to_host(t1).unwrap(), r1, "add must be bit-exact");
        let mut r2 = vec![0f32; n];
        crate::vec_ops::vec_mul_f32(n, &mut r2, &r1, &ys);
        let mut r3 = vec![0f32; n];
        crate::vec_ops::vec_silu_f32(n, &mut r3, &r2);
        let got2 = cb.copy_to_host(t2).unwrap();
        assert_close("mul+silu (in-place)", &got2, &r3, 1e-5);
        let mut r4 = vec![0f32; n];
        crate::vec_ops::vec_mul_f32(n, &mut r4, &r3, &ys);
        assert_close("swiglu", &cb.copy_to_host(t3).unwrap(), &r4, 1e-5);
    }

    #[test]
    fn cuda_norm_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // RmsNorm: d=64, nt=5
        let (d, nt) = (64usize, 5usize);
        let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.state.register_weight("nw", &wbytes);
        let mut wt = Tensor::from_data(TensorType::F32, &[d as i64, 1, 1, 1], wbytes.clone());
        wt.name = "nw".to_string();

        // QkNorm: hd=16, nh=4, nt=3 — rows (t*nh + h) form a contiguous
        // [nt*nh, hd] matrix, so the same rms_norm kernel covers it.
        let (hd, nh, nt2) = (16usize, 4usize, 3usize);
        let qw: Vec<f32> = (0..hd).map(|i| 1.0 / (1.0 + i as f32)).collect();
        let qbytes: Vec<u8> = qw.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.state.register_weight("qw", &qbytes);
        let mut qwt = Tensor::from_data(TensorType::F32, &[hd as i64, 1, 1, 1], qbytes);
        qwt.name = "qw".to_string();

        let mut b = GraphBuilder::new();
        let x = b.input("x", [d, nt, 1, 1], DType::F32);
        let rn = b.rms_norm(x, Some(&wt), 1e-5);
        let q = b.input("q", [hd * nh, nt2, 1, 1], DType::F32);
        let qn = b.qk_norm(q, Some(&qwt), hd, nh, 1e-5);
        b.output(rn);
        b.output(qn);
        let g = b.build();

        let xb = cb.alloc_buffer(d * nt);
        let xs: Vec<f32> = (0..d * nt)
            .map(|i| ((i * 53) % 31) as f32 / 7.0 - 2.0)
            .collect();
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(d * nt);
        cb.execute_node(&g.nodes[rn], &[xb], ob, None).unwrap();

        let qb = cb.alloc_buffer(hd * nh * nt2);
        let qs: Vec<f32> = (0..hd * nh * nt2)
            .map(|i| ((i * 71) % 29) as f32 / 6.0 - 2.5)
            .collect();
        cb.write_host(qb, &qs).unwrap();
        let qo = cb.alloc_buffer(hd * nh * nt2);
        cb.execute_node(&g.nodes[qn], &[qb], qo, None).unwrap();

        let mut want = vec![0f32; d * nt];
        for t in 0..nt {
            crate::vec_ops::rms_norm_fused_f32(
                d,
                &mut want[t * d..(t + 1) * d],
                &xs[t * d..(t + 1) * d],
                &w,
                1e-5,
            );
        }
        assert_close("rms_norm", &cb.copy_to_host(ob).unwrap(), &want, 1e-4);

        let mut want2 = vec![0f32; hd * nh * nt2];
        for r in 0..nh * nt2 {
            crate::vec_ops::rms_norm_fused_f32(
                hd,
                &mut want2[r * hd..(r + 1) * hd],
                &qs[r * hd..(r + 1) * hd],
                &qw,
                1e-5,
            );
        }
        assert_close("qk_norm", &cb.copy_to_host(qo).unwrap(), &want2, 1e-4);
    }

    #[test]
    fn cuda_matmul_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (od, id_, nt) = (32usize, 64usize, 3usize);
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
            .collect();
        let bias: Vec<f32> = (0..od).map(|i| (i % 5) as f32 / 10.0).collect();

        // Q8_0 weight [out][in] row-major, quantized per row
        let wf8: Vec<f32> = (0..od * id_)
            .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let mut w8b = Vec::new();
        for r in 0..od {
            w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
                &wf8[r * id_..(r + 1) * id_],
            ));
        }
        let mut w8 = Tensor::from_data(
            TensorType::Q8_0,
            &[id_ as i64, od as i64, 1, 1],
            w8b.clone(),
        );
        w8.name = "mw8".to_string();
        cb.state.register_weight("mw8", &w8b);
        let biasb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.state.register_weight("mb", &biasb);
        let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], biasb);
        bt.name = "mb".to_string();

        // Q4_0 weight (18 bytes per 32 values: f16 d + 16 nibbles)
        let wf4: Vec<f32> = (0..od * id_)
            .map(|i| ((i * 40503) % 991) as f32 / 496.0 - 1.0)
            .collect();
        let mut w4b = Vec::new();
        for r in 0..od {
            let row = &wf4[r * id_..(r + 1) * id_];
            for bi in 0..id_ / 32 {
                let blk = &row[bi * 32..bi * 32 + 32];
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let dsc = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
                w4b.extend_from_slice(&half::f16::from_f32(dsc).to_le_bytes());
                for j in 0..16 {
                    let q0 = ((blk[j] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                    let q1 = ((blk[j + 16] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                    w4b.push(q0 | (q1 << 4));
                }
            }
        }
        let mut w4 = Tensor::from_data(
            TensorType::Q4_0,
            &[id_ as i64, od as i64, 1, 1],
            w4b.clone(),
        );
        w4.name = "mw4".to_string();
        cb.state.register_weight("mw4", &w4b);

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id_, nt, 1, 1], DType::F32);
        let m8 = b.matmul(x, &w8, Some(&bt));
        let m4 = b.matmul(x, &w4, Some(&bt));
        b.output(m8);
        b.output(m4);
        let g = b.build();

        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let (o8, o4) = (cb.alloc_buffer(od * nt), cb.alloc_buffer(od * nt));
        cb.execute_node(&g.nodes[m8], &[xb], o8, None).unwrap();
        cb.execute_node(&g.nodes[m4], &[xb], o4, None).unwrap();

        // References: dequantized weight rows × f32 activations + bias
        // (embed_tokens doubles as the row dequantizer for these types).
        let mut dq8 = vec![0f32; od * id_];
        crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8, &mut dq8, id_);
        let mut dq4 = vec![0f32; od * id_];
        crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w4, &mut dq4, id_);
        for (name, o, dq, quantize_acts) in [
            ("q8_0 matmul", o8, &dq8, false),
            // 8c: prefill Q4_0 runs the Q8_0-activation GEMM (the CPU path
            // has always quantized activations too) — mirror that in the
            // reference and keep a tight tolerance.
            ("q4_0 matmul", o4, &dq4, true),
        ] {
            let got = cb.copy_to_host(o).unwrap();
            let mut want = vec![0f32; od * nt];
            for t in 0..nt {
                let xrow = &xs[t * id_..(t + 1) * id_];
                let acts: Vec<f32> = if quantize_acts {
                    let q8 = crate::quants::quantize_row_q8_0(xrow);
                    (0..id_)
                        .map(|i| {
                            let b = i / 32;
                            let d8 =
                                half::f16::from_le_bytes([q8[b * 34], q8[b * 34 + 1]]).to_f32();
                            d8 * q8[b * 34 + 2 + (i % 32)] as i8 as f32
                        })
                        .collect()
                } else {
                    xrow.to_vec()
                };
                for r in 0..od {
                    let mut acc = 0f32;
                    for i in 0..id_ {
                        acc += dq[r * id_ + i] * acts[i];
                    }
                    want[t * od + r] = acc + bias[r];
                }
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close(name, &got, &want, scale * 2e-2);
        }
    }

    /// 7e②: K-quant matmul parity (Q4_K + Q6_K). The reference dequantizes
    /// each row with an independent in-test implementation of the
    /// llama.cpp block layout and dots it with the f32 activations. The
    /// original scalar CUDA kernels and the 7e② vectorized ones must both
    /// agree with it (coverage gap found in 7e②: q6_K previously had NO
    /// parity test, which let a broken vectorized variant pass the suite).
    #[test]
    fn cuda_kquant_matmul_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // id_ = 512 = 2 super-blocks of 256; od = 8 rows (NR0 = 2 → 4 row
        // pairs across 2 warps per block).
        let (od, id_, nt) = (8usize, 512usize, 3usize);
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
            .collect();

        // get_scale_min_k4 (llama.cpp Q4_K scale packing, reimplemented
        // here independently of the kernel under test).
        fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }

        // ── Q4_K tensor: 144 bytes per 256-element super-block ──
        // layout: f16 d, f16 dmin, u8 scales[12], nibble bytes qs[128]
        let mut w4b = Vec::new();
        let mut w4dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..id_ / 256 {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                w4b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                w4b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    scb[j] = ((r * 31 + j * 17 + ib * 5) % 63) as u8;
                }
                w4b.extend_from_slice(&scb);
                let mut qs = [0u8; 128];
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    qs[j] = lo | (hi << 4);
                }
                w4b.extend_from_slice(&qs);
                // reference dequant: LOW nibbles of bytes[32j..32j+31] are
                // elements [64j..64j+31] (scale 2j), HIGH nibbles are
                // elements [64j+32..64j+63] (scale 2j+1);
                // value = d*sc*nibble - dmin*m
                for j in 0..4 {
                    let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                    let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b = qs[j * 32 + l];
                        let base = r * id_ + ib * 256 + j * 64;
                        w4dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                        w4dq[base + 32 + l] =
                            (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                    }
                }
            }
        }

        // ── Q6_K tensor: 210 bytes per 256-element super-block ──
        // layout: ql[128], qh[64], i8 scales[16], f16 d
        let mut w6b = Vec::new();
        let mut w6dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..id_ / 256 {
                let d = 0.027f32 + 0.004 * ((r * 11 + ib * 7) % 6) as f32;
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut sc = [0i8; 16];
                for i in 0..128 {
                    ql[i] = ((r * 29 + i * 7 + ib * 3) % 255) as u8;
                }
                for i in 0..64 {
                    qh[i] = ((r * 17 + i * 13 + ib * 11) % 255) as u8;
                }
                for i in 0..16 {
                    sc[i] = (((r * 5 + i * 3 + ib) % 15) as i8) - 7;
                }
                w6b.extend_from_slice(&ql);
                w6b.extend_from_slice(&qh);
                w6b.extend(sc.iter().map(|&x| x as u8));
                w6b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                // reference dequant (llama.cpp Q6_K layout):
                // value = d * sc[n*8 + l/16 + t*2] * (nibble|2bits<<4 - 32)
                for n in 0..2usize {
                    let qlh = &ql[n * 64..n * 64 + 64];
                    let qhh = &qh[n * 32..n * 32 + 32];
                    for l in 0..32usize {
                        let is = l / 16;
                        let q1 = ((qlh[l] & 0xF) as i32 | (((qhh[l] >> 0) as i32 & 3) << 4)) - 32;
                        let q2 =
                            ((qlh[l + 32] & 0xF) as i32 | (((qhh[l] >> 2) as i32 & 3) << 4)) - 32;
                        let q3 = ((qlh[l] >> 4) as i32 | (((qhh[l] >> 4) as i32 & 3) << 4)) - 32;
                        let q4 =
                            ((qlh[l + 32] >> 4) as i32 | (((qhh[l] >> 6) as i32 & 3) << 4)) - 32;
                        let base = r * id_ + ib * 256 + n * 128;
                        w6dq[base + l] = d * sc[n * 8 + is] as f32 * q1 as f32;
                        w6dq[base + l + 32] = d * sc[n * 8 + is + 2] as f32 * q2 as f32;
                        w6dq[base + l + 64] = d * sc[n * 8 + is + 4] as f32 * q3 as f32;
                        w6dq[base + l + 96] = d * sc[n * 8 + is + 6] as f32 * q4 as f32;
                    }
                }
            }
        }

        let mut w4t = Tensor::from_data(
            TensorType::Q4_K,
            &[id_ as i64, od as i64, 1, 1],
            w4b.clone(),
        );
        w4t.name = "mw4k".to_string();
        cb.state.register_weight("mw4k", &w4b);
        let mut w6t = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6t.name = "mw6k".to_string();
        cb.state.register_weight("mw6k", &w6b);
        // 7e② padded layout path (register_weight_q6k_padded)
        cb.state.register_weight_q6k_padded("mw6kp", &w6b, od, id_);
        assert!(cb.state.is_weight_padded("mw6kp"));

        let mut w6pt = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6pt.name = "mw6kp".to_string();

        // ── F32 weight (7e④): aligned id (512) and odd id (513, scalar path)
        let wfb: Vec<u8> = w4dq.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut wft =
            Tensor::from_data(TensorType::F32, &[id_ as i64, od as i64, 1, 1], wfb.clone());
        wft.name = "mwf32".to_string();
        cb.state.register_weight("mwf32", &wfb);
        // odd id: 513-wide rows (first 512 = w4dq, element 512 synthetic)
        let (od_o, id_o) = (8usize, 513usize);
        let mut wfo_vals = Vec::with_capacity(od_o * id_o);
        for r in 0..od_o {
            for i in 0..id_o {
                wfo_vals.push(if i < id_ {
                    w4dq[r * id_ + i]
                } else {
                    (r + 1) as f32 * 0.25
                });
            }
        }
        let wfo: Vec<u8> = wfo_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut wfot = Tensor::from_data(
            TensorType::F32,
            &[id_o as i64, od_o as i64, 1, 1],
            wfo.clone(),
        );
        wfot.name = "mwf32o".to_string();
        cb.state.register_weight("mwf32o", &wfo);

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id_, nt, 1, 1], DType::F32);
        let m4 = b.matmul(x, &w4t, None);
        let m6 = b.matmul(x, &w6t, None);
        let m6p = b.matmul(x, &w6pt, None);
        let mf = b.matmul(x, &wft, None);
        b.output(m4);
        b.output(m6);
        b.output(m6p);
        b.output(mf);
        // odd-id graph: x sliced to id_ = 513
        let xo = b.input("xo", [id_o, nt, 1, 1], DType::F32);
        let mfo = b.matmul(xo, &wfot, None);
        b.output(mfo);
        let g = b.build();

        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let (o4, o6, o6p) = (
            cb.alloc_buffer(od * nt),
            cb.alloc_buffer(od * nt),
            cb.alloc_buffer(od * nt),
        );
        let of = cb.alloc_buffer(od * nt);
        cb.execute_node(&g.nodes[m4], &[xb], o4, None).unwrap();
        cb.execute_node(&g.nodes[m6], &[xb], o6, None).unwrap();
        cb.execute_node(&g.nodes[m6p], &[xb], o6p, None).unwrap();
        cb.execute_node(&g.nodes[mf], &[xb], of, None).unwrap();
        let mut xso = xs.clone();
        xso.resize(id_o * nt, 0.25f32); // extend for the odd-id input
        let xob = cb.alloc_buffer(id_o * nt);
        cb.write_host(xob, &xso).unwrap();
        let ofo = cb.alloc_buffer(od_o * nt);
        cb.execute_node(&g.nodes[mfo], &[xob], ofo, None).unwrap();

        for (name, o, dq) in [
            ("q4_k matmul", o4, &w4dq),
            ("q6_k matmul", o6, &w6dq),
            ("q6_k padded matmul", o6p, &w6dq),
        ] {
            let got = cb.copy_to_host(o).unwrap();
            let mut want = vec![0f32; od * nt];
            for t in 0..nt {
                for r in 0..od {
                    let mut acc = 0f32;
                    for i in 0..id_ {
                        acc += dq[r * id_ + i] * xs[t * id_ + i];
                    }
                    want[t * od + r] = acc;
                }
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close(name, &got, &want, scale * 2e-3);
        }

        // F32 matmul (aligned + odd-id scalar path) vs the same reference rows
        {
            let got = cb.copy_to_host(of).unwrap();
            let mut want = vec![0f32; od * nt];
            for t in 0..nt {
                for r in 0..od {
                    let mut acc = 0f32;
                    for i in 0..id_ {
                        acc += w4dq[r * id_ + i] * xs[t * id_ + i];
                    }
                    want[t * od + r] = acc;
                }
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close("f32 matmul", &got, &want, scale * 2e-3);
        }
        {
            let got = cb.copy_to_host(ofo).unwrap();
            let mut want = vec![0f32; od_o * nt];
            for t in 0..nt {
                for r in 0..od_o {
                    let mut acc = 0f32;
                    for i in 0..id_o {
                        acc += wfo_vals[r * id_o + i] * xso[t * id_o + i];
                    }
                    want[t * od_o + r] = acc;
                }
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close("f32 matmul odd id", &got, &want, scale * 2e-3);
        }
    }

    /// 7e⑤: fused FFN gate+up parity — the concat matmul + in-place offset
    /// swiglu must equal silu(gate·x)·(up·x) computed on the host, for a
    /// plain-registered q4_K concat and a padded-repacked q6_K concat.
    /// The reference dequantizes with the same independent in-test block
    /// layouts as `cuda_kquant_matmul_parity`.
    #[test]
    fn cuda_q4k_decode_mmvq_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // the mmvq scratch (buf_q8_decode) is singleton state — serialize the
        // parallel mmvq parity tests so one test's scratch grow cannot free
        // the buffer another test just enqueued kernels against
        let _guard = crate::cuda::CudaState::model_load_guard();
        // 8e-reversal: decode (nt == 1) Q4_K dispatches to the MMVQ structure
        // kernel (q8 activations + dp4a, one 256-thread block per row) when
        // id >= 2048 && id % 32 == 0. Reference: independent dequant of the
        // same bytes + the SAME q8 activation quantization the kernel applies
        // (round-trip error ~2/127 per element), so the tolerance stays tight.
        let (od, id_, nt) = (5120usize, 3584usize, 1usize); // 7B fused-qkv shape
                                                            // activations with real-model spread (RMSNorm outputs reach ±4, and
                                                            // some 32-blocks are near-zero: exercises the f16 d8 denormal range)
        let mut rng_state = 12345u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0; // [-1, 1)
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

        fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }

        let mut w4b = Vec::new();
        let mut w4dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..id_ / 256 {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                w4b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                w4b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    scb[j] = ((r * 31 + j * 17 + ib * 5) % 63) as u8;
                }
                w4b.extend_from_slice(&scb);
                let mut qs = [0u8; 128];
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    qs[j] = lo | (hi << 4);
                }
                w4b.extend_from_slice(&qs);
                for j in 0..4 {
                    let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                    let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b = qs[j * 32 + l];
                        let base = r * id_ + ib * 256 + j * 64;
                        w4dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                        w4dq[base + 32 + l] =
                            (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                    }
                }
            }
        }

        // mirror quantize_q8_0_pad40: per 32-element block, d = amax/127,
        // payload at byte offset 4 (padded 40B blocks)
        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut w4t = Tensor::from_data(
            TensorType::Q4_K,
            &[id_ as i64, od as i64, 1, 1],
            w4b.clone(),
        );
        w4t.name = "mmw4k".to_string();
        cb.state.register_weight("mmw4k", &w4b);

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id_, nt, 1, 1], DType::F32);
        let m = b.matmul(x, &w4t, None);
        b.output(m);
        let g = b.build();

        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.execute_node(&g.nodes[m], &[xb], ob, None).unwrap();
        let got = cb.copy_to_host(ob).unwrap();

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w4dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        // q8 activation quantization noise (<= 2/127 per element, random
        // signs over 2048 elements) lands well under 1e-2 of the row scale.
        assert_close("q4_k decode mmvq", &got, &want, scale * 1e-2);

        // ── real 7B weights (Q4_K tensors of the decode path) ──
        // dumped via the ignored `dump_real_q4k_tensor` helper; skipped when
        // the file is absent so the suite stays hermetic.
        let real_shapes = [
            ("real_blk_0_attn_q_weight.bin", 3584usize, 3584usize),
            ("real_blk_0_attn_k_weight.bin", 3584usize, 512usize),
            ("real_blk_0_ffn_gate_weight.bin", 3584usize, 18944usize),
        ];
        for (file, id2, od2) in real_shapes {
            let Ok(wb) = std::fs::read(format!("/tmp/minfer_phase7/{file}")) else {
                break;
            };
            assert_eq!(wb.len(), od2 * (id2 / 256) * 144);
            let mut w2t = Tensor::from_data(
                TensorType::Q4_K,
                &[id2 as i64, od2 as i64, 1, 1],
                wb.clone(),
            );
            w2t.name = "realq4k".to_string();
            cb.state.register_weight("realq4k", &wb);
            let mut b2 = GraphBuilder::new();
            let x2 = b2.input("x2", [id2, 1, 1, 1], DType::F32);
            let m2 = b2.matmul(x2, &w2t, None);
            b2.output(m2);
            let g2 = b2.build();
            let xb2 = cb.alloc_buffer(id2);
            cb.write_host(xb2, &xs).unwrap();
            let ob2 = cb.alloc_buffer(od2);
            cb.execute_node(&g2.nodes[m2], &[xb2], ob2, None).unwrap();
            let got2 = cb.copy_to_host(ob2).unwrap();

            // CPU reference over the real bytes: dequant (same map as the
            // parity reference above) + q8-quantized activations
            let mut want2 = vec![0f32; od2];
            for r in 0..od2 {
                let mut acc = 0f32;
                for ib in 0..id2 / 256 {
                    let blk = &wb[(r * (id2 / 256) + ib) * 144..];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                    let mut scb = [0u8; 12];
                    scb.copy_from_slice(&blk[4..16]);
                    for j in 0..4 {
                        let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                        let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                        for l in 0..32 {
                            let b8 = blk[16 + j * 32 + l];
                            let base = ib * 256 + j * 64;
                            let v_lo = (b8 & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                            let v_hi = (b8 >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                            acc += v_lo * dq8(base + l);
                            acc += v_hi * dq8(base + 32 + l);
                        }
                    }
                }
                want2[r] = acc;
            }
            // activations here only cover id2=3584 — regenerate quantization
            // for the real id (xs has id_=3584 entries from the scaled-up test)
            let scale2 = want2.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            let mut worst = (0f32, 0usize);
            for r in 0..od2 {
                let e = (got2[r] - want2[r]).abs();
                if e > worst.0 {
                    worst = (e, r);
                }
            }
            println!(
                "real q4_k [{file}]: max err {:.4} at row {} (got {:.4} want {:.4}, scale {scale2:.3})",
                worst.0, worst.1, got2[worst.1], want2[worst.1]
            );
            assert!(
                worst.0 <= scale2 * 1e-2,
                "real q4_k mmvq err {} > {}",
                worst.0,
                scale2 * 1e-2
            );
        }
    }

    /// 8e follow-up: decode (nt == 1) Q6_K dispatches to the MMVQ structure
    /// kernel (16-element units over q8 activations). The synthetic block
    /// covers a partial tail super-block (id = 2176 = 8×256 + 128) and full
    /// 6-bit scale/nibble ranges; both strides go through direct
    /// `q6_k_decode_mmvq` calls — the synthetic shape (od=64) is far below
    /// the measured od*id >= 24M dispatch gate on purpose (small shapes
    /// stay on the f32 kernel at runtime), and gate coverage comes from the
    /// full-size 7B ffn_down real-weight section below.
    #[test]
    fn cuda_q6k_decode_mmvq_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // serialize against the other mmvq parity tests (shared scratch)
        let _guard = crate::cuda::CudaState::model_load_guard();
        let (od, id_, nt) = (64usize, 2176usize, 1usize); // partial tail super-block
        let mut rng_state = 12345u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

        let nbe = (id_ + 255) / 256;
        let mut w6b = Vec::new();
        let mut w6dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.02f32 + 0.004 * ((r * 7 + ib * 3) % 5) as f32;
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut sc = [0u8; 16];
                for s in 0..16usize {
                    sc[s] = (((r * 11 + s * 5 + ib * 3) % 64) as i32 - 32) as u8;
                }
                // only the real elements of a partial tail super-block
                let n_elem = 256usize.min(id_ - ib * 256);
                for e in 0..n_elem {
                    let s = e / 16;
                    let l = e % 16;
                    let chunk = s / 8;
                    let g = (s / 2) % 4;
                    let is = s % 2;
                    let q6 = ((r * 13 + e * 7 + ib * 3) % 64) as u8;
                    w6dq[r * id_ + ib * 256 + e] = d * (sc[s] as i8 as f32) * (q6 as f32 - 32.0);
                    let nib = q6 & 0xF;
                    let hi2 = (q6 >> 4) & 3;
                    let qpos = chunk * 64 + (g % 2) * 32 + is * 16 + l;
                    if g < 2 {
                        ql[qpos] |= nib;
                    } else {
                        ql[qpos] |= nib << 4;
                    }
                    let hpos = chunk * 32 + is * 16 + l;
                    qh[hpos] |= hi2 << (2 * g);
                }
                // block_q6_K field order: ql[128], qh[64], scales[16], d —
                // d is the LAST field (offset 208), not the first
                w6b.extend_from_slice(&ql);
                w6b.extend_from_slice(&qh);
                w6b.extend_from_slice(&sc);
                w6b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            }
        }
        assert_eq!(w6b.len(), od * nbe * 210);

        // mirror quantize_q8_0_pad40 (padded 40B blocks, payload at offset 4)
        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w6dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));

        // ── padded 224B registration, direct decode call (gate bypassed) ──
        let mut w6t = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6t.name = "mw6kp".to_string();
        cb.state.register_weight_q6k_padded("mw6kp", &w6b, od, id_);
        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.state.q6_k_decode_mmvq(
            cb.state.get_weight_ptr("mw6kp").unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(ob).unwrap(),
            od,
            id_,
            nt,
            true,
        );
        let got = cb.copy_to_host(ob).unwrap();
        let mut worst = (0f32, 0usize);
        for i in 0..od * nt {
            let e = (got[i] - want[i]).abs();
            if e > worst.0 {
                worst = (e, i);
            }
        }
        assert!(
            worst.0 <= scale * 1e-2,
            "q6_k mmvq padded err {} > {} at {}",
            worst.0,
            scale * 1e-2,
            worst.1
        );

        // ── raw 210B stride via a direct decode call (gate bypassed) ──
        let mut w6tr = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6tr.name = "mw6kr".to_string();
        cb.state.register_weight("mw6kr", &w6b);
        let obr = cb.alloc_buffer(od * nt);
        cb.state.q6_k_decode_mmvq(
            cb.state.get_weight_ptr("mw6kr").unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(obr).unwrap(),
            od,
            id_,
            nt,
            false,
        );
        let gotr = cb.copy_to_host(obr).unwrap();
        for i in 0..od * nt {
            assert!(
                (gotr[i] - want[i]).abs() <= scale * 1e-2,
                "q6_k mmvq raw stride mismatch at {i}"
            );
        }
        // ── real weights (Q6_K tensors of the decode path) ──
        // 7B blk.0.ffn_down (q4_k_m, od=3584 x id=18944 — the shape that
        // passes the od*id >= 24M dispatch gate, so this also covers the
        // graph-dispatch wiring), 7B blk.0.attn_v and 0.5B blk.0.ffn_down
        // (both BELOW the gate — direct calls, since runtime keeps those
        // on the f32 kernel). Dumped via the ignored helpers; skipped when
        // a file is absent so the suite stays hermetic. The padded
        // registration repacks the raw 210B bytes, matching the loader.
        let real_shapes: [(&str, usize, usize, bool); 3] = [
            ("real_blk_0_ffn_down_weight.bin", 18944, 3584, true),
            ("real_blk_0_attn_v_weight.bin", 3584, 512, false),
            ("real05_blk_0_ffn_down_weight.bin", 4864, 896, false),
        ];
        for (file, id2, od2, via_graph) in real_shapes {
            let Ok(wb) = std::fs::read(format!("/tmp/minfer_phase7/{file}")) else {
                println!("real q6_k [{file}]: absent — skipped");
                break;
            };
            assert_eq!(wb.len(), od2 * (id2 / 256) * 210);
            // fresh activations at the real width
            let mut rng2 = 777u64;
            let xs2: Vec<f32> = (0..id2)
                .map(|_| {
                    rng2 = rng2
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = ((rng2 >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                    let mag = if (rng2 >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                    (u as f32) * mag
                })
                .collect();
            let mut x82 = vec![0u8; (id2 / 32) * 40];
            for b in 0..id2 / 32 {
                let mut am = 0f32;
                for j in 0..32 {
                    am = am.max(xs2[b * 32 + j].abs());
                }
                let dd = am / 127.0;
                let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                x82[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                for j in 0..32 {
                    let q = (xs2[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                    x82[b * 40 + 4 + j] = q as u8;
                }
            }
            let dq82 = |i: usize| -> f32 {
                half::f16::from_le_bytes([x82[(i / 32) * 40], x82[(i / 32) * 40 + 1]]).to_f32()
                    * (x82[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
            };
            let mut w2t = Tensor::from_data(
                TensorType::Q6_K,
                &[id2 as i64, od2 as i64, 1, 1],
                wb.clone(),
            );
            w2t.name = "realq6k".to_string();
            cb.state
                .register_weight_q6k_padded("realq6k", &wb, od2, id2);
            let mut b2 = GraphBuilder::new();
            let x2 = b2.input("x2", [id2, 1, 1, 1], DType::F32);
            let m2 = b2.matmul(x2, &w2t, None);
            b2.output(m2);
            let g2 = b2.build();
            let xb2 = cb.alloc_buffer(id2);
            cb.write_host(xb2, &xs2).unwrap();
            let ob2 = cb.alloc_buffer(od2);
            if via_graph {
                // shape above the od*id gate — dispatch selects the MMVQ
                cb.execute_node(&g2.nodes[m2], &[xb2], ob2, None).unwrap();
            } else {
                // below the gate — runtime dispatch would keep the f32
                // kernel; call the decode path directly for kernel parity
                cb.state.q6_k_decode_mmvq(
                    cb.state.get_weight_ptr("realq6k").unwrap(),
                    cb.ptr_of(xb2).unwrap(),
                    cb.ptr_of(ob2).unwrap(),
                    od2,
                    id2,
                    1,
                    true,
                );
            }
            let got2 = cb.copy_to_host(ob2).unwrap();

            let mut want2 = vec![0f32; od2];
            for r in 0..od2 {
                let mut acc = 0f32;
                for ib in 0..id2 / 256 {
                    let blk = &wb[(r * (id2 / 256) + ib) * 210..];
                    let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                    for e in 0..256 {
                        let s = e / 16;
                        let l = e % 16;
                        let chunk = s / 8;
                        let g = (s / 2) % 4;
                        let is = s % 2;
                        let qlb = blk[chunk * 64 + (g % 2) * 32 + is * 16 + l];
                        let nib = if g < 2 { qlb & 0xF } else { qlb >> 4 };
                        let qhb = blk[128 + chunk * 32 + is * 16 + l];
                        let hi = (qhb >> (2 * g)) & 3;
                        acc += d
                            * (blk[192 + s] as i8 as f32)
                            * ((nib | (hi << 4)) as f32 - 32.0)
                            * dq82(ib * 256 + e);
                    }
                }
                want2[r] = acc;
            }
            let scale2 = want2.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            let mut worst = (0f32, 0usize);
            for r in 0..od2 {
                let e = (got2[r] - want2[r]).abs();
                if e > worst.0 {
                    worst = (e, r);
                }
            }
            println!(
                "real q6_k [{file}]: max err {:.4} at row {} (got {:.4} want {:.4}, scale {scale2:.3})",
                worst.0, worst.1, got2[worst.1], want2[worst.1]
            );
            assert!(
                worst.0 <= scale2 * 1e-2,
                "real q6_k mmvq err {} > {}",
                worst.0,
                scale2 * 1e-2
            );
        }
    }

    /// 8e follow-up: decode (nt == 1) Q5_K dispatches to the MMVQ structure
    /// kernel (q4_K shape with the q5 high-bit plane folded in). Scale bytes
    /// cover the full 0..255 range so the get_scale_min_k4 high-bit splicing
    /// (bits 6..7 of bytes 0..3 feeding the upper scales — the 8e lesson that
    /// %63 data never reaches those paths) is exercised. No local model
    /// carries Q5_K tensors (the "q5_k_m"-branded 0.5B GGUF actually stores
    /// Q5_1/Q8_0/Q6_K), so parity rests on this synthetic; real-weight
    /// parity lands when a model with Q5_K tensors is used.
    #[test]
    fn cuda_q5k_decode_mmvq_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // serialize against the other mmvq parity tests (shared scratch)
        let _guard = crate::cuda::CudaState::model_load_guard();
        let (od, id_, nt) = (128usize, 2176usize, 1usize); // partial tail super-block
        let mut rng_state = 54321u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

        fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }

        let nbe = (id_ + 255) / 256;
        let mut w5b = Vec::new();
        let mut w5dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                w5b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                w5b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    // full byte range — exercises the scale packing hi bits
                    scb[j] = ((r * 131 + j * 29 + ib * 7) % 256) as u8;
                }
                w5b.extend_from_slice(&scb);
                let mut qh = [0u8; 32];
                let mut qs = [0u8; 128];
                let n_elem = 256usize.min(id_ - ib * 256);
                for e in 0..n_elem {
                    let s = e / 32;
                    let l = e % 32;
                    let u = ((r * 13 + e * 7 + ib * 5) % 32) as u8;
                    let (s8, m8) = k4_scale(&scb, s);
                    w5dq[r * id_ + ib * 256 + e] = u as f32 * d * s8 as f32 - dmin * m8 as f32;
                    if s % 2 == 0 {
                        qs[(s >> 1) * 32 + l] |= u & 0xF;
                    } else {
                        qs[(s >> 1) * 32 + l] |= (u & 0xF) << 4;
                    }
                    qh[l] |= ((u >> 4) & 1) << s;
                }
                w5b.extend_from_slice(&qh);
                w5b.extend_from_slice(&qs);
            }
        }
        assert_eq!(w5b.len(), od * nbe * 176);

        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w5dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));

        let mut w5t = Tensor::from_data(
            TensorType::Q5_K,
            &[id_ as i64, od as i64, 1, 1],
            w5b.clone(),
        );
        w5t.name = "mw5k".to_string();
        cb.state.register_weight("mw5k", &w5b);
        // direct decode call: the synthetic shape (od=128) is far below the
        // measured od*id >= 24M dispatch gate (small shapes stay on the f32
        // kernel at runtime); no local model carries Q5_K tensors, so the
        // gate wiring for q5_K is covered by this kernel parity + the shared
        // dispatch code path with q6_K.
        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.state.q5_k_decode_mmvq(
            cb.state.get_weight_ptr("mw5k").unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(ob).unwrap(),
            od,
            id_,
            nt,
        );
        let got = cb.copy_to_host(ob).unwrap();
        for i in 0..od * nt {
            assert!(
                (got[i] - want[i]).abs() <= scale * 1e-2,
                "q5_k mmvq mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }

    #[test]
    fn cuda_fused_ffn_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (nf, id_) = (8usize, 512usize); // concat od = 16, decode nt = 1
        let xs: Vec<f32> = (0..id_)
            .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
            .collect();

        // ── q4_K gate/up weights (144-byte super-blocks, llama layout) ──
        // get_scale_min_k4 (llama.cpp Q4_K scale packing): the second half
        // of the 8 scale/min pairs is spliced across the 12 scale bytes.
        fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }
        fn build_q4k(seed: u64, rows: usize, id: usize, bytes: &mut Vec<u8>, dq: &mut Vec<f32>) {
            for r in 0..rows {
                for ib in 0..id / 256 {
                    let d = 0.031f32 + 0.005 * ((seed + (r * 7 + ib * 3) as u64) % 5) as f32;
                    let dmin = 0.002f32 + 0.001 * ((seed + (r * 3 + ib) as u64) % 4) as f32;
                    bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    bytes.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                    let mut scb = [0u8; 12];
                    for j in 0..12 {
                        scb[j] = ((seed as usize + r * 31 + j * 17 + ib * 5) % 63) as u8;
                    }
                    bytes.extend_from_slice(&scb);
                    let mut qs = [0u8; 128];
                    for j in 0..128 {
                        let lo = ((seed as usize + r * 13 + j * 7 + ib * 3) % 15) as u8;
                        let hi = ((seed as usize + r * 5 + j * 11 + ib * 2) % 15) as u8;
                        qs[j] = lo | (hi << 4);
                    }
                    bytes.extend_from_slice(&qs);
                    for j in 0..4 {
                        let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                        let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                        for l in 0..32 {
                            let b = qs[j * 32 + l];
                            let base = r * id + ib * 256 + j * 64;
                            dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                            dq[base + 32 + l] =
                                (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                        }
                    }
                }
            }
        }

        // ── q6_K gate/up weights (210-byte super-blocks, llama layout) ──
        fn build_q6k(seed: u64, rows: usize, id: usize, bytes: &mut Vec<u8>, dq: &mut Vec<f32>) {
            for r in 0..rows {
                for ib in 0..id / 256 {
                    let d = 0.027f32 + 0.004 * ((seed + (r * 11 + ib * 7) as u64) % 6) as f32;
                    let mut ql = [0u8; 128];
                    let mut qh = [0u8; 64];
                    let mut sc = [0i8; 16];
                    for i in 0..128 {
                        ql[i] = ((seed as usize + r * 29 + i * 7 + ib * 3) % 255) as u8;
                    }
                    for i in 0..64 {
                        qh[i] = ((seed as usize + r * 17 + i * 13 + ib * 11) % 255) as u8;
                    }
                    for i in 0..16 {
                        sc[i] = (((seed as usize + r * 5 + i * 3 + ib) % 15) as i8) - 7;
                    }
                    bytes.extend_from_slice(&ql);
                    bytes.extend_from_slice(&qh);
                    bytes.extend(sc.iter().map(|&x| x as u8));
                    bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    // reference dequant (llama.cpp Q6_K layout, same as
                    // the kquant test): four interleaved 32-element groups
                    // per 128-element half.
                    for n in 0..2usize {
                        let qlh = &ql[n * 64..n * 64 + 64];
                        let qhh = &qh[n * 32..n * 32 + 32];
                        for l in 0..32usize {
                            let is = l / 16;
                            let q1 =
                                ((qlh[l] & 0xF) as i32 | (((qhh[l] >> 0) as i32 & 3) << 4)) - 32;
                            let q2 = ((qlh[l + 32] & 0xF) as i32
                                | (((qhh[l] >> 2) as i32 & 3) << 4))
                                - 32;
                            let q3 =
                                ((qlh[l] >> 4) as i32 | (((qhh[l] >> 4) as i32 & 3) << 4)) - 32;
                            let q4 = ((qlh[l + 32] >> 4) as i32
                                | (((qhh[l] >> 6) as i32 & 3) << 4))
                                - 32;
                            let base = r * id + ib * 256 + n * 128;
                            dq[base + l] = d * sc[n * 8 + is] as f32 * q1 as f32;
                            dq[base + l + 32] = d * sc[n * 8 + is + 2] as f32 * q2 as f32;
                            dq[base + l + 64] = d * sc[n * 8 + is + 4] as f32 * q3 as f32;
                            dq[base + l + 96] = d * sc[n * 8 + is + 6] as f32 * q4 as f32;
                        }
                    }
                }
            }
        }

        let mut g4b = Vec::new();
        let mut g4dq = vec![0f32; nf * id_];
        build_q4k(1, nf, id_, &mut g4b, &mut g4dq);
        let mut u4b = Vec::new();
        let mut u4dq = vec![0f32; nf * id_];
        build_q4k(101, nf, id_, &mut u4b, &mut u4dq);
        let mut g6b = Vec::new();
        let mut g6dq = vec![0f32; nf * id_];
        build_q6k(7, nf, id_, &mut g6b, &mut g6dq);
        let mut u6b = Vec::new();
        let mut u6dq = vec![0f32; nf * id_];
        build_q6k(207, nf, id_, &mut u6b, &mut u6dq);

        // concat rows: gate rows then up rows (concat_rows semantics)
        let gu4: Vec<u8> = g4b.iter().chain(u4b.iter()).copied().collect();
        let gu6: Vec<u8> = g6b.iter().chain(u6b.iter()).copied().collect();
        cb.state.register_weight("mgu4", &gu4);
        // q6_K concat goes through the padded repack (7e② layout)
        cb.state
            .register_weight_q6k_padded("mgu6", &gu6, 2 * nf, id_);
        assert!(cb.state.is_weight_padded("mgu6"));

        let (xb, ogu4, ogu6) = (
            cb.alloc_buffer(id_),
            cb.alloc_buffer(2 * nf),
            cb.alloc_buffer(2 * nf),
        );
        cb.write_host(xb, &xs).unwrap();

        for (ttype, wname, ogu, gdq, udq) in [
            (crate::tensor::TensorType::Q4_K, "mgu4", ogu4, &g4dq, &u4dq),
            (crate::tensor::TensorType::Q6_K, "mgu6", ogu6, &g6dq, &u6dq),
        ] {
            let mut b = crate::graph::builder::GraphBuilder::new();
            let x = b.input("x", [id_, 1, 1, 1], crate::graph::DType::F32);
            let gu = b.fused_ffn(
                x,
                crate::graph::ops::FusedFfnMeta {
                    gu_weight: wname.to_string(),
                    weight_ttype: ttype,
                    in_dim: id_,
                    nf,
                },
            );
            b.output(gu);
            let g = b.build();
            cb.execute_node(&g.nodes[gu], &[xb], ogu, None).unwrap();

            // host reference: silu(gate·x) × (up·x)
            let got = cb.copy_to_host(ogu).unwrap();
            let mut want = vec![0f32; nf];
            for r in 0..nf {
                let mut ag = 0f32;
                let mut au = 0f32;
                for i in 0..id_ {
                    ag += gdq[r * id_ + i] * xs[i];
                    au += udq[r * id_ + i] * xs[i];
                }
                let s = ag / (1.0f32 + (-ag).exp());
                want[r] = s * au;
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close(
                &format!("{ttype:?} fused ffn"),
                &got[..nf],
                &want,
                scale * 1e-3,
            );
        }
    }

    /// 7e③: embedding / row-gather parity. Device embed kernels (one per
    /// supported weight type, incl. the padded Q6_K layout) must match
    /// `kernel::embed_tokens` — the CPU path these nodes used before 7e③ —
    /// and the generic f32 gather (G3 tail get_rows) must match a manual
    /// row copy.
    #[test]
    fn cuda_embed_getrows_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (vocab, n_embd, nt) = (6usize, 512usize, 3usize); // 2 super-blocks/row
        let ids: Vec<u32> = vec![0, 5, 2];
        let ids_f32: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();

        // ── build one tensor per supported type (rows = vocab) ──
        // f32
        let wf: Vec<f32> = (0..vocab * n_embd)
            .map(|i| (((i as u64) * 2654435761 % 1009) as f32) / 504.0 - 1.0)
            .collect();
        let wf_bytes: Vec<u8> = wf.iter().flat_map(|f| f.to_le_bytes()).collect();
        let mut tf = Tensor::from_data(
            TensorType::F32,
            &[n_embd as i64, vocab as i64, 1, 1],
            wf_bytes.clone(),
        );
        tf.name = "ewf32".to_string();

        // q8_0 (34B blocks: f16 d + 32 i8)
        let mut w8 = Vec::new();
        for r in 0..vocab {
            for ib in 0..n_embd / 32 {
                let d = 0.02f32 + 0.003 * ((r * 5 + ib) % 7) as f32;
                w8.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for i in 0..32 {
                    w8.push((((r * 37 + ib * 17 + i * 3) % 255) as i8 as u8).wrapping_add(0));
                }
            }
        }
        let mut t8 = Tensor::from_data(
            TensorType::Q8_0,
            &[n_embd as i64, vocab as i64, 1, 1],
            w8.clone(),
        );
        t8.name = "ewq8".to_string();

        // q4_0 (18B blocks: f16 d + 16 nibble bytes; elem j = LOW of byte j)
        let mut w40 = Vec::new();
        for r in 0..vocab {
            for ib in 0..n_embd / 32 {
                let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
                w40.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for i in 0..16 {
                    let lo = ((r * 11 + ib * 7 + i * 3) % 15) as u8;
                    let hi = ((r * 7 + ib * 5 + i) % 15) as u8;
                    w40.push(lo | (hi << 4));
                }
            }
        }
        let mut t40 = Tensor::from_data(
            TensorType::Q4_0,
            &[n_embd as i64, vocab as i64, 1, 1],
            w40.clone(),
        );
        t40.name = "ewq40".to_string();

        // q4_k (144B super-blocks) — same generator scheme as the matmul test
        let mut w4k = Vec::new();
        for r in 0..vocab {
            for ib in 0..n_embd / 256 {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                w4k.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                w4k.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                for j in 0..12 {
                    w4k.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
                }
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    w4k.push(lo | (hi << 4));
                }
            }
        }
        let mut t4k = Tensor::from_data(
            TensorType::Q4_K,
            &[n_embd as i64, vocab as i64, 1, 1],
            w4k.clone(),
        );
        t4k.name = "ewq4k".to_string();

        // q6_k (210B raw; also registered padded)
        let mut w6k = Vec::new();
        for r in 0..vocab {
            for ib in 0..n_embd / 256 {
                let d = 0.027f32 + 0.004 * ((r * 11 + ib * 7) % 6) as f32;
                for i in 0..128 {
                    w6k.push(((r * 29 + i * 7 + ib * 3) % 255) as u8);
                }
                for i in 0..64 {
                    w6k.push(((r * 17 + i * 13 + ib * 11) % 255) as u8);
                }
                for i in 0..16 {
                    w6k.push(((((r * 5 + i * 3 + ib) % 15) as i8) - 7) as u8);
                }
                w6k.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            }
        }
        let mut t6k = Tensor::from_data(
            TensorType::Q6_K,
            &[n_embd as i64, vocab as i64, 1, 1],
            w6k.clone(),
        );
        t6k.name = "ewq6k".to_string();
        let mut t6kp = Tensor::from_data(
            TensorType::Q6_K,
            &[n_embd as i64, vocab as i64, 1, 1],
            w6k.clone(),
        );
        t6kp.name = "ewq6kp".to_string();

        // ── register + build one graph with an embed node per type ──
        cb.state.register_weight("ewf32", &wf_bytes);
        cb.state.register_weight("ewq8", &w8);
        cb.state.register_weight("ewq40", &w40);
        cb.state.register_weight("ewq4k", &w4k);
        cb.state.register_weight("ewq6k", &w6k);
        cb.state
            .register_weight_q6k_padded("ewq6kp", &w6k, vocab, n_embd);
        assert!(cb.state.is_weight_padded("ewq6kp"));

        let mut b = GraphBuilder::new();
        let ids_in = b.input("ids", [nt, 1, 1, 1], DType::F32);
        let e_f32 = b.embedding(ids_in, &tf);
        let e_q8 = b.embedding(ids_in, &t8);
        let e_q40 = b.embedding(ids_in, &t40);
        let e_q4k = b.embedding(ids_in, &t4k);
        let e_q6k = b.embedding(ids_in, &t6k);
        let e_q6kp = b.embedding(ids_in, &t6kp);
        // ── 7e③ model-shape q4_0 case (0.5B): n_embd=896 (nb=28 blocks),
        // large ids — the exact shape that E2E first exercised ──
        let (mv, me) = (10000usize, 896usize);
        let mids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
        let mut mw = Vec::new();
        for r in 0..mv {
            for ib in 0..me / 32 {
                let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
                mw.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for i in 0..16 {
                    let lo = ((r * 11 + ib * 7 + i * 3) % 15) as u8;
                    let hi = ((r * 7 + ib * 5 + i) % 15) as u8;
                    mw.push(lo | (hi << 4));
                }
            }
        }
        let mut mt = Tensor::from_data(TensorType::Q4_0, &[me as i64, mv as i64, 1, 1], mw.clone());
        mt.name = "ewq40m".to_string();
        cb.state.register_weight("ewq40m", &mw);
        let mids_f32: Vec<f32> = mids.iter().map(|&i| f32::from_bits(i)).collect();
        let midb = cb.alloc_buffer(mids.len());
        cb.write_host(midb, &mids_f32).unwrap();
        let mout = cb.alloc_buffer(me * mids.len());
        let mut mb = GraphBuilder::new();
        let mi = mb.input("mids", [mids.len(), 1, 1, 1], DType::F32);
        let me_node = mb.embedding(mi, &mt);
        mb.output(me_node);
        let mg = mb.build();
        cb.execute_node(&mg.nodes[me_node], &[midb], mout, None)
            .unwrap();
        {
            let got = cb.copy_to_host(mout).unwrap();
            let mut want = vec![0f32; me * mids.len()];
            crate::kernel::embed_tokens(&mids, &mt, &mut want, me);
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close("embed q4_0 model-shape", &got, &want, scale * 2e-3);
        }

        // generic gather (G3 tail): x[ids[t]] — the source has vocab rows so
        // every id is in range
        let xin = b.input("x", [n_embd, vocab, 1, 1], DType::F32);
        let gr = b.get_rows(xin, ids_in, [n_embd, nt, 1, 1]);
        for n in [e_f32, e_q8, e_q40, e_q4k, e_q6k, e_q6kp, gr] {
            b.output(n);
        }
        let g = b.build();

        let idsb = cb.alloc_buffer(nt);
        cb.write_host(idsb, &ids_f32).unwrap();
        let xvals: Vec<f32> = (0..n_embd * vocab)
            .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
            .collect();
        let xb = cb.alloc_buffer(n_embd * vocab);
        cb.write_host(xb, &xvals).unwrap();

        let mut outs = Vec::new();
        for node in [e_f32, e_q8, e_q40, e_q4k, e_q6k, e_q6kp] {
            let out = cb.alloc_buffer(n_embd * nt);
            cb.execute_node(&g.nodes[node], &[idsb], out, None).unwrap();
            outs.push(out);
        }
        let grb = cb.alloc_buffer(n_embd * nt);
        cb.execute_node(&g.nodes[gr], &[xb, idsb], grb, None)
            .unwrap();

        // ── references ──
        let names = ["f32", "q8_0", "q4_0", "q4_k", "q6_k", "q6_k padded"];
        let tensors = [&tf, &t8, &t40, &t4k, &t6k, &t6kp];
        for ((name, t), &ob) in names.iter().zip(tensors).zip(outs.iter()) {
            let got = cb.copy_to_host(ob).unwrap();
            let mut want = vec![0f32; n_embd * nt];
            if t.ttype == TensorType::F32 {
                // embed_tokens handles the quantized types; f32 is a row copy
                for (ti, &id) in ids.iter().enumerate() {
                    let src = (id as usize) * n_embd;
                    want[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(&wf[src..src + n_embd]);
                }
            } else {
                crate::kernel::embed_tokens(&ids, t, &mut want, n_embd);
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close(&format!("embed {name}"), &got, &want, scale * 2e-3);
        }
        let got = cb.copy_to_host(grb).unwrap();
        for t in 0..nt {
            let id = ids[t] as usize;
            for i in 0..n_embd {
                let want = xvals[id * n_embd + i];
                assert!(
                    (got[t * n_embd + i] - want).abs() <= 1e-6 * (1.0 + want.abs()),
                    "gather [{t},{i}]: got {} want {want}",
                    got[t * n_embd + i]
                );
            }
        }
    }

    #[test]
    fn cuda_rope_kv_attn_roundtrip() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
        let nkt = nk_h * hd;
        let (nt, n_ctx) = (3usize, 32usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let pos: Vec<usize> = vec![1, 4, 9]; // sparse, exercises the scatter

        let mut b = GraphBuilder::new();
        let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
        let p = b.input("positions", [nt, 1, 1, 1], DType::I32);
        let store = b.kvcache_store(0, k, v, p, n_ctx);
        let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
        let qr = b.rope(
            q,
            p,
            RopeStyle::NonInterleaved,
            RoPEMeta {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_head: nh,
                hd,
            },
        );
        let at = b.attn(
            qr,
            load,
            p,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b.output(at);
        let g = b.build();

        let (xb_q, xb_k, xb_v) = (
            cb.alloc_buffer(nh * hd * nt),
            cb.alloc_buffer(nkt * nt),
            cb.alloc_buffer(nkt * nt),
        );
        let xb_p = cb.alloc_buffer(nt);
        let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
        let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

        let qs: Vec<f32> = (0..nh * hd * nt)
            .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
            .collect();
        let ks: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let vs: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let pb: Vec<f32> = pos.iter().map(|&pp| f32::from_bits(pp as u32)).collect();
        cb.write_host(xb_q, &qs).unwrap();
        cb.write_host(xb_k, &ks).unwrap();
        cb.write_host(xb_v, &vs).unwrap();
        cb.write_host(xb_p, &pb).unwrap();
        // Zero the KV regions first: rows the store never touches stay
        // uninitialized in a recycled cudaMalloc block, and the reference
        // below treats unwritten rows as zeros (deterministic vs pool state).
        cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
        cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

        cb.execute_node(
            &g.nodes[store],
            &[xb_k, xb_v, xb_p],
            kreg,
            Some((kreg, vreg)),
        )
        .unwrap();
        cb.execute_node(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
            .unwrap();
        cb.execute_node(
            &g.nodes[at],
            &[ob_qr, kreg, xb_p],
            ob_at,
            Some((kreg, vreg)),
        )
        .unwrap();

        // a) stored K rows are bit-exact at the scattered positions
        let kback = cb.copy_to_host(kreg).unwrap();
        for (t, &pp) in pos.iter().enumerate() {
            assert_eq!(
                &kback[pp * nkt..(pp + 1) * nkt],
                &ks[t * nkt..(t + 1) * nkt],
                "K row {pp}"
            );
        }
        // b) RoPE vs cpu_rope (also covers the non-alias D2D staging path)
        let qgot = cb.copy_to_host(ob_qr).unwrap();
        let mut qref = qs.clone();
        crate::graph::cpu_backend::cpu_rope(
            &mut qref,
            &pos,
            nh,
            hd,
            10000.0,
            1.0,
            RopeStyle::NonInterleaved,
        );
        assert_close("rope", &qgot, &qref, 1e-4);
        // c) GQA attention vs cpu_gqa_attn over the scattered KV regions
        let mut kfull = vec![0f32; nkt * n_ctx];
        let mut vfull = vec![0f32; nkt * n_ctx];
        for (t, &pp) in pos.iter().enumerate() {
            kfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&ks[t * nkt..(t + 1) * nkt]);
            vfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&vs[t * nkt..(t + 1) * nkt]);
        }
        let nkv = pos.iter().copied().max().unwrap() + 1;
        let mut aref = vec![0f32; nh * hd * nt];
        crate::graph::cpu_backend::cpu_gqa_attn(
            &qref, &kfull, &vfull, &pos, nt, nkv, nh, nk_h, hd, hd, nkt, &mut aref, scale,
        )
        .unwrap();
        let agot = cb.copy_to_host(ob_at).unwrap();
        assert_close("gqa_attn", &agot, &aref, 1e-4);
    }

    // 8b: f16 KV cache — the store rounds K/V to half and the attention
    // kernel reads half4. The reference builds its KV from the SAME
    // half-rounded values so the comparison isolates the kernel from the
    // f16 quantization noise (tolerance stays tight).
    #[test]
    fn cuda_kv_f16_roundtrip_attn() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        cb.set_kv_f16_for_test(true);
        let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
        let nkt = nk_h * hd;
        let (nt, n_ctx) = (3usize, 32usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let pos: Vec<usize> = vec![1, 4, 9];

        let mut b = GraphBuilder::new();
        let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
        let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
        let store = b.kvcache_store(0, k, v, pp, n_ctx);
        let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
        let qr = b.rope(
            q,
            pp,
            RopeStyle::NonInterleaved,
            RoPEMeta {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_head: nh,
                hd,
            },
        );
        let at = b.attn(
            qr,
            load,
            pp,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b.output(at);
        let g = b.build();

        let (xb_q, xb_k, xb_v) = (
            cb.alloc_buffer(nh * hd * nt),
            cb.alloc_buffer(nkt * nt),
            cb.alloc_buffer(nkt * nt),
        );
        let xb_p = cb.alloc_buffer(nt);
        let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
        let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

        let qs: Vec<f32> = (0..nh * hd * nt)
            .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
            .collect();
        let ks: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let vs: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
        // the reference KV: what the f16 store actually persists (f32→f16→f32)
        let to_half = |x: &[f32]| -> Vec<f32> {
            x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
        };
        let ks_h = to_half(&ks);
        let vs_h = to_half(&vs);
        cb.write_host(xb_q, &qs).unwrap();
        cb.write_host(xb_k, &ks).unwrap();
        cb.write_host(xb_v, &vs).unwrap();
        cb.write_host(xb_p, &pb).unwrap();
        // zero the regions (unwritten rows read as f16 zeros)
        cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
        cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

        cb.execute_node(
            &g.nodes[store],
            &[xb_k, xb_v, xb_p],
            kreg,
            Some((kreg, vreg)),
        )
        .unwrap();
        cb.execute_node(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
            .unwrap();
        cb.execute_node(
            &g.nodes[at],
            &[ob_qr, kreg, xb_p],
            ob_at,
            Some((kreg, vreg)),
        )
        .unwrap();

        // a) stored K rows equal the half-rounded values at the scatter positions
        let kback_f32 = cb.copy_to_host(kreg).unwrap();
        // reinterpret the region as f16 pairs (store wrote 2 bytes/elem)
        let kbytes: Vec<u8> = kback_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
        for (t, &p) in pos.iter().enumerate() {
            for j in 0..nkt {
                let byte_off = (p * nkt + j) * 2;
                let got = half::f16::from_le_bytes([kbytes[byte_off], kbytes[byte_off + 1]]);
                assert!(
                    (got.to_f32() - ks_h[t * nkt + j]).abs() < 1e-6,
                    "f16 K row {p}[{j}]"
                );
            }
        }
        // b) attention vs cpu_gqa_attn over the half-rounded KV
        let qgot = cb.copy_to_host(ob_qr).unwrap();
        let mut qref = qs.clone();
        crate::graph::cpu_backend::cpu_rope(
            &mut qref,
            &pos,
            nh,
            hd,
            10000.0,
            1.0,
            RopeStyle::NonInterleaved,
        );
        assert_close("rope(f16 kv)", &qgot, &qref, 1e-4);
        let mut kfull = vec![0f32; nkt * n_ctx];
        let mut vfull = vec![0f32; nkt * n_ctx];
        for (t, &p) in pos.iter().enumerate() {
            kfull[p * nkt..(p + 1) * nkt].copy_from_slice(&ks_h[t * nkt..(t + 1) * nkt]);
            vfull[p * nkt..(p + 1) * nkt].copy_from_slice(&vs_h[t * nkt..(t + 1) * nkt]);
        }
        let nkv = pos.iter().copied().max().unwrap() + 1;
        let mut aref = vec![0f32; nh * hd * nt];
        crate::graph::cpu_backend::cpu_gqa_attn(
            &qref, &kfull, &vfull, &pos, nt, nkv, nh, nk_h, hd, hd, nkt, &mut aref, scale,
        )
        .unwrap();
        let agot = cb.copy_to_host(ob_at).unwrap();
        assert_close("gqa_attn(f16 kv)", &agot, &aref, 1e-4);
    }

    // 8n: prefill attention (nt >= 64, hd == 128) routes through the
    // FA-style tiled kernel (wmma QK^T, online softmax, per-thread register
    // O accumulator). Reference: cpu_gqa_attn over the f16-rounded KV — the
    // kernel reads the same f16 cache; its q and probs carry f16 rounding,
    // measured ~1.4e-4 on the standalone harness, so 5e-3 leaves headroom.
    #[test]
    #[test]
    fn cuda_prefill_fused_b_bitparity() {
        // 8p: the fused dequant-in-GEMM path must be BIT-identical to the
        // legacy dequant-to-f16 two-pass path (same __float2half rounding,
        // same wmma accumulate). All 8 types x {1, 2} super-blocks; the
        // legacy path is reference-validated by cuda_prefill_f16_gemm_parity.
        let _guard = crate::cuda::CudaState::model_load_guard();
        crate::cuda::CudaState::init();
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let state = cb.state;
        let mut seed = 0x9E3779B9u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        for (od, id, nt) in [(70usize, 256usize, 33usize), (70usize, 512usize, 70usize)] {
            let nsp = id / 256;
            let xs: Vec<f32> = (0..id * nt)
                .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
                .collect();
            let mut mk =
                |nbytes: usize| -> Vec<u8> { (0..nbytes).map(|_| (rnd() & 0xFF) as u8).collect() };
            let xb = cb.alloc_buffer(id * nt);
            let out = cb.alloc_buffer(od * nt);
            cb.write_host(xb, &xs).unwrap();
            let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());
            let dbytes = |v: f32| half::f16::from_f32(v).to_le_bytes();

            // benign d (and m for the min-carrying types) per block
            let mut wq80 = mk(od * (id / 32) * 34);
            let mut wq40 = mk(od * (id / 32) * 18);
            let mut wq41 = mk(od * (id / 32) * 20);
            let mut wq50 = mk(od * (id / 32) * 22);
            let mut wq51 = mk(od * (id / 32) * 24);
            for g in 0..od * (id / 32) {
                let set = |w: &mut [u8], base: usize, off: usize, v: f32| {
                    let db = dbytes(v);
                    w[base + off] = db[0];
                    w[base + off + 1] = db[1];
                };
                let b32 = g * 34;
                set(&mut wq80, b32, 0, 0.01);
                let b18 = g * 18;
                set(&mut wq40, b18, 0, 0.05);
                let b20 = g * 20;
                set(&mut wq41, b20, 0, 0.05);
                set(&mut wq41, b20, 2, 0.1);
                let b22 = g * 22;
                set(&mut wq50, b22, 0, 0.05);
                let b24 = g * 24;
                set(&mut wq51, b24, 0, 0.05);
                set(&mut wq51, b24, 2, 0.1);
            }
            let mut wq4k = mk(od * nsp * 144);
            let mut wq5k = mk(od * nsp * 176);
            let mut wq6k = mk(od * nsp * 210);
            for r in 0..od {
                for sp in 0..nsp {
                    let base4 = (r * nsp + sp) * 144;
                    wq4k[base4..base4 + 2].copy_from_slice(&dbytes(0.01));
                    wq4k[base4 + 2..base4 + 4].copy_from_slice(&dbytes(0.005));
                    let base5 = (r * nsp + sp) * 176;
                    wq5k[base5..base5 + 2].copy_from_slice(&dbytes(0.01));
                    wq5k[base5 + 2..base5 + 4].copy_from_slice(&dbytes(0.005));
                    let base6 = (r * nsp + sp) * 210;
                    wq6k[base6 + 208..base6 + 210].copy_from_slice(&dbytes(0.01));
                }
            }

            state.register_weight("bp_w80", &wq80);
            state.register_weight("bp_w40", &wq40);
            state.register_weight("bp_w41", &wq41);
            state.register_weight("bp_w50", &wq50);
            state.register_weight("bp_w51", &wq51);
            state.register_weight("bp_w4k", &wq4k);
            state.register_weight("bp_w5k", &wq5k);
            state.register_weight("bp_w6k_raw", &wq6k);
            state.register_weight_q6k_padded("bp_w6k_pad", &wq6k, od, id);

            let cases: [(TensorType, &str, bool); 9] = [
                (TensorType::Q8_0, "bp_w80", false),
                (TensorType::Q4_0, "bp_w40", false),
                (TensorType::Q4_1, "bp_w41", false),
                (TensorType::Q5_0, "bp_w50", false),
                (TensorType::Q5_1, "bp_w51", false),
                (TensorType::Q4_K, "bp_w4k", false),
                (TensorType::Q5_K, "bp_w5k", false),
                (TensorType::Q6_K, "bp_w6k_raw", false),
                (TensorType::Q6_K, "bp_w6k_pad", true),
            ];
            for (ttype, name, padded) in cases {
                let wptr = state.get_weight_ptr(name).unwrap();
                state
                    .prefill_gemm_f16_inner(wptr, ttype, xptr, optr, od, id, nt, padded, true)
                    .unwrap();
                cb.synchronize();
                let gotf = cb.copy_to_host(out).unwrap();
                state
                    .prefill_gemm_f16_inner(wptr, ttype, xptr, optr, od, id, nt, padded, false)
                    .unwrap();
                cb.synchronize();
                let gotl = cb.copy_to_host(out).unwrap();
                assert_eq!(gotf.len(), gotl.len(), "{name} len");
                for (i, (a, b)) in gotf.iter().zip(gotl.iter()).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "{name} od={od} id={id} fused vs legacy bit mismatch at [{i}] ({a} vs {b})"
                    );
                }
            }
        }
    }

    fn cuda_fa_prefill_attention_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        cb.set_kv_f16_for_test(true);
        let (nh, nk_h, hd) = (4usize, 2usize, 128usize);
        let nkt = nk_h * hd;
        let (nt, n_ctx) = (100usize, 128usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let pos: Vec<usize> = (0..nt).collect();

        let mut b = GraphBuilder::new();
        let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
        let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
        let _store = b.kvcache_store(0, k, v, pp, n_ctx);
        let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
        let at = b.attn(
            q,
            load,
            pp,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b.output(at);
        let g = b.build();

        let (xb_q, xb_k, xb_v) = (
            cb.alloc_buffer(nh * hd * nt),
            cb.alloc_buffer(nkt * nt),
            cb.alloc_buffer(nkt * nt),
        );
        let xb_p = cb.alloc_buffer(nt);
        let ob_at = cb.alloc_buffer(nh * hd * nt);
        let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

        let qs: Vec<f32> = (0..nh * hd * nt)
            .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
            .collect();
        let ks: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let vs: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
        let to_half = |x: &[f32]| -> Vec<f32> {
            x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
        };
        let ks_h = to_half(&ks);
        let vs_h = to_half(&vs);
        cb.write_host(xb_q, &qs).unwrap();
        cb.write_host(xb_k, &ks).unwrap();
        cb.write_host(xb_v, &vs).unwrap();
        cb.write_host(xb_p, &pb).unwrap();
        cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
        cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

        cb.execute_node(
            &g.nodes[_store],
            &[xb_k, xb_v, xb_p],
            kreg,
            Some((kreg, vreg)),
        )
        .unwrap();
        cb.execute_node(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
            .unwrap();

        let mut kfull = vec![0f32; nkt * n_ctx];
        let mut vfull = vec![0f32; nkt * n_ctx];
        for (t, &p) in pos.iter().enumerate() {
            kfull[p * nkt..(p + 1) * nkt].copy_from_slice(&ks_h[t * nkt..(t + 1) * nkt]);
            vfull[p * nkt..(p + 1) * nkt].copy_from_slice(&vs_h[t * nkt..(t + 1) * nkt]);
        }
        let nkv = pos.iter().copied().max().unwrap() + 1;
        let mut aref = vec![0f32; nh * hd * nt];
        crate::graph::cpu_backend::cpu_gqa_attn(
            &qs, &kfull, &vfull, &pos, nt, nkv, nh, nk_h, hd, hd, nkt, &mut aref, scale,
        )
        .unwrap();
        let agot = cb.copy_to_host(ob_at).unwrap();
        let mut maxe = 0f32;
        for (a, r) in agot.iter().zip(aref.iter()) {
            maxe = maxe.max((a - r).abs());
        }
        println!("fa prefill attention: max err {maxe:.6}");
        assert_close("fa_prefill_f16kv", &agot, &aref, 5e-3);
    }

    // 8c: prefill Q4_0 matmul (nt > 1, id <= 8192) routes through the
    // Q8_0-activation GEMM. The reference builds the SAME Q8_0 activation
    // blocks and uses dot_q4_0_q8_0 — the kernel's exact math — so the
    // tolerance is tight. The nt == 1 call takes the f32-activation path
    // (decode); its reference dequantizes the weights.
    #[test]
    fn cuda_q4_0_prefill_q8_0_gemm_parity() {
        crate::cuda::CudaState::init();
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (od, id, nt) = (32usize, 64usize, 3usize);
        let nb = id / 32;

        // build a Q4_0 weight: d = amax/7, biased nibbles (v + 8)
        let wf: Vec<f32> = (0..od * id)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let mut wq = Vec::with_capacity(od * nb * 18);
        for r in 0..od {
            for b in 0..nb {
                let row = &wf[r * id + b * 32..r * id + (b + 1) * 32];
                let amax = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
                let d = amax / 7.0;
                let di = if d != 0.0 { 1.0 / d } else { 0.0 };
                let dbits = half::f16::from_f32(d).to_le_bytes();
                wq.push(dbits[0]);
                wq.push(dbits[1]);
                for j in 0..16 {
                    let q0 = (row[j] * di).round().clamp(-8.0, 7.0) as i8 + 8;
                    let q1 = (row[j + 16] * di).round().clamp(-8.0, 7.0) as i8 + 8;
                    wq.push(((q1 as u8) << 4) | (q0 as u8));
                }
            }
        }
        let state = cb.state;
        state.register_weight("w40", &wq);
        let wptr = state.get_weight_ptr("w40").unwrap();

        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        // per-token Q8_0 activation blocks (same layout the kernel reads)
        let q8s: Vec<Vec<u8>> = (0..nt)
            .map(|t| crate::quants::quantize_row_q8_0(&xs[t * id..(t + 1) * id]))
            .collect();

        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());

        // nt > 1: Q8_0-activation path
        state
            .matmul_f32_ptr(wptr, TensorType::Q4_0, xptr, optr, od, id, nt)
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        for t in 0..nt {
            for r in 0..od {
                let want =
                    crate::quants::dot_q4_0_q8_0(&wq[r * nb * 18..(r + 1) * nb * 18], &q8s[t]);
                assert!(
                    (got[t * od + r] - want).abs() < 1e-3,
                    "q8_0 path [{t}][{r}] {} vs {want}",
                    got[t * od + r]
                );
            }
        }

        // CPU cross-check: my hand dequant vs dot_q4_0_q8_0 (same wq bytes)
        let q8_tok0 = &q8s[0];
        for r in [0usize, 1, 17] {
            let via_dot =
                crate::quants::dot_q4_0_q8_0(&wq[r * nb * 18..(r + 1) * nb * 18], q8_tok0);
            let mut deq = 0f32;
            for b in 0..nb {
                let blk = &wq[r * nb * 18 + b * 18..r * nb * 18 + (b + 1) * 18];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let q8b = &q8_tok0[b * 34..(b + 1) * 34];
                let d8 = half::f16::from_le_bytes([q8b[0], q8b[1]]).to_f32();
                let mut si = 0i32;
                for j in 0..16 {
                    let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                    let v1 = (blk[2 + j] >> 4) as i32 - 8;
                    si += v0 * q8b[2 + j] as i8 as i32 + v1 * q8b[2 + j + 16] as i8 as i32;
                }
                deq += si as f32 * d * d8;
            }
            let deq_f32 = {
                // dequant-want against raw f32 x (what the f32 kernel reads);
                // weight block b pairs with x[b*32 .. b*32+32]
                let mut acc = 0f32;
                for b in 0..nb {
                    let blk = &wq[r * nb * 18 + b * 18..r * nb * 18 + (b + 1) * 18];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let xb = &xs[b * 32..(b + 1) * 32];
                    for j in 0..16 {
                        let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                        let v1 = (blk[2 + j] >> 4) as i32 - 8;
                        acc += d * (v0 as f32 * xb[j] + v1 as f32 * xb[j + 16]);
                    }
                }
                acc
            };
            assert!(
                (via_dot - deq).abs() < 1e-2 && (via_dot - deq_f32).abs() < 5e-2,
                "crosscheck r={r}: dot_q8 {via_dot} vs dequant-q8 {deq} vs dequant-f32 {deq_f32}"
            );
        }

        // nt == 1: f32-activation path (decode), reference dequantizes weights
        let out1 = cb.alloc_buffer(od);
        let (x1, o1) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out1).unwrap());
        state
            .matmul_f32_ptr(wptr, TensorType::Q4_0, x1, o1, od, id, 1)
            .unwrap();
        cb.synchronize();
        let got1 = cb.copy_to_host(out1).unwrap();
        for r in 0..od {
            let mut want = 0f32;
            for b in 0..nb {
                let blk = &wq[r * nb * 18 + b * 18..r * nb * 18 + (b + 1) * 18];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let xrow = &xs[b * 32..(b + 1) * 32];
                for j in 0..16 {
                    let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                    let v1 = (blk[2 + j] >> 4) as i32 - 8;
                    want += d * (v0 as f32 * xrow[j] + v1 as f32 * xrow[j + 16]);
                }
            }
            assert!(
                (got1[r] - want).abs() < 0.05,
                "f32 path [{r}] got {} want {want} diff {}",
                got1[r],
                got1[r] - want
            );
        }
    }

    /// 8m: the prefill f16 GEMM path (nt >= 16) for every supported quant
    /// type — random VALID block bytes with small d/dmin, reference computed
    /// in Rust by dequantizing those exact bytes (kernel-vs-reference parity;
    /// quantization quality is irrelevant). Tails: od=70, nt=33 (id stays
    /// %32==0 like every real tensor). Real 7B Q4_K check at the end, skipped
    /// when the dump is absent so the suite stays hermetic.
    #[test]
    fn cuda_prefill_f16_gemm_parity() {
        fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
            if j < 4 {
                (q[j] & 63, q[j + 4] & 63)
            } else {
                (
                    (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                    (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
                )
            }
        }
        let _guard = crate::cuda::CudaState::model_load_guard();
        crate::cuda::CudaState::init();
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (od, id, nt) = (70usize, 256usize, 33usize);
        let state = cb.state;

        // seeded pseudo-random source (deterministic across runs)
        let mut seed = 0x2545F491u32;
        let mut rnd = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed
        };
        let xs: Vec<f32> = (0..id * nt)
            .map(|_| (rnd() % 2000) as f32 / 1000.0 - 1.0)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        let (xptr, optr) = (cb.ptr_of(xb).unwrap(), cb.ptr_of(out).unwrap());

        // build [type → (wq bytes, dequant closure)]
        // d values are small so the f16 scratch never overflows.
        let mut mk =
            |nbytes: usize| -> Vec<u8> { (0..nbytes).map(|_| (rnd() & 0xFF) as u8).collect() };

        // Q8_0: d=0.01 + int8 q
        let mut wq80 = mk(od * (id / 32) * 34);
        for g in 0..od * (id / 32) {
            let db = half::f16::from_f32(0.01).to_le_bytes();
            wq80[g * 34] = db[0];
            wq80[g * 34 + 1] = db[1];
        }
        // Q4_0: d=0.05 + biased nibbles (kernel does nib - 8)
        let mut wq40 = mk(od * (id / 32) * 18);
        for g in 0..od * (id / 32) {
            let db = half::f16::from_f32(0.05).to_le_bytes();
            wq40[g * 18] = db[0];
            wq40[g * 18 + 1] = db[1];
        }
        // Q4_K: d=0.01, dmin=0.005, raw scales/nibbles
        let nsp = id / 256;
        let mut wq4k = mk(od * nsp * 144);
        for r in 0..od {
            for sp in 0..nsp {
                let blk = &mut wq4k[(r * nsp + sp) * 144..(r * nsp + sp) * 144 + 144];
                let db = half::f16::from_f32(0.01).to_le_bytes();
                blk[0] = db[0];
                blk[1] = db[1];
                let mb_ = half::f16::from_f32(0.005).to_le_bytes();
                blk[2] = mb_[0];
                blk[3] = mb_[1];
            }
        }
        // Q5_K: d=0.01, dmin=0.005 (176B blocks)
        let mut wq5k = mk(od * nsp * 176);
        for r in 0..od {
            for sp in 0..nsp {
                let blk = &mut wq5k[(r * nsp + sp) * 176..(r * nsp + sp) * 176 + 176];
                let db = half::f16::from_f32(0.01).to_le_bytes();
                blk[0] = db[0];
                blk[1] = db[1];
                let mb_ = half::f16::from_f32(0.005).to_le_bytes();
                blk[2] = mb_[0];
                blk[3] = mb_[1];
            }
        }
        // Q6_K: raw 210B blocks, d = 0.01 at offset 208 (LAST field)
        let mut wq6k = mk(od * nsp * 210);
        for r in 0..od {
            for sp in 0..nsp {
                let blk = &mut wq6k[(r * nsp + sp) * 210..(r * nsp + sp) * 210 + 210];
                let db = half::f16::from_f32(0.01).to_le_bytes();
                blk[208] = db[0];
                blk[209] = db[1];
            }
        }

        state.register_weight("gemm_w80", &wq80);
        state.register_weight("gemm_w40", &wq40);
        state.register_weight("gemm_w4k", &wq4k);
        state.register_weight("gemm_w5k", &wq5k);
        state.register_weight_q6k_padded("gemm_w6k", &wq6k, od, id);

        // ── run the GEMM path per type (nt=33 ≥ 16 hits the gate) ──
        let cases: [(TensorType, &str, bool); 5] = [
            (TensorType::Q8_0, "gemm_w80", false),
            (TensorType::Q4_0, "gemm_w40", false),
            (TensorType::Q4_K, "gemm_w4k", false),
            (TensorType::Q5_K, "gemm_w5k", false),
            (TensorType::Q6_K, "gemm_w6k", true),
        ];
        for (ttype, name, padded) in cases {
            let wptr = state.get_weight_ptr(name).unwrap();
            state
                .matmul_f32_ptr_layout(wptr, ttype, xptr, optr, od, id, nt, padded)
                .unwrap();
            cb.synchronize();
            let got = cb.copy_to_host(out).unwrap();

            // reference: dequant the same bytes, plain f32 dot with raw xs
            let mut want = vec![0f32; od * nt];
            for r in 0..od {
                for t in 0..nt {
                    let mut acc = 0f32;
                    match ttype {
                        TensorType::Q8_0 => {
                            for g in 0..id / 32 {
                                let blk = &wq80[(r * (id / 32) + g) * 34..][..34];
                                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                                for i in 0..32 {
                                    acc += d * (blk[2 + i] as i8 as f32) * xs[t * id + g * 32 + i];
                                }
                            }
                        }
                        TensorType::Q4_0 => {
                            for g in 0..id / 32 {
                                let blk = &wq40[(r * (id / 32) + g) * 18..][..18];
                                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                                for j in 0..16 {
                                    let v0 = (blk[2 + j] & 0x0F) as i32 - 8;
                                    let v1 = (blk[2 + j] >> 4) as i32 - 8;
                                    acc += d
                                        * (v0 as f32 * xs[t * id + g * 32 + j]
                                            + v1 as f32 * xs[t * id + g * 32 + j + 16]);
                                }
                            }
                        }
                        TensorType::Q4_K => {
                            for ib in 0..nsp {
                                let blk = &wq4k[(r * nsp + ib) * 144..][..144];
                                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                                let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                                let mut scb = [0u8; 12];
                                scb.copy_from_slice(&blk[4..16]);
                                for j in 0..4 {
                                    let (s0, m0) = k4_scale(&scb, 2 * j);
                                    let (s1, m1) = k4_scale(&scb, 2 * j + 1);
                                    for l in 0..32 {
                                        let b8 = blk[16 + j * 32 + l];
                                        let base = ib * 256 + j * 64;
                                        let v0 =
                                            (b8 & 0x0F) as f32 * d * s0 as f32 - dmin * m0 as f32;
                                        let v1 =
                                            (b8 >> 4) as f32 * d * s1 as f32 - dmin * m1 as f32;
                                        acc += v0 * xs[t * id + base + l];
                                        acc += v1 * xs[t * id + base + 32 + l];
                                    }
                                }
                            }
                        }
                        TensorType::Q5_K => {
                            for ib in 0..nsp {
                                let blk = &wq5k[(r * nsp + ib) * 176..][..176];
                                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                                let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                                let mut scb = [0u8; 12];
                                scb.copy_from_slice(&blk[4..16]);
                                for sub in 0..8 {
                                    let (scb_s, mb) = k4_scale(&scb, sub);
                                    let ci = sub >> 1;
                                    let hi = sub & 1;
                                    let q4 = &blk[48 + ci * 32..48 + ci * 32 + 32];
                                    let qh = &blk[16..48]; // 256 high bits = 32 bytes
                                    for l in 0..32 {
                                        let nib = if hi != 0 { q4[l] >> 4 } else { q4[l] & 0x0F };
                                        let wv = nib as f32 + 16.0 * ((qh[l] >> sub) & 1) as f32;
                                        let v = d * scb_s as f32 * wv - dmin * mb as f32;
                                        acc += v * xs[t * id + ib * 256 + sub * 32 + l];
                                    }
                                }
                            }
                        }
                        TensorType::Q6_K => {
                            for ib in 0..nsp {
                                let blk = &wq6k[(r * nsp + ib) * 210..][..210];
                                let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                                for sub in 0..16 {
                                    let n = sub / 8;
                                    let rem = sub % 8;
                                    let tt = rem / 2;
                                    let gq = rem % 2;
                                    let ql_off = n * 64 + (tt % 2) * 32 + gq * 16;
                                    // qh field lives at blk[128..192] (64 bytes,
                                    // 2 bits per element); qh_off is relative to it.
                                    let qh_off = 128 + n * 32 + gq * 16;
                                    let dsc = d * (blk[192 + n * 8 + tt * 2 + gq] as i8 as f32);
                                    for rr in 0..16 {
                                        let nib = if tt < 2 {
                                            (blk[ql_off + rr] & 0x0F) as i32
                                        } else {
                                            (blk[ql_off + rr] >> 4) as i32
                                        };
                                        let q2 = ((blk[qh_off + rr] >> (tt * 2)) & 3) as i32;
                                        let v = dsc * (((nib | (q2 << 4)) - 32) as f32);
                                        acc += v * xs
                                            [t * id + ib * 256 + n * 128 + tt * 32 + gq * 16 + rr];
                                    }
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                    want[t * od + r] = acc;
                }
            }

            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            let mut worst = (0f32, 0usize);
            for i in 0..got.len() {
                let e = (got[i] - want[i]).abs();
                if e > worst.0 {
                    worst = (e, i);
                }
            }
            println!(
                "prefill f16 gemm [{name:?}]: max err {:.5} at {} (got {:.4} want {:.4}, scale {scale:.3})",
                worst.0,
                worst.1,
                got[worst.1],
                want[worst.1]
            );
            // f16 weight/activation rounding (~2^-11 rel per element) over a
            // 256-length dot: well under 2% of the row scale.
            assert!(
                worst.0 <= scale * 2e-2,
                "prefill gemm {name:?}: err {} > {} at {}",
                worst.0,
                scale * 2e-2,
                worst.1
            );
        }

        // ── real 7B Q4_K weight (attn_q 3584×3584) through the GEMM path ──
        let Ok(wb) = std::fs::read("/tmp/minfer_phase7/real_blk_0_attn_q_weight.bin") else {
            eprintln!("real q4_k dump absent — skipping the real-weight GEMM check");
            return;
        };
        let (rod, rid) = (3584usize, 3584usize);
        assert_eq!(wb.len(), rod * (rid / 256) * 144);
        state.register_weight("gemm_realq4k", &wb);
        let wptr = state.get_weight_ptr("gemm_realq4k").unwrap();
        let rnt = 17usize;
        let rxs: Vec<f32> = (0..rid * rnt)
            .map(|i| ((i * 73) % 17) as f32 / 8.0 - 1.0)
            .collect();
        let rxb = cb.alloc_buffer(rid * rnt);
        let rout = cb.alloc_buffer(rod * rnt);
        cb.write_host(rxb, &rxs).unwrap();
        let (rxp, rop) = (cb.ptr_of(rxb).unwrap(), cb.ptr_of(rout).unwrap());
        state
            .matmul_f32_ptr_layout(wptr, TensorType::Q4_K, rxp, rop, rod, rid, rnt, false)
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(rout).unwrap();
        let mut worst = (0f32, 0usize);
        let mut scale = 1e-9f32;
        for t in 0..rnt {
            for r in 0..rod {
                let mut acc = 0f32;
                for ib in 0..rid / 256 {
                    let blk = &wb[(r * (rid / 256) + ib) * 144..][..144];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                    let mut scb = [0u8; 12];
                    scb.copy_from_slice(&blk[4..16]);
                    for j in 0..4 {
                        let (s0, m0) = k4_scale(&scb, 2 * j);
                        let (s1, m1) = k4_scale(&scb, 2 * j + 1);
                        for l in 0..32 {
                            let b8 = blk[16 + j * 32 + l];
                            let base = ib * 256 + j * 64;
                            let v0 = (b8 & 0x0F) as f32 * d * s0 as f32 - dmin * m0 as f32;
                            let v1 = (b8 >> 4) as f32 * d * s1 as f32 - dmin * m1 as f32;
                            acc += v0 * rxs[t * rid + base + l];
                            acc += v1 * rxs[t * rid + base + 32 + l];
                        }
                    }
                }
                scale = scale.max(acc.abs());
                let e = (got[t * rod + r] - acc).abs();
                if e > worst.0 {
                    worst = (e, t * rod + r);
                }
            }
        }
        println!(
            "real 7B q4_k f16 gemm: max err {:.4} at {} (scale {scale:.3})",
            worst.0, worst.1
        );
        assert!(
            worst.0 <= scale * 2e-2,
            "real q4_k f16 gemm err {}",
            worst.0
        );
    }

    // 8d: split-K decode attention parity (nt == 1 routes to the split path).
    // nkv = 3 exercises EMPTY splits (positions[0] = 2 → splits 3..7 have no
    // rows); nkv = 37 exercises a partial last split with SPLITS = 8. Both KV
    // layouts checked. Reference: cpu_gqa_attn over the same KV (zero rows +
    // one stored row).
    #[test]
    fn cuda_attn_split_decode_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
        let nkt = nk_h * hd;
        let n_ctx = 64usize;
        let scale = 1.0 / (hd as f32).sqrt();

        for kv_f16 in [false, true] {
            for pos0 in [2usize, 36usize] {
                let nkv = pos0 + 1;
                cb.set_kv_f16_for_test(kv_f16);
                let mut b = GraphBuilder::new();
                let q = b.input("q", [nh * hd, 1, 1, 1], DType::F32);
                let k = b.input("k", [nkt, 1, 1, 1], DType::F32);
                let v = b.input("v", [nkt, 1, 1, 1], DType::F32);
                let pp = b.input("positions", [1, 1, 1, 1], DType::I32);
                let store = b.kvcache_store(0, k, v, pp, n_ctx);
                let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
                let at = b.attn(
                    q,
                    load,
                    pp,
                    AttnMode::Gqa,
                    AttnMeta {
                        layer: 0,
                        n_head: nh,
                        n_head_kv: nk_h,
                        hd,
                        hd_kv: hd,
                        nkt,
                        scale,
                    },
                );
                b.output(at);
                let g = b.build();

                let (xb_q, xb_k, xb_v) = (
                    cb.alloc_buffer(nh * hd),
                    cb.alloc_buffer(nkt),
                    cb.alloc_buffer(nkt),
                );
                let xb_p = cb.alloc_buffer(1);
                let ob_at = cb.alloc_buffer(nh * hd);
                let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

                let qs: Vec<f32> = (0..nh * hd)
                    .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
                    .collect();
                let ks: Vec<f32> = (0..nkt)
                    .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
                    .collect();
                let vs: Vec<f32> = (0..nkt)
                    .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
                    .collect();
                let pb = vec![f32::from_bits(pos0 as u32)];
                let to_half = |x: &[f32]| -> Vec<f32> {
                    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
                };
                let (ks_r, vs_r) = if kv_f16 {
                    (to_half(&ks), to_half(&vs))
                } else {
                    (ks.clone(), vs.clone())
                };
                cb.write_host(xb_q, &qs).unwrap();
                cb.write_host(xb_k, &ks).unwrap();
                cb.write_host(xb_v, &vs).unwrap();
                cb.write_host(xb_p, &pb).unwrap();
                cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
                cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

                cb.execute_node(
                    &g.nodes[store],
                    &[xb_k, xb_v, xb_p],
                    kreg,
                    Some((kreg, vreg)),
                )
                .unwrap();
                cb.execute_node(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
                    .unwrap();

                let mut kfull = vec![0f32; nkt * n_ctx];
                let mut vfull = vec![0f32; nkt * n_ctx];
                kfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&ks_r);
                vfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&vs_r);
                let mut aref = vec![0f32; nh * hd];
                crate::graph::cpu_backend::cpu_gqa_attn(
                    &qs,
                    &kfull,
                    &vfull,
                    &[pos0],
                    1,
                    nkv,
                    nh,
                    nk_h,
                    hd,
                    hd,
                    nkt,
                    &mut aref,
                    scale,
                )
                .unwrap();
                let agot = cb.copy_to_host(ob_at).unwrap();
                assert_close(
                    &format!("attn_split(f16kv={kv_f16}, nkv={nkv})"),
                    &agot,
                    &aref,
                    1e-4,
                );
            }
        }
    }

    // 8f: Q5_1 / Q5_K f32-activation matmul parity (incl. the Q5_K partial
    // tail super-block at id = 896 = 3.5 × 256). The weight blocks are
    // quantized in-test against scales unpacked with the REAL
    // block::unpack_q4k_scales, so kernel and reference share the exact
    // decode math and the tolerance stays tight.
    #[test]
    fn cuda_q5_matmul_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let nt = 3usize;

        // ── Q5_1: od 8, id 64 (2 blocks / row) ──
        {
            let (od, id) = (8usize, 64usize);
            let nb = id / 32;
            let wf: Vec<f32> = (0..od * id)
                .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
                .collect();
            let mut wq = Vec::new();
            for r in 0..od {
                for b in 0..nb {
                    let row = &wf[r * id + b * 32..r * id + (b + 1) * 32];
                    let amax = row.iter().fold(0f32, |m, &v| m.max(v));
                    let amin = row.iter().fold(0f32, |m, &v| m.min(v));
                    let d = (amax - amin) / 31.0;
                    let di = if d != 0.0 { 1.0 / d } else { 0.0 };
                    wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                    wq.extend_from_slice(&half::f16::from_f32(amin).to_le_bytes());
                    let mut qh = 0u32;
                    let mut qs = [0u8; 16];
                    for j in 0..16 {
                        let u_lo = ((row[j] - amin) * di).round().clamp(0.0, 31.0) as u32;
                        let u_hi = ((row[j + 16] - amin) * di).round().clamp(0.0, 31.0) as u32;
                        qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                        qh |= ((u_lo >> 4) & 1) << j;
                        qh |= ((u_hi >> 4) & 1) << (j + 16);
                    }
                    wq.extend_from_slice(&qh.to_le_bytes());
                    wq.extend_from_slice(&qs);
                }
            }
            let state = cb.state;
            state.register_weight("w51", &wq);
            let wptr = state.get_weight_ptr("w51").unwrap();
            let xs: Vec<f32> = (0..id * nt)
                .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
                .collect();
            let xb = cb.alloc_buffer(id * nt);
            let out = cb.alloc_buffer(od * nt);
            cb.write_host(xb, &xs).unwrap();
            state
                .matmul_f32_ptr(
                    wptr,
                    TensorType::Q5_1,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                )
                .unwrap();
            cb.synchronize();
            let got = cb.copy_to_host(out).unwrap();
            // independent dequant reference
            for t in 0..nt {
                for r in 0..od {
                    let mut want = 0f32;
                    for b in 0..nb {
                        let blk = &wq[(r * nb + b) * 24..(r * nb + b) * 24 + 24];
                        let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                        let m = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                        let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                        let xrow = &xs[t * id + b * 32..t * id + (b + 1) * 32];
                        for j in 0..16 {
                            let u_lo = ((blk[8 + j] & 0xF) as f32) + 16.0 * ((qh >> j) & 1) as f32;
                            let u_hi =
                                ((blk[8 + j] >> 4) as f32) + 16.0 * ((qh >> (j + 16)) & 1) as f32;
                            want += d * (u_lo * xrow[j] + u_hi * xrow[j + 16])
                                + m * (xrow[j] + xrow[j + 16]);
                        }
                    }
                    assert!(
                        (got[t * od + r] - want).abs() < 5e-3,
                        "q5_1 [{t}][{r}] {} vs {want}",
                        got[t * od + r]
                    );
                }
            }
        }

        // ── Q5_K: od 8, id 896 (PARTIAL tail super-block: 3.5 × 256) ──
        // Weight values are GENERATED from the decode formula with random
        // per-sub w (0..31) against scales unpacked from random sc bytes —
        // the test targets the kernel's decode/indexing/tail-masking
        // correctness, not a quantizer.
        {
            let (od, id) = (8usize, 896usize);
            let nsp = (id + 255) / 256; // 4 — last one is partial (4 valid subs)
            let mut wf: Vec<f32> = (0..od * id)
                .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
                .collect();
            let mut wq = vec![0u8; od * nsp * 176];
            for r in 0..od {
                for sp in 0..nsp {
                    let blk_off = (r * nsp + sp) * 176;
                    let sc: [u8; 12] = core::array::from_fn(|i| ((i * 7 + 3) % 63 + 1) as u8);
                    let (scales, mins) = crate::block::unpack_q4k_scales(&sc);
                    wq[blk_off..blk_off + 4].copy_from_slice(&{
                        // d = 0.25, dmin = 0.25 (exact in f16)
                        let b = half::f16::from_f32(0.25).to_le_bytes();
                        [b[0], b[1], b[0], b[1]]
                    });
                    wq[blk_off + 4..blk_off + 16].copy_from_slice(&sc);
                    // qh/qs stay zero for invalid tail subs (masked out)
                    let valid = ((id - sp * 256).min(256) + 31) / 32;
                    for sub in 0..valid {
                        let base = sp * 256 + sub * 32;
                        let row = &wf[r * id + base..r * id + base + 32];
                        // invert the decode: v = d·s8·w − dmin·m8 →
                        // w = (v + dmin·m8) / (d·s8); needs w ∈ 0..31 —
                        // instead regenerate v FROM w so it is exact:
                        for l in 0..32 {
                            let seed = (r * 91 + base + l) % 32;
                            let v =
                                0.25 * scales[sub] as f32 * seed as f32 - 0.25 * mins[sub] as f32;
                            // overwrite wf so the reference dot uses exact values
                            wf[r * id + base + l] = v;
                            let wv = seed as u8;
                            let ci = sub >> 1;
                            if sub & 1 == 1 {
                                qs_byte(&mut wq[blk_off + 48..], ci, l, wv, true);
                            } else {
                                qs_byte(&mut wq[blk_off + 48..], ci, l, wv, false);
                            }
                            qh_byte(&mut wq[blk_off + 16..], l, sub, wv);
                        }
                    }
                }
            }
            let state = cb.state;
            state.register_weight("w5k", &wq);
            let wptr = state.get_weight_ptr("w5k").unwrap();
            let xs: Vec<f32> = (0..id * nt)
                .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
                .collect();
            let xb = cb.alloc_buffer(id * nt);
            let out = cb.alloc_buffer(od * nt);
            cb.write_host(xb, &xs).unwrap();
            state
                .matmul_f32_ptr(
                    wptr,
                    TensorType::Q5_K,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                )
                .unwrap();
            cb.synchronize();
            let got = cb.copy_to_host(out).unwrap();
            // independent dequant reference (mirrors the kernel decode)
            let deq = |r: usize| -> Vec<f32> {
                let mut outv = vec![0f32; id];
                for sp in 0..nsp {
                    let blk_off = (r * nsp + sp) * 176;
                    let d = half::f16::from_le_bytes([wq[blk_off], wq[blk_off + 1]]).to_f32();
                    let dmin =
                        half::f16::from_le_bytes([wq[blk_off + 2], wq[blk_off + 3]]).to_f32();
                    let sc: [u8; 12] = wq[blk_off + 4..blk_off + 16].try_into().unwrap();
                    let (scales, mins) = crate::block::unpack_q4k_scales(&sc);
                    let valid = ((id - sp * 256).min(256) + 31) / 32;
                    for sub in 0..valid {
                        let ci = sub >> 1;
                        for l in 0..32 {
                            let qbyte = wq[blk_off + 48 + ci * 32 + l];
                            let nib = if sub & 1 == 1 {
                                qbyte >> 4
                            } else {
                                qbyte & 0xF
                            };
                            let w =
                                nib as f32 + 16.0 * (((wq[blk_off + 16 + l] >> sub) & 1) as f32);
                            outv[sp * 256 + sub * 32 + l] =
                                d * scales[sub] as f32 * w - dmin * mins[sub] as f32;
                        }
                    }
                }
                outv
            };
            for t in 0..nt {
                for r in 0..od {
                    let dq = deq(r);
                    let mut want = 0f32;
                    for i in 0..id {
                        want += dq[i] * xs[t * id + i];
                    }
                    assert!(
                        (got[t * od + r] - want).abs() < 5e-3,
                        "q5_K [{t}][{r}] {} vs {want}",
                        got[t * od + r]
                    );
                }
            }
        }
    }

    // q5_K qs nibble packing: 4 chunks of 32 bytes; chunk ci, byte l:
    // low nibble = element l of sub 2ci, high = element l of sub 2ci+1
    fn qs_byte(qs: &mut [u8], ci: usize, l: usize, w: u8, hi: bool) {
        if hi {
            qs[ci * 32 + l] |= w << 4;
        } else {
            qs[ci * 32 + l] |= w & 0xF;
        }
    }
    // q5_K qh layout: byte l, bit sub = the >16 bit of element (sub, l)
    fn qh_byte(qh: &mut [u8], l: usize, sub: usize, w: u8) {
        qh[l] |= ((w >> 4) & 1) << sub;
    }

    #[test]
    fn cuda_scheduler_chain() {
        crate::cuda::CudaState::init();
        let Some(state) = crate::cuda::CudaState::get() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (id_, od, nt) = (64usize, 32usize, 2usize);
        let cw: Vec<f32> = (0..id_).map(|i| 0.8 + (i % 5) as f32 / 10.0).collect();
        let cwb: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
        state.register_weight("cw", &cwb);
        let mut cwt = Tensor::from_data(TensorType::F32, &[id_ as i64, 1, 1, 1], cwb);
        cwt.name = "cw".to_string();
        let wf: Vec<f32> = (0..od * id_)
            .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let mut w8b = Vec::new();
        for r in 0..od {
            w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
                &wf[r * id_..(r + 1) * id_],
            ));
        }
        state.register_weight("cw8", &w8b);
        let mut w8t = Tensor::from_data(TensorType::Q8_0, &[id_ as i64, od as i64, 1, 1], w8b);
        w8t.name = "cw8".to_string();
        let bias: Vec<f32> = (0..od).map(|i| (i % 3) as f32 / 7.0).collect();
        let bb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        state.register_weight("cb", &bb);
        let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], bb);
        bt.name = "cb".to_string();

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id_, nt, 1, 1], DType::F32);
        let n1 = b.rms_norm(x, Some(&cwt), 1e-5);
        let m = b.matmul(n1, &w8t, Some(&bt));
        let s = b.silu(m);
        b.output(s);
        let mut g = b.build();

        // Full pipeline: assign → alloc → fill → execute (no fusion needed).
        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = crate::graph::scheduler::BackendScheduler;
        sched.assign_backends(&mut g, &alloc);
        for (i, nd) in g.nodes.iter().enumerate() {
            assert_eq!(
                nd.backend,
                Some(crate::graph::Backend::Cuda),
                "node {i} ({})",
                nd.name
            );
        }
        alloc.alloc_graph(&g).unwrap();
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|i| ((i * 97) % 21) as f32 / 5.0 - 2.0)
            .collect();
        alloc.fill_input(&g, "x", &xs).unwrap();
        sched.execute(&g, &mut alloc).unwrap();

        // Host reference: rms → dequant matmul + bias → silu
        let mut rmsd = vec![0f32; id_ * nt];
        for t in 0..nt {
            crate::vec_ops::rms_norm_fused_f32(
                id_,
                &mut rmsd[t * id_..(t + 1) * id_],
                &xs[t * id_..(t + 1) * id_],
                &cw,
                1e-5,
            );
        }
        let mut dq = vec![0f32; od * id_];
        crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8t, &mut dq, id_);
        let mut mm = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += dq[r * id_ + i] * rmsd[t * id_ + i];
                }
                mm[t * od + r] = acc + bias[r];
            }
        }
        let mut want = vec![0f32; od * nt];
        crate::vec_ops::vec_silu_f32(od * nt, &mut want, &mm);
        let got = alloc.copy_to_cpu(s).unwrap();
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("scheduler chain", &got, &want, scale * 1e-3);
    }

    // ─── Phase 7d: CUDA Graph capture/replay ─────────────────────

    /// x, y → silu(x) + y: a weightless all-CUDA graph exercising the
    /// capture/replay bookkeeping without model weights.
    fn replay_graph() -> crate::graph::ComputeGraph {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], DType::F32);
        let y = b.input("y", [8, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let o = b.add(s, y);
        b.output(o);
        b.build()
    }

    fn replay_alloc(graphs_enabled: bool) -> GraphAllocator {
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda(), "cuda device required");
        if !graphs_enabled {
            alloc.cuda_mut().unwrap().set_graphs_enabled_for_test(false);
        }
        alloc
    }

    fn replay_step(
        sched: &BackendScheduler,
        graph: &crate::graph::ComputeGraph,
        alloc: &mut GraphAllocator,
        seed: f32,
    ) -> Vec<f32> {
        let xs: Vec<f32> = (0..8).map(|i| seed + i as f32).collect();
        let ys: Vec<f32> = (0..8).map(|i| (seed * 0.5) - i as f32).collect();
        alloc.fill_input(graph, "x", &xs).unwrap();
        alloc.fill_input(graph, "y", &ys).unwrap();
        sched.execute(graph, alloc).unwrap();
        alloc.copy_to_cpu(graph.outputs[0]).unwrap()
    }

    /// Warmup → capture → replay must be bit-identical to pure direct
    /// launches for every step (llama.cpp's core replay guarantee).
    #[test]
    fn cuda_graph_replay_bit_parity() {
        // 8m: serialize against other tests' stream users — capture on the shared
        // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = BackendScheduler;
        let mut g_cap = replay_graph();
        let mut g_ref = replay_graph();
        let mut cap = replay_alloc(true);
        let mut refr = replay_alloc(false);
        sched.assign_backends(&mut g_cap, &cap);
        sched.assign_backends(&mut g_ref, &refr);
        cap.alloc_graph(&g_cap).unwrap();
        refr.alloc_graph(&g_ref).unwrap();

        for step in 0..5u32 {
            let seed = 10.0 + 10.0 * step as f32;
            let got = replay_step(&sched, &g_cap, &mut cap, seed);
            let want = replay_step(&sched, &g_ref, &mut refr, seed);
            assert_eq!(
                got, want,
                "step {step}: replay path diverged from direct launches"
            );
        }
        // steps 1-2 direct, step 3 captured, steps 4-5 replayed
        assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);
    }

    /// 8g①: a prefill-shaped graph (any matmul with nt > 1) must NEVER open
    /// a capture window, even after 3+ executions of the same (uid, range) —
    /// a repeated identical-nt prefill (server scenario) would otherwise be
    /// captured without validation.
    #[test]
    fn cuda_prefill_shaped_graph_never_captures() {
        // 8m: serialize against other tests' stream users — capture on the shared
        // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut g = {
            let mut b = GraphBuilder::new();
            let x = b.input("x", [8, 8, 1, 1], DType::F32);
            let y = b.input("y", [8, 8, 1, 1], DType::F32);
            let s = b.silu(x);
            let a = b.add(s, y);
            let wb: Vec<u8> = (0..32)
                .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
                .collect();
            let mut w = Tensor::from_data(crate::tensor::TensorType::F32, &[8, 4, 1, 1], wb);
            w.name = "w".to_string();
            let o = b.matmul(a, &w, None);
            b.output(o);
            b.build()
        };
        assert_eq!(g.capture_nt_hint(), Some(8), "prefill-shaped hint");

        let sched = BackendScheduler;
        let mut cap = replay_alloc(true);
        sched.assign_backends(&mut g, &mut cap);
        cap.cuda_mut().unwrap().state.register_weight(
            "w",
            &(0..32)
                .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
                .collect::<Vec<u8>>(),
        );
        cap.alloc_graph(&g).unwrap();

        for step in 0..4u32 {
            let seed = 3.0 + 7.0 * step as f32;
            let xs: Vec<f32> = (0..64).map(|i| seed + i as f32).collect();
            let ys: Vec<f32> = (0..64).map(|i| seed * 0.25 - i as f32).collect();
            cap.fill_input(&g, "x", &xs).unwrap();
            cap.fill_input(&g, "y", &ys).unwrap();
            sched.execute(&g, &mut cap).unwrap();
        }
        let cb = cap.cuda_mut().unwrap();
        assert_eq!(
            cb.captured_count(),
            0,
            "prefill-shaped graph must never be captured"
        );
        assert!(cb.capturing.is_none());
    }

    /// 8i-1: MULTI-SPLIT capture. A CUDA op → CPU op → CUDA op graph yields
    /// two CUDA splits; each must capture and replay independently with
    /// bit-identical results vs pure direct launches (per-split capture is
    /// supported but 7d's parity only covered single-split graphs).
    #[test]
    fn cuda_multisplit_capture_bit_parity() {
        // 8m: serialize against other tests' stream users — capture on the shared
        // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        // Softmax has no CUDA kernel (stays on CPU) → forces a split between
        // two CUDA segments.
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], DType::F32);
        let y = b.input("y", [8, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let sm = b.softmax(s, 0);
        let o = b.add(sm, y);
        b.output(o);
        let mut g_cap = b.build();

        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], DType::F32);
        let y = b.input("y", [8, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let sm = b.softmax(s, 0);
        let o = b.add(sm, y);
        b.output(o);
        let mut g_ref = b.build();

        let sched = BackendScheduler;
        let mut cap = replay_alloc(true);
        let mut refr = replay_alloc(false);
        sched.assign_backends(&mut g_cap, &cap);
        sched.assign_backends(&mut g_ref, &refr);
        cap.alloc_graph(&g_cap).unwrap();
        refr.alloc_graph(&g_ref).unwrap();

        for step in 0..5u32 {
            let seed = 5.0 + 3.0 * step as f32;
            let got = replay_step(&sched, &g_cap, &mut cap, seed);
            let want = replay_step(&sched, &g_ref, &mut refr, seed);
            assert_eq!(
                got, want,
                "step {step}: multi-split replay diverged from direct launches"
            );
        }
        assert_eq!(
            cap.cuda_mut().unwrap().captured_count(),
            2,
            "both CUDA splits must be captured"
        );
    }

    /// 8g②: deliberate prefill capture — with the opt-in ON, a repeated
    /// identical-nt prefill-shaped graph captures after the 3-run protocol
    /// and replays BIT-IDENTICAL to direct launches, at both pp16 and pp300
    /// (the ~437-node real-prefill scale). Default remains OFF (8g① test).
    #[test]
    fn cuda_prefill_capture_bit_parity_pp16_pp300() {
        // 8m: serialize against other tests' stream users — capture on the shared
        // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let build = |nt: usize| -> crate::graph::ComputeGraph {
            let mut b = GraphBuilder::new();
            let x = b.input("x", [8, nt, 1, 1], DType::F32);
            let y = b.input("y", [8, nt, 1, 1], DType::F32);
            let s = b.silu(x);
            let a = b.add(s, y);
            let wb: Vec<u8> = (0..32)
                .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
                .collect();
            let mut w = Tensor::from_data(crate::tensor::TensorType::F32, &[8, 4, 1, 1], wb);
            w.name = "w".to_string();
            let o = b.matmul(a, &w, None);
            b.output(o);
            b.build()
        };

        let sched = BackendScheduler;
        for nt in [16usize, 300usize] {
            let mut g_cap = build(nt);
            let mut g_ref = build(nt);
            let mut cap = replay_alloc(true);
            let mut refr = replay_alloc(false);
            cap.cuda_mut().unwrap().set_prefill_capture_for_test(true);
            let wb: Vec<u8> = (0..32)
                .flat_map(|i| ((i as f32 - 16.0) / 32.0).to_le_bytes())
                .collect();
            cap.cuda_mut().unwrap().state.register_weight("w", &wb);
            refr.cuda_mut().unwrap().state.register_weight("w", &wb);
            sched.assign_backends(&mut g_cap, &cap);
            sched.assign_backends(&mut g_ref, &refr);
            cap.alloc_graph(&g_cap).unwrap();
            refr.alloc_graph(&g_ref).unwrap();

            for step in 0..5u32 {
                let seed = 1.0 + 2.0 * step as f32;
                let xs: Vec<f32> = (0..8 * nt).map(|i| seed + (i % 9) as f32).collect();
                let ys: Vec<f32> = (0..8 * nt).map(|i| seed * 0.5 - (i % 7) as f32).collect();
                cap.fill_input(&g_cap, "x", &xs).unwrap();
                cap.fill_input(&g_cap, "y", &ys).unwrap();
                refr.fill_input(&g_ref, "x", &xs).unwrap();
                refr.fill_input(&g_ref, "y", &ys).unwrap();
                sched.execute(&g_cap, &mut cap).unwrap();
                sched.execute(&g_ref, &mut refr).unwrap();
                let got = cap.copy_to_cpu(g_cap.outputs[0]).unwrap();
                let want = refr.copy_to_cpu(g_ref.outputs[0]).unwrap();
                assert_eq!(
                    got, want,
                    "pp{nt} step {step}: prefill replay diverged from direct launches"
                );
            }
            assert_eq!(
                cap.cuda_mut().unwrap().captured_count(),
                1,
                "pp{nt}: the prefill split must be captured exactly once"
            );
        }
    }

    /// Phase 8 review: an execute_node error during an open capture window
    /// must ABORT the window (the scheduler propagates before the boundary
    /// sync, so nothing else would close it). Driven directly here because
    /// no supported model can fail a node mid-capture today.
    #[test]
    fn cuda_capture_abort_on_error() {
        // 8m: serialize against other tests' stream users — capture on the shared
        // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = BackendScheduler;
        let mut g_cap = replay_graph();
        let mut g_ref = replay_graph();
        let mut cap = replay_alloc(true);
        let mut refr = replay_alloc(false);
        sched.assign_backends(&mut g_cap, &cap);
        sched.assign_backends(&mut g_ref, &refr);
        cap.alloc_graph(&g_cap).unwrap();
        refr.alloc_graph(&g_ref).unwrap();

        // open the window via the 3-run protocol WITHOUT executing nodes
        let cb = cap.cuda_mut().unwrap();
        for _ in 0..3 {
            cb.graph_replay_step(7, (0, 1), None);
        }
        assert!(cb.capturing.is_some(), "3rd run must open a capture window");
        assert!(cb.stream_guard.is_some(), "window holds the stream lock");

        // the error path: abort, not close
        cb.abort_capture("unit test");
        assert!(cb.capturing.is_none(), "window must be closed");
        assert!(cb.stream_guard.is_none(), "stream lock released");
        assert_eq!(
            cb.graphs_mode,
            GraphMode::Disabled,
            "graphs disabled after an aborted window"
        );
        assert_eq!(cb.captured_count(), 0, "aborted window must not be cached");
        assert!(
            !cb.graph_replay_step(7, (0, 1), None),
            "no replay after graphs are disabled"
        );

        // direct execution keeps working after the abort
        let got = replay_step(&sched, &g_cap, &mut cap, 99.0);
        let want = replay_step(&sched, &g_ref, &mut refr, 99.0);
        assert_eq!(got, want, "post-abort direct execution diverged");
    }

    /// A pool generation change after capture must invalidate the stored exec
    /// (conservative: pointers may differ) and re-capture on a later run.
    #[test]
    fn cuda_graph_recaptures_on_pool_gen_change() {
        // 8m: serialize against other tests' stream users — capture on the shared
        // stream is not thread-safe (race exposed by the prefill GEMM timing shift).
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = BackendScheduler;
        let mut g = replay_graph();
        let mut cap = replay_alloc(true);
        let mut refr = replay_alloc(false);
        sched.assign_backends(&mut g, &cap);
        cap.alloc_graph(&g).unwrap();
        refr.alloc_graph(&g).unwrap();

        for step in 0..3u32 {
            let seed = 1.0 + step as f32;
            let got = replay_step(&sched, &g, &mut cap, seed);
            let want = replay_step(&sched, &g, &mut refr, seed);
            assert_eq!(got, want, "warmup step {step}");
        }
        assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);

        // bump pool_gen behind the backend's back (as a new staging alloc
        // would). Invalidation is lazy: the stale exec is dropped at the next
        // graph_replay call, before it could ever be launched.
        let c = cap.cuda_mut().unwrap();
        let _fresh = Backend::alloc_fresh(c, 64);

        // run 4: graph_replay sees the pool_gen change → drops the exec and
        // runs direct (warmup restarts). Parity holds throughout.
        let got = replay_step(&sched, &g, &mut cap, 4.0);
        let want = replay_step(&sched, &g, &mut refr, 4.0);
        assert_eq!(got, want, "post-invalidation step 4");
        assert_eq!(
            cap.cuda_mut().unwrap().captured_count(),
            0,
            "stale exec must be dropped after pool churn"
        );

        // run 5 direct (warmup 2), run 6 re-captures — parity holds
        for step in 5..7u32 {
            let seed = step as f32;
            let got = replay_step(&sched, &g, &mut cap, seed);
            let want = replay_step(&sched, &g, &mut refr, seed);
            assert_eq!(got, want, "post-invalidation step {step}");
        }
        assert_eq!(cap.cuda_mut().unwrap().captured_count(), 1);
    }

    /// Real-model generation: two full generations (independent caches) must
    /// produce identical greedy tokens — the first loop mixes direct/capture/
    /// replay executions, the second replays everything, and a third loop
    /// with graphs force-disabled is the direct-launch reference.
    #[test]
    fn cuda_graph_generation_replay_parity_real_model() {
        use crate::models::qwen2::graph::Qwen2Graph;
        use crate::models::qwen2::Qwen2Model;

        crate::cuda::CudaState::init();
        if crate::cuda::CudaState::get().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut p = std::path::PathBuf::from(std::env::var("HOME").unwrap());
        p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
        if !p.exists() {
            eprintln!("skipping: qwen2.5-0.5b q4_0 not cached");
            return;
        }
        // Hold the model-load lock from BEFORE the load through the whole
        // comparison: a parallel test loading a different architecture
        // registers same-named tensors of a different size, which would swap
        // the weight registry underneath these loops and corrupt one of them.
        // (The guard is reentrant — load_model takes it again internally.)
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let gguf = crate::gguf::load_gguf_model(&p).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().unwrap();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        let nt = ids.len();
        // Full model context (32k) would size f32 KV regions at ~800 MB per
        // cache — x3 caches here. 4096 comfortably covers a 200-token decode
        // and keeps the parallel suite's device-memory footprint small.
        let n_ctx = 4096;

        fn generate(
            q2: &Qwen2Model,
            ids: &[u32],
            nt: usize,
            n_ctx: usize,
            steps: usize,
        ) -> (Vec<u32>, Vec<f32>) {
            let mut cache = GraphCache::new();
            let positions: Vec<usize> = (0..nt).collect();
            let mut logits = Qwen2Graph::forward_cached(q2, ids, &positions, 1, n_ctx, &mut cache);
            let mut toks = Vec::with_capacity(steps);
            for step in 0..steps {
                let next = logits
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .unwrap()
                    .0 as u32;
                toks.push(next);
                logits =
                    Qwen2Graph::forward_cached(q2, &[next], &[nt + step], 1, n_ctx, &mut cache);
            }
            (toks, logits)
        }

        // loop 1: warmup → capture → replay across the steps (200 tokens)
        let (toks1, last1) = generate(q2, &ids, nt, n_ctx, 200);
        // loop 2: everything replays (fresh cache, fresh backend bookkeeping)
        let (toks2, last2) = generate(q2, &ids, nt, n_ctx, 200);
        assert_eq!(toks1, toks2, "replay generation diverged from mixed-mode");
        assert_eq!(last1, last2, "final-step logits diverged bitwise");

        // loop 3: graphs force-disabled — the direct-launch reference. The
        // allocator must get its CUDA backend (and the disabled flag) before
        // the first forward_cached call, which would otherwise create it.
        let mut cache3 = GraphCache::new();
        cache3.alloc().disable_graphs_for_test();
        let positions: Vec<usize> = (0..nt).collect();
        let mut logits3 = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, n_ctx, &mut cache3);
        let mut toks3 = Vec::with_capacity(200);
        for step in 0..200 {
            let next = logits3
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .unwrap()
                .0 as u32;
            toks3.push(next);
            logits3 = Qwen2Graph::forward_cached(q2, &[next], &[nt + step], 1, n_ctx, &mut cache3);
        }
        assert_eq!(
            toks1, toks3,
            "graph-captured generation diverged from direct launches"
        );
    }
}
