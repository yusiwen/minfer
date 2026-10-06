//! Metal (MPS) backend — Phase 3.
//!
//! Executes IR nodes by dispatching to MpsState's existing per-op kernels
//! (rms_norm, quant_matmul_f32_on_gpu_buf, rope_f32, silu_f32, add_f32,
//! mul_f32, swiglu_f32, embed_tokens_gpu, store_kv, gqa_attn_f32_off).
//!
//! Buffers are shared-memory MTLBuffers (host + GPU visible), so read_host /
//! write_host are direct memory views — cross-backend copies are plain host
//! round trips. One `MpsCommandBuffer` is kept per split and submitted by
//! `synchronize()` (called at split boundaries), so ops within a split share a
//! single GPU submission — the plan's §15 "split shares one command buffer"
//! rule.
//!
//! GPU safety (docs/GPU_SAFETY.md): kernel-invariant violations return Err —
//! the caller must not treat them as a silent CPU fallback; supported-model
//! constraints are checked up front (nkt == nk*hd for attention).

use crate::metal::MpsState;
#[cfg(target_os = "macos")]
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLSharedEvent};

use super::backend::Backend;
use super::ops::{FusedOp, NodeMeta, Op};
use super::{BufRef, CNode, DType};

/// F5 (#137): the `MINFER_TEST_CALL_FAIL` site of the Metal cross-backend
/// staging copy ([`docs/GATE-CONTRACT.md`]). Under injection the enqueue
/// suppresses its `MTLSharedEvent` signal, so phase B's bounded wait genuinely
/// times out and the mutation gate reads the real timeout status. Unset in every
/// default, bench and device run.
///
/// [`docs/GATE-CONTRACT.md`]: ../docs/GATE-CONTRACT.md
pub(crate) const CROSS_COPY_SITE: &str = "metal_cross_copy";

/// F5 (#137): the bound on the phase-B host wait, in milliseconds. The same 10 s
/// order as the split submission's bound (`MpsCommandBuffer::submit`): a GPU
/// that never signals the event is a loud `Err`, never an unbounded host block
/// (`docs/GPU_SAFETY.md`).
const CROSS_WAIT_TIMEOUT_MS: u64 = 10_000;

/// F5 (#137): the value the staging blit signals and phase B waits for. Each
/// copy owns a fresh event, so 1 is always "this copy finished".
const CROSS_EVENT_VALUE: u64 = 1;

// ─── op profiler (MINFER_OP_PROFILE=1, debug aid) ────────────────────
// Host-side encode time per op label (accumulated across the process) plus
// per-submit GPU wait time. The first submit prints the full per-op table
// (prefill); later submits print one line each (decode = one submit per token).
// Zero overhead when the env var is unset.
use std::collections::BTreeMap;
static OP_ENC: std::sync::Mutex<BTreeMap<String, (u64, f64)>> =
    std::sync::Mutex::new(BTreeMap::new());
static GPU_MS: std::sync::Mutex<(u64, f64)> = std::sync::Mutex::new((0, 0.0));

fn op_profile_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_OP_PROFILE").map_or(false, |v| v == "1"))
}

/// Records host-side encode time per op label on drop (works with early returns).
struct EncTimer {
    key: String,
    t0: std::time::Instant,
}
impl Drop for EncTimer {
    fn drop(&mut self) {
        let ms = self.t0.elapsed().as_secs_f64() * 1e3;
        let mut m = OP_ENC.lock().unwrap();
        let e = m.entry(std::mem::take(&mut self.key)).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
    }
}

pub struct MetalBackend {
    state: &'static MpsState,
    /// f32-element pool: id → shared MTLBuffer (size * 4 bytes)
    pool: Vec<crate::metal::MetalBuffer>,
    free: Vec<usize>,
    /// P2/P3 capture staging: host-readable buffers written by per-split blits.
    /// Only allocated while a trace/live capture is armed.
    staging: Vec<crate::metal::MetalBuffer>,
    free_staging: Vec<usize>,
    /// Pending command buffer for the current split. Stored as a leaked box
    /// pointer (null = none) because MpsCommandBuffer is !Send/!Sync; all
    /// access happens sequentially through &self/&mut self methods on the
    /// scheduler thread, so the raw pointer is contained.
    cb_ptr: *mut crate::metal::MpsCommandBuffer<'static>,
    /// F5 (#137): in-flight cross-backend staging copies, keyed by the same
    /// `(graph uid, node, destination backend)` triple the allocator's `cross`
    /// map uses. Each record owns the staging buffer the blit wrote, the shared
    /// event the consumer waits on, and the command buffer it was encoded into
    /// (so phase B can report the GPU's real status). An entry is removed by
    /// phase B; `copy_across` makes a re-request while the entry is still here a
    /// no-op, so a record can never be silently replaced.
    cross: std::collections::HashMap<(u64, super::NodeId, super::Backend), MetalCrossCopy>,
    /// F5 (#137): how many times this backend handed a pool buffer's bytes to
    /// the host directly (`Backend::read_host`) — the synchronous staging path's
    /// readback. Metal's buffers are `StorageModeShared`, so there is no
    /// separate blocking-copy API to count; this is the device-level analogue of
    /// `CudaBackend::blocking_readback_count` and the async path never touches
    /// it (it publishes its own staging bytes instead). See
    /// `docs/BACKEND-REGISTRY-DESIGN.md` §11.3.
    sync_readbacks: std::sync::atomic::AtomicU64,
    /// C4 per-engine (issue #44 part (b); the Metal half of #99/#153): the KV
    /// element type this pool's regions store. Stamped from
    /// [`GraphAllocator::set_kv_format`](super::alloc::GraphAllocator::set_kv_format)
    /// — the loaded engine's resolved format — and read by every attention/store
    /// dispatch and the registry's `kv_format` hook, so two engines in one
    /// process hold their own layouts and a C5 session header describes the width
    /// its region really uses. `F32` until stamped (the pre-enable default).
    kv_format: super::kvformat::KvFormat,
}

/// F5 (#137): one in-flight Metal cross-backend staging copy (see
/// [`MetalBackend::cross`]).
struct MetalCrossCopy {
    /// The `StorageModeShared` buffer the GPU blit wrote; valid once `event`
    /// reaches `value`.
    staging: crate::metal::MetalBuffer,
    /// The `MTLSharedEvent` the blit signals and phase B waits on. Each copy has
    /// its own event, so the value space is never shared between concurrent
    /// copies.
    event: crate::metal::MetalSharedEvent,
    /// The command buffer the blit was encoded into, retained so phase B can
    /// report a real `status()` / `error()` after the event wait.
    cmd: crate::metal::MetalCommandBuffer,
    /// The value the blit signals (and the host waits for). Always 1 today; the
    /// field exists so the wait names the value it was waiting for.
    value: u64,
}

// Safety: every field is either owned (pool/free), a 'static reference
// (MpsState is a Sync singleton), or the command-buffer pointer which is only
// dereferenced inside &self/&mut self methods (sequential, single-threaded).
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// P2/P3 capture: encode blits copying `src_ids` (pool buffers) into
    /// staging buffers, at the END of this split's command buffer (after all
    /// kernels, so the data is this step's output). Returns the staging ids —
    /// their contents are valid only after the next `synchronize`.
    pub fn capture_split(&mut self, src_ids: &[usize]) -> Result<Vec<usize>, String> {
        let mut dst_ids = Vec::with_capacity(src_ids.len());
        for &sid in src_ids {
            let len = self.buf(sid).length() as usize;
            dst_ids.push(self.staging_alloc(len)?);
        }
        let pairs: Vec<(usize, usize)> = src_ids
            .iter()
            .copied()
            .zip(dst_ids.iter().copied())
            .collect();
        self.cb()
            .encode_captures(&pairs, &self.pool, &self.staging)?;
        Ok(dst_ids)
    }

    fn staging_alloc(&mut self, len_bytes: usize) -> Result<usize, String> {
        if let Some(pos) = self
            .free_staging
            .iter()
            .position(|&id| self.staging[id].length() as usize == len_bytes)
        {
            return Ok(self.free_staging.swap_remove(pos));
        }
        if len_bytes % 4 != 0 {
            return Err(format!("capture staging size {len_bytes} not 4-aligned"));
        }
        let buf = self.state.new_f32_buffer(len_bytes / 4);
        self.staging.push(buf);
        Ok(self.staging.len() - 1)
    }

    /// Read a staging buffer (valid after the split's command buffer was
    /// submitted by `synchronize`).
    pub fn read_staging(&self, id: usize) -> Option<&[f32]> {
        let buf = self.staging.get(id)?;
        let len = (buf.length() as usize) / 4;
        Some(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) })
    }

    /// Return all staging buffers to the free list (call after the readback of
    /// one split's captures; they may be reused by the next split).
    pub fn release_staging_all(&mut self) {
        self.free_staging = (0..self.staging.len()).collect();
    }

    /// F5 (#137) **phase A**: enqueue the staging copy of `src` (a Metal pool
    /// reference) for `(uid, node_id, dst_backend)` and return **without
    /// waiting** for it.
    ///
    /// One fresh `StorageModeShared` staging buffer and one fresh
    /// `MTLSharedEvent`, plus a `MTLBlitCommandEncoder` copy and
    /// `encodeSignalEvent` encoded into the split's own command buffer (a
    /// standalone one when no split buffer is open). Nothing here blocks the
    /// host, and the encoding is ordered behind the split's kernels in the same
    /// submission, so the copy reads exactly the bytes the split wrote.
    ///
    /// The record is keyed exactly as the allocator's staging map is, so a
    /// re-request (which `copy_across` already turns into a no-op while the
    /// entry is pending) could never overwrite a live event.
    pub(crate) fn cross_enqueue(
        &mut self,
        uid: u64,
        node_id: super::NodeId,
        dst_backend: super::Backend,
        src: BufRef,
    ) -> Result<(), String> {
        if self.cross.contains_key(&(uid, node_id, dst_backend)) {
            // `copy_across` no-ops a re-request while the entry is pending, so
            // this is a backstop, not a path: replacing the record would drop
            // the first copy's event and staging buffer on the floor.
            return Err(format!(
                "metal cross_enqueue: a staging copy for node {node_id} on {dst_backend:?} is \
                 already in flight"
            ));
        }
        let src_buf = self
            .pool
            .get(src.id)
            .ok_or_else(|| format!("metal: no buffer {}", src.id))?
            .clone();
        let bytes = src.len * 4;
        let staging = self.state.new_f32_buffer(src.len);
        let event = self
            .state
            .new_shared_event()
            .ok_or("MTLDevice.newSharedEvent returned nil")?;
        // The injection suppresses the signal, so phase B's *bounded* wait takes
        // its real timeout branch instead of succeeding. Off by default.
        let signal = !crate::testfail::requested(CROSS_COPY_SITE);
        // F5 (#137): encode the blit into the **producer split's** command buffer
        // when one is still open, so the copy is ordered behind that split's
        // kernels in the same submission (one command buffer per split, and the
        // source is read in the same submission that wrote it). At a boundary the
        // scheduler calls this before `Backend::retire` submits the split, so the
        // open buffer is the norm; a source with no open split buffer gets a
        // standalone command buffer that is committed here.
        let had_split_cb = !self.cb_ptr.is_null();
        let cb = self.cb();
        cb.encode_blit_signal(
            &src_buf,
            src.offset * 4,
            &staging,
            bytes,
            &event,
            CROSS_EVENT_VALUE,
            signal,
        )?;
        // Retain the underlying buffer for a real `status()` / `error()` on the
        // failure path before the (standalone) submission consumes `cb`.
        let cmd = cb.command_buffer();
        if !had_split_cb {
            self.submit_pending();
        }
        self.cross.insert(
            (uid, node_id, dst_backend),
            MetalCrossCopy {
                staging,
                event,
                cmd,
                value: CROSS_EVENT_VALUE,
            },
        );
        Ok(())
    }

    /// F5 (#137) **phase B**: wait on the event phase A recorded — once, at the
    /// documented synchronization point — and publish the staging bytes to the
    /// host.
    ///
    /// The wait is **bounded** (10 s, the same order as the split submission's
    /// bound); a GPU that never signals is a loud `Err` naming the value waited
    /// for, the observed `signaledValue`, and the command buffer's real
    /// `status()` / `error()` — never an unbounded block. A blit that faults on
    /// the device never signals its event, so it lands in exactly this branch
    /// with its real status attached; a successful signal **is** the completion
    /// guarantee for the shared staging buffer (the command buffer's own status
    /// can still read `Scheduled` at that instant, which is why it is not
    /// consulted on the success path). `Ok(None)` means this backend never
    /// enqueued a copy for the key (a device destination declined phase A),
    /// which is a no-op for the caller.
    pub(crate) fn cross_take(
        &mut self,
        uid: u64,
        node_id: super::NodeId,
        dst_backend: super::Backend,
    ) -> Result<Option<Vec<f32>>, String> {
        let Some(rec) = self.cross.remove(&(uid, node_id, dst_backend)) else {
            return Ok(None);
        };
        let injected = crate::testfail::requested(CROSS_COPY_SITE);
        // A 1 ms bound under injection keeps the mutation gate fast while still
        // exercising the real timeout branch.
        let timeout_ms = if injected { 1 } else { CROSS_WAIT_TIMEOUT_MS };
        let signaled = rec
            .event
            .waitUntilSignaledValue_timeoutMS(rec.value, timeout_ms);
        if !signaled {
            let status = rec.cmd.status();
            return Err(format!(
                "Metal cross-backend staging copy: the bounded wait on MTLSharedEvent timed out \
                 after {timeout_ms} ms (waited for value {}, observed signaledValue {}; command \
                 buffer status={status:?}{}){}",
                rec.value,
                rec.event.signaledValue(),
                match rec.cmd.error() {
                    Some(e) => format!(", error={e:?}"),
                    None => String::new(),
                },
                if injected {
                    format!(" — MINFER_TEST_CALL_FAIL={CROSS_COPY_SITE} suppressed the signal")
                } else {
                    String::new()
                }
            ));
        }
        let len = (rec.staging.length() as usize) / 4;
        let data = unsafe {
            std::slice::from_raw_parts(rec.staging.contents().as_ptr() as *const f32, len)
        };
        Ok(Some(data.to_vec()))
    }

    /// F5 (#137): how many in-flight staging copies this backend holds. The
    /// idempotence gate reads it to prove a re-request did not create a second
    /// record (and therefore did not leak a staging buffer or an event).
    /// Test-only (#238): driven by `graph::metal_backend::tests::staging::a_re_request_while_in_flight_is_the_same_transfer`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn cross_pending_len(&self) -> usize {
        self.cross.len()
    }

    /// F5 (#137): the device-level count of host readbacks of a Metal pool
    /// buffer (see the `sync_readbacks` field). The async boundary path must not
    /// move it; the synchronous reference must.
    /// Test-only (#238): driven by `graph::metal_backend::tests::staging::a_split_graph_waits_once_per_staged_copy_and_stays_bitwise` and the `#[ignore]`d real-model gate `models::qwen2::graph::tests::offload_copy::async_cross_copies_never_block_and_stay_bitwise_identical_on_metal`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn sync_readback_count(&self) -> u64 {
        self.sync_readbacks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// None when MPS is unavailable or not initialized (MpsState::init()).
    pub fn new() -> Option<Self> {
        let state = MpsState::get()?;
        Some(Self {
            state,
            pool: Vec::new(),
            free: Vec::new(),
            staging: Vec::new(),
            free_staging: Vec::new(),
            cb_ptr: std::ptr::null_mut(),
            cross: std::collections::HashMap::new(),
            sync_readbacks: std::sync::atomic::AtomicU64::new(0),
            kv_format: super::kvformat::KvFormat::F32,
        })
    }

    /// C4 per-engine: stamp the loaded engine's resolved KV format onto this
    /// pool (see the `kv_format` field). Idempotent.
    pub(crate) fn set_kv_format(&mut self, format: super::kvformat::KvFormat) {
        self.kv_format = format;
    }

    /// C4 per-engine: the KV element type this pool's regions store.
    ///
    /// The registry's `kv_format` hook reads the allocator's stamp, not this,
    /// because it must answer before the pool is enabled (C5's `load_slots`);
    /// this is the instance answer the dispatch reads through `kv_f16`.
    /// Test-only: driven by `graph::metal_backend::tests::copy_cells::metal_kv_format_is_per_engine`,
    /// which asserts the pool holds its own layout; `#[cfg(test)]` keeps it out
    /// of production builds.
    #[cfg(test)]
    pub(crate) fn kv_format(&self) -> super::kvformat::KvFormat {
        self.kv_format
    }

    /// Whether the KV kernels must read half-width rows. Metal refuses `Q8_0`
    /// (`READS_PACKED_KV = false`), so the only other answer is f32.
    fn kv_f16(&self) -> bool {
        self.kv_format == super::kvformat::KvFormat::F16
    }

    fn buf(&self, id: usize) -> &crate::metal::MetalBuffer {
        &self.pool[id]
    }

    /// The current split's command buffer (created on first op of a split).
    /// The box is leaked, so the returned reference is 'static and does not
    /// borrow `self` — callers can freely touch the pool afterwards.
    fn cb(&mut self) -> &'static mut crate::metal::MpsCommandBuffer<'static> {
        if self.cb_ptr.is_null() {
            let cb = Box::new(self.state.cmd_buffer());
            self.cb_ptr = Box::into_raw(cb);
        }
        // SAFETY: cb_ptr is null or points to a live box created here; all
        // callers hold &mut self, so no concurrent mutation.
        unsafe { &mut *self.cb_ptr }
    }

    /// Submit the pending command buffer (if any) and clear it.
    fn submit_pending(&mut self) {
        if !self.cb_ptr.is_null() {
            let t0 = std::time::Instant::now();
            // SAFETY: exclusive &mut self — take the box back and submit.
            let cb = unsafe { Box::from_raw(self.cb_ptr) };
            self.cb_ptr = std::ptr::null_mut();
            cb.submit()
                .expect("MPS: graph backend command-buffer submit error");
            if op_profile_enabled() {
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let n = {
                    let mut g = GPU_MS.lock().unwrap();
                    g.0 += 1;
                    g.1 += ms;
                    g.0
                };
                if n == 1 {
                    Self::print_profile();
                } else {
                    eprintln!(
                        "[MINFER_OP_PROFILE] submit #{n}: GPU {ms:.2} ms (total {:.1} ms)",
                        GPU_MS.lock().unwrap().1
                    );
                }
            }
        }
    }

    /// KV-parallel attention chunk count (decode), mirroring layer_gpu's
    /// adaptive rule: one chunk per 32 KV rows, capped at 16, with a
    /// MINFER_ATTN_CHUNKS override.
    fn attention_chunks(&self, positions: &crate::metal::MetalBuffer, count: usize) -> usize {
        let max_pos = Self::positions_max(positions, count);
        std::env::var("MINFER_ATTN_CHUNKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&c| c >= 1)
            .unwrap_or_else(|| ((max_pos + 1 + 31) / 32).clamp(1, 16))
    }

    /// `max(positions[..count])`, read on the host: the positions are input
    /// data, never GPU-computed, so a host read is safe here.
    ///
    /// `count` is the node's **logical** length (`BufRef::len`, i.e. `nt`), not
    /// the pool buffer's class-rounded length (E4 S2): a recycled positions
    /// buffer keeps an earlier graph's tail, so scanning the whole pool buffer
    /// would derive a stale `max_pos` and make a causal prefill attend to rows
    /// past its own window (the `reused_cache_across_prompts_matches_a_fresh_cache`
    /// failure). The pool is always at least `count` long; bound the scan.
    fn positions_max(positions: &crate::metal::MetalBuffer, count: usize) -> usize {
        let n = ((positions.length() as usize) / 4).min(count);
        let p =
            unsafe { std::slice::from_raw_parts(positions.contents().as_ptr() as *const u32, n) };
        p.iter().map(|&x| x as usize).max().unwrap_or(0)
    }

    /// #38: refuse a KV-store row that is past the layer's persistent region.
    ///
    /// `rows` is the I32 index buffer the store kernel dereferences — `cells`
    /// for a plain [`Op::KvcacheStore`], `positions` for the fused epilogue —
    /// and `n_ctx` the region's cell count. The allocator bounds the same input
    /// on the `fill_input_i32` path (`GraphAllocator::check_positions_bound`),
    /// but that is an upstream guard on one filler; the arm that indexes the
    /// persistent region is the last line before the kernel, so it owns the
    /// bound. A Metal `MTLBuffer` is `StorageModeShared`, so the small index
    /// buffer is read back here; a kernel-side range check would be a *silent*
    /// no-write, which `docs/GPU_SAFETY.md` forbids.
    fn check_kv_store_rows(
        &self,
        rows: &crate::metal::MetalBuffer,
        count: usize,
        n_ctx: usize,
        layer: usize,
    ) -> Result<(), String> {
        if count == 0 {
            return Ok(());
        }
        let idx =
            unsafe { std::slice::from_raw_parts(rows.contents().as_ptr() as *const u32, count) };
        for &cell in idx {
            if cell as usize >= n_ctx {
                return Err(format!(
                    "Metal KV store layer {layer}: cell {cell} is past the {n_ctx}-cell \
                     arena (refusing to write past the persistent region)"
                ));
            }
        }
        Ok(())
    }

    /// #40: resolve a norm node's weight `(buffer, offset)` or refuse loudly.
    ///
    /// Both `None` meanings are a missing weight, not a licence to run
    /// weightless: (a) `NormMeta::weight_name` is `None`, and (b) the name is
    /// set but `MpsState::weight_buf` has no registration for it on this device.
    /// The old arms fell through to the weightless `rms_norm` kernel in either
    /// case, turning a kernel-invariant violation into plausible-looking output
    /// from a wrong computation — the exact failure mode `docs/GPU_SAFETY.md`
    /// forbids. Both supported producers (`models/qwen2`, `models/qwen3`) always
    /// pass a weight, so no legitimate weightless path exists to preserve; this
    /// mirrors CUDA's `CudaBackend::norm_weight` (same two refusals).
    fn norm_weight(&self, node: &CNode) -> Result<(crate::metal::MetalBuffer, u64), String> {
        let name = match &node.meta {
            NodeMeta::Norm(m) => m.weight_name.as_deref(),
            other => {
                return Err(format!(
                    "metal: {} norm node missing NormMeta: {other:?}",
                    node.name
                ))
            }
        };
        let Some(name) = name else {
            return Err(format!(
                "metal: {} has no norm weight (the Metal rms_norm kernel requires one)",
                node.name
            ));
        };
        self.state
            .weight_buf(name)
            .ok_or_else(|| format!("metal: norm weight '{name}' not on GPU ({})", node.name))
    }

    fn copy_in(&self, dst: usize, src: usize) {
        // in-place-ish ops (silu/rope) may alias; snapshot to dst first
        let src_buf = self.buf(src);
        let dst_buf = self.buf(dst);
        let n = (src_buf.length().min(dst_buf.length()) / 4) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buf.contents().as_ptr() as *const f32,
                dst_buf.contents().as_ptr() as *mut f32,
                n,
            );
        }
    }

    fn print_profile() {
        let enc = OP_ENC.lock().unwrap();
        let gpu = GPU_MS.lock().unwrap();
        let mut rows: Vec<_> = enc.iter().collect();
        rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        eprintln!(
            "\n[MINFER_OP_PROFILE] host-encode per op (cumulative), GPU submits={}:",
            gpu.0
        );
        let mut enc_total = 0.0;
        for (k, (n, ms)) in rows.iter().take(20) {
            enc_total += ms;
            eprintln!("  {ms:9.3} ms  x{n:4}  {k}");
        }
        eprintln!(
            "  host encode (top20): {enc_total:.3} ms; GPU wait: {:.3} ms over {} submits",
            gpu.1, gpu.0
        );
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        // flush any pending command buffer (never leave an unterminated encoder)
        if !self.cb_ptr.is_null() {
            // SAFETY: dropping the backend — take the box back and submit.
            let cb = unsafe { Box::from_raw(self.cb_ptr) };
            self.cb_ptr = std::ptr::null_mut();
            let _ = cb.submit();
        }
        if op_profile_enabled() {
            Self::print_profile();
        }
    }
}

/// F4: the Metal capability matrix, as a free function.
///
/// This is the **authority** for the answer: the `Backend` trait method below
/// forwards to it (the function cannot take a `&self`), and assignment reads the
/// trait method. The registry does not carry a copy.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
    match op {
        Op::Input => true,
        Op::Add | Op::Mul | Op::Silu | Op::RmsNorm { .. } | Op::QkNorm { .. } | Op::SwiGLU => {
            dtype == DType::F32
        }
        Op::MatMul { .. } => {
            matches!(dtype, DType::F32) // activations are f32; weight type in meta
        }
        Op::GetRows | Op::RoPE { .. } | Op::Attn { .. } => dtype == DType::F32,
        Op::KvcacheStore { .. } | Op::KvcacheLoad { .. } => dtype == DType::F32,
        Op::FusedQKV { .. } | Op::FusedQkvNorm { .. } | Op::FusedFFN => dtype == DType::F32,
        // D1: Metal's kernels take a buffer and a length, with no element
        // offset, so it can only express an *exact* view (offset 0 == the
        // parent's own buffer, which is what the allocator maps). An offset
        // window would read the wrong bytes, so it is refused here rather
        // than silently mis-computed — the allocator backstops the partial
        // case, which `supports_op` cannot see (the parent's length is not
        // in the op). G5 is where Metal would learn offsets.
        Op::View { offset, .. } => *offset == 0,
        Op::Reshape { .. } | Op::Permute { .. } => true,
        Op::Scale(_) | Op::Softmax { .. } | Op::BatchMatMul => false,
        // Mixed-quant decode QKV epilogue (D3-8 class 2) is CUDA-only; on
        // Metal the graph builder never emits it (qkv_epilogue_ok = false
        // without `--features cuda`), so it is never assigned here.
        Op::QkvBiasRopeStore { .. } => false,
    }
}

/// F4: the fusion-pass capability, as a free function (see [`supports_op`]).
pub fn supports_fused(fused: &FusedOp) -> bool {
    // swiglu_f32 is the only fusion-pass kernel. The bias+rope+store
    // capability is a build-time fused node (FusedQKV/FusedQkvNorm), not a
    // FusionPass target, so it is not advertised here.
    matches!(fused, FusedOp::SwiGLU)
}

/// E1 `attn_span` read path (issue #44 part (a), G5a): Metal reads an explicit
/// one-range window — one `[lo, hi)` pair per query — with
/// `kernel_gqa_attn_window_f32/_f16` (`src/metal/kernels/attn_window.metal`),
/// covering `nt == 1`, `nt > 1`, both KV widths and a non-zero start.
///
/// The **other** explicit layout, `kv_map`'s `(cell, len)` runs (C8b S4), is
/// still refused: `Device::gathers_attn_map` stays false for Metal, so the model
/// builder never asks for a map and the `Op::Attn` arm backstops a map-sized
/// input with a loud `Err` rather than parsing it as spans. The write/move side
/// landed in part (b): `copy_cells` (C3 compaction), the `copy_kv_to_cpu` arm
/// (C2 shift / C5 sessions, f32-only) and the per-engine `kv_format`.
pub const SUPPORTS_ATTN_SPAN: bool = true;

/// C4: Metal addresses f32/f16 KV rows, so it does not read a packed `q8_0`
/// region; [#87] is the work that adds the kernel and flips this.
///
/// [#87]: https://github.com/yusiwen/minfer/issues/87
pub const READS_PACKED_KV: bool = false;

/// F5 ([#58], ported by [#137]): registry hook **phase A** of a cross-backend
/// staging copy out of Metal.
///
/// Metal is a **device source on unified memory**: its pool buffers are
/// `MTLBuffer`s with `StorageModeShared`, and every split's submission is closed
/// by `Backend::retire` (Metal's default `synchronize`, a bounded-wait
/// `submit`). So the transfer here is not a device→host DMA but a GPU-side blit
/// into a fresh shared staging buffer, plus an `MTLSharedEvent` the consumer
/// waits on — Metal's counterpart of CUDA's `cudaMemcpyAsync` + `cudaEventRecord`.
/// Phase A only **enqueues** (commits the dedicated command buffer and returns);
/// the host block happens once, in [`await_cross`], at the consumer's first read.
///
/// **Mechanism choice.** `MTLSharedEvent`, not `MTLEvent` and not a completion
/// handler: a completion handler can only notify the **host**, and a plain
/// `MTLEvent` cannot be waited on from the host at all (only `MTLSharedEvent`
/// exposes `waitUntilSignaledValue:timeoutMS:`) — and it is `MTLSharedEvent`
/// that `MTLCommandBuffer::encodeSignalEvent` / `encodeWaitForEvent` accept, so
/// it is the one primitive that covers *both* the host wait this port needs and
/// the device-side wait a device consumer would need. See §11.2.
///
/// Every destination other than the CPU **declines** (`Ok(false)`) and the
/// allocator's synchronous host round trip handles it, exactly as CUDA declines
/// a non-CPU destination: a Metal→Metal staging copy is unreachable
/// (`copy_across` early-returns when source and destination backends match), and
/// CUDA does not run on Apple Silicon, so no device→device pair exists on macOS.
/// The reserved device-side mechanism is named in §11.3; declining is deliberate
/// and the boundary counters record the resulting synchronous copy.
///
/// [#58]: https://github.com/yusiwen/minfer/issues/58
/// [#137]: https://github.com/yusiwen/minfer/issues/137
pub(crate) fn copy_cross(
    alloc: &mut super::alloc::GraphAllocator,
    uid: u64,
    node_id: super::NodeId,
    dst_backend: super::Backend,
) -> Result<bool, String> {
    if dst_backend != super::Backend::CPU {
        return Ok(false);
    }
    let src = alloc
        .node_buffer(node_id)
        .ok_or_else(|| format!("node {node_id} has no buffer"))?;
    if src.backend != super::Backend::METAL {
        return Err(format!(
            "metal copy_cross: node {node_id} is on {:?}, not Metal",
            src.backend
        ));
    }
    alloc
        .metal_mut()
        .ok_or("Metal backend not enabled")?
        .cross_enqueue(uid, node_id, dst_backend, src)?;
    alloc.cross_stats_mut().async_host_copies += 1;
    Ok(true)
}

/// F5 ([#58], ported by [#137]): registry hook **phase B** for a Metal source —
/// wait on the event phase A recorded, exactly once, then publish the staging
/// bytes into the destination's staging buffer.
///
/// This is the **one** host block of the async Metal device→host path, and it is
/// bounded: [`MetalBackend::cross_take`] waits `waitUntilSignaledValue` with a
/// 10 s timeout and turns a timeout or a non-`Completed` blit into a loud `Err`
/// naming the real status — never a silent read of in-flight bytes and never a
/// CPU fallback.
///
/// A no-op when this backend holds no record for the key (the pair declined
/// phase A, e.g. a device destination), matching the CUDA hook.
///
/// [#58]: https://github.com/yusiwen/minfer/issues/58
/// [#137]: https://github.com/yusiwen/minfer/issues/137
pub(crate) fn await_cross(
    alloc: &mut super::alloc::GraphAllocator,
    uid: u64,
    node_id: super::NodeId,
    dst_backend: super::Backend,
) -> Result<(), String> {
    if dst_backend != super::Backend::CPU {
        return Ok(());
    }
    let Some(data) = alloc
        .metal_mut()
        .ok_or("Metal backend not enabled")?
        .cross_take(uid, node_id, dst_backend)?
    else {
        return Ok(());
    };
    // Resolve the destination before borrowing the counters (the two borrows
    // cannot be live at once).
    let dst = alloc
        .cross_buffer(uid, node_id, dst_backend)
        .ok_or_else(|| format!("node {node_id} has no staging buffer on {dst_backend:?}"))?;
    alloc.cross_stats_mut().event_syncs += 1;
    alloc.write_cross_staging(dst, &data)
}

/// F4: this backend's registry entry (see `cpu_backend::entry`).
pub fn entry() -> super::registry::BackendEntry {
    use super::registry::{Backend as Handle, BackendCaps, BackendEntry, PRIORITY_METAL};
    BackendEntry {
        handle: Handle::METAL,
        name: "metal",
        priority: PRIORITY_METAL,
        caps: BackendCaps {
            reads_packed_kv: READS_PACKED_KV,
        },
        pool: |a| a.metal().map(|m| m as &dyn Backend),
        pool_mut: |a| a.metal_mut().map(|m| m as &mut dyn Backend),
        host_read: |a, id| a.metal().and_then(|m| m.read_host(id)).map(|s| s.to_vec()),
        // F5 (#137): Metal's async staging copy — a `MTLBlitCommandEncoder`
        // copy into a shared staging buffer plus an `MTLSharedEvent` signal
        // (phase A), waited on once at the consumer's first read (phase B).
        // See `copy_cross` / `await_cross` above for the mechanism and the
        // declined directions.
        copy_cross,
        await_cross,
        // C4 per-engine (issue #44 part (b)): the engine's stamped format, not a
        // process global — C5 records it as the session's KV element type, so a
        // session is described under the width its region really uses. Reading
        // the allocator's stamp rather than `metal().kv_format()` matters because
        // `kv_load` / `load_slots` run **before** the first forward, when the
        // Metal pool has not been enabled yet (`enable_metal` builds its format
        // from this same stamp, so the two cannot disagree). This is the CUDA
        // hook's shape (`cuda_backend::entry`).
        kv_format: |a| a.kv_format(),
        enable: |a| a.enable_metal(),
        unavailable: || {
            if metal_available() {
                None
            } else {
                Some("no Metal device, or MPS is unavailable (MINFER_DISABLE_MPS)")
            }
        },
    }
}

/// F4: register the Metal backend (macOS only).
pub fn register(registry: &mut super::registry::Registry) {
    registry.register_entry(entry());
}

impl Backend for MetalBackend {
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        supports_op(op, dtype)
    }

    fn supports_attn_span(&self) -> bool {
        SUPPORTS_ATTN_SPAN
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        supports_fused(fused)
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        if let Some(idx) = self
            .free
            .iter()
            .position(|&id| self.pool[id].length() as usize == size * 4)
        {
            return self.free.swap_remove(idx);
        }
        self.pool.push(self.state.new_f32_buffer(size));
        self.pool.len() - 1
    }

    fn free_buffer(&mut self, id: usize) {
        if !self.free.contains(&id) {
            self.free.push(id);
        }
    }

    fn pool_len(&self) -> usize {
        self.pool.len()
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // never recycled from the free list (see Backend::alloc_fresh)
        self.pool.push(self.state.new_f32_buffer(size));
        self.pool.len() - 1
    }

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[BufRef],
        out_buf: BufRef,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        let cb = self.cb();
        let _t = if op_profile_enabled() {
            Some(EncTimer {
                key: format!("{:?}", node.op),
                t0: std::time::Instant::now(),
            })
        } else {
            None
        };
        match &node.op {
            Op::Input => Ok(()),
            Op::Silu => {
                if in_bufs[0].id != out_buf.id {
                    self.copy_in(out_buf.id, in_bufs[0].id);
                }
                let n = self.pool[out_buf.id].length() as usize / 4;
                cb.silu_f32(self.buf(out_buf.id), n);
                Ok(())
            }
            Op::Add => {
                cb.add_f32(
                    self.buf(in_bufs[0].id),
                    self.buf(in_bufs[1].id),
                    self.buf(out_buf.id),
                    self.pool[out_buf.id].length() as usize / 4,
                );
                Ok(())
            }
            Op::Mul => {
                cb.mul_f32(
                    self.buf(in_bufs[0].id),
                    self.buf(in_bufs[1].id),
                    self.buf(out_buf.id),
                    self.pool[out_buf.id].length() as usize / 4,
                );
                Ok(())
            }
            Op::RmsNorm { eps } => {
                // #40: a missing norm weight is a loud refusal, never a
                // weightless fallthrough (see `MetalBackend::norm_weight`).
                let (wb, w_off) = self.norm_weight(node)?;
                let d = node.out_shape[0];
                let n = node.out_shape[1];
                // 256-thread kernel when enabled (METAL_OPTIMIZATIONS #16)
                if crate::metal::rms_norm_256_enabled() {
                    cb.rms_norm_256(
                        self.buf(in_bufs[0].id),
                        Some(&wb),
                        w_off,
                        self.buf(out_buf.id),
                        d,
                        n,
                        *eps,
                        0,
                        0,
                    );
                } else {
                    cb.rms_norm(
                        self.buf(in_bufs[0].id),
                        Some(&wb),
                        w_off,
                        self.buf(out_buf.id),
                        d,
                        n,
                        *eps,
                        0,
                        0,
                    );
                }
                Ok(())
            }
            Op::QkNorm { hd, nh, eps } => {
                // Per-head norm: contiguous [nt*nh, hd] rows — same kernel as
                // RmsNorm with d = hd and n = nt*nh (weight length hd).
                // #40: loud refusal on a missing weight, as in the RmsNorm arm.
                let (wb, w_off) = self.norm_weight(node)?;
                let d = *hd;
                let n = (self.pool[out_buf.id].length() as usize / 4) / d;
                let _ = nh;
                if crate::metal::rms_norm_256_enabled() {
                    cb.rms_norm_256(
                        self.buf(in_bufs[0].id),
                        Some(&wb),
                        w_off,
                        self.buf(out_buf.id),
                        d,
                        n,
                        *eps,
                        0,
                        0,
                    );
                } else {
                    cb.rms_norm(
                        self.buf(in_bufs[0].id),
                        Some(&wb),
                        w_off,
                        self.buf(out_buf.id),
                        d,
                        n,
                        *eps,
                        0,
                        0,
                    );
                }
                Ok(())
            }
            Op::MatMul { .. } => {
                let meta = match &node.meta {
                    NodeMeta::MatMul(m) => m,
                    other => return Err(format!("matmul node missing MatMulMeta: {other:?}")),
                };

                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.weight_name)
                    .ok_or_else(|| format!("weight '{}' not on GPU", meta.weight_name))?;
                let nt = node.out_shape[1];
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
                    meta.out_dim,
                    meta.in_dim,
                    nt,
                );
                if let Some(bname) = &meta.bias_name {
                    let (bb, b_off) = self
                        .state
                        .weight_buf(bname)
                        .ok_or_else(|| format!("bias '{}' not on GPU", bname))?;
                    cb.add_bias_f32(self.buf(out_buf.id), &bb, b_off, meta.out_dim, nt, 0);
                }
                Ok(())
            }
            Op::GetRows => {
                match &node.meta {
                    NodeMeta::Embed(m) => {
                        let (wb, w_off) = self
                            .state
                            .weight_buf(&m.weight_name)
                            .ok_or_else(|| format!("embedding '{}' not on GPU", m.weight_name))?;
                        let ne = node.out_shape[0];
                        let nt = node.out_shape[1];
                        cb.embed_tokens_gpu(
                            &wb,
                            w_off,
                            self.buf(in_bufs[0].id),
                            self.buf(out_buf.id),
                            ne,
                            nt,
                            m.weight_ttype,
                        );
                        Ok(())
                    }
                    NodeMeta::None => {
                        // generic row selection: out[t] = x[ids[t]] (n_out tail)
                        let ne = node.out_shape[0];
                        let nt = node.out_shape[1];
                        cb.get_rows_f32(
                            self.buf(in_bufs[0].id),
                            self.buf(in_bufs[1].id),
                            self.buf(out_buf.id),
                            ne,
                            nt,
                        );
                        Ok(())
                    }
                    other => Err(format!("get_rows node with unexpected meta: {other:?}")),
                }
            }
            Op::RoPE { style } => {
                let meta = match &node.meta {
                    NodeMeta::Rope(m) => m,
                    other => return Err(format!("rope node missing RoPEMeta: {other:?}")),
                };
                if in_bufs[0].id != out_buf.id {
                    self.copy_in(out_buf.id, in_bufs[0].id);
                }
                let nt = node.out_shape[1];
                cb.rope_f32(
                    self.buf(out_buf.id),
                    meta.n_head,
                    meta.hd,
                    nt,
                    meta.freq_base,
                    meta.freq_scale,
                    self.buf(in_bufs[1].id),
                    *style as i32,
                    0,
                );
                Ok(())
            }
            Op::SwiGLU => {
                let n = self.pool[out_buf.id].length() as usize / 4;
                cb.swiglu_f32(
                    self.buf(in_bufs[0].id),
                    self.buf(in_bufs[1].id),
                    self.buf(out_buf.id),
                    n,
                );
                Ok(())
            }
            Op::KvcacheStore { layer } => {
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                let nkt = node.out_shape[0];
                // Logical length, not the pool buffer's: the allocation is rounded
                // up to its size class (E4 S2), so `pool[id].length()` would
                // over-count `nt` whenever `nkt * nt` is not itself a class size
                // and the store would read past the filled `cells` into the
                // class's uninitialised tail, writing garbage rows into the arena.
                let nt = in_bufs[0].len / nkt;
                // #38: bound every row against the region the kernel writes into
                // (`node.out_shape[1]` is the `n_ctx` the region was sized with).
                self.check_kv_store_rows(self.buf(in_bufs[2].id), nt, node.out_shape[1], *layer)?;
                cb.store_kv(
                    self.buf(in_bufs[0].id),
                    self.buf(k_id),
                    nkt,
                    nt,
                    self.buf(in_bufs[2].id),
                    0,
                    self.kv_f16(),
                );
                cb.store_kv(
                    self.buf(in_bufs[1].id),
                    self.buf(v_id),
                    nkt,
                    nt,
                    self.buf(in_bufs[2].id),
                    0,
                    self.kv_f16(),
                );
                Ok(())
            }
            Op::KvcacheLoad { .. } => Ok(()), // view of the K region
            Op::Attn { explicit_span, .. } => {
                let meta = match &node.meta {
                    NodeMeta::Attn(m) => m,
                    other => return Err(format!("attn node missing AttnMeta: {other:?}")),
                };
                // GPU safety (H1): kernel_gqa_attn strides KV by nk*hd
                if meta.nkt != meta.n_head_kv * meta.hd {
                    return Err(format!(
                        "Metal attention: nkt={} != n_head_kv*hd={} (kernel_gqa_attn strides KV by nk*hd)",
                        meta.nkt, meta.n_head_kv * meta.hd
                    ));
                }
                if meta.hd != meta.hd_kv {
                    return Err(format!(
                        "Metal attention: hd={} != hd_kv={} (kernel_gqa_attn uses query head dim)",
                        meta.hd, meta.hd_kv
                    ));
                }
                let (k_id, v_id) = kv_pair
                    .ok_or_else(|| format!("KV regions for layer {} not allocated", meta.layer))?;
                let nt = node.out_shape[1];
                let k = self.buf(k_id);
                let v = self.buf(v_id);
                let q = self.buf(in_bufs[0].id);
                let o = self.buf(out_buf.id);
                // E1 (G5a): an explicit window is selected by the **size** of the
                // window input (topology, fixed at build time), mirroring CUDA's
                // arm. `attn_span` is one `[lo, hi)` pair per query (its `lo` at
                // `window[t]`, `hi` at `window[nt + t]`) and runs the windowed
                // kernel; a `kv_map`-sized input names `KV_MAP_MAX_SPANS` runs per
                // query and would be resolved to the wrong rows if read as spans
                // (risk 1), so it stays a loud refusal while `gathers_attn_map`
                // is false — never a guess, and never parsed as spans.
                if *explicit_span {
                    let win = in_bufs.get(3).ok_or_else(|| {
                        format!(
                            "Metal attention: {} is explicit-span but carries no window input",
                            node.name
                        )
                    })?;
                    if win.len == 2 * nt {
                        cb.gqa_attn_window(
                            q,
                            k,
                            v,
                            o,
                            self.buf(win.id),
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            nt,
                            self.kv_f16(),
                        );
                        return Ok(());
                    }
                    let kmax = crate::graph::kvcache::KV_MAP_MAX_SPANS;
                    if win.len == nt * kmax * 2 {
                        return Err(format!(
                            "Metal attention: {} carries a kv_map window (C8b S4), which Metal's \
                             kernel cannot gather (`Device::gathers_attn_map` is false) — backend \
                             assignment should have kept it on CPU or CUDA",
                            node.name
                        ));
                    }
                    return Err(format!(
                        "Metal attention: {}'s window input has {} values; one query needs either a \
                         single (lo, hi) pair ({}) or {} (cell, len) runs ({})",
                        node.name,
                        win.len,
                        2 * nt,
                        kmax,
                        nt * kmax * 2
                    ));
                }
                // G1: dispatch the fast attention kernels (flash / split /
                // parallel) exactly like the legacy layer_gpu path. The fast
                // paths are gated to the isolation-tested shapes (hd 64/128);
                // anything else falls back to the classic kernel. Every causal
                // path below is byte-untouched by the E1 window (G5a): only the
                // `explicit_span` branch above is new.
                let positions = self.buf(in_bufs[2].id);
                if nt == 1 {
                    if crate::metal::flash_attn_enabled(meta.hd) {
                        let chunks = self.attention_chunks(positions, nt);
                        cb.gqa_attn_flash(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            1,
                            chunks,
                            self.kv_f16(),
                        );
                    } else if (meta.hd == 64 || meta.hd == 128)
                        && !std::env::var("MINFER_NO_SPLIT_ATTN").map_or(false, |v| v == "1")
                    {
                        let chunks = self.attention_chunks(positions, nt);
                        cb.gqa_attn_split_f32(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            1,
                            chunks,
                            self.kv_f16(),
                        );
                    } else {
                        cb.gqa_attn_f32(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            1,
                            self.kv_f16(),
                        );
                    }
                } else if meta.hd == 64 || meta.hd == 128 {
                    let max_pos = Self::positions_max(positions, nt);
                    let nkv = max_pos + 1;
                    if crate::metal::prefill_flash_enabled(meta.hd) {
                        cb.attn_flash_prefill(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            nkv,
                            meta.nkt,
                            nt,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            self.kv_f16(),
                        );
                    } else if crate::metal::matmul_attn_enabled() {
                        cb.attn_parallel_prefill(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            nkv,
                            meta.nkt,
                            meta.n_head * meta.hd,
                            nt,
                            meta.n_head,
                            meta.hd,
                            meta.n_head / meta.n_head_kv,
                            meta.scale,
                        );
                    } else {
                        cb.gqa_attn_f32(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            nt,
                            self.kv_f16(),
                        );
                    }
                } else {
                    cb.gqa_attn_f32(
                        q,
                        k,
                        v,
                        o,
                        positions,
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        nt,
                        self.kv_f16(),
                    );
                }
                Ok(())
            }
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                // D1: the view is the parent's buffer; nothing to copy.
                if in_bufs[0].id != out_buf.id {
                    return Err(format!(
                        "metal: {} is a view but its output buffer is not its source's (D1 aliasing \
                         missing); refusing to copy silently",
                        node.name
                    ));
                }
                Ok(())
            }
            Op::FusedFFN => {
                // #39: the fused FFN kernel is decode-only, and `debug_assert!`
                // is compiled out in release. Shape validation is cheap and does
                // not depend on the weights, so it is the first thing checked
                // (kernel-invariant violations return Err, never assume).
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "metal: {}: FusedFFN is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::FusedFfn(m) => m,
                    other => return Err(format!("fused_ffn node missing FusedFfnMeta: {other:?}")),
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.gu_weight)
                    .ok_or_else(|| format!("gate+up weight '{}' not on GPU", meta.gu_weight))?;
                let od_total = 2 * meta.nf;
                // 1) concat matmul: x × [ffn_gate|ffn_up] → gate|up concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                // 2) swiglu in place: silu(gate rows 0..nf) * up rows nf..2*nf
                //    (llama ggml_swiglu_split); result written back to gate rows
                let n = nt * meta.nf;
                cb.swiglu_f32_off(
                    self.buf(out_buf.id),
                    self.buf(out_buf.id),
                    self.buf(out_buf.id),
                    n,
                    n,
                );
                if std::env::var("MINFER_FFNDEBUG").is_ok() {
                    self.submit_pending();
                    let nb = (self.pool[out_buf.id].length() as usize) / 4;
                    let ob = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(out_buf.id).contents().as_ptr() as *const f32,
                            nb,
                        )
                    };
                    eprintln!(
                        "[ffn-out] FusedFFN out_buf={} len={nb} first4={:?} last4={:?}",
                        out_buf.id,
                        &ob[..4],
                        &ob[nb - 4..]
                    );
                }
                Ok(())
            }
            Op::FusedQKV { layer } => {
                // #39: decode-only shape guard first — see the FusedFFN arm.
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "metal: {}: FusedQKV is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::FusedQkv(m) => m,
                    other => return Err(format!("fused_qkv node missing FusedQkvMeta: {other:?}")),
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.qkv_weight)
                    .ok_or_else(|| format!("qkv weight '{}' not on GPU", meta.qkv_weight))?;
                let od_total = meta.nqt + 2 * meta.nkt;
                // 1) concat matmul: x × [wq|wk|wv] → q|k|v concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                // 2) fused bias + rope + KV store in one kernel pass
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                // #38: the epilogue stores K/V at the row `positions` names
                // (Metal keeps the pre-C6 gate, so positions == cells here);
                // bound it against the region's cell count.
                self.check_kv_store_rows(
                    self.buf(in_bufs[1].id),
                    nt,
                    meta.kv_elems / meta.nkt.max(1),
                    *layer,
                )?;
                let bias_off =
                    |name: &Option<String>| -> Result<(crate::metal::MetalBuffer, u64), String> {
                        match name {
                            Some(n) => self
                                .state
                                .weight_buf(n)
                                .ok_or_else(|| format!("bias '{n}' not on GPU")),
                            None => Err("fused QKV bias missing".into()),
                        }
                    };
                let bq = bias_off(&meta.bias_q)?;
                let bk = bias_off(&meta.bias_k)?;
                let bv = bias_off(&meta.bias_v)?;
                let (bq_b, bq_o) = bq;
                let (bk_b, bk_o) = bk;
                let (bv_b, bv_o) = bv;
                let pos = {
                    let n = (self.buf(in_bufs[1].id).length() as usize) / 4;
                    let p = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(in_bufs[1].id).contents().as_ptr() as *const u32,
                            n,
                        )
                    };
                    p[0] as i32
                };

                cb.attn_bias_rope_store(
                    self.buf(out_buf.id),
                    &bq_b,
                    bq_o,
                    &bk_b,
                    bk_o,
                    &bv_b,
                    bv_o,
                    self.buf(k_id),
                    self.buf(v_id),
                    meta.nqt,
                    meta.nkt,
                    meta.hd,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                    meta.rope_style as i32,
                    self.kv_f16(),
                );
                Ok(())
            }
            Op::FusedQkvNorm { layer } => {
                // #39: decode-only shape guard first — see the FusedFFN arm.
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "metal: {}: FusedQkvNorm is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::FusedQkvNorm(m) => m,
                    other => {
                        return Err(format!(
                            "fused_qkv_norm node missing FusedQkvNormMeta: {other:?}"
                        ))
                    }
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.qkv_weight)
                    .ok_or_else(|| format!("qkv weight '{}' not on GPU", meta.qkv_weight))?;
                // per-head Q/K RMSNorm weights (llama attn_q_norm / attn_k_norm)
                let norm_off =
                    |name: &Option<String>| -> Result<(crate::metal::MetalBuffer, u64), String> {
                        match name {
                            Some(n) => self
                                .state
                                .weight_buf(n)
                                .ok_or_else(|| format!("norm '{n}' not on GPU")),
                            None => Err("fused QKV norm weight missing".into()),
                        }
                    };
                let (qn_b, qn_o) = norm_off(&meta.q_norm_name)?;
                let (kn_b, kn_o) = norm_off(&meta.k_norm_name)?;
                let od_total = meta.nqt + 2 * meta.nkt;
                // 1) concat matmul: x × [wq|wk|wv] → q|k|v concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                // #38: same store bound as the plain/fused QKV arms — the
                // no-bias epilogue writes the KV row `positions` names.
                self.check_kv_store_rows(
                    self.buf(in_bufs[1].id),
                    nt,
                    meta.kv_elems / meta.nkt.max(1),
                    *layer,
                )?;
                // 2) per-head RMSNorm on q/k IN PLACE on the concat buffer
                //    (llama build_norm(Qcur/Kcur, attn_q_norm/attn_k_norm) before
                //    ggml_rope_ext). q section is at byte offset 0; k section at
                //    byte offset nqt*4. Reuse the rms_norm_256 kernel (d=hd rows).
                let off_q = 0u64;
                let off_k = (meta.nqt * 4) as u64;
                let n_q = meta.nh;
                let n_k = meta.nk;
                if crate::metal::rms_norm_256_enabled() {
                    cb.rms_norm_256(
                        self.buf(out_buf.id),
                        Some(&qn_b),
                        qn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_q,
                        meta.eps,
                        off_q,
                        off_q,
                    );
                    cb.rms_norm_256(
                        self.buf(out_buf.id),
                        Some(&kn_b),
                        kn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_k,
                        meta.eps,
                        off_k,
                        off_k,
                    );
                } else {
                    cb.rms_norm(
                        self.buf(out_buf.id),
                        Some(&qn_b),
                        qn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_q,
                        meta.eps,
                        off_q,
                        off_q,
                    );
                    cb.rms_norm(
                        self.buf(out_buf.id),
                        Some(&kn_b),
                        kn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_k,
                        meta.eps,
                        off_k,
                        off_k,
                    );
                }
                // 3) no-bias rope + KV store (q in place, k rope+store, v store)
                let pos = {
                    let n = (self.buf(in_bufs[1].id).length() as usize) / 4;
                    let p = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(in_bufs[1].id).contents().as_ptr() as *const u32,
                            n,
                        )
                    };
                    p[0] as i32
                };
                cb.attn_rope_store(
                    self.buf(out_buf.id),
                    self.buf(k_id),
                    self.buf(v_id),
                    meta.nqt,
                    meta.nkt,
                    meta.hd,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                    meta.rope_style as i32,
                    self.kv_f16(),
                );
                Ok(())
            }
            Op::Scale(_) | Op::Softmax { .. } | Op::BatchMatMul => {
                Err(format!("op {:?} unsupported on Metal (Phase 3)", node.op))
            }
            // CUDA-only mixed-quant decode epilogue; never emitted on the Metal
            // graph (builder sets qkv_epilogue_ok = false without --features
            // cuda), so reaching here is a scheduling invariant violation.
            Op::QkvBiasRopeStore { .. } => {
                Err(format!("op {:?} unsupported on Metal (CUDA-only)", node.op))
            }
        }
    }

    /// C3 (issue #44 part (b)): move KV rows inside one arena on the device.
    ///
    /// One `MTLBlitCommandEncoder` copy per row, in the overlap-safe order
    /// (ascending when the run slides down, descending when it slides up — the
    /// same `dst_row <= src_row` branch CUDA's `kv_move_rows` takes), encoded
    /// into the **current** command buffer and submitted once. A separate
    /// submission would overlap the producer split (#137); `self.cb()` reuses the
    /// open one when there is one, otherwise opens exactly one.
    fn copy_cells(
        &mut self,
        dst: BufRef,
        src: BufRef,
        dst_row: usize,
        src_row: usize,
        rows: usize,
        elems_per_cell: usize,
    ) -> Result<(), String> {
        if dst.id != src.id {
            return Err(format!(
                "metal: copy_cells moves cells within one arena ({} -> {})",
                src.id, dst.id
            ));
        }
        if rows == 0 || dst_row == src_row {
            // A zero-row plan entry or a same-row move: nothing to copy (the
            // overlapping blit would also be undefined for a same-row move).
            return Ok(());
        }
        // An f16 region addresses `half[cell * nkt]` while `elems_per_cell` is
        // counted in f32 words (the unit the caller passes): a cell is `nkt / 2`
        // f32 words apart. CUDA's `copy_cells` halves the same stride; a Q8_0
        // cell is already whole words including padding and is passed through
        // unchanged (the caller passes `region.elems / n_ctx`), but Metal refuses
        // Q8_0 (`READS_PACKED_KV = false`) so only the f16 branch is live here.
        let elems_per_cell = if self.kv_f16() {
            (elems_per_cell / 2).max(1)
        } else {
            elems_per_cell
        };
        let row_bytes = elems_per_cell * 4;
        let src_buf = self.buf(src.id).clone();
        let dst_buf = self.buf(dst.id).clone();
        self.cb().encode_move_rows(
            &src_buf,
            &dst_buf,
            src.offset * 4,
            dst.offset * 4,
            src_row,
            dst_row,
            rows,
            row_bytes,
        )?;
        self.submit_pending();
        Ok(())
    }

    fn read_host(&self, id: usize) -> Option<&[f32]> {
        // F5 (#137): the synchronous staging path reaches a Metal source through
        // here (the registry entry's `host_read` hook), so this is the
        // device-level readback counter. The async path publishes its own
        // staging bytes in `cross_take` and never comes through here.
        self.sync_readbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let buf = self.pool.get(id)?;
        let len = (buf.length() as usize) / 4;
        Some(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) })
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let buf = self.pool.get(id).ok_or_else(|| format!("no buffer {id}"))?;
        let len = (buf.length() as usize) / 4;
        if len != data.len() {
            return Err(format!(
                "buffer {id}: expected {len} elements, got {}",
                data.len()
            ));
        }
        self.write_host_window(id, 0, data)
    }

    /// E4 S2: a pooled activation buffer is rounded to its size class, so a node
    /// writes its logical window into a buffer that may be longer.
    fn write_host_window(&mut self, id: usize, offset: usize, data: &[f32]) -> Result<(), String> {
        let buf = self.pool.get(id).ok_or_else(|| format!("no buffer {id}"))?;
        let len = (buf.length() as usize) / 4;
        let end = offset.checked_add(data.len()).ok_or_else(|| {
            format!(
                "buffer {id}: offset {offset} + {} elements overflows",
                data.len()
            )
        })?;
        if end > len {
            return Err(format!(
                "buffer {id}: writing {} elements at offset {offset} runs past the {len}-element pool buffer",
                data.len()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (buf.contents().as_ptr() as *mut f32).add(offset),
                data.len(),
            );
        }
        Ok(())
    }

    fn synchronize(&mut self) {
        self.submit_pending();
    }
}

/// The Metal backend exists only where the trait sees it; helper for the
/// allocator to know whether GPU is available.
pub fn metal_available() -> bool {
    MpsState::get().is_some()
}

#[cfg(test)]
mod tests;
