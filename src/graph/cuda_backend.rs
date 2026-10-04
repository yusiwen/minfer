//! CUDA graph backend (Phase 7).
//!
//! Wraps the [`crate::cuda::CudaState`] singleton in the graph [`Backend`]
//! trait contract (`src/graph/backend.rs`), mirroring `metal_backend.rs` where
//! the mechanics allow: a device buffer pool with a byte-length free list,
//! name → device-pointer weight resolution, sync H2D/D2H host transfers, and
//! per-op kernel dispatch on the shared stream. Design + rollout:
//! `docs/CUDA-BACKEND-DESIGN.md`.

use super::alloc::GraphAllocator;
use super::backend::Backend;
use super::ops::{FusedOp, NodeMeta, Op};
use super::{Backend as BackendTag, BufRef, CNode, DType, NodeId};
use crate::vec_ops::RopeStyle;

struct CudaBuf {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
}

pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,
    /// Issue #188: **this backend's own device stream**. Every launch, copy,
    /// event, capture/replay and synchronize this backend issues is bound to
    /// this stream for the duration of the operation
    /// (`crate::cuda::bind_stream`), so two engines in one process run — and
    /// capture — concurrently instead of sharing `CudaState`'s process-wide
    /// stream and its one capture window. Created `cudaStreamNonBlocking` so it
    /// does not join the legacy default stream's implicit global
    /// synchronization, which is what let one engine's host-side `cudaMemcpy`
    /// invalidate another engine's capture.
    ///
    /// It is freed in [`Drop`], after the pool, the scratches and the captured
    /// graphs are released (a `cudaFree` on a stream that still has queued work
    /// is ordered by the driver, but destroying the stream last keeps the
    /// teardown order obvious).
    stream: *mut std::ffi::c_void,
    /// 8b / C4 S2b / #153: the KV cache **layout** as a `crate::cuda::KV_LAYOUT_*` code
    /// (f32 / f16 / q8_0). It is **per engine**: the loaded model's resolved
    /// `KvFormat` reaches this backend through `GraphAllocator::set_kv_format` (the
    /// same stamp the CPU kernels get), so two engines with different formats hold
    /// different tags in one process. The tag is what selects the store, the
    /// attention kernel, the `copy_cells` stride **and** the captured-graph
    /// identity, so a packed region can never be addressed as f32 rows and a graph
    /// captured for one layout never replays for another.
    kv_layout: i32,
    pool: Vec<CudaBuf>,
    free: Vec<usize>,
    /// Bumped on every pool allocation (fresh or free-list reuse). A captured
    /// CUDA Graph (7d) is only valid while the node → device-pointer mapping
    /// is unchanged, and any alloc_buffer() call may change it.
    pool_gen: u64,
    /// Device scratch holding raw-int32 positions decoded from the f32-bits
    /// I32 input buffers (grown on demand; freed in Drop alongside the pool).
    /// i32 staging, **one buffer per converted source input** (fix, 2026-09-19).
    ///
    /// A single shared scratch was wrong. Inside one execution window a graph
    /// converts `positions` for the KV store and `attn_span` for attention; the
    /// second conversion overwrote the first *before its launch had run*, so the
    /// store wrote rows derived from the span's `lo` values instead of the
    /// positions. That is invisible when they coincide — which is exactly why the
    /// degenerate E1b fixture (`positions` `[0, 2]` == the span's `lo` block) and
    /// every causal graph passed, while a windowed graph with a real prompt
    /// produced garbage (plan §14). Keyed by source buffer id; the buffers are
    /// reused (a tiny pool) and re-converted every window, so a stale value can
    /// never be read.
    pos_scratch: std::collections::HashMap<usize, (*mut std::ffi::c_void, usize)>,
    /// D3-7 2c: one-execution-window memo for positions_i32 — (input buf id,
    /// pool_gen) of the last conversion. Every Rope/KvcacheStore/Attn node
    /// re-converted the same positions buffer (240 launches/step at 14B
    /// decode, ~0.28 ms of pure launch overhead); the i32 result is a pure
    /// function of the input content, identical for all consumers within one
    /// serial execution pass. Cleared in synchronize() next to the MmqCache
    /// clear (same split-boundary reuse-of-pool-ids lifecycle).
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
    /// `MINFER_NO_CUDA_GRAPH=1` (at construction) or a capture failure
    /// (session-wide) force the plain direct-launch path.
    graphs_mode: GraphMode,
    /// 8g②: prefill capture gate. R3-B (2026-08-31): DEFAULT ON — repeated
    /// identical-nt prefills (server/slot scenario) capture after the usual
    /// 3-run protocol; a one-shot CLI prefill never reaches 3 runs and pays
    /// nothing. `MINFER_NO_PREFILL_CAPTURE=1` restores the old default-off
    /// (`MINFER_CAPTURE_PREFILL=1` is now redundant but still accepted).
    prefill_capture: bool,
    /// #218: how many direct-launch warmup runs of a `(uid, range)` key happen
    /// before capture opens (default 3, llama.cpp's protocol). This is
    /// **load-bearing beyond the capture heuristic**: the prefill-GEMM >48 KiB
    /// dynamic-smem attribute is set lazily on an instantiation's first launch
    /// and cached, so the warmup is what guarantees that
    /// `cudaFuncSetAttribute` runs **outside** every capture window
    /// (`docs/CUDA-BACKEND-DESIGN.md` §2.4). Do not lower it without reading
    /// that section.
    ///
    /// Test-only seam: `MINFER_TEST_CAPTURE_WARMUP=<n>` lowers it so a gate can
    /// drive the opt-in into the window on purpose (the #218 mutation arm). A
    /// non-test build always uses 3.
    capture_warmup: u32,
    /// Viz/trace capture staging: async D2H of captured node outputs queued
    /// right after each node's launch (stream-ordered — pool buffers recycle
    /// intra-split), drained with one sync at the split boundary. Replaces
    /// the per-node `copy_to_host` full-stream sync in the scheduler.
    cap: crate::cuda::CaptureStaging,
    /// F5 (#58): pinned host slabs used as the intermediate of an **asynchronous**
    /// cross-backend D2H staging copy. D2H has no device-side destination, so the
    /// transfer always lands in host memory, and a pageable destination would make
    /// the driver bounce through its own pinned buffer and block — exactly what F5
    /// removes. Slabs are grown on demand and reused through `in_use`, so a
    /// steady-state boundary allocates nothing.
    cross_slabs: Vec<CrossSlab>,
    /// F5: the cross-backend staging copies this backend has **enqueued but not
    /// yet waited on**. The allocator's phase B (`await_cross`) drains them; a
    /// record left here past the boundary is the missing-wait bug, and the
    /// allocator's pending map turns the consumer's next read into a loud error.
    cross_pending: Vec<CrossPending>,
    /// F5: how many **blocking** device→host readbacks (`copy_to_host`) this
    /// backend performed. Device tests read it to prove the async boundary path
    /// issues none; it is the device-level twin of
    /// `copystats::CrossCopyStats::blocking_host_copies`. An atomic because the
    /// read path takes `&self` (the trait's `read_host` shape).
    blocking_readbacks: std::sync::atomic::AtomicU64,
    /// Issue #185: how many times **this backend** blocked the host on a whole
    /// stream (`CudaState::sync`). Per instance, never process-wide: the F5
    /// gates compare the async arm's stalls against the synchronous arm's, and
    /// under a parallel harness a *foreign* test's syncs landed between the two
    /// snapshots (run A: the async arm read 4160 stalls against the synchronous
    /// arm's 728 — the harness's own load, not the async path). Same hazard and
    /// same fix as `blocking_readbacks`, which was per-instance from the start.
    stream_syncs: std::sync::atomic::AtomicU64,
    /// #138: the high-water mark of [`Self::cross_slabs`] entries in use at once
    /// — i.e. how many cross-backend copies were **enqueued but not yet waited
    /// on** simultaneously. This is the device-side measurement of the ticket's
    /// overlap claim: a boundary that waits per input (F5) cannot exceed 1,
    /// because the wait releases the slab before the next copy takes one, while
    /// the deferred boundary reaches the number of staged inputs. Read only by
    /// the multi-input gate, so it is `#[cfg(test)]`-accessed like the other
    /// device counters.
    cross_inflight_peak: usize,
}

/// F5: one pinned host slab of the async D2H staging pool (see
/// [`CudaBackend::cross_slabs`]).
struct CrossSlab {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
    in_use: bool,
}

/// F5: one in-flight cross-backend staging copy, device → host.
struct CrossPending {
    uid: u64,
    node: NodeId,
    /// The destination staging buffer — the host-side buffer the consumer reads.
    dst: BufRef,
    /// Index into [`CudaBackend::cross_slabs`] holding this copy's bytes.
    slab: usize,
    bytes: usize,
    /// Recorded after the `cudaMemcpyAsync`; waited on by `await_cross`.
    event: *mut std::ffi::c_void,
}

/// An instantiated CUDA Graph exec with its capture identity.
///
/// #153 added `kv_layout`: the recorded kernels were instantiated for one layout
/// (`store_kv_f16` vs `store_kv_q8_0`, the layout-tagged attention), so replaying the
/// exec for a backend whose tag changed would run the wrong kernel over the regions.
/// The lookup refuses a mismatched tag exactly like a changed `pool_gen` — destroy the
/// exec and re-warm — so the identity is (uid, range, pool_gen, kv_layout).
struct CapturedGraph {
    exec: *mut std::ffi::c_void,
    uid: u64,
    range: (usize, usize),
    pool_gen: u64,
    kv_layout: i32,
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
    /// #153: build a backend whose kernels address the KV regions in `layout`
    /// (`crate::cuda::KV_LAYOUT_*`). `GraphAllocator::enable_cuda` derives it from
    /// the engine's resolved `KvFormat` (`cuda::layout_of`), so the tag is per
    /// engine and never a process global.
    pub fn with_layout(kv_layout: i32) -> Option<Self> {
        let state = crate::cuda::CudaState::get()?;
        // Issue #188: the backend's own stream. A device that cannot give us one
        // gives us no backend (every path here is stream-scoped).
        let stream = state.create_stream();
        if stream.is_null() {
            return None;
        }
        let graphs_mode = if std::env::var("MINFER_NO_CUDA_GRAPH").as_deref() == Ok("1") {
            GraphMode::Disabled
        } else {
            GraphMode::Enabled
        };
        let prefill_capture = std::env::var("MINFER_NO_PREFILL_CAPTURE").as_deref() != Ok("1");
        // #218: the warmup count is 3 in every real build; the `MINFER_TEST_*`
        // seam exists only so a gate can force capture onto the first run (which
        // drives the >48 KiB smem opt-in into the window — the mutation arm).
        #[cfg(test)]
        let capture_warmup = std::env::var("MINFER_TEST_CAPTURE_WARMUP")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .filter(|n| *n >= 1)
            .unwrap_or(3);
        #[cfg(not(test))]
        let capture_warmup = 3u32;
        Some(Self {
            state,
            stream,
            pool: Vec::new(),
            free: Vec::new(),
            pool_gen: 0,
            pos_scratch: std::collections::HashMap::new(),
            graph_execs: Vec::new(),
            graph_runs: std::collections::HashMap::new(),
            capturing: None,
            graphs_mode,
            prefill_capture,
            capture_warmup,
            kv_layout,
            cap: crate::cuda::CaptureStaging::new(),
            cross_slabs: Vec::new(),
            cross_pending: Vec::new(),
            blocking_readbacks: std::sync::atomic::AtomicU64::new(0),
            stream_syncs: std::sync::atomic::AtomicU64::new(0),
            cross_inflight_peak: 0,
        })
    }

    /// Issue #188: bind this backend's stream to the calling thread for the
    /// lifetime of the returned guard. Every device operation this backend
    /// performs starts with `let _bound = self.bind();`, after which
    /// `CudaState::stream()` (and everything built on it: launchers,
    /// `cudaMemcpyAsync` staging, events, capture/replay, `synchronize`) answers
    /// with **this instance's** stream.
    fn bind(&self) -> crate::cuda::StreamBinding {
        crate::cuda::bind_stream(self.stream)
    }

    /// The stream this backend issues its device work on (test introspection +
    /// the #188 concurrency gate's "two engines really do hold two streams" arm).
    /// Test-only (#238): driven by `graph::cuda_backend::tests::capture::cuda_capture_abort_on_error`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn device_stream(&self) -> *mut std::ffi::c_void {
        self.stream
    }

    /// Pool generation counter (CUDA Graph replay invalidation, Phase 7d).
    /// Test-only (#238): driven by `graph::alloc::tests::a_rebuild_does_not_touch_the_device_pool`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn pool_gen(&self) -> u64 {
        self.pool_gen
    }

    /// Number of captured graphs currently held (test introspection).
    #[cfg(test)]
    fn captured_count(&self) -> usize {
        self.graph_execs.len()
    }

    /// The layouts of the held execs, oldest first (test introspection for #153's
    /// captured-graph identity).
    #[cfg(test)]
    fn captured_layouts(&self) -> Vec<i32> {
        self.graph_execs.iter().map(|g| g.kv_layout).collect()
    }

    /// #153: set the KV layout this backend's kernels address. The allocator calls it
    /// on every forward with the engine's resolved format (idempotent in steady
    /// state), so the tag follows the engine rather than a process global.
    ///
    /// A **change** invalidates every captured graph: each exec was instantiated for
    /// the old tag (`store_kv_f16` vs `store_kv_q8_0`, the layout-tagged attention) and
    /// reporting it up to date would replay the wrong kernel. The warmup counter is
    /// dropped too, so the next runs re-warm and re-capture under the new tag — the
    /// [`CapturedGraph::kv_layout`] check in `graph_replay_step` is the second line of
    /// defence.
    pub fn set_kv_layout(&mut self, layout: i32) {
        if self.kv_layout == layout {
            return;
        }
        self.kv_layout = layout;
        self.graph_execs.clear();
        self.graph_runs.clear();
    }

    /// The `KvFormat` this backend's regions store (the registry's `kv_format`
    /// hook, C5's session header). The tag is set from the engine's resolved format,
    /// so this is the engine's answer, not a process-wide one.
    /// Test-only (#238): driven by `graph::alloc::tests::set_kv_format_stamps_the_cuda_layout_per_engine`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn kv_format(&self) -> super::kvformat::KvFormat {
        crate::cuda::format_of(self.kv_layout)
    }

    #[cfg(test)]
    /// 8b: flip the per-instance KV element type (device tests exercise both
    /// f32/f16 layouts in one process).
    pub(crate) fn set_kv_f16_for_test(&mut self, f16: bool) {
        self.set_kv_layout(if f16 {
            crate::cuda::KV_LAYOUT_F16
        } else {
            crate::cuda::KV_LAYOUT_F32
        });
    }

    #[cfg(test)]
    /// C4 S2b: the packed layout, per instance (same reason as
    /// [`Self::set_kv_f16_for_test`]).
    pub(crate) fn set_kv_q8_for_test(&mut self) {
        self.set_kv_layout(crate::cuda::KV_LAYOUT_Q8_0);
    }

    #[cfg(test)]
    /// Any layout, per instance — the three-way gates (`cuda_map_window_*`) sweep
    /// f32/f16/q8_0 through one backend.
    pub(crate) fn set_kv_layout_for_test(&mut self, layout: i32) {
        self.set_kv_layout(layout);
    }

    /// The KV layout tag this backend addresses its regions in.
    pub(crate) fn kv_layout(&self) -> i32 {
        self.kv_layout
    }

    /// Bytes one KV cell occupies on this backend's device region — the stride
    /// every kernel is given. f32 is `nkt * 4`, f16 `nkt * 2`, Q8_0 the word-padded
    /// packed width (`KvFormat::Q8_0.row_bytes(nkt)`).
    fn kv_row_bytes(&self, nkt: usize) -> usize {
        use super::kvformat::KvFormat;
        match self.kv_layout {
            crate::cuda::KV_LAYOUT_Q8_0 => KvFormat::Q8_0.row_bytes(nkt),
            crate::cuda::KV_LAYOUT_F16 => nkt * 2,
            _ => nkt * 4,
        }
    }

    /// #144 item 1: the fused decode QKV epilogue, dispatched on the KV layout.
    /// f32/f16 write one element per thread (`attn_bias_rope_store`); a packed
    /// cell needs whole Q8_0 blocks, so it takes the block-owning kernel
    /// (`attn_bias_rope_store_q8_0`), which quantizes with the store's own
    /// quantizer and writes the same bytes the unfused chain does.
    #[allow(clippy::too_many_arguments)]
    fn fused_qkv_epilogue(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        bq: *mut std::ffi::c_void,
        bk: *mut std::ffi::c_void,
        bv: *mut std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
        cells: *mut std::ffi::c_void,
    ) {
        if self.kv_layout == crate::cuda::KV_LAYOUT_Q8_0 {
            self.state.attn_bias_rope_store_q8_0(
                q,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                bq,
                bk,
                bv,
                kv_k,
                kv_v,
                nqt,
                nkt,
                hd,
                freq_base,
                freq_scale,
                positions,
                cells,
                self.kv_row_bytes(nkt),
            );
        } else {
            self.state.attn_bias_rope_store(
                q,
                k,
                v,
                bq,
                bk,
                bv,
                kv_k,
                kv_v,
                nqt,
                nkt,
                hd,
                freq_base,
                freq_scale,
                positions,
                cells,
                self.kv_layout,
            );
        }
    }

    /// 8g②: turn on deliberate prefill capture for tests (the pp16/pp300
    /// bit-parity harness is the validation gate).
    #[cfg(test)]
    pub(crate) fn set_prefill_capture_for_test(&mut self, enabled: bool) {
        self.prefill_capture = enabled;
    }

    #[cfg(test)]
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
        let _bound = self.bind();
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
            if self.graph_execs[pos].pool_gen != self.pool_gen
                || self.graph_execs[pos].kv_layout != self.kv_layout
            {
                // pool churned since capture (pointers may differ), or the KV
                // layout moved (the recorded kernels were instantiated for the old
                // tag: `store_kv_f16` vs `store_kv_q8_0`, the layout-tagged
                // attention) — replaying either would run the wrong kernel.
                // Re-capture.
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
        // 8g①: capture decode-shaped graphs by default (nt_hint None = no
        // matmul in the graph, synthetic tests). Prefill-shaped graphs
        // (nt > 1) capture only with the prefill_capture gate: 8g② made
        // that a deliberate opt-in after an audit caught unvalidated
        // capture; R3-B (2026-08-31) flips the default ON — the 3-run
        // protocol bounds the cost, A1 made the real prefill graph a
        // single split, and the pp16/pp300 bit-parity harness validated
        // replay at real-prefill scale. MINFER_NO_PREFILL_CAPTURE=1 opts
        // out (8g① semantics).
        //
        // #218 — the warmup count is **load-bearing**, not just a cost bound.
        // The prefill-GEMM >48 KiB dynamic-smem attribute is set lazily on an
        // instantiation's first launch and cached per instantiation
        // (`gemm_smem_optin`); the design's stated invariant is that the
        // attribute is in force *before* a window opens and is never set
        // *inside* one (`docs/CUDA-BACKEND-DESIGN.md` §2.4). Three things hold
        // it up, and changing any of them is a design change:
        //   1. `capture_warmup` (default 3) means the first launch of an
        //      instantiation happens on an uncaptured run — the one
        //      `cudaFuncSetAttribute` lands there;
        //   2. `graph_begin_capture` opens in `cudaStreamCaptureModeThreadLocal`
        //      (#188's measured choice), so a foreign thread's driver call
        //      cannot belong to this window;
        //   3. the per-instantiation cache means the in-window launch re-reads
        //      the answer instead of asking the driver again.
        // `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` pins
        // (1)+(3) for a real >48 KiB captured prefill; `MINFER_TEST_CAPTURE_WARMUP=1`
        // is the test-only seam that drives the opt-in into the window and turns
        // that gate red.
        if *runs >= self.capture_warmup
            && self.capturing.is_none()
            && nt_hint.map_or(true, |nt| nt == 1 || self.prefill_capture)
        {
            // Issue #188: no stream lock. The capture window is opened on THIS
            // backend's own stream, and the mode is thread-local, so no other
            // backend's work can be recorded into this graph or invalidate it.
            if self.state.graph_begin_capture() {
                self.capturing = Some(key);
            } else {
                eprintln!("CUDA: stream capture unavailable; graphs disabled for this session");
                self.graphs_mode = GraphMode::Disabled;
            }
        }
        false
    }

    /// Issue #188: stream-work serialization is gone. Every backend holds its
    /// own stream, so there is no shared stream to exclude another backend from.
    /// Kept as a `None`-returning shim so the historical call sites read as the
    /// no-op they now are; a caller must not rely on mutual exclusion here.
    fn stream_guard(&self) -> Option<()> {
        None
    }

    /// Close an open capture window (instantiate + launch once + cache), or
    /// fall back to a plain synchronize. Called at split boundaries (`block =
    /// false` since #138 — see the trait's `retire`) and after the last split
    /// (`block = true`) — never inside a capture window.
    ///
    /// `block` is the only difference between the boundary close and the drain:
    /// the capture must be *ended and launched* either way (capture records
    /// launches without executing them, and the copies that follow it on the same
    /// stream are ordered behind the launch), while the `cudaStreamSynchronize`
    /// that makes the host wait for it is exactly what #138 defers to the
    /// consumer's first read. The failure arm keeps its sync in both modes: the
    /// step's outputs are undefined there, and a caller that has just lost a graph
    /// should see the error before it reads them.
    fn close_capture_or_sync(&mut self, block: bool) {
        if let Some(key) = self.capturing.take() {
            let _bound = self.bind();
            let exec = self.state.graph_end_capture_to_exec();
            let ok = !exec.is_null() && self.state.graph_launch_exec(exec);
            if ok {
                self.graph_execs.push(CapturedGraph {
                    exec,
                    uid: key.0,
                    range: key.1,
                    pool_gen: self.pool_gen,
                    kv_layout: self.kv_layout,
                });
                if block {
                    self.state_sync();
                }
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
                self.state_sync();
                // NOTE: there is no poisoned-error mechanism — later steps
                // run direct-launch with graphs disabled; this step's outputs
                // were undefined and are consumed as-is. (Phase 8 review:
                // the old comment claimed otherwise.)
            }
            return;
        }
        if block {
            self.state_sync();
        }
    }

    /// D1: a reference's device pointer with its window applied. D1 views are
    /// `F32` (the allocator refuses anything else), so the element offset is
    /// scaled by 4 bytes.
    ///
    /// F5: also the F5 registry hooks' way to resolve a source pointer
    /// (`cuda_backend::copy_cross`) before they borrow the pool mutably.
    pub(crate) fn ptr_of_ref(&self, r: BufRef) -> Result<*mut std::ffi::c_void, String> {
        let base = self.ptr_of(r.id)?;
        if r.offset == 0 {
            return Ok(base);
        }
        Ok(unsafe { (base as *mut u8).add(r.offset * 4) as *mut std::ffi::c_void })
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
        let _bound = self.bind();
        let b = self.pool.get(id)?;
        if b.ptr.is_null() || b.bytes == 0 {
            return None;
        }
        let _sg = self.stream_guard();
        let mut out = vec![0f32; b.bytes / 4];
        self.blocking_readbacks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.state_sync();
        let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, b.bytes) };
        // R3-A2: read through the pinned staging buffer (pageable-memcpy
        // bounce removed); MINFER_NO_PINNED_READBACK=1 reverts.
        self.state.copy_from_device_pinned(b.ptr, dst);
        Some(out)
    }

    /// F5: the number of **blocking** device→host readbacks this backend has
    /// performed (`copy_to_host`, which syncs the stream and then copies). The
    /// hot path must issue none of these; the values that legitimately do are the
    /// enumerated host-visible ones (logits, KV session save, debug dumps,
    /// capture fallbacks) — see `docs/BACKEND-REGISTRY-DESIGN.md` §11.
    /// (The F5 gates are tests, so the production build has no caller.)
    /// Test-only (#238): driven by `graph::cuda_backend::tests::staging::a_split_graph_waits_once_per_staged_copy_and_stays_bitwise` and `models::qwen2::graph::tests::offload_copy::async_cross_copies_never_block_and_stay_bitwise_identical`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn blocking_readback_count(&self) -> u64 {
        self.blocking_readbacks
            .load(std::sync::atomic::Ordering::Relaxed)
    }

    /// #138: the high-water mark of cross-backend copies enqueued but not yet
    /// waited on (see [`Self::cross_inflight_peak`]). Test-only (#138): driven by
    /// `graph::cuda_backend::tests::staging::a_boundary_with_several_staged_inputs_defers_its_waits`;
    /// `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn cross_inflight_peak(&self) -> usize {
        self.cross_inflight_peak
    }

    /// #138: restart the in-flight high-water mark from the number of slabs
    /// currently in use, so one backend can measure two copy disciplines in a row
    /// (the gate drives the F5 enqueue-then-wait order by hand after the
    /// scheduler's deferred run). Test-only (#138), same gate as
    /// [`Self::cross_inflight_peak`].
    #[cfg(test)]
    pub(crate) fn reset_cross_inflight_peak(&mut self) {
        self.cross_inflight_peak = self.cross_slabs.iter().filter(|s| s.in_use).count();
    }

    /// Issue #185: this backend's own count of full-stream host stalls
    /// (`CudaState::sync`, the only `cudaStreamSynchronize` in the device layer).
    ///
    /// The F5 gates read this held per instance: a delta around one workload is
    /// then attributable to that workload, whereas a process-wide total would
    /// move whenever *any* other thread syncs (the parallel harness ran the F5
    /// gate beside ~27 other device tests). The process-wide counter this
    /// replaced was deleted in [#242].
    /// Test-only (#238): driven by `graph::cuda_backend::tests::capture::stream_sync_counts_are_per_backend_not_process_wide`; `#[cfg(test)]` keeps it out of production builds.
    ///
    /// [#242]: https://github.com/yusiwen/minfer/issues/242
    #[cfg(test)]
    pub(crate) fn stream_sync_count(&self) -> u64 {
        self.stream_syncs.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Issue #185: `CudaState::sync` with this backend's stall counter bumped in
    /// the same call, so the count and the stall cannot drift apart. Every sync
    /// this backend performs goes through here.
    fn state_sync(&self) {
        let _bound = self.bind();
        self.stream_syncs
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.state.sync();
    }

    /// F5: enqueue the **asynchronous** device→host staging copy of one node's
    /// output. Returns after queueing the transfer and recording its event; the
    /// bytes are not valid until [`Self::take_cross`] waits on that event.
    ///
    /// `src` is the source buffer's device pointer and `bytes` its length in
    /// bytes; `dst` is the destination backend's staging reference (the host
    /// buffer the consumer reads).
    fn enqueue_cross_host(
        &mut self,
        uid: u64,
        node: NodeId,
        dst: BufRef,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> Result<(), String> {
        let _bound = self.bind();
        if bytes == 0 {
            return Err(format!(
                "cuda: node {node} has an empty staging copy (0 bytes)"
            ));
        }
        let _sg = self.stream_guard();
        let slab = self.take_cross_slab(bytes)?;
        let ptr = self.cross_slabs[slab].ptr;
        self.state.copy_to_host_async(src, ptr, bytes)?;
        // The event is recorded *after* the copy on the same stream, so waiting
        // on it is exactly "the transfer has landed" — never an earlier point.
        let event = self.state.record_event()?;
        self.cross_pending.push(CrossPending {
            uid,
            node,
            dst,
            slab,
            bytes,
            event,
        });
        Ok(())
    }

    /// F5: wait on a pending cross copy's event and hand its bytes back, then
    /// release the slab and the event. `None` when this backend has no pending
    /// copy for `(uid, node, dst)` — the pair took the allocator's synchronous
    /// path, so its phase B is a no-op.
    fn take_cross(
        &mut self,
        uid: u64,
        node: NodeId,
        dst: BackendTag,
    ) -> Result<Option<Vec<f32>>, String> {
        let _bound = self.bind();
        let Some(pos) = self
            .cross_pending
            .iter()
            .position(|p| p.uid == uid && p.node == node && p.dst.backend == dst)
        else {
            return Ok(None);
        };
        let rec = self.cross_pending.remove(pos);
        let _sg = self.stream_guard();
        let r = self.state.wait_event(rec.event);
        // Release the event and the slab whatever the wait said: a failed wait
        // must not leak the pinned memory, and the caller turns the `Err` into a
        // loud boundary failure.
        self.state.event_destroy(rec.event);
        self.cross_slabs[rec.slab].in_use = false;
        r?;
        let bytes = rec.bytes;
        let data = unsafe {
            std::slice::from_raw_parts(self.cross_slabs[rec.slab].ptr as *const f32, bytes / 4)
        };
        Ok(Some(data.to_vec()))
    }

    /// F5: an idle pinned slab of at least `bytes`, growing the pool when every
    /// slab is in use. The smallest sufficient slab is chosen so a 4 KB staging
    /// copy does not occupy the 4 MB slab a prefill left behind.
    fn take_cross_slab(&mut self, bytes: usize) -> Result<usize, String> {
        let _bound = self.bind();
        let mut best: Option<usize> = None;
        for (i, s) in self.cross_slabs.iter().enumerate() {
            if s.in_use || s.bytes < bytes {
                continue;
            }
            if best.is_none_or(|b| s.bytes < self.cross_slabs[b].bytes) {
                best = Some(i);
            }
        }
        if let Some(i) = best {
            self.cross_slabs[i].in_use = true;
            self.note_cross_inflight();
            return Ok(i);
        }
        // 64 KiB floor: a decode-step boundary copies KB-scale activations, and
        // rounding up avoids a realloc per layer.
        let need = bytes.max(64 << 10);
        let ptr = self.state.host_alloc(need).ok_or_else(|| {
            format!("cuda: pinned staging allocation of {need} bytes failed (cudaHostAlloc)")
        })?;
        self.cross_slabs.push(CrossSlab {
            ptr: ptr as *mut std::ffi::c_void,
            bytes: need,
            in_use: true,
        });
        self.note_cross_inflight();
        Ok(self.cross_slabs.len() - 1)
    }

    /// #138: record how many cross slabs are in use right now (see
    /// [`Self::cross_inflight_peak`]). Called on every slab acquisition, which is
    /// the only point the count can rise.
    fn note_cross_inflight(&mut self) {
        let n = self.cross_slabs.iter().filter(|s| s.in_use).count();
        self.cross_inflight_peak = self.cross_inflight_peak.max(n);
    }

    /// Viz/trace capture: queue an async D2H of pool buffer `id` into the
    /// pinned capture staging (see `CaptureStaging`). `false` = refused
    /// (unknown/empty buffer, or it does not fit under the staging ceiling) —
    /// the caller falls back to the per-node sync `copy_to_host`.
    pub fn capture_enq(&mut self, id: usize) -> bool {
        let _bound = self.bind();
        let Some(b) = self.pool.get(id) else {
            return false;
        };
        if b.ptr.is_null() || b.bytes == 0 {
            return false;
        }
        let _sg = self.stream_guard();
        self.cap.queue(self.state.stream(), b.ptr, b.bytes)
    }

    /// One stream sync, then drain the capture staging: one `Vec<f32>` per
    /// successfully enqueued buffer, in enqueue order.
    pub fn capture_drain(&mut self) -> Vec<Vec<f32>> {
        // Issue #188: bind first — `CaptureStaging::drain` syncs a stream, and it
        // must be the stream the queued D2H copies were issued on.
        let _bound = self.bind();
        self.cap.drain(self.state)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Issue #188: bind this backend's stream so the teardown's host-side
        // transfers (and any queued work still in flight) are scoped to it.
        // `cudaFree`/`cudaFreeHost` are context-wide; with the stream
        // non-blocking they no longer implicitly sync against another engine's
        // capturing stream, and the capture mode is thread-local.
        let _bound = self.bind();
        // Pools live as long as the backend (inside GraphCache); only real
        // teardown frees device memory. free_buffer() only recycles.
        for b in &self.pool {
            Self::state_free(b.ptr);
        }
        self.pool.clear();
        self.free.clear();
        for (_, (ptr, _)) in self.pos_scratch.drain() {
            Self::state_free(ptr);
        }
        // F5: release the async staging pool (a pending copy's event is destroyed
        // first — `cudaFreeHost` syncs, so the bytes are gone either way).
        for p in self.cross_pending.drain(..) {
            self.state.event_destroy(p.event);
        }
        for s in self.cross_slabs.drain(..) {
            self.state.host_free(s.ptr as *mut u8);
        }
        for g in &self.graph_execs {
            self.state.graph_destroy(g.exec);
        }
        self.graph_execs.clear();
        self.capturing = None;
        // Last, so every operation above ran on a live stream.
        self.state.destroy_stream(self.stream);
        self.stream = std::ptr::null_mut();
    }
}

impl CudaBackend {
    /// End an open capture window WITHOUT launching it (error path, Phase 8
    /// review): the recorded launches never executed, so the split's outputs
    /// are invalid. Disables graph capture for the session.
    fn abort_capture(&mut self, cause: &str) {
        if let Some(key) = self.capturing.take() {
            let _bound = self.bind();
            let exec = self.state.graph_end_capture_to_exec();
            if !exec.is_null() {
                self.state.graph_destroy(exec);
            }
            self.graphs_mode = GraphMode::Disabled;
            eprintln!(
                "CUDA: node error inside capture window (split {key:?}); capture aborted, \
                 graphs disabled for this session: {cause}"
            );
            self.state_sync();
        }
    }

    fn execute_node_inner(
        &mut self,
        node: &CNode,
        in_bufs: &[BufRef],
        out_buf: BufRef,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        // one lock acquisition per node; None while this backend itself is
        // capturing (its own enqueues are the recorded work)
        let _sg = self.stream_guard();
        // r49: the MMQ A-quantize memoization is valid only across CONSECUTIVE
        // MatMul nodes sharing the same src A. ANY other node kind clears it
        // (conservative — no write tracking): a matmul src buffer is immutable
        // within its own run, but the allocator's buffer-id reuse could
        // otherwise alias a late matmul's A onto a cached one. Transposed-b
        // matmuls also route here (they error below) but never consult MMQ.
        // D3-5 1a: Op::FusedFFN joins the preserve set — its input is the
        // ffn_norm rms output whose pad40 plane the fused rms epilogue just
        // recorded, and the internal gu matmul is that src's first consumer
        // (same consecutive-consumer window as MatMul→MatMul; any node
        // between the producer and here would have cleared the entry).
        if !matches!(&node.op, Op::MatMul { .. } | Op::FusedFFN) {
            self.state.clear_mmq_cache();
        }
        match &node.op {
            // Inputs are host-filled by the allocator; KvcacheLoad is a view
            // of the persistent K region (out_buf IS the region — no kernel).
            Op::Input | Op::KvcacheLoad { .. } => Ok(()),
            // D1: view-like nodes own nothing — the allocator mapped the node
            // onto its parent's buffer, so there is no copy to perform (and a
            // d2d copy from a buffer to itself would be wasted traffic at best).
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                let src = *in_bufs
                    .first()
                    .ok_or_else(|| format!("cuda: {} without source buffer", node.name))?;
                if src.id != out_buf.id {
                    return Err(format!(
                        "cuda: {} is a view but its output buffer is not its source's (D1 aliasing                          missing); refusing to copy silently",
                        node.name
                    ));
                }
                Ok(())
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
                        self.ptr_of_ref(in_bufs[0])?,
                        self.ptr_of_ref(out_buf)?,
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
                        self.ptr_of_ref(in_bufs[0])?,
                        self.ptr_of_ref(in_bufs[1])?,
                        self.ptr_of_ref(out_buf)?,
                        n_embd,
                        nt,
                    );
                    Ok(())
                }
                other => Err(format!("get_rows node with unexpected meta: {other:?}")),
            },

            Op::Add => {
                let n = out_buf.len;
                if in_bufs[0].len != n || in_bufs[1].len != n {
                    return Err(format!("cuda: {}: add input size mismatch", node.name));
                }
                self.state.add_f32(
                    self.ptr_of_ref(in_bufs[0])?,
                    self.ptr_of_ref(in_bufs[1])?,
                    self.ptr_of_ref(out_buf)?,
                    n,
                );
                Ok(())
            }
            Op::Mul => {
                let n = out_buf.len;
                if in_bufs[0].len != n || in_bufs[1].len != n {
                    return Err(format!("cuda: {}: mul input size mismatch", node.name));
                }
                self.state.mul_f32(
                    self.ptr_of_ref(in_bufs[0])?,
                    self.ptr_of_ref(in_bufs[1])?,
                    self.ptr_of_ref(out_buf)?,
                    n,
                );
                Ok(())
            }
            // In-place op (alias rule, graph rules §5): stage via D2D copy when
            // the allocator did not alias the input, then run on the output.
            Op::Silu => {
                if in_bufs[0].id != out_buf.id {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                self.state.silu_f32(self.ptr_of_ref(out_buf)?, out_buf.len);
                Ok(())
            }
            Op::SwiGLU => {
                let n = out_buf.len;
                if in_bufs[0].len != n || in_bufs[1].len != n {
                    return Err(format!("cuda: {}: swiglu input size mismatch", node.name));
                }
                // r51/r52: producer-fused A-quantize (MINFER_MMQ_A_FUSE + full
                // MMQ gate set; r60: default-on, "0" opts out): in the prefill
                // graph the swiglu output is
                // EXCLUSIVELY the down projection's GEMM input, so compute
                // the pad40_t plane in the same pass and register it in the
                // MmqCache — prefill_mmq consumes it with no re-quantize.
                // Rows >= 16 keeps decode (nt==1, capture window, FusedFFN)
                // and short-prefill (nt < 16 never reaches prefill_mmq) on
                // the unfused pair; dim % 256 mirrors the transposed GEMM's
                // nchunk % 8 requirement.
                // r52 mode 2 additionally SKIPS the f32 output write: the
                // output's sole consumers are the immediately following
                // consecutive MatMul nodes reading the plane via the MmqCache
                // (window-safety proof: docs/CUDA_OPTIMIZATION.md P6 r52; the
                // dead_write cache guard turns any window violation into a
                // loud error instead of reading the unwritten buffer). Plane
                // OOM degrades to mode 1, then to the unfused pair.
                let dim = node.out_shape[0];
                let rows = if dim > 0 { n / dim } else { 0 };
                if dim > 0 && n == dim * rows && rows >= 16 && dim % 256 == 0 {
                    match self.state.mmq_a_fuse_mode() {
                        2 => {
                            if self
                                .state
                                .swiglu_quant_nw(
                                    self.ptr_of_ref(in_bufs[0])?,
                                    self.ptr_of_ref(in_bufs[1])?,
                                    self.ptr_of_ref(out_buf)?,
                                    dim,
                                    rows,
                                )
                                .is_ok()
                            {
                                return Ok(());
                            }
                            if self
                                .state
                                .swiglu_quant(
                                    self.ptr_of_ref(in_bufs[0])?,
                                    self.ptr_of_ref(in_bufs[1])?,
                                    self.ptr_of_ref(out_buf)?,
                                    dim,
                                    rows,
                                )
                                .is_ok()
                            {
                                return Ok(());
                            }
                        }
                        1 => {
                            if self
                                .state
                                .swiglu_quant(
                                    self.ptr_of_ref(in_bufs[0])?,
                                    self.ptr_of_ref(in_bufs[1])?,
                                    self.ptr_of_ref(out_buf)?,
                                    dim,
                                    rows,
                                )
                                .is_ok()
                            {
                                return Ok(());
                            }
                        }
                        _ => {}
                    }
                }
                self.state.swiglu_f32(
                    self.ptr_of_ref(in_bufs[0])?,
                    self.ptr_of_ref(in_bufs[1])?,
                    self.ptr_of_ref(out_buf)?,
                    n,
                );
                Ok(())
            }

            Op::RmsNorm { eps } => {
                let d = node.out_shape[0];
                // #169: the weight must be f32 of exactly `d` elements.
                let wptr = self.norm_weight(node, d)?;
                if d % 4 != 0 || d == 0 || out_buf.len % d != 0 {
                    return Err(format!(
                        "cuda: {}: rms_norm dim {d} must be a nonzero multiple of 4 (float4 kernel)",
                        node.name
                    ));
                }
                let n = out_buf.len / d;
                // r51/r52: producer-fused A-quantize — see the Op::SwiGLU arm.
                // Every rms_norm output in the prefill graph is exclusively
                // a GEMM input (q/k/v, gate/up; the output-norm/lm_head and
                // last-layer FFN run on the G3-tail n_out=1 rows, below the
                // n >= 16 gate). r52 mode 2 additionally skips the f32 output
                // write — same window-safety contract as the swiglu arm.
                if n >= 16 && d % 256 == 0 {
                    match self.state.mmq_a_fuse_mode() {
                        2 => {
                            if self
                                .state
                                .rms_norm_quant_nw(
                                    self.ptr_of_ref(in_bufs[0])?,
                                    wptr,
                                    self.ptr_of_ref(out_buf)?,
                                    d,
                                    n,
                                    *eps,
                                )
                                .is_ok()
                            {
                                return Ok(());
                            }
                            if self
                                .state
                                .rms_norm_quant(
                                    self.ptr_of_ref(in_bufs[0])?,
                                    wptr,
                                    self.ptr_of_ref(out_buf)?,
                                    d,
                                    n,
                                    *eps,
                                )
                                .is_ok()
                            {
                                return Ok(());
                            }
                        }
                        1 => {
                            if self
                                .state
                                .rms_norm_quant(
                                    self.ptr_of_ref(in_bufs[0])?,
                                    wptr,
                                    self.ptr_of_ref(out_buf)?,
                                    d,
                                    n,
                                    *eps,
                                )
                                .is_ok()
                            {
                                return Ok(());
                            }
                        }
                        _ => {}
                    }
                }
                // D3-5 1a: decode (n==1) producers fuse the pad40 q8
                // epilogue — bit-identical f32 y and q8 bytes, one launch
                // fewer per producer, and the following decode matmul group
                // skips its standalone quantize.
                if n == 1 && d % 32 == 0 && !crate::cuda::CudaState::no_decode_a_fuse() {
                    self.state.rms_norm_quant_on_gpu(
                        self.ptr_of_ref(in_bufs[0])?,
                        wptr,
                        self.ptr_of_ref(out_buf)?,
                        d,
                        n,
                        *eps,
                    );
                    return Ok(());
                }
                self.state.rms_norm(
                    self.ptr_of_ref(in_bufs[0])?,
                    Some(wptr),
                    self.ptr_of_ref(out_buf)?,
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
                let d = *hd;
                // #169: the per-head norm weight must be f32 of exactly `hd` elements.
                let wptr = self.norm_weight(node, d)?;
                if d % 4 != 0 || d == 0 || out_buf.len % d != 0 {
                    return Err(format!(
                        "cuda: {}: qk_norm head dim {d} must be a nonzero multiple of 4 (float4 kernel)",
                        node.name
                    ));
                }
                let n = out_buf.len / d;
                self.state.rms_norm(
                    self.ptr_of_ref(in_bufs[0])?,
                    Some(wptr),
                    self.ptr_of_ref(out_buf)?,
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
                    self.ptr_of_ref(in_bufs[0])?,
                    self.ptr_of_ref(out_buf)?,
                    od_total,
                    meta.in_dim,
                    nt,
                    self.state.is_weight_padded(&meta.gu_weight),
                )?;
                // 2) in-place swiglu: silu(rows 0..nf) × (rows nf..2*nf)
                let n = nt * meta.nf;
                let buf = self.ptr_of_ref(out_buf)?;
                // D3-5 1a: fuse the pad40 q8 epilogue for the following down
                // matmul (same fused-producer form as the rms arm).
                if n % 32 == 0 && !crate::cuda::CudaState::no_decode_a_fuse() {
                    self.state.swiglu_quant_off_on_gpu(buf, n, n);
                } else {
                    self.state.swiglu_f32_off_on_gpu(buf, n, n);
                }
                Ok(())
            }
            // D3-8: decode QKV fusion (G4 CUDA port of the Metal path): one
            // concat matmul (blk.{i}.attn_qkv = wq|wk|wv rows) + one fused
            // bias+rope+store pass, replacing the 7-launch unfused chain
            // (3 matmuls + add_bias×3 + rope×2 + store×2).
            // Bitwise argument: (1) the concat matmul — the decode MMVQ
            // kernels map one 256-thread block per row (row = blockIdx.x) and
            // the dispatch depends on (ttype, id, nt) only, so per-row results
            // cannot depend on od (probe: cuda_fused_qkv_concat_matmul_bitwise);
            // (2) the epilogue — math verbatim add_bias_f32 + rope_f32 +
            // store_kv_f32/f16 (probe: cuda_fused_qkv_epilogue_bitwise).
            // D3-8: mixed-quant QKV epilogue (class 2): q/k/v come from
            // three SEPARATE matmuls (mixed quant types — e.g. Q6_K attn_v —
            // cannot share the concat matmul); the SAME pointer-form kernel as
            // FusedQKV's epilogue applies the three biases, ropes q/k in
            // place, and stores k/v. The allocator aliases the output to q's
            // input buffer (the builder wires attention to this node, so q's
            // matmul buffer has exactly one consumer); fall back to a D2D copy
            // if it ever doesn't (RoPE arm pattern). Replaces the 7-launch
            // small-kernel tail (add_bias×3 + rope×2 + store×2) → 1 launch.
            Op::QkvBiasRopeStore { layer } => {
                let meta = match &node.meta {
                    NodeMeta::QkvBiasRopeStore(m) => m,
                    other => {
                        return Err(format!(
                            "qkv_bias_rope_store node missing QkvBiasRopeStoreMeta: {other:?}"
                        ));
                    }
                };
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "cuda: {}: QkvBiasRopeStore is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                if !matches!(meta.rope_style, RopeStyle::NonInterleaved) {
                    return Err(format!(
                        "cuda: {}: qkv epilogue rope style {:#?} not supported (rope_f32 is neox/non-interleaved only)",
                        node.name, meta.rope_style
                    ));
                }
                if meta.hd == 0 || meta.hd % 2 != 0 {
                    return Err(format!(
                        "cuda: {}: qkv epilogue head dim {} must be even",
                        node.name, meta.hd
                    ));
                }
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                // q: in-place (out aliases the q input; copy when it doesn't)
                if in_bufs[0].id != out_buf.id {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                let bias_ptr = |name: &Option<String>| -> Result<*mut std::ffi::c_void, String> {
                    match name {
                        Some(n) => self.state.get_weight_ptr(n).ok_or_else(|| {
                            format!("cuda: qkv epilogue bias '{n}' not registered on CUDA")
                        }),
                        None => Err("cuda: qkv epilogue bias missing".into()),
                    }
                };
                let bq = bias_ptr(&meta.bias_q)?;
                let bk = bias_ptr(&meta.bias_k)?;
                let bv = bias_ptr(&meta.bias_v)?;
                let pos = self.positions_i32(in_bufs[3].id)?;
                let cells = self.positions_i32(in_bufs[4].id)?;
                self.fused_qkv_epilogue(
                    self.ptr_of_ref(out_buf)?,
                    self.ptr_of_ref(in_bufs[1])?,
                    self.ptr_of_ref(in_bufs[2])?,
                    bq,
                    bk,
                    bv,
                    self.ptr_of(k_id)?,
                    self.ptr_of(v_id)?,
                    meta.nqt,
                    meta.nkt,
                    meta.hd,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                    cells,
                );
                Ok(())
            }
            Op::FusedQKV { layer } => {
                let meta = match &node.meta {
                    NodeMeta::FusedQkv(m) => m,
                    other => {
                        return Err(format!("fused_qkv node missing FusedQkvMeta: {other:?}"));
                    }
                };
                let nt = node.out_shape[1];
                if nt != 1 {
                    return Err(format!(
                        "cuda: {}: FusedQKV is decode (nt==1) only, got nt={nt}",
                        node.name
                    ));
                }
                if !matches!(meta.rope_style, RopeStyle::NonInterleaved) {
                    return Err(format!(
                        "cuda: {}: fused qkv rope style {:#?} not supported (rope_f32 is neox/non-interleaved only)",
                        node.name, meta.rope_style
                    ));
                }
                if meta.hd == 0 || meta.hd % 2 != 0 {
                    return Err(format!(
                        "cuda: {}: fused qkv head dim {} must be even",
                        node.name, meta.hd
                    ));
                }
                let wptr = self.state.get_weight_ptr(&meta.qkv_weight).ok_or_else(|| {
                    format!(
                        "cuda: qkv weight '{}' not registered on CUDA ({})",
                        meta.qkv_weight, node.name
                    )
                })?;
                let od_total = meta.nqt + 2 * meta.nkt;
                // 1) concat matmul: x × [wq|wk|wv] → q|k|v concat buffer
                self.state.matmul_f32_ptr_layout(
                    wptr,
                    meta.weight_ttype,
                    self.ptr_of_ref(in_bufs[0])?,
                    self.ptr_of_ref(out_buf)?,
                    od_total,
                    meta.in_dim,
                    nt,
                    self.state.is_weight_padded(&meta.qkv_weight),
                )?;
                // 2) fused bias + rope + KV store in one kernel pass
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                let bias_ptr = |name: &Option<String>| -> Result<*mut std::ffi::c_void, String> {
                    match name {
                        Some(n) => self.state.get_weight_ptr(n).ok_or_else(|| {
                            format!("cuda: fused qkv bias '{n}' not registered on CUDA")
                        }),
                        None => Err("cuda: fused qkv bias missing".into()),
                    }
                };
                let bq = bias_ptr(&meta.bias_q)?;
                let bk = bias_ptr(&meta.bias_k)?;
                let bv = bias_ptr(&meta.bias_v)?;
                let pos = self.positions_i32(in_bufs[1].id)?;
                let cells = self.positions_i32(in_bufs[2].id)?;
                // pointer-form section bases into the concat output
                // [q|k|v]: q at 0, k at nqt, v at nqt+nkt (in-bounds by
                // construction: out = od_total = nqt + 2*nkt f32)
                let q_ptr = self.ptr_of_ref(out_buf)? as *mut f32;
                let (k_ptr, v_ptr) = unsafe {
                    (
                        q_ptr.add(meta.nqt) as *mut std::ffi::c_void,
                        q_ptr.add(meta.nqt + meta.nkt) as *mut std::ffi::c_void,
                    )
                };
                self.fused_qkv_epilogue(
                    q_ptr as *mut std::ffi::c_void,
                    k_ptr,
                    v_ptr,
                    bq,
                    bk,
                    bv,
                    self.ptr_of(k_id)?,
                    self.ptr_of(v_id)?,
                    meta.nqt,
                    meta.nkt,
                    meta.hd,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                    cells,
                );
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
                if in_bufs[0].len < id * nt || out_buf.len < od * nt {
                    return Err(format!(
                        "cuda: {}: buffer size mismatch for [{od}x{id}] x nt={nt}",
                        node.name
                    ));
                }
                self.state.matmul_f32_ptr_layout(
                    wptr,
                    meta.weight_ttype,
                    self.ptr_of_ref(in_bufs[0])?,
                    self.ptr_of_ref(out_buf)?,
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
                    self.state
                        .add_bias_f32(self.ptr_of_ref(out_buf)?, bptr, od, nt);
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
                if in_bufs[0].id != out_buf.id {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                let nt = node.out_shape[1];
                let pos = self.positions_i32(in_bufs[1].id)?;
                self.state.rope_f32(
                    self.ptr_of_ref(out_buf)?,
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
                if out_buf.id != k_id {
                    return Err(format!(
                        "cuda: kv store output buffer {} is not the K region {k_id}",
                        out_buf.id
                    ));
                }
                let nkt = node.out_shape[0];
                if nkt == 0 || in_bufs[0].len % nkt != 0 {
                    return Err(format!(
                        "cuda: kv store k input {} elems not a multiple of nkt {nkt}",
                        in_bufs[0].len
                    ));
                }
                let nt = in_bufs[0].len / nkt;
                let pos = self.positions_i32(in_bufs[2].id)?;
                // Note: positions >= n_ctx are not validated here (device-side
                // data); the CPU backend checks them, the GPU backends trust
                // session-level clamping like Metal's store_kv dispatch.
                // 8b: f16 KV stores into the same persistent region viewed as
                // half (2 bytes/elem) — halves attention read bandwidth, same
                // trade-off as Metal's store_kv dispatch.
                let (sk, sv) = (self.ptr_of_ref(in_bufs[0])?, self.ptr_of_ref(in_bufs[1])?);
                let (dk, dv) = (self.ptr_of(k_id)?, self.ptr_of(v_id)?);
                // C4 S2b: the layout picks the store. `store_kv_q8_0` needs the
                // packed cell's byte width, which is the same number `ensure_kv`
                // sized the region with (`KvFormat::Q8_0.row_bytes(nkt)`).
                match self.kv_layout() {
                    crate::cuda::KV_LAYOUT_Q8_0 => {
                        let row_bytes = self.kv_row_bytes(nkt);
                        self.state.store_kv_q8_0(sk, dk, nkt, nt, row_bytes, pos);
                        self.state.store_kv_q8_0(sv, dv, nkt, nt, row_bytes, pos);
                    }
                    crate::cuda::KV_LAYOUT_F16 => {
                        self.state.store_kv_f16(sk, dk, nkt, nt, pos);
                        self.state.store_kv_f16(sv, dv, nkt, nt, pos);
                    }
                    _ => {
                        self.state.store_kv_f32(sk, dk, nkt, nt, pos);
                        self.state.store_kv_f32(sv, dv, nkt, nt, pos);
                    }
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
                // Issue #122: the prefill entries index `q` as `nt` token rows of
                // `n_head * hd` (`fa_prefill_f16kv`: `q[t * nh * hd + h * hd + d]`), so a
                // q input shorter than that is an out-of-bounds device read. Refuse it
                // loudly here instead of letting the kernel do it: the fixture that
                // triggered #122 read ~7 MB past its q buffer and latched
                // `cudaErrorIllegalAddress` (700), which corrupted the CUDA context and
                // surfaced only later as a misleading "0 byte budget". The length contract
                // is `BufRef::len` (rule 13).
                let q_need = nt * meta.n_head * meta.hd;
                if in_bufs[0].len < q_need {
                    return Err(format!(
                        "cuda: attention q input has {} values but a {} token prefill needs \
                         n_tokens * n_head * hd = {q_need}; refusing rather than reading past \
                         the buffer (issue #122)",
                        in_bufs[0].len, nt
                    ));
                }
                // E1b: a multi-sequence node carries an explicit `[lo, hi)` span
                // (input 3) and runs the windowed kernel instantiations; every
                // other node is the causal case and reads `positions` (input 2)
                // exactly as before — same kernels, same instructions, same
                // captures. The bound still never crosses to the host
                // (precondition for CUDA Graph replay, Phase 7d).
                let windowed = matches!(
                    &node.op,
                    Op::Attn {
                        explicit_span: true,
                        ..
                    }
                );
                // C8b S2/S4: an explicit window is either one `(lo, hi)` pair per
                // query (`attn_span`) or `KV_MAP_MAX_SPANS` `(cell, len)` runs per
                // query (`kv_map`) — the **size** says which, and each mode is its own
                // kernel instantiation. A size that matches neither is refused rather
                // than mis-strided: a `(cell, len)` pair read as `(lo, hi)` would
                // attend to the wrong rows silently.
                let mode = if !windowed {
                    crate::cuda::AttnWindow::Causal
                } else if in_bufs[3].len == 2 * nt {
                    crate::cuda::AttnWindow::Span
                } else if in_bufs[3].len == nt * super::kvcache::KV_MAP_MAX_SPANS * 2 {
                    crate::cuda::AttnWindow::Map
                } else {
                    return Err(format!(
                        "cuda: attention window input has {} values; one query needs either a \
                         single (lo, hi) pair ({}) or {} (cell, len) runs ({})",
                        in_bufs[3].len,
                        2 * nt,
                        super::kvcache::KV_MAP_MAX_SPANS,
                        nt * super::kvcache::KV_MAP_MAX_SPANS * 2
                    ));
                };
                let bound_buf = if windowed { in_bufs[3] } else { in_bufs[2] };
                let pos = self.positions_i32(bound_buf.id)?;
                // 8d: decode (nt == 1) uses split-K flash-decoding — the
                // single-warp kernel leaves the GPU idle at nt == 1 (nsys:
                // 48% of the 7B decode step at 2K ctx). Fixed grid +
                // device-side range split keeps CUDA Graph capture valid;
                // the partials scratch is size-stable (nh/hd constants).
                if nt == 1 {
                    self.state.gqa_attn_split(
                        self.ptr_of_ref(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of_ref(out_buf)?,
                        pos,
                        mode.code(),
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        self.kv_layout(),
                        self.kv_row_bytes(meta.nkt),
                    );
                    return Ok(());
                }
                // doc 94: verify shapes (1 < nt <= 16) route through the
                // batched split path — bitwise-equal per position to the
                // nt=1 decode path (the greedy identity). The incumbent
                // single-warp-per-(token, head) kernel walks the keys with
                // a different reduction schedule, which flipped argmax on
                // near-ties and made spec output diverge from sequential
                // greedy decode. Prefill (nt > 16) keeps the incumbent.
                //
                // C4 S2b: a **packed** cache does not take this path. The batched
                // split kernel's whole purpose is that unmeasured-here bitwise
                // identity, so instead of claiming it for Q8_0, the verify band
                // falls through to `gqa_attn_f32` — and a speculative session
                // refuses a packed cache outright (`spec::draft` load gate), because
                // its greedy identity would no longer be the thing this kernel
                // guarantees.
                if nt <= 16 && self.kv_layout() != crate::cuda::KV_LAYOUT_Q8_0 {
                    self.state.gqa_attn_split_batched(
                        self.ptr_of_ref(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of_ref(out_buf)?,
                        pos,
                        mode.code(),
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        self.kv_layout() == crate::cuda::KV_LAYOUT_F16,
                        nt,
                    );
                    return Ok(());
                }
                // nt > 16 (prefill): FA-style tiled attention (f16 tile staging,
                // tensor-core QK^T/P·V) for the f16 and packed layouts alike —
                // #144 item 3 stages a packed cell by dequantizing it into the
                // same f16 tile, with the general layout-tagged kernel as the
                // documented fallback. f32 stays on the general kernel (its
                // staging would round the cache to f16 for no reason).
                if self.kv_layout() != crate::cuda::KV_LAYOUT_F32 {
                    self.state.gqa_attn_kv_prefill(
                        self.ptr_of_ref(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of_ref(out_buf)?,
                        pos,
                        mode.code(),
                        self.kv_layout(),
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        self.kv_row_bytes(meta.nkt),
                        nt,
                    );
                } else {
                    self.state.gqa_attn_f32(
                        self.ptr_of_ref(in_bufs[0])?,
                        self.ptr_of(k_id)?,
                        self.ptr_of(v_id)?,
                        self.ptr_of_ref(out_buf)?,
                        pos,
                        mode.code(),
                        self.kv_layout(),
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        self.kv_row_bytes(meta.nkt),
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

    fn copy_d2d(&self, src: BufRef, dst: BufRef) -> Result<(), String> {
        let _bound = self.bind();
        let (s, d) = (self.ptr_of_ref(src)?, self.ptr_of_ref(dst)?);
        let (sb, db) = (src.len * 4, dst.len * 4);
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
        // One staging buffer **per source input**, converted on every request: a
        // graph converts `positions` for the store and `attn_span` for attention
        // inside the same window, and sharing one buffer made the second
        // conversion clobber the first before its launch ran (see the field
        // comment). Buffers are reused across windows, so this is a kernel launch
        // plus a small memcpy per i32 input per step — the same cost the memo hit
        // saved only for the *repeated* consumer of one input.
        let src = self.ptr_of(id)?;
        let bytes = self.pool[id].bytes;
        let slot = match self.pos_scratch.get(&id) {
            Some(&(ptr, have)) if have >= bytes && !ptr.is_null() => (ptr, have),
            _ => {
                if let Some((old, _)) = self.pos_scratch.remove(&id) {
                    if !old.is_null() {
                        Self::state_free(old);
                    }
                }
                let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
                if ptr.is_null() {
                    return Err("cuda: positions scratch allocation failed".to_string());
                }
                // the freed scratch pointer may be embedded in captured graph
                // execs — invalidate them so they re-capture against the new
                // address (Phase 8 review)
                self.pool_gen += 1;
                self.pos_scratch.insert(id, (ptr, bytes));
                (ptr, bytes)
            }
        };
        self.state.bits_to_i32(src, slot.0, bytes / 4);
        Ok(slot.0)
    }

    /// Resolve a NormMeta weight by name on the CUDA registry. Unlike Metal
    /// (which silently degrades to a weightless norm when the weight is not on
    /// the backend), a declared-but-missing weight is an invariant violation
    /// here and returns Err (docs/GPU_SAFETY.md).
    ///
    /// `elems` is the number of f32 elements the kernel will read (the norm
    /// dim). The type is part of the invariant: the rms_norm kernel indexes the
    /// weight as `d * 4` bytes, so a weight registered with a different length
    /// — an f16 norm (2 B/element) from a hand-made GGUF, which the f16 weight
    /// path has no kernel for (#169) — would read past its end silently. This
    /// returns Err instead, naming both lengths (kernel-invariant violations
    /// are refusals, never a silent wrong path).
    fn norm_weight(&self, node: &CNode, elems: usize) -> Result<*mut std::ffi::c_void, String> {
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
        let ptr = self.state.get_weight_ptr(name).ok_or_else(|| {
            format!(
                "cuda: weight '{name}' not registered on CUDA ({})",
                node.name
            )
        })?;
        let want = elems * 4;
        let got = self.state.weight_size(name);
        if got != Some(want) {
            return Err(format!(
                "cuda: norm weight '{name}' is {} B but the rms_norm kernel reads {elems} f32 \
                 elements ({want} B) for '{}'; the f16 weight path has no f16-norm kernel (#169)",
                got.map(|n| n.to_string())
                    .unwrap_or_else(|| "unregistered".to_string()),
                node.name
            ));
        }
        Ok(ptr)
    }
}

/// F4: the CUDA capability matrix, as a free function.
///
/// This is the **authority** for the answer: the `Backend` trait method below
/// forwards to it (the function cannot take a `&self`), and assignment reads the
/// trait method. The registry does not carry a copy.
///
/// Capability matrix (docs/CUDA-BACKEND-DESIGN.md §4.3): the full per-layer
/// chain runs on CUDA, including the embedding/tail gather (7e③) and the
/// decode fusions FusedQKV/QkvBiasRopeStore/FusedFFN. Scale, Softmax,
/// BatchMatMul and the Qwen3-only FusedQkvNorm have no kernels and stay on
/// the CPU backend; weight-quant eligibility is the model-level
/// all-weights-registered gate. RoPE is gated to the neox
/// (non-interleaved) layout — the only style the supported architectures
/// emit.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
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
        // D3-8: decode QKV fusion (G4 CUDA port) — concat matmul + fused
        // bias/rope/store epilogue (nt==1; the builder emits the node only
        // when the loader registered blk.{i}.attn_qkv, the same gate as
        // Metal — see qkv_concat_available / CParams.fuse_qkv).
        | Op::FusedQKV { .. }
        // D3-8: mixed-quant QKV epilogue (class 2) — same pointer-form
        // kernel as FusedQKV's epilogue on three separate matmul outputs
        | Op::QkvBiasRopeStore { .. }
        | Op::FusedFFN => true,
        // MatMul ttype gating happens at the model level (weights must all
        // be registered on CUDA — same all-or-nothing rule as Metal).
        Op::RoPE { style } => matches!(style, RopeStyle::NonInterleaved),
        _ => false,
    }
}

/// F4: the fusion-pass capability, as a free function (see [`supports_op`]).
pub fn supports_fused(fused: &FusedOp) -> bool {
    matches!(fused, FusedOp::SwiGLU)
}

/// E1b: the attention kernels gained a windowed instantiation that reads the
/// `[lo, hi)` span, so CUDA can take a multi-sequence attention node.
pub const SUPPORTS_ATTN_SPAN: bool = true;

/// C4 S2b: CUDA reads a packed `q8_0` KV region. The kernels gained a layout tag
/// (`KV_LAYOUT_F32/F16/Q8_0`) and a byte-addressed row accessor (`kv_row` +
/// `kv4<LAYOUT>`), a `store_kv_q8_0` that uses the CPU's own quantizer, and a
/// `copy_cells` stride that leaves a word-padded packed cell alone. The dispatch
/// cuts (decode through the converted 1-warp split body with `rpw_gate = 0`; the
/// verify band and prefill through the layout-tagged general kernel; no fused QKV
/// epilogue) are stated in `docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 C4 S2b.
pub const READS_PACKED_KV: bool = true;

/// F5 ([#58]): registry hook **phase A** — enqueue one cross-backend staging
/// copy out of CUDA.
///
/// The device→host direction is the one with a device leg, and it is the whole
/// point of the ticket: before F5 the generic path read the source with
/// `copy_to_host`, which **synchronized the whole stream** and then issued a
/// blocking `cudaMemcpy` — two host stalls per staged input, per boundary. Here
/// the transfer is an `cudaMemcpyAsync` into a pinned slab plus a recorded event,
/// so the host is not blocked at all; `await_cross` below is where it waits, once,
/// at the documented point.
///
/// Every other destination **declines** (`Ok(false)`) and the allocator's
/// synchronous host round trip handles it exactly as before F5: a device→device
/// staging copy (unreachable today — `copy_across` early-returns when source and
/// destination backends match), CUDA→Metal on a macOS+CUDA build, and anything
/// else a future device adds. Declining is deliberate, not a silent fallback: the
/// allocator *does* perform that pair, just synchronously, and the boundary
/// counters record it as a blocking copy.
pub(crate) fn copy_cross(
    alloc: &mut GraphAllocator,
    uid: u64,
    node_id: NodeId,
    dst_backend: BackendTag,
) -> Result<bool, String> {
    if dst_backend != BackendTag::CPU {
        return Ok(false);
    }
    let src = alloc
        .node_buffer(node_id)
        .ok_or_else(|| format!("node {node_id} has no buffer"))?;
    let dst = alloc
        .cross_buffer(uid, node_id, dst_backend)
        .ok_or_else(|| format!("node {node_id} has no staging buffer on {dst_backend:?}"))?;
    // Resolve the source device pointer before borrowing the pool mutably (the
    // two borrows cannot be live at once).
    let src_ptr = alloc
        .cuda()
        .ok_or("CUDA backend not enabled")?
        .ptr_of_ref(src)? as *const std::ffi::c_void;
    let bytes = src.len * 4;
    alloc
        .cuda_mut()
        .ok_or("CUDA backend not enabled")?
        .enqueue_cross_host(uid, node_id, dst, src_ptr, bytes)?;
    alloc.cross_stats_mut().async_host_copies += 1;
    Ok(true)
}

/// F5 ([#58]): registry hook **phase B** — wait on the event phase A recorded.
///
/// This is the synchronization point the ticket's second acceptance line is
/// about: the D2H copy is in flight, and reading its destination before this wait
/// is a read of undefined data. `take_cross` waits on the recorded event (the
/// **only** host block in the async device→host path) and releases the slab, after
/// which the allocator publishes the bytes into the staging buffer the consumer
/// reads.
///
/// A no-op when this backend has no pending copy for `(uid, node, dst)` — the pair
/// declined phase A and the allocator's synchronous path already produced the
/// bytes.
pub(crate) fn await_cross(
    alloc: &mut GraphAllocator,
    uid: u64,
    node_id: NodeId,
    dst_backend: BackendTag,
) -> Result<(), String> {
    let Some(data) = alloc
        .cuda_mut()
        .ok_or("CUDA backend not enabled")?
        .take_cross(uid, node_id, dst_backend)?
    else {
        return Ok(());
    };
    // Copy out of the released slab before the mutable borrow below: the bytes are
    // already in host memory, so this is a plain memcpy, not a device transfer.
    let dst = alloc
        .cross_buffer(uid, node_id, dst_backend)
        .ok_or_else(|| format!("node {node_id} has no staging buffer on {dst_backend:?}"))?;
    alloc.cross_stats_mut().event_syncs += 1;
    alloc.write_cross_staging(dst, &data)
}

/// F4: this backend's registry entry (see `cpu_backend::entry`).
pub fn entry() -> super::registry::BackendEntry {
    use super::registry::{Backend as Handle, BackendCaps, BackendEntry, PRIORITY_CUDA};
    BackendEntry {
        handle: Handle::CUDA,
        name: "cuda",
        priority: PRIORITY_CUDA,
        caps: BackendCaps {
            reads_packed_kv: READS_PACKED_KV,
        },
        pool: |a| a.cuda().map(|c| c as &dyn Backend),
        pool_mut: |a| a.cuda_mut().map(|c| c as &mut dyn Backend),
        // A borrowed `&[f32]` into device memory is not expressible, so the
        // trait's `read_host` is `None` and the pool's own stream-ordered
        // `copy_to_host` is the host-read path (F4: it is a registry hook for
        // exactly this reason).
        host_read: |a, id| a.cuda().and_then(|c| c.copy_to_host(id)),
        // F5: the split boundary's async staging copy. CUDA is the backend the
        // device leg of the ticket is asserted on: an `cudaMemcpyAsync` D2H into a
        // pinned slab plus a recorded event, waited on once at the boundary's
        // documented synchronization point. See `copy_cross` / `await_cross` above
        // for what each destination gets.
        copy_cross,
        await_cross,
        // #153: the answer is the **engine's** stamped format, not a process global
        // (C5 records it as the session's KV element type, so a packed session is
        // saved and resumed as Q8_0 — the container's FLAG_PACKED bit — and an f16
        // one is the separate gap [#130] tracks). Reading the allocator's stamp
        // rather than `cuda().kv_format()` matters because `kv_load` / `load_slots`
        // run **before** the first forward, when this backend has not been enabled:
        // `enable_cuda` builds its tag from this same stamp (`cuda::layout_of`), so
        // the two cannot disagree, and the pre-forward answer is the engine's.
        kv_format: |a| a.kv_format(),
        enable: |a| a.enable_cuda(),
        unavailable: || {
            if crate::cuda::CudaState::get().is_some() {
                None
            } else {
                Some("no CUDA device is available, or CUDA is disabled (MINFER_DISABLE_CUDA)")
            }
        },
    }
}

/// F4: register the CUDA backend (`--features cuda`).
pub fn register(registry: &mut super::registry::Registry) {
    registry.register_entry(entry());
}

impl Backend for CudaBackend {
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        supports_op(op, dtype)
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        supports_fused(fused)
    }

    fn supports_attn_span(&self) -> bool {
        SUPPORTS_ATTN_SPAN
    }

    /// Device bytes the CUDA weight registry holds (E4).
    fn weights_bytes(&self) -> usize {
        crate::cuda::CudaState::get().map_or(0, |c| c.weights_bytes())
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        let _bound = self.bind();
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

    fn pool_len(&self) -> usize {
        self.pool.len()
    }

    fn free_buffer(&mut self, id: usize) {
        let _bound = self.bind();
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
        in_bufs: &[BufRef],
        out_buf: BufRef,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        let _bound = self.bind();
        let result = self.execute_node_inner(node, in_bufs, out_buf, kv_pair);
        // #162: drain the sticky required-launch record in BOTH arms. A launch
        // that failed for real set it (the C site has already printed the site,
        // the instantiation and `cudaGetErrorName`, and cleared the CUDA latch);
        // a required kernel must not let the op proceed with a stale output, so
        // the node fails here with the site's own message. Draining unconditionally
        // keeps a stale record from being blamed on the *next* node, and on the
        // `Err` arm the node's own error is the real one.
        let launch = self.state.take_launch_failure();
        match result {
            Ok(()) => match launch {
                Some(msg) => {
                    let e = format!(
                        "cuda: a required kernel launch failed in node {:?}: {msg} — the node \
                         produced no valid output (issue #162)",
                        node.op
                    );
                    if self.capturing.is_some() {
                        self.abort_capture(&e);
                    }
                    Err(e)
                }
                None => Ok(()),
            },
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

    /// C3: move KV rows inside one arena on the device (the compaction copy).
    ///
    /// Issued on the backend's own stream, so it is ordered after every kernel
    /// the previous forward launched — the compacted bytes are the ones the
    /// forwards wrote. Overlap is safe by the kernel's construction (ascending
    /// rows with a barrier), which is why this does not stage through a temp.
    fn copy_cells(
        &mut self,
        dst: BufRef,
        src: BufRef,
        dst_row: usize,
        src_row: usize,
        rows: usize,
        elems_per_cell: usize,
    ) -> Result<(), String> {
        let _bound = self.bind();
        if dst.id != src.id {
            return Err(format!(
                "cuda: copy_cells moves cells within one arena ({} -> {})",
                src.id, dst.id
            ));
        }
        let dst_ptr = self.ptr_of_ref(dst)?;
        let src_ptr = self.ptr_of_ref(src)? as *const std::ffi::c_void;
        // With an f16 KV cache the rows are stored as halves (`store_kv_f16` indexes
        // a row by `nkt` half slots), while the move kernel strides in **f32**
        // elements — the unit `elems_per_cell` is expressed in. A row is `nkt / 2`
        // f32 then, so passing `nkt` walked twice as far per row and every moved row
        // landed in the wrong cell: a copy-on-write or a compaction on an f16 device
        // silently corrupted the arena. (Found by the C8b S4 real-model gate on a
        // model whose KV is f16 — the copy gates all ran on an f32-KV model.)
        //
        // C4 S2b: a **packed** cell needs no conversion. The caller passes the
        // region's own `elems / n_ctx`, which `KvCache` sets to
        // `KvFormat::Q8_0.row_elems(nkt)` for a packed region — already a whole
        // number of f32 words, which is exactly the unit `kv_move_rows` strides in.
        // The move is therefore a plain word copy of the padded cell (the 2 bytes of
        // tail padding move with it, harmlessly). `cuda_f16_kv_cell_move_strides_by_row_bytes`
        // and its Q8_0 twin are the gates that pin both statements.
        let elems_per_cell = if self.kv_layout() == crate::cuda::KV_LAYOUT_F16 {
            (elems_per_cell / 2).max(1)
        } else {
            elems_per_cell
        };
        self.state
            .kv_move_rows(dst_ptr, src_ptr, dst_row, src_row, rows, elems_per_cell)
    }

    fn read_host(&self, _id: usize) -> Option<&[f32]> {
        // A staged D2H transfer cannot return a borrowed slice (this method
        // takes &self; the host staging buffer would escape its guard). Use
        // `copy_to_host` via alloc.rs's copy_to_cpu CUDA arm instead.
        None
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let _bound = self.bind();
        // CUDA's fill has always allowed a prefix (the pool buffer may be longer
        // than the data), which is exactly the E4 S2 window contract at offset 0.
        self.write_host_window(id, 0, data)
    }

    /// E4 S2: pooled activation buffers are rounded to their size class, so a
    /// node writes its logical window into a buffer that may be longer; `offset`
    /// is that window's element offset (0 for an owning node, non-zero for a D1
    /// view).
    fn write_host_window(&mut self, id: usize, offset: usize, data: &[f32]) -> Result<(), String> {
        let _bound = self.bind();
        let _sg = self.stream_guard();
        let bytes = data.len() * 4;
        let base = offset * 4;
        let dst = self.ptr_of(id)?;
        let have = self.pool[id].bytes;
        if base.saturating_add(bytes) > have {
            return Err(format!(
                "cuda: buffer {id} too small: writing {bytes} bytes at offset {base} runs past {have} bytes"
            ));
        }
        // 7e⑥: pinned-staged async fill (same-stream ordering makes this
        // race-free with the kernels that read the input; the ring syncs
        // only if more than STAGING_SLOTS fills queue up without a sync).
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
        let dst = unsafe { (dst as *mut u8).add(base) as *mut std::ffi::c_void };
        self.state.write_input_async(src, dst);
        Ok(())
    }

    fn synchronize(&mut self) {
        let _bound = self.bind();
        if self.capturing.is_none() {
            let _sg = self.stream_guard();
        }
        // r49: the MMQ A-quantize memoization is bounded to ONE graph execution.
        // A split boundary / next execution reuses the same pool buffer ids for
        // different data, so the cached (src,nt,id) must not leak across it.
        self.state.clear_mmq_cache();
        // No i32-memo reset is needed any more: `positions_i32` re-converts on
        // every request into a stable per-input buffer, so a stale value cannot
        // survive a boundary (the buffers themselves are reused).
        self.close_capture_or_sync(true);
    }

    /// #138 ([#138]): the boundary close **without the host block**.
    ///
    /// Everything `synchronize` does that orders the following staging copies is
    /// still done — an open capture window is instantiated and launched (its
    /// launches are stream-ordered before the copies, which run on this same
    /// backend stream), and the MMQ memoization is dropped at the boundary exactly
    /// as before. What is *not* done is the `cudaStreamSynchronize`: the copies
    /// are already ordered behind this split's work by stream order, so blocking
    /// the host there buys nothing, and the one wait that is required (the
    /// device→host event) is deferred to the consumer's first read. Measured on
    /// the 0.5B mixed-offload gate: one full stream sync per boundary removed.
    ///
    /// [#138]: https://github.com/yusiwen/minfer/issues/138
    fn retire(&mut self) {
        let _bound = self.bind();
        if self.capturing.is_none() {
            let _sg = self.stream_guard();
        }
        self.state.clear_mmq_cache();
        self.close_capture_or_sync(false);
    }

    fn graph_replay(&mut self, uid: u64, range: (usize, usize), nt_hint: Option<usize>) -> bool {
        let _bound = self.bind();
        self.graph_replay_step(uid, range, nt_hint)
    }
}

#[cfg(test)]
mod tests;
