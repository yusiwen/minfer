// CUDA (NVIDIA GPU) backend for x86-64 Linux/Windows.
//
// The graph backend lives in graph/cuda_backend.rs (Phase 7a-7d); this module
// hosts the CudaState singleton it wraps: device probes, the weight registry,
// streams, per-op kernel entry points and the CUDA Graph capture API. The
// legacy layer_gpu/`init_kv_cache` pre-alloc path is no longer driven by
// main() (the graph allocator owns KV regions since Phase 7c); the legacy
// surface carries targeted #![allow]s at its use sites instead of a
// module-wide opt-out (7e⑦).

use crate::block::Q8B;
use crate::device_tier;
use crate::q4k_dsc::q4k_dsc_payload_ok;
use crate::tensor::{Tensor, TensorType};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// How an attention node's KV window is expressed (mirrors `ATTN_WIN_*` in
/// `cuda_kernels.cu`). The **size** of the node's window input is what selects it
/// (C8b S2's departure 2: the layout is topology, so it is fixed at build time),
/// and each mode is a separate template instantiation — the causal one keeps the
/// pre-E1 instruction stream (E1b).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttnWindow {
    /// `positions`: rows `[0, positions[t] + 1)` of the contiguous arena.
    Causal,
    /// `attn_span`: one `[lo, hi)` pair per query (E1).
    Span,
    /// `kv_map`: a zero-padded list of `(cell, len)` runs per query (C8b S4).
    Map,
}

impl AttnWindow {
    /// The C-side mode constant.
    pub fn code(self) -> i32 {
        match self {
            AttnWindow::Causal => 0,
            AttnWindow::Span => 1,
            AttnWindow::Map => 2,
        }
    }
}

/// Issue #189's mutation seam for the S4 map-window A/B: how many times a
/// **map-mode** attention launch is issued.
///
/// `MINFER_S4_AB_MAP_REPS=2` makes the gate's timed map arm pay twice the work,
/// which is gate contract rule 3's "break the implementation, watch the gate go
/// red" made reproducible (the one-line form of #123's map-work doubling).
/// Every other mode, and every run with the variable unset, returns `1`. Read
/// once per process — a mutation run exports the variable before the process
/// starts, so the cached value can never race a timing round.
fn s4_ab_map_reps(mode: i32) -> usize {
    if mode != AttnWindow::Map.code() {
        return 1;
    }
    static REPS: OnceLock<usize> = OnceLock::new();
    *REPS.get_or_init(|| {
        std::env::var("MINFER_S4_AB_MAP_REPS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n >= 1)
            .unwrap_or(1)
    })
}

/// Wrapper to make `*mut c_void` Send+Sync for use in Mutex.
#[derive(Clone, Copy)]
struct CudaPtr(*mut std::ffi::c_void);
unsafe impl Send for CudaPtr {}
unsafe impl Sync for CudaPtr {}

// Sized buffer for cudaGetDeviceProperties (avoids fragile field-by-field layout).
// Only the first 256 bytes (device name) are read; extra padding handles any CUDA version.
#[repr(C)]
struct CudaDevicePropBuf([u8; 4096]);

// The CUDA runtime/driver FFI declarations live in `cuda/ffi_runtime.rs`
// and the family `impl CudaState` blocks in `cuda/methods/`; `cuda.rs`
// stays the parent of both, so every `crate::cuda::*` path is unchanged.

mod ffi_runtime;
mod methods;

use self::ffi_runtime::*;
#[cfg(test)]
pub(crate) use self::ffi_runtime::{gemm_smem_opted_in, gemm_smem_optin_in_capture_count};
#[cfg(test)]
pub(crate) use self::methods::*;

// cudaMemcpyKind values (https://docs.nvidia.com/cuda/runtime-api/group__CUDART__TYPES.html)
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

const CUDA_DEV_ATTR_COMPUTE_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_COMPUTE_MINOR: i32 = 76;
const CUDA_DEV_ATTR_MULTIPROC_COUNT: i32 = 16;

/// The symbolic name of a CUDA error code (`cudaGetErrorName`), e.g.
/// `cudaErrorIllegalAddress` for 700. Falls back to `cudaError<code>` if the runtime
/// returns null, so a diagnostic can always name *something* specific.
pub(crate) fn cuda_error_name(code: i32) -> &'static str {
    // The runtime's strings are static and owned by it, so leaking the formatted
    // fallback once is bounded by the number of distinct error codes ever seen.
    let p = unsafe { cudaGetErrorName(code) };
    if p.is_null() {
        return Box::leak(format!("cudaError{code}").into_boxed_str());
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
    match s {
        std::borrow::Cow::Borrowed(b) => b,
        std::borrow::Cow::Owned(o) => Box::leak(o.into_boxed_str()),
    }
}

/// Copy a NUL-terminated C string into an owned `String` (empty for null).
/// Used by the #162 launch-failure records, whose C side owns the storage.
fn cstr_owned(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

// ─── CudaState singleton ───────────────────────────────────────

static CUDA: OnceLock<Option<CudaState>> = OnceLock::new();

// ─── Issue #188: the per-instance device stream ──────────────────
//
// `CudaState` is the process-wide **context**: the device, the name-keyed
// weight registry and the derived weight planes are genuinely context-scoped
// and stay shared (a second copy of a 7B model's weights is not an option).
// The **stream**, the capture window and the activation scratches are not
// context-scoped, and making them process-wide is what made the parallel
// device suite fault: `cudaStreamCaptureModeGlobal` says another thread's
// driver call belongs to the capture, so a weight registration's blocking
// `cudaMemcpy` on a second thread either invalidated the window (901) or
// faulted inside the driver (`cuMemcpyHtoD_v2`, issue #188's backtrace).
//
// The stream therefore travels with the **backend instance**. Every
// `CudaBackend` owns its own non-blocking stream, and binds it for the
// duration of each of its device operations; `CudaState::stream()` then
// answers with the bound stream. That keeps the ~60 launch/copy/event helpers
// in this module unchanged in signature while giving every consumer the
// backend's stream, which is what makes two engines in one process able to
// run (and capture) at the same time.

thread_local! {
    /// The device stream bound on this thread by the innermost
    /// [`bind_stream`] guard; null means "use `CudaState`'s own stream".
    static BOUND_STREAM: Cell<*mut std::ffi::c_void> = const { Cell::new(std::ptr::null_mut()) };
}

/// The default (non-backend) stream pointer, published by `CudaState::try_new`.
/// The per-stream scratch maps key on this when no backend has bound a stream
/// (legacy layer path, direct `CudaState` tests), so an unbound caller still
/// gets a stable, private scratch set.
static DEFAULT_STREAM: AtomicUsize = AtomicUsize::new(0);

/// RAII binding of a device stream to the current thread. Restores the previous
/// binding on drop, so a backend method may freely nest.
///
/// `!Send` on purpose: a binding describes the thread that made it, and moving
/// it to another thread would restore the wrong stream there.
pub struct StreamBinding {
    prev: *mut std::ffi::c_void,
    _not_send: std::marker::PhantomData<*const ()>,
}

impl Drop for StreamBinding {
    fn drop(&mut self) {
        BOUND_STREAM.with(|c| c.set(self.prev));
    }
}

/// Bind `stream` as this thread's device stream until the guard drops.
pub fn bind_stream(stream: *mut std::ffi::c_void) -> StreamBinding {
    let prev = BOUND_STREAM.with(|c| c.replace(stream));
    StreamBinding {
        prev,
        _not_send: std::marker::PhantomData,
    }
}

/// The stream bound on this thread, or null when none is bound.
pub fn bound_stream() -> *mut std::ffi::c_void {
    BOUND_STREAM.with(|c| c.get())
}

/// Key for the per-stream scratch maps: the bound stream, or the context's own
/// stream for an unbound caller.
fn current_stream_key() -> usize {
    let bound = bound_stream() as usize;
    if bound != 0 {
        bound
    } else {
        DEFAULT_STREAM.load(Ordering::Relaxed)
    }
}

/// Grow-on-demand activation scratch that is **private to a device stream**
/// (issue #188). Every `(ptr, size)` pair used to be a process-wide slot, so
/// two engines running concurrently on two streams overwrote each other's
/// quantized activations before the consuming launch read them. The public
/// shape (`get_or_grow(&slot, need)`) is unchanged; only the key moved.
struct StreamScratch {
    map: Mutex<HashMap<usize, (CudaPtr, usize)>>,
}

impl StreamScratch {
    fn new() -> Self {
        Self {
            map: Mutex::new(HashMap::new()),
        }
    }
}

/// Issue #188: the CUDA stream capture mode the device layer opens windows
/// with. `cudaStreamCaptureModeGlobal` (1, the pre-#188 value) makes *another*
/// thread's non-capture-safe driver call invalidate the window; both
/// alternatives scope invalidation to the capturing thread —
/// `cudaStreamCaptureModeThreadLocal` (2) keeps the capturing thread's own
/// mistakes fatal, `cudaStreamCaptureModeRelaxed` (0) does not.
pub const CAPTURE_MODE_RELAXED: i32 = 0;
pub const CAPTURE_MODE_GLOBAL: i32 = 1;
pub const CAPTURE_MODE_THREAD_LOCAL: i32 = 2;

/// The capture mode a window is opened with, read once per process.
/// `MINFER_CUDA_CAPTURE_MODE=0|1|2` overrides it (the #188 probe's per-mode
/// measurement); the default is `ThreadLocal` — see the record in
/// `docs/CUDA-BACKEND-DESIGN.md` §"Per-instance streams and capture".
fn capture_mode() -> i32 {
    static MODE: OnceLock<i32> = OnceLock::new();
    *MODE.get_or_init(|| {
        match std::env::var("MINFER_CUDA_CAPTURE_MODE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            Some(m @ (CAPTURE_MODE_RELAXED | CAPTURE_MODE_GLOBAL | CAPTURE_MODE_THREAD_LOCAL)) => m,
            Some(other) => {
                eprintln!(
                    "CUDA: ignoring MINFER_CUDA_CAPTURE_MODE={other} (want 0 relaxed, 1 global, \
                     2 thread-local); using thread-local"
                );
                CAPTURE_MODE_THREAD_LOCAL
            }
            None => CAPTURE_MODE_THREAD_LOCAL,
        }
    })
}

/// Issue #188: the raw return code of the last `cudaStreamEndCapture` the graph
/// backend performed. `0` = the window closed cleanly; `901`
/// (`cudaErrorStreamCaptureInvalidated`) = another thread's driver call was not
/// capture-safe under the mode that was open. The acceptance probe reads it; the
/// production paths ignore it (a null exec is the failure signal there).
static LAST_CAPTURE_END_CODE: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);

/// Issue #188: see [`LAST_CAPTURE_END_CODE`].
/// Test-only (#238): driven by `graph::cuda_backend::tests::capture::capture_window_on_one_thread_survives_a_weight_registration_on_another`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn last_capture_end_code() -> i32 {
    LAST_CAPTURE_END_CODE.load(Ordering::Relaxed)
}

/// Issue #145: how many times `CudaState::sync` found an error **already latched**
/// by `cudaGetLastError` (i.e. not caused by the kernel that just ran). The
/// message names the observer, never a launch; this counter is how the gate
/// proves the error was surfaced rather than dropped.
static LATCHED_API_ERRORS: AtomicU64 = AtomicU64::new(0);

/// The honest label for an error that `cudaGetLastError` found **already
/// latched** at a sync point.
///
/// `sync` cannot know which call set it — it may be a launch, an attribute
/// request, a graph destroy, or anything else several operations ago — so the
/// message names the observer (`cudaGetLastError`), the `cudaGetErrorName`
/// symbolic name and the code, and never claims "kernel launch". Pure, so the
/// wording is pinned by a unit test with no device.
pub fn latched_api_error_message(err: i32) -> String {
    format!(
        "CUDA: latched API error observed by cudaGetLastError: {} ({err}); NOT attributed to a \
         kernel — an earlier CUDA call on this thread did not check its return value (fix that \
         call site rather than the kernel)",
        cuda_error_name(err)
    )
}

/// Issue #147: the message for a `cudaGraphDestroy` that returned an error.
///
/// The handle destroyed here is the `cudaGraph_t` from `cudaStreamEndCapture`;
/// a `cudaGraphExec_t` (from `cudaGraphInstantiate`) goes to
/// `cudaGraphExecDestroy` instead — mixing them returns `cudaErrorInvalidValue`
/// and leaks the handle (issue #145). A failed destroy leaks the graph but
/// leaves the instantiated exec valid, so it is named and cleared here rather
/// than turned into a refusal of the (usable) exec.
///
/// Pure, so the wording is pinned by a unit test with no device — and so a gate
/// that asserts the message can be mutation-checked against a version that
/// names the wrong call.
pub fn graph_destroy_failure_message(err: i32) -> String {
    format!(
        "CUDA: cudaGraphDestroy(cudaGraph_t from cudaStreamEndCapture) failed: {} ({err}); the \
         graph handle leaks, the instantiated exec stays valid, and the error was named and \
         cleared here so it cannot resurface as a latched API error (issue #147)",
        cuda_error_name(err)
    )
}

/// Issue #147 test injection: the `MINFER_TEST_CALL_FAIL` query for `site`
/// (test-only; unset in every default run, including the `compute-sanitizer`
/// one).
///
/// #171 absorbed the matcher into `crate::testfail::injection_names_site` — one
/// matcher for the Rust chokepoints and the device-side `launch:*`/`attr:*`
/// sites, with its exact-token tests in `src/testfail.rs` — so this is a thin
/// alias. Its direct reader is production `graph_end_capture_to_exec` (the
/// `destroy:graph_destroy` injection); the #147 device gates are other call
/// sites, not the only ones.
fn test_call_failure_requested(site: &str) -> bool {
    crate::testfail::requested(site)
}

/// A small per-thread reentrant lock guarding model weight registration.
///
/// std's `Mutex` is not reentrant, but the natural usage nests: a test holds
/// the lock across several forwards while `load_model` (called inside) takes
/// it again to register weights. This guard tracks the owning thread plus a
/// depth counter — recursive acquisition on the same thread is free; other
/// threads block until the outermost guard drops.
pub struct ModelLoadGuard {
    _inner: Option<MutexGuard<'static, ()>>,
}

static MODEL_LOAD_MUTEX: Mutex<()> = Mutex::new(());
static MODEL_LOAD_OWNER: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static MODEL_LOAD_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl ModelLoadGuard {
    fn acquire() -> Self {
        let tid = std::thread::current().id();
        // ThreadId is opaque; use its Debug value as a stable discriminator
        // for the owner slot (collision-free within a process).
        let tid_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            tid.hash(&mut h);
            h.finish()
        };
        let depth = MODEL_LOAD_DEPTH.with(|d| d.get());
        if depth > 0 && MODEL_LOAD_OWNER.load(Ordering::Acquire) == tid_hash {
            MODEL_LOAD_DEPTH.with(|d| d.set(depth + 1));
            return ModelLoadGuard { _inner: None };
        }
        // Poison-immune: a panicking holder (test assertion) must not wedge
        // every later loader — the registry has no inconsistent state to
        // recover from (entries are atomically replaced).
        let inner = MODEL_LOAD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        MODEL_LOAD_OWNER.store(tid_hash, Ordering::Release);
        MODEL_LOAD_DEPTH.with(|d| d.set(1));
        ModelLoadGuard {
            _inner: Some(inner),
        }
    }
}

impl Drop for ModelLoadGuard {
    fn drop(&mut self) {
        let depth = MODEL_LOAD_DEPTH.with(|d| d.get());
        if depth > 1 {
            MODEL_LOAD_DEPTH.with(|d| d.set(depth - 1));
            return;
        }
        if depth == 1 {
            MODEL_LOAD_OWNER.store(0, Ordering::Release);
            MODEL_LOAD_DEPTH.with(|d| d.set(0));
        }
    }
}

/// Pinned-host staging slots for async H2D input fills (7e⑥). Pageable
/// `cudaMemcpy` forces the driver to bounce through an internal pinned
/// buffer AND blocks until the copy lands; copying into our own pinned
/// slot + `cudaMemcpyAsync` returns immediately and lets the copy overlap
/// the subsequent kernel launches on the stream. Slots form a ring: when
/// the ring wraps, one stream sync retires every in-flight copy before a
/// slot is reused (in practice input fills are KB-scale and the ring
/// never wraps within a step).
struct PinnedPool {
    ptrs: Vec<*mut u8>,
    slot_bytes: usize,
    next: usize,
}
impl Drop for PinnedPool {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            unsafe { cudaFreeHost(p as *mut std::ffi::c_void) };
        }
    }
}

unsafe impl Send for PinnedPool {}
unsafe impl Sync for PinnedPool {}

/// R3-A2: single grow-on-demand pinned staging buffer for device→host
/// readbacks (the graph logits path runs this once per decode step).
struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
}
impl Drop for PinnedBuf {
    fn drop(&mut self) {
        unsafe { cudaFreeHost(self.ptr as *mut std::ffi::c_void) };
    }
}
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

/// Viz/trace capture staging (scheduler's per-node `copy_to_host` full-stream
/// sync replaced by batched async D2H). The scheduler enqueues one async copy
/// into this pinned buffer immediately after each captured node's launch —
/// stream order preserves the value even though the pool recycles buffers
/// intra-split — and drains everything with ONE sync at the split boundary.
/// Bounded: a node larger than the remaining capacity is refused (the caller
/// falls back to the per-node sync copy); growth only happens between drains
/// (reallocating under undrained pending copies would lose them).
pub struct CaptureStaging {
    ptr: *mut u8,
    bytes: usize,
    used: usize,
    /// (offset, bytes) per enqueued copy, in enqueue order.
    pending: Vec<(usize, usize)>,
}

/// 128 MB of pinned host memory is the ceiling for one split's captured node
/// outputs; 7B decode steps (~150 nodes × ≤76 KB) fit ~20× over, prefill's
/// GB-scale FFN tensors fall back to the sync path.
const CAPTURE_STAGING_MAX: usize = 128 << 20;

impl CaptureStaging {
    pub const fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            used: 0,
            pending: Vec::new(),
        }
    }

    /// Queue an async D2H of `bytes` from device `src` into the staging.
    /// `false` = refused (alloc failure or no room): the caller must fall
    /// back to the per-node sync copy for this buffer.
    pub fn queue(
        &mut self,
        stream: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> bool {
        if bytes == 0 || self.used + bytes > CAPTURE_STAGING_MAX {
            return false;
        }
        if self.bytes < self.used + bytes {
            // grow only between drains (pending must be empty here — a full
            // buffer with pending entries was refused above)
            debug_assert!(self.pending.is_empty());
            if !self.pending.is_empty() {
                return false;
            }
            let need = (self.used + bytes).max(8 << 20);
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            if unsafe { cudaHostAlloc(&mut p, need, 0) } != 0 {
                return false;
            }
            if !self.ptr.is_null() {
                unsafe { cudaFreeHost(self.ptr as *mut std::ffi::c_void) };
            }
            self.ptr = p as *mut u8;
            self.bytes = need;
            self.used = 0;
        }
        let err = unsafe {
            cudaMemcpyAsync(
                self.ptr.add(self.used) as *mut std::ffi::c_void,
                src,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if err != 0 {
            return false;
        }
        self.pending.push((self.used, bytes));
        self.used += bytes;
        true
    }

    /// One stream sync, then hand out the queued copies (enqueue order).
    /// Empty when nothing was queued.
    pub fn drain(&mut self, state: &CudaState) -> Vec<Vec<f32>> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        state.sync();
        let mut out = Vec::with_capacity(self.pending.len());
        for &(off, bytes) in &self.pending {
            let slice =
                unsafe { std::slice::from_raw_parts(self.ptr.add(off) as *const f32, bytes / 4) };
            out.push(slice.to_vec());
        }
        self.pending.clear();
        self.used = 0;
        out
    }
}

impl Drop for CaptureStaging {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFreeHost(self.ptr as *mut std::ffi::c_void) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for CaptureStaging {}
unsafe impl Sync for CaptureStaging {}

/// r49: consecutive-window memoization of the MMQ A-quantize prepass.
///
/// The three attention GEMMs (q/k/v) all consume the SAME `normed` activation
/// and the two FFN GEMMs (gate/up) consume the same `normed2`; the MMQ path
/// re-quantizes (transpose) that A per matmul. This cache holds the quantized
/// (transposed) form of the last MMQ A so consecutive same-A matmuls reuse it
/// instead of re-launching `quantize_q8_0_pad40_t`.
///
/// Correctness relies on two rules (both conservative, no write-tracking):
///   * It is valid ONLY across CONSECUTIVE prefill-MMQ MatMul nodes. Any other
///     node kind clears it (see `CudaBackend::execute_node_inner`), so a late
///     buffer-id reuse by the liveness allocator can never alias the cached A.
///   * It is cleared at split boundaries (`CudaBackend::synchronize`) so a
///     cache from a previous graph execution never leaks stale data into a
///     later one (the same pool buffer id holds different data each step).
///
/// The buffers are the state-level `buf_qa8_t`/`buf_sda_t` (transposed) and
/// `buf_q8_prefill` (native) scratch — dedicated allocations OUTSIDE the
/// graph allocator pool, so they never alias a node output buffer. The quantize
/// output is a pure function of (src, nt, id), so a hit is byte-identical to a
/// recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MmqCache {
    /// True once a quantize has been recorded into `key`.
    active: bool,
    /// (src device pointer, nt, id) of the cached A.
    key: (usize, usize, usize),
    /// True when the cached buffers hold the transposed pad40_t layout
    /// (buf_qa8_t + buf_sda_t); false => native pad40 (buf_q8_prefill).
    transposed: bool,
    /// r52: true when the entry was recorded by a mode-2 (skip-write) fused
    /// producer — the f32 src was NEVER written, so the plane is the only
    /// valid form of that activation. Any cache path that would re-quantize
    /// the src (transposed miss / native prepass) REFUSES on such an entry
    /// (loud failure) instead of silently reading the dead buffer's garbage.
    dead_write: bool,
    /// Physical buffer pointers of the cached quantized A (validated on hit —
    /// `get_or_grow` may realloc on a larger miss).
    qa8: usize,
    sda: usize,
    q8: usize,
}

impl Default for MmqCache {
    fn default() -> Self {
        MmqCache {
            active: false,
            key: (0, 0, 0),
            transposed: false,
            dead_write: false,
            qa8: 0,
            sda: 0,
            q8: 0,
        }
    }
}

/// 8p: warm the f16 weight cache only for models whose quantized matmul
/// weights total at least this much (7B q4_k_m = 4.4 GB warms; the 0.5-1.5B
/// test fixtures do not, keeping their footprint at pre-8p levels).
pub const W16_ENABLE_BYTES: usize = 2 << 30;

pub struct CudaState {
    stream: Mutex<CudaPtr>,
    /// Lazy pinned staging ring (7e⑥), **one per device stream** since #188:
    /// the ring's slots are written by `cudaMemcpyAsync` on the bound stream, so
    /// a shared ring would let one engine's fill reuse a slot another engine's
    /// in-flight copy is still reading. Empty until the first async fill on a
    /// stream; a stream whose `cudaHostAlloc` failed falls back to a sync copy.
    staging: Mutex<HashMap<usize, PinnedPool>>,
    /// R3-A2: pinned D2H readback buffer (grown on demand; None until the
    /// first pinned read, stays None on cudaHostAlloc failure → the pageable
    /// fallback). A blocking `cudaMemcpy` into a PAGEABLE destination bounces
    /// through a driver-internal pinned buffer (see write_input_async's
    /// comment); reading into our own pinned slot skips that bounce. This is
    /// the per-decode-step logits readback (608 KB on 0.5B/7B-class vocab).
    readback: Mutex<Option<PinnedBuf>>,
    weights: Mutex<HashMap<String, (CudaPtr, usize)>>,
    /// 8p: persistent per-weight f16 dequant cache, keyed by the device
    /// weight pointer → (f16 copy, bytes). The two-pass prefill GEMM used
    /// to dequantize W on EVERY call (288 ms per 7B @2K forward); weights
    /// are immutable after registration (register_weight reuses the same
    /// device copy for same name+size and never frees on replace), so a
    /// wptr key is stable and the dequant runs once per weight per process.
    /// Adds 2 B/element (~8.6 GB on 7B q4_k_m) — MINFER_NO_W16CACHE=1
    /// reverts to the per-call scratch.
    w16_cache: Mutex<HashMap<usize, (CudaPtr, usize)>>,
    /// 8p: the f16 cache is enabled only for models whose quantized matmul
    /// weights total >= W16_ENABLE_BYTES (set by the loader's warm pass).
    /// Small models keep the per-call scratch: the test suite keeps several
    /// loaded models resident on a shared overcommitted CUDA pool and a
    /// +1-2 GB cache per loaded model tipped it over (probability OOMs).
    w16_enabled: std::sync::atomic::AtomicBool,
    /// R1: device compute capability ×100 in `major*100 + minor` encoding
    /// (GB10 sm_12.1 → 1201; note: NOT the llama.cpp tier-key encoding
    /// 1210 — see `device_tier::llama_key`), read once at init. Gates the
    /// int8-mma MMQ prefill path (needs sm_80+ — mma.m16n8k32).
    /// Its only reader is the `#[cfg(test)]` accessor [`Self::cc`]
    /// (`graph/cuda_backend/tests.rs`); the tier selector consumes the value at
    /// init before the field is stored, so a non-test build stores it unread —
    /// `#[cfg_attr(not(test), ...)]` names that configuration (#243).
    #[cfg_attr(not(test), allow(dead_code))]
    cc: std::sync::atomic::AtomicI32,
    /// T1: effective MMQ gate — the tier's own flag, or `cc >= 800` for the
    /// GENERIC row (unknown architectures keep the conservative gate).
    tier_mmq: bool,
    /// T2: SM count (queried at init, previously print-only) — feeds the
    /// auto-ksplit target parameterization.
    sm_count: i32,
    /// r60: true while every quantized weight registered on this device is
    /// NB-BT-consumable (q4_K / q6_K) — the r52 mode-2 (skip-write fused
    /// producer) window proof assumes the fused pad40_t plane is consumed
    /// ONLY by the raw NB-BT GEMM paths. On any other quant mix (Q4_0/Q5_K/
    /// Q8_0/... or a 2-D F32 matmul weight) a fused rms_norm/swiglu output
    /// can feed a generic mmq_nt consumer whose native re-quantize would
    /// read the skipped f32 — deterministic loud refusal (or silent F32
    /// garbage). The loaders clear this at registration; `mmq_a_fuse_mode`
    /// degrades mode 2 -> 1 (r51 fused semantics, writes the f32) when it
    /// is false. All-q4_K/q6_K models (7B q4_k_m: embed q4_K + output q6_K)
    /// keep mode 2 unchanged.
    nb_bt_only: std::sync::atomic::AtomicBool,
    /// Names registered through `register_weight_q6k_padded` (device layout
    /// is 224-byte-padded Q6_K, not the raw GGUF byte stream) → the
    /// ORIGINAL raw byte length, so `has_weight_of_size` can still match
    /// tensors by their raw GGUF size.
    padded_weights: Mutex<HashMap<String, usize>>,
    // doc 104: q8_0 p32 split planes — original weight device ptr -> (payload, d)
    q80_p32: Mutex<HashMap<usize, (usize, usize)>>,
    /// r53: pre-expanded q6_K B planes (dense centered-int8, `od * id` bytes,
    /// row stride = id, super-block stride = 256) built at padded registration
    /// under `MINFER_MMQ_Q6K_NB` (r60: default-on) AND `MINFER_MMQ_Q6K_EXP
    /// != "0"` (r54: explicit "0" skips the ~1.5 GB plane build entirely),
    /// keyed by the
    /// PADDED weight's device pointer. `prefill_mmq` looks the plane up by
    /// `wptr` and passes it to the NB-BT q6_K launcher; a miss (gate off at
    /// load, allocation failure, raw 210-B layout) keeps the r41 in-kernel
    /// expand.
    q6k_exp: Mutex<HashMap<usize, CudaPtr>>,
    /// r53: the W_exp allocation-failure warning prints once per process.
    q6k_exp_warned: std::sync::atomic::AtomicBool,
    /// r56 (Session E item 2b): per-tensor precomputed dsc f32 pairs for the
    /// NB-BT q6_K kernel — plane[c*od + j] = float2(d*sc0, d*sc1), chunk-major
    /// so the per-kt staging is a contiguous 16-B cp.async stream. Keyed by the
    /// PADDED weight's device pointer exactly like `q6k_exp`; a miss (gate off,
    /// alloc failure, odd od) keeps the r41 scalar dsc path.
    q6k_dpl: Mutex<HashMap<usize, CudaPtr>>,
    q6k_dsc: Mutex<HashMap<usize, CudaPtr>>,
    /// r56: the W_dsc allocation-failure warning prints once per process.
    q6k_dsc_warned: std::sync::atomic::AtomicBool,
    /// r59: q4_K W_dsc f32-pair planes — float2(d*sc, -(dmin*m)) per
    /// (chunk, od-row), chunk-major — keyed by the RAW weight's device
    /// pointer (the r56 q6_K map pattern). A map miss keeps the in-kernel
    /// get_scale_min_k4 decode (the DSC=false instantiation).
    q4k_dsc: Mutex<HashMap<usize, CudaPtr>>,
    /// r59: the q4_K W_dsc allocation-failure warning prints once per process.
    q4k_dsc_warned: std::sync::atomic::AtomicBool,
    /// r59 rider: max id/32 seen at K-quant registration — the pre-warm
    /// MmqCache scratch sizing hint (0 until a K-quant weight registers).
    max_nchunk: std::sync::atomic::AtomicUsize,
    /// 8c: prefill Q8_0-activation scratch (quantized activations for the
    /// Q4_0×Q8_0 GEMM, nt > 1). Grown on demand like the layer-path buffers.
    buf_q8_prefill: StreamScratch,
    /// P6 r34: transposed-A q8_0 prepass scratch — the swizzled qs plane
    /// ([ntb][nchunk][2048]) and the packed d|ssum scale ([ntb][nchunk][256]),
    /// consumed by mmq_raw_nb_bt_kernel's bulk staging (MINFER_MMQ_A_TRANSPOSE).
    buf_qa8_t: StreamScratch,
    buf_sda_t: StreamScratch,
    /// doc 92: K-split fp32 partials ([ksplit][nt][od]) for the BT GEMM's
    /// block-starvation fix at small nt. Grown on demand like the other
    /// prepass scratches; the reduce kernel consumes it on the same stream.
    buf_mmq_ksplit: StreamScratch,
    /// r49: consecutive-window memoization of the MMQ A-quantize prepass (see
    /// [`MmqCache`]). **Per stream since #188**: an entry records the device
    /// pointer of the per-stream scratch plane, so a shared memo would hand one
    /// engine's plane to another. Still keyed on the (src, nt, id) window, so
    /// the CUDA backend invalidates its own stream's entry between non-MMQ nodes
    /// / graph executions.
    mmq_cache: Mutex<HashMap<usize, MmqCache>>,
    /// 8d: split-K attention partials ([8][nh][pstr] floats, nh/hd are graph
    /// constants so the size is stable — grown during warmup, never inside a
    /// capture window).
    buf_attn_partial: StreamScratch,
    /// 8e-reversal: decode MMVQ q8 activation scratch (nt=1, so id/32 * 40B
    /// per token — size-stable per graph, grown during warmup runs).
    buf_q8_decode: StreamScratch,
    /// 8m: prefill f16 GEMM scratch — dequantized weights (od*id halves) and
    /// converted activations (nt*id halves), grown on demand. Prefill never
    /// enters a CUDA Graph capture window (8g①), so the grow is capture-safe
    /// (same assumption as the 8c buf_q8_prefill).
    buf_f16_w: StreamScratch,
    buf_f16_x: StreamScratch,
}

/// Quant block element count (ggml block_q): 256 for K-quants, 32 otherwise.
fn quant_block_q(t: TensorType) -> usize {
    match t {
        TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K => 256,
        _ => 32,
    }
}

/// Quant block byte size — matches ggml type_size (Q4_0=18, Q4_1=20, Q5_0=22,
/// Q5_1=24, Q8_0=34, Q4_K=144, Q5_K=176, Q6_K=210).
fn quant_block_bytes(t: TensorType) -> usize {
    match t {
        TensorType::Q4_0 => 18,
        TensorType::Q4_1 => 20,
        TensorType::Q5_0 => 22,
        TensorType::Q5_1 => 24,
        TensorType::Q8_0 => 34,
        TensorType::Q4_K => 144,
        TensorType::Q5_K => 176,
        TensorType::Q6_K => 210,
        _ => 0,
    }
}

/// Concatenate raw quantized weights along the output (row) dimension into one
/// weight buffer for a fused matmul (nt==1 decode): the matmul kernel lays
/// weights out as [out rows][blocks][block bytes], so a row-major concat is
/// contiguous. Returns None when the weights can't share a single matmul
/// (different types, different input dims, or an unsized type).
pub fn concat_rows(tensors: &[&Tensor]) -> Option<Vec<u8>> {
    if tensors.len() < 2 {
        return None;
    }
    let tt = tensors[0].ttype;
    if tensors.iter().any(|t| t.ttype != tt) {
        return None;
    }
    let bq = quant_block_q(tt);
    let bb = quant_block_bytes(tt);
    if bb == 0 {
        return None;
    }
    let ne0 = tensors[0].shape[0] as usize;
    if tensors.iter().any(|t| t.shape[0] != ne0 as i64) {
        return None;
    }
    if ne0 % bq != 0 {
        return None;
    }
    let row = (ne0 / bq) * bb;
    let rows: usize = tensors.iter().map(|t| t.shape[1] as usize).sum();
    let mut out = Vec::with_capacity(rows * row);
    for t in tensors {
        out.extend_from_slice(t.data());
    }
    if out.len() != rows * row {
        return None;
    }
    Some(out)
}

/// Metadata-only `concat_rows` feasibility check — no byte copying. The
/// decode-graph build probes concat availability per layer; the eager
/// variant re-concatenated ~1.9 GB (28 ffn gate/up pairs on 7B) on every
/// decode graph build, measured as a ~920 ms one-time stall at the
/// prefill→decode switch. The loader performs the real concatenation once
/// at model load and registers `blk.{i}.ffn_gu`; both paths must agree, so
/// this mirrors concat_rows' precondition checks exactly (the per-tensor
/// data-length check subsumes the final `out.len() != rows * row` test).
pub fn concat_rows_feasible(tensors: &[&Tensor]) -> bool {
    if tensors.len() < 2 {
        return false;
    }
    let tt = tensors[0].ttype;
    if tensors.iter().any(|t| t.ttype != tt) {
        return false;
    }
    let bq = quant_block_q(tt);
    let bb = quant_block_bytes(tt);
    if bb == 0 {
        return false;
    }
    let ne0 = tensors[0].shape[0] as usize;
    if tensors.iter().any(|t| t.shape[0] != ne0 as i64) {
        return false;
    }
    if ne0 % bq != 0 {
        return false;
    }
    let row = (ne0 / bq) * bb;
    tensors
        .iter()
        .all(|t| t.data().len() == row * (t.shape[1] as usize))
}

/// 8b / C4 S2b: the GPU KV cache **layout** tag the kernels are templated on. The
/// three codes are the `KvFormat` discriminants, so the same number names the same
/// layout on both sides of the FFI boundary:
///
/// - `0` f32 — one f32 per element;
/// - `1` f16 — one f16 per element in the first half of the f32-shaped region;
/// - `2` q8_0 — packed 34-byte Q8_0 blocks, one cell rounded up to whole f32 words.
///
/// Before C4 S2b this was a bool and anything that was not exactly `f16` became
/// `false` — so a `q8_0` region would have been addressed as f32 rows. The layout
/// is now a first-class three-valued tag and `q8_0` is never silently folded
/// into f32.
pub const KV_LAYOUT_F32: i32 = 0;
pub const KV_LAYOUT_F16: i32 = 1;
pub const KV_LAYOUT_Q8_0: i32 = 2;

/// #186: the packed Q8_0 **decode** K dot's same-binary A/B control, and where the
/// launcher's answer comes from. `MINFER_NO_DP4A_Q8_KV=1` selects the incumbent
/// convert-based `kv4<KV_LAYOUT_Q8_0>` load; unset (or any other value) selects the
/// `__dp4a` int accumulation against the lane's quantized query.
///
/// Read **once per process** and passed to the launcher as a value: the decode graph
/// is captured, and an answer that flipped mid-process would select a different
/// kernel than the one the captured exec recorded — exactly the class of change
/// `cuda_backend.rs::graph_replay_step` refuses for a moved layout. Resolving it here
/// (rather than inside the `.cu`) also lets the Rust side bump the observation
/// counter only when the arm it launched is the one under test.
pub fn q8_kv_dp4a_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MINFER_NO_DP4A_Q8_KV").as_deref() != Ok("1"))
}

/// #202: the packed Q8_0 **decode** K/V four-quant load's same-binary A/B control.
/// `MINFER_NO_Q8_KV_WIDE=1` selects the incumbent four byte loads
/// (`q8_0_load4_bytes`); unset (or any other value) selects the two 16-bit loads
/// (`q8_0_load4_wide`) that halve the L1 request count for the same bytes.
///
/// The load arm is only meaningful together with the `__dp4a` K dot, so
/// `q8_kv_dp4a_enabled() == false` already implies the byte form — the launcher
/// checks `dp4a && wide`. Read **once per process** for the same captured-graph
/// reason as [`q8_kv_dp4a_enabled`].
pub fn q8_kv_wide_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("MINFER_NO_Q8_KV_WIDE").as_deref() != Ok("1"))
}

/// The `KV_LAYOUT_*` tag for a `KvFormat` — the one place the two names are tied
/// together, exhaustive over the enum so a fourth format cannot be added without a
/// compile error here.
///
/// **Per engine since #153.** The tag used to be a `static KV_LAYOUT` this module
/// owned and every device kernel read; the loaded engine's resolved `KvFormat` now
/// reaches its own `CudaBackend` through `GraphAllocator::set_kv_format`, so this is
/// a pure translation, not a policy. A process that loads two engines with different
/// formats gives each `CudaBackend` its own tag, and the captured-graph identity
/// records it.
pub fn layout_of(format: crate::graph::kvformat::KvFormat) -> i32 {
    use crate::graph::kvformat::KvFormat;
    match format {
        KvFormat::F32 => KV_LAYOUT_F32,
        KvFormat::F16 => KV_LAYOUT_F16,
        KvFormat::Q8_0 => KV_LAYOUT_Q8_0,
    }
}

/// The `KvFormat` a `KV_LAYOUT_*` tag names. The inverse of [`layout_of`]; an
/// unknown tag is F32, the pre-C4 reading, and is only reachable from an internal
/// bug (the tag is never parsed from a file or the environment).
/// Test-only (#238): driven by `cuda::kv_dtype_tests::the_layout_tag_is_the_format_discriminant`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn format_of(layout: i32) -> crate::graph::kvformat::KvFormat {
    use crate::graph::kvformat::KvFormat;
    match layout {
        KV_LAYOUT_F16 => KvFormat::F16,
        KV_LAYOUT_Q8_0 => KvFormat::Q8_0,
        _ => KvFormat::F32,
    }
}

#[cfg(test)]
mod kv_dtype_tests;

/// Issue #223 control: `MINFER_NO_GEMM_PREWARM=1` skips the eager prefill-GEMM
/// dynamic-smem pre-warm at context creation. It exists so the **same binary**
/// can measure the pre-warm on/off (the performance A/B) and so the "lazy path
/// alone" arms of the #218 gates can run in a fresh process. Unset — or any
/// value other than `1` — is the production default: pre-warm. The lazy
/// per-launch opt-in is unaffected either way; it is the universal fallback.
fn gemm_prewarm_disabled() -> bool {
    std::env::var("MINFER_NO_GEMM_PREWARM").as_deref() == Ok("1")
}

#[cfg(test)]
mod d35_probe_tests;

// ────────────────────────────────────────────────────────────────────
// D3-8 probes: FusedQKV decode fusion (CUDA port of the Metal G4 path)
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod d38_probe_tests;

// ────────────────────────────────────────────────────────────────────
// Issue #145: the eager prefill-GEMM smem opt-in is checked, and a latched
// API error is reported with its real origin (never as a kernel launch).
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod issue145_tests;

// ────────────────────────────────────────────────────────────────────
// Issue #147: the remaining unchecked attribute / launch / destroy returns.
//
// The C++ sites (cuda_kernels.cu) report through `minfer_site_fail_*`; the Rust
// destroy site formats its own message (`graph_destroy_failure_message`). The
// deliberate failures are env-gated behind `MINFER_TEST_ISSUE147=1` because they
// *really* fail a CUDA call — a `compute-sanitizer --tool memcheck` run must not
// see them (the same reason #145's latch gate is gated). With the knob off these
// gates skip; the two pure gates always run.
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod issue147_tests;

// ────────────────────────────────────────────────────────────────────
// #162: the launch-return gate. Every `<<<>>>` in src/cuda_kernels.cu now reads
// its own error through `minfer_launch_ok` / `minfer_launch_ok_opt` and carries
// an injection lever (`minfer_launch_block` / `minfer_launch_smem`), which
// `scripts/check_cuda_launch_returns.py` audits statically and the CI job runs.
// This gate is the runtime half: it drives every launcher through the branches
// that reach each audited site with that site armed, and asserts the site named
// itself, named its kernel instantiation and `cudaGetErrorName`, and left no
// latch for `CudaState::sync`.
//
// Deliberate-failure gates make REAL CUDA calls fail, so they are gated behind
// `MINFER_TEST_ISSUE162=1` and never run in a compute-sanitizer pass. The pure
// `severity` test below runs in a default (and CI) run.
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod issue162_tests;

// ────────────────────────────────────────────────────────────────────
// #218 → #223: the prefill-GEMM dynamic-smem opt-in invariant.
//
// `gemm_prefill_smem_init` — the eager startup sweep — had no production
// caller after #188 and was annotated away by the dead-code campaign. #218
// (plan B) kept the lazy per-launch opt-in (`gemm_smem_optin`, reached through
// `launch_gemm_f16`) and made the invariant explicit and gated. #223 restores
// the **runtime guarantee** on top: `CudaState::try_new` now drives the same
// production entry eagerly, once per process, before any stream or capture
// window can exist, so the invariant no longer depends only on the three
// emergent mechanisms (the 3-run warmup, `cudaStreamCaptureModeThreadLocal`,
// the per-instantiation cache), which are demoted to defence in depth.
//
// The #218 arms below keep their claims by running against the documented
// control `MINFER_NO_GEMM_PREWARM=1` (the "lazy path alone" configuration),
// which is where the pre-#223 preconditions (`opted_in == 0` before the first
// launch) are observable:
//
// - `cuda_prefill_smem_optin_is_done_by_production` — with the pre-warm off, a
//   real prefill forward still opts a >48 KiB instantiation in, asserted
//   through the device's own read-back, with the "not already opted in"
//   precondition established by running in a fresh process (the tile config and
//   the attribute are both process-scoped);
// - `cuda_prefill_smem_optin_refusal_fails_the_prefill` — the control arm: the
//   `attr:gemm_f16_f16` injection makes the prefill refuse the launch loudly;
// - `graph::cuda_backend::tests::capture::
//   cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` — with the
//   pre-warm off, a >48 KiB prefill-shaped graph is captured and replays
//   bitwise, and the opt-in is shown to run **before** the window opens, never
//   inside it;
// - `issue145_tests::cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation`
//   — every launchable >48 KiB instantiation reads back opted in through the
//   production function (a cache-keying regression detector, whichever path set
//   the attribute).
//
// #223's own gate lives in `issue223_tests`: with the pre-warm ON (the default
// process), every launchable >48 KiB instantiation already reads back opted in
// immediately after context creation and before any launch.
//
// `test_child` is the fresh-process harness; the sibling #145 gate in
// `issue145_tests` drives the same lazy production opt-in.
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
pub(crate) mod issue218_tests;
#[cfg(test)]
pub(crate) mod issue223_tests;
#[cfg(test)]
pub(crate) mod test_child;

// #239: the moved bucket-B helpers and test-only FFI declarations.
#[cfg(test)]
mod tests;
