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

// ─── FFI declarations for CUDA runtime API ────────────────────

// Sized buffer for cudaGetDeviceProperties (avoids fragile field-by-field layout).
// Only the first 256 bytes (device name) are read; extra padding handles any CUDA version.
#[repr(C)]
struct CudaDevicePropBuf([u8; 4096]);

extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: std::ffi::c_int) -> *mut std::ffi::c_void;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaHostAlloc(ptr: *mut *mut std::ffi::c_void, size: usize, flags: i32) -> i32;
    fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    // Issue #188: a **per-backend** stream is created with
    // `cudaStreamNonBlocking` so it does not take part in the legacy default
    // stream's implicit global synchronization. Without that flag every
    // explicit stream implicitly synchronizes with the null stream, and a
    // host-side blocking `cudaMemcpy` (weight registration, readback) on one
    // thread would serialize — and, inside another thread's capture window,
    // invalidate — every engine's stream.
    fn cudaStreamCreateWithFlags(stream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    // Issue #188: is `stream` currently inside a capture window? The only reader is
    // `CudaState::stream_is_capturing` below, whose only caller is the #188 probe in
    // `graph/cuda_backend/tests.rs`; the production capture bookkeeping is the
    // per-instance `CudaBackend::capturing` field. (The older note here claimed a
    // registration-path refusal that no longer exists — corrected by #238.)
    /// Test-only (#238): driven by `graph::cuda_backend::tests::cuda_capture_abort_on_error`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    fn cudaStreamIsCapturing(stream: *mut std::ffi::c_void, status: *mut i32) -> i32;
    // F5 (#58): events, the synchronization primitive the split boundary's async
    // staging copies need. `cudaEventRecord` marks a point on the stream;
    // `cudaStreamWaitEvent` makes a later consumer wait on it **without blocking
    // the host**; `cudaEventSynchronize` is the host-side wait and is the one
    // documented synchronization point of a device→host staging copy.
    fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn cudaEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
    /// Kept for the one copy class no backend implements yet, and named by the
    /// census brief [#244](https://github.com/yusiwen/minfer/issues/244):
    /// `docs/BACKEND-REGISTRY-DESIGN.md` §11 documents it as a device consumer's
    /// synchronization point.
    ///
    /// Still dead in every compilable configuration. #138 (the F5 deferred wait,
    /// landed 2026-10-04) was the ticket expected to wire it, and it did **not**
    /// create a caller: the only pair that could express a device destination
    /// would need two device backends, `copy_across` early-returns on a
    /// same-backend pair, and Metal declines phase A. The reachable
    /// device-consumer direction (CPU → device) needs no event — its fill is
    /// stream-ordered on the consuming pool's own stream.
    #[allow(dead_code)]
    fn cudaStreamWaitEvent(
        stream: *mut std::ffi::c_void,
        event: *mut std::ffi::c_void,
        flags: u32,
    ) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    // Issue #122: the *name* of a CUDA error code, so a failed memory query can say
    // `cudaErrorIllegalAddress (700)` instead of a bare number (or nothing at all).
    fn cudaGetErrorName(error: i32) -> *const std::os::raw::c_char;
    fn cudaGetDeviceProperties(prop: *mut CudaDevicePropBuf, device: i32) -> i32;
    // T2 device-adaptation queries (plan §6): smem feasibility for the BT
    // tile config. The externs query the CURRENT device (R2 fix).
    // cuda_shared_per_sm stays C-side only: per-block optin <= per-SM on
    // every arch, so the per-block check below subsumes it.
    fn cuda_shared_per_block_optin() -> i32;
    fn cuda_mmq_smem_bytes() -> i32;
    // CUDA Graph APIs
    fn cudaStreamBeginCapture(stream: *mut std::ffi::c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(
        stream: *mut std::ffi::c_void,
        graph: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn cudaGraphInstantiate(
        exec: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        error_node: *mut std::ffi::c_void,
        log_buf: *mut u8,
        buf_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    // A `cudaGraph_t` (the result of `cudaStreamEndCapture`) and a
    // `cudaGraphExec_t` (the result of `cudaGraphInstantiate`) are different
    // handle types with different destroy calls. Passing an exec to
    // `cudaGraphDestroy` returns `cudaErrorInvalidValue` and leaks the exec
    // (issue #145).
    fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
    fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;
}

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

// ─── FFI declarations for kernel launch wrappers ───────────

extern "C" {
    fn launch_q4_0_q8_0_matmul(
        weights: *const u8,
        acts: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_1_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_f32_matmul_padded(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_quantize_q8_0(
        x: *const f32,
        y: *mut u8,
        dim: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_f32_f32_matmul(
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // #141: f16 weights × f32 activations (raw half bits). Both return 0 on
    // success and non-zero when the launch itself failed (#147 checking; the
    // caller turns it into an `Err`, never a silent fallback).
    fn launch_f16_f32_matmul(
        w: *const u8,
        x: *const f32,
        out: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn launch_embed_rows_f16(
        w: *const u8,
        ids: *const f32,
        out: *mut f32,
        n_embd: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn launch_swiglu_f32_off(buf: *mut f32, n: i32, off: i32, stream: *mut std::ffi::c_void);
    // D3-5 1a: fused-producer decode A-quantize (rms_norm/swiglu + pad40
    // epilogue; q8 bytes bit-identical to quantize_q8_0_pad40).
    fn launch_rms_norm_quant_pad40(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        q8: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_swiglu_quant_pad40(
        buf: *mut f32,
        q8: *mut u8,
        n: i32,
        off: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gather_rows_f32(
        src: *const f32,
        ids: *const f32,
        out: *mut f32,
        n: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_embed_rows(
        w: *const u8,
        ids: *const f32,
        out: *mut f32,
        n_embd: i32,
        nt: i32,
        type_id: i32,
        block_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_rms_norm_f32(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        d: i32,
        eps: f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_add_bias_f32(
        y: *mut f32,
        b: *const f32,
        d: i32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_add_f32(
        x: *const f32,
        y: *const f32,
        z: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mul_f32(
        x: *const f32,
        y: *const f32,
        z: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_silu_f32(y: *mut f32, n: i32, stream: *mut std::ffi::c_void);
    fn launch_swiglu_f32(
        gate: *const f32,
        up: *const f32,
        dst: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    // r51: producer-fused rms_norm/swiglu + pad40_t A-quantize (MINFER_MMQ_A_FUSE)
    fn launch_rms_norm_quant_f32_t(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        yqs: *mut u8,
        ysda: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_swiglu_quant_f32_t(
        gate: *const f32,
        up: *const f32,
        dst: *mut f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    // r52: mode-2 skip-write variants (MINFER_MMQ_A_FUSE=2) — same planes, no
    // f32 output write (window-safety proof: docs/CUDA_OPTIMIZATION.md P6 r52).
    fn launch_rms_norm_quant_nw_f32_t(
        x: *const f32,
        w: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_swiglu_quant_nw_f32_t(
        gate: *const f32,
        up: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_f32_bits_to_i32(
        src: *const f32,
        dst: *mut i32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_rope_f32(
        x: *mut f32,
        n_head: i32,
        n_dims: i32,
        nt: i32,
        freq_base: f32,
        freq_scale: f32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_store_kv_f32(
        src: *const f32,
        dst: *mut f32,
        nkt: i32,
        nt: i32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_f32(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        bound: *const i32,
        mode: i32,
        layout: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        row_bytes: usize,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // 8m: prefill dequant-to-f16 + wmma HGEMM. The f16 pointers cross the
    // boundary as c_void (Rust has no __half); the type_id mapping is
    // documented at launch_dequant_f16 in cuda_kernels.cu.
    fn launch_dequant_f16(
        type_id: i32,
        w: *const u8,
        out: *mut std::ffi::c_void,
        od: i32,
        id: i32,
        block_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_convert_f16(
        x: *const f32,
        out: *mut std::ffi::c_void,
        n: i64,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gemm_f16(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        af32: bool,
    ) -> i32;
    // #223: the **production** eager pre-warm entry, called once per process
    // from `CudaState::try_new` for every launchable `(tm, ks, af32)`. It drives
    // the same `gemm_smem_optin<TM,KS,AF32>` cache `launch_gemm_f16` reads, so
    // the pre-warm and the lazy path are one mechanism, not two. Outcome:
    //  1 = the attribute is in force (set now, or already cached);
    //  0 = refused — `minfer_smem_optin` named the instantiation, the requested
    //      bytes, the device limit and `cudaGetErrorName`, and cleared the latch;
    // -1 = the combination is not in the fatbin;
    // -2 = deliberately skipped: the request exceeds the device's
    //      `cudaDevAttrMaxSharedMemoryPerBlockOptin`, so the attribute was never
    //      called (the reason is named by `minfer_smem_optin`).
    fn gemm_prefill_smem_prewarm_one(tm: i32, ks: i32, af32: i32) -> i32;
    // #218: the removed eager sweep's (`gemm_prefill_smem_init`) introspection
    // lives in the `#[cfg(test)] extern "C"` block below, so no test-only
    // declaration is carried by a non-test build. The `checked`/`skipped`
    // counters did not come back with #223's pre-warm.
    //
    // #145/#147/#162 test introspection (`cuda_test_latch_oversized_smem`,
    // `minfer_site_fail_*`, `minfer_site_hist_{len,site,name,msg}`) moved to
    // `cuda::tests` by #239: their only callers are the `cuda::*_tests` modules.
    // P6: A arrives as f32 activations; the GEMM converts on stage — the
    // separate convert_f32_f16 pass disappears for every prefill matmul.
    // #147: the same shape for the af32 path — 1 = launched and accepted; 0 =
    // the opt-in or the launch was refused (named at the site), which the
    // caller turns into an `Err`.
    fn launch_gemm_f32a(
        a: *const f32,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // #162: the sticky required-launch failure and the ordered launch-site
    // history. `minfer_launch_ok` sets the sticky for a REQUIRED site;
    // `CudaBackend::execute_node` drains it and returns an `Err` naming the site,
    // so one Rust-side check covers every launcher. `minfer_launch_ok_opt` (a
    // documented fallback) never sets it. The history records every named launch
    // failure so the #162 gate can see more than one site per call.
    fn minfer_launch_fail_pending() -> i32;
    fn minfer_launch_fail_site() -> *const std::os::raw::c_char;
    fn minfer_launch_fail_name() -> *const std::os::raw::c_char;
    fn minfer_launch_fail_code() -> i32;
    fn minfer_launch_fail_clear();
    // 8p: fused dequant-in-GEMM — B tiles dequantize raw quantized bytes
    // in-register (no f16 weight scratch round trip). type_id mapping as in
    // launch_dequant_f16; q6_stride = 210 raw / 224 padded (only Q6_K reads
    // it). Requires id % 256 == 0 (host gate).
    fn launch_gemm_qb_nt(
        a: *const std::ffi::c_void,
        w: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        type_id: i32,
        q6_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // R1: int8 MMQ prefill GEMM — q8_0-quantized activations (pad40 blocks
    // with the per-block int sum at offset 36) × raw quantized weights, tiled
    // mma.m16n8k32/m16n8k16 (s8) with per-k-block scale rescale. type_id as
    // in launch_dequant_f16; q6_stride = 210 raw / 224 padded (Q6_K only).
    // Requires id % 32 == 0 and sm_80+ (int8 mma; sm_75 falls back).
    // #147: 1 = launched and accepted, 0 = refused (named at the site).
    fn launch_mmq_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        q6_stride: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // P6: raw-byte staging MMQ (q4_K, whole 256-k super-blocks).
    fn launch_mmq_raw_wide_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // P6 Direction-A: NB raw-nibble MMQ (q4_K, KD=8 native, 64x128, 2 blk/SM).
    // Returns 1 when it ran (KD=8), 0 on clean fallback (KD!=8 / smem / regs).
    fn launch_mmq_raw_nb_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // P6 r34: NB kernel whose A staging is bulk LDG->STS over the
    // PRE-TRANSPOSED qa8/sda buffers (MINFER_MMQ_A_TRANSPOSE=1). Returns 1 on
    // KD=8, 0 on clean fallback (KD!=8 / null transposed buffers / smem).
    // r59 rider: kernel module pre-load (cudaFuncGetAttributes over the
    // MMQ/FA/fused launch set) — see CudaState::prewarm_prefill.
    fn minfer_prewarm_kernels();
    // r59: `w_dsc` selects the DSC=true instantiation (registration-time
    // W_dsc f32-pair plane; null = in-kernel scalar decode, DSC=false).
    fn launch_mmq_raw_nb_bt_nt(
        type_id: i32,
        w: *const u8,
        w_dsc: *const u8,
        qa8g: *const u8,
        sdag: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        nchunk: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
        cpart: *mut f32,
        ksplit: i32,
    ) -> i32;
    // P6 r38: q6_K BT kernel — the same bulk-LDG->STS A staging as r34, but the
    // B weight is EXPANDED to centered int8 (256 B/row) in staging and the mma is
    // m16n8k16 (KSPLIT=2) with a per-16-sub dsc rescale. r53: `w_exp` selects
    // the pre-expanded-B cp.async instantiation (null = r41 in-kernel expand).
    // Returns 1 on KD=8, 0 on clean fallback.
    fn launch_mmq_raw_nb_bt_q6k_nt(
        type_id: i32,
        w: *const u8,
        w_exp: *const u8,
        w_dsc: *const u8,
        qa8g: *const u8,
        sdag: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        nchunk: i32,
        bstride: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
        cpart: *mut f32,
        ksplit: i32,
    ) -> i32;
    // #147: 1 = launched and accepted, 0 = refused (the opt-in or the launch
    // failed and was named at the site); the caller turns a 0 into an `Err`.
    fn launch_mmq_raw_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // 8n: FA-style prefill attention. Returns -1 when the >48KB dynamic
    // shared-memory opt-in fails (then Rust falls back to the legacy kernel).
    // #144 item 3: `layout` is the KV tag the staging reads (f16 or packed
    // Q8_0 — the packed arm dequantizes each cell into the same f16 tile);
    // `row_bytes` is the cell's byte width and is ignored by the f16 arm.
    fn launch_fa_prefill_kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        layout: i32,
        row_bytes: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn launch_store_kv_f16(
        src: *const f32,
        dst: *mut std::ffi::c_void,
        nkt: i32,
        nt: i32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    // C4 S2b: the packed store. `row_bytes` is the Q8_0 cell's byte width
    // (`KvFormat::Q8_0.row_bytes(nkt)`), so the kernel writes each 34-byte block
    // at its cell's own stride.
    fn launch_store_kv_q8_0(
        src: *const f32,
        dst: *mut std::ffi::c_void,
        nkt: i32,
        nt: i32,
        row_bytes: usize,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    // D3-8: fused decode QKV epilogue — bias×3 + rope×2 + store×2 in one
    // launch (CUDA port of Metal's attn_bias_rope_store, G4 FusedQKV)
    fn launch_attn_bias_rope_store(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        bias_q: *const std::ffi::c_void,
        bias_k: *const std::ffi::c_void,
        bias_v: *const std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: i32,
        nkt: i32,
        hd: i32,
        freq_base: f32,
        freq_scale: f32,
        positions: *const i32,
        cells: *const i32,
        kv_is_f16: i32,
        stream: *mut std::ffi::c_void,
    );
    // #144 item 1: the packed arm of the fused decode QKV epilogue. One thread
    // per (head, 32-element K block) and per V block, so a whole Q8_0 block is
    // quantized by its owner; `row_bytes` is the packed cell's byte width.
    fn launch_attn_bias_rope_store_q8_0(
        q: *mut f32,
        k: *const f32,
        v: *const f32,
        bias_q: *const std::ffi::c_void,
        bias_k: *const std::ffi::c_void,
        bias_v: *const std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: i32,
        nkt: i32,
        hd: i32,
        freq_base: f32,
        freq_scale: f32,
        positions: *const i32,
        cells: *const i32,
        row_bytes: usize,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_f32_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_kv_move_rows(
        dst: *mut f32,
        src: *const f32,
        dst_row: i32,
        src_row: i32,
        rows: i32,
        elems: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn launch_gqa_attn_split_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_quantize_q8_0_pad40(
        x: *const f32,
        y: *mut u8,
        dim: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // P6 r34: transposed-A q8_0 quantize prepass — emits the qs plane swizzled
    // per-64-token-block ([ntb][nchunk][2048]) and the d|ssum packed scale
    // ([ntb][nchunk][256]) for mmq_raw_nb_bt_kernel's bulk staging.
    fn launch_quantize_q8_0_pad40_t(
        x: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // doc 103: q4_0 / q8_0 decode MMVQ (the 8e structure on the legacy
    // f32-activation types; new code only — see cuda_kernels.cu tail).
    fn launch_q4_0_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_0_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // doc 104: q8_0 p32 split-plane variants (payload plane + dense d plane)
    fn launch_q8_0_p32_q8_mmvq(
        plane_p: *const u8,
        plane_d: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_p32_q8_mmvq_multi(
        plane_p: *const u8,
        plane_d: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // Step 82: multi-token (nt in [2, 8]) MMVQ variants — the token loop is
    // in-block (one block per weight row, grid.y = 1), so the weight stream
    // is paid once regardless of nt (the doc 81 D5-1a dispatch hole).
    fn launch_q4_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // R2: weight-streaming rework (one weight byte loaded exactly once per
    // row; uint4 loads; q6_K needs the padded 224B stride for alignment).
    fn launch_q4_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // D3b-1b: pipelined q6_K MMVQ for tall rows (npair > 256); bitwise-identical
    fn launch_q6_k_q8_mmvq_v2_pf(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // D4-4 L1: dense split-plane (dpl) decode kernels — bitwise-identical
    // to the padded forms, no 224B pad sectors (see the kernel comments).
    fn launch_q6_k_q8_mmvq_v2_pf_dpl(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        nbe: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_v2_dpl(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        nbe: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_1_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_split_f32kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        stream: *mut std::ffi::c_void,
    );
    // C4 S2b: the packed decode path (nt == 1). One 1-warp split-K launch per
    // window mode, `rpw_gate = 0` — the hybrid 4-warp body is f16-typed.
    fn launch_gqa_attn_split_q8_0(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        row_bytes: usize,
        dp4a: i32,
        wide: i32,
        stream: *mut std::ffi::c_void,
    );
    // doc 94: batched split attention for the verify shapes (1 < nt <= 16) —
    // bitwise-equal per position to the nt=1 decode split path.
    fn launch_gqa_attn_split_batched_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn launch_gqa_attn_split_batched_f32kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

// #218: introspection for the prefill-GEMM dynamic-smem gates, test-only by
// construction (`#[cfg(test)]`), so a non-test build carries neither the
// declaration nor an `allow(dead_code)` for it. `gemm_smem_need` is the
// single-source byte formula the launcher reads; `gemm_smem_opted_in` is the
// device's own `cudaFuncGetAttributes().maxDynamicSharedSizeBytes` read back;
// `gemm_prefill_smem_limit` is the queried
// `cudaDevAttrMaxSharedMemoryPerBlockOptin`;
// `gemm_smem_optin_in_capture_count` counts opt-in attempts made while the
// launch stream was capturing (the design says never);
// `gemm_prefill_smem_optin_one_for_test` drives production's
// per-instantiation `gemm_smem_optin` for one compiled combination.
#[cfg(test)]
extern "C" {
    pub(crate) fn gemm_prefill_smem_limit() -> i32;
    pub(crate) fn gemm_smem_need(tm: i32, ks: i32, af32: i32) -> usize;
    pub(crate) fn gemm_smem_opted_in(tm: i32, ks: i32, af32: i32) -> i32;
    pub(crate) fn gemm_smem_optin_in_capture_count() -> i32;
    pub(crate) fn gemm_prefill_smem_optin_one_for_test(
        tm: i32,
        ks: i32,
        af32: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
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
/// Test-only (#238): driven by `graph::cuda_backend::tests::capture_window_on_one_thread_survives_a_weight_registration_on_another`; `#[cfg(test)]` keeps it out of production builds.
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

impl CudaState {
    /// Preload the NVIDIA driver library.
    ///
    /// libcudart resolves `libcuda.so.1` through the loader's default search,
    /// which misses distribution-specific driver paths — and nix shells do not
    /// consult `/etc/ld.so.cache` at all (the binary then fails with
    /// cudaGetDeviceCount err 35, CUDA_ERROR_LIBRARY_NOT_FOUND). Loading it up
    /// front from the well-known locations makes cudart's own dlopen re-use
    /// the resident object (SONAME match). Harmless no-op when the driver is
    /// already loadable; libcuda's glibc-stub dependencies resolve via the
    /// binary's DT_RPATH (nix glibc dir) or the system default paths.
    fn preload_driver() {
        const RTLD_NOW: std::ffi::c_int = 2;
        const RTLD_GLOBAL: std::ffi::c_int = 0x100;
        const CANDIDATES: &[&str] = &[
            "libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib/libcuda.so.1",
        ];
        for c in CANDIDATES {
            let Ok(cstr) = std::ffi::CString::new(*c) else {
                continue;
            };
            let handle = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if !handle.is_null() {
                return;
            }
        }
    }

    fn try_new(requested: Option<i32>) -> Option<Self> {
        Self::preload_driver();
        if std::env::var("MINFER_DISABLE_CUDA").is_ok() {
            eprintln!("CUDA: disabled by MINFER_DISABLE_CUDA");
            return None;
        }

        let mut count: i32 = 0;
        let err = unsafe { cudaGetDeviceCount(&mut count) };
        if err != 0 || count == 0 {
            eprintln!("CUDA: no CUDA devices found (cudaGetDeviceCount err {err}, count {count})");
            return None;
        }

        // Device selection: honor `--gpu N` when given and in range; otherwise
        // auto-select the device with the highest compute capability.
        let best_device: i32 = match requested {
            Some(n) if (0..count).contains(&n) => n,
            _ => {
                if let Some(n) = requested {
                    eprintln!(
                        "CUDA: --gpu {n} out of range (found {count} device(s)); auto-selecting"
                    );
                }
                let mut best_device: i32 = 0;
                let mut best_score: i32 = 0;
                for dev in 0..count {
                    let mut major: i32 = 0;
                    let mut minor: i32 = 0;
                    unsafe {
                        cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_COMPUTE_MAJOR, dev);
                        cudaDeviceGetAttribute(&mut minor, CUDA_DEV_ATTR_COMPUTE_MINOR, dev);
                    }
                    let score = major * 100 + minor;
                    if score > best_score {
                        best_score = score;
                        best_device = dev;
                    }
                }
                best_device
            }
        };

        let err = unsafe { cudaSetDevice(best_device) };
        if err != 0 {
            eprintln!("CUDA: failed to set device {}", best_device);
            return None;
        }

        let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamCreate(&mut stream) };
        if err != 0 || stream.is_null() {
            eprintln!("CUDA: failed to create stream");
            return None;
        }

        // Query device properties
        fn get_attr(attr: i32, dev: i32) -> i32 {
            let mut v: i32 = 0;
            unsafe {
                cudaDeviceGetAttribute(&mut v, attr, dev);
            }
            v
        }
        let major = get_attr(CUDA_DEV_ATTR_COMPUTE_MAJOR, best_device);
        let minor = get_attr(CUDA_DEV_ATTR_COMPUTE_MINOR, best_device);
        let sm_count = get_attr(CUDA_DEV_ATTR_MULTIPROC_COUNT, best_device);
        let mut free_mem: usize = 0;
        let mut total_mem: usize = 0;
        // The banner's total is a convenience, but the return code is checked all the
        // same (issue #122): a failed query here must not print a bare "0 MB" device.
        let mem_rc = unsafe { cudaMemGetInfo(&mut free_mem, &mut total_mem) };
        // Read device name (first 256 bytes of the oversized buffer)
        let mut name_buf = CudaDevicePropBuf([0u8; 4096]);
        unsafe {
            cudaGetDeviceProperties(&mut name_buf, best_device);
        }
        let name = name_buf.0[..256]
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8 as char)
            .collect::<String>();
        eprintln!(
            "CUDA: using {} (SM {}.{}, {} MB, {} SMs)",
            name,
            major,
            minor,
            total_mem / 1048576,
            sm_count
        );
        if mem_rc != 0 {
            eprintln!(
                "CUDA: device memory query at init failed with {} (code {}); the banner's \
                 size is not a measurement",
                cuda_error_name(mem_rc),
                mem_rc
            );
        }
        // T1: resolve the device tier (plan §5.3). MINFER_DEVICE_TIER=<key>
        // forces a row by llama.cpp-style key (1210/1200/890/870/860/750) —
        // the forced-tier soak runs the whole suite under a foreign tier to
        // prove gates only ever choose among correct kernels.
        let cc_val = major * 100 + minor;
        let tier_mmq = match std::env::var("MINFER_DEVICE_TIER")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            Some(key) => {
                let s = device_tier::select_forced(key);
                eprintln!(
                    "CUDA: device tier FORCED {} ({:?}, mmq {}) — key {}",
                    s.tier.name, s.tier.provenance, s.mmq_available, key
                );
                s.mmq_available
            }
            None => {
                let s = device_tier::select(cc_val);
                eprintln!(
                    "CUDA: device tier {} ({:?}, mmq {})",
                    s.tier.name, s.tier.provenance, s.mmq_available
                );
                s.mmq_available
            }
        };
        // T2 (plan §6.2): BT dynamic-smem feasibility — the tile config's
        // demand (single-source formula in cuda_kernels.cu) must fit the
        // device's opt-in limit. Degrading here routes prefill to the f16
        // GEMM path instead of failing launches on 100 KB-class devices.
        // GB10 passes (identical to the previous unconditional behavior).
        let mut tier_mmq = tier_mmq;
        if tier_mmq {
            let smem_need = unsafe { cuda_mmq_smem_bytes() };
            let smem_have = unsafe { cuda_shared_per_block_optin() };
            if smem_need > smem_have {
                eprintln!(
                    "CUDA: BT tile smem {smem_need} B > device optin {smem_have} B — MMQ prefill disabled, f16 GEMM path serves"
                );
                tier_mmq = false;
            }
        }

        // ── Issue #223: eager prefill-GEMM dynamic-smem pre-warm ──────────────
        //
        // #188 deleted the #145 sweep's call from exactly this point and said so
        // nowhere; #218 removed the orphan and made the invariant gated but still
        // only *emergent* (a tested property, not an enforced one). This restores
        // the runtime guarantee at the same site.
        //
        // **Placement, by construction.** `try_new` runs once per process under
        // `CUDA.get_or_init`, before the state is published, before any
        // `CudaBackend` exists, and therefore before any per-instance stream — the
        // only place `graph_begin_capture` can open a window — can exist. So "the
        // attribute is set outside any capture window" is true by construction
        // here, not inferred from the 3-run warmup or the thread-local capture
        // mode; those two remain as defence in depth, and the lazy per-launch
        // opt-in (`gemm_smem_optin`) stays too, so a process that sets
        // `MINFER_NO_GEMM_PREWARM=1` (the documented A/B control) behaves exactly
        // as it did after #218.
        //
        // **One mechanism, not two.** Every entry drives the **production**
        // `gemm_prefill_smem_prewarm_one` → `gemm_smem_optin<TM,KS,AF32>`, i.e.
        // the same per-instantiation cache the launcher reads. A cache-keying
        // regression (the #218 `template <typename K>` bug) therefore cannot be
        // masked by the pre-warm: it makes the pre-warm itself leave
        // instantiations un-opted-in, which the #223 gate and the #218 coverage
        // gate both read back from the device.
        //
        // **No banner.** A fully admitted pre-warm prints nothing; each failure
        // or deliberate skip is named per instantiation (the instantiation, the
        // requested bytes, the queried device limit and `cudaGetErrorName`), and
        // the removed `checked`/`skipped` counters do not come back.
        if !gemm_prewarm_disabled() {
            const GEMM_PREWARM_SET: [(i32, i32, i32); 12] = [
                (64, 32, 0),
                (64, 32, 1),
                (64, 64, 0),
                (64, 64, 1),
                (128, 32, 0),
                (128, 32, 1),
                (128, 64, 0),
                (128, 64, 1),
                (256, 32, 0),
                (256, 32, 1),
                (256, 64, 0),
                (256, 64, 1),
            ];
            let t0 = std::time::Instant::now();
            let mut named = 0usize; // failures + deliberate skips (diagnostic only)
            for &(tm, ks, af32) in GEMM_PREWARM_SET.iter() {
                match unsafe { gemm_prefill_smem_prewarm_one(tm, ks, af32) } {
                    1 => {}  // admitted (set now, or already cached)
                    -1 => {} // not compiled into this fatbin — nothing to do
                    // -2 = over the device limit, deliberately not called;
                    // 0 = the attribute call failed. Both were named by
                    // `minfer_smem_optin` at the `prewarm:gemm_f16` site.
                    -2 | 0 => named += 1,
                    other => {
                        eprintln!(
                            "CUDA: prefill-GEMM smem pre-warm for gemm_f16_nt_kernel_t<{tm},{ks},{}> \
                             returned an unexpected outcome {other}",
                            af32 != 0
                        );
                        named += 1;
                    }
                }
            }
            if crate::optiming::flag_from_env(std::env::var_os("MINFER_OP_TIMING").as_ref()) {
                eprintln!(
                    "CUDA: prefill-GEMM smem pre-warm ({} instantiation(s), {named} refused/skipped \
                     and named above) took {} µs",
                    GEMM_PREWARM_SET.len(),
                    t0.elapsed().as_micros()
                );
            }
        }

        // Issue #188: publish the context stream so the per-stream scratch
        // maps can key unbound callers (legacy layer path, direct tests).
        DEFAULT_STREAM.store(stream as usize, Ordering::Relaxed);
        Some(CudaState {
            stream: Mutex::new(CudaPtr(stream)),
            staging: Mutex::new(HashMap::new()),
            readback: Mutex::new(None),
            weights: Mutex::new(HashMap::new()),
            w16_cache: Mutex::new(HashMap::new()),
            w16_enabled: std::sync::atomic::AtomicBool::new(false),
            cc: std::sync::atomic::AtomicI32::new(major * 100 + minor),
            tier_mmq,
            sm_count,
            nb_bt_only: std::sync::atomic::AtomicBool::new(true),
            padded_weights: Mutex::new(HashMap::new()),
            q80_p32: Mutex::new(HashMap::new()),
            q6k_exp: Mutex::new(HashMap::new()),
            q6k_exp_warned: std::sync::atomic::AtomicBool::new(false),
            q6k_dpl: Mutex::new(HashMap::new()),
            q6k_dsc: Mutex::new(HashMap::new()),
            q6k_dsc_warned: std::sync::atomic::AtomicBool::new(false),
            q4k_dsc: Mutex::new(HashMap::new()),
            q4k_dsc_warned: std::sync::atomic::AtomicBool::new(false),
            max_nchunk: std::sync::atomic::AtomicUsize::new(0),
            buf_q8_prefill: StreamScratch::new(),
            buf_qa8_t: StreamScratch::new(),
            buf_sda_t: StreamScratch::new(),
            buf_mmq_ksplit: StreamScratch::new(),
            mmq_cache: Mutex::new(HashMap::new()),
            buf_attn_partial: StreamScratch::new(),
            buf_q8_decode: StreamScratch::new(),
            buf_f16_w: StreamScratch::new(),
            buf_f16_x: StreamScratch::new(),
        })
    }

    pub fn get() -> Option<&'static Self> {
        CUDA.get().and_then(|s| s.as_ref())
    }

    /// Default: auto-select the device (highest compute capability).
    pub fn init() {
        Self::init_with_gpu(None);
    }

    /// Auto-select, or honor an explicit `--gpu N` device index.
    ///
    /// `requested` is the CUDA device index from `--gpu`; `None` = auto-select.
    /// An out-of-range index warns and falls back to auto-select.
    pub fn init_with_gpu(requested: Option<i32>) {
        CUDA.get_or_init(|| {
            let s = Self::try_new(requested);
            if s.is_some() {
                eprintln!("CUDA: GPU acceleration enabled");
            } else {
                eprintln!("CUDA: not available, using CPU fallback");
            }
            s
        });
    }

    /// Bytes of device-resident weights this state holds (E4: the feasibility gate
    /// charges the budget for them, so "weights + activations" is one comparison).
    /// Sums the registry; the padded-plane companions are accounted by their own
    /// registers and are deliberately not double-counted here.
    ///
    /// A poisoned lock is recovered rather than answered with `0`: this registry is
    /// append-only, so the map behind a poison is still valid, and reporting 0 would
    /// silently *under*-charge the budget by every resident weight (issue #122's
    /// fail-open twin).
    pub fn weights_bytes(&self) -> usize {
        crate::graph::alloc::weights_from_lock(self.weights.lock(), "CUDA weight registry", |w| {
            w.values().map(|(_, size)| *size).sum()
        })
    }

    /// Device memory free/total in bytes, queried now (E4's default budget uses `free`).
    ///
    /// The `cudaMemGetInfo` return code is **not** discarded (issue #122). A pre-#122
    /// failure left `free` at 0 and was indistinguishable from "the device is full";
    /// the E4 feasibility gate then refused every later device allocation with
    /// "exceeds the 0 byte budget (0 MiB)" while the real cause — a sticky CUDA error
    /// such as `cudaErrorIllegalAddress` (700) — was thrown away.
    pub fn device_memory(&self) -> crate::graph::allocplan::DeviceMemory {
        let (mut free, mut total) = (0usize, 0usize);
        let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if rc != 0 {
            return crate::graph::allocplan::DeviceMemory::QueryFailed {
                code: rc,
                name: cuda_error_name(rc).to_string(),
            };
        }
        crate::graph::allocplan::DeviceMemory::Reported { free, total }
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        {
            let w = self.weights.lock().unwrap();
            if let Some((_, size)) = w.get(name) {
                if *size == data.len() {
                    // Device weights are immutable: same name + size ⇒ the
                    // same GGUF tensor (single-model-per-process today, and
                    // unit tests reload the same file). Reuse the existing
                    // device copy instead of leaking one buffer per load.
                    return;
                }
                // Different size (a different architecture registered the
                // same tensor name): replace the entry. The stale buffer is
                // deliberately NOT freed — a live captured graph may still
                // reference it; the leak is bounded by the number of
                // distinct (arch, tensor) shapes ever loaded.
            }
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
        if err != 0 || ptr.is_null() {
            eprintln!(
                "CUDA: failed to allocate {} bytes for '{}'",
                data.len(),
                name
            );
            return;
        }
        let err = unsafe {
            // Issue #188: stream-ordered, not the legacy-null-stream blocking
            // `cudaMemcpy`. The blocking form is not a stream operation at all:
            // while another thread holds a capture window open on a different
            // stream, it participates in the legacy default stream's implicit
            // global synchronization — under `cudaStreamCaptureModeGlobal` that
            // is exactly the call that invalidated the capture (901) or faulted
            // in `cuMemcpyHtoD_v2`. Queuing the copy on the context's own stream
            // and waiting on that stream keeps the registration a bounded,
            // stream-scoped operation.
            cudaMemcpyAsync(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.context_stream(),
            )
        };
        if err == 0 {
            let serr = unsafe { cudaStreamSynchronize(self.context_stream()) };
            if serr != 0 {
                eprintln!(
                    "CUDA: weight-registration stream sync failed for '{}': {} ({serr})",
                    name,
                    cuda_error_name(serr)
                );
                unsafe {
                    cudaFree(ptr);
                }
                return;
            }
        }
        if err != 0 {
            eprintln!("CUDA: failed to copy '{}' to device", name);
            unsafe {
                cudaFree(ptr);
            }
            return;
        }
        // a plain (unpadded) registration must clear any stale padded flag
        // for the same name: a second model reusing the tensor name with a
        // non-Q6_K type would otherwise dispatch the padded-224 kernel on a
        // raw-210 buffer (Phase 8 review finding)
        self.padded_weights.lock().unwrap().remove(name);
        self.weights
            .lock()
            .unwrap()
            .insert(name.to_string(), (CudaPtr(ptr), data.len()));
    }

    /// Issue #188 probe, **test-only**: the pre-#188 registration H2D path, a
    /// blocking `cudaMemcpy` (which the driver issues on the legacy default
    /// stream, not on any explicit stream). Kept so the acceptance probe can
    /// measure the capture mode against the *historical* setup — a shared
    /// capture stream plus this copy — instead of against a setup the fix
    /// already changed, which would make the mode look irrelevant.
    #[cfg(test)]
    pub fn register_weight_blocking_legacy(&self, name: &str, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
        if err != 0 || ptr.is_null() {
            panic!("probe: cudaMalloc({}) failed ({err})", data.len());
        }
        let err = unsafe {
            cudaMemcpy(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 {
            unsafe { cudaFree(ptr) };
            panic!("probe: blocking cudaMemcpy failed ({err})");
        }
        self.weights
            .lock()
            .unwrap()
            .insert(name.to_string(), (CudaPtr(ptr), data.len()));
    }

    /// 7e②: register a Q6_K tensor in the PADDED device layout (each
    /// 210-byte block in a 224-byte slot) so the matmul kernel can use
    /// 16-byte-aligned uint4 weight loads. `od`/`id` are the matmul output/
    /// input dims (GGUF shape [in, out] → id = shape[0], od = shape[1]).
    pub fn register_weight_q6k_padded(&self, name: &str, data: &[u8], od: usize, id: usize) {
        // r59 rider: track the max nchunk (= id/32) for the pre-warm scratch
        // sizing (see prewarm_prefill).
        self.max_nchunk
            .fetch_max(id / 32, std::sync::atomic::Ordering::Relaxed);
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id.div_ceil(256);
        let row_len = nbe * Q6KB;
        if od == 0 || id == 0 || data.len() < od * row_len {
            eprintln!(
                "CUDA: q6_k padded registration skipped for '{}' ({} bytes, od={od} id={id})",
                name,
                data.len()
            );
            return;
        }
        let mut padded = vec![0u8; od * nbe * Q6KPB];
        for r in 0..od {
            for ib in 0..nbe {
                let src = r * row_len + ib * Q6KB;
                let dst = r * nbe * Q6KPB + ib * Q6KPB;
                padded[dst..dst + Q6KB].copy_from_slice(&data[src..src + Q6KB]);
            }
        }
        self.register_weight(name, &padded);
        self.padded_weights
            .lock()
            .unwrap()
            .insert(name.to_string(), data.len());
        // D4-4 L1: dense split-plane (dpl) decode sibling plane — per row
        // [ql nbe*128][qh nbe*64][sc nbe*16][d nbe*2] at a 16B-aligned row
        // stride, 210B of content per 256-elem block (no 224B pad sectors).
        // The decode MMVQ reads it when present; per-unit values and the
        // accumulation order are unchanged, so outputs stay bitwise-identical
        // (probe /tmp/d4/probe_l1_dpl.cu). Requires id % 256 == 0 (exact
        // nbe, like the W_exp plane); MINFER_Q6K_DPL=0 skips the build and
        // decode keeps the padded kernels.
        if Self::mmq_gate_on("MINFER_Q6K_DPL") && id % 256 == 0 {
            let nbe = id / 256;
            let dpl_row = (nbe * 210 + 15) & !15usize;
            let raw_row = nbe * 210;
            let mut dpl = vec![0u8; od * dpl_row];
            for r in 0..od {
                let src = &data[r * raw_row..(r + 1) * raw_row];
                let dst = &mut dpl[r * dpl_row..(r + 1) * dpl_row];
                let (ql, rest) = dst.split_at_mut(nbe * 128);
                let (qh, rest) = rest.split_at_mut(nbe * 64);
                let (sc, dd) = rest.split_at_mut(nbe * 16);
                for ib in 0..nbe {
                    let blk = &src[ib * 210..(ib + 1) * 210];
                    ql[ib * 128..(ib + 1) * 128].copy_from_slice(&blk[..128]);
                    qh[ib * 64..(ib + 1) * 64].copy_from_slice(&blk[128..192]);
                    sc[ib * 16..(ib + 1) * 16].copy_from_slice(&blk[192..208]);
                    dd[ib * 2..(ib + 1) * 2].copy_from_slice(&blk[208..210]);
                }
            }
            let dpl_name = format!("{name}__dpl");
            self.register_weight(&dpl_name, &dpl);
            if let (Some(wp), Some(ep)) =
                (self.get_weight_ptr(name), self.get_weight_ptr(&dpl_name))
            {
                if !wp.is_null() && !ep.is_null() {
                    self.q6k_dpl
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(ep));
                }
            }
        }
        // r53: pre-expand B into the dense centered-int8 plane (P6 r44) so the
        // NB-BT q6_K kernel's B staging is a pure cp.async copy. Ships with the
        // MINFER_MMQ_Q6K_NB gate (the NB-BT kernel is its only consumer; the
        // A/B baseline is the unchanged env set) and requires id % 256 == 0
        // (the kernel's own launch gate, which also keeps the dense index
        // 16B-aligned). Dense bytes = od * id — ~2.4 GiB total on 7B q4_k_m
        // (ffn_down 67.9 MB x 28 + output 545 MB + attn_v 1.8 MB x 28).
        // r54: MINFER_MMQ_Q6K_EXP decouples the plane from the kernel gate —
        // unset/"1" keeps the r53 default (build it), explicit "0" skips the
        // build entirely (registration early-returns; device memory stays at
        // the pre-r53 level) so dispatch map-misses into the EXP=false r41
        // in-kernel expand. ANDed with Q6K_NB: EXP only matters when the NB
        // kernel is live (r60: Q6K_NB is default-on — "0" opts out of the
        // kernel AND both planes).
        if Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")
            && std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() != Ok("0")
            && id % 256 == 0
        {
            self.register_weight_q6k_exp(name, &padded, od, id);
            // r56 (Session E item 2b): the dsc f32 plane rides the same gate
            // (+ od % 2 == 0: the kernel stages row PAIRS per 16-B cp.async
            // chunk and zero-fills whole pairs, so an odd od row would lose
            // its scale — such tensors keep the scalar path via map miss).
            if od % 2 == 0 {
                self.register_weight_q6k_dsc(name, &padded, od, id);
            }
        }
    }

    /// r53: build + upload the dense pre-expanded B plane for one padded q6_K
    /// tensor and map it from the padded weight's device pointer. Called from
    /// [`Self::register_weight_q6k_padded`] under the `MINFER_MMQ_Q6K_NB`
    /// (r60: default-on) + `MINFER_MMQ_Q6K_EXP != "0"` (r54) +
    /// `id % 256 == 0` gate; also `pub`
    /// for the gate-on byte-exactness test. An alloc/upload failure leaves the
    /// map empty: the kernel falls back to the r41 in-kernel expand with a
    /// once-per-process loud eprintln.
    pub fn register_weight_q6k_exp(&self, name: &str, padded: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name: a same-name different-shape
        // re-registration can never collide with (and silently reuse) a stale
        // plane of the same byte size but a different od/id layout.
        let exp_name = format!("{name}__exp{od}x{id}");
        // T2: budget-gate BEFORE the host expansion (review NIT #6) — the
        // dense plane is od*id bytes; a tripped device must not pay the
        // full host build only to discard it.
        if !self.plane_budget_ok(od * id) {
            return;
        }
        let exp = Self::expand_q6k_dense(padded, od, id);
        debug_assert_eq!(exp.len(), od * id);
        self.register_weight(&exp_name, &exp);
        // the MAP is keyed by the PADDED weight's device pointer (what
        // prefill_mmq holds); the value is the W_exp plane's pointer
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(ep) = self.get_weight_ptr(&exp_name) {
                if !wp.is_null() && !ep.is_null() {
                    self.q6k_exp
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(ep));
                    return;
                }
            }
        }
        if !self
            .q6k_exp_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q6_K W_exp pre-expand unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt_q6k falls back to the \
                 r41 in-kernel expand"
            );
        }
    }

    /// r56 (Session E item 2b): build + upload the precomputed dsc f32-pair
    /// plane for one padded q6_K tensor and map it from the padded weight's
    /// device pointer. Called from [`Self::register_weight_q6k_padded`] under
    /// the same gate as `register_weight_q6k_exp` (+ `od % 2 == 0`). An
    /// alloc/upload failure leaves the map empty: the kernel falls back to the
    /// r41 scalar dsc path with a once-per-process loud eprintln.
    pub fn register_weight_q6k_dsc(&self, name: &str, padded: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name (same rationale as the W_exp name).
        let dsc_name = format!("{name}__dsc{od}x{id}");
        // T2: budget-gate BEFORE the host expansion (review NIT #6) — the
        // dsc plane is (id/32)*od*8 bytes.
        if !self.plane_budget_ok((id / 32) * od * 8) {
            return;
        }
        let dsc = Self::expand_q6k_dsc(padded, od, id);
        debug_assert_eq!(dsc.len(), (id / 32) * od * 8);
        self.register_weight(&dsc_name, &dsc);
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(dp) = self.get_weight_ptr(&dsc_name) {
                if !wp.is_null() && !dp.is_null() {
                    self.q6k_dsc
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(dp));
                    return;
                }
            }
        }
        if !self
            .q6k_dsc_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q6_K W_dsc plane unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt_q6k keeps the r41 \
                 scalar dsc path"
            );
        }
    }

    /// r56 (Session E item 2b): precomputed dsc pairs of one padded q6_K
    /// tensor. Output: `nchunk * od * 8` bytes, `out[(c*od + j)*8..+8]` =
    /// float2(d*sc[2(c&7)], d*sc[2(c&7)+1]) — chunk-major so the kernel's
    /// per-kt staging (rows j0..j0+MMQ_NBJ of chunk c0+kd contiguous) is a
    /// pure 16-B cp.async stream. Bit-identical to the in-kernel r41 scalar
    /// computation: exact f16->f32 (half::f16, = __half2float), exact
    /// i8->f32, one IEEE f32 multiply, no FMA contraction on either side.
    pub fn expand_q6k_dsc(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id / 256;
        let row_len = nbe * Q6KPB;
        let mut out = vec![0u8; (id / 32) * od * 8];
        for j in 0..od {
            let prow = &padded[j * row_len..(j + 1) * row_len];
            for sb in 0..nbe {
                let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
                let d_bits = u16::from_le_bytes([blk[208], blk[209]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                for cc in 0..8usize {
                    let s0 = 2 * cc;
                    let sc0 = blk[192 + s0] as i8 as f32;
                    let sc1 = blk[192 + s0 + 1] as i8 as f32;
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    out[idx..idx + 4].copy_from_slice(&(d * sc0).to_bits().to_le_bytes());
                    out[idx + 4..idx + 8].copy_from_slice(&(d * sc1).to_bits().to_le_bytes());
                }
            }
        }
        out
    }

    /// r59 (Session F item 1), #165: build + upload the precomputed dsc f32-pair
    /// plane for one RAW q4_K tensor and map it from the raw weight's device
    /// pointer. Called from the qwen2 loader under the NB-BT gate set
    /// (MINFER_MMQ_RAW_NB=1 + MINFER_MMQ_A_TRANSPOSE=1 + MINFER_MMQ_Q4K_DSC
    /// != "0") + [`crate::q4k_dsc::q4k_dsc_plane_admitted`] (type q4_K + `id % 256 == 0`) +
    /// `od % 2 == 0`. An alloc/upload failure leaves the map empty: the kernel
    /// falls back to the in-kernel scalar decode (DSC=false) with a
    /// once-per-process loud eprintln. A payload that is not exactly
    /// [`crate::q4k_dsc::q4k_dsc_payload_bytes`] is refused **before** the budget query and
    /// before the host expansion: its bytes are another type's (the #165
    /// misread) or too few for the row arithmetic (the #165 latent OOB read).
    pub fn register_weight_q4k_dsc(&self, name: &str, raw: &[u8], od: usize, id: usize) {
        // #165: the payload contract first. It costs nothing and is the only check that
        // can refuse a smaller-ratio payload before `expand_q4k_dsc` would index past
        // `raw`; `q4k_dsc_plane_admitted` in the loader is the same rule plus the type.
        if !q4k_dsc_payload_ok(raw.len(), od, id) {
            return;
        }
        // geometry-encoded sibling name (same rationale as the W_exp name).
        let dsc_name = format!("{name}__q4dsc{od}x{id}");
        // T2: budget-gate BEFORE the host expansion (review NIT #6) — the
        // dsc plane is (id/32)*od*8 bytes.
        if !self.plane_budget_ok((id / 32) * od * 8) {
            return;
        }
        let Some(dsc) = Self::expand_q4k_dsc(raw, od, id) else {
            return;
        };
        debug_assert_eq!(dsc.len(), (id / 32) * od * 8);
        self.register_weight(&dsc_name, &dsc);
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(dp) = self.get_weight_ptr(&dsc_name) {
                if !wp.is_null() && !dp.is_null() {
                    self.q4k_dsc
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(dp));
                    return;
                }
            }
        }
        if !self
            .q4k_dsc_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q4_K W_dsc plane unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt keeps the in-kernel \
                 scalar dsc decode"
            );
        }
    }

    /// r59 (Session F item 1): precomputed dsc pairs of one RAW q4_K tensor.
    /// Output: `(id / 32) * od * 8` bytes, `out[(c*od + j)*8..+8]` =
    /// float2(d*sc[c&7], -(dmin*m[c&7])) — chunk-major so the kernel's
    /// per-kt staging (rows j0..j0+MMQ_NBJ of chunk c0+kd contiguous) is a
    /// pure 16-B cp.async stream. Bit-identical to the in-kernel scalar
    /// computation: exact f16->f32 (half::f16 = __half2float), exact
    /// u8->f32, ONE IEEE f32 multiply, exact negation, no FMA contraction
    /// on either side.
    ///
    /// #165: **`None` when `raw` is not exactly [`crate::q4k_dsc::q4k_dsc_payload_bytes`] for the
    /// geometry** — `raw`'s rows are then not 144-byte q4_K super-block rows, and the
    /// indexing below (`raw[j * row_len..(j + 1) * row_len]`) would either misread
    /// another type's bytes or run past the slice. The caller
    /// ([`Self::register_weight_q4k_dsc`]) registers nothing in that case.
    pub fn expand_q4k_dsc(raw: &[u8], od: usize, id: usize) -> Option<Vec<u8>> {
        if !q4k_dsc_payload_ok(raw.len(), od, id) {
            return None;
        }
        const Q4KB: usize = 144;
        let nsb = id / 256;
        let nchunk = id / 32;
        let row_len = nsb * Q4KB;
        let mut out = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            let prow = &raw[j * row_len..(j + 1) * row_len];
            for sb in 0..nsb {
                let blk = &prow[sb * Q4KB..sb * Q4KB + Q4KB];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
                let sc = &blk[4..16]; // 12 packed 6-bit scales+mins
                for cc in 0..8usize {
                    // host mirror of the device get_scale_min_k4 (cuda_kernels.cu)
                    let (s, m) = if cc < 4 {
                        (sc[cc] & 63, sc[cc + 4] & 63)
                    } else {
                        (
                            (sc[cc + 4] & 0xF) | ((sc[cc - 4] >> 6) << 4),
                            (sc[cc + 4] >> 4) | ((sc[cc] >> 6) << 4),
                        )
                    };
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    out[idx..idx + 4].copy_from_slice(&(d * (s as f32)).to_bits().to_le_bytes());
                    out[idx + 4..idx + 8]
                        .copy_from_slice(&(-(dmin * (m as f32))).to_bits().to_le_bytes());
                }
            }
        }
        Some(out)
    }

    /// r59 (Session F riders, r57 items 4+5): move the one-time first-launch
    /// costs out of the measured prefill window. Called once at the end of
    /// model weight registration.
    /// (1) kernel module pre-load — cudaFuncGetAttributes over the
    ///     MMQ/FA/fused launch set forces the fatbin to load now instead of
    ///     at the first dispatch (r58 CUPTI: ~3 ms host stalls bracketing
    ///     the first mode-2 swiglu / first bt matmul);
    /// (2) pinned D2H readback pre-grow — the grow-on-demand 4 MB
    ///     cudaHostAlloc was the "0.78 ms tail malloc" at the n_out=1
    ///     logits readback;
    /// (3) MmqCache scratch pre-grow — buf_q8_prefill / buf_qa8_t /
    ///     buf_sda_t sized for a nominal 4096-token prefill (the default
    ///     n_ctx) at the max registered nchunk, so the first prefill's
    ///     get_or_grow hits instead of cudaMalloc-ing ~150 MB mid-window
    ///     (a larger prompt grows in-window exactly as before). MMQ-gated:
    ///     the planes are dead weight when the MMQ path is off.
    pub fn prewarm_prefill(&self) {
        unsafe {
            minfer_prewarm_kernels();
        }
        // (2) pinned readback pre-grow (same 4 MB floor as
        // copy_from_device_pinned; failure is non-fatal — the first readback
        // retries the alloc there).
        {
            let mut guard = self.readback.lock().unwrap();
            if guard.is_none() {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = unsafe { cudaHostAlloc(&mut p, 4 * 1024 * 1024, 0) };
                if err == 0 && !p.is_null() {
                    *guard = Some(PinnedBuf {
                        ptr: p as *mut u8,
                        bytes: 4 * 1024 * 1024,
                    });
                }
            }
        }
        // (3) MmqCache scratch pre-grow
        if Self::mmq_gate_on("MINFER_MMQ") {
            let max_nchunk = self.max_nchunk.load(std::sync::atomic::Ordering::Relaxed);
            if max_nchunk > 0 {
                const NT_PREWARM: usize = 4096; // default n_ctx
                let ntb = NT_PREWARM.div_ceil(64);
                Self::get_or_grow(&self.buf_q8_prefill, NT_PREWARM * max_nchunk * 40);
                Self::get_or_grow(&self.buf_qa8_t, ntb * max_nchunk * 2048);
                Self::get_or_grow(&self.buf_sda_t, ntb * max_nchunk * 256);
            }
        }
    }

    /// r53: dense centered-int8 pre-expansion of one padded q6_K tensor — the
    /// host mirror of the device `expand_q6_elem` (P6 r44 / MMQ-analysis
    /// §11.24). Output: `od * id` bytes, `out[j * id + sb * 256 + e]` =
    /// super-block element e of row j — the exact tile the kernel's staging
    /// used to recomb. Requires `id % 256 == 0` (the NB-BT launch gate).
    /// Two output elements per (ql, qh) byte pair: e = it*128+r and
    /// e = it*128+r+64 share ql[it*64+r] (nibble shifts 0/4) and
    /// qh[it*32 + (r&31)] (2-bit-field shifts 2*(r>>5) / 2*((r>>5)+2)).
    pub fn expand_q6k_dense(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id / 256;
        let row_len = nbe * Q6KPB;
        let mut out = vec![0u8; od * id];
        for j in 0..od {
            let prow = &padded[j * row_len..(j + 1) * row_len];
            let orow = &mut out[j * id..(j + 1) * id];
            for sb in 0..nbe {
                let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
                let (ql, qh) = blk.split_at(128);
                let obase = &mut orow[sb * 256..sb * 256 + 256];
                for it in 0..2usize {
                    for r in 0..64usize {
                        let qlb = ql[it * 64 + r];
                        let qhb = qh[it * 32 + (r & 31)];
                        let s0 = (r >> 5) * 2;
                        let e0 = it * 128 + r;
                        obase[e0] = ((qlb & 0xF) | (((qhb >> s0) & 3) << 4)).wrapping_sub(32);
                        obase[e0 + 64] =
                            (((qlb >> 4) & 0xF) | (((qhb >> (s0 + 4)) & 3) << 4)).wrapping_sub(32);
                    }
                }
            }
        }
        out
    }

    /// Whether `name` was registered in the padded Q6_K layout.
    pub fn is_weight_padded(&self, name: &str) -> bool {
        self.padded_weights.lock().unwrap().contains_key(name)
    }

    /// #165: the q4_K `W_dsc` planes currently in the registry (`*__q4dsc*`) as
    /// `(name, bytes)`. The plane's only consumer is the NB-BT q4_K kernel, so this is
    /// the exact set of device buffers a load that is *not* q4_K must leave empty — the
    /// registry query the #165 gate and its before/after accounting read by name.
    /// Test-only (#238): driven by `graph::cuda_backend::tests::cuda_q4dsc_plane_is_q4k_only and tooling::tests::f167_qwen3_q4k_registers_the_dsc_plane_exactly`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn q4dsc_planes(&self) -> Vec<(String, usize)> {
        self.weights
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _)| n.contains("__q4dsc"))
            .map(|(n, (_, size))| (n.clone(), *size))
            .collect()
    }

    pub fn get_weight_ptr(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        self.weights.lock().unwrap().get(name).map(|(cp, _)| cp.0)
    }

    /// The registered byte length of the device weight `name` (#169).
    ///
    /// A Q6_K entry registered through `register_weight_q6k_padded` lives on the
    /// device with a larger stride, so it reports its **original raw** length —
    /// the same convention `has_weight_of_size` uses. `None` when the name is
    /// not registered at all. The norm path reads this to refuse a weight that
    /// is not the f32 it will index as (the f16-norm hazard of #169).
    pub fn weight_size(&self, name: &str) -> Option<usize> {
        if let Some(&raw) = self.padded_weights.lock().unwrap().get(name) {
            return Some(raw);
        }
        self.weights
            .lock()
            .unwrap()
            .get(name)
            .map(|(_, size)| *size)
    }

    /// #167: whether the NB-BT q4_K kernel would find a `W_dsc` plane for the raw weight
    /// `name`. The kernel keys its map on the **raw weight's device pointer** (`q4k_dsc`,
    /// read at `mmq_raw_nb_bt`'s launch site: a null `w_dsc` selects the in-kernel scalar
    /// decode), so this performs exactly that lookup rather than looking the sibling name
    /// up in the registry — which is the part a name-only assertion cannot see. `None`
    /// when the weight is not registered at all.
    /// Test-only (#238): driven by `tooling::tests::f167_qwen3_q4k_registers_the_dsc_plane_exactly`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn q4dsc_plane_for(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        let wp = self.get_weight_ptr(name)?;
        if wp.is_null() {
            return None;
        }
        self.q4k_dsc
            .lock()
            .unwrap()
            .get(&(wp as usize))
            .map(|cp| cp.0)
    }

    /// Process-wide model-load serialization: loaders hold this while
    /// registering weights, so two models with same-named tensors (qwen2 0.5B
    /// vs qwen3 0.6B in parallel tests) cannot interleave their registrations.
    /// The graph-path tests that span multiple forwards hold it for their body
    /// to keep the weight registry stable underneath them. REENTRANT per
    /// thread: `load_model` takes it inside callers that already hold it.
    pub fn model_load_guard() -> ModelLoadGuard {
        ModelLoadGuard::acquire()
    }

    /// Size-aware registry check for the graph-path gate: a same-name entry
    /// with a DIFFERENT byte size belongs to another architecture's model and
    /// must read as "not registered" so that model cleanly falls back to CPU.
    pub fn has_weight_of_size(&self, name: &str, bytes: usize) -> bool {
        // Padded Q6_K entries live on the device with a larger (224-byte
        // stride) footprint; match them by their ORIGINAL raw length so the
        // weights gate sees them as registered.
        if let Some(&raw) = self.padded_weights.lock().unwrap().get(name) {
            return raw == bytes;
        }
        self.weights
            .lock()
            .unwrap()
            .get(name)
            .is_some_and(|(_, size)| *size == bytes)
    }

    /// The device stream this call's work belongs on.
    ///
    /// **Per instance since #188.** A `CudaBackend` binds its own non-blocking
    /// stream for the duration of each device operation (`crate::cuda::bind_stream`),
    /// so this answers with *that* backend's stream; only an unbound caller
    /// (legacy layer path, direct `CudaState` tests, weight registration) sees
    /// the context's own stream. Every launch, copy, event, capture and
    /// synchronize in this module goes through here, which is what makes the
    /// stream follow the backend rather than the process.
    pub fn stream(&self) -> *mut std::ffi::c_void {
        let bound = bound_stream();
        if bound.is_null() {
            self.stream.lock().unwrap().0
        } else {
            bound
        }
    }

    /// The **context's own** stream — never a backend's bound stream. Weight
    /// registration (and any other context-level transfer) uses this so it can
    /// never enqueue into a backend's open capture window.
    fn context_stream(&self) -> *mut std::ffi::c_void {
        self.stream.lock().unwrap().0
    }

    /// Issue #188: create a **non-blocking** stream for a backend instance. The
    /// flag matters: a blocking stream implicitly synchronizes with the legacy
    /// default stream, so a host-side `cudaMemcpy` for one engine would join
    /// every other engine's stream — and inside another thread's capture window
    /// that implicit join is what invalidates it.
    pub fn create_stream(&self) -> *mut std::ffi::c_void {
        let mut s: *mut std::ffi::c_void = std::ptr::null_mut();
        // cudaStreamNonBlocking == 1 (cudaStreamDefault == 0).
        let err = unsafe { cudaStreamCreateWithFlags(&mut s, 1) };
        if err != 0 || s.is_null() {
            eprintln!(
                "CUDA: cudaStreamCreateWithFlags(non-blocking) failed: {} ({err})",
                cuda_error_name(err)
            );
            return std::ptr::null_mut();
        }
        s
    }

    /// Issue #188: release a stream from [`Self::create_stream`] (no-op on null).
    pub fn destroy_stream(&self, stream: *mut std::ffi::c_void) {
        if !stream.is_null() {
            unsafe {
                cudaStreamDestroy(stream);
            }
        }
    }

    /// Issue #188: is `stream` inside a capture window right now? Read only by the
    /// #188 probe; a failed query reads as "not capturing" (the caller's own
    /// per-instance capture bookkeeping is authoritative). The note here used to
    /// claim a registration-path inventory that does not call it — corrected by
    /// #238.
    /// Test-only (#238): driven by `graph::cuda_backend::tests::cuda_capture_abort_on_error`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn stream_is_capturing(stream: *mut std::ffi::c_void) -> bool {
        let mut status: i32 = 0; // cudaStreamCaptureStatusNone == 0
        let err = unsafe { cudaStreamIsCapturing(stream, &mut status) };
        err == 0 && status != 0
    }

    // ─── Persistent buffer management ─────────────────────────

    /// The live persistent-scratch allocator: every `buf_*`/staging slot the
    /// MMQ, attention and f16 paths use is grown here, per stream (#188), so no
    /// `cudaMalloc` lands mid-capture. (This replaces a stale "legacy surface
    /// (7e⑦)" note.)
    fn get_or_grow(slot: &StreamScratch, need: usize) -> *mut std::ffi::c_void {
        // Issue #188: the slot is private to the **current stream**, so two
        // engines' concurrent launches never read each other's staging.
        let key = current_stream_key();
        let mut map = slot.map.lock().unwrap();
        let (ptr, size) = map
            .entry(key)
            .or_insert((CudaPtr(std::ptr::null_mut()), 0usize));
        if ptr.0.is_null() || *size < need {
            if !ptr.0.is_null() {
                unsafe {
                    cudaFree(ptr.0);
                }
            }
            let mut new_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { cudaMalloc(&mut new_ptr, need) };
            if err != 0 || new_ptr.is_null() {
                eprintln!("CUDA: OOM allocating {} bytes (cuda err {err})", need);
                *ptr = CudaPtr(std::ptr::null_mut());
                *size = 0;
                return std::ptr::null_mut();
            }
            *ptr = CudaPtr(new_ptr);
            *size = need;
            new_ptr
        } else {
            ptr.0
        }
    }

    /// Allocate device memory (graph-backend pool helper). Null on failure.
    pub fn cuda_malloc(size: usize) -> *mut std::ffi::c_void {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, size) };
        if err != 0 || ptr.is_null() {
            eprintln!("CUDA: OOM allocating {} bytes", size);
            return std::ptr::null_mut();
        }
        ptr
    }

    /// Free device memory allocated via [`Self::cuda_malloc`] (no-op on null).
    pub fn cuda_free(ptr: *mut std::ffi::c_void) {
        if !ptr.is_null() {
            unsafe {
                cudaFree(ptr);
            }
        }
    }

    // ─── Copy helpers ─────────────────────────────────────────

    pub fn copy_to_device(&self, src: &[u8], dst: *mut std::ffi::c_void) {
        unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
        }
    }

    /// 7e⑥: async H2D input fill through a pinned staging slot. The data
    /// is copied into pinned host memory (cheap, CPU-side), then
    /// `cudaMemcpyAsync` queues the transfer on the stream — the call
    /// returns before the copy lands; same-stream ordering guarantees the
    /// fill completes before the kernels that read the buffer. Falls back
    /// to a synchronous pageable copy for oversized inputs or if pinned
    /// allocation failed.
    pub fn write_input_async(&self, data: &[u8], dst: *mut std::ffi::c_void) {
        const STAGING_SLOTS: usize = 8;
        const STAGING_SLOT_BYTES: usize = 2 * 1024 * 1024;
        // #188: the ring is keyed on the bound stream — its slots are the
        // source of a `cudaMemcpyAsync` on that stream, so two engines must not
        // share one.
        let key = current_stream_key();
        let mut map = self.staging.lock().unwrap();
        if !map.contains_key(&key) {
            let mut ptrs = Vec::new();
            let mut alloc_err = 0i32;
            for _ in 0..STAGING_SLOTS {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                alloc_err = unsafe { cudaHostAlloc(&mut p, STAGING_SLOT_BYTES, 0) };
                if alloc_err != 0 {
                    break;
                }
                ptrs.push(p as *mut u8);
            }
            // a shrunken ring silently degrades to a stream sync every
            // `ptrs.len()` fills — surface it (Phase 8 review)
            if ptrs.len() != STAGING_SLOTS {
                eprintln!(
                    "CUDA: pinned staging ring shrunk to {}/{} slots (cudaHostAlloc err {});                      fills beyond the ring fall back to sync copies",
                    ptrs.len(),
                    STAGING_SLOTS,
                    alloc_err
                );
            }
            if !ptrs.is_empty() {
                map.insert(
                    key,
                    PinnedPool {
                        ptrs,
                        slot_bytes: STAGING_SLOT_BYTES,
                        next: 0,
                    },
                );
            }
        }
        let fits = map.get(&key).is_some_and(|p| data.len() <= p.slot_bytes);
        if !fits {
            drop(map);
            self.copy_to_device(data, dst);
            return;
        }
        // ring wrap: retire all in-flight copies before reusing slot 0. The
        // reset is re-checked under the re-lock so two threads that both
        // observed the full ring cannot both take slot 0 (Phase 8 review).
        if map.get(&key).is_some_and(|p| p.next == p.ptrs.len()) {
            drop(map);
            self.sync();
            map = self.staging.lock().unwrap();
        }
        let slot = {
            let pool = map.get_mut(&key).expect("the ring exists (checked above)");
            if pool.next >= pool.ptrs.len() {
                pool.next = 0;
            }
            let p = pool.ptrs[pool.next];
            pool.next += 1;
            p
        };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), slot, data.len());
            cudaMemcpyAsync(
                dst,
                slot as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream(),
            );
        }
    }

    pub fn copy_from_device(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                src,
                dst.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
        }
    }

    /// R3-A2: D2H read through our own pinned staging buffer. The caller has
    /// already synchronized the stream; the copy is a blocking `cudaMemcpy`
    /// whose DESTINATION is pinned — no driver-internal bounce buffer, no
    /// pageable staging — followed by a plain CPU copy out to the caller's
    /// (pageable) slice. `MINFER_NO_PINNED_READBACK=1` or a cudaHostAlloc
    /// failure falls back to the pageable path (`copy_from_device`).
    pub fn copy_from_device_pinned(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        static FALLBACK_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if std::env::var("MINFER_NO_PINNED_READBACK").as_deref() == Ok("1") {
            self.copy_from_device(src, dst);
            return;
        }
        // headroom so small size changes don't churn the allocation
        let need = dst.len().max(4 * 1024 * 1024);
        let mut guard = self.readback.lock().unwrap();
        if guard.as_ref().map_or(true, |b| b.bytes < need) {
            if let Some(old) = guard.take() {
                drop(old); // cudaFreeHost
            }
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { cudaHostAlloc(&mut p, need, 0) };
            if err != 0 {
                drop(guard);
                if !FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "CUDA: pinned readback alloc failed (err {err}); pageable D2H fallback"
                    );
                }
                self.copy_from_device(src, dst);
                return;
            }
            *guard = Some(PinnedBuf {
                ptr: p as *mut u8,
                bytes: need,
            });
        }
        let buf = guard.as_mut().unwrap();
        unsafe {
            cudaMemcpy(
                buf.ptr as *mut std::ffi::c_void,
                src,
                dst.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            std::ptr::copy_nonoverlapping(buf.ptr, dst.as_mut_ptr(), dst.len());
        }
    }

    pub fn copy_device_to_device(
        &self,
        src: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        size: usize,
    ) {
        unsafe {
            // Stream-ordered (not the legacy-sync cudaMemcpy): capturable
            // inside a CUDA Graph capture window and race-free with replay.
            cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src,
                size,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.stream(),
            );
        }
    }

    // ─── F5 (#58): events + asynchronous host transfers ────────
    //
    // The split boundary is the only place the engine moves a value between
    // backends. Before F5 every such move read the source through
    // `copy_from_device_pinned`, which first **synchronized the whole stream**
    // and then issued a blocking `cudaMemcpy` — so one cross-backend hop cost
    // the host two stalls and forbade any overlap. These are the primitives that
    // replace it: a stream event plus a stream-ordered `cudaMemcpyAsync`, with
    // the single host wait moved to the consumer
    // (`docs/BACKEND-REGISTRY-DESIGN.md` §11).

    /// F5: allocate a pinned host slab (the intermediate of an async D2H copy —
    /// a pageable destination would make the driver bounce through its own
    /// pinned buffer and block, which is exactly what this avoids).
    pub fn host_alloc(&self, bytes: usize) -> Option<*mut u8> {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaHostAlloc(&mut p, bytes, 0) };
        if err != 0 || p.is_null() {
            return None;
        }
        Some(p as *mut u8)
    }

    /// F5: release a slab from [`Self::host_alloc`].
    pub fn host_free(&self, ptr: *mut u8) {
        if !ptr.is_null() {
            unsafe { cudaFreeHost(ptr as *mut std::ffi::c_void) };
        }
    }

    /// F5: create an event and record it on the stream. The returned handle must
    /// be released with [`Self::event_destroy`]; a failure is a loud `Err` naming
    /// the `cudaGetErrorName` (never a silently missing synchronization).
    pub fn record_event(&self) -> Result<*mut std::ffi::c_void, String> {
        let mut ev: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaEventCreate(&mut ev) };
        if err != 0 || ev.is_null() {
            return Err(format!(
                "cudaEventCreate failed: {} ({err})",
                cuda_error_name(err)
            ));
        }
        let err = unsafe { cudaEventRecord(ev, self.stream()) };
        if err != 0 {
            unsafe { cudaEventDestroy(ev) };
            return Err(format!(
                "cudaEventRecord failed: {} ({err})",
                cuda_error_name(err)
            ));
        }
        Ok(ev)
    }

    /// F5: **block the host** until `ev` has completed. This is the one
    /// documented synchronization point of a device→host staging copy, and it is
    /// a wait on the copy — not the copy itself — that blocks.
    pub fn wait_event(&self, ev: *mut std::ffi::c_void) -> Result<(), String> {
        let err = unsafe { cudaEventSynchronize(ev) };
        if err != 0 {
            return Err(format!(
                "cudaEventSynchronize failed: {} ({err})",
                cuda_error_name(err)
            ));
        }
        Ok(())
    }

    /// F5: make every later operation on the stream wait for `ev`, **without
    /// blocking the host** — the synchronization point of a device consumer.
    ///
    /// Still dead in every compilable configuration. #138 (the F5 deferred wait,
    /// landed 2026-10-04) was the ticket expected to wire it into a device→device
    /// staging copy, and it deliberately left it unwired: the only pair that could
    /// express a device destination would need two device backends, `copy_across`
    /// early-returns on a same-backend pair, and CUDA→Metal declines phase A — so
    /// a call site would be unreachable code rather than a caller. The reachable
    /// device-consumer direction (CPU → device) needs no event: its fill is issued
    /// on the consuming pool's own stream, so stream order is the whole wait. Kept
    /// because the census brief names it and
    /// `docs/BACKEND-REGISTRY-DESIGN.md` §11 documents it as that copy class's
    /// mechanism.
    #[allow(dead_code)]
    pub fn stream_wait_event(&self, ev: *mut std::ffi::c_void) -> Result<(), String> {
        let err = unsafe { cudaStreamWaitEvent(self.stream(), ev, 0) };
        if err != 0 {
            return Err(format!(
                "cudaStreamWaitEvent failed: {} ({err})",
                cuda_error_name(err)
            ));
        }
        Ok(())
    }

    /// F5: release an event handle (a no-op on null).
    pub fn event_destroy(&self, ev: *mut std::ffi::c_void) {
        if !ev.is_null() {
            unsafe { cudaEventDestroy(ev) };
        }
    }

    /// F5: enqueue a device→host copy on the stream. The call returns as soon as
    /// the transfer is queued; `dst` must be pinned (see [`Self::host_alloc`]) and
    /// must stay alive until the event that follows it has been waited on.
    pub fn copy_to_host_async(
        &self,
        src: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        bytes: usize,
    ) -> Result<(), String> {
        let err =
            unsafe { cudaMemcpyAsync(dst, src, bytes, CUDA_MEMCPY_DEVICE_TO_HOST, self.stream()) };
        if err != 0 {
            return Err(format!(
                "cudaMemcpyAsync (D2H) failed: {} ({err})",
                cuda_error_name(err)
            ));
        }
        Ok(())
    }

    pub fn sync(&self) {
        // #145: `cudaGetLastError` reports whatever an earlier call latched —
        // it is NOT evidence about the kernel that just ran. Name the observer
        // and the real API error; the counting keeps the error visible instead
        // of dropping it.
        let err = unsafe { cudaGetLastError() };
        if err != 0 {
            LATCHED_API_ERRORS.fetch_add(1, Ordering::Relaxed);
            eprintln!("{}", latched_api_error_message(err));
        }
        let err = unsafe { cudaStreamSynchronize(self.stream()) };
        if err != 0 {
            eprintln!("CUDA stream sync error: {} ({err})", cuda_error_name(err));
        }
    }

    /// #162: the sticky "a REQUIRED kernel launch failed" record, drained.
    ///
    /// `minfer_launch_ok` (a required site) sets it; the site has already named
    /// itself, the instantiation and `cudaGetErrorName` on stderr and cleared the
    /// CUDA latch, so this is the *op-level* consequence. `execute_node` turns it
    /// into an `Err`, which is the one Rust-side check that covers every
    /// launcher — no signature churn, and no consumer ever reads a stale output.
    /// `minfer_launch_ok_opt` (a documented fallback) does not set it.
    ///
    /// Draining (rather than peeking) is what keeps a stale record from poisoning
    /// the *next* node: `execute_node` drains it on both the `Ok` and `Err` arms.
    pub fn take_launch_failure(&self) -> Option<String> {
        if unsafe { minfer_launch_fail_pending() } == 0 {
            return None;
        }
        let site = cstr_owned(unsafe { minfer_launch_fail_site() });
        let name = cstr_owned(unsafe { minfer_launch_fail_name() });
        let code = unsafe { minfer_launch_fail_code() };
        unsafe { minfer_launch_fail_clear() };
        Some(format!(
            "kernel launch {name} failed: {} ({code}) at site {site}",
            cuda_error_name(code)
        ))
    }

    // ─── CUDA Graph (decode step batch) ───────────────────────

    pub fn graph_begin_capture(&self) -> bool {
        let stream = self.stream();
        // Issue #188: the mode is the whole point. Global (1) made another
        // thread's capture-unsafe driver call invalidate this window (or fault
        // inside the driver); thread-local (2) scopes invalidation to the
        // capturing thread. `MINFER_CUDA_CAPTURE_MODE` overrides it for the
        // probe's per-mode measurement; see `capture_mode`.
        let err = unsafe { cudaStreamBeginCapture(stream, capture_mode()) };
        if err != 0 {
            unsafe {
                cudaGetLastError();
            }
            false
        } else {
            true
        }
    }

    /// Close a capture window and return the instantiated exec handle (null
    /// on failure, after clearing the CUDA error state). Used by the
    /// graph-path backend, which owns per-(uid, range) exec storage; the
    /// legacy `graph_end_capture` single-slot flow is unchanged.
    pub fn graph_end_capture_to_exec(&self) -> *mut std::ffi::c_void {
        let stream = self.stream();

        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        LAST_CAPTURE_END_CODE.store(err, Ordering::Relaxed);
        if err != 0 || graph.is_null() {
            if err != 0 {
                unsafe {
                    cudaGetLastError();
                }
            }
            eprintln!("CUDA: stream capture end failed (err {err})");
            // Issue #188: record the raw code so the acceptance probe can
            // distinguish 901 (`cudaErrorStreamCaptureInvalidated`) from every
            // other end-capture failure.
            LAST_CAPTURE_END_CODE.store(err, Ordering::Relaxed);
            return std::ptr::null_mut();
        }

        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe {
            cudaGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        // Issue #147: read `cudaGraphDestroy`'s own return value. `graph` is
        // the `cudaGraph_t` from `cudaStreamEndCapture`, so this is the matching
        // API; a failure here leaks the graph handle but leaves the exec valid,
        // so it is named with `cudaGetErrorName` and cleared instead of being
        // refused. Pre-#145 this call site passed the *exec* (the wrong handle,
        // returning `cudaErrorInvalidValue`), which is exactly what the test
        // knob re-injects to prove this message is produced and the latch gone.
        let destroyed = if test_call_failure_requested("destroy:graph_destroy") {
            exec
        } else {
            graph
        };
        let derr = unsafe { cudaGraphDestroy(destroyed) };
        if derr != 0 {
            eprintln!("{}", graph_destroy_failure_message(derr));
            unsafe {
                cudaGetLastError(); // this site owns the error
            }
        }
        if err != 0 || exec.is_null() {
            eprintln!("CUDA: graph instantiate failed (err {err})");
            return std::ptr::null_mut();
        }
        exec
    }

    /// Free an instantiated graph exec (Phase 7d cache invalidation).
    ///
    /// The handle comes from `cudaGraphInstantiate`, so it is a
    /// `cudaGraphExec_t` and must go to `cudaGraphExecDestroy`. Passing it to
    /// `cudaGraphDestroy` (which takes the `cudaGraph_t` from
    /// `cudaStreamEndCapture`) returns `cudaErrorInvalidValue`, leaks the exec,
    /// and latches an error that the next `sync()` used to report as a kernel
    /// launch failure — issue #145.
    ///
    /// Returns `false` when the destroy call failed (and was named here, then
    /// cleared). The gate asserts this return value, not the latch: the
    /// failure is deliberately cleared here so it cannot resurface as a
    /// phantom launch error, which would otherwise make a
    /// "no latched error" assertion pass for the wrong reason.
    pub fn graph_destroy(&self, exec: *mut std::ffi::c_void) -> bool {
        if exec.is_null() {
            return true;
        }
        let err = unsafe { cudaGraphExecDestroy(exec) };
        if err != 0 {
            // Name it here, then clear it: this call site owns the error.
            eprintln!(
                "CUDA: cudaGraphExecDestroy failed: {} ({err})",
                cuda_error_name(err)
            );
            unsafe {
                cudaGetLastError();
            }
            return false;
        }
        true
    }

    /// Launch an arbitrary instantiated graph exec on the backend stream.
    pub fn graph_launch_exec(&self, exec: *mut std::ffi::c_void) -> bool {
        if exec.is_null() {
            return false;
        }
        let stream = self.stream();
        let err = unsafe { cudaGraphLaunch(exec, stream) };
        if err != 0 {
            unsafe {
                cudaGetLastError();
            }
            return false;
        }
        true
    }

    // ─── Kernel launch operations (called from CudaCommandBuffer) ──

    /// f32-activation matmul dispatch by raw weight pointer + tensor type.
    /// The graph backend (graph/cuda_backend.rs) resolves weights by name and
    /// holds no Tensor, so dispatch takes (ptr, ttype) directly.
    ///
    /// Test-only (#240): production goes through `matmul_f32_ptr_layout`
    /// directly; the only callers of this `padded_q6k: false` shorthand are the
    /// device gates in `graph::cuda_backend::tests`, so `#[cfg(test)] pub(crate)`
    /// is the T1 (#238) form — a `#[cfg(test)]` module in another file is the
    /// caller, so it cannot move into `cuda/tests.rs`.
    #[cfg(test)]
    pub(crate) fn matmul_f32_ptr(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) -> Result<(), String> {
        self.matmul_f32_ptr_layout(wptr, ttype, x, out, od, id, nt, false)
    }

    /// `padded_q6k`: the weight buffer was registered via
    /// `register_weight_q6k_padded` (224-byte block stride).
    pub fn matmul_f32_ptr_layout(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
    ) -> Result<(), String> {
        // 8m: prefill (nt >= 16 — Step 82 lowered the gate to >= 9: batches
        // of 2..8 run the multi-token MMVQ / token-looped legacy kernels,
        // which are weights-once without the M-tile padding waste; doc 81)
        // runs ONE tiled GEMM for every quantized
        // weight type. R1 (2026-08-31): the int8 MMQ GEMM — activations
        // quantized to q8_0 once per call, raw weight bytes staged per tile,
        // mma.m16n8k32 (s8) with per-k-block scale rescale (llama.cpp's MMQ
        // structure; see mmq_nt_kernel). MINFER_MMQ-gated (r60 PROMOTION:
        // default ON — the promoted 1.080x path; `MINFER_MMQ=0` = the f16
        // wmma path). MINFER_NO_PREFILL_GEMM=1 still forces the legacy
        // per-type kernels. id % 32 == 0 covers the block math of every type
        // (q6_K runs as k32 chunks with dual 16-sub rescale inside).
        // doc 91 experiment gate: route nt 2..8 to the mma BT path. Off by
        // default — at nt<=64 the BT grid has only od/NBJ blocks (40 on the
        // 14B shapes) and runs ~1.7x over the multi-MMVQ path; the K-split
        // follow-up raises the block count before this can win.
        let small_m_gemm =
            std::env::var("MINFER_SMALL_M_GEMM").map_or(false, |v| v == "1") && nt >= 2;
        if (nt >= 9 || small_m_gemm)
            && id % 32 == 0
            && !Self::no_prefill_gemm()
            && matches!(
                ttype,
                TensorType::Q4_0
                    | TensorType::Q4_1
                    | TensorType::Q5_0
                    | TensorType::Q5_1
                    | TensorType::Q8_0
                    | TensorType::Q4_K
                    | TensorType::Q5_K
                    | TensorType::Q6_K
            )
        {
            if self.mmq_active() {
                // doc 92 resolution: auto-ksplit is the DEFAULT for every
                // route into the BT GEMM. The auto formula activates only
                // while the grid is M-starved (nt <= 64 => ntb == 1) and
                // falls back to 1 above that, so prefill at nt >= 65 keeps
                // the unsplit single-pass association. Deterministic per
                // (nt, od, id), so capture and replay follow the same
                // partial-sum order.
                let ksplit_req = -1;
                let _ = small_m_gemm;
                return self.prefill_mmq(wptr, ttype, x, out, od, id, nt, padded_q6k, ksplit_req);
            }
            return self.prefill_gemm_f16(wptr, ttype, x, out, od, id, nt, padded_q6k);
        }
        let stream = self.stream();
        // T1 tier note (plan §5.4/§14 R8): the resolved tier's per-type batch
        // limits are tabled and unit-tested (`device_tier::mmvq_cap`) but NOT
        // wired into the decode arms yet — a limit < 8 has no destination for
        // the vacated nt range (BT starts at nt >= 9; the MINFER_SMALL_M_GEMM
        // experiment measured ~1.7x over multi-MMVQ at small nt, doc 91) and
        // would strip the spec identity family (R3). Activation waits for a
        // small-nt BT destination (T2 tile candidates) or field A/B data.
        macro_rules! launch {
            ($f:ident) => {{
                unsafe {
                    $f(
                        wptr as *const u8,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    );
                }
                Ok(())
            }};
        }
        match ttype {
            TensorType::Q4_0 => {
                // 8c: prefill Q8_0-activation GEMM — quantize activations once
                // and run the int8-dot kernel. Standalone A/B (7e② method,
                // bench8c): +38–44% at id ≤ 8192 (activation-heavy shapes:
                // 0.5B all, 7B attn/qkv/o, 7B ffn_gu +4.7%); −63% at
                // 7B ffn_down (id=18944, weight-stream-bound — the q8_0
                // kernel streams weight bytes slower), so the shape gate
                // excludes it. Decode (nt == 1) keeps the f32 kernel.
                // Prefill never enters a CUDA Graph capture window (8g①
                // decode-only gate), so the on-demand scratch grow is safe.
                if nt > 1 && id <= 8192 {
                    let q8 = Self::get_or_grow(
                        &self.buf_q8_prefill,
                        nt * (id / 32) * Q8B,
                    );
                    self.quantize_q8_0(x, q8, id, nt);
                    unsafe {
                        launch_q4_0_q8_0_matmul(
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            od as i32,
                            id as i32,
                            nt as i32,
                            stream,
                        );
                    }
                    Ok(())
                } else if nt == 1 && id >= 2048 && id % 32 == 0 && !Self::no_q40_mmvq() {
                    // doc 103: decode MMVQ (the 8c int8 branch above owns
                    // nt > 1 && id <= 8192 and is untouched; these branches
                    // only claim shapes that previously ran the f32 kernel).
                    self.q4_0_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id > 8192 && id % 32 == 0 && !Self::no_q40_mmvq() {
                    self.q4_0_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q4_0_f32_matmul)
                }
            }
            TensorType::Q8_0 => {
                // doc 103: decode joins the MMVQ structure (dp4a over the
                // shared pad40 q8 activations; the 34-B block stride keeps
                // the f32 kernel's coalesced weight loop out of the picture
                // — measured crossover in doc 103). The f32 kernel stays as
                // the fallback (MINFER_NO_Q80_MMVQ=1) and for every shape
                // the gates exclude.
                if nt == 1 && id >= 2048 && id % 32 == 0 && !Self::no_q80_mmvq() {
                    // doc 104: prefer the p32 split planes when registered
                    // (byte-equal output, +6-10% kernel bandwidth); the raw
                    // doc 103 kernel stays as the fallback.
                    if !Self::no_q80_p32() {
                        if let Some((pp, pd)) = self.q80_p32_planes(wptr as usize) {
                            self.q8_0_p32_decode_mmvq(pp, pd, x, out, od, id, nt);
                            return Ok(());
                        }
                    }
                    self.q8_0_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id >= 2048 && id % 32 == 0 && !Self::no_q80_mmvq() {
                    // size floor as in nt == 1: below ~2048 the extra
                    // quantize launch and the q8 activation rounding lose to
                    // the f32 kernel (and tiny shapes keep the exact f32
                    // numerics the small-shape tests pin down)
                    if !Self::no_q80_p32() {
                        if let Some((pp, pd)) = self.q80_p32_planes(wptr as usize) {
                            self.q8_0_p32_decode_mmvq_multi(pp, pd, x, out, od, id, nt);
                            return Ok(());
                        }
                    }
                    self.q8_0_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q8_0_f32_matmul)
                }
            }
            TensorType::Q4_1 => launch!(launch_q4_1_f32_matmul),
            TensorType::Q4_K => {
                // 8e-reversal: decode (nt == 1) runs the MMVQ structure
                // (dp4a over q8 activations, one row per 256-thread block) —
                // +74–77% at 7B shapes (bench8e2); id >= 2048 gate (below
                // that it is launch-latency noise), id % 32 == 0 for the
                // sub-block tail granularity. Prefill keeps the f32 kernel.
                if nt == 1 && id >= 2048 && id % 32 == 0 {
                    self.q4_k_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id % 32 == 0 {
                    // Step 82: multi-token MMVQ (in-block token loop).
                    self.q4_k_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q4_k_f32_matmul)
                }
            }
            TensorType::Q5_1 => launch!(launch_q5_1_f32_matmul),
            TensorType::Q5_0 => launch!(launch_q5_0_f32_matmul),
            TensorType::Q5_K => {
                // 8f: partial tail super-blocks are masked at 32-element
                // granularity inside the kernel — finer tails unsupported.
                if id % 32 != 0 {
                    return Err(format!(
                        "cuda: Q5_K id {id} not a multiple of 32 (tail masking granularity)"
                    ));
                }
                // 8e follow-up: decode (nt == 1) joins the MMVQ structure
                // (dp4a over q8 activations, one row per 256-thread block).
                // Shape gate measured on-device (dbg micro-bench, padded f32
                // vs mmvq): od*id < ~24M elements loses (od 512 → 4.5x
                // slower, 896 → 3.0x, 2048x4864 → 1.66x) because 1-2 units
                // per thread expose the uncoalesced q5/q6 byte loads; large
                // shapes win (7B ffn_down 3584x18944 → 1.5x faster, lm_head
                // 152064x3584 → 1.4x). MINFER_NO_KQ_MMVQ=1 forces f32.
                if nt == 1 && od * id >= 24_000_000 && !Self::no_kq_mmvq() {
                    self.q5_k_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && !Self::no_kq_mmvq() {
                    // Step 82: multi-token MMVQ — weights-once at any shape
                    // (the 24M nt == 1 crossover does not apply: the in-block
                    // token loop amortizes the uncoalesced load latency).
                    self.q5_k_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q5_k_f32_matmul)
                }
            }
            TensorType::Q6_K => {
                // 8e follow-up: decode (nt == 1) joins the MMVQ structure —
                // 16-element units over q8 activations. blk_stride follows
                // the weight registration (224 padded 7e② repack / 210 raw).
                // Shape gate: see the Q5_K arm comment (measured od*id
                // crossover ~24M elements; below it the padded f32 kernel's
                // coalesced loop wins, above it MMVQ's dp4a wins).
                // D3-7 2b: gate lowered 24M -> 4M for the attn_v class —
                // the 14B attn_v (od 1024 x id 5120 = 5.24M, 11 layers)
                // sat on the padded-f32 kernel at 134.8 GB/s (D3-1 census)
                // while its q6_K MMVQ siblings sustain the 200-225 GB/s
                // class; attn_v/attn_o share the attention-output buffer
                // and id, so the D3-5 MmqCache dedupes their standalone
                // quantize to one launch. GGUF census: no other q6_K shape
                // falls in (4M, 24M) (7B attn_v 1.8M stays padded-f32).
                // Tolerance-gated (f32->q8 activation rounding): D3a
                // package; MINFER_NO_KQ_MMVQ=1 keeps the padded kernel.
                if nt == 1 && id % 32 == 0 && od * id >= 4_000_000 && !Self::no_kq_mmvq() {
                    self.q6_k_decode_mmvq(wptr, x, out, od, id, nt, padded_q6k);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id % 32 == 0 && !Self::no_kq_mmvq() {
                    // Step 82: multi-token MMVQ (v2 needs the padded 224B
                    // stride; the v1 form handles raw 210B too). The 4M
                    // nt == 1 crossover does not apply — weights-once at
                    // any shape once nt >= 2.
                    self.q6_k_decode_mmvq_multi(wptr, x, out, od, id, nt, padded_q6k);
                    Ok(())
                } else if padded_q6k {
                    launch!(launch_q6_k_f32_matmul_padded)
                } else {
                    launch!(launch_q6_k_f32_matmul)
                }
            }
            TensorType::F32 => {
                unsafe {
                    launch_f32_f32_matmul(
                        wptr as *const f32,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    );
                }
                Ok(())
            }
            // #141: an f16 weight matmul. The weights stay 2 B/element on the
            // device (no registration-time f32 copy — that would give up the
            // memory the f16 file exists to save), converted in-register by the
            // kernel. It never enters the int8 MMQ prefill GEMM above: MMQ
            // streams *quantized* bytes and f16 is not one of its formats.
            TensorType::F16 => {
                let rc = unsafe {
                    launch_f16_f32_matmul(
                        wptr as *const u8,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    )
                };
                if rc != 0 {
                    // #147 rule: the launch named itself at the site (see the
                    // `minfer/cuda: kernel launch …` line on stderr); refuse
                    // here instead of running on into a checked error.
                    return Err(format!(
                        "cuda: f16 matmul launch failed for [{od}x{id}] x nt={nt} \
                         (site launch:f16_f32_matmul_*)"
                    ));
                }
                Ok(())
            }
            other => Err(format!(
                "cuda: weight type {other:?} has no f32-activation matmul kernel (supported: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K)"
            )),
        }
    }

    /// r60: the promoted MMQ gate semantics — unset / any non-"0" value =
    /// ON (the verified 1.080x path), explicit "0" = opt-out to the pre-r60
    /// disabled/f16 behavior (the r54 `MINFER_MMQ_Q6K_EXP` pattern).
    /// Single-sourced: every promoted MINFER_MMQ_* dispatch and
    /// plane-registration read goes through this.
    pub fn mmq_gate_on(name: &str) -> bool {
        std::env::var(name).map_or(true, |v| v != "0")
    }

    /// r60: the loaders call this when they register a quantized weight that
    /// is NOT NB-BT-consumable (not q4_K/q6_K), or a 2-D F32 matmul weight:
    /// mode-2 skip-write fused producers become unsound for such mixes (see
    /// `nb_bt_only`) and degrade to mode 1 for the rest of the process.
    /// Registration happens before the first forward, so no per-node cost.
    pub fn clear_mmq_nb_bt_only(&self) {
        self.nb_bt_only
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// R1: `MINFER_MMQ` gates the prefill path INTO the int8 MMQ GEMM.
    /// r60 PROMOTION (2026-09-06): default ON — the full r34-r59 MMQ stack
    /// is parity-green and measures 3590.8 tok/s clean on 7B q4_K_m @3314
    /// (1.080x vs-llama, docs/CUDA_OPTIMIZATION.md P6 r59b); `MINFER_MMQ=0`
    /// opts out to the legacy f16 w16-cache prefill (~2353 tok/s), the
    /// 2026-08-31 default from when the untuned kernel measured ~2.5-3
    /// GMAC/s per matmul under GPU contention vs ~8-11 for the f16
    /// w16-cache path (7B @2K: ~155 vs ~630-880 tok/s). All other A/B
    /// escapes still work: `MINFER_NO_PREFILL_GEMM=1` (legacy per-type
    /// kernels).
    fn mmq_enabled() -> bool {
        Self::mmq_gate_on("MINFER_MMQ")
    }

    /// R1: the int8 MMQ prefill GEMM is active — sm_80+ (mma.m16n8k32 s8;
    /// sm_75 only has k16) and opted in via MINFER_MMQ=1. The loader also
    /// uses this to skip the f16 cache warm pass (MMQ streams raw weight
    /// bytes; the w16 copy would be dead weight).
    /// T1: the sm_80+ check is now the resolved tier's MMQ flag
    /// (`tier_mmq`: table flag, or `cc >= 800` for the GENERIC row) — same
    /// verdict on every currently-built-for device, tier-aware elsewhere.
    pub fn mmq_active(&self) -> bool {
        self.tier_mmq && Self::mmq_enabled()
    }

    /// Device compute capability in `major*100 + minor` encoding (GB10
    /// sm_12.1 = 1201); 0 when no device is initialized. Used by tests to
    /// gate sm_80+ kernels.
    #[cfg(test)]
    pub fn cc(&self) -> i32 {
        self.cc.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// r51/r52: producer-fused A-quantize mode — 0 off, 1 = fused + f32
    /// output write (MINFER_MMQ_A_FUSE=1 semantics), 2 = fused + SKIP the f32
    /// output write (MINFER_MMQ_A_FUSE=2; r60: mode 2 is the DEFAULT when
    /// unset — still degrading to mode 1 under any window-reader/fallback
    /// condition; "0" = off). The base gate is the full r49 MMQ
    /// gate set (the fused plane is only CONSUMED by the raw NB-BT (q4_K) /
    /// q6_K NB transposed GEMM paths, all of which these gates enable; fusing
    /// with any of them off would write a plane the matmul re-quantizes
    /// natively — correct but pure waste).
    ///
    /// Mode 2 additionally requires the r52 window-safety conditions: the f32
    /// output is never written, so (a) every debug/trace reader of node
    /// buffers must be off (MINFER_GRAPH_DUMP layer-0 node dumps, the
    /// debug_dump feature's MINFER_DUMP_DIR, MINFER_TRACE per-node capture,
    /// --viz live capture), and (b) no GEMM path that reads the f32 A may be
    /// reachable (MINFER_NO_PREFILL_GEMM=1 legacy kernels). If any condition
    /// fails, mode 2 degrades to mode-1 semantics (fused, writes the f32) —
    /// still the r51 win, just without the skip. The remaining window
    /// guarantee (producers' outputs consumed only by their immediately
    /// consecutive MatMul group via the MmqCache plane; any dead-write cache
    /// miss REFUSES loudly) is enforced structurally — see MmqCache::dead_write
    /// and docs/CUDA_OPTIMIZATION.md P6 r52 for the full proof.
    pub fn mmq_a_fuse_mode(&self) -> u8 {
        if !(self.mmq_active()
            && Self::mmq_gate_on("MINFER_MMQ_RAW")
            && Self::mmq_gate_on("MINFER_MMQ_RAW_NB")
            && Self::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE")
            && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB"))
        {
            return 0;
        }
        // r60 promotion: unset = mode 2 (the verified-best skip-write fused
        // producers); "1"/"2" keep the r51/r52 override semantics; "0" (and
        // any other unrecognized value, as before r60) = off.
        let requested = match std::env::var("MINFER_MMQ_A_FUSE").as_deref() {
            Err(std::env::VarError::NotPresent) => 2,
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        // r60: mode 2 additionally requires the NB-BT-only weight mix (see
        // `nb_bt_only`) — a mixed-quant model degrades to mode 1 regardless
        // of how mode 2 was requested (default or explicit "2").
        let mode2_possible = requested == 2
            && !Self::no_prefill_gemm()
            && self.nb_bt_only.load(std::sync::atomic::Ordering::Relaxed);
        match requested {
            1 => 1,
            2 if mode2_possible
                && std::env::var_os("MINFER_GRAPH_DUMP").is_none()
                && std::env::var_os("MINFER_DUMP_DIR").is_none()
                && !crate::trace::enabled()
                && !crate::live::enabled() =>
            {
                2
            }
            // Mode 2 requested but a window-reader/fallback condition is
            // active: keep the r51 fused semantics (write the f32 output).
            2 => 1,
            _ => 0,
        }
    }

    /// r49: invalidate the MMQ A-quantize memoization. Called by the CUDA
    /// backend between non-MMQ nodes (conservative consecutive-window rule)
    /// and at split boundaries (cross-execution staleness).
    pub fn clear_mmq_cache(&self) {
        // #188: only this stream's entry — a foreign engine's window is not ours
        // to invalidate.
        self.mmq_cache.lock().unwrap().remove(&current_stream_key());
    }

    /// r49: transposed-A helper for the MMQ quantize prepass — returns
    /// `(qa8, sda)` device pointers holding `x`'s q8_0 pad40_t-transposed
    /// quantized form, launching `quantize_q8_0_pad40_t` ONLY when the
    /// last MMQ A (consecutive, same (src,nt,id)) is not already there. A
    /// cache hit therefore skips the prepass launch entirely — the caller
    /// still feeds the returned pointers to the GEMM, which is byte-identical
    /// (quantize depends only on x/nt/id, not the weight type).
    unsafe fn mmq_quantize_transposed(
        &self,
        x: *const f32,
        id: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    ) -> (usize, usize) {
        let need_qa8 = (ntb as usize) * (id as usize / 32) * 2048;
        let need_sda = (ntb as usize) * (id as usize / 32) * 256;
        let key = (x as usize, nt as usize, id as usize);
        let mut map = self.mmq_cache.lock().unwrap();
        let cache = map.entry(current_stream_key()).or_default();
        if cache.active && cache.key == key && cache.transposed {
            // get_or_grow may have reallocated on a larger miss: validate the
            // physical pointers so a grown buffer is never reused stale.
            let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8) as usize;
            let sda = Self::get_or_grow(&self.buf_sda_t, need_sda) as usize;
            if qa8 == cache.qa8 && sda == cache.sda {
                return (qa8, sda);
            }
        }
        // r52: a mode-2 (skip-write) fused producer left the f32 src UNWRITTEN
        // and promised its consumers would hit this cache. Reaching the launch
        // path with a dead-write entry for the SAME buffer means the window
        // guarantee broke (split boundary, exotic fallback, pointer drift) —
        // re-quantizing would read the dead buffer's garbage. Refuse: the
        // callers treat (0, 0) as a failed path and prefill_mmq errors out
        // loudly instead of silently producing wrong results.
        if cache.active && cache.dead_write && cache.key.0 == x as usize {
            eprintln!(
                "minfer/cuda: MMQ A-quantize refused: mode-2 dead-write A \
                 (ptr {:#x}) missed the MmqCache window (MINFER_MMQ_A_FUSE=2 \
                 safety guard; A falls back and errors out)",
                x as usize
            );
            return (0, 0);
        }
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8);
        let sda = Self::get_or_grow(&self.buf_sda_t, need_sda);
        launch_quantize_q8_0_pad40_t(
            x,
            qa8 as *mut u8,
            sda as *mut u8,
            id,
            nt,
            nchunk,
            ntb,
            stream,
        );
        cache.active = true;
        cache.key = key;
        cache.transposed = true;
        cache.dead_write = false;
        cache.qa8 = qa8 as usize;
        cache.sda = sda as usize;
        cache.q8 = 0;
        (qa8 as usize, sda as usize)
    }

    /// r49: native (non-transposed) counterpart of
    /// [`Self::mmq_quantize_transposed`] — the pad40 q8_0 `q8` buffer used by
    /// the fallback NB/wide/narrow kernels. Same consecutive-window rule; the
    /// quantize output is again a pure function of (x, nt, id).
    unsafe fn mmq_quantize_native(
        &self,
        x: *const f32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> usize {
        let need = (nt as usize) * (id as usize / 32) * 40;
        let key = (x as usize, id as usize, nt as usize);
        let mut map = self.mmq_cache.lock().unwrap();
        let cache = map.entry(current_stream_key()).or_default();
        if cache.active && cache.key == key && !cache.transposed {
            let q8 = Self::get_or_grow(&self.buf_q8_prefill, need) as usize;
            if q8 == cache.q8 {
                return q8;
            }
        }
        // r52: same dead-write refusal as mmq_quantize_transposed — a mode-2
        // fused producer's f32 src was never written; re-quantizing it here
        // would read garbage. Return 0 (callers fail loudly).
        if cache.active && cache.dead_write && cache.key.0 == x as usize {
            eprintln!(
                "minfer/cuda: native A-quantize refused: mode-2 dead-write A \
                 (ptr {:#x}) (MINFER_MMQ_A_FUSE=2 safety guard)",
                x as usize
            );
            return 0;
        }
        let q8 = Self::get_or_grow(&self.buf_q8_prefill, need);
        launch_quantize_q8_0_pad40(x, q8 as *mut u8, id, nt, stream);
        cache.active = true;
        cache.key = key;
        cache.transposed = false;
        cache.dead_write = false;
        cache.qa8 = 0;
        cache.sda = 0;
        cache.q8 = q8 as usize;
        q8 as usize
    }

    /// D3-5 1a: decode-side native pad40 A-quantize with r49-style
    /// consecutive-consumer memoization. The fused decode producers
    /// ([`Self::rms_norm_quant_on_gpu`] / [`Self::swiglu_quant_off_on_gpu`])
    /// record their pad40 plane here keyed on the producer's f32 output
    /// pointer; the decode matmul group that consumes it hits the cache and
    /// skips the standalone `quantize_q8_0_pad40` launch. Same window-safety
    /// contract as the prefill MmqCache: the plane is a pure function of
    /// (src, nt, id), any non-(MatMul|FusedFFN) node clears the entry, and
    /// `synchronize` clears it at execution boundaries. The hit path
    /// re-validates the physical pointer (a `get_or_grow` between record and
    /// consult must not have moved the plane — a grown-over entry falls back
    /// to the standalone launch, which re-quantizes from the live f32 src).
    /// `MINFER_NO_DECODE_A_FUSE=1` skips the consult AND the record, restoring
    /// the exact pre-D3-5 per-matmul standalone quantize (A/B gate).
    fn decode_quantize_native(&self, x: *const f32, id: usize, nt: usize) -> *mut u8 {
        let need = nt * (id / 32) * 40;
        let key = (x as usize, nt, id);
        if !Self::no_decode_a_fuse() {
            let map = self.mmq_cache.lock().unwrap();
            if let Some(cache) = map.get(&current_stream_key()) {
                if cache.active && !cache.transposed && !cache.dead_write && cache.key == key {
                    let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as usize;
                    if q8 == cache.q8 {
                        return q8 as *mut u8;
                    }
                }
            }
        }
        let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as *mut u8;
        let stream = self.stream();
        unsafe {
            launch_quantize_q8_0_pad40(x, q8, id as i32, nt as i32, stream);
        }
        if !Self::no_decode_a_fuse() {
            self.record_mmq_cache_native(key.0, nt, id, q8 as usize);
        }
        q8
    }

    /// D3-5 1a: record a fused-producer NATIVE (pad40, non-transposed) plane
    /// into the MmqCache (decode counterpart of
    /// [`Self::record_mmq_cache_transposed`]).
    fn record_mmq_cache_native(&self, src: usize, nt: usize, id: usize, q8: usize) {
        let mut map = self.mmq_cache.lock().unwrap();
        let c = map.entry(current_stream_key()).or_default();
        c.active = true;
        c.key = (src, nt, id);
        c.transposed = false;
        c.dead_write = false;
        c.qa8 = 0;
        c.sda = 0;
        c.q8 = q8;
    }

    /// D3-5 1a: opt-out of the decode fused-producer A-quantize (A/B escape
    /// hatch): plain rms_norm/swiglu producers + unconditional standalone
    /// quantize per matmul.
    pub fn no_decode_a_fuse() -> bool {
        std::env::var("MINFER_NO_DECODE_A_FUSE").map_or(false, |v| v == "1")
    }

    /// r51: record a PRODUCER-FUSED quantize result into the r49 MmqCache.
    /// The fused rms_norm/swiglu kernel wrote the pad40_t plane for `src`
    /// (the producer's f32 output device pointer) directly, so the following
    /// matmul group's `mmq_quantize_transposed` hits the cache (same
    /// (src, nt, id) key, same buf_qa8_t/buf_sda_t pointers) and skips its
    /// own prepass launch. The cache-window rules are unchanged: any
    /// non-MatMul node clears the entry (the producer's own execution is the
    /// set point — the backend's clear runs before the node executes), and
    /// `synchronize` clears it at split boundaries, so a stale plane can
    /// never be consumed. r52: `dead_write` marks mode-2 entries (the f32
    /// src was NOT written — see the mode-2 refusal guards in
    /// `mmq_quantize_transposed`/`mmq_quantize_native`).
    fn record_mmq_cache_transposed(
        &self,
        src: usize,
        nt: usize,
        id: usize,
        qa8: usize,
        sda: usize,
        dead_write: bool,
    ) {
        let mut map = self.mmq_cache.lock().unwrap();
        let c = map.entry(current_stream_key()).or_default();
        c.active = true;
        c.key = (src, nt, id);
        c.transposed = true;
        c.dead_write = dead_write;
        c.qa8 = qa8;
        c.sda = sda;
        c.q8 = 0;
    }

    /// r51: fused rms_norm + pad40_t A-quantize (MINFER_MMQ_A_FUSE=1, full
    /// MMQ gate set). Writes the f32 rms output `y` (bit-identical to
    /// [`Self::rms_norm`]) AND the transposed q8_0 plane of `y` in ONE pass,
    /// then registers the plane in the MmqCache keyed on `y`'s device
    /// pointer — the following matmul group's `prefill_mmq` hits the cache
    /// and skips its quantize launch. `n` rows × `d` dims; the caller gates
    /// on n >= 16 (prefill_mmq territory; decode/capture paths keep the
    /// unfused pair) and d % 256 == 0 (the transposed GEMM's nchunk % 8
    /// requirement). Plane scratch sizes match `mmq_quantize_transposed`'s
    /// exactly so the hit validation's get_or_grow returns the same pointer.
    /// Err = plane OOM (caller falls back to the unfused pair).
    pub fn rms_norm_quant(
        &self,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        let nchunk = d / 32;
        let ntb = n.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: rms_norm_quant plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_rms_norm_quant_f32_t(
                x as *const f32,
                w as *const f32,
                y as *mut f32,
                qa8 as *mut u8,
                sda as *mut u8,
                d as i32,
                eps,
                n as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(y as usize, n, d, qa8 as usize, sda as usize, false);
        Ok(())
    }

    /// r51: fused swiglu + pad40_t A-quantize — the SwiGLU counterpart of
    /// [`Self::rms_norm_quant`]; `dst` (silu(gate)*up, `dim` = nf) is both
    /// the f32 node output and the quantize source. Registered for the
    /// immediately following down projection.
    pub fn swiglu_quant(
        &self,
        gate: *mut std::ffi::c_void,
        up: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        dim: usize,
        nt: usize,
    ) -> Result<(), String> {
        let nchunk = dim / 32;
        let ntb = nt.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: swiglu_quant plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_swiglu_quant_f32_t(
                gate as *const f32,
                up as *const f32,
                dst as *mut f32,
                qa8 as *mut u8,
                sda as *mut u8,
                dim as i32,
                nt as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(dst as usize, nt, dim, qa8 as usize, sda as usize, false);
        Ok(())
    }

    /// r52: mode-2 (MINFER_MMQ_A_FUSE=2) variant of [`Self::rms_norm_quant`]
    /// — SAME pad40_t plane, but the f32 output `y` is NOT written. `y` is
    /// passed only as the MmqCache key (the consuming matmuls' A pointer);
    /// its buffer is expected to be dead: consumed exclusively by the
    /// immediately following consecutive MatMul group through the plane. The
    /// entry is recorded with `dead_write = true`, so any window violation
    /// (a GEMM path that would re-quantize the unwritten buffer) refuses and
    /// fails loudly instead of reading garbage — see
    /// [`Self::mmq_a_fuse_mode`] for the dispatch conditions and
    /// docs/CUDA_OPTIMIZATION.md P6 r52 for the proof. Err = plane OOM
    /// (caller degrades to mode 1 / the unfused pair, both of which write).
    pub fn rms_norm_quant_nw(
        &self,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        let nchunk = d / 32;
        let ntb = n.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: rms_norm_quant_nw plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_rms_norm_quant_nw_f32_t(
                x as *const f32,
                w as *const f32,
                qa8 as *mut u8,
                sda as *mut u8,
                d as i32,
                eps,
                n as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(y as usize, n, d, qa8 as usize, sda as usize, true);
        Ok(())
    }

    /// r52: mode-2 counterpart of [`Self::swiglu_quant`] — same plane, the
    /// f32 `dst` is NOT written (key only). See [`Self::rms_norm_quant_nw`].
    pub fn swiglu_quant_nw(
        &self,
        gate: *mut std::ffi::c_void,
        up: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        dim: usize,
        nt: usize,
    ) -> Result<(), String> {
        let nchunk = dim / 32;
        let ntb = nt.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: swiglu_quant_nw plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_swiglu_quant_nw_f32_t(
                gate as *const f32,
                up as *const f32,
                qa8 as *mut u8,
                sda as *mut u8,
                dim as i32,
                nt as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(dst as usize, nt, dim, qa8 as usize, sda as usize, true);
        Ok(())
    }

    /// R1: int8 MMQ prefill GEMM — quantize activations to q8_0 (pad40
    /// blocks; the quantize kernel also emits the per-block int sum used by
    /// the min-term correction), then one tiled mma.m16n8k32 (s8) launch per
    /// weight (see mmq_nt_kernel). Caller guarantees nt >= 16, id % 32 == 0,
    /// and a supported quant type. The q8 scratch follows the same
    /// grow-on-demand lifecycle as the f16 path's buf_f16_x: the 3-run
    /// capture protocol sizes it before the capture window opens.
    /// doc 92 auto-ksplit, parameterized (T2, plan §6.1): the M-starve gate
    /// stays `nt <= 64` (ntb == 1 — a single M tile row) and the resident-
    /// block target becomes `max(256, 2*SM)`. On GB10 (48 SMs) that is
    /// exactly the calibrated 256 — zero behavior change — while larger SM
    /// counts scale the target proportionally. `MINFER_MMQ_KSPLIT_TARGET`
    /// still overrides everything (explicit human choice beats the formula).
    fn auto_ksplit(&self, nt: usize, nbt_y: usize, nktile: usize) -> usize {
        if nt > 64 || nbt_y == 0 || nktile <= 1 {
            return 1;
        }
        let target: usize = std::env::var("MINFER_MMQ_KSPLIT_TARGET")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| (256usize).max(2 * self.sm_count as usize));
        let want = (target + nbt_y - 1) / nbt_y;
        want.clamp(2, nktile)
    }

    pub fn prefill_mmq(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
        ksplit_req: i32,
    ) -> Result<(), String> {
        let type_id = match ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            other => return Err(format!("cuda: prefill MMQ got unsupported type {other:?}")),
        };
        // r49: the native pad40 buffer is sized upfront so an OOM surfaces here
        // (before any GEMM launch) instead of as a null deref in the q4_K
        // fallback's final `launch_mmq_raw_nt`. The fallbacks re-derive it via
        // `mmq_quantize_native` (which dedups); this call only guards OOM.
        if Self::get_or_grow(&self.buf_q8_prefill, nt * (id / 32) * 40).is_null() {
            return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
        }
        // Q6_K reads whichever layout the weight was registered with
        // (224-byte padded 7e② repack or raw 210); all others are raw.
        let block_stride: i32 = if ttype == TensorType::Q6_K && padded_q6k {
            224
        } else {
            210
        };
        let stream = self.stream();
        // P6 r38: q6_K on a raw-byte BT-style kernel (expanded centered-int8 B,
        // m16n8k16 KSPLIT=2). Same A prepass as r34; the B path is q6_K-specific.
        // Gated on MINFER_MMQ_Q6K_NB + MINFER_MMQ_RAW (r60: default-on,
        // "0" opts out; the q6k path always uses the transposed-A raster).
        // On any cap/mismatch the launcher returns 0 -> clean fall through
        // to the generic mmq_nt<7,2>.
        if type_id == 7
            && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")
            && Self::mmq_gate_on("MINFER_MMQ_RAW")
            && (id / 32) % 8 == 0
        {
            let nchunk = (id / 32) as i32;
            let ntb = ((nt as i64 + 63) / 64) as i32;
            unsafe {
                // r49: A-quantize prepass via the consecutive-window cache —
                // a same-A (>q/k/v, gate/up) matmul reuses qa8g/sdag without a
                // fresh quantize launch.
                let (qa8g, sdag) = self.mmq_quantize_transposed(
                    x as *const f32,
                    id as i32,
                    nt as i32,
                    nchunk,
                    ntb,
                    stream,
                );
                // r53: pre-expanded B plane lookup by the padded weight's
                // device pointer (null on miss -> launcher selects the r41
                // in-kernel-expand instantiation).
                let w_exp = self
                    .q6k_exp
                    .lock()
                    .unwrap()
                    .get(&(wptr as usize))
                    .map(|cp| cp.0)
                    .unwrap_or(std::ptr::null_mut());
                // r56 (Session E item 2b): the precomputed dsc f32-pair plane
                // (null on miss -> the r41 scalar dsc path in-kernel).
                let w_dsc = self
                    .q6k_dsc
                    .lock()
                    .unwrap()
                    .get(&(wptr as usize))
                    .map(|cp| cp.0)
                    .unwrap_or(std::ptr::null_mut());
                // doc 92: same K-split contract as the q4_K path (T2: the
                // target is SM-count-parameterized — see auto_ksplit).
                let ksplit: usize = if ksplit_req < 0 {
                    self.auto_ksplit(nt, (od + 127) / 128, (nchunk as usize + 1) / 2)
                } else {
                    ksplit_req.max(1) as usize
                };
                let cpart = if ksplit > 1 {
                    Self::get_or_grow(&self.buf_mmq_ksplit, ksplit * nt * od * 4) as *mut f32
                } else {
                    std::ptr::null_mut()
                };
                if qa8g != 0
                    && sdag != 0
                    && (ksplit == 1 || !cpart.is_null())
                    && launch_mmq_raw_nb_bt_q6k_nt(
                        type_id,
                        wptr as *const u8,
                        w_exp as *const u8,
                        w_dsc as *const u8,
                        qa8g as *const u8,
                        sdag as *const u8,
                        out as *mut f32,
                        nt as i32,
                        od as i32,
                        id as i32,
                        nchunk,
                        block_stride,
                        stream,
                        8,
                        cpart,
                        ksplit as i32,
                    ) == 1
                {
                    if std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1") {
                        // r54: name WHY the r41 in-kernel expand is running —
                        // "exp=off" is the intentional MINFER_MMQ_Q6K_EXP=0
                        // switch; "fallback!" means a W_exp build was expected
                        // (padded weight, EXP gate on) but the map missed
                        // (alloc/upload failure or a registration bug). Raw
                        // 210-B weights never get a plane -> kept unqualified.
                        let b = if !w_exp.is_null() {
                            "W_exp-cp.async"
                        } else if !padded_q6k {
                            "in-kernel-expand"
                        } else if std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() == Ok("0") {
                            "in-kernel-expand(exp=off)"
                        } else {
                            "in-kernel-expand(fallback!)"
                        };
                        // r56: name the A/dsc staging paths too (liveness
                        // check per the r53 lesson — a fallback-correct fast
                        // path needs a visible label, parity cannot see it).
                        let a = if !w_dsc.is_null() {
                            "A=cp.async DSC=f32-plane"
                        } else {
                            "A=cp.async DSC=scalar"
                        };
                        eprintln!(
                            "minfer/cuda: mmq raw NB-BT q6_K kernel active \
                             (r56 {}, r54 B={}, r39 KDR=2 double-buffer, \
                             A-transpose)",
                            a, b
                        );
                    }
                    return Ok(());
                }
            }
        }
        // P6: raw-byte staging variant (q4_K, whole super-blocks only).
        // Same quantized activations; the GEMM stages RAW weight bytes via
        // cp.async and dequants in registers (docs/CUDA_OPTIMIZATION.md).
        if type_id == 5 && Self::mmq_gate_on("MINFER_MMQ_RAW") && (id / 32) % 8 == 0 {
            let kd: i32 = std::env::var("MINFER_MMQ_RAW_KD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
            let wide = std::env::var("MINFER_MMQ_RAW_WIDE").as_deref() == Ok("1");
            let nb = Self::mmq_gate_on("MINFER_MMQ_RAW_NB");
            let nb_debug = std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1");
            let at = Self::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE");
            unsafe {
                // P6 r34: relocate the A-side layout transform out of the mma
                // kernel into a quantize-transpose prepass (llama.cpp's design).
                // Under MINFER_MMQ_A_TRANSPOSE=1 the activations are emitted
                // PRE-TRANSPOSED (quantize_q8_0_pad40_t) so the NB kernel's A
                // staging is a bulk LDG->STS; the native q8 buffer is only
                // filled on the (rare) bb-bt fallback below.
                let mut nb_ok = false;
                if nb && at && kd == 8 {
                    let nchunk = (id / 32) as i32;
                    let ntb = ((nt as i64 + 63) / 64) as i32;
                    // r49: cache-backed transposed A-quantize prepass (reuses
                    // the previous same-A matmul's qa8g/sdag when consecutive).
                    let (qa8g, sdag) = self.mmq_quantize_transposed(
                        x as *const f32,
                        id as i32,
                        nt as i32,
                        nchunk,
                        ntb,
                        stream,
                    );
                    // r59: the q4_K W_dsc f32-pair plane (null on miss ->
                    // the DSC=false in-kernel scalar decode instantiation).
                    let w_dsc = self
                        .q4k_dsc
                        .lock()
                        .unwrap()
                        .get(&(wptr as usize))
                        .map(|cp| cp.0)
                        .unwrap_or(std::ptr::null_mut());
                    // doc 92: K-split the K range across grid.z slots when
                    // the grid is M-starved (ntb == 1 => only od/128 blocks).
                    // ksplit_req < 0 = auto (small-M gate): target >= 256
                    // resident blocks (about 2/SM); 1 = the unsplit path
                    // (default prefill, bitwise unchanged).
                    // T2: the auto formula moved into auto_ksplit (shared
                    // with the q6_K NB path; SM-count-parameterized target).
                    let ksplit: usize = if ksplit_req < 0 {
                        self.auto_ksplit(nt, (od + 127) / 128, (nchunk as usize + 7) / 8)
                    } else {
                        ksplit_req.max(1) as usize
                    };
                    let cpart = if ksplit > 1 {
                        Self::get_or_grow(&self.buf_mmq_ksplit, ksplit * nt * od * 4) as *mut f32
                    } else {
                        std::ptr::null_mut()
                    };
                    nb_ok = qa8g != 0
                        && sdag != 0
                        && (ksplit == 1 || !cpart.is_null())
                        && launch_mmq_raw_nb_bt_nt(
                            type_id,
                            wptr as *const u8,
                            w_dsc as *const u8,
                            qa8g as *const u8,
                            sdag as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            nchunk,
                            stream,
                            kd,
                            cpart,
                            ksplit as i32,
                        ) == 1;
                    if nb_ok && nb_debug {
                        // r59: name the dsc staging path (liveness check per
                        // the r53 lesson — a fallback-correct fast path needs
                        // a visible label, parity cannot see it). "fallback!"
                        // means a plane was expected (gate on) but the map
                        // missed (alloc/upload failure or registration bug).
                        let d = if !w_dsc.is_null() {
                            "DSC=f32-plane"
                        } else if std::env::var("MINFER_MMQ_Q4K_DSC").as_deref() == Ok("0") {
                            "DSC=in-kernel(dsc=off)"
                        } else {
                            "DSC=in-kernel(fallback!)"
                        };
                        eprintln!(
                            "minfer/cuda: mmq raw NB-BT kernel active \
                             (KD=8, A-transpose, r59 {d})"
                        );
                    }
                }
                if !nb_ok {
                    // r49: cache-backed native A-quantize prepass. The helper
                    // returns 0 (no buffer) on OOM; the launchers below only
                    // run when a buffer is present (the original unconditional
                    // `q8` came from a pre-call get_or_grow, now inside the
                    // helper).
                    let q8 =
                        self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
                    // Direction-A NB raw-nibble kernel is KD=8-native; it
                    // activates only under the full MMQ gate set. launcher
                    // returns 0 on KD!=8 or smem/reg cap failure -> clean
                    // fallback to the wide/narrow raw path below.
                    nb_ok = q8 != 0
                        && nb
                        && launch_mmq_raw_nb_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 1;
                    if nb_ok && nb_debug {
                        eprintln!("minfer/cuda: mmq raw NB kernel active (KD=8)");
                    }
                }
                if !nb_ok {
                    let q8 =
                        self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
                    if q8 == 0 {
                        // r52: mmq_quantize_native refuses a mode-2 dead-write
                        // A (and plain q8 OOM is pre-checked at fn entry) —
                        // never fall through to a GEMM on a null/garbage A.
                        return Err("cuda: prefill MMQ: A-quantize unavailable (mode-2 \
                             dead-write A refused or q8 OOM); MINFER_MMQ_A_FUSE=2 \
                             window violated"
                            .to_string());
                    }
                    let wide_ok = q8 != 0
                        && wide
                        && launch_mmq_raw_wide_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 1;
                    if !wide_ok
                        && launch_mmq_raw_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 0
                    {
                        // #147: the terminal raw-narrow launcher refused the
                        // launch (its dynamic-smem opt-in failed or the launch
                        // itself errored). It named the call, the instantiation
                        // and `cudaGetErrorName` on stderr — do not let the
                        // graph continue on an unwritten output.
                        return Err("cuda: prefill MMQ: launch_mmq_raw_nt refused the launch \
                                    (see the named CUDA error on stderr); no kernel ran"
                            .to_string());
                    }
                }
            }
            return Ok(());
        }
        unsafe {
            let q8 = self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
            if q8 == 0 {
                return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
            }
            if launch_mmq_nt(
                type_id,
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                nt as i32,
                od as i32,
                id as i32,
                block_stride,
                stream,
            ) == 0
            {
                // #147: same contract as launch_mmq_raw_nt above.
                return Err(
                    "cuda: prefill MMQ: launch_mmq_nt refused the launch (see the \
                            named CUDA error on stderr); no kernel ran"
                        .to_string(),
                );
            }
        }
        Ok(())
    }

    /// 8m: prefill GEMM — dequant the weight to f16 scratch once, convert
    /// activations to f16, then one tensor-core GEMM (see the dispatch-gate
    /// comment in `matmul_f32_ptr_layout`). Caller guarantees nt >= 16 and
    /// id % 32 == 0 for the supported quant types.
    fn prefill_gemm_f16(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
    ) -> Result<(), String> {
        // 8p: default is the cp.async f16 GEMM over a PERSISTENT per-weight
        // f16 copy (dequant runs once per weight per process — the per-call
        // 288 ms dequant is gone). MINFER_FUSED_B=1 opts into the
        // dequant-in-GEMM kernel instead (memory-lean: no f16 weight cache,
        // but re-dequantizes every B tile per nt sweep — slower on large
        // nt). Both require id % 256 == 0 for alignment/coverage.
        let fused = id % 256 == 0 && Self::fused_b_on();
        self.prefill_gemm_f16_inner(wptr, ttype, x, out, od, id, nt, padded_q6k, fused)
    }

    /// 8p: opt the process into the f16 weight cache (loader-side, for
    /// models big enough to amortize it — see W16_ENABLE_BYTES).
    pub fn enable_w16_cache(&self) {
        self.w16_enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// 8p: warm the persistent f16 cache for one registered matmul weight
    /// at LOAD time (the loader has the tensor metadata; the lazy path
    /// would otherwise put ~8.6 GB of cudaMalloc + the full dequant inside
    /// the first — timed — prefill). No-op for non-quant types and for
    /// rows not covered by the fused-GEMM alignment gate.
    pub fn warm_w16(&self, name: &str, t: &crate::tensor::Tensor) -> bool {
        let type_id = match t.ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            _ => return false,
        };
        // GGUF convention: metadata [in, out] → od = shape[1], id = shape[0].
        let od = t.shape[1] as usize;
        let id = t.shape[0] as usize;
        if od == 0 || id == 0 || id % 256 != 0 {
            return false;
        }
        let padded_q6k =
            t.ttype == TensorType::Q6_K && self.padded_weights.lock().unwrap().contains_key(name);
        let block_stride = if padded_q6k { 224 } else { 210 };
        let Some(wptr) = self.get_weight_ptr(name) else {
            return false;
        };
        self.w16_get(wptr, type_id, od, id, block_stride).is_some()
    }

    /// Persistent f16 copy of a registered quantized weight (8p). Returns
    /// None when the allocation fails (caller falls back to the per-call
    /// scratch) or when MINFER_NO_W16CACHE=1.
    fn w16_get(
        &self,
        wptr: *mut std::ffi::c_void,
        type_id: i32,
        od: usize,
        id: usize,
        block_stride: i32,
    ) -> Option<*mut std::ffi::c_void> {
        if Self::no_w16cache() || !self.w16_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let bytes = od * id * 2;
        if let Some((p, sz)) = self.w16_cache.lock().unwrap().get(&(wptr as usize)) {
            if *sz == bytes {
                return Some(p.0);
            }
        }
        // Memory-pressure valve: the cache doubles the resident weight
        // footprint; skip it (caller falls back to the per-call scratch =
        // pre-8p behavior) unless free memory comfortably covers the copy —
        // the test suite keeps several loaded models resident (registry
        // entries are never freed), and +1 GB caches per model exhausted the
        // ~23 GB free pool and broke later models' weight uploads.
        {
            let (mut free_mem, mut total_mem) = (0usize, 0usize);
            let rc = unsafe { cudaMemGetInfo(&mut free_mem, &mut total_mem) };
            // A failed query is *not* "there is room" (issue #122): the pre-#122
            // `rc == 0 && …` shape let a failure fall through to the allocation,
            // which is the fail-open direction for an optional cache. Take the
            // documented conservative branch instead — no measurement, no cache.
            if rc != 0 || free_mem < 2 * bytes + (4usize << 30) {
                return None;
            }
        }
        let ptr = Self::cuda_malloc(bytes);
        if ptr.is_null() {
            eprintln!("CUDA: w16 cache OOM allocating {bytes} bytes");
            return None;
        }
        unsafe {
            launch_dequant_f16(
                type_id,
                wptr as *const u8,
                ptr,
                od as i32,
                id as i32,
                block_stride,
                self.stream(),
            );
        }
        self.w16_cache
            .lock()
            .unwrap()
            .insert(wptr as usize, (CudaPtr(ptr), bytes));
        Some(ptr)
    }

    /// Prefill GEMM with an explicit fused/legacy switch (test entry point
    /// for the bit-parity check between the two paths).
    pub(crate) fn prefill_gemm_f16_inner(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
        fused: bool,
    ) -> Result<(), String> {
        let stream = self.stream();
        let type_id = match ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            other => return Err(format!("cuda: prefill GEMM got unsupported type {other:?}")),
        };
        // Q6_K reads whichever layout the weight was registered with
        // (224-byte padded 7e② repack or raw 210); all others are raw.
        let block_stride: i32 = if ttype == TensorType::Q6_K && padded_q6k {
            224
        } else {
            210
        };
        if fused {
            // 8p: A converts to f16 once (the convert pass stays); the B
            // side dequantizes raw quantized bytes inside the GEMM — no
            // buf_f16_w round trip, no launch_dequant_f16.
            let x16 = Self::get_or_grow(&self.buf_f16_x, nt * id * 2);
            unsafe {
                launch_convert_f16(x as *const f32, x16, (nt * id) as i64, stream);
                launch_gemm_qb_nt(
                    x16,
                    wptr as *const u8,
                    out as *mut f32,
                    nt as i32,
                    od as i32,
                    id as i32,
                    type_id,
                    block_stride,
                    stream,
                );
            }
            return Ok(());
        }
        let w16 = match self.w16_get(wptr, type_id, od, id, block_stride) {
            Some(p) => p, // persistent copy, dequant already done
            None => {
                let w16 = Self::get_or_grow(&self.buf_f16_w, od * id * 2);
                unsafe {
                    launch_dequant_f16(
                        type_id,
                        wptr as *const u8,
                        w16,
                        od as i32,
                        id as i32,
                        block_stride,
                        stream,
                    );
                }
                w16
            }
        };
        // P6 negative result: in-kernel f32->f16 A conversion measured
        // ~8% SLOWER end-to-end (2172-2177 vs 2365 tok/s) — the A stage
        // loses cp.async and its synchronous 32B loads stall every k-tile,
        // costing more than the 56 ms convert pass it removes. Kept behind
        // MINFER_GEMM_A32=1 for the smem-mirror variant follow-up (cp.async
        // the f32 tile to smem, convert smem->smem — no global round trip).
        let af32 = std::env::var("MINFER_GEMM_A32")
            .map(|v| v == "1")
            .unwrap_or(false);
        if af32 {
            let launched = unsafe {
                std::env::var("MINFER_A32_DEBUG")
                    .is_ok()
                    .then(|| eprintln!("minfer/cuda: af32 gemm nt={nt} od={od} id={id}"));
                launch_gemm_f32a(
                    x as *const f32,
                    w16,
                    out as *mut f32,
                    nt as i32,
                    od as i32,
                    id as i32,
                    stream,
                )
            };
            if launched == 0 {
                // #147: a refused/failed prefill GEMM launch is an error, not a
                // silent pass over an unwritten output. The site named it.
                return Err(
                    "cuda: prefill GEMM (af32): launch_gemm_f32a refused the launch \
                            (see the named CUDA error on stderr); no kernel ran"
                        .to_string(),
                );
            }
            return Ok(());
        }
        let x16 = Self::get_or_grow(&self.buf_f16_x, nt * id * 2);
        let launched = unsafe {
            launch_convert_f16(x as *const f32, x16, (nt * id) as i64, stream);
            launch_gemm_f16(
                x16,
                w16,
                out as *mut f32,
                nt as i32,
                od as i32,
                id as i32,
                stream,
                false,
            )
        };
        if launched == 0 {
            // #147: same contract as the af32 arm above.
            return Err(
                "cuda: prefill GEMM (f16): launch_gemm_f16 refused the launch \
                        (see the named CUDA error on stderr); no kernel ran"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn quantize_q8_0(
        &self,
        x: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        dim: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_quantize_q8_0(x as *const f32, y as *mut u8, dim as i32, nt as i32, stream);
        }
    }

    /// 7e⑤: in-place split swiglu over one buffer (llama
    /// `ggml_swiglu_split`): buf[i] = silu(buf[i]) * buf[off + i] for
    /// i in 0..n. Used by the fused FFN decode path where the concat
    /// matmul output carries gate rows 0..nf and up rows nf..2*nf.
    pub fn swiglu_f32_off_on_gpu(&self, buf: *mut std::ffi::c_void, n: usize, off: usize) {
        let stream = self.stream();
        unsafe {
            launch_swiglu_f32_off(buf as *mut f32, n as i32, off as i32, stream);
        }
    }

    /// D3-5 1a: decode fused rms_norm + pad40 q8 epilogue — the f32 y write
    /// is bit-identical to [`Self::rms_norm`] (same kernel body) and the q8
    /// bytes are bit-identical to `quantize_q8_0_pad40` (epilogue = the
    /// standalone per-block body verbatim). Records the plane for the
    /// following decode matmul group (decode_quantize_native). Callers gate
    /// on the decode shape (n == 1, d % 32 == 0) and MINFER_NO_DECODE_A_FUSE.
    pub fn rms_norm_quant_on_gpu(
        &self,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) {
        let q8 = Self::get_or_grow(&self.buf_q8_decode, n * (d / 32) * 40);
        let stream = self.stream();
        unsafe {
            launch_rms_norm_quant_pad40(
                x as *const f32,
                w as *const f32,
                y as *mut f32,
                q8 as *mut u8,
                d as i32,
                eps,
                n as i32,
                stream,
            );
        }
        self.record_mmq_cache_native(y as usize, n, d, q8 as usize);
    }

    /// D3-5 1a: decode fused swiglu + pad40 q8 epilogue — the in-place swiglu
    /// write is bit-identical to [`Self::swiglu_f32_off_on_gpu`] and the q8
    /// bytes to `quantize_q8_0_pad40`. n % 32 == 0 is the caller's gate
    /// (whole 32-element blocks; the fused-FFN intermediate is always a
    /// multiple of 32 on the supported models).
    pub fn swiglu_quant_off_on_gpu(&self, buf: *mut std::ffi::c_void, n: usize, off: usize) {
        let q8 = Self::get_or_grow(&self.buf_q8_decode, (n / 32) * 40);
        let stream = self.stream();
        unsafe {
            launch_swiglu_quant_pad40(buf as *mut f32, q8 as *mut u8, n as i32, off as i32, stream);
        }
        self.record_mmq_cache_native(buf as usize, 1, n, q8 as usize);
    }

    /// 7e③: generic f32 row gather on device (`get_rows`: out[t*n+i] =
    /// src[ids[t]*n+i]; ids are I32-as-f32 bit patterns on device).
    pub fn gather_rows_f32_on_gpu(
        &self,
        src: *mut std::ffi::c_void,
        ids: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_gather_rows_f32(
                src as *const f32,
                ids as *const f32,
                out as *mut f32,
                n as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// 7e③: embedding gather + dequantize on device. `padded_q6k` selects the
    /// padded (224-byte) block stride for Q6_K weights registered via
    /// `register_weight_q6k_padded`.
    pub fn embed_rows_on_gpu(
        &self,
        ttype: TensorType,
        wptr: *mut std::ffi::c_void,
        ids: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n_embd: usize,
        nt: usize,
        padded_q6k: bool,
    ) -> Result<(), String> {
        let stream = self.stream();
        let (type_id, block_stride) = match ttype {
            TensorType::Q8_0 => (0i32, 34i32),
            TensorType::Q4_0 => (1, 18),
            TensorType::Q4_1 => (7, 20),
            TensorType::Q4_K => (2, 144),
            TensorType::Q5_0 => (6, 22),
            TensorType::Q5_1 => (4, 24),
            TensorType::Q5_K => (5, 176),
            TensorType::Q6_K => (3, if padded_q6k { 224 } else { 210 }),
            TensorType::F32 => {
                // f32 tok_embd is a plain gather of weight rows
                self.gather_rows_f32_on_gpu(wptr, ids, out, n_embd, nt);
                return Ok(());
            }
            // #141: f16 tok_embd rows gather + convert in one kernel; without
            // this a converted f16 GGUF would fail the all-or-nothing
            // `weights_on_cuda` embed check and run the whole model on the CPU.
            TensorType::F16 => {
                let rc = unsafe {
                    launch_embed_rows_f16(
                        wptr as *const u8,
                        ids as *const f32,
                        out as *mut f32,
                        n_embd as i32,
                        nt as i32,
                        stream,
                    )
                };
                if rc != 0 {
                    return Err(format!(
                        "cuda: f16 embed gather launch failed for n_embd={n_embd} nt={nt} \
                         (site launch:embed_rows_f16)"
                    ));
                }
                return Ok(());
            }
            other => {
                return Err(format!(
                    "cuda: no embed_rows kernel for weight type {other:?}"
                ));
            }
        };
        unsafe {
            launch_embed_rows(
                wptr as *const u8,
                ids as *const f32,
                out as *mut f32,
                n_embd as i32,
                nt as i32,
                type_id,
                block_stride,
                stream,
            );
        }
        Ok(())
    }

    pub fn rms_norm(
        &self,
        x: *mut std::ffi::c_void,
        w: Option<*mut std::ffi::c_void>,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) {
        let wptr =
            w.expect("CUDA rms_norm: weight required (no-weights variant not yet implemented)");
        let stream = self.stream();
        unsafe {
            launch_rms_norm_f32(
                x as *const f32,
                wptr as *const f32,
                y as *mut f32,
                d as i32,
                eps,
                n as i32,
                stream,
            );
        }
    }

    pub fn add_f32(
        &self,
        x: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        z: *mut std::ffi::c_void,
        n: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_add_f32(
                x as *const f32,
                y as *const f32,
                z as *mut f32,
                n as i32,
                stream,
            );
        }
    }

    /// Add a per-row bias to a token-major `[rows][d]` buffer: `y[t][i] += b[i]`.
    /// `rows` is the ROW COUNT (token count) — the kernel grid maps one block
    /// row per token, so passing the total element count writes out of bounds.
    pub fn add_bias_f32(
        &self,
        y: *mut std::ffi::c_void,
        b: *mut std::ffi::c_void,
        d: usize,
        rows: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_add_bias_f32(
                y as *mut f32,
                b as *const f32,
                d as i32,
                rows as i32,
                stream,
            );
        }
    }

    pub fn mul_f32(
        &self,
        x: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        z: *mut std::ffi::c_void,
        n: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_mul_f32(
                x as *const f32,
                y as *const f32,
                z as *mut f32,
                n as i32,
                stream,
            );
        }
    }

    pub fn silu_f32(&self, y: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_silu_f32(y as *mut f32, n as i32, stream);
        }
    }

    pub fn swiglu_f32(
        &self,
        gate: *mut std::ffi::c_void,
        up: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        n: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_swiglu_f32(
                gate as *const f32,
                up as *const f32,
                dst as *mut f32,
                n as i32,
                stream,
            );
        }
    }

    pub fn rope_f32(
        &self,
        x: *mut std::ffi::c_void,
        n_head: usize,
        n_dims: usize,
        nt: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_rope_f32(
                x as *mut f32,
                n_head as i32,
                n_dims as i32,
                nt as i32,
                freq_base,
                freq_scale,
                positions as *const i32,
                stream,
            );
        }
    }

    /// C4 S2b: the general nt > 1 attention kernel, layout-tagged. `row_bytes` is
    /// the KV cell's byte width (`nk * hd * {4,2}` for f32/f16,
    /// `KvFormat::Q8_0.row_bytes(nkt)` for a packed region).
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attn_f32(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        layout: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        row_bytes: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_gqa_attn_f32(
                q as *const f32,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                o as *mut f32,
                positions as *const i32,
                mode,
                layout,
                nh as i32,
                nk as i32,
                hd as i32,
                scale,
                row_bytes,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 94: batched split attention for the verify shapes (1 < nt <= 16).
    /// Per-token nkv = positions[t]+1 with the decode path's exact
    /// attn_split_1w_body arithmetic and combine merge order, so a verify
    /// batch's logits are bitwise-equal to running the nt=1 decode path at
    /// each position (the greedy identity). The partials scratch grows to
    /// nt * SPLITS * nh * pstr — sized at warmup for the verify nt, stable
    /// within a capture window.
    pub fn gqa_attn_split_batched(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        f16_kv: bool,
        nt: usize,
    ) {
        let pstr = ((4 + hd + 3) & !3) as i32;
        const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
        let need = nt * ATTN_SPLITS * nh * (pstr as usize) * 4;
        let partial = Self::get_or_grow(&self.buf_attn_partial, need);
        let stream = self.stream();
        unsafe {
            if f16_kv {
                launch_gqa_attn_split_batched_f16kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    nt as i32,
                    stream,
                );
            } else {
                launch_gqa_attn_split_batched_f32kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// 8b: GQA attention over a **staged** KV cache (K/V materialized into the
    /// FA f16 tile). `layout` is the KV tag: `KV_LAYOUT_F16` reads halves and
    /// `KV_LAYOUT_Q8_0` dequantizes each packed cell while staging (#144 item 3);
    /// q/o stay f32 in both. Matches Metal's pl_gqa_attn_f16 precision class
    /// (f16 storage, f32 accumulate). `row_bytes` is the packed cell's byte
    /// width and is ignored by the f16 arm.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attn_kv_prefill(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        layout: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        row_bytes: usize,
        nt: usize,
    ) {
        // #189's mutation seam: 1 unless `MINFER_S4_AB_MAP_REPS=2` doubles the
        // map arm's work (`s4_ab_map_reps`).
        for _ in 0..s4_ab_map_reps(mode) {
            self.gqa_attn_kv_prefill_once(
                q, k, v, o, positions, mode, layout, nh, nk, hd, scale, row_bytes, nt,
            );
        }
        // The observation half of gate contract rule 3: the map instantiation
        // really ran. The S4 A/B resets this counter, runs one map call and one
        // span call, and asserts the map call moved it while the span did not.
        if mode == AttnWindow::Map.code() {
            crate::testfail::note_checked("cuda_attn_map_window");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_attn_kv_prefill_once(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        layout: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        row_bytes: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        // 8n: prefill (nt >= 64) runs the FA-style tiled attention. The
        // legacy kernel is one block per (token, head) — K re-read per token
        // per head (7B @2K: ~132 GB/layer) with a 128-register accumulator —
        // and measured 176 ms/layer, 76% of the whole 2K prefill. hd % 16
        // is hard-wired (FA_HQ = hd/4 = 32) and the shared-memory opt-in
        // can fail on constrained devices, hence the rc fallback.
        //
        // D5-R stage 4 (doc 86): the verify shapes (nt = d+1, i.e. 2..=9)
        // used to fall through to the legacy per-(token,head) kernel — the
        // doc 85 ledger prices that hole at ~9 ms of the 17.6 ms nt=3
        // marginal (the nt=1 split-KV path does the same KV read in 0.8 ms).
        // fa_prefill masks rows causally from the positions array ("positions
        // are data"), which is exactly the verify block's structure, so the
        // gate is lowered to nt >= 2; nt == 1 keeps the split-KV decode path.
        //
        // C8b S4: a `kv_map` window is gathered here too — the staging resolves
        // each linear window index through the runs, and the tile's per-row mask is
        // that query's row count (a map window is a prefix of the sequence's address
        // space, so the existing `gcol < limit` form stays exact).
        //
        // #144 item 3: the same FA body serves a packed cache — the staging
        // dequantizes each Q8_0 cell into the f16 tile instead of copying halves,
        // so the tensor-core QK^T/P·V stream is unchanged. The general
        // layout-tagged kernel remains the documented fallback (the `rc == -1`
        // arm below), and the packed arm reaches it with the same rounded
        // window modes.
        if nt >= 2 && hd == 128 && !Self::no_fa_prefill() && layout != crate::cuda::KV_LAYOUT_F32 {
            let rc = unsafe {
                launch_fa_prefill_kv(
                    q as *const f32,
                    k as *const std::ffi::c_void,
                    v as *const std::ffi::c_void,
                    o as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    nt as i32,
                    layout,
                    row_bytes,
                    stream,
                )
            };
            if rc == 0 {
                // #144 item 3: the observation half (gate contract rule 3) — a gate
                // that must prove the packed FA path *ran* cannot read the dispatch's
                // own answer, so the chokepoint records it and
                // `cuda_q8_0_fa_prefill_attention_parity` asserts the counter moved.
                if layout == crate::cuda::KV_LAYOUT_Q8_0 {
                    crate::testfail::note_checked("cuda_fa_prefill_q8_0");
                }
                return;
            }
        }
        if layout == crate::cuda::KV_LAYOUT_Q8_0 {
            // #144: the packed fallback is the general layout-tagged kernel, not
            // the f16-typed one — `launch_gqa_attn_f32_f16kv` would read the packed
            // bytes as halves.
            self.gqa_attn_f32(
                q, k, v, o, positions, mode, layout, nh, nk, hd, scale, row_bytes, nt,
            );
            return;
        }
        unsafe {
            launch_gqa_attn_f32_f16kv(
                q as *const f32,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                o as *mut f32,
                positions as *const i32,
                mode,
                nh as i32,
                nk as i32,
                hd as i32,
                scale,
                nt as i32,
                stream,
            );
        }
    }

    /// 8d: split-K decode attention (nt == 1). `pstr` = (4 + hd + 3) & !3 —
    /// the partials' row stride keeps the oc section 16-byte aligned. The
    /// scratch must be at least ATTN_SPLITS * nh * pstr floats (see
    /// buf_attn_partial). ATTN_SPLITS must mirror the `ATTN_SPLITS` define
    /// in cuda_kernels.cu (fixed grid — the graph-replay capture depends on
    /// it; idle splits write an mx=-INF/S=0 partial the combine weights to
    /// zero).
    /// C3: move `rows` rows of `elems` f32 elements inside one KV arena, from
    /// `src_row` down to `dst_row` (`dst_row <= src_row`; overlapping is fine).
    ///
    /// The kernel walks rows ascending with a barrier between them; a contract
    /// violation is an `Err` here rather than a silent no-op, so the allocator's
    /// compaction fails before it renumbers anything.
    pub fn kv_move_rows(
        &self,
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        dst_row: usize,
        src_row: usize,
        rows: usize,
        elems: usize,
    ) -> Result<(), String> {
        if rows == 0 || elems == 0 {
            return Ok(());
        }
        let rc = unsafe {
            launch_kv_move_rows(
                dst as *mut f32,
                src as *const f32,
                dst_row as i32,
                src_row as i32,
                rows as i32,
                elems as i32,
                self.stream(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "cuda: kv_move_rows({dst_row}<-{src_row}, {rows} rows x {elems} elements) failed \
                 (rc {rc})"
            ));
        }
        Ok(())
    }

    /// C4 S2b: the decode (nt == 1) split-K path, layout-tagged. A packed cache
    /// takes `launch_gqa_attn_split_q8_0`, whose `rpw_gate = 0` skips the hybrid
    /// 4-warp body (f16-typed); f32/f16 keep their pre-C4 launchers unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attn_split(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        layout: i32,
        row_bytes: usize,
    ) {
        // #189's mutation seam: 1 unless `MINFER_S4_AB_MAP_REPS=2` doubles the
        // map arm's work (`s4_ab_map_reps`). The launch is idempotent, so the
        // second issue writes the same bytes and only the cost changes.
        for _ in 0..s4_ab_map_reps(mode) {
            self.gqa_attn_split_once(
                q, k, v, o, positions, mode, nh, nk, hd, scale, layout, row_bytes,
            );
        }
        // The observation half of gate contract rule 3 (see `gqa_attn_kv_prefill`).
        if mode == AttnWindow::Map.code() {
            crate::testfail::note_checked("cuda_attn_map_window");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_attn_split_once(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        layout: i32,
        row_bytes: usize,
    ) {
        let pstr = ((4 + hd + 3) & !3) as i32;
        const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
        let need = ATTN_SPLITS * nh * (pstr as usize) * 4;
        let partial = Self::get_or_grow(&self.buf_attn_partial, need);
        let stream = self.stream();
        // #186: the Q8_0 decode arm's same-binary A/B control. The flag is resolved
        // once per process (`cuda::q8_kv_dp4a_enabled`) and passed as a value; the
        // launcher picks the `__dp4a` or the convert-based instantiation from it.
        // #202 adds a second, independent arm (`wide`): the K/V four-quant groups
        // are loaded with two 16-bit loads instead of four byte loads. It is only
        // meaningful on the dp4a arm, so `dp4a` already gates it in the launcher.
        let dp4a = q8_kv_dp4a_enabled();
        let wide = q8_kv_wide_enabled();
        unsafe {
            if layout == KV_LAYOUT_Q8_0 {
                launch_gqa_attn_split_q8_0(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    row_bytes,
                    dp4a as i32,
                    wide as i32,
                    stream,
                );
                // The observation half of gate contract rule 3: a gate that must
                // prove the int dot ran cannot read the launch's own report. The
                // chokepoint is bumped only when this call launched the dp4a arm.
                if dp4a {
                    crate::testfail::note_checked("cuda_q8_kv_dp4a");
                }
            } else if layout == KV_LAYOUT_F16 {
                launch_gqa_attn_split_f16kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    stream,
                );
            } else {
                launch_gqa_attn_split_f32kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    stream,
                );
            }
        }
    }

    /// 8e follow-up: `MINFER_NO_KQ_MMVQ=1` forces the K-quant decode matmuls
    /// back onto the f32 kernels (A/B escape hatch, MINFER_NO_FUSE_* style).
    fn no_kq_mmvq() -> bool {
        std::env::var("MINFER_NO_KQ_MMVQ").map_or(false, |v| v == "1")
    }
    // doc 103: per-type opt-outs for the new q4_0/q8_0 MMVQ decode paths
    // (A/B switches; the f32-activation kernels remain the fallback).
    fn no_q40_mmvq() -> bool {
        std::env::var("MINFER_NO_Q40_MMVQ").map_or(false, |v| v == "1")
    }
    fn no_q80_mmvq() -> bool {
        std::env::var("MINFER_NO_Q80_MMVQ").map_or(false, |v| v == "1")
    }
    fn no_q80_p32() -> bool {
        std::env::var("MINFER_NO_Q80_P32").map_or(false, |v| v == "1")
    }
    /// doc 104: register a Q8_0 tensor ALSO as the p32 split-plane layout —
    /// payload plane (32 B/block, 16B-aligned for uint4 loads) + dense d
    /// plane (2 B/block). The raw registration stays untouched (the f32
    /// fallback and every existing kernel keep reading it); the planes are
    /// extra device memory (~+94% of the tensor) keyed by the original
    /// weight pointer for the decode dispatch. Host repack at load, like
    /// the q6_K padded precedent — no capture-window hazard. Registered
    /// under private keys (`\u{1}`-prefixed suffixes) so they can never
    /// collide with a real tensor name (the doc 102 lesson).
    /// T2 (plan §6.3): free-VRAM budget gate for OPTIONAL weight planes
    /// (q8_0 p32 pairs, q4_K/q6_K dsc + dense expansions). Requires free >
    /// extra + extra/4 — the headroom covers the activation/KV working set
    /// that lands after registration. On GB10's 128 GB this always passes
    /// (zero change); on 8 GB unified-memory devices (Orin Nano) it
    /// self-disables the planes and the raw paths serve (raw lookup miss).
    /// Best-effort by design — every consumer falls back to its raw path on a
    /// map miss — so a failed query keeps the planes **off** (the conservative
    /// branch, unchanged) but no longer silently (issue #122): the discarded
    /// return code used to make a broken query look like "not enough memory".
    fn plane_budget_ok(&self, extra_bytes: usize) -> bool {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if rc != 0 {
            static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if WARNED.set(()).is_ok() {
                eprintln!(
                    "CUDA: optional weight-plane budget query failed with {} (code {}); \
                     keeping the optional planes off",
                    cuda_error_name(rc),
                    rc
                );
            }
            return false;
        }
        free > extra_bytes + extra_bytes / 4
    }

    pub fn register_weight_q80_p32(&self, name: &str, data: &[u8], od: usize, id: usize) {
        if od == 0
            || id == 0
            || id % 32 != 0
            || id < 2048
            || std::env::var("MINFER_NO_Q80_P32").map_or(false, |v| v == "1")
        {
            return;
        }
        let nb = id / 32;
        if data.len() < od * nb * 34 {
            return;
        }
        // T2: the p32 pair adds od*nb*34 B (pp 32 + pd 2) = +100% of the raw
        // weight — budget-gate before building/uploading (plan §6.3).
        if !self.plane_budget_ok(od * nb * 34) {
            return;
        }
        let mut pp = vec![0u8; od * nb * 32];
        let mut pd = vec![0u8; od * nb * 2];
        for r in 0..od {
            let row = r * nb;
            for b in 0..nb {
                let src = (row + b) * 34;
                pp[(row + b) * 32..(row + b) * 32 + 32].copy_from_slice(&data[src + 2..src + 34]);
                pd[(row + b) * 2..(row + b) * 2 + 2].copy_from_slice(&data[src..src + 2]);
            }
        }
        let orig = {
            let w = self.weights.lock().unwrap();
            match w.get(name) {
                Some((p, sz)) if *sz == data.len() => p.0 as usize,
                _ => return,
            }
        };
        let pname = format!("{name}\u{1}p32");
        let dname = format!("{name}\u{1}p32d");
        self.register_weight(&pname, &pp);
        self.register_weight(&dname, &pd);
        let w = self.weights.lock().unwrap();
        if let (Some((ppp, _)), Some((pdp, _))) = (w.get(&pname), w.get(&dname)) {
            self.q80_p32
                .lock()
                .unwrap()
                .insert(orig, (ppp.0 as usize, pdp.0 as usize));
        }
    }

    /// doc 104: p32 plane pair for a registered q8_0 weight, if built.
    pub fn q80_p32_planes(&self, wptr: usize) -> Option<(*const u8, *const u8)> {
        let map = self.q80_p32.lock().unwrap();
        map.get(&wptr)
            .map(|(pp, pd)| (*pp as *const u8, *pd as *const u8))
    }

    /// doc 103: decode (nt == 1) q4_0 matmul via the MMVQ structure — dp4a
    /// over the shared pad40 q8 activation plane, one row per 256-thread
    /// block, nibble offset -8 folded into the dot via the 8*sx correction.
    pub fn q4_0_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_mmvq(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 103: multi-token (nt in [2, 8]) q4_0 MMVQ — same in-block token
    /// loop as the K-quant multi variants; bitwise-consistent with the
    /// nt == 1 kernel (same per-u order and reduction).
    pub fn q4_0_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_mmvq_multi(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 103: decode (nt == 1) q8_0 matmul via the MMVQ structure — the
    /// 34-B block stride is only 2B-aligned, so the payload reads use the
    /// q6_K two-u16-halves pattern.
    pub fn q8_0_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_q8_mmvq(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 103: multi-token (nt in [2, 8]) q8_0 MMVQ.
    pub fn q8_0_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_q8_mmvq_multi(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 104: decode q8_0 matmul over the p32 split planes — byte-equal
    /// arithmetic to `q8_0_decode_mmvq`, wider weight loads (uint4 x2 per
    /// 32-element unit instead of 16 scattered u16s).
    pub fn q8_0_p32_decode_mmvq(
        &self,
        pp: *const u8,
        pd: *const u8,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_p32_q8_mmvq(
                pp,
                pd,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 104: multi-token p32 variant (weight words hoisted out of the
    /// token loop; bitwise-consistent with the nt == 1 p32 kernel).
    pub fn q8_0_p32_decode_mmvq_multi(
        &self,
        pp: *const u8,
        pd: *const u8,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_p32_q8_mmvq_multi(
                pp,
                pd,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// 8m: force the legacy per-type prefill kernels (A/B escape hatch).
    // 8p: opt-in dequant-in-GEMM (memory-lean alternative to the f16 cache).
    fn fused_b_on() -> bool {
        std::env::var("MINFER_FUSED_B").map_or(false, |v| v == "1")
    }

    // 8p: disable the persistent per-weight f16 dequant cache.
    fn no_w16cache() -> bool {
        std::env::var("MINFER_NO_W16CACHE").map_or(false, |v| v == "1")
    }

    fn no_prefill_gemm() -> bool {
        std::env::var("MINFER_NO_PREFILL_GEMM").map_or(false, |v| v == "1")
    }

    /// 8n: force the legacy per-token prefill attention kernel (A/B escape).
    fn no_fa_prefill() -> bool {
        std::env::var("MINFER_NO_FA_PREFILL").map_or(false, |v| v == "1")
    }

    /// 8e-reversal: decode (nt == 1) q4_K matmul via the llama.cpp MMVQ
    /// structure — quantize the activation row to padded 40B q8 blocks in
    /// `buf_q8_decode`, then run the dp4a one-row-per-block kernel. The
    /// caller gates on id % 32 == 0 (sub-block granularity) and id >= 2048
    /// (below that the structure win is launch-latency noise).
    pub fn q4_k_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        // D3-5 1a: consult the MmqCache first — the fused decode producers
        // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
        // pad40 plane, so the matmul group following the producer skips the
        // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
        // restores the unconditional standalone launch).
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q4_k_q8_mmvq_v2(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q4_k_q8_mmvq(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// 8e follow-up: decode (nt == 1) q6_K matmul via the same MMVQ
    /// structure — the shared q8 activation scratch, then the 16-element
    /// unit dp4a kernel. `blk_stride` follows the weight registration:
    /// 224 for the padded 7e② repack, 210 for raw bytes.
    pub fn q6_k_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        blk_stride_padded: bool,
    ) {
        // D3-5 1a: consult the MmqCache first — the fused decode producers
        // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
        // pad40 plane, so the matmul group following the producer skips the
        // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
        // restores the unconditional standalone launch).
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        // D4-4 L1: dense split-plane fast path (bitwise — see the
        // registration comment). Falls through to the padded kernels when
        // the plane is absent (MINFER_Q6K_DPL=0 / id not a multiple of
        // 256 / map miss). The pf-vs-loop shape gate mirrors the padded
        // dispatch below (same MINFER_Q6K_PF opt-out semantics).
        if blk_stride_padded && Self::mmq_gate_on("MINFER_Q6K_DPL") {
            let dwp = self
                .q6k_dpl
                .lock()
                .unwrap()
                .get(&(wptr as usize))
                .map(|cp| cp.0);
            if let Some(dwp) = dwp {
                unsafe {
                    let nbe = (id >> 8) as i32;
                    if id > 8192
                        && id <= 16384
                        && !std::env::var("MINFER_Q6K_PF").map_or(false, |v| v == "0")
                    {
                        launch_q6_k_q8_mmvq_v2_pf_dpl(
                            dwp as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            od as i32,
                            id as i32,
                            nt as i32,
                            nbe,
                            stream,
                        );
                    } else {
                        launch_q6_k_q8_mmvq_v2_dpl(
                            dwp as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            od as i32,
                            id as i32,
                            nt as i32,
                            nbe,
                            stream,
                        );
                    }
                }
                return;
            }
        }
        unsafe {
            if Self::mmvq_v2(id) && blk_stride_padded {
                // v2's uint4 ql/qh loads need the padded 224B stride
                // D4-2 B0 correctness fix: v2_pf processes exactly TWO units
                // per thread (u = tid, tid+256 → npair ≤ 512), but the old
                // gate (id > 8192) had no upper bound — npair=592 shapes
                // (7B ffn_down id 18944, 10 q6_K layers) silently dropped
                // units 512..591, corrupting 7B decode since D3b-1b. Guard
                // the upper bound; taller rows take the v2 loop form, which
                // walks any npair with identical per-unit arithmetic and the
                // same ascending-u accumulation order (bitwise for every
                // npair ≤ 512 shape, which keep the pipelined kernel).
                // D4-2 B1c: MINFER_Q6K_PF=0 A/Bs the pipelined form against
                // the v2 loop at these shapes (both cover the full unit set).
                if id > 8192
                    && id <= 16384
                    && !std::env::var("MINFER_Q6K_PF").map_or(false, |v| v == "0")
                {
                    // D3b-1b: tall rows (npair > 256, e.g. ffn_down id 13824)
                    // run the pipelined variant (bitwise-identical, loads for
                    // both serial units issue up front).
                    launch_q6_k_q8_mmvq_v2_pf(
                        wptr as *const u8,
                        q8 as *const u8,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        224,
                        stream,
                    );
                } else {
                    launch_q6_k_q8_mmvq_v2(
                        wptr as *const u8,
                        q8 as *const u8,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        224,
                        stream,
                    );
                }
            } else {
                launch_q6_k_q8_mmvq(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    if blk_stride_padded { 224 } else { 210 },
                    stream,
                );
            }
        }
    }

    /// Step 82: multi-token (nt in 2..=8) q4_K matmul via the MMVQ
    /// structure with an in-block token loop — the weight stream is paid
    /// once regardless of nt (the doc 81 D5-1a dispatch hole). The
    /// nt == 1 id >= 2048 shape gate does not apply: at nt >= 2 a
    /// weights-once kernel wins at any shape.
    pub fn q4_k_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        // decode_quantize_native is nt-parameterized (the pad40 q8 plane
        // covers all nt rows), so no scratch change is needed here.
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q4_k_q8_mmvq_v2_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q4_k_q8_mmvq_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// Step 82: multi-token (nt in 2..=8) q5_K matmul — the q4_K multi
    /// structure with the q5 high-bit plane folded in.
    pub fn q5_k_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q5_k_q8_mmvq_v2_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q5_k_q8_mmvq_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// Step 82: multi-token (nt in 2..=8) q6_K matmul — 16-element units
    /// over q8 activations with an in-block token loop. The v2 form needs
    /// the padded 224B block stride (u32/uint4 loads); the v1 form handles
    /// the raw 210B stride too (2-byte loads).
    pub fn q6_k_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        blk_stride_padded: bool,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        let stride: i32 = if blk_stride_padded { 224 } else { 210 };
        unsafe {
            if blk_stride_padded && Self::mmvq_v2(id) {
                launch_q6_k_q8_mmvq_v2_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stride,
                    stream,
                );
            } else {
                launch_q6_k_q8_mmvq_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stride,
                    stream,
                );
            }
        }
    }

    /// 8e follow-up: decode (nt == 1) q5_K matmul via the same MMVQ
    /// structure (q4_K shape with the q5 high-bit plane folded in).
    pub fn q5_k_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        // D3-5 1a: consult the MmqCache first — the fused decode producers
        // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
        // pad40 plane, so the matmul group following the producer skips the
        // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
        // restores the unconditional standalone launch).
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q5_k_q8_mmvq_v2(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q5_k_q8_mmvq(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// R2: the v2 weight-streaming MMVQ kernels need full 256-element
    /// super-blocks (chunk/pair indexing) — `MINFER_MMVQ_V1=1` forces the
    /// 8e kernels for A/B on shapes that would otherwise take v2.
    fn mmvq_v2(id: usize) -> bool {
        id % 256 == 0 && !std::env::var("MINFER_MMVQ_V1").map_or(false, |v| v == "1")
    }

    /// Decode I32 graph inputs (f32::from_bits bit patterns, alloc.rs
    /// fill_input_i32) into raw int32 for the rope/store/attention kernels.
    /// Fully device-side: no host sync, capture-safe (Phase 7d).
    pub fn bits_to_i32(&self, src: *mut std::ffi::c_void, dst: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_f32_bits_to_i32(src as *const f32, dst as *mut i32, n as i32, stream);
        }
    }

    /// 8b: f16 KV cache — see `kv_cache_is_f16`. Same trade-off as Metal:
    /// halves attention KV read bandwidth; the region stays f32-sized (the
    /// f16 view uses the first half of the bytes).
    pub fn store_kv_f16(
        &self,
        src: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        nkt: usize,
        nt: usize,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_f16(
                src as *const f32,
                dst as *mut std::ffi::c_void,
                nkt as i32,
                nt as i32,
                positions as *const i32,
                stream,
            );
        }
    }

    /// C4 S2b: the packed store — one thread per (row, 32-element block), with the
    /// CPU's own quantizer (`amax/127`, f16 scale, round-ties-even). `row_bytes` is
    /// the packed cell's byte width (`KvFormat::Q8_0.row_bytes(nkt)`), which is what
    /// makes the kernel address the cell's 34-byte blocks inside a word-padded row.
    pub fn store_kv_q8_0(
        &self,
        src: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        nkt: usize,
        nt: usize,
        row_bytes: usize,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_q8_0(
                src as *const f32,
                dst,
                nkt as i32,
                nt as i32,
                row_bytes,
                positions as *const i32,
                stream,
            );
        }
    }

    /// D3-8: fused decode QKV epilogue (G4 CUDA port of Metal's
    /// `attn_bias_rope_store`). `positions` drives RoPE (sequence-relative);
    /// `cells` is the allocator-resolved KV row for the store (C6) — the two
    /// differ when the run does not start at cell 0.
    ///
    /// `q`/`k`/`v` are POINTER-FORM section bases:
    /// the concat class passes sections of the concat matmul output [q|k|v]
    /// (nt==1), the mixed-quant class passes the three separate matmul
    /// outputs. Biases added per section, q/k roped in place (math verbatim
    /// `rope_f32`), k/v stored into the persistent regions at the same
    /// addresses as `store_kv_f32`/`store_kv_f16` (f32 or f16 per `layout`).
    ///
    /// C4 S2b: **f32/f16 only**, and a packed layout is refused here rather than
    /// silently stored per element. The fused epilogue writes one K/V element at a
    /// time — a Q8_0 block's scale needs all 32 of its elements before any of them
    /// can be quantized — so a packed cache never builds this node (the model
    /// builders' `layer_gpu` gate gains `&& !packed`) and a Q8_0 decode runs the
    /// unfused bias/rope/store chain through [`Self::store_kv_q8_0`]. Reaching this
    /// function with `KV_LAYOUT_Q8_0` is a builder bug, so it is an `Err`-shaped
    /// refusal in the caller's terms: the function is infallible in the pre-C4
    /// signature, so it panics with the reason instead of writing the wrong bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_bias_rope_store(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        bias_q: *mut std::ffi::c_void,
        bias_k: *mut std::ffi::c_void,
        bias_v: *mut std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
        cells: *mut std::ffi::c_void,
        layout: i32,
    ) {
        assert!(
            layout != KV_LAYOUT_Q8_0,
            "cuda: the f32/f16 fused bias/rope/store epilogue has no packed store (it writes one \
             element at a time; a Q8_0 block needs all 32) — a packed engine must call \
             attn_bias_rope_store_q8_0 instead (issue #144)"
        );
        let stream = self.stream();
        unsafe {
            launch_attn_bias_rope_store(
                q as *mut f32,
                k as *mut f32,
                v as *mut f32,
                bias_q as *const std::ffi::c_void,
                bias_k as *const std::ffi::c_void,
                bias_v as *const std::ffi::c_void,
                kv_k,
                kv_v,
                nqt as i32,
                nkt as i32,
                hd as i32,
                freq_base,
                freq_scale,
                positions as *const i32,
                cells as *const i32,
                (layout == KV_LAYOUT_F16) as i32,
                stream,
            );
        }
    }

    /// #144 item 1: the **packed** arm of the fused decode QKV epilogue. Same
    /// bias+rope contract as [`Self::attn_bias_rope_store`], but K and V are
    /// written as whole Q8_0 blocks: one thread per (head, 32-element K block)
    /// computes the block's roped values itself and hands them to the store's own
    /// quantizer, and one thread per V block does bias + quantize. The K buffer is
    /// left roped-free (its readers are gone in both fused classes) and the bytes
    /// written to the packed regions are `add_bias`+`rope`+`store_kv_q8_0`'s
    /// verbatim.
    ///
    /// `row_bytes` is `KvFormat::Q8_0.row_bytes(nkt)`, the same byte width
    /// [`Self::store_kv_q8_0`] and `ensure_kv` use.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_bias_rope_store_q8_0(
        &self,
        q: *mut std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        bias_q: *mut std::ffi::c_void,
        bias_k: *mut std::ffi::c_void,
        bias_v: *mut std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
        cells: *mut std::ffi::c_void,
        row_bytes: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_attn_bias_rope_store_q8_0(
                q as *mut f32,
                k as *const f32,
                v as *const f32,
                bias_q as *const std::ffi::c_void,
                bias_k as *const std::ffi::c_void,
                bias_v as *const std::ffi::c_void,
                kv_k,
                kv_v,
                nqt as i32,
                nkt as i32,
                hd as i32,
                freq_base,
                freq_scale,
                positions as *const i32,
                cells as *const i32,
                row_bytes,
                stream,
            );
        }
    }

    pub fn store_kv_f32(
        &self,
        src: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        nkt: usize,
        nt: usize,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_f32(
                src as *const f32,
                dst as *mut f32,
                nkt as i32,
                nt as i32,
                positions as *const i32,
                stream,
            );
        }
    }
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
// - `graph::cuda_backend::tests::
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
