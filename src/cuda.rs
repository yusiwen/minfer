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
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
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
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    // F5 (#58): events, the synchronization primitive the split boundary's async
    // staging copies need. `cudaEventRecord` marks a point on the stream;
    // `cudaStreamWaitEvent` makes a later consumer wait on it **without blocking
    // the host**; `cudaEventSynchronize` is the host-side wait and is the one
    // documented synchronization point of a device→host staging copy.
    fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    fn cudaEventRecord(event: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
    fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
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
    fn cudaGetErrorString(error: i32) -> *const std::os::raw::c_char;
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
fn cuda_error_name(code: i32) -> &'static str {
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

/// The human-readable description of a CUDA error code (`cudaGetErrorString`).
#[allow(dead_code)] // for diagnostics that want the prose form
fn cuda_error_string(code: i32) -> &'static str {
    let p = unsafe { cudaGetErrorString(code) };
    if p.is_null() {
        return "";
    }
    let s = unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy();
    match s {
        std::borrow::Cow::Borrowed(b) => b,
        std::borrow::Cow::Owned(o) => Box::leak(o.into_boxed_str()),
    }
}

// Legacy layer_gpu debug tracing (7e⑦): the graph path syncs via
// `CudaState::sync()`; MINFER_CUDA_DEBUG tracing stays for the legacy
// surface only (debug_sync in the impl below).
static CUDA_DEBUG: OnceLock<bool> = OnceLock::new();
fn cuda_debug_enabled() -> bool {
    *CUDA_DEBUG.get_or_init(|| std::env::var("MINFER_CUDA_DEBUG").is_ok())
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
    // P6: A arrives as f32 activations; the GEMM converts on stage — the
    // separate convert_f32_f16 pass disappears for every prefill matmul.
    //
    // #145: returns the number of `cudaFuncSetAttribute` calls that failed (each
    // already named on stderr with `cudaGetErrorName` where it was made). The
    // caller reports the count; the attribute requests that exceed the device's
    // opt-in limit are deliberately skipped with the reason printed.
    fn gemm_prefill_smem_init() -> i32;
    // #145 introspection: the opt-in decision, for the startup report and the
    // `cuda_prefill_smem_optin_*` gate. `gemm_smem_need` is the single-source
    // formula the launcher reads; `gemm_smem_opted_in` reads the device's own
    // `cudaFuncGetAttributes().maxDynamicSharedSizeBytes` back.
    #[allow(dead_code)] // read by the #145 device gate, not by a non-test build
    fn gemm_prefill_smem_checked() -> i32;
    fn gemm_prefill_smem_skipped() -> i32;
    #[allow(dead_code)] // read by the #145 device gate
    fn gemm_prefill_smem_limit() -> i32;
    #[allow(dead_code)] // read by the #145 device gate
    fn gemm_smem_need(tm: i32, ks: i32, af32: i32) -> usize;
    #[allow(dead_code)] // read by the #145 device gate
    fn gemm_smem_opted_in(tm: i32, ks: i32, af32: i32) -> i32;
    // #145 test injection: latch a real `cudaErrorInvalidValue` on purpose
    // (request the device limit + 4096 B) without clearing it. Only the gate
    // calls this.
    #[allow(dead_code)]
    fn cuda_test_latch_oversized_smem() -> i32;
    // #147: the same shape for the af32 path — 1 = launched and accepted.
    fn launch_gemm_f32a(
        a: *const f32,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // #147 site-failure introspection: the last dynamic-smem / launch failure a
    // hardened C++ site named, so the `issue147_tests` device gates can assert
    // the site, the requested value and `cudaGetErrorName` without parsing
    // stderr. `kind`: 1 = attribute, 2 = launch, 3 = a latched error found
    // before a launch.
    #[allow(dead_code)] // read by the #147 device gates
    fn minfer_site_fail_count() -> i32;
    #[allow(dead_code)]
    fn minfer_site_fail_kind() -> i32;
    #[allow(dead_code)]
    fn minfer_site_fail_code() -> i32;
    #[allow(dead_code)]
    fn minfer_site_fail_bytes() -> i32;
    #[allow(dead_code)]
    fn minfer_site_fail_limit() -> i32;
    #[allow(dead_code)]
    fn minfer_site_fail_site() -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_site_fail_message() -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_site_fail_reset();
    // #162: the sticky required-launch failure and the ordered launch-site
    // history. `minfer_launch_ok` sets the sticky for a REQUIRED site;
    // `CudaBackend::execute_node` drains it and returns an `Err` naming the site,
    // so one Rust-side check covers every launcher. `minfer_launch_ok_opt` (a
    // documented fallback) never sets it. The history records every named launch
    // failure so the #162 gate can see more than one site per call.
    #[allow(dead_code)]
    fn minfer_launch_fail_pending() -> i32;
    #[allow(dead_code)]
    fn minfer_launch_fail_site() -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_launch_fail_name() -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_launch_fail_code() -> i32;
    #[allow(dead_code)]
    fn minfer_launch_fail_clear();
    #[allow(dead_code)]
    fn minfer_site_hist_len() -> i32;
    #[allow(dead_code)]
    fn minfer_site_hist_site(i: i32) -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_site_hist_name(i: i32) -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_site_hist_msg(i: i32) -> *const std::os::raw::c_char;
    #[allow(dead_code)]
    fn minfer_site_hist_reset();
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

// ─── CudaState singleton ───────────────────────────────────────

static CUDA: OnceLock<Option<CudaState>> = OnceLock::new();

/// F5 (#58): how many times this process has **blocked the host** on the stream
/// (`CudaState::sync` — the only full `cudaStreamSynchronize` in the device
/// layer). Process-wide and monotonic on purpose: it is the "host stalls"
/// measurement the ticket's before/after is stated in, and a reader compares a
/// delta around one workload rather than an absolute count. A relaxed atomic
/// increment is the whole cost on the hot path.
static STREAM_SYNCS: AtomicU64 = AtomicU64::new(0);

/// F5 (#58): the process-wide stream-synchronization count (see [`STREAM_SYNCS`]).
///
/// **A gate must not read this** (issue #185). It is a lifetime total across every
/// thread, so a delta around one workload also contains whatever any concurrent
/// device test synced — the F5 gate's async arm read 4160 stalls against the
/// synchronous arm's 728 under the parallel harness. Read the backend's own
/// `CudaBackend::stream_sync_count()` instead; the counter lives per instance,
/// like `blocking_readbacks` and like `copystats`' accumulators.
#[allow(dead_code)] // kept as the process-wide total; the F5 gates read the backend's own (#185)
pub fn stream_sync_count() -> u64 {
    STREAM_SYNCS.load(Ordering::Relaxed)
}

/// Issue #145: how many times `CudaState::sync` found an error **already latched**
/// by `cudaGetLastError` (i.e. not caused by the kernel that just ran). The
/// message names the observer, never a launch; this counter is how the gate
/// proves the error was surfaced rather than dropped.
static LATCHED_API_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Issue #145: the process-wide count of latched API errors `sync` has reported.
#[allow(dead_code)] // read by the #145 device gate
pub fn latched_api_error_count() -> u64 {
    LATCHED_API_ERRORS.load(Ordering::Relaxed)
}

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
/// sites, with its exact-token tests in `src/testfail.rs` — so this is now a
/// thin alias that keeps the #147 device gates' call site unchanged.
#[allow(dead_code)] // read by the #147 device gates
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
    /// Lazy pinned staging ring (7e⑥); None until the first async fill,
    /// and stays None if cudaHostAlloc fails (sync fallback).
    staging: Mutex<Option<PinnedPool>>,
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
    /// Test-only readers today (cuda_backend probe tests); the tier selector
    /// consumes the value at init before the field is stored.
    #[allow(dead_code)]
    cc: std::sync::atomic::AtomicI32,
    /// T1: resolved device tier (plan §5) — exact key match → family
    /// inheritance → GENERIC. Resolved once at init (also honors the
    /// MINFER_DEVICE_TIER override); dispatch reads plain fields, never
    /// re-scans the table. Direct consumers arrive with the batch-cap
    /// activation (plan §14 R8); the effective gate travels via `tier_mmq`.
    #[allow(dead_code)]
    tier: &'static device_tier::DeviceTier,
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
    // Persistent activation buffers (grown on demand) with size tracking
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_hidden: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bn: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bq: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bk: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bv: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_ba: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bf: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bg: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_q8_bn: Mutex<(CudaPtr, usize)>,
    /// 8c: prefill Q8_0-activation scratch (quantized activations for the
    /// Q4_0×Q8_0 GEMM, nt > 1). Grown on demand like the layer-path buffers.
    buf_q8_prefill: Mutex<(CudaPtr, usize)>,
    /// P6 r34: transposed-A q8_0 prepass scratch — the swizzled qs plane
    /// ([ntb][nchunk][2048]) and the packed d|ssum scale ([ntb][nchunk][256]),
    /// consumed by mmq_raw_nb_bt_kernel's bulk staging (MINFER_MMQ_A_TRANSPOSE).
    buf_qa8_t: Mutex<(CudaPtr, usize)>,
    buf_sda_t: Mutex<(CudaPtr, usize)>,
    /// doc 92: K-split fp32 partials ([ksplit][nt][od]) for the BT GEMM's
    /// block-starvation fix at small nt. Grown on demand like the other
    /// prepass scratches; the reduce kernel consumes it on the same stream.
    buf_mmq_ksplit: Mutex<(CudaPtr, usize)>,
    /// r49: consecutive-window memoization of the MMQ A-quantize prepass (see
    /// [`MmqCache`]). Lives on the process singleton so the CUDA backend can
    /// invalidate it between non-MMQ nodes / graph executions.
    mmq_cache: Mutex<MmqCache>,
    /// 8d: split-K attention partials ([8][nh][pstr] floats, nh/hd are graph
    /// constants so the size is stable — grown during warmup, never inside a
    /// capture window).
    buf_attn_partial: Mutex<(CudaPtr, usize)>,
    /// 8e-reversal: decode MMVQ q8 activation scratch (nt=1, so id/32 * 40B
    /// per token — size-stable per graph, grown during warmup runs).
    buf_q8_decode: Mutex<(CudaPtr, usize)>,
    /// 8m: prefill f16 GEMM scratch — dequantized weights (od*id halves) and
    /// converted activations (nt*id halves), grown on demand. Prefill never
    /// enters a CUDA Graph capture window (8g①), so the grow is capture-safe
    /// (same assumption as the 8c buf_q8_prefill).
    buf_f16_w: Mutex<(CudaPtr, usize)>,
    buf_f16_x: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_q8_ba: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_positions: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_logits: Mutex<(CudaPtr, usize)>,
    // Persistent per-layer GPU KV cache (k, v) and current size
    kv_k: Mutex<Vec<CudaPtr>>,
    kv_v: Mutex<Vec<CudaPtr>>,
    kv_size: Mutex<Vec<usize>>,
    // CUDA Graph for decode step (capture once, replay for each token)
    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    decode_graph_exec: Mutex<CudaPtr>,
    /// Process-wide stream serialization for the graph-path backend (Phase
    /// 7d): stream capture is per-stream, so while one backend holds an open
    /// capture window, every OTHER backend's stream work (fills, copies,
    /// launches, allocs) must block instead of being recorded into that
    /// graph. The capturing backend holds this lock across its window; its
    /// own enqueues skip re-locking (they are the recorded work).
    stream_lock: Mutex<()>,
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
pub fn format_of(layout: i32) -> crate::graph::kvformat::KvFormat {
    use crate::graph::kvformat::KvFormat;
    match layout {
        KV_LAYOUT_F16 => KvFormat::F16,
        KV_LAYOUT_Q8_0 => KvFormat::Q8_0,
        _ => KvFormat::F32,
    }
}

#[cfg(test)]
mod kv_dtype_tests {
    use super::{format_of, layout_of, KV_LAYOUT_F16, KV_LAYOUT_F32, KV_LAYOUT_Q8_0};
    use crate::graph::kvformat::KvFormat;

    /// The three layouts are a total, one-to-one mapping — the tag a `CudaBackend`
    /// holds and the format its engine resolved cannot disagree.
    #[test]
    fn the_layout_tag_is_the_format_discriminant() {
        assert_eq!(layout_of(KvFormat::F32), KV_LAYOUT_F32);
        assert_eq!(layout_of(KvFormat::F16), KV_LAYOUT_F16);
        assert_eq!(layout_of(KvFormat::Q8_0), KV_LAYOUT_Q8_0);
        for f in [KvFormat::F32, KvFormat::F16, KvFormat::Q8_0] {
            assert_eq!(format_of(layout_of(f)), f, "{f:?} round trip");
        }
    }

    /// C4 S2b: the packed layout is a third value, not `false`. The pre-S2b bool
    /// mapped anything that was not exactly `f16` to f32, so a `q8_0` region would
    /// have been handed to the f32 kernels.
    #[test]
    fn the_packed_layout_is_not_the_f32_one() {
        assert_eq!(format_of(KV_LAYOUT_Q8_0), KvFormat::Q8_0);
        assert_ne!(format_of(KV_LAYOUT_Q8_0), KvFormat::F32);
        assert_ne!(format_of(KV_LAYOUT_Q8_0), KvFormat::F16);
    }
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
        let (tier, tier_mmq) = match std::env::var("MINFER_DEVICE_TIER")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            Some(key) => {
                let s = device_tier::select_forced(key);
                eprintln!(
                    "CUDA: device tier FORCED {} ({:?}, mmq {}) — key {}",
                    s.tier.name, s.tier.provenance, s.mmq_available, key
                );
                (s.tier, s.mmq_available)
            }
            None => {
                let s = device_tier::select(cc_val);
                eprintln!(
                    "CUDA: device tier {} ({:?}, mmq {})",
                    s.tier.name, s.tier.provenance, s.mmq_available
                );
                (s.tier, s.mmq_available)
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

        let dummy = (CudaPtr(std::ptr::null_mut()), 0usize);
        // Eager dynamic-smem opt-in for the prefill GEMM instantiations:
        // must happen BEFORE any stream capture — capture mode Global
        // forbids cudaFuncSetAttribute, so a lazy first-use opt-in fails
        // and the >48KB launch poisons the context (error 700).
        //
        // #145: every attribute call's return value is checked inside
        // `gemm_prefill_smem_init` and named there (`cudaFuncSetAttribute(
        // gemm_f16_nt_kernel_t<..>, cudaFuncAttributeMaxDynamicSharedMemorySize,
        // .. B) failed: cudaError... (1)`); a request over the device's opt-in
        // limit is skipped there, with the reason. The failure count is a
        // startup fact, not something the next `sync()` should re-blame on a
        // kernel.
        let smem_failures = unsafe { gemm_prefill_smem_init() };
        if smem_failures > 0 {
            eprintln!(
                "CUDA: {smem_failures} prefill-GEMM dynamic-smem opt-in call(s) failed at init \
                 (each named above); the affected >48 KiB instantiation(s) keep the 48 KiB default \
                 and must not be selected for a captured launch"
            );
        }
        let smem_skipped = unsafe { gemm_prefill_smem_skipped() };
        if smem_skipped > 0 {
            eprintln!(
                "CUDA: {smem_skipped} prefill-GEMM dynamic-smem opt-in request(s) skipped because \
                 they exceed this device's limit (see the reasons above)"
            );
        }
        Some(CudaState {
            stream: Mutex::new(CudaPtr(stream)),
            staging: Mutex::new(None),
            readback: Mutex::new(None),
            weights: Mutex::new(HashMap::new()),
            w16_cache: Mutex::new(HashMap::new()),
            w16_enabled: std::sync::atomic::AtomicBool::new(false),
            cc: std::sync::atomic::AtomicI32::new(major * 100 + minor),
            tier,
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
            buf_hidden: Mutex::new(dummy),
            buf_bn: Mutex::new(dummy),
            buf_bq: Mutex::new(dummy),
            buf_bk: Mutex::new(dummy),
            buf_bv: Mutex::new(dummy),
            buf_ba: Mutex::new(dummy),
            buf_bf: Mutex::new(dummy),
            buf_bg: Mutex::new(dummy),
            buf_q8_bn: Mutex::new(dummy),
            buf_q8_prefill: Mutex::new(dummy),
            buf_qa8_t: Mutex::new(dummy),
            buf_sda_t: Mutex::new(dummy),
            buf_mmq_ksplit: Mutex::new(dummy),
            mmq_cache: Mutex::new(MmqCache::default()),
            buf_attn_partial: Mutex::new(dummy),
            buf_q8_decode: Mutex::new(dummy),
            buf_f16_w: Mutex::new(dummy),
            buf_f16_x: Mutex::new(dummy),
            buf_q8_ba: Mutex::new(dummy),
            buf_positions: Mutex::new(dummy),
            buf_logits: Mutex::new(dummy),
            kv_k: Mutex::new(Vec::new()),
            kv_v: Mutex::new(Vec::new()),
            kv_size: Mutex::new(Vec::new()),
            decode_graph_exec: Mutex::new(CudaPtr(std::ptr::null_mut())),
            stream_lock: Mutex::new(()),
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

    #[allow(dead_code)] // legacy surface (7e⑦): used by layer_gpu
    pub fn has_weight(&self, name: &str) -> bool {
        self.weights.lock().unwrap().contains_key(name)
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

    /// Free device bytes, or `None` when the query failed (E5's `auto` fit refuses rather
    /// than reading a failure as "0 bytes free"; issue #122).
    pub fn device_free_bytes(&self) -> Option<usize> {
        self.device_memory().free_bytes()
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
            cudaMemcpy(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
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
    pub fn q4dsc_planes(&self) -> Vec<(String, usize)> {
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
    pub fn q4dsc_plane_for(&self, name: &str) -> Option<*mut std::ffi::c_void> {
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

    pub fn stream(&self) -> *mut std::ffi::c_void {
        self.stream.lock().unwrap().0
    }

    // ─── Persistent buffer management ─────────────────────────

    #[allow(dead_code)] // legacy surface (7e⑦)
    fn get_or_grow(slot: &Mutex<(CudaPtr, usize)>, need: usize) -> *mut std::ffi::c_void {
        let mut guard = slot.lock().unwrap();
        let (ptr, size) = &mut *guard;
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
        let mut guard = self.staging.lock().unwrap();
        if guard.is_none() {
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
                *guard = Some(PinnedPool {
                    ptrs,
                    slot_bytes: STAGING_SLOT_BYTES,
                    next: 0,
                });
            }
        }
        let pool = match guard.as_mut() {
            Some(p) if data.len() <= p.slot_bytes => p,
            _ => {
                drop(guard);
                self.copy_to_device(data, dst);
                return;
            }
        };
        // ring wrap: retire all in-flight copies before reusing slot 0. The
        // reset is re-checked under the re-lock so two threads that both
        // observed the full ring cannot both take slot 0 (Phase 8 review).
        if pool.next == pool.ptrs.len() {
            drop(guard);
            self.sync();
            guard = self.staging.lock().unwrap();
        }
        let slot = {
            let pool = guard.as_mut().unwrap();
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

    /// Process-wide stream serialization handle (see the field docs). The
    /// returned reference is `&'static` at every call site because `CudaState`
    /// itself is only ever built as `&'static` (Box::leak in `get`), so the
    /// elided lifetime there is `'static` — guards may be stored.
    pub fn stream_lock(&self) -> &Mutex<()> {
        &self.stream_lock
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
        // F5: a full stream sync is a host stall — count it. It is the number the
        // split-boundary before/after is stated in (`stream_sync_count`).
        STREAM_SYNCS.fetch_add(1, Ordering::Relaxed);
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

    /// Clear and return the CUDA per-thread "last error" latch (`cudaGetLastError`).
    ///
    /// Used by the device gates that must assert a call left **no** error
    /// behind (e.g. `cuda_graph_destroy_*`, issue #145), and by diagnostics.
    #[allow(dead_code)] // read by the #145 device gates
    pub fn take_last_error(&self) -> i32 {
        unsafe { cudaGetLastError() }
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
    #[allow(dead_code)] // read by the backend's execute_node and the #162 gates
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

    /// Drop a pending required-launch record without reporting it. Used by a
    /// documented fallback path (`minfer_launch_ok_opt` never sets one, but an
    /// earlier required site in the same call may have) and by the `Err` arm of
    /// `execute_node`, whose own message is the real error.
    #[allow(dead_code)]
    pub fn clear_launch_failure(&self) {
        unsafe { minfer_launch_fail_clear() };
    }

    /// Debug sync: print label, then sync and report error.
    /// `il` = layer index, or negative for non-layer steps (e.g. output norm).
    /// Only active when MINFER_CUDA_DEBUG is set.
    #[allow(dead_code)]
    pub fn debug_sync(&self, il: i32, label: &str) {
        if !cuda_debug_enabled() {
            return;
        }
        let err = unsafe { cudaGetLastError() };
        if il >= 0 {
            let tag = format!("l{il}: ");
            if err != 0 {
                eprintln!("CUDA DEBUG: {tag}{label} -- latched API error: {err}");
            }
            let err = unsafe { cudaStreamSynchronize(self.stream()) };
            if err != 0 {
                eprintln!("CUDA DEBUG: {tag}{label} -- sync error: {err}");
            } else {
                eprintln!("CUDA DEBUG: {tag}{label} OK");
            }
        } else {
            if err != 0 {
                eprintln!("CUDA DEBUG: {label} -- latched API error: {err}");
            }
            let err = unsafe { cudaStreamSynchronize(self.stream()) };
            if err != 0 {
                eprintln!("CUDA DEBUG: {label} -- sync error: {err}");
            } else {
                eprintln!("CUDA DEBUG: {label} OK");
            }
        }
    }

    // ─── Upload/download for forward pass ─────────────────────

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn upload_hidden(&self, hidden: &[f32]) {
        let need = hidden.len() * 4;
        let ptr = Self::get_or_grow(&self.buf_hidden, need);
        self.copy_to_device(
            unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, need) },
            ptr,
        );
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn download_hidden(&self, hidden: &mut [f32]) {
        let need = hidden.len() * 4;
        let guard = self.buf_hidden.lock().unwrap();
        let ptr = guard.0 .0;
        if ptr.is_null() {
            return;
        }
        self.copy_from_device(ptr as *const std::ffi::c_void, unsafe {
            std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, need)
        });
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn upload_positions(&self, positions: &[usize]) {
        let ints: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let need = ints.len() * 4;
        let ptr = Self::get_or_grow(&self.buf_positions, need);
        self.copy_to_device(
            unsafe { std::slice::from_raw_parts(ints.as_ptr() as *const u8, need) },
            ptr,
        );
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn get_positions_buf(&self) -> *mut std::ffi::c_void {
        self.buf_positions.lock().unwrap().0 .0
    }

    // ─── KV cache management ─────────────────────────────────

    /// Pre-allocate GPU KV cache for all layers to n_ctx entries.
    /// Must be called after model loading (when n_layer, n_ctx, nkt are known)
    /// but before the first forward pass. Eliminates O(n²) incremental growth.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn init_kv_cache(&self, n_layer: usize, n_ctx: usize, nkt: usize) {
        let need = n_ctx * nkt * 4;
        let mut kvec = self.kv_k.lock().unwrap();
        let mut vvec = self.kv_v.lock().unwrap();
        let mut szvec = self.kv_size.lock().unwrap();
        for il in 0..n_layer {
            let new_k = Self::cuda_malloc(need);
            let new_v = Self::cuda_malloc(need);
            if new_k.is_null() || new_v.is_null() {
                eprintln!("CUDA: failed to pre-allocate KV cache for layer {}", il);
                return;
            }
            kvec.push(CudaPtr(new_k));
            vvec.push(CudaPtr(new_v));
            szvec.push(n_ctx);
        }
        let total_kb = (n_layer * need * 2) / 1024;
        eprintln!(
            "CUDA: pre-allocated KV cache for {} layers ({:.1} MB)",
            n_layer,
            total_kb as f64 / 1024.0
        );
    }

    /// Verify KV cache has enough room for `max_nkv` entries at layer `il`.
    /// Returns false if capacity is exceeded (should never happen with pre-allocation).
    #[allow(dead_code)] // legacy surface (7e⑦)
    fn kv_ensure_layer(&self, il: usize, max_nkv: usize) -> bool {
        let szvec = self.kv_size.lock().unwrap();
        let size = szvec.get(il).copied().unwrap_or(0);
        if max_nkv > size {
            eprintln!(
                "CUDA: KV cache overflow at layer {}: need {} but allocated {}",
                il, max_nkv, size
            );
            return false;
        }
        true
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn get_kv_size(&self, il: usize) -> usize {
        let szvec = self.kv_size.lock().unwrap();
        szvec.get(il).copied().unwrap_or(0)
    }

    /// Download logits from GPU after layer loop.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn download_logits(&self, logits: &mut [f32]) {
        let need = logits.len() * 4;
        let guard = self.buf_logits.lock().unwrap();
        let ptr = guard.0 .0;
        if ptr.is_null() {
            return;
        }
        self.copy_from_device(ptr as *const std::ffi::c_void, unsafe {
            std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, need)
        });
    }

    // ─── CUDA Graph (decode step batch) ───────────────────────

    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    pub fn graph_available(&self) -> bool {
        !self.decode_graph_exec.lock().unwrap().0.is_null()
    }

    pub fn graph_begin_capture(&self) -> bool {
        let stream = self.stream();
        let err = unsafe { cudaStreamBeginCapture(stream, 1) };
        if err != 0 {
            unsafe {
                cudaGetLastError();
            }
            false
        } else {
            true
        }
    }

    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    pub fn graph_end_capture(&self) {
        let stream = self.stream();

        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        if err != 0 || graph.is_null() {
            if err != 0 {
                unsafe {
                    cudaGetLastError();
                }
            }
            return;
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
        if err != 0 || exec.is_null() {
            // #147: `cudaGraphDestroy` takes the `cudaGraph_t` from
            // `cudaStreamEndCapture`; read its own return value here.
            let derr = unsafe { cudaGraphDestroy(graph) };
            if derr != 0 {
                eprintln!("{}", graph_destroy_failure_message(derr));
                unsafe {
                    cudaGetLastError(); // this site owns the error
                }
            }
            return;
        }

        let derr = unsafe { cudaGraphDestroy(graph) };
        if derr != 0 {
            eprintln!("{}", graph_destroy_failure_message(derr));
            unsafe {
                cudaGetLastError(); // this site owns the error
            }
        }
        *self.decode_graph_exec.lock().unwrap() = CudaPtr(exec);
    }

    /// Close a capture window and return the instantiated exec handle (null
    /// on failure, after clearing the CUDA error state). Used by the
    /// graph-path backend, which owns per-(uid, range) exec storage; the
    /// legacy `graph_end_capture` single-slot flow is unchanged.
    pub fn graph_end_capture_to_exec(&self) -> *mut std::ffi::c_void {
        let stream = self.stream();

        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        if err != 0 || graph.is_null() {
            if err != 0 {
                unsafe {
                    cudaGetLastError();
                }
            }
            eprintln!("CUDA: stream capture end failed (err {err})");
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

    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    pub fn graph_launch(&self) -> bool {
        let exec = self.decode_graph_exec.lock().unwrap().0;
        if exec.is_null() {
            return false;
        }
        let stream = self.stream();
        let err = unsafe { cudaGraphLaunch(exec, stream) };
        if err != 0 {
            return false;
        }
        true
    }

    // ─── Kernel launch operations (called from CudaCommandBuffer) ──

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn quant_matmul_q8(
        &self,
        w: &Tensor,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let wptr = self.get_weight_ptr(&w.name).expect("weight not on GPU");
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_0_matmul(
                wptr as *const u8,
                x as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// f32-activation matmul dispatch by raw weight pointer + tensor type.
    /// The graph backend (graph/cuda_backend.rs) resolves weights by name and
    /// holds no Tensor, so dispatch takes (ptr, ttype) directly; the legacy
    /// Tensor-taking entry point below delegates here.
    pub fn matmul_f32_ptr(
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
        let mut c = self.mmq_cache.lock().unwrap();
        c.active = false;
        c.key = (0, 0, 0);
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
        let mut cache = self.mmq_cache.lock().unwrap();
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
        let mut cache = self.mmq_cache.lock().unwrap();
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
            let cache = self.mmq_cache.lock().unwrap();
            if cache.active && !cache.transposed && !cache.dead_write && cache.key == key {
                let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as usize;
                if q8 == cache.q8 {
                    return q8 as *mut u8;
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
        let mut c = self.mmq_cache.lock().unwrap();
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
        let mut c = self.mmq_cache.lock().unwrap();
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

    pub fn quant_matmul_f32_on_gpu(
        &self,
        w: &Tensor,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let wptr = self.get_weight_ptr(&w.name).expect("weight not on GPU");
        self.matmul_f32_ptr(wptr, w.ttype, x, out, od, id, nt)
            .unwrap_or_else(|e| panic!("CUDA: {e}"));
    }

    pub fn matmul_on_gpu(
        &self,
        w: &Tensor,
        q8_x: *mut std::ffi::c_void,
        f32_x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        if w.ttype == TensorType::Q4_0 {
            self.quant_matmul_q8(w, q8_x, out, od, id, nt);
        } else if w.ttype == TensorType::Q8_0 {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        } else if w.ttype == TensorType::Q4_1 {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        } else {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        }
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
        let pstr = ((4 + hd + 3) & !3) as i32;
        const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
        let need = ATTN_SPLITS * nh * (pstr as usize) * 4;
        let partial = Self::get_or_grow(&self.buf_attn_partial, need);
        let stream = self.stream();
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
                    stream,
                );
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

    // ─── Batch quant_matmul (for Q/K/V projection) ────────────

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn quant_matmul_f32_batch(
        &self,
        mats: &mut [(
            /*weight*/ &Tensor,
            /*output*/ &mut [f32],
            /*od*/ usize,
        )],
        x: &[f32],
        id: usize,
        nt: usize,
    ) {
        // For batch Q4_0 matmuls: quantize activations once, then launch each matmul
        if mats.iter().any(|m| m.0.ttype != TensorType::Q4_0) {
            // Fall back to CPU for non-Q4_0 types
            for mat in mats.iter_mut() {
                crate::kernel::cpu_quant_matmul_f32(mat.0, x, mat.1, mat.2, id, nt);
            }
            return;
        }

        let nb = id / 32;
        let q8_len = nt * nb * Q8B;
        let mut q8 = vec![0u8; q8_len];
        crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut q8);

        let xbuf = Self::get_or_grow(&self.buf_hidden, q8_len);
        self.copy_to_device(&q8, xbuf);

        // Launch each matmul and read back results
        for (_i, mat) in mats.iter_mut().enumerate() {
            let out_len = nt * mat.2 * 4;
            let obuf = Self::get_or_grow(&self.buf_bq, out_len);
            self.quant_matmul_q8(mat.0, xbuf, obuf, mat.2, id, nt);
            self.sync();
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(mat.1.as_mut_ptr() as *mut u8, out_len) };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        }
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn quant_matmul_f32(
        &self,
        w: &Tensor,
        x: &[f32],
        out: &mut [f32],
        od: usize,
        id: usize,
        nt: usize,
    ) {
        if w.ttype == TensorType::Q4_0 {
            let nb = id / 32;
            let q8_len = nt * nb * Q8B;
            let out_len = nt * od * 4;

            let mut q8 = vec![0u8; q8_len];
            crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut q8);

            let xbuf = Self::get_or_grow(&self.buf_hidden, q8_len);
            let obuf = Self::get_or_grow(&self.buf_logits, out_len);

            self.copy_to_device(&q8, xbuf);
            self.quant_matmul_q8(w, xbuf, obuf, od, id, nt);
            self.sync();
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_len) };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        } else if w.ttype == TensorType::Q8_0 {
            let out_len = nt * od * 4;
            let x_len = nt * id * 4;
            let xbuf = Self::get_or_grow(&self.buf_hidden, x_len);
            let obuf = Self::get_or_grow(&self.buf_logits, out_len);
            self.copy_to_device(
                unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x_len) },
                xbuf,
            );
            self.quant_matmul_f32_on_gpu(w, xbuf, obuf, od, id, nt);
            self.sync();
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_len) };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        } else {
            crate::kernel::cpu_quant_matmul_f32(w, x, out, od, id, nt);
        }
    }

    // ─── Full-layer GPU pass ──────────────────────────────────

    /// Encode one transformer layer onto the CUDA stream.
    /// Returns false if any weight is missing from GPU.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn layer_gpu(
        &self,
        il: usize,
        l: &crate::models::qwen2::loader::LayerWeights,
        positions: &[usize],
        ne: usize,
        nqt: usize,
        nkt: usize,
        nf: usize,
        nt: usize,
        nh: usize,
        nk: usize,
        hd: usize,
        eps: f32,
        freq_base: f32,
        freq_scale: f32,
    ) -> bool {
        let attn_norm = match &l.attn_norm {
            Some(t) => t,
            None => return false,
        };
        let ffn_norm = match &l.ffn_norm {
            Some(t) => t,
            None => return false,
        };
        let wq = l.wq.as_ref().unwrap();
        let wk = l.wk.as_ref().unwrap();
        let wv = l.wv.as_ref().unwrap();
        let wo = l.wo.as_ref().unwrap();
        let ffn_gate = l.ffn_gate.as_ref().unwrap();
        let ffn_up = l.ffn_up.as_ref().unwrap();
        let ffn_down = l.ffn_down.as_ref().unwrap();

        // Accept Q4_0/Q4_1 group or Q4_K/Q6_K group (no mixing between groups)
        fn is_q4(t: TensorType) -> bool {
            t == TensorType::Q4_0 || t == TensorType::Q4_1
        }
        fn is_qk(t: TensorType) -> bool {
            t == TensorType::Q4_K || t == TensorType::Q6_K
        }
        let all_q4 = is_q4(wq.ttype)
            && is_q4(wk.ttype)
            && is_q4(wv.ttype)
            && is_q4(wo.ttype)
            && is_q4(ffn_gate.ttype)
            && is_q4(ffn_up.ttype)
            && is_q4(ffn_down.ttype);
        let all_qk = is_qk(wq.ttype)
            && is_qk(wk.ttype)
            && is_qk(wv.ttype)
            && is_qk(wo.ttype)
            && is_qk(ffn_gate.ttype)
            && is_qk(ffn_up.ttype)
            && is_qk(ffn_down.ttype);
        if !all_q4 && !all_qk {
            return false;
        }

        if !self.has_weight(&wq.name)
            || !self.has_weight(&wk.name)
            || !self.has_weight(&wv.name)
            || !self.has_weight(&wo.name)
            || !self.has_weight(&ffn_gate.name)
            || !self.has_weight(&ffn_up.name)
            || !self.has_weight(&ffn_down.name)
        {
            return false;
        }
        let norm_attn_w = match self.get_weight_ptr(&attn_norm.name) {
            Some(p) => p,
            None => return false,
        };
        let norm_ffn_w = match self.get_weight_ptr(&ffn_norm.name) {
            Some(p) => p,
            None => return false,
        };
        let bq_bias = l.bq.as_ref().and_then(|b| self.get_weight_ptr(&b.name));
        let bk_bias = l.bk.as_ref().and_then(|b| self.get_weight_ptr(&b.name));
        let bv_bias = l.bv.as_ref().and_then(|b| self.get_weight_ptr(&b.name));

        let max_pos = positions.iter().copied().max().unwrap_or(0);
        if !self.kv_ensure_layer(il, max_pos + 1) {}

        let hidden_len = nt * ne * 4;
        let bn_len = hidden_len;
        let bq_len = nt * nqt * 4;
        let bk_len = nt * nkt * 4;
        let bv_len = bk_len;
        let ba_len = nt * ne * 4;
        let bf_len = nt * nf.max(ne) * 4;
        let bg_len = nt * nf * 4;
        let q8_bn_len = nt * (ne / 32) * Q8B;
        let q8_ba_len = nt * (nf.max(ne) / 32) * Q8B;

        let hidden = Self::get_or_grow(&self.buf_hidden, hidden_len);
        let bn = Self::get_or_grow(&self.buf_bn, bn_len);
        let bq_buf = Self::get_or_grow(&self.buf_bq, bq_len);
        let bk_buf = Self::get_or_grow(&self.buf_bk, bk_len);
        let bv_buf = Self::get_or_grow(&self.buf_bv, bv_len);
        let ba_buf = Self::get_or_grow(&self.buf_ba, ba_len);
        let bf_buf = Self::get_or_grow(&self.buf_bf, bf_len);
        let bg_buf = Self::get_or_grow(&self.buf_bg, bg_len);
        let q8_bn = Self::get_or_grow(&self.buf_q8_bn, q8_bn_len);
        let q8_ba = Self::get_or_grow(&self.buf_q8_ba, q8_ba_len);
        let pos_buf = self.get_positions_buf();
        let kv_k = self.kv_k.lock().unwrap()[il].0;
        let kv_v = self.kv_v.lock().unwrap()[il].0;

        // Attention branch
        self.rms_norm(hidden, Some(norm_attn_w), bn, ne, nt, eps);
        self.debug_sync(il as i32, "rms_norm(attn)");

        self.quantize_q8_0(bn, q8_bn, ne, nt);
        self.debug_sync(il as i32, "quantize_q8_0(attn)");
        self.matmul_on_gpu(wq, q8_bn, bn, bq_buf, nqt, ne, nt);
        self.debug_sync(il as i32, "wq matmul");
        if let Some(bb) = bq_bias {
            self.add_bias_f32(bq_buf, bb, nqt, nt);
            self.debug_sync(il as i32, "bq bias");
        }
        self.matmul_on_gpu(wk, q8_bn, bn, bk_buf, nkt, ne, nt);
        self.debug_sync(il as i32, "wk matmul");
        if let Some(bb) = bk_bias {
            self.add_bias_f32(bk_buf, bb, nkt, nt);
            self.debug_sync(il as i32, "bk bias");
        }
        self.matmul_on_gpu(wv, q8_bn, bn, bv_buf, nkt, ne, nt);
        self.debug_sync(il as i32, "wv matmul");
        if let Some(bb) = bv_bias {
            self.add_bias_f32(bv_buf, bb, nkt, nt);
            self.debug_sync(il as i32, "bv bias");
        }
        self.rope_f32(bq_buf, nh, hd, nt, freq_base, freq_scale, pos_buf);
        self.debug_sync(il as i32, "rope q");
        self.rope_f32(bk_buf, nk, hd, nt, freq_base, freq_scale, pos_buf);
        self.debug_sync(il as i32, "rope k");
        self.store_kv_f32(bk_buf, kv_k as *mut std::ffi::c_void, nkt, nt, pos_buf);
        self.debug_sync(il as i32, "store_kv k");
        self.store_kv_f32(bv_buf, kv_v as *mut std::ffi::c_void, nkt, nt, pos_buf);
        self.debug_sync(il as i32, "store_kv v");
        let scale = 1.0 / (hd as f32).sqrt();
        // The layer-level fused path is single-sequence (its caller is the
        // graph's FusedQKV/QkvBiasRopeStore layer, which has no span input), so
        // it always takes the causal instantiation.
        self.gqa_attn_f32(
            bq_buf,
            kv_k as *mut std::ffi::c_void,
            kv_v as *mut std::ffi::c_void,
            ba_buf,
            pos_buf,
            AttnWindow::Causal.code(),
            KV_LAYOUT_F32,
            nh,
            nk,
            hd,
            scale,
            (nk * hd * 4) as usize,
            nt,
        );
        self.debug_sync(il as i32, "gqa_attn");

        // wo projection
        self.quantize_q8_0(ba_buf, q8_ba, ne, nt);
        self.debug_sync(il as i32, "quantize_q8_0(wo)");
        self.matmul_on_gpu(wo, q8_ba, ba_buf, bn, ne, ne, nt);
        self.debug_sync(il as i32, "wo matmul");
        self.add_f32(hidden, bn, hidden, nt * ne);
        self.debug_sync(il as i32, "add(residual attn)");

        // FFN branch
        self.rms_norm(hidden, Some(norm_ffn_w), ba_buf, ne, nt, eps);
        self.debug_sync(il as i32, "rms_norm(ffn)");
        self.quantize_q8_0(ba_buf, q8_ba, ne, nt);
        self.debug_sync(il as i32, "quantize_q8_0(ffn)");
        self.matmul_on_gpu(ffn_gate, q8_ba, ba_buf, bg_buf, nf, ne, nt);
        self.debug_sync(il as i32, "ffn_gate matmul");
        self.matmul_on_gpu(ffn_up, q8_ba, ba_buf, bf_buf, nf, ne, nt);
        self.debug_sync(il as i32, "ffn_up matmul");
        self.swiglu_f32(bg_buf, bf_buf, bg_buf, nt * nf);
        self.debug_sync(il as i32, "swiglu");
        self.quantize_q8_0(bg_buf, q8_ba, nf, nt);
        self.debug_sync(il as i32, "quantize_q8_0(ffn_down)");
        self.matmul_on_gpu(ffn_down, q8_ba, bg_buf, bn, ne, nf, nt);
        self.debug_sync(il as i32, "ffn_down matmul");
        self.add_f32(hidden, bn, hidden, nt * ne);
        self.debug_sync(il as i32, "add(residual ffn)");

        true
    }

    /// Final RMSNorm + output matmul on GPU.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn output_norm_gpu(
        &self,
        output: &Tensor,
        output_norm: Option<&Tensor>,
        output_b: Option<&Tensor>,
        ne: usize,
        nv: usize,
        nt: usize,
        n_out: usize,
        eps: f32,
    ) -> bool {
        let norm_w = match output_norm {
            Some(t) => match self.get_weight_ptr(&t.name) {
                Some(w) => w,
                None => return false,
            },
            None => return false,
        };
        if !self.has_weight(&output.name) {
            return false;
        }
        if output.ttype != TensorType::Q4_0
            && output.ttype != TensorType::Q8_0
            && output.ttype != TensorType::Q4_1
            && output.ttype != TensorType::Q4_K
            && output.ttype != TensorType::Q6_K
        {
            return false;
        }
        debug_assert!(n_out <= nt, "n_out={n_out} > nt={nt}");

        // Output rows = last n_out tokens (single-sequence [nt][ne] row-major).
        let hid_off = (nt - n_out) * ne * 4;

        let hidden = Self::get_or_grow(&self.buf_hidden, nt * ne * 4);
        let bn = Self::get_or_grow(&self.buf_bn, n_out * ne * 4);
        let logits = Self::get_or_grow(&self.buf_logits, n_out * nv * 4);

        let hidden_off = unsafe { (hidden as *mut u8).add(hid_off) } as *mut std::ffi::c_void;
        self.rms_norm(hidden_off, Some(norm_w), bn, ne, n_out, eps);
        self.debug_sync(-1, "output: rms_norm");

        if output.ttype == TensorType::Q4_0 {
            let q8_len = n_out * (ne / 32) * Q8B;
            let q8_bn = Self::get_or_grow(&self.buf_q8_bn, q8_len);
            self.quantize_q8_0(bn, q8_bn, ne, n_out);
            self.debug_sync(-1, "output: quantize_q8_0");
            self.quant_matmul_q8(output, q8_bn, logits, nv, ne, n_out);
            self.debug_sync(-1, "output: q4_0 matmul");
        } else {
            self.quant_matmul_f32_on_gpu(output, bn, logits, nv, ne, n_out);
            self.debug_sync(-1, "output: f32 matmul");
        }

        if let Some(ob) = output_b {
            if let Some(bias_buf) = self.get_weight_ptr(&ob.name) {
                self.add_bias_f32(logits, bias_buf, nv, n_out);
                self.debug_sync(-1, "output: bias");
            }
        }
        true
    }
}

#[cfg(test)]
mod d35_probe_tests {
    use super::*;

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    fn dev_alloc(bytes: usize) -> *mut std::ffi::c_void {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut p, bytes) };
        assert_eq!(err, 0, "cudaMalloc failed");
        p
    }

    fn h2d_f32(dst: *mut std::ffi::c_void, src: &[f32]) {
        let err = unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len() * 4,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(err, 0);
    }

    fn d2h_f32(src: *mut std::ffi::c_void, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        let err = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                n * 4,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(err, 0);
        out
    }

    /// D2H readback of the shared decode q8 scratch (private-field probe).
    fn read_q8(st: &CudaState, bytes: usize) -> Vec<u8> {
        let guard = st.buf_q8_decode.lock().unwrap();
        let (ptr, size) = &*guard;
        assert!(*size >= bytes, "q8 scratch smaller than probe readback");
        let mut out = vec![0u8; bytes];
        let err = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ptr.0 as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(err, 0);
        out
    }

    /// Deterministic activation with real-model spread (RMSNorm outputs reach
    /// ±3 and some 32-blocks are near-zero).
    fn gen_acts(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (s >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                (u as f32) * mag
            })
            .collect()
    }

    /// Valid finite q4_K bytes (od x id), deterministic per (r, ib).
    fn gen_q4_k(od: usize, id: usize) -> Vec<u8> {
        let nbe = id / 256;
        let mut b = Vec::with_capacity(od * nbe * 144);
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                for j in 0..12 {
                    b.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
                }
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    b.push(lo | (hi << 4));
                }
            }
        }
        b
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// D3-5 1a bitwise probe: fused-producer q8 epilogues vs the standalone
    /// quantize kernel — identical pad40 bytes, identical f32 producer
    /// outputs, bit-identical MMVQ outputs through the cache-hit path.
    #[test]
    fn cuda_embed_rows_q4_1_reference() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        let (vocab, d) = (5usize, 64usize);
        let nb = d / 32;
        // synthetic Q4_1 blocks (20B: f16 d, f16 m, 16 nibble bytes)
        let mut w = vec![0u8; vocab * nb * 20];
        for r in 0..vocab {
            for b in 0..nb {
                let off = (r * nb + b) * 20;
                let dbits = 0x3800u16.wrapping_add(((r * 7 + b * 3) as u16) * 64);
                let mbits = 0x3800u16.wrapping_add(((r * 5 + b) as u16) * 32);
                w[off..off + 2].copy_from_slice(&dbits.to_le_bytes());
                w[off + 2..off + 4].copy_from_slice(&mbits.to_le_bytes());
                for j in 0..16usize {
                    w[off + 4 + j] = ((j * 17 + r * 13 + b * 29) & 0xFF) as u8;
                }
            }
        }
        st.register_weight("q41_embed_w", &w);
        let wptr = st.get_weight_ptr("q41_embed_w").unwrap();
        // row ids travel as I32-as-f32 bit patterns (graph rule §4)
        let ids: Vec<i32> = vec![0, 3, 4, 1];
        let nt = ids.len();
        let ids_f: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i as u32)).collect();
        let dids = dev_alloc(nt * 4);
        h2d_f32(dids, &ids_f);
        let dout = dev_alloc(nt * d * 4);
        st.embed_rows_on_gpu(TensorType::Q4_1, wptr, dids, dout, d, nt, false)
            .unwrap();
        let got = d2h_f32(dout, nt * d);
        for (t, &row) in ids.iter().enumerate() {
            for b in 0..nb {
                let off = (row as usize * nb + b) * 20;
                let dv = half::f16::from_bits(u16::from_le_bytes([w[off], w[off + 1]])).to_f32();
                let mv =
                    half::f16::from_bits(u16::from_le_bytes([w[off + 2], w[off + 3]])).to_f32();
                for j in 0..16usize {
                    let lo = (w[off + 4 + j] & 0x0F) as f32;
                    let hi = (w[off + 4 + j] >> 4) as f32;
                    // the kernel's d*q+m may compile to a fused multiply-add;
                    // accept either rounding (both are one-ulp forms)
                    for (e, v) in [(b * 32 + j, lo), (b * 32 + j + 16, hi)] {
                        let sep = dv * v + mv;
                        let fma = dv.mul_add(v, mv);
                        assert!(
                            got[t * d + e] == sep || got[t * d + e] == fma,
                            "row {row} elem {e}: got {} want {sep} (fma {fma})",
                            got[t * d + e]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cuda_decode_a_quant_fuse_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        let eps = 1e-5f32;

        // ---- rms probe: d 5120 (14B hidden) -> q4_K matmul (MMVQ gate) ----
        let (d, od) = (5120usize, 256usize);
        let x = gen_acts(d, 0x9E37);
        let w: Vec<f32> = (0..d).map(|i| 0.5 + 0.001 * (i % 17) as f32).collect();
        let dx = dev_alloc(d * 4);
        h2d_f32(dx, &x);
        let dw = dev_alloc(d * 4);
        h2d_f32(dw, &w);
        let dy_a = dev_alloc(d * 4);
        let dy_b = dev_alloc(d * 4);
        let do_a = dev_alloc(od * 4);
        let do_b = dev_alloc(od * 4);
        st.register_weight("d35_probe_w4", &gen_q4_k(od, d));
        let wq = st
            .get_weight_ptr("d35_probe_w4")
            .expect("weight registered");

        // path A (pre-D3-5 shape): plain rms; matmul cache-cleared -> the
        // standalone quantize launch inside the decode matmul
        st.clear_mmq_cache();
        st.rms_norm(dx, Some(dw), dy_a, d, 1, eps);
        st.matmul_f32_ptr_layout(wq, TensorType::Q4_K, dy_a, do_a, od, d, 1, false)
            .unwrap();
        let q8_a = read_q8(st, (d / 32) * 40);

        // path B: fused rms (records the plane) + matmul (cache hit)
        st.clear_mmq_cache();
        st.rms_norm_quant_on_gpu(dx, dw, dy_b, d, 1, eps);
        st.matmul_f32_ptr_layout(wq, TensorType::Q4_K, dy_b, do_b, od, d, 1, false)
            .unwrap();
        let q8_b = read_q8(st, (d / 32) * 40);

        assert_eq!(q8_a, q8_b, "rms epilogue q8 bytes differ from standalone");
        let y_a = d2h_f32(dy_a, d);
        let y_b = d2h_f32(dy_b, d);
        assert!(
            bits_eq(&y_a, &y_b),
            "fused rms f32 output not bit-identical"
        );
        let o_a = d2h_f32(do_a, od);
        let o_b = d2h_f32(do_b, od);
        assert!(bits_eq(&o_a, &o_b), "MMVQ output diverged (rms path)");

        // ---- swiglu probe: n 2048 (down id, q4_K MMVQ gate) ----
        let (nf, odf) = (2048usize, 256usize);
        let gate = gen_acts(nf, 0x1234);
        let up = gen_acts(nf, 0x5678);
        let mut buf = vec![0f32; 2 * nf];
        buf[..nf].copy_from_slice(&gate);
        buf[nf..].copy_from_slice(&up);
        let db_a = dev_alloc(2 * nf * 4);
        h2d_f32(db_a, &buf);
        let db_b = dev_alloc(2 * nf * 4);
        h2d_f32(db_b, &buf);
        let da_a = dev_alloc(odf * 4);
        let da_b = dev_alloc(odf * 4);
        st.register_weight("d35_probe_w4b", &gen_q4_k(odf, nf));
        let wq2 = st
            .get_weight_ptr("d35_probe_w4b")
            .expect("weight registered");

        st.clear_mmq_cache();
        st.swiglu_f32_off_on_gpu(db_a, nf, nf);
        st.matmul_f32_ptr_layout(wq2, TensorType::Q4_K, db_a, da_a, odf, nf, 1, false)
            .unwrap();
        let q8_a2 = read_q8(st, (nf / 32) * 40);

        st.clear_mmq_cache();
        st.swiglu_quant_off_on_gpu(db_b, nf, nf);
        st.matmul_f32_ptr_layout(wq2, TensorType::Q4_K, db_b, da_b, odf, nf, 1, false)
            .unwrap();
        let q8_b2 = read_q8(st, (nf / 32) * 40);

        assert_eq!(
            q8_a2, q8_b2,
            "swiglu epilogue q8 bytes differ from standalone"
        );
        let bu_a = d2h_f32(db_a, 2 * nf);
        let bu_b = d2h_f32(db_b, 2 * nf);
        assert!(
            bits_eq(&bu_a, &bu_b),
            "fused swiglu f32 output not bit-identical"
        );
        let oa_a = d2h_f32(da_a, odf);
        let oa_b = d2h_f32(da_b, odf);
        assert!(bits_eq(&oa_a, &oa_b), "MMVQ output diverged (swiglu path)");
    }
}

// ────────────────────────────────────────────────────────────────────
// D3-8 probes: FusedQKV decode fusion (CUDA port of the Metal G4 path)
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod d38_probe_tests {
    use super::*;
    use crate::tensor::TensorType;

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    fn dev_alloc(bytes: usize) -> *mut std::ffi::c_void {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut p, bytes) };
        assert_eq!(err, 0, "cudaMalloc failed");
        p
    }

    fn h2d(dst: *mut std::ffi::c_void, src: &[u8]) {
        let err = unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(err, 0);
    }

    fn h2d_f32(dst: *mut std::ffi::c_void, src: &[f32]) {
        let mut bytes = Vec::with_capacity(src.len() * 4);
        for v in src {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        h2d(dst, &bytes);
    }

    fn d2h(src: *mut std::ffi::c_void, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        let err = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(err, 0);
        out
    }

    fn d2h_f32(src: *mut std::ffi::c_void, n: usize) -> Vec<f32> {
        let bytes = d2h(src, n * 4);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Deterministic activation with real-model spread (same generator class
    /// as the D3-5 probes).
    fn gen_acts(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (s >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                (u as f32) * mag
            })
            .collect()
    }

    /// Valid finite q4_K bytes (od x id), deterministic per (r, ib).
    fn gen_q4_k(od: usize, id: usize) -> Vec<u8> {
        let nbe = id / 256;
        let mut b = Vec::with_capacity(od * nbe * 144);
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                for j in 0..12 {
                    b.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
                }
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    b.push(lo | (hi << 4));
                }
            }
        }
        b
    }

    /// D3-8 probe A: fused `attn_bias_rope_store` vs the unfused chain
    /// (add_bias×3 + rope×2 + store_kv×2) on the SAME concat-matmul output —
    /// bitwise on the q/k/v sections AND both KV regions, f32 + f16 KV,
    /// 14B + 7B shapes, BOTH pointer forms (concat sections AND three
    /// separate buffers — the mixed-quant class-2 wiring).
    #[test]
    fn cuda_fused_qkv_epilogue_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        let (freq_base, freq_scale) = (10000.0f32, 1.0f32);
        let pos0: i32 = 1234;

        // device positions buffer in the graph's i32 form (the kernel reads
        // positions[0]; the graph path converts via f32_bits_to_i32)
        let dpos = dev_alloc(4);
        h2d(dpos, &pos0.to_le_bytes());

        // (nh, nk, hd, tag): 14B GQA 40:8, 7B GQA 28:4 — hd 128 both
        for &(nh, nk, hd, tag) in &[(40usize, 8usize, 128usize, "14B"), (28, 4, 128, "7B")] {
            let (nqt, nkt) = (nh * hd, nk * hd);
            let total = nqt + 2 * nkt;
            let ctx_elems = nkt * (pos0 as usize + 2); // room for the store row
            let acts = gen_acts(total, 0xD380 + nh as u64);
            let bq = gen_acts(nqt, 0xB10 + nh as u64);
            let bk = gen_acts(nkt, 0xB20 + nh as u64);
            let bv = gen_acts(nkt, 0xB30 + nh as u64);

            let dbq = dev_alloc(nqt * 4);
            h2d_f32(dbq, &bq);
            let dbk = dev_alloc(nkt * 4);
            h2d_f32(dbk, &bk);
            let dbv = dev_alloc(nkt * 4);
            h2d_f32(dbv, &bv);

            for kv_f16 in [false, true] {
                let kv_bytes = if kv_f16 { 2 } else { 4 };

                // ---- FUSED form 1: concat buffer, section pointers ----
                let d_fused = dev_alloc(total * 4);
                h2d_f32(d_fused, &acts);
                let base = d_fused as *mut u8;
                let (q1, k1, v1) = (
                    d_fused,
                    unsafe { base.add(nqt * 4) } as *mut std::ffi::c_void,
                    unsafe { base.add((nqt + nkt) * 4) } as *mut std::ffi::c_void,
                );
                let dk_f = dev_alloc(ctx_elems * kv_bytes);
                let dv_f = dev_alloc(ctx_elems * kv_bytes);
                st.attn_bias_rope_store(
                    q1,
                    k1,
                    v1,
                    dbq,
                    dbk,
                    dbv,
                    dk_f,
                    dv_f,
                    nqt,
                    nkt,
                    hd,
                    freq_base,
                    freq_scale,
                    dpos,
                    dpos,
                    if kv_f16 { KV_LAYOUT_F16 } else { KV_LAYOUT_F32 },
                );

                // ---- FUSED form 2: three separate buffers (class-2 shape) --
                let d_q2 = dev_alloc(nqt * 4);
                h2d_f32(d_q2, &acts[..nqt]);
                let d_k2 = dev_alloc(nkt * 4);
                h2d_f32(d_k2, &acts[nqt..nqt + nkt]);
                let d_v2 = dev_alloc(nkt * 4);
                h2d_f32(d_v2, &acts[nqt + nkt..]);
                let dk_f2 = dev_alloc(ctx_elems * kv_bytes);
                let dv_f2 = dev_alloc(ctx_elems * kv_bytes);
                st.attn_bias_rope_store(
                    d_q2,
                    d_k2,
                    d_v2,
                    dbq,
                    dbk,
                    dbv,
                    dk_f2,
                    dv_f2,
                    nqt,
                    nkt,
                    hd,
                    freq_base,
                    freq_scale,
                    dpos,
                    dpos,
                    if kv_f16 { KV_LAYOUT_F16 } else { KV_LAYOUT_F32 },
                );

                // ---- UNFUSED: the 7-launch chain on split sections ----
                let d_q = dev_alloc(nqt * 4);
                h2d_f32(d_q, &acts[..nqt]);
                let d_k = dev_alloc(nkt * 4);
                h2d_f32(d_k, &acts[nqt..nqt + nkt]);
                let d_v = dev_alloc(nkt * 4);
                h2d_f32(d_v, &acts[nqt + nkt..]);
                st.add_bias_f32(d_q, dbq, nqt, 1);
                st.add_bias_f32(d_k, dbk, nkt, 1);
                st.add_bias_f32(d_v, dbv, nkt, 1);
                st.rope_f32(d_q, nh, hd, 1, freq_base, freq_scale, dpos);
                st.rope_f32(d_k, nk, hd, 1, freq_base, freq_scale, dpos);
                let dk_u = dev_alloc(ctx_elems * kv_bytes);
                let dv_u = dev_alloc(ctx_elems * kv_bytes);
                if kv_f16 {
                    st.store_kv_f16(d_k, dk_u, nkt, 1, dpos);
                    st.store_kv_f16(d_v, dv_u, nkt, 1, dpos);
                } else {
                    st.store_kv_f32(d_k, dk_u, nkt, 1, dpos);
                    st.store_kv_f32(d_v, dv_u, nkt, 1, dpos);
                }

                // ---- compare: q/k/v sections + both KV rows ----
                let (rq, rk, rv) = (d2h_f32(d_q, nqt), d2h_f32(d_k, nkt), d2h_f32(d_v, nkt));
                let f1 = d2h_f32(d_fused, total);
                let (qf, kf, vf) = (
                    &f1[..nqt],
                    &f1[nqt..nqt + nkt],
                    &f1[nqt + nkt..nqt + 2 * nkt],
                );
                assert!(
                    bits_eq(qf, &rq),
                    "{tag} f16={kv_f16}: concat q section diverged"
                );
                assert!(
                    bits_eq(kf, &rk),
                    "{tag} f16={kv_f16}: concat k section diverged"
                );
                assert!(
                    bits_eq(vf, &rv),
                    "{tag} f16={kv_f16}: concat v section diverged"
                );
                assert!(
                    bits_eq(&d2h_f32(d_q2, nqt), &rq)
                        && bits_eq(&d2h_f32(d_k2, nkt), &rk)
                        && bits_eq(&d2h_f32(d_v2, nkt), &rv),
                    "{tag} f16={kv_f16}: separate-buffer form diverged"
                );
                let row = (pos0 as usize) * nkt;
                if kv_f16 {
                    for (f, u, what) in [
                        (dk_f, dk_u, "K"),
                        (dv_f, dv_u, "V"),
                        (dk_f2, dk_u, "K form2"),
                        (dv_f2, dv_u, "V form2"),
                    ] {
                        let fh = d2h(f, ctx_elems * 2);
                        let uh = d2h(u, ctx_elems * 2);
                        assert_eq!(
                            fh[row * 2..(row + nkt) * 2],
                            uh[row * 2..(row + nkt) * 2],
                            "{tag}: f16 {what} region diverged"
                        );
                    }
                } else {
                    for (f, u, what) in [
                        (dk_f, dk_u, "K"),
                        (dv_f, dv_u, "V"),
                        (dk_f2, dk_u, "K form2"),
                        (dv_f2, dv_u, "V form2"),
                    ] {
                        assert!(
                            bits_eq(
                                &d2h_f32(f, ctx_elems)[row..row + nkt],
                                &d2h_f32(u, ctx_elems)[row..row + nkt],
                            ),
                            "{tag}: f32 {what} region diverged"
                        );
                    }
                }
                eprintln!("{tag} kv_f16={kv_f16}: epilogue bitwise OK (both pointer forms)");
            }
        }
    }

    /// D3-8 probe B: the concat matmul (one launch over wq|wk|wv rows) is
    /// per-row bit-identical to the three separate decode matmuls — the MMVQ
    /// kernels map one block per row and dispatch on (ttype, id, nt) only.
    #[test]
    fn cuda_fused_qkv_concat_matmul_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();

        // (oq, okv, id, tag): 14B wq|wk|wv = 5120|1024|1024 id 5120;
        // 7B = 3584|512|512 id 3584. Both decode-MMVQ (id ≥ 2048, % 32 == 0).
        for &(oq, okv, id, tag) in &[
            (5120usize, 1024usize, 5120usize, "14B"),
            (3584, 512, 3584, "7B"),
        ] {
            let od_total = oq + 2 * okv;
            let row_bytes = (id / 256) * 144; // q4_K
            let concat = gen_q4_k(od_total, id);
            st.register_weight("d38_concat", &concat);
            st.register_weight("d38_wq", &concat[..row_bytes * oq]);
            st.register_weight("d38_wk", &concat[row_bytes * oq..row_bytes * (oq + okv)]);
            st.register_weight(
                "d38_wv",
                &concat[row_bytes * (oq + okv)..row_bytes * od_total],
            );
            let w_cat = st.get_weight_ptr("d38_concat").expect("concat registered");
            let w_q = st.get_weight_ptr("d38_wq").expect("wq registered");
            let w_k = st.get_weight_ptr("d38_wk").expect("wk registered");
            let w_v = st.get_weight_ptr("d38_wv").expect("wv registered");
            assert!(!st.is_weight_padded("d38_concat"));

            let x = gen_acts(id, 0xC0DE + id as u64);
            let dx = dev_alloc(id * 4);
            h2d_f32(dx, &x);
            let d_cat = dev_alloc(od_total * 4);
            // unfused: wq/wk/wv write separate output buffers (the live path)
            let d_q = dev_alloc(oq * 4);
            let d_k = dev_alloc(okv * 4);
            let d_v = dev_alloc(okv * 4);

            // unfused: three separate decode matmuls (share the A-quantize
            // MmqCache exactly like the live wq/wk/wv group)
            st.clear_mmq_cache();
            st.matmul_f32_ptr_layout(w_q, TensorType::Q4_K, dx, d_q, oq, id, 1, false)
                .unwrap();
            st.matmul_f32_ptr_layout(w_k, TensorType::Q4_K, dx, d_k, okv, id, 1, false)
                .unwrap();
            st.matmul_f32_ptr_layout(w_v, TensorType::Q4_K, dx, d_v, okv, id, 1, false)
                .unwrap();

            // fused: one concat matmul (fresh cache window, standalone quantize)
            st.clear_mmq_cache();
            st.matmul_f32_ptr_layout(w_cat, TensorType::Q4_K, dx, d_cat, od_total, id, 1, false)
                .unwrap();

            let cat = d2h_f32(d_cat, od_total);
            let sq = d2h_f32(d_q, oq);
            let sk = d2h_f32(d_k, okv);
            let sv = d2h_f32(d_v, okv);
            assert!(bits_eq(&cat[..oq], &sq), "{tag}: q rows diverged");
            assert!(bits_eq(&cat[oq..oq + okv], &sk), "{tag}: k rows diverged");
            assert!(bits_eq(&cat[oq + okv..], &sv), "{tag}: v rows diverged");
            eprintln!("{tag}: concat matmul bitwise OK ({od_total} rows)");
        }
    }

    // D4-4 L1: the dense split-plane (dpl) decode path must be bitwise
    // identical to the padded path. Registers the same raw q6_K bytes twice
    // (dpl built vs MINFER_Q6K_DPL=0) and compares decode outputs bit-exact.
    #[test]
    fn cuda_q6k_dpl_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        // (od, id): a v2_pf_dpl shape (id in (8192, 16384]) and a v2_dpl
        // loop shape; both must clear the decode MMVQ gate (nt==1, id%32==0,
        // od*id >= 4M).
        for (od, id) in [(512usize, 8960usize), (4096usize, 1024usize)] {
            let raw = gen_q6_k_raw(od, id);
            std::env::remove_var("MINFER_Q6K_DPL");
            st.register_weight_q6k_padded("d44_dpl_a", &raw, od, id);
            std::env::set_var("MINFER_Q6K_DPL", "0");
            st.register_weight_q6k_padded("d44_dpl_b", &raw, od, id);
            std::env::remove_var("MINFER_Q6K_DPL");
            let wa = st.get_weight_ptr("d44_dpl_a").expect("a registered");
            let wb = st.get_weight_ptr("d44_dpl_b").expect("b registered");
            let x: Vec<f32> = (0..id).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();
            let dx = dev_alloc(id * 4);
            h2d_f32(dx, &x);
            let oa = dev_alloc(od * 4);
            let ob = dev_alloc(od * 4);
            st.matmul_f32_ptr_layout(wa, TensorType::Q6_K, dx, oa, od, id, 1, true)
                .unwrap();
            st.matmul_f32_ptr_layout(wb, TensorType::Q6_K, dx, ob, od, id, 1, true)
                .unwrap();
            let ra = d2h_f32(oa, od);
            let rb = d2h_f32(ob, od);
            assert_eq!(
                ra, rb,
                "dpl-vs-padded decode outputs must be bit-identical (od {od} id {id})"
            );
            eprintln!("d44_dpl: od {od} id {id} bitwise OK");
        }
    }

    // raw GGUF-layout q6_K bytes: random ql/qh nibbles, small int8 scales,
    // d = 1.0 (finite outputs so the bit-exact compare is meaningful).
    fn gen_q6_k_raw(od: usize, id: usize) -> Vec<u8> {
        let nbe = id.div_ceil(256);
        let mut v = vec![0u8; od * nbe * 210];
        let mut s: u32 = 0x5DEE_CE6D;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for r in 0..od {
            for b in 0..nbe {
                let blk = &mut v[(r * nbe + b) * 210..(r * nbe + b + 1) * 210];
                for i in 0..192 {
                    blk[i] = (next() & 0x55) as u8;
                }
                for i in 0..16 {
                    blk[192 + i] = (next() % 15) as u8;
                }
                blk[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
            }
        }
        v
    }
}

// ────────────────────────────────────────────────────────────────────
// Issue #145: the eager prefill-GEMM smem opt-in is checked, and a latched
// API error is reported with its real origin (never as a kernel launch).
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod issue145_tests {
    use super::*;

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    /// The sync message must name the *observer* (`cudaGetLastError`) and the
    /// real API error, and must not claim a kernel launch. Pure — no device.
    #[test]
    fn the_latched_error_message_never_blames_a_kernel() {
        let msg = latched_api_error_message(1);
        assert!(
            msg.contains("cudaGetLastError"),
            "must name the observer: {msg}"
        );
        assert!(
            msg.contains("cudaErrorInvalidValue"),
            "must name the error symbolically: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("kernel launch"),
            "a latched error must not be attributed to a kernel: {msg}"
        );
    }

    /// The single-source smem formula, pinned against the kernel's own byte
    /// layout (As 256*KS + Am 512*KS [AF32] + Bs 4*TM*KS + Cs 8192, TN=64,
    /// NW=8). A silent shrink of the formula is what under-declares a launch's
    /// dynamic smem; this is the arm that sees it. Pure — no device.
    #[test]
    fn the_gemm_smem_formula_matches_the_kernel_layout() {
        let cases = [
            (64, 32, false, 24576usize),
            (64, 32, true, 40960),
            (64, 64, false, 40960),
            (64, 64, true, 73728),
            (128, 32, false, 32768),
            (128, 32, true, 49152),
            (128, 64, false, 57344),
            (128, 64, true, 90112),
            (256, 32, false, 49152),
            (256, 32, true, 65536),
            (256, 64, false, 90112),
            (256, 64, true, 122880),
        ];
        for (tm, ks, af32, want) in cases {
            assert_eq!(
                unsafe { gemm_smem_need(tm, ks, af32 as i32) },
                want,
                "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> dynamic-smem need"
            );
        }
    }

    /// Every (tm, ks, af32) combination the launcher can select whose request
    /// exceeds the 48 KiB default must actually be admitted by the device's
    /// `cudaFuncGetAttributes().maxDynamicSharedSizeBytes` — otherwise the
    /// prefill GEMM's >48 KiB launch (which capture mode forces to be opted in
    /// eagerly) fails. A request over the device limit must be skipped, never
    /// called. Device gate.
    #[test]
    fn cuda_prefill_smem_optin_covers_every_launchable_instantiation() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        // Re-run the eager init: idempotent, and its return value is the count
        // of attribute calls that failed.
        let failures = unsafe { gemm_prefill_smem_init() };
        assert_eq!(failures, 0, "a request the device admits must not fail");
        let limit = unsafe { gemm_prefill_smem_limit() };
        assert!(limit >= 48 * 1024, "queried opt-in limit {limit} B");
        let mut covered = 0usize;
        let mut over_limit = 0usize;
        for tm in [64, 128, 256] {
            for ks in [32, 64] {
                for af32 in [false, true] {
                    let need = unsafe { gemm_smem_need(tm, ks, af32 as i32) };
                    if need <= 48 * 1024 {
                        continue; // the default cap admits it; no opt-in needed
                    }
                    let opted = unsafe { gemm_smem_opted_in(tm, ks, af32 as i32) } == 1;
                    if need > limit as usize {
                        over_limit += 1;
                        assert!(
                            !opted,
                            "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> needs {need} B > the \
                             {limit} B device limit but reads back as opted in"
                        );
                        continue;
                    }
                    assert!(
                        opted,
                        "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> needs {need} B and the device \
                         admits {limit} B, but cudaFuncGetAttributes reports its \
                         maxDynamicSharedSizeBytes below that — the >48 KiB prefill launch \
                         would fail (capture mode cannot set the attribute lazily)"
                    );
                    covered += 1;
                }
            }
        }
        assert!(
            covered >= 5,
            "expected the launchable >48 KiB instantiations to be opted in, got {covered}"
        );
        assert_eq!(
            unsafe { gemm_prefill_smem_checked() } as usize,
            covered,
            "every admitted >48 KiB request must have been attempted exactly once"
        );
        assert_eq!(
            unsafe { gemm_prefill_smem_skipped() } as usize,
            over_limit,
            "every over-limit request must be skipped with a reason"
        );
    }

    /// `graph_destroy` is handed the `cudaGraphExec_t` from
    /// `cudaGraphInstantiate`; destroying it with `cudaGraphDestroy` returns
    /// `cudaErrorInvalidValue`, leaks the exec, and used to surface later as a
    /// phantom "kernel launch error" (#145). Device gate.
    #[test]
    fn cuda_graph_exec_destroy_leaves_no_latched_error() {
        let _model_load_guard = CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let s = device().unwrap();
        let _ = s.take_last_error(); // start from a clean latch
        assert!(s.graph_begin_capture(), "stream capture should begin");
        let exec = s.graph_end_capture_to_exec();
        assert!(
            !exec.is_null(),
            "an empty captured graph should instantiate"
        );
        // Assert the destroy CALL's own result, not just the latch: the failure
        // is named and cleared inside `graph_destroy`, so a latch-only
        // assertion would pass even with the wrong destructor (the gate would
        // pass for the wrong reason).
        assert!(
            s.graph_destroy(exec),
            "destroying a cudaGraphExec_t must succeed — cudaGraphExecDestroy, \
             not cudaGraphDestroy (which returns cudaErrorInvalidValue and \
             leaks the exec)"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "destroying a cudaGraphExec_t must not latch an API error"
        );
    }

    /// A latched error must still be *visible* at the next sync (not dropped),
    /// reported as latched, and cleared. Device gate.
    ///
    /// Env-gated (`MINFER_TEST_LATCH_ERROR=1`) **because it deliberately
    /// latches a real CUDA API error**: the default suite run must stay clean
    /// under `compute-sanitizer --tool memcheck`, which counts every such call.
    #[test]
    fn cuda_sync_surfaces_a_latched_error_as_latched() {
        let _model_load_guard = CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if std::env::var("MINFER_TEST_LATCH_ERROR").is_err() {
            eprintln!(
                "skipping: set MINFER_TEST_LATCH_ERROR=1 to run this deliberate-latch gate \
                 (it must not pollute a compute-sanitizer run)"
            );
            return;
        }
        let s = device().unwrap();
        let _ = s.take_last_error(); // start from a clean latch
        let before = latched_api_error_count();
        let injected = unsafe { cuda_test_latch_oversized_smem() };
        assert_eq!(
            injected, 1,
            "the injector must latch cudaErrorInvalidValue (1), got {injected}"
        );
        s.sync();
        assert_eq!(
            latched_api_error_count(),
            before + 1,
            "sync must report the latched error instead of dropping it"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "sync must clear the latch it reported"
        );
    }
}

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
mod issue147_tests {
    use super::*;

    /// `minfer_site_fail_kind` values (cuda_kernels.cu).
    const SITE_ATTR: i32 = 1;
    const SITE_LAUNCH: i32 = 2;

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    fn cstr(p: *const std::os::raw::c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    /// The deliberate-failure gates must not run in a default (or sanitizer)
    /// run: they make a real CUDA call fail for real.
    fn issue147_gate_enabled() -> bool {
        if std::env::var("MINFER_TEST_ISSUE147").is_err() {
            eprintln!(
                "skipping: set MINFER_TEST_ISSUE147=1 to run the deliberate-failure gates (they \
                 make real CUDA calls fail; a compute-sanitizer run must not set it)"
            );
            return false;
        }
        true
    }

    /// Restores `MINFER_TEST_CALL_FAIL` on drop, so a panicking gate cannot
    /// leave the injection armed for the rest of the process.
    struct InjectionGuard {
        prev: Option<String>,
    }

    impl InjectionGuard {
        fn arm(site: &str) -> Self {
            let prev = std::env::var("MINFER_TEST_CALL_FAIL").ok();
            std::env::set_var("MINFER_TEST_CALL_FAIL", site);
            Self { prev }
        }
    }

    impl Drop for InjectionGuard {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("MINFER_TEST_CALL_FAIL", v),
                None => std::env::remove_var("MINFER_TEST_CALL_FAIL"),
            }
        }
    }

    /// What one injected call reported, plus its own return value.
    struct Observed {
        ret: i32,
        count: i32,
        kind: i32,
        code: i32,
        bytes: i32,
        limit: i32,
        site: String,
        msg: String,
    }

    /// Clear the latch, arm `site`, run `call`, and read back the site's own
    /// report. The latch is cleared first so the final assertion is about this
    /// call alone.
    fn inject(s: &CudaState, site: &str, call: impl FnOnce() -> i32) -> Observed {
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        let ret = {
            let _g = InjectionGuard::arm(site);
            call()
        };
        Observed {
            ret,
            count: unsafe { minfer_site_fail_count() },
            kind: unsafe { minfer_site_fail_kind() },
            code: unsafe { minfer_site_fail_code() },
            bytes: unsafe { minfer_site_fail_bytes() },
            limit: unsafe { minfer_site_fail_limit() },
            site: cstr(unsafe { minfer_site_fail_site() }),
            msg: cstr(unsafe { minfer_site_fail_message() }),
        }
    }

    fn zeroed_dev(bytes: usize) -> *mut std::ffi::c_void {
        let p = CudaState::cuda_malloc(bytes);
        assert!(!p.is_null(), "cudaMalloc({bytes}) failed");
        let zeros = vec![0u8; bytes];
        let e = unsafe {
            cudaMemcpy(
                p,
                zeros.as_ptr() as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(e, 0, "zero-fill failed: {}", cuda_error_name(e));
        p
    }

    /// The formatter for the Rust-site `cudaGraphDestroy` failure must name the
    /// matching destructor and the error, and must not name the wrong call (a
    /// gate that only asserts "a message appeared" cannot see that). Pure.
    #[test]
    fn the_graph_destroy_failure_message_names_the_matching_destructor() {
        let msg = graph_destroy_failure_message(1);
        assert!(
            msg.contains("cudaGraphDestroy"),
            "must name the call: {msg}"
        );
        assert!(
            msg.contains("cudaGraph_t from cudaStreamEndCapture"),
            "must name the handle and where it came from: {msg}"
        );
        assert!(
            !msg.contains("cudaGraphExecDestroy"),
            "must not name the wrong destructor: {msg}"
        );
        assert!(
            msg.contains("cudaErrorInvalidValue"),
            "must name the error symbolically: {msg}"
        );
        assert!(
            !msg.to_lowercase().contains("kernel launch"),
            "a destroy failure must not be confused with a launch: {msg}"
        );
        assert!(msg.contains("issue #147"), "must carry the ticket: {msg}");
    }

    // The injection selector's exact-token matching moved to
    // `testfail::tests::the_matcher_is_exact_and_comma_separated` (#171): one
    // matcher, unit-tested on the CPU job as well. The device half keeps its
    // own gates below.

    /// Every dynamic-smem opt-in site: the injected (real, over-limit) attribute
    /// call must be named with the API, the attribute, the instantiation and
    /// `cudaGetErrorName`, the launcher must refuse, and the latch must be gone.
    fn assert_attr_failure(
        s: &CudaState,
        site: &str,
        kernel_frag: &str,
        call: impl FnOnce() -> i32,
    ) {
        let o = inject(s, site, call);
        assert_eq!(
            o.count, 1,
            "[{site}] exactly one site failure must be reported: {}",
            o.msg
        );
        assert_eq!(
            o.kind, SITE_ATTR,
            "[{site}] must be the attribute-call failure: {}",
            o.msg
        );
        assert_eq!(o.site, site, "[{site}] the report must name this site");
        assert_eq!(
            cuda_error_name(o.code),
            "cudaErrorInvalidValue",
            "[{site}] the injected call must return cudaErrorInvalidValue: {}",
            o.msg
        );
        assert!(
            o.limit > 0,
            "[{site}] the device opt-in limit must have been queried: {}",
            o.msg
        );
        assert!(
            o.bytes > o.limit,
            "[{site}] the injected request {} must exceed the device limit {}: {}",
            o.bytes,
            o.limit,
            o.msg
        );
        assert!(
            o.msg.contains("cudaFuncSetAttribute"),
            "[{site}] must name the call: {}",
            o.msg
        );
        assert!(
            o.msg
                .contains("cudaFuncAttributeMaxDynamicSharedMemorySize"),
            "[{site}] must name the attribute: {}",
            o.msg
        );
        assert!(
            o.msg.contains("cudaErrorInvalidValue"),
            "[{site}] must name the error: {}",
            o.msg
        );
        assert!(
            o.msg.contains(kernel_frag),
            "[{site}] must name the instantiation {kernel_frag}: {}",
            o.msg
        );
        assert!(
            !o.msg.contains("SKIPPED"),
            "[{site}] an injected failure is a real failing call, not a deliberate skip: {}",
            o.msg
        );
        assert_eq!(o.ret, 0, "[{site}] the launcher must refuse the launch");
        assert_eq!(
            s.take_last_error(),
            0,
            "[{site}] the site must clear the latch it named (it must not reach sync)"
        );
    }

    /// Every launch site: the injected launch must be named, the launcher must
    /// refuse, and the latch must be gone.
    fn assert_launch_failure(
        s: &CudaState,
        site: &str,
        kernel_frag: &str,
        call: impl FnOnce() -> i32,
    ) {
        let o = inject(s, site, call);
        assert_eq!(
            o.count, 1,
            "[{site}] exactly one site failure must be reported: {}",
            o.msg
        );
        assert_eq!(
            o.kind, SITE_LAUNCH,
            "[{site}] must be the launch failure: {}",
            o.msg
        );
        assert_eq!(o.site, site, "[{site}] the report must name this site");
        assert_eq!(
            cuda_error_name(o.code),
            "cudaErrorInvalidValue",
            "[{site}] the injected launch must return cudaErrorInvalidValue: {}",
            o.msg
        );
        assert!(
            o.msg.contains("kernel launch"),
            "[{site}] must say the launch failed: {}",
            o.msg
        );
        assert!(
            o.msg.contains(kernel_frag),
            "[{site}] must name the instantiation {kernel_frag}: {}",
            o.msg
        );
        assert!(
            o.msg.contains("cudaErrorInvalidValue"),
            "[{site}] must name the error: {}",
            o.msg
        );
        assert_eq!(o.ret, 0, "[{site}] the launcher must refuse the launch");
        assert_eq!(
            s.take_last_error(),
            0,
            "[{site}] the site must clear the latch it named (it must not reach sync)"
        );
    }

    /// Issue #147 acceptance, site by site: a deliberately failed
    /// `cudaFuncSetAttribute` at every dynamic-smem site is named where it is
    /// made and the following launch is refused. Device + env-gated.
    #[test]
    fn cuda_issue147_attribute_sites_name_the_call_and_refuse_the_launch() {
        let _model_load_guard = CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !issue147_gate_enabled() {
            return;
        }
        let s = device().unwrap();
        let w = zeroed_dev(1 << 20);
        let q8 = zeroed_dev(1 << 20);
        let c = zeroed_dev(1 << 20);
        let stream = s.stream();

        // launch_mmq_raw_nt: both kd branches of the terminal raw launcher.
        assert_attr_failure(
            s,
            "attr:mmq_raw_nt_kd4",
            "mmq_raw_nt_kernel<4>",
            || unsafe {
                launch_mmq_raw_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    stream,
                    4,
                )
            },
        );
        assert_attr_failure(
            s,
            "attr:mmq_raw_nt_kd8",
            "mmq_raw_nt_kernel<8>",
            || unsafe {
                launch_mmq_raw_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    stream,
                    8,
                )
            },
        );
        // launch_mmq_nt (the MMQ_LAUNCH macro; one call per quant type).
        assert_attr_failure(s, "attr:mmq_nt", "mmq_nt_kernel<0,1,false>", || unsafe {
            launch_mmq_nt(
                0,
                w as *const u8,
                q8 as *const u8,
                c as *mut f32,
                1,
                64,
                64,
                40,
                stream,
            )
        });
        // launch_mmq_raw_nb_nt.
        assert_attr_failure(s, "attr:mmq_raw_nb", "mmq_raw_nb_kernel<8>", || unsafe {
            launch_mmq_raw_nb_nt(
                5,
                w as *const u8,
                q8 as *const u8,
                c as *mut f32,
                1,
                64,
                256,
                stream,
                8,
            )
        });
        // launch_mmq_raw_nb_bt_nt (both DSC branches share one site token).
        assert_attr_failure(
            s,
            "attr:mmq_raw_nb_bt",
            "mmq_raw_nb_bt_kernel<8,false>",
            || unsafe {
                launch_mmq_raw_nb_bt_nt(
                    5,
                    w as *const u8,
                    std::ptr::null(),
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    8,
                    stream,
                    8,
                    std::ptr::null_mut(),
                    1,
                )
            },
        );
        // launch_mmq_raw_nb_bt_q6k_nt.
        assert_attr_failure(
            s,
            "attr:mmq_raw_nb_bt_q6k",
            "mmq_raw_nb_bt_q6k_kernel<2,false>",
            || unsafe {
                launch_mmq_raw_nb_bt_q6k_nt(
                    7,
                    w as *const u8,
                    std::ptr::null(),
                    std::ptr::null(),
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    2,
                    210,
                    stream,
                    8,
                    std::ptr::null_mut(),
                    1,
                )
            },
        );
        // launch_mmq_raw_wide_nt: both kd branches.
        assert_attr_failure(
            s,
            "attr:mmq_raw_wide_kd4",
            "mmq_raw_wide_nt_kernel<4>",
            || unsafe {
                launch_mmq_raw_wide_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    128,
                    256,
                    stream,
                    4,
                )
            },
        );
        assert_attr_failure(
            s,
            "attr:mmq_raw_wide_kd8",
            "mmq_raw_wide_nt_kernel<8>",
            || unsafe {
                launch_mmq_raw_wide_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    128,
                    256,
                    stream,
                    8,
                )
            },
        );
        // launch_gemm_f16: the family is tm/ks-dependent (their env knobs are
        // read once per process), so the assertion pins the family and the
        // af32 instantiation suffix rather than one exact (tm, ks).
        assert_attr_failure(s, "attr:gemm_f16_f16", "gemm_f16_nt_kernel_t<", || unsafe {
            launch_gemm_f16(
                w as *const std::ffi::c_void,
                q8 as *const std::ffi::c_void,
                c as *mut f32,
                1,
                64,
                32,
                stream,
                false,
            )
        });
        assert_attr_failure(s, "attr:gemm_f16_a32", ",true>", || unsafe {
            launch_gemm_f16(
                w as *const std::ffi::c_void,
                q8 as *const std::ffi::c_void,
                c as *mut f32,
                1,
                64,
                32,
                stream,
                true,
            )
        });

        // Positive control: with the knob off, the same terminal launcher must
        // still launch. A launcher that always refused would pass the loop above
        // for the wrong reason.
        let _ = s.take_last_error();
        assert_eq!(
            unsafe {
                launch_mmq_raw_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    stream,
                    8,
                )
            },
            1,
            "the non-injected raw-narrow launcher must launch"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "the non-injected launch must leave no latch"
        );

        unsafe {
            cudaFree(w);
            cudaFree(q8);
            cudaFree(c);
        }
    }

    /// Issue #147 acceptance for the launches themselves: a deliberately failed
    /// `<<<>>>` is named at the site, the launcher refuses, and the latch is
    /// cleared. Device + env-gated.
    #[test]
    fn cuda_issue147_launch_sites_name_the_call_and_refuse_the_launch() {
        let _model_load_guard = CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !issue147_gate_enabled() {
            return;
        }
        let s = device().unwrap();
        let w = zeroed_dev(1 << 20);
        let q8 = zeroed_dev(1 << 20);
        let c = zeroed_dev(1 << 20);
        let stream = s.stream();

        assert_launch_failure(
            s,
            "launch:mmq_raw_nt_kd4",
            "mmq_raw_nt_kernel<4>",
            || unsafe {
                launch_mmq_raw_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    stream,
                    4,
                )
            },
        );
        assert_launch_failure(
            s,
            "launch:mmq_raw_nt_kd8",
            "mmq_raw_nt_kernel<8>",
            || unsafe {
                launch_mmq_raw_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    stream,
                    8,
                )
            },
        );
        assert_launch_failure(s, "launch:mmq_nt", "mmq_nt_kernel<0,1,false>", || unsafe {
            launch_mmq_nt(
                0,
                w as *const u8,
                q8 as *const u8,
                c as *mut f32,
                1,
                64,
                64,
                40,
                stream,
            )
        });
        assert_launch_failure(s, "launch:mmq_raw_nb", "mmq_raw_nb_kernel<8>", || unsafe {
            launch_mmq_raw_nb_nt(
                5,
                w as *const u8,
                q8 as *const u8,
                c as *mut f32,
                1,
                64,
                256,
                stream,
                8,
            )
        });
        assert_launch_failure(
            s,
            "launch:mmq_raw_nb_bt",
            "mmq_raw_nb_bt_kernel<8,false>",
            || unsafe {
                launch_mmq_raw_nb_bt_nt(
                    5,
                    w as *const u8,
                    std::ptr::null(),
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    8,
                    stream,
                    8,
                    std::ptr::null_mut(),
                    1,
                )
            },
        );
        assert_launch_failure(
            s,
            "launch:mmq_raw_nb_bt_q6k",
            "mmq_raw_nb_bt_q6k_kernel<2,false>",
            || unsafe {
                launch_mmq_raw_nb_bt_q6k_nt(
                    7,
                    w as *const u8,
                    std::ptr::null(),
                    std::ptr::null(),
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    64,
                    256,
                    2,
                    210,
                    stream,
                    8,
                    std::ptr::null_mut(),
                    1,
                )
            },
        );
        assert_launch_failure(
            s,
            "launch:mmq_raw_wide_kd4",
            "mmq_raw_wide_nt_kernel<4>",
            || unsafe {
                launch_mmq_raw_wide_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    128,
                    256,
                    stream,
                    4,
                )
            },
        );
        assert_launch_failure(
            s,
            "launch:mmq_raw_wide_kd8",
            "mmq_raw_wide_nt_kernel<8>",
            || unsafe {
                launch_mmq_raw_wide_nt(
                    5,
                    w as *const u8,
                    q8 as *const u8,
                    c as *mut f32,
                    1,
                    128,
                    256,
                    stream,
                    8,
                )
            },
        );
        assert_launch_failure(
            s,
            "launch:gemm_f16_f16",
            "gemm_f16_nt_kernel_t<",
            || unsafe {
                launch_gemm_f16(
                    w as *const std::ffi::c_void,
                    q8 as *const std::ffi::c_void,
                    c as *mut f32,
                    1,
                    64,
                    32,
                    stream,
                    false,
                )
            },
        );
        assert_launch_failure(s, "launch:gemm_f16_a32", ",true>", || unsafe {
            launch_gemm_f16(
                w as *const std::ffi::c_void,
                q8 as *const std::ffi::c_void,
                c as *mut f32,
                1,
                64,
                32,
                stream,
                true,
            )
        });

        // Positive control: the same GEMM launcher must still launch with the
        // knob off (otherwise the loop above would pass on an always-refusing
        // launcher).
        let _ = s.take_last_error();
        assert_eq!(
            unsafe {
                launch_gemm_f16(
                    w as *const std::ffi::c_void,
                    q8 as *const std::ffi::c_void,
                    c as *mut f32,
                    1,
                    64,
                    32,
                    stream,
                    false,
                )
            },
            1,
            "the non-injected GEMM launcher must launch"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "the non-injected launch must leave no latch"
        );

        unsafe {
            cudaFree(w);
            cudaFree(q8);
            cudaFree(c);
        }
    }

    /// Issue #147 acceptance for the Rust destroy site: an injected failed
    /// `cudaGraphDestroy` (the pre-#145 wrong-destructor call) is named and
    /// cleared, and the instantiated exec — which the failed destroy does not
    /// touch — is still returned and still destroyable. Device + env-gated.
    #[test]
    fn cuda_issue147_graph_destroy_failure_is_named_and_the_exec_survives() {
        let _model_load_guard = CudaState::model_load_guard();
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !issue147_gate_enabled() {
            return;
        }
        let s = device().unwrap();
        let _ = s.take_last_error();
        assert!(s.graph_begin_capture(), "stream capture should begin");
        let exec = {
            let _g = InjectionGuard::arm("destroy:graph_destroy");
            s.graph_end_capture_to_exec()
        };
        // The injected call is `cudaGraphDestroy(exec)` — the pre-#145 bug — so
        // the *graph* leaks while the exec stays valid. Refusing the exec would
        // be wrong (only the graph handle is lost), so the site names the
        // failure, clears the latch, and still returns the exec.
        assert!(
            !exec.is_null(),
            "a failed cudaGraphDestroy(graph) must not refuse the valid exec"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "the destroy site must not leave its error for CudaState::sync"
        );
        assert!(
            s.graph_destroy(exec),
            "the returned exec must still be a destroyable cudaGraphExec_t"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "and that destroy must leave no latch"
        );
    }
}

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
mod issue162_tests {
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    /// `minfer_site_fail_kind` for a kernel launch (`cuda_kernels.cu`).
    const SITE_LAUNCH: i32 = 2;
    /// `MINFER_SITE_*` / `ATTN_WIN_*` mirrors (cuda_kernels.cu).
    const CAUSAL: i32 = 0;
    const SPAN: i32 = 1;
    const MAP: i32 = 2;
    const ERR_INVALID_VALUE: i32 = 1; // cudaErrorInvalidValue

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    fn cstr(p: *const std::os::raw::c_char) -> String {
        if p.is_null() {
            return String::new();
        }
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }

    fn gate_enabled() -> bool {
        if std::env::var("MINFER_TEST_ISSUE162").is_err() {
            eprintln!(
                "skipping: set MINFER_TEST_ISSUE162=1 to run the #162 launch-failure gates \
                 (they drive real CUDA launches into failure; a compute-sanitizer run must not \
                 set it)"
            );
            return false;
        }
        true
    }

    /// The committed audit fixture (`scripts/check_cuda_launch_returns.py
    /// --fixture tests/fixtures/cuda_launch_sites.tsv`): `line, owner, site,
    /// kernel-fragment` for every `<<<` site in `cuda_kernels.cu`. The gate
    /// asserts the *driven* set against it, so a launcher the driver misses, or a
    /// new site added without a driver, is a red gate rather than a silent gap.
    fn fixture() -> Vec<(String, String, String, String)> {
        include_str!("../tests/fixtures/cuda_launch_sites.tsv")
            .lines()
            .filter(|l| !l.trim().is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                assert_eq!(f.len(), 4, "fixture row: {l}");
                (
                    f[0].to_string(),
                    f[1].to_string(),
                    f[2].to_string(),
                    f[3].to_string(),
                )
            })
            .collect()
    }

    /// Restores `MINFER_TEST_CALL_FAIL` on drop, so a panicking gate cannot leave
    /// the injection armed for the rest of the process.
    struct Arm {
        prev: Option<String>,
    }
    impl Arm {
        fn new(list: &str) -> Self {
            let prev = std::env::var("MINFER_TEST_CALL_FAIL").ok();
            std::env::set_var("MINFER_TEST_CALL_FAIL", list);
            Self { prev }
        }
    }
    impl Drop for Arm {
        fn drop(&mut self) {
            match &self.prev {
                Some(v) => std::env::set_var("MINFER_TEST_CALL_FAIL", v),
                None => std::env::remove_var("MINFER_TEST_CALL_FAIL"),
            }
        }
    }

    /// Zeroed device scratch. An injected launch never runs; the two `_opt`
    /// k-split sites the driver reaches alone run their kernel for real, so the
    /// buffers are 1 MiB each — well past every shape the driver passes.
    struct Ctx {
        stream: *mut std::ffi::c_void,
        bufs: Vec<*mut std::ffi::c_void>,
    }
    impl Ctx {
        fn new(s: &CudaState) -> Self {
            let n = 1 << 20;
            let mut bufs = Vec::new();
            for _ in 0..12 {
                let p = CudaState::cuda_malloc(n);
                assert!(!p.is_null(), "cudaMalloc({n}) failed");
                let zeros = vec![0u8; n];
                let e = unsafe {
                    cudaMemcpy(
                        p,
                        zeros.as_ptr() as *const std::ffi::c_void,
                        n,
                        CUDA_MEMCPY_HOST_TO_DEVICE,
                    )
                };
                assert_eq!(e, 0, "zero-fill failed: {}", cuda_error_name(e));
                bufs.push(p);
            }
            Self {
                stream: s.stream(),
                bufs,
            }
        }
        fn p(&self, i: usize) -> *mut std::ffi::c_void {
            self.bufs[i]
        }
        fn f(&self, i: usize) -> *mut f32 {
            self.bufs[i] as *mut f32
        }
        fn cf(&self, i: usize) -> *const f32 {
            self.bufs[i] as *const f32
        }
        fn u(&self, i: usize) -> *const u8 {
            self.bufs[i] as *const u8
        }
        fn mu(&self, i: usize) -> *mut u8 {
            self.bufs[i] as *mut u8
        }
        fn i32(&self, i: usize) -> *mut i32 {
            self.bufs[i] as *mut i32
        }
        fn ci32(&self, i: usize) -> *const i32 {
            self.bufs[i] as *const i32
        }
    }

    /// Drive one dispatch with `arm` armed, then assert: the armed set is exactly
    /// the observed set, every report is a loud named launch failure at its own
    /// site naming its kernel instantiation, and nothing latched.
    fn run(
        s: &CudaState,
        frag: &HashMap<String, String>,
        arm: &[&str],
        seen: &mut BTreeSet<String>,
        call: impl FnOnce(),
    ) {
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        {
            let _g = Arm::new(&arm.join(","));
            call();
        }
        let n = unsafe { minfer_site_hist_len() };
        let mut observed = BTreeSet::new();
        for i in 0..n {
            let site = cstr(unsafe { minfer_site_hist_site(i) });
            let name = cstr(unsafe { minfer_site_hist_name(i) });
            let msg = cstr(unsafe { minfer_site_hist_msg(i) });
            assert!(
                msg.contains("kernel launch"),
                "[{site}] must say the launch failed: {msg}"
            );
            assert!(
                msg.contains("cudaErrorInvalidValue"),
                "[{site}] must name the error with cudaGetErrorName: {msg}"
            );
            assert!(
                msg.contains(&format!("#162/{site}")),
                "[{site}] must carry its own incident tag: {msg}"
            );
            let f = frag.get(&site).unwrap_or_else(|| {
                panic!("a report came from a site not in the audit fixture: {site}")
            });
            if !f.is_empty() {
                assert!(
                    msg.contains(f.as_str()),
                    "[{site}] must name the instantiation fragment `{f}` (reported `{name}`): {msg}"
                );
            }
            assert!(
                observed.insert(site.clone()),
                "[{site}] reported twice in one dispatch"
            );
            seen.insert(site);
        }
        let expected: BTreeSet<String> = arm.iter().map(|s| s.to_string()).collect();
        assert_eq!(
            observed, expected,
            "armed/observed site mismatch (armed {arm:?}); a site the call did not reach, or an \
             unarmed site that failed, is a red gate"
        );
        assert_eq!(
            unsafe { minfer_site_fail_kind() },
            SITE_LAUNCH,
            "kind after arming {arm:?}"
        );
        assert_eq!(
            unsafe { minfer_site_fail_code() },
            ERR_INVALID_VALUE,
            "the injected geometry must return cudaErrorInvalidValue after arming {arm:?}"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "arming {arm:?} must leave no latched error for CudaState::sync"
        );
    }

    /// Issue #162 acceptance, site by site: arming a site drives a REAL failing
    /// launch, and the site names itself and its instantiation and clears the
    /// latch. Every `<<<` site in the audit fixture must be reached.
    #[test]
    fn cuda_issue162_every_launch_site_names_itself_and_leaves_no_latch() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !gate_enabled() {
            return;
        }
        let s = device().unwrap();
        let rows = fixture();
        let frag: HashMap<String, String> =
            rows.iter().map(|r| (r.2.clone(), r.3.clone())).collect();
        let ctx = Ctx::new(s);
        let mut seen: BTreeSet<String> = BTreeSet::new();
        let st = ctx.stream;

        macro_rules! go {
            ($arm:expr, $body:expr) => {
                run(s, &frag, $arm, &mut seen, $body)
            };
        }

        // ── the split-attention families ────────────────────────────────
        for (mode, tag) in [(MAP, "map"), (SPAN, "span"), (CAUSAL, "causal")] {
            let arm: Vec<&str> = vec![
                match tag {
                    "map" => "launch:gqa_attn_split_batched_kv__partial_map",
                    "span" => "launch:gqa_attn_split_batched_kv__partial_span",
                    _ => "launch:gqa_attn_split_batched_kv__partial_causal",
                },
                "launch:gqa_attn_split_batched_kv__combine",
            ];
            go!(&arm, || unsafe {
                launch_gqa_attn_split_batched_f16kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    4,
                    st,
                );
            });
        }

        // ── embedding gather (8 quantized cases) ────────────────────────
        for (type_id, token) in [
            (0, "launch:embed_rows__q8_0"),
            (1, "launch:embed_rows__q4_0"),
            (2, "launch:embed_rows__q4_k"),
            (7, "launch:embed_rows__q4_1"),
            (4, "launch:embed_rows__q5_1"),
            (5, "launch:embed_rows__q5_k"),
            (6, "launch:embed_rows__q5_0"),
            (3, "launch:embed_rows__q6_k"),
        ] {
            go!(&[token], || unsafe {
                launch_embed_rows(ctx.u(0), ctx.cf(1), ctx.f(2), 256, 2, type_id, 210, st);
            });
        }
        go!(&["launch:embed_rows_f16"], || unsafe {
            launch_embed_rows_f16(ctx.u(0), ctx.cf(1), ctx.f(2), 256, 2, st);
        });

        // ── f32 / f16 matmul shape branches ─────────────────────────────
        for (id, token) in [
            (8, "launch:f32_f32_matmul__vec"),
            (7, "launch:f32_f32_matmul__scalar"),
        ] {
            go!(&[token], || unsafe {
                launch_f32_f32_matmul(ctx.cf(0), ctx.cf(1), ctx.f(2), 8, id, 2, st);
            });
        }
        for (id, token) in [
            (8, "launch:f16_f32_matmul_vec"),
            (7, "launch:f16_f32_matmul_scalar"),
        ] {
            go!(&[token], || unsafe {
                launch_f16_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, id, 2, st);
            });
        }

        // ── every single-site launcher ──────────────────────────────────
        go!(&["launch:q4_0_q8_0_matmul"], || unsafe {
            launch_q4_0_q8_0_matmul(ctx.u(0), ctx.u(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q4_0_f32_matmul"], || unsafe {
            launch_q4_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q8_0_f32_matmul"], || unsafe {
            launch_q8_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q4_1_f32_matmul"], || unsafe {
            launch_q4_1_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q4_k_f32_matmul"], || unsafe {
            launch_q4_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q5_1_f32_matmul"], || unsafe {
            launch_q5_1_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q5_0_f32_matmul"], || unsafe {
            launch_q5_0_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q5_k_f32_matmul"], || unsafe {
            launch_q5_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q6_k_f32_matmul"], || unsafe {
            launch_q6_k_f32_matmul(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:q6_k_f32_matmul_padded"], || unsafe {
            launch_q6_k_f32_matmul_padded(ctx.u(0), ctx.cf(1), ctx.f(2), 8, 64, 2, st);
        });
        go!(&["launch:swiglu_f32_off"], || unsafe {
            launch_swiglu_f32_off(ctx.f(0), 16, 0, st);
        });
        go!(&["launch:swiglu_quant_pad40"], || unsafe {
            launch_swiglu_quant_pad40(ctx.f(0), ctx.mu(1), 16, 0, st);
        });
        go!(&["launch:gather_rows_f32"], || unsafe {
            launch_gather_rows_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, 2, st);
        });
        go!(&["launch:quantize_q8_0_pad40"], || unsafe {
            launch_quantize_q8_0_pad40(ctx.cf(0), ctx.mu(1), 256, 2, st);
        });
        go!(&["launch:quantize_q8_0_pad40_t"], || unsafe {
            launch_quantize_q8_0_pad40_t(ctx.cf(0), ctx.mu(1), ctx.mu(2), 256, 2, 8, 1, st);
        });
        go!(&["launch:quantize_q8_0"], || unsafe {
            launch_quantize_q8_0(ctx.cf(0), ctx.mu(1), 256, 2, st);
        });
        go!(&["launch:rms_norm_f32"], || unsafe {
            launch_rms_norm_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, 1e-6, 2, st);
        });
        go!(&["launch:rms_norm_quant_pad40"], || unsafe {
            launch_rms_norm_quant_pad40(ctx.cf(0), ctx.cf(1), ctx.f(2), ctx.mu(3), 64, 1e-6, 8, st);
        });
        go!(&["launch:add_bias_f32"], || unsafe {
            launch_add_bias_f32(ctx.f(0), ctx.cf(1), 64, 8, st);
        });
        go!(&["launch:add_f32"], || unsafe {
            launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
        });
        go!(&["launch:mul_f32"], || unsafe {
            launch_mul_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
        });
        go!(&["launch:silu_f32"], || unsafe {
            launch_silu_f32(ctx.f(0), 64, st);
        });
        go!(&["launch:swiglu_f32"], || unsafe {
            launch_swiglu_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st);
        });
        go!(&["launch:rms_norm_quant_f32_t"], || unsafe {
            launch_rms_norm_quant_f32_t(
                ctx.cf(0),
                ctx.cf(1),
                ctx.f(2),
                ctx.mu(3),
                ctx.mu(4),
                256,
                1e-6,
                8,
                8,
                1,
                st,
            );
        });
        go!(&["launch:rms_norm_quant_nw_f32_t"], || unsafe {
            launch_rms_norm_quant_nw_f32_t(
                ctx.cf(0),
                ctx.cf(1),
                ctx.mu(2),
                ctx.mu(3),
                256,
                1e-6,
                8,
                8,
                1,
                st,
            );
        });
        go!(&["launch:swiglu_quant_f32_t"], || unsafe {
            launch_swiglu_quant_f32_t(
                ctx.cf(0),
                ctx.cf(1),
                ctx.f(2),
                ctx.mu(3),
                ctx.mu(4),
                256,
                2,
                8,
                1,
                st,
            );
        });
        go!(&["launch:swiglu_quant_nw_f32_t"], || unsafe {
            launch_swiglu_quant_nw_f32_t(
                ctx.cf(0),
                ctx.cf(1),
                ctx.mu(2),
                ctx.mu(3),
                256,
                2,
                8,
                1,
                st,
            );
        });
        go!(&["launch:f32_bits_to_i32"], || unsafe {
            launch_f32_bits_to_i32(ctx.cf(0), ctx.i32(1), 64, st);
        });
        go!(&["launch:rope_f32"], || unsafe {
            launch_rope_f32(ctx.f(0), 4, 64, 2, 10000.0, 1.0, ctx.ci32(1), st);
        });
        go!(&["launch:store_kv_f32"], || unsafe {
            launch_store_kv_f32(ctx.cf(0), ctx.f(1), 64, 2, ctx.ci32(2), st);
        });
        go!(&["launch:store_kv_f16"], || unsafe {
            launch_store_kv_f16(ctx.cf(0), ctx.p(1), 64, 2, ctx.ci32(2), st);
        });
        go!(&["launch:store_kv_q8_0"], || unsafe {
            launch_store_kv_q8_0(ctx.cf(0), ctx.p(1), 64, 2, 68, ctx.ci32(2), st);
        });
        go!(&["launch:attn_bias_rope_store"], || unsafe {
            launch_attn_bias_rope_store(
                ctx.f(0),
                ctx.f(1),
                ctx.f(2),
                ctx.p(3),
                ctx.p(4),
                ctx.p(5),
                ctx.p(6),
                ctx.p(7),
                2,
                4,
                64,
                10000.0,
                1.0,
                ctx.ci32(8),
                ctx.ci32(9),
                0,
                st,
            );
        });
        // #144 item 1: the packed arm of the same epilogue.
        go!(&["launch:attn_bias_rope_store_q8_0"], || unsafe {
            launch_attn_bias_rope_store_q8_0(
                ctx.f(0),
                ctx.cf(1),
                ctx.cf(2),
                ctx.p(3),
                ctx.p(4),
                ctx.p(5),
                ctx.p(6),
                ctx.p(7),
                2,
                4,
                64,
                10000.0,
                1.0,
                ctx.ci32(8),
                ctx.ci32(9),
                68,
                st,
            );
        });
        for (mode, token) in [
            (MAP, "launch:gqa_attn_f32_f16kv__map"),
            (SPAN, "launch:gqa_attn_f32_f16kv__span"),
            (CAUSAL, "launch:gqa_attn_f32_f16kv__causal"),
        ] {
            go!(&[token], || unsafe {
                launch_gqa_attn_f32_f16kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.ci32(4),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    2,
                    st,
                );
            });
        }
        for (mode, token) in [
            (MAP, "launch:gqa_attn_split_f32kv__map"),
            (SPAN, "launch:gqa_attn_split_f32kv__span"),
            (CAUSAL, "launch:gqa_attn_split_f32kv__causal"),
        ] {
            go!(
                &[token, "launch:gqa_attn_split_f32kv__combine"],
                || unsafe {
                    launch_gqa_attn_split_f32kv(
                        ctx.cf(0),
                        ctx.p(1),
                        ctx.p(2),
                        ctx.f(3),
                        ctx.f(4),
                        ctx.ci32(5),
                        mode,
                        4,
                        2,
                        64,
                        0.125,
                        68,
                        st,
                    );
                }
            );
        }
        for (mode, token) in [
            (MAP, "launch:gqa_attn_split_q8_0__map"),
            (SPAN, "launch:gqa_attn_split_q8_0__span"),
            (CAUSAL, "launch:gqa_attn_split_q8_0__causal"),
        ] {
            go!(&[token, "launch:gqa_attn_split_q8_0__combine"], || unsafe {
                launch_gqa_attn_split_q8_0(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.f(4),
                    ctx.ci32(5),
                    mode,
                    4,
                    2,
                    64,
                    0.125,
                    68,
                    68,
                    st,
                );
            });
        }
        // hd == 128 takes the dual-kernel (h4w + hybrid) path; hd != 128 the
        // single incumbent one. Both share `__combine`.
        for (mode, mtag) in [(MAP, "map"), (SPAN, "span"), (CAUSAL, "causal")] {
            let h4w = format!("launch:gqa_attn_split_f16kv__{mtag}_h4w");
            let hyb = format!("launch:gqa_attn_split_f16kv__hybrid_{mtag}");
            go!(
                &[
                    h4w.as_str(),
                    hyb.as_str(),
                    "launch:gqa_attn_split_f16kv__combine"
                ],
                || unsafe {
                    launch_gqa_attn_split_f16kv(
                        ctx.cf(0),
                        ctx.p(1),
                        ctx.p(2),
                        ctx.f(3),
                        ctx.f(4),
                        ctx.ci32(5),
                        mode,
                        4,
                        2,
                        128,
                        0.125,
                        68,
                        st,
                    );
                }
            );
            let plain = format!("launch:gqa_attn_split_f16kv__{mtag}");
            go!(
                &[plain.as_str(), "launch:gqa_attn_split_f16kv__combine"],
                || unsafe {
                    launch_gqa_attn_split_f16kv(
                        ctx.cf(0),
                        ctx.p(1),
                        ctx.p(2),
                        ctx.f(3),
                        ctx.f(4),
                        ctx.ci32(5),
                        mode,
                        4,
                        2,
                        64,
                        0.125,
                        68,
                        st,
                    );
                }
            );
        }
        // The layout-tagged general kernel: one source site, nine instantiations
        // (the message names the instantiation). Drive one.
        go!(&["launch:gqa_attn_f32"], || unsafe {
            launch_gqa_attn_f32(
                ctx.cf(0),
                ctx.p(1),
                ctx.p(2),
                ctx.f(3),
                ctx.ci32(4),
                CAUSAL,
                KV_LAYOUT_F32,
                4,
                2,
                64,
                0.125,
                256,
                2,
                st,
            );
        });
        // #144: the FA launcher serves both staged layouts from one source site
        // per (layout, mode) — six sites, all driven here.
        for (mode, layout, token) in [
            (
                MAP,
                crate::cuda::KV_LAYOUT_F16,
                "launch:fa_prefill_kv__f16_map",
            ),
            (
                SPAN,
                crate::cuda::KV_LAYOUT_F16,
                "launch:fa_prefill_kv__f16_span",
            ),
            (
                CAUSAL,
                crate::cuda::KV_LAYOUT_F16,
                "launch:fa_prefill_kv__f16_causal",
            ),
            (
                MAP,
                crate::cuda::KV_LAYOUT_Q8_0,
                "launch:fa_prefill_kv__q8_0_map",
            ),
            (
                SPAN,
                crate::cuda::KV_LAYOUT_Q8_0,
                "launch:fa_prefill_kv__q8_0_span",
            ),
            (
                CAUSAL,
                crate::cuda::KV_LAYOUT_Q8_0,
                "launch:fa_prefill_kv__q8_0_causal",
            ),
        ] {
            go!(&[token], || unsafe {
                launch_fa_prefill_kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.ci32(4),
                    mode,
                    4,
                    2,
                    128,
                    0.125,
                    2,
                    layout,
                    136,
                    st,
                );
            });
        }
        for (type_id, token) in [
            (0, "launch:dequant_f16__q8_0"),
            (1, "launch:dequant_f16__q4_0"),
            (2, "launch:dequant_f16__q4_1"),
            (3, "launch:dequant_f16__q5_0"),
            (4, "launch:dequant_f16__q5_1"),
            (5, "launch:dequant_f16__q4_k"),
            (6, "launch:dequant_f16__q5_k"),
            (7, "launch:dequant_f16__q6_k"),
        ] {
            go!(&[token], || unsafe {
                launch_dequant_f16(type_id, ctx.u(0), ctx.p(1), 8, 64, 210, st);
            });
        }
        go!(&["launch:convert_f16"], || unsafe {
            launch_convert_f16(ctx.cf(0), ctx.p(1), 256, st);
        });
        go!(&["launch:gemm_qb_nt"], || unsafe {
            launch_gemm_qb_nt(ctx.p(0), ctx.u(1), ctx.f(2), 2, 8, 64, 5, 212, st);
        });
        // `launch_gemm_f16`'s site token is chosen by `af32`; the audit resolves
        // the ternary to the first arm, so drive the `af32 = true` one.
        go!(&["launch:gemm_f16_a32"], || unsafe {
            launch_gemm_f16(ctx.p(0), ctx.p(1), ctx.f(2), 2, 8, 64, st, true);
        });
        // The MMQ fast paths: arm the launch token (not the attribute token), so
        // the opt-in succeeds and the launch itself is what fails.
        go!(&["launch:mmq_raw_nb"], || unsafe {
            launch_mmq_raw_nb_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 1, 8, 64, st, 8);
        });
        for (kd, token) in [
            (4, "launch:mmq_raw_wide_kd4"),
            (8, "launch:mmq_raw_wide_kd8"),
        ] {
            go!(&[token], || unsafe {
                launch_mmq_raw_wide_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, st, kd);
            });
        }
        for (kd, token) in [(4, "launch:mmq_raw_nt_kd4"), (8, "launch:mmq_raw_nt_kd8")] {
            go!(&[token], || unsafe {
                launch_mmq_raw_nt(5, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, st, kd);
            });
        }
        go!(&["launch:mmq_nt"], || unsafe {
            launch_mmq_nt(0, ctx.u(0), ctx.u(1), ctx.f(2), 2, 8, 64, 40, st);
        });
        // The two NB-BT launchers: `w_dsc`/`w_exp` selects the instantiation, and
        // the k-split reduce is its own token (the kernel's token returns before
        // it, so the reduce can only be reached alone).
        for (dsc, token) in [
            (false, "launch:mmq_raw_nb_bt"),
            (true, "launch:mmq_raw_nb_bt"),
        ] {
            go!(&[token], || unsafe {
                launch_mmq_raw_nb_bt_nt(
                    5,
                    ctx.u(0),
                    if dsc { ctx.u(1) } else { std::ptr::null() },
                    ctx.u(2),
                    ctx.u(3),
                    ctx.f(4),
                    1,
                    8,
                    64,
                    8,
                    st,
                    8,
                    ctx.f(5),
                    1,
                );
            });
        }
        go!(&["launch:mmq_raw_nb_bt_ksplit"], || unsafe {
            launch_mmq_raw_nb_bt_nt(
                5,
                ctx.u(0),
                ctx.u(1),
                ctx.u(2),
                ctx.u(3),
                ctx.f(4),
                1,
                8,
                64,
                16,
                st,
                8,
                ctx.f(5),
                2,
            );
        });
        for (exp, token) in [
            (false, "launch:mmq_raw_nb_bt_q6k"),
            (true, "launch:mmq_raw_nb_bt_q6k"),
        ] {
            go!(&[token], || unsafe {
                launch_mmq_raw_nb_bt_q6k_nt(
                    7,
                    ctx.u(0),
                    if exp { ctx.u(1) } else { std::ptr::null() },
                    ctx.u(2),
                    ctx.u(3),
                    ctx.u(4),
                    ctx.f(5),
                    1,
                    8,
                    64,
                    8,
                    212,
                    st,
                    8,
                    ctx.f(6),
                    1,
                );
            });
        }
        go!(&["launch:mmq_raw_nb_bt_q6k_ksplit"], || unsafe {
            launch_mmq_raw_nb_bt_q6k_nt(
                7,
                ctx.u(0),
                ctx.u(1),
                ctx.u(2),
                ctx.u(3),
                ctx.u(4),
                ctx.f(5),
                1,
                8,
                64,
                16,
                212,
                st,
                8,
                ctx.f(6),
                2,
            );
        });
        // ── the multi-token MMVQ family ─────────────────────────────────
        for (token, extra) in [
            ("launch:q4_k_q8_mmvq", 0),
            ("launch:q4_k_q8_mmvq_v2", 0),
            ("launch:q4_k_q8_mmvq_multi", 0),
            ("launch:q4_k_q8_mmvq_v2_multi", 0),
            ("launch:q5_k_q8_mmvq", 0),
            ("launch:q5_k_q8_mmvq_v2", 0),
            ("launch:q5_k_q8_mmvq_multi", 0),
            ("launch:q5_k_q8_mmvq_v2_multi", 0),
            ("launch:q4_0_q8_mmvq", 0),
            ("launch:q4_0_q8_mmvq_multi", 0),
            ("launch:q8_0_q8_mmvq", 0),
            ("launch:q8_0_q8_mmvq_multi", 0),
            ("launch:q6_k_q8_mmvq", 1),
            ("launch:q6_k_q8_mmvq_v2", 1),
            ("launch:q6_k_q8_mmvq_v2_pf", 1),
            ("launch:q6_k_q8_mmvq_multi", 1),
            ("launch:q6_k_q8_mmvq_v2_multi", 1),
            ("launch:q6_k_q8_mmvq_v2_dpl", 2),
            ("launch:q6_k_q8_mmvq_v2_pf_dpl", 2),
        ] {
            go!(&[token], || unsafe {
                let (w, a, o) = (ctx.u(0), ctx.u(1), ctx.f(2));
                match token {
                    "launch:q4_k_q8_mmvq" => launch_q4_k_q8_mmvq(w, a, o, 8, 64, 2, st),
                    "launch:q4_k_q8_mmvq_v2" => launch_q4_k_q8_mmvq_v2(w, a, o, 8, 64, 2, st),
                    "launch:q4_k_q8_mmvq_multi" => launch_q4_k_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                    "launch:q4_k_q8_mmvq_v2_multi" => {
                        launch_q4_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, st)
                    }
                    "launch:q5_k_q8_mmvq" => launch_q5_k_q8_mmvq(w, a, o, 8, 64, 2, st),
                    "launch:q5_k_q8_mmvq_v2" => launch_q5_k_q8_mmvq_v2(w, a, o, 8, 64, 2, st),
                    "launch:q5_k_q8_mmvq_multi" => launch_q5_k_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                    "launch:q5_k_q8_mmvq_v2_multi" => {
                        launch_q5_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, st)
                    }
                    "launch:q4_0_q8_mmvq" => launch_q4_0_q8_mmvq(w, a, o, 8, 64, 2, st),
                    "launch:q4_0_q8_mmvq_multi" => launch_q4_0_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                    "launch:q8_0_q8_mmvq" => launch_q8_0_q8_mmvq(w, a, o, 8, 64, 2, st),
                    "launch:q8_0_q8_mmvq_multi" => launch_q8_0_q8_mmvq_multi(w, a, o, 8, 64, 2, st),
                    "launch:q6_k_q8_mmvq" => launch_q6_k_q8_mmvq(w, a, o, 8, 64, 2, 210, st),
                    "launch:q6_k_q8_mmvq_v2" => launch_q6_k_q8_mmvq_v2(w, a, o, 8, 64, 2, 210, st),
                    "launch:q6_k_q8_mmvq_v2_pf" => {
                        launch_q6_k_q8_mmvq_v2_pf(w, a, o, 8, 64, 2, 210, st)
                    }
                    "launch:q6_k_q8_mmvq_multi" => {
                        launch_q6_k_q8_mmvq_multi(w, a, o, 8, 64, 2, 210, st)
                    }
                    "launch:q6_k_q8_mmvq_v2_multi" => {
                        launch_q6_k_q8_mmvq_v2_multi(w, a, o, 8, 64, 2, 210, st)
                    }
                    "launch:q6_k_q8_mmvq_v2_dpl" => {
                        launch_q6_k_q8_mmvq_v2_dpl(w, a, o, 8, 64, 2, 1, st)
                    }
                    _ => launch_q6_k_q8_mmvq_v2_pf_dpl(w, a, o, 8, 64, 2, 1, st),
                }
                let _ = extra;
            });
        }
        go!(&["launch:q8_0_p32_q8_mmvq"], || unsafe {
            launch_q8_0_p32_q8_mmvq(ctx.u(0), ctx.u(1), ctx.u(2), ctx.f(3), 8, 64, 2, st);
        });
        go!(&["launch:q8_0_p32_q8_mmvq_multi"], || unsafe {
            launch_q8_0_p32_q8_mmvq_multi(ctx.u(0), ctx.u(1), ctx.u(2), ctx.f(3), 8, 64, 2, st);
        });
        go!(&["launch:kv_move_rows"], || unsafe {
            launch_kv_move_rows(ctx.f(0), ctx.cf(1), 1, 0, 1, 64, st);
        });

        // ── the union assertion (rule 1: the expected set is a value, not a
        //    relation between two code paths) ─────────────────────────────
        let expected: BTreeSet<String> = rows.iter().map(|r| r.2.clone()).collect();
        let missing: Vec<&String> = expected.difference(&seen).collect();
        let extra: Vec<&String> = seen.difference(&expected).collect();
        assert!(
            missing.is_empty(),
            "the driver never reached {} audited <<< site(s): {missing:?}",
            missing.len()
        );
        assert!(
            extra.is_empty(),
            "the driver observed {} site(s) absent from the audit fixture: {extra:?}",
            extra.len()
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "no latch may survive the gate: a latched error reaching CudaState::sync is the bug"
        );
    }

    /// The required/fallback severity decision is in the helper, and the two
    /// kinds are distinguishable at the call site that matters. Device + gated.
    #[test]
    fn cuda_issue162_required_sites_set_the_sticky_opt_sites_do_not() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !gate_enabled() {
            return;
        }
        let s = device().unwrap();
        let ctx = Ctx::new(s);
        let st = ctx.stream;

        // A required site records the sticky that `execute_node` turns into Err.
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        {
            let _g = Arm::new("launch:add_f32");
            unsafe { launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, st) };
        }
        let msg = s
            .take_launch_failure()
            .expect("a required launch failure must leave a sticky record");
        assert!(
            msg.contains("launch:add_f32") && msg.contains("cudaErrorInvalidValue"),
            "the sticky must name the site and the error: {msg}"
        );
        assert!(
            s.take_launch_failure().is_none(),
            "draining must clear the sticky (a stale record would blame the next node)"
        );

        // A documented-fallback site names and clears, and sets no sticky.
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        let rc = {
            let _g = Arm::new("launch:fa_prefill_kv__f16_causal");
            unsafe {
                launch_fa_prefill_kv(
                    ctx.cf(0),
                    ctx.p(1),
                    ctx.p(2),
                    ctx.f(3),
                    ctx.ci32(4),
                    CAUSAL,
                    4,
                    2,
                    128,
                    0.125,
                    2,
                    crate::cuda::KV_LAYOUT_F16,
                    136,
                    st,
                )
            }
        };
        assert_eq!(rc, -1, "the fa-prefill fallback must refuse the launch");
        assert!(
            s.take_launch_failure().is_none(),
            "a documented-fallback site must not set the sticky"
        );
        assert_eq!(unsafe { minfer_site_fail_count() }, 1);
        assert_eq!(
            s.take_last_error(),
            0,
            "the fallback site cleared its latch"
        );
    }

    /// The positive control: with the knob off the same call launches for real
    /// and leaves no report and no latch. Without it, "the injected run reported"
    /// would not distinguish a working site from a site that always fails.
    #[test]
    fn cuda_issue162_positive_control_launches_cleanly() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !gate_enabled() {
            return;
        }
        let s = device().unwrap();
        let ctx = Ctx::new(s);
        let a = vec![1.0f32; 64];
        let b = vec![2.0f32; 64];
        for (i, src) in [&a, &b].iter().enumerate() {
            let e = unsafe {
                cudaMemcpy(
                    ctx.p(i),
                    src.as_ptr() as *const std::ffi::c_void,
                    64 * 4,
                    CUDA_MEMCPY_HOST_TO_DEVICE,
                )
            };
            assert_eq!(e, 0, "host fill failed: {}", cuda_error_name(e));
        }
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        unsafe { launch_add_f32(ctx.cf(0), ctx.cf(1), ctx.f(2), 64, ctx.stream) };
        assert_eq!(
            unsafe { minfer_site_fail_count() },
            0,
            "knob off: the launch must not report"
        );
        assert_eq!(
            s.take_launch_failure(),
            None,
            "knob off: no sticky required-launch failure"
        );
        assert_eq!(s.take_last_error(), 0, "knob off: no latch");
        s.sync();
        let mut out = vec![0.0f32; 64];
        let e = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ctx.p(2),
                64 * 4,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(e, 0, "readback failed: {}", cuda_error_name(e));
        assert_eq!(
            out,
            vec![3.0f32; 64],
            "the positive control must actually compute 1 + 2"
        );
    }

    /// The node-level consequence, and the acceptance's "no stale output" claim:
    /// a required launch failure inside a real op makes `execute_node` return
    /// `Err` naming the site. Device + gated.
    #[test]
    fn cuda_issue162_a_required_launch_failure_fails_the_node() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !gate_enabled() {
            return;
        }
        use crate::graph::alloc::GraphAllocator;
        use crate::graph::builder::GraphBuilder;
        use crate::graph::scheduler::BackendScheduler;

        let s = device().unwrap();
        let mut b = GraphBuilder::new();
        let x = b.input("x", [16, 1, 1, 1], crate::graph::DType::F32);
        let y = b.input("y", [16, 1, 1, 1], crate::graph::DType::F32);
        let z = b.add(x, y);
        b.output(z);
        let mut g = b.build();

        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = BackendScheduler::new();
        sched.assign_backends(&mut g, &alloc);
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &[1.0f32; 16]).unwrap();
        alloc.fill_input(&g, "y", &[2.0f32; 16]).unwrap();

        // Positive control: the node executes and produces 3.0.
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        sched.execute(&g, &mut alloc).unwrap();
        assert_eq!(alloc.copy_to_cpu(z).unwrap(), vec![3.0f32; 16]);

        // Injected: the required launch fails, so the op must not proceed on an
        // unwritten output.
        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        let err = {
            let _arm = Arm::new("launch:add_f32");
            sched
                .execute(&g, &mut alloc)
                .expect_err("a required launch failure must fail the node")
        };
        assert!(
            err.contains("launch:add_f32") && err.contains("add_f32"),
            "the node error must name the site: {err}"
        );
        assert_eq!(
            s.take_last_error(),
            0,
            "and the site's own latch must not reach CudaState::sync"
        );
        assert!(
            s.take_launch_failure().is_none(),
            "execute_node must drain the sticky even on its Err arm"
        );
    }

    /// The **Err** arm's drain, isolated: an f16 matmul's Rust wrapper turns the
    /// launcher's own `int` return into an `Err`, so `execute_node_inner` returns
    /// `Err` *with the sticky already set*. Without the unconditional drain the
    /// record would survive into the next `execute_node` and be blamed on it.
    /// Device + gated.
    #[test]
    fn cuda_issue162_the_err_arm_also_drains_the_sticky() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        if !gate_enabled() {
            return;
        }
        use crate::graph::alloc::GraphAllocator;
        use crate::graph::builder::GraphBuilder;
        use crate::graph::scheduler::BackendScheduler;

        let s = device().unwrap();
        // od=8, id=64: `id % 8 == 0` selects the vectorized f16 site.
        let (od, id) = (8usize, 64usize);
        let wb = vec![0u8; od * id * 2];
        let mut wt = Tensor::from_data(TensorType::F16, &[id as i64, od as i64, 1, 1], wb.clone());
        wt.name = "issue162_f16_w".to_string();
        s.register_weight(&wt.name, &wb);

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id, 1, 1, 1], crate::graph::DType::F32);
        let m = b.matmul(x, &wt, None);
        b.output(m);
        let mut g = b.build();
        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = BackendScheduler::new();
        sched.assign_backends(&mut g, &alloc);
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &vec![0.0f32; id]).unwrap();

        let _ = s.take_last_error();
        unsafe { minfer_site_fail_reset() };
        let err = {
            let _arm = Arm::new("launch:f16_f32_matmul_vec");
            sched
                .execute(&g, &mut alloc)
                .expect_err("the f16 matmul launcher's Err must reach the scheduler")
        };
        assert!(
            err.contains("f16 matmul"),
            "the node error is the launcher's own: {err}"
        );
        assert!(
            s.take_launch_failure().is_none(),
            "the Err arm must drain the sticky too, or the NEXT node would be blamed for this \
             launch (issue #162)"
        );
        assert_eq!(s.take_last_error(), 0);
    }
}
