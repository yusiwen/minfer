// The CUDA runtime/driver FFI declarations (pre-split `src/cuda.rs`).
//
// The `extern "C"` items are `pub(crate)` so `cuda` and its descendants --
// the `methods/*` family files -- can reach them; `src/cuda.rs` re-exports
// the test-only introspection symbols under their `crate::cuda::` paths.

use super::*;

// ─── FFI declarations for CUDA runtime API ────────────────────
extern "C" {
    pub(crate) fn dlopen(
        filename: *const std::ffi::c_char,
        flag: std::ffi::c_int,
    ) -> *mut std::ffi::c_void;
    pub(crate) fn cudaSetDevice(device: i32) -> i32;
    pub(crate) fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    pub(crate) fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    pub(crate) fn cudaHostAlloc(ptr: *mut *mut std::ffi::c_void, size: usize, flags: i32) -> i32;
    pub(crate) fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    // Issue #188: a **per-backend** stream is created with
    // `cudaStreamNonBlocking` so it does not take part in the legacy default
    // stream's implicit global synchronization. Without that flag every
    // explicit stream implicitly synchronizes with the null stream, and a
    // host-side blocking `cudaMemcpy` (weight registration, readback) on one
    // thread would serialize — and, inside another thread's capture window,
    // invalidate — every engine's stream.
    pub(crate) fn cudaStreamCreateWithFlags(stream: *mut *mut std::ffi::c_void, flags: u32) -> i32;
    pub(crate) fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    // Issue #188: is `stream` currently inside a capture window? The only reader is
    // `CudaState::stream_is_capturing` below, whose only caller is the #188 probe in
    // `graph/cuda_backend/tests.rs`; the production capture bookkeeping is the
    // per-instance `CudaBackend::capturing` field. (The older note here claimed a
    // registration-path refusal that no longer exists — corrected by #238.)
    /// Test-only (#238): driven by `graph::cuda_backend::tests::capture::cuda_capture_abort_on_error`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn cudaStreamIsCapturing(stream: *mut std::ffi::c_void, status: *mut i32) -> i32;
    // F5 (#58): events, the synchronization primitive the split boundary's async
    // staging copies need. `cudaEventRecord` marks a point on the stream;
    // `cudaStreamWaitEvent` makes a later consumer wait on it **without blocking
    // the host**; `cudaEventSynchronize` is the host-side wait and is the one
    // documented synchronization point of a device→host staging copy.
    pub(crate) fn cudaEventCreate(event: *mut *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaEventRecord(
        event: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn cudaEventSynchronize(event: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaEventDestroy(event: *mut std::ffi::c_void) -> i32;
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
    pub(crate) fn cudaStreamWaitEvent(
        stream: *mut std::ffi::c_void,
        event: *mut std::ffi::c_void,
        flags: u32,
    ) -> i32;
    pub(crate) fn cudaGetDeviceCount(count: *mut i32) -> i32;
    pub(crate) fn cudaGetLastError() -> i32;
    pub(crate) fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    pub(crate) fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    // Issue #122: the *name* of a CUDA error code, so a failed memory query can say
    // `cudaErrorIllegalAddress (700)` instead of a bare number (or nothing at all).
    pub(crate) fn cudaGetErrorName(error: i32) -> *const std::os::raw::c_char;
    pub(crate) fn cudaGetDeviceProperties(prop: *mut CudaDevicePropBuf, device: i32) -> i32;
    // T2 device-adaptation queries (plan §6): smem feasibility for the BT
    // tile config. The externs query the CURRENT device (R2 fix).
    // cuda_shared_per_sm stays C-side only: per-block optin <= per-SM on
    // every arch, so the per-block check below subsumes it.
    pub(crate) fn cuda_shared_per_block_optin() -> i32;
    pub(crate) fn cuda_mmq_smem_bytes() -> i32;
    // CUDA Graph APIs
    pub(crate) fn cudaStreamBeginCapture(stream: *mut std::ffi::c_void, mode: i32) -> i32;
    pub(crate) fn cudaStreamEndCapture(
        stream: *mut std::ffi::c_void,
        graph: *mut *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn cudaGraphInstantiate(
        exec: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        error_node: *mut std::ffi::c_void,
        log_buf: *mut u8,
        buf_size: usize,
    ) -> i32;
    pub(crate) fn cudaGraphLaunch(
        exec: *mut std::ffi::c_void,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // A `cudaGraph_t` (the result of `cudaStreamEndCapture`) and a
    // `cudaGraphExec_t` (the result of `cudaGraphInstantiate`) are different
    // handle types with different destroy calls. Passing an exec to
    // `cudaGraphDestroy` returns `cudaErrorInvalidValue` and leaks the exec
    // (issue #145).
    pub(crate) fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
    pub(crate) fn cudaGraphExecDestroy(exec: *mut std::ffi::c_void) -> i32;
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
