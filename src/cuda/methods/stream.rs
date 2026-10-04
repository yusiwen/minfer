// stream lifecycle (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

impl CudaState {
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
    /// Test-only (#238): driven by `graph::cuda_backend::tests::capture::cuda_capture_abort_on_error`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn stream_is_capturing(stream: *mut std::ffi::c_void) -> bool {
        let mut status: i32 = 0; // cudaStreamCaptureStatusNone == 0
        let err = unsafe { cudaStreamIsCapturing(stream, &mut status) };
        err == 0 && status != 0
    }
}
