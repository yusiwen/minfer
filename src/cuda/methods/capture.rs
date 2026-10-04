// CUDA-graph capture / replay (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

impl CudaState {
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
}
