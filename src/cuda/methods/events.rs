// events, async staging and the API-error latch (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // #162: the sticky required-launch failure and the ordered launch-site
    // history. `minfer_launch_ok` sets the sticky for a REQUIRED site;
    // `CudaBackend::execute_node` drains it and returns an `Err` naming the site,
    // so one Rust-side check covers every launcher. `minfer_launch_ok_opt` (a
    // documented fallback) never sets it. The history records every named launch
    // failure so the #162 gate can see more than one site per call.
    pub(crate) fn minfer_launch_fail_pending() -> i32;
    pub(crate) fn minfer_launch_fail_site() -> *const std::os::raw::c_char;
    pub(crate) fn minfer_launch_fail_name() -> *const std::os::raw::c_char;
    pub(crate) fn minfer_launch_fail_code() -> i32;
    pub(crate) fn minfer_launch_fail_clear();
}

impl CudaState {
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
}
