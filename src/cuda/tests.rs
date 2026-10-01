//! `#[cfg(test)] mod tests` for `src/cuda.rs` — extracted so a non-test build
//! does not parse it. See the parent module for the docs.
//!
//! [#239] moved here the test-only FFI declarations and the test-only
//! `CudaState` accessors whose callers are the sibling `cuda::*_tests` modules.
//! They are `pub(super)` — the narrowest visibility that reaches a sibling
//! module under `cuda`.
//!
//! [#239]: https://github.com/yusiwen/minfer/issues/239

use super::*;

// ─── #145 / #147 / #162 device-gate FFI declarations ────────────────────────
//
// The C++ sites in `cuda_kernels.cu` write these; only the device gates read
// them. Declaring them here (rather than in the production `extern "C"` block)
// keeps every test-only declaration out of a non-test build.
extern "C" {
    /// #145 test injection: latch a real `cudaErrorInvalidValue` on purpose
    /// (request the device limit + 4096 B) without clearing it.
    ///
    /// Test-only (#239): driven by
    /// `cuda::issue145_tests::cuda_sync_surfaces_a_latched_error_as_latched`.
    pub(super) fn cuda_test_latch_oversized_smem() -> i32;

    /// #147/#162 site-failure introspection: the last dynamic-smem / launch
    /// failure a hardened C++ site named. `kind`: 1 = attribute, 2 = launch,
    /// 3 = a latched error found before a launch.
    ///
    /// Test-only (#239): `count`/`kind`/`code` are read by the `inject` helper of
    /// `cuda::issue147_tests` (driven by
    /// `cuda_issue147_attribute_sites_name_the_call_and_refuse_the_launch` and
    /// `cuda_issue147_launch_sites_name_the_call_and_refuse_the_launch`) and by
    /// the `run` helper of `cuda::issue162_tests` (driven by
    /// `cuda_issue162_required_sites_set_the_sticky_opt_sites_do_not`).
    pub(super) fn minfer_site_fail_count() -> i32;
    /// Test-only (#239): read by `cuda::issue147_tests::inject` and
    /// `cuda::issue162_tests::run` (see `minfer_site_fail_count`).
    pub(super) fn minfer_site_fail_kind() -> i32;
    /// Test-only (#239): read by `cuda::issue147_tests::inject` and
    /// `cuda::issue162_tests::run`.
    pub(super) fn minfer_site_fail_code() -> i32;

    /// Test-only (#239): read by `cuda::issue147_tests::inject`.
    pub(super) fn minfer_site_fail_bytes() -> i32;
    /// Test-only (#239): read by `cuda::issue147_tests::inject`.
    pub(super) fn minfer_site_fail_limit() -> i32;
    /// Test-only (#239): read by `cuda::issue147_tests::inject`.
    pub(super) fn minfer_site_fail_site() -> *const std::os::raw::c_char;
    /// Test-only (#239): read by `cuda::issue147_tests::inject`.
    pub(super) fn minfer_site_fail_message() -> *const std::os::raw::c_char;

    /// Test-only (#239): cleared by `cuda::issue147_tests::inject` and by the
    /// `run` helper of `cuda::issue162_tests`, so each gate reads only its own
    /// call's report.
    pub(super) fn minfer_site_fail_reset();

    /// #162: the ordered launch-site history (`minfer_launch_ok` records it).
    ///
    /// Test-only (#239): read by the `run` helper of `cuda::issue162_tests`
    /// (driven by `cuda_issue162_required_sites_set_the_sticky_opt_sites_do_not`
    /// and `cuda_issue162_a_required_launch_failure_fails_the_node`).
    pub(super) fn minfer_site_hist_len() -> i32;
    /// Test-only (#239): read by `cuda::issue162_tests::run`.
    pub(super) fn minfer_site_hist_site(i: i32) -> *const std::os::raw::c_char;
    /// Test-only (#239): read by `cuda::issue162_tests::run`.
    pub(super) fn minfer_site_hist_name(i: i32) -> *const std::os::raw::c_char;
    /// Test-only (#239): read by `cuda::issue162_tests::run`.
    pub(super) fn minfer_site_hist_msg(i: i32) -> *const std::os::raw::c_char;
}

/// Issue #145: the process-wide count of latched API errors `sync` has reported.
///
/// Test-only (#239): driven by
/// `cuda::issue145_tests::cuda_sync_surfaces_a_latched_error_as_latched`.
pub(super) fn latched_api_error_count() -> u64 {
    LATCHED_API_ERRORS.load(Ordering::Relaxed)
}

impl CudaState {
    /// Clear and return the CUDA per-thread "last error" latch (`cudaGetLastError`).
    ///
    /// Used by the device gates that must assert a call left **no** error behind
    /// (`cuda_graph_exec_destroy_leaves_no_latched_error`, #145) and by the
    /// #147/#162 gates to start from a clean latch.
    ///
    /// Test-only (#239): driven by
    /// `cuda::issue145_tests::cuda_graph_exec_destroy_leaves_no_latched_error` and
    /// the sibling #147/#162 gates.
    pub(super) fn take_last_error(&self) -> i32 {
        unsafe { cudaGetLastError() }
    }
}

impl StreamScratch {
    /// The current stream's `(ptr, size)`, or `(null, 0)` when this stream has
    /// never grown the slot.
    ///
    /// Test-only (#240): [#239] deferred this move because the bucket-A legacy
    /// wrappers `upload_hidden` / `upload_positions` / `download_logits` /
    /// `get_positions_buf` still called it and were still compiled; [#240]
    /// deleted them, so the only remaining caller is `cuda::d35_probe_tests`
    /// (`st.buf_q8_decode.slot()`). `pub(super)` is the visibility the rest of
    /// this file uses for the sibling `cuda::*_tests` modules.
    ///
    /// [#239]: https://github.com/yusiwen/minfer/issues/239
    /// [#240]: https://github.com/yusiwen/minfer/issues/240
    pub(super) fn slot(&self) -> (CudaPtr, usize) {
        let key = current_stream_key();
        self.map
            .lock()
            .unwrap()
            .get(&key)
            .copied()
            .unwrap_or((CudaPtr(std::ptr::null_mut()), 0))
    }
}
