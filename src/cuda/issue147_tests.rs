//! `#[cfg(test)] mod issue147_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
// #239: declared in `cuda::tests` now (test-only FFI / accessors).
use super::tests::{
    minfer_site_fail_bytes, minfer_site_fail_code, minfer_site_fail_count, minfer_site_fail_kind,
    minfer_site_fail_limit, minfer_site_fail_message, minfer_site_fail_reset,
    minfer_site_fail_site,
};

/// `minfer_site_fail_kind` values (`src/cuda/kernels/*.cu`).
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
fn assert_attr_failure(s: &CudaState, site: &str, kernel_frag: &str, call: impl FnOnce() -> i32) {
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
fn assert_launch_failure(s: &CudaState, site: &str, kernel_frag: &str, call: impl FnOnce() -> i32) {
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
