//! `#[cfg(test)] mod issue145_tests` for `src/cuda.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
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
/// exceeds the 48 KiB default must be admitted by the **production** opt-in —
/// `gemm_smem_optin`, the per-instantiation function `launch_gemm_f16` itself
/// calls (driven here through the `gemm_prefill_smem_optin_one_for_test` seam so
/// arbitrary combinations can be selected) — and the device's own read-back
/// (`cudaFuncGetAttributes().maxDynamicSharedSizeBytes`) must agree. A request
/// over the device limit must be skipped, never called, and must not read back
/// opted in. Device gate.
///
/// #218: the pre-#218 version called `gemm_prefill_smem_init`, the **eager
/// sweep** #188 had orphaned (no production caller). The sweep is gone; this
/// gate now drives the lazy path production actually uses, and its name and
/// claim say so. The production *forward* arms live in `issue218_tests`:
/// `cuda_prefill_smem_optin_is_done_by_production` (a real prefill opts a
/// >48 KiB instantiation in, non-vacuously) and
/// `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` (a >48 KiB
/// captured prefill replays bitwise, with the opt-in shown to happen before the
/// window opened).
#[test]
fn cuda_prefill_smem_lazy_optin_admits_every_launchable_instantiation() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let s = device().unwrap();
    let _ = s.take_last_error(); // this gate must not inherit a latch
    let limit = unsafe { gemm_prefill_smem_limit() };
    assert!(limit >= 48 * 1024, "queried opt-in limit {limit} B");
    let mut covered = 0usize;
    for tm in [64, 128, 256] {
        for ks in [32, 64] {
            for af32 in [false, true] {
                let need = unsafe { gemm_smem_need(tm, ks, af32 as i32) };
                if need <= 48 * 1024 {
                    continue; // the default cap admits it; no opt-in needed
                }
                let admitted = unsafe {
                    gemm_prefill_smem_optin_one_for_test(tm, ks, af32 as i32, s.stream())
                };
                assert_ne!(
                    admitted, -1,
                    "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> must be compiled in"
                );
                let opted = unsafe { gemm_smem_opted_in(tm, ks, af32 as i32) } == 1;
                if need > limit as usize {
                    assert_eq!(
                        admitted, 0,
                        "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> needs {need} B > the \
                         {limit} B device limit and must be refused, not launched"
                    );
                    assert!(
                        !opted,
                        "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> needs {need} B > the \
                         {limit} B device limit but reads back as opted in"
                    );
                    continue;
                }
                assert_eq!(
                    admitted, 1,
                    "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> needs {need} B and the device \
                     admits {limit} B, but the production opt-in refused it"
                );
                assert!(
                    opted,
                    "gemm_f16_nt_kernel_t<{tm},{ks},{af32}> needs {need} B and the device \
                     admits {limit} B, but cudaFuncGetAttributes reports its \
                     maxDynamicSharedSizeBytes below that — the >48 KiB prefill launch \
                     would fail"
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
        s.take_last_error(),
        0,
        "the opt-in path must clear every latch it takes"
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
