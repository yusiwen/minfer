//! `#[cfg(test)] mod issue223_tests` for `src/cuda.rs` — the **eager**
//! prefill-GEMM dynamic-smem pre-warm, restored by #223 as a *runtime*
//! guarantee.
//!
//! [#218] made the invariant explicit and gated but left production upholding it
//! only through the lazy per-launch opt-in plus three emergent mechanisms (the
//! 3-run capture warmup, `cudaStreamCaptureModeThreadLocal`, the
//! per-instantiation cache). [#223] put the guarantee back where [#188] deleted
//! it: `CudaState::try_new` drives the production
//! `gemm_prefill_smem_prewarm_one` for every launchable `(tm, ks, af32)` once
//! per process, at the earliest point in the process, where no stream — and so
//! no capture window — can exist. The lazy path stays as defence in depth.
//!
//! The gate here is the runtime guarantee itself, not the lazy path's ability
//! to recover: **immediately after context creation and before any kernel
//! launch**, every launchable >48 KiB instantiation must already read back opted
//! in from the device (`cudaFuncGetAttributes().maxDynamicSharedSizeBytes`).
//!
//! **Why it cannot pass vacuously.** It runs in a **fresh process** (the
//! `test_child` harness), whose body calls `CudaState::init()` and then
//! immediately reads the device — it launches nothing, so "before any launch" is
//! true by construction. The second child is the **control**: the same fresh
//! process with the documented `MINFER_NO_GEMM_PREWARM=1` control, which asserts
//! the opposite (the >48 KiB instantiations are *not* opted in), so the
//! pre-warmed arm's `opted_in == 1` cannot be a property of the read-back
//! itself. The named load-bearing instantiation is
//! `gemm_f16_nt_kernel_t<128,64,false>` at 57344 B (the same one the #218 gates
//! use); the whole launchable set is checked, so a pre-warm that drops **any**
//! one of them goes red.
//!
//! **Mutation evidence (rule 3).** Remove one `(tm, ks, af32)` from
//! `GEMM_PREWARM_SET` in `src/cuda.rs` and run this gate: the child reports
//! `gemm_f16_nt_kernel_t<…> (… B > 48 KiB) must already read as opted in` with
//! `left: 0, right: 1`. The #218 coverage gate stays green under that mutation
//! (the lazy path opts the dropped instantiation in when the coverage gate
//! launches it), which is exactly why this gate exists — the skip would
//! otherwise be invisible. The `template <typename K>` cache regression, by
//! contrast, makes both this gate and the #218 coverage read-back go red.
//!
//! [#188]: https://github.com/yusiwen/minfer/issues/188
//! [#218]: https://github.com/yusiwen/minfer/issues/218
//! [#223]: https://github.com/yusiwen/minfer/issues/223
use super::*;

/// The launchable prefill-GEMM set, enumerated **independently of** the
/// production pre-warm's own list (`GEMM_PREWARM_SET` in `cuda.rs`): the gate
/// must be able to observe an instantiation the pre-warm dropped, so it cannot
/// share that list. The same 3×2×2 fatbin set the C++ `MINFER_GEMM_OPTIN_SET`
/// lists.
const LAUNCHABLE: [(i32, i32, i32); 12] = [
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

/// The instantiation the mutation arm targets and the assertion message names.
const KERNEL: &str = "gemm_f16_nt_kernel_t<128,64,false>";

/// The eager pre-warm is a **runtime guarantee**: after `CudaState::init()` and
/// before any kernel launch, every launchable >48 KiB instantiation reads back
/// opted in. Two fresh processes: the pre-warmed one (default) asserts the
/// guarantee; the control (`MINFER_NO_GEMM_PREWARM=1`) asserts its negation, so
/// the gate is not asserting something the read-back always answers.
#[test]
fn cuda_prefill_smem_prewarm_opts_in_every_launchable_instantiation_before_any_launch() {
    const FILTER: &str =
        "cuda_prefill_smem_prewarm_opts_in_every_launchable_instantiation_before_any_launch";
    match super::test_child::child_phase().as_deref() {
        Some("prewarm") => prewarm_child(),
        Some("control") => control_child(),
        _ => {
            let child = super::test_child::run_self(FILTER, "prewarm", &[]);
            child.verdict("the eager pre-warm's runtime guarantee");
            let control =
                super::test_child::run_self(FILTER, "control", &[("MINFER_NO_GEMM_PREWARM", "1")]);
            control.verdict("the pre-warm-off control");
        }
    }
}

/// The pre-warmed child: `init()` (the first device call, which runs the
/// pre-warm) then the read-back, with nothing launched in between.
fn prewarm_child() {
    CudaState::init();
    if CudaState::get().is_none() {
        super::test_child::child_skip("no CUDA device");
    }
    // By construction: `try_new` is the only thing this process has run on the
    // device, and the only code that can open a capture window is a
    // `CudaBackend`, which does not exist yet.
    assert_eq!(
        unsafe { gemm_smem_optin_in_capture_count() },
        0,
        "no capture window can exist at context creation, so no opt-in can have \
         happened inside one"
    );
    assert!(
        !gemm_prewarm_disabled(),
        "this child is the pre-warmed arm; MINFER_NO_GEMM_PREWARM must be unset"
    );
    let limit = unsafe { gemm_prefill_smem_limit() };
    assert!(limit >= 48 * 1024, "queried opt-in limit {limit} B");
    assert!(
        unsafe { gemm_smem_need(128, 64, 0) } == 57344,
        "the load-bearing instantiation's single-source need must be 57344 B"
    );
    let mut checked = 0usize;
    for (tm, ks, af32) in LAUNCHABLE {
        let need = unsafe { gemm_smem_need(tm, ks, af32) };
        if need <= 48 * 1024 {
            continue; // the default cap admits it; there is no attribute to set
        }
        if need > limit as usize {
            // The device cannot admit it; the pre-warm must have skipped it
            // without calling the attribute (named by `minfer_smem_optin`).
            assert_eq!(
                unsafe { gemm_smem_opted_in(tm, ks, af32) },
                0,
                "gemm_f16_nt_kernel_t<{tm},{ks},{}> needs {need} B > the {limit} B device \
                 limit, so it can never read as opted in",
                af32 != 0
            );
            continue;
        }
        let name = if (tm, ks, af32) == (128, 64, 0) {
            KERNEL.to_string()
        } else {
            format!("gemm_f16_nt_kernel_t<{tm},{ks},{}>", af32 != 0)
        };
        assert_eq!(
            unsafe { gemm_smem_opted_in(tm, ks, af32) },
            1,
            "immediately after context creation and before any kernel launch, {name} \
             ({need} B > 48 KiB) must already read as opted in — the eager pre-warm at \
             CudaState::try_new is the runtime guarantee this gate exists for"
        );
        if (tm, ks, af32) == (128, 64, 0) {
            checked += 1;
        }
    }
    assert!(
        checked == 1,
        "the gate must have checked the load-bearing {KERNEL}"
    );
    super::test_child::child_ok();
}

/// The control child: the same fresh process, but the pre-warm is skipped, so
/// nothing can have opted the >48 KiB instantiations in before a launch. This is
/// what makes the pre-warmed arm non-vacuous — the read-back is capable of
/// answering 0.
fn control_child() {
    CudaState::init();
    if CudaState::get().is_none() {
        super::test_child::child_skip("no CUDA device");
    }
    assert!(
        gemm_prewarm_disabled(),
        "this child is the control arm; MINFER_NO_GEMM_PREWARM=1 must be set"
    );
    let limit = unsafe { gemm_prefill_smem_limit() };
    let mut checked = 0usize;
    for (tm, ks, af32) in LAUNCHABLE {
        let need = unsafe { gemm_smem_need(tm, ks, af32) };
        if need <= 48 * 1024 || need > limit as usize {
            continue;
        }
        assert_eq!(
            unsafe { gemm_smem_opted_in(tm, ks, af32) },
            0,
            "control (MINFER_NO_GEMM_PREWARM=1): gemm_f16_nt_kernel_t<{tm},{ks},{}> \
             ({need} B > 48 KiB) must NOT be opted in before any launch; if it were, the \
             pre-warmed arm's assertion would be vacuous",
            af32 != 0
        );
        checked += 1;
    }
    assert!(
        checked >= 1,
        "the control must have checked at least one >48 KiB instantiation"
    );
    super::test_child::child_ok();
}
