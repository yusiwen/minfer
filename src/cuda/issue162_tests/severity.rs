//! Required vs documented-fallback sites: only a required launch failure leaves the sticky record `execute_node` turns into an `Err`.
//!
//! Split out of `src/cuda/issue162_tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
