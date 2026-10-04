//! The positive control: with the knob off the same launch runs for real, reports nothing, latches nothing and computes 1 + 2.
//!
//! Split out of `src/cuda/issue162_tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
