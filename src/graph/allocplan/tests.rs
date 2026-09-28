//! `#[cfg(test)] mod tests` for `src/graph/allocplan.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// Issue #122's gate: a failed device query must never become a zero budget.
///
/// The mutation this pins is the old
/// `device_memory().0 / 4 * 3` on a discarded return code: with `free` left at 0 by
/// a failed `cudaMemGetInfo`, that produced `Some(0)` and the E4 gate then refused
/// every later device allocation with "exceeds the 0 byte budget (0 MiB)".
#[test]
fn a_failed_device_query_is_not_a_zero_budget() {
    let failed = DeviceMemory::QueryFailed {
        code: 700,
        name: "cudaErrorIllegalAddress".to_string(),
    };
    let d = budget_decision(None, &failed);
    assert_ne!(
        d.budget,
        Some(0),
        "a failed query must not masquerade as a full device: {d:?}"
    );
    // The fallback is weights-only accounting (unbounded activations), so the gate
    // cannot refuse from a number that was never measured.
    assert_eq!(d.budget, Some(usize::MAX), "{d:?}");
    let note = d.note.expect("the fallback must carry its reason");
    assert!(note.contains("cudaErrorIllegalAddress"), "{note}");
    assert!(note.contains("700"), "the note must name the code: {note}");
    assert!(
        !note.contains("0 byte budget") && !note.contains("0 MiB"),
        "the note must name the cause, not a fabricated measurement: {note}"
    );
    // And the query outcome itself still exposes the real failure.
    assert_eq!(failed.free_bytes(), None);
    assert!(failed
        .failure_note("cudaMemGetInfo")
        .unwrap()
        .contains("cudaErrorIllegalAddress"));
}

/// The happy path is unchanged: a reported read keeps the exact pre-#122 default.
#[test]
fn a_reported_free_read_keeps_the_three_quarters_default() {
    let mem = DeviceMemory::Reported {
        free: 4 << 20,
        total: 8 << 20,
    };
    let d = budget_decision(None, &mem);
    assert_eq!(d.budget, Some(3 << 20));
    assert_eq!(d.note, None, "a measurement needs no note");
    assert_eq!(mem.free_bytes(), Some(4 << 20));
    assert_eq!(mem.failure_note("cudaMemGetInfo"), None);
    // Rounding is the pre-existing integer division, not a new formula.
    let odd = DeviceMemory::Reported {
        free: 4_000_001,
        total: 8 << 20,
    };
    assert_eq!(budget_decision(None, &odd).budget, Some(3_000_000));
}

/// A *measured* zero is a real zero and must still refuse: that is "the device is
/// full", which is exactly the case the gate exists for.
#[test]
fn a_measured_zero_free_read_is_still_a_zero_budget() {
    let full = DeviceMemory::Reported {
        free: 0,
        total: 8 << 20,
    };
    let d = budget_decision(None, &full);
    assert_eq!(d.budget, Some(0));
    assert_eq!(d.note, None, "a real measurement carries no excuse");
}

/// No device state is unbounded and silent, as before; an explicit override wins.
#[test]
fn no_device_state_and_an_explicit_budget_are_unchanged() {
    let d = budget_decision(None, &DeviceMemory::NoDevice);
    assert_eq!(d.budget, None);
    assert_eq!(d.note, None);
    // An explicit budget is the caller's number, taken as given on every outcome.
    for mem in [
        DeviceMemory::NoDevice,
        DeviceMemory::Reported { free: 0, total: 1 },
        DeviceMemory::QueryFailed {
            code: 700,
            name: "cudaErrorIllegalAddress".to_string(),
        },
    ] {
        assert_eq!(budget_decision(Some(4096), &mem).budget, Some(4096));
        assert_eq!(budget_decision(Some(4096), &mem).note, None);
    }
}

#[test]
fn the_ladder_rounds_up_without_wasting_a_step() {
    // Below the grain: powers of two, with a 1 KiB floor.
    assert_eq!(class_size(0), CLASS_MIN);
    assert_eq!(class_size(1), CLASS_MIN);
    assert_eq!(class_size(CLASS_MIN), CLASS_MIN);
    assert_eq!(class_size(CLASS_MIN + 1), 512);
    assert_eq!(class_size(1000), 1024);
    assert_eq!(class_size(CLASS_GRAIN), CLASS_GRAIN);
    // Above it: multiples of the grain, so the waste is under one step.
    assert_eq!(class_size(CLASS_GRAIN + 1), CLASS_GRAIN * 2);
    assert_eq!(
        class_size(14336),
        16384,
        "896 x 16 is one step over 3 grains"
    );
    for elems in [1usize, 257, 4095, 4097, 200_000, 1_835_008, 4_194_305] {
        let c = class_size(elems);
        assert!(c >= elems, "{c} < {elems}");
        assert!(
            c - elems < CLASS_GRAIN,
            "{elems} -> {c} wastes {} elements",
            c - elems
        );
    }
}

#[test]
fn two_shapes_in_one_class_share_a_buffer() {
    // 896 x 16 = 14336 and 896 x 17 = 15232 round to the same class, so the second
    // interval reuses the first one's buffer once it is free (the "no silent growth"
    // acceptance: the pool does not keep both).
    let a = 896 * 16;
    let b = 896 * 17;
    assert_eq!(class_size(a), class_size(b));
    let plan = AllocPlan::plan(&[(a, 0, 1), (b, 2, 3)]);
    assert_eq!(plan.buffers, 1, "{plan:?}");
    assert_eq!(plan.reused, 1);
    assert_eq!(plan.reserved_bytes, class_bytes(class_size(a)));
    // Overlapping lifetimes cannot share, so both are reserved.
    let plan = AllocPlan::plan(&[(a, 0, 5), (b, 2, 3)]);
    assert_eq!(plan.buffers, 2, "{plan:?}");
    assert_eq!(plan.reused, 0);
    assert_eq!(plan.reserved_bytes, 2 * class_bytes(class_size(a)));
}

#[test]
fn the_plan_is_deterministic_and_order_independent() {
    let a = 4096;
    let b = 8192;
    let one = AllocPlan::plan(&[(a, 0, 2), (b, 1, 3), (a, 4, 5)]);
    let two = AllocPlan::plan(&[(a, 0, 2), (b, 1, 3), (a, 4, 5)]);
    assert_eq!(one, two);
    // Intervals arriving out of step order are still placed in step order, so the
    // plan does not depend on how the caller enumerated them.
    let shuffled = AllocPlan::plan(&[(a, 4, 5), (b, 1, 3), (a, 0, 2)]);
    assert_eq!(shuffled.buffers, one.buffers);
    assert_eq!(shuffled.reused, one.reused);
    assert_eq!(shuffled.reserved_bytes, one.reserved_bytes);
}

#[test]
fn an_empty_plan_is_free() {
    let plan = AllocPlan::plan(&[]);
    assert_eq!(plan.buffers, 0);
    assert_eq!(plan.reserved_bytes, 0);
    assert_eq!(plan.live_peak_bytes, 0);
}

#[test]
fn the_live_peak_is_not_the_reserved_total() {
    // Three intervals of the same class, one at a time: the pool holds one buffer,
    // and the live peak is that one buffer too.
    let plan = AllocPlan::plan(&[(4096, 0, 0), (4096, 1, 1), (4096, 2, 2)]);
    assert_eq!(plan.buffers, 1);
    assert_eq!(plan.live_peak_bytes, class_bytes(4096));
    assert_eq!(plan.reserved_bytes, class_bytes(4096));
    assert_eq!(plan.reused, 2);
    // Two different classes alive at once: the peak is their sum, and it is the same
    // as the reserved total (nothing to recycle yet).
    let plan = AllocPlan::plan(&[(4096, 0, 1), (8192, 0, 1)]);
    assert_eq!(plan.live_peak_bytes, plan.reserved_bytes);
}
