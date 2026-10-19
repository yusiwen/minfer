//! `#[cfg(test)] mod tests` for `src/testfail.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

/// #171 acceptance: with the switch unset the seam is a **no-op** — this is
/// the property the mutation check breaks (make `requested` ignore the
/// environment) to prove the guards are real.
#[test]
fn the_seam_is_off_by_default() {
    assert!(
        std::env::var(ENV).is_err(),
        "{ENV} is set in the default suite; the seam must be off"
    );
    assert!(!requested("all"));
    assert!(!requested("forward_batch"));
    assert!(!requested("alloc_in_pool"));
    assert!(!requested("execute_node"));
    assert!(guard("forward_batch").is_ok());
    assert!(guard("alloc_in_pool").is_ok());
    assert!(guard("execute_node").is_ok());
}

/// The matcher is exact and comma-separated (#147's rule, moved here from
/// `cuda.rs`): no substring surprises, whitespace tolerated, `all` is the
/// only wildcard.
#[test]
fn the_matcher_is_exact_and_comma_separated() {
    assert!(injection_names_site("all", "attr:mmq_nt"));
    assert!(injection_names_site(
        "launch:gemm_f16_f16,launch:mmq",
        "launch:mmq"
    ));
    assert!(injection_names_site(" launch:mmq ", "launch:mmq"));
    assert!(!injection_names_site("launch:gemm_f16_f16", "launch:gemm"));
    assert!(!injection_names_site("small", "attr:mmq_nt"));
    assert!(!injection_names_site("", "attr:mmq_nt"));
    assert!(!injection_names_site("attr:mmq", "attr:mmq_nt"));
    assert!(!injection_names_site("attr:mmq_nt_extra", "attr:mmq_nt"));
    assert!(!injection_names_site(
        "destroy:graph_destroy_extra",
        "destroy:graph_destroy"
    ));
    assert!(injection_names_site(
        "all,destroy:graph_destroy",
        "destroy:graph_destroy"
    ));
    assert!(injection_names_site("all", "launch:gemm_f16_f16"));
}

/// The observation half counts per site, ignores other sites, and resets.
/// This is the gate the counter mutation breaks.
#[test]
fn the_observation_counter_counts_only_what_ran() {
    reset_checked();
    assert_eq!(checked("alloc_in_pool"), 0);
    note_checked("alloc_in_pool");
    note_checked("alloc_in_pool");
    note_checked("execute_node");
    assert_eq!(checked("alloc_in_pool"), 2);
    assert_eq!(checked("execute_node"), 1);
    assert_eq!(checked("forward_batch"), 0, "an un-noted site stays at 0");
    reset_checked();
    assert_eq!(checked("alloc_in_pool"), 0);
    assert_eq!(checked("execute_node"), 0);
}

/// #619: an arm is scoped to the **arming thread**.
///
/// This is the failure the 2026-10-19 parallel device run showed: the stall gate
/// armed `forward_batch` and a concurrent `kv_sharing` gate's tick met the
/// injection inside the real forward entry, so a gate failed for another gate's
/// mutation. `guard` is the decision the chokepoints make (`guard_panic` for
/// `forward_batch` is the same `requested` call), so asserting on it asserts the
/// seam, not a copy of it.
#[test]
fn an_arm_reaches_only_the_arming_thread() {
    let g = InjectionGuard::arm("forward_batch");
    // Positive control: on the arming thread the chokepoint fires.
    assert!(requested("forward_batch"), "the arming thread sees its arm");
    assert!(
        guard("forward_batch").is_err(),
        "positive control: the chokepoint fires here"
    );
    // The victim: another gate stepping a `BatchEngine` on another thread.
    let other =
        std::thread::spawn(|| (requested("forward_batch"), guard("forward_batch").is_err()))
            .join()
            .expect("the other thread");
    assert_eq!(
        other,
        (false, false),
        "an arm must not reach a gate on another thread (#619)"
    );
    drop(g);
    assert!(
        !requested("forward_batch"),
        "the arm is gone after the scope"
    );
    assert!(
        guard("forward_batch").is_ok(),
        "and the chokepoint is clean"
    );
}

/// #619's root cause, pinned: `arm` used to write `MINFER_TEST_CALL_FAIL`, which
/// is how one gate's arm reached every other gate in the process. The
/// environment stays the **out-of-process** lever (a mutation run on the command
/// line), never something an in-process gate writes.
#[test]
fn arming_never_writes_the_process_environment() {
    assert!(
        std::env::var(ENV).is_err(),
        "{ENV} is set in the default suite; the seam must be off"
    );
    let g = InjectionGuard::arm("forward_batch");
    assert!(requested("forward_batch"), "the arm is live in-process");
    assert!(
        std::env::var(ENV).is_err(),
        "an arm must stay in-process: writing {ENV} is what leaked it across gates (#619)"
    );
    drop(g);
    assert!(
        std::env::var(ENV).is_err(),
        "and it stays off after the scope"
    );
}

/// Nested arms (`arm()` inside an armed scope) unarm innermost-first, so a
/// gate composed of two injected phases does not disarm the outer one early.
#[test]
fn nested_arms_unarm_in_lifo_order() {
    let outer = InjectionGuard::arm("forward_batch");
    let inner = InjectionGuard::arm("execute_node");
    assert!(requested("forward_batch") && requested("execute_node"));
    drop(inner);
    assert!(requested("forward_batch"), "the outer arm survives");
    assert!(!requested("execute_node"), "the inner arm is gone");
    drop(outer);
    assert!(!requested("forward_batch"), "both arms are gone");
}

/// The scope rule itself, as a pure decision: an arm on this thread fires, the
/// environment (the command-line mutation lever) fires everywhere, and neither
/// implies the other. Mutating `decides` to drop either clause is the mutation
/// check for the two gates above.
#[test]
fn the_scope_rule_is_an_arm_or_the_environment() {
    assert!(decides(true, None, "forward_batch"), "an arm needs no env");
    assert!(
        decides(false, Some("forward_batch"), "forward_batch"),
        "the env lever fires without an arm"
    );
    assert!(
        decides(true, Some("all"), "forward_batch"),
        "an arm still fires"
    );
    assert!(
        !decides(false, None, "forward_batch"),
        "nothing armed, no lever"
    );
    assert!(
        !decides(false, Some("execute_node"), "forward_batch"),
        "the env lever names sites exactly"
    );
    assert!(
        decides(false, Some("all"), "execute_node"),
        "`all` stays the wildcard through the scope rule"
    );
}
