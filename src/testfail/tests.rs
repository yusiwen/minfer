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
