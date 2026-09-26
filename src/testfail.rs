//! The one failure-injection seam for gate mutation checks (issue #171).
//!
//! Rule 3 of the gate contract ([`docs/GATE-CONTRACT.md`]) is that every gate
//! needs **mutation evidence**: break the implementation, watch the gate go
//! red, revert. Before #171 the experiment cost a bespoke mock per ticket — a
//! test-local `ModelDef` here (`FailingForward`, #151), a private
//! `MINFER_TEST_*` variable there (#145, #147, #167) — so "make this gate
//! fail" cost half an hour and got skipped. This module is the one seam: a
//! chokepoint calls [`guard`] (or [`guard_panic`]) and the only thing that
//! makes it fire is the presence of `MINFER_TEST_CALL_FAIL`.
//!
//! ```text
//! // one line at the chokepoint …
//! crate::testfail::guard("alloc_in_pool")?;
//! // … and one environment variable on the command line:
//! MINFER_TEST_CALL_FAIL=alloc_in_pool cargo test --release …
//! ```
//!
//! # Off by default, test-only by contract
//!
//! [`requested`] is `false` whenever the variable is unset, which is every
//! default run: CI, a normal `cargo test`, and the `compute-sanitizer` run.
//! The property is pinned by [`tests::the_seam_is_off_by_default`]. A
//! chokepoint is an ordinary call that returns `Ok`/does nothing in
//! production; no production code sets the variable.
//!
//! # Matching rule
//!
//! Exact token, comma-separated: `MINFER_TEST_CALL_FAIL=forward_batch,launch:gemm_f16_f16`
//! fires those two sites, `all` fires every Rust chokepoint, and a token that
//! is a superstring of a site (`launch:gemm` against `launch:gemm_f16_f16`)
//! never matches. The device-side helper `minfer_test_call_fails` in
//! `src/cuda_kernels.cu` implements the identical rule for the CUDA launch and
//! attribute sites (`launch:*`, `attr:*`), because a C++ kernel cannot call
//! into Rust; the Rust half is the one that is unit-tested.
//!
//! # Chokepoints
//!
//! | site | where | what it injects |
//! |---|---|---|
//! | `forward_batch` | `server::chat::guarded_forward_batch` | panic, caught into the guarded 500 (#151's path) |
//! | `alloc_in_pool` | `graph::alloc::GraphAllocator::alloc_in_pool` | `Err` before the pool is touched |
//! | `execute_node` | `graph::scheduler::BackendScheduler::execute` | `Err` before the backend dispatch |
//! | `register_weight` | `models::weight_reg::register_cuda_weight` | panic in the CUDA weight registrar |
//! | `launch:*` / `attr:*` | `src/cuda_kernels.cu` | the **real** CUDA call is made to fail (#147) |
//!
//! # The observation half
//!
//! [`note_checked`] / [`checked`] are the seam's *observation* side. A gate that
//! must prove a path really executed cannot read the path's own answer: a
//! dispatch function naming its branch is self-certifying. #141's vectorized
//! f16 dot needed `F16_SIMD_PATH_CALLS` — bumped by the SIMD entry points
//! themselves — for exactly that reason, and `vec_ops::tests::f16_dot_uses_the_vectorized_path`
//! asserts the counter moved. Bumping [`note_checked`] at a chokepoint gives a
//! gate the same evidence cheaply: a thread-local counter, no allocation on the
//! hot path, and an assertion that it advanced.
//!
//! [`docs/GATE-CONTRACT.md`]: ../docs/GATE-CONTRACT.md

/// The environment variable that drives the seam — #147's original
/// `MINFER_TEST_CALL_FAIL`, kept because the #147 device gates are the contract.
pub const ENV: &str = "MINFER_TEST_CALL_FAIL";

/// Does the comma-separated `value` name `site`? Absorbed from
/// `cuda::injection_names_site` (#147) so there is one matcher, not two.
///
/// Pure on the variable's *value*, so it is unit-tested without a device and
/// without mutating the process environment. Token matching is exact: `all`
/// for every site, otherwise one site token.
pub fn injection_names_site(value: &str, site: &str) -> bool {
    value.split(',').any(|t| {
        let t = t.trim();
        !t.is_empty() && (t == "all" || t == site)
    })
}

/// Is the injection switch asking for `site`? `false` when the variable is
/// unset — the default in every non-mutation run.
pub fn requested(site: &str) -> bool {
    std::env::var(ENV).map_or(false, |v| injection_names_site(&v, site))
}

/// The error-returning chokepoint: `Ok(())` unless `site` is requested.
///
/// The message names the variable and the site, so a mutated run's failure
/// transcript identifies the injection rather than looking like a real fault.
pub fn guard(site: &str) -> Result<(), String> {
    if requested(site) {
        Err(format!(
            "[testfail] deliberate failure injected at site '{site}' \
             (MINFER_TEST_CALL_FAIL={site})"
        ))
    } else {
        Ok(())
    }
}

/// The panic-returning chokepoint, for a path whose failure channel is a panic:
/// `guarded_forward_batch` catches it and answers 500, so injecting here
/// exercises the **real** guard and the real error path.
pub fn guard_panic(site: &str) {
    if requested(site) {
        panic!(
            "[testfail] deliberate panic injected at site '{site}' \
             (MINFER_TEST_CALL_FAIL={site})"
        );
    }
}

thread_local! {
    /// Per-thread `(site, count)` pairs. A chokepoint records here; a gate reads
    /// it with [`checked`]. Thread-local because a graph executes on the
    /// calling thread and a gate's assertion runs there too.
    static CHECKED: std::cell::RefCell<Vec<(&'static str, u64)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Record that the `site` chokepoint reached this point on this thread.
///
/// `site` is `&'static str` on purpose: every call site is a literal, and a
/// static key keeps the counter allocation-free after the first note.
pub fn note_checked(site: &'static str) {
    CHECKED.with(|c| {
        let mut v = c.borrow_mut();
        match v.iter_mut().find(|(s, _)| *s == site) {
            Some((_, n)) => *n += 1,
            None => v.push((site, 1)),
        }
    });
}

/// How many times `site` ran on this thread since the last [`reset_checked`].
///
/// Gate-facing API: read by `#[cfg(test)]` gates and by the `#[ignore]`d
/// real-model set, so a non-test build sees it unused.
#[allow(dead_code)]
pub fn checked(site: &str) -> u64 {
    CHECKED.with(|c| {
        c.borrow()
            .iter()
            .find(|(s, _)| *s == site)
            .map_or(0, |(_, n)| *n)
    })
}

/// Clear every counter on this thread (a gate calls this before the run it
/// wants to observe, so a previous test on the same thread cannot fake it).
#[allow(dead_code)]
pub fn reset_checked() {
    CHECKED.with(|c| c.borrow_mut().clear());
}

/// Test-only RAII arm/disarm of the switch, so a panicking gate cannot leave
/// the injection armed for the rest of the process.
///
/// This mirrors `cuda::tests::InjectionGuard`, which the #147 device gates keep
/// for their own scoped arm; a new gate should use this one.
#[cfg(test)]
pub(crate) struct InjectionGuard {
    prev: Option<String>,
}

#[cfg(test)]
impl InjectionGuard {
    /// Arm `site` until the returned guard drops.
    pub(crate) fn arm(site: &str) -> Self {
        let prev = std::env::var(ENV).ok();
        std::env::set_var(ENV, site);
        Self { prev }
    }
}

#[cfg(test)]
impl Drop for InjectionGuard {
    fn drop(&mut self) {
        match &self.prev {
            Some(v) => std::env::set_var(ENV, v),
            None => std::env::remove_var(ENV),
        }
    }
}

#[cfg(test)]
mod tests {
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
}
