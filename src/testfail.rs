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
//! `src/cuda/kernels/*.cu` (the CUDA launch and attribute sites `launch:*` /
//! `attr:*`) **calls back into this module** ([`minfer_seam_site_requested`]) instead
//! of re-reading the environment, so there is one matcher and one scope rule for
//! both halves — the C++ side cannot own the scope, because the scope is a
//! thread-local (#619).
//!
//! # Scope: the arming thread
//!
//! An in-process arm is **thread-scoped** (#619). `MINFER_TEST_CALL_FAIL` stays
//! the *out-of-process* lever — a mutation run names a site on the command line
//! and every thread of that process sees it — but a gate arming the seam for its
//! own scope must not reach another gate: a parallel harness
//! (`cargo test --release …`, the `#[ignore]`d real-model sets, [#195]'s device
//! repeats) ticks engines in one process, and a process-global arm made one
//! gate's deliberate failure fail a different gate. The rule is [`decides`]: an
//! arm on this thread fires, or the environment fires; [`InjectionGuard::arm`]
//! therefore **never writes the environment**.
//!
//! [#195]: https://github.com/yusiwen/minfer/issues/195
//!
//! # Chokepoints
//!
//! | site | where | what it injects |
//! |---|---|---|
//! | `forward_batch` | `server::chat::guarded_forward_batch` | panic, caught into the guarded 500 (#151's path) |
//! | `alloc_in_pool` | `graph::alloc::GraphAllocator::alloc_in_pool` | `Err` before the pool is touched |
//! | `execute_node` | `graph::scheduler::BackendScheduler::execute` | `Err` before the backend dispatch |
//! | `register_weight` | `models::weight_reg::register_cuda_weight` | panic in the CUDA weight registrar |
//! | `metal_device_memory` | `metal::MpsState::device_memory` | `Reported` → `QueryFailed` (the E4/E5 device answer on macOS, [#53]) |
//! | `metal_cross_copy` | `graph::metal_backend::MetalBackend::cross_enqueue` | suppresses the staging copy's `MTLSharedEvent` signal, so phase B's bounded wait takes its real timeout branch ([#137]) |
//! | `launch:*` / `attr:*` | `src/cuda/kernels/*.cu` | the **real** CUDA call is made to fail (#147) |
//!
//! [#53]: https://github.com/yusiwen/minfer/issues/53
//! [#137]: https://github.com/yusiwen/minfer/issues/137
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

/// The scope rule, pure: `site` fires if it is armed **on this thread** or if the
/// process environment names it.
///
/// The two levers answer different questions and neither implies the other. An
/// arm is a gate saying "this scope is mine" and must not cross a thread
/// (`#619`); the environment is a whole-process mutation run and is deliberately
/// global. Pure, so the rule is unit-tested without mutating the environment or
/// spawning anything — and so a mutation of either clause is a one-line edit.
fn decides(armed_here: bool, env: Option<&str>, site: &str) -> bool {
    armed_here || env.map_or(false, |v| injection_names_site(v, site))
}

/// Is the injection switch asking for `site`? `false` when nothing is armed and
/// the variable is unset — the default in every non-mutation run.
///
/// Reads the caller's thread-local arms first (the same thread-local the
/// chokepoints and the CUDA helper share), then the environment.
pub fn requested(site: &str) -> bool {
    let armed_here = ARMED.with(|a| a.borrow().iter().any(|s| s == site));
    let env = std::env::var(ENV).ok();
    decides(armed_here, env.as_deref(), site)
}

/// The C++ half of the seam: `src/cuda/kernels/guard.cu` declares this **weak**
/// and forwards its `minfer_test_call_fails` to it, so the `launch:*` / `attr:*`
/// sites reach [`requested`] — the same decision, matcher and thread-scoped arms
/// as the Rust chokepoints, instead of a second `getenv` parser that could only
/// ever be process-wide (#619). Weak because `cargo run --example
/// cuda_graph_diag` links the kernels without the crate; there the symbol is null
/// and `guard.cu` answers "not injected".
#[cfg(feature = "cuda")]
#[no_mangle]
pub extern "C" fn minfer_seam_site_requested(site: *const std::ffi::c_char) -> bool {
    if site.is_null() {
        return false;
    }
    // SAFETY: every caller in `src/cuda/kernels/` passes a string literal, which
    // is NUL-terminated and alive for the whole program.
    let site = unsafe { std::ffi::CStr::from_ptr(site) };
    // A non-UTF-8 site cannot equal a Rust site token; treating it as "not
    // injected" keeps the seam off rather than guessing.
    site.to_str().map_or(false, |s| requested(s))
}

thread_local! {
    /// The sites armed **on this thread** ([`InjectionGuard::arm`], #619). Not
    /// the environment: writing that is what made one gate's arm reach every
    /// other gate in the process.
    static ARMED: std::cell::RefCell<Vec<String>> = const { std::cell::RefCell::new(Vec::new()) };
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
/// Test-only (#238): driven by `graph::scheduler::tests::the_execute_chokepoint_is_observable`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn checked(site: &str) -> u64 {
    CHECKED.with(|c| {
        c.borrow()
            .iter()
            .find(|(s, _)| *s == site)
            .map_or(0, |(_, n)| *n)
    })
}

/// Clear every counter on this thread (a gate calls this before the run it
/// wants to observe, so a previous test on the same thread cannot fake it).
/// Test-only (#238): driven by `graph::scheduler::tests::the_execute_chokepoint_is_observable`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn reset_checked() {
    CHECKED.with(|c| c.borrow_mut().clear());
}

/// Test-only RAII arm/disarm of the switch for **this thread**, so a panicking
/// gate cannot leave the injection armed for the rest of the run.
///
/// The scope is the thread (#619): `arm` writes the thread-local, never
/// `MINFER_TEST_CALL_FAIL`, because the environment is process-wide and a
/// parallel harness runs the gates in one process. The CUDA `launch:*` / `attr:*`
/// sites see it too — their helper forwards to [`requested`]
/// — so a device gate arms and launches on its own thread and stays inside it.
#[cfg(test)]
pub(crate) struct InjectionGuard {
    /// How long [`ARMED`] was before this arm: dropping truncates back to it, so
    /// nested arms unarm innermost-first.
    depth: usize,
}

#[cfg(test)]
impl InjectionGuard {
    /// Arm `site` on this thread until the returned guard drops.
    pub(crate) fn arm(site: &str) -> Self {
        Self::arm_all(&[site])
    }

    /// Arm several sites as **one** scope: a device dispatch that reaches more
    /// than one site in a single call arms them together, and one drop disarms
    /// the set.
    pub(crate) fn arm_all(sites: &[&str]) -> Self {
        let depth = ARMED.with(|a| {
            let mut v = a.borrow_mut();
            let depth = v.len();
            v.extend(sites.iter().map(|s| s.to_string()));
            depth
        });
        Self { depth }
    }
}

#[cfg(test)]
impl Drop for InjectionGuard {
    fn drop(&mut self) {
        ARMED.with(|a| a.borrow_mut().truncate(self.depth));
    }
}

#[cfg(test)]
mod tests;
