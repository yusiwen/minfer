//! #218: the fresh-process harness for the prefill-GEMM dynamic-smem gates.
//!
//! Why a child process. Two pieces of the mechanism under test are decided
//! **once per process**, and both are exactly what a non-vacuous gate must
//! establish a precondition on:
//!
//! - the prefill-GEMM tile is read from `MINFER_GEMM_TM` / `MINFER_GEMM_K64`
//!   into function-local `static`s on the first `launch_gemm_f16` call, so a
//!   test cannot switch to a >48 KiB instantiation mid-process; and
//! - `cudaFuncSetAttribute` sticks to the kernel function for the life of the
//!   process, so "the device read back `opted_in == 0` *before* this forward"
//!   can only be observed in a process where no earlier launch touched the
//!   instantiation.
//!
//! So a gate that must prove "production opts this in" spawns this same test
//! binary with the device knobs forced, runs a single test in it, and reads its
//! verdict. The child is one test, so nothing else can have opted the
//! instantiation in first.
//!
//! Only compiled under `#[cfg(test)]`; the gates below are the only callers.

use std::process::Command;

/// Set in the child to select which gate body to run. Absent in the parent.
pub(crate) const PHASE_ENV: &str = "MINFER_218_PHASE";

/// The sentinels a child prints on stdout (the harness passes `--nocapture`).
pub(crate) const OK: &str = "MINFER218_OK";
pub(crate) const SKIP: &str = "MINFER218_SKIP";

/// Env the child must not inherit: anything that would move the tile the gate
/// is about, disable the path under test, or arm a different injection. A gate
/// needs a *controlled* process, not whatever the caller happened to export.
const STRIPPED: &[&str] = &[
    "MINFER_TEST_CALL_FAIL",
    "MINFER_TEST_ISSUE147",
    "MINFER_TEST_ISSUE162",
    "MINFER_TEST_LATCH_ERROR",
    "MINFER_GEMM_TM",
    "MINFER_GEMM_A32",
    "MINFER_FUSED_B",
    "MINFER_NO_CUDA_GRAPH",
    "MINFER_NO_PREFILL_CAPTURE",
    "MINFER_CAPTURE_PREFILL",
    "MINFER_NO_PREFILL_GEMM",
    "MINFER_DEVICE_TIER",
    "MINFER_BACKENDS",
    "MINFER_GPU_LAYERS",
    "MINFER_DISABLE_CUDA",
    "MINFER_CACHE_TYPE",
    // NOTE: `MINFER_TEST_CAPTURE_WARMUP` is deliberately **not** stripped — it is
    // the documented mutation lever for
    // `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window` (set it to 1
    // and the child drives the >48 KiB opt-in into the window, turning the
    // counter assertion red).
];

pub(crate) struct Child {
    pub status: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

impl Child {
    pub(crate) fn ok(&self) -> bool {
        self.status == Some(0)
    }

    /// The child's verdict, or a panic that carries its whole output. A child
    /// that skipped (no device, no cached model) returns `None`; a child that
    /// finished without the OK sentinel is a failure, not a pass — a gate must
    /// not go green because its body silently returned.
    pub(crate) fn verdict(&self, what: &str) -> Option<()> {
        if !self.ok() {
            panic!(
                "{what}: the fresh-process arm failed (exit {:?}):\n--- stdout ---\n{}\n--- stderr ---\n{}",
                self.status, self.stdout, self.stderr
            );
        }
        if self.stdout.contains(SKIP) || self.stderr.contains(SKIP) {
            eprintln!(
                "{what}: skipping — {}",
                self.stdout
                    .lines()
                    .chain(self.stderr.lines())
                    .find(|l| l.contains(SKIP))
                    .unwrap_or("child skipped")
            );
            return None;
        }
        assert!(
            self.stdout.contains(OK),
            "{what}: the child exited 0 but never reached its assertion (no {OK} sentinel):\n\
             --- stdout ---\n{}\n--- stderr ---\n{}",
            self.stdout,
            self.stderr
        );
        Some(())
    }
}

/// Run `filter` (a unique test-name substring) in a fresh process of this test
/// binary, with `phase` selected and `MINFER_MMQ=0` + `MINFER_GEMM_K64=1` in
/// force. Those two gate the f16 wmma prefill GEMM (`gemm_f16_nt_kernel_t`) and
/// make the default `tm=128, ks=64` instantiation need 57344 B — over the
/// 48 KiB default cudaFuncSetAttribute cap, so the production opt-in is
/// exercised. `extra_env` is applied **after** [`STRIPPED`].
pub(crate) fn run_self(filter: &str, phase: &str, extra_env: &[(&str, &str)]) -> Child {
    let exe = std::env::current_exe().expect("current_exe for the fresh-process arm");
    let mut cmd = Command::new(exe);
    cmd.arg("--test-threads=1").arg("--nocapture").arg(filter);
    cmd.env(PHASE_ENV, phase);
    cmd.env("MINFER_MMQ", "0"); // the f16 wmma GEMM, not the int8 MMQ
    cmd.env("MINFER_GEMM_K64", "1"); // tm=128/ks=64 -> 57344 B > 48 KiB
    for k in STRIPPED {
        cmd.env_remove(k);
    }
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn the fresh-process arm");
    // `compute-sanitizer --target-processes all` follows this child too and
    // aggregates every process it launched into the ONE `ERROR SUMMARY` it
    // prints at the end, so the sanitizer log's `0 API errors` covers the child
    // as well — no per-child summary to re-emit here.
    Child {
        status: out.status.code(),
        stdout: String::from_utf8_lossy(&out.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// `Some(phase)` when this process was spawned by [`run_self`] for `phase`.
pub(crate) fn child_phase() -> Option<String> {
    std::env::var(PHASE_ENV).ok()
}

/// Child-side: this process has no CUDA device — tell the parent to skip.
pub(crate) fn child_skip(reason: &str) -> ! {
    eprintln!("{SKIP}: {reason}");
    std::process::exit(0);
}

/// Child-side: the body reached its end; hand the parent a pass.
pub(crate) fn child_ok() {
    println!("{OK}");
}
