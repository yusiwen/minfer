//! Process-level gates for F4's backend **name surface** (#57).
//!
//! The registry resolves names purely, so `graph::registry`'s unit tests cover
//! the matrix. These tests cover the thing unit tests cannot: that the *CLI and
//! the environment* spell the same surface, that an unknown name and a
//! compiled-out name are two different loud refusals, and that both happen
//! **before** the model is looked for (every case below points at a model path
//! that does not exist — a backend refusal must win the race).
//!
//! No model is loaded, so these are fast and run in the default suite.

use std::process::{Command, Stdio};

/// Spawn the built binary with a deadline (kill on hang) and return
/// `(stdout, stderr, exit_code)`.
fn run_cli(args: &[&str], env: &[(&str, &str)]) -> (String, String, i32) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_minfer"));
    cmd.args(args)
        .env_remove("MINFER_BACKENDS")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn minfer");
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            break;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            panic!("minfer did not exit within 60s (hang?)");
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    }
    let out = child.wait_with_output().expect("wait");
    (
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
        out.status.code().unwrap_or(-1),
    )
}

/// An **absolute** model path that cannot exist. Absolute on purpose: a bare
/// name is resolved against the HuggingFace cache (and could be downloaded),
/// while an absolute path that is missing fails immediately with "File not
/// found" — which is what lets a case assert that the backend gate ran and the
/// failure moved on to the model.
const NO_MODEL: &str = "/nonexistent/f4-gate-no-such-model.gguf";

/// F4 gate 3a: an unknown backend name is a loud startup refusal that names the
/// accepted set — from the CLI flag.
#[test]
fn an_unknown_backend_name_exits_loudly_from_the_flag() {
    let (_, err, code) = run_cli(&["--backend", "gpu2", NO_MODEL], &[]);
    assert_ne!(code, 0, "an unknown backend must not be accepted: {err}");
    assert!(
        err.contains("unknown backend 'gpu2'"),
        "the refusal must name the unknown name: {err}"
    );
    assert!(
        err.contains("known backends are: cpu, metal, cuda"),
        "the refusal must list the accepted names: {err}"
    );
    // …and it must be the *backend* refusal, not a missing-model error: the
    // backend gate runs before the model is resolved.
    assert!(
        !err.contains("File not found"),
        "the backend gate must run before the model is looked for: {err}"
    );
}

/// F4 gate 3b: the same surface through `MINFER_BACKENDS`, with the same message.
#[test]
fn an_unknown_backend_name_exits_loudly_from_the_environment() {
    let (_, err, code) = run_cli(&[NO_MODEL], &[("MINFER_BACKENDS", "cpu,gpu2")]);
    assert_ne!(code, 0);
    assert!(err.contains("unknown backend 'gpu2'"), "{err}");
    assert!(
        err.contains("known backends are: cpu, metal, cuda"),
        "{err}"
    );
}

/// F4 gate 3c: a name that *exists* but is not compiled into this build is a
/// **different** refusal, and it says which of the two situations it is. On a
/// non-macOS build `metal` is registered nowhere; on a non-`cuda` build neither
/// is `cuda`.
#[test]
fn a_compiled_out_backend_name_is_a_different_refusal() {
    #[cfg(not(target_os = "macos"))]
    {
        let (_, err, code) = run_cli(&["--backend", "metal", NO_MODEL], &[]);
        assert_ne!(code, 0);
        assert!(
            err.contains("backend 'metal' is known but not compiled into this build"),
            "the compiled-out refusal must be distinct from 'unknown': {err}"
        );
        assert!(
            !err.contains("unknown backend"),
            "a known-but-compiled-out name must not read as unknown: {err}"
        );
        assert!(
            !err.contains("File not found"),
            "the backend gate must run before the model is looked for: {err}"
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let (_, err, code) = run_cli(&["--backend", "cuda", NO_MODEL], &[]);
        assert_ne!(code, 0);
        assert!(
            err.contains("backend 'cuda' is known but not compiled into this build"),
            "{err}"
        );
        assert!(err.contains("--features cuda"), "{err}");
    }
}

/// F4 gate 3d: a known, compiled-in name passes the backend gate — proved by the
/// failure moving on to the model. `cpu` is always accepted, and it is also what
/// the fence keeps as the universal fallback.
#[test]
fn a_known_backend_name_passes_the_gate() {
    for args in [
        vec!["--backend", "cpu", NO_MODEL],
        // Comma-separated and repeatable spellings are the same request.
        vec!["--backend", "cpu,", NO_MODEL],
        vec!["--backend=cpu", NO_MODEL],
    ] {
        let (_, err, code) = run_cli(&args, &[]);
        assert_ne!(code, 0, "no model, so this must still fail: {err}");
        assert!(
            !err.contains("unknown backend") && !err.contains("not compiled into this build"),
            "a known name must pass the backend gate ({args:?}): {err}"
        );
        // …and the failure is the missing model, i.e. the gate let it through.
        assert!(
            err.contains("File not found"),
            "the run must have moved on to the model ({args:?}): {err}"
        );
    }
    // The environment spelling of the same name.
    let (_, err, code) = run_cli(&[NO_MODEL], &[("MINFER_BACKENDS", "cpu")]);
    assert_ne!(code, 0);
    assert!(
        !err.contains("unknown backend") && !err.contains("not compiled into this build"),
        "{err}"
    );
    assert!(err.contains("File not found"), "{err}");
}

/// F4 gate 3e: the flag is accepted before any subcommand dispatch, so a
/// subcommand with its own parser (`bench`) honours it too — an unknown name is
/// refused there as well, instead of being reported as an unknown *option*.
#[test]
fn the_backend_flag_reaches_the_bench_subcommand() {
    let (_, err, code) = run_cli(&["bench", "--backend", "gpu2", NO_MODEL], &[]);
    assert_ne!(code, 0);
    assert!(err.contains("unknown backend 'gpu2'"), "{err}");
    assert!(
        !err.contains("unknown option"),
        "the flag must be consumed before bench's own parser: {err}"
    );
}

/// F4 gate 3f: `--help` documents the surface (a user cannot discover a name
/// gate that is not in the usage text).
#[test]
fn help_documents_the_backend_flag() {
    let (_, err, code) = run_cli(&["--help"], &[]);
    assert_eq!(code, 0);
    assert!(err.contains("--backend"), "{err}");
    assert!(err.contains("MINFER_BACKENDS"), "{err}");
    assert!(
        err.contains("cpu, metal, cuda"),
        "the accepted names belong in the usage text: {err}"
    );
}

/// F4 gate 3g: **behaviour preservation for the pre-existing fences.** The
/// default request means "whatever this build can use", so a device the
/// `MINFER_DISABLE_CUDA` / `MINFER_DISABLE_MPS` flags fence off must keep
/// meaning "run on the CPU" — not "refuse to start". Only a backend the request
/// actually *names* is checked for availability (stage 2).
#[test]
fn a_disabled_device_is_not_refused_unless_it_was_named() {
    let (_, err, code) = run_cli(
        &[NO_MODEL],
        &[("MINFER_DISABLE_CUDA", "1"), ("MINFER_DISABLE_MPS", "1")],
    );
    assert_ne!(code, 0, "there is no model, so this must still fail");
    assert!(
        !err.contains("not available on this machine"),
        "an unnamed device that the pre-existing flags disable must not become a \
         startup refusal: {err}"
    );
    // …and the failure is the missing model, i.e. the run proceeded to the CPU.
    assert!(err.contains("File not found"), "{err}");

    // Naming it *is* the refusal, where the flag can actually disable it.
    #[cfg(feature = "cuda")]
    {
        let (_, err, code) = run_cli(
            &["--backend", "cuda", NO_MODEL],
            &[("MINFER_DISABLE_CUDA", "1")],
        );
        assert_ne!(code, 0);
        assert!(
            err.contains("backend 'cuda' is compiled in but not available on this machine"),
            "{err}"
        );
        assert!(err.contains("MINFER_DISABLE_CUDA"), "{err}");
    }
}
