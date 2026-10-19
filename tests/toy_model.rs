//! The checked-in toy GGUF must load, build a graph and decode one token (#609).
//!
//! This is the fixture's own gate: `tests/template_e2e.rs` uses the toy as the
//! model under test, so a fixture the engine cannot run would fail there too — but
//! with a confusing message. Here the failure names the fixture and the generator.

use std::path::{Path, PathBuf};
use std::process::Command;

fn toy() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/toy/toy-qwen2-f32.gguf")
}

#[test]
fn the_toy_model_loads_and_decodes_one_token() {
    let path = toy();
    assert!(
        path.exists(),
        "missing fixture {} — regenerate with `python3 scripts/gen_toy_gguf.py`",
        path.display()
    );
    let out = Command::new(env!("CARGO_BIN_EXE_minfer"))
        .arg(&path)
        .args(["hi", "-n", "1"])
        .output()
        .expect("spawn minfer on the toy model");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        out.status.code(),
        Some(0),
        "the toy must run; stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    // The toy has no chat template, so the run must announce ChatML and tokenize
    // the framed prompt (a raw "hi" would be far fewer tokens).
    assert!(
        stderr.contains("using the generic ChatML renderer"),
        "the toy has no chat template and must say so; stderr:\n{stderr}"
    );
    assert!(
        stdout.contains("Generated: 1 tokens"),
        "one decode step must complete; stdout:\n{stdout}"
    );
}

/// The fixture is generated, not hand-made: `--check` must reproduce it.
#[test]
fn the_toy_model_matches_its_generator() {
    let script = Path::new(env!("CARGO_MANIFEST_DIR")).join("scripts/gen_toy_gguf.py");
    let out = Command::new("python3")
        .arg(&script)
        .arg("--check")
        .output()
        .expect("spawn python3");
    assert_eq!(
        out.status.code(),
        Some(0),
        "the committed fixture drifted from its generator; stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
}
