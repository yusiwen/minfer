//! A `--chat-template` refusal is a startup refusal with a non-zero status (#601).
//!
//! Fast tier: the file half of the override is decided **before the model is
//! resolved**, so these tests need no model. The assertions that matter are the
//! exit code, the file named on stderr, the absence of a panic — and, for the
//! fast-fail, that the model was never loaded.

use std::io::Write;
use std::process::Command;

fn scratch(name: &str, body: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join("minfer-template-refusal");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join(name);
    let mut f = std::fs::File::create(&path).expect("write scratch template");
    f.write_all(body.as_bytes()).unwrap();
    drop(f);
    path
}

/// `minfer /nonexistent/model.gguf hi [--chat-template <file>] …`.
fn run(template: Option<&std::path::Path>, extra: &[&str]) -> (i32, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_minfer"));
    cmd.arg("/nonexistent/model.gguf").arg("hi");
    if let Some(t) = template {
        cmd.arg("--chat-template").arg(t);
    }
    cmd.args(extra);
    let out = cmd.output().expect("spawn minfer");
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn an_unrenderable_template_file_refuses_before_the_model_with_exit_1() {
    let path = scratch("bad.jinja", "{{ messages[0]['content'].format() }}");
    let (code, stderr) = run(Some(&path), &[]);
    assert_eq!(code, 1, "a refused template must exit 1; stderr:\n{stderr}");
    assert!(stderr.contains("bad.jinja"), "the file is named:\n{stderr}");
    assert!(
        stderr.contains("format"),
        "the construct is named:\n{stderr}"
    );
    assert!(!stderr.contains("panicked"), "{stderr}");
    // #601's fast-fail: the file is decided before the model is looked for, so
    // the model's own refusal is never reached.
    assert!(
        !stderr.contains("File not found"),
        "the template refusal must preempt the model; stderr was:\n{stderr}"
    );
}

#[test]
fn a_missing_template_file_refuses_with_exit_1() {
    let path = std::env::temp_dir().join("minfer-template-refusal/absent.jinja");
    let (code, stderr) = run(Some(&path), &[]);
    assert_eq!(code, 1, "{stderr}");
    assert!(stderr.contains("absent.jinja"), "{stderr}");
    assert!(!stderr.contains("panicked"), "{stderr}");
}

#[test]
fn a_good_template_passes_the_gate_and_the_model_path_is_the_next_refusal() {
    let path = scratch("good.jinja", "U:{{ messages[0]['content'] }}");
    let (code, stderr) = run(Some(&path), &[]);
    assert_ne!(code, 0, "the placeholder model path must still fail");
    assert!(
        !stderr.contains("chat template error"),
        "the template itself was fine; stderr was:\n{stderr}"
    );
    // The next refusal is the model, which is the proof the template gate passed.
    assert!(
        stderr.contains("File not found") && stderr.contains("/nonexistent/model.gguf"),
        "the model path is the next refusal; stderr was:\n{stderr}"
    );
}

#[test]
fn the_template_flags_are_refused_together() {
    let path = scratch("ok.jinja", "U:{{ messages[0]['content'] }}");
    let (code, stderr) = run(Some(&path), &["--no-template"]);
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stderr.contains("--chat-template") && stderr.contains("--no-template"),
        "both flags are named:\n{stderr}"
    );
}
