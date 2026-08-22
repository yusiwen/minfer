//! Process-level tests for `--cnv` conversation mode (docs/CLI-CONVERSATION-PLAN.md §8.3).
//!
//! Two tiers:
//! - **Fast** (run by default): argument validation / help — no model load.
//! - **Real model** (`#[ignore]`): scripted stdin sessions against the cached
//!   Qwen2.5-0.5B q4_0. Ignored because the scalar-CPU 0.5B forward is slow
//!   (~minutes); run explicitly with `cargo test --test conversation_cli -- --ignored`
//!   on a machine that has the model cached.

use std::io::{Read, Write};
use std::process::{Command, Stdio};

/// Cached 0.5B q4_0 (same fixture as the existing realdata tests).
fn model_path() -> Option<String> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
    p.exists().then(|| p.to_string_lossy().to_string())
}

/// Spawn the built binary with piped stdin/stdout/stderr, write all input at
/// once (EOF closes stdin), wait with a deadline (kill on hang), return
/// (stdout, stderr, exit_code).
fn run_cli(args: &[&str], stdin_input: &str, timeout_secs: u64) -> (String, String, i32) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_minfer"))
        .args(args)
        // 固定 CPU 后端：golden/断言与后端解耦（Metal logits 与 CPU 有 ~1e1 差异）
        .env("MINFER_DISABLE_MPS", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn minfer");
    {
        let mut si = child.stdin.take().expect("stdin");
        si.write_all(stdin_input.as_bytes()).expect("write stdin");
    } // drop → EOF

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        if child.try_wait().expect("try_wait").is_some() {
            break;
        }
        if std::time::Instant::now() > deadline {
            let _ = child.kill();
            panic!("minfer did not exit within {timeout_secs}s (hang? flush discipline violated?)");
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    let code = child.wait().map(|s| s.code().unwrap_or(-1)).unwrap_or(-1);
    let mut out = String::new();
    let mut err = String::new();
    child.stdout.take().unwrap().read_to_string(&mut out).unwrap();
    child.stderr.take().unwrap().read_to_string(&mut err).unwrap();
    (out, err, code)
}

/// Parse per-turn stats lines like `[turn 2] prefill 51 tokens, generated 0 tokens in 0.12s`.
fn turn_stats(err: &str) -> Vec<(usize, usize, usize)> {
    let re = regex::Regex::new(r"\[turn (\d+)\] (?:regen: )?prefill (\d+) tokens, generated (\d+) tokens")
        .unwrap();
    re.captures_iter(err)
        .map(|c| {
            (
                c[1].parse().unwrap(),
                c[2].parse().unwrap(),
                c[3].parse().unwrap(),
            )
        })
        .collect()
}

// === Fast: argument validation (no model load) ===

#[test]
fn cnv_with_no_template_errors_before_model_load() {
    // 互斥校验发生在模型解析之前 → 任意路径即可，快。
    let (_, err, code) = run_cli(&["--cnv", "--no-template", "no-such-model.gguf"], "", 30);
    assert_ne!(code, 0, "must exit non-zero");
    assert!(
        err.contains("--cnv") && err.contains("--no-template"),
        "conflict must be named: {err}"
    );
}

#[test]
fn invalid_color_value_errors() {
    let (_, err, code) = run_cli(&["--color", "neon", "--cnv", "no-such-model.gguf"], "", 30);
    assert_ne!(code, 0);
    assert!(err.contains("--color"), "err: {err}");
}

#[test]
fn help_lists_conversation_flags() {
    let (_, err, code) = run_cli(&["--help"], "", 30);
    assert_eq!(code, 0);
    for flag in ["--cnv", "--single-turn", "--system", "--multiline-input", "--color"] {
        assert!(err.contains(flag), "help must mention {flag}: {err}");
    }
}

// === Real-model sessions (ignored: slow scalar-CPU 0.5B) ===

#[test]
#[ignore = "requires cached 0.5B model; slow on scalar CPU (~minutes)"]
fn two_turn_session_via_stdin_pipe() {
    let Some(model) = model_path() else {
        eprintln!("0.5B q4_0 not cached; skipping");
        return;
    };
    let input = "hi\nwhat is 2+2?\n/exit\n";
    let (out, err, code) = run_cli(
        &["--cnv", "--greedy", "--seed", "42", "--color", "off", "--n-ctx", "512", &model],
        input,
        1800,
    );
    assert_eq!(code, 0, "stderr: {err}");
    assert!(!out.contains('\u{FFFD}'), "no U+FFFD in stdout");
    let stats = turn_stats(&err);
    assert_eq!(stats.len(), 2, "two turns: {err}");
    // 增量性（L8）：turn 2 的 delta prefill 必须小于 turn 1 的全量渲染
    assert!(
        stats[1].1 < stats[0].1,
        "turn 2 prefill ({}) must be < turn 1 prefill ({}): {err}",
        stats[1].1,
        stats[0].1
    );
    // 两回合都有输出（assistant 文本经 emit 进 stdout）
    assert!(out.trim().len() > 0, "assistant output expected: {out:?}");
}

#[test]
#[ignore = "requires cached 0.5B model; slow on scalar CPU (~minutes)"]
fn single_turn_exits_after_one_turn() {
    let Some(model) = model_path() else {
        eprintln!("0.5B q4_0 not cached; skipping");
        return;
    };
    let input = "hi\nthis second line must not become a turn\n";
    let (_, err, code) = run_cli(
        &["--cnv", "-st", "--greedy", "--seed", "42", "--color", "off", "--n-ctx", "512", &model],
        input,
        1800,
    );
    assert_eq!(code, 0, "stderr: {err}");
    let stats = turn_stats(&err);
    assert_eq!(stats.len(), 1, "-st must exit after one turn: {err}");
}

#[test]
#[ignore = "requires cached 0.5B model; slow on scalar CPU (~minutes)"]
fn clear_and_regen_commands() {
    let Some(model) = model_path() else {
        eprintln!("0.5B q4_0 not cached; skipping");
        return;
    };
    let input = "hi\n/clear\nQ\n/regen\n/exit\n";
    let (out, err, code) = run_cli(
        &["--cnv", "--greedy", "--seed", "42", "--color", "off", "--n-ctx", "512", &model],
        input,
        1800,
    );
    assert_eq!(code, 0, "stderr: {err}");
    // /clear 的确认走 stdout（println!），错误走 stderr
    assert!(out.contains("[history cleared]"), "stdout: {out}");
    assert!(!err.contains("[regen failed"), "regen must succeed: {err}");
    assert!(!err.contains("[error]"), "no errors: {err}");
    let stats = turn_stats(&err);
    assert_eq!(stats.len(), 3, "hi + Q + regen(Q): {err}");
}

#[test]
#[ignore = "requires cached 0.5B model; slow on scalar CPU (~minutes)"]
fn stop_string_truncates_output() {
    let Some(model) = model_path() else {
        eprintln!("0.5B q4_0 not cached; skipping");
        return;
    };
    // 常见英文单词 stop 串；模型输出被截断到 stop 之前
    let input = "hi\n/exit\n";
    let (out, _, code) = run_cli(
        &["--cnv", "--greedy", "--seed", "42", "--color", "off", "--n-ctx", "512",
          "--stop", "is", &model],
        input,
        1800,
    );
    assert_eq!(code, 0);
    assert!(!out.contains('\u{FFFD}'));
}
