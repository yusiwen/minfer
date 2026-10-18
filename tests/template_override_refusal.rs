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

/// The happy path on `serve` (#602): a good override starts the server and the
/// health endpoint answers. Needs a cached model, so it belongs to the
/// `#[ignore]`d set that `scripts/real_model_gates.sh` runs.
#[test]
#[ignore = "needs a cached model; run with scripts/real_model_gates.sh"]
fn serve_starts_and_answers_health_with_an_override_file() {
    use std::io::{BufRead, BufReader, Read};
    use std::net::TcpStream;
    use std::sync::mpsc;
    use std::time::Duration;

    let model = std::env::var("MINFER_BATCH_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    if !std::path::Path::new(&model).exists() {
        eprintln!("skipped: no cached model at {model}");
        return;
    }
    let template = scratch("serve-ok.jinja", "U:{{ messages[0]['content'] }}");
    let port: u16 = 18099;

    let mut child = Command::new(env!("CARGO_BIN_EXE_minfer"))
        .arg("serve")
        .arg(&model)
        .args(["--port", &port.to_string(), "--chat-template"])
        .arg(&template)
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("spawn minfer serve");

    // The listening banner is on stderr; wait for it, bounded.
    let stderr = child.stderr.take().expect("stderr pipe");
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        for line in BufReader::new(stderr).lines() {
            let Ok(line) = line else { break };
            if tx.send(line).is_err() {
                break;
            }
        }
    });
    let mut log = String::new();
    let listening = loop {
        match rx.recv_timeout(Duration::from_secs(180)) {
            Ok(line) => {
                log.push_str(&line);
                log.push('\n');
                if line.contains("listening on") {
                    break true;
                }
            }
            Err(_) => break false,
        }
    };

    let health = if listening {
        let mut stream = TcpStream::connect(("127.0.0.1", port)).expect("connect to serve");
        use std::io::Write as _;
        stream
            .write_all(b"GET /health HTTP/1.0\r\nHost: localhost\r\n\r\n")
            .unwrap();
        let mut resp = String::new();
        stream.read_to_string(&mut resp).unwrap();
        resp
    } else {
        String::new()
    };

    let _ = child.kill();
    let _ = child.wait();

    assert!(
        listening,
        "the server never printed its listening banner; log so far:\n{log}"
    );
    assert!(
        health.contains("\"status\":\"ok\""),
        "health response was:\n{health}"
    );
}
