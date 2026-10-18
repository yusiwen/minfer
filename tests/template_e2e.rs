//! End-to-end checks that a template choice reaches the running server (#602, #607).
//!
//! These need a cached model, so the whole file is `#[ignore]`d and belongs to the
//! set `scripts/real_model_gates.sh` runs. Two questions are answered here that the
//! unit tests cannot:
//!
//!   1. Does a `--chat-template <FILE>` actually drive a **request** on `serve` and
//!      `viz`? The marker template loads (its refusal only fires for a single
//!      message, which the load-time canary never sends) and refuses at request
//!      time, so a 400 carrying its text proves the file was used; a start without
//!      the override answers the same request 200.
//!   2. Is the no-template notice printed **exactly once, before the listening
//!      banner**, on each path? The artifact is a one-byte rename of the GGUF KV key
//!      `tokenizer.chat_template` — the key is length-prefixed, so the file keeps its
//!      size and every tensor offset stays valid, and the loader sees no template.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::time::Duration;

/// The notice text asserted below. It is `template::CHATML_NOTICE`, spelled here
/// because an integration test cannot link the binary crate's items.
const CHATML_NOTICE: &str = "using the generic ChatML renderer";

/// A long-lived `minfer serve` / `minfer viz` child with its stderr drained.
struct Server {
    child: Child,
    log: String,
    rx: Receiver<String>,
    port: u16,
    /// The line each mode prints once it is serving: `serve` says "listening on",
    /// `viz` says "viz server on".
    banner: &'static str,
}

impl Server {
    fn spawn(mode: &str, model: &str, port: u16, template: Option<&Path>) -> Server {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_minfer"));
        cmd.arg(mode).arg(model).args(["--port", &port.to_string()]);
        if let Some(t) = template {
            cmd.arg("--chat-template").arg(t);
        }
        let mut child = cmd
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("spawn minfer {mode}: {e}"));
        let stderr = child.stderr.take().expect("stderr pipe");
        let (tx, rx) = mpsc::channel();
        // The banner (and the notice) are on stderr.
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines() {
                let Ok(line) = line else { break };
                if tx.send(line).is_err() {
                    break;
                }
            }
        });
        Server {
            child,
            log: String::new(),
            rx,
            port,
            banner: if mode == "serve" {
                "listening on"
            } else {
                "viz server on"
            },
        }
    }

    /// Drain stderr until the listening banner appears; true when it did.
    fn wait_listening(&mut self) -> bool {
        loop {
            match self.rx.recv_timeout(Duration::from_secs(180)) {
                Ok(line) => {
                    let listening = line.contains(self.banner);
                    self.log.push_str(&line);
                    self.log.push('\n');
                    if listening {
                        return true;
                    }
                }
                Err(_) => return false,
            }
        }
    }

    /// One HTTP/1.0 request; returns `(status, body)`.
    fn request(&self, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
        let mut stream = TcpStream::connect(("127.0.0.1", self.port))
            .unwrap_or_else(|e| panic!("connect to {}: {e}", self.port));
        let mut req = format!("{method} {path} HTTP/1.0\r\nHost: localhost\r\n");
        if let Some(b) = body {
            req.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                b.len()
            ));
        }
        req.push_str("\r\n");
        if let Some(b) = body {
            req.push_str(b);
        }
        stream.write_all(req.as_bytes()).unwrap();
        let mut raw = String::new();
        stream.read_to_string(&mut raw).unwrap();
        let status = raw
            .split_whitespace()
            .nth(1)
            .and_then(|c| c.parse().ok())
            .unwrap_or(0);
        let body = raw.split("\r\n\r\n").nth(1).unwrap_or_default().to_string();
        (status, body)
    }

    fn log(&self) -> &str {
        &self.log
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn cached_model() -> Option<PathBuf> {
    let model = std::env::var("MINFER_BATCH_TEST_MODEL").unwrap_or_else(|_| {
        format!(
            "{}/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let path = PathBuf::from(model);
    path.exists().then_some(path)
}

/// The model without its `tokenizer.chat_template`, made by renaming the KV key's
/// first byte in a copy.
///
/// GGUF keys are length-prefixed, so a same-length rename keeps the file size and
/// every tensor offset valid; the loader ignores the unknown key and reports no
/// template. The assertions are the two properties the trick depends on: the key
/// occurs exactly once, and the size does not change.
fn template_less_model() -> Option<PathBuf> {
    let src = cached_model()?;
    let dst = std::env::temp_dir()
        .join("minfer-template-less")
        .join(src.file_name()?);
    if dst.exists() {
        return Some(dst);
    }
    std::fs::create_dir_all(dst.parent()?).ok()?;
    std::fs::copy(&src, &dst).ok()?;
    let before = std::fs::metadata(&dst).ok()?.len();
    let mut data = std::fs::read(&dst).ok()?;
    let needle = b"tokenizer.chat_template";
    let hits: Vec<usize> = data
        .windows(needle.len())
        .enumerate()
        .filter(|(_, w)| *w == needle)
        .map(|(i, _)| i)
        .collect();
    assert_eq!(
        hits.len(),
        1,
        "the KV key must occur exactly once for a same-length rename; found {}",
        hits.len()
    );
    data[hits[0]] = b'X';
    std::fs::write(&dst, &data).ok()?;
    assert_eq!(
        std::fs::metadata(&dst).ok()?.len(),
        before,
        "the rename must not change the file size"
    );
    Some(dst)
}

/// A template that loads (the canary sends 4 and 2 messages) but refuses a
/// one-message request — the shape both the single-turn CLI and `/viz/run` send.
const MARKER_TEMPLATE: &str = "{% if messages|length == 1 %}{{ raise_exception(\"override-marker\") }}{% endif %}U:{{ messages[0][\"content\"] }}";

fn marker_file() -> PathBuf {
    let dir = std::env::temp_dir().join("minfer-template-e2e");
    std::fs::create_dir_all(&dir).expect("scratch dir");
    let path = dir.join("marker.jinja");
    std::fs::write(&path, MARKER_TEMPLATE).expect("write marker template");
    path
}

#[test]
#[ignore = "needs a cached model; run with scripts/real_model_gates.sh"]
fn serve_starts_and_answers_health_with_an_override_file() {
    let Some(model) = cached_model() else {
        eprintln!("skipped: no cached model");
        return;
    };
    let template = marker_file();
    // The marker only fires on a one-message request; a plain /health never renders.
    let mut server = Server::spawn("serve", &model.to_string_lossy(), 18099, Some(&template));
    assert!(
        server.wait_listening(),
        "the server never printed its banner; log so far:\n{}",
        server.log()
    );
    let (status, body) = server.request("GET", "/health", None);
    assert_eq!(status, 200, "health status; body:\n{body}");
    assert!(body.contains("\"status\":\"ok\""), "health body:\n{body}");
}

#[test]
#[ignore = "needs a cached model; run with scripts/real_model_gates.sh"]
fn serve_renders_from_the_override_at_request_time() {
    let Some(model) = cached_model() else {
        eprintln!("skipped: no cached model");
        return;
    };
    let template = marker_file();
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":1}"#;

    let mut with = Server::spawn("serve", &model.to_string_lossy(), 18100, Some(&template));
    assert!(
        with.wait_listening(),
        "no banner; log so far:\n{}",
        with.log()
    );
    let (status, response) = with.request("POST", "/v1/chat/completions", Some(body));
    assert_eq!(
        status, 400,
        "the override's refusal must surface as 400; body:\n{response}"
    );
    assert!(
        response.contains("override-marker"),
        "the override template's own text must be in the error; body:\n{response}"
    );

    // Control: the same request against the GGUF's own template is not refused, so
    // the 400 above is the file's doing and not the artifact's.
    let mut without = Server::spawn("serve", &model.to_string_lossy(), 18101, None);
    assert!(
        without.wait_listening(),
        "no banner; log so far:\n{}",
        without.log()
    );
    let (status, response) = without.request("POST", "/v1/chat/completions", Some(body));
    assert_eq!(
        status, 200,
        "without the override the request must render; body:\n{response}"
    );
    assert!(!response.contains("override-marker"), "{response}");
}

#[test]
#[ignore = "needs a cached model; run with scripts/real_model_gates.sh"]
fn viz_renders_from_the_override_at_request_time() {
    let Some(model) = cached_model() else {
        eprintln!("skipped: no cached model");
        return;
    };
    let template = marker_file();
    let body = r#"{"prompt":"hi","max_tokens":1}"#;

    let mut with = Server::spawn("viz", &model.to_string_lossy(), 18102, Some(&template));
    assert!(
        with.wait_listening(),
        "no banner; log so far:\n{}",
        with.log()
    );
    let (status, response) = with.request("POST", "/viz/run", Some(body));
    assert_eq!(
        status, 400,
        "the override's refusal must surface as 400; body:\n{response}"
    );
    assert!(
        response.contains("override-marker"),
        "the override template's own text must be in the error; body:\n{response}"
    );

    let mut without = Server::spawn("viz", &model.to_string_lossy(), 18103, None);
    assert!(
        without.wait_listening(),
        "no banner; log so far:\n{}",
        without.log()
    );
    let (status, response) = without.request("POST", "/viz/run", Some(body));
    assert_eq!(
        status, 200,
        "without the override the request must render; body:\n{response}"
    );
    assert!(!response.contains("override-marker"), "{response}");
}

#[test]
#[ignore = "needs a cached model; run with scripts/real_model_gates.sh"]
fn the_chatml_notice_is_printed_once_before_the_banner_on_serve_and_viz() {
    let Some(model) = template_less_model() else {
        eprintln!("skipped: no cached model");
        return;
    };
    for (mode, port) in [("serve", 18104u16), ("viz", 18105)] {
        let mut server = Server::spawn(mode, &model.to_string_lossy(), port, None);
        assert!(
            server.wait_listening(),
            "{mode}: no banner; log so far:\n{}",
            server.log()
        );
        let log = server.log();
        let lines: Vec<&str> = log.lines().collect();
        let notices: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains(CHATML_NOTICE))
            .map(|(i, _)| i)
            .collect();
        let banner = lines
            .iter()
            .position(|l| l.contains(server.banner))
            .expect("the banner was seen");
        assert_eq!(
            notices.len(),
            1,
            "{mode}: the notice must be printed exactly once; log:\n{log}"
        );
        assert!(
            notices[0] < banner,
            "{mode}: the notice must precede the banner; log:\n{log}"
        );
    }
}
