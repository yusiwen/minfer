//! #726: a request's lifecycle **is** the log — one `request.admitted` and one
//! `request.finished` line per request, naming the same request id, and nothing
//! at all below the `info` threshold.
//!
//! The toy GGUF (`tests/fixtures/toy/`, 113 KB, checked in) is the model here on
//! purpose: CI has no cached real model, and a lifecycle gate that only ran on
//! the maintainer's box would not be a gate.

use super::*;

fn toy_model() -> Option<(Box<dyn ModelDef>, Tokenizer)> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/fixtures/toy/toy-qwen2-f32.gguf");
    let gguf = minfer::gguf::load_gguf_model(&path).ok()?;
    let tok = Tokenizer::load(&gguf.parts[0].ctx).ok()?;
    let model = minfer::models::load_model(&gguf).ok()?;
    Some((model, tok))
}

/// Drive exactly one request through the real `serve_loop`, returning its event
/// lines at `level` and the metrics snapshot afterwards.
fn one_request(
    level: crate::server::log::Level,
    id: &str,
) -> (Vec<String>, crate::server::metrics::ServerMetrics) {
    let Some((model, tok)) = toy_model() else {
        eprintln!("[#726] the toy fixture is missing; skipping");
        return (Vec::new(), crate::server::metrics::ServerMetrics::new());
    };
    // The engine and its channels live **outside** the capture: `capture` only
    // arms the calling thread's sink, and the lifecycle lines are emitted by
    // `serve_loop`, which runs on this thread.
    let mut engine = BatchEngine::new(&*model, 1, 64).expect("engine");
    let metrics = crate::server::metrics::ServerMetrics::new();
    let (job_tx, job_rx) = mpsc::channel::<Job>(4);
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(64);
    job_tx
        .blocking_send(Job {
            id: id.to_string(),
            admitted: Instant::now(),
            input_ids: tok.encode("hi"),
            params: sampling_params(2),
            tx,
        })
        .expect("queue the request");
    drop(job_tx);
    let (_, lines) = crate::server::log::capture(level, || {
        serve_loop(&*model, &tok, job_rx, &mut engine, &metrics, 1)
    });
    // The request finished normally: one `Finish`, no error frame.
    let mut finishes = 0;
    while let Ok(ev) = rx.try_recv() {
        match ev {
            StreamEvent::Finish { .. } => finishes += 1,
            StreamEvent::Err(e) => panic!("the toy request failed: {}", e.message),
            StreamEvent::Text(_) => {}
        }
    }
    assert_eq!(finishes, 1, "one request, one Finish");
    (lines, metrics)
}

fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let prefix = format!("{key}=");
    line.split(' ')
        .find_map(|part| part.strip_prefix(prefix.as_str()))
}

#[test]
fn info_prints_one_admitted_and_one_finished_line_with_the_same_id() {
    let id = "chatcmpl-726-test";
    let (lines, metrics) = one_request(crate::server::log::Level::Info, id);
    let admitted: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("event=request.admitted"))
        .collect();
    let finished: Vec<&String> = lines
        .iter()
        .filter(|l| l.contains("event=request.finished"))
        .collect();
    assert_eq!(admitted.len(), 1, "one admitted line: {lines:#?}");
    assert_eq!(finished.len(), 1, "one finished line: {lines:#?}");
    // Both lines name the request the client saw.
    for line in [admitted[0], finished[0]] {
        assert_eq!(field(line, "id"), Some(id), "{line}");
    }
    // The admitted line carries the slot and the queue depth; the finished one
    // the reason, the token count and the elapsed time.
    assert_eq!(field(admitted[0], "slot"), Some("0"));
    assert!(field(admitted[0], "queued").is_some(), "{admitted:?}");
    assert!(field(finished[0], "reason").is_some(), "{finished:?}");
    assert!(field(finished[0], "tokens").is_some(), "{finished:?}");
    assert!(field(finished[0], "ms").is_some(), "{finished:?}");
    // The log must not have changed what `/metrics` reports.
    let s = metrics.snapshot();
    assert_eq!(
        s.requests_total, 0,
        "the handler records this, not the worker"
    );
    assert_eq!(s.jobs_dropped_total, 0);
    assert_eq!(s.worker_pending, 0);
    assert_eq!(s.running, 0);
}

#[test]
fn warn_and_off_print_no_lifecycle_lines() {
    for level in [
        crate::server::log::Level::Warn,
        crate::server::log::Level::Off,
    ] {
        let (lines, _) = one_request(level, "chatcmpl-726-quiet");
        assert!(
            !lines
                .iter()
                .any(|l| l.contains("event=request.admitted")
                    || l.contains("event=request.finished")),
            "{level:?} must not print the lifecycle: {lines:#?}"
        );
    }
}
