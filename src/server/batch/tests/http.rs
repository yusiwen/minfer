//! HTTP refusals: the 503 answer and the SSE error frame.
//!
//! Split out of `src/server/batch/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// #121: the answer a rejected job gets, per transport.
///
/// Drives the real handler functions with the event [`reject`] produces:
/// `collect_response` must surface a **503** rather than read the closed
/// channel as a completed empty answer, and `stream_response` must emit a
/// `data:` **error frame** rather than an empty stream followed by `[DONE]`.
/// No model is involved, so this half runs in CI; the coupling — that a
/// saturated engine really calls `reject` — is the `#[ignore]`d real-model
/// gate below (`a_job_rejected_for_want_of_a_slot_is_answered_with_503`).
///
/// A plain `#[test]` with its own runtime: `reject` runs on the worker (a
/// non-async thread, as in production), and `blocking_send` correctly refuses
/// to block a runtime thread — so it is called *outside* `block_on`.
#[test]
fn a_rejected_job_answers_503_and_an_sse_error_frame() {
    use std::sync::Arc;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let job = |tx: mpsc::Sender<StreamEvent>| Job {
        input_ids: vec![1, 2, 3],
        params: sampling_params(4),
        tx,
    };

    // Non-streaming: the error reaches the handler as `Err(e)`, and `e` is a
    // 503 for the HTTP layer.
    let (tx, rx) = mpsc::channel::<StreamEvent>(4);
    let e = reject(job(tx), ApiError::unavailable("no idle slot"));
    assert_eq!(e.status, 503, "the engine rejects with 503");
    let non_streaming_metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let from_stream = rt
        .block_on(crate::server::collect_response(rx, &non_streaming_metrics))
        .expect_err("a rejected job must not read as a completed empty answer");
    assert_eq!(from_stream.status, 503, "{}", from_stream.message);
    assert!(
        from_stream.message.contains("no idle slot"),
        "the reason must survive: {}",
        from_stream.message
    );
    assert_eq!(
        crate::server::error_response(&from_stream).status(),
        axum::http::StatusCode::SERVICE_UNAVAILABLE
    );
    // #157: the non-streaming answer is counted, by class, exactly once — and the
    // completion counter is not the signal for it.
    assert_eq!(
        non_streaming_metrics.snapshot().requests_failed,
        vec![("unavailable_error", 1)]
    );

    // Streaming: the same event is an SSE error frame — not a silent close.
    // The `Sse` response itself is built inside the runtime (axum's keep-alive
    // arms a timer), while `reject` stays outside it.
    let (tx, rx) = mpsc::channel::<StreamEvent>(4);
    let _ = reject(job(tx), ApiError::unavailable("no idle slot"));
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let body = rt.block_on(async move {
        let resp = crate::server::stream_response(
            "chatcmpl-test",
            "test-model",
            0,
            rx,
            crate::server::InFlight::new(metrics.clone()),
            metrics,
            3,
        );
        axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("SSE body")
    });
    let body = String::from_utf8_lossy(&body);
    assert!(
        body.contains("no idle slot"),
        "the SSE error frame is missing: {body}"
    );
    assert!(
        body.contains("\"code\":503"),
        "the frame must carry the status: {body}"
    );
    assert!(
        !body.contains("\"finish_reason\":\"stop\""),
        "a rejected request must not look finished: {body}"
    );
}

/// #157: the streaming arm counts the error frame by class, and `/metrics`
/// renders it as a labelled series (never as a bare `…_total 0`).
#[test]
fn a_streamed_error_frame_is_counted_by_class() {
    use std::sync::Arc;

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("test runtime");
    let (tx, rx) = mpsc::channel::<StreamEvent>(4);
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let _ = reject(
        Job {
            input_ids: vec![1, 2, 3],
            params: sampling_params(4),
            tx,
        },
        ApiError::server("decode failed"),
    );
    let metrics_for_body = metrics.clone();
    rt.block_on(async move {
        let resp = crate::server::stream_response(
            "chatcmpl-test",
            "test-model",
            0,
            rx,
            crate::server::InFlight::new(metrics_for_body.clone()),
            metrics_for_body,
            3,
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .expect("SSE body");
        assert!(
            String::from_utf8_lossy(&body).contains("decode failed"),
            "the frame carries the error"
        );
    });

    assert_eq!(
        metrics.snapshot().requests_failed,
        vec![("server_error", 1)],
        "one request, one counted failure, in its own class"
    );
    let text = metrics.render();
    assert!(
        text.contains("# TYPE minfer_requests_failed_total counter"),
        "the family is declared: {text}"
    );
    assert!(
        text.contains("minfer_requests_failed_total{class=\"server_error\"} 1"),
        "the sample is labelled: {text}"
    );
    assert!(
        !text.contains("minfer_requests_failed_total 1"),
        "a labelled family must not also emit an unlabelled sample: {text}"
    );
}

/// #150: a job that arrives while the only slot is busy **waits** and is
/// served, instead of being answered `503 no idle slot`.
///
/// One slot, two jobs queued before the loop starts, so the second one is
/// deterministically behind a running request. Both must `Finish` with tokens,
/// `jobs_dropped_total` must stay 0, and the queue arithmetic must balance.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn a_job_queued_behind_a_busy_slot_is_served_when_it_frees() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[#150] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let mut engine = BatchEngine::new(&*model, 1, 512).expect("engine");
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let mut rxs = Vec::new();
    for prompt in ["Alpha", "Beta"] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
        rxs.push(rx);
        job_tx
            .blocking_send(Job {
                input_ids: tok.encode(prompt),
                params: sampling_params(8),
                tx,
            })
            .expect("queue job");
        // The HTTP handler's side of the contract (F8): accepted + in flight.
        metrics.requests_total.fetch_add(1, Ordering::SeqCst);
        metrics.in_flight.fetch_add(1, Ordering::SeqCst);
    }
    drop(job_tx); // no more senders: the loop drains and exits
    super::serve_loop(
        &*model,
        &tok,
        job_rx,
        &mut engine,
        &metrics,
        super::QUEUE_MAX,
    );

    let s = metrics.snapshot();
    assert_eq!(
        s.jobs_dropped_total, 0,
        "a job that waits for a slot was not dropped: queueing is #150"
    );
    assert_eq!(
        s.requests_total - s.queue_depth,
        2,
        "both jobs left the queue (one ran, one waited and then ran)"
    );
    assert_eq!(s.worker_pending, 0, "the queue drained");
    assert_eq!(s.running, 0, "nothing is running once the loop ends");
    assert_eq!(
        s.worker_stalled_total, 0,
        "a healthy one-slot run never trips the no-progress bound (#196)"
    );

    let mut served = 0usize;
    for mut rx in rxs {
        let mut errs = Vec::new();
        let mut finishes = Vec::new();
        while let Some(ev) = rx.blocking_recv() {
            match ev {
                StreamEvent::Err(e) => errs.push(e),
                StreamEvent::Finish { tokens, .. } => finishes.push(tokens),
                StreamEvent::Text(_) => {}
            }
        }
        assert!(
            errs.is_empty(),
            "a queued job must be served, not refused: {}",
            errs.iter()
                .map(|e| e.message.as_str())
                .collect::<Vec<_>>()
                .join("; ")
        );
        assert_eq!(finishes.len(), 1, "exactly one Finish per served job");
        assert!(finishes[0] > 0, "the served job produced tokens");
        served += 1;
    }
    assert_eq!(served, 2, "one slot serves both jobs when one waits");
}

/// #150: the queue is **bounded**, and a job beyond the bound is refused loudly.
///
/// One slot, three jobs and a queue depth of 1: `Alpha` runs, `Beta` waits, and
/// `Gamma` — the newest — is answered `503 the queue is full` through [`reject`]
/// and counted in `jobs_dropped_total`, exactly once. Two are served, so the
/// bound is a bound and not a rejection of everything after the first.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn a_job_beyond_the_queue_bound_is_answered_with_503() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[#150] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let mut engine = BatchEngine::new(&*model, 1, 512).expect("engine");
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let mut rxs = Vec::new();
    for prompt in ["Alpha", "Beta", "Gamma"] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
        rxs.push(rx);
        job_tx
            .blocking_send(Job {
                input_ids: tok.encode(prompt),
                params: sampling_params(8),
                tx,
            })
            .expect("queue job");
        metrics.requests_total.fetch_add(1, Ordering::SeqCst);
        metrics.in_flight.fetch_add(1, Ordering::SeqCst);
    }
    drop(job_tx);
    super::serve_loop(&*model, &tok, job_rx, &mut engine, &metrics, 1);

    let s = metrics.snapshot();
    assert_eq!(
        s.jobs_dropped_total, 1,
        "a queue depth of 1 drops exactly the one job beyond it"
    );
    assert_eq!(
        s.requests_total - s.queue_depth,
        3,
        "a refused job also left the queue, so the arithmetic balances"
    );

    let mut served = 0usize;
    let mut refused = 0usize;
    for mut rx in rxs {
        let mut errs = Vec::new();
        let mut finishes = 0usize;
        while let Some(ev) = rx.blocking_recv() {
            match ev {
                StreamEvent::Err(e) => errs.push(e),
                StreamEvent::Finish { .. } => finishes += 1,
                StreamEvent::Text(_) => {}
            }
        }
        match (errs.len(), finishes) {
            (0, 1) => served += 1,
            (1, 0) => {
                assert_eq!(errs[0].status, 503, "{}", errs[0].message);
                assert_eq!(
                    errs[0].message,
                    super::QUEUE_FULL_MESSAGE,
                    "the refusal names the bound the client hit"
                );
                refused += 1;
            }
            (e, f) => panic!("a job answered {e} error(s) and {f} Finish(es)"),
        }
    }
    assert_eq!(
        (served, refused),
        (2, 1),
        "the queue depth bounds the wait, it does not reject the first request"
    );
}

/// #150: a drain answers what it will not run.
///
/// A queued job that outlives the drain deadline would be abandoned without an
/// answer (#121's silent drop, one layer up), so `serve_loop` answers every
/// queued-but-unadmitted job with `503 server is draining` as soon as the flag
/// is set. Both jobs here were accepted before the drain and neither was
/// admitted, so both are answered — exactly once — and nothing runs.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn a_drain_answers_queued_jobs_with_503() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[#150] no cached model; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let mut engine = BatchEngine::new(&*model, 1, 512).expect("engine");
    let metrics = Arc::new(crate::server::metrics::ServerMetrics::new());
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let mut rxs = Vec::new();
    for prompt in ["Alpha", "Beta"] {
        let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
        rxs.push(rx);
        job_tx
            .blocking_send(Job {
                input_ids: tok.encode(prompt),
                params: sampling_params(8),
                tx,
            })
            .expect("queue job");
        metrics.requests_total.fetch_add(1, Ordering::SeqCst);
        metrics.in_flight.fetch_add(1, Ordering::SeqCst);
    }
    drop(job_tx);
    metrics.draining.store(true, Ordering::SeqCst);
    super::serve_loop(
        &*model,
        &tok,
        job_rx,
        &mut engine,
        &metrics,
        super::QUEUE_MAX,
    );

    let s = metrics.snapshot();
    assert_eq!(s.jobs_dropped_total, 2, "no queued job is silently dropped");
    assert_eq!(s.running, 0, "a draining worker admits nothing new");
    for (i, mut rx) in rxs.into_iter().enumerate() {
        let mut errs = Vec::new();
        let mut finishes = 0usize;
        while let Some(ev) = rx.blocking_recv() {
            match ev {
                StreamEvent::Err(e) => errs.push(e),
                StreamEvent::Finish { .. } => finishes += 1,
                StreamEvent::Text(_) => {}
            }
        }
        assert_eq!(finishes, 0, "job {i}: a drain does not run queued work");
        assert_eq!(errs.len(), 1, "job {i}: exactly one answer");
        assert_eq!(errs[0].status, 503, "job {i}: {}", errs[0].message);
        assert_eq!(errs[0].message, super::DRAINING_MESSAGE, "job {i}");
    }
}
