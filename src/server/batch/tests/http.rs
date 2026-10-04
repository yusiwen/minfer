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
    let from_stream = rt
        .block_on(crate::server::collect_response(rx))
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
/// #121: a saturated engine **answers** the job it cannot place.
///
/// One slot, two jobs queued before the loop starts, so the rejection is
/// deterministic (the setup F8's round 2 already used) — but here both
/// receivers are kept and read. `A` is served (`Finish`, tokens > 0); `B`
/// gets **exactly one** `StreamEvent::Err` with status 503 and the
/// `no idle slot` message, and nothing else. `jobs_dropped_total` moves by
/// exactly one.
///
/// Before the fix `B`'s channel closed with no events at all, which the
/// handler rendered as HTTP 200 with empty content.
///
/// Real-model gate (`#[ignore]`: CI has no cached GGUF).
#[test]
#[ignore = "needs the cached 0.5B GGUF (CI has no model)"]
fn a_job_rejected_for_want_of_a_slot_is_answered_with_503() {
    use std::sync::atomic::Ordering;
    use std::sync::Arc;

    let Some(path) = cached_model() else {
        eprintln!("[f8] no cached model; skipping");
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
                // Two tokens, so the served request finishes immediately and
                // the second job is rejected in the very first `admit` pass.
                params: sampling_params(2),
                tx,
            })
            .expect("queue job");
        // The HTTP handler's side of the contract (F8): accepted + in flight.
        metrics.requests_total.fetch_add(1, Ordering::SeqCst);
        metrics.in_flight.fetch_add(1, Ordering::SeqCst);
    }
    drop(job_tx); // no more senders: the loop drains and exits
    super::serve_loop(&*model, &tok, job_rx, &mut engine, &metrics);

    let s = metrics.snapshot();
    assert_eq!(
        s.jobs_dropped_total, 1,
        "exactly one job could not be placed on the one slot"
    );

    // Read both channels: one request was served, one was answered with an
    // error. Classify by content, not by index, so the gate cannot pass
    // because the two happened to be swapped — and name the silent-drop
    // shape explicitly, so the *mutation* that puts it back reports the
    // defect instead of a confusing "the served job must finish".
    let mut served = 0usize;
    let mut rejected = 0usize;
    for mut rx in rxs {
        let mut events = Vec::new();
        while let Some(ev) = rx.blocking_recv() {
            events.push(ev);
        }
        let err = events.iter().find_map(|ev| match ev {
            StreamEvent::Err(e) => Some(e),
            _ => None,
        });
        let finish = events.iter().find_map(|ev| match ev {
            StreamEvent::Finish { tokens, .. } => Some(*tokens),
            _ => None,
        });
        match (err, finish) {
            (Some(e), None) => {
                assert_eq!(
                    e.status, 503,
                    "a rejected job is unavailable: {}",
                    e.message
                );
                assert!(
                    e.message.contains("no idle slot"),
                    "the reason must reach the client: {}",
                    e.message
                );
                assert_eq!(
                    events.len(),
                    1,
                    "the rejected job gets exactly the error, then the channel closes"
                );
                rejected += 1;
            }
            (None, Some(tokens)) => {
                assert!(tokens > 0, "the served job produced {tokens} tokens");
                served += 1;
            }
            (Some(_), Some(_)) => panic!(
                "a job cannot be both rejected and finished ({} event(s))",
                events.len()
            ),
            (None, None) => panic!(
                "a job's channel closed with {} event(s) — the #121 silent drop is back \
                 (the handler would answer HTTP 200 with empty content)",
                events.len()
            ),
        }
    }
    assert_eq!(
        (served, rejected),
        (1, 1),
        "one slot serves one request and answers the other"
    );
    // #196: checked *after* the channels are read, so a wedge — which answers
    // the served run with `500 the worker stalled` instead of a `Finish` —
    // surfaces as the terminal error the client actually got, above, and this
    // is the assertion that names the guard itself.
    assert_eq!(
        s.worker_stalled_total, 0,
        "the worker tripped its counted no-progress bound: a one-slot run that \
         serves one job and rejects the other must never stall (#196)"
    );
}
