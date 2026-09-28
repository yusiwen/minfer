//! `#[cfg(test)] mod tests` for `src/server/mod.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use std::net::SocketAddr;
use tokio::io::{AsyncReadExt, AsyncWriteExt};

/// Bind an ephemeral port and serve `app` on it.
async fn serve(app: Router) -> SocketAddr {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    addr
}

/// One real HTTP/1.1 exchange, read to EOF (`Connection: close` makes the
/// server close, so `read_to_end` terminates). Returns `(head, body)`.
async fn request(addr: SocketAddr, req: &str) -> (String, String) {
    let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
    stream.write_all(req.as_bytes()).await.unwrap();
    let mut buf = Vec::new();
    stream.read_to_end(&mut buf).await.unwrap();
    let text = String::from_utf8_lossy(&buf).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").expect("response headers");
    (head.to_string(), body.to_string())
}

async fn get(addr: SocketAddr, path: &str) -> (String, String) {
    request(
        addr,
        &format!("GET {path} HTTP/1.1\r\nHost: minfer\r\nConnection: close\r\n\r\n"),
    )
    .await
}

async fn post_json(addr: SocketAddr, path: &str, body: &str) -> (String, String) {
    request(
        addr,
        &format!(
            "POST {path} HTTP/1.1\r\nHost: minfer\r\nContent-Type: application/json\r\n\
             Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        ),
    )
    .await
}

/// An `AppState` with no model: enough for the real router (F8's tests never
/// tokenize — see `Tokenizer::empty`). The returned receiver must stay alive
/// or `job_tx.send` fails, which is a different failure from the one under
/// test.
fn app_state(metrics: Arc<ServerMetrics>) -> (Arc<AppState>, mpsc::Receiver<Job>) {
    let (job_tx, job_rx) = mpsc::channel::<Job>(4);
    let state = Arc::new(AppState {
        job_tx,
        metrics,
        model_name: "test-model".to_string(),
        n_ctx: 64,
        n_ctx_slot: 64,
        tokenizer: Arc::new(Tokenizer::empty()),
        special: crate::models::SpecialTokens {
            eos: 0,
            im_end: None,
        },
        spec_enabled: false,
        chat_template: None,
        created: 0,
    });
    (state, job_rx)
}

/// F8: `/metrics` over a real HTTP connection, with live numbers in it.
#[tokio::test]
async fn metrics_endpoint_renders_over_http() {
    let metrics = Arc::new(ServerMetrics::new());
    metrics.requests_total.store(2, Ordering::SeqCst);
    metrics.jobs_admitted_total.store(1, Ordering::SeqCst);
    metrics.publish_kv(&metrics::KvSnapshot {
        layers: 24,
        rows: 512,
        region_bytes: 1 << 20,
        weights_bytes: 999,
        packed: true,
        ..metrics::KvSnapshot::default()
    });
    let addr = serve(metrics_router(metrics.clone())).await;
    let (head, body) = get(addr, "/metrics").await;

    assert!(head.starts_with("HTTP/1.1 200 OK"), "status line: {head}");
    assert!(
        head.to_ascii_lowercase()
            .contains("content-type: text/plain; version=0.0.4; charset=utf-8"),
        "content type: {head}"
    );
    assert!(body.contains("# TYPE minfer_requests_total counter\n"));
    assert!(body.contains("minfer_requests_total 2\n"));
    assert!(body.contains("minfer_queue_depth 1\n"));
    assert!(body.contains("minfer_kv_layers 24\n"));
    assert!(body.contains("minfer_kv_rows 512\n"));
    assert!(body.contains("minfer_kv_packed 1\n"));
    assert!(body.contains("minfer_memory_weights_bytes 999\n"));
}

/// The full production router carries `/metrics` (merged from its own state),
/// and the pre-existing endpoints are untouched.
#[tokio::test]
async fn the_full_router_serves_metrics_health_and_models() {
    let metrics = Arc::new(ServerMetrics::new());
    let (state, _rx) = app_state(metrics.clone());
    let addr = serve(router(state)).await;

    let (head, body) = get(addr, "/metrics").await;
    assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
    assert!(body.contains("minfer_requests_total"));

    let (head, body) = get(addr, "/health").await;
    assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
    assert!(body.contains("\"status\":\"ok\""), "{body}");

    let (head, body) = get(addr, "/v1/models").await;
    assert!(head.starts_with("HTTP/1.1 200 OK"), "{head}");
    assert!(body.contains("test-model"), "{body}");
}

/// F8: once a shutdown starts, new work is refused with `503` (not queued
/// behind the drain), the refusal is counted, and a scrape shows it.
#[tokio::test]
async fn a_draining_server_refuses_new_work_and_says_so_in_metrics() {
    let metrics = Arc::new(ServerMetrics::new());
    metrics.draining.store(true, Ordering::SeqCst);
    let (state, _rx) = app_state(metrics.clone());
    let addr = serve(router(state)).await;

    let (head, body) = post_json(
        addr,
        "/v1/chat/completions",
        r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}]}"#,
    )
    .await;
    assert!(head.starts_with("HTTP/1.1 503"), "status line: {head}");
    assert!(body.contains("draining"), "body: {body}");
    assert_eq!(metrics.requests_rejected_total.load(Ordering::SeqCst), 1);
    // Nothing was queued, so nothing is in flight.
    assert_eq!(metrics.requests_total.load(Ordering::SeqCst), 0);
    assert_eq!(metrics.in_flight.load(Ordering::SeqCst), 0);

    let (_, body) = get(addr, "/metrics").await;
    assert!(body.contains("minfer_draining 1\n"), "{body}");
    assert!(
        body.contains("minfer_requests_rejected_total 1\n"),
        "{body}"
    );
}

/// A drain that finishes in time reports `Clean`.
#[tokio::test]
async fn bounded_drain_is_clean_when_the_work_ends_in_time() {
    let metrics = ServerMetrics::new();
    let out = bounded_drain(async {}, &metrics, Duration::from_millis(500)).await;
    assert_eq!(out, DrainOutcome::Clean);
}

/// And a drain that does not finish is **bounded** and names what it left:
/// the serve future is `pending`, so only the deadline can end the wait.
#[tokio::test]
async fn bounded_drain_times_out_and_reports_the_abandoned_requests() {
    let metrics = Arc::new(ServerMetrics::new());
    metrics.in_flight.store(3, Ordering::SeqCst);
    let start = std::time::Instant::now();
    let out = bounded_drain(
        std::future::pending::<()>(),
        &metrics,
        Duration::from_millis(50),
    )
    .await;
    let elapsed = start.elapsed();
    assert_eq!(out, DrainOutcome::Forced { in_flight: 3 });
    assert!(
        elapsed >= Duration::from_millis(50),
        "must wait out the deadline, took {elapsed:?}"
    );
    assert!(
        elapsed < Duration::from_secs(10),
        "must be bounded, took {elapsed:?}"
    );
}

#[test]
fn drain_deadline_defaults_and_reports_a_bad_value() {
    assert_eq!(drain_deadline_ms(None), 30_000);
    assert_eq!(drain_deadline_ms(Some("250")), 250);
    assert_eq!(drain_deadline_ms(Some(" 100 ")), 100);
    assert_eq!(drain_deadline_ms(Some("soon")), 30_000);
    assert_eq!(drain_deadline_ms(Some("-1")), 30_000);
    // 0 is a legitimate "stop now" and must not be read as "unset".
    assert_eq!(drain_deadline_ms(Some("0")), 0);
}

/// The in-flight guard is what makes `in_flight` fall, and it is also what a
/// drain waits on — dropping it twice would break the count.
#[test]
fn the_in_flight_guard_counts_exactly_one_request() {
    let metrics = Arc::new(ServerMetrics::new());
    metrics.in_flight.store(1, Ordering::SeqCst);
    {
        let _g = InFlight::new(metrics.clone());
        assert_eq!(metrics.in_flight.load(Ordering::SeqCst), 1);
    }
    assert_eq!(metrics.in_flight.load(Ordering::SeqCst), 0);
    assert_eq!(metrics.requests_completed_total.load(Ordering::SeqCst), 1);
}
