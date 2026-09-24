//! OpenAI-compatible HTTP server (OPENAI-CHAT-API-PLAN.md).
//!
//! axum + tokio; a dedicated std worker thread runs inference serially and
//! pushes per-request events into per-request `tokio::sync::mpsc` channels,
//! which the async handlers drain (backpressure = client consumption speed).

pub mod batch;
pub mod chat;
pub mod metrics;
pub mod slot;
pub mod types;
pub mod viz;

use std::convert::Infallible;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use axum::extract::State;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures_util::stream::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;

use crate::gguf::GgufModel;
use crate::models::ModelDef;
use crate::tokenizer::Tokenizer;

use chat::{Job, StreamEvent};
use metrics::ServerMetrics;
use types::{ApiError, ChatCompletionRequest, SamplingParams};

/// Shared server state (handlers only; the worker owns the model + slots).
pub struct AppState {
    pub job_tx: mpsc::Sender<Job>,
    /// F8: the observability registry. The handler writes the request-lifecycle
    /// counters; the worker publishes the queue/engine/KV readings.
    pub metrics: Arc<ServerMetrics>,
    pub model_name: String,
    /// The whole arena. C7: an admission may repartition it, so the request bound
    /// is this, not `n_ctx_slot` (which is only the startup share a slot holds
    /// before a long request reclaims the idle slots above it).
    pub n_ctx: usize,
    pub n_ctx_slot: usize,
    pub tokenizer: Arc<Tokenizer>,
    /// F2 (#47): the EOG ids a grammar needs, resolved once from the model (the
    /// handler compiles the per-request grammar before queueing the job, so it
    /// needs them here rather than only inside the worker).
    pub special: crate::models::SpecialTokens,
    /// F2: whether the worker runs the speculative draft engine — a grammar and a
    /// verify round are mutually exclusive, and the handler must refuse the
    /// combination before queueing.
    pub spec_enabled: bool,
    pub chat_template: Option<String>,
    pub created: i64,
}

/// Build the axum router (pure OpenAI API; the viz live endpoints live under
/// `minfer viz`, see `server::viz`).
///
/// The two halves are stated and merged **after** `with_state`, so `/metrics`
/// carries its own `Arc<ServerMetrics>` state and needs neither the tokenizer nor
/// the job channel: a scrape keeps working while the model is busy, and the
/// endpoint is testable without a model ([`metrics_router`]).
pub fn router(state: Arc<AppState>) -> Router {
    let metrics = state.metrics.clone();
    let api = Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .with_state(state);
    api.merge(metrics_router(metrics))
        .layer(tower_http::cors::CorsLayer::permissive())
}

/// The `/metrics` route on its own state (F8) — see [`router`] for why it is
/// split out.
pub fn metrics_router(metrics: Arc<ServerMetrics>) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .with_state(metrics)
}

/// Prometheus text exposition; see [`metrics::render`] for the metric names and
/// units. Rendering only reads atomics, so scraping cannot perturb generation.
async fn metrics_handler(State(metrics): State<Arc<ServerMetrics>>) -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, metrics::CONTENT_TYPE)],
        metrics.render(),
    )
        .into_response()
}

/// Start the server: spawn the inference worker, then serve until a shutdown
/// signal, drain with a bound, and return. `model` and `slots` move into the
/// worker thread; handlers only share the job channel + tokenizer.
#[allow(clippy::too_many_arguments)]
pub fn run(
    model: Box<dyn ModelDef>,
    tokenizer: Tokenizer,
    gguf: &GgufModel,
    port: u16,
    n_ctx: usize,
    n_slots: usize,
    spec_cfg: Option<crate::spec::SpecConfig>,
    slots_file: Option<String>,
) {
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let slots = slot::new_slots(n_slots, n_ctx);
    let n_ctx_slot = n_ctx / n_slots.max(1);
    let metrics = Arc::new(ServerMetrics::new());

    // F7 (#50): the template is validated before the worker starts, so an
    // unrenderable one refuses to serve instead of 500-ing per request. A GGUF
    // with no template at all still uses ChatML, and says so once.
    let template = chat_template_from_gguf(&gguf.parts[0].data);
    match template.as_deref() {
        Some(t) => {
            if let Err(e) = crate::template::validate(t) {
                eprintln!("Error: {}", e.message());
                return;
            }
        }
        None => eprintln!(
            "Notice: this GGUF has no tokenizer.chat_template; using the generic ChatML renderer"
        ),
    }

    // F8: a completion signal, so `run` can wait for the worker *boundedly* after
    // a clean drain instead of joining it (see `serve_with_shutdown`).
    let (worker_done_tx, worker_done_rx) = std::sync::mpsc::channel::<()>();
    // F2: read what the handler needs before the model and the spec config move
    // into the worker thread.
    let special = model.special_tokens();
    let spec_enabled = spec_cfg.is_some();
    let worker_tokenizer = tokenizer.clone();
    let worker_metrics = metrics.clone();
    let worker = std::thread::spawn(move || {
        chat::worker_loop(
            model,
            worker_tokenizer,
            slots,
            job_rx,
            spec_cfg,
            slots_file,
            worker_metrics,
        );
        let _ = worker_done_tx.send(());
    });

    let model_name = gguf
        .parts
        .first()
        .and_then(|p| p.ctx.get_key_val_str("general.name"))
        .unwrap_or_else(|| "minfer-model".to_string());

    let state = Arc::new(AppState {
        job_tx,
        metrics: metrics.clone(),
        model_name,
        n_ctx,
        n_ctx_slot,
        tokenizer: Arc::new(tokenizer),
        special,
        spec_enabled,
        chat_template: template,
        created: now_unix(),
    });

    let drain = Duration::from_millis(drain_deadline_ms(
        std::env::var("MINFER_DRAIN_MS").ok().as_deref(),
    ));
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    let outcome = runtime.block_on(async move {
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| {
                eprintln!("Error: failed to bind {addr}: {e}");
                std::process::exit(1);
            });
        eprintln!("minfer server listening on http://{addr}");
        eprintln!(
            "[server] graceful drain: on SIGINT/SIGTERM, new requests are refused and in-flight \
             requests get up to {} ms (MINFER_DRAIN_MS)",
            drain.as_millis()
        );
        serve_with_shutdown(listener, router(state), metrics.clone(), drain).await
    });

    // Only a clean drain is worth waiting for the worker at all, and even then
    // bounded: an SSE client that never disconnects must not be able to keep the
    // process alive (the ticket's named trap). A forced drain returns immediately.
    if outcome == DrainOutcome::Clean {
        match worker_done_rx.recv_timeout(drain) {
            Ok(()) => eprintln!("[server] worker stopped; exiting"),
            Err(_) => eprintln!(
                "[server] worker did not stop within {} ms after a clean drain; exiting anyway",
                drain.as_millis()
            ),
        }
    }
    let _ = worker;
}

/// What a drain did: every in-flight request finished, or the deadline expired
/// with `in_flight` still running.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    Clean,
    Forced { in_flight: u64 },
}

/// `MINFER_DRAIN_MS` — how long a graceful drain may take. Default 30 s.
///
/// A value that is not a whole number of milliseconds is reported and the
/// default is used: silently clamping a typo to 0 would turn every shutdown into
/// an abrupt one, which is the opposite of what the flag is for. Pure, so the
/// parse is unit-tested without a server.
pub fn drain_deadline_ms(raw: Option<&str>) -> u64 {
    const DEFAULT: u64 = 30_000;
    match raw {
        None => DEFAULT,
        Some(v) => v.trim().parse::<u64>().unwrap_or_else(|_| {
            eprintln!(
                "[server] MINFER_DRAIN_MS={v:?} is not a whole number of milliseconds; \
                 using {DEFAULT}"
            );
            DEFAULT
        }),
    }
}

/// Wait for `done` (the graceful server future) up to `deadline`, then report
/// what was still in flight.
///
/// This is the **bounded** half of the drain: the alternative — awaiting the
/// serve future unboundedly, then joining the worker — hangs for as long as the
/// slowest client keeps its connection (an SSE stream can be forever), because
/// that client's handler holds an `Arc<AppState>` and therefore keeps the job
/// channel's sender side open. Split out from [`serve_with_shutdown`] so the
/// bound itself is a unit test with no socket and no model.
pub async fn bounded_drain<F>(done: F, metrics: &ServerMetrics, deadline: Duration) -> DrainOutcome
where
    F: std::future::Future<Output = ()>,
{
    match tokio::time::timeout(deadline, done).await {
        Ok(()) => DrainOutcome::Clean,
        Err(_) => DrainOutcome::Forced {
            in_flight: metrics.in_flight.load(Ordering::SeqCst),
        },
    }
}

/// Serve `app` on `listener` until a shutdown signal, then drain within
/// `deadline`. Without a signal this never returns — the pre-F8 behaviour.
pub async fn serve_with_shutdown(
    listener: tokio::net::TcpListener,
    app: Router,
    metrics: Arc<ServerMetrics>,
    deadline: Duration,
) -> DrainOutcome {
    let (shutdown_tx, shutdown_rx) = tokio::sync::watch::channel(false);
    let signal_metrics = metrics.clone();
    tokio::spawn(async move {
        let sig = shutdown_signal().await;
        // Set `draining` *before* the watch fires, so a request that races the
        // signal sees the refusal rather than queueing behind the shutdown.
        signal_metrics.draining.store(true, Ordering::SeqCst);
        eprintln!(
            "[server] {sig}: draining — refusing new requests; {} request(s) in flight, \
             up to {} ms to finish",
            signal_metrics.in_flight.load(Ordering::SeqCst),
            deadline.as_millis()
        );
        let _ = shutdown_tx.send(true);
    });

    // axum's graceful shutdown stops accepting connections and runs the in-flight
    // ones to completion; the watch makes it start on our signal.
    let mut serve_rx = shutdown_rx.clone();
    let server = axum::serve(listener, app).with_graceful_shutdown(async move {
        while !*serve_rx.borrow() {
            if serve_rx.changed().await.is_err() {
                break;
            }
        }
    });
    let serve_task = tokio::spawn(async move {
        if let Err(e) = server.await {
            eprintln!("[server] serve error: {e}");
        }
    });

    // Phase 1 — wait for the signal. Without one this is the old `serve(...).await`:
    // the loop never ends.
    let mut wait_rx = shutdown_rx.clone();
    while !*wait_rx.borrow() {
        if wait_rx.changed().await.is_err() {
            std::future::pending::<()>().await;
        }
    }

    // Phase 2 — bounded drain.
    let outcome = bounded_drain(
        async move {
            let _ = serve_task.await;
        },
        &metrics,
        deadline,
    )
    .await;
    match outcome {
        DrainOutcome::Clean => {
            eprintln!("[server] drain complete: every in-flight request finished")
        }
        DrainOutcome::Forced { in_flight } => {
            metrics.drain_abandoned.store(in_flight, Ordering::SeqCst);
            eprintln!(
                "[server] drain deadline ({} ms) reached with {in_flight} request(s) still in \
                 flight; abandoning them and exiting",
                deadline.as_millis()
            );
        }
    }
    outcome
}

/// Wait for SIGINT or SIGTERM, naming which one arrived.
#[cfg(unix)]
async fn shutdown_signal() -> &'static str {
    use tokio::signal::unix::{signal, SignalKind};
    let mut term = match signal(SignalKind::terminate()) {
        Ok(s) => s,
        Err(e) => {
            // Cannot listen for SIGTERM: keep SIGINT working rather than dying at
            // startup, and say why.
            eprintln!("[server] cannot install the SIGTERM handler ({e}); SIGINT only");
            let _ = tokio::signal::ctrl_c().await;
            return "SIGINT";
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => "SIGINT",
        _ = term.recv() => "SIGTERM",
    }
}

#[cfg(not(unix))]
async fn shutdown_signal() -> &'static str {
    let _ = tokio::signal::ctrl_c().await;
    "SIGINT"
}

/// C8b S2/S4 A/B gate: `MINFER_NO_KV_SHARE=1` (presence-checked, the campaign's
/// standing env-gate rule) forces C8a's **copy** where the device's attention
/// kernel could read another sequence's prefix in place.
///
/// It is what lets the real-model gate compare a shared run against a
/// *shape-matched* copied one, and it is the switch that measures the share
/// itself. Presence-checked on purpose: `MINFER_NO_KV_SHARE=0` would otherwise
/// read as "off" while looking like a declaration.
pub(crate) fn kv_share_disabled() -> bool {
    std::env::var("MINFER_NO_KV_SHARE").map_or(false, |v| v == "1")
}

pub(crate) fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Read `tokenizer.chat_template` from GGUF metadata (same lookup the CLI uses).
pub(crate) fn chat_template_from_gguf(data: &[u8]) -> Option<String> {
    let ctx = crate::gguf::GgufContext::init_from_data(data)?;
    ctx.kv
        .iter()
        .find(|kv| kv.key == "tokenizer.chat_template")
        .map(|kv| kv.get_val_str(0).to_string())
}

// === Handlers ===

async fn chat_completions(State(state): State<Arc<AppState>>, body: String) -> Response {
    // F8: a shutdown refuses new work immediately — the drain waits for what is
    // already accepted and nothing else, so a request that arrived after the
    // signal must not extend the deadline.
    if state.metrics.draining.load(Ordering::SeqCst) {
        state
            .metrics
            .requests_rejected_total
            .fetch_add(1, Ordering::SeqCst);
        return error_response(&ApiError::unavailable("server is draining"));
    }
    let req = match ChatCompletionRequest::parse(body.as_bytes()) {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };
    let stream = req.stream.unwrap_or(false);
    let mut params: SamplingParams = match req.resolve(rand::random::<u64>()) {
        Ok(p) => p,
        Err(e) => return error_response(&e),
    };
    // F3 (#48): refuse a nonsensical sampler configuration as a 400 before the
    // request occupies a worker slot.
    if let Err(e) = params.validate(state.tokenizer.vocab_size()) {
        return error_response(&e);
    }
    // F2 (#47): compile the grammar/schema now (the vocabulary is here), so an
    // unsupported construct is a 400 before the job is queued. Speculative
    // decoding samples several rows from one automaton state, so a grammar and
    // the server's draft engine are mutually exclusive.
    if params.grammar_source.is_some() && state.spec_enabled {
        return error_response(&ApiError::invalid_request(
            "a grammar/JSON schema cannot be combined with the server's speculative draft \
             engine (a verify round samples several rows from one automaton state)"
                .to_string(),
        ));
    }
    if let Err(e) = params.compile_grammar(&state.tokenizer, &state.special) {
        return error_response(&e);
    }
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix(); // per-request timestamp (OpenAI semantics)
    let model_name = req
        .model
        .clone()
        .unwrap_or_else(|| state.model_name.clone());

    // Render the chat template + tokenize (cheap, do it on the handler side so
    // context-overflow is rejected before the job occupies the worker queue).
    let bos = state.tokenizer.bos_text();
    let messages: Vec<(String, Option<String>)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt = match crate::template::render_messages_opt(
        state.chat_template.as_deref(),
        &messages,
        true,
        &bos,
    ) {
        Ok(p) => p,
        // F7 (#50): a template refusal is a client-visible error naming the
        // construct; the startup gate above means this is a backstop.
        Err(e) => return error_response(&ApiError::invalid_request(e.message())),
    };
    let input_ids = state.tokenizer.encode(&prompt);
    if input_ids.is_empty() {
        return error_response(&ApiError::invalid_request(
            "rendered prompt tokenizes to nothing",
        ));
    }
    // C7: the bound is the whole arena, not a slot's share. The batched engine
    // repartitions on admission (an idle slot's cells are reclaimable) and answers
    // with its own `exceed_context` when even that cannot serve the request; the
    // serial path (batching off) still holds only its slot's share and reports the
    // same error from `generate_seq`.
    let prompt_tokens = input_ids.len();
    if prompt_tokens > state.n_ctx {
        return error_response(&ApiError::exceed_context(format!(
            "prompt of {prompt_tokens} tokens exceeds the context of {}",
            state.n_ctx
        )));
    }

    let (tx, rx) = mpsc::channel::<StreamEvent>(64);
    let job = Job {
        input_ids,
        params,
        tx,
    };
    if state.job_tx.send(job).await.is_err() {
        state
            .metrics
            .requests_rejected_total
            .fetch_add(1, Ordering::SeqCst);
        return error_response(&ApiError::unavailable("server shutting down"));
    }
    state.metrics.requests_total.fetch_add(1, Ordering::SeqCst);
    state.metrics.in_flight.fetch_add(1, Ordering::SeqCst);
    // F8: `in_flight` falls when the response is finished — the body sent, the SSE
    // stream closed, or the client gone. That is the set a drain waits for.
    let guard = InFlight::new(state.metrics.clone());

    if stream {
        stream_response(
            &id,
            &model_name,
            created,
            rx,
            guard,
            state.metrics.clone(),
            prompt_tokens as u64,
        )
    } else {
        let _guard = guard;
        match collect_response(rx).await {
            Ok((text, reason, completion_tokens)) => {
                // F8: token accounting happens where the response is *produced*,
                // so it covers the batched and serial paths uniformly and needs
                // nothing plumbed through the engine.
                state
                    .metrics
                    .record_tokens(prompt_tokens as u64, completion_tokens as u64);
                let resp = types::build_response(
                    &id,
                    &model_name,
                    created,
                    text,
                    &reason,
                    prompt_tokens,
                    completion_tokens,
                );
                (StatusCode::OK, axum::Json(resp)).into_response()
            }
            Err(e) => error_response(&e),
        }
    }
}

/// F8: one accepted request, counted until its response is finished. Dropping
/// this is what makes `in_flight` fall, and therefore what a graceful drain
/// waits for — including the streaming path, where the guard is owned by the SSE
/// stream and drops when the client is done or disconnects.
struct InFlight(Arc<ServerMetrics>);

impl InFlight {
    fn new(metrics: Arc<ServerMetrics>) -> Self {
        Self(metrics)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        self.0.in_flight.fetch_sub(1, Ordering::SeqCst);
        self.0
            .requests_completed_total
            .fetch_add(1, Ordering::SeqCst);
    }
}

/// Drain a job's event stream to a single non-streaming response.
async fn collect_response(
    mut rx: mpsc::Receiver<StreamEvent>,
) -> Result<(String, String, usize), ApiError> {
    let mut text = String::new();
    let mut reason = String::from("stop");
    let mut completion_tokens = 0usize;
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Text(t) => text.push_str(&t),
            StreamEvent::Finish { reason: r, tokens } => {
                reason = r;
                completion_tokens = tokens;
            }
            StreamEvent::Err(e) => return Err(e),
        }
    }
    Ok((text, reason, completion_tokens))
}

/// SSE stream: role chunk -> content chunks -> finish chunk -> [DONE].
///
/// `guard` rides the stream so `in_flight` only falls when the stream is dropped
/// (client done, or disconnected mid-answer) — dropping it here instead would
/// report the request finished while its tokens are still being written.
fn stream_response(
    id: &str,
    model: &str,
    created: i64,
    rx: mpsc::Receiver<StreamEvent>,
    guard: InFlight,
    tokens: Arc<ServerMetrics>,
    prompt_tokens: u64,
) -> Response {
    let role = ok(Event::default().data(types::chunk_role(id, model, created)));
    let done = ok(Event::default().data("[DONE]"));
    let id_owned = id.to_string();
    let model_owned = model.to_string();
    let stream = futures_util::stream::iter(vec![role])
        .chain(
            tokio_stream::wrappers::ReceiverStream::new(rx).map(move |ev| {
                let _keep = &guard;
                // F8: the worker sends `Finish` exactly once per request, so this
                // is the streaming path's token-accounting point. A client that
                // disconnects first drops the stream and its tokens are not
                // counted — they were never delivered.
                if let StreamEvent::Finish { tokens: n, .. } = &ev {
                    tokens.record_tokens(prompt_tokens, *n as u64);
                }
                to_event(&id_owned, &model_owned, created, ev)
            }),
        )
        .chain(futures_util::stream::iter(vec![done]));
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

fn to_event(id: &str, model: &str, created: i64, ev: StreamEvent) -> Result<Event, Infallible> {
    let data = match ev {
        StreamEvent::Text(t) => types::chunk_content(id, model, created, &t),
        StreamEvent::Finish { reason, .. } => types::chunk_finish(id, model, created, &reason),
        StreamEvent::Err(e) => e.json(),
    };
    Ok(Event::default().data(data))
}

fn ok(e: Event) -> Result<Event, Infallible> {
    Ok(e)
}

async fn list_models(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    axum::Json(json!({
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": state.created,
            "owned_by": "minfer",
        }],
    }))
}

async fn health() -> impl IntoResponse {
    axum::Json(json!({ "status": "ok" }))
}

fn error_response(e: &ApiError) -> Response {
    let status = StatusCode::from_u16(e.status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR);
    let value: serde_json::Value = serde_json::from_str(&e.json()).unwrap_or(json!({
        "error": { "message": e.message, "type": e.error_type, "code": e.status }
    }));
    (status, axum::Json(value)).into_response()
}

#[cfg(test)]
mod tests {
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
}
