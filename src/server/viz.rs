//! Self-contained visualization server — `minfer viz [--port N] <model>`.
//!
//! One process serves everything the web visualizer needs:
//! - the viz/ page itself (embedded at build time; samples/ optionally from
//!   disk via `MINFER_VIZ_DIR`, default `viz`)
//! - `GET /viz/graph` — prefill + decode graphs (same shape as a trace doc)
//! - `GET /viz/events` — SSE live inference events
//! - `POST /viz/run` — `{prompt, max_tokens?, temperature?}` trigger (chat
//!   template rendered server-side, same slot/worker machinery as `serve`)
//!
//! Live capture is lazy: `live::enabled()` is true only while an SSE client is
//! connected, so idle viz servers pay nothing.

use std::convert::Infallible;
use std::io::Read;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::{Path, State};
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::Router;
use futures_util::stream::Stream;
use futures_util::StreamExt;
use serde_json::json;
use tokio::sync::mpsc;

use crate::gguf::GgufModel;
use crate::models::ModelDef;
use crate::tokenizer::Tokenizer;

use super::chat::{Job, StreamEvent};
use super::types::SamplingParams;
use super::AppState;

/// Embedded viz/ assets (page + logic; samples are optional disk data).
const INDEX_HTML: &str = include_str!("../../viz/index.html");
const STYLE_CSS: &str = include_str!("../../viz/style.css");
const APP_JS: &str = include_str!("../../viz/app.js");
// The end-to-end shape-flow view (viz/e2e.html) and its data layer.
const E2E_HTML: &str = include_str!("../../viz/e2e.html");
const E2E_CSS: &str = include_str!("../../viz/e2e.css");
const E2E_MODEL_JS: &str = include_str!("../../viz/e2e-model.js");
const E2E_JS: &str = include_str!("../../viz/e2e.js");

pub struct VizState {
    pub app: Arc<AppState>,
    pub graphs: Arc<serde_json::Value>,
}

#[derive(serde::Deserialize)]
struct RunRequest {
    prompt: String,
    max_tokens: Option<i64>,
    temperature: Option<f32>,
}

/// Start the viz server: arm live, precompute graphs, spawn the inference
/// worker, then serve the page + live endpoints forever.
pub fn run_viz(
    model: Box<dyn ModelDef>,
    tokenizer: Tokenizer,
    gguf: &GgufModel,
    port: u16,
    n_ctx: usize,
    n_slots: usize,
) {
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let slots = super::slot::new_slots(n_slots, n_ctx);
    let n_ctx_slot = n_ctx / n_slots.max(1);

    // Arm the broadcaster + precompute graphs BEFORE the model moves into the
    // worker thread (canonical prefill nt=16: topology matches any run, shapes
    // approximate for the live prefill view).
    let (live_tx, _) = tokio::sync::broadcast::channel::<String>(8192);
    crate::live::init(live_tx.clone());
    // Mirror the engine's CParams construction (qwen2/qwen3 graph.rs): GPU
    // participation is Metal OR CUDA; QKV fusion stays Metal-only (the fused
    // bias+rope+store kernel has no CUDA counterpart — the engine's own gate
    // is `fuse_qkv = nt==1 && metal_on`); FFN gate+up fusion runs on both
    // (7e⑤). The preview builds through the same graph builder, so the live
    // per-node events (matched by node id) only line up when these flags do.
    #[cfg(target_os = "macos")]
    let metal_on = crate::graph::metal_backend::metal_available()
        && !std::env::var("MINFER_DISABLE_MPS").map_or(false, |v| v == "1");
    #[cfg(not(target_os = "macos"))]
    let metal_on = false;
    #[cfg(feature = "cuda")]
    let cuda_on = crate::cuda::CudaState::get().is_some();
    #[cfg(not(feature = "cuda"))]
    let cuda_on = false;
    let gpu = metal_on || cuda_on;
    let fuse_qkv = metal_on && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1");
    let fuse_ffn = gpu && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1");
    let model_name = gguf
        .parts
        .first()
        .and_then(|p| p.ctx.get_key_val_str("general.name"))
        .unwrap_or_else(|| "minfer-model".to_string());
    let graphs = Arc::new(json!({
        "format": "minfer-live",
        "version": 1,
        "model": model_name,
        "phases": [
            { "kind": "prefill",
              "graph": crate::graph::json::export_graph_json(&*model, &model_name, 16, n_ctx_slot, gpu, false, false),
              "steps": [] },
            { "kind": "decode",
              "graph": crate::graph::json::export_graph_json(&*model, &model_name, 1, n_ctx_slot, gpu, fuse_qkv, fuse_ffn),
              "steps": [] },
        ],
    }));

    // F7 (#50): refuse to start with an unrenderable template, before the worker
    // thread exists.
    let template = super::chat_template_from_gguf(&gguf.parts[0].data);
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

    let worker_tokenizer = tokenizer.clone();
    // F8: the viz server publishes into the same registry shape as `minfer serve`
    // (its own instance — the two commands never run in one process).
    let metrics = Arc::new(super::metrics::ServerMetrics::new());
    let worker_metrics = metrics.clone();
    // doc 97: the viz path wires no speculative draft.
    // F2: the handler needs the EOG set, and the model moves into the worker.
    let special = model.special_tokens();
    std::thread::spawn(move || {
        super::chat::worker_loop(
            model,
            worker_tokenizer,
            slots,
            job_rx,
            None,
            None,
            worker_metrics,
        )
    });

    let app = Arc::new(AppState {
        job_tx,
        metrics,
        model_name: gguf
            .parts
            .first()
            .and_then(|p| p.ctx.get_key_val_str("general.name"))
            .unwrap_or_else(|| "minfer-model".to_string()),
        n_ctx,
        n_ctx_slot,
        tokenizer: Arc::new(tokenizer),
        special,
        spec_enabled: false,
        chat_template: template,
        created: super::now_unix(),
    });
    let state = Arc::new(VizState { app, graphs });

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async move {
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr)
            .await
            .unwrap_or_else(|e| {
                eprintln!("Error: failed to bind {addr}: {e}");
                std::process::exit(1);
            });
        eprintln!("minfer viz server on http://{addr}  (open in a browser to interact)");
        axum::serve(listener, viz_router(state))
            .await
            .expect("server");
    });
}

fn viz_router(state: Arc<VizState>) -> Router {
    Router::new()
        .route("/", get(page))
        .route("/index.html", get(page))
        .route("/favicon.ico", get(favicon))
        .route("/style.css", get(style_asset))
        .route("/app.js", get(app_asset))
        .route("/e2e.html", get(e2e_page))
        .route("/e2e.css", get(e2e_style_asset))
        .route("/e2e-model.js", get(e2e_model_asset))
        .route("/e2e.js", get(e2e_app_asset))
        .route("/samples/:name", get(sample_asset))
        .route("/viz/graph", get(viz_graph))
        .route("/viz/events", get(viz_events))
        .route("/viz/run", post(viz_run))
        .with_state(state)
}

// === static assets ===

async fn page() -> Response {
    html(INDEX_HTML.to_string())
}
async fn style_asset() -> Response {
    asset("text/css", STYLE_CSS)
}
async fn app_asset() -> Response {
    asset("application/javascript", APP_JS)
}
async fn e2e_page() -> Response {
    html(E2E_HTML.to_string())
}
async fn e2e_style_asset() -> Response {
    asset("text/css", E2E_CSS)
}
async fn e2e_model_asset() -> Response {
    asset("application/javascript", E2E_MODEL_JS)
}
async fn e2e_app_asset() -> Response {
    asset("application/javascript", E2E_JS)
}

/// Serve one embedded asset. These are a development surface that is edited in
/// place, so they are `no-store`: a cached script beside a fresh page is how the
/// "Shape flow" jump silently stopped working (the listener lived in the stale
/// script). Samples stay cacheable -- they are data, not code.
fn asset(content_type: &'static str, body: &'static str) -> Response {
    (
        [
            (axum::http::header::CONTENT_TYPE, content_type),
            (axum::http::header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}
/// Tiny inline SVG favicon (kills the browser's /favicon.ico 404 noise).
async fn favicon() -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, "image/svg+xml")],
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect width='16' height='16' rx='3' fill='%230ea5e9'/><path d='M4 11l3-6 2 3 1.5-2 1.5 5z' fill='white'/></svg>",
    )
        .into_response()
}

fn html(body: String) -> Response {
    (
        [
            (axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (axum::http::header::CACHE_CONTROL, "no-store"),
        ],
        body,
    )
        .into_response()
}

/// Serve samples/ from disk (`MINFER_VIZ_DIR` or `viz` relative to CWD).
async fn sample_asset(Path(name): Path<String>, headers: HeaderMap) -> Response {
    let dir = std::env::var("MINFER_VIZ_DIR").unwrap_or_else(|_| "viz".to_string());
    let if_none_match = headers
        .get(axum::http::header::IF_NONE_MATCH)
        .and_then(|value| value.to_str().ok());
    sample_response(std::path::Path::new(&dir), &name, if_none_match)
}

/// A strong validator for one sample: its length and modification time.
///
/// `If-None-Match` needs no HTTP-date formatting, so this is a few lines of `fstat`
/// rather than a date library — and a browser that holds the tag revalidates with it
/// in preference to any `If-Modified-Since` it was given (RFC 9110 §13.1.3). A
/// platform that cannot report a modification time falls back to the length alone.
fn etag_of(metadata: &std::fs::Metadata) -> String {
    let mtime = metadata
        .modified()
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("\"{}-{}\"", metadata.len(), mtime)
}

/// Whether an `If-None-Match` value names `etag`, by *weak* comparison over the
/// comma-separated list a browser, a proxy or `curl` may send (RFC 9110 §13.1.2).
/// `*` matches any current representation; a missing or unparsable value matches
/// nothing, so a garbage header is a `200`, never a `304`.
fn etag_matches(if_none_match: Option<&str>, etag: &str) -> bool {
    let Some(value) = if_none_match else {
        return false;
    };
    let wanted = etag.strip_prefix("W/").unwrap_or(etag);
    value.split(',').any(|candidate| {
        let candidate = candidate.trim();
        candidate == "*" || candidate.strip_prefix("W/").unwrap_or(candidate) == wanted
    })
}

/// The body of [`sample_asset`], with the directory and the client's validator passed
/// in, so the JSON contract and the `304` arm are testable without the process-wide
/// `MINFER_VIZ_DIR` (#546, #548).
///
/// Samples are data, not code: one may be regenerated in place under the same name,
/// and a browser holding a copy has no way to know. `no-cache` says *revalidate* —
/// a client that asks gets the file on disk — where `no-store` (the code assets'
/// rule, [`asset`]) would forbid keeping the copy at all. The `ETag` is what makes
/// that revalidation cheap: a matching `If-None-Match` is answered from the file's
/// metadata, without reading the body. The `viz/` pages ask for the revalidation
/// themselves (`cache: 'no-cache'` on every runtime fetch), which is what covers a
/// plain static server too — that server is not ours and sends no `Cache-Control`.
fn sample_response(dir: &std::path::Path, name: &str, if_none_match: Option<&str>) -> Response {
    if name.contains('/') || name.contains("..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let mut file = match std::fs::File::open(dir.join("samples").join(name)) {
        Ok(file) => file,
        Err(_) => return (StatusCode::NOT_FOUND, "sample not found").into_response(),
    };
    // `fstat` on the handle that would be read, so the tag and the bytes describe
    // one file rather than two lookups of it.
    let metadata = match file.metadata() {
        Ok(metadata) => metadata,
        Err(_) => return (StatusCode::NOT_FOUND, "sample not found").into_response(),
    };
    let etag = etag_of(&metadata);
    let headers = [
        (
            axum::http::header::CONTENT_TYPE,
            "application/json".to_string(),
        ),
        (axum::http::header::CACHE_CONTROL, "no-cache".to_string()),
        (axum::http::header::ETAG, etag.clone()),
    ];
    if etag_matches(if_none_match, &etag) {
        // RFC 9110 §15.4.5: a `304` repeats the validators its `200` would have sent.
        return (StatusCode::NOT_MODIFIED, headers, Vec::new()).into_response();
    }
    let mut bytes = Vec::new();
    if file.read_to_end(&mut bytes).is_err() {
        return (StatusCode::NOT_FOUND, "sample not found").into_response();
    }
    (headers, bytes).into_response()
}

// === live endpoints ===

async fn viz_graph(State(state): State<Arc<VizState>>) -> impl IntoResponse {
    axum::Json((*state.graphs).clone())
}

/// Unsubscribe when the client disconnects (the stream is dropped), so capture stops automatically.
struct LiveGuard;
impl Drop for LiveGuard {
    fn drop(&mut self) {
        crate::live::detach_client();
    }
}
struct Guarded<S>(S, LiveGuard);
impl<S: Stream + Unpin> Stream for Guarded<S> {
    type Item = S::Item;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<S::Item>> {
        self.get_mut().0.poll_next_unpin(cx)
    }
}

async fn viz_events(State(state): State<Arc<VizState>>) -> Response {
    let rx = crate::live::subscribe(); // clients += 1: begin capture
    let hello = Event::default().data(crate::live::hello(&state.app.model_name, 0));
    let stream = Guarded(
        futures_util::stream::iter(vec![Ok::<_, Infallible>(hello)]).chain(
            tokio_stream::wrappers::BroadcastStream::new(rx).map(|r| match r {
                Ok(line) => Ok(Event::default().data(line)),
                Err(_lagged) => Ok(Event::default().data("{\"type\":\"lag\"}")),
            }),
        ),
        LiveGuard,
    );
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// POST /viz/run {prompt, max_tokens?, temperature?} — dedicated inference trigger endpoint.
async fn viz_run(State(state): State<Arc<VizState>>, body: String) -> Response {
    let req: RunRequest = match serde_json::from_str(&body) {
        Ok(r) => r,
        Err(_) => return json_err("invalid body: {prompt, max_tokens?, temperature?}"),
    };
    if req.prompt.trim().is_empty() {
        return json_err("prompt required");
    }
    let bos = state.app.tokenizer.bos_text();
    let prompt = match crate::template::render_messages_opt(
        state.app.chat_template.as_deref(),
        &[("user".to_string(), Some(req.prompt.clone()))],
        true,
        &bos,
    ) {
        Ok(p) => p,
        Err(e) => return json_err(e.message()),
    };
    let input_ids = state.app.tokenizer.encode(&prompt);
    if input_ids.is_empty() {
        return json_err("prompt tokenizes to nothing");
    }
    if input_ids.len() > state.app.n_ctx_slot {
        return json_err(format!(
            "prompt of {} tokens exceeds slot context of {}",
            input_ids.len(),
            state.app.n_ctx_slot
        ));
    }
    let params = SamplingParams {
        temp: req.temperature.unwrap_or(super::types::DEFAULT_TEMP),
        top_k: super::types::DEFAULT_TOP_K,
        top_p: super::types::DEFAULT_TOP_P,
        repeat_penalty: super::types::DEFAULT_REPEAT_PENALTY,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        // F3 (#48): the viz request surface carries only temperature and
        // max_tokens, so the whole new sampler set stays at its no-op default.
        min_p: 0.0,
        typical_p: 1.0,
        xtc_probability: 0.0,
        xtc_threshold: 0.5,
        dry_multiplier: 0.0,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_penalty_last_n: 64,
        dry_breakers: Vec::new(),
        mirostat: crate::sampler::MirostatMode::Off,
        mirostat_tau: 5.0,
        mirostat_eta: 0.1,
        mirostat_m: 100,
        logit_bias: Vec::new(),
        // F2 (#47): the viz request surface has no grammar field, so it stays
        // unconstrained (bit-identical to the pre-F2 path).
        grammar_source: None,
        grammar: None,
        seed: rand::random::<u64>(),
        stop_strings: Vec::new(),
        max_tokens: req.max_tokens.unwrap_or(32),
    };
    let (tx, mut rx) = mpsc::channel::<StreamEvent>(64);
    if state
        .app
        .job_tx
        .send(Job {
            input_ids,
            params,
            tx,
        })
        .await
        .is_err()
    {
        return json_err("server shutting down");
    }
    // Events drive the page over SSE; only the response body is summarized here (the page also shows the text).
    let mut text = String::new();
    let mut reason = String::from("stop");
    let mut tokens = 0usize;
    while let Some(ev) = rx.recv().await {
        match ev {
            StreamEvent::Text(t) => text.push_str(&t),
            StreamEvent::Finish {
                reason: r,
                tokens: n,
            } => {
                reason = r;
                tokens = n;
            }
            StreamEvent::Err(e) => return json_err(&e.message),
        }
    }
    axum::Json(json!({
        "ok": true,
        "text": text,
        "finish_reason": reason,
        "completion_tokens": tokens,
    }))
    .into_response()
}

fn json_err(msg: impl Into<String>) -> Response {
    (
        StatusCode::BAD_REQUEST,
        axum::Json(json!({ "ok": false, "error": msg.into() })),
    )
        .into_response()
}

#[cfg(test)]
mod tests;
