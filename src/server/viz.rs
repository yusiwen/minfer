//! Self-contained visualization server — `minfer --viz [port]`.
//!
//! One process serves everything the web visualizer needs:
//! - the viz/ page itself (embedded at build time; samples/ optionally from
//!   disk via `MINFER_VIZ_DIR`, default `viz`)
//! - `GET /viz/graph` — prefill + decode graphs (same shape as a trace doc)
//! - `GET /viz/events` — SSE live inference events
//! - `POST /viz/run` — `{prompt, max_tokens?, temperature?}` trigger (chat
//!   template rendered server-side, same slot/worker machinery as `--server`)
//!
//! Live capture is lazy: `live::enabled()` is true only while an SSE client is
//! connected, so idle viz servers pay nothing.

use std::convert::Infallible;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use axum::extract::{Path, State};
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
    #[cfg(target_os = "macos")]
    let gpu = crate::graph::metal_backend::metal_available()
        && !std::env::var("MINFER_DISABLE_MPS").map_or(false, |v| v == "1");
    #[cfg(not(target_os = "macos"))]
    let gpu = false;
    let fuse_qkv = gpu && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1");
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
              "graph": crate::graph::json::export_graph_json(&*model, &model_name, 16, n_ctx_slot, gpu, false),
              "steps": [] },
            { "kind": "decode",
              "graph": crate::graph::json::export_graph_json(&*model, &model_name, 1, n_ctx_slot, gpu, fuse_qkv),
              "steps": [] },
        ],
    }));

    let worker_tokenizer = tokenizer.clone();
    std::thread::spawn(move || super::chat::worker_loop(model, worker_tokenizer, slots, job_rx));

    let template = super::chat_template_from_gguf(&gguf.parts[0].data);
    let app = Arc::new(AppState {
        job_tx,
        model_name: gguf
            .parts
            .first()
            .and_then(|p| p.ctx.get_key_val_str("general.name"))
            .unwrap_or_else(|| "minfer-model".to_string()),
        n_ctx_slot,
        tokenizer: Arc::new(tokenizer),
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
    ([(axum::http::header::CONTENT_TYPE, "text/css")], STYLE_CSS).into_response()
}
async fn app_asset() -> Response {
    (
        [(axum::http::header::CONTENT_TYPE, "application/javascript")],
        APP_JS,
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
        [(axum::http::header::CONTENT_TYPE, "text/html; charset=utf-8")],
        body,
    )
        .into_response()
}

/// Serve samples/ from disk (`MINFER_VIZ_DIR` or `viz` relative to CWD).
async fn sample_asset(Path(name): Path<String>) -> Response {
    if name.contains('/') || name.contains("..") {
        return (StatusCode::BAD_REQUEST, "bad path").into_response();
    }
    let dir = std::env::var("MINFER_VIZ_DIR").unwrap_or_else(|_| "viz".to_string());
    let path = std::path::Path::new(&dir).join("samples").join(&name);
    match std::fs::read(&path) {
        Ok(bytes) => (
            [(axum::http::header::CONTENT_TYPE, "application/json")],
            bytes,
        )
            .into_response(),
        Err(_) => (StatusCode::NOT_FOUND, "sample not found").into_response(),
    }
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
    let prompt = crate::template::render_messages(
        state.app.chat_template.as_deref().unwrap_or(""),
        &[("user".to_string(), Some(req.prompt.clone()))],
        true,
        &bos,
    );
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
