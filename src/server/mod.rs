//! OpenAI-compatible HTTP server (OPENAI-CHAT-API-PLAN.md).
//!
//! axum + tokio; a dedicated std worker thread runs inference serially and
//! pushes per-request events into per-request `tokio::sync::mpsc` channels,
//! which the async handlers drain (backpressure = client consumption speed).

pub mod chat;
pub mod slot;
pub mod types;

use std::convert::Infallible;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

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
use types::{ApiError, ChatCompletionRequest, SamplingParams};

/// Shared server state (handlers only; the worker owns the model + slots).
pub struct AppState {
    pub job_tx: mpsc::Sender<Job>,
    pub model_name: String,
    pub n_ctx_slot: usize,
    pub tokenizer: Arc<Tokenizer>,
    pub chat_template: Option<String>,
    pub created: i64,
}

/// Build the axum router.
pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(list_models))
        .route("/health", get(health))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state)
}

/// Start the server: spawn the inference worker, then serve forever.
/// `model` and `slots` move into the worker thread; handlers only share the
/// job channel + tokenizer.
pub fn run(
    model: Box<dyn ModelDef>,
    tokenizer: Tokenizer,
    gguf: &GgufModel,
    port: u16,
    n_ctx: usize,
    n_slots: usize,
) {
    let (job_tx, job_rx) = mpsc::channel::<Job>(64);
    let slots = slot::new_slots(n_slots, n_ctx);
    let n_ctx_slot = n_ctx / n_slots.max(1);
    let worker_tokenizer = tokenizer.clone();
    let worker = std::thread::spawn(move || {
        chat::worker_loop(model, worker_tokenizer, slots, job_rx)
    });
    let _ = worker;

    let template = chat_template_from_gguf(&gguf.parts[0].data);
    let model_name = gguf
        .parts
        .first()
        .and_then(|p| p.ctx.get_key_val_str("general.name"))
        .unwrap_or_else(|| "minfer-model".to_string());

    let state = Arc::new(AppState {
        job_tx,
        model_name,
        n_ctx_slot,
        tokenizer: Arc::new(tokenizer),
        chat_template: template,
        created: now_unix(),
    });

    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    runtime.block_on(async move {
        let addr = std::net::SocketAddr::from(([0, 0, 0, 0], port));
        let listener = tokio::net::TcpListener::bind(addr).await.unwrap_or_else(|e| {
            eprintln!("Error: failed to bind {addr}: {e}");
            std::process::exit(1);
        });
        eprintln!("minfer server listening on http://{addr}");
        axum::serve(listener, router(state)).await.expect("server");
    });
}

fn now_unix() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

/// Read `tokenizer.chat_template` from GGUF metadata (same lookup the CLI uses).
fn chat_template_from_gguf(data: &[u8]) -> Option<String> {
    let ctx = crate::gguf::GgufContext::init_from_data(data)?;
    ctx.kv.iter().find(|kv| kv.key == "tokenizer.chat_template")
        .map(|kv| kv.get_val_str(0).to_string())
}

// === Handlers ===

async fn chat_completions(State(state): State<Arc<AppState>>, body: String) -> Response {
    let req = match ChatCompletionRequest::parse(body.as_bytes()) {
        Ok(r) => r,
        Err(e) => return error_response(&e),
    };
    let stream = req.stream.unwrap_or(false);
    let params: SamplingParams = req.resolve(rand::random::<u64>());
    let id = format!("chatcmpl-{}", uuid::Uuid::new_v4().simple());
    let created = now_unix(); // per-request timestamp (OpenAI semantics)
    let model_name = req.model.clone().unwrap_or_else(|| state.model_name.clone());

    // Render the chat template + tokenize (cheap, do it on the handler side so
    // context-overflow is rejected before the job occupies the worker queue).
    let bos = state.tokenizer.bos_text();
    let messages: Vec<(String, Option<String>)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();
    let prompt = crate::template::render_messages(
        state.chat_template.as_deref().unwrap_or(""),
        &messages,
        true,
        &bos,
    );
    let input_ids = state.tokenizer.encode(&prompt);
    if input_ids.is_empty() {
        return error_response(&ApiError::invalid_request(
            "rendered prompt tokenizes to nothing",
        ));
    }
    let prompt_tokens = input_ids.len();
    if prompt_tokens > state.n_ctx_slot {
        return error_response(&ApiError::exceed_context(format!(
            "prompt of {prompt_tokens} tokens exceeds slot context of {}",
            state.n_ctx_slot
        )));
    }

    let (tx, rx) = mpsc::channel::<StreamEvent>(64);
    let job = Job { input_ids, params, tx };
    if state.job_tx.send(job).await.is_err() {
        return error_response(&ApiError::unavailable("server shutting down"));
    }

    if stream {
        stream_response(&id, &model_name, created, rx)
    } else {
        match collect_response(rx).await {
            Ok((text, reason, completion_tokens)) => {
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
fn stream_response(
    id: &str,
    model: &str,
    created: i64,
    rx: mpsc::Receiver<StreamEvent>,
) -> Response {
    let role = ok(Event::default().data(types::chunk_role(id, model, created)));
    let done = ok(Event::default().data("[DONE]"));
    let id_owned = id.to_string();
    let model_owned = model.to_string();
    let stream = futures_util::stream::iter(vec![role])
        .chain(
            tokio_stream::wrappers::ReceiverStream::new(rx)
                .map(move |ev| to_event(&id_owned, &model_owned, created, ev)),
        )
        .chain(futures_util::stream::iter(vec![done]));
    Sse::new(stream).keep_alive(KeepAlive::default()).into_response()
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
