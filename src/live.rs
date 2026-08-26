//! P3 live event broadcast — the web visualizer watches inference in real
//! time over SSE (`/viz/events`, served by `minfer --viz`).
//!
//! The scheduler emits one event per executed node (reusing the P2 read-back);
//! the generation loop emits phase / step / token / logits events. Events are
//! pre-serialized JSON strings on a `tokio::sync::broadcast` channel (send is
//! sync; drops when no client is subscribed).
//!
//! **Lazy arming**: capture only happens while at least one SSE client is
//! connected (`clients > 0`). Normal inference (CLI single-shot, `--server`)
//! never arms the broadcaster, and a `--viz` server with nobody watching pays
//! nothing either — `enabled()` is a cheap uncontended lock + bool check,
//! evaluated once per `execute()`.
//!
//! Execution is single-threaded, so the global `Mutex<Live>` is uncontended.

use std::sync::{Mutex, OnceLock};
use tokio::sync::broadcast::{Receiver, Sender};

type Tx = Sender<String>;

struct Live {
    tx: Option<Tx>,
    clients: usize,
    phase: String,
    step: usize,
    pending_token: Option<u32>,
    pending_text: String,
}

static LIVE: OnceLock<Mutex<Live>> = OnceLock::new();

fn live() -> &'static Mutex<Live> {
    LIVE.get_or_init(|| {
        Mutex::new(Live {
            tx: None,
            clients: 0,
            phase: String::new(),
            step: 0,
            pending_token: None,
            pending_text: String::new(),
        })
    })
}

/// Server startup (`--viz`): arm the broadcaster.
pub fn init(tx: Tx) {
    live().lock().unwrap().tx = Some(tx);
}

/// Whether live capture is active: armed AND at least one SSE client is
/// watching. Checked once per `execute()`.
pub fn enabled() -> bool {
    let l = live().lock().unwrap();
    l.tx.is_some() && l.clients > 0
}

/// SSE handler: a client connected — bump the counter and return a receiver.
pub fn subscribe() -> Receiver<String> {
    let mut l = live().lock().unwrap();
    l.clients += 1;
    l.tx.clone().expect("live not armed").subscribe()
}

/// SSE handler: a client disconnected (stream dropped).
pub fn detach_client() {
    let mut l = live().lock().unwrap();
    l.clients = l.clients.saturating_sub(1);
}

fn emit(l: &Live, ev: &serde_json::Value) {
    if let Some(tx) = &l.tx {
        if let Ok(s) = serde_json::to_string(ev) {
            let _ = tx.send(s);
        }
    }
}

/// CLI/server: mark a phase boundary (prefill / decode). Emits a `phase` event.
pub fn begin_phase(kind: &str) {
    let mut l = live().lock().unwrap();
    l.phase = kind.into();
    l.step = 0;
    emit(
        &l,
        &serde_json::json!({ "type": "phase", "kind": kind, "index": 0 }),
    );
}

/// Scheduler: one `execute()` = one step. Emits nothing (node events carry the
/// step index; the `step` event arrives at attach_step).
pub fn begin_step() {
    let mut l = live().lock().unwrap();
    l.step += 1;
}

/// Scheduler: one node's output. `values` is the strided sample (P2 analyze).
pub fn record_node(
    id: usize,
    name: &str,
    op: &str,
    dtype: &str,
    stats: [f64; 4],
    values: &[f32],
    stride: usize,
    n_total: usize,
) {
    let l = live().lock().unwrap();
    let step = l.step;
    emit(
        &l,
        &serde_json::json!({
            "type": "node",
            "step": step,
            "id": id,
            "name": name,
            "op": op,
            "dtype": dtype,
            "stats": { "min": stats[0], "max": stats[1], "mean": stats[2], "absmean": stats[3] },
            "values": values,
            "stride": stride,
            "n": n_total,
        }),
    );
}

/// Server (decode): remember the token about to become the next step's input.
pub fn set_token(id: u32, text: &str) {
    let mut l = live().lock().unwrap();
    l.pending_token = Some(id);
    l.pending_text = text.to_string();
}

/// Server: emit the step boundary event (token + this step's logits top-5).
/// Called after each forward returns.
pub fn attach_step(logits: &[f32]) {
    let mut l = live().lock().unwrap();
    let step = l.step;
    let tok = l.pending_token;
    let txt = l.pending_text.clone();
    l.pending_token = None;
    l.pending_text.clear();
    let top = crate::trace::topk(logits, 5);
    emit(
        &l,
        &serde_json::json!({
            "type": "step",
            "phase": l.phase,
            "step": step,
            "token": tok,
            "text": txt,
            "logits_top": top,
        }),
    );
}

/// Server: generation finished.
pub fn finish(reason: &str, tokens: usize, text: &str) {
    let l = live().lock().unwrap();
    emit(
        &l,
        &serde_json::json!({
            "type": "finish",
            "reason": reason,
            "tokens": tokens,
            "text": text,
        }),
    );
}

/// SSE connect greeting.
pub fn hello(model: &str, prompt_tokens: usize) -> String {
    serde_json::json!({
        "type": "hello",
        "model": model,
        "prompt_tokens": prompt_tokens,
    })
    .to_string()
}
