//! Per-node inference trace for the web visualizer (P2) — `MINFER_TRACE=<path>`.
//!
//! During graph execution the scheduler records every node's output: full-buffer
//! stats (min/max/mean/abs-mean) + a strided downsampled value sample (for the
//! page's heatmaps). The CLI marks phases (`prefill` / `decode`), attaches the
//! sampled token + logits top-k to each decode step, and `finish()` writes one
//! `minfer-trace` JSON (graph structure included, one graph per phase).
//!
//! Execution is single-threaded, so a global `Mutex<Trace>` is uncontended —
//! this avoids plumbing a collector through the scheduler/allocator signatures.
//! Recording is best-effort: no entry point other than the CLI single-shot path
//! calls `finish()`, so an un-phased trace simply never gets written.

use std::sync::{Mutex, OnceLock};
use serde_json::{json, Value};

/// Downsampled value sample cap per node (page heatmap grid ~64 cells).
pub const MAX_SAMPLE: usize = 64;

#[derive(Default)]
pub struct NodeData {
    pub dtype: &'static str, // "f32" | "i32"
    pub stats: [f64; 4],     // min, max, mean, abs-mean (display values)
    pub values: Vec<f32>,    // strided sample
    pub stride: usize,
    pub n_total: usize,
}

#[derive(Default)]
pub struct Step {
    pub nodes: Vec<(usize, NodeData)>,
    /// Input token of this step (decode); None for prefill.
    pub token: Option<u32>,
    pub token_text: String,
    /// (token_id, probability) — top-k softmax of THIS step's logits output.
    pub logits_top: Vec<(u32, f32)>,
}

#[derive(Default)]
pub struct Phase {
    pub kind: String, // "prefill" | "decode"
    pub steps: Vec<Step>,
    /// Attached by `finish()` (the exported graph for this phase).
    pub graph: Option<Value>,
}

#[derive(Default)]
pub struct Trace {
    pub phases: Vec<Phase>,
    pending_token: Option<u32>,
    pending_text: String,
}

static TRACE: OnceLock<Mutex<Trace>> = OnceLock::new();

fn trace() -> &'static Mutex<Trace> {
    TRACE.get_or_init(Default::default)
}

pub fn enabled() -> bool {
    std::env::var_os("MINFER_TRACE").is_some()
}

pub fn path() -> Option<String> {
    std::env::var("MINFER_TRACE").ok()
}

/// CLI: mark the start of a phase (prefill / decode). Repeated calls within the
/// same phase are no-ops; a kind change starts a new phase.
pub fn begin_phase(kind: &str) {
    let mut t = trace().lock().unwrap();
    match t.phases.last() {
        Some(p) if p.kind == kind => {}
        _ => t.phases.push(Phase { kind: kind.into(), steps: Vec::new(), graph: None }),
    }
}

/// Scheduler: start a new step (one per `execute()` call — prefill is one
/// step, each decode forward is one step). No-op outside a CLI phase.
pub fn begin_step() {
    let mut t = trace().lock().unwrap();
    let Some(phase) = t.phases.last_mut() else { return };
    phase.steps.push(Step::default());
}

/// Scheduler: record one node's output into the current step. No-op when no
/// phase/step has been opened.
pub fn record_node(
    id: usize,
    dtype: &'static str,
    stats: [f64; 4],
    values: Vec<f32>,
    stride: usize,
    n_total: usize,
) {
    let mut t = trace().lock().unwrap();
    let Some(phase) = t.phases.last_mut() else { return };
    let Some(step) = phase.steps.last_mut() else { return };
    step.nodes.push((
        id,
        NodeData { dtype, stats, values, stride, n_total },
    ));
}

/// CLI (decode): remember the token about to become the next step's input.
pub fn set_token(id: u32, text: &str) {
    let mut t = trace().lock().unwrap();
    t.pending_token = Some(id);
    t.pending_text = text.to_string();
}

/// CLI: attach token + logits top-k to the step the scheduler just recorded.
/// `token` None ⇒ prefill step (no input token; logits still shown).
pub fn attach_step(logits: &[f32]) {
    let mut t = trace().lock().unwrap();
    let tok = t.pending_token;
    let txt = t.pending_text.clone();
    t.pending_token = None;
    t.pending_text.clear();
    let Some(phase) = t.phases.last_mut() else { return };
    let Some(step) = phase.steps.last_mut() else { return };
    step.token = tok;
    step.token_text = txt;
    step.logits_top = topk(logits, 5);
}

/// Softmax over all logits, then the top-k (id, probability).
pub fn topk(logits: &[f32], k: usize) -> Vec<(u32, f32)> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut exps = vec![0f64; logits.len()];
    let mut sum = 0f64;
    for (i, l) in logits.iter().enumerate() {
        let e = ((l - max) as f64).exp();
        exps[i] = e;
        sum += e;
    }
    let mut idx: Vec<u32> = (0..logits.len() as u32).collect();
    idx.sort_by(|&a, &b| exps[b as usize].partial_cmp(&exps[a as usize]).unwrap());
    idx.truncate(k);
    idx.into_iter()
        .map(|i| (i, (exps[i as usize] / sum) as f32))
        .collect()
}

/// Full-buffer stats + strided sample. i32 inputs (token ids) are stored as
/// f32 bit patterns — reinterpret to integers for display values/stats.
/// Shared with the P3 live broadcaster.
pub(crate) fn analyze(dtype: &str, data: &[f32]) -> ([f64; 4], Vec<f32>, usize, usize) {
    let n = data.len();
    let stride = if n <= MAX_SAMPLE { 1 } else { (n + MAX_SAMPLE - 1) / MAX_SAMPLE };
    let mut values = Vec::with_capacity(n / stride + 1);
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut sum = 0f64;
    let mut asum = 0f64;
    let is_i32 = dtype == "i32";
    for (i, &v) in data.iter().enumerate() {
        let x: f64 = if is_i32 {
            (v.to_bits() as i32) as f64
        } else {
            v as f64
        };
        if i % stride == 0 {
            // Round sample values (heatmap preview; keeps the trace compact).
            values.push(((x as f32) * 1e6).round() / 1e6);
        }
        min = min.min(x);
        max = max.max(x);
        sum += x;
        asum += x.abs();
    }
    let nf = n.max(1) as f64;
    ([if n > 0 { min } else { 0.0 }, if n > 0 { max } else { 0.0 }, sum / nf, asum / nf], values, stride, n)
}

/// Serialize the whole trace to `<path>` (compact JSON — traces are large).
pub fn finish(path: &str, model: &str, prompt: &str, graph_prefill: Value, graph_decode: Value) {
    let mut t = trace().lock().unwrap();
    for p in &mut t.phases {
        p.graph = match p.kind.as_str() {
            "prefill" => Some(graph_prefill.clone()),
            "decode" => Some(graph_decode.clone()),
            _ => None,
        };
    }
    let phases: Vec<Value> = t
        .phases
        .iter()
        .map(|p| {
            json!({
                "kind": p.kind,
                "graph": p.graph.clone(),
                "steps": p.steps.iter().map(|s| {
                    json!({
                        "token": s.token,
                        "text": s.token_text,
                        "logits_top": s.logits_top,
                        "nodes": s.nodes.iter().map(|(id, nd)| {
                            json!({
                                "id": id,
                                "dtype": nd.dtype,
                                "stats": { "min": nd.stats[0], "max": nd.stats[1], "mean": nd.stats[2], "absmean": nd.stats[3] },
                                "values": nd.values,
                                "stride": nd.stride,
                                "n": nd.n_total,
                            })
                        }).collect::<Vec<_>>(),
                    })
                }).collect::<Vec<_>>(),
            })
        })
        .collect();
    let doc = json!({
        "format": "minfer-trace",
        "version": 1,
        "model": model,
        "prompt": prompt,
        "phases": phases,
    });
    match serde_json::to_string(&doc) {
        Ok(s) => {
            if let Err(e) = std::fs::write(path, s) {
                eprintln!("[trace] failed to write {path}: {e}");
            } else {
                eprintln!("[trace] wrote {path} ({} phases)", t.phases.len());
            }
        }
        Err(e) => eprintln!("[trace] serialize failed: {e}"),
    }
}
