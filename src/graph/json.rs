//! JSON graph export for the interactive web visualizer (`viz/`).
//!
//! P1 scope: graph structure + full metadata (op payloads, shapes, dtypes,
//! backend assignment, weight info). The schema reserves per-node
//! `stats` / `values` for P2 trace data (real tensor stats + downsampled
//! values) without breaking the page — the visualizer treats absent fields as
//! "no data".
//!
//! The export runs AFTER backend assignment (scheduler `assign_backends`), so
//! the page shows real CPU / Metal colors matching a runtime run.

use super::ops::{NodeMeta, Op};
use super::{Backend, ComputeGraph, DType};
use crate::graph::alloc::GraphAllocator;
use crate::graph::params::{CParams, GraphParams, GraphType};
use crate::graph::scheduler::BackendScheduler;
use crate::models::ModelDef;
use serde_json::{json, Value};

/// The GraphParams the runtime uses for a forward with `n_tokens` (mirrors the
/// model cache's construction; `gpu`/`fuse_qkv` follow the env toggles).
pub fn runtime_gparams(n_tokens: usize, n_ctx: usize, gpu: bool, fuse_qkv: bool) -> GraphParams {
    GraphParams {
        n_tokens,
        n_seqs: 1,
        n_out: 1,
        gtype: if n_tokens == 1 {
            GraphType::Decode
        } else {
            GraphType::Prefill
        },
        cparams: CParams {
            n_ctx,
            n_batch: n_tokens,
            flash_attn: false,
            gpu,
            fuse_qkv,
        },
        weights_version: 1,
    }
}

/// Build the graph the runtime would execute for `gparams` (mirroring the
/// cache: build → assign → FusionPass), so node ids/ops always match what the
/// scheduler actually ran. Used by `--dump-graph*`, the P2 trace, and the P3
/// live server.
pub fn build_runtime_graph(model: &dyn ModelDef, gparams: &GraphParams) -> ComputeGraph {
    let mut g = model.build_graph(gparams);
    #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
    let mut alloc = GraphAllocator::new();
    #[cfg(target_os = "macos")]
    if gparams.cparams.gpu {
        alloc.enable_metal();
    }
    BackendScheduler::new().assign_backends(&mut g, &alloc);
    // FusionPass so the exported graph matches the executed graph (the cache
    // applies it after assignment — silu+mul → SwiGLU etc.).
    {
        use crate::graph::backend::Backend as BackendTrait;
        use crate::graph::fusion::FusionPass;
        let backends: Vec<&dyn BackendTrait> = {
            #[cfg_attr(not(target_os = "macos"), allow(unused_mut))]
            let mut v: Vec<&dyn BackendTrait> = vec![alloc.cpu()];
            #[cfg(target_os = "macos")]
            if gparams.cparams.gpu {
                if let Some(m) = alloc.metal() {
                    v.push(m);
                }
            }
            v
        };
        FusionPass::new().run(&mut g, &backends, &|g, id| match g.node(id).backend {
            Some(Backend::CPU) => Some(0),
            Some(Backend::Metal) => Some(1),
            _ => None,
        });
    }
    g
}

/// Build + export the runtime graph as the web-visualizer JSON. Single source
/// for `--dump-graph-json`, the P2 trace graphs, and the P3 live server.
pub fn export_graph_json(
    model: &dyn ModelDef,
    model_name: &str,
    n_tokens: usize,
    n_ctx: usize,
    gpu: bool,
    fuse_qkv: bool,
) -> Value {
    let gparams = runtime_gparams(n_tokens, n_ctx, gpu, fuse_qkv);
    let g = build_runtime_graph(model, &gparams);
    let kind = if n_tokens == 1 { "decode" } else { "prefill" };
    g.export_json(model_name, kind)
}

impl ComputeGraph {
    /// Export the graph as the web-visualizer JSON document.
    /// `model` = display name (e.g. "qwen2.5-0.5b-instruct-q4_k_m"),
    /// `kind`  = "prefill" | "decode" (the graph type that produced it).
    pub fn export_json(&self, model: &str, kind: &str) -> Value {
        let nodes: Vec<Value> = self
            .nodes
            .iter()
            .map(|n| {
                json!({
                    "id": n.id,
                    "name": n.name,
                    "op": op_name(&n.op),
                    "detail": op_detail(&n.op),
                    "shape": n.out_shape,
                    "dtype": dtype_name(n.out_dtype),
                    "backend": n.backend.map(backend_name),
                    "src": n.src,
                    "meta": meta_json(&n.meta),
                })
            })
            .collect();
        json!({
            "format": "minfer-graph",
            "version": 1,
            "model": model,
            "kind": kind,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "nodes": nodes,
        })
    }
}

fn backend_name(b: Backend) -> &'static str {
    match b {
        Backend::CPU => "cpu",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
    }
}

pub(crate) fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F32 => "f32",
        DType::F16 => "f16",
        DType::I32 => "i32",
        DType::Q8_0 => "q8_0",
    }
}

pub(crate) fn op_name(op: &Op) -> &'static str {
    match op {
        Op::Input => "input",
        Op::Add => "add",
        Op::Mul => "mul",
        Op::Scale(_) => "scale",
        Op::Silu => "silu",
        Op::Softmax { .. } => "softmax",
        Op::RmsNorm { .. } => "rms_norm",
        Op::QkNorm { .. } => "qk_norm",
        Op::MatMul { .. } => "matmul",
        Op::GetRows => "get_rows",
        Op::RoPE { .. } => "rope",
        Op::Attn { .. } => "attn",
        Op::KvcacheStore { .. } => "kvcache_store",
        Op::KvcacheLoad { .. } => "kvcache_load",
        Op::View { .. } => "view",
        Op::Reshape { .. } => "reshape",
        Op::Permute { .. } => "permute",
        Op::SwiGLU => "swiglu",
        Op::FusedBiasRope => "fused_bias_rope",
        Op::BatchMatMul => "batch_matmul",
        Op::FusedQKV { .. } => "fused_qkv",
        Op::FusedQkvNorm { .. } => "fused_qkv_norm",
        Op::FusedFFN => "fused_ffn",
    }
}

/// Op payload as a flat JSON object (page shows it under "detail").
fn op_detail(op: &Op) -> Value {
    match op {
        Op::Input
        | Op::Add
        | Op::Mul
        | Op::Silu
        | Op::GetRows
        | Op::SwiGLU
        | Op::FusedBiasRope
        | Op::FusedFFN
        | Op::BatchMatMul
        | Op::FusedQkvNorm { .. } => json!({}),
        Op::Scale(s) => json!({ "scale": s }),
        Op::Softmax { dim } => json!({ "dim": dim }),
        Op::RmsNorm { eps } => json!({ "eps": eps }),
        Op::QkNorm { hd, nh, eps } => json!({ "hd": hd, "nh": nh, "eps": eps }),
        Op::MatMul { transpose_b } => json!({ "transpose_b": transpose_b }),
        Op::RoPE { style } => json!({ "style": format!("{style:?}") }),
        Op::Attn { mode } => json!({ "mode": format!("{mode:?}") }),
        Op::KvcacheStore { layer } => json!({ "layer": layer }),
        Op::KvcacheLoad { layer } => json!({ "layer": layer }),
        Op::View { offset, shape } => json!({ "offset": offset, "shape": shape }),
        Op::Reshape { shape } => json!({ "shape": shape }),
        Op::Permute { dims } => json!({ "dims": dims }),
        Op::FusedQKV { layer } => json!({ "layer": layer }),
    }
}

/// Node metadata → weight / dims info (what the op reads/writes).
fn meta_json(meta: &NodeMeta) -> Value {
    match meta {
        NodeMeta::None => json!({}),
        NodeMeta::MatMul(m) => json!({
            "weight": m.weight_name,
            "bias": m.bias_name,
            "wtype": m.weight_ttype.name(),
            "in_dim": m.in_dim,
            "out_dim": m.out_dim,
        }),
        NodeMeta::Norm(n) => json!({
            "weight": n.weight_name,
            "bias": n.bias_name,
        }),
        NodeMeta::Rope(r) => json!({
            "freq_base": r.freq_base,
            "freq_scale": r.freq_scale,
            "n_head": r.n_head,
            "hd": r.hd,
        }),
        NodeMeta::Attn(a) => json!({
            "layer": a.layer,
            "n_head": a.n_head,
            "n_head_kv": a.n_head_kv,
            "hd": a.hd,
            "hd_kv": a.hd_kv,
            "nkt": a.nkt,
            "scale": a.scale,
        }),
        NodeMeta::Kvcache(k) => json!({
            "n_embd": k.n_embd,
            "n_head_kv": k.n_head_kv,
        }),
        NodeMeta::Embed(e) => json!({
            "vocab_size": e.vocab_size,
            "weight": e.weight_name,
            "wtype": e.weight_ttype.name(),
        }),
        NodeMeta::FusedQkv(f) => json!({
            "weight": f.qkv_weight,
            "bias_q": f.bias_q,
            "bias_k": f.bias_k,
            "bias_v": f.bias_v,
            "wtype": f.weight_ttype.name(),
            "in_dim": f.in_dim,
            "nqt": f.nqt,
            "nkt": f.nkt,
            "hd": f.hd,
            "nh": f.nh,
            "nk": f.nk,
            "freq_base": f.freq_base,
            "freq_scale": f.freq_scale,
            "kv_elems": f.kv_elems,
        }),
        NodeMeta::FusedFfn(f) => json!({
            "weight": f.gu_weight,
            "wtype": f.weight_ttype.name(),
            "in_dim": f.in_dim,
            "nf": f.nf,
        }),
        NodeMeta::FusedQkvNorm(f) => json!({
            "weight": f.qkv_weight,
            "q_norm": f.q_norm_name,
            "k_norm": f.k_norm_name,
            "wtype": f.weight_ttype.name(),
            "in_dim": f.in_dim,
            "nqt": f.nqt,
            "nkt": f.nkt,
            "hd": f.hd,
            "nh": f.nh,
            "nk": f.nk,
            "freq_base": f.freq_base,
            "freq_scale": f.freq_scale,
            "kv_elems": f.kv_elems,
            "eps": f.eps,
        }),
    }
}
