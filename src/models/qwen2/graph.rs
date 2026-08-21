//! Qwen2 compute-graph construction (Phase 5) + graph forward path (Phase 6).
//!
//! `build_graph` mirrors llama.cpp's `llama_model_qwen2::graph::graph()`
//! (src/models/qwen2.cpp:53) and the plan's §4 example: a declarative IR of
//! the full forward pass. `forward_graph` runs it through the graph stack
//! (builder → assign → fuse → alloc → execute) with a params-only reuse cache.
//!
//! Deviations recorded in docs/GRAPH-REFACTOR-PLAN.md §17:
//! - Full-nt computation (the n_out tail-row `GetRows` optimization is
//!   deferred; tail-row extraction happens on the logits only, which is
//!   numerically identical for the sampled rows).
//! - KV cache lives in the graph allocator's persistent regions (the caller's
//!   `KVCache` is ignored by the graph path).
//! - `weights_version` is static (1) until LoRA support lands.

use std::sync::{Mutex, OnceLock};

use crate::cache::KVCache;
use crate::graph::alloc::GraphAllocator;
use crate::graph::cache::GraphCache;
use crate::graph::backend::Backend;
use crate::graph::fusion::FusionPass;
use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
use crate::graph::params::{CParams, GraphParams, GraphType};
use crate::graph::scheduler::BackendScheduler;
use crate::graph::ComputeGraph;

use super::Qwen2Model;

/// Qwen2 graph construction + execution (Phase 5/6).
pub struct Qwen2Graph;

impl Qwen2Graph {
    /// Build the declarative graph for one forward step (deterministic in
    /// `params` — the reuse invariant).
    pub fn build(model: &Qwen2Model, params: &GraphParams) -> ComputeGraph {
        let hp = &model.hparams;
        let nt = params.n_tokens;
        let ne = hp.n_embd as usize;
        let nh = hp.n_head as usize;
        let nk = hp.n_head_kv as usize;
        let hd = hp.n_embd_head() as usize;
        let nkt = hp.n_kv_embd as usize;
        let hd_kv = nkt / nk;
        let nf = hp.n_ff as usize;
        let eps = hp.f_norm_rms_eps;
        let n_ctx = params.cparams.n_ctx;

        let mut b = crate::graph::builder::GraphBuilder::new();

        let inp_ids = b.input("token_ids", [nt, 1, 1, 1], crate::graph::DType::I32);
        let inp_pos = b.input("positions", [nt, 1, 1, 1], crate::graph::DType::I32);

        let mut h = b.embedding(inp_ids, model.tok_embd.as_ref().unwrap());

        let mode = if params.cparams.flash_attn { AttnMode::Flash } else { AttnMode::Gqa };
        let attn_scale = hp.attention_scale();

        for (il, l) in model.layers.iter().enumerate() {
            let residual = h;

            // pre-norm
            let normed = b.rms_norm(h, l.attn_norm.as_ref(), eps);

            // Q/K/V projections (+ biases)
            let q = b.matmul(normed, l.wq.as_ref().unwrap(), l.bq.as_ref());
            let k = b.matmul(normed, l.wk.as_ref().unwrap(), l.bk.as_ref());
            let v = b.matmul(normed, l.wv.as_ref().unwrap(), l.bv.as_ref());

            // RoPE (Q: nh heads, K: nk heads, both hd dims)
            let q = b.rope(
                q, inp_pos, hp.rope_style,
                RoPEMeta { freq_base: hp.rope_freq_base, freq_scale: hp.rope_freq_scale, n_head: nh, hd },
            );
            let k = b.rope(
                k, inp_pos, hp.rope_style,
                RoPEMeta { freq_base: hp.rope_freq_base, freq_scale: hp.rope_freq_scale, n_head: nk, hd },
            );

            // KV cache (persistent region; positions are data)
            b.kvcache_store(il, k, v, inp_pos, n_ctx);
            let kv = b.kvcache_load(il, nkt, n_ctx, nk);

            // attention
            let attn_out = b.attn(
                q, kv, inp_pos, mode,
                AttnMeta { n_head: nh, n_head_kv: nk, hd, hd_kv, nkt, scale: attn_scale },
            );

            // output projection + residual
            let wo = b.matmul(attn_out, l.wo.as_ref().unwrap(), None);
            h = b.add(residual, wo);

            // FFN (SwiGLU); built as silu+mul so the fusion pass folds it
            let residual = h;
            let normed = b.rms_norm(h, l.ffn_norm.as_ref(), eps);
            let gate = b.matmul(normed, l.ffn_gate.as_ref().unwrap(), None);
            let up = b.matmul(normed, l.ffn_up.as_ref().unwrap(), None);
            let g = b.silu(gate);
            let ffn_out = b.mul(g, up);
            let ffn_out = b.matmul(ffn_out, l.ffn_down.as_ref().unwrap(), None);
            h = b.add(residual, ffn_out);
        }

        // output: norm + lm_head
        let normed = b.rms_norm(h, model.output_norm.as_ref(), eps);
        let logits = b.matmul(normed, model.output.as_ref().unwrap(), model.output_b.as_ref());
        b.output(logits);

        b.build()
    }

    /// Register every weight the graph references on the allocator's backend.
    fn register_graph_weights(model: &Qwen2Model, alloc: &mut GraphAllocator) {
        for t in [
            &model.tok_embd,
            &model.output_norm,
            &model.output,
            &model.output_b,
        ] {
            if let Some(t) = t {
                let name = t.name.clone();
                alloc.register_weight(&name, t.clone());
            }
        }
        for l in &model.layers {
            for t in [
                &l.attn_norm, &l.wq, &l.bq, &l.wk, &l.bk, &l.wv, &l.bv, &l.wo,
                &l.ffn_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down,
            ] {
                if let Some(t) = t {
                    let name = t.name.clone();
                    alloc.register_weight(&name, t.clone());
                }
            }
        }
    }

    /// Graph-based forward: build/assign/fuse/alloc/execute with reuse.
    /// `kv` is ignored (the graph owns its KV in persistent regions).
    pub fn forward(
        model: &Qwen2Model,
        tokens: &[u32],
        positions: &[usize],
        _kv: &mut KVCache,
        n_out: usize,
    ) -> Vec<f32> {
        let nt = tokens.len();
        debug_assert!(n_out <= nt);
        let params = GraphParams {
            n_tokens: nt,
            n_seqs: 1,
            gtype: if nt == 1 { GraphType::Decode } else { GraphType::Prefill },
            cparams: CParams {
                n_ctx: model.hparams.max_seq_len as usize,
                n_batch: nt,
                flash_attn: false,
            },
            weights_version: 1,
        };

        let mut guard = graph_cache().lock().unwrap();
        let cache = &mut *guard;

        if !cache.try_reuse(&params) {
            let mut graph = Self::build(model, &params);
            let sched = BackendScheduler::new();
            {
                let alloc = cache.alloc();
                Self::register_graph_weights(model, alloc);
                sched.assign_backends(&mut graph, alloc);
                // fusion pass: fold silu+gate*mul into SwiGLU (CPU supports it)
                let backends: [&dyn Backend; 1] = [alloc.cpu()];
                FusionPass::new().run(&mut graph, &backends, &|_, _| Some(0));
                alloc.alloc_graph(&graph).unwrap();
            }
            cache.replace_graph(graph, params);
        }

        let (graph, alloc) = cache.current().unwrap();

        // refresh input data (positions/ids are data, not topology)
        let ids: Vec<u32> = tokens.to_vec();
        alloc.fill_input_i32(graph, "token_ids", &ids).unwrap();
        let pos: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        alloc.fill_input_i32(graph, "positions", &pos).unwrap();

        let sched = BackendScheduler::new();
        sched.execute(graph, alloc).unwrap();

        // extract the last n_out rows of logits ([nv, nt] → [n_out * nv])
        let nv = model.hparams.n_vocab as usize;
        let logits = alloc.get_buffer(graph, graph.outputs[0]).expect("logits buffer");
        let off = (nt - n_out) * nv;
        logits[off..off + n_out * nv].to_vec()
    }
}

/// Process-wide graph cache: the allocator inside it owns the KV cache, so it
/// must survive across steps (and rebuilds). Single model per process today.
fn graph_cache() -> &'static Mutex<GraphCache> {
    static CACHE: OnceLock<Mutex<GraphCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(GraphCache::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Path to the locally cached Qwen2.5-0.5B Q4_0 (downloaded via
    /// `minfer download hf Qwen/Qwen2.5-0.5B-Instruct-GGUF Q4_0`).
    fn cached_model_path() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
        if p.exists() { Some(p) } else { None }
    }

    fn argmax(x: &[f32]) -> u32 {
        let mut best = 0usize;
        for i in 1..x.len() {
            if x[i] > x[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Compare the last logits row (n_out=1 → the whole returned vector).
    fn compare(tag: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "[{tag}] logits length mismatch");
        let mut maxd = 0.0f32;
        for i in 0..a.len() {
            maxd = maxd.max((a[i] - b[i]).abs());
        }
        eprintln!("[{tag}] logits max abs diff: {maxd:.3e}");
        assert!(maxd < 1e-3, "[{tag}] graph vs forward logits diverge (max diff {maxd:.3e})");
    }

    /// Phase 6 verification: the graph path must reproduce forward.rs logits on
    /// a real model, for prefill and for a decode step (KV carried across).
    #[test]
    fn graph_logits_match_forward_real_model() {
        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping graph-logits test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");

        let ctx = &gguf.parts[0].ctx;
        let tok = crate::tokenizer::Tokenizer::load(ctx);
        let ids = tok.encode("The capital of France is");
        assert!(!ids.is_empty());
        let positions: Vec<usize> = (0..ids.len()).collect();

        // prefill (both paths run on their own KV state)
        let mut kv_f = KVCache::new(model.n_layer(), model.n_kv_embd(), 4096);
        let lf = model.forward(&ids, &positions, &mut kv_f, 1);
        let mut kv_g = KVCache::new(model.n_layer(), model.n_kv_embd(), 4096);
        let lg = model.forward_graph(&ids, &positions, &mut kv_g, 1);
        compare("prefill", &lf, &lg);

        // decode step with the argmax token (KV carries over in both paths)
        let next = argmax(&lf);
        let pos = ids.len();
        let lf2 = model.forward(&[next], &[pos], &mut kv_f, 1);
        let lg2 = model.forward_graph(&[next], &[pos], &mut kv_g, 1);
        compare("decode", &lf2, &lg2);
    }
}
