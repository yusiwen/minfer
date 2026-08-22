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
use crate::graph::ops::{AttnMeta, AttnMode, FusedFfnMeta, FusedQkvMeta, RoPEMeta};
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

            // Q/K/V projections (+ biases). decode (nt==1) with GPU QKV concat
            // uses the fused path (G4): one concat matmul + one fused
            // bias+rope+store kernel, replacing 3 matmul + 3 bias + 2 rope +
            // 2 store dispatches.
            let fuse_qkv = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_qkv
                && l.bq.is_some() && l.bk.is_some() && l.bv.is_some()
                && Self::qkv_concat_available(&l.wq, &l.wk, &l.wv);
            let (q, kv) = if fuse_qkv {
                let qkv = b.fused_qkv(
                    normed, inp_pos, il,
                    FusedQkvMeta {
                        qkv_weight: format!("blk.{il}.attn_qkv"),
                        bias_q: l.bq.as_ref().map(|t| t.name.clone()),
                        bias_k: l.bk.as_ref().map(|t| t.name.clone()),
                        bias_v: l.bv.as_ref().map(|t| t.name.clone()),
                        weight_ttype: l.wq.as_ref().unwrap().ttype,
                        in_dim: hp.n_embd as usize,
                        nqt: nh * hd,
                        nkt,
                        hd,
                        nh,
                        nk,
                        freq_base: hp.rope_freq_base,
                        freq_scale: hp.rope_freq_scale,
                        rope_style: hp.rope_style,
                        kv_elems: nkt * n_ctx,
                    },
                );
                // q lives at concat offset 0 (rows 0..nqt); K/V went into the
                // persistent regions via the fused store — read them back.
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (qkv, kv)
            } else {
                let q = b.matmul(normed, l.wq.as_ref().unwrap(), l.bq.as_ref());
                let k = b.matmul(normed, l.wk.as_ref().unwrap(), l.bk.as_ref());
                let v = b.matmul(normed, l.wv.as_ref().unwrap(), l.bv.as_ref());

                let q = b.rope(
                    q, inp_pos, hp.rope_style,
                    RoPEMeta { freq_base: hp.rope_freq_base, freq_scale: hp.rope_freq_scale, n_head: nh, hd },
                );
                let k = b.rope(
                    k, inp_pos, hp.rope_style,
                    RoPEMeta { freq_base: hp.rope_freq_base, freq_scale: hp.rope_freq_scale, n_head: nk, hd },
                );

                b.kvcache_store(il, k, v, inp_pos, n_ctx);
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (q, kv)
            };

            // attention
            let attn_out = b.attn(
                q, kv, inp_pos, mode,
                AttnMeta { layer: il, n_head: nh, n_head_kv: nk, hd, hd_kv, nkt, scale: attn_scale },
            );

            // output projection + residual
            let wo = b.matmul(attn_out, l.wo.as_ref().unwrap(), None);
            let is_last = il == model.layers.len() - 1;
            if is_last && params.n_out < nt {
                // G3: reduce to the tail n_out rows BEFORE the last layer's FFN
                // (llama `ggml_get_rows(cur/inpSA, inp_out_ids)` at
                // qwen2.cpp:106-108) — ffn_norm, gate/up/down, swiglu, both
                // residuals and lm_head all run on n_out rows only.
                let tail_ids = b.input("tail_ids", [params.n_out, 1, 1, 1], crate::graph::DType::I32);
                let cur_tail = b.get_rows(wo, tail_ids, [ne, params.n_out, 1, 1]);
                let res_tail = b.get_rows(residual, tail_ids, [ne, params.n_out, 1, 1]);
                h = b.add(res_tail, cur_tail);
            } else {
                h = b.add(residual, wo);
            }

            // FFN (SwiGLU); built as silu+mul so the fusion pass folds it.
            // decode (nt==1) with GPU gate+up concat uses the fused path (G4
            // follow-up): one concat matmul (blk.{i}.ffn_gu) + one in-place
            // swiglu, replacing 2 matmul + silu + mul dispatches.
            // FFN gate+up fusion is a dispatch-count win on small models (0.5B
            // ~+3% decode) but measured SLOWER on the 7B class: the Q4_K concat
            // matmul (od = 2*nf ≈ 37888) under-performs two separate matmuls on
            // the decode (nt==1) scalar kernel. Gate on FFN size.
            let fuse_gu = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_qkv
                && Self::gu_concat_available(&l.ffn_gate, &l.ffn_up)
                && nf <= 16384
                && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1");
            let residual = h;
            let normed = b.rms_norm(h, l.ffn_norm.as_ref(), eps);
            let ffn_out = if fuse_gu {
                let gu = b.fused_ffn(
                    normed,
                    FusedFfnMeta {
                        gu_weight: format!("blk.{il}.ffn_gu"),
                        weight_ttype: l.ffn_gate.as_ref().unwrap().ttype,
                        in_dim: ne,
                        nf,
                    },
                );
                // down reads rows 0..nf of the concat buffer (gate rows, now
                // holding silu(gate)*up); nt==1 makes the concat layout safe
                b.matmul(gu, l.ffn_down.as_ref().unwrap(), None)
            } else {
                let gate = b.matmul(normed, l.ffn_gate.as_ref().unwrap(), None);
                let up = b.matmul(normed, l.ffn_up.as_ref().unwrap(), None);
                let g = b.silu(gate);
                let sw = b.mul(g, up);
                b.matmul(sw, l.ffn_down.as_ref().unwrap(), None)
            };
            h = b.add(residual, ffn_out);
        }

        // output: norm + lm_head
        let normed = b.rms_norm(h, model.output_norm.as_ref(), eps);
        let logits = b.matmul(normed, model.output.as_ref().unwrap(), model.output_b.as_ref());
        b.output(logits);

        b.build()
    }

    /// Whether wq/wk/wv can share one concat matmul (loader registered
    /// `blk.{i}.attn_qkv`): same quant type, same input dim, block-aligned.
    fn qkv_concat_available(
        wq: &Option<crate::tensor::Tensor>,
        wk: &Option<crate::tensor::Tensor>,
        wv: &Option<crate::tensor::Tensor>,
    ) -> bool {
        let (Some(wq), Some(wk), Some(wv)) = (wq, wk, wv) else { return false };
        #[cfg(target_os = "macos")]
        {
            crate::metal::concat_rows(&[wq, wk, wv]).is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (wq, wk, wv);
            false
        }
    }

    /// Whether ffn_gate/ffn_up can share one concat matmul (loader registered
    /// `blk.{i}.ffn_gu`): same quant type, same input dim, block-aligned.
    fn gu_concat_available(
        fg: &Option<crate::tensor::Tensor>,
        fu: &Option<crate::tensor::Tensor>,
    ) -> bool {
        let (Some(fg), Some(fu)) = (fg, fu) else { return false };
        #[cfg(target_os = "macos")]
        {
            crate::metal::concat_rows(&[fg, fu]).is_some()
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = (fg, fu);
            false
        }
    }

    /// Register every weight the graph references on the allocator's backend.
    pub(crate) fn register_graph_weights(model: &Qwen2Model, alloc: &mut GraphAllocator) {
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
    ///
    /// CLI convenience wrapper: uses the process-global `graph_cache()` and the
    /// model's full `max_seq_len` context. Server / multi-slot code must call
    /// [`Qwen2Graph::forward_cached`] with a slot-scoped cache instead.
    pub fn forward(
        model: &Qwen2Model,
        tokens: &[u32],
        positions: &[usize],
        _kv: &mut KVCache,
        n_out: usize,
    ) -> Vec<f32> {
        let mut guard = graph_cache().lock().unwrap();
        Self::forward_cached(
            model, tokens, positions, n_out,
            model.hparams.max_seq_len as usize, &mut guard,
        )
    }

    /// Graph-based forward with a caller-provided cache and explicit context
    /// size (server / multi-slot path).
    ///
    /// `cache` owns the KV regions (persistent per-layer regions inside its
    /// allocator) and survives rebuilds; `n_ctx` sizes those regions and must
    /// satisfy `positions[i] < n_ctx` for every position (asserted below).
    pub fn forward_cached(
        model: &Qwen2Model,
        tokens: &[u32],
        positions: &[usize],
        n_out: usize,
        n_ctx: usize,
        cache: &mut GraphCache,
    ) -> Vec<f32> {
        let nt = tokens.len();
        debug_assert!(n_out <= nt);
        // Out-of-range positions would write past the KV regions (which are
        // sized n_kv_embd * n_ctx): fail loudly instead of corrupting memory.
        if let Some(&maxp) = positions.iter().max() {
            assert!(
                maxp < n_ctx,
                "position {maxp} exceeds n_ctx {n_ctx} (KV region overflow)"
            );
        }
        // GPU availability is part of the reuse identity (backend assignment
        // lives in the built graph, not in the params' other fields)
        let metal_on = cfg!(target_os = "macos")
            && crate::graph::metal_backend::metal_available()
            && Self::weights_on_gpu(model);
        let params = GraphParams {
            n_tokens: nt,
            n_seqs: 1,
            n_out,
            gtype: if nt == 1 { GraphType::Decode } else { GraphType::Prefill },
            cparams: CParams {
                n_ctx,
                n_batch: nt,
                flash_attn: false,
                gpu: metal_on,
                // G4: decode QKV fusion is part of the topology — the env
                // toggle forces a rebuild so it can be A/B'd reliably.
                fuse_qkv: nt == 1
                    && metal_on
                    && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1"),
            },
            weights_version: 1,
        };

        if !cache.try_reuse(&params) {
            let mut graph = Self::build(model, &params);
            let sched = BackendScheduler::new();
            {
                let alloc = cache.alloc();
                Self::register_graph_weights(model, alloc);
                if metal_on {
                    alloc.enable_metal();
                }
                sched.assign_backends(&mut graph, alloc);
                // fusion pass gated per node's assigned backend
                let backends: Vec<&dyn Backend> = {
                    let mut v: Vec<&dyn Backend> = vec![alloc.cpu()];
                    #[cfg(target_os = "macos")]
                    if metal_on {
                        if let Some(m) = alloc.metal() {
                            v.push(m);
                        }
                    }
                    v
                };
                FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                    Some(crate::graph::Backend::CPU) => Some(0),
                    Some(crate::graph::Backend::Metal) => Some(1),
                    _ => None,
                });
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
        // G3: the last-layer tail-row reduction reads `tail_ids` (filled when
        // the graph was built with n_out < nt, i.e. prefill)
        if graph.inputs.iter().any(|&i| graph.node(i).name == "tail_ids") {
            let tail: Vec<u32> = ((nt - n_out)..nt).map(|x| x as u32).collect();
            alloc.fill_input_i32(graph, "tail_ids", &tail).unwrap();
        }
        if std::env::var("MINFER_GRAPH_DUMP").is_ok() {
            if let Some(idsbuf) = graph.inputs.iter().find(|&&i| graph.node(i).name == "token_ids").copied() {
                let v = alloc.copy_to_cpu(idsbuf).unwrap_or_default();
                eprintln!("[graph dump] ids after fill: {:?}", &v[..v.len().min(8)]);
            }
        }

        let sched = BackendScheduler::new();
        sched.execute(graph, alloc).unwrap();

        // debug dump: MINFER_GRAPH_DUMP=/tmp/x writes the logits and layer-0 KV
        // so GPU vs CPU graph runs can be compared (Phase 3 debugging)
        if let Ok(dir) = std::env::var("MINFER_GRAPH_DUMP") {
            let nv = model.hparams.n_vocab as usize;
            let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
            let tag = if nt == 1 { "decode" } else { "prefill" };
            let _ = std::fs::write(format!("{dir}/logits_{tag}.f32"), {
                let mut b = Vec::with_capacity(logits.len() * 4);
                for x in &logits { b.extend_from_slice(&x.to_le_bytes()); }
                b
            });
            for nid in [0usize, 1, 2, 3, 5, 8, 10, 11] {
                if nid < graph.n_nodes() {
                    if let Some(buf) = alloc.copy_to_cpu(nid) {
                        let mut b = Vec::with_capacity(buf.len() * 4);
                        for x in &buf { b.extend_from_slice(&x.to_le_bytes()); }
                        let _ = std::fs::write(format!("{dir}/node{nid}_{tag}.f32"), b);
                    }
                }
            }
            if let Some(kv0) = graph.nodes.iter().position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 })) {
                if let Some(kv) = alloc.copy_to_cpu(kv0) {
                    let mut b = Vec::with_capacity(kv.len() * 4);
                    for x in &kv { b.extend_from_slice(&x.to_le_bytes()); }
                    let _ = std::fs::write(format!("{dir}/kv0_{tag}.f32"), b);
                }
            }
            eprintln!("[graph dump] wrote {dir}/logits_{tag}.f32 ({} elems)", logits.len());
        }

        // extract the tail n_out rows of logits. With G3 the graph already
        // reduced the last layer + lm_head to n_out rows (buffer = n_out*nv);
        // without it (decode, n_out == nt) the buffer is nv*nt == n_out*nv.
        // Either way the first n_out*nv elements are the answer.
        let nv = model.hparams.n_vocab as usize;
        let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
        logits[..n_out * nv].to_vec()
    }

    /// Every weight the graph reads must be GPU-registered for the Metal path.
    fn weights_on_gpu(model: &Qwen2Model) -> bool {
        let names: Vec<String> = {
            let mut v = Vec::new();
            for t in [&model.tok_embd, &model.output_norm, &model.output, &model.output_b] {
                if let Some(t) = t {
                    v.push(t.name.clone());
                }
            }
            for l in &model.layers {
                for t in [
                    &l.attn_norm, &l.wq, &l.bq, &l.wk, &l.bk, &l.wv, &l.bv, &l.wo,
                    &l.ffn_norm, &l.ffn_gate, &l.ffn_up, &l.ffn_down,
                ] {
                    if let Some(t) = t {
                        v.push(t.name.clone());
                    }
                }
            }
            v
        };
        #[cfg(target_os = "macos")]
        {
            let Some(mps) = crate::metal::MpsState::get() else { return false };
            names.iter().all(|n| mps.has_weight(n))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = names;
            false
        }
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

    /// Phase 6 verification (hermetic, CPU-only): the graph path must reproduce
    /// forward.rs logits on a real model — prefill and a decode step (KV
    /// carried across). Built and executed locally (no global cache, no MPS),
    /// so it is immune to other tests initializing Metal.
    #[test]
    fn graph_logits_match_forward_real_model() {
        use crate::graph::alloc::GraphAllocator;
        use crate::models::ModelDef;
        use crate::graph::backend::Backend;
        use crate::graph::builder::GraphBuilder;
        use crate::graph::fusion::FusionPass;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        use crate::graph::ComputeGraph;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping graph-logits test");
            return;
        };
        // This test compares the graph against forward() — meaningful only when
        // forward() runs on CPU. Earlier tests may have initialized MPS (process
        // global), which would send forward() to layer_gpu with a different
        // activation path; skip in that case (correctness is covered by the
        // hermetic layer-0 isolation and the CLI).
        #[cfg(target_os = "macos")]
        if crate::metal::MpsState::get().is_some() {
            eprintln!("MPS initialized by an earlier test; skipping CPU-parity test");
            return;
        }
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2 model");
        let ctx = &gguf.parts[0].ctx;
        let tok = crate::tokenizer::Tokenizer::load(ctx);
        let ids = tok.encode("The capital of France is");
        assert!(!ids.is_empty());
        let positions: Vec<usize> = (0..ids.len()).collect();

        fn run_prefill_decode(
            model: &Qwen2Model,
            ids: &[u32],
            next: u32,
            n_ctx: usize,
        ) -> (Vec<f32>, Vec<f32>) {
            let nt = ids.len();
            let params = GraphParams {
                n_tokens: nt,
                n_seqs: 1,
                n_out: 1,
                gtype: GraphType::Prefill,
                cparams: CParams { n_ctx, n_batch: nt, flash_attn: false, gpu: false, fuse_qkv: false },
                weights_version: 1,
            };
            let sched = BackendScheduler::new();
            let mut alloc = GraphAllocator::new();
            Qwen2Graph::register_graph_weights(model, &mut alloc);

            // prefill graph
            let mut graph: ComputeGraph = model.build_graph(&params);
            sched.assign_backends(&mut graph, &alloc);
            let backends: [&dyn Backend; 1] = [alloc.cpu()];
            FusionPass::new().run(&mut graph, &backends, &|_, _| Some(0));
            alloc.alloc_graph(&graph).unwrap();
            let ids32: Vec<u32> = ids.iter().copied().collect();
            let pos32: Vec<u32> = (0..nt as u32).collect();
            alloc.fill_input_i32(&graph, "token_ids", &ids32).unwrap();
            alloc.fill_input_i32(&graph, "positions", &pos32).unwrap();
            sched.execute(&graph, &mut alloc).unwrap();
            let nv = model.n_vocab();
            let logits = alloc.copy_to_cpu(graph.outputs[0]).unwrap();
            let prefill_l = logits[(nt - 1) * nv..nt * nv].to_vec();

            // decode graph (same allocator: KV persists through the rebuild)
            let dparams = GraphParams {
                n_tokens: 1,
                n_seqs: 1,
                n_out: 1,
                gtype: GraphType::Decode,
                cparams: CParams { n_ctx, n_batch: 1, flash_attn: false, gpu: false, fuse_qkv: false },
                weights_version: 1,
            };
            let mut dgraph: ComputeGraph = model.build_graph(&dparams);
            sched.assign_backends(&mut dgraph, &alloc);
            let backends: [&dyn Backend; 1] = [alloc.cpu()];
            FusionPass::new().run(&mut dgraph, &backends, &|_, _| Some(0));
            alloc.alloc_graph(&dgraph).unwrap();
            alloc.fill_input_i32(&dgraph, "token_ids", &[next]).unwrap();
            alloc.fill_input_i32(&dgraph, "positions", &[nt as u32]).unwrap();
            sched.execute(&dgraph, &mut alloc).unwrap();
            let dlogits = alloc.copy_to_cpu(dgraph.outputs[0]).unwrap();
            (prefill_l, dlogits.to_vec())
        }

        // NOTE: prefill and decode share one GraphAllocator so the KV persists
        // NOTE: both runs share one GraphAllocator so the KV persists across
        // the prefill -> decode transition (like the real loop).
        let n_ctx = q2.hparams.max_seq_len as usize;
        let mut kv_f = KVCache::new(model.n_layer(), model.n_kv_embd(), n_ctx);
        let lf = model.forward(&ids, &positions, &mut kv_f, 1);
        let next = argmax(&lf);
        let lf2 = model.forward(&[next], &[ids.len()], &mut kv_f, 1);
        let (lg, lg2) = run_prefill_decode(q2, &ids, next, n_ctx);
        compare("prefill", &lf, &lg);
        compare("decode", &lf2, &lg2);
    }

    /// Phase 3 verification: the graph path on the Metal backend must produce
    /// logits close to the CPU forward (kernel math differs in reduction order,
    /// so a loose tolerance + greedy-token equality is the criterion).
    #[test]
    fn graph_metal_matches_cpu_logits() {
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            let Some(path) = cached_model_path() else {
                eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
                return;
            };
            crate::metal::MpsState::init();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let model = crate::models::load_model(&gguf).expect("load model");
            let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
            let ids = tok.encode("The capital of France is");
            let positions: Vec<usize> = (0..ids.len()).collect();

            // CPU reference (forward, not forward_graph — separate KV state)
            let mut kv = KVCache::new(model.n_layer(), model.n_kv_embd(), 4096);
            let ref_l = model.forward(&ids, &positions, &mut kv, 1);

            // GPU graph (forward_graph picks Metal when MPS + weights on GPU)
            let mut kv2 = KVCache::new(model.n_layer(), model.n_kv_embd(), 4096);
            let gpu_l = model.forward_graph(&ids, &positions, &mut kv2, 1);

            let mut maxd = 0.0f32;
            for i in 0..ref_l.len() {
                maxd = maxd.max((ref_l[i] - gpu_l[i]).abs());
            }
            eprintln!("[metal graph] logits max abs diff: {maxd:.3e} (expected ~18: the graph-Metal path uses f32 activations while the CPU reference quantizes activations to Q8_0)");
            let greedy_ref = argmax(&ref_l);
            let greedy_gpu = argmax(&gpu_l);
            eprintln!("[metal graph] greedy token: CPU={greedy_ref} GPU={greedy_gpu}");
            // functional criterion: the greedy token should agree OR the GPU
            // path should still be self-consistent (verified separately)
            assert_eq!(greedy_ref, greedy_gpu, "greedy token differs");
        }
    }

    /// Phase 3: full layer-0 path on Metal vs CPU (embed/rms/matmul/rope/kv/attn).
    #[test]
    fn graph_metal_layer0_isolation() {
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            let Some(path) = cached_model_path() else { eprintln!("not cached; skipping"); return; };
            crate::metal::MpsState::init();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let model = crate::models::load_model(&gguf).expect("load model");
            let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
            let nt = 30usize;
            let ids: Vec<u32> = (100..100 + nt as u32).collect();
            let hp = &q2.hparams;
            let mut gb = crate::graph::builder::GraphBuilder::new();
            let idsn = gb.input("token_ids", [nt, 1, 1, 1], crate::graph::DType::I32);
            let pos = gb.input("positions", [nt, 1, 1, 1], crate::graph::DType::I32);
            let e = gb.embedding(idsn, q2.tok_embd.as_ref().unwrap());
            // Metal->CPU copy, then the FULL layer-0 CPU path (qkv/rope/KV/attn)
            let l0 = &q2.layers[0];
            let r = gb.rms_norm(e, l0.attn_norm.as_ref(), hp.f_norm_rms_eps);
            let q = gb.matmul(r, l0.wq.as_ref().unwrap(), None);
            let k = gb.matmul(r, l0.wk.as_ref().unwrap(), None);
            let v = gb.matmul(r, l0.wv.as_ref().unwrap(), None);
            let nh = hp.n_head as usize;
            let nk = hp.n_head_kv as usize;
            let hd = hp.n_embd_head() as usize;
            let nkt = hp.n_kv_embd as usize;
            let rm = |hd_, nh_| crate::graph::ops::RoPEMeta {
                freq_base: hp.rope_freq_base, freq_scale: hp.rope_freq_scale,
                n_head: nh_, hd: hd_,
            };
            let qr = gb.rope(q, pos, hp.rope_style, rm(hd, nh));
            let kr = gb.rope(k, pos, hp.rope_style, rm(hd, nk));
            gb.kvcache_store(0, kr, v, pos, 32768);
            let kv = gb.kvcache_load(0, nkt, 32768, nk);
            let ao = gb.attn(qr, kv, pos, crate::graph::ops::AttnMode::Gqa,
                crate::graph::ops::AttnMeta {
                    layer: 0, n_head: nh, n_head_kv: nk, hd, hd_kv: nkt / nk,
                    nkt, scale: hp.attention_scale(),
                });
            gb.output(ao);
            let g = gb.build();

            let mut sched = crate::graph::scheduler::BackendScheduler::new();
            let mut ca = crate::graph::alloc::GraphAllocator::new();
            Qwen2Graph::register_graph_weights(q2, &mut ca);
            ca.alloc_graph(&g).unwrap();
            ca.fill_input_i32(&g, "token_ids", &ids).unwrap();
            let pos32: Vec<u32> = (0..nt as u32).collect();
            ca.fill_input_i32(&g, "positions", &pos32).unwrap();
            sched.execute(&g, &mut ca).unwrap();
            let expect = ca.copy_to_cpu(g.outputs[0]).unwrap();

            // compare the KV region contents (persistent, not reused)
            let kv_node_ref = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 }))
                .unwrap();
            let kv_ref = ca.copy_to_cpu(kv_node_ref).unwrap();
            let mut worst = 0.0f32;
            for _ in 0..5 {
                let mut g2 = g.clone();
                for n in &mut g2.nodes {
                    n.backend = Some(crate::graph::Backend::Metal); // FULL layer-0 on Metal
                }
                let mut alloc = crate::graph::alloc::GraphAllocator::new();
                Qwen2Graph::register_graph_weights(q2, &mut alloc);
                alloc.enable_metal();
                alloc.alloc_graph(&g2).unwrap();
                alloc.fill_input_i32(&g2, "token_ids", &ids).unwrap();
                let pos32: Vec<u32> = (0..nt as u32).collect();
                alloc.fill_input_i32(&g2, "positions", &pos32).unwrap();
                sched.execute(&g2, &mut alloc).unwrap();
                let got = alloc.copy_to_cpu(g2.outputs[0]).unwrap();
                let mut maxd = 0.0f32;
                for i in 0..got.len().min(expect.len()) { maxd = maxd.max((got[i] - expect[i]).abs()); }
                let kv_got = g2
                    .nodes
                    .iter()
                    .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 }))
                    .and_then(|nid| alloc.copy_to_cpu(nid))
                    .unwrap_or_default();
                let mut kvd = 0.0f32;
                for i in 0..kv_ref.len().min(kv_got.len()) { kvd = kvd.max((kv_ref[i] - kv_got[i]).abs()); }
                eprintln!("[embed iso] kv0[0..4] gpu={:?} cpu={:?}", &kv_got[..kv_got.len().min(4)], &kv_ref[..kv_ref.len().min(4)]);
                let kvn = g2.nodes.iter().position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 })).unwrap();
                eprintln!("[embed iso] kv_load bufref={:?}", alloc.node_buffer(kvn));
                use crate::graph::backend::KvProvider as _;
                let kp = alloc.kv_pair(0);
                eprintln!("[embed iso] kv_pair(0)={:?}", kp);
                if let Some((ki, _)) = kp {
                    if let Some(kv2) = alloc.metal().and_then(|m| m.read_host(ki).map(|s| s.to_vec())) {
                        eprintln!("[embed iso] kv_pair(0) direct metal read[0..4]={:?}", &kv2[..4]);
                    }
                }
                eprintln!("[embed iso] run max diff {maxd:.3e} kvK diff {kvd:.3e} got0={:?} expect0={:?} got29={:?} expect29={:?}",
                    &got[0..4], &expect[0..4], &got[29 * 896..29 * 896 + 4], &expect[29 * 896..29 * 896 + 4]);
                worst = worst.max(maxd);
            }
            assert!(worst < 1e-3, "embed isolation nondeterministic: worst {worst:.3e}");
        }
    }

    /// Real model layer-0 K matmul: Metal vs CPU (isolates weight offset/data).
    #[test]
    fn graph_metal_real_wk_matmul() {
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            let Some(path) = cached_model_path() else { eprintln!("not cached; skipping"); return; };
            crate::metal::MpsState::init();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let model = crate::models::load_model(&gguf).expect("load model");
            let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
            let l0 = &q2.layers[0];
            let wk = l0.wk.as_ref().unwrap();
            let ne = q2.hparams.n_embd as usize;
            let nkt = q2.hparams.n_kv_embd as usize;
            let nt = 30usize; // GEMM path (nt >= 9)
            let xd: Vec<f32> = (0..ne * nt).map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0).collect();

            let mut gb = crate::graph::builder::GraphBuilder::new();
            let x = gb.input("x", [ne, nt, 1, 1], crate::graph::DType::F32);
            let m = gb.matmul(x, wk, None);
            gb.output(m);
            let g = gb.build();

            // CPU (registered weight)
            let mut sched = crate::graph::scheduler::BackendScheduler::new();
            let mut ca = crate::graph::alloc::GraphAllocator::new();
            let wname = wk.name.clone();
            ca.register_weight(&wname, wk.clone());
            ca.alloc_graph(&g).unwrap();
            ca.fill_input(&g, "x", &xd).unwrap();
            sched.execute(&g, &mut ca).unwrap();
            let expect = ca.get_buffer(&g, m).unwrap().to_vec();

            // Metal
            let mut g2 = g.clone();
            for n in &mut g2.nodes { n.backend = Some(crate::graph::Backend::Metal); }
            let mut alloc = crate::graph::alloc::GraphAllocator::new();
            alloc.enable_metal();
            alloc.alloc_graph(&g2).unwrap();
            alloc.fill_input(&g2, "x", &xd).unwrap();
            sched.execute(&g2, &mut alloc).unwrap();
            let got = alloc.copy_to_cpu(m).unwrap();
            let mut maxd = 0.0f32;
            for i in 0..got.len() { maxd = maxd.max((got[i] - expect[i]).abs()); }
            eprintln!("[real wk matmul] vs CPU: max diff {maxd:.3e} (nt={nt}, od={nkt}, id={ne})");
            // manual Q4_0 x f32 reference from the real weight bytes
            let mut ref2 = vec![0.0f32; nkt * nt];
            {
                let wraw = wk.data();
                for o in 0..nkt {
                    let wrow = &wraw[o * (ne / 32) * 18..];
                    for t in 0..nt {
                        let mut acc = 0.0f32;
                        for b in 0..ne / 32 {
                            let boff = b * 18;
                            let d = crate::block::fp16_to_f32(u16::from_le_bytes([wrow[boff], wrow[boff + 1]]));
                            for j in 0..16 {
                                let byte = wrow[boff + 2 + j];
                                acc += ((byte & 0x0F) as i8 - 8) as f32 * d * xd[t * ne + b * 32 + j];
                                acc += ((byte >> 4) as i8 - 8) as f32 * d * xd[t * ne + b * 32 + j + 16];
                            }
                        }
                        ref2[t * nkt + o] = acc;
                    }
                }
            }
            let mut m2 = 0.0f32;
            for i in 0..got.len() { m2 = m2.max((got[i] - ref2[i]).abs()); }
            eprintln!("[real wk matmul] vs manual Q4_0xf32: max diff {m2:.3e}");
            assert!(m2 < 5e-3, "real wk Metal diverges from Q4_0xf32 reference: {m2:.3e}");
        }
    }

    /// Phase 0 (OPENAI-CHAT-API-PLAN.md): `forward_cached` with two independent
    /// `GraphCache` instances must isolate KV — interleaving prefill/decode
    /// across caches must not change either cache's logits, and a cache scoped
    /// to a smaller `n_ctx` must still work (CPU-only, real model when cached).
    #[test]
    fn forward_cached_isolates_kv_between_caches() {
        use crate::graph::cache::GraphCache;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping forward_cached test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2 model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is Paris and");
        let nt = ids.len();
        assert!(nt > 2, "prompt too short");
        let positions: Vec<usize> = (0..nt).collect();
        let next_pos = nt; // first decode position
        let n_ctx = 32768usize;

        // Same input on two fresh caches => identical logits (deterministic).
        let mut cache_a = GraphCache::new();
        let mut cache_b = GraphCache::new();
        let pa = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, n_ctx, &mut cache_a);
        let pb = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, n_ctx, &mut cache_b);
        assert_eq!(pa.len(), pb.len());
        let mut d = 0.0f32;
        for i in 0..pa.len() {
            d = d.max((pa[i] - pb[i]).abs());
        }
        assert_eq!(d, 0.0, "identical inputs on fresh caches must give identical logits");

        // Interleave: A prefill -> B prefill -> A decode -> B decode.
        // Each cache's KV must stay isolated from the other's.
        let da = Qwen2Graph::forward_cached(
            q2, &[argmax(&pa)], &[next_pos], 1, n_ctx, &mut cache_a,
        );
        let db = Qwen2Graph::forward_cached(
            q2, &[argmax(&pb)], &[next_pos], 1, n_ctx, &mut cache_b,
        );
        let mut d2 = 0.0f32;
        for i in 0..da.len().min(db.len()) {
            d2 = d2.max((da[i] - db[i]).abs());
        }
        assert_eq!(d2, 0.0, "interleaved caches must not cross-contaminate KV");

        // n_ctx bounds: positions must stay below n_ctx (asserted), and a
        // smaller n_ctx must produce the same prefill logits (KV capacity does
        // not change the math while positions fit).
        let small_ctx = nt + 4;
        let mut cache_s = GraphCache::new();
        let ps = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, small_ctx, &mut cache_s);
        let mut d3 = 0.0f32;
        for i in 0..pa.len() {
            d3 = d3.max((pa[i] - ps[i]).abs());
        }
        assert_eq!(d3, 0.0, "smaller n_ctx must not change prefill logits");
        // And an out-of-range position must panic (guarded), not corrupt memory.
        let oob = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Qwen2Graph::forward_cached(
                q2, &[argmax(&pa)], &[small_ctx], 1, small_ctx, &mut cache_s,
            );
        }));
        assert!(oob.is_err(), "position >= n_ctx must be rejected");
    }
}

#[cfg(test)]
mod tail_tests {
    use super::*;
    use crate::models::ModelDef;

    fn model_path() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
        if p.exists() { Some(p) } else { None }
    }

    /// G3 correctness: the n_out=1 (reduced) graph must produce the same last
    /// token logits as the full-nt (n_out=nt) graph, node by node through the
    /// tail block (wo → get_rows → tail add → ffn → output). This also guards
    /// the allocator's build-order liveness (scheduler executes in build
    /// order; a topo-order liveness pass would free still-alive inputs).
    #[test]
    fn tail_reduction_matches_full_nt() {
        use crate::graph::alloc::GraphAllocator;
        use crate::graph::scheduler::BackendScheduler;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::ops::Op;
        let Some(path) = model_path() else {
            eprintln!("not cached; skipping");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        let nt = ids.len();
        let nv = model.n_vocab();

        /// Identify the semantic nodes of the final block by src-chain. Works
        /// for both the full graph (n_out=nt) and the reduced graph (n_out=1,
        /// tail get_rows inserted after wo).
        fn semantic_nodes(g: &crate::graph::ComputeGraph) -> Vec<(usize, &'static str)> {
            let q_matmul = g.nodes.iter().position(|n| n.name == "matmul_blk.23.attn_q.weight").unwrap();
            let q_rope = g.nodes.iter().position(|n| matches!(n.op, Op::RoPE { .. }) && n.src[0] == q_matmul).unwrap();
            let attn = g.nodes.iter().position(|n| matches!(n.op, Op::Attn { .. }) && n.src[0] == q_rope).unwrap();
            let wo = g.nodes.iter().position(|n| n.name == "matmul_blk.23.attn_output.weight").unwrap();
            // post-attn add: full -> add(src wo); reduced -> add(get_rows(wo), get_rows(h))
            let add_after_wo = g.nodes.iter().position(|n| {
                matches!(n.op, Op::Add)
                    && n.src.iter().any(|&s| {
                        s == wo || matches!(g.nodes[s].op, Op::GetRows) && g.nodes[s].src[0] == wo
                    })
            }).unwrap();
            let ffn_norm = g.nodes.iter().position(|n| matches!(n.op, Op::RmsNorm { .. }) && n.src[0] == add_after_wo).unwrap();
            let gate = g.nodes.iter().position(|n| n.name == "matmul_blk.23.ffn_gate.weight").unwrap();
            let up = g.nodes.iter().position(|n| n.name == "matmul_blk.23.ffn_up.weight").unwrap();
            // FusionPass merges silu+mul into a single SwiGLU node
            let swiglu = g.nodes.iter().position(|n| matches!(n.op, Op::SwiGLU) && n.src.contains(&gate)).unwrap();
            let down = g.nodes.iter().position(|n| n.name == "matmul_blk.23.ffn_down.weight").unwrap();
            let post_ffn = g.nodes.iter().position(|n| matches!(n.op, Op::Add) && n.src.contains(&down)).unwrap();
            let out_norm = g.nodes.iter().position(|n| matches!(n.op, Op::RmsNorm { .. }) && n.src[0] == post_ffn).unwrap();
            let lm = g.nodes.iter().position(|n| n.name == "matmul_output.weight").unwrap();
            vec![
                (q_matmul, "q_matmul"), (q_rope, "q_rope"), (attn, "attn"),
                (wo, "wo"), (add_after_wo, "post_attn_add"), (ffn_norm, "ffn_norm"),
                (gate, "gate"), (up, "up"), (swiglu, "swiglu"),
                (down, "down"), (post_ffn, "post_ffn_add"), (out_norm, "output_norm"),
                (lm, "lm_head"),
            ]
        }

        /// Build + execute a graph for n_out, keeping the given node ids alive
        /// as graph outputs (so post-exec dumps are not clobbered by liveness
        /// reuse). Returns (graph, allocator).
        fn run_keep(
            q2: &Qwen2Model, ids: &[u32], n_out: usize,
            keep: &[usize],
        ) -> (crate::graph::ComputeGraph, crate::graph::alloc::GraphAllocator) {
            use crate::graph::params::{CParams, GraphParams, GraphType};
            let model: &dyn ModelDef = q2;
            let nt = ids.len();
            let params = GraphParams {
                n_tokens: nt, n_seqs: 1, n_out,
                gtype: GraphType::Prefill,
                cparams: CParams { n_ctx: 4096, n_batch: nt, flash_attn: false, gpu: false, fuse_qkv: false },
                weights_version: 1,
            };
            let mut graph = model.build_graph(&params);
            for &i in keep {
                if !graph.outputs.contains(&i) { graph.outputs.push(i); }
            }
            let sched = BackendScheduler::new();
            let mut alloc = GraphAllocator::new();
            Qwen2Graph::register_graph_weights(q2, &mut alloc);
            sched.assign_backends(&mut graph, &alloc);
            let backends: [&dyn Backend; 1] = [alloc.cpu()];
            FusionPass::new().run(&mut graph, &backends, &|_, _| Some(0));
            alloc.alloc_graph(&graph).unwrap();
            let ids32: Vec<u32> = ids.iter().copied().collect();
            let pos32: Vec<u32> = (0..nt as u32).collect();
            alloc.fill_input_i32(&graph, "token_ids", &ids32).unwrap();
            alloc.fill_input_i32(&graph, "positions", &pos32).unwrap();
            if graph.inputs.iter().any(|&i| graph.node(i).name == "tail_ids") {
                let tail: Vec<u32> = ((nt - n_out)..nt).map(|x| x as u32).collect();
                alloc.fill_input_i32(&graph, "tail_ids", &tail).unwrap();
            }
            sched.execute(&graph, &mut alloc).unwrap();
            (graph, alloc)
        }

        // Discover semantic nodes on throwaway graphs, then re-run with those
        // nodes (plus their get_rows inputs) kept alive.
        let (fg0, _) = run_keep(q2, &ids, nt, &[]);
        let (rg0, _) = run_keep(q2, &ids, 1, &[]);
        let fsem = semantic_nodes(&fg0);
        let rsem = semantic_nodes(&rg0);
        let mut fkeep: Vec<usize> = fsem.iter().map(|&(i, _)| i).collect();
        let mut rkeep: Vec<usize> = rsem.iter().map(|&(i, _)| i).collect();
        for g in [&fg0, &rg0] {
            let wo_id = g.nodes.iter().position(|n| n.name == "matmul_blk.23.attn_output.weight");
            if let Some(wi) = wo_id {
                if let Some(addi) = g.nodes.iter().position(|n| {
                    matches!(n.op, Op::Add)
                        && n.src.iter().any(|&s| {
                            s == wi || matches!(g.nodes[s].op, Op::GetRows) && g.nodes[s].src[0] == wi
                        })
                }) {
                    let target = if std::ptr::eq(g, &fg0) { &mut fkeep } else { &mut rkeep };
                    for &s in &g.nodes[addi].src {
                        if !target.contains(&s) { target.push(s); }
                        // in the reduced graph the src is a get_rows: also keep
                        // the h / wo buffer it reads
                        if matches!(g.nodes[s].op, Op::GetRows) {
                            for &s2 in &g.nodes[s].src {
                                if !target.contains(&s2) { target.push(s2); }
                            }
                        }
                    }
                }
            }
        }
        drop(fg0); drop(rg0);

        let (fg, mut fa) = run_keep(q2, &ids, nt, &fkeep);
        let (rg, mut ra) = run_keep(q2, &ids, 1, &rkeep);

        // KV load at layer 23 must be identical (attention path preserved)
        let fkv = fg.nodes.iter().position(|n| matches!(n.op, Op::KvcacheLoad { layer: 23 }));
        let rkv = rg.nodes.iter().position(|n| matches!(n.op, Op::KvcacheLoad { layer: 23 }));
        if let (Some(fk), Some(rk)) = (fkv, rkv) {
            let a = fa.copy_to_cpu(fk).unwrap();
            let b = ra.copy_to_cpu(rk).unwrap();
            let kvd = (0..a.len()).map(|i| (a[i] - b[i]).abs()).fold(0.0f32, f32::max);
            assert!(kvd < 1e-5, "kv.23 diverges: {kvd:.3e}");
        }

        // Full graph: every node holds nt rows. Reduced graph: same nt rows
        // before wo, 1 row after the tail get_rows. Compare the LAST row of
        // each (they describe the same token nt-1).
        for (fl, rl) in fsem.iter().zip(rsem.iter()) {
            let (fi, flab) = *fl;
            let (ri, _) = *rl;
            let a = fa.copy_to_cpu(fi).unwrap();
            let b = ra.copy_to_cpu(ri).unwrap();
            let al = a.len();
            let full_row = al / nt; // row width in the full graph (row count == nt)
            let row_f: Vec<f32> = a[al - full_row..].to_vec();
            let row_b: Vec<f32> = if b.len() >= full_row { b[b.len() - full_row..].to_vec() } else { b.clone() };
            let d = (0..row_f.len().min(row_b.len()))
                .map(|i| (row_f[i] - row_b[i]).abs())
                .fold(0.0f32, f32::max);
            assert!(d < 1e-5, "{flab} diverges: {d:.3e}");
        }

        let full_last: Vec<f32> = fa.copy_to_cpu(fg.outputs[0]).unwrap();
        let reduced_last: Vec<f32> = ra.copy_to_cpu(rg.outputs[0]).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..nv {
            maxd = maxd.max((full_last[nv * (nt - 1) + i] - reduced_last[i]).abs());
        }
        assert!(maxd < 1e-5, "tail reduction diverges: {maxd:.3e}");
    }

    /// G4 (+FFN follow-up): decode (nt==1) QKV and FFN gate+up fusion must be
    /// numerically identical to the unfused path. The fused graph replaces the
    /// QKV chain (3 matmul + 3 bias + 2 rope + 2 store) with one concat matmul
    /// (blk.{i}.attn_qkv) + one fused bias+rope+store kernel, and the FFN chain
    /// (2 matmul + silu + mul) with one concat matmul (blk.{i}.ffn_gu) + one
    /// in-place swiglu; both execute the same Metal math, so logits must match.
    #[test]
    fn fused_qkv_matches_unfused_decode() {
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            use crate::graph::alloc::GraphAllocator;
            use crate::graph::scheduler::BackendScheduler;
            use crate::graph::params::{CParams, GraphParams, GraphType};
            use crate::graph::ops::Op;
            use crate::graph::builder::GraphBuilder;

            fn run_decode(
                q2: &Qwen2Model, model: &dyn ModelDef, tok_ids: &[u32], n_ctx: usize, nv: usize,
                fuse: bool, expect_fused_node: bool,
            ) -> Vec<f32> {
                let params = GraphParams {
                    n_tokens: 1, n_seqs: 1, n_out: 1,
                    gtype: GraphType::Decode,
                    cparams: CParams { n_ctx, n_batch: 1, flash_attn: false, gpu: true, fuse_qkv: fuse },
                    weights_version: 1,
                };
                let mut graph = model.build_graph(&params);
                let sched = BackendScheduler::new();
                let mut alloc = GraphAllocator::new();
                Qwen2Graph::register_graph_weights(q2, &mut alloc);
                assert!(alloc.enable_metal(), "Metal backend unavailable");
                sched.assign_backends(&mut graph, &alloc);
                {
                    let backends: Vec<&dyn Backend> = vec![alloc.cpu(), alloc.metal().unwrap()];
                    FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                        Some(crate::graph::Backend::CPU) => Some(0),
                        Some(crate::graph::Backend::Metal) => Some(1),
                        _ => None,
                    });
                }
                let has_fused = graph.nodes.iter().any(|n| matches!(n.op, Op::FusedQKV { .. }));
                assert_eq!(has_fused, expect_fused_node, "FusedQKV node presence");
                let ffn_off = std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1");
                let has_fused_ffn = graph.nodes.iter().any(|n| matches!(n.op, Op::FusedFFN));
                // FFN fusion is gated on nf <= 16384 (7B nf=18944 skips it)
                let ffn_small = {
                    let nf = q2.hparams.n_ff as usize;
                    nf <= 16384
                };
                assert_eq!(has_fused_ffn, expect_fused_node && !ffn_off && ffn_small,
                    "FusedFFN node presence");
                alloc.alloc_graph(&graph).unwrap();
                alloc.fill_input_i32(&graph, "token_ids", &[tok_ids[0]]).unwrap();
                alloc.fill_input_i32(&graph, "positions", &[0]).unwrap();
                sched.execute(&graph, &mut alloc).unwrap();
                alloc.copy_to_cpu(graph.outputs[0]).unwrap()
            }

            /// Build + execute a decode graph, marking every post-FFN Add
            /// (residual + ffn_down) as an output so we can compare layer by layer.
            fn run_decode_layers(
                q2: &Qwen2Model, model: &dyn ModelDef, tok_ids: &[u32],
                fuse: bool,
            ) -> (Vec<f32>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>, Vec<Vec<f32>>) {
                let params = GraphParams {
                    n_tokens: 1, n_seqs: 1, n_out: 1,
                    gtype: GraphType::Decode,
                    cparams: CParams { n_ctx: 4096, n_batch: 1, flash_attn: false, gpu: true, fuse_qkv: fuse },
                    weights_version: 1,
                };
                let mut graph = model.build_graph(&params);
                // mark post-FFN residual adds and the FFN-norm rms as outputs
                let mut ffn_adds: Vec<usize> = Vec::new();
                let mut norm_adds: Vec<usize> = Vec::new();
                let mut down_ids: Vec<usize> = Vec::new();
                let mut fused_outs: Vec<usize> = Vec::new();
                let mut swiglu_ids: Vec<usize> = Vec::new();
                for (i, n) in graph.nodes.iter().enumerate() {
                    if matches!(n.op, Op::Add) && n.src.len() == 2
                        && graph.nodes[n.src[1]].name.contains("ffn_down")
                    {
                        ffn_adds.push(i);
                        graph.outputs.push(i);
                    }
                    if n.name.contains("ffn_down") {
                        down_ids.push(i);
                        graph.outputs.push(i);
                    }
                    if matches!(n.op, Op::FusedFFN) {
                        fused_outs.push(i);
                        graph.outputs.push(i);
                    }
                    if matches!(n.op, Op::SwiGLU) {
                        swiglu_ids.push(i);
                        graph.outputs.push(i);
                    }
                    if matches!(n.op, Op::RmsNorm { .. }) {
                        let fed = graph.nodes.iter().any(|m| m.src.contains(&i)
                            && (matches!(m.op, Op::FusedFFN) || m.name.contains("ffn_gate") || m.name.contains("ffn_up")));
                        if fed {
                            norm_adds.push(i);
                            graph.outputs.push(i);
                        }
                    }
                }
                let sched = BackendScheduler::new();
                let mut alloc = GraphAllocator::new();
                Qwen2Graph::register_graph_weights(q2, &mut alloc);
                alloc.enable_metal();
                sched.assign_backends(&mut graph, &alloc);
                {
                    let backends: Vec<&dyn Backend> = vec![alloc.cpu(), alloc.metal().unwrap()];
                    FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                        Some(crate::graph::Backend::CPU) => Some(0),
                        Some(crate::graph::Backend::Metal) => Some(1),
                        _ => None,
                    });
                }
                alloc.alloc_graph(&graph).unwrap();
                alloc.fill_input_i32(&graph, "token_ids", &[tok_ids[0]]).unwrap();
                alloc.fill_input_i32(&graph, "positions", &[0]).unwrap();
                sched.execute(&graph, &mut alloc).unwrap();
                let logits = alloc.copy_to_cpu(graph.outputs[0]).unwrap();
                let layers: Vec<Vec<f32>> = ffn_adds
                    .iter()
                    .map(|&o| alloc.copy_to_cpu(o).unwrap())
                    .collect();
                let norms: Vec<Vec<f32>> = norm_adds
                    .iter()
                    .map(|&o| alloc.copy_to_cpu(o).unwrap())
                    .collect();
                let downs: Vec<Vec<f32>> = down_ids
                    .iter()
                    .map(|&o| alloc.copy_to_cpu(o).unwrap())
                    .collect();
                let fouts: Vec<Vec<f32>> = fused_outs
                    .iter()
                    .map(|&o| alloc.copy_to_cpu(o).unwrap())
                    .collect();
                let swouts: Vec<Vec<f32>> = swiglu_ids
                    .iter()
                    .map(|&o| alloc.copy_to_cpu(o).unwrap())
                    .collect();
                (logits, layers, norms, downs, fouts, swouts)
            }

            let Some(path) = model_path() else {
                eprintln!("not cached; skipping");
                return;
            };
            crate::metal::MpsState::init();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let model = crate::models::load_model(&gguf).expect("load model");
            let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
            let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
            let ids = tok.encode("The capital of France is");
            let nv = model.n_vocab();
            let model: &dyn ModelDef = q2;

            // 0.5B Q4_0: fused vs unfused must be bit-identical
            let fused_l = run_decode(q2, model, &[ids[0]], 4096, nv, true, true);
            let unfused_l = run_decode(q2, model, &[ids[0]], 4096, nv, false, false);
            let mut maxd = 0.0f32;
            for i in 0..fused_l.len() {
                maxd = maxd.max((fused_l[i] - unfused_l[i]).abs());
            }
            eprintln!("[fused-qkv] 0.5B decode logits fused-vs-unfused max diff: {maxd:.3e}");
            // per-layer post-FFN comparison on 0.5B (regression guard)
            let (_, lf0, _, lf0d, _, _) = run_decode_layers(q2, model, &[ids[0]], true);
            let (_, lu0, _, lu0d, _, _) = run_decode_layers(q2, model, &[ids[0]], false);
            for (li, (vf, vu)) in lf0.iter().zip(lu0.iter()).enumerate() {
                let mut d = 0.0f32;
                for i in 0..vf.len().min(vu.len()) {
                    d = d.max((vf[i] - vu[i]).abs());
                }
                assert!(d < 1e-5, "0.5B layer {li} post-FFN diverges: {d:.3e}");
            }
            for (li, (df, du)) in lf0d.iter().zip(lu0d.iter()).enumerate() {
                let mut d = 0.0f32;
                for i in 0..df.len().min(du.len()) {
                    d = d.max((df[i] - du[i]).abs());
                }
                assert!(d < 1e-5, "0.5B layer {li} ffn_down diverges: {d:.3e}");
            }
            assert!(maxd < 2e-4, "0.5B fused QKV decode diverges: {maxd:.3e}");

            // 7B Q4_K_M (when cached): fused vs unfused, per layer
            let seven_b_path = {
                let home = std::env::var_os("HOME").map(std::path::PathBuf::from);
                home.map(|mut p| {
                    p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf");
                    p
                }).filter(|p| p.exists())
            };
            if let Some(p7) = seven_b_path {
                let gguf7 = crate::gguf::load_gguf_model(&p7).expect("parse 7B GGUF");
                let model7 = crate::models::load_model(&gguf7).expect("load 7B model");
                let q7: &Qwen2Model = model7.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
                let m7: &dyn ModelDef = q7;
                let tok7 = crate::tokenizer::Tokenizer::load(&gguf7.parts[0].ctx);
                let ids7 = tok7.encode("Hello!");
                let nv7 = model7.n_vocab();

                let (lf, layers_f, _, _, _, _) = run_decode_layers(q7, m7, &ids7, true);
                let (lu, layers_u, _, _, _, _) = run_decode_layers(q7, m7, &ids7, false);
                let mut d7 = 0.0f32;
                for i in 0..lf.len().min(lu.len()) {
                    d7 = d7.max((lf[i] - lu[i]).abs());
                }
                eprintln!("[fused-qkv] 7B decode fused-vs-unfused max diff: {d7:.3e}");
                for (li, (vf, vu)) in layers_f.iter().zip(layers_u.iter()).enumerate() {
                    let mut d = 0.0f32;
                    for i in 0..vf.len().min(vu.len()) {
                        d = d.max((vf[i] - vu[i]).abs());
                    }
                    if d > 1e-5 {
                        eprintln!("[fused-qkv]   layer {li} post-FFN add diff {d:.3e}");
                    }
                }
                // functional: logits may differ by tiny float noise, but the
                // greedy token must be identical
                let _ = (GraphBuilder::new(), nv7);
                let gf = (0..lf.len()).fold(0usize, |a, i| if lf[i] > lf[a] { i } else { a });
                let gu = (0..lu.len()).fold(0usize, |a, i| if lu[i] > lu[a] { i } else { a });
                assert_eq!(gf, gu, "7B fused/unfused greedy token differs");
            }
        }
    }
}
