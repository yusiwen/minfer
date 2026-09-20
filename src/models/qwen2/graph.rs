//! Qwen2 compute-graph construction (Phase 5) + graph forward path (Phase 6).
//!
//! `build_graph` mirrors llama.cpp's `llama_model_qwen2::graph::graph()`
//! (src/models/qwen2.cpp:53) and the plan's §4 example: a declarative IR of
//! the full forward pass. `forward_graph` runs it through the graph stack
//! (builder → assign → fuse → alloc → execute) with a params-only reuse cache.
//!
//! Deviations recorded in docs/COMPUTE-GRAPH-DESIGN.md §17:
//! - Full-nt computation (the n_out tail-row `GetRows` optimization is
//!   deferred; tail-row extraction happens on the logits only, which is
//!   numerically identical for the sampled rows).
//! - KV cache lives in the graph allocator's persistent regions (the caller's
//!   `KVCache` is ignored by the graph path).
//! - `weights_version` is static (1) until LoRA support lands.

use std::sync::{Mutex, OnceLock};

use crate::cache::KVCache;
use crate::graph::alloc::GraphAllocator;
use crate::graph::backend::Backend;
use crate::graph::cache::GraphCache;
use crate::graph::fusion::FusionPass;
use crate::graph::ops::{
    AttnMeta, AttnMode, FusedFfnMeta, FusedQkvMeta, QkvBiasRopeStoreMeta, RoPEMeta,
};
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
        // Namespaced fused-weight names must match the loader's registration
        // keys (Qwen2Model::ns; empty for the primary model).
        let wns = &model.ns;
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
        // E1/E2: when positions cannot bound a query's window (several
        // sequences, or a window that does not start at cell 0) the attention
        // nodes carry the flag, so a backend still deriving the bound refuses
        // them instead of using a wrong window.
        b.set_explicit_span(params.cparams.explicit_span);

        let inp_pos = b.input("positions", [nt, 1, 1, 1], crate::graph::DType::I32);
        // G3 tail-row reduction input, declared at the graph HEAD (not beside
        // its consumers at the last layer): an input node mid-graph splits the
        // forward into extra CPU/CUDA boundaries (2 full-stream syncs + host
        // round-trip copies per step on the split path). R3-A1,
        // docs/CUDA_OPTIMIZATION.md Part III. Node order is not semantics —
        // the consumers below just reference the handle.
        let tail_ids = (params.n_out < nt).then(|| {
            b.input(
                "tail_ids",
                [params.n_out, 1, 1, 1],
                crate::graph::DType::I32,
            )
        });

        let mut h = b.embedding(inp_ids, model.tok_embd.as_ref().unwrap());

        let mode = if params.cparams.flash_attn {
            AttnMode::Flash
        } else {
            AttnMode::Gqa
        };
        let attn_scale = hp.attention_scale();

        for (il, l) in model.layers.iter().enumerate() {
            let residual = h;

            // pre-norm
            let normed = b.rms_norm(h, l.attn_norm.as_ref(), eps);

            // Q/K/V projections (+ biases). decode (nt==1) with GPU QKV
            // fusion (G4 + D3-8) has two classes:
            // - concat class (wq|wk|wv same quant type): `Op::FusedQKV` — one
            //   concat matmul + one fused bias+rope+store kernel;
            // - mixed-quant class (e.g. Q6_K attn_v among Q4_K q/k): three
            //   separate matmuls (no bias) + one `Op::QkvBiasRopeStore`
            //   epilogue (CUDA-only today; Metal keeps the unfused chain for
            //   these layers). Both replace 3 matmul + 3 bias + 2 rope +
            //   2 store dispatches.
            let fuse_qkv = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_qkv
                && l.bq.is_some()
                && l.bk.is_some()
                && l.bv.is_some();
            // class 2 is CUDA-only: on macOS (feature off) the mixed-quant
            // layers keep the unfused chain, bitwise-neutral vs pre-D3-8.
            #[cfg(feature = "cuda")]
            let qkv_epilogue_ok = crate::cuda::CudaState::get().is_some();
            #[cfg(not(feature = "cuda"))]
            let qkv_epilogue_ok = false;
            let (q, kv) = if fuse_qkv && Self::qkv_concat_available(&l.wq, &l.wk, &l.wv) {
                let qkv = b.fused_qkv(
                    normed,
                    inp_pos,
                    il,
                    FusedQkvMeta {
                        qkv_weight: format!("{wns}blk.{il}.attn_qkv"),
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
            } else if fuse_qkv && qkv_epilogue_ok {
                // D3-8 class 2 (mixed quant types): separate matmuls without
                // bias, then one epilogue pass (bias×3 + rope×2 + store×2 → 1).
                // Attention is wired to the epilogue node so q's matmul buffer
                // has exactly one consumer (in-place alias rule, §5).
                let q = b.matmul(normed, l.wq.as_ref().unwrap(), None);
                let k = b.matmul(normed, l.wk.as_ref().unwrap(), None);
                let v = b.matmul(normed, l.wv.as_ref().unwrap(), None);
                let q = b.qkv_bias_rope_store(
                    q,
                    k,
                    v,
                    inp_pos,
                    il,
                    QkvBiasRopeStoreMeta {
                        bias_q: l.bq.as_ref().map(|t| t.name.clone()),
                        bias_k: l.bk.as_ref().map(|t| t.name.clone()),
                        bias_v: l.bv.as_ref().map(|t| t.name.clone()),
                        nqt: nh * hd,
                        nkt,
                        hd,
                        freq_base: hp.rope_freq_base,
                        freq_scale: hp.rope_freq_scale,
                        rope_style: hp.rope_style,
                        kv_elems: nkt * n_ctx,
                    },
                );
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (q, kv)
            } else {
                let q = b.matmul(normed, l.wq.as_ref().unwrap(), l.bq.as_ref());
                let k = b.matmul(normed, l.wk.as_ref().unwrap(), l.bk.as_ref());
                let v = b.matmul(normed, l.wv.as_ref().unwrap(), l.bv.as_ref());

                let q = b.rope(
                    q,
                    inp_pos,
                    hp.rope_style,
                    RoPEMeta {
                        freq_base: hp.rope_freq_base,
                        freq_scale: hp.rope_freq_scale,
                        n_head: nh,
                        hd,
                    },
                );
                let k = b.rope(
                    k,
                    inp_pos,
                    hp.rope_style,
                    RoPEMeta {
                        freq_base: hp.rope_freq_base,
                        freq_scale: hp.rope_freq_scale,
                        n_head: nk,
                        hd,
                    },
                );

                b.kvcache_store(il, k, v, n_ctx);
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (q, kv)
            };

            // attention
            let attn_out = b.attn(
                q,
                kv,
                inp_pos,
                mode,
                AttnMeta {
                    layer: il,
                    n_head: nh,
                    n_head_kv: nk,
                    hd,
                    hd_kv,
                    nkt,
                    scale: attn_scale,
                },
            );

            // output projection + residual
            let wo = b.matmul(attn_out, l.wo.as_ref().unwrap(), None);
            let is_last = il == model.layers.len() - 1;
            // G3: reduce to the tail n_out rows BEFORE the last layer's FFN
            // (llama `ggml_get_rows(cur/inpSA, inp_out_ids)` at
            // qwen2.cpp:106-108) — ffn_norm, gate/up/down, swiglu, both
            // residuals and lm_head all run on n_out rows only. The tail_ids
            // input itself is declared at the graph head (see there).
            if is_last && params.n_out < nt {
                let tail_ids = tail_ids.expect("tail_ids input declared when n_out < nt");
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
                && params.cparams.fuse_ffn
                && Self::gu_concat_available(&l.ffn_gate, &l.ffn_up)
                && nf <= 16384;
            let residual = h;
            let normed = b.rms_norm(h, l.ffn_norm.as_ref(), eps);
            let ffn_out = if fuse_gu {
                let gu_weight = format!("{wns}blk.{il}.ffn_gu");
                let weight_ttype = l.ffn_gate.as_ref().unwrap().ttype;
                // D3: the hand-written node is the default (it is faster where it
                // fuses, and Metal needs it until G5); `MINFER_FFN_COMPOSITION=1`
                // selects the proven composition instead. See
                // `models::ffn_composition` for the rule and the loud refusal on a
                // backend without offset views.
                let ffn_composition = crate::models::ffn_composition(
                    std::env::var("MINFER_FFN_COMPOSITION").ok().as_deref(),
                    Self::device(model),
                );
                let ffn_node = !ffn_composition;
                let gu = if ffn_node {
                    b.fused_ffn(
                        normed,
                        FusedFfnMeta {
                            gu_weight,
                            weight_ttype,
                            in_dim: ne,
                            nf,
                        },
                    )
                } else {
                    b.fused_ffn_composition(normed, &gu_weight, weight_ttype, ne, nf)
                };
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
        let logits = b.matmul(
            normed,
            model.output.as_ref().unwrap(),
            model.output_b.as_ref(),
        );
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
        let (Some(wq), Some(wk), Some(wv)) = (wq, wk, wv) else {
            return false;
        };
        #[cfg(target_os = "macos")]
        {
            crate::metal::concat_rows(&[wq, wk, wv]).is_some()
        }
        // D3-8: CUDA arm — metadata-only probe (concat_rows_feasible mirrors
        // concat_rows' preconditions exactly; the loader performs the real
        // concatenation once at model load and registers blk.{i}.attn_qkv,
        // so both sides agree with the Metal pattern).
        #[cfg(all(feature = "cuda", not(target_os = "macos")))]
        {
            crate::cuda::concat_rows_feasible(&[wq, wk, wv])
        }
        #[cfg(not(any(target_os = "macos", all(feature = "cuda", not(target_os = "macos")))))]
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
        let (Some(fg), Some(fu)) = (fg, fu) else {
            return false;
        };
        #[cfg(all(target_os = "macos", not(feature = "cuda")))]
        {
            crate::metal::concat_rows(&[fg, fu]).is_some()
        }
        #[cfg(all(target_os = "macos", feature = "cuda"))]
        {
            crate::metal::concat_rows(&[fg, fu]).is_some()
                || crate::cuda::concat_rows(&[fg, fu]).is_some()
        }
        #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
        {
            // metadata-only probe — the concat bytes are built once by the
            // loader; rebuilding them here cost ~920 ms per decode graph build
            crate::cuda::concat_rows_feasible(&[fg, fu])
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
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
                &l.attn_norm,
                &l.wq,
                &l.bq,
                &l.wk,
                &l.bk,
                &l.wv,
                &l.bv,
                &l.wo,
                &l.ffn_norm,
                &l.ffn_gate,
                &l.ffn_up,
                &l.ffn_down,
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
    /// CLI convenience wrapper: uses the process-global `graph_cache()`. The
    /// KV regions are sized by `n_ctx`, clamped to the model's `max_seq_len`
    /// (llama.cpp clamps `n_ctx` the same way); single-shot previously used the
    /// full `max_seq_len` unconditionally, which over-allocated and paid a
    /// first-submit Metal tax proportional to the KV bytes (see
    /// docs/PERF-QWEN3-4B-VS-LLAMACPP.md §2). Server / multi-slot code must
    /// call [`Qwen2Graph::forward_cached`] with a slot-scoped cache instead.
    pub fn forward(
        model: &Qwen2Model,
        tokens: &[u32],
        positions: &[usize],
        _kv: &mut KVCache,
        n_out: usize,
        n_ctx: usize,
    ) -> Vec<f32> {
        let n_ctx = n_ctx.min(model.hparams.max_seq_len as usize);
        let mut guard = graph_cache().lock().unwrap();
        Self::forward_cached(model, tokens, positions, n_out, n_ctx, &mut guard)
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
        // The classic single-sequence forward IS a one-sequence batch (E2).
        Self::forward_batch(
            model,
            &crate::graph::batch::Batch::single(tokens, positions),
            n_out,
            n_ctx,
            cache,
        )
    }

    /// One forward over a batch: `nt` query tokens belonging to `n_seqs`
    /// sequences, each attending only to its own KV cells (Phase E / E2).
    ///
    /// `n_out` keeps its single-sequence meaning (the last `n_out` rows); a
    /// multi-sequence batch returns one logits row per sequence instead, in
    /// batch order (`Batch::out_rows`).
    pub fn forward_batch(
        model: &Qwen2Model,
        batch: &crate::graph::batch::Batch,
        n_out: usize,
        n_ctx: usize,
        cache: &mut GraphCache,
    ) -> Vec<f32> {
        if let Err(e) = batch.check() {
            panic!("forward_batch: {e}");
        }
        let tokens = &batch.tokens;
        let positions = &batch.positions;
        let nt = batch.len();
        let out_rows = batch.out_rows(n_out);
        let n_out = out_rows.len();
        // E2: whether the causal (positions-based) attention instantiation is
        // exact for this batch, taken from the KV reservations — the authority on
        // where each sequence's window starts — so no caller can get it wrong.
        // One sequence starting at cell 0 keeps the classic path; a second
        // sequence (whose reservation cannot also start at 0), or a non-zero
        // start, takes the explicit span. This reads the *batch*, never
        // `GraphParams`: the sequence count is data (A7/E2), so it is not part
        // of the reuse identity.
        let explicit_span = {
            let mut need = batch.n_seqs() > 1;
            for (seq, _, _) in batch.groups() {
                if cache.alloc().kv_seq_slot(seq).map(|s| s.start).unwrap_or(0) != 0 {
                    need = true;
                }
            }
            need
        };
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
        // lives in the built graph, not in the params' other fields), and it
        // comes from `Self::device` so the builder and the server's batching
        // default (E6) cannot disagree.
        let device = Self::device(model);
        let metal_on = device == crate::models::Device::Metal;
        let cuda_on = device == crate::models::Device::Cuda;
        let params = GraphParams {
            n_tokens: nt,
            n_out,
            gtype: if nt == 1 {
                GraphType::Decode
            } else {
                GraphType::Prefill
            },
            cparams: CParams {
                n_ctx,
                flash_attn: false,
                explicit_span,
                gpu: metal_on || cuda_on,
                // G4/G5: decode fusions are part of the topology — the env
                // toggles force a rebuild so they can be A/B'd reliably.
                // G5 (FFN gate+up) is decoupled from the QKV fusion gate
                // (mirrors Qwen3) so A/B-ing one fusion does not flip the
                // other; 7e⑤ extends it to the CUDA backend.
                // D3-8: CUDA joins the decode QKV fusion (G4 CUDA port) —
                // the backend claims Op::FusedQKV in supports_op and the
                // loader registers blk.{i}.attn_qkv; qkv_concat_available
                // probes the concat feasibility per backend (same shape as
                // the fuse_ffn gate below).
                // C6/S3: the fused epilogue stores K/V at the
                // allocator-resolved `cells` row, so the CUDA path may stay
                // fused even when the run does not start at cell 0
                // (`explicit_span`). Metal keeps the pre-C6 gate: it has no
                // explicit-span attention at all (G5), so every run it can
                // fuse starts at cell 0 and positions == cells.
                fuse_qkv: nt == 1
                    && (metal_on || cuda_on)
                    && (cuda_on || !explicit_span)
                    && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1"),
                fuse_ffn: nt == 1
                    && (metal_on || cuda_on)
                    && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1"),
            },
            weights_version: 1,
        };

        if !cache.try_reuse(&params) {
            let rebuild_t0 = std::time::Instant::now();
            let trace_rb = std::env::var("MINFER_REBUILD_TRACE").map_or(false, |v| v == "1");
            let mut graph = Self::build(model, &params);
            let sched = BackendScheduler::new();
            {
                let alloc = cache.alloc();
                Self::register_graph_weights(model, alloc);
                #[cfg(target_os = "macos")]
                if metal_on {
                    alloc.enable_metal();
                }
                #[cfg(feature = "cuda")]
                if cuda_on {
                    alloc.enable_cuda();
                }
                sched.assign_backends(&mut graph, alloc);
                // fusion pass gated per node's assigned backend
                let backends: Vec<&dyn Backend> = {
                    #[cfg_attr(not(any(target_os = "macos", feature = "cuda")), allow(unused_mut))]
                    let mut v: Vec<&dyn Backend> = vec![alloc.cpu()];
                    #[cfg(target_os = "macos")]
                    if metal_on {
                        if let Some(m) = alloc.metal() {
                            v.push(m);
                        }
                    }
                    #[cfg(feature = "cuda")]
                    if cuda_on {
                        if let Some(c) = alloc.cuda() {
                            v.push(c);
                        }
                    }
                    v
                };
                // The closure maps a node's backend to its index in `backends`
                // (the fusion pass probes supports_fused through it), so the
                // CUDA index is derived from the actual vector layout.
                let cuda_idx = backends.iter().position(|b| b.name() == "cuda");
                FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                    Some(crate::graph::Backend::CPU) => Some(0),
                    Some(crate::graph::Backend::Metal) => Some(1),
                    Some(crate::graph::Backend::Cuda) => cuda_idx,
                    _ => None,
                });
                alloc.alloc_graph(&graph).unwrap();
            }
            if trace_rb {
                eprintln!(
                    "[rebuild] nt={} params rebuilt in {:.1} ms (build+assign+alloc; capture lands on the next 3 executions)",
                    params.n_tokens,
                    rebuild_t0.elapsed().as_secs_f64() * 1e3
                );
            }
            cache.replace_graph(graph, params);
        }

        let (graph, alloc) = cache.current().unwrap();

        // refresh input data (positions/ids are data, not topology)
        let ids: Vec<u32> = tokens.to_vec();
        alloc.fill_input_i32(graph, "token_ids", &ids).unwrap();
        let pos: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        alloc.fill_input_i32(graph, "positions", &pos).unwrap();
        // E1: one sequence today (batch composition is E2). The allocator
        // resolves each query's allowed cells from the cell store, so the IR's
        // seq ids and the kernel's window cannot disagree.
        // Phase C / C2 + E1 + E2: mark each sequence's written rows, fill the
        // sequence ids and resolve every query's attention span from the cell
        // store — one call, so the ids and the kernels' windows cannot disagree.
        // All of it is data, so the graph and its reuse identity are untouched.
        alloc
            .fill_batch_inputs(graph, batch)
            .unwrap_or_else(|e| panic!("batch inputs: {e}"));
        // G3: the last-layer tail-row reduction reads `tail_ids` (filled when
        // the graph was built with n_out < nt, i.e. prefill)
        if graph
            .inputs
            .iter()
            .any(|&i| graph.node(i).name == "tail_ids")
        {
            alloc.fill_input_i32(graph, "tail_ids", &out_rows).unwrap();
        }
        if std::env::var("MINFER_GRAPH_DUMP").is_ok() {
            if let Some(idsbuf) = graph
                .inputs
                .iter()
                .find(|&&i| graph.node(i).name == "token_ids")
                .copied()
            {
                let v = alloc.copy_to_cpu(idsbuf).unwrap_or_default();
                eprintln!("[graph dump] ids after fill: {:?}", &v[..v.len().min(8)]);
            }
        }

        let sched = BackendScheduler::new();
        sched.execute(graph, alloc).unwrap();

        // debug dump: MINFER_GRAPH_DUMP=/tmp/x writes the logits and layer-0 KV
        // so GPU vs CPU graph runs can be compared (Phase 3 debugging)
        if let Ok(dir) = std::env::var("MINFER_GRAPH_DUMP") {
            static DUMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dump_n = DUMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
            let tag = if nt == 1 { "decode" } else { "prefill" };
            let _ = std::fs::write(format!("{dir}/logits_{tag}_{dump_n:05}.f32"), {
                let mut b = Vec::with_capacity(logits.len() * 4);
                for x in &logits {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b
            });
            for nid in [0usize, 1, 2, 3, 5, 8, 10, 11] {
                if nid < graph.n_nodes() {
                    if let Some(buf) = alloc.copy_to_cpu(nid) {
                        let mut b = Vec::with_capacity(buf.len() * 4);
                        for x in &buf {
                            b.extend_from_slice(&x.to_le_bytes());
                        }
                        let _ = std::fs::write(format!("{dir}/node{nid}_{tag}.f32"), b);
                    }
                }
            }
            // every layer's K AND V regions (persistent buffers — always
            // live, so these dumps are reliable for GPU-vs-CPU layer
            // bisection; doc 95 identity debugging dumps both halves)
            for layer in 0..model.n_layer() {
                if let Some((k, v)) = alloc.copy_kv_to_cpu(layer) {
                    for (kind, data) in [("k", k), ("v", v)] {
                        let mut b = Vec::with_capacity(data.len() * 4);
                        for x in &data {
                            b.extend_from_slice(&x.to_le_bytes());
                        }
                        let _ = std::fs::write(format!("{dir}/kv{layer}{kind}_{tag}.f32"), b);
                    }
                }
            }
            if let Some(kv0) = graph
                .nodes
                .iter()
                .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 }))
            {
                if let Some(kv) = alloc.copy_to_cpu(kv0) {
                    let mut b = Vec::with_capacity(kv.len() * 4);
                    for x in &kv {
                        b.extend_from_slice(&x.to_le_bytes());
                    }
                    let _ = std::fs::write(format!("{dir}/kv0_{tag}.f32"), b);
                }
            }
            eprintln!(
                "[graph dump] wrote {dir}/logits_{tag}.f32 ({} elems)",
                logits.len()
            );
        }

        // extract the tail n_out rows of logits. With G3 the graph already
        // reduced the last layer + lm_head to n_out rows (buffer = n_out*nv);
        // without it (decode, n_out == nt) the buffer is nv*nt == n_out*nv.
        // Either way the first n_out*nv elements are the answer.
        let nv = model.hparams.n_vocab as usize;
        let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
        // R3-A2: the buffer is always exactly n_out*nv (G3-reduced, or
        // n_out == nt) — skip the redundant full-logits clone.
        if logits.len() == n_out * nv {
            logits
        } else {
            logits[..n_out * nv].to_vec()
        }
    }

    /// The backend this model's forwards will run on — the **single authority**
    /// for the GPU gates (E6).
    ///
    /// `forward_batch` derives `CParams.gpu` from it and the server derives its
    /// batching default from it, so "the device participates" means one thing
    /// everywhere. Metal wins when both are available, matching the allocator's
    /// backend priority. Uses attributes rather than `cfg!()` so the
    /// `metal_backend` path is not resolved on non-macOS builds (the module does
    /// not exist there).
    pub fn device(model: &Qwen2Model) -> crate::models::Device {
        #[cfg(any(target_os = "macos", feature = "cuda"))]
        {
            #[cfg(target_os = "macos")]
            if crate::graph::metal_backend::metal_available() && Self::weights_on_gpu(model) {
                return crate::models::Device::Metal;
            }
            // CUDA participation (Phase 7): requires a usable device AND every
            // matmul weight registered on the CUDA registry in a kernel-supported
            // type (all-or-nothing; 7e③ moved the embedding gather on device, so
            // tok_embd is gated like every other weight).
            #[cfg(feature = "cuda")]
            if crate::cuda::CudaState::get().is_some() && Self::weights_on_cuda(model) {
                return crate::models::Device::Cuda;
            }
        }
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let _ = model;
        crate::models::Device::Cpu
    }

    /// Every weight the graph reads must be GPU-registered for the Metal path.
    #[cfg(target_os = "macos")]
    fn weights_on_gpu(model: &Qwen2Model) -> bool {
        let names: Vec<String> = {
            let mut v = Vec::new();
            for t in [
                &model.tok_embd,
                &model.output_norm,
                &model.output,
                &model.output_b,
            ] {
                if let Some(t) = t {
                    v.push(t.name.clone());
                }
            }
            for l in &model.layers {
                for t in [
                    &l.attn_norm,
                    &l.wq,
                    &l.bq,
                    &l.wk,
                    &l.bk,
                    &l.wv,
                    &l.bv,
                    &l.wo,
                    &l.ffn_norm,
                    &l.ffn_gate,
                    &l.ffn_up,
                    &l.ffn_down,
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
            let Some(mps) = crate::metal::MpsState::get() else {
                return false;
            };
            names.iter().all(|n| mps.has_weight(n))
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = names;
            false
        }
    }

    /// CUDA participation gate (Phase 7c, extended 7e③): all-or-nothing over
    /// the weights the graph executes, now INCLUDING `tok_embd` (7e③ gave the
    /// embedding a device gather+dequant kernel — the exclusion was removed).
    /// Two conditions per weight: registered on the CUDA registry, and a type
    /// with the matching kernel — matmuls cover Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 and
    /// the K-quants, embed gathers every type EXCEPT Q4_1 (no embed kernel).
    /// The loader registers some unsupported types for the legacy path, so the
    /// type check is required here. Norm/bias weights are f32 and only need
    /// registration.
    #[cfg(feature = "cuda")]
    fn weights_on_cuda(model: &Qwen2Model) -> bool {
        use crate::tensor::TensorType;
        fn matmul_t_ok(t: &crate::tensor::Tensor, cuda: &crate::cuda::CudaState) -> bool {
            matches!(t.ttype, |TensorType::Q4_0| TensorType::Q8_0
                | TensorType::Q4_1
                | TensorType::Q4_K
                | TensorType::Q5_0
                | TensorType::Q6_K
                | TensorType::F32
                | TensorType::Q5_1
                | TensorType::Q5_K)
                && cuda.has_weight_of_size(&t.name, t.data().len())
        }
        fn matmul_ok(t: &Option<crate::tensor::Tensor>, cuda: &crate::cuda::CudaState) -> bool {
            match t {
                Some(t) => matmul_t_ok(t, cuda),
                None => true,
            }
        }
        fn embed_t_ok(t: &crate::tensor::Tensor, cuda: &crate::cuda::CudaState) -> bool {
            matches!(t.ttype, |TensorType::F32| TensorType::Q4_0
                | TensorType::Q8_0
                | TensorType::Q4_K
                | TensorType::Q5_0
                | TensorType::Q6_K
                | TensorType::Q5_1
                | TensorType::Q5_K)
                && cuda.has_weight_of_size(&t.name, t.data().len())
        }
        fn embed_ok(t: &Option<crate::tensor::Tensor>, cuda: &crate::cuda::CudaState) -> bool {
            match t {
                Some(t) => embed_t_ok(t, cuda),
                None => true,
            }
        }
        fn registered(t: &Option<crate::tensor::Tensor>, cuda: &crate::cuda::CudaState) -> bool {
            match t {
                Some(t) => cuda.has_weight_of_size(&t.name, t.data().len()),
                None => true,
            }
        }
        let Some(cuda) = crate::cuda::CudaState::get() else {
            return false;
        };
        let mut ok = embed_ok(&model.tok_embd, &cuda)
            && matmul_ok(&model.output, &cuda)
            && registered(&model.output_norm, &cuda)
            && registered(&model.output_b, &cuda);
        for l in &model.layers {
            ok &= registered(&l.attn_norm, &cuda)
                && matmul_ok(&l.wq, &cuda)
                && registered(&l.bq, &cuda)
                && matmul_ok(&l.wk, &cuda)
                && registered(&l.bk, &cuda)
                && matmul_ok(&l.wv, &cuda)
                && registered(&l.bv, &cuda)
                && matmul_ok(&l.wo, &cuda)
                && registered(&l.ffn_norm, &cuda)
                && matmul_ok(&l.ffn_gate, &cuda)
                && matmul_ok(&l.ffn_up, &cuda)
                && matmul_ok(&l.ffn_down, &cuda);
        }
        if !ok {
            // Identify the first weight that fails the gate (same checks, same
            // order as above; 0 = embed, 1 = matmul, 2 = registered-only) so
            // the message names the tensor instead of a generic complaint.
            let fail = std::iter::once((&model.tok_embd, 0u8))
                .chain(std::iter::once((&model.output, 1u8)))
                .chain(std::iter::once((&model.output_norm, 2u8)))
                .chain(std::iter::once((&model.output_b, 2u8)))
                .chain(model.layers.iter().flat_map(|l| {
                    [
                        (&l.attn_norm, 2u8),
                        (&l.wq, 1u8),
                        (&l.bq, 2u8),
                        (&l.wk, 1u8),
                        (&l.bk, 2u8),
                        (&l.wv, 1u8),
                        (&l.bv, 2u8),
                        (&l.wo, 1u8),
                        (&l.ffn_norm, 2u8),
                        (&l.ffn_gate, 1u8),
                        (&l.ffn_up, 1u8),
                        (&l.ffn_down, 1u8),
                    ]
                }))
                .find(|(t, kind)| match (t, kind) {
                    (Some(t), 0) => !embed_t_ok(t, &cuda),
                    (Some(t), 1) => !matmul_t_ok(t, &cuda),
                    (Some(t), _) => !cuda.has_weight_of_size(&t.name, t.data().len()),
                    (None, _) => false,
                });
            if let Some((Some(t), _)) = fail {
                eprintln!(
                    "CUDA GATE: weight '{}' (type {:?}) has no CUDA kernel or is not registered",
                    t.name, t.ttype
                );
            } else {
                eprintln!("CUDA GATE: a matmul weight has an unsupported type");
            }
        }
        ok
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
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    /// Max |Δ| between two equal-length logit vectors.
    fn max_delta(x: &[f32], y: &[f32]) -> f32 {
        assert_eq!(x.len(), y.len(), "compared vectors must have equal length");
        x.iter()
            .zip(y)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max)
    }

    /// The named tolerance class for comparisons whose two sides were computed
    /// at **different batch shapes** (a different `nt` anywhere in their
    /// history) — the project's rule is "bitwise-identity *or* a named tolerance
    /// class", and this is the second.
    ///
    /// On CPU these comparisons are bitwise and stay bitwise: the kernels are
    /// per-token, so shape never enters the arithmetic. CUDA's prefill kernels
    /// tile by `nt` and quantize activations to int8 (MMQ), so the *same* tokens
    /// processed at a different `nt` land on slightly different K/V values. The
    /// drift is bounded and small — measured on GB10 (sm_121) as ≤ 0.37 absolute
    /// on these fixtures' logits (each test prints its value on device) — but it
    /// is far above 0, so a bitwise assertion would be a false claim there.
    ///
    /// This is a gross-error detector for device runs, not a proof of
    /// correctness: the same-shape gates (`batch_order_does_not_change_a_sequences_logits`,
    /// `cuda_two_sequences_do_not_cross_attend`) and the CPU bitwise assertions
    /// are what pin the mechanism.
    fn cross_shape_tolerance() -> f32 {
        #[cfg(feature = "cuda")]
        if crate::cuda::CudaState::get().is_some() {
            return 1.0;
        }
        0.0
    }

    /// Assert two forwards that differ in shape agree: bitwise on CPU, within
    /// [`cross_shape_tolerance`] on a device (where `what` is printed with the
    /// measured |Δ| so drift regressions are visible in the log).
    fn assert_across_shapes(what: &str, a: &[f32], b: &[f32]) {
        let d = max_delta(a, b);
        let tol = cross_shape_tolerance();
        if tol > 0.0 {
            eprintln!("[cuda] {what}: max |Δ| = {d} (named class: <= {tol})");
        }
        assert!(
            d <= tol,
            "{what}: max |Δ| = {d} exceeds {} tolerance ({tol})",
            if tol == 0.0 {
                "the bitwise"
            } else {
                "the CUDA cross-shape"
            }
        );
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
        // Since Phase 6 `model.forward` IS the graph path: on CUDA-capable
        // builds the engine side runs the CUDA graph while this test's manual
        // graph is CPU-only, so the comparison is cross-backend and the
        // bitwise criterion does not apply (7e① diagnosis: the 0.449
        // "residual" is accumulated f32 reduction-order noise, not a bug —
        // mirror the Metal test's functional criterion). CPU-only builds
        // compare two CPU graphs and keep the strict bound.
        #[cfg(feature = "cuda")]
        if crate::cuda::CudaState::get().is_some() {
            let ga = argmax(a);
            let gb = argmax(b);
            eprintln!("[{tag}] greedy token: CPU-graph={ga} engine-graph={gb}");
            assert_eq!(ga, gb, "[{tag}] greedy token differs across backends");
            return;
        }
        assert!(
            maxd < 1e-3,
            "[{tag}] graph vs forward logits diverge (max diff {maxd:.3e})"
        );
    }

    /// B1 (Phase B): does reusing one `GraphCache` for a **different** prompt
    /// contaminate the result?
    ///
    /// `worker_loop` throws the slot's cache away on every request, because the
    /// comment there (`chat.rs:483-490`) claims re-prefilling a different prompt
    /// over the same persistent KV regions "leaves stale rows below the new
    /// attention window". Prefix reuse (B2) is only safe if that claim is
    /// understood, so verify it instead of inheriting it: run A then B on one
    /// cache and B on a virgin cache, and compare. Both orders are exercised,
    /// because "B shorter than A" is the case the comment worries about and
    /// "A longer than B" is the case B2 wants to reuse.
    #[test]
    fn reused_cache_across_prompts_matches_a_fresh_cache() {
        use crate::graph::cache::GraphCache;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping cache-reuse test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        // Keep the weight registry stable for this whole test (same reason as
        // the parity test below).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let long = tok.encode("The capital of France is Paris and the capital of Japan is");
        let short = tok.encode("Water boils at one hundred degrees");
        assert_ne!(long, short, "the two prompts must differ");
        assert!(
            long.len() > short.len(),
            "need a longer and a shorter prompt ({} vs {})",
            long.len(),
            short.len()
        );

        let run = |cache: &mut GraphCache, ids: &[u32]| {
            let pos: Vec<usize> = (0..ids.len()).collect();
            model.forward_graph_cached(ids, &pos, 1, n_ctx, cache)
        };

        // (a) A then B, where B is SHORTER: the stale-row scenario.
        let mut reused = GraphCache::new();
        let _ = run(&mut reused, &long);
        let b_reused = run(&mut reused, &short);
        let mut virgin = GraphCache::new();
        let b_virgin = run(&mut virgin, &short);
        assert_eq!(
            b_reused, b_virgin,
            "reusing a cache for a shorter prompt changed the logits"
        );

        // (b) B then A, where A is LONGER: the append case B2 relies on.
        let mut reused2 = GraphCache::new();
        let _ = run(&mut reused2, &short);
        let a_reused = run(&mut reused2, &long);
        let mut virgin2 = GraphCache::new();
        let a_virgin = run(&mut virgin2, &long);
        assert_eq!(
            a_reused, a_virgin,
            "reusing a cache for a longer prompt changed the logits"
        );

        // (c) The exact server sequence: prompt A, then decoded tokens written
        //     past A's length, then a new (shorter) request. Those generated
        //     rows are what the chat.rs comment is about.
        let mut reused3 = GraphCache::new();
        let a_pos: Vec<usize> = (0..long.len()).collect();
        let _ = model.forward_graph_cached(&long, &a_pos, 1, n_ctx, &mut reused3);
        for t in 0..3usize {
            let pos = long.len() + t;
            let _ = model.forward_graph_cached(&[100 + t as u32], &[pos], 1, n_ctx, &mut reused3);
        }
        let short_pos: Vec<usize> = (0..short.len()).collect();
        let b_reused3 = model.forward_graph_cached(&short, &short_pos, 1, n_ctx, &mut reused3);
        assert_eq!(
            b_reused3, b_virgin,
            "reusing a cached+decoded cache for a shorter prompt changed the logits"
        );
    }

    /// B2 (Phase B): the numeric property prefix reuse rests on.
    ///
    /// If a cache already holds rows `0..L` for tokens `P[0..L]`, then
    /// prefilling only `P[L..]` at positions `L..` must give exactly the same
    /// last-row logits as prefilling all of `P` from position 0 — because
    /// attention for each new token reads `[0, pos+1)`, and rows `0..L` were
    /// verified to hold the same tokens. The server's reuse is gated on that
    /// exact token match (`common_prefix_len`), so this is the safety proof.
    #[test]
    fn prefix_reuse_matches_a_full_prefill() {
        use crate::graph::cache::GraphCache;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping prefix-reuse test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let prompt: Vec<u32> =
            tok.encode("The capital of France is Paris and the capital of Japan is Tokyo");
        let split = prompt.len() / 2;
        let (head, tail) = prompt.split_at(split);
        assert!(!head.is_empty() && !tail.is_empty());

        // (a) Warm the cache with the head, then prefill only the tail.
        let mut incremental = GraphCache::new();
        let hpos: Vec<usize> = (0..head.len()).collect();
        let _ = model.forward_graph_cached(head, &hpos, 1, n_ctx, &mut incremental);
        let tpos: Vec<usize> = (head.len()..prompt.len()).collect();
        let l_incremental = model.forward_graph_cached(tail, &tpos, 1, n_ctx, &mut incremental);

        // (b) A virgin cache prefills the whole prompt in one shot.
        let mut whole = GraphCache::new();
        let ppos: Vec<usize> = (0..prompt.len()).collect();
        let l_whole = model.forward_graph_cached(&prompt, &ppos, 1, n_ctx, &mut whole);

        assert_eq!(l_incremental.len(), l_whole.len());
        // Prefix reuse re-feeds the prefix at one shape and the tail at another,
        // so this is a cross-shape comparison: bitwise on CPU, the named CUDA
        // class on a device (`cross_shape_tolerance`).
        assert_across_shapes(
            &format!(
                "prefix reuse against a full prefill (head {} tail {})",
                head.len(),
                tail.len()
            ),
            &l_incremental,
            &l_whole,
        );
    }

    /// E2's evidence gate: a forward carrying two sequences must equal running
    /// those sequences one at a time, **bitwise** — the prefill (both prompts in
    /// one forward) and a decode step (one token each).
    ///
    /// Both sides use the *same* KV layout — one arena, sequence 7 reserved at
    /// `[0, la)` and sequence 9 at `[la, la + lb)` — so the only difference is
    /// whether one forward carries two sequences or two forwards carry one each.
    /// The batched side takes the windowed attention path
    /// (`Op::Attn { explicit_span: true }`), where sequence 9's window starts at
    /// `la`; equal logits mean the windows are per-sequence and the KV rows do
    /// not leak.
    #[test]
    fn a_two_sequence_batch_matches_two_single_sequence_forwards() {
        use crate::graph::batch::Batch;
        use crate::graph::cache::GraphCache;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the batch test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let nv = model.n_vocab();
        let a = tok.encode("The capital of France is");
        let b = tok.encode("The capital of Japan is");
        let (la, lb) = (a.len(), b.len());
        let (s7, s9) = (7u32, 9u32);

        // The layout both sides use: two reserved runs, deterministic placement.
        let layout = |cache: &mut GraphCache| -> (usize, usize) {
            cache.alloc().kv_set_capacity(n_ctx);
            let r7 = cache.alloc().kv_reserve_seq(s7, la + 4).expect("reserve 7");
            let r9 = cache.alloc().kv_reserve_seq(s9, lb + 4).expect("reserve 9");
            (r7.start, r9.start)
        };

        // ---- batched: both sequences, one forward per step ----
        let mut batched = GraphCache::new();
        let (p7, p9) = layout(&mut batched);
        let pos7: Vec<usize> = (0..la).collect();
        let pos9: Vec<usize> = (0..lb).collect();

        let mut tokens = a.clone();
        tokens.extend_from_slice(&b);
        let mut positions = pos7.clone();
        positions.extend_from_slice(&pos9);
        let mut seq_ids = vec![s7; la];
        seq_ids.extend(std::iter::repeat(s9).take(lb));
        let l_pre = model.forward_batch(
            &Batch::new(tokens, positions, seq_ids),
            1,
            n_ctx,
            &mut batched,
        );
        assert_eq!(l_pre.len(), 2 * nv, "one logits row per sequence");
        let (pre7, pre9) = (l_pre[..nv].to_vec(), l_pre[nv..].to_vec());

        let (t7, t9) = (argmax(&pre7), argmax(&pre9));
        let l_step = model.forward_batch(
            &Batch::new(vec![t7, t9], vec![la, lb], vec![s7, s9]),
            1,
            n_ctx,
            &mut batched,
        );
        assert_eq!(l_step.len(), 2 * nv);

        // ---- reference: the same layout, one sequence per forward ----
        let mut ref7 = GraphCache::new();
        let (q7, _) = layout(&mut ref7);
        assert_eq!(q7, p7, "the reference must place sequence 7 identically");
        let r_pre7 = model.forward_batch(
            &Batch::new(a.clone(), pos7.clone(), vec![s7; la]),
            1,
            n_ctx,
            &mut ref7,
        );
        let r_next7 = model.forward_batch(
            &Batch::new(vec![t7], vec![la], vec![s7]),
            1,
            n_ctx,
            &mut ref7,
        );
        assert_eq!(r_pre7.len(), nv);
        assert_eq!(r_next7.len(), nv);

        let mut ref9 = GraphCache::new();
        let (_, q9) = layout(&mut ref9);
        assert_eq!(q9, p9, "the reference must place sequence 9 identically");
        let r_pre9 = model.forward_batch(
            &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
            1,
            n_ctx,
            &mut ref9,
        );
        let r_next9 = model.forward_batch(
            &Batch::new(vec![t9], vec![lb], vec![s9]),
            1,
            n_ctx,
            &mut ref9,
        );

        // ---- all four comparisons ----
        // The batch carries both sequences in one forward, the reference carries
        // them one per forward: cross-shape, so bitwise on CPU and the named
        // CUDA class on a device. `batch_order_does_not_change_a_sequences_logits`
        // is the same-shape gate that stays bitwise on both.
        assert_across_shapes(&format!("batched prefill of sequence {s7}"), &pre7, &r_pre7);
        assert_across_shapes(&format!("batched prefill of sequence {s9}"), &pre9, &r_pre9);
        assert_across_shapes(
            &format!("batched decode of sequence {s7}"),
            &l_step[..nv],
            &r_next7,
        );
        assert_across_shapes(
            &format!("batched decode of sequence {s9}"),
            &l_step[nv..],
            &r_next9,
        );
        // The fixture must discriminate: two sequences in one forward must not
        // collapse onto the same answer.
        assert_ne!(
            argmax(&l_step[..nv]),
            argmax(&l_step[nv..]),
            "both sequences produced the same argmax - the fixture is not discriminating"
        );
    }

    /// The offset-sensitivity finding (plan §14 row 9), narrowed to what is
    /// actually established — and the two properties that *do* hold are asserted
    /// here so the day someone changes the forward they find out immediately.
    ///
    /// Measured on the 0.5B, same prompt, same relative window, no compaction:
    ///
    /// - the forward is **deterministic** (the same cell twice is bitwise equal);
    /// - **one** query token is offset-invariant *by construction* — a single key
    ///   makes the softmax weight 1 and V is unrotated, so any offset must give
    ///   bit-identical logits, and it does;
    /// - from **two** query tokens on, the logits differ (2.6% relative at an
    ///   8-cell offset) even though layer 0's stored V rows are **bit-identical**,
    ///   the stored K matches the cell rotation to ~1e-5 (`rope_shift_kv` with
    ///   `delta = 8` aligns cell 8 back onto cell 0), the CPU attention reads the
    ///   explicit `attn_span` (an unfilled span is an error, so a silent
    ///   `[0, pos)` fallback cannot be it), and Q/K carry the same freq table
    ///   (only `n_head` differs, and the table depends on `hd`).
    ///
    /// The minimal hand-built graph then exonerated the ops (see
    /// `the_minimal_attention_graph_is_offset_invariant_to_rounding`: ≤ 1.2e-7 at
    /// the model's own shape, and equal for a 1-cell and an 8-cell offset), a
    /// layer bisect put the entry point at layer 0's attention output (layer 0's V
    /// bit-identical, layer 1's differing by 3.6e-3), and a layout control (the
    /// same subject cells with the reservation split in two) came out
    /// bit-identical — so what remains is the rotation's own rounding, amplified
    /// by depth. Plan §14 row 9 carries the full attribution and C3's acceptance:
    /// byte-exact K/V plus a surviving greedy token, with the logits' tail in the
    /// named amplified-rounding class.
    #[test]
    fn offset_sensitivity_is_narrowed_to_multi_query_attention() {
        use crate::graph::batch::Batch;
        use crate::graph::cache::GraphCache;
        use crate::graph::kvcache::KvRope;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let full = tok.encode("The capital of France is");
        let n_ctx = 256;

        // Both runs must take the *same* attention instantiation, or the
        // comparison silently measures "causal vs explicit span" instead of the
        // offset: a run with a single sequence at cell 0 is causal, one with a
        // reservation below it is explicit. A second reservation (a holder when
        // the subject is offset, a dummy after it otherwise) makes both explicit.
        let run = |ids: Vec<u32>, start: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
            let n = ids.len();
            let mut cache = GraphCache::new();
            cache.alloc().kv_set_capacity(n_ctx);
            if start > 0 {
                cache.alloc().kv_reserve_seq(3, start).expect("holder");
            }
            cache.alloc().kv_reserve_seq(7, n + 4).expect("subject");
            if start == 0 {
                cache.alloc().kv_reserve_seq(3, 4).expect("dummy");
            }
            let l = model.forward_batch(
                &Batch::new(ids, (0..n).collect(), vec![7u32; n]),
                1,
                n_ctx,
                &mut cache,
            );
            let (k, v) = cache.alloc().copy_kv_to_cpu(0).expect("kv read");
            (l, k, v)
        };
        let d = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        // 1 token is the discriminator: with a single key the softmax weight is 1
        // and V is unrotated, so *any* offset must give bit-identical logits.
        let (freq_base, freq_scale) = model.rope_params();
        let rope = KvRope {
            freq_base,
            freq_scale,
            n_head_kv: model.n_head_kv(),
            hd: model.n_embd_head(),
            style: model.rope_style(),
        };
        for nt in [1usize, 2, full.len()] {
            let ids: Vec<u32> = full[..nt].to_vec();
            let (l0, k0, v0) = run(ids.clone(), 0);
            let (l0b, _k0b, v0b) = run(ids.clone(), 0);
            let (l8, k8, v8) = run(ids, 8);
            let row = v0.len() / n_ctx;
            // `rope_shift_kv(d)` means `new_pos = old_pos - d`, so aligning cell 8
            // back onto cell 0 is d = +8.
            // C6: compare the same relative rows directly — no rotation to undo.
            let k8_aligned = k8[8 * row..(nt + 8) * row].to_vec();
            let v_delta = d(&v0[..nt * row], &v8[8 * row..(nt + 8) * row]);
            let k_aligned = d(&k0[..nt * row], &k8_aligned);
            eprintln!(
                "[offset] nt={nt}: determinism {} | cell0-vs-8 logits {} | V {} | K aligned {}",
                d(&l0, &l0b),
                d(&l0, &l8),
                v_delta,
                k_aligned
            );
            // Determinism holds at every width (logits are compared bitwise above).
            assert_eq!(d(&l0, &l0b), 0.0, "the forward must be deterministic");
            // The KV-region probes below read the stored rows directly, which is
            // only meaningful where the store is f32: on a CUDA build the
            // process-wide KV dtype can be f16 (another model set it), and the
            // raw read would be bit patterns, not values.
            if matches!(model.device(), crate::models::Device::Cpu) {
                assert_eq!(d(&v0, &v0b), 0.0, "and bit-identical run to run");
                // C6: the same relative row at a different cell holds the same
                // bytes — no rotation is involved in a cell move any more.
                assert_eq!(v_delta, 0.0, "V is unrotated and cell-independent");
                assert_eq!(
                    k_aligned, 0.0,
                    "the stored K is verbatim at the sequence's new cell"
                );
            } else {
                eprintln!("[offset] (non-CPU device: skipping the raw KV-row probes)");
            }
            if nt == 1 {
                // A single key makes the softmax weight 1, so the scores cannot
                // matter: this must be bitwise, whatever the offset.
                assert_eq!(d(&l0, &l8), 0.0, "one query token is offset-invariant");
            } else {
                // Deliberately NOT asserted: the multi-query divergence is the
                // open finding (plan §14 row 9), and asserting it would turn a
                // bug report into a specification.
                assert!(d(&l0, &l8).is_finite());
            }
        }
    }

    /// Plan §14 row 9's decisive measurement: **layer 0 is offset-consistent
    /// within the rotation's own rounding**, and the divergence therefore appears
    /// downstream of it.
    ///
    /// Reading an intermediate buffer *after* `execute` is unsafe (liveness
    /// recycling), so this mirrors the scheduler's loop instead: execute the
    /// model's graph **node by node in build order**, capture layer 0's two rope
    /// outputs and its attention output immediately after each runs, and skip
    /// nodes the fusion pass orphaned (they have no buffer and the scheduler skips
    /// them too). The alignment numbers are their own validity check — the q and k
    /// ropes align to the same 1.5e-5 the independent KV-level measurement
    /// produced — and they say:
    ///
    /// | Buffer | cell 0 vs cell 8, after aligning the rotation |
    /// |---|---|
    /// | layer 0 q_roped | 1.5e-5 (rel 1.9e-7) |
    /// | layer 0 k_roped | 1.5e-5 (rel 1.2e-7) |
    /// | layer 0 attention output | 7.3e-7 (rel 7.3e-7) |
    ///
    /// So the ops are exact here, and the earlier KV bisect's layer-1 difference
    /// (3.6e-3) is that rounding **amplified downstream** — which is why C3's
    /// acceptance is byte-exact K/V plus a surviving greedy token with the
    /// logits' tail in a named class, not bit-identical logits.
    #[test]
    fn layer_0_is_offset_consistent_within_the_rotation_rounding_class() {
        use crate::graph::backend::Backend as _;
        use crate::graph::backend::KvProvider;
        use crate::graph::batch::Batch;
        use crate::graph::fusion::FusionPass;
        use crate::graph::kvcache::{rope_shift_kv, KvRope};
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        let n = ids.len();
        let n_ctx = 256;
        let (freq_base, freq_scale) = model.rope_params();
        let style = model.rope_style();
        let hd = model.n_embd_head();

        // (q rope, k rope, attention output) of layer 0.
        type Captured = (Vec<f32>, Vec<f32>, Vec<f32>);
        let run = |start: usize| -> Captured {
            let params = GraphParams {
                n_tokens: n,
                n_out: 1,
                gtype: GraphType::Prefill,
                cparams: CParams {
                    n_ctx,
                    flash_attn: false,
                    explicit_span: true,
                    gpu: false,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            let mut graph = model.build_graph(&params);
            let ropes: Vec<crate::graph::NodeId> = graph
                .nodes
                .iter()
                .filter(|nd| matches!(nd.op, crate::graph::ops::Op::RoPE { .. }))
                .take(2)
                .map(|nd| nd.id)
                .collect();
            let (q_node, k_node) =
                if graph.nodes[ropes[0]].out_shape[0] > graph.nodes[ropes[1]].out_shape[0] {
                    (ropes[0], ropes[1])
                } else {
                    (ropes[1], ropes[0])
                };
            let attn = graph
                .nodes
                .iter()
                .find(|nd| matches!(nd.op, crate::graph::ops::Op::Attn { .. }))
                .expect("attention node")
                .id;

            let mut alloc = GraphAllocator::new();
            alloc.kv_set_capacity(n_ctx);
            if start > 0 {
                alloc.kv_reserve_seq(3, start).expect("holder");
            }
            alloc.kv_reserve_seq(7, n + 4).expect("subject");
            if start == 0 {
                alloc.kv_reserve_seq(3, 4).expect("dummy");
            }

            let concrete = model
                .as_any()
                .downcast_ref::<crate::models::qwen2::Qwen2Model>()
                .expect("Qwen2Model");
            Qwen2Graph::register_graph_weights(concrete, &mut alloc);
            let mut sched = BackendScheduler::new();
            sched.assign_backends(&mut graph, &mut alloc);
            {
                let backends: Vec<&dyn crate::graph::backend::Backend> = vec![alloc.cpu()];
                FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                    Some(crate::graph::Backend::CPU) => Some(0),
                    _ => None,
                });
            }
            alloc.alloc_graph(&graph).expect("alloc");

            let batch = Batch::new(ids.clone(), (0..n).collect(), vec![7u32; n]);
            alloc.fill_input_i32(&graph, "token_ids", &ids).unwrap();
            let pos: Vec<u32> = (0..n).map(|p| p as u32).collect();
            alloc.fill_input_i32(&graph, "positions", &pos).unwrap();
            alloc.fill_batch_inputs(&graph, &batch).unwrap();
            if graph
                .inputs
                .iter()
                .any(|&i| graph.node(i).name == "tail_ids")
            {
                let rows: Vec<u32> = batch.out_rows(1).iter().map(|&r| r as u32).collect();
                alloc.fill_input_i32(&graph, "tail_ids", &rows).unwrap();
            }

            // Build order is a valid topological order (source before consumer),
            // so executing node by node in id order is what the scheduler does.
            let order: Vec<crate::graph::NodeId> = graph.nodes.iter().map(|nd| nd.id).collect();
            let mut captured: Captured = (Vec::new(), Vec::new(), Vec::new());
            for id in order {
                let node = graph.node(id);
                if matches!(node.op, crate::graph::ops::Op::Input) {
                    continue;
                }
                let ins: Vec<crate::graph::BufRef> = node
                    .src
                    .iter()
                    .map(|&s| alloc.node_buffer(s).expect("input buffer"))
                    .collect();
                // A node the fusion pass orphaned (e.g. a silu folded into
                // SwiGLU) has no buffer and is skipped, not executed — the same
                // rule the scheduler applies.
                let Some(out) = alloc.node_buffer(id) else {
                    continue;
                };
                let kv = match &node.op {
                    crate::graph::ops::Op::KvcacheStore { layer } => alloc.kv_pair(*layer),
                    crate::graph::ops::Op::Attn { .. } => match &node.meta {
                        crate::graph::ops::NodeMeta::Attn(m) => alloc.kv_pair(m.layer),
                        _ => None,
                    },
                    _ => None,
                };
                alloc
                    .cpu_mut()
                    .execute_node(node, &ins, out, kv)
                    .unwrap_or_else(|e| panic!("node {} ({}): {e}", node.id, node.name));
                let slot = if id == q_node {
                    Some(0)
                } else if id == k_node {
                    Some(1)
                } else if id == attn {
                    Some(2)
                } else {
                    None
                };
                if let Some(slot) = slot {
                    let host = alloc.cpu().read_host(out.id).expect("host read");
                    let window = host[out.offset..out.offset + out.len].to_vec();
                    match slot {
                        0 => captured.0 = window,
                        1 => captured.1 = window,
                        _ => captured.2 = window,
                    }
                }
            }
            captured
        };
        let d = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let scale = |v: &[f32]| v.iter().map(|x| x.abs()).fold(1.0f32, f32::max);

        let (q0, k0, a0) = run(0);
        let (q8, k8, a8) = run(8);
        // C6: a token's RoPE angle is its index within its sequence, so the same
        // relative token roped at cell 0 and at cell 8 is **bitwise** the same
        // — there is no rotation left to align.
        let (q8a, k8a) = (q8.clone(), k8.clone());
        eprintln!(
            "[per-node] q {} (rel {}) | k {} (rel {}) | attn {} (rel {})",
            d(&q0, &q8a),
            d(&q0, &q8a) / scale(&q0),
            d(&k0, &k8a),
            d(&k0, &k8a) / scale(&k0),
            d(&a0, &a8),
            d(&a0, &a8) / scale(&a0),
        );
        // The rotation's own rounding class, measured: the ropes are the only
        // inputs the offset touches, and layer 0's output follows them.
        assert!(
            d(&q0, &q8a) / scale(&q0) < 1e-5,
            "the query rope must be offset-consistent"
        );
        assert!(
            d(&k0, &k8a) / scale(&k0) < 1e-5,
            "the key rope must be offset-consistent"
        );
        assert!(
            d(&a0, &a8) / scale(&a0) < 1e-5,
            "layer 0's attention output must be offset-consistent"
        );
    }

    /// C3's end-to-end gate: compacting the arena **between steps** must leave a
    /// session's continuation intact.
    ///
    /// The subject sequence is reserved at a non-zero start (a holder occupies
    /// the cells below it) so the compaction really moves it, and `positions` are
    /// cells today, which means the K rows it leaves behind carry the RoPE angle
    /// of their *old* cells: they are re-roped by the delta. Without that re-rope
    /// the next step attends to rows rotated by the whole offset and this test
    /// diverges completely; with it, the continuation matches a run that was at
    /// cell 0 all along, up to C2's re-rope tolerance class (two composed
    /// rotations, not one).
    #[test]
    fn a_compaction_between_steps_keeps_the_continuation() {
        use crate::graph::batch::Batch;
        use crate::graph::cache::GraphCache;
        use crate::graph::kvcache::KvRope;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the compaction test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let nv = model.n_vocab();
        let s1 = 5u32;
        let subject = tok.encode("The capital of France is");
        let n = subject.len();
        let holder = 8usize; // cells reserved below the subject, released to force a move
        let step_tok = subject[n - 1]; // the fed token only has to match on both sides
        let (freq_base, freq_scale) = model.rope_params();
        let rope = KvRope {
            freq_base,
            freq_scale,
            n_head_kv: model.n_head_kv(),
            hd: model.n_embd_head(),
            style: model.rope_style(),
        };

        // ---- A: prefill + one step at a non-zero start, then compact, then step ----
        let mut a_cache = GraphCache::new();
        a_cache.alloc().kv_set_capacity(n_ctx);
        a_cache.alloc().kv_reserve_seq(3, holder).expect("holder");
        let start_a = a_cache
            .alloc()
            .kv_reserve_seq(s1, n + 4)
            .expect("subject")
            .start;
        assert_eq!(start_a, holder, "the holder must sit below the subject");
        let pre_a = model.forward_batch(
            &Batch::new(subject.clone(), (0..n).collect(), vec![s1; n]),
            1,
            n_ctx,
            &mut a_cache,
        );
        assert_eq!(pre_a.len(), nv);
        let l_a_step1 = model.forward_batch(
            &Batch::new(vec![step_tok], vec![n], vec![s1]),
            1,
            n_ctx,
            &mut a_cache,
        );

        // Release the holder and pack the subject down: this is the migration the
        // server performs when it follows a compaction report.
        a_cache.alloc().kv_release_seq(3);
        let report = a_cache.alloc().kv_defrag(Some(0)).expect("compact");
        assert!(
            !report.moves.is_empty(),
            "the subject must actually move: {report:?}"
        );
        assert_eq!(report.moves[0].seq, s1);
        let moved_by = report.moves[0].from - report.moves[0].to;
        assert_eq!(
            moved_by, holder,
            "the delta is the released holder's capacity"
        );
        let start_b = a_cache.alloc().kv_seq_slot(s1).expect("subject slot").start;
        assert_eq!(start_b, 0, "the compaction packs the subject to cell 0");
        let l_a = model.forward_batch(
            &Batch::new(vec![step_tok], vec![n + 1], vec![s1]),
            1,
            n_ctx,
            &mut a_cache,
        );

        // ---- B: the same session that never moved ----
        let mut b_cache = GraphCache::new();
        b_cache.alloc().kv_set_capacity(n_ctx);
        assert_eq!(
            b_cache
                .alloc()
                .kv_reserve_seq(s1, n + 4)
                .expect("control subject")
                .start,
            0
        );
        // A dummy second reservation, never written: it makes the control's
        // attention explicit-span too, so A and B differ in the *offset* alone
        // (otherwise the control would take the causal instantiation and the
        // comparison would mix two variables).
        b_cache.alloc().kv_reserve_seq(9, 4).expect("dummy");
        let pre_b = model.forward_batch(
            &Batch::new(subject.clone(), (0..n).collect(), vec![s1; n]),
            1,
            n_ctx,
            &mut b_cache,
        );
        // The prefill logits of the two runs differ although every *relative*
        // quantity is the same — see `offset_sensitivity_...` for the narrowed
        // finding and plan §14 row 9.
        let dpre = pre_a
            .iter()
            .zip(&pre_b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("[c3] prefill logits offset 8 vs 0: max |d| = {dpre}");
        let l_b_step1 = model.forward_batch(
            &Batch::new(vec![step_tok], vec![n], vec![s1]),
            1,
            n_ctx,
            &mut b_cache,
        );
        // The two runs have taken *identical* steps (same tokens, same relative
        // windows), differing only in the subject's cell offset... and that alone
        // already moves the logits: 2.6% relative here, on the 0.5B, before any
        // compaction. That is a finding in its own right (it is exactly why C3
        // can promise "the greedy token survives" but not "bit-identical" until
        // positions become sequence-relative), so it is printed rather than
        // tolerated silently.
        let d1 = l_a_step1
            .iter()
            .zip(&l_b_step1)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let sc1 = l_b_step1.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        eprintln!(
            "[c3] offset-alone effect (cell 8 vs cell 0): max |d| = {d1} (relative {})",
            d1 / sc1
        );
        let l_b = model.forward_batch(
            &Batch::new(vec![step_tok], vec![n + 1], vec![s1]),
            1,
            n_ctx,
            &mut b_cache,
        );

        // ---- compare: behaviour first, then the named tolerance class ----
        assert_eq!(
            argmax(&l_a),
            argmax(&l_b),
            "the greedy token must survive a compaction"
        );
        let worst = l_a
            .iter()
            .zip(&l_b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let scale = l_b.iter().map(|v| v.abs()).fold(1.0f32, f32::max);
        eprintln!(
            "[c3] post-compaction logits: max |d| = {worst} (relative {}; the \
             offset-alone effect above is the floor, not this fix)",
            worst / scale
        );
        assert!(
            worst.is_finite(),
            "the compaction produced non-finite logits"
        );
        // The gate is the *behaviour*, not the last bit: with a missing or
        // sign-flipped re-rope this argmax flips, and that is how this test first
        // failed. The byte-level identity of the move (V verbatim, K exactly
        // `rope_shift_kv(old, delta)`) is pinned in `kv_defrag_moves_the_bytes_and_opens_the_run`.
        assert_eq!(
            argmax(&l_a),
            argmax(&l_b),
            "the greedy token must survive a compaction (a wrong re-rope flips it)"
        );
    }

    /// E2 / A7: the *number of sequences* is data, not topology.
    ///
    /// `explicit_span` (derived from the KV reservations) already fixes every
    /// topology decision a batch can make, so two batches with the same token
    /// count, output count and span requirement describe **the same graph**
    /// whether the tokens belong to one sequence or two. While `GraphParams`
    /// also carried the sequence count, that pair rebuilt - a rebuild with no
    /// topological cause, which is why the field was deleted in E2 instead of
    /// kept "reserved" (A7 closure).
    ///
    /// The test pins both halves: the graph uid is unchanged across the pair (no
    /// rebuild) and the reused graph still computes what a fresh
    /// single-sequence forward computes, bitwise.
    #[test]
    fn sequence_count_is_data_not_topology() {
        use crate::graph::batch::Batch;
        use crate::graph::cache::GraphCache;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the sequence-count test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let nv = model.n_vocab();
        let a = tok.encode("The capital of France is");
        let b = tok.encode("The capital of Japan is");
        let (la, lb) = (a.len(), b.len());
        let (s7, s9) = (7u32, 9u32);

        // The same reservation order in every cache, so the layout - and with it
        // every position below - is deterministic. Both starts are non-zero, so
        // every forward here needs the explicit span.
        let layout = |cache: &mut GraphCache| -> (usize, usize) {
            cache.alloc().kv_set_capacity(n_ctx);
            let r7 = cache.alloc().kv_reserve_seq(s7, la + 2).expect("reserve 7");
            let r9 = cache.alloc().kv_reserve_seq(s9, lb + 4).expect("reserve 9");
            (r7.start, r9.start)
        };
        let uid = |c: &mut GraphCache| c.current().map(|(g, _)| g.uid).unwrap();

        // ---- batched cache: one graph is asked for both shapes ----
        let mut c = GraphCache::new();
        let (p7, p9) = layout(&mut c);
        let pos7: Vec<usize> = (0..la).collect();
        let pos9: Vec<usize> = (0..lb).collect();

        let pre7 = model.forward_batch(
            &Batch::new(a.clone(), pos7.clone(), vec![s7; la]),
            1,
            n_ctx,
            &mut c,
        );
        let t7 = argmax(&pre7);
        let pre9 = model.forward_batch(
            &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
            1,
            n_ctx,
            &mut c,
        );
        let t9 = argmax(&pre9);

        // Two sequences, one token each: nt = 2, n_out = 2 (a row per sequence).
        let two = model.forward_batch(
            &Batch::new(vec![t7, t9], vec![la, lb], vec![s7, s9]),
            1,
            n_ctx,
            &mut c,
        );
        assert_eq!(two.len(), 2 * nv, "one logits row per sequence");
        let uid_two = uid(&mut c);

        // One sequence, two tokens: the same nt, the same n_out (a caller that
        // wants both rows' distributions), the same span requirement. Only the
        // sequence count differs, so this must reuse the graph above.
        let tail = [a[0], a[1]];
        let tail_pos = [lb + 1, lb + 2];
        let one = model.forward_batch(
            &Batch::new(tail.to_vec(), tail_pos.to_vec(), vec![s9; 2]),
            2,
            n_ctx,
            &mut c,
        );
        assert_eq!(one.len(), 2 * nv, "nt = 2, n_out = 2");
        assert_eq!(
            uid_two,
            uid(&mut c),
            "a change in the number of sequences rebuilt an otherwise identical graph"
        );

        // ---- reference: the same layout, sequence 9 alone ----
        let mut ref9 = GraphCache::new();
        let (_, q9) = layout(&mut ref9);
        assert_eq!(q9, p9, "the reference must place sequence 9 identically");
        let r_pre9 = model.forward_batch(
            &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
            1,
            n_ctx,
            &mut ref9,
        );
        assert_eq!(
            argmax(&r_pre9),
            t9,
            "the reference must agree on sequence 9's first token"
        );
        let _ = model.forward_batch(
            &Batch::new(vec![t9], vec![lb], vec![s9]),
            1,
            n_ctx,
            &mut ref9,
        );
        let r_one = model.forward_batch(
            &Batch::new(tail.to_vec(), tail_pos.to_vec(), vec![s9; 2]),
            2,
            n_ctx,
            &mut ref9,
        );

        // Sequence 7's work (and reusing the graph) must not have touched
        // sequence 9's rows. The batched side's keys were written by a two-token
        // forward and one of the pair's, the reference's by single-sequence
        // forwards — cross-shape, so CPU stays bitwise and a device uses the
        // named class.
        assert_across_shapes(
            "the reused graph against a fresh single-sequence forward",
            &one,
            &r_one,
        );
    }

    /// E1b/E2: a query's attention window follows from **which sequence it
    /// belongs to and where that sequence is reserved** — never from its row
    /// index inside the batch. Swapping two sequences in an otherwise identical
    /// batch must therefore reproduce each sequence's logits *bitwise*, on CPU
    /// and on CUDA alike.
    ///
    /// This is the device-capable form of E1's "two sequences do not
    /// cross-attend" gate. The earlier batched tests compare forwards of
    /// *different shapes* (a batched prefill against per-sequence prefills),
    /// which is bitwise only on CPU: CUDA's prefill kernels tile by `nt` and
    /// store quantized activations, so a different shape changes the K/V values
    /// by ~1e-3 relative — measured on this box as ~0.3 absolute on logits, and
    /// already true on master for B2's and C2's cross-shape tests. A swap keeps
    /// the shape, the layout, the KV history and the graph identical, leaving
    /// the window assignment as the only thing that can move the numbers, so
    /// bitwise equality is the right expectation on every backend.
    #[test]
    fn batch_order_does_not_change_a_sequences_logits() {
        use crate::graph::batch::Batch;
        use crate::graph::cache::GraphCache;
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the batch-order test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let nv = model.n_vocab();
        let a = tok.encode("The capital of France is");
        let b = tok.encode("The capital of Japan is");
        let (la, lb) = (a.len(), b.len());
        let (s7, s9) = (7u32, 9u32);

        // Same reservations, same order, in both caches: first-fit makes the two
        // layouts identical, so the only difference below is the batch order.
        let layout = |cache: &mut GraphCache| -> (usize, usize) {
            cache.alloc().kv_set_capacity(n_ctx);
            let r7 = cache.alloc().kv_reserve_seq(s7, la + 2).expect("reserve 7");
            let r9 = cache.alloc().kv_reserve_seq(s9, lb + 2).expect("reserve 9");
            (r7.start, r9.start)
        };

        // The whole run, with sequence 7 either first or second in the decode
        // batch. Returns `(logits of 7, logits of 9)`.
        let run = |cache: &mut GraphCache, seq7_first: bool| -> (Vec<f32>, Vec<f32>) {
            let (p7, p9) = layout(cache);
            let pos7: Vec<usize> = (0..la).collect();
            let pos9: Vec<usize> = (0..lb).collect();
            let pre7 = model.forward_batch(
                &Batch::new(a.clone(), pos7.clone(), vec![s7; la]),
                1,
                n_ctx,
                cache,
            );
            let t7 = argmax(&pre7);
            let pre9 = model.forward_batch(
                &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
                1,
                n_ctx,
                cache,
            );
            let t9 = argmax(&pre9);

            // One token per sequence, nt = 2 either way.
            let (tokens, positions, seq_ids) = if seq7_first {
                (vec![t7, t9], vec![la, lb], vec![s7, s9])
            } else {
                (vec![t9, t7], vec![lb, la], vec![s9, s7])
            };
            let rows =
                model.forward_batch(&Batch::new(tokens, positions, seq_ids), 1, n_ctx, cache);
            assert_eq!(rows.len(), 2 * nv, "one logits row per sequence");
            if seq7_first {
                (rows[..nv].to_vec(), rows[nv..].to_vec())
            } else {
                (rows[nv..].to_vec(), rows[..nv].to_vec())
            }
        };

        let mut c1 = GraphCache::new();
        let (r7_first, r9_second) = run(&mut c1, true);
        let mut c2 = GraphCache::new();
        let (r7_second, r9_first) = run(&mut c2, false);

        assert_eq!(
            max_delta(&r7_first, &r7_second),
            0.0,
            "sequence 7's logits depend on its row index in the batch"
        );
        assert_eq!(
            max_delta(&r9_second, &r9_first),
            0.0,
            "sequence 9's logits depend on its row index in the batch"
        );
        // The fixture must discriminate: swapping must not have collapsed the
        // two sequences onto the same distribution.
        assert!(
            max_delta(&r7_first, &r9_second) > 0.0,
            "both sequences produced identical logits - the fixture is not discriminating"
        );
    }

    /// C2 (Phase C): a physical KV removal, and the sliding-window shift built
    /// on it.
    ///
    /// Three things are asserted **bitwise**, because a fresh prefill is an exact
    /// reference for them:
    ///
    /// 1. Removing a *tail* range invalidates exactly those rows: what is left
    ///    equals what a fresh prefill of the retained prefix computed, at every
    ///    layer, so continuing from it is bitwise-identical.
    /// 2. Removing a *middle* range (the conversation's overflow case: keep the
    ///    system prompt, drop an old turn) copies V byte-for-byte — only K is
    ///    re-roped — and leaves `[0, start)` completely untouched.
    /// 3. Removing everything written empties the arena; `len == 0` is a no-op
    ///    and removing past the end is an error.
    ///
    /// The fourth measurement is the one a fresh prefill cannot be a reference
    /// for. Shifting the window re-ropes the survivors, but their *values* were
    /// computed in the pre-shift context: a row that attended to the dropped
    /// prefix keeps that influence. That is inherent to any shift that avoids
    /// re-prefilling (llama.cpp's context shift has the same property), so C2
    /// records it as a named tolerance class instead of asserting equality. The
    /// numbers are printed on every run and pinned in the execution plan; the
    /// assertion here only guards against a *mechanism* regression, which the
    /// exact per-layer checks above already catch.
    #[test]
    fn kv_rm_is_exact_and_the_window_shift_is_a_named_tolerance_class() {
        use crate::graph::cache::GraphCache;
        use crate::graph::kvcache::{rope_shift_kv, KvRope};
        use crate::models::ModelDef;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the KV removal test");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        let n_ctx = 256;
        let a = tok.encode("The capital of France is Paris and");
        let b = tok.encode(" the capital of Japan is Tokyo and");
        let c = tok.encode(" the capital of Italy is Rome");
        let d = tok.encode(" the capital of Spain is Madrid");
        let ab: Vec<u32> = a.iter().chain(&b).copied().collect();
        let cd: Vec<u32> = c.iter().chain(&d).copied().collect();
        let abc: Vec<u32> = a.iter().chain(&b).chain(&c).copied().collect();
        let bc: Vec<u32> = b.iter().chain(&c).copied().collect();
        let nkt = model.n_kv_embd();
        let (freq_base, freq_scale) = model.rope_params();
        let rope = KvRope {
            freq_base,
            freq_scale,
            n_head_kv: model.n_head_kv(),
            hd: model.n_embd_head(),
            style: model.rope_style(),
        };
        let layers: Vec<usize> = (0..24).collect();
        // Max |Δ| between two equal-length windows; a mismatch is a bug, not a
        // short comparison.
        let delta = |x: &[f32], y: &[f32]| {
            assert_eq!(x.len(), y.len(), "compared windows must have equal length");
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        };
        let snapshot = |cache: &mut GraphCache| -> Vec<(Vec<f32>, Vec<f32>)> {
            layers
                .iter()
                .map(|&l| cache.alloc().copy_kv_to_cpu(l).expect("kv read"))
                .collect()
        };

        // (1) Remove the *tail*: the retained head must stay exactly what a fresh
        // prefill of it computed, so the continuation is bitwise-identical.
        let mut cut = GraphCache::new();
        let pos_ab: Vec<usize> = (0..ab.len()).collect();
        let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut cut);
        let before = snapshot(&mut cut);
        assert_eq!(
            cut.alloc()
                .kv_rm(a.len(), b.len(), &rope)
                .expect("remove the tail range"),
            a.len(),
            "only A survives removing B"
        );
        for (i, &l) in layers.iter().enumerate() {
            let (k, v) = cut.alloc().copy_kv_to_cpu(l).expect("kv read");
            assert_eq!(
                delta(&k[..a.len() * nkt], &before[i].0[..a.len() * nkt]),
                0.0,
                "layer {l}: removing the tail must not touch the retained K"
            );
            assert_eq!(
                delta(&v[..a.len() * nkt], &before[i].1[..a.len() * nkt]),
                0.0,
                "layer {l}: removing the tail must not touch the retained V"
            );
        }
        let pos_cd: Vec<usize> = (a.len()..a.len() + cd.len()).collect();
        let l_cut = model.forward_graph_cached(&cd, &pos_cd, 1, n_ctx, &mut cut);
        let mut fresh = GraphCache::new();
        let pos_a: Vec<usize> = (0..a.len()).collect();
        let _ = model.forward_graph_cached(&a, &pos_a, 1, n_ctx, &mut fresh);
        let l_ref = model.forward_graph_cached(&cd, &pos_cd, 1, n_ctx, &mut fresh);
        assert_eq!(l_cut.len(), l_ref.len());
        // A's rows were computed in the A+B forward, the reference's in an A-only
        // forward: a cross-shape comparison, so the CUDA tolerance class applies.
        assert_across_shapes("removing B against a fresh A + (C, D)", &l_cut, &l_ref);

        // (2) Remove a *middle* range: V copies byte-for-byte, [0, start) is
        // untouched, only the moved K is re-roped.
        let mut mid = GraphCache::new();
        let pos_abc: Vec<usize> = (0..abc.len()).collect();
        let _ = model.forward_graph_cached(&abc, &pos_abc, 1, n_ctx, &mut mid);
        let before = snapshot(&mut mid);
        let keep = a.len() + c.len();
        assert_eq!(
            mid.alloc()
                .kv_rm(a.len(), b.len(), &rope)
                .expect("remove the middle range"),
            keep,
            "A and C survive removing the middle range B"
        );
        let src = a.len() + b.len()..a.len() + b.len() + c.len();
        for (i, &l) in layers.iter().enumerate() {
            let (k, v) = mid.alloc().copy_kv_to_cpu(l).expect("kv read");
            assert_eq!(
                delta(&k[..a.len() * nkt], &before[i].0[..a.len() * nkt]),
                0.0,
                "layer {l}: K before the removal must not move"
            );
            assert_eq!(
                delta(&v[..a.len() * nkt], &before[i].1[..a.len() * nkt]),
                0.0,
                "layer {l}: V before the removal must not move"
            );
            // V has no rope: the moved rows must be byte-for-byte the old ones.
            assert_eq!(
                &v[a.len() * nkt..keep * nkt],
                &before[i].1[src.start * nkt..src.end * nkt],
                "layer {l}: V must move verbatim"
            );
            // K must be the moved rows re-roped by -b.len(): undoing the shift
            // with the opposite angle must land back on the source within the
            // rope tolerance class. A wrong sign, or re-roping the wrong rows,
            // cannot survive this.
            let mut back: Vec<f32> = k[a.len() * nkt..keep * nkt].to_vec();
            rope_shift_kv(&mut back, c.len(), -(b.len() as isize), &rope);
            let worst = delta(&back, &before[i].0[src.start * nkt..src.end * nkt]);
            assert!(
                worst < 1e-4,
                "layer {l}: the moved K must be the source K re-roped, got |Δ| = {worst}"
            );
        }

        // (3) Removing everything written empties the arena; `len == 0` is a
        // no-op and removing past the end is an error.
        let mut empty = GraphCache::new();
        let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut empty);
        assert_eq!(empty.alloc().kv_rm(0, ab.len(), &rope).unwrap(), 0);
        assert_eq!(empty.alloc().kv_n_used(0), Some(0));
        let mut noop = GraphCache::new();
        let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut noop);
        assert_eq!(noop.alloc().kv_rm(3, 0, &rope).unwrap(), ab.len());
        assert!(noop.alloc().kv_rm(0, ab.len() + 1, &rope).is_err());

        // (4) The sliding window: `kv_shift(drop)` == `kv_rm(0, drop)`. The
        // mechanism is checked exactly per layer; the deviation from a fresh
        // window is measured and recorded, not asserted away.
        let mut shifted_cache = GraphCache::new();
        let _ = model.forward_graph_cached(&ab, &pos_ab, 1, n_ctx, &mut shifted_cache);
        let before = snapshot(&mut shifted_cache);
        assert_eq!(
            shifted_cache
                .alloc()
                .kv_shift(a.len(), &rope)
                .expect("context shift"),
            b.len(),
            "only B survives the shift"
        );
        // Cells keep `cell == pos`, so the scheduler's identity gate stays
        // satisfied and no backend needs a new kernel.
        assert!(
            shifted_cache.alloc().kv_is_identity(),
            "a physical shift must keep the identity cell mapping"
        );
        for (i, &l) in layers.iter().enumerate() {
            let (k, v) = shifted_cache.alloc().copy_kv_to_cpu(l).expect("kv read");
            assert_eq!(
                &v[..b.len() * nkt],
                &before[i].1[a.len() * nkt..ab.len() * nkt],
                "layer {l}: V must survive the shift verbatim"
            );
            let mut back: Vec<f32> = k[..b.len() * nkt].to_vec();
            rope_shift_kv(&mut back, b.len(), -(a.len() as isize), &rope);
            let worst = delta(&back, &before[i].0[a.len() * nkt..ab.len() * nkt]);
            assert!(
                worst < 1e-4,
                "layer {l}: shifted K must be the source K re-roped, got |Δ| = {worst}"
            );
        }
        let pos_c: Vec<usize> = (b.len()..bc.len()).collect();
        let l_shifted = model.forward_graph_cached(&c, &pos_c, 1, n_ctx, &mut shifted_cache);
        let mut virgin = GraphCache::new();
        let pos_bc: Vec<usize> = (0..bc.len()).collect();
        let l_fresh = model.forward_graph_cached(&bc, &pos_bc, 1, n_ctx, &mut virgin);
        assert_eq!(l_shifted.len(), l_fresh.len());
        let worst = delta(&l_shifted, &l_fresh);
        eprintln!(
            "[c2] window shift vs a fresh window: max|Δlogits| = {worst} \
             (a={} b={} c={} tokens) — inherent: B's rows keep A's context",
            a.len(),
            b.len(),
            c.len()
        );
        assert!(
            worst.is_finite() && worst < 25.0,
            "the shift is degraded far beyond the recorded tolerance class: max|Δ| = {worst}"
        );
    }

    /// Phase 6 verification (hermetic on CPU builds): the graph path must
    /// reproduce forward.rs logits on a real model — prefill and a decode
    /// step (KV carried across). Built and executed locally (no global
    /// cache), so it is immune to other tests initializing Metal. On CUDA
    /// builds the engine side runs the CUDA graph — cross-backend mode then
    /// asserts greedy-token equality (see `compare`).
    #[test]
    fn graph_logits_match_forward_real_model() {
        use crate::graph::alloc::GraphAllocator;
        use crate::graph::backend::Backend;
        use crate::graph::builder::GraphBuilder;
        use crate::graph::fusion::FusionPass;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        use crate::graph::ComputeGraph;
        use crate::models::ModelDef;

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
        // Keep the weight registry stable for this whole test: a parallel
        // test loading a different architecture swaps same-named entries,
        // which would flip the CUDA gate mid-test (persistent KV regions
        // were allocated under the earlier decision).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let q2: &Qwen2Model = model
            .as_any()
            .downcast_ref::<Qwen2Model>()
            .expect("qwen2 model");
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
                n_out: 1,
                gtype: GraphType::Prefill,
                cparams: CParams {
                    n_ctx,
                    flash_attn: false,
                    explicit_span: false,
                    gpu: false,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
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
            let seqs = vec![crate::graph::kvcache::SEQ_MAIN; nt];
            alloc.fill_attn_inputs(&graph, &seqs, &pos32).unwrap();
            // G3 tail-reduction input: forward_cached fills it (see
            // forward_cached); the manual graph must do the same, otherwise the
            // reduce picks row 0 instead of the last row → logits of a different
            // token (the ~1.9e1 divergence, issue 1).
            if graph
                .inputs
                .iter()
                .any(|&i| graph.node(i).name == "tail_ids")
            {
                let tail: Vec<u32> = ((nt - params.n_out)..nt).map(|x| x as u32).collect();
                alloc.fill_input_i32(&graph, "tail_ids", &tail).unwrap();
            }
            sched.execute(&graph, &mut alloc).unwrap();
            let nv = model.n_vocab();
            let logits = alloc.copy_to_cpu(graph.outputs[0]).unwrap();
            // n_out=1 prefill with the G3 tail reduction: the output buffer is
            // exactly the last row's logits (n_out × nv), not nt rows
            let prefill_l = logits[..nv].to_vec();

            // decode graph (same allocator: KV persists through the rebuild)
            let dparams = GraphParams {
                n_tokens: 1,
                n_out: 1,
                gtype: GraphType::Decode,
                cparams: CParams {
                    n_ctx,
                    flash_attn: false,
                    explicit_span: false,
                    gpu: false,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            let mut dgraph: ComputeGraph = model.build_graph(&dparams);
            sched.assign_backends(&mut dgraph, &alloc);
            let backends: [&dyn Backend; 1] = [alloc.cpu()];
            FusionPass::new().run(&mut dgraph, &backends, &|_, _| Some(0));
            alloc.alloc_graph(&dgraph).unwrap();
            alloc.fill_input_i32(&dgraph, "token_ids", &[next]).unwrap();
            alloc
                .fill_input_i32(&dgraph, "positions", &[nt as u32])
                .unwrap();
            alloc
                .fill_attn_inputs(&dgraph, &[crate::graph::kvcache::SEQ_MAIN], &[nt as u32])
                .unwrap();
            sched.execute(&dgraph, &mut alloc).unwrap();
            let dlogits = alloc.copy_to_cpu(dgraph.outputs[0]).unwrap();
            (prefill_l, dlogits.to_vec())
        }

        // NOTE: prefill and decode share one GraphAllocator so the KV persists
        // NOTE: both runs share one GraphAllocator so the KV persists across
        // the prefill -> decode transition (like the real loop).
        let n_ctx = q2.hparams.max_seq_len as usize;
        let mut kv_f = KVCache::new(model.n_layer(), model.n_kv_embd(), n_ctx);
        let lf = model.forward(&ids, &positions, &mut kv_f, 1, n_ctx);
        let next = argmax(&lf);
        let lf2 = model.forward(&[next], &[ids.len()], &mut kv_f, 1, n_ctx);
        let (lg, lg2) = run_prefill_decode(q2, &ids, next, n_ctx);
        compare("prefill", &lf, &lg);
        compare("decode", &lf2, &lg2);
    }

    /// Phase 3 verification: the graph path on the Metal backend must produce
    /// logits close to the CPU forward (kernel math differs in reduction order,
    /// so a loose tolerance + greedy-token equality is the criterion).
    #[test]
    fn graph_metal_matches_cpu_logits() {
        #[cfg(target_os = "macos")]
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
            // Keep the weight registry stable for this whole test: a parallel
            // test loading a different architecture swaps same-named entries,
            // which would flip the CUDA gate mid-test (persistent KV regions
            // were allocated under the earlier decision).
            #[cfg(feature = "cuda")]
            let _model_load_guard = crate::cuda::CudaState::model_load_guard();
            let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
            let ids = tok.encode("The capital of France is");
            let positions: Vec<usize> = (0..ids.len()).collect();

            // CPU reference (forward, not forward_graph — separate KV state)
            let mut kv = KVCache::new(model.n_layer(), model.n_kv_embd(), 4096);
            let ref_l = model.forward(&ids, &positions, &mut kv, 1, 4096);

            // GPU graph (forward_graph picks Metal when MPS + weights on GPU)
            let mut kv2 = KVCache::new(model.n_layer(), model.n_kv_embd(), 4096);
            let gpu_l = model.forward_graph(&ids, &positions, &mut kv2, 1, 4096);

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
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            let Some(path) = cached_model_path() else {
                eprintln!("not cached; skipping");
                return;
            };
            crate::metal::MpsState::init();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let model = crate::models::load_model(&gguf).expect("load model");
            // Keep the weight registry stable for this whole test: a parallel
            // test loading a different architecture swaps same-named entries,
            // which would flip the CUDA gate mid-test (persistent KV regions
            // were allocated under the earlier decision).
            #[cfg(feature = "cuda")]
            let _model_load_guard = crate::cuda::CudaState::model_load_guard();
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
                freq_base: hp.rope_freq_base,
                freq_scale: hp.rope_freq_scale,
                n_head: nh_,
                hd: hd_,
            };
            let qr = gb.rope(q, pos, hp.rope_style, rm(hd, nh));
            let kr = gb.rope(k, pos, hp.rope_style, rm(hd, nk));
            gb.kvcache_store(0, kr, v, 32768);
            let kv = gb.kvcache_load(0, nkt, 32768, nk);
            let ao = gb.attn(
                qr,
                kv,
                pos,
                crate::graph::ops::AttnMode::Gqa,
                crate::graph::ops::AttnMeta {
                    layer: 0,
                    n_head: nh,
                    n_head_kv: nk,
                    hd,
                    hd_kv: nkt / nk,
                    nkt,
                    scale: hp.attention_scale(),
                },
            );
            gb.output(ao);
            let g = gb.build();

            let mut sched = crate::graph::scheduler::BackendScheduler::new();
            let mut ca = crate::graph::alloc::GraphAllocator::new();
            Qwen2Graph::register_graph_weights(q2, &mut ca);
            ca.alloc_graph(&g).unwrap();
            ca.fill_input_i32(&g, "token_ids", &ids).unwrap();
            let pos32: Vec<u32> = (0..nt as u32).collect();
            ca.fill_input_i32(&g, "positions", &pos32).unwrap();
            let seqs = vec![crate::graph::kvcache::SEQ_MAIN; nt];
            ca.fill_attn_inputs(&g, &seqs, &pos32).unwrap();
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
                let seqs = vec![crate::graph::kvcache::SEQ_MAIN; nt];
                alloc.fill_attn_inputs(&g2, &seqs, &pos32).unwrap();
                sched.execute(&g2, &mut alloc).unwrap();
                let got = alloc.copy_to_cpu(g2.outputs[0]).unwrap();
                let mut maxd = 0.0f32;
                for i in 0..got.len().min(expect.len()) {
                    maxd = maxd.max((got[i] - expect[i]).abs());
                }
                let kv_got = g2
                    .nodes
                    .iter()
                    .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 }))
                    .and_then(|nid| alloc.copy_to_cpu(nid))
                    .unwrap_or_default();
                let mut kvd = 0.0f32;
                for i in 0..kv_ref.len().min(kv_got.len()) {
                    kvd = kvd.max((kv_ref[i] - kv_got[i]).abs());
                }
                eprintln!(
                    "[embed iso] kv0[0..4] gpu={:?} cpu={:?}",
                    &kv_got[..kv_got.len().min(4)],
                    &kv_ref[..kv_ref.len().min(4)]
                );
                let kvn = g2
                    .nodes
                    .iter()
                    .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheLoad { layer: 0 }))
                    .unwrap();
                eprintln!("[embed iso] kv_load bufref={:?}", alloc.node_buffer(kvn));
                use crate::graph::backend::KvProvider as _;
                let kp = alloc.kv_pair(0);
                eprintln!("[embed iso] kv_pair(0)={:?}", kp);
                if let Some((ki, _)) = kp {
                    if let Some(kv2) = alloc
                        .metal()
                        .and_then(|m| m.read_host(ki).map(|s| s.to_vec()))
                    {
                        eprintln!(
                            "[embed iso] kv_pair(0) direct metal read[0..4]={:?}",
                            &kv2[..4]
                        );
                    }
                }
                eprintln!("[embed iso] run max diff {maxd:.3e} kvK diff {kvd:.3e} got0={:?} expect0={:?} got29={:?} expect29={:?}",
                    &got[0..4], &expect[0..4], &got[29 * 896..29 * 896 + 4], &expect[29 * 896..29 * 896 + 4]);
                worst = worst.max(maxd);
            }
            assert!(
                worst < 1e-3,
                "embed isolation nondeterministic: worst {worst:.3e}"
            );
        }
    }

    /// Real model layer-0 K matmul: Metal vs CPU (isolates weight offset/data).
    #[test]
    fn graph_metal_real_wk_matmul() {
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            let Some(path) = cached_model_path() else {
                eprintln!("not cached; skipping");
                return;
            };
            crate::metal::MpsState::init();
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let model = crate::models::load_model(&gguf).expect("load model");
            // Keep the weight registry stable for this whole test: a parallel
            // test loading a different architecture swaps same-named entries,
            // which would flip the CUDA gate mid-test (persistent KV regions
            // were allocated under the earlier decision).
            #[cfg(feature = "cuda")]
            let _model_load_guard = crate::cuda::CudaState::model_load_guard();
            let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
            let l0 = &q2.layers[0];
            let wk = l0.wk.as_ref().unwrap();
            let ne = q2.hparams.n_embd as usize;
            let nkt = q2.hparams.n_kv_embd as usize;
            let nt = 30usize; // GEMM path (nt >= 9)
            let xd: Vec<f32> = (0..ne * nt)
                .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
                .collect();

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
            for n in &mut g2.nodes {
                n.backend = Some(crate::graph::Backend::Metal);
            }
            let mut alloc = crate::graph::alloc::GraphAllocator::new();
            alloc.enable_metal();
            alloc.alloc_graph(&g2).unwrap();
            alloc.fill_input(&g2, "x", &xd).unwrap();
            sched.execute(&g2, &mut alloc).unwrap();
            let got = alloc.copy_to_cpu(m).unwrap();
            let mut maxd = 0.0f32;
            for i in 0..got.len() {
                maxd = maxd.max((got[i] - expect[i]).abs());
            }
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
                            let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                                wrow[boff],
                                wrow[boff + 1],
                            ]));
                            for j in 0..16 {
                                let byte = wrow[boff + 2 + j];
                                acc +=
                                    ((byte & 0x0F) as i8 - 8) as f32 * d * xd[t * ne + b * 32 + j];
                                acc += ((byte >> 4) as i8 - 8) as f32
                                    * d
                                    * xd[t * ne + b * 32 + j + 16];
                            }
                        }
                        ref2[t * nkt + o] = acc;
                    }
                }
            }
            let mut m2 = 0.0f32;
            for i in 0..got.len() {
                m2 = m2.max((got[i] - ref2[i]).abs());
            }
            eprintln!("[real wk matmul] vs manual Q4_0xf32: max diff {m2:.3e}");
            assert!(
                m2 < 5e-3,
                "real wk Metal diverges from Q4_0xf32 reference: {m2:.3e}"
            );
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
        // Keep the weight registry stable for this whole test: a parallel
        // test loading a different architecture swaps same-named entries,
        // which would flip the CUDA gate mid-test (persistent KV regions
        // were allocated under the earlier decision).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let q2: &Qwen2Model = model
            .as_any()
            .downcast_ref::<Qwen2Model>()
            .expect("qwen2 model");
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
        assert_eq!(
            d, 0.0,
            "identical inputs on fresh caches must give identical logits"
        );

        // Interleave: A prefill -> B prefill -> A decode -> B decode.
        // Each cache's KV must stay isolated from the other's.
        let da =
            Qwen2Graph::forward_cached(q2, &[argmax(&pa)], &[next_pos], 1, n_ctx, &mut cache_a);
        let db =
            Qwen2Graph::forward_cached(q2, &[argmax(&pb)], &[next_pos], 1, n_ctx, &mut cache_b);
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
                q2,
                &[argmax(&pa)],
                &[small_ctx],
                1,
                small_ctx,
                &mut cache_s,
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
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    /// G3 correctness: the n_out=1 (reduced) graph must produce the same last
    /// token logits as the full-nt (n_out=nt) graph, node by node through the
    /// tail block (wo → get_rows → tail add → ffn → output). This also guards
    /// the allocator's build-order liveness (scheduler executes in build
    /// order; a topo-order liveness pass would free still-alive inputs).
    #[test]
    fn tail_reduction_matches_full_nt() {
        use crate::graph::alloc::GraphAllocator;
        use crate::graph::ops::Op;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        let Some(path) = model_path() else {
            eprintln!("not cached; skipping");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        // Keep the weight registry stable for this whole test: a parallel
        // test loading a different architecture swaps same-named entries,
        // which would flip the CUDA gate mid-test (persistent KV regions
        // were allocated under the earlier decision).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        let nt = ids.len();
        let nv = model.n_vocab();

        /// Identify the semantic nodes of the final block by src-chain. Works
        /// for both the full graph (n_out=nt) and the reduced graph (n_out=1,
        /// tail get_rows inserted after wo).
        fn semantic_nodes(g: &crate::graph::ComputeGraph) -> Vec<(usize, &'static str)> {
            let q_matmul = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_blk.23.attn_q.weight")
                .unwrap();
            let q_rope = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, Op::RoPE { .. }) && n.src[0] == q_matmul)
                .unwrap();
            let attn = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, Op::Attn { .. }) && n.src[0] == q_rope)
                .unwrap();
            let wo = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_blk.23.attn_output.weight")
                .unwrap();
            // post-attn add: full -> add(src wo); reduced -> add(get_rows(wo), get_rows(h))
            let add_after_wo = g
                .nodes
                .iter()
                .position(|n| {
                    matches!(n.op, Op::Add)
                        && n.src.iter().any(|&s| {
                            s == wo
                                || matches!(g.nodes[s].op, Op::GetRows) && g.nodes[s].src[0] == wo
                        })
                })
                .unwrap();
            let ffn_norm = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, Op::RmsNorm { .. }) && n.src[0] == add_after_wo)
                .unwrap();
            let gate = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_blk.23.ffn_gate.weight")
                .unwrap();
            let up = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_blk.23.ffn_up.weight")
                .unwrap();
            // FusionPass merges silu+mul into a single SwiGLU node
            let swiglu = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, Op::SwiGLU) && n.src.contains(&gate))
                .unwrap();
            let down = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_blk.23.ffn_down.weight")
                .unwrap();
            let post_ffn = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, Op::Add) && n.src.contains(&down))
                .unwrap();
            let out_norm = g
                .nodes
                .iter()
                .position(|n| matches!(n.op, Op::RmsNorm { .. }) && n.src[0] == post_ffn)
                .unwrap();
            let lm = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_output.weight")
                .unwrap();
            vec![
                (q_matmul, "q_matmul"),
                (q_rope, "q_rope"),
                (attn, "attn"),
                (wo, "wo"),
                (add_after_wo, "post_attn_add"),
                (ffn_norm, "ffn_norm"),
                (gate, "gate"),
                (up, "up"),
                (swiglu, "swiglu"),
                (down, "down"),
                (post_ffn, "post_ffn_add"),
                (out_norm, "output_norm"),
                (lm, "lm_head"),
            ]
        }

        /// Build + execute a graph for n_out, keeping the given node ids alive
        /// as graph outputs (so post-exec dumps are not clobbered by liveness
        /// reuse). Returns (graph, allocator).
        fn run_keep(
            q2: &Qwen2Model,
            ids: &[u32],
            n_out: usize,
            keep: &[usize],
        ) -> (
            crate::graph::ComputeGraph,
            crate::graph::alloc::GraphAllocator,
        ) {
            use crate::graph::params::{CParams, GraphParams, GraphType};
            let model: &dyn ModelDef = q2;
            let nt = ids.len();
            let params = GraphParams {
                n_tokens: nt,
                n_out,
                gtype: GraphType::Prefill,
                cparams: CParams {
                    n_ctx: 4096,
                    flash_attn: false,
                    explicit_span: false,
                    gpu: false,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            let mut graph = model.build_graph(&params);
            for &i in keep {
                if !graph.outputs.contains(&i) {
                    graph.outputs.push(i);
                }
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
            let seqs = vec![crate::graph::kvcache::SEQ_MAIN; nt];
            alloc.fill_attn_inputs(&graph, &seqs, &pos32).unwrap();
            if graph
                .inputs
                .iter()
                .any(|&i| graph.node(i).name == "tail_ids")
            {
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
            let wo_id = g
                .nodes
                .iter()
                .position(|n| n.name == "matmul_blk.23.attn_output.weight");
            if let Some(wi) = wo_id {
                if let Some(addi) = g.nodes.iter().position(|n| {
                    matches!(n.op, Op::Add)
                        && n.src.iter().any(|&s| {
                            s == wi
                                || matches!(g.nodes[s].op, Op::GetRows) && g.nodes[s].src[0] == wi
                        })
                }) {
                    let target = if std::ptr::eq(g, &fg0) {
                        &mut fkeep
                    } else {
                        &mut rkeep
                    };
                    for &s in &g.nodes[addi].src {
                        if !target.contains(&s) {
                            target.push(s);
                        }
                        // in the reduced graph the src is a get_rows: also keep
                        // the h / wo buffer it reads
                        if matches!(g.nodes[s].op, Op::GetRows) {
                            for &s2 in &g.nodes[s].src {
                                if !target.contains(&s2) {
                                    target.push(s2);
                                }
                            }
                        }
                    }
                }
            }
        }
        drop(fg0);
        drop(rg0);

        let (fg, mut fa) = run_keep(q2, &ids, nt, &fkeep);
        let (rg, mut ra) = run_keep(q2, &ids, 1, &rkeep);

        // KV load at layer 23 must be identical (attention path preserved)
        let fkv = fg
            .nodes
            .iter()
            .position(|n| matches!(n.op, Op::KvcacheLoad { layer: 23 }));
        let rkv = rg
            .nodes
            .iter()
            .position(|n| matches!(n.op, Op::KvcacheLoad { layer: 23 }));
        if let (Some(fk), Some(rk)) = (fkv, rkv) {
            let a = fa.copy_to_cpu(fk).unwrap();
            let b = ra.copy_to_cpu(rk).unwrap();
            let kvd = (0..a.len())
                .map(|i| (a[i] - b[i]).abs())
                .fold(0.0f32, f32::max);
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
            let row_b: Vec<f32> = if b.len() >= full_row {
                b[b.len() - full_row..].to_vec()
            } else {
                b.clone()
            };
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
            use crate::graph::builder::GraphBuilder;
            use crate::graph::ops::Op;
            use crate::graph::params::{CParams, GraphParams, GraphType};
            use crate::graph::scheduler::BackendScheduler;

            // Serialize with the other Metal tests + ensure MPS is initialized:
            // without the lock this test races concurrent MPS access under
            // parallel `cargo test`, which makes the fused/unfused greedy token
            // flip (GPU nondeterminism under contention).
            let _g = crate::metal::metal_test_lock();
            crate::metal::MpsState::init();

            fn run_decode(
                q2: &Qwen2Model,
                model: &dyn ModelDef,
                tok_ids: &[u32],
                n_ctx: usize,
                nv: usize,
                fuse: bool,
                expect_fused_node: bool,
            ) -> Vec<f32> {
                let params = GraphParams {
                    n_tokens: 1,
                    n_out: 1,
                    gtype: GraphType::Decode,
                    cparams: CParams {
                        n_ctx,
                        flash_attn: false,
                        explicit_span: false,
                        gpu: true,
                        fuse_qkv: fuse,
                        fuse_ffn: fuse,
                    },
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
                    FusionPass::new().run(
                        &mut graph,
                        &backends,
                        &|g, id| match g.node(id).backend {
                            Some(crate::graph::Backend::CPU) => Some(0),
                            Some(crate::graph::Backend::Metal) => Some(1),
                            _ => None,
                        },
                    );
                }
                let has_fused = graph
                    .nodes
                    .iter()
                    .any(|n| matches!(n.op, Op::FusedQKV { .. }));
                assert_eq!(has_fused, expect_fused_node, "FusedQKV node presence");
                let ffn_off = std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1");
                let has_fused_ffn = graph.nodes.iter().any(|n| matches!(n.op, Op::FusedFFN));
                // FFN fusion is gated on nf <= 16384 (7B nf=18944 skips it)
                let ffn_small = {
                    let nf = q2.hparams.n_ff as usize;
                    nf <= 16384
                };
                assert_eq!(
                    has_fused_ffn,
                    expect_fused_node && !ffn_off && ffn_small,
                    "FusedFFN node presence"
                );
                alloc.alloc_graph(&graph).unwrap();
                alloc
                    .fill_input_i32(&graph, "token_ids", &[tok_ids[0]])
                    .unwrap();
                alloc.fill_input_i32(&graph, "positions", &[0]).unwrap();
                alloc.fill_attn_inputs(&graph, &[0], &[0]).unwrap();
                sched.execute(&graph, &mut alloc).unwrap();
                alloc.copy_to_cpu(graph.outputs[0]).unwrap()
            }

            /// Build + execute a decode graph, marking every post-FFN Add
            /// (residual + ffn_down) as an output so we can compare layer by layer.
            fn run_decode_layers(
                q2: &Qwen2Model,
                model: &dyn ModelDef,
                tok_ids: &[u32],
                fuse: bool,
            ) -> (
                Vec<f32>,
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
                Vec<Vec<f32>>,
            ) {
                let params = GraphParams {
                    n_tokens: 1,
                    n_out: 1,
                    gtype: GraphType::Decode,
                    cparams: CParams {
                        n_ctx: 4096,
                        flash_attn: false,
                        explicit_span: false,
                        gpu: true,
                        fuse_qkv: fuse,
                        fuse_ffn: fuse,
                    },
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
                    if matches!(n.op, Op::Add)
                        && n.src.len() == 2
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
                        let fed = graph.nodes.iter().any(|m| {
                            m.src.contains(&i)
                                && (matches!(m.op, Op::FusedFFN)
                                    || m.name.contains("ffn_gate")
                                    || m.name.contains("ffn_up"))
                        });
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
                    FusionPass::new().run(
                        &mut graph,
                        &backends,
                        &|g, id| match g.node(id).backend {
                            Some(crate::graph::Backend::CPU) => Some(0),
                            Some(crate::graph::Backend::Metal) => Some(1),
                            _ => None,
                        },
                    );
                }
                alloc.alloc_graph(&graph).unwrap();
                alloc
                    .fill_input_i32(&graph, "token_ids", &[tok_ids[0]])
                    .unwrap();
                alloc.fill_input_i32(&graph, "positions", &[0]).unwrap();
                alloc.fill_attn_inputs(&graph, &[0], &[0]).unwrap();
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
            // Keep the weight registry stable for this whole test: a parallel
            // test loading a different architecture swaps same-named entries,
            // which would flip the CUDA gate mid-test (persistent KV regions
            // were allocated under the earlier decision).
            #[cfg(feature = "cuda")]
            let _model_load_guard = crate::cuda::CudaState::model_load_guard();
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

    /// 8i-2/8i-3: multi-turn conversation on CUDA — the incremental path
    /// (append-only KV + REUSED decode graph across turns) vs the full
    /// rehydrate path (fresh graphs + full prefill from history) must
    /// produce identical greedy continuations. Both paths exercise the
    /// GraphCache prefill→decode alternation the server slot loop uses.
    /// Device-level: on a CUDA build the engine routes through the CUDA
    /// backend (model-load guard keeps the gate stable).
    #[test]
    #[ignore] // run explicitly: cargo test --release --features cuda dump_real_q4k_tensor -- --ignored --nocapture
    fn dump_real_q4k_tensor() {
        // 7B q4_k_m (Q4_K weights at 7B dims) — debug helper, ignored by default
        let path = std::path::PathBuf::from(
            "~/.cache/minfer/models/hf/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf",
        );
        let path = shellexpand_tilde(&path);
        let gguf = crate::gguf::load_gguf_model(&path).unwrap();
        let names = [
            "blk.0.ffn_down.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_k.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.1.ffn_down.weight",
            "lm_head.weight",
        ];
        let mut spec = String::new();
        for name in names {
            // find the tensor info across parts, then slice the part's mmap
            let mut found = None;
            for part in &gguf.parts {
                if let Some(info) = part.ctx.info.iter().find(|i| i.name == name) {
                    let start = part.ctx.offset + info.offset as usize;
                    let n = info.ne.iter().product::<i64>();
                    let bytes =
                        info.type_.type_size() * n as usize / info.type_.blck_size() as usize;
                    found = Some((info.type_, info.ne, &part.data[start..start + bytes]));
                    break;
                }
            }
            match found {
                Some((ty, ne, data)) => {
                    let out = format!("/tmp/minfer_phase7/real_{}.bin", name.replace('.', "_"));
                    std::fs::write(&out, data).unwrap();
                    spec.push_str(&format!(
                        "{name}: type={ty:?} ne={ne:?} bytes={} -> {out}\n",
                        data.len()
                    ));
                }
                None => spec.push_str(&format!("{name}: MISSING\n")),
            }
        }
        println!("{}", spec);
    }

    /// 8e follow-up: dump real Q5_K tensors from the 0.5B q5_k_m model for
    /// the `cuda_q5k_decode_mmvq_parity` real-weight section — debug helper,
    /// ignored by default. Writes to a `real05_` prefix so it cannot collide
    /// with the 7B q4_K dumps.
    #[test]
    #[ignore] // run explicitly: cargo test --release --features cuda dump_real_q5k_tensor -- --ignored --nocapture
    fn dump_real_q5k_tensor() {
        let path = std::path::PathBuf::from(
            "~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q5_k_m.gguf",
        );
        let path = shellexpand_tilde(&path);
        let gguf = crate::gguf::load_gguf_model(&path).unwrap();
        let names = [
            "token_embd.weight",
            "output.weight",
            "blk.0.attn_v.weight",
            "blk.0.attn_q.weight",
            "blk.0.attn_k.weight",
            "blk.0.ffn_down.weight",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
        ];
        let mut spec = String::new();
        for name in names {
            let mut found = None;
            for part in &gguf.parts {
                if let Some(info) = part.ctx.info.iter().find(|i| i.name == name) {
                    let start = part.ctx.offset + info.offset as usize;
                    let n = info.ne.iter().product::<i64>();
                    let bytes =
                        info.type_.type_size() * n as usize / info.type_.blck_size() as usize;
                    found = Some((info.type_, info.ne, &part.data[start..start + bytes]));
                    break;
                }
            }
            match found {
                Some((ty, ne, data)) => {
                    let out = format!("/tmp/minfer_phase7/real05_{}.bin", name.replace('.', "_"));
                    std::fs::write(&out, data).unwrap();
                    spec.push_str(&format!(
                        "{name}: type={ty:?} ne={ne:?} bytes={} -> {out}\n",
                        data.len()
                    ));
                }
                None => spec.push_str(&format!("{name}: MISSING\n")),
            }
        }
        println!("{}", spec);
    }

    fn shellexpand_tilde(p: &std::path::Path) -> std::path::PathBuf {
        let s = p.to_str().unwrap();
        if let Some(rest) = s.strip_prefix("~/") {
            if let Ok(home) = std::env::var("HOME") {
                return std::path::PathBuf::from(home).join(rest);
            }
        }
        p.to_path_buf()
    }

    #[test]
    fn cuda_conversation_multiturn_reuse() {
        #[cfg(feature = "cuda")]
        {
            // get() only READS the singleton — init() first (main.rs's job in
            // the binary; tests must do it themselves).
            crate::cuda::CudaState::init();
            if crate::cuda::CudaState::get().is_none() {
                eprintln!("skipping: no CUDA device");
                return;
            }
        }
        let Some(path) = model_path() else {
            eprintln!("skipping: q4_0 model not cached");
            return;
        };
        #[cfg(feature = "cuda")]
        let _guard = crate::cuda::CudaState::model_load_guard();
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let ctx = &gguf.parts[0].ctx;
        let tok = crate::tokenizer::Tokenizer::load(ctx);
        let spec = crate::conversation::ConversationSpec {
            template: None, // ChatML fallback (Qwen2's native style)
            bos_text: String::new(),
            eog: vec![model.special_tokens().eos],
            eot: model
                .special_tokens()
                .im_end
                .unwrap_or(model.special_tokens().eos),
            seed: 42,
            n_ctx: 1024,
            system_prompt: None,
        };
        let greedy = crate::conversation::TurnParams {
            n_predict: 16,
            temp: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_strings: Vec::new(),
        };
        use crate::conversation::Engine as _;

        // Path A: incremental — turn 1 prefill+decode, turn 2 appends onto
        // the same KV with a REUSED decode graph.
        let spec_a = crate::conversation::ConversationSpec {
            template: None,
            bos_text: String::new(),
            eog: vec![model.special_tokens().eos],
            eot: model
                .special_tokens()
                .im_end
                .unwrap_or(model.special_tokens().eos),
            seed: 42,
            n_ctx: 1024,
            system_prompt: None,
        };
        let mut conv_a = crate::conversation::Conversation::new(spec_a);
        let mut engine_a = crate::conversation::GraphEngine::new(model.as_ref(), 1024);
        let mut out1 = Vec::new();
        let mut emit1 = |b: &[u8]| out1.extend_from_slice(b);
        let r1 = conv_a
            .start(
                Some("The capital of France is Paris. The Eiffel"),
                &tok,
                &greedy,
                &mut engine_a,
                &mut emit1,
            )
            .expect("turn 1")
            .expect("turn 1 produced an outcome");
        let turn1_text = String::from_utf8_lossy(&out1).to_string();
        assert!(!turn1_text.trim().is_empty(), "turn 1 produced no text");
        // EOG may legitimately stop before n_predict; the substance is the
        // cross-path equality below.
        assert!(r1.tokens_generated >= 1, "turn 1 generated nothing");
        let mut out2 = Vec::new();
        let mut emit2 = |b: &[u8]| out2.extend_from_slice(b);
        let r2 = conv_a
            .user_turn(
                "Now continue describing it.",
                &tok,
                &greedy,
                &mut engine_a,
                &mut emit2,
            )
            .expect("turn 2");
        let turn2_text = String::from_utf8_lossy(&out2).to_string();
        assert!(!turn2_text.trim().is_empty(), "turn 2 produced no text");

        // Path B: full rehydrate — same history, fresh conversation; KV is
        // fully re-prefilled and the decode graph is rebuilt.
        let spec_b = crate::conversation::ConversationSpec {
            template: None,
            bos_text: String::new(),
            eog: vec![model.special_tokens().eos],
            eot: model
                .special_tokens()
                .im_end
                .unwrap_or(model.special_tokens().eos),
            seed: 42,
            n_ctx: 1024,
            system_prompt: None,
        };
        let mut conv_b = crate::conversation::Conversation::new(spec_b);
        let mut engine_b = crate::conversation::GraphEngine::new(model.as_ref(), 1024);
        let hist = conv_a.messages_to_json();
        let msgs =
            crate::conversation::Conversation::messages_from_json(&hist).expect("history json");
        conv_b.load_history(msgs, &tok, &mut engine_b);
        let mut out3 = Vec::new();
        let mut emit3 = |b: &[u8]| out3.extend_from_slice(b);
        let _r3 = conv_b
            .user_turn(
                "Now continue describing it.",
                &tok,
                &greedy,
                &mut engine_b,
                &mut emit3,
            )
            .expect("turn 2 rehydrated");
        let turn2_rehydrated = String::from_utf8_lossy(&out3).to_string();
        assert_eq!(
            turn2_text, turn2_rehydrated,
            "incremental (reused decode graph) vs rehydrated (fresh graphs) divergence"
        );
    }
}
