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
//! - KV cache lives in the graph allocator's persistent regions (the allocator
//!   owns it; the pre-[#252] legacy `KVCache` argument is gone).
//! - `weights_version` is static (1) until LoRA support lands.
//!
//! [#252]: https://github.com/yusiwen/minfer/issues/252

use std::sync::{Mutex, OnceLock};

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
        // C8b S2: the same flag the caller set for this device — a sharing
        // sequence's window is a list of cell runs, not one range.
        b.set_kv_map(params.cparams.kv_map);
        // C4 per-engine (issue #99): the storage format of this graph's KV regions
        // is a parameter of the build, not a process global — it decides each KV
        // node's cell width, so it is part of the reuse identity (`CParams`).
        b.set_kv_format(params.cparams.kv_format);

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
            // E5: tag this block's nodes so the assignment pass can keep a non-offloaded
            // block off the device, and decide here (not per node) whether this block's
            // device-only fused forms are available at all.
            b.set_layer(Some(il));
            let layer_gpu = params.cparams.gpu && il < params.cparams.gpu_layers;
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
            // #144 item 1: a packed cache builds the fused epilogue too — CUDA's
            // `attn_bias_rope_store_q8_0` gives one thread a whole (head,
            // 32-element K block) and one V block, so it can quantize a cell's
            // blocks in place. Pre-#144 the gate read `&& !b.kv_is_packed()` and a
            // Q8_0 decode ran the unfused bias/rope/store chain.
            //
            // #310: that packed fused epilogue is CUDA-only. Metal's fused kernel
            // is f32/f16-only, so a packed Metal cache takes the unfused chain
            // (`qkv_epilogue_ok` is false without the CUDA feature), exactly as
            // Qwen3's `fused_qkv_norm` already gates on `!b.kv_is_packed()`.
            // class 2 is CUDA-only: on macOS (feature off) the mixed-quant
            // layers keep the unfused chain, bitwise-neutral vs pre-D3-8.
            #[cfg(feature = "cuda")]
            let qkv_epilogue_ok = crate::cuda::CudaState::get().is_some();
            #[cfg(not(feature = "cuda"))]
            let qkv_epilogue_ok = false;
            let fuse_qkv = nt == 1
                && layer_gpu
                && params.cparams.fuse_qkv
                && (!b.kv_is_packed() || qkv_epilogue_ok)
                && l.bq.is_some()
                && l.bk.is_some()
                && l.bv.is_some();
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
                        row_elems: params.cparams.kv_format.row_elems(nkt),
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
                        row_elems: params.cparams.kv_format.row_elems(nkt),
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
                && layer_gpu
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
        // E5: everything from here on is outside any block (the final norm and `lm_head`);
        // a partial plan keeps those on the CPU (`OffloadPlan::device_holds_unblocked`).
        b.set_layer(None);

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
    /// KV is owned by the graph (persistent regions), not by the caller.
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
        // C8b S2/S4: a sequence that reads part of its prefix in place needs the
        // window as a list of cell runs rather than one range. Only a device whose
        // kernel can gather a map is asked (CPU, CUDA and, since #362, Metal), and
        // *whether* to share is the caller's decision — this only reflects it,
        // read from the reservations the cache already holds.
        let kv_map = device.gathers_attn_map()
            && batch.groups().iter().any(|&(seq, _, _)| {
                cache
                    .alloc()
                    .kv_seq_slot(seq)
                    .map(|s| s.shared.rows > 0)
                    .unwrap_or(false)
            });
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
                // C8b S2: the window as a list of cell runs. The caller sets it on
                // a `CParams` only for a device that can gather a map
                // (`Device::gathers_attn_map`).
                kv_map,
                gpu: metal_on || cuda_on,
                // E5: how many blocks the device holds. With no device this is 0 (nothing
                // is offloaded, and the field is inert), otherwise it is the plan the load
                // settled on — part of `CParams` because the *assignment* is topology.
                gpu_layers: if metal_on || cuda_on {
                    model.offload.plan.gpu_layers
                } else {
                    0
                },
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
                // (`explicit_span`). Metal's fused epilogue stores at a single
                // host-read `pos` (the pre-C6 form), so Metal keeps the
                // `!explicit_span` gate even though its attention *read* side
                // now handles an explicit span (#44 part (a), 2026-10-06): a
                // fused Metal store at `pos != cell` would write the wrong row.
                fuse_qkv: nt == 1
                    && (metal_on || cuda_on)
                    && (cuda_on || !explicit_span)
                    && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1"),
                fuse_ffn: nt == 1
                    && (metal_on || cuda_on)
                    && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1"),
                // C4 per-engine (issue #99): this engine's resolved KV format. It
                // sizes every KV node's cell and is part of the reuse identity, so a
                // cached graph built for one format is never reused for another.
                kv_format: model.kv_format,
            },
            weights_version: 1,
        };

        // C4 per-engine (issue #99): hand this engine's resolved format to the
        // allocator's CPU kernels. The graph nodes above already carry the same
        // format, so the region width and the store/attention dispatch cannot
        // disagree — and no other engine in the process can change either.
        cache.alloc().set_kv_format(model.kv_format);

        if !cache
            .try_reuse(&params)
            .expect("re-map onto a cached graph")
        {
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
                // E5: put the plan in force before assignment — it is what keeps a
                // non-offloaded block's nodes off the device (a device-only fused node is
                // never even built there, but every ordinary op still has to be placed).
                alloc.set_offload_plan(Some(model.offload.plan));
                sched.assign_backends(&mut graph, alloc);
                // fusion pass gated per node's assigned backend.
                //
                // F4: the backend list and the node → index map come from the
                // allocator's registry view (the enabled entries in identity
                // order), replacing the hand-built vector and its `name() ==
                // "cuda"` position lookup.
                let backends: Vec<&dyn Backend> = alloc.fusion_backends();
                FusionPass::new().run(&mut graph, &backends, &|g, id| {
                    g.node(id)
                        .backend
                        .and_then(|b| alloc.fusion_backend_index(b))
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
            // F4: a backend the run fenced off (`--backend` / `MINFER_BACKENDS`)
            // does not participate, so the builder never emits a device-only
            // fused node the CPU cannot execute. The fence is the same one
            // `GraphAllocator::supports_for` reads, so the two cannot disagree.
            let filter = crate::graph::registry::active_filter();
            #[cfg(target_os = "macos")]
            if filter.allows(crate::graph::Backend::METAL)
                && crate::graph::metal_backend::metal_available()
                && Self::weights_on_gpu(model)
            {
                return crate::models::Device::Metal;
            }
            // CUDA participation (Phase 7): requires a usable device AND every
            // matmul weight registered on the CUDA registry in a kernel-supported
            // type (all-or-nothing; 7e③ moved the embedding gather on device, so
            // tok_embd is gated like every other weight).
            #[cfg(feature = "cuda")]
            if filter.allows(crate::graph::Backend::CUDA)
                && crate::cuda::CudaState::get().is_some()
                && Self::weights_on_cuda(model)
            {
                return crate::models::Device::Cuda;
            }
        }
        #[cfg(not(any(target_os = "macos", feature = "cuda")))]
        let _ = model;
        crate::models::Device::Cpu
    }

    /// Every weight the graph reads must be GPU-registered for the Metal path.
    ///
    /// E5: "the graph reads" means the **offloaded** blocks, and the unblocked tensors only
    /// under a full plan — the loader did not register anything else for a partial plan.
    #[cfg(target_os = "macos")]
    fn weights_on_gpu(model: &Qwen2Model) -> bool {
        let plan = model.offload.plan;
        if plan.is_cpu_only() {
            return false;
        }
        let names: Vec<String> = {
            let mut v = Vec::new();
            if plan.device_holds_unblocked() {
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
            }
            for (i, l) in model.layers.iter().enumerate() {
                if !plan.on_device(i) {
                    continue;
                }
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
    /// with the matching kernel — matmuls cover Q4_0/Q4_1/Q5_0/Q5_1/Q8_0, the
    /// K-quants and the 2 B/element f16/bf16 weights; embed gathers every type
    /// EXCEPT Q4_1 (no embed kernel). The loader registers some unsupported
    /// types for the legacy path, so the type check is required here.
    /// Norm/bias weights are f32 and only need registration.
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
                | TensorType::F16
                // #208 (CUDA half): the loader registers bf16 raw and
                // `bf16_f32_matmul_vec` / `_scalar` are its device kernels. Both
                // supported architectures list it (qwen3's twin below), because
                // the shared registration rule (`models::weight_reg`) admits it
                // for both.
                | TensorType::BF16
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
                | TensorType::F16
                // #208 (CUDA half): `embed_rows_bf16` is the device gather.
                | TensorType::BF16
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
        // E5: only the **offloaded** blocks have to be usable on the device, and the
        // unblocked tensors only under a full plan — the loader deliberately did not
        // register anything else, so checking them would always fail a partial plan.
        let plan = model.offload.plan;
        if plan.is_cpu_only() {
            return false;
        }
        let mut ok = true;
        if plan.device_holds_unblocked() {
            ok = embed_ok(&model.tok_embd, &cuda)
                && matmul_ok(&model.output, &cuda)
                && registered(&model.output_norm, &cuda)
                && registered(&model.output_b, &cuda);
        }
        for (i, l) in model.layers.iter().enumerate() {
            if !plan.on_device(i) {
                continue; // this block runs on the CPU; its weights are not registered
            }
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
            // The same weights the check above looked at, in the same order (0 = embed,
            // 1 = matmul, 2 = registered-only): only a partial plan leaves the unblocked
            // tensors and the CPU-side blocks out.
            let mut named: Vec<(&Option<crate::tensor::Tensor>, u8)> = Vec::new();
            if plan.device_holds_unblocked() {
                named.extend([
                    (&model.tok_embd, 0u8),
                    (&model.output, 1u8),
                    (&model.output_norm, 2u8),
                    (&model.output_b, 2u8),
                ]);
            }
            for (i, l) in model.layers.iter().enumerate() {
                if !plan.on_device(i) {
                    continue;
                }
                named.extend([
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
                ]);
            }
            let fail = named.into_iter().find(|(t, kind)| match (t, kind) {
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
mod tests;

#[cfg(test)]
mod tail_tests;
