//! Qwen3 (dense) compute-graph construction + graph forward path.
//!
//! Mirrors `qwen2/graph.rs` with the two Qwen3 deltas
//! (docs/QWEN3-SUPPORT-PLAN.md §2):
//!   1. head dim `hd` comes from `hparams.n_embd_head` (128), not
//!      `n_embd / n_head` (64) — this drives the Q/K/V/wo widths, the RoPE dims,
//!      the attention scale and the KV row stride.
//!   2. per-head Q/K RMSNorm (`Op::QkNorm`) after the projections, before RoPE.
//!
//! The decode fused-QKV path (`Op::FusedQKV`) is intentionally NOT used for
//! Qwen3: the `attn_bias_rope_store` kernel does bias+rope+store in one pass
//! and cannot express the per-head norm between projection and rope (Phase E
//! follow-up). The unfused 3-matmul path is correct on both backends. The
//! fused-FFN decode path (`Op::FusedFFN`) IS reused (FFN is Qwen2-identical).

use std::sync::{Mutex, OnceLock};

use crate::cache::KVCache;
use crate::graph::alloc::GraphAllocator;
use crate::graph::backend::Backend;
use crate::graph::cache::GraphCache;
use crate::graph::fusion::FusionPass;
use crate::graph::ops::{AttnMeta, AttnMode, FusedQkvNormMeta, RoPEMeta};
use crate::graph::params::{CParams, GraphParams, GraphType};
use crate::graph::scheduler::BackendScheduler;
use crate::graph::ComputeGraph;

use super::Qwen3Model;

/// Qwen3 graph construction + execution.
pub struct Qwen3Graph;

impl Qwen3Graph {
    /// Build the declarative graph for one forward step (deterministic in
    /// `params` — the reuse invariant).
    pub fn build(model: &Qwen3Model, params: &GraphParams) -> ComputeGraph {
        let hp = &model.hparams;
        let nt = params.n_tokens;
        let ne = hp.n_embd as usize;
        let nh = hp.n_head as usize;
        let nk = hp.n_head_kv as usize;
        let hd = hp.n_embd_head() as usize; // 128 (decoupled; NOT ne/nh = 64)
        let nkt = hp.n_kv_embd as usize; // 1024
        let hd_kv = nkt / nk; // 128
        let nf = hp.n_ff as usize;
        let eps = hp.f_norm_rms_eps;
        let n_ctx = params.cparams.n_ctx;

        let mut b = crate::graph::builder::GraphBuilder::new();

        let inp_ids = b.input("token_ids", [nt, 1, 1, 1], crate::graph::DType::I32);
        let inp_pos = b.input("positions", [nt, 1, 1, 1], crate::graph::DType::I32);

        let mut h = b.embedding(inp_ids, model.tok_embd.as_ref().unwrap());

        let mode = if params.cparams.flash_attn {
            AttnMode::Flash
        } else {
            AttnMode::Gqa
        };
        let attn_scale = hp.attention_scale(); // 1/sqrt(128)

        for (il, l) in model.layers.iter().enumerate() {
            let residual = h;

            // pre-norm
            let normed = b.rms_norm(h, l.attn_norm.as_ref(), eps);

            // Q/K/V projections (no biases in Qwen3) + per-head Q/K RMSNorm +
            // RoPE over the full head dim. decode (nt==1) with GPU QKV concat
            // uses the fused path (Op::FusedQkvNorm): one concat matmul + a
            // fused per-head norm + no-bias rope+store, replacing 3 matmul +
            // 2 qk_norm + 2 rope + 2 store dispatches. Qwen3 has no attention
            // biases, so the fused kernel applies the per-head norm (which the
            // Qwen2 bias+rope+store kernel cannot express).
            let fuse_qkv_norm = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_qkv
                && l.q_norm.is_some()
                && l.k_norm.is_some()
                && Self::qkv_concat_available(&l.wq, &l.wk, &l.wv);
            let (q, kv) = if fuse_qkv_norm {
                let qkv = b.fused_qkv_norm(
                    normed,
                    inp_pos,
                    il,
                    FusedQkvNormMeta {
                        qkv_weight: format!("blk.{il}.attn_qkv"),
                        q_norm_name: l.q_norm.as_ref().map(|t| t.name.clone()),
                        k_norm_name: l.k_norm.as_ref().map(|t| t.name.clone()),
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
                        eps,
                    },
                );
                // q lives at concat offset 0 (rows 0..nqt); K/V went into the
                // persistent regions via the fused store — read them back.
                let kv = b.kvcache_load(il, nkt, n_ctx, nk);
                (qkv, kv)
            } else {
                let q = b.matmul(normed, l.wq.as_ref().unwrap(), None);
                let k = b.matmul(normed, l.wk.as_ref().unwrap(), None);
                let v = b.matmul(normed, l.wv.as_ref().unwrap(), None);

                // Qwen3: per-head RMSNorm on Q and K before RoPE (llama.cpp
                // `build_norm(Qcur, attn_q_norm, ...)` on the [hd, n_head, nt]
                // reshaped buffer — contiguous rows in our flat layout).
                let q = b.qk_norm(q, l.q_norm.as_ref(), hd, nh, eps);
                let k = b.qk_norm(k, l.k_norm.as_ref(), hd, nk, eps);

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

                b.kvcache_store(il, k, v, inp_pos, n_ctx);
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
            if is_last && params.n_out < nt {
                // G3: reduce to the tail n_out rows BEFORE the last layer's FFN
                let tail_ids = b.input(
                    "tail_ids",
                    [params.n_out, 1, 1, 1],
                    crate::graph::DType::I32,
                );
                let cur_tail = b.get_rows(wo, tail_ids, [ne, params.n_out, 1, 1]);
                let res_tail = b.get_rows(residual, tail_ids, [ne, params.n_out, 1, 1]);
                h = b.add(res_tail, cur_tail);
            } else {
                h = b.add(residual, wo);
            }

            // FFN (SwiGLU); fused gate+up decode path reused from Qwen2
            // (nf = 3072 ≤ 16384 gate; FFN is Qwen2-identical). Decoupled from
            // the QKV fusion gate (`fuse_qkv`) so A/B-ing the QKV fusion
            // (MINFER_NO_FUSE_QKV) does not also flip the FFN fusion.
            // fuse_ffn lives in CParams (the reuse identity) so the
            // MINFER_NO_FUSE_FFN toggle reliably forces a rebuild.
            let fuse_gu = nt == 1
                && params.cparams.gpu
                && params.cparams.fuse_ffn
                && Self::gu_concat_available(&l.ffn_gate, &l.ffn_up)
                && nf <= 16384;
            let residual = h;
            let normed = b.rms_norm(h, l.ffn_norm.as_ref(), eps);
            let ffn_out = if fuse_gu {
                let gu = b.fused_ffn(
                    normed,
                    crate::graph::ops::FusedFfnMeta {
                        gu_weight: format!("blk.{il}.ffn_gu"),
                        weight_ttype: l.ffn_gate.as_ref().unwrap().ttype,
                        in_dim: ne,
                        nf,
                    },
                );
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
            crate::cuda::concat_rows(&[fg, fu]).is_some()
        }
        #[cfg(all(not(target_os = "macos"), not(feature = "cuda")))]
        {
            let _ = (fg, fu);
            false
        }
    }

    /// Register every weight the graph references on the allocator's backend.
    pub(crate) fn register_graph_weights(model: &Qwen3Model, alloc: &mut GraphAllocator) {
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
                &l.wk,
                &l.wv,
                &l.wo,
                &l.q_norm,
                &l.k_norm,
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
    /// full `max_seq_len` unconditionally (40960 on Qwen3-4B → 12.1 GB of f32
    /// KV buffers + a ~275 ms first-submit Metal tax, see
    /// docs/PERF-QWEN3-4B-VS-LLAMACPP.md §2). Server / multi-slot code must
    /// call [`Qwen3Graph::forward_cached`] with a slot-scoped cache instead.
    pub fn forward(
        model: &Qwen3Model,
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
    pub fn forward_cached(
        model: &Qwen3Model,
        tokens: &[u32],
        positions: &[usize],
        n_out: usize,
        n_ctx: usize,
        cache: &mut GraphCache,
    ) -> Vec<f32> {
        let nt = tokens.len();
        debug_assert!(n_out <= nt);
        if let Some(&maxp) = positions.iter().max() {
            assert!(
                maxp < n_ctx,
                "position {maxp} exceeds n_ctx {n_ctx} (KV region overflow)"
            );
        }
        #[cfg(target_os = "macos")]
        let metal_on =
            crate::graph::metal_backend::metal_available() && Self::weights_on_gpu(model);
        #[cfg(not(target_os = "macos"))]
        let metal_on = false;
        // CUDA participation (Phase 7): requires a usable device AND every
        // matmul weight registered on the CUDA registry in a kernel-supported
        // type (all-or-nothing; 7e③ moved the embedding gather on device, so
        // tok_embd is gated like every other weight).
        #[cfg(feature = "cuda")]
        let cuda_on = crate::cuda::CudaState::get().is_some() && Self::weights_on_cuda(model);
        #[cfg(not(feature = "cuda"))]
        let cuda_on = false;
        let params = GraphParams {
            n_tokens: nt,
            n_seqs: 1,
            n_out,
            gtype: if nt == 1 {
                GraphType::Decode
            } else {
                GraphType::Prefill
            },
            cparams: CParams {
                n_ctx,
                n_batch: nt,
                flash_attn: false,
                gpu: metal_on || cuda_on,
                // decode (nt==1) QKV fusion (Op::FusedQkvNorm — per-head Q/K norm
                // + no-bias rope+store) is part of the topology; the env toggle
                // forces a rebuild so it can be A/B'd reliably (like qwen2's
                // FusedQKV). Prefill (nt>1) never uses it.
                fuse_qkv: nt == 1
                    && metal_on
                    && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1"),
                fuse_ffn: nt == 1
                    && (metal_on || cuda_on)
                    && !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1"),
            },
            weights_version: 1,
        };

        if !cache.try_reuse(&params) {
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
            cache.replace_graph(graph, params);
        }

        let (graph, alloc) = cache.current().unwrap();

        let ids: Vec<u32> = tokens.to_vec();
        alloc.fill_input_i32(graph, "token_ids", &ids).unwrap();
        let pos: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
        alloc.fill_input_i32(graph, "positions", &pos).unwrap();
        if graph
            .inputs
            .iter()
            .any(|&i| graph.node(i).name == "tail_ids")
        {
            let tail: Vec<u32> = ((nt - n_out)..nt).map(|x| x as u32).collect();
            alloc.fill_input_i32(graph, "tail_ids", &tail).unwrap();
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
        // so GPU vs CPU graph runs can be compared (Phase D verification).
        // Decode steps are written per-position (logits_decode_{pos}.f32) so a
        // full generation can be inspected step by step.
        if let Ok(dir) = std::env::var("MINFER_GRAPH_DUMP") {
            let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
            let tag = if nt == 1 {
                format!("decode_{}", positions[0])
            } else {
                "prefill".to_string()
            };
            let _ = std::fs::write(format!("{dir}/logits_{tag}.f32"), {
                let mut b = Vec::with_capacity(logits.len() * 4);
                for x in &logits {
                    b.extend_from_slice(&x.to_le_bytes());
                }
                b
            });
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
            // intermediate nodes (layer-0 Q/K path + attention) for decode-vs-
            // prefill debugging (mirrors qwen2's node dump)
            for nid in [2usize, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
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
            eprintln!(
                "[graph dump] wrote {dir}/logits_{tag}.f32 ({} elems)",
                logits.len()
            );
        }

        let nv = model.hparams.n_vocab as usize;
        let logits = alloc.copy_to_cpu(graph.outputs[0]).expect("logits buffer");
        logits[..n_out * nv].to_vec()
    }

    /// Every weight the graph reads must be GPU-registered for the Metal path.
    #[cfg(target_os = "macos")]
    fn weights_on_gpu(model: &Qwen3Model) -> bool {
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
                    &l.wk,
                    &l.wv,
                    &l.wo,
                    &l.q_norm,
                    &l.k_norm,
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
    /// embedding a device gather+dequant kernel). Matmul weights additionally
    /// require a type with an f32-activation kernel (Q5_0/Q5_1/Q5_K/F32 have
    /// none — the loader registers them for the legacy path, so the type
    /// check is required here); tok_embd requires an embed-kernel type
    /// (F32/Q4_0/Q8_0/Q4_K/Q6_K). Norm/bias weights are f32 and only need
    /// registration; q_norm/k_norm feed Op::QkNorm on CUDA.
    #[cfg(feature = "cuda")]
    fn weights_on_cuda(model: &Qwen3Model) -> bool {
        use crate::tensor::TensorType;
        fn matmul_ok(t: &Option<crate::tensor::Tensor>, cuda: &crate::cuda::CudaState) -> bool {
            match t {
                Some(t) => {
                    matches!(t.ttype, |TensorType::Q4_0| TensorType::Q8_0
                        | TensorType::Q4_1
                        | TensorType::Q4_K
                        | TensorType::Q6_K
                        | TensorType::F32
                        | TensorType::Q5_1
                        | TensorType::Q5_K)
                        && cuda.has_weight_of_size(&t.name, t.data().len())
                }
                None => true,
            }
        }
        fn embed_ok(t: &Option<crate::tensor::Tensor>, cuda: &crate::cuda::CudaState) -> bool {
            match t {
                Some(t) => {
                    matches!(t.ttype, |TensorType::F32| TensorType::Q4_0
                        | TensorType::Q8_0
                        | TensorType::Q4_K
                        | TensorType::Q6_K
                        | TensorType::Q5_1
                        | TensorType::Q5_K)
                        && cuda.has_weight_of_size(&t.name, t.data().len())
                }
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
                && matmul_ok(&l.wk, &cuda)
                && matmul_ok(&l.wv, &cuda)
                && matmul_ok(&l.wo, &cuda)
                && registered(&l.q_norm, &cuda)
                && registered(&l.k_norm, &cuda)
                && registered(&l.ffn_norm, &cuda)
                && matmul_ok(&l.ffn_gate, &cuda)
                && matmul_ok(&l.ffn_up, &cuda)
                && matmul_ok(&l.ffn_down, &cuda);
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
    use crate::models::ModelDef;

    /// Path to the locally cached Qwen3-0.6B Q8_0.
    fn cached_model_path() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf");
        if p.exists() {
            Some(p)
        } else {
            None
        }
    }

    fn load_qwen3() -> Option<(crate::gguf::GgufModel, Qwen3Model, Vec<u32>, Vec<usize>)> {
        let path = cached_model_path()?;
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let q3: &Qwen3Model = model
            .as_any()
            .downcast_ref::<Qwen3Model>()
            .expect("qwen3 model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        assert!(!ids.is_empty());
        let positions: Vec<usize> = (0..ids.len()).collect();
        Some((gguf, q3.clone(), ids, positions))
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

    fn compare(tag: &str, a: &[f32], b: &[f32]) {
        assert_eq!(a.len(), b.len(), "[{tag}] logits length mismatch");
        let mut maxd = 0.0f32;
        for i in 0..a.len() {
            maxd = maxd.max((a[i] - b[i]).abs());
        }
        eprintln!("[{tag}] logits max abs diff: {maxd:.3e}");
        assert!(
            maxd < 1e-3,
            "[{tag}] graph logits diverge (max diff {maxd:.3e})"
        );
    }

    /// Qwen3 hermetic CPU verification (the model's forward IS the graph path,
    /// so two independent GraphCache runs — prefill + decode — must agree
    /// bit-for-bit: deterministic build/execute, params-only reuse).
    ///
    /// CPU-only by design: skipped once a Metal test has initialized MPS in
    /// this process (the loader would then register weights on the GPU and the
    /// run would exercise the Metal path instead — covered by the Metal tests).
    #[test]
    fn graph_cpu_self_consistency_real_model() {
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(target_os = "macos")]
        if crate::metal::MpsState::get().is_some() {
            eprintln!("MPS initialized by an earlier test; skipping CPU-only test");
            return;
        }
        let Some((_, q3, ids, positions)) = load_qwen3() else {
            eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
            return;
        };
        // Keep the weight registry stable for this whole test (see qwen2 tests).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let n_ctx = q3.hparams.max_seq_len as usize;
        let nt = ids.len();

        let mut cache_a = GraphCache::new();
        let mut cache_b = GraphCache::new();
        let pa = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_a);
        let pb = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_b);
        compare("prefill", &pa, &pb);

        let next = argmax(&pa);
        let da = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut cache_a);
        let db = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut cache_b);
        compare("decode", &da, &db);
    }

    /// KV isolation between caches (mirrors qwen2): interleaving prefill/decode
    /// across two caches must not cross-contaminate, and a smaller n_ctx must
    /// not change prefill logits while positions fit. CPU-only (see above).
    #[test]
    fn forward_cached_isolates_kv_between_caches() {
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(target_os = "macos")]
        if crate::metal::MpsState::get().is_some() {
            eprintln!("MPS initialized by an earlier test; skipping CPU-only test");
            return;
        }
        let Some((_, q3, ids, positions)) = load_qwen3() else {
            eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
            return;
        };
        // Keep the weight registry stable for this whole test (see qwen2 tests).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let nt = ids.len();
        let n_ctx = q3.hparams.max_seq_len as usize;

        let mut cache_a = GraphCache::new();
        let mut cache_b = GraphCache::new();
        let pa = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_a);
        let pb = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut cache_b);
        let na = argmax(&pa);
        let nb = argmax(&pb);

        let da = Qwen3Graph::forward_cached(&q3, &[na], &[nt], 1, n_ctx, &mut cache_a);
        let db = Qwen3Graph::forward_cached(&q3, &[nb], &[nt], 1, n_ctx, &mut cache_b);
        assert_eq!(na, nb, "identical inputs must give the same greedy token");
        compare("interleaved decode", &da, &db);

        let small_ctx = nt + 4;
        let mut cache_s = GraphCache::new();
        let ps = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, small_ctx, &mut cache_s);
        compare("smaller n_ctx prefill", &pa, &ps);

        let oob = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Qwen3Graph::forward_cached(&q3, &[na], &[small_ctx], 1, small_ctx, &mut cache_s);
        }));
        assert!(oob.is_err(), "position >= n_ctx must be rejected");
    }

    /// Decode fused-QKV-with-per-head-norm (Op::FusedQkvNorm) must be
    /// numerically identical to the unfused path, and the fused node must be
    /// present/absent exactly per the fuse flag. Qwen3 fuses only the QKV
    /// projection chain (3 matmul + 2 qk_norm + 2 rope + 2 store → one concat
    /// matmul + fused per-head norm + no-bias rope+store); the FFN fusion is
    /// decoupled (always on when gated), so the fused/unfused graphs differ
    /// ONLY in the QKV spine, and the decode logits must match bit-for-bit
    /// (both execute the same Metal math — concat vs separate per-row matmul,
    /// and rms_norm_256 / attn_rope_store vs QkNorm / rope_f32 / store_kv).
    #[test]
    fn fused_qkv_norm_matches_unfused_decode() {
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            use crate::graph::ops::Op;
            use crate::graph::params::{CParams, GraphParams, GraphType};
            crate::metal::MpsState::init();
            let Some((_, q3, ids, positions)) = load_qwen3() else {
                eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
                return;
            };
            assert!(
                crate::graph::metal_backend::metal_available() && Qwen3Graph::weights_on_gpu(&q3),
                "Metal path must actually run (weights on GPU)"
            );
            let n_ctx = q3.hparams.max_seq_len as usize;
            let nt = ids.len();
            let nv = q3.hparams.n_vocab as usize;

            // --- node presence (build only, no execute) ---
            let fused_params = GraphParams {
                n_tokens: 1,
                n_seqs: 1,
                n_out: 1,
                gtype: GraphType::Decode,
                cparams: CParams {
                    n_ctx,
                    n_batch: 1,
                    flash_attn: false,
                    gpu: true,
                    fuse_qkv: true,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            let fg = Qwen3Graph::build(&q3, &fused_params);
            let has_fused = fg
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
            assert!(
                has_fused,
                "decode graph with fuse_qkv=true must contain FusedQkvNorm"
            );

            let unfused_params = GraphParams {
                cparams: CParams {
                    fuse_qkv: false,
                    fuse_ffn: false,
                    ..fused_params.cparams
                },
                ..fused_params
            };
            let ug = Qwen3Graph::build(&q3, &unfused_params);
            let has_unfused = ug
                .nodes
                .iter()
                .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
            assert!(
                !has_unfused,
                "decode graph with fuse_qkv=false must NOT contain FusedQkvNorm"
            );

            // --- numeric: fused (default) vs unfused (MINFER_NO_FUSE_QKV=1) ---
            let _ = std::env::remove_var("MINFER_NO_FUSE_QKV");
            let mut a = GraphCache::new();
            let pa = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut a);
            let next = argmax(&pa);
            let da = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut a);
            {
                let (g, _) = a.current().unwrap();
                let fused_in_cache = g
                    .nodes
                    .iter()
                    .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
                assert!(fused_in_cache, "the default decode graph must be fused");
            }

            std::env::set_var("MINFER_NO_FUSE_QKV", "1");
            let mut b = GraphCache::new();
            let pb = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, &mut b);
            let db = Qwen3Graph::forward_cached(&q3, &[next], &[nt], 1, n_ctx, &mut b);
            std::env::remove_var("MINFER_NO_FUSE_QKV");

            // prefill never fuses (nt>1) → both prefills must be bit-identical;
            // this isolates the decode-only divergence.
            {
                let mut pd = 0.0f32;
                for (x, y) in pa.iter().zip(pb.iter()) {
                    pd = pd.max((x - y).abs());
                }
                eprintln!("[qwen3 fused test] prefill a vs b max diff: {pd:.3e}");
                assert_eq!(pa.len(), pb.len());
                assert!(pd < 1e-5, "prefill a/b diverge (max {pd:.3e})");
            }
            {
                let (g, _) = b.current().unwrap();
                let unfused_in_cache = g
                    .nodes
                    .iter()
                    .any(|n| matches!(n.op, Op::FusedQkvNorm { .. }));
                assert!(
                    !unfused_in_cache,
                    "the MINFER_NO_FUSE_QKV=1 decode graph must be unfused"
                );
            }

            // fused and unfused decode must produce the SAME greedy token (the
            // hard correctness oracle — the graph_metal_matches_llama_reference
            // test additionally pins the whole sequence against llama.cpp), and
            // logits must agree within float noise. NOTE: unlike Qwen2's Q4_0
            // (fused-vs-unfused max diff 0.0), the Q8_0 concat matmul (od_total
            // vs separate od) is NOT bit-exact — the per-row reduction stays in
            // a different threadgroup grouping, giving ~1e-3 float noise on the
            // final logits. This does not flip the argmax (verified: greedy is
            // byte-identical, and matches llama.cpp exactly).
            assert_eq!(
                da.len(),
                nv,
                "decode logits length must be nv (=1 output token)"
            );
            assert_eq!(
                argmax(&da),
                argmax(&db),
                "fused vs unfused decode must select the same next token"
            );
            let mut maxd = 0.0f32;
            for (x, y) in da.iter().zip(db.iter()) {
                maxd = maxd.max((x - y).abs());
            }
            assert!(
                maxd < 1e-2,
                "fused vs unfused decode logits diverge (max diff {maxd:.3e})"
            );
        }
    }

    /// Metal greedy generation must reproduce the reference token sequence
    /// verified against llama.cpp (same Q8_0 GGUF, raw prompt, temp 0, no
    /// penalties — 60 tokens were byte-identical; the first 9 are pinned here
    /// as a hermetic regression oracle). Also asserts Metal is deterministic:
    /// two independent caches must produce the identical sequence.
    #[test]
    fn graph_metal_matches_llama_reference() {
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            crate::metal::MpsState::init();
            let Some((_, q3, ids, positions)) = load_qwen3() else {
                eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
                return;
            };
            assert!(
                crate::graph::metal_backend::metal_available() && Qwen3Graph::weights_on_gpu(&q3),
                "Metal path must actually run (weights on GPU)"
            );
            let n_ctx = q3.hparams.max_seq_len as usize;
            let nt = ids.len();

            // Reference greedy tokens for the raw prompt "The capital of France
            // is" (temp 0, no penalties): " Paris. The capital of France is also
            // the capital..." — verified token-for-token against llama.cpp.
            let reference: [u32; 9] = [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279];

            for (cache, tag) in [(&mut GraphCache::new(), "a"), (&mut GraphCache::new(), "b")] {
                let l = Qwen3Graph::forward_cached(&q3, &ids, &positions, 1, n_ctx, cache);
                let mut toks = vec![argmax(&l)];
                for i in 0..reference.len() - 1 {
                    let p = (nt + i) as usize;
                    let l = Qwen3Graph::forward_cached(&q3, &[toks[i]], &[p], 1, n_ctx, cache);
                    toks.push(argmax(&l));
                }
                eprintln!("[qwen3 metal {tag}] greedy={toks:?}");
                assert_eq!(
                    toks,
                    reference.to_vec(),
                    "Metal greedy diverges from the llama.cpp-verified reference (cache {tag})"
                );
            }
        }
    }

    /// Metal prefill must be bitwise deterministic across runs. Regression
    /// test for the Q8_0 multi-token matmul race (missing trailing
    /// threadgroup_barrier in `kernel_q8_0_f32_matmul_multi` zeroed shmem
    /// while slow threads were still reducing it, flipping output elements to
    /// 0 between runs). Keeps the layer-0 K-path nodes alive and compares two
    /// independent executions element by element.
    #[test]
    fn metal_prefill_determinism() {
        #[cfg(target_os = "macos")]
        let _g = crate::metal::metal_test_lock();
        #[cfg(not(target_os = "macos"))]
        {
            eprintln!("not macOS; skipping");
            return;
        }
        #[cfg(target_os = "macos")]
        {
            crate::metal::MpsState::init();
            let Some((_, q3, ids, positions)) = load_qwen3() else {
                eprintln!("Qwen3-0.6B q8_0 not cached; skipping");
                return;
            };
            let nt = ids.len();
            let params = GraphParams {
                n_tokens: nt,
                n_seqs: 1,
                n_out: 1,
                gtype: GraphType::Prefill,
                cparams: CParams {
                    n_ctx: 512,
                    n_batch: nt,
                    flash_attn: false,
                    gpu: true,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            // layer-0: embed, attn rms, q/k/v matmul, qk_norm q/k, rope q/k
            let keep = [2usize, 3, 4, 5, 6, 7, 8, 9, 10];
            let mut dumps: Vec<Vec<Vec<f32>>> = Vec::new();
            for _ in 0..2 {
                let mut graph = Qwen3Graph::build(&q3, &params);
                for &nid in &keep {
                    if !graph.outputs.contains(&nid) {
                        graph.outputs.push(nid);
                    }
                }
                let sched = BackendScheduler::new();
                let mut alloc = GraphAllocator::new();
                Qwen3Graph::register_graph_weights(&q3, &mut alloc);
                alloc.enable_metal();
                sched.assign_backends(&mut graph, &alloc);
                alloc.alloc_graph(&graph).unwrap();
                let ids32: Vec<u32> = ids.iter().copied().collect();
                alloc.fill_input_i32(&graph, "token_ids", &ids32).unwrap();
                let pos32: Vec<u32> = positions.iter().map(|&p| p as u32).collect();
                alloc.fill_input_i32(&graph, "positions", &pos32).unwrap();
                sched.execute(&graph, &mut alloc).unwrap();
                let mut run_dumps = Vec::new();
                for &nid in &keep {
                    run_dumps.push(alloc.copy_to_cpu(nid).unwrap());
                }
                dumps.push(run_dumps);
            }
            for (i, &nid) in keep.iter().enumerate() {
                let (a, b) = (&dumps[0][i], &dumps[1][i]);
                let neq = a.iter().zip(b.iter()).filter(|(x, y)| x != y).count();
                eprintln!("[determinism] node{nid}: neq={neq}");
                assert_eq!(
                    neq, 0,
                    "Metal prefill node {nid} is not deterministic across runs"
                );
            }
        }
    }
}
