//! #200: the Qwen3 fused decode QKV spine on CUDA.
//!
//! `Op::FusedQkvNorm` (one concat matmul + the per-head Q/K RMSNorm + the
//! no-bias rope/store epilogue) was Metal-only until #200. This gate is the
//! CUDA arm of the property the Metal test
//! (`fused_qkv_norm_matches_unfused_decode`) pins, and it runs it on **two
//! engines**: the default f32 KV and a packed q8_0 one, loaded through
//! `load_model_configured`'s explicit cache type — no `MINFER_CACHE_TYPE`
//! mutation, no process global (#99, #153).
//!
//! The comparison class is **bit equality**, which CUDA can demand where Metal
//! cannot: the concat matmul (od_total rows in one launch) and the three separate
//! matmuls reduce each output row with the same fixed thread mapping, independent
//! of the launch's row count, so the fused logits are the unfused chain's exactly
//! (measured 0.0 on GB10). Metal's twin records ~1e-3 for Q8_0 because its concat
//! grouping does change the per-row reduction, and there it asserts greedy
//! identity plus a 1e-2 bound instead. The kernel's own bytes are pinned one
//! level down by `cuda_qkv_norm_epilogue_matches_the_cpu_quantizer` (the store's
//! quantizer over the host's rope).
//!
//! The whole module is `#[cfg(feature = "cuda")]` (declared that way in the
//! parent): a CPU-only build has no fused CUDA arm to gate, and a no-op test
//! would only inflate the CPU suite's count.

use super::*;

/// Fused vs `MINFER_NO_FUSE_QKV=1` on the device, f32 and q8_0 KV.
///
/// `MINFER_NO_FUSE_QKV` is process-wide, so this gate assumes the serial CUDA
/// suite (`scripts/cuda_test.sh`); the Metal twin makes the same assumption.
#[test]
fn fused_qkv_norm_matches_unfused_decode_on_cuda() {
    use crate::graph::ops::Op;
    use crate::graph::params::{CParams, GraphParams, GraphType};
    use crate::models::{Device, ModelDef};

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen3-0.6B q8_0 not cached; skipping the #200 gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    assert!(!ids.is_empty());
    let positions: Vec<usize> = (0..ids.len()).collect();

    for cache_type in [None, Some("q8_0")] {
        let tag = cache_type.unwrap_or("default-f32");
        let model = crate::models::load_model_configured(
            &gguf,
            "",
            crate::graph::offload::OffloadRequest::Default,
            cache_type,
        )
        .expect("load a #200 gate engine");
        assert_eq!(
            model.device(),
            Device::Cuda,
            "[{tag}] the #200 gate needs the CUDA path; a device-gate fallback would \
             report a CPU number as a device one"
        );
        let q3: &Qwen3Model = model
            .as_any()
            .downcast_ref::<Qwen3Model>()
            .expect("qwen3 model");
        if cache_type == Some("q8_0") {
            assert_eq!(
                q3.kv_format,
                crate::graph::kvformat::KvFormat::Q8_0,
                "[{tag}] the engine must carry the format it was asked for"
            );
        }
        let n_ctx = q3.hparams.max_seq_len as usize;
        let nt = ids.len();

        // --- node presence (build only, no execute) ---
        let cparams = CParams {
            n_ctx,
            flash_attn: false,
            explicit_span: false,
            kv_map: false,
            gpu: true,
            gpu_layers: usize::MAX, // E5 fixture: no offload limit
            kv_format: q3.kv_format,
            fuse_qkv: true,
            fuse_ffn: false,
        };
        let params = GraphParams {
            n_tokens: 1,
            n_out: 1,
            gtype: GraphType::Decode,
            cparams,
            weights_version: 1,
        };
        let fg = Qwen3Graph::build(q3, &params);
        assert!(
            fg.nodes
                .iter()
                .any(|n| matches!(n.op, Op::FusedQkvNorm { .. })),
            "[{tag}] decode graph with fuse_qkv=true must contain FusedQkvNorm on CUDA"
        );
        let ug = Qwen3Graph::build(
            q3,
            &GraphParams {
                cparams: CParams {
                    fuse_qkv: false,
                    ..cparams
                },
                ..params.clone()
            },
        );
        assert!(
            !ug.nodes
                .iter()
                .any(|n| matches!(n.op, Op::FusedQkvNorm { .. })),
            "[{tag}] decode graph with fuse_qkv=false must NOT contain FusedQkvNorm"
        );

        // --- numeric A/B: the default (fused) vs MINFER_NO_FUSE_QKV=1 ---
        std::env::remove_var("MINFER_NO_FUSE_QKV");
        let mut a = GraphCache::new();
        let pa = Qwen3Graph::forward_cached(q3, &ids, &positions, 1, n_ctx, &mut a);
        let next = argmax(&pa);
        let da = Qwen3Graph::forward_cached(q3, &[next], &[nt], 1, n_ctx, &mut a);
        {
            let (g, _) = a.current().unwrap();
            assert!(
                g.nodes
                    .iter()
                    .any(|n| matches!(n.op, Op::FusedQkvNorm { .. })),
                "[{tag}] the default decode graph must be fused"
            );
        }

        std::env::set_var("MINFER_NO_FUSE_QKV", "1");
        let mut b = GraphCache::new();
        let pb = Qwen3Graph::forward_cached(q3, &ids, &positions, 1, n_ctx, &mut b);
        let db = Qwen3Graph::forward_cached(q3, &[next], &[nt], 1, n_ctx, &mut b);
        std::env::remove_var("MINFER_NO_FUSE_QKV");
        {
            let (g, _) = b.current().unwrap();
            assert!(
                !g.nodes
                    .iter()
                    .any(|n| matches!(n.op, Op::FusedQkvNorm { .. })),
                "[{tag}] the MINFER_NO_FUSE_QKV=1 decode graph must be unfused"
            );
        }

        // prefill never fuses (nt>1): both graphs run the same nodes, so this
        // isolates the decode-only divergence the two arms below pin.
        assert_eq!(
            pa, pb,
            "[{tag}] prefill a/b must be bit-identical; the fused node is decode-only"
        );

        // CUDA's concat matmul is bit-identical to the three separate ones: each
        // output row's dot is computed by the same fixed thread mapping,
        // independent of how many rows the launch has (unlike Metal, where the
        // grouping moves the Q8_0 result by ~1e-3 — see the Metal twin's comment).
        // Measured 0.0 on GB10; a non-zero value here means the matmul dispatch
        // changed its per-row reduction, which is a numerics change worth
        // stopping on.
        assert_eq!(
            argmax(&da),
            argmax(&db),
            "[{tag}] fused vs unfused decode must select the same next token"
        );
        assert_eq!(
            da, db,
            "[{tag}] fused vs unfused decode logits must be bit-identical on CUDA"
        );
        eprintln!("[#200 {tag}] prefill and decode logits are bit-identical");
    }
}
