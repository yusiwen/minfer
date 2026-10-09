//! The f16-on-Metal device + real-model gates (#164).
//!
//! The Metal twin of `f141_device.rs`: an f16 GGUF runs on the **device**
//! (all blocks on Metal, every f16 node assigned `Backend::METAL`) and its
//! device logits agree with the same file's CPU logits within the stated
//! backend tolerance, with an **identical greedy continuation**.
//!
//! Ignored because it needs a Metal device (a Mac) and a converted f16 GGUF.
//! Produce the Qwen2 fixture from the cached 0.5B with (provenance is the
//! `f16` output type, the cached q4_0 source, and this exact command):
//!
//! ```text
//! ./target/release/minfer quantize \
//!   ~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf \
//!   ~/.cache/minfer/f6-src/f164/minfer-f16.gguf --type f16
//! ```
//!
//! and the Qwen3 fixture identically from the cached Qwen3-0.6B Q8_0
//! (`Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf` -> `~/.cache/minfer/f6-src/f164/qwen3-f16.gguf`).
//! `minfer quantize --type f16` keeps 1-D tensors f32 and writes 2-D f16, which
//! is the file contract the kernels read. Override the path with
//! `MINFER_F164_F16_GGUF` / `MINFER_F164_QWEN3_F16_GGUF`.
//!
//! Run them alone (a Metal build, serial):
//!
//! ```text
//! cargo test --release --bin minfer f164 -- --ignored --test-threads=1
//! ```

use super::*;

/// Greedy continuation `steps` tokens past `ids`, returning the final-step
/// logits (whole vocabulary) and the sampled token ids. Generic over the
/// architecture through `ModelDef::forward_graph_cached`, so the same code
/// drives the Qwen2 and Qwen3 arms.
fn f164_greedy(
    model: &dyn crate::models::ModelDef,
    ids: &[u32],
    steps: usize,
    n_ctx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let nt = ids.len();
    let mut cache = crate::graph::cache::GraphCache::new();
    let mut positions: Vec<usize> = (0..nt).collect();
    let mut logits = model.forward_graph_cached(ids, &positions, 1, n_ctx, &mut cache);
    let mut toks = Vec::with_capacity(steps);
    for step in 0..steps {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        toks.push(next);
        positions = vec![nt + step];
        logits = model.forward_graph_cached(&[next], &positions, 1, n_ctx, &mut cache);
    }
    (logits, toks)
}

/// The shared body of both gates.
fn f164_run(env_key: &str, default: &str, ns: &str, n_ctx: usize, steps: usize) {
    use crate::graph::offload::OffloadRequest;
    use crate::graph::params::{CParams, GraphParams, GraphType};
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::Backend;
    use crate::models::Device;

    #[cfg(not(target_os = "macos"))]
    {
        let _ = (env_key, default, ns, n_ctx, steps);
        eprintln!("not a macOS build; the #164 Metal gate is a no-op here");
    }
    #[cfg(target_os = "macos")]
    {
        let _g = crate::metal::metal_test_lock();
        // The Metal device must be up **before** the load, or the loader decides
        // the weights are not usable there and answers `Device::Cpu`.
        crate::metal::MpsState::init();
        if crate::metal::MpsState::get().is_none() {
            eprintln!("no Metal device; skipping the #164 gate");
            return;
        }
        let Some(path) = env_path(env_key, default) else {
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("f16 GGUF parses");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
            .expect("strict tokenizer accepts the converted file");
        let ids = tok.encode(PROMPT);
        assert!(!ids.is_empty());

        // The file contract: every 2-D weight is f16, every 1-D norm/bias f32.
        let mut n_f16 = 0usize;
        let mut n_other = 0usize;
        for ti in &gguf.parts[0].ctx.info {
            if ti.ne[1] > 1 {
                if ti.type_ == crate::gguf::GgmlType::F16 {
                    n_f16 += 1;
                } else {
                    n_other += 1;
                }
            }
        }
        assert!(n_f16 > 0, "the f16 file has no 2-D f16 weight");
        assert_eq!(n_other, 0, "every 2-D weight must be f16 in this file");

        // --- device arm: an explicit full plan (no env involvement) --------
        let dev = crate::models::load_model_with(&gguf, ns, OffloadRequest::Layers(usize::MAX))
            .expect("device load");
        assert_eq!(
            dev.device(),
            Device::Metal,
            "an f16 GGUF did not make the model a Metal model — its weights are not \
             registered/consumable on the device"
        );
        let report = dev.offload_report().expect("offload report");
        assert!(
            report.contains("on metal") && !report.contains("cpu only"),
            "offload report does not put the blocks on the device: {report}"
        );

        // Every f16 node the scheduler assigns must be METAL. A *registered*
        // weight with no kernel would route the node to the CPU here and the
        // assert fires (or the count drops).
        let params = GraphParams {
            n_tokens: ids.len(),
            n_out: 1,
            gtype: GraphType::Prefill,
            cparams: CParams {
                n_ctx,
                flash_attn: false,
                explicit_span: false,
                kv_map: false,
                gpu: true,
                gpu_layers: usize::MAX,
                kv_format: crate::graph::kvformat::KvFormat::F32,
                fuse_qkv: true,
                fuse_ffn: true,
            },
            weights_version: 0,
        };
        let mut graph = dev.build_graph(&params);
        let mut alloc = crate::graph::alloc::GraphAllocator::new();
        assert!(alloc.enable_metal(), "the Metal allocator did not come up");
        BackendScheduler::new().assign_backends(&mut graph, &alloc);
        let mut f16_matmul = 0usize;
        let mut f16_embed = 0usize;
        for nd in &graph.nodes {
            let wt = match &nd.meta {
                crate::graph::ops::NodeMeta::MatMul(m) => Some(m.weight_ttype),
                crate::graph::ops::NodeMeta::Embed(m) => Some(m.weight_ttype),
                _ => None,
            };
            if wt != Some(crate::tensor::TensorType::F16) {
                continue;
            }
            match nd.op {
                crate::graph::ops::Op::MatMul { .. } => f16_matmul += 1,
                crate::graph::ops::Op::GetRows => f16_embed += 1,
                _ => {}
            }
            assert_eq!(
                nd.backend,
                Some(Backend::METAL),
                "f16 node '{}' ({:?}) was not assigned the device",
                nd.name,
                nd.op
            );
        }
        let n_layer = dev.n_layer();
        assert!(
            f16_matmul >= n_layer * 7,
            "only {f16_matmul} f16 matmul nodes in the graph (n_layer={n_layer})"
        );
        assert!(f16_embed >= 1, "the f16 embedding node is missing");

        // --- CPU arm: the same file, the same forward, Layers(0) ----------
        let cpu =
            crate::models::load_model_with(&gguf, ns, OffloadRequest::Layers(0)).expect("cpu load");
        assert_eq!(
            cpu.device(),
            Device::Cpu,
            "the CPU arm must not be a device model"
        );

        let (ld, gd) = f164_greedy(dev.as_ref(), &ids, steps, n_ctx);
        let (lc, gc) = f164_greedy(cpu.as_ref(), &ids, steps, n_ctx);
        assert_eq!(gd, gc, "greedy continuation differs between device and CPU");
        assert_eq!(ld.len(), lc.len());
        let max_abs = ld
            .iter()
            .zip(lc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_logit = ld.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "[f164] {n_f16} f16 tensors, {f16_matmul} f16 matmul + {f16_embed} embed nodes on \
             METAL, n_layer={n_layer}, max |Δlogit| = {max_abs}, max |logit| = {max_logit}, \
             greedy {gd:?}"
        );
        // Bar named before measuring, in the assertion comment: the two paths
        // differ only in accumulation order and the attention exp/softmax
        // kernels (both compute f32 activations against f16 weights — an f16
        // weight has no integer form, so the CPU does NOT quantize its
        // activations as it does for the quantized types). The CUDA f16 record
        // (#141) measured 7.34e-5 absolute on its converted file; Metal's scalar
        // row kernels accumulate in a different lane order and its attention
        // kernels are independent, so the observed drift here is larger
        // (max 2.4e-3 on the 0.5B, 7.9e-3 on Qwen3-0.6B, at a logit scale of
        // ~20). The bar is **0.05 absolute / 5e-3 relative** — about 6x the
        // observed, and still ~200x below any weight-level fault (a wrong f16
        // row moves logits by O(1)). The identical greedy check below is the
        // load-bearing correctness claim; this is a gross-error detector.
        assert!(max_abs <= 0.05, "max |Δlogit| = {max_abs}");
        assert!(
            max_abs / max_logit.max(1.0) <= 5e-3,
            "max relative Δlogit = {}",
            max_abs / max_logit
        );
    }
}

/// #164: f16 weights execute on the Metal **device** (Qwen2 twin of #141's
/// `f141_f16_weights_run_on_the_cuda_device`).
#[test]
#[ignore = "requires a converted f16 Qwen2 GGUF and a Metal device (#164)"]
fn f164_f16_weights_run_on_the_metal_device() {
    f164_run(
        "MINFER_F164_F16_GGUF",
        "~/.cache/minfer/f6-src/f164/minfer-f16.gguf",
        "f164:",
        512,
        4,
    );
}

/// #164: the same acceptance for the Qwen3 architecture — the loader branch and
/// the graph are separate code paths, and both loaders had to flip together.
#[test]
#[ignore = "requires a converted f16 Qwen3 GGUF and a Metal device (#164)"]
fn f164_f16_weights_run_on_the_metal_device_qwen3() {
    f164_run(
        "MINFER_F164_QWEN3_F16_GGUF",
        "~/.cache/minfer/f6-src/f164/qwen3-f16.gguf",
        "f164q3:",
        512,
        4,
    );
}
