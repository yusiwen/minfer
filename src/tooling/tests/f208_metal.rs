//! The bf16-on-Metal device + real-model gate (#208, the Metal half).
//!
//! The Metal twin of `f208_device.rs` (the CUDA half): a bf16 GGUF runs on the
//! **device** (all blocks on Metal, every bf16 node assigned `Backend::METAL`)
//! and its device logits agree with the same file's CPU logits within the stated
//! backend tolerance, with an **identical greedy continuation**.
//!
//! Ignored because it needs a Metal device (a Mac) and a converted bf16 GGUF.
//! `minfer quantize` has no bf16 encoder, so the fixture comes from
//! `minfer convert --outtype bf16 <hf-model-dir>` — the F6 writer
//! (`src/tooling/tests/bf16.rs` uses the same provenance). The gate takes a ready
//! file from `$MINFER_F208_BF16_GGUF`, else converts
//! `~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct` (override with
//! `$MINFER_F208_HF_DIR`), and skips **loudly** when neither exists.
//!
//! No Qwen3 arm exists because `minfer convert` supports `Qwen2ForCausalLM` only
//! (`src/convert.rs`), so there is no producer for a bf16 **Qwen3** file; the
//! Qwen3 loader arm is the same one-line registration change (see the loader
//! twins) and is covered by the shared shape, not a second file.
//!
//! Run it alone (a Metal build, serial):
//!
//! ```text
//! cargo test --release --bin minfer f208 -- --ignored --test-threads=1
//! ```

use super::*;

/// Greedy continuation `steps` tokens past `ids`, returning the final-step
/// logits (whole vocabulary) and the sampled token ids. Generic over the
/// architecture through `ModelDef::forward_graph_cached` (the `f164_metal`
/// helper's twin).
fn f208_metal_greedy(
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

/// The bf16 fixture: `minfer convert --outtype bf16 <hf-dir>` (`minfer quantize`
/// has no bf16 encoder, so the F6 writer is the only producer).
///
/// `$MINFER_F208_BF16_GGUF` points at a **ready** file and skips the conversion;
/// otherwise the cached HF checkpoint is converted into the scratch dir
/// `/tmp/f6-work/f208/`. The
/// helper skips loudly (naming the path) when neither exists, the same shape as
/// `env_path` in the parent module.
#[cfg(target_os = "macos")]
fn f208_metal_fixture() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("MINFER_F208_BF16_GGUF") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            // #205: this early return bypasses the shared resolver, so it verifies
            // the fixture itself rather than handing an unrecorded file to the gate.
            super::f6_fixtures::check(&p);
            return Some(p);
        }
        eprintln!(
            "MINFER_F208_BF16_GGUF={} does not exist; skipping the #208 Metal gate",
            p.display()
        );
        return None;
    }
    let Some(hf_dir) = env_path(
        "MINFER_F208_HF_DIR",
        "~/.cache/minfer/f6-src/hf/Qwen2.5-0.5B-Instruct",
    ) else {
        return None;
    };
    let out = work_dir("f208").join("f208-bf16.gguf");
    if !out.exists() {
        crate::convert::Conversion::plan(&hf_dir, OutType::Bf16)
            .expect("plan the bf16 conversion")
            .write_single(&out)
            .expect("write the bf16 GGUF");
    }
    Some(out)
}

/// #208 acceptance (Metal half): a bf16 GGUF executes on the Metal **device**,
/// and the device logits agree with the same file's CPU logits within the stated
/// backend tolerance and an identical greedy continuation.
///
/// Placement is the load-bearing claim, so it is asserted directly:
/// `ModelDef::device()` must answer `Metal` (no inferred placement), the loader's
/// offload report must say the blocks are on the device, and every `BF16` matmul /
/// embedding node must be `Backend::METAL`. Drop `TensorType::BF16` from the
/// Metal registration arms (`src/models/qwen2/loader.rs` /
/// `src/models/qwen3/loader.rs`) and `device()` answers `Cpu`: this gate then
/// fails on its first assertion instead of quietly measuring the CPU — that is
/// mutation 1 of #208's evidence. The exactness gate
/// (`metal::tests::bf16_matmul_matches_the_exact_shift_reference`) is what catches
/// a registered-but-wrong kernel (mutation 2).
///
/// **Bar, named before measuring.** The two paths differ only in accumulation
/// order and the attention exp/softmax kernels (graph rule §9), and both arms read
/// the *same* bf16 weights, so weight precision is not part of the difference. The
/// CPU bf16-vs-f16 record is 2.29e-5 / 1.24e-6, but Metal's scalar row kernels
/// accumulate in a different lane order, which is the dominant term — #164's f16
/// Metal gate recorded max 2.4e-3 absolute at a logit scale of ~20 against a 0.05
/// bar. Stated bound: **max |Δlogit| ≤ 0.05 and max relative ≤ 5e-3, with the
/// greedy continuation identical** (the f16 Metal gate's bar, not loosened). The
/// measured values are printed first, and the identical greedy check is the
/// load-bearing correctness claim.
#[test]
#[ignore = "requires a converted bf16 GGUF and a Metal device (#208)"]
fn f208_bf16_weights_run_on_the_metal_device() {
    #[cfg(not(target_os = "macos"))]
    eprintln!("not a macOS build; #208's Metal gate is a no-op here");
    #[cfg(target_os = "macos")]
    {
        use crate::graph::offload::OffloadRequest;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        use crate::graph::Backend;
        use crate::models::Device;

        let _g = crate::metal::metal_test_lock();
        // The Metal device must be up **before** the load, or the loader decides
        // the weights are not usable there and answers `Device::Cpu`.
        crate::metal::MpsState::init();
        if crate::metal::MpsState::get().is_none() {
            eprintln!("no Metal device; skipping the #208 Metal gate");
            return;
        }
        let Some(path) = f208_metal_fixture() else {
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("bf16 GGUF parses");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
            .expect("strict tokenizer accepts the converted file");
        let ids = tok.encode(PROMPT);
        assert!(!ids.is_empty());

        // The file contract: every 2-D weight is bf16, every 1-D norm/bias f32.
        let mut n_bf16 = 0usize;
        let mut n_other = 0usize;
        let mut n_1d = 0usize;
        for ti in &gguf.parts[0].ctx.info {
            if ti.ne[1] > 1 {
                if ti.type_ == crate::gguf::GgmlType::BF16 {
                    n_bf16 += 1;
                } else {
                    n_other += 1;
                }
            } else {
                assert_eq!(
                    ti.type_,
                    crate::gguf::GgmlType::F32,
                    "1-D tensor '{}' must stay f32 in a bf16 file",
                    ti.name
                );
                n_1d += 1;
            }
        }
        assert!(n_bf16 > 0, "the bf16 file has no 2-D bf16 weight");
        assert_eq!(n_other, 0, "every 2-D weight must be bf16 in this file");
        assert!(n_1d > 0, "the bf16 file has no 1-D f32 norm/bias");

        // --- device arm: an explicit full plan (no env involvement) --------
        //
        // Both arms load under a **namespace**. The Metal weight registry is
        // process-global and name-keyed; the other real-model gates in this set
        // load the cached 0.5B under the default `ns=""`, sharing the f32
        // norm/bias names (but not their values). The namespace keeps the device
        // arm on this file's norms — the same reason `f164_metal` / `f141` use one.
        let dev =
            crate::models::load_model_with(&gguf, "f208m:", OffloadRequest::Layers(usize::MAX))
                .expect("device load");
        assert_eq!(
            dev.device(),
            Device::Metal,
            "a bf16 GGUF did not make the model a Metal model — its weights are not \
             registered/consumable on the device"
        );
        let report = dev.offload_report().expect("offload report");
        assert!(
            report.contains("on metal") && !report.contains("cpu only"),
            "offload report does not put the blocks on the device: {report}"
        );

        // Every bf16 node the scheduler assigns must be METAL. bf16 does not fuse
        // (the fused QKV/FFN device forms are gated on the CUDA path), so the
        // expected census is the unfused shape: n_layer × 7 matmuls + lm_head,
        // plus the embedding gather.
        let params = GraphParams {
            n_tokens: ids.len(),
            n_out: 1,
            gtype: GraphType::Prefill,
            cparams: CParams {
                n_ctx: 512,
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
        let mut bf16_matmul = 0usize;
        let mut bf16_embed = 0usize;
        let mut bf16_elsewhere = 0usize;
        for nd in &graph.nodes {
            let wt = match &nd.meta {
                crate::graph::ops::NodeMeta::MatMul(m) => Some(m.weight_ttype),
                crate::graph::ops::NodeMeta::Embed(m) => Some(m.weight_ttype),
                _ => None,
            };
            if wt != Some(crate::tensor::TensorType::BF16) {
                continue;
            }
            match nd.op {
                crate::graph::ops::Op::MatMul { .. } => bf16_matmul += 1,
                crate::graph::ops::Op::GetRows => bf16_embed += 1,
                _ => bf16_elsewhere += 1,
            }
            assert_eq!(
                nd.backend,
                Some(Backend::METAL),
                "bf16 node '{}' ({:?}) was not assigned the device",
                nd.name,
                nd.op
            );
        }
        let n_layer = dev.n_layer();
        assert!(
            bf16_matmul >= n_layer * 7,
            "only {bf16_matmul} bf16 matmul nodes in the graph (n_layer={n_layer})"
        );
        assert!(bf16_embed >= 1, "the bf16 embedding node is missing");
        assert_eq!(
            bf16_elsewhere, 0,
            "a bf16 node outside MatMul/GetRows — the census is not the unfused shape"
        );

        // --- CPU arm: the same file, the same forward, Layers(0) ----------
        let cpu = crate::models::load_model_with(&gguf, "f208m:", OffloadRequest::Layers(0))
            .expect("cpu load");
        assert_eq!(
            cpu.device(),
            Device::Cpu,
            "the CPU arm must not be a device model"
        );

        let (ld, gd) = f208_metal_greedy(dev.as_ref(), &ids, 4, 512);
        let (lc, gc) = f208_metal_greedy(cpu.as_ref(), &ids, 4, 512);
        assert_eq!(gd, gc, "greedy continuation differs between device and CPU");
        assert_eq!(ld.len(), lc.len());
        let max_abs = ld
            .iter()
            .zip(lc.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mean_abs = ld
            .iter()
            .zip(lc.iter())
            .map(|(a, b)| (a - b).abs())
            .sum::<f32>()
            / ld.len() as f32;
        let max_logit = ld.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        eprintln!(
            "[f208-metal] {n_bf16} 2-D bf16 + {n_1d} 1-D f32 tensors, {bf16_matmul} bf16 matmul + \
             {bf16_embed} embed nodes on METAL, n_layer={n_layer}, max |Δlogit| = {max_abs} \
             (mean {mean_abs}), max |logit| = {max_logit} ({:.3e} relative), greedy {gd:?}",
            max_abs / max_logit.max(1.0),
        );
        assert!(max_abs <= 0.05, "max |Δlogit| = {max_abs} (bar 0.05)");
        assert!(
            max_abs / max_logit.max(1.0) <= 5e-3,
            "max relative Δlogit = {} (bar 5e-3)",
            max_abs / max_logit
        );
    }
}
