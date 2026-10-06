//! The bf16 device gate (#208, the CUDA half) — needs a CUDA device and a
//! converted bf16 GGUF.
//!
//! The twin of `f141_device.rs` for the second 2 B/element dtype. The two gates
//! are deliberately separate rather than parameterised: each names its own type
//! in its own assertions, so a change that admits one dtype and breaks the other
//! cannot pass by sharing a body.

use super::*;

/// #208 acceptance: a bf16 GGUF executes on the CUDA **device**, and the device
/// logits agree with the same file's CPU logits within the stated backend
/// tolerance and an identical greedy continuation.
///
/// Placement is the load-bearing claim, so it is asserted directly rather than
/// inferred from a timing: `Qwen2Graph::device()` must answer `Cuda` (no
/// inferred placement), the loader's offload report must say every block is on
/// the device, and every `BF16` matmul / embedding node the scheduler assigns
/// must be `Backend::CUDA`. Drop `TensorType::BF16` from the CUDA branch of
/// `models::weight_reg::cuda_weight_reg` (the loader's registration arm) and
/// `Qwen2Graph::device()` answers `Cpu`: this gate then fails on its first
/// assertion instead of quietly measuring the CPU — that is mutation 1 of #208's
/// evidence.
///
/// The model comes from `minfer convert --outtype bf16` on the HF checkpoint
/// (`minfer quantize` has no bf16 encoder), i.e. the F6 acceptance's own file —
/// see `MINFER_F208_BF16_GGUF` and the `f208_bf16_fixture` helper below for the
/// exact command and its provenance.
///
/// **Bar, named before measuring.** CPU and CUDA reduce in different orders and
/// run different attention exp/softmax kernels (graph rule §9), so the two paths
/// are not bit-equal by design — and both arms read the *same* bf16 weights, so
/// weight precision is not part of the difference at all. The recorded reference
/// is #141's f16-device measurement on the same 0.5B (7.34e-5 absolute / 4.0e-6
/// relative) and #142's CPU bf16-vs-f16 record (2.29e-5 / 1.24e-6); bf16's
/// device path has the same f32 accumulation shape as f16's. Stated bound:
/// **max |Δlogit| ≤ 0.01 and max relative ≤ 1e-3, with the greedy continuation
/// identical.** The measured values are printed first.
#[test]
#[ignore = "requires a converted bf16 GGUF and a CUDA device (#208)"]
fn f208_bf16_weights_run_on_the_cuda_device() {
    #[cfg(not(feature = "cuda"))]
    eprintln!("not a CUDA build; #208's device gate is a no-op here");
    #[cfg(feature = "cuda")]
    {
        use crate::graph::offload::OffloadRequest;
        use crate::graph::Backend;
        use crate::models::qwen2::graph::Qwen2Graph;
        use crate::models::Device;
        use crate::tensor::TensorType;

        // Bring CUDA up the way a load would (the loader calls this too);
        // `CudaState::get()` is None until something initializes it.
        crate::cuda::CudaState::init();
        if crate::cuda::CudaState::get().is_none() {
            eprintln!("no CUDA device; skipping the #208 device gate");
            return;
        }
        let Some(path) = f208_bf16_fixture() else {
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("bf16 GGUF parses");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
            .expect("strict tokenizer accepts the converted file");
        let ids = tok.encode(PROMPT);
        assert!(!ids.is_empty());

        // Fully typed weights only: the converted file's 2-D tensors are bf16
        // and its 1-D norms/biases f32 (the F6 file contract, #142). A silent
        // fallback would mean the device plan is not honest, so state it.
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
        // Both arms load under a **namespace**. The CUDA weight registry is
        // process-global and name-keyed, and `register_weight` reuses a
        // same-name+same-size device copy; the other real-model gates in this
        // set load the cached q4_k_m 0.5B under the default `ns=""`, which
        // shares 121 f32 norm/bias names *and* their byte sizes with this
        // file — but not their values. Without the namespace the device arm
        // silently computes with another file's norms and the greedy
        // continuation diverges — the process-global hazard #64 documents, and
        // why the loader has `ns` (the same reason f141 uses `"f141:"`).
        let dev =
            crate::models::load_model_with(&gguf, "f208:", OffloadRequest::Layers(usize::MAX))
                .expect("device load");
        let q = dev
            .as_any()
            .downcast_ref::<crate::models::qwen2::Qwen2Model>()
            .expect("Qwen2 model");
        assert_eq!(
            Qwen2Graph::device(q),
            Device::Cuda,
            "a bf16 GGUF did not make the model a CUDA model — its weights are not \
             registered/consumable on the device"
        );
        let report = dev.offload_report().expect("offload report");
        assert!(
            report.contains("on cuda") && !report.contains("cpu only"),
            "offload report does not put the blocks on the device: {report}"
        );
        assert_eq!(
            q.offload.plan.gpu_layers, q.offload.plan.n_layers,
            "not every block is on the device: {report}"
        );

        // Every bf16 node the scheduler assigns must be CUDA. This is the
        // assertion that a *registered* weight with no kernel would break:
        // `supports_op` would route the node to the CPU and the count below
        // would drop (or the backend assert would fire). bf16 does not fuse
        // (`cuda::concat_rows_feasible` has no 2 B/element arm), so the
        // expected census is the unfused shape: 24 blocks × 7 matmuls + lm_head.
        let params = crate::graph::params::GraphParams {
            n_tokens: ids.len(),
            n_out: 1,
            gtype: crate::graph::params::GraphType::Prefill,
            cparams: crate::graph::params::CParams {
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
        let mut graph =
            <crate::models::qwen2::Qwen2Model as crate::models::ModelDef>::build_graph(q, &params);
        let mut alloc = crate::graph::alloc::GraphAllocator::new();
        assert!(alloc.enable_cuda(), "the CUDA allocator did not come up");
        crate::graph::scheduler::BackendScheduler::new().assign_backends(&mut graph, &alloc);
        let mut bf16_matmul = 0usize;
        let mut bf16_embed = 0usize;
        let mut bf16_elsewhere = 0usize;
        for nd in &graph.nodes {
            let wt = match &nd.meta {
                crate::graph::ops::NodeMeta::MatMul(m) => Some(m.weight_ttype),
                crate::graph::ops::NodeMeta::Embed(m) => Some(m.weight_ttype),
                _ => None,
            };
            if wt != Some(TensorType::BF16) {
                continue;
            }
            match nd.op {
                crate::graph::ops::Op::MatMul { .. } => bf16_matmul += 1,
                crate::graph::ops::Op::GetRows => bf16_embed += 1,
                _ => bf16_elsewhere += 1,
            }
            assert_eq!(
                nd.backend,
                Some(Backend::CUDA),
                "bf16 node '{}' ({:?}) was not assigned the device",
                nd.name,
                nd.op
            );
        }
        // 24 blocks × (q,k,v,o,gate,up,down) + lm_head; the f16 twin measures
        // 169 matmul + 1 embed on the same 0.5B. The floor guards against a
        // vacuous pass on an empty node set.
        assert!(
            bf16_matmul >= 24 * 7,
            "only {bf16_matmul} bf16 matmul nodes in the graph"
        );
        assert!(bf16_embed >= 1, "the bf16 embedding node is missing");
        assert_eq!(
            bf16_elsewhere, 0,
            "a bf16 node outside MatMul/GetRows — the census is not the unfused shape"
        );

        // --- CPU arm: the same file, the same forward, Layers(0) ----------
        let cpu = crate::models::load_model_with(&gguf, "f208:", OffloadRequest::Layers(0))
            .expect("cpu load");
        let qc = cpu
            .as_any()
            .downcast_ref::<crate::models::qwen2::Qwen2Model>()
            .expect("Qwen2 model");
        assert_eq!(
            Qwen2Graph::device(qc),
            Device::Cpu,
            "the CPU arm must not be a device model"
        );

        let (ld, gd) = logits_greedy_on(q, &ids, 4, 512);
        let (lc, gc) = logits_greedy_on(qc, &ids, 4, 512);
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
            "f208 bf16 device: {n_bf16} 2-D bf16 + {n_1d} 1-D f32 tensors, {bf16_matmul} bf16 \
             matmul + {bf16_embed} embed nodes on CUDA, max |Δlogit| = {max_abs} (mean \
             {mean_abs}), max |logit| = {max_logit} ({:.3e} relative), greedy {:?}",
            max_abs / max_logit.max(1.0),
            gd
        );
        assert!(max_abs <= 0.01, "max |Δlogit| = {max_abs} (bar 0.01)");
        assert!(
            max_abs / max_logit.max(1.0) <= 1e-3,
            "max relative Δlogit = {} (bar 1e-3)",
            max_abs / max_logit
        );
    }
}

/// The bf16 fixture: `minfer convert --outtype bf16 <hf-dir>` (there is no bf16
/// encoder in `minfer quantize`, so the F6 writer is the only producer).
///
/// `$MINFER_F208_BF16_GGUF` points at a **ready** file and skips the conversion;
/// otherwise the cached HF checkpoint is converted into `/tmp/f6-work/f208/`.
/// The gate skips **loudly** (an `eprintln` naming the path) when neither
/// exists, the same shape as `env_path` in the parent module.
#[cfg(feature = "cuda")]
fn f208_bf16_fixture() -> Option<std::path::PathBuf> {
    if let Ok(p) = std::env::var("MINFER_F208_BF16_GGUF") {
        let p = std::path::PathBuf::from(p);
        if p.exists() {
            return Some(p);
        }
        eprintln!(
            "MINFER_F208_BF16_GGUF={} does not exist; skipping the #208 device gate",
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
