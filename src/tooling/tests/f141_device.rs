//! The f16 device gate (needs a CUDA device and a converted GGUF).
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// G2d (#141): the f16 weights execute on the CUDA **device**, and the
/// device logits agree with the same file's CPU logits within the stated
/// backend tolerance.
///
/// Placement is the load-bearing claim, so it is asserted directly rather
/// than inferred from a timing: the loader's offload report must say every
/// block is on the device, and every `F16` matmul / embedding node the
/// scheduler assigns must be `Backend::CUDA`. Break the loader's f16
/// registration (drop `TensorType::F16` from the CUDA branch) and
/// `Qwen2Model::device()` answers `Cpu`: this gate then fails on its first
/// assertion instead of quietly measuring the CPU.
///
/// The model comes from `minfer convert --outtype f16` on the HF
/// checkpoint, i.e. the F6 acceptance's own file, not a quantize-produced
/// one — see `MINFER_F141_F16_GGUF`.
///
/// Tolerance: CPU and CUDA reduce in different orders and run different
/// exp/softmax kernels (graph rule §9), so the two paths are not bit-equal
/// by design. What must hold is the *decision*: the greedy continuation is
/// identical. The numeric bound is printed and asserted with headroom.
#[test]
#[ignore = "requires a converted f16 GGUF and a CUDA device (#141)"]
fn f141_f16_weights_run_on_the_cuda_device() {
    #[cfg(not(feature = "cuda"))]
    eprintln!("not a CUDA build; #141's device gate is a no-op here");
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
            eprintln!("no CUDA device; skipping the #141 device gate");
            return;
        }
        let Some(path) = env_path("MINFER_F141_F16_GGUF", "/tmp/f141-work/minfer-f16.gguf") else {
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("f16 GGUF parses");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
            .expect("strict tokenizer accepts the converted file");
        let ids = tok.encode(PROMPT);
        assert!(!ids.is_empty());

        // Fully typed weights only: the converted file's 2-D tensors are f16
        // and its 1-D norms/biases f32. A silent fallback would mean the
        // device plan is not honest, so state it.
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
        //
        // Both arms load under a **namespace**. The CUDA weight registry is
        // process-global and name-keyed, and `register_weight` reuses a
        // same-name+same-size device copy; the other real-model gates in this
        // set load the cached q4_k_m 0.5B under the default `ns=""`, which
        // shares 121 f32 norm/bias names *and* their byte sizes with this
        // file — but not their values (the cached file's f32 tensors are a
        // different checkpoint revision: `blk.0.attn_norm.weight[0]` is
        // -0.046875 here, matching the HF bf16 safetensors, and -0.082947
        // there). Without the namespace the device arm silently computes with
        // another file's norms and the greedy continuation diverges — the
        // process-global hazard #64 documents, and why the loader has `ns`.
        let dev =
            crate::models::load_model_with(&gguf, "f141:", OffloadRequest::Layers(usize::MAX))
                .expect("device load");
        let q = dev
            .as_any()
            .downcast_ref::<crate::models::qwen2::Qwen2Model>()
            .expect("Qwen2 model");
        assert_eq!(
            Qwen2Graph::device(q),
            Device::Cuda,
            "an f16 GGUF did not make the model a CUDA model — its weights are not \
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

        // Every f16 node the scheduler assigns must be CUDA. This is the
        // assertion that a *registered* weight with no kernel would break:
        // `supports_op` would route the node to the CPU and the count below
        // would drop (or the backend assert would fire).
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
        let mut f16_matmul = 0usize;
        let mut f16_embed = 0usize;
        for nd in &graph.nodes {
            let wt = match &nd.meta {
                crate::graph::ops::NodeMeta::MatMul(m) => Some(m.weight_ttype),
                crate::graph::ops::NodeMeta::Embed(m) => Some(m.weight_ttype),
                _ => None,
            };
            if wt != Some(TensorType::F16) {
                continue;
            }
            match nd.op {
                crate::graph::ops::Op::MatMul { .. } => f16_matmul += 1,
                crate::graph::ops::Op::GetRows => f16_embed += 1,
                _ => {}
            }
            assert_eq!(
                nd.backend,
                Some(Backend::CUDA),
                "f16 node '{}' ({:?}) was not assigned the device",
                nd.name,
                nd.op
            );
        }
        // 24 blocks × (q,k,v,o,gate,up,down) + lm_head; measured 169 matmul
        // + 1 embed f16 nodes on the 0.5B. The floor guards against a
        // vacuous pass on an empty node set.
        assert!(
            f16_matmul >= 24 * 7,
            "only {f16_matmul} f16 matmul nodes in the graph"
        );
        assert!(f16_embed >= 1, "the f16 embedding node is missing");

        // --- CPU arm: the same file, the same forward, Layers(0) ----------
        let cpu = crate::models::load_model_with(&gguf, "f141:", OffloadRequest::Layers(0))
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
            "f141 f16 device: {n_f16} f16 tensors, {f16_matmul} f16 matmul + {f16_embed} \
             embed nodes on CUDA, max |Δlogit| = {max_abs} (mean {mean_abs}), max |logit| = \
             {max_logit}, greedy {:?}",
            gd
        );
        // The stated backend tolerance. Both paths compute f32 activations
        // against f16 weights (an f16 weight has no integer form, so the CPU
        // does *not* quantize its activations as it does for the quantized
        // types), so the difference is accumulation order plus the attention
        // exp/softmax kernel — not weight precision. Measured on the 0.5B at
        // ctx 512: 7.34e-5 absolute, 4.0e-6 relative to the largest logit.
        // The bound keeps ~130x headroom over that, which is still far below
        // any weight-level fault (a wrong f16 row moves logits by O(1)).
        assert!(max_abs <= 0.01, "max |Δlogit| = {max_abs}");
        assert!(
            max_abs / max_logit.max(1.0) <= 1e-3,
            "max relative Δlogit = {}",
            max_abs / max_logit
        );
    }
}
