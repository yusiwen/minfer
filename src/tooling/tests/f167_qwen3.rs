//! Qwen3 f16 and q4_K device registration.
//!
//! Split out of `src/tooling/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

/// #167: the Qwen3 twin of the #141 gate. An f16 **Qwen3** GGUF runs on the
/// CUDA device — the loader registers the f16 weights raw and
/// `Qwen3Graph::weights_on_cuda` admits the type — and its device logits agree
/// with the same file's CPU logits within a stated bound (max |Δlogit| ≤ 0.05
/// and ≤ 1e-3 relative; the gate prints both measured values).
///
/// Why a separate gate: `f141_f16_weights_run_on_the_cuda_device` is
/// qwen2-specific (it downcasts to `Qwen2Model`), and the qwen3 loader was
/// exactly the copy that lacked the f16 arm. Placement is asserted directly,
/// not inferred from a timing: every `F16` matmul / embedding node the
/// scheduler assigns must be `Backend::CUDA`, so a silent CPU fallback fails on
/// the first such node instead of reporting a CPU number as a device one.
///
/// The model is `llama-quantize --allow-requantize … F16` on the cached
/// `Qwen3-0.6B-Q8_0.gguf` (a Q8_0 file is not a K-quant, so re-quantizing is
/// allowed). **Not** `minfer quantize --type f16`: that path converts 1-D norms to
/// f16 too, while the engine's f16 contract (and llama.cpp's rule) keeps them f32 —
/// the assertion below pins the contract, and the tooling deviation is filed
/// separately. See `MINFER_F167_F16_GGUF`.
#[test]
#[ignore = "requires a converted f16 Qwen3 GGUF and a CUDA device (#167)"]
fn f167_f16_qwen3_weights_run_on_the_cuda_device() {
    #[cfg(not(feature = "cuda"))]
    eprintln!("not a CUDA build; #167's f16 Qwen3 gate is a no-op here");
    #[cfg(feature = "cuda")]
    {
        use crate::graph::offload::OffloadRequest;
        use crate::graph::Backend;
        use crate::models::qwen3::graph::Qwen3Graph;
        use crate::models::Device;
        use crate::tensor::TensorType;

        crate::cuda::CudaState::init();
        if crate::cuda::CudaState::get().is_none() {
            eprintln!("no CUDA device; skipping the #167 f16 Qwen3 gate");
            return;
        }
        let Some(path) = env_path("MINFER_F167_F16_GGUF", "/tmp/f167-work/qwen3-f16.gguf") else {
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("f16 GGUF parses");
        // The gate must not pass by loading some other architecture's file.
        assert_eq!(
            gguf.parts[0]
                .ctx
                .get_key_val_str("general.architecture")
                .as_deref(),
            Some("qwen3"),
            "#167's f16 gate requires a Qwen3 GGUF"
        );
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx)
            .expect("strict tokenizer accepts the converted file");
        let ids = tok.encode(PROMPT);
        assert!(!ids.is_empty());

        let mut n_f16 = 0usize;
        let mut n_other = 0usize;
        let mut n_1d_f32 = 0usize;
        let mut n_1d_other = 0usize;
        for ti in &gguf.parts[0].ctx.info {
            if ti.ne[1] > 1 {
                if ti.type_ == crate::gguf::GgmlType::F16 {
                    n_f16 += 1;
                } else {
                    n_other += 1;
                }
            } else if ti.type_ == crate::gguf::GgmlType::F32 {
                n_1d_f32 += 1;
            } else {
                n_1d_other += 1;
            }
        }
        assert!(n_f16 > 0, "the f16 file has no 2-D f16 weight");
        assert_eq!(n_other, 0, "every 2-D weight must be f16 in this file");
        // The engine's f16 contract: 1-D norms/biases stay f32 (the converter's and
        // llama.cpp's rule; `mat_mul_f16`/`embed_rows_f16` have no f16-norm sibling,
        // and the CPU/device RMSNorm reads f32 weights). Asserting it here is what
        // stops the gate from accepting an f16 file the engine cannot actually run.
        assert!(n_1d_f32 > 0, "the f16 file has no 1-D f32 norm/bias");
        assert_eq!(
            n_1d_other, 0,
            "every 1-D norm/bias must stay f32 in an f16 file"
        );

        // A namespace, for the same process-global-registry reason as f141: the
        // other real-model gates register same-named 0.5B/0.6B tensors under ns="".
        let dev =
            crate::models::load_model_with(&gguf, "f167f16:", OffloadRequest::Layers(usize::MAX))
                .expect("device load");
        let q = dev
            .as_any()
            .downcast_ref::<crate::models::qwen3::Qwen3Model>()
            .expect("Qwen3 model");
        assert_eq!(
            Qwen3Graph::device(q),
            Device::Cuda,
            "an f16 Qwen3 GGUF did not make the model a CUDA model — the loader's f16 \
             registration or the graph's f16 type gate is missing"
        );
        let report = dev.offload_report().expect("offload report");
        assert!(
            report.contains("on cuda") && !report.contains("cpu only"),
            "offload report does not put the blocks on the device: {report}"
        );
        let n_layers = q.offload.plan.n_layers;
        assert_eq!(
            q.offload.plan.gpu_layers, n_layers,
            "not every block is on the device: {report}"
        );

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
            <crate::models::qwen3::Qwen3Model as crate::models::ModelDef>::build_graph(q, &params);
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
        // 7 f16 matmuls per block plus lm_head; the floor guards against a
        // vacuous pass on an empty node set (the count is model-derived).
        assert!(
            f16_matmul >= n_layers * 7,
            "only {f16_matmul} f16 matmul nodes in the graph for {n_layers} layers"
        );
        assert!(f16_embed >= 1, "the f16 embedding node is missing");

        let cpu = crate::models::load_model_with(&gguf, "f167f16:", OffloadRequest::Layers(0))
            .expect("cpu load");
        let qc = cpu
            .as_any()
            .downcast_ref::<crate::models::qwen3::Qwen3Model>()
            .expect("Qwen3 model");
        assert_eq!(
            Qwen3Graph::device(qc),
            Device::Cpu,
            "the CPU arm must not be a device model"
        );

        let (ld, gd) = logits_greedy_on_qwen3(q, &ids, 4, 512);
        let (lc, gc) = logits_greedy_on_qwen3(qc, &ids, 4, 512);
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
            "f167 f16 qwen3 device: {n_f16} f16 tensors, {f16_matmul} f16 matmul + \
             {f16_embed} embed nodes on CUDA, max |Δlogit| = {max_abs} (mean {mean_abs}), \
             max |logit| = {max_logit}, greedy {:?}",
            gd
        );
        // Same comparison class as #141: CPU and CUDA reduce in different orders
        // and run different exp/softmax kernels (graph rule §9), so the paths are
        // not bit-equal by design. Qwen3's spread is ~100× the 0.5B f16 gate's
        // (measured: max |Δlogit| 1.7e-2 / relative 5e-4 here against f141's
        // 7.34e-5 / 4.0e-6) because Qwen3 runs **four** norms per layer
        // (attn_norm + per-head q_norm/k_norm + ffn_norm), and the CPU's rms_norm
        // (8-lane AVX2 FMA plus an f64 tail, then `1/sqrt`) and the device's
        // (warp-shuffle f32 reduction, then `rsqrtf`) differ in both reduction
        // order and reciprocal-sqrt form — an order-of-accumulation difference that
        // compounds through 28 layers, not a weight fault. The bounds are stated
        // here and printed above; a wrong f16 row moves logits by O(1), which the
        // greedy continuation would also catch.
        assert!(max_abs <= 0.05, "max |Δlogit| = {max_abs}");
        assert!(
            max_abs / max_logit.max(1.0) <= 1e-3,
            "max relative Δlogit = {}",
            max_abs / max_logit
        );
    }
}
/// #167 acceptance: the **qwen3** loader registers a `W_dsc` plane for exactly its
/// admissible q4_K weights — the set the GGUF index predicts, by name and count, not
/// a non-zero count — and the NB-BT kernel's own lookup (the raw weight pointer in
/// `CudaState::q4k_dsc`) finds each one.
///
/// The expected set is restated here **from the GGUF index**, independently of
/// `models::weight_reg`/`q4k_dsc`: a gate that derived `want` from the predicate under
/// test would stay green while both sides were wrong.
///
/// The negative arm loads the cached Qwen3-Q8_0 file under its own namespace and
/// requires zero planes, so an "always register" mutation cannot pass.
#[test]
#[ignore = "requires a q4_K Qwen3 GGUF and a CUDA device (#167)"]
fn f167_qwen3_q4k_registers_the_dsc_plane_exactly() {
    #[cfg(not(feature = "cuda"))]
    eprintln!("not a CUDA build; #167's q4_K Qwen3 gate is a no-op here");
    #[cfg(feature = "cuda")]
    {
        use crate::gguf::GgmlType;
        // `get()` is None until something initializes the process-wide state, and a
        // gate that skipped here would pass while measuring nothing (this gate is
        // runnable on its own, not only after the f16 arm warmed the device).
        crate::cuda::CudaState::init();
        let Some(state) = crate::cuda::CudaState::get() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let Some(path) = env_path("MINFER_F167_Q4K_GGUF", "/tmp/f167-work/qwen3-q4k.gguf") else {
            return;
        };
        let ns = "f167q4k:";
        let neg_ns = "f167neg:";
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse q4_K GGUF");
        assert_eq!(
            gguf.parts[0]
                .ctx
                .get_key_val_str("general.architecture")
                .as_deref(),
            Some("qwen3"),
            "#167's q4_K gate requires a Qwen3 GGUF"
        );

        // The expected plane set, from the index: q4_K, a whole number of 256-element
        // super-blocks per row, an even row count (the r59 row-pair staging), and a
        // payload of exactly od*(id/256)*144 bytes. Types are counted so the gate can
        // state that the file really exercises the type gate.
        let mut want: Vec<String> = Vec::new();
        let mut n_q4k = 0usize;
        let mut n_other_quant = 0usize;
        for part in &gguf.parts {
            for ti in &part.ctx.info {
                if ti.type_ == GgmlType::Q4_K {
                    n_q4k += 1;
                } else if ti.type_ != GgmlType::F32 && ti.ne[1] > 1 {
                    n_other_quant += 1;
                }
                if ti.type_ != GgmlType::Q4_K {
                    continue;
                }
                let (id, od) = (ti.ne[0] as usize, ti.ne[1] as usize);
                if id == 0 || id % 256 != 0 || od == 0 || od % 2 != 0 {
                    continue;
                }
                if ti.nbytes() != od * (id / 256) * 144 {
                    continue;
                }
                want.push(format!("{ns}{}__q4dsc{od}x{id}", ti.name));
            }
        }
        want.sort();
        assert!(
            !want.is_empty(),
            "the file has no admissible q4_K weight to gate on ({n_q4k} q4_K tensors)"
        );
        assert!(
            n_other_quant > 0,
            "the file has no non-q4_K quantized 2-D weight, so the type gate is untested"
        );

        let model = crate::models::load_model_with(
            &gguf,
            ns,
            crate::graph::offload::OffloadRequest::Layers(usize::MAX),
        )
        .expect("load q4_K model");
        assert!(
            matches!(
                crate::models::ModelDef::device(model.as_ref()),
                crate::models::Device::Cuda
            ),
            "the q4_K Qwen3 model did not come up on the device"
        );
        drop(model);

        let mut got: Vec<String> = state
            .q4dsc_planes()
            .into_iter()
            .map(|(n, _)| n)
            .filter(|n| n.starts_with(ns))
            .collect();
        got.sort();
        let bytes: usize = state
            .q4dsc_planes()
            .into_iter()
            .filter(|(n, _)| n.starts_with(ns))
            .map(|(_, b)| b)
            .sum();
        eprintln!(
            "f167 q4_K qwen3 planes: {} expected ({} q4_K + {} other quantized 2-D), \
             {} registered, {bytes} bytes in {}",
            want.len(),
            n_q4k,
            n_other_quant,
            got.len(),
            path.display()
        );
        assert_eq!(
            got, want,
            "the W_dsc plane set must be exactly the model's admissible q4_K weights"
        );
        // The NB-BT kernel's lookup is keyed on the raw weight pointer, not on the
        // plane's name: assert the map the kernel reads, per weight.
        let names: Vec<String> = want
            .iter()
            .map(|n| {
                n.trim_start_matches(ns)
                    .split("__q4dsc")
                    .next()
                    .unwrap()
                    .to_string()
            })
            .collect();
        for raw in &names {
            let key = format!("{ns}{raw}");
            assert!(
                state.q4dsc_plane_for(&key).is_some(),
                "the NB-BT kernel cannot find the plane for {key}"
            );
        }
        // Every plane is non-null and distinct (a name-only set could be aliases).
        let mut ptrs: Vec<*mut std::ffi::c_void> = Vec::new();
        for raw in &names {
            let p = state.q4dsc_plane_for(&format!("{ns}{raw}")).unwrap();
            assert!(!p.is_null(), "null plane for {raw}");
            ptrs.push(p);
        }
        ptrs.sort();
        ptrs.dedup();
        assert_eq!(
            ptrs.len(),
            names.len(),
            "two weights share one plane buffer"
        );

        // Negative arms, each under its own namespace: a q8_0 Qwen3 (whose payload
        // is *longer* than q4_K's, so the payload gate is the only refuser) and a
        // **q4_0** Qwen3 (`minfer quantize --type q4_0` of the same Q8_0 source).
        // q4_0's bytes/element ratio equals q4_K's exactly (18/32 == 144/256), so a
        // q4_0 payload passes the payload contract — the *type* gate is the only
        // thing that can refuse it, and this arm is what makes a type-gate bypass
        // show up in a real-model gate rather than only in the pure test.
        let mut neg_arm = |label: &str, key: &str, default: &str, arm_ns: &str| {
            if let Some(neg) = env_path(key, default) {
                let ngguf = crate::gguf::load_gguf_model(&neg).expect("parse negative GGUF");
                assert_eq!(
                    ngguf.parts[0]
                        .ctx
                        .get_key_val_str("general.architecture")
                        .as_deref(),
                    Some("qwen3"),
                    "the {label} negative arm requires a Qwen3 GGUF"
                );
                let nq = crate::models::load_model_with(
                    &ngguf,
                    arm_ns,
                    crate::graph::offload::OffloadRequest::Layers(usize::MAX),
                )
                .expect("load negative model");
                drop(nq);
                let leaked: Vec<String> = state
                    .q4dsc_planes()
                    .into_iter()
                    .map(|(n, _)| n)
                    .filter(|n| n.starts_with(arm_ns))
                    .collect();
                assert!(
                    leaked.is_empty(),
                    "a {label} Qwen3 model registered q4dsc planes: {leaked:?}"
                );
                eprintln!("f167 q4_K gate: {label} negative arm registers 0 planes");
            } else {
                eprintln!("f167 q4_K gate: no {label} negative arm ({key} absent)");
            }
        };
        neg_arm(
            "q8_0",
            "MINFER_F167_NEG_GGUF",
            "~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf",
            neg_ns,
        );
        neg_arm(
            "q4_0",
            "MINFER_F167_NEG_Q40_GGUF",
            "/tmp/f167-work/qwen3-q4_0.gguf",
            "f167negq40:",
        );
    }
}
