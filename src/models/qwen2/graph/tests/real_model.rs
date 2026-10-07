//! Real-model and Metal logits parity gates.
//!
//! Split out of `src/models/qwen2/graph/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

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
    use crate::graph::batch::Batch;
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
    let tok = crate::tokenizer::Tokenizer::load(ctx).expect("tokenizer load");
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
                kv_map: false,
                gpu: false,
                gpu_layers: usize::MAX, // E5: no offload limit in this fixture
                kv_format: crate::graph::kvformat::KvFormat::F32,
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
        let pos: Vec<usize> = (0..nt).collect();
        let pos32: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
        alloc.fill_input_i32(&graph, "token_ids", &ids32).unwrap();
        alloc.fill_input_i32(&graph, "positions", &pos32).unwrap();
        // E2 — the entry point `forward_cached`/`forward_batch` themselves drive.
        alloc
            .fill_batch_inputs(&graph, &Batch::single(&ids32, &pos))
            .unwrap();
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
                kv_map: false,
                gpu: false,
                gpu_layers: usize::MAX, // E5: no offload limit in this fixture
                kv_format: crate::graph::kvformat::KvFormat::F32,
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
        // E2: the decode step is a one-token batch on the same sequence.
        alloc
            .fill_batch_inputs(&dgraph, &Batch::single(&[next], &[nt]))
            .unwrap();
        sched.execute(&dgraph, &mut alloc).unwrap();
        let dlogits = alloc.copy_to_cpu(dgraph.outputs[0]).unwrap();
        (prefill_l, dlogits.to_vec())
    }

    // NOTE: prefill and decode share one GraphAllocator so the KV persists
    // NOTE: both runs share one GraphAllocator so the KV persists across
    // the prefill -> decode transition (like the real loop).
    let n_ctx = q2.hparams.max_seq_len as usize;
    let lf = model.forward(&ids, &positions, 1, n_ctx);
    let next = argmax(&lf);
    let lf2 = model.forward(&[next], &[ids.len()], 1, n_ctx);
    let (lg, lg2) = run_prefill_decode(q2, &ids, next, n_ctx);
    compare("prefill", &lf, &lg);
    compare("decode", &lf2, &lg2);
}
/// Phase 3 / [#324]: the model-level CPU-vs-Metal logits gate, restored.
///
/// The pre-#324 body called `model.forward` and `model.forward_graph`, but in
/// the graph era both dispatch to `Qwen2Graph::forward` under the same
/// `Qwen2Graph::device` decision, so it compared the Metal graph with itself
/// (measured max |Δ| = 0 — a vacuous gate). This version loads the **same
/// cached 0.5B Q4_0** into two engines — a `Layers(0)` CPU engine and an
/// explicit full-plan Metal engine (`OffloadRequest::Layers(usize::MAX)`) —
/// drives the same greedy continuation through the production
/// `ModelDef::forward_graph_cached`, asserts the two graphs genuinely differ in
/// backend assignment (the CPU graph has no `METAL` node; the Metal graph
/// assigns `METAL` nodes), and compares the final-step logits plus the greedy
/// continuation. This is the shape of the CUDA #141 gate and the Metal #164
/// f16 gate.
///
/// Bar named before measuring (gate-contract rules 3/5): unlike the f16 gate,
/// this compares the **quantized** Q4_0 path, where the CPU quantizes
/// activations to Q8_0 on the fly while Metal reads f32 (AGENTS rule 9), so
/// the class is quantization, not accumulation order. The bar is `MAX_ABS`
/// absolute / `MAX_REL` relative (taken from the q8_0 weight-quantisation
/// class of `docs/GGUF-TOOLING.md`: 1.0 absolute, 3x the observed), with the
/// load-bearing arm the greedy continuation pinned to the literal the CPU arm
/// produces. Measured on `macbook (macOS 27.0.1, Apple M4 Pro)`, 2026-10-07,
/// `cargo test --release --bin minfer -- --nocapture
/// graph_metal_matches_cpu_logits`: max |Δlogit| **0.347** absolute /
/// **1.57e-2** relative against max |logit| 22.04, greedy
/// `[12095, 11, 323, 432]` on both engines, metal-arm 248 METAL / 0 CPU nodes
/// and cpu-arm 0 METAL / 440 CPU nodes. The record lives in
/// `docs/METAL-BACKEND-DESIGN.md` §7.3.
#[test]
fn graph_metal_matches_cpu_logits() {
    /// Absolute logits bar (gross-error detector).
    const MAX_ABS: f32 = 1.0;
    /// Relative logits bar against the run's max |logit|.
    const MAX_REL: f32 = 5e-2;
    /// Greedy continuation the CPU arm must reproduce — the value arm
    /// (rule 1), independent of the Metal path. Pinned for the cached 0.5B
    /// Q4_0 on "The capital of France is".
    const PINNED_GREEDY: [u32; 4] = [12095, 11, 323, 432];

    #[cfg(target_os = "macos")]
    let _g = crate::metal::metal_test_lock();
    #[cfg(not(target_os = "macos"))]
    {
        eprintln!("not macOS; skipping");
        return;
    }
    #[cfg(target_os = "macos")]
    {
        use crate::graph::cache::GraphCache;
        use crate::graph::offload::OffloadRequest;
        use crate::graph::Backend;
        use crate::models::{Device, ModelDef};

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
            return;
        };
        // The Metal device must be up **before** the load, or the loader decides
        // the weights are not usable there and answers `Device::Cpu`.
        crate::metal::MpsState::init();
        if crate::metal::MpsState::get().is_none() {
            eprintln!("no Metal device; skipping");
            return;
        }
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        // Keep the weight registry stable for this whole test: a parallel
        // test loading a different architecture swaps same-named entries,
        // which would flip the CUDA gate mid-test (persistent KV regions
        // were allocated under the earlier decision).
        #[cfg(feature = "cuda")]
        let _model_load_guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
        let ids = tok.encode("The capital of France is");
        assert!(!ids.is_empty());
        let n_ctx = 512usize;
        let steps = PINNED_GREEDY.len();

        // Metal arm: the full plan, explicit (no environment involvement). Its
        // weights are registered under the primary (empty) namespace.
        let dev = crate::models::load_model_with(&gguf, "", OffloadRequest::Layers(usize::MAX))
            .expect("Metal load");
        assert_eq!(
            dev.device(),
            Device::Metal,
            "the 0.5B Q4_0 must be a Metal model on this box"
        );
        // CPU arm: `Layers(0)` — the same engine, fenced to no device block, so
        // `Qwen2Graph::device` answers Cpu. A distinct namespace keeps the
        // name-keyed Metal registry from making this engine look registered.
        let cpu = crate::models::load_model_with(&gguf, "cpu324.", OffloadRequest::Layers(0))
            .expect("CPU load");
        assert_eq!(
            cpu.device(),
            Device::Cpu,
            "the CPU arm must not be a device model"
        );

        // Greedy continuation + the built graph's assignment, per engine.
        let run = |model: &dyn ModelDef| -> (Vec<f32>, Vec<u32>, usize, usize) {
            let mut cache = GraphCache::new();
            let n = ids.len();
            let mut positions: Vec<usize> = (0..n).collect();
            let mut logits = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
            let mut toks = Vec::with_capacity(steps);
            for step in 0..steps {
                let next = argmax(&logits);
                toks.push(next);
                positions = vec![n + step];
                logits = model.forward_graph_cached(&[next], &positions, 1, n_ctx, &mut cache);
            }
            let (graph, _alloc) = cache.current().expect("a built graph");
            let n_metal = graph
                .nodes
                .iter()
                .filter(|nd| nd.backend == Some(Backend::METAL))
                .count();
            let n_cpu = graph
                .nodes
                .iter()
                .filter(|nd| nd.backend == Some(Backend::CPU))
                .count();
            (logits, toks, n_metal, n_cpu)
        };

        let (logits_metal, toks_metal, n_metal, n_metal_cpu) = run(dev.as_ref());
        let (logits_cpu, toks_cpu, n_cpu_metal, n_cpu_cpu) = run(cpu.as_ref());

        // Control arm (rule 2): the two sides must differ in the property under
        // test — backend assignment. A CPU engine cannot run a Metal graph, and
        // the Metal engine must actually place nodes on the device.
        eprintln!(
            "[metal graph] graph nodes: metal-arm {n_metal} METAL / {n_metal_cpu} CPU; cpu-arm {n_cpu_metal} METAL / {n_cpu_cpu} CPU"
        );
        assert!(
            n_metal > 0,
            "the Metal arm built no METAL node (vacuous gate)"
        );
        assert_eq!(n_cpu_metal, 0, "the CPU arm must not place a node on Metal");

        // Value arm (rule 1): the CPU engine's continuation is pinned.
        assert_eq!(
            toks_cpu.as_slice(),
            &PINNED_GREEDY,
            "CPU greedy continuation moved"
        );
        // The load-bearing parity claim.
        assert_eq!(
            toks_metal, toks_cpu,
            "greedy continuation differs CPU vs Metal"
        );

        assert_eq!(logits_metal.len(), logits_cpu.len());
        let max_abs = logits_metal
            .iter()
            .zip(logits_cpu.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let max_logit = logits_metal.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let max_rel = max_abs / max_logit.max(1.0);
        eprintln!(
            "[metal graph] final-step logits: max |Δ| = {max_abs:.3e} (bar {MAX_ABS}), relative {max_rel:.3e} (bar {MAX_REL}), max |logit| = {max_logit:.3e}; greedy CPU={toks_cpu:?} Metal={toks_metal:?}"
        );
        assert!(
            max_abs <= MAX_ABS,
            "CPU vs Metal max |Δlogit| = {max_abs:.3e} exceeds bar {MAX_ABS}"
        );
        assert!(
            max_rel <= MAX_REL,
            "CPU vs Metal relative Δlogit = {max_rel:.3e} exceeds bar {MAX_REL}"
        );
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
        let pos: Vec<usize> = (0..nt).collect();
        let pos32: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
        ca.fill_input_i32(&g, "positions", &pos32).unwrap();
        ca.fill_batch_inputs(&g, &crate::graph::batch::Batch::single(&ids, &pos))
            .unwrap();
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
                n.backend = Some(crate::graph::Backend::METAL); // FULL layer-0 on Metal
            }
            let mut alloc = crate::graph::alloc::GraphAllocator::new();
            Qwen2Graph::register_graph_weights(q2, &mut alloc);
            alloc.enable_metal();
            alloc.alloc_graph(&g2).unwrap();
            alloc.fill_input_i32(&g2, "token_ids", &ids).unwrap();
            let pos: Vec<usize> = (0..nt).collect();
            let pos32: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
            alloc.fill_input_i32(&g2, "positions", &pos32).unwrap();
            alloc
                .fill_batch_inputs(&g2, &crate::graph::batch::Batch::single(&ids, &pos))
                .unwrap();
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
            n.backend = Some(crate::graph::Backend::METAL);
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
                            acc += ((byte & 0x0F) as i8 - 8) as f32 * d * xd[t * ne + b * 32 + j];
                            acc +=
                                ((byte >> 4) as i8 - 8) as f32 * d * xd[t * ne + b * 32 + j + 16];
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
