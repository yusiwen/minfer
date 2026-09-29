//! `#[cfg(test)] mod tail_tests` for `src/models/qwen2/graph.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::batch::Batch;
use crate::models::ModelDef;

/// Dump directory for the `dump_real_*` debug helpers.
///
/// They exist to hand real tensor slices to an external parity check, so the
/// files are meant to stay on disk. Nothing created the directory, so both
/// helpers panicked on the *write* instead of doing their job (issue #82);
/// creating it is the whole fix.
fn dump_dir() -> std::path::PathBuf {
    let dir = std::path::PathBuf::from("/tmp/minfer_phase7");
    std::fs::create_dir_all(&dir).expect("create the dump directory");
    dir
}

fn model_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
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
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
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
                        s == wo || matches!(g.nodes[s].op, Op::GetRows) && g.nodes[s].src[0] == wo
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
                kv_map: false,
                gpu: false,
                gpu_layers: usize::MAX, // E5: no offload limit in this fixture
                kv_format: crate::graph::kvformat::KvFormat::F32,
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
        let pos: Vec<usize> = (0..nt).collect();
        let pos32: Vec<u32> = pos.iter().map(|&p| p as u32).collect();
        alloc.fill_input_i32(&graph, "token_ids", &ids32).unwrap();
        alloc.fill_input_i32(&graph, "positions", &pos32).unwrap();
        // E2: one sequence, one batch — the production fill entry point.
        alloc
            .fill_batch_inputs(&graph, &Batch::single(&ids32, &pos))
            .unwrap();
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
                        s == wi || matches!(g.nodes[s].op, Op::GetRows) && g.nodes[s].src[0] == wi
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
                    kv_map: false,
                    gpu: true,
                    gpu_layers: usize::MAX, // E5 fixture: no offload limit
                    kv_format: crate::graph::kvformat::KvFormat::F32,
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
                FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                    Some(crate::graph::Backend::CPU) => Some(0),
                    Some(crate::graph::Backend::METAL) => Some(1),
                    _ => None,
                });
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
            alloc
                .fill_batch_inputs(&graph, &Batch::single(&[tok_ids[0]], &[0]))
                .unwrap();
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
                    kv_map: false,
                    gpu: true,
                    gpu_layers: usize::MAX, // E5 fixture: no offload limit
                    kv_format: crate::graph::kvformat::KvFormat::F32,
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
                FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                    Some(crate::graph::Backend::CPU) => Some(0),
                    Some(crate::graph::Backend::METAL) => Some(1),
                    _ => None,
                });
            }
            alloc.alloc_graph(&graph).unwrap();
            alloc
                .fill_input_i32(&graph, "token_ids", &[tok_ids[0]])
                .unwrap();
            alloc.fill_input_i32(&graph, "positions", &[0]).unwrap();
            alloc
                .fill_batch_inputs(&graph, &Batch::single(&[tok_ids[0]], &[0]))
                .unwrap();
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
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
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
            let tok7 =
                crate::tokenizer::Tokenizer::load(&gguf7.parts[0].ctx).expect("tokenizer load");
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
                let bytes = info.type_.type_size() * n as usize / info.type_.blck_size() as usize;
                found = Some((info.type_, info.ne, &part.data[start..start + bytes]));
                break;
            }
        }
        match found {
            Some((ty, ne, data)) => {
                let out = dump_dir().join(format!("real_{}.bin", name.replace('.', "_")));
                std::fs::write(&out, data).unwrap();
                spec.push_str(&format!(
                    "{name}: type={ty:?} ne={ne:?} bytes={} -> {}\n",
                    data.len(),
                    out.display()
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
                let bytes = info.type_.type_size() * n as usize / info.type_.blck_size() as usize;
                found = Some((info.type_, info.ne, &part.data[start..start + bytes]));
                break;
            }
        }
        match found {
            Some((ty, ne, data)) => {
                let out = dump_dir().join(format!("real05_{}.bin", name.replace('.', "_")));
                std::fs::write(&out, data).unwrap();
                spec.push_str(&format!(
                    "{name}: type={ty:?} ne={ne:?} bytes={} -> {}\n",
                    data.len(),
                    out.display()
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
    let tok = crate::tokenizer::Tokenizer::load(ctx).expect("tokenizer load");
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
        mirostat_tau: 5.0,
        system_prompt: None,
    };
    let greedy = crate::conversation::TurnParams {
        n_predict: 16,
        sampler: crate::sampler::SamplerConfig {
            temp: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            ..crate::sampler::SamplerConfig::default()
        },
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
        mirostat_tau: 5.0,
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
        mirostat_tau: 5.0,
        system_prompt: None,
    };
    let mut conv_b = crate::conversation::Conversation::new(spec_b);
    let mut engine_b = crate::conversation::GraphEngine::new(model.as_ref(), 1024);
    let hist = conv_a.messages_to_json();
    let msgs = crate::conversation::Conversation::messages_from_json(&hist).expect("history json");
    conv_b
        .load_history(msgs, &tok, &mut engine_b)
        .expect("load history");
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
