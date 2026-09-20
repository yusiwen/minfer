// Logical-positions experiments — pre-C6 research artifact (see README.md).
//
// NOT part of the crate: this file lives outside `src/` and `tests/`, so cargo never
// compiles it. The two functions below are the verbatim test-module code that was
// developed on `feat/logical-positions` before C6 and kept in a stash; they are archived
// here as the record of how the cell-offset divergence was attributed to RoPE. C6
// removed that coupling, so they are no longer gates — the offset tests in
// `src/models/qwen2/graph.rs` now assert bitwise equality.
//
// To run them, append them (keeping their 4-space indentation, i.e. as members of
// `mod tests`) to `src/models/qwen2/graph.rs` and use the commands in README.md. They do
// not compile against post-C6 master unchanged: the runner builds positions as
// `start..start + n` and relies on the store writing at those rows, whereas C6 makes
// positions sequence-relative and resolves the KV row through the `cells` input.

    /// EXPERIMENT 1 (plan §14 row 9, causal injection): is the offset divergence
    /// caused by **nothing but** the rope's own rounding?
    ///
    /// Both runs execute the model's graph node by node at cell 0 and cell 8. The
    /// only input the offset touches is the rope (`positions` feeds the two rope
    /// nodes and the store; the store's *rows* do not change any value, and the
    /// relative windows are equal). So if run C's rope outputs are **overwritten
    /// with run A's captured values** right after each rope node executes, every
    /// remaining difference should vanish: the logits must come out **bitwise
    /// identical** to A's. If they do not, something else is position-dependent
    /// and the attribution is wrong.
    ///
    /// Test-only by construction: no production code changes, and the injection
    /// uses the existing `Backend::write_host` (a read-modify-write of the node's
    /// window), so the release build is untouched.
    #[test]
    fn offset_divergence_is_caused_by_the_rope_rounding_alone() {
        use crate::graph::batch::Batch;
        use crate::graph::backend::{Backend as _, KvProvider};
        use crate::graph::fusion::FusionPass;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        use crate::models::ModelDef;
        use std::collections::HashMap;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        let n = ids.len();
        let n_ctx = 256;

        struct RunOut {
            logits: Vec<f32>,
            ropes: Vec<(crate::graph::NodeId, Vec<f32>)>,
        }
        let run_nodes = |start: usize,
                         inject: Option<&HashMap<crate::graph::NodeId, Vec<f32>>>,
                         add: Option<f32>|
         -> RunOut {
            let params = GraphParams {
                n_tokens: n,
                n_out: 1,
                gtype: GraphType::Prefill,
                cparams: CParams {
                    n_ctx,
                    flash_attn: false,
                    explicit_span: true,
                    gpu: false,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            let mut graph = model.build_graph(&params);
            let rope_ids: Vec<crate::graph::NodeId> = graph
                .nodes
                .iter()
                .filter(|nd| matches!(nd.op, crate::graph::ops::Op::RoPE { .. }))
                .map(|nd| nd.id)
                .collect();
            let out0 = graph.outputs[0];

            let mut alloc = GraphAllocator::new();
            alloc.kv_set_capacity(n_ctx);
            if start > 0 {
                alloc.kv_reserve_seq(3, start).expect("holder");
            }
            alloc.kv_reserve_seq(7, n + 4).expect("subject");
            if start == 0 {
                alloc.kv_reserve_seq(3, 4).expect("dummy");
            }
            let concrete = model
                .as_any()
                .downcast_ref::<crate::models::qwen2::Qwen2Model>()
                .expect("Qwen2Model");
            Qwen2Graph::register_graph_weights(concrete, &mut alloc);
            let mut sched = BackendScheduler::new();
            sched.assign_backends(&mut graph, &mut alloc);
            {
                let backends: Vec<&dyn crate::graph::backend::Backend> = vec![alloc.cpu()];
                FusionPass::new().run(&mut graph, &backends, &|g, id| {
                    match g.node(id).backend {
                        Some(crate::graph::Backend::CPU) => Some(0),
                        _ => None,
                    }
                });
            }
            alloc.alloc_graph(&graph).expect("alloc");

            let batch = Batch::new(ids.clone(), (start..start + n).collect(), vec![7u32; n]);
            alloc.fill_input_i32(&graph, "token_ids", &ids).unwrap();
            let pos: Vec<u32> = (start..start + n).map(|p| p as u32).collect();
            alloc.fill_input_i32(&graph, "positions", &pos).unwrap();
            alloc.fill_batch_inputs(&graph, &batch).unwrap();
            if graph
                .inputs
                .iter()
                .any(|&i| graph.node(i).name == "tail_ids")
            {
                let rows: Vec<u32> = batch.out_rows(1).iter().map(|&r| r as u32).collect();
                alloc.fill_input_i32(&graph, "tail_ids", &rows).unwrap();
            }

            let order: Vec<crate::graph::NodeId> =
                graph.nodes.iter().map(|nd| nd.id).collect();
            let mut ropes: Vec<(crate::graph::NodeId, Vec<f32>)> = Vec::new();
            for id in order {
                let node = graph.node(id);
                if matches!(node.op, crate::graph::ops::Op::Input) {
                    continue;
                }
                let Some(out) = alloc.node_buffer(id) else {
                    continue;
                };
                let ins: Vec<crate::graph::BufRef> = node
                    .src
                    .iter()
                    .map(|&s| alloc.node_buffer(s).expect("input buffer"))
                    .collect();
                let kv = match &node.op {
                    crate::graph::ops::Op::KvcacheStore { layer } => alloc.kv_pair(*layer),
                    crate::graph::ops::Op::Attn { .. } => match &node.meta {
                        crate::graph::ops::NodeMeta::Attn(m) => alloc.kv_pair(m.layer),
                        _ => None,
                    },
                    _ => None,
                };
                alloc
                    .cpu_mut()
                    .execute_node(node, &ins, out, kv)
                    .unwrap_or_else(|e| panic!("node {id} ({}): {e}", node.name));

                if rope_ids.contains(&id) {
                    let mut buf = alloc.cpu().read_host(out.id).expect("rope read").to_vec();
                    let window = buf[out.offset..out.offset + out.len].to_vec();
                    let mut changed = false;
                    if let Some(map) = inject {
                        if let Some(src) = map.get(&id) {
                            buf[out.offset..out.offset + out.len].copy_from_slice(src);
                            changed = true;
                        }
                    }
                    if let Some(d) = add {
                        for v in buf[out.offset..out.offset + out.len].iter_mut() {
                            *v += d;
                        }
                        changed = true;
                    }
                    if changed {
                        alloc.cpu_mut().write_host(out.id, &buf).expect("rope write");
                    }
                    ropes.push((id, window));
                }
            }
            let logits = alloc.copy_to_cpu(out0).expect("logits");
            RunOut { logits, ropes }
        };
        let d = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        let a = run_nodes(0, None, None);
        let c = run_nodes(8, None, None);
        let offset_delta = d(&a.logits, &c.logits);
        let injected: HashMap<crate::graph::NodeId, Vec<f32>> = a.ropes.iter().cloned().collect();
        let ci = run_nodes(8, Some(&injected), None);
        let injected_delta = d(&a.logits, &ci.logits);
        eprintln!(
            "[inject] rope nodes injected: {} | logits delta offset {} -> injected {}",
            injected.len(),
            offset_delta,
            injected_delta
        );
        assert!(offset_delta > 0.0, "the offset effect must be present");
        assert_eq!(
            injected_delta, 0.0,
            "injecting A's rope outputs into C must make the logits bitwise identical; \
             a non-zero result means something else is position-dependent"
        );
    }

    /// EXPERIMENT 2 (plan §14 row 9): does a **distributed** perturbation of the
    /// rope outputs, of the magnitude the offset itself produces, move the logits
    /// as much as the offset does — and does the response jump at the scale of the
    /// CPU's Q8_0 activation quantisation step?
    ///
    /// The round-5 single-element probe was not equivalent to the offset (one
    /// element nudged by 1e-5 can stay inside its quantisation bin), so this adds
    /// the same delta to **every element of every rope output** and sweeps the
    /// magnitude. Printed, not asserted: the shape of the response is the result.
    #[test]
    fn a_distributed_rope_perturbation_scales_like_the_offset() {
        use crate::graph::batch::Batch;
        use crate::graph::backend::{Backend as _, KvProvider};
        use crate::graph::fusion::FusionPass;
        use crate::graph::params::{CParams, GraphParams, GraphType};
        use crate::graph::scheduler::BackendScheduler;
        use crate::models::ModelDef;
        use std::collections::HashMap;

        let Some(path) = cached_model_path() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode("The capital of France is");
        let n = ids.len();
        let n_ctx = 256;

        // Same runner, without the rope capture (experiment 1 keeps it).
        let run = |start: usize, add: Option<f32>| -> Vec<f32> {
            let params = GraphParams {
                n_tokens: n,
                n_out: 1,
                gtype: GraphType::Prefill,
                cparams: CParams {
                    n_ctx,
                    flash_attn: false,
                    explicit_span: true,
                    gpu: false,
                    fuse_qkv: false,
                    fuse_ffn: false,
                },
                weights_version: 1,
            };
            let mut graph = model.build_graph(&params);
            let rope_ids: Vec<crate::graph::NodeId> = graph
                .nodes
                .iter()
                .filter(|nd| matches!(nd.op, crate::graph::ops::Op::RoPE { .. }))
                .map(|nd| nd.id)
                .collect();
            let out0 = graph.outputs[0];
            let mut alloc = GraphAllocator::new();
            alloc.kv_set_capacity(n_ctx);
            if start > 0 {
                alloc.kv_reserve_seq(3, start).expect("holder");
            }
            alloc.kv_reserve_seq(7, n + 4).expect("subject");
            if start == 0 {
                alloc.kv_reserve_seq(3, 4).expect("dummy");
            }
            let concrete = model
                .as_any()
                .downcast_ref::<crate::models::qwen2::Qwen2Model>()
                .expect("Qwen2Model");
            Qwen2Graph::register_graph_weights(concrete, &mut alloc);
            let mut sched = BackendScheduler::new();
            sched.assign_backends(&mut graph, &mut alloc);
            {
                let backends: Vec<&dyn crate::graph::backend::Backend> = vec![alloc.cpu()];
                FusionPass::new().run(&mut graph, &backends, &|g, id| {
                    match g.node(id).backend {
                        Some(crate::graph::Backend::CPU) => Some(0),
                        _ => None,
                    }
                });
            }
            alloc.alloc_graph(&graph).expect("alloc");
            let batch = Batch::new(ids.clone(), (start..start + n).collect(), vec![7u32; n]);
            alloc.fill_input_i32(&graph, "token_ids", &ids).unwrap();
            let pos: Vec<u32> = (start..start + n).map(|p| p as u32).collect();
            alloc.fill_input_i32(&graph, "positions", &pos).unwrap();
            alloc.fill_batch_inputs(&graph, &batch).unwrap();
            if graph
                .inputs
                .iter()
                .any(|&i| graph.node(i).name == "tail_ids")
            {
                let rows: Vec<u32> = batch.out_rows(1).iter().map(|&r| r as u32).collect();
                alloc.fill_input_i32(&graph, "tail_ids", &rows).unwrap();
            }
            for id in graph.nodes.iter().map(|nd| nd.id).collect::<Vec<_>>() {
                let node = graph.node(id);
                if matches!(node.op, crate::graph::ops::Op::Input) {
                    continue;
                }
                let Some(out) = alloc.node_buffer(id) else {
                    continue;
                };
                let ins: Vec<crate::graph::BufRef> = node
                    .src
                    .iter()
                    .map(|&s| alloc.node_buffer(s).expect("input buffer"))
                    .collect();
                let kv = match &node.op {
                    crate::graph::ops::Op::KvcacheStore { layer } => alloc.kv_pair(*layer),
                    crate::graph::ops::Op::Attn { .. } => match &node.meta {
                        crate::graph::ops::NodeMeta::Attn(m) => alloc.kv_pair(m.layer),
                        _ => None,
                    },
                    _ => None,
                };
                alloc
                    .cpu_mut()
                    .execute_node(node, &ins, out, kv)
                    .unwrap_or_else(|e| panic!("node {id} ({}): {e}", node.name));
                if rope_ids.contains(&id) {
                    if let Some(delta) = add {
                        let mut buf = alloc.cpu().read_host(out.id).expect("read").to_vec();
                        for v in buf[out.offset..out.offset + out.len].iter_mut() {
                            *v += delta;
                        }
                        alloc.cpu_mut().write_host(out.id, &buf).expect("write");
                    }
                }
            }
            alloc.copy_to_cpu(out0).expect("logits")
        };
        let d = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };

        let argmax = |v: &[f32]| -> usize {
            v.iter()
                .enumerate()
                .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &x)| {
                    if x > bv {
                        (i, x)
                    } else {
                        (bi, bv)
                    }
                })
                .0
        };
        let reference = run(0, None);
        let offset_run = run(8, None);
        eprintln!(
            "[distributed] offset 0-vs-8: {} | greedy {} vs {}",
            d(&reference, &offset_run),
            argmax(&reference),
            argmax(&offset_run)
        );
        let mut previous = 0.0f32;
        for delta in [0.0f32, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2] {
            let got_run = run(0, Some(delta));
            let got = d(&reference, &got_run);
            eprintln!(
                "[distributed] rope +{delta:e}: logits delta {got} (prev {previous}) | greedy {}",
                argmax(&got_run)
            );
            previous = got;
        }
        assert!(d(&reference, &offset_run) > 0.0);
    }

