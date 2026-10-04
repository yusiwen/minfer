//! Batch composition, sequence-count independence and offset sensitivity.
//!
//! Split out of `src/models/qwen2/graph/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// E2's evidence gate: a forward carrying two sequences must equal running
/// those sequences one at a time, **bitwise** — the prefill (both prompts in
/// one forward) and a decode step (one token each).
///
/// Both sides use the *same* KV layout — one arena, sequence 7 reserved at
/// `[0, la)` and sequence 9 at `[la, la + lb)` — so the only difference is
/// whether one forward carries two sequences or two forwards carry one each.
/// The batched side takes the windowed attention path
/// (`Op::Attn { explicit_span: true }`), where sequence 9's window starts at
/// `la`; equal logits mean the windows are per-sequence and the KV rows do
/// not leak.
#[test]
fn a_two_sequence_batch_matches_two_single_sequence_forwards() {
    use crate::graph::batch::Batch;
    use crate::graph::cache::GraphCache;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the batch test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let nv = model.n_vocab();
    let a = tok.encode("The capital of France is");
    let b = tok.encode("The capital of Japan is");
    let (la, lb) = (a.len(), b.len());
    let (s7, s9) = (7u32, 9u32);

    // The layout both sides use: two reserved runs, deterministic placement.
    let layout = |cache: &mut GraphCache| -> (usize, usize) {
        cache.alloc().kv_set_capacity(n_ctx);
        let r7 = cache.alloc().kv_reserve_seq(s7, la + 4).expect("reserve 7");
        let r9 = cache.alloc().kv_reserve_seq(s9, lb + 4).expect("reserve 9");
        (r7.start, r9.start)
    };

    // ---- batched: both sequences, one forward per step ----
    let mut batched = GraphCache::new();
    let (p7, p9) = layout(&mut batched);
    let pos7: Vec<usize> = (0..la).collect();
    let pos9: Vec<usize> = (0..lb).collect();

    let mut tokens = a.clone();
    tokens.extend_from_slice(&b);
    let mut positions = pos7.clone();
    positions.extend_from_slice(&pos9);
    let mut seq_ids = vec![s7; la];
    seq_ids.extend(std::iter::repeat(s9).take(lb));
    let l_pre = model.forward_batch(
        &Batch::new(tokens, positions, seq_ids),
        1,
        n_ctx,
        &mut batched,
    );
    assert_eq!(l_pre.len(), 2 * nv, "one logits row per sequence");
    let (pre7, pre9) = (l_pre[..nv].to_vec(), l_pre[nv..].to_vec());

    let (t7, t9) = (argmax(&pre7), argmax(&pre9));
    let l_step = model.forward_batch(
        &Batch::new(vec![t7, t9], vec![la, lb], vec![s7, s9]),
        1,
        n_ctx,
        &mut batched,
    );
    assert_eq!(l_step.len(), 2 * nv);

    // ---- reference: the same layout, one sequence per forward ----
    let mut ref7 = GraphCache::new();
    let (q7, _) = layout(&mut ref7);
    assert_eq!(q7, p7, "the reference must place sequence 7 identically");
    let r_pre7 = model.forward_batch(
        &Batch::new(a.clone(), pos7.clone(), vec![s7; la]),
        1,
        n_ctx,
        &mut ref7,
    );
    let r_next7 = model.forward_batch(
        &Batch::new(vec![t7], vec![la], vec![s7]),
        1,
        n_ctx,
        &mut ref7,
    );
    assert_eq!(r_pre7.len(), nv);
    assert_eq!(r_next7.len(), nv);

    let mut ref9 = GraphCache::new();
    let (_, q9) = layout(&mut ref9);
    assert_eq!(q9, p9, "the reference must place sequence 9 identically");
    let r_pre9 = model.forward_batch(
        &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
        1,
        n_ctx,
        &mut ref9,
    );
    let r_next9 = model.forward_batch(
        &Batch::new(vec![t9], vec![lb], vec![s9]),
        1,
        n_ctx,
        &mut ref9,
    );

    // ---- all four comparisons ----
    // The batch carries both sequences in one forward, the reference carries
    // them one per forward: cross-shape, so bitwise on CPU and the named
    // CUDA class on a device. `batch_order_does_not_change_a_sequences_logits`
    // is the same-shape gate that stays bitwise on both.
    assert_across_shapes(&format!("batched prefill of sequence {s7}"), &pre7, &r_pre7);
    assert_across_shapes(&format!("batched prefill of sequence {s9}"), &pre9, &r_pre9);
    assert_across_shapes(
        &format!("batched decode of sequence {s7}"),
        &l_step[..nv],
        &r_next7,
    );
    assert_across_shapes(
        &format!("batched decode of sequence {s9}"),
        &l_step[nv..],
        &r_next9,
    );
    // The fixture must discriminate: two sequences in one forward must not
    // collapse onto the same answer.
    assert_ne!(
        argmax(&l_step[..nv]),
        argmax(&l_step[nv..]),
        "both sequences produced the same argmax - the fixture is not discriminating"
    );
}
/// The offset-sensitivity finding (plan §14 row 9), narrowed to what is
/// actually established — and the two properties that *do* hold are asserted
/// here so the day someone changes the forward they find out immediately.
///
/// Measured on the 0.5B, same prompt, same relative window, no compaction:
///
/// - the forward is **deterministic** (the same cell twice is bitwise equal);
/// - **one** query token is offset-invariant *by construction* — a single key
///   makes the softmax weight 1 and V is unrotated, so any offset must give
///   bit-identical logits, and it does;
/// - from **two** query tokens on, the logits differ (2.6% relative at an
///   8-cell offset) even though layer 0's stored V rows are **bit-identical**,
///   the stored K matches the cell rotation to ~1e-5 (`rope_shift_kv` with
///   `delta = 8` aligns cell 8 back onto cell 0), the CPU attention reads the
///   explicit `attn_span` (an unfilled span is an error, so a silent
///   `[0, pos)` fallback cannot be it), and Q/K carry the same freq table
///   (only `n_head` differs, and the table depends on `hd`).
///
/// The minimal hand-built graph then exonerated the ops (see
/// `the_minimal_attention_graph_is_offset_invariant_to_rounding`: ≤ 1.2e-7 at
/// the model's own shape, and equal for a 1-cell and an 8-cell offset), a
/// layer bisect put the entry point at layer 0's attention output (layer 0's V
/// bit-identical, layer 1's differing by 3.6e-3), and a layout control (the
/// same subject cells with the reservation split in two) came out
/// bit-identical — so what remains is the rotation's own rounding, amplified
/// by depth. Plan §14 row 9 carries the full attribution and C3's acceptance:
/// byte-exact K/V plus a surviving greedy token, with the logits' tail in the
/// named amplified-rounding class.
#[test]
fn offset_sensitivity_is_narrowed_to_multi_query_attention() {
    use crate::graph::batch::Batch;
    use crate::graph::cache::GraphCache;
    use crate::graph::kvcache::KvRope;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let full = tok.encode("The capital of France is");
    let n_ctx = 256;

    // Both runs must take the *same* attention instantiation, or the
    // comparison silently measures "causal vs explicit span" instead of the
    // offset: a run with a single sequence at cell 0 is causal, one with a
    // reservation below it is explicit. A second reservation (a holder when
    // the subject is offset, a dummy after it otherwise) makes both explicit.
    let run = |ids: Vec<u32>, start: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let n = ids.len();
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        if start > 0 {
            cache.alloc().kv_reserve_seq(3, start).expect("holder");
        }
        cache.alloc().kv_reserve_seq(7, n + 4).expect("subject");
        if start == 0 {
            cache.alloc().kv_reserve_seq(3, 4).expect("dummy");
        }
        let l = model.forward_batch(
            &Batch::new(ids, (0..n).collect(), vec![7u32; n]),
            1,
            n_ctx,
            &mut cache,
        );
        let (k, v) = cache.alloc().copy_kv_to_cpu(0).expect("kv read");
        (l, k, v)
    };
    let d = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };

    // 1 token is the discriminator: with a single key the softmax weight is 1
    // and V is unrotated, so *any* offset must give bit-identical logits.
    let (freq_base, freq_scale) = model.rope_params();
    let rope = KvRope {
        freq_base,
        freq_scale,
        n_head_kv: model.n_head_kv(),
        hd: model.n_embd_head(),
        style: model.rope_style(),
    };
    for nt in [1usize, 2, full.len()] {
        let ids: Vec<u32> = full[..nt].to_vec();
        let (l0, k0, v0) = run(ids.clone(), 0);
        let (l0b, _k0b, v0b) = run(ids.clone(), 0);
        let (l8, k8, v8) = run(ids, 8);
        let row = v0.len() / n_ctx;
        // `rope_shift_kv(d)` means `new_pos = old_pos - d`, so aligning cell 8
        // back onto cell 0 is d = +8.
        // C6: compare the same relative rows directly — no rotation to undo.
        let k8_aligned = k8[8 * row..(nt + 8) * row].to_vec();
        let v_delta = d(&v0[..nt * row], &v8[8 * row..(nt + 8) * row]);
        let k_aligned = d(&k0[..nt * row], &k8_aligned);
        eprintln!(
            "[offset] nt={nt}: determinism {} | cell0-vs-8 logits {} | V {} | K aligned {}",
            d(&l0, &l0b),
            d(&l0, &l8),
            v_delta,
            k_aligned
        );
        // Determinism holds at every width (logits are compared bitwise above).
        assert_eq!(d(&l0, &l0b), 0.0, "the forward must be deterministic");
        // The KV-region probes below read the stored rows directly, which is
        // only meaningful where the store is f32: on a CUDA build the
        // process-wide KV dtype can be f16 (another model set it), and the
        // raw read would be bit patterns, not values.
        if matches!(model.device(), crate::models::Device::Cpu) {
            assert_eq!(d(&v0, &v0b), 0.0, "and bit-identical run to run");
            // C6: the same relative row at a different cell holds the same
            // bytes — no rotation is involved in a cell move any more.
            assert_eq!(v_delta, 0.0, "V is unrotated and cell-independent");
            assert_eq!(
                k_aligned, 0.0,
                "the stored K is verbatim at the sequence's new cell"
            );
        } else {
            eprintln!("[offset] (non-CPU device: skipping the raw KV-row probes)");
        }
        if nt == 1 {
            // A single key makes the softmax weight 1, so the scores cannot
            // matter: this must be bitwise, whatever the offset.
            assert_eq!(d(&l0, &l8), 0.0, "one query token is offset-invariant");
        } else {
            // Deliberately NOT asserted: the multi-query divergence is the
            // open finding (plan §14 row 9), and asserting it would turn a
            // bug report into a specification.
            assert!(d(&l0, &l8).is_finite());
        }
    }
}
/// Plan §14 row 9's decisive measurement: **layer 0 is offset-consistent
/// within the rotation's own rounding**, and the divergence therefore appears
/// downstream of it.
///
/// Reading an intermediate buffer *after* `execute` is unsafe (liveness
/// recycling), so this mirrors the scheduler's loop instead: execute the
/// model's graph **node by node in build order**, capture layer 0's two rope
/// outputs and its attention output immediately after each runs, and skip
/// nodes the fusion pass orphaned (they have no buffer and the scheduler skips
/// them too). The alignment numbers are their own validity check — the q and k
/// ropes align to the same 1.5e-5 the independent KV-level measurement
/// produced — and they say:
///
/// | Buffer | cell 0 vs cell 8, after aligning the rotation |
/// |---|---|
/// | layer 0 q_roped | 1.5e-5 (rel 1.9e-7) |
/// | layer 0 k_roped | 1.5e-5 (rel 1.2e-7) |
/// | layer 0 attention output | 7.3e-7 (rel 7.3e-7) |
///
/// So the ops are exact here, and the earlier KV bisect's layer-1 difference
/// (3.6e-3) is that rounding **amplified downstream** — which is why C3's
/// acceptance is byte-exact K/V plus a surviving greedy token with the
/// logits' tail in a named class, not bit-identical logits.
#[test]
fn layer_0_is_offset_consistent_within_the_rotation_rounding_class() {
    use crate::graph::backend::Backend as _;
    use crate::graph::backend::KvProvider;
    use crate::graph::batch::Batch;
    use crate::graph::fusion::FusionPass;
    use crate::graph::kvcache::{rope_shift_kv, KvRope};
    use crate::graph::params::{CParams, GraphParams, GraphType};
    use crate::graph::scheduler::BackendScheduler;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let n_ctx = 256;
    let (freq_base, freq_scale) = model.rope_params();
    let style = model.rope_style();
    let hd = model.n_embd_head();

    // (q rope, k rope, attention output) of layer 0.
    type Captured = (Vec<f32>, Vec<f32>, Vec<f32>);
    let run = |start: usize| -> Captured {
        let params = GraphParams {
            n_tokens: n,
            n_out: 1,
            gtype: GraphType::Prefill,
            cparams: CParams {
                n_ctx,
                flash_attn: false,
                explicit_span: true,
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
        let ropes: Vec<crate::graph::NodeId> = graph
            .nodes
            .iter()
            .filter(|nd| matches!(nd.op, crate::graph::ops::Op::RoPE { .. }))
            .take(2)
            .map(|nd| nd.id)
            .collect();
        let (q_node, k_node) =
            if graph.nodes[ropes[0]].out_shape[0] > graph.nodes[ropes[1]].out_shape[0] {
                (ropes[0], ropes[1])
            } else {
                (ropes[1], ropes[0])
            };
        let attn = graph
            .nodes
            .iter()
            .find(|nd| matches!(nd.op, crate::graph::ops::Op::Attn { .. }))
            .expect("attention node")
            .id;

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
            FusionPass::new().run(&mut graph, &backends, &|g, id| match g.node(id).backend {
                Some(crate::graph::Backend::CPU) => Some(0),
                _ => None,
            });
        }
        alloc.alloc_graph(&graph).expect("alloc");

        let batch = Batch::new(ids.clone(), (0..n).collect(), vec![7u32; n]);
        alloc.fill_input_i32(&graph, "token_ids", &ids).unwrap();
        let pos: Vec<u32> = (0..n).map(|p| p as u32).collect();
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

        // Build order is a valid topological order (source before consumer),
        // so executing node by node in id order is what the scheduler does.
        let order: Vec<crate::graph::NodeId> = graph.nodes.iter().map(|nd| nd.id).collect();
        let mut captured: Captured = (Vec::new(), Vec::new(), Vec::new());
        for id in order {
            let node = graph.node(id);
            if matches!(node.op, crate::graph::ops::Op::Input) {
                continue;
            }
            let ins: Vec<crate::graph::BufRef> = node
                .src
                .iter()
                .map(|&s| alloc.node_buffer(s).expect("input buffer"))
                .collect();
            // A node the fusion pass orphaned (e.g. a silu folded into
            // SwiGLU) has no buffer and is skipped, not executed — the same
            // rule the scheduler applies.
            let Some(out) = alloc.node_buffer(id) else {
                continue;
            };
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
                .unwrap_or_else(|e| panic!("node {} ({}): {e}", node.id, node.name));
            let slot = if id == q_node {
                Some(0)
            } else if id == k_node {
                Some(1)
            } else if id == attn {
                Some(2)
            } else {
                None
            };
            if let Some(slot) = slot {
                let host = alloc.cpu().read_host(out.id).expect("host read");
                let window = host[out.offset..out.offset + out.len].to_vec();
                match slot {
                    0 => captured.0 = window,
                    1 => captured.1 = window,
                    _ => captured.2 = window,
                }
            }
        }
        captured
    };
    let d = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    let scale = |v: &[f32]| v.iter().map(|x| x.abs()).fold(1.0f32, f32::max);

    let (q0, k0, a0) = run(0);
    let (q8, k8, a8) = run(8);
    // C6: a token's RoPE angle is its index within its sequence, so the same
    // relative token roped at cell 0 and at cell 8 is **bitwise** the same
    // — there is no rotation left to align.
    let (q8a, k8a) = (q8.clone(), k8.clone());
    eprintln!(
        "[per-node] q {} (rel {}) | k {} (rel {}) | attn {} (rel {})",
        d(&q0, &q8a),
        d(&q0, &q8a) / scale(&q0),
        d(&k0, &k8a),
        d(&k0, &k8a) / scale(&k0),
        d(&a0, &a8),
        d(&a0, &a8) / scale(&a0),
    );
    // The rotation's own rounding class, measured: the ropes are the only
    // inputs the offset touches, and layer 0's output follows them.
    assert!(
        d(&q0, &q8a) / scale(&q0) < 1e-5,
        "the query rope must be offset-consistent"
    );
    assert!(
        d(&k0, &k8a) / scale(&k0) < 1e-5,
        "the key rope must be offset-consistent"
    );
    assert!(
        d(&a0, &a8) / scale(&a0) < 1e-5,
        "layer 0's attention output must be offset-consistent"
    );
}
/// E2 / A7: the *number of sequences* is data, not topology.
///
/// `explicit_span` (derived from the KV reservations) already fixes every
/// topology decision a batch can make, so two batches with the same token
/// count, output count and span requirement describe **the same graph**
/// whether the tokens belong to one sequence or two. While `GraphParams`
/// also carried the sequence count, that pair rebuilt - a rebuild with no
/// topological cause, which is why the field was deleted in E2 instead of
/// kept "reserved" (A7 closure).
///
/// The test pins both halves: the graph uid is unchanged across the pair (no
/// rebuild) and the reused graph still computes what a fresh
/// single-sequence forward computes, bitwise.
#[test]
fn sequence_count_is_data_not_topology() {
    use crate::graph::batch::Batch;
    use crate::graph::cache::GraphCache;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the sequence-count test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let nv = model.n_vocab();
    let a = tok.encode("The capital of France is");
    let b = tok.encode("The capital of Japan is");
    let (la, lb) = (a.len(), b.len());
    let (s7, s9) = (7u32, 9u32);

    // The same reservation order in every cache, so the layout - and with it
    // every position below - is deterministic. Both starts are non-zero, so
    // every forward here needs the explicit span.
    let layout = |cache: &mut GraphCache| -> (usize, usize) {
        cache.alloc().kv_set_capacity(n_ctx);
        let r7 = cache.alloc().kv_reserve_seq(s7, la + 2).expect("reserve 7");
        let r9 = cache.alloc().kv_reserve_seq(s9, lb + 4).expect("reserve 9");
        (r7.start, r9.start)
    };
    let uid = |c: &mut GraphCache| c.current().map(|(g, _)| g.uid).unwrap();

    // ---- batched cache: one graph is asked for both shapes ----
    let mut c = GraphCache::new();
    let (p7, p9) = layout(&mut c);
    let pos7: Vec<usize> = (0..la).collect();
    let pos9: Vec<usize> = (0..lb).collect();

    let pre7 = model.forward_batch(
        &Batch::new(a.clone(), pos7.clone(), vec![s7; la]),
        1,
        n_ctx,
        &mut c,
    );
    let t7 = argmax(&pre7);
    let pre9 = model.forward_batch(
        &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
        1,
        n_ctx,
        &mut c,
    );
    let t9 = argmax(&pre9);

    // Two sequences, one token each: nt = 2, n_out = 2 (a row per sequence).
    let two = model.forward_batch(
        &Batch::new(vec![t7, t9], vec![la, lb], vec![s7, s9]),
        1,
        n_ctx,
        &mut c,
    );
    assert_eq!(two.len(), 2 * nv, "one logits row per sequence");
    let uid_two = uid(&mut c);

    // One sequence, two tokens: the same nt, the same n_out (a caller that
    // wants both rows' distributions), the same span requirement. Only the
    // sequence count differs, so this must reuse the graph above.
    let tail = [a[0], a[1]];
    let tail_pos = [lb + 1, lb + 2];
    let one = model.forward_batch(
        &Batch::new(tail.to_vec(), tail_pos.to_vec(), vec![s9; 2]),
        2,
        n_ctx,
        &mut c,
    );
    assert_eq!(one.len(), 2 * nv, "nt = 2, n_out = 2");
    assert_eq!(
        uid_two,
        uid(&mut c),
        "a change in the number of sequences rebuilt an otherwise identical graph"
    );

    // ---- reference: the same layout, sequence 9 alone ----
    let mut ref9 = GraphCache::new();
    let (_, q9) = layout(&mut ref9);
    assert_eq!(q9, p9, "the reference must place sequence 9 identically");
    let r_pre9 = model.forward_batch(
        &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
        1,
        n_ctx,
        &mut ref9,
    );
    assert_eq!(
        argmax(&r_pre9),
        t9,
        "the reference must agree on sequence 9's first token"
    );
    let _ = model.forward_batch(
        &Batch::new(vec![t9], vec![lb], vec![s9]),
        1,
        n_ctx,
        &mut ref9,
    );
    let r_one = model.forward_batch(
        &Batch::new(tail.to_vec(), tail_pos.to_vec(), vec![s9; 2]),
        2,
        n_ctx,
        &mut ref9,
    );

    // Sequence 7's work (and reusing the graph) must not have touched
    // sequence 9's rows. The batched side's keys were written by a two-token
    // forward and one of the pair's, the reference's by single-sequence
    // forwards — cross-shape, so CPU stays bitwise and a device uses the
    // named class.
    assert_across_shapes(
        "the reused graph against a fresh single-sequence forward",
        &one,
        &r_one,
    );
}
/// E1b/E2: a query's attention window follows from **which sequence it
/// belongs to and where that sequence is reserved** — never from its row
/// index inside the batch. Swapping two sequences in an otherwise identical
/// batch must therefore reproduce each sequence's logits *bitwise*, on CPU
/// and on CUDA alike.
///
/// This is the device-capable form of E1's "two sequences do not
/// cross-attend" gate. The earlier batched tests compare forwards of
/// *different shapes* (a batched prefill against per-sequence prefills),
/// which is bitwise only on CPU: CUDA's prefill kernels tile by `nt` and
/// store quantized activations, so a different shape changes the K/V values
/// by ~1e-3 relative — measured on dgxspark as ~0.3 absolute on logits, and
/// already true on master for B2's and C2's cross-shape tests. A swap keeps
/// the shape, the layout, the KV history and the graph identical, leaving
/// the window assignment as the only thing that can move the numbers, so
/// bitwise equality is the right expectation on every backend.
#[test]
fn batch_order_does_not_change_a_sequences_logits() {
    use crate::graph::batch::Batch;
    use crate::graph::cache::GraphCache;
    use crate::models::ModelDef;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the batch-order test");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let model = crate::models::load_model(&gguf).expect("load model");
    #[cfg(feature = "cuda")]
    let _model_load_guard = crate::cuda::CudaState::model_load_guard();
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");

    let n_ctx = 256;
    let nv = model.n_vocab();
    let a = tok.encode("The capital of France is");
    let b = tok.encode("The capital of Japan is");
    let (la, lb) = (a.len(), b.len());
    let (s7, s9) = (7u32, 9u32);

    // Same reservations, same order, in both caches: first-fit makes the two
    // layouts identical, so the only difference below is the batch order.
    let layout = |cache: &mut GraphCache| -> (usize, usize) {
        cache.alloc().kv_set_capacity(n_ctx);
        let r7 = cache.alloc().kv_reserve_seq(s7, la + 2).expect("reserve 7");
        let r9 = cache.alloc().kv_reserve_seq(s9, lb + 2).expect("reserve 9");
        (r7.start, r9.start)
    };

    // The whole run, with sequence 7 either first or second in the decode
    // batch. Returns `(logits of 7, logits of 9)`.
    let run = |cache: &mut GraphCache, seq7_first: bool| -> (Vec<f32>, Vec<f32>) {
        let (p7, p9) = layout(cache);
        let pos7: Vec<usize> = (0..la).collect();
        let pos9: Vec<usize> = (0..lb).collect();
        let pre7 = model.forward_batch(
            &Batch::new(a.clone(), pos7.clone(), vec![s7; la]),
            1,
            n_ctx,
            cache,
        );
        let t7 = argmax(&pre7);
        let pre9 = model.forward_batch(
            &Batch::new(b.clone(), pos9.clone(), vec![s9; lb]),
            1,
            n_ctx,
            cache,
        );
        let t9 = argmax(&pre9);

        // One token per sequence, nt = 2 either way.
        let (tokens, positions, seq_ids) = if seq7_first {
            (vec![t7, t9], vec![la, lb], vec![s7, s9])
        } else {
            (vec![t9, t7], vec![lb, la], vec![s9, s7])
        };
        let rows = model.forward_batch(&Batch::new(tokens, positions, seq_ids), 1, n_ctx, cache);
        assert_eq!(rows.len(), 2 * nv, "one logits row per sequence");
        if seq7_first {
            (rows[..nv].to_vec(), rows[nv..].to_vec())
        } else {
            (rows[nv..].to_vec(), rows[..nv].to_vec())
        }
    };

    let mut c1 = GraphCache::new();
    let (r7_first, r9_second) = run(&mut c1, true);
    let mut c2 = GraphCache::new();
    let (r7_second, r9_first) = run(&mut c2, false);

    assert_eq!(
        max_delta(&r7_first, &r7_second),
        0.0,
        "sequence 7's logits depend on its row index in the batch"
    );
    assert_eq!(
        max_delta(&r9_second, &r9_first),
        0.0,
        "sequence 9's logits depend on its row index in the batch"
    );
    // The fixture must discriminate: swapping must not have collapsed the
    // two sequences onto the same distribution.
    assert!(
        max_delta(&r7_first, &r9_second) > 0.0,
        "both sequences produced identical logits - the fixture is not discriminating"
    );
}
/// Phase 0 (OPENAI-CHAT-API-PLAN.md): `forward_cached` with two independent
/// `GraphCache` instances must isolate KV — interleaving prefill/decode
/// across caches must not change either cache's logits, and a cache scoped
/// to a smaller `n_ctx` must still work (CPU-only, real model when cached).
#[test]
fn forward_cached_isolates_kv_between_caches() {
    use crate::graph::cache::GraphCache;

    let Some(path) = cached_model_path() else {
        eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping forward_cached test");
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
    let q2: &Qwen2Model = model
        .as_any()
        .downcast_ref::<Qwen2Model>()
        .expect("qwen2 model");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let ids = tok.encode("The capital of France is Paris and");
    let nt = ids.len();
    assert!(nt > 2, "prompt too short");
    let positions: Vec<usize> = (0..nt).collect();
    let next_pos = nt; // first decode position
    let n_ctx = 32768usize;

    // Same input on two fresh caches => identical logits (deterministic).
    let mut cache_a = GraphCache::new();
    let mut cache_b = GraphCache::new();
    let pa = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, n_ctx, &mut cache_a);
    let pb = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, n_ctx, &mut cache_b);
    assert_eq!(pa.len(), pb.len());
    let mut d = 0.0f32;
    for i in 0..pa.len() {
        d = d.max((pa[i] - pb[i]).abs());
    }
    assert_eq!(
        d, 0.0,
        "identical inputs on fresh caches must give identical logits"
    );

    // Interleave: A prefill -> B prefill -> A decode -> B decode.
    // Each cache's KV must stay isolated from the other's.
    let da = Qwen2Graph::forward_cached(q2, &[argmax(&pa)], &[next_pos], 1, n_ctx, &mut cache_a);
    let db = Qwen2Graph::forward_cached(q2, &[argmax(&pb)], &[next_pos], 1, n_ctx, &mut cache_b);
    let mut d2 = 0.0f32;
    for i in 0..da.len().min(db.len()) {
        d2 = d2.max((da[i] - db[i]).abs());
    }
    assert_eq!(d2, 0.0, "interleaved caches must not cross-contaminate KV");

    // n_ctx bounds: positions must stay below n_ctx (asserted), and a
    // smaller n_ctx must produce the same prefill logits (KV capacity does
    // not change the math while positions fit).
    let small_ctx = nt + 4;
    let mut cache_s = GraphCache::new();
    let ps = Qwen2Graph::forward_cached(q2, &ids, &positions, 1, small_ctx, &mut cache_s);
    let mut d3 = 0.0f32;
    for i in 0..pa.len() {
        d3 = d3.max((pa[i] - ps[i]).abs());
    }
    assert_eq!(d3, 0.0, "smaller n_ctx must not change prefill logits");
    // And an out-of-range position must panic (guarded), not corrupt memory.
    let oob = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        Qwen2Graph::forward_cached(q2, &[argmax(&pa)], &[small_ctx], 1, small_ctx, &mut cache_s);
    }));
    assert!(oob.is_err(), "position >= n_ctx must be rejected");
}
