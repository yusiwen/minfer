//! `#[cfg(test)] mod tests` for `src/graph/alloc.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::batch::Batch;
use crate::graph::builder::GraphBuilder;
use crate::graph::DType;

/// F4: a fresh allocator inherits the run's backend fence, so the two places
/// the fence is enforced (this allocator's assignment and the graph
/// builders' `Device`) read the same answer. The default — nothing installed
/// — is every backend, i.e. the pre-F4 behaviour.
#[test]
fn a_fresh_allocator_inherits_the_runs_backend_filter() {
    let alloc = GraphAllocator::new();
    assert_eq!(
        alloc.backend_filter(),
        crate::graph::registry::active_filter()
    );
    // The process-wide default is unfiltered in the test binary (nothing
    // installs a fence here), so the assignment is unchanged for every
    // existing gate.
    assert!(crate::graph::registry::active_filter().is_unfiltered());
    assert_eq!(alloc.supports(&Op::Input, DType::F32), Some(Backend::CPU));
}

/// F4: the fence reaches the **assignment pass**, not just the name parser.
///
/// Needs a usable device, so it is `#[ignore]`d like the other device gates
/// (run it serially: the CUDA device state is a process-wide singleton).
/// Without a fence the device takes an op it supports; with the device
/// fenced off the *same* op must land on the CPU — and the device stays
/// enabled, because the fence is a policy, not a teardown.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn cuda_the_backend_fence_moves_assignment_off_a_usable_device() {
    crate::cuda::CudaState::init();
    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("no CUDA device; this gate needs one");
        return;
    }
    assert_eq!(
        alloc.supports(&Op::Silu, DType::F32),
        Some(Backend::CUDA),
        "an unfenced allocator must offer the device"
    );
    alloc.set_backend_filter(crate::graph::registry::BackendFilter::from_names(["cpu"]).unwrap());
    assert_eq!(
        alloc.supports(&Op::Silu, DType::F32),
        Some(Backend::CPU),
        "a fenced allocator must not offer the device"
    );
    assert!(
        alloc.cuda().is_some(),
        "the fence must not tear the device pool down"
    );
}

/// C8a S1: the prefix copy refuses what it cannot do instead of copying a row that
/// does not exist or writing past the destination's reservation.
///
/// These are the checks that run before any backend work. The data path, and the
/// written/capacity bounds (which need real KV regions), are covered by the S2 gate:
/// the same prompt served via a copy and via a re-prefill must produce byte-identical
/// continuations.
#[test]
fn copying_a_prefix_refuses_what_it_cannot_copy() {
    let mut a = GraphAllocator::new();
    a.kv_set_capacity(16);
    // Neither sequence holds a run yet.
    let err = a.kv_copy_prefix(0, 1, 4).unwrap_err();
    assert!(err.contains("holds no run"), "got: {err}");
    a.kv_reserve_seq(0, 8).unwrap();
    // The destination has no run.
    let err = a.kv_copy_prefix(0, 1, 4).unwrap_err();
    assert!(err.contains("holds no run"), "got: {err}");
    a.kv_reserve_seq(1, 4).unwrap();
    // Zero rows is a no-op, whatever the two runs are.
    a.kv_copy_prefix(0, 0, 0).expect("zero rows");
    // A source with nothing written is refused before any copy is attempted.
    let err = a.kv_copy_prefix(0, 1, 2).unwrap_err();
    assert!(err.contains("has written 0 rows"), "got: {err}");
}

/// C6/C7: `cells` is an absolute arena row, so it is bounded by the arena — not
/// by the per-sequence position rule it used to be checked with. The two bounds
/// coincide today, which is exactly why the difference is written down: sharing a
/// cell range across sequences is what stops them coinciding.
#[test]
fn the_cells_input_is_bounded_by_the_arena() {
    let mut b = GraphBuilder::new();
    let cells = b.input("cells", [1, 1, 1, 1], crate::graph::DType::I32);
    b.output(cells);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(8);
    alloc.alloc_graph(&g).expect("alloc the graph");
    alloc
        .fill_input_i32(&g, "cells", &[7])
        .expect("the last cell of an 8-cell arena is valid");
    let err = alloc.fill_input_i32(&g, "cells", &[8]).unwrap_err();
    assert!(err.contains("cell 8 is past the 8-cell arena"), "{err}");
    // Without an arena there is nothing to bound against, and the resolver reports
    // that case itself when it runs.
    let mut fresh = GraphAllocator::new();
    fresh.alloc_graph(&g).expect("alloc the graph");
    fresh
        .fill_input_i32(&g, "cells", &[3])
        .expect("no arena, no bound to check");
}

/// D1: a graph whose output is a *view* of an input, so the allocator must
/// alias rather than copy. `offset`/`shape` are parameters so the refusal
/// cases can be built too.
fn view_graph(offset: usize, shape: [usize; 4], parent_shape: [usize; 4]) -> ComputeGraph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", parent_shape, crate::graph::DType::F32);
    let v = b.node(
        "view",
        Op::View { offset, shape },
        &[x],
        shape,
        crate::graph::DType::F32,
        NodeMeta::None,
    );
    b.output(v);
    b.build()
}

/// D1 acceptance: the view is **zero-copy** — it maps onto its parent's
/// buffer, and the parent is not recycled while the view lives.
#[test]
fn a_view_aliases_its_parent_buffer_and_keeps_it_alive() {
    let g = view_graph(0, [4, 1, 1, 1], [4, 1, 1, 1]);
    assert!(
        g.node(1).view.is_some(),
        "the builder must mark View as an alias"
    );
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).expect("alloc");
    let parent = alloc.node_buffer(0).expect("parent buffer");
    let view = alloc.node_buffer(1).expect("view buffer");
    assert_eq!(
        (view.backend, view.id),
        (parent.backend, parent.id),
        "the view must reuse its parent's buffer id (zero-copy)"
    );
    // Liveness: the parent's deadline covers the view (the view is an
    // output here, so both run to the end of the graph).
    let parent_deadline = alloc.buf_alive.get(&(parent.backend, parent.id)).copied();
    assert!(
        parent_deadline.is_some(),
        "the aliased parent must be registered as alive"
    );
}

/// D1 increment 2: a partial window at a non-zero offset maps onto the
/// parent's buffer with exactly that window (the op matrix's `View offset`
/// case proves the bytes; this pins the bookkeeping).
#[test]
fn an_offset_view_names_a_window_of_its_parent() {
    let g = view_graph(2, [4, 1, 1, 1], [8, 1, 1, 1]);
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).expect("alloc");
    let parent = alloc.node_buffer(0).expect("parent");
    let view = alloc.node_buffer(1).expect("view");
    assert_eq!(
        (view.id, view.offset, view.len),
        (parent.id, 2, 4),
        "the view must be the parent's buffer, windowed to [2, 6)"
    );
}

/// D1: what cannot be expressed is refused **loudly** — never silently
/// copied or mis-read (standing rule 2). Increment 2 accepts offset and
/// partial windows, so what remains here is a window that does not fit.
#[test]
fn views_that_do_not_fit_their_parent_are_refused() {
    let g = view_graph(4, [4, 1, 1, 1], [6, 1, 1, 1]);
    let err = GraphAllocator::new()
        .alloc_graph(&g)
        .expect_err("a window past the parent must be refused");
    assert!(err.contains("does not fit"), "{err}");
}

/// D2: the FFN composition's `SwiGLU` writes into its **gate window**, so the
/// node's buffer *is* the concat buffer at offset 0 — the same bytes the
/// hand-written `Op::FusedFFN` leaves its result in, which is what makes the
/// two paths comparable. No GPU needed: this is allocator bookkeeping.
#[test]
fn ffn_composition_swiglu_aliases_the_gate_window() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
    let out = b.fused_ffn_composition(
        x,
        "blk.0.ffn_gu",
        crate::tensor::TensorType::F32,
        4, // in_dim
        3, // nf
    );
    b.output(out);
    let g = b.build();
    let names: Vec<&str> = g.nodes.iter().map(|n| n.name.as_str()).collect();
    assert!(names.contains(&"matmul_named"), "{names:?}");
    assert!(names.contains(&"ffn_gate_window"), "{names:?}");
    assert!(names.contains(&"ffn_up_window"), "{names:?}");
    assert!(names.contains(&"ffn_swiglu"), "{names:?}");

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).expect("alloc");
    let concat = alloc
        .node_buffer(
            g.nodes
                .iter()
                .position(|n| n.name == "matmul_named")
                .unwrap(),
        )
        .expect("concat");
    let gate = alloc
        .node_buffer(
            g.nodes
                .iter()
                .position(|n| n.name == "ffn_gate_window")
                .unwrap(),
        )
        .expect("gate");
    let up = alloc
        .node_buffer(
            g.nodes
                .iter()
                .position(|n| n.name == "ffn_up_window")
                .unwrap(),
        )
        .expect("up");
    let swiglu = alloc
        .node_buffer(g.nodes.iter().position(|n| n.name == "ffn_swiglu").unwrap())
        .expect("swiglu");
    assert_eq!(
        (gate.id, gate.offset, gate.len),
        (concat.id, 0, 3),
        "the gate is the concat's first window"
    );
    assert_eq!(
        (up.id, up.offset, up.len),
        (concat.id, 3, 3),
        "the up half is the concat's second window"
    );
    assert_eq!(
        (swiglu.id, swiglu.offset, swiglu.len),
        (concat.id, 0, 3),
        "in-place swiglu writes into the gate window, exactly where the fused node puts its result"
    );
}

/// D1: an in-place op on a view writes through to the buffer it is a window
/// of, so liveness must follow the chain (view -> parent).
#[test]
fn in_place_on_a_view_extends_the_parents_liveness() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
    let v = b.node(
        "view",
        Op::View {
            offset: 0,
            shape: [4, 1, 1, 1],
        },
        &[x],
        [4, 1, 1, 1],
        crate::graph::DType::F32,
        NodeMeta::None,
    );
    // sole consumer of the view, same backend -> the in-place rule aliases
    let s = b.silu(v);
    b.output(s);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).expect("alloc");
    let parent = alloc.node_buffer(0).expect("parent");
    assert_eq!(
        alloc.node_buffer(2).expect("silu").id,
        parent.id,
        "the in-place op must alias through the view onto the parent"
    );
    assert!(
        alloc.buf_alive.contains_key(&(parent.backend, parent.id)),
        "parent must still be alive for the in-place consumer"
    );
}

fn chain(n_ops: usize) -> ComputeGraph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
    let mut h = x;
    for i in 0..n_ops {
        h = if i % 2 == 0 { b.silu(h) } else { b.add(h, x) };
    }
    b.output(h);
    b.build()
}

#[test]
fn liveness_reuses_buffers_along_chain() {
    let g = chain(6);
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    assert!(
        alloc.n_mapped_buffers() < g.n_nodes(),
        "expected reuse, got {} buffers for {} nodes",
        alloc.n_mapped_buffers(),
        g.n_nodes()
    );
    for id in 0..g.n_nodes() {
        assert!(alloc.node_buffer(id).is_some(), "node {id} missing buffer");
    }
}

#[test]
fn parallel_chains_do_not_share() {
    let mut b = GraphBuilder::new();
    let a0 = b.input("a0", [4, 1, 1, 1], crate::graph::DType::F32);
    let b0 = b.input("b0", [4, 1, 1, 1], crate::graph::DType::F32);
    let a1 = b.silu(a0);
    let b1 = b.silu(b0);
    let a2 = b.add(a1, a0);
    let b2 = b.add(b1, b0);
    let out = b.add(a2, b2);
    b.output(out);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    let n_chain6 = {
        let g2 = chain(6);
        let mut al = GraphAllocator::new();
        al.alloc_graph(&g2).unwrap();
        al.n_mapped_buffers()
    };
    assert!(
        alloc.n_mapped_buffers() > n_chain6,
        "parallel chains should not share"
    );
}

#[test]
fn fill_and_read_input() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(alloc.get_buffer(&g, x).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    assert!(alloc.fill_input(&g, "x", &[1.0, 2.0]).is_err());
    assert!(alloc.fill_input(&g, "nope", &[]).is_err());
}

#[test]
fn kv_regions_two_per_layer() {
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, 1024);
    let load = b.kvcache_load(0, 16, 1024, 2);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    // C6: the store now also consumes a `cells` input, so node ids shifted —
    // look the nodes up by name instead of by position.
    let store = g.nodes.iter().position(|n| n.name == "kv_store.0").unwrap();
    let load = g.nodes.iter().position(|n| n.name == "kv_load.0").unwrap();
    // store and load share the K region; V is a sibling
    assert_eq!(alloc.node_buffer(store), alloc.node_buffer(load));
    let pair = alloc.kv_pair(0).unwrap();
    assert_eq!(alloc.node_buffer(store).unwrap().id, pair.0);
    assert_ne!(pair.0, pair.1);
    assert_eq!(alloc.persistent.len(), 2);
    assert_eq!(alloc.persistent[0].name, "kv.0.k");
    assert_eq!(alloc.persistent[1].name, "kv.0.v");
    // mapped buffers: positions/k/v/cells (4 liveness) + K region (shared) = 5
    assert_eq!(alloc.n_mapped_buffers(), 5);
}

/// C5 end to end on CPU: a session's region bytes **and** its run table
/// (reservation, ownership, written extent) survive a save and a load into a
/// **fresh** allocator, and the loaded arena resolves the same cells.
///
/// The refusals are the other half of the ticket: a file that does not describe
/// this arena is rejected loudly, and because `kv_load` verifies the whole file
/// before it applies anything, a rejected load leaves no arena behind.
#[test]
fn kv_session_round_trips_the_rows_and_the_run_table() {
    const N_CTX: usize = 16;
    const ROW: usize = 4;
    const SEQ: u32 = 7;
    let kk: Vec<f32> = (0..ROW * 2).map(|i| 0.25 + i as f32 * 0.5).collect();
    let vv: Vec<f32> = (0..ROW * 2).map(|i| 1.0 / (i as f32 + 1.0)).collect();

    let graph = |alloc: &mut GraphAllocator| -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [2, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, N_CTX);
        let load = b.kvcache_load(0, ROW, N_CTX, 1);
        b.output(load);
        let g = b.build();
        alloc.kv_set_capacity(N_CTX);
        alloc.alloc_graph(&g).unwrap();
        g
    };

    // ---- the session that stays in memory ----
    let mut a = GraphAllocator::new();
    let ga = graph(&mut a);
    a.kv_reserve_seq(SEQ, N_CTX).unwrap();
    a.fill_input_i32(&ga, "positions", &[1, 3]).unwrap();
    // E2, the production spelling: the reservation above means the fill owns
    // positions 1 and 3 in sequence `SEQ` without reserving anything else (the
    // deleted E1 helper recorded a prefix on `SEQ_MAIN` here instead).
    a.fill_batch_inputs(&ga, &Batch::new(vec![0, 0], vec![1, 3], vec![SEQ, SEQ]))
        .unwrap();
    a.fill_input(&ga, "k", &kk).unwrap();
    a.fill_input(&ga, "v", &vv).unwrap();
    let mut sched = crate::graph::scheduler::BackendScheduler::new();
    sched.execute(&ga, &mut a).unwrap();
    a.kv.own_range(SEQ, 0, 4);
    let want_kv = a.copy_kv_to_cpu(0).unwrap();
    let want_stats = a.kv_arena_stats();
    let want_cells = a.kv_cells_for_seq(&[SEQ, SEQ], &[1, 3]).unwrap();

    let path = std::env::temp_dir().join(format!(
        "minfer-c5-alloc-{}-roundtrip.bin",
        std::process::id()
    ));
    let report = a.kv_save(&path).unwrap();
    assert_eq!(report.layers, 1);
    assert_eq!(report.cells, N_CTX);
    assert_eq!(report.written, 4, "positions 0..4 are written");
    assert_eq!(report.bytes, std::fs::metadata(&path).unwrap().len());

    // ---- a fresh allocator, restored from the file ----
    let expect = KvSessionExpect {
        backend: Backend::CPU,
        n_ctx: N_CTX,
        n_embd: ROW,
    };
    let mut b = GraphAllocator::new();
    let gb = graph(&mut b);
    let loaded = b.kv_load(&path, &expect).unwrap();
    assert_eq!(loaded, report);
    assert_eq!(
        b.copy_kv_to_cpu(0).unwrap(),
        want_kv,
        "the region bytes must be identical"
    );
    assert_eq!(b.kv_arena_stats(), want_stats);
    assert_eq!(
        b.kv_cells_for_seq(&[SEQ, SEQ], &[1, 3]).unwrap(),
        want_cells,
        "the restored run table must resolve the same cells"
    );
    assert_eq!(b.kv_n_used(0), a.kv_n_used(0));
    let _ = gb;

    // ---- refusals ----
    for (what, expect_bad, needle) in [
        (
            "a different n_ctx",
            KvSessionExpect {
                n_ctx: N_CTX + 1,
                ..expect
            },
            "-cell arena",
        ),
        (
            "a different row width",
            KvSessionExpect {
                n_embd: ROW + 4,
                ..expect
            },
            "KV rows",
        ),
        (
            "another backend",
            KvSessionExpect {
                backend: Backend::METAL,
                ..expect
            },
            "session",
        ),
    ] {
        let mut fresh = GraphAllocator::new();
        let err = fresh.kv_load(&path, &expect_bad).unwrap_err();
        assert!(err.contains(needle), "{what}: {err}");
        assert!(
            fresh.kv_n_used(0).is_none(),
            "{what}: a refused load must not create an arena"
        );
    }
    // A different KV element type for the same shape.
    let mut fresh = GraphAllocator::new();
    fresh
        .cpu_mut()
        .set_kv_format(super::super::kvformat::KvFormat::Q8_0);
    let err = fresh.kv_load(&path, &expect).unwrap_err();
    assert!(err.contains("element type"), "{err}");
    assert!(fresh.kv_n_used(0).is_none());

    // A truncated file: rejected, and (because `kv_load` verifies first) the
    // allocator is untouched.
    let full = std::fs::read(&path).unwrap();
    std::fs::write(&path, &full[..full.len() / 2]).unwrap();
    let mut fresh = GraphAllocator::new();
    let err = fresh.kv_load(&path, &expect).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
    assert!(fresh.kv_n_used(0).is_none());
    std::fs::remove_file(&path).ok();
}

/// #130: the same `kv_save` → `kv_load` round trip under an **f16** element
/// type — the policy a CUDA box's auto rule picks for the 7B class
/// (`n_layers * n_kv_embd >= 8192`). The store here is the CPU's (which copies
/// words), so this gates the *container* half with no device: the format the
/// allocator reports, the header's flags, the reader's decode and the
/// `header.format != live` check. Before `FLAG_F16` the load refused the file
/// the save had just produced. The device half is the Qwen3-0.6B real-model gate.
#[test]
fn an_f16_session_round_trips_through_save_and_load() {
    use crate::graph::kvformat::KvFormat;
    const N_CTX: usize = 8;
    const ROW: usize = 4;
    const SEQ: u32 = 3;
    let kk: Vec<f32> = (0..ROW * 2).map(|i| 0.5 + i as f32).collect();
    let vv: Vec<f32> = (0..ROW * 2).map(|i| 1.0 / (i as f32 + 1.0)).collect();

    let build = |alloc: &mut GraphAllocator| -> ComputeGraph {
        alloc.cpu_mut().set_kv_format(KvFormat::F16);
        let mut b = GraphBuilder::new();
        b.set_kv_format(KvFormat::F16);
        let pos = b.input("positions", [2, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [ROW, 2, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, N_CTX);
        let load = b.kvcache_load(0, ROW, N_CTX, 1);
        b.output(load);
        let g = b.build();
        alloc.kv_set_capacity(N_CTX);
        alloc.alloc_graph(&g).unwrap();
        g
    };

    let mut a = GraphAllocator::new();
    let ga = build(&mut a);
    a.kv_reserve_seq(SEQ, N_CTX).unwrap();
    a.fill_input_i32(&ga, "positions", &[0, 1]).unwrap();
    // E2, the production spelling (see the f32 round trip above).
    a.fill_batch_inputs(&ga, &Batch::new(vec![0, 0], vec![0, 1], vec![SEQ, SEQ]))
        .unwrap();
    a.fill_input(&ga, "k", &kk).unwrap();
    a.fill_input(&ga, "v", &vv).unwrap();
    let mut sched = crate::graph::scheduler::BackendScheduler::new();
    sched.execute(&ga, &mut a).unwrap();
    a.kv.own_range(SEQ, 0, 2);
    let want_kv = a.copy_kv_to_cpu(0).unwrap();

    let path = std::env::temp_dir().join(format!("minfer-c5-alloc-{}-f16.bin", std::process::id()));
    let report = a.kv_save(&path).unwrap();
    // The bytes on disk name f16 — the flag the reader must decode.
    let bytes = std::fs::read(&path).unwrap();
    let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    assert_eq!(flags, 1 << 1, "the header must carry FLAG_F16");

    // A fresh allocator under the same policy resumes it, rows bit-identical.
    let expect = KvSessionExpect {
        backend: Backend::CPU,
        n_ctx: N_CTX,
        n_embd: ROW,
    };
    let mut b = GraphAllocator::new();
    let gb = build(&mut b);
    let loaded = b.kv_load(&path, &expect).unwrap();
    assert_eq!(loaded, report);
    assert_eq!(b.copy_kv_to_cpu(0).unwrap(), want_kv);
    let _ = gb;

    // An f32 reader refuses it by element type — not silently, and without
    // creating an arena.
    let mut c = GraphAllocator::new();
    c.cpu_mut().set_kv_format(KvFormat::F32);
    let err = c.kv_load(&path, &expect).unwrap_err();
    assert!(err.contains("element type"), "{err}");
    assert!(err.contains("f16"), "{err}");
    assert!(c.kv_n_used(0).is_none());
    std::fs::remove_file(&path).ok();
}

/// C3 end to end on CPU: a fragmented arena refuses an 8-cell reservation,
/// the compaction moves the real KV bytes *and* renumbers the runs, and the
/// reservation then fits — with the moved rows byte-identical at their new
/// cells.
#[test]
fn kv_defrag_moves_the_bytes_and_opens_the_run() {
    const N_CTX: usize = 16;
    const ROW: usize = 4; // elements per cell (n_kv_embd)
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, N_CTX);
    let load = b.kvcache_load(0, ROW, N_CTX, 1);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g).unwrap();
    for seq in 1u32..=3 {
        assert_eq!(
            alloc.kv_reserve_seq(seq, 4).unwrap().start,
            (seq as usize - 1) * 4
        );
        alloc
            .kv
            .own_range(seq, (seq as usize - 1) * 4, seq as usize * 4);
    }
    let rope = crate::graph::kvcache::KvRope {
        freq_base: 10_000.0,
        freq_scale: 1.0,
        n_head_kv: 1,
        hd: ROW,
        style: crate::vec_ops::RopeStyle::NonInterleaved,
    };
    let (kid, vid) = alloc.kv_pair(0).unwrap();
    let pattern = |base: f32| -> Vec<f32> {
        (0..N_CTX * ROW)
            .map(|i| base + (i / ROW) as f32 + (i % ROW) as f32 / 10.0)
            .collect()
    };
    alloc
        .write_pool(crate::graph::Backend::CPU, kid, &pattern(0.0))
        .unwrap();
    alloc
        .write_pool(crate::graph::Backend::CPU, vid, &pattern(100.0))
        .unwrap();
    // Release the middle run: free runs [4,8) and [12,16), 8 cells, none 8 long.
    alloc.kv_release_seq(2);
    let before = alloc.kv_arena_stats();
    assert_eq!((before.free_cells, before.free_runs), (8, 2));
    assert!(alloc.kv_reserve_seq(4, 8).is_err());

    let report = alloc.kv_defrag(Some(8)).unwrap();
    assert_eq!(report.moves.len(), 1, "{report:?}");
    assert_eq!(report.moves[0].seq, 3);
    assert_eq!((report.moves[0].from, report.moves[0].to), (8, 4));
    assert_eq!(report.rows_moved, 4);
    assert_eq!((report.before.free_runs, report.after.free_runs), (2, 1));
    assert_eq!(report.after.largest_free_run, 8);
    assert_eq!((report.after.defrags, report.after.cells_moved), (1, 4));
    // C6: both K and V move **verbatim** — a cell move no longer changes any
    // rotation, because a token's angle is its index within its sequence.
    let (k_now, v_now) = alloc.copy_kv_to_cpu(0).unwrap();
    for e in 0..ROW {
        assert_eq!(v_now[4 * ROW + e], 108.0 + e as f32 / 10.0, "V element {e}");
    }
    let k_expect: Vec<f32> = (0..ROW).map(|e| 8.0 + e as f32 / 10.0).collect();
    for e in 0..ROW {
        assert_eq!(k_now[4 * ROW + e], k_expect[e], "K element {e}");
    }
    // The reservation that first-fit refused now fits, in the opened tail.
    assert_eq!(alloc.kv_reserve_seq(4, 8).unwrap().start, 8);
    // The same helper, on a fresh fragmentation: free the lowest run and the
    // 8-cell one, and ask for 12 contiguous cells. First-fit fails (the free
    // space is two runs); compacting the single survivor down opens the tail,
    // and the helper reports the move it relied on.
    alloc.kv_release_seq(1);
    alloc.kv_release_seq(4);
    let (slot, moves) = alloc.kv_reserve_seq_with_defrag(5, 12).unwrap();
    assert_eq!(
        slot.start, 4,
        "the retry packs the survivor down, then takes the tail"
    );
    assert_eq!(moves.len(), 1, "{moves:?}");
    assert_eq!((moves[0].seq, moves[0].from, moves[0].to), (3, 4, 0));
    // The helper also answers "it still does not fit" with the original
    // first-fit error, after compaction moved nothing.
    let err = alloc.kv_reserve_seq_with_defrag(6, 4).unwrap_err();
    assert!(err.contains("no free run of 4 cells"), "{err}");
}

#[test]
fn the_defrag_gate_is_off_only_when_the_flag_is_present() {
    assert!(super::kv_defrag_enabled_from(None));
    assert!(!super::kv_defrag_enabled_from(Some(std::ffi::OsStr::new(
        "1"
    ))));
}

/// C8b S3 end to end on CPU: a sequence that reads a prefix in place cannot
/// store into it, the copy-on-write moves **its own** rows up inside its run,
/// and the donor's cells come out of it byte-identical.
#[test]
fn a_copy_on_write_moves_the_rows_and_never_writes_through() {
    const N_CTX: usize = 16;
    const ROW: usize = 4; // elements per cell (n_kv_embd)
    let mut b = GraphBuilder::new();
    let _pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, N_CTX);
    let load = b.kvcache_load(0, ROW, N_CTX, 1);
    b.output(load);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g).unwrap();
    let (kid, vid) = alloc.kv_pair(0).unwrap();
    // Every cell holds the row index and the element, so a moved row is
    // identifiable wherever it lands.
    let pattern = |base: f32| -> Vec<f32> {
        (0..N_CTX * ROW)
            .map(|i| base + (i / ROW) as f32 * 100.0 + (i % ROW) as f32 / 10.0)
            .collect()
    };
    alloc
        .write_pool(crate::graph::Backend::CPU, kid, &pattern(0.0))
        .unwrap();
    alloc
        .write_pool(crate::graph::Backend::CPU, vid, &pattern(1000.0))
        .unwrap();
    // Sequence 1 computes four rows at [0, 4); sequence 2 reserves [4, 10),
    // reads those four in place, and writes two rows of its own at [4, 6).
    assert_eq!(alloc.kv_reserve_seq(1, 4).unwrap().start, 0);
    alloc.kv.own_range(1, 0, 4);
    assert_eq!(alloc.kv_reserve_seq(2, 6).unwrap().start, 4);
    assert_eq!(alloc.kv_share_prefix(1, 2, 4).unwrap(), 4);
    alloc.kv.own_range(2, 4, 6);

    // The store resolver refuses a position inside the share: that refusal is
    // what makes a write-through impossible rather than merely unlikely.
    let err = alloc.kv_cells_for_seq(&[2], &[1]).unwrap_err();
    assert!(err.contains("shared prefix"), "got: {err}");
    assert!(err.contains("C8b S3"), "got: {err}");

    let shift = alloc
        .kv_private_row_for(2, 1)
        .unwrap()
        .expect("position 1 must copy-on-write");
    assert_eq!((shift.base, shift.from, shift.to, shift.rows), (1, 4, 7, 2));
    assert_eq!(alloc.kv_arena_stats().cows, 1);
    // From position 1 on everything is private, and the rows that were at
    // [4, 6) kept their positions at [7, 9) — while position 0 still reads the
    // donor's cell 0.
    assert_eq!(alloc.kv.cell_of(2, 0), Some(0), "still shared");
    assert_eq!(
        alloc
            .kv_cells_for_seq(&[2, 2, 2, 2, 2], &[1, 2, 3, 4, 5])
            .unwrap(),
        vec![4, 5, 6, 7, 8]
    );
    // A store at 0 would still land in the donor's cells, so it is still
    // refused — the share is smaller, not gone.
    assert!(alloc.kv_cells_for_seq(&[2], &[0]).is_err());
    // The data moved verbatim (K and V), and the donor's four rows are what
    // they were — nothing wrote through them.
    let (k_now, v_now) = alloc.copy_kv_to_cpu(0).unwrap();
    for e in 0..ROW {
        let row = |cell: usize| cell * ROW + e;
        assert_eq!(k_now[row(7)], 400.0 + e as f32 / 10.0, "K at [7]");
        assert_eq!(k_now[row(8)], 500.0 + e as f32 / 10.0, "K at [8]");
        assert_eq!(v_now[row(7)], 1400.0 + e as f32 / 10.0, "V at [7]");
        assert_eq!(v_now[row(8)], 1500.0 + e as f32 / 10.0, "V at [8]");
        for cell in 0..4 {
            assert_eq!(
                k_now[row(cell)],
                cell as f32 * 100.0 + e as f32 / 10.0,
                "donor K cell {cell} changed"
            );
            assert_eq!(
                v_now[row(cell)],
                1000.0 + cell as f32 * 100.0 + e as f32 / 10.0,
                "donor V cell {cell} changed"
            );
        }
    }
    // Idempotent for a position that is already private.
    assert_eq!(alloc.kv_private_row_for(2, 3).unwrap(), None);
    assert_eq!(alloc.kv_arena_stats().cows, 1);
    // And a sequence with no run at all is a no-op, not an error — that is what
    // keeps the classic single-sequence path (which never reserves) untouched.
    assert_eq!(alloc.kv_private_row_for(9, 0).unwrap(), None);
    // The production entry point gets the same rule: `fill_batch_inputs` copies
    // for a batch that still names a shared position, and then resolves the
    // private cells. Position 0 is the last shared one, so this is the second
    // (and final) copy-on-write. A `Batch::single` would name `SEQ_MAIN` (0) and
    // try to reserve the whole arena for it — the wrong sequence, and the arena is
    // already full — so this spells the batch out for sequence 2.
    alloc
        .fill_batch_inputs(&g, &Batch::new(vec![0], vec![0], vec![2]))
        .unwrap();
    assert_eq!(alloc.kv_arena_stats().cows, 2);
    assert_eq!(alloc.kv.spans_of(2), &[(0, 4, 6)], "the share is gone");
    assert_eq!(alloc.kv.cell_of(2, 0), Some(4));
    assert_eq!(alloc.kv.cell_of(2, 5), Some(9));
    let (k_end, _) = alloc.copy_kv_to_cpu(0).unwrap();
    for e in 0..ROW {
        assert_eq!(
            k_end[9 * ROW + e],
            500.0 + e as f32 / 10.0,
            "the rows shifted up once more"
        );
    }
}

/// D1 increment 3, structural half: `split_parts` maps each part onto the
/// owner's buffer as a window, so a producer's one output feeds several
/// consumers without a copy and without a second output on `CNode`.
#[test]
fn split_parts_maps_every_part_onto_the_owners_buffer() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
    // A real producer whose single kernel output carries both parts (the D2
    // shape: one concat matmul, two halves).
    let concat = b.matmul_by_name(x, "blk.0.w", crate::tensor::TensorType::F32, 6, 4);
    let parts = b.split_parts(concat, &[3, 3]);
    let (gate, up) = (parts[0], parts[1]);
    b.output(up);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).expect("alloc");
    let owner = alloc.node_buffer(concat).expect("owner");
    assert_eq!(
        (
            alloc.node_buffer(gate).unwrap().offset,
            alloc.node_buffer(gate).unwrap().len
        ),
        (0, 3),
        "the first part is the owner's first window"
    );
    assert_eq!(
        (
            alloc.node_buffer(up).unwrap().offset,
            alloc.node_buffer(up).unwrap().len
        ),
        (3, 3),
        "the second part starts where the first ends"
    );
    assert_eq!(
        alloc.node_buffer(gate).unwrap().id,
        owner.id,
        "a part aliases the owner's buffer"
    );
    assert_eq!(alloc.node_buffer(up).unwrap().id, owner.id);
    assert_eq!(
        g.nodes[gate].view.as_ref().map(|v| (v.src, v.offset)),
        Some((concat, 0)),
        "and it is recorded as a view, so liveness follows the owner"
    );
}

/// D1 increment 3, functional half: the parts are independently consumable —
/// here the second part is an **indices** part (i32 bit patterns in an f32
/// buffer, rule 4) driving `Op::GetRows` over the first part. That is the
/// shape MoE routing needs (a router's top-k indices selecting expert rows)
/// and it exercises the mechanism end to end: one owner, two parts, two
/// different consumers, no second output and no backend change.
#[test]
fn split_parts_parts_feed_independent_consumers() {
    let mut b = GraphBuilder::new();
    // 4 value rows of width 2, then 2 indices: one flat buffer.
    let both = b.input("both", [10, 1, 1, 1], crate::graph::DType::F32);
    let parts = b.split_parts(both, &[8, 2]);
    let (values, indices) = (parts[0], parts[1]);
    let gathered = b.get_rows(values, indices, [2, 2, 1, 1]);
    b.output(gathered);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).expect("alloc");
    let mut data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    data.push(f32::from_bits(3)); // gather row 3
    data.push(f32::from_bits(0)); // then row 0
    alloc.fill_input(&g, "both", &data).expect("fill");
    let mut sched = crate::graph::scheduler::BackendScheduler::new();
    sched.execute(&g, &mut alloc).expect("execute");
    assert_eq!(
        alloc.copy_to_cpu(gathered).expect("read"),
        vec![7.0, 8.0, 1.0, 2.0],
        "rows 3 and 0 of the values part, in the indices part's order"
    );
}

/// The KV regions are persistent across rebuilds (they ARE the cache), so a
/// graph that asks for a different `n_ctx` on the same allocator must be a
/// loud error, not a silent reuse of the older, smaller region.
#[test]
fn kv_region_size_change_is_a_loud_error() {
    fn kv_graph(n_ctx: usize) -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, n_ctx);
        let load = b.kvcache_load(0, 16, n_ctx, 2);
        b.output(load);
        b.build()
    }

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&kv_graph(1024)).unwrap();
    // Unchanged shape: reuse is fine (this is the decode-reuse path).
    alloc.alloc_graph(&kv_graph(1024)).unwrap();
    // Changed n_ctx on a live cache: must fail loudly.
    let err = alloc.alloc_graph(&kv_graph(2048)).unwrap_err();
    assert!(err.contains("KV region for layer 0"), "got: {err}");
}

/// The KV row input is `cells` (C6), and a cell is an index into the arena, so a
/// value at or past the arena is an out-of-bounds write on every backend that does
/// not re-check it (the GPU ones). It is bounded by the *arena* (C7/#60), not by the
/// per-sequence position rule — the two only coincide while a cell equals its
/// position. The guard lives in the allocator so all three backends share it.
#[test]
fn a_cell_beyond_the_arena_is_rejected() {
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
    let _store = b.kvcache_store(0, k, v, 1024);
    let load = b.kvcache_load(0, 16, 1024, 2);
    b.output(load);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();

    // The last legal row is n_ctx - 1.
    alloc.fill_input_i32(&g, "cells", &[1023]).unwrap();
    let err = alloc.fill_input_i32(&g, "cells", &[1024]).unwrap_err();
    assert!(
        err.contains("cell 1024 is past the 1024-cell arena"),
        "got: {err}"
    );
}

/// The same guard must NOT bound `token_ids`: a vocabulary is routinely
/// larger than `n_ctx`.
#[test]
fn token_ids_are_not_bounded_by_n_ctx() {
    let mut b = GraphBuilder::new();
    let ids = b.input("token_ids", [1, 1, 1, 1], crate::graph::DType::I32);
    let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
    let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
    let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
    let emb = b.get_rows(k, ids, [16, 1, 1, 1]);
    let _store = b.kvcache_store(0, emb, v, 8);
    let load = b.kvcache_load(0, 16, 8, 2);
    b.output(load);
    let g = b.build();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();

    alloc
        .fill_input_i32(&g, "token_ids", &[50_000])
        .expect("token ids are not positions");
    let err = alloc.fill_input_i32(&g, "cells", &[8]).unwrap_err();
    assert!(
        err.contains("cell 8 is past the 8-cell arena"),
        "got: {err}"
    );
}

/// A staging buffer is keyed by (node, destination backend): one node
/// feeding two foreign backends gets one buffer each, and a consumer is
/// never offered the other backend's copy. The old single-entry map forced
/// the scheduler to filter by backend on every read (and could not serve
/// two foreign consumers at all).
#[test]
fn staging_is_keyed_by_destination_backend() {
    let mut alloc = GraphAllocator::new();
    alloc.stage_cross_for_test(1, 7, Backend::CPU, 3, 4);
    assert_eq!(
        alloc.cross_buffer(1, 7, Backend::CPU).map(|b| b.id),
        Some(3)
    );
    assert!(
        alloc.cross_buffer(1, 7, Backend::CUDA).is_none(),
        "a CPU staging buffer must not be offered to a CUDA consumer"
    );
    alloc.stage_cross_for_test(1, 7, Backend::CUDA, 4, 4);
    assert_eq!(
        alloc.cross_buffer(1, 7, Backend::CUDA).map(|b| b.id),
        Some(4)
    );
    assert_eq!(
        alloc.cross_buffer(1, 7, Backend::CPU).map(|b| b.id),
        Some(3),
        "staging for a second backend must not clobber the first"
    );
}

/// F5 ([#58]) gate, allocator half: **a staged entry whose boundary wait has
/// not been issued cannot be handed to a consumer, and issuing the wait
/// publishes it.**
///
/// This is the mechanism the missing-wait gate rests on. It is deterministic
/// and needs no device: the pending flag is the state between phase A
/// (`copy_across`) and phase B (`await_cross`), and `cross_input` — the only
/// accessor the scheduler's consumer path uses — refuses it by name. The
/// graph-level version (the same refusal through the real `execute`) is
/// `scheduler::tests::a_staged_boundary_input_cannot_be_consumed_before_its_wait`.
#[test]
fn a_pending_staged_copy_is_refused_until_its_wait_is_issued() {
    let g = {
        use crate::graph::builder::GraphBuilder;
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], super::super::DType::F32);
        let s = b.silu(x);
        b.output(s);
        b.build()
    };
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    // Node 1 (the silu) is on the CPU; the staging entry is for a notional
    // device consumer, so `await_cross` dispatches to the CPU entry's
    // registered no-op (the CPU has no device transfer to wait on) and the
    // test stays device-free.
    const N: NodeId = 1;
    alloc.stage_cross_for_test(g.uid, N, Backend::CUDA, 4, 8);
    alloc.mark_cross_pending_for_test(g.uid, N, Backend::CUDA);

    // Phase B has not run: the entry is not readable.
    let err = alloc
        .cross_input(g.uid, N, Backend::CUDA)
        .expect_err("a pending staging entry must not be published");
    assert!(err.contains("before its boundary wait"), "{err}");
    assert!(err.contains("#58"), "{err}");
    // A key with no entry is still "nothing was staged", not an error.
    assert_eq!(alloc.cross_input(g.uid, N, Backend::CPU).unwrap(), None);

    // A different (uid, node, backend) triple is untouched by the pending flag.
    alloc.stage_cross_for_test(2, N, Backend::CUDA, 5, 8);
    assert!(alloc.cross_input(2, N, Backend::CUDA).unwrap().is_some());

    // Phase B publishes it and the counter records the wait.
    alloc.await_cross(g.uid, N, Backend::CUDA).unwrap();
    let staged = alloc
        .cross_input(g.uid, N, Backend::CUDA)
        .expect("the wait published the entry")
        .expect("the staging buffer exists");
    assert_eq!((staged.id, staged.len), (4, 8));
    assert_eq!(
        alloc.cross_stats(),
        CrossCopyStats {
            copies: 0,
            waits: 1,
            ..CrossCopyStats::default()
        },
        "the wait is counted; no copy was issued here (the test injected the state)"
    );
}

/// F5: `await_cross` is idempotent and self-cleaning — including when the pair
/// never crossed a backend (the scheduler does list a same-backend node whose
/// split differs, e.g. CPU → Metal → CPU). Such an entry is not a *copy*, so it
/// must not be counted as one, or `copies == waits` would stop meaning what the
/// gate reads it as.
#[test]
fn a_same_backend_boundary_input_is_neither_copied_nor_counted() {
    let mut alloc = GraphAllocator::new();
    alloc.mark_cross_pending_for_test(1, 7, Backend::CPU);
    // No buffer for node 7 exists, so this is the "unknown node" refusal.
    assert!(alloc.await_cross(1, 7, Backend::CPU).is_err());

    // With a real node and a same-backend destination, both phases are no-ops.
    let g = {
        use crate::graph::builder::GraphBuilder;
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], super::super::DType::F32);
        let s = b.silu(x);
        b.output(s);
        b.build()
    };
    alloc.alloc_graph(&g).unwrap();
    alloc.copy_across(g.uid, 1, Backend::CPU).unwrap();
    alloc.await_cross(g.uid, 1, Backend::CPU).unwrap();
    assert_eq!(alloc.cross_stats(), CrossCopyStats::default());
    assert!(
        alloc.cross_input(g.uid, 1, Backend::CPU).unwrap().is_none(),
        "a same-backend input reads its canonical buffer, not staging"
    );
}

#[test]
fn cycle_graph_allocation_fails() {
    let mut g = ComputeGraph::default();
    g.nodes.push(super::super::CNode {
        id: 0,
        name: "a".into(),
        op: Op::Add,
        src: vec![1],
        out_shape: [1, 1, 1, 1],
        out_dtype: super::super::DType::F32,
        backend: None,
        meta: super::super::ops::NodeMeta::None,
        view: None,
        layer: None,
    });
    g.nodes.push(super::super::CNode {
        id: 1,
        name: "b".into(),
        op: Op::Add,
        src: vec![0],
        out_shape: [1, 1, 1, 1],
        out_dtype: super::super::DType::F32,
        backend: None,
        meta: super::super::ops::NodeMeta::None,
        view: None,
        layer: None,
    });
    let mut alloc = GraphAllocator::new();
    assert!(alloc.alloc_graph(&g).is_err());
}

/// Issue #122's fail-open twin: a **poisoned** weights lock must not report 0 bytes.
///
/// The mutation this pins is `weights.lock().map(|w| …).unwrap_or(0)`: a panic in
/// another thread poisons the mutex, the sum became 0, and `weights + activations >
/// budget` then under-charged the budget by every resident weight. The registry is
/// append-only, so the value behind the poison is still valid and is recovered.
#[test]
fn a_poisoned_registry_does_not_report_zero_weights() {
    let registry: std::sync::Mutex<Vec<(u64, usize)>> =
        std::sync::Mutex::new(vec![(1, 7), (2, 11)]);
    // Poison it the way a panic inside a holder would.
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _g = registry.lock().unwrap();
        panic!("poison the registry");
    }));
    assert!(registry.is_poisoned(), "the test must actually poison it");
    let bytes = weights_from_lock(registry.lock(), "test registry", |r| {
        r.iter().map(|(_, size)| *size).sum()
    });
    assert_eq!(
        bytes, 18,
        "the poisoned registry must report its real bytes, not 0"
    );
}

/// E4's safety half: a request that would exceed the backend's budget is refused
/// **before the pool is touched**, with the numbers (weights, pooled, this request,
/// budget), instead of surfacing later as a null pointer at execute time.
#[test]
fn a_graph_that_cannot_fit_is_refused_with_its_numbers() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [1024, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    // One byte short of a 4 KiB activation: the gate must fire (the class of 1024
    // elements is 1024 exactly, so this is the requirement minus one).
    alloc.set_memory_budget(Backend::CPU, Some(4095));
    let err = alloc.alloc_graph(&g).unwrap_err();
    assert!(err.contains("out of CPU memory"), "{err}");
    assert!(err.contains("budget"), "{err}");
    assert!(
        err.contains("MiB"),
        "the error must carry the numbers, not a bare failure: {err}"
    );
    // Nothing was allocated: the gate runs before the pool is asked for anything.
    assert_eq!(
        alloc.n_cpu_buffers(),
        0,
        "the refused graph must not touch the pool"
    );
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(report.pool_bytes, 0);
    assert_eq!(report.budget, Some(4095));
    assert_eq!(report.headroom_bytes(), Some(4095));

    // The same graph fits once the budget does.
    alloc.set_memory_budget(Backend::CPU, Some(1 << 20));
    alloc.alloc_graph(&g).unwrap();
    let report = alloc.memory_report(Backend::CPU);
    assert!(report.pool_bytes > 0, "{report:?}");
    assert!(report.live_bytes > 0 && report.live_bytes <= report.pool_bytes);
    assert!(report.peak_live_bytes >= report.live_bytes);
    assert_eq!(report.budget, Some(1 << 20));
    assert!(report.headroom_bytes().unwrap() < 1 << 20);
}

/// E4's accounting half: weights and activations are one comparison, and the peak is
/// a number the caller can read (the ticket's "peak memory is reported").
#[test]
fn the_budget_counts_weights_and_activations_together() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [256, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let weight = tensor_f32("w", [256, 1, 1, 1], vec![0.5; 256]);
    let weight_bytes = weight.data.len();

    let mut alloc = GraphAllocator::new();
    alloc.register_weight("w", weight);
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(report.weights_bytes, weight_bytes, "{report:?}");
    assert_eq!(report.pool_bytes, 0, "nothing is allocated before a build");

    // A budget that covers the weight but not the activation is still a refusal.
    let activation_class =
        crate::graph::allocplan::class_bytes(crate::graph::allocplan::class_size(256 * 4 / 4));
    alloc.set_memory_budget(Backend::CPU, Some(weight_bytes + activation_class - 1));
    let err = alloc.alloc_graph(&g).unwrap_err();
    assert!(err.contains("out of CPU memory"), "{err}");
    assert!(
        err.contains(&format!("{} MiB", weight_bytes / (1024 * 1024))),
        "the refusal must name the weight bytes: {err}"
    );

    // One byte more and it fits.
    alloc.set_memory_budget(Backend::CPU, Some(weight_bytes + activation_class));
    alloc.alloc_graph(&g).unwrap();
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(report.weights_bytes, weight_bytes);
    assert!(
        report.weights_bytes + report.pool_bytes <= report.budget.unwrap(),
        "{report:?}"
    );
}

/// E4 S2's second acceptance: pooled activations are rounded up to their size class,
/// so a graph whose shapes move **inside** one class recycles the pool's buffers
/// instead of reserving one per shape ("no silent growth"). Before this, the free
/// list matched lengths exactly and every shape of a rebuild added a buffer.
#[test]
fn a_rebuild_inside_one_class_reuses_the_pool() {
    let mut alloc = GraphAllocator::new();
    let build = |alloc: &mut GraphAllocator, rows: usize| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [896, rows, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();
        alloc.alloc_graph(&g).unwrap();
    };
    // 896 x 16 = 14336 and 896 x 17 = 15232 both round to the 16384-element class.
    let a = 896 * 16;
    let class = allocplan::class_size(a);
    assert_eq!(allocplan::class_size(896 * 17), class);
    build(&mut alloc, 16);
    let first = alloc.memory_report(Backend::CPU);
    assert_eq!(first.pool_bytes, allocplan::class_bytes(class), "{first:?}");
    let first_buffers = alloc.n_cpu_buffers();

    build(&mut alloc, 17);
    let second = alloc.memory_report(Backend::CPU);
    assert_eq!(
        second.pool_bytes, first.pool_bytes,
        "a shape inside the same class must recycle the class buffer, not reserve another"
    );
    assert_eq!(
        alloc.n_cpu_buffers(),
        first_buffers,
        "the pool grew for a shape that fits an existing class"
    );

    // The ladder is not a no-op: a shape that leaves the class does reserve more.
    build(&mut alloc, 20); // 896 x 20 = 17920 -> the next 16 KiB step
    assert!(
        alloc.memory_report(Backend::CPU).pool_bytes > first.pool_bytes,
        "a bigger class must actually reserve"
    );
}

/// E4 S2: the length contract moved to the owning `BufRef`, so a fill is checked
/// against the node's **logical** length — the pool buffer is rounded up to a class
/// and is routinely longer, which makes "equals the physical length" wrong and
/// "fits in the physical length" too weak.
#[test]
fn a_fill_must_match_the_nodes_logical_length() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [3, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    let report = alloc.memory_report(Backend::CPU);
    assert_eq!(
        report.pool_bytes,
        allocplan::class_bytes(allocplan::class_size(3)),
        "a 3-element activation still occupies a whole class: {report:?}"
    );

    alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0]).unwrap();
    for wrong in [vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32, 2.0]] {
        let err = alloc.fill_input(&g, "x", &wrong).unwrap_err();
        assert!(
            err.contains(&format!(
                "{} elements were supplied but the node holds 3",
                wrong.len()
            )),
            "got: {err}"
        );
    }
    // The refused fills wrote nothing, and the read is the window, not the class.
    assert_eq!(alloc.get_buffer(&g, x).unwrap(), &[1.0, 2.0, 3.0]);
}

/// E4 S2: an input is host-filled **before** execution, so it must never take a buffer
/// that this build's `sweep` released — the previous owner writes that buffer during
/// execution, i.e. after the fill, and the input's value is gone by the time its
/// consumer reads it. Here `t`'s buffer becomes free before `y` is placed, and `t`
/// writes it at step 1.
///
/// This test was originally written with two distinct inputs (`add(x, w)`) because
/// `add(x, x)` could not be built at all: `topo_order` reported a false cycle on a
/// repeated source (#98). The repeated source is legal now and covered end to end by
/// [`a_repeated_source_allocates_and_executes_as_two_reads`]; the two-input form is
/// kept here on purpose so this gate keeps pinning *its* property (input placement)
/// rather than the duplicate-source path.
#[test]
fn an_input_never_takes_a_buffer_the_walk_released() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
    let w = b.input("w", [8, 1, 1, 1], crate::graph::DType::F32);
    let t = b.add(x, w); // its own buffer; this is the only node that writes it
    let c = b.mul(t, w); // last use of `t`, so its buffer is released right after
    let y = b.input("y", [8, 1, 1, 1], crate::graph::DType::F32);
    let o = b.mul(c, y);
    b.output(o);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    let xs: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    let ws = vec![1.0f32; 8];
    let ys: Vec<f32> = (1..=8).map(|v| (v * 10) as f32).collect();
    alloc.fill_input(&g, "x", &xs).unwrap();
    alloc.fill_input(&g, "w", &ws).unwrap();
    alloc.fill_input(&g, "y", &ys).unwrap();
    crate::graph::scheduler::BackendScheduler::new()
        .execute(&g, &mut alloc)
        .expect("execute");
    let want: Vec<f32> = xs
        .iter()
        .zip(&ys)
        .map(|(a, b)| (a + 1.0) * 1.0 * b)
        .collect();
    assert_eq!(
        alloc.copy_to_cpu(o).expect("read"),
        want,
        "`y` must still hold its fill value after `t` executed"
    );
}

/// #98: a node may read the same source twice — `add(x, x)` is `2 * x`
/// written as an addition, `mul(x, x)` is `x * x`, and `GraphBuilder::add`
/// / `mul` pass `&[a, b]` straight through. `alloc_graph` validates with
/// `topo_order()` on every build, so before the fix this graph never
/// reached execution at all (it failed with a false cycle). The assertion
/// is the **value**: `is_ok()` alone would also pass a graph that dropped
/// the second read, and `2 * x` would come back as `x`.
#[test]
fn a_repeated_source_allocates_and_executes_as_two_reads() {
    let xs = [1.5f32, -2.0, 3.25, 0.0];
    let wants = [
        ("add", xs.map(|v| 2.0 * v).to_vec()),
        ("mul", xs.map(|v| v * v).to_vec()),
    ];
    for (name, want) in wants {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        let out = if name == "mul" {
            b.mul(x, x)
        } else {
            b.add(x, x)
        };
        b.output(out);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc
            .alloc_graph(&g)
            .unwrap_or_else(|e| panic!("{name}(x, x) must allocate, got: {e}"));
        alloc.fill_input(&g, "x", &xs).unwrap();
        crate::graph::scheduler::BackendScheduler::new()
            .execute(&g, &mut alloc)
            .unwrap_or_else(|e| panic!("{name}(x, x) must execute, got: {e}"));
        assert_eq!(
            alloc.copy_to_cpu(out).expect("read"),
            want,
            "{name}(x, x) must read x twice, not drop the repeated source"
        );
    }
}

/// E4 S2: a view is a window of its parent's buffer, so the parent must stay alive
/// through the **view's** last reader. D1 wrote this extension starting from the view
/// itself — where the walk stops on its first check, because the view's own liveness
/// is already the bound — so it never ran. Here `s` would take the parent's buffer
/// (same class) and overwrite the window `p0` is read through.
#[test]
fn a_view_keeps_its_parents_buffer_alive_through_later_consumers() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
    let w = b.input("w", [8, 1, 1, 1], crate::graph::DType::F32);
    let a = b.input("a", [4, 1, 1, 1], crate::graph::DType::F32);
    let a4 = b.input("a4", [4, 1, 1, 1], crate::graph::DType::F32);
    let t = b.add(x, w); // its own buffer, 8 elements (one class)
    let parts = b.split_parts(t, &[4, 4]);
    let p0 = parts[0]; // window [0, 4) of `t`'s buffer
    let s = b.mul(a, a4); // same class, placed after `t`'s own last use
    let o = b.mul(p0, s);
    b.output(o);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    let xs: Vec<f32> = (1..=8).map(|v| v as f32).collect();
    alloc.fill_input(&g, "x", &xs).unwrap();
    alloc.fill_input(&g, "w", &vec![1.0f32; 8]).unwrap();
    alloc.fill_input(&g, "a", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    alloc.fill_input(&g, "a4", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    crate::graph::scheduler::BackendScheduler::new()
        .execute(&g, &mut alloc)
        .expect("execute");
    // p0 = (x + w)[0..4] = [2, 3, 4, 5] with w = 1; s = a * a4 = [1, 4, 9, 16].
    let want = vec![2.0, 12.0, 36.0, 80.0];
    assert_eq!(
        alloc.copy_to_cpu(o).expect("read"),
        want,
        "the window must still hold the parent's rows, not the recycled buffer's"
    );
}

/// E5: with a plan in force, the device is only offered for a node whose block the plan
/// offloaded — and a node outside any block only under a full plan. Needs a device to be
/// meaningful (without one the answer is CPU either way), so it is gated on the CUDA
/// build and skips when no device is present.
#[test]
#[cfg(feature = "cuda")]
fn the_offload_plan_keeps_late_blocks_off_the_device() {
    use crate::graph::offload::OffloadPlan;
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("no CUDA device; skipping the E5 placement gate");
        return;
    }
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_cuda());
    // No plan: the pre-E5 behaviour — the device takes every node it can.
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(3)),
        Some(Backend::CUDA)
    );
    alloc.set_offload_plan(Some(OffloadPlan {
        gpu_layers: 2,
        n_layers: 4,
    }));
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(0)),
        Some(Backend::CUDA),
        "an offloaded block may use the device"
    );
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(2)),
        Some(Backend::CPU),
        "a block past the plan stays on the CPU"
    );
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, None),
        Some(Backend::CPU),
        "a partial plan keeps the unblocked tensors (embed/output) on the CPU"
    );
    // A full plan puts the unblocked tensors back on the device.
    alloc.set_offload_plan(Some(OffloadPlan {
        gpu_layers: 4,
        n_layers: 4,
    }));
    assert_eq!(
        alloc.supports_for(&Op::Silu, crate::graph::DType::F32, None),
        Some(Backend::CUDA)
    );
}

/// #153: the allocator stamps the engine's resolved format onto its **CUDA**
/// backend — both when the backend already exists (`set_kv_format`) and when it is
/// created afterwards (`enable_cuda`), so the tag is always the engine's and never
/// a process global. Device-gated.
#[test]
#[cfg(feature = "cuda")]
fn set_kv_format_stamps_the_cuda_layout_per_engine() {
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("no CUDA device; skipping the #153 layout-stamp gate");
        return;
    }
    // (a) the format is known before the backend exists: `enable_cuda` must build
    // it with the stamped layout, not a default.
    let mut alloc = GraphAllocator::new();
    assert_eq!(
        alloc.kv_format(),
        KvFormat::F32,
        "a fresh allocator defaults to f32"
    );
    alloc.set_kv_format(KvFormat::Q8_0);
    assert!(alloc.enable_cuda());
    assert_eq!(
        alloc.cuda().unwrap().kv_layout(),
        crate::cuda::KV_LAYOUT_Q8_0,
        "the backend must be built with the engine's stamped format"
    );
    assert_eq!(alloc.cuda().unwrap().kv_format(), KvFormat::Q8_0);

    // (b) the format changes while the backend exists: `set_kv_format` re-stamps it
    // (an engine's format never moves in production, but the stamp must be total).
    alloc.set_kv_format(KvFormat::F16);
    assert_eq!(
        alloc.cuda().unwrap().kv_layout(),
        crate::cuda::KV_LAYOUT_F16,
        "an existing backend must follow the stamp"
    );
    assert_eq!(alloc.kv_format(), KvFormat::F16);
    assert_eq!(alloc.cuda().unwrap().kv_format(), KvFormat::F16);
}

/// E5: a KV session is one arena (the container carries a single backend tag), so a mixed
/// CPU/device plan cannot resume one. The refusal happens before the file is read — the
/// path below does not exist, which is how the test proves the order.
#[test]
fn a_mixed_offload_plan_refuses_to_resume_a_session() {
    use crate::graph::kvsession::KvSessionExpect;
    let expect = KvSessionExpect {
        backend: Backend::CPU,
        n_ctx: 16,
        n_embd: 8,
    };
    let mut alloc = GraphAllocator::new();
    alloc.set_offload_plan(Some(crate::graph::offload::OffloadPlan {
        gpu_layers: 2,
        n_layers: 4,
    }));
    let err = alloc
        .kv_load(std::path::Path::new("/nonexistent/e5-mixed.bin"), &expect)
        .unwrap_err();
    assert!(err.contains("offload plan"), "got: {err}");
    // An all-CPU or all-device plan proceeds to the file and fails there instead.
    for plan in [
        crate::graph::offload::OffloadPlan {
            gpu_layers: 0,
            n_layers: 4,
        },
        crate::graph::offload::OffloadPlan {
            gpu_layers: 4,
            n_layers: 4,
        },
    ] {
        alloc.set_offload_plan(Some(plan));
        let err = alloc
            .kv_load(std::path::Path::new("/nonexistent/e5-mixed.bin"), &expect)
            .unwrap_err();
        assert!(!err.contains("offload plan"), "{plan:?}: {err}");
    }
}

/// E4 S3's acceptance: a rebuild **re-maps**. The reservation table hands the same
/// class-sized buffers back, so the pool creates nothing (`CpuBackend::alloc_count`, the
/// CPU twin of CUDA's `pool_gen`) and the node → slot mapping is identical — the
/// "reserve, then assign" split, observable.
#[test]
fn a_rebuild_remaps_instead_of_reallocating() {
    /// The buffers a graph's nodes were assigned, by node id: `(backend, pool id)` —
    /// deliberately not the lengths, which differ between two shapes in a class.
    fn slots(alloc: &GraphAllocator, g: &ComputeGraph) -> Vec<(NodeId, Option<(Backend, usize)>)> {
        (0..g.n_nodes())
            .map(|i| (i, alloc.node_buffer(i).map(|b| (b.backend, b.id))))
            .collect()
    }
    let graph = |rows: usize| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [896, rows, 1, 1], crate::graph::DType::F32);
        let w = b.input("w", [896, rows, 1, 1], crate::graph::DType::F32);
        let t = b.add(x, w);
        let y = b.silu(t); // in-place: stays in `t`'s slot
        b.output(y);
        b.build()
    };

    let mut alloc = GraphAllocator::new();
    let g = graph(16);
    alloc.alloc_graph(&g).unwrap();
    let first = slots(&alloc, &g);
    let (allocs, buffers) = (alloc.n_cpu_allocs(), alloc.n_cpu_buffers());
    assert!(allocs > 0 && buffers > 0, "the first build does allocate");

    // The same graph, and a neighbour **in the same class** (896 x 17 = 15232 rounds to
    // the same 16384-element class as 896 x 16 = 14336): both re-map onto the reservation.
    for rows in [16, 17] {
        let g = graph(rows);
        alloc.alloc_graph(&g).unwrap();
        assert_eq!(
            alloc.n_cpu_allocs(),
            allocs,
            "a re-map must not create a pool buffer (rows = {rows})"
        );
        assert_eq!(alloc.n_cpu_buffers(), buffers);
        assert_eq!(
            slots(&alloc, &g),
            first,
            "the same topology gets the same slots (rows = {rows})"
        );
    }

    // A shape that leaves the class reserves a new buffer — the table is not a no-op.
    alloc.alloc_graph(&graph(64)).unwrap();
    assert!(alloc.n_cpu_allocs() > allocs, "a bigger class must reserve");
    assert!(alloc.n_cpu_buffers() > buffers);
}

/// E4 S3: the **reservation** is visible — a released classed buffer stays in the pool and
/// goes to the idle list, so `live_bytes` drops back while `pool_bytes` does not, and the
/// idle slot is what the next graph is assigned.
#[test]
fn a_released_buffer_stays_reserved_and_idle() {
    let mut alloc = GraphAllocator::new();
    let mut b = GraphBuilder::new();
    let x = b.input("x", [256, 1, 1, 1], crate::graph::DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();
    alloc.alloc_graph(&g).unwrap();
    let built = alloc.memory_report(Backend::CPU);
    assert!(built.pool_bytes > 0 && built.live_bytes > 0);
    // Rebuilding the same graph frees every classed buffer at the start and re-assigns
    // them, so at the end the reservation and the live set are exactly what they were.
    alloc.alloc_graph(&g).unwrap();
    let again = alloc.memory_report(Backend::CPU);
    assert_eq!(again.pool_bytes, built.pool_bytes);
    assert_eq!(again.live_bytes, built.live_bytes);
    assert_eq!(again.idle_slots, built.idle_slots);
}

/// E4 S3, CUDA half: a rebuild must not touch the device pool at all. `pool_gen` is the
/// generation counter CUDA invalidates its captured graphs on, so "the plan is untouched"
/// is exactly "the capture survives a rebuild".
#[test]
#[cfg(feature = "cuda")]
fn a_rebuild_does_not_touch_the_device_pool() {
    crate::cuda::CudaState::init();
    if crate::cuda::CudaState::get().is_none() {
        eprintln!("no CUDA device; skipping the E4 S3 device gate");
        return;
    }
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_cuda());
    let graph = |rows: usize| {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [512, rows, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        let z = b.add(y, x);
        b.output(z);
        let mut g = b.build();
        // The device pool is what this gate is about: every node on CUDA.
        for n in g.nodes.iter_mut() {
            n.backend = Some(Backend::CUDA);
        }
        g
    };
    let g = graph(4);
    alloc.alloc_graph(&g).unwrap();
    let gen = alloc.cuda().unwrap().pool_gen();
    let mapping: Vec<_> = (0..g.n_nodes())
        .map(|i| alloc.node_buffer(i).map(|b| (b.backend, b.id)))
        .collect();
    // Same shape: no device traffic.
    alloc.alloc_graph(&g).unwrap();
    assert_eq!(
        alloc.cuda().unwrap().pool_gen(),
        gen,
        "a same-shape rebuild must not allocate or free on the device"
    );
    // Same class, different shape: still no device traffic. (512 x 4 = 2048 and
    // 512 x 3 = 1536 both round to the 2048-element class — asserted, so the case cannot
    // silently stop being the one it claims to test.)
    assert_eq!(
        crate::graph::allocplan::class_size(512 * 4),
        crate::graph::allocplan::class_size(512 * 3)
    );
    alloc.alloc_graph(&graph(3)).unwrap();
    assert_eq!(
        alloc.cuda().unwrap().pool_gen(),
        gen,
        "a shape in the same class re-maps onto the reserved device buffer"
    );
    let same: Vec<_> = (0..g.n_nodes())
        .map(|i| alloc.node_buffer(i).map(|b| (b.backend, b.id)))
        .collect();
    assert_eq!(same, mapping, "and onto the same slots");
}

/// The tiny f32 tensor the accounting tests register (`Tensor` carries raw bytes).
fn tensor_f32(name: &str, shape: [i64; 4], data: Vec<f32>) -> crate::tensor::Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let mut t = crate::tensor::Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
    t.name = name.to_string();
    t
}
