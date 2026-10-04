//! D1 views: zero-copy windows, their liveness, and `split_parts`.
//!
//! Split out of `src/graph/alloc/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
