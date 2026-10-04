//! Liveness and reuse: buffer reuse along a chain, parallel chains, inputs and cycles.
//!
//! Split out of `src/graph/alloc/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
fn cycle_graph_allocation_fails() {
    let mut g = ComputeGraph::default();
    g.nodes.push(crate::graph::CNode {
        id: 0,
        name: "a".into(),
        op: Op::Add,
        src: vec![1],
        out_shape: [1, 1, 1, 1],
        out_dtype: crate::graph::DType::F32,
        backend: None,
        meta: crate::graph::ops::NodeMeta::None,
        view: None,
        layer: None,
    });
    g.nodes.push(crate::graph::CNode {
        id: 1,
        name: "b".into(),
        op: Op::Add,
        src: vec![0],
        out_shape: [1, 1, 1, 1],
        out_dtype: crate::graph::DType::F32,
        backend: None,
        meta: crate::graph::ops::NodeMeta::None,
        view: None,
        layer: None,
    });
    let mut alloc = GraphAllocator::new();
    assert!(alloc.alloc_graph(&g).is_err());
}
