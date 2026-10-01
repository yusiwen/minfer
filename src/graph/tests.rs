//! `#[cfg(test)] mod tests` for `src/graph/mod.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::builder::GraphBuilder;

fn chain_graph(n_ops: usize) -> ComputeGraph {
    // input -> silu -> add -> silu -> add -> ... (linear chain)
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let mut h = x;
    for i in 0..n_ops {
        h = if i % 2 == 0 { b.silu(h) } else { b.add(h, x) };
    }
    b.output(h);
    b.build()
}

#[test]
fn topo_order_validates_chain() {
    let g = chain_graph(4);
    let order = g.topo_order().unwrap();
    assert_eq!(order.len(), g.n_nodes());
    // every node appears before its consumers
    for node in &g.nodes {
        for &s in &node.src {
            let ps = order.iter().position(|&x| x == s).unwrap();
            let pn = order.iter().position(|&x| x == node.id).unwrap();
            assert!(ps < pn, "src {} must precede consumer {}", s, node.id);
        }
    }
}

/// #98: `add(x, x)` lists one predecessor twice, which is one edge read
/// twice — not two edges. The in-degree pass must count it once (the
/// release pass already decrements once per node), or the validator reports
/// a false cycle and `GraphAllocator::alloc_graph` refuses a legal DAG.
#[test]
fn topo_order_accepts_a_repeated_source() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let double = b.add(x, x);
    b.output(double);
    let g = b.build();
    let order = g.topo_order().expect("add(x, x) is acyclic");
    assert_eq!(order.len(), g.n_nodes(), "every node must be ordered");
    let px = order.iter().position(|&i| i == x).unwrap();
    let pd = order.iter().position(|&i| i == double).unwrap();
    assert!(px < pd, "the source must precede its consumer");
}

#[test]
fn topo_order_detects_cycle() {
    // hand-built cycle: a -> b -> a
    let mut g = ComputeGraph::default();
    g.nodes.push(CNode {
        id: 0,
        name: "a".into(),
        op: Op::Add,
        src: vec![1],
        out_shape: [1, 1, 1, 1],
        out_dtype: DType::F32,
        backend: None,
        meta: NodeMeta::None,
        view: None,
        layer: None,
    });
    g.nodes.push(CNode {
        id: 1,
        name: "b".into(),
        op: Op::Add,
        src: vec![0],
        out_shape: [1, 1, 1, 1],
        out_dtype: DType::F32,
        backend: None,
        meta: NodeMeta::None,
        view: None,
        layer: None,
    });
    assert_eq!(
        g.topo_order().unwrap_err(),
        "cycle detected: 0/2 nodes ordered",
        "a genuine cycle must still be refused by the cycle check itself, \
         not by some other error the repeated-source fix might have introduced"
    );
}

#[test]
fn op_partial_eq_compares_payloads() {
    assert_eq!(Op::RmsNorm { eps: 1e-5 }, Op::RmsNorm { eps: 1e-5 });
    assert_ne!(Op::RmsNorm { eps: 1e-5 }, Op::RmsNorm { eps: 1e-6 });
    assert_ne!(
        Op::MatMul { transpose_b: true },
        Op::MatMul { transpose_b: false }
    );
    assert_eq!(Op::KvcacheLoad { layer: 2 }, Op::KvcacheLoad { layer: 2 });
    assert_ne!(Op::KvcacheLoad { layer: 2 }, Op::KvcacheLoad { layer: 3 });
}

#[test]
fn dtype_size() {
    assert_eq!(DType::F32.size(), 4);
    assert_eq!(DType::F16.size(), 2);
    assert_eq!(DType::I32.size(), 4);
    assert_eq!(DType::Q8_0.size(), 1);
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `mod.rs` (bucket B of the dead-code census — the only
// test caller, `graph::tests::dtype_size`, already lives in this subtree).
// ────────────────────────────────────────────────────────────────────────────

impl DType {
    /// Bytes per element (Q8_0 = 1 byte per quantized element block member;
    /// actual block layout is a backend concern).
    ///
    /// Test-only (#239): driven by `graph::tests::dtype_size`.
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I32 => 4,
            DType::Q8_0 => 1,
        }
    }
}
