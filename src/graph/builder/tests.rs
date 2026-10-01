//! `#[cfg(test)] mod tests` for `src/graph/builder.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

fn f32_tensor(name: &str, shape: [i64; 4]) -> Tensor {
    let mut t = Tensor::new(crate::tensor::TensorType::F32, &shape);
    // [#244]: this helper is `Tensor::new`'s only consumer, and every assertion in
    // the tests below reads `name` or a graph shape — never the strides or the
    // allocation `Tensor::new` computes. Without these the body could be wrong
    // (a zeroed stride, a mis-sized buffer) with every gate still green; #239
    // recorded the gap, this closes it. `nbytes()` is derived from the strides,
    // so the two assertions together observe both halves of the body.
    //
    // [#244]: https://github.com/yusiwen/minfer/issues/244
    assert_eq!(t.strides[0], 4, "f32 element stride");
    assert_eq!(t.strides[1], 4 * shape[0] as usize);
    assert_eq!(t.strides[2], t.strides[1] * shape[1] as usize);
    assert_eq!(t.strides[3], t.strides[2] * shape[2] as usize);
    assert_eq!(
        t.nbytes(),
        4 * shape.iter().map(|&d| d as usize).product::<usize>(),
        "a dense f32 tensor of {shape:?}"
    );
    assert_eq!(t.data.len(), t.nbytes(), "the buffer is allocated");
    t.name = name.to_string();
    t
}

#[test]
fn builder_creates_topo_sorted_graph() {
    let mut b = GraphBuilder::new();
    let ids = b.input("token_ids", [2, 1, 1, 1], DType::I32);
    let w = f32_tensor("tok_embd", [16, 8, 1, 1]);
    let h = b.embedding(ids, &w);
    let n = f32_tensor("attn_norm", [16, 1, 1, 1]);
    let h = b.rms_norm(h, Some(&n), 1e-5);
    let wq = f32_tensor("blk.0.attn_q", [16, 16, 1, 1]);
    let q = b.matmul(h, &wq, None);
    b.output(q);

    let g = b.build();
    assert_eq!(g.n_nodes(), 4); // ids, embed, rms_norm, matmul
    assert_eq!(g.inputs, vec![0]);
    assert_eq!(g.outputs, vec![3]);
    // topological order is the node order itself
    assert_eq!(g.topo_order().unwrap(), vec![0, 1, 2, 3]);
    // output shapes
    assert_eq!(g.node(1).out_shape, [16, 2, 1, 1]); // embed [n_embd, nt]
    assert_eq!(g.node(3).out_shape, [16, 2, 1, 1]);
    // metadata payloads
    let meta = match &g.node(3).meta {
        NodeMeta::MatMul(m) => m,
        other => panic!("expected MatMulMeta, got {other:?}"),
    };
    assert_eq!(meta.weight_name, "blk.0.attn_q");
    assert_eq!(meta.bias_name, None);
}

#[test]
fn kv_nodes_carry_layer_only() {
    let mut b = GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
    let k = b.input("k", [16, 1, 1, 1], DType::F32);
    let v = b.input("v", [16, 1, 1, 1], DType::F32);
    let store = b.kvcache_store(3, k, v, 1024);
    let load = b.kvcache_load(3, 16, 1024, 2);
    let g = b.build();

    assert_eq!(g.node(store).op, Op::KvcacheStore { layer: 3 });
    assert_eq!(g.node(load).op, Op::KvcacheLoad { layer: 3 });
    assert_eq!(g.node(load).out_shape, [16, 1024, 1, 1]);
    // no n_past anywhere in the IR: payloads only carry the layer index
    assert_ne!(g.node(store).op, Op::KvcacheStore { layer: 4 });
}

/// E5: `set_layer` stamps the nodes created after it, so the offload policy can place a
/// node by its block. Nothing else in the IR carries the block (the KV ops' meta layer is
/// only the KV layers), which is why this is a builder-level contract.
#[test]
fn set_layer_tags_the_nodes_created_after_it() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let before = b.silu(x);
    b.set_layer(Some(2));
    let in_block = b.silu(x);
    let view = b.split_parts(in_block, &[2, 2])[0];
    b.set_layer(None);
    let after = b.silu(x);
    let g = b.build();
    assert_eq!(g.node(x).layer, None, "an input is outside any block");
    assert_eq!(g.node(before).layer, None);
    assert_eq!(g.node(in_block).layer, Some(2));
    assert_eq!(
        g.node(view).layer,
        Some(2),
        "views inherit the block they were made in"
    );
    assert_eq!(g.node(after).layer, None, "cleared after the block");
}

#[test]
fn swiglu_builder_and_meta() {
    let mut b = GraphBuilder::new();
    let g_ = b.input("gate", [8, 1, 1, 1], DType::F32);
    let u = b.input("up", [8, 1, 1, 1], DType::F32);
    let s = b.swiglu(g_, u);
    let g = b.build();
    assert_eq!(g.node(s).op, Op::SwiGLU);
    assert_eq!(g.node(s).src, vec![0, 1]);
}
