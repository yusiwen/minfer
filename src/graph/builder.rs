//! Declarative graph builder (Phase 1).
//!
//! Mirrors llama.cpp's `llm_graph_context` builder methods (src/llama-graph.h:950).
//! The builder is pure: it only appends nodes to the graph, it never computes.

use crate::tensor::Tensor;
use crate::vec_ops::RopeStyle;

use super::ops::{
    AttnMeta, AttnMode, EmbedMeta, FusedFfnMeta, FusedQkvMeta, KvcacheMeta, MatMulMeta, NodeMeta, NormMeta,
    Op, RoPEMeta,
};
use super::{CNode, ComputeGraph, DType, NodeId};

pub struct GraphBuilder {
    graph: ComputeGraph,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self { graph: ComputeGraph::default() }
    }

    /// Create an operator node; returns its id.
    pub fn node(
        &mut self,
        name: &str,
        op: Op,
        src: &[NodeId],
        out_shape: [usize; 4],
        out_dtype: DType,
        meta: NodeMeta,
    ) -> NodeId {
        let id = self.graph.nodes.len();
        self.graph.nodes.push(CNode {
            id,
            name: name.to_string(),
            op,
            src: src.to_vec(),
            out_shape,
            out_dtype,
            backend: None,
            meta,
        });
        id
    }

    /// Leaf input node — filled externally every step; never part of the
    /// topology (so `n_past`/positions changes never force a graph rebuild).
    pub fn input(&mut self, name: &str, shape: [usize; 4], dtype: DType) -> NodeId {
        let id = self.node(name, Op::Input, &[], shape, dtype, NodeMeta::None);
        self.graph.inputs.push(id);
        id
    }

    // ---- convenience methods (shape helpers mirror llama.cpp layouts) ----

    /// Embedding lookup: token ids → rows of `weight` (`GetRows`).
    /// Output `[n_embd, nt, 1, 1]`.
    pub fn embedding(&mut self, ids: NodeId, weight: &Tensor) -> NodeId {
        let n_embd = weight.shape[0] as usize;
        let nt = self.graph.nodes[ids].out_shape[0];
        let out_shape = [n_embd, nt, 1, 1];
        self.node(
            "embed",
            Op::GetRows,
            &[ids],
            out_shape,
            DType::F32,
            NodeMeta::Embed(EmbedMeta {
                vocab_size: weight.shape[1] as usize,
                weight_name: weight.name.clone(),
                weight_ttype: weight.ttype,
            }),
        )
    }

    /// RMSNorm over the leading dimension; output shape = input shape.
    pub fn rms_norm(&mut self, x: NodeId, weight: Option<&Tensor>, eps: f32) -> NodeId {
        let shape = self.graph.nodes[x].out_shape;
        self.node(
            "rms_norm",
            Op::RmsNorm { eps },
            &[x],
            shape,
            DType::F32,
            NodeMeta::Norm(NormMeta {
                weight_name: weight.map(|t| t.name.clone()),
                bias_name: None,
            }),
        )
    }

    /// Per-head RMSNorm (Qwen3 Q/K norms): normalizes each contiguous `hd`-wide
    /// head row of the flat `[nt*nh*hd]` buffer with a weight of length `hd`.
    /// Output shape = input shape.
    pub fn qk_norm(&mut self, x: NodeId, weight: Option<&Tensor>, hd: usize, nh: usize, eps: f32) -> NodeId {
        let shape = self.graph.nodes[x].out_shape;
        self.node(
            "qk_norm",
            Op::QkNorm { hd, nh, eps },
            &[x],
            shape,
            DType::F32,
            NodeMeta::Norm(NormMeta {
                weight_name: weight.map(|t| t.name.clone()),
                bias_name: None,
            }),
        )
    }

    /// Matrix multiply `w @ x` (+ optional bias).
    ///
    /// Weight convention (llama.cpp/GGUF): the tensor metadata is `[in, out]`
    /// (ne[0] = input dim, fastest) while memory is `[out][in]` row-major —
    /// i.e. the output dim is `shape[1]`. Activations are `[n_embd, nt, 1, 1]`
    /// (features × tokens), so the output is `[shape[1], nt, 1, 1]`.
    pub fn matmul(&mut self, x: NodeId, w: &Tensor, bias: Option<&Tensor>) -> NodeId {
        let out = w.shape[1] as usize;
        let nt = self.graph.nodes[x].out_shape[1];
        let name = format!("matmul_{}", w.name);
        self.node(
            &name,
            Op::MatMul { transpose_b: false },
            &[x],
            [out, nt, 1, 1],
            DType::F32,
            NodeMeta::MatMul(MatMulMeta {
                weight_name: w.name.clone(),
                bias_name: bias.map(|b| b.name.clone()),
                weight_ttype: w.ttype,
                in_dim: w.shape[0] as usize,
                out_dim: w.shape[1] as usize,
            }),
        )
    }

    /// Matmul against a GPU-registered weight by name (probes/tests only):
    /// builds a MatMul node whose meta references `weight_name` directly.
    #[allow(dead_code)]
    pub fn matmul_by_name(
        &mut self, x: NodeId, weight_name: &str, ttype: crate::tensor::TensorType,
        out_dim: usize, in_dim: usize,
    ) -> NodeId {
        let nt = self.graph.nodes[x].out_shape[1];
        self.node(
            "matmul_named",
            Op::MatMul { transpose_b: false },
            &[x],
            [out_dim, nt, 1, 1],
            DType::F32,
            NodeMeta::MatMul(MatMulMeta {
                weight_name: weight_name.to_string(),
                bias_name: None,
                weight_ttype: ttype,
                in_dim,
                out_dim,
            }),
        )
    }

    /// Generic row selection: `out[t] = x[ids[t]]` (llama `ggml_get_rows`;
    /// also used for the n_out tail-row reduction). `ids` is an I32 input.
    pub fn get_rows(&mut self, x: NodeId, ids: NodeId, out_shape: [usize; 4]) -> NodeId {
        self.node("get_rows", Op::GetRows, &[x, ids], out_shape, DType::F32, NodeMeta::None)
    }

    /// RoPE. `pos` is an input node carrying per-token positions (data).
    /// decode (nt==1) fused QKV: one concat matmul (wq|wk|wv) whose output
    /// buffer carries q (rows 0..nqt), k (nqt..nqt+nkt), v (nqt+nkt..) after
    /// bias+rope; the backend also stores K/V into the layer's persistent
    /// regions (kv_pair). Output shape = [nqt+nkt+nkt, nt].
    pub fn fused_qkv(&mut self, x: NodeId, pos: NodeId, layer: usize, meta: FusedQkvMeta) -> NodeId {
        let nt = self.graph.nodes[x].out_shape[1];
        let od_total = meta.nqt + 2 * meta.nkt;
        self.node(
            "fused_qkv",
            Op::FusedQKV { layer },
            &[x, pos],
            [od_total, nt, 1, 1],
            DType::F32,
            NodeMeta::FusedQkv(meta),
        )
    }

    /// decode (nt==1) fused FFN gate+up: one concat matmul (`ffn_gu`) whose
    /// output buffer carries gate (rows 0..nf) and up (nf..2*nf); a single
    /// in-place swiglu pass folds silu(gate)*up into the gate rows. The next
    /// down matmul reads rows 0..nf (od = nf). Output shape = [2*nf, nt].
    pub fn fused_ffn(&mut self, x: NodeId, meta: FusedFfnMeta) -> NodeId {
        let nt = self.graph.nodes[x].out_shape[1];
        self.node(
            "fused_ffn",
            Op::FusedFFN,
            &[x],
            [2 * meta.nf, nt, 1, 1],
            DType::F32,
            NodeMeta::FusedFfn(meta),
        )
    }

    pub fn rope(&mut self, x: NodeId, pos: NodeId, style: RopeStyle, meta: RoPEMeta) -> NodeId {
        let shape = self.graph.nodes[x].out_shape;
        self.node("rope", Op::RoPE { style }, &[x, pos], shape, DType::F32, NodeMeta::Rope(meta))
    }

    pub fn silu(&mut self, x: NodeId) -> NodeId {
        let shape = self.graph.nodes[x].out_shape;
        self.node("silu", Op::Silu, &[x], shape, DType::F32, NodeMeta::None)
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let shape = self.graph.nodes[a].out_shape;
        self.node("add", Op::Add, &[a, b], shape, DType::F32, NodeMeta::None)
    }

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let shape = self.graph.nodes[a].out_shape;
        self.node("mul", Op::Mul, &[a, b], shape, DType::F32, NodeMeta::None)
    }

    /// Softmax builder (op vocabulary; the fused attention kernels softmax
    /// internally, so no live graph emits a standalone softmax node today).
    #[allow(dead_code)]
    pub fn softmax(&mut self, x: NodeId, dim: usize) -> NodeId {
        let shape = self.graph.nodes[x].out_shape;
        self.node("softmax", Op::Softmax { dim }, &[x], shape, DType::F32, NodeMeta::None)
    }

    /// Attention over a KV region produced by `kvcache_load`. `pos` carries the
    /// per-token write positions (I32 input), needed for causal masking
    /// (`vl = pos[t]+1`). Output shape = q shape.
    pub fn attn(
        &mut self,
        q: NodeId,
        kv: NodeId,
        pos: NodeId,
        mode: AttnMode,
        meta: AttnMeta,
    ) -> NodeId {
        // Attention output is one row per query head: [n_head*hd, nt]. The q
        // input may be a larger fused concat buffer (G4 FusedQKV carries
        // q|k|v), so the output shape comes from the meta, not from q.
        let nt = self.graph.nodes[q].out_shape[1];
        self.node(
            "attn",
            Op::Attn { mode },
            &[q, kv, pos],
            [meta.n_head * meta.hd, nt, 1, 1],
            DType::F32,
            NodeMeta::Attn(meta),
        )
    }

    /// Fused SwiGLU (fusion-pass target; backends without a fused kernel
    /// decompose to silu+mul at execution). Used by tests; the Qwen2 graph
    /// builder emits gate/up separately and lets the fusion pass combine them.
    #[allow(dead_code)]
    pub fn swiglu(&mut self, gate: NodeId, up: NodeId) -> NodeId {
        let shape = self.graph.nodes[gate].out_shape;
        self.node("swiglu", Op::SwiGLU, &[gate, up], shape, DType::F32, NodeMeta::None)
    }

    /// Write this step's K/V into the layer's persistent KV region at the
    /// positions carried by `pos`. `n_ctx` sizes the persistent region.
    pub fn kvcache_store(
        &mut self,
        layer: usize,
        k: NodeId,
        v: NodeId,
        pos: NodeId,
        n_ctx: usize,
    ) -> NodeId {
        let n_embd = self.graph.nodes[k].out_shape[0];
        // shape mirrors the persistent region so the allocator can size it
        self.node(
            &format!("kv_store.{layer}"),
            Op::KvcacheStore { layer },
            &[k, v, pos],
            [n_embd, n_ctx, 1, 1],
            DType::F32,
            NodeMeta::Kvcache(KvcacheMeta { n_embd, n_head_kv: 0 }),
        )
    }

    /// View of the layer's persistent KV region (K rows `[n_embd, n_ctx]`;
    /// the executor exposes only the written prefix). Topology independent of
    /// `n_past`.
    pub fn kvcache_load(
        &mut self,
        layer: usize,
        n_embd: usize,
        n_ctx: usize,
        n_head_kv: usize,
    ) -> NodeId {
        self.node(
            &format!("kv_load.{layer}"),
            Op::KvcacheLoad { layer },
            &[],
            [n_embd, n_ctx, 1, 1],
            DType::F32,
            NodeMeta::Kvcache(KvcacheMeta { n_embd, n_head_kv }),
        )
    }

    /// Mark `node` as a graph output (e.g. logits).
    pub fn output(&mut self, node: NodeId) {
        if !self.graph.outputs.contains(&node) {
            self.graph.outputs.push(node);
        }
    }

    pub fn build(self) -> ComputeGraph {
        self.graph
    }
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_tensor(name: &str, shape: [i64; 4]) -> Tensor {
        let mut t = Tensor::new(crate::tensor::TensorType::F32, &shape);
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
        let store = b.kvcache_store(3, k, v, pos, 1024);
        let load = b.kvcache_load(3, 16, 1024, 2);
        let g = b.build();

        assert_eq!(g.node(store).op, Op::KvcacheStore { layer: 3 });
        assert_eq!(g.node(load).op, Op::KvcacheLoad { layer: 3 });
        assert_eq!(g.node(load).out_shape, [16, 1024, 1, 1]);
        // no n_past anywhere in the IR: payloads only carry the layer index
        assert_ne!(g.node(store).op, Op::KvcacheStore { layer: 4 });
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
}
