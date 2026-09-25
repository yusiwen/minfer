//! Declarative graph builder (Phase 1).
//!
//! Mirrors llama.cpp's `llm_graph_context` builder methods (src/llama-graph.h:950).
//! The builder is pure: it only appends nodes to the graph, it never computes.

use crate::tensor::Tensor;
use crate::vec_ops::RopeStyle;

use super::kvformat::KvFormat;
use super::ops::{
    AttnMeta, AttnMode, EmbedMeta, FusedFfnMeta, FusedQkvMeta, FusedQkvNormMeta, KvcacheMeta,
    MatMulMeta, NodeMeta, NormMeta, Op, QkvBiasRopeStoreMeta, RoPEMeta,
};
use super::{CNode, ComputeGraph, DType, NodeId, ViewAlias};

pub struct GraphBuilder {
    graph: ComputeGraph,
    /// E1: the per-query sequence ids / allowed cell spans. Created on first use
    /// (one node for the whole graph, however many layers call `attn`) and filled
    /// per step by the allocator, like `positions`.
    seq_ids: Option<NodeId>,
    attn_span: Option<NodeId>,
    /// C6: the KV row input, created by `kvcache_store`.
    cells: Option<NodeId>,
    /// Whether this graph's attention window cannot be derived from `positions`
    /// alone (more than one sequence, or a window not starting at cell 0) — set
    /// by the model builders from the *batch* and its KV reservations, never from
    /// `GraphParams` (E2/A7: the sequence count is data). Recorded in the op so a
    /// backend that still derives its bound from positions refuses it.
    explicit_span: bool,
    /// C8b S2: build the window as the `kv_map` input — a list of cell runs per
    /// query (a shared prefix plus a private run) — instead of `attn_span`, one
    /// `[lo, hi)` range. Same op, and the window input's layout *is* the
    /// difference: a backend that cannot gather a map validates the input's size
    /// and refuses the node (CPU can; CUDA is C8b S4; Metal is G5).
    kv_map: bool,
    /// C4 per-engine (issue #99): the storage format of this graph's persistent KV
    /// regions. It is a **parameter** (`params.cparams.kv_format`), not a process
    /// global: it decides a region's *cell width* (`KvcacheMeta::row_elems`), so a
    /// graph built for one format and executed against an allocator that sized the
    /// other refuses loudly in `ensure_kv`. The default is `F32`; the model builders
    /// stamp the loaded engine's resolved format with [`Self::set_kv_format`].
    kv_format: KvFormat,
    /// E5: the transformer block the nodes being created belong to. The model
    /// builders set it once per block (`set_layer(Some(il))`) and clear it after
    /// the loop, so every node the offload policy needs to place carries its
    /// block. `None` for everything outside a block.
    cur_layer: Option<usize>,
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            graph: ComputeGraph::default(),
            seq_ids: None,
            attn_span: None,
            cells: None,
            explicit_span: false,
            kv_map: false,
            // C4 per-engine: F32 until the model builder stamps its own format.
            kv_format: KvFormat::F32,
            cur_layer: None,
        }
    }

    /// E5: stamp every node created from here on with `layer` (`None` = outside any
    /// block). The model builders call this once per block; the offload policy reads
    /// `CNode.layer` to decide which backend may take a node.
    pub fn set_layer(&mut self, layer: Option<usize>) {
        self.cur_layer = layer;
    }

    /// C8b S2: read attention windows from the `kv_map` input (a list of cell runs
    /// per query) instead of `attn_span` (one range). Must be called before the
    /// first `attn` node; the model builders pass `params.cparams.kv_map`, which
    /// the caller sets only for a device that can gather a map.
    pub fn set_kv_map(&mut self, on: bool) {
        self.kv_map = on;
    }

    /// C4 per-engine (issue #99): build this graph's KV nodes for `format`. The
    /// model builders call it once, from `params.cparams.kv_format` — the loaded
    /// engine's resolved policy. A test that wants a packed region asks for one here
    /// instead of mutating process state every other test reads.
    pub fn set_kv_format(&mut self, format: KvFormat) {
        self.kv_format = format;
    }

    /// C4 S2b: whether this graph's KV regions store **packed** cells. A model
    /// builder reads it to keep the device-only fused QKV forms out of a packed
    /// graph: their epilogue writes one K/V element at a time, and a Q8_0 block's
    /// scale needs all 32 of its elements, so a packed decode takes the unfused
    /// bias/rope/store chain (through `store_kv_q8_0`) instead.
    pub fn kv_is_packed(&self) -> bool {
        self.kv_format.is_packed()
    }

    /// Declare that this graph's attention must read the explicit span (E1/E2):
    /// the batch spans more than one sequence, or a window does not start at
    /// cell 0. Must be called before the first `attn`/fused-attention node, since
    /// it is part of the op the builder emits; the model builders pass
    /// `params.cparams.explicit_span`.
    pub fn set_explicit_span(&mut self, on: bool) {
        self.explicit_span = on;
    }

    /// The per-query sequence-id input, created on first call.
    fn seq_ids_input(&mut self, nt: usize) -> NodeId {
        if let Some(id) = self.seq_ids {
            return id;
        }
        let id = self.input("seq_ids", [nt, 1, 1, 1], DType::I32);
        self.seq_ids = Some(id);
        id
    }

    /// The per-query allowed-cell-span input, created on first call: `lo` of
    /// token `t` at `[t]`, `hi` at `[nt + t]`.
    /// The KV **row** input, created on first call: the cell each query token
    /// must be written to. C6: `positions` is the token's index within its
    /// sequence (which is what RoPE needs) and this is the cell the allocator
    /// resolves for `(sequence, position)` — they coincide only while a run
    /// starts at cell 0.
    fn cells_input(&mut self, nt: usize) -> NodeId {
        if let Some(id) = self.cells {
            return id;
        }
        let id = self.input("cells", [nt, 1, 1, 1], DType::I32);
        self.cells = Some(id);
        id
    }

    fn attn_span_input(&mut self, nt: usize) -> NodeId {
        if let Some(id) = self.attn_span {
            return id;
        }
        let id = self.input("attn_span", [2 * nt, 1, 1, 1], DType::I32);
        self.attn_span = Some(id);
        id
    }

    /// C8b S2: the per-query window as a list of `KV_MAP_MAX_SPANS` `(cell, len)`
    /// runs, zero-padded — the layout a sharing sequence needs. One node per graph,
    /// filled per step by the allocator like `attn_span`; the length is what tells
    /// a backend which layout it is reading.
    fn kv_map_input(&mut self, nt: usize) -> NodeId {
        if let Some(id) = self.attn_span {
            return id;
        }
        let id = self.input(
            "kv_map",
            [nt * crate::graph::kvcache::KV_MAP_MAX_SPANS * 2, 1, 1, 1],
            DType::I32,
        );
        self.attn_span = Some(id);
        id
    }

    /// D1 increment 3: expose `owner`'s **single** output as independent **part**
    /// tensors — contiguous runs along the leading (element) axis, each a
    /// zero-copy window.
    ///
    /// This is the "multi-output node" the plan reserved for this increment,
    /// realized without giving `CNode` a second output. A node whose one kernel
    /// writes several logical tensors (MoE's router logits plus the expert rows
    /// they select, MLA's per-head latents) writes them into one output buffer;
    /// this helper turns that buffer into the parts a consumer binds to. The
    /// allocator, the scheduler and all three backends therefore keep their
    /// single-output contract, no backend grows a second-output path, and the
    /// parts inherit the owner's liveness (`extend_through_views`). The backend
    /// never sees a part node at all: `Op::View` is an alias (`CNode::view`), and
    /// its arm is a debug assertion.
    ///
    /// `sizes` are **element counts along the leading axis** and must sum to the
    /// owner's leading dimension; each part keeps the owner's trailing dims. A
    /// part's dtype is the owner's, and an indices part is i32 stored as f32 bit
    /// patterns (rule 4) — which is exactly what makes a `(values, indices)`
    /// pair expressible without a second output dtype, so a part can drive
    /// `Op::GetRows` (see `split_parts_parts_feed_independent_consumers`).
    ///
    /// The composition is proven in production by D2: `fused_ffn_composition` is
    /// a concat matmul whose gate/up halves are consumed through two such
    /// windows.
    pub fn split_parts(&mut self, owner: NodeId, sizes: &[usize]) -> Vec<NodeId> {
        let out_shape = self.graph.nodes[owner].out_shape;
        let total: usize = sizes.iter().sum();
        assert_eq!(
            total, out_shape[0],
            "split_parts: parts sum to {total} but node '{}' has leading dim {}",
            self.graph.nodes[owner].name, out_shape[0]
        );
        let owner_name = self.graph.nodes[owner].name.clone();
        let dtype = self.graph.nodes[owner].out_dtype;
        let mut offset = 0usize;
        let mut parts = Vec::with_capacity(sizes.len());
        for (i, &len) in sizes.iter().enumerate() {
            let shape = [len, out_shape[1], out_shape[2], out_shape[3]];
            parts.push(self.node(
                &format!("{owner_name}_part{i}"),
                Op::View { offset, shape },
                &[owner],
                shape,
                dtype,
                NodeMeta::None,
            ));
            offset += len;
        }
        parts
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
        // D1: the view-like ops do not own their output — they are windows into
        // their single source's buffer. Recorded here (the one construction
        // point, so hand-built graphs get it too) and honoured by the allocator,
        // which maps the node onto the parent's buffer and extends the parent's
        // liveness instead of allocating a copy.
        let view = match (&op, src.len()) {
            (Op::View { offset, .. }, 1) => Some(ViewAlias {
                src: src[0],
                offset: *offset,
            }),
            (Op::Reshape { .. }, 1) | (Op::Permute { .. }, 1) => Some(ViewAlias {
                src: src[0],
                offset: 0,
            }),
            _ => None,
        };
        self.graph.nodes.push(CNode {
            id,
            name: name.to_string(),
            op,
            src: src.to_vec(),
            out_shape,
            out_dtype,
            backend: None,
            meta,
            view,
            // E5: the block this node belongs to (set by the model builders per block).
            layer: self.cur_layer,
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
    pub fn qk_norm(
        &mut self,
        x: NodeId,
        weight: Option<&Tensor>,
        hd: usize,
        nh: usize,
        eps: f32,
    ) -> NodeId {
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

    /// Matmul against a GPU-registered weight by name: builds a MatMul node
    /// whose meta references `weight_name` directly (D2's composition uses it for
    /// the concatenated gate|up weight, which is registered on device by name).
    pub fn matmul_by_name(
        &mut self,
        x: NodeId,
        weight_name: &str,
        ttype: crate::tensor::TensorType,
        out_dim: usize,
        in_dim: usize,
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
        self.node(
            "get_rows",
            Op::GetRows,
            &[x, ids],
            out_shape,
            DType::F32,
            NodeMeta::None,
        )
    }

    /// RoPE. `pos` is an input node carrying per-token positions (data).
    /// decode (nt==1) fused QKV: one concat matmul (wq|wk|wv) whose output
    /// buffer carries q (rows 0..nqt), k (nqt..nqt+nkt), v (nqt+nkt..) after
    /// bias+rope; the backend also stores K/V into the layer's persistent
    /// regions (kv_pair). Output shape = [nqt+nkt+nkt, nt].
    pub fn fused_qkv(
        &mut self,
        x: NodeId,
        pos: NodeId,
        layer: usize,
        meta: FusedQkvMeta,
    ) -> NodeId {
        let nt = self.graph.nodes[x].out_shape[1];
        let od_total = meta.nqt + 2 * meta.nkt;
        // C6: the epilogue stores K/V at allocator-resolved rows, so the node
        // takes the same shared `cells` input `kvcache_store` uses.
        let cells = self.cells_input(nt);
        self.node(
            "fused_qkv",
            Op::FusedQKV { layer },
            &[x, pos, cells],
            [od_total, nt, 1, 1],
            DType::F32,
            NodeMeta::FusedQkv(meta),
        )
    }

    /// decode (nt==1) fused FFN gate+up: one concat matmul (`ffn_gu`) whose
    /// output buffer carries gate (rows 0..nf) and up (nf..2*nf); a single
    /// in-place swiglu pass folds silu(gate)*up into the gate rows. The next
    /// down matmul reads rows 0..nf (od = nf). Output shape = [2*nf, nt].
    /// decode (nt==1) QKV epilogue for mixed-quant layers (D3-8): q/k/v are
    /// the THREE SEPARATE wq/wk/wv matmul outputs (no concat matmul — mixed
    /// quant types); the kernel applies the three biases, ropes q/k in place,
    /// and stores k/v into the persistent regions (kv_pair). Output aliases
    /// q's buffer (in-place; the allocator's sole-consumer alias applies
    /// because attention reads THIS node). Output shape = q's shape [nqt, nt].
    pub fn qkv_bias_rope_store(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        pos: NodeId,
        layer: usize,
        meta: QkvBiasRopeStoreMeta,
    ) -> NodeId {
        let shape = self.graph.nodes[q].out_shape;
        // C6: same split as `fused_qkv` — `pos` for RoPE, `cells` for the store.
        let cells = self.cells_input(shape[1]);
        self.node(
            "qkv_bias_rope_store",
            Op::QkvBiasRopeStore { layer },
            &[q, k, v, pos, cells],
            shape,
            DType::F32,
            NodeMeta::QkvBiasRopeStore(meta),
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

    /// D2: the FFN decode fusion expressed as a **composition** — concat
    /// `MatMul` → gate/up windows → in-place `SwiGLU`.
    ///
    /// This is the same arithmetic `Op::FusedFFN` performs (one matmul over the
    /// concatenated gate|up weight, then `silu(gate) * up`), and the result lands
    /// in the **same bytes**: the gate window is rows `0..nf` of the concat
    /// buffer, which is where the fused kernel leaves its result, so downstream
    /// reads are unchanged and the two paths are comparable element by element.
    /// It needs D1's partial windows at a non-zero offset — before those, the
    /// gate/up halves could only have been copies.
    pub fn fused_ffn_composition(
        &mut self,
        x: NodeId,
        gu_weight: &str,
        ttype: crate::tensor::TensorType,
        in_dim: usize,
        nf: usize,
    ) -> NodeId {
        let nt = self.graph.nodes[x].out_shape[1];
        let concat = self.matmul_by_name(x, gu_weight, ttype, 2 * nf, in_dim);
        let gate = self.node(
            "ffn_gate_window",
            Op::View {
                offset: 0,
                shape: [nf, nt, 1, 1],
            },
            &[concat],
            [nf, nt, 1, 1],
            DType::F32,
            NodeMeta::None,
        );
        let up = self.node(
            "ffn_up_window",
            Op::View {
                offset: nf,
                shape: [nf, nt, 1, 1],
            },
            &[concat],
            [nf, nt, 1, 1],
            DType::F32,
            NodeMeta::None,
        );
        // in place into the gate window (the allocator aliases a view input)
        self.node(
            "ffn_swiglu",
            Op::SwiGLU,
            &[gate, up],
            [nf, nt, 1, 1],
            DType::F32,
            NodeMeta::None,
        )
    }

    /// decode (nt==1) fused QKV with per-head Q/K RMSNorm (Qwen3): one concat
    /// matmul (wq|wk|wv) whose output buffer carries q (rows 0..nqt), k
    /// (nqt..nqt+nkt), v (nqt+nkt..) after per-head norm + rope + store. The
    /// backend normalizes q/k per head (llama `attn_q_norm`/`attn_k_norm`),
    /// then ropes q in place, ropes + stores K, and stores V into the layer's
    /// persistent regions (kv_pair). Output shape = [nqt+nkt+nkt, nt].
    pub fn fused_qkv_norm(
        &mut self,
        x: NodeId,
        pos: NodeId,
        layer: usize,
        meta: FusedQkvNormMeta,
    ) -> NodeId {
        let nt = self.graph.nodes[x].out_shape[1];
        let od_total = meta.nqt + 2 * meta.nkt;
        self.node(
            "fused_qkv_norm",
            Op::FusedQkvNorm { layer },
            &[x, pos],
            [od_total, nt, 1, 1],
            DType::F32,
            NodeMeta::FusedQkvNorm(meta),
        )
    }

    pub fn rope(&mut self, x: NodeId, pos: NodeId, style: RopeStyle, meta: RoPEMeta) -> NodeId {
        let shape = self.graph.nodes[x].out_shape;
        self.node(
            "rope",
            Op::RoPE { style },
            &[x, pos],
            shape,
            DType::F32,
            NodeMeta::Rope(meta),
        )
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
        self.node(
            "softmax",
            Op::Softmax { dim },
            &[x],
            shape,
            DType::F32,
            NodeMeta::None,
        )
    }

    /// Attention over a KV region produced by `kvcache_load`. `pos` carries the
    /// per-token write positions (I32 input), needed for causal masking
    /// (`vl = pos[t]+1`). Output shape = q shape.
    /// Attention over the cells the `attn_span` input names.
    ///
    /// Inputs are `[q, kv, pos, span]`: the span (index 3) is the bound E1 makes
    /// explicit, and `positions` stays at index 2 because the Metal and CUDA arms
    /// still read it there — they derive the bound host-side/on device and are
    /// refused a multi-sequence node until their port lands
    /// (`Backend::supports_attn_span`). The span and seq-id inputs are created on
    /// first use and filled per step.
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
        let span = if self.kv_map {
            self.kv_map_input(nt)
        } else {
            self.attn_span_input(nt)
        };
        self.seq_ids_input(nt);
        self.node(
            "attn",
            Op::Attn {
                mode,
                explicit_span: self.explicit_span,
            },
            &[q, kv, pos, span],
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
        self.node(
            "swiglu",
            Op::SwiGLU,
            &[gate, up],
            shape,
            DType::F32,
            NodeMeta::None,
        )
    }

    /// Write this step's K/V into the layer's persistent KV region at the
    /// positions carried by `pos`. `n_ctx` sizes the persistent region.
    /// Write this forward's K/V into the layer's persistent regions. C6: the row
    /// input is the allocator-resolved `cells` (created here, like `attn_span`),
    /// not `positions` — callers no longer pass a buffer, because the mapping
    /// `(sequence, position) -> cell` belongs to the cell store.
    pub fn kvcache_store(&mut self, layer: usize, k: NodeId, v: NodeId, n_ctx: usize) -> NodeId {
        let n_embd = self.graph.nodes[k].out_shape[0];
        let nt = self.graph.nodes[k].out_shape[1];
        let cells = self.cells_input(nt);
        // shape mirrors the *logical* region ([n_kv_embd, n_ctx]) so it is what the
        // store's K/V input means; the allocator sizes the persistent region from
        // this meta's `row_elems` (C4: a packed region is narrower per cell than
        // `n_embd`).
        self.node(
            &format!("kv_store.{layer}"),
            Op::KvcacheStore { layer },
            &[k, v, cells],
            [n_embd, n_ctx, 1, 1],
            DType::F32,
            NodeMeta::Kvcache(KvcacheMeta {
                n_embd,
                n_head_kv: 0,
                row_elems: self.kv_format.row_elems(n_embd),
            }),
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
            NodeMeta::Kvcache(KvcacheMeta {
                n_embd,
                n_head_kv,
                row_elems: self.kv_format.row_elems(n_embd),
            }),
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
}
