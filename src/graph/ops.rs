//! Operator definitions and node metadata (Phase 1).
//!
//! Mirrors the `Op` enum in docs/GRAPH-REFACTOR-PLAN.md §3.1. Key invariant:
//! **KV cache positions are data, not structure** — `KvcacheStore`/`KvcacheLoad`
//! carry only the layer index; the write position is injected via an input node
//! (`positions`), so the graph topology never depends on `n_past`.

use crate::tensor::TensorType;
use crate::vec_ops::RopeStyle;

/// Attention mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnMode {
    Mha,
    Gqa,
    Flash,
}

/// Fused-op capability tag: drives the fusion pass (Phase 4) — a fusion is only
/// applied when the target backend reports `supports_fused(FusedOp)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOp {
    SwiGLU,     // silu(gate) * up
    BiasRope,   // add_bias + rope (+ kv store on GPU: attn_bias_rope_store)
    BatchMatMul, // multiple matmuls sharing one quantized activation
}

/// Operator type. Implements full `PartialEq` (payloads included) so debug
/// builds can verify graph-rebuild structural identity; the production graph
/// reuse decision is params-only (see docs/GRAPH-REFACTOR-PLAN.md §6).
#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    /// Leaf input node (token ids, positions, KV idx, ...). Filled externally
    /// each step via `GraphAllocator::fill_input`; never part of the topology.
    Input,

    // ---- element-wise ----
    Add,
    Mul,
    Scale(f32),
    Silu,

    // ---- reduction ----
    Softmax { dim: usize },

    // ---- normalization ----
    RmsNorm { eps: f32 },

    // ---- linear algebra ----
    MatMul { transpose_b: bool },

    // ---- indexing ----
    GetRows,

    // ---- positional encoding ----
    RoPE { style: RopeStyle },

    // ---- attention ----
    Attn { mode: AttnMode },

    // ---- KV cache (persistent external buffer; positions are data) ----
    KvcacheStore { layer: usize },
    KvcacheLoad { layer: usize },

    // ---- view / reshape ----
    View { offset: usize, shape: [usize; 4] },
    Reshape { shape: [usize; 4] },
    Permute { dims: [usize; 4] },

    // ---- fused ops (fusion pass output, gated by backend supports_fused) ----
    SwiGLU,
    FusedBiasRope,
    BatchMatMul,
}

/// Per-node metadata.
///
/// Deviation from the plan's `meta: Box<dyn Any + Send + Sync>`: a concrete
/// enum is used instead — it is `PartialEq` (needed for the debug graph-reuse
/// structural check), avoids downcast panics, and keeps `CNode` `Clone`.
#[derive(Debug, Clone, PartialEq)]
pub enum NodeMeta {
    None,
    MatMul(MatMulMeta),
    Norm(NormMeta),
    Rope(RoPEMeta),
    Attn(AttnMeta),
    Kvcache(KvcacheMeta),
    Embed(EmbedMeta),
}

/// Matmul target weight (+ optional bias) — resolved to a backend buffer by
/// name at execution time; `weight_ttype` lets GPU backends pick the kernel
/// without holding the whole Tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct MatMulMeta {
    pub weight_name: String,
    pub bias_name: Option<String>,
    pub weight_ttype: TensorType,
    /// Weight dims under the GGUF convention: memory `[out][in]` row-major,
    /// metadata `[in, out]` — so `in_dim = shape[0]`, `out_dim = shape[1]`.
    pub in_dim: usize,
    pub out_dim: usize,
}

/// Normalization weight/bias names.
#[derive(Debug, Clone, PartialEq)]
pub struct NormMeta {
    pub weight_name: Option<String>,
    pub bias_name: Option<String>,
}

/// RoPE params. `n_head`/`hd` are needed by the kernel (the rope applies per
/// head over `hd` dims), so they ride in the metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct RoPEMeta {
    pub freq_base: f32,
    pub freq_scale: f32,
    pub n_head: usize,
    pub hd: usize,
}

/// Attention params. `layer` lets the backend resolve the layer's K/V regions
/// (each layer has two persistent regions: K and V). `nkt` = KV row stride
/// (n_embd), `scale` = QK scale.
#[derive(Debug, Clone, PartialEq)]
pub struct AttnMeta {
    pub layer: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub hd: usize,
    pub hd_kv: usize,
    pub nkt: usize,
    pub scale: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct KvcacheMeta {
    pub n_embd: usize,
    pub n_head_kv: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedMeta {
    pub vocab_size: usize,
    pub weight_name: String,
    pub weight_ttype: TensorType,
}
