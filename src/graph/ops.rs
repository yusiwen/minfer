//! Operator definitions and node metadata (Phase 1).
//!
//! Mirrors the `Op` enum in docs/GRAPH-REFACTOR-PLAN.md §3.1. Key invariant:
//! **KV cache positions are data, not structure** — `KvcacheStore`/`KvcacheLoad`
//! carry only the layer index; the write position is injected via an input node
//! (`positions`), so the graph topology never depends on `n_past`.

use crate::tensor::TensorType;
use crate::vec_ops::RopeStyle;

/// Attention mode.
// Mha is part of the full attention-mode vocabulary (ggml parity); only Gqa /
// Flash are constructed by the supported architectures today.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttnMode {
    #[allow(dead_code)]
    Mha,
    Gqa,
    Flash,
}

/// Fused-op capability tag: drives the fusion pass (Phase 4) — a fusion is only
/// applied when the target backend reports `supports_fused(FusedOp)`.
// BatchMatMul / QKVBiasRopeStore are the planned fused variants (the decode
// path uses FusedQKV today).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedOp {
    SwiGLU,     // silu(gate) * up
    BiasRope,   // add_bias + rope (+ kv store on GPU: attn_bias_rope_store)
    #[allow(dead_code)]
    BatchMatMul, // multiple matmuls sharing one quantized activation
    #[allow(dead_code)]
    QKVBiasRopeStore, // decode QKV: concat matmul + bias+rope+store (nt==1)
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
    /// Part of the full op vocabulary (ggml parity); no supported architecture
    /// emits a scale node yet.
    #[allow(dead_code)]
    Scale(f32),
    Silu,

    // ---- reduction ----
    /// Softmax is part of the op vocabulary; the attention kernels fuse the
    /// softmax internally, so no standalone softmax node is emitted today.
    #[allow(dead_code)]
    Softmax { dim: usize },

    // ---- normalization ----
    RmsNorm { eps: f32 },
    /// Per-head RMSNorm (Qwen3 `attn_q_norm`/`attn_k_norm`): the input is a
    /// flat token-major buffer `[nt * nh * hd]`; each contiguous `hd`-wide row
    /// (`t*nh + h`) is RMS-normalized with a weight of length `hd`. The buffer
    /// layout makes this a contiguous `[nt*nh, hd]` matrix, so execution reuses
    /// the RMSNorm kernels with `d = hd`, `n = nt*nh`.
    QkNorm { hd: usize, nh: usize, eps: f32 },

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
    /// View / Reshape / Permute / BatchMatMul are part of the full ggml op
    /// vocabulary; the Qwen2 graph builder doesn't emit them (yet).
    #[allow(dead_code)]
    View { offset: usize, shape: [usize; 4] },
    #[allow(dead_code)]
    Reshape { shape: [usize; 4] },
    #[allow(dead_code)]
    Permute { dims: [usize; 4] },

    // ---- fused ops (fusion pass output, gated by backend supports_fused) ----
    SwiGLU,
    FusedBiasRope,
    #[allow(dead_code)]
    BatchMatMul,
    /// decode (nt==1) fused QKV: one concat matmul (wq/wk/wv) + bias+rope+store
    /// in one kernel pass (llama `attn_bias_rope_store`). Carries the layer so
    /// the scheduler can resolve the persistent K/V regions (kv_pair).
    FusedQKV { layer: usize },
    /// decode (nt==1) fused FFN gate+up: one concat matmul (ffn_gate|ffn_up,
    /// loader-registered `blk.{i}.ffn_gu`) whose output buffer carries gate
    /// (rows 0..nf) and up (nf..2*nf); a single swiglu pass (silu(gate)*up,
    /// llama `ggml_swiglu_split`) runs in place. The following down matmul
    /// reads rows 0..nf (od = nf; nt==1 makes the concat layout safe).
    FusedFFN,
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
    FusedQkv(FusedQkvMeta),
    FusedFfn(FusedFfnMeta),
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

/// decode QKV fusion metadata: concat weight (wq|wk|wv rows), the three
/// biases, and the rope/store parameters. `qkv_weight` is the loader-registered
/// concat tensor (`blk.{i}.attn_qkv`); `od_total = nqt + 2*nkt`.
#[derive(Debug, Clone, PartialEq)]
pub struct FusedQkvMeta {
    pub qkv_weight: String,
    pub bias_q: Option<String>,
    pub bias_k: Option<String>,
    pub bias_v: Option<String>,
    pub weight_ttype: TensorType,
    /// Weight dims (GGUF convention): `in_dim = shape[0]`, concat `out_dim =
    /// shape[1]` sum = nqt + nkt + nkt.
    pub in_dim: usize,
    pub nqt: usize,
    pub nkt: usize,
    pub hd: usize,
    pub nh: usize,
    pub nk: usize,
    pub freq_base: f32,
    pub freq_scale: f32,
    pub rope_style: RopeStyle,
    /// KV region element count (nkt * n_ctx) — for the allocator's ensure_kv.
    pub kv_elems: usize,
}

/// decode FFN gate+up fusion metadata: the loader-registered concat weight
/// `blk.{i}.ffn_gu` (ffn_gate|ffn_up rows), the shared input dim, and the FFN
/// output dim nf (gate rows 0..nf, up rows nf..2*nf in the output buffer).
#[derive(Debug, Clone, PartialEq)]
pub struct FusedFfnMeta {
    pub gu_weight: String,
    pub weight_ttype: TensorType,
    /// Weight dims (GGUF convention): `in_dim = shape[0]` (== n_embd), the
    /// concat output has `2*nf` rows (`nf = shape[1]` of either weight).
    pub in_dim: usize,
    pub nf: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EmbedMeta {
    pub vocab_size: usize,
    pub weight_name: String,
    pub weight_ttype: TensorType,
}
