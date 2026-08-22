// Qwen3 model definition + ModelDef impl (dense architecture).
//
// Qwen3 dense differs from Qwen2.5 in exactly two ways (see
// docs/QWEN3-SUPPORT-PLAN.md §2):
//   1. head_dim is decoupled from embedding_length / n_head (128 vs 64 for the
//      0.6B) — read from `qwen3.attention.key_length` / `value_length`.
//   2. per-head Q/K RMSNorm (`blk.{i}.attn_q_norm` / `attn_k_norm`) applied
//      after the projections, before RoPE.
// Everything else (RMSNorm pre-norm, SwiGLU FFN, no biases, GQA attention,
// NeoX non-interleaved RoPE, tied lm_head) is identical to Qwen2 and reuses
// the same graph primitives.

pub mod graph;
pub mod loader;

use crate::cache::KVCache;
use crate::models::{ModelDef, SpecialTokens};
use crate::tensor::Tensor;

pub use loader::{HParams, LayerWeights};

/// Qwen3 model with all weights loaded.
#[derive(Clone)]
pub struct Qwen3Model {
    pub hparams: HParams,
    pub tok_embd: Option<Tensor>,
    pub output_norm: Option<Tensor>,
    pub output: Option<Tensor>,
    pub output_b: Option<Tensor>,
    pub layers: Vec<LayerWeights>,
}

impl Qwen3Model {
    /// Inherent accessor — callers use the trait's `n_layer()` (the graph tests
    /// go through `Box<dyn ModelDef>`), so this stays as a concrete-type helper.
    #[allow(dead_code)]
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
}

impl ModelDef for Qwen3Model {
    fn forward(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache, n_out: usize) -> Vec<f32> {
        graph::Qwen3Graph::forward(self, tokens, positions, kv, n_out)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn build_graph(&self, params: &crate::graph::params::GraphParams) -> crate::graph::ComputeGraph {
        graph::Qwen3Graph::build(self, params)
    }

    fn forward_graph(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache, n_out: usize) -> Vec<f32> {
        graph::Qwen3Graph::forward(self, tokens, positions, kv, n_out)
    }

    fn forward_graph_cached(
        &self,
        tokens: &[u32],
        positions: &[usize],
        n_out: usize,
        n_ctx: usize,
        cache: &mut crate::graph::cache::GraphCache,
    ) -> Vec<f32> {
        graph::Qwen3Graph::forward_cached(self, tokens, positions, n_out, n_ctx, cache)
    }

    fn format_chat(&self, messages: &[(String, String)]) -> String {
        format_chatml(messages)
    }

    fn special_tokens(&self) -> SpecialTokens {
        let eos = self.hparams.eos_token_id;
        let im_end = self.hparams.im_end_token_id;
        SpecialTokens { eos, im_end }
    }

    fn n_layer(&self) -> usize { self.hparams.n_layer as usize }
    fn n_head_kv(&self) -> usize { self.hparams.n_head_kv as usize }
    fn n_embd_head(&self) -> usize { self.hparams.n_embd_head() as usize }
    fn n_kv_embd(&self) -> usize { self.hparams.n_kv_embd as usize }
    fn n_vocab(&self) -> usize { self.hparams.n_vocab as usize }
    fn rope_style(&self) -> crate::vec_ops::RopeStyle { self.hparams.rope_style }
}

/// Simple ChatML formatting (fallback; template.rs renders the model's own
/// GGUF chat template — the Qwen3 template with `<think>` tags works there).
#[allow(dead_code)]
fn format_chatml(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        prompt.push_str(&format!(
            "<|im_start|>{}\n{}<|im_end|>\n", role, content
        ));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}

// ============================================================
// GGUF tensor name mapping
// ============================================================

pub mod tensor_names {
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    pub const OUTPUT: &str = "output.weight";
    pub const OUTPUT_BIAS: &str = "output.bias";

    pub fn attn_norm(i: usize) -> String { format!("blk.{}.attn_norm.weight", i) }
    pub fn attn_q(i: usize) -> String { format!("blk.{}.attn_q.weight", i) }
    pub fn attn_k(i: usize) -> String { format!("blk.{}.attn_k.weight", i) }
    pub fn attn_v(i: usize) -> String { format!("blk.{}.attn_v.weight", i) }
    pub fn attn_out(i: usize) -> String { format!("blk.{}.attn_output.weight", i) }
    /// Per-head Q RMSNorm weight (Qwen3): shape [n_embd_head].
    pub fn attn_q_norm(i: usize) -> String { format!("blk.{}.attn_q_norm.weight", i) }
    /// Per-head K RMSNorm weight (Qwen3): shape [n_embd_head].
    pub fn attn_k_norm(i: usize) -> String { format!("blk.{}.attn_k_norm.weight", i) }
    pub fn ffn_norm(i: usize) -> String { format!("blk.{}.ffn_norm.weight", i) }
    pub fn ffn_gate(i: usize) -> String { format!("blk.{}.ffn_gate.weight", i) }
    pub fn ffn_up(i: usize) -> String { format!("blk.{}.ffn_up.weight", i) }
    pub fn ffn_down(i: usize) -> String { format!("blk.{}.ffn_down.weight", i) }
}
