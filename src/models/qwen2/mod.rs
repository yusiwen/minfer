// Qwen2 model definition + ModelDef impl
// Translated from: llama.cpp/src/models/qwen2.cpp

pub mod forward;
pub mod loader;

use crate::cache::KVCache;
use crate::models::{ModelDef, SpecialTokens};
use crate::tensor::Tensor;

pub use loader::{HParams, LayerWeights};

/// Qwen2 model with all weights loaded.
#[derive(Clone)]
pub struct Qwen2Model {
    pub hparams: HParams,
    pub tok_embd: Option<Tensor>,
    pub output_norm: Option<Tensor>,
    pub output: Option<Tensor>,
    pub output_b: Option<Tensor>,
    pub layers: Vec<LayerWeights>,
}

impl Qwen2Model {
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
}

impl ModelDef for Qwen2Model {
    fn forward(&self, tokens: &[u32], positions: &[usize], kv: &mut KVCache) -> Vec<f32> {
        forward::forward(self, tokens, positions, kv)
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
    fn n_vocab(&self) -> usize { self.hparams.n_vocab as usize }
}

/// Simple ChatML formatting.
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
    pub fn attn_q_bias(i: usize) -> String { format!("blk.{}.attn_q.bias", i) }
    pub fn attn_k(i: usize) -> String { format!("blk.{}.attn_k.weight", i) }
    pub fn attn_k_bias(i: usize) -> String { format!("blk.{}.attn_k.bias", i) }
    pub fn attn_v(i: usize) -> String { format!("blk.{}.attn_v.weight", i) }
    pub fn attn_v_bias(i: usize) -> String { format!("blk.{}.attn_v.bias", i) }
    pub fn attn_out(i: usize) -> String { format!("blk.{}.attn_output.weight", i) }
    pub fn ffn_norm(i: usize) -> String { format!("blk.{}.ffn_norm.weight", i) }
    pub fn ffn_gate(i: usize) -> String { format!("blk.{}.ffn_gate.weight", i) }
    pub fn ffn_up(i: usize) -> String { format!("blk.{}.ffn_up.weight", i) }
    pub fn ffn_down(i: usize) -> String { format!("blk.{}.ffn_down.weight", i) }
}
