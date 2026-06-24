// Phase 6: Qwen2 Model Structure
// Translated from: llama.cpp/src/models/qwen2.cpp (load_arch_hparams, load_arch_tensors, build_arch_graph)
//   + llama.cpp/src/llama-hparams.h (hyperparameters)
//   + llama.cpp/src/llama-arch.h (architecture enumeration)
// Strict 1:1 translation — no extra code, no design changes

use crate::tensor::{Tensor, TensorType};

// === Architecture types (llama-arch.h) ===

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Qwen2,
    Unknown,
}

// === Hyperparameters (llama-hparams.h) ===

#[derive(Debug, Clone)]
pub struct HParams {
    pub n_embd: i64,        // embedding / hidden dimension
    pub n_head: i64,        // number of query heads
    pub n_head_kv: i64,     // number of key/value heads (GQA)
    pub n_layer: i64,       // number of transformer layers
    pub n_ff: i64,          // feed-forward hidden dimension
    pub n_vocab: i64,       // vocabulary size

    pub max_seq_len: i64,   // maximum sequence length

    pub f_norm_rms_eps: f32, // RMS norm epsilon

    pub rope_freq_base: f32,    // RoPE base frequency
    pub rope_freq_scale: f32,   // RoPE frequency scaling (usually 1.0)
}

impl HParams {
    pub fn new() -> Self {
        Self {
            n_embd: 0,
            n_head: 0,
            n_head_kv: 0,
            n_layer: 0,
            n_ff: 0,
            n_vocab: 0,
            max_seq_len: 0,
            f_norm_rms_eps: 1e-6,
            rope_freq_base: 10000.0,
            rope_freq_scale: 1.0,
        }
    }

    /// Head dimension (shared for Q, K, V in Qwen2)
    pub fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }

    /// n_rot: number of rotary dimensions (full head dim in Qwen2)
    pub fn n_rot(&self) -> i64 {
        self.n_embd_head()
    }
}

// === Per-layer weights (qwen2.cpp lines 33-46) ===

/// Weights for one transformer layer
#[derive(Clone)]
pub struct LayerWeights {
    // Attention
    pub attn_norm: Option<Tensor>,   // RMSNorm weight, shape [n_embd]

    pub wq: Option<Tensor>,          // Q projection, shape [n_embd, n_embd]
    pub bq: Option<Tensor>,          // Q bias, shape [n_embd]
    pub wk: Option<Tensor>,          // K projection, shape [n_embd, n_embd_kv]
    pub bk: Option<Tensor>,          // K bias, shape [n_embd_kv]
    pub wv: Option<Tensor>,          // V projection, shape [n_embd, n_embd_kv]
    pub bv: Option<Tensor>,          // V bias, shape [n_embd_kv]
    pub wo: Option<Tensor>,          // Output projection, shape [n_embd, n_embd]

    // FFN (SwiGLU: gate, up, down)
    pub ffn_norm: Option<Tensor>,    // RMSNorm weight, shape [n_embd]

    pub ffn_gate: Option<Tensor>,    // Gate projection, shape [n_embd, n_ff]
    pub ffn_up: Option<Tensor>,      // Up projection, shape [n_embd, n_ff]
    pub ffn_down: Option<Tensor>,    // Down projection, shape [n_ff, n_embd]
}

impl LayerWeights {
    pub fn new() -> Self {
        Self {
            attn_norm: None,
            wq: None,
            bq: None,
            wk: None,
            bk: None,
            wv: None,
            bv: None,
            wo: None,
            ffn_norm: None,
            ffn_gate: None,
            ffn_up: None,
            ffn_down: None,
        }
    }
}

// === Full model (qwen2.cpp) ===

#[derive(Clone)]
pub struct Model {
    pub hparams: HParams,
    pub arch: ModelType,

    // Embeddings
    pub tok_embd: Option<Tensor>,    // token embedding, shape [n_embd, n_vocab]

    // Output
    pub output_norm: Option<Tensor>, // final RMSNorm weight, shape [n_embd]
    pub output: Option<Tensor>,      // LM head weight, shape [n_embd, n_vocab]
    pub output_b: Option<Tensor>,    // LM head bias, shape [n_vocab] (usually None)

    // Per-layer weights
    pub layers: Vec<LayerWeights>,
}

impl Model {
    /// Create a new model from hyperparameters
    pub fn new(hparams: HParams) -> Self {
        let n_layer = hparams.n_layer as usize;
        Self {
            hparams,
            arch: ModelType::Qwen2,
            tok_embd: None,
            output_norm: None,
            output: None,
            output_b: None,
            layers: vec![LayerWeights::new(); n_layer],
        }
    }

    /// Set token embedding weight (from loaded GGUF tensor)
    pub fn set_tok_embd(&mut self, tensor: Tensor) {
        self.tok_embd = Some(tensor);
    }

    /// Set output norm weight
    pub fn set_output_norm(&mut self, tensor: Tensor) {
        self.output_norm = Some(tensor);
    }

    /// Set output / LM head weight
    pub fn set_output(&mut self, tensor: Tensor) {
        self.output = Some(tensor);
    }

    /// Set output bias
    pub fn set_output_b(&mut self, tensor: Tensor) {
        self.output_b = Some(tensor);
    }

    /// Access a specific layer
    pub fn layer(&self, idx: usize) -> &LayerWeights {
        &self.layers[idx]
    }

    pub fn layer_mut(&mut self, idx: usize) -> &mut LayerWeights {
        &mut self.layers[idx]
    }

    /// Number of layers
    pub fn n_layer(&self) -> usize {
        self.layers.len()
    }
}

// === GGUF tensor name mapping (for lookup by name) ===

/// Tensor name constants matching GGUF naming convention
pub mod tensor_names {
    pub const TOKEN_EMBD: &str = "token_embd.weight";
    pub const OUTPUT_NORM: &str = "output_norm.weight";
    pub const OUTPUT: &str = "output.weight";
    pub const OUTPUT_BIAS: &str = "output.bias";

    pub fn attn_norm(i: usize) -> String {
        format!("blk.{}.attn_norm.weight", i)
    }

    pub fn attn_q(i: usize) -> String {
        format!("blk.{}.attn_q.weight", i)
    }

    pub fn attn_q_bias(i: usize) -> String {
        format!("blk.{}.attn_q.bias", i)
    }

    pub fn attn_k(i: usize) -> String {
        format!("blk.{}.attn_k.weight", i)
    }

    pub fn attn_k_bias(i: usize) -> String {
        format!("blk.{}.attn_k.bias", i)
    }

    pub fn attn_v(i: usize) -> String {
        format!("blk.{}.attn_v.weight", i)
    }

    pub fn attn_v_bias(i: usize) -> String {
        format!("blk.{}.attn_v.bias", i)
    }

    pub fn attn_out(i: usize) -> String {
        format!("blk.{}.attn_output.weight", i)
    }

    pub fn ffn_norm(i: usize) -> String {
        format!("blk.{}.ffn_norm.weight", i)
    }

    pub fn ffn_gate(i: usize) -> String {
        format!("blk.{}.ffn_gate.weight", i)
    }

    pub fn ffn_up(i: usize) -> String {
        format!("blk.{}.ffn_up.weight", i)
    }

    pub fn ffn_down(i: usize) -> String {
        format!("blk.{}.ffn_down.weight", i)
    }
}

// === GGUF KV helper: extract hparams from GGUF context ===

/// Extract HParams from a parsed GGUF context
pub fn hparams_from_gguf(ctx: &crate::gguf::GgufContext) -> Option<HParams> {
    let get_i64 = |key: &str| -> Option<i64> {
        ctx.get_key_val_i64(key)
    };

    let get_f32 = |key: &str| -> Option<f32> {
        ctx.get_key_val_f32(key)
    };

    // Get vocab size from tokenizer.ggml.tokens array length
    let n_vocab: i64 = {
        let mut found = 0i64;
        for kv in &ctx.kv {
            if kv.key == "tokenizer.ggml.tokens" && kv.is_array {
                found = kv.get_ne() as i64;
                break;
            }
        }
        found
    };
    if n_vocab == 0 {
        eprintln!("Warning: could not determine vocabulary size from GGUF");
    }

    // Verify architecture is Qwen2
    let arch = ctx.get_key_val_str("general.architecture")?;
    if arch != "qwen2" {
        eprintln!("Warning: expected architecture 'qwen2', got '{}'", arch);
    }

    let hparams = HParams {
        n_embd:       get_i64("qwen2.embedding_length")?,
        n_head:       get_i64("qwen2.attention.head_count")?,
        n_head_kv:    get_i64("qwen2.attention.head_count_kv")?,
        n_layer:      get_i64("qwen2.block_count")?,
        n_ff:         get_i64("qwen2.feed_forward_length")?,
        n_vocab,
        max_seq_len:  get_i64("qwen2.context_length").unwrap_or(32768),

        f_norm_rms_eps: get_f32("qwen2.attention.layer_norm_rms_epsilon").unwrap_or(1e-6),

        rope_freq_base:  get_f32("qwen2.rope.freq_base").unwrap_or(10000.0),
        rope_freq_scale: get_f32("qwen2.rope.frequency_scale").unwrap_or(1.0),
    };

    Some(hparams)
}

/// Extract HParams using a simpler approach — read from GGUF KV pairs by key
impl crate::gguf::GgufContext {
    /// Get a string KV value by key
    pub fn get_key_val_str(&self, key: &str) -> Option<String> {
        for kv in &self.kv {
            if kv.key == key {
                return kv.get_string();
            }
        }
        None
    }

    /// Get an i64 KV value by key
    pub fn get_key_val_i64(&self, key: &str) -> Option<i64> {
        for kv in &self.kv {
            if kv.key == key {
                return kv.get_i64();
            }
        }
        None
    }

    /// Get an f32 KV value by key
    pub fn get_key_val_f32(&self, key: &str) -> Option<f32> {
        for kv in &self.kv {
            if kv.key == key {
                return kv.get_f32();
            }
        }
        None
    }
}

impl crate::gguf::GgufKv {
    pub fn get_string(&self) -> Option<String> {
        if self.type_ == crate::gguf::GgufType::String && !self.data_string.is_empty() {
            Some(self.data_string[0].clone())
        } else {
            None
        }
    }

    pub fn get_i64(&self) -> Option<i64> {
        match self.type_ {
            crate::gguf::GgufType::Int64 => Some(self.get_val_i64(0)),
            crate::gguf::GgufType::Uint32 => Some(self.get_val_u32(0) as i64),
            crate::gguf::GgufType::Int32 => Some(self.get_val_i32(0) as i64),
            crate::gguf::GgufType::Uint64 => Some(self.get_val_u64(0) as i64),
            _ => None,
        }
    }

    pub fn get_f32(&self) -> Option<f32> {
        match self.type_ {
            crate::gguf::GgufType::Float32 => Some(self.get_val_f32(0)),
            crate::gguf::GgufType::Float64 => Some(self.get_val_f64(0) as f32),
            _ => None,
        }
    }
}
