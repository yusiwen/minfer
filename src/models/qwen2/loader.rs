// Qwen2 GGUF tensor loader
// Translated from: llama.cpp/src/models/qwen2.cpp (load_arch_tensors)

use crate::gguf::GgufContext;
use crate::tensor::{Tensor, TensorType};

use super::tensor_names as tn;

/// Qwen2 hyperparameters — read from GGUF metadata.
#[derive(Debug, Clone)]
pub struct HParams {
    pub n_embd: i64,
    pub n_head: i64,
    pub n_head_kv: i64,
    pub n_layer: i64,
    pub n_ff: i64,
    pub n_vocab: i64,
    pub max_seq_len: i64,
    pub f_norm_rms_eps: f32,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub eos_token_id: u32,
    pub im_end_token_id: Option<u32>,
}

impl HParams {
    pub fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }
}

/// Per-layer weights for Qwen2.
#[derive(Clone)]
pub struct LayerWeights {
    pub attn_norm: Option<Tensor>,
    pub wq: Option<Tensor>,
    pub bq: Option<Tensor>,
    pub wk: Option<Tensor>,
    pub bk: Option<Tensor>,
    pub wv: Option<Tensor>,
    pub bv: Option<Tensor>,
    pub wo: Option<Tensor>,
    pub ffn_norm: Option<Tensor>,
    pub ffn_gate: Option<Tensor>,
    pub ffn_up: Option<Tensor>,
    pub ffn_down: Option<Tensor>,
}

impl LayerWeights {
    pub fn new() -> Self {
        Self {
            attn_norm: None,
            wq: None, bq: None,
            wk: None, bk: None,
            wv: None, bv: None,
            wo: None,
            ffn_norm: None,
            ffn_gate: None, ffn_up: None, ffn_down: None,
        }
    }
}

// ============================================================
// HParams extraction from GGUF
// ============================================================

fn get_i64(ctx: &GgufContext, key: &str) -> Option<i64> {
    ctx.get_key_val_i64(key)
}
fn get_f32(ctx: &GgufContext, key: &str) -> Option<f32> {
    ctx.get_key_val_f32(key)
}
fn get_u32(ctx: &GgufContext, key: &str) -> Option<u32> {
    for kv in &ctx.kv {
        if kv.key == key {
            return match kv.type_ {
                crate::gguf::GgufType::Uint32 => Some(kv.get_val_u32(0)),
                crate::gguf::GgufType::Int32 => Some(kv.get_val_i32(0) as u32),
                crate::gguf::GgufType::Uint64 => Some(kv.get_val_u64(0) as u32),
                crate::gguf::GgufType::Int64 => Some(kv.get_val_i64(0) as u32),
                _ => None,
            };
        }
    }
    None
}

pub fn hparams_from_gguf(ctx: &GgufContext) -> Option<HParams> {
    let n_vocab = {
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

    // Try qwen2 prefix first, fall back to llama/generic
    let n_embd = get_i64(ctx, "qwen2.embedding_length")
        .or_else(|| get_i64(ctx, "llama.embedding_length"))?;
    let n_head = get_i64(ctx, "qwen2.attention.head_count")
        .or_else(|| get_i64(ctx, "llama.attention.head_count"))?;
    let n_head_kv = get_i64(ctx, "qwen2.attention.head_count_kv")
        .or_else(|| get_i64(ctx, "llama.attention.head_count_kv"))
        .unwrap_or(n_head);
    let n_layer = get_i64(ctx, "qwen2.block_count")
        .or_else(|| get_i64(ctx, "llama.block_count"))?;
    let n_ff = get_i64(ctx, "qwen2.feed_forward_length")
        .or_else(|| get_i64(ctx, "llama.feed_forward_length"))?;

    let eos = get_u32(ctx, "tokenizer.ggml.eos_token_id").unwrap_or(0);
    let im_end = find_token_id(ctx, "<|im_end|>").or(Some(eos));

    Some(HParams {
        n_embd, n_head, n_head_kv, n_layer, n_ff, n_vocab,
        max_seq_len: get_i64(ctx, "qwen2.context_length")
            .or_else(|| get_i64(ctx, "llama.context_length"))
            .unwrap_or(32768),
        f_norm_rms_eps: get_f32(ctx, "qwen2.attention.layer_norm_rms_epsilon")
            .or_else(|| get_f32(ctx, "llama.attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-6),
        rope_freq_base: get_f32(ctx, "qwen2.rope.freq_base")
            .or_else(|| get_f32(ctx, "llama.rope.freq_base"))
            .unwrap_or(10000.0),
        rope_freq_scale: get_f32(ctx, "qwen2.rope.frequency_scale")
            .or_else(|| get_f32(ctx, "llama.rope.frequency_scale"))
            .unwrap_or(1.0),
        eos_token_id: eos,
        im_end_token_id: im_end,
    })
}

fn find_token_id(ctx: &GgufContext, target: &str) -> Option<u32> {
    for kv in &ctx.kv {
        if kv.key == "tokenizer.ggml.tokens" && kv.is_array {
            for i in 0..kv.get_ne() {
                if kv.get_val_str(i) == target {
                    return Some(i as u32);
                }
            }
        }
    }
    None
}

// ============================================================
// Tensor loading
// ============================================================

fn load_tensor(ctx: &GgufContext, raw: &[u8], ti: &crate::gguf::GgufTensorInfo) -> Tensor {
    let ttype = TensorType::from_ggml_type(ti.type_);
    let mut shape = [1i64; 4];
    for j in 0..4 { shape[j] = ti.ne[j]; }
    let off = ctx.offset + ti.offset as usize;
    let ts = ttype.type_size();
    let bs = ttype.blck_size() as usize;
    let n = (shape[0] * shape[1] * shape[2] * shape[3]) as usize;
    let nbytes = (n / bs) * ts;
    let src = &raw[off..off + nbytes];

    let mut data = Vec::with_capacity(src.len());
    data.extend_from_slice(src);

    let mut strides = [0usize; 4];
    strides[0] = ts;
    strides[1] = strides[0] * (shape[0] / bs as i64) as usize;
    for j in 2..4 {
        strides[j] = strides[j - 1] * shape[j - 1] as usize;
    }

    let mut tensor = Tensor::from_data_with_strides(ttype, &shape, &strides, data);
    tensor.set_name(&ti.name);

    // Register weight tensors with MPS (Apple Silicon GPU).
    // Quantized weights are used by matmul kernels; f32 norm weights are used by RMSNorm.
    if let Some(mps) = crate::metal::MpsState::get() {
        if matches!(ttype, TensorType::Q4_0 | TensorType::Q4_1 | TensorType::Q4_K | TensorType::Q6_K | TensorType::Q8_0) {
            mps.register_weight(&ti.name, tensor.data());
        } else if ttype == TensorType::F32 {
            mps.register_weight(&ti.name, tensor.data());
        }
    }

    tensor
}

// ============================================================
// Architecture loader
// ============================================================

pub fn load(ctx: &GgufContext, raw: &[u8]) -> Option<super::Qwen2Model> {
    let hparams = hparams_from_gguf(ctx)?;

    let mut tensor_map = std::collections::HashMap::<String, &crate::gguf::GgufTensorInfo>::new();
    for ti in &ctx.info {
        tensor_map.insert(ti.name.clone(), ti);
    }

    let load_one = |n: &str| -> Option<Tensor> {
        tensor_map.get(n).map(|ti| load_tensor(ctx, raw, ti))
    };

    // Token embedding
    let tok_embd = load_one(tn::TOKEN_EMBD)?;

    // Output norm
    let output_norm = load_one(tn::OUTPUT_NORM);

    // Output weight (with weight tying fallback)
    let output = load_one(tn::OUTPUT).unwrap_or_else(|| tok_embd.clone());

    // Output bias (optional)
    let output_b = load_one(tn::OUTPUT_BIAS);

    // Per-layer weights
    let n_layer = hparams.n_layer as usize;
    let mut layers = Vec::with_capacity(n_layer);
    for i in 0..n_layer {
        let mut layer = crate::models::qwen2::loader::LayerWeights::new();

        if let Some(ti) = tensor_map.get(&tn::attn_norm(i)) {
            layer.attn_norm = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q(i)) {
            layer.wq = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q_bias(i)) {
            layer.bq = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k(i)) {
            layer.wk = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k_bias(i)) {
            layer.bk = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v(i)) {
            layer.wv = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v_bias(i)) {
            layer.bv = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_out(i)) {
            layer.wo = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_norm(i)) {
            layer.ffn_norm = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_gate(i)) {
            layer.ffn_gate = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_up(i)) {
            layer.ffn_up = Some(load_tensor(ctx, raw, ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_down(i)) {
            layer.ffn_down = Some(load_tensor(ctx, raw, ti));
        }
        layers.push(layer);
    }

    println!("Loaded: {} layers", n_layer);

    Some(super::Qwen2Model {
        hparams,
        tok_embd: Some(tok_embd),
        output_norm,
        output: Some(output),
        output_b,
        layers,
    })
}
