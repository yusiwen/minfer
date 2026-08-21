// Qwen2 GGUF tensor loader

use crate::gguf::GgufContext;
use crate::tensor::{Tensor, TensorType};
use crate::vec_ops::RopeStyle;

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
    pub rope_style: RopeStyle,
    /// Actual KV embedding dimension (ne[1] of K weight).
    /// May differ from `n_head_kv * n_embd_head()` when the model uses
    /// separate KV head dimensions (e.g. Qwen2.5-0.5B: kv_dim=128 vs n_embd=896).
    pub n_kv_embd: i64,
}

impl HParams {
    pub fn n_embd_head(&self) -> i64 {
        self.n_embd / self.n_head
    }

    pub fn attention_scale(&self) -> f32 {
        1.0 / (self.n_embd_head() as f32).sqrt()
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
        rope_style: RopeStyle::NonInterleaved,
        n_kv_embd: n_head_kv * (n_embd / n_head), // default, updated from K weight shape below
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

fn load_tensor(ctx: &GgufContext, raw: &'static [u8], ti: &crate::gguf::GgufTensorInfo) -> Tensor {
    let ttype = TensorType::from_ggml_type(ti.type_);
    let mut shape = [1i64; 4];
    for j in 0..4 { shape[j] = ti.ne[j]; }
    let off = ctx.offset + ti.offset as usize;
    // Use GGML type for byte-size calculation — always correct regardless of TensorType mapping
    let ts = ti.type_.type_size();
    let bs = ti.type_.blck_size() as usize;
    let n = (shape[0] * shape[1] * shape[2] * shape[3]) as usize;
    let nbytes = (n / bs) * ts;
    // Borrow the tensor bytes straight from the mmap'd part file (zero-copy —
    // the file pages are shared with the CPU and GPU instead of a per-tensor copy).
    let src = &raw[off..off + nbytes];

    let mut strides = [0usize; 4];
    strides[0] = ts;
    strides[1] = strides[0] * (shape[0] / bs as i64) as usize;
    for j in 2..4 {
        strides[j] = strides[j - 1] * shape[j - 1] as usize;
    }

    let mut tensor = Tensor::from_data_borrowed_with_strides(ttype, &shape, &strides, src);
    tensor.set_name(&ti.name);

    // Register weight tensors with GPU backends.
    #[cfg(target_os = "macos")]
    if let Some(mps) = crate::metal::MpsState::get() {
        if matches!(ttype, TensorType::Q4_0 | TensorType::Q4_1 | TensorType::Q4_K | TensorType::Q5_0 | TensorType::Q5_1 | TensorType::Q5_K | TensorType::Q6_K | TensorType::Q8_0) {
            mps.register_weight(&ti.name, tensor.data());
        } else if ttype == TensorType::F32 {
            mps.register_weight(&ti.name, tensor.data());
        }
    }
    #[cfg(feature = "cuda")]
    if let Some(cuda) = crate::cuda::CudaState::get() {
        if matches!(ttype, TensorType::Q4_0 | TensorType::Q4_1 | TensorType::Q4_K | TensorType::Q5_0 | TensorType::Q5_1 | TensorType::Q6_K | TensorType::Q8_0) {
            cuda.register_weight(&ti.name, tensor.data());
        } else if ttype == TensorType::F32 {
            cuda.register_weight(&ti.name, tensor.data());
        }
    }

    tensor
}

// ============================================================
// Architecture loader
// ============================================================

pub fn load(model: &crate::gguf::GgufModel) -> Option<super::Qwen2Model> {
    let ctx = &model.parts[0].ctx;
    let mut hparams = hparams_from_gguf(ctx)?;

    // Zero-copy weight registration: tell the Metal backend about each mmap'd
    // part (page-aligned base) BEFORE any weight is registered, so weights are
    // wrapped as (buffer, offset) into the part buffer instead of being copied.
    #[cfg(target_os = "macos")]
    if let Some(mps) = crate::metal::MpsState::get() {
        for part in &model.parts {
            mps.register_part(part.data);
        }
    }

    // Merged tensor index across all split parts (llama.cpp weights_map): each
    // tensor lives in the part that lists it, read from that part's own data.
    let mut tensor_map = std::collections::HashMap::<String, (usize, &crate::gguf::GgufTensorInfo)>::new();
    for (pi, part) in model.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            tensor_map.insert(ti.name.clone(), (pi, ti));
        }
    }
    let load_one = |n: &str| -> Option<Tensor> {
        tensor_map.get(n).map(|(pi, ti)| {
            let part = &model.parts[*pi];
            load_tensor(&part.ctx, &part.data, ti)
        })
    };
    let load_ti = |(pi, ti): &(usize, &crate::gguf::GgufTensorInfo)| -> Tensor {
        let part = &model.parts[*pi];
        load_tensor(&part.ctx, &part.data, ti)
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
            layer.attn_norm = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q(i)) {
            layer.wq = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_q_bias(i)) {
            layer.bq = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k(i)) {
            layer.wk = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_k_bias(i)) {
            layer.bk = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v(i)) {
            layer.wv = Some(load_ti(ti));
        }

        // Fused QKV projection (nt==1 decode): concatenate Wq/Wk/Wv along the
        // output (row) dim into one GPU buffer so a single matmul produces
        // q+k+v. Only registered when the three weights share a matmul
        // (same quant type + same input dim); layer_gpu falls back to three
        // separate matmuls otherwise.
        #[cfg(target_os = "macos")]
        if let Some(mps) = crate::metal::MpsState::get() {
            if let (Some(wq), Some(wk), Some(wv)) = (&layer.wq, &layer.wk, &layer.wv) {
                if let Some(data) = crate::metal::concat_rows(&[wq, wk, wv]) {
                    mps.register_weight(&format!("blk.{i}.attn_qkv"), &data);
                }
            }
        }
        if let Some(ti) = tensor_map.get(&tn::attn_v_bias(i)) {
            layer.bv = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::attn_out(i)) {
            layer.wo = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_norm(i)) {
            layer.ffn_norm = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_gate(i)) {
            layer.ffn_gate = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_up(i)) {
            layer.ffn_up = Some(load_ti(ti));
        }
        if let Some(ti) = tensor_map.get(&tn::ffn_down(i)) {
            layer.ffn_down = Some(load_ti(ti));
        }

        // Fused FFN gate+up (nt==1 decode): one matmul produces both gate and
        // up from a concatenated weight (same pattern as attn_qkv).
        #[cfg(target_os = "macos")]
        if let Some(mps) = crate::metal::MpsState::get() {
            if let (Some(fg), Some(fu)) = (&layer.ffn_gate, &layer.ffn_up) {
                if let Some(data) = crate::metal::concat_rows(&[fg, fu]) {
                    mps.register_weight(&format!("blk.{i}.ffn_gu"), &data);
                }
            }
        }
        layers.push(layer);
    }

    // Override n_kv_embd from layer 0 K weight's actual output dimension
    if let Some((_, ti)) = tensor_map.get(&tn::attn_k(0)) {
        hparams.n_kv_embd = ti.ne[1];
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
