// Qwen2 forward pass — batch-aware, with shared quantization
// Translated from: llama.cpp/src/models/qwen2.cpp (build_arch_graph)
// Based on minfer2/src/models/qwen2/forward.rs

use crate::block::{self, BlockQ8_0};
use crate::cache::KVCache;
use crate::tensor::{Tensor, TensorType};

use super::loader::HParams;
use super::LayerWeights;

// ============================================================
// Public API: forward a batch of tokens
// ============================================================

/// Run the Qwen2 transformer for a batch of tokens.
/// tokens: token IDs
/// positions: position indices (one per token)
/// Returns: logits [n_tokens, n_vocab]
pub fn forward(
    model: &super::Qwen2Model,
    token_ids: &[u32],
    positions: &[usize],
    kv_cache: &mut KVCache,
) -> Vec<f32> {
    let hp = &model.hparams;
    let n_tokens = token_ids.len();
    let n_embd = hp.n_embd as usize;
    let n_head = hp.n_head as usize;
    let n_head_kv = hp.n_head_kv as usize;
    let n_embd_head = hp.n_embd_head() as usize;
    let n_vocab = hp.n_vocab as usize;
    let n_gqa = n_head / n_head_kv;

    // 1. Token embedding lookup
    let mut hidden = embed_tokens(token_ids, model.tok_embd.as_ref().unwrap(), hp);

    // 2. Per-layer processing
    for il in 0..model.n_layer() {
        let layer = &model.layers[il];

        // 2a. Attention RMSNorm
        let normed = rms_norm(&hidden, hp.f_norm_rms_eps, n_tokens, n_embd);
        let normed = apply_weight_owned(normed, &layer.attn_norm, n_tokens, n_embd);

        // 2b. Quantize normed ONCE, reuse for Q/K/V (minfer2 pattern)
        let normed_q8 = quantize_activation(&normed, n_tokens, n_embd);

        // 2c. Q/K/V projections (Q4_0 × Q8_0 + F32 bias)
        let mut q = linear_q4_b(layer.wq.as_ref().unwrap(), &normed_q8, &layer.bq, n_embd, n_embd, n_tokens);
        let mut k = linear_q4_b(layer.wk.as_ref().unwrap(), &normed_q8, &layer.bk, n_head_kv * n_embd_head, n_embd, n_tokens);
        let v = linear_q4_b(layer.wv.as_ref().unwrap(), &normed_q8, &layer.bv, n_head_kv * n_embd_head, n_embd, n_tokens);

        // 2d. RoPE on Q and K
        apply_rope(&mut q, positions, n_head, n_embd_head, hp.rope_freq_base);
        let mut k_rope = k.clone();
        apply_rope(&mut k_rope, positions, n_head_kv, n_embd_head, hp.rope_freq_base);

        // 2e. KV cache: store
        kv_cache.layers[il].store_multi(positions, &k_rope, &v);

        // 2f. Attention
        let n_kv = kv_cache.layers[il].size;
        let k_all = &kv_cache.layers[il].k[..n_kv * n_head_kv * n_embd_head];
        let v_all = &kv_cache.layers[il].v[..n_kv * n_head_kv * n_embd_head];
        let attn_out = gqa_attention_batch(&q, k_all, v_all, positions, n_tokens, n_kv,
            n_head, n_head_kv, n_embd_head);

        // 2g. Output projection: quantize attn_out, then Q4_0 x Q8_0
        let attn_q8 = quantize_activation(&attn_out, n_tokens, n_embd);
        let proj_out = linear_q4(layer.wo.as_ref().unwrap(), &attn_q8, n_embd, n_embd, n_tokens);

        // 2h. Residual
        for i in 0..hidden.len() {
            hidden[i] += proj_out[i];
        }

        // 2i. FFN RMSNorm
        let ffn_normed = rms_norm(&hidden, hp.f_norm_rms_eps, n_tokens, n_embd);
        let ffn_normed = apply_weight_owned(ffn_normed, &layer.ffn_norm, n_tokens, n_embd);

        // 2j. SwiGLU FFN (shared quantize for gate/up)
        let ffn_q8 = quantize_activation(&ffn_normed, n_tokens, n_embd);
        let ffn_out = swiglu_ffn(ffn_q8, layer, n_tokens, n_embd, hp.n_ff as usize);

        // 2k. FFN residual
        for i in 0..hidden.len() {
            hidden[i] += ffn_out[i];
        }
    }

    // 3. Final RMS norm
    let final_normed = rms_norm(&hidden, hp.f_norm_rms_eps, n_tokens, n_embd);
    let final_normed = apply_weight_owned(final_normed, &model.output_norm, n_tokens, n_embd);

    // 4. LM head: quantize once, then Q8_0/Q4_0 × Q8_0
    if let Some(output) = &model.output {
        let final_q8 = quantize_activation(&final_normed, n_tokens, n_embd);
        match output.ttype {
            TensorType::Q8_0 => linear_q8(output, &final_q8, n_vocab, n_embd, n_tokens),
            TensorType::Q4_0 => linear_q4(output, &final_q8, n_vocab, n_embd, n_tokens),
            _ => unreachable!("unsupported LM head type: {:?}", output.ttype),
        }
    } else {
        vec![0.0f32; n_tokens * n_vocab]
    }
}

// ============================================================
// Quantization helper: f32 → Q8_0 blocks, shareable across projections
// ============================================================

/// Quantize a [n_tokens, dim] f32 matrix to a contiguous `Vec<BlockQ8_0>`.
/// Each row produces `dim/32` blocks. Returns `[n_tokens * nb]` blocks.
fn quantize_activation(x: &[f32], n_tokens: usize, dim: usize) -> Vec<BlockQ8_0> {
    crate::avx2::quantize_row_q8_0_batch(x, n_tokens, dim)
}

// ============================================================
// Linear projections (pre-quantized Q8_0 activation, no internal quantize)
// ============================================================

/// Q4_0 weight × pre-quantized Q8_0 activation, no bias.
/// weight must be TensorType::Q4_0, x_q8 must be [n_tokens, in_dim/32] blocks.
fn linear_q4(weight: &Tensor, x_q8: &[BlockQ8_0], out_dim: usize, in_dim: usize, n_tokens: usize) -> Vec<f32> {
    let nb = in_dim / 32;
    let blocks = weight.data_q4_0();
    let mut out = vec![0.0f32; n_tokens * out_dim];

    for o in 0..out_dim {
        let w_row = &blocks[o * nb..(o + 1) * nb];
        for t in 0..n_tokens {
            out[t * out_dim + o] = crate::avx2::vec_dot_q4_0_q8_0(
                in_dim as i32, w_row, &x_q8[t * nb..(t + 1) * nb]);
        }
    }
    out
}

/// Q4_0 weight × pre-quantized Q8_0 activation + F32 bias.
fn linear_q4_b(weight: &Tensor, x_q8: &[BlockQ8_0], bias: &Option<Tensor>,
               out_dim: usize, in_dim: usize, n_tokens: usize) -> Vec<f32> {
    let mut out = linear_q4(weight, x_q8, out_dim, in_dim, n_tokens);
    if let Some(b) = bias {
        let bd = b.data_f32();
        for t in 0..n_tokens {
            let base = t * out_dim;
            for i in 0..out_dim.min(bd.len()) {
                out[base + i] += bd[i];
            }
        }
    }
    out
}

/// Q8_0 weight × pre-quantized Q8_0 activation.
fn linear_q8(weight: &Tensor, x_q8: &[BlockQ8_0], out_dim: usize, in_dim: usize, n_tokens: usize) -> Vec<f32> {
    let nb = in_dim / 32;
    let blocks = weight.data_q8_0();
    let mut out = vec![0.0f32; n_tokens * out_dim];

    for o in 0..out_dim {
        let w_row = &blocks[o * nb..(o + 1) * nb];
        for t in 0..n_tokens {
            out[t * out_dim + o] = crate::avx2::vec_dot_q8_0_q8_0(
                in_dim as i32, w_row, &x_q8[t * nb..(t + 1) * nb]);
        }
    }
    out
}

// ============================================================
// Legacy project_row (f32 fallback, kept for external callers)
// ============================================================

pub fn project_row(
    input: &[f32], output: &mut [f32], weight: &Tensor,
    input_dim: usize, output_dim: usize, n_tokens: usize,
) {
    match weight.ttype {
        TensorType::F32 => {
            let w = weight.data_f32();
            for t in 0..n_tokens {
                let inp = &input[t * input_dim..(t + 1) * input_dim];
                let out = &mut output[t * output_dim..(t + 1) * output_dim];
                for o in 0..output_dim {
                    let mut sum = 0.0f32;
                    let w_row = &w[o * input_dim..(o + 1) * input_dim];
                    for i in 0..input_dim { sum += inp[i] * w_row[i]; }
                    out[o] = sum;
                }
            }
        }
        _ => panic!("project_row: unsupported type {:?}", weight.ttype),
    }
}

// ============================================================
// Embedding lookup
// ============================================================

fn embed_tokens(token_ids: &[u32], tok_embd: &Tensor, hp: &HParams) -> Vec<f32> {
    let n_embd = hp.n_embd as usize;
    let n_tokens = token_ids.len();
    let mut output = vec![0.0f32; n_tokens * n_embd];
    let blk_size = 32usize;
    let n_blocks_per_row = (n_embd + blk_size - 1) / blk_size;
    let blk_bytes = tok_embd.ttype.type_size();
    let is_q8_0 = tok_embd.ttype == TensorType::Q8_0;

    for (t, &token_id) in token_ids.iter().enumerate() {
        let token_idx = token_id as usize;
        let dst_offset = t * n_embd;
        for b in 0..n_blocks_per_row {
            let block_idx = token_idx * n_blocks_per_row + b;
            let off = block_idx * blk_bytes;
            let d = u16::from_le_bytes([tok_embd.data[off], tok_embd.data[off + 1]]);
            let d_f32 = block::fp16_to_f32(d);
            let max_v = core::cmp::min(blk_size, n_embd - b * blk_size);

            if is_q8_0 {
                for j in 0..max_v {
                    let qs = tok_embd.data[off + 2 + j] as i8;
                    output[dst_offset + b * blk_size + j] = qs as f32 * d_f32;
                }
            } else {
                for j in 0..max_v {
                    let nibble = if j % 2 == 0 {
                        tok_embd.data[off + 2 + j / 2] & 0x0F
                    } else {
                        tok_embd.data[off + 2 + j / 2] >> 4
                    };
                    output[dst_offset + b * blk_size + j] = (nibble as i8 - 8) as f32 * d_f32;
                }
            }
        }
    }
    output
}

// ============================================================
// RMSNorm
// ============================================================

fn rms_norm(x: &[f32], eps: f32, n_tokens: usize, dim: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n_tokens * dim];
    for t in 0..n_tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let dst = &mut out[t * dim..(t + 1) * dim];
        let mut sum_sq = 0.0f64;
        for i in 0..dim { sum_sq += (row[i] as f64) * (row[i] as f64); }
        let scale = 1.0 / ((sum_sq / dim as f64) as f32 + eps).sqrt();
        for i in 0..dim { dst[i] = row[i] * scale; }
    }
    out
}

fn apply_weight_owned(mut x: Vec<f32>, w: &Option<Tensor>, n_tokens: usize, dim: usize) -> Vec<f32> {
    if let Some(weight) = w {
        let wd = weight.data_f32();
        for t in 0..n_tokens {
            let base = t * dim;
            for i in 0..dim { x[base + i] *= wd[i]; }
        }
    }
    x
}

// ============================================================
// RoPE (Neox style)
// ============================================================

fn apply_rope(x: &mut [f32], positions: &[usize], n_head: usize, n_embd_head: usize, freq_base: f32) {
    let half = n_embd_head / 2;
    for t in 0..positions.len() {
        let p = positions[t] as f32;
        for h in 0..n_head {
            let base = t * n_head * n_embd_head + h * n_embd_head;
            for i in 0..half {
                let freq = 1.0 / freq_base.powf((2 * i) as f32 / n_embd_head as f32);
                let theta = p * freq;
                let (sin, cos) = theta.sin_cos();
                let i0 = base + i;
                let i1 = base + i + half;
                let x0 = x[i0];
                let x1 = x[i1];
                x[i0] = x0 * cos - x1 * sin;
                x[i1] = x0 * sin + x1 * cos;
            }
        }
    }
}

// ============================================================
// GQA attention (batch version)
// ============================================================

fn gqa_attention_batch(
    q: &[f32], k_all: &[f32], v_all: &[f32],
    positions: &[usize], n_tokens: usize, n_kv: usize,
    n_head: usize, n_head_kv: usize, n_embd_head: usize,
) -> Vec<f32> {
    let n_gqa = n_head / n_head_kv;
    let n_embd = n_head * n_embd_head;
    let scale = 1.0 / (n_embd_head as f32).sqrt();
    let mut out = vec![0.0f32; n_tokens * n_embd];
    let mut scores = vec![0.0f32; n_kv];

    for h in 0..n_head {
        let h_kv = h / n_gqa;
        for t in 0..n_tokens {
            let q_start = t * n_embd + h * n_embd_head;
            let cur_pos = positions[t];
            let valid = (cur_pos + 1).min(n_kv);

            let mut max_score = f32::NEG_INFINITY;
            for kv in 0..valid {
                let k_start = kv * n_head_kv * n_embd_head + h_kv * n_embd_head;
                let mut s = 0.0f32;
                for d in 0..n_embd_head { s += q[q_start + d] * k_all[k_start + d]; }
                s *= scale;
                scores[kv] = s;
                if s > max_score { max_score = s; }
            }
            for kv in valid..n_kv { scores[kv] = f32::NEG_INFINITY; }

            // Softmax
            let mut sum = 0.0f64;
            for s in &mut scores { *s = (*s - max_score).exp(); sum += *s as f64; }
            let inv_sum = (1.0 / sum) as f32;
            for s in &mut scores { *s *= inv_sum; }

            // Weighted sum of V
            let out_start = t * n_embd + h * n_embd_head;
            for d in 0..n_embd_head {
                let mut acc = 0.0f32;
                for kv in 0..n_kv {
                    acc += scores[kv] * v_all[kv * n_head_kv * n_embd_head + h_kv * n_embd_head + d];
                }
                out[out_start + d] = acc;
            }
        }
    }
    out
}

// ============================================================
// SwiGLU FFN (takes pre-quantized input)
// ============================================================

/// SwiGLU FFN with shared quantized activation.
/// x_q8: pre-quantized input [n_tokens, n_embd] as BlockQ8_0 blocks.
fn swiglu_ffn(
    x_q8: Vec<BlockQ8_0>,
    layer: &LayerWeights,
    n_tokens: usize,
    n_embd: usize,
    n_ff: usize,
) -> Vec<f32> {
    // gate = Q4_0(gate) × Q8_0(x)
    let mut gate = linear_q4(layer.ffn_gate.as_ref().unwrap(), &x_q8, n_ff, n_embd, n_tokens);
    // up = Q4_0(up) × Q8_0(x)
    let up = linear_q4(layer.ffn_up.as_ref().unwrap(), &x_q8, n_ff, n_embd, n_tokens);

    // SiLU(x) * up — in-place on gate
    for i in 0..n_tokens * n_ff {
        gate[i] = silu(gate[i]) * up[i];
    }

    // down = Q4_0(down) × Q8_0(gate*up), quantize gate once
    let gate_q8 = quantize_activation(&gate, n_tokens, n_ff);
    linear_q4(layer.ffn_down.as_ref().unwrap(), &gate_q8, n_embd, n_ff, n_tokens)
}

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
