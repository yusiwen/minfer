// Qwen2 forward pass — batch-aware version
// Translated from: llama.cpp/src/models/qwen2.cpp (build_arch_graph)
// Based on minfer2/src/models/qwen2/forward.rs

use crate::block;
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
    let eps = hp.f_norm_rms_eps;
    let rope_base = hp.rope_freq_base;

    // 1. Token embedding lookup
    let mut hidden = embed_tokens(token_ids, model.tok_embd.as_ref().unwrap(), hp);

    // 2. Per-layer processing
    for il in 0..model.n_layer() {
        let layer = &model.layers[il];

        // 2a. Attention RMS norm
        let mut normed = rms_norm(&hidden, eps, n_tokens, n_embd);
        if let Some(w) = &layer.attn_norm {
            apply_weight(n_tokens, n_embd, &mut normed, w.data_f32());
        }

        // 2b. QKV projections
        let (mut q, k, v) = project_qkv(&normed, layer, hp);

        // 2c. RoPE
        apply_rope(&mut q, positions, n_head, n_embd_head, rope_base);
        let mut k_rope = k.clone();
        apply_rope(&mut k_rope, positions, n_head_kv, n_embd_head, rope_base);

        // 2d. KV cache: store
        kv_cache.layers[il].store_multi(positions, &k_rope, &v);

        // 2e. Attention
        let n_kv = kv_cache.layers[il].size;
        let k_all = &kv_cache.layers[il].k[..n_kv * n_head_kv * n_embd_head];
        let v_all = &kv_cache.layers[il].v[..n_kv * n_head_kv * n_embd_head];
        let attn_out = gqa_attention_batch(&q, k_all, v_all, positions, n_tokens, n_kv,
            n_head, n_head_kv, n_embd_head);

        // 2f. Output projection
        let attn_proj = {
            let mut buf = vec![0.0f32; n_tokens * n_embd];
            project_row(&attn_out, &mut buf, layer.wo.as_ref().unwrap(), n_embd, n_embd, n_tokens);
            buf
        };

        // 2g. Residual
        for i in 0..hidden.len() {
            hidden[i] += attn_proj[i];
        }

        // 2h. FFN RMS norm
        let mut ffn_normed = rms_norm(&hidden, eps, n_tokens, n_embd);
        if let Some(w) = &layer.ffn_norm {
            apply_weight(n_tokens, n_embd, &mut ffn_normed, w.data_f32());
        }

        // 2i. SwiGLU FFN
        let ffn_out = ffn_swiglu(&ffn_normed, layer, hp);

        // 2j. FFN residual
        for i in 0..hidden.len() {
            hidden[i] += ffn_out[i];
        }
    }

    // 3. Final RMS norm
    let mut final_normed = rms_norm(&hidden, eps, n_tokens, n_embd);
    if let Some(w) = &model.output_norm {
        apply_weight(n_tokens, n_embd, &mut final_normed, w.data_f32());
    }

    // 4. LM head
    if let Some(output) = &model.output {
        let mut logits = vec![0.0f32; n_tokens * n_vocab];
        project_row(&final_normed, &mut logits, output, n_embd, n_vocab, n_tokens);
        logits
    } else {
        vec![0.0f32; n_tokens * n_vocab]
    }
}

// ============================================================
// Embedding lookup
// ============================================================

fn embed_tokens(token_ids: &[u32], tok_embd: &Tensor, hp: &HParams) -> Vec<f32> {
    let n_embd = hp.n_embd as usize;
    let n_tokens = token_ids.len();
    let mut output = vec![0.0f32; n_tokens * n_embd];

    // For each token, dequantize the embedding row
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
            let d = u16::from_le_bytes([
                tok_embd.data[off],
                tok_embd.data[off + 1],
            ]);
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
        for i in 0..dim {
            sum_sq += (row[i] as f64) * (row[i] as f64);
        }
        let scale = 1.0 / ((sum_sq / dim as f64) as f32 + eps).sqrt();
        for i in 0..dim {
            dst[i] = row[i] * scale;
        }
    }
    out
}

fn apply_weight(n_tokens: usize, dim: usize, x: &mut [f32], w: &[f32]) {
    for t in 0..n_tokens {
        let base = t * dim;
        for i in 0..dim {
            x[base + i] *= w[i];
        }
    }
}

// ============================================================
// QKV projections
// ============================================================

fn project_qkv(
    input: &[f32], layer: &LayerWeights, hp: &HParams,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_embd = hp.n_embd as usize;
    let n_head = hp.n_head as usize;
    let n_head_kv = hp.n_head_kv as usize;
    let n_embd_head = hp.n_embd_head() as usize;
    let n_tokens = input.len() / n_embd;

    let n_q_total = n_head * n_embd_head;
    let n_kv_total = n_head_kv * n_embd_head;

    let mut q = vec![0.0f32; n_tokens * n_q_total];
    let mut k = vec![0.0f32; n_tokens * n_kv_total];
    let mut v = vec![0.0f32; n_tokens * n_kv_total];

    crate::models::qwen2::forward::project_row(
        input, &mut q, layer.wq.as_ref().unwrap(), n_embd, n_q_total, n_tokens);
    add_bias(&mut q, &layer.bq, n_tokens, n_q_total);

    crate::models::qwen2::forward::project_row(
        input, &mut k, layer.wk.as_ref().unwrap(), n_embd, n_kv_total, n_tokens);
    add_bias(&mut k, &layer.bk, n_tokens, n_kv_total);

    crate::models::qwen2::forward::project_row(
        input, &mut v, layer.wv.as_ref().unwrap(), n_embd, n_kv_total, n_tokens);
    add_bias(&mut v, &layer.bv, n_tokens, n_kv_total);

    (q, k, v)
}

fn add_bias(act: &mut [f32], bias: &Option<Tensor>, n_tokens: usize, dim: usize) {
    if let Some(b) = bias {
        let bd = b.data_f32();
        for t in 0..n_tokens {
            let base = t * dim;
            for i in 0..dim.min(bd.len()) {
                act[base + i] += bd[i];
            }
        }
    }
}

// ============================================================
// Projection: quantized matmul
// ============================================================

pub fn project_row(
    input: &[f32],
    output: &mut [f32],
    weight: &Tensor,
    input_dim: usize,
    output_dim: usize,
    n_tokens: usize,
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
                    for i in 0..input_dim {
                        sum += inp[i] * w_row[i];
                    }
                    out[o] = sum;
                }
            }
        }
        TensorType::Q4_0 => {
            let qk = 32;
            let nb = input_dim / qk;
            let blocks = weight.data_q4_0();
            let q8 = crate::avx2::quantize_row_q8_0_batch(input, n_tokens, input_dim);

            for t in 0..n_tokens {
                let q8_row = &q8[t * nb..(t + 1) * nb];
                let out = &mut output[t * output_dim..(t + 1) * output_dim];
                for o in 0..output_dim {
                    let row_blocks = &blocks[o * nb..(o + 1) * nb];
                    out[o] = crate::avx2::vec_dot_q4_0_q8_0(input_dim as i32, row_blocks, q8_row);
                }
            }
        }
        TensorType::Q8_0 => {
            let qk = 32;
            let nb = input_dim / qk;
            let blocks = weight.data_q8_0();
            let q8 = crate::avx2::quantize_row_q8_0_batch(input, n_tokens, input_dim);

            for t in 0..n_tokens {
                let q8_row = &q8[t * nb..(t + 1) * nb];
                let out = &mut output[t * output_dim..(t + 1) * output_dim];
                for o in 0..output_dim {
                    let row_blocks = &blocks[o * nb..(o + 1) * nb];
                    out[o] = crate::avx2::vec_dot_q8_0_q8_0(input_dim as i32, row_blocks, q8_row);
                }
            }
        }
        _ => panic!("project_row: unsupported type {:?}", weight.ttype),
    }
}

// ============================================================
// RoPE (Neox style)
// ============================================================

fn apply_rope(x: &mut [f32], positions: &[usize], n_head: usize, n_embd_head: usize, freq_base: f32) {
    let half = n_embd_head / 2;
    let n_tokens = positions.len();
    for t in 0..n_tokens {
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
    q: &[f32],
    k_all: &[f32],
    v_all: &[f32],
    positions: &[usize],
    n_tokens: usize,
    n_kv: usize,
    n_head: usize,
    n_head_kv: usize,
    n_embd_head: usize,
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
                for d in 0..n_embd_head {
                    s += q[q_start + d] * k_all[k_start + d];
                }
                s *= scale;
                scores[kv] = s;
                if s > max_score { max_score = s; }
            }
            for kv in valid..n_kv {
                scores[kv] = f32::NEG_INFINITY;
            }

            // Softmax
            let mut sum = 0.0f64;
            for s in &mut scores {
                *s = (*s - max_score).exp();
                sum += *s as f64;
            }
            let inv_sum = (1.0 / sum) as f32;
            for s in &mut scores {
                *s *= inv_sum;
            }

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
// SwiGLU FFN
// ============================================================

fn ffn_swiglu(input: &[f32], layer: &LayerWeights, hp: &HParams) -> Vec<f32> {
    let n_embd = hp.n_embd as usize;
    let n_ff = hp.n_ff as usize;
    let n_tokens = input.len() / n_embd;

    let mut gate = vec![0.0f32; n_tokens * n_ff];
    let mut up = vec![0.0f32; n_tokens * n_ff];
    let mut output = vec![0.0f32; n_tokens * n_embd];

    project_row(input, &mut gate, layer.ffn_gate.as_ref().unwrap(), n_embd, n_ff, n_tokens);
    project_row(input, &mut up, layer.ffn_up.as_ref().unwrap(), n_embd, n_ff, n_tokens);

    // SiLU(x) * up
    for i in 0..n_tokens * n_ff {
        gate[i] = silu(gate[i]) * up[i];
    }

    project_row(&gate, &mut output, layer.ffn_down.as_ref().unwrap(), n_ff, n_embd, n_tokens);
    output
}

#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}
