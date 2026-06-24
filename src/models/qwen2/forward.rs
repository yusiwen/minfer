// Qwen2 forward pass — zero-alloc (all buffers pre-allocated in forward())
// Translated from: llama.cpp/src/models/qwen2.cpp (build_arch_graph)
// Based on minfer2/src/models/qwen2/forward.rs

use crate::block::{self, BlockQ8_0};
use crate::cache::KVCache;
use crate::tensor::{Tensor, TensorType};

use super::loader::HParams;
use super::LayerWeights;

/// Maximum working buffer size across all layers (n_ff > n_embd).
fn max_buf_dim(n_embd: usize, n_ff: usize) -> usize { n_ff.max(n_embd) }

pub fn forward(
    model: &super::Qwen2Model,
    token_ids: &[u32], positions: &[usize],
    kv_cache: &mut KVCache,
) -> Vec<f32> {
    let hp = &model.hparams;
    let n_tokens = token_ids.len();
    let n_embd = hp.n_embd as usize;
    let n_head = hp.n_head as usize;
    let n_head_kv = hp.n_head_kv as usize;
    let n_embd_head = hp.n_embd_head() as usize;
    let n_vocab = hp.n_vocab as usize;
    let n_ff = hp.n_ff as usize;
    let n_gqa = n_head / n_head_kv;
    let nb_embd = n_embd / 32;           // blocks per row for n_embd
    let nb_ff = n_ff / 32;               // blocks per row for n_ff
    let n_q_total = n_head * n_embd_head;
    let n_kv_total = n_head_kv * n_embd_head;

    // Pre-allocate working buffers (live across all layers).
    let max_dim = max_buf_dim(n_embd, n_ff);
    let max_nb = max_dim / 32;
    // f32 scratch buffers
    let mut b_norm = vec![0.0f32; n_tokens * n_embd];
    let mut b_q    = vec![0.0f32; n_tokens * n_q_total];
    let mut b_k    = vec![0.0f32; n_tokens * n_kv_total];
    let mut b_v    = vec![0.0f32; n_tokens * n_kv_total];
    let mut b_attn = vec![0.0f32; n_tokens * n_embd];
    let mut b_ffn  = vec![0.0f32; n_tokens * max_dim];
    let mut b_gate = vec![0.0f32; n_tokens * max_dim];
    // q8 scratch (reused by all quantize_activation calls)
    let mut q8_buf = vec![BlockQ8_0::default(); n_tokens * max_nb];

    // 1. Token embedding
    let mut hidden = vec![0.0f32; n_tokens * n_embd];
    embed_tokens(token_ids, model.tok_embd.as_ref().unwrap(), &mut hidden, hp);

    // 2. Per-layer loop
    for il in 0..model.n_layer() {
        let layer = &model.layers[il];

        // --- Attention RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut b_norm, n_tokens, n_embd);
        if let Some(w) = &layer.attn_norm {
            apply_weight(&mut b_norm, w.data_f32(), n_tokens, n_embd);
        }

        // --- Quantize ONCE for QKV ---
        let (q8, q8_rest) = q8_buf.split_at_mut(n_tokens * nb_embd);
        quantize_activation(&b_norm, n_tokens, n_embd, q8);

        // --- QKV ---
        linear_q4(layer.wq.as_ref().unwrap(), q8, &mut b_q, n_q_total, n_embd, n_tokens);
        if let Some(b) = &layer.bq { add_bias(&mut b_q, b.data_f32(), n_tokens, n_q_total); }
        linear_q4(layer.wk.as_ref().unwrap(), q8, &mut b_k, n_kv_total, n_embd, n_tokens);
        if let Some(b) = &layer.bk { add_bias(&mut b_k, b.data_f32(), n_tokens, n_kv_total); }
        linear_q4(layer.wv.as_ref().unwrap(), q8, &mut b_v, n_kv_total, n_embd, n_tokens);
        if let Some(b) = &layer.bv { add_bias(&mut b_v, b.data_f32(), n_tokens, n_kv_total); }

        // --- RoPE on Q and K ---
        apply_rope(&mut b_q, positions, n_head, n_embd_head, hp.rope_freq_base);
        apply_rope(&mut b_k, positions, n_head_kv, n_embd_head, hp.rope_freq_base);

        // --- KV cache store ---
        kv_cache.layers[il].store_multi(positions, &b_k, &b_v);

        // --- Attention ---
        let n_kv = kv_cache.layers[il].size;
        let k_all = &kv_cache.layers[il].k[..n_kv * n_kv_total];
        let v_all = &kv_cache.layers[il].v[..n_kv * n_kv_total];
        gqa_attention_batch(&b_q, k_all, v_all, positions, n_tokens, n_kv,
            n_head, n_head_kv, n_embd_head, &mut b_attn);

        // --- Output projection (quantize attn, Q4_0 x Q8_0) ---
        quantize_activation(&b_attn, n_tokens, n_embd, &mut q8_buf[..n_tokens * nb_embd]);
        linear_q4(layer.wo.as_ref().unwrap(), &q8_buf[..n_tokens * nb_embd],
            &mut b_norm, n_embd, n_embd, n_tokens);
        for i in 0..hidden.len() { hidden[i] += b_norm[i]; }

        // --- FFN RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut b_ffn[..n_tokens * n_embd], n_tokens, n_embd);
        if let Some(w) = &layer.ffn_norm {
            apply_weight(&mut b_ffn[..n_tokens * n_embd], w.data_f32(), n_tokens, n_embd);
        }

        // --- SwiGLU FFN ---
        quantize_activation(&b_ffn[..n_tokens * n_embd], n_tokens, n_embd,
            &mut q8_buf[..n_tokens * nb_embd]);
        // gate = Q4_0(gate) × Q8_0(x)
        linear_q4(layer.ffn_gate.as_ref().unwrap(), &q8_buf[..n_tokens * nb_embd],
            &mut b_gate, n_ff, n_embd, n_tokens);
        // up = Q4_0(up) × Q8_0(x)
        linear_q4(layer.ffn_up.as_ref().unwrap(), &q8_buf[..n_tokens * nb_embd],
            &mut b_ffn, n_ff, n_embd, n_tokens);
        // SiLU(gate) * up → gate
        for i in 0..n_tokens * n_ff { b_gate[i] = silu(b_gate[i]) * b_ffn[i]; }
        // down = Q4_0(down) × Q8_0(gate*up)
        quantize_activation(&b_gate[..n_tokens * n_ff], n_tokens, n_ff,
            &mut q8_buf[..n_tokens * nb_ff]);
        linear_q4(layer.ffn_down.as_ref().unwrap(), &q8_buf[..n_tokens * nb_ff],
            &mut b_norm, n_embd, n_ff, n_tokens);
        for i in 0..hidden.len() { hidden[i] += b_norm[i]; }
    }

    // 3. Final RMSNorm
    rms_norm(&hidden, hp.f_norm_rms_eps, &mut b_norm, n_tokens, n_embd);
    if let Some(w) = &model.output_norm {
        apply_weight(&mut b_norm, w.data_f32(), n_tokens, n_embd);
    }

    // 4. LM head
    if let Some(output) = &model.output {
        quantize_activation(&b_norm, n_tokens, n_embd,
            &mut q8_buf[..n_tokens * nb_embd]);
        match output.ttype {
            TensorType::Q8_0 => {
                let mut logits = vec![0.0f32; n_tokens * n_vocab];
                linear_q8(output, &q8_buf[..n_tokens * nb_embd],
                    &mut logits, n_vocab, n_embd, n_tokens);
                logits
            }
            TensorType::Q4_0 => {
                let mut logits = vec![0.0f32; n_tokens * n_vocab];
                linear_q4(output, &q8_buf[..n_tokens * nb_embd],
                    &mut logits, n_vocab, n_embd, n_tokens);
                logits
            }
            _ => unreachable!(),
        }
    } else {
        vec![0.0f32; n_tokens * n_vocab]
    }
}

// ============================================================
// Embedding lookup — write to pre-allocated out
// ============================================================

fn embed_tokens(token_ids: &[u32], tok_embd: &Tensor, out: &mut [f32], hp: &HParams) {
    let n_embd = hp.n_embd as usize;
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
                    out[dst_offset + b * blk_size + j] =
                        (tok_embd.data[off + 2 + j] as i8) as f32 * d_f32;
                }
            } else {
                for j in 0..max_v {
                    let nibble = if j % 2 == 0 {
                        tok_embd.data[off + 2 + j / 2] & 0x0F
                    } else {
                        tok_embd.data[off + 2 + j / 2] >> 4
                    };
                    out[dst_offset + b * blk_size + j] = (nibble as i8 - 8) as f32 * d_f32;
                }
            }
        }
    }
}

// ============================================================
// RMSNorm — write to pre-allocated out
// ============================================================

fn rms_norm(x: &[f32], eps: f32, out: &mut [f32], n_tokens: usize, dim: usize) {
    for t in 0..n_tokens {
        let row = &x[t * dim..(t + 1) * dim];
        let dst = &mut out[t * dim..(t + 1) * dim];
        let mut sum_sq = 0.0f64;
        for i in 0..dim { sum_sq += (row[i] as f64) * (row[i] as f64); }
        let scale = 1.0 / ((sum_sq / dim as f64) as f32 + eps).sqrt();
        for i in 0..dim { dst[i] = row[i] * scale; }
    }
}

fn apply_weight(x: &mut [f32], w: &[f32], n_tokens: usize, dim: usize) {
    for t in 0..n_tokens {
        let base = t * dim;
        for i in 0..dim { x[base + i] *= w[i]; }
    }
}

fn add_bias(x: &mut [f32], b: &[f32], n_tokens: usize, dim: usize) {
    for t in 0..n_tokens {
        let base = t * dim;
        for i in 0..dim.min(b.len()) { x[base + i] += b[i]; }
    }
}

// ============================================================
// Quantization — write to pre-allocated q8
// ============================================================

fn quantize_activation(x: &[f32], n_tokens: usize, dim: usize, q8: &mut [BlockQ8_0]) {
    crate::avx2::quantize_row_q8_0_buf(x, n_tokens, dim, q8);
}

// ============================================================
// Projections — write to pre-allocated out
// ============================================================

/// Q4_0 × pre-quantized Q8_0, write to `out`.
fn linear_q4(weight: &Tensor, x_q8: &[BlockQ8_0], out: &mut [f32],
             out_dim: usize, in_dim: usize, n_tokens: usize) {
    let nb = in_dim / 32;
    let blocks = weight.data_q4_0();
    for o in 0..out_dim {
        let w_row = &blocks[o * nb..(o + 1) * nb];
        for t in 0..n_tokens {
            out[t * out_dim + o] = crate::avx2::vec_dot_q4_0_q8_0(
                in_dim as i32, w_row, &x_q8[t * nb..(t + 1) * nb]);
        }
    }
}

/// Q8_0 × pre-quantized Q8_0, write to `out`.
fn linear_q8(weight: &Tensor, x_q8: &[BlockQ8_0], out: &mut [f32],
             out_dim: usize, in_dim: usize, n_tokens: usize) {
    let nb = in_dim / 32;
    let blocks = weight.data_q8_0();
    for o in 0..out_dim {
        let w_row = &blocks[o * nb..(o + 1) * nb];
        for t in 0..n_tokens {
            out[t * out_dim + o] = crate::avx2::vec_dot_q8_0_q8_0(
                in_dim as i32, w_row, &x_q8[t * nb..(t + 1) * nb]);
        }
    }
}

// ============================================================
// RoPE (Neox style) — in-place
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
// GQA attention — write to pre-allocated out
// ============================================================

fn gqa_attention_batch(
    q: &[f32], k_all: &[f32], v_all: &[f32],
    positions: &[usize], n_tokens: usize, n_kv: usize,
    n_head: usize, n_head_kv: usize, n_embd_head: usize,
    out: &mut [f32],
) {
    let n_gqa = n_head / n_head_kv;
    let n_embd = n_head * n_embd_head;
    let scale = 1.0 / (n_embd_head as f32).sqrt();
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

            // Softmax (f64 accumulator for numerical stability)
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
}

/// u8 silu
#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }

/// Legacy project_row (f32 fallback only).
pub fn project_row(input: &[f32], output: &mut [f32], weight: &Tensor,
                   input_dim: usize, output_dim: usize, n_tokens: usize) {
    match weight.ttype {
        TensorType::F32 => {
            let w = weight.data_f32();
            for t in 0..n_tokens {
                let inp = &input[t * input_dim..(t + 1) * input_dim];
                let out = &mut output[t * output_dim..(t + 1) * output_dim];
                for o in 0..output_dim {
                    let w_row = &w[o * input_dim..(o + 1) * input_dim];
                    let mut sum = 0.0f32;
                    for i in 0..input_dim { sum += inp[i] * w_row[i]; }
                    out[o] = sum;
                }
            }
        }
        _ => panic!("project_row: unsupported type {:?}", weight.ttype),
    }
}
