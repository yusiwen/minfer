// Qwen2 forward pass — zero-alloc, pre-extracted weight blocks
// Reference: minfer2/src/models/qwen2/forward.rs

use crate::block::{self, BlockQ4_0, BlockQ8_0};
use crate::cache::KVCache;
use crate::tensor::{Tensor, TensorType};

use super::loader::HParams;
use super::LayerWeights;

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
    let nb_embd = n_embd / 32;
    let nb_ff = n_ff / 32;
    let n_q_total = n_head * n_embd_head;
    let n_kv_total = n_head_kv * n_embd_head;
    let max_dim = n_ff.max(n_embd);
    let max_nb = max_dim / 32;

    // Pre-allocated working buffers
    let mut b_norm = vec![0.0f32; n_tokens * n_embd];
    let mut b_q    = vec![0.0f32; n_tokens * n_q_total];
    let mut b_k    = vec![0.0f32; n_tokens * n_kv_total];
    let mut b_v    = vec![0.0f32; n_tokens * n_kv_total];
    let mut b_attn = vec![0.0f32; n_tokens * n_embd];
    let mut b_ffn  = vec![0.0f32; n_tokens * max_dim];
    let mut b_gate = vec![0.0f32; n_tokens * max_dim];
    let mut q8_buf = vec![BlockQ8_0::default(); n_tokens * max_nb];

    // 1. Embedding
    let mut hidden = vec![0.0f32; n_tokens * n_embd];
    embed_tokens(token_ids, model.tok_embd.as_ref().unwrap(), &mut hidden, hp);

    // 2. Per-layer loop
    for il in 0..model.n_layer() {
        let layer = &model.layers[il];

        // --- Extract weight blocks ONCE per layer ---
        let wq    = layer.wq.as_ref().map(|t| t.data_q4_0()).unwrap();
        let wk    = layer.wk.as_ref().map(|t| t.data_q4_0()).unwrap();
        let wv    = layer.wv.as_ref().map(|t| t.data_q4_0()).unwrap();
        let wo    = layer.wo.as_ref().map(|t| t.data_q4_0()).unwrap();
        let fg    = layer.ffn_gate.as_ref().map(|t| t.data_q4_0()).unwrap();
        let fu    = layer.ffn_up.as_ref().map(|t| t.data_q4_0()).unwrap();
        let fd    = layer.ffn_down.as_ref().map(|t| t.data_q4_0()).unwrap();

        // --- RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut b_norm, n_tokens, n_embd);
        if let Some(w) = &layer.attn_norm { apply_weight(&mut b_norm, w.data_f32(), n_tokens, n_embd); }

        // --- Quantize ONCE for QKV ---
        let (q8, _) = q8_buf.split_at_mut(n_tokens * nb_embd);
        quantize_activation(&b_norm, n_tokens, n_embd, q8);

        // --- QKV ---
        linear_q4_blk(wq, q8, &mut b_q, n_q_total, n_embd, n_tokens);
        if let Some(b) = &layer.bq { add_bias(&mut b_q, b.data_f32(), n_tokens, n_q_total); }
        linear_q4_blk(wk, q8, &mut b_k, n_kv_total, n_embd, n_tokens);
        if let Some(b) = &layer.bk { add_bias(&mut b_k, b.data_f32(), n_tokens, n_kv_total); }
        linear_q4_blk(wv, q8, &mut b_v, n_kv_total, n_embd, n_tokens);
        if let Some(b) = &layer.bv { add_bias(&mut b_v, b.data_f32(), n_tokens, n_kv_total); }

        // --- RoPE ---
        apply_rope(&mut b_q, positions, n_head, n_embd_head, hp.rope_freq_base);
        apply_rope(&mut b_k, positions, n_head_kv, n_embd_head, hp.rope_freq_base);

        // --- KV cache ---
        kv_cache.layers[il].store_multi(positions, &b_k, &b_v);

        // --- Attention ---
        let n_kv = kv_cache.layers[il].size;
        gqa_attention_batch(&b_q, &kv_cache.layers[il].k[..n_kv * n_kv_total],
            &kv_cache.layers[il].v[..n_kv * n_kv_total], positions, n_tokens, n_kv,
            n_head, n_head_kv, n_embd_head, &mut b_attn);

        // --- Wo projection ---
        let (q8_attn, _) = q8_buf.split_at_mut(n_tokens * nb_embd);
        quantize_activation(&b_attn, n_tokens, n_embd, q8_attn);
        linear_q4_blk(wo, q8_attn, &mut b_norm, n_embd, n_embd, n_tokens);
        for i in 0..hidden.len() { hidden[i] += b_norm[i]; }

        // --- FFN RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut b_ffn[..n_tokens * n_embd], n_tokens, n_embd);
        if let Some(w) = &layer.ffn_norm { apply_weight(&mut b_ffn[..n_tokens * n_embd], w.data_f32(), n_tokens, n_embd); }

        // --- SwiGLU FFN ---
        let (q8_ffn, _) = q8_buf.split_at_mut(n_tokens * nb_embd);
        quantize_activation(&b_ffn[..n_tokens * n_embd], n_tokens, n_embd, q8_ffn);
        linear_q4_blk(fg, q8_ffn, &mut b_gate, n_ff, n_embd, n_tokens);
        linear_q4_blk(fu, q8_ffn, &mut b_ffn, n_ff, n_embd, n_tokens);
        for i in 0..n_tokens * n_ff { b_gate[i] = silu(b_gate[i]) * b_ffn[i]; }

        let (q8_gate, _) = q8_buf.split_at_mut(n_tokens * nb_ff);
        quantize_activation(&b_gate[..n_tokens * n_ff], n_tokens, n_ff, q8_gate);
        linear_q4_blk(fd, q8_gate, &mut b_norm, n_embd, n_ff, n_tokens);
        for i in 0..hidden.len() { hidden[i] += b_norm[i]; }
    }

    // 3. Final RMSNorm
    rms_norm(&hidden, hp.f_norm_rms_eps, &mut b_norm, n_tokens, n_embd);
    if let Some(w) = &model.output_norm { apply_weight(&mut b_norm, w.data_f32(), n_tokens, n_embd); }

    // 4. LM head
    if let Some(output) = &model.output {
        let (q8_embd, _) = q8_buf.split_at_mut(n_tokens * nb_embd);
        quantize_activation(&b_norm, n_tokens, n_embd, q8_embd);
        match output.ttype {
            TensorType::Q8_0 => {
                let mut logits = vec![0.0f32; n_tokens * n_vocab];
                linear_q8_blk(output.data_q8_0(), q8_embd, &mut logits, n_vocab, n_embd, n_tokens);
                logits
            }
            TensorType::Q4_0 => {
                let mut logits = vec![0.0f32; n_tokens * n_vocab];
                linear_q4_blk(output.data_q4_0(), q8_embd, &mut logits, n_vocab, n_embd, n_tokens);
                logits
            }
            _ => unreachable!(),
        }
    } else { vec![] }
}

// ============================================================
// Embedding lookup
// ============================================================

fn embed_tokens(ids: &[u32], t: &Tensor, out: &mut [f32], hp: &HParams) {
    let n_embd = hp.n_embd as usize;
    let blk = 32usize;
    let nbp = (n_embd + blk - 1) / blk;
    let bb = t.ttype.type_size();
    let is8 = t.ttype == TensorType::Q8_0;
    for (ti, &id) in ids.iter().enumerate() {
        let idx = id as usize;
        let doff = ti * n_embd;
        for b in 0..nbp {
            let off = (idx * nbp + b) * bb;
            let d = block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
            let mv = blk.min(n_embd - b * blk);
            if is8 {
                for j in 0..mv { out[doff + b * blk + j] = (t.data[off + 2 + j] as i8) as f32 * d; }
            } else {
                for j in 0..mv {
                    let nib = if j % 2 == 0 { t.data[off + 2 + j / 2] & 0x0F } else { t.data[off + 2 + j / 2] >> 4 };
                    out[doff + b * blk + j] = (nib as i8 - 8) as f32 * d;
                }
            }
        }
    }
}

// ============================================================
// RMSNorm
// ============================================================

fn rms_norm(x: &[f32], eps: f32, out: &mut [f32], n: usize, d: usize) {
    for t in 0..n {
        let row = &x[t * d..(t + 1) * d];
        let dst = &mut out[t * d..(t + 1) * d];
        let mut ss = 0.0f64;
        for i in 0..d { ss += (row[i] as f64) * (row[i] as f64); }
        let sc = 1.0 / ((ss / d as f64) as f32 + eps).sqrt();
        for i in 0..d { dst[i] = row[i] * sc; }
    }
}

fn apply_weight(x: &mut [f32], w: &[f32], n: usize, d: usize) {
    for t in 0..n { let b = t * d; for i in 0..d { x[b + i] *= w[i]; } }
}

fn add_bias(x: &mut [f32], b: &[f32], n: usize, d: usize) {
    for t in 0..n { let base = t * d; for i in 0..d.min(b.len()) { x[base + i] += b[i]; } }
}

// ============================================================
// Quantization
// ============================================================

fn quantize_activation(x: &[f32], n: usize, d: usize, q: &mut [BlockQ8_0]) {
    crate::avx2::quantize_row_q8_0_buf(x, n, d, q);
}

// ============================================================
// Q4_0 × Q8_0 projection (pre-extracted blocks)
// ============================================================

/// Q4_0 weight blocks × pre-quantized Q8_0 activations, write to `out`.
#[inline(always)]
fn linear_q4_blk(blocks: &[BlockQ4_0], x_q8: &[BlockQ8_0], out: &mut [f32],
                 out_dim: usize, in_dim: usize, n_tokens: usize) {
    let nb = in_dim / 32;
    for o in 0..out_dim {
        let wr = &blocks[o * nb..(o + 1) * nb];
        for t in 0..n_tokens {
            out[t * out_dim + o] = crate::avx2::vec_dot_q4_0_q8_0(in_dim as i32, wr, &x_q8[t * nb..(t + 1) * nb]);
        }
    }
}

/// Q8_0 weight blocks × pre-quantized Q8_0 activations, write to `out`.
#[inline(always)]
fn linear_q8_blk(blocks: &[BlockQ8_0], x_q8: &[BlockQ8_0], out: &mut [f32],
                 out_dim: usize, in_dim: usize, n_tokens: usize) {
    let nb = in_dim / 32;
    for o in 0..out_dim {
        let wr = &blocks[o * nb..(o + 1) * nb];
        for t in 0..n_tokens {
            out[t * out_dim + o] = crate::avx2::vec_dot_q8_0_q8_0(in_dim as i32, wr, &x_q8[t * nb..(t + 1) * nb]);
        }
    }
}

// ============================================================
// RoPE (Neox style)
// ============================================================

fn apply_rope(x: &mut [f32], pos: &[usize], nh: usize, hd: usize, fb: f32) {
    let half = hd / 2;
    for t in 0..pos.len() {
        let p = pos[t] as f32;
        for h in 0..nh {
            let b = t * nh * hd + h * hd;
            for i in 0..half {
                let fr = 1.0 / fb.powf((2 * i) as f32 / hd as f32);
                let th = p * fr;
                let (sn, cs) = th.sin_cos();
                let i0 = b + i;
                let i1 = b + i + half;
                let x0 = x[i0];
                let x1 = x[i1];
                x[i0] = x0 * cs - x1 * sn;
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }
}

// ============================================================
// GQA attention
// ============================================================

fn gqa_attention_batch(q: &[f32], k_all: &[f32], v_all: &[f32],
    pos: &[usize], nt: usize, nkv: usize,
    nh: usize, nh_kv: usize, hd: usize, out: &mut [f32]) {
    let n_gqa = nh / nh_kv;
    let ne = nh * hd;
    let sc = 1.0 / (hd as f32).sqrt();
    let mut scrs = vec![0.0f32; nkv];
    for h in 0..nh {
        let hk = h / n_gqa;
        for t in 0..nt {
            let qs = t * ne + h * hd;
            let cp = pos[t];
            let vl = (cp + 1).min(nkv);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks = kv * nh_kv * hd + hk * hd;
                let mut s = 0.0f32;
                for d in 0..hd { s += q[qs + d] * k_all[ks + d]; }
                s *= sc; scrs[kv] = s;
                if s > mx { mx = s; }
            }
            for kv in vl..nkv { scrs[kv] = f32::NEG_INFINITY; }
            let mut sm = 0.0f64;
            for s in &mut scrs { *s = (*s - mx).exp(); sm += *s as f64; }
            let is = (1.0 / sm) as f32;
            for s in &mut scrs { *s *= is; }
            let os = t * ne + h * hd;
            for d in 0..hd {
                let mut ac = 0.0f32;
                for kv in 0..nkv { ac += scrs[kv] * v_all[kv * nh_kv * hd + hk * hd + d]; }
                out[os + d] = ac;
            }
        }
    }
}

#[inline]
fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
