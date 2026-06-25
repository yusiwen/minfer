// Qwen2 forward pass — &[u8] everywhere (minfer2 pattern)

use crate::cache::KVCache;
use crate::tensor::TensorType;

const Q4B: usize = 18;
const Q8B: usize = 34;

pub fn forward(
    model: &super::Qwen2Model,
    token_ids: &[u32], positions: &[usize],
    kv_cache: &mut KVCache,
) -> Vec<f32> {
    let hp = &model.hparams;
    let nt = token_ids.len();
    let ne = hp.n_embd as usize;
    let nh = hp.n_head as usize;
    let nk = hp.n_head_kv as usize;
    let hd = hp.n_embd_head() as usize;
    let nv = hp.n_vocab as usize;
    let nf = hp.n_ff as usize;
    let nbe = ne / 32;
    let nbf = nf / 32;
    let nqt = nh * hd;
    let nkt = nk * hd;
    let maxd = nf.max(ne);
    let maxnb = maxd / 32;

    let mut bn = vec![0.0f32; nt * ne];
    let mut bq = vec![0.0f32; nt * nqt];
    let mut bk = vec![0.0f32; nt * nkt];
    let mut bv = vec![0.0f32; nt * nkt];
    let mut ba = vec![0.0f32; nt * ne];
    let mut bf = vec![0.0f32; nt * maxd];
    let mut bg = vec![0.0f32; nt * maxd];
    // Q8 activation buffer: raw bytes (minfer2 pattern, no BlockQ8_0)
    let mut qb = vec![0u8; nt * maxnb * Q8B];
    let max_seq = hp.max_seq_len as usize;
    let mut scrs_buf = vec![0.0f32; max_seq];

    // 1. Embedding
    let mut hidden = vec![0.0f32; nt * ne];
    embed_tokens(token_ids, model.tok_embd.as_ref().unwrap(), &mut hidden, ne);

    // 2. Per-layer loop
    for il in 0..model.n_layer() {
        let l = &model.layers[il];
        let (wq, wk, wv, wo, fg, fu, fd) = (
            l.wq.as_ref().map(|t| t.data_q4_0()).unwrap(),
            l.wk.as_ref().map(|t| t.data_q4_0()).unwrap(),
            l.wv.as_ref().map(|t| t.data_q4_0()).unwrap(),
            l.wo.as_ref().map(|t| t.data_q4_0()).unwrap(),
            l.ffn_gate.as_ref().map(|t| t.data_q4_0()).unwrap(),
            l.ffn_up.as_ref().map(|t| t.data_q4_0()).unwrap(),
            l.ffn_down.as_ref().map(|t| t.data_q4_0()).unwrap(),
        );

        // --- RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut bn, nt, ne, l.attn_norm.as_ref().map(|t| t.data_f32()));

        // --- Quantize ONCE for QKV → raw &[u8] ---
        let (q8, rest) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bn, nt, ne, q8);

        // --- QKV ---
        lq4_blk(wq, q8, &mut bq, nqt, ne, nt);
        if let Some(b) = &l.bq { add_bias(&mut bq, b.data_f32(), nt, nqt); }
        lq4_blk(wk, q8, &mut bk, nkt, ne, nt);
        if let Some(b) = &l.bk { add_bias(&mut bk, b.data_f32(), nt, nkt); }
        lq4_blk(wv, q8, &mut bv, nkt, ne, nt);
        if let Some(b) = &l.bv { add_bias(&mut bv, b.data_f32(), nt, nkt); }

        // --- RoPE ---
        apply_rope(&mut bq, positions, nh, hd, hp.rope_freq_base);
        apply_rope(&mut bk, positions, nk, hd, hp.rope_freq_base);

        // --- KV cache ---
        kv_cache.layers[il].store_multi(positions, &bk, &bv);
        let nkv = kv_cache.layers[il].size;

        // --- Attention ---
        gqa_attn(&bq, &kv_cache.layers[il].k[..nkv * nkt], &kv_cache.layers[il].v[..nkv * nkt],
            positions, nt, nkv, nh, nk, hd, &mut ba, &mut scrs_buf[..nkv]);

        // --- Wo ---
        let (q8a, _) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&ba, nt, ne, q8a);
        lq4_blk(wo, q8a, &mut bn, ne, ne, nt);
        for i in 0..hidden.len() { hidden[i] += bn[i]; }

        // --- FFN RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut bf[..nt * ne], nt, ne, l.ffn_norm.as_ref().map(|t| t.data_f32()));

        // --- SwiGLU ---
        let (q8f, _) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bf[..nt * ne], nt, ne, q8f);
        lq4_blk(fg, q8f, &mut bg, nf, ne, nt);
        lq4_blk(fu, q8f, &mut bf, nf, ne, nt);
        let len = nt * nf;
        let bp = bg.as_mut_ptr();
        unsafe {
            crate::vec_ops::vec_silu_f32(len,
                std::slice::from_raw_parts_mut(bp, len),
                std::slice::from_raw_parts(bp as *const f32, len));
        }
        for i in 0..len { bg[i] *= bf[i]; }

        let (q8g, _) = qb.split_at_mut(nt * nbf * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bg[..nt * nf], nt, nf, q8g);
        lq4_blk(fd, q8g, &mut bn, ne, nf, nt);
        for i in 0..hidden.len() { hidden[i] += bn[i]; }
    }

    // 3. Final RMSNorm
    rms_norm(&hidden, hp.f_norm_rms_eps, &mut bn, nt, ne, model.output_norm.as_ref().map(|t| t.data_f32()));

    // 4. LM head
    if let Some(output) = &model.output {
        let (q8e, _) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bn, nt, ne, q8e);
        let mut logits = vec![0.0f32; nt * nv];
        match output.ttype {
            TensorType::Q8_0 => lq8_blk(output.data_q8_0(), q8e, &mut logits, nv, ne, nt),
            TensorType::Q4_0 => lq4_blk(output.data_q4_0(), q8e, &mut logits, nv, ne, nt),
            _ => unreachable!(),
        }
        logits
    } else { vec![] }
}

fn embed_tokens(ids: &[u32], t: &crate::tensor::Tensor, out: &mut [f32], ne: usize) {
    let blk = 32usize;
    let nbp = (ne + blk - 1) / blk;
    let bb = t.ttype.type_size();
    let is8 = t.ttype == TensorType::Q8_0;
    for (ti, &id) in ids.iter().enumerate() {
        let idx = id as usize;
        let doff = ti * ne;
        for b in 0..nbp {
            let off = (idx * nbp + b) * bb;
            let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
            let mv = blk.min(ne - b * blk);
            if is8 {
                for j in 0..mv { out[doff + b * blk + j] = (t.data[off + 2 + j] as i8) as f32 * d; }
            } else {
                for j in (0..mv).step_by(2) {
                    let byte = t.data[off + 2 + j / 2];
                    let lo = (byte & 0x0F) as i8 - 8;
                    let hi = (byte >> 4) as i8 - 8;
                    out[doff + b * blk + j] = lo as f32 * d;
                    if j + 1 < mv { out[doff + b * blk + j + 1] = hi as f32 * d; }
                }
            }
        }
    }
}

fn rms_norm(x: &[f32], eps: f32, out: &mut [f32], n: usize, d: usize, w: Option<&[f32]>) {
    for t in 0..n {
        let row = &x[t * d..(t + 1) * d];
        let dst = &mut out[t * d..(t + 1) * d];
        let mut ss = 0.0f64;
        for i in 0..d { ss += (row[i] as f64) * (row[i] as f64); }
        let sc = 1.0 / ((ss / d as f64) as f32 + eps).sqrt();
        match w {
            Some(w) => { for i in 0..d { dst[i] = row[i] * sc * w[i]; } }
            None => { for i in 0..d { dst[i] = row[i] * sc; } }
        }
    }
}

fn add_bias(x: &mut [f32], b: &[f32], n: usize, d: usize) {
    for t in 0..n { let base = t * d; for i in 0..d.min(b.len()) { x[base + i] += b[i]; } }
}

fn lq4_blk(w: &[u8], x: &[u8], out: &mut [f32], od: usize, id: usize, nt: usize) {
    let nb = id / 32;
    for o in 0..od {
        let wb = &w[o * nb * Q4B..(o + 1) * nb * Q4B];
        for t in 0..nt {
            let xb = &x[t * nb * Q8B..(t + 1) * nb * Q8B];
            out[t * od + o] = crate::avx2::dot_q4_0_q8_0(wb, xb);
        }
    }
}

fn lq8_blk(w: &[u8], x: &[u8], out: &mut [f32], od: usize, id: usize, nt: usize) {
    let nb = id / 32;
    for o in 0..od {
        let wb = &w[o * nb * Q8B..(o + 1) * nb * Q8B];
        for t in 0..nt {
            let xb = &x[t * nb * Q8B..(t + 1) * nb * Q8B];
            out[t * od + o] = crate::avx2::dot_q8_0_q8_0(wb, xb);
        }
    }
}

fn apply_rope(x: &mut [f32], pos: &[usize], nh: usize, hd: usize, fb: f32) {
    let half = hd / 2;
    let mut freqs = [0.0f32; 64]; // max half = 64 for hd <= 128
    for i in 0..half { freqs[i] = 1.0 / fb.powf((2 * i) as f32 / hd as f32); }
    for t in 0..pos.len() {
        let p = pos[t] as f32;
        for h in 0..nh {
            let b = t * nh * hd + h * hd;
            for i in 0..half {
                let th = p * freqs[i]; let (sn, cs) = th.sin_cos();
                let (i0, i1) = (b + i, b + i + half);
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * cs - x1 * sn;
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }
}

fn gqa_attn(q: &[f32], ka: &[f32], va: &[f32], pos: &[usize], nt: usize, nkv: usize,
    nh: usize, nk: usize, hd: usize, out: &mut [f32], scrs: &mut [f32]) {
    let gqa = nh / nk; let ne = nh * hd; let sc = 1.0 / (hd as f32).sqrt();
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qs = t * ne + h * hd;
            let vl = (pos[t] + 1).min(nkv);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks = kv * nk * hd + hk * hd;
                let s = crate::vec_ops::vec_dot_f32(hd, &q[qs..qs + hd], &ka[ks..ks + hd]) * sc;
                scrs[kv] = s; if s > mx { mx = s; }
            }
            for kv in vl..nkv { scrs[kv] = f32::NEG_INFINITY; }
            let sp = scrs.as_mut_ptr();
            let sm = unsafe {
                crate::vec_ops::vec_soft_max_f32(nkv,
                    std::slice::from_raw_parts_mut(sp, nkv),
                    std::slice::from_raw_parts(sp as *const f32, nkv),
                    mx)
            };
            let is = (1.0 / sm) as f32;
            crate::vec_ops::vec_scale_f32(nkv, scrs, is);
            let os = t * ne + h * hd;
            let slice = &mut out[os..os + hd];
            for d in 0..hd { slice[d] = 0.0; }
            let stride = nk * hd;
            let vs_base = hk * hd;
            for kv in 0..nkv {
                crate::vec_ops::vec_muladd_f32(hd, slice,
                    &va[kv * stride + vs_base..kv * stride + vs_base + hd], scrs[kv]);
            }
        }
    }
}


