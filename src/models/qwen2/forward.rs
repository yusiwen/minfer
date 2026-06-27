// Qwen2 forward pass — &[u8] everywhere (minfer2 pattern)

use crate::cache::KVCache;
use crate::tensor::{Tensor, TensorType};

const Q4B: usize = 18;
const Q41B: usize = 20;
const Q8B: usize = 34;
const Q4KB: usize = 144;
const Q6KB: usize = 210;

// ============================================================
// Dispatched quantized matmul: weight tensor × Q8_0 activation
// ============================================================

fn quant_matmul(w: &Tensor, x: &[u8], out: &mut [f32], od: usize, id: usize, nt: usize) {
    match w.ttype {
        TensorType::Q4_0 => {
            let nb = id / 32;
            let ws = nb * Q4B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_0_q8_0(wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        TensorType::Q4_1 => {
            let nb = id / 32;
            let ws = nb * Q41B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_1_q8_0(wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        TensorType::Q4_K => {
            let nk = id / 256;
            let ws = nk * Q4KB;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_k_q8_0(wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
                }
            }
        }
        TensorType::Q6_K => {
            let nk = id / 256;
            let ws = nk * Q6KB;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q6_k_q8_0(wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
                }
            }
        }
        TensorType::Q8_0 => {
            let nb = id / 32;
            let ws = nb * Q8B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q8_0_q8_0(wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in quant_matmul", w.ttype),
    }
}

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

        // --- RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut bn, nt, ne, l.attn_norm.as_ref().map(|t| t.data_f32()));

        // --- Quantize ONCE for QKV → raw &[u8] ---
        let (q8, _rest) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bn, nt, ne, q8);

        // --- QKV ---
        quant_matmul(l.wq.as_ref().unwrap(), q8, &mut bq, nqt, ne, nt);
        if let Some(b) = &l.bq { add_bias(&mut bq, b.data_f32(), nt, nqt); }
        quant_matmul(l.wk.as_ref().unwrap(), q8, &mut bk, nkt, ne, nt);
        if let Some(b) = &l.bk { add_bias(&mut bk, b.data_f32(), nt, nkt); }
        quant_matmul(l.wv.as_ref().unwrap(), q8, &mut bv, nkt, ne, nt);
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
        let (q8a, _rest) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&ba, nt, ne, q8a);
        quant_matmul(l.wo.as_ref().unwrap(), q8a, &mut bn, ne, ne, nt);
        for i in 0..hidden.len() { hidden[i] += bn[i]; }

        // --- FFN RMSNorm ---
        rms_norm(&hidden, hp.f_norm_rms_eps, &mut bf[..nt * ne], nt, ne, l.ffn_norm.as_ref().map(|t| t.data_f32()));

        // --- SwiGLU ---
        let (q8f, _rest) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bf[..nt * ne], nt, ne, q8f);
        quant_matmul(l.ffn_gate.as_ref().unwrap(), q8f, &mut bg, nf, ne, nt);
        quant_matmul(l.ffn_up.as_ref().unwrap(), q8f, &mut bf, nf, ne, nt);
        let len = nt * nf;
        let bp = bg.as_mut_ptr();
        unsafe {
            crate::vec_ops::vec_silu_f32(len,
                std::slice::from_raw_parts_mut(bp, len),
                std::slice::from_raw_parts(bp as *const f32, len));
        }
        for i in 0..len { bg[i] *= bf[i]; }

        let (q8g, _rest) = qb.split_at_mut(nt * nbf * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bg[..nt * nf], nt, nf, q8g);
        quant_matmul(l.ffn_down.as_ref().unwrap(), q8g, &mut bn, ne, nf, nt);
        for i in 0..hidden.len() { hidden[i] += bn[i]; }
    }

    // 3. Final RMSNorm
    rms_norm(&hidden, hp.f_norm_rms_eps, &mut bn, nt, ne, model.output_norm.as_ref().map(|t| t.data_f32()));

    // 4. LM head
    if let Some(output) = &model.output {
        let (q8e, _rest) = qb.split_at_mut(nt * nbe * Q8B);
        crate::avx2::quantize_row_q8_0_buf(&bn, nt, ne, q8e);
        let mut logits = vec![0.0f32; nt * nv];
        quant_matmul(output, q8e, &mut logits, nv, ne, nt);
        logits
    } else { vec![] }
}

fn embed_tokens(ids: &[u32], t: &crate::tensor::Tensor, out: &mut [f32], ne: usize) {
    match t.ttype {
        TensorType::Q4_0 | TensorType::Q8_0 | TensorType::Q4_1 => {
            let is_q4_1 = t.ttype == TensorType::Q4_1;
            let blk = 32usize;
            let nbp = (ne + blk - 1) / blk;
            let bb = t.ttype.type_size();
            let is8 = t.ttype == TensorType::Q8_0;
            let d_off = if is_q4_1 { 2 } else { 0 };
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                    let m = if is_q4_1 {
                        crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 2], t.data[off + 3]]))
                    } else { 0.0 };
                    let mv = blk.min(ne - b * blk);
                    if is8 {
                        for j in 0..mv { out[doff + b * blk + j] = (t.data[off + 2 + j] as i8) as f32 * d; }
                    } else if is_q4_1 {
                        // Q4_1: value = q * d + m (unsigned nibbles 0..15)
                        for j in (0..mv).step_by(2) {
                            let byte = t.data[off + 4 + j / 2];
                            let lo = (byte & 0x0F) as f32;
                            let hi = (byte >> 4) as f32;
                            out[doff + b * blk + j] = lo * d + m;
                            if j + 1 < mv { out[doff + b * blk + j + 1] = hi * d + m; }
                        }
                    } else {
                        // Q4_0: value = (nibble - 8) * d
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
        TensorType::Q4_K => {
            // Q4_K embedding: ne=1536 → 6 superblocks of 256 elements each
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize;
                let doff = ti * ne;
                    for s in 0..n_super {
                        let off = (idx * n_super + s) * Q4KB;
                        let d    = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                        let dmin = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 2], t.data[off + 3]]));
                        let sc_bytes = <&[u8; 12]>::try_from(&t.data[off + 4..off + 16]).unwrap();
                        let qs       = &t.data[off + 16..off + 144];

                        // Unpack 12 bytes of scales/mins → 4 × uint32_t
                        let mut u: [u32; 4] = [0; 4];
                        unsafe {
                            std::ptr::copy_nonoverlapping(sc_bytes.as_ptr(), u.as_mut_ptr() as *mut u8, 12);
                        }
                        const KMASK1: u32 = 0x3f3f3f3f;
                        const KMASK2: u32 = 0x0f0f0f0f;
                        const KMASK3: u32 = 0x03030303;
                        u[3] = ((u[2] >> 4) & KMASK2) | (((u[1] >> 6) & KMASK3) << 4);
                        let uaux = u[1] & KMASK1;
                        u[1] = (u[2] & KMASK2) | (((u[0] >> 6) & KMASK3) << 4);
                        u[2] = uaux;
                        u[0] &= KMASK1;

                        for sub in 0..8 {
                            let sc_idx = sub / 4;
                            let sc_off = sub % 4;
                            let sc_val = ((u[sc_idx] >> (6 * sc_off)) & 0x3F) as i32;
                            let mm_val = ((u[2 + sc_idx] >> (6 * sc_off)) & 0x3F) as i32;
                        let dl = d * sc_val as f32;
                        let ml = dmin * mm_val as f32;

                        let base = doff + s * 256 + sub * 32;
                        let q4_sub = &qs[sub * 16..];
                        for j in 0..32 {
                            let nib = if j % 2 == 0 {
                                q4_sub[j / 2] & 0x0F
                            } else {
                                q4_sub[j / 2] >> 4
                            };
                            let qval = nib as i8;
                            out[base + j] = dl * qval as f32 - ml;
                        }
                    }
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in embed_tokens", t.ttype),
    }
}

/// 6-bit extract helper (copied from avx2.rs for embedding dequant)
/// Little-endian packing: val0=src[0][5:0], val1=src[0][7:6]|src[1][3:0], ...
#[inline]
fn get6bit_embed(src: &[u8; 3], idx: usize) -> i32 {
    let a = src[0] as i32;
    let b = src[1] as i32;
    let c = src[2] as i32;
    match idx {
        0 => a & 0x3F,
        1 => ((b & 0x0F) << 2) | ((a >> 6) & 0x03),
        2 => ((c & 0x03) << 4) | ((b >> 4) & 0x0F),
        3 => (c >> 2) & 0x3F,
        _ => unreachable!(),
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
