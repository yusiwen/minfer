use crate::cache::KVCache;
use crate::tensor::TensorType;
use crate::block::{Q4KB, Q6KB};

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
    let nqt = nh * hd;
    let nkt = nk * hd;
    let eps = hp.f_norm_rms_eps;

    let mut bn = vec![0.0f32; nt * ne];
    let mut bq = vec![0.0f32; nt * nqt];
    let mut bk = vec![0.0f32; nt * nkt];
    let mut bv = vec![0.0f32; nt * nkt];
    let mut ba = vec![0.0f32; nt * ne];
    let mut bf = vec![0.0f32; nt * nf.max(ne)];
    let mut bg = vec![0.0f32; nt * nf];
    let mut hidden = vec![0.0f32; nt * ne];
    let max_seq = hp.max_seq_len as usize;
    let mut scrs_buf = vec![0.0f32; max_seq];

    embed_tokens(token_ids, model.tok_embd.as_ref().unwrap(), &mut hidden, ne);

    // ─── MPS GPU path (hidden stays on GPU across layers) ────────
    #[cfg(target_os = "macos")]
    let use_gpu = if let Some(mps) = crate::metal::MpsState::get() {
        let l0 = &model.layers[0];
        let wq = l0.wq.as_ref().unwrap();
        let wk = l0.wk.as_ref().unwrap();
        let wv = l0.wv.as_ref().unwrap();
        let wo = l0.wo.as_ref().unwrap();
        let fg = l0.ffn_gate.as_ref().unwrap();
        let fu = l0.ffn_up.as_ref().unwrap();
        let fd = l0.ffn_down.as_ref().unwrap();
        mps.has_weight(&wq.name) && mps.has_weight(&wk.name) && mps.has_weight(&wv.name)
            && mps.has_weight(&wo.name) && mps.has_weight(&fg.name)
            && mps.has_weight(&fu.name) && mps.has_weight(&fd.name)
            && l0.attn_norm.as_ref().map_or(false, |t| mps.has_weight(&t.name))
            && l0.ffn_norm.as_ref().map_or(false, |t| mps.has_weight(&t.name))
            && l0.bq.as_ref().map_or(true, |t| mps.has_weight(&t.name))
            && l0.bk.as_ref().map_or(true, |t| mps.has_weight(&t.name))
            && l0.bv.as_ref().map_or(true, |t| mps.has_weight(&t.name))
    } else {
        false
    };
    #[cfg(not(target_os = "macos"))]
    let _use_gpu = false;

    let mut run_cpu = true;

    #[cfg(target_os = "macos")]
    if use_gpu {
        let mps = crate::metal::MpsState::get().unwrap();
        mps.upload_hidden(&hidden);
        mps.upload_positions(positions);
        let cb = mps.cmd_buffer();
        for il in 0..model.n_layer() {
            let l = &model.layers[il];
            if !mps.layer_gpu(&cb, il, l, positions, ne, nqt, nkt, nf, nt, nh, nk, hd, eps, hp.rope_freq_base, hp.rope_freq_scale) {
                eprintln!("layer_gpu returned false at layer {}", il);
                return vec![];
            }
        }
        let gpu_output = mps.output_norm_gpu(
            &cb, model.output.as_ref().unwrap(), model.output_norm.as_ref(),
            model.output_b.as_ref(),
            ne, nv, nt, eps,
        );
        cb.submit();
        if gpu_output {
            let mut logits = vec![0.0f32; nt * nv];
            mps.download_logits(&mut logits);
            return logits;
        }
        mps.download_hidden(&mut hidden);
        run_cpu = false;
    }

    if run_cpu {
        // ─── CPU path ──────────────────────────────────────────────
        for il in 0..model.n_layer() {
            let l = &model.layers[il];
            rms_norm(&hidden, eps, &mut bn, nt, ne, l.attn_norm.as_ref().map(|t| t.data_f32()));
            crate::kernel::quant_matmul_f32_batch(&mut [
                (l.wq.as_ref().unwrap(), &mut bq, nqt),
                (l.wk.as_ref().unwrap(), &mut bk, nkt),
                (l.wv.as_ref().unwrap(), &mut bv, nkt),
            ], &bn, ne, nt);
            if let Some(b) = &l.bq { add_bias(&mut bq, b.data_f32(), nt, nqt); }
            if let Some(b) = &l.bk { add_bias(&mut bk, b.data_f32(), nt, nkt); }
            if let Some(b) = &l.bv { add_bias(&mut bv, b.data_f32(), nt, nkt); }
            apply_rope(&mut bq, positions, nh, hd, hp.rope_freq_base, hp.rope_freq_scale);
            apply_rope(&mut bk, positions, nk, hd, hp.rope_freq_base, hp.rope_freq_scale);
            kv_cache.layers[il].store_multi(positions, &bk, &bv);
            let nkv = kv_cache.layers[il].size;
            gqa_attn(&bq, &kv_cache.layers[il].k[..nkv * nkt], &kv_cache.layers[il].v[..nkv * nkt],
                positions, nt, nkv, nh, nk, hd, &mut ba, &mut scrs_buf[..nkv]);
            crate::kernel::quant_matmul_f32(l.wo.as_ref().unwrap(), &ba, &mut bn, ne, ne, nt);
            unsafe {
                crate::vec_ops::vec_add_f32(hidden.len(),
                    std::slice::from_raw_parts_mut(hidden.as_mut_ptr(), hidden.len()),
                    std::slice::from_raw_parts(hidden.as_ptr(), hidden.len()),
                    &bn);
            }
            rms_norm(&hidden, eps, &mut bf[..nt * ne], nt, ne, l.ffn_norm.as_ref().map(|t| t.data_f32()));
            let ffn_in = bf[..nt * ne].to_vec();
            crate::kernel::quant_matmul_f32_batch(&mut [
                (l.ffn_gate.as_ref().unwrap(), &mut bg, nf),
                (l.ffn_up.as_ref().unwrap(),   &mut bf, nf),
            ], &ffn_in, ne, nt);
            let len = nt * nf;
            unsafe {
                crate::vec_ops::vec_silu_f32(len,
                    std::slice::from_raw_parts_mut(bg.as_mut_ptr(), len),
                    std::slice::from_raw_parts(bg.as_ptr(), len));
                crate::vec_ops::vec_mul_f32(len,
                    std::slice::from_raw_parts_mut(bg.as_mut_ptr(), len),
                    std::slice::from_raw_parts(bg.as_ptr(), len),
                    &bf);
            }
            crate::kernel::quant_matmul_f32(l.ffn_down.as_ref().unwrap(), &bg[..nt * nf], &mut bn, ne, nf, nt);
            unsafe {
                crate::vec_ops::vec_add_f32(hidden.len(),
                    std::slice::from_raw_parts_mut(hidden.as_mut_ptr(), hidden.len()),
                    std::slice::from_raw_parts(hidden.as_ptr(), hidden.len()),
                    &bn);
            }
        }
    }
    rms_norm(&hidden, eps, &mut bn, nt, ne, model.output_norm.as_ref().map(|t| t.data_f32()));
    if let Some(output) = &model.output {
        let mut logits = vec![0.0f32; nt * nv];
        crate::kernel::quant_matmul_f32(output, &bn, &mut logits, nv, ne, nt);
        if let Some(ob) = &model.output_b {
            let b = ob.data_f32();
            for t in 0..nt { let base = t * nv; for i in 0..nv.min(b.len()) { logits[base + i] += b[i]; } }
        }
        return logits;
    }
    vec![]
}

// ─── CPU helpers ────────────────────────────────────────────────────

fn embed_tokens(ids: &[u32], t: &crate::tensor::Tensor, out: &mut [f32], ne: usize) {
    match t.ttype {
        TensorType::Q4_0 | TensorType::Q8_0 | TensorType::Q4_1 => {
            let is_q4_1 = t.ttype == TensorType::Q4_1;
            let blk = 32usize; let nbp = (ne + blk - 1) / blk; let bb = t.ttype.type_size();
            let is8 = t.ttype == TensorType::Q8_0;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                    let m = if is_q4_1 { crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 2], t.data[off + 3]])) } else { 0.0 };
                    let mv = blk.min(ne - b * blk);
                    if is8 { for j in 0..mv { out[doff + b * blk + j] = (t.data[off + 2 + j] as i8) as f32 * d; } }
                    else if is_q4_1 {
                        for j in 0..16 {
                            let byte = t.data[off + 4 + j];
                            if j < mv { out[doff + b * blk + j] = (byte & 0x0F) as f32 * d + m; }
                            if j + 16 < mv { out[doff + b * blk + j + 16] = (byte >> 4) as f32 * d + m; }
                        }
                    } else {
                        for j in 0..16 {
                            let byte = t.data[off + 2 + j];
                            if j < mv { out[doff + b * blk + j] = ((byte & 0x0F) as i8 - 8) as f32 * d; }
                            if j + 16 < mv { out[doff + b * blk + j + 16] = ((byte >> 4) as i8 - 8) as f32 * d; }
                        }
                    }
                }
            }
        }
        TensorType::Q4_K => {
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * Q4KB;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                    let dmin = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 2], t.data[off + 3]]));
                    let sc = <&[u8; 12]>::try_from(&t.data[off + 4..off + 16]).unwrap();
                    let qs = &t.data[off + 16..off + 144];
                    let mut u: [u32; 4] = [0; 4];
                    unsafe { std::ptr::copy_nonoverlapping(sc.as_ptr(), u.as_mut_ptr() as *mut u8, 12); }
                    const KMASK1: u32 = 0x3f3f3f3f; const KMASK2: u32 = 0x0f0f0f0f; const KMASK3: u32 = 0x03030303;
                    u[3] = ((u[2] >> 4) & KMASK2) | (((u[1] >> 6) & KMASK3) << 4);
                    let uaux = u[1] & KMASK1; u[1] = (u[2] & KMASK2) | (((u[0] >> 6) & KMASK3) << 4); u[2] = uaux; u[0] &= KMASK1;
                    for sub in 0..8 {
                        let sc_idx = sub / 4; let sc_off = sub % 4;
                        let sc_val = ((u[sc_idx] >> (6 * sc_off)) & 0x3F) as i32;
                        let mm_val = ((u[2 + sc_idx] >> (6 * sc_off)) & 0x3F) as i32;
                        let dl = d * sc_val as f32; let ml = dmin * mm_val as f32;
                        let base = doff + s * 256 + sub * 32;
                        let q4_sub = &qs[sub * 16..];
                        for j in 0..32 {
                            let nib = if j % 2 == 0 { q4_sub[j / 2] & 0x0F } else { q4_sub[j / 2] >> 4 };
                            out[base + j] = dl * (nib as i8) as f32 - ml;
                        }
                    }
                }
            }
        }
        TensorType::Q6_K => {
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * Q6KB;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 208], t.data[off + 209]]));
                    let mut out_pos = doff + s * 256; let mut ql_pos = off; let mut qh_pos = off + 128; let mut sc_pos = off + 192;
                    for _ in 0..2 {
                        for l in 0..32 {
                            let g = l / 16;
                            let ql0 = t.data[ql_pos + l] as i32; let ql1 = t.data[ql_pos + l + 32] as i32; let qh = t.data[qh_pos + l] as i32;
                            out[out_pos + l + 0]  = d * (t.data[sc_pos + g * 8 + 0] as i8 as f32) * (((ql0 & 0x0F) | ((qh & 3) << 4)) - 32) as f32;
                            out[out_pos + l + 32] = d * (t.data[sc_pos + g * 8 + 2] as i8 as f32) * (((ql1 & 0x0F) | ((qh >> 2) & 3) << 4) - 32) as f32;
                            out[out_pos + l + 64] = d * (t.data[sc_pos + g * 8 + 4] as i8 as f32) * (((ql0 >> 4)   | ((qh >> 4) & 3) << 4) - 32) as f32;
                            out[out_pos + l + 96] = d * (t.data[sc_pos + g * 8 + 6] as i8 as f32) * (((ql1 >> 4)   | ((qh >> 6) & 3) << 4) - 32) as f32;
                        }
                        out_pos += 128; ql_pos += 64; qh_pos += 32; sc_pos += 8;
                    }
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in embed_tokens", t.ttype),
    }
}

fn rms_norm(x: &[f32], eps: f32, out: &mut [f32], n: usize, d: usize, w: Option<&[f32]>) {
    for t in 0..n {
        let row = &x[t * d..(t + 1) * d];
        let dst = &mut out[t * d..(t + 1) * d];
        match w {
            Some(w) => crate::vec_ops::rms_norm_fused_f32(d, dst, row, w, eps),
            None => crate::vec_ops::rms_norm_f32(d, dst, row, eps),
        }
    }
}

fn add_bias(x: &mut [f32], b: &[f32], n: usize, d: usize) {
    for t in 0..n { let base = t * d; for i in 0..d.min(b.len()) { x[base + i] += b[i]; } }
}

fn apply_rope(x: &mut [f32], pos: &[usize], nh: usize, hd: usize, fb: f32, freq_scale: f32) {
    let half = hd / 2;
    let mut freqs = [0.0f32; 64];
    for i in 0..half { freqs[i] = freq_scale / fb.powf((2 * i) as f32 / hd as f32); }
    let mut sin_cache = vec![0.0f32; half];
    let mut cos_cache = vec![0.0f32; half];
    for t in 0..pos.len() {
        let p = pos[t] as f32;
        for i in 0..half {
            let th = p * freqs[i];
            let (sn, cs) = th.sin_cos();
            sin_cache[i] = sn;
            cos_cache[i] = cs;
        }
        for h in 0..nh {
            let b = t * nh * hd + h * hd;
            for i in 0..half {
                let (sn, cs) = (sin_cache[i], cos_cache[i]);
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
    let gqa = nh / nk; let ne_q = nh * hd; let sc = 1.0 / (hd as f32).sqrt();
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qs = t * ne_q + h * hd; let vl = (pos[t] + 1).min(nkv);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks = kv * nk * hd + hk * hd;
                let s = crate::vec_ops::vec_dot_f32(hd, &q[qs..qs + hd], &ka[ks..ks + hd]) * sc;
                scrs[kv] = s; if s > mx { mx = s; }
            }
            for kv in vl..nkv { scrs[kv] = f32::NEG_INFINITY; }
            let sp = scrs.as_mut_ptr();
            let sm = unsafe { crate::vec_ops::vec_soft_max_f32(nkv, std::slice::from_raw_parts_mut(sp, nkv), std::slice::from_raw_parts(sp as *const f32, nkv), mx) };
            let is = (1.0 / sm) as f32; crate::vec_ops::vec_scale_f32(nkv, scrs, is);
            let os = t * ne_q + h * hd; let slice = &mut out[os..os + hd];
            for d in 0..hd { slice[d] = 0.0; }
            let stride = nk * hd; let vs_base = hk * hd;
            for kv in 0..nkv { crate::vec_ops::vec_muladd_f32(hd, slice, &va[kv * stride + vs_base..kv * stride + vs_base + hd], scrs[kv]); }
        }
    }
}
