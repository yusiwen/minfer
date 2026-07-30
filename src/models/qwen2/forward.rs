use crate::cache::KVCache;
use crate::tensor::TensorType;
use crate::block::{Q4KB, Q6KB};
use crate::vec_ops::RopeStyle;

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
    let nkt = hp.n_kv_embd as usize;
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

    crate::dump::maybe_dump_prefill_or_gen0("minfer_dump_embed_out", &hidden, nt);

    let mut run_cpu = true;

    // ─── MPS (Apple Silicon) GPU path ──────────────────────────
    #[cfg(target_os = "macos")]
    {
        let use_gpu = crate::metal::MpsState::get().map_or(false, |mps| {
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
        });
        if use_gpu {
            let mps = crate::metal::MpsState::get().unwrap();
            // GPU embedding lookup (Q4_0): writes directly to buf_hidden on GPU
            let gpu_embd = mps.embed_tokens_gpu(
                model.tok_embd.as_ref().unwrap(), token_ids, nt, ne
            );
            if !gpu_embd {
                // Fallback: upload CPU-computed hidden state
                mps.upload_hidden(&hidden);
            }
            mps.upload_positions(positions);
            let mut cb = mps.cmd_buffer();
            let mut gpu_failed = false;
            for il in 0..model.n_layer() {
                let l = &model.layers[il];
                if !mps.layer_gpu(&cb, il, l, positions, ne, nqt, nkt, nf, nt, nh, nk, hd, eps, hp.attention_scale(), hp.rope_freq_base, hp.rope_freq_scale, hp.rope_style as i32) {
                    gpu_failed = true;
                    break;
                }
                #[cfg(feature = "debug_dump")]
                {
                    cb.submit();
                    mps.download_hidden(&mut hidden);
                    crate::dump::maybe_dump_prefill_or_gen0(
                        &format!("minfer_gpu_dump_layer{}_out", il), &hidden, nt
                    );
                    crate::dump::maybe_dump_prefill_or_gen0(
                        &format!("minfer_gpu_dump_layer{}_attn_out", il), &hidden, nt
                    );
                    cb = mps.cmd_buffer();
                }
            }
            if gpu_failed {
                // GPU path failed — sync KV and fall back to CPU
                mps.sync_kv_to_cpu(kv_cache, model.n_layer());
                // cb dropped without submit; hidden was not uploaded so stays CPU-valid
            } else {
                let gpu_output = mps.output_norm_gpu(
                    &cb, model.output.as_ref().unwrap(), model.output_norm.as_ref(),
                    model.output_b.as_ref(),
                    ne, nv, nt, eps,
                );
                cb.submit();
                if gpu_output {
                    let mut logits = vec![0.0f32; nt * nv];
                    mps.download_logits(&mut logits);
                    mps.sync_kv_to_cpu(kv_cache, model.n_layer());

                    #[cfg(feature = "debug_dump")]
                    {
                        mps.download_hidden(&mut hidden);
                        rms_norm(&hidden, eps, &mut bn, nt, ne, model.output_norm.as_ref().map(|t| t.data_f32()));
                        crate::dump::maybe_dump_prefill_or_gen0("minfer_gpu_dump_last_norm", &bn, nt);
                        crate::dump::maybe_dump_prefill_or_gen0("minfer_gpu_dump_logits", &logits, nt);
                    }

                    return logits;
                }
                mps.download_hidden(&mut hidden);
                mps.sync_kv_to_cpu(kv_cache, model.n_layer());
                run_cpu = false;
            }
        }
    }

    // ─── CUDA (NVIDIA) GPU path ────────────────────────────────
    #[cfg(feature = "cuda")]
    {
        let use_gpu = crate::cuda::CudaState::get().map_or(false, |cuda| {
            model.layers.iter().all(|l| {
                let wq = match &l.wq { Some(t) => t, None => return false };
                let wk = match &l.wk { Some(t) => t, None => return false };
                let wv = match &l.wv { Some(t) => t, None => return false };
                let wo = match &l.wo { Some(t) => t, None => return false };
                let fg = match &l.ffn_gate { Some(t) => t, None => return false };
                let fu = match &l.ffn_up { Some(t) => t, None => return false };
                let fd = match &l.ffn_down { Some(t) => t, None => return false };
                fn is_q4(t: TensorType) -> bool { t == TensorType::Q4_0 || t == TensorType::Q4_1 }
                fn is_qk(t: TensorType) -> bool { t == TensorType::Q4_K || t == TensorType::Q6_K }
                let all_q4 = is_q4(wq.ttype) && is_q4(wk.ttype)
                    && is_q4(wv.ttype) && is_q4(wo.ttype)
                    && is_q4(fg.ttype) && is_q4(fu.ttype)
                    && is_q4(fd.ttype);
                let all_qk = is_qk(wq.ttype) && is_qk(wk.ttype)
                    && is_qk(wv.ttype) && is_qk(wo.ttype)
                    && is_qk(fg.ttype) && is_qk(fu.ttype)
                    && is_qk(fd.ttype);
                (all_q4 || all_qk)
                    && cuda.has_weight(&wq.name) && cuda.has_weight(&wk.name) && cuda.has_weight(&wv.name)
                    && cuda.has_weight(&wo.name) && cuda.has_weight(&fg.name)
                    && cuda.has_weight(&fu.name) && cuda.has_weight(&fd.name)
                    && l.attn_norm.as_ref().map_or(false, |t| cuda.has_weight(&t.name))
                    && l.ffn_norm.as_ref().map_or(false, |t| cuda.has_weight(&t.name))
                    && l.bq.as_ref().map_or(true, |t| cuda.has_weight(&t.name))
                    && l.bk.as_ref().map_or(true, |t| cuda.has_weight(&t.name))
                    && l.bv.as_ref().map_or(true, |t| cuda.has_weight(&t.name))
            })
        });
        if use_gpu {
            let cuda = crate::cuda::CudaState::get().unwrap();

            // Fast path: replay captured decode graph
            if nt == 1 && cuda.graph_available() {
                cuda.upload_hidden(&hidden);
                cuda.upload_positions(positions);
                cuda.graph_launch();
                cuda.sync();
                let mut logits = vec![0.0f32; nt * nv];
                cuda.download_logits(&mut logits);
                return logits;
            }

            cuda.upload_hidden(&hidden);
            cuda.upload_positions(positions);

            let capture = nt == 1 && !cuda.graph_available() && cuda.graph_begin_capture();

            run_cpu = false; // assume GPU path succeeds; reset on failure below

            for il in 0..model.n_layer() {
                let l = &model.layers[il];
                if !cuda.layer_gpu(il, l, positions, ne, nqt, nkt, nf, nt, nh, nk, hd, eps, hp.rope_freq_base, hp.rope_freq_scale) {
                    eprintln!("layer_gpu returned false at layer {} — falling back to CPU for all layers", il);
                    cuda.sync();
                    run_cpu = true;
                    break;
                }
            }

            if !run_cpu {
                let gpu_output = cuda.output_norm_gpu(
                    model.output.as_ref().unwrap(), model.output_norm.as_ref(),
                    model.output_b.as_ref(),
                    ne, nv, nt, eps,
                );

                if capture {
                    cuda.graph_end_capture();
                    cuda.graph_launch();
                }

                cuda.sync();
                if gpu_output {
                    let mut logits = vec![0.0f32; nt * nv];
                    cuda.download_logits(&mut logits);
                    return logits;
                }
                cuda.download_hidden(&mut hidden);
                run_cpu = false;
            }
        }
    }

    if run_cpu {
        // ─── CPU path ──────────────────────────────────────────────
        for il in 0..model.n_layer() {
            let l = &model.layers[il];
            rms_norm(&hidden, eps, &mut bn, nt, ne, l.attn_norm.as_ref().map(|t| t.data_f32()));
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_bn", &bn, nt, il == 0);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_bn", &bn, nt, il == 2);
            crate::kernel::quant_matmul_f32_batch(&mut [
                (l.wq.as_ref().unwrap(), &mut bq, nqt),
                (l.wk.as_ref().unwrap(), &mut bk, nkt),
                (l.wv.as_ref().unwrap(), &mut bv, nkt),
            ], &bn, ne, nt);
            if let Some(b) = &l.bq { add_bias(&mut bq, b.data_f32(), nt, nqt); }
            if let Some(b) = &l.bk { add_bias(&mut bk, b.data_f32(), nt, nkt); }
            if let Some(b) = &l.bv { add_bias(&mut bv, b.data_f32(), nt, nkt); }
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_bq", &bq, nt, il == 0);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_bq", &bq, nt, il == 2);
            apply_rope(&mut bq, positions, nh, hd, hp.rope_freq_base, hp.rope_freq_scale, hp.rope_style);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_bq_rope", &bq, nt, il == 0);
            apply_rope(&mut bk, positions, nk, hd, hp.rope_freq_base, hp.rope_freq_scale, hp.rope_style);
            kv_cache.layers[il].store_multi(positions, &bk, &bv);
            let nkv = kv_cache.layers[il].size;
            let hd_kv = nkt / nk;
            gqa_attn(&bq, &kv_cache.layers[il].k[..nkv * nkt], &kv_cache.layers[il].v[..nkv * nkt],
                positions, nt, nkv, nh, nk, hd, hd_kv, nkt, &mut ba, &mut scrs_buf[..nkv], hp.attention_scale());
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_ba", &ba, nt, il == 0);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_ba", &ba, nt, il == 2);
            crate::kernel::quant_matmul_f32(l.wo.as_ref().unwrap(), &ba, &mut bn, ne, ne, nt);
            unsafe {
                crate::vec_ops::vec_add_f32(hidden.len(),
                    std::slice::from_raw_parts_mut(hidden.as_mut_ptr(), hidden.len()),
                    std::slice::from_raw_parts(hidden.as_ptr(), hidden.len()),
                    &bn);
            }
        crate::dump::maybe_dump_prefill_or_gen0(&format!("minfer_dump_layer{}_attn_out", il), &hidden, nt);
            rms_norm(&hidden, eps, &mut bf[..nt * ne], nt, ne, l.ffn_norm.as_ref().map(|t| t.data_f32()));
            let ffn_in = bf[..nt * ne].to_vec();
            crate::kernel::quant_matmul_f32_batch(&mut [
                (l.ffn_gate.as_ref().unwrap(), &mut bg, nf),
                (l.ffn_up.as_ref().unwrap(),   &mut bf, nf),
            ], &ffn_in, ne, nt);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_bg", &bg, nt, il == 0);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_bg", &bg, nt, il == 2);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_bf", &bf, nt, il == 2);
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
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_swiglu", &bg, nt, il == 0);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_swiglu", &bg, nt, il == 2);
            crate::kernel::quant_matmul_f32(l.ffn_down.as_ref().unwrap(), &bg[..nt * nf], &mut bn, ne, nf, nt);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer0_fd", &bn, nt, il == 0);
            crate::dump::maybe_dump_prefill_or_gen0_if("minfer_dump_layer2_fd", &bn, nt, il == 2);
            unsafe {
                crate::vec_ops::vec_add_f32(hidden.len(),
                    std::slice::from_raw_parts_mut(hidden.as_mut_ptr(), hidden.len()),
                    std::slice::from_raw_parts(hidden.as_ptr(), hidden.len()),
                    &bn);
            }
        crate::dump::maybe_dump_prefill_or_gen0(&format!("minfer_dump_layer{}_out", il), &hidden, nt);
        }
    }
    rms_norm(&hidden, eps, &mut bn, nt, ne, model.output_norm.as_ref().map(|t| t.data_f32()));
    crate::dump::maybe_dump_prefill_or_gen0("minfer_dump_last_norm", &bn, nt);
    if let Some(output) = &model.output {
        let mut logits = vec![0.0f32; nt * nv];
        crate::kernel::quant_matmul_f32(output, &bn, &mut logits, nv, ne, nt);
        if let Some(ob) = &model.output_b {
            let b = ob.data_f32();
            for t in 0..nt { let base = t * nv; for i in 0..nv.min(b.len()) { logits[base + i] += b[i]; } }
        }
        crate::dump::maybe_dump_prefill_or_gen0("minfer_dump_logits", &logits, nt);
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
        TensorType::Q5_0 => {
            let blk = 32usize; let nbp = (ne + blk - 1) / blk; let bb = 22usize;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                    let qh = u32::from_le_bytes([t.data[off+2], t.data[off+3], t.data[off+4], t.data[off+5]]);
                    let qs = &t.data[off + 6..off + 22];
                    let mv = blk.min(ne - b * blk);
                    for j in 0..16 {
                        let xh0 = ((qh >> j) & 1) as u32;
                        let xh1 = ((qh >> (j + 16)) & 1) as u32;
                        if j < mv {
                            out[doff + b * blk + j] = (((qs[j] & 0x0F) as i32 | ((xh0 << 4) as i32)) - 16) as f32 * d;
                        }
                        if j + 16 < mv {
                            out[doff + b * blk + j + 16] = ((((qs[j] >> 4) & 0x0F) as i32 | ((xh1 << 4) as i32)) - 16) as f32 * d;
                        }
                    }
                }
            }
        }
        TensorType::Q5_1 => {
            let blk = 32usize; let nbp = (ne + blk - 1) / blk; let bb = 24usize;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for b in 0..nbp {
                    let off = (idx * nbp + b) * bb;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                    let m = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 2], t.data[off + 3]]));
                    let qh = u32::from_le_bytes([t.data[off+4], t.data[off+5], t.data[off+6], t.data[off+7]]);
                    let qs = &t.data[off + 8..off + 24];
                    let mv = blk.min(ne - b * blk);
                    for j in 0..16 {
                        let xh0 = ((qh >> j) & 1) as u32;
                        let xh1 = ((qh >> (j + 16)) & 1) as u32;
                        if j < mv {
                            out[doff + b * blk + j] = ((qs[j] & 0x0F) as i32 | ((xh0 << 4) as i32)) as f32 * d + m;
                        }
                        if j + 16 < mv {
                            out[doff + b * blk + j + 16] = (((qs[j] >> 4) & 0x0F) as i32 | ((xh1 << 4) as i32)) as f32 * d + m;
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
                    let sc_arr: &[u8; 12] = t.data[off + 4..off + 16].try_into().unwrap();
                    let (scales, mins) = crate::block::unpack_q4k_scales(sc_arr);
                    let qs = &t.data[off + 16..off + 144];

                    // Deinterleave qs: 4 chunks of 32 bytes, each covers 2 subblocks
                    // chunk[l] lo nibble → sub 2*chunk, elem l
                    // chunk[l] hi nibble → sub 2*chunk+1, elem l
                    let mut nibbles = [0i32; 256];
                    for chunk_idx in 0..4 {
                        let chunk = &qs[chunk_idx * 32..chunk_idx * 32 + 32];
                        for l in 0..32 {
                            nibbles[(2 * chunk_idx) * 32 + l] = (chunk[l] & 0x0F) as i32;
                            nibbles[(2 * chunk_idx + 1) * 32 + l] = (chunk[l] >> 4) as i32;
                        }
                    }

                    for sub in 0..8 {
                        let sc_val = scales[sub];
                        let mm_val = mins[sub];
                        let dl = d * sc_val as f32; let ml = dmin * mm_val as f32;
                        let base = doff + s * 256 + sub * 32;
                        for k in 0..32 {
                            out[base + k] = dl * nibbles[sub * 32 + k] as f32 - ml;
                        }
                    }
                }
            }
        }
        TensorType::Q5_K => {
            let Q5KB: usize = 176;
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * Q5KB;
                    let d = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off], t.data[off + 1]]));
                    let dmin = crate::block::fp16_to_f32(u16::from_le_bytes([t.data[off + 2], t.data[off + 3]]));
                    let sc_arr: &[u8; 12] = t.data[off + 4..off + 16].try_into().unwrap();
                    let (scales, mins) = crate::block::unpack_q4k_scales(sc_arr);
                    let qh = &t.data[off + 16..off + 48];
                    let qs = &t.data[off + 48..off + 176];

                    // Deinterleave qs nibbles: 4 chunks of 32 bytes, covering 2 subblocks each
                    let mut nb = [0u8; 256];
                    for ci in 0..4 {
                        let chunk = &qs[ci * 32..ci * 32 + 32];
                        for l in 0..32 {
                            nb[(2 * ci) * 32 + l] = chunk[l] & 0x0F;
                            nb[(2 * ci + 1) * 32 + l] = chunk[l] >> 4;
                        }
                    }

                    for sub in 0..8 {
                        let dl = d * scales[sub] as f32;
                        let ml = dmin * mins[sub] as f32;
                        let base = doff + s * 256 + sub * 32;
                        let h0_base = sub * 4;
                        for j in 0..32 {
                            let hidx = h0_base + j / 8;
                            let shift = j % 8;
                            let hi_bit = ((qh[hidx + if j < 16 { 0 } else { 2 }] >> shift) & 1) as u8;
                            // Q5_K signed: (unsigned_5bit - 16) * dl - ml
                            let w = nb[sub * 32 + j] as f32 + 16.0 * hi_bit as f32 - 16.0;
                            out[base + j] = dl * w - ml;
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
                    let base_out = doff + s * 256;

                    let ql = &t.data[off..off + 128];
                    let qh = &t.data[off + 128..off + 192];
                    let sc = &t.data[off + 192..off + 208];

                    for n in 0..2 {
                        let ql_off = n * 64;
                        let qh_off = n * 32;
                        let out_off = n * 128;
                        for l in 0..32 {
                            let is = l / 16;
                            let si = is + n * 8;

                            let q0 = (((ql[ql_off + l] & 0xF) as i32) | ((((qh[qh_off + l] >> 0) & 3) as i32) << 4)) - 32;
                            let q1 = (((ql[ql_off + l + 32] & 0xF) as i32) | ((((qh[qh_off + l] >> 2) & 3) as i32) << 4)) - 32;
                            let q2 = (((ql[ql_off + l] >> 4) as i32) | ((((qh[qh_off + l] >> 4) & 3) as i32) << 4)) - 32;
                            let q3 = (((ql[ql_off + l + 32] >> 4) as i32) | ((((qh[qh_off + l] >> 6) & 3) as i32) << 4)) - 32;

                            out[base_out + out_off + l]      = d * (sc[si + 0] as i8 as f32) * q0 as f32;
                            out[base_out + out_off + l + 32] = d * (sc[si + 2] as i8 as f32) * q1 as f32;
                            out[base_out + out_off + l + 64] = d * (sc[si + 4] as i8 as f32) * q2 as f32;
                            out[base_out + out_off + l + 96] = d * (sc[si + 6] as i8 as f32) * q3 as f32;
                        }
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

fn apply_rope(x: &mut [f32], pos: &[usize], nh: usize, hd: usize, fb: f32, freq_scale: f32, style: RopeStyle) {
    let half = hd / 2;
    let mut freqs = [0.0f32; 128];
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
                let (i0, i1) = match style {
                    RopeStyle::NonInterleaved => (b + i, b + i + half),
                    RopeStyle::Interleaved => (b + 2 * i, b + 2 * i + 1),
                };
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * cs - x1 * sn;
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }
}

fn gqa_attn(q: &[f32], ka: &[f32], va: &[f32], pos: &[usize], nt: usize, nkv: usize,
    nh: usize, nk: usize, hd: usize, hd_kv: usize, nkt: usize, out: &mut [f32], scrs: &mut [f32], scale: f32) {
    let gqa = nh / nk; let ne_q = nh * hd;
    assert!(hd >= hd_kv, "Q head dim ({}) must be >= KV head dim ({})", hd, hd_kv);
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qs = t * ne_q + h * hd; let vl = (pos[t] + 1).min(nkv);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks = kv * nkt + hk * hd_kv;
                let s = crate::vec_ops::vec_dot_f32(hd_kv, &q[qs..qs + hd_kv], &ka[ks..ks + hd_kv]) * scale;
                scrs[kv] = s; if s > mx { mx = s; }
            }
            for kv in vl..nkv { scrs[kv] = f32::NEG_INFINITY; }
            let sp = scrs.as_mut_ptr();
            let sm = unsafe { crate::vec_ops::vec_soft_max_f32(nkv, std::slice::from_raw_parts_mut(sp, nkv), std::slice::from_raw_parts(sp as *const f32, nkv), mx) };
            let is = (1.0 / sm) as f32; crate::vec_ops::vec_scale_f32(nkv, scrs, is);
            let os = t * ne_q + h * hd; let slice = &mut out[os..os + hd];
            for d in 0..hd { slice[d] = 0.0; }
            let vs_base = hk * hd_kv;
            for kv in 0..nkv { crate::vec_ops::vec_muladd_f32(hd_kv, &mut slice[..hd_kv], &va[kv * nkt + vs_base..kv * nkt + vs_base + hd_kv], scrs[kv]); }
        }
    }
}
