// Compute kernel dispatch layer.
//   CPU (AVX2/scalar) is always available as fallback.
//   MPS (Apple Silicon GPU) is enabled at runtime when Metal is available.

use crate::tensor::{Tensor, TensorType};
use crate::block::{Q4B, Q41B, Q8B, Q4KB, Q6KB};

/// CPU fallback for f32 activation: quantize → call existing dot product.
pub fn cpu_quant_matmul_f32(
    w: &Tensor, x: &[f32], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    let nbe = id / 32;
    let mut qb = vec![0u8; nt * nbe * Q8B];
    crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut qb);
    cpu_quant_matmul(w, &qb, out, od, id, nt)
}

pub fn cpu_quant_matmul(
    w: &Tensor, x: &[u8], out: &mut [f32],
    od: usize, id: usize, nt: usize,
) {
    match w.ttype {
        TensorType::Q4_0 => {
            let nb = id / 32;
            let ws = nb * Q4B;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q4_0_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
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
                    out[t * od + o] = crate::avx2::dot_q4_1_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
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
                    out[t * od + o] = crate::avx2::dot_q4_k_q8_0(
                        wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
                }
            }
        }
        TensorType::Q5_K => {
            let nk = id / 256;
            let ws = nk * 176;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q5_k_q8_0(
                        wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
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
                    out[t * od + o] = crate::avx2::dot_q6_k_q8_0(
                        wrow, &x[t * (id / 32) * Q8B..(t + 1) * (id / 32) * Q8B]);
                }
            }
        }
        TensorType::Q5_0 => {
            let nb = id / 32;
            let ws = nb * 22;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q5_0_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        TensorType::Q5_1 => {
            let nb = id / 32;
            let ws = nb * 24;
            let wb = w.data();
            for o in 0..od {
                let wrow = &wb[o * ws..(o + 1) * ws];
                for t in 0..nt {
                    out[t * od + o] = crate::avx2::dot_q5_1_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
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
                    out[t * od + o] = crate::avx2::dot_q8_0_q8_0(
                        wrow, &x[t * nb * Q8B..(t + 1) * nb * Q8B]);
                }
            }
        }
        _ => panic!("unsupported weight type {:?} in quant_matmul", w.ttype),
    }
}

// Shared embedding row getter (moved from qwen2/forward.rs when the
// imperative forward was replaced by the graph path — Phase 6).
pub fn embed_tokens(ids: &[u32], t: &crate::tensor::Tensor, out: &mut [f32], ne: usize) {
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
            let q5_kb: usize = 176;
            let n_super = (ne + 255) / 256;
            for (ti, &id) in ids.iter().enumerate() {
                let idx = id as usize; let doff = ti * ne;
                for s in 0..n_super {
                    let off = (idx * n_super + s) * q5_kb;
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
                        for j in 0..32 {
                            // Q5_K qh layout: element (sub s, pos j) high bit = qh[j] bit s
                            let hi_bit = ((qh[j] >> sub) & 1) as u8;
                            // Q5_K unsigned (no -16, unlike Q5_0): w = unsigned_5bit * dl - ml
                            let w = nb[sub * 32 + j] as f32 + 16.0 * hi_bit as f32;
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
