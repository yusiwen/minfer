//! Dense and quantized matmul parity (Q4_K/Q5_K/Q6_K/Q5_0, multi-token and fused FFN).
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn cuda_matmul_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (od, id_, nt) = (32usize, 64usize, 3usize);
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
        .collect();
    let bias: Vec<f32> = (0..od).map(|i| (i % 5) as f32 / 10.0).collect();

    // Q8_0 weight [out][in] row-major, quantized per row
    let wf8: Vec<f32> = (0..od * id_)
        .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
        .collect();
    let mut w8b = Vec::new();
    for r in 0..od {
        w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
            &wf8[r * id_..(r + 1) * id_],
        ));
    }
    let mut w8 = Tensor::from_data(
        TensorType::Q8_0,
        &[id_ as i64, od as i64, 1, 1],
        w8b.clone(),
    );
    w8.name = "mw8".to_string();
    cb.state.register_weight("mw8", &w8b);
    let biasb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
    cb.state.register_weight("mb", &biasb);
    let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], biasb);
    bt.name = "mb".to_string();

    // Q4_0 weight (18 bytes per 32 values: f16 d + 16 nibbles)
    let wf4: Vec<f32> = (0..od * id_)
        .map(|i| ((i * 40503) % 991) as f32 / 496.0 - 1.0)
        .collect();
    let mut w4b = Vec::new();
    for r in 0..od {
        let row = &wf4[r * id_..(r + 1) * id_];
        for bi in 0..id_ / 32 {
            let blk = &row[bi * 32..bi * 32 + 32];
            let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let dsc = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
            w4b.extend_from_slice(&half::f16::from_f32(dsc).to_le_bytes());
            for j in 0..16 {
                let q0 = ((blk[j] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                let q1 = ((blk[j + 16] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                w4b.push(q0 | (q1 << 4));
            }
        }
    }
    let mut w4 = Tensor::from_data(
        TensorType::Q4_0,
        &[id_ as i64, od as i64, 1, 1],
        w4b.clone(),
    );
    w4.name = "mw4".to_string();
    cb.state.register_weight("mw4", &w4b);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let m8 = b.matmul(x, &w8, Some(&bt));
    let m4 = b.matmul(x, &w4, Some(&bt));
    b.output(m8);
    b.output(m4);
    let g = b.build();

    let xb = cb.alloc_buffer(id_ * nt);
    cb.write_host(xb, &xs).unwrap();
    let (o8, o4) = (cb.alloc_buffer(od * nt), cb.alloc_buffer(od * nt));
    cb.exec_ids(&g.nodes[m8], &[xb], o8, None).unwrap();
    cb.exec_ids(&g.nodes[m4], &[xb], o4, None).unwrap();

    // References: dequantized weight rows × f32 activations + bias
    // (embed_tokens doubles as the row dequantizer for these types).
    let mut dq8 = vec![0f32; od * id_];
    crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8, &mut dq8, id_);
    let mut dq4 = vec![0f32; od * id_];
    crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w4, &mut dq4, id_);
    for (name, o, dq, quantize_acts) in [
        ("q8_0 matmul", o8, &dq8, false),
        // 8c: prefill Q4_0 runs the Q8_0-activation GEMM (the CPU path
        // has always quantized activations too) — mirror that in the
        // reference and keep a tight tolerance.
        ("q4_0 matmul", o4, &dq4, true),
    ] {
        let got = cb.copy_to_host(o).unwrap();
        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            let xrow = &xs[t * id_..(t + 1) * id_];
            let acts: Vec<f32> = if quantize_acts {
                let q8 = crate::quants::quantize_row_q8_0(xrow);
                (0..id_)
                    .map(|i| {
                        let b = i / 32;
                        let d8 = half::f16::from_le_bytes([q8[b * 34], q8[b * 34 + 1]]).to_f32();
                        d8 * q8[b * 34 + 2 + (i % 32)] as i8 as f32
                    })
                    .collect()
            } else {
                xrow.to_vec()
            };
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += dq[r * id_ + i] * acts[i];
                }
                want[t * od + r] = acc + bias[r];
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(name, &got, &want, scale * 2e-2);
    }
}
/// 7e②: K-quant matmul parity (Q4_K + Q6_K). The reference dequantizes
/// each row with an independent in-test implementation of the
/// llama.cpp block layout and dots it with the f32 activations. The
/// original scalar CUDA kernels and the 7e② vectorized ones must both
/// agree with it (coverage gap found in 7e②: q6_K previously had NO
/// parity test, which let a broken vectorized variant pass the suite).
#[test]
fn cuda_kquant_matmul_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // id_ = 512 = 2 super-blocks of 256; od = 8 rows (NR0 = 2 → 4 row
    // pairs across 2 warps per block).
    let (od, id_, nt) = (8usize, 512usize, 3usize);
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
        .collect();

    // get_scale_min_k4 (llama.cpp Q4_K scale packing, reimplemented
    // here independently of the kernel under test).
    fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }

    // ── Q4_K tensor: 144 bytes per 256-element super-block ──
    // layout: f16 d, f16 dmin, u8 scales[12], nibble bytes qs[128]
    let mut w4b = Vec::new();
    let mut w4dq = vec![0f32; od * id_];
    for r in 0..od {
        for ib in 0..id_ / 256 {
            let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
            let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
            w4b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            w4b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            let mut scb = [0u8; 12];
            for j in 0..12 {
                scb[j] = ((r * 31 + j * 17 + ib * 5) % 63) as u8;
            }
            w4b.extend_from_slice(&scb);
            let mut qs = [0u8; 128];
            for j in 0..128 {
                let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                qs[j] = lo | (hi << 4);
            }
            w4b.extend_from_slice(&qs);
            // reference dequant: LOW nibbles of bytes[32j..32j+31] are
            // elements [64j..64j+31] (scale 2j), HIGH nibbles are
            // elements [64j+32..64j+63] (scale 2j+1);
            // value = d*sc*nibble - dmin*m
            for j in 0..4 {
                let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                for l in 0..32 {
                    let b = qs[j * 32 + l];
                    let base = r * id_ + ib * 256 + j * 64;
                    w4dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                    w4dq[base + 32 + l] = (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                }
            }
        }
    }

    // ── Q6_K tensor: 210 bytes per 256-element super-block ──
    // layout: ql[128], qh[64], i8 scales[16], f16 d
    let mut w6b = Vec::new();
    let mut w6dq = vec![0f32; od * id_];
    for r in 0..od {
        for ib in 0..id_ / 256 {
            let d = 0.027f32 + 0.004 * ((r * 11 + ib * 7) % 6) as f32;
            let mut ql = [0u8; 128];
            let mut qh = [0u8; 64];
            let mut sc = [0i8; 16];
            for i in 0..128 {
                ql[i] = ((r * 29 + i * 7 + ib * 3) % 255) as u8;
            }
            for i in 0..64 {
                qh[i] = ((r * 17 + i * 13 + ib * 11) % 255) as u8;
            }
            for i in 0..16 {
                sc[i] = (((r * 5 + i * 3 + ib) % 15) as i8) - 7;
            }
            w6b.extend_from_slice(&ql);
            w6b.extend_from_slice(&qh);
            w6b.extend(sc.iter().map(|&x| x as u8));
            w6b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            // reference dequant (llama.cpp Q6_K layout):
            // value = d * sc[n*8 + l/16 + t*2] * (nibble|2bits<<4 - 32)
            for n in 0..2usize {
                let qlh = &ql[n * 64..n * 64 + 64];
                let qhh = &qh[n * 32..n * 32 + 32];
                for l in 0..32usize {
                    let is = l / 16;
                    let q1 = ((qlh[l] & 0xF) as i32 | (((qhh[l] >> 0) as i32 & 3) << 4)) - 32;
                    let q2 = ((qlh[l + 32] & 0xF) as i32 | (((qhh[l] >> 2) as i32 & 3) << 4)) - 32;
                    let q3 = ((qlh[l] >> 4) as i32 | (((qhh[l] >> 4) as i32 & 3) << 4)) - 32;
                    let q4 = ((qlh[l + 32] >> 4) as i32 | (((qhh[l] >> 6) as i32 & 3) << 4)) - 32;
                    let base = r * id_ + ib * 256 + n * 128;
                    w6dq[base + l] = d * sc[n * 8 + is] as f32 * q1 as f32;
                    w6dq[base + l + 32] = d * sc[n * 8 + is + 2] as f32 * q2 as f32;
                    w6dq[base + l + 64] = d * sc[n * 8 + is + 4] as f32 * q3 as f32;
                    w6dq[base + l + 96] = d * sc[n * 8 + is + 6] as f32 * q4 as f32;
                }
            }
        }
    }

    let mut w4t = Tensor::from_data(
        TensorType::Q4_K,
        &[id_ as i64, od as i64, 1, 1],
        w4b.clone(),
    );
    w4t.name = "mw4k".to_string();
    cb.state.register_weight("mw4k", &w4b);
    let mut w6t = Tensor::from_data(
        TensorType::Q6_K,
        &[id_ as i64, od as i64, 1, 1],
        w6b.clone(),
    );
    w6t.name = "mw6k".to_string();
    cb.state.register_weight("mw6k", &w6b);
    // 7e② padded layout path (register_weight_q6k_padded)
    cb.state.register_weight_q6k_padded("mw6kp", &w6b, od, id_);
    assert!(cb.state.is_weight_padded("mw6kp"));

    let mut w6pt = Tensor::from_data(
        TensorType::Q6_K,
        &[id_ as i64, od as i64, 1, 1],
        w6b.clone(),
    );
    w6pt.name = "mw6kp".to_string();

    // ── F32 weight (7e④): aligned id (512) and odd id (513, scalar path)
    let wfb: Vec<u8> = w4dq.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut wft = Tensor::from_data(TensorType::F32, &[id_ as i64, od as i64, 1, 1], wfb.clone());
    wft.name = "mwf32".to_string();
    cb.state.register_weight("mwf32", &wfb);
    // odd id: 513-wide rows (first 512 = w4dq, element 512 synthetic)
    let (od_o, id_o) = (8usize, 513usize);
    let mut wfo_vals = Vec::with_capacity(od_o * id_o);
    for r in 0..od_o {
        for i in 0..id_o {
            wfo_vals.push(if i < id_ {
                w4dq[r * id_ + i]
            } else {
                (r + 1) as f32 * 0.25
            });
        }
    }
    let wfo: Vec<u8> = wfo_vals.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut wfot = Tensor::from_data(
        TensorType::F32,
        &[id_o as i64, od_o as i64, 1, 1],
        wfo.clone(),
    );
    wfot.name = "mwf32o".to_string();
    cb.state.register_weight("mwf32o", &wfo);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let m4 = b.matmul(x, &w4t, None);
    let m6 = b.matmul(x, &w6t, None);
    let m6p = b.matmul(x, &w6pt, None);
    let mf = b.matmul(x, &wft, None);
    b.output(m4);
    b.output(m6);
    b.output(m6p);
    b.output(mf);
    // odd-id graph: x sliced to id_ = 513
    let xo = b.input("xo", [id_o, nt, 1, 1], DType::F32);
    let mfo = b.matmul(xo, &wfot, None);
    b.output(mfo);
    let g = b.build();

    let xb = cb.alloc_buffer(id_ * nt);
    cb.write_host(xb, &xs).unwrap();
    let (o4, o6, o6p) = (
        cb.alloc_buffer(od * nt),
        cb.alloc_buffer(od * nt),
        cb.alloc_buffer(od * nt),
    );
    let of = cb.alloc_buffer(od * nt);
    cb.exec_ids(&g.nodes[m4], &[xb], o4, None).unwrap();
    cb.exec_ids(&g.nodes[m6], &[xb], o6, None).unwrap();
    cb.exec_ids(&g.nodes[m6p], &[xb], o6p, None).unwrap();
    cb.exec_ids(&g.nodes[mf], &[xb], of, None).unwrap();
    let mut xso = xs.clone();
    xso.resize(id_o * nt, 0.25f32); // extend for the odd-id input
    let xob = cb.alloc_buffer(id_o * nt);
    cb.write_host(xob, &xso).unwrap();
    let ofo = cb.alloc_buffer(od_o * nt);
    cb.exec_ids(&g.nodes[mfo], &[xob], ofo, None).unwrap();

    for (name, o, dq) in [
        ("q4_k matmul", o4, &w4dq),
        ("q6_k matmul", o6, &w6dq),
        ("q6_k padded matmul", o6p, &w6dq),
    ] {
        let got = cb.copy_to_host(o).unwrap();
        // Step 82: at nt in [2, 8] the q4_K/q6_K arms dispatch to the
        // multi-token MMVQ kernels, whose activations are the pad40 q8
        // plane (f16 d + 2B pad + 32 i8 per 32-element block, per
        // token) — the reference dots the dequantized q8 values, with
        // the same 1e-2 relative tolerance the mmvq parity tests use
        // for the kernel-side quantization rounding.
        let mut x8 = vec![0u8; nt * (id_ / 32) * 40];
        for t in 0..nt {
            for blk in 0..id_ / 32 {
                let base = t * id_ + blk * 32;
                let mut am = 0f32;
                for j in 0..32 {
                    am = am.max(xs[base + j].abs());
                }
                let dd = am / 127.0;
                let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                let off = (t * (id_ / 32) + blk) * 40;
                x8[off..off + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                for j in 0..32 {
                    let q = (xs[base + j] * di).round().clamp(-128.0, 127.0) as i8;
                    x8[off + 4 + j] = q as u8;
                }
            }
        }
        let dq8 = |t: usize, i: usize| -> f32 {
            let off = (t * (id_ / 32) + i / 32) * 40;
            half::f16::from_le_bytes([x8[off], x8[off + 1]]).to_f32()
                * (x8[off + 4 + (i % 32)] as i8) as f32
        };
        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += dq[r * id_ + i] * dq8(t, i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(name, &got, &want, scale * 1e-2);
    }

    // F32 matmul (aligned + odd-id scalar path) vs the same reference rows
    {
        let got = cb.copy_to_host(of).unwrap();
        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w4dq[r * id_ + i] * xs[t * id_ + i];
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("f32 matmul", &got, &want, scale * 2e-3);
    }
    {
        let got = cb.copy_to_host(ofo).unwrap();
        let mut want = vec![0f32; od_o * nt];
        for t in 0..nt {
            for r in 0..od_o {
                let mut acc = 0f32;
                for i in 0..id_o {
                    acc += wfo_vals[r * id_o + i] * xso[t * id_o + i];
                }
                want[t * od_o + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("f32 matmul odd id", &got, &want, scale * 2e-3);
    }
}
#[test]
fn cuda_multi_token_matmul_bitwise() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    // doc 95: sweep the verify depths the adaptive controller may pick —
    // the identity needs multi-MMVQ bitwise-equal to single at every nt,
    // not just the historical nt=3 probe.
    for nt in [3usize, 5, 8] {
        fn gen_f32(n: usize, seed: u64) -> Vec<f32> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                    let mag = if (s >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                    (u as f32) * mag
                })
                .collect()
        }

        fn gen_bytes(n: usize, seed: u64) -> Vec<u8> {
            let mut s = seed;
            (0..n)
                .map(|_| {
                    s = s
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    (s >> 33) as u8
                })
                .collect()
        }

        // (label, type, od, id, q6_padded). Weight-byte lengths per type:
        // K-quants ceil(id/256) blocks per row (144/176/210 B), the rest
        // id/32 blocks per row (18/20/22/24/34 B), f32 raw.
        let cases: Vec<(&str, TensorType, usize, usize, bool)> = vec![
            ("q4k_v2", TensorType::Q4_K, 2048, 3584, false),
            ("q4k_v1", TensorType::Q4_K, 512, 3904, false),
            ("q5k_v2", TensorType::Q5_K, 8192, 3072, false),
            ("q5k_v1", TensorType::Q5_K, 8192, 3104, false),
            ("q6k_padded", TensorType::Q6_K, 2048, 2048, true),
            ("q6k_raw", TensorType::Q6_K, 2048, 2048, false),
            ("q8_0", TensorType::Q8_0, 512, 2048, false),
            ("q4_0_big", TensorType::Q4_0, 512, 9216, false),
            // 14B decode/verify shapes (doc 94): the identity chain needs
            // single-MMVQ (nt=1) == multi-MMVQ (nt 2..8) bitwise at the real
            // Qwen2.5-14B q4_k_m dims, not just the small fixtures.
            ("q4k_14b_attn", TensorType::Q4_K, 5120, 5120, false),
            ("q4k_14b_gu", TensorType::Q4_K, 13824, 5120, false),
            ("q6k_14b_down", TensorType::Q6_K, 5120, 13824, false),
            ("q4_1", TensorType::Q4_1, 512, 2048, false),
            ("q5_0", TensorType::Q5_0, 512, 2048, false),
            ("q5_1", TensorType::Q5_1, 512, 2048, false),
            ("f32", TensorType::F32, 512, 2048, false),
        ];

        for (i, (label, tt, od, id_, padded)) in cases.into_iter().enumerate() {
            let nbe = (id_ + 255) / 256;
            let row_bytes = match tt {
                TensorType::Q4_K => nbe * 144,
                TensorType::Q5_K => nbe * 176,
                TensorType::Q6_K => nbe * 210,
                TensorType::Q8_0 => (id_ / 32) * 34,
                TensorType::Q4_0 => (id_ / 32) * 18,
                TensorType::Q4_1 => (id_ / 32) * 20,
                TensorType::Q5_0 => (id_ / 32) * 22,
                TensorType::Q5_1 => (id_ / 32) * 24,
                TensorType::F32 => id_ * 4,
                other => panic!("unexpected type {other:?}"),
            };
            let wb = gen_bytes(od * row_bytes, 0x5EED_0000 + i as u64);
            let xs = gen_f32(id_ * nt, 0xA11C_0000 + i as u64);

            let wt_name = format!("wbit{i}");
            let mut wt = Tensor::from_data(tt, &[id_ as i64, od as i64, 1, 1], wb.clone());
            wt.name = wt_name.clone();
            if tt == TensorType::Q6_K && padded {
                cb.state.register_weight_q6k_padded(&wt_name, &wb, od, id_);
            } else {
                cb.state.register_weight(&wt_name, &wb);
            }

            // batched: one nt = 3 forward
            let mut b = GraphBuilder::new();
            let x = b.input("x", [id_, nt, 1, 1], DType::F32);
            let m = b.matmul(x, &wt, None);
            b.output(m);
            let g = b.build();
            let xb = cb.alloc_buffer(id_ * nt);
            cb.write_host(xb, &xs).unwrap();
            let ob = cb.alloc_buffer(od * nt);
            cb.exec_ids(&g.nodes[m], &[xb], ob, None).unwrap();
            let got = cb.copy_to_host(ob).unwrap();

            // reference: nt separate nt = 1 forwards over the same weights
            let mut refs: Vec<Vec<f32>> = Vec::with_capacity(nt);
            for t in 0..nt {
                let mut b1 = GraphBuilder::new();
                let x1 = b1.input("x1", [id_, 1, 1, 1], DType::F32);
                let m1 = b1.matmul(x1, &wt, None);
                b1.output(m1);
                let g1 = b1.build();
                let xb1 = cb.alloc_buffer(id_);
                cb.write_host(xb1, &xs[t * id_..(t + 1) * id_]).unwrap();
                let ob1 = cb.alloc_buffer(od);
                cb.exec_ids(&g1.nodes[m1], &[xb1], ob1, None).unwrap();
                refs.push(cb.copy_to_host(ob1).unwrap());
            }

            for t in 0..nt {
                for (r, (a, bref)) in got[t * od..(t + 1) * od]
                    .iter()
                    .zip(refs[t].iter())
                    .enumerate()
                {
                    assert_eq!(
                        a.to_bits(),
                        bref.to_bits(),
                        "{label} token {t} row {r}: batched nt={nt} vs single nt=1 mismatch"
                    );
                }
            }
        }

        // ── 8c q4_0 × q8-GEMM arm (nt > 1, id <= 8192) — tolerance vs the
        // independent host reference (dequant + the same q8 activation
        // quantization the kernel applies), the cuda_kquant_matmul_parity
        // method. The in-block token loop does not change per-token math.
        {
            let (od, id_) = (512usize, 2048usize);
            let wb = gen_bytes(od * (id_ / 32) * 18, 0x5EED_00C0);
            let xs = gen_f32(id_ * nt, 0xA11C_00C0);
            let mut wt =
                Tensor::from_data(TensorType::Q4_0, &[id_ as i64, od as i64, 1, 1], wb.clone());
            wt.name = "w8c".to_string();
            cb.state.register_weight("w8c", &wb);

            let mut b = GraphBuilder::new();
            let x = b.input("x", [id_, nt, 1, 1], DType::F32);
            let m = b.matmul(x, &wt, None);
            b.output(m);
            let g = b.build();
            let xb = cb.alloc_buffer(id_ * nt);
            cb.write_host(xb, &xs).unwrap();
            let ob = cb.alloc_buffer(od * nt);
            cb.exec_ids(&g.nodes[m], &[xb], ob, None).unwrap();
            let got = cb.copy_to_host(ob).unwrap();

            // host reference: q4_0 dequant (val = (nib - 8) * d) dotted with
            // the q8-quantized activations
            let mut x8 = vec![0u8; nt * (id_ / 32) * 40];
            for t in 0..nt {
                for blk in 0..id_ / 32 {
                    let base = t * id_ + blk * 32;
                    let mut am = 0f32;
                    for j in 0..32 {
                        am = am.max(xs[base + j].abs());
                    }
                    let dd = am / 127.0;
                    let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                    let off = (t * (id_ / 32) + blk) * 40;
                    x8[off..off + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                    for j in 0..32 {
                        let q = (xs[base + j] * di).round().clamp(-128.0, 127.0) as i8;
                        x8[off + 4 + j] = q as u8;
                    }
                }
            }
            let dq8 = |t: usize, i: usize| -> f32 {
                let off = (t * (id_ / 32) + i / 32) * 40;
                half::f16::from_le_bytes([x8[off], x8[off + 1]]).to_f32()
                    * (x8[off + 4 + (i % 32)] as i8) as f32
            };
            // The per-block contraction order mirrors the kernel (d applied
            // per block); compare with the standard q8 tolerance.
            let mut want = vec![0f32; od * nt];
            let mut scale = 1e-9f32;
            for t in 0..nt {
                for r in 0..od {
                    let mut acc = 0f32;
                    for blk in 0..id_ / 32 {
                        let blkb = &wb[(r * (id_ / 32) + blk) * 18..];
                        let d = half::f16::from_le_bytes([blkb[0], blkb[1]]).to_f32();
                        let mut sdot = 0f32;
                        for j in 0..16 {
                            let b0 = blkb[2 + j];
                            sdot += ((b0 & 0x0F) as f32 - 8.0) * dq8(t, blk * 32 + j)
                                + ((b0 >> 4) as f32 - 8.0) * dq8(t, blk * 32 + 16 + j);
                        }
                        acc += d * sdot;
                    }
                    want[t * od + r] = acc;
                    scale = scale.max(acc.abs());
                }
            }
            assert_close("q4_0 8c multi-token", &got, &want, scale * 1e-2);
        }
    } // nt sweep
}
#[test]
fn cuda_fused_ffn_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (nf, id_) = (8usize, 512usize); // concat od = 16, decode nt = 1
    let xs: Vec<f32> = (0..id_)
        .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
        .collect();

    // ── q4_K gate/up weights (144-byte super-blocks, llama layout) ──
    // get_scale_min_k4 (llama.cpp Q4_K scale packing): the second half
    // of the 8 scale/min pairs is spliced across the 12 scale bytes.
    fn k4_scale(q: &[u8; 12], j: usize) -> (u8, u8) {
        if j < 4 {
            (q[j] & 63, q[j + 4] & 63)
        } else {
            (
                (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4),
                (q[j + 4] >> 4) | ((q[j] >> 6) << 4),
            )
        }
    }
    fn build_q4k(seed: u64, rows: usize, id: usize, bytes: &mut Vec<u8>, dq: &mut Vec<f32>) {
        for r in 0..rows {
            for ib in 0..id / 256 {
                let d = 0.031f32 + 0.005 * ((seed + (r * 7 + ib * 3) as u64) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((seed + (r * 3 + ib) as u64) % 4) as f32;
                bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                bytes.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    scb[j] = ((seed as usize + r * 31 + j * 17 + ib * 5) % 63) as u8;
                }
                bytes.extend_from_slice(&scb);
                let mut qs = [0u8; 128];
                for j in 0..128 {
                    let lo = ((seed as usize + r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((seed as usize + r * 5 + j * 11 + ib * 2) % 15) as u8;
                    qs[j] = lo | (hi << 4);
                }
                bytes.extend_from_slice(&qs);
                for j in 0..4 {
                    let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                    let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b = qs[j * 32 + l];
                        let base = r * id + ib * 256 + j * 64;
                        dq[base + l] = (b & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                        dq[base + 32 + l] = (b >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                    }
                }
            }
        }
    }

    // ── q6_K gate/up weights (210-byte super-blocks, llama layout) ──
    fn build_q6k(seed: u64, rows: usize, id: usize, bytes: &mut Vec<u8>, dq: &mut Vec<f32>) {
        for r in 0..rows {
            for ib in 0..id / 256 {
                let d = 0.027f32 + 0.004 * ((seed + (r * 11 + ib * 7) as u64) % 6) as f32;
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut sc = [0i8; 16];
                for i in 0..128 {
                    ql[i] = ((seed as usize + r * 29 + i * 7 + ib * 3) % 255) as u8;
                }
                for i in 0..64 {
                    qh[i] = ((seed as usize + r * 17 + i * 13 + ib * 11) % 255) as u8;
                }
                for i in 0..16 {
                    sc[i] = (((seed as usize + r * 5 + i * 3 + ib) % 15) as i8) - 7;
                }
                bytes.extend_from_slice(&ql);
                bytes.extend_from_slice(&qh);
                bytes.extend(sc.iter().map(|&x| x as u8));
                bytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                // reference dequant (llama.cpp Q6_K layout, same as
                // the kquant test): four interleaved 32-element groups
                // per 128-element half.
                for n in 0..2usize {
                    let qlh = &ql[n * 64..n * 64 + 64];
                    let qhh = &qh[n * 32..n * 32 + 32];
                    for l in 0..32usize {
                        let is = l / 16;
                        let q1 = ((qlh[l] & 0xF) as i32 | (((qhh[l] >> 0) as i32 & 3) << 4)) - 32;
                        let q2 =
                            ((qlh[l + 32] & 0xF) as i32 | (((qhh[l] >> 2) as i32 & 3) << 4)) - 32;
                        let q3 = ((qlh[l] >> 4) as i32 | (((qhh[l] >> 4) as i32 & 3) << 4)) - 32;
                        let q4 =
                            ((qlh[l + 32] >> 4) as i32 | (((qhh[l] >> 6) as i32 & 3) << 4)) - 32;
                        let base = r * id + ib * 256 + n * 128;
                        dq[base + l] = d * sc[n * 8 + is] as f32 * q1 as f32;
                        dq[base + l + 32] = d * sc[n * 8 + is + 2] as f32 * q2 as f32;
                        dq[base + l + 64] = d * sc[n * 8 + is + 4] as f32 * q3 as f32;
                        dq[base + l + 96] = d * sc[n * 8 + is + 6] as f32 * q4 as f32;
                    }
                }
            }
        }
    }

    let mut g4b = Vec::new();
    let mut g4dq = vec![0f32; nf * id_];
    build_q4k(1, nf, id_, &mut g4b, &mut g4dq);
    let mut u4b = Vec::new();
    let mut u4dq = vec![0f32; nf * id_];
    build_q4k(101, nf, id_, &mut u4b, &mut u4dq);
    let mut g6b = Vec::new();
    let mut g6dq = vec![0f32; nf * id_];
    build_q6k(7, nf, id_, &mut g6b, &mut g6dq);
    let mut u6b = Vec::new();
    let mut u6dq = vec![0f32; nf * id_];
    build_q6k(207, nf, id_, &mut u6b, &mut u6dq);

    // concat rows: gate rows then up rows (concat_rows semantics)
    let gu4: Vec<u8> = g4b.iter().chain(u4b.iter()).copied().collect();
    let gu6: Vec<u8> = g6b.iter().chain(u6b.iter()).copied().collect();
    cb.state.register_weight("mgu4", &gu4);
    // q6_K concat goes through the padded repack (7e② layout)
    cb.state
        .register_weight_q6k_padded("mgu6", &gu6, 2 * nf, id_);
    assert!(cb.state.is_weight_padded("mgu6"));

    let (xb, ogu4, ogu6) = (
        cb.alloc_buffer(id_),
        cb.alloc_buffer(2 * nf),
        cb.alloc_buffer(2 * nf),
    );
    cb.write_host(xb, &xs).unwrap();

    for (ttype, wname, ogu, gdq, udq) in [
        (crate::tensor::TensorType::Q4_K, "mgu4", ogu4, &g4dq, &u4dq),
        (crate::tensor::TensorType::Q6_K, "mgu6", ogu6, &g6dq, &u6dq),
    ] {
        let mut b = crate::graph::builder::GraphBuilder::new();
        let x = b.input("x", [id_, 1, 1, 1], crate::graph::DType::F32);
        let gu = b.fused_ffn(
            x,
            crate::graph::ops::FusedFfnMeta {
                gu_weight: wname.to_string(),
                weight_ttype: ttype,
                in_dim: id_,
                nf,
            },
        );
        b.output(gu);
        let g = b.build();
        cb.exec_ids(&g.nodes[gu], &[xb], ogu, None).unwrap();

        // host reference: silu(gate·x) × (up·x)
        let got = cb.copy_to_host(ogu).unwrap();
        let mut want = vec![0f32; nf];
        for r in 0..nf {
            let mut ag = 0f32;
            let mut au = 0f32;
            for i in 0..id_ {
                ag += gdq[r * id_ + i] * xs[i];
                au += udq[r * id_ + i] * xs[i];
            }
            let s = ag / (1.0f32 + (-ag).exp());
            want[r] = s * au;
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(
            &format!("{ttype:?} fused ffn"),
            &got[..nf],
            &want,
            scale * 1e-3,
        );
    }
}
// 8f: Q5_1 / Q5_K f32-activation matmul parity (incl. the Q5_K partial
// tail super-block at id = 896 = 3.5 × 256). The weight blocks are
// quantized in-test against scales unpacked with the REAL
// block::unpack_q4k_scales, so kernel and reference share the exact
// decode math and the tolerance stays tight.
#[test]
fn cuda_q5_matmul_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let nt = 3usize;

    // ── Q5_1: od 8, id 64 (2 blocks / row) ──
    {
        let (od, id) = (8usize, 64usize);
        let nb = id / 32;
        let wf: Vec<f32> = (0..od * id)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let mut wq = Vec::new();
        for r in 0..od {
            for b in 0..nb {
                let row = &wf[r * id + b * 32..r * id + (b + 1) * 32];
                let amax = row.iter().fold(0f32, |m, &v| m.max(v));
                let amin = row.iter().fold(0f32, |m, &v| m.min(v));
                let d = (amax - amin) / 31.0;
                let di = if d != 0.0 { 1.0 / d } else { 0.0 };
                wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                wq.extend_from_slice(&half::f16::from_f32(amin).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((row[j] - amin) * di).round().clamp(0.0, 31.0) as u32;
                    let u_hi = ((row[j + 16] - amin) * di).round().clamp(0.0, 31.0) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                wq.extend_from_slice(&qh.to_le_bytes());
                wq.extend_from_slice(&qs);
            }
        }
        let state = cb.state;
        state.register_weight("w51", &wq);
        let wptr = state.get_weight_ptr("w51").unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        state
            .matmul_f32_ptr(
                wptr,
                TensorType::Q5_1,
                cb.ptr_of(xb).unwrap(),
                cb.ptr_of(out).unwrap(),
                od,
                id,
                nt,
            )
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        // independent dequant reference
        for t in 0..nt {
            for r in 0..od {
                let mut want = 0f32;
                for b in 0..nb {
                    let blk = &wq[(r * nb + b) * 24..(r * nb + b) * 24 + 24];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let m = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                    let qh = u32::from_le_bytes([blk[4], blk[5], blk[6], blk[7]]);
                    let xrow = &xs[t * id + b * 32..t * id + (b + 1) * 32];
                    for j in 0..16 {
                        let u_lo = ((blk[8 + j] & 0xF) as f32) + 16.0 * ((qh >> j) & 1) as f32;
                        let u_hi =
                            ((blk[8 + j] >> 4) as f32) + 16.0 * ((qh >> (j + 16)) & 1) as f32;
                        want += d * (u_lo * xrow[j] + u_hi * xrow[j + 16])
                            + m * (xrow[j] + xrow[j + 16]);
                    }
                }
                assert!(
                    (got[t * od + r] - want).abs() < 5e-3,
                    "q5_1 [{t}][{r}] {} vs {want}",
                    got[t * od + r]
                );
            }
        }
    }

    // ── Q5_0: od 8, id 64 (2 blocks / row) — the tok_embd type of the
    // 0.5B q4_k_m GGUFs; decode f32-activation kernel parity ──
    {
        let (od, id) = (8usize, 64usize);
        let nb = id / 32;
        let mut wq = Vec::new();
        for r in 0..od {
            for b in 0..nb {
                let d = 0.02f32 + 0.003 * ((r * 5 + b) % 7) as f32;
                wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((r * 11 + b * 7 + j * 3) % 32) as u32;
                    let u_hi = ((r * 7 + b * 5 + j) % 32) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                wq.extend_from_slice(&qh.to_le_bytes());
                wq.extend_from_slice(&qs);
            }
        }
        let state = cb.state;
        state.register_weight("w50", &wq);
        let wptr = state.get_weight_ptr("w50").unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        state
            .matmul_f32_ptr(
                wptr,
                TensorType::Q5_0,
                cb.ptr_of(xb).unwrap(),
                cb.ptr_of(out).unwrap(),
                od,
                id,
                nt,
            )
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        // independent dequant reference
        for t in 0..nt {
            for r in 0..od {
                let mut want = 0f32;
                for b in 0..nb {
                    let blk = &wq[(r * nb + b) * 22..(r * nb + b) * 22 + 22];
                    let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                    let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
                    let xrow = &xs[t * id + b * 32..t * id + (b + 1) * 32];
                    for j in 0..16 {
                        let v_lo =
                            ((blk[6 + j] & 0xF) as f32) + 16.0 * ((qh >> j) & 1) as f32 - 16.0;
                        let v_hi = ((blk[6 + j] >> 4) as f32)
                            + 16.0 * ((qh >> (j + 16)) & 1) as f32
                            - 16.0;
                        want += d * (v_lo * xrow[j] + v_hi * xrow[j + 16]);
                    }
                }
                assert!(
                    (got[t * od + r] - want).abs() < 5e-3,
                    "q5_0 [{t}][{r}] {} vs {want}",
                    got[t * od + r]
                );
            }
        }
    }

    // ── Q5_K: od 8, id 896 (PARTIAL tail super-block: 3.5 × 256) ──
    // Weight values are GENERATED from the decode formula with random
    // per-sub w (0..31) against scales unpacked from random sc bytes —
    // the test targets the kernel's decode/indexing/tail-masking
    // correctness, not a quantizer.
    {
        let (od, id) = (8usize, 896usize);
        let nsp = (id + 255) / 256; // 4 — last one is partial (4 valid subs)
        let mut wf: Vec<f32> = (0..od * id)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let mut wq = vec![0u8; od * nsp * 176];
        for r in 0..od {
            for sp in 0..nsp {
                let blk_off = (r * nsp + sp) * 176;
                let sc: [u8; 12] = core::array::from_fn(|i| ((i * 7 + 3) % 63 + 1) as u8);
                let (scales, mins) = crate::block::unpack_q4k_scales(&sc);
                wq[blk_off..blk_off + 4].copy_from_slice(&{
                    // d = 0.25, dmin = 0.25 (exact in f16)
                    let b = half::f16::from_f32(0.25).to_le_bytes();
                    [b[0], b[1], b[0], b[1]]
                });
                wq[blk_off + 4..blk_off + 16].copy_from_slice(&sc);
                // qh/qs stay zero for invalid tail subs (masked out)
                let valid = ((id - sp * 256).min(256) + 31) / 32;
                for sub in 0..valid {
                    let base = sp * 256 + sub * 32;
                    let row = &wf[r * id + base..r * id + base + 32];
                    // invert the decode: v = d·s8·w − dmin·m8 →
                    // w = (v + dmin·m8) / (d·s8); needs w ∈ 0..31 —
                    // instead regenerate v FROM w so it is exact:
                    for l in 0..32 {
                        let seed = (r * 91 + base + l) % 32;
                        let v = 0.25 * scales[sub] as f32 * seed as f32 - 0.25 * mins[sub] as f32;
                        // overwrite wf so the reference dot uses exact values
                        wf[r * id + base + l] = v;
                        let wv = seed as u8;
                        let ci = sub >> 1;
                        if sub & 1 == 1 {
                            qs_byte(&mut wq[blk_off + 48..], ci, l, wv, true);
                        } else {
                            qs_byte(&mut wq[blk_off + 48..], ci, l, wv, false);
                        }
                        qh_byte(&mut wq[blk_off + 16..], l, sub, wv);
                    }
                }
            }
        }
        let state = cb.state;
        state.register_weight("w5k", &wq);
        let wptr = state.get_weight_ptr("w5k").unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        // Step 82: nt = 3 dispatches Q5_K to the multi-token MMVQ
        // kernel (weights-once; the 24M nt == 1 crossover does not
        // apply in-block) — the reference dots the pad40 q8 activation
        // round-trip, tolerance as in the mmvq parity tests.
        let mut x8 = vec![0u8; nt * (id / 32) * 40];
        for t in 0..nt {
            for blk in 0..id / 32 {
                let base = t * id + blk * 32;
                let mut am = 0f32;
                for j in 0..32 {
                    am = am.max(xs[base + j].abs());
                }
                let dd = am / 127.0;
                let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
                let off = (t * (id / 32) + blk) * 40;
                x8[off..off + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
                for j in 0..32 {
                    let q = (xs[base + j] * di).round().clamp(-128.0, 127.0) as i8;
                    x8[off + 4 + j] = q as u8;
                }
            }
        }
        let dq8 = |t: usize, i: usize| -> f32 {
            let off = (t * (id / 32) + i / 32) * 40;
            half::f16::from_le_bytes([x8[off], x8[off + 1]]).to_f32()
                * (x8[off + 4 + (i % 32)] as i8) as f32
        };
        let xb = cb.alloc_buffer(id * nt);
        let out = cb.alloc_buffer(od * nt);
        cb.write_host(xb, &xs).unwrap();
        state
            .matmul_f32_ptr(
                wptr,
                TensorType::Q5_K,
                cb.ptr_of(xb).unwrap(),
                cb.ptr_of(out).unwrap(),
                od,
                id,
                nt,
            )
            .unwrap();
        cb.synchronize();
        let got = cb.copy_to_host(out).unwrap();
        // independent dequant reference (mirrors the kernel decode)
        let deq = |r: usize| -> Vec<f32> {
            let mut outv = vec![0f32; id];
            for sp in 0..nsp {
                let blk_off = (r * nsp + sp) * 176;
                let d = half::f16::from_le_bytes([wq[blk_off], wq[blk_off + 1]]).to_f32();
                let dmin = half::f16::from_le_bytes([wq[blk_off + 2], wq[blk_off + 3]]).to_f32();
                let sc: [u8; 12] = wq[blk_off + 4..blk_off + 16].try_into().unwrap();
                let (scales, mins) = crate::block::unpack_q4k_scales(&sc);
                let valid = ((id - sp * 256).min(256) + 31) / 32;
                for sub in 0..valid {
                    let ci = sub >> 1;
                    for l in 0..32 {
                        let qbyte = wq[blk_off + 48 + ci * 32 + l];
                        let nib = if sub & 1 == 1 {
                            qbyte >> 4
                        } else {
                            qbyte & 0xF
                        };
                        let w = nib as f32 + 16.0 * (((wq[blk_off + 16 + l] >> sub) & 1) as f32);
                        outv[sp * 256 + sub * 32 + l] =
                            d * scales[sub] as f32 * w - dmin * mins[sub] as f32;
                    }
                }
            }
            outv
        };
        let mut wants = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let dq = deq(r);
                let mut want = 0f32;
                for i in 0..id {
                    want += dq[i] * dq8(t, i);
                }
                wants[t * od + r] = want;
            }
        }
        let scale = wants.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        for t in 0..nt {
            for r in 0..od {
                assert!(
                    (got[t * od + r] - wants[t * od + r]).abs() < scale * 1e-2,
                    "q5_K [{t}][{r}] {} vs {}",
                    got[t * od + r],
                    wants[t * od + r]
                );
            }
        }
    }
}
// q5_K qs nibble packing: 4 chunks of 32 bytes; chunk ci, byte l:
// low nibble = element l of sub 2ci, high = element l of sub 2ci+1
fn qs_byte(qs: &mut [u8], ci: usize, l: usize, w: u8, hi: bool) {
    if hi {
        qs[ci * 32 + l] |= w << 4;
    } else {
        qs[ci * 32 + l] |= w & 0xF;
    }
}
// q5_K qh layout: byte l, bit sub = the >16 bit of element (sub, l)
fn qh_byte(qh: &mut [u8], l: usize, sub: usize, w: u8) {
    qh[l] |= ((w >> 4) & 1) << sub;
}
/// Q5_0 real-shape isolation: 0.5B q4_k_m was the first model to reach
/// CUDA with Q5_0 weights, and an end-to-end run died with a sticky
/// cudaErrorMisalignedAddress (716). Run every Q5_0 device path at the
/// model's REAL shapes with a sync after each step so the first faulting
/// path is identified exactly (small-shape parity above already proves
/// the math; this test targets shape/alignment coverage).
#[test]
fn cuda_q5_0_realshape_isolation() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let nt = 30usize; // the "Hello" prompt length
    macro_rules! step {
        ($tag:expr, $run:expr) => {{
            // Backend::synchronize() does NOT wait on the stream outside a
            // capture window — use the real state sync so an async fault
            // surfaces HERE, not at the next cudaMalloc.
            eprintln!("[isolation] begin {}", $tag);
            $run;
            cb.state.sync();
            eprintln!("[isolation] end {}", $tag);
        }};
    }
    cb.state.sync(); // baseline: context healthy after CudaBackend::new()
    eprintln!("[isolation] baseline sync done");

    // ── 0. embed bisect: isolate the fault dimension. Parity (6-row
    //    table, ids [0,5,2], nt=3, n_embd=512) is clean; the model-real
    //    (4096-row table, ids [7,1020,2033]) faults. Vary one dimension
    //    at a time: table size, id values. ──
    let n_embd_b = 512usize;
    let nb_b = n_embd_b / 32; // 16
    let build_table = |rows: usize| -> Vec<u8> {
        let mut t = Vec::new();
        for r in 0..rows {
            for ib in 0..nb_b {
                let d = 0.02f32 + 0.003 * ((r * 5 + ib) % 7) as f32;
                t.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((r * 11 + ib * 7 + j * 3) % 32) as u32;
                    let u_hi = ((r * 7 + ib * 5 + j) % 32) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                t.extend_from_slice(&qh.to_le_bytes());
                t.extend_from_slice(&qs);
            }
        }
        t
    };
    let cases: [(&str, usize, &[u32]); 5] = [
        ("a_smalltable_parityids", 6, &[0, 5, 2]),
        ("b_bigtable_parityids", 4096, &[0, 5, 2]),
        ("c_bigtable_bigids", 4096, &[7, 1020, 2033]),
        ("d_smalltable_midids", 16, &[7, 12, 15]),
        ("e_bigtable_row7only", 4096, &[7, 7, 7]),
    ];
    for &(cname, rows, ids) in &cases {
        let tbl = build_table(rows);
        let name = format!("iso_emb_{cname}");
        cb.state.register_weight(&name, &tbl);
        let wptr = cb.state.get_weight_ptr(&name).unwrap();
        let ids_f: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();
        let idb = cb.alloc_buffer(ids.len());
        cb.write_host(idb, &ids_f).unwrap();
        let ob = cb.alloc_buffer(n_embd_b * ids.len());
        step!(format!("embed {cname}"), {
            cb.state
                .embed_rows_on_gpu(
                    TensorType::Q5_0,
                    wptr,
                    cb.ptr_of(idb).unwrap(),
                    cb.ptr_of(ob).unwrap(),
                    n_embd_b,
                    ids.len(),
                    false,
                )
                .unwrap();
        });
    }

    // ── 2-5. prefill + decode matmuls at the model's real matmul shapes
    //    (attn_q 896x896, attn_k 896x128, ffn_gu 896x9728, ffn_down-class
    //    896x4864) through all three prefill paths ──
    let shapes = [
        (896usize, 896usize),
        (896usize, 128usize),
        (4864usize, 896usize),
    ];
    for (si, &(od, id)) in shapes.iter().enumerate() {
        let nb = id / 32;
        let mut wq = Vec::new();
        for r in 0..od {
            for b in 0..nb {
                let d = 0.02f32 + 0.003 * ((r * 5 + b + si) % 7) as f32;
                wq.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                let mut qh = 0u32;
                let mut qs = [0u8; 16];
                for j in 0..16 {
                    let u_lo = ((r * 11 + b * 7 + j * 3) % 32) as u32;
                    let u_hi = ((r * 7 + b * 5 + j) % 32) as u32;
                    qs[j] = ((u_lo & 0xF) | ((u_hi & 0xF) << 4)) as u8;
                    qh |= ((u_lo >> 4) & 1) << j;
                    qh |= ((u_hi >> 4) & 1) << (j + 16);
                }
                wq.extend_from_slice(&qh.to_le_bytes());
                wq.extend_from_slice(&qs);
            }
        }
        let name = format!("iso_w{si}");
        cb.state.register_weight(&name, &wq);
        let wptr = cb.state.get_weight_ptr(&name).unwrap();
        let xs: Vec<f32> = (0..id * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let xb = cb.alloc_buffer(id * nt);
        cb.write_host(xb, &xs).unwrap();
        let out = cb.alloc_buffer(od * nt);

        // 2. legacy f32-activation kernel (also the decode kernel)
        step!(format!("legacy f32 matmul od={od} id={id} nt={nt}"), {
            cb.state
                .matmul_f32_ptr(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                )
                .unwrap();
        });

        // 3. f16 wmma GEMM path (MINFER_MMQ=0 territory)
        step!(format!("f16 GEMM od={od} id={id} nt={nt}"), {
            cb.state
                .prefill_gemm_f16_inner(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                    false,
                    false,
                )
                .unwrap();
        });

        // 4. MMQ int8 GEMM path (the r60 default)
        step!(format!("MMQ od={od} id={id} nt={nt}"), {
            cb.state
                .prefill_mmq(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(xb).unwrap(),
                    cb.ptr_of(out).unwrap(),
                    od,
                    id,
                    nt,
                    false,
                    1,
                )
                .unwrap();
        });

        // 5. decode nt==1 through the top dispatch (routing check)
        let x1 = cb.alloc_buffer(id);
        cb.write_host(x1, &xs[..id]).unwrap();
        let o1 = cb.alloc_buffer(od);
        step!(format!("decode dispatch od={od} id={id} nt=1"), {
            cb.state
                .matmul_f32_ptr(
                    wptr,
                    TensorType::Q5_0,
                    cb.ptr_of(x1).unwrap(),
                    cb.ptr_of(o1).unwrap(),
                    od,
                    id,
                    1,
                )
                .unwrap();
        });
    }
}
