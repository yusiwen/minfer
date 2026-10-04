//! Decode-time (nt == 1) MMVQ parity for the K-quants.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// 7e⑤: fused FFN gate+up parity — the concat matmul + in-place offset
/// swiglu must equal silu(gate·x)·(up·x) computed on the host, for a
/// plain-registered q4_K concat and a padded-repacked q6_K concat.
/// The reference dequantizes with the same independent in-test block
/// layouts as `cuda_kquant_matmul_parity`.
#[test]
fn cuda_q4k_decode_mmvq_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // the mmvq scratch (buf_q8_decode) is singleton state — serialize the
    // parallel mmvq parity tests so one test's scratch grow cannot free
    // the buffer another test just enqueued kernels against
    let _guard = crate::cuda::CudaState::model_load_guard();
    // 8e-reversal: decode (nt == 1) Q4_K dispatches to the MMVQ structure
    // kernel (q8 activations + dp4a, one 256-thread block per row) when
    // id >= 2048 && id % 32 == 0. Reference: independent dequant of the
    // same bytes + the SAME q8 activation quantization the kernel applies
    // (round-trip error ~2/127 per element), so the tolerance stays tight.
    let (od, id_, nt) = (5120usize, 3584usize, 1usize); // 7B fused-qkv shape
                                                        // activations with real-model spread (RMSNorm outputs reach ±4, and
                                                        // some 32-blocks are near-zero: exercises the f16 d8 denormal range)
    let mut rng_state = 12345u64;
    let xs: Vec<f32> = (0..id_ * nt)
        .map(|_| {
            rng_state = rng_state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0; // [-1, 1)
            let mag = if (rng_state >> 60) & 7 == 0 {
                1e-5
            } else {
                3.0
            };
            (u as f32) * mag
        })
        .collect();

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

    // mirror quantize_q8_0_pad40: per 32-element block, d = amax/127,
    // payload at byte offset 4 (padded 40B blocks)
    let mut x8 = vec![0u8; (id_ / 32) * 40];
    for b in 0..id_ / 32 {
        let mut am = 0f32;
        for j in 0..32 {
            am = am.max(xs[b * 32 + j].abs());
        }
        let dd = am / 127.0;
        let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
        x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
        for j in 0..32 {
            let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
            x8[b * 40 + 4 + j] = q as u8;
        }
    }
    let dq8 = |i: usize| -> f32 {
        half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
            * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
    };

    let mut w4t = Tensor::from_data(
        TensorType::Q4_K,
        &[id_ as i64, od as i64, 1, 1],
        w4b.clone(),
    );
    w4t.name = "mmw4k".to_string();
    cb.state.register_weight("mmw4k", &w4b);

    let mut b = GraphBuilder::new();
    let x = b.input("x", [id_, nt, 1, 1], DType::F32);
    let m = b.matmul(x, &w4t, None);
    b.output(m);
    let g = b.build();

    let xb = cb.alloc_buffer(id_ * nt);
    cb.write_host(xb, &xs).unwrap();
    let ob = cb.alloc_buffer(od * nt);
    cb.exec_ids(&g.nodes[m], &[xb], ob, None).unwrap();
    let got = cb.copy_to_host(ob).unwrap();

    let mut want = vec![0f32; od * nt];
    for t in 0..nt {
        for r in 0..od {
            let mut acc = 0f32;
            for i in 0..id_ {
                acc += w4dq[r * id_ + i] * dq8(t * id_ + i);
            }
            want[t * od + r] = acc;
        }
    }
    let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
    // q8 activation quantization noise (<= 2/127 per element, random
    // signs over 2048 elements) lands well under 1e-2 of the row scale.
    assert_close("q4_k decode mmvq", &got, &want, scale * 1e-2);

    // ── real 7B weights (Q4_K tensors of the decode path) ──
    // dumped via the ignored `dump_real_q4k_tensor` helper; skipped when
    // the file is absent so the suite stays hermetic.
    let real_shapes = [
        ("real_blk_0_attn_q_weight.bin", 3584usize, 3584usize),
        ("real_blk_0_attn_k_weight.bin", 3584usize, 512usize),
        ("real_blk_0_ffn_gate_weight.bin", 3584usize, 18944usize),
    ];
    for (file, id2, od2) in real_shapes {
        let Ok(wb) = std::fs::read(format!("/tmp/minfer_phase7/{file}")) else {
            break;
        };
        assert_eq!(wb.len(), od2 * (id2 / 256) * 144);
        let mut w2t = Tensor::from_data(
            TensorType::Q4_K,
            &[id2 as i64, od2 as i64, 1, 1],
            wb.clone(),
        );
        w2t.name = "realq4k".to_string();
        cb.state.register_weight("realq4k", &wb);
        let mut b2 = GraphBuilder::new();
        let x2 = b2.input("x2", [id2, 1, 1, 1], DType::F32);
        let m2 = b2.matmul(x2, &w2t, None);
        b2.output(m2);
        let g2 = b2.build();
        let xb2 = cb.alloc_buffer(id2);
        cb.write_host(xb2, &xs).unwrap();
        let ob2 = cb.alloc_buffer(od2);
        cb.exec_ids(&g2.nodes[m2], &[xb2], ob2, None).unwrap();
        let got2 = cb.copy_to_host(ob2).unwrap();

        // CPU reference over the real bytes: dequant (same map as the
        // parity reference above) + q8-quantized activations
        let mut want2 = vec![0f32; od2];
        for r in 0..od2 {
            let mut acc = 0f32;
            for ib in 0..id2 / 256 {
                let blk = &wb[(r * (id2 / 256) + ib) * 144..];
                let d = half::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                let dmin = half::f16::from_le_bytes([blk[2], blk[3]]).to_f32();
                let mut scb = [0u8; 12];
                scb.copy_from_slice(&blk[4..16]);
                for j in 0..4 {
                    let (s_lo, m_lo) = k4_scale(&scb, 2 * j);
                    let (s_hi, m_hi) = k4_scale(&scb, 2 * j + 1);
                    for l in 0..32 {
                        let b8 = blk[16 + j * 32 + l];
                        let base = ib * 256 + j * 64;
                        let v_lo = (b8 & 0x0F) as f32 * d * s_lo as f32 - dmin * m_lo as f32;
                        let v_hi = (b8 >> 4) as f32 * d * s_hi as f32 - dmin * m_hi as f32;
                        acc += v_lo * dq8(base + l);
                        acc += v_hi * dq8(base + 32 + l);
                    }
                }
            }
            want2[r] = acc;
        }
        // activations here only cover id2=3584 — regenerate quantization
        // for the real id (xs has id_=3584 entries from the scaled-up test)
        let scale2 = want2.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        let mut worst = (0f32, 0usize);
        for r in 0..od2 {
            let e = (got2[r] - want2[r]).abs();
            if e > worst.0 {
                worst = (e, r);
            }
        }
        println!(
            "real q4_k [{file}]: max err {:.4} at row {} (got {:.4} want {:.4}, scale {scale2:.3})",
            worst.0, worst.1, got2[worst.1], want2[worst.1]
        );
        assert!(
            worst.0 <= scale2 * 1e-2,
            "real q4_k mmvq err {} > {}",
            worst.0,
            scale2 * 1e-2
        );
    }
}
/// 8e follow-up: decode (nt == 1) Q6_K dispatches to the MMVQ structure
/// kernel (16-element units over q8 activations). The synthetic block
/// covers a partial tail super-block (id = 2176 = 8×256 + 128) and full
/// 6-bit scale/nibble ranges; both strides go through direct
/// `q6_k_decode_mmvq` calls — the synthetic shape (od=64) is far below
/// the measured od*id >= 24M dispatch gate on purpose (small shapes
/// stay on the f32 kernel at runtime), and gate coverage comes from the
/// full-size 7B ffn_down real-weight section below.
#[test]
fn cuda_q6k_decode_mmvq_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    // serialize against the other mmvq parity tests (shared scratch)
    let _guard = crate::cuda::CudaState::model_load_guard();
    // two shapes: 2176 (partial tail super-block → v1 kernels) and 2560
    // (full super-blocks → the R2 v2 weight-streaming kernels)
    for (i, (od, id_, nt)) in [(64usize, 2176usize, 1usize), (64usize, 2560usize, 1usize)]
        .into_iter()
        .enumerate()
    {
        let mut rng_state = 12345u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

        let nbe = (id_ + 255) / 256;
        let mut w6b = Vec::new();
        let mut w6dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.02f32 + 0.004 * ((r * 7 + ib * 3) % 5) as f32;
                let mut ql = [0u8; 128];
                let mut qh = [0u8; 64];
                let mut sc = [0u8; 16];
                for s in 0..16usize {
                    sc[s] = (((r * 11 + s * 5 + ib * 3) % 64) as i32 - 32) as u8;
                }
                // only the real elements of a partial tail super-block
                let n_elem = 256usize.min(id_ - ib * 256);
                for e in 0..n_elem {
                    let s = e / 16;
                    let l = e % 16;
                    let chunk = s / 8;
                    let g = (s / 2) % 4;
                    let is = s % 2;
                    let q6 = ((r * 13 + e * 7 + ib * 3) % 64) as u8;
                    w6dq[r * id_ + ib * 256 + e] = d * (sc[s] as i8 as f32) * (q6 as f32 - 32.0);
                    let nib = q6 & 0xF;
                    let hi2 = (q6 >> 4) & 3;
                    let qpos = chunk * 64 + (g % 2) * 32 + is * 16 + l;
                    if g < 2 {
                        ql[qpos] |= nib;
                    } else {
                        ql[qpos] |= nib << 4;
                    }
                    let hpos = chunk * 32 + is * 16 + l;
                    qh[hpos] |= hi2 << (2 * g);
                }
                // block_q6_K field order: ql[128], qh[64], scales[16], d —
                // d is the LAST field (offset 208), not the first
                w6b.extend_from_slice(&ql);
                w6b.extend_from_slice(&qh);
                w6b.extend_from_slice(&sc);
                w6b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            }
        }
        assert_eq!(w6b.len(), od * nbe * 210);

        // mirror quantize_q8_0_pad40 (padded 40B blocks, payload at offset 4)
        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w6dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));

        // ── padded 224B registration, direct decode call (gate bypassed) ──
        let mut w6t = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6t.name = format!("mw6kp{i}");
        cb.state
            .register_weight_q6k_padded(&format!("mw6kp{i}"), &w6b, od, id_);
        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.state.q6_k_decode_mmvq(
            cb.state.get_weight_ptr(&format!("mw6kp{i}")).unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(ob).unwrap(),
            od,
            id_,
            nt,
            true,
        );
        let got = cb.copy_to_host(ob).unwrap();
        let mut worst = (0f32, 0usize);
        for i in 0..od * nt {
            let e = (got[i] - want[i]).abs();
            if e > worst.0 {
                worst = (e, i);
            }
        }
        assert!(
            worst.0 <= scale * 1e-2,
            "q6_k mmvq padded err {} > {} at {}",
            worst.0,
            scale * 1e-2,
            worst.1
        );

        // ── raw 210B stride via a direct decode call (gate bypassed) ──
        let mut w6tr = Tensor::from_data(
            TensorType::Q6_K,
            &[id_ as i64, od as i64, 1, 1],
            w6b.clone(),
        );
        w6tr.name = format!("mw6kr{i}");
        cb.state.register_weight(&format!("mw6kr{i}"), &w6b);
        let obr = cb.alloc_buffer(od * nt);
        cb.state.q6_k_decode_mmvq(
            cb.state.get_weight_ptr(&format!("mw6kr{i}")).unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(obr).unwrap(),
            od,
            id_,
            nt,
            false,
        );
        let gotr = cb.copy_to_host(obr).unwrap();
        for i in 0..od * nt {
            assert!(
                (gotr[i] - want[i]).abs() <= scale * 1e-2,
                "q6_k mmvq raw stride mismatch at {i}"
            );
        }
    }
    // ── real weights (Q6_K tensors of the decode path) ──
    // 7B blk.0.ffn_down (q4_k_m, od=3584 x id=18944 — the shape that
    // passes the od*id >= 24M dispatch gate, so this also covers the
    // graph-dispatch wiring), 7B blk.0.attn_v and 0.5B blk.0.ffn_down
    // (both BELOW the gate — direct calls, since runtime keeps those
    // on the f32 kernel). Dumped via the ignored helpers; skipped when
    // a file is absent so the suite stays hermetic. The padded
    // registration repacks the raw 210B bytes, matching the loader.
    let real_shapes: [(&str, usize, usize, bool); 3] = [
        ("real_blk_0_ffn_down_weight.bin", 18944, 3584, true),
        ("real_blk_0_attn_v_weight.bin", 3584, 512, false),
        ("real05_blk_0_ffn_down_weight.bin", 4864, 896, false),
    ];
    for (file, id2, od2, via_graph) in real_shapes {
        let Ok(wb) = std::fs::read(format!("/tmp/minfer_phase7/{file}")) else {
            println!("real q6_k [{file}]: absent — skipped");
            break;
        };
        assert_eq!(wb.len(), od2 * (id2 / 256) * 210);
        // fresh activations at the real width
        let mut rng2 = 777u64;
        let xs2: Vec<f32> = (0..id2)
            .map(|_| {
                rng2 = rng2
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng2 >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng2 >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                (u as f32) * mag
            })
            .collect();
        let mut x82 = vec![0u8; (id2 / 32) * 40];
        for b in 0..id2 / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs2[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x82[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs2[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x82[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq82 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x82[(i / 32) * 40], x82[(i / 32) * 40 + 1]]).to_f32()
                * (x82[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };
        let mut w2t = Tensor::from_data(
            TensorType::Q6_K,
            &[id2 as i64, od2 as i64, 1, 1],
            wb.clone(),
        );
        w2t.name = "realq6k".to_string();
        cb.state
            .register_weight_q6k_padded("realq6k", &wb, od2, id2);
        let mut b2 = GraphBuilder::new();
        let x2 = b2.input("x2", [id2, 1, 1, 1], DType::F32);
        let m2 = b2.matmul(x2, &w2t, None);
        b2.output(m2);
        let g2 = b2.build();
        let xb2 = cb.alloc_buffer(id2);
        cb.write_host(xb2, &xs2).unwrap();
        let ob2 = cb.alloc_buffer(od2);
        if via_graph {
            // shape above the od*id gate — dispatch selects the MMVQ
            cb.exec_ids(&g2.nodes[m2], &[xb2], ob2, None).unwrap();
        } else {
            // below the gate — runtime dispatch would keep the f32
            // kernel; call the decode path directly for kernel parity
            cb.state.q6_k_decode_mmvq(
                cb.state.get_weight_ptr("realq6k").unwrap(),
                cb.ptr_of(xb2).unwrap(),
                cb.ptr_of(ob2).unwrap(),
                od2,
                id2,
                1,
                true,
            );
        }
        let got2 = cb.copy_to_host(ob2).unwrap();

        let mut want2 = vec![0f32; od2];
        for r in 0..od2 {
            let mut acc = 0f32;
            for ib in 0..id2 / 256 {
                let blk = &wb[(r * (id2 / 256) + ib) * 210..];
                let d = half::f16::from_le_bytes([blk[208], blk[209]]).to_f32();
                for e in 0..256 {
                    let s = e / 16;
                    let l = e % 16;
                    let chunk = s / 8;
                    let g = (s / 2) % 4;
                    let is = s % 2;
                    let qlb = blk[chunk * 64 + (g % 2) * 32 + is * 16 + l];
                    let nib = if g < 2 { qlb & 0xF } else { qlb >> 4 };
                    let qhb = blk[128 + chunk * 32 + is * 16 + l];
                    let hi = (qhb >> (2 * g)) & 3;
                    acc += d
                        * (blk[192 + s] as i8 as f32)
                        * ((nib | (hi << 4)) as f32 - 32.0)
                        * dq82(ib * 256 + e);
                }
            }
            want2[r] = acc;
        }
        let scale2 = want2.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        let mut worst = (0f32, 0usize);
        for r in 0..od2 {
            let e = (got2[r] - want2[r]).abs();
            if e > worst.0 {
                worst = (e, r);
            }
        }
        println!(
            "real q6_k [{file}]: max err {:.4} at row {} (got {:.4} want {:.4}, scale {scale2:.3})",
            worst.0, worst.1, got2[worst.1], want2[worst.1]
        );
        assert!(
            worst.0 <= scale2 * 1e-2,
            "real q6_k mmvq err {} > {}",
            worst.0,
            scale2 * 1e-2
        );
    }
}
/// 8e follow-up: decode (nt == 1) Q5_K dispatches to the MMVQ structure
/// kernel (q4_K shape with the q5 high-bit plane folded in). Scale bytes
/// cover the full 0..255 range so the get_scale_min_k4 high-bit splicing
/// (bits 6..7 of bytes 0..3 feeding the upper scales — the 8e lesson that
/// %63 data never reaches those paths) is exercised. No local model
/// carries Q5_K tensors (the "q5_k_m"-branded 0.5B GGUF actually stores
/// Q5_1/Q8_0/Q6_K), so parity rests on this synthetic; real-weight
/// parity lands when a model with Q5_K tensors is used.
#[test]
fn cuda_q5k_decode_mmvq_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    // serialize against the other mmvq parity tests (shared scratch)
    let _guard = crate::cuda::CudaState::model_load_guard();
    // two shapes: 2176 (partial tail super-block → v1 kernels) and 2560
    // (full super-blocks → the R2 v2 weight-streaming kernels)
    for (i, (od, id_, nt)) in [(128usize, 2176usize, 1usize), (128usize, 2560usize, 1usize)]
        .into_iter()
        .enumerate()
    {
        let mut rng_state = 54321u64;
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|_| {
                rng_state = rng_state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((rng_state >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (rng_state >> 60) & 7 == 0 {
                    1e-5
                } else {
                    3.0
                };
                (u as f32) * mag
            })
            .collect();

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

        let nbe = (id_ + 255) / 256;
        let mut w5b = Vec::new();
        let mut w5dq = vec![0f32; od * id_];
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                w5b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                w5b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                let mut scb = [0u8; 12];
                for j in 0..12 {
                    // full byte range — exercises the scale packing hi bits
                    scb[j] = ((r * 131 + j * 29 + ib * 7) % 256) as u8;
                }
                w5b.extend_from_slice(&scb);
                let mut qh = [0u8; 32];
                let mut qs = [0u8; 128];
                let n_elem = 256usize.min(id_ - ib * 256);
                for e in 0..n_elem {
                    let s = e / 32;
                    let l = e % 32;
                    let u = ((r * 13 + e * 7 + ib * 5) % 32) as u8;
                    let (s8, m8) = k4_scale(&scb, s);
                    w5dq[r * id_ + ib * 256 + e] = u as f32 * d * s8 as f32 - dmin * m8 as f32;
                    if s % 2 == 0 {
                        qs[(s >> 1) * 32 + l] |= u & 0xF;
                    } else {
                        qs[(s >> 1) * 32 + l] |= (u & 0xF) << 4;
                    }
                    qh[l] |= ((u >> 4) & 1) << s;
                }
                w5b.extend_from_slice(&qh);
                w5b.extend_from_slice(&qs);
            }
        }
        assert_eq!(w5b.len(), od * nbe * 176);

        let mut x8 = vec![0u8; (id_ / 32) * 40];
        for b in 0..id_ / 32 {
            let mut am = 0f32;
            for j in 0..32 {
                am = am.max(xs[b * 32 + j].abs());
            }
            let dd = am / 127.0;
            let di = if dd != 0.0 { 1.0 / dd } else { 0.0 };
            x8[b * 40..b * 40 + 2].copy_from_slice(&half::f16::from_f32(dd).to_le_bytes());
            for j in 0..32 {
                let q = (xs[b * 32 + j] * di).round().clamp(-128.0, 127.0) as i8;
                x8[b * 40 + 4 + j] = q as u8;
            }
        }
        let dq8 = |i: usize| -> f32 {
            half::f16::from_le_bytes([x8[(i / 32) * 40], x8[(i / 32) * 40 + 1]]).to_f32()
                * (x8[(i / 32) * 40 + 4 + (i % 32)] as i8) as f32
        };

        let mut want = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += w5dq[r * id_ + i] * dq8(t * id_ + i);
                }
                want[t * od + r] = acc;
            }
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));

        let mut w5t = Tensor::from_data(
            TensorType::Q5_K,
            &[id_ as i64, od as i64, 1, 1],
            w5b.clone(),
        );
        w5t.name = format!("mw5k{i}");
        cb.state.register_weight(&format!("mw5k{i}"), &w5b);
        // direct decode call: the synthetic shape (od=128) is far below the
        // measured od*id >= 24M dispatch gate (small shapes stay on the f32
        // kernel at runtime); no local model carries Q5_K tensors, so the
        // gate wiring for q5_K is covered by this kernel parity + the shared
        // dispatch code path with q6_K.
        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(od * nt);
        cb.state.q5_k_decode_mmvq(
            cb.state.get_weight_ptr(&format!("mw5k{i}")).unwrap(),
            cb.ptr_of(xb).unwrap(),
            cb.ptr_of(ob).unwrap(),
            od,
            id_,
            nt,
        );
        let got = cb.copy_to_host(ob).unwrap();
        for i in 0..od * nt {
            assert!(
                (got[i] - want[i]).abs() <= scale * 1e-2,
                "q5_k mmvq mismatch at {i}: got {} want {}",
                got[i],
                want[i]
            );
        }
    }
}
