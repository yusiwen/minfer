//! Element-wise, norm and embedding-row parity.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// D5-R follow-up (doc 89): row-marginal localization bench for the
/// multi-token matmul kernels. Runs the REAL dispatch path (graph
/// execute_node -> quantize + kernel) at nt = 1..8 over real 14B shapes
/// with a cold-L2 protocol: each nt owns NC independent weight copies
/// (>L2 aggregate) cycled so no copy is revisited within 2 runs — L2 is
/// evicted between uses exactly like in a real forward, and runs per
/// (uid, range) stay below the 3-run capture trigger so timing is never
/// capture/replay. Per-run cost = one synchronized burst / R; the
/// per-row marginal = (t(nt) - t(1)) / (nt - 1), attributable to extra
/// in-kernel row work only (launch count is nt-invariant).
#[test]
fn cuda_row_marginal_bench() {
    if std::env::var("MINFER_BENCH_ROW_MARGINAL").is_err() {
        return;
    }
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();

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

    // (label, type, od, id, padded, weight copies). Aggregate weight
    // footprint per case > 126 MB L2, and >= 8x per-copy footprint
    // streams between revisits.
    let cases: Vec<(&str, TensorType, usize, usize, bool, usize)> = vec![
        ("q4k_attn_qo", TensorType::Q4_K, 5120, 5120, false, 40),
        ("q4k_ffn_up", TensorType::Q4_K, 13824, 5120, false, 40),
        ("q6k_ffn_down", TensorType::Q6_K, 5120, 13824, true, 24),
    ];
    let nts = [1usize, 2, 3, 4, 6, 8];

    for (ci, (label, tt, od, id_, padded, nc)) in cases.into_iter().enumerate() {
        let nbe = (id_ + 255) / 256;
        let row_bytes = match tt {
            TensorType::Q4_K => nbe * 144,
            TensorType::Q6_K => nbe * 210,
            other => panic!("unexpected {other:?}"),
        };
        let wb = gen_bytes(od * row_bytes, 0x5EED_C0DE + ci as u64);
        let mut wts = Vec::with_capacity(nc);
        for j in 0..nc {
            let name = format!("w{ci}_{j}");
            let mut wt = Tensor::from_data(tt, &[id_ as i64, od as i64, 1, 1], wb.clone());
            wt.name = name.clone();
            if tt == TensorType::Q6_K && padded {
                cb.state.register_weight_q6k_padded(&name, &wb, od, id_);
            } else {
                cb.state.register_weight(&name, &wb);
            }
            wts.push(wt);
        }

        let mut lines = Vec::new();
        for &nt in nts.iter() {
            // NC graphs (one per weight copy) sharing one x/out buffer.
            let xs: Vec<f32> = (0..id_ * nt)
                .map(|i| ((i * 2654435761 % 2000) as f32 / 1000.0 - 1.0))
                .collect();
            let mut graphs = Vec::with_capacity(nc);
            for wt in &wts {
                let mut b = GraphBuilder::new();
                let x = b.input("x", [id_, nt, 1, 1], DType::F32);
                let m = b.matmul(x, wt, None);
                b.output(m);
                let g = b.build();
                graphs.push(g);
            }
            let xb = cb.alloc_buffer(id_ * nt);
            cb.write_host(xb, &xs).unwrap();
            let ob = cb.alloc_buffer(od * nt);

            let reps = 2 * nc; // < 3 runs per (uid, range): no capture
            let t0 = std::time::Instant::now();
            for r in 0..reps {
                cb.exec_ids(
                    &graphs[r % nc].nodes[graphs[r % nc].outputs[0]],
                    &[xb],
                    ob,
                    None,
                )
                .unwrap();
            }
            cb.synchronize();
            let per_us = t0.elapsed().as_secs_f64() * 1e6 / reps as f64;
            lines.push((nt, per_us));
            let _ = cb.copy_to_host(ob).unwrap(); // keep result live
        }
        let t1 = lines[0].1;
        let gb = (od * row_bytes) as f64 / 1e9;
        eprintln!("[bench] {label} ({tt:?} {od}x{id_}, {gb:.1} MB/copy, NC={nc}):");
        for (nt, us) in &lines {
            let bw = gb * 1e3 / (us / 1e3) / 1e3;
            eprintln!(
                "[bench]   nt={nt}: {:9.1} us/run  ({bw:5.0} GB/s w-stream)",
                us
            );
        }
        for w in [2usize, 3, 4] {
            let tn = lines[w - 1].1;
            eprintln!(
                "[bench]   marginal/row (1->{w}): {:6.2} us per matmul per forward",
                (tn - t1) / (w - 1) as f64
            );
        }
        let _ = &cb;
    }
}
#[test]
fn cuda_elementwise_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let n = 257usize; // odd size exercises the elementwise tail guard
    let mut b = GraphBuilder::new();
    let a = b.input("a", [n, 1, 1, 1], DType::F32);
    let c = b.input("c", [n, 1, 1, 1], DType::F32);
    let add = b.add(a, c);
    let mul = b.mul(add, c);
    let sw = b.swiglu(mul, c); // gate is the RAW pre-activation
    let silu = b.silu(mul);
    b.output(silu);
    b.output(sw);
    let g = b.build();

    let (x, y) = (cb.alloc_buffer(n), cb.alloc_buffer(n));
    let (t1, t2, t3) = (cb.alloc_buffer(n), cb.alloc_buffer(n), cb.alloc_buffer(n));
    let xs: Vec<f32> = (0..n).map(|i| ((i * 37) % 23) as f32 / 4.0 - 2.5).collect();
    let ys: Vec<f32> = (0..n).map(|i| ((i * 91) % 17) as f32 / 3.0 - 2.0).collect();
    cb.write_host(x, &xs).unwrap();
    cb.write_host(y, &ys).unwrap();

    cb.exec_ids(&g.nodes[add], &[x, y], t1, None).unwrap();
    cb.exec_ids(&g.nodes[mul], &[t1, y], t2, None).unwrap();
    // SwiGLU consumes the RAW mul output, so it must run before the
    // in-place Silu overwrites t2 (alias path, graph rules §5).
    cb.exec_ids(&g.nodes[sw], &[t2, y], t3, None).unwrap();
    cb.exec_ids(&g.nodes[silu], &[t2], t2, None).unwrap();

    // Host reference through the same vec_ops the CPU backend uses.
    let mut r1 = vec![0f32; n];
    crate::vec_ops::vec_add_f32(n, &mut r1, &xs, &ys);
    assert_eq!(cb.copy_to_host(t1).unwrap(), r1, "add must be bit-exact");
    let mut r2 = vec![0f32; n];
    crate::vec_ops::vec_mul_f32(n, &mut r2, &r1, &ys);
    let mut r3 = vec![0f32; n];
    crate::vec_ops::vec_silu_f32(n, &mut r3, &r2);
    let got2 = cb.copy_to_host(t2).unwrap();
    assert_close("mul+silu (in-place)", &got2, &r3, 1e-5);
    let mut r4 = vec![0f32; n];
    crate::vec_ops::vec_mul_f32(n, &mut r4, &r3, &ys);
    assert_close("swiglu", &cb.copy_to_host(t3).unwrap(), &r4, 1e-5);
}
#[test]
fn cuda_norm_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // RmsNorm: d=64, nt=5
    let (d, nt) = (64usize, 5usize);
    let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
    let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
    cb.state.register_weight("nw", &wbytes);
    let mut wt = Tensor::from_data(TensorType::F32, &[d as i64, 1, 1, 1], wbytes.clone());
    wt.name = "nw".to_string();

    // QkNorm: hd=16, nh=4, nt=3 — rows (t*nh + h) form a contiguous
    // [nt*nh, hd] matrix, so the same rms_norm kernel covers it.
    let (hd, nh, nt2) = (16usize, 4usize, 3usize);
    let qw: Vec<f32> = (0..hd).map(|i| 1.0 / (1.0 + i as f32)).collect();
    let qbytes: Vec<u8> = qw.iter().flat_map(|v| v.to_le_bytes()).collect();
    cb.state.register_weight("qw", &qbytes);
    let mut qwt = Tensor::from_data(TensorType::F32, &[hd as i64, 1, 1, 1], qbytes);
    qwt.name = "qw".to_string();

    let mut b = GraphBuilder::new();
    let x = b.input("x", [d, nt, 1, 1], DType::F32);
    let rn = b.rms_norm(x, Some(&wt), 1e-5);
    let q = b.input("q", [hd * nh, nt2, 1, 1], DType::F32);
    let qn = b.qk_norm(q, Some(&qwt), hd, nh, 1e-5);
    b.output(rn);
    b.output(qn);
    let g = b.build();

    let xb = cb.alloc_buffer(d * nt);
    let xs: Vec<f32> = (0..d * nt)
        .map(|i| ((i * 53) % 31) as f32 / 7.0 - 2.0)
        .collect();
    cb.write_host(xb, &xs).unwrap();
    let ob = cb.alloc_buffer(d * nt);
    cb.exec_ids(&g.nodes[rn], &[xb], ob, None).unwrap();

    let qb = cb.alloc_buffer(hd * nh * nt2);
    let qs: Vec<f32> = (0..hd * nh * nt2)
        .map(|i| ((i * 71) % 29) as f32 / 6.0 - 2.5)
        .collect();
    cb.write_host(qb, &qs).unwrap();
    let qo = cb.alloc_buffer(hd * nh * nt2);
    cb.exec_ids(&g.nodes[qn], &[qb], qo, None).unwrap();

    let mut want = vec![0f32; d * nt];
    for t in 0..nt {
        crate::vec_ops::rms_norm_fused_f32(
            d,
            &mut want[t * d..(t + 1) * d],
            &xs[t * d..(t + 1) * d],
            &w,
            1e-5,
        );
    }
    assert_close("rms_norm", &cb.copy_to_host(ob).unwrap(), &want, 1e-4);

    let mut want2 = vec![0f32; hd * nh * nt2];
    for r in 0..nh * nt2 {
        crate::vec_ops::rms_norm_fused_f32(
            hd,
            &mut want2[r * hd..(r + 1) * hd],
            &qs[r * hd..(r + 1) * hd],
            &qw,
            1e-5,
        );
    }
    assert_close("qk_norm", &cb.copy_to_host(qo).unwrap(), &want2, 1e-4);
}
/// 7e③: embedding / row-gather parity. Device embed kernels (one per
/// supported weight type, incl. the padded Q6_K layout) must match
/// `kernel::embed_tokens` — the CPU path these nodes used before 7e③ —
/// and the generic f32 gather (G3 tail get_rows) must match a manual
/// row copy.
#[test]
fn cuda_embed_getrows_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (vocab, n_embd, nt) = (6usize, 512usize, 3usize); // 2 super-blocks/row
    let ids: Vec<u32> = vec![0, 5, 2];
    let ids_f32: Vec<f32> = ids.iter().map(|&i| f32::from_bits(i)).collect();

    // ── build one tensor per supported type (rows = vocab) ──
    // f32
    let wf: Vec<f32> = (0..vocab * n_embd)
        .map(|i| (((i as u64) * 2654435761 % 1009) as f32) / 504.0 - 1.0)
        .collect();
    let wf_bytes: Vec<u8> = wf.iter().flat_map(|f| f.to_le_bytes()).collect();
    let mut tf = Tensor::from_data(
        TensorType::F32,
        &[n_embd as i64, vocab as i64, 1, 1],
        wf_bytes.clone(),
    );
    tf.name = "ewf32".to_string();

    // q8_0 (34B blocks: f16 d + 32 i8)
    let mut w8 = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 32 {
            let d = 0.02f32 + 0.003 * ((r * 5 + ib) % 7) as f32;
            w8.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for i in 0..32 {
                w8.push((((r * 37 + ib * 17 + i * 3) % 255) as i8 as u8).wrapping_add(0));
            }
        }
    }
    let mut t8 = Tensor::from_data(
        TensorType::Q8_0,
        &[n_embd as i64, vocab as i64, 1, 1],
        w8.clone(),
    );
    t8.name = "ewq8".to_string();

    // q4_0 (18B blocks: f16 d + 16 nibble bytes; elem j = LOW of byte j)
    let mut w40 = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 32 {
            let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
            w40.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for i in 0..16 {
                let lo = ((r * 11 + ib * 7 + i * 3) % 15) as u8;
                let hi = ((r * 7 + ib * 5 + i) % 15) as u8;
                w40.push(lo | (hi << 4));
            }
        }
    }
    let mut t40 = Tensor::from_data(
        TensorType::Q4_0,
        &[n_embd as i64, vocab as i64, 1, 1],
        w40.clone(),
    );
    t40.name = "ewq40".to_string();

    // q5_0 (22B blocks: f16 d + u32 qh + 16 nibble bytes; value =
    // nibble + 16*high_bit - 16) — the tok_embd type of 0.5B q4_k_m GGUFs
    let mut w50 = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 32 {
            let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
            w50.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            let qh: u32 = (((r * 13 + ib * 7) % 5) as u32) << 17
                | (((r * 5 + ib * 3) % 7) as u32) << 3
                | 0b101;
            w50.extend_from_slice(&qh.to_le_bytes());
            for i in 0..16 {
                let lo = ((r * 11 + ib * 7 + i * 3) % 31) as u8;
                let hi = ((r * 7 + ib * 5 + i) % 31) as u8;
                w50.push(lo | (hi << 4));
            }
        }
    }
    let mut t50 = Tensor::from_data(
        TensorType::Q5_0,
        &[n_embd as i64, vocab as i64, 1, 1],
        w50.clone(),
    );
    t50.name = "ewq50".to_string();

    // q4_k (144B super-blocks) — same generator scheme as the matmul test
    let mut w4k = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 256 {
            let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
            let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
            w4k.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            w4k.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
            for j in 0..12 {
                w4k.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
            }
            for j in 0..128 {
                let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                w4k.push(lo | (hi << 4));
            }
        }
    }
    let mut t4k = Tensor::from_data(
        TensorType::Q4_K,
        &[n_embd as i64, vocab as i64, 1, 1],
        w4k.clone(),
    );
    t4k.name = "ewq4k".to_string();

    // q6_k (210B raw; also registered padded)
    let mut w6k = Vec::new();
    for r in 0..vocab {
        for ib in 0..n_embd / 256 {
            let d = 0.027f32 + 0.004 * ((r * 11 + ib * 7) % 6) as f32;
            for i in 0..128 {
                w6k.push(((r * 29 + i * 7 + ib * 3) % 255) as u8);
            }
            for i in 0..64 {
                w6k.push(((r * 17 + i * 13 + ib * 11) % 255) as u8);
            }
            for i in 0..16 {
                w6k.push(((((r * 5 + i * 3 + ib) % 15) as i8) - 7) as u8);
            }
            w6k.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
        }
    }
    let mut t6k = Tensor::from_data(
        TensorType::Q6_K,
        &[n_embd as i64, vocab as i64, 1, 1],
        w6k.clone(),
    );
    t6k.name = "ewq6k".to_string();
    let mut t6kp = Tensor::from_data(
        TensorType::Q6_K,
        &[n_embd as i64, vocab as i64, 1, 1],
        w6k.clone(),
    );
    t6kp.name = "ewq6kp".to_string();

    // ── register + build one graph with an embed node per type ──
    cb.state.register_weight("ewf32", &wf_bytes);
    cb.state.register_weight("ewq8", &w8);
    cb.state.register_weight("ewq40", &w40);
    cb.state.register_weight("ewq50", &w50);
    cb.state.register_weight("ewq4k", &w4k);
    cb.state.register_weight("ewq6k", &w6k);
    cb.state
        .register_weight_q6k_padded("ewq6kp", &w6k, vocab, n_embd);
    assert!(cb.state.is_weight_padded("ewq6kp"));

    let mut b = GraphBuilder::new();
    let ids_in = b.input("ids", [nt, 1, 1, 1], DType::F32);
    let e_f32 = b.embedding(ids_in, &tf);
    let e_q8 = b.embedding(ids_in, &t8);
    let e_q40 = b.embedding(ids_in, &t40);
    let e_q50 = b.embedding(ids_in, &t50);
    let e_q4k = b.embedding(ids_in, &t4k);
    let e_q6k = b.embedding(ids_in, &t6k);
    let e_q6kp = b.embedding(ids_in, &t6kp);
    // ── 7e③ model-shape q4_0 case (0.5B): n_embd=896 (nb=28 blocks),
    // large ids — the exact shape that E2E first exercised ──
    let (mv, me) = (10000usize, 896usize);
    let mids: Vec<u32> = vec![785, 6722, 315, 9625, 374];
    let mut mw = Vec::new();
    for r in 0..mv {
        for ib in 0..me / 32 {
            let d = 0.03f32 + 0.004 * ((r * 3 + ib * 2) % 5) as f32;
            mw.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for i in 0..16 {
                let lo = ((r * 11 + ib * 7 + i * 3) % 15) as u8;
                let hi = ((r * 7 + ib * 5 + i) % 15) as u8;
                mw.push(lo | (hi << 4));
            }
        }
    }
    let mut mt = Tensor::from_data(TensorType::Q4_0, &[me as i64, mv as i64, 1, 1], mw.clone());
    mt.name = "ewq40m".to_string();
    cb.state.register_weight("ewq40m", &mw);
    let mids_f32: Vec<f32> = mids.iter().map(|&i| f32::from_bits(i)).collect();
    let midb = cb.alloc_buffer(mids.len());
    cb.write_host(midb, &mids_f32).unwrap();
    let mout = cb.alloc_buffer(me * mids.len());
    let mut mb = GraphBuilder::new();
    let mi = mb.input("mids", [mids.len(), 1, 1, 1], DType::F32);
    let me_node = mb.embedding(mi, &mt);
    mb.output(me_node);
    let mg = mb.build();
    cb.exec_ids(&mg.nodes[me_node], &[midb], mout, None)
        .unwrap();
    {
        let got = cb.copy_to_host(mout).unwrap();
        let mut want = vec![0f32; me * mids.len()];
        crate::kernel::embed_tokens(&mids, &mt, &mut want, me);
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("embed q4_0 model-shape", &got, &want, scale * 2e-3);
    }

    // generic gather (G3 tail): x[ids[t]] — the source has vocab rows so
    // every id is in range
    let xin = b.input("x", [n_embd, vocab, 1, 1], DType::F32);
    let gr = b.get_rows(xin, ids_in, [n_embd, nt, 1, 1]);
    for n in [e_f32, e_q8, e_q40, e_q50, e_q4k, e_q6k, e_q6kp, gr] {
        b.output(n);
    }
    let g = b.build();

    let idsb = cb.alloc_buffer(nt);
    cb.write_host(idsb, &ids_f32).unwrap();
    let xvals: Vec<f32> = (0..n_embd * vocab)
        .map(|i| (((i as u64) * 1103515245 % 997) as f32) / 500.0 - 1.0)
        .collect();
    let xb = cb.alloc_buffer(n_embd * vocab);
    cb.write_host(xb, &xvals).unwrap();

    let mut outs = Vec::new();
    for node in [e_f32, e_q8, e_q40, e_q50, e_q4k, e_q6k, e_q6kp] {
        let out = cb.alloc_buffer(n_embd * nt);
        cb.exec_ids(&g.nodes[node], &[idsb], out, None).unwrap();
        outs.push(out);
    }
    let grb = cb.alloc_buffer(n_embd * nt);
    cb.exec_ids(&g.nodes[gr], &[xb, idsb], grb, None).unwrap();

    // ── references ──
    let names = ["f32", "q8_0", "q4_0", "q5_0", "q4_k", "q6_k", "q6_k padded"];
    let tensors = [&tf, &t8, &t40, &t50, &t4k, &t6k, &t6kp];
    for ((name, t), &ob) in names.iter().zip(tensors).zip(outs.iter()) {
        let got = cb.copy_to_host(ob).unwrap();
        let mut want = vec![0f32; n_embd * nt];
        if t.ttype == TensorType::F32 {
            // embed_tokens handles the quantized types; f32 is a row copy
            for (ti, &id) in ids.iter().enumerate() {
                let src = (id as usize) * n_embd;
                want[ti * n_embd..(ti + 1) * n_embd].copy_from_slice(&wf[src..src + n_embd]);
            }
        } else {
            crate::kernel::embed_tokens(&ids, t, &mut want, n_embd);
        }
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close(&format!("embed {name}"), &got, &want, scale * 2e-3);
    }
    let got = cb.copy_to_host(grb).unwrap();
    for t in 0..nt {
        let id = ids[t] as usize;
        for i in 0..n_embd {
            let want = xvals[id * n_embd + i];
            assert!(
                (got[t * n_embd + i] - want).abs() <= 1e-6 * (1.0 + want.abs()),
                "gather [{t},{i}]: got {} want {want}",
                got[t * n_embd + i]
            );
        }
    }
}
