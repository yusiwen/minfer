//! Attention parity: FA prefill, split decode and cross-sequence isolation.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// Step 82: multi-token matmul dispatch — for every quant type, one
/// nt = 3 batched forward must be BITWISE-equal to three nt = 1
/// forwards over the same weight bytes and the same per-token
/// activations. The Step 82 kernels (multi-token MMVQ for the
/// K-quants, in-block token loops for the legacy f32 kernels and the
/// 8c q8-GEMM) preserve the per-(row, token) op order by
/// construction; this test pins it. Shapes are chosen so the nt = 1
/// and nt = 3 paths share the kernel family:
///   - q4_K id 3584 (3584 % 256 == 0 → v2 family) and id 3904
///     (id % 256 != 0 → v1 family), both above the nt == 1 id >= 2048
///     gate,
///   - q5_K od·id >= 24M so nt == 1 rides MMVQ too (v2: id 3072,
///     v1: id 3104),
///   - q6_K od·id >= 4M (padded 224B registration → v2 family; raw
///     210B → v1 family),
///   - the legacy f32 kernels (q8_0 / q4_0 with id > 8192 so the 8c
///     q8-GEMM gate is out / q4_1 / q5_0 / q5_1 / f32) run the same
///     token-looped kernel at nt == 1 and nt == 3.
/// The 8c q4_0 × q8-GEMM arm (nt > 1, id <= 8192) has no nt == 1
/// sibling, so it is checked against an independent host dequant
/// E1b: the CUDA windowed instantiations must give each sequence its own
/// window, matching `cpu_backend`'s `two_sequences_do_not_cross_attend`.
/// Device-gated — CI has no GPU, so it compiles there and runs where one
/// exists (on dgxspark it passes on GB10/sm_121; the original `hd = 2`
/// fixture could not have, see the E1b record).
#[test]
fn cuda_two_sequences_do_not_cross_attend() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_f16_for_test(false); // f32 KV keeps the store and the attention in one dtype

    // One head, hd = 4, two sequences: sequence 0 owns row 0, sequence 1
    // owns row 2. The values make a leak change the answer — query 1 scores
    // 1.0 against sequence 0's key, so a window starting at 0 would blend
    // V(0) into the result instead of returning V(2).
    //
    // hd must be a multiple of 4: the CUDA attention kernels reject anything
    // else (`attention head dim ... outside the kernel's supported range`),
    // so the original hd = 2 fixture could never execute on a device — the
    // test compiled and skipped everywhere until dgxspark got a working GPU.
    let (nh, nk, hd, nt, n_ctx) = (1usize, 1usize, 4usize, 2usize, 4usize);
    let nkt = nk * hd;
    let mut gb = GraphBuilder::new();
    gb.set_explicit_span(true);
    let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
    let st = gb.kvcache_store(0, k, v, n_ctx);
    let kv = gb.kvcache_load(0, nkt, n_ctx, nk);
    let at = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: hd,
            nkt,
            scale: 1.0,
        },
    );
    gb.output(at);
    let g = gb.build();
    assert!(
        g.nodes.iter().any(|n| matches!(
            n.op,
            crate::graph::ops::Op::Attn {
                explicit_span: true,
                ..
            }
        )),
        "the attention node must declare explicit_span"
    );

    let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let kreg = cb.alloc_buffer(n_ctx * nkt);
    let vreg = cb.alloc_buffer(n_ctx * nkt);
    let kb = cb.alloc_buffer(nkt * nt);
    let vb = cb.alloc_buffer(nkt * nt);
    let qb = cb.alloc_buffer(nh * hd * nt);
    let pb = cb.alloc_buffer(nt);
    let sb = cb.alloc_buffer(2 * nt);
    let ob = cb.alloc_buffer(nh * hd * nt);
    // token 0 = [1,0,0,0] (sequence 0), token 1 = [1,0,0,0] (sequence 1)
    cb.write_host(qb, &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
        .unwrap();
    // row 0 = k [1,0,0,0] / v [1,0,0,0]; row 2 = k [0,1,0,0] / v [0,1,0,0]
    cb.write_host(kb, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    cb.write_host(vb, &[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
        .unwrap();
    cb.write_host(pb, &i32bits(&[0, 2])).unwrap();
    cb.write_host(sb, &i32bits(&[0, 2, 1, 3])).unwrap(); // lo block, hi block

    cb.exec_ids(&g.nodes[st], &[kb, vb, pb], kreg, Some((kreg, vreg)))
        .unwrap();
    cb.exec_ids(&g.nodes[at], &[qb, kreg, pb, sb], ob, Some((kreg, vreg)))
        .unwrap();
    // `copy_to_host`, not the trait's `read_host`: CUDA cannot return a
    // borrowed slice of device memory, so its `read_host` is `None` by
    // design (alloc.rs's `copy_to_cpu` CUDA arm does the same copy).
    let got = cb.copy_to_host(ob).unwrap();
    assert!(
        (got[0] - 1.0).abs() < 1e-4
            && got[1].abs() < 1e-4
            && got[2].abs() < 1e-4
            && got[3].abs() < 1e-4,
        "token 0 must attend to its own row: {got:?}"
    );
    assert!(
        got[4].abs() < 1e-4
            && (got[5] - 1.0).abs() < 1e-4
            && got[6].abs() < 1e-4
            && got[7].abs() < 1e-4,
        "token 1 saw the other sequence: {got:?}"
    );
}
#[test]
fn cuda_fa_prefill_attention_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    cb.set_kv_f16_for_test(true);
    let (nh, nk_h, hd) = (4usize, 2usize, 128usize);
    let nkt = nk_h * hd;
    let (nt, n_ctx) = (100usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = (0..nt).collect();

    let mut b = GraphBuilder::new();
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let _store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let at = b.attn(
        q,
        load,
        pp,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let ob_at = cb.alloc_buffer(nh * hd * nt);
    let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
    let to_half =
        |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect() };
    let ks_h = to_half(&ks);
    let vs_h = to_half(&vs);
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
    cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

    cb.exec_ids(
        &g.nodes[_store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
        .unwrap();

    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for (t, &p) in pos.iter().enumerate() {
        kfull[p * nkt..(p + 1) * nkt].copy_from_slice(&ks_h[t * nkt..(t + 1) * nkt]);
        vfull[p * nkt..(p + 1) * nkt].copy_from_slice(&vs_h[t * nkt..(t + 1) * nkt]);
    }
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qs, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    let mut maxe = 0f32;
    for (a, r) in agot.iter().zip(aref.iter()) {
        maxe = maxe.max((a - r).abs());
    }
    println!("fa prefill attention: max err {maxe:.6}");
    assert_close("fa_prefill_f16kv", &agot, &aref, 5e-3);
}
/// #144 item 3: FA prefill over a **packed** cache. Same shape and reference as
/// [`Self::cuda_fa_prefill_attention_parity`] — `nt = 100 > 16` and `hd = 128`,
/// the only combination that reaches the FA prefill — but the cache is Q8_0, so
/// the staging dequantizes each packed cell into the f16 tile instead of
/// copying halves. The reference is the CPU attention over the **same packed
/// bytes** dequantized on the host, so the only difference is the f16 staging:
/// the class the f16 FA gate already pins.
///
/// The observation arm is what makes this a gate about the *packed FA route*
/// rather than about "some attention kernel": the parity arm alone would pass
/// if the launch silently fell back to the general layout-tagged kernel, so the
/// chokepoint's `testfail::note_checked` counter is asserted to have moved.
#[test]
fn cuda_q8_0_fa_prefill_attention_parity() {
    use crate::graph::kvformat::KvFormat;
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_q8_for_test();
    let (nh, nk_h, hd) = (4usize, 2usize, 128usize);
    let nkt = nk_h * hd;
    let row_words = KvFormat::Q8_0.row_elems(nkt);
    let (nt, n_ctx) = (100usize, 128usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = (0..nt).collect();

    let mut b = GraphBuilder::new();
    b.set_kv_format(KvFormat::Q8_0);
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let at = b.attn(
        q,
        load,
        pp,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk_h,
            hd,
            hd_kv: hd,
            nkt,
            scale,
        },
    );
    b.output(at);
    let g = b.build();

    let (xb_q, xb_k, xb_v) = (
        cb.alloc_buffer(nh * hd * nt),
        cb.alloc_buffer(nkt * nt),
        cb.alloc_buffer(nkt * nt),
    );
    let xb_p = cb.alloc_buffer(nt);
    let ob_at = cb.alloc_buffer(nh * hd * nt);
    let (kreg, vreg) = (
        cb.alloc_buffer(n_ctx * row_words),
        cb.alloc_buffer(n_ctx * row_words),
    );

    let qs: Vec<f32> = (0..nh * hd * nt)
        .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
        .collect();
    let ks: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
        .collect();
    let vs: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
        .collect();
    let pb: Vec<f32> = pos.iter().map(|&p| f32::from_bits(p as u32)).collect();
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    cb.write_host(kreg, &vec![0f32; n_ctx * row_words]).unwrap();
    cb.write_host(vreg, &vec![0f32; n_ctx * row_words]).unwrap();

    crate::testfail::reset_checked();
    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
        .unwrap();

    // The reference reads the *same* packed bytes the device wrote (the store is
    // byte-exact against the CPU quantizer, its own gate), dequantized on the
    // host, so the only difference left is the f16 staging.
    let pk = cb.copy_to_host(kreg).unwrap();
    let pv = cb.copy_to_host(vreg).unwrap();
    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for &p in &pos {
        crate::graph::kvformat::unpack_q8_0_cells(
            &pk,
            nkt,
            p,
            1,
            &mut kfull[p * nkt..(p + 1) * nkt],
        );
        crate::graph::kvformat::unpack_q8_0_cells(
            &pv,
            nkt,
            p,
            1,
            &mut vfull[p * nkt..(p + 1) * nkt],
        );
    }
    assert!(
        kfull.iter().any(|x| *x != 0.0),
        "the dequantized reference is all zero; the comparison would be vacuous"
    );
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qs, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    let mut maxe = 0f32;
    for (a, r) in agot.iter().zip(aref.iter()) {
        maxe = maxe.max((a - r).abs());
    }
    println!("q8_0 fa prefill attention: max err {maxe:.6}");
    assert_close("fa_prefill_q8kv", &agot, &aref, 5e-3);

    // Observation arm: the dispatch must have taken the FA launch, not the
    // general layout-tagged fallback.
    assert!(
        crate::testfail::checked("cuda_fa_prefill_q8_0") > 0,
        "the packed prefill did not reach the FA launch (it fell back to the general \
         layout-tagged kernel); this gate would otherwise pass on the old route"
    );
}
// 8d: split-K decode attention parity (nt == 1 routes to the split path).
// nkv = 3 exercises EMPTY splits (positions[0] = 2 → splits 3..7 have no
// rows); nkv = 37 exercises a partial last split with SPLITS = 8. Both KV
// layouts checked. Reference: cpu_gqa_attn over the same KV (zero rows +
// one stored row).
#[test]
fn cuda_attn_split_decode_parity() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // D3-4 L1: shape 2 (hd=128) drives the hybrid kernel on the
    // kv_f16=true arm — n_ctx 4200 covers the runtime rpw dispatch
    // boundary (nkv 1920 -> rpw 15 -> 1-warp body; nkv 1921 -> rpw 16 ->
    // 4-warp body), full 32-row windows and chunk boundaries; shape 1
    // (hd=8) keeps covering the plain 1-warp kernel. n_ctx of shape 1 is
    // sized so pos0 can sweep the ATTN_SPLITS=32 chunk boundaries: full
    // splits, a partially-filled split, and trailing idle splits
    // (mx=-INF/S=0 partials) all get exercised (nkv = pos0 + 1).
    for (nh, nk_h, hd, n_ctx, pos0s) in [
        (
            4usize,
            2usize,
            8usize,
            208usize,
            [2usize, 32, 62, 63, 64, 126, 127, 128, 190, 206, 207],
        ),
        (
            4usize,
            2usize,
            128usize,
            4200usize,
            [2usize, 32, 63, 64, 127, 128, 1023, 1919, 1920, 4094, 4095],
        ),
        // D3-6 2a: the 14B GQA geometry (40:8, gqa=5) drives the
        // GQA-batched kernel (grid (ATTN_SPLITS, 8), 160 threads) on the
        // same pos0 sweep — the 1920/1921 boundary picks between the
        // bitwise 1-warp incumbent (nkv 1920) and the batched body
        // (nkv 1921), and 4094/4095 cover full-window chunk tails.
        (
            40usize,
            8usize,
            128usize,
            4200usize,
            [2usize, 32, 63, 64, 127, 128, 1023, 1919, 1920, 4094, 4095],
        ),
        // D3-6 2a: the 7B GQA geometry (28:4, gqa=7 → 224-thread blocks).
        // nkv 2808 (pos0 2807) reproduces the in-situ decode-step shape at
        // the divergence point seen in the 7B greedy gate.
        (
            28usize,
            4usize,
            128usize,
            4200usize,
            [2usize, 32, 63, 64, 127, 128, 1919, 1920, 2807, 4094, 4095],
        ),
    ] {
        let nkt = nk_h * hd;
        let scale = 1.0 / (hd as f32).sqrt();

        for kv_f16 in [false, true] {
            for pos0 in pos0s {
                let nkv = pos0 + 1;
                cb.set_kv_f16_for_test(kv_f16);
                let mut b = GraphBuilder::new();
                let q = b.input("q", [nh * hd, 1, 1, 1], DType::F32);
                let k = b.input("k", [nkt, 1, 1, 1], DType::F32);
                let v = b.input("v", [nkt, 1, 1, 1], DType::F32);
                let pp = b.input("positions", [1, 1, 1, 1], DType::I32);
                let store = b.kvcache_store(0, k, v, n_ctx);
                let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
                let at = b.attn(
                    q,
                    load,
                    pp,
                    AttnMode::Gqa,
                    AttnMeta {
                        layer: 0,
                        n_head: nh,
                        n_head_kv: nk_h,
                        hd,
                        hd_kv: hd,
                        nkt,
                        scale,
                    },
                );
                b.output(at);
                let g = b.build();

                let (xb_q, xb_k, xb_v) = (
                    cb.alloc_buffer(nh * hd),
                    cb.alloc_buffer(nkt),
                    cb.alloc_buffer(nkt),
                );
                let xb_p = cb.alloc_buffer(1);
                let ob_at = cb.alloc_buffer(nh * hd);
                let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

                let qs: Vec<f32> = (0..nh * hd)
                    .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
                    .collect();
                let ks: Vec<f32> = (0..nkt)
                    .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
                    .collect();
                let vs: Vec<f32> = (0..nkt)
                    .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
                    .collect();
                let pb = vec![f32::from_bits(pos0 as u32)];
                let to_half = |x: &[f32]| -> Vec<f32> {
                    x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
                };
                let (ks_r, vs_r) = if kv_f16 {
                    (to_half(&ks), to_half(&vs))
                } else {
                    (ks.clone(), vs.clone())
                };
                cb.write_host(xb_q, &qs).unwrap();
                cb.write_host(xb_k, &ks).unwrap();
                cb.write_host(xb_v, &vs).unwrap();
                cb.write_host(xb_p, &pb).unwrap();
                cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
                cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

                cb.exec_ids(
                    &g.nodes[store],
                    &[xb_k, xb_v, xb_p],
                    kreg,
                    Some((kreg, vreg)),
                )
                .unwrap();
                cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
                    .unwrap();

                let mut kfull = vec![0f32; nkt * n_ctx];
                let mut vfull = vec![0f32; nkt * n_ctx];
                kfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&ks_r);
                vfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&vs_r);
                let mut aref = vec![0f32; nh * hd];
                crate::graph::cpu_backend::cpu_gqa_attn(
                    &qs,
                    &kfull,
                    &vfull,
                    &crate::graph::cpu_backend::causal_span(&[pos0]),
                    1,
                    nh,
                    nk_h,
                    hd,
                    hd,
                    nkt,
                    &mut aref,
                    scale,
                )
                .unwrap();
                let agot = cb.copy_to_host(ob_at).unwrap();
                assert_close(
                    &format!("attn_split(f16kv={kv_f16}, nkv={nkv})"),
                    &agot,
                    &aref,
                    1e-4,
                );
            }
        }
    }

    // D3-6 2a: kernel-level gate of the calibrated tolerance package on
    // realistic outlier-scale data (docs/CUDA_OPTIMIZATION.md §2D D3a:
    // residual |q|~50, V outliers ±127 — the h4w body measured 6.5e-5
    // vs CPU on this class, the incumbent 3.8e-5). The GQA-batched body
    // shares the h4w window loop verbatim, so the same ≤1e-4 bound
    // applies; 14B geometry (40:8, gqa=5) inside the batched regime
    // (nkv 1921 / 4096, f16 KV).
    for pos0 in [1920usize, 4095usize] {
        let (nh, nk_h, hd, n_ctx) = (40usize, 8usize, 128usize, 4200usize);
        let nkt = nk_h * hd;
        let scale = 1.0 / (hd as f32).sqrt();
        let nkv = pos0 + 1;
        cb.set_kv_f16_for_test(true);
        let mut b = GraphBuilder::new();
        let q = b.input("q", [nh * hd, 1, 1, 1], DType::F32);
        let k = b.input("k", [nkt, 1, 1, 1], DType::F32);
        let v = b.input("v", [nkt, 1, 1, 1], DType::F32);
        let pp = b.input("positions", [1, 1, 1, 1], DType::I32);
        let store = b.kvcache_store(0, k, v, n_ctx);
        let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
        let at = b.attn(
            q,
            load,
            pp,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b.output(at);
        let g = b.build();

        let (xb_q, xb_k, xb_v) = (
            cb.alloc_buffer(nh * hd),
            cb.alloc_buffer(nkt),
            cb.alloc_buffer(nkt),
        );
        let xb_p = cb.alloc_buffer(1);
        let ob_at = cb.alloc_buffer(nh * hd);
        let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

        // Outlier scale: q residual |q|~50-60, V outliers |v|~140 (f16
        // representable); K stays at the tame scale like the D3a probe.
        let qs: Vec<f32> = (0..nh * hd)
            .map(|i| (((i * 37) % 19) as f32 / 5.0 - 1.9) * 30.0)
            .collect();
        let ks: Vec<f32> = (0..nkt)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let vs: Vec<f32> = (0..nkt)
            .map(|i| (((i * 57) % 11) as f32 / 3.0 - 1.8) * 80.0)
            .collect();
        let pb = vec![f32::from_bits(pos0 as u32)];
        let to_half = |x: &[f32]| -> Vec<f32> {
            x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect()
        };
        let (ks_r, vs_r) = (to_half(&ks), to_half(&vs));
        cb.write_host(xb_q, &qs).unwrap();
        cb.write_host(xb_k, &ks).unwrap();
        cb.write_host(xb_v, &vs).unwrap();
        cb.write_host(xb_p, &pb).unwrap();
        cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
        cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

        cb.exec_ids(
            &g.nodes[store],
            &[xb_k, xb_v, xb_p],
            kreg,
            Some((kreg, vreg)),
        )
        .unwrap();
        cb.exec_ids(&g.nodes[at], &[xb_q, kreg, xb_p], ob_at, Some((kreg, vreg)))
            .unwrap();

        let mut kfull = vec![0f32; nkt * n_ctx];
        let mut vfull = vec![0f32; nkt * n_ctx];
        kfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&ks_r);
        vfull[pos0 * nkt..(pos0 + 1) * nkt].copy_from_slice(&vs_r);
        let mut aref = vec![0f32; nh * hd];
        crate::graph::cpu_backend::cpu_gqa_attn(
            &qs,
            &kfull,
            &vfull,
            &crate::graph::cpu_backend::causal_span(&[pos0]),
            1,
            nh,
            nk_h,
            hd,
            hd,
            nkt,
            &mut aref,
            scale,
        )
        .unwrap();
        let agot = cb.copy_to_host(ob_at).unwrap();
        assert_close(
            &format!("attn_split_gqa_batched_outlier(nkv={nkv})"),
            &agot,
            &aref,
            1e-4,
        );
    }
}
