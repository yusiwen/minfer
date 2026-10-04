//! KV cells, the packed store, and the attention round trips over each KV dtype.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn kv_persistent_regions_survive_realloc() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut b = crate::graph::builder::GraphBuilder::new();
    let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
    let k = b.input("k", [16, 1, 1, 1], DType::F32);
    let v = b.input("v", [16, 1, 1, 1], DType::F32);
    let store = b.kvcache_store(0, k, v, 1024);
    let load = b.kvcache_load(0, 16, 1024, 2);
    b.output(load);
    let mut g = b.build();
    g.nodes[store].backend = Some(crate::graph::Backend::CUDA);
    g.nodes[load].backend = Some(crate::graph::Backend::CUDA);

    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    alloc.alloc_graph(&g).unwrap();
    let pair = alloc.kv_pair(0).unwrap();

    // the store node's buffer IS the K region, on the CUDA pool
    let kbuf = alloc.node_buffer(store).unwrap();
    assert_eq!(kbuf.backend, crate::graph::Backend::CUDA);
    assert_eq!(kbuf.id, pair.0);
    {
        let c = alloc.cuda_mut().unwrap();
        c.write_host(kbuf.id, &[7.5f32; 16]).unwrap();
    }

    // rebuild: liveness buffers recycle, KV regions survive unchanged
    alloc.alloc_graph(&g).unwrap();
    assert_eq!(alloc.kv_pair(0).unwrap(), pair);
    let back = alloc.copy_to_cpu(store).unwrap();
    assert_eq!(&back[..16], &[7.5f32; 16]);
}
/// C8b S4: a cell move must stride by a **row's** size, whatever the KV dtype
/// stores. With an f16 cache a row is `nkt` halves = `nkt / 2` f32 elements,
/// while `copy_cells`'s `elems_per_cell` is a count of f32 — the unit the move
/// kernel walks by. Passing `nkt` there moved every row twice as far as it
/// should, so a copy-on-write (or a compaction) on an f16 device wrote the
/// wrong cells; the real-model hd = 128 gate found it, and this pins it.
#[test]
fn cuda_f16_kv_cell_move_strides_by_row_bytes() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 64;
    // A row of `nkt` halves, stored in the first `nkt / 2` f32 slots.
    let nkt = 8usize;
    cb.set_kv_f16_for_test(true);
    let (src_row, dst_row, rows) = (2usize, 10usize, 4usize);
    let slots = N_CTX * (nkt / 2);
    let region = cb.alloc_buffer(slots);
    // Every f32 slot carries its own index in its low 16 bits, so the halves a
    // kernel would read are distinct and the *slot* layout is verifiable.
    let pattern: Vec<f32> = (0..slots)
        .map(|i| f32::from_bits(((i as u32) * 7 + 1) & 0xFFFF))
        .collect();
    cb.write_host(region, &pattern).unwrap();
    let before = pattern.clone();
    let r = BufRef::own(crate::graph::Backend::CUDA, region, slots);
    cb.copy_cells(r, r, dst_row, src_row, rows, nkt).unwrap();
    let after = cb.copy_to_host(region).unwrap();
    assert_eq!(after.len(), slots);
    for r in 0..rows {
        // Slot view of a row: `row * nkt / 2` f32.
        let sd = (src_row + r) * (nkt / 2);
        let dd = (dst_row + r) * (nkt / 2);
        assert_eq!(
            &after[dd..dd + nkt / 2],
            &before[sd..sd + nkt / 2],
            "row {r} did not land at row {} (f16 KV strides in f32 elements)",
            dst_row + r
        );
    }
    // Nothing outside the destination rows may move.
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        let moved = (dst_row * nkt / 2..(dst_row + rows) * nkt / 2).contains(&i);
        if !moved {
            assert_eq!(x, y, "cell move touched f32 slot {i} outside its rows");
        }
    }
}
/// C4 S2b: the packed twin of
/// [`Self::cuda_f16_kv_cell_move_strides_by_row_bytes`]. A Q8_0 cell is a whole
/// number of f32 words (`KvFormat::Q8_0.row_elems(nkt)` = ceil(nkt/32*34 / 4)),
/// and that is exactly the unit `copy_cells`'s `elems_per_cell` already carries
/// — the caller (`GraphAllocator::kv_set_cap_with_defrag`, the compaction and
/// the copy-on-write) passes `region.elems / n_ctx`. So the device backend must
/// pass it through **unchanged**: applying the f16 halving here would move
/// `row_elems / 2` words per row, land every moved cell short of its slot and
/// silently corrupt the arena on the first compaction.
///
/// `nkt = 64` is chosen so the two candidate strides cannot be confused:
/// `row_elems` is 17 words, while the pre-C4 `nkt` (and the f16 half of it) are
/// 64 and 32.
#[test]
fn cuda_q8_0_kv_cell_move_strides_by_row_bytes() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 64;
    let nkt = 64usize;
    let row_elems = crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt);
    assert_eq!(row_elems, 17, "the fixture assumes 2 Q8_0 blocks of 34 B");
    cb.set_kv_q8_for_test();
    let (src_row, dst_row, rows) = (2usize, 10usize, 4usize);
    let slots = N_CTX * row_elems;
    let region = cb.alloc_buffer(slots);
    // Distinct word per slot, so a move of the wrong length is visible as a
    // mismatched suffix *and* as a stray write outside the destination rows.
    let pattern: Vec<f32> = (0..slots)
        .map(|i| f32::from_bits(((i as u32) * 2654435761) | 1))
        .collect();
    cb.write_host(region, &pattern).unwrap();
    let before = pattern.clone();
    let r = BufRef::own(crate::graph::Backend::CUDA, region, slots);
    // The host passes the packed cell's word count, exactly as alloc.rs does.
    cb.copy_cells(r, r, dst_row, src_row, rows, row_elems)
        .unwrap();
    let after = cb.copy_to_host(region).unwrap();
    assert_eq!(after.len(), slots);
    for r in 0..rows {
        let sd = (src_row + r) * row_elems;
        let dd = (dst_row + r) * row_elems;
        // Compared as **bits**: the pattern is arbitrary words and some are NaN,
        // for which `==` is never true (the same reason C4's V-verbatim gate
        // compares bits).
        let same = after[dd..dd + row_elems]
            .iter()
            .zip(&before[sd..sd + row_elems])
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            same,
            "packed row {r} did not land at row {} (a Q8_0 cell is {row_elems} f32 words)",
            dst_row + r
        );
    }
    for (i, (x, y)) in before.iter().zip(&after).enumerate() {
        let moved = (dst_row * row_elems..(dst_row + rows) * row_elems).contains(&i);
        if !moved {
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "packed cell move touched f32 slot {i} outside its rows"
            );
        }
    }
}
/// C4 S2b: the Q8_0 store, two ways. (a) The bytes the device writes for a row
/// are **exactly** `kvformat::pack_q8_0_cell`'s — the CPU store's own output —
/// so a CPU/device Q8_0 comparison is a layout check, not a tolerance question.
/// (b) Those bytes survive the pool round trip (`copy_to_host` returns the same
/// words), which is what a physical shift's dequantize → re-rope → requantize
/// reads and writes back.
///
/// Mutation-checked by hand: pointing the kernel at `elem % 32` instead of
/// `(elem >> 5, elem & 31)` (i.e. dropping the block base) fails (a) on the
/// second block; storing the scale at byte 2 and the quants at 0 fails (a) on
/// every element.
#[test]
fn cuda_q8_0_store_matches_the_cpu_quantizer() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 64;
    const NT: usize = 5;
    // hd = 64 (two blocks per row) and hd = 32 (one) so a block-base mistake is
    // exercised, not just an offset one.
    for nkt in [64usize, 32] {
        let row_elems = crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt);
        let row_bytes = crate::graph::kvformat::KvFormat::Q8_0.row_bytes(nkt);
        cb.set_kv_q8_for_test();
        // Rows with a wide dynamic range inside each block, so `amax/127` is not
        // a degenerate 0 or 1 and the quants exercise the round/clamp.
        let rows: Vec<f32> = (0..NT * nkt)
            .map(|i| {
                let b = (i % nkt) / 32;
                let sign = if (i / 7) % 2 == 0 { 1.0 } else { -1.0 };
                sign * (((i * 37 % 101) as f32) / 101.0) * (b as f32 + 1.0) * 2.5
            })
            .collect();
        let pos: Vec<usize> = vec![0, 3, 7, 11, 15];
        let region = cb.alloc_buffer(N_CTX * row_elems);
        let src = cb.alloc_buffer(NT * nkt);
        let posb = cb.alloc_buffer(NT);
        cb.write_host(src, &rows).unwrap();
        cb.write_host(
            posb,
            &pos.iter()
                .map(|&p| f32::from_bits(p as u32))
                .collect::<Vec<f32>>(),
        )
        .unwrap();
        cb.write_host(region, &vec![0f32; N_CTX * row_elems])
            .unwrap();
        cb.state.store_kv_q8_0(
            cb.ptr_of(src).unwrap(),
            cb.ptr_of(region).unwrap(),
            nkt,
            NT,
            row_bytes,
            cb.ptr_of(posb).unwrap(),
        );

        let after = cb.copy_to_host(region).unwrap();
        // (a) byte-for-byte against the CPU quantizer, per real row. Compared as
        // **bits**: a packed word is an f16 scale and int8 quants, so as f32 it
        // is frequently NaN and `==` on it is never true.
        for (t, &p) in pos.iter().enumerate() {
            let row_f32 = &rows[t * nkt..(t + 1) * nkt];
            let expected: Vec<f32> = {
                let mut w = vec![0f32; row_elems];
                crate::graph::kvformat::pack_q8_0_cell(&mut w, nkt, row_f32);
                w
            };
            let same = after[p * row_elems..(p + 1) * row_elems]
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.to_bits() == b.to_bits());
            assert!(
                same,
                "nkt={nkt} row {t} (cell {p}): the device store is not the CPU quantizer's bytes"
            );
            assert_ne!(
                expected.iter().fold(0u32, |m, x| m | x.to_bits()),
                0,
                "the fixture wrote all-zero packed bytes; the comparison would be vacuous"
            );
        }
        // (b) unwritten cells stay zero (the store writes only its rows).
        for cell in 0..N_CTX {
            if pos.contains(&cell) {
                continue;
            }
            assert!(
                after[cell * row_elems..(cell + 1) * row_elems]
                    .iter()
                    .all(|x| x.to_bits() == 0),
                "the store touched cell {cell}, which no position named"
            );
        }
    }
}
/// C4 S2b: the Q8_0 store → attention round trip, the packed twin of
/// [`Self::cuda_kv_f16_roundtrip_attn`]. The device writes real packed cells
/// through `Op::KvcacheStore`, then attention reads them back through the
/// layout-tagged split path — and the output must match the **f32** kernel run
/// over the same cells dequantized on the host: `kv4<Q8_0>` reconstructs
/// exactly `d * q[i]`, which is what `kvformat::unpack_q8_0_cells` puts in the
/// reference region. A wrong block index, scale offset or quant offset moves a
/// value by whole quant steps (the 0.117-magnitude outputs here), so the 1e-6
/// bound is many orders tighter than any layout fault while leaving room for
/// the one real difference between the two kernels: nvcc may contract the
/// packed accessor's `d * q` into the following FMA chain, and the f32 kernel
/// reads that product from memory, so the two can differ by one ulp. Measured:
/// exactly one ULP on one element (2.38e-7 of 0.117). The **byte-exact** claim
/// for the store lives in [`Self::cuda_q8_0_store_matches_the_cpu_quantizer`],
/// where it belongs.
#[test]
fn cuda_kv_q8_0_roundtrip_attn() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    let (nh, nk_h, hd) = (4usize, 2usize, 64usize);
    let nkt = nk_h * hd;
    let row_elems = crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt);
    let (nt, n_ctx) = (3usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = vec![1, 4, 9];

    let mut b = GraphBuilder::new();
    b.set_kv_format(crate::graph::kvformat::KvFormat::Q8_0);
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let qr = b.rope(
        q,
        pp,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: nh,
            hd,
        },
    );
    let at = b.attn(
        qr,
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
    let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
    let (kreg, vreg) = (
        cb.alloc_buffer(n_ctx * row_elems),
        cb.alloc_buffer(n_ctx * row_elems),
    );
    // The f32 reference regions the dequantized cells are written into, read by
    // the same kernel instantiated on KV_LAYOUT_F32.
    let (kf32, vf32) = (cb.alloc_buffer(n_ctx * nkt), cb.alloc_buffer(n_ctx * nkt));

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
    cb.write_host(kreg, &vec![0f32; n_ctx * row_elems]).unwrap();
    cb.write_host(vreg, &vec![0f32; n_ctx * row_elems]).unwrap();

    // Q8_0 arm: store through the kernel, attention over the packed cells.
    cb.set_kv_q8_for_test();
    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
        .unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kreg, xb_p],
        ob_at,
        Some((kreg, vreg)),
    )
    .unwrap();
    let got_q8 = cb.copy_to_host(ob_at).unwrap();

    // Reference arm: dequantize the *same packed bytes* the device wrote on the
    // host, put them in f32-shaped regions, and run the f32 kernel. The two arms
    // share every byte of K/V, so the only variable is the layout arithmetic.
    let packed_k = cb.copy_to_host(kreg).unwrap();
    let packed_v = cb.copy_to_host(vreg).unwrap();
    let mut ref_k = vec![0f32; n_ctx * nkt];
    let mut ref_v = vec![0f32; n_ctx * nkt];
    for &p in &pos {
        crate::graph::kvformat::unpack_q8_0_cells(
            &packed_k,
            nkt,
            p,
            1,
            &mut ref_k[p * nkt..(p + 1) * nkt],
        );
        crate::graph::kvformat::unpack_q8_0_cells(
            &packed_v,
            nkt,
            p,
            1,
            &mut ref_v[p * nkt..(p + 1) * nkt],
        );
    }
    assert!(
        ref_k.iter().any(|x| *x != 0.0),
        "the dequantized reference is all zero; the comparison would be vacuous"
    );
    cb.set_kv_layout_for_test(crate::cuda::KV_LAYOUT_F32);
    cb.write_host(kf32, &ref_k).unwrap();
    cb.write_host(vf32, &ref_v).unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kf32, xb_p],
        ob_at,
        Some((kf32, vf32)),
    )
    .unwrap();
    let want = cb.copy_to_host(ob_at).unwrap();

    let worst = got_q8
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let first = got_q8
        .iter()
        .zip(&want)
        .position(|(a, b)| a != b)
        .map(|i| (i, got_q8[i], want[i]));
    assert!(
        worst <= 1e-6,
        "the packed attention read is not the f32 kernel over the dequantized \
         cells: max |Δ| = {worst}, first {first:?}"
    );
    eprintln!(
        "[c4s2] packed vs dequantized f32 attention: max |Δ| = {worst} ({} of {} \
         elements differ)",
        got_q8.iter().zip(&want).filter(|(a, b)| a != b).count(),
        got_q8.len()
    );

    // …and the arm is real: a zero-output kernel would also compare equal to a
    // zero reference, so require non-trivial output.
    assert!(
        got_q8.iter().any(|x| x.abs() > 1e-6),
        "the packed attention returned all zeros"
    );

    // ── #186: the `__dp4a` decode arm ───────────────────────────────────────
    //
    // The int-dot path lives only in the `nt == 1` split-K kernel, which the
    // `nt = 3` arms above never reach. This arm runs it at `nt = 1` with an
    // explicit span `[0, 2)` over **two** stored cells, so the query's K scores
    // actually reach the output through the softmax: with a single key the score
    // cancels and the arm would be vacuous (the first version of this arm used
    // `positions = [0]`/one cell and a mutated block base still passed — the
    // mutation is what found it).
    //
    // `positions = [0]` keeps the rope the identity permutation, and the query is
    // built **exactly Q8_0-representable** (`amax = 1`, values in `{-1, 0, 1}`):
    // its block scale is then `1/127` and its quants are `±127`, so the int dot's
    // answer equals the f32 kernel's over the dequantized cells and the bound
    // stays tight enough to catch a wrong block base, scale offset or byte pack.
    // The query quantization itself is a numerics change with its own class — it
    // is measured by the real-model gate
    // `a_packed_kv_cache_answers_like_the_f32_one`, not pinned here.
    {
        const NCELL: usize = 2;
        let mut b1 = GraphBuilder::new();
        b1.set_kv_format(crate::graph::kvformat::KvFormat::Q8_0);
        b1.set_explicit_span(true);
        let q1 = b1.input("q", [nh * hd, 1, 1, 1], DType::F32);
        let k1 = b1.input("k", [nkt, NCELL, 1, 1], DType::F32);
        let v1 = b1.input("v", [nkt, NCELL, 1, 1], DType::F32);
        let p1 = b1.input("positions", [1, 1, 1, 1], DType::I32);
        let st1 = b1.kvcache_store(0, k1, v1, n_ctx);
        let ld1 = b1.kvcache_load(0, nkt, n_ctx, nk_h);
        let qr1 = b1.rope(
            q1,
            p1,
            RopeStyle::NonInterleaved,
            RoPEMeta {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_head: nh,
                hd,
            },
        );
        let at1 = b1.attn(
            qr1,
            ld1,
            p1,
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
        b1.output(at1);
        let g1 = b1.build();

        // The Attn node's 4th input is the builder's span node; one `[lo, hi)`
        // pair per query, here `[0, 2)` over both stored cells.
        let span1: Vec<f32> = [0u32, NCELL as u32]
            .iter()
            .map(|&x| f32::from_bits(x))
            .collect();

        let qv1: Vec<f32> = (0..nh * hd)
            .map(|i| match i % 32 {
                0 => 1.0,
                1 => -1.0,
                _ => 0.0,
            })
            .collect();
        assert!(
            qv1.chunks(32).all(|c| c.iter().any(|x| *x != 0.0)),
            "every query block must have a non-zero amax, or its scale is 0"
        );
        let s1_q = cb.alloc_buffer(nh * hd);
        let s1_k = cb.alloc_buffer(nkt * NCELL);
        let s1_v = cb.alloc_buffer(nkt * NCELL);
        let s1_p = cb.alloc_buffer(1);
        let s1_c = cb.alloc_buffer(NCELL);
        let s1_w = cb.alloc_buffer(2);
        let o1_q = cb.alloc_buffer(nh * hd);
        let o1_a = cb.alloc_buffer(nh * hd);
        let rk1 = cb.alloc_buffer(n_ctx * row_elems);
        let rv1 = cb.alloc_buffer(n_ctx * row_elems);
        let fk1 = cb.alloc_buffer(n_ctx * nkt);
        let fv1 = cb.alloc_buffer(n_ctx * nkt);
        let bits1 = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
        cb.write_host(s1_q, &qv1).unwrap();
        cb.write_host(s1_k, &ks[..nkt * NCELL]).unwrap();
        cb.write_host(s1_v, &vs[..nkt * NCELL]).unwrap();
        cb.write_host(s1_p, &bits1(&[0])).unwrap();
        cb.write_host(s1_c, &bits1(&[0, 1])).unwrap();
        cb.write_host(s1_w, &span1).unwrap();
        cb.write_host(rk1, &vec![0f32; n_ctx * row_elems]).unwrap();
        cb.write_host(rv1, &vec![0f32; n_ctx * row_elems]).unwrap();

        cb.set_kv_q8_for_test();
        cb.exec_ids(&g1.nodes[st1], &[s1_k, s1_v, s1_c], rk1, Some((rk1, rv1)))
            .unwrap();
        cb.exec_ids(&g1.nodes[qr1], &[s1_q, s1_p], o1_q, None)
            .unwrap();
        crate::testfail::reset_checked();
        cb.exec_ids(
            &g1.nodes[at1],
            &[o1_q, rk1, s1_p, s1_w],
            o1_a,
            Some((rk1, rv1)),
        )
        .unwrap();
        let got1 = cb.copy_to_host(o1_a).unwrap();
        // Rule 3's observation half: the counter is bumped by the launch
        // chokepoint, so it proves the arm the A/B selects actually ran — and
        // tracks the control (`MINFER_NO_DP4A_Q8_KV=1`) rather than lying about
        // it.
        assert_eq!(
            crate::testfail::checked("cuda_q8_kv_dp4a"),
            u64::from(crate::cuda::q8_kv_dp4a_enabled()),
            "the Q8_0 decode launch's dp4a observation counter must match its control"
        );

        let pk1 = cb.copy_to_host(rk1).unwrap();
        let pv1 = cb.copy_to_host(rv1).unwrap();
        let mut ref_k1 = vec![0f32; n_ctx * nkt];
        let mut ref_v1 = vec![0f32; n_ctx * nkt];
        crate::graph::kvformat::unpack_q8_0_cells(&pk1, nkt, 0, NCELL, &mut ref_k1[..nkt * NCELL]);
        crate::graph::kvformat::unpack_q8_0_cells(&pv1, nkt, 0, NCELL, &mut ref_v1[..nkt * NCELL]);
        cb.set_kv_layout_for_test(crate::cuda::KV_LAYOUT_F32);
        cb.write_host(fk1, &ref_k1).unwrap();
        cb.write_host(fv1, &ref_v1).unwrap();
        cb.exec_ids(
            &g1.nodes[at1],
            &[o1_q, fk1, s1_p, s1_w],
            o1_a,
            Some((fk1, fv1)),
        )
        .unwrap();
        let want1 = cb.copy_to_host(o1_a).unwrap();
        let worst1 = got1
            .iter()
            .zip(&want1)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            got1.iter().any(|x| x.abs() > 1e-6),
            "the dp4a decode arm returned all zeros"
        );
        assert!(
            worst1 <= 1e-5,
            "the dp4a decode K dot is not the f32 kernel over the dequantized \
             cells with an exactly-representable query: max |Δ| = {worst1}"
        );
        eprintln!(
            "[c4s2] dp4a decode (nt=1, span over {} cells) vs dequantized f32 \
             attention: max |Δ| = {worst1}, dp4a arm {}",
            NCELL,
            crate::cuda::q8_kv_dp4a_enabled()
        );
    }
}
/// #144 item 1: the **packed fused decode epilogue** must write the unfused
/// chain's bytes. The reference is the CPU quantizer (`quants::
/// quantize_row_q8_0`) run over the roped / bias-added values computed on the
/// host — an implementation the kernel does not share — so a wrong block
/// index, rope pairing, scale offset or quant offset moves a value by whole
/// quant steps.
///
/// Two arms:
/// - `pos = 0` (the rope is the identity permutation: `cs = 1`, `sn = 0`), where
///   K and V are compared **byte for byte** — pairing, block math and the
///   quantizer have no transcendental to hide behind;
/// - `pos = 7`, where the angle is real: q (never quantized) is compared as a
///   value, and K is compared after dequantization at one quant step's class,
///   because the device's `cosf`/`sinf` may differ from the host's in the last
///   ulp and that can flip a quant sitting on a boundary.
#[test]
fn cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer() {
    use crate::graph::kvformat::KvFormat;
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    let (nh, nk_h, hd) = (4usize, 2usize, 64usize);
    let (nqt, nkt) = (nh * hd, nk_h * hd);
    let row_words = KvFormat::Q8_0.row_elems(nkt);
    let row_bytes = KvFormat::Q8_0.row_bytes(nkt);
    let half = hd / 2;
    let (fb, fs) = (10000.0f32, 1.0f32);

    let vals = |seed: usize, n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| ((i * seed + 7) % 17) as f32 / 4.0 - 2.0)
            .collect()
    };
    let (qs, ks, vs) = (vals(37, nqt), vals(41, nkt), vals(57, nkt));
    let (bq, bk, bv) = (vals(13, nqt), vals(19, nkt), vals(23, nkt));

    let (b_q, b_k, b_v) = (
        cb.alloc_buffer(nqt),
        cb.alloc_buffer(nkt),
        cb.alloc_buffer(nkt),
    );
    let (b_bq, b_bk, b_bv) = (
        cb.alloc_buffer(nqt),
        cb.alloc_buffer(nkt),
        cb.alloc_buffer(nkt),
    );
    let b_p = cb.alloc_buffer(1);
    let b_c = cb.alloc_buffer(1);
    // Two packed rows so the two arms cannot alias.
    let (kreg, vreg) = (
        cb.alloc_buffer(2 * row_words),
        cb.alloc_buffer(2 * row_words),
    );
    cb.write_host(b_q, &qs).unwrap();
    cb.write_host(b_k, &ks).unwrap();
    cb.write_host(b_v, &vs).unwrap();
    cb.write_host(b_bq, &bq).unwrap();
    cb.write_host(b_bk, &bk).unwrap();
    cb.write_host(b_bv, &bv).unwrap();
    cb.write_host(kreg, &vec![0f32; 2 * row_words]).unwrap();
    cb.write_host(vreg, &vec![0f32; 2 * row_words]).unwrap();

    // The host's rope + bias, per head (neox pairing d <-> d + hd/2).
    let roped = |src: &[f32], bias: &[f32], heads: usize, pos: usize| -> Vec<f32> {
        let mut out = vec![0f32; heads * hd];
        for h in 0..heads {
            for d in 0..hd {
                let dd = if d < half { d } else { d - half };
                let ja = h * hd + dd;
                let jb = ja + half;
                let x0 = src[ja] + bias[ja];
                let x1 = src[jb] + bias[jb];
                let theta = pos as f32 * fs / fb.powf((2.0 * dd as f32) / hd as f32);
                let (cs, sn) = (theta.cos(), theta.sin());
                out[h * hd + d] = if d < half {
                    x0 * cs - x1 * sn
                } else {
                    x0 * sn + x1 * cs
                };
            }
        }
        out
    };
    // The packed payload the CPU quantizer produces for a flat value row.
    let pack = |v: &[f32]| -> Vec<u8> {
        let nblk = v.len() / 32;
        let mut raw = vec![0u8; nblk * 34];
        for b in 0..nblk {
            let q = crate::quants::quantize_row_q8_0(&v[b * 32..(b + 1) * 32]);
            raw[b * 34..(b + 1) * 34].copy_from_slice(&q);
        }
        raw
    };
    let cell_bytes = |region: &[f32], row: usize| -> Vec<u8> {
        let words = &region[row * row_words..(row + 1) * row_words];
        let mut out = Vec::with_capacity(words.len() * 4);
        for w in words {
            out.extend_from_slice(&w.to_le_bytes());
        }
        out
    };

    let run = |cb: &mut CudaBackend, pos: usize, row: usize| {
        // #188: the direct `attn_bias_rope_store*` calls below bypass
        // `execute_node`, so bind this backend's stream explicitly.
        let _bound = cb.bind();
        // q is roped IN PLACE, so each arm starts from the original input.
        cb.write_host(b_q, &qs).unwrap();
        cb.write_host(b_p, &[f32::from_bits(pos as u32)]).unwrap();
        cb.write_host(b_c, &[f32::from_bits(row as u32)]).unwrap();
        let (q, k, v) = (
            cb.ptr_of(b_q).unwrap(),
            cb.ptr_of(b_k).unwrap(),
            cb.ptr_of(b_v).unwrap(),
        );
        let (pq, pk, pv) = (
            cb.ptr_of(b_bq).unwrap(),
            cb.ptr_of(b_bk).unwrap(),
            cb.ptr_of(b_bv).unwrap(),
        );
        let pp = cb.ptr_of(b_p).unwrap();
        let cc = cb.ptr_of(b_c).unwrap();
        cb.state.attn_bias_rope_store_q8_0(
            q,
            k,
            v,
            pq,
            pk,
            pv,
            cb.ptr_of(kreg).unwrap(),
            cb.ptr_of(vreg).unwrap(),
            nqt,
            nkt,
            hd,
            fb,
            fs,
            pp,
            cc,
            row_bytes,
        );
        cb.synchronize();
        (
            cb.copy_to_host(b_q).unwrap(),
            cb.copy_to_host(kreg).unwrap(),
            cb.copy_to_host(vreg).unwrap(),
        )
    };

    // ── arm 1: pos = 0 — K and V byte for byte ──────────────────────────
    let (q0, k0, v0) = run(&mut cb, 0, 0);
    let want_k = pack(&roped(&ks, &bk, nk_h, 0));
    let want_v = pack(&(0..nkt).map(|i| vs[i] + bv[i]).collect::<Vec<f32>>());
    assert_eq!(
        &cell_bytes(&k0, 0)[..want_k.len()],
        &want_k[..],
        "the packed K cell is not the CPU quantizer's bytes at pos = 0"
    );
    assert_eq!(
        &cell_bytes(&v0, 0)[..want_v.len()],
        &want_v[..],
        "the packed V cell is not the CPU quantizer's bytes at pos = 0"
    );
    let want_q0 = roped(&qs, &bq, nh, 0);
    let qdelta0 = q0
        .iter()
        .zip(&want_q0)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        qdelta0 <= 1e-6,
        "the roped q buffer is not the host's identity-rope result: max |Δ| = {qdelta0}"
    );

    // ── arm 2: pos = 7 — the angle is real ─────────────────────────────
    let (q7, k7, v7) = run(&mut cb, 7, 1);
    let want_q7 = roped(&qs, &bq, nh, 7);
    let qdelta = q7
        .iter()
        .zip(&want_q7)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        qdelta <= 1e-5,
        "the roped q buffer diverges from the host's rope: max |Δ| = {qdelta}"
    );
    assert_eq!(
        &cell_bytes(&v7, 1)[..want_v.len()],
        &want_v[..],
        "V does not depend on the rope angle, so its bytes must still match"
    );
    let mut got_k7 = vec![0f32; nkt];
    crate::graph::kvformat::unpack_q8_0_cells(&k7, nkt, 1, 1, &mut got_k7);
    let kdelta = got_k7
        .iter()
        .zip(&roped(&ks, &bk, nk_h, 7))
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        kdelta <= 0.03,
        "the packed K cell is not the host's roped values at one quant step's class: \
         max |Δ| = {kdelta}"
    );
    assert!(
        got_k7.iter().any(|x| x.abs() > 1e-3),
        "the packed K cell is all zeros; the comparison would be vacuous"
    );
}
#[test]
fn cuda_rope_kv_attn_roundtrip() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
    let nkt = nk_h * hd;
    let (nt, n_ctx) = (3usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = vec![1, 4, 9]; // sparse, exercises the scatter

    let mut b = GraphBuilder::new();
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let p = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let qr = b.rope(
        q,
        p,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: nh,
            hd,
        },
    );
    let at = b.attn(
        qr,
        load,
        p,
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
    let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
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
    let pb: Vec<f32> = pos.iter().map(|&pp| f32::from_bits(pp as u32)).collect();
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    // Zero the KV regions first: rows the store never touches stay
    // uninitialized in a recycled cudaMalloc block, and the reference
    // below treats unwritten rows as zeros (deterministic vs pool state).
    cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
    cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
        .unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kreg, xb_p],
        ob_at,
        Some((kreg, vreg)),
    )
    .unwrap();

    // a) stored K rows are bit-exact at the scattered positions
    let kback = cb.copy_to_host(kreg).unwrap();
    for (t, &pp) in pos.iter().enumerate() {
        assert_eq!(
            &kback[pp * nkt..(pp + 1) * nkt],
            &ks[t * nkt..(t + 1) * nkt],
            "K row {pp}"
        );
    }
    // b) RoPE vs cpu_rope (also covers the non-alias D2D staging path)
    let qgot = cb.copy_to_host(ob_qr).unwrap();
    let mut qref = qs.clone();
    crate::graph::cpu_backend::cpu_rope(
        &mut qref,
        &pos,
        nh,
        hd,
        10000.0,
        1.0,
        RopeStyle::NonInterleaved,
    );
    assert_close("rope", &qgot, &qref, 1e-4);
    // c) GQA attention vs cpu_gqa_attn over the scattered KV regions
    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for (t, &pp) in pos.iter().enumerate() {
        kfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&ks[t * nkt..(t + 1) * nkt]);
        vfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&vs[t * nkt..(t + 1) * nkt]);
    }
    // E1: the CPU reference takes the allowed-cell span; this is the
    // single-sequence causal window the test compares against.
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qref, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    assert_close("gqa_attn", &agot, &aref, 1e-4);
}
// 8b: f16 KV cache — the store rounds K/V to half and the attention
// kernel reads half4. The reference builds its KV from the SAME
// half-rounded values so the comparison isolates the kernel from the
// f16 quantization noise (tolerance stays tight).
/// C3's copy primitive on the device: the rows land where the plan says,
/// *including when source and destination overlap* — the case a bulk
/// device-to-device copy cannot express (CUDA documents overlapping
/// `cudaMemcpyAsync` as undefined).
#[test]
fn cuda_copy_cells_moves_overlapping_rows_in_both_directions() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    // 6 rows x 4 elements; a row's value identifies the row it came from.
    let id = cb.alloc_buffer(24);
    let data: Vec<f32> = (0..24)
        .map(|i| (i / 4) as f32 + (i % 4) as f32 / 10.0)
        .collect();
    cb.write_host(id, &data).unwrap();
    let r = BufRef::own(crate::graph::Backend::CUDA, id, 24);
    // Rows [1, 4) -> rows [0, 3): rows 1 and 2 are both read and overwritten.
    cb.copy_cells(r, r, 0, 1, 3, 4).unwrap();
    let got = cb.copy_to_host(id).unwrap();
    let want: Vec<f32> = (0..24)
        .map(|i| {
            let row = if i / 4 < 3 { i / 4 + 1 } else { i / 4 };
            row as f32 + (i % 4) as f32 / 10.0
        })
        .collect();
    assert_eq!(got, want, "the moved rows must be byte-identical");
    // C7b: the upward direction works too. The kernel walks the rows in the
    // order the overlap requires — descending here — so the result is what a
    // copy through a temporary would give.
    cb.write_host(id, &data).unwrap();
    cb.copy_cells(r, r, 1, 0, 3, 4).unwrap();
    let got = cb.copy_to_host(id).unwrap();
    let want: Vec<f32> = (0..24)
        .map(|i| {
            let row = match i / 4 {
                0 => 0,
                n if n <= 3 => n - 1,
                n => n,
            };
            row as f32 + (i % 4) as f32 / 10.0
        })
        .collect();
    assert_eq!(
        got, want,
        "an upward overlapping move must be byte-identical"
    );
}
#[test]
fn cuda_kv_f16_roundtrip_attn() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    cb.set_kv_f16_for_test(true);
    let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
    let nkt = nk_h * hd;
    let (nt, n_ctx) = (3usize, 32usize);
    let scale = 1.0 / (hd as f32).sqrt();
    let pos: Vec<usize> = vec![1, 4, 9];

    let mut b = GraphBuilder::new();
    let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
    let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
    let pp = b.input("positions", [nt, 1, 1, 1], DType::I32);
    let store = b.kvcache_store(0, k, v, n_ctx);
    let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
    let qr = b.rope(
        q,
        pp,
        RopeStyle::NonInterleaved,
        RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: nh,
            hd,
        },
    );
    let at = b.attn(
        qr,
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
    let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
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
    // the reference KV: what the f16 store actually persists (f32→f16→f32)
    let to_half =
        |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect() };
    let ks_h = to_half(&ks);
    let vs_h = to_half(&vs);
    cb.write_host(xb_q, &qs).unwrap();
    cb.write_host(xb_k, &ks).unwrap();
    cb.write_host(xb_v, &vs).unwrap();
    cb.write_host(xb_p, &pb).unwrap();
    // zero the regions (unwritten rows read as f16 zeros)
    cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
    cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

    cb.exec_ids(
        &g.nodes[store],
        &[xb_k, xb_v, xb_p],
        kreg,
        Some((kreg, vreg)),
    )
    .unwrap();
    cb.exec_ids(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
        .unwrap();
    cb.exec_ids(
        &g.nodes[at],
        &[ob_qr, kreg, xb_p],
        ob_at,
        Some((kreg, vreg)),
    )
    .unwrap();

    // a) stored K rows equal the half-rounded values at the scatter positions
    let kback_f32 = cb.copy_to_host(kreg).unwrap();
    // reinterpret the region as f16 pairs (store wrote 2 bytes/elem)
    let kbytes: Vec<u8> = kback_f32.iter().flat_map(|f| f.to_le_bytes()).collect();
    for (t, &p) in pos.iter().enumerate() {
        for j in 0..nkt {
            let byte_off = (p * nkt + j) * 2;
            let got = half::f16::from_le_bytes([kbytes[byte_off], kbytes[byte_off + 1]]);
            assert!(
                (got.to_f32() - ks_h[t * nkt + j]).abs() < 1e-6,
                "f16 K row {p}[{j}]"
            );
        }
    }
    // b) attention vs cpu_gqa_attn over the half-rounded KV
    let qgot = cb.copy_to_host(ob_qr).unwrap();
    let mut qref = qs.clone();
    crate::graph::cpu_backend::cpu_rope(
        &mut qref,
        &pos,
        nh,
        hd,
        10000.0,
        1.0,
        RopeStyle::NonInterleaved,
    );
    assert_close("rope(f16 kv)", &qgot, &qref, 1e-4);
    let mut kfull = vec![0f32; nkt * n_ctx];
    let mut vfull = vec![0f32; nkt * n_ctx];
    for (t, &p) in pos.iter().enumerate() {
        kfull[p * nkt..(p + 1) * nkt].copy_from_slice(&ks_h[t * nkt..(t + 1) * nkt]);
        vfull[p * nkt..(p + 1) * nkt].copy_from_slice(&vs_h[t * nkt..(t + 1) * nkt]);
    }
    // E1: the CPU reference takes the allowed-cell span; this is the
    // single-sequence causal window the test compares against.
    let span = crate::graph::cpu_backend::causal_span(&pos);
    let mut aref = vec![0f32; nh * hd * nt];
    crate::graph::cpu_backend::cpu_gqa_attn(
        &qref, &kfull, &vfull, &span, nt, nh, nk_h, hd, hd, nkt, &mut aref, scale,
    )
    .unwrap();
    let agot = cb.copy_to_host(ob_at).unwrap();
    assert_close("gqa_attn(f16 kv)", &agot, &aref, 1e-4);
}
