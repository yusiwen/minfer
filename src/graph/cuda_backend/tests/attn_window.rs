//! The causal/windowed and `kv_map` attention windows, including the S4 A/B statistic.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

/// Median of `v` (does not reorder the caller's slice).
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}
/// Matched pairs per phase of the S4 map-window A/B, fixed in advance
/// (issue #189).
const PAIRS: usize = 9;
/// The S4 A/B's sign-test threshold: the gate refuses when at least this many
/// of the `PAIRS` matched pairs put the map arm above the bar. For 9 pairs the
/// one-sided binomial tail under the null (a fair coin, i.e. map == bar *
/// span) is `P(X >= 7) = 46/512 = 0.090`. A median of the same 9 pairs flips
/// once 5 are disturbed — a minority cannot decide the sign test.
const SIGN_TEST_REFUSALS: usize = 7;
/// The map-window A/B's verdict statistic (issue #189), shared by both its
/// phases.
///
/// `span` and `map` hold µs/launch for the **same** round index, which is why
/// the callers interleave the rounds: every ratio is a matched pair measured
/// next to each other on one machine state. A pair is a *refusal* when the map
/// arm spent more than `bar` times its matched span arm. The verdict is the
/// refusal **count**, and the gate refuses only at [`SIGN_TEST_REFUSALS`] —
/// the one-sided sign-test threshold for [`PAIRS`] pairs at `alpha = 0.090`.
/// A median flips once half the pairs are disturbed; the sign test needs a
/// two-thirds supermajority, so a load spike that moves a minority (or even a
/// bare majority) of pairs cannot decide it. Returns
/// `(refusals, span_median, map_median, sorted_ratios)` so the gate prints
/// every sample and not just the verdict.
fn sign_test_ratio(span: &[f64], map: &[f64], bar: f64) -> (usize, f64, f64, Vec<f64>) {
    assert_eq!(span.len(), map.len(), "one ratio per matched pair");
    assert_eq!(span.len(), PAIRS, "the pair count is fixed in advance");
    let mut ratios: Vec<f64> = span.iter().zip(map).map(|(s, m)| m / s).collect();
    let refusals = ratios.iter().filter(|r| **r > bar).count();
    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (refusals, median(span), median(map), ratios)
}
/// The statistic itself, on the recorded failing distribution (issue #189).
///
/// The pre-#189 gate asserted the **median** of these nine ratios and went
/// red at 1.398x on a loaded parallel device run (GB10 sm_121, 2026-09-26).
/// The paired sign test must (a) pass that same distribution — 5 refusals is
/// below its threshold of 7 — while (b) refusing a real regression, where the
/// map arm's work doubles and every matched pair moves. Pure, so it runs in
/// the always-run CUDA unit suite with no device (rule 1 of the gate contract:
/// the gate must assert the value, not just a relation between two paths).
#[test]
fn the_s4_ab_statistic_absorbs_a_loaded_run_and_still_refuses_a_real_regression() {
    // The recorded prefill ratios, verbatim.
    let recorded = [
        0.632f64, 0.697, 0.989, 1.091, 1.398, 1.440, 1.466, 2.198, 6.695,
    ];
    let span = vec![100.0f64; PAIRS];
    let map: Vec<f64> = recorded.iter().map(|r| r * 100.0).collect();
    let (refusals, _, _, ratios) = sign_test_ratio(&span, &map, 1.25);
    assert_eq!(
        refusals, 5,
        "the recorded run has 5 of 9 pairs above 1.25x (ratios {ratios:?})"
    );
    assert!(
        median(&ratios) > 1.25,
        "the old median statistic must be red on the recorded distribution — otherwise \
         the recorded failure could not have happened (median {})",
        median(&ratios)
    );
    assert!(
        refusals < SIGN_TEST_REFUSALS,
        "the sign test must absorb the recorded loaded run ({refusals} < \
         {SIGN_TEST_REFUSALS})"
    );

    // A doubled map cost moves every pair, and the sign test refuses all nine.
    let doubled: Vec<f64> = span.iter().map(|s| s * 2.0).collect();
    let (d_refusals, ..) = sign_test_ratio(&span, &doubled, 1.25);
    assert_eq!(
        d_refusals, PAIRS,
        "a doubled map cost must be refused on every matched pair"
    );
}
/// E1b's recorded residual gap, closed: the **causal** and the **windowed**
/// instantiation must compute the same numbers over the same rows.
///
/// The equivalence rests on row arithmetic being the only difference between
/// the two kernels (the SASS comparison showed identical instruction counts
/// and opcode histograms). This is the direct check: one query per token, a
/// window that starts at cell 0 (so `positions[t] + 1` and the explicit span
/// describe the *same* rows), the same K/V, executed through both
/// instantiations — the outputs must be bitwise equal.
///
/// Without this, "the windowed path is correct" rested on the SASS identity
/// plus a windowed-only test with `lo = 0`; the E1b record named that as the
/// gap to close on the first device session.
#[test]
fn cuda_causal_and_windowed_agree_on_the_same_rows() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_f16_for_test(false); // f32 KV keeps the store and the attention in one dtype

    // Two tokens at cells 0 and 1: token 0's window is [0, 1), token 1's is
    // [0, 2) — exactly what `positions[t] + 1` gives, so both instantiations
    // must agree. hd = 4 (CUDA requires a nonzero multiple of 4).
    //
    // NOTE (2026-09-19, corrected): this fixture is **degenerate** — one-hot
    // queries and values make the output insensitive to the scores, so it is
    // weak evidence on its own and was never the windowed path's gate. The
    // randomized `cuda_windowed_attention_matches_causal_for_long_windows` is
    // that gate, and it is green (see its note: it first ran red because of a
    // harness bug — one advancing LCG shared by both calls — not the kernel).
    let (nh, nk, hd, nt, n_ctx) = (1usize, 1usize, 4usize, 2usize, 4usize);
    let nkt = nk * hd;
    let meta = crate::graph::ops::AttnMeta {
        layer: 0,
        n_head: nh,
        n_head_kv: nk,
        hd,
        hd_kv: hd,
        nkt,
        scale: 1.0,
    };
    let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let q = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]; // token 0 = e0, token 1 = e1
    let k = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
    let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let positions = [0u32, 1u32];

    // Build one graph per instantiation; both store into and read the same
    // region, so they see identical K/V.
    let build = |explicit: bool| -> crate::graph::ComputeGraph {
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(explicit);
        let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let qq = gb.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let kk = gb.input("k", [nkt, nt, 1, 1], DType::F32);
        let vv = gb.input("v", [nkt, nt, 1, 1], DType::F32);
        let _st = gb.kvcache_store(0, kk, vv, n_ctx);
        let kv = gb.kvcache_load(0, nkt, n_ctx, nk);
        let at = gb.attn(qq, kv, pos, crate::graph::ops::AttnMode::Gqa, meta.clone());
        gb.output(at);
        gb.build()
    };
    let causal = build(false);
    let windowed = build(true);
    assert!(
        !causal.nodes.iter().any(|n| matches!(
            n.op,
            crate::graph::ops::Op::Attn {
                explicit_span: true,
                ..
            }
        )),
        "the causal graph must not declare an explicit span"
    );
    assert!(
        windowed.nodes.iter().any(|n| matches!(
            n.op,
            crate::graph::ops::Op::Attn {
                explicit_span: true,
                ..
            }
        )),
        "the windowed graph must declare an explicit span"
    );

    let mut run = |g: &crate::graph::ComputeGraph| -> Vec<f32> {
        let kreg = cb.alloc_buffer(n_ctx * nkt);
        let vreg = cb.alloc_buffer(n_ctx * nkt);
        let kb = cb.alloc_buffer(nkt * nt);
        let vb = cb.alloc_buffer(nkt * nt);
        let qb = cb.alloc_buffer(nh * hd * nt);
        let pb = cb.alloc_buffer(nt);
        let sb = cb.alloc_buffer(2 * nt);
        let ob = cb.alloc_buffer(nh * hd * nt);
        cb.write_host(qb, &q).unwrap();
        cb.write_host(kb, &k).unwrap();
        cb.write_host(vb, &v).unwrap();
        cb.write_host(pb, &i32bits(&positions)).unwrap();
        // token 0: [0, 1); token 1: [0, 2) — the same rows `positions` names.
        cb.write_host(sb, &i32bits(&[0, 0, 1, 2])).unwrap();
        let sti = g
            .nodes
            .iter()
            .position(|n| matches!(n.op, crate::graph::ops::Op::KvcacheStore { .. }))
            .expect("store node");
        let ati = g
            .nodes
            .iter()
            .position(|n| matches!(n.op, crate::graph::ops::Op::Attn { .. }))
            .expect("attn node");
        cb.exec_ids(&g.nodes[sti], &[kb, vb, pb], kreg, Some((kreg, vreg)))
            .unwrap();
        cb.exec_ids(&g.nodes[ati], &[qb, kreg, pb, sb], ob, Some((kreg, vreg)))
            .unwrap();
        cb.copy_to_host(ob).unwrap()
    };
    let a = run(&causal);
    let b = run(&windowed);
    assert_eq!(a.len(), b.len());
    let worst = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert_eq!(
        worst, 0.0,
        "the causal and windowed instantiations disagree over the same rows: \
         causal {a:?} vs windowed {b:?}"
    );
}
/// The 2026-09-19 blocker, isolated: the **windowed** attention
/// instantiation must agree with the causal one over the *same relative
/// rows*, and it does not once the window is longer than a handful of rows.
///
/// Why this shape: the server puts every slot but the first at a non-zero
/// KV offset, so every slot but the first uses the windowed instantiation.
/// Because the kernels' row arithmetic is the only intended difference, the
/// assertion is **bitwise** equality against the causal run over rows `0..n`.
///
/// Note (2026-09-19): this test first ran RED, and the failure was the
/// **harness**, not the kernel — the two `run` calls shared one advancing LCG,
/// so "causal" and "windowed" were compared over *different* q/k/v (the
/// causal-vs-`V(row 0)` check still passed, because a single-key softmax is
/// that identity whatever the data). With the data now seeded per shape inside
/// `run`, every case is bitwise equal; plan §14's "narrowed to the windowed
/// instantiation" entry is retracted on that evidence.
#[test]
fn cuda_windowed_attention_matches_causal_for_long_windows() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    cb.set_kv_f16_for_test(false);

    // Real-ish shapes: the 7B decodes with hd 128 / 4 KV heads, which is a
    // larger `hd` and `nkv` than the 0.5B's 64 / 2 — the other reason this
    // was model-dependent.
    let n_ctx = 512usize;
    let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };

    // Deterministic pseudo-random data (an LCG, so a failure is reproducible).
    let mut seed = 0x1234_5678u32;
    let mut next = || {
        seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
        ((seed >> 8) as f32 / 8_388_608.0) - 1.0
    };

    let mut run = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                   explicit: bool,
                   start: usize,
                   n: usize,
                   shape: (usize, usize, usize)|
     -> (Vec<f32>, Vec<f32>) {
        let (nh, nk, hd) = shape;
        let nkt = nk * hd;
        // The two calls this test compares (causal / windowed) MUST see the
        // same q/k/v: the outer LCG advances on every call, so using it here
        // compared different data and reported a divergence that was an
        // artefact of the harness. Seed a local LCG from the shape only —
        // never from `start`/`explicit`, which are what the calls differ in.
        let mut lseed = 0x9e37_79b9u32
            ^ (n as u32).wrapping_mul(0x85eb_ca6b)
            ^ (nh as u32).wrapping_mul(0xc2b2_ae35)
            ^ (hd as u32).wrapping_mul(0x27d4_eb2f);
        let mut next = || {
            lseed = lseed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((lseed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let meta = crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: hd,
            nkt,
            scale: 1.0,
        };
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(explicit);
        let pos = gb.input("positions", [n, 1, 1, 1], DType::I32);
        let qq = gb.input("q", [nh * hd, n, 1, 1], DType::F32);
        let kk = gb.input("k", [nkt, n, 1, 1], DType::F32);
        let vv = gb.input("v", [nkt, n, 1, 1], DType::F32);
        let _st = gb.kvcache_store(0, kk, vv, n_ctx);
        let kv = gb.kvcache_load(0, nkt, n_ctx, nk);
        let at = gb.attn(qq, kv, pos, crate::graph::ops::AttnMode::Gqa, meta);
        gb.output(at);
        let g = gb.build();

        let posv: Vec<usize> = (start..start + n).collect();
        let qv: Vec<f32> = (0..nh * hd * n).map(|_| next()).collect();
        let kvv: Vec<f32> = (0..nkt * n).map(|_| next()).collect();
        let vvv: Vec<f32> = (0..nkt * n).map(|_| next()).collect();
        let span: Vec<u32> = (0..n)
            .flat_map(|t| [(start as u32), (start + t + 1) as u32])
            .collect();
        // The span array is laid out as all `lo`s then all `hi`s.
        let mut span_u32 = vec![0u32; 2 * n];
        for t in 0..n {
            span_u32[t] = start as u32;
            span_u32[n + t] = (start + t + 1) as u32;
        }
        let _ = span;

        let kreg = cb.alloc_buffer(n_ctx * nkt);
        let vreg = cb.alloc_buffer(n_ctx * nkt);
        let kb = cb.alloc_buffer(nkt * n);
        let vb = cb.alloc_buffer(nkt * n);
        let qb = cb.alloc_buffer(nh * hd * n);
        let pb = cb.alloc_buffer(n);
        let sb = cb.alloc_buffer(2 * n);
        let ob = cb.alloc_buffer(nh * hd * n);
        cb.write_host(qb, &qv).unwrap();
        cb.write_host(kb, &kvv).unwrap();
        cb.write_host(vb, &vvv).unwrap();
        cb.write_host(
            pb,
            &i32bits(&posv.iter().map(|&p| p as u32).collect::<Vec<u32>>()),
        )
        .unwrap();
        cb.write_host(sb, &i32bits(&span_u32)).unwrap();
        // Harness self-check: the buffers the launch will read must hold
        // exactly what this test wrote. If they do, the inputs are not the
        // explanation for the divergence and the launch/kernel is.
        let pos_back = cb.copy_to_host(pb).unwrap();
        let want_pos: Vec<f32> = i32bits(&posv.iter().map(|&p| p as u32).collect::<Vec<u32>>());
        assert_eq!(pos_back, want_pos, "positions buffer read back wrong");
        let span_back = cb.copy_to_host(sb).unwrap();
        assert_eq!(span_back, i32bits(&span_u32), "span buffer read back wrong");
        let sti = g
            .nodes
            .iter()
            .position(|nd| matches!(nd.op, crate::graph::ops::Op::KvcacheStore { .. }))
            .unwrap();
        let ati = g
            .nodes
            .iter()
            .position(|nd| matches!(nd.op, crate::graph::ops::Op::Attn { .. }))
            .unwrap();
        cb.exec_ids(&g.nodes[sti], &[kb, vb, pb], kreg, Some((kreg, vreg)))
            .unwrap();
        cb.exec_ids(&g.nodes[ati], &[qb, kreg, pb, sb], ob, Some((kreg, vreg)))
            .unwrap();
        let out = cb.copy_to_host(ob).unwrap();
        // Token t's V block is contiguous (`[t*nkt, (t+1)*nkt)`), so this is
        // the exact expected output of query 0, whose window is a single row.
        (out, vvv[..hd].to_vec())
    };

    // (window length, non-zero start): the lengths sweep the kernel variants
    // the 7B selects; the starts are the server's kind of offset.
    let mut bad: Vec<String> = Vec::new();
    // Both KV dtypes: the production default is the engine's resolved format
    // (`kvformat::auto_device_format`: f16 when `n_layers * n_kv_embd >= 8192`) —
    // so the 7B runs f16 KV while the 0.5B runs f32, and a gate that forces f32
    // cannot see a windowed-f16 fault at all.
    for f16 in [false, true] {
        for (shape, n, start) in [
            ((1usize, 1usize, 4usize), 1usize, 0usize),
            ((1, 1, 4), 2, 0),
            ((1, 1, 4), 2, 64),
            ((1, 1, 128), 1, 0),
            ((1, 1, 128), 2, 0),
            ((2, 2, 64), 1, 0),
            ((4, 4, 128), 1, 0),
            ((4, 4, 128), 2, 0),
            ((4, 4, 128), 2, 64),
            ((4, 4, 128), 16, 64),
            ((4, 4, 128), 34, 64),
        ] {
            cb.set_kv_f16_for_test(f16);
            let (causal, v_row0) = run(&mut cb, false, 0, n, shape);
            let (windowed, _) = run(&mut cb, true, start, n, shape);
            let hd = shape.2;
            let ref_delta = |x: &[f32]| {
                x[..hd]
                    .iter()
                    .zip(&v_row0)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max)
            };
            // The f16 KV rounds V, so the exactness check is f32-only; the
            // causal-vs-windowed comparison below is unaffected (bitwise for
            // both dtypes, since both runs see the same stored K/V).
            assert!(
                f16 || ref_delta(&causal) == 0.0,
                "the CAUSAL instantiation does not return V(row 0) for a single-row window (n={n})"
            );
            let worst = causal
                .iter()
                .zip(&windowed)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            if worst == 0.0 {
                eprintln!("[window] kv_f16={f16} {shape:?} n={n} start={start}: bitwise equal");
            } else {
                let first = causal
                    .iter()
                    .zip(&windowed)
                    .position(|(a, b)| a != b)
                    .map(|k| (k, causal[k], windowed[k]));
                eprintln!(
                    "[window] kv_f16={f16} {shape:?} n={n} start={start}: DIVERGES max|d|={worst} first={first:?}"
                );
                bad.push(format!(
                    "kv_f16={f16} {shape:?} n={n} start={start} max|d|={worst} first={first:?}"
                ));
            }
        }
    }
    assert!(
        bad.is_empty(),
        "the windowed instantiation diverges from the causal one over the same relative rows \
         in {} case(s):\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
}
/// C8b S4: a `kv_map` window must give each query exactly the rows the
/// equivalent `attn_span` gives it — the same cells, named through a run list
/// instead of one range. **Bitwise**, both KV dtypes: the two modes differ in
/// how they resolve a row, not in what they compute over it.
///
/// The shapes sweep the kernel families a map can land in: `nt == 1` (split-K
/// flash decoding — a sharing slot's decode step), `1 < nt <= 16` (the batched
/// split path) and `nt > 16` (prefill: FA when hd = 128, the legacy per-(token,
/// head) kernel otherwise). Both a prefill-shaped batch and a **decode-shaped**
/// one are run: a single token at the end of the window, whose window is several
/// runs. A first draft of this test only ever gave one token at position 0,
/// which touches the first run alone — the multi-run decode case is exactly what
/// the real-model gate caught it missing.
#[test]
fn cuda_map_window_matches_the_span_over_the_same_rows() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    let _guard = crate::cuda::CudaState::model_load_guard();
    const N_CTX: usize = 512;
    const KMAX: usize = crate::graph::kvcache::KV_MAP_MAX_SPANS;

    // Reference rows [0, n) of both regions, seeded from (n, shape) alone so
    // every call compared here sees the same bytes.
    let reference = |n: usize, shape: (usize, usize, usize)| -> (Vec<f32>, Vec<f32>) {
        let (nh, nk, hd) = shape;
        let nkt = nk * hd;
        let mut seed = 0x51ed_2701u32
            ^ (n as u32).wrapping_mul(0x9e37_79b9)
            ^ (nh as u32).wrapping_mul(0x85eb_ca6b)
            ^ (nkt as u32).wrapping_mul(0xc2b2_ae35);
        let mut next = || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let k: Vec<f32> = (0..n * nkt).map(|_| next()).collect();
        let v: Vec<f32> = (0..n * nkt).map(|_| next()).collect();
        (k, v)
    };

    // One attention call. `runs` are `(cell, len, base)`: `len` cells from
    // `cell` hold the reference rows starting at `base`, which is how a
    // non-contiguous window is compared against the span over the same bytes.
    let mut run = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                   map: bool,
                   pos: &[u32],
                   window: &[u32],
                   shape: (usize, usize, usize),
                   runs: &[(usize, usize, usize)],
                   reference: &(Vec<f32>, Vec<f32>),
                   layout: i32|
     -> Vec<f32> {
        let nt = pos.len();
        let (nh, nk, hd) = shape;
        let nkt = nk * hd;
        // C4 S2b: the packed cell's word width. Every region here is built
        // with the *layout's* own size — `N_CTX * row_elems` — which is also
        // the discipline issue #122 asks for (an under-sized buffer plus a
        // device read past it latches `cudaErrorIllegalAddress` and poisons
        // every later allocation in the process).
        let row_elems = if layout == crate::cuda::KV_LAYOUT_Q8_0 {
            crate::graph::kvformat::KvFormat::Q8_0.row_elems(nkt)
        } else {
            nkt
        };
        let meta = crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: hd,
            nkt,
            scale: 1.0,
        };
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(true);
        gb.set_kv_map(map);
        let p = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let qq = gb.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let _kk = gb.input("k", [nkt, nt, 1, 1], crate::graph::DType::F32);
        let _vv = gb.input("v", [nkt, nt, 1, 1], crate::graph::DType::F32);
        let kv = gb.kvcache_load(0, nkt, N_CTX, nk);
        let at = gb.attn(qq, kv, p, crate::graph::ops::AttnMode::Gqa, meta);
        gb.output(at);
        let g = gb.build();

        // The KV region as the runs describe it. With an f16 cache the kernel
        // reads the *low half* of each 4-byte slot, so a value is written as the
        // half's bit pattern there: clean f16 data rather than the garbage an
        // arbitrary f32 write leaves behind — which would make a wrong-row
        // comparison compare zero to zero. With Q8_0 the row is packed through
        // the same quantizer the store kernel uses (`pack_q8_0_cell`), so the
        // kernel reads exactly the bytes a real store would have written.
        let (rk, rv) = reference;
        let mut region_k = vec![0.0f32; N_CTX * row_elems];
        let mut region_v = vec![0.0f32; N_CTX * row_elems];
        let enc = |x: f32| -> f32 {
            if layout == crate::cuda::KV_LAYOUT_F16 {
                f32::from_bits(half::f16::from_f32(x).to_bits() as u32)
            } else {
                x
            }
        };
        for &(cell, len, base) in runs {
            for i in 0..len {
                let src = (base + i) * nkt;
                let dst = (cell + i) * row_elems;
                if layout == crate::cuda::KV_LAYOUT_Q8_0 {
                    crate::graph::kvformat::pack_q8_0_cell(
                        &mut region_k[dst..dst + row_elems],
                        nkt,
                        &rk[src..src + nkt],
                    );
                    crate::graph::kvformat::pack_q8_0_cell(
                        &mut region_v[dst..dst + row_elems],
                        nkt,
                        &rv[src..src + nkt],
                    );
                } else {
                    for e in 0..nkt {
                        region_k[dst + e] = enc(rk[src + e]);
                        region_v[dst + e] = enc(rv[src + e]);
                    }
                }
            }
        }
        // q is its own data (the comparison is over the same q in both modes).
        let mut seed = 0x1234_abcdu32
            ^ (nt as u32).wrapping_mul(0x27d4_eb2f)
            ^ (nh as u32).wrapping_mul(0x9e37_79b9);
        let qv: Vec<f32> = (0..nh * hd * nt)
            .map(|_| {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 8) as f32 / 8_388_608.0) - 1.0
            })
            .collect();

        let kreg = cb.alloc_buffer(N_CTX * row_elems);
        let vreg = cb.alloc_buffer(N_CTX * row_elems);
        let qb = cb.alloc_buffer(nh * hd * nt);
        let pb = cb.alloc_buffer(nt);
        let wb = cb.alloc_buffer(window.len());
        let ob = cb.alloc_buffer(nh * hd * nt);
        let i32bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
        cb.write_host(qb, &qv).unwrap();
        cb.write_host(kreg, &region_k).unwrap();
        cb.write_host(vreg, &region_v).unwrap();
        cb.write_host(pb, &i32bits(pos)).unwrap();
        cb.write_host(wb, &i32bits(window)).unwrap();
        let ati = g
            .nodes
            .iter()
            .position(|nd| matches!(nd.op, crate::graph::ops::Op::Attn { .. }))
            .unwrap();
        cb.exec_ids(&g.nodes[ati], &[qb, kreg, pb, wb], ob, Some((kreg, vreg)))
            .unwrap();
        cb.copy_to_host(ob).unwrap()
    };

    // A span window for a batch at `pos`: one `[lo, hi)` pair per query.
    let span_batch = |pos: &[u32]| -> Vec<u32> {
        let mut w = vec![0u32; 2 * pos.len()];
        for (i, &p) in pos.iter().enumerate() {
            w[i] = 64;
            w[pos.len() + i] = 64 + p + 1;
        }
        w
    };
    // ...and the runs `attn_map` would emit for it (in position order, each
    // query's own run clipped to its row).
    let map_batch = |runs: &[(usize, usize, usize)], pos: &[u32]| -> Vec<u32> {
        let mut w = Vec::with_capacity(pos.len() * KMAX * 2);
        for &p in pos {
            let mut left = p as usize + 1;
            let mut kk = 0usize;
            for &(cell, len, _) in runs {
                let take = len.min(left);
                if take == 0 {
                    continue;
                }
                assert!(kk < KMAX, "the fixture needs more than {KMAX} runs");
                w.push(cell as u32);
                w.push(take as u32);
                left -= take;
                kk += 1;
            }
            assert_eq!(left, 0, "the runs must cover query {p}'s window");
            while kk < KMAX {
                w.push(0);
                w.push(0);
                kk += 1;
            }
        }
        w
    };

    let mut bad: Vec<String> = Vec::new();
    let mut checked = 0usize;
    // C4 S2b: all three layouts. Q8_0 rows go through exactly the same
    // modes — the run list is resolved before the load, so the packed accessor
    // cannot change which rows a window names.
    for layout in [
        crate::cuda::KV_LAYOUT_F32,
        crate::cuda::KV_LAYOUT_F16,
        crate::cuda::KV_LAYOUT_Q8_0,
    ] {
        cb.set_kv_layout_for_test(layout);
        let layout_name = match layout {
            crate::cuda::KV_LAYOUT_Q8_0 => "q8_0",
            crate::cuda::KV_LAYOUT_F16 => "f16",
            _ => "f32",
        };
        for (shape, n) in [
            ((1usize, 1usize, 4usize), 1usize),
            ((2, 2, 64), 1),
            ((4, 4, 128), 1),
            ((2, 2, 64), 6),
            ((4, 4, 128), 8),  // 1 < nt <= 16: the batched split path (f32/f16)
            ((4, 4, 128), 34), // nt > 16: FA prefill (hd = 128) for f16, general otherwise
            ((2, 2, 64), 34),  // nt > 16 with f32 KV: the legacy kernel
        ] {
            // A Q8_0 cell is a whole number of 32-element blocks, so `nkt` must
            // be a multiple of 32 — the invariant `ensure_kv`'s `check_width`
            // enforces (`nkt` is `n_head_kv * hd` and every supported arch has
            // `hd % 32 == 0`). The `hd = 4` fixture is a kernel-shape probe that
            // a packed format cannot express at all; it is skipped rather than
            // silently run with a zero-word cell.
            if layout == crate::cuda::KV_LAYOUT_Q8_0 && (shape.1 * shape.2) % 32 != 0 {
                continue;
            }
            let reference = reference(n, shape);
            let mut variants: Vec<(&str, Vec<(usize, usize, usize)>)> = vec![
                ("one run", vec![(64, n, 0)]),
                (
                    "two runs",
                    vec![(64, 4.min(n), 0), (300, n - 4.min(n), 4.min(n))],
                ),
            ];
            if n >= 6 {
                // Three runs, so the walk is exercised past its second entry.
                let (l0, l1) = (n / 3, n / 3);
                variants.push((
                    "three runs",
                    vec![(64, l0, 0), (200, l1, l0), (300, n - l0 - l1, l0 + l1)],
                ));
            }
            for (label, runs) in variants {
                if runs.iter().map(|r| r.1).sum::<usize>() != n {
                    continue; // a window this short cannot be split that far
                }
                // (a) a prefill-shaped batch (one query per window row) and
                // (b) a decode-shaped one (a single token at the window's end,
                //     which is the multi-run case a sharing slot decodes).
                let prefill: Vec<u32> = (0..n as u32).collect();
                let decode = [n as u32 - 1];
                for (which, pos) in [("prefill", &prefill[..]), ("decode", &decode[..])] {
                    let span = run(
                        &mut cb,
                        false,
                        pos,
                        &span_batch(pos),
                        shape,
                        &[(64, n, 0)],
                        &reference,
                        layout,
                    );
                    let map = run(
                        &mut cb,
                        true,
                        pos,
                        &map_batch(&runs, pos),
                        shape,
                        &runs,
                        &reference,
                        layout,
                    );
                    checked += 1;
                    let worst = span
                        .iter()
                        .zip(map.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    if worst != 0.0 {
                        bad.push(format!(
                            "kv={layout_name} {shape:?} n={n} {label} {which}: max|d|={worst}"
                        ));
                    }
                }
            }
        }
    }
    assert!(checked >= 30, "the fixture checked only {checked} cases");
    assert!(
        bad.is_empty(),
        "the map window disagrees with the span over the same bytes in {} case(s):\n  {}",
        bad.len(),
        bad.join("\n  ")
    );
    // A window that attended to *no* row would compare equal trivially (both
    // sides zero), so the gate has to see real output too.
    cb.set_kv_f16_for_test(false);
    let reference8 = reference(8, (2, 2, 64));
    let out = run(
        &mut cb,
        false,
        &(0..8u32).collect::<Vec<u32>>(),
        &span_batch(&(0..8u32).collect::<Vec<u32>>()),
        (2, 2, 64),
        &[(64, 8, 0)],
        &reference8,
        crate::cuda::KV_LAYOUT_F32,
    );
    assert!(
        out.iter().any(|&x| x != 0.0 && x.is_finite()),
        "the fixture attends to no row — the comparison above would be vacuous"
    );
    // Every comparison above is **between modes over the same bytes**, so a
    // value-level fault in the layout accessor shifts both sides identically and
    // the bitwise equality still holds. Mutation-checked by hand: deleting the
    // block base from `kv4<KV_LAYOUT_Q8_0>` (reading block 0 for every group)
    // left this test green while `cuda_kv_q8_0_roundtrip_attn` failed — the F6
    // lesson, a gate passing for the wrong reason. So one single-row window is
    // compared against an **independently computed** reference: the dequantized
    // V row the fixture packed. A single-row window's softmax is 1, so the output
    // is exactly `d * q[i]` of the cell — the same expression `kv4<Q8_0>`
    // evaluates — and this arm now fails on a wrong block, scale or quant offset.
    {
        use crate::graph::kvformat::{pack_q8_0_cell, unpack_q8_0_cells, KvFormat};
        let shape = (2usize, 2usize, 64usize);
        let nkt = shape.1 * shape.2;
        let row_elems = KvFormat::Q8_0.row_elems(nkt);
        let reference1 = reference(1, shape);
        let mut cell_v = vec![0f32; row_elems];
        pack_q8_0_cell(&mut cell_v, nkt, &reference1.1);
        let mut want_v = vec![0f32; nkt];
        unpack_q8_0_cells(&cell_v, nkt, 0, 1, &mut want_v);
        cb.set_kv_layout_for_test(crate::cuda::KV_LAYOUT_Q8_0);
        let out = run(
            &mut cb,
            false,
            &[0u32],
            &span_batch(&[0u32]),
            shape,
            &[(64, 1, 0)],
            &reference1,
            crate::cuda::KV_LAYOUT_Q8_0,
        );
        let worst = out[..shape.2]
            .iter()
            .zip(&want_v[..shape.2])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            out[..shape.2].iter().any(|x| x.abs() > 1e-6),
            "the single-row Q8_0 fixture returned all zeros — vacuous"
        );
        assert_eq!(
            worst, 0.0,
            "the Q8_0 single-row window does not return the dequantized V cell row: \
             max |Δ| = {worst}"
        );
    }
    eprintln!("[map] {checked} cases bitwise-equal to the span over the same rows");
}
/// C8b S4 device A/B — what naming a row through the run list costs over the
/// span's `row0 + i`, at the decode shape that pays it on every step.
///
/// **The value arm is the primary signal (gate contract rule 1).** The gate
/// used to be a pure stopwatch: it timed two arms and never looked at what
/// they computed, so two equally-wrong kernels would pass and a loaded box
/// could decide the verdict. Before any timing it now asserts that
/// - a **one-row** map window at a non-zero cell returns exactly that row's
///   V — an absolute value computed on the host, not a relation between the
///   two modes (softmax over one key is exactly 1.0, so the kernel is the
///   identity on V);
/// - a **two-run** map window with a non-zero base cell returns the span's
///   bytes over the same rows, bit for bit — one run is indistinguishable to
///   a resolver that reads `(cell, len)` as `(lo, hi)`, two runs are not;
/// - the map instantiation actually ran (the observation half of rule 3):
///   `testfail::note_checked("cuda_attn_map_window")` moves on a map call and
///   not on a span call.
///
/// **The timing arm is a paired sign test, not a median (issue #189).** Both
/// phases (decode and prefill) interleave matched rounds of the two modes and
/// count how many pairs put map above `1.25x` span. The gate refuses only when
/// [`SIGN_TEST_REFUSALS`] = 7 of [`PAIRS`] = 9 pairs do — the one-sided sign
/// test at `alpha = 46/512 = 0.090`. A median of the same 9 pairs flips at 5,
/// and the recorded parallel device run (GB10 sm_121, 2026-09-26) measured
/// exactly 5 disturbed pairs (`[0.632, 0.697, 0.989, 1.091, 1.398, 1.440,
/// 1.466, 2.198, 6.695]`, median 1.398x) and failed a kernel that was not
/// slower; the sign test passes that run and still fails a doubled map cost
/// (`MINFER_S4_AB_MAP_REPS=2`, the reproducible form of #123's map-work
/// doubling), which moves every pair.
///
/// The bar is unchanged at 1.25x and the timing fixture is the pre-#189 one
/// (one run at cell 0), so the recorded margins stay comparable. Run with
/// `--ignored --nocapture` to see every sample and the refusal count.
#[test]
#[ignore = "timing: needs a CUDA device"]
fn cuda_map_window_costs_no_more_than_the_span_it_replaces() {
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    // Issue #188: this gate drives `CudaBackend`/`CudaState` **directly** — no
    // `scheduler::execute`. It no longer takes the #185 device-entry guard:
    // the direct `attn_bias_rope_store` calls below bind `cb`'s own stream
    // (see `run`), so they no longer land on the context stream that another
    // test could be capturing. The process-wide `model_load_guard` stays —
    // this gate is a full model-load-shaped device workload.
    let _guard = crate::cuda::CudaState::model_load_guard();
    const KMAX: usize = crate::graph::kvcache::KV_MAP_MAX_SPANS;
    // The 7B decode shape (hd 128, 4 KV heads) at a 2K window: long enough
    // that the split-K body does real work, short enough that the 4-warp
    // hybrid gate stays out of it.
    let (nh, nk, hd, nkv) = (28usize, 4usize, 128usize, 2048usize);
    let nkt = nk * hd;
    let n_ctx = 4096usize;
    let bits = |v: &[u32]| -> Vec<f32> { v.iter().map(|&x| f32::from_bits(x)).collect() };
    let scale = 1.0 / (hd as f32).sqrt();
    let (reps, warmup) = (100usize, 3usize);
    let (preps, pwarm) = (50usize, 3usize);
    // The prefill fixture's `nt`. `q` is read as `nt` token rows of
    // `nh * hd` floats (`fa_prefill_f16kv`: `q[t * nh * hd + h * hd + d]`,
    // `t < nt`), so it needs its own `nt`-row q buffer. Reusing the decode
    // phase's single-row `qb` here made the kernel read ~7 MB past it: a
    // silent read of whatever device memory followed when those pages were
    // mapped, and a latched `cudaErrorIllegalAddress` (700) when they were
    // not — which then failed every later `cudaMemGetInfo` in the process.
    let nt = 512usize;
    let qb = cb.alloc_buffer(nh * hd);
    let kreg = cb.alloc_buffer(n_ctx * nkt);
    let vreg = cb.alloc_buffer(n_ctx * nkt);
    let ob = cb.alloc_buffer(nh * hd);
    let spb = cb.alloc_buffer(2);
    let mpb = cb.alloc_buffer(KMAX * 2);
    let spb2 = cb.alloc_buffer(2 * nt);
    let mpb2 = cb.alloc_buffer(nt * KMAX * 2);
    let qb2 = cb.alloc_buffer(nh * hd * nt);
    let ob2 = cb.alloc_buffer(nh * hd * nt);
    cb.write_host(qb, &vec![0.01f32; nh * hd]).unwrap();
    let enc = |x: f32| -> f32 { f32::from_bits(half::f16::from_f32(x).to_bits() as u32) };

    // ── the value arms (gate contract rule 1) ────────────────────────────
    //
    // The fixture is harder than the timed one on purpose: a non-zero base
    // cell (so a resolver that ignores a run's `cell` cannot alias row 0) and
    // two ascending runs (so a resolver that reads the map as one `(lo, hi)`
    // pair cannot cover the window). Rows [VBASE, VBASE + nkv) carry distinct
    // K/V, so resolving the wrong cell changes the output instead of aliasing
    // the same constant.
    const VBASE: usize = 512;
    let distinct = |seed: u32, r: usize| -> Vec<f32> {
        (0..nkt)
            .map(|i| {
                (((i as u32)
                    .wrapping_mul(seed)
                    .wrapping_add((r as u32).wrapping_mul(2_654_435_761)))
                    % 101) as f32
                    / 101.0
                    - 0.5
            })
            .collect()
    };
    let mut kfill = vec![0.02f32; n_ctx * nkt];
    let mut vfill = vec![0.03f32; n_ctx * nkt];
    for r in VBASE..VBASE + nkv {
        kfill[r * nkt..(r + 1) * nkt].copy_from_slice(&distinct(31, r));
        vfill[r * nkt..(r + 1) * nkt].copy_from_slice(&distinct(57, r));
    }
    cb.set_kv_f16_for_test(false);
    cb.write_host(kreg, &kfill).unwrap();
    cb.write_host(vreg, &vfill).unwrap();

    // One decode-window call, returning the output buffer.
    let run = |cb: &CudaBackend, mode: crate::cuda::AttnWindow, win: usize| -> Vec<f32> {
        cb.state.gqa_attn_split(
            cb.ptr_of(qb).unwrap(),
            cb.ptr_of(kreg).unwrap(),
            cb.ptr_of(vreg).unwrap(),
            cb.ptr_of(ob).unwrap(),
            cb.ptr_of(win).unwrap(),
            mode.code(),
            nh,
            nk,
            hd,
            scale,
            crate::cuda::KV_LAYOUT_F32,
            nkt * 4,
        );
        cb.copy_to_host(ob).unwrap()
    };

    // (a) the absolute arm: a window of one row returns that row's V exactly.
    cb.write_host(spb, &bits(&[VBASE as u32, 1])).unwrap();
    let mut one_map = vec![0u32; KMAX * 2];
    one_map[0] = VBASE as u32;
    one_map[1] = 1;
    cb.write_host(mpb, &bits(&one_map)).unwrap();
    let got = run(&cb, crate::cuda::AttnWindow::Map, mpb);
    let gqa = nh / nk;
    let want = &vfill[VBASE * nkt..(VBASE + 1) * nkt];
    for h in 0..nh {
        let kw = (h / gqa) * hd;
        for d in 0..hd {
            assert_eq!(
                got[h * hd + d].to_bits(),
                want[kw + d].to_bits(),
                "a one-row map window at cell {VBASE} did not return that row's V \
                 (head {h}, dim {d})"
            );
        }
    }
    assert!(
        got.iter().any(|x| x.abs() > 1e-3),
        "the one-row map window returned all zeros; the comparison is vacuous"
    );

    // (b) the relation arm: a two-run window with a non-zero base equals the
    // span over the same rows, bit for bit.
    const VW: usize = 64;
    const VSPLIT: usize = 17;
    cb.write_host(spb, &bits(&[VBASE as u32, (VBASE + VW) as u32]))
        .unwrap();
    let mut map_win = vec![0u32; KMAX * 2];
    map_win[0] = VBASE as u32;
    map_win[1] = VSPLIT as u32;
    map_win[2] = (VBASE + VSPLIT) as u32;
    map_win[3] = (VW - VSPLIT) as u32;
    cb.write_host(mpb, &bits(&map_win)).unwrap();
    let o_span = run(&cb, crate::cuda::AttnWindow::Span, spb);
    let o_map = run(&cb, crate::cuda::AttnWindow::Map, mpb);
    let first = o_span
        .iter()
        .zip(&o_map)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, o_span[i], o_map[i]));
    assert!(
        first.is_none(),
        "a two-run map window over cells [{VBASE}, {}) diverges from the span over the \
         same rows: first {first:?}",
        VBASE + VW
    );
    assert!(
        o_map.iter().any(|x| x.abs() > 1e-3),
        "the two-run map window returned all zeros; the comparison is vacuous"
    );

    // (c) the counted observation (rule 3's observation half): the map
    // instantiation ran, and a span call does not touch the map counter.
    crate::testfail::reset_checked();
    let _ = run(&cb, crate::cuda::AttnWindow::Span, spb);
    assert_eq!(
        crate::testfail::checked("cuda_attn_map_window"),
        0,
        "a span call must not bump the map-window chokepoint"
    );
    let _ = run(&cb, crate::cuda::AttnWindow::Map, mpb);
    assert_eq!(
        crate::testfail::checked("cuda_attn_map_window"),
        1,
        "the map-mode attention launch never reached its chokepoint; the arm under test \
         did not run"
    );

    // (d) the same relation on the *prefill* path the parallel run failed on:
    // an f16 KV region with distinct rows and one two-run map window per query.
    cb.set_kv_f16_for_test(true);
    let mut kfill16 = vec![enc(0.02f32); n_ctx * nkt];
    let mut vfill16 = vec![enc(0.03f32); n_ctx * nkt];
    for r in VBASE..VBASE + nt {
        for (e, x) in distinct(31, r).iter().enumerate() {
            kfill16[r * nkt + e] = enc(*x);
        }
        for (e, x) in distinct(57, r).iter().enumerate() {
            vfill16[r * nkt + e] = enc(*x);
        }
    }
    cb.write_host(kreg, &kfill16).unwrap();
    cb.write_host(vreg, &vfill16).unwrap();
    cb.write_host(qb2, &vec![enc(0.01f32); nh * hd * nt])
        .unwrap();
    let mut span3 = vec![0u32; 2 * nt];
    let mut map3 = vec![0u32; nt * KMAX * 2];
    for t in 0..nt {
        let n = t + 1;
        let a = n / 2;
        span3[t] = VBASE as u32;
        span3[nt + t] = (VBASE + n) as u32;
        let at = t * KMAX * 2;
        map3[at] = VBASE as u32;
        map3[at + 1] = a as u32;
        map3[at + 2] = (VBASE + a) as u32;
        map3[at + 3] = (n - a) as u32;
    }
    cb.write_host(spb2, &bits(&span3)).unwrap();
    cb.write_host(mpb2, &bits(&map3)).unwrap();
    let prefill_run = |cb: &CudaBackend, mode: crate::cuda::AttnWindow, win: usize| -> Vec<f32> {
        cb.state.gqa_attn_kv_prefill(
            cb.ptr_of(qb2).unwrap(),
            cb.ptr_of(kreg).unwrap(),
            cb.ptr_of(vreg).unwrap(),
            cb.ptr_of(ob2).unwrap(),
            cb.ptr_of(win).unwrap(),
            mode.code(),
            crate::cuda::KV_LAYOUT_F16,
            nh,
            nk,
            hd,
            scale,
            nkt * 2,
            nt,
        );
        cb.copy_to_host(ob2).unwrap()
    };
    let p_span_v = prefill_run(&cb, crate::cuda::AttnWindow::Span, spb2);
    let p_map_v = prefill_run(&cb, crate::cuda::AttnWindow::Map, mpb2);
    let p_first = p_span_v
        .iter()
        .zip(&p_map_v)
        .position(|(a, b)| a.to_bits() != b.to_bits())
        .map(|i| (i, p_span_v[i], p_map_v[i]));
    assert!(
        p_first.is_none(),
        "a two-run map prefill over cells [{VBASE}, {}) diverges from the span over the \
         same rows: first {p_first:?}",
        VBASE + nt
    );
    assert!(
        p_map_v.iter().any(|x| x.abs() > 1e-3),
        "the two-run map prefill returned all zeros; the comparison is vacuous"
    );
    crate::testfail::reset_checked();
    let _ = prefill_run(&cb, crate::cuda::AttnWindow::Map, mpb2);
    assert_eq!(
        crate::testfail::checked("cuda_attn_map_window"),
        1,
        "the map-mode prefill launch never reached its chokepoint"
    );

    // ── the timing fixture (the pre-#189 one: constant K/V, one run at cell
    // 0), so the recorded margins stay comparable ─────────────────────────
    cb.set_kv_f16_for_test(false);
    cb.write_host(kreg, &vec![0.02f32; n_ctx * nkt]).unwrap();
    cb.write_host(vreg, &vec![0.03f32; n_ctx * nkt]).unwrap();
    cb.write_host(spb, &bits(&[0, nkv as u32])).unwrap();
    let mut map = vec![0u32; KMAX * 2];
    map[0] = 0;
    map[1] = nkv as u32;
    cb.write_host(mpb, &bits(&map)).unwrap();
    // µs/launch for one timed round of `reps` launches, after `warmup`
    // untimed ones. A first launch pays the module load, which is not the
    // measurement (the prewarm list covers the production instantiations, not
    // necessarily these).
    let time = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                mode: crate::cuda::AttnWindow,
                reps: usize,
                warmup: usize|
     -> f64 {
        let win = if mode == crate::cuda::AttnWindow::Map {
            mpb
        } else {
            spb
        };
        let mut call = |cb: &mut crate::graph::cuda_backend::CudaBackend| {
            cb.state.gqa_attn_split(
                cb.ptr_of(qb).unwrap(),
                cb.ptr_of(kreg).unwrap(),
                cb.ptr_of(vreg).unwrap(),
                cb.ptr_of(ob).unwrap(),
                cb.ptr_of(win).unwrap(),
                mode.code(),
                nh,
                nk,
                hd,
                scale,
                crate::cuda::KV_LAYOUT_F32,
                nkt * 4,
            );
        };
        for _ in 0..warmup {
            call(cb);
        }
        cb.state.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            call(cb);
        }
        cb.state.sync();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };

    // The timing statistic (issue #189). The old form asserted the **median of
    // the per-round ratios**, which a loaded harness flips once `rounds / 2`
    // pairs are disturbed: the recorded parallel device run (GB10 sm_121,
    // 2026-09-26) measured 5 of 9 pairs above 1.25x (median 1.398x) and failed
    // a kernel that was not slower. The verdict is now a **paired sign test**:
    // the count of pairs above the bar, fixed in advance at `PAIRS = 9`, and
    // the gate refuses only at `SIGN_TEST_REFUSALS = 7` — the one-sided
    // binomial tail `P(X >= 7 | fair coin) = 46/512 = 0.090`. A minority of
    // disturbed rounds cannot decide it; the recorded run's 5 does not, and
    // doubling the map work (`MINFER_S4_AB_MAP_REPS=2`) moves all 9 and does.
    //
    // The bar is **not** widened: it is the pre-#123 gate's 1.25x, and the
    // margin stays justified by measurement. On an idle GB10 the median ratio
    // is 1.001-1.004 (decode) and 1.087-1.107 (prefill) over 6 runs; with 16
    // CPU spinners plus two concurrent CUDA attention loops it stays
    // 1.001-1.018 and 1.079-1.145, although individual rounds reach 1.4-8.6x.
    const MAX_MAP_OVER_SPAN: f64 = 1.25;
    let mut span_us: Vec<f64> = Vec::with_capacity(PAIRS);
    let mut map_us: Vec<f64> = Vec::with_capacity(PAIRS);
    for _ in 0..PAIRS {
        span_us.push(time(&mut cb, crate::cuda::AttnWindow::Span, reps, warmup));
        map_us.push(time(&mut cb, crate::cuda::AttnWindow::Map, reps, warmup));
    }
    let (d_refusals, d_span, d_map, d_ratios) =
        sign_test_ratio(&span_us, &map_us, MAX_MAP_OVER_SPAN);
    eprintln!(
        "[s4-ab] decode nkv={nkv} nh={nh} nk={nk} hd={hd}: span {d_span:.1} / map {d_map:.1} \
         us/launch ({PAIRS} interleaved matched pairs of {reps}); per-round ratios \
         {d_ratios:?} — {d_refusals}/{PAIRS} above {MAX_MAP_OVER_SPAN}x (sign test refuses at \
         {SIGN_TEST_REFUSALS})"
    );
    assert!(
        d_refusals < SIGN_TEST_REFUSALS,
        "the map window is above {MAX_MAP_OVER_SPAN}x the span in {d_refusals} of {PAIRS} \
         matched pairs (medians {d_map:.1} vs {d_span:.1} us/launch; per-round ratios \
         {d_ratios:?}); the sign test refuses at {SIGN_TEST_REFUSALS}, so this is a systematic \
         map cost, not a load spike"
    );

    // The other half of the A/B: a *prefill* window (each query's whole
    // prefix), where the window is walked tile by tile. Both modes run FA here
    // since S4 taught its staging loop to resolve runs, so this measures the
    // resolution cost on the prefill path too. The buffers were allocated with
    // the value arms above, so this only rewrites the timing fixture.
    cb.set_kv_f16_for_test(true);
    cb.write_host(kreg, &vec![enc(0.02f32); n_ctx * nkt])
        .unwrap();
    cb.write_host(vreg, &vec![enc(0.03f32); n_ctx * nkt])
        .unwrap();
    cb.write_host(qb2, &vec![enc(0.01f32); nh * hd * nt])
        .unwrap();
    let mut span2 = vec![0u32; 2 * nt];
    let mut map2 = vec![0u32; nt * KMAX * 2];
    for t in 0..nt {
        span2[t] = 0;
        span2[nt + t] = (t + 1) as u32;
        map2[t * KMAX * 2] = 0;
        map2[t * KMAX * 2 + 1] = (t + 1) as u32;
    }
    cb.write_host(spb2, &bits(&span2)).unwrap();
    cb.write_host(mpb2, &bits(&map2)).unwrap();
    let time_prefill = |cb: &mut crate::graph::cuda_backend::CudaBackend,
                        mode: crate::cuda::AttnWindow,
                        reps: usize,
                        warmup: usize|
     -> f64 {
        let win = if mode == crate::cuda::AttnWindow::Map {
            mpb2
        } else {
            spb2
        };
        let mut call = |cb: &mut crate::graph::cuda_backend::CudaBackend| {
            cb.state.gqa_attn_kv_prefill(
                cb.ptr_of(qb2).unwrap(),
                cb.ptr_of(kreg).unwrap(),
                cb.ptr_of(vreg).unwrap(),
                cb.ptr_of(ob2).unwrap(),
                cb.ptr_of(win).unwrap(),
                mode.code(),
                crate::cuda::KV_LAYOUT_F16,
                nh,
                nk,
                hd,
                scale,
                nkt * 2,
                nt,
            );
        };
        for _ in 0..warmup {
            call(cb);
        }
        cb.state.sync();
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            call(cb);
        }
        cb.state.sync();
        t0.elapsed().as_secs_f64() * 1e6 / reps as f64
    };
    // The same paired sign test as the decode half — this is the assertion
    // that failed on the loaded parallel GB10, and the old form was weaker
    // here than there (a single 20-launch block per mode, no interleaving and
    // no round-to-round statistic at all). 9 matched pairs × 50 launches at
    // ~85 µs/launch is ~40 ms per mode.
    if hd == 128 {
        let mut pspan_us: Vec<f64> = Vec::with_capacity(PAIRS);
        let mut pmap_us: Vec<f64> = Vec::with_capacity(PAIRS);
        for _ in 0..PAIRS {
            pspan_us.push(time_prefill(
                &mut cb,
                crate::cuda::AttnWindow::Span,
                preps,
                pwarm,
            ));
            pmap_us.push(time_prefill(
                &mut cb,
                crate::cuda::AttnWindow::Map,
                preps,
                pwarm,
            ));
        }
        let (p_refusals, p_span, p_map, p_ratios) =
            sign_test_ratio(&pspan_us, &pmap_us, MAX_MAP_OVER_SPAN);
        eprintln!(
            "[s4-ab] prefill nt={nt} nkv={nt} hd={hd}: span {p_span:.1} / map {p_map:.1} \
             us/launch ({PAIRS} interleaved matched pairs of {preps}); per-round ratios \
             {p_ratios:?} — {p_refusals}/{PAIRS} above {MAX_MAP_OVER_SPAN}x (sign test refuses \
             at {SIGN_TEST_REFUSALS})"
        );
        assert!(
            p_refusals < SIGN_TEST_REFUSALS,
            "the map prefill is above {MAX_MAP_OVER_SPAN}x the span in {p_refusals} of {PAIRS} \
             matched pairs (medians {p_map:.1} vs {p_span:.1} us/launch; per-round ratios \
             {p_ratios:?}); the sign test refuses at {SIGN_TEST_REFUSALS}"
        );
    }
}
/// reference with the standard q8-activation tolerance instead.
#[test]
fn cuda_verify_attention_nt_invariance() {
    // doc 94: the verify batch's attention must be bitwise-equal to the
    // nt=1 decode path at every position — the greedy identity at the
    // kernel level. Same KV buffer (prefix + the 3 intra-batch rows),
    // same queries; batched (positions [P, P+1, P+2]) vs three decode
    // calls; compare outputs bitwise. Both KV dtype variants, the small
    // parity fixture AND the 14B decode dims (hd=128 -> the decode
    // path's dual-kernel gate is live; prefix 512 keeps rpw < 16 so the
    // incumbent 1-warp body owns both paths).
    let Some(mut cb) = pool() else {
        eprintln!("skipping: no CUDA device");
        return;
    };
    // Issue #188: direct `cb.state.*` kernel calls must land on THIS backend's
    // stream, so the following `cb.copy_to_host` / `cb.state.sync()` waits on
    // them (a context-stream launch + an instance-stream sync is a race).
    let _bound = cb.bind();
    let _guard = crate::cuda::CudaState::model_load_guard();

    let nts = [3usize, 8usize];
    let shapes: [(usize, usize, usize, usize, f32); 2] = [
        (4, 2, 8, 100, 0.3),                    // parity fixture dims
        (40, 8, 128, 512, 0.08838834764831845), // 14B decode dims
    ];
    for (nh, nk, hd, prefix, scale) in shapes {
        for nt in nts {
            let rows = prefix + nt;

            // deterministic inputs
            let mut s: u64 = 0x9E3779B97F4A7C15;
            let mut next = move || {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                s
            };
            let q: Vec<f32> = (0..nt * nh * hd)
                .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
                .collect();
            // f16 K/V bit patterns packed two-per-f32
            let kv_f16: Vec<u16> = (0..rows * nk * hd)
                .map(|_| half::f16::from_f32(((next() % 2001) as f32 - 1000.0) / 1000.0).to_bits())
                .collect();
            let kv_f32: Vec<f32> = (0..rows * nk * hd)
                .map(|_| ((next() % 2001) as f32 - 1000.0) / 1000.0)
                .collect();

            let pack_u16 = |v: &[u16]| -> Vec<f32> {
                v.chunks(2)
                    .map(|c| {
                        f32::from_bits(
                            (*c.first().unwrap()) as u32 | ((*c.get(1).unwrap_or(&0)) as u32) << 16,
                        )
                    })
                    .collect()
            };

            for f16_kv in [true, false] {
                // f32 KV needs rows*nk*hd f32 slots; f16 needs half — allocate
                // the max and write the right byte count per variant
                let kb = cb.alloc_buffer(rows * nk * hd);
                let vb = cb.alloc_buffer(rows * nk * hd);
                if f16_kv {
                    cb.write_host(kb, &pack_u16(&kv_f16)).unwrap();
                    cb.write_host(vb, &pack_u16(&kv_f16)).unwrap();
                } else {
                    cb.write_host(kb, &kv_f32).unwrap();
                    cb.write_host(vb, &kv_f32).unwrap();
                }
                let qb = cb.alloc_buffer(nt * nh * hd);
                cb.write_host(qb, &q).unwrap();
                let obt = cb.alloc_buffer(nt * nh * hd);
                let oseq = cb.alloc_buffer(nt * nh * hd);
                let post = cb.alloc_buffer(nt);
                let posv: Vec<f32> = (0..nt)
                    .map(|t| f32::from_bits((prefix as i32 + t as i32) as u32))
                    .collect();
                cb.write_host(post, &posv).unwrap();

                // batched verify: one call, positions [P, P+1, P+2]
                cb.state.gqa_attn_split_batched(
                    cb.ptr_of(qb).unwrap(),
                    cb.ptr_of(kb).unwrap(),
                    cb.ptr_of(vb).unwrap(),
                    cb.ptr_of(obt).unwrap(),
                    cb.ptr_of(post).unwrap(),
                    crate::cuda::AttnWindow::Causal.code(), // single-sequence
                    nh,
                    nk,
                    hd,
                    scale,
                    f16_kv,
                    nt,
                );

                // sequential decode: three nt=1 calls at the same positions
                let qrow = cb.alloc_buffer(nh * hd);
                let o1 = cb.alloc_buffer(nh * hd);
                let p1 = cb.alloc_buffer(1);
                for t in 0..nt {
                    cb.write_host(qrow, &q[t * nh * hd..(t + 1) * nh * hd])
                        .unwrap();
                    cb.write_host(p1, &[f32::from_bits((prefix as i32 + t as i32) as u32)])
                        .unwrap();
                    cb.state.gqa_attn_split(
                        cb.ptr_of(qrow).unwrap(),
                        cb.ptr_of(kb).unwrap(),
                        cb.ptr_of(vb).unwrap(),
                        cb.ptr_of(o1).unwrap(),
                        cb.ptr_of(p1).unwrap(),
                        crate::cuda::AttnWindow::Causal.code(), // single-sequence
                        nh,
                        nk,
                        hd,
                        scale,
                        if f16_kv {
                            crate::cuda::KV_LAYOUT_F16
                        } else {
                            crate::cuda::KV_LAYOUT_F32
                        },
                        if f16_kv { nk * hd * 2 } else { nk * hd * 4 },
                    );
                    let got = cb.copy_to_host(o1).unwrap();
                    let mut full = cb.copy_to_host(oseq).unwrap();
                    full[t * nh * hd..(t + 1) * nh * hd].copy_from_slice(&got);
                    cb.write_host(oseq, &full).unwrap();
                }

                let bt = cb.copy_to_host(obt).unwrap();
                let sq = cb.copy_to_host(oseq).unwrap();
                for (i, (a, b)) in bt.iter().zip(sq.iter()).enumerate() {
                    assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "nh={nh} nk={nk} hd={hd} prefix={prefix} kv_f16={f16_kv} elem {i}: batched {a} vs sequential {b}"
                );
                }
            }
        }
    } // nt sweep
}
