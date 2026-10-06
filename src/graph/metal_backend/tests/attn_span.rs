//! E1 `attn_span` windowed-attention gates for the Metal backend (issue #44
//! part (a), G5a).
//!
//! Before this round Metal refused every `Op::Attn { explicit_span: true }`
//! node (`SUPPORTS_ATTN_SPAN = false`), so a multi-sequence batch and a
//! non-zero-start run could not run on the device at all. These gates drive the
//! **new** `kernel_gqa_attn_window_f32/_f16` through the production graph path
//! and compare it against the CPU reference.
//!
//! `metal_attn_span_matches_cpu` is the row-resolution proof: each query is a
//! one-cell window at a distinct cell, so a single-key softmax is exactly 1 and
//! the output is exactly that cell's V row. That makes the comparison **bitwise**
//! on both KV widths and turns the window into the observable — a kernel that
//! fell back to `positions`, or read one cell too many, returns a different row.
//! A growing-window arm covers the multi-key softmax at a named tolerance (the
//! two sides are different f32 reduction orders; see the comment there).
//!
//! `metal_attn_span_nonzero_start` pins the decode-at-an-offset shape the server
//! emits, bitwise.

use super::*;
use crate::graph::kvformat::KvFormat;
use crate::graph::ops::{AttnMeta, AttnMode};

const NH: usize = 2;
const NK: usize = 2;
const HD: usize = 4;
const NKT: usize = NK * HD;
const NQT: usize = NH * HD;
const N_CTX: usize = 128;

/// Windows that differ row by row: when a window resolves to the wrong cell the
/// returned V row changes, so a wrong window cannot alias the expected one.
fn fixture_row(seed: usize, row: usize) -> f32 {
    ((seed + row * 37) % 41) as f32 / 41.0 - 0.5
}

fn q_data(nt: usize) -> Vec<f32> {
    (0..NQT * nt).map(|i| fixture_row(3, i)).collect()
}
/// K/V rounded through f16 so the *same* fixture bytes are exact on both widths
/// (a plain f32 would be truncated by the Metal f16 store and no longer match
/// the CPU's f32 store).
fn kv_data(seed: usize, nt: usize) -> Vec<f32> {
    (0..NKT * nt)
        .map(|i| half::f16::from_f32(fixture_row(seed, i)).to_f32())
        .collect()
}

/// Build `store -> load -> attn(explicit_span)` and return the graph plus the
/// attention node id.
fn make_graph(nt: usize, fmt: KvFormat) -> (ComputeGraph, usize) {
    let mut gb = GraphBuilder::new();
    gb.set_explicit_span(true);
    gb.set_kv_format(fmt);
    let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
    let q = gb.input("q", [NQT, nt, 1, 1], DType::F32);
    let k = gb.input("k", [NKT, nt, 1, 1], DType::F32);
    let v = gb.input("v", [NKT, nt, 1, 1], DType::F32);
    gb.kvcache_store(0, k, v, N_CTX);
    let kv = gb.kvcache_load(0, NKT, N_CTX, NK);
    let o = gb.attn(
        q,
        kv,
        pos,
        AttnMode::Gqa,
        AttnMeta {
            layer: 0,
            n_head: NH,
            n_head_kv: NK,
            hd: HD,
            hd_kv: HD,
            nkt: NKT,
            scale: 1.0 / (HD as f32).sqrt(),
        },
    );
    gb.output(o);
    (gb.build(), o)
}

/// Run one forward through the production fill entry point
/// (`GraphAllocator::fill_batch_inputs`) on CPU or Metal, with `reservations`
/// taken before the graph allocates the arena.
#[allow(clippy::too_many_arguments)]
fn run(
    g: &ComputeGraph,
    out: usize,
    batch: &Batch,
    reservations: &[(u32, usize)],
    fmt: KvFormat,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    positions: &[u32],
    metal: bool,
) -> Vec<f32> {
    let mut alloc = GraphAllocator::new();
    if metal {
        assert!(alloc.enable_metal(), "MPS unavailable; the gate cannot run");
    }
    // `BackendScheduler::execute` partitions by `node.backend` and does **not**
    // assign (an unset node defaults to CPU), so stamp Metal explicitly — the
    // sibling Metal fixtures do the same. Without this the "Metal" arm would run
    // on the CPU and the gate would pass vacuously.
    let stamped = metal.then(|| {
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::METAL);
        }
        g2
    });
    let g = stamped.as_ref().unwrap_or(g);
    alloc.set_kv_format(fmt);
    alloc.kv_set_capacity(N_CTX);
    for &(seq, cap) in reservations {
        alloc.kv_reserve_seq(seq, cap).expect("reserve a run");
    }
    alloc.alloc_graph(g).expect("alloc graph");
    alloc
        .fill_input_i32(g, "positions", positions)
        .expect("fill positions");
    alloc.fill_input(g, "q", q).expect("fill q");
    alloc.fill_input(g, "k", k).expect("fill k");
    alloc.fill_input(g, "v", v).expect("fill v");
    alloc
        .fill_batch_inputs(g, batch)
        .expect("resolve cells + attn_span");
    BackendScheduler::new()
        .execute(g, &mut alloc)
        .expect("execute");
    alloc.copy_to_cpu(out).expect("read output")
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Every query's V row, token-major — the exact output of a one-cell window
/// (softmax over one key is exactly 1.0, whatever q/K are).
fn single_key_reference(v: &[f32], nt: usize) -> Vec<f32> {
    let mut want = vec![0.0f32; NQT * nt];
    for t in 0..nt {
        for h in 0..NH {
            for d in 0..HD {
                want[t * NQT + h * HD + d] = v[t * NKT + h * HD + d];
            }
        }
    }
    want
}

/// E1 `attn_span`: a batch of one-token sequences, each at its own cell, must
/// return each sequence's own V row **bitwise** on CPU and on Metal, in both KV
/// widths. `nt` sweeps the decode (1), a small batch (8) and a prefill-sized
/// batch (40); first-fit reservations put sequence `i` at cell `i`, so every
/// sequence but the first also exercises a non-zero start.
///
/// Bar named before measuring: max|Δ| **== 0.0** (bitwise). A one-key softmax is
/// exactly 1, so the arithmetic has no reduction-order freedom to hide a wrong
/// window; the second half of the test additionally compares the Metal output
/// to the V rows computed on the host, so a kernel that resolved every query to
/// row 0 could not pass by aliasing a shared constant.
#[test]
fn metal_attn_span_matches_cpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    for fmt in [KvFormat::F32, KvFormat::F16] {
        for nt in [1usize, 8, 40] {
            let (g, out) = make_graph(nt, fmt);
            let q = q_data(nt);
            let k = kv_data(11, nt);
            let v = kv_data(23, nt);

            // Sequence `i`, one token at its own position 0, reserved first-fit
            // so it lands at cell `i`.
            let seq_ids: Vec<u32> = (0..nt as u32).collect();
            let positions = vec![0u32; nt];
            let batch = Batch::new(
                vec![0u32; nt],
                positions.iter().map(|&p| p as usize).collect(),
                seq_ids.clone(),
            );
            let reservations: Vec<(u32, usize)> = seq_ids.iter().map(|&s| (s, 1)).collect();

            let cpu = run(
                &g,
                out,
                &batch,
                &reservations,
                fmt,
                &q,
                &k,
                &v,
                &positions,
                false,
            );
            let met = run(
                &g,
                out,
                &batch,
                &reservations,
                fmt,
                &q,
                &k,
                &v,
                &positions,
                true,
            );

            let want = single_key_reference(&v, nt);
            assert_eq!(
                max_delta(&cpu, &want),
                0.0,
                "{fmt:?} nt={nt}: the CPU reference is not the V rows — the fixture is wrong"
            );
            let d = max_delta(&met, &cpu);
            eprintln!("[attn_span] {fmt:?} nt={nt} one-cell windows: max|Δ|={d}");
            assert_eq!(
                d, 0.0,
                "{fmt:?} nt={nt}: Metal's attn_span window disagrees with the CPU reference \
                 (max|Δ| = {d})"
            );
            // Non-vacuous: the output must be the distinct V rows, not zeros.
            assert!(
                want.iter().any(|&x| x != 0.0) && met.iter().any(|&x| x != 0.0),
                "{fmt:?} nt={nt}: the fixture returned all zeros; the comparison is vacuous"
            );
        }
    }
}

/// E1 `attn_span`: a **growing** window (one sequence, `nt` tokens at
/// positions `0..nt`, run at a non-zero start) exercises the multi-key online
/// softmax — the `nt > 1` prefill arm the brief's sweep asks for.
///
/// Bar named before measuring: max|Δ| **<= 1e-4**, the named class
/// `metal_attn_kv_matches_cpu` already uses for Metal-vs-CPU attention. The two
/// sides are different f32 reduction orders (the CPU reduces a whole row
/// sequentially; Metal reduces per-lane and then `simd_sum`s), so a bitwise bar
/// is not achievable for a real softmax — the one-cell gate above is where
/// `== 0.0` is the honest bar. The fixture's K/V are f16-representable, so even
/// the f16 arm differs only in reduction order, and the output is asserted
/// non-vacuous.
#[test]
fn metal_attn_span_multi_key_matches_cpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const START: usize = 9;
    for fmt in [KvFormat::F32, KvFormat::F16] {
        for nt in [1usize, 8, 40] {
            let (g, out) = make_graph(nt, fmt);
            let q = q_data(nt);
            let k = kv_data(11, nt);
            let v = kv_data(23, nt);

            // A holder owns [0, START); the subject's run starts at START.
            let positions: Vec<u32> = (0..nt as u32).collect();
            let batch = Batch::new(
                vec![0u32; nt],
                positions.iter().map(|&p| p as usize).collect(),
                vec![7u32; nt],
            );
            let reservations = [(99u32, START), (7u32, nt + 1)];

            let cpu = run(
                &g,
                out,
                &batch,
                &reservations,
                fmt,
                &q,
                &k,
                &v,
                &positions,
                false,
            );
            let met = run(
                &g,
                out,
                &batch,
                &reservations,
                fmt,
                &q,
                &k,
                &v,
                &positions,
                true,
            );

            let d = max_delta(&met, &cpu);
            eprintln!("[attn_span multi] {fmt:?} nt={nt} start={START}: max|Δ|={d}");
            assert!(
                met.iter().any(|&x| x != 0.0) && cpu.iter().any(|&x| x != 0.0),
                "{fmt:?} nt={nt}: the multi-key fixture returned all zeros; vacuous"
            );
            assert!(
                d <= 1e-4,
                "{fmt:?} nt={nt}: Metal's growing-window attention diverges from the CPU \
                 reference (max|Δ| = {d} > 1e-4)"
            );
        }
    }
}

/// E1 `attn_span`: a single sequence whose run starts at a **non-zero cell**,
/// decode shape (`nt = 1`). The output is exactly the stored V row of cell
/// `START`, so the bar is **max|Δ| == 0.0** on both KV widths. This is the shape
/// the server emits for a slot after another slot's run.
#[test]
fn metal_attn_span_nonzero_start() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const START: usize = 37;
    for fmt in [KvFormat::F32, KvFormat::F16] {
        let nt = 1usize;
        let (g, out) = make_graph(nt, fmt);
        let q = q_data(nt);
        let k = kv_data(11, nt);
        let v = kv_data(23, nt);

        // A holder owns [0, START); the subject's one token lives at START.
        let batch = Batch::new(vec![0u32], vec![0usize], vec![7u32]);
        let reservations = [(99u32, START), (7u32, 2)];

        let cpu = run(&g, out, &batch, &reservations, fmt, &q, &k, &v, &[0], false);
        let met = run(&g, out, &batch, &reservations, fmt, &q, &k, &v, &[0], true);

        let want = single_key_reference(&v, nt);
        assert_eq!(
            max_delta(&cpu, &want),
            0.0,
            "{fmt:?}: the CPU reference is not V(cell {START})"
        );
        let d = max_delta(&met, &cpu);
        eprintln!("[attn_span nonzero] {fmt:?} start={START}: max|Δ|={d}");
        assert_eq!(
            d, 0.0,
            "{fmt:?}: a non-zero-start attn_span window disagrees with the CPU reference \
             (max|Δ| = {d})"
        );
    }
}
