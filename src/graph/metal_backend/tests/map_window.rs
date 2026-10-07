//! C8b S4 `kv_map` windowed-attention gates for the Metal backend (issue #362).
//!
//! Before this ticket Metal refused every `Op::Attn { explicit_span: true }`
//! node whose window input was the `kv_map` layout, so the server copied a
//! shared prefix (C8a) instead of reading it in place, and the macOS real-model
//! set's `a_store_inside_a_shared_prefix_takes_a_private_row` was its one red.
//!
//! These gates drive the **new** `kernel_gqa_attn_map_f32/_f16`
//! (`src/metal/kernels/attn_window.metal`) through the production graph path and
//! compare it against both an independently computed value and the CPU
//! reference.
//!
//! - [`metal_map_single_cell_matches_the_v_row`] is the value arm (rule 1): each
//!   query names exactly one cell, so a single-key softmax is exactly 1 and the
//!   output is that cell's V row **bitwise**. The expected row is computed on the
//!   host, independent of both backends, so a kernel that resolved every run to
//!   row 0 (or read `lo + ki` as if it were a span) cannot alias a shared value.
//! - [`metal_map_matches_the_span_and_a_wrong_base_differs`] is the control arm
//!   (rule 2): the same cells named as a one-range `attn_span` and as a `kv_map`
//!   run list must agree **bitwise** (the two modes differ in how they resolve a
//!   row, not in what they compute over it); a map whose first run names the
//!   wrong base cell must differ; and a multi-run window must not degenerate to
//!   one key (the extra rows must move the output).

use super::*;
use crate::graph::kvformat::KvFormat;
use crate::graph::ops::{AttnMeta, AttnMode};

const NH: usize = 2;
const NK: usize = 2;
const HD: usize = 4;
const NKT: usize = NK * HD;
const NQT: usize = NH * HD;
const N_CTX: usize = 128;
const KMAX: usize = crate::graph::kvcache::KV_MAP_MAX_SPANS;

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
/// the host reference).
fn kv_data(seed: usize, rows: usize) -> Vec<f32> {
    (0..NKT * rows)
        .map(|i| half::f16::from_f32(fixture_row(seed, i)).to_f32())
        .collect()
}

/// Build `store(rows) -> load -> attn(explicit window)`. With `map` the graph's
/// window input is the `kv_map` layout; without it, the one-range `attn_span`
/// layout. `nt_q` is the query count (the attention output); `rows` is how many
/// K/V rows are stored (they may differ — a decode query against a long window).
fn make_graph(nt_q: usize, rows: usize, fmt: KvFormat, map: bool) -> (ComputeGraph, usize) {
    let mut gb = GraphBuilder::new();
    gb.set_explicit_span(true);
    gb.set_kv_map(map);
    gb.set_kv_format(fmt);
    let pos = gb.input("positions", [nt_q, 1, 1, 1], DType::I32);
    let q = gb.input("q", [NQT, nt_q, 1, 1], DType::F32);
    let k = gb.input("k", [NKT, rows, 1, 1], DType::F32);
    let v = gb.input("v", [NKT, rows, 1, 1], DType::F32);
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

/// Run one forward through the graph allocator + scheduler on CPU or Metal. The
/// store rows go to cells `0..rows`; the window is filled by name (`kv_map` for a
/// map graph, `attn_span` otherwise). The input's **size** selects the read
/// kernel, exactly as production does.
#[allow(clippy::too_many_arguments)]
fn run(
    g: &ComputeGraph,
    out: usize,
    fmt: KvFormat,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cells: &[u32],
    window: &[u32],
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
    alloc.alloc_graph(g).expect("alloc graph");
    alloc.fill_input(g, "q", q).expect("fill q");
    alloc.fill_input(g, "k", k).expect("fill k");
    alloc.fill_input(g, "v", v).expect("fill v");
    alloc.fill_input_i32(g, "cells", cells).expect("fill cells");
    let is_map = g.inputs.iter().any(|&i| g.node(i).name == "kv_map");
    if is_map {
        alloc
            .fill_input_i32(g, "kv_map", window)
            .expect("fill kv_map");
    } else {
        alloc
            .fill_input_i32(g, "attn_span", window)
            .expect("fill attn_span");
    }
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

/// Every query's V row **bitwise** — the exact output of a one-cell window
/// (softmax over one key is exactly 1.0, whatever q/K are). `named[t]` is the
/// cell query `t`'s window resolves to.
fn v_rows(v: &[f32], named: &[usize]) -> Vec<f32> {
    let nt = named.len();
    let mut want = vec![0.0f32; NQT * nt];
    for t in 0..nt {
        for h in 0..NH {
            for d in 0..HD {
                want[t * NQT + h * HD + d] = v[named[t] * NKT + h * HD + d];
            }
        }
    }
    want
}

/// A `kv_map` window with one `(cell, len)` run per query, padded to
/// `KV_MAP_MAX_SPANS`. `runs[t]` is `(cell, len)` — a zero length is a padding
/// slot, so a query always has exactly the runs it is given.
fn map_window(runs: &[&[(u32, u32)]]) -> Vec<u32> {
    let mut w = vec![0u32; runs.len() * KMAX * 2];
    for (t, rs) in runs.iter().enumerate() {
        assert!(rs.len() <= KMAX, "the fixture needs more than {KMAX} runs");
        for (s, &(cell, len)) in rs.iter().enumerate() {
            w[(t * KMAX + s) * 2] = cell;
            w[(t * KMAX + s) * 2 + 1] = len;
        }
    }
    w
}

/// One query's one-range `attn_span` window: `[lo, hi)`.
fn span_window(spans: &[(u32, u32)]) -> Vec<u32> {
    let nt = spans.len();
    let mut w = vec![0u32; 2 * nt];
    for (t, &(lo, hi)) in spans.iter().enumerate() {
        w[t] = lo;
        w[nt + t] = hi;
    }
    w
}

/// E1/C8b S4 rule 1 for Metal: a batch of queries, each naming **one** cell
/// through the `kv_map` layout, must return each cell's V row **bitwise** on
/// both KV widths. The expected row is computed on the host, independent of both
/// backends.
///
/// Bar named before measuring: max|Δ| **== 0.0** (bitwise). A one-key softmax is
/// exactly 1, so there is no reduction-order freedom to hide a wrong window; a
/// kernel that resolved every run to cell 0, or read the map as a span
/// (`lo + ki`), returns a different row and is red.
#[test]
fn metal_map_single_cell_matches_the_v_row() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const N: usize = 6;
    for fmt in [KvFormat::F32, KvFormat::F16] {
        let (g, out) = make_graph(N, N, fmt, true);
        let q = q_data(N);
        let k = kv_data(11, N);
        let v = kv_data(23, N);
        let cells: Vec<u32> = (0..N as u32).collect();
        // Query `t` names cell `t` alone.
        let run_data: Vec<Vec<(u32, u32)>> = (0..N as u32).map(|t| vec![(t, 1u32)]).collect();
        let runs: Vec<&[(u32, u32)]> = run_data.iter().map(|r| r.as_slice()).collect();
        let map = map_window(&runs);
        // A one-range span for the SAME cells: query `t` = `[t, t+1)`.
        let spans: Vec<(u32, u32)> = (0..N as u32).map(|t| (t, t + 1)).collect();
        let span = span_window(&spans);

        let want = v_rows(&v, &(0..N).collect::<Vec<_>>());

        // The span graph's output is the same value through the existing path —
        // a fixture check, not the bar.
        let (gs, outs) = make_graph(N, N, fmt, false);
        let span_cpu = run(&gs, outs, fmt, &q, &k, &v, &cells, &span, false);
        let map_cpu = run(&g, out, fmt, &q, &k, &v, &cells, &map, false);
        let map_met = run(&g, out, fmt, &q, &k, &v, &cells, &map, true);

        let d_cpu = max_delta(&map_cpu, &want);
        let d_met = max_delta(&map_met, &want);
        eprintln!(
            "[map single] {fmt:?} n={N}: cpu-vs-Vrow={d_cpu} metal-vs-Vrow={d_met} \
             span-vs-map={}",
            max_delta(&span_cpu, &map_cpu)
        );
        assert_eq!(
            d_cpu, 0.0,
            "{fmt:?}: the CPU map output is not the host V rows — the fixture is wrong"
        );
        assert_eq!(
            d_met, 0.0,
            "{fmt:?}: Metal's kv_map single-cell window disagrees with the host V row \
             (max|Δ| = {d_met})"
        );
        // Non-vacuous: the output must be the distinct V rows, not zeros.
        assert!(
            want.iter().any(|&x| x != 0.0) && map_met.iter().any(|&x| x != 0.0),
            "{fmt:?}: the fixture returned all zeros; the comparison is vacuous"
        );
    }
}

/// C8b S4 rule 2 + value arm for Metal. Two fixtures:
///
/// - **Value**: a one-query window whose runs are **non-adjacent** (cells
///   `{0, 1, 4, 5}`), compared to the CPU reference over the *same* runs.
///   Non-adjacency is what makes the run walk load-bearing: a kernel that
///   resolved the window as one contiguous range from the first run's cell
///   (`base + ki`) would read cells `{0, 1, 2, 3}` and is far outside the named
///   tolerance. Bar: max|Δ| **<= 1e-4** against the CPU reference — the class
///   `metal_attn_kv_matches_cpu` uses, because the two sides are different f32
///   reduction orders. (The single-cell arm above is the independent value
///   oracle; this arm's reference is the CPU kernel.)
/// - **Layout equivalence**: the same **contiguous** cells named as a one-range
///   `attn_span` and as a `kv_map` run list must agree **bitwise** on Metal —
///   the two layouts resolve the same rows and run the same arithmetic.
///
/// Controls: a map whose first run names the wrong base cell must differ; the
/// multi-run window must not degenerate to one key (the extra rows must move the
/// output away from the one-cell result).
#[test]
fn metal_map_matches_the_span_and_a_wrong_base_differs() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    /// The named cross-backend attention class (`metal_attn_kv_matches_cpu`).
    const TOL: f32 = 1e-4;

    const R: usize = 6;
    for fmt in [KvFormat::F32, KvFormat::F16] {
        let (gm, outm) = make_graph(1, R, fmt, true);
        let (gs, outs) = make_graph(1, R, fmt, false);
        let q = q_data(1);
        let k = kv_data(11, R);
        let v = kv_data(23, R);
        let cells: Vec<u32> = (0..R as u32).collect();

        // (1) Value arm: two NON-adjacent runs -> cells {0,1,4,5}. A kernel that
        // ignored the run boundary would read {0,1,2,3} and is red here.
        let runs_na = map_window(&[&[(0, 2), (4, 2)]]);
        let na_cpu = run(&gm, outm, fmt, &q, &k, &v, &cells, &runs_na, false);
        let na_met = run(&gm, outm, fmt, &q, &k, &v, &cells, &runs_na, true);
        let d_na = max_delta(&na_met, &na_cpu);

        // (2) Layout equivalence: the SAME contiguous cells {0..6} named as a
        // one-range span and as a two-run map, bitwise on Metal.
        let runs_c = map_window(&[&[(0, 3), (3, 3)]]);
        let span_c = span_window(&[(0, R as u32)]);
        let c_map = run(&gm, outm, fmt, &q, &k, &v, &cells, &runs_c, true);
        let c_span = run(&gs, outs, fmt, &q, &k, &v, &cells, &span_c, true);
        let d_modes = max_delta(&c_map, &c_span);

        eprintln!(
            "[map multi] {fmt:?} R={R}: metal-vs-cpu (non-adjacent runs)={d_na} \
             span-vs-map (contiguous)={d_modes} (tol {TOL})"
        );
        assert!(
            d_na <= TOL,
            "{fmt:?}: Metal's kv_map non-adjacent run window diverges from the CPU reference \
             (max|Δ| = {d_na} > {TOL})"
        );
        assert_eq!(
            d_modes, 0.0,
            "{fmt:?}: the same cells named as a span and as a kv_map disagree on Metal \
             (max|Δ| = {d_modes})"
        );

        // Control (a): moving the first run's base cell 0 -> 2 ({2,3,4,5}) must
        // change the output.
        let wrong = map_window(&[&[(2, 2), (4, 2)]]);
        let wrong_met = run(&gm, outm, fmt, &q, &k, &v, &cells, &wrong, true);
        let d_wrong = max_delta(&wrong_met, &na_met);
        assert!(
            d_wrong > 0.0,
            "{fmt:?}: a map naming the wrong base cell gave the same output; the window is \
             not observable in the fixture"
        );

        // Control (b): the multi-run window must not degenerate to one key. A
        // map naming only the first cell is a one-key softmax, so its output is
        // that cell's V row; the two-run output must differ from it (the extra
        // rows move the softmax weights).
        let single = map_window(&[&[(0, 1)]]);
        let single_met = run(&gm, outm, fmt, &q, &k, &v, &cells, &single, true);
        assert!(
            max_delta(&single_met, &na_met) > 0.0,
            "{fmt:?}: the multi-run window returned the single-key result; the fixture \
             degenerated to one key"
        );
    }
}
