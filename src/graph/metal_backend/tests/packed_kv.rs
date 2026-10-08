//! C4 S2b Metal packed `q8_0` KV gates (issue #310).
//!
//! The packed path is **implemented on this branch but not enabled**: production
//! keeps `READS_PACKED_KV = false` (the fast causal/windowed families refuse a
//! packed region and the classic fallback measures 4–17× slower than f16 — see
//! `docs/METAL-BACKEND-DESIGN.md` §4.4), so `MINFER_CACHE_TYPE=q8_0` is still
//! refused at load and `GraphAllocator::ensure_kv` still refuses a packed region
//! by default. These gates therefore run under a **documented `#[cfg(test)]`
//! capability seam** ([`PackedKvEnabled`], the registry's thread-local
//! `set_force_packed_kv`) that lets them drive the real production entry points
//! without flipping the shipped capability.
//!
//! #310 adds the packed store (`kernel_store_kv_q8_0`) and the packed reads
//! (`kernel_gqa_attn_q8_0` for the causal classic path and
//! `kernel_gqa_attn_window_q8_0` / `kernel_gqa_attn_map_q8_0` for the explicit
//! window layouts), selected whenever the engine's KV format is Q8_0.
//!
//! These gates drive the production graph path:
//!
//! - [`metal_packed_store_is_byte_identical_to_the_cpu_quantizer`] is the
//!   value arm for the store (rule 1): the bytes Metal writes are compared to
//!   `kvformat::pack_q8_0_cell`'s (the CPU quantizer, `amax/127` f16 scale,
//!   round-ties-even) **bitwise**.
//! - [`metal_packed_decode_matches_the_dequantized_reference`] and
//!   [`metal_packed_attn_span_matches_the_v_row`] are the value arms for the
//!   reads: a one-key window's softmax is exactly 1, so the output is the cell's
//!   dequantized V row **bitwise**, computed on the host independently of
//!   Metal. `HD = 64` is deliberate — that is the shape whose causal/window fast
//!   families (flash / prefill / windowed flash) the dispatch bypasses for a
//!   packed region; a kernel that took the f32 fast path over packed bytes would
//!   return a different value and be red.
//! - [`metal_packed_prefill_matches_the_dequantized_reference`] covers the
//!   multi-key softmax at a named tolerance.
//! - [`metal_packed_session_round_trips`] is the C5 `FLAG_PACKED` round trip: a
//!   Metal arena saves and restores bitwise, and a fresh F32 allocator refuses
//!   the file (the header really carries Q8_0), which is the store + container
//!   half.
//! - [`metal_q8_0_kv_answers_like_f32_on_a_real_model`] (ignored) is the
//!   real-model end-to-end arm.

use super::*;
use crate::graph::kvformat::{pack_q8_0_cell, unpack_q8_0_cells, KvFormat};
use crate::graph::kvsession::KvSessionExpect;
use crate::graph::ops::{AttnMeta, AttnMode};

/// #310 gate seam: enable packed KV for the calling thread while a test runs.
///
/// Production ships `READS_PACKED_KV = false` (the fast families refuse packed,
/// so the classic fallback is 4–17× slower), but the implementation is real and
/// must stay exercised. This guard flips the registry's **thread-local**
/// override, so it drives the production `resolve` / `ensure_kv` / session
/// entry points without changing the shipped answer and without leaking into a
/// concurrently-running test. `Drop` restores it, so a panic cannot leave it on.
struct PackedKvEnabled;

impl PackedKvEnabled {
    fn on() -> Self {
        crate::graph::registry::set_force_packed_kv(true);
        Self
    }
}

impl Drop for PackedKvEnabled {
    fn drop(&mut self) {
        crate::graph::registry::set_force_packed_kv(false);
    }
}

const NH: usize = 2;
const NK: usize = 2;
const HD: usize = 64;
const NKT: usize = NK * HD;
const NQT: usize = NH * HD;
const N_CTX: usize = 128;
const KMAX: usize = crate::graph::kvcache::KV_MAP_MAX_SPANS;

/// Rows that differ element by element: a wrong window resolves to a different V
/// row, so it cannot alias the expected one.
fn fixture_row(seed: usize, i: usize) -> f32 {
    ((seed + i * 37) % 41) as f32 / 41.0 - 0.5
}

fn q_data(nt: usize) -> Vec<f32> {
    (0..NQT * nt).map(|i| fixture_row(3, i)).collect()
}

fn kv_data(seed: usize, rows: usize) -> Vec<f32> {
    (0..NKT * rows).map(|i| fixture_row(seed, i)).collect()
}

/// `store(rows) -> load -> attn`. With `map` the window input is the `kv_map`
/// layout; a packed graph takes the Q8_0 read kernels either way.
fn make_graph(nt_q: usize, rows: usize, map: bool) -> (ComputeGraph, usize) {
    let mut gb = GraphBuilder::new();
    gb.set_explicit_span(true);
    gb.set_kv_map(map);
    gb.set_kv_format(KvFormat::Q8_0);
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

/// A causal graph (`explicit_span = false`): the packed read is
/// `kernel_gqa_attn_q8_0`.
fn make_causal_graph(nt: usize) -> (ComputeGraph, usize) {
    let mut gb = GraphBuilder::new();
    gb.set_kv_format(KvFormat::Q8_0);
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

/// Build the allocator, stamp every node Metal, fill the window by name and run.
#[allow(clippy::too_many_arguments)]
fn run(
    g: &ComputeGraph,
    out: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cells: &[u32],
    window: &[u32],
) -> Vec<f32> {
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_metal(), "MPS unavailable; the gate cannot run");
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    alloc.set_kv_format(KvFormat::Q8_0);
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g2).expect("alloc graph");
    alloc.fill_input(&g2, "q", q).expect("fill q");
    alloc.fill_input(&g2, "k", k).expect("fill k");
    alloc.fill_input(&g2, "v", v).expect("fill v");
    alloc
        .fill_input_i32(&g2, "cells", cells)
        .expect("fill cells");
    let is_map = g2.inputs.iter().any(|&i| g2.node(i).name == "kv_map");
    if is_map {
        alloc
            .fill_input_i32(&g2, "kv_map", window)
            .expect("fill kv_map");
    } else {
        alloc
            .fill_input_i32(&g2, "attn_span", window)
            .expect("fill attn_span");
    }
    BackendScheduler::new()
        .execute(&g2, &mut alloc)
        .expect("execute");
    alloc.copy_to_cpu(out).expect("read output")
}

fn max_delta(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// The CPU quantizer's round trip of `rows` f32 rows, laid out token-major —
/// the exact value a one-cell window returns and the reference for the
/// multi-key oracle.
fn q8_roundtrip(src: &[f32], rows: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; NKT * rows];
    for r in 0..rows {
        let mut cell = vec![0.0f32; KvFormat::Q8_0.row_elems(NKT)];
        pack_q8_0_cell(&mut cell, NKT, &src[r * NKT..(r + 1) * NKT]);
        unpack_q8_0_cells(&cell, NKT, 0, 1, &mut out[r * NKT..(r + 1) * NKT]);
    }
    out
}

/// Every query's one named V row, token-major — the exact output of a one-cell
/// window (softmax over one key is exactly 1.0).
fn v_rows(v: &[f32], named: &[usize]) -> Vec<f32> {
    let mut want = vec![0.0f32; NQT * named.len()];
    for (t, &cell) in named.iter().enumerate() {
        for h in 0..NH {
            for d in 0..HD {
                want[t * NQT + h * HD + d] = v[cell * NKT + h * HD + d];
            }
        }
    }
    want
}

/// An independent host attention over dequantized K/V, one `(lo, hi)` window per
/// query (rule 1's oracle). Sequential f32 accumulation, so a multi-key
/// comparison carries the usual reduction-order tolerance.
#[allow(clippy::too_many_arguments)]
fn host_attention(q: &[f32], k: &[f32], v: &[f32], spans: &[(u32, u32)], scale: f32) -> Vec<f32> {
    let nt = spans.len();
    let gqa = NH / NK;
    let mut out = vec![0.0f32; NQT * nt];
    for t in 0..nt {
        let (lo, hi) = spans[t];
        for h in 0..NH {
            let hk = h / gqa;
            let mut e = Vec::new();
            let mut mx = f32::NEG_INFINITY;
            for cell in lo..hi {
                let mut dot = 0.0f32;
                for d in 0..HD {
                    dot += q[t * NQT + h * HD + d] * k[cell as usize * NKT + hk * HD + d];
                }
                dot *= scale;
                mx = mx.max(dot);
                e.push(dot);
            }
            let exps: Vec<f32> = e.iter().map(|&x| (x - mx).exp()).collect();
            let s: f32 = exps.iter().sum();
            for d in 0..HD {
                let mut acc = 0.0f32;
                for (i, cell) in (lo..hi).enumerate() {
                    acc += exps[i] / s * v[cell as usize * NKT + hk * HD + d];
                }
                out[t * NQT + h * HD + d] = acc;
            }
        }
    }
    out
}

/// A one-cell `kv_map` window: query `t` names cell `t` alone.
fn map_window(cells: &[u32]) -> Vec<u32> {
    let mut w = vec![0u32; cells.len() * KMAX * 2];
    for (t, &c) in cells.iter().enumerate() {
        w[t * KMAX * 2] = c;
        w[t * KMAX * 2 + 1] = 1;
    }
    w
}

fn span_window(spans: &[(u32, u32)]) -> Vec<u32> {
    let mut w = vec![0u32; 2 * spans.len()];
    for (t, &(lo, hi)) in spans.iter().enumerate() {
        w[t] = lo;
        w[spans.len() + t] = hi;
    }
    w
}

/// Rule 1, store half: the bytes `kernel_store_kv_q8_0` writes are the CPU
/// quantizer's (`kvformat::pack_q8_0_cell`) **bitwise**, cell word for cell word.
/// The row is read back through the production host-read hook
/// (`GraphAllocator::copy_kv_to_cpu`) after a real forward, so this drives the
/// launch site, not a mirror.
///
/// Bar named before measuring: exact `f32::to_bits` equality on every word of
/// every written cell.
#[test]
fn metal_packed_store_is_byte_identical_to_the_cpu_quantizer() {
    let _g = crate::metal::metal_test_lock();
    let _packed = PackedKvEnabled::on();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const ROWS: usize = 6;
    let (g, out) = make_graph(1, ROWS, false);
    let q = q_data(1);
    let k = kv_data(11, ROWS);
    let v = kv_data(23, ROWS);
    let cells: Vec<u32> = (0..ROWS as u32).collect();
    let span = span_window(&[(0, 1)]);

    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_metal(), "MPS unavailable");
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    alloc.set_kv_format(KvFormat::Q8_0);
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "q", &q).unwrap();
    alloc.fill_input(&g2, "k", &k).unwrap();
    alloc.fill_input(&g2, "v", &v).unwrap();
    alloc.fill_input_i32(&g2, "cells", &cells).unwrap();
    alloc.fill_input_i32(&g2, "attn_span", &span).unwrap();
    BackendScheduler::new().execute(&g2, &mut alloc).unwrap();
    let _ = alloc.copy_to_cpu(out).unwrap();

    let row_elems = KvFormat::Q8_0.row_elems(NKT);
    let (k_words, v_words) = alloc.copy_kv_to_cpu(0).unwrap();
    for src in [&k, &v] {
        let words = if std::ptr::eq(src, &k) {
            &k_words
        } else {
            &v_words
        };
        for cell in 0..ROWS {
            let mut want = vec![0.0f32; row_elems];
            pack_q8_0_cell(&mut want, NKT, &src[cell * NKT..(cell + 1) * NKT]);
            let got = &words[cell * row_elems..(cell + 1) * row_elems];
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "cell {cell} word {i}: Metal {:08x} != CPU quantizer {:08x} \
                     (the Q8_0 store must be byte-identical)",
                    g.to_bits(),
                    w.to_bits()
                );
            }
        }
    }
    // Non-vacuous: the packed cell is not all zeros.
    assert!(
        k_words[..row_elems].iter().any(|&x| x.to_bits() != 0),
        "the stored K cell is all zeros; the comparison is vacuous"
    );
}

/// Rule 1, causal read half (`kernel_gqa_attn_q8_0`, decode shape): a one-token
/// sequence with a one-cell causal window has a one-key softmax (exactly 1), so
/// the output is the cell's dequantized V row **bitwise**. The expected row is
/// computed on the host from the CPU quantizer, independent of Metal.
#[test]
fn metal_packed_decode_matches_the_dequantized_reference() {
    let _g = crate::metal::metal_test_lock();
    let _packed = PackedKvEnabled::on();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    let (g, out) = make_causal_graph(1);
    let q = q_data(1);
    let k = kv_data(11, 1);
    let v = kv_data(23, 1);
    let cells = [0u32];
    // The causal graph needs `positions` = [0]; the window input is ignored.
    let got = run_causal(&g, out, &q, &k, &v, &cells, &[0]);

    let v_deq = q8_roundtrip(&v, 1);
    let want = v_rows(&v_deq, &[0]);
    let d = max_delta(&got, &want);
    eprintln!("[packed decode] one-key: max|Δ|={d}");
    assert!(
        want.iter().any(|&x| x != 0.0) && got.iter().any(|&x| x != 0.0),
        "the fixture returned all zeros; the comparison is vacuous"
    );
    assert_eq!(
        d, 0.0,
        "Metal's packed causal decode is not the dequantized V row (max|Δ| = {d})"
    );
}

/// Rule 1, causal read half, multi-key: an `nt`-token prefill at `HD = 64` (the
/// shape whose `kernel_flash_attn_blk_*` family would be chosen for f32/f16) is
/// bypassed for a packed region and runs `kernel_gqa_attn_q8_0`. The host oracle
/// runs a scalar softmax over the dequantized K/V, so a packed read that used
/// the wrong block base or stride is red.
///
/// Bar named before measuring: max|Δ| ≤ 1e-4, the class `metal_attn_span_multi_key_matches_cpu`
/// already uses for Metal-vs-CPU attention (different f32 reduction orders).
#[test]
fn metal_packed_prefill_matches_the_dequantized_reference() {
    let _g = crate::metal::metal_test_lock();
    let _packed = PackedKvEnabled::on();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const NT: usize = 8;
    let (g, out) = make_causal_graph(NT);
    let q = q_data(NT);
    let k = kv_data(11, NT);
    let v = kv_data(23, NT);
    let cells: Vec<u32> = (0..NT as u32).collect();
    let positions: Vec<u32> = (0..NT as u32).collect();
    let got = run_causal(&g, out, &q, &k, &v, &cells, &positions);

    let k_deq = q8_roundtrip(&k, NT);
    let v_deq = q8_roundtrip(&v, NT);
    let spans: Vec<(u32, u32)> = (0..NT as u32).map(|t| (0, t + 1)).collect();
    let want = host_attention(&q, &k_deq, &v_deq, &spans, 1.0 / (HD as f32).sqrt());
    let d = max_delta(&got, &want);
    eprintln!("[packed prefill] nt={NT}: max|Δ|={d}");
    assert!(
        want.iter().any(|&x| x != 0.0) && got.iter().any(|&x| x != 0.0),
        "the fixture returned all zeros; the comparison is vacuous"
    );
    assert!(
        d <= 1e-4,
        "Metal's packed causal prefill diverges from the dequantized reference (max|Δ| = {d})"
    );
}

/// Causal run helper: fill `positions` too (the classic kernel derives its window
/// from them), then `attn_span` is left at zero and unused (`explicit_span` false).
fn run_causal(
    g: &ComputeGraph,
    out: usize,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    cells: &[u32],
    positions: &[u32],
) -> Vec<f32> {
    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_metal(), "MPS unavailable");
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    alloc.set_kv_format(KvFormat::Q8_0);
    alloc.kv_set_capacity(N_CTX);
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "q", q).unwrap();
    alloc.fill_input(&g2, "k", k).unwrap();
    alloc.fill_input(&g2, "v", v).unwrap();
    alloc.fill_input_i32(&g2, "cells", cells).unwrap();
    alloc.fill_input_i32(&g2, "positions", positions).unwrap();
    // The causal graph still carries an `attn_span` input; fill it so the
    // allocation check passes even though the causal kernel ignores it.
    let nt = positions.len();
    let span = span_window(&(0..nt).map(|t| (0u32, (t + 1) as u32)).collect::<Vec<_>>());
    alloc.fill_input_i32(&g2, "attn_span", &span).unwrap();
    BackendScheduler::new().execute(&g2, &mut alloc).unwrap();
    alloc.copy_to_cpu(out).unwrap()
}

/// Rule 1, window read half (`kernel_gqa_attn_window_q8_0` and its `kv_map`
/// sibling): one-cell windows resolve to the dequantized V row **bitwise**, in
/// both explicit layouts, at `HD = 64` (the shape whose fast windowed-flash
/// family the packed dispatch bypasses).
#[test]
fn metal_packed_attn_span_matches_the_v_row() {
    let _g = crate::metal::metal_test_lock();
    let _packed = PackedKvEnabled::on();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const N: usize = 6;
    for map in [false, true] {
        let (g, out) = make_graph(N, N, map);
        let q = q_data(N);
        let k = kv_data(11, N);
        let v = kv_data(23, N);
        let cells: Vec<u32> = (0..N as u32).collect();
        let window = if map {
            map_window(&cells)
        } else {
            span_window(&(0..N as u32).map(|t| (t, t + 1)).collect::<Vec<_>>())
        };
        let got = run(&g, out, &q, &k, &v, &cells, &window);

        let v_deq = q8_roundtrip(&v, N);
        let want = v_rows(&v_deq, &(0..N).collect::<Vec<_>>());
        let d = max_delta(&got, &want);
        eprintln!("[packed window map={map}] n={N}: max|Δ|={d}");
        assert!(
            want.iter().any(|&x| x != 0.0) && got.iter().any(|&x| x != 0.0),
            "map={map}: the fixture returned all zeros; the comparison is vacuous"
        );
        assert_eq!(
            d, 0.0,
            "map={map}: Metal's packed one-cell window is not the dequantized V row \
             (max|Δ| = {d})"
        );
    }
}

/// The default cached gate model (Qwen2.5-0.5B Q4_0), overridable by
/// `MINFER_C4_MODEL` / `MINFER_BATCH_TEST_MODEL`.
fn cached_gate_model() -> Option<std::path::PathBuf> {
    for key in ["MINFER_C4_MODEL", "MINFER_BATCH_TEST_MODEL"] {
        if let Some(p) = std::env::var_os(key) {
            let p = std::path::PathBuf::from(p);
            if p.exists() {
                return Some(p);
            }
        }
    }
    let home = std::env::var_os("HOME")?;
    let p = std::path::PathBuf::from(home).join(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    p.exists().then_some(p)
}

fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in x.iter().enumerate() {
        if v > x[best] {
            best = i;
        }
    }
    best as u32
}

/// Deliverable #4: a **real model** loaded twice on Metal — one engine resolved
/// `f32`, one `q8_0` — runs the same prefill + decode, and the packed engine's
/// logits stay in the C4 tolerance class while its regions are at least 3x
/// smaller. This is the synthetic gates' real-model counterpart and the Metal
/// twin of `qwen2::graph::tests::cuda_kv::a_packed_kv_cache_answers_like_the_f32_one`;
/// it asserts the engine really landed on Metal (a silent CPU fallback would
/// measure the CPU's packed path).
///
/// Bars (named from the CUDA C4 gate, the same quantization on the same cells):
/// at the f32 reference's argmax |Δ| ≤ 1.0, and over the whole logit vector
/// |Δ| ≤ 4.0 (a gross-error detector — a wrong cell width is off by the spread).
/// Ignored: needs the cached model and a Metal device.
///
/// It enables the packed capability for its own run via [`PackedKvEnabled`] — the
/// explicit enabling path an ignored arm is allowed, because production keeps
/// `READS_PACKED_KV = false`.
#[test]
#[ignore = "requires the cached 0.5B model and a Metal device"]
fn metal_q8_0_kv_answers_like_f32_on_a_real_model() {
    use crate::graph::cache::GraphCache;
    use crate::graph::offload::OffloadRequest;
    use crate::models::{Device, ModelDef};

    let _g = crate::metal::metal_test_lock();
    let _packed = PackedKvEnabled::on();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let Some(path) = cached_gate_model() else {
        eprintln!("no cached gate model; skipping the #310 real-model gate");
        return;
    };
    let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
    let f32_model = crate::models::load_model_configured(
        &gguf,
        "m310.f32.",
        OffloadRequest::Default,
        Some("f32"),
    )
    .expect("load the f32 engine");
    let q8_model = crate::models::load_model_configured(
        &gguf,
        "m310.q8.",
        OffloadRequest::Default,
        Some("q8_0"),
    )
    .expect("load the q8_0 engine");
    for (m, want, name) in [
        (&*f32_model, KvFormat::F32, "f32"),
        (&*q8_model, KvFormat::Q8_0, "q8_0"),
    ] {
        assert_eq!(m.kv_format(), want, "the {name} engine's resolved format");
        assert_eq!(
            m.device(),
            Device::Metal,
            "the {name} engine must run on Metal — a silent CPU fallback would leave the \
             packed Metal kernels untested"
        );
    }

    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("tokenizer load");
    let n_ctx = 256usize;
    let steps = 8usize;
    let ids = tok.encode("The capital of France is");
    let n = ids.len();
    let positions: Vec<usize> = (0..n).collect();

    let run = |model: &dyn ModelDef| -> (Vec<Vec<f32>>, usize) {
        let mut cache = GraphCache::new();
        cache.alloc().kv_set_capacity(n_ctx);
        let mut l = model.forward_graph_cached(&ids, &positions, 1, n_ctx, &mut cache);
        let mut out = vec![l.clone()];
        let mut next = argmax(&l);
        for s in 0..steps {
            l = model.forward_graph_cached(&[next], &[n + s], 1, n_ctx, &mut cache);
            out.push(l.clone());
            next = argmax(&l);
        }
        let bytes = cache.alloc().kv_region_bytes();
        let _ = next;
        (out, bytes)
    };

    let (l_f32, b_f32) = run(f32_model.as_ref());
    let (l_q8, b_q8) = run(q8_model.as_ref());
    let worst = l_f32
        .iter()
        .zip(&l_q8)
        .map(|(a, b)| max_delta(a, b))
        .fold(0.0f32, f32::max);
    let at_argmax = l_f32
        .iter()
        .zip(&l_q8)
        .map(|(a, b)| {
            let i = argmax(a) as usize;
            (a[i] - b[i]).abs()
        })
        .fold(0.0f32, f32::max);
    let spread = l_f32
        .iter()
        .flatten()
        .fold(f32::NEG_INFINITY, |m, x| m.max(*x))
        - l_f32.iter().flatten().fold(f32::INFINITY, |m, x| m.min(*x));
    eprintln!(
        "[310 real] {}: KV regions f32 {b_f32} B vs q8_0 {b_q8} B ({:.2}x smaller); \
         max |Δlogit| = {worst} of a {spread} spread; at the argmax {at_argmax}",
        path.file_name().unwrap().to_string_lossy(),
        b_f32 as f64 / b_q8 as f64
    );
    assert!(
        b_q8 * 3 <= b_f32,
        "packed regions must be at least 3x smaller: {b_q8} vs {b_f32}"
    );
    assert!(
        at_argmax <= 1.0,
        "packed vs f32 at the argmax: |Δ| = {at_argmax}"
    );
    assert!(
        worst <= 4.0,
        "packed vs f32 logits: max |Δ| = {worst} of a {spread} spread"
    );
}

/// C5 `FLAG_PACKED` on Metal: a packed arena saves and restores **bitwise**, and
/// a fresh F32 allocator refuses the file (so the header really carries the Q8_0
/// element type). This drives the container's save/load over the Metal pool.
#[test]
fn metal_packed_session_round_trips() {
    let _g = crate::metal::metal_test_lock();
    let _packed = PackedKvEnabled::on();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }

    const ROWS: usize = 4;
    let (g, out) = make_causal_graph(ROWS);
    let q = q_data(ROWS);
    let k = kv_data(11, ROWS);
    let v = kv_data(23, ROWS);
    let cells: Vec<u32> = (0..ROWS as u32).collect();
    let positions: Vec<u32> = (0..ROWS as u32).collect();

    let mut a = GraphAllocator::new();
    assert!(a.enable_metal(), "MPS unavailable");
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    a.set_kv_format(KvFormat::Q8_0);
    a.kv_set_capacity(N_CTX);
    a.alloc_graph(&g2).unwrap();
    a.fill_input(&g2, "q", &q).unwrap();
    a.fill_input(&g2, "k", &k).unwrap();
    a.fill_input(&g2, "v", &v).unwrap();
    a.fill_input_i32(&g2, "cells", &cells).unwrap();
    a.fill_input_i32(&g2, "positions", &positions).unwrap();
    let span = span_window(&(0..ROWS as u32).map(|t| (0, t + 1)).collect::<Vec<_>>());
    a.fill_input_i32(&g2, "attn_span", &span).unwrap();
    BackendScheduler::new().execute(&g2, &mut a).unwrap();
    let _ = a.copy_to_cpu(out).unwrap();
    let want = a.copy_kv_to_cpu(0).unwrap();

    let path = std::env::temp_dir().join(format!(
        "minfer-310-metal-packed-{}.bin",
        std::process::id()
    ));
    a.kv_save_with_host(&path, b"host-state").unwrap();

    // The header carries Q8_0: a fresh F32 allocator refuses it.
    let mut f32_alloc = GraphAllocator::new();
    assert!(f32_alloc.enable_metal());
    f32_alloc.set_kv_format(KvFormat::F32);
    let expect_f32 = KvSessionExpect {
        backend: Tag::METAL,
        n_ctx: N_CTX,
        n_embd: NKT,
    };
    let err = f32_alloc.kv_load_with_host(&path, &expect_f32).unwrap_err();
    assert!(err.contains("element type"), "{err}");

    // A fresh Q8_0 allocator restores bitwise, host state and all.
    let mut b = GraphAllocator::new();
    assert!(b.enable_metal());
    let gb = {
        let mut g3 = g.clone();
        for n in &mut g3.nodes {
            n.backend = Some(Tag::METAL);
        }
        g3
    };
    b.set_kv_format(KvFormat::Q8_0);
    b.kv_set_capacity(N_CTX);
    b.alloc_graph(&gb).unwrap();
    let expect = KvSessionExpect {
        backend: Tag::METAL,
        n_ctx: N_CTX,
        n_embd: NKT,
    };
    let (host, report) = b.kv_load_with_host(&path, &expect).unwrap();
    assert_eq!(host, b"host-state");
    assert!(report.bytes > 0);
    let (got_k, got_v) = b.copy_kv_to_cpu(0).unwrap();
    // A packed region's pool words are raw Q8_0 bytes, so an `f32` `==` would
    // trip on a NaN bit pattern; compare the words bit for bit.
    for (region, got, want) in [("K", &got_k, &want.0), ("V", &got_v, &want.1)] {
        assert_eq!(got.len(), want.len(), "{region} region length changed");
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "{region} word {i}: the restored packed arena must be bitwise identical"
            );
        }
    }
    std::fs::remove_file(&path).ok();
}
