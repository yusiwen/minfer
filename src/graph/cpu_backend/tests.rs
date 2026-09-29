//! `#[cfg(test)] mod tests` for `src/graph/cpu_backend.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::batch::Batch;

/// C8b S2: a window given as **several runs** gathers exactly what the same cells
/// given as one range gather — the equivalence the map path rests on — and the
/// runs are visited in the order the store lists them.
#[test]
fn a_window_split_into_runs_gathers_like_one_range() {
    let (nh, nk, hd, nkt, n_ctx, nt) = (2usize, 1usize, 4usize, 4usize, 6usize, 2usize);
    let k: Vec<f32> = (0..n_ctx * nkt).map(|i| i as f32 * 0.05 + 0.1).collect();
    let v: Vec<f32> = (0..n_ctx * nkt).map(|i| i as f32 * -0.03 + 0.7).collect();
    let q: Vec<f32> = (0..nt * nh * hd).map(|i| i as f32 * 0.11 - 0.4).collect();
    let mut one = vec![0.0f32; nt * nh * hd];
    let mut split = vec![0.0f32; nt * nh * hd];
    // Query 0 sees cells [0, 3); query 1 sees [1, 6).
    let span = vec![(0usize, 3usize), (1, 6)];
    cpu_gqa_attn(&q, &k, &v, &span, nt, nh, nk, hd, hd, nkt, &mut one, 0.5).unwrap();
    // The same two windows as runs: [0, 2) + [2, 1), and [1, 3) + [4, 2).
    let runs = vec![(0usize, 2usize), (2, 1), (1, 3), (4, 2)];
    let off = vec![0usize, 2, 4];
    cpu_gqa_attn_runs(
        &q, &k, &v, &runs, &off, nt, nh, nk, hd, hd, nkt, &mut split, 0.5,
    )
    .unwrap();
    assert_eq!(one, split, "the window is the union of its runs, in order");
    // The `kv_map` layout decodes back to exactly those runs, and a padded slot
    // (length 0) is skipped.
    let k_max = crate::graph::kvcache::KV_MAP_MAX_SPANS;
    let mut flat = vec![0.0f32; nt * k_max * 2];
    for (i, &(cell, len)) in runs.iter().enumerate() {
        let at = (i / 2 * k_max + i % 2) * 2;
        flat[at] = f32::from_bits(cell as u32);
        flat[at + 1] = f32::from_bits(len as u32);
    }
    let (decoded, offsets) = decode_window(&flat, nt, n_ctx).unwrap();
    assert_eq!((decoded, offsets), (runs, off));
    // A layout that is neither form is refused, not guessed.
    let err = decode_window(&flat[1..], nt, n_ctx).unwrap_err();
    assert!(err.contains("expected"), "got: {err}");
}

/// Plan §14 row 9's exoneration of the ops, as a gate.
///
/// The minimal graph — `q`/`k`/`v` inputs → rope → store → attn, no model —
/// is run at cell 0, 1 and 8 with the same data, the same relative window and
/// the explicit span in every run, swept over shapes including the model's own
/// `(nh = 14, nk = 2, hd = 64)`. The only difference between the runs is the
/// rotation's own rounding, so the outputs must agree to well below anything a
/// model could amplify: measured <= 1.2e-7 at the widest shape, asserted < 1e-6.
///
/// This is what rules the kernels out as the source of the model-level offset
/// divergence (§14 row 9 finds it entering at layer 0's attention output and
/// still open): the same op, shape and window structure are exact in isolation.
#[test]
fn the_minimal_attention_graph_is_offset_invariant_to_rounding() {
    use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
    use crate::graph::DType;
    use crate::vec_ops::RopeStyle;

    let n = 5usize;
    let n_ctx = 64usize;
    let freq_base = 10_000.0f32;
    let freq_scale = 1.0f32;

    for (nh, nk, hd) in [
        (2usize, 2usize, 4usize),
        (2, 2, 64),
        (14, 2, 4),
        (14, 2, 64),
        (14, 2, 128),
    ] {
        let nkt = nk * hd;
        let mut b = GraphBuilder::new();
        b.set_explicit_span(true);
        let pos = b.input("positions", [n, 1, 1, 1], DType::I32);
        let qq = b.input("q", [nh * hd, n, 1, 1], DType::F32);
        let kk = b.input("k", [nkt, n, 1, 1], DType::F32);
        let vv = b.input("v", [nkt, n, 1, 1], DType::F32);
        let rope = |nh_: usize| RoPEMeta {
            freq_base,
            freq_scale,
            n_head: nh_,
            hd,
        };
        let q_r = b.rope(qq, pos, RopeStyle::NonInterleaved, rope(nh));
        let k_r = b.rope(kk, pos, RopeStyle::NonInterleaved, rope(nk));
        let _store = b.kvcache_store(0, k_r, vv, n_ctx);
        let kv = b.kvcache_load(0, nkt, n_ctx, nk);
        let at = b.attn(
            q_r,
            kv,
            pos,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk,
                hd,
                hd_kv: hd,
                nkt,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        b.output(at);
        let g = b.build();

        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((seed >> 8) as f32 / 8_388_608.0) - 1.0
        };
        let q: Vec<f32> = (0..nh * hd * n).map(|_| next()).collect();
        let k: Vec<f32> = (0..nkt * n).map(|_| next()).collect();
        let v: Vec<f32> = (0..nkt * n).map(|_| next()).collect();

        let run = |start: usize| -> Vec<f32> {
            let mut alloc = GraphAllocator::new();
            alloc.kv_set_capacity(n_ctx);
            alloc.alloc_graph(&g).expect("alloc");
            alloc.fill_input(&g, "q", &q).unwrap();
            alloc.fill_input(&g, "k", &k).unwrap();
            alloc.fill_input(&g, "v", &v).unwrap();
            // C6: `positions` are sequence-relative (the token's index), the
            // KV rows are the resolved `cells`, and the span stays a cell
            // range. The rotation therefore sees the same angles at every
            // `start`, which is what makes the comparison bitwise.
            let positions: Vec<u32> = (0..n).map(|p| p as u32).collect();
            let cells: Vec<u32> = (start..start + n).map(|p| p as u32).collect();
            let mut span = vec![start as u32; n];
            span.extend((0..n).map(|t| (start + t + 1) as u32));
            alloc.fill_input_i32(&g, "positions", &positions).unwrap();
            alloc.fill_input_i32(&g, "cells", &cells).unwrap();
            alloc.fill_input_i32(&g, "attn_span", &span).unwrap();
            let mut sched = crate::graph::scheduler::BackendScheduler::new();
            sched.execute(&g, &mut alloc).expect("execute");
            alloc.copy_to_cpu(at).expect("read")
        };
        let d = |a: &[f32], b: &[f32]| -> f32 {
            a.iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max)
        };
        let a0 = run(0);
        let a1 = run(1);
        let a8 = run(8);
        let (near, far) = (d(&a0, &a1), d(&a0, &a8));
        eprintln!("[minimal] nh={nh} nk={nk} hd={hd} n={n}: cell0-vs-1 {near} | cell0-vs-8 {far}");
        // C6: with relative positions a cell placement changes nothing at
        // all — the rows are written verbatim and read through the same
        // relative window, so this is bitwise, not a tolerance class. (Before
        // C6 the same comparison differed by the rotation's own rounding,
        // measured <= 1.2e-7; plan §14 row 9.)
        assert_eq!(
            (near, far),
            (0.0, 0.0),
            "a cell placement must not change the attention output"
        );
    }
}

#[test]
fn overlapping_rows_move_safely_in_both_directions() {
    let mut b = CpuBackend::new();
    let id = b.alloc_buffer(16);
    b.write_host(id, &(0..16).map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();
    let r = BufRef::own(crate::graph::Backend::CPU, id, 16);
    // Rows of 4 elements: row 0 <- row 1, overlapping by design.
    b.copy_cells(r, r, 0, 1, 2, 4).unwrap();
    assert_eq!(
        b.read_host(id).unwrap(),
        &[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0]
    );
    // C7b: the upward direction is supported as well — `copy_within` is a
    // memmove — and an overlapping upward move must land exactly where a copy
    // through a temporary would (row 1 <- row 0, overlapping).
    b.write_host(id, &(0..16).map(|i| i as f32).collect::<Vec<_>>())
        .unwrap();
    b.copy_cells(r, r, 1, 0, 2, 4).unwrap();
    assert_eq!(
        b.read_host(id).unwrap(),
        &[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0]
    );
    // A range leaving the buffer is an error, never a truncation.
    assert!(b.copy_cells(r, r, 0, 3, 4, 4).is_err());
    assert!(b.copy_cells(r, r, 0, 0, 1, 32).is_err());
}
use crate::graph::alloc::GraphAllocator;
use crate::graph::builder::GraphBuilder;
use crate::graph::scheduler::BackendScheduler;
use crate::graph::{DType, NodeId};

fn tensor_f32(name: &str, shape: [i64; 4], data: Vec<f32>) -> Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let mut t = Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
    t.name = name.to_string();
    t
}

struct Harness {
    sched: BackendScheduler,
    alloc: GraphAllocator,
}

impl Harness {
    fn new() -> Self {
        Self {
            sched: BackendScheduler::new(),
            alloc: GraphAllocator::new(),
        }
    }
    fn reg(&mut self, t: Tensor) {
        let name = t.name.clone();
        self.alloc.register_weight(&name, t);
    }
    fn run(&mut self, graph: &crate::graph::ComputeGraph, fills: &[(&str, Vec<f32>)]) {
        self.alloc.alloc_graph(graph).unwrap();
        for (name, data) in fills {
            self.alloc.fill_input(graph, name, data).unwrap();
        }
        self.sched.execute(graph, &mut self.alloc).unwrap();
    }
    fn out(&self, graph: &crate::graph::ComputeGraph, id: NodeId) -> Vec<f32> {
        self.alloc.get_buffer(graph, id).unwrap().to_vec()
    }
}

#[test]
// 8a② regression: an F32-weight matmul with nt > 1 must produce
// token-major output [nt][od]. The old mat_mul_f32 wrote [od][nt] —
// only visible for nt > 1 (decode nt==1 was accidentally correct).
#[test]
fn f32_matmul_nt2_token_major() {
    let mut h = Harness::new();
    // W [in=4, out=3]: row o selects (o+1) * x[o]
    h.reg(tensor_f32(
        "W",
        [4, 3, 1, 1],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
    ));
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [4, 2, 1, 1], DType::F32);
    let m = gb.matmul(x, h.alloc.cpu().weight("W").unwrap(), None);
    gb.output(m);
    let g = gb.build();
    let xdata = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, -0.25, 2.0];
    h.run(&g, &[("x", xdata)]);
    let got = h.out(&g, m);
    // token 0: [1, 4, 9]; token 1: [-1, 1, -0.75]
    let expect = [1.0, 4.0, 9.0, -1.0, 1.0, -0.75];
    for i in 0..6 {
        assert!(
            (got[i] - expect[i]).abs() < 1e-5,
            "out[{i}]={} expect {}",
            got[i],
            expect[i]
        );
    }
}

fn matmul_add_silu_scale() {
    // x [4,1] * W [3,4] + b [3] -> [3,1]; silu; *2
    let mut h = Harness::new();
    // weight metadata [in=4, out=3]; memory = 3 rows of 4 (out-major)
    h.reg(tensor_f32(
        "W",
        [4, 3, 1, 1],
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
    ));
    h.reg(tensor_f32("b", [3, 1, 1, 1], vec![0.5, -1.0, 0.25]));

    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [4, 1, 1, 1], DType::F32);
    let m = gb.matmul(
        x,
        h.alloc.cpu().weight("W").unwrap(),
        Some(h.alloc.cpu().weight("b").unwrap()),
    );
    let s = gb.silu(m);
    let o = gb.node(
        "scale2",
        Op::Scale(2.0),
        &[s],
        [3, 1, 1, 1],
        DType::F32,
        NodeMeta::None,
    );
    gb.output(o);
    let g = gb.build();

    h.run(&g, &[("x", vec![1.0, 2.0, 3.0, 4.0])]);
    let silu = |v: f32| v / (1.0 + (-v).exp());
    let expect = [silu(1.5) * 2.0, silu(3.0) * 2.0, silu(9.25) * 2.0];
    let got = h.out(&g, o);
    for i in 0..3 {
        assert!(
            (got[i] - expect[i]).abs() < 1e-4,
            "out[{i}]={} expect {}",
            got[i],
            expect[i]
        );
    }
}

#[test]
fn rms_norm_matches_reference() {
    let mut h = Harness::new();
    h.reg(tensor_f32("nw", [4, 1, 1, 1], vec![1.0, 1.0, 1.0, 1.0]));
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [4, 2, 1, 1], DType::F32);
    let r = gb.rms_norm(x, Some(h.alloc.cpu().weight("nw").unwrap()), 1e-5);
    gb.output(r);
    let g = gb.build();
    let data = vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 2.0, -3.0];
    h.run(&g, &[("x", data.clone())]);
    let got = h.out(&g, r);
    // reference via vec_ops directly
    let mut ref_out = vec![0.0f32; 8];
    for t in 0..2 {
        crate::vec_ops::rms_norm_fused_f32(
            4,
            &mut ref_out[t * 4..(t + 1) * 4],
            &data[t * 4..(t + 1) * 4],
            &[1.0, 1.0, 1.0, 1.0],
            1e-5,
        );
    }
    for i in 0..8 {
        assert!(
            (got[i] - ref_out[i]).abs() < 1e-6,
            "norm[{i}] {} vs {}",
            got[i],
            ref_out[i]
        );
    }
}

#[test]
fn embedding_and_rope() {
    // vocab 4, n_embd 4: ids [0,2] -> rows, then rope per 2 heads of hd 2
    let mut h = Harness::new();
    h.reg(tensor_f32(
        "tok_embd",
        [4, 4, 1, 1],
        vec![
            0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4,
        ],
    ));
    let mut gb = GraphBuilder::new();
    let ids = gb.input("token_ids", [2, 1, 1, 1], DType::I32);
    let emb = gb.embedding(ids, h.alloc.cpu().weight("tok_embd").unwrap());
    let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
    let rope = gb.rope(
        emb,
        pos,
        RopeStyle::NonInterleaved,
        super::super::ops::RoPEMeta {
            freq_base: 10000.0,
            freq_scale: 1.0,
            n_head: 2,
            hd: 2,
        },
    );
    gb.output(rope);
    let g = gb.build();
    // ids 0,2 at positions 0,1 (I32 inputs via bit patterns)
    h.alloc.alloc_graph(&g).unwrap();
    h.alloc.fill_input_i32(&g, "token_ids", &[0, 2]).unwrap();
    h.alloc.fill_input_i32(&g, "positions", &[0, 1]).unwrap();
    // No KV store and no attention: this fixture's graph has no `cells`,
    // `seq_ids`, `kv_map` or `attn_span` input, so there is no window to resolve
    // and no arena to reserve. Production's `fill_batch_inputs` therefore cannot
    // drive it — its per-group reservation would ask for a 0-cell run — and the
    // rope-only shape is kept as the `#[cfg(test)]`-scoped
    // `GraphAllocator::fill_attn_inputs_without_cells` (the deleted E1 helper's
    // `has_cells == false` branch). The claim is unchanged: filling must not
    // require a KV arena for a graph that stores no K/V.
    h.alloc
        .fill_attn_inputs_without_cells(&g, &[0, 0], &[0, 1])
        .unwrap();
    h.sched.execute(&g, &mut h.alloc).unwrap();
    let got = h.out(&g, rope);
    // reference: embed rows then rope per head
    let mut ref_x = vec![
        0.1, 0.2, 0.3, 0.4, // id 0
        2.1, 2.2, 2.3, 2.4, // id 2
    ];
    cpu_rope(
        &mut ref_x,
        &[0, 1],
        2,
        2,
        10000.0,
        1.0,
        RopeStyle::NonInterleaved,
    );
    for i in 0..8 {
        assert!(
            (got[i] - ref_x[i]).abs() < 1e-5,
            "rope[{i}] {} vs {}",
            got[i],
            ref_x[i]
        );
    }
}

#[test]
fn kvcache_store_load_and_attn_roundtrip() {
    // one layer: q [hd=2, nh=2, nt=1] vs stored k/v at pos 0; GQA nh=2 nk=2
    let mut h = Harness::new();
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [1, 1, 1, 1], DType::I32);
    let q = gb.input("q", [4, 1, 1, 1], DType::F32);
    let k = gb.input("k", [4, 1, 1, 1], DType::F32);
    let v = gb.input("v", [4, 1, 1, 1], DType::F32);
    let _st = gb.kvcache_store(0, k, v, 8);
    let kv = gb.kvcache_load(0, 4, 8, 2);
    let out = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        super::super::ops::AttnMeta {
            layer: 0,
            n_head: 2,
            n_head_kv: 2,
            hd: 2,
            hd_kv: 2,
            nkt: 4,
            scale: 0.5,
        },
    );
    gb.output(out);
    let g = gb.build();

    // q = [1,0, 0,1], k = [1,0, 0,1], v = [0.5,0.5, 0.25,0.75] at pos 0
    h.alloc.alloc_graph(&g).unwrap();
    h.alloc.fill_input_i32(&g, "positions", &[0]).unwrap();
    h.alloc
        .fill_batch_inputs(&g, &Batch::single(&[0], &[0]))
        .unwrap();
    h.alloc.fill_input(&g, "q", &[1.0, 0.0, 0.0, 1.0]).unwrap();
    h.alloc.fill_input(&g, "k", &[1.0, 0.0, 0.0, 1.0]).unwrap();
    h.alloc
        .fill_input(&g, "v", &[0.5, 0.5, 0.25, 0.75])
        .unwrap();
    h.sched.execute(&g, &mut h.alloc).unwrap();
    let got = h.out(&g, out);
    // scores: h0: dot([1,0],[1,0])*0.5 = 0.5; h1: dot([0,1],[0,1])*0.5 = 0.5
    // softmax([0.5]) = 1.0 -> out = v
    assert!((got[0] - 0.5).abs() < 1e-5, "got[0]={}", got[0]);
    assert!((got[1] - 0.5).abs() < 1e-5, "got[1]={}", got[1]);
    assert!((got[2] - 0.25).abs() < 1e-5, "got[2]={}", got[2]);
    assert!((got[3] - 0.75).abs() < 1e-5, "got[3]={}", got[3]);
}

/// C4: a packed Q8_0 KV region answers like the f32 one and occupies about a
/// third of the memory. The store quantizes, the attention read dequantizes the
/// window it is about to use; three rows and three causal queries make the
/// softmax weights depend on the *quantized scores*, so a broken K read cannot
/// pass by returning V verbatim.
#[test]
fn a_packed_kv_region_answers_like_the_f32_one_and_is_smaller() {
    use super::super::kvformat::KvFormat;
    let nkt = 32usize; // Q8_0 quantizes in 32-element blocks
    let hd = 32usize;
    let nt = 3usize;
    let n_ctx = 8usize;
    let qv: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i as f32) * 0.13).sin() * 0.7)
        .collect();
    let kk: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i as f32) * 0.29).cos() * 1.3 + 0.04 * (i % 7) as f32)
        .collect();
    let vv: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5 - 0.02 * (i % 3) as f32)
        .collect();

    let run = |format: KvFormat| -> (Vec<f32>, usize, Vec<f32>) {
        let mut h = Harness::new();
        // Both halves of the decision, without touching the process-wide policy:
        // the builder stamps the cell width, the backend reads it.
        h.alloc.cpu_mut().set_kv_format(format);
        let mut gb = GraphBuilder::new();
        gb.set_kv_format(format);
        let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let q = gb.input("q", [nkt, nt, 1, 1], DType::F32);
        let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
        let store = gb.kvcache_store(0, k, v, n_ctx);
        let kv = gb.kvcache_load(0, nkt, n_ctx, 1);
        let out = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            super::super::ops::AttnMeta {
                layer: 0,
                n_head: 1,
                n_head_kv: 1,
                hd,
                hd_kv: hd,
                nkt,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        gb.output(out);
        let g = gb.build();
        h.alloc.alloc_graph(&g).unwrap();
        h.alloc.fill_input_i32(&g, "positions", &[0, 1, 2]).unwrap();
        h.alloc
            .fill_batch_inputs(&g, &Batch::single(&[0, 0, 0], &[0, 1, 2]))
            .unwrap();
        h.alloc.fill_input(&g, "q", &qv).unwrap();
        h.alloc.fill_input(&g, "k", &kk).unwrap();
        h.alloc.fill_input(&g, "v", &vv).unwrap();
        h.sched.execute(&g, &mut h.alloc).unwrap();
        // The store node's buffer *is* the K region, so this is what the
        // attention read dequantized.
        (h.out(&g, out), h.alloc.kv_region_bytes(), h.out(&g, store))
    };

    let (f32_out, f32_bytes, f32_region) = run(KvFormat::F32);
    let (q8_out, q8_bytes, q8_region) = run(KvFormat::Q8_0);
    assert_eq!(f32_out.len(), nkt * nt);
    assert_eq!(q8_out.len(), nkt * nt);
    // Footprint: 4 bytes/element against ceil(34/4) words per 32 elements.
    assert_eq!(f32_bytes, n_ctx * nkt * 4 * 2);
    assert_eq!(q8_bytes, n_ctx * 9 * 4 * 2);
    assert!(
        q8_bytes * 3 <= f32_bytes,
        "the packed region must be at least 3x smaller: {q8_bytes} vs {f32_bytes}"
    );
    // The stored cell must be exactly the Q8_0 quantizate of the input row —
    // bitwise, not merely close. This is what separates "the packed layout is
    // addressed correctly" from "the numbers happen to be near": the packed
    // store's bytes and an independent `quantize -> dequantize` of the same
    // row must agree to the last bit, and the f32 run's region must hold the
    // *un*quantized row.
    let row_zero = &kk[..nkt];
    let mut want = vec![0.0f32; nkt];
    let bytes = crate::quants::quantize_row_q8_0(row_zero);
    crate::quants::dequantize_row_q8_0(&bytes, &mut want);
    let mut got = vec![0.0f32; nkt];
    super::super::kvformat::unpack_q8_0_cells(&q8_region, nkt, 0, 1, &mut got);
    assert_eq!(got, want, "the packed cell is not the Q8_0 quantizate");
    assert_eq!(
        &f32_region[..nkt],
        row_zero,
        "the f32 region must hold the row verbatim"
    );
    // Tolerance class: the per-block step of Q8_0, softened by the softmax over
    // three rows (a wrong cell width or a byte-order slip is off by orders of
    // magnitude more, which is what the gate is for).
    let worst = (0..f32_out.len())
        .map(|i| (f32_out[i] - q8_out[i]).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst < 5e-2,
        "packed KV vs f32 KV: max |Δ| = {worst} over {} outputs",
        f32_out.len()
    );
}

/// C4 S2: the fused read addresses **each KV head's blocks** and agrees with the
/// S1 path it replaces.
///
/// The oracle here is S1's own mechanism — `unpack_q8_0_cells` →
/// `cpu_gqa_attn_runs` — so the only permitted difference is the *query's* Q8_0
/// quantization (the K score is now a `dot_q8_0_q8_0`). Two KV heads with very
/// different K rows make a wrong head base (`hk * hd`, the block offset the
/// fused path computes itself) fail by the whole spread instead of by an ulp.
#[test]
fn the_fused_q8_read_matches_the_dequantizing_reference() {
    use super::super::kvformat::{self, KvFormat};
    let (nh, nk, hd, nt, n_ctx) = (2usize, 2usize, 32usize, 3usize, 6usize);
    let nkt = nk * hd; // 64 → two heads, each exactly one Q8_0 block wide
    let scale = 1.0 / (hd as f32).sqrt();
    let kk: Vec<f32> = (0..n_ctx * nkt)
        .map(|i| {
            if (i % nkt) / hd == 0 {
                ((i as f32) * 0.31).sin() * 0.9
            } else {
                ((i as f32) * 0.17).cos() * 2.5 - 1.0
            }
        })
        .collect();
    let vv: Vec<f32> = (0..n_ctx * nkt)
        .map(|i| ((i as f32) * 0.11).sin() * 0.6 - (i % 5) as f32 * 0.03)
        .collect();
    let q: Vec<f32> = (0..nt * nh * hd)
        .map(|i| ((i as f32) * 0.23).cos() * 0.8)
        .collect();

    // Pack exactly as the store does: one cell per row, heads block-aligned.
    let row_elems = KvFormat::Q8_0.row_elems(nkt);
    let mut kreg = vec![0.0f32; n_ctx * row_elems];
    let mut vreg = vec![0.0f32; n_ctx * row_elems];
    for cell in 0..n_ctx {
        let w = cell * row_elems..(cell + 1) * row_elems;
        kvformat::pack_q8_0_cell(&mut kreg[w.clone()], nkt, &kk[cell * nkt..(cell + 1) * nkt]);
        kvformat::pack_q8_0_cell(&mut vreg[w], nkt, &vv[cell * nkt..(cell + 1) * nkt]);
    }

    // Causal windows, one run per query (`off[t]..off[t + 1]`).
    let runs: Vec<(usize, usize)> = (0..nt).map(|t| (0usize, t + 1)).collect();
    let off: Vec<usize> = (0..=nt).collect();

    let mut kf = vec![0.0f32; n_ctx * nkt];
    let mut vf = vec![0.0f32; n_ctx * nkt];
    kvformat::unpack_q8_0_cells(&kreg, nkt, 0, n_ctx, &mut kf);
    kvformat::unpack_q8_0_cells(&vreg, nkt, 0, n_ctx, &mut vf);
    let mut want = vec![0.0f32; nt * nh * hd];
    cpu_gqa_attn_runs(
        &q, &kf, &vf, &runs, &off, nt, nh, nk, hd, hd, nkt, &mut want, scale,
    )
    .unwrap();

    let mut got = vec![0.0f32; nt * nh * hd];
    cpu_gqa_attn_runs_q8(
        &q, &kreg, &vreg, &runs, &off, nt, nh, nk, hd, hd, nkt, &mut got, scale,
    )
    .unwrap();

    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let spread = want.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x))
        - want.iter().fold(f32::INFINITY, |m, x| m.min(*x));
    eprintln!("[c4s2] fused vs dequantizing reference: max |Δ| = {worst} of a {spread} spread");
    assert!(
        worst < 1e-3,
        "the fused read may only differ by the query's Q8_0 quantization: max |Δ| = \
         {worst} of a {spread} spread"
    );
}

/// C4 S2: a physical shift on a **packed** region moves every surviving cell
/// verbatim (V is never re-roped, so it is bitwise), and re-ropes the survivors'
/// K through dequantize → rope → requantize.
///
/// The gate is exact on both sides: V must equal the old cells byte-for-byte, and
/// the new K cell must be the Q8_0 quantizate of the re-roped f32 row — computed
/// here independently, so a shift that forgets the requantize (or re-ropes the
/// packed bytes) cannot pass.
#[test]
fn a_packed_physical_shift_moves_v_verbatim_and_requantizes_k() {
    use super::super::kvformat::{self, KvFormat};
    use crate::graph::kvcache::{rope_shift_kv, KvRope};
    use crate::vec_ops::RopeStyle;

    let nkt = 32usize;
    let hd = 32usize;
    let nt = 3usize;
    let n_ctx = 8usize;
    let kk: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i as f32) * 0.29).cos() * 1.3 + 0.04 * (i % 7) as f32)
        .collect();
    let vv: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i as f32) * 0.07).sin() * 0.5 - 0.02 * (i % 3) as f32)
        .collect();

    let mut h = Harness::new();
    h.alloc.cpu_mut().set_kv_format(KvFormat::Q8_0);
    let mut gb = GraphBuilder::new();
    gb.set_kv_format(KvFormat::Q8_0);
    let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nkt, nt, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
    let store = gb.kvcache_store(0, k, v, n_ctx);
    gb.output(store);
    let g = gb.build();
    h.alloc.alloc_graph(&g).unwrap();
    h.alloc.fill_input_i32(&g, "positions", &[0, 1, 2]).unwrap();
    // `kvcache_store` consumes the builder's own `cells` input (C6): filling
    // `positions` alone left every row going to cell 0, which made this gate pass
    // on a region with one live row — exactly the kind of false green the nonzero
    // assertions below exist to prevent. E2 is what resolves them now (and records
    // the written extent through `own_positions`, so the old `kv_note_used`
    // post-execute call is gone with it).
    h.alloc
        .fill_batch_inputs(&g, &Batch::single(&[0, 0, 0], &[0, 1, 2]))
        .unwrap();
    h.alloc.fill_input(&g, "k", &kk).unwrap();
    h.alloc.fill_input(&g, "v", &vv).unwrap();
    h.sched.execute(&g, &mut h.alloc).unwrap();

    let rope = KvRope {
        freq_base: 10_000.0,
        freq_scale: 1.0,
        n_head_kv: 1,
        hd,
        style: RopeStyle::NonInterleaved,
    };
    let row_elems = KvFormat::Q8_0.row_elems(nkt);
    let (before_k_words, before_v_words) = h.alloc.copy_kv_to_cpu(0).unwrap();
    let mut before_k = vec![0.0f32; nt * nkt];
    let mut before_v = vec![0.0f32; nt * nkt];
    kvformat::unpack_q8_0_cells(&before_k_words, nkt, 0, nt, &mut before_k);
    kvformat::unpack_q8_0_cells(&before_v_words, nkt, 0, nt, &mut before_v);

    // The fixture must be real: three written rows, each with data, or every
    // assertion below would pass on zeros.
    for r in 0..nt {
        assert!(
            before_k[r * nkt..(r + 1) * nkt].iter().any(|x| *x != 0.0)
                && before_v[r * nkt..(r + 1) * nkt].iter().any(|x| *x != 0.0),
            "row {r} was not written (the store did not get its cells input)"
        );
    }
    // Drop the oldest row: cells 1..3 slide to 0..2 and are re-roped by -1.
    let left = h.alloc.kv_rm(0, 1, &rope).unwrap();
    assert_eq!(left, nt - 1, "one row removed");
    let (after_k_words, after_v_words) = h.alloc.copy_kv_to_cpu(0).unwrap();
    let mut after_k = vec![0.0f32; nt * nkt];
    let mut after_v = vec![0.0f32; nt * nkt];
    kvformat::unpack_q8_0_cells(&after_k_words, nkt, 0, left, &mut after_k);
    kvformat::unpack_q8_0_cells(&after_v_words, nkt, 0, left, &mut after_v);

    // V: the cells moved verbatim, so the dequantized values are bitwise equal.
    assert_eq!(
        &after_v[..left * nkt],
        &before_v[nkt..(left + 1) * nkt],
        "V must move verbatim under a packed shift"
    );
    // The packed words themselves must be the old cells' words: a cell is a whole
    // number of words, which is exactly why the move needs no format knowledge.
    // Compared as **bits**: a packed word is an f16 scale plus int8 quants, so as an
    // f32 it is frequently a NaN, and `==` on it is never true.
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&after_v_words[..left * row_elems]),
        bits(&before_v_words[row_elems..(left + 1) * row_elems]),
        "the V cells must move verbatim, byte for byte"
    );
    // K: exactly the Q8_0 quantizate of the old row, re-roped in f32.
    for r in 0..left {
        let mut want = before_k[(r + 1) * nkt..(r + 2) * nkt].to_vec();
        rope_shift_kv(&mut want, 1, 1, &rope);
        let bytes = crate::quants::quantize_row_q8_0(&want);
        let mut quantized = vec![0.0f32; nkt];
        crate::quants::dequantize_row_q8_0(&bytes, &mut quantized);
        let got = &after_k[r * nkt..(r + 1) * nkt];
        assert_eq!(
            got,
            &quantized[..],
            "row {r}: the shifted K must be the Q8_0 quantizate of the re-roped row"
        );
        // And the honest approximation class: the re-rope of a *quantized* row
        // differs from the quantum step of the stored cell.
        let step = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            step < 0.2,
            "row {r}: re-rope vs the f32 re-rope differs by {step}, more than one Q8_0 step"
        );
    }
    // The tail must be cleared: no position may address a stale row.
    for cell in left..n_ctx {
        assert!(
            after_k_words[cell * row_elems..(cell + 1) * row_elems]
                .iter()
                .all(|x| *x == 0.0)
                && after_v_words[cell * row_elems..(cell + 1) * row_elems]
                    .iter()
                    .all(|x| *x == 0.0),
            "cell {cell} must be zeroed after the shift"
        );
    }
}

/// C4: a row width Q8_0 cannot express is refused where the region is sized,
/// not truncated into a layout that would mis-address every cell.
#[test]
fn a_packed_region_refuses_a_width_q8_0_cannot_express() {
    use super::super::kvformat::KvFormat;
    let mut h = Harness::new();
    h.alloc.cpu_mut().set_kv_format(KvFormat::Q8_0);
    let mut gb = GraphBuilder::new();
    gb.set_kv_format(KvFormat::Q8_0);
    let pos = gb.input("positions", [1, 1, 1, 1], DType::I32);
    let q = gb.input("q", [4, 1, 1, 1], DType::F32);
    let k = gb.input("k", [4, 1, 1, 1], DType::F32);
    let v = gb.input("v", [4, 1, 1, 1], DType::F32);
    let _store = gb.kvcache_store(0, k, v, 8);
    let kv = gb.kvcache_load(0, 4, 8, 2);
    let out = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        super::super::ops::AttnMeta {
            layer: 0,
            n_head: 2,
            n_head_kv: 2,
            hd: 2,
            hd_kv: 2,
            nkt: 4,
            scale: 0.5,
        },
    );
    gb.output(out);
    let g = gb.build();
    let err = h.alloc.alloc_graph(&g).unwrap_err();
    assert!(err.contains("multiple of 32"), "{err}");
}

/// C4 regression: `ensure_kv` takes a **logical** cell width and a **stored**
/// one, and the D3-8 mixed-quant epilogue — which also stores K/V — must hand in
/// the same `n_kv_embd` the store node does. Passing `n_ctx` in its place made
/// `packed` compare two unrelated numbers, so a *CPU* graph with this node failed
/// with "the node declares N words per cell but the Q8_0 layout packs ..."; no
/// cached model here builds the epilogue (their q/k/v share a quant type), which
/// is why only this hand-built graph covers the line.
#[test]
fn the_qkv_epilogue_sizes_its_kv_region_by_n_kv_embd() {
    let (nkt, nqt, n_ctx) = (4usize, 8usize, 16usize);
    let mut h = Harness::new();
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [1, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nqt, 1, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, 1, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, 1, 1, 1], DType::F32);
    let ep = gb.qkv_bias_rope_store(
        q,
        k,
        v,
        pos,
        0,
        crate::graph::ops::QkvBiasRopeStoreMeta {
            bias_q: None,
            bias_k: None,
            bias_v: None,
            nqt,
            nkt,
            hd: 4,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            rope_style: crate::vec_ops::RopeStyle::NonInterleaved,
            kv_elems: nkt * n_ctx,
            row_elems: crate::graph::kvformat::KvFormat::F32.row_elems(nkt),
        },
    );
    let kv = gb.kvcache_load(0, nkt, n_ctx, 1);
    let out = gb.attn(
        ep,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        super::super::ops::AttnMeta {
            layer: 0,
            n_head: 1,
            n_head_kv: 1,
            hd: nkt,
            hd_kv: nkt,
            nkt,
            scale: 1.0,
        },
    );
    gb.output(out);
    let g = gb.build();
    h.alloc.alloc_graph(&g).unwrap();
    // Unpacked: one f32 word per element, K and V.
    assert_eq!(h.alloc.kv_region_bytes(), n_ctx * nkt * 4 * 2);
    assert!(!h.alloc.kv_is_packed());
}

/// E1's acceptance: two sequences sharing one KV arena must not see each
/// other. The windows are the input, so the test supplies them directly —
/// the resolver that produces them is covered by `graph::kvcache`'s tests.
///
/// The values are chosen so a leak *changes the answer*: query 1 would score
/// 1.0 against sequence 0's key if its window wrongly reached cell 0, which
/// would blend V(0) into the output instead of returning V(2).
#[test]
fn two_sequences_do_not_cross_attend() {
    let mut h = Harness::new();
    let mut gb = GraphBuilder::new();
    gb.set_explicit_span(true);
    // k/v are [nkt, nt] so the store node sizes the region as nkt * n_ctx.
    let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
    let q = gb.input("q", [2, 2, 1, 1], DType::F32);
    let k = gb.input("k", [2, 2, 1, 1], DType::F32);
    let v = gb.input("v", [2, 2, 1, 1], DType::F32);
    let _st = gb.kvcache_store(0, k, v, 4);
    let kv = gb.kvcache_load(0, 2, 4, 1);
    let out = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        super::super::ops::AttnMeta {
            layer: 0,
            n_head: 1,
            n_head_kv: 1,
            hd: 2,
            hd_kv: 2,
            nkt: 2,
            scale: 1.0,
        },
    );
    gb.output(out);
    let g = gb.build();
    // The graph must carry the multi-sequence flag: it is what stops a
    // backend that still derives its bound from positions from taking it.
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

    h.alloc.alloc_graph(&g).unwrap();
    // Sequence 0 owns row 0, sequence 1 owns row 2 (rows 1 and 3 stay free).
    // C6: `positions` are each token's index within its sequence, while the
    // store's rows come from the resolved `cells`.
    h.alloc.fill_input_i32(&g, "positions", &[0, 0]).unwrap();
    h.alloc.fill_input_i32(&g, "seq_ids", &[0, 1]).unwrap();
    h.alloc.fill_input_i32(&g, "cells", &[0, 2]).unwrap();
    // The span layout is the `lo` block then the `hi` block: token 0 gets
    // `[0, 1)` (sequence 0's only row) and token 1 `[2, 3)` (sequence 1's).
    h.alloc
        .fill_input_i32(&g, "attn_span", &[0, 2, 1, 3])
        .unwrap();
    // token 0 = [1, 0], token 1 = [1, 0]
    h.alloc.fill_input(&g, "q", &[1.0, 0.0, 1.0, 0.0]).unwrap();
    // row 0 = k [1, 0] / v [1, 0]; row 2 = k [0, 1] / v [0, 1]
    h.alloc.fill_input(&g, "k", &[1.0, 0.0, 0.0, 1.0]).unwrap();
    h.alloc.fill_input(&g, "v", &[1.0, 0.0, 0.0, 1.0]).unwrap();
    h.sched.execute(&g, &mut h.alloc).unwrap();
    let got = h.out(&g, out);
    assert_eq!(got.len(), 4);
    // Token 0 attends to its own row only -> V(0).
    assert!(
        (got[0] - 1.0).abs() < 1e-6 && got[1].abs() < 1e-6,
        "token 0: {got:?}"
    );
    // Token 1 attends to its own row only -> V(2). A cross-sequence leak
    // would show up here as roughly [0.73, 0.27].
    assert!(
        got[2].abs() < 1e-6 && (got[3] - 1.0).abs() < 1e-6,
        "token 1 saw the other sequence: {got:?}"
    );
}

/// Generic get_rows (n_out tail selection): out[t] = x[ids[t]].
#[test]
fn cpu_generic_get_rows() {
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [4, 3, 1, 1], DType::F32);
    let ids = gb.input("ids", [1, 1, 1, 1], DType::I32);
    let r = gb.get_rows(x, ids, [4, 1, 1, 1]);
    gb.output(r);
    let g = gb.build();

    let mut sched = BackendScheduler::new();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    // rows: r0=[1,2,3,4] r1=[10,20,30,40] r2=[100,200,300,400]
    alloc
        .fill_input(
            &g,
            "x",
            &[
                1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
            ],
        )
        .unwrap();
    alloc.fill_input_i32(&g, "ids", &[2]).unwrap();
    sched.execute(&g, &mut alloc).unwrap();
    let got = alloc.get_buffer(&g, r).unwrap();
    assert_eq!(
        got,
        &[100.0, 200.0, 300.0, 400.0],
        "get_rows should select row 2"
    );

    // ids = [0]
    alloc.fill_input_i32(&g, "ids", &[0]).unwrap();
    sched.execute(&g, &mut alloc).unwrap();
    let got = alloc.get_buffer(&g, r).unwrap();
    assert_eq!(got, &[1.0, 2.0, 3.0, 4.0], "get_rows should select row 0");
}
