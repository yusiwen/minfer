//! `#[cfg(test)] mod tests` for `src/graph/metal_backend.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::alloc::GraphAllocator;
use crate::graph::backend::Backend;
use crate::graph::batch::Batch;
use crate::graph::builder::GraphBuilder;
use crate::graph::scheduler::BackendScheduler;
use crate::graph::{Backend as Tag, ComputeGraph, DType};

mod attn_span;
mod copy_cells;
mod fusion_shape;
mod norm_weight;
mod staging;
fn f32t(name: &str, shape: [i64; 4], data: Vec<f32>) -> crate::tensor::Tensor {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for x in data {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    let mut t = crate::tensor::Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
    t.name = name.to_string();
    t
}

/// Fill a single-sequence graph's inputs the way **production** does: the
/// sequence-relative `positions`, then `GraphAllocator::fill_batch_inputs`
/// (`Batch::single`), which resolves `seq_ids`, `cells` and `attn_span` from the
/// cell store (E1; the migration [#228] put the other macOS-only call sites on).
///
/// The fixtures below build through `GraphBuilder::attn`, whose `attn_span`
/// window input is explicit, so filling only `positions` leaves the window at
/// its zero initialisation and the CPU reference arm refuses
/// (`decode_window`). A hand-rolled span here would re-create exactly the drift
/// [#228] removed, so this goes through the production entry point.
///
/// `tokens` is unused by the fill (these graphs have no token input), but
/// `Batch::single` requires one row per query.
fn fill_seq_inputs(alloc: &mut GraphAllocator, g: &ComputeGraph, positions: &[u32]) {
    alloc.fill_input_i32(g, "positions", positions).unwrap();
    let tokens = vec![0u32; positions.len()];
    let rel: Vec<usize> = positions.iter().map(|&p| p as usize).collect();
    alloc
        .fill_batch_inputs(g, &Batch::single(&tokens, &rel))
        .unwrap();
}

/// GPU graph (silu + add) must match the CPU graph bit-for-bit.
#[test]
fn metal_elementwise_matches_cpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(backend) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [4, 1, 1, 1], DType::F32);
    let s = gb.silu(x);
    let o = gb.add(s, x);
    gb.output(o);
    let g = gb.build();

    // CPU run
    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    ca.fill_input(&g, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
    sched.assign_backends(&mut g.clone(), &ca);
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();

    // Metal run (assign all to Metal)
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
    let splits = sched.split_graph(&g2);
    for (si, sp) in splits.iter().enumerate() {
        eprintln!(
            "[dbg] split {si}: {:?} range {:?} inputs {:?}",
            sp.backend, sp.node_range, sp.inputs
        );
    }
    sched.execute(&g2, &mut alloc).unwrap();
    eprintln!(
        "[dbg] silu out (node 1) = {:?}",
        alloc.copy_to_cpu(1).unwrap()
    );
    eprintln!(
        "[dbg] add out (node 2) = {:?}",
        alloc.copy_to_cpu(2).unwrap()
    );
    let got = alloc.copy_to_cpu(o).unwrap();
    for i in 0..4 {
        assert!(
            (got[i] - expect[i]).abs() < 1e-6,
            "out[{i}] {} vs {}",
            got[i],
            expect[i]
        );
    }
    let _ = backend;
}

/// rms_norm on Metal must match CPU within float tolerance.
#[test]
fn metal_rmsnorm_matches_cpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    // register a norm weight on MPS (name must resolve in weight_buf)
    let wdata: Vec<f32> = (0..8).map(|i| 0.5 + i as f32 * 0.1).collect();
    let mut bytes = Vec::new();
    for x in &wdata {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("nw", &bytes);

    let nw = f32t("nw", [8, 1, 1, 1], wdata);
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 3, 1, 1], DType::F32);
    let r = gb.rms_norm(x, Some(&nw), 1e-5);
    gb.output(r);
    let g = gb.build();

    // CPU
    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.register_weight("nw", nw);
    ca.alloc_graph(&g).unwrap();
    let data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.3).collect();
    ca.fill_input(&g, "x", &data).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, r).unwrap().to_vec();

    // Metal
    let mut sched = BackendScheduler::new();
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &data).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(r).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[rms_norm] max diff {maxd:.3e}");
    assert!(maxd < 1e-4, "rms_norm Metal diverges: {maxd:.3e}");
}

/// Cross-backend copies: silu on Metal, input/add on CPU.
#[test]
fn metal_cross_backend_copy() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [4, 1, 1, 1], DType::F32);
    let s = gb.silu(x);
    let o = gb.add(s, x);
    gb.output(o);
    let g = gb.build();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    ca.fill_input(&g, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();

    // only silu on Metal -> its input (CPU) and output (CPU consumer) cross backends
    let mut g2 = g.clone();
    g2.nodes[0].backend = Some(Tag::CPU);
    g2.nodes[1].backend = Some(Tag::METAL);
    g2.nodes[2].backend = Some(Tag::CPU);
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
    let splits = sched.split_graph(&g2);
    for (si, sp) in splits.iter().enumerate() {
        eprintln!(
            "[dbg] split {si}: {:?} range {:?} inputs {:?}",
            sp.backend, sp.node_range, sp.inputs
        );
    }
    sched.execute(&g2, &mut alloc).unwrap();
    eprintln!(
        "[dbg] silu out (node 1) = {:?}",
        alloc.copy_to_cpu(1).unwrap()
    );
    eprintln!(
        "[dbg] add out (node 2) = {:?}",
        alloc.copy_to_cpu(2).unwrap()
    );
    let got = alloc.copy_to_cpu(o).unwrap();
    for i in 0..4 {
        assert!(
            (got[i] - expect[i]).abs() < 1e-6,
            "cross-backend out[{i}] {} vs {}",
            got[i],
            expect[i]
        );
    }
}

/// Real-scale matmul (Q8_0 weight, 896×128 like wk): Metal vs CPU.
#[test]
fn metal_matmul_q8_matches_cpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let od = 128usize; // output dim
    let inn = 896usize; // input dim // id
                        // random-ish weight [out][in] row-major -> quantize each row to Q8_0
    let wf: Vec<f32> = (0..od * inn)
        .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let mut wbytes = Vec::new();
    for r in 0..od {
        let row = &wf[r * inn..(r + 1) * inn];
        wbytes.extend_from_slice(&crate::quants::quantize_row_q8_0(row));
    }
    let mut wt = crate::tensor::Tensor::from_data(
        crate::tensor::TensorType::Q8_0,
        &[inn as i64, od as i64, 1, 1],
        wbytes,
    );
    wt.name = "wq8".to_string();
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("wq8", wt.data());

    let nt = 8usize;
    let xd: Vec<f32> = (0..inn * nt)
        .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
        .collect();

    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [inn, nt, 1, 1], DType::F32);
    let m = gb.matmul(x, &wt, None);
    gb.output(m);
    let g = gb.build();

    // manual Q8_0 x f32 reference (weight rows dequantized, f32 activations)
    let mut expect = vec![0.0f32; od * nt];
    {
        let wraw = wt.data();
        let bsz = 34usize; // Q8_0 block: 2 (d) + 32 qs
        for o in 0..od {
            let wrow = &wraw[o * (inn / 32) * bsz..];
            for t in 0..nt {
                let mut acc = 0.0f32;
                for b in 0..inn / 32 {
                    let boff = b * bsz;
                    let d =
                        crate::block::fp16_to_f32(u16::from_le_bytes([wrow[boff], wrow[boff + 1]]));
                    let qs = &wrow[boff + 2..boff + 34];
                    for j in 0..32 {
                        let q = (qs[j] as i8) as f32;
                        acc += q * d * xd[t * inn + b * 32 + j];
                    }
                }
                expect[t * od + o] = acc; // token-major [nt][od]
            }
        }
    }

    // Metal
    let mut sched = BackendScheduler::new();
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &xd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(m).unwrap();
    let mut maxd = 0.0f32;
    let mut worst = 0usize;
    let mut nonzero = 0usize;
    for i in 0..got.len() {
        let d = (got[i] - expect[i]).abs();
        if got[i] != 0.0 {
            nonzero += 1;
        }
        if d > maxd {
            maxd = d;
            worst = i;
        }
    }
    eprintln!(
        "[matmul q8] Metal vs manual-Q8x f32 max diff {maxd:.3e} (nonzero {nonzero}/{})",
        got.len()
    );
    assert!(
        maxd < 1e-3,
        "matmul Metal diverges from Q8_0xf32 reference: {maxd:.3e} (worst idx {worst} (t={}, o={}): got {} expect {})",
        worst / od,
        worst % od,
        got[worst],
        expect[worst]
    );
}

/// rms_norm at REAL scale (d=896, nt=8, like attn_norm) Metal vs CPU.
#[test]
fn metal_rmsnorm_real_scale() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let d = 896usize;
    let nt = 8usize;
    let wdata: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
    let mut bytes = Vec::new();
    for x in &wdata {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("nw896", &bytes);
    let nw = f32t("nw896", [d as i64, 1, 1, 1], wdata);
    let xd: Vec<f32> = (0..d * nt)
        .map(|i| ((i * 97) % 200) as f32 / 100.0 - 1.0)
        .collect();

    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [d, nt, 1, 1], DType::F32);
    let r = gb.rms_norm(x, Some(&nw), 1e-6);
    gb.output(r);
    let g = gb.build();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.register_weight("nw896", nw);
    ca.alloc_graph(&g).unwrap();
    ca.fill_input(&g, "x", &xd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, r).unwrap().to_vec();

    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &xd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(r).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[rms_norm 896] max diff {maxd:.3e}");
    assert!(maxd < 1e-4, "rms_norm(896) Metal diverges: {maxd:.3e}");
}

/// Cross-backend copy at scale: silu of [896, 8] on Metal.
#[test]
fn metal_cross_backend_copy_large() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let d = 896usize;
    let nt = 8usize;
    let xd: Vec<f32> = (0..d * nt)
        .map(|i| ((i * 97) % 200) as f32 / 100.0 - 1.0)
        .collect();
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [d, nt, 1, 1], DType::F32);
    let s = gb.silu(x);
    let o = gb.add(s, x);
    gb.output(o);
    let g = gb.build();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    ca.fill_input(&g, "x", &xd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();

    let mut g2 = g.clone();
    g2.nodes[0].backend = Some(Tag::CPU);
    g2.nodes[1].backend = Some(Tag::METAL);
    g2.nodes[2].backend = Some(Tag::CPU);
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &xd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(o).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[cross large] max diff {maxd:.3e}");
    assert!(maxd < 1e-6, "cross-backend large diverges: {maxd:.3e}");
}

/// Real pattern: Q8_0 embedding (CPU) -> rms_norm (Metal) with a
/// cross-backend copy of the embed output in between.
#[test]
fn metal_embed_then_rmsnorm_cross_backend() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let ne = 32usize;
    let vocab = 8usize;
    let nt = 4usize;
    // Q8_0 embedding [ne, vocab]
    let ef: Vec<f32> = (0..ne * vocab)
        .map(|i| ((i * 31) % 97) as f32 / 50.0 - 1.0)
        .collect();
    let mut ebytes = Vec::new();
    for r in 0..vocab {
        let row = &ef[r * ne..(r + 1) * ne];
        ebytes.extend_from_slice(&crate::quants::quantize_row_q8_0(row));
    }
    let mut emb = crate::tensor::Tensor::from_data(
        crate::tensor::TensorType::Q8_0,
        &[ne as i64, vocab as i64, 1, 1],
        ebytes,
    );
    emb.name = "embq8".to_string();
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("embq8", emb.data());
    let wdata: Vec<f32> = (0..ne).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();
    let mut wbytes = Vec::new();
    for x in &wdata {
        wbytes.extend_from_slice(&x.to_le_bytes());
    }
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("nwE", &wbytes);
    let nw = f32t("nwE", [ne as i64, 1, 1, 1], wdata);
    let nw2 = nw.clone();
    let ids: Vec<u32> = vec![1, 3, 5, 2];

    let mut gb = GraphBuilder::new();
    let idsn = gb.input("token_ids", [nt, 1, 1, 1], DType::I32);
    let e = gb.embedding(idsn, &emb);
    let r = gb.rms_norm(e, Some(&nw), 1e-5);
    gb.output(r);
    let g = gb.build();

    // all-CPU reference
    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.register_weight("embq8", emb.clone());
    ca.register_weight("nwE", nw);
    ca.alloc_graph(&g).unwrap();
    ca.fill_input_i32(&g, "token_ids", &ids).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, r).unwrap().to_vec();

    // embed CPU + rms_norm Metal
    let mut g2 = g.clone();
    g2.nodes[0].backend = Some(Tag::CPU);
    g2.nodes[1].backend = Some(Tag::CPU);
    g2.nodes[2].backend = Some(Tag::METAL);
    let mut alloc = GraphAllocator::new();
    alloc.register_weight("embq8", emb);
    alloc.register_weight("nwE", nw2);
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input_i32(&g2, "token_ids", &ids).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(r).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[embed+rms] max diff {maxd:.3e}");
    assert!(maxd < 1e-3, "embed->rms cross-backend diverges: {maxd:.3e}");
}

/// Multiple Metal nodes alternating with CPU nodes (multi-split sync/copy).
#[test]
fn metal_multi_split_alternation() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let d = 64usize;
    let nt = 4usize;
    let wdata: Vec<f32> = (0..d).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();
    let mut wbytes = Vec::new();
    for x in &wdata {
        wbytes.extend_from_slice(&x.to_le_bytes());
    }
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("nwM", &wbytes);
    let nw = f32t("nwM", [d as i64, 1, 1, 1], wdata);
    let nw2 = nw.clone();
    let xd: Vec<f32> = (0..d * nt)
        .map(|i| ((i * 41) % 199) as f32 / 100.0 - 1.0)
        .collect();

    // x(CPU) -> rms(Metal) -> silu(CPU) -> rms(Metal) -> add(CPU) -> rms(Metal)
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [d, nt, 1, 1], DType::F32);
    let a = gb.rms_norm(x, Some(&nw), 1e-5);
    let b = gb.silu(a);
    let c = gb.rms_norm(b, Some(&nw), 1e-5);
    let e = gb.add(c, x);
    let f = gb.rms_norm(e, Some(&nw), 1e-5);
    gb.output(f);
    let g = gb.build();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.register_weight("nwM", nw);
    ca.alloc_graph(&g).unwrap();
    ca.fill_input(&g, "x", &xd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, f).unwrap().to_vec();

    let mut g2 = g.clone();
    // 0 CPU, 1 Metal, 2 CPU, 3 Metal, 4 CPU, 5 Metal
    g2.nodes[0].backend = Some(Tag::CPU);
    g2.nodes[1].backend = Some(Tag::METAL);
    g2.nodes[2].backend = Some(Tag::CPU);
    g2.nodes[3].backend = Some(Tag::METAL);
    g2.nodes[4].backend = Some(Tag::CPU);
    g2.nodes[5].backend = Some(Tag::METAL);
    let mut alloc = GraphAllocator::new();
    alloc.register_weight("nwM", nw2);
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &xd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(f).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[multi-split] max diff {maxd:.3e}");
    assert!(maxd < 1e-3, "multi-split alternation diverges: {maxd:.3e}");
}

/// Metal KV store + GQA attention vs CPU (F32 inputs, bit-exact check).
#[test]
fn metal_attn_kv_matches_cpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
    let q = gb.input("q", [8, 2, 1, 1], DType::F32);
    let k = gb.input("k", [8, 2, 1, 1], DType::F32);
    let v = gb.input("v", [8, 2, 1, 1], DType::F32);
    gb.kvcache_store(0, k, v, 16);
    let kv = gb.kvcache_load(0, 8, 16, 2);
    let o = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: 2,
            n_head_kv: 2,
            hd: 4,
            hd_kv: 4,
            nkt: 8,
            scale: 0.5,
        },
    );
    gb.output(o);
    let g = gb.build();

    let qd: Vec<f32> = (0..16)
        .map(|i| ((i * 13) % 29) as f32 / 10.0 - 1.4)
        .collect();
    let kd: Vec<f32> = (0..16)
        .map(|i| ((i * 17) % 31) as f32 / 10.0 - 1.5)
        .collect();
    let vd: Vec<f32> = (0..16)
        .map(|i| ((i * 19) % 37) as f32 / 10.0 - 1.8)
        .collect();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    fill_seq_inputs(&mut ca, &g, &[0, 1]);
    ca.fill_input(&g, "q", &qd).unwrap();
    ca.fill_input(&g, "k", &kd).unwrap();
    ca.fill_input(&g, "v", &vd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();

    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    fill_seq_inputs(&mut alloc, &g2, &[0, 1]);
    alloc.fill_input(&g2, "q", &qd).unwrap();
    alloc.fill_input(&g2, "k", &kd).unwrap();
    alloc.fill_input(&g2, "v", &vd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(o).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[attn kv] max diff {maxd:.3e}");
    assert!(maxd < 1e-4, "Metal attention diverges: {maxd:.3e}");
}

/// Real Q4_0 matmul (layer-0 wq: [896, 896]) Metal vs manual reference.
#[test]
fn metal_matmul_q4_matches_reference() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let od = 896usize;
    let inn = 896usize;
    let nt = 8usize;
    let wf: Vec<f32> = (0..od * inn)
        .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
        .collect();
    // quantize to Q4_0: 18 bytes per 32 values (d f16 + 16 nibbles)
    let mut wbytes = Vec::new();
    for r in 0..od {
        let row = &wf[r * inn..(r + 1) * inn];
        for b in 0..inn / 32 {
            let blk = &row[b * 32..b * 32 + 32];
            let mut amax = 0.0f32;
            for &v in blk {
                amax = amax.max(v.abs());
            }
            let d = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
            wbytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
            for j in 0..16 {
                // Q4_0 quantize: q = round(v/d) + 8 in [0,15]. Use i32 for
                // the +8 offset — v/d can reach ±127 when the row's amax is
                // near 1 (d = amax/127), so an i8 intermediate overflows
                // (127+8 > i8::MAX) and panics in debug builds.
                let q0 = ((blk[j] / d).round() as i32 + 8).clamp(0, 15) as u8;
                let q1 = ((blk[j + 16] / d).round() as i32 + 8).clamp(0, 15) as u8;
                wbytes.push(q0 | (q1 << 4));
            }
        }
    }
    let mut wt = crate::tensor::Tensor::from_data(
        crate::tensor::TensorType::Q4_0,
        &[inn as i64, od as i64, 1, 1],
        wbytes,
    );
    wt.name = "wq4".to_string();
    crate::metal::MpsState::get()
        .unwrap()
        .register_weight("wq4", wt.data());

    let xd: Vec<f32> = (0..inn * nt)
        .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
        .collect();

    // manual Q4_0 x f32 reference
    let mut expect = vec![0.0f32; od * nt];
    {
        let wraw = wt.data();
        for o in 0..od {
            let wrow = &wraw[o * (inn / 32) * 18..];
            for t in 0..nt {
                let mut acc = 0.0f32;
                for b in 0..inn / 32 {
                    let boff = b * 18;
                    let d =
                        crate::block::fp16_to_f32(u16::from_le_bytes([wrow[boff], wrow[boff + 1]]));
                    for j in 0..16 {
                        let byte = wrow[boff + 2 + j];
                        let q0 = ((byte & 0x0F) as i8 - 8) as f32;
                        let q1 = ((byte >> 4) as i8 - 8) as f32;
                        acc += q0 * d * xd[t * inn + b * 32 + j];
                        acc += q1 * d * xd[t * inn + b * 32 + j + 16];
                    }
                }
                expect[t * od + o] = acc;
            }
        }
    }

    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [inn, nt, 1, 1], DType::F32);
    let m = gb.matmul(x, &wt, None);
    gb.output(m);
    let g = gb.build();

    let mut sched = BackendScheduler::new();
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", &xd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(m).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[matmul q4] Metal vs manual Q4_0xf32 max diff {maxd:.3e}");
    assert!(maxd < 1e-3, "Q4_0 matmul Metal diverges: {maxd:.3e}");
}

/// Metal KV store + GQA attention at REAL scale (nh=14, nk=2, hd=64,
/// nkt=128, nt=30) vs CPU.
#[test]
fn metal_attn_kv_real_scale() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let (nh, nk, hd, nkt) = (14usize, 2usize, 64usize, 128usize);
    let nt = 30usize;
    let nqt = nh * hd;
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nqt, nt, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
    gb.kvcache_store(0, k, v, 4096);
    let kv = gb.kvcache_load(0, nkt, 4096, nk);
    let o = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: nkt / nk,
            nkt,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    gb.output(o);
    let g = gb.build();

    let qd: Vec<f32> = (0..nqt * nt)
        .map(|i| ((i * 13) % 997) as f32 / 400.0 - 1.2)
        .collect();
    let kd: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 17) % 991) as f32 / 400.0 - 1.3)
        .collect();
    let vd: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 19) % 983) as f32 / 400.0 - 1.1)
        .collect();
    let posd: Vec<u32> = (0..nt as u32).collect();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    fill_seq_inputs(&mut ca, &g, &posd);
    ca.fill_input(&g, "q", &qd).unwrap();
    ca.fill_input(&g, "k", &kd).unwrap();
    ca.fill_input(&g, "v", &vd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();

    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    fill_seq_inputs(&mut alloc, &g2, &posd);
    alloc.fill_input(&g2, "q", &qd).unwrap();
    alloc.fill_input(&g2, "k", &kd).unwrap();
    alloc.fill_input(&g2, "v", &vd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(o).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[attn kv real] max diff {maxd:.3e}");
    assert!(
        maxd < 1e-3,
        "Metal attention at real scale diverges: {maxd:.3e}"
    );
}

/// Metal decode-step attention: nt=1 with 30 already-stored KV rows.
#[test]
fn metal_attn_decode_step() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let (nh, nk, hd, nkt) = (14usize, 2usize, 64usize, 128usize);
    let nqt = nh * hd;
    let nkv_prev = 30usize; // KV already filled by the prefill

    // Build a graph that stores 30 tokens (positions 0..29) and 1 decode
    // token (position 30), then attends with nt=1.
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [nkv_prev + 1, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nqt, nkv_prev + 1, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, nkv_prev + 1, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nkv_prev + 1, 1, 1], DType::F32);
    gb.kvcache_store(0, k, v, 4096);
    let kv = gb.kvcache_load(0, nkt, 4096, nk);
    let o = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: nkt / nk,
            nkt,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    gb.output(o);
    let g = gb.build();

    let qd: Vec<f32> = (0..nqt * (nkv_prev + 1))
        .map(|i| ((i * 13) % 997) as f32 / 400.0 - 1.2)
        .collect();
    let kd: Vec<f32> = (0..nkt * (nkv_prev + 1))
        .map(|i| ((i * 17) % 991) as f32 / 400.0 - 1.3)
        .collect();
    let vd: Vec<f32> = (0..nkt * (nkv_prev + 1))
        .map(|i| ((i * 19) % 983) as f32 / 400.0 - 1.1)
        .collect();
    let posd: Vec<u32> = (0..=nkv_prev as u32).collect();

    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    fill_seq_inputs(&mut ca, &g, &posd);
    ca.fill_input(&g, "q", &qd).unwrap();
    ca.fill_input(&g, "k", &kd).unwrap();
    ca.fill_input(&g, "v", &vd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();

    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    fill_seq_inputs(&mut alloc, &g2, &posd);
    alloc.fill_input(&g2, "q", &qd).unwrap();
    alloc.fill_input(&g2, "k", &kd).unwrap();
    alloc.fill_input(&g2, "v", &vd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(o).unwrap();
    // compare ONLY the decode row (last token)
    let off = nkv_prev * nqt;
    let mut maxd = 0.0f32;
    for i in off..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[attn decode] decode-row max diff {maxd:.3e}");
    assert!(maxd < 1e-3, "Metal decode attention diverges: {maxd:.3e}");
}

/// KV store whose K input is a GPU-computed op (silu) — not host-filled.
#[test]
fn metal_store_after_gpu_op() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
    let q = gb.input("q", [8, 2, 1, 1], DType::F32);
    let k = gb.input("k", [8, 2, 1, 1], DType::F32);
    let v = gb.input("v", [8, 2, 1, 1], DType::F32);
    let ks = gb.silu(k); // GPU-computed K input
    gb.kvcache_store(0, ks, v, 16);
    let kv = gb.kvcache_load(0, 8, 16, 2);
    let o = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: 2,
            n_head_kv: 2,
            hd: 4,
            hd_kv: 4,
            nkt: 8,
            scale: 0.5,
        },
    );
    gb.output(o);
    let g = gb.build();
    let qd: Vec<f32> = (0..16)
        .map(|i| ((i * 13) % 29) as f32 / 10.0 - 1.4)
        .collect();
    let kd: Vec<f32> = (0..16)
        .map(|i| ((i * 17) % 31) as f32 / 10.0 - 1.5)
        .collect();
    let vd: Vec<f32> = (0..16)
        .map(|i| ((i * 19) % 37) as f32 / 10.0 - 1.8)
        .collect();
    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    fill_seq_inputs(&mut ca, &g, &[0, 1]);
    ca.fill_input(&g, "q", &qd).unwrap();
    ca.fill_input(&g, "k", &kd).unwrap();
    ca.fill_input(&g, "v", &vd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    fill_seq_inputs(&mut alloc, &g2, &[0, 1]);
    alloc.fill_input(&g2, "q", &qd).unwrap();
    alloc.fill_input(&g2, "k", &kd).unwrap();
    alloc.fill_input(&g2, "v", &vd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(o).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[store after gpu op] max diff {maxd:.3e}");
    assert!(maxd < 1e-4, "store-after-gpu-op diverges: {maxd:.3e}");
}

/// KV store+attn at REAL dims (nkt=128, n_ctx=32768, nt=30) hand-filled.
#[test]
fn metal_store_real_dims() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let (nh, nk, hd, nkt) = (14usize, 2usize, 64usize, 128usize);
    let nt = 30usize;
    let nqt = nh * hd;
    let mut gb = GraphBuilder::new();
    let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
    let q = gb.input("q", [nqt, nt, 1, 1], DType::F32);
    let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
    gb.kvcache_store(0, k, v, 32768);
    let kv = gb.kvcache_load(0, nkt, 32768, nk);
    let o = gb.attn(
        q,
        kv,
        pos,
        crate::graph::ops::AttnMode::Gqa,
        crate::graph::ops::AttnMeta {
            layer: 0,
            n_head: nh,
            n_head_kv: nk,
            hd,
            hd_kv: nkt / nk,
            nkt,
            scale: 1.0 / (hd as f32).sqrt(),
        },
    );
    gb.output(o);
    let g = gb.build();
    let qd: Vec<f32> = (0..nqt * nt)
        .map(|i| ((i * 13) % 997) as f32 / 400.0 - 1.2)
        .collect();
    let kd: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 17) % 991) as f32 / 400.0 - 1.3)
        .collect();
    let vd: Vec<f32> = (0..nkt * nt)
        .map(|i| ((i * 19) % 983) as f32 / 400.0 - 1.1)
        .collect();
    let posd: Vec<u32> = (0..nt as u32).collect();
    let mut sched = BackendScheduler::new();
    let mut ca = GraphAllocator::new();
    ca.alloc_graph(&g).unwrap();
    fill_seq_inputs(&mut ca, &g, &posd);
    ca.fill_input(&g, "q", &qd).unwrap();
    ca.fill_input(&g, "k", &kd).unwrap();
    ca.fill_input(&g, "v", &vd).unwrap();
    sched.execute(&g, &mut ca).unwrap();
    let expect = ca.get_buffer(&g, o).unwrap().to_vec();
    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    fill_seq_inputs(&mut alloc, &g2, &posd);
    alloc.fill_input(&g2, "q", &qd).unwrap();
    alloc.fill_input(&g2, "k", &kd).unwrap();
    alloc.fill_input(&g2, "v", &vd).unwrap();
    sched.execute(&g2, &mut alloc).unwrap();
    let got = alloc.copy_to_cpu(o).unwrap();
    let mut maxd = 0.0f32;
    for i in 0..got.len() {
        maxd = maxd.max((got[i] - expect[i]).abs());
    }
    eprintln!("[store real dims] max diff {maxd:.3e}");
    assert!(maxd < 1e-3, "store at real dims diverges: {maxd:.3e}");
}

/// #38: a Metal `KvcacheStore` whose row is past the arena refuses loudly —
/// naming the actual cell and the arena size — and never dispatches the store.
///
/// The allocator bounds the `cells` input when it is filled through
/// `fill_input_i32` (`GraphAllocator::check_positions_bound`). This gate
/// deliberately fills it through the generic f32 `fill_input`, which carries
/// the same `f32::from_bits` layout (compute-graph rule 4) but not the
/// i32-specific check, so the Metal arm is reached with a row the allocator
/// guard does not see — the remaining input path the ticket closes. Gate
/// contract rule 2: the control must not be refused by an earlier check, which
/// is exactly why `fill_input_i32` is *not* used here.
#[test]
fn metal_kvcache_store_refuses_a_cell_past_the_arena() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    let Some(_b) = MetalBackend::new() else {
        eprintln!("MPS unavailable; skipping");
        return;
    };
    let nkt = 8usize;
    let n_ctx = 16usize;
    let nt = 2usize;
    let mut gb = GraphBuilder::new();
    let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
    let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
    let store = gb.kvcache_store(0, k, v, n_ctx);
    gb.output(store);
    let g = gb.build();

    let mut g2 = g.clone();
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    let kd: Vec<f32> = (0..nkt * nt).map(|i| i as f32 * 0.25).collect();
    alloc.fill_input(&g2, "k", &kd).unwrap();
    alloc.fill_input(&g2, "v", &kd).unwrap();
    // Row 1 is the last cell of the 16-cell arena plus one (cell 16).
    let cells: [u32; 2] = [0, n_ctx as u32];
    let bits: Vec<f32> = cells.iter().map(|&c| f32::from_bits(c)).collect();
    alloc.fill_input(&g2, "cells", &bits).unwrap();

    let mut sched = BackendScheduler::new();
    let err = sched
        .execute(&g2, &mut alloc)
        .expect_err("a store at cell 16 of a 16-cell arena must refuse, not write");
    assert!(
        err.contains("cell 16") && err.contains("16-cell arena"),
        "the refusal must name the actual cell and the arena size, got: {err}"
    );
}
