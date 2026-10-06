//! #39: Metal's decode-fusion shape guards.
//!
//! `FusedFFN` / `FusedQKV` / `FusedQkvNorm` are decode-only (`nt == 1`). Before
//! #39 each arm asserted that with `debug_assert!`, which a **release** build
//! compiles out — so a release build proceeded with a shape the kernel cannot
//! handle, against the rule that a kernel-invariant violation returns `Err` from
//! `execute_node`, never assumes. Each arm now refuses `nt != 1` up front, before
//! the weight lookup (shape validation is weight-independent and cheaper).
//!
//! The gates below drive the real `BackendScheduler::execute` → `execute_node`
//! path on a Metal build with a prefill-shaped fusion node (`nt == 2`) and assert
//! the loud refusal names the node and the observed `nt`. They are split out of
//! `src/graph/metal_backend/tests.rs` (issue #267 pattern); the fixtures live in
//! the parent module and are reached through `use super::*;`.
//!
//! `#![cfg(target_os = "macos")]` is deliberately **not** used: these tests
//! compile on every platform and skip at runtime when MPS is unavailable, exactly
//! like the rest of the module. CI's `build-macos` therefore only type-checks
//! them; the assertions run on a Mac (record in the PR).

use super::*;
use crate::graph::ops::{FusedFfnMeta, FusedQkvMeta, FusedQkvNormMeta};
use crate::graph::scheduler::BackendScheduler;
use crate::graph::{Backend as Tag, DType};
use crate::tensor::TensorType;
use crate::vec_ops::RopeStyle;

/// Put every node on Metal and drive the graph through the real scheduler. The
/// caller has already filled the graph's inputs.
fn run_all_metal(g: crate::graph::ComputeGraph) -> String {
    let sched = BackendScheduler::new();
    let mut g2 = g;
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    // The guard fires before any input is read, but fill them so the fixture is
    // a well-formed graph (gate contract rule 2: nothing earlier may refuse it).
    alloc.fill_input(&g2, "x", &vec![0.0f32; 8 * 2]).unwrap();
    if g2.nodes.iter().any(|n| n.name == "positions") {
        alloc.fill_input_i32(&g2, "positions", &[0, 1]).unwrap();
    }
    if g2.nodes.iter().any(|n| n.name == "cells") {
        alloc.fill_input_i32(&g2, "cells", &[0, 1]).unwrap();
    }
    sched
        .execute(&g2, &mut alloc)
        .expect_err("a decode-only fusion at nt=2 must refuse, not dispatch")
}

#[test]
fn metal_fused_ffn_refuses_nt_other_than_one() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 2, 1, 1], DType::F32);
    let n = gb.fused_ffn(
        x,
        FusedFfnMeta {
            gu_weight: "gu_not_registered".to_string(),
            weight_ttype: TensorType::Q8_0,
            in_dim: 8,
            nf: 4,
        },
    );
    gb.output(n);
    let err = run_all_metal(gb.build());
    assert!(
        err.contains("fused_ffn")
            && err.contains("FusedFFN is decode (nt==1) only")
            && err.contains("got nt=2"),
        "FusedFFN must name the node and the observed nt, got: {err}"
    );
}

#[test]
fn metal_fused_qkv_refuses_nt_other_than_one() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 2, 1, 1], DType::F32);
    let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
    let n = gb.fused_qkv(
        x,
        pos,
        0,
        FusedQkvMeta {
            qkv_weight: "qkv_not_registered".to_string(),
            bias_q: None,
            bias_k: None,
            bias_v: None,
            weight_ttype: TensorType::Q8_0,
            in_dim: 8,
            nqt: 8,
            nkt: 4,
            hd: 4,
            nh: 2,
            nk: 1,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            rope_style: RopeStyle::NonInterleaved,
            kv_elems: 4 * 16,
            row_elems: 4,
        },
    );
    gb.output(n);
    let err = run_all_metal(gb.build());
    assert!(
        err.contains("fused_qkv")
            && err.contains("FusedQKV is decode (nt==1) only")
            && err.contains("got nt=2"),
        "FusedQKV must name the node and the observed nt, got: {err}"
    );
}

#[test]
fn metal_fused_qkv_norm_refuses_nt_other_than_one() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 2, 1, 1], DType::F32);
    let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
    let n = gb.fused_qkv_norm(
        x,
        pos,
        0,
        FusedQkvNormMeta {
            qkv_weight: "qkv_not_registered".to_string(),
            q_norm_name: Some("qn_not_registered".to_string()),
            k_norm_name: Some("kn_not_registered".to_string()),
            weight_ttype: TensorType::Q8_0,
            in_dim: 8,
            nqt: 8,
            nkt: 4,
            hd: 4,
            nh: 2,
            nk: 1,
            freq_base: 10_000.0,
            freq_scale: 1.0,
            rope_style: RopeStyle::NonInterleaved,
            kv_elems: 4 * 16,
            row_elems: 4,
            eps: 1e-6,
        },
    );
    gb.output(n);
    let err = run_all_metal(gb.build());
    assert!(
        err.contains("fused_qkv_norm")
            && err.contains("FusedQkvNorm is decode (nt==1) only")
            && err.contains("got nt=2"),
        "FusedQkvNorm must name the node and the observed nt, got: {err}"
    );
}
