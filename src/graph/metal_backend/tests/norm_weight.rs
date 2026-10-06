//! #40: Metal's norm-weight guards.
//!
//! `Op::RmsNorm` / `Op::QkNorm` used to fall through to the *weightless*
//! `rms_norm` kernel whenever the gain could not be resolved — either because
//! `NormMeta::weight_name` was `None`, or because the name was set but the
//! device had no registration for it. Both are a missing gain, and the
//! fallthrough produced plausible-looking output from a wrong computation, the
//! failure mode `docs/GPU_SAFETY.md` forbids. Both arms now call
//! `MetalBackend::norm_weight`, which refuses loudly naming the node and the
//! missing tensor (the CUDA twin is `CudaBackend::norm_weight`).
//!
//! The gates drive the real `BackendScheduler::execute` → `execute_node` path on
//! a Metal build. The unregistered-name case is deliberately built by leaving
//! `MpsState` without the registration, which is exactly the
//! assignment/offload mistake the guard must catch; the weightless case is built
//! through the builder's `None` weight. `#![cfg(target_os = "macos")]` is
//! deliberately **not** used, so CI's `build-macos` type-checks them and a Mac
//! runs them (record in the PR).

use super::*;
use crate::graph::scheduler::BackendScheduler;
use crate::graph::{Backend as Tag, DType};

/// Put every node on Metal and drive the graph through the real scheduler,
/// returning the loud refusal. The weight lookup fails before the input is read,
/// but the input is filled so the fixture is a well-formed graph (gate contract
/// rule 2: nothing earlier may refuse it).
fn run_all_metal_expect_err(g: crate::graph::ComputeGraph, input: &[f32]) -> String {
    let sched = BackendScheduler::new();
    let mut g2 = g;
    for n in &mut g2.nodes {
        n.backend = Some(Tag::METAL);
    }
    let mut alloc = GraphAllocator::new();
    alloc.enable_metal();
    alloc.alloc_graph(&g2).unwrap();
    alloc.fill_input(&g2, "x", input).unwrap();
    sched
        .execute(&g2, &mut alloc)
        .expect_err("a norm with no resolvable gain must refuse, not run weightless")
}

/// The name is set but the weight was never registered on this Metal device
/// (an assignment/offload mistake). Must name the missing tensor and the node.
#[test]
fn metal_rms_norm_refuses_a_weight_not_on_gpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    // Deliberately never registered on MpsState.
    let nw = f32t("nw_missing", [8, 1, 1, 1], vec![1.0; 8]);
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 3, 1, 1], DType::F32);
    let r = gb.rms_norm(x, Some(&nw), 1e-5);
    gb.output(r);
    let err = run_all_metal_expect_err(gb.build(), &vec![0.5f32; 24]);
    assert!(
        err.contains("nw_missing") && err.contains("rms_norm"),
        "the refusal must name the missing tensor and the node, got: {err}"
    );
}

/// `NormMeta::weight_name` is `None`: no producer in the tree builds a
/// weightless norm, so this is a malformed graph, not a legitimate compute.
#[test]
fn metal_rms_norm_refuses_a_weightless_node() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 3, 1, 1], DType::F32);
    let r = gb.rms_norm(x, None, 1e-5);
    gb.output(r);
    let err = run_all_metal_expect_err(gb.build(), &vec![0.5f32; 24]);
    assert!(
        err.contains("rms_norm") && err.contains("has no norm weight"),
        "the refusal must name the node and say the weight is missing, got: {err}"
    );
}

/// The sibling `Op::QkNorm` arm had the same `None` fallthrough; it must refuse
/// the same way.
#[test]
fn metal_qk_norm_refuses_a_weight_not_on_gpu() {
    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let nw = f32t("qk_nw_missing", [4, 1, 1, 1], vec![1.0; 4]);
    let mut gb = GraphBuilder::new();
    let x = gb.input("x", [8, 3, 1, 1], DType::F32);
    let q = gb.qk_norm(x, Some(&nw), 4, 2, 1e-5);
    gb.output(q);
    let err = run_all_metal_expect_err(gb.build(), &vec![0.5f32; 24]);
    assert!(
        err.contains("qk_nw_missing") && err.contains("qk_norm"),
        "the refusal must name the missing tensor and the node, got: {err}"
    );
}
