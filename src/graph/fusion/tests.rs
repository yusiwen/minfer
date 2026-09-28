//! `#[cfg(test)] mod tests` for `src/graph/fusion.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::graph::builder::GraphBuilder;
use crate::graph::cpu_backend::CpuBackend;
use crate::graph::DType;

fn swiglu_pattern_graph() -> ComputeGraph {
    let mut b = GraphBuilder::new();
    let x = b.input("x", [8, 4, 1, 1], DType::F32);
    let gate = b.input("gate", [8, 4, 1, 1], DType::F32);
    let up = b.input("up", [8, 4, 1, 1], DType::F32);
    let _silu = b.silu(gate);
    // reuse builder: silu(gate) then mul by up
    let mut h = b.silu(x);
    h = b.mul(h, up);
    b.output(h);
    b.build()
}

#[test]
fn swiglu_fusion_applies_when_backend_supports() {
    let mut g = swiglu_pattern_graph();
    let cpu = CpuBackend::new();
    // cpu supports SwiGLU fused
    let backends: [&dyn Backend; 1] = [&cpu];
    let backend_of = |_: &ComputeGraph, _: usize| Some(0);
    let pass = FusionPass::new();
    let n = pass.run(&mut g, &backends, &backend_of);
    assert_eq!(n, 1, "expected one SwiGLU fusion");
    assert!(g.nodes.iter().any(|nd| matches!(nd.op, Op::SwiGLU)));
    // the fused node's src = [gate, up]
    let fused = g
        .nodes
        .iter()
        .find(|nd| matches!(nd.op, Op::SwiGLU))
        .unwrap();
    assert_eq!(fused.src.len(), 2);
}

#[test]
fn swiglu_fusion_skipped_when_backend_does_not_support() {
    let mut g = swiglu_pattern_graph();
    let cpu = CpuBackend::new();
    let backends: [&dyn Backend; 1] = [&cpu];
    let backend_of = |_: &ComputeGraph, _: usize| None; // unassigned -> no fusion
    let pass = FusionPass::new();
    let n = pass.run(&mut g, &backends, &backend_of);
    assert_eq!(n, 0);
    assert!(!g.nodes.iter().any(|nd| matches!(nd.op, Op::SwiGLU)));
}
