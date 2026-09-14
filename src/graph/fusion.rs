//! Operator fusion pass (Phase 4).
//!
//! Graph-pattern matching over the IR. Every fusion is gated by the target
//! backend's `supports_fused` so fused IR nodes are only produced where a
//! kernel exists (no double-fusion with existing hand-written GPU kernels —
//! minfer's Metal `swiglu_f32` maps 1:1 to `Op::SwiGLU`). Fusions are applied
//! per node's assigned backend.
//!
//! Scope: the SwiGLU rewrite is the only rule. The plan's `RoPE(Add(x, b))` →
//! `FusedBiasRope` rule was removed once it became clear no backend would claim
//! the capability (the bias+rope work is covered by the build-time fused nodes:
//! Metal/CUDA `attn_bias_rope_store`). `BatchMatMul` remains deferred — the
//! single-output IR cannot express a multi-output node (see
//! docs/COMPUTE-GRAPH-DESIGN.md §5.4).

use super::backend::Backend;
use super::ops::{FusedOp, Op};
use super::ComputeGraph;

pub struct FusionPass;

impl FusionPass {
    pub fn new() -> Self {
        Self
    }

    /// Run all supported fusions over the graph. Returns the number of nodes
    /// rewritten. `backend_for` returns the backend a node is assigned to
    /// (used to gate fusions per backend capability).
    pub fn run(
        &self,
        graph: &mut ComputeGraph,
        backends: &[&dyn Backend],
        backend_of: &dyn Fn(&ComputeGraph, usize) -> Option<usize>, // node id -> backend index
    ) -> usize {
        let mut n = 0;
        n += self.fuse_swiglu(graph, backends, backend_of);
        n
    }

    /// Pattern: `Mul(Silu(X), Y)` → `SwiGLU(X, Y)`.
    /// Gate: backend supports `FusedOp::SwiGLU`.
    fn fuse_swiglu(
        &self,
        graph: &mut ComputeGraph,
        backends: &[&dyn Backend],
        backend_of: &dyn Fn(&ComputeGraph, usize) -> Option<usize>,
    ) -> usize {
        let n = graph.n_nodes();
        let mut replaced = 0usize;
        let mut new_ops: Vec<Option<Op>> = vec![None; n];
        for id in 0..n {
            if !matches!(graph.node(id).op, Op::Mul) {
                continue;
            }
            let mul = graph.node(id);
            if mul.src.len() != 2 {
                continue;
            }
            let (s, y) = (mul.src[0], mul.src[1]);
            let is_silu = |x: usize| matches!(graph.node(x).op, Op::Silu);
            let (silu_in, gate, up) = if is_silu(s) {
                (graph.node(s).src[0], graph.node(s).src[0], y)
            } else if is_silu(y) {
                (graph.node(y).src[0], graph.node(y).src[0], s)
            } else {
                continue;
            };
            let _ = silu_in;
            // gate the fusion on the mul node's backend capability
            let ok = match backend_of(graph, id) {
                Some(bi) => backends[bi].supports_fused(&FusedOp::SwiGLU),
                None => false,
            };
            if !ok {
                continue;
            }
            // replace Mul with SwiGLU(gate, up)
            new_ops[id] = Some(Op::SwiGLU);
            if let Some(node) = graph.nodes.get_mut(id) {
                node.src = vec![gate, up];
            }
            replaced += 1;
        }
        for (id, op) in new_ops.into_iter().enumerate() {
            if let Some(op) = op {
                graph.nodes[id].op = op;
            }
        }
        replaced
    }
}

impl Default for FusionPass {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
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
}
