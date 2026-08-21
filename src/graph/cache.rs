//! Graph reuse cache (Phase 4): params-only deterministic reuse.
//!
//! Mirrors llama.cpp's `llm_graph_params::allow_reuse` + `llm_graph_result`
//! reuse path. Key invariant: **graph topology is a deterministic function of
//! `GraphParams`** — equal params ⇒ identical topology ⇒ the graph and its
//! buffer allocation are reused as-is, only input data is refreshed. `n_past`
//! never appears here (it is execution data).

use super::alloc::GraphAllocator;
use super::params::GraphParams;
use super::{ComputeGraph, NodeId};

pub struct GraphCache {
    graph: Option<ComputeGraph>,
    alloc: Option<GraphAllocator>,
    prev_params: Option<GraphParams>,
}

impl Default for GraphCache {
    fn default() -> Self {
        Self { graph: None, alloc: None, prev_params: None }
    }
}

impl GraphCache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Params-only reuse check. On success the previously stored graph + alloc
    /// are reused without rebuilding (caller then refreshes input data).
    pub fn try_reuse(&mut self, params: &GraphParams) -> bool {
        match (&self.prev_params, &self.graph, &self.alloc) {
            (Some(prev), Some(_), Some(_)) if Self::params_match(prev, params) => {
                self.prev_params = Some(params.clone());
                true
            }
            _ => false,
        }
    }

    fn params_match(a: &GraphParams, b: &GraphParams) -> bool {
        a.n_tokens == b.n_tokens
            && a.n_seqs == b.n_seqs
            && a.gtype == b.gtype
            && a.cparams == b.cparams
            && a.weights_version == b.weights_version
    }

    /// Store a freshly built graph + allocation.
    pub fn store(&mut self, graph: ComputeGraph, alloc: GraphAllocator, params: GraphParams) {
        self.graph = Some(graph);
        self.alloc = Some(alloc);
        self.prev_params = Some(params);
    }

    /// Take the current graph + allocator for execution.
    pub fn current(&mut self) -> Option<(&ComputeGraph, &mut GraphAllocator)> {
        match (&self.graph, &mut self.alloc) {
            (Some(g), Some(a)) => Some((g, a)),
            _ => None,
        }
    }

    /// Debug-only structural check: two graphs built from equal params must be
    /// identical (op sequence with full payloads, shapes, dependencies).
    #[cfg(debug_assertions)]
    pub fn verify_structural(&self, graph: &ComputeGraph) -> bool {
        let Some(prev) = &self.graph else { return true };
        if prev.nodes.len() != graph.nodes.len() {
            return false;
        }
        prev.nodes.iter().zip(graph.nodes.iter()).all(|(a, b)| {
            a.op == b.op && a.out_shape == b.out_shape && a.src == b.src
        })
    }

    /// Debug-only: if params match but the new build differs structurally, the
    /// builder is not deterministic — panic loudly.
    #[cfg(debug_assertions)]
    pub fn assert_consistent_rebuild(&self, params: &GraphParams, graph: &ComputeGraph) {
        if let Some(prev) = &self.prev_params {
            if Self::params_match(prev, params) && !self.verify_structural(graph) {
                panic!(
                    "graph rebuild with identical params produced a different topology \
                     (builder is not deterministic)"
                );
            }
        }
    }

    /// Node ids of the stored graph's outputs (convenience for the caller).
    pub fn outputs(&self) -> Vec<NodeId> {
        self.graph.as_ref().map(|g| g.outputs.clone()).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::params::GraphType;
    use crate::graph::DType;

    fn params(n_tokens: usize, weights_version: u64) -> GraphParams {
        GraphParams {
            n_tokens,
            n_seqs: 1,
            gtype: if n_tokens == 1 { GraphType::Decode } else { GraphType::Prefill },
            cparams: Default::default(),
            weights_version,
        }
    }

    fn tiny_graph() -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        b.output(s);
        b.build()
    }

    #[test]
    fn reuse_requires_equal_params() {
        let mut cache = GraphCache::new();
        assert!(!cache.try_reuse(&params(1, 1)), "nothing stored yet");

        let g = tiny_graph();
        let outputs_expected = g.outputs.clone();
        let alloc = GraphAllocator::new();
        cache.store(g, alloc, params(1, 1));

        // same params -> reuse
        assert!(cache.try_reuse(&params(1, 1)));
        // n_tokens changed -> rebuild
        assert!(!cache.try_reuse(&params(4, 1)));
        // weights version changed (LoRA switch) -> rebuild
        assert!(!cache.try_reuse(&params(1, 2)));
        // gtype differs (n_tokens 4 => prefill vs decode) -> rebuild
        let p = GraphParams {
            n_tokens: 1,
            n_seqs: 1,
            gtype: GraphType::Prefill,
            cparams: Default::default(),
            weights_version: 1,
        };
        assert!(!cache.try_reuse(&p));

        // current() yields the stored graph
        let (g2, _) = cache.current().unwrap();
        assert_eq!(g2.outputs, outputs_expected);
    }

    #[test]
    fn n_past_is_not_part_of_reuse() {
        // n_past never appears in GraphParams — the graph is reused regardless
        // of KV position. Here we simply assert params without n_past compare
        // equal: two identical builds reuse.
        let mut cache = GraphCache::new();
        cache.store(tiny_graph(), GraphAllocator::new(), params(1, 1));
        assert!(cache.try_reuse(&params(1, 1)));
    }

    #[test]
    fn structural_check_detects_different_graph() {
        #[cfg(debug_assertions)]
        {
            let mut cache = GraphCache::new();
            cache.store(tiny_graph(), GraphAllocator::new(), params(1, 1));
            // a different topology with the same params must fail verification
            let mut b = GraphBuilder::new();
            let x = b.input("x", [4, 1, 1, 1], DType::F32);
            let s = b.silu(x);
            let o = b.add(s, x); // extra node -> different topology
            b.output(o);
            let g2 = b.build();
            assert!(!cache.verify_structural(&g2));
        }
    }
}
