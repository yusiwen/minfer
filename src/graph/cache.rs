//! Graph reuse cache (Phase 4→6): params-only deterministic reuse.
//!
//! Mirrors llama.cpp's `llm_graph_params::allow_reuse` + `llm_graph_result`
//! reuse path. Key invariant: **graph topology is a deterministic function of
//! `GraphParams`** — equal params ⇒ identical topology ⇒ the graph is reused
//! as-is, only input data is refreshed. `n_past` never appears here (it is
//! execution data).
//!
//! The allocator lives inside the cache and **survives graph rebuilds**: the
//! persistent KV regions are exactly the KV cache, so a prefill→decode
//! transition (different `n_tokens`/`gtype` ⇒ rebuild) must not lose them.
//! Only the node/buffer mapping is recomputed on rebuild.

use super::alloc::GraphAllocator;
use super::params::GraphParams;
use super::{ComputeGraph, NodeId};

pub struct GraphCache {
    graph: Option<ComputeGraph>,
    alloc: GraphAllocator,
    prev_params: Option<GraphParams>,
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCache {
    pub fn new() -> Self {
        Self { graph: None, alloc: GraphAllocator::new(), prev_params: None }
    }

    /// Params-only reuse check. On success the previously stored graph is
    /// reused without rebuilding (caller then refreshes input data).
    pub fn try_reuse(&mut self, params: &GraphParams) -> bool {
        match (&self.prev_params, &self.graph) {
            (Some(prev), Some(_)) if Self::params_match(prev, params) => {
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

    /// Store a freshly built graph. The allocator is kept (KV regions persist);
    /// its liveness mapping is recomputed by the caller via `alloc_graph`.
    pub fn replace_graph(&mut self, graph: ComputeGraph, params: GraphParams) {
        self.graph = Some(graph);
        self.prev_params = Some(params);
    }

    /// The allocator (weight registration before first `alloc_graph`).
    pub fn alloc(&mut self) -> &mut GraphAllocator {
        &mut self.alloc
    }

    /// Take the current graph + allocator for execution.
    pub fn current(&mut self) -> Option<(&ComputeGraph, &mut GraphAllocator)> {
        match &self.graph {
            Some(g) => Some((g, &mut self.alloc)),
            None => None,
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
        cache.replace_graph(g, params(1, 1));

        // same params -> reuse
        assert!(cache.try_reuse(&params(1, 1)));
        // n_tokens changed -> rebuild
        assert!(!cache.try_reuse(&params(4, 1)));
        // weights version changed (LoRA switch) -> rebuild
        assert!(!cache.try_reuse(&params(1, 2)));
        // gtype differs -> rebuild
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
    fn allocator_survives_rebuild() {
        let mut cache = GraphCache::new();
        cache.replace_graph(tiny_graph(), params(1, 1));
        // rebuild with different params: allocator object identity persists
        // (a KV persistent region registered before must still be there)
        cache.alloc().alloc_persistent("kv.test", 16);
        assert!(cache.alloc().get_persistent("kv.test").is_some());
        cache.replace_graph(tiny_graph(), params(4, 1));
        assert!(
            cache.alloc().get_persistent("kv.test").is_some(),
            "persistent regions must survive graph rebuilds"
        );
    }

    #[test]
    fn structural_check_detects_different_graph() {
        #[cfg(debug_assertions)]
        {
            let mut cache = GraphCache::new();
            cache.replace_graph(tiny_graph(), params(1, 1));
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
