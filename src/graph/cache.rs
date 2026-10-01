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
//!
//! E4 S3: the cache holds **several** graphs, one per distinct `GraphParams`
//! (MRU first, bounded), and switching between them is the *assign* half of
//! reserve/assign: the target graph already carries its backend assignment, so a
//! switch is `alloc_graph` — liveness plus a re-map onto the allocator's reserved
//! slots — with no graph build, no fusion pass and no pool traffic. That is what
//! stops a server alternating 1-wide and N-wide decode steps (roadmap §14 row 3)
//! and a chunked prefill whose chunk sizes repeat (E3) from rebuilding every
//! time.

use super::alloc::GraphAllocator;
use super::params::GraphParams;
use super::ComputeGraph;
use std::sync::atomic::{AtomicU64, Ordering};

/// Monotonic graph identity for CUDA Graph caching (llama.cpp
/// `ggml_graph_next_uid` analog): assigned when a NEW graph is stored in the
/// cache; a reused graph keeps its uid. Starts at 1 (0 = "no uid").
static NEXT_GRAPH_UID: AtomicU64 = AtomicU64::new(1);

/// How many distinct `GraphParams` a cache keeps (E4 S3). Small on purpose: the shapes a
/// session alternates are a handful (1-wide decode, N-wide decode, the prefill chunk sizes),
/// and a cached graph is its node vector — kilobytes, not buffers.
pub const MAX_CACHED_GRAPHS: usize = 8;

pub struct GraphCache {
    /// E4 S3: one graph per distinct params, MRU first; `graphs[0]` is the current one.
    graphs: Vec<(GraphParams, ComputeGraph)>,
    alloc: GraphAllocator,
    /// Builds vs reuses, for the gate that proves a switch stopped rebuilding.
    builds: usize,
    reuses: usize,
}

impl Default for GraphCache {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphCache {
    pub fn new() -> Self {
        Self {
            graphs: Vec::new(),
            alloc: GraphAllocator::new(),
            builds: 0,
            reuses: 0,
        }
    }

    /// Params-only reuse check. On success a **cached** graph with these params becomes
    /// current — and the allocator is re-mapped onto it (`alloc_graph`: liveness + slots, the
    /// assign half) instead of rebuilding. The caller then refreshes input data.
    ///
    /// E4 S3: matching is over the whole cache, not just the previous graph, so alternating
    /// two shapes hits from the second switch on.
    pub fn try_reuse(&mut self, params: &GraphParams) -> Result<bool, String> {
        let Some(pos) = self
            .graphs
            .iter()
            .position(|(p, _)| Self::params_match(p, params))
        else {
            return Ok(false);
        };
        let (_, graph) = self.graphs.remove(pos);
        // The re-map can fail (the budget gate), and a failed switch must not silently leave
        // the allocator pointing at the old graph's buffers — the error is returned to the
        // caller, which then rebuilds.
        self.alloc.alloc_graph(&graph)?;
        self.graphs.insert(0, (params.clone(), graph));
        self.reuses += 1;
        Ok(true)
    }

    /// (builds, reuses) since the cache was created — the observable behind "a switch stopped
    /// rebuilding" (E4 S3's acceptance).
    /// Test-only (#238): driven by `graph::cache::tests::switching_between_cached_graphs_re_maps_instead_of_rebuilding`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn stats(&self) -> (usize, usize) {
        (self.builds, self.reuses)
    }

    /// How many graphs are cached right now.
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn cached_graphs(&self) -> usize {
        self.graphs.len()
    }

    fn params_match(a: &GraphParams, b: &GraphParams) -> bool {
        a.n_tokens == b.n_tokens
            && a.n_out == b.n_out
            && a.gtype == b.gtype
            && a.cparams == b.cparams
            && a.weights_version == b.weights_version
    }

    /// Store a freshly built graph. The allocator is kept (KV regions persist);
    /// its liveness mapping is recomputed by the caller via `alloc_graph`.
    /// The graph gets a fresh monotonic uid (CUDA Graph cache key part).
    pub fn replace_graph(&mut self, mut graph: ComputeGraph, params: GraphParams) {
        graph.uid = NEXT_GRAPH_UID.fetch_add(1, Ordering::Relaxed);
        self.graphs.retain(|(p, _)| !Self::params_match(p, &params));
        self.graphs.insert(0, (params, graph));
        self.builds += 1;
        // Bounded: the oldest graph is dropped (its buffers are already back in the
        // allocator's slot table — the next switch re-assigns them).
        self.graphs.truncate(MAX_CACHED_GRAPHS);
    }

    /// The allocator (weight registration before first `alloc_graph`).
    pub fn alloc(&mut self) -> &mut GraphAllocator {
        &mut self.alloc
    }

    /// Take the current graph + allocator for execution.
    pub fn current(&mut self) -> Option<(&ComputeGraph, &mut GraphAllocator)> {
        match self.graphs.first() {
            Some((_, g)) => Some((g, &mut self.alloc)),
            None => None,
        }
    }

    /// Debug-only structural check: two graphs built from equal params must be
    /// identical (op sequence with full payloads, shapes, dependencies). Used
    /// only from `mod tests` in debug builds, so it is gated off a normal
    /// (non-test) binary build.
    #[cfg(all(test, debug_assertions))]
    pub fn verify_structural(&self, graph: &ComputeGraph) -> bool {
        let Some((_, prev)) = self.graphs.first() else {
            return true;
        };
        if prev.nodes.len() != graph.nodes.len() {
            return false;
        }
        prev.nodes
            .iter()
            .zip(graph.nodes.iter())
            .all(|(a, b)| a.op == b.op && a.out_shape == b.out_shape && a.src == b.src)
    }
}

#[cfg(test)]
mod tests;
