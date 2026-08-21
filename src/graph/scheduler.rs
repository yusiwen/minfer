//! Backend scheduler (Phase 4: assign → split → execute; fusion is separate).
//!
//! Mirrors llama.cpp's `ggml_backend_sched_split_graph` + `compute_splits`
//! (simplified: single backend today, cross-backend transfers land with the
//! Metal backend in Phase 3):
//! - `assign_backends`: per-op capability-driven assignment (all-CPU for now)
//! - `split_graph`: partition the node sequence into contiguous same-backend
//!   splits, recording cross-split inputs/outputs
//! - `execute`: run splits in order, copying inputs at boundaries (no-op when
//!   everything is on one backend)
//!
//! The buffer pools live in `GraphAllocator` (single source of truth — the
//! allocator is also the capability registry); the scheduler orchestrates.

use super::alloc::GraphAllocator;
use super::backend::Backend;
use super::{Backend as BackendTag, ComputeGraph, NodeId};

/// A contiguous subgraph executed on one backend.
#[derive(Debug, Clone)]
pub struct Split {
    pub backend: BackendTag,
    /// Node id range [start, end) in graph.nodes.
    pub node_range: (usize, usize),
    /// Nodes whose source values live on another backend (copied in).
    pub inputs: Vec<NodeId>,
    /// Nodes consumed by a later split on another backend (copied out).
    pub outputs: Vec<NodeId>,
}

impl Split {
    fn new(backend: BackendTag, start: usize) -> Self {
        Self { backend, node_range: (start, start), inputs: Vec::new(), outputs: Vec::new() }
    }
}

pub struct BackendScheduler;

impl Default for BackendScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl BackendScheduler {
    pub fn new() -> Self {
        Self
    }

    /// Assign every node to the best backend that supports it (capability
    /// driven via the allocator's backend registry; all-CPU for now).
    pub fn assign_backends(&self, graph: &mut ComputeGraph, alloc: &GraphAllocator) {
        for node in &mut graph.nodes {
            if node.backend.is_some() {
                continue; // keep explicit assignments
            }
            node.backend = alloc.supports(&node.op, node.out_dtype).or(Some(BackendTag::CPU));
        }
    }

    /// Partition the graph into contiguous same-backend splits.
    /// Cross-split inputs/outputs are derived from the src edges.
    pub fn split_graph(&self, graph: &ComputeGraph) -> Vec<Split> {
        let n = graph.n_nodes();
        let mut splits: Vec<Split> = Vec::new();
        let mut cur: Option<BackendTag> = None;
        let mut split = Split::new(BackendTag::CPU, 0);
        for id in 0..n {
            let node = graph.node(id);
            let b = node.backend.unwrap_or_else(|| cur.unwrap_or(BackendTag::CPU));
            if let Some(c) = cur {
                if b != c {
                    split.node_range.1 = id;
                    splits.push(std::mem::replace(&mut split, Split::new(b, id)));
                }
            } else {
                split = Split::new(b, id);
            }
            cur = Some(b);
        }
        split.node_range.1 = n;
        splits.push(split);

        // cross-split edges: a node's src on a different split's backend
        let mut split_of = vec![0usize; n];
        for (si, s) in splits.iter().enumerate() {
            for id in s.node_range.0..s.node_range.1 {
                split_of[id] = si;
            }
        }
        for si in 0..splits.len() {
            for id in splits[si].node_range.0..splits[si].node_range.1 {
                let node = graph.node(id);
                for &src in &node.src {
                    let src_split = split_of[src];
                    if src_split != si {
                        if !splits[si].inputs.contains(&src) {
                            splits[si].inputs.push(src);
                        }
                        if !splits[src_split].outputs.contains(&src) {
                            splits[src_split].outputs.push(src);
                        }
                    }
                }
            }
        }
        splits
    }

    /// Execute the graph split by split. With a single backend there is one
    /// split and copies are no-ops; the loop structure matches llama.cpp's
    /// `compute_splits` for when cross-backend transfers arrive (Phase 3).
    pub fn execute(
        &self,
        graph: &ComputeGraph,
        alloc: &mut GraphAllocator,
    ) -> Result<(), String> {
        let splits = self.split_graph(graph);
        let mut prev_backend: Option<BackendTag> = None;
        for split in &splits {
            if let Some(pb) = prev_backend {
                if pb != split.backend {
                    return Err(format!(
                        "cross-backend split ({pb:?} -> {:?}) needs Phase 3 transfer support",
                        split.backend
                    ));
                }
            }
            for id in split.node_range.0..split.node_range.1 {
                let node = graph.node(id);
                if node.is_input() {
                    continue;
                }
                let mut in_bufs = Vec::with_capacity(node.src.len());
                for &s in &node.src {
                    let br = alloc
                        .node_buffer(s)
                        .ok_or_else(|| format!("node {s} has no allocated buffer"))?;
                    in_bufs.push(br.id);
                }
                let br = alloc
                    .node_buffer(id)
                    .ok_or_else(|| format!("node {id} has no allocated buffer"))?;
                match split.backend {
                    BackendTag::CPU => alloc.cpu_mut().execute_node(node, &in_bufs, br.id)?,
                    other => {
                        return Err(format!(
                            "backend {other:?} execution needs Phase 3 (metal_backend)"
                        ))
                    }
                }
            }
            prev_backend = Some(split.backend);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::DType;

    fn small_graph() -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let o = b.add(s, x);
        b.output(o);
        b.build()
    }

    #[test]
    fn assign_all_cpu_and_single_split() {
        let mut g = small_graph();
        let sched = BackendScheduler::new();
        let alloc = GraphAllocator::new();
        sched.assign_backends(&mut g, &alloc);
        assert!(g.nodes.iter().all(|n| n.backend == Some(BackendTag::CPU)));
        let splits = sched.split_graph(&g);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].node_range, (0, g.n_nodes()));
        assert!(splits[0].inputs.is_empty());
        assert!(splits[0].outputs.is_empty());
    }

    #[test]
    fn split_on_backend_change() {
        // hand-assign: x(input) CPU, silu Metal, add CPU -> 3 splits
        let mut g = small_graph();
        g.nodes[0].backend = Some(BackendTag::CPU);
        g.nodes[1].backend = Some(BackendTag::Metal);
        g.nodes[2].backend = Some(BackendTag::CPU);
        let sched = BackendScheduler::new();
        let splits = sched.split_graph(&g);
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].backend, BackendTag::CPU);
        assert_eq!(splits[1].backend, BackendTag::Metal);
        assert_eq!(splits[2].backend, BackendTag::CPU);
        // cross-split edges: x (0) feeds the Metal silu; silu out (1) feeds the
        // CPU add
        assert_eq!(splits[0].outputs, vec![0]);
        assert_eq!(splits[1].inputs, vec![0]);
        assert_eq!(splits[1].outputs, vec![1]);
        // add(silu(x), x): consumes both the silu output (1) and x (0)
        assert_eq!(splits[2].inputs, vec![1, 0]);
    }

    #[test]
    fn execute_single_backend_graph() {
        let g = small_graph();
        let sched = BackendScheduler::new();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        sched.execute(&g, &mut alloc).unwrap();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let got = alloc.get_buffer(&g, 2).unwrap();
        for i in 0..4 {
            assert!((got[i] - (silu((i + 1) as f32) + (i + 1) as f32)).abs() < 1e-5);
        }
    }

    #[test]
    fn cross_backend_execute_rejected_for_now() {
        let mut g = small_graph();
        g.nodes[1].backend = Some(BackendTag::Metal); // would need Phase 3 transfer
        let sched = BackendScheduler::new();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(sched.execute(&g, &mut alloc).is_err());
    }
}
