//! Backend scheduler (Phase 3→4: assign → split → execute).
//!
//! Mirrors llama.cpp's `ggml_backend_sched_split_graph` + `compute_splits`:
//! - `assign_backends`: per-op capability-driven assignment
//! - `split_graph`: partition into contiguous same-backend splits, deriving
//!   cross-split inputs/outputs
//! - `execute`: per split — sync the previous backend, copy split inputs
//!   across backends, run the nodes (each backend batches its ops into one
//!   command buffer, flushed at the boundary via `synchronize`), then a final
//!   sync.
//!
//! Nodes execute in **build order** (the builder appends sources before
//! consumers, so this is a valid topological order — matching ggml, which
//! executes `nodes[0..n_nodes]` in order). This guarantees e.g. that a KV
//! store node executes before the attention that reads the KV view.

use super::alloc::GraphAllocator;
use super::backend::{Backend, KvProvider};
use super::ops::{NodeMeta, Op};
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
    /// driven via the allocator's backend registry).
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

    /// Execute the graph split by split (llama.cpp `compute_splits` shape).
    pub fn execute(
        &self,
        graph: &ComputeGraph,
        alloc: &mut GraphAllocator,
    ) -> Result<(), String> {
        #[cfg(debug_assertions)]
        debug_assert!(graph.topo_order().is_ok(), "graph is not a valid DAG");
        let splits = self.split_graph(graph);
        if std::env::var("MINFER_GRAPH_TRACE").is_ok() {
            for (si, s) in splits.iter().enumerate() {
                eprintln!("[graph] split {si}: {:?} nodes {}-{}", s.backend, s.node_range.0, s.node_range.1);
            }
            let mut by = std::collections::BTreeMap::new();
            for n in &graph.nodes {
                *by.entry((format!("{:?}", n.op).split('{').next().unwrap().to_string(), n.backend.map(|b| format!("{b:?}")).unwrap_or_default()))
                    .or_insert(0usize) += 1;
            }
            for (k, v) in by {
                eprintln!("[graph] op {:<12} backend {:<7} x{v}", k.0, k.1);
            }
        }
        let mut prev_backend: Option<BackendTag> = None;
        for split in &splits {
            if let Some(pb) = prev_backend {
                if pb != split.backend {
                    // 1. flush the previous backend's async work
                    alloc.sync_backend(pb);
                    // 2. copy this split's inputs across backends
                    for &inp in &split.inputs {
                        alloc.copy_across(inp, split.backend)?;
                    }
                }
            }
            for id in split.node_range.0..split.node_range.1 {
                let node = graph.node(id);
                if node.is_input() {
                    continue; // data pre-filled by the allocator
                }
                // dead nodes (no consumers, not outputs) get no buffer — the
                // fusion pass can orphan them (e.g. silu folded into SwiGLU);
                // they are skipped, not executed
                let Some(br) = alloc.node_buffer(id) else { continue };
                if br.backend != split.backend {
                    return Err(format!(
                        "node {id} buffer on {:?} but executing split is {:?} (assignment/alloc mismatch)",
                        br.backend, split.backend
                    ));
                }
                let mut in_bufs = Vec::with_capacity(node.src.len());
                for &s in &node.src {
                    let sbr = alloc
                        .node_buffer(s)
                        .ok_or_else(|| format!("node {s} has no allocated buffer"))?;
                    in_bufs.push(sbr.id);
                }
                // resolve the layer's KV region pair BEFORE the mutable backend
                // borrow (the backend needs it for KV store / attention)
                let kv_pair = match &node.op {
                    Op::KvcacheStore { layer } => alloc.kv_pair(*layer),
                    Op::FusedQKV { layer } => alloc.kv_pair(*layer),
                    Op::Attn { .. } => match &node.meta {
                        NodeMeta::Attn(m) => alloc.kv_pair(m.layer),
                        _ => None,
                    },
                    _ => None,
                };
                // NOTE: execution follows node id order (build order), which is
                // the graph's topological order by construction.
                match split.backend {
                    BackendTag::CPU => alloc.cpu_mut().execute_node(node, &in_bufs, br.id, kv_pair)?,
                    #[cfg(target_os = "macos")]
                    BackendTag::Metal => {
                        let m = alloc
                            .metal_mut()
                            .ok_or("Metal backend not enabled")?;
                        m.execute_node(node, &in_bufs, br.id, kv_pair)?;
                    }
                    #[cfg(not(target_os = "macos"))]
                    BackendTag::Metal => return Err("Metal unavailable".into()),
                    BackendTag::Cuda => return Err("CUDA backend not implemented".into()),
                }
            }
            prev_backend = Some(split.backend);
        }
        if let Some(pb) = prev_backend {
            alloc.sync_backend(pb);
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
        assert_eq!(splits[0].outputs, vec![0]);
        assert_eq!(splits[1].inputs, vec![0]);
        assert_eq!(splits[1].outputs, vec![1]);
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
}
