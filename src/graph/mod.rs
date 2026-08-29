//! Declarative compute graph IR (Phase 1).
//!
//! Mirrors llama.cpp's `ggml_cgraph` + `llm_graph_context` layering; see
//! docs/GRAPH-REFACTOR-PLAN.md. The IR is pure data: building a
//! `ComputeGraph` is side-effect free, and execution is delegated to the
//! backend scheduler (Phase 4).

pub mod alloc;
pub mod backend;
pub mod builder;
pub mod cache;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
pub mod dot;
pub mod fusion;
pub mod json;
#[cfg(target_os = "macos")]
pub mod metal_backend;
pub mod ops;
pub mod params;
pub mod scheduler;

use ops::{NodeMeta, Op};

/// Node index in `ComputeGraph::nodes`.
pub type NodeId = usize;

/// Activation dtype carried by node outputs (weight tensors keep their own
/// `TensorType`; IR activations are F32 except where noted). F16 / Q8_0 are
/// part of the full IR dtype vocabulary (ggml parity) but no supported op
/// constructs them today.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    I32,
    Q8_0,
}

#[allow(dead_code)]
impl DType {
    /// Bytes per element (Q8_0 = 1 byte per quantized element block member;
    /// actual block layout is a backend concern).
    pub fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::I32 => 4,
            DType::Q8_0 => 1,
        }
    }
}

/// Execution backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Backend {
    CPU,
    #[cfg_attr(not(target_os = "macos"), allow(dead_code))]
    Metal,
    /// CUDA backend (Phase 7, `--features cuda`).
    #[allow(dead_code)]
    Cuda,
}

/// Backend buffer handle: a buffer id inside a specific backend's pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufRef {
    pub backend: Backend,
    pub id: usize,
}

/// Persistent (never-freed) region — KV cache, etc.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PersistentBuf {
    pub name: String,
    pub backend: Backend,
    pub id: usize,
}

/// Single compute node.
#[derive(Debug, Clone)]
pub struct CNode {
    pub id: NodeId,
    pub name: String,
    pub op: Op,
    pub src: Vec<NodeId>,
    pub out_shape: [usize; 4],
    pub out_dtype: DType,
    /// Backend assigned by the scheduler (Phase 4); None = undecided.
    pub backend: Option<Backend>,
    pub meta: NodeMeta,
}

impl CNode {
    pub fn n_elements(&self) -> usize {
        self.out_shape.iter().product()
    }
    pub fn is_input(&self) -> bool {
        matches!(self.op, Op::Input)
    }
}

/// Compute graph: topologically ordered node sequence + input/output sets.
#[derive(Debug, Clone, Default)]
pub struct ComputeGraph {
    pub nodes: Vec<CNode>,
    pub inputs: Vec<NodeId>,
    pub outputs: Vec<NodeId>,
    /// Graph identifier for reuse detection (CUDA Graph caching etc.).
    /// Populated by `GraphCache::replace_graph` (monotonic per process);
    /// a reused graph keeps its uid.
    pub uid: u64,
}

impl ComputeGraph {
    pub fn node(&self, id: NodeId) -> &CNode {
        &self.nodes[id]
    }
    /// Mutable node access (fusion pass / debug tooling).
    #[allow(dead_code)]
    pub fn node_mut(&mut self, id: NodeId) -> &mut CNode {
        &mut self.nodes[id]
    }
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    /// Element count of a node output (assertion / debug helper).
    #[allow(dead_code)]
    pub fn n_elements(&self, id: NodeId) -> usize {
        self.nodes[id].n_elements()
    }

    /// Kahn topological sort. The builder appends sources before consumers, so
    /// `nodes` is already topologically ordered; this validates the invariant
    /// and returns a stable order (used by the allocator).
    pub fn topo_order(&self) -> Result<Vec<NodeId>, String> {
        let n = self.nodes.len();
        let mut indeg = vec![0usize; n];
        for node in &self.nodes {
            if node.id >= n {
                return Err(format!("node id {} out of range", node.id));
            }
            for &s in &node.src {
                if s >= n {
                    return Err(format!("node {}: src {s} out of range", node.id));
                }
                indeg[node.id] += 1;
            }
        }
        let mut queue: Vec<NodeId> = (0..n).filter(|&i| indeg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        let mut qi = 0;
        while qi < queue.len() {
            let u = queue[qi];
            qi += 1;
            order.push(u);
            for v in 0..n {
                if self.nodes[v].src.contains(&u) {
                    indeg[v] -= 1;
                    if indeg[v] == 0 {
                        queue.push(v);
                    }
                }
            }
        }
        if order.len() != n {
            return Err(format!(
                "cycle detected: {}/{} nodes ordered",
                order.len(),
                n
            ));
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;

    fn chain_graph(n_ops: usize) -> ComputeGraph {
        // input -> silu -> add -> silu -> add -> ... (linear chain)
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let mut h = x;
        for i in 0..n_ops {
            h = if i % 2 == 0 { b.silu(h) } else { b.add(h, x) };
        }
        b.output(h);
        b.build()
    }

    #[test]
    fn topo_order_validates_chain() {
        let g = chain_graph(4);
        let order = g.topo_order().unwrap();
        assert_eq!(order.len(), g.n_nodes());
        // every node appears before its consumers
        for node in &g.nodes {
            for &s in &node.src {
                let ps = order.iter().position(|&x| x == s).unwrap();
                let pn = order.iter().position(|&x| x == node.id).unwrap();
                assert!(ps < pn, "src {} must precede consumer {}", s, node.id);
            }
        }
    }

    #[test]
    fn topo_order_detects_cycle() {
        // hand-built cycle: a -> b -> a
        let mut g = ComputeGraph::default();
        g.nodes.push(CNode {
            id: 0,
            name: "a".into(),
            op: Op::Add,
            src: vec![1],
            out_shape: [1, 1, 1, 1],
            out_dtype: DType::F32,
            backend: None,
            meta: NodeMeta::None,
        });
        g.nodes.push(CNode {
            id: 1,
            name: "b".into(),
            op: Op::Add,
            src: vec![0],
            out_shape: [1, 1, 1, 1],
            out_dtype: DType::F32,
            backend: None,
            meta: NodeMeta::None,
        });
        assert!(g.topo_order().is_err());
    }

    #[test]
    fn op_partial_eq_compares_payloads() {
        assert_eq!(Op::RmsNorm { eps: 1e-5 }, Op::RmsNorm { eps: 1e-5 });
        assert_ne!(Op::RmsNorm { eps: 1e-5 }, Op::RmsNorm { eps: 1e-6 });
        assert_ne!(
            Op::MatMul { transpose_b: true },
            Op::MatMul { transpose_b: false }
        );
        assert_eq!(Op::KvcacheLoad { layer: 2 }, Op::KvcacheLoad { layer: 2 });
        assert_ne!(Op::KvcacheLoad { layer: 2 }, Op::KvcacheLoad { layer: 3 });
    }

    #[test]
    fn dtype_size() {
        assert_eq!(DType::F32.size(), 4);
        assert_eq!(DType::F16.size(), 2);
        assert_eq!(DType::I32.size(), 4);
        assert_eq!(DType::Q8_0.size(), 1);
    }
}
