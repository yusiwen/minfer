//! Liveness-based per-backend buffer allocator (Phase 1: CPU pool).
//!
//! Mirrors llama.cpp's `ggml_gallocr`: buffers are shared between nodes whose
//! live ranges do not overlap, and persistent regions (KV cache) are allocated
//! once and never freed. KV positions (`n_past`) never influence allocation —
//! the KV region is sized by `n_ctx` at build time.

use std::collections::HashMap;

use super::ops::Op;
use super::{Backend, BufRef, ComputeGraph, NodeId, PersistentBuf};

/// A CPU pool buffer.
struct CpuBuf {
    data: Vec<f32>,
    /// Last node-exec index (in topo order) at which this buffer is still live.
    alive_until: usize,
    in_use: bool,
}

/// Per-backend liveness allocator.
pub struct GraphAllocator {
    cpu_buffers: Vec<CpuBuf>,
    node_to_buf: HashMap<NodeId, BufRef>,
    /// KV persistent region per layer (sized n_embd * n_ctx; the V region is a
    /// Phase 5 executor concern and lives adjacent to K in the same region).
    kv: HashMap<usize, BufRef>,
    /// All persistent regions (never freed).
    pub persistent: Vec<PersistentBuf>,
}

impl Default for GraphAllocator {
    fn default() -> Self {
        Self {
            cpu_buffers: Vec::new(),
            node_to_buf: HashMap::new(),
            kv: HashMap::new(),
            persistent: Vec::new(),
        }
    }
}

impl GraphAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Liveness analysis + allocation for every node buffer.
    pub fn alloc_graph(&mut self, graph: &ComputeGraph) -> Result<(), String> {
        self.node_to_buf.clear();
        let order = graph.topo_order()?;
        let n = graph.n_nodes();

        // exec index per node
        let mut exec = vec![0usize; n];
        for (i, &id) in order.iter().enumerate() {
            exec[id] = i;
        }

        // last use = max exec index over consumers (own index if none);
        // outputs stay live until the end
        let mut last_use = exec.clone();
        for (i, &id) in order.iter().enumerate() {
            let node = graph.node(id);
            for &s in &node.src {
                if last_use[s] < i {
                    last_use[s] = i;
                }
            }
        }
        for &o in &graph.outputs {
            last_use[o] = order.len();
        }

        for (i, &id) in order.iter().enumerate() {
            // free buffers whose liveness ended before this node
            self.sweep(i);
            let node = graph.node(id);
            match node.op {
                Op::KvcacheStore { layer } | Op::KvcacheLoad { layer } => {
                    // persistent view — no liveness allocation
                    let br = self.ensure_kv(layer, node.n_elements());
                    self.node_to_buf.insert(id, br);
                }
                _ => {
                    if last_use[id] > i {
                        // produce a live value: allocate (or reuse a freed one)
                        let size = node.n_elements();
                        let br = self.alloc_cpu(size, last_use[id]);
                        self.node_to_buf.insert(id, br);
                    }
                }
            }
        }
        Ok(())
    }

    /// Buffer handle for a node (None if the node is dead / not allocated).
    pub fn node_buffer(&self, id: NodeId) -> Option<BufRef> {
        self.node_to_buf.get(&id).copied()
    }

    /// KV persistent region for a layer; created on first use.
    fn ensure_kv(&mut self, layer: usize, size: usize) -> BufRef {
        if let Some(&br) = self.kv.get(&layer) {
            return br;
        }
        let br = self.alloc_persistent(&format!("kv.{layer}"), size);
        self.kv.insert(layer, br);
        br
    }

    /// Allocate a persistent (never-freed) CPU region.
    pub fn alloc_persistent(&mut self, name: &str, size: usize) -> BufRef {
        let id = self.cpu_buffers.len();
        self.cpu_buffers.push(CpuBuf {
            data: vec![0.0f32; size],
            alive_until: usize::MAX,
            in_use: true,
        });
        self.persistent.push(PersistentBuf {
            name: name.to_string(),
            backend: Backend::CPU,
            id,
        });
        BufRef { backend: Backend::CPU, id }
    }

    /// Allocate a liveness-tracked CPU buffer of exactly `size` elements,
    /// reusing a freed buffer when one matches.
    fn alloc_cpu(&mut self, size: usize, alive_until: usize) -> BufRef {
        if let Some(id) = self
            .cpu_buffers
            .iter()
            .position(|b| !b.in_use && b.data.len() == size)
        {
            let b = &mut self.cpu_buffers[id];
            b.in_use = true;
            b.alive_until = alive_until;
            return BufRef { backend: Backend::CPU, id };
        }
        let id = self.cpu_buffers.len();
        self.cpu_buffers.push(CpuBuf {
            data: vec![0.0f32; size],
            alive_until,
            in_use: true,
        });
        BufRef { backend: Backend::CPU, id }
    }

    /// Mark buffers whose liveness ended before exec index `i` as reusable.
    fn sweep(&mut self, i: usize) {
        for b in &mut self.cpu_buffers {
            if b.in_use && b.alive_until < i {
                b.in_use = false;
            }
        }
    }

    /// Fill an input node's buffer from host data (CPU path).
    pub fn fill_input(&mut self, graph: &ComputeGraph, name: &str, data: &[f32]) -> Result<(), String> {
        let id = graph
            .inputs
            .iter()
            .copied()
            .find(|&i| graph.node(i).name == name)
            .ok_or_else(|| format!("no input node named '{name}'"))?;
        let br = self
            .node_buffer(id)
            .ok_or_else(|| format!("input '{name}' has no buffer (not allocated)"))?;
        if br.backend != Backend::CPU {
            return Err(format!("input '{name}' is on {:?}, not CPU", br.backend));
        }
        let buf = &mut self.cpu_buffers[br.id].data;
        let want = buf.len();
        if data.len() != want {
            return Err(format!(
                "input '{name}': expected {want} elements, got {}",
                data.len()
            ));
        }
        buf.copy_from_slice(data);
        Ok(())
    }

    /// Host view of a CPU node's buffer.
    pub fn get_buffer(&self, graph: &ComputeGraph, id: NodeId) -> Option<&[f32]> {
        let br = self.node_buffer(id)?;
        if br.backend != Backend::CPU {
            return None;
        }
        Some(&self.cpu_buffers[br.id].data)
    }

    /// Host view of a persistent region by name.
    pub fn get_persistent(&self, name: &str) -> Option<&[f32]> {
        self.persistent
            .iter()
            .find(|p| p.name == name)
            .map(|p| self.cpu_buffers[p.id].data.as_slice())
    }

    /// Number of distinct CPU buffers currently allocated (for tests).
    pub fn n_cpu_buffers(&self) -> usize {
        self.cpu_buffers.len()
    }

    /// Number of distinct buffers actually mapped to nodes (for tests).
    pub fn n_mapped_buffers(&self) -> usize {
        let mut s: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for br in self.node_to_buf.values() {
            if br.backend == Backend::CPU {
                s.insert(br.id);
            }
        }
        s.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;

    fn chain(n_ops: usize) -> ComputeGraph {
        // input x -> silu -> add(x) -> silu -> add(x) -> ...
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        let mut h = x;
        for i in 0..n_ops {
            h = if i % 2 == 0 { b.silu(h) } else { b.add(h, x) };
        }
        b.output(h);
        b.build()
    }

    #[test]
    fn liveness_reuses_buffers_along_chain() {
        let g = chain(6);
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        // 6 ops + 1 input = 7 nodes; liveness reuse must keep distinct
        // buffers well below the node count
        assert!(alloc.n_mapped_buffers() < g.n_nodes(), "expected reuse, got {} buffers for {} nodes", alloc.n_mapped_buffers(), g.n_nodes());
        // every live node got a buffer
        for id in 0..g.n_nodes() {
            assert!(alloc.node_buffer(id).is_some(), "node {id} missing buffer");
        }
    }

    #[test]
    fn parallel_chains_do_not_share() {
        // two independent chains: more buffers than a single chain
        let mut b = GraphBuilder::new();
        let a0 = b.input("a0", [4, 1, 1, 1], crate::graph::DType::F32);
        let b0 = b.input("b0", [4, 1, 1, 1], crate::graph::DType::F32);
        let a1 = b.silu(a0);
        let b1 = b.silu(b0);
        let a2 = b.add(a1, a0);
        let b2 = b.add(b1, b0);
        let out = b.add(a2, b2);
        b.output(out);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        let n_chain6 = {
            let g2 = chain(6);
            let mut al = GraphAllocator::new();
            al.alloc_graph(&g2).unwrap();
            al.n_mapped_buffers()
        };
        assert!(alloc.n_mapped_buffers() > n_chain6, "parallel chains should not share");
    }

    #[test]
    fn fill_and_read_input() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert_eq!(alloc.get_buffer(&g, x).unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        // wrong size rejected
        assert!(alloc.fill_input(&g, "x", &[1.0, 2.0]).is_err());
        // unknown input rejected
        assert!(alloc.fill_input(&g, "nope", &[]).is_err());
    }

    #[test]
    fn kv_region_is_persistent_and_shared() {
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, pos, 1024);
        let load = b.kvcache_load(0, 16, 1024, 2);
        b.output(load);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        // store and load share one persistent region, never liveness-freed
        assert_eq!(alloc.node_buffer(3), alloc.node_buffer(4));
        assert_eq!(alloc.persistent.len(), 1);
        assert_eq!(alloc.persistent[0].name, "kv.0");
        assert_eq!(alloc.get_persistent("kv.0").unwrap().len(), 16 * 1024);
        // mapped buffers: positions/k/v (3 liveness) + kv region (shared) = 4
        assert_eq!(alloc.n_mapped_buffers(), 4);
    }

    #[test]
    fn cycle_graph_allocation_fails() {
        let mut g = ComputeGraph::default();
        g.nodes.push(super::super::CNode {
            id: 0, name: "a".into(), op: Op::Add, src: vec![1],
            out_shape: [1, 1, 1, 1], out_dtype: super::super::DType::F32,
            backend: None, meta: super::super::ops::NodeMeta::None,
        });
        g.nodes.push(super::super::CNode {
            id: 1, name: "b".into(), op: Op::Add, src: vec![0],
            out_shape: [1, 1, 1, 1], out_dtype: super::super::DType::F32,
            backend: None, meta: super::super::ops::NodeMeta::None,
        });
        let mut alloc = GraphAllocator::new();
        assert!(alloc.alloc_graph(&g).is_err());
    }
}
