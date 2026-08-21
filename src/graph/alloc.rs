//! Liveness-based per-backend buffer allocator (Phase 1→2).
//!
//! Mirrors llama.cpp's `ggml_gallocr`: buffers are shared between nodes whose
//! live ranges do not overlap, and persistent regions (KV cache) are allocated
//! once and never freed. KV positions (`n_past`) never influence allocation —
//! the KV region is sized by `n_ctx` at build time.
//!
//! Since Phase 2 the storage lives in the backends' own pools (`CpuBackend`);
//! the allocator only tracks liveness and the node → buffer mapping.

use std::collections::HashMap;

use super::backend::Backend as BackendTrait;
use super::cpu_backend::CpuBackend;
use super::ops::Op;
use super::{Backend, BufRef, ComputeGraph, NodeId, PersistentBuf};

/// Per-backend liveness allocator.
pub struct GraphAllocator {
    cpu: CpuBackend,
    node_to_buf: HashMap<NodeId, BufRef>,
    /// cpu pool id → last exec index it stays alive until
    buf_alive: HashMap<usize, usize>,
    /// KV persistent region per layer (layout `[K | V]`, each n_embd*n_ctx)
    kv: HashMap<usize, BufRef>,
    /// All persistent regions (never freed).
    pub persistent: Vec<PersistentBuf>,
}

impl Default for GraphAllocator {
    fn default() -> Self {
        Self {
            cpu: CpuBackend::new(),
            node_to_buf: HashMap::new(),
            buf_alive: HashMap::new(),
            kv: HashMap::new(),
            persistent: Vec::new(),
        }
    }
}

impl GraphAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// The CPU backend (register weights here / host access).
    pub fn cpu(&self) -> &CpuBackend {
        &self.cpu
    }

    /// Mutable CPU backend (weight registration, execution).
    pub fn cpu_mut(&mut self) -> &mut CpuBackend {
        &mut self.cpu
    }

    /// Register a weight tensor by name (delegates to the CPU backend).
    pub fn register_weight(&mut self, name: &str, t: crate::tensor::Tensor) {
        self.cpu.register_weight(name, t);
    }

    /// Which backend supports this op/dtype (highest priority first).
    /// With only the CPU backend registered this always yields CPU or None.
    pub fn supports(&self, op: &Op, dtype: crate::graph::DType) -> Option<Backend> {
        if self.cpu.supports_op(op, dtype) {
            return Some(Backend::CPU);
        }
        None
    }

    /// Liveness analysis + allocation for every node buffer.
    ///
    /// Runs on every graph (re)build: previous liveness buffers are released
    /// back to the pool, while **persistent regions (KV cache) survive** — they
    /// are the KV cache and must persist across prefill→decode rebuilds.
    pub fn alloc_graph(&mut self, graph: &ComputeGraph) -> Result<(), String> {
        // release previous liveness buffers (persistent/KV regions are not in
        // buf_alive and stay untouched)
        let prev: Vec<usize> = self.buf_alive.keys().copied().collect();
        for id in prev {
            self.cpu.free_buffer(id);
        }
        self.buf_alive.clear();
        self.node_to_buf.clear();
        let order = graph.topo_order()?;
        let n = graph.n_nodes();

        // exec index per node
        let mut exec = vec![0usize; n];
        for (i, &id) in order.iter().enumerate() {
            exec[id] = i;
        }

        // last use = max exec index over consumers; outputs stay live to the end
        let mut last_use = exec.clone();
        for (i, &id) in order.iter().enumerate() {
            for &s in &graph.node(id).src {
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
            match graph.node(id).op {
                Op::KvcacheStore { layer } | Op::KvcacheLoad { layer } => {
                    // persistent view — no liveness allocation
                    let br = self.ensure_kv(layer, graph.node(id).n_elements());
                    self.node_to_buf.insert(id, br);
                }
                _ => {
                    if last_use[id] > i {
                        // produce a live value: allocate (or reuse a freed one)
                        let size = graph.node(id).n_elements();
                        let pid = self.cpu.alloc_buffer(size);
                        self.buf_alive.insert(pid, last_use[id]);
                        self.node_to_buf.insert(id, BufRef { backend: Backend::CPU, id: pid });
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
    /// Layout: `[K (n_embd*n_ctx) | V (n_embd*n_ctx)]` — contiguous so the
    /// attention kernel can view K and V from one buffer.
    fn ensure_kv(&mut self, layer: usize, size: usize) -> BufRef {
        if let Some(&br) = self.kv.get(&layer) {
            return br;
        }
        let pid = self.cpu.alloc_buffer(2 * size);
        self.persistent.push(PersistentBuf {
            name: format!("kv.{layer}"),
            backend: Backend::CPU,
            id: pid,
        });
        let br = BufRef { backend: Backend::CPU, id: pid };
        self.kv.insert(layer, br);
        br
    }

    /// Allocate a persistent (never-freed) CPU region.
    pub fn alloc_persistent(&mut self, name: &str, size: usize) -> BufRef {
        let pid = self.cpu.alloc_buffer(size);
        self.persistent.push(PersistentBuf {
            name: name.to_string(),
            backend: Backend::CPU,
            id: pid,
        });
        BufRef { backend: Backend::CPU, id: pid }
    }

    /// Mark buffers whose liveness ended before exec index `i` as reusable.
    fn sweep(&mut self, i: usize) {
        let expired: Vec<usize> = self
            .buf_alive
            .iter()
            .filter(|(_, &al)| al < i)
            .map(|(&id, _)| id)
            .collect();
        for id in expired {
            self.buf_alive.remove(&id);
            self.cpu.free_buffer(id);
        }
    }

    /// Fill an input node's buffer from host data (CPU path).
    pub fn fill_input(
        &mut self,
        graph: &ComputeGraph,
        name: &str,
        data: &[f32],
    ) -> Result<(), String> {
        self.fill_input_impl(graph, name, data)
    }

    /// Fill an I32 input (token ids / positions). Stored as `f32::from_bits`
    /// patterns — exact for |v| < 2^24.
    pub fn fill_input_i32(
        &mut self,
        graph: &ComputeGraph,
        name: &str,
        data: &[u32],
    ) -> Result<(), String> {
        let bits: Vec<f32> = data.iter().map(|&v| f32::from_bits(v)).collect();
        self.fill_input_impl(graph, name, &bits)
    }

    fn fill_input_impl(
        &mut self,
        graph: &ComputeGraph,
        name: &str,
        data: &[f32],
    ) -> Result<(), String> {
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
        self.cpu.write_host(br.id, data)
    }

    /// Host view of a CPU node's buffer.
    pub fn get_buffer(&self, graph: &ComputeGraph, id: NodeId) -> Option<&[f32]> {
        let br = self.node_buffer(id)?;
        if br.backend != Backend::CPU {
            return None;
        }
        self.cpu.read_host(br.id)
    }

    /// Host view of a persistent region by name.
    pub fn get_persistent(&self, name: &str) -> Option<&[f32]> {
        self.persistent
            .iter()
            .find(|p| p.name == name)
            .and_then(|p| self.cpu.read_host(p.id))
    }

    /// Number of distinct CPU buffers currently allocated (for tests).
    pub fn n_cpu_buffers(&self) -> usize {
        self.cpu.pool_len()
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
        assert!(
            alloc.n_mapped_buffers() < g.n_nodes(),
            "expected reuse, got {} buffers for {} nodes",
            alloc.n_mapped_buffers(),
            g.n_nodes()
        );
        for id in 0..g.n_nodes() {
            assert!(alloc.node_buffer(id).is_some(), "node {id} missing buffer");
        }
    }

    #[test]
    fn parallel_chains_do_not_share() {
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
        assert!(alloc.fill_input(&g, "x", &[1.0, 2.0]).is_err());
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
        // [K | V] contiguous: 2 * n_embd * n_ctx
        assert_eq!(alloc.get_persistent("kv.0").unwrap().len(), 2 * 16 * 1024);
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
