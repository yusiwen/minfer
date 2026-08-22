//! Liveness-based per-backend buffer allocator (Phase 1→3).
//!
//! Mirrors llama.cpp's `ggml_gallocr`: buffers are shared between nodes whose
//! live ranges do not overlap, and persistent regions (KV cache) are allocated
//! once and never freed. KV positions (`n_past`) never influence allocation.
//!
//! Storage lives in the backends' own pools (CPU `Vec<f32>`, Metal shared
//! MTLBuffers); the allocator tracks liveness, the node → buffer mapping, and
//! the per-layer KV regions (each layer owns TWO persistent regions: K and V —
//! symmetric across backends). Persistent regions survive graph rebuilds.

use std::collections::HashMap;

use super::backend::Backend as BackendTrait;
use super::backend::KvProvider;
use super::cpu_backend::CpuBackend;
use super::ops::{NodeMeta, Op};
use super::{Backend, BufRef, ComputeGraph, NodeId, PersistentBuf};

/// Per-backend liveness allocator.
pub struct GraphAllocator {
    cpu: CpuBackend,
    #[cfg(target_os = "macos")]
    metal: Option<super::metal_backend::MetalBackend>,
    node_to_buf: HashMap<NodeId, BufRef>,
    /// (backend, pool id) → last exec index it stays alive until
    buf_alive: HashMap<(Backend, usize), usize>,
    /// per-layer KV persistent regions: [k, v]
    kv: HashMap<usize, [BufRef; 2]>,
    /// All persistent regions (never freed).
    pub persistent: Vec<PersistentBuf>,
}

impl Default for GraphAllocator {
    fn default() -> Self {
        Self {
            cpu: CpuBackend::new(),
            #[cfg(target_os = "macos")]
            metal: None,
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

    /// Enable the Metal backend (called when MPS is initialized and the model
    /// weights are GPU-registered).
    #[cfg(target_os = "macos")]
    pub fn enable_metal(&mut self) -> bool {
        if self.metal.is_none() {
            self.metal = super::metal_backend::MetalBackend::new();
        }
        self.metal.is_some()
    }

    /// The CPU backend (register weights here / host access).
    pub fn cpu(&self) -> &CpuBackend {
        &self.cpu
    }

    /// Mutable CPU backend (weight registration, execution).
    pub fn cpu_mut(&mut self) -> &mut CpuBackend {
        &mut self.cpu
    }

    /// Mutable Metal backend (None until enabled / MPS unavailable).
    #[cfg(target_os = "macos")]
    pub fn metal_mut(&mut self) -> Option<&mut super::metal_backend::MetalBackend> {
        self.metal.as_mut()
    }

    /// Immutable Metal backend.
    #[cfg(target_os = "macos")]
    pub fn metal(&self) -> Option<&super::metal_backend::MetalBackend> {
        self.metal.as_ref()
    }

    /// Register a weight tensor by name (delegates to the CPU backend).
    pub fn register_weight(&mut self, name: &str, t: crate::tensor::Tensor) {
        self.cpu.register_weight(name, t);
    }

    /// Which backend supports this op/dtype (highest priority first).
    pub fn supports(&self, op: &Op, dtype: crate::graph::DType) -> Option<Backend> {
        #[cfg(target_os = "macos")]
        if let Some(m) = &self.metal {
            if m.supports_op(op, dtype) {
                return Some(Backend::Metal);
            }
        }
        if self.cpu.supports_op(op, dtype) {
            return Some(Backend::CPU);
        }
        None
    }

    /// Liveness analysis + allocation for every node buffer.
    ///
    /// Runs on every graph (re)build: previous liveness buffers are released
    /// back to their pools, while **persistent regions (KV cache) survive** —
    /// they are the KV cache and must persist across prefill→decode rebuilds.
    pub fn alloc_graph(&mut self, graph: &ComputeGraph) -> Result<(), String> {
        let prev: Vec<(Backend, usize)> = self.buf_alive.keys().copied().collect();
        for (b, id) in prev {
            self.free_in_pool(b, id);
        }
        self.buf_alive.clear();
        self.node_to_buf.clear();

        // The scheduler executes nodes in BUILD order (node id order — the
        // builder appends sources before consumers), so liveness must use the
        // same order: topo_order() can reorder srcless nodes (kv_load) ahead,
        // which would let a later consumer's buffer reuse clobber an input the
        // scheduler has not yet read (G3 tail get_rows regression). Validate
        // acyclicity, but keep build order.
        graph.topo_order()?;
        let order: Vec<NodeId> = (0..graph.n_nodes()).collect();
        let n = graph.n_nodes();

        let mut exec = vec![0usize; n];
        for (i, &id) in order.iter().enumerate() {
            exec[id] = i;
        }
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
        // Inputs are filled on the host BEFORE execution starts, so every
        // input buffer is live at fill time; liveness (which tracks execution
        // order) must never reuse an input's buffer for another input — the
        // later fill would clobber the earlier one. Treat inputs like outputs.
        for &i in &graph.inputs {
            last_use[i] = order.len();
        }

        // consumer counts (for in-place alias safety: an input may only be
        // overwritten in place when this op is its ONLY consumer)
        let mut n_consumers = vec![0usize; n];
        for node in &graph.nodes {
            for &s in &node.src {
                n_consumers[s] += 1;
            }
        }

        for (i, &id) in order.iter().enumerate() {
            self.sweep(i);
            let node = graph.node(id);
            let backend = node.backend.unwrap_or(Backend::CPU);
            match node.op {
                Op::KvcacheStore { layer } | Op::KvcacheLoad { layer } => {
                    let pair = self.ensure_kv(layer, backend, node.n_elements());
                    // the node's buffer = the K region
                    self.node_to_buf.insert(id, pair[0]);
                }
                Op::FusedQKV { layer } => {
                    // fused decode QKV: also needs the layer's persistent KV
                    // regions (the kernel stores K/V), but its output is a
                    // normal concat buffer (q|k|v), not the K region.
                    let kv_elems = match &node.meta {
                        NodeMeta::FusedQkv(m) => m.kv_elems,
                        _ => node.n_elements(),
                    };
                    self.ensure_kv(layer, backend, kv_elems);
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size);
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef { backend, id: pid });
                    }
                }
                Op::Silu | Op::RoPE { .. } => {
                    // In-place elementwise transforms: alias the input buffer
                    // (llama.cpp executes rope/silu in place). Same-backend
                    // aliasing avoids a host-side copy between a pending GPU
                    // producer and this kernel — it reads/writes the buffer the
                    // producer wrote, in kernel order. Cross-backend inputs get
                    // a fresh buffer: the producer completed before the split
                    // boundary, so the backend's host copy is safe there.
                    if last_use[id] > i {
                        let in_ref = self
                            .node_to_buf
                            .get(&node.src[0])
                            .copied()
                            .ok_or_else(|| format!("in-place op src buffer missing (node {id})"))?;
                        // alias only when the input's sole consumer is this op
                        // (in-place overwrites the input) AND it is on the same
                        // backend
                        if in_ref.backend == backend && n_consumers[node.src[0]] == 1 {
                            self.node_to_buf.insert(id, in_ref);
                            // the aliased input must stay alive through this
                            // node's consumers
                            last_use[node.src[0]] = last_use[node.src[0]].max(last_use[id]);
                        } else {
                            let size = node.n_elements();
                            let pid = self.alloc_in_pool(backend, size);
                            self.buf_alive.insert((backend, pid), last_use[id]);
                            self.node_to_buf.insert(id, BufRef { backend, id: pid });
                        }
                    }
                }
                _ => {
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size);
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef { backend, id: pid });
                    }
                }
            }
        }
        Ok(())
    }

    fn alloc_in_pool(&mut self, backend: Backend, size: usize) -> usize {
        match backend {
            Backend::CPU => self.cpu.alloc_buffer(size),
            #[cfg(target_os = "macos")]
            Backend::Metal => self
                .metal
                .as_mut()
                .expect("Metal pool not enabled")
                .alloc_buffer(size),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => unreachable!(),
            Backend::Cuda => unreachable!("CUDA pool not implemented"),
        }
    }

    fn free_in_pool(&mut self, backend: Backend, id: usize) {
        match backend {
            Backend::CPU => self.cpu.free_buffer(id),
            #[cfg(target_os = "macos")]
            Backend::Metal => {
                if let Some(m) = &mut self.metal {
                    m.free_buffer(id);
                }
            }
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => {}
            Backend::Cuda => {}
        }
    }

    /// Per-layer KV persistent regions (K and V), created on first use on the
    /// layer's assigned backend.
    fn ensure_kv(&mut self, layer: usize, backend: Backend, size: usize) -> [BufRef; 2] {
        if let Some(&pair) = self.kv.get(&layer) {
            return pair;
        }
        let k = self.alloc_persistent(&format!("kv.{layer}.k"), backend, size);
        let v = self.alloc_persistent(&format!("kv.{layer}.v"), backend, size);
        self.kv.insert(layer, [k, v]);
        [k, v]
    }

    /// Allocate a persistent (never-freed) region on a backend.
    pub fn alloc_persistent(&mut self, name: &str, backend: Backend, size: usize) -> BufRef {
        let id = self.alloc_in_pool(backend, size);
        self.persistent.push(PersistentBuf {
            name: name.to_string(),
            backend,
            id,
        });
        BufRef { backend, id }
    }

    /// Buffer handle for a node (None if the node is dead / not allocated).
    pub fn node_buffer(&self, id: NodeId) -> Option<BufRef> {
        self.node_to_buf.get(&id).copied()
    }

    /// Mark buffers whose liveness ended before exec index `i` as reusable.
    fn sweep(&mut self, i: usize) {
        let expired: Vec<(Backend, usize)> = self
            .buf_alive
            .iter()
            .filter(|(_, &al)| al < i)
            .map(|(&key, _)| key)
            .collect();
        for key in expired {
            self.buf_alive.remove(&key);
            self.free_in_pool(key.0, key.1);
        }
    }

    /// Fill an input node's buffer from host data (routes to the node's pool).
    /// (Test / debug helper — the generation loop fills I32 inputs.)
    #[allow(dead_code)]
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
        match br.backend {
            Backend::CPU => self.cpu.write_host(br.id, data),
            #[cfg(target_os = "macos")]
            Backend::Metal => self
                .metal
                .as_mut()
                .expect("Metal pool not enabled")
                .write_host(br.id, data),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => Err("Metal unavailable".into()),
            Backend::Cuda => Err("CUDA unavailable".into()),
        }
    }

    /// Host view of a CPU node's buffer (Metal nodes: use copy_to_cpu).
    /// (Test helper.)
    #[allow(dead_code)]
    pub fn get_buffer(&self, _graph: &ComputeGraph, id: NodeId) -> Option<&[f32]> {
        let br = self.node_buffer(id)?;
        match br.backend {
            Backend::CPU => self.cpu.read_host(br.id),
            _ => None,
        }
    }

    /// Host copy of any node's buffer (cross-backend reads).
    pub fn copy_to_cpu(&mut self, id: NodeId) -> Option<Vec<f32>> {
        let br = self.node_buffer(id)?;
        match br.backend {
            Backend::CPU => self.cpu.read_host(br.id).map(|s| s.to_vec()),
            #[cfg(target_os = "macos")]
            Backend::Metal => self.metal.as_mut().and_then(|m| m.read_host(br.id)).map(|s| s.to_vec()),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => None,
            Backend::Cuda => None,
        }
    }

    /// Host view of a persistent region by name (CPU pool).
    /// (Test helper.)
    #[allow(dead_code)]
    pub fn get_persistent(&self, name: &str) -> Option<&[f32]> {
        self.persistent
            .iter()
            .find(|p| p.name == name && p.backend == Backend::CPU)
            .and_then(|p| self.cpu.read_host(p.id))
    }

    /// Number of distinct buffers currently allocated (for tests).
    #[allow(dead_code)]
    pub fn n_cpu_buffers(&self) -> usize {
        self.cpu.pool_len()
    }

    /// Number of distinct buffers actually mapped to nodes (for tests).
    #[allow(dead_code)]
    pub fn n_mapped_buffers(&self) -> usize {
        let mut s: std::collections::BTreeSet<(Backend, usize)> = std::collections::BTreeSet::new();
        for br in self.node_to_buf.values() {
            s.insert((br.backend, br.id));
        }
        s.len()
    }

    /// Flush a backend's pending async work (split boundary / end).
    pub fn sync_backend(&mut self, backend: Backend) {
        match backend {
            Backend::CPU => {}
            #[cfg(target_os = "macos")]
            Backend::Metal => {
                if let Some(m) = &mut self.metal {
                    m.synchronize();
                }
            }
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => {}
            Backend::Cuda => {}
        }
    }

    /// Cross-backend copy of a node's buffer into `dst_backend`'s pool:
    /// host round trip through read_host/write_host (shared-memory GPU
    /// buffers make this a plain memcpy both ways).
    pub fn copy_across(
        &mut self,
        node_id: NodeId,
        dst_backend: Backend,
    ) -> Result<(), String> {
        let br = self
            .node_buffer(node_id)
            .ok_or_else(|| format!("node {node_id} has no buffer"))?;
        if br.backend == dst_backend {
            return Ok(());
        }
        let data = self
            .copy_to_cpu(node_id)
            .ok_or_else(|| format!("node {node_id} host read failed"))?;
        let new_id = self.alloc_in_pool(dst_backend, data.len());
        // write into the destination backend pool
        match dst_backend {
            Backend::CPU => self.cpu.write_host(new_id, &data)?,
            #[cfg(target_os = "macos")]
            Backend::Metal => self.metal.as_mut().unwrap().write_host(new_id, &data)?,
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => return Err("Metal unavailable".into()),
            Backend::Cuda => return Err("CUDA unavailable".into()),
        }
        // remap the node to the destination buffer
        self.node_to_buf.insert(node_id, BufRef { backend: dst_backend, id: new_id });
        // release the old buffer
        self.buf_alive.remove(&(br.backend, br.id));
        self.free_in_pool(br.backend, br.id);
        Ok(())
    }
}

impl KvProvider for GraphAllocator {
    fn kv_pair(&self, layer: usize) -> Option<(usize, usize)> {
        self.kv.get(&layer).map(|pair| (pair[0].id, pair[1].id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;

    fn chain(n_ops: usize) -> ComputeGraph {
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
    fn kv_regions_two_per_layer() {
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
        // store and load share the K region; V is a sibling
        assert_eq!(alloc.node_buffer(3), alloc.node_buffer(4));
        let pair = alloc.kv_pair(0).unwrap();
        assert_eq!(alloc.node_buffer(3).unwrap().id, pair.0);
        assert_ne!(pair.0, pair.1);
        assert_eq!(alloc.persistent.len(), 2);
        assert_eq!(alloc.persistent[0].name, "kv.0.k");
        assert_eq!(alloc.persistent[1].name, "kv.0.v");
        // mapped buffers: positions/k/v (3 liveness) + K region (shared) = 4
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
