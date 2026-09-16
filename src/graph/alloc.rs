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
    #[cfg(feature = "cuda")]
    cuda: Option<super::cuda_backend::CudaBackend>,
    node_to_buf: HashMap<NodeId, BufRef>,
    /// Cross-backend copies for the CURRENT graph (split-boundary staging):
    /// `(node, destination backend)` → buffer on that backend. NOT part of the
    /// node's canonical assignment — node_to_buf must stay re-executable (a
    /// remap would break the next execute of a reused graph, whose producing
    /// split would find its buffer on another backend). The same staging buffer
    /// is rewritten on every execute (no per-step allocation). Keying on the
    /// destination as well is what lets one node feed two foreign backends.
    cross: HashMap<(NodeId, Backend), BufRef>,
    /// (backend, pool id) → last exec index it stays alive until
    buf_alive: HashMap<(Backend, usize), usize>,
    /// Per-layer KV **cell store**: the persistent regions plus per-cell
    /// sequence ownership (Phase C / C1). The allocator only allocates the
    /// arenas; the store owns their bookkeeping and the `position -> cell`
    /// resolution.
    kv: super::kvcache::KvCache,
    /// All persistent regions (never freed).
    pub persistent: Vec<PersistentBuf>,
}

impl Default for GraphAllocator {
    fn default() -> Self {
        Self {
            cpu: CpuBackend::new(),
            #[cfg(target_os = "macos")]
            metal: None,
            #[cfg(feature = "cuda")]
            cuda: None,
            node_to_buf: HashMap::new(),
            cross: HashMap::new(),
            buf_alive: HashMap::new(),
            kv: super::kvcache::KvCache::new(),
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

    /// Enable the CUDA backend (device presence + `MINFER_DISABLE_CUDA` are
    /// checked by `CudaBackend::new` via the CudaState singleton).
    /// (Model-side wiring lands in Phase 7c; tests use it meanwhile.)
    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    pub fn enable_cuda(&mut self) -> bool {
        if self.cuda.is_none() {
            self.cuda = super::cuda_backend::CudaBackend::new();
        }
        self.cuda.is_some()
    }

    /// Test hook: ensure the CUDA backend exists and force its CUDA Graph
    /// capture/replay off (the direct-launch reference in A/B tests).
    #[cfg(all(feature = "cuda", test))]
    pub fn disable_graphs_for_test(&mut self) {
        if !self.enable_cuda() {
            panic!("disable_graphs_for_test: no CUDA device");
        }
        if let Some(c) = self.cuda.as_mut() {
            c.set_graphs_enabled_for_test(false);
        }
    }

    /// Mutable CUDA backend (None when unavailable / feature off).
    #[cfg(feature = "cuda")]
    pub fn cuda_mut(&mut self) -> Option<&mut super::cuda_backend::CudaBackend> {
        self.cuda.as_mut()
    }

    /// Immutable CUDA backend. (Phase 7c: used by the model-side gate.)
    #[allow(dead_code)]
    #[cfg(feature = "cuda")]
    pub fn cuda(&self) -> Option<&super::cuda_backend::CudaBackend> {
        self.cuda.as_ref()
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
        #[cfg(feature = "cuda")]
        if let Some(c) = &self.cuda {
            if c.supports_op(op, dtype) {
                return Some(Backend::Cuda);
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
        // cross-backend staging buffers belong to the previous graph (their
        // sizes follow that graph's shapes) — free and re-materialize on the
        // first execute of the new graph
        let prev_cross: Vec<((NodeId, Backend), BufRef)> = self.cross.drain().collect();
        for (_, cb) in prev_cross {
            self.free_in_pool(cb.backend, cb.id);
        }

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
                    let pair =
                        self.ensure_kv(layer, backend, node.n_elements(), node.out_shape[1])?;
                    // the node's buffer = the K region
                    self.node_to_buf.insert(id, pair[0]);
                }
                Op::FusedQKV { layer } => {
                    // fused decode QKV: also needs the layer's persistent KV
                    // regions (the kernel stores K/V), but its output is a
                    // normal concat buffer (q|k|v), not the K region.
                    let (kv_elems, n_ctx) = match &node.meta {
                        NodeMeta::FusedQkv(m) => (m.kv_elems, m.kv_elems / m.nkt.max(1)),
                        _ => (node.n_elements(), node.out_shape[1]),
                    };
                    self.ensure_kv(layer, backend, kv_elems, n_ctx)?;
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size);
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef { backend, id: pid });
                    }
                }
                Op::FusedQkvNorm { layer } => {
                    // fused decode QKV with per-head Q/K RMSNorm (Qwen3): same
                    // layout as FusedQKV — persistent KV regions + a normal
                    // concat (q|k|v) output buffer for the attention q input.
                    let (kv_elems, n_ctx) = match &node.meta {
                        NodeMeta::FusedQkvNorm(m) => (m.kv_elems, m.kv_elems / m.nkt.max(1)),
                        _ => (node.n_elements(), node.out_shape[1]),
                    };
                    self.ensure_kv(layer, backend, kv_elems, n_ctx)?;
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size);
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef { backend, id: pid });
                    }
                }
                Op::Silu | Op::RoPE { .. } | Op::QkvBiasRopeStore { .. } => {
                    // D3-8: the mixed-quant QKV epilogue also needs the layer's
                    // persistent KV regions (it stores k/v like FusedQKV).
                    if let Op::QkvBiasRopeStore { layer } = &node.op {
                        let (kv_elems, n_ctx) = match &node.meta {
                            NodeMeta::QkvBiasRopeStore(m) => {
                                (m.kv_elems, m.kv_elems / m.nkt.max(1))
                            }
                            _ => (node.n_elements(), node.out_shape[1]),
                        };
                        self.ensure_kv(*layer, backend, kv_elems, n_ctx)?;
                    }
                    // In-place elementwise transforms: alias the input buffer
                    // (llama.cpp executes rope/silu in place). Same-backend
                    // aliasing avoids a host-side copy between a pending GPU
                    // producer and this kernel — it reads/writes the buffer the
                    // producer wrote, in kernel order. Cross-backend inputs get
                    // a fresh buffer: the producer completed before the split
                    // boundary, so the backend's host copy is safe there.
                    if last_use[id] > i {
                        let in_ref =
                            self.node_to_buf.get(&node.src[0]).copied().ok_or_else(|| {
                                format!("in-place op src buffer missing (node {id})")
                            })?;
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
            #[cfg(feature = "cuda")]
            Backend::Cuda => self
                .cuda
                .as_mut()
                .expect("CUDA pool not enabled")
                .alloc_buffer(size),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => unreachable!("CUDA pool not implemented"),
        }
    }

    /// Fresh (never recycled) buffer on a backend's pool — split-boundary
    /// staging only. See Backend::alloc_fresh.
    fn alloc_fresh_in(&mut self, backend: Backend, size: usize) -> usize {
        match backend {
            Backend::CPU => self.cpu.alloc_fresh(size),
            #[cfg(target_os = "macos")]
            Backend::Metal => self
                .metal
                .as_mut()
                .expect("Metal pool not enabled")
                .alloc_fresh(size),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => unreachable!(),
            #[cfg(feature = "cuda")]
            Backend::Cuda => self
                .cuda
                .as_mut()
                .expect("CUDA pool not enabled")
                .alloc_fresh(size),
            #[cfg(not(feature = "cuda"))]
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
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                if let Some(c) = &mut self.cuda {
                    c.free_buffer(id);
                }
            }
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => {}
        }
    }

    /// Per-layer KV persistent regions (K and V), created on first use on the
    /// layer's assigned backend.
    ///
    /// Returns `Err` when the layer already owns a region whose element count
    /// or backend differs from the request. The regions outlive graph rebuilds
    /// (they ARE the KV cache), so a session that changes `n_ctx` — or moves a
    /// layer to another backend — must fail loudly here rather than silently
    /// reuse an under-sized or wrong-backend region and corrupt memory
    /// (`docs/ARCHITECTURE-ROADMAP.md` §2.4).
    fn ensure_kv(
        &mut self,
        layer: usize,
        backend: Backend,
        elems: usize,
        n_ctx: usize,
    ) -> Result<[BufRef; 2], String> {
        if let Some(region) = self.kv.get(layer) {
            if region.elems != elems {
                return Err(format!(
                    "KV region for layer {layer} was allocated with {} elements but {elems} are \
                     requested (n_ctx changed on a live GraphCache; the regions are persistent)",
                    region.elems
                ));
            }
            if region.k.backend != backend {
                return Err(format!(
                    "KV region for layer {layer} lives on {:?} but this graph assigns it to {:?}",
                    region.k.backend, backend
                ));
            }
            return Ok([region.k, region.v]);
        }
        let k = self.alloc_persistent(&format!("kv.{layer}.k"), backend, elems);
        let v = self.alloc_persistent(&format!("kv.{layer}.v"), backend, elems);
        self.kv.insert(layer, k, v, elems, n_ctx);
        Ok([k, v])
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

    /// True while the KV mapping is the identity, i.e. while a backend may keep
    /// indexing the arenas with the raw `positions` input (Phase C / C1). The
    /// scheduler refuses to execute once this is false, because no backend
    /// consumes the resolved cell array yet.
    pub fn kv_is_identity(&self) -> bool {
        self.kv.is_identity()
    }

    /// Host-side `position -> cell` resolution for a layer (Phase C / C1).
    /// Today the identity; C2 is the only thing that changes its behaviour.
    #[allow(dead_code)] // the resolver's C2 consumers are the backends
    pub fn kv_cells_for(&self, layer: usize, positions: &[usize]) -> Result<Vec<u32>, String> {
        self.kv.cells_for(layer, positions)
    }

    /// Drop the identity fast path (Phase C / C2). After this the scheduler
    /// refuses to execute until the backends consume the resolved cell array,
    /// so a half-ported C2 fails loudly instead of writing the wrong row.
    #[allow(dead_code)] // C2 calls it
    pub fn kv_clear_identity(&mut self) {
        self.kv.clear_identity();
    }

    /// Context removal (Phase C / C2): remove rows `[start, start + len)` from
    /// every layer's KV arena and re-rope the rows after them by `-len`, so a
    /// token that was at position `p >= start + len` becomes addressable at
    /// `p - len`. Rows `[0, start)` keep both their cells and their positions —
    /// that is what lets a sliding window keep a prefix (a system prompt) while
    /// dropping a middle range, which is the conversation's overflow case
    /// (`docs/ARCHITECTURE-EXECUTION-PLAN.md` C2). [`Self::kv_shift`] is the
    /// `start == 0` special case.
    ///
    /// Host-side by design. A *physical* removal leaves `cell == pos` for every
    /// surviving row, so the mapping stays the identity and no backend needs a
    /// new kernel — the only backend involvement is the existing
    /// `copy_kv_to_cpu` / `write_host` pair. That is why this is the one Phase C
    /// operation that can land without touching CUDA or Metal. Cost is O(n_ctx)
    /// per layer, once per overflow, against a re-prefill of the whole window.
    ///
    /// Two things C2 does **not** make exact, both recorded in the plan:
    /// the re-rope is not bitwise with a fresh prefill (two composed rotations
    /// vs one), and the surviving rows keep the values they were computed with —
    /// so rows that attended to the removed ones are a documented approximation
    /// of a fresh prefill, not an equivalent of it.
    pub fn kv_rm(
        &mut self,
        start: usize,
        len: usize,
        rope: &super::kvcache::KvRope,
    ) -> Result<usize, String> {
        let layers: Vec<usize> = self.kv.iter().map(|(l, _)| l).collect();
        if layers.is_empty() {
            return Err("kv_rm: no KV arena allocated".into());
        }
        // Validate every layer first: a rejected removal must not leave the
        // arenas half-shifted.
        for &layer in &layers {
            let l = self
                .kv
                .get(layer)
                .ok_or_else(|| format!("kv_rm: no arena for layer {layer}"))?;
            if start + len > l.n_used {
                return Err(format!(
                    "kv_rm: cannot remove {len} rows at {start} of {} written rows (layer {layer})",
                    l.n_used
                ));
            }
        }
        let mut new_used = 0usize;
        for layer in layers {
            let (n_used, row, kref, vref) = {
                let l = self
                    .kv
                    .get(layer)
                    .ok_or_else(|| format!("kv_rm: no arena for layer {layer}"))?;
                (l.n_used, l.elems / l.n_ctx.max(1), l.k, l.v)
            };
            let (mut k, mut v) = self
                .copy_kv_to_cpu(layer)
                .ok_or_else(|| format!("kv_rm: layer {layer} host read failed"))?;
            new_used = n_used - len;
            // Slide the survivors down, re-rope their K (rows [0, start) do not
            // move and are not touched), and clear the tail: the arena must
            // never hold rows that no position can address.
            k.copy_within((start + len) * row..n_used * row, start * row);
            v.copy_within((start + len) * row..n_used * row, start * row);
            super::kvcache::rope_shift_kv(
                &mut k[start * row..],
                new_used - start,
                len as isize,
                rope,
            );
            for x in &mut k[new_used * row..] {
                *x = 0.0;
            }
            for x in &mut v[new_used * row..] {
                *x = 0.0;
            }
            self.write_pool(kref.backend, kref.id, &k)?;
            self.write_pool(vref.backend, vref.id, &v)?;
        }
        let n = self.kv.after_rm(start, len)?;
        debug_assert_eq!(n, new_used);
        Ok(n)
    }

    /// Sliding-window special case of [`Self::kv_rm`]: drop the oldest `drop`
    /// rows of every layer's KV arena and re-rope the survivors by `-drop`, so
    /// the same tokens become addressable at `pos - drop`. Returns the new
    /// written-row count.
    pub fn kv_shift(
        &mut self,
        drop: usize,
        rope: &super::kvcache::KvRope,
    ) -> Result<usize, String> {
        self.kv_rm(0, drop, rope)
    }

    /// Written rows in a layer's arena (`n_used`), or `None` before allocation.
    pub fn kv_n_used(&self, layer: usize) -> Option<usize> {
        self.kv.get(layer).map(|l| l.n_used)
    }

    /// Record that the arena now holds rows `0..n_used` (Phase C / C2). The
    /// model calls this after a forward with `max(positions) + 1`, which is the
    /// only place that knows how far the KV store wrote.
    pub fn kv_note_used(&mut self, n_used: usize) {
        self.kv.own_prefix(super::kvcache::SEQ_MAIN, n_used);
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
        // Positions are the one I32 input that indexes a persistent region: the
        // KV store writes row `positions[t]` of an `[n_kv_embd, n_ctx]` region,
        // so a position >= n_ctx is an out-of-bounds write. The CPU backend
        // re-checks this per store node; the GPU backends write unconditionally.
        // Validating here — the single point where positions become graph data —
        // covers every backend without a per-backend guard, Metal included, and
        // costs one O(nt) scan over data that is already on the host.
        self.check_positions_bound(graph, name, data)?;
        let bits: Vec<f32> = data.iter().map(|&v| f32::from_bits(v)).collect();
        self.fill_input_impl(graph, name, &bits)
    }

    /// Reject values in an I32 input that would index past the KV region (see
    /// `fill_input_i32`). Only inputs actually consumed by a KV-writing or
    /// attention node are bounded — `token_ids` is I32 too, and vocabularies are
    /// routinely larger than `n_ctx`.
    fn check_positions_bound(
        &self,
        graph: &ComputeGraph,
        name: &str,
        data: &[u32],
    ) -> Result<(), String> {
        let Some(id) = graph
            .inputs
            .iter()
            .copied()
            .find(|&i| graph.node(i).name == name)
        else {
            return Ok(()); // unknown name: fill_input_impl reports it
        };
        let indexes_kv = graph.nodes.iter().any(|n| {
            n.src.contains(&id)
                && matches!(
                    n.op,
                    Op::KvcacheStore { .. }
                        | Op::FusedQKV { .. }
                        | Op::FusedQkvNorm { .. }
                        | Op::QkvBiasRopeStore { .. }
                        | Op::Attn { .. }
                )
        });
        if !indexes_kv {
            return Ok(());
        }
        // Resolve through the cell store rather than re-deriving the bound from
        // a node's shape: the store owns `n_ctx` and the position→cell mapping,
        // so when C2 makes that mapping non-identity this check keeps meaning
        // the same thing (it becomes an ownership check) without being touched.
        let mut layers: Vec<usize> = graph
            .nodes
            .iter()
            .filter_map(|n| match &n.op {
                Op::KvcacheStore { layer }
                | Op::KvcacheLoad { layer }
                | Op::FusedQKV { layer }
                | Op::FusedQkvNorm { layer }
                | Op::QkvBiasRopeStore { layer } => Some(*layer),
                Op::Attn { .. } => match &n.meta {
                    NodeMeta::Attn(m) => Some(m.layer),
                    _ => None,
                },
                _ => None,
            })
            .collect();
        if layers.is_empty() {
            return Ok(());
        }
        layers.sort_unstable();
        layers.dedup();
        let positions: Vec<usize> = data.iter().map(|&p| p as usize).collect();
        for layer in layers {
            if self.kv.contains(layer) {
                self.kv.cells_for(layer, &positions).map_err(|e| {
                    format!("input '{name}': {e} (`docs/ARCHITECTURE-ROADMAP.md` §2.4)")
                })?;
            }
        }
        Ok(())
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
            #[cfg(feature = "cuda")]
            Backend::Cuda => self
                .cuda
                .as_mut()
                .expect("CUDA pool not enabled")
                .write_host(br.id, data),
            #[cfg(not(feature = "cuda"))]
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
            Backend::Metal => self
                .metal
                .as_mut()
                .and_then(|m| m.read_host(br.id))
                .map(|s| s.to_vec()),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => None,
            #[cfg(feature = "cuda")]
            Backend::Cuda => self.cuda.as_ref().and_then(|c| c.copy_to_host(br.id)),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => None,
        }
    }

    /// Host copy of a layer's persistent KV regions (K, V) by pool buffer
    /// id — doc 95 identity debugging (the graph holds one KvcacheLoad node
    /// per layer, so the V half is reachable only through `kv_pair`).
    pub fn copy_kv_to_cpu(&mut self, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        let rd = |s: &mut Self, br: BufRef| -> Option<Vec<f32>> {
            match br.backend {
                Backend::CPU => s.cpu.read_host(br.id).map(|x| x.to_vec()),
                #[cfg(feature = "cuda")]
                Backend::Cuda => s.cuda.as_ref().and_then(|c| c.copy_to_host(br.id)),
                #[allow(unreachable_patterns)]
                _ => None,
            }
        };
        let l = self.kv.get(layer)?;
        let pair = [l.k, l.v];
        Some((rd(self, pair[0])?, rd(self, pair[1])?))
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
            #[cfg(feature = "cuda")]
            Backend::Cuda => {
                if let Some(c) = &mut self.cuda {
                    c.synchronize();
                }
            }
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => {}
        }
    }

    /// Cross-backend copy of a node's buffer into `dst_backend`'s pool:
    /// host round trip through read_host/write_host (shared-memory GPU
    /// buffers make this a plain memcpy both ways).
    ///
    /// The node's canonical buffer (node_to_buf) is left untouched — the copy
    /// lands in the `cross` staging map, which the scheduler consults when a
    /// CONSUMER on `dst_backend` resolves its inputs. Consumers on the node's
    /// own backend keep reading the original buffer. This keeps a reused graph
    /// re-executable: the producing split always finds its buffer where the
    /// allocator put it, and the staging buffer (allocated once per graph
    /// rebuild) is simply rewritten on each execute.
    pub fn copy_across(&mut self, node_id: NodeId, dst_backend: Backend) -> Result<(), String> {
        let br = self
            .node_buffer(node_id)
            .ok_or_else(|| format!("node {node_id} has no buffer"))?;
        if br.backend == dst_backend {
            return Ok(());
        }
        let data = self
            .copy_to_cpu(node_id)
            .ok_or_else(|| format!("node {node_id} host read failed"))?;
        // One staging buffer per (node, dst backend) per graph, reused on every
        // execute. Keyed by the destination too, because a node consumed by two
        // different foreign backends needs two buffers — the old single-entry
        // map forced the consumer-side filter in the scheduler, which `None`
        // here now expresses structurally.
        let dst_id = match self.cross.get(&(node_id, dst_backend)) {
            Some(&cb) => cb.id,
            None => {
                let id = self.alloc_fresh_in(dst_backend, data.len());
                self.cross.insert(
                    (node_id, dst_backend),
                    BufRef {
                        backend: dst_backend,
                        id,
                    },
                );
                id
            }
        };
        self.write_pool(dst_backend, dst_id, &data)
    }

    /// Write host data into a pool buffer of `backend` (shared by the staging
    /// paths of `copy_across`).
    fn write_pool(&mut self, backend: Backend, id: usize, data: &[f32]) -> Result<(), String> {
        match backend {
            Backend::CPU => self.cpu.write_host(id, data),
            #[cfg(target_os = "macos")]
            Backend::Metal => self.metal.as_mut().unwrap().write_host(id, data),
            #[cfg(not(target_os = "macos"))]
            Backend::Metal => Err("Metal unavailable".into()),
            #[cfg(feature = "cuda")]
            Backend::Cuda => self
                .cuda
                .as_mut()
                .expect("CUDA pool not enabled")
                .write_host(id, data),
            #[cfg(not(feature = "cuda"))]
            Backend::Cuda => Err("CUDA unavailable".into()),
        }
    }

    /// The staging buffer a consumer on `backend` must read for `node_id`, if a
    /// split boundary copied it for the current graph. Consumers on the node's
    /// own backend read the canonical buffer instead.
    pub fn cross_buffer(&self, node_id: NodeId, backend: Backend) -> Option<BufRef> {
        self.cross.get(&(node_id, backend)).copied()
    }

    /// Test hook: stage `node_id`'s output on `backend` as if a split boundary
    /// had copied it. The real path needs a second usable backend, which a
    /// CPU-only build does not have.
    #[cfg(test)]
    pub fn stage_cross_for_test(&mut self, node_id: NodeId, backend: Backend, id: usize) {
        self.cross
            .insert((node_id, backend), BufRef { backend, id });
    }
}

impl KvProvider for GraphAllocator {
    fn kv_pair(&self, layer: usize) -> Option<(usize, usize)> {
        self.kv.get(layer).map(|r| (r.k.id, r.v.id))
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
        assert!(
            alloc.n_mapped_buffers() > n_chain6,
            "parallel chains should not share"
        );
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

    /// The KV regions are persistent across rebuilds (they ARE the cache), so a
    /// graph that asks for a different `n_ctx` on the same allocator must be a
    /// loud error, not a silent reuse of the older, smaller region.
    #[test]
    fn kv_region_size_change_is_a_loud_error() {
        fn kv_graph(n_ctx: usize) -> ComputeGraph {
            let mut b = GraphBuilder::new();
            let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
            let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
            let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
            let _store = b.kvcache_store(0, k, v, pos, n_ctx);
            let load = b.kvcache_load(0, 16, n_ctx, 2);
            b.output(load);
            b.build()
        }

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&kv_graph(1024)).unwrap();
        // Unchanged shape: reuse is fine (this is the decode-reuse path).
        alloc.alloc_graph(&kv_graph(1024)).unwrap();
        // Changed n_ctx on a live cache: must fail loudly.
        let err = alloc.alloc_graph(&kv_graph(2048)).unwrap_err();
        assert!(err.contains("KV region for layer 0"), "got: {err}");
    }

    /// Positions index the persistent KV region, so a value past `n_ctx` is an
    /// out-of-bounds write on every backend that does not re-check it (the GPU
    /// ones). The guard lives in the allocator so all three backends share it.
    #[test]
    fn position_beyond_n_ctx_is_rejected() {
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

        // The last legal row is n_ctx - 1.
        alloc.fill_input_i32(&g, "positions", &[1023]).unwrap();
        let err = alloc.fill_input_i32(&g, "positions", &[1024]).unwrap_err();
        assert!(err.contains(">= n_ctx 1024"), "got: {err}");
    }

    /// The same guard must NOT bound `token_ids`: a vocabulary is routinely
    /// larger than `n_ctx`.
    #[test]
    fn token_ids_are_not_bounded_by_n_ctx() {
        let mut b = GraphBuilder::new();
        let ids = b.input("token_ids", [1, 1, 1, 1], crate::graph::DType::I32);
        let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
        let emb = b.get_rows(k, ids, [16, 1, 1, 1]);
        let _store = b.kvcache_store(0, emb, v, pos, 8);
        let load = b.kvcache_load(0, 16, 8, 2);
        b.output(load);
        let g = b.build();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();

        alloc
            .fill_input_i32(&g, "token_ids", &[50_000])
            .expect("token ids are not positions");
        let err = alloc.fill_input_i32(&g, "positions", &[8]).unwrap_err();
        assert!(err.contains(">= n_ctx 8"), "got: {err}");
    }

    /// A staging buffer is keyed by (node, destination backend): one node
    /// feeding two foreign backends gets one buffer each, and a consumer is
    /// never offered the other backend's copy. The old single-entry map forced
    /// the scheduler to filter by backend on every read (and could not serve
    /// two foreign consumers at all).
    #[test]
    fn staging_is_keyed_by_destination_backend() {
        let mut alloc = GraphAllocator::new();
        alloc.stage_cross_for_test(7, Backend::CPU, 3);
        assert_eq!(alloc.cross_buffer(7, Backend::CPU).map(|b| b.id), Some(3));
        assert!(
            alloc.cross_buffer(7, Backend::Cuda).is_none(),
            "a CPU staging buffer must not be offered to a CUDA consumer"
        );
        alloc.stage_cross_for_test(7, Backend::Cuda, 4);
        assert_eq!(alloc.cross_buffer(7, Backend::Cuda).map(|b| b.id), Some(4));
        assert_eq!(
            alloc.cross_buffer(7, Backend::CPU).map(|b| b.id),
            Some(3),
            "staging for a second backend must not clobber the first"
        );
    }

    #[test]
    fn cycle_graph_allocation_fails() {
        let mut g = ComputeGraph::default();
        g.nodes.push(super::super::CNode {
            id: 0,
            name: "a".into(),
            op: Op::Add,
            src: vec![1],
            out_shape: [1, 1, 1, 1],
            out_dtype: super::super::DType::F32,
            backend: None,
            meta: super::super::ops::NodeMeta::None,
        });
        g.nodes.push(super::super::CNode {
            id: 1,
            name: "b".into(),
            op: Op::Add,
            src: vec![0],
            out_shape: [1, 1, 1, 1],
            out_dtype: super::super::DType::F32,
            backend: None,
            meta: super::super::ops::NodeMeta::None,
        });
        let mut alloc = GraphAllocator::new();
        assert!(alloc.alloc_graph(&g).is_err());
    }
}
