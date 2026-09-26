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

use std::collections::{HashMap, HashSet};
use std::path::Path;

use super::allocplan;
use super::backend::Backend as BackendTrait;
use super::backend::KvProvider;
use super::copystats::{self, CrossCopyStats};
use super::cpu_backend::CpuBackend;
use super::kvformat::KvFormat;
use super::kvsession::{
    KvSessionExpect, KvSessionHeader, KvSessionReader, KvSessionReport, KvSessionWriter, VERSION,
};
use super::ops::{NodeMeta, Op};
use super::{Backend, BufRef, ComputeGraph, NodeId, PersistentBuf};

/// E4's accounting for one backend: what the pool holds, what is live right now, the
/// peak live set, the registered weights, and the budget the allocator checks against.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryReport {
    pub weights_bytes: usize,
    pub pool_bytes: usize,
    pub live_bytes: usize,
    pub peak_live_bytes: usize,
    pub budget: Option<usize>,
    /// E4 S3: reserved class-sized buffers that are idle right now — the reservation table's
    /// depth. A rebuild that re-maps leaves this where it was; a build that reserves more
    /// raises it.
    pub idle_slots: usize,
    /// E4 S3: how many (backend, size class) reservations the table holds.
    pub reserved_classes: usize,
}

impl MemoryReport {
    /// Whether `budget` is a real bound the gate compares against, as opposed to
    /// "unbounded" — `None` (CPU/Metal), or the `usize::MAX` an *unaccounted* device falls
    /// back to when its free-memory query failed. The metrics surface omits the
    /// budget/headroom families when this is false rather than publishing a number that
    /// was never measured (issue #122).
    pub fn budget_is_bounded(&self) -> bool {
        matches!(self.budget, Some(b) if b != usize::MAX)
    }

    /// Bytes the budget would refuse next: `budget - (weights + pool)`, or `None` when
    /// there is no bound.
    pub fn headroom_bytes(&self) -> Option<usize> {
        self.budget
            .filter(|b| *b != usize::MAX)
            .map(|b| b.saturating_sub(self.weights_bytes + self.pool_bytes))
    }
}

/// Print the reason for an unaccounted (fallback) budget **once per process**.
///
/// `memory_budget` runs per allocation, so a device whose memory query keeps failing
/// would otherwise print this on every node; once is loud enough to be found and quiet
/// enough not to bury the log. `OnceLock` rather than an atomic flag so the note itself
/// is what gets printed, whichever call site sees the failure first.
fn unaccounted_budget_note(note: &str) {
    static PRINTED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    if PRINTED.set(()).is_ok() {
        eprintln!("minfer: E4 memory accounting is unmeasured: {note}");
    }
}

/// The byte total behind a possibly-poisoned registry lock, recovering (and naming) the
/// poison instead of reporting zero.
///
/// A poisoned mutex means a previous holder panicked. The registries this is used on are
/// append-only, so the map behind the poison is still consistent — and the old
/// `.map(|w| …).unwrap_or(0)` behaviour was the fail-*open* twin of issue #122's failed
/// device query: it silently reported **0 registered weights**, so `weights + activations
/// > budget` under-charged the budget by every resident weight. Recovering the value and
/// saying so keeps the accounting honest.
pub fn weights_from_lock<T, F>(lock: std::sync::LockResult<T>, what: &str, sum: F) -> usize
where
    F: FnOnce(&T) -> usize,
{
    match lock {
        Ok(v) => sum(&v),
        Err(poisoned) => {
            eprintln!(
                "minfer: the {what} lock was poisoned by an earlier panic; recovering the \
                 entries it still holds rather than reporting 0 bytes"
            );
            sum(&poisoned.into_inner())
        }
    }
}

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
    /// E4 S3: keyed by the graph's **uid** first, because node ids restart per graph and the
    /// cache now holds several: a staging buffer belongs to one graph's node, at one size.
    /// The entries survive a re-map (`alloc_graph` no longer drops them) — re-creating them
    /// per switch would leak, since staging is allocated fresh, never from the slot table.
    cross: HashMap<(u64, NodeId, Backend), BufRef>,
    /// F5 ([#58]): the staging entries whose async copy has been enqueued but
    /// whose event has not been waited on yet — i.e. the boundary inputs that
    /// still **owe** their phase-B wait. `copy_across` inserts, `await_cross`
    /// removes; `cross_input` (the consumer's read) refuses a pending entry
    /// loudly, so a boundary that drops its wait is a named error instead of a
    /// read of in-flight device data.
    ///
    /// Cleared at every `alloc_graph` (a rebuild starts a fresh execution, and no
    /// copy can be in flight across it — the boundary either completed or failed).
    cross_pending: HashSet<(u64, NodeId, Backend)>,
    /// F5: what this allocator's split boundaries did — the measured half of "no
    /// host-side blocking copy on the hot path". See
    /// [`super::copystats::CrossCopyStats`].
    cross_stats: CrossCopyStats,
    /// (backend, pool id) → last exec index it stays alive until
    buf_alive: HashMap<(Backend, usize), usize>,
    /// E4: bytes each pooled buffer occupies (its size class), keyed by `(backend, id)`
    /// so a release can subtract exactly what the allocation added.
    buf_bytes: HashMap<(Backend, usize), usize>,
    /// E4 S3: the **reservation** half of reserve/assign — idle class-sized pool buffers,
    /// keyed by `(backend, class in elements)`. A buffer released by liveness goes here
    /// instead of back to the backend, so the next graph that needs that class re-maps onto
    /// it without touching the pool (`alloc_buffer`/`free_buffer` are not called at all in
    /// the steady state, which is what keeps CUDA's `pool_gen` — and therefore its captured
    /// graphs — stable across rebuilds).
    ///
    /// A `BTreeSet` so the assignment is **deterministic**: liveness releases buffers in
    /// `HashMap` order, so a LIFO/FIFO list would hand a rebuild different ids depending on
    /// the iteration order, and a graph switched back into the cache would move its nodes
    /// around. The smallest idle id is always taken first, which makes a given shape's
    /// assignment reproducible and pins a cached graph's mapping.
    slots: HashMap<(Backend, usize), std::collections::BTreeSet<usize>>,
    /// E4 S3: the class (in elements) of a **classed** allocation, so a release knows whether
    /// it returns to `slots` or to the backend's own free list (staging buffers are exact and
    /// stay with the backend).
    buf_class: HashMap<(Backend, usize), usize>,
    /// E4 accounting per backend: reserved (the pool's high-water mark), live, peak live.
    pool_bytes: HashMap<Backend, usize>,
    live_bytes: HashMap<Backend, usize>,
    peak_bytes: HashMap<Backend, usize>,
    /// E4: an explicit memory budget per backend; unset = the backend's own default.
    budget: HashMap<Backend, usize>,
    /// E5: the layer offload plan in force (None = the pre-E5 "the device takes whatever it
    /// can"). Read by `supports_for`, so it decides every node's backend.
    offload: Option<super::offload::OffloadPlan>,
    /// Per-layer KV **cell store**: the persistent regions plus per-cell
    /// sequence ownership (Phase C / C1). The allocator only allocates the
    /// arenas; the store owns their bookkeeping and the `position -> cell`
    /// resolution.
    kv: super::kvcache::KvCache,
    /// F4: the backends this run may use (`--backend` / `MINFER_BACKENDS`).
    ///
    /// Read by `supports_for`, so a fenced backend is never offered. It is the
    /// same fence the graph builders apply to `Device` (`registry::active_filter`,
    /// captured at construction), which is what keeps the two from disagreeing.
    filter: super::registry::BackendFilter,
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
            cross_pending: HashSet::new(),
            cross_stats: CrossCopyStats::default(),
            buf_alive: HashMap::new(),
            buf_bytes: HashMap::new(),
            slots: HashMap::new(),
            buf_class: HashMap::new(),
            pool_bytes: HashMap::new(),
            live_bytes: HashMap::new(),
            peak_bytes: HashMap::new(),
            budget: HashMap::new(),
            offload: None,
            kv: super::kvcache::KvCache::new(),
            filter: super::registry::active_filter().clone(),
            persistent: Vec::new(),
        }
    }
}

/// D1: extend a node's liveness past `to`, walking its `view_src` ancestors —
/// a view shares its parent's buffer, so the parent must not be recycled while
/// the view (or anything aliasing it) is still read.
///
/// Returns the nodes whose liveness actually grew (in walk order, the node
/// first), so the caller can keep the pool's own deadline in step: `buf_alive`
/// is set from `last_use` when a buffer is allocated, and a later extension that
/// does not touch it would let `sweep` hand the buffer to another node while the
/// alias still reads it (E4 S2: rounding makes such a hand-over likely, because
/// a class match no longer needs an exact size coincidence — the bug was latent
/// before it).
fn extend_through_views(
    graph: &ComputeGraph,
    last_use: &mut [usize],
    from: NodeId,
    to: usize,
) -> Vec<NodeId> {
    let mut grown = Vec::new();
    let mut cur = from;
    loop {
        if last_use[cur] >= to {
            break;
        }
        last_use[cur] = to;
        grown.push(cur);
        match graph.node(cur).view {
            Some(v) => cur = v.src,
            None => break,
        }
    }
    grown
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

    /// C4 per-engine (issue #99): tell this allocator's **CPU** kernels which KV
    /// format the engine they serve resolved. The model's graph stamps the same
    /// format into every KV node's `KvcacheMeta::row_elems`, so the store/attention
    /// dispatch and the region width cannot disagree within one engine.
    ///
    /// Honest scope: only the CPU half is per-engine. The CUDA and Metal device
    /// layers still hold a process-wide layout tag (`cuda::KV_LAYOUT`,
    /// `metal::kv_cache_is_f16`) that their kernels read, so a device run keeps the
    /// documented serial discipline; making the device layout per-graph is filed as
    /// its own follow-up.
    pub fn set_kv_format(&mut self, format: KvFormat) {
        self.cpu.set_kv_format(format);
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
    ///
    /// E5: `supports_for` with `layer = None` (no offload policy) — kept for callers that
    /// only ask "does any backend have this op", like the op matrix.
    pub fn supports(&self, op: &Op, dtype: crate::graph::DType) -> Option<Backend> {
        self.supports_for(op, dtype, None)
    }

    /// Which backend may take a node of `(op, dtype)` that belongs to `layer` — the
    /// assignment rule with the E5 offload policy and F4's backend fence applied.
    ///
    /// With a plan in force, a node whose block is **not** offloaded never reaches the
    /// device: the answer starts at the CPU, so a partial plan cannot silently run a
    /// non-offloaded block's op on a device that never registered its weights. A node
    /// outside any block (`layer = None`) follows the device only when every block is
    /// offloaded (`OffloadPlan::device_holds_unblocked`).
    ///
    /// F4: the device order is the registry's **priority** order, not a statement
    /// order in this function — Metal (300), CUDA (200), CPU (100) — and a backend
    /// the run fenced off (`--backend` / `MINFER_BACKENDS`) or whose pool is not
    /// enabled is skipped. The CPU is tried last and is always allowed: it is the
    /// universal fallback (`BackendFilter::from_names` re-admits it), so the order
    /// and the answer are identical to the pre-F4 chain.
    pub fn supports_for(
        &self,
        op: &Op,
        dtype: crate::graph::DType,
        layer: Option<usize>,
    ) -> Option<Backend> {
        let eligible = |b: &dyn BackendTrait| -> bool { super::backend_takes(b, op, dtype) };
        let device_ok = match (self.offload, layer) {
            (None, _) => true, // no policy: the pre-E5 all-or-nothing behaviour
            (Some(p), Some(l)) => p.on_device(l),
            (Some(p), None) => p.device_holds_unblocked(),
        };
        if device_ok {
            for &backend in super::registry::registry().by_priority() {
                if backend == Backend::CPU {
                    continue; // the fallback, tried below even when device_ok is false
                }
                if !self.filter.allows(backend) {
                    continue;
                }
                if let Some(pool) = self.pool(backend) {
                    if eligible(pool) {
                        return Some(backend);
                    }
                }
            }
        }
        if self.filter.allows(Backend::CPU) && eligible(&self.cpu) {
            return Some(Backend::CPU);
        }
        None
    }

    /// F4: the backends this run may offer (the `--backend` / `MINFER_BACKENDS`
    /// fence). Production installs it once at startup through
    /// `registry::install_filter`, before any allocator exists — so the field's
    /// default is already the run's fence. These two are the per-allocator test
    /// surface (the device-gated assignment gate needs to fence one allocator
    /// without touching the process).
    #[cfg(test)]
    pub fn set_backend_filter(&mut self, filter: super::registry::BackendFilter) {
        self.filter = filter;
    }

    /// F4: the filter in force on this allocator (test accessor).
    #[cfg(test)]
    pub fn backend_filter(&self) -> &super::registry::BackendFilter {
        &self.filter
    }

    /// F4: this backend's pool as the trait object every pool operation goes
    /// through, or `None` when the backend is not compiled in or its pool is not
    /// enabled on this allocator.
    ///
    /// This pair of helpers is what removed the twelve `match backend { … }`
    /// dispatch sites from this file: the registry entry says how to reach the
    /// pool, and the operation is a `Backend` trait call on it.
    pub fn pool(&self, backend: Backend) -> Option<&dyn BackendTrait> {
        (backend.entry()?.pool)(self)
    }

    /// The mutable form of [`Self::pool`].
    pub fn pool_mut(&mut self, backend: Backend) -> Option<&mut dyn BackendTrait> {
        (backend.entry()?.pool_mut)(self)
    }

    /// [`Self::pool_mut`] for a call site that cannot proceed without the pool:
    /// an internal invariant violation, so it panics naming the backend rather
    /// than silently skipping the operation.
    fn require_pool_mut(&mut self, backend: Backend) -> &mut dyn BackendTrait {
        let why = match backend.entry() {
            Some(_) => "pool not enabled",
            None => "not compiled into this build",
        };
        self.pool_mut(backend)
            .unwrap_or_else(|| panic!("{} backend {why}", backend.name()))
    }

    /// F4: the enabled backends in **identity** order — the vector the fusion
    /// pass probes, replacing the hand-built `[cpu, metal?, cuda?]` in three
    /// call sites. [`Self::fusion_backend_index`] is the matching node → index
    /// map, so the two cannot drift apart.
    pub fn fusion_backends(&self) -> Vec<&dyn BackendTrait> {
        super::registry::registry()
            .iter()
            .filter_map(|e| self.pool(e.handle))
            .collect()
    }

    /// F4: a node backend's index in [`Self::fusion_backends`].
    pub fn fusion_backend_index(&self, backend: Backend) -> Option<usize> {
        super::registry::registry()
            .iter()
            .filter(|e| self.pool(e.handle).is_some())
            .position(|e| e.handle == backend)
    }

    /// E5: put an offload plan in force. `None` restores the pre-E5 behaviour (no layer
    /// policy — the device takes whatever it can). The plan is *topology* (it decides every
    /// node's backend), so the caller also carries it in `CParams.gpu_layers` for the reuse
    /// check; this setter is what the assignment pass reads.
    pub fn set_offload_plan(&mut self, plan: Option<super::offload::OffloadPlan>) {
        self.offload = plan;
    }

    /// The offload plan in force (E5).
    pub fn offload_plan(&self) -> Option<super::offload::OffloadPlan> {
        self.offload
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
        // F5: a rebuild starts a fresh execution — no staging copy can be in
        // flight across it (the boundary either completed or failed loudly).
        self.cross_pending.clear();
        // Cross-backend staging buffers are keyed by `(graph uid, node, backend)` and
        // survive both a rebuild and a re-map (E4 S3): their size follows the shape of the
        // node in *that* graph, so an entry is only valid for its own graph, and re-creating
        // them per switch would leak (staging is `alloc_fresh`, which by design never recycles
        // from the free list). Entries of evicted graphs stay allocated — a handful of
        // boundary-sized buffers, bounded by the cache's graph count.

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

        // E4 S2: every input gets its buffer **before** the walk allocates anything.
        //
        // An input is host-filled before execution starts, so it must never take a buffer
        // that this build's `sweep` releases: the previous owner writes its output during
        // execution — *after* the fill — and would clobber the input before its consumer
        // reads it. Placing inputs first is enough, because at this moment the free list
        // holds only buffers released by the *previous* graph, whose writers have finished.
        // (Before size classes an input could only reuse an exactly-equal-size buffer, so
        // the window was narrow; a class match is common, which is how this surfaced.)
        for node in graph.nodes.iter().filter(|n| n.is_input()) {
            let id = node.id;
            let backend = node.backend.unwrap_or(Backend::CPU);
            let size = node.n_elements();
            let pid = self.alloc_in_pool(backend, size)?;
            self.buf_alive.insert((backend, pid), last_use[id]);
            self.node_to_buf.insert(id, BufRef::own(backend, pid, size));
        }

        for (i, &id) in order.iter().enumerate() {
            self.sweep(i);
            let node = graph.node(id);
            let backend = node.backend.unwrap_or(Backend::CPU);
            // inputs were placed above, before any sweep could recycle a buffer into them
            if node.is_input() {
                continue;
            }
            // D1: a view allocates nothing. It *is* its parent's buffer, so the
            // parent's liveness must cover it (and every consumer of it) — the
            // two refusals below are loud on purpose: a view this increment
            // cannot express must not silently become a copy.
            if let Some(v) = node.view {
                let parent = self.node_to_buf.get(&v.src).copied().ok_or_else(|| {
                    format!(
                        "view node {id} ('{}') has no buffer for its parent {}",
                        node.name, v.src
                    )
                })?;
                if parent.backend != backend {
                    return Err(format!(
                        "view node {id} ('{}') is assigned to {backend:?} but its parent {} lives on \
                         {:?}: a cross-backend view would need a staged copy, so build a materialized \
                         copy instead of a view",
                        node.name,
                        v.src,
                        parent.backend
                    ));
                }
                // D1 increment 2: the window may start at a non-zero offset and
                // be shorter than its parent, so the window must fit *inside* it
                // (`offset + len <= parent`), and the reference carries both so
                // every consumer slices exactly.
                let parent_elems = graph.node(v.src).n_elements();
                let window = node.n_elements();
                // D1: Metal's kernels have no element offset, so it can express
                // an exact view only (offset 0 and the parent's own length) —
                // `supports_op` already refuses a non-zero offset there, and this
                // backstops the partial window, which the op cannot see. Loud,
                // because the alternative is reading the wrong bytes.
                if parent.backend == Backend::METAL && (v.offset != 0 || window != parent_elems) {
                    return Err(format!(
                        "view node {id} ('{}') is a window of {} elements at offset {} of a \
                         {parent_elems}-element buffer on Metal, whose kernels take a buffer and a \
                         length with no offset; Metal supports exact views only (G5)",
                        node.name, window, v.offset
                    ));
                }
                if v.offset + window > parent_elems {
                    return Err(format!(
                        "view node {id} ('{}') spans [{}, {}) but its parent {} has {parent_elems} \
                         elements — the window does not fit",
                        node.name,
                        v.offset,
                        v.offset + window,
                        v.src
                    ));
                }
                // The parent must stay alive through the view's *readers*: the view is
                // just a window of the parent's buffer, so its bytes are read at the
                // consumer's step, not at the view's own. Walk from the **parent**: the
                // walk stops at the first node whose liveness already covers `to`, and
                // starting from the view (which by definition ends at `to`) made the
                // whole call a no-op — D1's parent extension never ran (D1/D3, found by
                // E4 S2, where class rounding turned the stale deadline into a real
                // hand-over).
                let lu = last_use[id];
                self.node_to_buf.insert(id, parent.window(v.offset, window));
                let grown = extend_through_views(graph, &mut last_use, v.src, lu);
                self.extend_buffer_alive(&grown, lu);
                continue;
            }
            match node.op {
                Op::KvcacheStore { layer } | Op::KvcacheLoad { layer } => {
                    // C4: the persistent region is sized by the meta's *cell width*
                    // (`row_elems` — packed under Q8_0) times the node's `n_ctx`; the
                    // node's own element count stays the logical `n_embd * n_ctx`.
                    let (n_embd, row_elems) = match &node.meta {
                        NodeMeta::Kvcache(m) => (m.n_embd, m.row_elems),
                        _ => (node.out_shape[0], node.out_shape[0]),
                    };
                    let pair =
                        self.ensure_kv(layer, backend, n_embd, row_elems, node.out_shape[1])?;
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
                    // The fusion is GPU-only, so its logical width is also its cell
                    // width. Under a packed format this layer's region would be sized
                    // differently from its store node's, and `ensure_kv` refuses that
                    // mismatch loudly — the two node kinds cannot disagree.
                    self.ensure_kv(
                        layer,
                        backend,
                        kv_elems / n_ctx.max(1),
                        kv_elems / n_ctx.max(1),
                        n_ctx,
                    )?;
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size)?;
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef::own(backend, pid, size));
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
                    self.ensure_kv(
                        layer,
                        backend,
                        kv_elems / n_ctx.max(1),
                        kv_elems / n_ctx.max(1),
                        n_ctx,
                    )?;
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size)?;
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef::own(backend, pid, size));
                    }
                }
                Op::Silu | Op::RoPE { .. } | Op::QkvBiasRopeStore { .. } | Op::SwiGLU => {
                    // D3-8: the mixed-quant QKV epilogue also needs the layer's
                    // persistent KV regions (it stores k/v like FusedQKV).
                    if let Op::QkvBiasRopeStore { layer } = &node.op {
                        let (kv_elems, n_ctx) = match &node.meta {
                            NodeMeta::QkvBiasRopeStore(m) => {
                                (m.kv_elems, m.kv_elems / m.nkt.max(1))
                            }
                            _ => (node.n_elements(), node.out_shape[1]),
                        };
                        // The epilogue stores K/V like the fusion, so it too hands in
                        // its logical width as its cell width (a packed format never
                        // reaches it: the graph would then also carry a store node
                        // whose region size disagrees, which `ensure_kv` refuses).
                        self.ensure_kv(
                            *layer,
                            backend,
                            kv_elems / n_ctx.max(1),
                            kv_elems / n_ctx.max(1),
                            n_ctx,
                        )?;
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
                        // D2: the FFN composition's SwiGLU writes into its *gate
                        // window* — the same bytes the fused node leaves its result
                        // in — so it joins the in-place set, but only when its first
                        // input is a view. A non-view SwiGLU (the fusion pass's, on
                        // CPU) keeps its own buffer, so existing graphs do not move.
                        let in_place_ok = match &node.op {
                            Op::SwiGLU => graph.node(node.src[0]).view.is_some(),
                            _ => true,
                        };
                        if in_place_ok && in_ref.backend == backend && n_consumers[node.src[0]] == 1
                        {
                            self.node_to_buf.insert(id, in_ref);
                            // the aliased input must stay alive through this
                            // node's consumers — and, if it is itself a view,
                            // so must the buffer it is a window of (D1)
                            let lu = last_use[id];
                            let grown = extend_through_views(graph, &mut last_use, node.src[0], lu);
                            self.extend_buffer_alive(&grown, lu);
                        } else {
                            let size = node.n_elements();
                            let pid = self.alloc_in_pool(backend, size)?;
                            self.buf_alive.insert((backend, pid), last_use[id]);
                            self.node_to_buf.insert(id, BufRef::own(backend, pid, size));
                        }
                    }
                }
                _ => {
                    if last_use[id] > i {
                        let size = node.n_elements();
                        let pid = self.alloc_in_pool(backend, size)?;
                        self.buf_alive.insert((backend, pid), last_use[id]);
                        self.node_to_buf.insert(id, BufRef::own(backend, pid, size));
                    }
                }
            }
        }
        Ok(())
    }

    /// Keep the pool's free-list deadline in step with a liveness extension (E4 S2):
    /// `buf_alive` is set from `last_use` when a buffer is allocated, so a node whose
    /// life was extended afterwards (an in-place alias or a D1 view sharing its buffer)
    /// must have its buffer's deadline moved too — otherwise `sweep` recycles a buffer
    /// another live node still reads. Rounding to size classes makes that match easy
    /// (any two shapes in a class), which is how the latent bug surfaced.
    fn extend_buffer_alive(&mut self, grown: &[NodeId], to: usize) {
        for &n in grown {
            if let Some(br) = self.node_to_buf.get(&n).copied() {
                if let Some(al) = self.buf_alive.get_mut(&(br.backend, br.id)) {
                    *al = (*al).max(to);
                }
            }
        }
    }

    /// Ask a pool for a buffer of `size` elements, after checking the request against the
    /// backend's memory budget (E4).
    ///
    /// The check is the ticket's safety half: `weights + pooled buffers + this allocation`
    /// is compared against the budget **before** the backend is touched, so a graph that
    /// cannot fit is refused with its numbers instead of surfacing later as a CUDA null
    /// pointer. The allocation is charged — and made — at its **size class**, the ladder
    /// `allocplan::class_size` defines, so the accounting and the pool agree on what was
    /// reserved and two shapes in one class share one buffer.
    ///
    /// #171: a failure-injection chokepoint. `MINFER_TEST_CALL_FAIL=alloc_in_pool`
    /// refuses here (before the pool is touched), and the site is observable through
    /// `testfail::checked("alloc_in_pool")`.
    fn alloc_in_pool(&mut self, backend: Backend, size: usize) -> Result<usize, String> {
        crate::testfail::note_checked("alloc_in_pool");
        crate::testfail::guard("alloc_in_pool")?;
        let want = allocplan::class_bytes(allocplan::class_size(size));
        if let Some(budget) = self.memory_budget(backend) {
            let pool = self.pool_bytes.get(&backend).copied().unwrap_or(0);
            let weights = self.weights_bytes(backend);
            if pool + weights + want > budget {
                return Err(format!(
                    "out of {backend:?} memory: {} MiB of weights + {} MiB of pooled buffers +                      {} MiB for this activation exceeds the {budget} byte budget ({} MiB);                      reduce --n-ctx/--n-batch, use a smaller quant, or free a session",
                    weights / (1024 * 1024),
                    pool / (1024 * 1024),
                    want / (1024 * 1024),
                    budget / (1024 * 1024)
                ));
            }
        }
        let before = self.pool_len_of(backend);
        let class = allocplan::class_size(size);
        let id = self.alloc_class_in_pool(backend, size)?;
        self.buf_bytes.insert((backend, id), want);
        self.buf_class.insert((backend, id), class);
        // E4 S2: `pool_bytes` is what the pool **holds**, not the sum of the requests it
        // served. A recycled class buffer is already resident, so charging it again would
        // make the report grow on every rebuild even though the pool did not — and the
        // budget gate would refuse graphs that fit.
        if self.pool_len_of(backend) > before {
            *self.pool_bytes.entry(backend).or_insert(0) += want;
        }
        let live = self.live_bytes.entry(backend).or_insert(0);
        *live += want;
        let peak = self.peak_bytes.entry(backend).or_insert(0);
        *peak = (*peak).max(*live);
        Ok(id)
    }

    /// Buffers resident in a backend's pool (E4 S2: the "did the pool grow?" probe behind
    /// the resident-bytes accounting).
    fn pool_len_of(&self, backend: Backend) -> usize {
        self.pool(backend).map_or(0, |p| p.pool_len())
    }

    /// The backend dispatch behind [`Self::alloc_in_pool`]: the pool is asked for the
    /// node's **size class**, not its exact element count (E4 S2).
    ///
    /// That is what makes recycling work: a pool's free list matches buffer lengths
    /// exactly, so two activation shapes in one class would each grow the pool while the
    /// other's buffer sat free. Rounded, the second shape finds the first one's buffer and
    /// the pool stops growing per shape. The node keeps its *logical* length in its
    /// [`BufRef`] — every consumer reads/writes that window, never the physical length —
    /// which is why this one line is the whole allocator change.
    ///
    /// Persistent regions (the KV arenas) do **not** come through here: they are sized
    /// exactly by `ensure_kv`, because a cell's width is a layout contract (`row_elems`),
    /// not a tuning knob.
    fn alloc_class_in_pool(&mut self, backend: Backend, size: usize) -> Result<usize, String> {
        let class = allocplan::class_size(size);
        // E4 S3 — assign: an idle buffer of this class from the reservation table. Only a
        // class with nothing idle asks the backend for a new buffer (reserve).
        if let Some(id) = self
            .slots
            .get_mut(&(backend, class))
            .and_then(|idle| idle.pop_first())
        {
            return Ok(id);
        }
        Ok(self.alloc_exact_in_pool(backend, class))
    }

    /// Bytes one backend's registered weights occupy (0 for a backend that does not
    /// track them) — the "weights" half of `weights + activations > budget`.
    fn weights_bytes(&self, backend: Backend) -> usize {
        self.pool(backend).map_or(0, |p| p.weights_bytes())
    }

    /// The memory budget for a backend (E4): an explicit one if it was set, else the
    /// device's *free* bytes with a quarter held back, else unbounded. The default is the
    /// device's own answer, not a guess: the query already reports what is left after
    /// every weight and KV region is resident.
    ///
    /// The device's answer is an explicit [`allocplan::DeviceMemory`], so a **failed**
    /// query can no longer arrive here as `free = 0`. It resolves through the pure
    /// [`allocplan::budget_decision`]: a reported read keeps the three-quarters default
    /// (issue #122's happy path, byte for byte), a measured zero still refuses, and a
    /// failed query falls back to weights-only accounting with its reason printed once
    /// (see `unaccounted_budget_note`). Nothing derives a budget from a non-measurement.
    fn memory_budget(&self, backend: Backend) -> Option<usize> {
        let explicit = self.budget.get(&backend).copied();
        #[cfg(feature = "cuda")]
        let mem = if backend == Backend::CUDA {
            match crate::cuda::CudaState::get() {
                Some(c) => c.device_memory(),
                // A CUDA graph with no device state is a configuration error the
                // assignment pass catches; the budget is simply unbounded here.
                None => allocplan::DeviceMemory::NoDevice,
            }
        } else {
            allocplan::DeviceMemory::NoDevice
        };
        #[cfg(not(feature = "cuda"))]
        let mem = allocplan::DeviceMemory::NoDevice;
        let decision = allocplan::budget_decision(explicit, &mem);
        if let Some(note) = &decision.note {
            unaccounted_budget_note(note);
        }
        decision.budget
    }

    /// Set (or clear, with `None`) a backend's memory budget. Tests and a future
    /// offload policy use it; `None` restores the default.
    pub fn set_memory_budget(&mut self, backend: Backend, budget: Option<usize>) {
        match budget {
            Some(b) => {
                self.budget.insert(backend, b);
            }
            None => {
                self.budget.remove(&backend);
            }
        }
    }

    /// E4's accounting for one backend: what is resident, what is live, the peak, and the
    /// budget the allocator checks against.
    pub fn memory_report(&self, backend: Backend) -> MemoryReport {
        let (mut idle_slots, mut reserved_classes) = (0usize, 0usize);
        for ((b, _), ids) in &self.slots {
            if *b != backend || ids.is_empty() {
                continue;
            }
            idle_slots += ids.len();
            reserved_classes += 1;
        }
        MemoryReport {
            weights_bytes: self.weights_bytes(backend),
            pool_bytes: self.pool_bytes.get(&backend).copied().unwrap_or(0),
            live_bytes: self.live_bytes.get(&backend).copied().unwrap_or(0),
            peak_live_bytes: self.peak_bytes.get(&backend).copied().unwrap_or(0),
            budget: self.memory_budget(backend),
            idle_slots,
            reserved_classes,
        }
    }

    /// Exact-size allocation (persistent KV regions and any caller that cannot be rounded).
    fn alloc_exact_in_pool(&mut self, backend: Backend, size: usize) -> usize {
        self.require_pool_mut(backend).alloc_buffer(size)
    }

    /// Fresh (never recycled) buffer on a backend's pool — split-boundary
    /// staging only. See Backend::alloc_fresh.
    fn alloc_fresh_in(&mut self, backend: Backend, size: usize) -> usize {
        self.require_pool_mut(backend).alloc_fresh(size)
    }

    fn free_in_pool(&mut self, backend: Backend, id: usize) {
        // E4: a freed buffer stays reserved (the pool is a high-water mark) but is no
        // longer live, so the accounting separates the two.
        if let Some(bytes) = self.buf_bytes.remove(&(backend, id)) {
            let live = self.live_bytes.entry(backend).or_insert(0);
            *live = live.saturating_sub(bytes);
        }
        // E4 S3 — reserve: a **classed** buffer goes back to the reservation table, not to
        // the backend. It keeps its bytes (the pool never returns memory anyway) and the next
        // graph needing that class is assigned this exact id, so a rebuild re-maps instead of
        // freeing and re-allocating. Staging buffers (exact, no class) keep the old path.
        if let Some(class) = self.buf_class.remove(&(backend, id)) {
            self.slots.entry((backend, class)).or_default().insert(id);
            return;
        }
        // A pool that is not enabled has nothing to free from (the pre-F4 Metal/CUDA
        // arms were no-ops in exactly that case).
        if let Some(pool) = self.pool_mut(backend) {
            pool.free_buffer(id);
        }
    }

    /// Per-layer KV persistent regions (K and V), created on first use on the
    /// layer's assigned backend.
    ///
    /// `n_embd` is the node's *logical* row width (`n_kv_embd`, what its K/V input
    /// carries) and `row_elems` is the width the region stores one cell in — the same
    /// value for f32/f16, a packed Q8_0 cell's word count under `MINFER_CACHE_TYPE=q8_0`
    /// (C4). The persistent region is `row_elems * n_ctx` elements.
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
        n_embd: usize,
        row_elems: usize,
        n_ctx: usize,
    ) -> Result<[BufRef; 2], String> {
        // `row_elems != n_embd` is what says "this node stores packed cells": the
        // graph is the authority (a builder under a Q8_0 policy stamps the packed
        // width), not a process global read a second time here.
        let packed = row_elems != n_embd;
        if packed {
            // Two loud checks, because both would otherwise corrupt silently: a row
            // width Q8_0 cannot express, and a KV node that does not carry the packed
            // width (the decode QKV fusion is GPU-only, so it hands in its logical
            // width) sizing a region whose kernel writes the other layout.
            let format = super::kvformat::KvFormat::Q8_0;
            format.check_width(n_embd)?;
            // F4: the capability is the registry's answer (`BackendCaps::
            // reads_packed_kv`), the same field `KvFormat::supports` reads — the
            // pre-F4 `backend != Backend::CPU` hardcode and the C4 format gate
            // can no longer disagree. C4 S2a gave the CPU the fused read and S2b
            // the CUDA kernels, so today the registry says yes for CPU and CUDA
            // and no for Metal ([#44], G5).
            //
            // [#44]: https://github.com/yusiwen/minfer/issues/44
            if !super::registry::reads_packed_kv(backend) {
                return Err(format!(
                    "KV region for layer {layer} would live on {backend:?}, which has no kernel \
                     that reads a packed {} region (the CPU and CUDA attention kernels do; Metal \
                     is G5 on issue #44); refusing rather than sizing a region its kernels would \
                     address as f32 rows",
                    format.name()
                ));
            }
            let expected = format.row_elems(n_embd);
            if row_elems != expected {
                return Err(format!(
                    "KV region for layer {layer}: the node declares {row_elems} words per cell but \
                     the {} layout packs one cell of {n_embd} elements into {expected}; only a node \
                     that carries `KvcacheMeta::row_elems` may size a packed region (the decode \
                     QKV fusion is GPU-only and never runs packed)",
                    format.name()
                ));
            }
        }
        let elems = row_elems * n_ctx;
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
            if region.packed != packed {
                return Err(format!(
                    "KV region for layer {layer} was allocated {} but this graph asks for {} \
                     (the KV format changed on a live GraphCache; the regions are persistent)",
                    if region.packed { "packed" } else { "unpacked" },
                    if packed {
                        "packed Q8_0"
                    } else {
                        "unpacked f32"
                    }
                ));
            }
            return Ok([region.k, region.v]);
        }
        let k = self.alloc_persistent(&format!("kv.{layer}.k"), backend, elems);
        let v = self.alloc_persistent(&format!("kv.{layer}.v"), backend, elems);
        self.kv
            .insert(layer, k, v, n_embd, row_elems, n_ctx, packed);
        Ok([k, v])
    }

    /// Allocate a persistent (never-freed) region on a backend.
    pub fn alloc_persistent(&mut self, name: &str, backend: Backend, size: usize) -> BufRef {
        let id = self.alloc_exact_in_pool(backend, size);
        self.persistent.push(PersistentBuf {
            name: name.to_string(),
            backend,
            id,
        });
        BufRef::own(backend, id, size)
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
        // C4 S2: a packed Q8_0 region is not f32 rows, so the re-rope below cannot
        // reinterpret its bytes. The survivors are still moved **verbatim** (a cell is
        // a whole number of words, which is exactly why that move needs no format
        // knowledge), and then each survivor's K row is dequantized, re-roped in f32
        // and quantized back — see `kvformat::map_q8_0_cells`.
        let packed = self.kv.any_packed();
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
            if packed {
                let nkt = self
                    .kv
                    .get(layer)
                    .map(|l| l.n_embd)
                    .ok_or_else(|| format!("kv_rm: no arena for layer {layer}"))?;
                let mut f32row = vec![0.0f32; nkt];
                super::kvformat::map_q8_0_cells(
                    &mut k,
                    nkt,
                    start,
                    new_used - start,
                    &mut f32row,
                    |r| super::kvcache::rope_shift_kv(r, 1, len as isize, rope),
                );
            } else {
                super::kvcache::rope_shift_kv(
                    &mut k[start * row..],
                    new_used - start,
                    len as isize,
                    rope,
                );
            }
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

    /// Reserve a contiguous cell run for `seq` (Phase E / E2). Several
    /// sequences share one arena this way; `Err` when no run fits — see
    /// [`Self::kv_reserve_seq_with_defrag`] for the C3 retry.
    pub fn kv_reserve_seq(
        &mut self,
        seq: super::kvcache::SeqId,
        cap: usize,
    ) -> Result<super::kvcache::SeqSlot, String> {
        self.kv.reserve_seq(seq, cap)
    }

    /// Reserve `cap` cells, compacting the arena once (C3) when first-fit
    /// cannot — fragmentation, not capacity, is what first-fit trips over.
    ///
    /// Returns the slot **and** the runs the compaction moved: a caller that
    /// keeps a run's `start` (E2's server keeps one per slot) must apply those
    /// moves to its own bookkeeping before the next forward, and the return
    /// type is what makes that unavoidable. `MINFER_NO_KV_DEFRAG` (presence
    /// checked, the A/B gate of standing rule 3) disables the retry.
    ///
    /// A compaction that moves nothing is a real answer ("it does not fit"), so
    /// the original reservation error is the one reported.
    pub fn kv_reserve_seq_with_defrag(
        &mut self,
        seq: super::kvcache::SeqId,
        cap: usize,
    ) -> Result<(super::kvcache::SeqSlot, Vec<super::kvcache::KvMove>), String> {
        let first = match self.kv.reserve_seq(seq, cap) {
            Ok(slot) => return Ok((slot, Vec::new())),
            Err(e) => e,
        };
        if !kv_defrag_enabled() {
            return Err(format!("{first} (KV defragmentation disabled)"));
        }
        let report = self.kv_defrag(Some(cap))?;
        if report.moves.is_empty() {
            return Err(first);
        }
        let slot = self.kv.reserve_seq(seq, cap).map_err(|e| {
            format!(
                "{e} (after a compaction that moved {} run(s): {first})",
                report.moves.len()
            )
        })?;
        Ok((slot, report.moves))
    }

    /// Compact the KV arena downward (C3): slide every live run to the lowest
    /// free gap, moving its written rows with [`BackendTrait::copy_cells`].
    ///
    /// `need = Some(n)` compacts only as far as opening a free run of `n` cells
    /// (the shortest prefix of the plan); `None` compacts fully. The data copy
    /// happens **before** the run table is renumbered, and a backend that cannot
    /// move cells fails the whole call — no renumbering without the copy.
    ///
    /// The returned moves are mandatory for whoever reserved a run: the graph
    /// itself needs nothing (every forward derives `attn_span` from the run
    /// table), but a caller that passes `start + pos` as a store position would
    /// otherwise write to cells that now belong to someone else.
    ///
    /// **No re-rope (C6).** `positions` are sequence-relative now, so a token's
    /// RoPE angle is its index within its sequence and a *cell* move changes
    /// nothing: the rows are copied verbatim and a mid-session compaction is
    /// bitwise identical. (Before C6 the angles were cell indices, which is why
    /// this function used to re-rope every moved K row — see the plan's C3 record.
    /// C2's `kv_rm`/`kv_shift` still re-rope: those move *positions*.)
    /// C7b: resize a sequence's run, moving whatever is in the way — in **either**
    /// direction — and return the moves the caller must follow. Rows are copied
    /// before the run table is renumbered, exactly like [`Self::kv_defrag`].
    pub fn kv_set_cap_with_defrag(
        &mut self,
        seq: super::kvcache::SeqId,
        cap: usize,
    ) -> Result<(super::kvcache::SeqSlot, Vec<super::kvcache::KvMove>), String> {
        let old = self
            .kv
            .seq_slot(seq)
            .map(|s| s.cap)
            .ok_or_else(|| format!("kv_set_cap: sequence {seq} holds no run"))?;
        let moves = self.kv.set_cap(seq, cap)?;
        if moves.is_empty() {
            let slot = self
                .kv
                .seq_slot(seq)
                .ok_or_else(|| "kv_set_cap: the run vanished".to_string())?;
            return Ok((slot, moves));
        }
        if !kv_defrag_enabled() {
            // The gate means "never compact": undo the resize and refuse loudly
            // rather than leave a run table the rows do not match.
            self.kv.set_cap(seq, old)?;
            return Err(format!(
                "resizing sequence {seq} to {cap} cells needs a compaction and KV \
                 defragmentation is disabled"
            ));
        }
        // Order matters wherever ranges overlap: runs moving **up** go top-down and
        // runs moving **down** bottom-up, so a destination never lands on a row that
        // has not been copied yet. The plan is emitted in ascending `from` order, so
        // reorder here and hand the same order to `apply_moves`, which shifts the
        // owner table by exactly these ranges.
        let mut ordered = moves.clone();
        super::kvcache::order_moves(&mut ordered);
        let regions: Vec<(BufRef, BufRef, usize)> = self
            .kv
            .iter()
            .map(|(_, l)| (l.k, l.v, (l.elems / l.n_ctx.max(1)).max(1)))
            .collect();
        for &(k, v, elems_per_cell) in &regions {
            for m in &ordered {
                if m.rows == 0 {
                    continue;
                }
                for region in [k, v] {
                    self.copy_cells_in_pool(
                        region.backend,
                        region,
                        region,
                        m.to,
                        m.from,
                        m.rows,
                        elems_per_cell,
                    )
                    .map_err(|e| {
                        format!(
                            "kv_set_cap: moving sequence {} rows {}..{} -> {} failed: {e}",
                            m.seq,
                            m.from,
                            m.from + m.rows,
                            m.to
                        )
                    })?;
                }
            }
        }
        self.kv.apply_moves(&ordered)?;
        let slot = self
            .kv
            .seq_slot(seq)
            .ok_or_else(|| "kv_set_cap: the run vanished".to_string())?;
        Ok((slot, ordered))
    }

    /// C8a: copy the first `rows` written K/V rows of `src` into `dst`'s run and
    /// give `dst` their ownership, so a prefix that another slot already computed is
    /// **not prefilled again**.
    ///
    /// This is the cheap half of cross-sequence sharing: the rows are duplicated
    /// (memory stays per-sequence) and **no read path changes**, because `dst` keeps
    /// one contiguous run — the copy exists precisely so it can. True sharing, where a
    /// block is stored once and the attention kernels gather it, is C8b.
    ///
    /// Both runs must be contiguous, which they are today (`cells[t] = start + t`); the
    /// move is bounded by `src`'s written rows and by `dst`'s capacity, and it fails
    /// loudly rather than copying a row that does not exist. The whole copy is within
    /// one region per layer, so `copy_cells`' overlap-safe ordering applies (a single
    /// move per region, in either direction).
    pub fn kv_copy_prefix(
        &mut self,
        src: super::kvcache::SeqId,
        dst: super::kvcache::SeqId,
        rows: usize,
    ) -> Result<(), String> {
        if rows == 0 {
            return Ok(());
        }
        let from = self
            .kv
            .seq_slot(src)
            .ok_or_else(|| format!("kv_copy_prefix: source sequence {src} holds no run"))?;
        // C8b S2: a copy is contiguous rows, so a source that reads part of its
        // prefix in place has no single run to copy. Refuse rather than copy the
        // wrong rows — sharing replaces this path where the device can gather.
        if from.shared.rows > 0 {
            return Err(format!(
                "kv_copy_prefix: source sequence {src} shares a {}-row prefix in place; there is \
                 no contiguous run to copy",
                from.shared.rows
            ));
        }
        let to = self
            .kv
            .seq_slot(dst)
            .ok_or_else(|| format!("kv_copy_prefix: destination sequence {dst} holds no run"))?;
        let written = self.kv.written_rows(src);
        if rows > written {
            return Err(format!(
                "kv_copy_prefix: sequence {src} has written {written} rows, {rows} requested"
            ));
        }
        if rows > to.cap {
            return Err(format!(
                "kv_copy_prefix: destination sequence {dst} reserved {} cells, {rows} requested",
                to.cap
            ));
        }
        if from.start == to.start {
            return Ok(()); // same run: nothing to copy
        }
        let regions: Vec<(BufRef, BufRef, usize, crate::graph::Backend)> = self
            .kv
            .iter()
            .map(|(_, l)| (l.k, l.v, (l.elems / l.n_ctx.max(1)).max(1), l.k.backend))
            .collect();
        for (k, v, elems_per_cell, backend) in regions {
            for region in [k, v] {
                self.copy_cells_in_pool(
                    backend,
                    region,
                    region,
                    to.start,
                    from.start,
                    rows,
                    elems_per_cell,
                )
                .map_err(|e| {
                    format!(
                        "kv_copy_prefix: copying {rows} rows of sequence {src} ({}) into \
                         sequence {dst} ({}) failed: {e}",
                        from.start, to.start
                    )
                })?;
            }
        }
        // One call for every layer: `own_range` sets the owner in all of them.
        self.kv.own_range(dst, to.start, to.start + rows);
        Ok(())
    }

    /// C8b S2: let `dst` read the first `rows` positions of `src` **in place** —
    /// no copy, one shared copy of the arena's bytes. Returns the cells the sharing
    /// pinned, for the caller's log. The store refuses what it cannot express (see
    /// `KvCache::share_prefix`), and the caller falls back to `kv_copy_prefix`.
    pub fn kv_share_prefix(
        &mut self,
        src: super::kvcache::SeqId,
        dst: super::kvcache::SeqId,
        rows: usize,
    ) -> Result<usize, String> {
        self.kv.share_prefix(src, dst, rows)?;
        Ok(rows)
    }

    /// C8b S3: copy-on-write — make position `t` of `seq` private because a store is
    /// about to land there and its current row belongs to the sequence it shares its
    /// prefix with.
    ///
    /// [`KvCache::private_row_for`] plans the rebase, this drives the data half with
    /// [`BackendTrait::copy_cells`], and [`KvCache::apply_private_row`] renumbers. The
    /// sequence's own rows move **up inside its own run** (never across sequences), and
    /// `copy_cells` is overlap-safe (C7b), so the shift needs no staging buffer. A
    /// `None` answer means `t` was already private and nothing moved; the donor's cells
    /// are never touched, which is the whole point.
    pub fn kv_private_row_for(
        &mut self,
        seq: super::kvcache::SeqId,
        t: usize,
    ) -> Result<Option<super::kvcache::KvShift>, String> {
        let Some(shift) = self.kv.private_row_for(seq, t)? else {
            return Ok(None);
        };
        if shift.rows > 0 {
            let regions: Vec<(BufRef, BufRef, usize, Backend)> = self
                .kv
                .iter()
                .map(|(_, l)| (l.k, l.v, (l.elems / l.n_ctx.max(1)).max(1), l.k.backend))
                .collect();
            for (k, v, elems_per_cell, backend) in regions {
                for region in [k, v] {
                    self.copy_cells_in_pool(
                        backend,
                        region,
                        region,
                        shift.to,
                        shift.from,
                        shift.rows,
                        elems_per_cell,
                    )
                    .map_err(|e| {
                        format!(
                            "kv_private_row_for: moving sequence {seq} rows {}..{} -> {} failed: \
                             {e}",
                            shift.from,
                            shift.from + shift.rows,
                            shift.to
                        )
                    })?;
                }
            }
        }
        self.kv.apply_private_row(&shift)?;
        Ok(Some(shift))
    }

    pub fn kv_defrag(&mut self, need: Option<usize>) -> Result<KvDefragReport, String> {
        let before = self.kv.arena_stats();
        let moves = self.kv.compaction_plan(need);
        if moves.is_empty() {
            return Ok(KvDefragReport {
                moves,
                rows_moved: 0,
                before: before.clone(),
                after: before,
            });
        }
        // Snapshot the regions: `kv.iter()` borrows the store, and the copies
        // below need `&mut self`.
        let regions: Vec<(BufRef, BufRef, usize)> = self
            .kv
            .iter()
            .map(|(_, l)| (l.k, l.v, (l.elems / l.n_ctx.max(1)).max(1)))
            .collect();
        for &(k, v, elems_per_cell) in &regions {
            for m in &moves {
                if m.rows == 0 {
                    continue;
                }
                for region in [k, v] {
                    self.copy_cells_in_pool(
                        region.backend,
                        region,
                        region,
                        m.to,
                        m.from,
                        m.rows,
                        elems_per_cell,
                    )
                    .map_err(|e| {
                        format!(
                            "kv_defrag: moving sequence {} rows {}..{} -> {} failed: {e}",
                            m.seq,
                            m.from,
                            m.from + m.rows,
                            m.to
                        )
                    })?;
                }
            }
        }
        let rows_moved = self.kv.apply_moves(&moves)?;
        let after = self.kv.arena_stats();
        Ok(KvDefragReport {
            moves,
            rows_moved,
            before,
            after,
        })
    }

    /// Fragmentation and utilisation counters for every arena layer's shared
    /// cell table (C3's acceptance surface; F8 exports the same numbers).
    pub fn kv_arena_stats(&self) -> super::kvcache::KvArenaStats {
        self.kv.arena_stats()
    }

    /// Bytes the persistent KV regions occupy, summed over layers and both regions
    /// (C4's footprint surface: a packed Q8_0 cache is measured against f32 here).
    pub fn kv_region_bytes(&self) -> usize {
        self.kv.region_bytes()
    }

    /// F8: layers with an allocated KV arena — one half of the arena shape
    /// (the other is [`Self::kv_n_ctx`]'s row count).
    pub fn kv_layer_count(&self) -> usize {
        self.kv.layer_count()
    }

    /// Whether the persistent KV regions store packed cells (C4).
    pub fn kv_is_packed(&self) -> bool {
        self.kv.any_packed()
    }

    /// Release everything `seq` reserved and owned; returns the freed capacity.
    pub fn kv_release_seq(&mut self, seq: super::kvcache::SeqId) -> usize {
        self.kv.release_seq(seq)
    }

    /// The run `seq` holds, or `None`.
    pub fn kv_seq_slot(&self, seq: super::kvcache::SeqId) -> Option<super::kvcache::SeqSlot> {
        self.kv.seq_slot(seq)
    }

    /// Mark cells `[from, to)` as written by `seq` in every layer (E2's batched
    /// forwards write several sequences per step).
    pub fn kv_own_range(&mut self, seq: super::kvcache::SeqId, from: usize, to: usize) {
        self.kv.own_range(seq, from, to);
    }

    /// Arena capacity in rows (`n_ctx`), 0 before allocation.
    pub fn kv_n_ctx(&self) -> usize {
        self.kv.n_ctx()
    }

    /// Declare the KV arena's capacity before the first `alloc_graph`, so
    /// sequences can be reserved up front (E2's server does this per slot).
    /// Must match the `n_ctx` the graphs are built with; `ensure_kv` re-checks
    /// the region size on every rebuild.
    pub fn kv_set_capacity(&mut self, n_ctx: usize) {
        self.kv.set_n_ctx(n_ctx);
    }

    /// Fill a batch's inputs: mark each sequence's written rows, fill `seq_ids`
    /// and resolve `attn_span` (Phase E / E2).
    ///
    /// A sequence with no reservation is refused unless it is the only one — the
    /// single-sequence path takes the whole arena, which is exactly E1's
    /// behaviour and keeps the classic forward bitwise; a batched caller must
    /// reserve each run first (`kv_reserve_seq`), so a missing reservation is a
    /// loud error rather than an overlap.
    pub fn fill_batch_inputs(
        &mut self,
        graph: &ComputeGraph,
        batch: &super::batch::Batch,
    ) -> Result<(), String> {
        // C8b S3: a store may not land in a prefix this sequence reads in place, so
        // every copy-on-write runs **before the first cell is resolved** — the rows
        // move inside the run, and a cell resolved before the move would be stale.
        // `private_row_for` is a no-op for a sequence with no run or nothing shared,
        // which is what keeps the classic single-sequence path untouched.
        for (seq, from, to) in batch.groups() {
            if let Some(t) = batch.positions[from..to].iter().copied().min() {
                self.kv_private_row_for(seq, t)?;
            }
        }
        for (seq, from, to) in batch.groups() {
            if self.kv.seq_slot(seq).is_none() {
                if batch.n_seqs() > 1 {
                    return Err(format!(
                        "sequence {seq} holds no KV cells; reserve it before batching                          (GraphAllocator::kv_reserve_seq)"
                    ));
                }
                let cap = self.kv.n_ctx();
                self.kv.reserve_seq(seq, cap)?;
            }
            // C6/C8b S2: `positions` are sequence-relative, so the rows this forward
            // writes are exactly those positions — resolved through the sequence's
            // span list, because a sharing sequence's cells are not `start + pos`.
            self.kv.own_positions(seq, &batch.positions[from..to])?;
        }
        let positions: Vec<usize> = batch.positions.clone();
        self.fill_seq_ids(graph, &batch.seq_ids, &positions)
    }

    /// Fill the E1 attention inputs and record how far this forward writes:
    /// `positions` are cell indices, `seq_ids` names each query's sequence.
    ///
    /// This is the one call a caller needs — the model path and hand-built graphs
    /// both use it — so the seq ids, the resolved span and the store's ownership
    /// cannot drift apart. `max(positions) + 1` is exactly the row count the KV
    /// store writes in this forward, recorded *before* the span is resolved
    /// because the span has to describe the cache the store is about to fill.
    pub fn fill_attn_inputs(
        &mut self,
        graph: &ComputeGraph,
        seq_ids: &[u32],
        positions: &[u32],
    ) -> Result<(), String> {
        let pos: Vec<usize> = positions.iter().map(|&p| p as usize).collect();
        // C6: the written extent is a *cell* extent, resolved through the runs —
        // but only a graph that actually stores K/V has a `cells` input (a
        // rope-only fixture does not, and must not need a KV arena).
        let has_cells = graph.inputs.iter().any(|&i| graph.node(i).name == "cells");
        if has_cells {
            // C8b S3: same rule as `fill_batch_inputs` — the copy-on-write runs before
            // any cell is resolved, keyed on each sequence's lowest position here. A
            // graph with no store has nothing to write, so it needs no copy either.
            let mut first: Vec<(u32, usize)> = Vec::new();
            for (&seq, &p) in seq_ids.iter().zip(&pos) {
                match first.iter_mut().find(|(s, _)| *s == seq) {
                    Some((_, low)) => *low = (*low).min(p),
                    None => first.push((seq, p)),
                }
            }
            for (seq, t) in first {
                self.kv_private_row_for(seq, t)?;
            }
        }
        if has_cells && !pos.is_empty() {
            let cells = self.kv_cells_for_seq(seq_ids, &pos)?;
            if let Some(&maxc) = cells.iter().max() {
                self.kv_note_used(maxc as usize + 1);
            }
        }
        self.fill_seq_ids(graph, seq_ids, &pos)
    }

    /// Resolve each query's **cell** from the store: `start + position`, for the
    /// runs the caller reserved (C6). This is what the KV store writes to, while
    /// `positions` stays the token's index within its sequence (what RoPE needs).
    pub fn kv_cells_for_seq(
        &self,
        seq_ids: &[u32],
        positions: &[usize],
    ) -> Result<Vec<u32>, String> {
        if seq_ids.len() != positions.len() {
            return Err(format!(
                "kv_cells_for_seq: {} sequence ids but {} positions",
                seq_ids.len(),
                positions.len()
            ));
        }
        let mut cells = Vec::with_capacity(positions.len());
        // The classic single-sequence path holds no reservation: the sequence owns
        // the whole arena, so `cell == position` (C1's identity). Any other
        // unreserved sequence is an error — the same rule `fill_batch_inputs`
        // applies when it reserves implicitly.
        let classic = self.kv.arena_stats().sequences == 0;
        let n_ctx = self.kv.n_ctx();
        for (t, (&seq, &rel)) in seq_ids.iter().zip(positions).enumerate() {
            let Some(slot) = self.kv.seq_slot(seq) else {
                if !classic {
                    return Err(format!(
                        "kv_cells_for_seq: query {t} has no reserved run (seq {seq})"
                    ));
                }
                if n_ctx == 0 {
                    return Err("kv_cells_for_seq: no KV arena allocated".to_string());
                }
                if rel >= n_ctx {
                    return Err(format!(
                        "kv_cells_for_seq: query {t} position {rel} is past the {n_ctx}-cell arena"
                    ));
                }
                cells.push(rel as u32);
                continue;
            };
            // C8b S3: a run holds positions `[shared.rows, shared.rows + cap)` — the
            // lower positions are read from the donor's cells, and a store there would
            // write **through** the shared prefix and corrupt every sharer. The caller
            // has to copy-on-write first (`kv_private_row_for`), which the two fill
            // entry points do; anything else is a loud error, never a silent write.
            if rel < slot.shared.rows {
                return Err(format!(
                    "kv_cells_for_seq: query {t} position {rel} would be written into sequence \
                     {seq}'s {}-row shared prefix (cells {}..{}); a store must not write through \
                     a shared prefix (C8b S3 copies the row first)",
                    slot.shared.rows,
                    slot.shared.cell,
                    slot.shared.cell + slot.shared.rows
                ));
            }
            if rel >= slot.shared.rows + slot.cap {
                return Err(format!(
                    "kv_cells_for_seq: query {t} position {rel} is past sequence {seq}'s \
                     reserved run ({} cells at {}, holding positions {}..{})",
                    slot.cap,
                    slot.start,
                    slot.shared.rows,
                    slot.shared.rows + slot.cap
                ));
            }
            // C8b S1: resolve through the sequence's span list. With one span this is
            // exactly `slot.start + rel`; a list that does not cover the position is an
            // invariant violation, so it is refused loudly instead of falling back to
            // the contiguous form.
            let cell = self.kv.cell_of(seq, rel).ok_or_else(|| {
                format!(
                    "kv_cells_for_seq: query {t} position {rel} is outside sequence {seq}'s \
                     span list ({:?})",
                    self.kv.spans_of(seq)
                )
            })?;
            cells.push(cell as u32);
        }
        Ok(cells)
    }

    /// The cell position `pos` of sequence `seq` lives in, or `None` when no span of
    /// that sequence covers it (C8b S1).
    ///
    /// The read-side twin of [`Self::kv_cells_for_seq`]: that one is the **store**
    /// resolver and refuses a position inside a shared prefix (a write there would go
    /// through to the donor), while this answers "where would a reader look?", which
    /// is what a caller snapshotting a sharing sequence's rows needs.
    pub fn kv_cell_of(&self, seq: super::kvcache::SeqId, pos: usize) -> Option<usize> {
        self.kv.cell_of(seq, pos)
    }

    /// Fill the per-query sequence ids and resolve the matching `attn_span`
    /// input from the KV cell store (Phase E / E1).
    ///
    /// `attn_span` is *derived*, never supplied by the caller: the store owns the
    /// per-sequence cell ranges, so resolving both here is what keeps the IR's
    /// seq ids and the kernel's window from disagreeing. A graph with no
    /// attention (no `attn_span` input) only records the ids.
    pub fn fill_seq_ids(
        &mut self,
        graph: &ComputeGraph,
        seq_ids: &[u32],
        positions: &[usize],
    ) -> Result<(), String> {
        let has = |name: &str| graph.inputs.iter().any(|&i| graph.node(i).name == name);
        if has("seq_ids") {
            self.fill_input_i32(graph, "seq_ids", seq_ids)?;
        }
        // C6: the store writes at the cells this resolves, while `positions`
        // (already filled by the caller) stays sequence-relative.
        if has("cells") {
            let cells = self.kv_cells_for_seq(seq_ids, positions)?;
            self.fill_input_i32(graph, "cells", &cells)?;
        }
        if has("kv_map") {
            // C8b S2: this graph reads its window as a list of cell runs, which is
            // what a sequence sharing a prefix needs (one `[lo, hi)` cannot name it).
            let map = self.kv.attn_map(seq_ids, positions)?;
            return self.fill_input_i32(graph, "kv_map", &map);
        }
        if !has("attn_span") {
            return Ok(());
        }
        let span = self.kv.attn_span(seq_ids, positions)?;
        self.fill_input_i32(graph, "attn_span", &span)
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
    /// `fill_input_i32`). An input consumed by `Op::Attn` is the E1 **span**
    /// (`[lo, hi)` pairs, checked as such); one consumed by the KV-writing ops is
    /// positions, and the `cells` input they also consume is the **arena row** —
    /// bounded by the arena, with its own message, because the two only coincide
    /// while a cell equals its position. Everything else (`token_ids`, `seq_ids`, …)
    /// is unbounded — vocabularies and sequence counts are routinely larger than
    /// `n_ctx`.
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
        // The span input's name is the IR contract (`fill_seq_ids` looks it up
        // that way too), and it is the fourth input of an `Attn` node: `positions`
        // stays at index 2 for the backends that still derive from it.
        if name == "attn_span" {
            return self.check_attn_span(graph, name, data);
        }
        if name == "cells" {
            let n_ctx = self.kv.n_ctx();
            if n_ctx == 0 {
                // No arena to bound against; `fill_seq_ids` reports that case when it
                // tries to resolve the input, so nothing is silent here.
                return Ok(());
            }
            if let Some(&cell) = data.iter().max() {
                if cell as usize >= n_ctx {
                    return Err(format!(
                        "input 'cells': cell {cell} is past the {n_ctx}-cell arena \
                         (`docs/ARCHITECTURE-ROADMAP.md` §2.4)"
                    ));
                }
            }
            return Ok(());
        }
        if name == "kv_map" {
            // C8b S2: `(cell, len)` runs, not positions — the generic arm below
            // would resolve them through the store as if they were token indices.
            let n_ctx = self.kv.n_ctx();
            for pair in data.chunks_exact(2) {
                let (cell, len) = (pair[0] as usize, pair[1] as usize);
                if len == 0 {
                    continue; // padding slot
                }
                if cell + len > n_ctx {
                    return Err(format!(
                        "input 'kv_map': run [{cell}, {}) is past the {n_ctx}-cell arena",
                        cell + len
                    ));
                }
            }
            return Ok(());
        }
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

    /// Validate the E1 attention span: `2 * nt` values, `lo <= hi <= n_ctx`. The
    /// arena size comes from the cell store of the attention node's layer, so the
    /// check keeps meaning the same thing if the mapping stops being the identity.
    fn check_attn_span(
        &self,
        graph: &ComputeGraph,
        name: &str,
        data: &[u32],
    ) -> Result<(), String> {
        if data.len() % 2 != 0 {
            return Err(format!(
                "input '{name}': attention span has {} values (expected lo/hi pairs)",
                data.len()
            ));
        }
        let layer = graph.nodes.iter().find_map(|n| match (&n.op, &n.meta) {
            (Op::Attn { .. }, NodeMeta::Attn(m)) => Some(m.layer),
            _ => None,
        });
        let Some(n_ctx) = layer
            .and_then(|l| self.kv.get(l))
            .map(|l| l.n_ctx)
            .or_else(|| {
                graph.nodes.iter().find_map(|n| match &n.op {
                    Op::KvcacheStore { layer } => self.kv.get(*layer).map(|l| l.n_ctx),
                    _ => None,
                })
            })
        else {
            return Ok(()); // no arena yet: nothing to bound against
        };
        let n = data.len() / 2;
        for t in 0..n {
            let (lo, hi) = (data[t] as usize, data[n + t] as usize);
            if lo > hi || hi > n_ctx {
                return Err(format!(
                    "input '{name}': query {t} has span [{lo}, {hi}) outside the {n_ctx}-cell arena"
                ));
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
        // E4 S2: the length contract lives in the node's `BufRef`. The pool buffer is
        // rounded up to the node's size class, so an equality check against the physical
        // length would refuse every correct fill — and a prefix-only check would accept a
        // wrong-sized one. The logical length is the one number that answers both.
        if data.len() != br.len {
            return Err(format!(
                "input '{name}': {} elements were supplied but the node holds {} (its pool buffer \
                 is rounded up to the {} element class)",
                data.len(),
                br.len,
                allocplan::class_size(br.len)
            ));
        }
        let (backend, id, offset) = (br.backend, br.id, br.offset);
        match self.pool_mut(backend) {
            Some(pool) => pool.write_host_window(id, offset, data),
            None => Err(format!(
                "{} is not usable on this allocator: {}",
                backend.name(),
                super::registry::unavailable_reason(backend)
                    .unwrap_or("the backend's pool is not enabled")
            )),
        }
    }

    /// Host view of a CPU node's buffer (Metal nodes: use copy_to_cpu).
    /// (Test helper.) The reference is a **window**: since E4 S2 a pool buffer is
    /// rounded up to its size class, so the physical slice is longer than the
    /// node and handing it back whole would leak another allocation's padding
    /// into every comparison.
    #[allow(dead_code)]
    pub fn get_buffer(&self, _graph: &ComputeGraph, id: NodeId) -> Option<&[f32]> {
        let br = self.node_buffer(id)?;
        // Only the CPU pool lends a host slice: a device pool's read is a copy
        // (`copy_to_cpu`), which cannot be returned by reference.
        if br.backend != Backend::CPU {
            return None;
        }
        self.cpu
            .read_host(br.id)?
            .get(br.offset..br.offset + br.len)
    }

    /// Host copy of any node's buffer (cross-backend reads).
    pub fn copy_to_cpu(&mut self, id: NodeId) -> Option<Vec<f32>> {
        let br = self.node_buffer(id)?;
        // D1: a view's reference is a window, so read exactly it — the parent's
        // buffer is longer (and may hold another view's data).
        let window = |v: Vec<f32>| -> Vec<f32> { v[br.offset..br.offset + br.len].to_vec() };
        // F4: one call for every backend — the entry's `host_read` is CPU/Metal's
        // borrowed read and CUDA's stream-ordered `copy_to_host`.
        (br.backend.entry()?.host_read)(self, br.id).map(window)
    }

    /// Host copy of a layer's persistent KV regions (K, V) by pool buffer
    /// id — doc 95 identity debugging (the graph holds one KvcacheLoad node
    /// per layer, so the V half is reachable only through `kv_pair`).
    pub fn copy_kv_to_cpu(&mut self, layer: usize) -> Option<(Vec<f32>, Vec<f32>)> {
        // Kept CPU/CUDA-only as before (the pre-F4 arm for Metal was `None`): this
        // is a CPU identity-debug helper, and widening it is not this ticket's job.
        let rd = |s: &mut Self, br: BufRef| -> Option<Vec<f32>> {
            if br.backend != Backend::CPU && br.backend != Backend::CUDA {
                return None;
            }
            (br.backend.entry()?.host_read)(s, br.id)
        };
        let l = self.kv.get(layer)?;
        let pair = [l.k, l.v];
        Some((rd(self, pair[0])?, rd(self, pair[1])?))
    }

    /// The KV element type a session on `backend` stores its rows in (C5). A
    /// session records it so a file written under one width cannot be resumed
    /// under another — an f16 region's second half is not meaningful data.
    ///
    /// F4: the answer is the registry entry's `kv_format` hook. A backend this
    /// build does not contain answers F32, as the pre-F4 `#[cfg]` arms did.
    /// Per-engine (issue #99): the CPU hook is this allocator's own stamped format;
    /// the device hooks still read the process-wide device layout.
    fn kv_element_format(&self, backend: Backend) -> KvFormat {
        match backend.entry() {
            Some(entry) => (entry.kv_format)(self),
            None => KvFormat::F32,
        }
    }

    /// Write the whole KV session — every layer's K/V region bytes plus the
    /// arena's run table, owner map and written extents — to `path` (C5).
    ///
    /// The header records the shape, the KV element type and the backend, so a
    /// load can refuse a file that does not describe *this* arena instead of
    /// applying it. All layers must agree on the shape: a session is one arena.
    pub fn kv_save(&mut self, path: &Path) -> Result<KvSessionReport, String> {
        self.kv_save_with_host(path, &[])
    }

    /// [`Self::kv_save`] with the caller's **host state** attached (C5 S2): the KV rows
    /// belong to a host state (a conversation, a server slot), so the container carries
    /// both and a restore that found only one of them is refused by the reader's own
    /// checks instead of resuming a session nobody owns. The bytes are opaque here.
    pub fn kv_save_with_host(
        &mut self,
        path: &Path,
        host: &[u8],
    ) -> Result<KvSessionReport, String> {
        let layers: Vec<usize> = self.kv.iter().map(|(l, _)| l).collect();
        let first = self
            .kv
            .get(*layers.first().ok_or("KV session: no KV arena to save")?)
            .ok_or("KV session: no KV arena to save")?;
        let (n_ctx, n_embd, backend) = (first.n_ctx, first.n_embd, first.k.backend);
        let row_elems = first.elems / first.n_ctx.max(1);
        let format = self.kv_element_format(backend);
        let header = KvSessionHeader {
            version: VERSION,
            format,
            backend,
            n_layer: layers.len(),
            n_ctx,
            n_embd,
            row_elems,
        };
        for &layer in &layers {
            let l = self.kv.get(layer).ok_or("KV session: layer vanished")?;
            if l.n_ctx != n_ctx || l.n_embd != n_embd || l.k.backend != backend {
                return Err(format!(
                    "KV session: layer {layer} is {}x{} cells on {:?} while the first layer is \
                     {n_ctx}x{n_embd} on {backend:?} — a session is one arena",
                    l.n_ctx, l.n_embd, l.k.backend
                ));
            }
            if l.elems / l.n_ctx.max(1) != row_elems {
                return Err(format!(
                    "KV session: layer {layer} stores {} words per cell, the first layer \
                     {row_elems}",
                    l.elems / l.n_ctx.max(1)
                ));
            }
        }
        let mut w = KvSessionWriter::create(path, &header)?;
        w.set_host(host);
        for &layer in &layers {
            let (k, v) = self
                .copy_kv_to_cpu(layer)
                .ok_or_else(|| format!("KV session: layer {layer} host read failed"))?;
            w.layer(layer, &k, &v)?;
        }
        w.finish(&self.kv.session_state())
    }

    /// Restore this allocator's KV arena from a session file (C5), creating the
    /// regions if they do not exist yet.
    ///
    /// The file is walked end to end (header, payload lengths, bookkeeping,
    /// checksum, end-of-file) **before** a single byte is written into a pool, so
    /// a truncated, corrupted or foreign file leaves the allocator untouched. The
    /// header must match what the caller knows from the model (`expect`).
    pub fn kv_load(
        &mut self,
        path: &Path,
        expect: &KvSessionExpect,
    ) -> Result<KvSessionReport, String> {
        self.kv_load_with_host(path, expect)
            .map(|(_, report)| report)
    }

    /// [`Self::kv_load`], also handing back the **host state** the file carries (C5 S2).
    /// The returned `Vec<u8>` is whatever [`Self::kv_save_with_host`] was given; a file
    /// with no host state returns an empty one, and the caller decides whether that is
    /// acceptable (the CLI's resume path treats it as "no companion state" and re-seeds).
    pub fn kv_load_with_host(
        &mut self,
        path: &Path,
        expect: &KvSessionExpect,
    ) -> Result<(Vec<u8>, KvSessionReport), String> {
        // E5: a session is one arena — the container carries a single backend tag and one
        // element type. A mixed CPU/device offload plan has regions on both, so the file's
        // backend would be forced onto layers that run elsewhere. Refused before anything
        // is applied (like every other `kv_load` refusal: a failed load is a no-op).
        if let Some(p) = self.offload {
            if p.is_mixed() {
                return Err(format!(
                    "KV session: this model's {} blocks are split across backends (E5 offload plan: {} on the device, {} on the CPU); a session is one arena, so it cannot be resumed — offload every block or none",
                    p.n_layers,
                    p.gpu_layers,
                    p.cpu_layers()
                ));
            }
        }
        let header = super::kvsession::verify(path)?;
        if header.backend != expect.backend {
            return Err(format!(
                "KV session: the file holds a {:?} session, this run uses {:?}",
                header.backend, expect.backend
            ));
        }
        if header.n_ctx != expect.n_ctx {
            return Err(format!(
                "KV session: the file describes a {}-cell arena, this run has {} (--n-ctx)",
                header.n_ctx, expect.n_ctx
            ));
        }
        if header.n_embd != expect.n_embd {
            return Err(format!(
                "KV session: the file holds {}-element KV rows, this model has {} (a session \
                 belongs to the model that wrote it)",
                header.n_embd, expect.n_embd
            ));
        }
        let live = self.kv_element_format(expect.backend);
        if header.format != live {
            return Err(format!(
                "KV session: the file was written with the {} KV element type, this run uses \
                 {} (MINFER_CACHE_TYPE)",
                header.format.name(),
                live.name()
            ));
        }
        let mut r = KvSessionReader::open(path)?;
        // A restore happens before the first graph is built, so the pool the file
        // names may not exist yet: enable it here (the same lazy enable the graph
        // builder does), or refuse when this build/box cannot have it.
        //
        // F4: one path for every backend — the entry's `enable` hook — with the
        // refusal still distinguishing "not compiled into this build" from
        // "compiled in but unavailable here", because those are different facts.
        match expect.backend.entry() {
            None => {
                return Err(format!(
                    "KV session: the file holds a {} session and this build has no {} backend ({})",
                    expect.backend.name(),
                    expect.backend.name(),
                    super::registry::unavailable_reason(expect.backend)
                        .unwrap_or("not compiled into this build")
                ));
            }
            Some(entry) => {
                if !(entry.enable)(self) {
                    return Err(format!(
                        "KV session: the file holds a {} session but {} is not available here: {}",
                        expect.backend.name(),
                        expect.backend.name(),
                        super::registry::unavailable_reason(expect.backend)
                            .unwrap_or("the backend's pool could not be enabled")
                    ));
                }
            }
        }
        let mut seen = 0usize;
        while let Some((layer, k, v)) = r.next_layer()? {
            let [kref, vref] = self.ensure_kv(
                layer,
                expect.backend,
                header.n_embd,
                header.row_elems,
                header.n_ctx,
            )?;
            self.write_pool(kref.backend, kref.id, &k)?;
            self.write_pool(vref.backend, vref.id, &v)?;
            seen += 1;
        }
        let body = r.finish()?;
        let (state, report) = (body.state, body.report);
        if state.layers.len() != seen {
            return Err(format!(
                "KV session: {seen} layers were read but the bookkeeping describes {}",
                state.layers.len()
            ));
        }
        self.kv.restore_session(&state)?;
        Ok((body.host, report))
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

    /// E4 S3: how many pool buffers the CPU backend has created (a re-map creates none).
    /// (Test helper.)
    #[allow(dead_code)]
    pub fn n_cpu_allocs(&self) -> usize {
        self.cpu.alloc_count()
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
        // The CPU's `synchronize` is a no-op, and a pool that is not enabled has
        // no pending work — so one path serves every backend.
        if let Some(pool) = self.pool_mut(backend) {
            pool.synchronize();
        }
    }

    /// F5 ([#58]) **phase A** of a cross-backend staging copy: *enqueue* the
    /// transfer of a node's buffer into `dst_backend`'s staging buffer.
    ///
    /// The node's canonical buffer (node_to_buf) is left untouched — the copy
    /// lands in the `cross` staging map, which the scheduler consults when a
    /// CONSUMER on `dst_backend` resolves its inputs. Consumers on the node's
    /// own backend keep reading the original buffer. This keeps a reused graph
    /// re-executable: the producing split always finds its buffer where the
    /// allocator put it, and the staging buffer (allocated once per graph
    /// rebuild) is simply rewritten on each execute.
    ///
    /// **The transfer itself is the source backend's registered hook**
    /// (`BackendEntry::copy_cross`), so there is no `match backend` here: a
    /// backend with device memory enqueues it asynchronously and records an
    /// event, a backend without device memory performs the synchronous host round
    /// trip it always did, and a backend that has not been ported *declines*
    /// (`Ok(false)`) and gets the synchronous path. The entry is inserted into
    /// `cross_pending` **before** the hook runs, so the contract "every staged
    /// input owes exactly one wait" holds even when the hook fails.
    ///
    /// Phase B is [`Self::await_cross`]; the consumer's read is
    /// [`Self::cross_input`], which refuses to hand out a still-pending entry.
    pub fn copy_across(
        &mut self,
        uid: u64,
        node_id: NodeId,
        dst_backend: Backend,
    ) -> Result<(), String> {
        let br = self
            .node_buffer(node_id)
            .ok_or_else(|| format!("node {node_id} has no buffer"))?;
        if br.backend == dst_backend {
            return Ok(());
        }
        let dst = self.cross_staging(uid, node_id, dst_backend)?;
        self.cross_pending.insert((uid, node_id, dst_backend));
        self.cross_stats.copies += 1;
        if copystats::async_copies_enabled() {
            let entry = br.backend.entry().ok_or_else(|| {
                format!(
                    "copy_across: {} is not compiled into this build",
                    br.backend.name()
                )
            })?;
            if (entry.copy_cross)(self, uid, node_id, dst_backend)? {
                return Ok(());
            }
        }
        // The synchronous host round trip: the pre-F5 path, the metal path until
        // it is ported, and the `MINFER_SYNC_COPIES=1` reference side of the
        // bitwise A/B.
        self.copy_across_blocking(node_id, dst)
    }

    /// The staging buffer a boundary copies `node_id` into for `dst_backend`,
    /// allocating it on first use. One staging buffer per (graph, node,
    /// destination backend) per graph: the size is part of the entry's identity (a
    /// graph's shapes never change under one uid, so the mismatch branch is a
    /// backstop, not a hot path).
    fn cross_staging(
        &mut self,
        uid: u64,
        node_id: NodeId,
        dst_backend: Backend,
    ) -> Result<BufRef, String> {
        let len = self
            .node_buffer(node_id)
            .ok_or_else(|| format!("node {node_id} has no buffer"))?
            .len;
        let key = (uid, node_id, dst_backend);
        if let Some(&cb) = self.cross.get(&key) {
            if cb.len == len {
                return Ok(cb);
            }
            // The size is part of the entry's identity (a graph's shapes never
            // change under one uid, so this is a backstop, not a hot path).
            self.cross.remove(&key);
            self.free_in_pool(cb.backend, cb.id);
        }
        let id = self.alloc_fresh_in(dst_backend, len);
        // Staging is exact (it is not an activation, so the class ladder does not
        // apply), but it is still pool memory the budget must see: charge it as
        // resident and live until the next rebuild frees it (E4 S2).
        let bytes = len * 4;
        self.buf_bytes.insert((dst_backend, id), bytes);
        *self.pool_bytes.entry(dst_backend).or_insert(0) += bytes;
        let live = self.live_bytes.entry(dst_backend).or_insert(0);
        *live += bytes;
        let peak = self.peak_bytes.entry(dst_backend).or_insert(0);
        *peak = (*peak).max(*live);
        let staged = BufRef::own(dst_backend, id, len);
        self.cross.insert(key, staged);
        Ok(staged)
    }

    /// F5: the **pre-F5 synchronous** host round trip of one staging copy — host
    /// read of the source through the source entry's `host_read`, then a write
    /// into the destination pool. Reached by `MINFER_SYNC_COPIES=1`, by a backend
    /// whose phase-A hook declined, and by [`Self::host_round_trip_cross`] (the
    /// CPU source's registered phase A).
    ///
    /// A device source's read here is a blocking copy *by construction*
    /// (`copy_to_host` syncs the stream, then issues a blocking `cudaMemcpy`), so
    /// it is what the `blocking_host_copies` counter counts — that counter is the
    /// "before" number of the ticket's acceptance line.
    fn copy_across_blocking(&mut self, node_id: NodeId, dst: BufRef) -> Result<(), String> {
        let br = self
            .node_buffer(node_id)
            .ok_or_else(|| format!("node {node_id} has no buffer"))?;
        if br.backend != Backend::CPU {
            self.cross_stats.blocking_host_copies += 1;
        }
        let data = self
            .copy_to_cpu(node_id)
            .ok_or_else(|| format!("node {node_id} host read failed"))?;
        self.write_cross_staging(dst, &data)
    }

    /// F5: the CPU source's registered **phase A** (`cpu_backend::copy_cross`) —
    /// the synchronous host round trip, named for what it is. The CPU has no
    /// device memory, so there is no transfer to make asynchronous; the device leg
    /// of a CPU→device copy is the destination pool's own stream-ordered
    /// `write_host`, which never blocked the host either.
    pub fn host_round_trip_cross(
        &mut self,
        uid: u64,
        node_id: NodeId,
        dst_backend: Backend,
    ) -> Result<(), String> {
        let dst = self.cross_staging(uid, node_id, dst_backend)?;
        self.copy_across_blocking(node_id, dst)
    }

    /// F5 **phase B**: wait on the event [`Self::copy_across`]'s hook recorded,
    /// exactly once per staged input, before the consuming split executes.
    ///
    /// The wait itself is the source backend's registered `await_cross` hook: a
    /// device source waits on its recorded event (a host block for a device→host
    /// copy — the one documented synchronization point — or a device-side
    /// `cudaStreamWaitEvent` for a device consumer), a backend without device
    /// memory is a documented no-op, and a backend that declined phase A has
    /// nothing to wait for.
    ///
    /// The allocator owns the *contract*: it counts the wait and clears the
    /// pending flag regardless of backend, which is what makes "the boundary path
    /// issues a wait for every staged input" a backend-independent, CI-covered
    /// assertion (see `scheduler`'s boundary and this module's tests). The pending
    /// flag is cleared **after** the hook succeeds, so a failed wait leaves the
    /// entry pending and the consumer's read errors instead of reading garbage.
    pub fn await_cross(
        &mut self,
        uid: u64,
        node_id: NodeId,
        dst_backend: Backend,
    ) -> Result<(), String> {
        let br = self
            .node_buffer(node_id)
            .ok_or_else(|| format!("node {node_id} has no buffer"))?;
        if br.backend == dst_backend {
            // Not a staged copy: the consumer already reads the canonical buffer
            // on its own backend. Clear defensively, so an `await_cross` is
            // idempotent and a stale flag can never outlive its boundary.
            self.cross_pending.remove(&(uid, node_id, dst_backend));
            return Ok(());
        }
        let hook = br
            .backend
            .entry()
            .ok_or_else(|| {
                format!(
                    "await_cross: {} is not compiled into this build",
                    br.backend.name()
                )
            })?
            .await_cross;
        hook(self, uid, node_id, dst_backend)?;
        self.cross_pending.remove(&(uid, node_id, dst_backend));
        self.cross_stats.waits += 1;
        Ok(())
    }

    /// Handle asynchronously-fetched bytes to the destination staging buffer
    /// (F5's phase B writes what phase A transferred).
    pub(crate) fn write_cross_staging(&mut self, dst: BufRef, data: &[f32]) -> Result<(), String> {
        self.write_pool(dst.backend, dst.id, data)
    }

    /// F5: what this allocator's split boundaries did. The gate reads a **delta**
    /// around one workload ([`CrossCopyStats::delta`]), because the counters are
    /// cumulative for the allocator's life. (The F5 gates are tests, so the
    /// production build has no caller.)
    #[allow(dead_code)]
    pub fn cross_stats(&self) -> CrossCopyStats {
        self.cross_stats
    }

    /// F5: the counters are mutated by the boundary and by the registry hooks;
    /// this is the hook-facing accessor.
    pub fn cross_stats_mut(&mut self) -> &mut CrossCopyStats {
        &mut self.cross_stats
    }

    /// F5: forget what the boundaries did so far (a gate that wants an absolute
    /// number rather than a delta).
    #[allow(dead_code)]
    pub fn reset_cross_stats(&mut self) {
        self.cross_stats = CrossCopyStats::default();
    }

    /// F5: the staged buffer a consumer on `backend` must read for `node_id`.
    ///
    /// Unlike [`Self::cross_buffer`], this **refuses** an entry whose phase-B wait
    /// has not been issued: reading it would read a transfer that may still be in
    /// flight. The scheduler uses this (never the raw accessor) precisely so that
    /// dropping the boundary's wait is a loud, named failure — "no host-side
    /// blocking copy" must never be bought with a missing synchronization.
    pub fn cross_input(
        &self,
        uid: u64,
        node_id: NodeId,
        backend: Backend,
    ) -> Result<Option<BufRef>, String> {
        let key = (uid, node_id, backend);
        let Some(staged) = self.cross.get(&key).copied() else {
            return Ok(None);
        };
        if self.cross_pending.contains(&key) {
            return Err(format!(
                "staged cross-backend input {node_id} for {backend:?} was read before its boundary \
                 wait: every copy_across owes one await_cross (F5, #58)"
            ));
        }
        Ok(Some(staged))
    }

    /// The staging buffer a consumer on `backend` must read for `node_id`, if a
    /// split boundary copied it for the current graph. Consumers on the node's
    /// own backend read the canonical buffer instead.
    ///
    /// **Raw accessor**: it does not check the phase-B contract (F5) — use
    /// [`Self::cross_input`] on the consumer path.
    pub fn cross_buffer(&self, uid: u64, node_id: NodeId, backend: Backend) -> Option<BufRef> {
        self.cross.get(&(uid, node_id, backend)).copied()
    }

    /// Test hook: stage `node_id`'s output on `backend` as if a split boundary
    /// had copied it. The real path needs a second usable backend, which a
    /// CPU-only build does not have.
    #[cfg(test)]
    pub fn stage_cross_for_test(
        &mut self,
        uid: u64,
        node_id: NodeId,
        backend: Backend,
        id: usize,
        len: usize,
    ) {
        self.cross
            .insert((uid, node_id, backend), BufRef::own(backend, id, len));
    }

    /// Test hook (F5): mark a staged entry as having an **un-waited** copy, i.e.
    /// behave exactly as the moment between `copy_across` and `await_cross`. Lets a
    /// CPU-only build gate the missing-wait invariant, which otherwise needs two
    /// usable backends.
    #[cfg(test)]
    pub fn mark_cross_pending_for_test(&mut self, uid: u64, node_id: NodeId, backend: Backend) {
        self.cross_pending.insert((uid, node_id, backend));
    }

    /// Dispatch `Backend::copy_cells` to the pool that owns the region (C3).
    ///
    /// F4: the "arm" is the registry entry's pool hook, and the refusal when
    /// there is none distinguishes "not compiled into this build" from "compiled
    /// in but its pool is not enabled here" — a backend that is compiled out is
    /// never silently treated as absent.
    #[allow(clippy::too_many_arguments)]
    fn copy_cells_in_pool(
        &mut self,
        backend: Backend,
        dst: BufRef,
        src: BufRef,
        dst_row: usize,
        src_row: usize,
        rows: usize,
        elems_per_cell: usize,
    ) -> Result<(), String> {
        let Some(pool) = self.pool_mut(backend) else {
            return Err(format!(
                "copy_cells: {} is {}",
                backend.name(),
                match backend.entry() {
                    None => "not compiled into this build".to_string(),
                    Some(_) => format!(
                        "not enabled on this allocator ({})",
                        super::registry::unavailable_reason(backend)
                            .unwrap_or("the pool is not enabled")
                    ),
                }
            ));
        };
        pool.copy_cells(dst, src, dst_row, src_row, rows, elems_per_cell)
    }

    /// Host read of one pool region by reference (the read side of
    /// [`Self::write_pool`]): C3's re-rope needs the K rows on the host.
    fn read_pool(&mut self, r: BufRef) -> Option<Vec<f32>> {
        let window = |s: &[f32]| -> Vec<f32> {
            let end = (r.offset + r.len).min(s.len());
            s[r.offset.min(end)..end].to_vec()
        };
        (r.backend.entry()?.host_read)(self, r.id).map(|v| window(&v))
    }

    /// Write host data into a pool buffer of `backend` (shared by the staging
    /// paths of `copy_across`).
    fn write_pool(&mut self, backend: Backend, id: usize, data: &[f32]) -> Result<(), String> {
        match self.pool_mut(backend) {
            Some(pool) => pool.write_host(id, data),
            None => Err(format!(
                "{} is not usable on this allocator: {}",
                backend.name(),
                super::registry::unavailable_reason(backend)
                    .unwrap_or("the backend's pool is not enabled")
            )),
        }
    }
}

impl KvProvider for GraphAllocator {
    fn kv_pair(&self, layer: usize) -> Option<(usize, usize)> {
        self.kv.get(layer).map(|r| (r.k.id, r.v.id))
    }
}

/// What a C3 compaction did, with the arena's counters before and after.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvDefragReport {
    /// The applied relocations, in application order — a caller holding a
    /// run's `start` must follow these.
    pub moves: Vec<super::kvcache::KvMove>,
    /// Written rows copied per layer (0 when only reservations moved).
    pub rows_moved: usize,
    pub before: super::kvcache::KvArenaStats,
    pub after: super::kvcache::KvArenaStats,
}

/// `MINFER_NO_KV_DEFRAG` (presence-checked, like the other `NO_*` gates):
/// disables the C3 compaction retry, so the same workload can be A/B'd with and
/// without it (standing rule 3).
pub fn kv_defrag_enabled() -> bool {
    kv_defrag_enabled_from(std::env::var_os("MINFER_NO_KV_DEFRAG").as_deref())
}

/// The pure half of [`kv_defrag_enabled`], so the gate's meaning is unit-tested
/// instead of only existing at runtime.
fn kv_defrag_enabled_from(no_flag: Option<&std::ffi::OsStr>) -> bool {
    no_flag.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::DType;

    /// F4: a fresh allocator inherits the run's backend fence, so the two places
    /// the fence is enforced (this allocator's assignment and the graph
    /// builders' `Device`) read the same answer. The default — nothing installed
    /// — is every backend, i.e. the pre-F4 behaviour.
    #[test]
    fn a_fresh_allocator_inherits_the_runs_backend_filter() {
        let alloc = GraphAllocator::new();
        assert_eq!(
            alloc.backend_filter(),
            crate::graph::registry::active_filter()
        );
        // The process-wide default is unfiltered in the test binary (nothing
        // installs a fence here), so the assignment is unchanged for every
        // existing gate.
        assert!(crate::graph::registry::active_filter().is_unfiltered());
        assert_eq!(alloc.supports(&Op::Input, DType::F32), Some(Backend::CPU));
    }

    /// F4: the fence reaches the **assignment pass**, not just the name parser.
    ///
    /// Needs a usable device, so it is `#[ignore]`d like the other device gates
    /// (run it serially: the CUDA device state is a process-wide singleton).
    /// Without a fence the device takes an op it supports; with the device
    /// fenced off the *same* op must land on the CPU — and the device stays
    /// enabled, because the fence is a policy, not a teardown.
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn cuda_the_backend_fence_moves_assignment_off_a_usable_device() {
        crate::cuda::CudaState::init();
        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("no CUDA device; this gate needs one");
            return;
        }
        assert_eq!(
            alloc.supports(&Op::Silu, DType::F32),
            Some(Backend::CUDA),
            "an unfenced allocator must offer the device"
        );
        alloc.set_backend_filter(
            crate::graph::registry::BackendFilter::from_names(["cpu"]).unwrap(),
        );
        assert_eq!(
            alloc.supports(&Op::Silu, DType::F32),
            Some(Backend::CPU),
            "a fenced allocator must not offer the device"
        );
        assert!(
            alloc.cuda().is_some(),
            "the fence must not tear the device pool down"
        );
    }

    /// C8a S1: the prefix copy refuses what it cannot do instead of copying a row that
    /// does not exist or writing past the destination's reservation.
    ///
    /// These are the checks that run before any backend work. The data path, and the
    /// written/capacity bounds (which need real KV regions), are covered by the S2 gate:
    /// the same prompt served via a copy and via a re-prefill must produce byte-identical
    /// continuations.
    #[test]
    fn copying_a_prefix_refuses_what_it_cannot_copy() {
        let mut a = GraphAllocator::new();
        a.kv_set_capacity(16);
        // Neither sequence holds a run yet.
        let err = a.kv_copy_prefix(0, 1, 4).unwrap_err();
        assert!(err.contains("holds no run"), "got: {err}");
        a.kv_reserve_seq(0, 8).unwrap();
        // The destination has no run.
        let err = a.kv_copy_prefix(0, 1, 4).unwrap_err();
        assert!(err.contains("holds no run"), "got: {err}");
        a.kv_reserve_seq(1, 4).unwrap();
        // Zero rows is a no-op, whatever the two runs are.
        a.kv_copy_prefix(0, 0, 0).expect("zero rows");
        // A source with nothing written is refused before any copy is attempted.
        let err = a.kv_copy_prefix(0, 1, 2).unwrap_err();
        assert!(err.contains("has written 0 rows"), "got: {err}");
    }

    /// C6/C7: `cells` is an absolute arena row, so it is bounded by the arena — not
    /// by the per-sequence position rule it used to be checked with. The two bounds
    /// coincide today, which is exactly why the difference is written down: sharing a
    /// cell range across sequences is what stops them coinciding.
    #[test]
    fn the_cells_input_is_bounded_by_the_arena() {
        let mut b = GraphBuilder::new();
        let cells = b.input("cells", [1, 1, 1, 1], crate::graph::DType::I32);
        b.output(cells);
        let g = b.build();
        let mut alloc = GraphAllocator::new();
        alloc.kv_set_capacity(8);
        alloc.alloc_graph(&g).expect("alloc the graph");
        alloc
            .fill_input_i32(&g, "cells", &[7])
            .expect("the last cell of an 8-cell arena is valid");
        let err = alloc.fill_input_i32(&g, "cells", &[8]).unwrap_err();
        assert!(err.contains("cell 8 is past the 8-cell arena"), "{err}");
        // Without an arena there is nothing to bound against, and the resolver reports
        // that case itself when it runs.
        let mut fresh = GraphAllocator::new();
        fresh.alloc_graph(&g).expect("alloc the graph");
        fresh
            .fill_input_i32(&g, "cells", &[3])
            .expect("no arena, no bound to check");
    }

    /// D1: a graph whose output is a *view* of an input, so the allocator must
    /// alias rather than copy. `offset`/`shape` are parameters so the refusal
    /// cases can be built too.
    fn view_graph(offset: usize, shape: [usize; 4], parent_shape: [usize; 4]) -> ComputeGraph {
        let mut b = GraphBuilder::new();
        let x = b.input("x", parent_shape, crate::graph::DType::F32);
        let v = b.node(
            "view",
            Op::View { offset, shape },
            &[x],
            shape,
            crate::graph::DType::F32,
            NodeMeta::None,
        );
        b.output(v);
        b.build()
    }

    /// D1 acceptance: the view is **zero-copy** — it maps onto its parent's
    /// buffer, and the parent is not recycled while the view lives.
    #[test]
    fn a_view_aliases_its_parent_buffer_and_keeps_it_alive() {
        let g = view_graph(0, [4, 1, 1, 1], [4, 1, 1, 1]);
        assert!(
            g.node(1).view.is_some(),
            "the builder must mark View as an alias"
        );
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).expect("alloc");
        let parent = alloc.node_buffer(0).expect("parent buffer");
        let view = alloc.node_buffer(1).expect("view buffer");
        assert_eq!(
            (view.backend, view.id),
            (parent.backend, parent.id),
            "the view must reuse its parent's buffer id (zero-copy)"
        );
        // Liveness: the parent's deadline covers the view (the view is an
        // output here, so both run to the end of the graph).
        let parent_deadline = alloc.buf_alive.get(&(parent.backend, parent.id)).copied();
        assert!(
            parent_deadline.is_some(),
            "the aliased parent must be registered as alive"
        );
    }

    /// D1 increment 2: a partial window at a non-zero offset maps onto the
    /// parent's buffer with exactly that window (the op matrix's `View offset`
    /// case proves the bytes; this pins the bookkeeping).
    #[test]
    fn an_offset_view_names_a_window_of_its_parent() {
        let g = view_graph(2, [4, 1, 1, 1], [8, 1, 1, 1]);
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).expect("alloc");
        let parent = alloc.node_buffer(0).expect("parent");
        let view = alloc.node_buffer(1).expect("view");
        assert_eq!(
            (view.id, view.offset, view.len),
            (parent.id, 2, 4),
            "the view must be the parent's buffer, windowed to [2, 6)"
        );
    }

    /// D1: what cannot be expressed is refused **loudly** — never silently
    /// copied or mis-read (standing rule 2). Increment 2 accepts offset and
    /// partial windows, so what remains here is a window that does not fit.
    #[test]
    fn views_that_do_not_fit_their_parent_are_refused() {
        let g = view_graph(4, [4, 1, 1, 1], [6, 1, 1, 1]);
        let err = GraphAllocator::new()
            .alloc_graph(&g)
            .expect_err("a window past the parent must be refused");
        assert!(err.contains("does not fit"), "{err}");
    }

    /// D2: the FFN composition's `SwiGLU` writes into its **gate window**, so the
    /// node's buffer *is* the concat buffer at offset 0 — the same bytes the
    /// hand-written `Op::FusedFFN` leaves its result in, which is what makes the
    /// two paths comparable. No GPU needed: this is allocator bookkeeping.
    #[test]
    fn ffn_composition_swiglu_aliases_the_gate_window() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        let out = b.fused_ffn_composition(
            x,
            "blk.0.ffn_gu",
            crate::tensor::TensorType::F32,
            4, // in_dim
            3, // nf
        );
        b.output(out);
        let g = b.build();
        let names: Vec<&str> = g.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"matmul_named"), "{names:?}");
        assert!(names.contains(&"ffn_gate_window"), "{names:?}");
        assert!(names.contains(&"ffn_up_window"), "{names:?}");
        assert!(names.contains(&"ffn_swiglu"), "{names:?}");

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).expect("alloc");
        let concat = alloc
            .node_buffer(
                g.nodes
                    .iter()
                    .position(|n| n.name == "matmul_named")
                    .unwrap(),
            )
            .expect("concat");
        let gate = alloc
            .node_buffer(
                g.nodes
                    .iter()
                    .position(|n| n.name == "ffn_gate_window")
                    .unwrap(),
            )
            .expect("gate");
        let up = alloc
            .node_buffer(
                g.nodes
                    .iter()
                    .position(|n| n.name == "ffn_up_window")
                    .unwrap(),
            )
            .expect("up");
        let swiglu = alloc
            .node_buffer(g.nodes.iter().position(|n| n.name == "ffn_swiglu").unwrap())
            .expect("swiglu");
        assert_eq!(
            (gate.id, gate.offset, gate.len),
            (concat.id, 0, 3),
            "the gate is the concat's first window"
        );
        assert_eq!(
            (up.id, up.offset, up.len),
            (concat.id, 3, 3),
            "the up half is the concat's second window"
        );
        assert_eq!(
            (swiglu.id, swiglu.offset, swiglu.len),
            (concat.id, 0, 3),
            "in-place swiglu writes into the gate window, exactly where the fused node puts its result"
        );
    }

    /// D1: an in-place op on a view writes through to the buffer it is a window
    /// of, so liveness must follow the chain (view -> parent).
    #[test]
    fn in_place_on_a_view_extends_the_parents_liveness() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        let v = b.node(
            "view",
            Op::View {
                offset: 0,
                shape: [4, 1, 1, 1],
            },
            &[x],
            [4, 1, 1, 1],
            crate::graph::DType::F32,
            NodeMeta::None,
        );
        // sole consumer of the view, same backend -> the in-place rule aliases
        let s = b.silu(v);
        b.output(s);
        let g = b.build();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).expect("alloc");
        let parent = alloc.node_buffer(0).expect("parent");
        assert_eq!(
            alloc.node_buffer(2).expect("silu").id,
            parent.id,
            "the in-place op must alias through the view onto the parent"
        );
        assert!(
            alloc.buf_alive.contains_key(&(parent.backend, parent.id)),
            "parent must still be alive for the in-place consumer"
        );
    }

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
        let _store = b.kvcache_store(0, k, v, 1024);
        let load = b.kvcache_load(0, 16, 1024, 2);
        b.output(load);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        // C6: the store now also consumes a `cells` input, so node ids shifted —
        // look the nodes up by name instead of by position.
        let store = g.nodes.iter().position(|n| n.name == "kv_store.0").unwrap();
        let load = g.nodes.iter().position(|n| n.name == "kv_load.0").unwrap();
        // store and load share the K region; V is a sibling
        assert_eq!(alloc.node_buffer(store), alloc.node_buffer(load));
        let pair = alloc.kv_pair(0).unwrap();
        assert_eq!(alloc.node_buffer(store).unwrap().id, pair.0);
        assert_ne!(pair.0, pair.1);
        assert_eq!(alloc.persistent.len(), 2);
        assert_eq!(alloc.persistent[0].name, "kv.0.k");
        assert_eq!(alloc.persistent[1].name, "kv.0.v");
        // mapped buffers: positions/k/v/cells (4 liveness) + K region (shared) = 5
        assert_eq!(alloc.n_mapped_buffers(), 5);
    }

    /// C5 end to end on CPU: a session's region bytes **and** its run table
    /// (reservation, ownership, written extent) survive a save and a load into a
    /// **fresh** allocator, and the loaded arena resolves the same cells.
    ///
    /// The refusals are the other half of the ticket: a file that does not describe
    /// this arena is rejected loudly, and because `kv_load` verifies the whole file
    /// before it applies anything, a rejected load leaves no arena behind.
    #[test]
    fn kv_session_round_trips_the_rows_and_the_run_table() {
        const N_CTX: usize = 16;
        const ROW: usize = 4;
        const SEQ: u32 = 7;
        let kk: Vec<f32> = (0..ROW * 2).map(|i| 0.25 + i as f32 * 0.5).collect();
        let vv: Vec<f32> = (0..ROW * 2).map(|i| 1.0 / (i as f32 + 1.0)).collect();

        let graph = |alloc: &mut GraphAllocator| -> ComputeGraph {
            let mut b = GraphBuilder::new();
            let pos = b.input("positions", [2, 1, 1, 1], crate::graph::DType::I32);
            let k = b.input("k", [ROW, 2, 1, 1], crate::graph::DType::F32);
            let v = b.input("v", [ROW, 2, 1, 1], crate::graph::DType::F32);
            let _store = b.kvcache_store(0, k, v, N_CTX);
            let load = b.kvcache_load(0, ROW, N_CTX, 1);
            b.output(load);
            let g = b.build();
            alloc.kv_set_capacity(N_CTX);
            alloc.alloc_graph(&g).unwrap();
            g
        };

        // ---- the session that stays in memory ----
        let mut a = GraphAllocator::new();
        let ga = graph(&mut a);
        a.kv_reserve_seq(SEQ, N_CTX).unwrap();
        a.fill_input_i32(&ga, "positions", &[1, 3]).unwrap();
        a.fill_attn_inputs(&ga, &[SEQ, SEQ], &[1, 3]).unwrap();
        a.fill_input(&ga, "k", &kk).unwrap();
        a.fill_input(&ga, "v", &vv).unwrap();
        let mut sched = crate::graph::scheduler::BackendScheduler::new();
        sched.execute(&ga, &mut a).unwrap();
        a.kv_own_range(SEQ, 0, 4);
        let want_kv = a.copy_kv_to_cpu(0).unwrap();
        let want_stats = a.kv_arena_stats();
        let want_cells = a.kv_cells_for_seq(&[SEQ, SEQ], &[1, 3]).unwrap();

        let path = std::env::temp_dir().join(format!(
            "minfer-c5-alloc-{}-roundtrip.bin",
            std::process::id()
        ));
        let report = a.kv_save(&path).unwrap();
        assert_eq!(report.layers, 1);
        assert_eq!(report.cells, N_CTX);
        assert_eq!(report.written, 4, "positions 0..4 are written");
        assert_eq!(report.bytes, std::fs::metadata(&path).unwrap().len());

        // ---- a fresh allocator, restored from the file ----
        let expect = KvSessionExpect {
            backend: Backend::CPU,
            n_ctx: N_CTX,
            n_embd: ROW,
        };
        let mut b = GraphAllocator::new();
        let gb = graph(&mut b);
        let loaded = b.kv_load(&path, &expect).unwrap();
        assert_eq!(loaded, report);
        assert_eq!(
            b.copy_kv_to_cpu(0).unwrap(),
            want_kv,
            "the region bytes must be identical"
        );
        assert_eq!(b.kv_arena_stats(), want_stats);
        assert_eq!(
            b.kv_cells_for_seq(&[SEQ, SEQ], &[1, 3]).unwrap(),
            want_cells,
            "the restored run table must resolve the same cells"
        );
        assert_eq!(b.kv_n_used(0), a.kv_n_used(0));
        let _ = gb;

        // ---- refusals ----
        for (what, expect_bad, needle) in [
            (
                "a different n_ctx",
                KvSessionExpect {
                    n_ctx: N_CTX + 1,
                    ..expect
                },
                "-cell arena",
            ),
            (
                "a different row width",
                KvSessionExpect {
                    n_embd: ROW + 4,
                    ..expect
                },
                "KV rows",
            ),
            (
                "another backend",
                KvSessionExpect {
                    backend: Backend::METAL,
                    ..expect
                },
                "session",
            ),
        ] {
            let mut fresh = GraphAllocator::new();
            let err = fresh.kv_load(&path, &expect_bad).unwrap_err();
            assert!(err.contains(needle), "{what}: {err}");
            assert!(
                fresh.kv_n_used(0).is_none(),
                "{what}: a refused load must not create an arena"
            );
        }
        // A different KV element type for the same shape.
        let mut fresh = GraphAllocator::new();
        fresh
            .cpu_mut()
            .set_kv_format(super::super::kvformat::KvFormat::Q8_0);
        let err = fresh.kv_load(&path, &expect).unwrap_err();
        assert!(err.contains("element type"), "{err}");
        assert!(fresh.kv_n_used(0).is_none());

        // A truncated file: rejected, and (because `kv_load` verifies first) the
        // allocator is untouched.
        let full = std::fs::read(&path).unwrap();
        std::fs::write(&path, &full[..full.len() / 2]).unwrap();
        let mut fresh = GraphAllocator::new();
        let err = fresh.kv_load(&path, &expect).unwrap_err();
        assert!(err.contains("truncated"), "{err}");
        assert!(fresh.kv_n_used(0).is_none());
        std::fs::remove_file(&path).ok();
    }

    /// #130: the same `kv_save` → `kv_load` round trip under an **f16** element
    /// type — the policy a CUDA box's auto rule picks for the 7B class
    /// (`n_layers * n_kv_embd >= 8192`). The store here is the CPU's (which copies
    /// words), so this gates the *container* half with no device: the format the
    /// allocator reports, the header's flags, the reader's decode and the
    /// `header.format != live` check. Before `FLAG_F16` the load refused the file
    /// the save had just produced. The device half is the Qwen3-0.6B real-model gate.
    #[test]
    fn an_f16_session_round_trips_through_save_and_load() {
        use crate::graph::kvformat::KvFormat;
        const N_CTX: usize = 8;
        const ROW: usize = 4;
        const SEQ: u32 = 3;
        let kk: Vec<f32> = (0..ROW * 2).map(|i| 0.5 + i as f32).collect();
        let vv: Vec<f32> = (0..ROW * 2).map(|i| 1.0 / (i as f32 + 1.0)).collect();

        let build = |alloc: &mut GraphAllocator| -> ComputeGraph {
            alloc.cpu_mut().set_kv_format(KvFormat::F16);
            let mut b = GraphBuilder::new();
            b.set_kv_format(KvFormat::F16);
            let pos = b.input("positions", [2, 1, 1, 1], crate::graph::DType::I32);
            let k = b.input("k", [ROW, 2, 1, 1], crate::graph::DType::F32);
            let v = b.input("v", [ROW, 2, 1, 1], crate::graph::DType::F32);
            let _store = b.kvcache_store(0, k, v, N_CTX);
            let load = b.kvcache_load(0, ROW, N_CTX, 1);
            b.output(load);
            let g = b.build();
            alloc.kv_set_capacity(N_CTX);
            alloc.alloc_graph(&g).unwrap();
            g
        };

        let mut a = GraphAllocator::new();
        let ga = build(&mut a);
        a.kv_reserve_seq(SEQ, N_CTX).unwrap();
        a.fill_input_i32(&ga, "positions", &[0, 1]).unwrap();
        a.fill_attn_inputs(&ga, &[SEQ, SEQ], &[0, 1]).unwrap();
        a.fill_input(&ga, "k", &kk).unwrap();
        a.fill_input(&ga, "v", &vv).unwrap();
        let mut sched = crate::graph::scheduler::BackendScheduler::new();
        sched.execute(&ga, &mut a).unwrap();
        a.kv_own_range(SEQ, 0, 2);
        let want_kv = a.copy_kv_to_cpu(0).unwrap();

        let path =
            std::env::temp_dir().join(format!("minfer-c5-alloc-{}-f16.bin", std::process::id()));
        let report = a.kv_save(&path).unwrap();
        // The bytes on disk name f16 — the flag the reader must decode.
        let bytes = std::fs::read(&path).unwrap();
        let flags = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        assert_eq!(flags, 1 << 1, "the header must carry FLAG_F16");

        // A fresh allocator under the same policy resumes it, rows bit-identical.
        let expect = KvSessionExpect {
            backend: Backend::CPU,
            n_ctx: N_CTX,
            n_embd: ROW,
        };
        let mut b = GraphAllocator::new();
        let gb = build(&mut b);
        let loaded = b.kv_load(&path, &expect).unwrap();
        assert_eq!(loaded, report);
        assert_eq!(b.copy_kv_to_cpu(0).unwrap(), want_kv);
        let _ = gb;

        // An f32 reader refuses it by element type — not silently, and without
        // creating an arena.
        let mut c = GraphAllocator::new();
        c.cpu_mut().set_kv_format(KvFormat::F32);
        let err = c.kv_load(&path, &expect).unwrap_err();
        assert!(err.contains("element type"), "{err}");
        assert!(err.contains("f16"), "{err}");
        assert!(c.kv_n_used(0).is_none());
        std::fs::remove_file(&path).ok();
    }

    /// C3 end to end on CPU: a fragmented arena refuses an 8-cell reservation,
    /// the compaction moves the real KV bytes *and* renumbers the runs, and the
    /// reservation then fits — with the moved rows byte-identical at their new
    /// cells.
    #[test]
    fn kv_defrag_moves_the_bytes_and_opens_the_run() {
        const N_CTX: usize = 16;
        const ROW: usize = 4; // elements per cell (n_kv_embd)
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, N_CTX);
        let load = b.kvcache_load(0, ROW, N_CTX, 1);
        b.output(load);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.kv_set_capacity(N_CTX);
        alloc.alloc_graph(&g).unwrap();
        for seq in 1u32..=3 {
            assert_eq!(
                alloc.kv_reserve_seq(seq, 4).unwrap().start,
                (seq as usize - 1) * 4
            );
            alloc.kv_own_range(seq, (seq as usize - 1) * 4, seq as usize * 4);
        }
        let rope = crate::graph::kvcache::KvRope {
            freq_base: 10_000.0,
            freq_scale: 1.0,
            n_head_kv: 1,
            hd: ROW,
            style: crate::vec_ops::RopeStyle::NonInterleaved,
        };
        let (kid, vid) = alloc.kv_pair(0).unwrap();
        let pattern = |base: f32| -> Vec<f32> {
            (0..N_CTX * ROW)
                .map(|i| base + (i / ROW) as f32 + (i % ROW) as f32 / 10.0)
                .collect()
        };
        alloc
            .write_pool(crate::graph::Backend::CPU, kid, &pattern(0.0))
            .unwrap();
        alloc
            .write_pool(crate::graph::Backend::CPU, vid, &pattern(100.0))
            .unwrap();
        // Release the middle run: free runs [4,8) and [12,16), 8 cells, none 8 long.
        alloc.kv_release_seq(2);
        let before = alloc.kv_arena_stats();
        assert_eq!((before.free_cells, before.free_runs), (8, 2));
        assert!(alloc.kv_reserve_seq(4, 8).is_err());

        let report = alloc.kv_defrag(Some(8)).unwrap();
        assert_eq!(report.moves.len(), 1, "{report:?}");
        assert_eq!(report.moves[0].seq, 3);
        assert_eq!((report.moves[0].from, report.moves[0].to), (8, 4));
        assert_eq!(report.rows_moved, 4);
        assert_eq!((report.before.free_runs, report.after.free_runs), (2, 1));
        assert_eq!(report.after.largest_free_run, 8);
        assert_eq!((report.after.defrags, report.after.cells_moved), (1, 4));
        // C6: both K and V move **verbatim** — a cell move no longer changes any
        // rotation, because a token's angle is its index within its sequence.
        let (k_now, v_now) = alloc.copy_kv_to_cpu(0).unwrap();
        for e in 0..ROW {
            assert_eq!(v_now[4 * ROW + e], 108.0 + e as f32 / 10.0, "V element {e}");
        }
        let k_expect: Vec<f32> = (0..ROW).map(|e| 8.0 + e as f32 / 10.0).collect();
        for e in 0..ROW {
            assert_eq!(k_now[4 * ROW + e], k_expect[e], "K element {e}");
        }
        // The reservation that first-fit refused now fits, in the opened tail.
        assert_eq!(alloc.kv_reserve_seq(4, 8).unwrap().start, 8);
        // The same helper, on a fresh fragmentation: free the lowest run and the
        // 8-cell one, and ask for 12 contiguous cells. First-fit fails (the free
        // space is two runs); compacting the single survivor down opens the tail,
        // and the helper reports the move it relied on.
        alloc.kv_release_seq(1);
        alloc.kv_release_seq(4);
        let (slot, moves) = alloc.kv_reserve_seq_with_defrag(5, 12).unwrap();
        assert_eq!(
            slot.start, 4,
            "the retry packs the survivor down, then takes the tail"
        );
        assert_eq!(moves.len(), 1, "{moves:?}");
        assert_eq!((moves[0].seq, moves[0].from, moves[0].to), (3, 4, 0));
        // The helper also answers "it still does not fit" with the original
        // first-fit error, after compaction moved nothing.
        let err = alloc.kv_reserve_seq_with_defrag(6, 4).unwrap_err();
        assert!(err.contains("no free run of 4 cells"), "{err}");
    }

    #[test]
    fn the_defrag_gate_is_off_only_when_the_flag_is_present() {
        assert!(super::kv_defrag_enabled_from(None));
        assert!(!super::kv_defrag_enabled_from(Some(std::ffi::OsStr::new(
            "1"
        ))));
    }

    /// C8b S3 end to end on CPU: a sequence that reads a prefix in place cannot
    /// store into it, the copy-on-write moves **its own** rows up inside its run,
    /// and the donor's cells come out of it byte-identical.
    #[test]
    fn a_copy_on_write_moves_the_rows_and_never_writes_through() {
        const N_CTX: usize = 16;
        const ROW: usize = 4; // elements per cell (n_kv_embd)
        let mut b = GraphBuilder::new();
        let _pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [ROW, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [ROW, 1, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, N_CTX);
        let load = b.kvcache_load(0, ROW, N_CTX, 1);
        b.output(load);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.kv_set_capacity(N_CTX);
        alloc.alloc_graph(&g).unwrap();
        let (kid, vid) = alloc.kv_pair(0).unwrap();
        // Every cell holds the row index and the element, so a moved row is
        // identifiable wherever it lands.
        let pattern = |base: f32| -> Vec<f32> {
            (0..N_CTX * ROW)
                .map(|i| base + (i / ROW) as f32 * 100.0 + (i % ROW) as f32 / 10.0)
                .collect()
        };
        alloc
            .write_pool(crate::graph::Backend::CPU, kid, &pattern(0.0))
            .unwrap();
        alloc
            .write_pool(crate::graph::Backend::CPU, vid, &pattern(1000.0))
            .unwrap();
        // Sequence 1 computes four rows at [0, 4); sequence 2 reserves [4, 10),
        // reads those four in place, and writes two rows of its own at [4, 6).
        assert_eq!(alloc.kv_reserve_seq(1, 4).unwrap().start, 0);
        alloc.kv_own_range(1, 0, 4);
        assert_eq!(alloc.kv_reserve_seq(2, 6).unwrap().start, 4);
        assert_eq!(alloc.kv_share_prefix(1, 2, 4).unwrap(), 4);
        alloc.kv_own_range(2, 4, 6);

        // The store resolver refuses a position inside the share: that refusal is
        // what makes a write-through impossible rather than merely unlikely.
        let err = alloc.kv_cells_for_seq(&[2], &[1]).unwrap_err();
        assert!(err.contains("shared prefix"), "got: {err}");
        assert!(err.contains("C8b S3"), "got: {err}");

        let shift = alloc
            .kv_private_row_for(2, 1)
            .unwrap()
            .expect("position 1 must copy-on-write");
        assert_eq!((shift.base, shift.from, shift.to, shift.rows), (1, 4, 7, 2));
        assert_eq!(alloc.kv_arena_stats().cows, 1);
        // From position 1 on everything is private, and the rows that were at
        // [4, 6) kept their positions at [7, 9) — while position 0 still reads the
        // donor's cell 0.
        assert_eq!(alloc.kv.cell_of(2, 0), Some(0), "still shared");
        assert_eq!(
            alloc
                .kv_cells_for_seq(&[2, 2, 2, 2, 2], &[1, 2, 3, 4, 5])
                .unwrap(),
            vec![4, 5, 6, 7, 8]
        );
        // A store at 0 would still land in the donor's cells, so it is still
        // refused — the share is smaller, not gone.
        assert!(alloc.kv_cells_for_seq(&[2], &[0]).is_err());
        // The data moved verbatim (K and V), and the donor's four rows are what
        // they were — nothing wrote through them.
        let (k_now, v_now) = alloc.copy_kv_to_cpu(0).unwrap();
        for e in 0..ROW {
            let row = |cell: usize| cell * ROW + e;
            assert_eq!(k_now[row(7)], 400.0 + e as f32 / 10.0, "K at [7]");
            assert_eq!(k_now[row(8)], 500.0 + e as f32 / 10.0, "K at [8]");
            assert_eq!(v_now[row(7)], 1400.0 + e as f32 / 10.0, "V at [7]");
            assert_eq!(v_now[row(8)], 1500.0 + e as f32 / 10.0, "V at [8]");
            for cell in 0..4 {
                assert_eq!(
                    k_now[row(cell)],
                    cell as f32 * 100.0 + e as f32 / 10.0,
                    "donor K cell {cell} changed"
                );
                assert_eq!(
                    v_now[row(cell)],
                    1000.0 + cell as f32 * 100.0 + e as f32 / 10.0,
                    "donor V cell {cell} changed"
                );
            }
        }
        // Idempotent for a position that is already private.
        assert_eq!(alloc.kv_private_row_for(2, 3).unwrap(), None);
        assert_eq!(alloc.kv_arena_stats().cows, 1);
        // And a sequence with no run at all is a no-op, not an error — that is what
        // keeps the classic single-sequence path (which never reserves) untouched.
        assert_eq!(alloc.kv_private_row_for(9, 0).unwrap(), None);
        // The other entry point a caller drives the allocator with gets the same
        // rule: `fill_attn_inputs` copies for a batch that still names a shared
        // position, and then resolves the private cells. Position 0 is the last
        // shared one, so this is the second (and final) copy-on-write.
        alloc.fill_attn_inputs(&g, &[2], &[0]).unwrap();
        assert_eq!(alloc.kv_arena_stats().cows, 2);
        assert_eq!(alloc.kv.spans_of(2), &[(0, 4, 6)], "the share is gone");
        assert_eq!(alloc.kv.cell_of(2, 0), Some(4));
        assert_eq!(alloc.kv.cell_of(2, 5), Some(9));
        let (k_end, _) = alloc.copy_kv_to_cpu(0).unwrap();
        for e in 0..ROW {
            assert_eq!(
                k_end[9 * ROW + e],
                500.0 + e as f32 / 10.0,
                "the rows shifted up once more"
            );
        }
    }

    /// D1 increment 3, structural half: `split_parts` maps each part onto the
    /// owner's buffer as a window, so a producer's one output feeds several
    /// consumers without a copy and without a second output on `CNode`.
    #[test]
    fn split_parts_maps_every_part_onto_the_owners_buffer() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
        // A real producer whose single kernel output carries both parts (the D2
        // shape: one concat matmul, two halves).
        let concat = b.matmul_by_name(x, "blk.0.w", crate::tensor::TensorType::F32, 6, 4);
        let parts = b.split_parts(concat, &[3, 3]);
        let (gate, up) = (parts[0], parts[1]);
        b.output(up);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).expect("alloc");
        let owner = alloc.node_buffer(concat).expect("owner");
        assert_eq!(
            (
                alloc.node_buffer(gate).unwrap().offset,
                alloc.node_buffer(gate).unwrap().len
            ),
            (0, 3),
            "the first part is the owner's first window"
        );
        assert_eq!(
            (
                alloc.node_buffer(up).unwrap().offset,
                alloc.node_buffer(up).unwrap().len
            ),
            (3, 3),
            "the second part starts where the first ends"
        );
        assert_eq!(
            alloc.node_buffer(gate).unwrap().id,
            owner.id,
            "a part aliases the owner's buffer"
        );
        assert_eq!(alloc.node_buffer(up).unwrap().id, owner.id);
        assert_eq!(
            g.nodes[gate].view.as_ref().map(|v| (v.src, v.offset)),
            Some((concat, 0)),
            "and it is recorded as a view, so liveness follows the owner"
        );
    }

    /// D1 increment 3, functional half: the parts are independently consumable —
    /// here the second part is an **indices** part (i32 bit patterns in an f32
    /// buffer, rule 4) driving `Op::GetRows` over the first part. That is the
    /// shape MoE routing needs (a router's top-k indices selecting expert rows)
    /// and it exercises the mechanism end to end: one owner, two parts, two
    /// different consumers, no second output and no backend change.
    #[test]
    fn split_parts_parts_feed_independent_consumers() {
        let mut b = GraphBuilder::new();
        // 4 value rows of width 2, then 2 indices: one flat buffer.
        let both = b.input("both", [10, 1, 1, 1], crate::graph::DType::F32);
        let parts = b.split_parts(both, &[8, 2]);
        let (values, indices) = (parts[0], parts[1]);
        let gathered = b.get_rows(values, indices, [2, 2, 1, 1]);
        b.output(gathered);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).expect("alloc");
        let mut data: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        data.push(f32::from_bits(3)); // gather row 3
        data.push(f32::from_bits(0)); // then row 0
        alloc.fill_input(&g, "both", &data).expect("fill");
        let mut sched = crate::graph::scheduler::BackendScheduler::new();
        sched.execute(&g, &mut alloc).expect("execute");
        assert_eq!(
            alloc.copy_to_cpu(gathered).expect("read"),
            vec![7.0, 8.0, 1.0, 2.0],
            "rows 3 and 0 of the values part, in the indices part's order"
        );
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
            let _store = b.kvcache_store(0, k, v, n_ctx);
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

    /// The KV row input is `cells` (C6), and a cell is an index into the arena, so a
    /// value at or past the arena is an out-of-bounds write on every backend that does
    /// not re-check it (the GPU ones). It is bounded by the *arena* (C7/#60), not by the
    /// per-sequence position rule — the two only coincide while a cell equals its
    /// position. The guard lives in the allocator so all three backends share it.
    #[test]
    fn a_cell_beyond_the_arena_is_rejected() {
        let mut b = GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], crate::graph::DType::I32);
        let k = b.input("k", [16, 1, 1, 1], crate::graph::DType::F32);
        let v = b.input("v", [16, 1, 1, 1], crate::graph::DType::F32);
        let _store = b.kvcache_store(0, k, v, 1024);
        let load = b.kvcache_load(0, 16, 1024, 2);
        b.output(load);
        let g = b.build();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();

        // The last legal row is n_ctx - 1.
        alloc.fill_input_i32(&g, "cells", &[1023]).unwrap();
        let err = alloc.fill_input_i32(&g, "cells", &[1024]).unwrap_err();
        assert!(
            err.contains("cell 1024 is past the 1024-cell arena"),
            "got: {err}"
        );
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
        let _store = b.kvcache_store(0, emb, v, 8);
        let load = b.kvcache_load(0, 16, 8, 2);
        b.output(load);
        let g = b.build();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();

        alloc
            .fill_input_i32(&g, "token_ids", &[50_000])
            .expect("token ids are not positions");
        let err = alloc.fill_input_i32(&g, "cells", &[8]).unwrap_err();
        assert!(
            err.contains("cell 8 is past the 8-cell arena"),
            "got: {err}"
        );
    }

    /// A staging buffer is keyed by (node, destination backend): one node
    /// feeding two foreign backends gets one buffer each, and a consumer is
    /// never offered the other backend's copy. The old single-entry map forced
    /// the scheduler to filter by backend on every read (and could not serve
    /// two foreign consumers at all).
    #[test]
    fn staging_is_keyed_by_destination_backend() {
        let mut alloc = GraphAllocator::new();
        alloc.stage_cross_for_test(1, 7, Backend::CPU, 3, 4);
        assert_eq!(
            alloc.cross_buffer(1, 7, Backend::CPU).map(|b| b.id),
            Some(3)
        );
        assert!(
            alloc.cross_buffer(1, 7, Backend::CUDA).is_none(),
            "a CPU staging buffer must not be offered to a CUDA consumer"
        );
        alloc.stage_cross_for_test(1, 7, Backend::CUDA, 4, 4);
        assert_eq!(
            alloc.cross_buffer(1, 7, Backend::CUDA).map(|b| b.id),
            Some(4)
        );
        assert_eq!(
            alloc.cross_buffer(1, 7, Backend::CPU).map(|b| b.id),
            Some(3),
            "staging for a second backend must not clobber the first"
        );
    }

    /// F5 ([#58]) gate, allocator half: **a staged entry whose boundary wait has
    /// not been issued cannot be handed to a consumer, and issuing the wait
    /// publishes it.**
    ///
    /// This is the mechanism the missing-wait gate rests on. It is deterministic
    /// and needs no device: the pending flag is the state between phase A
    /// (`copy_across`) and phase B (`await_cross`), and `cross_input` — the only
    /// accessor the scheduler's consumer path uses — refuses it by name. The
    /// graph-level version (the same refusal through the real `execute`) is
    /// `scheduler::tests::a_staged_boundary_input_cannot_be_consumed_before_its_wait`.
    #[test]
    fn a_pending_staged_copy_is_refused_until_its_wait_is_issued() {
        let g = {
            use crate::graph::builder::GraphBuilder;
            let mut b = GraphBuilder::new();
            let x = b.input("x", [8, 1, 1, 1], super::super::DType::F32);
            let s = b.silu(x);
            b.output(s);
            b.build()
        };
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        // Node 1 (the silu) is on the CPU; the staging entry is for a notional
        // device consumer, so `await_cross` dispatches to the CPU entry's
        // registered no-op (the CPU has no device transfer to wait on) and the
        // test stays device-free.
        const N: NodeId = 1;
        alloc.stage_cross_for_test(g.uid, N, Backend::CUDA, 4, 8);
        alloc.mark_cross_pending_for_test(g.uid, N, Backend::CUDA);

        // Phase B has not run: the entry is not readable.
        let err = alloc
            .cross_input(g.uid, N, Backend::CUDA)
            .expect_err("a pending staging entry must not be published");
        assert!(err.contains("before its boundary wait"), "{err}");
        assert!(err.contains("#58"), "{err}");
        // A key with no entry is still "nothing was staged", not an error.
        assert_eq!(alloc.cross_input(g.uid, N, Backend::CPU).unwrap(), None);

        // A different (uid, node, backend) triple is untouched by the pending flag.
        alloc.stage_cross_for_test(2, N, Backend::CUDA, 5, 8);
        assert!(alloc.cross_input(2, N, Backend::CUDA).unwrap().is_some());

        // Phase B publishes it and the counter records the wait.
        alloc.await_cross(g.uid, N, Backend::CUDA).unwrap();
        let staged = alloc
            .cross_input(g.uid, N, Backend::CUDA)
            .expect("the wait published the entry")
            .expect("the staging buffer exists");
        assert_eq!((staged.id, staged.len), (4, 8));
        assert_eq!(
            alloc.cross_stats(),
            CrossCopyStats {
                copies: 0,
                waits: 1,
                ..CrossCopyStats::default()
            },
            "the wait is counted; no copy was issued here (the test injected the state)"
        );
    }

    /// F5: `await_cross` is idempotent and self-cleaning — including when the pair
    /// never crossed a backend (the scheduler does list a same-backend node whose
    /// split differs, e.g. CPU → Metal → CPU). Such an entry is not a *copy*, so it
    /// must not be counted as one, or `copies == waits` would stop meaning what the
    /// gate reads it as.
    #[test]
    fn a_same_backend_boundary_input_is_neither_copied_nor_counted() {
        let mut alloc = GraphAllocator::new();
        alloc.mark_cross_pending_for_test(1, 7, Backend::CPU);
        // No buffer for node 7 exists, so this is the "unknown node" refusal.
        assert!(alloc.await_cross(1, 7, Backend::CPU).is_err());

        // With a real node and a same-backend destination, both phases are no-ops.
        let g = {
            use crate::graph::builder::GraphBuilder;
            let mut b = GraphBuilder::new();
            let x = b.input("x", [4, 1, 1, 1], super::super::DType::F32);
            let s = b.silu(x);
            b.output(s);
            b.build()
        };
        alloc.alloc_graph(&g).unwrap();
        alloc.copy_across(g.uid, 1, Backend::CPU).unwrap();
        alloc.await_cross(g.uid, 1, Backend::CPU).unwrap();
        assert_eq!(alloc.cross_stats(), CrossCopyStats::default());
        assert!(
            alloc.cross_input(g.uid, 1, Backend::CPU).unwrap().is_none(),
            "a same-backend input reads its canonical buffer, not staging"
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
            view: None,
            layer: None,
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
            view: None,
            layer: None,
        });
        let mut alloc = GraphAllocator::new();
        assert!(alloc.alloc_graph(&g).is_err());
    }

    /// Issue #122's fail-open twin: a **poisoned** weights lock must not report 0 bytes.
    ///
    /// The mutation this pins is `weights.lock().map(|w| …).unwrap_or(0)`: a panic in
    /// another thread poisons the mutex, the sum became 0, and `weights + activations >
    /// budget` then under-charged the budget by every resident weight. The registry is
    /// append-only, so the value behind the poison is still valid and is recovered.
    #[test]
    fn a_poisoned_registry_does_not_report_zero_weights() {
        let registry: std::sync::Mutex<Vec<(u64, usize)>> =
            std::sync::Mutex::new(vec![(1, 7), (2, 11)]);
        // Poison it the way a panic inside a holder would.
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _g = registry.lock().unwrap();
            panic!("poison the registry");
        }));
        assert!(registry.is_poisoned(), "the test must actually poison it");
        let bytes = weights_from_lock(registry.lock(), "test registry", |r| {
            r.iter().map(|(_, size)| *size).sum()
        });
        assert_eq!(
            bytes, 18,
            "the poisoned registry must report its real bytes, not 0"
        );
    }

    /// E4's safety half: a request that would exceed the backend's budget is refused
    /// **before the pool is touched**, with the numbers (weights, pooled, this request,
    /// budget), instead of surfacing later as a null pointer at execute time.
    #[test]
    fn a_graph_that_cannot_fit_is_refused_with_its_numbers() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [1024, 1, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        // One byte short of a 4 KiB activation: the gate must fire (the class of 1024
        // elements is 1024 exactly, so this is the requirement minus one).
        alloc.set_memory_budget(Backend::CPU, Some(4095));
        let err = alloc.alloc_graph(&g).unwrap_err();
        assert!(err.contains("out of CPU memory"), "{err}");
        assert!(err.contains("budget"), "{err}");
        assert!(
            err.contains("MiB"),
            "the error must carry the numbers, not a bare failure: {err}"
        );
        // Nothing was allocated: the gate runs before the pool is asked for anything.
        assert_eq!(
            alloc.n_cpu_buffers(),
            0,
            "the refused graph must not touch the pool"
        );
        let report = alloc.memory_report(Backend::CPU);
        assert_eq!(report.pool_bytes, 0);
        assert_eq!(report.budget, Some(4095));
        assert_eq!(report.headroom_bytes(), Some(4095));

        // The same graph fits once the budget does.
        alloc.set_memory_budget(Backend::CPU, Some(1 << 20));
        alloc.alloc_graph(&g).unwrap();
        let report = alloc.memory_report(Backend::CPU);
        assert!(report.pool_bytes > 0, "{report:?}");
        assert!(report.live_bytes > 0 && report.live_bytes <= report.pool_bytes);
        assert!(report.peak_live_bytes >= report.live_bytes);
        assert_eq!(report.budget, Some(1 << 20));
        assert!(report.headroom_bytes().unwrap() < 1 << 20);
    }

    /// E4's accounting half: weights and activations are one comparison, and the peak is
    /// a number the caller can read (the ticket's "peak memory is reported").
    #[test]
    fn the_budget_counts_weights_and_activations_together() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [256, 1, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();

        let weight = tensor_f32("w", [256, 1, 1, 1], vec![0.5; 256]);
        let weight_bytes = weight.data.len();

        let mut alloc = GraphAllocator::new();
        alloc.register_weight("w", weight);
        let report = alloc.memory_report(Backend::CPU);
        assert_eq!(report.weights_bytes, weight_bytes, "{report:?}");
        assert_eq!(report.pool_bytes, 0, "nothing is allocated before a build");

        // A budget that covers the weight but not the activation is still a refusal.
        let activation_class =
            crate::graph::allocplan::class_bytes(crate::graph::allocplan::class_size(256 * 4 / 4));
        alloc.set_memory_budget(Backend::CPU, Some(weight_bytes + activation_class - 1));
        let err = alloc.alloc_graph(&g).unwrap_err();
        assert!(err.contains("out of CPU memory"), "{err}");
        assert!(
            err.contains(&format!("{} MiB", weight_bytes / (1024 * 1024))),
            "the refusal must name the weight bytes: {err}"
        );

        // One byte more and it fits.
        alloc.set_memory_budget(Backend::CPU, Some(weight_bytes + activation_class));
        alloc.alloc_graph(&g).unwrap();
        let report = alloc.memory_report(Backend::CPU);
        assert_eq!(report.weights_bytes, weight_bytes);
        assert!(
            report.weights_bytes + report.pool_bytes <= report.budget.unwrap(),
            "{report:?}"
        );
    }

    /// E4 S2's second acceptance: pooled activations are rounded up to their size class,
    /// so a graph whose shapes move **inside** one class recycles the pool's buffers
    /// instead of reserving one per shape ("no silent growth"). Before this, the free
    /// list matched lengths exactly and every shape of a rebuild added a buffer.
    #[test]
    fn a_rebuild_inside_one_class_reuses_the_pool() {
        let mut alloc = GraphAllocator::new();
        let build = |alloc: &mut GraphAllocator, rows: usize| {
            let mut b = GraphBuilder::new();
            let x = b.input("x", [896, rows, 1, 1], crate::graph::DType::F32);
            let y = b.silu(x);
            b.output(y);
            let g = b.build();
            alloc.alloc_graph(&g).unwrap();
        };
        // 896 x 16 = 14336 and 896 x 17 = 15232 both round to the 16384-element class.
        let a = 896 * 16;
        let class = allocplan::class_size(a);
        assert_eq!(allocplan::class_size(896 * 17), class);
        build(&mut alloc, 16);
        let first = alloc.memory_report(Backend::CPU);
        assert_eq!(first.pool_bytes, allocplan::class_bytes(class), "{first:?}");
        let first_buffers = alloc.n_cpu_buffers();

        build(&mut alloc, 17);
        let second = alloc.memory_report(Backend::CPU);
        assert_eq!(
            second.pool_bytes, first.pool_bytes,
            "a shape inside the same class must recycle the class buffer, not reserve another"
        );
        assert_eq!(
            alloc.n_cpu_buffers(),
            first_buffers,
            "the pool grew for a shape that fits an existing class"
        );

        // The ladder is not a no-op: a shape that leaves the class does reserve more.
        build(&mut alloc, 20); // 896 x 20 = 17920 -> the next 16 KiB step
        assert!(
            alloc.memory_report(Backend::CPU).pool_bytes > first.pool_bytes,
            "a bigger class must actually reserve"
        );
    }

    /// E4 S2: the length contract moved to the owning `BufRef`, so a fill is checked
    /// against the node's **logical** length — the pool buffer is rounded up to a class
    /// and is routinely longer, which makes "equals the physical length" wrong and
    /// "fits in the physical length" too weak.
    #[test]
    fn a_fill_must_match_the_nodes_logical_length() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [3, 1, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        let report = alloc.memory_report(Backend::CPU);
        assert_eq!(
            report.pool_bytes,
            allocplan::class_bytes(allocplan::class_size(3)),
            "a 3-element activation still occupies a whole class: {report:?}"
        );

        alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0]).unwrap();
        for wrong in [vec![1.0f32, 2.0, 3.0, 4.0], vec![1.0f32, 2.0]] {
            let err = alloc.fill_input(&g, "x", &wrong).unwrap_err();
            assert!(
                err.contains(&format!(
                    "{} elements were supplied but the node holds 3",
                    wrong.len()
                )),
                "got: {err}"
            );
        }
        // The refused fills wrote nothing, and the read is the window, not the class.
        assert_eq!(alloc.get_buffer(&g, x).unwrap(), &[1.0, 2.0, 3.0]);
    }

    /// E4 S2: an input is host-filled **before** execution, so it must never take a buffer
    /// that this build's `sweep` released — the previous owner writes that buffer during
    /// execution, i.e. after the fill, and the input's value is gone by the time its
    /// consumer reads it. Here `t`'s buffer becomes free before `y` is placed, and `t`
    /// writes it at step 1.
    #[test]
    fn an_input_never_takes_a_buffer_the_walk_released() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
        let w = b.input("w", [8, 1, 1, 1], crate::graph::DType::F32);
        let t = b.add(x, w); // its own buffer; this is the only node that writes it
        let c = b.mul(t, w); // last use of `t`, so its buffer is released right after
        let y = b.input("y", [8, 1, 1, 1], crate::graph::DType::F32);
        let o = b.mul(c, y);
        b.output(o);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        let xs: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        let ws = vec![1.0f32; 8];
        let ys: Vec<f32> = (1..=8).map(|v| (v * 10) as f32).collect();
        alloc.fill_input(&g, "x", &xs).unwrap();
        alloc.fill_input(&g, "w", &ws).unwrap();
        alloc.fill_input(&g, "y", &ys).unwrap();
        crate::graph::scheduler::BackendScheduler::new()
            .execute(&g, &mut alloc)
            .expect("execute");
        let want: Vec<f32> = xs
            .iter()
            .zip(&ys)
            .map(|(a, b)| (a + 1.0) * 1.0 * b)
            .collect();
        assert_eq!(
            alloc.copy_to_cpu(o).expect("read"),
            want,
            "`y` must still hold its fill value after `t` executed"
        );
    }

    /// E4 S2: a view is a window of its parent's buffer, so the parent must stay alive
    /// through the **view's** last reader. D1 wrote this extension starting from the view
    /// itself — where the walk stops on its first check, because the view's own liveness
    /// is already the bound — so it never ran. Here `s` would take the parent's buffer
    /// (same class) and overwrite the window `p0` is read through.
    #[test]
    fn a_view_keeps_its_parents_buffer_alive_through_later_consumers() {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
        let w = b.input("w", [8, 1, 1, 1], crate::graph::DType::F32);
        let a = b.input("a", [4, 1, 1, 1], crate::graph::DType::F32);
        let a4 = b.input("a4", [4, 1, 1, 1], crate::graph::DType::F32);
        let t = b.add(x, w); // its own buffer, 8 elements (one class)
        let parts = b.split_parts(t, &[4, 4]);
        let p0 = parts[0]; // window [0, 4) of `t`'s buffer
        let s = b.mul(a, a4); // same class, placed after `t`'s own last use
        let o = b.mul(p0, s);
        b.output(o);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        let xs: Vec<f32> = (1..=8).map(|v| v as f32).collect();
        alloc.fill_input(&g, "x", &xs).unwrap();
        alloc.fill_input(&g, "w", &vec![1.0f32; 8]).unwrap();
        alloc.fill_input(&g, "a", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        alloc.fill_input(&g, "a4", &[1.0, 2.0, 3.0, 4.0]).unwrap();
        crate::graph::scheduler::BackendScheduler::new()
            .execute(&g, &mut alloc)
            .expect("execute");
        // p0 = (x + w)[0..4] = [2, 3, 4, 5] with w = 1; s = a * a4 = [1, 4, 9, 16].
        let want = vec![2.0, 12.0, 36.0, 80.0];
        assert_eq!(
            alloc.copy_to_cpu(o).expect("read"),
            want,
            "the window must still hold the parent's rows, not the recycled buffer's"
        );
    }

    /// E5: with a plan in force, the device is only offered for a node whose block the plan
    /// offloaded — and a node outside any block only under a full plan. Needs a device to be
    /// meaningful (without one the answer is CPU either way), so it is gated on the CUDA
    /// build and skips when no device is present.
    #[test]
    #[cfg(feature = "cuda")]
    fn the_offload_plan_keeps_late_blocks_off_the_device() {
        use crate::graph::offload::OffloadPlan;
        crate::cuda::CudaState::init();
        if crate::cuda::CudaState::get().is_none() {
            eprintln!("no CUDA device; skipping the E5 placement gate");
            return;
        }
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda());
        // No plan: the pre-E5 behaviour — the device takes every node it can.
        assert_eq!(
            alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(3)),
            Some(Backend::CUDA)
        );
        alloc.set_offload_plan(Some(OffloadPlan {
            gpu_layers: 2,
            n_layers: 4,
        }));
        assert_eq!(
            alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(0)),
            Some(Backend::CUDA),
            "an offloaded block may use the device"
        );
        assert_eq!(
            alloc.supports_for(&Op::Silu, crate::graph::DType::F32, Some(2)),
            Some(Backend::CPU),
            "a block past the plan stays on the CPU"
        );
        assert_eq!(
            alloc.supports_for(&Op::Silu, crate::graph::DType::F32, None),
            Some(Backend::CPU),
            "a partial plan keeps the unblocked tensors (embed/output) on the CPU"
        );
        // A full plan puts the unblocked tensors back on the device.
        alloc.set_offload_plan(Some(OffloadPlan {
            gpu_layers: 4,
            n_layers: 4,
        }));
        assert_eq!(
            alloc.supports_for(&Op::Silu, crate::graph::DType::F32, None),
            Some(Backend::CUDA)
        );
    }

    /// E5: a KV session is one arena (the container carries a single backend tag), so a mixed
    /// CPU/device plan cannot resume one. The refusal happens before the file is read — the
    /// path below does not exist, which is how the test proves the order.
    #[test]
    fn a_mixed_offload_plan_refuses_to_resume_a_session() {
        use crate::graph::kvsession::KvSessionExpect;
        let expect = KvSessionExpect {
            backend: Backend::CPU,
            n_ctx: 16,
            n_embd: 8,
        };
        let mut alloc = GraphAllocator::new();
        alloc.set_offload_plan(Some(crate::graph::offload::OffloadPlan {
            gpu_layers: 2,
            n_layers: 4,
        }));
        let err = alloc
            .kv_load(std::path::Path::new("/nonexistent/e5-mixed.bin"), &expect)
            .unwrap_err();
        assert!(err.contains("offload plan"), "got: {err}");
        // An all-CPU or all-device plan proceeds to the file and fails there instead.
        for plan in [
            crate::graph::offload::OffloadPlan {
                gpu_layers: 0,
                n_layers: 4,
            },
            crate::graph::offload::OffloadPlan {
                gpu_layers: 4,
                n_layers: 4,
            },
        ] {
            alloc.set_offload_plan(Some(plan));
            let err = alloc
                .kv_load(std::path::Path::new("/nonexistent/e5-mixed.bin"), &expect)
                .unwrap_err();
            assert!(!err.contains("offload plan"), "{plan:?}: {err}");
        }
    }

    /// E4 S3's acceptance: a rebuild **re-maps**. The reservation table hands the same
    /// class-sized buffers back, so the pool creates nothing (`CpuBackend::alloc_count`, the
    /// CPU twin of CUDA's `pool_gen`) and the node → slot mapping is identical — the
    /// "reserve, then assign" split, observable.
    #[test]
    fn a_rebuild_remaps_instead_of_reallocating() {
        /// The buffers a graph's nodes were assigned, by node id: `(backend, pool id)` —
        /// deliberately not the lengths, which differ between two shapes in a class.
        fn slots(
            alloc: &GraphAllocator,
            g: &ComputeGraph,
        ) -> Vec<(NodeId, Option<(Backend, usize)>)> {
            (0..g.n_nodes())
                .map(|i| (i, alloc.node_buffer(i).map(|b| (b.backend, b.id))))
                .collect()
        }
        let graph = |rows: usize| {
            let mut b = GraphBuilder::new();
            let x = b.input("x", [896, rows, 1, 1], crate::graph::DType::F32);
            let w = b.input("w", [896, rows, 1, 1], crate::graph::DType::F32);
            let t = b.add(x, w);
            let y = b.silu(t); // in-place: stays in `t`'s slot
            b.output(y);
            b.build()
        };

        let mut alloc = GraphAllocator::new();
        let g = graph(16);
        alloc.alloc_graph(&g).unwrap();
        let first = slots(&alloc, &g);
        let (allocs, buffers) = (alloc.n_cpu_allocs(), alloc.n_cpu_buffers());
        assert!(allocs > 0 && buffers > 0, "the first build does allocate");

        // The same graph, and a neighbour **in the same class** (896 x 17 = 15232 rounds to
        // the same 16384-element class as 896 x 16 = 14336): both re-map onto the reservation.
        for rows in [16, 17] {
            let g = graph(rows);
            alloc.alloc_graph(&g).unwrap();
            assert_eq!(
                alloc.n_cpu_allocs(),
                allocs,
                "a re-map must not create a pool buffer (rows = {rows})"
            );
            assert_eq!(alloc.n_cpu_buffers(), buffers);
            assert_eq!(
                slots(&alloc, &g),
                first,
                "the same topology gets the same slots (rows = {rows})"
            );
        }

        // A shape that leaves the class reserves a new buffer — the table is not a no-op.
        alloc.alloc_graph(&graph(64)).unwrap();
        assert!(alloc.n_cpu_allocs() > allocs, "a bigger class must reserve");
        assert!(alloc.n_cpu_buffers() > buffers);
    }

    /// E4 S3: the **reservation** is visible — a released classed buffer stays in the pool and
    /// goes to the idle list, so `live_bytes` drops back while `pool_bytes` does not, and the
    /// idle slot is what the next graph is assigned.
    #[test]
    fn a_released_buffer_stays_reserved_and_idle() {
        let mut alloc = GraphAllocator::new();
        let mut b = GraphBuilder::new();
        let x = b.input("x", [256, 1, 1, 1], crate::graph::DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();
        alloc.alloc_graph(&g).unwrap();
        let built = alloc.memory_report(Backend::CPU);
        assert!(built.pool_bytes > 0 && built.live_bytes > 0);
        // Rebuilding the same graph frees every classed buffer at the start and re-assigns
        // them, so at the end the reservation and the live set are exactly what they were.
        alloc.alloc_graph(&g).unwrap();
        let again = alloc.memory_report(Backend::CPU);
        assert_eq!(again.pool_bytes, built.pool_bytes);
        assert_eq!(again.live_bytes, built.live_bytes);
        assert_eq!(again.idle_slots, built.idle_slots);
    }

    /// E4 S3, CUDA half: a rebuild must not touch the device pool at all. `pool_gen` is the
    /// generation counter CUDA invalidates its captured graphs on, so "the plan is untouched"
    /// is exactly "the capture survives a rebuild".
    #[test]
    #[cfg(feature = "cuda")]
    fn a_rebuild_does_not_touch_the_device_pool() {
        crate::cuda::CudaState::init();
        if crate::cuda::CudaState::get().is_none() {
            eprintln!("no CUDA device; skipping the E4 S3 device gate");
            return;
        }
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda());
        let graph = |rows: usize| {
            let mut b = GraphBuilder::new();
            let x = b.input("x", [512, rows, 1, 1], crate::graph::DType::F32);
            let y = b.silu(x);
            let z = b.add(y, x);
            b.output(z);
            let mut g = b.build();
            // The device pool is what this gate is about: every node on CUDA.
            for n in g.nodes.iter_mut() {
                n.backend = Some(Backend::CUDA);
            }
            g
        };
        let g = graph(4);
        alloc.alloc_graph(&g).unwrap();
        let gen = alloc.cuda().unwrap().pool_gen();
        let mapping: Vec<_> = (0..g.n_nodes())
            .map(|i| alloc.node_buffer(i).map(|b| (b.backend, b.id)))
            .collect();
        // Same shape: no device traffic.
        alloc.alloc_graph(&g).unwrap();
        assert_eq!(
            alloc.cuda().unwrap().pool_gen(),
            gen,
            "a same-shape rebuild must not allocate or free on the device"
        );
        // Same class, different shape: still no device traffic. (512 x 4 = 2048 and
        // 512 x 3 = 1536 both round to the 2048-element class — asserted, so the case cannot
        // silently stop being the one it claims to test.)
        assert_eq!(
            crate::graph::allocplan::class_size(512 * 4),
            crate::graph::allocplan::class_size(512 * 3)
        );
        alloc.alloc_graph(&graph(3)).unwrap();
        assert_eq!(
            alloc.cuda().unwrap().pool_gen(),
            gen,
            "a shape in the same class re-maps onto the reserved device buffer"
        );
        let same: Vec<_> = (0..g.n_nodes())
            .map(|i| alloc.node_buffer(i).map(|b| (b.backend, b.id)))
            .collect();
        assert_eq!(same, mapping, "and onto the same slots");
    }

    /// The tiny f32 tensor the accounting tests register (`Tensor` carries raw bytes).
    fn tensor_f32(name: &str, shape: [i64; 4], data: Vec<f32>) -> crate::tensor::Tensor {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut t = crate::tensor::Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
        t.name = name.to_string();
        t
    }
}
