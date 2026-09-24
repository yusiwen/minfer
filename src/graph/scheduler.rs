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
use super::{Backend as BackendTag, BufRef, CNode, ComputeGraph, NodeId};

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
        Self {
            backend,
            node_range: (start, start),
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
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
    ///
    /// E5: the node's block is part of the question — `supports_for` keeps a node whose
    /// block the offload plan left on the CPU off the device, so a partially offloaded
    /// model never runs a block whose weights were never registered there.
    pub fn assign_backends(&self, graph: &mut ComputeGraph, alloc: &GraphAllocator) {
        for node in &mut graph.nodes {
            if node.backend.is_some() {
                continue; // keep explicit assignments
            }
            node.backend = alloc
                .supports_for(&node.op, node.out_dtype, node.layer)
                .or(Some(BackendTag::CPU));
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
            let b = node
                .backend
                .unwrap_or_else(|| cur.unwrap_or(BackendTag::CPU));
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
    pub fn execute(&self, graph: &ComputeGraph, alloc: &mut GraphAllocator) -> Result<(), String> {
        // C1 gate (Phase C). While the KV mapping is the identity, a backend may
        // index the arenas with the raw `positions` input — which is what all
        // three implement. Once C2 introduces a hole or a window the mapping
        // stops being the identity, the resolved cell array must reach the
        // kernel instead, and a backend that has not been ported has to refuse
        // the node rather than write the wrong row (standing rule 2: never a
        // silent wrong answer). It never fires today; it exists so C2 cannot
        // ship the hazard by accident, and the day it fires is the day a backend
        // is missing its port.
        if !alloc.kv_is_identity() {
            return Err(
                "KV cell mapping is no longer the identity: the resolved cell array must be \
                 passed to the backend, and no backend consumes it yet (Phase C / C2)"
                    .to_string(),
            );
        }
        #[cfg(debug_assertions)]
        debug_assert!(graph.topo_order().is_ok(), "graph is not a valid DAG");
        let splits = self.split_graph(graph);
        if std::env::var("MINFER_GRAPH_TRACE").is_ok() {
            for (si, s) in splits.iter().enumerate() {
                eprintln!(
                    "[graph] split {si}: {:?} nodes {}-{}",
                    s.backend, s.node_range.0, s.node_range.1
                );
            }
            let mut by = std::collections::BTreeMap::new();
            for n in &graph.nodes {
                *by.entry((
                    format!("{:?}", n.op).split('{').next().unwrap().to_string(),
                    n.backend.map(|b| format!("{b:?}")).unwrap_or_default(),
                ))
                .or_insert(0usize) += 1;
            }
            for (k, v) in by {
                eprintln!("[graph] op {:<12} backend {:<7} x{v}", k.0, k.1);
            }
        }
        let mut prev_backend: Option<BackendTag> = None;
        // P2 trace (MINFER_TRACE) + P3 live (viz): read back every node's
        // output (stats + downsampled sample). Checked once per execute() call;
        // one step per execute() (prefill = 1 step, each decode forward = 1).
        // Capture happens AFTER each node executes (this step's data):
        //   CPU → read immediately; Metal → blit into staging at split end,
        //   read after the split's command buffer is submitted (one submit per
        //   split — no per-node GPU flush). KV regions are skipped on Metal
        //   (staging would be huge; captured in full on CPU). Inputs are
        //   host-filled by the allocator → read directly.
        let trace_on = crate::trace::enabled();
        let live_on = crate::live::enabled();
        let capture = trace_on || live_on;
        // F8: resolved once per execute() (a cached relaxed load), so the
        // per-node check is a branch and the clock is only read when it is on.
        let op_timing = crate::optiming::enabled();
        if trace_on {
            crate::trace::begin_step();
        }
        if live_on {
            crate::live::begin_step();
        }
        // (node_id, src_buf_id) for the current Metal split, then (node_id,
        // staging_id) awaiting readback after that split's sync.
        #[cfg(target_os = "macos")]
        let mut metal_srcs: Vec<(usize, usize)> = Vec::new();
        let mut staged: Vec<(usize, usize)> = Vec::new();
        // CUDA capture: node ids whose outputs were queued into the pinned
        // capture staging this split, drained (one sync) at the boundary.
        // Always empty without the `cuda` feature (the push sites below are
        // gated the same way) — declared unconditionally so the boundary
        // `flush_cuda_captures` calls compile on every configuration.
        let mut cuda_caps: Vec<usize> = Vec::new();
        for split in &splits {
            if let Some(pb) = prev_backend {
                if pb != split.backend {
                    // 1. flush the previous backend's async work
                    alloc.sync_backend(pb);
                    // 1b. staged Metal/CUDA captures are valid now — read back
                    flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
                    flush_cuda_captures(graph, alloc, &mut cuda_caps, trace_on, live_on);
                    // 2. F5 phase A — enqueue this split's cross-backend staging
                    //    copies. A device source's transfer is an async
                    //    `cudaMemcpyAsync` plus a recorded event; nothing here
                    //    blocks the host on a copy.
                    for &inp in &split.inputs {
                        alloc.copy_across(graph.uid, inp, split.backend)?;
                    }
                    // 3. F5 phase B — the split boundary's **synchronization
                    //    points**: one wait per staged input, in the same order,
                    //    before any consumer can read the staging buffer. This is
                    //    the wait the missing-wait gate is about: dropping it
                    //    leaves the staging entry pending, and the consumer's read
                    //    below (`cross_input`) fails loudly instead of reading a
                    //    transfer that may still be in flight. See
                    //    `docs/BACKEND-REGISTRY-DESIGN.md` §11.
                    for &inp in &split.inputs {
                        alloc.await_cross(graph.uid, inp, split.backend)?;
                    }
                }
            }
            // CUDA Graph replay (Phase 7d): a captured split replays its whole
            // node loop as one launch. Disabled under MINFER_TRACE/viz
            // capture (per-node host readbacks inside a capture window are
            // illegal — they would corrupt the recorded graph).
            #[cfg(feature = "cuda")]
            let replayed = if capture || split.backend != BackendTag::CUDA {
                false
            } else {
                let c = alloc.cuda_mut().ok_or("CUDA backend not enabled")?;
                c.graph_replay(graph.uid, split.node_range, graph.capture_nt_hint())
            };
            #[cfg(not(feature = "cuda"))]
            let replayed = false;
            if replayed {
                // the captured launch covers every node of this split (the
                // boundary sync of the NEXT split still closes it out)
                prev_backend = Some(split.backend);
                continue;
            }
            for id in split.node_range.0..split.node_range.1 {
                let node = graph.node(id);
                if capture && node.is_input() {
                    // inputs are host-filled before execute — no pending GPU
                    // work, so reading them here is always current
                    if let Some(br) = alloc.node_buffer(id) {
                        if let Some(d) = read_host_buffer(alloc, br.backend, br.id)
                            .and_then(|d| window_of(br, d))
                        {
                            record_node_data(node, d, trace_on, live_on);
                        }
                    }
                }
                if node.is_input() {
                    continue; // data pre-filled by the allocator
                }
                // dead nodes (no consumers, not outputs) get no buffer — the
                // fusion pass can orphan them (e.g. silu folded into SwiGLU);
                // they are skipped, not executed
                let Some(br) = alloc.node_buffer(id) else {
                    continue;
                };
                if br.backend != split.backend {
                    let op_full = format!("{:?}", node.op);
                    let op = op_full.split(['(', '{']).next().unwrap_or("?");
                    return Err(format!(
                        "node {id} ({op}) buffer on {:?} but executing split is {:?} (assignment/alloc mismatch)",
                        br.backend, split.backend
                    ));
                }
                let mut in_bufs = Vec::with_capacity(node.src.len());
                for &s in &node.src {
                    // A split-boundary staging copy takes precedence, but only
                    // the one made FOR this split's backend: the staging map is
                    // keyed by (node, destination backend), so a node consumed
                    // by two different backends has one buffer each and neither
                    // consumer can pick up the other's. Otherwise fall back to
                    // the node's canonical buffer (already on this split's
                    // backend when no copy was needed).
                    //
                    // F5: `cross_input` (not the raw `cross_buffer`) so a staged
                    // entry whose boundary wait was skipped is a loud error, never
                    // a read of an in-flight transfer.
                    let sbr = alloc
                        .cross_input(graph.uid, s, split.backend)?
                        .or_else(|| alloc.node_buffer(s))
                        .ok_or_else(|| format!("node {s} has no allocated buffer"))?;
                    in_bufs.push(sbr);
                }
                // resolve the layer's KV region pair BEFORE the mutable backend
                // borrow (the backend needs it for KV store / attention)
                let kv_pair = match &node.op {
                    Op::KvcacheStore { layer } => alloc.kv_pair(*layer),
                    Op::FusedQKV { layer } => alloc.kv_pair(*layer),
                    Op::QkvBiasRopeStore { layer } => alloc.kv_pair(*layer),
                    Op::FusedQkvNorm { layer } => alloc.kv_pair(*layer),
                    Op::Attn { .. } => match &node.meta {
                        NodeMeta::Attn(m) => alloc.kv_pair(m.layer),
                        _ => None,
                    },
                    _ => None,
                };
                // NOTE: execution follows node id order (build order), which is
                // the graph's topological order by construction.
                //
                // F8: per-op timing wraps exactly this dispatch when
                // `MINFER_OP_TIMING` is set. `t0` is `None` on the default path,
                // so the flag off means the clock is never read. See
                // `crate::optiming` for what the interval does and does not
                // attribute.
                let t0 = if op_timing {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // F4: one dispatch for every backend — the registry entry's pool
                // hook, then the trait's `execute_node` on it. The per-backend
                // `#[cfg]` arms (and their "unavailable" strings) are gone; a
                // backend whose pool is not enabled is a loud error naming it.
                let pool = alloc.pool_mut(split.backend).ok_or_else(|| {
                    format!(
                        "{} backend not enabled: {}",
                        split.backend.name(),
                        crate::graph::registry::unavailable_reason(split.backend)
                            .unwrap_or("the backend's pool is not enabled")
                    )
                })?;
                pool.execute_node(node, &in_bufs, br, kv_pair)?;
                if let Some(t0) = t0 {
                    crate::optiming::record(crate::optiming::op_index(&node.op), t0.elapsed());
                }
                // CAPTURE AFTER EXECUTION — this step's output
                if capture {
                    // KV regions are huge (n_embd × n_ctx per layer) — skipped
                    // on both backends (page shows "no data for this node in this
                    // step"); everything else is captured.
                    let is_kv = matches!(node.op, Op::KvcacheStore { .. } | Op::KvcacheLoad { .. });
                    if !is_kv {
                        match br.backend {
                            BackendTag::CPU => {
                                if let Some(d) =
                                    alloc.cpu().read_host(br.id).and_then(|d| window_of(br, d))
                                {
                                    record_node_data(node, d, trace_on, live_on);
                                }
                            }
                            #[cfg(target_os = "macos")]
                            BackendTag::METAL => {
                                metal_srcs.push((id, br.id));
                            }
                            // CUDA: queue an async D2H into the pinned capture
                            // staging (stream-ordered — pool buffers recycle
                            // intra-split, so the copy must sit behind the
                            // producing kernel); ONE sync at the split
                            // boundary drains it all. Buffers that do not fit
                            // under the staging ceiling fall back to the
                            // per-node sync copy (GB-scale prefill tensors).
                            #[cfg(feature = "cuda")]
                            BackendTag::CUDA => {
                                if let Some(c) = alloc.cuda_mut() {
                                    if c.capture_enq(br.id) {
                                        cuda_caps.push(id);
                                    } else if let Some(v) = c.copy_to_host(br.id) {
                                        if let Some(d) = window_of(br, &v) {
                                            record_node_data(node, d, trace_on, live_on);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
            }
            // encode this split's Metal captures as one blit pass (after all of
            // its kernels, so the staging holds this step's output)
            #[cfg(target_os = "macos")]
            if !metal_srcs.is_empty() {
                let src_ids: Vec<usize> = metal_srcs.iter().map(|&(_, b)| b).collect();
                let dsts = alloc
                    .metal_mut()
                    .ok_or("Metal backend not enabled")?
                    .capture_split(&src_ids)?;
                for ((nid, _), st) in metal_srcs.iter().zip(dsts.into_iter()) {
                    staged.push((*nid, st));
                }
                metal_srcs.clear();
            }
            prev_backend = Some(split.backend);
        }
        if let Some(pb) = prev_backend {
            alloc.sync_backend(pb);
            flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
            flush_cuda_captures(graph, alloc, &mut cuda_caps, trace_on, live_on);
        }
        Ok(())
    }
}

/// Slice a physical host read of a pool buffer down to the window its node owns.
///
/// E4 S2 rounds a pooled activation buffer up to its size class, so the read is
/// routinely longer than the node's data: capture must report the node's
/// elements, never the padding (which holds another buffer's stale bytes and
/// would silently change the viz's counts and statistics). `None` when the read
/// is shorter than the reference, which would be an allocation bug.
fn window_of(br: BufRef, data: &[f32]) -> Option<&[f32]> {
    data.get(br.offset..br.offset + br.len)
}

/// Read a buffer's host data. CPU: direct. Metal: only safe for host-filled
/// inputs (no pending GPU work); staged Metal outputs go through
/// `flush_metal_captures` instead.
///
/// F4: the trait's borrowed `read_host` through the registry's pool hook — not
/// the `host_read` hook, because a device whose read is a *copy* (CUDA) has no
/// borrowed form (`read_host` is `None` there), which is exactly the pre-F4
/// answer for this path.
fn read_host_buffer(alloc: &GraphAllocator, backend: BackendTag, id: usize) -> Option<&[f32]> {
    alloc.pool(backend)?.read_host(id)
}

/// Analyze + record one node's output (shared by the immediate and the staged
/// Metal paths).
fn record_node_data(node: &CNode, data: &[f32], trace_on: bool, live_on: bool) {
    let dn = crate::graph::json::dtype_name(node.out_dtype);
    let (stats, values, stride, n_total) = crate::trace::analyze(dn, data);
    if trace_on {
        crate::trace::record_node(node.id, dn, stats, values.clone(), stride, n_total);
    }
    if live_on {
        crate::live::record_node(
            node.id,
            &node.name,
            crate::graph::json::op_name(&node.op),
            dn,
            stats,
            &values,
            stride,
            n_total,
        );
    }
}

/// Read back staged Metal captures (valid after their split's command buffer
/// was submitted by `sync_backend`), then return the staging buffers to the
/// free list.
fn flush_metal_captures(
    graph: &ComputeGraph,
    alloc: &mut GraphAllocator,
    staged: &mut Vec<(usize, usize)>,
    trace_on: bool,
    live_on: bool,
) {
    // `staged` is only ever populated by Metal splits (above), so on non-macOS
    // it is always empty; the whole read-back body lives on macOS only.
    #[cfg(target_os = "macos")]
    {
        for (id, st) in staged.drain(..) {
            let node = graph.node(id);
            let br = alloc.node_buffer(id);
            if let (Some(d), Some(br)) = (alloc.metal().and_then(|m| m.read_staging(st)), br) {
                if let Some(d) = window_of(br, d) {
                    record_node_data(node, d, trace_on, live_on);
                }
            }
        }
        if let Some(m) = alloc.metal_mut() {
            m.release_staging_all();
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (graph, alloc, staged, trace_on, live_on);
}

/// Read back the CUDA capture staging (one stream sync inside `capture_drain`,
/// valid after the split's `sync_backend` boundary) and emit the queued node
/// events in enqueue order.
#[cfg(feature = "cuda")]
fn flush_cuda_captures(
    graph: &ComputeGraph,
    alloc: &mut GraphAllocator,
    caps: &mut Vec<usize>,
    trace_on: bool,
    live_on: bool,
) {
    if caps.is_empty() {
        return;
    }
    let Some(c) = alloc.cuda_mut() else {
        caps.clear();
        return;
    };
    let data = c.capture_drain();
    for (nid, v) in caps.drain(..).zip(data) {
        let node = graph.node(nid);
        let Some(br) = alloc.node_buffer(nid) else {
            continue;
        };
        if let Some(d) = window_of(br, &v) {
            record_node_data(node, d, trace_on, live_on);
        }
    }
}

/// Non-CUDA stub: `caps` is only ever populated by CUDA splits, so without
/// the `cuda` feature it is always empty here (same pattern as the
/// non-macOS stub of `flush_metal_captures`).
#[cfg(not(feature = "cuda"))]
fn flush_cuda_captures(
    graph: &ComputeGraph,
    alloc: &mut GraphAllocator,
    caps: &mut Vec<usize>,
    trace_on: bool,
    live_on: bool,
) {
    let _ = (graph, alloc, caps, trace_on, live_on);
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
        g.nodes[1].backend = Some(BackendTag::METAL);
        g.nodes[2].backend = Some(BackendTag::CPU);
        let sched = BackendScheduler::new();
        let splits = sched.split_graph(&g);
        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].backend, BackendTag::CPU);
        assert_eq!(splits[1].backend, BackendTag::METAL);
        assert_eq!(splits[2].backend, BackendTag::CPU);
        assert_eq!(splits[0].outputs, vec![0]);
        assert_eq!(splits[1].inputs, vec![0]);
        assert_eq!(splits[1].outputs, vec![1]);
        assert_eq!(splits[2].inputs, vec![1, 0]);
    }

    /// C1 gate (Phase C): the executor must refuse a graph whose KV mapping is
    /// no longer the identity, because no backend consumes the resolved cell
    /// array yet — a half-ported C2 has to fail loudly, not write the wrong row.
    #[test]
    fn a_non_identity_kv_mapping_is_refused() {
        let g = small_graph();
        let sched = BackendScheduler::new();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();

        // The identity mapping is what every backend implements: it executes.
        sched.execute(&g, &mut alloc).unwrap();

        // Flip the gate the way C2 would (a hole, a window, …).
        alloc.kv_clear_identity();
        let err = sched.execute(&g, &mut alloc).unwrap_err();
        assert!(
            err.contains("no longer the identity"),
            "expected the C1 gate to fire, got: {err}"
        );
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

    /// F8 (#51): `MINFER_OP_TIMING` reports numbers, it never computes them.
    ///
    /// The gate runs the *same* graph twice through the *same* scheduler, once
    /// with the flag off and once on, and asserts the output buffer is
    /// bit-identical — the flagged path may only differ in the table it fills.
    /// The second half asserts the table actually moved, so the equality above is
    /// not the equality of "timing is broken and never ran".
    ///
    /// Both halves take `optiming::GATE` because `force` is process-global: a
    /// parallel graph test executing while the flag is forced on would otherwise
    /// pollute the delta.
    #[test]
    fn op_timing_does_not_change_the_result_but_does_accumulate() {
        let _g = crate::optiming::gate();
        let g = small_graph();
        let sched = BackendScheduler::new();
        let input = [1.0f32, 2.0, 3.0, 4.0];

        let run = |timing: bool| -> Vec<f32> {
            crate::optiming::force(timing);
            let mut alloc = GraphAllocator::new();
            alloc.alloc_graph(&g).unwrap();
            alloc.fill_input(&g, "x", &input).unwrap();
            sched.execute(&g, &mut alloc).unwrap();
            alloc.get_buffer(&g, 2).unwrap().to_vec()
        };

        // Off first, so the on-run is the only thing that can fill the table.
        let off = run(false);
        let before = crate::optiming::snapshot();
        let calls_before = before
            .iter()
            .find(|e| e.name == "silu")
            .map_or(0, |e| e.calls);
        let on = run(true);
        crate::optiming::force(false);
        let after = crate::optiming::snapshot();
        let calls_after = after
            .iter()
            .find(|e| e.name == "silu")
            .map_or(0, |e| e.calls);

        assert_eq!(
            off, on,
            "the timing flag must not perturb the computation, only the report"
        );
        assert!(
            calls_after > calls_before,
            "the flagged run must have recorded silu executions ({calls_before} -> {calls_after})"
        );
        // And with the flag off again, the scheduler reads the clock zero times:
        // the table is unchanged by a run.
        let quiet = run(false);
        assert_eq!(quiet, off);
        let now = crate::optiming::snapshot()
            .iter()
            .find(|e| e.name == "silu")
            .map_or(0, |e| e.calls);
        assert_eq!(now, calls_after, "a run with the flag off records nothing");
    }

    /// F5 ([#58]) gate: **a staged cross-backend input cannot be consumed before
    /// its boundary wait**.
    ///
    /// This is the ticket's second acceptance line made deterministic on a
    /// CPU-only build. The test puts the allocator in exactly the state between
    /// phase A and phase B (a staged entry that has not been waited on), runs the
    /// **real** `execute` — whose node loop resolves every input through
    /// `cross_input` — and asserts it fails **loudly**, naming the missing wait,
    /// instead of consuming the staging buffer. Issuing the wait publishes the
    /// entry and the same graph then executes to completion.
    ///
    /// Why the state is injected with a test hook rather than produced by a real
    /// boundary: a boundary needs two usable backends, and the only backend a
    /// CPU-only CI can enable is the CPU (a cross-backend split on this build
    /// would have to be a device). The CUDA
    /// `async_cross_copies_never_block_and_stay_bitwise_identical` gate produces
    /// the state end-to-end on a device; together they cover the invariant from
    /// both ends. The mutation evidence (dropping the boundary's `await_cross`
    /// loop) is recorded in `docs/ARCHITECTURE-EXECUTION-PLAN.md`.
    #[test]
    fn a_staged_boundary_input_cannot_be_consumed_before_its_wait() {
        let g = small_graph();
        let sched = BackendScheduler::new();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();

        // Stage node 1 (the silu, a source of node 2) on the CPU — the same
        // backend, so it is the canonical buffer — and mark its copy un-waited.
        let br = alloc.node_buffer(1).unwrap();
        alloc.stage_cross_for_test(g.uid, 1, BackendTag::CPU, br.id, br.len);
        alloc.mark_cross_pending_for_test(g.uid, 1, BackendTag::CPU);

        let err = sched
            .execute(&g, &mut alloc)
            .expect_err("a pending staged input must not be consumed");
        assert!(
            err.contains("was read before its boundary wait"),
            "the refusal must name the missing wait, got: {err}"
        );
        assert!(err.contains("#58"), "{err}");

        // Phase B publishes it; the graph then runs. (The staged reference is the
        // canonical buffer, so the values are unchanged.)
        alloc.await_cross(g.uid, 1, BackendTag::CPU).unwrap();
        sched.execute(&g, &mut alloc).unwrap();
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let got = alloc.get_buffer(&g, 2).unwrap();
        for i in 0..4 {
            assert!((got[i] - (silu((i + 1) as f32) + (i + 1) as f32)).abs() < 1e-5);
        }
    }

    /// F5 ([#58]) gate: **the CPU-only path is provably untouched.**
    ///
    /// A CPU-only graph is a single split, so the boundary never runs: the
    /// counters must stay at zero in *both* modes, and forcing the synchronous
    /// reference (`MINFER_SYNC_COPIES=1`'s code path) must not change a byte.
    /// There is no device copy to overlap, and this is the measurement that says
    /// so rather than asserting it.
    ///
    /// The mode is set programmatically: the environment is process-wide and the
    /// parallel harness shares it.
    #[test]
    fn the_cpu_path_never_enters_the_cross_copy_machinery() {
        let g = small_graph();
        let sched = BackendScheduler::new();
        let input = [1.0f32, -2.0, 3.5, 4.25];

        let run = || -> (Vec<f32>, crate::graph::copystats::CrossCopyStats) {
            let mut alloc = GraphAllocator::new();
            alloc.alloc_graph(&g).unwrap();
            alloc.fill_input(&g, "x", &input).unwrap();
            sched.execute(&g, &mut alloc).unwrap();
            (
                alloc.get_buffer(&g, 2).unwrap().to_vec(),
                alloc.cross_stats(),
            )
        };

        let (async_bytes, async_stats) = {
            let _g = crate::graph::copystats::set_sync_for_test(false);
            run()
        };
        let (sync_bytes, sync_stats) = {
            let _g = crate::graph::copystats::set_sync_for_test(true);
            run()
        };

        assert_eq!(
            async_bytes, sync_bytes,
            "the sync reference must be bitwise identical on the CPU path"
        );
        assert_eq!(
            async_stats,
            crate::graph::copystats::CrossCopyStats::default(),
            "a CPU-only graph has no boundary, so no copy and no wait may be counted"
        );
        assert_eq!(sync_stats, async_stats);
        assert!(
            async_stats.all_copies_awaited(),
            "the contract is trivially satisfied when nothing crossed"
        );
    }
}
