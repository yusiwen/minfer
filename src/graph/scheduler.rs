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
use super::{Backend as BackendTag, CNode, ComputeGraph, NodeId};

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
    pub fn assign_backends(&self, graph: &mut ComputeGraph, alloc: &GraphAllocator) {
        for node in &mut graph.nodes {
            if node.backend.is_some() {
                continue; // keep explicit assignments
            }
            node.backend = alloc
                .supports(&node.op, node.out_dtype)
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
        // P2 trace (MINFER_TRACE) + P3 live (--viz): read back every node's
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
        for split in &splits {
            if let Some(pb) = prev_backend {
                if pb != split.backend {
                    // 1. flush the previous backend's async work
                    alloc.sync_backend(pb);
                    // 1b. staged Metal captures are valid now — read them back
                    flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
                    // 2. copy this split's inputs across backends
                    for &inp in &split.inputs {
                        alloc.copy_across(inp, split.backend)?;
                    }
                }
            }
            for id in split.node_range.0..split.node_range.1 {
                let node = graph.node(id);
                if capture && node.is_input() {
                    // inputs are host-filled before execute — no pending GPU
                    // work, so reading them here is always current
                    if let Some(br) = alloc.node_buffer(id) {
                        if let Some(d) = read_host_buffer(alloc, br.backend, br.id) {
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
                    BackendTag::CPU => alloc
                        .cpu_mut()
                        .execute_node(node, &in_bufs, br.id, kv_pair)?,
                    #[cfg(target_os = "macos")]
                    BackendTag::Metal => {
                        let m = alloc.metal_mut().ok_or("Metal backend not enabled")?;
                        m.execute_node(node, &in_bufs, br.id, kv_pair)?;
                    }
                    #[cfg(not(target_os = "macos"))]
                    BackendTag::Metal => return Err("Metal unavailable".into()),
                    BackendTag::Cuda => return Err("CUDA backend not implemented".into()),
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
                                if let Some(d) = alloc.cpu().read_host(br.id) {
                                    record_node_data(node, d, trace_on, live_on);
                                }
                            }
                            #[cfg(target_os = "macos")]
                            BackendTag::Metal => {
                                metal_srcs.push((id, br.id));
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
        }
        Ok(())
    }
}

/// Read a buffer's host data. CPU: direct. Metal: only safe for host-filled
/// inputs (no pending GPU work); staged Metal outputs go through
/// `flush_metal_captures` instead.
fn read_host_buffer(alloc: &GraphAllocator, backend: BackendTag, id: usize) -> Option<&[f32]> {
    match backend {
        BackendTag::CPU => alloc.cpu().read_host(id),
        #[cfg(target_os = "macos")]
        BackendTag::Metal => alloc.metal().and_then(|m| m.read_host(id)),
        _ => None,
    }
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
            if let Some(d) = alloc.metal().and_then(|m| m.read_staging(st)) {
                record_node_data(node, d, trace_on, live_on);
            }
        }
        if let Some(m) = alloc.metal_mut() {
            m.release_staging_all();
        }
    }
    #[cfg(not(target_os = "macos"))]
    let _ = (graph, alloc, staged, trace_on, live_on);
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
