//! Cross-backend staging: the F5 (#58) split-boundary copies and the #138 deferred wait.
//!
//! Split out of `src/graph/cuda_backend/tests.rs` (issue #267): a pure move, so
//! the fixtures live in the parent module and are reached through `use super::*;`.

use super::*;

#[test]
fn cuda_pinned_readback_roundtrip() {
    // 5.6 MB > the 4 MiB initial pinned readback buffer: exercises the
    // grow-on-demand path of copy_from_device_pinned (R3-A2).
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut cb = CudaBackend::new().expect("backend after device init");
    let n = 1_400_000usize; // elements — alloc_buffer takes an element count
    let id = cb.alloc_buffer(n);
    let data: Vec<f32> = (0..n).map(|i| (i % 997) as f32 + 0.5).collect();
    cb.write_host(id, &data).unwrap();
    let got = cb.copy_to_host(id).unwrap();
    assert_eq!(got.len(), n);
    assert_eq!(got, data, "full roundtrip mismatch");
}
#[test]
fn copy_across_cpu_to_cuda_and_back() {
    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let mut b = crate::graph::builder::GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let y = b.silu(x);
    b.output(y);
    let g = b.build();

    let mut alloc = GraphAllocator::new();
    if !alloc.enable_cuda() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    alloc.alloc_graph(&g).unwrap();
    let data = [1.0f32, -2.0, 3.0, 4.0];
    alloc.fill_input(&g, "x", &data).unwrap();

    let canon = alloc.node_buffer(x).unwrap();
    // CPU → CUDA staging copy: canonical buffer untouched, cross map holds
    // the device copy (Phase 7c: the old remap-into-node_to_buf semantics
    // broke re-execution of reused graphs — the producing split found its
    // buffer remapped to another backend on the next execute)
    alloc
        .copy_across(1, x, crate::graph::Backend::CUDA)
        .unwrap();
    let cross = alloc
        .cross_buffer(1, x, crate::graph::Backend::CUDA)
        .expect("cross staging buffer");
    assert_eq!(cross.backend, crate::graph::Backend::CUDA);
    assert_eq!(
        alloc.node_buffer(x).unwrap(),
        canon,
        "canonical buffer must not be remapped"
    );
    assert_eq!(alloc.copy_to_cpu(x).unwrap(), data.to_vec());
    // re-copy (same dst) reuses the same staging buffer id
    alloc
        .copy_across(1, x, crate::graph::Backend::CUDA)
        .unwrap();
    assert_eq!(
        alloc
            .cross_buffer(1, x, crate::graph::Backend::CUDA)
            .unwrap()
            .id,
        cross.id
    );
    // a same-backend copy is a no-op and does not create staging for CPU
    alloc.copy_across(1, x, crate::graph::Backend::CPU).unwrap();
    assert!(
        alloc
            .cross_buffer(1, x, crate::graph::Backend::CPU)
            .is_none(),
        "the copy was already on the destination backend"
    );
    assert!(alloc
        .cross_buffer(1, x, crate::graph::Backend::CUDA)
        .is_some());
    // E4 S3: a rebuild keeps the staging buffer — it is keyed by (graph uid, node,
    // backend), so it belongs to this graph's shape and survives a re-map. Re-creating it
    // per rebuild leaked (staging is `alloc_fresh`, which never recycles).
    alloc.alloc_graph(&g).unwrap();
    assert_eq!(
        alloc
            .cross_buffer(1, x, crate::graph::Backend::CUDA)
            .expect("staging survives a re-map")
            .id,
        cross.id,
        "and it is the same buffer, not a fresh one"
    );
    // A different graph (its own uid) gets its own entry: node ids restart per graph.
    assert!(alloc
        .cross_buffer(2, x, crate::graph::Backend::CUDA)
        .is_none());
}
/// F5 ([#58]) device gate: **a real split graph's boundary copies out of the
/// device are asynchronous, every staged input owes exactly one wait — issued
/// since #138 at the consumer's first use rather than at the boundary — the
/// boundary close no longer blocks the host, and the result is bitwise identical
/// to the synchronous reference.**
///
/// This is the cheap, focused half of the F5 acceptance (the real-model half is
/// `models::qwen2::graph::async_cross_copies_never_block_and_stay_bitwise_identical`):
/// a three-node graph whose middle node is pinned to the device makes the
/// scheduler alternate CPU → CUDA → CPU, so the boundary stages one value *into*
/// the device and one *out of* it. Both modes run the same kernels in the same
/// order — only the transfer differs — so bitwise equality is the honest claim.
///
/// The counters are deterministic, which is why this is the mutation gate: with
/// the consumer path's resolver removed (a bare `cross_input` in the node loop),
/// `copies == waits` fails **and** the consumer's read of the still-pending
/// staging entry is a loud error.
#[test]
fn a_split_graph_waits_once_per_staged_copy_and_stays_bitwise() {
    use crate::graph::copystats::{self, CrossCopyStats};
    use crate::graph::scheduler::BackendScheduler;

    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let data = [1.0f32, -2.0, 3.5, 4.25];
    // x → silu (device) → add(silu, x) (CPU). Every backend is set explicitly:
    // `split_graph` makes an unassigned node inherit the previous one's backend.
    let build = || {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let o = b.add(s, x);
        b.output(o);
        let mut g = b.build();
        g.nodes[0].backend = Some(crate::graph::Backend::CPU);
        g.nodes[1].backend = Some(crate::graph::Backend::CUDA);
        g.nodes[2].backend = Some(crate::graph::Backend::CPU);
        g
    };
    // The mode override is process-wide, so the whole measurement holds the
    // gate (the parallel harness shares the process).
    let gate = copystats::gate();
    let run = |sync: bool| -> (Vec<f32>, CrossCopyStats, u64, u64) {
        let _mode = copystats::set_sync_for_test(sync);
        let g = build();
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda(), "a CUDA device answered the probe");
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &data).unwrap();
        let before = alloc.cross_stats();
        let readbacks_before = alloc.cuda().unwrap().blocking_readback_count();
        // #185: read the count through **this** backend, never a process-wide
        // total — a concurrent device test's syncs would otherwise land inside
        // this delta (the failure the ticket records).
        let syncs_before = alloc.cuda().unwrap().stream_sync_count();

        BackendScheduler::new().execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, 2).unwrap().to_vec();
        let stats = alloc.cross_stats().delta(before);
        let readbacks = alloc.cuda().unwrap().blocking_readback_count() - readbacks_before;
        (
            got,
            stats,
            readbacks,
            alloc.cuda().unwrap().stream_sync_count() - syncs_before,
        )
    };

    let (async_out, a, a_readbacks, a_syncs) = run(false);
    let (sync_out, s, s_readbacks, s_syncs) = run(true);
    drop(gate);

    eprintln!(
        "[f5] split graph: async copies={} waits={} blocking={} async_host={} event_syncs={} \
         reads={} syncs={} | sync copies={} waits={} blocking={} reads={} syncs={}",
        a.copies,
        a.waits,
        a.blocking_host_copies,
        a.async_host_copies,
        a.event_syncs,
        a_readbacks,
        a_syncs,
        s.copies,
        s.waits,
        s.blocking_host_copies,
        s_readbacks,
        s_syncs
    );

    // The graph really crosses the boundary in both directions: one CPU→CUDA
    // copy (node 0 into the device split) and one CUDA→CPU copy (node 1 back
    // out). Node 0 feeding the last split is a same-backend input and must not
    // be counted.
    assert_eq!(a.copies, 2, "one copy in, one copy out");
    assert_eq!(
        a.async_host_copies, 1,
        "the CUDA→CPU copy took the async path"
    );
    assert_eq!(a.event_syncs, 1, "…and owes exactly one event wait");
    assert_eq!(
        a.deferred_waits, a.copies,
        "every staged input's wait is issued at the consumer's first use (#138), \
         so the boundary itself waits for none of them"
    );
    assert!(
        a.all_copies_awaited(),
        "one wait per staged input: {} copies, {} waits",
        a.copies,
        a.waits
    );
    assert_eq!(
        a.blocking_host_copies, 0,
        "no blocking device→host copy on the boundary"
    );
    assert_eq!(a_readbacks, 0, "no blocking readback on the boundary");
    assert_eq!(
        a_syncs, 0,
        "the boundary close no longer blocks the host (#138): the copies are \
         stream-ordered behind the producer's work, so the wait is deferred"
    );
    assert!(
        a_syncs < s_syncs,
        "the async boundary removes the per-copy stream sync ({a_syncs} vs {s_syncs})"
    );
    assert_eq!(
        s.blocking_host_copies, 1,
        "the synchronous reference blocks on the device→host copy — the 'before' number"
    );
    assert!(s_readbacks >= 1, "the device counter sees the sync path");
    assert_eq!(
        async_out, sync_out,
        "the async staging copies must be bitwise identical to the synchronous reference"
    );
    let silu = |v: f32| v / (1.0 + (-v).exp());
    for (i, v) in data.iter().enumerate() {
        assert!((async_out[i] - (silu(*v) + v)).abs() < 1e-5);
    }
}
/// #138 ([#138]) device gate: **a boundary with several staged inputs enqueues
/// them all before waiting on any of them, so copy N+1 is in flight while copy N
/// (and the consumer's independent work) is still running — measurably fewer
/// serialized host stalls than the F5 enqueue-then-wait boundary.**
///
/// The graph makes the device split produce **two** values that the CPU split
/// consumes, with an independent CPU node between the boundary and the first
/// staged read:
///
/// ```text
/// x ──► silu (CUDA) ──► p ─┐
///   └──► silu (CUDA) ──► q ─┼─► add(p, ind) ──► add(q, ·)   (CPU)
///   └──► add(x, x) ──► ind ─┘
/// ```
///
/// The measured metrics, both device-side and deterministic:
///
/// * **in-flight copies** — [`CudaBackend::cross_inflight_peak`], the high-water
///   mark of pinned slabs held by copies that have been enqueued but not waited
///   on. The deferred boundary reaches **2** (both staged inputs); the F5
///   discipline, which the gate then drives by hand on the same buffers
///   (`copy_across` + `await_cross` per input), cannot exceed **1**, because the
///   wait releases the slab before the next copy takes one.
/// * **host stalls** — `stream_syncs` is **0** for the deferred boundary against
///   **2** for the synchronous reference over the same two copies; the F5
///   boundary paid one full stream sync per boundary *plus* its per-copy event
///   waits (that sync is what this ticket removes — it is redundant once the copy
///   is stream-ordered behind the producer).
///
/// `deferred_waits == copies` and `copies == waits` together say every wait moved
/// to a consumer read and none was dropped.
#[test]
fn a_boundary_with_several_staged_inputs_defers_its_waits() {
    use crate::graph::copystats::{self, CrossCopyStats};
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::Backend as BTag;

    if device().is_none() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let data = [1.0f32, -2.0, 3.5, 4.25];
    // Every backend is set explicitly: `split_graph` makes an unassigned node
    // inherit the previous one's backend.
    let build = || {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let p = b.silu(x); // 1: device producer
        let q = b.silu(x); // 2: second device producer
        let ind = b.add(x, x); // 3: independent CPU work, no staged input
        let m = b.add(p, ind); // 4: first use of the first staged input
        let o = b.add(q, m); // 5: first use of the second
        b.output(o);
        let mut g = b.build();
        g.nodes[0].backend = Some(BTag::CPU);
        g.nodes[1].backend = Some(BTag::CUDA);
        g.nodes[2].backend = Some(BTag::CUDA);
        for n in 3..=5 {
            g.nodes[n].backend = Some(BTag::CPU);
        }
        g
    };
    // The mode override is process-wide, so the whole measurement holds the gate
    // (the parallel harness shares the process).
    let gate = copystats::gate();
    let run = |sync: bool| -> (Vec<f32>, CrossCopyStats, u64, usize) {
        let _mode = copystats::set_sync_for_test(sync);
        let g = build();
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda(), "a CUDA device answered the probe");
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &data).unwrap();
        let before = alloc.cross_stats();
        let syncs_before = alloc.cuda().unwrap().stream_sync_count();

        BackendScheduler::new().execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, 5).unwrap().to_vec();
        let stats = alloc.cross_stats().delta(before);
        let syncs = alloc.cuda().unwrap().stream_sync_count() - syncs_before;
        let peak = alloc.cuda().unwrap().cross_inflight_peak();
        (got, stats, syncs, peak)
    };

    let (deferred_out, d, d_syncs, deferred_peak) = run(false);
    let (sync_out, s, s_syncs, sync_peak) = run(true);

    // Two disciplines measured on the same buffers, each on a fresh backend so
    // the high-water mark starts at zero.
    let (f5_peak, re_request) = {
        let _mode = copystats::set_sync_for_test(false);
        let g = build();
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_cuda());
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &data).unwrap();
        // Produce p and q on the device (the CPU→CUDA copies are not the metric).
        BackendScheduler::new().execute(&g, &mut alloc).unwrap();

        // (a) The deferred state makes a second request for the same
        // `(graph, node, destination)` reachable while the first is in flight —
        // unreachable under the F5 boundary, which awaited before re-enqueuing.
        // It is the same transfer into the same staging buffer, so it must not be
        // counted twice (that would also leave the first record's slab and event
        // behind: the allocator clears one pending key per entry).
        let before = alloc.cross_stats();
        alloc.copy_across(g.uid, 1, BTag::CPU).unwrap();
        alloc.copy_across(g.uid, 1, BTag::CPU).unwrap();
        let in_flight = alloc.cross_stats().delta(before);
        alloc.await_cross(g.uid, 1, BTag::CPU).unwrap();
        let settled = alloc.cross_stats().delta(before);
        let re_request = (in_flight.copies, settled.copies, settled.waits);

        // (b) The F5 copy discipline: enqueue one, wait on it, then the next —
        // the wait releases the slab, so two copies can never be in flight at
        // once. This is the "before" of the overlap metric.
        alloc.cuda_mut().unwrap().reset_cross_inflight_peak();
        for node in [1usize, 2] {
            alloc.copy_across(g.uid, node, BTag::CPU).unwrap();
            alloc.await_cross(g.uid, node, BTag::CPU).unwrap();
        }
        (alloc.cuda().unwrap().cross_inflight_peak(), re_request)
    };
    drop(gate);

    eprintln!(
        "[#138] deferred: copies={} waits={} deferred_waits={} async_host={} event_syncs={} \
         stream_syncs={} inflight_peak={} | sync: copies={} waits={} blocking={} stream_syncs={} \
         inflight_peak={} | F5 discipline inflight_peak={} | re-request while in flight: \
         copies={} (after its one wait: copies={} waits={})",
        d.copies,
        d.waits,
        d.deferred_waits,
        d.async_host_copies,
        d.event_syncs,
        d_syncs,
        deferred_peak,
        s.copies,
        s.waits,
        s.blocking_host_copies,
        s_syncs,
        sync_peak,
        f5_peak,
        re_request.0,
        re_request.1,
        re_request.2
    );
    assert_eq!(
        (re_request.0, re_request.1, re_request.2),
        (1, 1, 1),
        "a re-request while in flight is the same copy and owes the same single wait"
    );

    // The boundary really staged several inputs in both directions: one CPU→CUDA
    // (x, deduplicated across the two device producers) and two CUDA→CPU (p, q).
    assert_eq!(d.copies, 3, "one copy in, two out: {d:?}");
    assert_eq!(d.async_host_copies, 2, "both device→host copies are async");
    assert_eq!(d.event_syncs, 2, "and each owes exactly one event wait");
    assert_eq!(
        d.deferred_waits, d.copies,
        "every wait moved to a consumer read (#138): {d:?}"
    );
    assert!(
        d.all_copies_awaited(),
        "one wait per staged input: {} copies, {} waits",
        d.copies,
        d.waits
    );
    assert_eq!(
        d.blocking_host_copies, 0,
        "no blocking D2H copy on the path"
    );
    assert_eq!(
        d_syncs, 0,
        "the deferred boundary blocks the host nowhere; the F5 boundary paid one \
         full stream sync per boundary on top of its per-copy event waits"
    );
    assert_eq!(
        deferred_peak, 2,
        "both staged copies were in flight at once (copy N+1 enqueued while copy N \
         was still in flight)"
    );
    assert_eq!(
        f5_peak, 1,
        "the F5 enqueue-then-wait discipline cannot hold two copies in flight"
    );
    // The synchronous reference: no async copies, so no slabs, and its two
    // blocking readbacks pay a stream sync each.
    assert_eq!(sync_peak, 0, "the synchronous path issues no async copy");
    assert_eq!(s.blocking_host_copies, 2, "the reference blocks per copy");
    assert!(
        s_syncs >= 2,
        "one stream sync per blocking copy ({s_syncs})"
    );
    assert_eq!(
        deferred_out, sync_out,
        "the deferred staging copies must be bitwise identical to the synchronous reference"
    );
    let silu = |v: f32| v / (1.0 + (-v).exp());
    for (i, v) in data.iter().enumerate() {
        let p = silu(*v);
        let q = silu(*v);
        let ind = v + v;
        assert!((deferred_out[i] - (q + (p + ind))).abs() < 1e-5);
    }
}
