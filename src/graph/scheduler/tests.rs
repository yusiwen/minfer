//! `#[cfg(test)] mod tests` for `src/graph/scheduler.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
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

/// Run `small_graph` once through a scheduler whose timing goes to `sink`
/// (`enabled` is its flag), returning the output row. #173's isolation is
/// that every caller owns `sink`, so the assertions read only their own rows.
fn run_small_graph(sink: std::sync::Arc<crate::optiming::TimingSink>, enabled: bool) -> Vec<f32> {
    let g = small_graph();
    let sched =
        BackendScheduler::with_timing(crate::optiming::TimingMode::Private { sink, enabled });
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    sched.execute(&g, &mut alloc).unwrap();
    alloc.get_buffer(&g, 2).unwrap().to_vec()
}

/// The `calls` value for `name` in `sink`, or 0 when the op never ran.
fn calls_in(sink: &crate::optiming::TimingSink, name: &str) -> u64 {
    sink.snapshot()
        .into_iter()
        .find(|e| e.name == name)
        .map_or(0, |e| e.calls)
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

/// #171: the observation half of the failure-injection seam.
///
/// A gate that must prove the backend execute entry **ran** cannot read the
/// dispatch's own answer — that would be self-certifying — so the chokepoint
/// bumps `testfail::note_checked("execute_node")` and the gate asserts the
/// counter advanced. The expected count is the graph's non-input nodes (an
/// input is host-filled, never dispatched); the counter mutation (making
/// `note_checked` a no-op) is what this test catches.
#[test]
fn the_execute_chokepoint_is_observable() {
    crate::testfail::reset_checked();
    assert_eq!(crate::testfail::checked("execute_node"), 0);

    let g = small_graph();
    // An input node is host-filled by the allocator and never dispatched, so
    // the expected count is the non-input nodes of the graph.
    let dispatched = g.nodes.iter().filter(|n| !n.is_input()).count() as u64;
    assert!(dispatched > 0);
    let sched = BackendScheduler::new();
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0, 2.0, 3.0, 4.0]).unwrap();
    sched.execute(&g, &mut alloc).unwrap();

    assert_eq!(
        crate::testfail::checked("execute_node"),
        dispatched,
        "every dispatched node must be observed at the one execute entry"
    );
    // A site that did not run stays at zero: the counter is per site, not a
    // single "something executed" flag.
    assert_eq!(crate::testfail::checked("forward_batch"), 0);
}

/// F8 (#51): `MINFER_OP_TIMING` reports numbers, it never computes them.
///
/// The gate runs the *same* graph through two schedulers with **private**
/// sinks: the first with timing off, the second on. The output buffer must be
/// bit-identical — the timed path may only differ in the sink it fills — and
/// the assertions read **values** from the sink the run under test wrote:
/// exactly one `silu` and one `add` (`small_graph` is input → silu → add, and
/// an input is never dispatched), nothing after the timing-off run. That is
/// gate-contract rule 1: the old relation between two snapshots of shared
/// state was exactly what a concurrent execution could move (#173).
#[test]
fn op_timing_does_not_change_the_result_but_does_accumulate() {
    let off_sink = std::sync::Arc::new(crate::optiming::TimingSink::new());
    let on_sink = std::sync::Arc::new(crate::optiming::TimingSink::new());

    // Off first, and assert the *value*: a scheduler whose timing is off
    // records nothing into the sink it owns.
    let off = run_small_graph(off_sink.clone(), false);
    assert!(
        off_sink.snapshot().is_empty(),
        "a run with timing off records nothing: {:?}",
        off_sink.snapshot()
    );

    let on = run_small_graph(on_sink.clone(), true);
    assert_eq!(
        off, on,
        "the timing sink must not perturb the computation, only the report"
    );
    // One silu and one add per execute(), read from this run's own sink.
    assert_eq!(
        calls_in(&on_sink, "silu"),
        1,
        "one silu per execute: {:?}",
        on_sink.snapshot()
    );
    assert_eq!(
        calls_in(&on_sink, "add"),
        1,
        "one add per execute: {:?}",
        on_sink.snapshot()
    );
}

/// #173 control arm: **a concurrent graph execution cannot move a private
/// sink's verdict.**
///
/// Several load threads run the *real* `execute` at once and record into the
/// **process-global** sink — the shared destination the original fault
/// polluted. While they run, this thread asserts exact values from a private
/// sink. Had the gate read the shared table, the load's `silu` records would
/// already make `== 1` false; a per-scheduler sink cannot see them. The load
/// is bounded by work (`LOAD_THREADS * LOAD_ITERS` executes), every thread is
/// joined by `scope`, and there are no sleeps or clocks — gate rule 4.
///
/// The shared sink is reset first and its exact total asserted, so the test
/// also proves the control load really recorded (and that the free `record` /
/// `reset` / `snapshot` still address the one shared table `/metrics` reads).
#[test]
fn a_concurrent_graph_load_cannot_move_a_private_sink() {
    const LOAD_THREADS: usize = 4;
    const LOAD_ITERS: usize = 64;
    const CHECK_ROUNDS: usize = 8;

    crate::optiming::reset();
    // One extra record through the free function: the baseline the load adds
    // to, and the proof the free entry point writes the same shared table.
    crate::optiming::record(
        crate::optiming::op_index(&Op::Silu),
        std::time::Duration::from_nanos(1),
    );
    let global = crate::optiming::global_sink();

    std::thread::scope(|scope| {
        for _ in 0..LOAD_THREADS {
            scope.spawn(|| {
                for _ in 0..LOAD_ITERS {
                    // Each thread owns its graph and records into the shared
                    // global sink — the load, not a private one.
                    let _ = run_small_graph(global.clone(), true);
                }
            });
        }

        // While the load runs, the gate's own sink holds exact values.
        for round in 0..CHECK_ROUNDS {
            let sink = std::sync::Arc::new(crate::optiming::TimingSink::new());
            let _ = run_small_graph(sink.clone(), true);
            let snap = sink.snapshot();
            assert_eq!(
                calls_in(&sink, "silu"),
                1,
                "round {round}: the concurrent load moved a private sink: {snap:?}"
            );
            assert_eq!(snap.len(), 2, "round {round}: exactly silu + add: {snap:?}");
        }
    });

    // The control load really hammered the shared destination: had the gate
    // read it, its `== 1` would already be false.
    let shared = crate::optiming::snapshot();
    assert_eq!(
        calls_in(&crate::optiming::global_sink(), "silu"),
        1 + (LOAD_THREADS * LOAD_ITERS) as u64,
        "the control load must have recorded into the shared sink: {shared:?}"
    );
    crate::optiming::reset(); // leave the shared sink as we found it
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
/// The mode is set programmatically and the whole test holds
/// `copystats::gate()`: the override is process-wide and the parallel
/// harness shares it.
#[test]
fn the_cpu_path_never_enters_the_cross_copy_machinery() {
    let _gate = crate::graph::copystats::gate();
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
        let _m = crate::graph::copystats::set_sync_for_test(false);
        run()
    };
    let (sync_bytes, sync_stats) = {
        let _m = crate::graph::copystats::set_sync_for_test(true);
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

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `scheduler.rs` (bucket B of the dead-code census —
// the only test caller lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl BackendScheduler {
    /// A scheduler that times wherever `mode` says. The #173 isolation seam: a
    /// test hands in its own sink so its verdict reads only the rows its own
    /// scheduler wrote, and a concurrent graph execution cannot move them.
    ///
    /// Test-only (#239): driven by the `run_small_graph` helper of
    /// `graph::scheduler::tests` (called by
    /// `op_timing_does_not_change_the_result_but_does_accumulate` and
    /// `a_concurrent_graph_load_cannot_move_a_private_sink`).
    pub fn with_timing(mode: crate::optiming::TimingMode) -> Self {
        Self { timing: mode }
    }
}
