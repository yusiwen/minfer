//! F5 staging: the destination key, the pending copy and the end-of-run drain.
//!
//! Split out of `src/graph/alloc/tests.rs` (issue #267): a pure move, so the fixtures live in
//! the parent module and are reached through `use super::*;`.

use super::*;

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
/// `scheduler::tests::a_staged_boundary_input_is_waited_on_at_its_first_use`.
#[test]
fn a_pending_staged_copy_is_refused_until_its_wait_is_issued() {
    let g = {
        use crate::graph::builder::GraphBuilder;
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
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
/// #138 gate: **the end-of-execution drain issues the wait for a staged entry
/// nothing read**, so `copies == waits` cannot depend on what the consumer did
/// with its inputs.
///
/// A deferred wait is issued at the consumer's first read, and the scheduler
/// skips node loops (a replayed CUDA split) and dead nodes, so an entry can go
/// unread. Without the drain it would stay pending forever — a pinned slab and an
/// event held past the execution, and the missing-wait gate reading
/// `copies > waits`. The drain is deliberately *not* counted as a deferred wait:
/// it is not issued at a use.
#[test]
fn the_drain_waits_on_a_staged_entry_nothing_read() {
    let g = {
        use crate::graph::builder::GraphBuilder;
        let mut b = GraphBuilder::new();
        let x = b.input("x", [8, 1, 1, 1], crate::graph::DType::F32);
        let s = b.silu(x);
        b.output(s);
        b.build()
    };
    let mut alloc = GraphAllocator::new();
    alloc.alloc_graph(&g).unwrap();
    // Two staged entries for a notional device consumer; `await_cross`
    // dispatches to the CPU entry's registered no-op (the CPU has no device
    // transfer to wait on), so the test stays device-free.
    alloc.stage_cross_for_test(g.uid, 1, Backend::CUDA, 4, 8);
    alloc.mark_cross_pending_for_test(g.uid, 1, Backend::CUDA);
    alloc.stage_cross_for_test(g.uid, 0, Backend::CUDA, 5, 8);
    alloc.mark_cross_pending_for_test(g.uid, 0, Backend::CUDA);

    // The consumer reads one of them through the deferred path.
    assert!(alloc
        .cross_input_ready(g.uid, 1, Backend::CUDA)
        .unwrap()
        .is_some());
    // The other is still pending — and still unreadable.
    assert!(alloc.cross_input(g.uid, 0, Backend::CUDA).is_err());

    alloc.drain_cross_pending().unwrap();
    assert!(
        alloc
            .cross_input(g.uid, 0, Backend::CUDA)
            .unwrap()
            .is_some(),
        "the drain published the entry nothing read"
    );
    assert_eq!(
        alloc.cross_stats(),
        CrossCopyStats {
            copies: 0, // injected state: no phase A ran
            waits: 2,
            deferred_waits: 1, // only the read one was deferred
            ..CrossCopyStats::default()
        }
    );
    // Draining again is a no-op (nothing pending).
    alloc.drain_cross_pending().unwrap();
    assert_eq!(alloc.cross_stats().waits, 2);
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
        let x = b.input("x", [4, 1, 1, 1], crate::graph::DType::F32);
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
