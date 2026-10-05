//! Metal cross-backend staging: the F5 (#58) split-boundary copies as ported by
//! #137 (`MTLBlitCommandEncoder` + `MTLSharedEvent`), plus the #138 deferred
//! wait. Split out of `src/graph/metal_backend/tests.rs` (issue #267 pattern), so
//! the fixtures live in the parent module and are reached through `use super::*;`.
//!
//! The `#[ignore]`d real-model half of the same acceptance is
//! `models::qwen2::graph::tests::offload_copy::async_cross_copies_never_block_and_stay_bitwise_identical_on_metal`;
//! these are the cheap, deterministic device gates, and each one is the named
//! gate of a mutation in the ticket.

use super::*;

/// F5 (#137) device gate: **a real split graph's boundary copies out of Metal
/// are asynchronous, every staged input owes exactly one wait (issued since #138
/// at the consumer's first use), and the result is bitwise identical to the
/// synchronous reference.**
///
/// The graph `x → silu (Metal) → add(silu, x) (CPU)` alternates CPU → Metal →
/// CPU, so the boundary stages one value *into* the device (the CPU source's
/// host round trip) and one *out of* it (the Metal source's blit + shared
/// event). Both modes run the same kernels in the same order — only the transfer
/// differs — so bitwise equality is the honest claim.
///
/// The device-level evidence is
/// [`MetalBackend::sync_readback_count`], Metal's counterpart of
/// `CudaBackend::blocking_readback_count`: the async path publishes its own
/// staging bytes and never reads the source pool buffer back, so the counter
/// must not move; the synchronous reference reads the source through
/// `read_host`, so it must.
///
/// Mutations this gate catches: dropping the consumer path's wait makes
/// `copies == waits` fail (and the checked reader refuse); letting phase A take
/// the synchronous fallback moves `blocking_host_copies` off zero and makes the
/// device counter move in async mode.
#[test]
fn a_split_graph_waits_once_per_staged_copy_and_stays_bitwise() {
    use crate::graph::copystats::{self, CrossCopyStats};
    use crate::graph::scheduler::BackendScheduler;

    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let data = [1.0f32, -2.0, 3.5, 4.25];
    // Every backend is set explicitly: `split_graph` makes an unassigned node
    // inherit the previous one's backend.
    let build = || {
        let mut b = GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let s = b.silu(x);
        let o = b.add(s, x);
        b.output(o);
        let mut g = b.build();
        g.nodes[0].backend = Some(Tag::CPU);
        g.nodes[1].backend = Some(Tag::METAL);
        g.nodes[2].backend = Some(Tag::CPU);
        (g, x, s, o)
    };
    // The mode override is process-wide, so the whole measurement holds the gate
    // (the parallel harness shares the process; `metal_test_lock` does too).
    let gate = copystats::gate();
    let run = |sync: bool| -> (Vec<f32>, CrossCopyStats, u64) {
        let _mode = copystats::set_sync_for_test(sync);
        let (g, _x, _s, o) = build();
        let mut alloc = GraphAllocator::new();
        assert!(alloc.enable_metal(), "a Metal device answered the probe");
        alloc.alloc_graph(&g).unwrap();
        alloc.fill_input(&g, "x", &data).unwrap();
        let before = alloc.cross_stats();
        let readbacks_before = alloc.metal().unwrap().sync_readback_count();

        BackendScheduler::new().execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, o).unwrap().to_vec();
        let stats = alloc.cross_stats().delta(before);
        let readbacks = alloc.metal().unwrap().sync_readback_count() - readbacks_before;
        (got, stats, readbacks)
    };

    let (async_out, a, a_readbacks) = run(false);
    let (sync_out, s, s_readbacks) = run(true);
    drop(gate);

    eprintln!(
        "[f5-metal] async copies={} waits={} deferred={} blocking={} async_host={} event_syncs={} \
         sync_readbacks={} | sync copies={} waits={} blocking={} sync_readbacks={}",
        a.copies,
        a.waits,
        a.deferred_waits,
        a.blocking_host_copies,
        a.async_host_copies,
        a.event_syncs,
        a_readbacks,
        s.copies,
        s.waits,
        s.blocking_host_copies,
        s_readbacks
    );

    // One CPU→Metal copy in, one Metal→CPU copy out; a same-backend input is
    // not counted.
    assert_eq!(a.copies, 2, "one copy in, one copy out");
    assert_eq!(a.async_host_copies, 1, "the Metal→CPU copy is async");
    assert_eq!(a.event_syncs, 1, "…and owes exactly one event wait");
    assert!(
        a.all_copies_awaited(),
        "one wait per staged input: {} copies, {} waits",
        a.copies,
        a.waits
    );
    assert_eq!(
        a.deferred_waits, a.copies,
        "every staged input's wait is issued at the consumer's first use (#138)"
    );
    assert_eq!(
        a.blocking_host_copies, 0,
        "no blocking device→host copy on the async boundary"
    );
    assert_eq!(
        a_readbacks, 0,
        "the async path must not read the source buffer back to the host"
    );

    assert_eq!(
        s.blocking_host_copies, 1,
        "the synchronous reference blocks on the Metal→host copy — the 'before' number"
    );
    assert!(
        s_readbacks >= 1,
        "the device-level counter sees the synchronous path's readback"
    );
    assert_eq!(
        async_out, sync_out,
        "the async staging copies must be bitwise identical to the synchronous reference"
    );
    let silu = |v: f32| v / (1.0 + (-v).exp());
    for (i, v) in data.iter().enumerate() {
        assert!((async_out[i] - (silu(*v) + v)).abs() < 1e-5);
    }
}

/// #138 mutation gate: **a re-request for the same `(graph, node, destination)`
/// while the first copy is still in flight is the same transfer** — one record,
/// one wait, no duplicated transfer and no leaked staging buffer / event.
///
/// `copy_across` makes the re-request a no-op through `cross_pending`; this gate
/// runs the boundary by hand on a real Metal source so it can also assert the
/// backend-side record count, which the allocator's counter alone cannot see.
#[test]
fn a_re_request_while_in_flight_is_the_same_transfer() {
    use crate::graph::scheduler::BackendScheduler;

    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    let o = b.add(s, x);
    b.output(o);
    let mut g = b.build();
    g.nodes[0].backend = Some(Tag::CPU);
    g.nodes[1].backend = Some(Tag::METAL);
    g.nodes[2].backend = Some(Tag::CPU);

    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_metal(), "a Metal device answered the probe");
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0, -2.0, 3.5, 4.25]).unwrap();
    // The Metal split produced `s`; retire it exactly as the scheduler does, so
    // the source buffer holds this run's data.
    BackendScheduler::new().execute(&g, &mut alloc).unwrap();

    let before = alloc.cross_stats();
    let readbacks_before = alloc.metal().unwrap().sync_readback_count();
    alloc.copy_across(g.uid, s, Tag::CPU).unwrap();
    alloc.copy_across(g.uid, s, Tag::CPU).unwrap();
    let in_flight = alloc.cross_stats().delta(before);
    assert_eq!(
        in_flight.copies, 1,
        "the re-request is the same transfer and must not be counted twice"
    );
    assert_eq!(
        alloc.metal().unwrap().cross_pending_len(),
        1,
        "…and must not create a second staging record (a leaked buffer/event)"
    );

    alloc.await_cross(g.uid, s, Tag::CPU).unwrap();
    let settled = alloc.cross_stats().delta(before);
    assert_eq!(
        (settled.copies, settled.waits),
        (1, 1),
        "the same single copy owes the same single wait"
    );
    assert_eq!(
        alloc.metal().unwrap().cross_pending_len(),
        0,
        "the record is released by its one wait"
    );
    assert_eq!(
        alloc.metal().unwrap().sync_readback_count(),
        readbacks_before,
        "the async transfer reads only its own staging bytes"
    );
}

/// The missing-wait gate on the Metal source: **a staged entry may not be read
/// before its boundary wait**. `GraphAllocator::cross_input` is the checked
/// reader; the consumer path (`cross_input_ready`) is the only thing that issues
/// the wait, and it does so at the first read.
///
/// Mutation: delete the `cross_input_ready` call in the scheduler's node loop
/// (use the raw `cross_buffer`) and this is the loud error the consumer gets —
/// never a silent read of in-flight bytes.
#[test]
fn a_staged_entry_read_before_its_wait_is_refused() {
    use crate::graph::scheduler::BackendScheduler;

    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    let o = b.add(s, x);
    b.output(o);
    let mut g = b.build();
    g.nodes[0].backend = Some(Tag::CPU);
    g.nodes[1].backend = Some(Tag::METAL);
    g.nodes[2].backend = Some(Tag::CPU);

    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_metal(), "a Metal device answered the probe");
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0, -2.0, 3.5, 4.25]).unwrap();
    BackendScheduler::new().execute(&g, &mut alloc).unwrap();

    // Phase A only: the Metal copy is in flight.
    alloc.copy_across(g.uid, s, Tag::CPU).unwrap();
    let err = alloc
        .cross_input(g.uid, s, Tag::CPU)
        .expect_err("the checked reader must refuse a staged entry that was not awaited");
    assert!(
        err.contains("read before its boundary wait"),
        "the refusal must name the missing wait, got: {err}"
    );

    // The resolver issues the wait at the first read and then the same read
    // succeeds — so the missing-wait path is a guard, not the only path.
    let staged = alloc.cross_input_ready(g.uid, s, Tag::CPU).unwrap();
    assert!(
        staged.is_some(),
        "the resolved staging buffer is now readable"
    );
    assert!(alloc.cross_input(g.uid, s, Tag::CPU).unwrap().is_some());
}

/// The failure-channel gate: **a staging copy whose wait does not complete is a
/// loud `Err` naming the real status — never a silent fallback.**
///
/// `MINFER_TEST_CALL_FAIL=metal_cross_copy` ([`docs/GATE-CONTRACT.md`]) makes
/// phase A suppress its `MTLSharedEvent` signal, so phase B's *bounded* wait
/// takes its genuine timeout branch (a 1 ms bound under injection, so the gate
/// stays fast) and reports the value it waited for and the observed
/// `signaledValue`. The injected blit is real, the wait is real, and the error
/// is the production timeout path — not a mock.
///
/// [`docs/GATE-CONTRACT.md`]: ../../../docs/GATE-CONTRACT.md
#[test]
fn a_timed_out_cross_wait_is_a_loud_err() {
    use crate::graph::scheduler::BackendScheduler;

    let _g = crate::metal::metal_test_lock();
    crate::metal::MpsState::init();
    if MetalBackend::new().is_none() {
        eprintln!("MPS unavailable; skipping");
        return;
    }
    let mut b = GraphBuilder::new();
    let x = b.input("x", [4, 1, 1, 1], DType::F32);
    let s = b.silu(x);
    let o = b.add(s, x);
    b.output(o);
    let mut g = b.build();
    g.nodes[0].backend = Some(Tag::CPU);
    g.nodes[1].backend = Some(Tag::METAL);
    g.nodes[2].backend = Some(Tag::CPU);

    let mut alloc = GraphAllocator::new();
    assert!(alloc.enable_metal(), "a Metal device answered the probe");
    alloc.alloc_graph(&g).unwrap();
    alloc.fill_input(&g, "x", &[1.0, -2.0, 3.5, 4.25]).unwrap();

    let _inject = crate::testfail::InjectionGuard::arm(CROSS_COPY_SITE);
    let err = BackendScheduler::new()
        .execute(&g, &mut alloc)
        .expect_err("the injected timeout must surface as a loud Err");
    assert!(
        err.contains("MTLSharedEvent") && err.contains("timed out"),
        "the error must report the real wait status, got: {err}"
    );
    assert!(
        err.contains(CROSS_COPY_SITE),
        "the error must name the injection site so a mutated run is identifiable, got: {err}"
    );
    assert!(
        err.contains("observed signaledValue"),
        "the error must report the observed status, got: {err}"
    );
    // The record was released on the failure path, so the run leaks no event or
    // staging buffer.
    assert_eq!(alloc.metal().unwrap().cross_pending_len(), 0);
}
