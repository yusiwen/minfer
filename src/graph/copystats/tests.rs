//! `#[cfg(test)] mod tests` for `src/graph/copystats.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

#[test]
fn the_two_phases_are_counted_separately_and_the_delta_is_exact() {
    let a = CrossCopyStats {
        copies: 3,
        waits: 3,
        deferred_waits: 3,
        blocking_host_copies: 0,
        async_host_copies: 2,
        event_syncs: 2,
        stream_waits: 1,
    };
    assert!(a.all_copies_awaited());
    // A dropped wait is visible: this is the invariant the gate reads.
    let dropped = CrossCopyStats { waits: 2, ..a };
    assert!(!dropped.all_copies_awaited());

    let b = CrossCopyStats {
        copies: 5,
        waits: 5,
        deferred_waits: 4,
        blocking_host_copies: 1,
        async_host_copies: 4,
        event_syncs: 4,
        stream_waits: 1,
    };
    assert_eq!(
        b.delta(a),
        CrossCopyStats {
            copies: 2,
            waits: 2,
            deferred_waits: 1,
            blocking_host_copies: 1,
            async_host_copies: 2,
            event_syncs: 2,
            stream_waits: 0,
        }
    );
    // A delta is saturating, so a caller that snapshots after the fact
    // reads zero rather than wrapping.
    assert_eq!(a.delta(b), CrossCopyStats::default());
}

/// The gate is a mode switch, not a one-way door: forcing synchronous and
/// back leaves the default (async) in force. The whole check holds
/// [`gate`], because the override is process-wide and the parallel harness
/// shares it.
#[test]
fn the_sync_mode_override_restores_itself() {
    let _gate = gate();
    assert!(async_copies_enabled(), "the async substrate is the default");
    {
        let _m = set_sync_for_test(true);
        assert!(sync_copies());
        assert!(!async_copies_enabled());
    }
    assert!(async_copies_enabled());
}
