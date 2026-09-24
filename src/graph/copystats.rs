//! F5 ([#58]): cross-backend staging-copy instrumentation and the
//! synchronous-reference gate.
//!
//! The scheduler's split boundary is the only place the engine moves a value
//! between backends. Since F5 that move has **two phases**:
//!
//! - **phase A** (`GraphAllocator::copy_across`) *enqueues* the transfer. A
//!   backend whose source has device memory does it asynchronously and records
//!   an event; a backend without device memory performs the synchronous host
//!   round trip it always did.
//! - **phase B** (`GraphAllocator::await_cross`) *waits* on that event, exactly
//!   once per staged input, at the documented synchronization point (before the
//!   consuming split executes).
//!
//! This module owns the counters those two phases bump, so "no host-side
//! blocking copy on the hot path" is a number a gate can read instead of a
//! claim, and it owns the process-wide switch that restores the pre-F5
//! synchronous path for the bitwise A/B.
//!
//! **The counters live on the allocator, not in a process static.** A static
//! would make two tests running in parallel pollute each other's numbers
//! (`optiming` needs a mutex gate for exactly that reason); an allocator owns
//! one graph's execution, which is the scope these counters describe.
//!
//! [#58]: https://github.com/yusiwen/minfer/issues/58

use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

/// What the split boundary did in one execution, per allocator.
///
/// Read it as two pairs: `copies`/`waits` are the two phases of the contract
/// (one wait per copy — [`CrossCopyStats::all_copies_awaited`]), and
/// `blocking_host_copies`/`async_host_copies` classify the *device→host*
/// half — the one the ticket is about.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CrossCopyStats {
    /// Phase A: staging copies issued at a split boundary.
    pub copies: u64,
    /// Phase B: waits issued at a split boundary. One per staged input, so it
    /// equals `copies` on every path that honours the contract.
    pub waits: u64,
    /// Phase A copies that blocked the host on a device→host read: the pre-F5
    /// path (a stream sync plus a blocking `cudaMemcpy`) and the fallback for a
    /// direction no backend has made asynchronous yet. **This is the counter the
    /// "no blocking copy on the hot path" gate asserts is zero.**
    pub blocking_host_copies: u64,
    /// Phase A device→host copies enqueued asynchronously (no host block here).
    pub async_host_copies: u64,
    /// Phase B waits that blocked the host on a device event — the *documented*
    /// synchronization point of a device→host staging copy. The data cannot be
    /// read before this, so it is a wait, not a copy: the transfer itself never
    /// blocked.
    pub event_syncs: u64,
    /// Phase B waits that only inserted a device-side wait (a device consumer).
    /// The copy is ordered on the consuming stream, so the host never blocks.
    pub stream_waits: u64,
}

// Read by the F5 gates (tests) and by a caller that wants a delta rather than a
// lifetime total; production code only bumps the counters.
#[allow(dead_code)]
impl CrossCopyStats {
    /// The change from `before` to `self` — what a gate asserts on, so an
    /// unrelated earlier execution cannot shift the numbers.
    pub fn delta(self, before: CrossCopyStats) -> CrossCopyStats {
        CrossCopyStats {
            copies: self.copies.saturating_sub(before.copies),
            waits: self.waits.saturating_sub(before.waits),
            blocking_host_copies: self
                .blocking_host_copies
                .saturating_sub(before.blocking_host_copies),
            async_host_copies: self
                .async_host_copies
                .saturating_sub(before.async_host_copies),
            event_syncs: self.event_syncs.saturating_sub(before.event_syncs),
            stream_waits: self.stream_waits.saturating_sub(before.stream_waits),
        }
    }

    /// Whether phase B was issued for every phase-A copy. The missing-wait gate
    /// reads this: a boundary that drops its wait leaves `waits < copies`.
    pub fn all_copies_awaited(self) -> bool {
        self.waits == self.copies
    }
}

/// `MINFER_SYNC_COPIES=1` restores the **pre-F5 synchronous** host round trip for
/// every cross-backend staging copy. It is the reference side of the bitwise A/B
/// (same graph, same input, both modes), and it is how the "before" numbers on
/// the blocking-copy counters are reproduced on the F5 binary itself.
///
/// Read once per process (like the other `MINFER_*` gates); `0` is the default
/// (the async substrate).
static ENV_SYNC: OnceLock<bool> = OnceLock::new();

/// A test's programmatic override of [`sync_copies`], so a gate can flip the
/// mode without touching the process environment (which is shared by every
/// test in the binary). `0` = follow [`ENV_SYNC`], `1` = force async, `2` =
/// force synchronous.
static OVERRIDE: AtomicU8 = AtomicU8::new(0);

/// Whether staging copies should take the pre-F5 synchronous path.
pub fn sync_copies() -> bool {
    match OVERRIDE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => *ENV_SYNC.get_or_init(|| std::env::var("MINFER_SYNC_COPIES").as_deref() == Ok("1")),
    }
}

/// Whether the async substrate is in force (the default).
pub fn async_copies_enabled() -> bool {
    !sync_copies()
}

/// Set the mode programmatically for as long as the returned guard lives.
///
/// The guard restores the previous override even if the test panics, so a
/// failed assertion cannot leave the rest of the binary in the other mode.
#[cfg(test)]
#[must_use = "the guard restores the previous mode on drop"]
pub fn set_sync_for_test(sync: bool) -> SyncModeGuard {
    let prev = OVERRIDE.swap(if sync { 2 } else { 1 }, Ordering::Relaxed);
    SyncModeGuard { prev }
}

/// See [`set_sync_for_test`].
#[cfg(test)]
pub struct SyncModeGuard {
    prev: u8,
}

#[cfg(test)]
impl Drop for SyncModeGuard {
    fn drop(&mut self) {
        OVERRIDE.store(self.prev, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_phases_are_counted_separately_and_the_delta_is_exact() {
        let a = CrossCopyStats {
            copies: 3,
            waits: 3,
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
    /// back leaves the default (async) in force.
    #[test]
    fn the_sync_mode_override_restores_itself() {
        assert!(async_copies_enabled(), "the async substrate is the default");
        {
            let _g = set_sync_for_test(true);
            assert!(sync_copies());
            assert!(!async_copies_enabled());
        }
        assert!(async_copies_enabled());
    }
}
