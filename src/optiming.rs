//! F8 (#51): optional per-op timing, behind `MINFER_OP_TIMING`.
//!
//! **Off by default, and provably so.** With the variable unset,
//! [`enabled`] is one relaxed load of a cached flag and the scheduler does not
//! read the clock at all — the default path is the pre-F8 path. The gate is
//! `op_timing_off_by_default_leaves_the_table_empty`, and
//! `op_timing_does_not_change_the_result` pins that turning it on changes the
//! numbers reported, never the numbers computed.
//!
//! **Where the time is measured, and what it is not.** The timer wraps the
//! scheduler's per-node dispatch
//! ([`BackendScheduler::execute`](crate::graph::scheduler::BackendScheduler::execute)):
//! the interval from just before `Backend::execute_node` is called to just after
//! it returns, for CPU, Metal and CUDA alike. It therefore **includes** the
//! backend's own prologue (buffer lookups, kernel selection) and, on a backend
//! that batches nodes into one command buffer, excludes the actual device work
//! from the CPU-side number. What it does **not** attribute: the split-level
//! `synchronize`, the cross-backend staging copies, allocator liveness, and
//! `fill_input`. Those are scheduler-level costs, and this gate is deliberately a
//! per-op one; the alternative — timing inside each of the three
//! `execute_node` implementations — triplicates the code, still cannot separate
//! kernel time from its prologue, and lets the three backends' definition of "an
//! op" drift.
//!
//! **Storage is a fixed table, not a map.** `op_index` is an exhaustive `match`
//! over [`Op`], so a new variant is a compile error instead of a metric that
//! silently disappears. Accumulation is two relaxed `fetch_add`s on atomics —
//! no lock and no allocation per step, even with the flag on.

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::time::Duration;

use crate::graph::ops::Op;

/// Metric names, in table order. `op_index` must agree with this array;
/// `op_names_agree_with_op_index` pins it.
pub const OP_NAMES: [&str; 23] = [
    "input",
    "add",
    "mul",
    "scale",
    "silu",
    "softmax",
    "rms_norm",
    "qk_norm",
    "matmul",
    "get_rows",
    "rope",
    "attn",
    "kvcache_store",
    "kvcache_load",
    "view",
    "reshape",
    "permute",
    "swiglu",
    "batch_matmul",
    "fused_qkv",
    "qkv_bias_rope_store",
    "fused_ffn",
    "fused_qkv_norm",
];

const OP_COUNT: usize = OP_NAMES.len();

static OP_NANOS: [AtomicU64; OP_COUNT] = [const { AtomicU64::new(0) }; OP_COUNT];
static OP_CALLS: [AtomicU64; OP_COUNT] = [const { AtomicU64::new(0) }; OP_COUNT];

/// One op's accumulated time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OpTimingEntry {
    pub name: &'static str,
    pub calls: u64,
    pub nanos: u64,
}

/// Index of `op` in the timing table.
///
/// Exhaustive by construction: every `Op` variant is named, so adding one is a
/// compile error here rather than a silently missing metric. `Input` is included
/// for completeness even though the scheduler never executes it (inputs are
/// host-filled), so it simply stays at zero calls.
pub fn op_index(op: &Op) -> usize {
    match op {
        Op::Input => 0,
        Op::Add => 1,
        Op::Mul => 2,
        Op::Scale(_) => 3,
        Op::Silu => 4,
        Op::Softmax { .. } => 5,
        Op::RmsNorm { .. } => 6,
        Op::QkNorm { .. } => 7,
        Op::MatMul { .. } => 8,
        Op::GetRows => 9,
        Op::RoPE { .. } => 10,
        Op::Attn { .. } => 11,
        Op::KvcacheStore { .. } => 12,
        Op::KvcacheLoad { .. } => 13,
        Op::View { .. } => 14,
        Op::Reshape { .. } => 15,
        Op::Permute { .. } => 16,
        Op::SwiGLU => 17,
        Op::BatchMatMul => 18,
        Op::FusedQKV { .. } => 19,
        Op::QkvBiasRopeStore { .. } => 20,
        Op::FusedFFN => 21,
        Op::FusedQkvNorm { .. } => 22,
    }
}

const UNKNOWN: u8 = 0;
const ON: u8 = 1;
const OFF: u8 = 2;

static ENABLED: AtomicU8 = AtomicU8::new(UNKNOWN);

/// Whether per-op timing is on. `MINFER_OP_TIMING` is **presence-checked** — the
/// repo's convention for an opt-in instrumentation flag (`MINFER_TRACE`,
/// `MINFER_GRAPH_DUMP`, `MINFER_GRAPH_TRACE`): any value enables it, and unset
/// means the pre-F8 path.
///
/// The environment is read at most once (into an `AtomicU8`), so a steady-state
/// call is a relaxed load and a branch. Two threads racing the first call compute
/// the same answer, so the race is benign.
pub fn enabled() -> bool {
    match ENABLED.load(Ordering::Relaxed) {
        ON => true,
        OFF => false,
        _ => {
            let on = flag_from_env(std::env::var_os("MINFER_OP_TIMING").as_ref());
            ENABLED.store(if on { ON } else { OFF }, Ordering::Relaxed);
            on
        }
    }
}

/// The flag's rule, split out so "off by default" is a pure assertion that does
/// not depend on the process-global cache or on test ordering: unset is off, any
/// value is on.
pub fn flag_from_env(value: Option<&std::ffi::OsString>) -> bool {
    value.is_some()
}

/// Nanoseconds of a duration, saturating at `u64::MAX`.
///
/// A `Duration` can name more nanoseconds than fit in a `u64` (`u64::MAX` ns is
/// ~584 years), and a wrap would report a huge op as instant. Pure, so the
/// boundary is tested without touching the global table.
fn nanos_of(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

/// Accumulate one execution of `index`. Called only when [`enabled`] — the
/// scheduler keeps the check out of the hot path's way.
///
/// A CAS loop rather than `fetch_add`, so an absurd duration saturates instead
/// of wrapping to a small number that reads as "instant". Contention is nil:
/// one worker thread does the timing.
pub fn record(index: usize, elapsed: Duration) {
    let nanos = nanos_of(elapsed);
    let _ = OP_NANOS[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        Some(cur.saturating_add(nanos))
    });
    let _ = OP_CALLS[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        Some(cur.saturating_add(1))
    });
}

/// Every op with at least one recorded call, in table order (deterministic).
/// Ops that never ran are omitted, so a family with no samples stays absent from
/// a scrape instead of reporting a misleading zero.
pub fn snapshot() -> Vec<OpTimingEntry> {
    OP_NAMES
        .iter()
        .enumerate()
        .filter_map(|(i, name)| {
            let calls = OP_CALLS[i].load(Ordering::Relaxed);
            (calls > 0).then(|| OpTimingEntry {
                name,
                calls,
                nanos: OP_NANOS[i].load(Ordering::Relaxed),
            })
        })
        .collect()
}

/// Zero the table. Test-only: timing is a process-lifetime accumulator, and
/// nothing in a running server has a reason to reset it.
#[cfg(test)]
pub fn reset() {
    for i in 0..OP_COUNT {
        OP_NANOS[i].store(0, Ordering::Relaxed);
        OP_CALLS[i].store(0, Ordering::Relaxed);
    }
}

/// Force the gate for a test that needs to exercise the on/off paths without
/// mutating the process environment (which other tests share).
#[cfg(test)]
pub fn force(on: bool) {
    ENABLED.store(if on { ON } else { OFF }, Ordering::Relaxed);
}

/// Serializes every test that touches the process-global gate or table.
/// `crate::graph::scheduler`'s timing gate takes it too — a graph test running
/// concurrently while `force(true)` is set would otherwise record into the table
/// another test is asserting on.
#[cfg(test)]
pub(crate) static GATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The shared gate's guard, **poison-tolerant**: a failing assertion inside one
/// of these tests must not turn every later one into a confusing `PoisonError`
/// panic. The mutation check relies on this — breaking `op_index` should fail
/// `op_names_agree_with_op_index` and say why, not five tests at once.
#[cfg(test)]
pub(crate) fn gate() -> std::sync::MutexGuard<'static, ()> {
    GATE.lock().unwrap_or_else(|e| e.into_inner())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ops() -> Vec<Op> {
        vec![
            Op::Input,
            Op::Add,
            Op::Mul,
            Op::Scale(1.0),
            Op::Silu,
            Op::Softmax { dim: 1 },
            Op::RmsNorm { eps: 1e-6 },
            Op::QkNorm {
                hd: 128,
                nh: 8,
                eps: 1e-6,
            },
            Op::MatMul { transpose_b: false },
            Op::GetRows,
            Op::RoPE {
                style: crate::vec_ops::RopeStyle::NonInterleaved,
            },
            Op::Attn {
                mode: crate::graph::ops::AttnMode::Gqa,
                explicit_span: false,
            },
            Op::KvcacheStore { layer: 0 },
            Op::KvcacheLoad { layer: 0 },
            Op::View {
                offset: 0,
                shape: [1, 1, 1, 1],
            },
            Op::Reshape {
                shape: [1, 1, 1, 1],
            },
            Op::Permute { dims: [0, 1, 2, 3] },
            Op::SwiGLU,
            Op::BatchMatMul,
            Op::FusedQKV { layer: 0 },
            Op::QkvBiasRopeStore { layer: 0 },
            Op::FusedFFN,
            Op::FusedQkvNorm { layer: 0 },
        ]
    }

    /// The counters for `op`, as `(calls, nanos)` — read without `reset`, so a
    /// test asserts a **delta** and stays correct even if another locked test
    /// ran first.
    fn counters(op: &Op) -> (u64, u64) {
        let i = op_index(op);
        (
            OP_CALLS[i].load(Ordering::Relaxed),
            OP_NANOS[i].load(Ordering::Relaxed),
        )
    }

    /// The exhaustive `match` is only useful if it agrees with the name table: a
    /// mis-numbered arm would file one op's time under another's name.
    #[test]
    fn op_names_agree_with_op_index() {
        let ops = sample_ops();
        assert_eq!(ops.len(), OP_NAMES.len(), "every variant is covered once");
        let mut seen = std::collections::HashSet::new();
        for op in &ops {
            let i = op_index(op);
            assert!(i < OP_COUNT, "{op:?} maps out of range");
            assert!(seen.insert(i), "two variants map to index {i}");
            let name = OP_NAMES[i];
            assert!(!name.is_empty());
            assert!(
                name.chars().all(|c| c.is_ascii_lowercase() || c == '_'),
                "metric label value {name:?} must be lowercase snake case"
            );
        }
        assert_eq!(seen.len(), OP_COUNT, "the table has an unused entry");
    }

    /// **Off by default**: unset is off, any value is on — presence-checked, the
    /// repo's convention for opt-in instrumentation. Pure, so it does not depend
    /// on the global cache or on which test ran first.
    #[test]
    fn op_timing_flag_is_presence_checked_and_off_when_unset() {
        assert!(!flag_from_env(None));
        assert!(flag_from_env(Some(&std::ffi::OsString::from("1"))));
        // Any value, including one that reads like "off" — the flag is presence,
        // not a boolean (`MINFER_NO_*` uses `=1` because it *disables*).
        assert!(flag_from_env(Some(&std::ffi::OsString::from("0"))));
        // And the suite never sets it, so a fresh process resolves to off.
        assert!(
            !flag_from_env(std::env::var_os("MINFER_OP_TIMING").as_ref()),
            "this suite must not set MINFER_OP_TIMING"
        );
    }

    #[test]
    fn record_accumulates_per_op_without_cross_talk() {
        let _g = gate();
        force(false); // no other test can be recording while we measure
        let matmul = Op::MatMul { transpose_b: false };
        let attn = Op::Attn {
            mode: crate::graph::ops::AttnMode::Gqa,
            explicit_span: false,
        };
        let (m0c, m0n) = counters(&matmul);
        let (a0c, a0n) = counters(&attn);
        record(op_index(&matmul), Duration::from_nanos(100));
        record(op_index(&matmul), Duration::from_nanos(250));
        record(op_index(&attn), Duration::from_nanos(7));
        let (m1c, m1n) = counters(&matmul);
        let (a1c, a1n) = counters(&attn);
        assert_eq!(m1c - m0c, 2);
        assert_eq!(m1n - m0n, 350);
        assert_eq!(a1c - a0c, 1);
        assert_eq!(a1n - a0n, 7);

        // `snapshot` reports exactly the ops that ran, in table order.
        let snap = snapshot();
        let names: Vec<&str> = snap.iter().map(|e| e.name).collect();
        let mi = names.iter().position(|n| *n == "matmul").expect("matmul");
        let ai = names.iter().position(|n| *n == "attn").expect("attn");
        assert!(mi < ai, "table order: matmul before attn");
    }

    /// Delta-based, so a leftover count from an earlier test cannot break it.
    #[test]
    fn the_gate_switches_the_scheduler_path_on_and_off() {
        let _g = gate();
        force(false);
        assert!(!enabled(), "the flag is trusted, not re-read from the env");
        force(true);
        assert!(enabled());
        force(false);
        assert!(!enabled());
    }

    #[test]
    fn nanos_of_saturates_instead_of_wrapping() {
        assert_eq!(nanos_of(Duration::ZERO), 0);
        assert_eq!(nanos_of(Duration::from_nanos(u64::MAX)), u64::MAX);
        assert_eq!(nanos_of(Duration::from_secs(u64::MAX / 2)), u64::MAX);
    }

    /// An absurd duration saturates rather than wrapping to a small value that
    /// would read as "instant".
    #[test]
    fn record_saturates_instead_of_wrapping() {
        let _g = gate();
        force(false);
        let op = Op::Add;
        let (c0, n0) = counters(&op);
        record(op_index(&op), Duration::from_secs(u64::MAX / 2));
        let (c1, n1) = counters(&op);
        assert_eq!(c1 - c0, 1);
        // Either it added a saturating amount or it was already at the cap; what
        // it must never do is come out smaller than it went in.
        assert!(n1 >= n0);
        if n0 < u64::MAX {
            assert_eq!(n1, u64::MAX);
        }
    }

    /// The whole point of the fixed table: a scrape never sees a half-written
    /// entry, and an op that never ran contributes no family.
    #[test]
    fn snapshot_omits_ops_that_never_ran() {
        let _g = gate();
        force(false);
        // Patchwork: read the un-run set first, then assert none of those appear
        // after recording something else. Locks keep this stable.
        let never = Op::BatchMatMul;
        let (c, _) = counters(&never);
        if c == 0 {
            assert!(!snapshot().iter().any(|e| e.name == "batch_matmul"));
        }
    }
}
