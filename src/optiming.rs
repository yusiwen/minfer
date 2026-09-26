//! F8 (#51): optional per-op timing, behind `MINFER_OP_TIMING`.
//!
//! **Off by default, and provably so.** With the variable unset,
//! [`enabled`] is one relaxed load of a cached flag and the scheduler does not
//! read the clock at all — the default path is the pre-F8 path. The flag's rule
//! is pinned by `op_timing_flag_is_presence_checked_and_off_when_unset`, the
//! on/off decision it feeds by `off_mode_never_records` /
//! `global_mode_follows_the_flag`, and the scheduler's use of it by
//! `op_timing_does_not_change_the_result_but_does_accumulate`, which also pins
//! that turning it on changes the numbers reported, never the numbers computed.
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
//! **A sink is a destination, not a process global (issue [#173]).** The
//! accumulators live in a [`TimingSink`]; the engine records into the
//! process-global one ([`snapshot`] reads it, so `/metrics` is unchanged), but a
//! caller can own a sink and hand it to its own scheduler through
//! [`TimingMode::Private`]. That is what a gate needs: with one shared table, a
//! graph executed by *any other test thread* while the timing flag was forced on
//! moved the gate's numbers, so its verdict depended on the schedule. Now the
//! gate reads only the rows its own scheduler wrote, and a concurrent execution
//! cannot reach them — pinned by
//! [`a_concurrent_graph_load_cannot_move_a_private_sink`](crate::graph::scheduler::tests).
//!
//! **Storage is a fixed table, not a map.** `op_index` is an exhaustive `match`
//! over [`Op`], so a new variant is a compile error instead of a metric that
//! silently disappears. Accumulation is two relaxed `fetch_add`s on atomics —
//! no lock and no allocation per step, even with the flag on.
//!
//! [#173]: https://github.com/yusiwen/minfer/issues/173

use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
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

/// The per-op accumulators, one instance per destination.
///
/// A fixed table (see the module doc), and an *instance* rather than a static:
/// the engine shares the process-global sink, while a caller — a test, a second
/// engine — can own one and read only its own rows. `const fn new()` so the
/// global needs no lazy per-call initialisation.
pub struct TimingSink {
    nanos: [AtomicU64; OP_COUNT],
    calls: [AtomicU64; OP_COUNT],
}

impl Default for TimingSink {
    fn default() -> Self {
        Self::new()
    }
}

impl TimingSink {
    /// An empty sink. `const`, so a `static` needs no lazy initialisation.
    pub const fn new() -> Self {
        Self {
            nanos: [const { AtomicU64::new(0) }; OP_COUNT],
            calls: [const { AtomicU64::new(0) }; OP_COUNT],
        }
    }

    /// Accumulate one execution of `index`. Called only when timing is on — the
    /// scheduler resolves the sink up front and skips the clock otherwise.
    ///
    /// A CAS loop rather than `fetch_add`, so an absurd duration saturates
    /// instead of wrapping to a small number that reads as "instant".
    /// Contention is nil: one worker thread does the timing.
    pub fn record(&self, index: usize, elapsed: Duration) {
        let nanos = nanos_of(elapsed);
        let _ = self.calls[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_add(1))
        });
        let _ = self.nanos[index].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
            Some(cur.saturating_add(nanos))
        });
    }

    /// Every op with at least one recorded call, in table order (deterministic).
    /// Ops that never ran are omitted, so a family with no samples stays absent
    /// from a scrape instead of reporting a misleading zero.
    pub fn snapshot(&self) -> Vec<OpTimingEntry> {
        OP_NAMES
            .iter()
            .enumerate()
            .filter_map(|(i, name)| {
                let calls = self.calls[i].load(Ordering::Relaxed);
                (calls > 0).then(|| OpTimingEntry {
                    name,
                    calls,
                    nanos: self.nanos[i].load(Ordering::Relaxed),
                })
            })
            .collect()
    }

    /// Zero the sink. Test-only: timing is an accumulator and nothing in a
    /// running server has a reason to reset it.
    #[cfg(test)]
    pub fn reset(&self) {
        for i in 0..OP_COUNT {
            self.nanos[i].store(0, Ordering::Relaxed);
            self.calls[i].store(0, Ordering::Relaxed);
        }
    }
}

/// The process-global sink: the destination `/metrics` reads. Behind an
/// `OnceLock<Arc<_>>` so it can be shared with a caller that wants the *shared*
/// destination explicitly (the #173 control arm) without exposing a bare static.
fn global() -> &'static Arc<TimingSink> {
    static GLOBAL_SINK: OnceLock<Arc<TimingSink>> = OnceLock::new();
    GLOBAL_SINK.get_or_init(|| Arc::new(TimingSink::new()))
}

/// The process-global sink, so a caller can hand the shared destination to a
/// scheduler explicitly. The #173 control arm records into it from several
/// threads at once — the load a private-sink gate must be immune to.
#[cfg(test)]
pub fn global_sink() -> Arc<TimingSink> {
    global().clone()
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

/// Where a scheduler's per-op timings go, and whether it times at all.
///
/// Chosen when the scheduler is constructed, so the decision is a value its
/// owner holds rather than a process-wide flag another thread can flip under it.
/// The default is [`TimingMode::Global`], the pre-#173 production behaviour.
/// `Off`/`Private` are constructed by tests and by a caller that wants its own
/// destination, so a non-test build sees only `Global`.
#[derive(Clone, Default)]
#[cfg_attr(not(test), allow(dead_code))]
pub enum TimingMode {
    /// Production: follow `MINFER_OP_TIMING` (once per `execute`) and record
    /// into the process-global sink when it is on.
    #[default]
    Global,
    /// Never read the clock and never record.
    Off,
    /// Record into a caller-owned sink, gated by a caller-owned flag. This is
    /// the isolation seam: a test owns both the destination and the on/off
    /// decision, so its assertions read only the rows its own scheduler wrote.
    Private {
        sink: Arc<TimingSink>,
        enabled: bool,
    },
}

impl TimingMode {
    /// The sink to record into, given an already-resolved global flag and the
    /// global sink. Pure, so the on/off rule is testable without touching a
    /// table or the environment.
    pub(crate) fn resolve_for<'a>(
        &'a self,
        global_flag: bool,
        global: &'a TimingSink,
    ) -> Option<&'a TimingSink> {
        match self {
            TimingMode::Global => global_flag.then_some(global),
            TimingMode::Off => None,
            TimingMode::Private { sink, enabled } => enabled.then(|| sink.as_ref()),
        }
    }

    /// The sink for the next `execute`, or `None` to leave the clock unread.
    /// The process flag is resolved here, once per call — the F8 property.
    pub(crate) fn resolve(&self) -> Option<&TimingSink> {
        self.resolve_for(enabled(), global().as_ref())
    }
}

/// Nanoseconds of a duration, saturating at `u64::MAX`.
///
/// A `Duration` can name more nanoseconds than fit in a `u64` (`u64::MAX` ns is
/// ~584 years), and a wrap would report a huge op as instant. Pure, so the
/// boundary is tested without touching any table.
fn nanos_of(elapsed: Duration) -> u64 {
    elapsed.as_nanos().min(u64::MAX as u128) as u64
}

/// Accumulate one execution of `index` into the process-global sink. The
/// scheduler goes through a resolved [`TimingSink`] so a private sink is
/// possible; this free function is the module's unchanged entry point for
/// anything that just wants the shared table.
#[cfg_attr(not(test), allow(dead_code))]
pub fn record(index: usize, elapsed: Duration) {
    global().record(index, elapsed);
}

/// Every op with at least one recorded call in the process-global sink.
pub fn snapshot() -> Vec<OpTimingEntry> {
    global().snapshot()
}

/// Zero the process-global sink. Test-only.
#[cfg(test)]
pub fn reset() {
    global().reset();
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

    /// A fresh sink per test, so nothing here shares a destination with any
    /// other test — the fault #173 fixes.
    fn sink() -> TimingSink {
        TimingSink::new()
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

    /// The mode-to-sink rule, asserted **as a value**: `Off` never records;
    /// `Global` follows the flag and lands in the global sink; `Private` follows
    /// its own flag and lands in its own sink. Pure, so "off by default leaves
    /// the table empty" is pinned without touching a table another test could
    /// write.
    #[test]
    fn off_mode_never_records() {
        let global = sink();
        assert!(TimingMode::Off.resolve_for(true, &global).is_none());
        assert!(TimingMode::Off.resolve_for(false, &global).is_none());
    }

    #[test]
    fn global_mode_follows_the_flag() {
        let global = sink();
        assert!(TimingMode::Global.resolve_for(false, &global).is_none());
        assert!(std::ptr::eq(
            TimingMode::Global.resolve_for(true, &global).unwrap(),
            &global
        ));
    }

    #[test]
    fn private_mode_uses_its_own_sink_and_flag() {
        let global = sink();
        let mine = Arc::new(sink());
        let off = TimingMode::Private {
            sink: mine.clone(),
            enabled: false,
        };
        let on = TimingMode::Private {
            sink: mine.clone(),
            enabled: true,
        };
        assert!(off.resolve_for(true, &global).is_none());
        assert!(std::ptr::eq(
            on.resolve_for(false, &global).unwrap(),
            mine.as_ref()
        ));
    }

    /// Exact values from a sink only this test wrote: no reset, no delta, and no
    /// other thread can move them (issue #173).
    #[test]
    fn record_accumulates_per_op_without_cross_talk() {
        let s = sink();
        let matmul = Op::MatMul { transpose_b: false };
        let attn = Op::Attn {
            mode: crate::graph::ops::AttnMode::Gqa,
            explicit_span: false,
        };
        s.record(op_index(&matmul), Duration::from_nanos(100));
        s.record(op_index(&matmul), Duration::from_nanos(250));
        s.record(op_index(&attn), Duration::from_nanos(7));

        assert_eq!(
            s.snapshot(),
            vec![
                OpTimingEntry {
                    name: "matmul",
                    calls: 2,
                    nanos: 350,
                },
                OpTimingEntry {
                    name: "attn",
                    calls: 1,
                    nanos: 7,
                },
            ],
            "table order, only the ops that ran"
        );
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
        let s = sink();
        s.record(op_index(&Op::Add), Duration::from_secs(u64::MAX / 2));
        let snap = s.snapshot();
        assert_eq!(snap.len(), 1);
        assert_eq!(snap[0].name, "add");
        assert_eq!(snap[0].calls, 1);
        assert_eq!(snap[0].nanos, u64::MAX);
    }

    /// The whole point of the fixed table: an op that never ran contributes no
    /// family, and a scrape never sees a half-written entry.
    #[test]
    fn snapshot_omits_ops_that_never_ran() {
        let s = sink();
        s.record(op_index(&Op::Silu), Duration::from_nanos(5));
        assert_eq!(s.snapshot().len(), 1);
        assert!(!s.snapshot().iter().any(|e| e.name == "batch_matmul"));
    }
}
