//! `#[cfg(test)] mod tests` for `src/optiming.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
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
