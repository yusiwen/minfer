//! `#[cfg(test)] mod tests` for `src/server/metrics.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::optiming::OpTimingEntry;

fn empty_snapshot() -> MetricsSnapshot {
    MetricsSnapshot {
        requests_total: 0,
        requests_rejected_total: 0,
        requests_completed_total: 0,
        in_flight: 0,
        draining: false,
        drain_abandoned: 0,
        queue_depth: 0,
        worker_pending: 0,
        running: 0,
        jobs_dropped_total: 0,
        worker_stalled_total: 0,
        prompt_tokens_total: 0,
        completion_tokens_total: 0,
        completion_tokens_per_second: 0.0,
        kv: KvSnapshot::default(),
        ops: Vec::new(),
    }
}

/// Every family a provider's parser needs: a `# TYPE` line per family name,
/// and no sample without one.
fn assert_well_formed(text: &str) {
    for line in text.lines() {
        assert!(!line.is_empty(), "no blank lines in the exposition format");
    }
    let typed: std::collections::HashSet<&str> = text
        .lines()
        .filter_map(|l| l.strip_prefix("# TYPE "))
        .map(|l| l.split(' ').next().unwrap())
        .collect();
    for line in text.lines() {
        if line.starts_with('#') {
            assert!(
                line.starts_with("# HELP ") || line.starts_with("# TYPE "),
                "unexpected comment line: {line}"
            );
            continue;
        }
        let name = line.split(['{', ' ']).next().unwrap();
        assert!(typed.contains(name), "sample {name} has no # TYPE line");
    }
    // A `# HELP` immediately precedes its `# TYPE`, and carries a non-empty
    // help text (Prometheus rejects a HELP with no text before the newline).
    let lines: Vec<&str> = text.lines().collect();
    for (i, l) in lines.iter().enumerate() {
        if let Some(rest) = l.strip_prefix("# TYPE ") {
            let name = rest.split(' ').next().unwrap();
            assert!(i > 0, "TYPE with no preceding HELP");
            let prefix = format!("# HELP {name} ");
            assert!(
                lines[i - 1].starts_with(&prefix),
                "TYPE {name} is not preceded by its HELP (line {}: {:?})",
                i - 1,
                lines[i - 1]
            );
            assert!(
                lines[i - 1].len() > prefix.len(),
                "HELP text for {name} is empty"
            );
        }
    }
}

#[test]
fn render_is_well_formed_when_nothing_has_happened_yet() {
    let text = render(&empty_snapshot());
    assert_well_formed(&text);
    // The zero state is a valid scrape, not an empty document.
    assert!(text.contains("minfer_requests_total 0\n"));
    assert!(text.contains("minfer_queue_depth 0\n"));
    assert!(text.contains("minfer_draining 0\n"));
    assert!(text.contains("minfer_kv_packed 0\n"));
    // Nothing ran, so no timing family at all.
    assert!(!text.contains("minfer_op_seconds_total"));
}

#[test]
fn render_names_and_units_are_the_documented_ones() {
    let text = render(&empty_snapshot());
    for name in [
        "minfer_requests_total",
        "minfer_requests_completed_total",
        "minfer_requests_rejected_total",
        "minfer_requests_in_flight",
        "minfer_jobs_dropped_total",
        "minfer_worker_stalled_total",
        "minfer_queue_depth",
        "minfer_worker_pending_jobs",
        "minfer_requests_running",
        "minfer_draining",
        "minfer_drain_abandoned_requests",
        "minfer_prompt_tokens_total",
        "minfer_completion_tokens_total",
        "minfer_completion_tokens_per_second",
        "minfer_memory_weights_bytes",
        "minfer_memory_pool_bytes",
        "minfer_memory_live_bytes",
        "minfer_memory_peak_live_bytes",
        "minfer_memory_idle_slots",
        "minfer_memory_reserved_classes",
        "minfer_kv_layers",
        "minfer_kv_rows",
        "minfer_kv_region_bytes",
        "minfer_kv_packed",
        "minfer_kv_reserved_cells",
        "minfer_kv_owned_cells",
        "minfer_kv_shared_cells",
        "minfer_kv_free_cells",
        "minfer_kv_free_runs",
        "minfer_kv_sequences",
        "minfer_kv_defrags_total",
        "minfer_kv_cells_moved_total",
        "minfer_kv_cows_total",
        "minfer_kv_cow_cells_total",
    ] {
        assert!(text.contains(&format!("# TYPE {name} ")), "missing {name}");
    }
}

/// An unbounded backend has no budget; emitting `0` would read as "no
/// headroom" and page an operator for nothing.
#[test]
fn an_unbounded_backend_omits_the_budget_family() {
    let mut m = empty_snapshot();
    m.kv.weights_bytes = 100;
    m.kv.pool_bytes = 20;
    let unbounded = render(&m);
    assert!(!unbounded.contains("minfer_memory_budget_bytes"));
    assert!(!unbounded.contains("minfer_memory_headroom_bytes"));

    m.kv.budget_bytes = Some(1024);
    m.kv.headroom_bytes = Some(904);
    let bounded = render(&m);
    assert!(bounded.contains("minfer_memory_budget_bytes 1024\n"));
    assert!(bounded.contains("minfer_memory_headroom_bytes 904\n"));
}

/// Boundary: the queue depth is `accepted - admitted` and never wraps, even if
/// a scrape catches `admitted` ahead of `accepted` (a real race between two
/// independent relaxed counters — the renderer must not print 2^64).
#[test]
fn queue_depth_is_a_saturating_difference() {
    let metrics = ServerMetrics::new();
    metrics.requests_total.store(5, Ordering::Relaxed);
    metrics.jobs_admitted_total.store(2, Ordering::Relaxed);
    assert_eq!(metrics.snapshot().queue_depth, 3);
    metrics.jobs_admitted_total.store(9, Ordering::Relaxed);
    assert_eq!(metrics.snapshot().queue_depth, 0);
}

#[test]
fn draining_and_abandoned_are_visible() {
    let metrics = ServerMetrics::new();
    assert!(render(&metrics.snapshot()).contains("minfer_draining 0\n"));
    metrics.draining.store(true, Ordering::Relaxed);
    metrics.drain_abandoned.store(3, Ordering::Relaxed);
    let text = render(&metrics.snapshot());
    assert!(text.contains("minfer_draining 1\n"));
    assert!(text.contains("minfer_drain_abandoned_requests 3\n"));
}

/// The `KvMetrics` atomics round-trip every field, including the two
/// "unset means unbounded" pairs.
#[test]
fn kv_metrics_round_trip() {
    let kv = KvMetrics::default();
    let full = KvSnapshot {
        weights_bytes: 1,
        pool_bytes: 2,
        live_bytes: 3,
        peak_live_bytes: 4,
        budget_bytes: Some(5),
        headroom_bytes: Some(6),
        idle_slots: 7,
        reserved_classes: 8,
        layers: 9,
        rows: 10,
        region_bytes: 11,
        packed: true,
        reserved_cells: 12,
        shared_cells: 13,
        owned_cells: 14,
        free_cells: 15,
        free_runs: 16,
        sequences: 17,
        defrags: 18,
        cells_moved: 19,
        cows: 20,
        cow_cells: 21,
    };
    kv.publish(&full);
    assert_eq!(kv.snapshot(), full);
    // Publishing again with the budgets unset clears the "has" flags, so a
    // scrape after a backend change cannot report a stale budget.
    let none = KvSnapshot {
        weights_bytes: 1,
        budget_bytes: None,
        headroom_bytes: None,
        ..full
    };
    kv.publish(&none);
    let back = kv.snapshot();
    assert_eq!(back.budget_bytes, None);
    assert_eq!(back.headroom_bytes, None);
    assert_eq!(back.weights_bytes, 1);
    assert!(back.packed);
}

/// Timing samples: the seconds value keeps nanosecond resolution (a float
/// round-trip would lose the low digits) and the `op` label is the breakdown.
#[test]
fn op_timing_family_renders_seconds_with_nanosecond_resolution() {
    let mut m = empty_snapshot();
    m.ops = vec![
        OpTimingEntry {
            name: "matmul",
            calls: 8,
            nanos: 1_500_000_123,
        },
        OpTimingEntry {
            name: "attn",
            calls: 2,
            nanos: 7,
        },
    ];
    let text = render(&m);
    assert_well_formed(&text);
    assert!(text.contains("minfer_op_seconds_total{op=\"matmul\"} 1.500000123\n"));
    assert!(text.contains("minfer_op_calls_total{op=\"matmul\"} 8\n"));
    assert!(text.contains("minfer_op_seconds_total{op=\"attn\"} 0.000000007\n"));
    assert!(text.contains("minfer_op_calls_total{op=\"attn\"} 2\n"));
}

#[test]
fn seconds_is_exact_for_the_boundaries() {
    assert_eq!(seconds(0), "0.000000000");
    assert_eq!(seconds(1), "0.000000001");
    assert_eq!(seconds(999_999_999), "0.999999999");
    assert_eq!(seconds(1_000_000_000), "1.000000000");
    assert_eq!(seconds(u64::MAX), "18446744073.709551615");
}

/// F8 / issue #51 requires `tokens/s`. The counters are monotone and the rate
/// is a **trailing window**, so an idle server decays to 0/s instead of
/// reporting its last burst forever.
#[test]
fn token_counters_and_the_trailing_rate() {
    let metrics = ServerMetrics::new();
    let now = 1_000_000u64;

    // Nothing delivered yet: zero counters, zero rate (a scalar, not NaN).
    assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.completion_tokens_total.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.token_window.rate(now), 0.0);

    // 160 generated tokens inside one second is 160/16 = 10 tokens/s over the
    // 16-second window.
    metrics.token_window.add(160, now);
    assert!((metrics.token_window.rate(now) - 10.0).abs() < 1e-9);
    // The same reading one second later still sees the bucket.
    assert!((metrics.token_window.rate(now + 1) - 10.0).abs() < 1e-9);
    // And 16 seconds later the bucket has aged out.
    assert_eq!(metrics.token_window.rate(now + W as u64), 0.0);

    // `record_tokens` drives both counters and only the completion side of the
    // window.
    metrics.record_tokens(7, 4);
    assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), 7);
    assert_eq!(metrics.completion_tokens_total.load(Ordering::Relaxed), 4);
    assert!(metrics.completion_tokens_per_second() > 0.0);

    // A zero-token completion (an immediate stop) counts the prompt and adds
    // nothing to the window, and a zero prompt is not counted at all.
    let before = metrics.prompt_tokens_total.load(Ordering::Relaxed);
    metrics.record_tokens(0, 0);
    assert_eq!(metrics.prompt_tokens_total.load(Ordering::Relaxed), before);
}

/// Two tokens in the same second land in one bucket; tokens in different
/// seconds land in different buckets; and the total is the sum.
#[test]
fn the_token_window_buckets_by_second() {
    let w = TokenWindow::default();
    w.add(10, 500);
    w.add(6, 500);
    w.add(8, 501);
    assert_eq!(w.tokens[(500 % W)].load(Ordering::Relaxed), 16);
    assert_eq!(w.tokens[(501 % W)].load(Ordering::Relaxed), 8);
    // (16 + 8) / 16
    assert!((w.rate(501) - 1.5).abs() < 1e-9);

    // A bucket is recycled after exactly W seconds: the same slot, a new
    // second, and the old tokens are gone (not added to the new reading).
    w.add(4, 500 + W as u64);
    assert_eq!(w.tokens[(500 % W)].load(Ordering::Relaxed), 4);
    assert!((w.rate(500 + W as u64) - 12.0 / W as f64).abs() < 1e-9);
}

/// The rendered family carries the rate and the counters, so a scrape sees
/// them without a Prometheus server computing anything.
#[test]
fn the_token_families_render() {
    let metrics = ServerMetrics::new();
    metrics.record_tokens(11, 22);
    let text = render(&metrics.snapshot());
    assert_well_formed(&text);
    assert!(text.contains("minfer_prompt_tokens_total 11\n"));
    assert!(text.contains("minfer_completion_tokens_total 22\n"));
    let line = text
        .lines()
        .find(|l| l.starts_with("minfer_completion_tokens_per_second "))
        .expect("rate family");
    let v: f64 = line
        .split(' ')
        .nth(1)
        .unwrap()
        .parse()
        .expect("rate is a float");
    assert!(v > 0.0, "rate must be positive after a completion: {line}");
}

/// A metric moved by both sides shows up in one scrape: the handler's
/// accepted count and the worker's admitted count are independent atomics.
#[test]
fn both_threads_write_into_one_registry() {
    let metrics = ServerMetrics::new();
    // handler side
    metrics.requests_total.fetch_add(3, Ordering::Relaxed);
    metrics.in_flight.fetch_add(3, Ordering::Relaxed);
    // worker side
    metrics.jobs_admitted_total.fetch_add(1, Ordering::Relaxed);
    metrics.publish_kv(&KvSnapshot {
        layers: 24,
        rows: 512,
        packed: true,
        ..KvSnapshot::default()
    });
    let text = render(&metrics.snapshot());
    assert!(text.contains("minfer_requests_total 3\n"));
    assert!(text.contains("minfer_requests_in_flight 3\n"));
    assert!(text.contains("minfer_queue_depth 2\n"));
    assert!(text.contains("minfer_kv_layers 24\n"));
    assert!(text.contains("minfer_kv_rows 512\n"));
    assert!(text.contains("minfer_kv_packed 1\n"));
}

// ────────────────────────────────────────────────────────────────────────────
// #239: items moved out of `metrics.rs` (bucket B of the dead-code census — the
// only test caller lives in this module's subtree).
// ────────────────────────────────────────────────────────────────────────────

impl ServerMetrics {
    /// The current trailing-window rate, without building a whole snapshot.
    ///
    /// Test-only (#239): driven by
    /// `server::metrics::tests::token_counters_and_the_trailing_rate`.
    pub fn completion_tokens_per_second(&self) -> f64 {
        self.token_window.rate(now_secs())
    }
}
