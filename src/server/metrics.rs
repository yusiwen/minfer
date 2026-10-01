//! F8 (#51): the server's observability surface — the Prometheus text `/metrics`
//! endpoint and the shared registry behind it.
//!
//! **Why a shared struct of atomics.** The HTTP side and the worker side are two
//! threads with different reach: the handler owns the job sender and the
//! tokenizer, the worker owns the model, the `GraphCache` and the
//! [`BatchEngine`](super::batch::BatchEngine). A model reading cannot happen on
//! the handler thread (it would need to lock the worker) and the channel backlog
//! is not visible on the worker thread (a `tokio::sync::mpsc::Sender` cannot see
//! how many jobs are in flight). So both sides write into one
//! `Arc<ServerMetrics>` and the renderer reads it. Every field is a *relaxed*
//! atomic: the counters are independent, each is monotone by construction, and a
//! scrape that catches two of them one step apart is exactly what a scrape of a
//! live system is. No lock is taken on any path, and nothing here allocates
//! per step — [`ServerMetrics::publish_kv`] is ~20 relaxed stores.
//!
//! **Rendering is pure.** [`render`] takes a [`MetricsSnapshot`] and returns the
//! text; it touches no global state, takes no lock and reads no model, so it
//! cannot perturb generation. The handler only has to build the snapshot.
//!
//! **Metric names and units.** Every family is `minfer_`-prefixed. Byte counters
//! and gauges are bytes; durations are seconds (Prometheus' base unit, so
//! `rate()` is directly usable); the only label is `op` on the per-op timing
//! family, which is the per-op breakdown itself.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::graph::alloc::GraphAllocator;
use crate::graph::Backend;
use crate::optiming::OpTimingEntry;

/// Prometheus text exposition format 0.0.4 (what the `text/plain` scrape
/// protocol expects; the version parameter is optional but conventional).
pub const CONTENT_TYPE: &str = "text/plain; version=0.0.4; charset=utf-8";

/// Seconds of history `minfer_completion_tokens_per_second` averages over.
///
/// A trailing window rather than a lifetime average, because a lifetime average
/// is not a throughput: a server that has been idle for an hour would still
/// report the burst it served at startup. Sixteen seconds is long enough that a
/// scrape never sees an empty window on a busy server and short enough that an
/// idle one decays to 0/s within one scrape interval.
pub const TOKEN_RATE_WINDOW_SECS: usize = 16;
const W: usize = TOKEN_RATE_WINDOW_SECS;

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// A trailing window of generated tokens, one bucket per second.
///
/// **One writer** (the HTTP handler finishing a response) and **many readers** (a
/// scrape), so no lock is needed: each bucket carries the second it belongs to and
/// [`rate`](Self::rate) ignores buckets older than the window. `add` resets a
/// bucket whose stamp is stale *before* adding — that is what makes an idle server
/// decay to 0/s instead of reporting its last burst forever.
#[derive(Debug)]
struct TokenWindow {
    tokens: [AtomicU64; W],
    /// Epoch second each bucket holds; 0 = never written (a real second is never
    /// 0 for any clock this will run against, and the check also skips it).
    stamp: [AtomicU64; W],
}

impl Default for TokenWindow {
    fn default() -> Self {
        Self {
            tokens: [const { AtomicU64::new(0) }; W],
            stamp: [const { AtomicU64::new(0) }; W],
        }
    }
}

impl TokenWindow {
    fn add(&self, tokens: u64, now: u64) {
        if tokens == 0 {
            return;
        }
        let i = (now as usize) % W;
        if self.stamp[i].load(Ordering::Relaxed) != now {
            // Recycle the bucket. The stamp is written after the zero, so a
            // concurrent scrape can only see a bucket that is empty-but-new (an
            // undercount for one scrape) and never tokens attributed to the wrong
            // second.
            self.tokens[i].store(0, Ordering::Relaxed);
            self.stamp[i].store(now, Ordering::Relaxed);
        }
        self.tokens[i].fetch_add(tokens, Ordering::Relaxed);
    }

    /// Generated tokens per second over the trailing [`TOKEN_RATE_WINDOW_SECS`].
    fn rate(&self, now: u64) -> f64 {
        let mut sum = 0u64;
        for i in 0..W {
            let s = self.stamp[i].load(Ordering::Relaxed);
            if s != 0 && now.saturating_sub(s) < W as u64 {
                sum += self.tokens[i].load(Ordering::Relaxed);
            }
        }
        sum as f64 / W as f64
    }
}

/// A plain snapshot of the allocator's accounting plus the KV arena's shape and
/// C3/C8b counters — the numbers `/metrics` exports, decoupled from the atomics
/// so [`render`] is a pure function that is testable without a server.
///
/// Units: `*_bytes` are bytes, `*_cells`/`*_slots`/`*_classes`/`layers`/`rows`/
/// `sequences` are counts, `packed` is 0/1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct KvSnapshot {
    // E4 `MemoryReport`.
    pub weights_bytes: u64,
    pub pool_bytes: u64,
    pub live_bytes: u64,
    pub peak_live_bytes: u64,
    /// `None` = the backend is unbounded (CPU/Metal unless `set_memory_budget`
    /// set one). The budget/headroom families are then omitted rather than
    /// reported as 0, because 0 would read as "no headroom".
    pub budget_bytes: Option<u64>,
    pub headroom_bytes: Option<u64>,
    /// E4 S3: reserved class-sized buffers idle right now (the reservation
    /// table's depth) and the number of `(backend, class)` reservations.
    pub idle_slots: u64,
    pub reserved_classes: u64,
    // Arena shape (C4 the packing, C3 the counters).
    pub layers: u64,
    pub rows: u64,
    pub region_bytes: u64,
    pub packed: bool,
    pub reserved_cells: u64,
    pub shared_cells: u64,
    pub owned_cells: u64,
    pub free_cells: u64,
    pub free_runs: u64,
    pub sequences: u64,
    pub defrags: u64,
    pub cells_moved: u64,
    pub cows: u64,
    pub cow_cells: u64,
}

/// Read one allocator's occupancy into a snapshot.
///
/// This is the single place the mapping `MemoryReport` + `KvArenaStats` →
/// metric fields lives, so every publisher — the batched engine's one shared
/// arena, the serial path's per-slot arenas — reports the same numbers the same
/// way, and `minfer_memory_headroom_bytes` stays the allocator's own
/// [`MemoryReport::headroom_bytes`](crate::graph::alloc::MemoryReport::headroom_bytes)
/// rather than a second implementation of the same subtraction.
pub fn kv_snapshot_from(alloc: &GraphAllocator, backend: Backend) -> KvSnapshot {
    let r = alloc.memory_report(backend);
    let arena = alloc.kv_arena_stats();
    // `MemoryReport::budget_is_bounded` is the single authority for "is this a real
    // number": an unaccounted device (a failed free-memory query, #122) falls back to
    // an unbounded budget, and publishing `usize::MAX` as a gauge would be a number
    // that was never measured.
    let bounded = r.budget_is_bounded();
    KvSnapshot {
        weights_bytes: r.weights_bytes as u64,
        pool_bytes: r.pool_bytes as u64,
        live_bytes: r.live_bytes as u64,
        peak_live_bytes: r.peak_live_bytes as u64,
        budget_bytes: bounded.then(|| r.budget.unwrap() as u64),
        headroom_bytes: r.headroom_bytes().map(|h| h as u64),
        idle_slots: r.idle_slots as u64,
        reserved_classes: r.reserved_classes as u64,
        layers: alloc.kv_layer_count() as u64,
        rows: alloc.kv_n_ctx() as u64,
        region_bytes: alloc.kv_region_bytes() as u64,
        packed: alloc.kv_is_packed(),
        reserved_cells: arena.reserved_cells as u64,
        shared_cells: arena.shared_cells as u64,
        owned_cells: arena.owned_cells as u64,
        free_cells: arena.free_cells as u64,
        free_runs: arena.free_runs as u64,
        sequences: arena.sequences as u64,
        defrags: arena.defrags,
        cells_moved: arena.cells_moved,
        cows: arena.cows,
        cow_cells: arena.cow_cells,
    }
}

/// The same fields as [`KvSnapshot`], as atomics written by the worker thread and
/// read by the handler thread.
#[derive(Debug, Default)]
pub struct KvMetrics {
    weights_bytes: AtomicU64,
    pool_bytes: AtomicU64,
    live_bytes: AtomicU64,
    peak_live_bytes: AtomicU64,
    budget_bytes: AtomicU64,
    /// Whether `budget_bytes` means anything (see `KvSnapshot::budget_bytes`).
    has_budget: AtomicBool,
    headroom_bytes: AtomicU64,
    has_headroom: AtomicBool,
    idle_slots: AtomicU64,
    reserved_classes: AtomicU64,
    layers: AtomicU64,
    rows: AtomicU64,
    region_bytes: AtomicU64,
    packed: AtomicBool,
    reserved_cells: AtomicU64,
    shared_cells: AtomicU64,
    owned_cells: AtomicU64,
    free_cells: AtomicU64,
    free_runs: AtomicU64,
    sequences: AtomicU64,
    defrags: AtomicU64,
    cells_moved: AtomicU64,
    cows: AtomicU64,
    cow_cells: AtomicU64,
}

impl KvMetrics {
    /// Publish one reading. Relaxed on purpose: the fields are independent
    /// counters, and a scrape is not a transaction.
    pub fn publish(&self, s: &KvSnapshot) {
        let r = Ordering::Relaxed;
        self.weights_bytes.store(s.weights_bytes, r);
        self.pool_bytes.store(s.pool_bytes, r);
        self.live_bytes.store(s.live_bytes, r);
        self.peak_live_bytes.store(s.peak_live_bytes, r);
        match s.budget_bytes {
            Some(b) => {
                self.budget_bytes.store(b, r);
                self.has_budget.store(true, r);
            }
            None => self.has_budget.store(false, r),
        }
        match s.headroom_bytes {
            Some(h) => {
                self.headroom_bytes.store(h, r);
                self.has_headroom.store(true, r);
            }
            None => self.has_headroom.store(false, r),
        }
        self.idle_slots.store(s.idle_slots, r);
        self.reserved_classes.store(s.reserved_classes, r);
        self.layers.store(s.layers, r);
        self.rows.store(s.rows, r);
        self.region_bytes.store(s.region_bytes, r);
        self.packed.store(s.packed, r);
        self.reserved_cells.store(s.reserved_cells, r);
        self.shared_cells.store(s.shared_cells, r);
        self.owned_cells.store(s.owned_cells, r);
        self.free_cells.store(s.free_cells, r);
        self.free_runs.store(s.free_runs, r);
        self.sequences.store(s.sequences, r);
        self.defrags.store(s.defrags, r);
        self.cells_moved.store(s.cells_moved, r);
        self.cows.store(s.cows, r);
        self.cow_cells.store(s.cow_cells, r);
    }

    pub fn snapshot(&self) -> KvSnapshot {
        let r = Ordering::Relaxed;
        KvSnapshot {
            weights_bytes: self.weights_bytes.load(r),
            pool_bytes: self.pool_bytes.load(r),
            live_bytes: self.live_bytes.load(r),
            peak_live_bytes: self.peak_live_bytes.load(r),
            budget_bytes: self.has_budget.load(r).then(|| self.budget_bytes.load(r)),
            headroom_bytes: self
                .has_headroom
                .load(r)
                .then(|| self.headroom_bytes.load(r)),
            idle_slots: self.idle_slots.load(r),
            reserved_classes: self.reserved_classes.load(r),
            layers: self.layers.load(r),
            rows: self.rows.load(r),
            region_bytes: self.region_bytes.load(r),
            packed: self.packed.load(r),
            reserved_cells: self.reserved_cells.load(r),
            shared_cells: self.shared_cells.load(r),
            owned_cells: self.owned_cells.load(r),
            free_cells: self.free_cells.load(r),
            free_runs: self.free_runs.load(r),
            sequences: self.sequences.load(r),
            defrags: self.defrags.load(r),
            cells_moved: self.cells_moved.load(r),
            cows: self.cows.load(r),
            cow_cells: self.cow_cells.load(r),
        }
    }
}

/// The server's shared metrics registry. One instance per server process, shared
/// as `Arc<ServerMetrics>` by the axum router and the worker thread.
#[derive(Debug, Default)]
pub struct ServerMetrics {
    /// HTTP requests accepted: a job was queued for the worker.
    pub requests_total: AtomicU64,
    /// HTTP requests refused *before* queueing (the server is draining, or the
    /// worker is gone). Client errors (`400`) are not counted here — they never
    /// reached the queue.
    pub requests_rejected_total: AtomicU64,
    /// Accepted and not yet finished — queued or running. This is the set a
    /// graceful drain has to wait for, so it is also the drain's observable.
    pub in_flight: AtomicU64,
    /// Responses that finished (body sent, stream closed, or client gone).
    pub requests_completed_total: AtomicU64,
    /// 1 once a shutdown signal was received and new work is refused.
    pub draining: AtomicBool,
    /// Requests still in flight when the drain deadline expired — 0 is a clean
    /// drain, and the value names what was abandoned on a forced one.
    pub drain_abandoned: AtomicU64,

    /// Jobs the worker took off its `pending` deque (admitted or rejected).
    /// `requests_total - jobs_admitted_total` is the queue depth: the channel
    /// backlog plus the worker's own deque, which is the one quantity neither
    /// thread can see alone.
    pub jobs_admitted_total: AtomicU64,
    /// Jobs the worker could not place in a slot and answered with an error.
    pub jobs_dropped_total: AtomicU64,
    /// #196: times `serve_loop`'s counted no-progress bound tripped — the worker
    /// took `STALL_STEP_LIMIT` consecutive steps that left the engine busy with a
    /// frozen `BatchEngine::work_units`, answered every live and queued request
    /// with `500 server_error`, and stopped. Zero is the healthy value on every
    /// engine, because a `tick` that leaves the engine busy has forwarded a row or
    /// committed a token (and so moved the counter). Distinct from
    /// `jobs_dropped_total`: that counts saturation rejections (`503 no idle
    /// slot`), this counts the worker itself giving up.
    pub worker_stalled_total: AtomicU64,
    /// Jobs in the worker's `pending` deque right now (exact, worker-published).
    pub worker_pending: AtomicU64,
    /// Requests occupying an engine slot right now (worker-published).
    pub running: AtomicU64,

    /// Prompt tokens of the requests whose response was produced.
    pub prompt_tokens_total: AtomicU64,
    /// Completion tokens actually delivered to a client.
    pub completion_tokens_total: AtomicU64,
    /// Trailing-window bucket store for `minfer_completion_tokens_per_second`.
    token_window: TokenWindow,

    /// KV / allocator occupancy, republished by the worker after every step.
    pub kv: KvMetrics,
}

/// A plain snapshot of [`ServerMetrics`] — what [`render`] consumes.
#[derive(Debug, Clone, PartialEq)]
pub struct MetricsSnapshot {
    pub requests_total: u64,
    pub requests_rejected_total: u64,
    pub requests_completed_total: u64,
    pub in_flight: u64,
    pub draining: bool,
    pub drain_abandoned: u64,
    /// `requests_total - jobs_admitted_total`, saturating (see the field docs).
    pub queue_depth: u64,
    pub worker_pending: u64,
    pub running: u64,
    pub jobs_dropped_total: u64,
    pub worker_stalled_total: u64,
    pub prompt_tokens_total: u64,
    pub completion_tokens_total: u64,
    /// Generated tokens/s over the trailing [`TOKEN_RATE_WINDOW_SECS`] (f64: it
    /// is a rate, and the "no metric without a `# TYPE`" rule is the only shape
    /// constraint Prometheus puts on it).
    pub completion_tokens_per_second: f64,
    pub kv: KvSnapshot,
    /// Per-op accumulations. Empty unless `MINFER_OP_TIMING` is set — the whole
    /// timing family is absent then, which is how "off by default" is visible in
    /// a scrape.
    pub ops: Vec<OpTimingEntry>,
}

impl ServerMetrics {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record one finished request's token counts, and add its completion tokens
    /// to the throughput window.
    ///
    /// Called by the HTTP side when a response is *produced*, so it covers the
    /// batched and the serial path uniformly and needs no plumbing through the
    /// engine. A client that disconnects before its answer is complete is not
    /// counted: those tokens were never delivered, and counting them would report
    /// throughput no user received.
    pub fn record_tokens(&self, prompt: u64, completion: u64) {
        let r = Ordering::Relaxed;
        if prompt > 0 {
            self.prompt_tokens_total.fetch_add(prompt, r);
        }
        if completion > 0 {
            self.completion_tokens_total.fetch_add(completion, r);
            self.token_window.add(completion, now_secs());
        }
    }

    /// Read every atomic into one snapshot (relaxed; see the module docs).
    pub fn snapshot(&self) -> MetricsSnapshot {
        let r = Ordering::Relaxed;
        let accepted = self.requests_total.load(r);
        let admitted = self.jobs_admitted_total.load(r);
        MetricsSnapshot {
            requests_total: accepted,
            requests_rejected_total: self.requests_rejected_total.load(r),
            requests_completed_total: self.requests_completed_total.load(r),
            in_flight: self.in_flight.load(r),
            draining: self.draining.load(r),
            drain_abandoned: self.drain_abandoned.load(r),
            queue_depth: accepted.saturating_sub(admitted),
            worker_pending: self.worker_pending.load(r),
            running: self.running.load(r),
            jobs_dropped_total: self.jobs_dropped_total.load(r),
            worker_stalled_total: self.worker_stalled_total.load(r),
            prompt_tokens_total: self.prompt_tokens_total.load(r),
            completion_tokens_total: self.completion_tokens_total.load(r),
            completion_tokens_per_second: self.token_window.rate(now_secs()),
            kv: self.kv.snapshot(),
            ops: crate::optiming::snapshot(),
        }
    }

    /// The `/metrics` body.
    pub fn render(&self) -> String {
        render(&self.snapshot())
    }

    /// Publish one KV/allocator reading (called by the worker thread).
    pub fn publish_kv(&self, s: &KvSnapshot) {
        self.kv.publish(s);
    }
}

/// Append one metric family: `# HELP`, `# TYPE`, then the single sample.
fn sample(out: &mut String, name: &str, help: &str, kind: &str, value: impl std::fmt::Display) {
    out.push_str("# HELP ");
    out.push_str(name);
    out.push(' ');
    out.push_str(help);
    out.push_str("\n# TYPE ");
    out.push_str(name);
    out.push(' ');
    out.push_str(kind);
    out.push('\n');
    out.push_str(name);
    out.push(' ');
    let _ = std::fmt::Write::write_fmt(out, format_args!("{value}"));
    out.push('\n');
}

/// Exact `nanoseconds` as a Prometheus seconds value (`<sec>.<nanos:09>`), so the
/// sample keeps nanosecond resolution instead of rounding through an `f64`.
fn seconds(nanos: u64) -> String {
    format!("{}.{:09}", nanos / 1_000_000_000, nanos % 1_000_000_000)
}

/// Render a snapshot as Prometheus text exposition format 0.0.4.
///
/// Pure: no global state, no lock, no model. The order is fixed so the output is
/// deterministic and diffable.
pub fn render(m: &MetricsSnapshot) -> String {
    let mut out = String::with_capacity(2048);
    // --- request lifecycle ---
    sample(
        &mut out,
        "minfer_requests_total",
        "HTTP requests accepted and queued for the worker.",
        "counter",
        m.requests_total,
    );
    sample(
        &mut out,
        "minfer_requests_completed_total",
        "HTTP responses that finished (body sent, stream closed, or client gone).",
        "counter",
        m.requests_completed_total,
    );
    sample(
        &mut out,
        "minfer_requests_rejected_total",
        "HTTP requests refused before queueing (server draining, or the worker gone).",
        "counter",
        m.requests_rejected_total,
    );
    sample(
        &mut out,
        "minfer_requests_in_flight",
        "HTTP requests accepted and not yet finished, queued or running (the drain surface).",
        "gauge",
        m.in_flight,
    );
    sample(
        &mut out,
        "minfer_jobs_dropped_total",
        "Jobs the worker could not place in a slot and answered with an error.",
        "counter",
        m.jobs_dropped_total,
    );
    sample(
        &mut out,
        "minfer_worker_stalled_total",
        "Times the worker's counted no-progress bound tripped: consecutive steps left the engine busy without advancing its work counter, so every live and queued request was answered 500 and the worker stopped.",
        "counter",
        m.worker_stalled_total,
    );
    sample(
        &mut out,
        "minfer_queue_depth",
        "Jobs accepted by the HTTP side and not yet taken by the worker: the channel backlog plus the worker's pending deque.",
        "gauge",
        m.queue_depth,
    );
    sample(
        &mut out,
        "minfer_worker_pending_jobs",
        "Jobs sitting in the worker's pending deque right now.",
        "gauge",
        m.worker_pending,
    );
    sample(
        &mut out,
        "minfer_requests_running",
        "Requests currently occupying an engine slot.",
        "gauge",
        m.running,
    );
    // --- throughput ---
    sample(
        &mut out,
        "minfer_prompt_tokens_total",
        "Prompt tokens of the requests whose response was produced.",
        "counter",
        m.prompt_tokens_total,
    );
    sample(
        &mut out,
        "minfer_completion_tokens_total",
        "Completion tokens delivered to a client.",
        "counter",
        m.completion_tokens_total,
    );
    sample(
        &mut out,
        "minfer_completion_tokens_per_second",
        "Generated tokens per second over a trailing 16-second window (a lifetime average is not a throughput).",
        "gauge",
        format!("{:.3}", m.completion_tokens_per_second),
    );
    // --- drain ---
    sample(
        &mut out,
        "minfer_draining",
        "1 once a shutdown signal was received and the server stopped accepting work.",
        "gauge",
        u8::from(m.draining),
    );
    sample(
        &mut out,
        "minfer_drain_abandoned_requests",
        "Requests still in flight when the drain deadline expired (0 = a clean drain).",
        "gauge",
        m.drain_abandoned,
    );
    // --- allocator accounting (E4 MemoryReport), for the server's backend ---
    sample(
        &mut out,
        "minfer_memory_weights_bytes",
        "Model weights resident on the server's backend.",
        "gauge",
        m.kv.weights_bytes,
    );
    sample(
        &mut out,
        "minfer_memory_pool_bytes",
        "Activation/arena bytes the allocator's pools hold on the server's backend.",
        "gauge",
        m.kv.pool_bytes,
    );
    sample(
        &mut out,
        "minfer_memory_live_bytes",
        "Allocator bytes live in the current graph step.",
        "gauge",
        m.kv.live_bytes,
    );
    sample(
        &mut out,
        "minfer_memory_peak_live_bytes",
        "High-water mark of live allocator bytes.",
        "gauge",
        m.kv.peak_live_bytes,
    );
    // An unbounded backend reports no budget rather than 0, which would read as
    // "no headroom" and page an operator for nothing.
    if let Some(budget) = m.kv.budget_bytes {
        sample(
            &mut out,
            "minfer_memory_budget_bytes",
            "Memory budget the allocator checks allocations against.",
            "gauge",
            budget,
        );
    }
    if let Some(headroom) = m.kv.headroom_bytes {
        sample(
            &mut out,
            "minfer_memory_headroom_bytes",
            "Bytes the budget would still accept (budget - weights - pool).",
            "gauge",
            headroom,
        );
    }
    sample(
        &mut out,
        "minfer_memory_idle_slots",
        "Idle class-sized pool buffers in the E4 S3 reservation table (its depth).",
        "gauge",
        m.kv.idle_slots,
    );
    sample(
        &mut out,
        "minfer_memory_reserved_classes",
        "Distinct (backend, size class) reservations the pool holds.",
        "gauge",
        m.kv.reserved_classes,
    );
    // --- KV arena shape + C3/C8b counters ---
    sample(
        &mut out,
        "minfer_kv_layers",
        "Layers with an allocated KV arena.",
        "gauge",
        m.kv.layers,
    );
    sample(
        &mut out,
        "minfer_kv_rows",
        "Rows (cells) the KV arena holds per layer, i.e. the arena's n_ctx.",
        "gauge",
        m.kv.rows,
    );
    sample(
        &mut out,
        "minfer_kv_region_bytes",
        "Bytes the persistent KV regions occupy, summed over both regions of every layer.",
        "gauge",
        m.kv.region_bytes,
    );
    sample(
        &mut out,
        "minfer_kv_packed",
        "1 when the KV regions store packed (Q8_0) cells, 0 for f32/f16 words.",
        "gauge",
        u8::from(m.kv.packed),
    );
    sample(
        &mut out,
        "minfer_kv_reserved_cells",
        "KV cells reserved by live sequences.",
        "gauge",
        m.kv.reserved_cells,
    );
    sample(
        &mut out,
        "minfer_kv_owned_cells",
        "KV cells actually written (shared rows counted once).",
        "gauge",
        m.kv.owned_cells,
    );
    sample(
        &mut out,
        "minfer_kv_shared_cells",
        "KV rows a sequence reads in place from another sequence's run (C8b S2).",
        "gauge",
        m.kv.shared_cells,
    );
    sample(
        &mut out,
        "minfer_kv_free_cells",
        "KV cells neither reserved nor owned.",
        "gauge",
        m.kv.free_cells,
    );
    sample(
        &mut out,
        "minfer_kv_free_runs",
        "Maximal free KV runs — the fragmentation a compaction reduces (C3).",
        "gauge",
        m.kv.free_runs,
    );
    sample(
        &mut out,
        "minfer_kv_sequences",
        "Sequences with a KV reservation.",
        "gauge",
        m.kv.sequences,
    );
    sample(
        &mut out,
        "minfer_kv_defrags_total",
        "KV arena compactions run (C3).",
        "counter",
        m.kv.defrags,
    );
    sample(
        &mut out,
        "minfer_kv_cells_moved_total",
        "KV cell rows a compaction copied, summed over layers (C3).",
        "counter",
        m.kv.cells_moved,
    );
    sample(
        &mut out,
        "minfer_kv_cows_total",
        "KV copy-on-write events (a store landed inside a prefix a sequence shared).",
        "counter",
        m.kv.cows,
    );
    sample(
        &mut out,
        "minfer_kv_cow_cells_total",
        "Sequence rows a copy-on-write moved, summed over layers.",
        "counter",
        m.kv.cow_cells,
    );
    // --- per-op timing (present only when MINFER_OP_TIMING is set) ---
    if !m.ops.is_empty() {
        out.push_str(
            "# HELP minfer_op_seconds_total Wall time spent executing each op, at the \
             scheduler's per-node dispatch (includes dispatch, not split syncs or cross-backend copies).\n",
        );
        out.push_str("# TYPE minfer_op_seconds_total counter\n");
        for e in &m.ops {
            out.push_str("minfer_op_seconds_total{op=\"");
            out.push_str(e.name);
            out.push_str("\"} ");
            out.push_str(&seconds(e.nanos));
            out.push('\n');
        }
        out.push_str("# HELP minfer_op_calls_total Executed nodes per op.\n");
        out.push_str("# TYPE minfer_op_calls_total counter\n");
        for e in &m.ops {
            out.push_str("minfer_op_calls_total{op=\"");
            out.push_str(e.name);
            out.push_str("\"} ");
            out.push_str(&e.calls.to_string());
            out.push('\n');
        }
    }
    out
}

#[cfg(test)]
mod tests;
