//! Continuous batching for the server (Phase E / E2).
//!
//! Before E2 `--n-slots N` bought nothing: each `Slot` owned its own
//! `GraphCache` and the worker ran one request to completion, so `N` only carved
//! the context budget into `N` pieces. Here one arena holds every slot as a
//! *reservation* (`KvCache::reserve_seq`, C1/E1/E2), each slot keeps its KV and
//! its `cached_tokens` across requests (so B2's prefix reuse still works), and
//! one `forward_batch` per step carries every ready slot's next token — one
//! weight pass for all of them.
//!
//! Scope of this increment: the plain (non-speculative) decode path. A session
//! with a speculative draft keeps the per-slot caches and the run-to-completion
//! loop, because doc 94/97's identity contract is per request; batching it is a
//! follow-up. Prefill stays per slot (mixing prefill into a decode batch is
//! E3's chunked prefill), and a step's batch is capped at
//! [`MAX_BATCH`] tokens so CUDA rides its batched split-attention path (E1b
//! documents why a query tile must not span two sequences).

use std::collections::VecDeque;
use tokio::sync::mpsc::error::TryRecvError;

use rand::rngs::StdRng;
use rand::SeedableRng;
use tokio::sync::mpsc;

use crate::graph::batch::Batch;
use crate::graph::cache::GraphCache;
use crate::graph::kvcache::SeqId;
use crate::models::{ModelDef, SpecialTokens};
use crate::tokenizer::Tokenizer;

use super::chat::{
    common_prefix_len, guarded_forward_batch, is_stop_token, prefill_span, Job, StreamEvent,
};
use super::types::{ApiError, SamplingParams};

/// What one token did to its request.
enum StepOutcome {
    Continue,
    /// The turn ended; the string is the OpenAI `finish_reason`.
    Finish(&'static str),
}

/// Slots per step. CUDA's batched split-attention path covers `1 < nt <= 16`
/// (E1b); keeping the batch within it avoids the flash-attention path, whose
/// query tile must not span two sequences.
pub const MAX_BATCH: usize = 16;

/// Largest combined prompt (in fed tokens) that one prefill forward carries.
/// Above this the prefills go one at a time, so the graph width does not churn.
pub const MAX_PREFILL_BATCH: usize = 512;

/// E3: the default prefill chunk, in fed tokens — `n_batch` made real.
///
/// A prefill used to be **one** forward over the whole prompt, so activation memory
/// scaled with the prompt and a long prompt blocked every other slot's decode for its
/// whole duration. Chunking bounds the first and interleaves the second, at a measured
/// cost: each chunk re-streams the weights and re-fills the graph (the graph is keyed
/// on `n_tokens`, and E4's allocator work is what makes several live at once). 2048
/// keeps the default identical to the pre-E3 behaviour for every prompt that fits it,
/// which is what makes this safe to land on by default.
pub const DEFAULT_PREFILL_CHUNK: usize = 2048;

const REPEAT_LAST_N: usize = 64;

/// The prefill chunk size for `MINFER_N_BATCH` (`None`/empty → the default).
/// `0` disables chunking — one forward per prefill, the pre-E3 behaviour — and a
/// value that is not a number falls back to the default. Pure, so its matrix is
/// covered without a server.
pub fn prefill_chunk_size(requested: Option<&str>) -> usize {
    match requested {
        None | Some("") => DEFAULT_PREFILL_CHUNK,
        Some(v) => v.trim().parse::<usize>().unwrap_or(DEFAULT_PREFILL_CHUNK),
    }
}

/// Split the fed tokens `[from, total)` into prefill forwards of at most `chunk`
/// tokens, in order (E3).
///
/// `chunk == 0`, or a suffix no longer than one chunk, yields the whole suffix as a
/// single span: that is "chunking off", and it is also what keeps short prompts
/// bit-for-bit on the pre-E3 path. The last span is the remainder, so the final
/// forward carries the tail row whose logits the request samples from.
pub fn prefill_chunks(from: usize, total: usize, chunk: usize) -> Vec<(usize, usize)> {
    if from >= total {
        return Vec::new();
    }
    if chunk == 0 || total - from <= chunk {
        return vec![(from, total)];
    }
    let mut spans = Vec::new();
    let mut at = from;
    while at < total {
        let end = (at + chunk).min(total);
        spans.push((at, end));
        at = end;
    }
    spans
}

/// One slot's persistent state: its reservation, the tokens its rows hold, and
/// the request currently running on it.
struct SlotState {
    seq: SeqId,
    start: usize,
    cap: usize,
    /// B2: the token sequence this slot's KV rows hold (survives requests).
    cached_tokens: Vec<u32>,
    run: Option<Run>,
}

/// One in-flight request's decoding state.
struct Run {
    tx: mpsc::Sender<StreamEvent>,
    params: SamplingParams,
    /// F3 (#48): the request's sampler configuration, resolved once (it owns the
    /// DRY breakers / logit bias) and reused for every decode step.
    cfg: crate::sampler::SamplerConfig,
    /// F3: mirostat's running surprise budget, per request.
    mirostat: crate::sampler::MirostatState,
    /// F2 (#47): the request's grammar automaton, per slot/request.
    grammar_state: Option<crate::grammar::GrammarState>,
    rng: StdRng,
    prev_tokens: Vec<u32>,
    stop_bytes: Vec<Vec<u8>>,
    /// Bytes generated so far and how many were already sent as `Text`.
    full: Vec<u8>,
    emitted: usize,
    completion_tokens: usize,
    /// Rows written in this slot (`cached_tokens.len()` tracks them too).
    current_pos: usize,
    /// Logits of the row at `current_pos - 1`, waiting to be sampled.
    last_logits: Vec<f32>,
    /// A committed token whose row has not been written yet: the next step's
    /// forward carries it (that is what makes the step batched).
    needs_forward: Option<u32>,
    live_on: bool,
}

/// Version of the slot-table JSON that rides inside the KV container's host section
/// (C5 S2). A table this build does not understand is refused, never guessed.
pub const SLOTS_SNAPSHOT_VERSION: u32 = 1;

/// One slot's context in a snapshot: its sequence id, the reservation the rows live in,
/// and the token sequence those rows hold.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotRow {
    pub seq: SeqId,
    pub start: usize,
    pub cap: usize,
    pub cached_tokens: Vec<u32>,
}

/// The slot table a restart resumes (C5 S2 / E2).
///
/// The **in-flight request is deliberately not here**. A `Run` owns the response
/// channel to a client that a restart has already disconnected, plus its RNG and the
/// row's logits; none of that survives a process boundary, and pretending otherwise
/// would mean serving a stream nobody is reading. What survives is the *context*: the
/// reservation and the tokens each slot's KV rows hold, so a request that arrives after
/// the restart with the same prefix is admitted onto the restored rows and prefills only
/// its own delta (B2's cross-request prefix reuse, with the rows coming from disk).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SlotsSnapshot {
    pub n_slots: usize,
    pub n_ctx_total: usize,
    pub slots: Vec<SlotRow>,
}

pub struct BatchEngine {
    cache: GraphCache,
    n_ctx_total: usize,
    slots: Vec<SlotState>,
    special: SpecialTokens,
    /// C8a: prefix rows copied from another slot instead of being prefilled.
    prefix_rows_copied: usize,
    /// E3: prefill forwards carry at most this many fed tokens (`0` = no chunking).
    n_batch: usize,
    /// E3: prefill forwards run so far, and the largest `nt` any of them carried —
    /// the activation bound the ticket demands is an observable, not a claim.
    prefill_forwards: usize,
    prefill_max_nt: usize,
    /// B2/C5 S2: prompt tokens actually **fed** to a prefill (everything a slot did not
    /// reuse from its KV), summed over requests. The snapshot gate asserts on it: a
    /// restored context must feed only the request's own delta.
    prefill_fed: usize,
    /// E3: decode steps run *between* the chunks of a prefill — the interleaving's
    /// observable (0 whenever chunking is off).
    interleaved_ticks: u64,
    /// C5 S2: where the slot snapshot is written after each completed request.
    /// `None` = no snapshot (the default). Opt-in, because a snapshot rewrites the
    /// whole arena — see the cost line the worker prints at startup.
    slots_file: Option<std::path::PathBuf>,
    /// F8: where this engine's forwards run, so `/metrics` reports the occupancy
    /// of the backend the server actually uses rather than the CPU's by default.
    device: crate::models::Device,
    /// F8/#158: a monotone count of the work this engine has actually done — one
    /// unit per row a decode forward wrote, one per token [`Self::advance`]
    /// committed. A `tick` that leaves the engine busy always moves it (either a
    /// forward ran or some slot's `advance` returned `Continue`, and `Continue`
    /// commits exactly one token), which is what lets a real-model gate bound
    /// *work* instead of wall-clock seconds: a slow box runs the same number of
    /// units, only for longer, while a wedged engine stops moving it.
    work_units: u64,
}

/// C7: the cells a request wants reserved — pure, so the growth policy is
/// unit-tested without a model (the same reason `batch_mode` is pure).
///
/// A bounded request asks for its prompt plus its whole `max_tokens`; an
/// unbounded one asks for the prompt plus the slot's current capacity as
/// headroom, which is enough to serve it without claiming the arena for an
/// answer that may stop after a few tokens. The result is clamped to the arena
/// and never falls below the prompt: a prompt that does not fit at all stays the
/// caller's loud `exceed_context` error instead of becoming a smaller number here.
fn wanted_cells_from(nt: usize, max_tokens: i64, slot_cap: usize, n_ctx_total: usize) -> usize {
    let headroom = if max_tokens < 0 {
        slot_cap
    } else {
        max_tokens as usize
    };
    // The bound is exact: a request may use `prompt + max_tokens` cells, because
    // `advance` decides whether another token fits *before* committing it (see the
    // commit-time check there), so no forward is issued for a token that could only
    // be discarded.
    nt.saturating_add(headroom).min(n_ctx_total).max(nt)
}

/// Follow the run moves a KV operation returned (C3/C7): every slot that caches
/// its run's `start` adopts the new position. The engine is the one place that
/// keeps a `start` across calls, so this is where the contract `kv_defrag`
/// documents is discharged.
fn apply_run_moves(slots: &mut [SlotState], moves: &[crate::graph::kvcache::KvMove]) {
    for m in moves {
        if let Some(s) = slots.iter_mut().find(|s| s.seq == m.seq) {
            s.start = m.to;
        }
    }
}

/// Answer a job the engine will not run: send its error on the job's own
/// channel **before** its sender is dropped, and hand the error back so the
/// caller can still count the rejection.
///
/// #121: a `Job` carries the only `mpsc::Sender<StreamEvent>` its HTTP handler
/// listens on. Dropping it silently closes the channel, and the handler reads a
/// closed channel as a *completed* response — HTTP 200 with empty content
/// (non-streaming) or an empty SSE stream followed by `[DONE]` (streaming) —
/// which a client cannot tell from "the model produced nothing". Every path
/// that gives up on a job goes through here.
fn reject(job: Job, e: ApiError) -> ApiError {
    let _ = job.tx.blocking_send(StreamEvent::Err(e.clone()));
    e
}

/// The environment variable behind [`tick_seam`] — #160's gate-mutation seam.
pub const TICK_SEAM_ENV: &str = "MINFER_TEST_TICK";

/// #160: what `MINFER_TEST_TICK` makes [`BatchEngine::tick`] do, so a wedge can
/// be injected into the real step and the work-bounded gates can be shown to
/// fail *fast* (gate contract rule 3).
///
/// The two arms are the two ways a bounded drive can be wrong:
///
/// - [`TickSeam::Wedge`] (`MINFER_TEST_TICK=wedge`) returns from the step
///   without forwarding or committing anything, so the engine stays busy with a
///   frozen `BatchEngine::work_units`. That is the arm the per-step progress
///   assertion in the gates' shared `WorkBound` catches: the step that wedges
///   the engine is the step that panics.
/// - [`TickSeam::Spin`] (`MINFER_TEST_TICK=spin`) advances `work_units` but never
///   completes a run, so the drive keeps "moving" along a path that cannot
///   terminate. The progress assertion is satisfied by construction, and only
///   the step budget (`step_budget`) catches it.
///
/// [`TickSeam::Off`] is every run with the variable unset — production, CI, the
/// default suite and the unmutated real-model set — and the arms are inert then.
/// The variable is read once per process (like `cuda::s4_ab_map_reps`), so a
/// mutation run exports it before the process starts and a gate cannot race it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TickSeam {
    Off,
    Wedge,
    Spin,
}

/// Read [`TICK_SEAM_ENV`] once per process. An unknown value is `Off` — a typo
/// must not wedge the engine by accident.
pub fn tick_seam() -> TickSeam {
    static SEAM: std::sync::OnceLock<TickSeam> = std::sync::OnceLock::new();
    *SEAM.get_or_init(|| match std::env::var(TICK_SEAM_ENV).ok().as_deref() {
        Some("wedge") => TickSeam::Wedge,
        Some("spin") => TickSeam::Spin,
        _ => TickSeam::Off,
    })
}

impl BatchEngine {
    /// Reserve one run per slot in a single arena. The reservations are made
    /// once and kept: a slot that finishes a request keeps its rows and its
    /// `cached_tokens`, which is exactly what B2's cross-request prefix reuse
    /// needs.
    pub fn new(model: &dyn ModelDef, n_slots: usize, n_ctx_total: usize) -> Result<Self, String> {
        let n_slots = n_slots.max(1);
        let cap = n_ctx_total / n_slots;
        if cap == 0 {
            return Err(format!(
                "continuous batching needs at least one row per slot (n_ctx {n_ctx_total} over {n_slots} slots)"
            ));
        }
        let mut cache = GraphCache::new();
        // #153: stamp the engine's KV format **before** any forward. `load_slots`
        // (the server's startup path) reads the session header's element type through
        // the registry's `kv_format` hook, which is the allocator's stamped format —
        // on master the loader's process-wide tag made this true implicitly, and a
        // per-engine format has to be installed by the caller that knows the engine.
        cache.alloc().set_kv_format(model.kv_format());
        cache.alloc().kv_set_capacity(n_ctx_total);
        let mut slots: Vec<SlotState> = Vec::with_capacity(n_slots);
        for i in 0..n_slots {
            // Sequence ids start at 1: 0 is SEQ_MAIN's, and a slot must never
            // look like the classic single-sequence path.
            let seq = 1 + i as SeqId;
            // C3: a reservation that only fits after a compaction returns the
            // runs it moved, and the slots already handed out must follow them.
            // (At startup the runs are exact-fit and packed, so nothing moves;
            // this is the path that stays correct once runs are dynamic.)
            let (slot, moves) = cache
                .alloc()
                .kv_reserve_seq_with_defrag(seq, cap)
                .map_err(|e| format!("slot {i}: {e}"))?;
            apply_run_moves(&mut slots, &moves);
            slots.push(SlotState {
                seq,
                start: slot.start,
                cap,
                cached_tokens: Vec::new(),
                run: None,
            });
        }
        Ok(Self {
            cache,
            n_ctx_total,
            slots,
            special: model.special_tokens(),
            prefix_rows_copied: 0,
            n_batch: DEFAULT_PREFILL_CHUNK,
            prefill_forwards: 0,
            prefill_max_nt: 0,
            prefill_fed: 0,
            interleaved_ticks: 0,
            slots_file: None,
            device: model.device(),
            work_units: 0,
        })
    }

    /// F8: publish this engine's live reading — the allocator's `MemoryReport`
    /// (weights / pool / live / peak / budget / headroom / reservation depth),
    /// the arena shape, the C3/C8b counters, and how many slots are running now.
    ///
    /// Called from `serve_loop` after every step and every admission group. The
    /// cost is a handful of `HashMap` lookups plus ~20 relaxed stores, i.e. far
    /// below one forward, so a live reading does not perturb generation (the same
    /// reason the renderer takes no lock).
    pub fn publish_metrics(&mut self, metrics: &super::metrics::ServerMetrics) {
        // The borrow checker wants the slot count before `cache.alloc_mut()`.
        let running = self.running_slots() as u64;
        let backend = crate::graph::kvsession::backend_of(self.device);
        let snapshot = super::metrics::kv_snapshot_from(self.cache.alloc(), backend);
        metrics.publish_kv(&snapshot);
        metrics
            .running
            .store(running, std::sync::atomic::Ordering::Relaxed);
    }

    /// Slots currently holding a live request.
    pub fn running_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.run.is_some()).count()
    }

    /// The slot table as JSON (the KV container's opaque host section).
    pub fn slots_to_json(&self) -> String {
        let slots: Vec<serde_json::Value> = self
            .slots
            .iter()
            .map(|s| {
                serde_json::json!({
                    "seq": s.seq,
                    "start": s.start,
                    "cap": s.cap,
                    "cached_tokens": s.cached_tokens,
                })
            })
            .collect();
        serde_json::json!({
            "version": SLOTS_SNAPSHOT_VERSION,
            "n_slots": self.slots.len(),
            "n_ctx_total": self.n_ctx_total,
            "slots": slots,
        })
        .to_string()
    }

    /// Parse a slot table. `None` for anything this build does not understand — the
    /// caller refuses the snapshot and says why, rather than resuming half of it.
    pub fn slots_from_json(json: &str) -> Option<SlotsSnapshot> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        if v.get("version")?.as_u64()? != SLOTS_SNAPSHOT_VERSION as u64 {
            return None;
        }
        let slots = v
            .get("slots")?
            .as_array()?
            .iter()
            .map(|s| {
                Some(SlotRow {
                    seq: s.get("seq")?.as_u64()? as SeqId,
                    start: s.get("start")?.as_u64()? as usize,
                    cap: s.get("cap")?.as_u64()? as usize,
                    cached_tokens: s
                        .get("cached_tokens")?
                        .as_array()?
                        .iter()
                        .map(|t| t.as_u64().map(|n| n as u32))
                        .collect::<Option<Vec<u32>>>()?,
                })
            })
            .collect::<Option<Vec<SlotRow>>>()?;
        Some(SlotsSnapshot {
            n_slots: v.get("n_slots")?.as_u64()? as usize,
            n_ctx_total: v.get("n_ctx_total")?.as_u64()? as usize,
            slots,
        })
    }

    /// Write the slot table **and** the arena it describes into one file (C5's
    /// container, with the table as its host section). Returns `(bytes, slots)`.
    ///
    /// The snapshot is taken when it is called: a slot whose request is still running
    /// snapshots the rows it has *written* (`cached_tokens` tracks exactly those), so a
    /// restored server serves a client that resends its conversation.
    pub fn save_slots(&mut self, path: &std::path::Path) -> Result<(u64, usize), String> {
        let host = self.slots_to_json();
        let report = self
            .cache
            .alloc()
            .kv_save_with_host(path, host.as_bytes())?;
        Ok((report.bytes, self.slots.len()))
    }

    /// Restore a snapshot written by [`Self::save_slots`] (C5 S2).
    ///
    /// Refused, loudly and without touching the arena, unless the file describes *this*
    /// run: another `--n-slots`/`--n-ctx`, another model, another KV element type (the
    /// container's own header checks), a slot table whose sequence ids or written extents
    /// disagree with the arena it rode in with, or a version this build does not know.
    pub fn load_slots(
        &mut self,
        path: &std::path::Path,
        model: &dyn crate::models::ModelDef,
    ) -> Result<(usize, u64), String> {
        let expect = crate::graph::kvsession::expect_for(model, self.n_ctx_total);
        let (host, report) = self.cache.alloc().kv_load_with_host(path, &expect)?;
        let json = String::from_utf8_lossy(&host).into_owned();
        let Some(snap) = Self::slots_from_json(&json) else {
            return Err(format!(
                "{} carries no slot table this build understands (version {} expected)",
                path.display(),
                SLOTS_SNAPSHOT_VERSION
            ));
        };
        if snap.n_slots != self.slots.len() {
            return Err(format!(
                "{} was written by a {}-slot server, this one has {} (--n-slots)",
                path.display(),
                snap.n_slots,
                self.slots.len()
            ));
        }
        if snap.n_ctx_total != self.n_ctx_total {
            return Err(format!(
                "{} was written with n_ctx {}, this server has {} (--n-ctx)",
                path.display(),
                snap.n_ctx_total,
                self.n_ctx_total
            ));
        }
        for (i, row) in snap.slots.iter().enumerate() {
            if row.seq != self.slots[i].seq {
                return Err(format!(
                    "{}: slot {i} names sequence {}, this server reserved {}",
                    path.display(),
                    row.seq,
                    self.slots[i].seq
                ));
            }
            let restored = self.cache.alloc().kv_seq_slot(row.seq).ok_or_else(|| {
                format!(
                    "{}: slot {i} names sequence {} and the arena has no such run",
                    path.display(),
                    row.seq
                )
            })?;
            // The mirror may be *shorter* than the rows: a failed request clears it
            // (the rows it wrote are never read again, and the next request rewrites
            // them from `cached_tokens.len()` on). It may never be longer — that would
            // claim rows the arena does not have.
            if row.cached_tokens.len() > restored.written {
                return Err(format!(
                    "{}: slot {i}'s slot table claims {} token(s) but its rows hold only {} \
                     written row(s) — the two halves of the file disagree",
                    path.display(),
                    row.cached_tokens.len(),
                    restored.written
                ));
            }
            if row.cached_tokens.len() > restored.cap {
                return Err(format!(
                    "{}: slot {i} claims {} token(s) in a {}-cell reservation",
                    path.display(),
                    row.cached_tokens.len(),
                    restored.cap
                ));
            }
            // The snapshot is authoritative about where its rows are.
            self.slots[i].start = restored.start;
            self.slots[i].cap = restored.cap;
            self.slots[i].cached_tokens = row.cached_tokens.clone();
        }
        Ok((snap.slots.len(), report.bytes))
    }

    /// Configure (or clear) the slot snapshot file (C5 S2). The worker sets this once,
    /// before the serve loop; every completed request then rewrites it.
    pub fn set_slots_file(&mut self, path: Option<std::path::PathBuf>) {
        self.slots_file = path;
    }

    /// Bytes a snapshot of this arena takes (the cost of one save), for the startup line.
    pub fn snapshot_bytes(&mut self) -> usize {
        self.cache.alloc().kv_region_bytes()
    }

    /// Write the snapshot when one is configured (C5 S2). Called when a request
    /// **completes**: that is when the slot's context is stable, and it is also the
    /// last moment the rows are known-good, so a server killed without warning still
    /// resumes the conversations that had finished. The request that was in flight is
    /// the one thing a snapshot cannot carry (its response stream belongs to a client a
    /// restart has disconnected) and it is simply re-sent by that client.
    fn save_slots_if_configured(&mut self) {
        let Some(path) = self.slots_file.clone() else {
            return;
        };
        match self.save_slots(&path) {
            Ok((bytes, slots)) => eprintln!(
                "[server] slot snapshot: {slots} slot(s), {bytes} byte(s) → {}",
                path.display()
            ),
            Err(e) => eprintln!("[server] slot snapshot failed ({}): {e}", path.display()),
        }
    }

    /// C7: make slot `idx` hold at least `want` cells, by **reclaiming idle
    /// capacity above it** and re-reserving the slot in place.
    ///
    /// Why this shape. C6 makes a cell move free of arithmetic — a moved row
    /// keeps its sequence-relative position, so no other slot's logits change and
    /// nothing re-ropes — which is what makes an elastic partition possible at
    /// all. The reservation still has to be *contiguous per sequence*, and the
    /// arena is packed from cell 0, so growing a slot means the runs above it must
    /// get out of the way. This increment moves them by **releasing the idle ones**
    /// (their only cost is B2's prefix hint, which the slot re-prefills when it is
    /// next used) and then re-reserving this slot's run at its existing `start`,
    /// where its written rows already are. A *busy* run above the slot is never
    /// touched: that case needs an upward row move, which `Backend::copy_cells`
    /// does not implement yet, so it stays a loud refusal (recorded as C7's
    /// follow-up) rather than a silent repartition.
    ///
    /// Failing soft is deliberate: a slot that cannot grow keeps its rows and its
    /// capacity when they can be restored, and otherwise drops to "no reservation"
    /// and re-prefills on its next request. What it never does is keep reading
    /// rows at an offset its run no longer has.
    fn ensure_slot_capacity(&mut self, idx: usize, want: usize) {
        let want = want.min(self.n_ctx_total);
        if want == 0 || self.slots[idx].cap >= want {
            return;
        }
        // A slot whose run was reclaimed earlier (or that never had one) reserves
        // from scratch; first-fit plus C3's compaction decides where it lands.
        if self.slots[idx].cap == 0 {
            let seq = self.slots[idx].seq;
            if let Ok((slot, moves)) = self.cache.alloc().kv_reserve_seq_with_defrag(seq, want) {
                apply_run_moves(&mut self.slots, &moves);
                self.slots[idx].start = slot.start;
                self.slots[idx].cap = slot.cap;
            }
            return;
        }
        let seq = self.slots[idx].seq;
        // Capacity has to come from somewhere. Idle runs contribute theirs (their
        // only cost is B2's prefix hint, which the slot re-prefills); a **busy** run
        // is never touched — C7b moves the rows around it instead.
        let need = want - self.slots[idx].cap;
        let mut reclaimed = 0usize;
        for j in 0..self.slots.len() {
            if j == idx || self.slots[j].run.is_some() || self.slots[j].cap == 0 {
                continue;
            }
            if self.cache.alloc().kv_arena_stats().free_cells >= need {
                break;
            }
            self.cache.alloc().kv_release_seq(self.slots[j].seq);
            self.slots[j].cached_tokens.clear();
            self.slots[j].cap = 0;
            reclaimed += 1;
        }
        let before = self.slots[idx].cap;
        match self.cache.alloc().kv_set_cap_with_defrag(seq, want) {
            Ok((slot, moves)) => {
                apply_run_moves(&mut self.slots, &moves);
                self.slots[idx].start = slot.start;
                self.slots[idx].cap = slot.cap;
                // Say it once per growth: a reclaimed slot's prefix hint is gone and
                // rows may have moved, and silence would make both invisible.
                eprintln!(
                    "[server] slot {idx}: capacity {before} -> {} cells for a request wanting \
                     {want} (released {reclaimed} idle slot(s); {} run(s) moved)",
                    slot.cap,
                    moves.len()
                );
            }
            Err(e) => eprintln!("[server] slot {idx}: cannot grow to {want} cells ({e})"),
        }
    }

    /// C7: the cells a request wants reserved, clamped to the arena. A bounded
    /// request asks for its prompt plus its whole `max_tokens`; an unbounded one
    /// asks for the prompt plus the slot's current capacity as headroom — enough
    /// to serve it without claiming the arena for an answer that may stop early.
    /// A prompt that already fits its slot asks for nothing new, so the common
    /// case does not repartition at all.
    fn wanted_cells(&self, idx: usize, nt: usize, max_tokens: i64) -> usize {
        wanted_cells_from(nt, max_tokens, self.slots[idx].cap, self.n_ctx_total)
    }

    pub fn n_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn busy(&self) -> bool {
        self.slots.iter().any(|s| s.run.is_some())
    }

    /// Admit a group of requests that arrived together: place each one on an
    /// idle slot (reuse-aware) and prefill it, or — when every slot is busy —
    /// answer it `unavailable("no idle slot")` through [`reject`], which is the
    /// #121 semantics: a saturated server rejects **loudly** (503), it does not
    /// queue and it never answers an empty 200.
    ///
    /// Requests are placed first (reuse-aware), then their prefills go through
    /// **one** forward where that pays: prefill is a weight-bound share of a
    /// served request, so four prompts sharing one weight pass cost far less
    /// than four passes. They are only combined when the group fits
    /// [`MAX_PREFILL_BATCH`]; otherwise each is prefilled alone, because a
    /// varying batch width rebuilds the graph (one graph per cache today — E4's
    /// allocator work is what lifts that).
    ///
    /// Returns one result per job, in the order the jobs were given.
    pub fn admit(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        jobs: Vec<Job>,
    ) -> Vec<Result<usize, ApiError>> {
        let mut answers: Vec<Option<Result<usize, ApiError>>> =
            (0..jobs.len()).map(|_| None).collect();
        // Place the jobs: within the group, an already-taken slot is skipped, so
        // two jobs never land on one slot.
        let mut taken: Vec<bool> = self.slots.iter().map(|s| s.run.is_some()).collect();
        let mut placed: Vec<(usize, Job, usize)> = Vec::new(); // (job index, job, slot)
        for (i, job) in jobs.into_iter().enumerate() {
            let mut best: Option<(usize, usize)> = None; // (prefix match, slot)
            for slot in 0..self.slots.len() {
                if taken[slot] {
                    continue;
                }
                let n = common_prefix_len(&self.slots[slot].cached_tokens, &job.input_ids);
                if best.map_or(true, |(bn, _)| n > bn) {
                    best = Some((n, slot));
                }
            }
            match best {
                Some((_, slot)) => {
                    taken[slot] = true;
                    placed.push((i, job, slot));
                }
                None => answers[i] = Some(Err(reject(job, ApiError::unavailable("no idle slot")))),
            }
        }
        let total: usize = placed
            .iter()
            .map(|(_, job, slot)| self.feed_span(*slot, &job.input_ids).0)
            .sum();
        // A group forward is still one forward, so it must respect the chunk too:
        // with `n_batch` below the group cap the requests are prefilled one at a
        // time instead (each of them chunked), which keeps a single chunking path.
        let group_cap = if self.n_batch == 0 {
            MAX_PREFILL_BATCH
        } else {
            self.n_batch.min(MAX_PREFILL_BATCH)
        };
        if placed.len() > 1 && total <= group_cap && self.prefill_batch_ok() {
            match self.prefill_group(model, &placed) {
                Ok(()) => {
                    for (i, _, slot) in &placed {
                        answers[*i] = Some(Ok(*slot));
                    }
                }
                Err(e) => {
                    // A failed group prefill installs nothing (the install loop is
                    // its last, infallible step), so every job in the group is
                    // still here and must be answered, not dropped.
                    for (i, job, _) in placed {
                        answers[i] = Some(Err(reject(job, ApiError::server(e.clone()))));
                    }
                }
            }
        } else {
            for (i, job, slot) in placed {
                answers[i] = Some(self.submit_on(model, tokenizer, slot, job));
            }
        }
        answers
            .into_iter()
            .map(|a| a.expect("every job has an answer"))
            .collect()
    }

    /// Tokens this slot would have to feed for `input_ids` (reuse-aware): the
    /// suffix after the longest prefix its KV already holds.
    fn feed_span(&self, slot: usize, input_ids: &[u32]) -> (usize, usize) {
        let track = !std::env::var("MINFER_NO_PREFIX_REUSE").map_or(false, |v| v == "1");
        let reuse = if track {
            common_prefix_len(&self.slots[slot].cached_tokens, input_ids)
        } else {
            0
        };
        prefill_span(input_ids.len(), reuse)
    }

    /// C8a: the longest prefix this slot may start from, and the slot holding it when
    /// that is not this slot's own rows. `own` is what this slot already has (0 when
    /// prefix reuse is switched off), so disabling reuse disables the copy path with it.
    fn shared_prefix(&self, idx: usize, prompt: &[u32], own: usize) -> (usize, Option<usize>) {
        if own == 0 && std::env::var("MINFER_NO_PREFIX_REUSE").map_or(false, |v| v == "1") {
            return (0, None);
        }
        // One token must always be fed, or the forward has nothing to run and produces
        // no logits to sample from — the same bound the slot's own reuse lives under.
        let usable = prompt.len().saturating_sub(1);
        let mut best = (own.min(usable), None);
        for (j, slot) in self.slots.iter().enumerate() {
            if j == idx || slot.cached_tokens.is_empty() {
                continue;
            }
            let n = common_prefix_len(&slot.cached_tokens, prompt).min(usable);
            if n > best.0 {
                best = (n, Some(j));
            }
        }
        best
    }

    /// C8a: copy `rows` rows from `src`'s run into this slot's and record the tokens they
    /// hold, so the prefill only feeds the suffix. Returns the rows actually reusable —
    /// `own` when the copy fails, because a failed copy must not lose the slot's own
    /// cache, and the request then simply prefills what it cannot reuse.
    fn copy_prefix_from(
        &mut self,
        src: usize,
        idx: usize,
        rows: usize,
        own: usize,
        gathers: bool,
    ) -> usize {
        let (src_seq, dst_seq) = (self.slots[src].seq, self.slots[idx].seq);
        // C8b S2: a device whose kernel gathers a `kv_map` **shares** the
        // donor's rows instead of copying them — one copy of the bytes read by both
        // sequences. All three backends gather it (CPU, CUDA and, since #362,
        // Metal); `MINFER_NO_KV_SHARE` or a non-gathering device keeps C8a's copy.
        let (verb, fail) = if gathers {
            ("shared", "sharing")
        } else {
            ("copied", "copy")
        };
        let done = if gathers {
            self.cache
                .alloc()
                .kv_share_prefix(src_seq, dst_seq, rows)
                .map(|_| ())
        } else {
            self.cache.alloc().kv_copy_prefix(src_seq, dst_seq, rows)
        };
        match done {
            Ok(()) => {
                self.slots[idx].cached_tokens = self.slots[src].cached_tokens[..rows].to_vec();
                self.prefix_rows_copied += rows;
                eprintln!(
                    "[server] slot {idx}: {verb} {rows} prefix row(s) from slot {src} instead of \
                     prefilling them"
                );
                rows
            }
            Err(e) => {
                eprintln!(
                    "[server] slot {idx}: prefix {fail} from slot {src} failed ({e}); prefilling"
                );
                own
            }
        }
    }

    /// C8b S2/S4: whether this device's kernel gathers a `kv_map` — the CPU and CUDA
    /// ones do (since S4), so a prefix another slot already computed is read in place
    /// instead of copied. Metal keeps C8a's copy until G5. The answer comes from
    /// `Device::gathers_attn_map`, the same authority the model's graph builder reads,
    /// so the share and the window layout cannot disagree.
    ///
    /// `MINFER_NO_KV_SHARE=1` (presence-checked) forces C8a's copy where the device
    /// could read in place: the A/B gate for the share itself, and the shape-matched
    /// baseline the real-model S3 gate compares a shared run against.
    fn gathers_kv_map(model: &dyn ModelDef) -> bool {
        !super::kv_share_disabled() && model.device().gathers_attn_map()
    }

    /// C8a/C8b S2: prefix rows shared or copied rather than prefilled (tests assert
    /// that the reuse happened).
    #[cfg(test)]
    pub fn prefix_rows_copied(&self) -> usize {
        self.prefix_rows_copied
    }

    /// C8b S3: copy-on-write events and the rows they moved — the counter that lets
    /// a test assert the gate's mechanism actually ran (a gate that cannot see its
    /// own precondition is not a gate).
    #[cfg(test)]
    pub fn cow_stats(&mut self) -> (u64, u64) {
        let s = self.cache.alloc().kv_arena_stats();
        (s.cows, s.cow_cells)
    }

    /// Whether several sequences may share one prefill forward on the active
    /// backend. CUDA's flash-attention prefill stages a query tile per block and
    /// a tile must not span two sequences (E1b), so with a CUDA device the
    /// prefills stay per request until that port lands; CPU has no such limit.
    fn prefill_batch_ok(&self) -> bool {
        #[cfg(feature = "cuda")]
        if crate::cuda::CudaState::get().is_some() {
            return false;
        }
        true
    }

    /// One forward for several requests' prefills. Each sequence's suffix is
    /// contiguous in the batch, its positions are its index *within its
    /// sequence* (C6 — the KV row comes from the slot's run, not from the
    /// position), and the logits come back one row per sequence, in `placed`
    /// order.
    fn prefill_group(
        &mut self,
        model: &dyn ModelDef,
        placed: &[(usize, Job, usize)],
    ) -> Result<(), String> {
        let mut tokens: Vec<u32> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        let mut seq_ids: Vec<SeqId> = Vec::new();
        let mut feeds: Vec<usize> = Vec::new();
        for (_, job, slot) in placed {
            let (own, _) = self.feed_span(*slot, &job.input_ids);
            let (want_rows, donor) = self.shared_prefix(*slot, &job.input_ids, own);
            let feed_from = match donor {
                Some(src) => {
                    self.copy_prefix_from(src, *slot, want_rows, own, Self::gathers_kv_map(model))
                }
                None => want_rows,
            };
            let nt = job.input_ids.len();
            let want = self.wanted_cells(*slot, nt, job.params.max_tokens);
            self.ensure_slot_capacity(*slot, want);
            let cap = self.slots[*slot].cap;
            if nt > cap {
                return Err(format!(
                    "prompt of {nt} tokens exceeds slot context of {cap}"
                ));
            }
            tokens.extend_from_slice(&job.input_ids[feed_from..]);
            // C6: positions are the token's index within its sequence; the
            // allocator resolves the KV row from the slot's run.
            positions.extend(feed_from..nt);
            seq_ids.extend(std::iter::repeat(self.slots[*slot].seq).take(nt - feed_from));
            feeds.push(feed_from);
        }
        let batch = Batch::new(tokens, positions, seq_ids);
        let live_on = crate::live::enabled();
        if live_on {
            crate::live::begin_phase("prefill");
        }
        let trace = std::env::var("MINFER_BATCH_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let logits = guarded_forward_batch(model, &batch, 1, self.n_ctx_total, &mut self.cache)
            .map_err(|e| e.message)?;
        if trace {
            eprintln!(
                "[batch] batched prefill: {} prompts, {} tokens, {:.0} ms",
                placed.len(),
                batch.len(),
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        let k = placed.len();
        if logits.len() % k != 0 {
            return Err(format!(
                "batched prefill returned {} logits for {k} sequences",
                logits.len()
            ));
        }
        let nv = logits.len() / k;
        if live_on {
            crate::live::attach_step(&logits[..nv.min(logits.len())]);
        }
        for (r, (_, job, slot)) in placed.iter().enumerate() {
            self.install_run(
                *slot,
                job.clone(),
                logits[r * nv..(r + 1) * nv].to_vec(),
                feeds[r],
            );
        }
        Ok(())
    }

    /// E3: set the prefill chunk size (fed tokens per prefill forward; `0` = one
    /// forward per prefill, the pre-E3 behaviour).
    pub fn set_prefill_chunk(&mut self, n_batch: usize) {
        self.n_batch = n_batch;
    }

    /// #158: the engine's monotone work counter (see the field). A gate drives the
    /// engine until it is idle and asserts this moved on every `tick` that left it
    /// busy — a load-proof replacement for an absolute wall-clock deadline, which a
    /// slow machine could exceed while a wedged engine could not.
    pub fn work_units(&self) -> u64 {
        self.work_units
    }

    /// Whether an already admitted request has a token waiting for its decode
    /// forward — what interleaving a prefill is *for*.
    fn has_pending_decode(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.run.as_ref().is_some_and(|r| r.needs_forward.is_some()))
    }

    /// Admit a request on a **specific** slot.
    ///
    /// A slot's cell offset is part of a request's determinism: RoPE at cell
    /// 256 and at cell 5 agree in exact arithmetic but not in `f32`, so the same
    /// prompt on a different slot can differ in the last ulps — and flip an
    /// argmax on a near-tie. A slot is therefore a *session's KV home*, not an
    /// interchangeable resource, which is also what makes B2's cross-request
    /// prefix reuse possible; this entry point is how a caller pins one.
    ///
    /// Every failure that gives up on the request answers it through [`reject`]
    /// before its sender is dropped — the caller (`admit`) turns the returned
    /// `Err` into the F8 drop count, not into the client's answer.
    pub fn submit_on(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        idx: usize,
        job: Job,
    ) -> Result<usize, ApiError> {
        if idx >= self.slots.len() {
            return Err(reject(
                job,
                ApiError::invalid_request(format!(
                    "slot {idx} does not exist ({} slots)",
                    self.slots.len()
                )),
            ));
        }
        if self.slots[idx].run.is_some() {
            return Err(reject(
                job,
                ApiError::unavailable(format!("slot {idx} is busy")),
            ));
        }
        let nt = job.input_ids.len();
        // C7: size the slot from this request instead of leaving the startup
        // partition in place (see `ensure_slot_capacity`).
        let want = self.wanted_cells(idx, nt, job.params.max_tokens);
        self.ensure_slot_capacity(idx, want);
        let cap = self.slots[idx].cap;
        if nt > cap {
            return Err(reject(
                job,
                ApiError::exceed_context(format!(
                    "prompt of {nt} tokens exceeds slot context of {cap}"
                )),
            ));
        }
        let seq = self.slots[idx].seq;
        // C8a: the rows this request can start from may live in *another* slot — the
        // slot that already computed the same prefix (possibly while still generating:
        // its rows are stable for the duration of the copy). The match is verified
        // against the donor's recorded tokens, so the rows are known to hold this
        // prompt's prefix, and C6 makes the donor's different run start irrelevant to
        // this slot's arithmetic — which is why the gate can demand byte equality.
        let (own, _) = self.feed_span(idx, &job.input_ids);
        let (want_rows, donor) = self.shared_prefix(idx, &job.input_ids, own);
        let feed_from = match donor {
            Some(src) => {
                self.copy_prefix_from(src, idx, want_rows, own, Self::gathers_kv_map(model))
            }
            None => want_rows,
        };
        // C6: positions are a token's index *within its sequence* — what RoPE
        // rotates by — so the slot's reservation start is not part of them. The
        // allocator resolves the KV row from the run table (`cells` =
        // `start + position`); adding `start` here would double-count it and
        // `kv_cells_for_seq` rejects the result (a run of `cap` cells at `start`
        // cannot hold position `start + i`).
        // E3: feed the suffix in chunks of at most `n_batch` tokens; `n_batch = 0`
        // keeps the pre-E3 single forward, which is what the equality gate compares
        // against. Every chunk asks for one output row: `n_out` is part of
        // `GraphParams`, so varying it per chunk would rebuild the graph once more
        // for nothing (the lm_head over one row is noise next to the prefill).
        let spans = prefill_chunks(feed_from, nt, self.n_batch);
        let n_spans = spans.len();
        let live_on = crate::live::enabled();
        if live_on {
            crate::live::begin_phase("prefill");
        }
        let trace = std::env::var("MINFER_BATCH_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let mut last_logits: Vec<f32> = Vec::new();
        for (i, (from, to)) in spans.into_iter().enumerate() {
            let batch = Batch::new(
                job.input_ids[from..to].to_vec(),
                (from..to).collect(),
                vec![seq; to - from],
            );
            self.prefill_forwards += 1;
            self.prefill_max_nt = self.prefill_max_nt.max(batch.len());
            let logits =
                match guarded_forward_batch(model, &batch, 1, self.n_ctx_total, &mut self.cache) {
                    Ok(l) => l,
                    // The prefix is not installed yet, so this request has not
                    // been admitted and must be answered, not dropped.
                    Err(e) => return Err(reject(job, e)),
                };
            if i + 1 == n_spans {
                last_logits = logits;
            }
            // E3: a long prompt must not stall the conversations already running.
            // Between chunks — never before the first or after the last — the other
            // slots take their decode step; the chunk's rows are already written, so
            // this is exactly the tick `serve_loop` would have run next. A failure
            // there belongs to *that* request, not this one, so it is logged and the
            // prefill carries on (mirroring `serve_loop`).
            if i + 1 < n_spans && self.has_pending_decode() {
                self.interleaved_ticks += 1;
                if let Err(e) = self.tick(model, tokenizer) {
                    eprintln!("[server] interleaved decode step failed: {}", e.message);
                }
            }
        }
        if trace {
            eprintln!(
                "[batch] single prefill: slot {idx}, {} tokens in {n_spans} forward(s) \
                 (n_batch {}), {:.0} ms",
                nt - feed_from,
                self.n_batch,
                t0.elapsed().as_secs_f64() * 1e3
            );
        }
        if live_on {
            crate::live::attach_step(&last_logits);
        }
        self.install_run(idx, job, last_logits, feed_from);
        Ok(idx)
    }

    /// Put a request into `slot` with the logits of its last prompt row.
    fn install_run(&mut self, idx: usize, job: Job, last_logits: Vec<f32>, feed_from: usize) {
        let nt = job.input_ids.len();
        self.prefill_fed += nt - feed_from;
        let track = !std::env::var("MINFER_NO_PREFIX_REUSE").map_or(false, |v| v == "1");
        if track {
            eprintln!(
                "[server] slot {idx} prefill fed {}/{} prompt tokens ({} reused from its KV)",
                nt - feed_from,
                nt,
                feed_from
            );
        }
        self.slots[idx].cached_tokens.clear();
        if track {
            self.slots[idx]
                .cached_tokens
                .extend_from_slice(&job.input_ids);
        }
        debug_assert_eq!(self.slots[idx].cached_tokens.len(), nt);
        let cfg = job.params.sampler_config();
        let grammar_state = cfg.grammar.as_ref().map(|g| g.state());
        self.slots[idx].run = Some(Run {
            tx: job.tx,
            mirostat: crate::sampler::MirostatState::new(cfg.mirostat_tau),
            grammar_state,
            cfg,
            rng: StdRng::seed_from_u64(job.params.seed),
            prev_tokens: super::chat::sampler_recent_window(&job.input_ids, REPEAT_LAST_N),
            stop_bytes: job
                .params
                .stop_strings
                .iter()
                .map(|s| s.as_bytes().to_vec())
                .collect(),
            params: job.params,
            full: Vec::new(),
            emitted: 0,
            completion_tokens: 0,
            current_pos: nt,
            last_logits,
            needs_forward: None,
            live_on: crate::live::enabled(),
        });
    }

    /// One step: forward every ready slot's pending token in a single batch,
    /// then sample, commit and queue the next token for each of them.
    pub fn tick(&mut self, model: &dyn ModelDef, tokenizer: &Tokenizer) -> Result<(), ApiError> {
        // #160: the gate-mutation seam (see `tick_seam`). `wedge` returns with
        // the engine still busy and the work counter frozen; `spin` advances the
        // counter without ever completing a run. Both fire before any real work,
        // and neither branch is taken when the variable is unset (every
        // production, CI and default run).
        match tick_seam() {
            TickSeam::Off => {}
            TickSeam::Wedge => return Ok(()),
            TickSeam::Spin => {
                self.work_units += 1;
                return Ok(());
            }
        }
        // (1) one forward for every slot with a committed-but-unwritten token.
        let mut tokens: Vec<u32> = Vec::new();
        let mut positions: Vec<usize> = Vec::new();
        let mut seq_ids: Vec<SeqId> = Vec::new();
        let mut rows: Vec<usize> = Vec::new(); // slot index per batch row
        for (i, slot) in self.slots.iter().enumerate() {
            if let Some(run) = &slot.run {
                if let Some(tok) = run.needs_forward {
                    if tokens.len() == MAX_BATCH {
                        break;
                    }
                    tokens.push(tok);
                    positions.push(run.current_pos);
                    seq_ids.push(slot.seq);
                    rows.push(i);
                }
            }
        }
        if !tokens.is_empty() {
            let batch = Batch::new(tokens, positions, seq_ids);
            // The decode trace is what makes E2's acceptance observable: the
            // batched-decode claim ("N sequences in one weight pass") is a
            // property of *this* forward, and the prefill lines say nothing about
            // it. `MINFER_BATCH_TRACE=1`.
            let trace = std::env::var("MINFER_BATCH_TRACE").is_ok();
            let t0 = std::time::Instant::now();
            let logits =
                match guarded_forward_batch(model, &batch, 1, self.n_ctx_total, &mut self.cache) {
                    Ok(l) => l,
                    // #151: the batch is one weight pass, so a failed forward is every
                    // row's failure. Answer each run whose row was in it — through its
                    // own sender, exactly as `fail` does for a sampling error — and
                    // release the slots *before* returning. Returning first (the
                    // pre-fix shape) left every `needs_forward` set, so the next pass
                    // rebuilt this same batch and retried this same forward forever:
                    // a deterministic failure spun the worker at 100% CPU, no client
                    // ever heard, `in_flight` stayed ≥ 1 and a drain ran out its
                    // deadline. `rows` is the membership test — a run whose row was
                    // not in the batch keeps its state.
                    Err(e) => {
                        self.fail_batch(&rows, &e);
                        return Err(e);
                    }
                };
            // #158: this forward wrote one row per batch entry; count the work
            // before the bookkeeping below consumes the vector.
            self.work_units += rows.len() as u64;
            if trace {
                let rows_desc: Vec<String> = rows
                    .iter()
                    .enumerate()
                    .map(|(r, &slot)| {
                        format!(
                            "slot{slot}/seq{}/pos{}",
                            self.slots[slot].seq, batch.positions[r]
                        )
                    })
                    .collect();
                eprintln!("[batch] rows: {}", rows_desc.join(" "));
                eprintln!(
                    "[batch] decode step: {} sequence(s), {} tokens, {:.1} ms ({:.1} ms/token)",
                    rows.len(),
                    batch.len(),
                    t0.elapsed().as_secs_f64() * 1e3,
                    t0.elapsed().as_secs_f64() * 1e3 / batch.len() as f64
                );
            }
            let nv = logits.len() / rows.len();
            debug_assert_eq!(logits.len(), nv * rows.len());
            for (r, &slot_idx) in rows.iter().enumerate() {
                let slot = &mut self.slots[slot_idx];
                let run = slot.run.as_mut().expect("run present");
                // This row now holds the token that was pending: the slot's
                // `cached_tokens` mirror (B2) advances with the write.
                let tok = run.needs_forward.take().expect("batched row had a token");
                let pos = run.current_pos;
                run.last_logits = logits[r * nv..(r + 1) * nv].to_vec();
                let live_on = run.live_on;
                slot.cached_tokens.push(tok);
                // The row this forward wrote is exactly position `pos`, so the
                // slot's mirror now covers `0..=pos` and the next token belongs at
                // `pos + 1`. Advancing here — where the row exists — is what keeps
                // the mirror and the KV layout in step: `advance` used to move it
                // one step early, so the first token after a prefill was written at
                // `nt + 1` and position `nt` was never written at all (a stale row
                // the next step's attention then read).
                debug_assert_eq!(slot.cached_tokens.len(), pos + 1);
                if let Some(run) = slot.run.as_mut() {
                    run.current_pos = pos + 1;
                }
                if live_on {
                    crate::live::attach_step(&slot.run.as_ref().expect("run").last_logits);
                }
            }
        }

        // (2) sample, commit and queue per slot.
        for idx in 0..self.slots.len() {
            if self.slots[idx].run.is_none() {
                continue;
            }
            match self.advance(idx, tokenizer) {
                // #158: a `Continue` committed exactly one token (every other
                // `advance` outcome ends the run and takes it), so this is the
                // second half of the work counter.
                Ok(StepOutcome::Continue) => self.work_units += 1,
                Ok(StepOutcome::Finish(reason)) => self.finish(idx, reason),
                Err(e) => self.fail(idx, e),
            }
        }
        Ok(())
    }

    /// One token of one slot: the server's per-token rules, unchanged —
    /// max_tokens, the slot's context bound, EOG (which ends the turn without
    /// writing a row), stop strings on the full byte stream, complete-UTF-8
    /// emission, and the penalty window.
    fn advance(&mut self, idx: usize, tokenizer: &Tokenizer) -> Result<StepOutcome, ApiError> {
        let special = self.special.clone();
        let cap = self.slots[idx].cap;
        let Run {
            params,
            cfg,
            mirostat,
            grammar_state,
            rng,
            prev_tokens,
            stop_bytes,
            full,
            emitted,
            completion_tokens,
            current_pos,
            last_logits,
            tx,
            needs_forward,
            live_on,
            ..
        } = self.slots[idx].run.as_mut().expect("caller checked");

        if params.max_tokens >= 0 && *completion_tokens as i64 >= params.max_tokens {
            return Ok(StepOutcome::Finish("length"));
        }
        // Safety net: admission guarantees `cap` covers the request, and the
        // commit-time check below stops a generation exactly at its last usable
        // cell, so this fires only if a caller ever admits a wider request than the
        // run it reserved (then it ends the turn instead of writing past the run).
        if *current_pos >= cap {
            return Ok(StepOutcome::Finish("length"));
        }
        let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();
        let sampled = match crate::sampler::sample_with_config_grammar(
            last_logits,
            cfg,
            prev_tokens,
            mirostat,
            grammar_state,
            rng,
        ) {
            Ok(s) => s,
            Err(e) => {
                // The grammar allows nothing more: end the turn with the reason
                // printed. Everything already delivered is a valid prefix.
                eprintln!("[grammar] {e}");
                return Ok(StepOutcome::Finish("stop"));
            }
        };
        let tok = sampled.token_id;
        if is_stop_token(tok, &special) {
            return Ok(StepOutcome::Finish("stop"));
        }
        *completion_tokens += 1;
        prev_tokens.push(tok);
        if prev_tokens.len() > REPEAT_LAST_N {
            prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
        }
        full.extend_from_slice(&tokenizer.decode_bytes(&[tok]));
        if crate::sampler::match_stop_suffix(full, &stop_refs).is_some() {
            // Stop strings are not part of the canonical text: the token is not
            // forwarded, so its row is never written.
            return Ok(StepOutcome::Finish("stop"));
        }
        // Emit only the complete-UTF-8 prefix (hold back a split character).
        let complete = *emitted + crate::tokenizer::complete_utf8_prefix_len(&full[*emitted..]);
        if complete > *emitted {
            let chunk = String::from_utf8_lossy(&full[*emitted..complete]).into_owned();
            if tx.blocking_send(StreamEvent::Text(chunk)).is_err() {
                return Err(ApiError::server("client disconnected"));
            }
            *emitted = complete;
        }
        if *live_on {
            let text = String::from_utf8_lossy(&tokenizer.decode_bytes(&[tok])).into_owned();
            crate::live::set_token(tok, &text);
        }
        // Decide *now* whether another token can still be used, instead of committing
        // one and discovering on the next step that it must be discarded: that wasted
        // a whole weight pass and wrote a cell nothing would ever read (C7 reserved an
        // extra cell to make room for it). `current_pos` is the cell the committed
        // token occupies, so continuing is only legal while the *next* one (`+ 1`)
        // still fits the run — the two bounds are the request's token budget and the
        // run's capacity.
        let budget_done = params.max_tokens >= 0 && *completion_tokens as i64 >= params.max_tokens;
        if budget_done || *current_pos + 1 >= cap {
            return Ok(StepOutcome::Finish("length"));
        }
        // `tok`'s row is written by the NEXT step's batch, at `current_pos` — which
        // this function deliberately does not move: the forward that writes the row
        // is the one that advances the position (`tick`), so a token's position and
        // its row cannot drift apart.
        *needs_forward = Some(tok);
        Ok(StepOutcome::Continue)
    }

    /// A real failure: tell the client and drop the slot's cached-token record,
    /// because a failed request may have written part of a row.
    fn fail(&mut self, idx: usize, e: ApiError) {
        if let Some(run) = self.slots[idx].run.take() {
            let _ = run.tx.blocking_send(StreamEvent::Err(e));
        }
        self.slots[idx].cached_tokens.clear();
    }

    /// #151: answer every run whose row was in a **failed decode batch**.
    ///
    /// `rows` is the slot index per batch row, built by `tick` as it collects the
    /// pending tokens — so it is exactly the affected set, and a run whose
    /// `needs_forward` was unset (already sampled) or that did not fit `MAX_BATCH`
    /// is not in it and is left alone.
    ///
    /// Each run goes through [`Self::fail`], the same primitive `advance`'s error
    /// path uses: one `StreamEvent::Err` through the run's own sender (#121's
    /// per-job shape), the run taken so the slot is free, and `cached_tokens`
    /// cleared because a forward that failed part-way may have written rows the
    /// mirror does not describe — reusing that prefix is the one thing a failure
    /// must not lead to. Clearing the slot is also what stops the retry: with
    /// `needs_forward` gone, the next `tick` builds a different batch (or none),
    /// so the worker cannot spin on the same shape.
    ///
    /// **No retry, deliberately.** Every failure reachable here is deterministic
    /// and request-fatal — a kernel-invariant violation, an E4 activation-budget
    /// refusal, an `ensure_kv` format/width mismatch, or a panic caught by
    /// `guarded_forward_batch` (whose own contract is that the shared arena may be
    /// half-written) — so a retry of the same shape cannot succeed, and reading a
    /// half-written arena to try again is worse than answering. The failure is
    /// answered for **these runs only**: nothing is latched on the server, so a
    /// later request builds a fresh batch; a transient failure would still cost
    /// these particular requests their answer, which is the honest trade against
    /// the old infinite retry. If a transient class ever appears it belongs in the
    /// backend (a retry around the device op), not in a hidden loop here.
    fn fail_batch(&mut self, rows: &[usize], e: &ApiError) {
        if rows.is_empty() {
            return;
        }
        for &idx in rows {
            self.fail(idx, e.clone());
        }
        eprintln!(
            "[server] decode step failed ({}); answered {} run(s) and released their slot(s)",
            e.message,
            rows.len()
        );
    }

    /// #196: answer **every** run still occupying a slot — the `fail_batch` shape
    /// without a row list, for the case where the batch is the whole engine.
    ///
    /// `serve_loop`'s no-progress bound knows the engine is wedged but not which
    /// batch wedged it (a step that returned `Ok` may have built none), so the
    /// affected set is "every slot that still has a run". Each one goes through
    /// [`Self::fail`], the same primitive `advance`'s error path and `fail_batch`
    /// use: exactly one `StreamEvent::Err` through the run's own sender (#121's
    /// per-job shape), the run taken so the slot is free, and `cached_tokens`
    /// cleared because a wedged engine's KV rows cannot be trusted as a prefix.
    /// Returns how many runs it answered, for the log line and the gate.
    fn fail_all(&mut self, e: &ApiError) -> usize {
        let mut answered = 0;
        for idx in 0..self.slots.len() {
            if self.slots[idx].run.is_some() {
                self.fail(idx, e.clone());
                answered += 1;
            }
        }
        answered
    }

    /// Close a request: flush the tail, send `Finish`, and free the slot (its
    /// KV and `cached_tokens` stay, so the next request can reuse the prefix).
    fn finish(&mut self, idx: usize, reason: &str) {
        let Some(mut run) = self.slots[idx].run.take() else {
            return;
        };
        let reason = reason.to_string();
        let stop_bytes = run.stop_bytes.clone();
        let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();
        if crate::sampler::match_stop_suffix(&run.full, &stop_refs).is_some() {
            if let Some(cut) = crate::sampler::match_stop_suffix(&run.full, &stop_refs) {
                run.full.truncate(cut);
            }
        }
        if run.emitted < run.full.len() {
            // A response body must be valid UTF-8. A still-pending partial
            // character (one the generation ended in the middle of) can never
            // complete now, so it is dropped rather than decoded to U+FFFD — the
            // text stays a valid prefix of the grammar's language.
            let complete =
                run.emitted + crate::tokenizer::complete_utf8_prefix_len(&run.full[run.emitted..]);
            if complete > run.emitted {
                let chunk = String::from_utf8_lossy(&run.full[run.emitted..complete]).into_owned();
                if run.tx.blocking_send(StreamEvent::Text(chunk)).is_err() {
                    self.slots[idx].cached_tokens.clear();
                    return;
                }
            }
            if complete < run.full.len() {
                eprintln!(
                    "[server] dropped {} trailing byte(s) of an incomplete UTF-8 character",
                    run.full.len() - complete
                );
            }
        }
        let _ = run.tx.blocking_send(StreamEvent::Finish {
            reason: reason.clone(),
            tokens: run.completion_tokens,
        });
        if run.live_on {
            crate::live::finish(
                &reason,
                run.completion_tokens,
                &String::from_utf8_lossy(&run.full),
            );
        }
        // The EOG is never written (the row write is the *next* step's batch, and
        // an ending token is never queued), so rows `0..current_pos` hold exactly
        // `cached_tokens`.
        debug_assert_eq!(self.slots[idx].cached_tokens.len(), run.current_pos);
        self.save_slots_if_configured();
    }
}

/// #196: how many **consecutive** steps `serve_loop` tolerates leaving the engine
/// busy without advancing [`BatchEngine::work_units`] before it declares the worker
/// stalled, answers every live and queued request and ends the loop.
///
/// The healthy maximum is **0**: a `tick` that leaves the engine busy has either
/// forwarded a decode row or committed a token through `advance`'s `Continue`, and
/// both increment the counter (#158's invariant — the gates' shared `WorkBound`
/// asserts it per step, and this is the same invariant enforced in production). The
/// number is therefore deliberately loose rather than tuned: 64 steps of slack keep
/// a future engine change that legitimately defers work for a handful of steps from
/// being mistaken for a stall, while a genuinely wedged engine still ends the loop
/// promptly — a no-op step costs microseconds, so 64 of them are well under a
/// millisecond of extra spinning, and the loop was previously spinning forever at
/// 100% CPU. It is a **count** (steps), never a wall-clock number, per rule 4 of
/// the gate contract (`docs/GATE-CONTRACT.md`).
pub const STALL_STEP_LIMIT: u64 = 64;

/// The error every stuck request receives when `serve_loop`'s no-progress bound
/// trips (#196): a `500 server_error`, the same attribution #151 gave a failed
/// decode step — the *worker* stalled, which is not the client's fault and not the
/// saturation `503 unavailable_error` of #121/#150.
pub const WORKER_STALLED_MESSAGE: &str = "the worker stalled";

/// #196: answer every queued-but-unadmitted job when `serve_loop` gives up.
///
/// These jobs (the worker's own `pending` deque plus whatever is still in the
/// channel) have not been placed, and ending the loop drops their `Job` — and with
/// it the only `Sender<StreamEvent>` their handler listens on, which the handler
/// reads as a *completed* empty answer (#121's silent drop: HTTP 200 with empty
/// content). They are therefore answered with the same terminal error the live runs
/// got, through the same [`reject`] primitive a saturated engine uses. Each is
/// counted as admitted (it *left* the queue, so `queue_depth` stays
/// `accepted - admitted`) **and** as dropped (the worker could not place it) — the
/// two counters a rejected job already moves. Returns how many it answered.
fn reject_queued(
    pending: &mut VecDeque<Job>,
    job_rx: &mut mpsc::Receiver<Job>,
    e: &ApiError,
) -> u64 {
    let mut answered = 0;
    for job in pending.drain(..) {
        reject(job, e.clone());
        answered += 1;
    }
    loop {
        match job_rx.try_recv() {
            Ok(job) => {
                reject(job, e.clone());
                answered += 1;
            }
            Err(TryRecvError::Empty) | Err(TryRecvError::Disconnected) => break,
        }
    }
    answered
}

/// Drain jobs (blocking only when nothing is in flight) and step the batch.
///
/// F8: `metrics` is written on every pass — the queue/engine counters here, the
/// KV/allocator reading through `BatchEngine::publish_metrics`.
///
/// #196: the loop carries a **counted no-progress bound**. A `tick` that leaves
/// the engine busy without advancing `BatchEngine::work_units` is counted; after
/// [`STALL_STEP_LIMIT`] consecutive such steps the worker is declared stalled,
/// every live run and every queued job is answered **once** with a `500`, the
/// `worker_stalled_total` counter moves, and the loop ends. Before this, such a
/// step spun the loop at 100% CPU forever and no client ever heard (#151's shape,
/// one level up: the fix there answered a *failed* batch, this answers a batch
/// that never runs). The count is reset on every sign of progress — a wait for
/// work (`blocking_recv`), a step that moved the counter, and #151's `Err` step,
/// which answered its batch and released its slots — so a long-lived server never
/// accumulates toward the limit.
pub fn serve_loop(
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    mut job_rx: mpsc::Receiver<Job>,
    engine: &mut BatchEngine,
    metrics: &super::metrics::ServerMetrics,
) {
    let mut pending: VecDeque<Job> = VecDeque::new();
    // #196: consecutive steps that left the engine busy with a frozen work
    // counter. See the doc comment above; a healthy engine never gets past 0.
    let mut stalled_steps: u64 = 0;
    loop {
        if !engine.busy() {
            // Nothing to step: wait for work. An idle wait is not a spin, so any
            // earlier no-progress run of steps is over.
            stalled_steps = 0;
            let Some(job) = job_rx.blocking_recv() else {
                // No senders left: publish the final reading (idle slots, nothing
                // running) before leaving, so a scrape after the drain is not a
                // stale "busy".
                metrics
                    .worker_pending
                    .store(0, std::sync::atomic::Ordering::Relaxed);
                engine.publish_metrics(metrics);
                break;
            };
            pending.push_back(job);
        }
        // Admit as many as fit, then run one step.
        loop {
            match job_rx.try_recv() {
                Ok(job) => pending.push_back(job),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break,
            }
        }
        metrics
            .worker_pending
            .store(pending.len() as u64, std::sync::atomic::Ordering::Relaxed);
        // Admit everything that arrived as one group, so their prefills can
        // share a forward (`admit` places each request on its own slot and
        // combines the prefills when they fit).
        let group: Vec<Job> = pending.drain(..).collect();
        if !group.is_empty() {
            // `admitted` counts every job that *left* the queue (whether the
            // engine could place it or not), so `requests_total - admitted` is
            // exactly what is still waiting and cannot drift on a rejection.
            metrics
                .jobs_admitted_total
                .fetch_add(group.len() as u64, std::sync::atomic::Ordering::Relaxed);
            metrics
                .worker_pending
                .store(0, std::sync::atomic::Ordering::Relaxed);
            let answers = engine.admit(model, tokenizer, group);
            let dropped = answers.iter().filter(|r| r.is_err()).count() as u64;
            for r in answers {
                if let Err(e) = r {
                    // The engine could not place the request (no idle slot).
                    eprintln!("[server] job rejected: {}", e.message);
                }
            }
            if dropped > 0 {
                metrics
                    .jobs_dropped_total
                    .fetch_add(dropped, std::sync::atomic::Ordering::Relaxed);
            }
        }
        if engine.busy() {
            let work_before = engine.work_units();
            match engine.tick(model, tokenizer) {
                Ok(()) => {
                    // #196: the counted liveness bound. A step that leaves the
                    // engine busy must have moved the counter (#158); one that did
                    // not is the wedge that used to spin here forever. `Ok` only:
                    // #151's failed-batch arm is the `Err` branch below, and it is
                    // progress — it answered that batch and released its slots.
                    if engine.busy() && engine.work_units() == work_before {
                        stalled_steps += 1;
                        if stalled_steps >= STALL_STEP_LIMIT {
                            let e = ApiError::server(WORKER_STALLED_MESSAGE);
                            let runs = engine.fail_all(&e);
                            let queued = reject_queued(&mut pending, &mut job_rx, &e);
                            metrics
                                .worker_stalled_total
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            if queued > 0 {
                                metrics
                                    .jobs_admitted_total
                                    .fetch_add(queued, std::sync::atomic::Ordering::Relaxed);
                                metrics
                                    .jobs_dropped_total
                                    .fetch_add(queued, std::sync::atomic::Ordering::Relaxed);
                            }
                            metrics
                                .worker_pending
                                .store(0, std::sync::atomic::Ordering::Relaxed);
                            eprintln!(
                                "[server] the worker stalled: {STALL_STEP_LIMIT} consecutive \
                                 steps left the engine busy without advancing its work counter \
                                 ({} run(s), {} queued job(s) answered with 500); stopping the \
                                 worker",
                                runs, queued
                            );
                            engine.publish_metrics(metrics);
                            break;
                        }
                    } else {
                        stalled_steps = 0;
                    }
                }
                Err(e) => {
                    eprintln!("[server] step failed: {}", e.message);
                    // #151: `tick` answered the failed batch and released its
                    // slots before returning this error, so the engine moved even
                    // though `work_units` did not. That is progress, not a wedge.
                    stalled_steps = 0;
                }
            }
        }
        engine.publish_metrics(metrics);
        if !engine.busy() && job_rx.is_closed() && pending.is_empty() {
            break;
        }
    }
}

#[cfg(test)]
mod tests;
