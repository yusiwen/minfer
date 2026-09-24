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
        })
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

    /// Slots without a request (their reservation and KV stay).
    pub fn idle_slots(&self) -> usize {
        self.slots.iter().filter(|s| s.run.is_none()).count()
    }

    pub fn busy(&self) -> bool {
        self.slots.iter().any(|s| s.run.is_some())
    }

    /// Admit a request into an idle slot and prefill it. Returns the slot index,
    /// or `unavailable` when every slot is busy (today's behaviour).
    /// Admit a group of requests that arrived together.
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
                None => answers[i] = Some(Err(ApiError::unavailable("no idle slot"))),
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
                    for (i, _, _) in &placed {
                        answers[*i] = Some(Err(ApiError::server(e.clone())));
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

    /// Admit a single request: [`BatchEngine::admit`] with one job.
    pub fn submit(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        job: Job,
    ) -> Result<usize, ApiError> {
        self.admit(model, tokenizer, vec![job])
            .into_iter()
            .next()
            .expect("one job in, one answer out")
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
        // C8b S2: a device whose kernel gathers a `kv_map` (CPU) **shares** the
        // donor's rows instead of copying them — one copy of the bytes read by both
        // sequences. Other devices keep C8a's copy (CUDA is C8b S4, Metal is G5).
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

    /// Admit a request on a **specific** slot.
    ///
    /// A slot's cell offset is part of a request's determinism: RoPE at cell
    /// 256 and at cell 5 agree in exact arithmetic but not in `f32`, so the same
    /// prompt on a different slot can differ in the last ulps — and flip an
    /// argmax on a near-tie. A slot is therefore a *session's KV home*, not an
    /// interchangeable resource, which is also what makes B2's cross-request
    /// prefix reuse possible; this entry point is how a caller pins one.
    /// E3: set the prefill chunk size (fed tokens per prefill forward; `0` = one
    /// forward per prefill, the pre-E3 behaviour).
    pub fn set_prefill_chunk(&mut self, n_batch: usize) {
        self.n_batch = n_batch;
    }

    /// E3: `(largest nt any prefill forward carried, prefill forwards run)`. The
    /// activation-memory bound is `max_nt`, so the gate asserts on this rather than
    /// on a claim about buffers.
    pub fn prefill_stats(&self) -> (usize, usize) {
        (self.prefill_max_nt, self.prefill_forwards)
    }

    /// B2/C5 S2: prompt tokens this engine has fed to prefills (see `prefill_fed`).
    pub fn prefill_fed(&self) -> usize {
        self.prefill_fed
    }

    /// E3: decode steps run between the chunks of a prefill (0 with chunking off).
    pub fn interleaved_ticks(&self) -> u64 {
        self.interleaved_ticks
    }

    /// Whether an already admitted request has a token waiting for its decode
    /// forward — what interleaving a prefill is *for*.
    fn has_pending_decode(&self) -> bool {
        self.slots
            .iter()
            .any(|s| s.run.as_ref().is_some_and(|r| r.needs_forward.is_some()))
    }

    pub fn submit_on(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        idx: usize,
        job: Job,
    ) -> Result<usize, ApiError> {
        if idx >= self.slots.len() {
            return Err(ApiError::invalid_request(format!(
                "slot {idx} does not exist ({} slots)",
                self.slots.len()
            )));
        }
        if self.slots[idx].run.is_some() {
            return Err(ApiError::unavailable(format!("slot {idx} is busy")));
        }
        let nt = job.input_ids.len();
        // C7: size the slot from this request instead of leaving the startup
        // partition in place (see `ensure_slot_capacity`).
        let want = self.wanted_cells(idx, nt, job.params.max_tokens);
        self.ensure_slot_capacity(idx, want);
        let cap = self.slots[idx].cap;
        if nt > cap {
            return Err(ApiError::exceed_context(format!(
                "prompt of {nt} tokens exceeds slot context of {cap}"
            )));
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
                guarded_forward_batch(model, &batch, 1, self.n_ctx_total, &mut self.cache)?;
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
        self.slots[idx].run = Some(Run {
            tx: job.tx,
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
                guarded_forward_batch(model, &batch, 1, self.n_ctx_total, &mut self.cache)?;
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
                Ok(StepOutcome::Continue) => {}
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
        let sampled = crate::sampler::sample_with_penalties(
            last_logits,
            params.temp,
            params.top_k,
            params.top_p,
            params.repeat_penalty,
            params.frequency_penalty,
            params.presence_penalty,
            prev_tokens,
            rng,
        );
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
            let chunk = String::from_utf8_lossy(&run.full[run.emitted..]).into_owned();
            if run.tx.blocking_send(StreamEvent::Text(chunk)).is_err() {
                self.slots[idx].cached_tokens.clear();
                return;
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

/// Drain jobs (blocking only when nothing is in flight) and step the batch.
pub fn serve_loop(
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    mut job_rx: mpsc::Receiver<Job>,
    engine: &mut BatchEngine,
) {
    let mut pending: VecDeque<Job> = VecDeque::new();
    loop {
        if !engine.busy() {
            // Nothing to step: wait for work.
            let Some(job) = job_rx.blocking_recv() else {
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
        // Admit everything that arrived as one group, so their prefills can
        // share a forward (`admit` places each request on its own slot and
        // combines the prefills when they fit).
        let group: Vec<Job> = pending.drain(..).collect();
        if !group.is_empty() {
            for r in engine.admit(model, tokenizer, group) {
                if let Err(e) = r {
                    // The engine could not place the request (no idle slot).
                    eprintln!("[server] job rejected: {}", e.message);
                }
            }
        }
        if engine.busy() {
            if let Err(e) = engine.tick(model, tokenizer) {
                eprintln!("[server] step failed: {}", e.message);
            }
        } else if job_rx.is_closed() && pending.is_empty() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::ModelDef;
    use std::time::Instant;

    fn cached_model() -> Option<std::path::PathBuf> {
        // `MINFER_BATCH_TEST_MODEL` points the measurement at another cached
        // model (the 7B is where batching should pay: decode is weight-bandwidth
        // bound there, while the 0.5B's decode is kernel-compute bound).
        if let Ok(custom) = std::env::var("MINFER_BATCH_TEST_MODEL") {
            let p = std::path::PathBuf::from(custom);
            return p.exists().then_some(p);
        }
        let home = std::env::var_os("HOME")?;
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
        p.exists().then_some(p)
    }

    /// Drive `n` requests through the engine, returning each one's generated
    /// tokens and the wall time. Requests are submitted together (the engine
    /// admits what fits) and stepped until they all finish, which is what
    /// continuous batching does.
    #[derive(Debug, PartialEq)]
    struct Reply {
        text: String,
        tokens: usize,
        reason: String,
    }

    fn sampling_params(max_tokens: i64) -> SamplingParams {
        SamplingParams {
            // Greedy, identical on both sides of the comparison.
            temp: 0.0,
            top_k: 1,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: 7,
            stop_strings: Vec::new(),
            max_tokens,
        }
    }

    fn run_batched(
        model: &dyn ModelDef,
        tok: &Tokenizer,
        prompts: &[Vec<u32>],
        n_slots: usize,
        n_ctx: usize,
        max_tokens: i64,
        stagger: bool,
    ) -> (Vec<Reply>, f64) {
        let mut engine = BatchEngine::new(model, n_slots, n_ctx).expect("engine");
        let mut out: Vec<Option<Reply>> = (0..prompts.len()).map(|_| None).collect();
        let mut text: Vec<String> = vec![String::new(); prompts.len()];
        let mut pending: Vec<(usize, mpsc::Receiver<StreamEvent>)> = Vec::new();
        let mut queue: Vec<(usize, Job)> = Vec::new();
        for (i, p) in prompts.iter().enumerate() {
            let (tx, rx) = mpsc::channel::<StreamEvent>(1024);
            queue.push((
                i,
                Job {
                    input_ids: p.clone(),
                    params: sampling_params(max_tokens),
                    tx,
                },
            ));
            pending.push((i, rx));
        }
        let t0 = Instant::now();
        // Admit, then step until every request has finished.
        while !queue.is_empty() || engine.busy() {
            // `stagger` reproduces the **server's** admission pattern: requests
            // arrive while others decode, so `serve_loop` admits one per step and
            // the step sequence mixes widths (a 1-sequence step, then wider
            // ones). Without it, everything is admitted before the first tick and
            // every step has the same width — which is what this helper always
            // did, and why the blocker below hid from it.
            let mut admit = if stagger { 1 } else { engine.idle_slots() };
            while !queue.is_empty() && engine.idle_slots() > 0 && admit > 0 {
                let (i, job) = queue.remove(0);
                engine
                    .submit(model, tok, job)
                    .unwrap_or_else(|e| panic!("submit request {i}: {}", e.message));
                admit -= 1;
            }
            engine.tick(model, tok).expect("tick");
            // Drain events; a finished request is reported by `Finish`, and its
            // text is the concatenation of the `Text` events.
            for (i, rx) in pending.iter_mut() {
                loop {
                    match rx.try_recv() {
                        Ok(StreamEvent::Text(t)) => text[*i].push_str(&t),
                        Ok(StreamEvent::Finish { reason, tokens }) => {
                            out[*i] = Some(Reply {
                                text: std::mem::take(&mut text[*i]),
                                tokens,
                                reason,
                            });
                        }
                        Ok(StreamEvent::Err(e)) => panic!("request {i}: {}", e.message),
                        Err(_) => break,
                    }
                }
            }
        }
        (
            out.into_iter()
                .map(|r| r.expect("every request finished"))
                .collect(),
            t0.elapsed().as_secs_f64(),
        )
    }

    /// The serial baseline: the same four requests, each served alone, **on the
    /// slot it would occupy in the batched run** — a request's window offset is
    /// part of its determinism (see [`BatchEngine::submit_on`]), so comparing
    /// across offsets would measure arithmetic, not batching.
    fn run_serial(
        model: &dyn ModelDef,
        tok: &Tokenizer,
        prompts: &[Vec<u32>],
        n_slots: usize,
        n_ctx: usize,
        max_tokens: i64,
    ) -> (Vec<Reply>, f64) {
        // One engine, so the baseline pays the same one-time graph builds as the
        // batched run; each request is placed on the slot it would occupy.
        let t0 = Instant::now();
        let mut out = Vec::new();
        let mut engine = BatchEngine::new(model, n_slots, n_ctx).expect("engine");
        for (i, p) in prompts.iter().enumerate() {
            let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
            engine
                .submit_on(
                    model,
                    tok,
                    i,
                    Job {
                        input_ids: p.clone(),
                        params: sampling_params(max_tokens),
                        tx,
                    },
                )
                .unwrap_or_else(|e| panic!("submit request {i}: {}", e.message));
            let mut text = String::new();
            let mut done = None;
            while done.is_none() {
                engine.tick(model, tok).expect("tick");
                loop {
                    match rx.try_recv() {
                        Ok(StreamEvent::Text(t)) => text.push_str(&t),
                        Ok(StreamEvent::Finish { reason, tokens }) => {
                            done = Some(Reply {
                                text: std::mem::take(&mut text),
                                tokens,
                                reason,
                            });
                        }
                        Ok(StreamEvent::Err(e)) => panic!("request {i}: {}", e.message),
                        Err(_) => break,
                    }
                }
            }
            out.push(done.expect("finished"));
        }
        (out, t0.elapsed().as_secs_f64())
    }

    /// Whether a CUDA device participates in this process (`load_model` above
    /// initialises the state when the model is loaded).
    fn cuda_device_active() -> bool {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::CudaState::get().is_some()
        }
        #[cfg(not(feature = "cuda"))]
        {
            false
        }
    }

    /// E2's acceptance, measured at the engine: four requests served as one
    /// decode batch must generate exactly what four serial requests generate,
    /// and take materially less wall time.
    ///
    /// Ignored by default like the other real-model tests:
    ///   cargo test --release --bin minfer -- --ignored server_batch --nocapture
    /// C8a: admit `prompt` on `slot` and drive the engine until it finishes, returning
    /// the text the client would have streamed.
    fn serve_on(
        engine: &mut BatchEngine,
        model: &dyn ModelDef,
        tok: &Tokenizer,
        slot: usize,
        prompt: Vec<u32>,
        max_tokens: i64,
    ) -> String {
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        engine
            .submit_on(
                model,
                tok,
                slot,
                Job {
                    input_ids: prompt,
                    params: sampling_params(max_tokens),
                    tx,
                },
            )
            .expect("admit");
        let mut text = String::new();
        while engine.busy() {
            engine.tick(model, tok).expect("tick");
            while let Ok(ev) = rx.try_recv() {
                if let StreamEvent::Text(t) = ev {
                    text.push_str(&t);
                }
            }
        }
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Text(t) = ev {
                text.push_str(&t);
            }
        }
        text
    }

    /// C8a gate: a prompt served from *another* slot's rows must answer exactly as the
    /// same prompt served on a private run, and the rows must actually be copied rather
    /// than prefilled. The copy is what the counter proves; the equality is the whole
    /// point — C6 makes the donor's different run start arithmetic-free, so a copied
    /// prefix cannot change this slot's answer.
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn a_prefix_copied_from_another_slot_answers_identically() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the C8a gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let prompt = tok.encode("The capital of France is");
        let n_ctx = prompt.len() + 64;

        // Slot 0 computes the prompt; slot 1 must *copy* those rows.
        let mut engine = BatchEngine::new(&*model, 2, n_ctx).expect("engine");
        let first = serve_on(&mut engine, &*model, &tok, 0, prompt.clone(), 8);
        let before = engine.prefix_rows_copied();
        let second = serve_on(&mut engine, &*model, &tok, 1, prompt.clone(), 8);
        assert!(
            engine.prefix_rows_copied() > before,
            "slot 1 prefilled the prompt instead of copying slot 0's {} rows",
            prompt.len()
        );
        assert_eq!(first, second, "a copied prefix must not change the answer");

        // And the same prompt on a single-slot engine, which has nothing to copy.
        let mut alone = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
        let solo = serve_on(&mut alone, &*model, &tok, 0, prompt, 8);
        assert_eq!(
            second, solo,
            "the copied run must answer like a private one"
        );
    }

    /// C8b S3 gate: a request that diverges **inside** a prefix another slot
    /// computed must take private rows there (copy-on-write) instead of storing
    /// through the donor's cells.
    ///
    /// The failure this ticket forbids is invisible from the sharer alone: a store
    /// that wrote through the shared cells would still give the sharer the right
    /// answer (it reads those same cells), and only the **donor** would be corrupted.
    /// So the gate checks four things: the request is served at all (the store
    /// resolver refuses a shared position, so a missing copy-on-write is a loud
    /// error), the copy-on-write counter moved, the answer is the private run's, and
    /// the donor's rows come out byte-identical.
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn a_store_inside_a_shared_prefix_takes_a_private_row() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the C8b S3 gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        // Long enough that the share has rows worth copying.
        let prompt = tok.encode(
            "You are a helpful assistant. Answer in one short sentence. The capital of France is",
        );
        assert!(prompt.len() > 8, "the prompt has to be worth sharing");
        let n_ctx = prompt.len() + 64;

        // A request on slot 1 matching only the prompt's first three tokens: its
        // prefill starts inside the shared prefix, so the store has to copy.
        let mut diverging = prompt[..3].to_vec();
        diverging.extend(tok.encode(" and the capital of Italy is"));
        assert_eq!(
            common_prefix_len(&prompt, &diverging),
            3,
            "the divergence point is the gate's input"
        );

        // One whole scenario — donor, sharer, then the diverging request — with the
        // share either on (the ticket's path) or replaced by C8a's **copy** (the
        // A/B). The copy variant is the baseline the answer is compared against
        // because it is *shape-matched*: the same requests, the same fed positions
        // and the same K/V bytes, with the donor's rows duplicated instead of
        // referenced. A one-slot baseline cannot be: on CUDA a prefill's GEMM shape
        // changes the K/V it computes, so a run that feeds a different number of
        // tokens answers differently for reasons that have nothing to do with
        // sharing — the first form of this gate passed on CPU and failed on CUDA for
        // exactly that reason.
        let mut scenario =
            |share: bool| -> (String, usize, usize, u64, Vec<Vec<f32>>, Vec<Vec<f32>>) {
                if !share {
                    std::env::set_var("MINFER_NO_KV_SHARE", "1");
                }
                let mut engine = BatchEngine::new(&*model, 2, n_ctx).expect("engine");
                let (donor_seq, dst_seq) = (engine.slots[0].seq, engine.slots[1].seq);
                serve_on(&mut engine, &*model, &tok, 0, prompt.clone(), 8);
                serve_on(&mut engine, &*model, &tok, 1, prompt.clone(), 8);
                let shared_rows = engine
                    .cache
                    .alloc()
                    .kv_seq_slot(dst_seq)
                    .expect("slot 1 has a run")
                    .shared
                    .rows;
                let shared_cells = engine.cache.alloc().kv_arena_stats().shared_cells;
                let cows_before = engine.cow_stats().0;
                let donor_before = kv_rows_of(&mut engine, donor_seq);
                let answer = serve_on(&mut engine, &*model, &tok, 1, diverging.clone(), 8);
                let cows = engine.cow_stats().0 - cows_before;
                // The donor is still a live run — a reclaim would make the byte check
                // vacuous rather than wrong, which is the kind of silent pass this
                // assertion exists to prevent.
                assert!(engine.cache.alloc().kv_seq_slot(donor_seq).is_some());
                assert_eq!(
                    kv_rows_of(&mut engine, donor_seq),
                    donor_before,
                    "the diverging request wrote through the shared prefix"
                );
                let dst_rows = kv_rows_of(&mut engine, dst_seq);
                if !share {
                    std::env::remove_var("MINFER_NO_KV_SHARE");
                }
                (
                    answer,
                    shared_rows,
                    shared_cells,
                    cows,
                    donor_before,
                    dst_rows,
                )
            };

        let (shared_answer, shared_rows, shared_cells, cows, donor_rows, dst_a) = scenario(true);
        // Two identical share runs must agree byte for byte — the property that
        // caught the decode-position bug this gate first ran into: a skipped
        // position left an unwritten row that attention read, so the answer
        // depended on the arena's history.
        let (shared_answer2, _, _, _, _, dst_b) = scenario(true);
        assert_eq!(shared_answer, shared_answer2, "two share runs must agree");
        assert_eq!(dst_a, dst_b, "two share runs must write the same rows");
        assert!(
            shared_rows > 3,
            "slot 1 must read {shared_rows} rows in place for the gate to mean anything"
        );
        assert!(!donor_rows.is_empty(), "the donor must hold rows");
        // (1) The mechanism ran, (2) the rows really are shared in place on this
        // device (before S4 a CUDA run copied them), and (3) the donor survived it.
        assert!(cows > 0, "the store must have copied a private row");
        assert!(
            shared_cells >= shared_rows,
            "the prefix was copied, not shared, on device {}",
            model.device().name()
        );

        let (copied_answer, copied_rows, copied_cells, copied_cows, _, _) = scenario(false);
        assert_eq!(copied_rows, 0, "MINFER_NO_KV_SHARE must not share");
        assert_eq!(copied_cells, 0);
        assert_eq!(copied_cows, 0);
        // (4) And the answers agree: the shared run holds the same bytes in the same
        // order as the copied one, so a copy-on-write that moved the wrong rows (or
        // did not move them at all) shows up here.
        assert_eq!(
            shared_answer, copied_answer,
            "the shared run must answer like the shape-matched copied one"
        );
    }

    /// The K/V rows a sequence's written positions hold, one `Vec` per (layer, K or
    /// V, position) in a deterministic order. Cells are resolved through the span
    /// list on every call, so a relocation between two snapshots is not a difference;
    /// a sharing sequence's shared rows are read from wherever they live.
    fn kv_rows_of(engine: &mut BatchEngine, seq: SeqId) -> Vec<Vec<f32>> {
        let written = engine
            .cache
            .alloc()
            .kv_seq_slot(seq)
            .map_or(0, |s| s.written);
        let n_ctx = engine.cache.alloc().kv_n_ctx().max(1);
        let cells: Vec<usize> = (0..written)
            .map(|p| {
                engine
                    .cache
                    .alloc()
                    .kv_cell_of(seq, p)
                    .unwrap_or_else(|| panic!("no cell for sequence {seq} position {p}"))
            })
            .collect();
        // The region is over-allocated as f32 slots, but an f16 cache stores a row
        // as `nkt / 2` f32 slots (`store_kv_f16` indexes halves), so the window a
        // position covers is half as wide there. Reading it at the f32 width mixed
        // two rows per window and made this snapshot report differences in cells
        // nothing had written.
        let half_width = {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::kv_cache_is_f16()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let mut out: Vec<Vec<f32>> = Vec::new();
        let mut layer = 0;
        while let Some((k, v)) = engine.cache.alloc().copy_kv_to_cpu(layer) {
            let row = if half_width {
                (k.len() / n_ctx / 2).max(1)
            } else {
                k.len() / n_ctx
            };
            for &cell in &cells {
                let at = cell * row;
                out.push(k[at..at + row].to_vec());
                out.push(v[at..at + row].to_vec());
            }
            layer += 1;
        }
        out
    }

    /// C7: the growth policy is pure, and it plans from the request rather than
    /// from the startup partition.
    #[test]
    fn wanted_cells_plans_from_the_request() {
        // 4 slots over 2048 cells: 512 each. A prompt plus its answer that fit
        // ask for nothing new, so the common case never repartitions.
        assert_eq!(wanted_cells_from(100, 64, 512, 2048), 164);
        assert!(wanted_cells_from(100, 64, 512, 2048) <= 512);
        // An unbounded answer takes the slot's current capacity as headroom.
        assert_eq!(wanted_cells_from(100, -1, 512, 2048), 612);
        // A prompt past the partition asks for the prompt plus that headroom...
        assert_eq!(wanted_cells_from(1000, -1, 512, 2048), 1512);
        // ...clamped to the arena...
        assert_eq!(wanted_cells_from(1000, 100_000, 512, 2048), 2048);
        // ...and never below the prompt, so an impossible request stays the
        // caller's loud error instead of quietly shrinking into a smaller ask.
        assert_eq!(wanted_cells_from(4000, 8, 512, 2048), 4000);
    }

    /// C7 acceptance: a request whose prompt does not fit its share of the arena
    /// is served anyway, and the repartition cannot change the answer.
    ///
    /// Four slots over `n_ctx` give each slot `n_ctx/4`, and the prompt below
    /// needs most of the arena, so it can only be served by reclaiming the idle
    /// slots above it. Both admission paths are driven — `run_batched` goes
    /// through `submit`/`prefill_group`, `run_serial` through `submit_on`, which
    /// is the one the server uses — and both must agree with a one-slot engine,
    /// which needs no reclaim at all. That equality is the C6 payoff: a moved row
    /// keeps its sequence-relative position, so where the partition puts a
    /// sequence cannot show up in its logits.
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn a_long_request_may_use_the_whole_arena() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the C7 gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode(&"buffalo ".repeat(300));
        let n_ctx = ids.len() + 64;
        assert!(
            ids.len() > n_ctx / 2,
            "the prompt must not fit a quarter-slot partition: {} tokens over {n_ctx} cells",
            ids.len()
        );
        let (group, _) = run_batched(&*model, &tok, &[ids.clone()], 4, n_ctx, 8, false);
        let (slot, _) = run_serial(&*model, &tok, &[ids.clone()], 4, n_ctx, 8);
        let (alone, _) = run_batched(&*model, &tok, &[ids], 1, n_ctx, 8, false);
        assert!(
            !group[0].text.is_empty() && !slot[0].text.is_empty(),
            "the long request must be served, not rejected ({} / {})",
            group[0].reason,
            slot[0].reason
        );
        assert_eq!(
            group[0].text, alone[0].text,
            "reclaiming idle capacity changed the continuation ({} vs {} tokens)",
            group[0].tokens, alone[0].tokens
        );
        assert_eq!(
            slot[0].text, alone[0].text,
            "the server's own admission path disagrees after a reclaim ({} vs {} tokens)",
            slot[0].tokens, alone[0].tokens
        );

        // #59: a request with no token budget is bounded by its run alone, and it must
        // stop *at* its last cell instead of forwarding one past it. Before the
        // commit-time check that forward was issued, `kv_cells_for_seq` rejected the
        // batch, and this request failed instead of finishing — so `reason == "length"`
        // with exactly the run's capacity in tokens is the regression test.
        let counter = tok.encode("Count slowly from 1 to 400, one number per line: 1,");
        let budget = 48;
        let (open_ended, _) = run_batched(
            &*model,
            &tok,
            &[counter.clone()],
            1,
            counter.len() + budget,
            -1,
            false,
        );
        assert_eq!(
            open_ended[0].reason, "length",
            "an unbounded request must end on the context bound, not fail: {:?}",
            open_ended[0]
        );
        assert_eq!(
            open_ended[0].tokens, budget,
            "it must use exactly the cells its run has"
        );
    }

    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn server_batch_matches_serial_and_is_faster() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the batching measurement");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let texts = [
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Italy is",
            "The capital of Spain is",
        ];
        // Configurable so a bisect can put the *server's* exact configuration on
        // this path: the field bug of 2026-09-19 (plan §14) reproduces through the
        // server with the 7B and two slots, but not here with the 0.5B and four —
        // and these three knobs are the differences that are left.
        let templated = std::env::var("MINFER_BATCH_TEST_TEMPLATED").is_ok();
        let prompts: Vec<Vec<u32>> = texts
            .iter()
            .map(|p| {
                if templated {
                    // Exactly what the server does (`server::mod`): the GGUF's
                    // chat template, rendered with a generation prompt, then
                    // tokenized. `model.format_chat` is a *different* path and
                    // produced a 13-token prompt where the server's is 34 — which
                    // is why the first bisect compared unequal inputs.
                    let tpl = super::super::chat_template_from_gguf(&gguf.parts[0].data)
                        .unwrap_or_default();
                    let msgs = vec![("user".to_string(), Some(p.to_string()))];
                    tok.encode(&crate::template::render_messages(
                        &tpl,
                        &msgs,
                        true,
                        &tok.bos_text(),
                    ))
                } else {
                    tok.encode(p)
                }
            })
            .collect();
        let n_ctx: usize = std::env::var("MINFER_BATCH_TEST_CTX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let n_slots: usize = std::env::var("MINFER_BATCH_TEST_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(4);
        let n_req = n_slots.min(prompts.len());
        let prompts: Vec<Vec<u32>> = prompts.into_iter().take(n_req).collect();
        let max_tokens = 16;
        eprintln!(
            "[e2] config: {n_slots} slot(s), n_ctx {n_ctx}, {n_req} request(s), templated={templated}, \
             prompt len {}",
            prompts[0].len()
        );

        let (batched, t_batch) =
            run_batched(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens, false);
        let (serial, t_serial) = run_serial(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens);

        // Byte-equality is a **CPU** property: both sides drive the same engine
        // with the same slot reservations (`submit_on` pins each request to the
        // slot it would occupy), so on CPU the arithmetic is identical. On a
        // device it cannot be — the batched step is `nt = 4` and the serial step
        // `nt = 1`, and CUDA's kernels tile by `nt` (measured drift 0.22–0.37 on
        // logits; the plan records it as a named tolerance class), so a greedy
        // continuation may legitimately diverge after a few tokens.
        //
        // Device runs therefore assert the *structural* property that a wrong
        // window would break immediately — the two continuations must start
        // identically — and report how far they track; the window assignment
        // itself is pinned bitwise on device by
        // `batch_order_does_not_change_a_sequences_logits` (same shape, same
        // layout) and `cuda_two_sequences_do_not_cross_attend`.
        for (i, (b, s)) in batched.iter().zip(&serial).enumerate() {
            assert_eq!(b.reason, s.reason, "request {i}: finish reason differs");
            assert!(!b.text.is_empty(), "request {i}: batched generated nothing");
            assert!(!s.text.is_empty(), "request {i}: serial generated nothing");
            if cuda_device_active() {
                let common = b
                    .text
                    .bytes()
                    .zip(s.text.bytes())
                    .take_while(|(x, y)| x == y)
                    .count();
                assert!(
                    common > 0,
                    "request {i}: batched {b:?} and serial {s:?} diverge at the first byte on a \
                     device, which numerics cannot explain"
                );
                eprintln!(
                    "[e2] request {i}: {common} leading byte(s) shared on device; batched {:?} ({}) vs serial {:?} ({})",
                    b.text, b.tokens, s.text, s.tokens
                );
            } else {
                assert_eq!(
                    b.text, s.text,
                    "request {i}: batched {:?} ({}) vs serial {:?} ({})",
                    b.text, b.tokens, s.text, s.tokens
                );
                assert_eq!(b.tokens, s.tokens, "request {i}: token count differs");
            }
        }
        // ---- the server's pattern: staggered admission (mixed step widths) ----
        let (staggered, t_stag) =
            run_batched(&*model, &tok, &prompts, n_slots, n_ctx, max_tokens, true);
        let device = cuda_device_active();
        for (i, (st, s)) in staggered.iter().zip(&serial).enumerate() {
            assert_eq!(
                st.reason, s.reason,
                "staggered request {i}: finish reason differs"
            );
            assert!(
                !st.text.is_empty(),
                "staggered request {i} generated nothing"
            );
            if device {
                let common = st
                    .text
                    .bytes()
                    .zip(s.text.bytes())
                    .take_while(|(x, y)| x == y)
                    .count();
                assert!(
                    common > 0,
                    "staggered request {i} diverges from its serial reference at the first byte, \
                     which numerics cannot explain: staggered {:?} vs serial {:?}",
                    st.text,
                    s.text
                );
                eprintln!(
                    "[e2] staggered request {i}: {common} leading byte(s) shared; {:?} vs serial {:?}",
                    st.text, s.text
                );
            } else {
                assert_eq!(
                    st.text, s.text,
                    "staggered request {i}: {:?} vs serial {:?}",
                    st.text, s.text
                );
            }
        }
        eprintln!(
            "[e2] staggered {:.2}s vs simultaneous {:.2}s for {n_slots} slots",
            t_stag, t_batch
        );
        let total: usize = serial.iter().map(|r| r.tokens).sum();
        assert!(total > 0, "the workload generated nothing");
        eprintln!(
            "[e2] {n_slots} slots: batched {t_batch:.2}s vs serial {t_serial:.2}s for {total} tokens \
             (per-request {:?} batched / {:?} serial)",
            batched.iter().map(|r| r.tokens).collect::<Vec<_>>(),
            serial.iter().map(|r| r.tokens).collect::<Vec<_>>()
        );
        eprintln!(
            "[e2] throughput {:.1} tok/s batched vs {:.1} tok/s serial = {:.2}x",
            total as f64 / t_batch,
            total as f64 / t_serial,
            t_serial / t_batch
        );
        assert!(
            t_serial > t_batch,
            "batching must not be slower ({t_batch:.2}s vs {t_serial:.2}s)"
        );
    }
    /// E3: the chunk plan. Its boundaries are the whole contract — the last span is
    /// the remainder, `0` means "off" (one span, the pre-E3 behaviour), and a suffix
    /// that already fits is not split.
    #[test]
    fn prefill_chunks_split_a_suffix_by_the_chunk_size() {
        assert_eq!(prefill_chunks(0, 10, 0), vec![(0, 10)], "0 = chunking off");
        assert_eq!(prefill_chunks(0, 10, 10), vec![(0, 10)]);
        assert_eq!(
            prefill_chunks(0, 10, 16),
            vec![(0, 10)],
            "a suffix that fits stays one span"
        );
        assert_eq!(prefill_chunks(0, 10, 4), vec![(0, 4), (4, 8), (8, 10)]);
        assert_eq!(
            prefill_chunks(0, 8, 4),
            vec![(0, 4), (4, 8)],
            "an exact multiple must not emit an empty tail"
        );
        assert_eq!(
            prefill_chunks(3, 11, 4),
            vec![(3, 7), (7, 11)],
            "the reused prefix is not fed, so the split starts at `from`"
        );
        assert_eq!(prefill_chunks(5, 5, 4), Vec::<(usize, usize)>::new());
        assert_eq!(prefill_chunks(6, 5, 4), Vec::<(usize, usize)>::new());
        assert_eq!(prefill_chunks(0, 1, 8), vec![(0, 1)]);
        // Whatever the numbers, the spans tile `[from, total)` exactly and none is
        // wider than the chunk.
        for (from, total, chunk) in [(0usize, 100usize, 7usize), (0, 100, 1), (13, 91, 9)] {
            let spans = prefill_chunks(from, total, chunk);
            assert_eq!(spans.first().map(|s| s.0), Some(from));
            assert_eq!(spans.last().map(|s| s.1), Some(total));
            for (i, &(a, b)) in spans.iter().enumerate() {
                assert!(
                    a < b && b - a <= chunk,
                    "span {i} = ({a}, {b}) outside the chunk"
                );
                if i > 0 {
                    assert_eq!(
                        spans[i - 1].1,
                        a,
                        "span {i} does not continue the previous one"
                    );
                }
            }
        }
    }

    #[test]
    fn the_prefill_chunk_size_comes_from_the_env_or_the_default() {
        assert_eq!(prefill_chunk_size(None), DEFAULT_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_size(Some("")), DEFAULT_PREFILL_CHUNK);
        assert_eq!(prefill_chunk_size(Some(" 512 ")), 512);
        assert_eq!(prefill_chunk_size(Some("0")), 0, "0 is the off switch");
        assert_eq!(
            prefill_chunk_size(Some("banana")),
            DEFAULT_PREFILL_CHUNK,
            "a typo keeps the default rather than disabling chunking"
        );
        assert_eq!(prefill_chunk_size(Some("-1")), DEFAULT_PREFILL_CHUNK);
    }

    /// E3 gate helper: the bytes of `Text` a slot has been sent so far, drained
    /// without blocking (the events are already queued by the engine).
    /// C5 S2: the slot table is JSON in the container's host section — a version this
    /// build does not know, or a shape it cannot read, is `None` (the caller refuses the
    /// snapshot), never a half-parsed table.
    #[test]
    fn a_slot_table_round_trips_and_refuses_what_it_cannot_read() {
        let json = r#"{"version":1,"n_slots":2,"n_ctx_total":64,
            "slots":[{"seq":1,"start":0,"cap":32,"cached_tokens":[5,6,7]},
                     {"seq":2,"start":32,"cap":32,"cached_tokens":[]}]}"#;
        let snap = BatchEngine::slots_from_json(json).expect("parses");
        assert_eq!(snap.n_slots, 2);
        assert_eq!(snap.n_ctx_total, 64);
        assert_eq!(snap.slots[0].seq, 1);
        assert_eq!(snap.slots[0].cached_tokens, vec![5, 6, 7]);
        assert!(snap.slots[1].cached_tokens.is_empty());

        // Another version, a missing field, and a non-JSON blob are all `None`.
        let bumped = json.replace("\"version\":1", "\"version\":2");
        assert!(BatchEngine::slots_from_json(&bumped).is_none());
        assert!(BatchEngine::slots_from_json(r#"{"version":1,"n_slots":1}"#).is_none());
        assert!(BatchEngine::slots_from_json("not json").is_none());
    }

    /// C5 S2's acceptance on a real model: a snapshot written by one engine is resumed by
    /// another with the history **not** re-prefilled, and the continuation matches the run
    /// that never stopped. A snapshot from another `--n-slots` is refused loudly.
    #[test]
    #[ignore = "requires the cached 0.5B model and writes a snapshot file"]
    fn a_slot_snapshot_resumes_the_context_without_re_prefilling() {
        let Some(path) = cached_model() else {
            eprintln!("Qwen2.5-0.5B q4_0 not cached; skipping the slot-snapshot gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        #[cfg(feature = "cuda")]
        let _guard = crate::cuda::CudaState::model_load_guard();
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let n_ctx = 512usize;
        let n_slots = 2usize;
        let file = std::env::temp_dir().join(format!(
            "minfer-c5s2-slots-{}-{:?}.bin",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::remove_file(&file).ok();

        let prompt = tok.encode("The capital of France is");
        let second = tok.encode(" and the capital of Japan is");

        // Run one prompt to completion and let the engine write the snapshot.
        let (text_a, tokens_a, cold_fed) = {
            let mut a = BatchEngine::new(&*model, n_slots, n_ctx).expect("engine");
            a.set_slots_file(Some(file.clone()));
            let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
            a.submit(
                &*model,
                &tok,
                Job {
                    input_ids: prompt.clone(),
                    params: sampling_params(4),
                    tx,
                },
            )
            .expect("submit");
            while a.busy() {
                a.tick(&*model, &tok).expect("tick");
            }
            let cold_fed = a.prefill_fed();
            assert_eq!(
                cold_fed,
                prompt.len(),
                "the cold run must feed the whole prompt"
            );
            let mut text = String::new();
            let mut tokens = 0usize;
            while let Ok(ev) = rx.try_recv() {
                match ev {
                    StreamEvent::Text(t) => text.push_str(&t),
                    StreamEvent::Finish { tokens: n, .. } => tokens = n,
                    _ => {}
                }
            }
            (text, tokens, cold_fed)
        };
        assert!(file.exists(), "the snapshot must be written on completion");

        // A fresh engine resumes it: the same prompt prefills **nothing**.
        let mut b = BatchEngine::new(&*model, n_slots, n_ctx).expect("engine");
        let (slots, bytes) = b
            .load_slots(&file, &*model)
            .unwrap_or_else(|e| panic!("load_slots: {e}"));
        assert_eq!(slots, n_slots);
        assert!(bytes > 0);
        let fed_before = b.prefill_fed();
        let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
        b.submit(
            &*model,
            &tok,
            Job {
                input_ids: prompt.clone(),
                params: sampling_params(4),
                tx,
            },
        )
        .expect("submit");
        while b.busy() {
            b.tick(&*model, &tok).expect("tick");
        }
        let warm_fed = b.prefill_fed() - fed_before;
        assert_eq!(
            warm_fed,
            1,
            "the snapshot holds the history, so only the query token is fed \
             (the cold run fed {cold_fed} of {} token(s))",
            prompt.len()
        );
        let mut text_b = String::new();
        let mut tokens_b = 0usize;
        while let Ok(ev) = rx.try_recv() {
            match ev {
                StreamEvent::Text(t) => text_b.push_str(&t),
                StreamEvent::Finish { tokens: n, .. } => tokens_b = n,
                _ => {}
            }
        }
        assert_eq!(text_a, text_b, "the resumed continuation must match");
        assert_eq!(tokens_a, tokens_b);

        // ... and its *next* turn prefills only the delta, not the history: the
        // continuation carries the whole conversation, so a re-render would feed
        // `prompt + second` tokens.
        let mut continuation = prompt.clone();
        continuation.extend_from_slice(&second);
        let before_delta = b.prefill_fed();
        let (tx, _rx) = mpsc::channel::<StreamEvent>(1024);
        b.submit(
            &*model,
            &tok,
            Job {
                input_ids: continuation.clone(),
                params: sampling_params(2),
                tx,
            },
        )
        .expect("submit");
        while b.busy() {
            b.tick(&*model, &tok).expect("tick");
        }
        let fed_delta = b.prefill_fed() - before_delta;
        assert!(fed_delta > 0, "the delta still needs a forward");
        assert!(
            fed_delta < continuation.len(),
            "the next turn fed {fed_delta} of {} token(s) — the history came from the snapshot",
            continuation.len()
        );

        // A snapshot from another --n-slots is refused, loudly, naming both.
        let mut one = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
        let err = one
            .load_slots(&file, &*model)
            .expect_err("a 2-slot snapshot must not load into a 1-slot server");
        assert!(err.contains("2-slot"), "{err}");
        assert!(err.contains("--n-slots"), "{err}");
        // ... and one from another --n-ctx, too.
        let mut wide = BatchEngine::new(&*model, n_slots, n_ctx * 2).expect("engine");
        let err = wide
            .load_slots(&file, &*model)
            .expect_err("an n_ctx mismatch must be refused");
        assert!(
            err.contains(&n_ctx.to_string()) && err.contains(&(n_ctx * 2).to_string()),
            "the refusal must name both context lengths: {err}"
        );

        std::fs::remove_file(&file).ok();
    }

    fn drain_text_len(rx: &mut mpsc::Receiver<StreamEvent>) -> usize {
        let mut n = 0;
        while let Ok(ev) = rx.try_recv() {
            if let StreamEvent::Text(t) = ev {
                n += t.len();
            }
        }
        n
    }

    /// E3 acceptance: a prompt several times the chunk size is served with **every**
    /// prefill forward bounded by the chunk, and the continuation is the one the
    /// unchunked path produced.
    ///
    /// The comparison class is the repo's standing one: bitwise on CPU (its kernels
    /// are per-token, so shape never enters the arithmetic) and a named tolerance on
    /// CUDA, whose prefill GEMM tiles by `nt` and quantizes activations to int8. The
    /// gate also asserts that the *unchunked* run really did exceed the chunk —
    /// otherwise it would be proving nothing.
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn a_chunked_prefill_answers_like_an_unchunked_one() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the E3 equality gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode(&"buffalo ".repeat(96));
        let chunk = (ids.len() / 4).max(8);
        let n_ctx = ids.len() + 64;
        assert!(
            ids.len() > 4 * chunk / 2,
            "the prompt must be several chunks long: {} tokens, chunk {chunk}",
            ids.len()
        );

        // Returns the prefill's own tail-row logits (what the request samples its
        // first token from), the continuation, and the forward stats.
        let drive = |chunk: usize| -> (Vec<f32>, String, usize, usize) {
            let mut engine = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
            engine.set_prefill_chunk(chunk);
            let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
            engine
                .submit(
                    &*model,
                    &tok,
                    Job {
                        input_ids: ids.clone(),
                        params: sampling_params(8),
                        tx,
                    },
                )
                .expect("submit");
            let (max_nt, forwards) = engine.prefill_stats();
            let logits = engine.slots[0]
                .run
                .as_ref()
                .expect("the prefill installed a run")
                .last_logits
                .clone();
            while engine.busy() {
                engine.tick(&*model, &tok).expect("tick");
            }
            let mut text = String::new();
            while let Ok(ev) = rx.try_recv() {
                if let StreamEvent::Text(t) = ev {
                    text.push_str(&t);
                }
            }
            (logits, text, max_nt, forwards)
        };

        let (plain_logits, plain, max_plain, fwd_plain) = drive(0);
        let (chunked_logits, chunked, max_chunked, fwd_chunked) = drive(chunk);
        eprintln!(
            "[e3] {}-token prompt: chunked {fwd_chunked} forward(s), max nt {max_chunked}; \
             unchunked {fwd_plain} forward(s), max nt {max_plain}",
            ids.len()
        );
        assert!(
            fwd_chunked >= 4,
            "the chunked run must really split ({fwd_chunked} forwards for a {}-token prompt \
             at chunk {chunk})",
            ids.len()
        );
        assert!(
            max_chunked <= chunk,
            "a prefill forward carried {max_chunked} tokens, over the {chunk} chunk"
        );
        assert!(
            max_plain > chunk,
            "the unchunked run carried only {max_plain} tokens, so this gate proves nothing"
        );
        assert!(!chunked.is_empty(), "the chunked run produced no text");
        // The comparison class is the repo's standing one: **bitwise** on CPU (its
        // kernels are per-token, so shape never enters the arithmetic) and a named
        // tolerance on CUDA, whose prefill tiles by `nt` and quantizes activations to
        // int8 — the same tokens at a different width land on different scores
        // (measured <= 0.37 absolute on this repo's fixtures; 1.0 is the gross-error
        // bound `cross_shape_tolerance` uses). The *continuation* is asserted only on
        // CPU: on a degenerate repeated-token prompt a sub-tolerance logit shift can
        // flip an argmax, which is a fact about the prompt, not about chunking.
        assert_eq!(
            plain_logits.len(),
            chunked_logits.len(),
            "logit widths differ"
        );
        let worst = plain_logits
            .iter()
            .zip(&chunked_logits)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let on_cuda = {
            #[cfg(feature = "cuda")]
            {
                crate::cuda::CudaState::get().is_some()
            }
            #[cfg(not(feature = "cuda"))]
            {
                false
            }
        };
        let tol = if on_cuda { 1.0 } else { 0.0 };
        let agree = plain
            .bytes()
            .zip(chunked.bytes())
            .take_while(|(a, b)| a == b)
            .count();
        eprintln!(
            "[e3] prefill logits: max |Δ| = {worst} (class {tol}); continuations agree on the \
             first {agree} bytes"
        );
        assert!(
            worst <= tol,
            "chunking moved the prefill's logits by {worst} (class {tol})"
        );
        if !on_cuda {
            assert_eq!(
                plain, chunked,
                "chunking changed the continuation (must be bitwise on CPU)"
            );
        }
    }

    /// E4 S3 acceptance on the real path: a **repeated** request with the same chunk pattern
    /// stops rebuilding. The engine's `GraphCache` now keeps one graph per `GraphParams`, so
    /// the second request's prefill chunks (same sizes) hit it — before S3 each chunk was a
    /// fresh build, which is the "one forward's fixed overhead per chunk" the E3 record
    /// measured. The observable is the cache's own (builds, reuses).
    #[test]
    #[ignore = "requires the cached 0.5B model"]
    fn a_repeated_chunked_prefill_stops_rebuilding() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the E4 S3 rebuild gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ids = tok.encode(&"buffalo ".repeat(96));
        let chunk = (ids.len() / 4).max(8);
        let n_ctx = ids.len() + 64;

        let mut engine = BatchEngine::new(&*model, 1, n_ctx).expect("engine");
        engine.set_prefill_chunk(chunk);
        let mut submit_once = |engine: &mut BatchEngine| {
            let (tx, mut rx) = mpsc::channel::<StreamEvent>(1024);
            engine
                .submit(
                    &*model,
                    &tok,
                    Job {
                        input_ids: ids.clone(),
                        params: sampling_params(4),
                        tx,
                    },
                )
                .expect("submit");
            while engine.busy() {
                engine.tick(&*model, &tok).expect("tick");
            }
            while rx.try_recv().is_ok() {}
        };

        submit_once(&mut engine);
        let (b1, r1) = engine.cache.stats();
        assert!(b1 > 0, "the first request built its chunk graphs");
        submit_once(&mut engine);
        let (b2, r2) = engine.cache.stats();
        eprintln!("[e4-s3] request 1: {b1} builds / {r1} reuses; request 2: {b2} / {r2}");
        assert_eq!(
            b2, b1,
            "the second request must build nothing: its chunk shapes are cached"
        );
        assert!(r2 > r1, "and it must hit the cache instead");
    }

    /// E3 acceptance: while a long prompt prefills, the slots already serving keep
    /// taking their decode steps.
    ///
    /// The interleaving is observed the way the ticket states it — as tokens emitted
    /// by the *other* slot during the prefill call — with the A/B being chunking off,
    /// where one forward means no decode step can fit inside the prefill at all.
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn a_long_prefill_keeps_another_slot_decoding() {
        let Some(path) = cached_model() else {
            eprintln!("0.5B q4_0 not cached; skipping the E3 interleaving gate");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        // Both prompts are repeats: neither is expected to EOG, which keeps the
        // scenario about scheduling rather than about the model's mood.
        let short = tok.encode(&"buffalo ".repeat(48));
        let long = tok.encode(&"buffalo ".repeat(192));
        let chunk = (long.len() / 4).max(8);
        let n_ctx = long.len() + 256;

        let drive = |chunk: usize| -> (usize, u64, usize) {
            let mut engine = BatchEngine::new(&*model, 2, n_ctx).expect("engine");
            engine.set_prefill_chunk(chunk);
            // Slot 1 is already serving when the long request arrives.
            let (tx1, mut rx1) = mpsc::channel::<StreamEvent>(4096);
            engine
                .submit_on(
                    &*model,
                    &tok,
                    1,
                    Job {
                        input_ids: short.clone(),
                        params: sampling_params(48),
                        tx: tx1,
                    },
                )
                .expect("short submit");
            for _ in 0..3 {
                engine.tick(&*model, &tok).expect("tick");
            }
            // Drain what the three ticks produced: the next drain's *return* is then
            // exactly the bytes slot 1 gains while the long prefill runs (draining is
            // destructive, so nothing may be subtracted here).
            let before = drain_text_len(&mut rx1);
            assert!(
                before > 0,
                "slot 1 must have emitted something before the long prefill"
            );
            let (tx0, _rx0) = mpsc::channel::<StreamEvent>(4096);
            engine
                .submit_on(
                    &*model,
                    &tok,
                    0,
                    Job {
                        input_ids: long.clone(),
                        params: sampling_params(4),
                        tx: tx0,
                    },
                )
                .expect("long submit");
            let gained = drain_text_len(&mut rx1);
            (gained, engine.interleaved_ticks(), engine.prefill_stats().0)
        };

        let (grew, ticks, max_nt) = drive(chunk);
        let (grew_off, ticks_off, max_nt_off) = drive(0);
        eprintln!(
            "[e3] long prefill: chunked -> {ticks} interleaved step(s), slot 1 gained {grew} \
             bytes, max prefill nt {max_nt}; chunking off -> {ticks_off} step(s), gained \
             {grew_off}, max nt {max_nt_off}"
        );
        assert!(max_nt <= chunk, "chunked prefill carried {max_nt} tokens");
        assert!(
            ticks > 0 && grew > 0,
            "a chunked prefill must keep the other slot decoding (ticks {ticks}, bytes {grew})"
        );
        assert_eq!(
            ticks_off, 0,
            "chunking off is one forward, so nothing can interleave"
        );
        assert_eq!(
            grew_off, 0,
            "with chunking off the other slot cannot advance during the prefill"
        );
    }
}
