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
const REPEAT_LAST_N: usize = 64;

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

pub struct BatchEngine {
    cache: GraphCache,
    n_ctx_total: usize,
    slots: Vec<SlotState>,
    special: SpecialTokens,
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
        let mut slots = Vec::with_capacity(n_slots);
        for i in 0..n_slots {
            // Sequence ids start at 1: 0 is SEQ_MAIN's, and a slot must never
            // look like the classic single-sequence path.
            let seq = 1 + i as SeqId;
            let slot = cache
                .alloc()
                .kv_reserve_seq(seq, cap)
                .map_err(|e| format!("slot {i}: {e}"))?;
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
        })
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
        if placed.len() > 1 && total <= MAX_PREFILL_BATCH && self.prefill_batch_ok() {
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
    /// contiguous in the batch, its positions are its slot's cells, and the
    /// logits come back one row per sequence, in `placed` order.
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
            let (feed_from, _) = self.feed_span(*slot, &job.input_ids);
            let nt = job.input_ids.len();
            let cap = self.slots[*slot].cap;
            if nt > cap {
                return Err(format!(
                    "prompt of {nt} tokens exceeds slot context of {cap}"
                ));
            }
            let start = self.slots[*slot].start;
            tokens.extend_from_slice(&job.input_ids[feed_from..]);
            positions.extend((feed_from..nt).map(|i| start + i));
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
    pub fn submit_on(
        &mut self,
        model: &dyn ModelDef,
        tokenizer: &Tokenizer,
        idx: usize,
        job: Job,
    ) -> Result<usize, ApiError> {
        let _ = tokenizer;
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
        let cap = self.slots[idx].cap;
        if nt > cap {
            return Err(ApiError::exceed_context(format!(
                "prompt of {nt} tokens exceeds slot context of {cap}"
            )));
        }
        let (start, seq) = (self.slots[idx].start, self.slots[idx].seq);
        let (feed_from, _) = self.feed_span(idx, &job.input_ids);
        // Positions are cell indices in the shared arena: the slot's reservation
        // starts at `start`, so its row `i` is `start + i` (E1/E2).
        let positions: Vec<usize> = (feed_from..nt).map(|i| start + i).collect();
        let seq_ids = vec![seq; nt - feed_from];
        let batch = Batch::new(job.input_ids[feed_from..].to_vec(), positions, seq_ids);
        let live_on = crate::live::enabled();
        if live_on {
            crate::live::begin_phase("prefill");
        }
        let trace = std::env::var("MINFER_BATCH_TRACE").is_ok();
        let t0 = std::time::Instant::now();
        let last_logits =
            guarded_forward_batch(model, &batch, 1, self.n_ctx_total, &mut self.cache)?;
        if trace {
            eprintln!(
                "[batch] single prefill: slot {idx}, {} tokens, {:.0} ms",
                batch.len(),
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
                    positions.push(slot.start + run.current_pos);
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
                debug_assert_eq!(slot.cached_tokens.len(), pos);
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
        // `tok`'s row is written by the NEXT step's batch, at `current_pos`.
        *needs_forward = Some(tok);
        *current_pos += 1;
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
        let prompts: Vec<Vec<u32>> = [
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Italy is",
            "The capital of Spain is",
        ]
        .iter()
        .map(|p| tok.encode(p))
        .collect();
        let n_ctx = 512;
        let n_slots = 4;
        let max_tokens = 16;

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
}
