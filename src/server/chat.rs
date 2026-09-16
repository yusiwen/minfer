//! Chat generation loop (worker side) + streaming events.
//! OPENAI-CHAT-API-PLAN.md §Streaming, §Slot Management, Implementation Plan
//! Phase 3/4: serial execution on a dedicated worker thread; per-request
//! `tokio::sync::mpsc` channels give natural backpressure; UTF-8-safe
//! incremental decoding and stop-string truncation happen here, before any
//! chunk is emitted.

use rand::SeedableRng;
use tokio::sync::mpsc;

use crate::graph::cache::GraphCache;
use crate::models::{ModelDef, SpecialTokens};
use crate::server::slot::{Slot, SlotState};
use crate::server::types::{ApiError, SamplingParams};
use crate::tokenizer::Tokenizer;

/// Per-request event stream pushed by the worker to the HTTP handler.
pub enum StreamEvent {
    /// A UTF-8-safe text chunk (never ends mid-character).
    Text(String),
    /// Generation finished: reason is "stop" or "length"; `tokens` is the
    /// exact completion-token count (for `usage`).
    Finish { reason: String, tokens: usize },
    /// Terminal error (ApiError carries status/type for the HTTP response).
    Err(ApiError),
}

/// One queued inference task. `input_ids` is the rendered+tokenized prompt
/// (the handler tokenizes so context-overflow checks happen before queuing).
pub struct Job {
    pub input_ids: Vec<u32>,
    pub params: SamplingParams,
    pub tx: mpsc::Sender<StreamEvent>,
}

fn is_stop_token(id: u32, special: &SpecialTokens) -> bool {
    id == special.eos || Some(id) == special.im_end
}

/// B2: length of the common token prefix of the slot's recorded sequence and
/// the incoming prompt.
///
/// Prefix reuse is safe *because of this comparison*: the slot reuses rows
/// `0..reuse`, and every one of them holds the same token the new prompt has at
/// that position — the contents were verified, not assumed. A mismatch (or an
/// empty record) yields 0, which means a full prefill from position 0.
fn common_prefix_len(cached: &[u32], prompt: &[u32]) -> usize {
    cached
        .iter()
        .zip(prompt.iter())
        .take_while(|(a, b)| a == b)
        .count()
}

/// B2: which slice of the prompt this request must feed, given how many tokens
/// the slot's KV already holds. Returns `(feed_from, feed_len)`.
///
/// Always at least the last token: its logits are what the sampler consumes, so
/// a fully-cached prompt still costs one row (a decode step rather than a
/// prefill).
fn prefill_span(nt: usize, reuse: usize) -> (usize, usize) {
    let feed_from = reuse.min(nt.saturating_sub(1));
    (feed_from, nt - feed_from)
}

/// Run one job on `slot` to completion, pushing events into `job.tx`.
///
/// Termination: EOS / im_end -> "stop"; stop string -> "stop" (truncated);
/// `max_tokens` or slot context full -> "length". The slot's `GraphCache` is
/// passed by the caller and is only ever touched here (serial worker), so KV
/// isolation is structural. The generated text/usage travels over the event
/// channel; this function only reports errors.
pub fn generate(
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    cache: &mut GraphCache,
    cached_tokens: &mut Vec<u32>,
    n_ctx_slot: usize,
    input_ids: &[u32],
    params: &SamplingParams,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<(), ApiError> {
    if input_ids.is_empty() {
        return Err(ApiError::invalid_request("prompt tokenizes to nothing"));
    }
    generate_seq(
        model,
        tokenizer,
        cache,
        cached_tokens,
        n_ctx_slot,
        input_ids,
        params,
        tx,
        None,
    )
}

/// doc 97: speculative variant — `spec` carries the slot's draft engine.
/// Mirrors `generate_seq` token-for-token (the identity contract) while a
/// round commits 1..=d+1 pre-sampled tokens: the round writes the seed's KV
/// row plus every batch row except the last (the next seed); `current_pos`
/// advances by the committed count per break kind. The per-token machinery
/// (EOG / stop strings / UTF-8 holdback / Text events) is byte-identical to
/// the sequential loop. Stop-string tokens are never committed; their
/// spec-written rows are stale and overwritten before they are read.
#[allow(clippy::too_many_arguments)]
pub fn generate_spec(
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    cache: &mut GraphCache,
    cached_tokens: &mut Vec<u32>,
    n_ctx_slot: usize,
    input_ids: &[u32],
    params: &SamplingParams,
    tx: &mpsc::Sender<StreamEvent>,
    spec: &mut crate::spec::SpecEngine,
) -> Result<(), ApiError> {
    if input_ids.is_empty() {
        return Err(ApiError::invalid_request("prompt tokenizes to nothing"));
    }
    // Each request starts its draft context from scratch: the draft's KV
    // regions keep the previous request's rows and the draft never re-prefills
    // the new prompt (it drafts from the seed), so stale rows would sit below
    // every round's attention window. Resetting is cheap (~1 ms rebuild) and
    // keeps requests independent.
    spec.reset_draft();
    generate_seq(
        model,
        tokenizer,
        cache,
        cached_tokens,
        n_ctx_slot,
        input_ids,
        params,
        tx,
        Some(spec),
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_seq(
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    cache: &mut GraphCache,
    cached_tokens: &mut Vec<u32>,
    n_ctx_slot: usize,
    input_ids: &[u32],
    params: &SamplingParams,
    tx: &mpsc::Sender<StreamEvent>,
    mut spec: Option<&mut crate::spec::SpecEngine>,
) -> Result<(), ApiError> {
    if input_ids.is_empty() {
        return Err(ApiError::invalid_request("prompt tokenizes to nothing"));
    }
    if input_ids.len() > n_ctx_slot {
        return Err(ApiError::exceed_context(format!(
            "prompt of {} tokens exceeds slot context of {}",
            input_ids.len(),
            n_ctx_slot
        )));
    }

    let nt = input_ids.len();
    let special = model.special_tokens();

    // B2: how much of this prompt the slot's KV rows already hold.
    //
    // `cached_tokens` names the tokens those rows were written for, and reuse
    // is gated on an exact prefix match — so every row attention will read has
    // been verified to hold the same token. That is what makes reuse safe
    // regardless of the historical staleness question (see the B1 note in
    // `worker_loop`), and it is now bitwise too: the CPU attention is
    // nt-invariant (roadmap §4 item 14), so feeding a suffix at its own
    // positions reproduces a single-shot prefill exactly
    // (`prefix_reuse_matches_a_full_prefill`).
    //
    // The speculative path neither reuses nor records: a verify round writes
    // rows past the committed tokens, so the "row i holds cached_tokens[i]"
    // invariant is not one this path maintains for free.
    // `MINFER_NO_PREFIX_REUSE=1` disables reuse entirely (the pre-B2 behaviour);
    // it is an A/B switch for the measurement, not a topology change, so it is
    // not part of any graph identity.
    let track =
        spec.is_none() && !std::env::var("MINFER_NO_PREFIX_REUSE").map_or(false, |v| v == "1");
    let reuse = if track {
        common_prefix_len(cached_tokens, input_ids)
    } else {
        0
    };
    // Always feed at least the last token: its logits are what the sampler
    // consumes, so a fully-cached prompt still costs one row.
    let (feed_from, _feed_len) = prefill_span(nt, reuse);
    let positions: Vec<usize> = (feed_from..nt).collect();

    // P3 live (server): phase/token/logits events for the web visualizer.
    let live_on = crate::live::enabled();

    // Prefill (n_out=1: only the LAST token's logits are returned and used for
    // the first sample — passing nt here would hand the sampler all nt rows and
    // corrupt the first sampled token).
    if live_on {
        crate::live::begin_phase("prefill");
    }
    let last_logits = guarded_forward(
        model,
        &input_ids[feed_from..],
        &positions,
        1,
        n_ctx_slot,
        cache,
    )?;
    if live_on {
        crate::live::attach_step(&last_logits);
    }
    let mut logits = last_logits;
    let mut current_pos = nt;
    // One line per request: makes the reuse (or its absence) observable in
    // production, and it is what B3's measurement reads.
    eprintln!(
        "[server] prefill fed {}/{} prompt tokens ({} reused from the slot's KV)",
        nt - feed_from,
        nt,
        feed_from
    );
    // Rows 0..nt now hold `input_ids` (the reused prefix because it matched,
    // the fed suffix because it was just written).
    cached_tokens.clear();
    if track {
        cached_tokens.extend_from_slice(input_ids);
    }
    debug_assert!(cached_tokens.is_empty() || cached_tokens.len() == current_pos);

    let mut rng = rand::rngs::StdRng::seed_from_u64(params.seed);
    const REPEAT_LAST_N: usize = 64;
    let mut prev_tokens = sampler_recent_window(input_ids, REPEAT_LAST_N);

    let stop_bytes: Vec<Vec<u8>> = params
        .stop_strings
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();

    // doc 97: the speculative seed carried across rounds (the previous
    // batch's last token); None after any turn-ending break.
    let mut spec_seed: Option<u32> = None;

    // `full` accumulates ALL generated bytes; `emitted` tracks how many of them
    // were already sent as Text events. Stop detection runs on `full` (llama.cpp
    // checks the whole generated text, so a stop string split across tokens is
    // caught even though its earlier tokens were already emitted — they are
    // simply left in the client's buffer, exactly like llama.cpp).
    let mut full: Vec<u8> = Vec::new();
    let mut emitted: usize = 0;
    let mut completion_tokens = 0usize;
    let mut finish_reason = "stop";
    let mut stopped_by_string = false;

    if live_on {
        crate::live::begin_phase("decode");
    }

    // Emit `full[emitted..end]` as one Text event (complete UTF-8 by caller).
    let emit = |full: &[u8],
                emitted: usize,
                end: usize,
                tx: &mpsc::Sender<StreamEvent>|
     -> Result<(), ApiError> {
        if end > emitted {
            let chunk = String::from_utf8_lossy(&full[emitted..end]).into_owned();
            if tx.blocking_send(StreamEvent::Text(chunk)).is_err() {
                return Err(ApiError::server("client disconnected"));
            }
        }
        Ok(())
    };

    loop {
        // max_tokens limit
        if params.max_tokens >= 0 && completion_tokens as i64 >= params.max_tokens {
            finish_reason = "length";
            break;
        }
        // slot context full (KV write positions must stay < n_ctx_slot)
        if current_pos >= n_ctx_slot {
            finish_reason = "length";
            break;
        }

        // === doc 97: speculative iteration (a sibling of the plain body
        // below, mirroring it token-for-token — the identity contract). ===
        if let Some(sp) = spec.as_deref_mut() {
            // Seed: sampled from the last-row logits ONCE (first iteration);
            // afterwards the previous batch's last token is the seed — it was
            // already committed by the batch machinery below. (The logits var
            // is never refreshed in spec mode, so re-sampling here would
            // re-emit the first seed every round.)
            let seed_id = match spec_seed.take() {
                Some(id) => id,
                None => {
                    let sampled = crate::sampler::sample_with_penalties(
                        &mut logits,
                        params.temp,
                        params.top_k,
                        params.top_p,
                        params.repeat_penalty,
                        params.frequency_penalty,
                        params.presence_penalty,
                        &prev_tokens,
                        &mut rng,
                    );
                    let id = sampled.token_id;
                    if is_stop_token(id, &special) {
                        // The turn ends here; the slot's KV has no
                        // cross-request continuity (rows 0.. are rewritten
                        // before they are read), so no EOG KV write is needed.
                        finish_reason = "stop";
                        break;
                    }
                    completion_tokens += 1;
                    full.extend_from_slice(&tokenizer.decode_bytes(&[id]));
                    if let Some(cut) = crate::sampler::match_stop_suffix(&full, &stop_refs) {
                        stopped_by_string = true;
                        if cut > emitted {
                            emit(&full, emitted, cut, tx)?;
                            emitted = cut;
                        }
                        break;
                    }
                    prev_tokens.push(id);
                    if prev_tokens.len() > REPEAT_LAST_N {
                        prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
                    }
                    let complete =
                        emitted + crate::tokenizer::complete_utf8_prefix_len(&full[emitted..]);
                    if complete > emitted {
                        emit(&full, emitted, complete, tx)?;
                        emitted = complete;
                    }
                    if live_on {
                        let text =
                            String::from_utf8_lossy(&tokenizer.decode_bytes(&[id])).into_owned();
                        crate::live::set_token(id, &text);
                    }
                    id
                }
            };
            let sparams = crate::spec::SpecSampler {
                temp: params.temp,
                top_k: params.top_k,
                top_p: params.top_p,
                repeat_penalty: params.repeat_penalty,
                frequency_penalty: params.frequency_penalty,
                presence_penalty: params.presence_penalty,
            };
            let toks = sp.round(
                &*model,
                cache,
                seed_id,
                current_pos,
                n_ctx_slot,
                &sparams,
                &mut prev_tokens,
                &mut rng,
            );
            let mut batch_break = false;
            for (i, &tok) in toks.iter().enumerate() {
                if params.max_tokens >= 0 && completion_tokens as i64 >= params.max_tokens {
                    // Cap hit mid-batch: toks[0..i] were committed and their
                    // KV rows written; the tail is discarded (its rows, if
                    // any, are stale and rewritten before they are read).
                    finish_reason = "length";
                    current_pos += 1 + i;
                    batch_break = true;
                    break;
                }
                if is_stop_token(tok, &special) {
                    // The turn ends here (no cross-request KV continuity —
                    // see the seed-EOG note above).
                    finish_reason = "stop";
                    current_pos += 1 + i;
                    batch_break = true;
                    break;
                }
                completion_tokens += 1;
                full.extend_from_slice(&tokenizer.decode_bytes(&[tok]));
                if let Some(cut) = crate::sampler::match_stop_suffix(&full, &stop_refs) {
                    stopped_by_string = true;
                    if cut > emitted {
                        emit(&full, emitted, cut, tx)?;
                        emitted = cut;
                    }
                    // toks[i] is NOT committed (stop strings are not part of
                    // the canonical text); committed slots = seed + toks[0..i].
                    current_pos += 1 + i;
                    batch_break = true;
                    break;
                }
                // prev_tokens: the round's accept loop already pushed every
                // emitted token (pushing again would duplicate entries and
                // skew the repeat penalty — same as the conversation loop).
                let complete =
                    emitted + crate::tokenizer::complete_utf8_prefix_len(&full[emitted..]);
                if complete > emitted {
                    emit(&full, emitted, complete, tx)?;
                    emitted = complete;
                }
                if live_on {
                    let text =
                        String::from_utf8_lossy(&tokenizer.decode_bytes(&[tok])).into_owned();
                    crate::live::set_token(tok, &text);
                }
            }
            if batch_break {
                // A mid-batch break (length cap / EOG / stop string) ended the
                // TURN — leave the outer loop. (The plain path's equivalents
                // break out directly; without this the loop would run another
                // round after the turn was already finished, re-sampling a
                // seed from the stale prefill logits.)
                break;
            }
            // Full batch committed: toks[len-1] is the next seed (committed by
            // the batch machinery, its row written by the next round).
            current_pos += toks.len();
            spec_seed = toks.last().copied();
            continue;
        }

        let sampled = crate::sampler::sample_with_penalties(
            &mut logits,
            params.temp,
            params.top_k,
            params.top_p,
            params.repeat_penalty,
            params.frequency_penalty,
            params.presence_penalty,
            &prev_tokens,
            &mut rng,
        );
        if is_stop_token(sampled.token_id, &special) {
            finish_reason = "stop";
            break;
        }
        completion_tokens += 1;
        prev_tokens.push(sampled.token_id);
        if prev_tokens.len() > REPEAT_LAST_N {
            prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
        }

        // Stop-string detection on the FULL byte stream before emitting.
        full.extend_from_slice(&tokenizer.decode_bytes(&[sampled.token_id]));
        if let Some(cut) = crate::sampler::match_stop_suffix(&full, &stop_refs) {
            stopped_by_string = true;
            // Truncate at the stop string. If the stop's start lies before the
            // already-emitted bytes, the emitted prefix stays (unavoidable);
            // otherwise emit the untruncated tail up to `cut`.
            if cut > emitted {
                emit(&full, emitted, cut, tx)?;
                emitted = cut;
            }
            break;
        }

        // Emit only the complete-UTF-8 prefix; hold back a split multi-byte char.
        let complete = emitted + crate::tokenizer::complete_utf8_prefix_len(&full[emitted..]);
        if complete > emitted {
            emit(&full, emitted, complete, tx)?;
            emitted = complete;
        }

        // Decode step.
        if live_on {
            let text =
                String::from_utf8_lossy(&tokenizer.decode_bytes(&[sampled.token_id])).into_owned();
            crate::live::set_token(sampled.token_id, &text);
        }
        logits = guarded_forward(
            model,
            &[sampled.token_id],
            &[current_pos],
            1,
            n_ctx_slot,
            cache,
        )?;
        if live_on {
            crate::live::attach_step(&logits);
        }
        current_pos += 1;
        if track {
            // Row `current_pos - 1` now holds this token.
            cached_tokens.push(sampled.token_id);
            debug_assert_eq!(cached_tokens.len(), current_pos);
        }
    }

    // Final flush: everything after `emitted` — but if we stopped on a stop
    // string, nothing beyond `cut` (== full.len() after truncation below) may
    // leak; truncating `full` to `cut` makes the flush safe in both cases.
    if stopped_by_string {
        // `full` still holds bytes past the stop; drop them.
        if emitted < full.len() {
            // cut was captured above; recompute it the same way.
            if let Some(cut) = crate::sampler::match_stop_suffix(&full, &stop_refs) {
                full.truncate(cut);
            }
            if emitted < full.len() {
                emit(&full, emitted, full.len(), tx)?;
            }
        }
    } else if emitted < full.len() {
        emit(&full, emitted, full.len(), tx)?;
    }
    let _ = tx.blocking_send(StreamEvent::Finish {
        reason: finish_reason.to_string(),
        tokens: completion_tokens,
    });
    if live_on {
        crate::live::finish(
            finish_reason,
            completion_tokens,
            &String::from_utf8_lossy(&full),
        );
    }

    Ok(())
}

/// `forward_graph_cached` wrapped so an unexpected panic (e.g. an internal
/// invariant) becomes a 500 instead of killing the worker thread.
fn guarded_forward(
    model: &dyn ModelDef,
    tokens: &[u32],
    positions: &[usize],
    n_out: usize,
    n_ctx: usize,
    cache: &mut GraphCache,
) -> Result<Vec<f32>, ApiError> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        model.forward_graph_cached(tokens, positions, n_out, n_ctx, cache)
    }))
    .map_err(|_| ApiError::server("inference panicked"))
}

/// Run one job with panic isolation. Returns `true` when the job completed —
/// including a normal `Err`, which is forwarded to `tx` — and `false` when it
/// panicked, which is reported to `tx` as a 500 instead of unwinding the worker
/// thread (see the call site in `worker_loop`).
fn run_job_isolated<F>(tx: &mpsc::Sender<StreamEvent>, f: F) -> bool
where
    F: FnOnce() -> Result<(), ApiError>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            let _ = tx.blocking_send(StreamEvent::Err(e));
            true
        }
        Err(payload) => {
            // `&*payload`, not `&payload`: `Box<dyn Any + Send>` is itself an
            // `Any`, so passing the Box by reference would downcast against the
            // Box type and always report a non-string payload.
            let msg = panic_message(&*payload);
            eprintln!(
                "[server] job panicked: {msg} — the worker thread survives and the slot is released"
            );
            let _ = tx.blocking_send(StreamEvent::Err(ApiError::server(format!(
                "internal error: {msg}"
            ))));
            false
        }
    }
}

/// Human-readable panic payload. `panic!` with a literal or a formatted string
/// yields `&'static str` / `String`; anything else gets a generic label.
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "non-string panic payload".to_string()
    }
}

/// Worker thread: drains the job queue serially, one slot at a time.
/// Busy slots defer naturally — the queue is unbounded (llama.cpp semantics).
pub fn worker_loop(
    model: Box<dyn ModelDef>,
    tokenizer: Tokenizer,
    mut slots: Vec<Slot>,
    mut job_rx: mpsc::Receiver<Job>,
    spec_cfg: Option<crate::spec::SpecConfig>,
) {
    // doc 97: one draft engine per slot (each slot's draft KV is isolated
    // exactly like its target KV). The draft model is small (0.5B-class) and
    // loaded once at worker startup.
    let mut slot_specs: Vec<Option<crate::spec::SpecEngine>> = slots
        .iter()
        .map(|_| match &spec_cfg {
            Some(cfg) => crate::spec::SpecEngine::new(cfg, &tokenizer, model.n_vocab()).ok(),
            None => None,
        })
        .collect();
    loop {
        let Some(job) = job_rx.blocking_recv() else {
            break; // all senders dropped: server shutting down
        };
        let Some(slot_idx) = slots.iter().position(|s| s.state == SlotState::Idle) else {
            let _ = job
                .tx
                .blocking_send(StreamEvent::Err(ApiError::unavailable("no idle slot")));
            continue;
        };
        let slot = &mut slots[slot_idx];
        let slot_spec = slot_specs[slot_idx].as_mut();
        slot.state = SlotState::Processing;
        // B2: the slot KEEPS its cache and its `cached_tokens` record across
        // requests; `generate_seq` reuses the KV only when the new prompt
        // starts with exactly the recorded sequence, and prefills from position
        // 0 otherwise. The per-request reset this replaces came from commit
        // 39eceaa (doc 97), whose message described the cross-request bug as
        // "stale rows inside the new attention window". B1 tested that
        // mechanism directly (`reused_cache_across_prompts_matches_a_fresh_cache`
        // in `models/qwen2/graph.rs`: A→B, B→A, prefill+decode→B, each compared
        // bitwise against a virgin cache — all agree), so the reset was guarding
        // against something that does not happen on the plain path; the original
        // bug is consistent with the process-global cache the OpenAI plan's
        // revision notes record, which per-slot caches already fixed.
        //
        // `cached_tokens` is cleared on every error below: a failed request may
        // have written part of a row, and claiming otherwise is the one way this
        // could silently go wrong. The fused GPU stores (FusedQKV /
        // FusedQkvNorm / QkvBiasRopeStore) write K/V in-kernel and are
        // unverified on this box (A0: CUDA compile-only, Metal not built), so
        // the reuse gate stays conservative.
        //
        // Panic isolation for the WHOLE job, not just the forward call.
        // `guarded_forward` already contains a panic inside
        // `forward_graph_cached`, but the speculative path calls both models'
        // forwards directly (`spec.rs`), and the tokenizer, sampler, stop-string
        // and streaming paths are unguarded. Without this net one panic unwinds
        // this thread: `job_rx` is dropped, every later request is rejected, and
        // the requests already queued lose their event sender (an empty 200
        // instead of an error).
        let completed = run_job_isolated(&job.tx, || match slot_spec {
            Some(spec) => generate_spec(
                &*model,
                &tokenizer,
                &mut slot.cache,
                &mut slot.cached_tokens,
                slot.n_ctx_slot,
                &job.input_ids,
                &job.params,
                &job.tx,
                spec,
            ),
            None => generate(
                &*model,
                &tokenizer,
                &mut slot.cache,
                &mut slot.cached_tokens,
                slot.n_ctx_slot,
                &job.input_ids,
                &job.params,
                &job.tx,
            ),
        });
        // A panic may have written part of a KV row, so the record is no longer
        // trustworthy: drop it and make the next request prefill from position 0.
        // A *completed* job needs no such care — `generate_seq` only records a
        // token after its row was written, and returns early (before touching the
        // KV) for a rejected request, leaving the previous record valid.
        if !completed {
            slot.cached_tokens.clear();
        }
        slot.state = SlotState::Idle;
    }
}

fn sampler_recent_window(tokens: &[u32], last_n: usize) -> Vec<u32> {
    crate::sampler::recent_window(tokens, last_n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefill_span_always_feeds_the_last_token() {
        assert_eq!(prefill_span(10, 0), (0, 10), "cold slot: full prefill");
        assert_eq!(prefill_span(10, 4), (4, 6), "reuse a 4-token prefix");
        assert_eq!(
            prefill_span(10, 10),
            (9, 1),
            "fully cached: still one row for the logits"
        );
        assert_eq!(prefill_span(1, 0), (0, 1));
        assert_eq!(prefill_span(1, 1), (0, 1));
        assert_eq!(
            prefill_span(10, 99),
            (9, 1),
            "reuse cannot exceed the prompt"
        );
    }

    #[test]
    fn common_prefix_len_finds_the_exact_match() {
        assert_eq!(common_prefix_len(&[], &[1, 2, 3]), 0, "nothing recorded");
        assert_eq!(common_prefix_len(&[1, 2], &[]), 0, "empty prompt");
        assert_eq!(common_prefix_len(&[1, 2], &[1, 2]), 2, "identical");
        assert_eq!(
            common_prefix_len(&[1, 2], &[1, 2, 3]),
            2,
            "the record is a true prefix of the prompt"
        );
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 9]), 1, "diverges early");
        assert_eq!(
            common_prefix_len(&[5], &[1, 2, 3]),
            0,
            "a different first token must force a full prefill"
        );
    }

    #[test]
    fn stop_token_matches_eos_and_im_end() {
        let special = SpecialTokens {
            eos: 2,
            im_end: Some(7),
        };
        assert!(is_stop_token(2, &special));
        assert!(is_stop_token(7, &special));
        assert!(!is_stop_token(3, &special));
    }

    /// A panicking job must become a 500 on that request's stream — and the
    /// worker must live on. Before the fix the panic unwound `worker_loop`, so
    /// nothing was ever sent and the whole server degraded.
    #[test]
    fn isolated_job_turns_a_panic_into_an_error_event() {
        let (tx, mut rx) = mpsc::channel(4);
        let survived = run_job_isolated(&tx, || panic!("boom"));
        assert!(
            !survived,
            "a panicking job must report that it did not survive"
        );
        match rx.try_recv() {
            Ok(StreamEvent::Err(e)) => {
                let body = e.json();
                assert!(body.contains("boom"), "panic message lost: {body}");
                assert!(body.contains("server_error"), "wrong class: {body}");
            }
            _ => panic!("expected an Err event for the panicking job"),
        }
    }

    #[test]
    fn isolated_job_forwards_a_normal_error() {
        let (tx, mut rx) = mpsc::channel(4);
        assert!(run_job_isolated(&tx, || Err(ApiError::server("nope"))));
        match rx.try_recv() {
            Ok(StreamEvent::Err(e)) => assert!(e.json().contains("nope")),
            _ => panic!("expected the error to be forwarded"),
        }
    }

    #[test]
    fn isolated_job_passes_success_through_silently() {
        let (tx, mut rx) = mpsc::channel(4);
        assert!(run_job_isolated(&tx, || Ok(())));
        assert!(rx.try_recv().is_err(), "a clean job emits no error event");
    }
}
