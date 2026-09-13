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
    n_ctx_slot: usize,
    input_ids: &[u32],
    params: &SamplingParams,
    tx: &mpsc::Sender<StreamEvent>,
) -> Result<(), ApiError> {
    if input_ids.is_empty() {
        return Err(ApiError::invalid_request("prompt tokenizes to nothing"));
    }
    generate_seq(
        model, tokenizer, cache, n_ctx_slot, input_ids, params, tx, None,
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
    let positions: Vec<usize> = (0..nt).collect();
    let special = model.special_tokens();

    // P3 live (server): phase/token/logits events for the web visualizer.
    let live_on = crate::live::enabled();

    // Prefill (n_out=1: only the LAST token's logits are returned and used for
    // the first sample — passing nt here would hand the sampler all nt rows and
    // corrupt the first sampled token).
    if live_on {
        crate::live::begin_phase("prefill");
    }
    let last_logits = guarded_forward(model, input_ids, &positions, 1, n_ctx_slot, cache)?;
    if live_on {
        crate::live::attach_step(&last_logits);
    }
    let mut logits = last_logits;
    let mut current_pos = nt;

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
        // Each request starts from a fresh KV/graph state: the graph path's
        // persistent KV regions were built for append-only sessions (the
        // conversation resets on any full re-render) — re-prefilling a
        // DIFFERENT prompt over the same regions leaves stale rows below the
        // new attention window (observed as cross-request contamination on
        // the plain path too, doc 97 §2). A fresh cache costs one ~1 ms
        // graph rebuild per request.
        slot.cache = GraphCache::new();
        let result = match slot_spec {
            Some(spec) => generate_spec(
                &*model,
                &tokenizer,
                &mut slot.cache,
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
                slot.n_ctx_slot,
                &job.input_ids,
                &job.params,
                &job.tx,
            ),
        };
        if let Err(e) = result {
            let _ = job.tx.blocking_send(StreamEvent::Err(e));
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
    fn stop_token_matches_eos_and_im_end() {
        let special = SpecialTokens {
            eos: 2,
            im_end: Some(7),
        };
        assert!(is_stop_token(2, &special));
        assert!(is_stop_token(7, &special));
        assert!(!is_stop_token(3, &special));
    }
}
