//! The single-shot generation core (#763 S6b): prefill + decode, moved out of
//! `src/main.rs` so the CLI and an embedding host run the same loop.
//!
//! The CLI keeps the presentation (prompt rendering, the think-block
//! highlighter, the stats lines); this module keeps the inference: the prefill,
//! the sampler pipeline (penalties, grammar, mirostat, the speculative round
//! loop), stop strings, the trace marks and the KV writes.

use std::time::Instant;

use rand::SeedableRng;

use crate::graph::cache::GraphCache;
use crate::models::ModelDef;
use crate::sampler::{self, SamplerConfig};
use crate::spec::{SpecEngine, SpecSampler, SpecStats};
use crate::tokenizer::Tokenizer;
use crate::trace;

/// What one single-shot generation needs.
pub struct GenerateParams<'a> {
    /// The already-tokenized (and, for a chat model, already-rendered) prompt.
    pub tokens: &'a [u32],
    pub n_ctx: usize,
    pub n_predict: usize,
    /// The whole sampler pipeline (penalties, grammar, mirostat, …).
    pub sampler: &'a SamplerConfig,
    pub seed: u64,
    pub stop_strings: &'a [String],
    /// `MINFER_TIMING=1`: count sampling and forward time.
    pub timing: bool,
    /// Speculative decoding's adaptive-depth line, for the caller's report.
    pub spec_adaptive: bool,
    pub spec_d_max: usize,
}

/// What the run produced, for the caller's report.
#[derive(Debug, Default)]
pub struct GenerateOutcome {
    pub generated: usize,
    pub prefill_tokens: usize,
    pub prefill_secs: f64,
    pub decode_secs: f64,
    pub stopped_by_eog: bool,
    pub stopped_by_string: bool,
    pub hit_n_predict: bool,
    /// A grammar refusal (the emitted text is a valid prefix).
    pub grammar_error: Option<String>,
    /// `(sample_secs, forward_secs, counted_tokens)` when `timing` was on.
    pub timing: Option<(f64, f64, usize)>,
    /// Speculative-decoding counters, when a draft engine was supplied.
    pub spec: Option<SpecStatsOut>,
}

/// The speculative counters the CLI's `[spec]` line reports.
#[derive(Debug, Clone, Copy)]
pub struct SpecStatsOut {
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub repairs: u64,
    pub d_sum: u64,
    pub generated: u64,
    pub adaptive: bool,
    pub d_max: usize,
}

/// One decoded token's bytes, through the **checked** entry point. A refusal
/// here is a loader contract violation, so it is loud but cannot abort a host:
/// the bytes are empty and the caller's stream simply ends the token.
fn decode_bytes(tokenizer: &Tokenizer, id: u32) -> Vec<u8> {
    tokenizer.decode_bytes(&[id])
}

/// Prefill `p.tokens`, then decode up to `p.n_predict` tokens, streaming the
/// answer through `emit`.
pub fn generate(
    model: &dyn ModelDef,
    tokenizer: &Tokenizer,
    mut spec: Option<&mut SpecEngine>,
    p: &GenerateParams<'_>,
    emit: &mut dyn FnMut(&[u8]),
    on_prefill: &mut dyn FnMut(&[f32], f64),
) -> GenerateOutcome {
    let mut target_cache = GraphCache::new();
    let mut stopped_by_eog = false;
    let mut stopped_by_string = false;
    let mut grammar_error: Option<String> = None;
    let mut spec_stats: Option<SpecStatsOut> = None;
    let infer_start = std::time::Instant::now();
    let positions: Vec<usize> = (0..p.tokens.len()).collect();
    // forward() computes logits for only the LAST n_out tokens (n_out=1 here:
    // single sequence, only the final token is sampled). llama.cpp does the same
    // via ggml_get_rows(inp_out_ids) at the last layer, shrinking the lm_head
    // to n_outputs rows — saves the full-nt output GEMM + logits download.
    // n_ctx (--n-ctx, default 4096) sizes the graph KV regions — NOT the
    // model's max_seq_len, which would allocate 12 GB+ and pay a first-submit
    // Metal tax (docs/PERF-QWEN3-4B-VS-LLAMACPP.md §2). It never shrinks below
    // the prompt length, and the model's forward clamps it to max_seq_len.
    // Computed ONCE so prefill and decode size the same KV regions.
    let ctx = p.n_ctx.max(p.tokens.len());
    // P2 trace (MINFER_TRACE=<path>): the scheduler records per-node data; the
    // CLI marks phase boundaries and attaches tokens/logits for the page.
    // `trace_on` is hoisted out of the decode loop (one env read per run).
    let trace_on = crate::trace::enabled();
    if trace_on {
        crate::trace::begin_phase("prefill");
    }
    let logits = if spec.is_some() {
        // Spec mode: drive the target through forward_graph_cached so its KV
        // lives in the cache the verify rounds reuse; the draft prefills the
        // same tokens into its own cache.
        let l = model.forward_graph_cached(&p.tokens, &positions, 1, ctx, &mut target_cache);
        spec.as_mut()
            .expect("spec engine checked above")
            .prefill(&p.tokens, ctx);
        l
    } else {
        model.forward(&p.tokens, &positions, 1, ctx)
    };
    let last_logits: Vec<f32> = logits;
    if trace_on {
        crate::trace::attach_step(&last_logits);
    }

    let prefill_time = infer_start.elapsed();
    on_prefill(&last_logits, prefill_time.as_secs_f64());

    // === Generate ===
    let mut logits = last_logits;
    if trace_on {
        crate::trace::begin_phase("decode");
    }
    let gen_start = std::time::Instant::now(); // pure-decode start (llama "Generation" caliber)
    let mut generated: Vec<u32> = Vec::new();
    let special = model.special_tokens();
    let mut current_pos = p.tokens.len();

    // Seeded RNG for reproducible sampling; recent-token window for the
    // penalties (llama.cpp repeat_last_n default = 64), seeded with the prompt
    // tail so the first generated tokens are penalized too.
    let mut rng = rand::rngs::StdRng::seed_from_u64(p.seed);
    const REPEAT_LAST_N: usize = 64;
    let mut prev_tokens = sampler::recent_window(&p.tokens, REPEAT_LAST_N);

    // Stop strings (byte-level, llama.cpp antiprompt): `full` accumulates ALL
    // generated bytes and `emitted` tracks what was already written, so a stop
    // string split across tokens is caught even if its earlier tokens were
    // already flushed (they stay in the terminal, like llama.cpp). Output is
    // byte-wise (never lossy), so split multi-byte chars pass through intact.
    let stop_bytes: Vec<Vec<u8>> = p
        .stop_strings
        .iter()
        .map(|s| s.as_bytes().to_vec())
        .collect();
    let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();
    let mut full: Vec<u8> = Vec::new();
    let mut emitted: usize = 0;

    // MINFER_TIMING=1: decompose the per-token wall-clock into CPU sampling vs
    // forward() (CPU encode + GPU exec + logits download). Debug tool only.
    let timing = p.timing;
    let (mut t_samp, mut t_fwd, mut n_tok) = (0.0f64, 0.0f64, 0usize);
    // t0/t1 are (re)assigned at the top of each loop iteration before use.
    let (mut t0, mut t1);

    // Gray-highlight the Qwen3 `<think>…</think>` reasoning block so it reads
    // apart from the answer. Only when stdout is a terminal (no ANSI in pipes);
    // MINFER_COLOR=1 forces it on (e.g. piping to less -R or a file).

    // === D5-R spec round loop ===
    // Each round emits 1..=d+1 tokens (draft-accepted prefix + the target
    // sampler's bonus); the per-token output machinery below is the same
    // sequence the non-spec loop runs, applied per emitted token.
    let sampler_cfg = p.sampler.clone();
    let mut mirostat = sampler::MirostatState::new(sampler_cfg.mirostat_tau);
    // F2: one automaton state per run, exactly like mirostat's `mu`.
    let mut grammar_state = sampler_cfg.grammar.as_ref().map(|g| g.state());
    if let Some(engine) = spec.as_mut() {
        let sparams = SpecSampler {
            cfg: sampler_cfg.clone(),
        };
        // Seed token: the prefill's last-row logits, sampled exactly like the
        // normal path — G1 token identity starts at token 0.
        let mut seed_logits = std::mem::take(&mut logits);
        let sampled = sampler::sample_with_config(
            &mut seed_logits,
            &sampler_cfg,
            &prev_tokens,
            &mut mirostat,
            &mut rng,
        );
        let mut next_token = sampled.token_id;
        let mut stop = false;
        if special.is_stop(next_token) {
            stop = true;
        } else {
            generated.push(next_token);
            prev_tokens.push(next_token);
            full.extend_from_slice(&decode_bytes(tokenizer, next_token));
            if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                full.truncate(cut);
                if cut > emitted {
                    emit(&full[emitted..]);
                    emitted = full.len();
                }
                stopped_by_string = true;
                stop = true;
            }
            if !stop && emitted < full.len() {
                emit(&full[emitted..]);
                emitted = full.len();
            }
            if trace_on {
                let text =
                    String::from_utf8_lossy(&decode_bytes(tokenizer, next_token)).into_owned();
                crate::trace::set_token(next_token, &text);
            }
        }
        while !stop && generated.len() < p.n_predict {
            let t0r = std::time::Instant::now();
            let round_toks = engine.round(
                &*model,
                &mut target_cache,
                next_token,
                current_pos,
                ctx,
                &sparams,
                &mut prev_tokens,
                &mut rng,
            );
            if timing {
                t_fwd += t0r.elapsed().as_secs_f64();
                n_tok += round_toks.len();
            }
            let mut consumed = 0usize;
            while consumed < round_toks.len() && generated.len() < p.n_predict {
                let tok = round_toks[consumed];
                if special.is_stop(tok) {
                    stop = true;
                    break;
                }
                generated.push(tok);
                full.extend_from_slice(&decode_bytes(tokenizer, tok));
                if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                    full.truncate(cut);
                    if cut > emitted {
                        emit(&full[emitted..]);
                        emitted = full.len();
                    }
                    stop = true;
                    break;
                }
                if emitted < full.len() {
                    emit(&full[emitted..]);
                    emitted = full.len();
                }
                if trace_on {
                    let text = String::from_utf8_lossy(&decode_bytes(tokenizer, tok)).into_owned();
                    crate::trace::set_token(tok, &text);
                }
                consumed += 1;
            }
            if stop {
                break;
            }
            // The round already pushed every emitted token into prev_tokens
            // (the accept loop's penalty windows depend on it). Bookkeeping
            // follows what was actually emitted: an n_predict cut mid-round
            // discards the tail (we exit right after, so the KV state that
            // would have served those tokens is never read).
            next_token = round_toks[consumed - 1];
            current_pos += consumed;
        }
        let st = &engine.stats;
        spec_stats = Some(SpecStatsOut {
            rounds: st.rounds,
            drafted: st.drafted,
            accepted: st.accepted,
            repairs: st.repairs,
            d_sum: st.d_sum,
            generated: generated.len() as u64,
            adaptive: p.spec_adaptive,
            d_max: p.spec_d_max,
        });
    }
    if spec.is_none() {
        while generated.len() < p.n_predict {
            t0 = std::time::Instant::now();
            let sampled = match sampler::sample_with_config_grammar(
                &mut logits,
                &sampler_cfg,
                &prev_tokens,
                &mut mirostat,
                &mut grammar_state,
                &mut rng,
            ) {
                Ok(s) => s,
                Err(e) => {
                    // The emitted text is a valid prefix; the caller renders the
                    // refusal (the CLI prints it exactly as before).
                    grammar_error = Some(e.to_string());
                    break;
                }
            };
            if timing {
                t_samp += t0.elapsed().as_secs_f64();
            }

            if special.is_stop(sampled.token_id) {
                break;
            }
            generated.push(sampled.token_id);
            trace::token_trace(current_pos, sampled.token_id);
            prev_tokens.push(sampled.token_id);
            if prev_tokens.len() > REPEAT_LAST_N {
                prev_tokens.drain(0..prev_tokens.len() - REPEAT_LAST_N);
            }

            // Stop-string detection on the FULL byte stream before emitting.
            full.extend_from_slice(&decode_bytes(tokenizer, sampled.token_id));
            if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                full.truncate(cut);
                if cut > emitted {
                    emit(&full[emitted..]);
                    emitted = full.len();
                }
                stopped_by_string = true;
                break;
            }
            if emitted < full.len() {
                emit(&full[emitted..]);
                emitted = full.len();
            }

            // forward() returns n_out*nv logits (n_out=1 for single-token decode,
            // exactly n_vocab), so move the Vec in place instead of copying 607 KB/token.
            if trace_on {
                let text = String::from_utf8_lossy(&decode_bytes(tokenizer, sampled.token_id))
                    .into_owned();
                crate::trace::set_token(sampled.token_id, &text);
            }
            t1 = std::time::Instant::now();
            logits = model.forward(&[sampled.token_id], &[current_pos], 1, ctx);
            if trace_on {
                crate::trace::attach_step(&logits);
            }
            if timing {
                t_fwd += t1.elapsed().as_secs_f64();
                n_tok += 1;
            }
            current_pos += 1;
        }
    }
    // Final flush of any bytes not yet written (incl. a dangling partial
    // <think> marker, emitted raw).
    if emitted < full.len() {
        emit(&full[emitted..]);
    }

    let decode_secs = gen_start.elapsed().as_secs_f64();
    GenerateOutcome {
        generated: generated.len(),
        prefill_tokens: p.tokens.len(),
        prefill_secs: prefill_time.as_secs_f64(),
        decode_secs,
        stopped_by_eog,
        stopped_by_string,
        hit_n_predict: generated.len() >= p.n_predict && !stopped_by_eog && !stopped_by_string,
        grammar_error,
        timing: timing.then_some((t_samp, t_fwd, n_tok)),
        spec: spec_stats,
    }
}
