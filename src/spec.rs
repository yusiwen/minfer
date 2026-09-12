//! D5-R stage 1 — minimal greedy speculative decoding (`draft-simple`).
//!
//! docs/SPECULATIVE-DECODING-PLAN.md: one round = d sequential draft-model
//! forwards + ONE target verify forward at `nt = d+1` (`n_out = d+1`) through
//! the existing `forward_graph_cached` primitive, then accept-while-equal
//! against the target's own sampler chain. KV "rollback" is free by
//! construction: both graphs write KV rows in place and positions are data,
//! so rejected slots are simply overwritten by the next round (never read —
//! causal attention only touches rows ≤ the row being written).
//!
//! Correctness gate (plan G1): every emitted token is the target sampler's
//! OWN decision at that position — an accepted draft token survives only
//! because it EQUALS the target's sample — so at temp=0 the spec stream is
//! token-identical to the non-spec path.
//!
//! Round state invariant: `next_token` goes at `pos`; both KV caches are
//! valid through `pos - 1`. The round emits the tokens for `pos+1..pos+m`
//! (m = accepted + 1) and leaves the next round's `next_token` at `pos+m`.

use crate::graph::cache::GraphCache;
use crate::models::ModelDef;
use crate::sampler;
use crate::tokenizer::Tokenizer;
use rand::rngs::StdRng;
use rand::Rng;

/// CLI inputs (`--spec-draft <model>`, `--spec-draft-n <d>`).
pub struct SpecConfig {
    pub draft_path: String,
    pub draft_n: usize,
}

/// Per-round counters (reported on stderr after generation).
#[derive(Default)]
pub struct SpecStats {
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub repairs: u64,
}

/// Sampler inputs mirrored from GenParams so spec.rs does not depend on
/// main.rs's private struct.
pub struct SpecSampler {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
}

pub struct SpecEngine {
    draft: Box<dyn ModelDef>,
    draft_cache: GraphCache,
    draft_n: usize,
    pub stats: SpecStats,
}

impl SpecEngine {
    /// Load the draft model and gate on tokenizer compatibility (the D5
    /// analysis §4.2 hard gate: vocab size delta ≤ 128, same BOS/EOS, token
    /// texts equal). The draft proposes token ids the target must accept
    /// verbatim — a mismatch is silently-wrong output, so this fails loudly.
    pub fn new(
        cfg: &SpecConfig,
        target_tokenizer: &Tokenizer,
        target_vocab: usize,
    ) -> Result<Self, String> {
        let draft_path = crate::download::resolve(&cfg.draft_path)
            .map_err(|e| format!("spec draft model: {e}"))?
            .to_string_lossy()
            .into_owned();
        let gguf = crate::gguf::load_gguf_model(std::path::Path::new(&draft_path))
            .ok_or_else(|| format!("spec draft: failed to parse GGUF: {draft_path}"))?;
        // Namespaced load: the CUDA/Metal weight registries are process-global
        // and name-keyed, so the draft's tensors (same GGUF names as the
        // target's) must not collide — an unprefixed second load silently
        // fails the target's all-or-nothing CUDA check and drops it to CPU.
        let draft = crate::models::load_model_ns(&gguf, "draft.")
            .ok_or_else(|| "spec draft: load_model failed".to_string())?;
        let dtok = Tokenizer::load(&gguf.parts[0].ctx);
        let (tv, dv) = (target_vocab, dtok.vocab_size());
        if tv.abs_diff(dv) > 128 {
            return Err(format!(
                "spec draft: vocab size mismatch (target {tv} vs draft {dv}; max delta 128)"
            ));
        }
        if dtok.bos_token != target_tokenizer.bos_token
            || dtok.eos_token != target_tokenizer.eos_token
        {
            return Err(format!(
                "spec draft: special-token mismatch (draft bos {} eos {} vs target bos {} eos {})",
                dtok.bos_token,
                dtok.eos_token,
                target_tokenizer.bos_token,
                target_tokenizer.eos_token
            ));
        }
        for i in 0..tv.min(dv) {
            if dtok.id_to_token[i] != target_tokenizer.id_to_token[i] {
                return Err(format!(
                    "spec draft: token text mismatch at id {i}: '{}' vs '{}'",
                    dtok.id_to_token[i], target_tokenizer.id_to_token[i]
                ));
            }
        }
        Ok(Self {
            draft,
            draft_cache: GraphCache::new(),
            draft_n: cfg.draft_n,
            stats: SpecStats::default(),
        })
    }

    /// Draft-side prefill: same prompt tokens/positions as the target, so the
    /// draft KV ends at the same position and round drafting stays contiguous.
    pub fn prefill(&mut self, tokens: &[u32], n_ctx: usize) {
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let _ =
            self.draft
                .forward_graph_cached(tokens, &positions, 1, n_ctx, &mut self.draft_cache);
    }

    /// One speculative round. Returns 1..=d+1 sampled tokens for positions
    /// `pos+1..`; the caller emits them in order through its existing stop
    /// machinery. `prev_tokens` is extended IN HERE — the accept loop samples
    /// lazily so each row's penalty window sees exactly what the serial path
    /// would see at that position.
    #[allow(clippy::too_many_arguments)]
    pub fn round(
        &mut self,
        target: &dyn ModelDef,
        target_cache: &mut GraphCache,
        next_token: u32,
        pos: usize,
        n_ctx: usize,
        s: &SpecSampler,
        prev_tokens: &mut Vec<u32>,
        rng: &mut StdRng,
    ) -> Vec<u32> {
        // Clamp the draft depth to the KV horizon; at the context edge fall
        // back to a plain single-token step (same sampler, one row).
        let d = self.draft_n.min(n_ctx.saturating_sub(pos + 1));
        let debug = std::env::var("MINFER_SPEC_DEBUG").map_or(false, |v| v == "1" || v == "2");
        if d == 0 {
            let mut row =
                target.forward_graph_cached(&[next_token], &[pos], 1, n_ctx, target_cache);
            let t = sampler::sample_with_penalties(
                &mut row,
                s.temp,
                s.top_k,
                s.top_p,
                s.repeat_penalty,
                s.frequency_penalty,
                s.presence_penalty,
                prev_tokens,
                rng,
            );
            if debug {
                eprintln!(
                    "[spec] r{} pos={pos} next={next_token} d=0 -> [{}]",
                    self.stats.rounds, t.token_id
                );
            }
            prev_tokens.push(t.token_id);
            self.stats.rounds += 1;
            return vec![t.token_id];
        }

        // Draft phase: d forwards → d proposals for positions pos+1..pos+d.
        // Raw draft argmax (no penalties — penalties are target-sampler
        // semantics; a penalized proposal would only fail to match).
        let mut tok = next_token;
        let mut proposals = Vec::with_capacity(d);
        for k in 0..d {
            let logits = self.draft.forward_graph_cached(
                &[tok],
                &[pos + k],
                1,
                n_ctx,
                &mut self.draft_cache,
            );
            tok = argmax(&logits);
            proposals.push(tok);
        }
        self.stats.drafted += d as u64;

        // Verify: one nt=d+1 forward with logits on every row. Row i predicts
        // position pos+i+1.
        let mut rows = Vec::with_capacity(d + 1);
        rows.push(next_token);
        rows.extend_from_slice(&proposals);
        let positions: Vec<usize> = (pos..=pos + d).collect();
        let logits = target.forward_graph_cached(&rows, &positions, d + 1, n_ctx, target_cache);
        if debug {
            eprintln!(
                "[spec] r{} pos={pos} next={next_token} d={d} rows={rows:?} prop={proposals:?}",
                self.stats.rounds
            );
        }

        // Lazy accept: sample row i only after tokens 1..i are committed.
        let trace = std::env::var("MINFER_SPEC_DEBUG").map_or(0u8, |v| match v.as_str() {
            "2" => 2,
            _ => 1,
        });
        let (emitted, accepted) = accept_loop(
            &logits,
            &proposals,
            s,
            prev_tokens,
            rng,
            trace,
            self.stats.rounds,
        );
        self.stats.accepted += accepted as u64;
        self.stats.rounds += 1;

        // Full accept: the draft never wrote row pos+d (u_d was proposed but
        // not forwarded) — repair it so the next round's draft KV stays
        // contiguous. Partial accepts leave no hole: the draft KV already
        // reaches pos+k for every k ≤ d-1.
        if emitted.len() == d + 1 {
            let _ = self.draft.forward_graph_cached(
                &[proposals[d - 1]],
                &[pos + d],
                1,
                n_ctx,
                &mut self.draft_cache,
            );
            self.stats.repairs += 1;
        }
        emitted
    }
}

/// Lazy accept loop over the verify rows, extracted so the accept rule is
/// unit-testable with synthetic logits (plan gate G1a). Returns the emitted
/// token sequence (1..=d+1 entries) and the accepted-proposal count.
///
/// For row i (which predicts position pos+i+1): sample the target's own chain
/// (`sample_with_penalties` — the same sampler the non-spec path runs, with
/// `prev_tokens` including every token committed so far, lazily extended per
/// accepted proposal). Row i's sample is compared against proposals[i] only
/// when a proposal exists (i < d); the first mismatch — or the last row's
/// sample — becomes the bonus token that ends the round.
pub fn accept_loop<R: Rng>(
    logits: &[f32],
    proposals: &[u32],
    s: &SpecSampler,
    prev_tokens: &mut Vec<u32>,
    rng: &mut R,
    trace: u8,
    round: u64,
) -> (Vec<u32>, usize) {
    let d = proposals.len();
    let nv = logits.len() / (d + 1);
    let mut emitted = Vec::with_capacity(d + 1);
    let mut accepted = 0usize;
    for i in 0..=d {
        let mut row = logits[i * nv..(i + 1) * nv].to_vec();
        let (b1, b2) = top2(&row);
        let t = sampler::sample_with_penalties(
            &mut row,
            s.temp,
            s.top_k,
            s.top_p,
            s.repeat_penalty,
            s.frequency_penalty,
            s.presence_penalty,
            prev_tokens,
            rng,
        );
        if trace >= 2 {
            eprintln!(
                "[spec] r{round} row {i}: sample={} draft={} margin {:.4}",
                t.token_id,
                proposals.get(i).copied().unwrap_or(0),
                b1 - b2
            );
        }
        if i < d && t.token_id == proposals[i] {
            emitted.push(proposals[i]);
            prev_tokens.push(proposals[i]);
            accepted += 1;
        } else {
            // Mismatch (or the last row): this sample IS the next token.
            if trace >= 1 && i < d {
                eprintln!(
                    "[spec] r{round} REJECT i={i}: draft={} sample={} margin {:.4}",
                    proposals[i],
                    t.token_id,
                    b1 - b2
                );
            }
            emitted.push(t.token_id);
            prev_tokens.push(t.token_id);
            break;
        }
    }
    (emitted, accepted)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > bv {
            bv = v;
            best = i;
        }
    }
    best as u32
}

/// Two largest logits (for the MINFER_SPEC_DEBUG margin diagnostic).
fn top2(logits: &[f32]) -> (f32, f32) {
    let mut b1 = f32::NEG_INFINITY;
    let mut b2 = f32::NEG_INFINITY;
    for &v in logits {
        if v > b1 {
            b2 = b1;
            b1 = v;
        } else if v > b2 {
            b2 = v;
        }
    }
    (b1, b2)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn sampler() -> SpecSampler {
        SpecSampler {
            temp: 0.0,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
        }
    }

    /// Rows of d+1 tokens x nv logits; row i's argmax is exactly wants[i].
    fn rows(wants: &[u32], nv: usize) -> Vec<f32> {
        let mut v = vec![0.0f32; wants.len() * nv];
        for (i, &w) in wants.iter().enumerate() {
            v[i * nv + w as usize] = 2.0;
        }
        v
    }

    #[test]
    fn full_accept_emits_d_plus_1() {
        let mut prev = vec![7u32];
        let mut rng = StdRng::seed_from_u64(42);
        let logits = rows(&[10, 11, 12], 32);
        // proposals match rows 0 and 1 (d=2); row 2 is the bonus.
        let (emitted, accepted) =
            accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0);
        assert_eq!(emitted, vec![10, 11, 12]);
        assert_eq!(accepted, 2);
        // prev_tokens grew by the two accepted proposals + the bonus.
        assert_eq!(prev, vec![7, 10, 11, 12]);
    }

    #[test]
    fn reject_at_row1_emits_prefix_plus_bonus() {
        let mut prev = vec![];
        let mut rng = StdRng::seed_from_u64(42);
        let logits = rows(&[10, 31, 12], 32); // row 1 disagrees with proposal 11
        let (emitted, accepted) =
            accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0);
        assert_eq!(emitted, vec![10, 31]);
        assert_eq!(accepted, 1);
        assert_eq!(prev, vec![10, 31]);
    }

    #[test]
    fn reject_at_row0_emits_bonus_only() {
        let mut prev = vec![];
        let mut rng = StdRng::seed_from_u64(42);
        let logits = rows(&[31, 11, 12], 32);
        let (emitted, accepted) =
            accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0);
        assert_eq!(emitted, vec![31]);
        assert_eq!(accepted, 0);
    }

    /// The lazy rule: row 1's penalty window must include the accepted u1
    /// from row 0 — repeat-penalty demotes a repeated token so the bonus
    /// flips to the runner-up, which the serial path would also pick.
    #[test]
    fn lazy_penalty_window_sees_intra_round_tokens() {
        let mut prev = vec![];
        let mut rng = StdRng::seed_from_u64(42);
        let mut s = sampler();
        s.repeat_penalty = 1.5;
        // nv=4: row 0 argmax = 1 (accepted u1=1); row 1 raw argmax = 1 again,
        // runner-up = 2. With u1=1 in the penalty window, the penalized
        // row-1 sample must be 2, not 1.
        let mut logits = rows(&[1, 1], 4);
        logits[1 * 4 + 2] = 1.4; // runner-up wins once the 1.5 penalty demotes the repeat
        let (emitted, accepted) = accept_loop(&logits, &[1], &s, &mut prev, &mut rng, 0, 0);
        assert_eq!(accepted, 1);
        assert_eq!(emitted, vec![1, 2]);
        assert_eq!(prev, vec![1, 2]);
    }
}
