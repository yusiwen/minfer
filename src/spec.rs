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
    /// static depth, or the depth CAP when `adaptive` is set
    pub draft_n: usize,
    /// doc 95: adaptive depth controller (`--spec-draft-adaptive`)
    pub adaptive: bool,
}

/// Per-round counters (reported on stderr after generation).
#[derive(Default)]
pub struct SpecStats {
    pub rounds: u64,
    pub drafted: u64,
    pub accepted: u64,
    pub repairs: u64,
    /// sum of chosen depths (doc 95: dmean = d_sum / drafted rounds)
    pub d_sum: u64,
}

/// doc 95: adaptive draft-depth controller. Per-depth acceptance EWMA
/// (p_k = P(draft k accepted | drafts 1..k-1 accepted)) plus an online
/// verify-cost curve per nt and a draft-cost EWMA; each round picks the d
/// maximizing projected throughput E[tokens]/(V(d+1) + d*t_draft).
///
/// Greedy-identity note (doc 94): the verify forward is bitwise
/// position-invariant in nt and the penalty window is d-independent, so
/// varying d changes speed only — the emitted token stream stays
/// byte-identical to sequential decode.
pub struct AdaptiveD {
    d_max: usize,
    /// acceptance observations per depth: successes / trials (index 0 = depth 1)
    succ: Vec<u32>,
    n: Vec<u32>,
    /// verify-round wall time per nt: min over the last AD_COST_WINDOW
    /// samples (index 0 = nt 2), ms
    v: Vec<f64>,
    /// raw verify samples per nt (the min window)
    v_win: Vec<Vec<f64>>,
    /// draft per-token wall time EWMA, ms
    t_draft: f64,
    t_draft_win: Vec<f64>,
    rounds: u64,
    /// incumbent pick (hysteresis baseline; 0 = none yet)
    last_pick: usize,
    /// incumbent's score at its last decision
    last_score: f64,
    /// chosen-d histogram (index 0 = d 1) for the stats line
    d_counts: Vec<u64>,
}

/// Cost estimator: min over the last AD_COST_WINDOW samples — a deterministic
/// kernel's steady-state time; the min rejects capture/warm-up spikes on the
/// first execution after a depth switch (an EWMA would let those spikes
/// poison the cost curve and lock the controller shallow).
const AD_COST_WINDOW: usize = 4;
/// Beta prior for an observed depth's acceptance (Laplace smoothing): with a
/// handful of trials the mean stays near the prior and converges within a few
/// rounds — no EWMA alpha to tune, no separate exploration schedule.
const AD_BETA_A: f64 = 0.5;
const AD_BETA_B: f64 = 1.5;
const AD_PRIOR_P: f64 = 0.75;
const AD_PRIOR_T_DRAFT: f64 = 1.7;
/// A depth switch requires this score advantage — prevents the pick from
/// collapsing on early unlucky deep rounds (a 3-trial Laplace mean can read
/// a true-0.85 depth as 0.14; without hysteresis the pick then starves that
/// depth forever).
const AD_SWITCH_MARGIN: f64 = 0.10;
/// Trials before a depth's beta mean is trusted; below this the estimate is
/// floored at the depth-1 rate (optimism keeps the depth reachable).
const AD_MIN_TRIALS: u32 = 8;

/// Min over a bounded trailing window — the steady-state estimator for a
/// deterministic cost (rejects capture/warm-up spikes).
fn min_window_update(sample: f64, win: &mut Vec<f64>) -> f64 {
    win.push(sample);
    if win.len() > AD_COST_WINDOW {
        win.remove(0);
    }
    win.iter().cloned().fold(f64::INFINITY, f64::min)
}

impl AdaptiveD {
    pub fn new(d_max: usize) -> Self {
        let d_max = d_max.max(1);
        // Verify-cost priors: the doc-94 measured curve at 14B/GB10
        // (C_T(1)=39.7, 3=48.6, 5=56.2, 9=73.0 ms) interpolated per nt.
        // Machine/model-specific priors — the online EWMA corrects them for
        // every visited depth as rounds accumulate.
        let pts = [(1usize, 39.7f64), (3, 48.6), (5, 56.2), (9, 73.0)];
        let prior = |nt: usize| -> f64 {
            if nt <= 1 {
                return pts[0].1;
            }
            for w in pts.windows(2) {
                let (a, b) = (w[0].0, w[1].0);
                if nt <= b {
                    let t = (nt - a) as f64 / (b - a) as f64;
                    return w[0].1 + t * (w[1].1 - w[0].1);
                }
            }
            let (a, b) = (pts[pts.len() - 2], pts[pts.len() - 1]);
            let slope = (b.1 - a.1) / (b.0 - a.0) as f64;
            b.1 + slope * (nt - b.0) as f64
        };
        Self {
            d_max,
            succ: vec![0; d_max],
            n: vec![0; d_max],
            v: (2..=d_max + 1).map(|nt| prior(nt)).collect(),
            v_win: (2..=d_max + 1).map(|_| Vec::new()).collect(),
            t_draft: AD_PRIOR_T_DRAFT,
            t_draft_win: Vec::new(),
            rounds: 0,
            last_pick: 0,
            last_score: 0.0,
            d_counts: vec![0; d_max],
        }
    }

    pub fn observe_draft(&mut self, total_ms: f64, n: usize) {
        if n == 0 {
            return;
        }
        let per = total_ms / n as f64;
        self.t_draft = min_window_update(per, &mut self.t_draft_win);
    }

    pub fn observe_verify(&mut self, nt: usize, ms: f64) {
        if let Some(idx) = nt.checked_sub(2) {
            if idx < self.v.len() {
                self.v[idx] = min_window_update(ms, &mut self.v_win[idx]);
            }
        }
    }

    pub fn observe_round(&mut self, d_r: usize, accepted: usize) {
        for j in 0..d_r.min(self.succ.len()) {
            if j < accepted {
                self.succ[j] += 1;
            }
            self.n[j] = self.n[j].saturating_add(1);
        }
    }

    /// Pick the draft depth for this round (0 when the KV horizon is spent —
    /// the caller's d==0 fallback path). Exploration: the first
    /// AD_EXPLORE_ROUNDS rounds and every AD_EXPLORE_EVERY-th round run at
    /// d_max so deeper-depth estimates stay live.
    pub fn pick(&mut self, horizon: usize) -> usize {
        let d_cap = self.d_max.min(horizon);
        if d_cap == 0 {
            return 0;
        }
        self.rounds += 1;
        // Acceptance estimate per depth: Laplace-smoothed mean when observed;
        // an UNOBSERVED depth inherits the depth-1 rate (optimism that is
        // right for the flat acceptance curve of code-heavy text and merely
        // optimistic for collapsing prose — the first deep pick then observes
        // and corrects). Exploration is therefore emergent: a high depth-1
        // rate pulls unobserved depths up until real samples say otherwise.
        let p1 = if self.n[0] > 0 {
            (self.succ[0] as f64 + AD_BETA_A) / (self.n[0] as f64 + AD_BETA_B)
        } else {
            AD_PRIOR_P
        };
        let rate = |j: usize| -> f64 {
            if self.n[j] == 0 {
                p1
            } else {
                let mean = (self.succ[j] as f64 + AD_BETA_A) / (self.n[j] as f64 + AD_BETA_B);
                // optimism floor: an early-unlucky depth stays reachable
                mean.max(if self.n[j] < AD_MIN_TRIALS { p1 } else { mean })
            }
        };
        let mut best_d = 1usize;
        let mut best_score = f64::NEG_INFINITY;
        for cand in 1..=d_cap {
            let mut run_p = 1.0f64;
            let mut e = 1.0f64;
            for j in 0..cand {
                run_p *= rate(j);
                e += run_p;
            }
            let cost = self.v[cand - 1] + cand as f64 * self.t_draft;
            let score = e / cost;
            if score > best_score {
                best_score = score;
                best_d = cand;
            }
        }
        // Hysteresis: keep the incumbent depth unless the challenger wins by
        // AD_SWITCH_MARGIN (see the constant's note on starvation).
        let d = if self.last_pick >= 1
            && self.last_pick <= d_cap
            && best_d != self.last_pick
            && best_score < self.last_score * (1.0 + AD_SWITCH_MARGIN)
        {
            self.last_pick
        } else {
            best_d
        };
        self.last_score = if d == best_d {
            best_score
        } else {
            self.score_of(d, p1, d_cap)
        };
        self.d_counts[d - 1] += 1;
        d
    }

    /// Score of a specific depth under the current estimates (the
    /// hysteresis baseline for the incumbent pick).
    fn score_of(&self, cand: usize, p1: f64, _d_cap: usize) -> f64 {
        let rate = |j: usize| -> f64 {
            if self.n[j] == 0 {
                p1
            } else {
                (self.succ[j] as f64 + AD_BETA_A) / (self.n[j] as f64 + AD_BETA_B)
            }
        };
        let mut run_p = 1.0f64;
        let mut e = 1.0f64;
        for j in 0..cand {
            run_p *= rate(j);
            e += run_p;
        }
        e / (self.v[cand - 1] + cand as f64 * self.t_draft)
    }

    /// Mean chosen depth for the stats line (0 when nothing picked yet).
    pub fn mean_d(&self) -> f64 {
        let n: u64 = self.d_counts.iter().sum();
        if n == 0 {
            return 0.0;
        }
        self.d_counts
            .iter()
            .enumerate()
            .map(|(i, &c)| (i + 1) as f64 * c as f64)
            .sum::<f64>()
            / n as f64
    }
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
    adaptive: Option<AdaptiveD>,
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
        // Identity cap (doc 95): the greedy identity is bitwise-proven for
        // verify nt <= 8 (single + multi MMVQ, kernel-level tests at nt
        // 3/5/8). d=8 -> verify nt=9 crosses into the BT-MMQ GEMM family,
        // whose lm_head accumulation is tolerance-class, so the adaptive
        // controller is capped at d=7. A static --spec-draft-n 8 stays
        // available (max throughput, documented as not identity-safe).
        let cap = if cfg.adaptive {
            cfg.draft_n.min(7)
        } else {
            cfg.draft_n
        };
        Ok(Self {
            draft,
            draft_cache: GraphCache::new(),
            draft_n: cap,
            adaptive: cfg.adaptive.then(|| AdaptiveD::new(cap)),
            stats: SpecStats::default(),
        })
    }

    /// doc 97: drop the draft KV (conversation `/clear` / full re-render —
    /// draft positions are absolute and must rewind with the target).
    pub fn reset_draft(&mut self) {
        self.draft_cache = GraphCache::new();
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
        // back to a plain single-token step (same sampler, one row). doc 95:
        // adaptive mode picks the depth from the per-depth acceptance and
        // cost EWMA curve instead of the static draft_n cap.
        let horizon = n_ctx.saturating_sub(pos + 1);
        let d = match self.adaptive.as_mut() {
            Some(ad) => ad.pick(horizon),
            None => self.draft_n.min(horizon),
        };
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
            push_capped(prev_tokens, t.token_id);
            self.stats.rounds += 1;
            return vec![t.token_id];
        }

        // Draft phase: d forwards → d proposals for positions pos+1..pos+d.
        // Raw draft argmax (no penalties — penalties are target-sampler
        // semantics; a penalized proposal would only fail to match).
        let draft_t0 = std::time::Instant::now();
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
        let draft_ms = draft_t0.elapsed().as_secs_f64() * 1e3;
        if let Some(ad) = self.adaptive.as_mut() {
            ad.observe_draft(draft_ms, d);
        }
        self.stats.drafted += d as u64;
        self.stats.d_sum += d as u64;

        // Verify: one nt=d+1 forward with logits on every row. Row i predicts
        // position pos+i+1.
        let mut rows = Vec::with_capacity(d + 1);
        rows.push(next_token);
        rows.extend_from_slice(&proposals);
        let positions: Vec<usize> = (pos..=pos + d).collect();
        let verify_t0 = std::time::Instant::now();
        let logits = target.forward_graph_cached(&rows, &positions, d + 1, n_ctx, target_cache);
        let verify_ms = verify_t0.elapsed().as_secs_f64() * 1e3;
        if let Some(ad) = self.adaptive.as_mut() {
            ad.observe_verify(d + 1, verify_ms);
        }
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
            pos,
        );
        if let Some(ad) = self.adaptive.as_mut() {
            ad.observe_round(d, accepted);
        }
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
    base_pos: usize,
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
            push_capped(prev_tokens, proposals[i]);
            crate::token_trace(base_pos + i + 1, proposals[i]);
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
            push_capped(prev_tokens, t.token_id);
            crate::token_trace(base_pos + i + 1, t.token_id);
            break;
        }
    }
    (emitted, accepted)
}

/// Push a committed token into the penalty window, trimming to
/// `sampler::REPEAT_LAST_N` — the same window the sequential decode loop
/// maintains. Without the cap the spec path's window grows to the whole
/// generation and the repeat penalty reaches tokens the sequential run no
/// longer penalizes (doc 94: the greedy identity broke at the first
/// >64-distant repeat).
fn push_capped(prev_tokens: &mut Vec<u32>, t: u32) {
    prev_tokens.push(t);
    if prev_tokens.len() > crate::sampler::REPEAT_LAST_N {
        prev_tokens.drain(0..prev_tokens.len() - crate::sampler::REPEAT_LAST_N);
    }
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

    #[test]
    fn adaptive_picks_deep_when_acceptance_stays_high() {
        // code-like acceptance (~0.8 flat) with the doc-94 cost curve: the
        // deep-depth expected tokens amortize the verify round -> d_max wins.
        let mut ad = AdaptiveD::new(8);
        for _ in 0..40 {
            ad.observe_round(8, 8); // everything accepted at every depth
        }
        let d = ad.pick(64);
        assert_eq!(d, 8, "flat-high acceptance should pick the cap, got {d}");
    }

    #[test]
    fn adaptive_picks_shallow_when_acceptance_collapses() {
        // prose-like: p1 ~0.5, deeper depths collapse — deep rounds pay the
        // C_T(9) premium for nothing -> the controller must stay shallow.
        let mut ad = AdaptiveD::new(8);
        for _ in 0..40 {
            ad.observe_round(8, 1); // only the first draft ever survives
        }
        let d = ad.pick(64);
        assert!(d <= 2, "collapsing acceptance should pick d<=2, got {d}");
    }

    #[test]
    fn adaptive_respects_horizon_and_explores() {
        let mut ad = AdaptiveD::new(8);
        assert_eq!(ad.pick(0), 0, "spent horizon -> the d==0 fallback");
        assert_eq!(ad.pick(3), 3, "horizon caps the pick (prior curve)");
    }

    #[test]
    fn adaptive_online_cost_observations_shift_the_pick() {
        // With flat-high acceptance but an absurd observed deep-verify cost,
        // the controller must back off the cap (online correction beats the
        // hardcoded prior).
        let mut ad = AdaptiveD::new(8);
        for _ in 0..30 {
            ad.observe_round(8, 8);
        }
        for _ in 0..30 {
            ad.observe_verify(9, 400.0); // 400 ms per nt=9 verify
        }
        let d = ad.pick(64);
        assert!(d < 8, "prohibitive deep-verify cost must back off, got {d}");
    }

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
            accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0, 0);
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
            accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0, 0);
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
            accept_loop(&logits, &[10, 11], &sampler(), &mut prev, &mut rng, 0, 0, 0);
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
        let (emitted, accepted) = accept_loop(&logits, &[1], &s, &mut prev, &mut rng, 0, 0, 0);
        assert_eq!(accepted, 1);
        assert_eq!(emitted, vec![1, 2]);
        assert_eq!(prev, vec![1, 2]);
    }
}
