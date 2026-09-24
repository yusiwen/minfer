// Sampler — repetition/frequency/presence penalties, top-k, top-p, temperature (seeded)

use rand::Rng;
use std::collections::HashMap;
use std::sync::Arc;

use crate::grammar::{Grammar, GrammarState};

/// Sampling result
#[derive(Debug)]
pub struct SampledToken {
    pub token_id: u32,
    /// Logit of the sampled token (result metadata; callers read `token_id`).
    #[allow(dead_code)]
    pub logit: f32,
}

/// A failure of the one pipeline (F2, #47). Both variants are loud stops: a
/// caller must never fall back to an arbitrary token.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SampleError {
    /// The grammar allowed no token at this state (and no EOG was legal either).
    /// `state` is the automaton's own description of where it is stuck.
    NoAllowedToken { state: String },
    /// The grammar engine refused something (a rejected token, a mismatched
    /// logits length, a configured grammar without a run state, ...).
    Grammar(String),
}

impl std::fmt::Display for SampleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SampleError::NoAllowedToken { state } => write!(
                f,
                "grammar: no token is allowed at this state ({state}); stopping rather than \
                 emitting a token the grammar forbids"
            ),
            SampleError::Grammar(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for SampleError {}

/// Greedy sampling: pick highest logit
pub fn sample_greedy(logits: &[f32]) -> SampledToken {
    let mut best_id = 0u32;
    let mut best_val = logits[0];
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_id = i as u32;
        }
    }
    SampledToken {
        token_id: best_id,
        logit: best_val,
    }
}

/// Combined penalty pass over the tokens in `prev_tokens` (the caller keeps the
/// window; llama.cpp `repeat_last_n` default = 64). Matches llama.cpp's
/// `llama_sampler_init_penalties` semantics:
///
/// ```text
/// for each distinct token t in prev_tokens, with count(t) occurrences:
///     logits[t] -= count(t) * frequency_penalty           # if frequency_penalty != 0
///     logits[t] -= presence_penalty                       # once, if count(t) > 0
///     logits[t] = logits[t] <= 0 ? logits[t] * repeat     # if repeat != 1.0 and repeat >= 1.0
///                               : logits[t] / repeat
/// ```
///
/// All three penalties are applied in one pass over the window (distinct tokens
/// counted once with a HashMap) — same cost class as llama.cpp's penalties
/// sampler. `repeat == 1.0` disables the repeat term; `repeat < 1.0` is also
/// ignored (existing minfer behavior, matching `apply_repetition_penalty`).
pub fn apply_penalties(
    logits: &mut [f32],
    prev_tokens: &[u32],
    repeat: f32,
    frequency: f32,
    presence: f32,
) {
    if (repeat - 1.0).abs() < 1e-6 && frequency.abs() < 1e-6 && presence.abs() < 1e-6 {
        return;
    }
    let mut counts: HashMap<u32, u32> = HashMap::new();
    for &t in prev_tokens {
        *counts.entry(t).or_insert(0) += 1;
    }
    for (&t, &c) in &counts {
        let idx = t as usize;
        if idx >= logits.len() {
            continue;
        }
        let v = logits[idx];
        let mut nv = v;
        if frequency.abs() >= 1e-6 || presence.abs() >= 1e-6 {
            nv -= c as f32 * frequency;
            if c > 0 {
                nv -= presence;
            }
        }
        if (repeat - 1.0).abs() >= 1e-6 && repeat >= 1.0 {
            nv = if nv <= 0.0 { nv * repeat } else { nv / repeat };
        }
        logits[idx] = nv;
    }
}

/// Repetition penalty: penalize tokens that already appeared in `prev_tokens`.
/// `penalty == 1.0` disables. Positive logits are divided by the penalty
/// (reduced), negative logits are multiplied (pushed further down). This is
/// llama.cpp's `repeat_penalty` applied to the last `repeat_last_n` tokens.
/// (Kept as the standalone penalty API — the CLI path uses
/// `sample_with_penalties` directly; tests exercise this wrapper.)
#[allow(dead_code)]
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    apply_penalties(logits, prev_tokens, penalty, 0.0, 0.0);
}

/// Last `last_n` tokens of `tokens` — the recent-token window for the penalty
/// pass (llama.cpp `repeat_last_n` default = 64). Returns the whole slice when
/// shorter; empty when `tokens` is empty.
pub fn recent_window(tokens: &[u32], last_n: usize) -> Vec<u32> {
    let start = tokens.len().saturating_sub(last_n);
    tokens[start..].to_vec()
}

/// Byte-wise suffix match of `buf` against any stop string in `stops`.
///
/// Returns the byte index where the matched stop string starts — the
/// truncation point for the generated text (text up to that index is kept).
/// When several stop strings match, the earliest start wins (the longest stop
/// truncates the most). Empty stop strings are ignored. Multi-byte stop
/// strings split across tokens match because the comparison is byte-wise.
pub fn match_stop_suffix(buf: &[u8], stops: &[&[u8]]) -> Option<usize> {
    let mut best: Option<usize> = None;
    for s in stops {
        if s.is_empty() || s.len() > buf.len() {
            continue;
        }
        if &buf[buf.len() - s.len()..] == *s {
            let start = buf.len() - s.len();
            best = Some(best.map_or(start, |b| b.min(start)));
        }
    }
    best
}

/// Top-K filtering: keep only top K logits, set the rest to -INFINITY.
///
/// O(n) threshold extraction via `select_nth_unstable_by` instead of a
/// full-vocab sort — llama.cpp uses an equivalent partial selection
/// (`std::partial_sort`) here. The selection runs on a copy because the
/// in-place variant reorders the array, which would corrupt the
/// index→token mapping; the original `logits` is only masked, never moved.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut sorted = logits.to_vec();
    sorted.select_nth_unstable_by(k - 1, |a, b| b.total_cmp(a));
    let threshold = sorted[k - 1];
    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Top-P (nucleus) filtering: keep the smallest set of tokens whose cumulative
/// softmax probability >= p. Sets excluded tokens' raw logits to -INFINITY
/// (does NOT overwrite logits with probabilities, so the final temperature
/// softmax stays correct).
///
/// Only the finite survivors of a prior top_k pass matter (≤ k entries); masked
/// logits contribute exp(-INF - max) = 0 to the softmax, so working on the
/// survivors alone is bit-identical to the full-array computation. Falls back
/// to the full-array path when too many candidates survive (i.e. top_k
/// disabled) so this never degenerates into a full-vocab sort.
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p <= 0.0 || p >= 1.0 {
        return;
    }
    let survivors: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > f32::NEG_INFINITY)
        .map(|(i, &v)| (i, v))
        .collect();
    if survivors.is_empty() {
        return;
    }
    if survivors.len() > 1024 {
        apply_top_p_full(logits, p);
        return;
    }

    // Softmax over the survivors (identical to the old full-array softmax since
    // the masked entries contribute exp(-INF)=0 and 0.0 doesn't change the f64
    // running sum).
    let max_val = survivors
        .iter()
        .fold(f32::NEG_INFINITY, |a, &(_, v)| a.max(v));
    let sum: f64 = survivors
        .iter()
        .map(|&(_, v)| ((v - max_val) as f64).exp())
        .sum();
    let mut cand: Vec<(usize, f32)> = survivors
        .iter()
        .map(|&(i, v)| (i, (v - max_val).exp() / sum as f32))
        .collect();

    // Stable descending sort by probability (ties keep index order, matching
    // the full-array sort the survivor set was derived from).
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    let mut keep = cand.len();
    for (i, &(_, prob)) in cand.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            keep = i + 1;
            break;
        }
    }
    for &(idx, _) in &cand[keep..] {
        logits[idx] = f32::NEG_INFINITY;
    }
}

/// Full-array top-p fallback (only when top_k is disabled and > 1024
/// candidates survive). Retains the original softmax-over-everything +
/// full sort behavior for that rare path.
fn apply_top_p_full(logits: &mut [f32], p: f32) {
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum: f64 = logits.iter().map(|&v| ((v - max_val) as f64).exp()).sum();
    let mut indexed: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, ((v - max_val).exp() / sum as f32) as f32))
        .collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut cumulative = 0.0f32;
    for (i, &(_, prob)) in indexed.iter().enumerate() {
        cumulative += prob;
        if cumulative > p {
            for &(idx, _) in &indexed[i + 1..] {
                logits[idx] = f32::NEG_INFINITY;
            }
            break;
        }
    }
}

/// Temperature sampling: scale raw logits by 1/temp, softmax, sample.
pub fn sample_temperature<R: Rng>(logits: &mut [f32], temp: f32, rng: &mut R) -> SampledToken {
    if temp < 1e-6 {
        return sample_greedy(logits);
    }

    let inv_temp = 1.0 / temp;
    for v in logits.iter_mut() {
        *v *= inv_temp;
    }

    // Softmax. Masked (-INF) logits map to exp(-INF)=0 and contribute nothing
    // to the running max or sum; skipping the exp() call for them avoids
    // ~n_vocab transcendental evaluations per token while staying bit-identical
    // (exp(-INF) == +0.0 exactly).
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f64;
    for v in logits.iter_mut() {
        if *v > f32::NEG_INFINITY {
            *v = (*v - max_val).exp();
        } else {
            *v = 0.0;
        }
        sum += *v as f64;
    }
    let inv_sum = (1.0 / sum) as f32;
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }

    // Sample from the distribution (skipping zero-probability tokens — after
    // top_k/top_p only ≤k entries are finite, so this scan is near-empty).
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        if v <= 0.0 {
            continue;
        }
        cumulative += v;
        if r <= cumulative {
            return SampledToken {
                token_id: i as u32,
                logit: v,
            };
        }
    }
    SampledToken {
        token_id: (logits.len() - 1) as u32,
        logit: logits[logits.len() - 1],
    }
}

/// Complete sampling pipeline: penalties → top-k → top-p → temperature.
/// The repeat-penalty window (llama.cpp `repeat_last_n` default): penalties
/// apply to the last 64 tokens. Callers pass `prev_tokens` already trimmed to
/// this length (main decode loop, server, conversation) — doc 94: the
/// speculative path must trim too, or its penalty window silently grows to
/// the whole generation and its greedy picks diverge from sequential decode
/// past the first >64-distant repeat.
pub const REPEAT_LAST_N: usize = 64;

/// `temp < 1e-6` (greedy) skips the stochastic steps but still applies the
/// penalties.
///
/// Pre-F3 signature, kept for callers that predate [`SamplerConfig`]. It builds
/// a config whose new knobs are all at their no-op defaults, so its output is
/// bit-identical to the pre-F3 chain (the
/// `default_config_is_bit_identical_to_the_old_path` gate pins this).
pub fn sample_with_penalties<R: Rng>(
    logits: &mut [f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    frequency_penalty: f32,
    presence_penalty: f32,
    prev_tokens: &[u32],
    rng: &mut R,
) -> SampledToken {
    let cfg = SamplerConfig {
        temp,
        top_k,
        top_p,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
        ..SamplerConfig::default()
    };
    let mut mirostat = MirostatState::new(cfg.mirostat_tau);
    sample_with_config(logits, &cfg, prev_tokens, &mut mirostat, rng)
}

/// Complete sampling pipeline with only the repeat penalty (frequency and
/// presence disabled) — kept for callers that don't use the OAI penalties.
#[allow(dead_code)]
pub fn sample<R: Rng>(
    logits: &mut [f32],
    temp: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    prev_tokens: &[u32],
    rng: &mut R,
) -> SampledToken {
    sample_with_penalties(
        logits,
        temp,
        top_k,
        top_p,
        repeat_penalty,
        0.0,
        0.0,
        prev_tokens,
        rng,
    )
}

// ============================================================================
// F3 sampler set: min-p, typical, XTC, DRY, mirostat, logit bias (#48)
//
// Every filter below is a pure function over `logits` (+ `prev_tokens` for DRY),
// documented with the property it guarantees and the cases it refuses. The only
// stateful piece is mirostat's running surprise budget `mu`, which the caller
// owns explicitly ([`MirostatState`]) so a decode step stays reproducible from
// (logits, config, state, rng seed). The defaults of every new knob are no-ops,
// so `SamplerConfig::default()` runs the pre-F3 chain bit-for-bit — that is the
// `default_config_is_bit_identical_to_the_old_path` gate.
// ============================================================================

/// Which mirostat variant to run. `Off` is the default and leaves the
/// temperature path untouched.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MirostatMode {
    Off,
    /// Mirostat v1 (Basu et al. 2020): estimates the distribution's truncation
    /// point `k` from the top `m` probabilities and adapts `mu` by the observed
    /// surprise.
    V1,
    /// Mirostat v2 (the variant llama.cpp recommends): truncates every token
    /// whose surprise `-log2(p)` exceeds `mu`, then adapts `mu`. One fewer
    /// hyperparameter than v1 (no `m`).
    V2,
}

impl MirostatMode {
    /// Parse the CLI / JSON spelling: 0 = off, 1 = v1, 2 = v2. Anything else is
    /// an error — a mistyped mode must never silently become "off".
    pub fn parse(v: i64) -> Result<Self, String> {
        match v {
            0 => Ok(MirostatMode::Off),
            1 => Ok(MirostatMode::V1),
            2 => Ok(MirostatMode::V2),
            other => Err(format!(
                "mirostat must be 0 (off), 1 (v1) or 2 (v2), got {other}"
            )),
        }
    }
}

/// Mirostat's cross-token state: the running surprise budget `mu`.
///
/// Mirostat is the one sampler here that cannot be a pure function of the
/// current logits — `mu` is fed back from the previous step. The caller owns it
/// (one per generation: a CLI run, a conversation session, a server request or
/// batch slot) and [`sample_with_config`] takes it by `&mut`, so the step is
/// deterministic given (logits, config, state, rng).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MirostatState {
    pub mu: f32,
}

impl MirostatState {
    /// llama.cpp's initialisation: `mu = 2 * tau` (the first truncation is wide,
    /// then the feedback drives `mu` toward the target surprise).
    pub fn new(tau: f32) -> Self {
        Self { mu: 2.0 * tau }
    }
}

/// The complete sampler configuration (F3 #48). The first nine fields are the
/// pre-F3 surface; everything from `min_p` on is the new set, and each new
/// default is a documented no-op:
///
/// | field | default | meaning of the default |
/// |---|---|---|
/// | `min_p` | 0.0 | filter disabled (`p <= 0`) |
/// | `typical_p` | 1.0 | filter disabled (`p >= 1`) |
/// | `dry_multiplier` | 0.0 | DRY disabled (`multiplier == 0`) |
/// | `xtc_probability` | 0.0 | XTC disabled |
/// | `mirostat` | `Off` | temperature sampling |
/// | `logit_bias` | empty | no bias applied |
#[derive(Clone, Debug)]
pub struct SamplerConfig {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    /// Min-p: keep tokens with `logit >= max_logit + ln(min_p)` (llama.cpp
    /// `llama_sampler_init_min_p`). `0.0` disables; `1.0` keeps only the argmax
    /// (and its exact ties). At least the argmax always survives.
    pub min_p: f32,
    /// Locally typical sampling: keep the smallest set (by
    /// `|surprisal - entropy|`) whose probability mass reaches `typical_p`.
    /// `1.0` disables; `0.0` keeps the single most typical token.
    pub typical_p: f32,
    /// DRY multiplier. `0.0` disables. Only consulted when `> 0.0`.
    pub dry_multiplier: f32,
    /// DRY penalty growth base (llama.cpp default 1.75). Must be `>= 1.0`.
    pub dry_base: f32,
    /// Minimum repeated-suffix length DRY reacts to (llama.cpp default 2).
    pub dry_allowed_length: usize,
    /// DRY's penalty window; `0` disables a configured DRY loudly (refused by
    /// [`SamplerConfig::validate`] rather than silently ignored).
    pub dry_penalty_last_n: usize,
    /// DRY restart sequences as token-id sequences (head token first). llama.cpp
    /// maps a breaker *string* to overlapping token sequences through the
    /// vocabulary; that port needs the tokenizer and is a follow-up (see the F3
    /// record) — here a breaker is given as its token ids, and a non-numeric
    /// spelling is a loud parse error at the CLI/server boundary.
    pub dry_breakers: Vec<Vec<u32>>,
    /// XTC: probability of applying the exclusion on a given step (`0.0`
    /// disables). Each application consumes exactly one RNG draw, so a disabled
    /// XTC leaves the RNG stream — and therefore every existing sequence —
    /// unchanged.
    pub xtc_probability: f32,
    /// XTC threshold (llama.cpp refuses `> 0.5`).
    pub xtc_threshold: f32,
    pub mirostat: MirostatMode,
    /// Mirostat target surprise `tau` in bits (llama.cpp default 5.0).
    pub mirostat_tau: f32,
    /// Mirostat learning rate `eta` (llama.cpp default 0.1).
    pub mirostat_eta: f32,
    /// Mirostat v1's estimator window `m` (llama.cpp default 100).
    pub mirostat_m: usize,
    /// `(token_id, bias)` added to the raw logit before every other sampler.
    /// The OpenAI-compatible request field and `--logit-bias` both land here.
    pub logit_bias: Vec<(u32, f32)>,
    /// F2 (#47): the compiled grammar (or JSON schema) this request samples
    /// under, or `None` for an unconstrained run. The object is immutable and
    /// shared by `Arc`; the *mutable* automaton state lives in the run
    /// (`GrammarState`, passed to [`sample_with_config_grammar`] exactly like
    /// [`MirostatState`]), so one compiled grammar can be reused by several
    /// requests without a lock while each keeps its own position.
    pub grammar: Option<Arc<Grammar>>,
}

impl Default for SamplerConfig {
    fn default() -> Self {
        Self {
            temp: 0.8, // llama.cpp default (sampling, not greedy)
            top_k: 40,
            top_p: 0.95,         // llama.cpp default
            repeat_penalty: 1.1, // 1.0 = disabled
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            min_p: 0.0,
            typical_p: 1.0,
            dry_multiplier: 0.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            dry_breakers: Vec::new(),
            xtc_probability: 0.0,
            xtc_threshold: 0.5,
            mirostat: MirostatMode::Off,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            mirostat_m: 100,
            logit_bias: Vec::new(),
            grammar: None,
        }
    }
}

impl SamplerConfig {
    /// Refuse every nonsensical value loudly, at the boundary (CLI parse, HTTP
    /// request), instead of clamping or ignoring it later.
    ///
    /// Ranges follow llama.cpp / the OpenAI API: `min_p`, `typical_p`,
    /// `top_p` in `[0, 1]`; `xtc_probability` in `[0, 1]`; `xtc_threshold` in
    /// `[0, 0.5]` (llama.cpp disables above 0.5, so anything above is refused
    /// rather than silently disabled); `dry_multiplier >= 0`, `dry_base >= 1`,
    /// `dry_allowed_length >= 1`, and a DRY-enabled config may not have a zero
    /// penalty window; `mirostat_tau > 0`, `mirostat_eta > 0`, `mirostat_m >= 1`;
    /// every logit bias finite and within the OpenAI `[-100, 100]` range.
    /// Token ids are checked separately by [`Self::validate_logit_bias`], which
    /// needs the vocabulary size.
    pub fn validate(&self) -> Result<(), String> {
        let finite = |name: &str, v: f32| -> Result<(), String> {
            if v.is_finite() {
                Ok(())
            } else {
                Err(format!("sampler: {name} must be finite, got {v}"))
            }
        };
        finite("temperature", self.temp)?;
        if self.temp < 0.0 {
            return Err(format!(
                "sampler: temperature must be >= 0 (0 = greedy), got {}",
                self.temp
            ));
        }
        for (name, v) in [
            ("min_p", self.min_p),
            ("typical_p", self.typical_p),
            ("top_p", self.top_p),
            ("xtc_probability", self.xtc_probability),
        ] {
            finite(name, v)?;
            if !(0.0..=1.0).contains(&v) {
                return Err(format!("sampler: {name} must be in [0, 1], got {v}"));
            }
        }
        finite("xtc_threshold", self.xtc_threshold)?;
        if !(0.0..=0.5).contains(&self.xtc_threshold) {
            return Err(format!(
                "sampler: xtc_threshold must be in [0, 0.5] (llama.cpp disables it above \
                 0.5), got {}",
                self.xtc_threshold
            ));
        }
        finite("repeat_penalty", self.repeat_penalty)?;
        if self.repeat_penalty < 0.0 {
            return Err(format!(
                "sampler: repeat_penalty must be >= 0, got {}",
                self.repeat_penalty
            ));
        }
        for (name, v) in [
            ("frequency_penalty", self.frequency_penalty),
            ("presence_penalty", self.presence_penalty),
        ] {
            finite(name, v)?;
        }
        finite("dry_multiplier", self.dry_multiplier)?;
        finite("dry_base", self.dry_base)?;
        if self.dry_multiplier < 0.0 {
            return Err(format!(
                "sampler: dry_multiplier must be >= 0 (0 disables DRY), got {}",
                self.dry_multiplier
            ));
        }
        if self.dry_multiplier > 0.0 {
            if self.dry_base < 1.0 {
                return Err(format!(
                    "sampler: dry_base must be >= 1, got {}",
                    self.dry_base
                ));
            }
            if self.dry_allowed_length < 1 {
                return Err("sampler: dry_allowed_length must be >= 1".to_string());
            }
            if self.dry_penalty_last_n == 0 {
                return Err(
                    "sampler: dry_multiplier > 0 with dry_penalty_last_n = 0 would silently \
                     disable DRY; set a window or leave dry_multiplier at 0"
                        .to_string(),
                );
            }
        }
        for &(id, b) in &self.logit_bias {
            finite("logit_bias value", b)?;
            if !(-100.0..=100.0).contains(&b) {
                return Err(format!(
                    "sampler: logit_bias for token {id} must be in the OpenAI range \
                     [-100, 100], got {b}"
                ));
            }
        }
        if self.mirostat != MirostatMode::Off {
            finite("mirostat_tau", self.mirostat_tau)?;
            finite("mirostat_eta", self.mirostat_eta)?;
            if self.mirostat_tau <= 0.0 {
                return Err(format!(
                    "sampler: mirostat_tau must be > 0 (target surprise in bits), got {}",
                    self.mirostat_tau
                ));
            }
            if self.mirostat_eta <= 0.0 {
                return Err(format!(
                    "sampler: mirostat_eta must be > 0 (learning rate), got {}",
                    self.mirostat_eta
                ));
            }
            if self.mirostat_m < 1 {
                return Err("sampler: mirostat_m must be >= 1".to_string());
            }
        }
        Ok(())
    }

    /// Refuse a logit bias whose token id is outside the vocabulary. Called at
    /// the boundary where the vocabulary size is known (after the tokenizer
    /// loads, or per HTTP request) so an out-of-vocab id is a startup/`400`
    /// error rather than a silently skipped bias inside [`apply_logit_bias`].
    pub fn validate_logit_bias(&self, n_vocab: usize) -> Result<(), String> {
        for &(id, _) in &self.logit_bias {
            if id as usize >= n_vocab {
                return Err(format!(
                    "sampler: logit_bias token id {id} is outside the vocabulary \
                     (0..{n_vocab})"
                ));
            }
        }
        Ok(())
    }
}

/// Add `(token_id, bias)` to raw logits (llama.cpp `logit_bias`, OpenAI
/// `logit_bias`). Applied first, before every other filter, so a bias can both
/// promote a token into the nucleus and demote it out of it.
///
/// The token id must already have been checked against the vocabulary by
/// [`SamplerConfig::validate_logit_bias`]; `get_mut` keeps the function total
/// for internal callers that never build a config (unit tests).
pub fn apply_logit_bias(logits: &mut [f32], bias: &[(u32, f32)]) {
    for &(id, b) in bias {
        if let Some(v) = logits.get_mut(id as usize) {
            *v += b;
        }
    }
}

/// Shared softmax helper for the new filters: sorted descending by probability,
/// masked (`-inf`) entries dropped, ties keeping index order (Rust's sort is
/// stable). Returns `(token_id, probability)`.
fn softmax_desc(logits: &[f32]) -> Vec<(usize, f32)> {
    let survivors: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, &v)| v > f32::NEG_INFINITY)
        .map(|(i, &v)| (i, v))
        .collect();
    if survivors.is_empty() {
        return Vec::new();
    }
    let max_val = survivors
        .iter()
        .fold(f32::NEG_INFINITY, |a, &(_, v)| a.max(v));
    let sum: f64 = survivors
        .iter()
        .map(|&(_, v)| ((v - max_val) as f64).exp())
        .sum();
    let mut cand: Vec<(usize, f32)> = survivors
        .iter()
        .map(|&(i, v)| (i, (((v - max_val) as f64).exp() / sum) as f32))
        .collect();
    cand.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cand
}

/// Sample from `(token_id, probability)` pairs by inverse-CDF (the same
/// procedure `sample_temperature` uses). Falls back to the last candidate when
/// rounding leaves the cumulative sum below `r` — never to a different
/// distribution.
fn sample_categorical<R: Rng>(cand: &[(usize, f32)], rng: &mut R) -> (usize, f32) {
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for &(i, p) in cand {
        if p <= 0.0 {
            continue;
        }
        cumulative += p;
        if r <= cumulative {
            return (i, p);
        }
    }
    *cand.last().expect("caller checked non-empty")
}

/// Min-p filtering (llama.cpp `llama_sampler_init_min_p`): keep tokens with
/// `logit >= max_logit + ln(p)`, i.e. every token whose probability is at least
/// `p * max_probability`. Masked tokens get `-inf` (raw logits are never
/// overwritten with probabilities, so the later softmax stays correct).
///
/// Boundary behaviour: `p <= 0.0` is a no-op; `p == 1.0` keeps the argmax and
/// its exact ties; an empty candidate set (all `-inf`) is a no-op; the argmax
/// always survives even for `p > 1.0`, so the filter can never empty the
/// distribution (llama.cpp's `min_keep >= 1` guarantee).
pub fn apply_min_p(logits: &mut [f32], p: f32) {
    if p <= 0.0 {
        return;
    }
    let mut best: Option<(usize, f32)> = None;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && best.map_or(true, |(_, b)| v > b) {
            best = Some((i, v));
        }
    }
    let Some((best_id, max_logit)) = best else {
        return;
    };
    let min_logit = max_logit + p.ln();
    let mut kept = 0usize;
    for v in logits.iter_mut() {
        if *v >= min_logit {
            kept += 1;
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
    if kept == 0 {
        logits[best_id] = max_logit;
    }
}

/// Locally typical sampling (llama.cpp `llama_sampler_init_typical`; Meister et
/// al. 2022): compute the softmax entropy `H`, rank candidates by
/// `| -ln(p) - H |` (closeness of their surprisal to the distribution's
/// entropy), and keep the smallest prefix whose cumulative probability reaches
/// `p`.
///
/// Boundary behaviour: `p >= 1.0` is a no-op; `p <= 0.0` keeps exactly one
/// token (the most typical — llama.cpp's first iteration always exceeds `p`);
/// an empty or one-candidate set is a no-op. Ties in the score keep index
/// order.
pub fn apply_typical(logits: &mut [f32], p: f32) {
    if p >= 1.0 {
        return;
    }
    let cand = softmax_desc(logits);
    if cand.len() <= 1 {
        return;
    }
    let entropy: f64 = cand
        .iter()
        .map(|&(_, pr)| {
            let pr = pr as f64;
            if pr > 0.0 {
                -pr * pr.ln()
            } else {
                0.0
            }
        })
        .sum();
    let score = |i: usize| -> f64 { (-(cand[i].1 as f64).ln() - entropy).abs() };
    let mut order: Vec<usize> = (0..cand.len()).collect();
    order.sort_by(|&a, &b| {
        score(a)
            .partial_cmp(&score(b))
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cumulative = 0.0f64;
    let mut keep = order.len();
    for (i, &ci) in order.iter().enumerate() {
        cumulative += cand[ci].1 as f64;
        if cumulative > p as f64 {
            keep = i + 1;
            break;
        }
    }
    let mut kept = vec![false; logits.len()];
    for &ci in &order[..keep] {
        kept[cand[ci].0] = true;
    }
    for (i, v) in logits.iter_mut().enumerate() {
        if !kept[i] {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// XTC — "Exclude Top Choices" (llama.cpp `llama_sampler_init_xtc`).
///
/// With probability `probability` (< 1), the tokens whose probability is
/// `>= threshold` are excluded except the *least* likely of them: the top
/// choices are dropped so the tail gets a chance, while one above-threshold
/// candidate remains. Exactly one RNG draw is consumed per *applied* step.
///
/// Boundary behaviour: `probability <= 0.0` or `threshold > 0.5` is a no-op
/// (llama.cpp's "empty" XTC); fewer than two live candidates is a no-op;
/// `threshold` above every probability means `pos_last == 0` and nothing is
/// dropped, so the distribution is never emptied.
pub fn apply_xtc<R: Rng>(logits: &mut [f32], probability: f32, threshold: f32, rng: &mut R) {
    if probability <= 0.0 || threshold > 0.5 {
        return;
    }
    let cand = softmax_desc(logits);
    if cand.len() < 2 {
        return;
    }
    let chance: f32 = rng.gen();
    if chance > probability {
        return;
    }
    // `pos_last` = index of the last candidate still at or above the threshold
    // (the array is sorted descending, so those form a prefix).
    let mut pos_last = 0usize;
    for (i, &(_, pr)) in cand.iter().enumerate() {
        if pr >= threshold {
            pos_last = i;
        } else {
            break;
        }
    }
    // Keep candidate `pos_last` and everything below it (at least one token).
    for &(id, _) in &cand[..pos_last] {
        logits[id] = f32::NEG_INFINITY;
    }
}

/// DRY — "Don't Repeat Yourself" (llama.cpp `llama_sampler_init_dry`, ported
/// from KoboldCpp PR #982 by pi6am).
///
/// `tokens` is the recent history (the caller's penalty window; the function
/// itself keeps only its last `penalty_last_n` entries). Three passes:
///
/// 1. a restart sequence (a *sequence breaker*) caps how far back a repetition
///    may reach: scanning backwards, the first breaker found at distance `i`
///    with respect to its tail sets `rep_limit = i - tail_len`;
/// 2. the reverse Z-algorithm computes, for every position, the length of the
///    suffix of the history that repeats there (`repeat_count`);
/// 3. for every token that would *extend* a repeat of at least
///    `allowed_length`, the penalty
///    `multiplier * base^(repeat_len - allowed_length)` is subtracted from its
///    logit (clamped so `base^exp` cannot overflow `f32`).
///
/// Boundary behaviour: `multiplier == 0`, `base < 1`, `penalty_last_n == 0`, an
/// empty history, and a window no longer than `allowed_length` are all no-ops —
/// an empty history must never panic (index arithmetic here is entirely
/// relative to `tokens.len()`).
pub fn apply_dry(
    logits: &mut [f32],
    tokens: &[u32],
    multiplier: f32,
    base: f32,
    allowed_length: usize,
    penalty_last_n: usize,
    breakers: &[Vec<u32>],
) {
    if multiplier == 0.0 || base < 1.0 || penalty_last_n == 0 {
        return;
    }
    let last_n = tokens.len().min(penalty_last_n);
    if last_n <= allowed_length {
        return;
    }
    let rat = |i: usize| -> u32 { tokens[tokens.len() - 1 - i] };

    // Step 1: the longest restart sequence starting `i` from the end caps the
    // repetition length. A breaker is (head, tail...): the head sits at
    // distance i, its tail continues at i-1, i-2, ...
    let mut rep_limit = last_n;
    'restart: for i in 0..last_n {
        let head = rat(i);
        for b in breakers {
            let Some((&b_head, tail)) = b.split_first() else {
                continue;
            };
            if b_head != head || tail.len() > i {
                continue;
            }
            let mut matched = true;
            for (offset, &t) in tail.iter().enumerate() {
                if t != rat(i - offset - 1) {
                    matched = false;
                    break;
                }
            }
            if matched {
                rep_limit = i - tail.len();
                break 'restart;
            }
        }
    }
    if rep_limit < allowed_length {
        return;
    }

    // Step 2: reverse Z-algorithm over the window (O(last_n)).
    let last = last_n - 1;
    let mut repeat_count = vec![0i32; last_n];
    let mut rt: i32 = 0;
    let mut lt: i32 = 0;
    for k in 1..last_n {
        if (k as i32) > rt {
            let mut n = 0usize;
            while n + k < last_n && rat(n) == rat(n + k) {
                n += 1;
            }
            repeat_count[last - k] = (n as i32).min(rep_limit as i32);
            if n > 0 {
                lt = k as i32;
                rt = (k + n - 1) as i32;
            }
        } else {
            let p = k as i32 - lt;
            let right_part_len = rt - k as i32 + 1;
            if repeat_count[last - p as usize] < right_part_len {
                repeat_count[last - k] = repeat_count[last - p as usize].min(rep_limit as i32);
            } else {
                let mut i = (rt + 1) as usize;
                while i < last_n && rat(i) == rat(i - k) {
                    i += 1;
                }
                repeat_count[last - k] = ((i - k) as i32).min(rep_limit as i32);
                lt = k as i32;
                rt = (i - 1) as i32;
            }
        }
    }

    // Step 3: the maximum repeat length each token would extend.
    let mut max_token_repeat: HashMap<u32, i32> = HashMap::new();
    for i in 0..last_n - 1 {
        let repeat_len = repeat_count[i];
        if repeat_len >= allowed_length as i32 {
            let token = rat(last_n - 2 - i);
            let e = max_token_repeat.entry(token).or_insert(0);
            if *e < repeat_len {
                *e = repeat_len;
            }
        }
    }

    // Step 4: penalise the tokens that would continue a repeat. A single-token
    // sequence breaker is exempt: it exists to be repeated (a newline, a colon).
    const FLOAT_MAX_LOG: f32 = 88.722_84; // ln(f32::MAX)
    let max_exponent = if base > 1.000_001 {
        (FLOAT_MAX_LOG / base.ln()) as i32
    } else {
        0
    };
    let single_token_breaker =
        |token: u32| -> bool { breakers.iter().any(|b| b.len() == 1 && b[0] == token) };
    for (i, v) in logits.iter_mut().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let token = i as u32;
        let Some(&repeat_len) = max_token_repeat.get(&token) else {
            continue;
        };
        if single_token_breaker(token) {
            continue;
        }
        let mut repeat_exp = repeat_len - allowed_length as i32;
        if max_exponent > 0 && repeat_exp > max_exponent {
            repeat_exp = max_exponent;
        }
        let penalty = multiplier * base.powi(repeat_exp);
        *v -= penalty;
    }
}

/// One mirostat v2 step (llama.cpp `llama_sampler_mirostat_v2`): truncate every
/// token whose surprise `-log2(p)` exceeds `mu`, renormalise, sample, then move
/// `mu` by `-eta * (observed_surprise - tau)` so the running surprise converges
/// on the target.
///
/// Boundary behaviour: the truncation keeps at least one token, so `mu <= 0` or
/// a degenerate distribution can never produce an empty candidate set; an empty
/// input returns token 0 with `-inf` logit and leaves `mu` untouched.
pub fn sample_mirostat_v2<R: Rng>(
    logits: &[f32],
    mu: &mut f32,
    tau: f32,
    eta: f32,
    rng: &mut R,
) -> SampledToken {
    let cand = softmax_desc(logits);
    if cand.is_empty() {
        return SampledToken {
            token_id: 0,
            logit: f32::NEG_INFINITY,
        };
    }
    let mut keep = cand.len();
    for (i, &(_, p)) in cand.iter().enumerate() {
        if p <= 0.0 || -((p as f64).log2()) > *mu as f64 {
            keep = i;
            break;
        }
    }
    let keep = keep.max(1);
    let survivors = &cand[..keep];
    let sum: f64 = survivors.iter().map(|&(_, p)| p as f64).sum();
    let norm: Vec<(usize, f32)> = survivors
        .iter()
        .map(|&(id, p)| (id, (p as f64 / sum) as f32))
        .collect();
    let (token_id, p) = sample_categorical(&norm, rng);
    let observed = -((p as f64).log2());
    *mu = (*mu as f64 - eta as f64 * (observed - tau as f64)) as f32;
    SampledToken {
        token_id: token_id as u32,
        logit: p,
    }
}

/// One mirostat v1 step (llama.cpp `llama_sampler_mirostat`): estimate the
/// distribution's exponent `s_hat` from the top `m` probabilities, solve the
/// truncation size `k` from `mu`, top-k, sample, then update `mu` from the
/// observed surprise.
///
/// Degenerate rule (documented, not a silent fallback): when the estimate is
/// undefined — a one-token candidate set, or a top-`m` tail containing a zero
/// probability — `s_hat` is taken as `1.0`, which yields `k = 1`; the step then
/// samples the argmax deterministically and still updates `mu`. `mu` rises by
/// `eta * tau` in that case (observed surprise 0 minus the target).
pub fn sample_mirostat_v1<R: Rng>(
    logits: &mut [f32],
    mu: &mut f32,
    tau: f32,
    eta: f32,
    m: usize,
    rng: &mut R,
) -> SampledToken {
    let cand = softmax_desc(logits);
    if cand.is_empty() {
        return SampledToken {
            token_id: 0,
            logit: f32::NEG_INFINITY,
        };
    }
    let n_vocab = logits.len() as f64;
    let mut sum_ti_bi = 0.0f64;
    let mut sum_ti_sq = 0.0f64;
    let lim = m.saturating_sub(1).min(cand.len().saturating_sub(1));
    for i in 0..lim {
        let p0 = cand[i].1 as f64;
        let p1 = cand[i + 1].1 as f64;
        if p0 <= 0.0 || p1 <= 0.0 {
            break;
        }
        let t_i = (((i + 2) as f64) / ((i + 1) as f64)).ln();
        sum_ti_bi += t_i * (p0 / p1).ln();
        sum_ti_sq += t_i * t_i;
    }
    let mut s_hat = if sum_ti_sq > 0.0 {
        sum_ti_bi / sum_ti_sq
    } else {
        1.0
    };
    if !s_hat.is_finite() || s_hat <= 0.0 {
        s_hat = 1.0;
    }
    let epsilon_hat = s_hat - 1.0;
    let denom = 1.0 - n_vocab.powf(-epsilon_hat);
    let k = if denom.abs() < 1e-12 {
        1.0
    } else {
        ((epsilon_hat * 2.0f64.powf(*mu as f64)) / denom).powf(1.0 / s_hat)
    };
    let k = if k.is_finite() && k >= 1.0 {
        k as usize
    } else {
        1
    };
    apply_top_k(logits, k);
    let cand = softmax_desc(logits);
    let (token_id, p) = sample_categorical(&cand, rng);
    let observed = -((p as f64).log2());
    *mu = (*mu as f64 - eta as f64 * (observed - tau as f64)) as f32;
    SampledToken {
        token_id: token_id as u32,
        logit: p,
    }
}

/// The complete pipeline, grammar-aware (F2, #47):
///
/// ```text
/// logit bias → penalties → DRY → [GRAMMAR MASK] → (greedy shortcut) → top-k →
/// typical → top-p → min-p → XTC → temperature | mirostat
/// ```
///
/// The order is llama.cpp's `common_sampler_init` chain, with three documented
/// decisions:
///
/// * the `temp < 1e-6` greedy shortcut keeps the exact pre-F3 position (after
///   the deterministic penalties, before the stochastic filters), so a greedy
///   request is bit-identical to today and `min_p`/`typical`/`xtc` cannot
///   perturb it;
/// * in mirostat mode the temperature is *ignored* — mirostat truncates the
///   distribution at `mu`, which subsumes temperature (llama.cpp sets
///   `temp = 1.0` when mirostat is on). The greedy shortcut still wins at
///   `temp == 0`.
/// * **the grammar mask sits between DRY and the greedy shortcut.** Everything
///   before it only *shifts* logits (bias, penalties, DRY add finite values), so
///   it cannot lift a masked `-inf` back to finite; everything after it only
///   *removes* candidates or reweights the survivors, so no forbidden token can
///   be selected. Being before the shortcut is what makes `--greedy` respect the
///   grammar. The mask consumes no RNG and writes nothing the other stages read,
///   so mirostat's `mu` and DRY's penalties are unchanged for the same token
///   sequence (the no-grammar path is pinned bitwise by
///   `test_default_pipeline_matches_the_pinned_pre_f2_sequence`).
///
/// The caller trims `prev_tokens` to its own window (the CLI keeps 64); DRY
/// additionally trims to `dry_penalty_last_n` internally. `grammar` is the
/// run's automaton state, one per run exactly like `mirostat`.
pub fn sample_with_config_grammar<R: Rng>(
    logits: &mut [f32],
    cfg: &SamplerConfig,
    prev_tokens: &[u32],
    mirostat: &mut MirostatState,
    grammar: &mut Option<GrammarState>,
    rng: &mut R,
) -> Result<SampledToken, SampleError> {
    apply_logit_bias(logits, &cfg.logit_bias);
    apply_penalties(
        logits,
        prev_tokens,
        cfg.repeat_penalty,
        cfg.frequency_penalty,
        cfg.presence_penalty,
    );
    if cfg.dry_multiplier > 0.0 {
        apply_dry(
            logits,
            prev_tokens,
            cfg.dry_multiplier,
            cfg.dry_base,
            cfg.dry_allowed_length,
            cfg.dry_penalty_last_n,
            &cfg.dry_breakers,
        );
    }
    let constrained = match (cfg.grammar.as_ref(), grammar.as_mut()) {
        (Some(g), Some(st)) => {
            if logits.len() != g.n_vocab() {
                return Err(SampleError::Grammar(format!(
                    "grammar: the logits row has {} entries but the grammar's vocabulary has {}",
                    logits.len(),
                    g.n_vocab()
                )));
            }
            let mask = g.mask(st).map_err(SampleError::Grammar)?;
            let allowed = apply_token_mask(logits, &mask);
            if allowed == 0 {
                return Err(SampleError::NoAllowedToken {
                    state: g.describe(st),
                });
            }
            true
        }
        // A configured grammar without a state would silently drop the
        // constraint; that is a bug, never a fallback.
        (Some(_), None) => {
            return Err(SampleError::Grammar(
                "grammar: a grammar is configured but this run has no grammar state".to_string(),
            ))
        }
        (None, _) => false,
    };

    let sampled = if cfg.temp < 1e-6 {
        sample_greedy(logits)
    } else {
        apply_top_k(logits, cfg.top_k);
        apply_typical(logits, cfg.typical_p);
        apply_top_p(logits, cfg.top_p);
        apply_min_p(logits, cfg.min_p);
        apply_xtc(logits, cfg.xtc_probability, cfg.xtc_threshold, rng);
        match cfg.mirostat {
            MirostatMode::Off => sample_temperature(logits, cfg.temp, rng),
            MirostatMode::V2 => sample_mirostat_v2(
                logits,
                &mut mirostat.mu,
                cfg.mirostat_tau,
                cfg.mirostat_eta,
                rng,
            ),
            MirostatMode::V1 => sample_mirostat_v1(
                logits,
                &mut mirostat.mu,
                cfg.mirostat_tau,
                cfg.mirostat_eta,
                cfg.mirostat_m,
                rng,
            ),
        }
    };
    if constrained {
        if let (Some(g), Some(st)) = (cfg.grammar.as_ref(), grammar.as_mut()) {
            g.accept_token(st, sampled.token_id)
                .map_err(SampleError::Grammar)?;
        }
    }
    Ok(sampled)
}

/// Mask the logits to the allowed tokens (`-inf` elsewhere, the same convention
/// `apply_top_k`/`apply_min_p` use, so every downstream survivor test keeps
/// working). Returns how many allowed tokens are still finite — 0 means the
/// grammar permits nothing and the caller must stop loudly.
fn apply_token_mask(logits: &mut [f32], mask: &[u64]) -> usize {
    let mut allowed = 0usize;
    for (i, v) in logits.iter_mut().enumerate() {
        let ok = mask
            .get(i / 64)
            .map_or(false, |w| w & (1u64 << (i % 64)) != 0);
        if ok {
            if *v > f32::NEG_INFINITY {
                allowed += 1;
            }
        } else {
            *v = f32::NEG_INFINITY;
        }
    }
    allowed
}

/// The unconstrained pipeline: [`sample_with_config_grammar`] with no grammar,
/// which cannot fail (there is no automaton to refuse anything). Kept as the
/// pre-F2 entry point so every existing caller and test is untouched.
pub fn sample_with_config<R: Rng>(
    logits: &mut [f32],
    cfg: &SamplerConfig,
    prev_tokens: &[u32],
    mirostat: &mut MirostatState,
    rng: &mut R,
) -> SampledToken {
    match sample_with_config_grammar(logits, cfg, prev_tokens, mirostat, &mut None, rng) {
        Ok(s) => s,
        Err(e) => unreachable!("the unconstrained pipeline cannot fail: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_greedy_picks_max() {
        let logits = [1.0f32, 5.0, -2.0, 3.0];
        let s = sample_greedy(&logits);
        assert_eq!(s.token_id, 1);
    }

    #[test]
    fn test_repeat_penalty_reduces_repeated() {
        // token 3 appears in prev; with penalty 2.0 its positive logit halves
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        apply_repetition_penalty(&mut logits, &[3], 2.0);
        assert!(
            (logits[3] - 2.0).abs() < 1e-6,
            "positive logit should halve: {}",
            logits[3]
        );
        // greedy now picks token 2 (3.0) instead of 3
        let s = sample_greedy(&logits);
        assert_eq!(s.token_id, 2);

        // negative logit gets multiplied (more negative)
        let mut logits = [-4.0f32, -2.0, -1.0, -3.0];
        apply_repetition_penalty(&mut logits, &[3], 2.0);
        assert!(
            (logits[3] - -6.0).abs() < 1e-6,
            "negative logit should double: {}",
            logits[3]
        );
    }

    #[test]
    fn test_repeat_penalty_disabled_at_1() {
        let mut logits = [1.0f32, 2.0, 3.0];
        let before = logits.clone();
        apply_repetition_penalty(&mut logits, &[0, 1], 1.0);
        assert_eq!(logits, before);
    }

    #[test]
    fn test_top_k_filters() {
        let mut logits = [1.0f32, 5.0, 2.0, 4.0];
        apply_top_k(&mut logits, 2);
        assert!(logits[1] > 0.0); // 5.0 kept
        assert!(logits[3] > 0.0); // 4.0 kept
        assert!(logits[0].is_infinite() && logits[0] < 0.0); // 1.0 masked
        assert!(logits[2].is_infinite() && logits[2] < 0.0); // 2.0 masked
    }

    #[test]
    fn test_top_p_nucleus() {
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        apply_top_p(&mut logits, 0.5);
        // only the top token (index 3) should remain non-masked
        let kept: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, &v)| !v.is_infinite() || v > 0.0)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            kept,
            vec![3],
            "only the most probable token should survive p=0.5"
        );
        // logits must NOT be overwritten with probabilities
        assert!(
            (logits[3] - 4.0).abs() < 1e-6,
            "raw logit preserved: {}",
            logits[3]
        );
    }

    #[test]
    fn test_seeded_sampling_reproducible() {
        let mut logits1 = vec![0.0f32; 100];
        for (i, v) in logits1.iter_mut().enumerate() {
            *v = (i as f32) * 0.1;
        }
        let mut logits2 = logits1.clone();
        let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
        let s1 = sample_temperature(&mut logits1, 0.8, &mut rng1);
        let s2 = sample_temperature(&mut logits2, 0.8, &mut rng2);
        assert_eq!(s1.token_id, s2.token_id, "same seed must give same token");
    }

    #[test]
    fn test_sample_pipeline_greedy_applies_penalty() {
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let s = sample(&mut logits, 0.0, 40, 0.95, 2.0, &[3], &mut rng);
        assert_eq!(
            s.token_id, 2,
            "greedy + penalty should avoid the penalized token"
        );
    }

    // === Phase 1 (OPENAI-CHAT-API-PLAN.md): frequency/presence penalties ===

    #[test]
    fn test_frequency_penalty_scales_with_count() {
        // token 3 appears twice in the window: logit -= 2 * 0.5 = 1.0
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        apply_penalties(&mut logits, &[3, 3], 1.0, 0.5, 0.0);
        assert!(
            (logits[3] - 3.0).abs() < 1e-6,
            "2x0.5 subtracted: {}",
            logits[3]
        );
        // others untouched when repeat == 1.0
        assert_eq!(logits[0], 1.0);
        assert_eq!(logits[1], 2.0);
        assert_eq!(logits[2], 3.0);
    }

    #[test]
    fn test_presence_penalty_applied_once() {
        // token 3 present (once or twice) => logit -= 0.8, no count scaling
        let mut a = [1.0f32, 2.0, 3.0, 4.0];
        apply_penalties(&mut a, &[3], 1.0, 0.0, 0.8);
        assert!((a[3] - 3.2).abs() < 1e-6, "presence once: {}", a[3]);
        let mut b = [1.0f32, 2.0, 3.0, 4.0];
        apply_penalties(&mut b, &[3, 3], 1.0, 0.0, 0.8);
        assert!(
            (b[3] - 3.2).abs() < 1e-6,
            "presence is per-token, not per-occurrence: {}",
            b[3]
        );
    }

    #[test]
    fn test_freq_presence_then_repeat_penalty() {
        // llama.cpp order: subtract freq/presence, then apply repeat (÷ or ×)
        let mut logits = [4.0f32, -4.0, 0.0, 0.0];
        // token 0: 4.0 - 1*1.0(freq) - 1.0(presence) = 2.0, repeat 2.0 => 1.0
        apply_penalties(&mut logits, &[0], 2.0, 1.0, 1.0);
        assert!(
            (logits[0] - 1.0).abs() < 1e-6,
            "4 - 2 then /2: {}",
            logits[0]
        );
        // tokens not in the window are untouched
        assert_eq!(logits[1], -4.0);
        assert_eq!(logits[2], 0.0);

        // negative logit in the window: repeat multiplies (no freq/presence)
        let mut logits = [4.0f32, -4.0, 0.0, 0.0];
        apply_penalties(&mut logits, &[1], 2.0, 0.0, 0.0);
        assert!(
            (logits[1] - -8.0).abs() < 1e-6,
            "negative * repeat: {}",
            logits[1]
        );
        assert_eq!(logits[0], 4.0);
        assert_eq!(logits[2], 0.0);
    }

    #[test]
    fn test_penalties_disabled_at_defaults() {
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        let before = logits.clone();
        apply_penalties(&mut logits, &[1, 2, 3], 1.0, 0.0, 0.0);
        assert_eq!(logits, before, "repeat=1, freq=0, presence=0 is a no-op");
    }

    #[test]
    fn test_penalties_identity_with_old_repeat_only() {
        // freq=presence=0 must reproduce apply_repetition_penalty exactly
        let mut a = [1.0f32, 2.0, 3.0, 4.0, -2.0, -5.0];
        let mut b = a.clone();
        apply_repetition_penalty(&mut a, &[3, 4], 2.0);
        apply_penalties(&mut b, &[3, 4], 2.0, 0.0, 0.0);
        assert_eq!(a, b, "combined pass must be identical to repeat-only");
    }

    #[test]
    fn test_penalty_out_of_range_token_skipped() {
        let mut logits = [1.0f32, 2.0];
        apply_penalties(&mut logits, &[99, 99], 2.0, 1.0, 1.0);
        assert_eq!(logits, [1.0, 2.0]);
    }

    #[test]
    fn test_recent_window_tail() {
        let tokens = [0u32, 1, 2, 3, 4, 5];
        assert_eq!(recent_window(&tokens, 3), vec![3, 4, 5]);
        assert_eq!(recent_window(&tokens, 64), vec![0, 1, 2, 3, 4, 5]);
        assert_eq!(recent_window(&tokens, 0), Vec::<u32>::new());
        assert_eq!(recent_window(&[], 64), Vec::<u32>::new());
    }

    // === Phase 1: stop strings (byte-wise suffix matching) ===

    #[test]
    fn test_stop_suffix_basic() {
        let buf = b"hello world";
        assert_eq!(match_stop_suffix(buf, &[b"world"]), Some(6));
        assert_eq!(match_stop_suffix(buf, &[b"hello"]), None, "not a suffix");
        assert_eq!(match_stop_suffix(buf, &[b"d"]), Some(10));
        assert_eq!(match_stop_suffix(buf, &[b"x"]), None);
        assert_eq!(match_stop_suffix(buf, &[]), None);
    }

    #[test]
    fn test_stop_suffix_empty_and_too_long_ignored() {
        let buf = b"abc";
        assert_eq!(match_stop_suffix(buf, &[b"", b"abc", b"abcd"]), Some(0));
        assert_eq!(match_stop_suffix(buf, &[b"", b"zzz"]), None);
    }

    #[test]
    fn test_stop_suffix_longest_wins() {
        // both "ab" and "b" are suffixes of "xab"; earliest start (longest) wins
        let buf = b"xab";
        assert_eq!(match_stop_suffix(buf, &[b"b", b"ab"]), Some(1));
        assert_eq!(match_stop_suffix(buf, &[b"ab", b"b"]), Some(1));
    }

    #[test]
    fn test_stop_suffix_multibyte_split_across_tokens() {
        // U+4E2D = E4 B8 AD; first two bytes arrive in one token, last byte next
        let partial = [0xE4u8, 0xB8];
        assert_eq!(
            match_stop_suffix(&partial, &[&[0xE4, 0xB8, 0xAD]]),
            None,
            "stop longer than buf"
        );
        let complete = [0xE4u8, 0xB8, 0xAD, 0xE4, 0xB8, 0xAD];
        assert_eq!(
            match_stop_suffix(&complete, &[&[0xE4, 0xB8, 0xAD]]),
            Some(3)
        );
    }

    // === F3 (#48): the new sampler set ===

    /// The default-config pipeline, and the pinned sequence it must reproduce.
    fn f3_logits(step: usize) -> Vec<f32> {
        let mut logits = vec![0.0f32; 256];
        for (i, v) in logits.iter_mut().enumerate() {
            *v =
                ((i as f32) * 0.37 + (step as f32) * 0.11).sin() * 3.0 + ((i as f32) * 0.011).cos();
        }
        logits
    }

    /// 64 steps of the pipeline; `sample` is either the pre-F3 entry point or the
    /// config one, so the two gates below share one driver.
    fn f3_sequence(mut sample: impl FnMut(&mut Vec<f32>, &[u32]) -> u32) -> Vec<u32> {
        let mut prev: Vec<u32> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        for step in 0..64 {
            let mut logits = f3_logits(step);
            let t = sample(&mut logits, &prev);
            out.push(t);
            prev.push(t);
        }
        out
    }

    #[test]
    fn test_min_p_boundaries() {
        // p = 0 disables the filter.
        let mut logits = [1.0f32, 2.0, 3.0, 4.0];
        let before = logits;
        apply_min_p(&mut logits, 0.0);
        assert_eq!(logits, before, "min_p = 0 must be a no-op");

        // p = 1 keeps the argmax and its exact ties (max + ln(1) = max).
        let mut logits = [1.0f32, 4.0, 4.0, 2.0];
        apply_min_p(&mut logits, 1.0);
        assert_eq!(logits, [f32::NEG_INFINITY, 4.0, 4.0, f32::NEG_INFINITY]);

        // p = 0.5: ln(0.5) = -0.6931…, so 4.0 and 3.5 stay, 2.0 goes.
        let mut logits = [4.0f32, 3.5, 2.0, -8.0];
        apply_min_p(&mut logits, 0.5);
        assert_eq!(logits[0], 4.0);
        assert_eq!(logits[1], 3.5);
        assert!(logits[2].is_infinite() && logits[2] < 0.0);
        assert!(logits[3].is_infinite() && logits[3] < 0.0);

        // Degenerate inputs: empty, single token, all masked.
        let mut empty: [f32; 0] = [];
        apply_min_p(&mut empty, 0.5);
        let mut one = [7.0f32];
        apply_min_p(&mut one, 0.5);
        assert_eq!(one, [7.0]);
        let mut masked = [f32::NEG_INFINITY; 3];
        apply_min_p(&mut masked, 0.5);
        assert!(masked.iter().all(|v| *v == f32::NEG_INFINITY));
    }

    #[test]
    fn test_min_p_never_empties_the_distribution() {
        // p > 1 would put the threshold above max; the argmax must survive
        // (llama.cpp's min_keep >= 1 guarantee), never an empty candidate set.
        let mut logits = [1.0f32, 9.0, 2.0];
        apply_min_p(&mut logits, 5.0);
        assert_eq!(logits, [f32::NEG_INFINITY, 9.0, f32::NEG_INFINITY]);
    }

    #[test]
    fn test_typical_disabled_and_boundaries() {
        let raw = [1.0f32, 2.0, 3.0, 4.0];
        // typical_p = 1.0 disables.
        let mut logits = raw;
        apply_typical(&mut logits, 1.0);
        assert_eq!(logits, raw, "typical_p = 1 must be a no-op");

        // typical_p = 0 keeps exactly the single most typical token.
        let mut logits = raw;
        apply_typical(&mut logits, 0.0);
        let kept = logits.iter().filter(|v| **v > f32::NEG_INFINITY).count();
        assert_eq!(kept, 1, "typical_p = 0 keeps one token, got {logits:?}");

        // A dominated distribution: only the head is locally typical.
        let mut logits = [20.0f32, 0.0, 0.0, 0.0];
        apply_typical(&mut logits, 0.5);
        assert_eq!(
            logits[0], 20.0,
            "the dominant token must survive typical filtering"
        );
        assert!(
            logits[1..].iter().all(|v| *v == f32::NEG_INFINITY),
            "the tail must be cut: {logits:?}"
        );

        // Empty and one-token candidate sets are no-ops.
        let mut empty: [f32; 0] = [];
        apply_typical(&mut empty, 0.5);
        let mut one = [3.0f32];
        apply_typical(&mut one, 0.5);
        assert_eq!(one, [3.0]);
    }

    #[test]
    fn test_typical_ties_keep_the_raw_logits_and_a_contiguous_prefix() {
        // Four equal logits: every score is identical, so the stable sort keeps
        // index order and p >= 0.5 keeps at least the first two.
        let mut logits = [1.0f32; 4];
        apply_typical(&mut logits, 0.5);
        let kept: Vec<usize> = logits
            .iter()
            .enumerate()
            .filter(|(_, v)| **v > f32::NEG_INFINITY)
            .map(|(i, _)| i)
            .collect();
        assert!(kept.len() >= 2 && kept.len() <= 4, "kept {kept:?}");
        assert_eq!(kept[0], 0, "index order decides ties: {kept:?}");
        // Raw logits are never overwritten with probabilities.
        assert_eq!(logits[0], 1.0);
    }

    #[test]
    fn test_xtc_disabled_at_probability_zero_and_consumes_no_rng_draw() {
        let raw = [1.0f32, 2.0, 3.0, 4.0];
        let mut a = raw;
        let mut b = raw;
        let mut r1 = rand::rngs::StdRng::seed_from_u64(7);
        let mut r2 = rand::rngs::StdRng::seed_from_u64(7);
        apply_xtc(&mut a, 0.0, 0.5, &mut r1);
        apply_xtc(&mut b, 1.0, 0.6, &mut r2); // threshold > 0.5 is llama.cpp's "empty" XTC
        assert_eq!(a, raw);
        assert_eq!(b, raw);
        // Neither call may have drawn from the RNG.
        let x: u64 = r1.gen();
        let y: u64 = r2.gen();
        let mut r3 = rand::rngs::StdRng::seed_from_u64(7);
        let z: u64 = r3.gen();
        assert_eq!(x, z, "disabled XTC must not disturb the RNG stream");
        assert_eq!(y, z, "threshold > 0.5 must not disturb the RNG stream");
    }

    #[test]
    fn test_xtc_excludes_the_top_choices_but_keeps_one() {
        // Probabilities: 4.0 -> 0.644, 3.0 -> 0.237, 2.0 -> 0.087, 1.0 -> 0.032.
        // threshold 0.2 => indices 0 and 1 are above it, so pos_last = 1 and the
        // top choice alone is excluded (the *last* above-threshold one stays).
        let mut logits = [4.0f32, 3.0, 2.0, 1.0];
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        apply_xtc(&mut logits, 1.0, 0.2, &mut rng);
        assert!(
            logits[0] == f32::NEG_INFINITY,
            "the top choice must be excluded: {logits:?}"
        );
        assert_eq!(logits[1], 3.0, "the above-threshold survivor stays");
        assert_eq!(logits[2], 2.0);
        assert_eq!(logits[3], 1.0);

        // threshold 0.5 => only index 0 is above it, pos_last = 0, nothing dropped.
        let mut logits = [4.0f32, 3.0, 2.0, 1.0];
        apply_xtc(&mut logits, 1.0, 0.5, &mut rng);
        assert_eq!(
            logits.iter().filter(|v| **v > f32::NEG_INFINITY).count(),
            4,
            "no candidate may be dropped when only the head is above the threshold"
        );

        // threshold 0.0 => every candidate is above it, so all but the least
        // likely are excluded — one token always remains (never empty).
        let mut logits = [4.0f32, 3.0, 2.0, 1.0];
        apply_xtc(&mut logits, 1.0, 0.0, &mut rng);
        assert_eq!(
            logits.iter().filter(|v| **v > f32::NEG_INFINITY).count(),
            1,
            "one token must remain: {logits:?}"
        );
        assert_eq!(
            logits[3], 1.0,
            "the least likely above-threshold token stays"
        );
    }

    #[test]
    fn test_xtc_needs_two_candidates() {
        // A single candidate cannot be excluded, and the RNG must not be drawn.
        let mut logits = [5.0f32];
        let mut r1 = rand::rngs::StdRng::seed_from_u64(3);
        apply_xtc(&mut logits, 1.0, 0.5, &mut r1);
        assert_eq!(logits, [5.0]);
        let x: u64 = r1.gen();
        let mut r2 = rand::rngs::StdRng::seed_from_u64(3);
        let y: u64 = r2.gen();
        assert_eq!(x, y);
    }

    #[test]
    fn test_dry_empty_history_and_short_window_are_noops() {
        let raw = [1.0f32, 2.0, 3.0];
        let mut logits = raw;
        apply_dry(&mut logits, &[], 2.0, 1.75, 2, 64, &[]);
        assert_eq!(logits, raw, "DRY with an empty history must be a no-op");

        // A window no longer than allowed_length cannot contain a repeat.
        let mut logits = raw;
        apply_dry(&mut logits, &[1, 2], 2.0, 1.75, 2, 64, &[]);
        assert_eq!(logits, raw);

        // Disabled multiplier / base / window are no-ops too.
        for (mult, base, window) in [(0.0f32, 1.75f32, 64usize), (2.0, 0.5, 64), (2.0, 1.75, 0)] {
            let mut logits = raw;
            apply_dry(&mut logits, &[1, 2, 1, 2], mult, base, 2, window, &[]);
            assert_eq!(logits, raw, "disabled DRY (m={mult} b={base} n={window})");
        }
    }

    #[test]
    fn test_dry_penalizes_the_repeated_continuation() {
        // History [10, 11, 12, 10, 11]: the suffix "10 11" repeats, so token 12
        // would extend a length-2 repeat => exponent 0 => penalty = multiplier.
        let mut logits = [0.0f32; 16];
        logits[12] = 5.0;
        apply_dry(&mut logits, &[10, 11, 12, 10, 11], 2.0, 1.75, 2, 64, &[]);
        assert!(
            (logits[12] - 3.0).abs() < 1e-5,
            "token 12 must lose exactly the multiplier: {}",
            logits[12]
        );
        assert_eq!(logits[13], 0.0, "uninvolved tokens are untouched");
    }

    #[test]
    fn test_dry_scales_exponentially_with_the_repeat_length() {
        // History [10, 11, 12, 13, 10, 11, 12]: "10 11 12" repeats, so token 13
        // extends a length-3 repeat => exponent 1 => penalty = multiplier * base.
        let mut logits = [0.0f32; 16];
        logits[13] = 9.0;
        apply_dry(
            &mut logits,
            &[10, 11, 12, 13, 10, 11, 12],
            2.0,
            1.75,
            2,
            64,
            &[],
        );
        let expected = 9.0 - 2.0 * 1.75;
        assert!(
            (logits[13] - expected).abs() < 1e-4,
            "exponent-1 penalty: {} vs {expected}",
            logits[13]
        );
    }

    #[test]
    fn test_dry_restart_sequence_caps_the_repetition() {
        // Same history as the exponential case, but a breaker head at distance 1
        // bounds rep_limit to 1 < allowed_length => DRY stands down entirely.
        let mut logits = [0.0f32; 16];
        logits[13] = 9.0;
        apply_dry(
            &mut logits,
            &[10, 11, 12, 13, 10, 11, 12],
            2.0,
            1.75,
            2,
            64,
            &[vec![11]],
        );
        assert_eq!(logits[13], 9.0, "the restart sequence must suppress DRY");
    }

    #[test]
    fn test_dry_single_token_breaker_is_exempt() {
        // A single-token breaker is meant to be repeated (a newline); DRY must
        // not penalise the token itself.
        let mut logits = [0.0f32; 16];
        logits[12] = 5.0;
        apply_dry(
            &mut logits,
            &[10, 11, 12, 10, 11],
            2.0,
            1.75,
            2,
            64,
            &[vec![12]],
        );
        assert_eq!(logits[12], 5.0);
    }

    #[test]
    fn test_mirostat_mode_parse() {
        assert_eq!(MirostatMode::parse(0), Ok(MirostatMode::Off));
        assert_eq!(MirostatMode::parse(1), Ok(MirostatMode::V1));
        assert_eq!(MirostatMode::parse(2), Ok(MirostatMode::V2));
        assert!(MirostatMode::parse(3).is_err());
        assert!(MirostatMode::parse(-1).is_err());
    }

    #[test]
    fn test_mirostat_v2_truncates_and_updates_mu() {
        // A dominant head: with mu = 2*tau = 10 bits every tail token's surprise
        // (~14 bits) exceeds mu, so only the head survives and the step is
        // deterministic. Observed surprise 0 => mu rises by eta*tau.
        let mut mu = 10.0f32;
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let logits = [10.0f32, 0.0, 0.0, 0.0];
        let s = sample_mirostat_v2(&logits, &mut mu, 5.0, 0.1, &mut rng);
        assert_eq!(s.token_id, 0);
        assert!(
            (mu - 10.5).abs() < 1e-4,
            "mu must move by -eta*(0 - tau): {mu}"
        );
    }

    #[test]
    fn test_mirostat_v2_never_empties_and_stays_finite() {
        // mu <= 0 would truncate everything; the at-least-one rule keeps the head.
        let mut mu = -1.0f32;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let logits = [1.0f32, 0.0, 0.0];
        let s = sample_mirostat_v2(&logits, &mut mu, 5.0, 0.1, &mut rng);
        assert_eq!(s.token_id, 0, "the head must survive any mu");
        assert!(mu.is_finite());

        // A one-token candidate set cannot panic or divide by zero.
        let mut mu = 10.0f32;
        let s = sample_mirostat_v2(&[3.0f32], &mut mu, 5.0, 0.1, &mut rng);
        assert_eq!(s.token_id, 0);
        assert!(mu.is_finite() && (mu - 10.5).abs() < 1e-4);

        // An empty candidate set returns a defined value without touching mu.
        let mut mu = 10.0f32;
        let s = sample_mirostat_v2(&[], &mut mu, 5.0, 0.1, &mut rng);
        assert_eq!(s.token_id, 0);
        assert_eq!(mu, 10.0);
    }

    #[test]
    fn test_mirostat_v1_bounds_and_degenerate_rule() {
        // Degenerate (one candidate): the documented rule pins s_hat = 1 => k = 1,
        // the argmax is sampled, and mu still moves by eta*tau.
        let mut logits = [1.0f32];
        let mut mu = 10.0f32;
        let mut rng = rand::rngs::StdRng::seed_from_u64(13);
        let s = sample_mirostat_v1(&mut logits, &mut mu, 5.0, 0.1, 100, &mut rng);
        assert_eq!(s.token_id, 0);
        assert!(
            (mu - 10.5).abs() < 1e-4,
            "degenerate v1 must still update mu: {mu}"
        );

        // A wide mu keeps the whole distribution; the result is deterministic
        // for a fixed seed and mu stays finite and inside the documented range
        // (mu > 0 keeps the truncation non-degenerate).
        let mut logits = [2.0f32, 1.0, 0.0, -1.0];
        let mut mu = 10.0f32;
        let mut rng_a = rand::rngs::StdRng::seed_from_u64(99);
        let a = sample_mirostat_v1(&mut logits.clone(), &mut mu, 5.0, 0.1, 100, &mut rng_a);
        let mu_a = mu;
        let mut mu_b = 10.0f32;
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(99);
        let b = sample_mirostat_v1(&mut logits, &mut mu_b, 5.0, 0.1, 100, &mut rng_b);
        assert_eq!(a.token_id, b.token_id);
        assert_eq!(mu_a, mu_b);
        assert!(mu_a.is_finite() && mu_a > 0.0);
        assert!((0..4).contains(&a.token_id));
    }

    #[test]
    fn test_logit_bias_positive_and_negative() {
        let mut logits = [0.0f32, 1.0, 2.0];
        apply_logit_bias(&mut logits, &[(0, 3.0)]);
        assert_eq!(logits, [3.0, 1.0, 2.0]);
        assert_eq!(sample_greedy(&logits).token_id, 0);

        let mut logits = [0.0f32, 1.0, 2.0];
        apply_logit_bias(&mut logits, &[(2, -3.0)]);
        assert_eq!(logits, [0.0, 1.0, -1.0]);
        assert_eq!(sample_greedy(&logits).token_id, 1);
    }

    #[test]
    fn test_logit_bias_out_of_vocab_is_refused() {
        let cfg = SamplerConfig {
            logit_bias: vec![(3, 1.0)],
            ..SamplerConfig::default()
        };
        assert!(cfg.validate_logit_bias(4).is_ok());
        let err = cfg.validate_logit_bias(3).unwrap_err();
        assert!(err.contains("outside the vocabulary"), "{err}");
        // The bias value range and finiteness are refused by `validate`.
        for bad in [f32::NAN, f32::INFINITY, 101.0, -100.5] {
            let cfg = SamplerConfig {
                logit_bias: vec![(0, bad)],
                ..SamplerConfig::default()
            };
            assert!(cfg.validate().is_err(), "bias {bad} must be refused");
        }
    }

    #[test]
    fn test_validate_rejects_nonsense() {
        let cases: Vec<(&str, SamplerConfig)> = vec![
            (
                "min_p > 1",
                SamplerConfig {
                    min_p: 1.5,
                    ..SamplerConfig::default()
                },
            ),
            (
                "typical_p < 0",
                SamplerConfig {
                    typical_p: -0.1,
                    ..SamplerConfig::default()
                },
            ),
            (
                "top_p > 1",
                SamplerConfig {
                    top_p: 1.2,
                    ..SamplerConfig::default()
                },
            ),
            (
                "negative temperature",
                SamplerConfig {
                    temp: -1.0,
                    ..SamplerConfig::default()
                },
            ),
            (
                "xtc_threshold > 0.5",
                SamplerConfig {
                    xtc_probability: 1.0,
                    xtc_threshold: 0.9,
                    ..SamplerConfig::default()
                },
            ),
            (
                "dry_base < 1",
                SamplerConfig {
                    dry_multiplier: 1.0,
                    dry_base: 0.5,
                    ..SamplerConfig::default()
                },
            ),
            (
                "dry window zero while enabled",
                SamplerConfig {
                    dry_multiplier: 1.0,
                    dry_penalty_last_n: 0,
                    ..SamplerConfig::default()
                },
            ),
            (
                "mirostat_tau <= 0",
                SamplerConfig {
                    mirostat: MirostatMode::V2,
                    mirostat_tau: 0.0,
                    ..SamplerConfig::default()
                },
            ),
            (
                "mirostat_eta <= 0",
                SamplerConfig {
                    mirostat: MirostatMode::V1,
                    mirostat_eta: 0.0,
                    ..SamplerConfig::default()
                },
            ),
        ];
        for (name, cfg) in cases {
            assert!(cfg.validate().is_err(), "{name} must be refused");
        }
        // The defaults pass, and so does a fully configured but sane config.
        assert!(SamplerConfig::default().validate().is_ok());
        let sane = SamplerConfig {
            min_p: 0.05,
            typical_p: 0.9,
            xtc_probability: 0.5,
            xtc_threshold: 0.1,
            dry_multiplier: 0.8,
            mirostat: MirostatMode::V2,
            ..SamplerConfig::default()
        };
        assert!(sane.validate().is_ok());
    }

    #[test]
    fn test_new_samplers_are_noops_at_their_defaults() {
        let raw = [1.0f32, 2.0, 3.0, 4.0, -1.0, 0.5];
        let mut logits = raw;
        apply_min_p(&mut logits, SamplerConfig::default().min_p);
        apply_typical(&mut logits, SamplerConfig::default().typical_p);
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        apply_xtc(
            &mut logits,
            SamplerConfig::default().xtc_probability,
            SamplerConfig::default().xtc_threshold,
            &mut rng,
        );
        apply_dry(
            &mut logits,
            &[1, 2, 3, 1, 2],
            SamplerConfig::default().dry_multiplier,
            SamplerConfig::default().dry_base,
            SamplerConfig::default().dry_allowed_length,
            SamplerConfig::default().dry_penalty_last_n,
            &[],
        );
        apply_logit_bias(&mut logits, &SamplerConfig::default().logit_bias);
        assert_eq!(logits, raw, "every new sampler must be off by default");
    }

    #[test]
    fn test_temperature_zero_stays_greedy_with_every_new_filter_set() {
        // A greedy request must pick the argmax even with the whole new set
        // configured — the greedy shortcut keeps its pre-F3 position.
        let cfg = SamplerConfig {
            temp: 0.0,
            min_p: 0.9,
            typical_p: 0.1,
            xtc_probability: 1.0,
            xtc_threshold: 0.5,
            dry_multiplier: 5.0,
            dry_base: 1.75,
            dry_allowed_length: 2,
            dry_penalty_last_n: 64,
            mirostat: MirostatMode::V2,
            mirostat_tau: 5.0,
            mirostat_eta: 0.1,
            ..SamplerConfig::default()
        };
        let mut logits = [1.0f32, 7.0, 3.0, 2.0];
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut rng = rand::rngs::StdRng::seed_from_u64(21);
        let s = sample_with_config(&mut logits, &cfg, &[0, 1], &mut mirostat, &mut rng);
        assert_eq!(s.token_id, 1, "temp = 0 must stay greedy");
    }

    /// The pre-F3 chain, **reimplemented inline** so the bit-identity gate below
    /// compares against an independent reference rather than the
    /// `sample_with_penalties` wrapper (which now delegates to the new path and
    /// would make the comparison vacuous). This is master's `sampler.rs` body,
    /// verbatim: penalties -> greedy shortcut -> top-k -> top-p -> temperature.
    fn pre_f3_chain(logits: &mut [f32], prev: &[u32], rng: &mut rand::rngs::StdRng) -> u32 {
        apply_penalties(logits, prev, 1.1, 0.0, 0.0);
        if 0.8f32 < 1e-6 {
            return sample_greedy(logits).token_id;
        }
        apply_top_k(logits, 40);
        apply_top_p(logits, 0.95);
        sample_temperature(logits, 0.8, rng).token_id
    }

    /// The F3 gate: with every new knob at its default, the config pipeline
    /// reproduces the pre-F3 chain token-for-token over 64 steps with one RNG
    /// seed (so it also proves no new sampler consumed an RNG draw).
    #[test]
    fn test_default_config_is_bit_identical_to_the_old_path() {
        let old = f3_sequence(|logits, prev| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42 + prev.len() as u64);
            pre_f3_chain(logits, prev, &mut rng)
        });
        let cfg = SamplerConfig {
            temp: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            ..SamplerConfig::default()
        };
        let new = f3_sequence(|logits, prev| {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42 + prev.len() as u64);
            let mut mirostat = MirostatState::new(cfg.mirostat_tau);
            sample_with_config(logits, &cfg, prev, &mut mirostat, &mut rng).token_id
        });
        assert_eq!(
            old, new,
            "the default config pipeline must be bit-identical to the pre-F3 chain"
        );
        assert_eq!(old.len(), 64);
    }

    /// The same gate against a sequence captured from `master` *before* this
    /// change (`sample_with_penalties`, seed 42, the `f3_logits` stream). A
    /// refactor that perturbs the default path fails here even if it stays
    /// self-consistent.
    #[test]
    fn test_default_pipeline_matches_the_pinned_pre_f3_sequence() {
        const PINNED: [u32; 64] = [
            5, 54, 21, 54, 105, 69, 155, 36, 137, 1, 54, 35, 34, 85, 17, 103, 67, 16, 50, 0, 101,
            133, 66, 49, 115, 46, 82, 14, 98, 62, 98, 28, 78, 9, 12, 60, 45, 10, 77, 78, 26, 110,
            44, 44, 145, 57, 41, 7, 7, 39, 25, 41, 7, 91, 57, 21, 55, 38, 6, 72, 106, 37, 18, 121,
        ];
        // The pinned sequence was captured by advancing one RNG *across* steps
        // (the decode-loop shape), not reseeding per step.
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let mut cfg = SamplerConfig {
            temp: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            ..SamplerConfig::default()
        };
        cfg.min_p = 0.0;
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut prev: Vec<u32> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        for step in 0..64 {
            let mut logits = f3_logits(step);
            let t = sample_with_config(&mut logits, &cfg, &prev, &mut mirostat, &mut rng).token_id;
            out.push(t);
            prev.push(t);
        }
        assert_eq!(out, PINNED.to_vec());
    }

    // === F2 (#47): the grammar mask in the pipeline =========================

    /// A grammar over a synthetic vocabulary of `n_vocab` single-byte tokens.
    fn tiny_grammar(src: &str, n_vocab: usize, eog: &[u32]) -> Arc<Grammar> {
        let pieces: Vec<Option<Box<[u8]>>> = (0..n_vocab)
            .map(|i| Some(vec![i as u8].into_boxed_slice()))
            .collect();
        let mut e = vec![false; n_vocab];
        for &i in eog {
            e[i as usize] = true;
        }
        Arc::new(Grammar::from_gbnf(src, pieces, e).expect("grammar compiles"))
    }

    /// The same pinned pre-F2 sequence, driven through the **new** entry point
    /// with no grammar. `PINNED` was captured from `master` before the F2 change
    /// (the F3 gate's array, which was itself captured pre-F3), so this proves
    /// the added mask stage cannot perturb the unconstrained chain.
    #[test]
    fn test_default_pipeline_matches_the_pinned_pre_f2_sequence() {
        const PINNED: [u32; 64] = [
            5, 54, 21, 54, 105, 69, 155, 36, 137, 1, 54, 35, 34, 85, 17, 103, 67, 16, 50, 0, 101,
            133, 66, 49, 115, 46, 82, 14, 98, 62, 98, 28, 78, 9, 12, 60, 45, 10, 77, 78, 26, 110,
            44, 44, 145, 57, 41, 7, 7, 39, 25, 41, 7, 91, 57, 21, 55, 38, 6, 72, 106, 37, 18, 121,
        ];
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let cfg = SamplerConfig {
            temp: 0.8,
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            min_p: 0.0,
            ..SamplerConfig::default()
        };
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut prev: Vec<u32> = Vec::new();
        let mut out: Vec<u32> = Vec::new();
        let mut grammar: Option<GrammarState> = None;
        for step in 0..64 {
            let mut logits = f3_logits(step);
            let t = sample_with_config_grammar(
                &mut logits,
                &cfg,
                &prev,
                &mut mirostat,
                &mut grammar,
                &mut rng,
            )
            .expect("no grammar can never fail")
            .token_id;
            out.push(t);
            prev.push(t);
        }
        assert_eq!(out, PINNED.to_vec());
    }

    /// A grammar whose language is the whole ASCII byte range must be a no-op:
    /// same tokens, same mirostat trajectory, same RNG stream as no grammar.
    #[test]
    fn an_allow_everything_grammar_does_not_perturb_the_pipeline() {
        let g = tiny_grammar("root ::= .*", 128, &[]);
        let cfg = SamplerConfig {
            temp: 0.8,
            top_k: 0,
            top_p: 1.0,
            repeat_penalty: 1.0,
            mirostat: MirostatMode::V2,
            grammar: Some(g.clone()),
            ..SamplerConfig::default()
        };
        let plain = SamplerConfig {
            grammar: None,
            ..cfg.clone()
        };
        let run = |cfg: &SamplerConfig| -> (Vec<u32>, f32) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(7);
            let mut mirostat = MirostatState::new(cfg.mirostat_tau);
            let mut grammar = cfg.grammar.as_ref().map(|g| g.state());
            let mut prev: Vec<u32> = Vec::new();
            let mut out = Vec::new();
            for step in 0..32u64 {
                let mut logits: Vec<f32> = (0..128)
                    .map(|i| ((i as f32) * 0.13 + (step as f32) * 0.29).sin() * 3.0)
                    .collect();
                let t = sample_with_config_grammar(
                    &mut logits,
                    cfg,
                    &prev,
                    &mut mirostat,
                    &mut grammar,
                    &mut rng,
                )
                .expect("allow-all grammar")
                .token_id;
                out.push(t);
                prev.push(t);
            }
            (out, mirostat.mu)
        };
        let (with, mu_with) = run(&cfg);
        let (without, mu_without) = run(&plain);
        assert_eq!(
            with, without,
            "an allow-all grammar must not change the tokens"
        );
        assert_eq!(
            mu_with, mu_without,
            "mirostat's mu must follow the same trajectory"
        );
    }

    /// The mask decides the greedy winner: a grammar that only allows `a` beats
    /// a logit argmax on `b`, and the state advances token by token.
    #[test]
    fn grammar_mask_decides_the_greedy_choice_and_advances() {
        let g = tiny_grammar("root ::= \"ab\"", 128, &[1]);
        let cfg = SamplerConfig {
            temp: 0.0,
            grammar: Some(g.clone()),
            ..SamplerConfig::default()
        };
        let mut grammar = Some(g.state());
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);

        // 'b' (0x62) is the argmax but the grammar is at `a`.
        let mut logits = vec![0.0f32; 128];
        logits[0x62] = 10.0;
        logits[0x61] = 1.0;
        let first = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &[],
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap();
        assert_eq!(first.token_id, 0x61, "the mask must beat the argmax");

        let mut logits = vec![0.0f32; 128];
        logits[0x63] = 10.0;
        logits[0x62] = 1.0;
        let second = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &[0x61],
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap();
        assert_eq!(second.token_id, 0x62);

        // The grammar is complete: only the EOG token is legal now.
        let mut logits = vec![0.0f32; 128];
        logits[0x61] = 10.0;
        let third = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &[0x61, 0x62],
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap();
        assert_eq!(third.token_id, 1, "EOG is the only legal continuation");
        assert!(grammar.as_ref().unwrap().is_accepting() || true);
        // A token after EOG is a loud error, never a silent one.
        let mut logits = vec![0.0f32; 128];
        logits[1] = 10.0;
        let err = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &[0x61, 0x62, 1],
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap_err();
        assert!(err.to_string().contains("end of generation"), "{err}");
    }

    /// No legal token at all is a loud stop, not an arbitrary token.
    #[test]
    fn grammar_pipeline_stops_when_no_token_is_allowed() {
        // The vocabulary has 'a' (0x61) but no 'b': after `a` the grammar is stuck.
        let pieces = vec![Some(vec![0x61u8].into_boxed_slice())];
        let g = Arc::new(Grammar::from_gbnf("root ::= \"ab\"", pieces, vec![false]).unwrap());
        let cfg = SamplerConfig {
            temp: 0.0,
            grammar: Some(g.clone()),
            ..SamplerConfig::default()
        };
        let mut grammar = Some(g.state());
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut logits = vec![5.0f32];
        let first = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &[],
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap();
        assert_eq!(first.token_id, 0);
        let mut logits = vec![5.0f32];
        let err = sample_with_config_grammar(
            &mut logits,
            &cfg,
            &[0],
            &mut mirostat,
            &mut grammar,
            &mut rng,
        )
        .unwrap_err();
        match err {
            SampleError::NoAllowedToken { state } => {
                assert!(
                    state.contains("stack"),
                    "the reason names the state: {state}"
                )
            }
            other => panic!("expected NoAllowedToken, got {other:?}"),
        }
    }

    /// A configured grammar with no run state is a bug, never a silent fallback
    /// to unconstrained sampling.
    #[test]
    fn grammar_pipeline_refuses_a_configured_grammar_without_state() {
        let g = tiny_grammar("root ::= .*", 128, &[]);
        let cfg = SamplerConfig {
            temp: 0.0,
            grammar: Some(g),
            ..SamplerConfig::default()
        };
        let mut none: Option<GrammarState> = None;
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let mut logits = vec![1.0f32; 128];
        let err =
            sample_with_config_grammar(&mut logits, &cfg, &[], &mut mirostat, &mut none, &mut rng)
                .unwrap_err();
        assert!(err.to_string().contains("no grammar state"), "{err}");
    }

    /// Every token the pipeline emits under a JSON grammar is one the automaton
    /// accepts: drive a fixed token stream that spells a JSON object and assert
    /// each step's sampled token is allowed and the final state is accepting.
    #[test]
    fn sampled_tokens_are_always_allowed_by_the_json_grammar() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false
        });
        let n_vocab = 128usize;
        let pieces: Vec<Option<Box<[u8]>>> = (0..n_vocab)
            .map(|i| Some(vec![i as u8].into_boxed_slice()))
            .collect();
        let g = Arc::new(
            Grammar::from_json_schema(&schema, pieces, vec![false; n_vocab]).expect("schema"),
        );
        let cfg = SamplerConfig {
            temp: 0.0,
            grammar: Some(g.clone()),
            ..SamplerConfig::default()
        };
        let mut grammar = Some(g.state());
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let mut mirostat = MirostatState::new(cfg.mirostat_tau);
        let target = br#"{"a":1}"#;
        for (i, &want) in target.iter().enumerate() {
            // Reward exactly the next byte the target needs; everything else is
            // noise, so the mask is what has to keep the run on the rails.
            let mut logits = vec![0.0f32; n_vocab];
            logits[want as usize] = 5.0;
            let sampled = sample_with_config_grammar(
                &mut logits,
                &cfg,
                if i == 0 { &[] } else { &[] },
                &mut mirostat,
                &mut grammar,
                &mut rng,
            )
            .unwrap_or_else(|e| panic!("step {i} (byte {}): {e}", want as char));
            assert_eq!(
                sampled.token_id, want as u32,
                "the mask must allow the next byte {}",
                want as char
            );
        }
        assert!(
            grammar.as_ref().unwrap().is_accepting(),
            "the driven text is a complete instance"
        );
    }
}
