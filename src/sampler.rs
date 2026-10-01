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
mod tests;
