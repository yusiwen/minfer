// Sampler — repetition/frequency/presence penalties, top-k, top-p, temperature (seeded)

use rand::Rng;
use std::collections::HashMap;

/// Sampling result
#[derive(Debug)]
pub struct SampledToken {
    pub token_id: u32,
    /// Logit of the sampled token (result metadata; callers read `token_id`).
    #[allow(dead_code)]
    pub logit: f32,
}

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
/// `temp < 1e-6` (greedy) skips the stochastic steps but still applies the
/// penalties.
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
    apply_penalties(
        logits,
        prev_tokens,
        repeat_penalty,
        frequency_penalty,
        presence_penalty,
    );
    if temp < 1e-6 {
        return sample_greedy(logits);
    }
    apply_top_k(logits, top_k);
    apply_top_p(logits, top_p);
    sample_temperature(logits, temp, rng)
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
}
