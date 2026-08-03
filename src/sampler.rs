// Sampler — repetition penalty, top-k, top-p, temperature (seeded)
// Translated from: llama.cpp/src/llama-sampling.cpp (simplified)

use rand::Rng;

/// Sampling result
#[derive(Debug)]
pub struct SampledToken {
    pub token_id: u32,
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
    SampledToken { token_id: best_id, logit: best_val }
}

/// Repetition penalty: penalize tokens that already appeared in `prev_tokens`.
/// `penalty == 1.0` disables. Positive logits are divided by the penalty
/// (reduced), negative logits are multiplied (pushed further down). This is
/// llama.cpp's `repeat_penalty` applied to the last `repeat_last_n` tokens.
pub fn apply_repetition_penalty(logits: &mut [f32], prev_tokens: &[u32], penalty: f32) {
    if (penalty - 1.0).abs() < 1e-6 || penalty < 1.0 {
        return;
    }
    for &t in prev_tokens {
        let idx = t as usize;
        if idx >= logits.len() {
            continue;
        }
        let v = logits[idx];
        logits[idx] = if v <= 0.0 { v * penalty } else { v / penalty };
    }
}

/// Top-K filtering: keep only top K logits, set the rest to -INFINITY.
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Less));
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
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p <= 0.0 || p >= 1.0 {
        return;
    }
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    // Softmax into a temp array (logits stay raw).
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

    // Softmax
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum = 0.0f64;
    for v in logits.iter_mut() {
        *v = (*v - max_val).exp();
        sum += *v as f64;
    }
    let inv_sum = (1.0 / sum) as f32;
    for v in logits.iter_mut() {
        *v *= inv_sum;
    }

    // Sample from the distribution
    let r: f32 = rng.gen();
    let mut cumulative = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        cumulative += v;
        if r <= cumulative {
            return SampledToken { token_id: i as u32, logit: v };
        }
    }
    SampledToken { token_id: (logits.len() - 1) as u32, logit: logits[logits.len() - 1] }
}

/// Complete sampling pipeline: repetition penalty → top-k → top-p →
/// temperature. `temp < 1e-6` (greedy) skips the stochastic steps but still
/// applies the repetition penalty.
pub fn sample<R: Rng>(logits: &mut [f32], temp: f32, top_k: usize, top_p: f32,
    repeat_penalty: f32, prev_tokens: &[u32], rng: &mut R,
) -> SampledToken {
    apply_repetition_penalty(logits, prev_tokens, repeat_penalty);
    if temp < 1e-6 {
        return sample_greedy(logits);
    }
    apply_top_k(logits, top_k);
    apply_top_p(logits, top_p);
    sample_temperature(logits, temp, rng)
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
        assert!((logits[3] - 2.0).abs() < 1e-6, "positive logit should halve: {}", logits[3]);
        // greedy now picks token 2 (3.0) instead of 3
        let s = sample_greedy(&logits);
        assert_eq!(s.token_id, 2);

        // negative logit gets multiplied (more negative)
        let mut logits = [-4.0f32, -2.0, -1.0, -3.0];
        apply_repetition_penalty(&mut logits, &[3], 2.0);
        assert!((logits[3] - -6.0).abs() < 1e-6, "negative logit should double: {}", logits[3]);
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
        let kept: Vec<usize> = logits.iter().enumerate()
            .filter(|(_, &v)| !v.is_infinite() || v > 0.0).map(|(i, _)| i).collect();
        assert_eq!(kept, vec![3], "only the most probable token should survive p=0.5");
        // logits must NOT be overwritten with probabilities
        assert!((logits[3] - 4.0).abs() < 1e-6, "raw logit preserved: {}", logits[3]);
    }

    #[test]
    fn test_seeded_sampling_reproducible() {
        let mut logits1 = vec![0.0f32; 100];
        for (i, v) in logits1.iter_mut().enumerate() { *v = (i as f32) * 0.1; }
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
        assert_eq!(s.token_id, 2, "greedy + penalty should avoid the penalized token");
    }
}
