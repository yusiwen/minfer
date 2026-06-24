// Phase 8: Sampler — greedy, temperature, top-k, top-p
// Translated from: llama.cpp/src/llama-sampling.cpp (simplified)
// Strict 1:1 translation of sampling logic

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

/// Temperature sampling: apply softmax with temperature, then sample
pub fn sample_temperature(logits: &mut [f32], temp: f32) -> SampledToken {
    if temp < 1e-6 {
        return sample_greedy(logits);
    }

    // Apply temperature: divide by temp
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

    // Sample from distribution
    let r: f32 = rand::thread_rng().gen();
    let mut cumulative = 0.0f32;
    for (i, &v) in logits.iter().enumerate() {
        cumulative += v;
        if r <= cumulative {
            return SampledToken { token_id: i as u32, logit: v };
        }
    }
    // Fallback to last token
    SampledToken { token_id: (logits.len() - 1) as u32, logit: logits[logits.len() - 1] }
}

/// Top-K filtering: keep only top K logits, zero out the rest
pub fn apply_top_k(logits: &mut [f32], k: usize) {
    if k == 0 || k >= logits.len() { return; }

    // Find the k-th largest value (sort handles NaN by treating it as smallest)
    let mut sorted = logits.to_vec();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Less));
    let threshold = sorted[k - 1];

    for v in logits.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
}

/// Top-P (nucleus) filtering: keep smallest set of tokens with cumulative prob >= p
pub fn apply_top_p(logits: &mut [f32], p: f32) {
    if p <= 0.0 || p >= 1.0 { return; }

    // Find max (already applied softmax in previous step)
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

    // Sort by probability descending
    let mut indexed: Vec<(usize, f32)> = logits.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Accumulate until p is reached
    let mut cumulative = 0.0f32;
    for (i, &(_, v)) in indexed.iter().enumerate() {
        cumulative += v;
        if cumulative > p {
            // Zero out remaining tokens
            for &(idx, _) in &indexed[i + 1..] {
                logits[idx] = 0.0;
            }
            break;
        }
    }
}

/// Complete sampling: apply temperature, top-k, top-p, then sample
pub fn sample(logits: &mut [f32], temp: f32, top_k: usize, top_p: f32) -> SampledToken {
    if temp < 1e-6 {
        return sample_greedy(logits);
    }
    apply_top_k(logits, top_k);
    apply_top_p(logits, top_p);
    sample_temperature(logits, temp)
}
