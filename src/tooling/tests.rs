//! `#[cfg(test)] mod tests` for `src/tooling.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;
use crate::gguf::GgmlType;

mod bf16;
mod f141_device;
mod f167_qwen3;
mod f16_encode;
mod f6_roundtrip;
mod parse;
mod quantize_bounds;
// === F6 real-model gates (#[ignore]: CI has no checkpoint/model) ===

/// The prompt every F6 logit gate uses. Short enough to prefill quickly on
/// the 0.5B, long enough that a position/weight error moves a logit.
const PROMPT: &str = "The capital of France is";
fn env_path(key: &str, default: &str) -> Option<PathBuf> {
    let p = std::env::var_os(key)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(default.replace('~', &std::env::var("HOME").unwrap())));
    if p.exists() {
        Some(p)
    } else {
        eprintln!("{key} not found at {}; skipping this F6 gate", p.display());
        None
    }
}
fn work_dir(name: &str) -> PathBuf {
    let d = PathBuf::from("/tmp/f6-work").join(name);
    std::fs::create_dir_all(&d).expect("create work dir");
    d
}
fn cached_qwen05() -> PathBuf {
    let home = std::env::var("HOME").unwrap();
    PathBuf::from(home).join(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/\
         qwen2.5-0.5b-instruct-q4_k_m.gguf",
    )
}
/// Greedy continuation `steps` tokens past `prompt`, returning the
/// final-step logits (whole vocabulary) and the sampled token ids.
///
/// Forced onto the CPU plan (`OffloadRequest::Layers(0)`) so the bitwise
/// comparisons are device-independent and no device registration is done —
/// CI has no GPU, and the logit claims here are about *weights*, not backends.
fn logits_greedy(path: &Path, prompt: &str, steps: usize, n_ctx: usize) -> (Vec<f32>, Vec<u32>) {
    let gguf = crate::gguf::load_gguf_model(path).expect("parse GGUF");
    let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx).expect("strict tokenizer load");
    let model =
        crate::models::load_model_with(&gguf, "", crate::graph::offload::OffloadRequest::Layers(0))
            .expect("load model");
    let q = model
        .as_any()
        .downcast_ref::<crate::models::qwen2::Qwen2Model>()
        .expect("Qwen2 model");
    logits_greedy_on(q, &tok.encode(prompt), steps, n_ctx)
}
/// The run itself, on an already-loaded model. Shared by the CPU gates
/// (which load with `Layers(0)`) and #141's device arm (which loads with a
/// full device plan): the numeric comparison must be the same forward code
/// on both sides, differing only in the backend the scheduler assigns.
fn logits_greedy_on(
    q: &crate::models::qwen2::Qwen2Model,
    ids: &[u32],
    steps: usize,
    n_ctx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let nt = ids.len();
    let mut cache = crate::graph::cache::GraphCache::new();
    let mut positions: Vec<usize> = (0..nt).collect();
    let mut logits = crate::models::qwen2::graph::Qwen2Graph::forward_cached(
        q, ids, &positions, 1, n_ctx, &mut cache,
    );
    let mut toks = Vec::with_capacity(steps);
    for step in 0..steps {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        toks.push(next);
        positions = vec![nt + step];
        logits = crate::models::qwen2::graph::Qwen2Graph::forward_cached(
            q,
            &[next],
            &positions,
            1,
            n_ctx,
            &mut cache,
        );
    }
    let _ = positions;
    (logits, toks)
}
/// Compare every tensor payload of two GGUFs by name (offset-independent).
/// Returns the number of tensors compared.
fn assert_tensor_payloads_equal(a_path: &Path, b_path: &Path) -> usize {
    let ma = crate::gguf::load_gguf_model(a_path).expect("a parses");
    let mb = crate::gguf::load_gguf_model(b_path).expect("b parses");
    let mut index = std::collections::HashMap::new();
    for (pi, part) in ma.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            let off = part.ctx.offset + ti.offset as usize;
            index.insert(ti.name.clone(), (pi, off, ti.nbytes()));
        }
    }
    let mut n = 0;
    for (pi, part) in mb.parts.iter().enumerate() {
        for ti in &part.ctx.info {
            let off = part.ctx.offset + ti.offset as usize;
            let (pa, oa, na) = index
                .get(&ti.name)
                .unwrap_or_else(|| panic!("tensor '{}' only in the second file", ti.name));
            assert_eq!(*na, ti.nbytes(), "tensor {} size", ti.name);
            assert_eq!(
                &ma.parts[*pa].data[*oa..*oa + *na],
                &mb.parts[pi].data[off..off + ti.nbytes()],
                "tensor {} payload differs",
                ti.name
            );
            n += 1;
        }
    }
    assert_eq!(n, index.len(), "tensor set differs");
    assert!(n > 0);
    n
}
/// The Qwen3 twin of [`logits_greedy_on`] (that one is pinned to `Qwen2Model`).
/// Same forward code on both arms, so the only difference is the backend the
/// scheduler assigns.
fn logits_greedy_on_qwen3(
    q: &crate::models::qwen3::Qwen3Model,
    ids: &[u32],
    steps: usize,
    n_ctx: usize,
) -> (Vec<f32>, Vec<u32>) {
    let nt = ids.len();
    let mut cache = crate::graph::cache::GraphCache::new();
    let mut positions: Vec<usize> = (0..nt).collect();
    let mut logits = crate::models::qwen3::graph::Qwen3Graph::forward_cached(
        q, ids, &positions, 1, n_ctx, &mut cache,
    );
    let mut toks = Vec::with_capacity(steps);
    for step in 0..steps {
        let next = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0 as u32;
        toks.push(next);
        positions = vec![nt + step];
        logits = crate::models::qwen3::graph::Qwen3Graph::forward_cached(
            q,
            &[next],
            &positions,
            1,
            n_ctx,
            &mut cache,
        );
    }
    let _ = positions;
    (logits, toks)
}
