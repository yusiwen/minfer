//! `#[cfg(test)] mod tests` for `src/models/qwen2/graph.rs` — extracted so a non-test
//! build does not parse it. See the parent module for the docs.
use super::*;

mod batching;
mod cuda_kv;
mod kv_reuse;
mod offload_copy;
mod real_model;
/// Path to the locally cached Qwen2.5-0.5B Q4_0 (downloaded via
/// `minfer download hf Qwen/Qwen2.5-0.5B-Instruct-GGUF Q4_0`).
fn cached_model_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(
        ".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf",
    );
    if p.exists() {
        Some(p)
    } else {
        None
    }
}
/// Max |Δ| between two equal-length logit vectors.
fn max_delta(x: &[f32], y: &[f32]) -> f32 {
    assert_eq!(x.len(), y.len(), "compared vectors must have equal length");
    x.iter()
        .zip(y)
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max)
}
/// The named tolerance class for comparisons whose two sides were computed
/// at **different batch shapes** (a different `nt` anywhere in their
/// history) — the project's rule is "bitwise-identity *or* a named tolerance
/// class", and this is the second.
///
/// On CPU these comparisons are bitwise and stay bitwise: the kernels are
/// per-token, so shape never enters the arithmetic. CUDA's prefill kernels
/// tile by `nt` and quantize activations to int8 (MMQ), so the *same* tokens
/// processed at a different `nt` land on slightly different K/V values. The
/// drift is bounded and small — measured on GB10 (sm_121) as ≤ 0.37 absolute
/// on these fixtures' logits (each test prints its value on device) — but it
/// is far above 0, so a bitwise assertion would be a false claim there.
///
/// This is a gross-error detector for device runs, not a proof of
/// correctness: the same-shape gates (`batch_order_does_not_change_a_sequences_logits`,
/// `cuda_two_sequences_do_not_cross_attend`) and the CPU bitwise assertions
/// are what pin the mechanism.
fn cross_shape_tolerance() -> f32 {
    #[cfg(feature = "cuda")]
    if crate::cuda::CudaState::get().is_some() {
        return 1.0;
    }
    #[cfg(target_os = "macos")]
    if crate::metal::MpsState::get().is_some() {
        // Metal is a device too, so its cross-shape comparison uses a named
        // class for the same reason CUDA does: a batched forward runs the
        // windowed attention kernel (`kernel_gqa_attn_window_*`, E1/G5a) while a
        // single-sequence forward runs the causal flash/prefill kernel, and the
        // GEMMs tile by `nt`, so two shapes cannot agree bitwise. Named before
        // measuring: **0.1** (the observed max|Δ| across these qwen2 gates on
        // the 0.5B is <= 0.0153). The same-shape gate
        // `batch_order_does_not_change_a_sequences_logits` stays bitwise and is
        // what pins the window assignment.
        return 0.1;
    }
    0.0
}
/// Assert two forwards that differ in shape agree: bitwise on CPU, within
/// [`cross_shape_tolerance`] on a device (where `what` is printed with the
/// measured |Δ| so drift regressions are visible in the log).
fn assert_across_shapes(what: &str, a: &[f32], b: &[f32]) {
    let d = max_delta(a, b);
    let tol = cross_shape_tolerance();
    if tol > 0.0 {
        eprintln!("[cuda] {what}: max |Δ| = {d} (named class: <= {tol})");
    }
    assert!(
        d <= tol,
        "{what}: max |Δ| = {d} exceeds {} tolerance ({tol})",
        if tol == 0.0 {
            "the bitwise"
        } else {
            "the CUDA cross-shape"
        }
    );
}
fn argmax(x: &[f32]) -> u32 {
    let mut best = 0usize;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best as u32
}
/// Compare the last logits row (n_out=1 → the whole returned vector).
fn compare(tag: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "[{tag}] logits length mismatch");
    let mut maxd = 0.0f32;
    for i in 0..a.len() {
        maxd = maxd.max((a[i] - b[i]).abs());
    }
    eprintln!("[{tag}] logits max abs diff: {maxd:.3e}");
    // Since Phase 6 `model.forward` IS the graph path: on CUDA-capable
    // builds the engine side runs the CUDA graph while this test's manual
    // graph is CPU-only, so the comparison is cross-backend and the
    // bitwise criterion does not apply (7e① diagnosis: the 0.449
    // "residual" is accumulated f32 reduction-order noise, not a bug —
    // mirror the Metal test's functional criterion). CPU-only builds
    // compare two CPU graphs and keep the strict bound.
    #[cfg(feature = "cuda")]
    if crate::cuda::CudaState::get().is_some() {
        let ga = argmax(a);
        let gb = argmax(b);
        eprintln!("[{tag}] greedy token: CPU-graph={ga} engine-graph={gb}");
        assert_eq!(ga, gb, "[{tag}] greedy token differs across backends");
        return;
    }
    assert!(
        maxd < 1e-3,
        "[{tag}] graph vs forward logits diverge (max diff {maxd:.3e})"
    );
}
