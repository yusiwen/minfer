// Metal backend policy predicates (pre-split `src/metal.rs`).
//
// Moved verbatim by the #265 layout split. These are the pure `MINFER_*`
// switches the `graph/metal_backend.rs` op gate and the model loaders read; none
// of them holds device state.

/// KV cache element type for the GPU path. `MINFER_CACHE_TYPE=f16` forces a
/// half cache (llama.cpp's default); `MINFER_CACHE_TYPE=f32` forces f32. When
/// unset, `set_kv_cache_type` (called at model load with the model dims)
/// auto-selects: f16 for the 7B class (n_layers×n_kv_embd ≥ 8192 — KV
/// bandwidth-bound decode; measured 7B @2K ctx f16 ≈ −1 ms/token vs f32),
/// f32 for small models (0.5B measured f16 ~3% SLOWER — dispatch-latency-bound,
/// see §0 decided-not #8 / §2.5).
static KV_F16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Test-only override of [`kv_cache_is_f16`] (issue #44's `attn_span` gate): the
/// production answer is a process-wide `OnceLock` — the first `set_kv_cache_type`
/// wins — so one test cannot exercise both KV widths in a single process by
/// calling the production setter twice. `0` = defer to `KV_F16`, `1` = force f32,
/// `2` = force f16. The real machine has a Mac, and the production reader is
/// unchanged (the override is `#[cfg(test)]`-only); the override exists exactly
/// because a per-engine `kv_format` on Metal is part (b) [#306]/G5's write side,
/// not this read-side PR.
#[cfg(test)]
static KV_F16_TEST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

pub fn kv_cache_is_f16() -> bool {
    #[cfg(test)]
    {
        match KV_F16_TEST.load(std::sync::atomic::Ordering::Relaxed) {
            1 => return false,
            2 => return true,
            _ => {}
        }
    }
    *KV_F16.get_or_init(|| false)
}

/// Test-only: force [`kv_cache_is_f16`] to `f16`/`f32` for the current test.
/// Paired with [`clear_kv_f16_for_test`] (or a drop guard) so the override does
/// not leak into another test sharing the process.
#[cfg(test)]
pub fn set_kv_f16_for_test(f16: bool) {
    KV_F16_TEST.store(
        if f16 { 2 } else { 1 },
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Test-only: restore the production `OnceLock` answer.
#[cfg(test)]
pub fn clear_kv_f16_for_test() {
    KV_F16_TEST.store(0, std::sync::atomic::Ordering::Relaxed);
}

/// Called once at model load with the model dims, BEFORE the first forward:
/// sets the GPU KV cache element type (auto-select or MINFER_CACHE_TYPE).
pub fn set_kv_cache_type(n_layers: usize, n_kv_embd: usize) {
    let f16 = std::env::var("MINFER_CACHE_TYPE").map_or(
        n_layers * n_kv_embd >= 8192, // auto: 7B class → f16
        |v| v == "f16",
    );
    let _ = KV_F16.set(f16);
}

/// Use the 256-thread multi-simdgroup rms_norm in the decode path (P1 2026-08-10
/// A/B gate; ON by default after it measured ~2x faster than the 32-thread kernel).
pub fn rms_norm_256_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_RMS_256").map_or(true, |v| v != "1"))
}

/// Use matmul-based prefill attention (P1 2026-08-11): broadcast+quantize the
/// KV to Q8_0 and compute kq/kqv via the fast Q8_0 GEMM, replacing the
/// latency-bound classic kernel for nt>1. ON by default; MINFER_NO_MATMUL_ATTN=1
/// falls back to the classic kernel for A/B.
pub fn matmul_attn_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_MATMUL_ATTN").map_or(true, |v| v != "1"))
}

/// Use the llama flash-attention port (kernel_flash_attn_ext_f32/_f16 and the
/// hd=128 variants) for nt==1 decode. Fixed-shape kernels → requires hd==64
/// (DK=DV=64) or hd==128 (DK=DV=128); anything else falls back to the
/// split-attention path. ON by default; MINFER_NO_FLASH=1 reverts to the split
/// path for A/B.
pub fn flash_attn_enabled(hd: usize) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_FLASH").map_or(true, |v| v != "1"))
        && (hd == 64 || hd == 128)
}

/// Use the llama kernel_flash_attn_ext_blk port (kernel_flash_attn_blk_f32/_f16,
/// legacy simdgroup_matrix) for prefill attention when nt>1. Fixed-shape
/// (DK=DV=64 or DK=DV=128) kernel → requires hd==64 or hd==128; anything else
/// falls back to the 3-pass parallel attention. ON by default;
/// MINFER_NO_PREFILL_FLASH=1 reverts to the 3-pass path for A/B.
pub fn prefill_flash_enabled(hd: usize) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_PREFILL_FLASH").map_or(true, |v| v != "1"))
        && (hd == 64 || hd == 128)
}
