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

pub fn kv_cache_is_f16() -> bool {
    *KV_F16.get_or_init(|| false)
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
