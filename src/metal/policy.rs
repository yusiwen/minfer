// Metal backend policy predicates (pre-split `src/metal.rs`).
//
// Moved verbatim by the #265 layout split. These are the pure `MINFER_*`
// switches the `graph/metal_backend.rs` op gate and the model loaders read; none
// of them holds device state.

// The process-wide `KV_F16` OnceLock, `kv_cache_is_f16`, `set_kv_cache_type` and
// the `#[cfg(test)]` override lived here until issue #44 part (b). The GPU KV
// element type is now **per engine**: `MetalBackend::kv_format` is stamped from
// `GraphAllocator::set_kv_format` (the loaded engine's resolved
// `MINFER_CACHE_TYPE`/auto policy) and every kernel dispatch reads that instance
// — exactly the #99/#153 lesson. The old global let the first load win, so two
// engines could not hold different layouts and a C5 session could be described
// under the wrong width.

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

/// Use the windowed sibling of `kernel_flash_attn_blk_*`
/// (`kernel_flash_attn_window_blk_*`, `fa_window.metal`, issue #359) for an
/// explicit `attn_span` prefill. Fixed-shape like the causal prefill, so it
/// requires `hd == 64 || hd == 128`; every other shape keeps the #44
/// correctness kernel `kernel_gqa_attn_window_*`. ON by default;
/// `MINFER_NO_WINDOW_FLASH=1` reverts to the correctness kernel for A/B.
pub fn prefill_window_flash_enabled(hd: usize) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_WINDOW_FLASH").map_or(true, |v| v != "1"))
        && (hd == 64 || hd == 128)
}

/// #310: which kernel family a packed `q8_0` KV region's `Op::Attn` takes. The
/// fast families are f16/f32-only, so a packed region either reads packed cells
/// natively (mechanism A, decode only) or dequantizes the needed window into a
/// transient **f32** stage and runs the existing f32 fast kernel (mechanism B);
/// everything a fast family does not cover keeps the classic packed kernel.
///
/// This is the whole routing matrix, factored out so it is testable with no
/// Metal device. `hd == 64 || hd == 128` is folded in by the `*_enabled`
/// predicates, so `Classic` is also the answer for a small/odd `hd` and for any
/// `MINFER_NO_*` opt-out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackedRoute {
    /// `nt == 1` causal decode: the packed native flash
    /// (`kernel_flash_attn_ext_q8_0` / `_hd128_q8_0`).
    NativeDecode,
    /// `nt == 1` causal decode with the stage forced: dequantize, then the f32
    /// flash. The A/B twin of [`PackedRoute::NativeDecode`].
    StageThenFlash,
    /// `nt > 1` causal prefill: dequantize, then `kernel_flash_attn_blk_*`.
    StageThenPrefill,
    /// `nt > 1` explicit window (one-range `attn_span` or set-valued `kv_map`):
    /// dequantize, then `kernel_flash_attn_window_*`. The host picks the map or
    /// one-range kernel from the window input's size; the route is shared.
    StageThenWindow,
    /// The classic packed kernel, for every shape no fast family covers.
    Classic,
}

/// #310: the packed `Op::Attn` routing matrix (see [`PackedRoute`]). Pure: it
/// reads only `hd`/`nt`/`explicit_span`, the caller-supplied `stage_forced`
/// switch and the device-independent `*_enabled` policy predicates. It is the
/// single authority the Metal dispatch and its off-device gate share.
pub fn packed_attn_route(
    hd: usize,
    nt: usize,
    explicit_span: bool,
    stage_forced: bool,
) -> PackedRoute {
    if explicit_span {
        if nt > 1 && prefill_window_flash_enabled(hd) {
            PackedRoute::StageThenWindow
        } else {
            PackedRoute::Classic
        }
    } else if nt == 1 && flash_attn_enabled(hd) {
        if stage_forced {
            PackedRoute::StageThenFlash
        } else {
            PackedRoute::NativeDecode
        }
    } else if nt > 1 && prefill_flash_enabled(hd) {
        PackedRoute::StageThenPrefill
    } else {
        PackedRoute::Classic
    }
}

#[cfg(test)]
mod route_tests;
