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

use crate::tensor::TensorType;

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

/// Which matvec arm a quant family has for the multi-row band below the GEMM —
/// it decides the row floor of the boundary, because the two arms have different
/// per-extra-row costs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BandArm {
    /// `_multi` only: the token loop sits **inside** one threadgroup, so each
    /// extra row re-streams the weight block. All arms except Q4_K/Q6_K.
    MultiOnly,
    /// Also the shipped single-token kernel, which takes its token from
    /// `tgpig.y`: dispatching it with `grid.y = nt` gives one threadgroup per
    /// (output-row tile, **token**) and the waves share the weight stream
    /// through L2. Q4_K and Q6_K only ([#659]).
    SingleGrid,
}

/// Smallest `nt` at which the multi-row GEMM beats the per-row matvec family
/// **for a family with no `grid.y = nt` band** ([`BandArm::MultiOnly`]) — and an
/// output width that can fill the N tile (see [`GEMM_MIN_OD`]).
///
/// The GEMM's M tile is a fixed 32 rows (`MpsCommandBuffer::gemm_dispatch`), so a
/// call with fewer rows spends most of the tile on padding *and* pays the whole
/// tile's compute; the matvec kernels instead read the weights once and pay the
/// activation side per token. Measured on Metal (Qwen2.5-7B q4_K_M, M4 Pro, KV
/// depth 512, `minfer specverify`, 20 reps): `nt = 1` 20.8 ms, `nt = 3` 58.3 ms
/// against 100.8 ms for the GEMM at the same depth, and `nt = 5` 114.4 ms against
/// 103.1 ms. The crossover therefore sits between 3 and 4 rows.
/// [`GEMM_MIN_OD`] is the other half of the same measurement: at `nt = 4..8` a
/// narrow output row (the 0.5B's `attn_qkv`, od 1152) still loses to the matvec
/// family by 1.56x, because the GEMM's 64-wide N tile is then the half that
/// under-fills.
///
/// Consequence for the speculative decoder: `d = 2` (verify `nt = 3`) lands in the
/// matvec band and is bitwise against decode on every shape; `d >= 3` takes the
/// GEMM for wide targets (tolerance-class) and stays matvec for narrow ones. That
/// is measured, not assumed — the record lives in `docs/METAL_OPTIMIZATIONS.md`.
///
/// [#431]: https://github.com/yusiwen/minfer/issues/431
pub const GEMM_MIN_NT: usize = 4;

/// Smallest `nt` at which the GEMM beats the **single-token/`grid.y = nt`** arm
/// ([`BandArm::SingleGrid`], Q4_K/Q6_K).
///
/// That arm's per-extra-row cost is about half `_multi`'s, so the GEMM's fixed
/// 32-row tile takes two more rows to pay off. Measured 2026-10-23
/// (`single_grid_y_bandwidth_profile`, Q4_K, `n = 50` dispatches per timed command
/// buffer, min of 3 interleaved rounds) — the number is
/// `single grid / shipped GEMM`, below 1.00 meaning the matvec wins:

/// | od×id (target) | nt=4 | nt=5 | nt=6 | nt=7 | nt=8 |
/// |---|---|---|---|---|---|
/// | 3584×3584 (7B attn) | 0.46 | 0.56 | 0.66 | 0.78 | 0.90 |
/// | 18944×3584 (7B ffn) | 0.65 | 0.81 | 0.99 | 1.16 | 1.34 |
/// | 5120×13824 (14B ffn) | 0.62 | 0.77 | 0.93 | 1.09 | 1.26 |
/// | 152064×3584 (`output.weight`, Q6_K) | 0.94 | 1.18 | **1.41** | 1.65 | 1.89 |
/// | 152064×3584 (`output.weight`, Q4_K) | 0.73 | 0.93 | **1.13** | 1.33 | 1.54 |
///
/// `nt = 4` and `5` are unambiguous wins on every working-set shape, so the floor
/// is 6. `nt = 6` is neutral on two of the three working-set shapes (0.93-0.99, one
/// 1.5x win) — and the row that settled it is `output.weight`, the widest N in
/// every model, added to the profile on 2026-10-23: a `q4_K_M` file stores it as
/// Q6_K padded to **152064** (K-quants block by 256; the vocabulary's 151936 is not
/// a multiple, so 32-aligned types keep 151936 and K-quants pad). There the GEMM
/// wins outright from `nt = 6` up (1.41x at 6, 1.65 at 7), which kills the
/// floor-7 option and leaves 6 exact.
///
/// It also prices the *lower* option honestly: on that row, at `nt = 4, 5`, the
/// matvec is 6-18 % slower for Q6_K (the type the row actually is) while it is
/// still 7-27 % faster for Q4_K — and the row is 1 of ~170 matmuls, so the whole
/// question is worth ~0.3-0.7 % of the verify cost, not a third policy axis and
/// not the bitwise logits the matvec arm gives `d = 3, 4`.
pub const GEMM_MIN_NT_SINGLE_GRID: usize = 6;

/// Smallest output width (`od`, the matmul's N dimension) at which the GEMM's
/// 64-wide N tile is worth its M-tile padding.
///
/// Measured on the 0.5B q4_0 (`od` 896-1152): `nt = 4` costs 341 t/s through the
/// GEMM against 533 t/s through the matvec family — the GEMM loses 1.56x when both
/// tiles under-fill. Wide targets (a 7B's 3584-152064) are the GEMM's case.
pub const GEMM_MIN_OD: usize = 2048;

/// The row floor for a family's band arm. Both floors share [`GEMM_MIN_OD`].
pub fn gemm_min_nt(arm: BandArm) -> usize {
    match arm {
        BandArm::MultiOnly => GEMM_MIN_NT,
        BandArm::SingleGrid => GEMM_MIN_NT_SINGLE_GRID,
    }
}

/// The dispatch policy's **pure** half: does a call with `nt` rows over `od`
/// output columns want the GEMM, given which matvec arm the family has?
///
/// Split from [`gemm_enabled`] so the boundary is testable with **no device and no
/// environment mutation** (`src/metal/policy/gate_tests.rs`) — the Metal module
/// only compiles on macOS, but this half needs no GPU and no `MINFER_*` switch,
/// which is what lets one authority serve all seven quant arms.
pub fn gemm_preferred(nt: usize, od: usize, arm: BandArm) -> bool {
    nt >= gemm_min_nt(arm) && od >= GEMM_MIN_OD
}

/// llama's **few-row MMA** tiling for the 2..16 activation-row window (#673): an
/// 8x8 simdgroup-matrix kernel whose weight rows and activation rows are both tiled,
/// so the weights are read once per 8-row tile instead of once per token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FewRowMma {
    /// The pipeline suffix the kernel was instantiated under: `nt{N}_rt{R}`.
    pub nt: usize,
    pub rt: usize,
    pub nsg: usize,
    /// Threadgroup memory the reduction needs, in bytes.
    pub smem: usize,
}

/// Whether the few-row MMA arm is **enabled**: `MINFER_MV_MMA=1`. Off by default until
/// the arm is measured against the matvec on a Mac (#673); the tiling itself is pure so
/// it can be tested and reasoned about without the latch.
pub fn few_row_mma_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_MV_MMA").map_or(false, |v| v == "1"))
}

/// The tiling llama picks for this shape, or `None` when the shape is outside the arm.
///
/// Constants and the halving rule are `ggml-metal-device.cpp:805-865`: 8-row tiles,
/// at most 4 weight tiles, 16 KB of reduction buffer, 128 threadgroups wanted, and
/// `nsg` 32/16/8 by output-row count, halved until the K split has that many 64-weight
/// chunks. `k` must be a multiple of 64 (the kernel's chunk size).
pub fn few_row_mma(ttype: TensorType, nt: usize, od: usize, k: usize) -> Option<FewRowMma> {
    const NT_MAX: usize = 4;
    const SMEM_MAX: usize = 16384;
    const MIN_TG: usize = 128;
    const TILE: usize = 8;
    let rows_min = match ttype {
        TensorType::Q4_K => 3,
        TensorType::Q6_K => 2,
        _ => return None,
    };
    if nt < rows_min || nt > 16 || od < TILE || k < 64 || k % 64 != 0 {
        return None;
    }
    let rt = if nt > 8 { 2 } else { 1 };
    let n_steps = k / 64;
    // Only 8 and 16 are instantiated (see the kernel's MINFER_MMA_ONE list): 32 would
    // need a threadgroup of 1024 threads and no shape this arm serves picks it, so a
    // 32 is clamped — a mismatch between this number and the kernel's baked NSG drops
    // K chunks silently, which is why the value is checked by the tiling gate.
    let mut nsg = if od <= 6144 { 16 } else { 8 };
    while nsg > 1 && nsg > n_steps {
        nsg /= 2;
    }
    if nsg < 8 {
        nsg = 8;
    }
    // Probe (#673): NSG=1 gives every threadgroup its own output rows, so the store
    // path needs no reduction and no threadgroup memory. `nt` drops to 1 to keep
    // enough threadgroups in the grid (od/8 against od/32 at NT=4).
    if let Ok(v) = std::env::var("MINFER_MV_MMA_NSG") {
        if v.parse::<usize>().ok() == Some(1) {
            // `nt=4` measured best: 17.88 tokens/s against 17.16 at nt=2 and 15.36 at
            // nt=1 (14B, nt=3, same afternoon). The eight MMAs inside a chunk all
            // accumulate into one `mc[rt][t]`, so the tile count *is* the number of
            // independent dependency chains; the grid it costs (od/32 threadgroups at
            // 32 threads) is still enough to fill the device.
            return Some(FewRowMma {
                nt: 4,
                rt: 1,
                nsg: 1,
                smem: 1024,
            });
        }
    }
    let limit = (SMEM_MAX / (nsg * rt * 256)).min(od / (MIN_TG * TILE));
    // llama's `halve_to_limit(NT_MAX, min(nt_smem, nt_rows))`: start at the cap and
    // halve while the tile budget cannot hold it.
    let mut nt_tile = NT_MAX;
    while nt_tile > 1 && nt_tile > limit {
        nt_tile /= 2;
    }
    Some(FewRowMma {
        nt: nt_tile,
        rt,
        nsg,
        smem: nsg * nt_tile * rt * 256,
    })
}

/// Debug knob for the geometry sweep (#673): `MINFER_MV_MMA_NT=1|2|4` re-clamps the
/// weight-tile count, which changes how many threadgroups the dispatch has. The tiling
/// rule above is llama's; this exists to ask whether *our* GPU wants a narrower tile
/// than the rule picks, without a rebuild per answer.
pub fn few_row_mma_nt_override() -> Option<usize> {
    std::env::var("MINFER_MV_MMA_NT")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| matches!(n, 1 | 2 | 4))
}

/// Apply [`few_row_mma_nt_override`] to a tiling, keeping the reduction size consistent.
pub fn with_nt_override(mut t: FewRowMma) -> FewRowMma {
    if let Some(nt) = few_row_mma_nt_override() {
        if nt != t.nt {
            t.nt = nt;
            t.smem = t.nsg * t.nt * t.rt * 256;
        }
    }
    t
}

/// Whether the GEMM path is available at all: `MINFER_GEMM=0` forces the per-row
/// matvec family at every `nt` (the A/B switch). ON by default.
pub fn gemm_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_GEMM").map_or(true, |v| v != "0"))
}

#[cfg(test)]
mod gate_tests;
#[cfg(test)]
mod route_tests;
