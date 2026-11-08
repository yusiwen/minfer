//! Process-wide engine configuration (#763 S4).
//!
//! The engine used to read `MINFER_*` variables from arbitrary depth — a SIMD
//! dispatch, a graph builder, a model loader — which made the environment the
//! only way to configure inference. A library consumer cannot use that, and two
//! engines cannot differ.
//!
//! The rule now: **[`EngineConfig::from_env`] is the application's job** (the CLI
//! calls it once at startup and [`install`]s it); the engine reads [`get`]. A
//! consumer that installs nothing gets [`EngineConfig::default`], never the
//! host's environment.
//!
//! Scope: the whole engine. #763 S4 converted the platform-neutral core
//! (`graph`, `models`, `quants`, `vec_ops`, `conversation`); #769 converted the
//! device backends' own policy knobs (`src/cuda/`, `src/metal/`,
//! `src/graph/metal_backend.rs`, `src/models/*/graph.rs`), which used to read
//! `MINFER_*` directly at 60 sites. Nothing under `src/` reads the environment
//! for an engine knob any more — only [`EngineConfig::from_env`] and the
//! `#[cfg(test)]` harnesses do.

use crate::graph::registry::BackendFilter;
use std::sync::OnceLock;

/// Everything the engine core needs to know before it runs, in one value.
///
/// Field semantics follow the environment they replace: the `fuse_*`, `avx2`,
/// `neon` and `context_shift` flags mean "the optimisation is on", so a
/// `MINFER_NO_*=1` variable sets them to `false`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EngineConfig {
    // ── engine scope ────────────────────────────────────────────────────
    /// The backends this engine may use (`--backend` / `MINFER_BACKENDS`).
    /// The application resolves the names and sets this before installing;
    /// `from_env` leaves it as [`BackendFilter::all`].
    pub backends: BackendFilter,
    // ── graph shape / fusion ────────────────────────────────────────────
    /// `MINFER_NO_FUSE_QKV` (inverted).
    pub fuse_qkv: bool,
    /// `MINFER_NO_FUSE_FFN` (inverted).
    pub fuse_ffn: bool,
    /// `MINFER_FFN_COMPOSITION`; `Some("0"|"1")` forces the D2 composition.
    pub ffn_composition: Option<String>,
    /// `MINFER_NO_FUSED_Q8_KV` (inverted).
    pub fused_q8_kv: bool,
    /// `MINFER_NO_SPLIT_ATTN` (inverted).
    pub split_attn: bool,
    /// `MINFER_REBUILD_TRACE`.
    pub rebuild_trace: bool,
    // ── offload / cache ─────────────────────────────────────────────────
    /// `MINFER_CACHE_TYPE`.
    pub kv_cache_type: Option<String>,
    /// `MINFER_GPU_LAYERS` (`default` / `auto` / a block count).
    pub gpu_layers: Option<String>,
    /// `MINFER_GPU_MEM` in MiB.
    pub gpu_mem: Option<String>,
    /// `MINFER_NO_KV_DEFRAG` (inverted).
    pub kv_defrag: bool,
    // ── device execution ────────────────────────────────────────────────
    /// `MINFER_NO_CUDA_GRAPH` (inverted).
    pub cuda_graph: bool,
    /// `MINFER_NO_PREFILL_CAPTURE` (inverted).
    pub prefill_capture: bool,
    /// `MINFER_TEST_CAPTURE_WARMUP` (a test seam; `None` = the real 3).
    pub capture_warmup: Option<u32>,
    /// `MINFER_MMQ_Q4K_DSC` (default on; `0` disables).
    pub mmq_q4k_dsc: bool,
    /// `MINFER_PACKED_KV_STAGE=1`.
    pub packed_kv_stage: bool,
    /// `MINFER_METAL_SYNC_OPS=1`.
    pub metal_sync_ops: bool,
    /// `MINFER_ATTN_CHUNKS`.
    pub attn_chunks: Option<usize>,
    /// `MINFER_FFNDEBUG` (presence).
    pub ffndebug: bool,
    // ── device policy: CUDA kernel levers (all A/B escapes; #769) ───────
    /// `MINFER_MMQ` — the int8 MMQ prefill GEMM; on unless `0` (r60 promotion).
    pub mmq: bool,
    /// `MINFER_MMQ_RAW`; on unless `0`.
    pub mmq_raw: bool,
    /// `MINFER_MMQ_RAW_NB`; on unless `0`.
    pub mmq_raw_nb: bool,
    /// `MINFER_MMQ_A_TRANSPOSE`; on unless `0`.
    pub mmq_a_transpose: bool,
    /// `MINFER_MMQ_Q6K_NB`; on unless `0`.
    pub mmq_q6k_nb: bool,
    /// `MINFER_Q6K_DPL`; on unless `0`.
    pub q6k_dpl: bool,
    /// `MINFER_MMQ_A_FUSE` — 0 off, 1 fused, 2 fused + skip the f32 write
    /// (the default; any other value is 0).
    pub mmq_a_fuse: u8,
    /// `MINFER_MMQ_KSPLIT_TARGET` (`None` = the kernel's own SM-based rule).
    pub mmq_ksplit_target: Option<usize>,
    /// `MINFER_MMQ_RAW_KD` (the k-split depth; 8 when unset).
    pub mmq_raw_kd: i32,
    /// `MINFER_MMQ_RAW_WIDE=1`.
    pub mmq_raw_wide: bool,
    /// `MINFER_MMQ_RAW_NB_DEBUG=1`.
    pub mmq_raw_nb_debug: bool,
    /// `MINFER_MMQ_Q6K_EXP`; on unless `0`.
    pub mmq_q6k_exp: bool,
    /// `MINFER_NO_KQ_MMVQ=1` — force the f32 decode matmuls for K-quants.
    pub no_kq_mmvq: bool,
    /// `MINFER_NO_Q40_MMVQ=1`.
    pub no_q40_mmvq: bool,
    /// `MINFER_NO_Q80_MMVQ=1`.
    pub no_q80_mmvq: bool,
    /// `MINFER_NO_Q80_P32=1`.
    pub no_q80_p32: bool,
    /// `MINFER_NO_DECODE_A_FUSE=1`.
    pub no_decode_a_fuse: bool,
    /// `MINFER_FUSED_B=1`.
    pub fused_b: bool,
    /// `MINFER_NO_W16CACHE=1` — drop the persistent f16 dequant cache.
    pub no_w16cache: bool,
    /// `MINFER_NO_PREFILL_GEMM=1` — the legacy per-type prefill kernels.
    pub no_prefill_gemm: bool,
    /// `MINFER_NO_FA_PREFILL=1` — the legacy per-token prefill attention.
    pub no_fa_prefill: bool,
    /// `MINFER_SMALL_M_GEMM=1`.
    pub small_m_gemm: bool,
    /// `MINFER_Q6K_PF`; on unless `0` (the Q6_K prefetch decode kernel).
    pub q6k_prefetch: bool,
    /// `MINFER_MMVQ_V1=1` — force the v1 MMVQ kernels.
    pub mmvq_v1: bool,
    /// `MINFER_NO_PINNED_READBACK=1`.
    pub no_pinned_readback: bool,
    /// `MINFER_GEMM_A32=1` — the smem-mirror f32-A GEMM variant.
    pub gemm_a32: bool,
    /// `MINFER_A32_DEBUG` (presence).
    pub a32_debug: bool,
    /// `MINFER_DISABLE_CUDA` (presence: no CUDA device is initialised).
    pub disable_cuda: bool,
    /// `MINFER_DEVICE_TIER` — force one device-tier row by index.
    pub device_tier: Option<String>,
    /// `MINFER_OP_TIMING` (presence) — per-op timing in `/metrics`.
    pub op_timing: Option<String>,
    /// `MINFER_DUMP_DIR` — per-layer debug dumps (`debug_dump` feature).
    pub dump_dir: Option<String>,
    // ── device policy: Metal kernel levers (#769) ───────────────────────
    /// `MINFER_NO_RMS_256` (inverted: the 256-thread RMSNorm is on).
    pub rms_256: bool,
    /// `MINFER_NO_MATMUL_ATTN` (inverted).
    pub matmul_attn: bool,
    /// `MINFER_NO_FLASH` (inverted).
    pub flash_attn: bool,
    /// `MINFER_NO_PREFILL_FLASH` (inverted).
    pub prefill_flash: bool,
    /// `MINFER_NO_WINDOW_FLASH` (inverted).
    pub window_flash: bool,
    /// `MINFER_MV_MMA=1` — the few-row MMA decode arm.
    pub mv_mma: bool,
    /// `MINFER_MV_MMA_NSG` — the probe's threadgroup variant (`1` only).
    pub mv_mma_nsg: Option<usize>,
    /// `MINFER_MV_MMA_NT` — the weight-tile override (1, 2 or 4 only).
    pub mv_mma_nt: Option<usize>,
    /// `MINFER_GEMM`; on unless `0` (the GEMM path at every `nt`).
    pub gemm: bool,
    /// `MINFER_DISABLE_MPS` (presence: no Metal device is initialised).
    pub disable_mps: bool,
    /// `MINFER_METAL_CAPTURE` (presence: start a GPU capture).
    pub metal_capture: bool,
    /// `MINFER_METAL_LABELS=1` — name the encoder in the GPU trace (#673).
    pub metal_labels: bool,
    /// `MINFER_METALLIB_FILE` — a `.metallib` overriding the embedded one.
    pub metallib_file: Option<String>,
    /// `MINFER_WEIGHT_COPY=1` — force the host copy instead of the zero-copy view.
    pub weight_copy: bool,
    // ── device policy: CUDA runtime/capture (moved from `src/cuda.rs`) ───
    /// `MINFER_CUDA_CAPTURE_MODE` (0 relaxed, 1 global, 2 thread-local; the
    /// default and any invalid value is thread-local, ADR-0031).
    pub cuda_capture_mode: i32,
    /// `MINFER_NO_DP4A_Q8_KV` (inverted).
    pub dp4a_q8_kv: bool,
    /// `MINFER_NO_Q8_KV_WIDE` (inverted).
    pub q8_kv_wide: bool,
    /// `MINFER_NO_GEMM_PREWARM` (inverted).
    pub gemm_prewarm: bool,
    /// `MINFER_S4_AB_MAP_REPS` — the map-window A/B arm's repeat count (1 when unset).
    pub s4_ab_map_reps: usize,
    // ── diagnostics ─────────────────────────────────────────────────────
    /// `MINFER_GRAPH_TRACE` (presence).
    pub graph_trace: bool,
    /// `MINFER_SYNC_COPIES=1`.
    pub sync_copies: bool,
    /// `MINFER_OP_PROFILE=1`.
    pub op_profile: bool,
    /// `MINFER_GRAPH_DUMP` directory.
    pub graph_dump: Option<String>,
    // ── platform capability overrides ───────────────────────────────────
    /// `MINFER_NO_AVX2` (inverted).
    pub avx2: bool,
    /// `MINFER_NO_AVX512` (inverted).
    pub avx512: bool,
    /// `MINFER_NO_NEON` (inverted).
    pub neon: bool,
    /// `MINFER_NO_F16_ROWB` (inverted).
    pub f16_rowb: bool,
    // ── conversation ────────────────────────────────────────────────────
    /// `MINFER_NO_CONTEXT_SHIFT` (inverted).
    pub context_shift: bool,
}

impl Default for EngineConfig {
    /// The built-in defaults: every optimisation on, every diagnostic off.
    fn default() -> Self {
        Self {
            backends: BackendFilter::all(),
            fuse_qkv: true,
            fuse_ffn: true,
            ffn_composition: None,
            fused_q8_kv: true,
            split_attn: true,
            rebuild_trace: false,
            kv_cache_type: None,
            gpu_layers: None,
            gpu_mem: None,
            kv_defrag: true,
            cuda_graph: true,
            prefill_capture: true,
            capture_warmup: None,
            mmq_q4k_dsc: true,
            packed_kv_stage: false,
            metal_sync_ops: false,
            attn_chunks: None,
            ffndebug: false,
            mmq: true,
            mmq_raw: true,
            mmq_raw_nb: true,
            mmq_a_transpose: true,
            mmq_q6k_nb: true,
            q6k_dpl: true,
            mmq_a_fuse: 2,
            mmq_ksplit_target: None,
            mmq_raw_kd: 8,
            mmq_raw_wide: false,
            mmq_raw_nb_debug: false,
            mmq_q6k_exp: true,
            no_kq_mmvq: false,
            no_q40_mmvq: false,
            no_q80_mmvq: false,
            no_q80_p32: false,
            no_decode_a_fuse: false,
            fused_b: false,
            no_w16cache: false,
            no_prefill_gemm: false,
            no_fa_prefill: false,
            small_m_gemm: false,
            q6k_prefetch: true,
            mmvq_v1: false,
            no_pinned_readback: false,
            gemm_a32: false,
            a32_debug: false,
            disable_cuda: false,
            device_tier: None,
            op_timing: None,
            dump_dir: None,
            rms_256: true,
            matmul_attn: true,
            flash_attn: true,
            prefill_flash: true,
            window_flash: true,
            mv_mma: false,
            mv_mma_nsg: None,
            mv_mma_nt: None,
            gemm: true,
            disable_mps: false,
            metal_capture: false,
            metallib_file: None,
            metal_labels: false,
            weight_copy: false,
            cuda_capture_mode: 2,
            dp4a_q8_kv: true,
            q8_kv_wide: true,
            gemm_prewarm: true,
            s4_ab_map_reps: 1,
            graph_trace: false,
            sync_copies: false,
            op_profile: false,
            graph_dump: None,
            avx2: true,
            avx512: true,
            neon: true,
            f16_rowb: true,
            context_shift: true,
        }
    }
}

impl EngineConfig {
    /// Read the `MINFER_*` interface. **The application calls this** — the engine
    /// never reads the environment itself (ADR-0079).
    pub fn from_env() -> Self {
        // One literal read per knob: `scripts/check_env_vars.py` enumerates these
        // to prove every knob has a documented home, so a dynamic `var(name)`
        // helper would hide them from the gate.
        Self {
            backends: BackendFilter::all(),
            fuse_qkv: !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1"),
            fuse_ffn: !std::env::var("MINFER_NO_FUSE_FFN").map_or(false, |v| v == "1"),
            ffn_composition: std::env::var("MINFER_FFN_COMPOSITION").ok(),
            fused_q8_kv: !std::env::var("MINFER_NO_FUSED_Q8_KV").map_or(false, |v| v == "1"),
            split_attn: !std::env::var("MINFER_NO_SPLIT_ATTN").map_or(false, |v| v == "1"),
            rebuild_trace: std::env::var("MINFER_REBUILD_TRACE").map_or(false, |v| v == "1"),
            kv_cache_type: std::env::var("MINFER_CACHE_TYPE").ok(),
            gpu_layers: std::env::var("MINFER_GPU_LAYERS").ok(),
            gpu_mem: std::env::var("MINFER_GPU_MEM").ok(),
            kv_defrag: !std::env::var("MINFER_NO_KV_DEFRAG").map_or(false, |v| v == "1"),
            cuda_graph: !std::env::var("MINFER_NO_CUDA_GRAPH").map_or(false, |v| v == "1"),
            prefill_capture: !std::env::var("MINFER_NO_PREFILL_CAPTURE")
                .map_or(false, |v| v == "1"),
            capture_warmup: std::env::var("MINFER_TEST_CAPTURE_WARMUP")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .filter(|n| *n >= 1),
            mmq_q4k_dsc: std::env::var("MINFER_MMQ_Q4K_DSC").map_or(true, |v| v != "0"),
            packed_kv_stage: std::env::var("MINFER_PACKED_KV_STAGE").map_or(false, |v| v == "1"),
            metal_sync_ops: std::env::var("MINFER_METAL_SYNC_OPS").map_or(false, |v| v == "1"),
            attn_chunks: std::env::var("MINFER_ATTN_CHUNKS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&c| c >= 1),
            ffndebug: std::env::var("MINFER_FFNDEBUG").is_ok(),
            // ── device policy: CUDA (#769) ──────────────────────────────
            mmq: std::env::var("MINFER_MMQ").map_or(true, |v| v != "0"),
            mmq_raw: std::env::var("MINFER_MMQ_RAW").map_or(true, |v| v != "0"),
            mmq_raw_nb: std::env::var("MINFER_MMQ_RAW_NB").map_or(true, |v| v != "0"),
            mmq_a_transpose: std::env::var("MINFER_MMQ_A_TRANSPOSE").map_or(true, |v| v != "0"),
            mmq_q6k_nb: std::env::var("MINFER_MMQ_Q6K_NB").map_or(true, |v| v != "0"),
            q6k_dpl: std::env::var("MINFER_Q6K_DPL").map_or(true, |v| v != "0"),
            mmq_a_fuse: match std::env::var("MINFER_MMQ_A_FUSE").as_deref() {
                Err(std::env::VarError::NotPresent) => 2,
                Ok("1") => 1,
                Ok("2") => 2,
                _ => 0,
            },
            mmq_ksplit_target: std::env::var("MINFER_MMQ_KSPLIT_TARGET")
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
            mmq_raw_kd: std::env::var("MINFER_MMQ_RAW_KD")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
                .unwrap_or(8),
            mmq_raw_wide: std::env::var("MINFER_MMQ_RAW_WIDE").as_deref() == Ok("1"),
            mmq_raw_nb_debug: std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1"),
            mmq_q6k_exp: std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() != Ok("0"),
            no_kq_mmvq: std::env::var("MINFER_NO_KQ_MMVQ").map_or(false, |v| v == "1"),
            no_q40_mmvq: std::env::var("MINFER_NO_Q40_MMVQ").map_or(false, |v| v == "1"),
            no_q80_mmvq: std::env::var("MINFER_NO_Q80_MMVQ").map_or(false, |v| v == "1"),
            no_q80_p32: std::env::var("MINFER_NO_Q80_P32").map_or(false, |v| v == "1"),
            no_decode_a_fuse: std::env::var("MINFER_NO_DECODE_A_FUSE").map_or(false, |v| v == "1"),
            fused_b: std::env::var("MINFER_FUSED_B").map_or(false, |v| v == "1"),
            no_w16cache: std::env::var("MINFER_NO_W16CACHE").map_or(false, |v| v == "1"),
            no_prefill_gemm: std::env::var("MINFER_NO_PREFILL_GEMM").map_or(false, |v| v == "1"),
            no_fa_prefill: std::env::var("MINFER_NO_FA_PREFILL").map_or(false, |v| v == "1"),
            small_m_gemm: std::env::var("MINFER_SMALL_M_GEMM").map_or(false, |v| v == "1"),
            q6k_prefetch: std::env::var("MINFER_Q6K_PF").map_or(true, |v| v != "0"),
            mmvq_v1: std::env::var("MINFER_MMVQ_V1").map_or(false, |v| v == "1"),
            no_pinned_readback: std::env::var("MINFER_NO_PINNED_READBACK")
                .map_or(false, |v| v == "1"),
            gemm_a32: std::env::var("MINFER_GEMM_A32").map_or(false, |v| v == "1"),
            a32_debug: std::env::var("MINFER_A32_DEBUG").is_ok(),
            disable_cuda: std::env::var("MINFER_DISABLE_CUDA").is_ok(),
            device_tier: std::env::var("MINFER_DEVICE_TIER").ok(),
            op_timing: std::env::var("MINFER_OP_TIMING").ok(),
            dump_dir: std::env::var("MINFER_DUMP_DIR").ok(),
            // ── device policy: Metal (#769) ─────────────────────────────
            rms_256: std::env::var("MINFER_NO_RMS_256").map_or(true, |v| v != "1"),
            matmul_attn: std::env::var("MINFER_NO_MATMUL_ATTN").map_or(true, |v| v != "1"),
            flash_attn: std::env::var("MINFER_NO_FLASH").map_or(true, |v| v != "1"),
            prefill_flash: std::env::var("MINFER_NO_PREFILL_FLASH").map_or(true, |v| v != "1"),
            window_flash: std::env::var("MINFER_NO_WINDOW_FLASH").map_or(true, |v| v != "1"),
            mv_mma: std::env::var("MINFER_MV_MMA").map_or(false, |v| v == "1"),
            mv_mma_nsg: std::env::var("MINFER_MV_MMA_NSG")
                .ok()
                .and_then(|v| v.parse::<usize>().ok()),
            mv_mma_nt: std::env::var("MINFER_MV_MMA_NT")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| matches!(n, 1 | 2 | 4)),
            gemm: std::env::var("MINFER_GEMM").map_or(true, |v| v != "0"),
            disable_mps: std::env::var("MINFER_DISABLE_MPS").is_ok(),
            metal_capture: std::env::var("MINFER_METAL_CAPTURE").is_ok(),
            metallib_file: std::env::var("MINFER_METALLIB_FILE")
                .ok()
                .filter(|p| !p.is_empty()),
            weight_copy: std::env::var("MINFER_WEIGHT_COPY").map_or(false, |v| v == "1"),
            metal_labels: std::env::var("MINFER_METAL_LABELS").map_or(false, |v| v == "1"),
            // ── device policy: CUDA runtime/capture (from `src/cuda.rs`) ──
            cuda_capture_mode: match std::env::var("MINFER_CUDA_CAPTURE_MODE")
                .ok()
                .and_then(|v| v.parse::<i32>().ok())
            {
                Some(m @ (0 | 1 | 2)) => m,
                Some(other) => {
                    eprintln!(
                        "CUDA: ignoring MINFER_CUDA_CAPTURE_MODE={other} (want 0 relaxed, 1 global, \
                         2 thread-local); using thread-local"
                    );
                    2
                }
                None => 2,
            },
            dp4a_q8_kv: std::env::var("MINFER_NO_DP4A_Q8_KV").as_deref() != Ok("1"),
            q8_kv_wide: std::env::var("MINFER_NO_Q8_KV_WIDE").as_deref() != Ok("1"),
            gemm_prewarm: std::env::var("MINFER_NO_GEMM_PREWARM").as_deref() != Ok("1"),
            s4_ab_map_reps: std::env::var("MINFER_S4_AB_MAP_REPS")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|&n| n >= 1)
                .unwrap_or(1),
            graph_trace: std::env::var("MINFER_GRAPH_TRACE").is_ok(),
            sync_copies: std::env::var("MINFER_SYNC_COPIES").map_or(false, |v| v == "1"),
            op_profile: std::env::var("MINFER_OP_PROFILE").map_or(false, |v| v == "1"),
            graph_dump: std::env::var("MINFER_GRAPH_DUMP").ok(),
            avx2: !std::env::var("MINFER_NO_AVX2").map_or(false, |v| v == "1"),
            avx512: !std::env::var("MINFER_NO_AVX512").map_or(false, |v| v == "1"),
            neon: !std::env::var("MINFER_NO_NEON").map_or(false, |v| v == "1"),
            f16_rowb: !std::env::var("MINFER_NO_F16_ROWB").map_or(false, |v| v == "1"),
            context_shift: !std::env::var("MINFER_NO_CONTEXT_SHIFT").map_or(false, |v| v == "1"),
        }
    }
}

static GLOBAL: OnceLock<EngineConfig> = OnceLock::new();

/// Install the process configuration. Called once, by the application, before
/// any engine exists; a second call is ignored (the first wins).
pub fn install(cfg: EngineConfig) {
    let _ = GLOBAL.set(cfg);
}

/// The configuration in force: the installed one, or the built-in defaults when
/// nothing was installed (a library consumer that does not configure anything).
pub fn get() -> &'static EngineConfig {
    #[cfg(test)]
    if let Some(cfg) = TEST_OVERRIDE.with(|slot| slot.get()) {
        return cfg;
    }
    GLOBAL.get_or_init(EngineConfig::default)
}

#[cfg(test)]
thread_local! {
    /// A test-scoped configuration, so a gate can flip a knob without touching
    /// the process environment (shared by every test in the binary).
    static TEST_OVERRIDE: std::cell::Cell<Option<&'static EngineConfig>> =
        const { std::cell::Cell::new(None) };
}

/// Run `f` with `cfg` in force on this thread. Test-only; the config is leaked
/// (tests are short-lived, and a scoped reference must stay `'static` for
/// [`get`]).
#[cfg(test)]
pub fn with_config<T>(cfg: EngineConfig, f: impl FnOnce() -> T) -> T {
    let leaked: &'static EngineConfig = Box::leak(Box::new(cfg));
    let prev = TEST_OVERRIDE.with(|slot| slot.replace(Some(leaked)));
    let out = f();
    TEST_OVERRIDE.with(|slot| slot.set(prev));
    out
}

#[cfg(test)]
mod tests;
