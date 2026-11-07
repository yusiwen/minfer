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
//! Scope: the platform-neutral engine core (`graph`, `models`, `quants`,
//! `vec_ops`, `conversation`). The device backends' own policy knobs
//! (`src/metal/`, `src/cuda/`) stay where they are for now: they are
//! platform-gated, each carries its own cached `OnceLock`, and a Metal-only
//! change cannot be compiled on the Linux host this slice was written on.

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
