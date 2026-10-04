// MMQ/GEMM gate predicates (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

impl CudaState {
    /// r60: the promoted MMQ gate semantics — unset / any non-"0" value =
    /// ON (the verified 1.080x path), explicit "0" = opt-out to the pre-r60
    /// disabled/f16 behavior (the r54 `MINFER_MMQ_Q6K_EXP` pattern).
    /// Single-sourced: every promoted MINFER_MMQ_* dispatch and
    /// plane-registration read goes through this.
    pub fn mmq_gate_on(name: &str) -> bool {
        std::env::var(name).map_or(true, |v| v != "0")
    }

    /// R1: `MINFER_MMQ` gates the prefill path INTO the int8 MMQ GEMM.
    /// r60 PROMOTION (2026-09-06): default ON — the full r34-r59 MMQ stack
    /// is parity-green and measures 3590.8 tok/s clean on 7B q4_K_m @3314
    /// (1.080x vs-llama, docs/CUDA_OPTIMIZATION.md P6 r59b); `MINFER_MMQ=0`
    /// opts out to the legacy f16 w16-cache prefill (~2353 tok/s), the
    /// 2026-08-31 default from when the untuned kernel measured ~2.5-3
    /// GMAC/s per matmul under GPU contention vs ~8-11 for the f16
    /// w16-cache path (7B @2K: ~155 vs ~630-880 tok/s). All other A/B
    /// escapes still work: `MINFER_NO_PREFILL_GEMM=1` (legacy per-type
    /// kernels).
    fn mmq_enabled() -> bool {
        Self::mmq_gate_on("MINFER_MMQ")
    }

    /// R1: the int8 MMQ prefill GEMM is active — sm_80+ (mma.m16n8k32 s8;
    /// sm_75 only has k16) and opted in via MINFER_MMQ=1. The loader also
    /// uses this to skip the f16 cache warm pass (MMQ streams raw weight
    /// bytes; the w16 copy would be dead weight).
    /// T1: the sm_80+ check is now the resolved tier's MMQ flag
    /// (`tier_mmq`: table flag, or `cc >= 800` for the GENERIC row) — same
    /// verdict on every currently-built-for device, tier-aware elsewhere.
    pub fn mmq_active(&self) -> bool {
        self.tier_mmq && Self::mmq_enabled()
    }

    /// Device compute capability in `major*100 + minor` encoding (GB10
    /// sm_12.1 = 1201); 0 when no device is initialized. Used by tests to
    /// gate sm_80+ kernels.
    #[cfg(test)]
    pub fn cc(&self) -> i32 {
        self.cc.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// r51/r52: producer-fused A-quantize mode — 0 off, 1 = fused + f32
    /// output write (MINFER_MMQ_A_FUSE=1 semantics), 2 = fused + SKIP the f32
    /// output write (MINFER_MMQ_A_FUSE=2; r60: mode 2 is the DEFAULT when
    /// unset — still degrading to mode 1 under any window-reader/fallback
    /// condition; "0" = off). The base gate is the full r49 MMQ
    /// gate set (the fused plane is only CONSUMED by the raw NB-BT (q4_K) /
    /// q6_K NB transposed GEMM paths, all of which these gates enable; fusing
    /// with any of them off would write a plane the matmul re-quantizes
    /// natively — correct but pure waste).
    ///
    /// Mode 2 additionally requires the r52 window-safety conditions: the f32
    /// output is never written, so (a) every debug/trace reader of node
    /// buffers must be off (MINFER_GRAPH_DUMP layer-0 node dumps, the
    /// debug_dump feature's MINFER_DUMP_DIR, MINFER_TRACE per-node capture,
    /// --viz live capture), and (b) no GEMM path that reads the f32 A may be
    /// reachable (MINFER_NO_PREFILL_GEMM=1 legacy kernels). If any condition
    /// fails, mode 2 degrades to mode-1 semantics (fused, writes the f32) —
    /// still the r51 win, just without the skip. The remaining window
    /// guarantee (producers' outputs consumed only by their immediately
    /// consecutive MatMul group via the MmqCache plane; any dead-write cache
    /// miss REFUSES loudly) is enforced structurally — see MmqCache::dead_write
    /// and docs/CUDA_OPTIMIZATION.md P6 r52 for the full proof.
    pub fn mmq_a_fuse_mode(&self) -> u8 {
        if !(self.mmq_active()
            && Self::mmq_gate_on("MINFER_MMQ_RAW")
            && Self::mmq_gate_on("MINFER_MMQ_RAW_NB")
            && Self::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE")
            && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB"))
        {
            return 0;
        }
        // r60 promotion: unset = mode 2 (the verified-best skip-write fused
        // producers); "1"/"2" keep the r51/r52 override semantics; "0" (and
        // any other unrecognized value, as before r60) = off.
        let requested = match std::env::var("MINFER_MMQ_A_FUSE").as_deref() {
            Err(std::env::VarError::NotPresent) => 2,
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        // r60: mode 2 additionally requires the NB-BT-only weight mix (see
        // `nb_bt_only`) — a mixed-quant model degrades to mode 1 regardless
        // of how mode 2 was requested (default or explicit "2").
        let mode2_possible = requested == 2
            && !Self::no_prefill_gemm()
            && self.nb_bt_only.load(std::sync::atomic::Ordering::Relaxed);
        match requested {
            1 => 1,
            2 if mode2_possible
                && std::env::var_os("MINFER_GRAPH_DUMP").is_none()
                && std::env::var_os("MINFER_DUMP_DIR").is_none()
                && !crate::trace::enabled()
                && !crate::live::enabled() =>
            {
                2
            }
            // Mode 2 requested but a window-reader/fallback condition is
            // active: keep the r51 fused semantics (write the f32 output).
            2 => 1,
            _ => 0,
        }
    }
}
