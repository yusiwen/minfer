// attention (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    pub(crate) fn launch_gqa_attn_f32(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        bound: *const i32,
        mode: i32,
        layout: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        row_bytes: usize,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // 8n: FA-style prefill attention. Returns -1 when the >48KB dynamic
    // shared-memory opt-in fails (then Rust falls back to the legacy kernel).
    // #144 item 3: `layout` is the KV tag the staging reads (f16 or packed
    // Q8_0 — the packed arm dequantizes each cell into the same f16 tile);
    // `row_bytes` is the cell's byte width and is ignored by the f16 arm.
    pub(crate) fn launch_fa_prefill_kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        layout: i32,
        row_bytes: usize,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn launch_gqa_attn_f32_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_gqa_attn_split_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_gqa_attn_split_f32kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        stream: *mut std::ffi::c_void,
    );
    // C4 S2b: the packed decode path (nt == 1). One 1-warp split-K launch per
    // window mode, `rpw_gate = 0` — the hybrid 4-warp body is f16-typed.
    pub(crate) fn launch_gqa_attn_split_q8_0(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        row_bytes: usize,
        dp4a: i32,
        wide: i32,
        stream: *mut std::ffi::c_void,
    );
    // doc 94: batched split attention for the verify shapes (1 < nt <= 16) —
    // bitwise-equal per position to the nt=1 decode split path.
    pub(crate) fn launch_gqa_attn_split_batched_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn launch_gqa_attn_split_batched_f32kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        bound: *const i32,
        mode: i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

impl CudaState {
    /// C4 S2b: the general nt > 1 attention kernel, layout-tagged. `row_bytes` is
    /// the KV cell's byte width (`nk * hd * {4,2}` for f32/f16,
    /// `KvFormat::Q8_0.row_bytes(nkt)` for a packed region).
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attn_f32(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        layout: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        row_bytes: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_gqa_attn_f32(
                q as *const f32,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                o as *mut f32,
                positions as *const i32,
                mode,
                layout,
                nh as i32,
                nk as i32,
                hd as i32,
                scale,
                row_bytes,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 94: batched split attention for the verify shapes (1 < nt <= 16).
    /// Per-token nkv = positions[t]+1 with the decode path's exact
    /// attn_split_1w_body arithmetic and combine merge order, so a verify
    /// batch's logits are bitwise-equal to running the nt=1 decode path at
    /// each position (the greedy identity). The partials scratch grows to
    /// nt * SPLITS * nh * pstr — sized at warmup for the verify nt, stable
    /// within a capture window.
    pub fn gqa_attn_split_batched(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        f16_kv: bool,
        nt: usize,
    ) {
        let pstr = ((4 + hd + 3) & !3) as i32;
        const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
        let need = nt * ATTN_SPLITS * nh * (pstr as usize) * 4;
        let partial = Self::get_or_grow(&self.buf_attn_partial, need);
        let stream = self.stream();
        unsafe {
            if f16_kv {
                launch_gqa_attn_split_batched_f16kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    nt as i32,
                    stream,
                );
            } else {
                launch_gqa_attn_split_batched_f32kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// 8b: GQA attention over a **staged** KV cache (K/V materialized into the
    /// FA f16 tile). `layout` is the KV tag: `KV_LAYOUT_F16` reads halves and
    /// `KV_LAYOUT_Q8_0` dequantizes each packed cell while staging (#144 item 3);
    /// q/o stay f32 in both. Matches Metal's pl_gqa_attn_f16 precision class
    /// (f16 storage, f32 accumulate). `row_bytes` is the packed cell's byte
    /// width and is ignored by the f16 arm.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attn_kv_prefill(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        layout: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        row_bytes: usize,
        nt: usize,
    ) {
        // #189's mutation seam: 1 unless `MINFER_S4_AB_MAP_REPS=2` doubles the
        // map arm's work (`s4_ab_map_reps`).
        for _ in 0..s4_ab_map_reps(mode) {
            self.gqa_attn_kv_prefill_once(
                q, k, v, o, positions, mode, layout, nh, nk, hd, scale, row_bytes, nt,
            );
        }
        // The observation half of gate contract rule 3: the map instantiation
        // really ran. The S4 A/B resets this counter, runs one map call and one
        // span call, and asserts the map call moved it while the span did not.
        if mode == AttnWindow::Map.code() {
            crate::testfail::note_checked("cuda_attn_map_window");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_attn_kv_prefill_once(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        layout: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        row_bytes: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        // 8n: prefill (nt >= 64) runs the FA-style tiled attention. The
        // legacy kernel is one block per (token, head) — K re-read per token
        // per head (7B @2K: ~132 GB/layer) with a 128-register accumulator —
        // and measured 176 ms/layer, 76% of the whole 2K prefill. hd % 16
        // is hard-wired (FA_HQ = hd/4 = 32) and the shared-memory opt-in
        // can fail on constrained devices, hence the rc fallback.
        //
        // D5-R stage 4 (doc 86): the verify shapes (nt = d+1, i.e. 2..=9)
        // used to fall through to the legacy per-(token,head) kernel — the
        // doc 85 ledger prices that hole at ~9 ms of the 17.6 ms nt=3
        // marginal (the nt=1 split-KV path does the same KV read in 0.8 ms).
        // fa_prefill masks rows causally from the positions array ("positions
        // are data"), which is exactly the verify block's structure, so the
        // gate is lowered to nt >= 2; nt == 1 keeps the split-KV decode path.
        //
        // C8b S4: a `kv_map` window is gathered here too — the staging resolves
        // each linear window index through the runs, and the tile's per-row mask is
        // that query's row count (a map window is a prefix of the sequence's address
        // space, so the existing `gcol < limit` form stays exact).
        //
        // #144 item 3: the same FA body serves a packed cache — the staging
        // dequantizes each Q8_0 cell into the f16 tile instead of copying halves,
        // so the tensor-core QK^T/P·V stream is unchanged. The general
        // layout-tagged kernel remains the documented fallback (the `rc == -1`
        // arm below), and the packed arm reaches it with the same rounded
        // window modes.
        if nt >= 2 && hd == 128 && !Self::no_fa_prefill() && layout != crate::cuda::KV_LAYOUT_F32 {
            let rc = unsafe {
                launch_fa_prefill_kv(
                    q as *const f32,
                    k as *const std::ffi::c_void,
                    v as *const std::ffi::c_void,
                    o as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    nt as i32,
                    layout,
                    row_bytes,
                    stream,
                )
            };
            if rc == 0 {
                // #144 item 3: the observation half (gate contract rule 3) — a gate
                // that must prove the packed FA path *ran* cannot read the dispatch's
                // own answer, so the chokepoint records it and
                // `cuda_q8_0_fa_prefill_attention_parity` asserts the counter moved.
                if layout == crate::cuda::KV_LAYOUT_Q8_0 {
                    crate::testfail::note_checked("cuda_fa_prefill_q8_0");
                }
                return;
            }
        }
        if layout == crate::cuda::KV_LAYOUT_Q8_0 {
            // #144: the packed fallback is the general layout-tagged kernel, not
            // the f16-typed one — `launch_gqa_attn_f32_f16kv` would read the packed
            // bytes as halves.
            self.gqa_attn_f32(
                q, k, v, o, positions, mode, layout, nh, nk, hd, scale, row_bytes, nt,
            );
            return;
        }
        unsafe {
            launch_gqa_attn_f32_f16kv(
                q as *const f32,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                o as *mut f32,
                positions as *const i32,
                mode,
                nh as i32,
                nk as i32,
                hd as i32,
                scale,
                nt as i32,
                stream,
            );
        }
    }

    /// C4 S2b: the decode (nt == 1) split-K path, layout-tagged. A packed cache
    /// takes `launch_gqa_attn_split_q8_0`, whose `rpw_gate = 0` skips the hybrid
    /// 4-warp body (f16-typed); f32/f16 keep their pre-C4 launchers unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attn_split(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        layout: i32,
        row_bytes: usize,
    ) {
        // #189's mutation seam: 1 unless `MINFER_S4_AB_MAP_REPS=2` doubles the
        // map arm's work (`s4_ab_map_reps`). The launch is idempotent, so the
        // second issue writes the same bytes and only the cost changes.
        for _ in 0..s4_ab_map_reps(mode) {
            self.gqa_attn_split_once(
                q, k, v, o, positions, mode, nh, nk, hd, scale, layout, row_bytes,
            );
        }
        // The observation half of gate contract rule 3 (see `gqa_attn_kv_prefill`).
        if mode == AttnWindow::Map.code() {
            crate::testfail::note_checked("cuda_attn_map_window");
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn gqa_attn_split_once(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        mode: i32,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        layout: i32,
        row_bytes: usize,
    ) {
        let pstr = ((4 + hd + 3) & !3) as i32;
        const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
        let need = ATTN_SPLITS * nh * (pstr as usize) * 4;
        let partial = Self::get_or_grow(&self.buf_attn_partial, need);
        let stream = self.stream();
        // #186: the Q8_0 decode arm's same-binary A/B control. The flag is resolved
        // once per process (`cuda::q8_kv_dp4a_enabled`) and passed as a value; the
        // launcher picks the `__dp4a` or the convert-based instantiation from it.
        // #202 adds a second, independent arm (`wide`): the K/V four-quant groups
        // are loaded with two 16-bit loads instead of four byte loads. It is only
        // meaningful on the dp4a arm, so `dp4a` already gates it in the launcher.
        let dp4a = q8_kv_dp4a_enabled();
        let wide = q8_kv_wide_enabled();
        unsafe {
            if layout == KV_LAYOUT_Q8_0 {
                launch_gqa_attn_split_q8_0(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    row_bytes,
                    dp4a as i32,
                    wide as i32,
                    stream,
                );
                // The observation half of gate contract rule 3: a gate that must
                // prove the int dot ran cannot read the launch's own report. The
                // chokepoint is bumped only when this call launched the dp4a arm.
                if dp4a {
                    crate::testfail::note_checked("cuda_q8_kv_dp4a");
                }
            } else if layout == KV_LAYOUT_F16 {
                launch_gqa_attn_split_f16kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    stream,
                );
            } else {
                launch_gqa_attn_split_f32kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
                    mode,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    pstr,
                    stream,
                );
            }
        }
    }
}
