// f32-activation matmul dispatch (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // ─── FFI declarations for kernel launch wrappers ───────────
    pub(crate) fn launch_q4_0_q8_0_matmul(
        weights: *const u8,
        acts: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q4_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q8_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q4_1_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q4_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_f32_matmul_padded(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_f32_f32_matmul(
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // #141: f16 weights × f32 activations (raw half bits). Both return 0 on
    // success and non-zero when the launch itself failed (#147 checking; the
    // caller turns it into an `Err`, never a silent fallback).
    pub(crate) fn launch_f16_f32_matmul(
        w: *const u8,
        x: *const f32,
        out: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // #208: bf16 weights × f32 activations (raw 2 B/element words, `bits << 16`
    // in-register). Same checked-return contract as the f16 launcher above.
    pub(crate) fn launch_bf16_f32_matmul(
        w: *const u8,
        x: *const f32,
        out: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn launch_q5_1_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q5_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q5_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
}

impl CudaState {
    // ─── Kernel launch operations (called from graph/cuda_backend.rs) ──

    /// f32-activation matmul dispatch by raw weight pointer + tensor type.
    /// The graph backend (graph/cuda_backend.rs) resolves weights by name and
    /// holds no Tensor, so dispatch takes (ptr, ttype) directly.
    ///
    /// Test-only (#240): production goes through `matmul_f32_ptr_layout`
    /// directly; the only callers of this `padded_q6k: false` shorthand are the
    /// device gates in `graph::cuda_backend::tests`, so `#[cfg(test)] pub(crate)`
    /// is the T1 (#238) form — a `#[cfg(test)]` module in another file is the
    /// caller, so it cannot move into `cuda/tests.rs`.
    #[cfg(test)]
    pub(crate) fn matmul_f32_ptr(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) -> Result<(), String> {
        self.matmul_f32_ptr_layout(wptr, ttype, x, out, od, id, nt, false)
    }

    /// `padded_q6k`: the weight buffer was registered via
    /// `register_weight_q6k_padded` (224-byte block stride).
    pub fn matmul_f32_ptr_layout(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
    ) -> Result<(), String> {
        // 8m: prefill (nt >= 16 — Step 82 lowered the gate to >= 9: batches
        // of 2..8 run the multi-token MMVQ / token-looped legacy kernels,
        // which are weights-once without the M-tile padding waste; doc 81)
        // runs ONE tiled GEMM for every quantized
        // weight type. R1 (2026-08-31): the int8 MMQ GEMM — activations
        // quantized to q8_0 once per call, raw weight bytes staged per tile,
        // mma.m16n8k32 (s8) with per-k-block scale rescale (llama.cpp's MMQ
        // structure; see mmq_nt_kernel). MINFER_MMQ-gated (r60 PROMOTION:
        // default ON — the promoted 1.080x path; `MINFER_MMQ=0` = the f16
        // wmma path). MINFER_NO_PREFILL_GEMM=1 still forces the legacy
        // per-type kernels. id % 32 == 0 covers the block math of every type
        // (q6_K runs as k32 chunks with dual 16-sub rescale inside).
        // doc 91 experiment gate: route nt 2..8 to the mma BT path. Off by
        // default — at nt<=64 the BT grid has only od/NBJ blocks (40 on the
        // 14B shapes) and runs ~1.7x over the multi-MMVQ path; the K-split
        // follow-up raises the block count before this can win.
        let small_m_gemm =
            std::env::var("MINFER_SMALL_M_GEMM").map_or(false, |v| v == "1") && nt >= 2;
        if (nt >= 9 || small_m_gemm)
            && id % 32 == 0
            && !Self::no_prefill_gemm()
            && matches!(
                ttype,
                TensorType::Q4_0
                    | TensorType::Q4_1
                    | TensorType::Q5_0
                    | TensorType::Q5_1
                    | TensorType::Q8_0
                    | TensorType::Q4_K
                    | TensorType::Q5_K
                    | TensorType::Q6_K
            )
        {
            if self.mmq_active() {
                // doc 92 resolution: auto-ksplit is the DEFAULT for every
                // route into the BT GEMM. The auto formula activates only
                // while the grid is M-starved (nt <= 64 => ntb == 1) and
                // falls back to 1 above that, so prefill at nt >= 65 keeps
                // the unsplit single-pass association. Deterministic per
                // (nt, od, id), so capture and replay follow the same
                // partial-sum order.
                let ksplit_req = -1;
                let _ = small_m_gemm;
                return self.prefill_mmq(wptr, ttype, x, out, od, id, nt, padded_q6k, ksplit_req);
            }
            return self.prefill_gemm_f16(wptr, ttype, x, out, od, id, nt, padded_q6k);
        }
        let stream = self.stream();
        // T1 tier note (plan §5.4/§14 R8): the resolved tier's per-type batch
        // limits are tabled and unit-tested (`device_tier::mmvq_cap`) but NOT
        // wired into the decode arms yet — a limit < 8 has no destination for
        // the vacated nt range (BT starts at nt >= 9; the MINFER_SMALL_M_GEMM
        // experiment measured ~1.7x over multi-MMVQ at small nt, doc 91) and
        // would strip the spec identity family (R3). Activation waits for a
        // small-nt BT destination (T2 tile candidates) or field A/B data.
        macro_rules! launch {
            ($f:ident) => {{
                unsafe {
                    $f(
                        wptr as *const u8,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    );
                }
                Ok(())
            }};
        }
        match ttype {
            TensorType::Q4_0 => {
                // 8c: prefill Q8_0-activation GEMM — quantize activations once
                // and run the int8-dot kernel. Standalone A/B (7e② method,
                // bench8c): +38–44% at id ≤ 8192 (activation-heavy shapes:
                // 0.5B all, 7B attn/qkv/o, 7B ffn_gu +4.7%); −63% at
                // 7B ffn_down (id=18944, weight-stream-bound — the q8_0
                // kernel streams weight bytes slower), so the shape gate
                // excludes it. Decode (nt == 1) keeps the f32 kernel.
                // Prefill never enters a CUDA Graph capture window (8g①
                // decode-only gate), so the on-demand scratch grow is safe.
                if nt > 1 && id <= 8192 {
                    let q8 = Self::get_or_grow(
                        &self.buf_q8_prefill,
                        nt * (id / 32) * Q8B,
                    );
                    self.quantize_q8_0(x, q8, id, nt);
                    unsafe {
                        launch_q4_0_q8_0_matmul(
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            od as i32,
                            id as i32,
                            nt as i32,
                            stream,
                        );
                    }
                    Ok(())
                } else if nt == 1 && id >= 2048 && id % 32 == 0 && !Self::no_q40_mmvq() {
                    // doc 103: decode MMVQ (the 8c int8 branch above owns
                    // nt > 1 && id <= 8192 and is untouched; these branches
                    // only claim shapes that previously ran the f32 kernel).
                    self.q4_0_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id > 8192 && id % 32 == 0 && !Self::no_q40_mmvq() {
                    self.q4_0_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q4_0_f32_matmul)
                }
            }
            TensorType::Q8_0 => {
                // doc 103: decode joins the MMVQ structure (dp4a over the
                // shared pad40 q8 activations; the 34-B block stride keeps
                // the f32 kernel's coalesced weight loop out of the picture
                // — measured crossover in doc 103). The f32 kernel stays as
                // the fallback (MINFER_NO_Q80_MMVQ=1) and for every shape
                // the gates exclude.
                if nt == 1 && id >= 2048 && id % 32 == 0 && !Self::no_q80_mmvq() {
                    // doc 104: prefer the p32 split planes when registered
                    // (byte-equal output, +6-10% kernel bandwidth); the raw
                    // doc 103 kernel stays as the fallback.
                    if !Self::no_q80_p32() {
                        if let Some((pp, pd)) = self.q80_p32_planes(wptr as usize) {
                            self.q8_0_p32_decode_mmvq(pp, pd, x, out, od, id, nt);
                            return Ok(());
                        }
                    }
                    self.q8_0_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id >= 2048 && id % 32 == 0 && !Self::no_q80_mmvq() {
                    // size floor as in nt == 1: below ~2048 the extra
                    // quantize launch and the q8 activation rounding lose to
                    // the f32 kernel (and tiny shapes keep the exact f32
                    // numerics the small-shape tests pin down)
                    if !Self::no_q80_p32() {
                        if let Some((pp, pd)) = self.q80_p32_planes(wptr as usize) {
                            self.q8_0_p32_decode_mmvq_multi(pp, pd, x, out, od, id, nt);
                            return Ok(());
                        }
                    }
                    self.q8_0_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q8_0_f32_matmul)
                }
            }
            TensorType::Q4_1 => launch!(launch_q4_1_f32_matmul),
            TensorType::Q4_K => {
                // 8e-reversal: decode (nt == 1) runs the MMVQ structure
                // (dp4a over q8 activations, one row per 256-thread block) —
                // +74–77% at 7B shapes (bench8e2); id >= 2048 gate (below
                // that it is launch-latency noise), id % 32 == 0 for the
                // sub-block tail granularity. Prefill keeps the f32 kernel.
                if nt == 1 && id >= 2048 && id % 32 == 0 {
                    self.q4_k_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id % 32 == 0 {
                    // Step 82: multi-token MMVQ (in-block token loop).
                    self.q4_k_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q4_k_f32_matmul)
                }
            }
            TensorType::Q5_1 => launch!(launch_q5_1_f32_matmul),
            TensorType::Q5_0 => launch!(launch_q5_0_f32_matmul),
            TensorType::Q5_K => {
                // 8f: partial tail super-blocks are masked at 32-element
                // granularity inside the kernel — finer tails unsupported.
                if id % 32 != 0 {
                    return Err(format!(
                        "cuda: Q5_K id {id} not a multiple of 32 (tail masking granularity)"
                    ));
                }
                // 8e follow-up: decode (nt == 1) joins the MMVQ structure
                // (dp4a over q8 activations, one row per 256-thread block).
                // Shape gate measured on-device (dbg micro-bench, padded f32
                // vs mmvq): od*id < ~24M elements loses (od 512 → 4.5x
                // slower, 896 → 3.0x, 2048x4864 → 1.66x) because 1-2 units
                // per thread expose the uncoalesced q5/q6 byte loads; large
                // shapes win (7B ffn_down 3584x18944 → 1.5x faster, lm_head
                // 152064x3584 → 1.4x). MINFER_NO_KQ_MMVQ=1 forces f32.
                if nt == 1 && od * id >= 24_000_000 && !Self::no_kq_mmvq() {
                    self.q5_k_decode_mmvq(wptr, x, out, od, id, nt);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && !Self::no_kq_mmvq() {
                    // Step 82: multi-token MMVQ — weights-once at any shape
                    // (the 24M nt == 1 crossover does not apply: the in-block
                    // token loop amortizes the uncoalesced load latency).
                    self.q5_k_decode_mmvq_multi(wptr, x, out, od, id, nt);
                    Ok(())
                } else {
                    launch!(launch_q5_k_f32_matmul)
                }
            }
            TensorType::Q6_K => {
                // 8e follow-up: decode (nt == 1) joins the MMVQ structure —
                // 16-element units over q8 activations. blk_stride follows
                // the weight registration (224 padded 7e② repack / 210 raw).
                // Shape gate: see the Q5_K arm comment (measured od*id
                // crossover ~24M elements; below it the padded f32 kernel's
                // coalesced loop wins, above it MMVQ's dp4a wins).
                // D3-7 2b: gate lowered 24M -> 4M for the attn_v class —
                // the 14B attn_v (od 1024 x id 5120 = 5.24M, 11 layers)
                // sat on the padded-f32 kernel at 134.8 GB/s (D3-1 census)
                // while its q6_K MMVQ siblings sustain the 200-225 GB/s
                // class; attn_v/attn_o share the attention-output buffer
                // and id, so the D3-5 MmqCache dedupes their standalone
                // quantize to one launch. GGUF census: no other q6_K shape
                // falls in (4M, 24M) (7B attn_v 1.8M stays padded-f32).
                // Tolerance-gated (f32->q8 activation rounding): D3a
                // package; MINFER_NO_KQ_MMVQ=1 keeps the padded kernel.
                if nt == 1 && id % 32 == 0 && od * id >= 4_000_000 && !Self::no_kq_mmvq() {
                    self.q6_k_decode_mmvq(wptr, x, out, od, id, nt, padded_q6k);
                    Ok(())
                } else if nt >= 2 && nt <= 8 && id % 32 == 0 && !Self::no_kq_mmvq() {
                    // Step 82: multi-token MMVQ (v2 needs the padded 224B
                    // stride; the v1 form handles raw 210B too). The 4M
                    // nt == 1 crossover does not apply — weights-once at
                    // any shape once nt >= 2.
                    self.q6_k_decode_mmvq_multi(wptr, x, out, od, id, nt, padded_q6k);
                    Ok(())
                } else if padded_q6k {
                    launch!(launch_q6_k_f32_matmul_padded)
                } else {
                    launch!(launch_q6_k_f32_matmul)
                }
            }
            TensorType::F32 => {
                unsafe {
                    launch_f32_f32_matmul(
                        wptr as *const f32,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    );
                }
                Ok(())
            }
            // #141: an f16 weight matmul. The weights stay 2 B/element on the
            // device (no registration-time f32 copy — that would give up the
            // memory the f16 file exists to save), converted in-register by the
            // kernel. It never enters the int8 MMQ prefill GEMM above: MMQ
            // streams *quantized* bytes and f16 is not one of its formats.
            TensorType::F16 => {
                let rc = unsafe {
                    launch_f16_f32_matmul(
                        wptr as *const u8,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    )
                };
                if rc != 0 {
                    // #147 rule: the launch named itself at the site (see the
                    // `minfer/cuda: kernel launch …` line on stderr); refuse
                    // here instead of running on into a checked error.
                    return Err(format!(
                        "cuda: f16 matmul launch failed for [{od}x{id}] x nt={nt} \
                         (site launch:f16_f32_matmul_*)"
                    ));
                }
                Ok(())
            }
            // #208: a bf16 weight matmul — the f16 arm's exact sibling. Raw
            // 2 B/element words stay on the device; `bf16_f32_matmul_vec` /
            // `_scalar` promote in-register with `bits << 16`, which is exact.
            // Deliberately its own arm over its own kernel rather than a flag on
            // the f16 one: the two decodes differ (`__half22float2` vs a shift)
            // and folding them would put a per-element branch in the inner loop
            // of the hottest device kernel. Like f16 it never enters the int8
            // MMQ prefill GEMM: MMQ streams *quantized* bytes and bf16 is not one
            // of its formats.
            TensorType::BF16 => {
                let rc = unsafe {
                    launch_bf16_f32_matmul(
                        wptr as *const u8,
                        x as *const f32,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        stream,
                    )
                };
                if rc != 0 {
                    // #147 rule: the launch named itself at the site (see the
                    // `minfer/cuda: kernel launch …` line on stderr); refuse
                    // here instead of running on into a checked error.
                    return Err(format!(
                        "cuda: bf16 matmul launch failed for [{od}x{id}] x nt={nt} \
                         (site launch:bf16_f32_matmul_*)"
                    ));
                }
                Ok(())
            }
            other => Err(format!(
                "cuda: weight type {other:?} has no f32-activation matmul kernel (supported: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K)"
            )),
        }
    }
}
