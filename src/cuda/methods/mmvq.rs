// decode MMVQ matmuls (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    pub(crate) fn launch_q4_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q5_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // doc 103: q4_0 / q8_0 decode MMVQ (the 8e structure on the legacy
    // f32-activation types; new code only — see cuda_kernels.cu tail).
    pub(crate) fn launch_q4_0_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q4_0_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q8_0_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q8_0_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // doc 104: q8_0 p32 split-plane variants (payload plane + dense d plane)
    pub(crate) fn launch_q8_0_p32_q8_mmvq(
        plane_p: *const u8,
        plane_d: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q8_0_p32_q8_mmvq_multi(
        plane_p: *const u8,
        plane_d: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // Step 82: multi-token (nt in [2, 8]) MMVQ variants — the token loop is
    // in-block (one block per weight row, grid.y = 1), so the weight stream
    // is paid once regardless of nt (the doc 81 D5-1a dispatch hole).
    pub(crate) fn launch_q4_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q4_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q5_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q5_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // R2: weight-streaming rework (one weight byte loaded exactly once per
    // row; uint4 loads; q6_K needs the padded 224B stride for alignment).
    pub(crate) fn launch_q4_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // D3b-1b: pipelined q6_K MMVQ for tall rows (npair > 256); bitwise-identical
    pub(crate) fn launch_q6_k_q8_mmvq_v2_pf(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // D4-4 L1: dense split-plane (dpl) decode kernels — bitwise-identical
    // to the padded forms, no 224B pad sectors (see the kernel comments).
    pub(crate) fn launch_q6_k_q8_mmvq_v2_pf_dpl(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        nbe: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q6_k_q8_mmvq_v2_dpl(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        nbe: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_q5_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
}

impl CudaState {
    /// doc 103: decode (nt == 1) q4_0 matmul via the MMVQ structure — dp4a
    /// over the shared pad40 q8 activation plane, one row per 256-thread
    /// block, nibble offset -8 folded into the dot via the 8*sx correction.
    pub fn q4_0_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_mmvq(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 103: multi-token (nt in [2, 8]) q4_0 MMVQ — same in-block token
    /// loop as the K-quant multi variants; bitwise-consistent with the
    /// nt == 1 kernel (same per-u order and reduction).
    pub fn q4_0_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_mmvq_multi(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 103: decode (nt == 1) q8_0 matmul via the MMVQ structure — the
    /// 34-B block stride is only 2B-aligned, so the payload reads use the
    /// q6_K two-u16-halves pattern.
    pub fn q8_0_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_q8_mmvq(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 103: multi-token (nt in [2, 8]) q8_0 MMVQ.
    pub fn q8_0_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_q8_mmvq_multi(
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 104: decode q8_0 matmul over the p32 split planes — byte-equal
    /// arithmetic to `q8_0_decode_mmvq`, wider weight loads (uint4 x2 per
    /// 32-element unit instead of 16 scattered u16s).
    pub fn q8_0_p32_decode_mmvq(
        &self,
        pp: *const u8,
        pd: *const u8,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_p32_q8_mmvq(
                pp,
                pd,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// doc 104: multi-token p32 variant (weight words hoisted out of the
    /// token loop; bitwise-consistent with the nt == 1 p32 kernel).
    pub fn q8_0_p32_decode_mmvq_multi(
        &self,
        pp: *const u8,
        pd: *const u8,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            launch_q8_0_p32_q8_mmvq_multi(
                pp,
                pd,
                q8 as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// 8e-reversal: decode (nt == 1) q4_K matmul via the llama.cpp MMVQ
    /// structure — quantize the activation row to padded 40B q8 blocks in
    /// `buf_q8_decode`, then run the dp4a one-row-per-block kernel. The
    /// caller gates on id % 32 == 0 (sub-block granularity) and id >= 2048
    /// (below that the structure win is launch-latency noise).
    pub fn q4_k_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        // D3-5 1a: consult the MmqCache first — the fused decode producers
        // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
        // pad40 plane, so the matmul group following the producer skips the
        // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
        // restores the unconditional standalone launch).
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q4_k_q8_mmvq_v2(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q4_k_q8_mmvq(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// 8e follow-up: decode (nt == 1) q6_K matmul via the same MMVQ
    /// structure — the shared q8 activation scratch, then the 16-element
    /// unit dp4a kernel. `blk_stride` follows the weight registration:
    /// 224 for the padded 7e② repack, 210 for raw bytes.
    pub fn q6_k_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        blk_stride_padded: bool,
    ) {
        // D3-5 1a: consult the MmqCache first — the fused decode producers
        // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
        // pad40 plane, so the matmul group following the producer skips the
        // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
        // restores the unconditional standalone launch).
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        // D4-4 L1: dense split-plane fast path (bitwise — see the
        // registration comment). Falls through to the padded kernels when
        // the plane is absent (MINFER_Q6K_DPL=0 / id not a multiple of
        // 256 / map miss). The pf-vs-loop shape gate mirrors the padded
        // dispatch below (same MINFER_Q6K_PF opt-out semantics).
        if blk_stride_padded && Self::mmq_gate_on("MINFER_Q6K_DPL") {
            let dwp = self
                .q6k_dpl
                .lock()
                .unwrap()
                .get(&(wptr as usize))
                .map(|cp| cp.0);
            if let Some(dwp) = dwp {
                unsafe {
                    let nbe = (id >> 8) as i32;
                    if id > 8192
                        && id <= 16384
                        && !std::env::var("MINFER_Q6K_PF").map_or(false, |v| v == "0")
                    {
                        launch_q6_k_q8_mmvq_v2_pf_dpl(
                            dwp as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            od as i32,
                            id as i32,
                            nt as i32,
                            nbe,
                            stream,
                        );
                    } else {
                        launch_q6_k_q8_mmvq_v2_dpl(
                            dwp as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            od as i32,
                            id as i32,
                            nt as i32,
                            nbe,
                            stream,
                        );
                    }
                }
                return;
            }
        }
        unsafe {
            if Self::mmvq_v2(id) && blk_stride_padded {
                // v2's uint4 ql/qh loads need the padded 224B stride
                // D4-2 B0 correctness fix: v2_pf processes exactly TWO units
                // per thread (u = tid, tid+256 → npair ≤ 512), but the old
                // gate (id > 8192) had no upper bound — npair=592 shapes
                // (7B ffn_down id 18944, 10 q6_K layers) silently dropped
                // units 512..591, corrupting 7B decode since D3b-1b. Guard
                // the upper bound; taller rows take the v2 loop form, which
                // walks any npair with identical per-unit arithmetic and the
                // same ascending-u accumulation order (bitwise for every
                // npair ≤ 512 shape, which keep the pipelined kernel).
                // D4-2 B1c: MINFER_Q6K_PF=0 A/Bs the pipelined form against
                // the v2 loop at these shapes (both cover the full unit set).
                if id > 8192
                    && id <= 16384
                    && !std::env::var("MINFER_Q6K_PF").map_or(false, |v| v == "0")
                {
                    // D3b-1b: tall rows (npair > 256, e.g. ffn_down id 13824)
                    // run the pipelined variant (bitwise-identical, loads for
                    // both serial units issue up front).
                    launch_q6_k_q8_mmvq_v2_pf(
                        wptr as *const u8,
                        q8 as *const u8,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        224,
                        stream,
                    );
                } else {
                    launch_q6_k_q8_mmvq_v2(
                        wptr as *const u8,
                        q8 as *const u8,
                        out as *mut f32,
                        od as i32,
                        id as i32,
                        nt as i32,
                        224,
                        stream,
                    );
                }
            } else {
                launch_q6_k_q8_mmvq(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    if blk_stride_padded { 224 } else { 210 },
                    stream,
                );
            }
        }
    }

    /// Step 82: multi-token (nt in 2..=8) q4_K matmul via the MMVQ
    /// structure with an in-block token loop — the weight stream is paid
    /// once regardless of nt (the doc 81 D5-1a dispatch hole). The
    /// nt == 1 id >= 2048 shape gate does not apply: at nt >= 2 a
    /// weights-once kernel wins at any shape.
    pub fn q4_k_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        // decode_quantize_native is nt-parameterized (the pad40 q8 plane
        // covers all nt rows), so no scratch change is needed here.
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q4_k_q8_mmvq_v2_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q4_k_q8_mmvq_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// Step 82: multi-token (nt in 2..=8) q5_K matmul — the q4_K multi
    /// structure with the q5 high-bit plane folded in.
    pub fn q5_k_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q5_k_q8_mmvq_v2_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q5_k_q8_mmvq_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// Step 82: multi-token (nt in 2..=8) q6_K matmul — 16-element units
    /// over q8 activations with an in-block token loop. The v2 form needs
    /// the padded 224B block stride (u32/uint4 loads); the v1 form handles
    /// the raw 210B stride too (2-byte loads).
    pub fn q6_k_decode_mmvq_multi(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        blk_stride_padded: bool,
    ) {
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        let stride: i32 = if blk_stride_padded { 224 } else { 210 };
        unsafe {
            if blk_stride_padded && Self::mmvq_v2(id) {
                launch_q6_k_q8_mmvq_v2_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stride,
                    stream,
                );
            } else {
                launch_q6_k_q8_mmvq_multi(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stride,
                    stream,
                );
            }
        }
    }

    /// 8e follow-up: decode (nt == 1) q5_K matmul via the same MMVQ
    /// structure (q4_K shape with the q5 high-bit plane folded in).
    pub fn q5_k_decode_mmvq(
        &self,
        wptr: *mut std::ffi::c_void,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        // D3-5 1a: consult the MmqCache first — the fused decode producers
        // (rms_norm_quant_on_gpu / swiglu_quant_off_on_gpu) recorded their
        // pad40 plane, so the matmul group following the producer skips the
        // standalone quantize launch entirely (MINFER_NO_DECODE_A_FUSE=1
        // restores the unconditional standalone launch).
        let q8 = self.decode_quantize_native(x as *const f32, id, nt);
        let stream = self.stream();
        unsafe {
            if Self::mmvq_v2(id) {
                launch_q5_k_q8_mmvq_v2(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            } else {
                launch_q5_k_q8_mmvq(
                    wptr as *const u8,
                    q8 as *const u8,
                    out as *mut f32,
                    od as i32,
                    id as i32,
                    nt as i32,
                    stream,
                );
            }
        }
    }

    /// R2: the v2 weight-streaming MMVQ kernels need full 256-element
    /// super-blocks (chunk/pair indexing) — `MINFER_MMVQ_V1=1` forces the
    /// 8e kernels for A/B on shapes that would otherwise take v2.
    fn mmvq_v2(id: usize) -> bool {
        id % 256 == 0 && !std::env::var("MINFER_MMVQ_V1").map_or(false, |v| v == "1")
    }
}
