// KV store + fused QKV epilogue (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    pub(crate) fn launch_f32_bits_to_i32(
        src: *const f32,
        dst: *mut i32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_store_kv_f32(
        src: *const f32,
        dst: *mut f32,
        nkt: i32,
        nt: i32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_store_kv_f16(
        src: *const f32,
        dst: *mut std::ffi::c_void,
        nkt: i32,
        nt: i32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    // C4 S2b: the packed store. `row_bytes` is the Q8_0 cell's byte width
    // (`KvFormat::Q8_0.row_bytes(nkt)`), so the kernel writes each 34-byte block
    // at its cell's own stride.
    pub(crate) fn launch_store_kv_q8_0(
        src: *const f32,
        dst: *mut std::ffi::c_void,
        nkt: i32,
        nt: i32,
        row_bytes: usize,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    // D3-8: fused decode QKV epilogue — bias×3 + rope×2 + store×2 in one
    // launch (CUDA port of Metal's attn_bias_rope_store, G4 FusedQKV)
    pub(crate) fn launch_attn_bias_rope_store(
        q: *mut f32,
        k: *mut f32,
        v: *mut f32,
        bias_q: *const std::ffi::c_void,
        bias_k: *const std::ffi::c_void,
        bias_v: *const std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: i32,
        nkt: i32,
        hd: i32,
        freq_base: f32,
        freq_scale: f32,
        positions: *const i32,
        cells: *const i32,
        kv_is_f16: i32,
        stream: *mut std::ffi::c_void,
    );
    // #144 item 1: the packed arm of the fused decode QKV epilogue. One thread
    // per (head, 32-element K block) and per V block, so a whole Q8_0 block is
    // quantized by its owner; `row_bytes` is the packed cell's byte width.
    pub(crate) fn launch_attn_bias_rope_store_q8_0(
        q: *mut f32,
        k: *const f32,
        v: *const f32,
        bias_q: *const std::ffi::c_void,
        bias_k: *const std::ffi::c_void,
        bias_v: *const std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: i32,
        nkt: i32,
        hd: i32,
        freq_base: f32,
        freq_scale: f32,
        positions: *const i32,
        cells: *const i32,
        row_bytes: usize,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_kv_move_rows(
        dst: *mut f32,
        src: *const f32,
        dst_row: i32,
        src_row: i32,
        rows: i32,
        elems: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
}

impl CudaState {
    /// 8d: split-K decode attention (nt == 1). `pstr` = (4 + hd + 3) & !3 —
    /// the partials' row stride keeps the oc section 16-byte aligned. The
    /// scratch must be at least ATTN_SPLITS * nh * pstr floats (see
    /// buf_attn_partial). ATTN_SPLITS must mirror the `ATTN_SPLITS` define
    /// in src/cuda/kernels/kv_store.cu (fixed grid — the graph-replay capture depends on
    /// it; idle splits write an mx=-INF/S=0 partial the combine weights to
    /// zero).
    /// C3: move `rows` rows of `elems` f32 elements inside one KV arena, from
    /// `src_row` down to `dst_row` (`dst_row <= src_row`; overlapping is fine).
    ///
    /// The kernel walks rows ascending with a barrier between them; a contract
    /// violation is an `Err` here rather than a silent no-op, so the allocator's
    /// compaction fails before it renumbers anything.
    pub fn kv_move_rows(
        &self,
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        dst_row: usize,
        src_row: usize,
        rows: usize,
        elems: usize,
    ) -> Result<(), String> {
        if rows == 0 || elems == 0 {
            return Ok(());
        }
        let rc = unsafe {
            launch_kv_move_rows(
                dst as *mut f32,
                src as *const f32,
                dst_row as i32,
                src_row as i32,
                rows as i32,
                elems as i32,
                self.stream(),
            )
        };
        if rc != 0 {
            return Err(format!(
                "cuda: kv_move_rows({dst_row}<-{src_row}, {rows} rows x {elems} elements) failed \
                 (rc {rc})"
            ));
        }
        Ok(())
    }

    /// Decode I32 graph inputs (f32::from_bits bit patterns, alloc.rs
    /// fill_input_i32) into raw int32 for the rope/store/attention kernels.
    /// Fully device-side: no host sync, capture-safe (Phase 7d).
    pub fn bits_to_i32(&self, src: *mut std::ffi::c_void, dst: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_f32_bits_to_i32(src as *const f32, dst as *mut i32, n as i32, stream);
        }
    }

    /// 8b: f16 KV cache (the engine's per-instance `KvFormat::F16`). Same
    /// trade-off as Metal: halves attention KV read bandwidth; the region stays
    /// f32-sized (the f16 view uses the first half of the bytes).
    pub fn store_kv_f16(
        &self,
        src: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        nkt: usize,
        nt: usize,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_f16(
                src as *const f32,
                dst as *mut std::ffi::c_void,
                nkt as i32,
                nt as i32,
                positions as *const i32,
                stream,
            );
        }
    }

    /// C4 S2b: the packed store — one thread per (row, 32-element block), with the
    /// CPU's own quantizer (`amax/127`, f16 scale, round-ties-even). `row_bytes` is
    /// the packed cell's byte width (`KvFormat::Q8_0.row_bytes(nkt)`), which is what
    /// makes the kernel address the cell's 34-byte blocks inside a word-padded row.
    pub fn store_kv_q8_0(
        &self,
        src: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        nkt: usize,
        nt: usize,
        row_bytes: usize,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_q8_0(
                src as *const f32,
                dst,
                nkt as i32,
                nt as i32,
                row_bytes,
                positions as *const i32,
                stream,
            );
        }
    }

    /// D3-8: fused decode QKV epilogue (G4 CUDA port of Metal's
    /// `attn_bias_rope_store`). `positions` drives RoPE (sequence-relative);
    /// `cells` is the allocator-resolved KV row for the store (C6) — the two
    /// differ when the run does not start at cell 0.
    ///
    /// `q`/`k`/`v` are POINTER-FORM section bases:
    /// the concat class passes sections of the concat matmul output [q|k|v]
    /// (nt==1), the mixed-quant class passes the three separate matmul
    /// outputs. Biases added per section, q/k roped in place (math verbatim
    /// `rope_f32`), k/v stored into the persistent regions at the same
    /// addresses as `store_kv_f32`/`store_kv_f16` (f32 or f16 per `layout`).
    ///
    /// C4 S2b: **f32/f16 only**, and a packed layout is refused here rather than
    /// silently stored per element. The fused epilogue writes one K/V element at a
    /// time — a Q8_0 block's scale needs all 32 of its elements before any of them
    /// can be quantized — so a packed cache never builds this node (the model
    /// builders' `layer_gpu` gate gains `&& !packed`) and a Q8_0 decode runs the
    /// unfused bias/rope/store chain through [`Self::store_kv_q8_0`]. Reaching this
    /// function with `KV_LAYOUT_Q8_0` is a builder bug, so it is an `Err`-shaped
    /// refusal in the caller's terms: the function is infallible in the pre-C4
    /// signature, so it panics with the reason instead of writing the wrong bytes.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_bias_rope_store(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        bias_q: *mut std::ffi::c_void,
        bias_k: *mut std::ffi::c_void,
        bias_v: *mut std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
        cells: *mut std::ffi::c_void,
        layout: i32,
    ) {
        assert!(
            layout != KV_LAYOUT_Q8_0,
            "cuda: the f32/f16 fused bias/rope/store epilogue has no packed store (it writes one \
             element at a time; a Q8_0 block needs all 32) — a packed engine must call \
             attn_bias_rope_store_q8_0 instead (issue #144)"
        );
        let stream = self.stream();
        unsafe {
            launch_attn_bias_rope_store(
                q as *mut f32,
                k as *mut f32,
                v as *mut f32,
                bias_q as *const std::ffi::c_void,
                bias_k as *const std::ffi::c_void,
                bias_v as *const std::ffi::c_void,
                kv_k,
                kv_v,
                nqt as i32,
                nkt as i32,
                hd as i32,
                freq_base,
                freq_scale,
                positions as *const i32,
                cells as *const i32,
                (layout == KV_LAYOUT_F16) as i32,
                stream,
            );
        }
    }

    /// #144 item 1: the **packed** arm of the fused decode QKV epilogue. Same
    /// bias+rope contract as [`Self::attn_bias_rope_store`], but K and V are
    /// written as whole Q8_0 blocks: one thread per (head, 32-element K block)
    /// computes the block's roped values itself and hands them to the store's own
    /// quantizer, and one thread per V block does bias + quantize. The K buffer is
    /// left roped-free (its readers are gone in both fused classes) and the bytes
    /// written to the packed regions are `add_bias`+`rope`+`store_kv_q8_0`'s
    /// verbatim.
    ///
    /// `row_bytes` is `KvFormat::Q8_0.row_bytes(nkt)`, the same byte width
    /// [`Self::store_kv_q8_0`] and `ensure_kv` use.
    #[allow(clippy::too_many_arguments)]
    pub fn attn_bias_rope_store_q8_0(
        &self,
        q: *mut std::ffi::c_void,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        bias_q: *mut std::ffi::c_void,
        bias_k: *mut std::ffi::c_void,
        bias_v: *mut std::ffi::c_void,
        kv_k: *mut std::ffi::c_void,
        kv_v: *mut std::ffi::c_void,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
        cells: *mut std::ffi::c_void,
        row_bytes: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_attn_bias_rope_store_q8_0(
                q as *mut f32,
                k as *const f32,
                v as *const f32,
                bias_q as *const std::ffi::c_void,
                bias_k as *const std::ffi::c_void,
                bias_v as *const std::ffi::c_void,
                kv_k,
                kv_v,
                nqt as i32,
                nkt as i32,
                hd as i32,
                freq_base,
                freq_scale,
                positions as *const i32,
                cells as *const i32,
                row_bytes,
                stream,
            );
        }
    }

    pub fn store_kv_f32(
        &self,
        src: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        nkt: usize,
        nt: usize,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_f32(
                src as *const f32,
                dst as *mut f32,
                nkt as i32,
                nt as i32,
                positions as *const i32,
                stream,
            );
        }
    }
}
