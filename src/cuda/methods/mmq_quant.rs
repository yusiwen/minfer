// MMQ activation quantize + caches (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // r51: producer-fused rms_norm/swiglu + pad40_t A-quantize (MINFER_MMQ_A_FUSE)
    pub(crate) fn launch_rms_norm_quant_f32_t(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        yqs: *mut u8,
        ysda: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_swiglu_quant_f32_t(
        gate: *const f32,
        up: *const f32,
        dst: *mut f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    // r52: mode-2 skip-write variants (MINFER_MMQ_A_FUSE=2) — same planes, no
    // f32 output write (window-safety proof: docs/CUDA_OPTIMIZATION.md P6 r52).
    pub(crate) fn launch_rms_norm_quant_nw_f32_t(
        x: *const f32,
        w: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_swiglu_quant_nw_f32_t(
        gate: *const f32,
        up: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_quantize_q8_0_pad40(
        x: *const f32,
        y: *mut u8,
        dim: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // P6 r34: transposed-A q8_0 quantize prepass — emits the qs plane swizzled
    // per-64-token-block ([ntb][nchunk][2048]) and the d|ssum packed scale
    // ([ntb][nchunk][256]) for mmq_raw_nb_bt_kernel's bulk staging.
    pub(crate) fn launch_quantize_q8_0_pad40_t(
        x: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
}

impl CudaState {
    /// r60: the loaders call this when they register a quantized weight that
    /// is NOT NB-BT-consumable (not q4_K/q6_K), or a 2-D F32 matmul weight:
    /// mode-2 skip-write fused producers become unsound for such mixes (see
    /// `nb_bt_only`) and degrade to mode 1 for the rest of the process.
    /// Registration happens before the first forward, so no per-node cost.
    pub fn clear_mmq_nb_bt_only(&self) {
        self.nb_bt_only
            .store(false, std::sync::atomic::Ordering::Relaxed);
    }

    /// r49: invalidate the MMQ A-quantize memoization. Called by the CUDA
    /// backend between non-MMQ nodes (conservative consecutive-window rule)
    /// and at split boundaries (cross-execution staleness).
    pub fn clear_mmq_cache(&self) {
        // #188: only this stream's entry — a foreign engine's window is not ours
        // to invalidate.
        self.mmq_cache.lock().unwrap().remove(&current_stream_key());
    }

    /// D3-5 1a: opt-out of the decode fused-producer A-quantize (A/B escape
    /// hatch): plain rms_norm/swiglu producers + unconditional standalone
    /// quantize per matmul.
    pub fn no_decode_a_fuse() -> bool {
        std::env::var("MINFER_NO_DECODE_A_FUSE").map_or(false, |v| v == "1")
    }

    /// r51: record a PRODUCER-FUSED quantize result into the r49 MmqCache.
    /// The fused rms_norm/swiglu kernel wrote the pad40_t plane for `src`
    /// (the producer's f32 output device pointer) directly, so the following
    /// matmul group's `mmq_quantize_transposed` hits the cache (same
    /// (src, nt, id) key, same buf_qa8_t/buf_sda_t pointers) and skips its
    /// own prepass launch. The cache-window rules are unchanged: any
    /// non-MatMul node clears the entry (the producer's own execution is the
    /// set point — the backend's clear runs before the node executes), and
    /// `synchronize` clears it at split boundaries, so a stale plane can
    /// never be consumed. r52: `dead_write` marks mode-2 entries (the f32
    /// src was NOT written — see the mode-2 refusal guards in
    /// `mmq_quantize_transposed`/`mmq_quantize_native`).
    fn record_mmq_cache_transposed(
        &self,
        src: usize,
        nt: usize,
        id: usize,
        qa8: usize,
        sda: usize,
        dead_write: bool,
    ) {
        let mut map = self.mmq_cache.lock().unwrap();
        let c = map.entry(current_stream_key()).or_default();
        c.active = true;
        c.key = (src, nt, id);
        c.transposed = true;
        c.dead_write = dead_write;
        c.qa8 = qa8;
        c.sda = sda;
        c.q8 = 0;
    }

    /// r51: fused rms_norm + pad40_t A-quantize (MINFER_MMQ_A_FUSE=1, full
    /// MMQ gate set). Writes the f32 rms output `y` (bit-identical to
    /// [`Self::rms_norm`]) AND the transposed q8_0 plane of `y` in ONE pass,
    /// then registers the plane in the MmqCache keyed on `y`'s device
    /// pointer — the following matmul group's `prefill_mmq` hits the cache
    /// and skips its quantize launch. `n` rows × `d` dims; the caller gates
    /// on n >= 16 (prefill_mmq territory; decode/capture paths keep the
    /// unfused pair) and d % 256 == 0 (the transposed GEMM's nchunk % 8
    /// requirement). Plane scratch sizes match `mmq_quantize_transposed`'s
    /// exactly so the hit validation's get_or_grow returns the same pointer.
    /// Err = plane OOM (caller falls back to the unfused pair).
    pub fn rms_norm_quant(
        &self,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        let nchunk = d / 32;
        let ntb = n.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: rms_norm_quant plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_rms_norm_quant_f32_t(
                x as *const f32,
                w as *const f32,
                y as *mut f32,
                qa8 as *mut u8,
                sda as *mut u8,
                d as i32,
                eps,
                n as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(y as usize, n, d, qa8 as usize, sda as usize, false);
        Ok(())
    }

    /// r51: fused swiglu + pad40_t A-quantize — the SwiGLU counterpart of
    /// [`Self::rms_norm_quant`]; `dst` (silu(gate)*up, `dim` = nf) is both
    /// the f32 node output and the quantize source. Registered for the
    /// immediately following down projection.
    pub fn swiglu_quant(
        &self,
        gate: *mut std::ffi::c_void,
        up: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        dim: usize,
        nt: usize,
    ) -> Result<(), String> {
        let nchunk = dim / 32;
        let ntb = nt.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: swiglu_quant plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_swiglu_quant_f32_t(
                gate as *const f32,
                up as *const f32,
                dst as *mut f32,
                qa8 as *mut u8,
                sda as *mut u8,
                dim as i32,
                nt as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(dst as usize, nt, dim, qa8 as usize, sda as usize, false);
        Ok(())
    }

    /// r52: mode-2 (MINFER_MMQ_A_FUSE=2) variant of [`Self::rms_norm_quant`]
    /// — SAME pad40_t plane, but the f32 output `y` is NOT written. `y` is
    /// passed only as the MmqCache key (the consuming matmuls' A pointer);
    /// its buffer is expected to be dead: consumed exclusively by the
    /// immediately following consecutive MatMul group through the plane. The
    /// entry is recorded with `dead_write = true`, so any window violation
    /// (a GEMM path that would re-quantize the unwritten buffer) refuses and
    /// fails loudly instead of reading garbage — see
    /// [`Self::mmq_a_fuse_mode`] for the dispatch conditions and
    /// docs/CUDA_OPTIMIZATION.md P6 r52 for the proof. Err = plane OOM
    /// (caller degrades to mode 1 / the unfused pair, both of which write).
    pub fn rms_norm_quant_nw(
        &self,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) -> Result<(), String> {
        let nchunk = d / 32;
        let ntb = n.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: rms_norm_quant_nw plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_rms_norm_quant_nw_f32_t(
                x as *const f32,
                w as *const f32,
                qa8 as *mut u8,
                sda as *mut u8,
                d as i32,
                eps,
                n as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(y as usize, n, d, qa8 as usize, sda as usize, true);
        Ok(())
    }

    /// r52: mode-2 counterpart of [`Self::swiglu_quant`] — same plane, the
    /// f32 `dst` is NOT written (key only). See [`Self::rms_norm_quant_nw`].
    pub fn swiglu_quant_nw(
        &self,
        gate: *mut std::ffi::c_void,
        up: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        dim: usize,
        nt: usize,
    ) -> Result<(), String> {
        let nchunk = dim / 32;
        let ntb = nt.div_ceil(64);
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);
        let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
        if qa8.is_null() || sda.is_null() {
            return Err("cuda: swiglu_quant_nw plane OOM".to_string());
        }
        let stream = self.stream();
        unsafe {
            launch_swiglu_quant_nw_f32_t(
                gate as *const f32,
                up as *const f32,
                qa8 as *mut u8,
                sda as *mut u8,
                dim as i32,
                nt as i32,
                nchunk as i32,
                ntb as i32,
                stream,
            );
        }
        self.record_mmq_cache_transposed(dst as usize, nt, dim, qa8 as usize, sda as usize, true);
        Ok(())
    }
}
