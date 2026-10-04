// f16 GEMM + w16 cache (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // 8m: prefill dequant-to-f16 + wmma HGEMM. The f16 pointers cross the
    // boundary as c_void (Rust has no __half); the type_id mapping is
    // documented at launch_dequant_f16 in src/cuda/kernels/gemm_wmma.cu.
    pub(crate) fn launch_dequant_f16(
        type_id: i32,
        w: *const u8,
        out: *mut std::ffi::c_void,
        od: i32,
        id: i32,
        block_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_convert_f16(
        x: *const f32,
        out: *mut std::ffi::c_void,
        n: i64,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_gemm_f16(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        af32: bool,
    ) -> i32;
    // #218: the removed eager sweep's (`gemm_prefill_smem_init`) introspection
    // lives in the `#[cfg(test)] extern "C"` block below, so no test-only
    // declaration is carried by a non-test build. The `checked`/`skipped`
    // counters did not come back with #223's pre-warm.
    //
    // #145/#147/#162 test introspection (`cuda_test_latch_oversized_smem`,
    // `minfer_site_fail_*`, `minfer_site_hist_{len,site,name,msg}`) moved to
    // `cuda::tests` by #239: their only callers are the `cuda::*_tests` modules.
    // P6: A arrives as f32 activations; the GEMM converts on stage — the
    // separate convert_f32_f16 pass disappears for every prefill matmul.
    // #147: the same shape for the af32 path — 1 = launched and accepted; 0 =
    // the opt-in or the launch was refused (named at the site), which the
    // caller turns into an `Err`.
    pub(crate) fn launch_gemm_f32a(
        a: *const f32,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    // 8p: fused dequant-in-GEMM — B tiles dequantize raw quantized bytes
    // in-register (no f16 weight scratch round trip). type_id mapping as in
    // launch_dequant_f16; q6_stride = 210 raw / 224 padded (only Q6_K reads
    // it). Requires id % 256 == 0 (host gate).
    pub(crate) fn launch_gemm_qb_nt(
        a: *const std::ffi::c_void,
        w: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        type_id: i32,
        q6_stride: i32,
        stream: *mut std::ffi::c_void,
    );
}

impl CudaState {
    /// 8p: opt the process into the f16 weight cache (loader-side, for
    /// models big enough to amortize it — see W16_ENABLE_BYTES).
    pub fn enable_w16_cache(&self) {
        self.w16_enabled
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    /// 8p: warm the persistent f16 cache for one registered matmul weight
    /// at LOAD time (the loader has the tensor metadata; the lazy path
    /// would otherwise put ~8.6 GB of cudaMalloc + the full dequant inside
    /// the first — timed — prefill). No-op for non-quant types and for
    /// rows not covered by the fused-GEMM alignment gate.
    pub fn warm_w16(&self, name: &str, t: &crate::tensor::Tensor) -> bool {
        let type_id = match t.ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            _ => return false,
        };
        // GGUF convention: metadata [in, out] → od = shape[1], id = shape[0].
        let od = t.shape[1] as usize;
        let id = t.shape[0] as usize;
        if od == 0 || id == 0 || id % 256 != 0 {
            return false;
        }
        let padded_q6k =
            t.ttype == TensorType::Q6_K && self.padded_weights.lock().unwrap().contains_key(name);
        let block_stride = if padded_q6k { 224 } else { 210 };
        let Some(wptr) = self.get_weight_ptr(name) else {
            return false;
        };
        self.w16_get(wptr, type_id, od, id, block_stride).is_some()
    }

    /// Persistent f16 copy of a registered quantized weight (8p). Returns
    /// None when the allocation fails (caller falls back to the per-call
    /// scratch) or when MINFER_NO_W16CACHE=1.
    fn w16_get(
        &self,
        wptr: *mut std::ffi::c_void,
        type_id: i32,
        od: usize,
        id: usize,
        block_stride: i32,
    ) -> Option<*mut std::ffi::c_void> {
        if Self::no_w16cache() || !self.w16_enabled.load(std::sync::atomic::Ordering::Relaxed) {
            return None;
        }
        let bytes = od * id * 2;
        if let Some((p, sz)) = self.w16_cache.lock().unwrap().get(&(wptr as usize)) {
            if *sz == bytes {
                return Some(p.0);
            }
        }
        // Memory-pressure valve: the cache doubles the resident weight
        // footprint; skip it (caller falls back to the per-call scratch =
        // pre-8p behavior) unless free memory comfortably covers the copy —
        // the test suite keeps several loaded models resident (registry
        // entries are never freed), and +1 GB caches per model exhausted the
        // ~23 GB free pool and broke later models' weight uploads.
        {
            let (mut free_mem, mut total_mem) = (0usize, 0usize);
            let rc = unsafe { cudaMemGetInfo(&mut free_mem, &mut total_mem) };
            // A failed query is *not* "there is room" (issue #122): the pre-#122
            // `rc == 0 && …` shape let a failure fall through to the allocation,
            // which is the fail-open direction for an optional cache. Take the
            // documented conservative branch instead — no measurement, no cache.
            if rc != 0 || free_mem < 2 * bytes + (4usize << 30) {
                return None;
            }
        }
        let ptr = Self::cuda_malloc(bytes);
        if ptr.is_null() {
            eprintln!("CUDA: w16 cache OOM allocating {bytes} bytes");
            return None;
        }
        unsafe {
            launch_dequant_f16(
                type_id,
                wptr as *const u8,
                ptr,
                od as i32,
                id as i32,
                block_stride,
                self.stream(),
            );
        }
        self.w16_cache
            .lock()
            .unwrap()
            .insert(wptr as usize, (CudaPtr(ptr), bytes));
        Some(ptr)
    }

    /// Prefill GEMM with an explicit fused/legacy switch (test entry point
    /// for the bit-parity check between the two paths).
    pub(crate) fn prefill_gemm_f16_inner(
        &self,
        wptr: *mut std::ffi::c_void,
        ttype: TensorType,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
        padded_q6k: bool,
        fused: bool,
    ) -> Result<(), String> {
        let stream = self.stream();
        let type_id = match ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            other => return Err(format!("cuda: prefill GEMM got unsupported type {other:?}")),
        };
        // Q6_K reads whichever layout the weight was registered with
        // (224-byte padded 7e② repack or raw 210); all others are raw.
        let block_stride: i32 = if ttype == TensorType::Q6_K && padded_q6k {
            224
        } else {
            210
        };
        if fused {
            // 8p: A converts to f16 once (the convert pass stays); the B
            // side dequantizes raw quantized bytes inside the GEMM — no
            // buf_f16_w round trip, no launch_dequant_f16.
            let x16 = Self::get_or_grow(&self.buf_f16_x, nt * id * 2);
            unsafe {
                launch_convert_f16(x as *const f32, x16, (nt * id) as i64, stream);
                launch_gemm_qb_nt(
                    x16,
                    wptr as *const u8,
                    out as *mut f32,
                    nt as i32,
                    od as i32,
                    id as i32,
                    type_id,
                    block_stride,
                    stream,
                );
            }
            return Ok(());
        }
        let w16 = match self.w16_get(wptr, type_id, od, id, block_stride) {
            Some(p) => p, // persistent copy, dequant already done
            None => {
                let w16 = Self::get_or_grow(&self.buf_f16_w, od * id * 2);
                unsafe {
                    launch_dequant_f16(
                        type_id,
                        wptr as *const u8,
                        w16,
                        od as i32,
                        id as i32,
                        block_stride,
                        stream,
                    );
                }
                w16
            }
        };
        // P6 negative result: in-kernel f32->f16 A conversion measured
        // ~8% SLOWER end-to-end (2172-2177 vs 2365 tok/s) — the A stage
        // loses cp.async and its synchronous 32B loads stall every k-tile,
        // costing more than the 56 ms convert pass it removes. Kept behind
        // MINFER_GEMM_A32=1 for the smem-mirror variant follow-up (cp.async
        // the f32 tile to smem, convert smem->smem — no global round trip).
        let af32 = std::env::var("MINFER_GEMM_A32")
            .map(|v| v == "1")
            .unwrap_or(false);
        if af32 {
            let launched = unsafe {
                std::env::var("MINFER_A32_DEBUG")
                    .is_ok()
                    .then(|| eprintln!("minfer/cuda: af32 gemm nt={nt} od={od} id={id}"));
                launch_gemm_f32a(
                    x as *const f32,
                    w16,
                    out as *mut f32,
                    nt as i32,
                    od as i32,
                    id as i32,
                    stream,
                )
            };
            if launched == 0 {
                // #147: a refused/failed prefill GEMM launch is an error, not a
                // silent pass over an unwritten output. The site named it.
                return Err(
                    "cuda: prefill GEMM (af32): launch_gemm_f32a refused the launch \
                            (see the named CUDA error on stderr); no kernel ran"
                        .to_string(),
                );
            }
            return Ok(());
        }
        let x16 = Self::get_or_grow(&self.buf_f16_x, nt * id * 2);
        let launched = unsafe {
            launch_convert_f16(x as *const f32, x16, (nt * id) as i64, stream);
            launch_gemm_f16(
                x16,
                w16,
                out as *mut f32,
                nt as i32,
                od as i32,
                id as i32,
                stream,
                false,
            )
        };
        if launched == 0 {
            // #147: same contract as the af32 arm above.
            return Err(
                "cuda: prefill GEMM (f16): launch_gemm_f16 refused the launch \
                        (see the named CUDA error on stderr); no kernel ran"
                    .to_string(),
            );
        }
        Ok(())
    }
}
