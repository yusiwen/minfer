// The `impl CudaState` families (one file per kernel family, pre-split
// `src/cuda.rs`) plus the helpers whose callers span several families.
//
// This module is the parent of every `methods/<family>.rs`, so a method that
// is private here (or a private field of `CudaState`, declared in
// `src/cuda.rs`) is visible in all of them: the split needs no `pub(super)`.

use super::*;

impl CudaState {
    /// The **context's own** stream — never a backend's bound stream. Weight
    /// registration (and any other context-level transfer) uses this so it can
    /// never enqueue into a backend's open capture window.
    fn context_stream(&self) -> *mut std::ffi::c_void {
        self.stream.lock().unwrap().0
    }

    // ─── Persistent buffer management ─────────────────────────

    /// The live persistent-scratch allocator: every `buf_*`/staging slot the
    /// MMQ, attention and f16 paths use is grown here, per stream (#188), so no
    /// `cudaMalloc` lands mid-capture. (This replaces a stale "legacy surface
    /// (7e⑦)" note.)
    fn get_or_grow(slot: &StreamScratch, need: usize) -> *mut std::ffi::c_void {
        // Issue #188: the slot is private to the **current stream**, so two
        // engines' concurrent launches never read each other's staging.
        let key = current_stream_key();
        let mut map = slot.map.lock().unwrap();
        let (ptr, size) = map
            .entry(key)
            .or_insert((CudaPtr(std::ptr::null_mut()), 0usize));
        if ptr.0.is_null() || *size < need {
            if !ptr.0.is_null() {
                unsafe {
                    cudaFree(ptr.0);
                }
            }
            let mut new_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { cudaMalloc(&mut new_ptr, need) };
            if err != 0 || new_ptr.is_null() {
                eprintln!("CUDA: OOM allocating {} bytes (cuda err {err})", need);
                *ptr = CudaPtr(std::ptr::null_mut());
                *size = 0;
                return std::ptr::null_mut();
            }
            *ptr = CudaPtr(new_ptr);
            *size = need;
            new_ptr
        } else {
            ptr.0
        }
    }

    /// r49: transposed-A helper for the MMQ quantize prepass — returns
    /// `(qa8, sda)` device pointers holding `x`'s q8_0 pad40_t-transposed
    /// quantized form, launching `quantize_q8_0_pad40_t` ONLY when the
    /// last MMQ A (consecutive, same (src,nt,id)) is not already there. A
    /// cache hit therefore skips the prepass launch entirely — the caller
    /// still feeds the returned pointers to the GEMM, which is byte-identical
    /// (quantize depends only on x/nt/id, not the weight type).
    unsafe fn mmq_quantize_transposed(
        &self,
        x: *const f32,
        id: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    ) -> (usize, usize) {
        let need_qa8 = (ntb as usize) * (id as usize / 32) * 2048;
        let need_sda = (ntb as usize) * (id as usize / 32) * 256;
        let key = (x as usize, nt as usize, id as usize);
        let mut map = self.mmq_cache.lock().unwrap();
        let cache = map.entry(current_stream_key()).or_default();
        if cache.active && cache.key == key && cache.transposed {
            // get_or_grow may have reallocated on a larger miss: validate the
            // physical pointers so a grown buffer is never reused stale.
            let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8) as usize;
            let sda = Self::get_or_grow(&self.buf_sda_t, need_sda) as usize;
            if qa8 == cache.qa8 && sda == cache.sda {
                return (qa8, sda);
            }
        }
        // r52: a mode-2 (skip-write) fused producer left the f32 src UNWRITTEN
        // and promised its consumers would hit this cache. Reaching the launch
        // path with a dead-write entry for the SAME buffer means the window
        // guarantee broke (split boundary, exotic fallback, pointer drift) —
        // re-quantizing would read the dead buffer's garbage. Refuse: the
        // callers treat (0, 0) as a failed path and prefill_mmq errors out
        // loudly instead of silently producing wrong results.
        if cache.active && cache.dead_write && cache.key.0 == x as usize {
            eprintln!(
                "minfer/cuda: MMQ A-quantize refused: mode-2 dead-write A \
                 (ptr {:#x}) missed the MmqCache window (MINFER_MMQ_A_FUSE=2 \
                 safety guard; A falls back and errors out)",
                x as usize
            );
            return (0, 0);
        }
        let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8);
        let sda = Self::get_or_grow(&self.buf_sda_t, need_sda);
        launch_quantize_q8_0_pad40_t(
            x,
            qa8 as *mut u8,
            sda as *mut u8,
            id,
            nt,
            nchunk,
            ntb,
            stream,
        );
        cache.active = true;
        cache.key = key;
        cache.transposed = true;
        cache.dead_write = false;
        cache.qa8 = qa8 as usize;
        cache.sda = sda as usize;
        cache.q8 = 0;
        (qa8 as usize, sda as usize)
    }

    /// r49: native (non-transposed) counterpart of
    /// [`Self::mmq_quantize_transposed`] — the pad40 q8_0 `q8` buffer used by
    /// the fallback NB/wide/narrow kernels. Same consecutive-window rule; the
    /// quantize output is again a pure function of (x, nt, id).
    unsafe fn mmq_quantize_native(
        &self,
        x: *const f32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> usize {
        let need = (nt as usize) * (id as usize / 32) * 40;
        let key = (x as usize, id as usize, nt as usize);
        let mut map = self.mmq_cache.lock().unwrap();
        let cache = map.entry(current_stream_key()).or_default();
        if cache.active && cache.key == key && !cache.transposed {
            let q8 = Self::get_or_grow(&self.buf_q8_prefill, need) as usize;
            if q8 == cache.q8 {
                return q8;
            }
        }
        // r52: same dead-write refusal as mmq_quantize_transposed — a mode-2
        // fused producer's f32 src was never written; re-quantizing it here
        // would read garbage. Return 0 (callers fail loudly).
        if cache.active && cache.dead_write && cache.key.0 == x as usize {
            eprintln!(
                "minfer/cuda: native A-quantize refused: mode-2 dead-write A \
                 (ptr {:#x}) (MINFER_MMQ_A_FUSE=2 safety guard)",
                x as usize
            );
            return 0;
        }
        let q8 = Self::get_or_grow(&self.buf_q8_prefill, need);
        launch_quantize_q8_0_pad40(x, q8 as *mut u8, id, nt, stream);
        cache.active = true;
        cache.key = key;
        cache.transposed = false;
        cache.dead_write = false;
        cache.qa8 = 0;
        cache.sda = 0;
        cache.q8 = q8 as usize;
        q8 as usize
    }

    /// D3-5 1a: decode-side native pad40 A-quantize with r49-style
    /// consecutive-consumer memoization. The fused decode producers
    /// ([`Self::rms_norm_quant_on_gpu`] / [`Self::swiglu_quant_off_on_gpu`])
    /// record their pad40 plane here keyed on the producer's f32 output
    /// pointer; the decode matmul group that consumes it hits the cache and
    /// skips the standalone `quantize_q8_0_pad40` launch. Same window-safety
    /// contract as the prefill MmqCache: the plane is a pure function of
    /// (src, nt, id), any non-(MatMul|FusedFFN) node clears the entry, and
    /// `synchronize` clears it at execution boundaries. The hit path
    /// re-validates the physical pointer (a `get_or_grow` between record and
    /// consult must not have moved the plane — a grown-over entry falls back
    /// to the standalone launch, which re-quantizes from the live f32 src).
    /// `MINFER_NO_DECODE_A_FUSE=1` skips the consult AND the record, restoring
    /// the exact pre-D3-5 per-matmul standalone quantize (A/B gate).
    fn decode_quantize_native(&self, x: *const f32, id: usize, nt: usize) -> *mut u8 {
        let need = nt * (id / 32) * 40;
        let key = (x as usize, nt, id);
        if !Self::no_decode_a_fuse() {
            let map = self.mmq_cache.lock().unwrap();
            if let Some(cache) = map.get(&current_stream_key()) {
                if cache.active && !cache.transposed && !cache.dead_write && cache.key == key {
                    let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as usize;
                    if q8 == cache.q8 {
                        return q8 as *mut u8;
                    }
                }
            }
        }
        let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as *mut u8;
        let stream = self.stream();
        unsafe {
            launch_quantize_q8_0_pad40(x, q8, id as i32, nt as i32, stream);
        }
        if !Self::no_decode_a_fuse() {
            self.record_mmq_cache_native(key.0, nt, id, q8 as usize);
        }
        q8
    }

    /// D3-5 1a: record a fused-producer NATIVE (pad40, non-transposed) plane
    /// into the MmqCache (decode counterpart of
    /// [`Self::record_mmq_cache_transposed`]).
    fn record_mmq_cache_native(&self, src: usize, nt: usize, id: usize, q8: usize) {
        let mut map = self.mmq_cache.lock().unwrap();
        let c = map.entry(current_stream_key()).or_default();
        c.active = true;
        c.key = (src, nt, id);
        c.transposed = false;
        c.dead_write = false;
        c.qa8 = 0;
        c.sda = 0;
        c.q8 = q8;
    }

    /// 8m: prefill GEMM — dequant the weight to f16 scratch once, convert
    /// activations to f16, then one tensor-core GEMM (see the dispatch-gate
    /// comment in `matmul_f32_ptr_layout`). Caller guarantees nt >= 16 and
    /// id % 32 == 0 for the supported quant types.
    fn prefill_gemm_f16(
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
        // 8p: default is the cp.async f16 GEMM over a PERSISTENT per-weight
        // f16 copy (dequant runs once per weight per process — the per-call
        // 288 ms dequant is gone). MINFER_FUSED_B=1 opts into the
        // dequant-in-GEMM kernel instead (memory-lean: no f16 weight cache,
        // but re-dequantizes every B tile per nt sweep — slower on large
        // nt). Both require id % 256 == 0 for alignment/coverage.
        let fused = id % 256 == 0 && Self::fused_b_on();
        self.prefill_gemm_f16_inner(wptr, ttype, x, out, od, id, nt, padded_q6k, fused)
    }

    /// 8e follow-up: `MINFER_NO_KQ_MMVQ=1` forces the K-quant decode matmuls
    /// back onto the f32 kernels (A/B escape hatch, MINFER_NO_FUSE_* style).
    fn no_kq_mmvq() -> bool {
        std::env::var("MINFER_NO_KQ_MMVQ").map_or(false, |v| v == "1")
    }

    // doc 103: per-type opt-outs for the new q4_0/q8_0 MMVQ decode paths
    // (A/B switches; the f32-activation kernels remain the fallback).
    fn no_q40_mmvq() -> bool {
        std::env::var("MINFER_NO_Q40_MMVQ").map_or(false, |v| v == "1")
    }

    fn no_q80_mmvq() -> bool {
        std::env::var("MINFER_NO_Q80_MMVQ").map_or(false, |v| v == "1")
    }

    fn no_q80_p32() -> bool {
        std::env::var("MINFER_NO_Q80_P32").map_or(false, |v| v == "1")
    }

    /// 8m: force the legacy per-type prefill kernels (A/B escape hatch).
    // 8p: opt-in dequant-in-GEMM (memory-lean alternative to the f16 cache).
    fn fused_b_on() -> bool {
        std::env::var("MINFER_FUSED_B").map_or(false, |v| v == "1")
    }

    // 8p: disable the persistent per-weight f16 dequant cache.
    fn no_w16cache() -> bool {
        std::env::var("MINFER_NO_W16CACHE").map_or(false, |v| v == "1")
    }

    fn no_prefill_gemm() -> bool {
        std::env::var("MINFER_NO_PREFILL_GEMM").map_or(false, |v| v == "1")
    }

    /// 8n: force the legacy per-token prefill attention kernel (A/B escape).
    fn no_fa_prefill() -> bool {
        std::env::var("MINFER_NO_FA_PREFILL").map_or(false, |v| v == "1")
    }
}

mod accounting;
mod attention;
mod buffers;
mod capture;
mod copy;
mod dispatch;
mod elementwise;
mod events;
mod gpu_act;
mod init;
mod kvstore;
mod mmq_quant;
mod mmvq;
mod policy;
mod prefill_f16;
mod prefill_mmq;
mod stream;
mod weights;

#[cfg(test)]
pub(crate) use self::attention::*;
#[cfg(test)]
pub(crate) use self::dispatch::*;
#[cfg(test)]
pub(crate) use self::elementwise::*;
#[cfg(test)]
pub(crate) use self::gpu_act::*;
#[cfg(test)]
pub(crate) use self::init::*;
#[cfg(test)]
pub(crate) use self::kvstore::*;
pub(crate) use self::mmq_quant::*;
#[cfg(test)]
pub(crate) use self::mmvq::*;
#[cfg(test)]
pub(crate) use self::prefill_f16::*;
#[cfg(test)]
pub(crate) use self::prefill_mmq::*;
