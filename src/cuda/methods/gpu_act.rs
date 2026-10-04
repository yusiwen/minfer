// on-GPU activation quantize / gather / embed (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    pub(crate) fn launch_quantize_q8_0(
        x: *const f32,
        y: *mut u8,
        dim: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_embed_rows_f16(
        w: *const u8,
        ids: *const f32,
        out: *mut f32,
        n_embd: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    pub(crate) fn launch_swiglu_f32_off(
        buf: *mut f32,
        n: i32,
        off: i32,
        stream: *mut std::ffi::c_void,
    );
    // D3-5 1a: fused-producer decode A-quantize (rms_norm/swiglu + pad40
    // epilogue; q8 bytes bit-identical to quantize_q8_0_pad40).
    pub(crate) fn launch_rms_norm_quant_pad40(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        q8: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_swiglu_quant_pad40(
        buf: *mut f32,
        q8: *mut u8,
        n: i32,
        off: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_gather_rows_f32(
        src: *const f32,
        ids: *const f32,
        out: *mut f32,
        n: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    pub(crate) fn launch_embed_rows(
        w: *const u8,
        ids: *const f32,
        out: *mut f32,
        n_embd: i32,
        nt: i32,
        type_id: i32,
        block_stride: i32,
        stream: *mut std::ffi::c_void,
    );
}

impl CudaState {
    pub fn quantize_q8_0(
        &self,
        x: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        dim: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_quantize_q8_0(x as *const f32, y as *mut u8, dim as i32, nt as i32, stream);
        }
    }

    /// 7e⑤: in-place split swiglu over one buffer (llama
    /// `ggml_swiglu_split`): buf[i] = silu(buf[i]) * buf[off + i] for
    /// i in 0..n. Used by the fused FFN decode path where the concat
    /// matmul output carries gate rows 0..nf and up rows nf..2*nf.
    pub fn swiglu_f32_off_on_gpu(&self, buf: *mut std::ffi::c_void, n: usize, off: usize) {
        let stream = self.stream();
        unsafe {
            launch_swiglu_f32_off(buf as *mut f32, n as i32, off as i32, stream);
        }
    }

    /// D3-5 1a: decode fused rms_norm + pad40 q8 epilogue — the f32 y write
    /// is bit-identical to [`Self::rms_norm`] (same kernel body) and the q8
    /// bytes are bit-identical to `quantize_q8_0_pad40` (epilogue = the
    /// standalone per-block body verbatim). Records the plane for the
    /// following decode matmul group (decode_quantize_native). Callers gate
    /// on the decode shape (n == 1, d % 32 == 0) and MINFER_NO_DECODE_A_FUSE.
    pub fn rms_norm_quant_on_gpu(
        &self,
        x: *mut std::ffi::c_void,
        w: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) {
        let q8 = Self::get_or_grow(&self.buf_q8_decode, n * (d / 32) * 40);
        let stream = self.stream();
        unsafe {
            launch_rms_norm_quant_pad40(
                x as *const f32,
                w as *const f32,
                y as *mut f32,
                q8 as *mut u8,
                d as i32,
                eps,
                n as i32,
                stream,
            );
        }
        self.record_mmq_cache_native(y as usize, n, d, q8 as usize);
    }

    /// D3-5 1a: decode fused swiglu + pad40 q8 epilogue — the in-place swiglu
    /// write is bit-identical to [`Self::swiglu_f32_off_on_gpu`] and the q8
    /// bytes to `quantize_q8_0_pad40`. n % 32 == 0 is the caller's gate
    /// (whole 32-element blocks; the fused-FFN intermediate is always a
    /// multiple of 32 on the supported models).
    pub fn swiglu_quant_off_on_gpu(&self, buf: *mut std::ffi::c_void, n: usize, off: usize) {
        let q8 = Self::get_or_grow(&self.buf_q8_decode, (n / 32) * 40);
        let stream = self.stream();
        unsafe {
            launch_swiglu_quant_pad40(buf as *mut f32, q8 as *mut u8, n as i32, off as i32, stream);
        }
        self.record_mmq_cache_native(buf as usize, 1, n, q8 as usize);
    }

    /// 7e③: generic f32 row gather on device (`get_rows`: out[t*n+i] =
    /// src[ids[t]*n+i]; ids are I32-as-f32 bit patterns on device).
    pub fn gather_rows_f32_on_gpu(
        &self,
        src: *mut std::ffi::c_void,
        ids: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n: usize,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_gather_rows_f32(
                src as *const f32,
                ids as *const f32,
                out as *mut f32,
                n as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// 7e③: embedding gather + dequantize on device. `padded_q6k` selects the
    /// padded (224-byte) block stride for Q6_K weights registered via
    /// `register_weight_q6k_padded`.
    pub fn embed_rows_on_gpu(
        &self,
        ttype: TensorType,
        wptr: *mut std::ffi::c_void,
        ids: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        n_embd: usize,
        nt: usize,
        padded_q6k: bool,
    ) -> Result<(), String> {
        let stream = self.stream();
        let (type_id, block_stride) = match ttype {
            TensorType::Q8_0 => (0i32, 34i32),
            TensorType::Q4_0 => (1, 18),
            TensorType::Q4_1 => (7, 20),
            TensorType::Q4_K => (2, 144),
            TensorType::Q5_0 => (6, 22),
            TensorType::Q5_1 => (4, 24),
            TensorType::Q5_K => (5, 176),
            TensorType::Q6_K => (3, if padded_q6k { 224 } else { 210 }),
            TensorType::F32 => {
                // f32 tok_embd is a plain gather of weight rows
                self.gather_rows_f32_on_gpu(wptr, ids, out, n_embd, nt);
                return Ok(());
            }
            // #141: f16 tok_embd rows gather + convert in one kernel; without
            // this a converted f16 GGUF would fail the all-or-nothing
            // `weights_on_cuda` embed check and run the whole model on the CPU.
            TensorType::F16 => {
                let rc = unsafe {
                    launch_embed_rows_f16(
                        wptr as *const u8,
                        ids as *const f32,
                        out as *mut f32,
                        n_embd as i32,
                        nt as i32,
                        stream,
                    )
                };
                if rc != 0 {
                    return Err(format!(
                        "cuda: f16 embed gather launch failed for n_embd={n_embd} nt={nt} \
                         (site launch:embed_rows_f16)"
                    ));
                }
                return Ok(());
            }
            other => {
                return Err(format!(
                    "cuda: no embed_rows kernel for weight type {other:?}"
                ));
            }
        };
        unsafe {
            launch_embed_rows(
                wptr as *const u8,
                ids as *const f32,
                out as *mut f32,
                n_embd as i32,
                nt as i32,
                type_id,
                block_stride,
                stream,
            );
        }
        Ok(())
    }
}
