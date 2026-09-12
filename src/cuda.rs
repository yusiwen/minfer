// CUDA (NVIDIA GPU) backend for x86-64 Linux/Windows.
//
// The graph backend lives in graph/cuda_backend.rs (Phase 7a-7d); this module
// hosts the CudaState singleton it wraps: device probes, the weight registry,
// streams, per-op kernel entry points and the CUDA Graph capture API. The
// legacy layer_gpu/`init_kv_cache` pre-alloc path is no longer driven by
// main() (the graph allocator owns KV regions since Phase 7c); the legacy
// surface carries targeted #![allow]s at its use sites instead of a
// module-wide opt-out (7e⑦).

use crate::block::Q8B;
use crate::tensor::{Tensor, TensorType};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Wrapper to make `*mut c_void` Send+Sync for use in Mutex.
#[derive(Clone, Copy)]
struct CudaPtr(*mut std::ffi::c_void);
unsafe impl Send for CudaPtr {}
unsafe impl Sync for CudaPtr {}

// ─── FFI declarations for CUDA runtime API ────────────────────

// Sized buffer for cudaGetDeviceProperties (avoids fragile field-by-field layout).
// Only the first 256 bytes (device name) are read; extra padding handles any CUDA version.
#[repr(C)]
struct CudaDevicePropBuf([u8; 4096]);

extern "C" {
    fn dlopen(filename: *const std::ffi::c_char, flag: std::ffi::c_int) -> *mut std::ffi::c_void;
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaMemcpy(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
    ) -> i32;
    fn cudaHostAlloc(ptr: *mut *mut std::ffi::c_void, size: usize, flags: i32) -> i32;
    fn cudaFreeHost(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMemcpyAsync(
        dst: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        count: usize,
        kind: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetLastError() -> i32;
    fn cudaDeviceGetAttribute(value: *mut i32, attr: i32, device: i32) -> i32;
    fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
    fn cudaGetDeviceProperties(prop: *mut CudaDevicePropBuf, device: i32) -> i32;
    // CUDA Graph APIs
    fn cudaStreamBeginCapture(stream: *mut std::ffi::c_void, mode: i32) -> i32;
    fn cudaStreamEndCapture(
        stream: *mut std::ffi::c_void,
        graph: *mut *mut std::ffi::c_void,
    ) -> i32;
    fn cudaGraphInstantiate(
        exec: *mut *mut std::ffi::c_void,
        graph: *mut std::ffi::c_void,
        error_node: *mut std::ffi::c_void,
        log_buf: *mut u8,
        buf_size: usize,
    ) -> i32;
    fn cudaGraphLaunch(exec: *mut std::ffi::c_void, stream: *mut std::ffi::c_void) -> i32;
    fn cudaGraphDestroy(graph: *mut std::ffi::c_void) -> i32;
}

// cudaMemcpyKind values (https://docs.nvidia.com/cuda/runtime-api/group__CUDART__TYPES.html)
const CUDA_MEMCPY_HOST_TO_DEVICE: i32 = 1;
const CUDA_MEMCPY_DEVICE_TO_HOST: i32 = 2;
const CUDA_MEMCPY_DEVICE_TO_DEVICE: i32 = 3;

const CUDA_DEV_ATTR_COMPUTE_MAJOR: i32 = 75;
const CUDA_DEV_ATTR_COMPUTE_MINOR: i32 = 76;
const CUDA_DEV_ATTR_MULTIPROC_COUNT: i32 = 16;

// Legacy layer_gpu debug tracing (7e⑦): the graph path syncs via
// `CudaState::sync()`; MINFER_CUDA_DEBUG tracing stays for the legacy
// surface only (debug_sync in the impl below).
static CUDA_DEBUG: OnceLock<bool> = OnceLock::new();
fn cuda_debug_enabled() -> bool {
    *CUDA_DEBUG.get_or_init(|| std::env::var("MINFER_CUDA_DEBUG").is_ok())
}

// ─── FFI declarations for kernel launch wrappers ───────────

extern "C" {
    fn launch_q4_0_q8_0_matmul(
        weights: *const u8,
        acts: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q8_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_1_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_f32_matmul_padded(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_quantize_q8_0(
        x: *const f32,
        y: *mut u8,
        dim: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_f32_f32_matmul(
        w: *const f32,
        x: *const f32,
        out: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_swiglu_f32_off(buf: *mut f32, n: i32, off: i32, stream: *mut std::ffi::c_void);
    // D3-5 1a: fused-producer decode A-quantize (rms_norm/swiglu + pad40
    // epilogue; q8 bytes bit-identical to quantize_q8_0_pad40).
    fn launch_rms_norm_quant_pad40(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        q8: *mut u8,
        d: i32,
        eps: f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_swiglu_quant_pad40(
        buf: *mut f32,
        q8: *mut u8,
        n: i32,
        off: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gather_rows_f32(
        src: *const f32,
        ids: *const f32,
        out: *mut f32,
        n: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_embed_rows(
        w: *const u8,
        ids: *const f32,
        out: *mut f32,
        n_embd: i32,
        nt: i32,
        type_id: i32,
        block_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_rms_norm_f32(
        x: *const f32,
        w: *const f32,
        y: *mut f32,
        d: i32,
        eps: f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_add_bias_f32(
        y: *mut f32,
        b: *const f32,
        d: i32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_add_f32(
        x: *const f32,
        y: *const f32,
        z: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_mul_f32(
        x: *const f32,
        y: *const f32,
        z: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_silu_f32(y: *mut f32, n: i32, stream: *mut std::ffi::c_void);
    fn launch_swiglu_f32(
        gate: *const f32,
        up: *const f32,
        dst: *mut f32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    // r51: producer-fused rms_norm/swiglu + pad40_t A-quantize (MINFER_MMQ_A_FUSE)
    fn launch_rms_norm_quant_f32_t(
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
    fn launch_swiglu_quant_f32_t(
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
    fn launch_rms_norm_quant_nw_f32_t(
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
    fn launch_swiglu_quant_nw_f32_t(
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
    fn launch_f32_bits_to_i32(
        src: *const f32,
        dst: *mut i32,
        n: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_rope_f32(
        x: *mut f32,
        n_head: i32,
        n_dims: i32,
        nt: i32,
        freq_base: f32,
        freq_scale: f32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_store_kv_f32(
        src: *const f32,
        dst: *mut f32,
        nkt: i32,
        nt: i32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_f32(
        q: *const f32,
        k: *const f32,
        v: *const f32,
        o: *mut f32,
        positions: *const i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // 8m: prefill dequant-to-f16 + wmma HGEMM. The f16 pointers cross the
    // boundary as c_void (Rust has no __half); the type_id mapping is
    // documented at launch_dequant_f16 in cuda_kernels.cu.
    fn launch_dequant_f16(
        type_id: i32,
        w: *const u8,
        out: *mut std::ffi::c_void,
        od: i32,
        id: i32,
        block_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_convert_f16(
        x: *const f32,
        out: *mut std::ffi::c_void,
        n: i64,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gemm_f16(
        a: *const std::ffi::c_void,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        af32: bool,
    );
    // P6: A arrives as f32 activations; the GEMM converts on stage — the
    // separate convert_f32_f16 pass disappears for every prefill matmul.
    fn gemm_prefill_smem_init();
    fn launch_gemm_f32a(
        a: *const f32,
        b: *const std::ffi::c_void,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
    );
    // 8p: fused dequant-in-GEMM — B tiles dequantize raw quantized bytes
    // in-register (no f16 weight scratch round trip). type_id mapping as in
    // launch_dequant_f16; q6_stride = 210 raw / 224 padded (only Q6_K reads
    // it). Requires id % 256 == 0 (host gate).
    fn launch_gemm_qb_nt(
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
    // R1: int8 MMQ prefill GEMM — q8_0-quantized activations (pad40 blocks
    // with the per-block int sum at offset 36) × raw quantized weights, tiled
    // mma.m16n8k32/m16n8k16 (s8) with per-k-block scale rescale. type_id as
    // in launch_dequant_f16; q6_stride = 210 raw / 224 padded (Q6_K only).
    // Requires id % 32 == 0 and sm_80+ (int8 mma; sm_75 falls back).
    fn launch_mmq_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        q6_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    // P6: raw-byte staging MMQ (q4_K, whole 256-k super-blocks).
    fn launch_mmq_raw_wide_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // P6 Direction-A: NB raw-nibble MMQ (q4_K, KD=8 native, 64x128, 2 blk/SM).
    // Returns 1 when it ran (KD=8), 0 on clean fallback (KD!=8 / smem / regs).
    fn launch_mmq_raw_nb_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // P6 r34: NB kernel whose A staging is bulk LDG->STS over the
    // PRE-TRANSPOSED qa8/sda buffers (MINFER_MMQ_A_TRANSPOSE=1). Returns 1 on
    // KD=8, 0 on clean fallback (KD!=8 / null transposed buffers / smem).
    // r59 rider: kernel module pre-load (cudaFuncGetAttributes over the
    // MMQ/FA/fused launch set) — see CudaState::prewarm_prefill.
    fn minfer_prewarm_kernels();
    // r59: `w_dsc` selects the DSC=true instantiation (registration-time
    // W_dsc f32-pair plane; null = in-kernel scalar decode, DSC=false).
    fn launch_mmq_raw_nb_bt_nt(
        type_id: i32,
        w: *const u8,
        w_dsc: *const u8,
        qa8g: *const u8,
        sdag: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        nchunk: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    // P6 r38: q6_K BT kernel — the same bulk-LDG->STS A staging as r34, but the
    // B weight is EXPANDED to centered int8 (256 B/row) in staging and the mma is
    // m16n8k16 (KSPLIT=2) with a per-16-sub dsc rescale. r53: `w_exp` selects
    // the pre-expanded-B cp.async instantiation (null = r41 in-kernel expand).
    // Returns 1 on KD=8, 0 on clean fallback.
    fn launch_mmq_raw_nb_bt_q6k_nt(
        type_id: i32,
        w: *const u8,
        w_exp: *const u8,
        w_dsc: *const u8,
        qa8g: *const u8,
        sdag: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        nchunk: i32,
        bstride: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    ) -> i32;
    fn launch_mmq_raw_nt(
        type_id: i32,
        w: *const u8,
        q8: *const u8,
        c: *mut f32,
        nt: i32,
        od: i32,
        id: i32,
        stream: *mut std::ffi::c_void,
        kd: i32,
    );
    // 8n: FA-style prefill attention. Returns -1 when the >48KB dynamic
    // shared-memory opt-in fails (then Rust falls back to the legacy kernel).
    fn launch_fa_prefill_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        positions: *const i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    ) -> i32;
    fn launch_store_kv_f16(
        src: *const f32,
        dst: *mut std::ffi::c_void,
        nkt: i32,
        nt: i32,
        positions: *const i32,
        stream: *mut std::ffi::c_void,
    );
    // D3-8: fused decode QKV epilogue — bias×3 + rope×2 + store×2 in one
    // launch (CUDA port of Metal's attn_bias_rope_store, G4 FusedQKV)
    fn launch_attn_bias_rope_store(
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
        kv_is_f16: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_f32_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        positions: *const i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_split_f16kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        positions: *const i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_quantize_q8_0_pad40(
        x: *const f32,
        y: *mut u8,
        dim: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    // P6 r34: transposed-A q8_0 quantize prepass — emits the qs plane swizzled
    // per-64-token-block ([ntb][nchunk][2048]) and the d|ssum packed scale
    // ([ntb][nchunk][256]) for mmq_raw_nb_bt_kernel's bulk staging.
    fn launch_quantize_q8_0_pad40_t(
        x: *const f32,
        yqs: *mut u8,
        ysda: *mut u8,
        dim: i32,
        nt: i32,
        nchunk: i32,
        ntb: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq(
        weights: *const u8,
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
    fn launch_q4_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q4_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq_v2_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_multi(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        blk_stride: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_v2_multi(
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
    fn launch_q4_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_v2(
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
    fn launch_q6_k_q8_mmvq_v2_pf(
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
    fn launch_q6_k_q8_mmvq_v2_pf_dpl(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        nbe: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q6_k_q8_mmvq_v2_dpl(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        nbe: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_q8_mmvq_v2(
        weights: *const u8,
        acts8: *const u8,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_1_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_0_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_q5_k_f32_matmul(
        weights: *const u8,
        acts: *const f32,
        output: *mut f32,
        od: i32,
        id: i32,
        nt: i32,
        stream: *mut std::ffi::c_void,
    );
    fn launch_gqa_attn_split_f32kv(
        q: *const f32,
        k: *const std::ffi::c_void,
        v: *const std::ffi::c_void,
        o: *mut f32,
        partial: *mut f32,
        positions: *const i32,
        nh: i32,
        nk: i32,
        hd: i32,
        scale: f32,
        pstr: i32,
        stream: *mut std::ffi::c_void,
    );
}

// ─── CudaState singleton ───────────────────────────────────────

static CUDA: OnceLock<Option<CudaState>> = OnceLock::new();

/// A small per-thread reentrant lock guarding model weight registration.
///
/// std's `Mutex` is not reentrant, but the natural usage nests: a test holds
/// the lock across several forwards while `load_model` (called inside) takes
/// it again to register weights. This guard tracks the owning thread plus a
/// depth counter — recursive acquisition on the same thread is free; other
/// threads block until the outermost guard drops.
pub struct ModelLoadGuard {
    _inner: Option<MutexGuard<'static, ()>>,
}

static MODEL_LOAD_MUTEX: Mutex<()> = Mutex::new(());
static MODEL_LOAD_OWNER: AtomicU64 = AtomicU64::new(0);
thread_local! {
    static MODEL_LOAD_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

impl ModelLoadGuard {
    fn acquire() -> Self {
        let tid = std::thread::current().id();
        // ThreadId is opaque; use its Debug value as a stable discriminator
        // for the owner slot (collision-free within a process).
        let tid_hash = {
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            tid.hash(&mut h);
            h.finish()
        };
        let depth = MODEL_LOAD_DEPTH.with(|d| d.get());
        if depth > 0 && MODEL_LOAD_OWNER.load(Ordering::Acquire) == tid_hash {
            MODEL_LOAD_DEPTH.with(|d| d.set(depth + 1));
            return ModelLoadGuard { _inner: None };
        }
        // Poison-immune: a panicking holder (test assertion) must not wedge
        // every later loader — the registry has no inconsistent state to
        // recover from (entries are atomically replaced).
        let inner = MODEL_LOAD_MUTEX.lock().unwrap_or_else(|e| e.into_inner());
        MODEL_LOAD_OWNER.store(tid_hash, Ordering::Release);
        MODEL_LOAD_DEPTH.with(|d| d.set(1));
        ModelLoadGuard {
            _inner: Some(inner),
        }
    }
}

impl Drop for ModelLoadGuard {
    fn drop(&mut self) {
        let depth = MODEL_LOAD_DEPTH.with(|d| d.get());
        if depth > 1 {
            MODEL_LOAD_DEPTH.with(|d| d.set(depth - 1));
            return;
        }
        if depth == 1 {
            MODEL_LOAD_OWNER.store(0, Ordering::Release);
            MODEL_LOAD_DEPTH.with(|d| d.set(0));
        }
    }
}

/// Pinned-host staging slots for async H2D input fills (7e⑥). Pageable
/// `cudaMemcpy` forces the driver to bounce through an internal pinned
/// buffer AND blocks until the copy lands; copying into our own pinned
/// slot + `cudaMemcpyAsync` returns immediately and lets the copy overlap
/// the subsequent kernel launches on the stream. Slots form a ring: when
/// the ring wraps, one stream sync retires every in-flight copy before a
/// slot is reused (in practice input fills are KB-scale and the ring
/// never wraps within a step).
struct PinnedPool {
    ptrs: Vec<*mut u8>,
    slot_bytes: usize,
    next: usize,
}
impl Drop for PinnedPool {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            unsafe { cudaFreeHost(p as *mut std::ffi::c_void) };
        }
    }
}

unsafe impl Send for PinnedPool {}
unsafe impl Sync for PinnedPool {}

/// R3-A2: single grow-on-demand pinned staging buffer for device→host
/// readbacks (the graph logits path runs this once per decode step).
struct PinnedBuf {
    ptr: *mut u8,
    bytes: usize,
}
impl Drop for PinnedBuf {
    fn drop(&mut self) {
        unsafe { cudaFreeHost(self.ptr as *mut std::ffi::c_void) };
    }
}
unsafe impl Send for PinnedBuf {}
unsafe impl Sync for PinnedBuf {}

/// Viz/trace capture staging (scheduler's per-node `copy_to_host` full-stream
/// sync replaced by batched async D2H). The scheduler enqueues one async copy
/// into this pinned buffer immediately after each captured node's launch —
/// stream order preserves the value even though the pool recycles buffers
/// intra-split — and drains everything with ONE sync at the split boundary.
/// Bounded: a node larger than the remaining capacity is refused (the caller
/// falls back to the per-node sync copy); growth only happens between drains
/// (reallocating under undrained pending copies would lose them).
pub struct CaptureStaging {
    ptr: *mut u8,
    bytes: usize,
    used: usize,
    /// (offset, bytes) per enqueued copy, in enqueue order.
    pending: Vec<(usize, usize)>,
}

/// 128 MB of pinned host memory is the ceiling for one split's captured node
/// outputs; 7B decode steps (~150 nodes × ≤76 KB) fit ~20× over, prefill's
/// GB-scale FFN tensors fall back to the sync path.
const CAPTURE_STAGING_MAX: usize = 128 << 20;

impl CaptureStaging {
    pub const fn new() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            bytes: 0,
            used: 0,
            pending: Vec::new(),
        }
    }

    /// Queue an async D2H of `bytes` from device `src` into the staging.
    /// `false` = refused (alloc failure or no room): the caller must fall
    /// back to the per-node sync copy for this buffer.
    pub fn queue(
        &mut self,
        stream: *mut std::ffi::c_void,
        src: *const std::ffi::c_void,
        bytes: usize,
    ) -> bool {
        if bytes == 0 || self.used + bytes > CAPTURE_STAGING_MAX {
            return false;
        }
        if self.bytes < self.used + bytes {
            // grow only between drains (pending must be empty here — a full
            // buffer with pending entries was refused above)
            debug_assert!(self.pending.is_empty());
            if !self.pending.is_empty() {
                return false;
            }
            let need = (self.used + bytes).max(8 << 20);
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            if unsafe { cudaHostAlloc(&mut p, need, 0) } != 0 {
                return false;
            }
            if !self.ptr.is_null() {
                unsafe { cudaFreeHost(self.ptr as *mut std::ffi::c_void) };
            }
            self.ptr = p as *mut u8;
            self.bytes = need;
            self.used = 0;
        }
        let err = unsafe {
            cudaMemcpyAsync(
                self.ptr.add(self.used) as *mut std::ffi::c_void,
                src,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
                stream,
            )
        };
        if err != 0 {
            return false;
        }
        self.pending.push((self.used, bytes));
        self.used += bytes;
        true
    }

    /// One stream sync, then hand out the queued copies (enqueue order).
    /// Empty when nothing was queued.
    pub fn drain(&mut self, state: &CudaState) -> Vec<Vec<f32>> {
        if self.pending.is_empty() {
            return Vec::new();
        }
        state.sync();
        let mut out = Vec::with_capacity(self.pending.len());
        for &(off, bytes) in &self.pending {
            let slice =
                unsafe { std::slice::from_raw_parts(self.ptr.add(off) as *const f32, bytes / 4) };
            out.push(slice.to_vec());
        }
        self.pending.clear();
        self.used = 0;
        out
    }
}

impl Drop for CaptureStaging {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe { cudaFreeHost(self.ptr as *mut std::ffi::c_void) };
            self.ptr = std::ptr::null_mut();
        }
    }
}

unsafe impl Send for CaptureStaging {}
unsafe impl Sync for CaptureStaging {}

/// r49: consecutive-window memoization of the MMQ A-quantize prepass.
///
/// The three attention GEMMs (q/k/v) all consume the SAME `normed` activation
/// and the two FFN GEMMs (gate/up) consume the same `normed2`; the MMQ path
/// re-quantizes (transpose) that A per matmul. This cache holds the quantized
/// (transposed) form of the last MMQ A so consecutive same-A matmuls reuse it
/// instead of re-launching `quantize_q8_0_pad40_t`.
///
/// Correctness relies on two rules (both conservative, no write-tracking):
///   * It is valid ONLY across CONSECUTIVE prefill-MMQ MatMul nodes. Any other
///     node kind clears it (see `CudaBackend::execute_node_inner`), so a late
///     buffer-id reuse by the liveness allocator can never alias the cached A.
///   * It is cleared at split boundaries (`CudaBackend::synchronize`) so a
///     cache from a previous graph execution never leaks stale data into a
///     later one (the same pool buffer id holds different data each step).
///
/// The buffers are the state-level `buf_qa8_t`/`buf_sda_t` (transposed) and
/// `buf_q8_prefill` (native) scratch — dedicated allocations OUTSIDE the
/// graph allocator pool, so they never alias a node output buffer. The quantize
/// output is a pure function of (src, nt, id), so a hit is byte-identical to a
/// recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MmqCache {
    /// True once a quantize has been recorded into `key`.
    active: bool,
    /// (src device pointer, nt, id) of the cached A.
    key: (usize, usize, usize),
    /// True when the cached buffers hold the transposed pad40_t layout
    /// (buf_qa8_t + buf_sda_t); false => native pad40 (buf_q8_prefill).
    transposed: bool,
    /// r52: true when the entry was recorded by a mode-2 (skip-write) fused
    /// producer — the f32 src was NEVER written, so the plane is the only
    /// valid form of that activation. Any cache path that would re-quantize
    /// the src (transposed miss / native prepass) REFUSES on such an entry
    /// (loud failure) instead of silently reading the dead buffer's garbage.
    dead_write: bool,
    /// Physical buffer pointers of the cached quantized A (validated on hit —
    /// `get_or_grow` may realloc on a larger miss).
    qa8: usize,
    sda: usize,
    q8: usize,
}

impl Default for MmqCache {
    fn default() -> Self {
        MmqCache {
            active: false,
            key: (0, 0, 0),
            transposed: false,
            dead_write: false,
            qa8: 0,
            sda: 0,
            q8: 0,
        }
    }
}

/// 8p: warm the f16 weight cache only for models whose quantized matmul
/// weights total at least this much (7B q4_k_m = 4.4 GB warms; the 0.5-1.5B
/// test fixtures do not, keeping their footprint at pre-8p levels).
pub const W16_ENABLE_BYTES: usize = 2 << 30;

pub struct CudaState {
    stream: Mutex<CudaPtr>,
    /// Lazy pinned staging ring (7e⑥); None until the first async fill,
    /// and stays None if cudaHostAlloc fails (sync fallback).
    staging: Mutex<Option<PinnedPool>>,
    /// R3-A2: pinned D2H readback buffer (grown on demand; None until the
    /// first pinned read, stays None on cudaHostAlloc failure → the pageable
    /// fallback). A blocking `cudaMemcpy` into a PAGEABLE destination bounces
    /// through a driver-internal pinned buffer (see write_input_async's
    /// comment); reading into our own pinned slot skips that bounce. This is
    /// the per-decode-step logits readback (608 KB on 0.5B/7B-class vocab).
    readback: Mutex<Option<PinnedBuf>>,
    weights: Mutex<HashMap<String, (CudaPtr, usize)>>,
    /// 8p: persistent per-weight f16 dequant cache, keyed by the device
    /// weight pointer → (f16 copy, bytes). The two-pass prefill GEMM used
    /// to dequantize W on EVERY call (288 ms per 7B @2K forward); weights
    /// are immutable after registration (register_weight reuses the same
    /// device copy for same name+size and never frees on replace), so a
    /// wptr key is stable and the dequant runs once per weight per process.
    /// Adds 2 B/element (~8.6 GB on 7B q4_k_m) — MINFER_NO_W16CACHE=1
    /// reverts to the per-call scratch.
    w16_cache: Mutex<HashMap<usize, (CudaPtr, usize)>>,
    /// 8p: the f16 cache is enabled only for models whose quantized matmul
    /// weights total >= W16_ENABLE_BYTES (set by the loader's warm pass).
    /// Small models keep the per-call scratch: the test suite keeps several
    /// loaded models resident on a shared overcommitted CUDA pool and a
    /// +1-2 GB cache per loaded model tipped it over (probability OOMs).
    w16_enabled: std::sync::atomic::AtomicBool,
    /// R1: device compute capability ×100 (e.g. 1210 = sm_12.1), read once at
    /// init. Gates the int8-mma MMQ prefill path (needs sm_80+ — mma.m16n8k32).
    cc: std::sync::atomic::AtomicI32,
    /// r60: true while every quantized weight registered on this device is
    /// NB-BT-consumable (q4_K / q6_K) — the r52 mode-2 (skip-write fused
    /// producer) window proof assumes the fused pad40_t plane is consumed
    /// ONLY by the raw NB-BT GEMM paths. On any other quant mix (Q4_0/Q5_K/
    /// Q8_0/... or a 2-D F32 matmul weight) a fused rms_norm/swiglu output
    /// can feed a generic mmq_nt consumer whose native re-quantize would
    /// read the skipped f32 — deterministic loud refusal (or silent F32
    /// garbage). The loaders clear this at registration; `mmq_a_fuse_mode`
    /// degrades mode 2 -> 1 (r51 fused semantics, writes the f32) when it
    /// is false. All-q4_K/q6_K models (7B q4_k_m: embed q4_K + output q6_K)
    /// keep mode 2 unchanged.
    nb_bt_only: std::sync::atomic::AtomicBool,
    /// Names registered through `register_weight_q6k_padded` (device layout
    /// is 224-byte-padded Q6_K, not the raw GGUF byte stream) → the
    /// ORIGINAL raw byte length, so `has_weight_of_size` can still match
    /// tensors by their raw GGUF size.
    padded_weights: Mutex<HashMap<String, usize>>,
    /// r53: pre-expanded q6_K B planes (dense centered-int8, `od * id` bytes,
    /// row stride = id, super-block stride = 256) built at padded registration
    /// under `MINFER_MMQ_Q6K_NB` (r60: default-on) AND `MINFER_MMQ_Q6K_EXP
    /// != "0"` (r54: explicit "0" skips the ~1.5 GB plane build entirely),
    /// keyed by the
    /// PADDED weight's device pointer. `prefill_mmq` looks the plane up by
    /// `wptr` and passes it to the NB-BT q6_K launcher; a miss (gate off at
    /// load, allocation failure, raw 210-B layout) keeps the r41 in-kernel
    /// expand.
    q6k_exp: Mutex<HashMap<usize, CudaPtr>>,
    /// r53: the W_exp allocation-failure warning prints once per process.
    q6k_exp_warned: std::sync::atomic::AtomicBool,
    /// r56 (Session E item 2b): per-tensor precomputed dsc f32 pairs for the
    /// NB-BT q6_K kernel — plane[c*od + j] = float2(d*sc0, d*sc1), chunk-major
    /// so the per-kt staging is a contiguous 16-B cp.async stream. Keyed by the
    /// PADDED weight's device pointer exactly like `q6k_exp`; a miss (gate off,
    /// alloc failure, odd od) keeps the r41 scalar dsc path.
    q6k_dpl: Mutex<HashMap<usize, CudaPtr>>,
    q6k_dsc: Mutex<HashMap<usize, CudaPtr>>,
    /// r56: the W_dsc allocation-failure warning prints once per process.
    q6k_dsc_warned: std::sync::atomic::AtomicBool,
    /// r59: q4_K W_dsc f32-pair planes — float2(d*sc, -(dmin*m)) per
    /// (chunk, od-row), chunk-major — keyed by the RAW weight's device
    /// pointer (the r56 q6_K map pattern). A map miss keeps the in-kernel
    /// get_scale_min_k4 decode (the DSC=false instantiation).
    q4k_dsc: Mutex<HashMap<usize, CudaPtr>>,
    /// r59: the q4_K W_dsc allocation-failure warning prints once per process.
    q4k_dsc_warned: std::sync::atomic::AtomicBool,
    /// r59 rider: max id/32 seen at K-quant registration — the pre-warm
    /// MmqCache scratch sizing hint (0 until a K-quant weight registers).
    max_nchunk: std::sync::atomic::AtomicUsize,
    // Persistent activation buffers (grown on demand) with size tracking
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_hidden: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bn: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bq: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bk: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bv: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_ba: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bf: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_bg: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_q8_bn: Mutex<(CudaPtr, usize)>,
    /// 8c: prefill Q8_0-activation scratch (quantized activations for the
    /// Q4_0×Q8_0 GEMM, nt > 1). Grown on demand like the layer-path buffers.
    buf_q8_prefill: Mutex<(CudaPtr, usize)>,
    /// P6 r34: transposed-A q8_0 prepass scratch — the swizzled qs plane
    /// ([ntb][nchunk][2048]) and the packed d|ssum scale ([ntb][nchunk][256]),
    /// consumed by mmq_raw_nb_bt_kernel's bulk staging (MINFER_MMQ_A_TRANSPOSE).
    buf_qa8_t: Mutex<(CudaPtr, usize)>,
    buf_sda_t: Mutex<(CudaPtr, usize)>,
    /// r49: consecutive-window memoization of the MMQ A-quantize prepass (see
    /// [`MmqCache`]). Lives on the process singleton so the CUDA backend can
    /// invalidate it between non-MMQ nodes / graph executions.
    mmq_cache: Mutex<MmqCache>,
    /// 8d: split-K attention partials ([8][nh][pstr] floats, nh/hd are graph
    /// constants so the size is stable — grown during warmup, never inside a
    /// capture window).
    buf_attn_partial: Mutex<(CudaPtr, usize)>,
    /// 8e-reversal: decode MMVQ q8 activation scratch (nt=1, so id/32 * 40B
    /// per token — size-stable per graph, grown during warmup runs).
    buf_q8_decode: Mutex<(CudaPtr, usize)>,
    /// 8m: prefill f16 GEMM scratch — dequantized weights (od*id halves) and
    /// converted activations (nt*id halves), grown on demand. Prefill never
    /// enters a CUDA Graph capture window (8g①), so the grow is capture-safe
    /// (same assumption as the 8c buf_q8_prefill).
    buf_f16_w: Mutex<(CudaPtr, usize)>,
    buf_f16_x: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_q8_ba: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_positions: Mutex<(CudaPtr, usize)>,
    #[allow(dead_code)] // legacy surface (7e⑦)
    buf_logits: Mutex<(CudaPtr, usize)>,
    // Persistent per-layer GPU KV cache (k, v) and current size
    kv_k: Mutex<Vec<CudaPtr>>,
    kv_v: Mutex<Vec<CudaPtr>>,
    kv_size: Mutex<Vec<usize>>,
    // CUDA Graph for decode step (capture once, replay for each token)
    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    decode_graph_exec: Mutex<CudaPtr>,
    /// Process-wide stream serialization for the graph-path backend (Phase
    /// 7d): stream capture is per-stream, so while one backend holds an open
    /// capture window, every OTHER backend's stream work (fills, copies,
    /// launches, allocs) must block instead of being recorded into that
    /// graph. The capturing backend holds this lock across its window; its
    /// own enqueues skip re-locking (they are the recorded work).
    stream_lock: Mutex<()>,
}

/// Quant block element count (ggml block_q): 256 for K-quants, 32 otherwise.
fn quant_block_q(t: TensorType) -> usize {
    match t {
        TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K => 256,
        _ => 32,
    }
}

/// Quant block byte size — matches ggml type_size (Q4_0=18, Q4_1=20, Q5_0=22,
/// Q5_1=24, Q8_0=34, Q4_K=144, Q5_K=176, Q6_K=210).
fn quant_block_bytes(t: TensorType) -> usize {
    match t {
        TensorType::Q4_0 => 18,
        TensorType::Q4_1 => 20,
        TensorType::Q5_0 => 22,
        TensorType::Q5_1 => 24,
        TensorType::Q8_0 => 34,
        TensorType::Q4_K => 144,
        TensorType::Q5_K => 176,
        TensorType::Q6_K => 210,
        _ => 0,
    }
}

/// Concatenate raw quantized weights along the output (row) dimension into one
/// weight buffer for a fused matmul (nt==1 decode): the matmul kernel lays
/// weights out as [out rows][blocks][block bytes], so a row-major concat is
/// contiguous. Returns None when the weights can't share a single matmul
/// (different types, different input dims, or an unsized type).
pub fn concat_rows(tensors: &[&Tensor]) -> Option<Vec<u8>> {
    if tensors.len() < 2 {
        return None;
    }
    let tt = tensors[0].ttype;
    if tensors.iter().any(|t| t.ttype != tt) {
        return None;
    }
    let bq = quant_block_q(tt);
    let bb = quant_block_bytes(tt);
    if bb == 0 {
        return None;
    }
    let ne0 = tensors[0].shape[0] as usize;
    if tensors.iter().any(|t| t.shape[0] != ne0 as i64) {
        return None;
    }
    if ne0 % bq != 0 {
        return None;
    }
    let row = (ne0 / bq) * bb;
    let rows: usize = tensors.iter().map(|t| t.shape[1] as usize).sum();
    let mut out = Vec::with_capacity(rows * row);
    for t in tensors {
        out.extend_from_slice(t.data());
    }
    if out.len() != rows * row {
        return None;
    }
    Some(out)
}

/// Metadata-only `concat_rows` feasibility check — no byte copying. The
/// decode-graph build probes concat availability per layer; the eager
/// variant re-concatenated ~1.9 GB (28 ffn gate/up pairs on 7B) on every
/// decode graph build, measured as a ~920 ms one-time stall at the
/// prefill→decode switch. The loader performs the real concatenation once
/// at model load and registers `blk.{i}.ffn_gu`; both paths must agree, so
/// this mirrors concat_rows' precondition checks exactly (the per-tensor
/// data-length check subsumes the final `out.len() != rows * row` test).
pub fn concat_rows_feasible(tensors: &[&Tensor]) -> bool {
    if tensors.len() < 2 {
        return false;
    }
    let tt = tensors[0].ttype;
    if tensors.iter().any(|t| t.ttype != tt) {
        return false;
    }
    let bq = quant_block_q(tt);
    let bb = quant_block_bytes(tt);
    if bb == 0 {
        return false;
    }
    let ne0 = tensors[0].shape[0] as usize;
    if tensors.iter().any(|t| t.shape[0] != ne0 as i64) {
        return false;
    }
    if ne0 % bq != 0 {
        return false;
    }
    let row = (ne0 / bq) * bb;
    tensors
        .iter()
        .all(|t| t.data().len() == row * (t.shape[1] as usize))
}

/// 8b: GPU KV cache element type (CUDA side, mirrors `metal::kv_cache_is_f16`).
/// `MINFER_CACHE_TYPE=f16|f32` forces one; unset auto-selects f16 for the
/// 7B class (n_layers×n_kv_embd ≥ 8192 — KV-bandwidth-bound decode), f32 for
/// small models. Read once per CudaBackend at construction.
static KV_F16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn kv_cache_is_f16() -> bool {
    *KV_F16.get_or_init(|| false)
}

/// Called at model load with the model dims, BEFORE the first forward.
pub fn set_kv_cache_type(n_layers: usize, n_kv_embd: usize) {
    let f16 =
        std::env::var("MINFER_CACHE_TYPE").map_or(n_layers * n_kv_embd >= 8192, |v| v == "f16");
    let _ = KV_F16.set(f16);
}

impl CudaState {
    /// Preload the NVIDIA driver library.
    ///
    /// libcudart resolves `libcuda.so.1` through the loader's default search,
    /// which misses distribution-specific driver paths — and nix shells do not
    /// consult `/etc/ld.so.cache` at all (the binary then fails with
    /// cudaGetDeviceCount err 35, CUDA_ERROR_LIBRARY_NOT_FOUND). Loading it up
    /// front from the well-known locations makes cudart's own dlopen re-use
    /// the resident object (SONAME match). Harmless no-op when the driver is
    /// already loadable; libcuda's glibc-stub dependencies resolve via the
    /// binary's DT_RPATH (nix glibc dir) or the system default paths.
    fn preload_driver() {
        const RTLD_NOW: std::ffi::c_int = 2;
        const RTLD_GLOBAL: std::ffi::c_int = 0x100;
        const CANDIDATES: &[&str] = &[
            "libcuda.so.1",
            "/usr/lib/aarch64-linux-gnu/libcuda.so.1",
            "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
            "/usr/lib64/libcuda.so.1",
            "/usr/lib/libcuda.so.1",
        ];
        for c in CANDIDATES {
            let Ok(cstr) = std::ffi::CString::new(*c) else {
                continue;
            };
            let handle = unsafe { dlopen(cstr.as_ptr(), RTLD_NOW | RTLD_GLOBAL) };
            if !handle.is_null() {
                return;
            }
        }
    }

    fn try_new(requested: Option<i32>) -> Option<Self> {
        Self::preload_driver();
        if std::env::var("MINFER_DISABLE_CUDA").is_ok() {
            eprintln!("CUDA: disabled by MINFER_DISABLE_CUDA");
            return None;
        }

        let mut count: i32 = 0;
        let err = unsafe { cudaGetDeviceCount(&mut count) };
        if err != 0 || count == 0 {
            eprintln!("CUDA: no CUDA devices found (cudaGetDeviceCount err {err}, count {count})");
            return None;
        }

        // Device selection: honor `--gpu N` when given and in range; otherwise
        // auto-select the device with the highest compute capability.
        let best_device: i32 = match requested {
            Some(n) if (0..count).contains(&n) => n,
            _ => {
                if let Some(n) = requested {
                    eprintln!(
                        "CUDA: --gpu {n} out of range (found {count} device(s)); auto-selecting"
                    );
                }
                let mut best_device: i32 = 0;
                let mut best_score: i32 = 0;
                for dev in 0..count {
                    let mut major: i32 = 0;
                    let mut minor: i32 = 0;
                    unsafe {
                        cudaDeviceGetAttribute(&mut major, CUDA_DEV_ATTR_COMPUTE_MAJOR, dev);
                        cudaDeviceGetAttribute(&mut minor, CUDA_DEV_ATTR_COMPUTE_MINOR, dev);
                    }
                    let score = major * 100 + minor;
                    if score > best_score {
                        best_score = score;
                        best_device = dev;
                    }
                }
                best_device
            }
        };

        let err = unsafe { cudaSetDevice(best_device) };
        if err != 0 {
            eprintln!("CUDA: failed to set device {}", best_device);
            return None;
        }

        let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamCreate(&mut stream) };
        if err != 0 || stream.is_null() {
            eprintln!("CUDA: failed to create stream");
            return None;
        }

        // Query device properties
        fn get_attr(attr: i32, dev: i32) -> i32 {
            let mut v: i32 = 0;
            unsafe {
                cudaDeviceGetAttribute(&mut v, attr, dev);
            }
            v
        }
        let major = get_attr(CUDA_DEV_ATTR_COMPUTE_MAJOR, best_device);
        let minor = get_attr(CUDA_DEV_ATTR_COMPUTE_MINOR, best_device);
        let sm_count = get_attr(CUDA_DEV_ATTR_MULTIPROC_COUNT, best_device);
        let mut free_mem: usize = 0;
        let mut total_mem: usize = 0;
        unsafe {
            cudaMemGetInfo(&mut free_mem, &mut total_mem);
        }
        // Read device name (first 256 bytes of the oversized buffer)
        let mut name_buf = CudaDevicePropBuf([0u8; 4096]);
        unsafe {
            cudaGetDeviceProperties(&mut name_buf, best_device);
        }
        let name = name_buf.0[..256]
            .iter()
            .take_while(|&&c| c != 0)
            .map(|&c| c as u8 as char)
            .collect::<String>();
        eprintln!(
            "CUDA: using {} (SM {}.{}, {} MB, {} SMs)",
            name,
            major,
            minor,
            total_mem / 1048576,
            sm_count
        );

        let dummy = (CudaPtr(std::ptr::null_mut()), 0usize);
        // Eager dynamic-smem opt-in for the prefill GEMM instantiations:
        // must happen BEFORE any stream capture — capture mode Global
        // forbids cudaFuncSetAttribute, so a lazy first-use opt-in fails
        // and the >48KB launch poisons the context (error 700).
        unsafe {
            gemm_prefill_smem_init();
        }
        Some(CudaState {
            stream: Mutex::new(CudaPtr(stream)),
            staging: Mutex::new(None),
            readback: Mutex::new(None),
            weights: Mutex::new(HashMap::new()),
            w16_cache: Mutex::new(HashMap::new()),
            w16_enabled: std::sync::atomic::AtomicBool::new(false),
            cc: std::sync::atomic::AtomicI32::new(major * 100 + minor),
            nb_bt_only: std::sync::atomic::AtomicBool::new(true),
            padded_weights: Mutex::new(HashMap::new()),
            q6k_exp: Mutex::new(HashMap::new()),
            q6k_exp_warned: std::sync::atomic::AtomicBool::new(false),
            q6k_dpl: Mutex::new(HashMap::new()),
            q6k_dsc: Mutex::new(HashMap::new()),
            q6k_dsc_warned: std::sync::atomic::AtomicBool::new(false),
            q4k_dsc: Mutex::new(HashMap::new()),
            q4k_dsc_warned: std::sync::atomic::AtomicBool::new(false),
            max_nchunk: std::sync::atomic::AtomicUsize::new(0),
            buf_hidden: Mutex::new(dummy),
            buf_bn: Mutex::new(dummy),
            buf_bq: Mutex::new(dummy),
            buf_bk: Mutex::new(dummy),
            buf_bv: Mutex::new(dummy),
            buf_ba: Mutex::new(dummy),
            buf_bf: Mutex::new(dummy),
            buf_bg: Mutex::new(dummy),
            buf_q8_bn: Mutex::new(dummy),
            buf_q8_prefill: Mutex::new(dummy),
            buf_qa8_t: Mutex::new(dummy),
            buf_sda_t: Mutex::new(dummy),
            mmq_cache: Mutex::new(MmqCache::default()),
            buf_attn_partial: Mutex::new(dummy),
            buf_q8_decode: Mutex::new(dummy),
            buf_f16_w: Mutex::new(dummy),
            buf_f16_x: Mutex::new(dummy),
            buf_q8_ba: Mutex::new(dummy),
            buf_positions: Mutex::new(dummy),
            buf_logits: Mutex::new(dummy),
            kv_k: Mutex::new(Vec::new()),
            kv_v: Mutex::new(Vec::new()),
            kv_size: Mutex::new(Vec::new()),
            decode_graph_exec: Mutex::new(CudaPtr(std::ptr::null_mut())),
            stream_lock: Mutex::new(()),
        })
    }

    pub fn get() -> Option<&'static Self> {
        CUDA.get().and_then(|s| s.as_ref())
    }

    /// Default: auto-select the device (highest compute capability).
    pub fn init() {
        Self::init_with_gpu(None);
    }

    /// Auto-select, or honor an explicit `--gpu N` device index.
    ///
    /// `requested` is the CUDA device index from `--gpu`; `None` = auto-select.
    /// An out-of-range index warns and falls back to auto-select.
    pub fn init_with_gpu(requested: Option<i32>) {
        CUDA.get_or_init(|| {
            let s = Self::try_new(requested);
            if s.is_some() {
                eprintln!("CUDA: GPU acceleration enabled");
            } else {
                eprintln!("CUDA: not available, using CPU fallback");
            }
            s
        });
    }

    #[allow(dead_code)] // legacy surface (7e⑦): used by layer_gpu
    pub fn has_weight(&self, name: &str) -> bool {
        self.weights.lock().unwrap().contains_key(name)
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        {
            let w = self.weights.lock().unwrap();
            if let Some((_, size)) = w.get(name) {
                if *size == data.len() {
                    // Device weights are immutable: same name + size ⇒ the
                    // same GGUF tensor (single-model-per-process today, and
                    // unit tests reload the same file). Reuse the existing
                    // device copy instead of leaking one buffer per load.
                    return;
                }
                // Different size (a different architecture registered the
                // same tensor name): replace the entry. The stale buffer is
                // deliberately NOT freed — a live captured graph may still
                // reference it; the leak is bounded by the number of
                // distinct (arch, tensor) shapes ever loaded.
            }
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
        if err != 0 || ptr.is_null() {
            eprintln!(
                "CUDA: failed to allocate {} bytes for '{}'",
                data.len(),
                name
            );
            return;
        }
        let err = unsafe {
            cudaMemcpy(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        if err != 0 {
            eprintln!("CUDA: failed to copy '{}' to device", name);
            unsafe {
                cudaFree(ptr);
            }
            return;
        }
        // a plain (unpadded) registration must clear any stale padded flag
        // for the same name: a second model reusing the tensor name with a
        // non-Q6_K type would otherwise dispatch the padded-224 kernel on a
        // raw-210 buffer (Phase 8 review finding)
        self.padded_weights.lock().unwrap().remove(name);
        self.weights
            .lock()
            .unwrap()
            .insert(name.to_string(), (CudaPtr(ptr), data.len()));
    }

    /// 7e②: register a Q6_K tensor in the PADDED device layout (each
    /// 210-byte block in a 224-byte slot) so the matmul kernel can use
    /// 16-byte-aligned uint4 weight loads. `od`/`id` are the matmul output/
    /// input dims (GGUF shape [in, out] → id = shape[0], od = shape[1]).
    pub fn register_weight_q6k_padded(&self, name: &str, data: &[u8], od: usize, id: usize) {
        // r59 rider: track the max nchunk (= id/32) for the pre-warm scratch
        // sizing (see prewarm_prefill).
        self.max_nchunk
            .fetch_max(id / 32, std::sync::atomic::Ordering::Relaxed);
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id.div_ceil(256);
        let row_len = nbe * Q6KB;
        if od == 0 || id == 0 || data.len() < od * row_len {
            eprintln!(
                "CUDA: q6_k padded registration skipped for '{}' ({} bytes, od={od} id={id})",
                name,
                data.len()
            );
            return;
        }
        let mut padded = vec![0u8; od * nbe * Q6KPB];
        for r in 0..od {
            for ib in 0..nbe {
                let src = r * row_len + ib * Q6KB;
                let dst = r * nbe * Q6KPB + ib * Q6KPB;
                padded[dst..dst + Q6KB].copy_from_slice(&data[src..src + Q6KB]);
            }
        }
        self.register_weight(name, &padded);
        self.padded_weights
            .lock()
            .unwrap()
            .insert(name.to_string(), data.len());
        // D4-4 L1: dense split-plane (dpl) decode sibling plane — per row
        // [ql nbe*128][qh nbe*64][sc nbe*16][d nbe*2] at a 16B-aligned row
        // stride, 210B of content per 256-elem block (no 224B pad sectors).
        // The decode MMVQ reads it when present; per-unit values and the
        // accumulation order are unchanged, so outputs stay bitwise-identical
        // (probe /tmp/d4/probe_l1_dpl.cu). Requires id % 256 == 0 (exact
        // nbe, like the W_exp plane); MINFER_Q6K_DPL=0 skips the build and
        // decode keeps the padded kernels.
        if Self::mmq_gate_on("MINFER_Q6K_DPL") && id % 256 == 0 {
            let nbe = id / 256;
            let dpl_row = (nbe * 210 + 15) & !15usize;
            let raw_row = nbe * 210;
            let mut dpl = vec![0u8; od * dpl_row];
            for r in 0..od {
                let src = &data[r * raw_row..(r + 1) * raw_row];
                let dst = &mut dpl[r * dpl_row..(r + 1) * dpl_row];
                let (ql, rest) = dst.split_at_mut(nbe * 128);
                let (qh, rest) = rest.split_at_mut(nbe * 64);
                let (sc, dd) = rest.split_at_mut(nbe * 16);
                for ib in 0..nbe {
                    let blk = &src[ib * 210..(ib + 1) * 210];
                    ql[ib * 128..(ib + 1) * 128].copy_from_slice(&blk[..128]);
                    qh[ib * 64..(ib + 1) * 64].copy_from_slice(&blk[128..192]);
                    sc[ib * 16..(ib + 1) * 16].copy_from_slice(&blk[192..208]);
                    dd[ib * 2..(ib + 1) * 2].copy_from_slice(&blk[208..210]);
                }
            }
            let dpl_name = format!("{name}__dpl");
            self.register_weight(&dpl_name, &dpl);
            if let (Some(wp), Some(ep)) =
                (self.get_weight_ptr(name), self.get_weight_ptr(&dpl_name))
            {
                if !wp.is_null() && !ep.is_null() {
                    self.q6k_dpl
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(ep));
                }
            }
        }
        // r53: pre-expand B into the dense centered-int8 plane (P6 r44) so the
        // NB-BT q6_K kernel's B staging is a pure cp.async copy. Ships with the
        // MINFER_MMQ_Q6K_NB gate (the NB-BT kernel is its only consumer; the
        // A/B baseline is the unchanged env set) and requires id % 256 == 0
        // (the kernel's own launch gate, which also keeps the dense index
        // 16B-aligned). Dense bytes = od * id — ~2.4 GiB total on 7B q4_k_m
        // (ffn_down 67.9 MB x 28 + output 545 MB + attn_v 1.8 MB x 28).
        // r54: MINFER_MMQ_Q6K_EXP decouples the plane from the kernel gate —
        // unset/"1" keeps the r53 default (build it), explicit "0" skips the
        // build entirely (registration early-returns; device memory stays at
        // the pre-r53 level) so dispatch map-misses into the EXP=false r41
        // in-kernel expand. ANDed with Q6K_NB: EXP only matters when the NB
        // kernel is live (r60: Q6K_NB is default-on — "0" opts out of the
        // kernel AND both planes).
        if Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")
            && std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() != Ok("0")
            && id % 256 == 0
        {
            self.register_weight_q6k_exp(name, &padded, od, id);
            // r56 (Session E item 2b): the dsc f32 plane rides the same gate
            // (+ od % 2 == 0: the kernel stages row PAIRS per 16-B cp.async
            // chunk and zero-fills whole pairs, so an odd od row would lose
            // its scale — such tensors keep the scalar path via map miss).
            if od % 2 == 0 {
                self.register_weight_q6k_dsc(name, &padded, od, id);
            }
        }
    }

    /// r53: build + upload the dense pre-expanded B plane for one padded q6_K
    /// tensor and map it from the padded weight's device pointer. Called from
    /// [`Self::register_weight_q6k_padded`] under the `MINFER_MMQ_Q6K_NB`
    /// (r60: default-on) + `MINFER_MMQ_Q6K_EXP != "0"` (r54) +
    /// `id % 256 == 0` gate; also `pub`
    /// for the gate-on byte-exactness test. An alloc/upload failure leaves the
    /// map empty: the kernel falls back to the r41 in-kernel expand with a
    /// once-per-process loud eprintln.
    pub fn register_weight_q6k_exp(&self, name: &str, padded: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name: a same-name different-shape
        // re-registration can never collide with (and silently reuse) a stale
        // plane of the same byte size but a different od/id layout.
        let exp_name = format!("{name}__exp{od}x{id}");
        let exp = Self::expand_q6k_dense(padded, od, id);
        self.register_weight(&exp_name, &exp);
        // the MAP is keyed by the PADDED weight's device pointer (what
        // prefill_mmq holds); the value is the W_exp plane's pointer
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(ep) = self.get_weight_ptr(&exp_name) {
                if !wp.is_null() && !ep.is_null() {
                    self.q6k_exp
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(ep));
                    return;
                }
            }
        }
        if !self
            .q6k_exp_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q6_K W_exp pre-expand unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt_q6k falls back to the \
                 r41 in-kernel expand"
            );
        }
    }

    /// r56 (Session E item 2b): build + upload the precomputed dsc f32-pair
    /// plane for one padded q6_K tensor and map it from the padded weight's
    /// device pointer. Called from [`Self::register_weight_q6k_padded`] under
    /// the same gate as `register_weight_q6k_exp` (+ `od % 2 == 0`). An
    /// alloc/upload failure leaves the map empty: the kernel falls back to the
    /// r41 scalar dsc path with a once-per-process loud eprintln.
    pub fn register_weight_q6k_dsc(&self, name: &str, padded: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name (same rationale as the W_exp name).
        let dsc_name = format!("{name}__dsc{od}x{id}");
        let dsc = Self::expand_q6k_dsc(padded, od, id);
        self.register_weight(&dsc_name, &dsc);
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(dp) = self.get_weight_ptr(&dsc_name) {
                if !wp.is_null() && !dp.is_null() {
                    self.q6k_dsc
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(dp));
                    return;
                }
            }
        }
        if !self
            .q6k_dsc_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q6_K W_dsc plane unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt_q6k keeps the r41 \
                 scalar dsc path"
            );
        }
    }

    /// r56 (Session E item 2b): precomputed dsc pairs of one padded q6_K
    /// tensor. Output: `nchunk * od * 8` bytes, `out[(c*od + j)*8..+8]` =
    /// float2(d*sc[2(c&7)], d*sc[2(c&7)+1]) — chunk-major so the kernel's
    /// per-kt staging (rows j0..j0+MMQ_NBJ of chunk c0+kd contiguous) is a
    /// pure 16-B cp.async stream. Bit-identical to the in-kernel r41 scalar
    /// computation: exact f16->f32 (half::f16, = __half2float), exact
    /// i8->f32, one IEEE f32 multiply, no FMA contraction on either side.
    pub fn expand_q6k_dsc(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id / 256;
        let row_len = nbe * Q6KPB;
        let mut out = vec![0u8; (id / 32) * od * 8];
        for j in 0..od {
            let prow = &padded[j * row_len..(j + 1) * row_len];
            for sb in 0..nbe {
                let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
                let d_bits = u16::from_le_bytes([blk[208], blk[209]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                for cc in 0..8usize {
                    let s0 = 2 * cc;
                    let sc0 = blk[192 + s0] as i8 as f32;
                    let sc1 = blk[192 + s0 + 1] as i8 as f32;
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    out[idx..idx + 4].copy_from_slice(&(d * sc0).to_bits().to_le_bytes());
                    out[idx + 4..idx + 8].copy_from_slice(&(d * sc1).to_bits().to_le_bytes());
                }
            }
        }
        out
    }

    /// r59 (Session F item 1): build + upload the precomputed dsc f32-pair
    /// plane for one RAW q4_K tensor and map it from the raw weight's device
    /// pointer. Called from the qwen2 loader under the NB-BT gate set
    /// (MINFER_MMQ_RAW_NB=1 + MINFER_MMQ_A_TRANSPOSE=1 + MINFER_MMQ_Q4K_DSC
    /// != "0") + `id % 256 == 0` + `od % 2 == 0`. An alloc/upload failure
    /// leaves the map empty: the kernel falls back to the in-kernel scalar
    /// decode (DSC=false) with a once-per-process loud eprintln.
    pub fn register_weight_q4k_dsc(&self, name: &str, raw: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name (same rationale as the W_exp name).
        let dsc_name = format!("{name}__q4dsc{od}x{id}");
        let dsc = Self::expand_q4k_dsc(raw, od, id);
        self.register_weight(&dsc_name, &dsc);
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(dp) = self.get_weight_ptr(&dsc_name) {
                if !wp.is_null() && !dp.is_null() {
                    self.q4k_dsc
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(dp));
                    return;
                }
            }
        }
        if !self
            .q4k_dsc_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q4_K W_dsc plane unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt keeps the in-kernel \
                 scalar dsc decode"
            );
        }
    }

    /// r59 (Session F item 1): precomputed dsc pairs of one RAW q4_K tensor.
    /// Output: `(id / 32) * od * 8` bytes, `out[(c*od + j)*8..+8]` =
    /// float2(d*sc[c&7], -(dmin*m[c&7])) — chunk-major so the kernel's
    /// per-kt staging (rows j0..j0+MMQ_NBJ of chunk c0+kd contiguous) is a
    /// pure 16-B cp.async stream. Bit-identical to the in-kernel scalar
    /// computation: exact f16->f32 (half::f16 = __half2float), exact
    /// u8->f32, ONE IEEE f32 multiply, exact negation, no FMA contraction
    /// on either side.
    pub fn expand_q4k_dsc(raw: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q4KB: usize = 144;
        let nsb = id / 256;
        let nchunk = id / 32;
        let row_len = nsb * Q4KB;
        let mut out = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            let prow = &raw[j * row_len..(j + 1) * row_len];
            for sb in 0..nsb {
                let blk = &prow[sb * Q4KB..sb * Q4KB + Q4KB];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
                let sc = &blk[4..16]; // 12 packed 6-bit scales+mins
                for cc in 0..8usize {
                    // host mirror of the device get_scale_min_k4 (cuda_kernels.cu)
                    let (s, m) = if cc < 4 {
                        (sc[cc] & 63, sc[cc + 4] & 63)
                    } else {
                        (
                            (sc[cc + 4] & 0xF) | ((sc[cc - 4] >> 6) << 4),
                            (sc[cc + 4] >> 4) | ((sc[cc] >> 6) << 4),
                        )
                    };
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    out[idx..idx + 4].copy_from_slice(&(d * (s as f32)).to_bits().to_le_bytes());
                    out[idx + 4..idx + 8]
                        .copy_from_slice(&(-(dmin * (m as f32))).to_bits().to_le_bytes());
                }
            }
        }
        out
    }

    /// r59 (Session F riders, r57 items 4+5): move the one-time first-launch
    /// costs out of the measured prefill window. Called once at the end of
    /// model weight registration.
    /// (1) kernel module pre-load — cudaFuncGetAttributes over the
    ///     MMQ/FA/fused launch set forces the fatbin to load now instead of
    ///     at the first dispatch (r58 CUPTI: ~3 ms host stalls bracketing
    ///     the first mode-2 swiglu / first bt matmul);
    /// (2) pinned D2H readback pre-grow — the grow-on-demand 4 MB
    ///     cudaHostAlloc was the "0.78 ms tail malloc" at the n_out=1
    ///     logits readback;
    /// (3) MmqCache scratch pre-grow — buf_q8_prefill / buf_qa8_t /
    ///     buf_sda_t sized for a nominal 4096-token prefill (the default
    ///     n_ctx) at the max registered nchunk, so the first prefill's
    ///     get_or_grow hits instead of cudaMalloc-ing ~150 MB mid-window
    ///     (a larger prompt grows in-window exactly as before). MMQ-gated:
    ///     the planes are dead weight when the MMQ path is off.
    pub fn prewarm_prefill(&self) {
        unsafe {
            minfer_prewarm_kernels();
        }
        // (2) pinned readback pre-grow (same 4 MB floor as
        // copy_from_device_pinned; failure is non-fatal — the first readback
        // retries the alloc there).
        {
            let mut guard = self.readback.lock().unwrap();
            if guard.is_none() {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = unsafe { cudaHostAlloc(&mut p, 4 * 1024 * 1024, 0) };
                if err == 0 && !p.is_null() {
                    *guard = Some(PinnedBuf {
                        ptr: p as *mut u8,
                        bytes: 4 * 1024 * 1024,
                    });
                }
            }
        }
        // (3) MmqCache scratch pre-grow
        if Self::mmq_gate_on("MINFER_MMQ") {
            let max_nchunk = self.max_nchunk.load(std::sync::atomic::Ordering::Relaxed);
            if max_nchunk > 0 {
                const NT_PREWARM: usize = 4096; // default n_ctx
                let ntb = NT_PREWARM.div_ceil(64);
                Self::get_or_grow(&self.buf_q8_prefill, NT_PREWARM * max_nchunk * 40);
                Self::get_or_grow(&self.buf_qa8_t, ntb * max_nchunk * 2048);
                Self::get_or_grow(&self.buf_sda_t, ntb * max_nchunk * 256);
            }
        }
    }

    /// r53: dense centered-int8 pre-expansion of one padded q6_K tensor — the
    /// host mirror of the device `expand_q6_elem` (P6 r44 / MMQ-analysis
    /// §11.24). Output: `od * id` bytes, `out[j * id + sb * 256 + e]` =
    /// super-block element e of row j — the exact tile the kernel's staging
    /// used to recomb. Requires `id % 256 == 0` (the NB-BT launch gate).
    /// Two output elements per (ql, qh) byte pair: e = it*128+r and
    /// e = it*128+r+64 share ql[it*64+r] (nibble shifts 0/4) and
    /// qh[it*32 + (r&31)] (2-bit-field shifts 2*(r>>5) / 2*((r>>5)+2)).
    pub fn expand_q6k_dense(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id / 256;
        let row_len = nbe * Q6KPB;
        let mut out = vec![0u8; od * id];
        for j in 0..od {
            let prow = &padded[j * row_len..(j + 1) * row_len];
            let orow = &mut out[j * id..(j + 1) * id];
            for sb in 0..nbe {
                let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
                let (ql, qh) = blk.split_at(128);
                let obase = &mut orow[sb * 256..sb * 256 + 256];
                for it in 0..2usize {
                    for r in 0..64usize {
                        let qlb = ql[it * 64 + r];
                        let qhb = qh[it * 32 + (r & 31)];
                        let s0 = (r >> 5) * 2;
                        let e0 = it * 128 + r;
                        obase[e0] = ((qlb & 0xF) | (((qhb >> s0) & 3) << 4)).wrapping_sub(32);
                        obase[e0 + 64] =
                            (((qlb >> 4) & 0xF) | (((qhb >> (s0 + 4)) & 3) << 4)).wrapping_sub(32);
                    }
                }
            }
        }
        out
    }

    /// Whether `name` was registered in the padded Q6_K layout.
    pub fn is_weight_padded(&self, name: &str) -> bool {
        self.padded_weights.lock().unwrap().contains_key(name)
    }

    pub fn get_weight_ptr(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        self.weights.lock().unwrap().get(name).map(|(cp, _)| cp.0)
    }

    /// Process-wide model-load serialization: loaders hold this while
    /// registering weights, so two models with same-named tensors (qwen2 0.5B
    /// vs qwen3 0.6B in parallel tests) cannot interleave their registrations.
    /// The graph-path tests that span multiple forwards hold it for their body
    /// to keep the weight registry stable underneath them. REENTRANT per
    /// thread: `load_model` takes it inside callers that already hold it.
    pub fn model_load_guard() -> ModelLoadGuard {
        ModelLoadGuard::acquire()
    }

    /// Size-aware registry check for the graph-path gate: a same-name entry
    /// with a DIFFERENT byte size belongs to another architecture's model and
    /// must read as "not registered" so that model cleanly falls back to CPU.
    pub fn has_weight_of_size(&self, name: &str, bytes: usize) -> bool {
        // Padded Q6_K entries live on the device with a larger (224-byte
        // stride) footprint; match them by their ORIGINAL raw length so the
        // weights gate sees them as registered.
        if let Some(&raw) = self.padded_weights.lock().unwrap().get(name) {
            return raw == bytes;
        }
        self.weights
            .lock()
            .unwrap()
            .get(name)
            .is_some_and(|(_, size)| *size == bytes)
    }

    pub fn stream(&self) -> *mut std::ffi::c_void {
        self.stream.lock().unwrap().0
    }

    // ─── Persistent buffer management ─────────────────────────

    #[allow(dead_code)] // legacy surface (7e⑦)
    fn get_or_grow(slot: &Mutex<(CudaPtr, usize)>, need: usize) -> *mut std::ffi::c_void {
        let mut guard = slot.lock().unwrap();
        let (ptr, size) = &mut *guard;
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

    /// Allocate device memory (graph-backend pool helper). Null on failure.
    pub fn cuda_malloc(size: usize) -> *mut std::ffi::c_void {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, size) };
        if err != 0 || ptr.is_null() {
            eprintln!("CUDA: OOM allocating {} bytes", size);
            return std::ptr::null_mut();
        }
        ptr
    }

    /// Free device memory allocated via [`Self::cuda_malloc`] (no-op on null).
    pub fn cuda_free(ptr: *mut std::ffi::c_void) {
        if !ptr.is_null() {
            unsafe {
                cudaFree(ptr);
            }
        }
    }

    // ─── Copy helpers ─────────────────────────────────────────

    pub fn copy_to_device(&self, src: &[u8], dst: *mut std::ffi::c_void) {
        unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            );
        }
    }

    /// 7e⑥: async H2D input fill through a pinned staging slot. The data
    /// is copied into pinned host memory (cheap, CPU-side), then
    /// `cudaMemcpyAsync` queues the transfer on the stream — the call
    /// returns before the copy lands; same-stream ordering guarantees the
    /// fill completes before the kernels that read the buffer. Falls back
    /// to a synchronous pageable copy for oversized inputs or if pinned
    /// allocation failed.
    pub fn write_input_async(&self, data: &[u8], dst: *mut std::ffi::c_void) {
        const STAGING_SLOTS: usize = 8;
        const STAGING_SLOT_BYTES: usize = 2 * 1024 * 1024;
        let mut guard = self.staging.lock().unwrap();
        if guard.is_none() {
            let mut ptrs = Vec::new();
            let mut alloc_err = 0i32;
            for _ in 0..STAGING_SLOTS {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                alloc_err = unsafe { cudaHostAlloc(&mut p, STAGING_SLOT_BYTES, 0) };
                if alloc_err != 0 {
                    break;
                }
                ptrs.push(p as *mut u8);
            }
            // a shrunken ring silently degrades to a stream sync every
            // `ptrs.len()` fills — surface it (Phase 8 review)
            if ptrs.len() != STAGING_SLOTS {
                eprintln!(
                    "CUDA: pinned staging ring shrunk to {}/{} slots (cudaHostAlloc err {});                      fills beyond the ring fall back to sync copies",
                    ptrs.len(),
                    STAGING_SLOTS,
                    alloc_err
                );
            }
            if !ptrs.is_empty() {
                *guard = Some(PinnedPool {
                    ptrs,
                    slot_bytes: STAGING_SLOT_BYTES,
                    next: 0,
                });
            }
        }
        let pool = match guard.as_mut() {
            Some(p) if data.len() <= p.slot_bytes => p,
            _ => {
                drop(guard);
                self.copy_to_device(data, dst);
                return;
            }
        };
        // ring wrap: retire all in-flight copies before reusing slot 0. The
        // reset is re-checked under the re-lock so two threads that both
        // observed the full ring cannot both take slot 0 (Phase 8 review).
        if pool.next == pool.ptrs.len() {
            drop(guard);
            self.sync();
            guard = self.staging.lock().unwrap();
        }
        let slot = {
            let pool = guard.as_mut().unwrap();
            if pool.next >= pool.ptrs.len() {
                pool.next = 0;
            }
            let p = pool.ptrs[pool.next];
            pool.next += 1;
            p
        };
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), slot, data.len());
            cudaMemcpyAsync(
                dst,
                slot as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.stream(),
            );
        }
    }

    pub fn copy_from_device(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        unsafe {
            cudaMemcpy(
                dst.as_mut_ptr() as *mut std::ffi::c_void,
                src,
                dst.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
        }
    }

    /// R3-A2: D2H read through our own pinned staging buffer. The caller has
    /// already synchronized the stream; the copy is a blocking `cudaMemcpy`
    /// whose DESTINATION is pinned — no driver-internal bounce buffer, no
    /// pageable staging — followed by a plain CPU copy out to the caller's
    /// (pageable) slice. `MINFER_NO_PINNED_READBACK=1` or a cudaHostAlloc
    /// failure falls back to the pageable path (`copy_from_device`).
    pub fn copy_from_device_pinned(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        static FALLBACK_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if std::env::var("MINFER_NO_PINNED_READBACK").as_deref() == Ok("1") {
            self.copy_from_device(src, dst);
            return;
        }
        // headroom so small size changes don't churn the allocation
        let need = dst.len().max(4 * 1024 * 1024);
        let mut guard = self.readback.lock().unwrap();
        if guard.as_ref().map_or(true, |b| b.bytes < need) {
            if let Some(old) = guard.take() {
                drop(old); // cudaFreeHost
            }
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { cudaHostAlloc(&mut p, need, 0) };
            if err != 0 {
                drop(guard);
                if !FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "CUDA: pinned readback alloc failed (err {err}); pageable D2H fallback"
                    );
                }
                self.copy_from_device(src, dst);
                return;
            }
            *guard = Some(PinnedBuf {
                ptr: p as *mut u8,
                bytes: need,
            });
        }
        let buf = guard.as_mut().unwrap();
        unsafe {
            cudaMemcpy(
                buf.ptr as *mut std::ffi::c_void,
                src,
                dst.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            std::ptr::copy_nonoverlapping(buf.ptr, dst.as_mut_ptr(), dst.len());
        }
    }

    pub fn copy_device_to_device(
        &self,
        src: *const std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        size: usize,
    ) {
        unsafe {
            // Stream-ordered (not the legacy-sync cudaMemcpy): capturable
            // inside a CUDA Graph capture window and race-free with replay.
            cudaMemcpyAsync(
                dst as *mut std::ffi::c_void,
                src,
                size,
                CUDA_MEMCPY_DEVICE_TO_DEVICE,
                self.stream(),
            );
        }
    }

    /// Process-wide stream serialization handle (see the field docs). The
    /// returned reference is `&'static` at every call site because `CudaState`
    /// itself is only ever built as `&'static` (Box::leak in `get`), so the
    /// elided lifetime there is `'static` — guards may be stored.
    pub fn stream_lock(&self) -> &Mutex<()> {
        &self.stream_lock
    }

    pub fn sync(&self) {
        let err = unsafe { cudaGetLastError() };
        if err != 0 {
            eprintln!("CUDA kernel launch error: {}", err);
        }
        let err = unsafe { cudaStreamSynchronize(self.stream()) };
        if err != 0 {
            eprintln!("CUDA stream sync error: {}", err);
        }
    }

    /// Debug sync: print label, then sync and report error.
    /// `il` = layer index, or negative for non-layer steps (e.g. output norm).
    /// Only active when MINFER_CUDA_DEBUG is set.
    #[allow(dead_code)]
    pub fn debug_sync(&self, il: i32, label: &str) {
        if !cuda_debug_enabled() {
            return;
        }
        let err = unsafe { cudaGetLastError() };
        if il >= 0 {
            let tag = format!("l{il}: ");
            if err != 0 {
                eprintln!("CUDA DEBUG: {tag}{label} -- launch error: {err}");
            }
            let err = unsafe { cudaStreamSynchronize(self.stream()) };
            if err != 0 {
                eprintln!("CUDA DEBUG: {tag}{label} -- sync error: {err}");
            } else {
                eprintln!("CUDA DEBUG: {tag}{label} OK");
            }
        } else {
            if err != 0 {
                eprintln!("CUDA DEBUG: {label} -- launch error: {err}");
            }
            let err = unsafe { cudaStreamSynchronize(self.stream()) };
            if err != 0 {
                eprintln!("CUDA DEBUG: {label} -- sync error: {err}");
            } else {
                eprintln!("CUDA DEBUG: {label} OK");
            }
        }
    }

    // ─── Upload/download for forward pass ─────────────────────

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn upload_hidden(&self, hidden: &[f32]) {
        let need = hidden.len() * 4;
        let ptr = Self::get_or_grow(&self.buf_hidden, need);
        self.copy_to_device(
            unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, need) },
            ptr,
        );
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn download_hidden(&self, hidden: &mut [f32]) {
        let need = hidden.len() * 4;
        let guard = self.buf_hidden.lock().unwrap();
        let ptr = guard.0 .0;
        if ptr.is_null() {
            return;
        }
        self.copy_from_device(ptr as *const std::ffi::c_void, unsafe {
            std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, need)
        });
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn upload_positions(&self, positions: &[usize]) {
        let ints: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let need = ints.len() * 4;
        let ptr = Self::get_or_grow(&self.buf_positions, need);
        self.copy_to_device(
            unsafe { std::slice::from_raw_parts(ints.as_ptr() as *const u8, need) },
            ptr,
        );
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn get_positions_buf(&self) -> *mut std::ffi::c_void {
        self.buf_positions.lock().unwrap().0 .0
    }

    // ─── KV cache management ─────────────────────────────────

    /// Pre-allocate GPU KV cache for all layers to n_ctx entries.
    /// Must be called after model loading (when n_layer, n_ctx, nkt are known)
    /// but before the first forward pass. Eliminates O(n²) incremental growth.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn init_kv_cache(&self, n_layer: usize, n_ctx: usize, nkt: usize) {
        let need = n_ctx * nkt * 4;
        let mut kvec = self.kv_k.lock().unwrap();
        let mut vvec = self.kv_v.lock().unwrap();
        let mut szvec = self.kv_size.lock().unwrap();
        for il in 0..n_layer {
            let new_k = Self::cuda_malloc(need);
            let new_v = Self::cuda_malloc(need);
            if new_k.is_null() || new_v.is_null() {
                eprintln!("CUDA: failed to pre-allocate KV cache for layer {}", il);
                return;
            }
            kvec.push(CudaPtr(new_k));
            vvec.push(CudaPtr(new_v));
            szvec.push(n_ctx);
        }
        let total_kb = (n_layer * need * 2) / 1024;
        eprintln!(
            "CUDA: pre-allocated KV cache for {} layers ({:.1} MB)",
            n_layer,
            total_kb as f64 / 1024.0
        );
    }

    /// Verify KV cache has enough room for `max_nkv` entries at layer `il`.
    /// Returns false if capacity is exceeded (should never happen with pre-allocation).
    #[allow(dead_code)] // legacy surface (7e⑦)
    fn kv_ensure_layer(&self, il: usize, max_nkv: usize) -> bool {
        let szvec = self.kv_size.lock().unwrap();
        let size = szvec.get(il).copied().unwrap_or(0);
        if max_nkv > size {
            eprintln!(
                "CUDA: KV cache overflow at layer {}: need {} but allocated {}",
                il, max_nkv, size
            );
            return false;
        }
        true
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn get_kv_size(&self, il: usize) -> usize {
        let szvec = self.kv_size.lock().unwrap();
        szvec.get(il).copied().unwrap_or(0)
    }

    /// Download logits from GPU after layer loop.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn download_logits(&self, logits: &mut [f32]) {
        let need = logits.len() * 4;
        let guard = self.buf_logits.lock().unwrap();
        let ptr = guard.0 .0;
        if ptr.is_null() {
            return;
        }
        self.copy_from_device(ptr as *const std::ffi::c_void, unsafe {
            std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, need)
        });
    }

    // ─── CUDA Graph (decode step batch) ───────────────────────

    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    pub fn graph_available(&self) -> bool {
        !self.decode_graph_exec.lock().unwrap().0.is_null()
    }

    pub fn graph_begin_capture(&self) -> bool {
        let stream = self.stream();
        let err = unsafe { cudaStreamBeginCapture(stream, 1) };
        if err != 0 {
            unsafe {
                cudaGetLastError();
            }
            false
        } else {
            true
        }
    }

    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    pub fn graph_end_capture(&self) {
        let stream = self.stream();

        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        if err != 0 || graph.is_null() {
            if err != 0 {
                unsafe {
                    cudaGetLastError();
                }
            }
            return;
        }

        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe {
            cudaGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        if err != 0 || exec.is_null() {
            unsafe {
                cudaGraphDestroy(graph);
            }
            return;
        }

        unsafe {
            cudaGraphDestroy(graph);
        }
        *self.decode_graph_exec.lock().unwrap() = CudaPtr(exec);
    }

    /// Close a capture window and return the instantiated exec handle (null
    /// on failure, after clearing the CUDA error state). Used by the
    /// graph-path backend, which owns per-(uid, range) exec storage; the
    /// legacy `graph_end_capture` single-slot flow is unchanged.
    pub fn graph_end_capture_to_exec(&self) -> *mut std::ffi::c_void {
        let stream = self.stream();

        let mut graph: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamEndCapture(stream, &mut graph) };
        if err != 0 || graph.is_null() {
            if err != 0 {
                unsafe {
                    cudaGetLastError();
                }
            }
            eprintln!("CUDA: stream capture end failed (err {err})");
            return std::ptr::null_mut();
        }

        let mut exec: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe {
            cudaGraphInstantiate(
                &mut exec,
                graph,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                0,
            )
        };
        unsafe {
            cudaGraphDestroy(graph);
        }
        if err != 0 || exec.is_null() {
            eprintln!("CUDA: graph instantiate failed (err {err})");
            return std::ptr::null_mut();
        }
        exec
    }

    /// Free an instantiated graph exec (Phase 7d cache invalidation).
    pub fn graph_destroy(&self, exec: *mut std::ffi::c_void) {
        if !exec.is_null() {
            unsafe {
                cudaGraphDestroy(exec);
            }
        }
    }

    /// Launch an arbitrary instantiated graph exec on the backend stream.
    pub fn graph_launch_exec(&self, exec: *mut std::ffi::c_void) -> bool {
        if exec.is_null() {
            return false;
        }
        let stream = self.stream();
        let err = unsafe { cudaGraphLaunch(exec, stream) };
        if err != 0 {
            unsafe {
                cudaGetLastError();
            }
            return false;
        }
        true
    }

    #[allow(dead_code)] // legacy single-slot capture flow (7e⑦)
    pub fn graph_launch(&self) -> bool {
        let exec = self.decode_graph_exec.lock().unwrap().0;
        if exec.is_null() {
            return false;
        }
        let stream = self.stream();
        let err = unsafe { cudaGraphLaunch(exec, stream) };
        if err != 0 {
            return false;
        }
        true
    }

    // ─── Kernel launch operations (called from CudaCommandBuffer) ──

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn quant_matmul_q8(
        &self,
        w: &Tensor,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let wptr = self.get_weight_ptr(&w.name).expect("weight not on GPU");
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_0_matmul(
                wptr as *const u8,
                x as *const u8,
                out as *mut f32,
                od as i32,
                id as i32,
                nt as i32,
                stream,
            );
        }
    }

    /// f32-activation matmul dispatch by raw weight pointer + tensor type.
    /// The graph backend (graph/cuda_backend.rs) resolves weights by name and
    /// holds no Tensor, so dispatch takes (ptr, ttype) directly; the legacy
    /// Tensor-taking entry point below delegates here.
    pub fn matmul_f32_ptr(
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
        if nt >= 9
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
                return self.prefill_mmq(wptr, ttype, x, out, od, id, nt, padded_q6k);
            }
            return self.prefill_gemm_f16(wptr, ttype, x, out, od, id, nt, padded_q6k);
        }
        let stream = self.stream();
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
                } else {
                    launch!(launch_q4_0_f32_matmul)
                }
            }
            TensorType::Q8_0 => launch!(launch_q8_0_f32_matmul),
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
            other => Err(format!(
                "cuda: weight type {other:?} has no f32-activation matmul kernel (supported: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K)"
            )),
        }
    }

    /// r60: the promoted MMQ gate semantics — unset / any non-"0" value =
    /// ON (the verified 1.080x path), explicit "0" = opt-out to the pre-r60
    /// disabled/f16 behavior (the r54 `MINFER_MMQ_Q6K_EXP` pattern).
    /// Single-sourced: every promoted MINFER_MMQ_* dispatch and
    /// plane-registration read goes through this.
    pub fn mmq_gate_on(name: &str) -> bool {
        std::env::var(name).map_or(true, |v| v != "0")
    }

    /// r60: the loaders call this when they register a quantized weight that
    /// is NOT NB-BT-consumable (not q4_K/q6_K), or a 2-D F32 matmul weight:
    /// mode-2 skip-write fused producers become unsound for such mixes (see
    /// `nb_bt_only`) and degrade to mode 1 for the rest of the process.
    /// Registration happens before the first forward, so no per-node cost.
    pub fn clear_mmq_nb_bt_only(&self) {
        self.nb_bt_only
            .store(false, std::sync::atomic::Ordering::Relaxed);
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
    pub fn mmq_active(&self) -> bool {
        self.cc.load(std::sync::atomic::Ordering::Relaxed) >= 800 && Self::mmq_enabled()
    }

    /// Device compute capability × 100 (e.g. 1210 = sm_121); 0 when no
    /// device is initialized. Used by tests to gate sm_80+ kernels.
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

    /// r49: invalidate the MMQ A-quantize memoization. Called by the CUDA
    /// backend between non-MMQ nodes (conservative consecutive-window rule)
    /// and at split boundaries (cross-execution staleness).
    pub fn clear_mmq_cache(&self) {
        let mut c = self.mmq_cache.lock().unwrap();
        c.active = false;
        c.key = (0, 0, 0);
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
        let mut cache = self.mmq_cache.lock().unwrap();
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
        let mut cache = self.mmq_cache.lock().unwrap();
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
            let cache = self.mmq_cache.lock().unwrap();
            if cache.active && !cache.transposed && !cache.dead_write && cache.key == key {
                let q8 = Self::get_or_grow(&self.buf_q8_decode, need) as usize;
                if q8 == cache.q8 {
                    return q8 as *mut u8;
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
        let mut c = self.mmq_cache.lock().unwrap();
        c.active = true;
        c.key = (src, nt, id);
        c.transposed = false;
        c.dead_write = false;
        c.qa8 = 0;
        c.sda = 0;
        c.q8 = q8;
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
        let mut c = self.mmq_cache.lock().unwrap();
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

    /// R1: int8 MMQ prefill GEMM — quantize activations to q8_0 (pad40
    /// blocks; the quantize kernel also emits the per-block int sum used by
    /// the min-term correction), then one tiled mma.m16n8k32 (s8) launch per
    /// weight (see mmq_nt_kernel). Caller guarantees nt >= 16, id % 32 == 0,
    /// and a supported quant type. The q8 scratch follows the same
    /// grow-on-demand lifecycle as the f16 path's buf_f16_x: the 3-run
    /// capture protocol sizes it before the capture window opens.
    pub fn prefill_mmq(
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
        let type_id = match ttype {
            TensorType::Q8_0 => 0,
            TensorType::Q4_0 => 1,
            TensorType::Q4_1 => 2,
            TensorType::Q5_0 => 3,
            TensorType::Q5_1 => 4,
            TensorType::Q4_K => 5,
            TensorType::Q5_K => 6,
            TensorType::Q6_K => 7,
            other => return Err(format!("cuda: prefill MMQ got unsupported type {other:?}")),
        };
        // r49: the native pad40 buffer is sized upfront so an OOM surfaces here
        // (before any GEMM launch) instead of as a null deref in the q4_K
        // fallback's final `launch_mmq_raw_nt`. The fallbacks re-derive it via
        // `mmq_quantize_native` (which dedups); this call only guards OOM.
        if Self::get_or_grow(&self.buf_q8_prefill, nt * (id / 32) * 40).is_null() {
            return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
        }
        // Q6_K reads whichever layout the weight was registered with
        // (224-byte padded 7e② repack or raw 210); all others are raw.
        let block_stride: i32 = if ttype == TensorType::Q6_K && padded_q6k {
            224
        } else {
            210
        };
        let stream = self.stream();
        // P6 r38: q6_K on a raw-byte BT-style kernel (expanded centered-int8 B,
        // m16n8k16 KSPLIT=2). Same A prepass as r34; the B path is q6_K-specific.
        // Gated on MINFER_MMQ_Q6K_NB + MINFER_MMQ_RAW (r60: default-on,
        // "0" opts out; the q6k path always uses the transposed-A raster).
        // On any cap/mismatch the launcher returns 0 -> clean fall through
        // to the generic mmq_nt<7,2>.
        if type_id == 7
            && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")
            && Self::mmq_gate_on("MINFER_MMQ_RAW")
            && (id / 32) % 8 == 0
        {
            let nchunk = (id / 32) as i32;
            let ntb = ((nt as i64 + 63) / 64) as i32;
            unsafe {
                // r49: A-quantize prepass via the consecutive-window cache —
                // a same-A (>q/k/v, gate/up) matmul reuses qa8g/sdag without a
                // fresh quantize launch.
                let (qa8g, sdag) = self.mmq_quantize_transposed(
                    x as *const f32,
                    id as i32,
                    nt as i32,
                    nchunk,
                    ntb,
                    stream,
                );
                // r53: pre-expanded B plane lookup by the padded weight's
                // device pointer (null on miss -> launcher selects the r41
                // in-kernel-expand instantiation).
                let w_exp = self
                    .q6k_exp
                    .lock()
                    .unwrap()
                    .get(&(wptr as usize))
                    .map(|cp| cp.0)
                    .unwrap_or(std::ptr::null_mut());
                // r56 (Session E item 2b): the precomputed dsc f32-pair plane
                // (null on miss -> the r41 scalar dsc path in-kernel).
                let w_dsc = self
                    .q6k_dsc
                    .lock()
                    .unwrap()
                    .get(&(wptr as usize))
                    .map(|cp| cp.0)
                    .unwrap_or(std::ptr::null_mut());
                if qa8g != 0
                    && sdag != 0
                    && launch_mmq_raw_nb_bt_q6k_nt(
                        type_id,
                        wptr as *const u8,
                        w_exp as *const u8,
                        w_dsc as *const u8,
                        qa8g as *const u8,
                        sdag as *const u8,
                        out as *mut f32,
                        nt as i32,
                        od as i32,
                        id as i32,
                        nchunk,
                        block_stride,
                        stream,
                        8,
                    ) == 1
                {
                    if std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1") {
                        // r54: name WHY the r41 in-kernel expand is running —
                        // "exp=off" is the intentional MINFER_MMQ_Q6K_EXP=0
                        // switch; "fallback!" means a W_exp build was expected
                        // (padded weight, EXP gate on) but the map missed
                        // (alloc/upload failure or a registration bug). Raw
                        // 210-B weights never get a plane -> kept unqualified.
                        let b = if !w_exp.is_null() {
                            "W_exp-cp.async"
                        } else if !padded_q6k {
                            "in-kernel-expand"
                        } else if std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() == Ok("0") {
                            "in-kernel-expand(exp=off)"
                        } else {
                            "in-kernel-expand(fallback!)"
                        };
                        // r56: name the A/dsc staging paths too (liveness
                        // check per the r53 lesson — a fallback-correct fast
                        // path needs a visible label, parity cannot see it).
                        let a = if !w_dsc.is_null() {
                            "A=cp.async DSC=f32-plane"
                        } else {
                            "A=cp.async DSC=scalar"
                        };
                        eprintln!(
                            "minfer/cuda: mmq raw NB-BT q6_K kernel active \
                             (r56 {}, r54 B={}, r39 KDR=2 double-buffer, \
                             A-transpose)",
                            a, b
                        );
                    }
                    return Ok(());
                }
            }
        }
        // P6: raw-byte staging variant (q4_K, whole super-blocks only).
        // Same quantized activations; the GEMM stages RAW weight bytes via
        // cp.async and dequants in registers (docs/CUDA_OPTIMIZATION.md).
        if type_id == 5 && Self::mmq_gate_on("MINFER_MMQ_RAW") && (id / 32) % 8 == 0 {
            let kd: i32 = std::env::var("MINFER_MMQ_RAW_KD")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(8);
            let wide = std::env::var("MINFER_MMQ_RAW_WIDE").as_deref() == Ok("1");
            let nb = Self::mmq_gate_on("MINFER_MMQ_RAW_NB");
            let nb_debug = std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1");
            let at = Self::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE");
            unsafe {
                // P6 r34: relocate the A-side layout transform out of the mma
                // kernel into a quantize-transpose prepass (llama.cpp's design).
                // Under MINFER_MMQ_A_TRANSPOSE=1 the activations are emitted
                // PRE-TRANSPOSED (quantize_q8_0_pad40_t) so the NB kernel's A
                // staging is a bulk LDG->STS; the native q8 buffer is only
                // filled on the (rare) bb-bt fallback below.
                let mut nb_ok = false;
                if nb && at && kd == 8 {
                    let nchunk = (id / 32) as i32;
                    let ntb = ((nt as i64 + 63) / 64) as i32;
                    // r49: cache-backed transposed A-quantize prepass (reuses
                    // the previous same-A matmul's qa8g/sdag when consecutive).
                    let (qa8g, sdag) = self.mmq_quantize_transposed(
                        x as *const f32,
                        id as i32,
                        nt as i32,
                        nchunk,
                        ntb,
                        stream,
                    );
                    // r59: the q4_K W_dsc f32-pair plane (null on miss ->
                    // the DSC=false in-kernel scalar decode instantiation).
                    let w_dsc = self
                        .q4k_dsc
                        .lock()
                        .unwrap()
                        .get(&(wptr as usize))
                        .map(|cp| cp.0)
                        .unwrap_or(std::ptr::null_mut());
                    nb_ok = qa8g != 0
                        && sdag != 0
                        && launch_mmq_raw_nb_bt_nt(
                            type_id,
                            wptr as *const u8,
                            w_dsc as *const u8,
                            qa8g as *const u8,
                            sdag as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            nchunk,
                            stream,
                            kd,
                        ) == 1;
                    if nb_ok && nb_debug {
                        // r59: name the dsc staging path (liveness check per
                        // the r53 lesson — a fallback-correct fast path needs
                        // a visible label, parity cannot see it). "fallback!"
                        // means a plane was expected (gate on) but the map
                        // missed (alloc/upload failure or registration bug).
                        let d = if !w_dsc.is_null() {
                            "DSC=f32-plane"
                        } else if std::env::var("MINFER_MMQ_Q4K_DSC").as_deref() == Ok("0") {
                            "DSC=in-kernel(dsc=off)"
                        } else {
                            "DSC=in-kernel(fallback!)"
                        };
                        eprintln!(
                            "minfer/cuda: mmq raw NB-BT kernel active \
                             (KD=8, A-transpose, r59 {d})"
                        );
                    }
                }
                if !nb_ok {
                    // r49: cache-backed native A-quantize prepass. The helper
                    // returns 0 (no buffer) on OOM; the launchers below only
                    // run when a buffer is present (the original unconditional
                    // `q8` came from a pre-call get_or_grow, now inside the
                    // helper).
                    let q8 =
                        self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
                    // Direction-A NB raw-nibble kernel is KD=8-native; it
                    // activates only under the full MMQ gate set. launcher
                    // returns 0 on KD!=8 or smem/reg cap failure -> clean
                    // fallback to the wide/narrow raw path below.
                    nb_ok = q8 != 0
                        && nb
                        && launch_mmq_raw_nb_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 1;
                    if nb_ok && nb_debug {
                        eprintln!("minfer/cuda: mmq raw NB kernel active (KD=8)");
                    }
                }
                if !nb_ok {
                    let q8 =
                        self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
                    if q8 == 0 {
                        // r52: mmq_quantize_native refuses a mode-2 dead-write
                        // A (and plain q8 OOM is pre-checked at fn entry) —
                        // never fall through to a GEMM on a null/garbage A.
                        return Err("cuda: prefill MMQ: A-quantize unavailable (mode-2 \
                             dead-write A refused or q8 OOM); MINFER_MMQ_A_FUSE=2 \
                             window violated"
                            .to_string());
                    }
                    let wide_ok = q8 != 0
                        && wide
                        && launch_mmq_raw_wide_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        ) == 1;
                    if !wide_ok {
                        launch_mmq_raw_nt(
                            type_id,
                            wptr as *const u8,
                            q8 as *const u8,
                            out as *mut f32,
                            nt as i32,
                            od as i32,
                            id as i32,
                            stream,
                            kd,
                        );
                    }
                }
            }
            return Ok(());
        }
        unsafe {
            let q8 = self.mmq_quantize_native(x as *const f32, id as i32, nt as i32, stream);
            if q8 == 0 {
                return Err("cuda: prefill MMQ q8 scratch OOM".to_string());
            }
            launch_mmq_nt(
                type_id,
                wptr as *const u8,
                q8 as *const u8,
                out as *mut f32,
                nt as i32,
                od as i32,
                id as i32,
                block_stride,
                stream,
            );
        }
        Ok(())
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
            if rc == 0 && free_mem < 2 * bytes + (4usize << 30) {
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
            unsafe {
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
                );
            }
            return Ok(());
        }
        let x16 = Self::get_or_grow(&self.buf_f16_x, nt * id * 2);
        unsafe {
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
            );
        }
        Ok(())
    }

    pub fn quant_matmul_f32_on_gpu(
        &self,
        w: &Tensor,
        x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        let wptr = self.get_weight_ptr(&w.name).expect("weight not on GPU");
        self.matmul_f32_ptr(wptr, w.ttype, x, out, od, id, nt)
            .unwrap_or_else(|e| panic!("CUDA: {e}"));
    }

    pub fn matmul_on_gpu(
        &self,
        w: &Tensor,
        q8_x: *mut std::ffi::c_void,
        f32_x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        if w.ttype == TensorType::Q4_0 {
            self.quant_matmul_q8(w, q8_x, out, od, id, nt);
        } else if w.ttype == TensorType::Q8_0 {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        } else if w.ttype == TensorType::Q4_1 {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        } else {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        }
    }

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

    pub fn rms_norm(
        &self,
        x: *mut std::ffi::c_void,
        w: Option<*mut std::ffi::c_void>,
        y: *mut std::ffi::c_void,
        d: usize,
        n: usize,
        eps: f32,
    ) {
        let wptr =
            w.expect("CUDA rms_norm: weight required (no-weights variant not yet implemented)");
        let stream = self.stream();
        unsafe {
            launch_rms_norm_f32(
                x as *const f32,
                wptr as *const f32,
                y as *mut f32,
                d as i32,
                eps,
                n as i32,
                stream,
            );
        }
    }

    pub fn add_f32(
        &self,
        x: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        z: *mut std::ffi::c_void,
        n: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_add_f32(
                x as *const f32,
                y as *const f32,
                z as *mut f32,
                n as i32,
                stream,
            );
        }
    }

    /// Add a per-row bias to a token-major `[rows][d]` buffer: `y[t][i] += b[i]`.
    /// `rows` is the ROW COUNT (token count) — the kernel grid maps one block
    /// row per token, so passing the total element count writes out of bounds.
    pub fn add_bias_f32(
        &self,
        y: *mut std::ffi::c_void,
        b: *mut std::ffi::c_void,
        d: usize,
        rows: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_add_bias_f32(
                y as *mut f32,
                b as *const f32,
                d as i32,
                rows as i32,
                stream,
            );
        }
    }

    pub fn mul_f32(
        &self,
        x: *mut std::ffi::c_void,
        y: *mut std::ffi::c_void,
        z: *mut std::ffi::c_void,
        n: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_mul_f32(
                x as *const f32,
                y as *const f32,
                z as *mut f32,
                n as i32,
                stream,
            );
        }
    }

    pub fn silu_f32(&self, y: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_silu_f32(y as *mut f32, n as i32, stream);
        }
    }

    pub fn swiglu_f32(
        &self,
        gate: *mut std::ffi::c_void,
        up: *mut std::ffi::c_void,
        dst: *mut std::ffi::c_void,
        n: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_swiglu_f32(
                gate as *const f32,
                up as *const f32,
                dst as *mut f32,
                n as i32,
                stream,
            );
        }
    }

    pub fn rope_f32(
        &self,
        x: *mut std::ffi::c_void,
        n_head: usize,
        n_dims: usize,
        nt: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: *mut std::ffi::c_void,
    ) {
        let stream = self.stream();
        unsafe {
            launch_rope_f32(
                x as *mut f32,
                n_head as i32,
                n_dims as i32,
                nt as i32,
                freq_base,
                freq_scale,
                positions as *const i32,
                stream,
            );
        }
    }

    pub fn gqa_attn_f32(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
    ) {
        let stream = self.stream();
        unsafe {
            launch_gqa_attn_f32(
                q as *const f32,
                k as *const f32,
                v as *const f32,
                o as *mut f32,
                positions as *const i32,
                nh as i32,
                nk as i32,
                hd as i32,
                scale,
                nt as i32,
                stream,
            );
        }
    }

    /// 8b: GQA attention over an f16 KV cache (K/V read as half and
    /// converted to f32 in registers; q/o stay f32). Matches Metal's
    /// pl_gqa_attn_f16 precision class (f16 storage, f32 accumulate).
    pub fn gqa_attn_f16kv(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
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
        if nt >= 2 && hd == 128 && !Self::no_fa_prefill() {
            let rc = unsafe {
                launch_fa_prefill_f16kv(
                    q as *const f32,
                    k as *const std::ffi::c_void,
                    v as *const std::ffi::c_void,
                    o as *mut f32,
                    positions as *const i32,
                    nh as i32,
                    nk as i32,
                    hd as i32,
                    scale,
                    nt as i32,
                    stream,
                )
            };
            if rc == 0 {
                return;
            }
        }
        unsafe {
            launch_gqa_attn_f32_f16kv(
                q as *const f32,
                k as *const std::ffi::c_void,
                v as *const std::ffi::c_void,
                o as *mut f32,
                positions as *const i32,
                nh as i32,
                nk as i32,
                hd as i32,
                scale,
                nt as i32,
                stream,
            );
        }
    }

    /// 8d: split-K decode attention (nt == 1). `pstr` = (4 + hd + 3) & !3 —
    /// the partials' row stride keeps the oc section 16-byte aligned. The
    /// scratch must be at least ATTN_SPLITS * nh * pstr floats (see
    /// buf_attn_partial). ATTN_SPLITS must mirror the `ATTN_SPLITS` define
    /// in cuda_kernels.cu (fixed grid — the graph-replay capture depends on
    /// it; idle splits write an mx=-INF/S=0 partial the combine weights to
    /// zero).
    pub fn gqa_attn_split(
        &self,
        q: *mut std::ffi::c_void,
        k: *mut std::ffi::c_void,
        v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void,
        positions: *mut std::ffi::c_void,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        f16_kv: bool,
    ) {
        let pstr = ((4 + hd + 3) & !3) as i32;
        const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
        let need = ATTN_SPLITS * nh * (pstr as usize) * 4;
        let partial = Self::get_or_grow(&self.buf_attn_partial, need);
        let stream = self.stream();
        unsafe {
            if f16_kv {
                launch_gqa_attn_split_f16kv(
                    q as *const f32,
                    k,
                    v,
                    o as *mut f32,
                    partial as *mut f32,
                    positions as *const i32,
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

    /// 8e follow-up: `MINFER_NO_KQ_MMVQ=1` forces the K-quant decode matmuls
    /// back onto the f32 kernels (A/B escape hatch, MINFER_NO_FUSE_* style).
    fn no_kq_mmvq() -> bool {
        std::env::var("MINFER_NO_KQ_MMVQ").map_or(false, |v| v == "1")
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

    /// Decode I32 graph inputs (f32::from_bits bit patterns, alloc.rs
    /// fill_input_i32) into raw int32 for the rope/store/attention kernels.
    /// Fully device-side: no host sync, capture-safe (Phase 7d).
    pub fn bits_to_i32(&self, src: *mut std::ffi::c_void, dst: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_f32_bits_to_i32(src as *const f32, dst as *mut i32, n as i32, stream);
        }
    }

    /// 8b: f16 KV cache — see `kv_cache_is_f16`. Same trade-off as Metal:
    /// halves attention KV read bandwidth; the region stays f32-sized (the
    /// f16 view uses the first half of the bytes).
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

    /// D3-8: fused decode QKV epilogue (G4 CUDA port of Metal's
    /// `attn_bias_rope_store`). `q`/`k`/`v` are POINTER-FORM section bases:
    /// the concat class passes sections of the concat matmul output [q|k|v]
    /// (nt==1), the mixed-quant class passes the three separate matmul
    /// outputs. Biases added per section, q/k roped in place (math verbatim
    /// `rope_f32`), k/v stored into the persistent regions at the same
    /// addresses as `store_kv_f32`/`store_kv_f16` (f32 or f16 per `kv_is_f16`).
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
        kv_is_f16: bool,
    ) {
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
                kv_is_f16 as i32,
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

    // ─── Batch quant_matmul (for Q/K/V projection) ────────────

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn quant_matmul_f32_batch(
        &self,
        mats: &mut [(
            /*weight*/ &Tensor,
            /*output*/ &mut [f32],
            /*od*/ usize,
        )],
        x: &[f32],
        id: usize,
        nt: usize,
    ) {
        // For batch Q4_0 matmuls: quantize activations once, then launch each matmul
        if mats.iter().any(|m| m.0.ttype != TensorType::Q4_0) {
            // Fall back to CPU for non-Q4_0 types
            for mat in mats.iter_mut() {
                crate::kernel::cpu_quant_matmul_f32(mat.0, x, mat.1, mat.2, id, nt);
            }
            return;
        }

        let nb = id / 32;
        let q8_len = nt * nb * Q8B;
        let mut q8 = vec![0u8; q8_len];
        crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut q8);

        let xbuf = Self::get_or_grow(&self.buf_hidden, q8_len);
        self.copy_to_device(&q8, xbuf);

        // Launch each matmul and read back results
        for (_i, mat) in mats.iter_mut().enumerate() {
            let out_len = nt * mat.2 * 4;
            let obuf = Self::get_or_grow(&self.buf_bq, out_len);
            self.quant_matmul_q8(mat.0, xbuf, obuf, mat.2, id, nt);
            self.sync();
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(mat.1.as_mut_ptr() as *mut u8, out_len) };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        }
    }

    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn quant_matmul_f32(
        &self,
        w: &Tensor,
        x: &[f32],
        out: &mut [f32],
        od: usize,
        id: usize,
        nt: usize,
    ) {
        if w.ttype == TensorType::Q4_0 {
            let nb = id / 32;
            let q8_len = nt * nb * Q8B;
            let out_len = nt * od * 4;

            let mut q8 = vec![0u8; q8_len];
            crate::quants::quantize_row_q8_0_buf(x, nt, id, &mut q8);

            let xbuf = Self::get_or_grow(&self.buf_hidden, q8_len);
            let obuf = Self::get_or_grow(&self.buf_logits, out_len);

            self.copy_to_device(&q8, xbuf);
            self.quant_matmul_q8(w, xbuf, obuf, od, id, nt);
            self.sync();
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_len) };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        } else if w.ttype == TensorType::Q8_0 {
            let out_len = nt * od * 4;
            let x_len = nt * id * 4;
            let xbuf = Self::get_or_grow(&self.buf_hidden, x_len);
            let obuf = Self::get_or_grow(&self.buf_logits, out_len);
            self.copy_to_device(
                unsafe { std::slice::from_raw_parts(x.as_ptr() as *const u8, x_len) },
                xbuf,
            );
            self.quant_matmul_f32_on_gpu(w, xbuf, obuf, od, id, nt);
            self.sync();
            let out_bytes =
                unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_len) };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        } else {
            crate::kernel::cpu_quant_matmul_f32(w, x, out, od, id, nt);
        }
    }

    // ─── Full-layer GPU pass ──────────────────────────────────

    /// Encode one transformer layer onto the CUDA stream.
    /// Returns false if any weight is missing from GPU.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn layer_gpu(
        &self,
        il: usize,
        l: &crate::models::qwen2::loader::LayerWeights,
        positions: &[usize],
        ne: usize,
        nqt: usize,
        nkt: usize,
        nf: usize,
        nt: usize,
        nh: usize,
        nk: usize,
        hd: usize,
        eps: f32,
        freq_base: f32,
        freq_scale: f32,
    ) -> bool {
        let attn_norm = match &l.attn_norm {
            Some(t) => t,
            None => return false,
        };
        let ffn_norm = match &l.ffn_norm {
            Some(t) => t,
            None => return false,
        };
        let wq = l.wq.as_ref().unwrap();
        let wk = l.wk.as_ref().unwrap();
        let wv = l.wv.as_ref().unwrap();
        let wo = l.wo.as_ref().unwrap();
        let ffn_gate = l.ffn_gate.as_ref().unwrap();
        let ffn_up = l.ffn_up.as_ref().unwrap();
        let ffn_down = l.ffn_down.as_ref().unwrap();

        // Accept Q4_0/Q4_1 group or Q4_K/Q6_K group (no mixing between groups)
        fn is_q4(t: TensorType) -> bool {
            t == TensorType::Q4_0 || t == TensorType::Q4_1
        }
        fn is_qk(t: TensorType) -> bool {
            t == TensorType::Q4_K || t == TensorType::Q6_K
        }
        let all_q4 = is_q4(wq.ttype)
            && is_q4(wk.ttype)
            && is_q4(wv.ttype)
            && is_q4(wo.ttype)
            && is_q4(ffn_gate.ttype)
            && is_q4(ffn_up.ttype)
            && is_q4(ffn_down.ttype);
        let all_qk = is_qk(wq.ttype)
            && is_qk(wk.ttype)
            && is_qk(wv.ttype)
            && is_qk(wo.ttype)
            && is_qk(ffn_gate.ttype)
            && is_qk(ffn_up.ttype)
            && is_qk(ffn_down.ttype);
        if !all_q4 && !all_qk {
            return false;
        }

        if !self.has_weight(&wq.name)
            || !self.has_weight(&wk.name)
            || !self.has_weight(&wv.name)
            || !self.has_weight(&wo.name)
            || !self.has_weight(&ffn_gate.name)
            || !self.has_weight(&ffn_up.name)
            || !self.has_weight(&ffn_down.name)
        {
            return false;
        }
        let norm_attn_w = match self.get_weight_ptr(&attn_norm.name) {
            Some(p) => p,
            None => return false,
        };
        let norm_ffn_w = match self.get_weight_ptr(&ffn_norm.name) {
            Some(p) => p,
            None => return false,
        };
        let bq_bias = l.bq.as_ref().and_then(|b| self.get_weight_ptr(&b.name));
        let bk_bias = l.bk.as_ref().and_then(|b| self.get_weight_ptr(&b.name));
        let bv_bias = l.bv.as_ref().and_then(|b| self.get_weight_ptr(&b.name));

        let max_pos = positions.iter().copied().max().unwrap_or(0);
        if !self.kv_ensure_layer(il, max_pos + 1) {}

        let hidden_len = nt * ne * 4;
        let bn_len = hidden_len;
        let bq_len = nt * nqt * 4;
        let bk_len = nt * nkt * 4;
        let bv_len = bk_len;
        let ba_len = nt * ne * 4;
        let bf_len = nt * nf.max(ne) * 4;
        let bg_len = nt * nf * 4;
        let q8_bn_len = nt * (ne / 32) * Q8B;
        let q8_ba_len = nt * (nf.max(ne) / 32) * Q8B;

        let hidden = Self::get_or_grow(&self.buf_hidden, hidden_len);
        let bn = Self::get_or_grow(&self.buf_bn, bn_len);
        let bq_buf = Self::get_or_grow(&self.buf_bq, bq_len);
        let bk_buf = Self::get_or_grow(&self.buf_bk, bk_len);
        let bv_buf = Self::get_or_grow(&self.buf_bv, bv_len);
        let ba_buf = Self::get_or_grow(&self.buf_ba, ba_len);
        let bf_buf = Self::get_or_grow(&self.buf_bf, bf_len);
        let bg_buf = Self::get_or_grow(&self.buf_bg, bg_len);
        let q8_bn = Self::get_or_grow(&self.buf_q8_bn, q8_bn_len);
        let q8_ba = Self::get_or_grow(&self.buf_q8_ba, q8_ba_len);
        let pos_buf = self.get_positions_buf();
        let kv_k = self.kv_k.lock().unwrap()[il].0;
        let kv_v = self.kv_v.lock().unwrap()[il].0;

        // Attention branch
        self.rms_norm(hidden, Some(norm_attn_w), bn, ne, nt, eps);
        self.debug_sync(il as i32, "rms_norm(attn)");

        self.quantize_q8_0(bn, q8_bn, ne, nt);
        self.debug_sync(il as i32, "quantize_q8_0(attn)");
        self.matmul_on_gpu(wq, q8_bn, bn, bq_buf, nqt, ne, nt);
        self.debug_sync(il as i32, "wq matmul");
        if let Some(bb) = bq_bias {
            self.add_bias_f32(bq_buf, bb, nqt, nt);
            self.debug_sync(il as i32, "bq bias");
        }
        self.matmul_on_gpu(wk, q8_bn, bn, bk_buf, nkt, ne, nt);
        self.debug_sync(il as i32, "wk matmul");
        if let Some(bb) = bk_bias {
            self.add_bias_f32(bk_buf, bb, nkt, nt);
            self.debug_sync(il as i32, "bk bias");
        }
        self.matmul_on_gpu(wv, q8_bn, bn, bv_buf, nkt, ne, nt);
        self.debug_sync(il as i32, "wv matmul");
        if let Some(bb) = bv_bias {
            self.add_bias_f32(bv_buf, bb, nkt, nt);
            self.debug_sync(il as i32, "bv bias");
        }
        self.rope_f32(bq_buf, nh, hd, nt, freq_base, freq_scale, pos_buf);
        self.debug_sync(il as i32, "rope q");
        self.rope_f32(bk_buf, nk, hd, nt, freq_base, freq_scale, pos_buf);
        self.debug_sync(il as i32, "rope k");
        self.store_kv_f32(bk_buf, kv_k as *mut std::ffi::c_void, nkt, nt, pos_buf);
        self.debug_sync(il as i32, "store_kv k");
        self.store_kv_f32(bv_buf, kv_v as *mut std::ffi::c_void, nkt, nt, pos_buf);
        self.debug_sync(il as i32, "store_kv v");
        let scale = 1.0 / (hd as f32).sqrt();
        self.gqa_attn_f32(
            bq_buf,
            kv_k as *mut std::ffi::c_void,
            kv_v as *mut std::ffi::c_void,
            ba_buf,
            pos_buf,
            nh,
            nk,
            hd,
            scale,
            nt,
        );
        self.debug_sync(il as i32, "gqa_attn");

        // wo projection
        self.quantize_q8_0(ba_buf, q8_ba, ne, nt);
        self.debug_sync(il as i32, "quantize_q8_0(wo)");
        self.matmul_on_gpu(wo, q8_ba, ba_buf, bn, ne, ne, nt);
        self.debug_sync(il as i32, "wo matmul");
        self.add_f32(hidden, bn, hidden, nt * ne);
        self.debug_sync(il as i32, "add(residual attn)");

        // FFN branch
        self.rms_norm(hidden, Some(norm_ffn_w), ba_buf, ne, nt, eps);
        self.debug_sync(il as i32, "rms_norm(ffn)");
        self.quantize_q8_0(ba_buf, q8_ba, ne, nt);
        self.debug_sync(il as i32, "quantize_q8_0(ffn)");
        self.matmul_on_gpu(ffn_gate, q8_ba, ba_buf, bg_buf, nf, ne, nt);
        self.debug_sync(il as i32, "ffn_gate matmul");
        self.matmul_on_gpu(ffn_up, q8_ba, ba_buf, bf_buf, nf, ne, nt);
        self.debug_sync(il as i32, "ffn_up matmul");
        self.swiglu_f32(bg_buf, bf_buf, bg_buf, nt * nf);
        self.debug_sync(il as i32, "swiglu");
        self.quantize_q8_0(bg_buf, q8_ba, nf, nt);
        self.debug_sync(il as i32, "quantize_q8_0(ffn_down)");
        self.matmul_on_gpu(ffn_down, q8_ba, bg_buf, bn, ne, nf, nt);
        self.debug_sync(il as i32, "ffn_down matmul");
        self.add_f32(hidden, bn, hidden, nt * ne);
        self.debug_sync(il as i32, "add(residual ffn)");

        true
    }

    /// Final RMSNorm + output matmul on GPU.
    #[allow(dead_code)] // legacy surface (7e⑦)
    pub fn output_norm_gpu(
        &self,
        output: &Tensor,
        output_norm: Option<&Tensor>,
        output_b: Option<&Tensor>,
        ne: usize,
        nv: usize,
        nt: usize,
        n_out: usize,
        eps: f32,
    ) -> bool {
        let norm_w = match output_norm {
            Some(t) => match self.get_weight_ptr(&t.name) {
                Some(w) => w,
                None => return false,
            },
            None => return false,
        };
        if !self.has_weight(&output.name) {
            return false;
        }
        if output.ttype != TensorType::Q4_0
            && output.ttype != TensorType::Q8_0
            && output.ttype != TensorType::Q4_1
            && output.ttype != TensorType::Q4_K
            && output.ttype != TensorType::Q6_K
        {
            return false;
        }
        debug_assert!(n_out <= nt, "n_out={n_out} > nt={nt}");

        // Output rows = last n_out tokens (single-sequence [nt][ne] row-major).
        let hid_off = (nt - n_out) * ne * 4;

        let hidden = Self::get_or_grow(&self.buf_hidden, nt * ne * 4);
        let bn = Self::get_or_grow(&self.buf_bn, n_out * ne * 4);
        let logits = Self::get_or_grow(&self.buf_logits, n_out * nv * 4);

        let hidden_off = unsafe { (hidden as *mut u8).add(hid_off) } as *mut std::ffi::c_void;
        self.rms_norm(hidden_off, Some(norm_w), bn, ne, n_out, eps);
        self.debug_sync(-1, "output: rms_norm");

        if output.ttype == TensorType::Q4_0 {
            let q8_len = n_out * (ne / 32) * Q8B;
            let q8_bn = Self::get_or_grow(&self.buf_q8_bn, q8_len);
            self.quantize_q8_0(bn, q8_bn, ne, n_out);
            self.debug_sync(-1, "output: quantize_q8_0");
            self.quant_matmul_q8(output, q8_bn, logits, nv, ne, n_out);
            self.debug_sync(-1, "output: q4_0 matmul");
        } else {
            self.quant_matmul_f32_on_gpu(output, bn, logits, nv, ne, n_out);
            self.debug_sync(-1, "output: f32 matmul");
        }

        if let Some(ob) = output_b {
            if let Some(bias_buf) = self.get_weight_ptr(&ob.name) {
                self.add_bias_f32(logits, bias_buf, nv, n_out);
                self.debug_sync(-1, "output: bias");
            }
        }
        true
    }
}

#[cfg(test)]
mod d35_probe_tests {
    use super::*;

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    fn dev_alloc(bytes: usize) -> *mut std::ffi::c_void {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut p, bytes) };
        assert_eq!(err, 0, "cudaMalloc failed");
        p
    }

    fn h2d_f32(dst: *mut std::ffi::c_void, src: &[f32]) {
        let err = unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len() * 4,
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(err, 0);
    }

    fn d2h_f32(src: *mut std::ffi::c_void, n: usize) -> Vec<f32> {
        let mut out = vec![0f32; n];
        let err = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                n * 4,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(err, 0);
        out
    }

    /// D2H readback of the shared decode q8 scratch (private-field probe).
    fn read_q8(st: &CudaState, bytes: usize) -> Vec<u8> {
        let guard = st.buf_q8_decode.lock().unwrap();
        let (ptr, size) = &*guard;
        assert!(*size >= bytes, "q8 scratch smaller than probe readback");
        let mut out = vec![0u8; bytes];
        let err = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                ptr.0 as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(err, 0);
        out
    }

    /// Deterministic activation with real-model spread (RMSNorm outputs reach
    /// ±3 and some 32-blocks are near-zero).
    fn gen_acts(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (s >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                (u as f32) * mag
            })
            .collect()
    }

    /// Valid finite q4_K bytes (od x id), deterministic per (r, ib).
    fn gen_q4_k(od: usize, id: usize) -> Vec<u8> {
        let nbe = id / 256;
        let mut b = Vec::with_capacity(od * nbe * 144);
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                for j in 0..12 {
                    b.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
                }
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    b.push(lo | (hi << 4));
                }
            }
        }
        b
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// D3-5 1a bitwise probe: fused-producer q8 epilogues vs the standalone
    /// quantize kernel — identical pad40 bytes, identical f32 producer
    /// outputs, bit-identical MMVQ outputs through the cache-hit path.
    #[test]
    fn cuda_decode_a_quant_fuse_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        let eps = 1e-5f32;

        // ---- rms probe: d 5120 (14B hidden) -> q4_K matmul (MMVQ gate) ----
        let (d, od) = (5120usize, 256usize);
        let x = gen_acts(d, 0x9E37);
        let w: Vec<f32> = (0..d).map(|i| 0.5 + 0.001 * (i % 17) as f32).collect();
        let dx = dev_alloc(d * 4);
        h2d_f32(dx, &x);
        let dw = dev_alloc(d * 4);
        h2d_f32(dw, &w);
        let dy_a = dev_alloc(d * 4);
        let dy_b = dev_alloc(d * 4);
        let do_a = dev_alloc(od * 4);
        let do_b = dev_alloc(od * 4);
        st.register_weight("d35_probe_w4", &gen_q4_k(od, d));
        let wq = st
            .get_weight_ptr("d35_probe_w4")
            .expect("weight registered");

        // path A (pre-D3-5 shape): plain rms; matmul cache-cleared -> the
        // standalone quantize launch inside the decode matmul
        st.clear_mmq_cache();
        st.rms_norm(dx, Some(dw), dy_a, d, 1, eps);
        st.matmul_f32_ptr_layout(wq, TensorType::Q4_K, dy_a, do_a, od, d, 1, false)
            .unwrap();
        let q8_a = read_q8(st, (d / 32) * 40);

        // path B: fused rms (records the plane) + matmul (cache hit)
        st.clear_mmq_cache();
        st.rms_norm_quant_on_gpu(dx, dw, dy_b, d, 1, eps);
        st.matmul_f32_ptr_layout(wq, TensorType::Q4_K, dy_b, do_b, od, d, 1, false)
            .unwrap();
        let q8_b = read_q8(st, (d / 32) * 40);

        assert_eq!(q8_a, q8_b, "rms epilogue q8 bytes differ from standalone");
        let y_a = d2h_f32(dy_a, d);
        let y_b = d2h_f32(dy_b, d);
        assert!(
            bits_eq(&y_a, &y_b),
            "fused rms f32 output not bit-identical"
        );
        let o_a = d2h_f32(do_a, od);
        let o_b = d2h_f32(do_b, od);
        assert!(bits_eq(&o_a, &o_b), "MMVQ output diverged (rms path)");

        // ---- swiglu probe: n 2048 (down id, q4_K MMVQ gate) ----
        let (nf, odf) = (2048usize, 256usize);
        let gate = gen_acts(nf, 0x1234);
        let up = gen_acts(nf, 0x5678);
        let mut buf = vec![0f32; 2 * nf];
        buf[..nf].copy_from_slice(&gate);
        buf[nf..].copy_from_slice(&up);
        let db_a = dev_alloc(2 * nf * 4);
        h2d_f32(db_a, &buf);
        let db_b = dev_alloc(2 * nf * 4);
        h2d_f32(db_b, &buf);
        let da_a = dev_alloc(odf * 4);
        let da_b = dev_alloc(odf * 4);
        st.register_weight("d35_probe_w4b", &gen_q4_k(odf, nf));
        let wq2 = st
            .get_weight_ptr("d35_probe_w4b")
            .expect("weight registered");

        st.clear_mmq_cache();
        st.swiglu_f32_off_on_gpu(db_a, nf, nf);
        st.matmul_f32_ptr_layout(wq2, TensorType::Q4_K, db_a, da_a, odf, nf, 1, false)
            .unwrap();
        let q8_a2 = read_q8(st, (nf / 32) * 40);

        st.clear_mmq_cache();
        st.swiglu_quant_off_on_gpu(db_b, nf, nf);
        st.matmul_f32_ptr_layout(wq2, TensorType::Q4_K, db_b, da_b, odf, nf, 1, false)
            .unwrap();
        let q8_b2 = read_q8(st, (nf / 32) * 40);

        assert_eq!(
            q8_a2, q8_b2,
            "swiglu epilogue q8 bytes differ from standalone"
        );
        let bu_a = d2h_f32(db_a, 2 * nf);
        let bu_b = d2h_f32(db_b, 2 * nf);
        assert!(
            bits_eq(&bu_a, &bu_b),
            "fused swiglu f32 output not bit-identical"
        );
        let oa_a = d2h_f32(da_a, odf);
        let oa_b = d2h_f32(da_b, odf);
        assert!(bits_eq(&oa_a, &oa_b), "MMVQ output diverged (swiglu path)");
    }
}

// ────────────────────────────────────────────────────────────────────
// D3-8 probes: FusedQKV decode fusion (CUDA port of the Metal G4 path)
// ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod d38_probe_tests {
    use super::*;
    use crate::tensor::TensorType;

    fn device() -> Option<&'static CudaState> {
        CudaState::init();
        CudaState::get()
    }

    fn dev_alloc(bytes: usize) -> *mut std::ffi::c_void {
        let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut p, bytes) };
        assert_eq!(err, 0, "cudaMalloc failed");
        p
    }

    fn h2d(dst: *mut std::ffi::c_void, src: &[u8]) {
        let err = unsafe {
            cudaMemcpy(
                dst,
                src.as_ptr() as *const std::ffi::c_void,
                src.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
            )
        };
        assert_eq!(err, 0);
    }

    fn h2d_f32(dst: *mut std::ffi::c_void, src: &[f32]) {
        let mut bytes = Vec::with_capacity(src.len() * 4);
        for v in src {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        h2d(dst, &bytes);
    }

    fn d2h(src: *mut std::ffi::c_void, bytes: usize) -> Vec<u8> {
        let mut out = vec![0u8; bytes];
        let err = unsafe {
            cudaMemcpy(
                out.as_mut_ptr() as *mut std::ffi::c_void,
                src as *const std::ffi::c_void,
                bytes,
                CUDA_MEMCPY_DEVICE_TO_HOST,
            )
        };
        assert_eq!(err, 0);
        out
    }

    fn d2h_f32(src: *mut std::ffi::c_void, n: usize) -> Vec<f32> {
        let bytes = d2h(src, n * 4);
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn bits_eq(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Deterministic activation with real-model spread (same generator class
    /// as the D3-5 probes).
    fn gen_acts(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f64) / ((1u64 << 31) as f64) - 1.0;
                let mag = if (s >> 60) & 7 == 0 { 1e-5 } else { 3.0 };
                (u as f32) * mag
            })
            .collect()
    }

    /// Valid finite q4_K bytes (od x id), deterministic per (r, ib).
    fn gen_q4_k(od: usize, id: usize) -> Vec<u8> {
        let nbe = id / 256;
        let mut b = Vec::with_capacity(od * nbe * 144);
        for r in 0..od {
            for ib in 0..nbe {
                let d = 0.031f32 + 0.005 * ((r * 7 + ib * 3) % 5) as f32;
                let dmin = 0.002f32 + 0.001 * ((r * 3 + ib) % 4) as f32;
                b.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                b.extend_from_slice(&half::f16::from_f32(dmin).to_le_bytes());
                for j in 0..12 {
                    b.push(((r * 31 + j * 17 + ib * 5) % 63) as u8);
                }
                for j in 0..128 {
                    let lo = ((r * 13 + j * 7 + ib * 3) % 15) as u8;
                    let hi = ((r * 5 + j * 11 + ib * 2) % 15) as u8;
                    b.push(lo | (hi << 4));
                }
            }
        }
        b
    }

    /// D3-8 probe A: fused `attn_bias_rope_store` vs the unfused chain
    /// (add_bias×3 + rope×2 + store_kv×2) on the SAME concat-matmul output —
    /// bitwise on the q/k/v sections AND both KV regions, f32 + f16 KV,
    /// 14B + 7B shapes, BOTH pointer forms (concat sections AND three
    /// separate buffers — the mixed-quant class-2 wiring).
    #[test]
    fn cuda_fused_qkv_epilogue_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        let (freq_base, freq_scale) = (10000.0f32, 1.0f32);
        let pos0: i32 = 1234;

        // device positions buffer in the graph's i32 form (the kernel reads
        // positions[0]; the graph path converts via f32_bits_to_i32)
        let dpos = dev_alloc(4);
        h2d(dpos, &pos0.to_le_bytes());

        // (nh, nk, hd, tag): 14B GQA 40:8, 7B GQA 28:4 — hd 128 both
        for &(nh, nk, hd, tag) in &[(40usize, 8usize, 128usize, "14B"), (28, 4, 128, "7B")] {
            let (nqt, nkt) = (nh * hd, nk * hd);
            let total = nqt + 2 * nkt;
            let ctx_elems = nkt * (pos0 as usize + 2); // room for the store row
            let acts = gen_acts(total, 0xD380 + nh as u64);
            let bq = gen_acts(nqt, 0xB10 + nh as u64);
            let bk = gen_acts(nkt, 0xB20 + nh as u64);
            let bv = gen_acts(nkt, 0xB30 + nh as u64);

            let dbq = dev_alloc(nqt * 4);
            h2d_f32(dbq, &bq);
            let dbk = dev_alloc(nkt * 4);
            h2d_f32(dbk, &bk);
            let dbv = dev_alloc(nkt * 4);
            h2d_f32(dbv, &bv);

            for kv_f16 in [false, true] {
                let kv_bytes = if kv_f16 { 2 } else { 4 };

                // ---- FUSED form 1: concat buffer, section pointers ----
                let d_fused = dev_alloc(total * 4);
                h2d_f32(d_fused, &acts);
                let base = d_fused as *mut u8;
                let (q1, k1, v1) = (
                    d_fused,
                    unsafe { base.add(nqt * 4) } as *mut std::ffi::c_void,
                    unsafe { base.add((nqt + nkt) * 4) } as *mut std::ffi::c_void,
                );
                let dk_f = dev_alloc(ctx_elems * kv_bytes);
                let dv_f = dev_alloc(ctx_elems * kv_bytes);
                st.attn_bias_rope_store(
                    q1, k1, v1, dbq, dbk, dbv, dk_f, dv_f, nqt, nkt, hd, freq_base, freq_scale,
                    dpos, kv_f16,
                );

                // ---- FUSED form 2: three separate buffers (class-2 shape) --
                let d_q2 = dev_alloc(nqt * 4);
                h2d_f32(d_q2, &acts[..nqt]);
                let d_k2 = dev_alloc(nkt * 4);
                h2d_f32(d_k2, &acts[nqt..nqt + nkt]);
                let d_v2 = dev_alloc(nkt * 4);
                h2d_f32(d_v2, &acts[nqt + nkt..]);
                let dk_f2 = dev_alloc(ctx_elems * kv_bytes);
                let dv_f2 = dev_alloc(ctx_elems * kv_bytes);
                st.attn_bias_rope_store(
                    d_q2, d_k2, d_v2, dbq, dbk, dbv, dk_f2, dv_f2, nqt, nkt, hd, freq_base,
                    freq_scale, dpos, kv_f16,
                );

                // ---- UNFUSED: the 7-launch chain on split sections ----
                let d_q = dev_alloc(nqt * 4);
                h2d_f32(d_q, &acts[..nqt]);
                let d_k = dev_alloc(nkt * 4);
                h2d_f32(d_k, &acts[nqt..nqt + nkt]);
                let d_v = dev_alloc(nkt * 4);
                h2d_f32(d_v, &acts[nqt + nkt..]);
                st.add_bias_f32(d_q, dbq, nqt, 1);
                st.add_bias_f32(d_k, dbk, nkt, 1);
                st.add_bias_f32(d_v, dbv, nkt, 1);
                st.rope_f32(d_q, nh, hd, 1, freq_base, freq_scale, dpos);
                st.rope_f32(d_k, nk, hd, 1, freq_base, freq_scale, dpos);
                let dk_u = dev_alloc(ctx_elems * kv_bytes);
                let dv_u = dev_alloc(ctx_elems * kv_bytes);
                if kv_f16 {
                    st.store_kv_f16(d_k, dk_u, nkt, 1, dpos);
                    st.store_kv_f16(d_v, dv_u, nkt, 1, dpos);
                } else {
                    st.store_kv_f32(d_k, dk_u, nkt, 1, dpos);
                    st.store_kv_f32(d_v, dv_u, nkt, 1, dpos);
                }

                // ---- compare: q/k/v sections + both KV rows ----
                let (rq, rk, rv) = (d2h_f32(d_q, nqt), d2h_f32(d_k, nkt), d2h_f32(d_v, nkt));
                let f1 = d2h_f32(d_fused, total);
                let (qf, kf, vf) = (
                    &f1[..nqt],
                    &f1[nqt..nqt + nkt],
                    &f1[nqt + nkt..nqt + 2 * nkt],
                );
                assert!(
                    bits_eq(qf, &rq),
                    "{tag} f16={kv_f16}: concat q section diverged"
                );
                assert!(
                    bits_eq(kf, &rk),
                    "{tag} f16={kv_f16}: concat k section diverged"
                );
                assert!(
                    bits_eq(vf, &rv),
                    "{tag} f16={kv_f16}: concat v section diverged"
                );
                assert!(
                    bits_eq(&d2h_f32(d_q2, nqt), &rq)
                        && bits_eq(&d2h_f32(d_k2, nkt), &rk)
                        && bits_eq(&d2h_f32(d_v2, nkt), &rv),
                    "{tag} f16={kv_f16}: separate-buffer form diverged"
                );
                let row = (pos0 as usize) * nkt;
                if kv_f16 {
                    for (f, u, what) in [
                        (dk_f, dk_u, "K"),
                        (dv_f, dv_u, "V"),
                        (dk_f2, dk_u, "K form2"),
                        (dv_f2, dv_u, "V form2"),
                    ] {
                        let fh = d2h(f, ctx_elems * 2);
                        let uh = d2h(u, ctx_elems * 2);
                        assert_eq!(
                            fh[row * 2..(row + nkt) * 2],
                            uh[row * 2..(row + nkt) * 2],
                            "{tag}: f16 {what} region diverged"
                        );
                    }
                } else {
                    for (f, u, what) in [
                        (dk_f, dk_u, "K"),
                        (dv_f, dv_u, "V"),
                        (dk_f2, dk_u, "K form2"),
                        (dv_f2, dv_u, "V form2"),
                    ] {
                        assert!(
                            bits_eq(
                                &d2h_f32(f, ctx_elems)[row..row + nkt],
                                &d2h_f32(u, ctx_elems)[row..row + nkt],
                            ),
                            "{tag}: f32 {what} region diverged"
                        );
                    }
                }
                eprintln!("{tag} kv_f16={kv_f16}: epilogue bitwise OK (both pointer forms)");
            }
        }
    }

    /// D3-8 probe B: the concat matmul (one launch over wq|wk|wv rows) is
    /// per-row bit-identical to the three separate decode matmuls — the MMVQ
    /// kernels map one block per row and dispatch on (ttype, id, nt) only.
    #[test]
    fn cuda_fused_qkv_concat_matmul_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();

        // (oq, okv, id, tag): 14B wq|wk|wv = 5120|1024|1024 id 5120;
        // 7B = 3584|512|512 id 3584. Both decode-MMVQ (id ≥ 2048, % 32 == 0).
        for &(oq, okv, id, tag) in &[
            (5120usize, 1024usize, 5120usize, "14B"),
            (3584, 512, 3584, "7B"),
        ] {
            let od_total = oq + 2 * okv;
            let row_bytes = (id / 256) * 144; // q4_K
            let concat = gen_q4_k(od_total, id);
            st.register_weight("d38_concat", &concat);
            st.register_weight("d38_wq", &concat[..row_bytes * oq]);
            st.register_weight("d38_wk", &concat[row_bytes * oq..row_bytes * (oq + okv)]);
            st.register_weight(
                "d38_wv",
                &concat[row_bytes * (oq + okv)..row_bytes * od_total],
            );
            let w_cat = st.get_weight_ptr("d38_concat").expect("concat registered");
            let w_q = st.get_weight_ptr("d38_wq").expect("wq registered");
            let w_k = st.get_weight_ptr("d38_wk").expect("wk registered");
            let w_v = st.get_weight_ptr("d38_wv").expect("wv registered");
            assert!(!st.is_weight_padded("d38_concat"));

            let x = gen_acts(id, 0xC0DE + id as u64);
            let dx = dev_alloc(id * 4);
            h2d_f32(dx, &x);
            let d_cat = dev_alloc(od_total * 4);
            // unfused: wq/wk/wv write separate output buffers (the live path)
            let d_q = dev_alloc(oq * 4);
            let d_k = dev_alloc(okv * 4);
            let d_v = dev_alloc(okv * 4);

            // unfused: three separate decode matmuls (share the A-quantize
            // MmqCache exactly like the live wq/wk/wv group)
            st.clear_mmq_cache();
            st.matmul_f32_ptr_layout(w_q, TensorType::Q4_K, dx, d_q, oq, id, 1, false)
                .unwrap();
            st.matmul_f32_ptr_layout(w_k, TensorType::Q4_K, dx, d_k, okv, id, 1, false)
                .unwrap();
            st.matmul_f32_ptr_layout(w_v, TensorType::Q4_K, dx, d_v, okv, id, 1, false)
                .unwrap();

            // fused: one concat matmul (fresh cache window, standalone quantize)
            st.clear_mmq_cache();
            st.matmul_f32_ptr_layout(w_cat, TensorType::Q4_K, dx, d_cat, od_total, id, 1, false)
                .unwrap();

            let cat = d2h_f32(d_cat, od_total);
            let sq = d2h_f32(d_q, oq);
            let sk = d2h_f32(d_k, okv);
            let sv = d2h_f32(d_v, okv);
            assert!(bits_eq(&cat[..oq], &sq), "{tag}: q rows diverged");
            assert!(bits_eq(&cat[oq..oq + okv], &sk), "{tag}: k rows diverged");
            assert!(bits_eq(&cat[oq + okv..], &sv), "{tag}: v rows diverged");
            eprintln!("{tag}: concat matmul bitwise OK ({od_total} rows)");
        }
    }

    // D4-4 L1: the dense split-plane (dpl) decode path must be bitwise
    // identical to the padded path. Registers the same raw q6_K bytes twice
    // (dpl built vs MINFER_Q6K_DPL=0) and compares decode outputs bit-exact.
    #[test]
    fn cuda_q6k_dpl_bitwise() {
        let Some(st) = device() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let _guard = CudaState::model_load_guard();
        // (od, id): a v2_pf_dpl shape (id in (8192, 16384]) and a v2_dpl
        // loop shape; both must clear the decode MMVQ gate (nt==1, id%32==0,
        // od*id >= 4M).
        for (od, id) in [(512usize, 8960usize), (4096usize, 1024usize)] {
            let raw = gen_q6_k_raw(od, id);
            std::env::remove_var("MINFER_Q6K_DPL");
            st.register_weight_q6k_padded("d44_dpl_a", &raw, od, id);
            std::env::set_var("MINFER_Q6K_DPL", "0");
            st.register_weight_q6k_padded("d44_dpl_b", &raw, od, id);
            std::env::remove_var("MINFER_Q6K_DPL");
            let wa = st.get_weight_ptr("d44_dpl_a").expect("a registered");
            let wb = st.get_weight_ptr("d44_dpl_b").expect("b registered");
            let x: Vec<f32> = (0..id).map(|i| ((i % 13) as f32 - 6.0) * 0.125).collect();
            let dx = dev_alloc(id * 4);
            h2d_f32(dx, &x);
            let oa = dev_alloc(od * 4);
            let ob = dev_alloc(od * 4);
            st.matmul_f32_ptr_layout(wa, TensorType::Q6_K, dx, oa, od, id, 1, true)
                .unwrap();
            st.matmul_f32_ptr_layout(wb, TensorType::Q6_K, dx, ob, od, id, 1, true)
                .unwrap();
            let ra = d2h_f32(oa, od);
            let rb = d2h_f32(ob, od);
            assert_eq!(
                ra, rb,
                "dpl-vs-padded decode outputs must be bit-identical (od {od} id {id})"
            );
            eprintln!("d44_dpl: od {od} id {id} bitwise OK");
        }
    }

    // raw GGUF-layout q6_K bytes: random ql/qh nibbles, small int8 scales,
    // d = 1.0 (finite outputs so the bit-exact compare is meaningful).
    fn gen_q6_k_raw(od: usize, id: usize) -> Vec<u8> {
        let nbe = id.div_ceil(256);
        let mut v = vec![0u8; od * nbe * 210];
        let mut s: u32 = 0x5DEE_CE6D;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            s
        };
        for r in 0..od {
            for b in 0..nbe {
                let blk = &mut v[(r * nbe + b) * 210..(r * nbe + b + 1) * 210];
                for i in 0..192 {
                    blk[i] = (next() & 0x55) as u8;
                }
                for i in 0..16 {
                    blk[192 + i] = (next() % 15) as u8;
                }
                blk[208..210].copy_from_slice(&0x3C00u16.to_le_bytes());
            }
        }
        v
    }
}
