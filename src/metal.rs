// MPS (Metal) backend for Apple Silicon.
//
// Provides MpsCommandBuffer for batching all layer ops into one GPU submission.

use crate::tensor::{Tensor, TensorType};
use block2::RcBlock;
use dispatch2::DispatchData;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBarrierScope, MTLBlitCommandEncoder, MTLBuffer, MTLCaptureDescriptor, MTLCaptureDestination,
    MTLCaptureManager, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder,
    MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::OnceLock;

static MPS: OnceLock<Option<MpsState>> = OnceLock::new();

#[cfg(target_os = "macos")]
pub type MetalDevice = Retained<ProtocolObject<dyn MTLDevice>>;
#[cfg(target_os = "macos")]
pub type MetalCommandQueue = Retained<ProtocolObject<dyn MTLCommandQueue>>;
#[cfg(target_os = "macos")]
pub type MetalCommandBuffer = Retained<ProtocolObject<dyn MTLCommandBuffer>>;
#[cfg(target_os = "macos")]
pub type MetalComputeCommandEncoder = Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>;
#[cfg(target_os = "macos")]
pub type MetalBuffer = Retained<ProtocolObject<dyn MTLBuffer>>;
#[cfg(target_os = "macos")]
pub type MetalComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
#[cfg(target_os = "macos")]
pub type MetalLibrary = Retained<ProtocolObject<dyn MTLLibrary>>;
#[cfg(target_os = "macos")]

/// Serialize Metal-touching tests.
///
/// Parallel test threads submitting to the same MTLCommandQueue can make the
/// GPU intermittently drop kernel writes — observed on Apple M4 Pro as
/// `kernel_q8_0_f32_matmul_multi` losing whole threadgroup rows (two adjacent
/// output rows stay 0) while the command buffer still reports Completed, when
/// the heavy `prefill_gemm_throughput_profile` test (50-kernel batches, up to
/// ~700 MB of buffers per case) is running concurrently. Product code is
/// single-worker serial and never submits concurrently, so this is a
/// test-only guard: every test that touches MPS takes the lock.
#[cfg(test)]
pub(crate) fn metal_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    // Tolerate poisoning: the guard only serializes GPU access (no shared data),
    // and a panicking test (e.g. the pre-existing q4 overflow) would otherwise
    // poison the lock for every later test.
    LOCK.get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// Print a clear error for an unsafe/unsupported GPU configuration and exit.
/// All GPU safety guards (dimension misalignment, device-limit overruns,
/// kernel-array overflow) abort here so the user knows the GPU path cannot run
/// the model — never silently fall back to CPU (which would mask the problem).
fn gpu_abort(msg: &str) -> ! {
    eprintln!("MPS: unsupported GPU configuration — refusing to risk a GPU fault:");
    eprintln!("  {msg}");
    eprintln!("  (force CPU with MINFER_DISABLE_MPS=1)");
    std::process::exit(1);
}

/// Quant block width (elements per block) for the fused QKV matmul concat.
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

/// Concatenate the raw quantized weights along the output (row) dimension into
/// one weight buffer for a fused matmul (nt==1 decode). The matmul kernel lays
/// weights out as [out rows][in/block_q blocks][block bytes], so a row-major
/// concat is contiguous. Returns None when the weights can't share a single
/// matmul (different types, different input dims, or an unsized type).
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

/// KV cache element type for the GPU path. `MINFER_CACHE_TYPE=f16` forces a
/// half cache (llama.cpp's default); `MINFER_CACHE_TYPE=f32` forces f32. When
/// unset, `set_kv_cache_type` (called at model load with the model dims)
/// auto-selects: f16 for the 7B class (n_layers×n_kv_embd ≥ 8192 — KV
/// bandwidth-bound decode; measured 7B @2K ctx f16 ≈ −1 ms/token vs f32),
/// f32 for small models (0.5B measured f16 ~3% SLOWER — dispatch-latency-bound,
/// see §0 decided-not #8 / §2.5).
static KV_F16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

pub fn kv_cache_is_f16() -> bool {
    *KV_F16.get_or_init(|| false)
}

/// Called once at model load with the model dims, BEFORE the first forward:
/// sets the GPU KV cache element type (auto-select or MINFER_CACHE_TYPE).
pub fn set_kv_cache_type(n_layers: usize, n_kv_embd: usize) {
    let f16 = std::env::var("MINFER_CACHE_TYPE").map_or(
        n_layers * n_kv_embd >= 8192, // auto: 7B class → f16
        |v| v == "f16",
    );
    let _ = KV_F16.set(f16);
}

/// Use the 256-thread multi-simdgroup rms_norm in the decode path (P1 2026-08-10
/// A/B gate; ON by default after it measured ~2x faster than the 32-thread kernel).
pub fn rms_norm_256_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_RMS_256").map_or(true, |v| v != "1"))
}

/// Use matmul-based prefill attention (P1 2026-08-11): broadcast+quantize the
/// KV to Q8_0 and compute kq/kqv via the fast Q8_0 GEMM, replacing the
/// latency-bound classic kernel for nt>1. ON by default; MINFER_NO_MATMUL_ATTN=1
/// falls back to the classic kernel for A/B.
pub fn matmul_attn_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_MATMUL_ATTN").map_or(true, |v| v != "1"))
}

/// Use the llama flash-attention port (kernel_flash_attn_ext_f32/_f16 and the
/// hd=128 variants) for nt==1 decode. Fixed-shape kernels → requires hd==64
/// (DK=DV=64) or hd==128 (DK=DV=128); anything else falls back to the
/// split-attention path. ON by default; MINFER_NO_FLASH=1 reverts to the split
/// path for A/B.
pub fn flash_attn_enabled(hd: usize) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_FLASH").map_or(true, |v| v != "1"))
        && (hd == 64 || hd == 128)
}

/// Use the llama kernel_flash_attn_ext_blk port (kernel_flash_attn_blk_f32/_f16,
/// legacy simdgroup_matrix) for prefill attention when nt>1. Fixed-shape
/// (DK=DV=64 or DK=DV=128) kernel → requires hd==64 or hd==128; anything else
/// falls back to the 3-pass parallel attention. ON by default;
/// MINFER_NO_PREFILL_FLASH=1 reverts to the 3-pass path for A/B.
pub fn prefill_flash_enabled(hd: usize) -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_NO_PREFILL_FLASH").map_or(true, |v| v != "1"))
        && (hd == 64 || hd == 128)
}

pub struct MpsState {
    #[cfg(target_os = "macos")]
    inner: MpsStateInner,
}
// Metal objects (objc2) are not auto Send/Sync, but minfer touches MpsState only
// through the single scheduler thread (or metal_test_lock in tests), and Metal
// objects are thread-safe in reality — so assert it.
unsafe impl Send for MpsState {}
unsafe impl Sync for MpsState {}

#[cfg(target_os = "macos")]
struct MpsStateInner {
    device: MetalDevice,
    // Cached device capabilities (queried once at init, aligned with llama.cpp's
    // ggml-metal-device props). All dispatch-time guards compare against these.
    max_threadgroup_memory: u64,
    queue: MetalCommandQueue,
    pl_q4_0_f32: MetalComputePipelineState,
    pl_q4_0_f32_multi: MetalComputePipelineState,
    pl_q4_0_mm_f32: MetalComputePipelineState,
    pl_q4_1_f32: MetalComputePipelineState,
    pl_q4_1_f32_multi: MetalComputePipelineState,
    pl_q4_1_mm_f32: MetalComputePipelineState,
    pl_q8_0_f32: MetalComputePipelineState,
    pl_q8_0_f32_multi: MetalComputePipelineState,
    pl_q8_0_mm_f32: MetalComputePipelineState,
    pl_q4_k_f32: MetalComputePipelineState,
    pl_q4_k_f32_multi: MetalComputePipelineState,
    pl_q4_k_mm_f32: MetalComputePipelineState,
    pl_q6_k_f32: MetalComputePipelineState,
    pl_q6_k_f32_multi: MetalComputePipelineState,
    pl_q6_k_mm_f32: MetalComputePipelineState,
    pl_q5_0_f32: MetalComputePipelineState,
    pl_q5_0_f32_multi: MetalComputePipelineState,
    pl_q5_0_mm_f32: MetalComputePipelineState,
    pl_q5_1_f32: MetalComputePipelineState,
    pl_q5_1_f32_multi: MetalComputePipelineState,
    pl_q5_1_mm_f32: MetalComputePipelineState,
    pl_q5_k_f32: MetalComputePipelineState,
    pl_q5_k_f32_multi: MetalComputePipelineState,
    pl_q5_k_mm_f32: MetalComputePipelineState,
    pl_get_rows_q4_0: MetalComputePipelineState,
    pl_get_rows_f32: MetalComputePipelineState,
    pl_get_rows_q4_k: MetalComputePipelineState,
    pl_get_rows_q4_1: MetalComputePipelineState,
    pl_get_rows_q5_0: MetalComputePipelineState,
    pl_get_rows_q5_1: MetalComputePipelineState,
    pl_get_rows_q8_0: MetalComputePipelineState,
    pl_get_rows_q6_k: MetalComputePipelineState,
    pl_get_rows_q5_k: MetalComputePipelineState,
    pl_rms_norm: MetalComputePipelineState,
    pl_rms_norm_256: MetalComputePipelineState,
    pl_add: MetalComputePipelineState,
    pl_add_bias: MetalComputePipelineState,
    pl_mul: MetalComputePipelineState,
    pl_silu: MetalComputePipelineState,
    pl_swiglu: MetalComputePipelineState,
    pl_rope: MetalComputePipelineState,
    pl_gqa_attn: MetalComputePipelineState,
    pl_gqa_attn_f16: MetalComputePipelineState,
    pl_gqa_attn_partial: MetalComputePipelineState,
    pl_gqa_attn_partial_f16: MetalComputePipelineState,
    pl_gqa_attn_combine: MetalComputePipelineState,
    pl_flash_attn: MetalComputePipelineState,
    pl_flash_attn_f16: MetalComputePipelineState,
    pl_flash_attn_hd128: MetalComputePipelineState,
    pl_flash_attn_hd128_f16: MetalComputePipelineState,
    pl_flash_attn_blk: MetalComputePipelineState,
    pl_flash_attn_blk_f16: MetalComputePipelineState,
    pl_flash_attn_blk_hd128: MetalComputePipelineState,
    pl_flash_attn_blk_hd128_f16: MetalComputePipelineState,
    pl_kv_tail_pad: MetalComputePipelineState,
    pl_store_kv: MetalComputePipelineState,
    pl_store_kv_f16: MetalComputePipelineState,
    pl_attn_bsr: MetalComputePipelineState,
    pl_attn_rope_store: MetalComputePipelineState,
    pl_attn_scores: MetalComputePipelineState,
    pl_attn_output: MetalComputePipelineState,
    pl_softmax_attn: MetalComputePipelineState,
    pl_warmup: MetalComputePipelineState,
    // (buffer, byte-offset): weights live either in a per-weight copied buffer
    // (offset 0) or — since the 2026-08-21 mmap loader — as offsets into a
    // page-aligned NoCopy buffer over the mmap'd GGUF part (llama-style,
    // ggml-metal-device.m:1668; newBufferWithBytesNoCopy requires a page-aligned
    // base, so per-tensor offsets are passed via setBuffer:offset:).
    weights: std::sync::Mutex<std::collections::HashMap<String, (MetalBuffer, u64)>>,
    // Registered mmap'd GGUF parts: (base_ptr, len, Metal buffer). register_weight
    // resolves a weight slice to (buffer, offset) by pointer-range containment.
    mmap_parts: std::sync::Mutex<Vec<(usize, usize, MetalBuffer)>>,
    // Scratch for the mmap-part warmup dispatch (register_part).
    buf_positions: std::sync::Mutex<MetalBuffer>,
    buf_attn_partial: std::sync::Mutex<MetalBuffer>,
    // Prefill parallel-attention scratch (P1 2026-08-11): scores [nt][nh][nkv].
    buf_attn_scores: std::sync::Mutex<MetalBuffer>,
    // Flash-prefill tail pad (2026-08-14): [2][64][nkt] f32/f16 K-tail + V-tail.
    buf_attn_pad: std::sync::Mutex<MetalBuffer>,
    // Ring of recent dispatch op labels (for GPU-fault diagnosis, MINFER_TRACE only).
    dispatch_trace: std::sync::Mutex<std::collections::VecDeque<String>>,
}

// ─── MpsCommandBuffer: batch multiple ops in one GPU submission ──────

#[cfg(target_os = "macos")]
pub struct MpsCommandBuffer<'a> {
    state: &'a MpsStateInner,
    // The metal crate returns AUTORELEASED objects from `commandBuffer` /
    // `newComputeCommandEncoder` (not `new`). cmd_buffer() retains them (and
    // Drop releases), so the objects survive the creating thread's
    // autorelease-pool drain — required whenever a command buffer is created on
    // a background thread (parallel encoding) or handed across threads.
    cmd_buf: MetalCommandBuffer,
    enc: MetalComputeCommandEncoder,
    /// Whether the compute encoder is still open. `encode_captures` (P2/P3
    /// staging blits) ends it and encodes a blit pass; `submit` must then skip
    /// the (now illegal) second `end_encoding`.
    enc_open: bool,
}

#[cfg(target_os = "macos")]
#[cfg(target_os = "macos")]
impl MpsCommandBuffer<'_> {
    /// Record the current dispatch op label (only when MINFER_TRACE=1, so normal
    /// encode speed is unaffected). Used to print the faulting kernel on a
    /// Metal command-buffer error / timeout.
    fn trace_op(&self, op: &str) {
        static TRACE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*TRACE.get_or_init(|| std::env::var("MINFER_TRACE").is_ok()) {
            return;
        }
        let mut t = self.state.dispatch_trace.lock().unwrap();
        t.push_back(op.to_string());
        if t.len() > 16 {
            t.pop_front();
        }
    }

    fn set_params(&self, idx: u64, val: &i32) {
        unsafe {
            self.enc.setBytes_length_atIndex(
                NonNull::new(val as *const i32 as *const std::ffi::c_void as *mut c_void).unwrap(),
                (std::mem::size_of::<i32>() as u64) as usize,
                (idx) as usize,
            )
        };
    }

    /// GPU memory barrier (2026-08-19 fix). Metal does NOT guarantee
    /// write visibility between dispatches in a single compute command encoder;
    /// without an explicit barrier, a kernel that reads a buffer written by a
    /// preceding dispatch can race with that dispatch's last threadgroups,
    /// intermittently corrupting the tail rows (observed: last-2 token slots of
    /// layer0 bn on 1.5B/7B prefill). llama.cpp's Metal backend inserts the same
    /// barrier after every op. MTLBarrierScopeBuffers = 1 << 0.
    fn barrier(&self) {
        self.enc.memoryBarrierWithScope(MTLBarrierScope::Buffers);
    }

    /// End the compute pass. Call before encoding blits from the same command
    /// buffer (Metal allows only one active encoder at a time); `submit` then
    /// skips its own `end_encoding`.
    pub fn end_compute(&mut self) {
        if self.enc_open {
            self.enc.endEncoding();
            self.enc_open = false;
        }
    }

    /// Encode GPU→host staging copies (P2/P3 live/trace capture) into this
    /// command buffer, AFTER all kernels of the split. The destinations' data
    /// is valid once the command buffer is submitted (`synchronize`).
    pub fn encode_captures(
        &mut self,
        pairs: &[(usize, usize)],
        buffers: &[MetalBuffer],
        staging: &[MetalBuffer],
    ) -> Result<(), String> {
        if pairs.is_empty() {
            return Ok(());
        }
        self.end_compute();
        let blit_ref = self.cmd_buf.blitCommandEncoder().expect("blit encoder");
        for &(src, dst) in pairs {
            let src_buf = buffers
                .get(src)
                .ok_or_else(|| format!("capture: no buffer {src}"))?;
            let dst_buf = staging
                .get(dst)
                .ok_or_else(|| format!("capture: no staging {dst}"))?;
            let len = src_buf.length();
            unsafe {
                blit_ref.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                    &**src_buf, 0, &**dst_buf, 0, len,
                );
            }
        }
        blit_ref.endEncoding();
        Ok(())
    }

    fn dispatch_2d(&self, w: u64, h: u64, tw: u64, th: u64) {
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (w) as usize,
                height: (h) as usize,
                depth: (1) as usize,
            },
            MTLSize {
                width: (tw) as usize,
                height: (th) as usize,
                depth: (1) as usize,
            },
        );
        self.barrier();
    }

    fn dispatch_3d(&self, w: u64, h: u64, d: u64, tw: u64, th: u64, td: u64) {
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: (w) as usize,
                height: (h) as usize,
                depth: (d) as usize,
            },
            MTLSize {
                width: (tw) as usize,
                height: (th) as usize,
                depth: (td) as usize,
            },
        );
        self.barrier();
    }

    /// GEMM kernels (prefill nt>=16) are enabled unless MINFER_GEMM=0.
    fn gemm_enabled() -> bool {
        static GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GEMM.get_or_init(|| std::env::var("MINFER_GEMM").map_or(true, |v| v != "0"))
    }

    /// Dispatch a 64×32-tile simdgroup GEMM (NT≥16 prefill). GPU safety: the
    /// kernels stage 8 KB of threadgroup memory (sa 4 KB + sb 2 KB + bc_out
    /// 8 KB reusing sa/sb) — verified against the queried device limit.
    fn gemm_dispatch(
        &self,
        pl: &MetalComputePipelineState,
        wb: &MetalBuffer,
        w_off: u64,
        x: &MetalBuffer,
        x_off: u64,
        out: &MetalBuffer,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        if 8192 > self.state.max_threadgroup_memory {
            gpu_abort(&format!(
                "GEMM needs 8192 B threadgroup memory, device max is {} B",
                self.state.max_threadgroup_memory
            ));
        }
        self.enc.setComputePipelineState(&**(pl));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(wb)), (w_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (x_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
        };
        let mm_p = [od as i32, id as i32, nt as i32];
        unsafe {
            self.enc.setBytes_length_atIndex(
                NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void).unwrap(),
                (12) as usize,
                (3) as usize,
            )
        };
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((8192) as usize, (0) as usize)
        };
        self.dispatch_2d(((nt + 31) / 32) as u64, ((od + 63) / 64) as u64, 32, 4);
    }

    fn dispatch_1d(&self, n: u64, tg: u64) {
        self.enc.dispatchThreadgroups_threadsPerThreadgroup(
            MTLSize {
                width: ((n + tg - 1) / tg) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            },
            MTLSize {
                width: (tg) as usize,
                height: (1) as usize,
                depth: (1) as usize,
            },
        );
        self.barrier();
    }

    pub fn quant_matmul_f32_on_gpu_buf(
        &self,
        wb: &MetalBuffer,
        w_off: u64,
        ttype: TensorType,
        x: &MetalBuffer,
        x_off: u64,
        out: &MetalBuffer,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        self.trace_op("matmul");
        // GPU safety (M1): the K-quant (super-block) kernels index weights by
        // K/256 super-blocks (floor). A non-256-aligned id silently drops the
        // remainder (wrong results, not a fault) — refuse rather than risk it.
        if matches!(
            ttype,
            TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K
        ) && id % 256 != 0
        {
            gpu_abort(&format!(
                "matmul input dim id={id} is not 256-aligned for {ttype:?} (K-quant kernels use K/256 super-block floor)"
            ));
        }
        match ttype {
            TensorType::Q8_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q8_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q8_0_f32_multi
                        } else {
                            &self.state.pl_q8_0_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    const NW: u64 = 32;
                    const NSG: u64 = 4;
                    const NR0: u64 = 2;
                    const TG_MEM: u64 = NW * NR0 * std::mem::size_of::<f32>() as u64; // 256 bytes
                    unsafe {
                        self.enc
                            .setThreadgroupMemoryLength_atIndex((TG_MEM) as usize, (0) as usize)
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 1) / 2) as u64, grid_y, NW, NSG);
                }
            }
            TensorType::Q4_K | TensorType::Q6_K => {
                // Q6_K has a simdgroup GEMM (super-block); Q4_K still falls back
                // to the scalar f32 multi (no Q4_K in the shipped 0.5B K_M models).
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    // both Q4_K and Q6_K have simdgroup GEMMs
                    let pl = if ttype == TensorType::Q6_K {
                        &self.state.pl_q6_k_mm_f32
                    } else {
                        &self.state.pl_q4_k_mm_f32
                    };
                    self.gemm_dispatch(pl, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    let pl: &MetalComputePipelineState = if ttype == TensorType::Q4_K {
                        if nt > 1 {
                            &self.state.pl_q4_k_f32_multi
                        } else {
                            &self.state.pl_q4_k_f32
                        }
                    } else {
                        if nt > 1 {
                            &self.state.pl_q6_k_f32_multi
                        } else {
                            &self.state.pl_q6_k_f32
                        }
                    };
                    self.enc.setComputePipelineState(&**(pl));
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    // Q6_K/Q4_K: llama's kernel_mul_mv_q6_K/q4_K_f32_impl use
                    // TG(32, nsg=2); the stride-2 (q6_K) / stride-4 (q4_K) thread
                    // layout keeps all threads busy for small id (nb super-blocks),
                    // unlike the old stride-64 scalar loop.
                    if ttype == TensorType::Q6_K || ttype == TensorType::Q4_K {
                        self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 32, 2);
                    } else {
                        self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 64, 1);
                    }
                }
            }
            TensorType::Q4_1 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q4_1_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q4_1_f32_multi
                        } else {
                            &self.state.pl_q4_1_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q5_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q5_0_f32_multi
                        } else {
                            &self.state.pl_q5_0_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_1 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q5_1_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q5_1_f32_multi
                        } else {
                            &self.state.pl_q5_1_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_K => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q5_k_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q5_k_f32_multi
                        } else {
                            &self.state.pl_q5_k_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q4_0 => {
                // Prefill uses the simdgroup GEMM (faithful llama.cpp port, float
                // accumulation). MINFER_GEMM=0 disables it (f32 multi fallback) for
                // A/B comparison. GEMM wins for nt >= ~16 (fixed dispatch overhead
                // dominates for tiny prefills).
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(
                        &self.state.pl_q4_0_mm_f32,
                        wb,
                        w_off,
                        x,
                        x_off,
                        out,
                        od,
                        id,
                        nt,
                    );
                } else {
                    self.enc.setComputePipelineState(
                        &**(if nt > 1 {
                            &self.state.pl_q4_0_f32_multi
                        } else {
                            &self.state.pl_q4_0_f32
                        }),
                    );
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(wb)),
                            (w_off) as usize,
                            (0) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(x)),
                            (x_off) as usize,
                            (1) as usize,
                        )
                    };
                    unsafe {
                        self.enc.setBuffer_offset_atIndex(
                            Some(&**(out)),
                            (0) as usize,
                            (2) as usize,
                        )
                    };
                    let mm_p = [od as i32, id as i32, nt as i32];
                    unsafe {
                        self.enc.setBytes_length_atIndex(
                            NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                                .unwrap(),
                            (12) as usize,
                            (3) as usize,
                        )
                    };
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            _ => {
                self.enc.setComputePipelineState(
                    &**(if nt > 1 {
                        &self.state.pl_q4_0_f32_multi
                    } else {
                        &self.state.pl_q4_0_f32
                    }),
                );
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(wb)), (w_off) as usize, (0) as usize)
                };
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(x)), (x_off) as usize, (1) as usize)
                };
                unsafe {
                    self.enc
                        .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
                };
                let mm_p = [od as i32, id as i32, nt as i32];
                unsafe {
                    self.enc.setBytes_length_atIndex(
                        NonNull::new(mm_p.as_ptr() as *const std::ffi::c_void as *mut c_void)
                            .unwrap(),
                        (12) as usize,
                        (3) as usize,
                    )
                };
                let grid_y = if nt > 1 { 1 } else { nt as u64 };
                self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
            }
        }
    }

    /// Choose the f32-activation matmul for all weight types (including Q4_0,
    /// matching llama.cpp's Metal backend which does not Q8_0-quantize activations).
    /// Pre-looked-up weight buffer and type — avoids per-matmul HashMap locking.
    /// (Only exercised by the `matmul_bandwidth_profile` test; the graph backend
    /// calls `quant_matmul_f32_on_gpu_buf` directly.)
    #[allow(dead_code)]
    fn matmul_on_gpu_buf(
        &self,
        wb: &MetalBuffer,
        w_off: u64,
        ttype: TensorType,
        _q8_x: &MetalBuffer,
        f32_x: &MetalBuffer,
        x_off: u64,
        out: &MetalBuffer,
        od: usize,
        id: usize,
        nt: usize,
    ) {
        self.quant_matmul_f32_on_gpu_buf(wb, w_off, ttype, f32_x, x_off, out, od, id, nt);
    }

    /// GPU embedding lookup: dequantize Q4_0 embedding rows for nt token ids.
    /// Writes f32 hidden state [nt][ne] to dst (buf_hidden).
    pub fn embed_tokens_gpu(
        &self,
        wb: &MetalBuffer,
        w_off: u64,
        ids: &MetalBuffer,
        dst: &MetalBuffer,
        ne: usize,
        nt: usize,
        ttype: TensorType,
    ) {
        self.trace_op("embed");
        let (pl, nb) = match ttype {
            TensorType::Q4_0 => (&self.state.pl_get_rows_q4_0, ne / 32),
            TensorType::Q4_1 => (&self.state.pl_get_rows_q4_1, ne / 32),
            TensorType::Q5_0 => (&self.state.pl_get_rows_q5_0, ne / 32),
            TensorType::Q5_1 => (&self.state.pl_get_rows_q5_1, ne / 32),
            TensorType::Q8_0 => (&self.state.pl_get_rows_q8_0, ne / 32),
            TensorType::Q4_K => (&self.state.pl_get_rows_q4_k, (ne / 256) * 16),
            TensorType::Q6_K => (&self.state.pl_get_rows_q6_k, (ne / 256) * 16),
            TensorType::Q5_K => (&self.state.pl_get_rows_q5_k, (ne / 256) * 16),
            _ => unreachable!("embed_tokens_gpu called with unsupported type {ttype:?}"),
        };
        self.enc.setComputePipelineState(&**(pl));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(wb)), (w_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(ids)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(ne as i32));
        self.set_params(4, &(nt as i32));
        self.dispatch_1d((nt * nb) as u64, 256);
    }

    /// Generic f32 row selection: out[t] = x[ids[t]] (graph n_out tail rows).
    pub fn get_rows_f32(
        &self,
        x: &MetalBuffer,
        ids: &MetalBuffer,
        out: &MetalBuffer,
        ne: usize,
        nt: usize,
    ) {
        self.trace_op("get_rows_f32");
        self.enc
            .setComputePipelineState(&*self.state.pl_get_rows_f32);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(ids)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(ne as i32));
        self.dispatch_2d(nt as u64, ne as u64, 1, 1);
    }

    /// RMSNorm: y = x * rsqrt(mean(x²)+eps) * w
    pub fn rms_norm(
        &self,
        x: &MetalBuffer,
        w: Option<&MetalBuffer>,
        w_off: u64,
        y: &MetalBuffer,
        d: usize,
        n: usize,
        eps: f32,
        off: u64,
        y_off: u64,
    ) {
        self.trace_op("rms_norm");
        self.enc.setComputePipelineState(&*self.state.pl_rms_norm);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (off) as usize, (0) as usize)
        };
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(w.unwrap_or(y))),
                (w_off) as usize,
                (1) as usize,
            )
        }; // dummy if no weight
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (y_off) as usize, (2) as usize)
        };
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        self.dispatch_2d(n as u64, 1, 32, 1);
    }

    /// RMSNorm with a 256-thread multi-simdgroup kernel (P1 2026-08-10, llama
    /// transcription). Same math as rms_norm but the threadgroup is 256 threads
    /// so a single 896-element row isn't DRAM-latency-bound (the 32-thread
    /// kernel measured ~7x the per-dispatch cost of 256-thread elementwise ops).
    /// Requires a threadgroup buffer of n_simdgroups floats (8 for 256 threads).
    pub fn rms_norm_256(
        &self,
        x: &MetalBuffer,
        w: Option<&MetalBuffer>,
        w_off: u64,
        y: &MetalBuffer,
        d: usize,
        n: usize,
        eps: f32,
        off: u64,
        y_off: u64,
    ) {
        self.trace_op("rms_norm");
        self.enc
            .setComputePipelineState(&*self.state.pl_rms_norm_256);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (off) as usize, (0) as usize)
        };
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(w.unwrap_or(y))),
                (w_off) as usize,
                (1) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (y_off) as usize, (2) as usize)
        };
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((32 * 4) as usize, (0) as usize)
        };
        // 256 threads = 8 simdgroups; one threadgroup per row.
        self.dispatch_2d(n as u64, 1, 32, 8);
    }

    /// Element-wise add: z = x + y
    pub fn add_f32(&self, x: &MetalBuffer, y: &MetalBuffer, z: &MetalBuffer, n: usize) {
        self.add_f32_off(x, y, z, n, 0, 0, 0);
    }

    /// Element-wise add with per-buffer byte offsets (last-layer output-rows
    /// reduction: x/z read/write the tail n_out rows of `hidden`, y starts at 0).
    pub fn add_f32_off(
        &self,
        x: &MetalBuffer,
        y: &MetalBuffer,
        z: &MetalBuffer,
        n: usize,
        x_off: u64,
        y_off: u64,
        z_off: u64,
    ) {
        self.trace_op("add");
        self.enc.setComputePipelineState(&*self.state.pl_add);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (x_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (y_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(z)), (z_off) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        // float4 kernel: 4 elements/thread (ceil for the scalar tail)
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// Add 1-D bias to rows: y[t][i] += b[i]. `off` = element offset into `y`
    /// (used by the fused QKV path to bias the q/k/v sections of one buffer).
    pub fn add_bias_f32(
        &self,
        y: &MetalBuffer,
        b: &MetalBuffer,
        b_off: u64,
        d: usize,
        n: usize,
        off: usize,
    ) {
        self.trace_op("bias");
        self.enc.setComputePipelineState(&*self.state.pl_add_bias);
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(y)),
                ((off * 4) as u64) as usize,
                (0) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(b)), (b_off) as usize, (1) as usize)
        };
        self.set_params(2, &(d as i32));
        // float4 kernel: 4 dims/thread
        self.dispatch_2d(n as u64, ((d as u64) + 3) / 4, 1, 64);
    }

    /// Element-wise multiply: z = x * y
    pub fn mul_f32(&self, x: &MetalBuffer, y: &MetalBuffer, z: &MetalBuffer, n: usize) {
        self.enc.setComputePipelineState(&*self.state.pl_mul);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(x)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(z)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// SiLU in-place: y = y / (1 + exp(-y))
    pub fn silu_f32(&self, y: &MetalBuffer, n: usize) {
        self.enc.setComputePipelineState(&*self.state.pl_silu);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(y)), (0) as usize, (0) as usize)
        };
        self.set_params(1, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// SwiGLU fused: dst = silu(gate) * up  (dst may alias gate)
    pub fn swiglu_f32(&self, gate: &MetalBuffer, up: &MetalBuffer, dst: &MetalBuffer, n: usize) {
        self.trace_op("swiglu");
        self.enc.setComputePipelineState(&*self.state.pl_swiglu);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(gate)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(up)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// SwiGLU over a fused gate+up buffer: gate at offset 0, up at `up_off`
    /// elements (fused FFN gate+up path). Writes silu(gate)*up back to gate.
    pub fn swiglu_f32_off(
        &self,
        gate: &MetalBuffer,
        up: &MetalBuffer,
        dst: &MetalBuffer,
        n: usize,
        up_off: usize,
    ) {
        self.trace_op("swiglu");
        self.enc.setComputePipelineState(&*self.state.pl_swiglu);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(gate)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(up)),
                ((up_off * 4) as u64) as usize,
                (1) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(n as i32));
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// RoPE (in-place): x layout [nt][n_head][n_dims]. `off` = element offset
    /// into `x` (fused QKV: K section lives mid-buffer).
    /// rope_style: 0 = non-interleaved (Qwen2), 1 = interleaved (LLaMA).
    pub fn rope_f32(
        &self,
        x: &MetalBuffer,
        n_head: usize,
        n_dims: usize,
        nt: usize,
        freq_base: f32,
        freq_scale: f32,
        positions: &MetalBuffer,
        rope_style: i32,
        off: usize,
    ) {
        self.trace_op("rope");
        self.enc.setComputePipelineState(&*self.state.pl_rope);
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(x)),
                ((off * 4) as u64) as usize,
                (0) as usize,
            )
        };
        self.set_params(1, &(n_head as i32));
        self.set_params(2, &(n_dims as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(freq_base.to_bits() as i32));
        self.set_params(5, &(freq_scale.to_bits() as i32));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (6) as usize)
        };
        self.set_params(7, &rope_style);
        // P7: one thread per (dim, head, token) instead of one per (token, head)
        self.dispatch_3d((n_dims / 2) as u64, n_head as u64, nt as u64, 1, 1, 1);
    }

    /// Flash Attention: one threadgroup per (token, KV_head), tiled K/V
    /// with online softmax. Each simdgroup processes one query head.
    /// K/V tiles loaded into threadgroup-shared memory, reused by all
    /// query heads in the GQA group.
    pub fn gqa_attn_f32(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
    ) {
        self.gqa_attn_f32_off(q, 0, k, 0, v, 0, o, positions, nh, nk, hd, scale, nt);
    }

    /// Offset variant of `gqa_attn_f32` — K/V may live at byte offsets inside a
    /// shared buffer (the graph backend's `[K | V]` contiguous KV region).
    pub fn gqa_attn_f32_off(
        &self,
        q: &MetalBuffer,
        q_off: u64,
        k: &MetalBuffer,
        k_off: u64,
        v: &MetalBuffer,
        v_off: u64,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
    ) {
        self.trace_op("gqa_attn");
        let gqa = nh / nk;
        self.enc.setComputePipelineState(
            &**(if kv_cache_is_f16() {
                &self.state.pl_gqa_attn_f16
            } else {
                &self.state.pl_gqa_attn
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (q_off) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (k_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (v_off) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_2d(nt as u64, nk as u64, 32, gqa as u64);
    }

    /// KV-parallel split attention for nt==1 decode (the classic kernel's grid
    /// is only (1, nk) threadgroups that loop the KV sequentially — the measured
    /// #1 decode bottleneck). Two passes: partial per KV chunk (grid (nt,nk,P)),
    /// then combine (grid (nt,nh)). Requires the partials buffer (`buf_attn_partial`)
    /// sized for nt*nh*P*(2+hd) floats, grown on demand here.
    pub fn gqa_attn_split_f32(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        n_chunks: usize,
    ) {
        self.trace_op("gqa_attn_split");
        let gqa = nh / nk;
        let need = (nt * nh * n_chunks * (2 + hd) * 4) as u64;
        let partial = MpsState::get_or_grow(&self.state.buf_attn_partial, need, &self.state.device);

        // pass 1: partials per (token, KV_head, chunk) — f16 cache picks the
        // f16 partial kernel (K/V read as half, staged to f32 float4 tiles).
        self.enc.setComputePipelineState(
            &**(if kv_cache_is_f16() {
                &self.state.pl_gqa_attn_partial_f16
            } else {
                &self.state.pl_gqa_attn_partial
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        self.set_params(10, &(n_chunks as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_3d(nt as u64, nk as u64, n_chunks as u64, 32, gqa as u64, 1);

        // pass 2: combine
        self.enc
            .setComputePipelineState(&*self.state.pl_gqa_attn_combine);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nh as i32));
        self.set_params(3, &(hd as i32));
        self.set_params(4, &(nt as i32));
        self.set_params(5, &(n_chunks as i32));
        self.dispatch_2d(nt as u64, nh as u64, 32, 1);
    }

    /// Flash-attention port (llama kernel_flash_attn_ext_vec, NSG=1 fixed
    /// DK=DV=64/NE=2/C=32 shape) for nt==1 decode. Replaces the split pair with
    /// a single-simdgroup-per-(t,h,iwg) kernel whose Q*K^T reduce is
    /// shuffle-based (simd_shuffle_down 8,4,2,1 + broadcast) instead of
    /// threadgroup barriers — llama's structural advantage over the split
    /// attention (~7-10x isolated at nkv=430). Output partials are {M,S,O[hd]}
    /// in the SAME layout as kernel_gqa_attn_partial_f32, so the shared combine
    /// kernel merges them unchanged. Grid (nt, nh, n_chunks), 32 threads.
    /// Host guard: layer_gpu only dispatches this when hd==64 (fixed DK/DV);
    /// otherwise the split path is used.
    pub fn gqa_attn_flash(
        &self,
        q: &MetalBuffer,
        k: &MetalBuffer,
        v: &MetalBuffer,
        o: &MetalBuffer,
        positions: &MetalBuffer,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
        nt: usize,
        n_chunks: usize,
    ) {
        self.trace_op("gqa_attn_flash");
        let need = (nt * nh * n_chunks * (2 + hd) * 4) as u64;
        let partial = MpsState::get_or_grow(&self.state.buf_attn_partial, need, &self.state.device);

        // pass 1: flash partials — f16 cache reads the half K/V directly.
        self.enc.setComputePipelineState(
            &**(match (kv_cache_is_f16(), hd) {
                (false, 128) => &self.state.pl_flash_attn_hd128,
                (true, 128) => &self.state.pl_flash_attn_hd128_f16,
                (false, _) => &self.state.pl_flash_attn,
                (true, _) => &self.state.pl_flash_attn_f16,
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        self.set_params(10, &(n_chunks as i32));
        // shmem (hd=64): sq4 (16 float4 = 256 B) | ss (32 f32 = 128 B) | so4 (32 float4 = 512 B) = 896 → 1024
        // shmem (hd=128): sq4 (32 float4 = 512 B) | ss (32 f32 = 128 B) | so4 (32 float4 = 512 B) = 1152
        let shmem = if hd == 128 { 1152 } else { 1024 };
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_3d(nt as u64, nh as u64, n_chunks as u64, 32, 1, 1);

        // pass 2: combine (shared with the split path)
        self.enc
            .setComputePipelineState(&*self.state.pl_gqa_attn_combine);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*partial), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(o)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nh as i32));
        self.set_params(3, &(hd as i32));
        self.set_params(4, &(nt as i32));
        self.set_params(5, &(n_chunks as i32));
        self.dispatch_2d(nt as u64, nh as u64, 32, 1);
    }

    /// Scatter nt rows of src[nt][nkt] into dst[positions[t]][nkt].
    /// Writes f32 (default) or f16 (MINFER_CACHE_TYPE=f16) into the KV cache.
    pub fn store_kv(
        &self,
        src: &MetalBuffer,
        dst: &MetalBuffer,
        nkt: usize,
        nt: usize,
        positions: &MetalBuffer,
        off: usize,
    ) {
        self.trace_op("store_kv");
        self.enc.setComputePipelineState(
            &**(if kv_cache_is_f16() {
                &self.state.pl_store_kv_f16
            } else {
                &self.state.pl_store_kv
            }),
        );
        unsafe {
            self.enc.setBuffer_offset_atIndex(
                Some(&**(src)),
                ((off * 4) as u64) as usize,
                (0) as usize,
            )
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(dst)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nkt as i32));
        self.set_params(3, &(nt as i32));
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (4) as usize)
        };
        self.dispatch_2d(nt as u64, nkt as u64, 1, 1);
    }

    /// Prefill parallel attention (P1 2026-08-11): replaces the classic
    /// latency-bound attention kernel for nt>1 (grid (nt,nk), sequential KV loop
    /// with ~24K barriers at nt=430 → ~100ms, 48% of prefill, ~25x llama's).
    /// This 3-pass replacement is fully parallel (no threadgroup barriers):
    ///   1. scores[t][h][kv] = dot(q[t][h][0..hd], k[kv][hk*hd..]) * scale
    ///   2. masked softmax over kv per (t,h) row
    ///   3. out[t][h][0..hd] = Σ_kv softmax[t][h][kv] * v[kv][hk*hd..]
    /// q: [nt][nqt], kv_k/kv_v: [nkv][nkt], out: [nt][nqt]. nkv = real KV length
    /// (max_pos+1); the scores buffer is [nt][nh][nkv] (no padding needed — all
    /// three kernels handle arbitrary nkv).
    pub fn attn_parallel_prefill(
        &self,
        q: &MetalBuffer,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        out: &MetalBuffer,
        positions: &MetalBuffer,
        nkv: usize,
        nkt: usize,
        _nqt: usize,
        nt: usize,
        nh: usize,
        hd: usize,
        gqa: usize,
        scale: f32,
    ) {
        self.trace_op("attn_parallel");
        let dev = &self.state.device;
        let scores =
            MpsState::get_or_grow(&self.state.buf_attn_scores, (nt * nh * nkv * 4) as u64, dev);

        // pass 1: scores [nt*nh][nkv] — one 256-thread TG per (t,h) row
        self.enc
            .setComputePipelineState(&*self.state.pl_attn_scores);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*scores), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(nh as i32));
        self.set_params(4, &(hd as i32));
        self.set_params(5, &(nkv as i32));
        self.set_params(6, &(nt as i32));
        self.set_params(7, &(gqa as i32));
        self.set_params(8, &(nkt as i32));
        self.set_params(9, &(scale.to_bits() as i32));
        self.dispatch_2d((nt * nh) as u64, 1, 256, 1);

        // pass 2: masked softmax over kv per (t,h) row
        self.enc
            .setComputePipelineState(&*self.state.pl_softmax_attn);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*scores), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (1) as usize)
        };
        self.set_params(2, &(nkv as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(nh as i32));
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((32 * 4) as usize, (0) as usize)
        };
        self.dispatch_2d((nt * nh) as u64, 1, 32, 8);

        // pass 3: out = softmax · V — one 256-thread TG per (t,h) row
        self.enc
            .setComputePipelineState(&*self.state.pl_attn_output);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*scores), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(nh as i32));
        self.set_params(4, &(hd as i32));
        self.set_params(5, &(nkv as i32));
        self.set_params(6, &(nt as i32));
        self.set_params(7, &(gqa as i32));
        self.set_params(8, &(nkt as i32));
        self.dispatch_2d((nt * nh) as u64, 1, 256, 1);
    }

    /// Prefill flash attention (2026-08-14, llama kernel_flash_attn_ext_blk port):
    /// ONE kernel replaces the 3-pass parallel attention for nt>1 (measured 46 ms
    /// of 135 ms prefill GPU vs llama's ~3 ms). Fixed-shape NSG=4/Q=8/C=64/
    /// DK=DV=64: grid (ceil(nt/8), nh) of 128-thread threadgroups (32 lanes × 4
    /// simdgroups), each computing Q=8 query tokens × ALL KV for head h via
    /// simdgroup_matrix QK^T + online softmax + PV with an inline causal mask.
    /// GQA head hk = h/gqa is baked into the K/V base inside the kernel.
    /// The host copies the last partial KV block (nkv % 64 != 0) into a
    /// [2][64][nkt] tail-pad buffer first (kernel_kv_tail_pad); padded rows are
    /// zero + masked, so a pad buffer is always bound but only populated then.
    pub fn attn_flash_prefill(
        &self,
        q: &MetalBuffer,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        out: &MetalBuffer,
        positions: &MetalBuffer,
        nkv: usize,
        nkt: usize,
        nt: usize,
        nh: usize,
        nk: usize,
        hd: usize,
        scale: f32,
    ) {
        self.trace_op("attn_flash_blk");
        let dev = &self.state.device;
        let f16 = kv_cache_is_f16();
        let elem = if f16 { 2u64 } else { 4u64 };
        let pad =
            MpsState::get_or_grow(&self.state.buf_attn_pad, (2 * 64 * nkt as u64) * elem, dev);

        if nkv % 64 != 0 {
            self.enc
                .setComputePipelineState(&*self.state.pl_kv_tail_pad);
            unsafe {
                self.enc
                    .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (0) as usize)
            };
            unsafe {
                self.enc
                    .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (1) as usize)
            };
            unsafe {
                self.enc
                    .setBuffer_offset_atIndex(Some(&*pad), (0) as usize, (2) as usize)
            };
            self.set_params(3, &(nkv as i32));
            self.set_params(4, &(nkt as i32));
            self.set_params(5, &(if f16 { 1 } else { 0 }));
            self.dispatch_2d(nkt as u64, 64, 1, 1);
        }

        self.enc.setComputePipelineState(
            &**(if f16 {
                if hd == 128 {
                    &self.state.pl_flash_attn_blk_hd128_f16
                } else {
                    &self.state.pl_flash_attn_blk_f16
                }
            } else {
                if hd == 128 {
                    &self.state.pl_flash_attn_blk_hd128
                } else {
                    &self.state.pl_flash_attn_blk
                }
            }),
        );
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(q)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&*pad), (0) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(out)), (0) as usize, (4) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(positions)), (0) as usize, (5) as usize)
        };
        self.set_params(6, &(nh as i32));
        self.set_params(7, &(nk as i32));
        self.set_params(8, &(hd as i32));
        self.set_params(9, &(scale.to_bits() as i32));
        self.set_params(10, &(nt as i32));
        self.set_params(11, &(nkv as i32));
        // shmem: hd=64: sq (512 half = 1024 B) | so (512 f32 = 2048 B) | ss (1024 f32 = 4096 B);
        //        hd=128: sq (1024 half = 2048 B) | so (1024 f32 = 4096 B) | ss (1024 f32 = 4096 B)
        let shmem = if hd == 128 { 10240u64 } else { 7168u64 };
        unsafe {
            self.enc
                .setThreadgroupMemoryLength_atIndex((shmem) as usize, (0) as usize)
        };
        self.dispatch_2d(((nt + 7) / 8) as u64, nh as u64, 32, 4);
    }

    /// Fused bias-add + RoPE + KV-store for nt==1 decode: ONE kernel replaces
    /// add_bias×3 + rope×2 + store_kv×2 (7 dispatches). `bqkv` layout is
    /// [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt]; biases are the raw
    /// per-section buffers. `pos` = the single token position. The KV store
    /// writes f32 or f16 (per kv_cache_is_f16) into kv_k/kv_v.
    pub fn attn_bias_rope_store(
        &self,
        bqkv: &MetalBuffer,
        bias_q: &MetalBuffer,
        bq_off: u64,
        bias_k: &MetalBuffer,
        bk_off: u64,
        bias_v: &MetalBuffer,
        bv_off: u64,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        pos: i32,
        rope_style: i32,
    ) {
        self.trace_op("attn_bias_rope_store");
        self.enc.setComputePipelineState(&*self.state.pl_attn_bsr);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bqkv)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bias_q)), (bq_off) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bias_k)), (bk_off) as usize, (2) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bias_v)), (bv_off) as usize, (3) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (4) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (5) as usize)
        };
        self.set_params(6, &(nqt as i32));
        self.set_params(7, &(nkt as i32));
        self.set_params(8, &(hd as i32));
        self.set_params(9, &(freq_base.to_bits() as i32));
        self.set_params(10, &(freq_scale.to_bits() as i32));
        self.set_params(11, &pos);
        self.set_params(12, &rope_style);
        self.set_params(13, &(if kv_cache_is_f16() { 1 } else { 0 }));
        let grid = nqt / 2 + nkt / 2 + nkt;
        self.dispatch_1d(grid as u64, 256);
    }

    /// Fused decode QKV rope+store WITHOUT attention biases (Qwen3): the concat
    /// buffer `bqkv` holds q|k|v (q = rows 0..nqt, k = nqt..nqt+nkt,
    /// v = nqt+nkt..), and the per-head Q/K RMSNorm was already applied in place
    /// by the preceding rms_norm_256 dispatches. This kernel only RoPEs q in
    /// place, RoPEs + stores K, and stores V into the persistent KV regions.
    /// Grid: nqt/2 + nkt/2 + nkt, 256 threads. Buffer indices: 0=bqkv, 1=kv_k,
    /// 2=kv_v; params 3..=10.
    pub fn attn_rope_store(
        &self,
        bqkv: &MetalBuffer,
        kv_k: &MetalBuffer,
        kv_v: &MetalBuffer,
        nqt: usize,
        nkt: usize,
        hd: usize,
        freq_base: f32,
        freq_scale: f32,
        pos: i32,
        rope_style: i32,
    ) {
        self.trace_op("attn_rope_store");
        self.enc
            .setComputePipelineState(&*self.state.pl_attn_rope_store);
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(bqkv)), (0) as usize, (0) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_k)), (0) as usize, (1) as usize)
        };
        unsafe {
            self.enc
                .setBuffer_offset_atIndex(Some(&**(kv_v)), (0) as usize, (2) as usize)
        };
        self.set_params(3, &(nqt as i32));
        self.set_params(4, &(nkt as i32));
        self.set_params(5, &(hd as i32));
        self.set_params(6, &(freq_base.to_bits() as i32));
        self.set_params(7, &(freq_scale.to_bits() as i32));
        self.set_params(8, &pos);
        self.set_params(9, &rope_style);
        self.set_params(10, &(if kv_cache_is_f16() { 1 } else { 0 }));
        let grid = nqt / 2 + nkt / 2 + nkt;
        self.dispatch_1d(grid as u64, 256);
    }

    /// Commit GPU work and wait for completion using a semaphore completion handler.
    /// This avoids the ~20ms Metal scheduler wakeup overhead of wait_until_completed.
    pub fn submit(self) -> Result<(), String> {
        if self.enc_open {
            self.enc.endEncoding();
        }

        // dispatch_semaphore_t is already a reference-counted opaque pointer.
        let sem = unsafe { dispatch_semaphore_create(0) };
        let sem_val = sem as usize;

        let blk = RcBlock::new(
            move |_cb: NonNull<ProtocolObject<dyn MTLCommandBuffer>>| unsafe {
                dispatch_semaphore_signal(sem_val as *mut c_void);
            },
        );
        unsafe {
            self.cmd_buf.addCompletedHandler(RcBlock::into_raw(blk));
        }
        self.cmd_buf.commit();

        // Bounded wait (10 s). If the GPU hangs (hardware fault), the completion
        // handler never fires and we bail out instead of blocking forever.
        let timeout = unsafe { dispatch_time(0, 10_000_000_000i64) }; // 10 s from now
        let rc = unsafe { dispatch_semaphore_wait(sem, timeout) };
        unsafe {
            dispatch_release(sem);
        }

        if rc == 0 {
            // Command buffer finished (possibly with an error status).
            match self.cmd_buf.status() {
                MTLCommandBufferStatus::Completed => Ok(()),
                st => Err(format!(
                    "Metal command buffer status={st:?}. recent dispatches: {}",
                    self.recent_trace()
                )),
            }
        } else {
            // Timed out: the GPU did not complete the work.
            Err(format!(
                "Metal command buffer timed out after 10s (GPU hang). recent dispatches: {}",
                self.recent_trace()
            ))
        }
    }

    /// Join the recent dispatch trace into a printable string.
    fn recent_trace(&self) -> String {
        let t = self.state.dispatch_trace.lock().unwrap();
        t.iter().cloned().collect::<Vec<_>>().join(" -> ")
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn dispatch_semaphore_create(value: isize) -> *mut std::ffi::c_void;
    fn dispatch_semaphore_signal(sem: *mut std::ffi::c_void) -> isize;
    fn dispatch_semaphore_wait(sem: *mut std::ffi::c_void, timeout: u64) -> isize;
    fn dispatch_time(when: u64, delta: i64) -> u64;
    fn dispatch_release(obj: *mut std::ffi::c_void);
}

// ─── MpsState (global singleton) ─────────────────────────────────────

/// Compile metal.metal from source at runtime (fallback when the build-time
/// metallib is unavailable — see try_new). ~0.3-1 s per process start.
#[cfg(target_os = "macos")]
fn compile_metal_source(device: &MetalDevice) -> Option<MetalLibrary> {
    let src = include_str!("metal.metal");
    match device.newLibraryWithSource_options_error(&*NSString::from_str(src), None) {
        Ok(l) => Some(l),
        Err(e) => {
            eprintln!("MPS: shader compilation failed: {}", e);
            None
        }
    }
}

/// Load the embedded precompiled metallib, falling back to a runtime source
/// compile when it is empty or fails to load.
#[cfg(target_os = "macos")]
fn load_embedded_or_source(device: &MetalDevice, metallib: &[u8]) -> Option<MetalLibrary> {
    if !metallib.is_empty() {
        match device.newLibraryWithData_error(&*DispatchData::from_bytes(metallib)) {
            Ok(l) => return Some(l),
            Err(e) => eprintln!(
                "MPS: precompiled metallib load failed ({e}) — falling back to source compile"
            ),
        }
    }
    compile_metal_source(device)
}

// Retained layer-gpu / old-forward methods: the graph backend replaces them,
// but they stay for tests and the layer-gpu reference path (AGENTS.md).
#[allow(dead_code)]
impl MpsState {
    pub fn try_new() -> Option<Self> {
        if std::env::var("MINFER_DISABLE_MPS").is_ok() {
            eprintln!("MPS: disabled by MINFER_DISABLE_MPS");
            return None;
        }
        // dummy for non-macOS — never called due to cfg
        #[cfg(not(target_os = "macos"))]
        return None;

        #[cfg(target_os = "macos")]
        {
            let device = MTLCreateSystemDefaultDevice()?;

            // GPU trace capture: set MINFER_METAL_CAPTURE=1
            if std::env::var("MINFER_METAL_CAPTURE").is_ok() {
                let capture = unsafe { MTLCaptureManager::sharedCaptureManager() };
                let desc = MTLCaptureDescriptor::new();
                desc.set_capture_device(&*device);
                desc.setDestination(MTLCaptureDestination::DeveloperTools);
                let _ = capture.startCaptureWithDescriptor_error(&*desc);
                eprintln!("MPS: GPU capture started");
            }

            // Prefer the build-time precompiled metallib (build.rs compiles
            // src/metal.metal → minfer.metallib, llama-style: embedded
            // default.metallib, ggml-metal-device.m:128-234). The embedded
            // file is EMPTY when the Metal toolchain was unavailable at build
            // time → fall back to newLibraryWithSource (per-process compile,
            // ~0.3-1 s). Flags must match the runtime source-compile numerics
            // exactly — see build.rs for the chosen -O level.
            // MINFER_METALLIB_FILE overrides the embedded library at runtime
            // (debug/tuning hook to A/B different -O levels without rebuilds).
            static METALLIB: &[u8] = include_bytes!(env!("MINFER_METALLIB_PATH"));
            let override_file = std::env::var("MINFER_METALLIB_FILE")
                .ok()
                .filter(|p| !p.is_empty());
            let lib = if let Some(path) = override_file {
                match std::fs::read(&path).ok().and_then(|b| {
                    device
                        .newLibraryWithData_error(&*DispatchData::from_bytes(&b))
                        .ok()
                }) {
                    Some(l) => l,
                    None => {
                        eprintln!("MPS: metallib override {path} unreadable — falling back to embedded/source");
                        load_embedded_or_source(&device, METALLIB)?
                    }
                }
            } else {
                load_embedded_or_source(&device, METALLIB)?
            };

            let get_pl = |name: &str| {
                let f = match lib.newFunctionWithName(&*NSString::from_str(name)) {
                    Some(f) => f,
                    None => {
                        eprintln!("MPS: no function '{}'", name);
                        return None;
                    }
                };
                match device.newComputePipelineStateWithFunction_error(&*f) {
                    Ok(p) => Some(p),
                    Err(e) => {
                        eprintln!("MPS: pipeline '{}': {}", name, e);
                        None
                    }
                }
            };

            let pl_q4_0_f32 = get_pl("kernel_q4_0_f32_matmul")?;
            let pl_q4_0_f32_multi = get_pl("kernel_q4_0_f32_matmul_multi")?;
            let pl_q4_0_mm_f32 = get_pl("kernel_q4_0_mm_f32")?;
            let pl_q4_1_f32 = get_pl("kernel_q4_1_f32_matmul")?;
            let pl_q4_1_mm_f32 = get_pl("kernel_q4_1_mm_f32")?;
            let pl_q4_1_f32_multi = get_pl("kernel_q4_1_f32_matmul_multi")?;
            let pl_q8_0_f32 = get_pl("kernel_q8_0_f32_matmul")?;
            let pl_q8_0_mm_f32 = get_pl("kernel_q8_0_mm_f32")?;
            let pl_q8_0_f32_multi = get_pl("kernel_q8_0_f32_matmul_multi")?;
            let pl_q4_k_f32 = get_pl("kernel_q4_k_f32_matmul")?;
            let pl_q4_k_mm_f32 = get_pl("kernel_q4_k_mm_f32")?;
            let pl_q4_k_f32_multi = get_pl("kernel_q4_k_f32_matmul_multi")?;
            let pl_q6_k_f32 = get_pl("kernel_q6_k_f32_matmul")?;
            let pl_q6_k_mm_f32 = get_pl("kernel_q6_k_mm_f32")?;
            let pl_q6_k_f32_multi = get_pl("kernel_q6_k_f32_matmul_multi")?;
            let pl_q5_0_f32 = get_pl("kernel_q5_0_f32_matmul")?;
            let pl_q5_0_mm_f32 = get_pl("kernel_q5_0_mm_f32")?;
            let pl_q5_0_f32_multi = get_pl("kernel_q5_0_f32_matmul_multi")?;
            let pl_q5_1_f32 = get_pl("kernel_q5_1_f32_matmul")?;
            let pl_q5_1_mm_f32 = get_pl("kernel_q5_1_mm_f32")?;
            let pl_q5_1_f32_multi = get_pl("kernel_q5_1_f32_matmul_multi")?;
            let pl_q5_k_f32 = get_pl("kernel_q5_k_f32_matmul")?;
            let pl_q5_k_mm_f32 = get_pl("kernel_q5_k_mm_f32")?;
            let pl_q5_k_f32_multi = get_pl("kernel_q5_k_f32_matmul_multi")?;
            let pl_get_rows_q4_0 = get_pl("kernel_get_rows_q4_0")?;
            let pl_get_rows_f32 = get_pl("kernel_get_rows_f32")?;
            let pl_get_rows_q4_k = get_pl("kernel_get_rows_q4_k")?;
            let pl_get_rows_q4_1 = get_pl("kernel_get_rows_q4_1")?;
            let pl_get_rows_q5_0 = get_pl("kernel_get_rows_q5_0")?;
            let pl_get_rows_q5_1 = get_pl("kernel_get_rows_q5_1")?;
            let pl_get_rows_q8_0 = get_pl("kernel_get_rows_q8_0")?;
            let pl_get_rows_q6_k = get_pl("kernel_get_rows_q6_k")?;
            let pl_get_rows_q5_k = get_pl("kernel_get_rows_q5_k")?;
            let pl_rms_norm = get_pl("kernel_rms_norm_f32")?;
            let pl_rms_norm_256 = get_pl("kernel_rms_norm_f32_256")?;
            let pl_add = get_pl("kernel_add_f32")?;
            let pl_add_bias = get_pl("kernel_add_bias_f32")?;
            let pl_mul = get_pl("kernel_mul_f32")?;
            let pl_silu = get_pl("kernel_silu_f32")?;
            let pl_swiglu = get_pl("kernel_swiglu_f32")?;
            let pl_rope = get_pl("kernel_rope_f32")?;
            let pl_gqa_attn = get_pl("kernel_gqa_attn_f32")?;
            let pl_gqa_attn_f16 = get_pl("kernel_gqa_attn_f16")?;
            let pl_gqa_attn_partial = get_pl("kernel_gqa_attn_partial_f32")?;
            let pl_gqa_attn_partial_f16 = get_pl("kernel_gqa_attn_partial_f16")?;
            let pl_gqa_attn_combine = get_pl("kernel_gqa_attn_combine_f32")?;
            let pl_flash_attn = get_pl("kernel_flash_attn_ext_f32")?;
            let pl_flash_attn_f16 = get_pl("kernel_flash_attn_ext_f16")?;
            let pl_flash_attn_hd128 = get_pl("kernel_flash_attn_ext_hd128_f32")?;
            let pl_flash_attn_hd128_f16 = get_pl("kernel_flash_attn_ext_hd128_f16")?;
            let pl_flash_attn_blk = get_pl("kernel_flash_attn_blk_f32")?;
            let pl_flash_attn_blk_f16 = get_pl("kernel_flash_attn_blk_f16")?;
            let pl_flash_attn_blk_hd128 = get_pl("kernel_flash_attn_blk_hd128_f32")?;
            let pl_flash_attn_blk_hd128_f16 = get_pl("kernel_flash_attn_blk_hd128_f16")?;
            let pl_kv_tail_pad = get_pl("kernel_kv_tail_pad")?;
            let pl_store_kv = get_pl("kernel_store_kv_f32")?;
            let pl_store_kv_f16 = get_pl("kernel_store_kv_f16")?;
            let pl_attn_bsr = get_pl("kernel_attn_bias_rope_store")?;
            let pl_attn_rope_store = get_pl("kernel_attn_rope_store")?;
            let pl_attn_scores = get_pl("kernel_attn_scores")?;
            let pl_attn_output = get_pl("kernel_attn_output")?;
            let pl_softmax_attn = get_pl("kernel_softmax_attn")?;
            let pl_warmup = get_pl("kernel_warmup_read")?;
            let dummy_buf = device
                .newBufferWithLength_options((1) as usize, MTLResourceOptions::StorageModeShared)
                .unwrap();
            let m = MpsStateInner {
                device: device.clone(),
                max_threadgroup_memory: device.maxThreadgroupMemoryLength() as u64,
                queue: device.newCommandQueue().unwrap(),
                pl_q4_0_f32,
                pl_q4_0_f32_multi,
                pl_q4_0_mm_f32,
                pl_q4_1_f32,
                pl_q4_1_f32_multi,
                pl_q4_1_mm_f32,
                pl_q8_0_f32,
                pl_q8_0_mm_f32,
                pl_q8_0_f32_multi,
                pl_q4_k_f32,
                pl_q4_k_f32_multi,
                pl_q4_k_mm_f32,
                pl_q6_k_f32,
                pl_q6_k_f32_multi,
                pl_q6_k_mm_f32,
                pl_q5_0_f32,
                pl_q5_0_f32_multi,
                pl_q5_0_mm_f32,
                pl_q5_1_f32,
                pl_q5_1_f32_multi,
                pl_q5_1_mm_f32,
                pl_q5_k_f32,
                pl_q5_k_f32_multi,
                pl_q5_k_mm_f32,
                pl_get_rows_q4_0,
                pl_get_rows_f32,
                pl_get_rows_q4_k,
                pl_get_rows_q4_1,
                pl_get_rows_q5_0,
                pl_get_rows_q5_1,
                pl_get_rows_q8_0,
                pl_get_rows_q6_k,
                pl_get_rows_q5_k,
                pl_rms_norm,
                pl_rms_norm_256,
                pl_add,
                pl_add_bias,
                pl_mul,
                pl_silu,
                pl_swiglu,
                pl_rope,
                pl_gqa_attn,
                pl_gqa_attn_f16,
                pl_gqa_attn_partial,
                pl_gqa_attn_partial_f16,
                pl_gqa_attn_combine,
                pl_flash_attn,
                pl_flash_attn_f16,
                pl_flash_attn_hd128,
                pl_flash_attn_hd128_f16,
                pl_flash_attn_blk,
                pl_flash_attn_blk_f16,
                pl_flash_attn_blk_hd128,
                pl_flash_attn_blk_hd128_f16,
                pl_kv_tail_pad,
                pl_store_kv,
                pl_store_kv_f16,
                pl_attn_bsr,
                pl_attn_rope_store,
                pl_attn_scores,
                pl_attn_output,
                pl_softmax_attn,
                pl_warmup,
                weights: std::sync::Mutex::new(std::collections::HashMap::new()),
                mmap_parts: std::sync::Mutex::new(Vec::new()),
                buf_attn_partial: std::sync::Mutex::new(dummy_buf.clone()),
                buf_positions: std::sync::Mutex::new(dummy_buf.clone()),
                buf_attn_scores: std::sync::Mutex::new(dummy_buf.clone()),
                buf_attn_pad: std::sync::Mutex::new(dummy_buf.clone()),
                dispatch_trace: std::sync::Mutex::new(std::collections::VecDeque::new()),
            };
            eprintln!(
                "MPS: using Metal on {} (unified: {})",
                device.name().to_string(),
                if device.hasUnifiedMemory() {
                    "yes"
                } else {
                    "no"
                }
            );
            Some(MpsState { inner: m })
        }
    }

    pub fn get() -> Option<&'static Self> {
        MPS.get().and_then(|s| s.as_ref())
    }

    pub fn init() {
        MPS.get_or_init(|| {
            let s = Self::try_new();
            if s.is_some() {
                eprintln!("MPS: GPU acceleration enabled");
            } else {
                eprintln!("MPS: not available, using CPU fallback");
            }
            s
        });
    }

    pub fn has_weight(&self, name: &str) -> bool {
        #[cfg(not(target_os = "macos"))]
        {
            false
        }
        #[cfg(target_os = "macos")]
        {
            self.inner.weights.lock().unwrap().contains_key(name)
        }
    }

    /// Look up a registered weight's (buffer, byte offset) — used by the graph
    /// Metal backend to dispatch per-op kernels without holding the Tensor.
    pub fn weight_buf(&self, name: &str) -> Option<(MetalBuffer, u64)> {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = name;
            None
        }
        #[cfg(target_os = "macos")]
        {
            self.inner.weights.lock().unwrap().get(name).cloned()
        }
    }

    /// Allocate a shared-memory f32 buffer (visible to both CPU and GPU) for
    /// the graph backend's buffer pool.
    pub fn new_f32_buffer(&self, n_elements: usize) -> MetalBuffer {
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!()
        }
        #[cfg(target_os = "macos")]
        {
            let bytes = (n_elements * 4) as u64;
            self.inner
                .device
                .newBufferWithLength_options(
                    (bytes) as usize,
                    MTLResourceOptions::StorageModeShared,
                )
                .unwrap()
        }
    }

    /// Register an mmap'd GGUF part for zero-copy weight wrapping. The part
    /// data pointer must be page-aligned (mmap returns page-aligned addresses):
    /// newBufferWithBytesNoCopy requires a page-aligned base (llama
    /// ggml_metal_buffer_map, ggml-metal-device.m:1701). Weights from this part
    /// are then registered as (buffer, offset) into this one buffer.
    pub fn register_part(&self, data: &'static [u8]) {
        #[cfg(not(target_os = "macos"))]
        {
            let _ = data;
        }
        #[cfg(target_os = "macos")]
        {
            if data.is_empty() {
                return;
            }
            let page = 16384; // macOS page size on Apple Silicon
            let base = data.as_ptr() as usize;
            debug_assert!(base % page == 0, "mmap'd GGUF part not page-aligned");
            let buf = unsafe {
                self.inner
                    .device
                    .newBufferWithBytesNoCopy_length_options_deallocator(
                        NonNull::new(data.as_ptr() as *const std::ffi::c_void as *mut c_void)
                            .unwrap(),
                        (data.len() as u64) as usize,
                        MTLResourceOptions::StorageModeShared,
                        None,
                    )
                    .unwrap()
            };
            self.inner
                .mmap_parts
                .lock()
                .unwrap()
                .push((base, data.len(), buf.clone()));
            // GPU-side warm-up (METAL_OPTIMIZATIONS #39): the FIRST GPU access to
            // file-backed (mmap) pages costs ~44 ms of one-time page/TLB setup.
            // Doing a dummy full-buffer read HERE (at model load, outside the
            // CLI's Total timing) moves that cost out of the first prefill —
            // llama-bench's numbers are equally warm. ~5 ms bandwidth + the
            // setup, amortized into load.
            let cb = self.cmd_buffer();
            cb.trace_op("part_warmup");
            cb.enc.setComputePipelineState(&*self.inner.pl_warmup);
            unsafe {
                cb.enc
                    .setBuffer_offset_atIndex(Some(&*buf), (0) as usize, (0) as usize)
            };
            let tiny = self.inner.buf_positions.lock().unwrap().clone();
            unsafe {
                cb.enc
                    .setBuffer_offset_atIndex(Some(&*tiny), (0) as usize, (1) as usize)
            };
            let n = (buf.length() / 4) as u64;
            cb.dispatch_1d((n + 255) / 256, 256);
            let _ = cb.submit();
        }
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        #[cfg(not(target_os = "macos"))]
        {}
        #[cfg(target_os = "macos")]
        {
            if data.is_empty() {
                return;
            }
            let ptr = data.as_ptr() as usize;
            let force_copy = std::env::var("MINFER_WEIGHT_COPY").map_or(false, |v| v == "1");
            // Zero-copy path: the weight is a slice of a registered mmap'd part
            // → (part buffer, offset). The GPU reads the mapped file pages
            // directly (llama's shared mmap buffer, ggml-metal-device.m:1668) —
            // no CPU→GPU memcpy, no GPU-side allocation.
            let entry = if !force_copy {
                let parts = self.inner.mmap_parts.lock().unwrap();
                parts
                    .iter()
                    .find(|(base, len, _)| ptr >= *base && ptr + data.len() <= base + len)
                    .map(|(base, _, buf)| (buf.clone(), (ptr - base) as u64))
            } else {
                None
            };
            let (buf, off) = match entry {
                Some(e) => e,
                None => {
                    // Fallback: copy into a fresh per-weight buffer (offset 0).
                    let b = self
                        .inner
                        .device
                        .newBufferWithLength_options(
                            (data.len() as u64) as usize,
                            MTLResourceOptions::StorageModeShared,
                        )
                        .unwrap();
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            b.contents().as_ptr() as *mut u8,
                            data.len(),
                        );
                    }
                    (b, 0u64)
                }
            };
            self.inner
                .weights
                .lock()
                .unwrap()
                .insert(name.to_string(), (buf, off));
        }
    }

    /// Create a command buffer for batching operations.
    pub fn cmd_buffer(&self) -> MpsCommandBuffer<'_> {
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!()
        }
        #[cfg(target_os = "macos")]
        {
            let cmd_buf_ref = self.inner.queue.commandBuffer().unwrap();
            let enc_ref = cmd_buf_ref.computeCommandEncoder().unwrap();
            // objc2-metal returns owned Retained values — no manual retain/release.
            MpsCommandBuffer {
                state: &self.inner,
                cmd_buf: cmd_buf_ref,
                enc: enc_ref,
                enc_open: true,
            }
        }
    }

    /// Return a buffer with at least `need` bytes, growing the persistent pool
    /// if necessary. The underlying allocation is reused across calls.
    fn get_or_grow(
        slot: &std::sync::Mutex<MetalBuffer>,
        need: u64,
        dev: &MetalDevice,
    ) -> MetalBuffer {
        {
            let b = slot.lock().unwrap();
            if b.length() >= need as usize {
                return b.clone();
            }
        }
        let new = dev
            .newBufferWithLength_options((need) as usize, MTLResourceOptions::StorageModeShared)
            .unwrap();
        *slot.lock().unwrap() = new.clone();
        new
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod mmap_align_test;
