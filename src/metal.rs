// MPS (Metal) backend for Apple Silicon.
//
// Provides MpsCommandBuffer for batching all layer ops into one GPU submission.
//
// #265 split: the L1 runtime is `metal/runtime.rs`, the submission layer
// `metal/encode.rs`, the op → encoding table `metal/ops.rs` and the pure
// `MINFER_*` predicates `metal/policy.rs`. The type definitions, the private
// dispatch primitives and the test-only `matmul_on_gpu_buf` stay here, in the
// parent module, so the children reach private fields/methods with no
// visibility edit (the `CudaState`-stays-in-`cuda.rs` shape).

use crate::tensor::{Tensor, TensorType};
use objc2::{rc::Retained, runtime::ProtocolObject};
#[cfg(target_os = "macos")]
use objc2_metal::{
    MTLBarrierScope, MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSharedEvent, MTLSize,
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
/// F5 (#137): the synchronization primitive of the cross-backend staging copy.
/// `MTLSharedEvent` is chosen over a completion handler because it is the only
/// Metal event that supports a **device-side** wait (`encodeWaitForEvent`), and
/// it also supports a bounded **host** wait (`waitUntilSignaledValue`). See
/// `docs/BACKEND-REGISTRY-DESIGN.md` §11.2.
#[cfg(target_os = "macos")]
pub type MetalSharedEvent = Retained<ProtocolObject<dyn MTLSharedEvent>>;
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
    // #164: f16 weights stay 2 B/element on the device (matmul + embed).
    pl_f16_f32: MetalComputePipelineState,
    // #317: f32-weight matmul (parity with CUDA's `launch_f32_f32_matmul`).
    pl_f32_f32: MetalComputePipelineState,
    // #208: bf16 weights stay 2 B/element too (its own kernel, the exact shift
    // decode — the Metal twin of CUDA's #208 pair).
    pl_bf16_f32: MetalComputePipelineState,
    pl_get_rows_q4_0: MetalComputePipelineState,
    pl_get_rows_f32: MetalComputePipelineState,
    pl_get_rows_q4_k: MetalComputePipelineState,
    pl_get_rows_q4_1: MetalComputePipelineState,
    pl_get_rows_q5_0: MetalComputePipelineState,
    pl_get_rows_q5_1: MetalComputePipelineState,
    pl_get_rows_q8_0: MetalComputePipelineState,
    pl_get_rows_q6_k: MetalComputePipelineState,
    pl_get_rows_q5_k: MetalComputePipelineState,
    pl_get_rows_f16: MetalComputePipelineState,
    pl_get_rows_bf16: MetalComputePipelineState,
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
    // #310: the packed Q8_0 causal attention kernel (classic tiling). Since
    // #310 the fast families also read a packed region — natively (mechanism A)
    // or through the f32 stage (mechanism B) — so this is the fallback for the
    // shapes no fast family covers; see `metal_backend`'s Attn arm.
    pl_gqa_attn_q8_0: MetalComputePipelineState,
    // E1 `attn_span` read path (issue #44, G5a): the windowed kernel family.
    pl_gqa_attn_window: MetalComputePipelineState,
    pl_gqa_attn_window_f16: MetalComputePipelineState,
    pl_gqa_attn_window_q8_0: MetalComputePipelineState,
    // C8b S4 `kv_map` read path (issue #362): the set-valued window's sibling
    // kernels. Deliberately separate from the one-range window family above,
    // whose instruction stream is a measured contract (#315).
    pl_gqa_attn_map: MetalComputePipelineState,
    pl_gqa_attn_map_f16: MetalComputePipelineState,
    pl_gqa_attn_map_q8_0: MetalComputePipelineState,
    pl_gqa_attn_partial: MetalComputePipelineState,
    pl_gqa_attn_partial_f16: MetalComputePipelineState,
    pl_gqa_attn_combine: MetalComputePipelineState,
    pl_flash_attn: MetalComputePipelineState,
    pl_flash_attn_f16: MetalComputePipelineState,
    pl_flash_attn_hd128: MetalComputePipelineState,
    pl_flash_attn_hd128_f16: MetalComputePipelineState,
    // #310 mechanism A: the packed Q8_0 decode flash family (the f16 kernels'
    // twin, reading one block's four dequantized elements per lane).
    pl_flash_attn_q8_0: MetalComputePipelineState,
    pl_flash_attn_hd128_q8_0: MetalComputePipelineState,
    pl_flash_attn_blk: MetalComputePipelineState,
    pl_flash_attn_blk_f16: MetalComputePipelineState,
    pl_flash_attn_blk_hd128: MetalComputePipelineState,
    pl_flash_attn_blk_hd128_f16: MetalComputePipelineState,
    // #359: the fast explicit-span prefill. A *copy* of the causal blk family
    // (fa_prefill.metal, left byte-untouched) whose inline mask reads each
    // query's `[lo, hi)` window instead of `[0, pos+1)`.
    pl_flash_attn_window_blk: MetalComputePipelineState,
    pl_flash_attn_window_blk_f16: MetalComputePipelineState,
    pl_flash_attn_window_blk_hd128: MetalComputePipelineState,
    pl_flash_attn_window_blk_hd128_f16: MetalComputePipelineState,
    // #369: the set-valued `kv_map` siblings — the same tile with a
    // run-membership mask instead of `[lo, hi)`.
    pl_flash_attn_window_map: MetalComputePipelineState,
    pl_flash_attn_window_map_f16: MetalComputePipelineState,
    pl_flash_attn_window_map_hd128: MetalComputePipelineState,
    pl_flash_attn_window_map_hd128_f16: MetalComputePipelineState,
    pl_kv_tail_pad: MetalComputePipelineState,
    pl_store_kv: MetalComputePipelineState,
    pl_store_kv_f16: MetalComputePipelineState,
    pl_store_kv_q8_0: MetalComputePipelineState,
    // #310 mechanism B: packed Q8_0 cell window -> transient f32 stage.
    pl_dequant_kv_q8_0_to_f32: MetalComputePipelineState,
    pl_attn_bsr: MetalComputePipelineState,
    pl_attn_rope_store: MetalComputePipelineState,
    pl_attn_scores: MetalComputePipelineState,
    pl_attn_output: MetalComputePipelineState,
    pl_softmax_attn: MetalComputePipelineState,
    pl_warmup: MetalComputePipelineState,
    // (buffer, byte-offset, logical byte length): weights live either in a
    // per-weight copied buffer (offset 0) or — since the 2026-08-21 mmap loader —
    // as offsets into a page-aligned NoCopy buffer over the mmap'd GGUF part
    // (llama-style, ggml-metal-device.m:1668; newBufferWithBytesNoCopy requires a
    // page-aligned base, so per-tensor offsets are passed via setBuffer:offset:).
    // The byte length is the registered weight's own extent — the E4 gate's
    // "weights" term (issue #299), the analogue of CUDA's per-entry `size`.
    weights: std::sync::Mutex<std::collections::HashMap<String, (MetalBuffer, u64, usize)>>,
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
    // #310 mechanism B: transient f32 K/V windows dequantized from a packed Q8_0
    // region so the f32 fast families can read it. Grow-on-demand; never
    // shrunk (the `get_or_grow` contract).
    buf_kv_stage_k: std::sync::Mutex<MetalBuffer>,
    buf_kv_stage_v: std::sync::Mutex<MetalBuffer>,
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
    /// Choose the f32-activation matmul for all weight types (including Q4_0,
    /// matching llama.cpp's Metal backend which does not Q8_0-quantize activations).
    /// Pre-looked-up weight buffer and type — avoids per-matmul HashMap locking.
    /// (Only exercised by the `matmul_bandwidth_profile` test; the graph backend
    /// calls `quant_matmul_f32_on_gpu_buf` directly — so this wrapper is
    /// `#[cfg(test)]`, judged by the #255 macOS oracle.)
    #[cfg(test)]
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
        self.quant_matmul_f32_on_gpu_buf(wb, w_off, ttype, f32_x, x_off, out, od, id, nt)
            .expect("quant_matmul_f32_on_gpu_buf dispatch");
    }
}

impl MpsState {
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

mod encode;
mod ops;
mod policy;
mod runtime;

pub use policy::{
    flash_attn_enabled, matmul_attn_enabled, packed_attn_route, prefill_flash_enabled,
    prefill_window_flash_enabled, rms_norm_256_enabled, PackedRoute,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod mmap_align_test;
