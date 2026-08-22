// MPS (Metal) backend for Apple Silicon.
//
// Provides MpsCommandBuffer for batching all layer ops into one GPU submission.

use std::sync::OnceLock;
use crate::tensor::{Tensor, TensorType};
#[cfg(target_os = "macos")]
use metal::objc::{msg_send, sel, sel_impl};

static MPS: OnceLock<Option<MpsState>> = OnceLock::new();

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
    LOCK.get_or_init(|| std::sync::Mutex::new(())).lock().unwrap_or_else(|e| e.into_inner())
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
    if tensors.len() < 2 { return None; }
    let tt = tensors[0].ttype;
    if tensors.iter().any(|t| t.ttype != tt) { return None; }
    let bq = quant_block_q(tt);
    let bb = quant_block_bytes(tt);
    if bb == 0 { return None; }
    let ne0 = tensors[0].shape[0] as usize;
    if tensors.iter().any(|t| t.shape[0] != ne0 as i64) { return None; }
    if ne0 % bq != 0 { return None; }
    let row = (ne0 / bq) * bb;
    let rows: usize = tensors.iter().map(|t| t.shape[1] as usize).sum();
    let mut out = Vec::with_capacity(rows * row);
    for t in tensors {
        out.extend_from_slice(t.data());
    }
    if out.len() != rows * row { return None; }
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

#[cfg(target_os = "macos")]
struct MpsStateInner {
    device: metal::Device,
    // Cached device capabilities (queried once at init, aligned with llama.cpp's
    // ggml-metal-device props). All dispatch-time guards compare against these.
    max_threadgroup_memory: u64,
    queue: metal::CommandQueue,
    pl_q4_0_f32: metal::ComputePipelineState,
    pl_q4_0_f32_multi: metal::ComputePipelineState,
    pl_q4_0_mm_f32: metal::ComputePipelineState,
    pl_q4_1_f32: metal::ComputePipelineState,
    pl_q4_1_f32_multi: metal::ComputePipelineState,
    pl_q4_1_mm_f32: metal::ComputePipelineState,
    pl_q8_0_f32: metal::ComputePipelineState,
    pl_q8_0_f32_multi: metal::ComputePipelineState,
    pl_q8_0_mm_f32: metal::ComputePipelineState,
    pl_q4_k_f32: metal::ComputePipelineState,
    pl_q4_k_f32_multi: metal::ComputePipelineState,
    pl_q4_k_mm_f32: metal::ComputePipelineState,
    pl_q6_k_f32: metal::ComputePipelineState,
    pl_q6_k_f32_multi: metal::ComputePipelineState,
    pl_q6_k_mm_f32: metal::ComputePipelineState,
    pl_q5_0_f32: metal::ComputePipelineState,
    pl_q5_0_f32_multi: metal::ComputePipelineState,
    pl_q5_0_mm_f32: metal::ComputePipelineState,
    pl_q5_1_f32: metal::ComputePipelineState,
    pl_q5_1_f32_multi: metal::ComputePipelineState,
    pl_q5_1_mm_f32: metal::ComputePipelineState,
    pl_q5_k_f32: metal::ComputePipelineState,
    pl_q5_k_f32_multi: metal::ComputePipelineState,
    pl_q5_k_mm_f32: metal::ComputePipelineState,
    pl_get_rows_q4_0: metal::ComputePipelineState,
    pl_get_rows_f32: metal::ComputePipelineState,
    pl_get_rows_q4_k: metal::ComputePipelineState,
    pl_get_rows_q4_1: metal::ComputePipelineState,
    pl_get_rows_q5_0: metal::ComputePipelineState,
    pl_get_rows_q5_1: metal::ComputePipelineState,
    pl_get_rows_q8_0: metal::ComputePipelineState,
    pl_get_rows_q6_k: metal::ComputePipelineState,
    pl_get_rows_q5_k: metal::ComputePipelineState,
    pl_rms_norm: metal::ComputePipelineState,
    pl_rms_norm_256: metal::ComputePipelineState,
    pl_add: metal::ComputePipelineState,
    pl_add_bias: metal::ComputePipelineState,
    pl_mul: metal::ComputePipelineState,
    pl_silu: metal::ComputePipelineState,
    pl_swiglu: metal::ComputePipelineState,
    pl_rope: metal::ComputePipelineState,
    pl_gqa_attn: metal::ComputePipelineState,
    pl_gqa_attn_f16: metal::ComputePipelineState,
    pl_gqa_attn_partial: metal::ComputePipelineState,
    pl_gqa_attn_partial_f16: metal::ComputePipelineState,
    pl_gqa_attn_combine: metal::ComputePipelineState,
    pl_flash_attn: metal::ComputePipelineState,
    pl_flash_attn_f16: metal::ComputePipelineState,
    pl_flash_attn_hd128: metal::ComputePipelineState,
    pl_flash_attn_hd128_f16: metal::ComputePipelineState,
    pl_flash_attn_blk: metal::ComputePipelineState,
    pl_flash_attn_blk_f16: metal::ComputePipelineState,
    pl_flash_attn_blk_hd128: metal::ComputePipelineState,
    pl_flash_attn_blk_hd128_f16: metal::ComputePipelineState,
    pl_kv_tail_pad: metal::ComputePipelineState,
    pl_store_kv: metal::ComputePipelineState,
    pl_store_kv_f16: metal::ComputePipelineState,
    pl_attn_bsr: metal::ComputePipelineState,
    pl_attn_scores: metal::ComputePipelineState,
    pl_attn_output: metal::ComputePipelineState,
    pl_softmax_attn: metal::ComputePipelineState,
    pl_warmup: metal::ComputePipelineState,
    // (buffer, byte-offset): weights live either in a per-weight copied buffer
    // (offset 0) or — since the 2026-08-21 mmap loader — as offsets into a
    // page-aligned NoCopy buffer over the mmap'd GGUF part (llama-style,
    // ggml-metal-device.m:1668; newBufferWithBytesNoCopy requires a page-aligned
    // base, so per-tensor offsets are passed via setBuffer:offset:).
    weights: std::sync::Mutex<std::collections::HashMap<String, (metal::Buffer, u64)>>,
    // Registered mmap'd GGUF parts: (base_ptr, len, Metal buffer). register_weight
    // resolves a weight slice to (buffer, offset) by pointer-range containment.
    mmap_parts: std::sync::Mutex<Vec<(usize, usize, metal::Buffer)>>,
    // Scratch for the mmap-part warmup dispatch (register_part).
    buf_positions: std::sync::Mutex<metal::Buffer>,
    buf_attn_partial: std::sync::Mutex<metal::Buffer>,
    // Prefill parallel-attention scratch (P1 2026-08-11): scores [nt][nh][nkv].
    buf_attn_scores: std::sync::Mutex<metal::Buffer>,
    // Flash-prefill tail pad (2026-08-14): [2][64][nkt] f32/f16 K-tail + V-tail.
    buf_attn_pad: std::sync::Mutex<metal::Buffer>,
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
    cmd_buf: &'a metal::CommandBufferRef,
    enc: &'a metal::ComputeCommandEncoderRef,
}

#[cfg(target_os = "macos")]
impl Drop for MpsCommandBuffer<'_> {
    fn drop(&mut self) {
        // Release the retains taken in cmd_buffer(). This MUST happen after the
        // encoder is ended and the command buffer committed, so Metal sees a
        // clean lifecycle (no "encoder released without endEncoding").
        unsafe {
            let _: () = msg_send![self.cmd_buf, release];
            let _: () = msg_send![self.enc, release];
        }
    }
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
        self.enc.set_bytes(
            idx,
            std::mem::size_of::<i32>() as u64,
            val as *const i32 as *const std::ffi::c_void,
        );
    }

    /// GPU memory barrier (2026-08-19 fix). Metal does NOT guarantee
    /// write visibility between dispatches in a single compute command encoder;
    /// without an explicit barrier, a kernel that reads a buffer written by a
    /// preceding dispatch can race with that dispatch's last threadgroups,
    /// intermittently corrupting the tail rows (observed: last-2 token slots of
    /// layer0 bn on 1.5B/7B prefill). llama.cpp's Metal backend inserts the same
    /// barrier after every op. MTLBarrierScopeBuffers = 1 << 0.
    fn barrier(&self) {
        unsafe {
            let _: () = msg_send![self.enc, memoryBarrierWithScope: 1u64];
        }
    }

    fn dispatch_2d(&self, w: u64, h: u64, tw: u64, th: u64) {
        self.enc.dispatch_thread_groups(
            metal::MTLSize { width: w, height: h, depth: 1 },
            metal::MTLSize { width: tw, height: th, depth: 1 },
        );
        self.barrier();
    }

    fn dispatch_3d(&self, w: u64, h: u64, d: u64, tw: u64, th: u64, td: u64) {
        self.enc.dispatch_thread_groups(
            metal::MTLSize { width: w, height: h, depth: d },
            metal::MTLSize { width: tw, height: th, depth: td },
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
    fn gemm_dispatch(&self, pl: &metal::ComputePipelineState, wb: &metal::Buffer, w_off: u64,
        x: &metal::Buffer, x_off: u64, out: &metal::Buffer, od: usize, id: usize, nt: usize,
    ) {
        if 8192 > self.state.max_threadgroup_memory {
            gpu_abort(&format!(
                "GEMM needs 8192 B threadgroup memory, device max is {} B",
                self.state.max_threadgroup_memory
            ));
        }
        self.enc.set_compute_pipeline_state(pl);
        self.enc.set_buffer(0, Some(wb), w_off);
        self.enc.set_buffer(1, Some(x), x_off);
        self.enc.set_buffer(2, Some(out), 0);
        let mm_p = [od as i32, id as i32, nt as i32];
        self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
        self.enc.set_threadgroup_memory_length(0, 8192);
        self.dispatch_2d(((nt + 31) / 32) as u64, ((od + 63) / 64) as u64, 32, 4);
    }

    fn dispatch_1d(&self, n: u64, tg: u64) {
        self.enc.dispatch_thread_groups(
            metal::MTLSize { width: (n + tg - 1) / tg, height: 1, depth: 1 },
            metal::MTLSize { width: tg, height: 1, depth: 1 },
        );
        self.barrier();
    }

    pub fn quant_matmul_f32_on_gpu_buf(&self, wb: &metal::Buffer, w_off: u64, ttype: TensorType,
        x: &metal::Buffer, x_off: u64, out: &metal::Buffer, od: usize, id: usize, nt: usize,
    ) {
        self.trace_op("matmul");
        // GPU safety (M1): the K-quant (super-block) kernels index weights by
        // K/256 super-blocks (floor). A non-256-aligned id silently drops the
        // remainder (wrong results, not a fault) — refuse rather than risk it.
        if matches!(ttype, TensorType::Q4_K | TensorType::Q5_K | TensorType::Q6_K) && id % 256 != 0 {
            gpu_abort(&format!(
                "matmul input dim id={id} is not 256-aligned for {ttype:?} (K-quant kernels use K/256 super-block floor)"
            ));
        }
        match ttype {
            TensorType::Q8_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q8_0_mm_f32, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q8_0_f32_multi } else { &self.state.pl_q8_0_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    const NW: u64 = 32;
                    const NSG: u64 = 4;
                    const NR0: u64 = 2;
                    const TG_MEM: u64 = NW * NR0 * std::mem::size_of::<f32>() as u64; // 256 bytes
                    self.enc.set_threadgroup_memory_length(0, TG_MEM);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 1) / 2) as u64, grid_y, NW, NSG);
                }
            }
            TensorType::Q4_K | TensorType::Q6_K => {
                // Q6_K has a simdgroup GEMM (super-block); Q4_K still falls back
                // to the scalar f32 multi (no Q4_K in the shipped 0.5B K_M models).
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    // both Q4_K and Q6_K have simdgroup GEMMs
                    let pl = if ttype == TensorType::Q6_K { &self.state.pl_q6_k_mm_f32 } else { &self.state.pl_q4_k_mm_f32 };
                    self.gemm_dispatch(pl, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    let pl: &metal::ComputePipelineState = if ttype == TensorType::Q4_K {
                        if nt > 1 { &self.state.pl_q4_k_f32_multi } else { &self.state.pl_q4_k_f32 }
                    } else {
                        if nt > 1 { &self.state.pl_q6_k_f32_multi } else { &self.state.pl_q6_k_f32 }
                    };
                    self.enc.set_compute_pipeline_state(pl);
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
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
                    self.gemm_dispatch(&self.state.pl_q4_1_mm_f32, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q4_1_f32_multi } else { &self.state.pl_q4_1_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_0 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q5_0_mm_f32, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q5_0_f32_multi } else { &self.state.pl_q5_0_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_1 => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q5_1_mm_f32, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q5_1_f32_multi } else { &self.state.pl_q5_1_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_K => {
                if nt >= 2 && (od >= 2048 || nt >= 9) && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q5_k_mm_f32, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q5_k_f32_multi } else { &self.state.pl_q5_k_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
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
                    self.gemm_dispatch(&self.state.pl_q4_0_mm_f32, wb, w_off, x, x_off, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q4_0_f32_multi } else { &self.state.pl_q4_0_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), w_off);
                    self.enc.set_buffer(1, Some(x), x_off);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            _ => {
                self.enc.set_compute_pipeline_state(
                    if nt > 1 { &self.state.pl_q4_0_f32_multi } else { &self.state.pl_q4_0_f32 }
                );
                self.enc.set_buffer(0, Some(wb), w_off);
                self.enc.set_buffer(1, Some(x), x_off);
                self.enc.set_buffer(2, Some(out), 0);
                let mm_p = [od as i32, id as i32, nt as i32];
                self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
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
    fn matmul_on_gpu_buf(&self, wb: &metal::Buffer, w_off: u64, ttype: TensorType,
        _q8_x: &metal::Buffer, f32_x: &metal::Buffer, x_off: u64,
        out: &metal::Buffer, od: usize, id: usize, nt: usize,
    ) {
        self.quant_matmul_f32_on_gpu_buf(wb, w_off, ttype, f32_x, x_off, out, od, id, nt);
    }

    /// GPU embedding lookup: dequantize Q4_0 embedding rows for nt token ids.
    /// Writes f32 hidden state [nt][ne] to dst (buf_hidden).
    pub fn embed_tokens_gpu(&self, wb: &metal::Buffer, w_off: u64, ids: &metal::Buffer,
        dst: &metal::Buffer, ne: usize, nt: usize, ttype: TensorType,
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
        self.enc.set_compute_pipeline_state(pl);
        self.enc.set_buffer(0, Some(wb), w_off);
        self.enc.set_buffer(1, Some(ids), 0);
        self.enc.set_buffer(2, Some(dst), 0);
        self.set_params(3, &(ne as i32));
        self.set_params(4, &(nt as i32));
        self.dispatch_1d((nt * nb) as u64, 256);
    }

    /// Generic f32 row selection: out[t] = x[ids[t]] (graph n_out tail rows).
    pub fn get_rows_f32(&self, x: &metal::Buffer, ids: &metal::Buffer,
        out: &metal::Buffer, ne: usize, nt: usize,
    ) {
        self.trace_op("get_rows_f32");
        self.enc.set_compute_pipeline_state(&self.state.pl_get_rows_f32);
        self.enc.set_buffer(0, Some(x), 0);
        self.enc.set_buffer(1, Some(ids), 0);
        self.enc.set_buffer(2, Some(out), 0);
        self.set_params(3, &(ne as i32));
        self.dispatch_2d(nt as u64, ne as u64, 1, 1);
    }

    /// RMSNorm: y = x * rsqrt(mean(x²)+eps) * w
    pub fn rms_norm(&self, x: &metal::Buffer, w: Option<&metal::Buffer>, w_off: u64,
        y: &metal::Buffer, d: usize, n: usize, eps: f32, off: u64,
    ) {
        self.trace_op("rms_norm");
        self.enc.set_compute_pipeline_state(&self.state.pl_rms_norm);
        self.enc.set_buffer(0, Some(x), off);
        self.enc.set_buffer(1, Some(w.unwrap_or(y)), w_off); // dummy if no weight
        self.enc.set_buffer(2, Some(y), 0);
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        self.dispatch_2d(n as u64, 1, 32, 1);
    }

    /// RMSNorm with a 256-thread multi-simdgroup kernel (P1 2026-08-10, llama
    /// transcription). Same math as rms_norm but the threadgroup is 256 threads
    /// so a single 896-element row isn't DRAM-latency-bound (the 32-thread
    /// kernel measured ~7x the per-dispatch cost of 256-thread elementwise ops).
    /// Requires a threadgroup buffer of n_simdgroups floats (8 for 256 threads).
    pub fn rms_norm_256(&self, x: &metal::Buffer, w: Option<&metal::Buffer>, w_off: u64,
        y: &metal::Buffer, d: usize, n: usize, eps: f32, off: u64,
    ) {
        self.trace_op("rms_norm");
        self.enc.set_compute_pipeline_state(&self.state.pl_rms_norm_256);
        self.enc.set_buffer(0, Some(x), off);
        self.enc.set_buffer(1, Some(w.unwrap_or(y)), w_off);
        self.enc.set_buffer(2, Some(y), 0);
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        self.enc.set_threadgroup_memory_length(0, 32 * 4);
        // 256 threads = 8 simdgroups; one threadgroup per row.
        self.dispatch_2d(n as u64, 1, 32, 8);
    }

    /// Element-wise add: z = x + y
    pub fn add_f32(&self, x: &metal::Buffer, y: &metal::Buffer, z: &metal::Buffer, n: usize) {
        self.add_f32_off(x, y, z, n, 0, 0, 0);
    }

    /// Element-wise add with per-buffer byte offsets (last-layer output-rows
    /// reduction: x/z read/write the tail n_out rows of `hidden`, y starts at 0).
    pub fn add_f32_off(&self, x: &metal::Buffer, y: &metal::Buffer, z: &metal::Buffer,
        n: usize, x_off: u64, y_off: u64, z_off: u64,
    ) {
        self.trace_op("add");
        self.enc.set_compute_pipeline_state(&self.state.pl_add);
        self.enc.set_buffer(0, Some(x), x_off);
        self.enc.set_buffer(1, Some(y), y_off);
        self.enc.set_buffer(2, Some(z), z_off);
        self.set_params(3, &(n as i32));
        // float4 kernel: 4 elements/thread (ceil for the scalar tail)
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// Add 1-D bias to rows: y[t][i] += b[i]. `off` = element offset into `y`
    /// (used by the fused QKV path to bias the q/k/v sections of one buffer).
    pub fn add_bias_f32(&self, y: &metal::Buffer, b: &metal::Buffer, b_off: u64,
        d: usize, n: usize, off: usize,
    ) {
        self.trace_op("bias");
        self.enc.set_compute_pipeline_state(&self.state.pl_add_bias);
        self.enc.set_buffer(0, Some(y), (off * 4) as u64);
        self.enc.set_buffer(1, Some(b), b_off);
        self.set_params(2, &(d as i32));
        // float4 kernel: 4 dims/thread
        self.dispatch_2d(n as u64, ((d as u64) + 3) / 4, 1, 64);
    }

    /// Element-wise multiply: z = x * y
    pub fn mul_f32(&self, x: &metal::Buffer, y: &metal::Buffer, z: &metal::Buffer, n: usize) {
        self.enc.set_compute_pipeline_state(&self.state.pl_mul);
        self.enc.set_buffer(0, Some(x), 0);
        self.enc.set_buffer(1, Some(y), 0);
        self.enc.set_buffer(2, Some(z), 0);
        self.set_params(3, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// SiLU in-place: y = y / (1 + exp(-y))
    pub fn silu_f32(&self, y: &metal::Buffer, n: usize) {
        self.enc.set_compute_pipeline_state(&self.state.pl_silu);
        self.enc.set_buffer(0, Some(y), 0);
        self.set_params(1, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// SwiGLU fused: dst = silu(gate) * up  (dst may alias gate)
    pub fn swiglu_f32(&self, gate: &metal::Buffer, up: &metal::Buffer, dst: &metal::Buffer, n: usize) {
        self.trace_op("swiglu");
        self.enc.set_compute_pipeline_state(&self.state.pl_swiglu);
        self.enc.set_buffer(0, Some(gate), 0);
        self.enc.set_buffer(1, Some(up), 0);
        self.enc.set_buffer(2, Some(dst), 0);
        self.set_params(3, &(n as i32));
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// SwiGLU over a fused gate+up buffer: gate at offset 0, up at `up_off`
    /// elements (fused FFN gate+up path). Writes silu(gate)*up back to gate.
    pub fn swiglu_f32_off(&self, gate: &metal::Buffer, up: &metal::Buffer, dst: &metal::Buffer,
        n: usize, up_off: usize,
    ) {
        self.trace_op("swiglu");
        self.enc.set_compute_pipeline_state(&self.state.pl_swiglu);
        self.enc.set_buffer(0, Some(gate), 0);
        self.enc.set_buffer(1, Some(up), (up_off * 4) as u64);
        self.enc.set_buffer(2, Some(dst), 0);
        self.set_params(3, &(n as i32));
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// RoPE (in-place): x layout [nt][n_head][n_dims]. `off` = element offset
    /// into `x` (fused QKV: K section lives mid-buffer).
    /// rope_style: 0 = non-interleaved (Qwen2), 1 = interleaved (LLaMA).
    pub fn rope_f32(&self, x: &metal::Buffer, n_head: usize, n_dims: usize, nt: usize,
        freq_base: f32, freq_scale: f32, positions: &metal::Buffer,
        rope_style: i32, off: usize,
    ) {
        self.trace_op("rope");
        self.enc.set_compute_pipeline_state(&self.state.pl_rope);
        self.enc.set_buffer(0, Some(x), (off * 4) as u64);
        self.set_params(1, &(n_head as i32));
        self.set_params(2, &(n_dims as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(freq_base.to_bits() as i32));
        self.set_params(5, &(freq_scale.to_bits() as i32));
        self.enc.set_buffer(6, Some(positions), 0);
        self.set_params(7, &rope_style);
        // P7: one thread per (dim, head, token) instead of one per (token, head)
        self.dispatch_3d((n_dims / 2) as u64, n_head as u64, nt as u64, 1, 1, 1);
    }

    /// Flash Attention: one threadgroup per (token, KV_head), tiled K/V
    /// with online softmax. Each simdgroup processes one query head.
    /// K/V tiles loaded into threadgroup-shared memory, reused by all
    /// query heads in the GQA group.
    pub fn gqa_attn_f32(&self, q: &metal::Buffer, k: &metal::Buffer, v: &metal::Buffer,
        o: &metal::Buffer, positions: &metal::Buffer, nh: usize, nk: usize, hd: usize, scale: f32, nt: usize,
    ) {
        self.gqa_attn_f32_off(q, 0, k, 0, v, 0, o, positions, nh, nk, hd, scale, nt);
    }

    /// Offset variant of `gqa_attn_f32` — K/V may live at byte offsets inside a
    /// shared buffer (the graph backend's `[K | V]` contiguous KV region).
    pub fn gqa_attn_f32_off(&self, q: &metal::Buffer, q_off: u64,
        k: &metal::Buffer, k_off: u64, v: &metal::Buffer, v_off: u64,
        o: &metal::Buffer, positions: &metal::Buffer, nh: usize, nk: usize, hd: usize, scale: f32, nt: usize,
    ) {
        self.trace_op("gqa_attn");
        let gqa = nh / nk;
        self.enc.set_compute_pipeline_state(
            if kv_cache_is_f16() { &self.state.pl_gqa_attn_f16 } else { &self.state.pl_gqa_attn }
        );
        self.enc.set_buffer(0, Some(q), q_off);
        self.enc.set_buffer(1, Some(k), k_off);
        self.enc.set_buffer(2, Some(v), v_off);
        self.enc.set_buffer(3, Some(o), 0);
        self.enc.set_buffer(4, Some(positions), 0);
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        self.enc.set_threadgroup_memory_length(0, shmem);
        self.dispatch_2d(nt as u64, nk as u64, 32, gqa as u64);
    }

    /// KV-parallel split attention for nt==1 decode (the classic kernel's grid
    /// is only (1, nk) threadgroups that loop the KV sequentially — the measured
    /// #1 decode bottleneck). Two passes: partial per KV chunk (grid (nt,nk,P)),
    /// then combine (grid (nt,nh)). Requires the partials buffer (`buf_attn_partial`)
    /// sized for nt*nh*P*(2+hd) floats, grown on demand here.
    pub fn gqa_attn_split_f32(&self, q: &metal::Buffer, k: &metal::Buffer, v: &metal::Buffer,
        o: &metal::Buffer, positions: &metal::Buffer, nh: usize, nk: usize, hd: usize,
        scale: f32, nt: usize, n_chunks: usize,
    ) {
        self.trace_op("gqa_attn_split");
        let gqa = nh / nk;
        let need = (nt * nh * n_chunks * (2 + hd) * 4) as u64;
        let partial = MpsState::get_or_grow(&self.state.buf_attn_partial, need, &self.state.device);

        // pass 1: partials per (token, KV_head, chunk) — f16 cache picks the
        // f16 partial kernel (K/V read as half, staged to f32 float4 tiles).
        self.enc.set_compute_pipeline_state(
            if kv_cache_is_f16() { &self.state.pl_gqa_attn_partial_f16 } else { &self.state.pl_gqa_attn_partial }
        );
        self.enc.set_buffer(0, Some(q), 0);
        self.enc.set_buffer(1, Some(k), 0);
        self.enc.set_buffer(2, Some(v), 0);
        self.enc.set_buffer(3, Some(&partial), 0);
        self.enc.set_buffer(4, Some(positions), 0);
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        self.set_params(10, &(n_chunks as i32));
        const BC: u64 = 32;
        let shmem = BC * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
        self.enc.set_threadgroup_memory_length(0, shmem);
        self.dispatch_3d(nt as u64, nk as u64, n_chunks as u64, 32, gqa as u64, 1);

        // pass 2: combine
        self.enc.set_compute_pipeline_state(&self.state.pl_gqa_attn_combine);
        self.enc.set_buffer(0, Some(&partial), 0);
        self.enc.set_buffer(1, Some(o), 0);
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
    pub fn gqa_attn_flash(&self, q: &metal::Buffer, k: &metal::Buffer, v: &metal::Buffer,
        o: &metal::Buffer, positions: &metal::Buffer, nh: usize, nk: usize, hd: usize,
        scale: f32, nt: usize, n_chunks: usize,
    ) {
        self.trace_op("gqa_attn_flash");
        let need = (nt * nh * n_chunks * (2 + hd) * 4) as u64;
        let partial = MpsState::get_or_grow(&self.state.buf_attn_partial, need, &self.state.device);

        // pass 1: flash partials — f16 cache reads the half K/V directly.
        self.enc.set_compute_pipeline_state(match (kv_cache_is_f16(), hd) {
            (false, 128) => &self.state.pl_flash_attn_hd128,
            (true, 128) => &self.state.pl_flash_attn_hd128_f16,
            (false, _) => &self.state.pl_flash_attn,
            (true, _) => &self.state.pl_flash_attn_f16,
        });
        self.enc.set_buffer(0, Some(q), 0);
        self.enc.set_buffer(1, Some(k), 0);
        self.enc.set_buffer(2, Some(v), 0);
        self.enc.set_buffer(3, Some(&partial), 0);
        self.enc.set_buffer(4, Some(positions), 0);
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        self.set_params(10, &(n_chunks as i32));
        // shmem (hd=64): sq4 (16 float4 = 256 B) | ss (32 f32 = 128 B) | so4 (32 float4 = 512 B) = 896 → 1024
        // shmem (hd=128): sq4 (32 float4 = 512 B) | ss (32 f32 = 128 B) | so4 (32 float4 = 512 B) = 1152
        let shmem = if hd == 128 { 1152 } else { 1024 };
        self.enc.set_threadgroup_memory_length(0, shmem);
        self.dispatch_3d(nt as u64, nh as u64, n_chunks as u64, 32, 1, 1);

        // pass 2: combine (shared with the split path)
        self.enc.set_compute_pipeline_state(&self.state.pl_gqa_attn_combine);
        self.enc.set_buffer(0, Some(&partial), 0);
        self.enc.set_buffer(1, Some(o), 0);
        self.set_params(2, &(nh as i32));
        self.set_params(3, &(hd as i32));
        self.set_params(4, &(nt as i32));
        self.set_params(5, &(n_chunks as i32));
        self.dispatch_2d(nt as u64, nh as u64, 32, 1);
    }

    /// Scatter nt rows of src[nt][nkt] into dst[positions[t]][nkt].
    /// Writes f32 (default) or f16 (MINFER_CACHE_TYPE=f16) into the KV cache.
    pub fn store_kv(&self, src: &metal::Buffer, dst: &metal::Buffer, nkt: usize, nt: usize,
        positions: &metal::Buffer, off: usize,
    ) {
        self.trace_op("store_kv");
        self.enc.set_compute_pipeline_state(
            if kv_cache_is_f16() { &self.state.pl_store_kv_f16 } else { &self.state.pl_store_kv }
        );
        self.enc.set_buffer(0, Some(src), (off * 4) as u64);
        self.enc.set_buffer(1, Some(dst), 0);
        self.set_params(2, &(nkt as i32));
        self.set_params(3, &(nt as i32));
        self.enc.set_buffer(4, Some(positions), 0);
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
    pub fn attn_parallel_prefill(&self, q: &metal::Buffer, kv_k: &metal::Buffer, kv_v: &metal::Buffer,
        out: &metal::Buffer, positions: &metal::Buffer,
        nkv: usize, nkt: usize, _nqt: usize, nt: usize, nh: usize,
        hd: usize, gqa: usize, scale: f32,
    ) {
        self.trace_op("attn_parallel");
        let dev = &self.state.device;
        let scores = MpsState::get_or_grow(&self.state.buf_attn_scores,
            (nt * nh * nkv * 4) as u64, dev);

        // pass 1: scores [nt*nh][nkv] — one 256-thread TG per (t,h) row
        self.enc.set_compute_pipeline_state(&self.state.pl_attn_scores);
        self.enc.set_buffer(0, Some(q), 0);
        self.enc.set_buffer(1, Some(kv_k), 0);
        self.enc.set_buffer(2, Some(&scores), 0);
        self.set_params(3, &(nh as i32));
        self.set_params(4, &(hd as i32));
        self.set_params(5, &(nkv as i32));
        self.set_params(6, &(nt as i32));
        self.set_params(7, &(gqa as i32));
        self.set_params(8, &(nkt as i32));
        self.set_params(9, &(scale.to_bits() as i32));
        self.dispatch_2d((nt * nh) as u64, 1, 256, 1);

        // pass 2: masked softmax over kv per (t,h) row
        self.enc.set_compute_pipeline_state(&self.state.pl_softmax_attn);
        self.enc.set_buffer(0, Some(&scores), 0);
        self.enc.set_buffer(1, Some(positions), 0);
        self.set_params(2, &(nkv as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(nh as i32));
        self.enc.set_threadgroup_memory_length(0, 32 * 4);
        self.dispatch_2d((nt * nh) as u64, 1, 32, 8);

        // pass 3: out = softmax · V — one 256-thread TG per (t,h) row
        self.enc.set_compute_pipeline_state(&self.state.pl_attn_output);
        self.enc.set_buffer(0, Some(&scores), 0);
        self.enc.set_buffer(1, Some(kv_v), 0);
        self.enc.set_buffer(2, Some(out), 0);
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
    pub fn attn_flash_prefill(&self, q: &metal::Buffer, kv_k: &metal::Buffer, kv_v: &metal::Buffer,
        out: &metal::Buffer, positions: &metal::Buffer,
        nkv: usize, nkt: usize, nt: usize, nh: usize,
        nk: usize, hd: usize, scale: f32,
    ) {
        self.trace_op("attn_flash_blk");
        let dev = &self.state.device;
        let f16 = kv_cache_is_f16();
        let elem = if f16 { 2u64 } else { 4u64 };
        let pad = MpsState::get_or_grow(&self.state.buf_attn_pad,
            (2 * 64 * nkt as u64) * elem, dev);

        if nkv % 64 != 0 {
            self.enc.set_compute_pipeline_state(&self.state.pl_kv_tail_pad);
            self.enc.set_buffer(0, Some(kv_k), 0);
            self.enc.set_buffer(1, Some(kv_v), 0);
            self.enc.set_buffer(2, Some(&pad), 0);
            self.set_params(3, &(nkv as i32));
            self.set_params(4, &(nkt as i32));
            self.set_params(5, &(if f16 { 1 } else { 0 }));
            self.dispatch_2d(nkt as u64, 64, 1, 1);
        }

        self.enc.set_compute_pipeline_state(
            if f16 {
                if hd == 128 { &self.state.pl_flash_attn_blk_hd128_f16 } else { &self.state.pl_flash_attn_blk_f16 }
            } else {
                if hd == 128 { &self.state.pl_flash_attn_blk_hd128 } else { &self.state.pl_flash_attn_blk }
            }
        );
        self.enc.set_buffer(0, Some(q), 0);
        self.enc.set_buffer(1, Some(kv_k), 0);
        self.enc.set_buffer(2, Some(kv_v), 0);
        self.enc.set_buffer(3, Some(&pad), 0);
        self.enc.set_buffer(4, Some(out), 0);
        self.enc.set_buffer(5, Some(positions), 0);
        self.set_params(6, &(nh as i32));
        self.set_params(7, &(nk as i32));
        self.set_params(8, &(hd as i32));
        self.set_params(9, &(scale.to_bits() as i32));
        self.set_params(10, &(nt as i32));
        self.set_params(11, &(nkv as i32));
        // shmem: hd=64: sq (512 half = 1024 B) | so (512 f32 = 2048 B) | ss (1024 f32 = 4096 B);
        //        hd=128: sq (1024 half = 2048 B) | so (1024 f32 = 4096 B) | ss (1024 f32 = 4096 B)
        let shmem = if hd == 128 { 10240u64 } else { 7168u64 };
        self.enc.set_threadgroup_memory_length(0, shmem);
        self.dispatch_2d(((nt + 7) / 8) as u64, nh as u64, 32, 4);
    }

    /// Fused bias-add + RoPE + KV-store for nt==1 decode: ONE kernel replaces
    /// add_bias×3 + rope×2 + store_kv×2 (7 dispatches). `bqkv` layout is
    /// [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt]; biases are the raw
    /// per-section buffers. `pos` = the single token position. The KV store
    /// writes f32 or f16 (per kv_cache_is_f16) into kv_k/kv_v.
    pub fn attn_bias_rope_store(&self,
        bqkv: &metal::Buffer,
        bias_q: &metal::Buffer, bq_off: u64,
        bias_k: &metal::Buffer, bk_off: u64,
        bias_v: &metal::Buffer, bv_off: u64,
        kv_k: &metal::Buffer, kv_v: &metal::Buffer,
        nqt: usize, nkt: usize, hd: usize,
        freq_base: f32, freq_scale: f32, pos: i32, rope_style: i32,
    ) {
        self.trace_op("attn_bias_rope_store");
        self.enc.set_compute_pipeline_state(&self.state.pl_attn_bsr);
        self.enc.set_buffer(0, Some(bqkv), 0);
        self.enc.set_buffer(1, Some(bias_q), bq_off);
        self.enc.set_buffer(2, Some(bias_k), bk_off);
        self.enc.set_buffer(3, Some(bias_v), bv_off);
        self.enc.set_buffer(4, Some(kv_k), 0);
        self.enc.set_buffer(5, Some(kv_v), 0);
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

    /// Commit GPU work and wait for completion using a semaphore completion handler.
    /// This avoids the ~20ms Metal scheduler wakeup overhead of wait_until_completed.
    pub fn submit(self) -> Result<(), String> {
        self.enc.end_encoding();

        // dispatch_semaphore_t is already a reference-counted opaque pointer.
        let sem = unsafe { dispatch_semaphore_create(0) };
        let sem_val = sem as usize;

        use block::ConcreteBlock;
        let blk = ConcreteBlock::new(move |_buf: &metal::CommandBufferRef| {
            unsafe { dispatch_semaphore_signal(sem_val as *mut std::ffi::c_void); }
        });
        let blk = blk.copy();
        self.cmd_buf.add_completed_handler(&blk);
        self.cmd_buf.commit();

        // Bounded wait (10 s). If the GPU hangs (hardware fault), the completion
        // handler never fires and we bail out instead of blocking forever.
        let timeout = unsafe { dispatch_time(0, 10_000_000_000i64) }; // 10 s from now
        let rc = unsafe { dispatch_semaphore_wait(sem, timeout) };
        unsafe { dispatch_release(sem); }

        if rc == 0 {
            // Command buffer finished (possibly with an error status).
            match self.cmd_buf.status() {
                metal::MTLCommandBufferStatus::Completed => Ok(()),
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
fn compile_metal_source(device: &metal::Device) -> Option<metal::Library> {
    let src = include_str!("metal.metal");
    let opts = metal::CompileOptions::new();
    match device.new_library_with_source(src, &opts) {
        Ok(l) => Some(l),
        Err(e) => { eprintln!("MPS: shader compilation failed: {}", e); None }
    }
}

/// Load the embedded precompiled metallib, falling back to a runtime source
/// compile when it is empty or fails to load.
#[cfg(target_os = "macos")]
fn load_embedded_or_source(device: &metal::Device, metallib: &[u8]) -> Option<metal::Library> {
    if !metallib.is_empty() {
        match device.new_library_with_data(metallib) {
            Ok(l) => return Some(l),
            Err(e) => eprintln!("MPS: precompiled metallib load failed ({e}) — falling back to source compile"),
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
            let device = metal::Device::system_default()?;

            // GPU trace capture: set MINFER_METAL_CAPTURE=1
            if std::env::var("MINFER_METAL_CAPTURE").is_ok() {
                let capture = metal::CaptureManager::shared();
                let desc = metal::CaptureDescriptor::new();
                desc.set_capture_device(&device);
                desc.set_destination(metal::MTLCaptureDestination::DeveloperTools);
                capture.start_capture(&desc).ok();
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
            let override_file = std::env::var("MINFER_METALLIB_FILE").ok().filter(|p| !p.is_empty());
            let lib = if let Some(path) = override_file {
                match std::fs::read(&path).ok().and_then(|b| device.new_library_with_data(&b).ok()) {
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
                let f = match lib.get_function(name, None) {
                    Ok(f) => f,
                    Err(e) => { eprintln!("MPS: no function '{}': {}", name, e); return None; }
                };
                match device.new_compute_pipeline_state_with_function(&f) {
                    Ok(p) => Some(p),
                    Err(e) => { eprintln!("MPS: pipeline '{}': {}", name, e); None }
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
            let pl_add      = get_pl("kernel_add_f32")?;
            let pl_add_bias = get_pl("kernel_add_bias_f32")?;
            let pl_mul      = get_pl("kernel_mul_f32")?;
            let pl_silu     = get_pl("kernel_silu_f32")?;
            let pl_swiglu   = get_pl("kernel_swiglu_f32")?;
            let pl_rope     = get_pl("kernel_rope_f32")?;
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
            let pl_attn_scores = get_pl("kernel_attn_scores")?;
            let pl_attn_output = get_pl("kernel_attn_output")?;
            let pl_softmax_attn = get_pl("kernel_softmax_attn")?;
            let pl_warmup = get_pl("kernel_warmup_read")?;
            let dummy_buf = device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared);
            let m = MpsStateInner {
                device: device.clone(),
                max_threadgroup_memory: device.max_threadgroup_memory_length(),
                queue: device.new_command_queue(),
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
            eprintln!("MPS: using Metal on {} (unified: {})",
                device.name(), if device.has_unified_memory() { "yes" } else { "no" });
            Some(MpsState { inner: m })
        }
    }

    pub fn get() -> Option<&'static Self> {
        MPS.get().and_then(|s| s.as_ref())
    }

    pub fn init() {
        MPS.get_or_init(|| {
            let s = Self::try_new();
            if s.is_some() { eprintln!("MPS: GPU acceleration enabled"); }
            else { eprintln!("MPS: not available, using CPU fallback"); }
            s
        });
    }

    pub fn has_weight(&self, name: &str) -> bool {
        #[cfg(not(target_os = "macos"))] { false }
        #[cfg(target_os = "macos")]
        { self.inner.weights.lock().unwrap().contains_key(name) }
    }

    /// Look up a registered weight's (buffer, byte offset) — used by the graph
    /// Metal backend to dispatch per-op kernels without holding the Tensor.
    pub fn weight_buf(&self, name: &str) -> Option<(metal::Buffer, u64)> {
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
    pub fn new_f32_buffer(&self, n_elements: usize) -> metal::Buffer {
        #[cfg(not(target_os = "macos"))]
        {
            unreachable!()
        }
        #[cfg(target_os = "macos")]
        {
            let bytes = (n_elements * 4) as u64;
            self.inner.device.new_buffer(bytes, metal::MTLResourceOptions::StorageModeShared)
        }
    }

    /// Register an mmap'd GGUF part for zero-copy weight wrapping. The part
    /// data pointer must be page-aligned (mmap returns page-aligned addresses):
    /// newBufferWithBytesNoCopy requires a page-aligned base (llama
    /// ggml_metal_buffer_map, ggml-metal-device.m:1701). Weights from this part
    /// are then registered as (buffer, offset) into this one buffer.
    pub fn register_part(&self, data: &'static [u8]) {
        #[cfg(not(target_os = "macos"))] { let _ = data; }
        #[cfg(target_os = "macos")]
        {
            if data.is_empty() { return; }
            let page = 16384; // macOS page size on Apple Silicon
            let base = data.as_ptr() as usize;
            debug_assert!(base % page == 0, "mmap'd GGUF part not page-aligned");
            let buf = self.inner.device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const std::ffi::c_void,
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
                None,
            );
            self.inner.mmap_parts.lock().unwrap().push((base, data.len(), buf.clone()));
            // GPU-side warm-up (METAL_OPTIMIZATIONS #39): the FIRST GPU access to
            // file-backed (mmap) pages costs ~44 ms of one-time page/TLB setup.
            // Doing a dummy full-buffer read HERE (at model load, outside the
            // CLI's Total timing) moves that cost out of the first prefill —
            // llama-bench's numbers are equally warm. ~5 ms bandwidth + the
            // setup, amortized into load.
            let cb = self.cmd_buffer();
            cb.trace_op("part_warmup");
            cb.enc.set_compute_pipeline_state(&self.inner.pl_warmup);
            cb.enc.set_buffer(0, Some(&buf), 0);
            let tiny = self.inner.buf_positions.lock().unwrap().clone();
            cb.enc.set_buffer(1, Some(&tiny), 0);
            let n = (buf.length() / 4) as u64;
            cb.dispatch_1d((n + 255) / 256, 256);
            let _ = cb.submit();
        }
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        #[cfg(not(target_os = "macos"))] {}
        #[cfg(target_os = "macos")]
        {
            if data.is_empty() { return; }
            let ptr = data.as_ptr() as usize;
            let force_copy = std::env::var("MINFER_WEIGHT_COPY").map_or(false, |v| v == "1");
            // Zero-copy path: the weight is a slice of a registered mmap'd part
            // → (part buffer, offset). The GPU reads the mapped file pages
            // directly (llama's shared mmap buffer, ggml-metal-device.m:1668) —
            // no CPU→GPU memcpy, no GPU-side allocation.
            let entry = if !force_copy {
                let parts = self.inner.mmap_parts.lock().unwrap();
                parts.iter().find(|(base, len, _)| {
                    ptr >= *base && ptr + data.len() <= base + len
                }).map(|(base, _, buf)| (buf.clone(), (ptr - base) as u64))
            } else { None };
            let (buf, off) = match entry {
                Some(e) => e,
                None => {
                    // Fallback: copy into a fresh per-weight buffer (offset 0).
                    let b = self.inner.device.new_buffer(
                        data.len() as u64,
                        metal::MTLResourceOptions::StorageModeShared,
                    );
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            data.as_ptr(),
                            b.contents() as *mut u8,
                            data.len(),
                        );
                    }
                    (b, 0u64)
                }
            };
            self.inner.weights.lock().unwrap().insert(name.to_string(), (buf, off));
        }
    }

    /// Create a command buffer for batching operations.
    pub fn cmd_buffer(&self) -> MpsCommandBuffer<'_> {
        #[cfg(not(target_os = "macos"))] { unreachable!() }
        #[cfg(target_os = "macos")]
        {
            let cmd_buf_ref = self.inner.queue.new_command_buffer();
            let enc_ref = cmd_buf_ref.new_compute_command_encoder();
            // The metal crate returns autoreleased objects (`commandBuffer`, not
            // `newCommandBuffer`). Retain so the cb survives the creating
            // thread's autorelease-pool drain when it crosses threads; Drop
            // releases.
            unsafe {
                let _: *mut metal::objc::runtime::Object = msg_send![cmd_buf_ref, retain];
                let _: *mut metal::objc::runtime::Object = msg_send![enc_ref, retain];
            }
            MpsCommandBuffer { state: &self.inner, cmd_buf: cmd_buf_ref, enc: enc_ref }
        }
    }

    /// Return a buffer with at least `need` bytes, growing the persistent pool
    /// if necessary. The underlying allocation is reused across calls.
    fn get_or_grow(
        slot: &std::sync::Mutex<metal::Buffer>,
        need: u64,
        dev: &metal::Device,
    ) -> metal::Buffer {
        {
            let b = slot.lock().unwrap();
            if b.length() >= need {
                return b.clone();
            }
        }
        let new = dev.new_buffer(need, metal::MTLResourceOptions::StorageModeShared);
        *slot.lock().unwrap() = new.clone();
        new
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression guard: the Metal shader program (metal.metal) is compiled at
    /// RUNTIME by `try_new` — `cargo build` does NOT catch shader errors. A
    /// duplicate/missing kernel or a Metal compile error makes `MpsState::init`
    /// fall back to CPU silently, which looks like a "GPU throttling" slowdown
    /// (2026-08-06: the Q5_0 `block_q5_0_dot_y` redefine bug did exactly this).
    /// This test compiles every pipeline and fails if MPS is unavailable.
    #[test]
    fn metal_pipelines_compile() {
        let _g = crate::metal::metal_test_lock();
        MpsState::init();
        assert!(
            MpsState::get().is_some(),
            "MPS unavailable — Metal shader compilation failed (check src/metal.metal for \
             duplicate/missing kernel definitions); the model would run on CPU"
        );
    }

    /// Batched-cb bandwidth profile of each nt==1 matmul kernel (decode path).
    /// Dispatches the SAME matmul N times in one command buffer (per the
    /// 2026-08-03 methodology: a single dispatch is dominated by the ~165 µs
    /// cb launch+sync floor — batch dozens before trusting a per-matmul time),
    /// then reports GB/s of weight reads. Goal (2026-08-06 #1): find whether any
    /// specific matmul (output/QKV/O/GU/down) is far below the ~200 GB/s floor.
    #[test]
    fn matmul_bandwidth_profile() {
        let _g = crate::metal::metal_test_lock();
        MpsState::init();
        let mps = MpsState::get().expect("MPS must be active for the bandwidth profile");
        let dev = &mps.inner.device;

        // (label, ttype, od, id, batches) — Qwen2.5-0.5B Q4_K_M + 7B Q4_K decode dims.
        let cases: &[(&str, TensorType, usize, usize, usize)] = &[
            ("QKV  q5_0 (od=1152,id=896)",  TensorType::Q5_0, 1152,   896, 400),
            ("O    q5_0 (od=896, id=896)",  TensorType::Q5_0,  896,   896, 400),
            ("GU   q5_0 (od=9728,id=896)",  TensorType::Q5_0, 9728,   896, 100),
            ("down q6_K (od=896, id=4864)", TensorType::Q6_K,  896,  4864, 200),
            ("out  q8_0 (od=151936,id=896)", TensorType::Q8_0, 151936, 896, 6),
            // Q4_0 kernel (the "fast interleaved-ushort" one) at the SAME small
            // dims — isolates whether the low GB/s is the kernel or the small-od structure.
            ("QKV  q4_0 (od=1152,id=896)",  TensorType::Q4_0, 1152,   896, 400),
            ("GU   q4_0 (od=9728,id=896)",  TensorType::Q4_0, 9728,   896, 100),
            ("out  q4_0 (od=151936,id=896)", TensorType::Q4_0, 151936, 896, 6),
            // Q5_1 shares Q5_0's qh (variable-shift) handling + an m term.
            ("QKV  q5_1 (od=1152,id=896)",  TensorType::Q5_1, 1152,   896, 400),
            // Qwen2.5-7B Q4_K decode dims (to-do #7 pre-port baseline):
            // attn_q/attn_output (3584/3584), attn_k (3584/512), ffn_gate/up (18944/3584).
            ("7B attn_q q4_K (3584/3584)",   TensorType::Q4_K, 3584,  3584, 400),
            ("7B attn_k q4_K (3584/512)",    TensorType::Q4_K, 3584,   512, 400),
            ("7B ffn_g/u q4_K (18944/3584)", TensorType::Q4_K, 18944, 3584, 100),
        ];

        println!("\n=== nt==1 matmul bandwidth profile (batched cb, M4 Pro) ===");
        // Warm up the first pipeline (Q4_0) so the first measured case isn't a cold start.
        {
            let wb = dev.new_buffer(65536, metal::MTLResourceOptions::StorageModeShared);
            let acts = dev.new_buffer(4096, metal::MTLResourceOptions::StorageModeShared);
            let out = dev.new_buffer(65536, metal::MTLResourceOptions::StorageModeShared);
            let cb = mps.cmd_buffer();
            for _ in 0..50 { cb.matmul_on_gpu_buf(&wb, 0, TensorType::Q4_0, &acts, &acts, 0, &out, 2048, 128, 1); }
            cb.submit().expect("warmup");
        }
        for &(label, ttype, od, id, n) in cases {
            let bq = quant_block_q(ttype);
            let bb = quant_block_bytes(ttype);
            let nblocks = (id + bq - 1) / bq;
            let wbytes = nblocks * bb * od;
            let wb = dev.new_buffer(wbytes as u64, metal::MTLResourceOptions::StorageModeShared);
            // Deterministic fill: d bytes 0x3333 (finite half), data nibbles 3.
            unsafe { std::slice::from_raw_parts_mut(wb.contents() as *mut u8, wbytes).fill(0x33); }
            let acts = dev.new_buffer((id * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
            unsafe {
                let p = acts.contents() as *mut f32;
                for i in 0..id { *p.add(i) = 0.5; }
            }
            let out = dev.new_buffer((od * 4) as u64, metal::MTLResourceOptions::StorageModeShared);

            // Warm this kernel with a discard batch (GPU clock/pipeline ramp-up —
            // the first measurement of a kernel is up to ~4x slow otherwise).
            {
                let cb = mps.cmd_buffer();
                for _ in 0..(n / 2).max(16) { cb.matmul_on_gpu_buf(&wb, 0, ttype, &acts, &acts, 0, &out, od, id, 1); }
                cb.submit().expect("warmup");
            }

            // Measure TWICE; report the second (warm) value — even after the
            // warmup batch the very first timed cb can still be slow.
            let mut warm_gbs = 0.0f64;
            for rep in 0..2 {
                let cb = mps.cmd_buffer();
                for _ in 0..n {
                    cb.matmul_on_gpu_buf(&wb, 0, ttype, &acts, &acts, 0, &out, od, id, 1);
                }
                let t0 = std::time::Instant::now();
                cb.submit().expect("submit");
                let dt = t0.elapsed().as_secs_f64();
                warm_gbs = wbytes as f64 * n as f64 / dt / 1e9;
                if rep == 0 {
                    println!(
                        "  {label:<26} (cold run {rep}: {:>5.0} GB/s) — warming…",
                        warm_gbs
                    );
                }
            }
            println!(
                "  {label:<26} {:>7.1} MB  x{n:>3} = {:>6.0} MB  {:>5.0} GB/s  (warm)",
                wbytes as f64 / 1e6, wbytes as f64 * n as f64 / 1e6, warm_gbs
            );
        }
        println!("=== end profile ===");
    }

    /// Batched-cb per-kernel GPU time profile of the NON-MATMUL decode kernels
    /// (rms_norm, add, add_bias, swiglu, rope, store_kv, BSR, attn partial +
    /// combine). P0 of the "per-kernel GPU distribution" plan (2026-08-10):
    /// the final gap report says the ~1.2 ms non-matmul tail is "~340 kernels at
    /// ~4x llama" but the per-kernel distribution was UNKNOWN (xctrace CLI can't
    /// give per-kernel durations). This batches each kernel dozens-hundreds of
    /// times in ONE command buffer (the 2026-08-03 methodology: single-dispatch
    /// timing has a ~165 us cb launch+sync floor; batch to amortize), warms each
    /// kernel, measures twice, and reports per-kernel GPU time in us.
    #[test]
    fn non_matmul_bandwidth_profile() {
        let _g = crate::metal::metal_test_lock();
        MpsState::init();
        let mps = MpsState::get().expect("MPS must be active");
        let dev = &mps.inner.device;

        // Qwen2.5-0.5B decode dims.
        let (ne, nqt, nkt, nf) = (896usize, 896usize, 128usize, 4864usize);
        let (nh, nk, hd) = (14usize, 2usize, 64usize);
        let nkv = 430usize; // long-ish context (matches the -n 512 avg)

        // Shared activation buffers (f32).
        let x  = dev.new_buffer((ne * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let y  = dev.new_buffer((ne * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let bqkv = dev.new_buffer(((nqt + 2 * nkt) * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let w  = dev.new_buffer((ne * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let g  = dev.new_buffer((nf * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let u  = dev.new_buffer((nf * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let bq = dev.new_buffer((nqt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let bk = dev.new_buffer((nkt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let bv = dev.new_buffer((nkt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let kv  = dev.new_buffer((nkv * nkt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pos = dev.new_buffer(4, metal::MTLResourceOptions::StorageModeShared);
        let o   = dev.new_buffer((ne * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        // finite fill (0.5) so no denormal/NaN paths skew timing
        for b in [&x, &y, &bqkv, &w, &g, &u, &bq, &bk, &bv, &kv, &o] {
            unsafe { std::slice::from_raw_parts_mut(b.contents() as *mut f32, (b.length() / 4) as usize).fill(0.5); }
        }
        unsafe { std::slice::from_raw_parts_mut(pos.contents() as *mut i32, 1)[0] = (nkv - 1) as i32; }

        // (label, dispatch closure, batches) — each closure dispatches ONE kernel.
        let cases: Vec<(&str, Box<dyn Fn(&MpsCommandBuffer)>, usize)> = vec![
            ("rms_norm 32t  (d=896, 1 row)",  Box::new(|cb| cb.rms_norm(&x, Some(&w), 0, &y, ne, 1, 1e-6, 0)), 400),
            ("rms_norm 256t (d=896, 1 row)",  Box::new(|cb| cb.rms_norm_256(&x, Some(&w), 0, &y, ne, 1, 1e-6, 0)), 400),
            ("add_f32 (n=896, 256t)",        Box::new(|cb| cb.add_f32(&x, &y, &x, ne)), 400),
            ("add_bias_f32 (d=896, 64t)",    Box::new(|cb| cb.add_bias_f32(&x, &w, 0, ne, 1, 0)), 400),
            ("swiglu_f32 (n=4864, 256t)",    Box::new(|cb| cb.swiglu_f32(&g, &u, &g, nf)), 400),
            ("rope_f32 (q: 14h x 64d)",      Box::new(|cb| cb.rope_f32(&bqkv, nh, hd, 1, 1e6, 1.0, &pos, 0, 0)), 400),
            ("store_kv (nkt=128, 1t)",       Box::new(|cb| cb.store_kv(&bk, &kv, nkt, 1, &pos, 0)), 400),
            ("attn_bsr (q+k+v, 256t)",       Box::new(|cb| cb.attn_bias_rope_store(&bqkv, &bq, 0, &bk, 0, &bv, 0, &kv, &kv, nqt, nkt, hd, 1e6, 1.0, (nkv - 1) as i32, 0)), 400),
            // split = partial + combine as a PAIR (2 dispatches/layer, decode path)
            ("attn split p+c (c=16)",        Box::new(|cb| cb.gqa_attn_split_f32(&bqkv, &kv, &kv, &o, &pos, nh, nk, hd, 0.125, 1, 16)), 100),
        ];

        println!("\n=== nt==1 non-matmul GPU profile (batched cb, M4 Pro) ===");
        // warm the whole pipeline once
        {
            let cb = mps.cmd_buffer();
            for _ in 0..50 { cb.rms_norm(&x, Some(&w), 0, &y, ne, 1, 1e-6, 0); }
            cb.submit().expect("warmup");
        }

        for (i, (label, dispatch, n)) in cases.iter().enumerate() {
            // warm this kernel (pipeline/clock ramp)
            {
                let cb = mps.cmd_buffer();
                for _ in 0..(n / 2).max(16) { dispatch(&cb); }
                cb.submit().expect("warmup");
            }
            // median of 3 warm runs (the docs' methodology: batched-cb per-kernel
            // numbers vary ~2x run-to-run due to GPU clock — take the median).
            let mut us: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let cb = mps.cmd_buffer();
                for _ in 0..*n { dispatch(&cb); }
                let t0 = std::time::Instant::now();
                cb.submit().expect("submit");
                let dt = t0.elapsed().as_secs_f64();
                us.push(dt * 1e6 / *n as f64);
            }
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let med = us[1];
            println!("  [{i}] {label:<26} {:>7.2} us/kernel  (median, n={}, [{:.2},{:.2},{:.2}])",
                med, n, us[0], us[1], us[2]);
        }

        // Classic single-pass attention (baseline for the split pair):
        let cases2: Vec<(&str, Box<dyn Fn(&MpsCommandBuffer)>, usize)> = vec![
            ("attn classic (nkv=430)", Box::new(|cb| cb.gqa_attn_f32(&bqkv, &kv, &kv, &o, &pos, nh, nk, hd, 0.125, 1)), 100),
        ];
        for (i, (label, dispatch, n)) in cases2.iter().enumerate() {
            {
                let cb = mps.cmd_buffer();
                for _ in 0..(n / 2).max(16) { dispatch(&cb); }
                cb.submit().expect("warmup");
            }
            let mut us: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let cb = mps.cmd_buffer();
                for _ in 0..*n { dispatch(&cb); }
                let t0 = std::time::Instant::now();
                cb.submit().expect("submit");
                let dt = t0.elapsed().as_secs_f64();
                us.push(dt * 1e6 / *n as f64);
            }
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            println!("  [{i}] {label:<26} {:>7.2} us/kernel  (median, n={}, [{:.2},{:.2},{:.2}])",
                us[1], n, us[0], us[1], us[2]);
        }
        println!("=== end non-matmul profile ===");
    }

    /// Correctness of the 256-thread multi-simdgroup rms_norm vs a scalar CPU
    /// reference (and vs the 32-thread kernel). P1 (2026-08-10): the multi-
    /// simdgroup reduction (shmem + 2 barriers) is the riskiest new piece —
    /// must be byte-deterministic before it touches the decode path.
    #[test]
    fn rms_norm_256_correctness() {
        let _g = crate::metal::metal_test_lock();
        MpsState::init();
        let mps = MpsState::get().expect("MPS must be active");
        let dev = &mps.inner.device;
        let d = 896usize;

        let x = dev.new_buffer((d * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let w = dev.new_buffer((d * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let y32 = dev.new_buffer((d * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let y256 = dev.new_buffer((d * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        // Deterministic input: x = sin(i), w = cos(i/7) — exercises varied magnitudes.
        unsafe {
            let xp = x.contents() as *mut f32;
            let wp = w.contents() as *mut f32;
            for i in 0..d {
                *xp.add(i) = (i as f32 * 0.37).sin() * 3.0;
                *wp.add(i) = (i as f32 / 7.0).cos() + 1.0;
            }
        }

        for (label, buf, method) in [
            ("32t", &y32, 0),
            ("256t", &y256, 1),
        ] {
            let cb = mps.cmd_buffer();
            if method == 0 { cb.rms_norm(&x, Some(&w), 0, buf, d, 1, 1e-6, 0); }
            else { cb.rms_norm_256(&x, Some(&w), 0, buf, d, 1, 1e-6, 0); }
            cb.submit().expect("submit");
        }

        // CPU scalar reference: scale = 1/sqrt(mean(x^2)+eps); y = x*scale*w.
        let xs: Vec<f32> = (0..d).map(|i| (i as f32 * 0.37).sin() * 3.0).collect();
        let ws: Vec<f32> = (0..d).map(|i| (i as f32 / 7.0).cos() + 1.0).collect();
        let mean: f32 = xs.iter().map(|v| v * v).sum::<f32>() / d as f32;
        let scale = 1.0f32 / (mean + 1e-6f32).sqrt();
        let mut ref_y = vec![0.0f32; d];
        for i in 0..d { ref_y[i] = xs[i] * scale * ws[i]; }

        for (label, buf) in [("32t", &y32), ("256t", &y256)] {
            let mut got = vec![0.0f32; d];
            unsafe { std::ptr::copy_nonoverlapping(buf.contents() as *const f32, got.as_mut_ptr(), d); }
            let mut maxd = 0.0f32;
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            for i in 0..d {
                maxd = maxd.max((got[i] - ref_y[i]).abs());
                dot += got[i] * ref_y[i];
                na += got[i] * got[i];
            }
            let cos = dot / (na.sqrt() * ref_y.iter().map(|v| v * v).sum::<f32>().sqrt());
            println!("  rms_norm {label}: maxdiff={maxd:.3e} cos={cos:.9}");
            assert!(cos > 0.9999, "rms_norm {label} wrong vs CPU (cos={cos})");
            assert!(maxd < 1e-3, "rms_norm {label} maxdiff {maxd} > 1e-3");
        }
        // 32t vs 256t should be bit-close (same math, different reduction order).
        let mut y32v = vec![0.0f32; d];
        let mut y256v = vec![0.0f32; d];
        unsafe {
            std::ptr::copy_nonoverlapping(y32.contents() as *const f32, y32v.as_mut_ptr(), d);
            std::ptr::copy_nonoverlapping(y256.contents() as *const f32, y256v.as_mut_ptr(), d);
        }
        let maxdd: f32 = (0..d).map(|i| (y32v[i] - y256v[i]).abs()).fold(0.0, f32::max);
        println!("  rms_norm 32t vs 256t maxdiff={maxdd:.3e}");
        assert!(maxdd < 1e-3, "32t vs 256t diverge (maxdiff {maxdd})");
    }


    /// Correctness of the 3-pass parallel prefill attention vs a CPU reference
    /// using REAL dumped layer-0 activations (q, k, v). P1.
    #[test]
    fn attn_parallel_realdata_correctness() {
        let _g = crate::metal::metal_test_lock();
        use std::io::Read;
        let dir = std::env::var("MINFER_TEST_DUMP").unwrap_or_else(|_| "/tmp/dp3".into());
        // The dump files are generated by a debug_dump GPU run; skip (not fail)
        // when they are absent, like the other fixture-dependent tests.
        let (bq_path, bk_path, bv_path) = (
            format!("{dir}/minfer_gpu_dump_layer0_bq.f32"),
            format!("{dir}/minfer_gpu_dump_layer0_bk.f32"),
            format!("{dir}/minfer_gpu_dump_layer0_bv.f32"),
        );
        if !std::path::Path::new(&bq_path).exists()
            || !std::path::Path::new(&bk_path).exists()
            || !std::path::Path::new(&bv_path).exists()
        {
            eprintln!("layer-0 dump files not found in {dir}; skipping realdata attention test");
            return;
        }
        let mut bq = Vec::new();
        std::fs::File::open(bq_path).unwrap().read_to_end(&mut bq).unwrap();
        let bq: Vec<f32> = bq.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let mut bk = Vec::new();
        std::fs::File::open(bk_path).unwrap().read_to_end(&mut bk).unwrap();
        let bk: Vec<f32> = bk.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let mut bv = Vec::new();
        std::fs::File::open(bv_path).unwrap().read_to_end(&mut bv).unwrap();
        let bv: Vec<f32> = bv.chunks_exact(4).map(|c| f32::from_le_bytes([c[0],c[1],c[2],c[3]])).collect();
        let (nh, nk, hd, nkt, nqt) = (14usize, 2usize, 64usize, 128usize, 896usize);
        let nt = bq.len() / nqt;
        let nkv = bk.len() / nkt;
        let gqa = nh / nk;
        let scale = 1.0 / (hd as f32).sqrt();
        assert_eq!(nt, 35);
        MpsState::init();
        let mps = MpsState::get().expect("MPS");
        let dev = &mps.inner.device;
        let qb = dev.new_buffer_with_data(bq.as_ptr() as *const _, (bq.len()*4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let kb = dev.new_buffer_with_data(bk.as_ptr() as *const _, (bk.len()*4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let vb = dev.new_buffer_with_data(bv.as_ptr() as *const _, (bv.len()*4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let ob = dev.new_buffer((nt*nqt*4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pb = dev.new_buffer((nt*4) as u64, metal::MTLResourceOptions::StorageModeShared);
        unsafe { for t in 0..nt { (pb.contents() as *mut i32).add(t).write(t as i32); } }
        let cb = mps.cmd_buffer();
        cb.attn_parallel_prefill(&qb, &kb, &vb, &ob, &pb, nkv, nkt, nqt, nt, nh, hd, gqa, scale);
        cb.submit().expect("submit");
        let got: Vec<f32> = unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, nt*nqt) }.to_vec();
        let got: Vec<f32> = unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, nt*nqt) }.to_vec();
        let nan = got.iter().filter(|v| !v.is_finite()).count();
        println!("  realdata parallel: nan={nan} of {}", nt * nqt);
        assert!(nan == 0, "realdata parallel produced NaN");
        // CPU reference
        let mut ref_out = vec![0.0f32; nt*nqt];
        let mut scrs = vec![0.0f32; nkv];
        for h in 0..nh {
            let hk = h / gqa;
            for t in 0..nt {
                let qq = t*nqt + h*hd;
                let vl = (t+1).min(nkv);
                let mut mx = f32::NEG_INFINITY;
                for kv in 0..vl {
                    let ks_ = kv*nkt + hk*hd;
                    let s = (0..hd).map(|d| bq[qq+d]*bk[ks_+d]).sum::<f32>()*scale;
                    scrs[kv] = s; if s > mx { mx = s; }
                }
                for kv in vl..nkv { scrs[kv] = f32::NEG_INFINITY; }
                let mut sum = 0.0f32;
                for kv in 0..nkv { scrs[kv] = if scrs[kv]==f32::NEG_INFINITY {0.0} else {(scrs[kv]-mx).exp()}; sum += scrs[kv]; }
                for kv in 0..nkv { scrs[kv] /= sum; }
                let oo = t*nqt + h*hd;
                for d in 0..hd { ref_out[oo+d]=0.0; }
                for kv in 0..nkv { for d in 0..hd { ref_out[oo+d] += scrs[kv]*bv[kv*nkt+hk*hd+d]; } }
            }
        }
        let maxerr = (0..nt*nqt).map(|i| (got[i]-ref_out[i]).abs()).fold(0.0f32, f32::max);
        println!("  realdata parallel maxerr vs CPU: {maxerr:.5}");
        assert!(maxerr < 0.1, "realdata wrong (maxerr {maxerr})");
    }

    /// End-to-end correctness of the 3-pass parallel prefill attention vs a CPU
    /// scalar reference (the exact algorithm in forward.rs::gqa_attn). P1.
    #[test]
    fn attn_parallel_prefill_correctness() {
        let _g = crate::metal::metal_test_lock();
        MpsState::init();
        let mps = MpsState::get().expect("MPS must be active");
        let dev = &mps.inner.device;
        let (nh, nk, hd, nkt, nqt) = (14usize, 2usize, 64usize, 128usize, 896usize);
        let (nt, nkv_real) = (35usize, 35usize);
        let nkv_p = ((nkv_real + 31) / 32) * 32;
        let gqa = nh / nk;
        let scale = 1.0 / (hd as f32).sqrt();

        // deterministic q [nt][nqt], kv [nkv][nkt]
        let q = dev.new_buffer((nt * nqt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let k = dev.new_buffer((nkv_real * nkt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let v = dev.new_buffer((nkv_real * nkt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let out = dev.new_buffer((nt * nqt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        let pos = dev.new_buffer((nt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
        unsafe {
            let qp = q.contents() as *mut f32;
            for i in 0..(nt * nqt) { *qp.add(i) = ((i as f32) * 0.37).sin() * 1.5; }
            let kp = k.contents() as *mut f32;
            for i in 0..(nkv_real * nkt) { *kp.add(i) = ((i as f32) * 0.11).cos() * 1.2; }
            let vp = v.contents() as *mut f32;
            for i in 0..(nkv_real * nkt) { *vp.add(i) = ((i as f32) * 0.23).sin() * 0.9; }
            let pp = pos.contents() as *mut i32;
            for t in 0..nt { *pp.add(t) = t as i32; }
        }

        let cb = mps.cmd_buffer();
        cb.attn_parallel_prefill(&q, &k, &v, &out, &pos,
            nkv_real, nkt, nqt, nt, nh, hd, gqa, scale);
        cb.submit().expect("submit");

        // CPU reference (mirror of forward.rs::gqa_attn)
        let qs: Vec<f32> = (0..nt * nqt).map(|i| ((i as f32) * 0.37).sin() * 1.5).collect();
        let ks: Vec<f32> = (0..nkv_real * nkt).map(|i| ((i as f32) * 0.11).cos() * 1.2).collect();
        let vs: Vec<f32> = (0..nkv_real * nkt).map(|i| ((i as f32) * 0.23).sin() * 0.9).collect();
        let mut ref_out = vec![0.0f32; nt * nqt];
        let mut scrs = vec![0.0f32; nkv_real];
        for h in 0..nh {
            let hk = h / gqa;
            for t in 0..nt {
                let qq = t * nqt + h * hd;
                let vl = (t + 1).min(nkv_real);
                let mut mx = f32::NEG_INFINITY;
                for kv in 0..vl {
                    let ks_ = kv * nkt + hk * hd;
                    let s = (0..hd).map(|d| qs[qq + d] * ks[ks_ + d]).sum::<f32>() * scale;
                    scrs[kv] = s; if s > mx { mx = s; }
                }
                for kv in vl..nkv_real { scrs[kv] = f32::NEG_INFINITY; }
                let mut sum = 0.0f32;
                for kv in 0..nkv_real { scrs[kv] = if scrs[kv] == f32::NEG_INFINITY { 0.0 } else { (scrs[kv] - mx).exp() }; sum += scrs[kv]; }
                for kv in 0..nkv_real { scrs[kv] /= sum; }
                let oo = t * nqt + h * hd;
                for d in 0..hd { ref_out[oo + d] = 0.0; }
                for kv in 0..nkv_real {
                    let vbase = kv * nkt + hk * hd;
                    for d in 0..hd { ref_out[oo + d] += scrs[kv] * vs[vbase + d]; }
                }
            }
        }

        let mut got = vec![0.0f32; nt * nqt];
        unsafe { std::ptr::copy_nonoverlapping(out.contents() as *const f32, got.as_mut_ptr(), nt * nqt); }
        let mut maxerr = 0.0f32;
        for i in 0..nt * nqt {
            maxerr = maxerr.max((got[i] - ref_out[i]).abs());
        }
        println!("  attn_parallel_prefill: maxerr vs CPU {maxerr:.5}");
        assert!(maxerr < 0.1, "matmul attention wrong vs CPU (maxerr {maxerr})");
    }

    /// Prefill GEMM (nt=430) throughput — P1 prefill-gap investigation (2026-08-11):
    /// minfer pp430 ~1860 t/s vs llama-Metal ~6940 t/s (3.7x). GEMM params match
    /// (64x32 tile, 4 sg, both legacy-simdgroup on M4). This measures the GEMM
    /// kernel's achieved GB/s at the REAL Q4_K prefill dims to see if it's
    /// bandwidth-bound or latency/occupancy-bound vs llama.
    #[test]
    fn prefill_gemm_throughput_profile() {
        let _g = crate::metal::metal_test_lock();
        MpsState::init();
        let mps = MpsState::get().expect("MPS must be active");
        let dev = &mps.inner.device;
        // Qwen2.5-0.5B Q4_K_M prefill dims: attn_q=Q5_0 (od=896,id=896),
        // ffn_up=Q5_0 (od=18944,id=896), ffn_down=Q6_K (od=896,id=4864).
        // 7B Q4_K_M prefill GEMMs (od=ne[1], id=ne[0] from `minfer info`):
        //   q4_K: attn_q/attn_output (3584/3584), attn_k (512/3584), ffn_gate/up (18944/3584)
        //   q6_K: attn_v (512/3584), ffn_down (3584/18944), output (152064/3584)
        // Use the 64x32-tile GEMM (nt>=16) which is what prefill uses.
        let cases: &[(&str, TensorType, usize, usize)] = &[
            ("attn_q Q5_0  od=896    id=896   nt=430", TensorType::Q5_0, 896, 896),
            ("attn_q Q4_0  od=896    id=896   nt=430", TensorType::Q4_0, 896, 896),
            ("ffn_up Q5_0  od=18944  id=896   nt=430", TensorType::Q5_0, 18944, 896),
            ("ffn_up Q4_0  od=18944  id=896   nt=430", TensorType::Q4_0, 18944, 896),
            ("down  Q6_K   od=896    id=4864  nt=430", TensorType::Q6_K, 896, 4864),
            // 7B prefill GEMMs (2026-08-18, llama test-backend-ops A/B)
            ("7B attn_q Q4_K od=3584   id=3584  nt=430", TensorType::Q4_K, 3584, 3584),
            ("7B attn_k Q4_K od=512    id=3584  nt=430", TensorType::Q4_K, 512, 3584),
            ("7B ffn_gu Q4_K od=18944  id=3584  nt=430", TensorType::Q4_K, 18944, 3584),
            ("7B attn_v Q6_K od=512    id=3584  nt=430", TensorType::Q6_K, 512, 3584),
            ("7B ffn_down Q6_K od=3584 id=18944 nt=430", TensorType::Q6_K, 3584, 18944),
            ("7B output Q6_K od=152064 id=3584  nt=430", TensorType::Q6_K, 152064, 3584),
        ];
        let nt = 430usize;
        println!("\n=== prefill GEMM throughput (nt=430, batched cb) ===");
        for &(label, ttype, od, id) in cases {
            let bq = quant_block_q(ttype);
            let bb = quant_block_bytes(ttype);
            let nblocks = (id + bq - 1) / bq;
            let wbytes = nblocks * bb * od;
            let wb = dev.new_buffer(wbytes as u64, metal::MTLResourceOptions::StorageModeShared);
            // Fill with valid finite weights: each block's d (first 2 bytes, fp16)
            // = 1.0 (0x00 0x3C LE), remaining bytes 0x33 (finite nibbles). Avoids
            // the denormal-fp16 slow path that skews GEMM timing.
            unsafe { std::slice::from_raw_parts_mut(wb.contents() as *mut u8, wbytes).fill(0x33); }
            {
                let p = wb.contents() as *mut u8;
                let row = nblocks * bb;
                for r in 0..od {
                    for b in 0..nblocks {
                        let off = (r * row + b * bb) as isize;
                        unsafe { *p.offset(off) = 0x00; *p.offset(off + 1) = 0x3C; }
                    }
                }
            }
            let acts = dev.new_buffer((id * nt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
            unsafe { std::slice::from_raw_parts_mut(acts.contents() as *mut f32, id * nt).fill(0.5); }
            let out = dev.new_buffer((od * nt * 4) as u64, metal::MTLResourceOptions::StorageModeShared);
            // warm
            {
                let cb = mps.cmd_buffer();
                for _ in 0..20 { cb.quant_matmul_f32_on_gpu_buf(&wb, 0, ttype, &acts, 0, &out, od, id, nt); }
                cb.submit().expect("warmup");
            }
            let n = 50;
            let mut us: Vec<f64> = Vec::new();
            for _ in 0..3 {
                let cb = mps.cmd_buffer();
                for _ in 0..n { cb.quant_matmul_f32_on_gpu_buf(&wb, 0, ttype, &acts, 0, &out, od, id, nt); }
                let t0 = std::time::Instant::now();
                cb.submit().expect("submit");
                let dt = t0.elapsed().as_secs_f64();
                us.push(dt * 1e6 / n as f64);
            }
            us.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let per_us = us[1]; // median
            let gbs = wbytes as f64 / (per_us * 1e-6) / 1e9;
            let tflops = 2.0 * od as f64 * id as f64 * nt as f64 / (per_us * 1e-6) / 1e12;
            println!("  {label:<32} {:.1} MB {per_us:>8.1} us => {:>5.0} GB/s  {:>5.2} TFLOPS (warm, n=50)", wbytes as f64/1e6, gbs, tflops);
        }
    }
}

#[cfg(test)]
mod mmap_align_test {
    #[test]
    fn nocopy_alignment_probe() {
        let _g = crate::metal::metal_test_lock();
        let dev = metal::Device::system_default().unwrap();
        // A 16-aligned Vec base + 32 → 32-aligned, NOT 256-aligned
        let mut backing = vec![0u8; 8192 + 64];
        let base = backing.as_mut_ptr() as usize;
        let aligned = (base + 63) & !31usize; // 32-aligned pointer
        assert!(aligned % 32 == 0 && aligned % 256 != 0, "need a non-256-aligned 32-aligned ptr");
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(aligned as *mut u8, 4096) };
        for i in 0..4096 { buf_slice[i] = (i & 0xFF) as u8; }
        let b = dev.new_buffer_with_bytes_no_copy(
            aligned as *const std::ffi::c_void, 4096,
            metal::MTLResourceOptions::StorageModeShared, None);
        let contents = unsafe { std::slice::from_raw_parts(b.contents() as *const u8, 4096) };
        let mut ok = contents.len() == 4096;
        for i in 0..4096 { if contents[i] != (i & 0xFF) as u8 { ok = false; break; } }
        println!("CPU readback: ok={ok}");
        // GPU readback: dispatch a trivial copy kernel reading the buffer
        let lib = dev.new_library_with_source(
            "kernel void k(const device uchar *in [[buffer(0)]], device uchar *out [[buffer(1)]]) { out[0] = in[3]; }",
            &metal::CompileOptions::new()).unwrap();
        let pl = dev.new_compute_pipeline_state_with_function(&lib.get_function("k", None).unwrap()).unwrap();
        let out = dev.new_buffer(16, metal::MTLResourceOptions::StorageModeShared);
        let q = dev.new_command_queue();
        let cb = q.new_command_buffer();
        let enc = cb.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&pl);
        enc.set_buffer(0, Some(&b), 0);
        enc.set_buffer(1, Some(&out), 0);
        enc.dispatch_thread_groups(metal::MTLSize::new(1,1,1), metal::MTLSize::new(1,1,1));
        enc.end_encoding();
        cb.commit();
        cb.wait_until_completed();
        let got = unsafe { *(out.contents() as *const u8) };
        println!("GPU readback (expect 3): got={got} -> {}", if got == 3 { "OK" } else { "WRONG" });
        assert!(got == 3, "GPU readback at 32-aligned nocopy base is WRONG");
        // Same probe over an mmap'd FILE region (the actual weights path)
        {
            let path = std::env::temp_dir().join("minfer_mmap_probe.bin");
            let mut f = std::fs::File::create(&path).unwrap();
            use std::io::Write;
            let mut blob = vec![0u8; 8192 + 64];
            for i in 0..blob.len() { blob[i] = (i & 0xFF) as u8; }
            f.write_all(&blob).unwrap();
            drop(f);
            use std::os::unix::io::AsRawFd;
            extern "C" { fn mmap(a: *mut std::ffi::c_void, l: usize, p: i32, f: i32, fd: i32, o: i64) -> *mut std::ffi::c_void; }
            let file = std::fs::File::open(&path).unwrap();
            let m = unsafe { mmap(std::ptr::null_mut(), 8192, 0x1, 0x0002, file.as_raw_fd(), 0) };
            assert!(m as isize != -1);
            let mbase = m as usize;
            let mptr = (mbase + 63) & !31usize; // 32-aligned, not page-aligned
            assert!(mptr % 256 != 0, "need non-256-aligned");
            let bm = dev.new_buffer_with_bytes_no_copy(
                mptr as *const std::ffi::c_void, 4096,
                metal::MTLResourceOptions::StorageModeShared, None);
            let out2 = dev.new_buffer(64, metal::MTLResourceOptions::StorageModeShared);
            let cb2 = q.new_command_buffer();
            let enc2 = cb2.new_compute_command_encoder();
            enc2.set_compute_pipeline_state(&pl);
            enc2.set_buffer(0, Some(&bm), 0);
            enc2.set_buffer(1, Some(&out2), 0);
            enc2.dispatch_thread_groups(metal::MTLSize::new(1,1,1), metal::MTLSize::new(1,1,1));
            enc2.end_encoding();
            cb2.commit();
            cb2.wait_until_completed();
            let got2 = unsafe { *(out2.contents() as *const u8) };
            println!("mmap GPU readback (expect 3): got={got2} -> {}", if got2 == 3 { "OK" } else { "WRONG" });
            let _ = std::fs::remove_file(&path);
        }
    }

    // ─── Fixture generator for attn_parallel_realdata_correctness ───────────
    // The realdata test consumes layer-0 q/k/v dumps that the (now deleted)
    // layer_gpu dump path used to write. This generator rebuilds them with the
    // current graph path: a real 35-token prefill on the cached 0.5B q4_0,
    // dumping the layer-0 q/k/v matmul outputs (pre-RoPE, token-major) as f32
    // files. Run once with:
    //   cargo test --bin minfer gen_layer0_realdata_dump -- --ignored
    // (writes to $MINFER_TEST_DUMP or /tmp/dp3, matching the test's default).
    fn cached_qwen05_q4_0() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
        if p.exists() { Some(p) } else { None }
    }

    fn write_f32(path: &str, data: &[f32]) {
        let mut b = Vec::with_capacity(data.len() * 4);
        for x in data {
            b.extend_from_slice(&x.to_le_bytes());
        }
        std::fs::write(path, b).expect("write dump");
    }

    #[test]
    #[ignore = "one-time fixture generator (writes /tmp/dp3 files)"]
    fn gen_layer0_realdata_dump() {
        use crate::graph::alloc::GraphAllocator;
        use crate::graph::builder::GraphBuilder;
        use crate::graph::scheduler::BackendScheduler;
        use crate::graph::DType;
        use crate::models::qwen2::graph::Qwen2Graph;
        use crate::models::qwen2::Qwen2Model;
        use crate::models::ModelDef;

        let Some(path) = cached_qwen05_q4_0() else {
            eprintln!("0.5B q4_0 not cached; skipping fixture generator");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let q2: &Qwen2Model = model.as_any().downcast_ref::<Qwen2Model>().expect("qwen2");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);

        const NT: usize = 35; // the realdata test asserts nt == 35
        let mut ids = tok.encode(
            "The capital of France is Paris and the capital of Germany is Berlin. \
             The capital of Italy is Rome and the capital of Spain is Madrid and \
             the capital of the United Kingdom is London and the capital of Japan is Tokyo.",
        );
        assert!(ids.len() >= NT, "prompt tokenizes to {} < {NT} tokens", ids.len());
        ids.truncate(NT);

        let hp = &q2.hparams;
        let nh = hp.n_head as usize;
        let nk = hp.n_head_kv as usize;
        let hd = hp.n_embd_head() as usize;
        let nqt = nh * hd;
        let nkt = nk * hd;
        let l0 = &q2.layers[0];

        // embedding -> rms_norm -> q/k/v matmul. No RoPE on purpose: the
        // fixtures are pre-RoPE projections (the attention kernel and the CPU
        // reference in the realdata test don't apply RoPE).
        let mut b = GraphBuilder::new();
        let ids_n = b.input("token_ids", [NT, 1, 1, 1], DType::I32);
        let h = b.embedding(ids_n, q2.tok_embd.as_ref().unwrap());
        let normed = b.rms_norm(h, l0.attn_norm.as_ref(), hp.f_norm_rms_eps);
        let qn = b.matmul(normed, l0.wq.as_ref().unwrap(), l0.bq.as_ref());
        let kn = b.matmul(normed, l0.wk.as_ref().unwrap(), l0.bk.as_ref());
        let vn = b.matmul(normed, l0.wv.as_ref().unwrap(), l0.bv.as_ref());
        b.output(qn);
        b.output(kn);
        b.output(vn);
        let mut graph = b.build();

        let mut alloc = GraphAllocator::new();
        Qwen2Graph::register_graph_weights(q2, &mut alloc);
        let sched = BackendScheduler::new();
        sched.assign_backends(&mut graph, &mut alloc);
        alloc.alloc_graph(&graph).unwrap();
        alloc.fill_input_i32(&graph, "token_ids", &ids).unwrap();
        sched.execute(&graph, &mut alloc).unwrap();

        let q = alloc.copy_to_cpu(qn).expect("q buffer");
        let k = alloc.copy_to_cpu(kn).expect("k buffer");
        let v = alloc.copy_to_cpu(vn).expect("v buffer");
        assert_eq!(q.len(), NT * nqt, "q dims");
        assert_eq!(k.len(), NT * nkt, "k dims");
        assert_eq!(v.len(), NT * nkt, "v dims");
        eprintln!("[gen dump] layer0 q/k/v: {NT}x{nqt} / {NT}x{nkt} / {NT}x{nkt} (q[0]={})", q[0]);

        let dir = std::env::var("MINFER_TEST_DUMP").unwrap_or_else(|_| "/tmp/dp3".into());
        std::fs::create_dir_all(&dir).unwrap();
        write_f32(&format!("{dir}/minfer_gpu_dump_layer0_bq.f32"), &q);
        write_f32(&format!("{dir}/minfer_gpu_dump_layer0_bk.f32"), &k);
        write_f32(&format!("{dir}/minfer_gpu_dump_layer0_bv.f32"), &v);
        eprintln!("[gen dump] wrote {dir}/minfer_gpu_dump_layer0_b{{q,k,v}}.f32");
    }
}
