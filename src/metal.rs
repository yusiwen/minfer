// MPS (Metal) backend for Apple Silicon.
// Translated from: llama.cpp/ggml/src/ggml-metal
//
// Provides MpsCommandBuffer for batching all layer ops into one GPU submission.

use std::sync::OnceLock;
use crate::tensor::{Tensor, TensorType};
use crate::block::Q8B;
#[cfg(target_os = "macos")]
use metal::objc::{msg_send, sel, sel_impl};

static MPS: OnceLock<Option<MpsState>> = OnceLock::new();

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

/// KV cache element type for the GPU path. Defaults to f32; `MINFER_CACHE_TYPE=f16`
/// switches to a half cache (llama.cpp's default). f16 halves attention memory
/// bandwidth but showed a ~15% decode regression on the 0.5B model (decode is
/// dispatch-latency-bound, not KV-bandwidth-bound), so it is opt-in.
pub fn kv_cache_is_f16() -> bool {
    static F16: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *F16.get_or_init(|| std::env::var("MINFER_CACHE_TYPE").map_or(false, |v| v == "f16"))
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
    max_threads_per_threadgroup: u32,
    queue: metal::CommandQueue,
    pl_q4_0_q8: metal::ComputePipelineState,
    pl_q4_0_q8_multi: metal::ComputePipelineState,
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
    pl_rms_norm: metal::ComputePipelineState,
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
    pl_store_kv: metal::ComputePipelineState,
    pl_store_kv_f16: metal::ComputePipelineState,
    pl_attn_bsr: metal::ComputePipelineState,
    weights: std::sync::Mutex<std::collections::HashMap<String, metal::Buffer>>,
    // Persistent scratch buffers grown on demand; avoids per-call allocation.
    q8_buf: std::sync::Mutex<metal::Buffer>,
    out_buf: std::sync::Mutex<metal::Buffer>,
    // Pool of output buffers reused by batch matmuls (one slot per batch entry).
    out_pool: std::sync::Mutex<Vec<metal::Buffer>>,
    // Persistent activation buffers reused across transformer layers.
    buf_hidden: std::sync::Mutex<metal::Buffer>,
    buf_bn: std::sync::Mutex<metal::Buffer>,
    buf_bq: std::sync::Mutex<metal::Buffer>,
    buf_bk: std::sync::Mutex<metal::Buffer>,
    buf_bv: std::sync::Mutex<metal::Buffer>,
    buf_bqkv: std::sync::Mutex<metal::Buffer>,
    buf_ba: std::sync::Mutex<metal::Buffer>,
    buf_attn_partial: std::sync::Mutex<metal::Buffer>,
    buf_bf: std::sync::Mutex<metal::Buffer>,
    buf_bg: std::sync::Mutex<metal::Buffer>,
    buf_bgu: std::sync::Mutex<metal::Buffer>,
    buf_q8_bn: std::sync::Mutex<metal::Buffer>,
    buf_q8_ba: std::sync::Mutex<metal::Buffer>,
    buf_positions: std::sync::Mutex<metal::Buffer>,
    buf_token_ids: std::sync::Mutex<metal::Buffer>,
    buf_logits: std::sync::Mutex<metal::Buffer>,
    // Persistent per-layer GPU KV cache (k, v) and current size in KV positions.
    kv_k: std::sync::RwLock<Vec<metal::Buffer>>,
    kv_v: std::sync::RwLock<Vec<metal::Buffer>>,
    kv_size: std::sync::RwLock<Vec<usize>>,
    // Ring of recent dispatch op labels (for GPU-fault diagnosis, MINFER_TRACE only).
    dispatch_trace: std::sync::Mutex<std::collections::VecDeque<String>>,
}

/// Decode profiling gates (subtractive per-token timing). Each flag skips one
/// kernel group during decode (nt==1) when `MINFER_SKIP_{ATTN,MATMULS,SMALL}=1`.
/// The env is read once per process into a OnceLock (mirroring the MINFER_TRACE
/// pattern), so normal decode has ~zero overhead. Usage in the dispatch code is
/// constrained to gate each kernel IN ITS EXACT ORIGINAL POSITION — never move a
/// dispatch relative to its neighbors (a grouped gate moved the FFN down-matmul
/// before swiglu on 2026-08-06 and corrupted output).
#[derive(Clone, Copy, Default)]
pub struct DecodeSkips {
    pub attn: bool,
    pub matmul: bool,
    pub small: bool,
}

#[cfg(target_os = "macos")]
impl DecodeSkips {
    pub fn active(nt: usize) -> Self {
        if nt != 1 {
            return DecodeSkips::default();
        }
        static CACHE: std::sync::OnceLock<DecodeSkips> = std::sync::OnceLock::new();
        *CACHE.get_or_init(|| DecodeSkips {
            attn: std::env::var("MINFER_SKIP_ATTN").map_or(false, |v| v == "1"),
            matmul: std::env::var("MINFER_SKIP_MATMULS").map_or(false, |v| v == "1"),
            small: std::env::var("MINFER_SKIP_SMALL").map_or(false, |v| v == "1"),
        })
    }
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

    fn dispatch_2d(&self, w: u64, h: u64, tw: u64, th: u64) {
        self.enc.dispatch_thread_groups(
            metal::MTLSize { width: w, height: h, depth: 1 },
            metal::MTLSize { width: tw, height: th, depth: 1 },
        );
    }

    fn dispatch_3d(&self, w: u64, h: u64, d: u64, tw: u64, th: u64, td: u64) {
        self.enc.dispatch_thread_groups(
            metal::MTLSize { width: w, height: h, depth: d },
            metal::MTLSize { width: tw, height: th, depth: td },
        );
    }

    /// GEMM kernels (prefill nt>=16) are enabled unless MINFER_GEMM=0.
    fn gemm_enabled() -> bool {
        static GEMM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *GEMM.get_or_init(|| std::env::var("MINFER_GEMM").map_or(true, |v| v != "0"))
    }

    /// Dispatch a 64×32-tile simdgroup GEMM (NT≥16 prefill). GPU safety: the
    /// kernels stage 8 KB of threadgroup memory (sa 4 KB + sb 2 KB + bc_out
    /// 8 KB reusing sa/sb) — verified against the queried device limit.
    fn gemm_dispatch(&self, pl: &metal::ComputePipelineState, wb: &metal::Buffer,
        x: &metal::Buffer, out: &metal::Buffer, od: usize, id: usize, nt: usize,
    ) {
        if 8192 > self.state.max_threadgroup_memory {
            gpu_abort(&format!(
                "GEMM needs 8192 B threadgroup memory, device max is {} B",
                self.state.max_threadgroup_memory
            ));
        }
        self.enc.set_compute_pipeline_state(pl);
        self.enc.set_buffer(0, Some(wb), 0);
        self.enc.set_buffer(1, Some(x), 0);
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
    }

    /// Dispatch Q4_0/Q4_1/Q4_K/Q8_0 × f32 matmul (activations are f32).
    /// Q4_0/Q4_1: NR0=4, NSG=2, TG=64 threads, grid x = ceil(od / 8).
    /// Q4_K    : NR0=2, NSG=2, TG=64 threads, grid x = ceil(od / 4).
    /// Q8_0    : NR0=2, NSG=4, TG=128 threads, grid x = ceil(od / 2),
    ///           uses 256 bytes of threadgroup memory for cross-simdgroup reduction.
    pub fn quant_matmul_f32_on_gpu(&self, w: &Tensor, x: &metal::Buffer, out: &metal::Buffer,
        od: usize, id: usize, nt: usize,
    ) {
        let weights = self.state.weights.lock().unwrap();
        let wb = weights.get(&w.name).expect("weight not on GPU");
        self.quant_matmul_f32_on_gpu_buf(wb, w.ttype, x, out, od, id, nt);
    }

    pub fn quant_matmul_f32_on_gpu_buf(&self, wb: &metal::Buffer, ttype: TensorType,
        x: &metal::Buffer, out: &metal::Buffer, od: usize, id: usize, nt: usize,
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
                if nt >= 16 && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q8_0_mm_f32, wb, x, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q8_0_f32_multi } else { &self.state.pl_q8_0_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
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
                if nt >= 16 && Self::gemm_enabled() {
                    // both Q4_K and Q6_K have simdgroup GEMMs
                    let pl = if ttype == TensorType::Q6_K { &self.state.pl_q6_k_mm_f32 } else { &self.state.pl_q4_k_mm_f32 };
                    self.gemm_dispatch(pl, wb, x, out, od, id, nt);
                } else {
                    let pl: &metal::ComputePipelineState = if ttype == TensorType::Q4_K {
                        if nt > 1 { &self.state.pl_q4_k_f32_multi } else { &self.state.pl_q4_k_f32 }
                    } else {
                        if nt > 1 { &self.state.pl_q6_k_f32_multi } else { &self.state.pl_q6_k_f32 }
                    };
                    self.enc.set_compute_pipeline_state(pl);
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 3) / 4) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q4_1 => {
                if nt >= 16 && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q4_1_mm_f32, wb, x, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q4_1_f32_multi } else { &self.state.pl_q4_1_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_0 => {
                if nt >= 16 && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q5_0_mm_f32, wb, x, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q5_0_f32_multi } else { &self.state.pl_q5_0_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_1 => {
                if nt >= 16 && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q5_1_mm_f32, wb, x, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q5_1_f32_multi } else { &self.state.pl_q5_1_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
                    self.enc.set_buffer(2, Some(out), 0);
                    let mm_p = [od as i32, id as i32, nt as i32];
                    self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                    let grid_y = if nt > 1 { 1 } else { nt as u64 };
                    self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
                }
            }
            TensorType::Q5_K => {
                if nt >= 16 && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q5_k_mm_f32, wb, x, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q5_k_f32_multi } else { &self.state.pl_q5_k_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
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
                if nt >= 16 && Self::gemm_enabled() {
                    self.gemm_dispatch(&self.state.pl_q4_0_mm_f32, wb, x, out, od, id, nt);
                } else {
                    self.enc.set_compute_pipeline_state(
                        if nt > 1 { &self.state.pl_q4_0_f32_multi } else { &self.state.pl_q4_0_f32 }
                    );
                    self.enc.set_buffer(0, Some(wb), 0);
                    self.enc.set_buffer(1, Some(x), 0);
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
                self.enc.set_buffer(0, Some(wb), 0);
                self.enc.set_buffer(1, Some(x), 0);
                self.enc.set_buffer(2, Some(out), 0);
                let mm_p = [od as i32, id as i32, nt as i32];
                self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
                let grid_y = if nt > 1 { 1 } else { nt as u64 };
                self.dispatch_2d(((od + 7) / 8) as u64, grid_y, 64, 1);
            }
        }
    }

    /// Dispatch Q4_0 × Q8_0 matmul (bit-exact with CPU path).
    pub fn quant_matmul_q8(&self, w: &Tensor, x: &metal::Buffer, out: &metal::Buffer,
        od: usize, id: usize, nt: usize,
    ) {
        let weights = self.state.weights.lock().unwrap();
        let wb = weights.get(&w.name).expect("weight not on GPU");
        self.quant_matmul_q8_buf(wb, x, out, od, id, nt);
    }

    pub fn quant_matmul_q8_buf(&self, wb: &metal::Buffer, x: &metal::Buffer,
        out: &metal::Buffer, od: usize, id: usize, nt: usize,
    ) {
        if nt > 1 {
            self.enc.set_compute_pipeline_state(&self.state.pl_q4_0_q8_multi);
            self.enc.set_buffer(0, Some(wb), 0);
            self.enc.set_buffer(1, Some(x), 0);
            self.enc.set_buffer(2, Some(out), 0);
            let mm_p = [od as i32, id as i32, nt as i32];
            self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
            self.dispatch_2d((od as u64 + 7) / 8, 1, 64, 1);
        } else {
            self.enc.set_compute_pipeline_state(&self.state.pl_q4_0_q8);
            self.enc.set_buffer(0, Some(wb), 0);
            self.enc.set_buffer(1, Some(x), 0);
            self.enc.set_buffer(2, Some(out), 0);
            let mm_p = [od as i32, id as i32, nt as i32];
            self.enc.set_bytes(3, 12, mm_p.as_ptr() as *const std::ffi::c_void);
            self.dispatch_2d((od as u64 + 7) / 8, nt as u64, 64, 1);
        }
    }

    /// Choose the f32-activation matmul for all weight types (including Q4_0,
    /// matching llama.cpp's Metal backend which does not Q8_0-quantize activations).
    /// Pre-looked-up weight buffer and type — avoids per-matmul HashMap locking.
    fn matmul_on_gpu_buf(&self, wb: &metal::Buffer, ttype: TensorType,
        _q8_x: &metal::Buffer, f32_x: &metal::Buffer, out: &metal::Buffer,
        od: usize, id: usize, nt: usize,
    ) {
        self.quant_matmul_f32_on_gpu_buf(wb, ttype, f32_x, out, od, id, nt);
    }

    /// GPU embedding lookup: dequantize Q4_0 embedding rows for nt token ids.
    /// Writes f32 hidden state [nt][ne] to dst (buf_hidden).
    pub fn embed_tokens_gpu(&self, wb: &metal::Buffer, ids: &metal::Buffer,
        dst: &metal::Buffer, ne: usize, nt: usize,
    ) {
        self.trace_op("embed");
        self.enc.set_compute_pipeline_state(&self.state.pl_get_rows_q4_0);
        self.enc.set_buffer(0, Some(wb), 0);
        self.enc.set_buffer(1, Some(ids), 0);
        self.enc.set_buffer(2, Some(dst), 0);
        self.set_params(3, &(ne as i32));
        self.set_params(4, &(nt as i32));
        let nb = ne / 32;
        self.dispatch_1d((nt * nb) as u64, 256);
    }

    /// RMSNorm: y = x * rsqrt(mean(x²)+eps) * w
    pub fn rms_norm(&self, x: &metal::Buffer, w: Option<&metal::Buffer>, y: &metal::Buffer,
        d: usize, n: usize, eps: f32,
    ) {
        self.trace_op("rms_norm");
        self.enc.set_compute_pipeline_state(&self.state.pl_rms_norm);
        self.enc.set_buffer(0, Some(x), 0);
        self.enc.set_buffer(1, Some(w.unwrap_or(y)), 0); // dummy if no weight
        self.enc.set_buffer(2, Some(y), 0);
        self.set_params(3, &(d as i32));
        self.set_params(4, &(eps.to_bits() as i32));
        self.dispatch_2d(n as u64, 1, 32, 1);
    }

    /// Element-wise add: z = x + y
    pub fn add_f32(&self, x: &metal::Buffer, y: &metal::Buffer, z: &metal::Buffer, n: usize) {
        self.trace_op("add");
        self.enc.set_compute_pipeline_state(&self.state.pl_add);
        self.enc.set_buffer(0, Some(x), 0);
        self.enc.set_buffer(1, Some(y), 0);
        self.enc.set_buffer(2, Some(z), 0);
        self.set_params(3, &(n as i32));
        // float4 kernel: 4 elements/thread (ceil for the scalar tail)
        self.dispatch_1d(((n as u64) + 3) / 4, 256);
    }

    /// Add 1-D bias to rows: y[t][i] += b[i]. `off` = element offset into `y`
    /// (used by the fused QKV path to bias the q/k/v sections of one buffer).
    pub fn add_bias_f32(&self, y: &metal::Buffer, b: &metal::Buffer, d: usize, n: usize,
        off: usize,
    ) {
        self.trace_op("bias");
        self.enc.set_compute_pipeline_state(&self.state.pl_add_bias);
        self.enc.set_buffer(0, Some(y), (off * 4) as u64);
        self.enc.set_buffer(1, Some(b), 0);
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
        self.trace_op("gqa_attn");
        let gqa = nh / nk;
        self.enc.set_compute_pipeline_state(
            if kv_cache_is_f16() { &self.state.pl_gqa_attn_f16 } else { &self.state.pl_gqa_attn }
        );
        self.enc.set_buffer(0, Some(q), 0);
        self.enc.set_buffer(1, Some(k), 0);
        self.enc.set_buffer(2, Some(v), 0);
        self.enc.set_buffer(3, Some(o), 0);
        self.enc.set_buffer(4, Some(positions), 0);
        self.set_params(5, &(nh as i32));
        self.set_params(6, &(nk as i32));
        self.set_params(7, &(hd as i32));
        self.set_params(8, &(scale.to_bits() as i32));
        self.set_params(9, &(nt as i32));
        const Bc: u64 = 32;
        let shmem = Bc * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
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
        const Bc: u64 = 32;
        let shmem = Bc * hd as u64 * 2 * std::mem::size_of::<f32>() as u64;
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

    /// Fused bias-add + RoPE + KV-store for nt==1 decode: ONE kernel replaces
    /// add_bias×3 + rope×2 + store_kv×2 (7 dispatches). `bqkv` layout is
    /// [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt]; biases are the raw
    /// per-section buffers. `pos` = the single token position. The KV store
    /// writes f32 or f16 (per kv_cache_is_f16) into kv_k/kv_v.
    pub fn attn_bias_rope_store(&self,
        bqkv: &metal::Buffer,
        bias_q: &metal::Buffer, bias_k: &metal::Buffer, bias_v: &metal::Buffer,
        kv_k: &metal::Buffer, kv_v: &metal::Buffer,
        nqt: usize, nkt: usize, hd: usize,
        freq_base: f32, freq_scale: f32, pos: i32, rope_style: i32,
    ) {
        self.trace_op("attn_bias_rope_store");
        self.enc.set_compute_pipeline_state(&self.state.pl_attn_bsr);
        self.enc.set_buffer(0, Some(bqkv), 0);
        self.enc.set_buffer(1, Some(bias_q), 0);
        self.enc.set_buffer(2, Some(bias_k), 0);
        self.enc.set_buffer(3, Some(bias_v), 0);
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

            let src = include_str!("metal.metal");
            let opts = metal::CompileOptions::new();
            let lib = match device.new_library_with_source(src, &opts) {
                Ok(l) => l,
                Err(e) => { eprintln!("MPS: shader compilation failed: {}", e); return None; }
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

            let pl_q4_0_q8 = get_pl("kernel_q4_0_q8_0_matmul")?;
            let pl_q4_0_q8_multi = get_pl("kernel_q4_0_q8_0_matmul_multi")?;
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
            let pl_rms_norm = get_pl("kernel_rms_norm_f32")?;
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
            let pl_store_kv = get_pl("kernel_store_kv_f32")?;
            let pl_store_kv_f16 = get_pl("kernel_store_kv_f16")?;
            let pl_attn_bsr = get_pl("kernel_attn_bias_rope_store")?;
            let dummy_buf = device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared);
            let m = MpsStateInner {
                device: device.clone(),
                max_threadgroup_memory: device.max_threadgroup_memory_length(),
                max_threads_per_threadgroup: {
                    let t = device.max_threads_per_threadgroup();
                    (t.width * t.height * t.depth) as u32
                },
                queue: device.new_command_queue(),
                pl_q4_0_q8,
                pl_q4_0_q8_multi,
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
                pl_rms_norm,
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
                pl_store_kv,
                pl_store_kv_f16,
                pl_attn_bsr,
                weights: std::sync::Mutex::new(std::collections::HashMap::new()),
                q8_buf: std::sync::Mutex::new(dummy_buf.clone()),
                out_buf: std::sync::Mutex::new(dummy_buf.clone()),
                out_pool: std::sync::Mutex::new(Vec::new()),
                buf_hidden: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bn: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bq: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bk: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bv: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bqkv: std::sync::Mutex::new(dummy_buf.clone()),
                buf_ba: std::sync::Mutex::new(dummy_buf.clone()),
                buf_attn_partial: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bf: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bg: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bgu: std::sync::Mutex::new(dummy_buf.clone()),
                buf_q8_bn: std::sync::Mutex::new(dummy_buf.clone()),
                buf_q8_ba: std::sync::Mutex::new(dummy_buf.clone()),
                buf_positions: std::sync::Mutex::new(dummy_buf.clone()),
                buf_token_ids: std::sync::Mutex::new(dummy_buf.clone()),
                buf_logits: std::sync::Mutex::new(dummy_buf.clone()),
                kv_k: std::sync::RwLock::new(Vec::new()),
                kv_v: std::sync::RwLock::new(Vec::new()),
                kv_size: std::sync::RwLock::new(Vec::new()),
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

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        #[cfg(not(target_os = "macos"))] {}
        #[cfg(target_os = "macos")]
        {
            if data.is_empty() { return; }
            // Allocate a fresh GPU buffer and copy; the source Tensor Vec<u8> is untouched.
            let buf = self.inner.device.new_buffer(
                data.len() as u64,
                metal::MTLResourceOptions::StorageModeShared,
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buf.contents() as *mut u8,
                    data.len(),
                );
            }
            self.inner.weights.lock().unwrap().insert(name.to_string(), buf);
        }
    }

    /// Create a command buffer for batching operations.
    pub fn cmd_buffer(&self) -> MpsCommandBuffer {
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

    pub fn copy_to_gpu(src: &[f32], dst: &metal::Buffer) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr() as *const u8,
                dst.contents() as *mut u8,
                src.len() * 4,
            );
        }
    }

    pub fn copy_to_gpu_u8(src: &[u8], dst: &metal::Buffer) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.as_ptr(),
                dst.contents() as *mut u8,
                src.len(),
            );
        }
    }

    pub fn copy_from_gpu_u8(src: &metal::Buffer, dst: &mut [u8]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.contents() as *const u8,
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
    }

    pub fn copy_from_gpu_u8_part(src: &metal::Buffer, dst: &mut [u8], offset: u64, len: u64) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                (src.contents() as *const u8).add(offset as usize),
                dst.as_mut_ptr(),
                len as usize,
            );
        }
    }

    pub fn get_weight(&self, name: &str) -> Option<metal::Buffer> {
        #[cfg(not(target_os = "macos"))] { None }
        #[cfg(target_os = "macos")]
        {
            self.inner.weights.lock().unwrap().get(name).cloned()
        }
    }

    pub fn copy_from_gpu(src: &metal::Buffer, dst: &mut [f32]) {
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.contents() as *const f32,
                dst.as_mut_ptr(),
                dst.len(),
            );
        }
    }

    /// Copy `count` f16 elements from a GPU buffer, converting to f32.
    /// Used to read back the F16 KV cache for CPU fallback.
    pub fn copy_from_gpu_half_to_f32(src: &metal::Buffer, dst: &mut [f32], count: usize) {
        let p = src.contents() as *const u16;
        for i in 0..count {
            // SAFETY: the buffer holds at least `count` f16 values.
            dst[i] = f32::from(half::f16::from_bits(unsafe { *p.add(i) }));
        }
    }

    /// Create a temporary GPU buffer from CPU data (for norm weights, biases, etc.)
    pub fn temp_buffer(&self, data: &[f32]) -> metal::Buffer {
        #[cfg(not(target_os = "macos"))] { unreachable!() }
        #[cfg(target_os = "macos")]
        {
            self.inner.device.new_buffer_with_data(
                data.as_ptr() as *const std::ffi::c_void,
                (data.len() * 4) as u64,
                metal::MTLResourceOptions::StorageModeShared,
            )
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

    /// Batch several Q4_0 × f32 matmuls that share the same activation.
    /// Quantizes once, uploads once, encodes into one command buffer, submits once.
    pub fn quant_matmul_f32_batch(
        &self,
        mats: &mut [(/*weight*/ &Tensor, /*output*/ &mut [f32], /*od*/ usize)],
        x: &[f32], id: usize, nt: usize,
    ) {
        if mats.iter().any(|mat| mat.0.ttype != TensorType::Q4_0) {
            for mat in mats.iter_mut() {
                crate::kernel::cpu_quant_matmul_f32(mat.0, x, mat.1, mat.2, id, nt);
            }
            return;
        }

        let nb = id / 32;
        let q8_len = (nt * nb * Q8B) as u64;
        let mut q8 = vec![0u8; q8_len as usize];
        crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut q8);

        let dev = &self.inner.device;
        let xbuf = Self::get_or_grow(&self.inner.q8_buf, q8_len, dev);
        Self::copy_to_gpu_u8(&q8, &xbuf);

        let cb = self.cmd_buffer();

        // Acquire/grow persistent output buffers for this batch, then release
        // the pool lock before submitting GPU work.
        {
            let mut pool = self.inner.out_pool.lock().unwrap();
            let needed = mats.len();
            for _ in pool.len()..needed {
                pool.push(dev.new_buffer(1, metal::MTLResourceOptions::StorageModeShared));
            }
            for (i, mat) in mats.iter_mut().enumerate() {
                let out_len = (nt * mat.2 * std::mem::size_of::<f32>()) as u64;
                if pool[i].length() < out_len {
                    pool[i] = dev.new_buffer(out_len, metal::MTLResourceOptions::StorageModeShared);
                }
                cb.quant_matmul_q8(mat.0, &xbuf, &pool[i], mat.2, id, nt);
            }
        }
        cb.submit().unwrap_or_else(|e| {
            eprintln!("MPS: GPU submit error: {e}");
            std::process::exit(1);
        });

        {
            let pool = self.inner.out_pool.lock().unwrap();
            for (i, mat) in mats.iter_mut().enumerate() {
                Self::copy_from_gpu(&pool[i], mat.1);
            }
        }
    }

    /// Standalone Q4_0 × f32 matmul (CPU data → GPU → back).
    /// Quantizes activations to Q8_0 first so the GPU runs the same Q4_0×Q8_0
    /// dot product as the CPU AVX2 path.
    pub fn quant_matmul_f32(
        &self, w: &Tensor, x: &[f32], out: &mut [f32],
        od: usize, id: usize, nt: usize,
    ) {
        if w.ttype != TensorType::Q4_0 {
            return crate::kernel::cpu_quant_matmul_f32(w, x, out, od, id, nt);
        }

        let nb = id / 32;
        let q8_len = (nt * nb * Q8B) as u64;
        let out_len = (nt * od * std::mem::size_of::<f32>()) as u64;

        let mut q8 = vec![0u8; q8_len as usize];
        crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut q8);

        let dev = &self.inner.device;
        let xbuf = Self::get_or_grow(&self.inner.q8_buf, q8_len, dev);
        let obuf = Self::get_or_grow(&self.inner.out_buf, out_len, dev);

        Self::copy_to_gpu_u8(&q8, &xbuf);

        let cb = self.cmd_buffer();
        cb.quant_matmul_q8(w, &xbuf, &obuf, od, id, nt);
        cb.submit().unwrap_or_else(|e| {
            eprintln!("MPS: GPU submit error: {e}");
            std::process::exit(1);
        });
        Self::copy_from_gpu(&obuf, out);
    }

    // ─── Full-layer GPU pass (Phase 2) ─────────────────────────────────

    /// Upload the initial hidden state to GPU before the layer loop.
    pub fn upload_hidden(&self, hidden: &[f32]) {
        let buf = Self::get_or_grow(&self.inner.buf_hidden, (hidden.len() * 4) as u64, &self.inner.device);
        Self::copy_to_gpu(hidden, &buf);
    }

    /// Download the final hidden state from GPU after the layer loop.
    pub fn download_hidden(&self, hidden: &mut [f32]) {
        let buf = self.inner.buf_hidden.lock().unwrap();
        Self::copy_from_gpu(&buf, hidden);
    }

    /// Debug: download + dump layer-0 intermediates (mirrors the CPU path's
    /// minfer_dump_layer0_* dumps) so GPU vs CPU divergence can be localized.
    #[cfg(feature = "debug_dump")]
    pub fn dump_layer0_intermediates(&self, nt: usize, ne: usize, nqt: usize, nkt: usize, nf: usize) {
        use crate::dump;
        macro_rules! dump_buf {
            ($buf:expr, $name:expr, $n:expr) => {{
                let b = $buf.lock().unwrap();
                let mut data = vec![0.0f32; $n];
                Self::copy_from_gpu(&b, &mut data);
                dump::maybe_dump_prefill_or_gen0($name, &data, nt);
            }};
        }
        dump_buf!(self.inner.buf_bn, "minfer_gpu_dump_layer0_bn", nt * ne);
        dump_buf!(self.inner.buf_bq, "minfer_gpu_dump_layer0_bq", nt * nqt);
        dump_buf!(self.inner.buf_bk, "minfer_gpu_dump_layer0_bk", nt * nkt);
        dump_buf!(self.inner.buf_bv, "minfer_gpu_dump_layer0_bv", nt * nkt);
        dump_buf!(self.inner.buf_ba, "minfer_gpu_dump_layer0_ba", nt * ne);
        dump_buf!(self.inner.buf_bg, "minfer_gpu_dump_layer0_bg", nt * nf);
        dump_buf!(self.inner.buf_bf, "minfer_gpu_dump_layer0_bf", nt * nf);
    }

    /// Upload positions used by RoPE and causal attention for this forward call.
    pub fn upload_positions(&self, positions: &[usize]) {
        let need = (positions.len() * std::mem::size_of::<i32>()) as u64;
        let buf = Self::get_or_grow(&self.inner.buf_positions, need, &self.inner.device);
        let ints: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                ints.as_ptr(),
                buf.contents() as *mut i32,
                ints.len(),
            );
        }
    }

    /// Upload token ids for GPU-side embedding lookup.
    pub fn upload_token_ids(&self, token_ids: &[u32]) {
        let need = (token_ids.len() * std::mem::size_of::<i32>()) as u64;
        let buf = Self::get_or_grow(&self.inner.buf_token_ids, need, &self.inner.device);
        let ints: Vec<i32> = token_ids.iter().map(|&id| id as i32).collect();
        unsafe {
            std::ptr::copy_nonoverlapping(
                ints.as_ptr(),
                buf.contents() as *mut i32,
                ints.len(),
            );
        }
    }

    /// GPU embedding lookup: dequantize embedding rows and write to buf_hidden.
    /// Returns false if the embedding weight is not on GPU or not Q4_0.
    pub fn embed_tokens_gpu(&self, embd_weight: &Tensor, token_ids: &[u32], nt: usize, ne: usize) -> bool {
        // GPU safety (M2): kernel_get_rows_q4_0 indexes rows by token_id with no
        // in-kernel bound. A token_id >= vocab would read out of bounds — check
        // host-side and error-exit (the tokenizer normally guarantees valid ids).
        let vocab = embd_weight.shape[1] as usize;
        if let Some(&bad) = token_ids.iter().find(|&&id| id as usize >= vocab) {
            gpu_abort(&format!(
                "embedding token id {bad} >= vocab {vocab} (kernel_get_rows_q4_0 would read out of bounds)"
            ));
        }
        if embd_weight.ttype != TensorType::Q4_0 {
            return false;
        }
        let wb = match self.get_weight(&embd_weight.name) {
            Some(b) => b,
            None => return false,
        };
        self.upload_token_ids(token_ids);
        let dev = &self.inner.device;
        let hidden = Self::get_or_grow(&self.inner.buf_hidden, (nt * ne * 4) as u64, dev);
        let ids_buf = self.inner.buf_token_ids.lock().unwrap().clone();
        let cb = self.cmd_buffer();
        cb.embed_tokens_gpu(&wb, &ids_buf, &hidden, ne, nt);
        if cb.submit().is_err() {
            return false;
        }
        true
    }

    /// Ensure the GPU KV cache for layer `il` can hold at least `max_nkv` rows.
    fn kv_ensure_layer(&self, il: usize, max_nkv: usize, nkt: usize) {
        let elem = if kv_cache_is_f16() { 2u64 } else { 4u64 }; // F16 or F32 KV cache
        let need = (max_nkv * nkt) as u64 * elem;
        {
            let mut kvec = self.inner.kv_k.write().unwrap();
            let mut vvec = self.inner.kv_v.write().unwrap();
            let mut szvec = self.inner.kv_size.write().unwrap();
            while kvec.len() <= il {
                kvec.push(self.inner.device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared));
                vvec.push(self.inner.device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared));
                szvec.push(0);
            }
            if kvec[il].length() < need {
                // Grow GEOMETRICALLY (double) so the buffer isn't reallocated +
                // fully copied on EVERY decode token as the KV grows by one row
                // (that made the CPU encode cost O(n^2) in context length: a new
                // MTLBuffer + full old-KV memcpy per layer per token). Doubling
                // amortizes allocation+copy to O(log n) growth events.
                let new_len = need.max(kvec[il].length() * 2);
                // Preserve existing KV data when growing the buffer.
                let old_k = kvec[il].clone();
                let old_v = vvec[il].clone();
                let old_len = old_k.length().min(old_v.length());
                kvec[il] = self.inner.device.new_buffer(new_len, metal::MTLResourceOptions::StorageModeShared);
                vvec[il] = self.inner.device.new_buffer(new_len, metal::MTLResourceOptions::StorageModeShared);
                if old_len > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            old_k.contents() as *const u8,
                            kvec[il].contents() as *mut u8,
                            old_len as usize,
                        );
                        std::ptr::copy_nonoverlapping(
                            old_v.contents() as *const u8,
                            vvec[il].contents() as *mut u8,
                            old_len as usize,
                        );
                    }
                }
                // szvec[il] stays unchanged — existing KV entries remain valid.
            }
        }
    }

    /// Append one transformer layer to an existing command buffer. Attention + FFN,
    /// with hidden and KV cache kept on GPU. The caller is responsible for creating
    /// the command buffer (one per token) and committing it after all layers.
    pub fn layer_gpu(
        &self,
        cb: &MpsCommandBuffer,
        il: usize,
        l: &crate::models::qwen2::loader::LayerWeights,
        positions: &[usize],
        ne: usize, nqt: usize, nkt: usize, nf: usize, nt: usize,
        nh: usize, nk: usize, hd: usize,
        eps: f32, attn_scale: f32, freq_base: f32, freq_scale: f32,
        rope_style: i32,
    ) -> bool {
        let attn_norm = match &l.attn_norm { Some(t) => t, None => return false };
        let ffn_norm  = match &l.ffn_norm  { Some(t) => t, None => return false };
        let wq = l.wq.as_ref().unwrap();
        let wk = l.wk.as_ref().unwrap();
        let wv = l.wv.as_ref().unwrap();
        let wo = l.wo.as_ref().unwrap();
        let ffn_gate = l.ffn_gate.as_ref().unwrap();
        let ffn_up   = l.ffn_up.as_ref().unwrap();
        let ffn_down = l.ffn_down.as_ref().unwrap();

        // Runtime guards: the Metal kernels assume these constraints. On any
        // violation we error-exit (never silently fall back to CPU — that would
        // mask an unsupported GPU configuration). nh%nk==0 and hd<=256 are
        // required by the attention kernel (barrier participation + acc[256]);
        // id%32==0 by the quantized matmuls.
        if nh % nk != 0 {
            gpu_abort(&format!(
                "attention head mismatch: nh={nh}, nk={nk}, nh%nk != 0 (kernel_gqa_attn would deadlock the GPU)"
            ));
        }
        if hd > 256 {
            gpu_abort(&format!(
                "attention head dim {hd} > 256 (kernel_gqa_attn float acc[256] would overflow)"
            ));
        }
        if hd % 4 != 0 {
            gpu_abort(&format!(
                "attention head dim {hd} % 4 != 0 (kernel_gqa_attn_partial uses float4 vectorized acc)"
            ));
        }
        for (name, v) in [("ne", ne), ("nqt", nqt), ("nkt", nkt), ("nf", nf)] {
            if v % 32 != 0 {
                gpu_abort(&format!(
                    "{name}={v} is not 32-aligned (quantized matmul kernels would read out of bounds)"
                ));
            }
        }
        // Device-limit guards (queried values, not guessed): the attention
        // kernel's threadgroup needs 2*32*hd f32s and 32*gqa threads.
        let attn_smem = 2 * 32 * hd * 4;
        if attn_smem as u64 > self.inner.max_threadgroup_memory {
            gpu_abort(&format!(
                "attention needs {attn_smem} B threadgroup memory, device max is {} B",
                self.inner.max_threadgroup_memory
            ));
        }
        let gqa = nh / nk;
        if 32 * gqa as u32 > self.inner.max_threads_per_threadgroup {
            gpu_abort(&format!(
                "attention threadgroup needs {} threads (32 * gqa={}), device max is {}",
                 32 * gqa,
                 32 * gqa,
                self.inner.max_threads_per_threadgroup
            ));
        }
        // GPU safety (H1): the attention kernel strides the KV cache with
        // nk*hd, so it requires the KV head dim to equal the query head dim
        // (nkt == nk*hd). Models with a separate KV head dim would silently
        // read misaligned KV rows — refuse rather than risk wrong results.
        if nkt != nk * hd {
            gpu_abort(&format!(
                "attention KV head dim nkt={nkt} != nk*hd={} (kernel_gqa_attn strides KV by nk*hd; separate-KV-head models unsupported)",
                nk * hd
            ));
        }

        // Raw not supported in Metal shaders — fall back to CPU
        let all_types = [wq.ttype, wk.ttype, wv.ttype, wo.ttype,
                         ffn_gate.ttype, ffn_up.ttype, ffn_down.ttype];
        if all_types.iter().any(|t| *t == TensorType::Raw) {
            return false;
        }

        // Pre-lookup all weight buffers once to avoid per-matmul HashMap locking.
        let weights = self.inner.weights.lock().unwrap();
        let buf_wq = match weights.get(&wq.name) { Some(b) => b.clone(), None => return false };
        let buf_wk = match weights.get(&wk.name) { Some(b) => b.clone(), None => return false };
        let buf_wv = match weights.get(&wv.name) { Some(b) => b.clone(), None => return false };
        let buf_wo = match weights.get(&wo.name) { Some(b) => b.clone(), None => return false };
        let buf_fg = match weights.get(&ffn_gate.name) { Some(b) => b.clone(), None => return false };
        let buf_fu = match weights.get(&ffn_up.name) { Some(b) => b.clone(), None => return false };
        let buf_fd = match weights.get(&ffn_down.name) { Some(b) => b.clone(), None => return false };
        // Fused FFN gate+up buffer (registered at load when gate/up share a type).
        let buf_gu = weights.get(&format!("blk.{il}.ffn_gu")).cloned();
        let norm_attn_w = match weights.get(&attn_norm.name) { Some(b) => b.clone(), None => return false };
        let norm_ffn_w  = match weights.get(&ffn_norm.name)  { Some(b) => b.clone(), None => return false };
        let bq_bias = l.bq.as_ref().and_then(|b| weights.get(&b.name).cloned());
        let bk_bias = l.bk.as_ref().and_then(|b| weights.get(&b.name).cloned());
        let bv_bias = l.bv.as_ref().and_then(|b| weights.get(&b.name).cloned());
        // Fused QKV: concatenated Wq/Wk/Wv buffer (registered at load when the
        // three weights share a matmul type). One matmul → bqkv for nt==1 decode.
        let buf_qkv = weights.get(&format!("blk.{il}.attn_qkv")).cloned();
        drop(weights);
        if l.bq.is_some() && bq_bias.is_none() { return false; }
        if l.bk.is_some() && bk_bias.is_none() { return false; }
        if l.bv.is_some() && bv_bias.is_none() { return false; }

        let max_pos = positions.iter().copied().max().unwrap_or(0);
        self.kv_ensure_layer(il, max_pos + 1, nkt);

        let dev = &self.inner.device;
        let hidden_len = (nt * ne * 4) as u64;
        let bn_len = hidden_len;
        let bq_len = (nt * nqt * 4) as u64;
        let bk_len = (nt * nkt * 4) as u64;
        let bv_len = bk_len;
        let ba_len = (nt * ne * 4) as u64;
        let bf_len = (nt * nf.max(ne) * 4) as u64;
        let bg_len = (nt * nf * 4) as u64;
        let q8_bn_len = (nt * (ne / 32) * Q8B) as u64;
        let q8_ba_len = (nt * (nf.max(ne) / 32) * Q8B) as u64;

        let hidden = Self::get_or_grow(&self.inner.buf_hidden, hidden_len, dev);
        let bn = Self::get_or_grow(&self.inner.buf_bn, bn_len, dev);
        let bq_buf = Self::get_or_grow(&self.inner.buf_bq, bq_len, dev);
        let bk_buf = Self::get_or_grow(&self.inner.buf_bk, bk_len, dev);
        let bv_buf = Self::get_or_grow(&self.inner.buf_bv, bv_len, dev);
        let bqkv_len = (nt * (nqt + nkt + nkt) * 4) as u64;
        let bqkv_buf = Self::get_or_grow(&self.inner.buf_bqkv, bqkv_len, dev);
        let ba_buf = Self::get_or_grow(&self.inner.buf_ba, ba_len, dev);
        let bf_buf = Self::get_or_grow(&self.inner.buf_bf, bf_len, dev);
        let bg_buf = Self::get_or_grow(&self.inner.buf_bg, bg_len, dev);
        let bgu_len = (nt * (nf + nf) * 4) as u64;
        let bgu_buf = Self::get_or_grow(&self.inner.buf_bgu, bgu_len, dev);
        let q8_bn = Self::get_or_grow(&self.inner.buf_q8_bn, q8_bn_len, dev);
        let q8_ba = Self::get_or_grow(&self.inner.buf_q8_ba, q8_ba_len, dev);
        let pos_buf = self.inner.buf_positions.lock().unwrap();
        let kv_k = self.inner.kv_k.read().unwrap();
        let kv_v = self.inner.kv_v.read().unwrap();

        // Attention branch
        let attn_all_q4 = wq.ttype == TensorType::Q4_0 && wk.ttype == TensorType::Q4_0
            && wv.ttype == TensorType::Q4_0 && wo.ttype == TensorType::Q4_0;
        let attn_any_q4k = [wq.ttype, wk.ttype, wv.ttype, wo.ttype]
            .iter().any(|t| *t == TensorType::Q4_K || *t == TensorType::Q6_K);
        if !attn_all_q4 && attn_any_q4k {
            if wq.ttype == TensorType::Q4_0 || wk.ttype == TensorType::Q4_0
                || wv.ttype == TensorType::Q4_0 || wo.ttype == TensorType::Q4_0
                { return false; }
        }

        // ── decode profiling gates (subtractive per-token timing) ──────────
        // Flags are OnceLock-cached (see DecodeSkips). Each gated dispatch below
        // stays in its EXACT original position — never reorder across gates.
        let sk = DecodeSkips::active(nt);

        if !sk.small { cb.rms_norm(&hidden, Some(&norm_attn_w), &bn, ne, nt, eps); }

        // Fused QKV projection (nt==1 decode): one matmul on the concatenated
        // Wq/Wk/Wv produces q+k+v in one buffer; the rope/store read the q/k/v
        // sections via buffer offsets. The metal crate's set_buffer offset is a
        // fixed byte offset, which is only valid for a single token (per-token
        // sections are contiguous only when nt==1), so the fused path is
        // gated to nt==1. Prefill keeps three separate matmuls (GEMM for Q4_0).
        let use_fused_qkv = nt == 1 && buf_qkv.is_some()
            && wq.ttype == wk.ttype && wk.ttype == wv.ttype
            && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1");
        // KV-parallel attention chunk count (decode): adaptive to the current KV
        // length (max_pos+1) so long contexts get more parallelism; about 16 KV
        // rows per chunk, capped at 32. MINFER_ATTN_CHUNKS overrides for tuning.
        let split_chunks = std::env::var("MINFER_ATTN_CHUNKS").ok()
            .and_then(|v| v.parse::<usize>().ok()).filter(|&c| c >= 1)
            .unwrap_or_else(|| ((max_pos + 1 + 15) / 16).clamp(1, 32));
        if use_fused_qkv {
            let buf_qkv = buf_qkv.as_ref().unwrap();
            // GPU safety: the concat buffer must exactly match the fused output
            // rows, else the matmul reads out of bounds. Verify the byte length
            // against the expected row layout and error-exit (never fall back).
            let od_total = nqt + nkt + nkt;
            let row = (ne / quant_block_q(wq.ttype)) * quant_block_bytes(wq.ttype);
            let expect = (od_total * row) as u64;
            if buf_qkv.length() != expect {
                gpu_abort(&format!(
                    "attn_qkv buffer length {} B != expected {expect} B (fused QKV rows={od_total}, row={row})",
                    buf_qkv.length()
                ));
            }
            if bqkv_buf.length() < (od_total * 4) as u64 {
                gpu_abort(&format!(
                    "bqkv buffer length {} B < {} B (fused QKV needs nt*od_total*4)",
                    bqkv_buf.length(), (od_total * 4) as u64
                ));
            }
            if !sk.matmul { cb.matmul_on_gpu_buf(&buf_qkv, wq.ttype, &q8_bn, &bn, &bqkv_buf, od_total, ne, nt); }
            // Fused bias+RoPE+KV-store (nt==1 only): replaces the 7 separate
            // bias/rope/store dispatches below. Requires all three biases + even
            // hd (rope pair mapping); MINFER_NO_FUSE_BSR=1 disables it for A/B.
            let use_fused_bsr = nt == 1 && hd % 2 == 0
                && bq_bias.is_some() && bk_bias.is_some() && bv_bias.is_some()
                && !std::env::var("MINFER_NO_FUSE_BSR").map_or(false, |v| v == "1");
            if !sk.small {
                if use_fused_bsr {
                    cb.attn_bias_rope_store(&bqkv_buf,
                        bq_bias.as_ref().unwrap(), bk_bias.as_ref().unwrap(), bv_bias.as_ref().unwrap(),
                        &kv_k[il], &kv_v[il],
                        nqt, nkt, hd, freq_base, freq_scale, positions[0] as i32, rope_style);
                } else {
                    if let Some(bb) = &bq_bias { cb.add_bias_f32(&bqkv_buf, bb, nqt, nt, 0); }
                    if let Some(bb) = &bk_bias { cb.add_bias_f32(&bqkv_buf, bb, nkt, nt, nqt); }
                    if let Some(bb) = &bv_bias { cb.add_bias_f32(&bqkv_buf, bb, nkt, nt, nqt + nkt); }
                    cb.rope_f32(&bqkv_buf, nh, hd, nt, freq_base, freq_scale, &pos_buf, rope_style, 0);
                    cb.rope_f32(&bqkv_buf, nk, hd, nt, freq_base, freq_scale, &pos_buf, rope_style, nqt);
                    cb.store_kv(&bqkv_buf, &kv_k[il], nkt, nt, &pos_buf, nqt);
                    cb.store_kv(&bqkv_buf, &kv_v[il], nkt, nt, &pos_buf, nqt + nkt);
                }
            }
            if !sk.attn {
                if nt == 1
                    && !std::env::var("MINFER_NO_SPLIT_ATTN").map_or(false, |v| v == "1") {
                    cb.gqa_attn_split_f32(&bqkv_buf, &kv_k[il], &kv_v[il], &ba_buf, &pos_buf, nh, nk, hd, attn_scale, nt, split_chunks);
                } else {
                    cb.gqa_attn_f32(&bqkv_buf, &kv_k[il], &kv_v[il], &ba_buf, &pos_buf, nh, nk, hd, attn_scale, nt);
                }
            }
        } else {
            cb.matmul_on_gpu_buf(&buf_wq, wq.ttype, &q8_bn, &bn, &bq_buf, nqt, ne, nt);
            cb.matmul_on_gpu_buf(&buf_wk, wk.ttype, &q8_bn, &bn, &bk_buf, nkt, ne, nt);
            cb.matmul_on_gpu_buf(&buf_wv, wv.ttype, &q8_bn, &bn, &bv_buf, nkt, ne, nt);
            if !sk.small {
                if let Some(bb) = &bq_bias { cb.add_bias_f32(&bq_buf, bb, nqt, nt, 0); }
                if let Some(bb) = &bk_bias { cb.add_bias_f32(&bk_buf, bb, nkt, nt, 0); }
                if let Some(bb) = &bv_bias { cb.add_bias_f32(&bv_buf, bb, nkt, nt, 0); }
                cb.rope_f32(&bq_buf, nh, hd, nt, freq_base, freq_scale, &pos_buf, rope_style, 0);
                cb.rope_f32(&bk_buf, nk, hd, nt, freq_base, freq_scale, &pos_buf, rope_style, 0);
                cb.store_kv(&bk_buf, &kv_k[il], nkt, nt, &pos_buf, 0);
                cb.store_kv(&bv_buf, &kv_v[il], nkt, nt, &pos_buf, 0);
            }
            if !sk.attn {
                if nt == 1
                    && !std::env::var("MINFER_NO_SPLIT_ATTN").map_or(false, |v| v == "1") {
                    cb.gqa_attn_split_f32(&bq_buf, &kv_k[il], &kv_v[il], &ba_buf, &pos_buf, nh, nk, hd, attn_scale, nt, split_chunks);
                } else {
                    cb.gqa_attn_f32(&bq_buf, &kv_k[il], &kv_v[il], &ba_buf, &pos_buf, nh, nk, hd, attn_scale, nt);
                }
            }
        }
        // all weight types read f32 activations (Q4_0 included); no Q8_0 quantize pass.
        if !sk.matmul { cb.matmul_on_gpu_buf(&buf_wo, wo.ttype, &q8_ba, &ba_buf, &bn, ne, ne, nt); }
        if !sk.small { cb.add_f32(&hidden, &bn, &hidden, nt * ne); }

        // FFN branch
        let ffn_all_q4 = ffn_gate.ttype == TensorType::Q4_0 && ffn_up.ttype == TensorType::Q4_0;
        let ffn_any_q4k = [ffn_gate.ttype, ffn_up.ttype, ffn_down.ttype]
            .iter().any(|t| *t == TensorType::Q4_K || *t == TensorType::Q6_K);
        if !ffn_all_q4 && ffn_any_q4k {
            if ffn_gate.ttype == TensorType::Q4_0 || ffn_up.ttype == TensorType::Q4_0
                { return false; }
        }

        if !sk.small { cb.rms_norm(&hidden, Some(&norm_ffn_w), &ba_buf, ne, nt, eps); }
        // Fused FFN gate+up (nt==1 decode): one matmul on the concatenated
        // gate+up weight → bgu; swiglu reads gate at offset 0 and up at nf.
        // Same nt==1 gate + exact-buffer-length guard as the QKV fusion.
        let use_fused_gu = nt == 1 && buf_gu.is_some()
            && ffn_gate.ttype == ffn_up.ttype
            && !std::env::var("MINFER_NO_FUSE_QKV").map_or(false, |v| v == "1");
        if use_fused_gu {
            let buf_gu = buf_gu.as_ref().unwrap();
            let od_total = nf + nf;
            let row = (ne / quant_block_q(ffn_gate.ttype)) * quant_block_bytes(ffn_gate.ttype);
            let expect = (od_total * row) as u64;
            if buf_gu.length() != expect {
                gpu_abort(&format!(
                    "ffn_gu buffer length {} B != expected {expect} B (fused gate+up rows={od_total}, row={row})",
                    buf_gu.length()
                ));
            }
            if bgu_buf.length() < (od_total * 4) as u64 {
                gpu_abort(&format!(
                    "bgu buffer length {} B < {} B (fused gate+up needs nt*od_total*4)",
                    bgu_buf.length(), (od_total * 4) as u64
                ));
            }
            if !sk.matmul {
                cb.matmul_on_gpu_buf(&buf_gu, ffn_gate.ttype, &q8_ba, &ba_buf, &bgu_buf, od_total, ne, nt);
            }
            if !sk.small { cb.swiglu_f32_off(&bgu_buf, &bgu_buf, &bgu_buf, nt * nf, nt * nf); }
            if !sk.matmul {
                cb.matmul_on_gpu_buf(&buf_fd, ffn_down.ttype, &q8_ba, &bgu_buf, &bn, ne, nf, nt);
            }
        } else {
            if !sk.matmul {
                cb.matmul_on_gpu_buf(&buf_fg, ffn_gate.ttype, &q8_ba, &ba_buf, &bg_buf, nf, ne, nt);
                cb.matmul_on_gpu_buf(&buf_fu, ffn_up.ttype, &q8_ba, &ba_buf, &bf_buf, nf, ne, nt);
            }
            if !sk.small { cb.swiglu_f32(&bg_buf, &bf_buf, &bg_buf, nt * nf); }
            if !sk.matmul {
                cb.matmul_on_gpu_buf(&buf_fd, ffn_down.ttype, &q8_ba, &bg_buf, &bn, ne, nf, nt);
            }
        }
        if !sk.small { cb.add_f32(&hidden, &bn, &hidden, nt * ne); }

        self.inner.kv_size.write().unwrap()[il] = max_pos + 1;
        true
    }

    /// Final RMSNorm + output matmul on GPU. Returns false if GPU unavailable.
    /// Call download_logits() after cb.submit() to retrieve results.
    pub fn output_norm_gpu(
        &self,
        cb: &MpsCommandBuffer,
        output: &Tensor,
        output_norm: Option<&Tensor>,
        output_b: Option<&Tensor>,
        ne: usize, nv: usize, nt: usize, eps: f32,
    ) -> bool {
        // The output matmul quantized kernels require id (ne) % 32 == 0.
        if ne % 32 != 0 {
            gpu_abort(&format!(
                "output matmul input dim ne={ne} is not 32-aligned (quantized matmul would read out of bounds)"
            ));
        }
        let weights = self.inner.weights.lock().unwrap();
        let norm_w = match output_norm {
            Some(t) => match weights.get(&t.name) {
                Some(w) => w.clone(),
                None => return false,
            },
            None => return false,
        };
        let buf_output = match weights.get(&output.name) {
            Some(b) => b.clone(),
            None => return false,
        };
        let bias_buf = output_b.and_then(|ob| weights.get(&ob.name).cloned());
        drop(weights);
        if output.ttype == TensorType::Q5_K || output.ttype == TensorType::Raw {
            return false;
        }
        if output.ttype != TensorType::Q4_0 && output.ttype != TensorType::Q4_1
            && output.ttype != TensorType::Q8_0
            && output.ttype != TensorType::Q4_K && output.ttype != TensorType::Q6_K {
            return false;
        }

        let dev = &self.inner.device;
        let hidden = Self::get_or_grow(&self.inner.buf_hidden, (nt * ne * 4) as u64, dev);
        let bn = Self::get_or_grow(&self.inner.buf_bn, (nt * ne * 4) as u64, dev);
        let logits = Self::get_or_grow(&self.inner.buf_logits, (nt * nv * 4) as u64, dev);

        let sk = DecodeSkips::active(nt);

        if !sk.small { cb.rms_norm(&hidden, Some(&norm_w), &bn, ne, nt, eps); }

        // all output weight types read f32 activations (Q4_0 included); no Q8_0 quantize pass.
        if !sk.matmul { cb.quant_matmul_f32_on_gpu_buf(&buf_output, output.ttype, &bn, &logits, nv, ne, nt); }
        if !sk.small {
            if let Some(bb) = &bias_buf {
                cb.add_bias_f32(&logits, bb, nv, nt, 0);
            }
        }
        true
    }

    /// Download logits from GPU after command buffer submission.
    pub fn download_logits(&self, logits: &mut [f32]) {
        let buf = self.inner.buf_logits.lock().unwrap();
        Self::copy_from_gpu(&buf, logits);
    }

    /// Sync GPU KV cache to CPU KVCache for up to `num_layers` layers.
    /// Ensures CPU fallback has consistent KV state after GPU forward passes.
    /// The GPU cache is F16 (opt-in) or F32; the CPU cache is F32.
    pub fn sync_kv_to_cpu(&self, kv_cache: &mut crate::cache::KVCache, num_layers: usize) {
        let kv_k = self.inner.kv_k.read().unwrap();
        let kv_v = self.inner.kv_v.read().unwrap();
        let kv_size = self.inner.kv_size.read().unwrap();
        let f16 = kv_cache_is_f16();
        for il in 0..num_layers.min(kv_k.len()) {
            let sz = kv_size[il];
            if sz == 0 || il >= kv_cache.layers.len() {
                continue;
            }
            let layer = &mut kv_cache.layers[il];
            let n = sz * layer.dim;
            let elem = if f16 { 2u64 } else { 4u64 };
            if kv_k[il].length() >= (n as u64) * elem && layer.k.len() >= n {
                if f16 {
                    Self::copy_from_gpu_half_to_f32(&kv_k[il], &mut layer.k[..n], n);
                    Self::copy_from_gpu_half_to_f32(&kv_v[il], &mut layer.v[..n], n);
                } else {
                    Self::copy_from_gpu(&kv_k[il], &mut layer.k[..n]);
                    Self::copy_from_gpu(&kv_v[il], &mut layer.v[..n]);
                }
                layer.size = sz;
            }
        }
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
        MpsState::init();
        assert!(
            MpsState::get().is_some(),
            "MPS unavailable — Metal shader compilation failed (check src/metal.metal for \
             duplicate/missing kernel definitions); the model would run on CPU"
        );
    }
}
