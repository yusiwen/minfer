// MPS (Metal) backend for Apple Silicon.
// Translated from: llama.cpp/ggml/src/ggml-metal
//
// Provides MpsCommandBuffer for batching all layer ops into one GPU submission.

use std::sync::OnceLock;
use crate::tensor::{Tensor, TensorType};
use crate::block::{Q4B, Q4KB, Q8B};

static MPS: OnceLock<Option<MpsState>> = OnceLock::new();

pub struct MpsState {
    #[cfg(target_os = "macos")]
    inner: MpsStateInner,
}

#[cfg(target_os = "macos")]
struct MpsStateInner {
    device: metal::Device,
    queue: metal::CommandQueue,
    pl_q4_0_q8: metal::ComputePipelineState,
    pl_q4_0_f32: metal::ComputePipelineState,
    pl_q4_1_f32: metal::ComputePipelineState,
    pl_q8_0_f32: metal::ComputePipelineState,
    pl_q4_k_f32: metal::ComputePipelineState,
    pl_q6_k_f32: metal::ComputePipelineState,
    pl_quantize_q8_0: metal::ComputePipelineState,
    pl_rms_norm: metal::ComputePipelineState,
    pl_add: metal::ComputePipelineState,
    pl_add_bias: metal::ComputePipelineState,
    pl_mul: metal::ComputePipelineState,
    pl_silu: metal::ComputePipelineState,
    pl_rope: metal::ComputePipelineState,
    pl_gqa_attn: metal::ComputePipelineState,
    pl_store_kv: metal::ComputePipelineState,
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
    buf_ba: std::sync::Mutex<metal::Buffer>,
    buf_bf: std::sync::Mutex<metal::Buffer>,
    buf_bg: std::sync::Mutex<metal::Buffer>,
    buf_q8_bn: std::sync::Mutex<metal::Buffer>,
    buf_q8_ba: std::sync::Mutex<metal::Buffer>,
    buf_positions: std::sync::Mutex<metal::Buffer>,
    buf_logits: std::sync::Mutex<metal::Buffer>,
    // Persistent per-layer GPU KV cache (k, v) and current size in KV positions.
    kv_k: std::sync::Mutex<Vec<metal::Buffer>>,
    kv_v: std::sync::Mutex<Vec<metal::Buffer>>,
    kv_size: std::sync::Mutex<Vec<usize>>,
}

// ─── MpsCommandBuffer: batch multiple ops in one GPU submission ──────

#[cfg(target_os = "macos")]
pub struct MpsCommandBuffer<'a> {
    state: &'a MpsStateInner,
    cmd_buf: &'a metal::CommandBufferRef,
    enc: &'a metal::ComputeCommandEncoderRef,
}

#[cfg(target_os = "macos")]
impl Drop for MpsCommandBuffer<'_> {
    fn drop(&mut self) {}
}

#[cfg(target_os = "macos")]
impl MpsCommandBuffer<'_> {
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
        assert!(w.ttype == TensorType::Q4_0 || w.ttype == TensorType::Q4_1 || w.ttype == TensorType::Q4_K || w.ttype == TensorType::Q6_K || w.ttype == TensorType::Q8_0,
            "quant_matmul_f32_on_gpu: unsupported weight type {:?}", w.ttype);
        let weights = self.state.weights.lock().unwrap();
        let wb = weights.get(&w.name).expect("weight not on GPU");

        match w.ttype {
            TensorType::Q8_0 => {
                self.enc.set_compute_pipeline_state(&self.state.pl_q8_0_f32);
                self.enc.set_buffer(0, Some(wb), 0);
                self.enc.set_buffer(1, Some(x), 0);
                self.enc.set_buffer(2, Some(out), 0);
                self.set_params(3, &(od as i32));
                self.set_params(4, &(id as i32));
                self.set_params(5, &(nt as i32));
                const NW: u64 = 32;
                const NSG: u64 = 4;
                const NR0: u64 = 2;
                const TG_MEM: u64 = NW * NR0 * std::mem::size_of::<f32>() as u64; // 256 bytes
                self.enc.set_threadgroup_memory_length(0, TG_MEM);
                self.dispatch_2d(((od + 1) / 2) as u64, nt as u64, NW, NSG);
            }
            TensorType::Q4_K | TensorType::Q6_K => {
                let pl = if w.ttype == TensorType::Q4_K { &self.state.pl_q4_k_f32 } else { &self.state.pl_q6_k_f32 };
                self.enc.set_compute_pipeline_state(pl);
                self.enc.set_buffer(0, Some(wb), 0);
                self.enc.set_buffer(1, Some(x), 0);
                self.enc.set_buffer(2, Some(out), 0);
                self.set_params(3, &(od as i32));
                self.set_params(4, &(id as i32));
                self.set_params(5, &(nt as i32));
                self.dispatch_2d(((od + 3) / 4) as u64, nt as u64, 64, 1);
            }
            TensorType::Q4_1 => {
                self.enc.set_compute_pipeline_state(&self.state.pl_q4_1_f32);
                self.enc.set_buffer(0, Some(wb), 0);
                self.enc.set_buffer(1, Some(x), 0);
                self.enc.set_buffer(2, Some(out), 0);
                self.set_params(3, &(od as i32));
                self.set_params(4, &(id as i32));
                self.set_params(5, &(nt as i32));
                self.dispatch_2d(((od + 7) / 8) as u64, nt as u64, 64, 1);
            }
            _ => {
                self.enc.set_compute_pipeline_state(&self.state.pl_q4_0_f32);
                self.enc.set_buffer(0, Some(wb), 0);
                self.enc.set_buffer(1, Some(x), 0);
                self.enc.set_buffer(2, Some(out), 0);
                self.set_params(3, &(od as i32));
                self.set_params(4, &(id as i32));
                self.set_params(5, &(nt as i32));
                self.dispatch_2d(((od + 7) / 8) as u64, nt as u64, 64, 1);
            }
        }
    }

    /// Dispatch Q4_0 × Q8_0 matmul (bit-exact with CPU path).
    pub fn quant_matmul_q8(&self, w: &Tensor, x: &metal::Buffer, out: &metal::Buffer,
        od: usize, id: usize, nt: usize,
    ) {
        let weights = self.state.weights.lock().unwrap();
        let wb = weights.get(&w.name).expect("weight not on GPU");
        self.enc.set_compute_pipeline_state(&self.state.pl_q4_0_q8);
        self.enc.set_buffer(0, Some(wb), 0);
        self.enc.set_buffer(1, Some(x), 0);
        self.enc.set_buffer(2, Some(out), 0);
        self.set_params(3, &(od as i32));
        self.set_params(4, &(id as i32));
        self.set_params(5, &(nt as i32));
        // Grid: x = ceil(od / 8) row-groups, y = tokens; TG = 64 threads.
        self.dispatch_2d((od as u64 + 7) / 8, nt as u64, 64, 1);
    }

    /// Quantize f32 activations to Q8_0 on the GPU.
    /// Input layout: [nt][dim]; output layout: [nt][nb][Q8B].
    pub fn quantize_q8_0(&self, x: &metal::Buffer, y: &metal::Buffer, dim: usize, nt: usize) {
        self.enc.set_compute_pipeline_state(&self.state.pl_quantize_q8_0);
        self.enc.set_buffer(0, Some(x), 0);
        self.enc.set_buffer(1, Some(y), 0);
        self.set_params(2, &(dim as i32));
        self.set_params(3, &(nt as i32));
        let nb = dim / 32;
        self.dispatch_1d((nt * nb) as u64, 256);
    }

    /// Choose Q4_0×Q8_0 matmul when weight is Q4_0, otherwise fall back to f32-activation matmul.
    fn matmul_on_gpu(&self, w: &Tensor, q8_x: &metal::Buffer, f32_x: &metal::Buffer, out: &metal::Buffer,
        od: usize, id: usize, nt: usize,
    ) {
        if w.ttype == TensorType::Q4_0 {
            self.quant_matmul_q8(w, q8_x, out, od, id, nt);
        } else {
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        }
    }

    /// RMSNorm: y = x * rsqrt(mean(x²)+eps) * w
    pub fn rms_norm(&self, x: &metal::Buffer, w: Option<&metal::Buffer>, y: &metal::Buffer,
        d: usize, n: usize, eps: f32,
    ) {
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
        self.enc.set_compute_pipeline_state(&self.state.pl_add);
        self.enc.set_buffer(0, Some(x), 0);
        self.enc.set_buffer(1, Some(y), 0);
        self.enc.set_buffer(2, Some(z), 0);
        self.set_params(3, &(n as i32));
        self.dispatch_1d(n as u64, 256);
    }

    /// Add 1-D bias to rows: y[t][i] += b[i]
    pub fn add_bias_f32(&self, y: &metal::Buffer, b: &metal::Buffer, d: usize, n: usize) {
        self.enc.set_compute_pipeline_state(&self.state.pl_add_bias);
        self.enc.set_buffer(0, Some(y), 0);
        self.enc.set_buffer(1, Some(b), 0);
        self.set_params(2, &(d as i32));
        self.dispatch_2d(n as u64, d as u64, 1, 64);
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

    /// RoPE (in-place): x layout [nt][n_head][n_dims].
    pub fn rope_f32(&self, x: &metal::Buffer, n_head: usize, n_dims: usize, nt: usize,
        freq_base: f32, freq_scale: f32, positions: &metal::Buffer,
    ) {
        self.enc.set_compute_pipeline_state(&self.state.pl_rope);
        self.enc.set_buffer(0, Some(x), 0);
        self.set_params(1, &(n_head as i32));
        self.set_params(2, &(n_dims as i32));
        self.set_params(3, &(nt as i32));
        self.set_params(4, &(freq_base.to_bits() as i32));
        self.set_params(5, &(freq_scale.to_bits() as i32));
        self.enc.set_buffer(6, Some(positions), 0);
        self.dispatch_2d(nt as u64, n_head as u64, 1, 1);
    }

    /// GQA attention: q/k/v/o layout [nt][nh][hd]; k/v stored as [nkv][nk][hd].
    /// Per-token KV length is positions[t] + 1 (causal mask).
    pub fn gqa_attn_f32(&self, q: &metal::Buffer, k: &metal::Buffer, v: &metal::Buffer,
        o: &metal::Buffer, positions: &metal::Buffer, nh: usize, nk: usize, hd: usize, scale: f32, nt: usize,
    ) {
        self.enc.set_compute_pipeline_state(&self.state.pl_gqa_attn);
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
        self.dispatch_2d(nt as u64, nh as u64, 32, 1);
    }

    /// Scatter nt rows of src[nt][nkt] into dst[positions[t]][nkt].
    pub fn store_kv_f32(&self, src: &metal::Buffer, dst: &metal::Buffer, nkt: usize, nt: usize,
        positions: &metal::Buffer,
    ) {
        self.enc.set_compute_pipeline_state(&self.state.pl_store_kv);
        self.enc.set_buffer(0, Some(src), 0);
        self.enc.set_buffer(1, Some(dst), 0);
        self.set_params(2, &(nkt as i32));
        self.set_params(3, &(nt as i32));
        self.enc.set_buffer(4, Some(positions), 0);
        self.dispatch_2d(nt as u64, nkt as u64, 1, 1);
    }

    /// Commit GPU work and wait for completion using a semaphore completion handler.
    /// This avoids the ~20ms Metal scheduler wakeup overhead of wait_until_completed.
    pub fn submit(self) {
        self.enc.end_encoding();

        // dispatch_semaphore_t is already a reference-counted opaque pointer.
        // We create one with value 0, signal from the completion handler, and wait here.
        let sem = unsafe { dispatch_semaphore_create(0) };

        // We need to pass `sem` into the block. Capture it as a usize to satisfy
        // the block crate's Sync requirement.
        let sem_val = sem as usize;

        use block::ConcreteBlock;
        let blk = ConcreteBlock::new(move |_buf: &metal::CommandBufferRef| {
            // SAFETY: dispatch_semaphore_signal is thread-safe.
            unsafe { dispatch_semaphore_signal(sem_val as *mut std::ffi::c_void); }
        });
        let blk = blk.copy();
        self.cmd_buf.add_completed_handler(&blk);
        self.cmd_buf.commit();

        // DISPATCH_TIME_FOREVER = ~0u64 — wait indefinitely for GPU completion.
        unsafe { dispatch_semaphore_wait(sem, !0u64); }
        unsafe { dispatch_release(sem); }
    }
}

#[cfg(target_os = "macos")]
extern "C" {
    fn dispatch_semaphore_create(value: isize) -> *mut std::ffi::c_void;
    fn dispatch_semaphore_signal(sem: *mut std::ffi::c_void) -> isize;
    fn dispatch_semaphore_wait(sem: *mut std::ffi::c_void, timeout: u64) -> isize;
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
            let pl_q4_0_f32 = get_pl("kernel_q4_0_f32_matmul")?;
            let pl_q4_1_f32 = get_pl("kernel_q4_1_f32_matmul")?;
            let pl_q8_0_f32 = get_pl("kernel_q8_0_f32_matmul")?;
            let pl_q4_k_f32 = get_pl("kernel_q4_k_f32_matmul")?;
            let pl_q6_k_f32 = get_pl("kernel_q6_k_f32_matmul")?;
            let pl_quantize_q8_0 = get_pl("kernel_quantize_q8_0")?;
            let pl_rms_norm = get_pl("kernel_rms_norm_f32")?;
            let pl_add      = get_pl("kernel_add_f32")?;
            let pl_add_bias = get_pl("kernel_add_bias_f32")?;
            let pl_mul      = get_pl("kernel_mul_f32")?;
            let pl_silu     = get_pl("kernel_silu_f32")?;
            let pl_rope     = get_pl("kernel_rope_f32")?;
            let pl_gqa_attn = get_pl("kernel_gqa_attn_f32")?;
            let pl_store_kv = get_pl("kernel_store_kv_f32")?;
            let dummy_buf = device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared);
            let m = MpsStateInner {
                device: device.clone(),
                queue: device.new_command_queue(),
                pl_q4_0_q8,
                pl_q4_0_f32,
                pl_q4_1_f32,
                pl_q8_0_f32,
                pl_q4_k_f32,
                pl_q6_k_f32,
                pl_quantize_q8_0,
                pl_rms_norm,
                pl_add,
                pl_add_bias,
                pl_mul,
                pl_silu,
                pl_rope,
                pl_gqa_attn,
                pl_store_kv,
                weights: std::sync::Mutex::new(std::collections::HashMap::new()),
                q8_buf: std::sync::Mutex::new(dummy_buf.clone()),
                out_buf: std::sync::Mutex::new(dummy_buf.clone()),
                out_pool: std::sync::Mutex::new(Vec::new()),
                buf_hidden: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bn: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bq: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bk: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bv: std::sync::Mutex::new(dummy_buf.clone()),
                buf_ba: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bf: std::sync::Mutex::new(dummy_buf.clone()),
                buf_bg: std::sync::Mutex::new(dummy_buf.clone()),
                buf_q8_bn: std::sync::Mutex::new(dummy_buf.clone()),
                buf_q8_ba: std::sync::Mutex::new(dummy_buf.clone()),
                buf_positions: std::sync::Mutex::new(dummy_buf.clone()),
                buf_logits: std::sync::Mutex::new(dummy_buf.clone()),
                kv_k: std::sync::Mutex::new(Vec::new()),
                kv_v: std::sync::Mutex::new(Vec::new()),
                kv_size: std::sync::Mutex::new(Vec::new()),
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
            let cmd_buf = self.inner.queue.new_command_buffer();
            let enc = cmd_buf.new_compute_command_encoder();
            MpsCommandBuffer { state: &self.inner, cmd_buf, enc }
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
        cb.submit();

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
        cb.submit();
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

    /// Ensure the GPU KV cache for layer `il` can hold at least `max_nkv` rows.
    fn kv_ensure_layer(&self, il: usize, max_nkv: usize, nkt: usize) {
        let need = (max_nkv * nkt * 4) as u64;
        {
            let mut kvec = self.inner.kv_k.lock().unwrap();
            let mut vvec = self.inner.kv_v.lock().unwrap();
            let mut szvec = self.inner.kv_size.lock().unwrap();
            while kvec.len() <= il {
                kvec.push(self.inner.device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared));
                vvec.push(self.inner.device.new_buffer(1, metal::MTLResourceOptions::StorageModeShared));
                szvec.push(0);
            }
            if kvec[il].length() < need {
                // Preserve existing KV data when growing the buffer.
                let old_k = kvec[il].clone();
                let old_v = vvec[il].clone();
                let old_len = old_k.length().min(old_v.length());
                kvec[il] = self.inner.device.new_buffer(need, metal::MTLResourceOptions::StorageModeShared);
                vvec[il] = self.inner.device.new_buffer(need, metal::MTLResourceOptions::StorageModeShared);
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
        eps: f32, freq_base: f32, freq_scale: f32,
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
        if !self.has_weight(&wq.name) || !self.has_weight(&wk.name) || !self.has_weight(&wv.name)
            || !self.has_weight(&wo.name) || !self.has_weight(&ffn_gate.name)
            || !self.has_weight(&ffn_up.name) || !self.has_weight(&ffn_down.name) {
            return false;
        }
        let norm_attn_w = match self.get_weight(&attn_norm.name) { Some(b) => b, None => return false };
        let norm_ffn_w  = match self.get_weight(&ffn_norm.name)  { Some(b) => b, None => return false };
        // Validate bias weights before encoding
        let bq_bias = l.bq.as_ref().map(|b| self.get_weight(&b.name).ok_or(())).transpose().ok().flatten();
        let bk_bias = l.bk.as_ref().map(|b| self.get_weight(&b.name).ok_or(())).transpose().ok().flatten();
        let bv_bias = l.bv.as_ref().map(|b| self.get_weight(&b.name).ok_or(())).transpose().ok().flatten();
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
        let ba_buf = Self::get_or_grow(&self.inner.buf_ba, ba_len, dev);
        let bf_buf = Self::get_or_grow(&self.inner.buf_bf, bf_len, dev);
        let bg_buf = Self::get_or_grow(&self.inner.buf_bg, bg_len, dev);
        let q8_bn = Self::get_or_grow(&self.inner.buf_q8_bn, q8_bn_len, dev);
        let q8_ba = Self::get_or_grow(&self.inner.buf_q8_ba, q8_ba_len, dev);
        let pos_buf = self.inner.buf_positions.lock().unwrap();
        let kv_k = self.inner.kv_k.lock().unwrap();
        let kv_v = self.inner.kv_v.lock().unwrap();

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

        cb.rms_norm(&hidden, Some(&norm_attn_w), &bn, ne, nt, eps);
        if attn_all_q4 && !attn_any_q4k {
            cb.quantize_q8_0(&bn, &q8_bn, ne, nt);
        }
        cb.matmul_on_gpu(wq, &q8_bn, &bn, &bq_buf, nqt, ne, nt);
        if let Some(bb) = &bq_bias { cb.add_bias_f32(&bq_buf, bb, nqt, nt); }
        cb.matmul_on_gpu(wk, &q8_bn, &bn, &bk_buf, nkt, ne, nt);
        if let Some(bb) = &bk_bias { cb.add_bias_f32(&bk_buf, bb, nkt, nt); }
        cb.matmul_on_gpu(wv, &q8_bn, &bn, &bv_buf, nkt, ne, nt);
        if let Some(bb) = &bv_bias { cb.add_bias_f32(&bv_buf, bb, nkt, nt); }
        cb.rope_f32(&bq_buf, nh, hd, nt, freq_base, freq_scale, &pos_buf);
        cb.rope_f32(&bk_buf, nk, hd, nt, freq_base, freq_scale, &pos_buf);
        cb.store_kv_f32(&bk_buf, &kv_k[il], nkt, nt, &pos_buf);
        cb.store_kv_f32(&bv_buf, &kv_v[il], nkt, nt, &pos_buf);
        let scale = 1.0 / (hd as f32).sqrt();
        cb.gqa_attn_f32(&bq_buf, &kv_k[il], &kv_v[il], &ba_buf, &pos_buf, nh, nk, hd, scale, nt);
        // wo: Q4_0 needs Q8_0 quantized input; Q4_K reads f32 from ba_buf directly.
        if wo.ttype == TensorType::Q4_0 {
            cb.quantize_q8_0(&ba_buf, &q8_ba, ne, nt);
        }
        cb.matmul_on_gpu(wo, &q8_ba, &ba_buf, &bn, ne, ne, nt);
        cb.add_f32(&hidden, &bn, &hidden, nt * ne);

        // FFN branch
        let ffn_all_q4 = ffn_gate.ttype == TensorType::Q4_0 && ffn_up.ttype == TensorType::Q4_0;
        let ffn_any_q4k = [ffn_gate.ttype, ffn_up.ttype, ffn_down.ttype]
            .iter().any(|t| *t == TensorType::Q4_K || *t == TensorType::Q6_K);
        if !ffn_all_q4 && ffn_any_q4k {
            if ffn_gate.ttype == TensorType::Q4_0 || ffn_up.ttype == TensorType::Q4_0
                { return false; }
        }

        cb.rms_norm(&hidden, Some(&norm_ffn_w), &ba_buf, ne, nt, eps);
        if ffn_all_q4 && !ffn_any_q4k {
            cb.quantize_q8_0(&ba_buf, &q8_ba, ne, nt);
        }
        cb.matmul_on_gpu(ffn_gate, &q8_ba, &ba_buf, &bg_buf, nf, ne, nt);
        cb.matmul_on_gpu(ffn_up, &q8_ba, &ba_buf, &bf_buf, nf, ne, nt);
        cb.silu_f32(&bg_buf, nt * nf);
        cb.mul_f32(&bg_buf, &bf_buf, &bg_buf, nt * nf);
        if ffn_down.ttype == TensorType::Q4_0 {
            cb.quantize_q8_0(&bg_buf, &q8_ba, nf, nt);
        }
        cb.matmul_on_gpu(ffn_down, &q8_ba, &bg_buf, &bn, ne, nf, nt);
        cb.add_f32(&hidden, &bn, &hidden, nt * ne);

        self.inner.kv_size.lock().unwrap()[il] = max_pos + 1;
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
        let norm_w = match output_norm {
            Some(t) => match self.get_weight(&t.name) {
                Some(w) => w,
                None => return false,
            },
            None => return false,
        };
        if !self.has_weight(&output.name) {
            return false;
        }
        if output.ttype != TensorType::Q4_0 && output.ttype != TensorType::Q4_1
            && output.ttype != TensorType::Q8_0 {
            return false;
        }

        let dev = &self.inner.device;
        let hidden = Self::get_or_grow(&self.inner.buf_hidden, (nt * ne * 4) as u64, dev);
        let bn = Self::get_or_grow(&self.inner.buf_bn, (nt * ne * 4) as u64, dev);
        let logits = Self::get_or_grow(&self.inner.buf_logits, (nt * nv * 4) as u64, dev);

        cb.rms_norm(&hidden, Some(&norm_w), &bn, ne, nt, eps);

        if output.ttype == TensorType::Q4_0 {
            let q8_len = (nt * (ne / 32) * Q8B) as u64;
            let q8_bn = Self::get_or_grow(&self.inner.buf_q8_bn, q8_len, dev);
            cb.quantize_q8_0(&bn, &q8_bn, ne, nt);
            cb.quant_matmul_q8(output, &q8_bn, &logits, nv, ne, nt);
        } else {
            cb.quant_matmul_f32_on_gpu(output, &bn, &logits, nv, ne, nt);
        }
        if let Some(ob) = output_b {
            if let Some(bias_buf) = self.get_weight(&ob.name) {
                cb.add_bias_f32(&logits, &bias_buf, nv, nt);
            }
        }
        true
    }

    /// Download logits from GPU after command buffer submission.
    pub fn download_logits(&self, logits: &mut [f32]) {
        let buf = self.inner.buf_logits.lock().unwrap();
        Self::copy_from_gpu(&buf, logits);
    }
}
