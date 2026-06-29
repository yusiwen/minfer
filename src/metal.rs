// MPS (Metal Performance Shaders) backend for Apple Silicon.
// Translated from: llama.cpp/ggml/src/ggml-metal/ggml-metal-device.m
//
// Runtime auto-detection: MTLCreateSystemDefaultDevice() → None → CPU fallback.
// Shared memory (MTLResourceStorageModeShared) on Apple Silicon for zero-copy access.
//
// This module is compiled for all targets, but the #[cfg(target_os = "macos")]
// guard ensures the metal crate is only linked on Apple platforms.

use std::sync::OnceLock;
use crate::tensor::{Tensor, TensorType};

static MPS: OnceLock<Option<MpsState>> = OnceLock::new();

pub struct MpsState {
    #[cfg(target_os = "macos")]
    inner: MpsStateInner,
}

#[cfg(target_os = "macos")]
struct MpsStateInner {
    device: metal::Device,
    queue: metal::CommandQueue,
    pipeline_q4_0: metal::ComputePipelineState,
    // weight name → metal::Buffer (inserted at load time, read-only during inference)
    weights: std::sync::Mutex<std::collections::HashMap<String, metal::Buffer>>,
}

// MPS is only constructable on macOS; on other platforms MpsState::get() returns None.
impl MpsState {
    /// Try to initialize the MPS backend.
    /// Returns None if Metal is unavailable (non-Apple, no GPU, or device too old).
    #[cfg(target_os = "macos")]
    pub fn try_new() -> Option<Self> {
        let device = metal::Device::system_default()?;

        // Compile shader source at runtime (llama.cpp pattern: GGML_METAL_EMBED_LIBRARY)
        let src = include_str!("metal.metal");
        let opts = metal::CompileOptions::new();
        let lib = device.new_library_with_source(src, &opts).map_err(|e| {
            eprintln!("MPS: failed to compile shaders: {}", e);
        }).ok()?;

        let fn_q4_0 = lib.get_function("kernel_q4_0_q8_0_matmul", None).ok()?;
        let pipeline_q4_0 = device
            .new_compute_pipeline_state_with_function(&fn_q4_0)
            .map_err(|e| {
                eprintln!("MPS: failed to create pipeline: {}", e);
            })
            .ok()?;

        let queue = device.new_command_queue();

        eprintln!("MPS: using Metal on {} (unified memory: {})",
            device.name(),
            if device.has_unified_memory() { "yes" } else { "no" });

        Some(Self {
            inner: MpsStateInner {
                device,
                queue,
                pipeline_q4_0,
                weights: std::sync::Mutex::new(std::collections::HashMap::new()),
            },
        })
    }

    #[cfg(not(target_os = "macos"))]
    pub fn try_new() -> Option<Self> {
        None
    }

    /// Get the global MPS state (if initialized and available).
    pub fn get() -> Option<&'static Self> {
        MPS.get().and_then(|s| s.as_ref())
    }

    /// Initialize (or disable) the MPS backend. Called once at startup.
    pub fn init() {
        MPS.get_or_init(|| {
            match Self::try_new() {
                Some(s) => {
                    eprintln!("MPS: GPU acceleration enabled");
                    Some(s)
                }
                None => {
                    eprintln!("MPS: not available, using CPU fallback");
                    None
                }
            }
        });
    }

    /// Check if a weight tensor by name is registered on GPU.
    pub fn has_weight(&self, name: &str) -> bool {
        #[cfg(target_os = "macos")]
        { self.inner.weights.lock().unwrap().contains_key(name) }
        #[cfg(not(target_os = "macos"))]
        { false }
    }

    /// Register a weight tensor's data on the GPU.
    /// Called during model loading for each quantized weight tensor.
    #[cfg(target_os = "macos")]
    pub fn register_weight(&self, name: &str, data: &[u8]) {
        if data.is_empty() { return; }
        let buf = self.inner.device.new_buffer_with_data(
            data.as_ptr() as *const std::ffi::c_void,
            data.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        self.inner.weights.lock().unwrap().insert(name.to_string(), buf);
    }

    #[cfg(not(target_os = "macos"))]
    pub fn register_weight(&self, _name: &str, _data: &[u8]) {}

    /// Dispatch quantized matrix multiplication (Q4_0 × Q8_0) on GPU.
    /// Only Q4_0 weights are accelerated; other quantized types fall back to CPU.
    #[cfg(target_os = "macos")]
    pub fn quant_matmul(
        &self,
        w: &Tensor,
        x: &[u8],
        out: &mut [f32],
        od: usize,
        id: usize,
        nt: usize,
    ) {
        if w.ttype != TensorType::Q4_0 {
            return crate::kernel::cpu_quant_matmul(w, x, out, od, id, nt);
        }

        let inner = &self.inner;
        let weights = inner.weights.lock().unwrap();
        let wbuf = match weights.get(&w.name) {
            Some(buf) => buf,
            None => {
                return crate::kernel::cpu_quant_matmul(w, x, out, od, id, nt);
            }
        };

        let nb = id / 32;
        let x_len = nt * nb * 34;  // Q8B
        let out_len = nt * od;

        // Activation buffer: create a shared MTLBuffer from CPU memory
        let xbuf = inner.device.new_buffer_with_data(
            x.as_ptr() as *const std::ffi::c_void,
            x_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let obuf = inner.device.new_buffer(
            (out_len * 4) as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );

        let cmd_buf = inner.queue.new_command_buffer();
        let enc = cmd_buf.new_compute_command_encoder();

        enc.set_compute_pipeline_state(&inner.pipeline_q4_0);
        enc.set_buffer(0, Some(wbuf), 0);
        enc.set_buffer(1, Some(&xbuf), 0);
        enc.set_buffer(2, Some(&obuf), 0);

        let od_i: i32 = od as i32;
        let id_i: i32 = id as i32;
        let nt_i: i32 = nt as i32;

        let set = |idx: u64, val: &i32| {
            enc.set_bytes(
                idx,
                std::mem::size_of::<i32>() as u64,
                val as *const i32 as *const std::ffi::c_void,
            );
        };
        set(3, &od_i);
        set(4, &id_i);
        set(5, &nt_i);

        let thread_group_size = metal::MTLSize {
            width: 32u64,
            height: 1u64,
            depth: 1u64,
        };
        let thread_groups = metal::MTLSize {
            width: ((nt + 31) / 32) as u64,
            height: od as u64,
            depth: 1u64,
        };
        enc.dispatch_thread_groups(thread_groups, thread_group_size);

        enc.end_encoding();
        cmd_buf.commit();
        cmd_buf.wait_until_completed();

        unsafe {
            std::ptr::copy_nonoverlapping(
                obuf.contents() as *const f32,
                out.as_mut_ptr(),
                out_len,
            );
        }
    }

    #[cfg(not(target_os = "macos"))]
    pub fn quant_matmul(
        &self,
        w: &Tensor,
        x: &[u8],
        out: &mut [f32],
        od: usize,
        id: usize,
        nt: usize,
    ) {
        crate::kernel::cpu_quant_matmul(w, x, out, od, id, nt);
    }
}


