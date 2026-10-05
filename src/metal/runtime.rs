// Metal backend L1 device runtime (pre-split `src/metal.rs`).
//
// Moved verbatim by the #265 layout split: the `MpsState` singleton and the
// compile/load paths (`try_new` builds the pipeline cache; the embedded
// metallib is preferred and the SAME concatenated source build.rs compiled is
// the fallback). `MpsState` itself stays declared in `src/metal.rs`, the parent
// module, so its private `inner` field stays reachable here with no visibility
// edit — the same shape as `CudaState` in `src/cuda.rs`.

use super::*;
use dispatch2::DispatchData;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCaptureDescriptor, MTLCaptureDestination, MTLCaptureManager, MTLCreateSystemDefaultDevice,
    MTLResourceOptions,
};

// ─── MpsState (global singleton) ─────────────────────────────────────

/// Compile the shader source at runtime (fallback when the build-time metallib is
/// unavailable — see try_new). ~0.3-1 s per process start.
///
/// The source is the SAME concatenation `build.rs` compiles into the metallib:
/// build.rs writes `$OUT_DIR/minfer.metal` from its `SHADER_SOURCES` list (#265),
/// so the two compile entry points cannot drift apart.
#[cfg(target_os = "macos")]
fn compile_metal_source(device: &MetalDevice) -> Option<MetalLibrary> {
    let src = include_str!(concat!(env!("OUT_DIR"), "/minfer.metal"));
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

// L1 device/runtime surface: the singleton, weight registration, f32 buffer
// allocation and command-buffer creation. The #255 macOS oracle (`cargo check
// --release`, annotations stripped) reports no dead member here — loaders,
// `main.rs`, `graph/metal_backend.rs` and `get_or_grow`/`cmd_buffer` all read
// them — so the former container-level `#[allow(dead_code)]` is deleted rather
// than narrowed.
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
}
