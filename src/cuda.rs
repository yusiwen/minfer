// CUDA (NVIDIA GPU) backend for x86-64 Linux/Windows.
// Translated from: src/metal.rs (Metal backend pattern) and
//   llama.cpp/ggml/src/ggml-cuda (CUDA kernel dispatch)

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use crate::tensor::{Tensor, TensorType};
use crate::block::Q8B;

/// Wrapper to make `*mut c_void` Send+Sync for use in Mutex.
#[derive(Clone, Copy)]
struct CudaPtr(*mut std::ffi::c_void);
unsafe impl Send for CudaPtr {}
unsafe impl Sync for CudaPtr {}

// ─── FFI declarations for CUDA runtime API ────────────────────

#[repr(C)]
struct cudaIpcEventHandle_t([u8; 64]);

extern "C" {
    fn cudaSetDevice(device: i32) -> i32;
    fn cudaFree(ptr: *mut std::ffi::c_void) -> i32;
    fn cudaMalloc(ptr: *mut *mut std::ffi::c_void, size: usize) -> i32;
    fn cudaMemcpy(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32) -> i32;
    fn cudaMemcpyAsync(dst: *mut std::ffi::c_void, src: *const std::ffi::c_void, count: usize, kind: i32, stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamCreate(stream: *mut *mut std::ffi::c_void) -> i32;
    fn cudaStreamDestroy(stream: *mut std::ffi::c_void) -> i32;
    fn cudaStreamSynchronize(stream: *mut std::ffi::c_void) -> i32;
    fn cudaGetDeviceCount(count: *mut i32) -> i32;
    fn cudaGetDeviceProperties(prop: *mut cudaDeviceProp, device: i32) -> i32;
}

#[repr(C)]
struct cudaDeviceProp {
    name: [i8; 256],
    totalGlobalMem: usize,
    sharedMemPerBlock: usize,
    regsPerBlock: i32,
    warpSize: i32,
    memPitch: usize,
    maxThreadsPerBlock: i32,
    maxThreadsDim: [i32; 3],
    maxGridSize: [i32; 3],
    clockRate: i32,
    totalConstMem: usize,
    major: i32,
    minor: i32,
    textureAlignment: usize,
    texturePitchAlignment: usize,
    deviceOverlap: i32,
    multiProcessorCount: i32,
    kernelExecTimeoutEnabled: i32,
    integrated: i32,
    canMapHostMemory: i32,
    computeMode: i32,
    maxTexture1D: i32,
    maxTexture1DLinear: i32,
    maxTexture2D: [i32; 2],
    maxTexture2DLinear: [i32; 3],
    maxTexture2DGather: [i32; 2],
    maxTexture3D: [i32; 3],
    maxTextureCubemap: i32,
    maxTexture1DLayered: [i32; 2],
    maxTexture2DLayered: [i32; 3],
    maxTextureCubemapLayered: [i32; 2],
    maxSurface1D: i32,
    maxSurface2D: [i32; 2],
    maxSurface3D: [i32; 3],
    maxSurface1DLayered: [i32; 2],
    maxSurface2DLayered: [i32; 3],
    maxSurfaceCubemap: i32,
    maxSurfaceCubemapLayered: [i32; 2],
    surfaceAlignment: usize,
    concurrentKernels: i32,
    ECCEnabled: i32,
    pciBusID: i32,
    pciDeviceID: i32,
    pciDomainID: i32,
    tccDriver: i32,
    asyncEngineCount: i32,
    unifiedAddressing: i32,
    memoryClockRate: i32,
    memoryBusWidth: i32,
    l2CacheSize: i32,
    maxThreadsPerMultiProcessor: i32,
    streamPrioritiesSupported: i32,
    globalL1CacheSupported: i32,
    localL1CacheSupported: i32,
    sharedMemPerMultiprocessor: usize,
    regsPerMultiprocessor: i32,
    managedMemory: i32,
    isMultiGpuBoard: i32,
    multiGpuBoardGroupID: i32,
    hostNativeAtomicSupported: i32,
    singleToDoublePrecisionPerfRatio: i32,
    pageableMemoryAccess: i32,
    concurrentManagedAccess: i32,
    computePreemptionSupported: i32,
    canUseHostPointerForRegisteredMem: i32,
    cooperativeLaunch: i32,
    cooperativeMultiDeviceLaunch: i32,
    sharedMemPerBlockMultiprocessor: usize,
    maxRegsPerMultiprocessor: i32,
    managedMemoryPerDevice: usize,
    accessPolicyMaxWindowSize: i32,
    reserved: [i32; 448],
}

const cudaMemcpyHostToDevice: i32 = 1;
const cudaMemcpyDeviceToHost: i32 = 2;
const cudaMemcpyDeviceToDevice: i32 = 3;

fn cuda_check(err: i32, msg: &str) {
    if err != 0 {
        eprintln!("CUDA error ({}): {}", msg, err);
    }
}

// ─── FFI declarations for kernel launch wrappers ───────────

extern "C" {
    fn launch_q4_0_q8_0_matmul(
        weights: *const u8, acts: *const u8, output: *mut f32,
        od: i32, id: i32, nt: i32, stream: *mut std::ffi::c_void
    );
    fn launch_q4_0_f32_matmul(
        weights: *const u8, acts: *const f32, output: *mut f32,
        od: i32, id: i32, nt: i32, stream: *mut std::ffi::c_void
    );
    fn launch_quantize_q8_0(
        x: *const f32, y: *mut u8, dim: i32, nt: i32, stream: *mut std::ffi::c_void
    );
    fn launch_rms_norm_f32(
        x: *const f32, w: *const f32, y: *mut f32,
        d: i32, eps: f32, n: i32, stream: *mut std::ffi::c_void
    );
    fn launch_add_bias_f32(
        y: *mut f32, b: *const f32, d: i32, n: i32, stream: *mut std::ffi::c_void
    );
    fn launch_add_f32(
        x: *const f32, y: *const f32, z: *mut f32, n: i32, stream: *mut std::ffi::c_void
    );
    fn launch_mul_f32(
        x: *const f32, y: *const f32, z: *mut f32, n: i32, stream: *mut std::ffi::c_void
    );
    fn launch_silu_f32(
        y: *mut f32, n: i32, stream: *mut std::ffi::c_void
    );
    fn launch_swiglu_f32(
        gate: *const f32, up: *const f32, dst: *mut f32, n: i32, stream: *mut std::ffi::c_void
    );
    fn launch_rope_f32(
        x: *mut f32, n_head: i32, n_dims: i32, nt: i32,
        freq_base: f32, freq_scale: f32, positions: *const i32, stream: *mut std::ffi::c_void
    );
    fn launch_store_kv_f32(
        src: *const f32, dst: *mut f32, nkt: i32, nt: i32,
        positions: *const i32, stream: *mut std::ffi::c_void
    );
    fn launch_gqa_attn_f32(
        q: *const f32, k: *const f32, v: *const f32, o: *mut f32,
        positions: *const i32, nh: i32, nk: i32, hd: i32,
        scale: f32, nt: i32, stream: *mut std::ffi::c_void
    );
}

// ─── CudaState singleton ───────────────────────────────────────

static CUDA: OnceLock<Option<CudaState>> = OnceLock::new();

pub struct CudaState {
    stream: Mutex<CudaPtr>,
    weights: Mutex<HashMap<String, CudaPtr>>,
    // Persistent activation buffers (grown on demand) with size tracking
    buf_hidden: Mutex<(CudaPtr, usize)>,
    buf_bn: Mutex<(CudaPtr, usize)>,
    buf_bq: Mutex<(CudaPtr, usize)>,
    buf_bk: Mutex<(CudaPtr, usize)>,
    buf_bv: Mutex<(CudaPtr, usize)>,
    buf_ba: Mutex<(CudaPtr, usize)>,
    buf_bf: Mutex<(CudaPtr, usize)>,
    buf_bg: Mutex<(CudaPtr, usize)>,
    buf_q8_bn: Mutex<(CudaPtr, usize)>,
    buf_q8_ba: Mutex<(CudaPtr, usize)>,
    buf_positions: Mutex<(CudaPtr, usize)>,
    buf_logits: Mutex<(CudaPtr, usize)>,
    // Persistent per-layer GPU KV cache (k, v) and current size
    kv_k: Mutex<Vec<CudaPtr>>,
    kv_v: Mutex<Vec<CudaPtr>>,
    kv_size: Mutex<Vec<usize>>,
}

impl CudaState {
    fn try_new() -> Option<Self> {
        if std::env::var("MINFER_DISABLE_CUDA").is_ok() {
            eprintln!("CUDA: disabled by MINFER_DISABLE_CUDA");
            return None;
        }

        let mut count: i32 = 0;
        let err = unsafe { cudaGetDeviceCount(&mut count) };
        if err != 0 || count == 0 {
            eprintln!("CUDA: no CUDA devices found");
            return None;
        }

        let err = unsafe { cudaSetDevice(0) };
        if err != 0 {
            eprintln!("CUDA: failed to set device 0");
            return None;
        }

        let mut stream: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaStreamCreate(&mut stream) };
        if err != 0 || stream.is_null() {
            eprintln!("CUDA: failed to create stream");
            return None;
        }

        // Read device properties
        let mut prop: cudaDeviceProp = unsafe { std::mem::zeroed() };
        unsafe { cudaGetDeviceProperties(&mut prop, 0); }
        let name = prop.name.iter().take_while(|&&c| c != 0).map(|&c| c as u8 as char).collect::<String>();
        eprintln!("CUDA: using {} (SM {}.{}, {} MB, {} SMs)",
            name, prop.major, prop.minor,
            prop.totalGlobalMem / 1048576,
            prop.multiProcessorCount);

        let dummy = (CudaPtr(std::ptr::null_mut()), 0usize);
        Some(CudaState {
            stream: Mutex::new(CudaPtr(stream)),
            weights: Mutex::new(HashMap::new()),
            buf_hidden: Mutex::new(dummy),
            buf_bn: Mutex::new(dummy),
            buf_bq: Mutex::new(dummy),
            buf_bk: Mutex::new(dummy),
            buf_bv: Mutex::new(dummy),
            buf_ba: Mutex::new(dummy),
            buf_bf: Mutex::new(dummy),
            buf_bg: Mutex::new(dummy),
            buf_q8_bn: Mutex::new(dummy),
            buf_q8_ba: Mutex::new(dummy),
            buf_positions: Mutex::new(dummy),
            buf_logits: Mutex::new(dummy),
            kv_k: Mutex::new(Vec::new()),
            kv_v: Mutex::new(Vec::new()),
            kv_size: Mutex::new(Vec::new()),
        })
    }

    pub fn get() -> Option<&'static Self> {
        CUDA.get().and_then(|s| s.as_ref())
    }

    pub fn init() {
        CUDA.get_or_init(|| {
            let s = Self::try_new();
            if s.is_some() { eprintln!("CUDA: GPU acceleration enabled"); }
            else { eprintln!("CUDA: not available, using CPU fallback"); }
            s
        });
    }

    pub fn has_weight(&self, name: &str) -> bool {
        self.weights.lock().unwrap().contains_key(name)
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        if data.is_empty() { return; }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
        if err != 0 || ptr.is_null() {
            eprintln!("CUDA: failed to allocate {} bytes for '{}'", data.len(), name);
            return;
        }
        let err = unsafe {
            cudaMemcpy(ptr, data.as_ptr() as *const std::ffi::c_void, data.len(), cudaMemcpyHostToDevice)
        };
        if err != 0 {
            eprintln!("CUDA: failed to copy '{}' to device", name);
            unsafe { cudaFree(ptr); }
            return;
        }
        self.weights.lock().unwrap().insert(name.to_string(), CudaPtr(ptr));
    }

    pub fn get_weight_ptr(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        self.weights.lock().unwrap().get(name).map(|cp| cp.0)
    }

    pub fn stream(&self) -> *mut std::ffi::c_void {
        self.stream.lock().unwrap().0
    }

    // ─── Persistent buffer management ─────────────────────────

    fn get_or_grow(slot: &Mutex<(CudaPtr, usize)>, need: usize) -> *mut std::ffi::c_void {
        let mut guard = slot.lock().unwrap();
        let (ptr, size) = &mut *guard;
        if ptr.0.is_null() || *size < need {
            if !ptr.0.is_null() {
                unsafe { cudaFree(ptr.0); }
            }
            let mut new_ptr: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { cudaMalloc(&mut new_ptr, need) };
            if err != 0 || new_ptr.is_null() {
                eprintln!("CUDA: OOM allocating {} bytes", need);
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

    fn cuda_malloc(size: usize) -> *mut std::ffi::c_void {
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, size) };
        if err != 0 || ptr.is_null() {
            eprintln!("CUDA: OOM allocating {} bytes", size);
            return std::ptr::null_mut();
        }
        ptr
    }

    // ─── Copy helpers ─────────────────────────────────────────

    pub fn copy_to_device(&self, src: &[u8], dst: *mut std::ffi::c_void) {
        unsafe {
            cudaMemcpy(dst, src.as_ptr() as *const std::ffi::c_void, src.len(), cudaMemcpyHostToDevice);
        }
    }

    pub fn copy_to_device_async(&self, src: &[u8], dst: *mut std::ffi::c_void) {
        unsafe {
            cudaMemcpyAsync(dst, src.as_ptr() as *const std::ffi::c_void, src.len(), cudaMemcpyHostToDevice, self.stream());
        }
    }

    pub fn copy_from_device(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        unsafe {
            cudaMemcpy(dst.as_mut_ptr() as *mut std::ffi::c_void, src, dst.len(), cudaMemcpyDeviceToHost);
        }
    }

    pub fn copy_from_device_async(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        unsafe {
            cudaMemcpyAsync(dst.as_mut_ptr() as *mut std::ffi::c_void, src, dst.len(), cudaMemcpyDeviceToHost, self.stream());
        }
    }

    pub fn copy_device_to_device(&self, src: *const std::ffi::c_void, dst: *mut std::ffi::c_void, size: usize) {
        unsafe {
            cudaMemcpy(dst as *mut std::ffi::c_void, src, size, cudaMemcpyDeviceToDevice);
        }
    }

    pub fn sync(&self) {
        unsafe { cudaStreamSynchronize(self.stream()); }
    }

    // ─── Upload/download for forward pass ─────────────────────

    pub fn upload_hidden(&self, hidden: &[f32]) {
        let need = hidden.len() * 4;
        let ptr = Self::get_or_grow(&self.buf_hidden, need);
        self.copy_to_device(unsafe { std::slice::from_raw_parts(hidden.as_ptr() as *const u8, need) }, ptr);
    }

    pub fn download_hidden(&self, hidden: &mut [f32]) {
        let need = hidden.len() * 4;
        let guard = self.buf_hidden.lock().unwrap();
        let ptr = guard.0.0;
        if ptr.is_null() { return; }
        self.copy_from_device(ptr as *const std::ffi::c_void,
            unsafe { std::slice::from_raw_parts_mut(hidden.as_mut_ptr() as *mut u8, need) });
    }

    pub fn upload_positions(&self, positions: &[usize]) {
        let ints: Vec<i32> = positions.iter().map(|&p| p as i32).collect();
        let need = ints.len() * 4;
        let ptr = Self::get_or_grow(&self.buf_positions, need);
        self.copy_to_device(unsafe { std::slice::from_raw_parts(ints.as_ptr() as *const u8, need) }, ptr);
    }

    pub fn get_positions_buf(&self) -> *mut std::ffi::c_void {
        self.buf_positions.lock().unwrap().0.0
    }

    // ─── KV cache management ─────────────────────────────────

    fn kv_ensure_layer(&self, il: usize, max_nkv: usize, nkt: usize) {
        let need = max_nkv * nkt * 4;
        let mut kvec = self.kv_k.lock().unwrap();
        let mut vvec = self.kv_v.lock().unwrap();
        let mut szvec = self.kv_size.lock().unwrap();
        while kvec.len() <= il {
            kvec.push(CudaPtr(std::ptr::null_mut()));
            vvec.push(CudaPtr(std::ptr::null_mut()));
            szvec.push(0);
        }
        let old_size = szvec.get(il).copied().unwrap_or(0) * nkt * 4;
        if old_size >= need { return; }

        // Allocate new
        let new_k = Self::cuda_malloc(need);
        let new_v = Self::cuda_malloc(need);
        if new_k.is_null() || new_v.is_null() { return; }

        // Copy old data
        if old_size > 0 {
            self.copy_device_to_device(kvec[il].0 as *const std::ffi::c_void, new_k, old_size);
            self.copy_device_to_device(vvec[il].0 as *const std::ffi::c_void, new_v, old_size);
            unsafe { cudaFree(kvec[il].0); cudaFree(vvec[il].0); }
        }

        kvec[il] = CudaPtr(new_k);
        vvec[il] = CudaPtr(new_v);
    }

    pub fn get_kv_size(&self, il: usize) -> usize {
        let szvec = self.kv_size.lock().unwrap();
        szvec.get(il).copied().unwrap_or(0)
    }

    /// Download logits from GPU after layer loop.
    pub fn download_logits(&self, logits: &mut [f32]) {
        let need = logits.len() * 4;
        let guard = self.buf_logits.lock().unwrap();
        let ptr = guard.0.0;
        if ptr.is_null() { return; }
        self.copy_from_device(ptr as *const std::ffi::c_void,
            unsafe { std::slice::from_raw_parts_mut(logits.as_mut_ptr() as *mut u8, need) });
    }

    // ─── Kernel launch operations (called from CudaCommandBuffer) ──

    pub fn quant_matmul_q8(&self, w: &Tensor, x: *mut std::ffi::c_void, out: *mut std::ffi::c_void,
        od: usize, id: usize, nt: usize) {
        let wptr = self.get_weight_ptr(&w.name).expect("weight not on GPU");
        let stream = self.stream();
        unsafe {
            launch_q4_0_q8_0_matmul(
                wptr as *const u8, x as *const u8, out as *mut f32,
                od as i32, id as i32, nt as i32, stream);
        }
    }

    pub fn quant_matmul_f32_on_gpu(&self, w: &Tensor, x: *mut std::ffi::c_void, out: *mut std::ffi::c_void,
        od: usize, id: usize, nt: usize) {
        if w.ttype != TensorType::Q4_0 {
            // Non-Q4_0 weights not yet supported in CUDA f32 matmul;
            // caller should use quantize->Q8_0 path or CPU fallback.
            panic!("CUDA: unsupported weight type {:?} for f32 matmul", w.ttype);
        }
        let wptr = self.get_weight_ptr(&w.name).expect("weight not on GPU");
        let stream = self.stream();
        unsafe {
            launch_q4_0_f32_matmul(
                wptr as *const u8, x as *const f32, out as *mut f32,
                od as i32, id as i32, nt as i32, stream);
        }
    }

    pub fn matmul_on_gpu(&self, w: &Tensor, q8_x: *mut std::ffi::c_void, f32_x: *mut std::ffi::c_void,
        out: *mut std::ffi::c_void, od: usize, id: usize, nt: usize) {
        if w.ttype == TensorType::Q4_0 {
            self.quant_matmul_q8(w, q8_x, out, od, id, nt);
        } else if w.ttype == TensorType::Q4_1 {
            // Fall back to CPU for Q4_1 (not yet implemented in CUDA)
            // The caller must handle this. For now we just use the f32 matmul path
            // which will also fall back.
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        } else {
            // For other types (Q4_K, Q6_K, Q8_0), we need the f32 matmul path
            self.quant_matmul_f32_on_gpu(w, f32_x, out, od, id, nt);
        }
    }

    pub fn quantize_q8_0(&self, x: *mut std::ffi::c_void, y: *mut std::ffi::c_void, dim: usize, nt: usize) {
        let stream = self.stream();
        unsafe {
            launch_quantize_q8_0(x as *const f32, y as *mut u8, dim as i32, nt as i32, stream);
        }
    }

    pub fn rms_norm(&self, x: *mut std::ffi::c_void, w: Option<*mut std::ffi::c_void>, y: *mut std::ffi::c_void,
        d: usize, n: usize, eps: f32) {
        let wptr = w.unwrap_or(y); // dummy if no weight (but weight is always present for Qwen2)
        let stream = self.stream();
        unsafe {
            launch_rms_norm_f32(x as *const f32, wptr as *const f32, y as *mut f32,
                d as i32, eps, n as i32, stream);
        }
    }

    pub fn add_f32(&self, x: *mut std::ffi::c_void, y: *mut std::ffi::c_void, z: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_add_f32(x as *const f32, y as *const f32, z as *mut f32, n as i32, stream);
        }
    }

    pub fn add_bias_f32(&self, y: *mut std::ffi::c_void, b: *mut std::ffi::c_void, d: usize, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_add_bias_f32(y as *mut f32, b as *const f32, d as i32, n as i32, stream);
        }
    }

    pub fn mul_f32(&self, x: *mut std::ffi::c_void, y: *mut std::ffi::c_void, z: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_mul_f32(x as *const f32, y as *const f32, z as *mut f32, n as i32, stream);
        }
    }

    pub fn silu_f32(&self, y: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_silu_f32(y as *mut f32, n as i32, stream);
        }
    }

    pub fn swiglu_f32(&self, gate: *mut std::ffi::c_void, up: *mut std::ffi::c_void, dst: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_swiglu_f32(gate as *const f32, up as *const f32, dst as *mut f32, n as i32, stream);
        }
    }

    pub fn rope_f32(&self, x: *mut std::ffi::c_void, n_head: usize, n_dims: usize, nt: usize,
        freq_base: f32, freq_scale: f32, positions: *mut std::ffi::c_void) {
        let stream = self.stream();
        unsafe {
            launch_rope_f32(x as *mut f32, n_head as i32, n_dims as i32, nt as i32,
                freq_base, freq_scale, positions as *const i32, stream);
        }
    }

    pub fn gqa_attn_f32(&self, q: *mut std::ffi::c_void, k: *mut std::ffi::c_void, v: *mut std::ffi::c_void,
        o: *mut std::ffi::c_void, positions: *mut std::ffi::c_void, nh: usize, nk: usize, hd: usize, scale: f32, nt: usize) {
        let stream = self.stream();
        unsafe {
            launch_gqa_attn_f32(q as *const f32, k as *const f32, v as *const f32, o as *mut f32,
                positions as *const i32, nh as i32, nk as i32, hd as i32, scale, nt as i32, stream);
        }
    }

    pub fn store_kv_f32(&self, src: *mut std::ffi::c_void, dst: *mut std::ffi::c_void, nkt: usize, nt: usize,
        positions: *mut std::ffi::c_void) {
        let stream = self.stream();
        unsafe {
            launch_store_kv_f32(src as *const f32, dst as *mut f32, nkt as i32, nt as i32,
                positions as *const i32, stream);
        }
    }

    // ─── Batch quant_matmul (for Q/K/V projection) ────────────

    pub fn quant_matmul_f32_batch(
        &self,
        mats: &mut [(/*weight*/ &Tensor, /*output*/ &mut [f32], /*od*/ usize)],
        x: &[f32], id: usize, nt: usize,
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
        crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut q8);

        let xbuf = Self::get_or_grow(&self.buf_hidden, q8_len);
        self.copy_to_device(&q8, xbuf);

        // Launch each matmul and read back results
        for (i, mat) in mats.iter_mut().enumerate() {
            let out_len = nt * mat.2 * 4;
            let obuf = Self::get_or_grow(&self.buf_bq, out_len);
            self.quant_matmul_q8(mat.0, xbuf, obuf, mat.2, id, nt);
            self.sync();
            let out_bytes = unsafe {
                std::slice::from_raw_parts_mut(mat.1.as_mut_ptr() as *mut u8, out_len)
            };
            self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
        }
    }

    pub fn quant_matmul_f32(
        &self, w: &Tensor, x: &[f32], out: &mut [f32],
        od: usize, id: usize, nt: usize,
    ) {
        if w.ttype != TensorType::Q4_0 {
            return crate::kernel::cpu_quant_matmul_f32(w, x, out, od, id, nt);
        }

        let nb = id / 32;
        let q8_len = nt * nb * Q8B;
        let out_len = nt * od * 4;

        let mut q8 = vec![0u8; q8_len];
        crate::avx2::quantize_row_q8_0_buf(x, nt, id, &mut q8);

        let xbuf = Self::get_or_grow(&self.buf_hidden, q8_len);
        let obuf = Self::get_or_grow(&self.buf_logits, out_len);

        self.copy_to_device(&q8, xbuf);
        self.quant_matmul_q8(w, xbuf, obuf, od, id, nt);
        self.sync();
        let out_bytes = unsafe {
            std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, out_len)
        };
        self.copy_from_device(obuf as *const std::ffi::c_void, out_bytes);
    }

    // ─── Full-layer GPU pass ──────────────────────────────────

    /// Encode one transformer layer onto the CUDA stream.
    /// Returns false if any weight is missing from GPU.
    pub fn layer_gpu(
        &self,
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

        // CUDA currently only supports Q4_0 weight type in layer_gpu
        let all_q4_0 = wq.ttype == TensorType::Q4_0 && wk.ttype == TensorType::Q4_0
            && wv.ttype == TensorType::Q4_0 && wo.ttype == TensorType::Q4_0
            && ffn_gate.ttype == TensorType::Q4_0 && ffn_up.ttype == TensorType::Q4_0
            && ffn_down.ttype == TensorType::Q4_0;
        if !all_q4_0 { return false; }

        if !self.has_weight(&wq.name) || !self.has_weight(&wk.name) || !self.has_weight(&wv.name)
            || !self.has_weight(&wo.name) || !self.has_weight(&ffn_gate.name)
            || !self.has_weight(&ffn_up.name) || !self.has_weight(&ffn_down.name) {
            return false;
        }
        let norm_attn_w = match self.get_weight_ptr(&attn_norm.name) { Some(p) => p, None => return false };
        let norm_ffn_w  = match self.get_weight_ptr(&ffn_norm.name)  { Some(p) => p, None => return false };
        let bq_bias = l.bq.as_ref().and_then(|b| self.get_weight_ptr(&b.name));
        let bk_bias = l.bk.as_ref().and_then(|b| self.get_weight_ptr(&b.name));
        let bv_bias = l.bv.as_ref().and_then(|b| self.get_weight_ptr(&b.name));

        let max_pos = positions.iter().copied().max().unwrap_or(0);
        self.kv_ensure_layer(il, max_pos + 1, nkt);

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
        self.quantize_q8_0(bn, q8_bn, ne, nt);
        self.matmul_on_gpu(wq, q8_bn, bn, bq_buf, nqt, ne, nt);
        if let Some(bb) = bq_bias { self.add_bias_f32(bq_buf, bb, nqt, nt); }
        self.matmul_on_gpu(wk, q8_bn, bn, bk_buf, nkt, ne, nt);
        if let Some(bb) = bk_bias { self.add_bias_f32(bk_buf, bb, nkt, nt); }
        self.matmul_on_gpu(wv, q8_bn, bn, bv_buf, nkt, ne, nt);
        if let Some(bb) = bv_bias { self.add_bias_f32(bv_buf, bb, nkt, nt); }
        self.rope_f32(bq_buf, nh, hd, nt, freq_base, freq_scale, pos_buf);
        self.rope_f32(bk_buf, nk, hd, nt, freq_base, freq_scale, pos_buf);
        self.store_kv_f32(bk_buf, kv_k as *mut std::ffi::c_void, nkt, nt, pos_buf);
        self.store_kv_f32(bv_buf, kv_v as *mut std::ffi::c_void, nkt, nt, pos_buf);
        let scale = 1.0 / (hd as f32).sqrt();
        self.gqa_attn_f32(bq_buf, kv_k as *mut std::ffi::c_void, kv_v as *mut std::ffi::c_void,
            ba_buf, pos_buf, nh, nk, hd, scale, nt);

        // wo projection
        self.quantize_q8_0(ba_buf, q8_ba, ne, nt);
        self.matmul_on_gpu(wo, q8_ba, ba_buf, bn, ne, ne, nt);
        self.add_f32(hidden, bn, hidden, nt * ne);

        // FFN branch
        self.rms_norm(hidden, Some(norm_ffn_w), ba_buf, ne, nt, eps);
        self.quantize_q8_0(ba_buf, q8_ba, ne, nt);
        self.matmul_on_gpu(ffn_gate, q8_ba, ba_buf, bg_buf, nf, ne, nt);
        self.matmul_on_gpu(ffn_up, q8_ba, ba_buf, bf_buf, nf, ne, nt);
        self.swiglu_f32(bg_buf, bf_buf, bg_buf, nt * nf);
        self.quantize_q8_0(bg_buf, q8_ba, nf, nt);
        self.matmul_on_gpu(ffn_down, q8_ba, bg_buf, bn, ne, nf, nt);
        self.add_f32(hidden, bn, hidden, nt * ne);

        self.kv_size.lock().unwrap()[il] = max_pos + 1;
        true
    }

    /// Final RMSNorm + output matmul on GPU.
    pub fn output_norm_gpu(
        &self,
        output: &Tensor,
        output_norm: Option<&Tensor>,
        output_b: Option<&Tensor>,
        ne: usize, nv: usize, nt: usize, eps: f32,
    ) -> bool {
        let norm_w = match output_norm {
            Some(t) => match self.get_weight_ptr(&t.name) {
                Some(w) => w,
                None => return false,
            },
            None => return false,
        };
        if !self.has_weight(&output.name) { return false; }
        if output.ttype != TensorType::Q4_0 { return false; }

        let hidden = Self::get_or_grow(&self.buf_hidden, nt * ne * 4);
        let bn = Self::get_or_grow(&self.buf_bn, nt * ne * 4);
        let logits = Self::get_or_grow(&self.buf_logits, nt * nv * 4);

        self.rms_norm(hidden, Some(norm_w), bn, ne, nt, eps);

        // Output projection
        let q8_len = nt * (ne / 32) * Q8B;
        let q8_bn = Self::get_or_grow(&self.buf_q8_bn, q8_len);
        self.quantize_q8_0(bn, q8_bn, ne, nt);
        self.quant_matmul_q8(output, q8_bn, logits, nv, ne, nt);

        if let Some(ob) = output_b {
            if let Some(bias_buf) = self.get_weight_ptr(&ob.name) {
                self.add_bias_f32(logits, bias_buf, nv, nt);
            }
        }
        true
    }

}
