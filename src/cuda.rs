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

pub struct CudaState {
    stream: Mutex<CudaPtr>,
    /// Lazy pinned staging ring (7e⑥); None until the first async fill,
    /// and stays None if cudaHostAlloc fails (sync fallback).
    staging: Mutex<Option<PinnedPool>>,
    weights: Mutex<HashMap<String, (CudaPtr, usize)>>,
    /// Names registered through `register_weight_q6k_padded` (device layout
    /// is 224-byte-padded Q6_K, not the raw GGUF byte stream) → the
    /// ORIGINAL raw byte length, so `has_weight_of_size` can still match
    /// tensors by their raw GGUF size.
    padded_weights: Mutex<HashMap<String, usize>>,
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

    fn try_new() -> Option<Self> {
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

        // Auto-select the device with highest compute capability
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
        Some(CudaState {
            stream: Mutex::new(CudaPtr(stream)),
            staging: Mutex::new(None),
            weights: Mutex::new(HashMap::new()),
            padded_weights: Mutex::new(HashMap::new()),
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
            decode_graph_exec: Mutex::new(CudaPtr(std::ptr::null_mut())),
            stream_lock: Mutex::new(()),
        })
    }

    pub fn get() -> Option<&'static Self> {
        CUDA.get().and_then(|s| s.as_ref())
    }

    pub fn init() {
        CUDA.get_or_init(|| {
            let s = Self::try_new();
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
            for _ in 0..STAGING_SLOTS {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                if unsafe { cudaHostAlloc(&mut p, STAGING_SLOT_BYTES, 0) } != 0 {
                    break;
                }
                ptrs.push(p as *mut u8);
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
        // ring wrap: retire all in-flight copies before reusing slot 0
        if pool.next == pool.ptrs.len() {
            drop(guard);
            self.sync();
            guard = self.staging.lock().unwrap();
            let pool = guard.as_mut().unwrap();
            pool.next = 0;
        }
        let slot = {
            let pool = guard.as_mut().unwrap();
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
            TensorType::Q4_0 => launch!(launch_q4_0_f32_matmul),
            TensorType::Q8_0 => launch!(launch_q8_0_f32_matmul),
            TensorType::Q4_1 => launch!(launch_q4_1_f32_matmul),
            TensorType::Q4_K => launch!(launch_q4_k_f32_matmul),
            TensorType::Q6_K => {
                if padded_q6k {
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
                "cuda: weight type {other:?} has no f32-activation matmul kernel (supported: Q4_0/Q8_0/Q4_1/Q4_K/Q6_K)"
            )),
        }
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

    /// Decode I32 graph inputs (f32::from_bits bit patterns, alloc.rs
    /// fill_input_i32) into raw int32 for the rope/store/attention kernels.
    /// Fully device-side: no host sync, capture-safe (Phase 7d).
    pub fn bits_to_i32(&self, src: *mut std::ffi::c_void, dst: *mut std::ffi::c_void, n: usize) {
        let stream = self.stream();
        unsafe {
            launch_f32_bits_to_i32(src as *const f32, dst as *mut i32, n as i32, stream);
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
