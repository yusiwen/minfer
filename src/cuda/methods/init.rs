// device probe / tier selection / process singleton (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // #223: the **production** eager pre-warm entry, called once per process
    // from `CudaState::try_new` for every launchable `(tm, ks, af32)`. It drives
    // the same `gemm_smem_optin<TM,KS,AF32>` cache `launch_gemm_f16` reads, so
    // the pre-warm and the lazy path are one mechanism, not two. Outcome:
    //  1 = the attribute is in force (set now, or already cached);
    //  0 = refused — `minfer_smem_optin` named the instantiation, the requested
    //      bytes, the device limit and `cudaGetErrorName`, and cleared the latch;
    // -1 = the combination is not in the fatbin;
    // -2 = deliberately skipped: the request exceeds the device's
    //      `cudaDevAttrMaxSharedMemoryPerBlockOptin`, so the attribute was never
    //      called (the reason is named by `minfer_smem_optin`).
    pub(crate) fn gemm_prefill_smem_prewarm_one(tm: i32, ks: i32, af32: i32) -> i32;
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
        // The banner's total is a convenience, but the return code is checked all the
        // same (issue #122): a failed query here must not print a bare "0 MB" device.
        let mem_rc = unsafe { cudaMemGetInfo(&mut free_mem, &mut total_mem) };
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
        if mem_rc != 0 {
            eprintln!(
                "CUDA: device memory query at init failed with {} (code {}); the banner's \
                 size is not a measurement",
                cuda_error_name(mem_rc),
                mem_rc
            );
        }
        // T1: resolve the device tier (plan §5.3). MINFER_DEVICE_TIER=<key>
        // forces a row by llama.cpp-style key (1210/1200/890/870/860/750) —
        // the forced-tier soak runs the whole suite under a foreign tier to
        // prove gates only ever choose among correct kernels.
        let cc_val = major * 100 + minor;
        let tier_mmq = match std::env::var("MINFER_DEVICE_TIER")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            Some(key) => {
                let s = device_tier::select_forced(key);
                eprintln!(
                    "CUDA: device tier FORCED {} ({:?}, mmq {}) — key {}",
                    s.tier.name, s.tier.provenance, s.mmq_available, key
                );
                s.mmq_available
            }
            None => {
                let s = device_tier::select(cc_val);
                eprintln!(
                    "CUDA: device tier {} ({:?}, mmq {})",
                    s.tier.name, s.tier.provenance, s.mmq_available
                );
                s.mmq_available
            }
        };
        // T2 (plan §6.2): BT dynamic-smem feasibility — the tile config's
        // demand (single-source formula in cuda_kernels.cu) must fit the
        // device's opt-in limit. Degrading here routes prefill to the f16
        // GEMM path instead of failing launches on 100 KB-class devices.
        // GB10 passes (identical to the previous unconditional behavior).
        let mut tier_mmq = tier_mmq;
        if tier_mmq {
            let smem_need = unsafe { cuda_mmq_smem_bytes() };
            let smem_have = unsafe { cuda_shared_per_block_optin() };
            if smem_need > smem_have {
                eprintln!(
                    "CUDA: BT tile smem {smem_need} B > device optin {smem_have} B — MMQ prefill disabled, f16 GEMM path serves"
                );
                tier_mmq = false;
            }
        }

        // ── Issue #223: eager prefill-GEMM dynamic-smem pre-warm ──────────────
        //
        // #188 deleted the #145 sweep's call from exactly this point and said so
        // nowhere; #218 removed the orphan and made the invariant gated but still
        // only *emergent* (a tested property, not an enforced one). This restores
        // the runtime guarantee at the same site.
        //
        // **Placement, by construction.** `try_new` runs once per process under
        // `CUDA.get_or_init`, before the state is published, before any
        // `CudaBackend` exists, and therefore before any per-instance stream — the
        // only place `graph_begin_capture` can open a window — can exist. So "the
        // attribute is set outside any capture window" is true by construction
        // here, not inferred from the 3-run warmup or the thread-local capture
        // mode; those two remain as defence in depth, and the lazy per-launch
        // opt-in (`gemm_smem_optin`) stays too, so a process that sets
        // `MINFER_NO_GEMM_PREWARM=1` (the documented A/B control) behaves exactly
        // as it did after #218.
        //
        // **One mechanism, not two.** Every entry drives the **production**
        // `gemm_prefill_smem_prewarm_one` → `gemm_smem_optin<TM,KS,AF32>`, i.e.
        // the same per-instantiation cache the launcher reads. A cache-keying
        // regression (the #218 `template <typename K>` bug) therefore cannot be
        // masked by the pre-warm: it makes the pre-warm itself leave
        // instantiations un-opted-in, which the #223 gate and the #218 coverage
        // gate both read back from the device.
        //
        // **No banner.** A fully admitted pre-warm prints nothing; each failure
        // or deliberate skip is named per instantiation (the instantiation, the
        // requested bytes, the queried device limit and `cudaGetErrorName`), and
        // the removed `checked`/`skipped` counters do not come back.
        if !gemm_prewarm_disabled() {
            const GEMM_PREWARM_SET: [(i32, i32, i32); 12] = [
                (64, 32, 0),
                (64, 32, 1),
                (64, 64, 0),
                (64, 64, 1),
                (128, 32, 0),
                (128, 32, 1),
                (128, 64, 0),
                (128, 64, 1),
                (256, 32, 0),
                (256, 32, 1),
                (256, 64, 0),
                (256, 64, 1),
            ];
            let t0 = std::time::Instant::now();
            let mut named = 0usize; // failures + deliberate skips (diagnostic only)
            for &(tm, ks, af32) in GEMM_PREWARM_SET.iter() {
                match unsafe { gemm_prefill_smem_prewarm_one(tm, ks, af32) } {
                    1 => {}  // admitted (set now, or already cached)
                    -1 => {} // not compiled into this fatbin — nothing to do
                    // -2 = over the device limit, deliberately not called;
                    // 0 = the attribute call failed. Both were named by
                    // `minfer_smem_optin` at the `prewarm:gemm_f16` site.
                    -2 | 0 => named += 1,
                    other => {
                        eprintln!(
                            "CUDA: prefill-GEMM smem pre-warm for gemm_f16_nt_kernel_t<{tm},{ks},{}> \
                             returned an unexpected outcome {other}",
                            af32 != 0
                        );
                        named += 1;
                    }
                }
            }
            if crate::optiming::flag_from_env(std::env::var_os("MINFER_OP_TIMING").as_ref()) {
                eprintln!(
                    "CUDA: prefill-GEMM smem pre-warm ({} instantiation(s), {named} refused/skipped \
                     and named above) took {} µs",
                    GEMM_PREWARM_SET.len(),
                    t0.elapsed().as_micros()
                );
            }
        }

        // Issue #188: publish the context stream so the per-stream scratch
        // maps can key unbound callers (legacy layer path, direct tests).
        DEFAULT_STREAM.store(stream as usize, Ordering::Relaxed);
        Some(CudaState {
            stream: Mutex::new(CudaPtr(stream)),
            staging: Mutex::new(HashMap::new()),
            readback: Mutex::new(None),
            weights: Mutex::new(HashMap::new()),
            w16_cache: Mutex::new(HashMap::new()),
            w16_enabled: std::sync::atomic::AtomicBool::new(false),
            cc: std::sync::atomic::AtomicI32::new(major * 100 + minor),
            tier_mmq,
            sm_count,
            nb_bt_only: std::sync::atomic::AtomicBool::new(true),
            padded_weights: Mutex::new(HashMap::new()),
            q80_p32: Mutex::new(HashMap::new()),
            q6k_exp: Mutex::new(HashMap::new()),
            q6k_exp_warned: std::sync::atomic::AtomicBool::new(false),
            q6k_dpl: Mutex::new(HashMap::new()),
            q6k_dsc: Mutex::new(HashMap::new()),
            q6k_dsc_warned: std::sync::atomic::AtomicBool::new(false),
            q4k_dsc: Mutex::new(HashMap::new()),
            q4k_dsc_warned: std::sync::atomic::AtomicBool::new(false),
            max_nchunk: std::sync::atomic::AtomicUsize::new(0),
            buf_q8_prefill: StreamScratch::new(),
            buf_qa8_t: StreamScratch::new(),
            buf_sda_t: StreamScratch::new(),
            buf_mmq_ksplit: StreamScratch::new(),
            mmq_cache: Mutex::new(HashMap::new()),
            buf_attn_partial: StreamScratch::new(),
            buf_q8_decode: StreamScratch::new(),
            buf_f16_w: StreamScratch::new(),
            buf_f16_x: StreamScratch::new(),
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
}
