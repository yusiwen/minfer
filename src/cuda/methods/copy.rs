// host<->device copies (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

impl CudaState {
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
        // #188: the ring is keyed on the bound stream — its slots are the
        // source of a `cudaMemcpyAsync` on that stream, so two engines must not
        // share one.
        let key = current_stream_key();
        let mut map = self.staging.lock().unwrap();
        if !map.contains_key(&key) {
            let mut ptrs = Vec::new();
            let mut alloc_err = 0i32;
            for _ in 0..STAGING_SLOTS {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                alloc_err = unsafe { cudaHostAlloc(&mut p, STAGING_SLOT_BYTES, 0) };
                if alloc_err != 0 {
                    break;
                }
                ptrs.push(p as *mut u8);
            }
            // a shrunken ring silently degrades to a stream sync every
            // `ptrs.len()` fills — surface it (Phase 8 review)
            if ptrs.len() != STAGING_SLOTS {
                eprintln!(
                    "CUDA: pinned staging ring shrunk to {}/{} slots (cudaHostAlloc err {});                      fills beyond the ring fall back to sync copies",
                    ptrs.len(),
                    STAGING_SLOTS,
                    alloc_err
                );
            }
            if !ptrs.is_empty() {
                map.insert(
                    key,
                    PinnedPool {
                        ptrs,
                        slot_bytes: STAGING_SLOT_BYTES,
                        next: 0,
                    },
                );
            }
        }
        let fits = map.get(&key).is_some_and(|p| data.len() <= p.slot_bytes);
        if !fits {
            drop(map);
            self.copy_to_device(data, dst);
            return;
        }
        // ring wrap: retire all in-flight copies before reusing slot 0. The
        // reset is re-checked under the re-lock so two threads that both
        // observed the full ring cannot both take slot 0 (Phase 8 review).
        if map.get(&key).is_some_and(|p| p.next == p.ptrs.len()) {
            drop(map);
            self.sync();
            map = self.staging.lock().unwrap();
        }
        let slot = {
            let pool = map.get_mut(&key).expect("the ring exists (checked above)");
            if pool.next >= pool.ptrs.len() {
                pool.next = 0;
            }
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

    /// R3-A2: D2H read through our own pinned staging buffer. The caller has
    /// already synchronized the stream; the copy is a blocking `cudaMemcpy`
    /// whose DESTINATION is pinned — no driver-internal bounce buffer, no
    /// pageable staging — followed by a plain CPU copy out to the caller's
    /// (pageable) slice. `MINFER_NO_PINNED_READBACK=1` or a cudaHostAlloc
    /// failure falls back to the pageable path (`copy_from_device`).
    pub fn copy_from_device_pinned(&self, src: *const std::ffi::c_void, dst: &mut [u8]) {
        static FALLBACK_WARNED: std::sync::atomic::AtomicBool =
            std::sync::atomic::AtomicBool::new(false);
        if std::env::var("MINFER_NO_PINNED_READBACK").as_deref() == Ok("1") {
            self.copy_from_device(src, dst);
            return;
        }
        // headroom so small size changes don't churn the allocation
        let need = dst.len().max(4 * 1024 * 1024);
        let mut guard = self.readback.lock().unwrap();
        if guard.as_ref().map_or(true, |b| b.bytes < need) {
            if let Some(old) = guard.take() {
                drop(old); // cudaFreeHost
            }
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            let err = unsafe { cudaHostAlloc(&mut p, need, 0) };
            if err != 0 {
                drop(guard);
                if !FALLBACK_WARNED.swap(true, std::sync::atomic::Ordering::Relaxed) {
                    eprintln!(
                        "CUDA: pinned readback alloc failed (err {err}); pageable D2H fallback"
                    );
                }
                self.copy_from_device(src, dst);
                return;
            }
            *guard = Some(PinnedBuf {
                ptr: p as *mut u8,
                bytes: need,
            });
        }
        let buf = guard.as_mut().unwrap();
        unsafe {
            cudaMemcpy(
                buf.ptr as *mut std::ffi::c_void,
                src,
                dst.len(),
                CUDA_MEMCPY_DEVICE_TO_HOST,
            );
            std::ptr::copy_nonoverlapping(buf.ptr, dst.as_mut_ptr(), dst.len());
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
}
