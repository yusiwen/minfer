// weight registration + q6_k/q4_k quant expansion (pre-split `src/cuda.rs`).
//
// Moved verbatim by the #262 layout split; declared by `src/cuda/methods.rs`,
// so `use super::*` reaches the CUDA runtime declarations and the helpers.

use super::*;

extern "C" {
    // P6 r34: NB kernel whose A staging is bulk LDG->STS over the
    // PRE-TRANSPOSED qa8/sda buffers (MINFER_MMQ_A_TRANSPOSE=1). Returns 1 on
    // KD=8, 0 on clean fallback (KD!=8 / null transposed buffers / smem).
    // r59 rider: kernel module pre-load (cudaFuncGetAttributes over the
    // MMQ/FA/fused launch set) — see CudaState::prewarm_prefill.
    pub(crate) fn minfer_prewarm_kernels();
}

impl CudaState {
    /// [#758] start a new model registration and make it the current owner of
    /// everything the loader registers next.
    ///
    /// The returned handle is stored in the model. Its `Drop` calls
    /// [`Self::release_registration`], which frees the model's device buffers —
    /// immediately when no captured graph is live, or when the last capture is
    /// dropped if one still references them (the CaptureLease contract).
    ///
    /// The registration id is **not** the model fingerprint: two loads of the same
    /// file share the fingerprint (so the same-name+size device copy is reused) but
    /// are separate owners, and each must be releasable on its own.
    pub fn begin_model(&self, fingerprint: u64) -> std::sync::Arc<CudaModelRegistration> {
        let id = self
            .next_model_reg
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            + 1;
        self.model_id
            .store(id, std::sync::atomic::Ordering::Relaxed);
        self.model_fp
            .store(fingerprint, std::sync::atomic::Ordering::Relaxed);
        std::sync::Arc::new(CudaModelRegistration::new(id))
    }

    pub fn register_weight(&self, name: &str, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let model_id = self.model_id.load(std::sync::atomic::Ordering::Relaxed);
        let model_fp = self.model_fp.load(std::sync::atomic::Ordering::Relaxed);
        {
            let w = self.weights.lock().unwrap();
            if let Some((old_ptr, size)) = w.get(name) {
                if *size == data.len() {
                    let same_model =
                        self.weight_models.lock().unwrap().get(name).copied() == Some(model_fp);
                    if same_model {
                        // Device weights are immutable within a model: same
                        // name + size + model ⇒ the same GGUF tensor. Reuse the
                        // device copy instead of leaking one buffer per load —
                        // and add this registration as an owner, so a second
                        // live load of the same file cannot free it under the
                        // first.
                        let ptr = old_ptr.0 as usize;
                        drop(w);
                        self.add_owner(ptr, model_id);
                        return;
                    }
                    // A different model reusing the name and byte size: register
                    // a fresh device copy. The old buffer stays in the ownership
                    // ledger under its own registration (and is freed when that
                    // registration releases, or reaped when its captures go).
                }
                // Different size (a different architecture registered the same
                // tensor name): same replace-and-ledger rule.
            }
        }
        // ANCHOR: register_weight
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
            // Issue #188: stream-ordered, not the legacy-null-stream blocking
            // `cudaMemcpy`. The blocking form is not a stream operation at all:
            // while another thread holds a capture window open on a different
            // stream, it participates in the legacy default stream's implicit
            // global synchronization — under `cudaStreamCaptureModeGlobal` that
            // is exactly the call that invalidated the capture (901) or faulted
            // in `cuMemcpyHtoD_v2`. Queuing the copy on the context's own stream
            // and waiting on that stream keeps the registration a bounded,
            // stream-scoped operation.
            cudaMemcpyAsync(
                ptr,
                data.as_ptr() as *const std::ffi::c_void,
                data.len(),
                CUDA_MEMCPY_HOST_TO_DEVICE,
                self.context_stream(),
            )
        };
        if err == 0 {
            let serr = unsafe { cudaStreamSynchronize(self.context_stream()) };
            if serr != 0 {
                eprintln!(
                    "CUDA: weight-registration stream sync failed for '{}': {} ({serr})",
                    name,
                    cuda_error_name(serr)
                );
                unsafe {
                    cudaFree(ptr);
                }
                return;
            }
        }
        if err != 0 {
            eprintln!("CUDA: failed to copy '{}' to device", name);
            unsafe {
                cudaFree(ptr);
            }
            return;
        }
        // a plain (unpadded) registration must clear any stale padded flag
        // for the same name: a second model reusing the tensor name with a
        // non-Q6_K type would otherwise dispatch the padded-224 kernel on a
        // raw-210 buffer (Phase 8 review finding)
        self.padded_weights.lock().unwrap().remove(name);
        self.weights
            .lock()
            .unwrap()
            .insert(name.to_string(), (CudaPtr(ptr), data.len()));
        self.weight_models
            .lock()
            .unwrap()
            .insert(name.to_string(), model_fp);
        // [#758] record the buffer in the ownership ledger. The name map above is
        // only the "current buffer for this name" lookup; freeing follows owners,
        // so a replaced buffer is charged to the E4 budget until its model goes.
        let owners = if model_id == 0 {
            Vec::new()
        } else {
            vec![model_id]
        };
        self.model_allocs.lock().unwrap().insert(
            ptr as usize,
            DeviceAlloc {
                ptr: CudaPtr(ptr),
                size: data.len(),
                owners,
                retired: false,
            },
        );
        // ANCHOR_END: register_weight
    }

    /// [#758] add `reg_id` as an owner of the allocation at `ptr`, if the ledger
    /// knows it. This is the same-fingerprint reuse path: two live loads of one
    /// file share a device copy, so neither may free it while the other lives.
    fn add_owner(&self, ptr: usize, reg_id: u64) {
        if reg_id == 0 {
            return;
        }
        if let Some(alloc) = self.model_allocs.lock().unwrap().get_mut(&ptr) {
            if !alloc.owners.contains(&reg_id) {
                alloc.owners.push(reg_id);
            }
        }
    }

    /// [#758] record a lazily-built device cache allocation (the prefill f16
    /// weight copy, `w16_cache`) against the registrations that own its source
    /// weight, so a model drop frees it with the rest of that model's memory.
    /// `source_ptr` is the raw weight's device pointer the cache is keyed by.
    pub(crate) fn record_cache_alloc(
        &self,
        source_ptr: usize,
        ptr: *mut std::ffi::c_void,
        bytes: usize,
    ) {
        let mut allocs = self.model_allocs.lock().unwrap();
        let owners = match allocs.get(&source_ptr) {
            Some(source) => source.owners.clone(),
            None => {
                // A pre-ledger (test-only) registration has no owner; fall back to
                // the current model so a real load's cache still has a release
                // path, and an unattributed one simply stays for the process.
                let id = self.model_id.load(std::sync::atomic::Ordering::Relaxed);
                if id == 0 {
                    Vec::new()
                } else {
                    vec![id]
                }
            }
        };
        allocs.insert(
            ptr as usize,
            DeviceAlloc {
                ptr: CudaPtr(ptr),
                size: bytes,
                owners,
                retired: false,
            },
        );
    }

    /// [#758] release one model registration.
    ///
    /// Every allocation loses this owner; an allocation whose owner list becomes
    /// empty is either freed now (no live captured graph) or retired and kept for
    /// [`Self::reap_retired`] (a live capture may still replay a kernel over it).
    /// The name map entry is removed only while it still points at the released
    /// buffer: a newer model may already have replaced that name.
    pub(crate) fn release_registration(&self, id: u64) {
        let captures = self.live_captures.load(std::sync::atomic::Ordering::SeqCst);
        let mut freed: Vec<usize> = Vec::new();
        let mut released: Vec<(usize, usize)> = Vec::new();
        let mut freed_bytes = 0usize;
        let mut retained_bytes = 0usize;
        {
            let mut allocs = self.model_allocs.lock().unwrap();
            let keys: Vec<usize> = allocs.keys().copied().collect();
            for key in keys {
                let Some(alloc) = allocs.get_mut(&key) else {
                    continue;
                };
                if let Some(pos) = alloc.owners.iter().position(|owner| *owner == id) {
                    alloc.owners.remove(pos);
                    if alloc.owners.is_empty() {
                        released.push((key, alloc.size));
                    }
                }
            }
            for (ptr, _) in &released {
                if captures == 0 {
                    if let Some(done) = allocs.remove(ptr) {
                        freed_bytes += done.size;
                        freed.push(done.ptr.0 as usize);
                    }
                } else {
                    if let Some(alloc) = allocs.get_mut(ptr) {
                        alloc.retired = true;
                        retained_bytes += alloc.size;
                    }
                }
            }
        }
        if released.is_empty() {
            return;
        }
        let released: std::collections::HashSet<usize> =
            released.iter().map(|(ptr, _)| *ptr).collect();
        self.forget_released(&released);
        if !freed.is_empty() {
            self.free_device_allocs(&freed);
        }
        if retained_bytes > 0 {
            *self.last_release.lock().unwrap() = Some(ReleaseOutcome::Retained {
                bytes: retained_bytes,
                captures,
            });
            eprintln!(
                "CUDA: retained {retained_bytes} bytes of device weights after their model was \
                 dropped: {captures} captured graph(s) still reference them; they are freed when \
                 the last capture is dropped"
            );
        } else if freed_bytes > 0 {
            *self.last_release.lock().unwrap() = Some(ReleaseOutcome::Freed { bytes: freed_bytes });
        }
    }

    /// [#758] free every retired allocation once no capture can reference it.
    ///
    /// Called by the last [`CaptureLease`] drop, which is the only point where the
    /// answer changes from "a live exec may replay over these bytes" to "no live
    /// exec exists".
    pub(crate) fn reap_retired(&self) {
        if self.live_captures.load(std::sync::atomic::Ordering::SeqCst) > 0 {
            return;
        }
        let mut ptrs: Vec<usize> = Vec::new();
        let mut bytes = 0usize;
        {
            let mut allocs = self.model_allocs.lock().unwrap();
            let dead: Vec<usize> = allocs
                .iter()
                .filter(|(_, alloc)| alloc.retired && alloc.owners.is_empty())
                .map(|(ptr, _)| *ptr)
                .collect();
            for ptr in dead {
                if let Some(alloc) = allocs.remove(&ptr) {
                    bytes += alloc.size;
                    ptrs.push(alloc.ptr.0 as usize);
                }
            }
        }
        if ptrs.is_empty() {
            return;
        }
        let released: std::collections::HashSet<usize> = ptrs.iter().copied().collect();
        self.forget_released(&released);
        self.free_device_allocs(&ptrs);
        *self.last_release.lock().unwrap() = Some(ReleaseOutcome::Freed { bytes });
        eprintln!(
            "CUDA: released {bytes} bytes of retained device weights after the last captured \
             graph referencing them was dropped"
        );
    }

    /// [#758] remove every name and companion-plane entry that points at a
    /// released allocation.
    ///
    /// The `weights`/`weight_models` maps are the name → current buffer lookup, so
    /// only a name still pointing at the released buffer is removed — a newer
    /// model may have replaced it in the meantime.
    fn forget_released(&self, ptrs: &std::collections::HashSet<usize>) {
        let dead: Vec<String> = {
            let mut weights = self.weights.lock().unwrap();
            let dead: Vec<String> = weights
                .iter()
                .filter(|(_, (ptr, _))| ptrs.contains(&(ptr.0 as usize)))
                .map(|(name, _)| name.clone())
                .collect();
            for name in &dead {
                weights.remove(name);
            }
            dead
        };
        {
            let mut models = self.weight_models.lock().unwrap();
            for name in &dead {
                models.remove(name);
            }
        }
        {
            let mut padded = self.padded_weights.lock().unwrap();
            for name in &dead {
                padded.remove(name);
            }
        }
        // Companion planes: keyed by the main weight pointer (q6k/q4k/w16) or by
        // the raw q8_0 pointer (q80_p32 pair). Drop any entry whose key or value
        // names a released allocation.
        self.q6k_exp
            .lock()
            .unwrap()
            .retain(|k, v| !ptrs.contains(k) && !ptrs.contains(&(v.0 as usize)));
        self.q6k_dpl
            .lock()
            .unwrap()
            .retain(|k, v| !ptrs.contains(k) && !ptrs.contains(&(v.0 as usize)));
        self.q6k_dsc
            .lock()
            .unwrap()
            .retain(|k, v| !ptrs.contains(k) && !ptrs.contains(&(v.0 as usize)));
        self.q4k_dsc
            .lock()
            .unwrap()
            .retain(|k, v| !ptrs.contains(k) && !ptrs.contains(&(v.0 as usize)));
        self.q80_p32
            .lock()
            .unwrap()
            .retain(|k, v| !ptrs.contains(k) && !ptrs.contains(&v.0) && !ptrs.contains(&v.1));
        self.w16_cache
            .lock()
            .unwrap()
            .retain(|k, v| !ptrs.contains(k) && !ptrs.contains(&(v.0 .0 as usize)));
    }

    /// [#758] synchronize the context stream, then free the given allocations.
    ///
    /// Registration copied on the context's own stream, so draining that stream is
    /// what makes "no queued operation still writes these bytes" true before the
    /// driver recycles them.
    fn free_device_allocs(&self, ptrs: &[usize]) {
        unsafe {
            cudaStreamSynchronize(self.context_stream());
        }
        for ptr in ptrs {
            unsafe {
                cudaFree(*ptr as *mut std::ffi::c_void);
            }
        }
    }

    /// [#758] the outcome of the last registration release or reaping pass.
    /// Test-only (#238): the two device acceptance gates assert the named
    /// `Retained` outcome and the final `Freed` reaping.
    #[cfg(test)]
    pub(crate) fn last_release(&self) -> Option<ReleaseOutcome> {
        *self.last_release.lock().unwrap()
    }

    /// [#758] how many captured graphs currently hold a retention lease.
    /// Test-only (#238): the negative gate asserts a capture is live before it
    /// drops the model.
    #[cfg(test)]
    pub(crate) fn live_captures(&self) -> usize {
        self.live_captures.load(std::sync::atomic::Ordering::SeqCst)
    }

    /// Issue #188 probe, **test-only**: the pre-#188 registration H2D path, a
    /// blocking `cudaMemcpy` (which the driver issues on the legacy default
    /// stream, not on any explicit stream). Kept so the acceptance probe can
    /// measure the capture mode against the *historical* setup — a shared
    /// capture stream plus this copy — instead of against a setup the fix
    /// already changed, which would make the mode look irrelevant.
    #[cfg(test)]
    pub fn register_weight_blocking_legacy(&self, name: &str, data: &[u8]) {
        if data.is_empty() {
            return;
        }
        let mut ptr: *mut std::ffi::c_void = std::ptr::null_mut();
        let err = unsafe { cudaMalloc(&mut ptr, data.len()) };
        if err != 0 || ptr.is_null() {
            panic!("probe: cudaMalloc({}) failed ({err})", data.len());
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
            unsafe { cudaFree(ptr) };
            panic!("probe: blocking cudaMemcpy failed ({err})");
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
        // r59 rider: track the max nchunk (= id/32) for the pre-warm scratch
        // sizing (see prewarm_prefill).
        self.max_nchunk
            .fetch_max(id / 32, std::sync::atomic::Ordering::Relaxed);
        // ANCHOR: q6k_padded_repack
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
        // ANCHOR_END: q6k_padded_repack
        // D4-4 L1: dense split-plane (dpl) decode sibling plane — per row
        // [ql nbe*128][qh nbe*64][sc nbe*16][d nbe*2] at a 16B-aligned row
        // stride, 210B of content per 256-elem block (no 224B pad sectors).
        // The decode MMVQ reads it when present; per-unit values and the
        // accumulation order are unchanged, so outputs stay bitwise-identical
        // (probe /tmp/d4/probe_l1_dpl.cu). Requires id % 256 == 0 (exact
        // nbe, like the W_exp plane); MINFER_Q6K_DPL=0 skips the build and
        // decode keeps the padded kernels.
        if crate::config::get().q6k_dpl && id % 256 == 0 {
            let nbe = id / 256;
            let dpl_row = (nbe * 210 + 15) & !15usize;
            let raw_row = nbe * 210;
            let mut dpl = vec![0u8; od * dpl_row];
            for r in 0..od {
                let src = &data[r * raw_row..(r + 1) * raw_row];
                let dst = &mut dpl[r * dpl_row..(r + 1) * dpl_row];
                let (ql, rest) = dst.split_at_mut(nbe * 128);
                let (qh, rest) = rest.split_at_mut(nbe * 64);
                let (sc, dd) = rest.split_at_mut(nbe * 16);
                for ib in 0..nbe {
                    let blk = &src[ib * 210..(ib + 1) * 210];
                    ql[ib * 128..(ib + 1) * 128].copy_from_slice(&blk[..128]);
                    qh[ib * 64..(ib + 1) * 64].copy_from_slice(&blk[128..192]);
                    sc[ib * 16..(ib + 1) * 16].copy_from_slice(&blk[192..208]);
                    dd[ib * 2..(ib + 1) * 2].copy_from_slice(&blk[208..210]);
                }
            }
            let dpl_name = format!("{name}__dpl");
            self.register_weight(&dpl_name, &dpl);
            if let (Some(wp), Some(ep)) =
                (self.get_weight_ptr(name), self.get_weight_ptr(&dpl_name))
            {
                if !wp.is_null() && !ep.is_null() {
                    self.q6k_dpl
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(ep));
                }
            }
        }
        // r53: pre-expand B into the dense centered-int8 plane (P6 r44) so the
        // NB-BT q6_K kernel's B staging is a pure cp.async copy. Ships with the
        // MINFER_MMQ_Q6K_NB gate (the NB-BT kernel is its only consumer; the
        // A/B baseline is the unchanged env set) and requires id % 256 == 0
        // (the kernel's own launch gate, which also keeps the dense index
        // 16B-aligned). Dense bytes = od * id — ~2.4 GiB total on 7B q4_k_m
        // (ffn_down 67.9 MB x 28 + output 545 MB + attn_v 1.8 MB x 28).
        // r54: MINFER_MMQ_Q6K_EXP decouples the plane from the kernel gate —
        // unset/"1" keeps the r53 default (build it), explicit "0" skips the
        // build entirely (registration early-returns; device memory stays at
        // the pre-r53 level) so dispatch map-misses into the EXP=false r41
        // in-kernel expand. ANDed with Q6K_NB: EXP only matters when the NB
        // kernel is live (r60: Q6K_NB is default-on — "0" opts out of the
        // kernel AND both planes).
        if crate::config::get().mmq_q6k_nb && crate::config::get().mmq_q6k_exp && id % 256 == 0 {
            self.register_weight_q6k_exp(name, &padded, od, id);
            // r56 (Session E item 2b): the dsc f32 plane rides the same gate
            // (+ od % 2 == 0: the kernel stages row PAIRS per 16-B cp.async
            // chunk and zero-fills whole pairs, so an odd od row would lose
            // its scale — such tensors keep the scalar path via map miss).
            if od % 2 == 0 {
                self.register_weight_q6k_dsc(name, &padded, od, id);
            }
        }
    }

    /// r53: build + upload the dense pre-expanded B plane for one padded q6_K
    /// tensor and map it from the padded weight's device pointer. Called from
    /// [`Self::register_weight_q6k_padded`] under the `MINFER_MMQ_Q6K_NB`
    /// (r60: default-on) + `MINFER_MMQ_Q6K_EXP != "0"` (r54) +
    /// `id % 256 == 0` gate; also `pub`
    /// for the gate-on byte-exactness test. An alloc/upload failure leaves the
    /// map empty: the kernel falls back to the r41 in-kernel expand with a
    /// once-per-process loud eprintln.
    pub fn register_weight_q6k_exp(&self, name: &str, padded: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name: a same-name different-shape
        // re-registration can never collide with (and silently reuse) a stale
        // plane of the same byte size but a different od/id layout.
        let exp_name = format!("{name}__exp{od}x{id}");
        // T2: budget-gate BEFORE the host expansion (review NIT #6) — the
        // dense plane is od*id bytes; a tripped device must not pay the
        // full host build only to discard it.
        if !self.plane_budget_ok(od * id) {
            return;
        }
        let exp = Self::expand_q6k_dense(padded, od, id);
        debug_assert_eq!(exp.len(), od * id);
        self.register_weight(&exp_name, &exp);
        // the MAP is keyed by the PADDED weight's device pointer (what
        // prefill_mmq holds); the value is the W_exp plane's pointer
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(ep) = self.get_weight_ptr(&exp_name) {
                if !wp.is_null() && !ep.is_null() {
                    self.q6k_exp
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(ep));
                    return;
                }
            }
        }
        if !self
            .q6k_exp_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q6_K W_exp pre-expand unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt_q6k falls back to the \
                 r41 in-kernel expand"
            );
        }
    }

    /// r56 (Session E item 2b): build + upload the precomputed dsc f32-pair
    /// plane for one padded q6_K tensor and map it from the padded weight's
    /// device pointer. Called from [`Self::register_weight_q6k_padded`] under
    /// the same gate as `register_weight_q6k_exp` (+ `od % 2 == 0`). An
    /// alloc/upload failure leaves the map empty: the kernel falls back to the
    /// r41 scalar dsc path with a once-per-process loud eprintln.
    pub fn register_weight_q6k_dsc(&self, name: &str, padded: &[u8], od: usize, id: usize) {
        // geometry-encoded sibling name (same rationale as the W_exp name).
        let dsc_name = format!("{name}__dsc{od}x{id}");
        // T2: budget-gate BEFORE the host expansion (review NIT #6) — the
        // dsc plane is (id/32)*od*8 bytes.
        if !self.plane_budget_ok((id / 32) * od * 8) {
            return;
        }
        let dsc = Self::expand_q6k_dsc(padded, od, id);
        debug_assert_eq!(dsc.len(), (id / 32) * od * 8);
        self.register_weight(&dsc_name, &dsc);
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(dp) = self.get_weight_ptr(&dsc_name) {
                if !wp.is_null() && !dp.is_null() {
                    self.q6k_dsc
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(dp));
                    return;
                }
            }
        }
        if !self
            .q6k_dsc_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q6_K W_dsc plane unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt_q6k keeps the r41 \
                 scalar dsc path"
            );
        }
    }

    /// r56 (Session E item 2b): precomputed dsc pairs of one padded q6_K
    /// tensor. Output: `nchunk * od * 8` bytes, `out[(c*od + j)*8..+8]` =
    /// float2(d*sc[2(c&7)], d*sc[2(c&7)+1]) — chunk-major so the kernel's
    /// per-kt staging (rows j0..j0+MMQ_NBJ of chunk c0+kd contiguous) is a
    /// pure 16-B cp.async stream. Bit-identical to the in-kernel r41 scalar
    /// computation: exact f16->f32 (half::f16, = __half2float), exact
    /// i8->f32, one IEEE f32 multiply, no FMA contraction on either side.
    pub fn expand_q6k_dsc(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id / 256;
        let row_len = nbe * Q6KPB;
        let mut out = vec![0u8; (id / 32) * od * 8];
        for j in 0..od {
            let prow = &padded[j * row_len..(j + 1) * row_len];
            for sb in 0..nbe {
                let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
                let d_bits = u16::from_le_bytes([blk[208], blk[209]]);
                let d = half::f16::from_bits(d_bits).to_f32();
                for cc in 0..8usize {
                    let s0 = 2 * cc;
                    let sc0 = blk[192 + s0] as i8 as f32;
                    let sc1 = blk[192 + s0 + 1] as i8 as f32;
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    out[idx..idx + 4].copy_from_slice(&(d * sc0).to_bits().to_le_bytes());
                    out[idx + 4..idx + 8].copy_from_slice(&(d * sc1).to_bits().to_le_bytes());
                }
            }
        }
        out
    }

    /// r59 (Session F item 1), #165: build + upload the precomputed dsc f32-pair
    /// plane for one RAW q4_K tensor and map it from the raw weight's device
    /// pointer. Called from the qwen2 loader under the NB-BT gate set
    /// (MINFER_MMQ_RAW_NB=1 + MINFER_MMQ_A_TRANSPOSE=1 + MINFER_MMQ_Q4K_DSC
    /// != "0") + [`crate::q4k_dsc::q4k_dsc_plane_admitted`] (type q4_K + `id % 256 == 0`) +
    /// `od % 2 == 0`. An alloc/upload failure leaves the map empty: the kernel
    /// falls back to the in-kernel scalar decode (DSC=false) with a
    /// once-per-process loud eprintln. A payload that is not exactly
    /// [`crate::q4k_dsc::q4k_dsc_payload_bytes`] is refused **before** the budget query and
    /// before the host expansion: its bytes are another type's (the #165
    /// misread) or too few for the row arithmetic (the #165 latent OOB read).
    pub fn register_weight_q4k_dsc(&self, name: &str, raw: &[u8], od: usize, id: usize) {
        // #165: the payload contract first. It costs nothing and is the only check that
        // can refuse a smaller-ratio payload before `expand_q4k_dsc` would index past
        // `raw`; `q4k_dsc_plane_admitted` in the loader is the same rule plus the type.
        if !q4k_dsc_payload_ok(raw.len(), od, id) {
            return;
        }
        // geometry-encoded sibling name (same rationale as the W_exp name).
        let dsc_name = format!("{name}__q4dsc{od}x{id}");
        // T2: budget-gate BEFORE the host expansion (review NIT #6) — the
        // dsc plane is (id/32)*od*8 bytes.
        if !self.plane_budget_ok((id / 32) * od * 8) {
            return;
        }
        let Some(dsc) = Self::expand_q4k_dsc(raw, od, id) else {
            return;
        };
        debug_assert_eq!(dsc.len(), (id / 32) * od * 8);
        self.register_weight(&dsc_name, &dsc);
        if let Some(wp) = self.get_weight_ptr(name) {
            if let Some(dp) = self.get_weight_ptr(&dsc_name) {
                if !wp.is_null() && !dp.is_null() {
                    self.q4k_dsc
                        .lock()
                        .unwrap()
                        .insert(wp as usize, CudaPtr(dp));
                    return;
                }
            }
        }
        if !self
            .q4k_dsc_warned
            .swap(true, std::sync::atomic::Ordering::Relaxed)
        {
            eprintln!(
                "minfer/cuda: q4_K W_dsc plane unavailable for '{name}' \
                 (alloc/upload failed) - mmq_raw_nb_bt keeps the in-kernel \
                 scalar dsc decode"
            );
        }
    }

    /// r59 (Session F item 1): precomputed dsc pairs of one RAW q4_K tensor.
    /// Output: `(id / 32) * od * 8` bytes, `out[(c*od + j)*8..+8]` =
    /// float2(d*sc[c&7], -(dmin*m[c&7])) — chunk-major so the kernel's
    /// per-kt staging (rows j0..j0+MMQ_NBJ of chunk c0+kd contiguous) is a
    /// pure 16-B cp.async stream. Bit-identical to the in-kernel scalar
    /// computation: exact f16->f32 (half::f16 = __half2float), exact
    /// u8->f32, ONE IEEE f32 multiply, exact negation, no FMA contraction
    /// on either side.
    ///
    /// #165: **`None` when `raw` is not exactly [`crate::q4k_dsc::q4k_dsc_payload_bytes`] for the
    /// geometry** — `raw`'s rows are then not 144-byte q4_K super-block rows, and the
    /// indexing below (`raw[j * row_len..(j + 1) * row_len]`) would either misread
    /// another type's bytes or run past the slice. The caller
    /// ([`Self::register_weight_q4k_dsc`]) registers nothing in that case.
    pub fn expand_q4k_dsc(raw: &[u8], od: usize, id: usize) -> Option<Vec<u8>> {
        if !q4k_dsc_payload_ok(raw.len(), od, id) {
            return None;
        }
        const Q4KB: usize = 144;
        let nsb = id / 256;
        let nchunk = id / 32;
        let row_len = nsb * Q4KB;
        let mut out = vec![0u8; nchunk * od * 8];
        for j in 0..od {
            let prow = &raw[j * row_len..(j + 1) * row_len];
            for sb in 0..nsb {
                let blk = &prow[sb * Q4KB..sb * Q4KB + Q4KB];
                let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
                let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
                let sc = &blk[4..16]; // 12 packed 6-bit scales+mins
                for cc in 0..8usize {
                    // host mirror of the device get_scale_min_k4 (src/cuda/kernels/common.cuh)
                    let (s, m) = if cc < 4 {
                        (sc[cc] & 63, sc[cc + 4] & 63)
                    } else {
                        (
                            (sc[cc + 4] & 0xF) | ((sc[cc - 4] >> 6) << 4),
                            (sc[cc + 4] >> 4) | ((sc[cc] >> 6) << 4),
                        )
                    };
                    let idx = ((sb * 8 + cc) * od + j) * 8;
                    out[idx..idx + 4].copy_from_slice(&(d * (s as f32)).to_bits().to_le_bytes());
                    out[idx + 4..idx + 8]
                        .copy_from_slice(&(-(dmin * (m as f32))).to_bits().to_le_bytes());
                }
            }
        }
        Some(out)
    }

    /// r59 (Session F riders, r57 items 4+5): move the one-time first-launch
    /// costs out of the measured prefill window. Called once at the end of
    /// model weight registration.
    /// (1) kernel module pre-load — cudaFuncGetAttributes over the
    ///     MMQ/FA/fused launch set forces the fatbin to load now instead of
    ///     at the first dispatch (r58 CUPTI: ~3 ms host stalls bracketing
    ///     the first mode-2 swiglu / first bt matmul);
    /// (2) pinned D2H readback pre-grow — the grow-on-demand 4 MB
    ///     cudaHostAlloc was the "0.78 ms tail malloc" at the n_out=1
    ///     logits readback;
    /// (3) MmqCache scratch pre-grow — buf_q8_prefill / buf_qa8_t /
    ///     buf_sda_t sized for a nominal 4096-token prefill (the default
    ///     n_ctx) at the max registered nchunk, so the first prefill's
    ///     get_or_grow hits instead of cudaMalloc-ing ~150 MB mid-window
    ///     (a larger prompt grows in-window exactly as before). MMQ-gated:
    ///     the planes are dead weight when the MMQ path is off.
    pub fn prewarm_prefill(&self) {
        unsafe {
            minfer_prewarm_kernels();
        }
        // (2) pinned readback pre-grow (same 4 MB floor as
        // copy_from_device_pinned; failure is non-fatal — the first readback
        // retries the alloc there).
        {
            let mut guard = self.readback.lock().unwrap();
            if guard.is_none() {
                let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
                let err = unsafe { cudaHostAlloc(&mut p, 4 * 1024 * 1024, 0) };
                if err == 0 && !p.is_null() {
                    *guard = Some(PinnedBuf {
                        ptr: p as *mut u8,
                        bytes: 4 * 1024 * 1024,
                    });
                }
            }
        }
        // (3) MmqCache scratch pre-grow
        if crate::config::get().mmq {
            let max_nchunk = self.max_nchunk.load(std::sync::atomic::Ordering::Relaxed);
            if max_nchunk > 0 {
                const NT_PREWARM: usize = 4096; // default n_ctx
                let ntb = NT_PREWARM.div_ceil(64);
                Self::get_or_grow(&self.buf_q8_prefill, NT_PREWARM * max_nchunk * 40);
                Self::get_or_grow(&self.buf_qa8_t, ntb * max_nchunk * 2048);
                Self::get_or_grow(&self.buf_sda_t, ntb * max_nchunk * 256);
            }
        }
    }

    /// r53: dense centered-int8 pre-expansion of one padded q6_K tensor — the
    /// host mirror of the device `expand_q6_elem` (P6 r44 / MMQ-analysis
    /// §11.24). Output: `od * id` bytes, `out[j * id + sb * 256 + e]` =
    /// super-block element e of row j — the exact tile the kernel's staging
    /// used to recomb. Requires `id % 256 == 0` (the NB-BT launch gate).
    /// Two output elements per (ql, qh) byte pair: e = it*128+r and
    /// e = it*128+r+64 share ql[it*64+r] (nibble shifts 0/4) and
    /// qh[it*32 + (r&31)] (2-bit-field shifts 2*(r>>5) / 2*((r>>5)+2)).
    pub fn expand_q6k_dense(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
        const Q6KB: usize = 210;
        const Q6KPB: usize = 224;
        let nbe = id / 256;
        let row_len = nbe * Q6KPB;
        let mut out = vec![0u8; od * id];
        for j in 0..od {
            let prow = &padded[j * row_len..(j + 1) * row_len];
            let orow = &mut out[j * id..(j + 1) * id];
            for sb in 0..nbe {
                let blk = &prow[sb * Q6KPB..sb * Q6KPB + Q6KB];
                let (ql, qh) = blk.split_at(128);
                let obase = &mut orow[sb * 256..sb * 256 + 256];
                for it in 0..2usize {
                    for r in 0..64usize {
                        let qlb = ql[it * 64 + r];
                        let qhb = qh[it * 32 + (r & 31)];
                        let s0 = (r >> 5) * 2;
                        let e0 = it * 128 + r;
                        obase[e0] = ((qlb & 0xF) | (((qhb >> s0) & 3) << 4)).wrapping_sub(32);
                        obase[e0 + 64] =
                            (((qlb >> 4) & 0xF) | (((qhb >> (s0 + 4)) & 3) << 4)).wrapping_sub(32);
                    }
                }
            }
        }
        out
    }

    /// Whether `name` was registered in the padded Q6_K layout.
    pub fn is_weight_padded(&self, name: &str) -> bool {
        self.padded_weights.lock().unwrap().contains_key(name)
    }

    /// #165: the q4_K `W_dsc` planes currently in the registry (`*__q4dsc*`) as
    /// `(name, bytes)`. The plane's only consumer is the NB-BT q4_K kernel, so this is
    /// the exact set of device buffers a load that is *not* q4_K must leave empty — the
    /// registry query the #165 gate and its before/after accounting read by name.
    /// Test-only (#238): driven by `graph::cuda_backend::tests::weights::cuda_q4dsc_plane_is_q4k_only and tooling::tests::f167_qwen3_q4k_registers_the_dsc_plane_exactly`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn q4dsc_planes(&self) -> Vec<(String, usize)> {
        self.weights
            .lock()
            .unwrap()
            .iter()
            .filter(|(n, _)| n.contains("__q4dsc"))
            .map(|(n, (_, size))| (n.clone(), *size))
            .collect()
    }

    pub fn get_weight_ptr(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        self.weights.lock().unwrap().get(name).map(|(cp, _)| cp.0)
    }

    /// The registered byte length of the device weight `name` (#169).
    ///
    /// A Q6_K entry registered through `register_weight_q6k_padded` lives on the
    /// device with a larger stride, so it reports its **original raw** length —
    /// the same convention `has_weight_of_size` uses. `None` when the name is
    /// not registered at all. The norm path reads this to refuse a weight that
    /// is not the f32 it will index as (the f16-norm hazard of #169).
    pub fn weight_size(&self, name: &str) -> Option<usize> {
        if let Some(&raw) = self.padded_weights.lock().unwrap().get(name) {
            return Some(raw);
        }
        self.weights
            .lock()
            .unwrap()
            .get(name)
            .map(|(_, size)| *size)
    }

    /// #167: whether the NB-BT q4_K kernel would find a `W_dsc` plane for the raw weight
    /// `name`. The kernel keys its map on the **raw weight's device pointer** (`q4k_dsc`,
    /// read at `mmq_raw_nb_bt`'s launch site: a null `w_dsc` selects the in-kernel scalar
    /// decode), so this performs exactly that lookup rather than looking the sibling name
    /// up in the registry — which is the part a name-only assertion cannot see. `None`
    /// when the weight is not registered at all.
    /// Test-only (#238): driven by `tooling::tests::f167_qwen3_q4k_registers_the_dsc_plane_exactly`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn q4dsc_plane_for(&self, name: &str) -> Option<*mut std::ffi::c_void> {
        let wp = self.get_weight_ptr(name)?;
        if wp.is_null() {
            return None;
        }
        self.q4k_dsc
            .lock()
            .unwrap()
            .get(&(wp as usize))
            .map(|cp| cp.0)
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

    /// doc 104: register a Q8_0 tensor ALSO as the p32 split-plane layout —
    /// payload plane (32 B/block, 16B-aligned for uint4 loads) + dense d
    /// plane (2 B/block). The raw registration stays untouched (the f32
    /// fallback and every existing kernel keep reading it); the planes are
    /// extra device memory (~+94% of the tensor) keyed by the original
    /// weight pointer for the decode dispatch. Host repack at load, like
    /// the q6_K padded precedent — no capture-window hazard. Registered
    /// under private keys (`\u{1}`-prefixed suffixes) so they can never
    /// collide with a real tensor name (the doc 102 lesson).
    /// T2 (plan §6.3): free-VRAM budget gate for OPTIONAL weight planes
    /// (q8_0 p32 pairs, q4_K/q6_K dsc + dense expansions). Requires free >
    /// extra + extra/4 — the headroom covers the activation/KV working set
    /// that lands after registration. On GB10's 128 GB this always passes
    /// (zero change); on 8 GB unified-memory devices (Orin Nano) it
    /// self-disables the planes and the raw paths serve (raw lookup miss).
    /// Best-effort by design — every consumer falls back to its raw path on a
    /// map miss — so a failed query keeps the planes **off** (the conservative
    /// branch, unchanged) but no longer silently (issue #122): the discarded
    /// return code used to make a broken query look like "not enough memory".
    fn plane_budget_ok(&self, extra_bytes: usize) -> bool {
        let mut free: usize = 0;
        let mut total: usize = 0;
        let rc = unsafe { cudaMemGetInfo(&mut free, &mut total) };
        if rc != 0 {
            static WARNED: std::sync::OnceLock<()> = std::sync::OnceLock::new();
            if WARNED.set(()).is_ok() {
                eprintln!(
                    "CUDA: optional weight-plane budget query failed with {} (code {}); \
                     keeping the optional planes off",
                    cuda_error_name(rc),
                    rc
                );
            }
            return false;
        }
        free > extra_bytes + extra_bytes / 4
    }

    pub fn register_weight_q80_p32(&self, name: &str, data: &[u8], od: usize, id: usize) {
        if od == 0 || id == 0 || id % 32 != 0 || id < 2048 || crate::config::get().no_q80_p32 {
            return;
        }
        let nb = id / 32;
        if data.len() < od * nb * 34 {
            return;
        }
        // T2: the p32 pair adds od*nb*34 B (pp 32 + pd 2) = +100% of the raw
        // weight — budget-gate before building/uploading (plan §6.3).
        if !self.plane_budget_ok(od * nb * 34) {
            return;
        }
        let mut pp = vec![0u8; od * nb * 32];
        let mut pd = vec![0u8; od * nb * 2];
        for r in 0..od {
            let row = r * nb;
            for b in 0..nb {
                let src = (row + b) * 34;
                pp[(row + b) * 32..(row + b) * 32 + 32].copy_from_slice(&data[src + 2..src + 34]);
                pd[(row + b) * 2..(row + b) * 2 + 2].copy_from_slice(&data[src..src + 2]);
            }
        }
        let orig = {
            let w = self.weights.lock().unwrap();
            match w.get(name) {
                Some((p, sz)) if *sz == data.len() => p.0 as usize,
                _ => return,
            }
        };
        let pname = format!("{name}\u{1}p32");
        let dname = format!("{name}\u{1}p32d");
        self.register_weight(&pname, &pp);
        self.register_weight(&dname, &pd);
        let w = self.weights.lock().unwrap();
        if let (Some((ppp, _)), Some((pdp, _))) = (w.get(&pname), w.get(&dname)) {
            self.q80_p32
                .lock()
                .unwrap()
                .insert(orig, (ppp.0 as usize, pdp.0 as usize));
        }
    }

    /// doc 104: p32 plane pair for a registered q8_0 weight, if built.
    pub fn q80_p32_planes(&self, wptr: usize) -> Option<(*const u8, *const u8)> {
        let map = self.q80_p32.lock().unwrap();
        map.get(&wptr)
            .map(|(pp, pd)| (*pp as *const u8, *pd as *const u8))
    }
}
