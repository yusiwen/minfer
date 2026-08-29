//! CUDA graph backend (Phase 7).
//!
//! Wraps the [`crate::cuda::CudaState`] singleton in the graph [`Backend`]
//! trait contract (`src/graph/backend.rs`), mirroring `metal_backend.rs` where
//! the mechanics allow: a device buffer pool with a byte-length free list,
//! name → device-pointer weight resolution, sync H2D/D2H host transfers, and
//! per-op kernel dispatch on the shared stream. Design + rollout:
//! `docs/CUDA-BACKEND-PLAN.md`.

use super::backend::Backend;
use super::ops::{FusedOp, NodeMeta, Op};
use super::{CNode, DType};
use crate::vec_ops::RopeStyle;

struct CudaBuf {
    ptr: *mut std::ffi::c_void,
    bytes: usize,
}

pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,
    pool: Vec<CudaBuf>,
    free: Vec<usize>,
    /// Bumped on every pool allocation (fresh or free-list reuse). A captured
    /// CUDA Graph (7d) is only valid while the node → device-pointer mapping
    /// is unchanged, and any alloc_buffer() call may change it.
    pool_gen: u64,
    /// Device scratch holding raw-int32 positions decoded from the f32-bits
    /// I32 input buffers (grown on demand; freed in Drop alongside the pool).
    pos_scratch: *mut std::ffi::c_void,
    pos_scratch_bytes: usize,
}

// SAFETY: CudaBuf holds raw device pointers that are only dereferenced by the
// GPU; the backend is only mutated through &mut self (allocator/scheduler).
// Same reasoning as metal_backend.rs's unsafe Send/Sync.
unsafe impl Send for CudaBackend {}
unsafe impl Sync for CudaBackend {}

impl CudaBackend {
    /// `None` when CUDA is unavailable (no device, or disabled via
    /// `MINFER_DISABLE_CUDA` — both handled by `CudaState::try_new`).
    /// (Called by GraphAllocator::enable_cuda; bin builds see it as dead until
    /// the Phase 7c model wiring, tests use it meanwhile.)
    #[allow(dead_code)]
    pub fn new() -> Option<Self> {
        let state = crate::cuda::CudaState::get()?;
        Some(Self {
            state,
            pool: Vec::new(),
            free: Vec::new(),
            pool_gen: 0,
            pos_scratch: std::ptr::null_mut(),
            pos_scratch_bytes: 0,
        })
    }

    /// Pool generation counter (CUDA Graph replay invalidation, Phase 7d).
    #[allow(dead_code)]
    pub fn pool_gen(&self) -> u64 {
        self.pool_gen
    }

    fn ptr_of(&self, id: usize) -> Result<*mut std::ffi::c_void, String> {
        match self.pool.get(id) {
            Some(b) if !b.ptr.is_null() => Ok(b.ptr),
            Some(_) => Err(format!("cuda: buffer {id} has a null device pointer")),
            None => Err(format!("cuda: unknown buffer id {id}")),
        }
    }

    /// Sync D2H readback of a pool buffer as owned data. This is the alloc.rs
    /// `copy_to_cpu` CUDA arm (the trait's `read_host` cannot return a borrowed
    /// slice for a staged transfer). Explicit `sync()` first: never rely on the
    /// legacy-default-stream's implicit synchronization with blocking streams.
    pub fn copy_to_host(&self, id: usize) -> Option<Vec<f32>> {
        let b = self.pool.get(id)?;
        if b.ptr.is_null() || b.bytes == 0 {
            return None;
        }
        let mut out = vec![0f32; b.bytes / 4];
        self.state.sync();
        let dst = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, b.bytes) };
        self.state.copy_from_device(b.ptr, dst);
        Some(out)
    }
}

impl Drop for CudaBackend {
    fn drop(&mut self) {
        // Pools live as long as the backend (inside GraphCache); only real
        // teardown frees device memory. free_buffer() only recycles.
        for b in &self.pool {
            Self::state_free(b.ptr);
        }
        self.pool.clear();
        self.free.clear();
        if !self.pos_scratch.is_null() {
            Self::state_free(self.pos_scratch);
            self.pos_scratch = std::ptr::null_mut();
            self.pos_scratch_bytes = 0;
        }
    }
}

impl CudaBackend {
    fn state_free(ptr: *mut std::ffi::c_void) {
        <crate::cuda::CudaState>::cuda_free(ptr);
    }

    fn elems(&self, id: usize) -> usize {
        self.pool[id].bytes / 4
    }

    fn copy_d2d(&self, src: usize, dst: usize) -> Result<(), String> {
        let (s, d) = (self.ptr_of(src)?, self.ptr_of(dst)?);
        let (sb, db) = (self.pool[src].bytes, self.pool[dst].bytes);
        if sb != db {
            return Err(format!(
                "cuda: device copy size mismatch src {sb} vs dst {db} bytes"
            ));
        }
        self.state.copy_device_to_device(s, d, db);
        Ok(())
    }

    /// Decode an I32 input buffer (f32::from_bits bit patterns, alloc.rs
    /// fill_input_i32) into raw int32 on the device. The rope/store/attention
    /// kernels read `const int* positions`; one tiny elementwise pass keeps
    /// the whole path on-device — no host sync, and the pointer stays stable
    /// across steps (a precondition for CUDA Graph replay in Phase 7d).
    fn positions_i32(&mut self, id: usize) -> Result<*mut std::ffi::c_void, String> {
        let src = self.ptr_of(id)?;
        let bytes = self.pool[id].bytes;
        if self.pos_scratch_bytes < bytes {
            if !self.pos_scratch.is_null() {
                Self::state_free(self.pos_scratch);
            }
            let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
            if ptr.is_null() {
                self.pos_scratch_bytes = 0;
                return Err("cuda: positions scratch allocation failed".to_string());
            }
            self.pos_scratch = ptr;
            self.pos_scratch_bytes = bytes;
        }
        self.state.bits_to_i32(src, self.pos_scratch, bytes / 4);
        Ok(self.pos_scratch)
    }

    /// Resolve a NormMeta weight by name on the CUDA registry. Unlike Metal
    /// (which silently degrades to a weightless norm when the weight is not on
    /// the backend), a declared-but-missing weight is an invariant violation
    /// here and returns Err (docs/GPU_SAFETY.md).
    fn norm_weight(&self, node: &CNode) -> Result<*mut std::ffi::c_void, String> {
        let name = match &node.meta {
            NodeMeta::Norm(m) => m.weight_name.as_deref(),
            other => {
                return Err(format!(
                    "cuda: {} node missing NormMeta: {other:?}",
                    node.name
                ))
            }
        };
        let Some(name) = name else {
            return Err(format!(
                "cuda: {} has no norm weight (the CUDA rms_norm kernel requires one)",
                node.name
            ));
        };
        self.state.get_weight_ptr(name).ok_or_else(|| {
            format!(
                "cuda: weight '{name}' not registered on CUDA ({})",
                node.name
            )
        })
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    /// v1 capability matrix (docs/CUDA-BACKEND-PLAN.md §4.3): the full
    /// per-layer chain runs on CUDA; Embed/GetRows, Scale, Softmax and the
    /// fused decode ops have no kernels and stay on the CPU backend. RoPE is
    /// gated to the neox (non-interleaved) layout — the only style the
    /// supported architectures emit.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        if dtype != DType::F32 {
            return false;
        }
        match op {
            Op::Input
            | Op::Add
            | Op::Mul
            | Op::Silu
            | Op::SwiGLU
            | Op::RmsNorm { .. }
            | Op::QkNorm { .. }
            | Op::MatMul { .. }
            | Op::Attn { .. }
            | Op::KvcacheStore { .. }
            | Op::KvcacheLoad { .. }
            | Op::View { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. } => true,
            // MatMul ttype gating happens at the model level (weights must all
            // be registered on CUDA — same all-or-nothing rule as Metal).
            Op::RoPE { style } => matches!(style, RopeStyle::NonInterleaved),
            _ => false,
        }
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        matches!(fused, FusedOp::SwiGLU)
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        let bytes = size * 4;
        if let Some(pos) = self
            .free
            .iter()
            .position(|&id| self.pool[id].bytes == bytes)
        {
            let id = self.free.remove(pos);
            self.pool_gen += 1;
            return id;
        }
        let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
        self.pool.push(CudaBuf { ptr, bytes });
        self.pool_gen += 1;
        self.pool.len() - 1
    }

    fn free_buffer(&mut self, id: usize) {
        // Recycle, never cudaFree here: persistent KV regions survive rebuilds
        // and the pool keeps freed device memory for reuse (CPU/Metal alike).
        if !self.free.contains(&id) {
            self.free.push(id);
        }
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // bypass the free list entirely (see Backend::alloc_fresh): the ids in
        // it are still referenced by node_to_buf and physically live during
        // the execute that follows
        let bytes = size * 4;
        let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes);
        self.pool.push(CudaBuf { ptr, bytes });
        self.pool_gen += 1;
        self.pool.len() - 1
    }

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[usize],
        out_buf: usize,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        match &node.op {
            // Inputs are host-filled by the allocator; KvcacheLoad is a view
            // of the persistent K region (out_buf IS the region — no kernel).
            Op::Input | Op::KvcacheLoad { .. } => Ok(()),
            // Layout-only nodes: identity copy of the source buffer (same
            // semantics as cpu_backend's View/Reshape/Permute handling).
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                let src = *in_bufs
                    .first()
                    .ok_or_else(|| format!("cuda: {} without source buffer", node.name))?;
                self.copy_d2d(src, out_buf)
            }

            Op::Add => {
                let n = self.elems(out_buf);
                if self.elems(in_bufs[0]) != n || self.elems(in_bufs[1]) != n {
                    return Err(format!("cuda: {}: add input size mismatch", node.name));
                }
                self.state.add_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(in_bufs[1])?,
                    self.ptr_of(out_buf)?,
                    n,
                );
                Ok(())
            }
            Op::Mul => {
                let n = self.elems(out_buf);
                if self.elems(in_bufs[0]) != n || self.elems(in_bufs[1]) != n {
                    return Err(format!("cuda: {}: mul input size mismatch", node.name));
                }
                self.state.mul_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(in_bufs[1])?,
                    self.ptr_of(out_buf)?,
                    n,
                );
                Ok(())
            }
            // In-place op (alias rule, graph rules §5): stage via D2D copy when
            // the allocator did not alias the input, then run on the output.
            Op::Silu => {
                if in_bufs[0] != out_buf {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                self.state
                    .silu_f32(self.ptr_of(out_buf)?, self.elems(out_buf));
                Ok(())
            }
            Op::SwiGLU => {
                let n = self.elems(out_buf);
                if self.elems(in_bufs[0]) != n || self.elems(in_bufs[1]) != n {
                    return Err(format!("cuda: {}: swiglu input size mismatch", node.name));
                }
                self.state.swiglu_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(in_bufs[1])?,
                    self.ptr_of(out_buf)?,
                    n,
                );
                Ok(())
            }

            Op::RmsNorm { eps } => {
                let wptr = self.norm_weight(node)?;
                let d = node.out_shape[0];
                if d % 4 != 0 || d == 0 || self.elems(out_buf) % d != 0 {
                    return Err(format!(
                        "cuda: {}: rms_norm dim {d} must be a nonzero multiple of 4 (float4 kernel)",
                        node.name
                    ));
                }
                let n = self.elems(out_buf) / d;
                self.state.rms_norm(
                    self.ptr_of(in_bufs[0])?,
                    Some(wptr),
                    self.ptr_of(out_buf)?,
                    d,
                    n,
                    *eps,
                );
                Ok(())
            }
            // Per-head RMSNorm: the flat [nt*nh*hd] buffer is a contiguous
            // [nt*nh, hd] row matrix (t*(nh*hd) + h*hd == (t*nh+h)*hd), so the
            // same rms_norm kernel runs with d = hd (weight shared per head).
            Op::QkNorm { hd, eps, .. } => {
                let wptr = self.norm_weight(node)?;
                let d = *hd;
                if d % 4 != 0 || d == 0 || self.elems(out_buf) % d != 0 {
                    return Err(format!(
                        "cuda: {}: qk_norm head dim {d} must be a nonzero multiple of 4 (float4 kernel)",
                        node.name
                    ));
                }
                let n = self.elems(out_buf) / d;
                self.state.rms_norm(
                    self.ptr_of(in_bufs[0])?,
                    Some(wptr),
                    self.ptr_of(out_buf)?,
                    d,
                    n,
                    *eps,
                );
                Ok(())
            }

            Op::MatMul { transpose_b } => {
                if *transpose_b {
                    return Err(format!(
                        "cuda: {}: transposed matmul not supported",
                        node.name
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::MatMul(m) => m,
                    other => {
                        return Err(format!("matmul node missing MatMulMeta: {other:?}"));
                    }
                };
                let wptr = self
                    .state
                    .get_weight_ptr(&meta.weight_name)
                    .ok_or_else(|| {
                        format!(
                            "cuda: weight '{}' not registered on CUDA ({})",
                            meta.weight_name, node.name
                        )
                    })?;
                let (od, id) = (meta.out_dim, meta.in_dim);
                let nt = node.out_shape[1];
                if id % 32 != 0 {
                    return Err(format!(
                        "cuda: {}: matmul input dim {id} is not a multiple of the 32-element quant block",
                        node.name
                    ));
                }
                if self.elems(in_bufs[0]) < id * nt || self.elems(out_buf) < od * nt {
                    return Err(format!(
                        "cuda: {}: buffer size mismatch for [{od}x{id}] x nt={nt}",
                        node.name
                    ));
                }
                self.state.matmul_f32_ptr(
                    wptr,
                    meta.weight_ttype,
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(out_buf)?,
                    od,
                    id,
                    nt,
                )?;
                if let Some(bname) = &meta.bias_name {
                    let bptr = self.state.get_weight_ptr(bname).ok_or_else(|| {
                        format!(
                            "cuda: bias '{bname}' not registered on CUDA ({})",
                            node.name
                        )
                    })?;
                    // add_bias_f32's last argument is the ROW COUNT (nt), not
                    // the total element count — the kernel grid maps one block
                    // row per token (a wrong count writes out of bounds).
                    self.state.add_bias_f32(self.ptr_of(out_buf)?, bptr, od, nt);
                }
                Ok(())
            }

            Op::RoPE { style } => {
                if !matches!(style, RopeStyle::NonInterleaved) {
                    return Err(format!(
                        "cuda: rope style {style:?} not supported (kernel is neox/non-interleaved only)"
                    ));
                }
                let meta = match &node.meta {
                    NodeMeta::Rope(m) => m,
                    other => return Err(format!("rope node missing RoPEMeta: {other:?}")),
                };
                if meta.hd == 0 || meta.hd % 2 != 0 {
                    return Err(format!(
                        "cuda: {}: rope head dim {} must be even",
                        node.name, meta.hd
                    ));
                }
                if in_bufs[0] != out_buf {
                    self.copy_d2d(in_bufs[0], out_buf)?;
                }
                let nt = node.out_shape[1];
                let pos = self.positions_i32(in_bufs[1])?;
                self.state.rope_f32(
                    self.ptr_of(out_buf)?,
                    meta.n_head,
                    meta.hd,
                    nt,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                );
                Ok(())
            }

            Op::KvcacheStore { layer } => {
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                if out_buf != k_id {
                    return Err(format!(
                        "cuda: kv store output buffer {out_buf} is not the K region {k_id}"
                    ));
                }
                let nkt = node.out_shape[0];
                if nkt == 0 || self.elems(in_bufs[0]) % nkt != 0 {
                    return Err(format!(
                        "cuda: kv store k input {} elems not a multiple of nkt {nkt}",
                        self.elems(in_bufs[0])
                    ));
                }
                let nt = self.elems(in_bufs[0]) / nkt;
                let pos = self.positions_i32(in_bufs[2])?;
                // Note: positions >= n_ctx are not validated here (device-side
                // data); the CPU backend checks them, the GPU backends trust
                // session-level clamping like Metal's store_kv dispatch.
                self.state
                    .store_kv_f32(self.ptr_of(in_bufs[0])?, self.ptr_of(k_id)?, nkt, nt, pos);
                self.state
                    .store_kv_f32(self.ptr_of(in_bufs[1])?, self.ptr_of(v_id)?, nkt, nt, pos);
                Ok(())
            }

            Op::Attn { .. } => {
                let meta = match &node.meta {
                    NodeMeta::Attn(m) => m,
                    other => return Err(format!("attn node missing AttnMeta: {other:?}")),
                };
                // Same kernel-invariant guards as Metal (docs/GPU_SAFETY.md):
                // the kernel strides KV by nk*hd, uses the query head dim, and
                // keeps hd/4 accumulators in registers (oc[32] → hd ≤ 128).
                if meta.nkt != meta.n_head_kv * meta.hd {
                    return Err(format!(
                        "cuda: attention nkt={} != n_head_kv*hd={} (kernel strides KV by nk*hd)",
                        meta.nkt,
                        meta.n_head_kv * meta.hd
                    ));
                }
                if meta.hd != meta.hd_kv {
                    return Err(format!(
                        "cuda: attention hd={} != hd_kv={} (kernel uses the query head dim)",
                        meta.hd, meta.hd_kv
                    ));
                }
                if meta.hd == 0 || meta.hd > 128 || meta.hd % 4 != 0 {
                    return Err(format!(
                        "cuda: attention head dim {} outside the kernel's supported range (multiple of 4, 1..=128)",
                        meta.hd
                    ));
                }
                if meta.n_head_kv == 0 || meta.n_head % meta.n_head_kv != 0 {
                    return Err(format!(
                        "cuda: attention n_head {} not divisible by n_head_kv {}",
                        meta.n_head, meta.n_head_kv
                    ));
                }
                let (k_id, v_id) = kv_pair
                    .ok_or_else(|| format!("KV regions for layer {} not allocated", meta.layer))?;
                let nt = node.out_shape[1];
                let pos = self.positions_i32(in_bufs[2])?;
                // The causal bound (positions[t]+1) is derived from the device
                // positions inside the kernel — no host scalar crosses here
                // (precondition for CUDA Graph replay, Phase 7d).
                self.state.gqa_attn_f32(
                    self.ptr_of(in_bufs[0])?,
                    self.ptr_of(k_id)?,
                    self.ptr_of(v_id)?,
                    self.ptr_of(out_buf)?,
                    pos,
                    meta.n_head,
                    meta.n_head_kv,
                    meta.hd,
                    meta.scale,
                    nt,
                );
                Ok(())
            }

            op => Err(format!(
                "cuda: op {op:?} has no kernel (stays on the CPU backend per supports_op)"
            )),
        }
    }

    fn read_host(&self, _id: usize) -> Option<&[f32]> {
        // A staged D2H transfer cannot return a borrowed slice (this method
        // takes &self; the host staging buffer would escape its guard). Use
        // `copy_to_host` via alloc.rs's copy_to_cpu CUDA arm instead.
        None
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let bytes = data.len() * 4;
        let dst = self.ptr_of(id)?;
        if self.pool[id].bytes < bytes {
            return Err(format!(
                "cuda: buffer {id} too small: {} < {bytes} bytes",
                self.pool[id].bytes
            ));
        }
        let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
        self.state.copy_to_device(src, dst);
        Ok(())
    }

    fn synchronize(&mut self) {
        self.state.sync();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::backend::KvProvider;

    /// Init the CUDA singleton; silent-skip the test when no device answers
    /// (e.g. CI without a GPU). Run with --nocapture to see skips.
    fn device() -> Option<&'static crate::cuda::CudaState> {
        crate::cuda::CudaState::init();
        crate::cuda::CudaState::get()
    }

    #[test]
    fn cuda_pool_roundtrip() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut cb = CudaBackend::new().expect("backend after device init");
        let id = cb.alloc_buffer(16);
        cb.write_host(id, &[1.5f32; 16]).unwrap();
        assert_eq!(cb.copy_to_host(id).unwrap(), vec![1.5f32; 16]);
        // shorter than the buffer is fine, longer is rejected
        cb.write_host(id, &[2.0f32; 4]).unwrap();
        assert!(cb.write_host(id, &[2.0f32; 32]).is_err());
        // free-list reuse hands back the same id; pool_gen tracked both times
        cb.free_buffer(id);
        let id2 = cb.alloc_buffer(16);
        assert_eq!(id, id2);
        assert_eq!(cb.pool_gen, 2);
    }

    #[test]
    fn copy_across_cpu_to_cuda_and_back() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut b = crate::graph::builder::GraphBuilder::new();
        let x = b.input("x", [4, 1, 1, 1], DType::F32);
        let y = b.silu(x);
        b.output(y);
        let g = b.build();

        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        alloc.alloc_graph(&g).unwrap();
        let data = [1.0f32, -2.0, 3.0, 4.0];
        alloc.fill_input(&g, "x", &data).unwrap();

        let canon = alloc.node_buffer(x).unwrap();
        // CPU → CUDA staging copy: canonical buffer untouched, cross map holds
        // the device copy (Phase 7c: the old remap-into-node_to_buf semantics
        // broke re-execution of reused graphs — the producing split found its
        // buffer remapped to another backend on the next execute)
        alloc.copy_across(x, crate::graph::Backend::Cuda).unwrap();
        let cross = alloc.cross_buffer(x).expect("cross staging buffer");
        assert_eq!(cross.backend, crate::graph::Backend::Cuda);
        assert_eq!(
            alloc.node_buffer(x).unwrap(),
            canon,
            "canonical buffer must not be remapped"
        );
        assert_eq!(alloc.copy_to_cpu(x).unwrap(), data.to_vec());
        // re-copy (same dst) reuses the same staging buffer id
        alloc.copy_across(x, crate::graph::Backend::Cuda).unwrap();
        assert_eq!(alloc.cross_buffer(x).unwrap().id, cross.id);
        // same-backend copy is a no-op
        alloc.copy_across(x, crate::graph::Backend::CPU).unwrap();
        assert!(alloc.cross_buffer(x).unwrap().backend == crate::graph::Backend::Cuda);
        // rebuild clears staging (buffers freed, map empty)
        alloc.alloc_graph(&g).unwrap();
        assert!(alloc.cross_buffer(x).is_none());
    }

    #[test]
    fn kv_persistent_regions_survive_realloc() {
        if device().is_none() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let mut b = crate::graph::builder::GraphBuilder::new();
        let pos = b.input("positions", [1, 1, 1, 1], DType::I32);
        let k = b.input("k", [16, 1, 1, 1], DType::F32);
        let v = b.input("v", [16, 1, 1, 1], DType::F32);
        let store = b.kvcache_store(0, k, v, pos, 1024);
        let load = b.kvcache_load(0, 16, 1024, 2);
        b.output(load);
        let mut g = b.build();
        g.nodes[store].backend = Some(crate::graph::Backend::Cuda);
        g.nodes[load].backend = Some(crate::graph::Backend::Cuda);

        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        alloc.alloc_graph(&g).unwrap();
        let pair = alloc.kv_pair(0).unwrap();

        // the store node's buffer IS the K region, on the CUDA pool
        let kbuf = alloc.node_buffer(store).unwrap();
        assert_eq!(kbuf.backend, crate::graph::Backend::Cuda);
        assert_eq!(kbuf.id, pair.0);
        {
            let c = alloc.cuda_mut().unwrap();
            c.write_host(kbuf.id, &[7.5f32; 16]).unwrap();
        }

        // rebuild: liveness buffers recycle, KV regions survive unchanged
        alloc.alloc_graph(&g).unwrap();
        assert_eq!(alloc.kv_pair(0).unwrap(), pair);
        let back = alloc.copy_to_cpu(store).unwrap();
        assert_eq!(&back[..16], &[7.5f32; 16]);
    }

    // ─── Phase 7b: per-op dispatch parity ───────────────────────

    use crate::graph::builder::GraphBuilder;
    use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
    use crate::tensor::{Tensor, TensorType};

    /// Fresh backend on an initialized device (None → skip on no-GPU hosts).
    fn pool() -> Option<CudaBackend> {
        device()?;
        CudaBackend::new()
    }

    fn assert_close(name: &str, got: &[f32], want: &[f32], tol: f32) {
        assert_eq!(got.len(), want.len(), "{name}: length mismatch");
        let mut worst = (0.0f32, 0usize);
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            let d = (g - w).abs();
            if d > worst.0 {
                worst = (d, i);
            }
        }
        assert!(
            worst.0 <= tol,
            "{name}: max diff {} at {} (got {}, want {})",
            worst.0,
            worst.1,
            got[worst.1],
            want[worst.1]
        );
    }

    #[test]
    fn cuda_elementwise_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let n = 257usize; // odd size exercises the elementwise tail guard
        let mut b = GraphBuilder::new();
        let a = b.input("a", [n, 1, 1, 1], DType::F32);
        let c = b.input("c", [n, 1, 1, 1], DType::F32);
        let add = b.add(a, c);
        let mul = b.mul(add, c);
        let sw = b.swiglu(mul, c); // gate is the RAW pre-activation
        let silu = b.silu(mul);
        b.output(silu);
        b.output(sw);
        let g = b.build();

        let (x, y) = (cb.alloc_buffer(n), cb.alloc_buffer(n));
        let (t1, t2, t3) = (cb.alloc_buffer(n), cb.alloc_buffer(n), cb.alloc_buffer(n));
        let xs: Vec<f32> = (0..n).map(|i| ((i * 37) % 23) as f32 / 4.0 - 2.5).collect();
        let ys: Vec<f32> = (0..n).map(|i| ((i * 91) % 17) as f32 / 3.0 - 2.0).collect();
        cb.write_host(x, &xs).unwrap();
        cb.write_host(y, &ys).unwrap();

        cb.execute_node(&g.nodes[add], &[x, y], t1, None).unwrap();
        cb.execute_node(&g.nodes[mul], &[t1, y], t2, None).unwrap();
        // SwiGLU consumes the RAW mul output, so it must run before the
        // in-place Silu overwrites t2 (alias path, graph rules §5).
        cb.execute_node(&g.nodes[sw], &[t2, y], t3, None).unwrap();
        cb.execute_node(&g.nodes[silu], &[t2], t2, None).unwrap();

        // Host reference through the same vec_ops the CPU backend uses.
        let mut r1 = vec![0f32; n];
        crate::vec_ops::vec_add_f32(n, &mut r1, &xs, &ys);
        assert_eq!(cb.copy_to_host(t1).unwrap(), r1, "add must be bit-exact");
        let mut r2 = vec![0f32; n];
        crate::vec_ops::vec_mul_f32(n, &mut r2, &r1, &ys);
        let mut r3 = vec![0f32; n];
        crate::vec_ops::vec_silu_f32(n, &mut r3, &r2);
        let got2 = cb.copy_to_host(t2).unwrap();
        assert_close("mul+silu (in-place)", &got2, &r3, 1e-5);
        let mut r4 = vec![0f32; n];
        crate::vec_ops::vec_mul_f32(n, &mut r4, &r3, &ys);
        assert_close("swiglu", &cb.copy_to_host(t3).unwrap(), &r4, 1e-5);
    }

    #[test]
    fn cuda_norm_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        // RmsNorm: d=64, nt=5
        let (d, nt) = (64usize, 5usize);
        let w: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 / 8.0).collect();
        let wbytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.state.register_weight("nw", &wbytes);
        let mut wt = Tensor::from_data(TensorType::F32, &[d as i64, 1, 1, 1], wbytes.clone());
        wt.name = "nw".to_string();

        // QkNorm: hd=16, nh=4, nt=3 — rows (t*nh + h) form a contiguous
        // [nt*nh, hd] matrix, so the same rms_norm kernel covers it.
        let (hd, nh, nt2) = (16usize, 4usize, 3usize);
        let qw: Vec<f32> = (0..hd).map(|i| 1.0 / (1.0 + i as f32)).collect();
        let qbytes: Vec<u8> = qw.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.state.register_weight("qw", &qbytes);
        let mut qwt = Tensor::from_data(TensorType::F32, &[hd as i64, 1, 1, 1], qbytes);
        qwt.name = "qw".to_string();

        let mut b = GraphBuilder::new();
        let x = b.input("x", [d, nt, 1, 1], DType::F32);
        let rn = b.rms_norm(x, Some(&wt), 1e-5);
        let q = b.input("q", [hd * nh, nt2, 1, 1], DType::F32);
        let qn = b.qk_norm(q, Some(&qwt), hd, nh, 1e-5);
        b.output(rn);
        b.output(qn);
        let g = b.build();

        let xb = cb.alloc_buffer(d * nt);
        let xs: Vec<f32> = (0..d * nt)
            .map(|i| ((i * 53) % 31) as f32 / 7.0 - 2.0)
            .collect();
        cb.write_host(xb, &xs).unwrap();
        let ob = cb.alloc_buffer(d * nt);
        cb.execute_node(&g.nodes[rn], &[xb], ob, None).unwrap();

        let qb = cb.alloc_buffer(hd * nh * nt2);
        let qs: Vec<f32> = (0..hd * nh * nt2)
            .map(|i| ((i * 71) % 29) as f32 / 6.0 - 2.5)
            .collect();
        cb.write_host(qb, &qs).unwrap();
        let qo = cb.alloc_buffer(hd * nh * nt2);
        cb.execute_node(&g.nodes[qn], &[qb], qo, None).unwrap();

        let mut want = vec![0f32; d * nt];
        for t in 0..nt {
            crate::vec_ops::rms_norm_fused_f32(
                d,
                &mut want[t * d..(t + 1) * d],
                &xs[t * d..(t + 1) * d],
                &w,
                1e-5,
            );
        }
        assert_close("rms_norm", &cb.copy_to_host(ob).unwrap(), &want, 1e-4);

        let mut want2 = vec![0f32; hd * nh * nt2];
        for r in 0..nh * nt2 {
            crate::vec_ops::rms_norm_fused_f32(
                hd,
                &mut want2[r * hd..(r + 1) * hd],
                &qs[r * hd..(r + 1) * hd],
                &qw,
                1e-5,
            );
        }
        assert_close("qk_norm", &cb.copy_to_host(qo).unwrap(), &want2, 1e-4);
    }

    #[test]
    fn cuda_matmul_parity() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (od, id_, nt) = (32usize, 64usize, 3usize);
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
            .collect();
        let bias: Vec<f32> = (0..od).map(|i| (i % 5) as f32 / 10.0).collect();

        // Q8_0 weight [out][in] row-major, quantized per row
        let wf8: Vec<f32> = (0..od * id_)
            .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let mut w8b = Vec::new();
        for r in 0..od {
            w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
                &wf8[r * id_..(r + 1) * id_],
            ));
        }
        let mut w8 = Tensor::from_data(
            TensorType::Q8_0,
            &[id_ as i64, od as i64, 1, 1],
            w8b.clone(),
        );
        w8.name = "mw8".to_string();
        cb.state.register_weight("mw8", &w8b);
        let biasb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        cb.state.register_weight("mb", &biasb);
        let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], biasb);
        bt.name = "mb".to_string();

        // Q4_0 weight (18 bytes per 32 values: f16 d + 16 nibbles)
        let wf4: Vec<f32> = (0..od * id_)
            .map(|i| ((i * 40503) % 991) as f32 / 496.0 - 1.0)
            .collect();
        let mut w4b = Vec::new();
        for r in 0..od {
            let row = &wf4[r * id_..(r + 1) * id_];
            for bi in 0..id_ / 32 {
                let blk = &row[bi * 32..bi * 32 + 32];
                let amax = blk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let dsc = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
                w4b.extend_from_slice(&half::f16::from_f32(dsc).to_le_bytes());
                for j in 0..16 {
                    let q0 = ((blk[j] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                    let q1 = ((blk[j + 16] / dsc).round() as i32 + 8).clamp(0, 15) as u8;
                    w4b.push(q0 | (q1 << 4));
                }
            }
        }
        let mut w4 = Tensor::from_data(
            TensorType::Q4_0,
            &[id_ as i64, od as i64, 1, 1],
            w4b.clone(),
        );
        w4.name = "mw4".to_string();
        cb.state.register_weight("mw4", &w4b);

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id_, nt, 1, 1], DType::F32);
        let m8 = b.matmul(x, &w8, Some(&bt));
        let m4 = b.matmul(x, &w4, Some(&bt));
        b.output(m8);
        b.output(m4);
        let g = b.build();

        let xb = cb.alloc_buffer(id_ * nt);
        cb.write_host(xb, &xs).unwrap();
        let (o8, o4) = (cb.alloc_buffer(od * nt), cb.alloc_buffer(od * nt));
        cb.execute_node(&g.nodes[m8], &[xb], o8, None).unwrap();
        cb.execute_node(&g.nodes[m4], &[xb], o4, None).unwrap();

        // References: dequantized weight rows × f32 activations + bias
        // (embed_tokens doubles as the row dequantizer for these types).
        let mut dq8 = vec![0f32; od * id_];
        crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8, &mut dq8, id_);
        let mut dq4 = vec![0f32; od * id_];
        crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w4, &mut dq4, id_);
        for (name, o, dq) in [("q8_0 matmul", o8, &dq8), ("q4_0 matmul", o4, &dq4)] {
            let got = cb.copy_to_host(o).unwrap();
            let mut want = vec![0f32; od * nt];
            for t in 0..nt {
                for r in 0..od {
                    let mut acc = 0f32;
                    for i in 0..id_ {
                        acc += dq[r * id_ + i] * xs[t * id_ + i];
                    }
                    want[t * od + r] = acc + bias[r];
                }
            }
            let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
            assert_close(name, &got, &want, scale * 1e-3);
        }
    }

    #[test]
    fn cuda_rope_kv_attn_roundtrip() {
        let Some(mut cb) = pool() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (nh, nk_h, hd) = (4usize, 2usize, 8usize);
        let nkt = nk_h * hd;
        let (nt, n_ctx) = (3usize, 32usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let pos: Vec<usize> = vec![1, 4, 9]; // sparse, exercises the scatter

        let mut b = GraphBuilder::new();
        let q = b.input("q", [nh * hd, nt, 1, 1], DType::F32);
        let k = b.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = b.input("v", [nkt, nt, 1, 1], DType::F32);
        let p = b.input("positions", [nt, 1, 1, 1], DType::I32);
        let store = b.kvcache_store(0, k, v, p, n_ctx);
        let load = b.kvcache_load(0, nkt, n_ctx, nk_h);
        let qr = b.rope(
            q,
            p,
            RopeStyle::NonInterleaved,
            RoPEMeta {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_head: nh,
                hd,
            },
        );
        let at = b.attn(
            qr,
            load,
            p,
            AttnMode::Gqa,
            AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk_h,
                hd,
                hd_kv: hd,
                nkt,
                scale,
            },
        );
        b.output(at);
        let g = b.build();

        let (xb_q, xb_k, xb_v) = (
            cb.alloc_buffer(nh * hd * nt),
            cb.alloc_buffer(nkt * nt),
            cb.alloc_buffer(nkt * nt),
        );
        let xb_p = cb.alloc_buffer(nt);
        let (ob_qr, ob_at) = (cb.alloc_buffer(nh * hd * nt), cb.alloc_buffer(nh * hd * nt));
        let (kreg, vreg) = (cb.alloc_buffer(nkt * n_ctx), cb.alloc_buffer(nkt * n_ctx));

        let qs: Vec<f32> = (0..nh * hd * nt)
            .map(|i| ((i * 37) % 19) as f32 / 5.0 - 1.9)
            .collect();
        let ks: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 41) % 13) as f32 / 4.0 - 1.5)
            .collect();
        let vs: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 57) % 11) as f32 / 3.0 - 1.8)
            .collect();
        let pb: Vec<f32> = pos.iter().map(|&pp| f32::from_bits(pp as u32)).collect();
        cb.write_host(xb_q, &qs).unwrap();
        cb.write_host(xb_k, &ks).unwrap();
        cb.write_host(xb_v, &vs).unwrap();
        cb.write_host(xb_p, &pb).unwrap();
        // Zero the KV regions first: rows the store never touches stay
        // uninitialized in a recycled cudaMalloc block, and the reference
        // below treats unwritten rows as zeros (deterministic vs pool state).
        cb.write_host(kreg, &vec![0f32; nkt * n_ctx]).unwrap();
        cb.write_host(vreg, &vec![0f32; nkt * n_ctx]).unwrap();

        cb.execute_node(
            &g.nodes[store],
            &[xb_k, xb_v, xb_p],
            kreg,
            Some((kreg, vreg)),
        )
        .unwrap();
        cb.execute_node(&g.nodes[qr], &[xb_q, xb_p], ob_qr, None)
            .unwrap();
        cb.execute_node(
            &g.nodes[at],
            &[ob_qr, kreg, xb_p],
            ob_at,
            Some((kreg, vreg)),
        )
        .unwrap();

        // a) stored K rows are bit-exact at the scattered positions
        let kback = cb.copy_to_host(kreg).unwrap();
        for (t, &pp) in pos.iter().enumerate() {
            assert_eq!(
                &kback[pp * nkt..(pp + 1) * nkt],
                &ks[t * nkt..(t + 1) * nkt],
                "K row {pp}"
            );
        }
        // b) RoPE vs cpu_rope (also covers the non-alias D2D staging path)
        let qgot = cb.copy_to_host(ob_qr).unwrap();
        let mut qref = qs.clone();
        crate::graph::cpu_backend::cpu_rope(
            &mut qref,
            &pos,
            nh,
            hd,
            10000.0,
            1.0,
            RopeStyle::NonInterleaved,
        );
        assert_close("rope", &qgot, &qref, 1e-4);
        // c) GQA attention vs cpu_gqa_attn over the scattered KV regions
        let mut kfull = vec![0f32; nkt * n_ctx];
        let mut vfull = vec![0f32; nkt * n_ctx];
        for (t, &pp) in pos.iter().enumerate() {
            kfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&ks[t * nkt..(t + 1) * nkt]);
            vfull[pp * nkt..(pp + 1) * nkt].copy_from_slice(&vs[t * nkt..(t + 1) * nkt]);
        }
        let nkv = pos.iter().copied().max().unwrap() + 1;
        let mut aref = vec![0f32; nh * hd * nt];
        crate::graph::cpu_backend::cpu_gqa_attn(
            &qref, &kfull, &vfull, &pos, nt, nkv, nh, nk_h, hd, hd, nkt, &mut aref, scale,
        )
        .unwrap();
        let agot = cb.copy_to_host(ob_at).unwrap();
        assert_close("gqa_attn", &agot, &aref, 1e-4);
    }

    #[test]
    fn cuda_scheduler_chain() {
        crate::cuda::CudaState::init();
        let Some(state) = crate::cuda::CudaState::get() else {
            eprintln!("skipping: no CUDA device");
            return;
        };
        let (id_, od, nt) = (64usize, 32usize, 2usize);
        let cw: Vec<f32> = (0..id_).map(|i| 0.8 + (i % 5) as f32 / 10.0).collect();
        let cwb: Vec<u8> = cw.iter().flat_map(|v| v.to_le_bytes()).collect();
        state.register_weight("cw", &cwb);
        let mut cwt = Tensor::from_data(TensorType::F32, &[id_ as i64, 1, 1, 1], cwb);
        cwt.name = "cw".to_string();
        let wf: Vec<f32> = (0..od * id_)
            .map(|i| ((i * 2654435761 % 1000) as f32 / 500.0) - 1.0)
            .collect();
        let mut w8b = Vec::new();
        for r in 0..od {
            w8b.extend_from_slice(&crate::quants::quantize_row_q8_0(
                &wf[r * id_..(r + 1) * id_],
            ));
        }
        state.register_weight("cw8", &w8b);
        let mut w8t = Tensor::from_data(TensorType::Q8_0, &[id_ as i64, od as i64, 1, 1], w8b);
        w8t.name = "cw8".to_string();
        let bias: Vec<f32> = (0..od).map(|i| (i % 3) as f32 / 7.0).collect();
        let bb: Vec<u8> = bias.iter().flat_map(|v| v.to_le_bytes()).collect();
        state.register_weight("cb", &bb);
        let mut bt = Tensor::from_data(TensorType::F32, &[od as i64, 1, 1, 1], bb);
        bt.name = "cb".to_string();

        let mut b = GraphBuilder::new();
        let x = b.input("x", [id_, nt, 1, 1], DType::F32);
        let n1 = b.rms_norm(x, Some(&cwt), 1e-5);
        let m = b.matmul(n1, &w8t, Some(&bt));
        let s = b.silu(m);
        b.output(s);
        let mut g = b.build();

        // Full pipeline: assign → alloc → fill → execute (no fusion needed).
        let mut alloc = GraphAllocator::new();
        if !alloc.enable_cuda() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let sched = crate::graph::scheduler::BackendScheduler;
        sched.assign_backends(&mut g, &alloc);
        for (i, nd) in g.nodes.iter().enumerate() {
            assert_eq!(
                nd.backend,
                Some(crate::graph::Backend::Cuda),
                "node {i} ({})",
                nd.name
            );
        }
        alloc.alloc_graph(&g).unwrap();
        let xs: Vec<f32> = (0..id_ * nt)
            .map(|i| ((i * 97) % 21) as f32 / 5.0 - 2.0)
            .collect();
        alloc.fill_input(&g, "x", &xs).unwrap();
        sched.execute(&g, &mut alloc).unwrap();

        // Host reference: rms → dequant matmul + bias → silu
        let mut rmsd = vec![0f32; id_ * nt];
        for t in 0..nt {
            crate::vec_ops::rms_norm_fused_f32(
                id_,
                &mut rmsd[t * id_..(t + 1) * id_],
                &xs[t * id_..(t + 1) * id_],
                &cw,
                1e-5,
            );
        }
        let mut dq = vec![0f32; od * id_];
        crate::kernel::embed_tokens(&(0..od as u32).collect::<Vec<u32>>(), &w8t, &mut dq, id_);
        let mut mm = vec![0f32; od * nt];
        for t in 0..nt {
            for r in 0..od {
                let mut acc = 0f32;
                for i in 0..id_ {
                    acc += dq[r * id_ + i] * rmsd[t * id_ + i];
                }
                mm[t * od + r] = acc + bias[r];
            }
        }
        let mut want = vec![0f32; od * nt];
        crate::vec_ops::vec_silu_f32(od * nt, &mut want, &mm);
        let got = alloc.copy_to_cpu(s).unwrap();
        let scale = want.iter().fold(1e-9f32, |m, v| m.max(v.abs()));
        assert_close("scheduler chain", &got, &want, scale * 1e-3);
    }
}
