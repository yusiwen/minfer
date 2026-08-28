//! CUDA graph backend (Phase 7).
//!
//! Wraps the [`crate::cuda::CudaState`] singleton in the graph [`Backend`]
//! trait contract (`src/graph/backend.rs`), mirroring `metal_backend.rs` where
//! the mechanics allow: a device buffer pool with a byte-length free list,
//! name → device-pointer weight resolution, sync H2D/D2H host transfers, and
//! per-op kernel dispatch on the shared stream. Design + rollout:
//! `docs/CUDA-BACKEND-PLAN.md`.

use super::backend::Backend;
use super::ops::{FusedOp, Op};
use super::{CNode, DType};

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
    }
}

impl CudaBackend {
    fn state_free(ptr: *mut std::ffi::c_void) {
        <crate::cuda::CudaState>::cuda_free(ptr);
    }
}

impl Backend for CudaBackend {
    fn name(&self) -> &str {
        "cuda"
    }

    /// v1 capability matrix (docs/CUDA-BACKEND-PLAN.md §4.3): the full
    /// per-layer chain runs on CUDA; Embed/GetRows, Scale, Softmax and the
    /// fused decode ops have no kernels and stay on the CPU backend.
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        if dtype != DType::F32 {
            return false;
        }
        matches!(
            op,
            Op::Input
                | Op::Add
                | Op::Mul
                | Op::Silu
                | Op::SwiGLU
                | Op::RmsNorm { .. }
                | Op::QkNorm { .. }
                | Op::MatMul { .. }
                | Op::RoPE { .. }
                | Op::Attn { .. }
                | Op::KvcacheStore { .. }
                | Op::KvcacheLoad { .. }
                | Op::View { .. }
                | Op::Reshape { .. }
                | Op::Permute { .. }
        )
    }

    fn supports_fused(&self, _fused: &FusedOp) -> bool {
        // SwiGLU flips to true in Phase 7b when the dispatch lands (the
        // kernel already exists; 7a only wires the skeleton).
        false
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

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[usize],
        out_buf: usize,
        _kv_pair: Option<(usize, usize)>,
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
                let s = self.ptr_of(src)?;
                let d = self.ptr_of(out_buf)?;
                let (sb, db) = (self.pool[src].bytes, self.pool[out_buf].bytes);
                if sb != db {
                    return Err(format!(
                        "cuda: {}: size mismatch src {sb} vs out {db} bytes",
                        node.name
                    ));
                }
                self.state.copy_device_to_device(s, d, db);
                Ok(())
            }
            op => Err(format!(
                "cuda: op {op:?} not implemented yet (Phase 7b dispatch)"
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

        // CPU → CUDA → host readback
        alloc.copy_across(x, crate::graph::Backend::Cuda).unwrap();
        assert_eq!(alloc.copy_to_cpu(x).unwrap(), data.to_vec());
        // CUDA → CPU → host readback
        alloc.copy_across(x, crate::graph::Backend::CPU).unwrap();
        assert_eq!(alloc.copy_to_cpu(x).unwrap(), data.to_vec());
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
}
