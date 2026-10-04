//! CPU backend (Phase 2): executes IR nodes with the existing scalar/SIMD
//! kernels (vec_ops.rs, kernel.rs). Buffers are plain `Vec<f32>`; I32 input
//! data (token ids, positions) is stored as `f32::from_bits` bit patterns —
//! exact for |v| < 2^24 (vocab sizes and context lengths are far below that).
//!
//! KV region layout (per layer, contiguous): `[ K (n_embd*n_ctx) | V ]`.

use std::collections::HashMap;

use crate::kernel;
use crate::tensor::Tensor;
use crate::vec_ops::RopeStyle;

use super::backend::Backend;
use super::kvformat::{self, KvFormat};
use super::ops::{FusedOp, NodeMeta, Op};
use super::{BufRef, CNode, DType};

/// CPU buffer pool + weight registry.
pub struct CpuBackend {
    buffers: Vec<Vec<f32>>,
    free: Vec<usize>,
    weights: HashMap<String, Tensor>,
    /// C4: the KV storage format this backend's kernels speak, snapshotted from the
    /// process-wide policy at construction (like CUDA's `kv_f16`). A packed region
    /// makes the store quantize and the attention read dequantize.
    kv_format: KvFormat,
    /// Reusable f32 scratch the packed read path dequantizes a query window into
    /// (one per region). Kept across calls: it is one window, not the whole cache.
    /// Only the fallback path (`hd_kv` narrower than one Q8_0 block, or
    /// `MINFER_NO_FUSED_Q8_KV`) uses these.
    kv_scratch_k: Vec<f32>,
    kv_scratch_v: Vec<f32>,
    /// Reusable one-cell payload for the packed store, so packing `nt` cells
    /// allocates once rather than once per row.
    kv_pack_buf: Vec<u8>,
    /// E4 S3: how many times the pool actually created a buffer. A rebuild that
    /// re-maps onto reserved slots must not move this — it is the CPU-side twin of
    /// CUDA's `pool_gen`, and what the "reserve, then assign" gate asserts.
    allocs: usize,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            free: Vec::new(),
            weights: HashMap::new(),
            // C4 per-engine: F32 until `GraphAllocator::set_kv_format` stamps the
            // loaded engine's resolved format (issue #99 — no process global).
            kv_format: KvFormat::F32,
            kv_scratch_k: Vec::new(),
            kv_scratch_v: Vec::new(),
            kv_pack_buf: Vec::new(),
            allocs: 0,
        }
    }

    /// C4: the KV format this backend's kernels speak — the loaded engine's
    /// `ModelDef::kv_format`, stamped through `GraphAllocator::set_kv_format`. A
    /// packed region makes the store quantize and the attention read dequantize.
    pub fn kv_format(&self) -> KvFormat {
        self.kv_format
    }

    /// C4 per-engine: set the KV format this backend's kernels speak. The
    /// allocator calls it once per forward with the model's resolved format, so two
    /// engines with different formats no longer share a process-wide answer
    /// (issue #99). Device tests exercise one layout explicitly instead of through
    /// the process environment (`cuda_backend` does the same with
    /// `set_kv_f16_for_test`).
    pub fn set_kv_format(&mut self, format: KvFormat) {
        self.kv_format = format;
    }

    /// Register a weight tensor by name (Phase 6 wires this from the model).
    pub fn register_weight(&mut self, name: &str, t: Tensor) {
        // Skip re-registration of an already-known weight: Tensor carries its
        // bytes as Cow::Owned, so the `t.clone()` at the model call sites
        // deep-copies the full weight set (~4.4 GB on 7B) on EVERY graph
        // (re)build — measured as a ~635 ms pure-CPU stall at the
        // prefill→decode graph switch (no CUDA calls, no kernels). Model
        // weights are immutable after load (weights_version guards any future
        // change), so a same-name registration always carries the same data.
        if self.weights.contains_key(name) {
            return;
        }
        self.weights.insert(name.to_string(), t);
    }

    /// E4 S3: buffers this pool has created (never reused from the slot table).
    /// (Test / accounting helper.)
    /// Test-only (#238): driven by `graph::alloc::tests::slots`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn alloc_count(&self) -> usize {
        self.allocs
    }

    /// D1: the `f32` window a [`BufRef`] names — the whole buffer for an owning
    /// node (offset 0, len = its element count), the parent's bytes at the
    /// window for a view.
    fn window(&self, r: BufRef) -> &[f32] {
        &self.buffers[r.id][r.offset..r.offset + r.len]
    }

    /// Pool size (for tests).
    ///
    /// Test-only (#239): the only caller is the test helper
    /// `GraphAllocator::n_cpu_buffers`, itself moved into `graph::alloc::tests` by
    /// #239. `pub(in crate::graph)` is the narrowest spelling that reaches it — this
    /// item is what closed [#238]'s `pool_len` deferral.
    #[cfg(test)]
    pub(in crate::graph) fn pool_len(&self) -> usize {
        self.buffers.len()
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

/// F4: the CPU capability matrix, as a free function.
///
/// This is the **authority** for the answer: the `Backend` trait method below
/// forwards to it (the function cannot take a `&self`), and assignment reads the
/// trait method (`graph::backend_takes`). The registry does not carry a copy.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
    if dtype != DType::F32 {
        return false;
    }
    matches!(
        op,
        Op::Input
            | Op::Add
            | Op::Mul
            | Op::Scale(_)
            | Op::Silu
            | Op::Softmax { .. }
            | Op::RmsNorm { .. }
            | Op::QkNorm { .. }
            | Op::MatMul { .. }
            | Op::GetRows
            | Op::RoPE { .. }
            | Op::Attn { .. }
            | Op::KvcacheStore { .. }
            | Op::KvcacheLoad { .. }
            | Op::SwiGLU
            | Op::View { .. }
            | Op::Reshape { .. }
            | Op::Permute { .. }
    )
}

/// F4: the fusion-pass capability, as a free function (see [`supports_op`]).
pub fn supports_fused(fused: &FusedOp) -> bool {
    // The SwiGLU rewrite IS applied to CPU nodes (CPU is first in the
    // fusion pass's backend list); `Op::SwiGLU` below executes it as a
    // single pass. bias+rope and batch-matmul are not fused (batch QKV
    // quantize-sharing is a Phase 5+ win).
    matches!(fused, FusedOp::SwiGLU)
}

/// E1: the CPU attention kernel reads `attn_span` (the window the KV store
/// resolved), so it can take a multi-sequence attention node.
pub const SUPPORTS_ATTN_SPAN: bool = true;

/// C4: the CPU attention kernel is the one that reads a packed `q8_0` KV region
/// (it dots the stored K blocks against the quantized query and accumulates V
/// out of the cell). CUDA and Metal address f32/f16 rows — issue [#87] is the
/// work that flips their value and adds their kernels.
///
/// [#87]: https://github.com/yusiwen/minfer/issues/87
pub const READS_PACKED_KV: bool = true;

/// F5 ([#58]): registry hook **phase A** of a cross-backend staging copy, CPU
/// source.
///
/// The CPU backend has **no device memory**, so there is nothing to make
/// asynchronous: this is exactly the synchronous host round trip the boundary
/// always performed (read the source into a host vector, write it into the
/// destination's staging buffer). Declaring it here rather than as a
/// `if backend != CPU` branch in the allocator is the F4 shape — the fact that
/// the CPU's answer is "there is no transfer to overlap" is a *registered
/// answer*, not a special case.
///
/// The device leg of a CPU→CUDA staging copy is not in this hook: the
/// destination pool's own `write_host` is already the pinned, stream-ordered
/// `cudaMemcpyAsync` fill (7e⑥), so that direction never blocked the host either
/// — which is why the ticket's blocking-copy count is about the *device→host*
/// direction only.
///
/// [#58]: https://github.com/yusiwen/minfer/issues/58
pub(crate) fn copy_cross(
    alloc: &mut super::alloc::GraphAllocator,
    uid: u64,
    node_id: super::NodeId,
    dst_backend: super::Backend,
) -> Result<bool, String> {
    alloc.host_round_trip_cross(uid, node_id, dst_backend)?;
    Ok(true)
}

/// F5 ([#58]): registry hook **phase B** for a CPU source — a documented no-op.
///
/// The bytes were produced by phase A (a plain host memcpy) and the device leg,
/// when the destination is a device, is ordered on that device's stream behind
/// its own fill. There is no event to wait for and no host stall to take. The
/// allocator still counts the wait (phase B is issued exactly once per staged
/// input, whatever the backend), which is what makes "the boundary path issues a
/// wait for every staged input" a backend-independent, CI-covered assertion.
///
/// [#58]: https://github.com/yusiwen/minfer/issues/58
pub(crate) fn await_cross(
    _alloc: &mut super::alloc::GraphAllocator,
    _uid: u64,
    _node_id: super::NodeId,
    _dst_backend: super::Backend,
) -> Result<(), String> {
    Ok(())
}

/// F4: this backend's registry entry. Called by `Registry::build` at startup —
/// the only place the CPU backend is introduced to the registry.
pub fn entry() -> super::registry::BackendEntry {
    use super::registry::{Backend as Handle, BackendCaps, BackendEntry, PRIORITY_CPU};
    BackendEntry {
        handle: Handle::CPU,
        name: "cpu",
        priority: PRIORITY_CPU,
        caps: BackendCaps {
            reads_packed_kv: READS_PACKED_KV,
        },
        pool: |a| Some(a.cpu()),
        pool_mut: |a| Some(a.cpu_mut()),
        host_read: |a, id| a.cpu().read_host(id).map(|s| s.to_vec()),
        // F5: the CPU is a registered participant in the boundary contract, not a
        // special case: phase A is the synchronous host round trip, phase B is the
        // documented no-op. See `copy_cross` / `await_cross` above.
        copy_cross,
        await_cross,
        // Per-engine (issue #99): the allocator stamps the loaded engine's KV
        // format onto this pool; that stamped value is the live answer, so a
        // session header and the kernels cannot disagree.
        kv_format: |a| a.cpu().kv_format(),
        enable: |_| true,
        unavailable: || None,
    }
}

/// F4: register the CPU backend. Always present: it is the universal fallback.
pub fn register(registry: &mut super::registry::Registry) {
    registry.register_entry(entry());
}

impl Backend for CpuBackend {
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        supports_op(op, dtype)
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        supports_fused(fused)
    }

    fn supports_attn_span(&self) -> bool {
        SUPPORTS_ATTN_SPAN
    }

    /// Bytes of registered weights (E4: the feasibility gate charges the budget for
    /// them, so "weights + activations" is one comparison).
    fn weights_bytes(&self) -> usize {
        self.weights.values().map(|t| t.data.len()).sum()
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        if let Some(idx) = self
            .free
            .iter()
            .position(|&id| self.buffers[id].len() == size)
        {
            let id = self.free.swap_remove(idx);
            self.buffers[id].fill(0.0);
            return id;
        }
        self.buffers.push(vec![0.0f32; size]);
        self.allocs += 1;
        self.buffers.len() - 1
    }

    fn free_buffer(&mut self, id: usize) {
        if !self.free.contains(&id) {
            self.free.push(id);
        }
    }

    fn pool_len(&self) -> usize {
        self.buffers.len()
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // never recycled from the free list (see Backend::alloc_fresh)
        self.buffers.push(vec![0.0f32; size]);
        self.allocs += 1;
        self.buffers.len() - 1
    }

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[BufRef],
        out_buf: BufRef,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        // KV store needs mutable access to both K and V persistent regions
        // (the V region is a sibling buffer, not an input) — handle it before
        // the aliasing split below, which borrows the pool immutably.
        if let Op::KvcacheStore { layer } = &node.op {
            let (k_id, v_id) =
                kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
            if k_id != out_buf.id {
                return Err("KV store out buffer must be the K region".into());
            }
            // C4: the region's cell width comes from the meta (`row_elems`, packed
            // under a Q8_0 policy); the node's shapes stay logical, so `nt` is the K
            // input's logical row count either way.
            let nkt = match &node.meta {
                NodeMeta::Kvcache(m) => m.n_embd,
                other => return Err(format!("KV store node missing KvcacheMeta: {other:?}")),
            };
            let row_elems = self.kv_format.row_elems(nkt);
            let n_ctx = node.out_shape[1];
            let nt = in_bufs[0].len / nkt;
            let pos: Vec<usize> = self
                .window(in_bufs[2])
                .iter()
                .map(|b| b.to_bits() as usize)
                .collect();
            let k_src = self.window(in_bufs[0]).to_vec();
            let v_src = self.window(in_bufs[1]).to_vec();
            // k region = out_buf, v region = sibling; both in this pool.
            // Disjoint mutable access via split_at_mut.
            let (k_dst, v_dst): (&mut [f32], &mut [f32]) = if v_id < k_id {
                let (before, after) = self.buffers.split_at_mut(k_id);
                (&mut after[0], &mut before[v_id])
            } else {
                let (before, after) = self.buffers.split_at_mut(v_id);
                (&mut before[k_id], &mut after[0])
            };
            let packed = self.kv_format.is_packed();
            // C4 S2: one quantization scratch for the whole store, not one per cell.
            if packed {
                self.kv_pack_buf
                    .resize(KvFormat::Q8_0.payload_bytes(nkt), 0);
            }
            for t in 0..nt {
                let p = pos[t];
                if p >= n_ctx {
                    return Err(format!("KV store position {p} >= n_ctx {n_ctx}"));
                }
                let ks = p * row_elems;
                let src = t * nkt;
                if packed {
                    kvformat::pack_q8_0_cell_into(
                        &mut k_dst[ks..ks + row_elems],
                        nkt,
                        &k_src[src..src + nkt],
                        &mut self.kv_pack_buf,
                    );
                    kvformat::pack_q8_0_cell_into(
                        &mut v_dst[ks..ks + row_elems],
                        nkt,
                        &v_src[src..src + nkt],
                        &mut self.kv_pack_buf,
                    );
                } else {
                    k_dst[ks..ks + row_elems].copy_from_slice(&k_src[src..src + nkt]);
                    v_dst[ks..ks + row_elems].copy_from_slice(&v_src[src..src + nkt]);
                }
            }
            return Ok(());
        }

        // Aliasing safety: liveness reuse may map an input and the output to
        // the same physical buffer. Snapshot aliased inputs, then carve the
        // output region out of the pool with split_at_mut so the remaining
        // inputs can be borrowed immutably alongside it.
        // D1: every reference is a *window* (offset + len) of a pool buffer, so
        // an owning node's `out`/`ins` are exactly what they were before views
        // existed (offset 0, len = its element count) and a view's are the
        // parent's bytes at the window.
        let mut aliased: Vec<Vec<f32>> = Vec::new();
        let mut alias_of: Vec<Option<usize>> = in_bufs.iter().map(|_| None).collect();
        for (k, &i) in in_bufs.iter().enumerate() {
            if i.id == out_buf.id {
                alias_of[k] = Some(aliased.len());
                aliased.push(self.buffers[out_buf.id].clone());
            }
        }
        let (before, rest) = self.buffers.split_at_mut(out_buf.id);
        let (out0, after) = rest.split_at_mut(1);
        let out = &mut out0[0][out_buf.offset..out_buf.offset + out_buf.len];
        let mut ins: Vec<&[f32]> = Vec::with_capacity(in_bufs.len());
        for (k, &i) in in_bufs.iter().enumerate() {
            let (off, len) = (i.offset, i.len);
            match alias_of[k] {
                Some(ai) => ins.push(&aliased[ai][off..off + len]),
                None if i.id < out_buf.id => ins.push(&before[i.id][off..off + len]),
                _ => ins.push(&after[i.id - out_buf.id - 1][off..off + len]),
            }
        }

        match &node.op {
            Op::Input => Ok(()), // data pre-filled by the allocator

            Op::Silu => {
                crate::vec_ops::vec_silu_f32(out.len(), out, ins[0]);
                Ok(())
            }
            Op::Add => {
                crate::vec_ops::vec_add_f32(out.len(), out, ins[0], ins[1]);
                Ok(())
            }
            Op::Mul => {
                crate::vec_ops::vec_mul_f32(out.len(), out, ins[0], ins[1]);
                Ok(())
            }
            Op::Scale(s) => {
                out.copy_from_slice(ins[0]);
                crate::vec_ops::vec_scale_f32(out.len(), out, *s);
                Ok(())
            }
            Op::RmsNorm { eps } => {
                let w = match &node.meta {
                    NodeMeta::Norm(m) => m.weight_name.as_ref().and_then(|n| self.weights.get(n)),
                    _ => None,
                };
                let d = node.out_shape[0];
                let n = out.len() / d;
                for t in 0..n {
                    let row = &ins[0][t * d..(t + 1) * d];
                    let dst = &mut out[t * d..(t + 1) * d];
                    match w {
                        Some(w) => {
                            crate::vec_ops::rms_norm_fused_f32(d, dst, row, w.data_f32(), *eps)
                        }
                        None => crate::vec_ops::rms_norm_f32(d, dst, row, *eps),
                    }
                }
                Ok(())
            }
            Op::QkNorm { hd, nh, eps } => {
                // Per-head norm: the flat [nt*nh*hd] buffer viewed as a
                // contiguous [nt*nh, hd] matrix (t*(nh*hd) + h*hd == (t*nh+h)*hd),
                // so this is the same RMSNorm row loop with d = hd, n = nt*nh.
                let w = match &node.meta {
                    NodeMeta::Norm(m) => m.weight_name.as_ref().and_then(|n| self.weights.get(n)),
                    _ => None,
                };
                let _ = nh; // row count is derived from the buffer length
                let d = *hd;
                let n = out.len() / d;
                for t in 0..n {
                    let row = &ins[0][t * d..(t + 1) * d];
                    let dst = &mut out[t * d..(t + 1) * d];
                    match w {
                        Some(w) => {
                            crate::vec_ops::rms_norm_fused_f32(d, dst, row, w.data_f32(), *eps)
                        }
                        None => crate::vec_ops::rms_norm_f32(d, dst, row, *eps),
                    }
                }
                Ok(())
            }
            Op::MatMul { .. } => {
                let meta = match &node.meta {
                    NodeMeta::MatMul(m) => m,
                    other => return Err(format!("matmul node missing MatMulMeta: {other:?}")),
                };
                let w = self
                    .weights
                    .get(&meta.weight_name)
                    .ok_or_else(|| format!("weight '{}' not registered", meta.weight_name))?;
                // llama.cpp/GGUF convention: metadata [in, out], memory [out][in]
                let od = w.shape[1] as usize; // output dim
                let id = w.shape[0] as usize; // input dim
                let nt = node.out_shape[1];
                if w.ttype == crate::tensor::TensorType::F32 {
                    // plain f32 matmul: out[t*od+o] = dot(w[o], x[t])
                    crate::vec_ops::mat_mul_f32(od, nt, id, out, w.data_f32(), ins[0]);
                } else if w.ttype == crate::tensor::TensorType::F16 {
                    // F6: f16 weights are decoded a row at a time (no f32 copy).
                    crate::vec_ops::mat_mul_f16(od, nt, id, out, w.data(), ins[0]);
                } else if w.ttype == crate::tensor::TensorType::BF16 {
                    // #142: the bf16 twin — a row decode (`bits << 16`) then the
                    // plain f32 dot; still no f32 copy of the weight.
                    crate::vec_ops::mat_mul_bf16(od, nt, id, out, w.data(), ins[0]);
                } else {
                    // quantized weight × f32 activations (Q8_0-quantized on the fly)
                    kernel::cpu_quant_matmul_f32(w, ins[0], out, od, id, nt);
                }
                if let Some(bname) = &meta.bias_name {
                    let b = self
                        .weights
                        .get(bname)
                        .ok_or_else(|| format!("bias '{}' not registered", bname))?;
                    let bd = b.data_f32();
                    for t in 0..nt {
                        let base = t * od;
                        for i in 0..od.min(bd.len()) {
                            out[base + i] += bd[i];
                        }
                    }
                }
                Ok(())
            }
            Op::GetRows => {
                let meta = match &node.meta {
                    NodeMeta::Embed(m) => Some(m),
                    NodeMeta::None => None, // generic row selection: x[ids]
                    other => return Err(format!("get_rows node with unexpected meta: {other:?}")),
                };
                let Some(meta) = meta else {
                    // generic gather: out[t*n+i] = x[ids[t]*n+i]
                    let n_embd = node.out_shape[0];
                    let nt = node.out_shape[1];
                    for t in 0..nt {
                        let id = ins[1][t].to_bits() as usize; // ids (I32)
                        if (id + 1) * n_embd > ins[0].len() {
                            return Err(format!("get_rows index {id} out of range"));
                        }
                        out[t * n_embd..(t + 1) * n_embd]
                            .copy_from_slice(&ins[0][id * n_embd..(id + 1) * n_embd]);
                    }
                    return Ok(());
                };
                let w = self
                    .weights
                    .get(&meta.weight_name)
                    .ok_or_else(|| format!("embedding '{}' not registered", meta.weight_name))?;
                let n_embd = node.out_shape[0];
                let nt = node.out_shape[1];
                let ids: Vec<u32> = (0..nt).map(|t| ins[0][t].to_bits()).collect();
                if w.ttype == crate::tensor::TensorType::F32 {
                    let wf = w.data_f32();
                    let vocab = w.shape[1] as usize;
                    for (t, &id) in ids.iter().enumerate() {
                        if (id as usize) >= vocab {
                            return Err(format!("embedding id {id} >= vocab {vocab}"));
                        }
                        let src = &wf[id as usize * n_embd..(id as usize + 1) * n_embd];
                        out[t * n_embd..(t + 1) * n_embd].copy_from_slice(src);
                    }
                } else if w.ttype == crate::tensor::TensorType::F16 {
                    // F6: f16 embedding rows decoded in place.
                    let wd = w.data();
                    let vocab = w.shape[1] as usize;
                    for (t, &id) in ids.iter().enumerate() {
                        if (id as usize) >= vocab {
                            return Err(format!("embedding id {id} >= vocab {vocab}"));
                        }
                        let base = id as usize * n_embd * 2;
                        for j in 0..n_embd {
                            out[t * n_embd + j] = crate::block::fp16_to_f32(u16::from_le_bytes([
                                wd[base + 2 * j],
                                wd[base + 2 * j + 1],
                            ]));
                        }
                    }
                } else if w.ttype == crate::tensor::TensorType::BF16 {
                    // #142: bf16 embedding rows decoded in place (`bits << 16`).
                    let wd = w.data();
                    let vocab = w.shape[1] as usize;
                    for (t, &id) in ids.iter().enumerate() {
                        if (id as usize) >= vocab {
                            return Err(format!("embedding id {id} >= vocab {vocab}"));
                        }
                        let base = id as usize * n_embd * 2;
                        for j in 0..n_embd {
                            out[t * n_embd + j] = crate::block::bf16_to_f32(u16::from_le_bytes([
                                wd[base + 2 * j],
                                wd[base + 2 * j + 1],
                            ]));
                        }
                    }
                } else {
                    // quantized embedding: shared dequantization
                    crate::kernel::embed_tokens(&ids, w, out, n_embd);
                }
                Ok(())
            }
            Op::RoPE { style } => {
                let meta = match &node.meta {
                    NodeMeta::Rope(m) => m,
                    other => return Err(format!("rope node missing RoPEMeta: {other:?}")),
                };
                let nh = meta.n_head;
                let hd = meta.hd;
                let nt = node.out_shape[1];
                // positions are I32 bit patterns in ins[1]
                let pos: Vec<usize> = (0..nt).map(|t| ins[1][t].to_bits() as usize).collect();
                out.copy_from_slice(ins[0]);
                cpu_rope(out, &pos, nh, hd, meta.freq_base, meta.freq_scale, *style);
                Ok(())
            }
            Op::Softmax { dim } => {
                if *dim == 0 || *dim == 1 {
                    let mut mx = f32::NEG_INFINITY;
                    for &v in ins[0].iter() {
                        if v > mx {
                            mx = v;
                        }
                    }
                    out.copy_from_slice(ins[0]);
                    let s_in = out.to_vec();
                    // `vec_soft_max_f32` writes exp(x - max) and RETURNS the sum
                    // (same contract the attention kernels use); without this
                    // division the op produced unnormalised values. Caught by
                    // the op matrix (ticket A1) — no architecture emits a
                    // standalone Softmax node, so nothing else exercised it.
                    let sum = crate::vec_ops::vec_soft_max_f32(out.len(), out, &s_in, mx);
                    let inv = (1.0 / sum) as f32;
                    for v in out.iter_mut() {
                        *v *= inv;
                    }
                    Ok(())
                } else {
                    Err(format!("Softmax dim {dim} not supported (Phase 2)"))
                }
            }
            Op::SwiGLU => {
                // silu(gate) * up, one pass: bit-identical to the old
                // vec_silu_f32 + vec_mul_f32 pair (same formula and
                // per-element order) without its full-size temp buffer.
                crate::vec_ops::vec_swiglu_f32(out.len(), out, ins[0], ins[1]);
                Ok(())
            }
            Op::KvcacheStore { .. } => unreachable!("KV store handled in the dedicated path"),
            Op::KvcacheLoad { .. } => Ok(()), // view of the K region

            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                // D1: a view does not own its output — the allocator mapped it
                // onto its parent's buffer, so `out` and `ins[0]` are the same
                // allocation and there is nothing to copy. The assert is the
                // guard against a future non-view use of these ops silently
                // passing through.
                debug_assert!(
                    node.view.is_some(),
                    "{}: view-like op without a view alias (allocator should have refused)",
                    node.name
                );
                debug_assert_eq!(out.as_ptr(), ins[0].as_ptr(), "view is not aliased");
                Ok(())
            }
            Op::Attn { .. } => {
                let meta = match &node.meta {
                    NodeMeta::Attn(m) => m,
                    other => return Err(format!("attn node missing AttnMeta: {other:?}")),
                };
                let nt = node.out_shape[1];
                let nkt = meta.nkt;
                let row_elems = self.kv_format.row_elems(nkt);
                // K and V are separate persistent regions per layer
                let (k_id, v_id) = kv_pair
                    .ok_or_else(|| format!("KV regions for layer {} not allocated", meta.layer))?;
                let k_slice: &[f32] = if k_id < out_buf.id {
                    &before[k_id]
                } else {
                    &after[k_id - out_buf.id - 1]
                };
                let v_slice: &[f32] = if v_id < out_buf.id {
                    &before[v_id]
                } else {
                    &after[v_id - out_buf.id - 1]
                };
                // C4: the region holds `n_ctx * row_elems` words, not `n_ctx * nkt`.
                let n_ctx = k_slice.len() / row_elems;
                // E1/C8b S2: the allowed cells are an explicit input, not a bound
                // derived from `positions` — that is what lets a batch hold several
                // sequences, and (S2) what lets one query's window be a list of runs.
                // The allocator resolved it from the cell store; here it is only
                // validated and decoded.
                let (runs, off) = decode_window(ins[3], nt, n_ctx)?;
                if self.kv_format.is_packed() {
                    // C4 S2: read the packed blocks directly — the K score is a
                    // Q8_0 × Q8_0 dot against the quantized query, V accumulates out
                    // of the cell. No scratch, no second pass.
                    // `MINFER_NO_FUSED_Q8_KV` keeps S1's dequantizing path for the
                    // A/B standing rule 3 asks for (presence-checked).
                    if meta.hd_kv % kvformat::Q8_0_BLOCK == 0 && fused_q8_kv_enabled() {
                        return cpu_gqa_attn_runs_q8(
                            ins[0],
                            k_slice,
                            v_slice,
                            &runs,
                            &off,
                            nt,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.hd_kv,
                            nkt,
                            out,
                            meta.scale,
                        );
                    }
                    // A KV head narrower than one Q8_0 block cannot be read
                    // block-aligned (no supported architecture emits one — `check_width`
                    // already requires `nkt % 32 == 0`): dequantize the union of this
                    // batch's windows into the reusable scratch and run the unchanged
                    // f32 kernel over it. The runs stay cell-indexed, only rebased to
                    // the scratch's start.
                    let lo = runs.iter().map(|r| r.0).min().unwrap_or(0);
                    let hi = runs.iter().map(|r| r.0 + r.1).max().unwrap_or(lo);
                    let rows = hi.saturating_sub(lo);
                    self.kv_scratch_k.resize(rows * nkt, 0.0);
                    self.kv_scratch_v.resize(rows * nkt, 0.0);
                    kvformat::unpack_q8_0_cells(k_slice, nkt, lo, rows, &mut self.kv_scratch_k);
                    kvformat::unpack_q8_0_cells(v_slice, nkt, lo, rows, &mut self.kv_scratch_v);
                    let rebased: Vec<(usize, usize)> =
                        runs.iter().map(|&(cell, len)| (cell - lo, len)).collect();
                    cpu_gqa_attn_runs(
                        ins[0],
                        &self.kv_scratch_k,
                        &self.kv_scratch_v,
                        &rebased,
                        &off,
                        nt,
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.hd_kv,
                        nkt,
                        out,
                        meta.scale,
                    )?;
                    return Ok(());
                }
                cpu_gqa_attn_runs(
                    ins[0],
                    k_slice,
                    v_slice,
                    &runs,
                    &off,
                    nt,
                    meta.n_head,
                    meta.n_head_kv,
                    meta.hd,
                    meta.hd_kv,
                    nkt,
                    out,
                    meta.scale,
                )?;
                Ok(())
            }
            Op::BatchMatMul
            | Op::FusedQKV { .. }
            | Op::QkvBiasRopeStore { .. }
            | Op::FusedQkvNorm { .. }
            | Op::FusedFFN => Err(format!(
                "op {:?} unsupported on CPU (fusion not enabled for it)",
                node.op
            )),
        }
    }

    fn copy_cells(
        &mut self,
        dst: BufRef,
        src: BufRef,
        dst_row: usize,
        src_row: usize,
        rows: usize,
        elems_per_cell: usize,
    ) -> Result<(), String> {
        if dst.id != src.id {
            return Err(format!(
                "copy_cells: CPU moves cells within one buffer ({} -> {})",
                src.id, dst.id
            ));
        }
        let buf = self
            .buffers
            .get_mut(dst.id)
            .ok_or_else(|| format!("copy_cells: unknown buffer {}", dst.id))?;
        let src_start = src_row
            .checked_mul(elems_per_cell)
            .ok_or("copy_cells: src range overflows")?;
        let dst_start = dst_row
            .checked_mul(elems_per_cell)
            .ok_or("copy_cells: dst range overflows")?;
        let len = rows
            .checked_mul(elems_per_cell)
            .ok_or("copy_cells: length overflows")?;
        if src_start + len > buf.len() || dst_start + len > buf.len() {
            return Err(format!(
                "copy_cells: {} rows x {elems_per_cell} elements at src {src_start} / dst \
                 {dst_start} does not fit a {} element buffer",
                rows,
                buf.len()
            ));
        }
        // `copy_within` is a memmove: correct for overlapping ranges, whichever
        // way they overlap, which is what the contract needs.
        buf.copy_within(src_start..src_start + len, dst_start);
        Ok(())
    }

    fn read_host(&self, id: usize) -> Option<&[f32]> {
        self.buffers.get(id).map(|b| b.as_slice())
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let b = self
            .buffers
            .get(id)
            .ok_or_else(|| format!("no buffer {id}"))?;
        if b.len() != data.len() {
            return Err(format!(
                "buffer {id}: expected {} elements, got {}",
                b.len(),
                data.len()
            ));
        }
        self.write_host_window(id, 0, data)
    }

    /// E4 S2: a pooled activation buffer is rounded to its size class, so a node
    /// writes its logical window into a buffer that may be longer. Bounds, not
    /// equality.
    fn write_host_window(&mut self, id: usize, offset: usize, data: &[f32]) -> Result<(), String> {
        let b = self
            .buffers
            .get_mut(id)
            .ok_or_else(|| format!("no buffer {id}"))?;
        let end = offset.checked_add(data.len()).ok_or_else(|| {
            format!(
                "buffer {id}: offset {offset} + {} elements overflows",
                data.len()
            )
        })?;
        if end > b.len() {
            return Err(format!(
                "buffer {id}: writing {} elements at offset {offset} runs past the {}-element pool buffer",
                data.len(),
                b.len()
            ));
        }
        b[offset..end].copy_from_slice(data);
        Ok(())
    }

    fn synchronize(&mut self) {}
}

/// RoPE per-head (same math as forward.rs's apply_rope).
pub(crate) fn cpu_rope(
    x: &mut [f32],
    pos: &[usize],
    nh: usize,
    hd: usize,
    freq_base: f32,
    freq_scale: f32,
    style: RopeStyle,
) {
    let half = hd / 2;
    let mut freqs = [0.0f32; 128];
    for i in 0..half {
        freqs[i] = freq_scale / freq_base.powf((2 * i) as f32 / hd as f32);
    }
    for t in 0..pos.len() {
        let p = pos[t] as f32;
        for h in 0..nh {
            let b = t * nh * hd + h * hd;
            for i in 0..half {
                let th = p * freqs[i];
                let (sn, cs) = th.sin_cos();
                let (i0, i1) = match style {
                    RopeStyle::NonInterleaved => (b + i, b + i + half),
                    RopeStyle::Interleaved => (b + 2 * i, b + 2 * i + 1),
                };
                let (x0, x1) = (x[i0], x[i1]);
                x[i0] = x0 * cs - x1 * sn;
                x[i1] = x0 * sn + x1 * cs;
            }
        }
    }
}

/// GQA attention (same math as forward.rs's gqa_attn).
/// The causal span for a single sequence, in the `attn_span` layout E1 defines:
/// token `t` may see the cells `[0, pos[t] + 1)`. Test/reference helper — the
/// model path resolves the span from the KV store's ownership instead.
/// Test-only (#238): driven by `graph::cuda_backend::tests::kv::cuda_rope_kv_attn_roundtrip`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn causal_span(pos: &[usize]) -> Vec<(usize, usize)> {
    pos.iter().map(|&p| (0, p + 1)).collect()
}

/// Decode a query-window input into `(cell, len)` runs plus per-query offsets.
///
/// Two layouts reach the kernel, and their **sizes tell them apart** (an input's
/// size is graph topology, so this is fixed at build time, not a per-step choice):
/// `attn_span` carries one `(lo, hi)` per query (E1), and `kv_map` carries
/// `KV_MAP_MAX_SPANS` `(cell, len)` runs per query, zero-padded (C8b S2 — what a
/// sequence reads when it shares a prefix: the shared runs plus its own). Both are
/// validated here, and a length that is neither is an error rather than a guess.
fn decode_window(
    input: &[f32],
    nt: usize,
    n_ctx: usize,
) -> Result<(Vec<(usize, usize)>, Vec<usize>), String> {
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(nt);
    let mut off: Vec<usize> = Vec::with_capacity(nt + 1);
    off.push(0);
    if input.len() == 2 * nt {
        for t in 0..nt {
            let (lo, hi) = (
                input[t].to_bits() as usize,
                input[nt + t].to_bits() as usize,
            );
            if hi > n_ctx || hi <= lo {
                return Err(format!(
                    "attn: query {t} has span [{lo}, {hi}) — outside the {n_ctx}-cell arena, \
                     or empty (a span input that was never filled is all zeros)"
                ));
            }
            runs.push((lo, hi - lo));
            off.push(runs.len());
        }
        return Ok((runs, off));
    }
    let k = crate::graph::kvcache::KV_MAP_MAX_SPANS;
    if input.len() != nt * k * 2 {
        return Err(format!(
            "attn: window input has {} values, expected {} (attn_span: one (lo, hi) per query) \
             or {} (kv_map: {k} (cell, len) runs per query)",
            input.len(),
            2 * nt,
            nt * k * 2
        ));
    }
    for t in 0..nt {
        let mut any = false;
        for s in 0..k {
            let at = (t * k + s) * 2;
            let (cell, len) = (
                input[at].to_bits() as usize,
                input[at + 1].to_bits() as usize,
            );
            if len == 0 {
                continue; // padding slot
            }
            if cell + len > n_ctx {
                return Err(format!(
                    "attn: query {t} run {s} is [{cell}, {}) — past the {n_ctx}-cell arena",
                    cell + len
                ));
            }
            runs.push((cell, len));
            any = true;
        }
        if !any {
            return Err(format!(
                "attn: query {t} has no cell runs (a kv_map input that was never filled is all \
                 zeros)"
            ));
        }
        off.push(runs.len());
    }
    Ok((runs, off))
}

/// The one-range form of [`cpu_gqa_attn_runs`], kept for the reference callers
/// (tests and the causal fixture): `span[t] = (lo, hi)` is one run of `hi - lo`
/// cells. The Op::Attn path decodes the window input itself, so it can carry a
/// list of runs (C8b S2) and not only a contiguous range.
/// Test-only (#238): driven by `graph::cpu_backend::tests::a_window_split_into_runs_gathers_like_one_range`; `#[cfg(test)]` keeps it out of production builds.
#[cfg(test)]
pub(crate) fn cpu_gqa_attn(
    q: &[f32],
    ka: &[f32],
    va: &[f32],
    span: &[(usize, usize)],
    nt: usize,
    nh: usize,
    nk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: &mut [f32],
    scale: f32,
) -> Result<(), String> {
    if span.len() != nt {
        return Err(format!(
            "attention: {} spans for {nt} query tokens",
            span.len()
        ));
    }
    let mut runs: Vec<(usize, usize)> = Vec::with_capacity(nt);
    let mut off: Vec<usize> = Vec::with_capacity(nt + 1);
    off.push(0);
    for &(lo, hi) in span {
        if hi < lo {
            return Err(format!("attention: span [{lo}, {hi}) is empty"));
        }
        runs.push((lo, hi - lo));
        off.push(runs.len());
    }
    cpu_gqa_attn_runs(
        q, ka, va, &runs, &off, nt, nh, nk, hd, hd_kv, nkt, out, scale,
    )
}

/// GQA attention over each query's **list of `(cell, len)` runs** (C8b S2), with
/// `off[t]..off[t + 1]` naming query `t`'s runs — the union of the runs is the
/// query's window, in order.
pub(crate) fn cpu_gqa_attn_runs(
    q: &[f32],
    ka: &[f32],
    va: &[f32],
    runs: &[(usize, usize)],
    off: &[usize],
    nt: usize,
    nh: usize,
    nk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: &mut [f32],
    scale: f32,
) -> Result<(), String> {
    if hd < hd_kv {
        return Err(format!(
            "Q head dim ({hd}) must be >= KV head dim ({hd_kv})"
        ));
    }

    // Heads are independent: parallelize over head ranges on the CPU pool.
    // Each worker computes its own head range with a private scores buffer,
    // so the output is bit-identical to the single-threaded order (no
    // cross-head reduction).
    if off.len() != nt + 1 {
        return Err(format!(
            "attention: {} run offsets for {nt} query tokens",
            off.len()
        ));
    }
    let max_vl = (0..nt)
        .map(|t| (off[t]..off[t + 1]).map(|i| runs[i].1).sum::<usize>())
        .max();
    let ctx = AttnCtx {
        q: q.as_ptr(),
        ka: ka.as_ptr(),
        va: va.as_ptr(),
        runs: runs.as_ptr(),
        off: off.as_ptr(),
        nt,
        max_vl: max_vl.unwrap_or(0),
        nh,
        hk: nk,
        hd,
        hd_kv,
        nkt,
        out: out.as_mut_ptr(),
        scale,
    };
    if crate::kernel::cpu_threads() <= 1 || nh < 2 {
        unsafe { attn_heads(&ctx as *const _ as *const (), 0, nh) };
        return Ok(());
    }
    crate::kernel::par_for(nh, &ctx as *const _ as *const (), attn_heads);
    Ok(())
}

/// Raw context for the pooled attention worker (valid for the par_for call).
struct AttnCtx {
    q: *const f32,
    ka: *const f32,
    va: *const f32,
    /// Per-query window as `(cell, len)` runs, `off[t]..off[t + 1]` per query
    /// (E1's one range is one run; C8b S2 makes several possible).
    runs: *const (usize, usize),
    off: *const usize,
    nt: usize,
    /// Longest window in the batch: sizes the per-head score scratch.
    max_vl: usize,
    nh: usize,
    hk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: *mut f32,
    scale: f32,
}

/// Compute attention for heads [h0, h1). SAFETY: `ctx` points to a live
/// `AttnCtx` (whose `span` outlives the call) for the duration; each head h
/// writes the disjoint output range out[t*ne_q + h*hd .. +hd], so concurrent
/// calls over disjoint head ranges cannot race.
unsafe fn attn_heads(ctx: *const (), h0: usize, h1: usize) {
    let c = &*(ctx as *const AttnCtx);
    let gqa = c.nh / c.hk;
    let ne_q = c.nh * c.hd;
    let mut scrs = vec![0.0f32; c.max_vl.max(1)];
    for h in h0..h1 {
        let hk = h / gqa;
        for t in 0..c.nt {
            let qs = t * ne_q + h * c.hd;
            let (o0, o1) = (*c.off.add(t), *c.off.add(t + 1));
            let vl: usize = (o0..o1).map(|i| (*c.runs.add(i)).1).sum();
            let mut mx = f32::NEG_INFINITY;
            let mut i = 0usize;
            for r in o0..o1 {
                let (cell, len) = *c.runs.add(r);
                for kv in cell..cell + len {
                    let ks = kv * c.nkt + hk * c.hd_kv;
                    let s = crate::vec_ops::vec_dot_f32(
                        c.hd_kv,
                        std::slice::from_raw_parts(c.q.add(qs), c.hd_kv),
                        std::slice::from_raw_parts(c.ka.add(ks), c.hd_kv),
                    ) * c.scale;
                    scrs[i] = s;
                    i += 1;
                    if s > mx {
                        mx = s;
                    }
                }
            }
            // Softmax and accumulate over the token's OWN window `[lo, hi)`, the
            // one the span input names, rather than over a batch-wide range.
            // Padding the reduction and softmaxing over it makes every
            // reduction's *length* depend on how many tokens the batch holds,
            // which makes a token's result depend on `nt` (measured: a 6-token
            // and a 13-token batch diverge from layer 3 on). With the window a
            // function of the token alone, incremental prefill and decode
            // reproduce a single-shot prefill bitwise — and with one sequence
            // the window is exactly `[0, pos + 1)`, which is what the old
            // derivation produced.
            let sm = crate::vec_ops::vec_soft_max_inplace_f32(vl, &mut scrs, mx);
            let is = (1.0 / sm) as f32;
            crate::vec_ops::vec_scale_f32(vl, &mut scrs, is);
            let os = t * ne_q + h * c.hd;
            let out_slice = std::slice::from_raw_parts_mut(c.out.add(os), c.hd);
            out_slice.fill(0.0);
            let vs_base = hk * c.hd_kv;
            let mut i = 0usize;
            for r in o0..o1 {
                let (cell, len) = *c.runs.add(r);
                for kv in cell..cell + len {
                    crate::vec_ops::vec_muladd_f32(
                        c.hd_kv,
                        std::slice::from_raw_parts_mut(c.out.add(os), c.hd_kv),
                        std::slice::from_raw_parts(c.va.add(kv * c.nkt + vs_base), c.hd_kv),
                        scrs[i],
                    );
                    i += 1;
                }
            }
        }
    }
}

// ---- the fused packed (Q8_0) read path (C4 S2) -----------------------------

/// Whether the fused packed read is enabled (C4 S2).
///
/// `MINFER_NO_FUSED_Q8_KV` (presence-checked, like the other `MINFER_NO_*` knobs)
/// keeps S1's dequantize-into-a-scratch path available, which is the A/B standing
/// rule 3 asks for: the two paths answer within a named class, not bitwise, because
/// the fused one quantizes the query too.
fn fused_q8_kv_enabled() -> bool {
    std::env::var_os("MINFER_NO_FUSED_Q8_KV").is_none()
}

/// Attention over **packed Q8_0** K/V regions, reading the blocks directly.
///
/// C4 S1 dequantized the union of the batch's windows into a reusable f32 scratch
/// (two allocations and a write+read pass per attention node, per layer, per
/// forward) and ran the unchanged f32 kernel over it. S2 removes all of that[^bw]:
///
/// - the K score is `dot_q8_0_q8_0` between the **Q8_0-quantized query row** and the
///   stored K blocks — llama.cpp's form for a quantized cache, and the reason a
///   packed cache is worth having (34 bytes per 32 elements, SDOT in the NEON arm);
/// - V accumulates straight out of the packed cell, block scale included, with no
///   f32 staging at all.
///
/// [^bw]: The **query** is now quantized too, so a score carries the query's Q8_0
/// error on top of the stored cell's — a new term in the tolerance class this path
/// is measured against the f32 cache with (see `docs/ARCHITECTURE-EXECUTION-PLAN.md`
/// §5 C4 S2 for the measured number). The V side has no new term: it evaluates the
/// same `scale * quant` product the scratch path's dequantizer wrote.
///
/// `hd` must be a multiple of [`kvformat::Q8_0_BLOCK`] so a KV head's slice starts on
/// a block boundary; the caller keeps the S1 scratch path for anything else.
pub(crate) fn cpu_gqa_attn_runs_q8(
    q: &[f32],
    ka: &[f32],
    va: &[f32],
    runs: &[(usize, usize)],
    off: &[usize],
    nt: usize,
    nh: usize,
    nk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: &mut [f32],
    scale: f32,
) -> Result<(), String> {
    if hd < hd_kv {
        return Err(format!(
            "Q head dim ({hd}) must be >= KV head dim ({hd_kv})"
        ));
    }
    if hd_kv % kvformat::Q8_0_BLOCK != 0 {
        return Err(format!(
            "fused Q8_0 attention needs a KV head width that is a multiple of {} (got {hd_kv}): \
             a packed block covers {} elements and a head slice must not straddle one",
            kvformat::Q8_0_BLOCK,
            kvformat::Q8_0_BLOCK
        ));
    }
    if off.len() != nt + 1 {
        return Err(format!(
            "attention: {} run offsets for {nt} query tokens",
            off.len()
        ));
    }
    let max_vl = (0..nt)
        .map(|t| (off[t]..off[t + 1]).map(|i| runs[i].1).sum::<usize>())
        .max();
    let ctx = AttnQ8Ctx {
        q: q.as_ptr(),
        ka: ka.as_ptr(),
        ka_len: ka.len(),
        va: va.as_ptr(),
        va_len: va.len(),
        runs: runs.as_ptr(),
        off: off.as_ptr(),
        nt,
        max_vl: max_vl.unwrap_or(0),
        nh,
        hk: nk,
        hd,
        hd_kv,
        nkt,
        out: out.as_mut_ptr(),
        scale,
    };
    if crate::kernel::cpu_threads() <= 1 || nh < 2 {
        unsafe { attn_heads_q8(&ctx as *const _ as *const (), 0, nh) };
        return Ok(());
    }
    crate::kernel::par_for(nh, &ctx as *const _ as *const (), attn_heads_q8);
    Ok(())
}

/// Raw context for the pooled packed-attention worker (S2). Same fields as
/// [`AttnCtx`] plus the two region lengths the byte views need.
struct AttnQ8Ctx {
    q: *const f32,
    /// Packed K region, as pool words (read as bytes through
    /// [`kvformat::region_bytes`]).
    ka: *const f32,
    ka_len: usize,
    /// Packed V region, as pool words.
    va: *const f32,
    va_len: usize,
    runs: *const (usize, usize),
    off: *const usize,
    nt: usize,
    max_vl: usize,
    nh: usize,
    hk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: *mut f32,
    scale: f32,
}

/// Fused packed attention for heads [h0, h1). SAFETY: as [`attn_heads`] — `ctx`
/// points at a live [`AttnQ8Ctx`] for the call's duration, the head ranges are
/// disjoint, and each head only reads.
unsafe fn attn_heads_q8(ctx: *const (), h0: usize, h1: usize) {
    let c = &*(ctx as *const AttnQ8Ctx);
    let gqa = c.nh / c.hk;
    let ne_q = c.nh * c.hd;
    let ka = std::slice::from_raw_parts(c.ka, c.ka_len);
    let va = std::slice::from_raw_parts(c.va, c.va_len);
    let kbytes = kvformat::region_bytes(ka);
    let block_bytes = (c.hd_kv / kvformat::Q8_0_BLOCK) * kvformat::Q8_0_BLOCK_BYTES;
    // One Q8_0 block row per worker (not per query): the query is re-quantized per
    // (head, token) into this buffer and the K dot reads it back.
    let mut qq = vec![0u8; block_bytes];
    let mut scrs = vec![0.0f32; c.max_vl.max(1)];
    for h in h0..h1 {
        let hk = h / gqa;
        let head_elem = hk * c.hd_kv;
        for t in 0..c.nt {
            let qs = t * ne_q + h * c.hd;
            let qrow = std::slice::from_raw_parts(c.q.add(qs), c.hd_kv);
            crate::quants::quantize_row_q8_0_buf(qrow, 1, c.hd_kv, &mut qq);
            let (o0, o1) = (*c.off.add(t), *c.off.add(t + 1));
            let mut mx = f32::NEG_INFINITY;
            let mut i = 0usize;
            for r in o0..o1 {
                let (cell, len) = *c.runs.add(r);
                for kv in cell..cell + len {
                    let at = kvformat::q8_0_cell_offset(c.nkt, kv, head_elem);
                    let s =
                        crate::quants::dot_q8_0_q8_0(&qq, &kbytes[at..at + block_bytes]) * c.scale;
                    scrs[i] = s;
                    i += 1;
                    if s > mx {
                        mx = s;
                    }
                }
            }
            // The softmax and the accumulation run over the token's OWN window, in
            // the same order and with the same ops as the scratch path — only the
            // source of the scores and of the V rows changed.
            let vl = i;
            let sm = crate::vec_ops::vec_soft_max_inplace_f32(vl, &mut scrs, mx);
            let is = (1.0 / sm) as f32;
            crate::vec_ops::vec_scale_f32(vl, &mut scrs, is);
            let os = t * ne_q + h * c.hd;
            let out_slice = std::slice::from_raw_parts_mut(c.out.add(os), c.hd);
            out_slice.fill(0.0);
            let mut i = 0usize;
            for r in o0..o1 {
                let (cell, len) = *c.runs.add(r);
                for kv in cell..cell + len {
                    kvformat::accumulate_q8_0_row(
                        va,
                        c.nkt,
                        kv,
                        head_elem,
                        scrs[i],
                        &mut out_slice[..c.hd_kv],
                    );
                    i += 1;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;
