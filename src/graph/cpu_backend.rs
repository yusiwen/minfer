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

    /// Look up a registered weight tensor (test / debug helper).
    #[allow(dead_code)]
    pub fn weight(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// E4 S3: buffers this pool has created (never reused from the slot table).
    /// (Test / accounting helper.)
    #[allow(dead_code)]
    pub fn alloc_count(&self) -> usize {
        self.allocs
    }

    /// Pool size (for tests).
    #[allow(dead_code)]
    /// D1: the `f32` window a [`BufRef`] names — the whole buffer for an owning
    /// node (offset 0, len = its element count), the parent's bytes at the
    /// window for a view.
    fn window(&self, r: BufRef) -> &[f32] {
        &self.buffers[r.id][r.offset..r.offset + r.len]
    }

    pub fn pool_len(&self) -> usize {
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
/// The registry carries this as [`super::registry::BackendCaps::supports_op`],
/// which cannot take a `&self`; the trait method below forwards to it, so the
/// registry's answer and the trait's answer are the same code.
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
            supports_op,
            supports_fused,
            supports_attn_span: SUPPORTS_ATTN_SPAN,
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
    fn name(&self) -> &str {
        "cpu"
    }

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
#[allow(dead_code)]
pub fn causal_span(pos: &[usize]) -> Vec<(usize, usize)> {
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
mod tests {
    use super::*;

    /// C8b S2: a window given as **several runs** gathers exactly what the same cells
    /// given as one range gather — the equivalence the map path rests on — and the
    /// runs are visited in the order the store lists them.
    #[test]
    fn a_window_split_into_runs_gathers_like_one_range() {
        let (nh, nk, hd, nkt, n_ctx, nt) = (2usize, 1usize, 4usize, 4usize, 6usize, 2usize);
        let k: Vec<f32> = (0..n_ctx * nkt).map(|i| i as f32 * 0.05 + 0.1).collect();
        let v: Vec<f32> = (0..n_ctx * nkt).map(|i| i as f32 * -0.03 + 0.7).collect();
        let q: Vec<f32> = (0..nt * nh * hd).map(|i| i as f32 * 0.11 - 0.4).collect();
        let mut one = vec![0.0f32; nt * nh * hd];
        let mut split = vec![0.0f32; nt * nh * hd];
        // Query 0 sees cells [0, 3); query 1 sees [1, 6).
        let span = vec![(0usize, 3usize), (1, 6)];
        cpu_gqa_attn(&q, &k, &v, &span, nt, nh, nk, hd, hd, nkt, &mut one, 0.5).unwrap();
        // The same two windows as runs: [0, 2) + [2, 1), and [1, 3) + [4, 2).
        let runs = vec![(0usize, 2usize), (2, 1), (1, 3), (4, 2)];
        let off = vec![0usize, 2, 4];
        cpu_gqa_attn_runs(
            &q, &k, &v, &runs, &off, nt, nh, nk, hd, hd, nkt, &mut split, 0.5,
        )
        .unwrap();
        assert_eq!(one, split, "the window is the union of its runs, in order");
        // The `kv_map` layout decodes back to exactly those runs, and a padded slot
        // (length 0) is skipped.
        let k_max = crate::graph::kvcache::KV_MAP_MAX_SPANS;
        let mut flat = vec![0.0f32; nt * k_max * 2];
        for (i, &(cell, len)) in runs.iter().enumerate() {
            let at = (i / 2 * k_max + i % 2) * 2;
            flat[at] = f32::from_bits(cell as u32);
            flat[at + 1] = f32::from_bits(len as u32);
        }
        let (decoded, offsets) = decode_window(&flat, nt, n_ctx).unwrap();
        assert_eq!((decoded, offsets), (runs, off));
        // A layout that is neither form is refused, not guessed.
        let err = decode_window(&flat[1..], nt, n_ctx).unwrap_err();
        assert!(err.contains("expected"), "got: {err}");
    }

    /// Plan §14 row 9's exoneration of the ops, as a gate.
    ///
    /// The minimal graph — `q`/`k`/`v` inputs → rope → store → attn, no model —
    /// is run at cell 0, 1 and 8 with the same data, the same relative window and
    /// the explicit span in every run, swept over shapes including the model's own
    /// `(nh = 14, nk = 2, hd = 64)`. The only difference between the runs is the
    /// rotation's own rounding, so the outputs must agree to well below anything a
    /// model could amplify: measured <= 1.2e-7 at the widest shape, asserted < 1e-6.
    ///
    /// This is what rules the kernels out as the source of the model-level offset
    /// divergence (§14 row 9 finds it entering at layer 0's attention output and
    /// still open): the same op, shape and window structure are exact in isolation.
    #[test]
    fn the_minimal_attention_graph_is_offset_invariant_to_rounding() {
        use crate::graph::ops::{AttnMeta, AttnMode, RoPEMeta};
        use crate::graph::DType;
        use crate::vec_ops::RopeStyle;

        let n = 5usize;
        let n_ctx = 64usize;
        let freq_base = 10_000.0f32;
        let freq_scale = 1.0f32;

        for (nh, nk, hd) in [
            (2usize, 2usize, 4usize),
            (2, 2, 64),
            (14, 2, 4),
            (14, 2, 64),
            (14, 2, 128),
        ] {
            let nkt = nk * hd;
            let mut b = GraphBuilder::new();
            b.set_explicit_span(true);
            let pos = b.input("positions", [n, 1, 1, 1], DType::I32);
            let qq = b.input("q", [nh * hd, n, 1, 1], DType::F32);
            let kk = b.input("k", [nkt, n, 1, 1], DType::F32);
            let vv = b.input("v", [nkt, n, 1, 1], DType::F32);
            let rope = |nh_: usize| RoPEMeta {
                freq_base,
                freq_scale,
                n_head: nh_,
                hd,
            };
            let q_r = b.rope(qq, pos, RopeStyle::NonInterleaved, rope(nh));
            let k_r = b.rope(kk, pos, RopeStyle::NonInterleaved, rope(nk));
            let _store = b.kvcache_store(0, k_r, vv, n_ctx);
            let kv = b.kvcache_load(0, nkt, n_ctx, nk);
            let at = b.attn(
                q_r,
                kv,
                pos,
                AttnMode::Gqa,
                AttnMeta {
                    layer: 0,
                    n_head: nh,
                    n_head_kv: nk,
                    hd,
                    hd_kv: hd,
                    nkt,
                    scale: 1.0 / (hd as f32).sqrt(),
                },
            );
            b.output(at);
            let g = b.build();

            let mut seed = 0x1234_5678u32;
            let mut next = move || {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((seed >> 8) as f32 / 8_388_608.0) - 1.0
            };
            let q: Vec<f32> = (0..nh * hd * n).map(|_| next()).collect();
            let k: Vec<f32> = (0..nkt * n).map(|_| next()).collect();
            let v: Vec<f32> = (0..nkt * n).map(|_| next()).collect();

            let run = |start: usize| -> Vec<f32> {
                let mut alloc = GraphAllocator::new();
                alloc.kv_set_capacity(n_ctx);
                alloc.alloc_graph(&g).expect("alloc");
                alloc.fill_input(&g, "q", &q).unwrap();
                alloc.fill_input(&g, "k", &k).unwrap();
                alloc.fill_input(&g, "v", &v).unwrap();
                // C6: `positions` are sequence-relative (the token's index), the
                // KV rows are the resolved `cells`, and the span stays a cell
                // range. The rotation therefore sees the same angles at every
                // `start`, which is what makes the comparison bitwise.
                let positions: Vec<u32> = (0..n).map(|p| p as u32).collect();
                let cells: Vec<u32> = (start..start + n).map(|p| p as u32).collect();
                let mut span = vec![start as u32; n];
                span.extend((0..n).map(|t| (start + t + 1) as u32));
                alloc.fill_input_i32(&g, "positions", &positions).unwrap();
                alloc.fill_input_i32(&g, "cells", &cells).unwrap();
                alloc.fill_input_i32(&g, "attn_span", &span).unwrap();
                let mut sched = crate::graph::scheduler::BackendScheduler::new();
                sched.execute(&g, &mut alloc).expect("execute");
                alloc.copy_to_cpu(at).expect("read")
            };
            let d = |a: &[f32], b: &[f32]| -> f32 {
                a.iter()
                    .zip(b)
                    .map(|(x, y)| (x - y).abs())
                    .fold(0.0f32, f32::max)
            };
            let a0 = run(0);
            let a1 = run(1);
            let a8 = run(8);
            let (near, far) = (d(&a0, &a1), d(&a0, &a8));
            eprintln!(
                "[minimal] nh={nh} nk={nk} hd={hd} n={n}: cell0-vs-1 {near} | cell0-vs-8 {far}"
            );
            // C6: with relative positions a cell placement changes nothing at
            // all — the rows are written verbatim and read through the same
            // relative window, so this is bitwise, not a tolerance class. (Before
            // C6 the same comparison differed by the rotation's own rounding,
            // measured <= 1.2e-7; plan §14 row 9.)
            assert_eq!(
                (near, far),
                (0.0, 0.0),
                "a cell placement must not change the attention output"
            );
        }
    }

    #[test]
    fn overlapping_rows_move_safely_in_both_directions() {
        let mut b = CpuBackend::new();
        let id = b.alloc_buffer(16);
        b.write_host(id, &(0..16).map(|i| i as f32).collect::<Vec<_>>())
            .unwrap();
        let r = BufRef::own(crate::graph::Backend::CPU, id, 16);
        // Rows of 4 elements: row 0 <- row 1, overlapping by design.
        b.copy_cells(r, r, 0, 1, 2, 4).unwrap();
        assert_eq!(
            b.read_host(id).unwrap(),
            &[
                4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0
            ]
        );
        // C7b: the upward direction is supported as well — `copy_within` is a
        // memmove — and an overlapping upward move must land exactly where a copy
        // through a temporary would (row 1 <- row 0, overlapping).
        b.write_host(id, &(0..16).map(|i| i as f32).collect::<Vec<_>>())
            .unwrap();
        b.copy_cells(r, r, 1, 0, 2, 4).unwrap();
        assert_eq!(
            b.read_host(id).unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0]
        );
        // A range leaving the buffer is an error, never a truncation.
        assert!(b.copy_cells(r, r, 0, 3, 4, 4).is_err());
        assert!(b.copy_cells(r, r, 0, 0, 1, 32).is_err());
    }
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::{DType, NodeId};

    fn tensor_f32(name: &str, shape: [i64; 4], data: Vec<f32>) -> Tensor {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut t = Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
        t.name = name.to_string();
        t
    }

    struct Harness {
        sched: BackendScheduler,
        alloc: GraphAllocator,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                sched: BackendScheduler::new(),
                alloc: GraphAllocator::new(),
            }
        }
        fn reg(&mut self, t: Tensor) {
            let name = t.name.clone();
            self.alloc.register_weight(&name, t);
        }
        fn run(&mut self, graph: &crate::graph::ComputeGraph, fills: &[(&str, Vec<f32>)]) {
            self.alloc.alloc_graph(graph).unwrap();
            for (name, data) in fills {
                self.alloc.fill_input(graph, name, data).unwrap();
            }
            self.sched.execute(graph, &mut self.alloc).unwrap();
        }
        fn out(&self, graph: &crate::graph::ComputeGraph, id: NodeId) -> Vec<f32> {
            self.alloc.get_buffer(graph, id).unwrap().to_vec()
        }
    }

    #[test]
    // 8a② regression: an F32-weight matmul with nt > 1 must produce
    // token-major output [nt][od]. The old mat_mul_f32 wrote [od][nt] —
    // only visible for nt > 1 (decode nt==1 was accidentally correct).
    #[test]
    fn f32_matmul_nt2_token_major() {
        let mut h = Harness::new();
        // W [in=4, out=3]: row o selects (o+1) * x[o]
        h.reg(tensor_f32(
            "W",
            [4, 3, 1, 1],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        ));
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [4, 2, 1, 1], DType::F32);
        let m = gb.matmul(x, h.alloc.cpu().weight("W").unwrap(), None);
        gb.output(m);
        let g = gb.build();
        let xdata = vec![1.0, 2.0, 3.0, 4.0, -1.0, 0.5, -0.25, 2.0];
        h.run(&g, &[("x", xdata)]);
        let got = h.out(&g, m);
        // token 0: [1, 4, 9]; token 1: [-1, 1, -0.75]
        let expect = [1.0, 4.0, 9.0, -1.0, 1.0, -0.75];
        for i in 0..6 {
            assert!(
                (got[i] - expect[i]).abs() < 1e-5,
                "out[{i}]={} expect {}",
                got[i],
                expect[i]
            );
        }
    }

    fn matmul_add_silu_scale() {
        // x [4,1] * W [3,4] + b [3] -> [3,1]; silu; *2
        let mut h = Harness::new();
        // weight metadata [in=4, out=3]; memory = 3 rows of 4 (out-major)
        h.reg(tensor_f32(
            "W",
            [4, 3, 1, 1],
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0],
        ));
        h.reg(tensor_f32("b", [3, 1, 1, 1], vec![0.5, -1.0, 0.25]));

        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [4, 1, 1, 1], DType::F32);
        let m = gb.matmul(
            x,
            h.alloc.cpu().weight("W").unwrap(),
            Some(h.alloc.cpu().weight("b").unwrap()),
        );
        let s = gb.silu(m);
        let o = gb.node(
            "scale2",
            Op::Scale(2.0),
            &[s],
            [3, 1, 1, 1],
            DType::F32,
            NodeMeta::None,
        );
        gb.output(o);
        let g = gb.build();

        h.run(&g, &[("x", vec![1.0, 2.0, 3.0, 4.0])]);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let expect = [silu(1.5) * 2.0, silu(3.0) * 2.0, silu(9.25) * 2.0];
        let got = h.out(&g, o);
        for i in 0..3 {
            assert!(
                (got[i] - expect[i]).abs() < 1e-4,
                "out[{i}]={} expect {}",
                got[i],
                expect[i]
            );
        }
    }

    #[test]
    fn rms_norm_matches_reference() {
        let mut h = Harness::new();
        h.reg(tensor_f32("nw", [4, 1, 1, 1], vec![1.0, 1.0, 1.0, 1.0]));
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [4, 2, 1, 1], DType::F32);
        let r = gb.rms_norm(x, Some(h.alloc.cpu().weight("nw").unwrap()), 1e-5);
        gb.output(r);
        let g = gb.build();
        let data = vec![1.0, 2.0, 3.0, 4.0, 0.5, -0.5, 2.0, -3.0];
        h.run(&g, &[("x", data.clone())]);
        let got = h.out(&g, r);
        // reference via vec_ops directly
        let mut ref_out = vec![0.0f32; 8];
        for t in 0..2 {
            crate::vec_ops::rms_norm_fused_f32(
                4,
                &mut ref_out[t * 4..(t + 1) * 4],
                &data[t * 4..(t + 1) * 4],
                &[1.0, 1.0, 1.0, 1.0],
                1e-5,
            );
        }
        for i in 0..8 {
            assert!(
                (got[i] - ref_out[i]).abs() < 1e-6,
                "norm[{i}] {} vs {}",
                got[i],
                ref_out[i]
            );
        }
    }

    #[test]
    fn embedding_and_rope() {
        // vocab 4, n_embd 4: ids [0,2] -> rows, then rope per 2 heads of hd 2
        let mut h = Harness::new();
        h.reg(tensor_f32(
            "tok_embd",
            [4, 4, 1, 1],
            vec![
                0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4,
            ],
        ));
        let mut gb = GraphBuilder::new();
        let ids = gb.input("token_ids", [2, 1, 1, 1], DType::I32);
        let emb = gb.embedding(ids, h.alloc.cpu().weight("tok_embd").unwrap());
        let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
        let rope = gb.rope(
            emb,
            pos,
            RopeStyle::NonInterleaved,
            super::super::ops::RoPEMeta {
                freq_base: 10000.0,
                freq_scale: 1.0,
                n_head: 2,
                hd: 2,
            },
        );
        gb.output(rope);
        let g = gb.build();
        // ids 0,2 at positions 0,1 (I32 inputs via bit patterns)
        h.alloc.alloc_graph(&g).unwrap();
        h.alloc.fill_input_i32(&g, "token_ids", &[0, 2]).unwrap();
        h.alloc.fill_input_i32(&g, "positions", &[0, 1]).unwrap();
        // E1: the attention window is data the graph carries, not a bound it
        // derives — a hand-built graph fills it like the model path does.
        h.alloc.fill_attn_inputs(&g, &[0, 0], &[0, 1]).unwrap();
        h.sched.execute(&g, &mut h.alloc).unwrap();
        let got = h.out(&g, rope);
        // reference: embed rows then rope per head
        let mut ref_x = vec![
            0.1, 0.2, 0.3, 0.4, // id 0
            2.1, 2.2, 2.3, 2.4, // id 2
        ];
        cpu_rope(
            &mut ref_x,
            &[0, 1],
            2,
            2,
            10000.0,
            1.0,
            RopeStyle::NonInterleaved,
        );
        for i in 0..8 {
            assert!(
                (got[i] - ref_x[i]).abs() < 1e-5,
                "rope[{i}] {} vs {}",
                got[i],
                ref_x[i]
            );
        }
    }

    #[test]
    fn kvcache_store_load_and_attn_roundtrip() {
        // one layer: q [hd=2, nh=2, nt=1] vs stored k/v at pos 0; GQA nh=2 nk=2
        let mut h = Harness::new();
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [1, 1, 1, 1], DType::I32);
        let q = gb.input("q", [4, 1, 1, 1], DType::F32);
        let k = gb.input("k", [4, 1, 1, 1], DType::F32);
        let v = gb.input("v", [4, 1, 1, 1], DType::F32);
        let _st = gb.kvcache_store(0, k, v, 8);
        let kv = gb.kvcache_load(0, 4, 8, 2);
        let out = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            super::super::ops::AttnMeta {
                layer: 0,
                n_head: 2,
                n_head_kv: 2,
                hd: 2,
                hd_kv: 2,
                nkt: 4,
                scale: 0.5,
            },
        );
        gb.output(out);
        let g = gb.build();

        // q = [1,0, 0,1], k = [1,0, 0,1], v = [0.5,0.5, 0.25,0.75] at pos 0
        h.alloc.alloc_graph(&g).unwrap();
        h.alloc.fill_input_i32(&g, "positions", &[0]).unwrap();
        h.alloc.fill_attn_inputs(&g, &[0], &[0]).unwrap();
        h.alloc.fill_input(&g, "q", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        h.alloc.fill_input(&g, "k", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        h.alloc
            .fill_input(&g, "v", &[0.5, 0.5, 0.25, 0.75])
            .unwrap();
        h.sched.execute(&g, &mut h.alloc).unwrap();
        let got = h.out(&g, out);
        // scores: h0: dot([1,0],[1,0])*0.5 = 0.5; h1: dot([0,1],[0,1])*0.5 = 0.5
        // softmax([0.5]) = 1.0 -> out = v
        assert!((got[0] - 0.5).abs() < 1e-5, "got[0]={}", got[0]);
        assert!((got[1] - 0.5).abs() < 1e-5, "got[1]={}", got[1]);
        assert!((got[2] - 0.25).abs() < 1e-5, "got[2]={}", got[2]);
        assert!((got[3] - 0.75).abs() < 1e-5, "got[3]={}", got[3]);
    }

    /// C4: a packed Q8_0 KV region answers like the f32 one and occupies about a
    /// third of the memory. The store quantizes, the attention read dequantizes the
    /// window it is about to use; three rows and three causal queries make the
    /// softmax weights depend on the *quantized scores*, so a broken K read cannot
    /// pass by returning V verbatim.
    #[test]
    fn a_packed_kv_region_answers_like_the_f32_one_and_is_smaller() {
        use super::super::kvformat::KvFormat;
        let nkt = 32usize; // Q8_0 quantizes in 32-element blocks
        let hd = 32usize;
        let nt = 3usize;
        let n_ctx = 8usize;
        let qv: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i as f32) * 0.13).sin() * 0.7)
            .collect();
        let kk: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i as f32) * 0.29).cos() * 1.3 + 0.04 * (i % 7) as f32)
            .collect();
        let vv: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5 - 0.02 * (i % 3) as f32)
            .collect();

        let run = |format: KvFormat| -> (Vec<f32>, usize, Vec<f32>) {
            let mut h = Harness::new();
            // Both halves of the decision, without touching the process-wide policy:
            // the builder stamps the cell width, the backend reads it.
            h.alloc.cpu_mut().set_kv_format(format);
            let mut gb = GraphBuilder::new();
            gb.set_kv_format(format);
            let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
            let q = gb.input("q", [nkt, nt, 1, 1], DType::F32);
            let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
            let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
            let store = gb.kvcache_store(0, k, v, n_ctx);
            let kv = gb.kvcache_load(0, nkt, n_ctx, 1);
            let out = gb.attn(
                q,
                kv,
                pos,
                crate::graph::ops::AttnMode::Gqa,
                super::super::ops::AttnMeta {
                    layer: 0,
                    n_head: 1,
                    n_head_kv: 1,
                    hd,
                    hd_kv: hd,
                    nkt,
                    scale: 1.0 / (hd as f32).sqrt(),
                },
            );
            gb.output(out);
            let g = gb.build();
            h.alloc.alloc_graph(&g).unwrap();
            h.alloc.fill_input_i32(&g, "positions", &[0, 1, 2]).unwrap();
            h.alloc
                .fill_attn_inputs(&g, &[0, 0, 0], &[0, 1, 2])
                .unwrap();
            h.alloc.fill_input(&g, "q", &qv).unwrap();
            h.alloc.fill_input(&g, "k", &kk).unwrap();
            h.alloc.fill_input(&g, "v", &vv).unwrap();
            h.sched.execute(&g, &mut h.alloc).unwrap();
            // The store node's buffer *is* the K region, so this is what the
            // attention read dequantized.
            (h.out(&g, out), h.alloc.kv_region_bytes(), h.out(&g, store))
        };

        let (f32_out, f32_bytes, f32_region) = run(KvFormat::F32);
        let (q8_out, q8_bytes, q8_region) = run(KvFormat::Q8_0);
        assert_eq!(f32_out.len(), nkt * nt);
        assert_eq!(q8_out.len(), nkt * nt);
        // Footprint: 4 bytes/element against ceil(34/4) words per 32 elements.
        assert_eq!(f32_bytes, n_ctx * nkt * 4 * 2);
        assert_eq!(q8_bytes, n_ctx * 9 * 4 * 2);
        assert!(
            q8_bytes * 3 <= f32_bytes,
            "the packed region must be at least 3x smaller: {q8_bytes} vs {f32_bytes}"
        );
        // The stored cell must be exactly the Q8_0 quantizate of the input row —
        // bitwise, not merely close. This is what separates "the packed layout is
        // addressed correctly" from "the numbers happen to be near": the packed
        // store's bytes and an independent `quantize -> dequantize` of the same
        // row must agree to the last bit, and the f32 run's region must hold the
        // *un*quantized row.
        let row_zero = &kk[..nkt];
        let mut want = vec![0.0f32; nkt];
        let bytes = crate::quants::quantize_row_q8_0(row_zero);
        crate::quants::dequantize_row_q8_0(&bytes, &mut want);
        let mut got = vec![0.0f32; nkt];
        super::super::kvformat::unpack_q8_0_cells(&q8_region, nkt, 0, 1, &mut got);
        assert_eq!(got, want, "the packed cell is not the Q8_0 quantizate");
        assert_eq!(
            &f32_region[..nkt],
            row_zero,
            "the f32 region must hold the row verbatim"
        );
        // Tolerance class: the per-block step of Q8_0, softened by the softmax over
        // three rows (a wrong cell width or a byte-order slip is off by orders of
        // magnitude more, which is what the gate is for).
        let worst = (0..f32_out.len())
            .map(|i| (f32_out[i] - q8_out[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(
            worst < 5e-2,
            "packed KV vs f32 KV: max |Δ| = {worst} over {} outputs",
            f32_out.len()
        );
    }

    /// C4 S2: the fused read addresses **each KV head's blocks** and agrees with the
    /// S1 path it replaces.
    ///
    /// The oracle here is S1's own mechanism — `unpack_q8_0_cells` →
    /// `cpu_gqa_attn_runs` — so the only permitted difference is the *query's* Q8_0
    /// quantization (the K score is now a `dot_q8_0_q8_0`). Two KV heads with very
    /// different K rows make a wrong head base (`hk * hd`, the block offset the
    /// fused path computes itself) fail by the whole spread instead of by an ulp.
    #[test]
    fn the_fused_q8_read_matches_the_dequantizing_reference() {
        use super::super::kvformat::{self, KvFormat};
        let (nh, nk, hd, nt, n_ctx) = (2usize, 2usize, 32usize, 3usize, 6usize);
        let nkt = nk * hd; // 64 → two heads, each exactly one Q8_0 block wide
        let scale = 1.0 / (hd as f32).sqrt();
        let kk: Vec<f32> = (0..n_ctx * nkt)
            .map(|i| {
                if (i % nkt) / hd == 0 {
                    ((i as f32) * 0.31).sin() * 0.9
                } else {
                    ((i as f32) * 0.17).cos() * 2.5 - 1.0
                }
            })
            .collect();
        let vv: Vec<f32> = (0..n_ctx * nkt)
            .map(|i| ((i as f32) * 0.11).sin() * 0.6 - (i % 5) as f32 * 0.03)
            .collect();
        let q: Vec<f32> = (0..nt * nh * hd)
            .map(|i| ((i as f32) * 0.23).cos() * 0.8)
            .collect();

        // Pack exactly as the store does: one cell per row, heads block-aligned.
        let row_elems = KvFormat::Q8_0.row_elems(nkt);
        let mut kreg = vec![0.0f32; n_ctx * row_elems];
        let mut vreg = vec![0.0f32; n_ctx * row_elems];
        for cell in 0..n_ctx {
            let w = cell * row_elems..(cell + 1) * row_elems;
            kvformat::pack_q8_0_cell(&mut kreg[w.clone()], nkt, &kk[cell * nkt..(cell + 1) * nkt]);
            kvformat::pack_q8_0_cell(&mut vreg[w], nkt, &vv[cell * nkt..(cell + 1) * nkt]);
        }

        // Causal windows, one run per query (`off[t]..off[t + 1]`).
        let runs: Vec<(usize, usize)> = (0..nt).map(|t| (0usize, t + 1)).collect();
        let off: Vec<usize> = (0..=nt).collect();

        let mut kf = vec![0.0f32; n_ctx * nkt];
        let mut vf = vec![0.0f32; n_ctx * nkt];
        kvformat::unpack_q8_0_cells(&kreg, nkt, 0, n_ctx, &mut kf);
        kvformat::unpack_q8_0_cells(&vreg, nkt, 0, n_ctx, &mut vf);
        let mut want = vec![0.0f32; nt * nh * hd];
        cpu_gqa_attn_runs(
            &q, &kf, &vf, &runs, &off, nt, nh, nk, hd, hd, nkt, &mut want, scale,
        )
        .unwrap();

        let mut got = vec![0.0f32; nt * nh * hd];
        cpu_gqa_attn_runs_q8(
            &q, &kreg, &vreg, &runs, &off, nt, nh, nk, hd, hd, nkt, &mut got, scale,
        )
        .unwrap();

        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let spread = want.iter().fold(f32::NEG_INFINITY, |m, x| m.max(*x))
            - want.iter().fold(f32::INFINITY, |m, x| m.min(*x));
        eprintln!("[c4s2] fused vs dequantizing reference: max |Δ| = {worst} of a {spread} spread");
        assert!(
            worst < 1e-3,
            "the fused read may only differ by the query's Q8_0 quantization: max |Δ| = \
             {worst} of a {spread} spread"
        );
    }

    /// C4 S2: a physical shift on a **packed** region moves every surviving cell
    /// verbatim (V is never re-roped, so it is bitwise), and re-ropes the survivors'
    /// K through dequantize → rope → requantize.
    ///
    /// The gate is exact on both sides: V must equal the old cells byte-for-byte, and
    /// the new K cell must be the Q8_0 quantizate of the re-roped f32 row — computed
    /// here independently, so a shift that forgets the requantize (or re-ropes the
    /// packed bytes) cannot pass.
    #[test]
    fn a_packed_physical_shift_moves_v_verbatim_and_requantizes_k() {
        use super::super::kvformat::{self, KvFormat};
        use crate::graph::kvcache::{rope_shift_kv, KvRope};
        use crate::vec_ops::RopeStyle;

        let nkt = 32usize;
        let hd = 32usize;
        let nt = 3usize;
        let n_ctx = 8usize;
        let kk: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i as f32) * 0.29).cos() * 1.3 + 0.04 * (i % 7) as f32)
            .collect();
        let vv: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i as f32) * 0.07).sin() * 0.5 - 0.02 * (i % 3) as f32)
            .collect();

        let mut h = Harness::new();
        h.alloc.cpu_mut().set_kv_format(KvFormat::Q8_0);
        let mut gb = GraphBuilder::new();
        gb.set_kv_format(KvFormat::Q8_0);
        let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let q = gb.input("q", [nkt, nt, 1, 1], DType::F32);
        let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
        let store = gb.kvcache_store(0, k, v, n_ctx);
        gb.output(store);
        let g = gb.build();
        h.alloc.alloc_graph(&g).unwrap();
        h.alloc.fill_input_i32(&g, "positions", &[0, 1, 2]).unwrap();
        // `kvcache_store` consumes the builder's own `cells` input (C6): filling
        // `positions` alone left every row going to cell 0, which made this gate pass
        // on a region with one live row — exactly the kind of false green the nonzero
        // assertions below exist to prevent.
        h.alloc
            .fill_attn_inputs(&g, &[0, 0, 0], &[0, 1, 2])
            .unwrap();
        h.alloc.fill_input(&g, "k", &kk).unwrap();
        h.alloc.fill_input(&g, "v", &vv).unwrap();
        h.sched.execute(&g, &mut h.alloc).unwrap();
        h.alloc.kv_note_used(nt);

        let rope = KvRope {
            freq_base: 10_000.0,
            freq_scale: 1.0,
            n_head_kv: 1,
            hd,
            style: RopeStyle::NonInterleaved,
        };
        let row_elems = KvFormat::Q8_0.row_elems(nkt);
        let (before_k_words, before_v_words) = h.alloc.copy_kv_to_cpu(0).unwrap();
        let mut before_k = vec![0.0f32; nt * nkt];
        let mut before_v = vec![0.0f32; nt * nkt];
        kvformat::unpack_q8_0_cells(&before_k_words, nkt, 0, nt, &mut before_k);
        kvformat::unpack_q8_0_cells(&before_v_words, nkt, 0, nt, &mut before_v);

        // The fixture must be real: three written rows, each with data, or every
        // assertion below would pass on zeros.
        for r in 0..nt {
            assert!(
                before_k[r * nkt..(r + 1) * nkt].iter().any(|x| *x != 0.0)
                    && before_v[r * nkt..(r + 1) * nkt].iter().any(|x| *x != 0.0),
                "row {r} was not written (the store did not get its cells input)"
            );
        }
        // Drop the oldest row: cells 1..3 slide to 0..2 and are re-roped by -1.
        let left = h.alloc.kv_rm(0, 1, &rope).unwrap();
        assert_eq!(left, nt - 1, "one row removed");
        let (after_k_words, after_v_words) = h.alloc.copy_kv_to_cpu(0).unwrap();
        let mut after_k = vec![0.0f32; nt * nkt];
        let mut after_v = vec![0.0f32; nt * nkt];
        kvformat::unpack_q8_0_cells(&after_k_words, nkt, 0, left, &mut after_k);
        kvformat::unpack_q8_0_cells(&after_v_words, nkt, 0, left, &mut after_v);

        // V: the cells moved verbatim, so the dequantized values are bitwise equal.
        assert_eq!(
            &after_v[..left * nkt],
            &before_v[nkt..(left + 1) * nkt],
            "V must move verbatim under a packed shift"
        );
        // The packed words themselves must be the old cells' words: a cell is a whole
        // number of words, which is exactly why the move needs no format knowledge.
        // Compared as **bits**: a packed word is an f16 scale plus int8 quants, so as an
        // f32 it is frequently a NaN, and `==` on it is never true.
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(
            bits(&after_v_words[..left * row_elems]),
            bits(&before_v_words[row_elems..(left + 1) * row_elems]),
            "the V cells must move verbatim, byte for byte"
        );
        // K: exactly the Q8_0 quantizate of the old row, re-roped in f32.
        for r in 0..left {
            let mut want = before_k[(r + 1) * nkt..(r + 2) * nkt].to_vec();
            rope_shift_kv(&mut want, 1, 1, &rope);
            let bytes = crate::quants::quantize_row_q8_0(&want);
            let mut quantized = vec![0.0f32; nkt];
            crate::quants::dequantize_row_q8_0(&bytes, &mut quantized);
            let got = &after_k[r * nkt..(r + 1) * nkt];
            assert_eq!(
                got,
                &quantized[..],
                "row {r}: the shifted K must be the Q8_0 quantizate of the re-roped row"
            );
            // And the honest approximation class: the re-rope of a *quantized* row
            // differs from the quantum step of the stored cell.
            let step = got
                .iter()
                .zip(&want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                step < 0.2,
                "row {r}: re-rope vs the f32 re-rope differs by {step}, more than one Q8_0 step"
            );
        }
        // The tail must be cleared: no position may address a stale row.
        for cell in left..n_ctx {
            assert!(
                after_k_words[cell * row_elems..(cell + 1) * row_elems]
                    .iter()
                    .all(|x| *x == 0.0)
                    && after_v_words[cell * row_elems..(cell + 1) * row_elems]
                        .iter()
                        .all(|x| *x == 0.0),
                "cell {cell} must be zeroed after the shift"
            );
        }
    }

    /// C4: a row width Q8_0 cannot express is refused where the region is sized,
    /// not truncated into a layout that would mis-address every cell.
    #[test]
    fn a_packed_region_refuses_a_width_q8_0_cannot_express() {
        use super::super::kvformat::KvFormat;
        let mut h = Harness::new();
        h.alloc.cpu_mut().set_kv_format(KvFormat::Q8_0);
        let mut gb = GraphBuilder::new();
        gb.set_kv_format(KvFormat::Q8_0);
        let pos = gb.input("positions", [1, 1, 1, 1], DType::I32);
        let q = gb.input("q", [4, 1, 1, 1], DType::F32);
        let k = gb.input("k", [4, 1, 1, 1], DType::F32);
        let v = gb.input("v", [4, 1, 1, 1], DType::F32);
        let _store = gb.kvcache_store(0, k, v, 8);
        let kv = gb.kvcache_load(0, 4, 8, 2);
        let out = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            super::super::ops::AttnMeta {
                layer: 0,
                n_head: 2,
                n_head_kv: 2,
                hd: 2,
                hd_kv: 2,
                nkt: 4,
                scale: 0.5,
            },
        );
        gb.output(out);
        let g = gb.build();
        let err = h.alloc.alloc_graph(&g).unwrap_err();
        assert!(err.contains("multiple of 32"), "{err}");
    }

    /// C4 regression: `ensure_kv` takes a **logical** cell width and a **stored**
    /// one, and the D3-8 mixed-quant epilogue — which also stores K/V — must hand in
    /// the same `n_kv_embd` the store node does. Passing `n_ctx` in its place made
    /// `packed` compare two unrelated numbers, so a *CPU* graph with this node failed
    /// with "the node declares N words per cell but the Q8_0 layout packs ..."; no
    /// cached model here builds the epilogue (their q/k/v share a quant type), which
    /// is why only this hand-built graph covers the line.
    #[test]
    fn the_qkv_epilogue_sizes_its_kv_region_by_n_kv_embd() {
        let (nkt, nqt, n_ctx) = (4usize, 8usize, 16usize);
        let mut h = Harness::new();
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [1, 1, 1, 1], DType::I32);
        let q = gb.input("q", [nqt, 1, 1, 1], DType::F32);
        let k = gb.input("k", [nkt, 1, 1, 1], DType::F32);
        let v = gb.input("v", [nkt, 1, 1, 1], DType::F32);
        let ep = gb.qkv_bias_rope_store(
            q,
            k,
            v,
            pos,
            0,
            crate::graph::ops::QkvBiasRopeStoreMeta {
                bias_q: None,
                bias_k: None,
                bias_v: None,
                nqt,
                nkt,
                hd: 4,
                freq_base: 10_000.0,
                freq_scale: 1.0,
                rope_style: crate::vec_ops::RopeStyle::NonInterleaved,
                kv_elems: nkt * n_ctx,
                row_elems: crate::graph::kvformat::KvFormat::F32.row_elems(nkt),
            },
        );
        let kv = gb.kvcache_load(0, nkt, n_ctx, 1);
        let out = gb.attn(
            ep,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            super::super::ops::AttnMeta {
                layer: 0,
                n_head: 1,
                n_head_kv: 1,
                hd: nkt,
                hd_kv: nkt,
                nkt,
                scale: 1.0,
            },
        );
        gb.output(out);
        let g = gb.build();
        h.alloc.alloc_graph(&g).unwrap();
        // Unpacked: one f32 word per element, K and V.
        assert_eq!(h.alloc.kv_region_bytes(), n_ctx * nkt * 4 * 2);
        assert!(!h.alloc.kv_is_packed());
    }

    /// E1's acceptance: two sequences sharing one KV arena must not see each
    /// other. The windows are the input, so the test supplies them directly —
    /// the resolver that produces them is covered by `graph::kvcache`'s tests.
    ///
    /// The values are chosen so a leak *changes the answer*: query 1 would score
    /// 1.0 against sequence 0's key if its window wrongly reached cell 0, which
    /// would blend V(0) into the output instead of returning V(2).
    #[test]
    fn two_sequences_do_not_cross_attend() {
        let mut h = Harness::new();
        let mut gb = GraphBuilder::new();
        gb.set_explicit_span(true);
        // k/v are [nkt, nt] so the store node sizes the region as nkt * n_ctx.
        let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
        let q = gb.input("q", [2, 2, 1, 1], DType::F32);
        let k = gb.input("k", [2, 2, 1, 1], DType::F32);
        let v = gb.input("v", [2, 2, 1, 1], DType::F32);
        let _st = gb.kvcache_store(0, k, v, 4);
        let kv = gb.kvcache_load(0, 2, 4, 1);
        let out = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            super::super::ops::AttnMeta {
                layer: 0,
                n_head: 1,
                n_head_kv: 1,
                hd: 2,
                hd_kv: 2,
                nkt: 2,
                scale: 1.0,
            },
        );
        gb.output(out);
        let g = gb.build();
        // The graph must carry the multi-sequence flag: it is what stops a
        // backend that still derives its bound from positions from taking it.
        assert!(
            g.nodes.iter().any(|n| matches!(
                n.op,
                crate::graph::ops::Op::Attn {
                    explicit_span: true,
                    ..
                }
            )),
            "the attention node must declare explicit_span"
        );

        h.alloc.alloc_graph(&g).unwrap();
        // Sequence 0 owns row 0, sequence 1 owns row 2 (rows 1 and 3 stay free).
        // C6: `positions` are each token's index within its sequence, while the
        // store's rows come from the resolved `cells`.
        h.alloc.fill_input_i32(&g, "positions", &[0, 0]).unwrap();
        h.alloc.fill_input_i32(&g, "seq_ids", &[0, 1]).unwrap();
        h.alloc.fill_input_i32(&g, "cells", &[0, 2]).unwrap();
        // The span layout is the `lo` block then the `hi` block: token 0 gets
        // `[0, 1)` (sequence 0's only row) and token 1 `[2, 3)` (sequence 1's).
        h.alloc
            .fill_input_i32(&g, "attn_span", &[0, 2, 1, 3])
            .unwrap();
        // token 0 = [1, 0], token 1 = [1, 0]
        h.alloc.fill_input(&g, "q", &[1.0, 0.0, 1.0, 0.0]).unwrap();
        // row 0 = k [1, 0] / v [1, 0]; row 2 = k [0, 1] / v [0, 1]
        h.alloc.fill_input(&g, "k", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        h.alloc.fill_input(&g, "v", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        h.sched.execute(&g, &mut h.alloc).unwrap();
        let got = h.out(&g, out);
        assert_eq!(got.len(), 4);
        // Token 0 attends to its own row only -> V(0).
        assert!(
            (got[0] - 1.0).abs() < 1e-6 && got[1].abs() < 1e-6,
            "token 0: {got:?}"
        );
        // Token 1 attends to its own row only -> V(2). A cross-sequence leak
        // would show up here as roughly [0.73, 0.27].
        assert!(
            got[2].abs() < 1e-6 && (got[3] - 1.0).abs() < 1e-6,
            "token 1 saw the other sequence: {got:?}"
        );
    }

    /// Generic get_rows (n_out tail selection): out[t] = x[ids[t]].
    #[test]
    fn cpu_generic_get_rows() {
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [4, 3, 1, 1], DType::F32);
        let ids = gb.input("ids", [1, 1, 1, 1], DType::I32);
        let r = gb.get_rows(x, ids, [4, 1, 1, 1]);
        gb.output(r);
        let g = gb.build();

        let mut sched = BackendScheduler::new();
        let mut alloc = GraphAllocator::new();
        alloc.alloc_graph(&g).unwrap();
        // rows: r0=[1,2,3,4] r1=[10,20,30,40] r2=[100,200,300,400]
        alloc
            .fill_input(
                &g,
                "x",
                &[
                    1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 100.0, 200.0, 300.0, 400.0,
                ],
            )
            .unwrap();
        alloc.fill_input_i32(&g, "ids", &[2]).unwrap();
        sched.execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, r).unwrap();
        assert_eq!(
            got,
            &[100.0, 200.0, 300.0, 400.0],
            "get_rows should select row 2"
        );

        // ids = [0]
        alloc.fill_input_i32(&g, "ids", &[0]).unwrap();
        sched.execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, r).unwrap();
        assert_eq!(got, &[1.0, 2.0, 3.0, 4.0], "get_rows should select row 0");
    }
}
