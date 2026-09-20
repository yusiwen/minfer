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
use super::ops::{FusedOp, NodeMeta, Op};
use super::{BufRef, CNode, DType};

/// CPU buffer pool + weight registry.
pub struct CpuBackend {
    buffers: Vec<Vec<f32>>,
    free: Vec<usize>,
    weights: HashMap<String, Tensor>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self {
            buffers: Vec::new(),
            free: Vec::new(),
            weights: HashMap::new(),
        }
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

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
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

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        // The SwiGLU rewrite IS applied to CPU nodes (CPU is first in the
        // fusion pass's backend list); `Op::SwiGLU` below executes it as a
        // single pass. bias+rope and batch-matmul are not fused (batch QKV
        // quantize-sharing is a Phase 5+ win).
        matches!(fused, FusedOp::SwiGLU)
    }

    fn supports_attn_span(&self) -> bool {
        // E1: the CPU attention kernel reads `attn_span` (the window the KV
        // store resolved), so it can take a multi-sequence attention node.
        true
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
        self.buffers.len() - 1
    }

    fn free_buffer(&mut self, id: usize) {
        if !self.free.contains(&id) {
            self.free.push(id);
        }
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // never recycled from the free list (see Backend::alloc_fresh)
        self.buffers.push(vec![0.0f32; size]);
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
            let nkt = node.out_shape[0];
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
            for t in 0..nt {
                let p = pos[t];
                if p >= n_ctx {
                    return Err(format!("KV store position {p} >= n_ctx {n_ctx}"));
                }
                let ks = p * nkt;
                k_dst[ks..ks + nkt].copy_from_slice(&k_src[t * nkt..(t + 1) * nkt]);
                v_dst[ks..ks + nkt].copy_from_slice(&v_src[t * nkt..(t + 1) * nkt]);
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
                let n_ctx = k_slice.len() / nkt;
                // E1: the allowed cells are an explicit input (`attn_span`), not
                // a bound derived from `positions` — that is what lets a batch
                // hold several sequences. The allocator resolved it from the
                // cell store's ownership; here it is only validated.
                let span_in = ins[3];
                if span_in.len() != 2 * nt {
                    return Err(format!(
                        "attn: span input has {} values, expected {} (2 per query token)",
                        span_in.len(),
                        2 * nt
                    ));
                }
                let span: Vec<(usize, usize)> = (0..nt)
                    .map(|t| {
                        (
                            span_in[t].to_bits() as usize,
                            span_in[nt + t].to_bits() as usize,
                        )
                    })
                    .collect();
                for (t, &(lo, hi)) in span.iter().enumerate() {
                    if hi > n_ctx || hi <= lo {
                        return Err(format!(
                            "attn: query {t} has span [{lo}, {hi}) — outside the {n_ctx}-cell \
                             arena, or empty (a span input that was never filled is all zeros)"
                        ));
                    }
                }
                cpu_gqa_attn(
                    ins[0],
                    k_slice,
                    v_slice,
                    &span,
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
        if dst_row > src_row {
            return Err(format!(
                "copy_cells: dst row {dst_row} is below src row {src_row}; the contract is \
                 downward-only (ascending copies are the safe direction)"
            ));
        }
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
            .get_mut(id)
            .ok_or_else(|| format!("no buffer {id}"))?;
        if b.len() != data.len() {
            return Err(format!(
                "buffer {id}: expected {} elements, got {}",
                b.len(),
                data.len()
            ));
        }
        b.copy_from_slice(data);
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
    if hd < hd_kv {
        return Err(format!(
            "Q head dim ({hd}) must be >= KV head dim ({hd_kv})"
        ));
    }

    // Heads are independent: parallelize over head ranges on the CPU pool.
    // Each worker computes its own head range with a private scores buffer,
    // so the output is bit-identical to the single-threaded order (no
    // cross-head reduction).
    if span.len() != nt {
        return Err(format!(
            "attention: {} spans for {nt} query tokens",
            span.len()
        ));
    }
    let max_vl = span.iter().map(|&(lo, hi)| hi.saturating_sub(lo)).max();
    let ctx = AttnCtx {
        q: q.as_ptr(),
        ka: ka.as_ptr(),
        va: va.as_ptr(),
        span: span.as_ptr(),
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
    /// Per-query allowed cell range `[lo, hi)` (E1).
    span: *const (usize, usize),
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
            let (lo, hi) = *c.span.add(t);
            let vl = hi.saturating_sub(lo);
            let mut mx = f32::NEG_INFINITY;
            for (i, kv) in (lo..hi).enumerate() {
                let ks = kv * c.nkt + hk * c.hd_kv;
                let s = crate::vec_ops::vec_dot_f32(
                    c.hd_kv,
                    std::slice::from_raw_parts(c.q.add(qs), c.hd_kv),
                    std::slice::from_raw_parts(c.ka.add(ks), c.hd_kv),
                ) * c.scale;
                scrs[i] = s;
                if s > mx {
                    mx = s;
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
            for (i, kv) in (lo..hi).enumerate() {
                crate::vec_ops::vec_muladd_f32(
                    c.hd_kv,
                    std::slice::from_raw_parts_mut(c.out.add(os), c.hd_kv),
                    std::slice::from_raw_parts(c.va.add(kv * c.nkt + vs_base), c.hd_kv),
                    scrs[i],
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn overlapping_rows_move_down_safely() {
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
        // The safe direction is downward: an upward move is refused, not guessed.
        let err = b.copy_cells(r, r, 1, 0, 2, 4).unwrap_err();
        assert!(err.contains("downward-only"), "{err}");
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
