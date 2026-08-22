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
use super::{CNode, DType};

/// CPU buffer pool + weight registry.
pub struct CpuBackend {
    buffers: Vec<Vec<f32>>,
    free: Vec<usize>,
    weights: HashMap<String, Tensor>,
}

impl CpuBackend {
    pub fn new() -> Self {
        Self { buffers: Vec::new(), free: Vec::new(), weights: HashMap::new() }
    }

    /// Register a weight tensor by name (Phase 6 wires this from the model).
    pub fn register_weight(&mut self, name: &str, t: Tensor) {
        self.weights.insert(name.to_string(), t);
    }

    /// Look up a registered weight tensor (test / debug helper).
    #[allow(dead_code)]
    pub fn weight(&self, name: &str) -> Option<&Tensor> {
        self.weights.get(name)
    }

    /// Pool size (for tests).
    #[allow(dead_code)]
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
        // CPU has no dedicated fused kernels yet: silu+mul stays decomposed
        // (the fusion pass leaves it as-is on CPU); bias+rope and batch-matmul
        // are not fused either (batch QKV quantize-sharing is a Phase 5+ win).
        matches!(fused, FusedOp::SwiGLU)
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        if let Some(idx) = self.free.iter().position(|&id| self.buffers[id].len() == size) {
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

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[usize],
        out_buf: usize,
        kv_pair: Option<(usize, usize)>,
    ) -> Result<(), String> {
        // KV store needs mutable access to both K and V persistent regions
        // (the V region is a sibling buffer, not an input) — handle it before
        // the aliasing split below, which borrows the pool immutably.
        if let Op::KvcacheStore { layer } = &node.op {
            let (k_id, v_id) = kv_pair
                .ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
            if k_id != out_buf {
                return Err("KV store out buffer must be the K region".into());
            }
            let nkt = node.out_shape[0];
            let n_ctx = node.out_shape[1];
            let nt = self.buffers[in_bufs[0]].len() / nkt;
            let pos: Vec<usize> = self.buffers[in_bufs[2]].iter().map(|b| b.to_bits() as usize).collect();
            let k_src = self.buffers[in_bufs[0]].clone();
            let v_src = self.buffers[in_bufs[1]].clone();
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
        let mut aliased: Vec<Vec<f32>> = Vec::new();
        let mut alias_of: Vec<Option<usize>> = in_bufs.iter().map(|_| None).collect();
        for (k, &i) in in_bufs.iter().enumerate() {
            if i == out_buf {
                alias_of[k] = Some(aliased.len());
                aliased.push(self.buffers[out_buf].clone());
            }
        }
        let (before, rest) = self.buffers.split_at_mut(out_buf);
        let (out0, after) = rest.split_at_mut(1);
        let out = &mut out0[0];
        let mut ins: Vec<&[f32]> = Vec::with_capacity(in_bufs.len());
        for (k, &i) in in_bufs.iter().enumerate() {
            match alias_of[k] {
                Some(ai) => ins.push(&aliased[ai]),
                None if i < out_buf => ins.push(&before[i]),
                _ => ins.push(&after[i - out_buf - 1]),
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
                let pos: Vec<usize> =
                    (0..nt).map(|t| ins[1][t].to_bits() as usize).collect();
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
                    crate::vec_ops::vec_soft_max_f32(out.len(), out, &s_in, mx);
                    Ok(())
                } else {
                    Err(format!("Softmax dim {dim} not supported (Phase 2)"))
                }
            }
            Op::SwiGLU => {
                // silu(gate) * up
                crate::vec_ops::vec_silu_f32(out.len(), out, ins[0]);
                let g = out.to_vec();
                crate::vec_ops::vec_mul_f32(out.len(), out, &g, ins[1]);
                Ok(())
            }
            Op::KvcacheStore { .. } => unreachable!("KV store handled in the dedicated path"),
            Op::KvcacheLoad { .. } => Ok(()), // view of the K region

            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                // Phase 2: identity (shape/layout metadata; data unchanged)
                out.copy_from_slice(ins[0]);
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
                let k_slice: &[f32] = if k_id < out_buf {
                    &before[k_id]
                } else {
                    &after[k_id - out_buf - 1]
                };
                let v_slice: &[f32] = if v_id < out_buf {
                    &before[v_id]
                } else {
                    &after[v_id - out_buf - 1]
                };
                let n_ctx = k_slice.len() / nkt;
                // current KV size = max position + 1
                let nkv = (0..nt)
                    .map(|t| ins[2][t].to_bits() as usize + 1)
                    .max()
                    .unwrap_or(0)
                    .min(n_ctx);
                let pos: Vec<usize> = (0..nt).map(|t| ins[2][t].to_bits() as usize).collect();
                let mut scrs = vec![0.0f32; nkv.max(1)];
                cpu_gqa_attn(
                    ins[0], k_slice, v_slice, &pos, nt, nkv, meta.n_head, meta.n_head_kv, meta.hd,
                    meta.hd_kv, nkt, out, &mut scrs, meta.scale,
                )?;
                Ok(())
            }
            Op::FusedBiasRope | Op::BatchMatMul | Op::FusedQKV { .. } | Op::FusedFFN => {
                Err(format!("op {:?} unsupported on CPU (fusion not enabled for it)", node.op))
            }
        }
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
pub(crate) fn cpu_gqa_attn(
    q: &[f32],
    ka: &[f32],
    va: &[f32],
    pos: &[usize],
    nt: usize,
    nkv: usize,
    nh: usize,
    nk: usize,
    hd: usize,
    hd_kv: usize,
    nkt: usize,
    out: &mut [f32],
    scrs: &mut [f32],
    scale: f32,
) -> Result<(), String> {
    if hd < hd_kv {
        return Err(format!("Q head dim ({hd}) must be >= KV head dim ({hd_kv})"));
    }
    let gqa = nh / nk;
    let ne_q = nh * hd;
    for h in 0..nh {
        let hk = h / gqa;
        for t in 0..nt {
            let qs = t * ne_q + h * hd;
            let vl = (pos[t] + 1).min(nkv);
            let mut mx = f32::NEG_INFINITY;
            for kv in 0..vl {
                let ks = kv * nkt + hk * hd_kv;
                let s = crate::vec_ops::vec_dot_f32(hd_kv, &q[qs..qs + hd_kv], &ka[ks..ks + hd_kv])
                    * scale;
                scrs[kv] = s;
                if s > mx {
                    mx = s;
                }
            }
            for kv in vl..nkv {
                scrs[kv] = f32::NEG_INFINITY;
            }
            let s_in = scrs[..nkv].to_vec();
            let sm = crate::vec_ops::vec_soft_max_f32(nkv, scrs, &s_in, mx);
            let is = (1.0 / sm) as f32;
            crate::vec_ops::vec_scale_f32(nkv, scrs, is);
            let os = t * ne_q + h * hd;
            out[os..os + hd].fill(0.0);
            let vs_base = hk * hd_kv;
            for kv in 0..nkv {
                crate::vec_ops::vec_muladd_f32(
                    hd_kv,
                    &mut out[os..os + hd_kv],
                    &va[kv * nkt + vs_base..kv * nkt + vs_base + hd_kv],
                    scrs[kv],
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
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
            Self { sched: BackendScheduler::new(), alloc: GraphAllocator::new() }
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
        let m = gb.matmul(x, h.alloc.cpu().weight("W").unwrap(), Some(h.alloc.cpu().weight("b").unwrap()));
        let s = gb.silu(m);
        let o = gb.node("scale2", Op::Scale(2.0), &[s], [3, 1, 1, 1], DType::F32, NodeMeta::None);
        gb.output(o);
        let g = gb.build();

        h.run(&g, &[("x", vec![1.0, 2.0, 3.0, 4.0])]);
        let silu = |v: f32| v / (1.0 + (-v).exp());
        let expect = [silu(1.5) * 2.0, silu(3.0) * 2.0, silu(9.25) * 2.0];
        let got = h.out(&g, o);
        for i in 0..3 {
            assert!((got[i] - expect[i]).abs() < 1e-4, "out[{i}]={} expect {}", got[i], expect[i]);
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
            assert!((got[i] - ref_out[i]).abs() < 1e-6, "norm[{i}] {} vs {}", got[i], ref_out[i]);
        }
    }

    #[test]
    fn embedding_and_rope() {
        // vocab 4, n_embd 4: ids [0,2] -> rows, then rope per 2 heads of hd 2
        let mut h = Harness::new();
        h.reg(tensor_f32(
            "tok_embd",
            [4, 4, 1, 1],
            vec![0.1, 0.2, 0.3, 0.4, 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3, 2.4, 3.1, 3.2, 3.3, 3.4],
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
        h.sched.execute(&g, &mut h.alloc).unwrap();
        let got = h.out(&g, rope);
        // reference: embed rows then rope per head
        let mut ref_x = vec![
            0.1, 0.2, 0.3, 0.4, // id 0
            2.1, 2.2, 2.3, 2.4, // id 2
        ];
        cpu_rope(&mut ref_x, &[0, 1], 2, 2, 10000.0, 1.0, RopeStyle::NonInterleaved);
        for i in 0..8 {
            assert!((got[i] - ref_x[i]).abs() < 1e-5, "rope[{i}] {} vs {}", got[i], ref_x[i]);
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
        let _st = gb.kvcache_store(0, k, v, pos, 8);
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
        h.alloc.fill_input(&g, "q", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        h.alloc.fill_input(&g, "k", &[1.0, 0.0, 0.0, 1.0]).unwrap();
        h.alloc.fill_input(&g, "v", &[0.5, 0.5, 0.25, 0.75]).unwrap();
        h.sched.execute(&g, &mut h.alloc).unwrap();
        let got = h.out(&g, out);
        // scores: h0: dot([1,0],[1,0])*0.5 = 0.5; h1: dot([0,1],[0,1])*0.5 = 0.5
        // softmax([0.5]) = 1.0 -> out = v
        assert!((got[0] - 0.5).abs() < 1e-5, "got[0]={}", got[0]);
        assert!((got[1] - 0.5).abs() < 1e-5, "got[1]={}", got[1]);
        assert!((got[2] - 0.25).abs() < 1e-5, "got[2]={}", got[2]);
        assert!((got[3] - 0.75).abs() < 1e-5, "got[3]={}", got[3]);
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
        alloc.fill_input(&g, "x", &[1.0,2.0,3.0,4.0, 10.0,20.0,30.0,40.0, 100.0,200.0,300.0,400.0]).unwrap();
        alloc.fill_input_i32(&g, "ids", &[2]).unwrap();
        sched.execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, r).unwrap();
        assert_eq!(got, &[100.0, 200.0, 300.0, 400.0], "get_rows should select row 2");

        // ids = [0]
        alloc.fill_input_i32(&g, "ids", &[0]).unwrap();
        sched.execute(&g, &mut alloc).unwrap();
        let got = alloc.get_buffer(&g, r).unwrap();
        assert_eq!(got, &[1.0, 2.0, 3.0, 4.0], "get_rows should select row 0");
    }
}
