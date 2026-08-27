//! Metal (MPS) backend — Phase 3.
//!
//! Executes IR nodes by dispatching to MpsState's existing per-op kernels
//! (rms_norm, quant_matmul_f32_on_gpu_buf, rope_f32, silu_f32, add_f32,
//! mul_f32, swiglu_f32, embed_tokens_gpu, store_kv, gqa_attn_f32_off).
//!
//! Buffers are shared-memory MTLBuffers (host + GPU visible), so read_host /
//! write_host are direct memory views — cross-backend copies are plain host
//! round trips. One `MpsCommandBuffer` is kept per split and submitted by
//! `synchronize()` (called at split boundaries), so ops within a split share a
//! single GPU submission — the plan's §15 "split shares one command buffer"
//! rule.
//!
//! GPU safety (docs/GPU_SAFETY.md): kernel-invariant violations return Err —
//! the caller must not treat them as a silent CPU fallback; supported-model
//! constraints are checked up front (nkt == nk*hd for attention).

use crate::metal::MpsState;
#[cfg(target_os = "macos")]
use objc2_metal::MTLBuffer;

use super::backend::Backend;
use super::ops::{FusedOp, NodeMeta, Op};
use super::{CNode, DType};

// ─── op profiler (MINFER_OP_PROFILE=1, debug aid) ────────────────────
// Host-side encode time per op label (accumulated across the process) plus
// per-submit GPU wait time. The first submit prints the full per-op table
// (prefill); later submits print one line each (decode = one submit per token).
// Zero overhead when the env var is unset.
use std::collections::BTreeMap;
static OP_ENC: std::sync::Mutex<BTreeMap<String, (u64, f64)>> =
    std::sync::Mutex::new(BTreeMap::new());
static GPU_MS: std::sync::Mutex<(u64, f64)> = std::sync::Mutex::new((0, 0.0));

fn op_profile_enabled() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("MINFER_OP_PROFILE").map_or(false, |v| v == "1"))
}

/// Records host-side encode time per op label on drop (works with early returns).
struct EncTimer {
    key: String,
    t0: std::time::Instant,
}
impl Drop for EncTimer {
    fn drop(&mut self) {
        let ms = self.t0.elapsed().as_secs_f64() * 1e3;
        let mut m = OP_ENC.lock().unwrap();
        let e = m.entry(std::mem::take(&mut self.key)).or_insert((0, 0.0));
        e.0 += 1;
        e.1 += ms;
    }
}

pub struct MetalBackend {
    state: &'static MpsState,
    /// f32-element pool: id → shared MTLBuffer (size * 4 bytes)
    pool: Vec<crate::metal::MetalBuffer>,
    free: Vec<usize>,
    /// P2/P3 capture staging: host-readable buffers written by per-split blits.
    /// Only allocated while a trace/live capture is armed.
    staging: Vec<crate::metal::MetalBuffer>,
    free_staging: Vec<usize>,
    /// Pending command buffer for the current split. Stored as a leaked box
    /// pointer (null = none) because MpsCommandBuffer is !Send/!Sync; all
    /// access happens sequentially through &self/&mut self methods on the
    /// scheduler thread, so the raw pointer is contained.
    cb_ptr: *mut crate::metal::MpsCommandBuffer<'static>,
}

// Safety: every field is either owned (pool/free), a 'static reference
// (MpsState is a Sync singleton), or the command-buffer pointer which is only
// dereferenced inside &self/&mut self methods (sequential, single-threaded).
unsafe impl Send for MetalBackend {}
unsafe impl Sync for MetalBackend {}

impl MetalBackend {
    /// P2/P3 capture: encode blits copying `src_ids` (pool buffers) into
    /// staging buffers, at the END of this split's command buffer (after all
    /// kernels, so the data is this step's output). Returns the staging ids —
    /// their contents are valid only after the next `synchronize`.
    pub fn capture_split(&mut self, src_ids: &[usize]) -> Result<Vec<usize>, String> {
        let mut dst_ids = Vec::with_capacity(src_ids.len());
        for &sid in src_ids {
            let len = self.buf(sid).length() as usize;
            dst_ids.push(self.staging_alloc(len)?);
        }
        let pairs: Vec<(usize, usize)> = src_ids
            .iter()
            .copied()
            .zip(dst_ids.iter().copied())
            .collect();
        self.cb()
            .encode_captures(&pairs, &self.pool, &self.staging)?;
        Ok(dst_ids)
    }

    fn staging_alloc(&mut self, len_bytes: usize) -> Result<usize, String> {
        if let Some(pos) = self
            .free_staging
            .iter()
            .position(|&id| self.staging[id].length() as usize == len_bytes)
        {
            return Ok(self.free_staging.swap_remove(pos));
        }
        if len_bytes % 4 != 0 {
            return Err(format!("capture staging size {len_bytes} not 4-aligned"));
        }
        let buf = self.state.new_f32_buffer(len_bytes / 4);
        self.staging.push(buf);
        Ok(self.staging.len() - 1)
    }

    /// Read a staging buffer (valid after the split's command buffer was
    /// submitted by `synchronize`).
    pub fn read_staging(&self, id: usize) -> Option<&[f32]> {
        let buf = self.staging.get(id)?;
        let len = (buf.length() as usize) / 4;
        Some(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) })
    }

    /// Return all staging buffers to the free list (call after the readback of
    /// one split's captures; they may be reused by the next split).
    pub fn release_staging_all(&mut self) {
        self.free_staging = (0..self.staging.len()).collect();
    }

    /// None when MPS is unavailable or not initialized (MpsState::init()).
    pub fn new() -> Option<Self> {
        let state = MpsState::get()?;
        Some(Self {
            state,
            pool: Vec::new(),
            free: Vec::new(),
            staging: Vec::new(),
            free_staging: Vec::new(),
            cb_ptr: std::ptr::null_mut(),
        })
    }

    fn buf(&self, id: usize) -> &crate::metal::MetalBuffer {
        &self.pool[id]
    }

    /// The current split's command buffer (created on first op of a split).
    /// The box is leaked, so the returned reference is 'static and does not
    /// borrow `self` — callers can freely touch the pool afterwards.
    fn cb(&mut self) -> &'static mut crate::metal::MpsCommandBuffer<'static> {
        if self.cb_ptr.is_null() {
            let cb = Box::new(self.state.cmd_buffer());
            self.cb_ptr = Box::into_raw(cb);
        }
        // SAFETY: cb_ptr is null or points to a live box created here; all
        // callers hold &mut self, so no concurrent mutation.
        unsafe { &mut *self.cb_ptr }
    }

    /// Submit the pending command buffer (if any) and clear it.
    fn submit_pending(&mut self) {
        if !self.cb_ptr.is_null() {
            let t0 = std::time::Instant::now();
            // SAFETY: exclusive &mut self — take the box back and submit.
            let cb = unsafe { Box::from_raw(self.cb_ptr) };
            self.cb_ptr = std::ptr::null_mut();
            cb.submit()
                .expect("MPS: graph backend command-buffer submit error");
            if op_profile_enabled() {
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                let n = {
                    let mut g = GPU_MS.lock().unwrap();
                    g.0 += 1;
                    g.1 += ms;
                    g.0
                };
                if n == 1 {
                    Self::print_profile();
                } else {
                    eprintln!(
                        "[MINFER_OP_PROFILE] submit #{n}: GPU {ms:.2} ms (total {:.1} ms)",
                        GPU_MS.lock().unwrap().1
                    );
                }
            }
        }
    }

    /// KV-parallel attention chunk count (decode), mirroring layer_gpu's
    /// adaptive rule: one chunk per 32 KV rows, capped at 16, with a
    /// MINFER_ATTN_CHUNKS override.
    fn attention_chunks(&self, positions: &crate::metal::MetalBuffer) -> usize {
        let max_pos = Self::positions_max(positions);
        std::env::var("MINFER_ATTN_CHUNKS")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&c| c >= 1)
            .unwrap_or_else(|| ((max_pos + 1 + 31) / 32).clamp(1, 16))
    }

    /// max(positions) + host-side read of the (host-written) I32 positions
    /// buffer — the positions are input data, never GPU-computed, so a host
    /// read is safe here.
    fn positions_max(positions: &crate::metal::MetalBuffer) -> usize {
        let n = (positions.length() as usize) / 4;
        let p =
            unsafe { std::slice::from_raw_parts(positions.contents().as_ptr() as *const u32, n) };
        p.iter().map(|&x| x as usize).max().unwrap_or(0)
    }

    fn copy_in(&self, dst: usize, src: usize) {
        // in-place-ish ops (silu/rope) may alias; snapshot to dst first
        let src_buf = self.buf(src);
        let dst_buf = self.buf(dst);
        let n = (src_buf.length().min(dst_buf.length()) / 4) as usize;
        unsafe {
            std::ptr::copy_nonoverlapping(
                src_buf.contents().as_ptr() as *const f32,
                dst_buf.contents().as_ptr() as *mut f32,
                n,
            );
        }
    }

    fn print_profile() {
        let enc = OP_ENC.lock().unwrap();
        let gpu = GPU_MS.lock().unwrap();
        let mut rows: Vec<_> = enc.iter().collect();
        rows.sort_by(|a, b| b.1 .1.partial_cmp(&a.1 .1).unwrap());
        eprintln!(
            "\n[MINFER_OP_PROFILE] host-encode per op (cumulative), GPU submits={}:",
            gpu.0
        );
        let mut enc_total = 0.0;
        for (k, (n, ms)) in rows.iter().take(20) {
            enc_total += ms;
            eprintln!("  {ms:9.3} ms  x{n:4}  {k}");
        }
        eprintln!(
            "  host encode (top20): {enc_total:.3} ms; GPU wait: {:.3} ms over {} submits",
            gpu.1, gpu.0
        );
    }
}

impl Drop for MetalBackend {
    fn drop(&mut self) {
        // flush any pending command buffer (never leave an unterminated encoder)
        if !self.cb_ptr.is_null() {
            // SAFETY: dropping the backend — take the box back and submit.
            let cb = unsafe { Box::from_raw(self.cb_ptr) };
            self.cb_ptr = std::ptr::null_mut();
            let _ = cb.submit();
        }
        if op_profile_enabled() {
            Self::print_profile();
        }
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &str {
        "metal"
    }

    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        match op {
            Op::Input => true,
            Op::Add | Op::Mul | Op::Silu | Op::RmsNorm { .. } | Op::QkNorm { .. } | Op::SwiGLU => {
                dtype == DType::F32
            }
            Op::MatMul { .. } => {
                matches!(dtype, DType::F32) // activations are f32; weight type in meta
            }
            Op::GetRows | Op::RoPE { .. } | Op::Attn { .. } => dtype == DType::F32,
            Op::KvcacheStore { .. } | Op::KvcacheLoad { .. } => dtype == DType::F32,
            Op::FusedQKV { .. } | Op::FusedQkvNorm { .. } | Op::FusedFFN => dtype == DType::F32,
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => true,
            Op::Scale(_) | Op::Softmax { .. } | Op::FusedBiasRope | Op::BatchMatMul => false,
        }
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        // swiglu_f32 and attn_bias_rope_store kernels exist (the latter is the
        // fused decode QKV store path, nt==1 only)
        matches!(fused, FusedOp::SwiGLU | FusedOp::QKVBiasRopeStore)
    }

    fn alloc_buffer(&mut self, size: usize) -> usize {
        if let Some(idx) = self
            .free
            .iter()
            .position(|&id| self.pool[id].length() as usize == size * 4)
        {
            return self.free.swap_remove(idx);
        }
        self.pool.push(self.state.new_f32_buffer(size));
        self.pool.len() - 1
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
        let cb = self.cb();
        let _t = if op_profile_enabled() {
            Some(EncTimer {
                key: format!("{:?}", node.op),
                t0: std::time::Instant::now(),
            })
        } else {
            None
        };
        match &node.op {
            Op::Input => Ok(()),
            Op::Silu => {
                if in_bufs[0] != out_buf {
                    self.copy_in(out_buf, in_bufs[0]);
                }
                let n = self.pool[out_buf].length() as usize / 4;
                cb.silu_f32(self.buf(out_buf), n);
                Ok(())
            }
            Op::Add => {
                cb.add_f32(
                    self.buf(in_bufs[0]),
                    self.buf(in_bufs[1]),
                    self.buf(out_buf),
                    self.pool[out_buf].length() as usize / 4,
                );
                Ok(())
            }
            Op::Mul => {
                cb.mul_f32(
                    self.buf(in_bufs[0]),
                    self.buf(in_bufs[1]),
                    self.buf(out_buf),
                    self.pool[out_buf].length() as usize / 4,
                );
                Ok(())
            }
            Op::RmsNorm { eps } => {
                let w = match &node.meta {
                    NodeMeta::Norm(m) => m
                        .weight_name
                        .as_ref()
                        .and_then(|n| self.state.weight_buf(n)),
                    _ => None,
                };
                let d = node.out_shape[0];
                let n = node.out_shape[1];
                // G2: 256-thread kernel when enabled (METAL_OPTIMIZATIONS #16)
                match w {
                    Some((wb, w_off)) => {
                        if crate::metal::rms_norm_256_enabled() {
                            cb.rms_norm_256(
                                self.buf(in_bufs[0]),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        } else {
                            cb.rms_norm(
                                self.buf(in_bufs[0]),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        }
                    }
                    None => cb.rms_norm(
                        self.buf(in_bufs[0]),
                        None,
                        0,
                        self.buf(out_buf),
                        d,
                        n,
                        *eps,
                        0,
                        0,
                    ),
                }
                Ok(())
            }
            Op::QkNorm { hd, nh, eps } => {
                // Per-head norm: contiguous [nt*nh, hd] rows — same kernel as
                // RmsNorm with d = hd and n = nt*nh (weight length hd).
                let w = match &node.meta {
                    NodeMeta::Norm(m) => m
                        .weight_name
                        .as_ref()
                        .and_then(|n| self.state.weight_buf(n)),
                    _ => None,
                };
                let d = *hd;
                let n = (self.pool[out_buf].length() as usize / 4) / d;
                let _ = nh;
                match w {
                    Some((wb, w_off)) => {
                        if crate::metal::rms_norm_256_enabled() {
                            cb.rms_norm_256(
                                self.buf(in_bufs[0]),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        } else {
                            cb.rms_norm(
                                self.buf(in_bufs[0]),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        }
                    }
                    None => cb.rms_norm(
                        self.buf(in_bufs[0]),
                        None,
                        0,
                        self.buf(out_buf),
                        d,
                        n,
                        *eps,
                        0,
                        0,
                    ),
                }
                Ok(())
            }
            Op::MatMul { .. } => {
                let meta = match &node.meta {
                    NodeMeta::MatMul(m) => m,
                    other => return Err(format!("matmul node missing MatMulMeta: {other:?}")),
                };

                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.weight_name)
                    .ok_or_else(|| format!("weight '{}' not on GPU", meta.weight_name))?;
                let nt = node.out_shape[1];
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0]),
                    0,
                    self.buf(out_buf),
                    meta.out_dim,
                    meta.in_dim,
                    nt,
                );
                if let Some(bname) = &meta.bias_name {
                    let (bb, b_off) = self
                        .state
                        .weight_buf(bname)
                        .ok_or_else(|| format!("bias '{}' not on GPU", bname))?;
                    cb.add_bias_f32(self.buf(out_buf), &bb, b_off, meta.out_dim, nt, 0);
                }
                Ok(())
            }
            Op::GetRows => {
                match &node.meta {
                    NodeMeta::Embed(m) => {
                        let (wb, w_off) = self
                            .state
                            .weight_buf(&m.weight_name)
                            .ok_or_else(|| format!("embedding '{}' not on GPU", m.weight_name))?;
                        let ne = node.out_shape[0];
                        let nt = node.out_shape[1];
                        cb.embed_tokens_gpu(
                            &wb,
                            w_off,
                            self.buf(in_bufs[0]),
                            self.buf(out_buf),
                            ne,
                            nt,
                            m.weight_ttype,
                        );
                        Ok(())
                    }
                    NodeMeta::None => {
                        // generic row selection: out[t] = x[ids[t]] (n_out tail)
                        let ne = node.out_shape[0];
                        let nt = node.out_shape[1];
                        cb.get_rows_f32(
                            self.buf(in_bufs[0]),
                            self.buf(in_bufs[1]),
                            self.buf(out_buf),
                            ne,
                            nt,
                        );
                        Ok(())
                    }
                    other => Err(format!("get_rows node with unexpected meta: {other:?}")),
                }
            }
            Op::RoPE { style } => {
                let meta = match &node.meta {
                    NodeMeta::Rope(m) => m,
                    other => return Err(format!("rope node missing RoPEMeta: {other:?}")),
                };
                if in_bufs[0] != out_buf {
                    self.copy_in(out_buf, in_bufs[0]);
                }
                let nt = node.out_shape[1];
                cb.rope_f32(
                    self.buf(out_buf),
                    meta.n_head,
                    meta.hd,
                    nt,
                    meta.freq_base,
                    meta.freq_scale,
                    self.buf(in_bufs[1]),
                    *style as i32,
                    0,
                );
                Ok(())
            }
            Op::SwiGLU => {
                let n = self.pool[out_buf].length() as usize / 4;
                cb.swiglu_f32(
                    self.buf(in_bufs[0]),
                    self.buf(in_bufs[1]),
                    self.buf(out_buf),
                    n,
                );
                Ok(())
            }
            Op::KvcacheStore { layer } => {
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                let nkt = node.out_shape[0];
                let nt = (self.pool[in_bufs[0]].length() as usize / 4) / nkt;
                cb.store_kv(
                    self.buf(in_bufs[0]),
                    self.buf(k_id),
                    nkt,
                    nt,
                    self.buf(in_bufs[2]),
                    0,
                );
                cb.store_kv(
                    self.buf(in_bufs[1]),
                    self.buf(v_id),
                    nkt,
                    nt,
                    self.buf(in_bufs[2]),
                    0,
                );
                Ok(())
            }
            Op::KvcacheLoad { .. } => Ok(()), // view of the K region
            Op::Attn { .. } => {
                let meta = match &node.meta {
                    NodeMeta::Attn(m) => m,
                    other => return Err(format!("attn node missing AttnMeta: {other:?}")),
                };
                // GPU safety (H1): kernel_gqa_attn strides KV by nk*hd
                if meta.nkt != meta.n_head_kv * meta.hd {
                    return Err(format!(
                        "Metal attention: nkt={} != n_head_kv*hd={} (kernel_gqa_attn strides KV by nk*hd)",
                        meta.nkt, meta.n_head_kv * meta.hd
                    ));
                }
                if meta.hd != meta.hd_kv {
                    return Err(format!(
                        "Metal attention: hd={} != hd_kv={} (kernel_gqa_attn uses query head dim)",
                        meta.hd, meta.hd_kv
                    ));
                }
                let (k_id, v_id) = kv_pair
                    .ok_or_else(|| format!("KV regions for layer {} not allocated", meta.layer))?;
                let nt = node.out_shape[1];
                // G1: dispatch the fast attention kernels (flash / split /
                // parallel) exactly like the legacy layer_gpu path. The fast
                // paths are gated to the isolation-tested shapes (hd 64/128);
                // anything else falls back to the classic kernel.
                let k = self.buf(k_id);
                let v = self.buf(v_id);
                let q = self.buf(in_bufs[0]);
                let o = self.buf(out_buf);
                let positions = self.buf(in_bufs[2]);
                if nt == 1 {
                    if crate::metal::flash_attn_enabled(meta.hd) {
                        let chunks = self.attention_chunks(positions);
                        cb.gqa_attn_flash(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            1,
                            chunks,
                        );
                    } else if (meta.hd == 64 || meta.hd == 128)
                        && !std::env::var("MINFER_NO_SPLIT_ATTN").map_or(false, |v| v == "1")
                    {
                        let chunks = self.attention_chunks(positions);
                        cb.gqa_attn_split_f32(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            1,
                            chunks,
                        );
                    } else {
                        cb.gqa_attn_f32(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            1,
                        );
                    }
                } else if meta.hd == 64 || meta.hd == 128 {
                    let max_pos = Self::positions_max(positions);
                    let nkv = max_pos + 1;
                    if crate::metal::prefill_flash_enabled(meta.hd) {
                        cb.attn_flash_prefill(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            nkv,
                            meta.nkt,
                            nt,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                        );
                    } else if crate::metal::matmul_attn_enabled() {
                        cb.attn_parallel_prefill(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            nkv,
                            meta.nkt,
                            meta.n_head * meta.hd,
                            nt,
                            meta.n_head,
                            meta.hd,
                            meta.n_head / meta.n_head_kv,
                            meta.scale,
                        );
                    } else {
                        cb.gqa_attn_f32(
                            q,
                            k,
                            v,
                            o,
                            positions,
                            meta.n_head,
                            meta.n_head_kv,
                            meta.hd,
                            meta.scale,
                            nt,
                        );
                    }
                } else {
                    cb.gqa_attn_f32(
                        q,
                        k,
                        v,
                        o,
                        positions,
                        meta.n_head,
                        meta.n_head_kv,
                        meta.hd,
                        meta.scale,
                        nt,
                    );
                }
                Ok(())
            }
            Op::View { .. } | Op::Reshape { .. } | Op::Permute { .. } => {
                if in_bufs[0] != out_buf {
                    self.copy_in(out_buf, in_bufs[0]);
                }
                Ok(())
            }
            Op::FusedFFN => {
                let meta = match &node.meta {
                    NodeMeta::FusedFfn(m) => m,
                    other => return Err(format!("fused_ffn node missing FusedFfnMeta: {other:?}")),
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.gu_weight)
                    .ok_or_else(|| format!("gate+up weight '{}' not on GPU", meta.gu_weight))?;
                let nt = node.out_shape[1];
                debug_assert!(nt == 1, "FusedFFN is decode (nt==1) only, got nt={nt}");
                let od_total = 2 * meta.nf;
                // 1) concat matmul: x × [ffn_gate|ffn_up] → gate|up concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0]),
                    0,
                    self.buf(out_buf),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                // 2) swiglu in place: silu(gate rows 0..nf) * up rows nf..2*nf
                //    (llama ggml_swiglu_split); result written back to gate rows
                let n = nt * meta.nf;
                cb.swiglu_f32_off(
                    self.buf(out_buf),
                    self.buf(out_buf),
                    self.buf(out_buf),
                    n,
                    n,
                );
                if std::env::var("MINFER_FFNDEBUG").is_ok() {
                    self.submit_pending();
                    let nb = (self.pool[out_buf].length() as usize) / 4;
                    let ob = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(out_buf).contents().as_ptr() as *const f32,
                            nb,
                        )
                    };
                    eprintln!(
                        "[ffn-out] FusedFFN out_buf={out_buf} len={nb} first4={:?} last4={:?}",
                        &ob[..4],
                        &ob[nb - 4..]
                    );
                }
                Ok(())
            }
            Op::FusedQKV { layer } => {
                let meta = match &node.meta {
                    NodeMeta::FusedQkv(m) => m,
                    other => return Err(format!("fused_qkv node missing FusedQkvMeta: {other:?}")),
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.qkv_weight)
                    .ok_or_else(|| format!("qkv weight '{}' not on GPU", meta.qkv_weight))?;
                let nt = node.out_shape[1];
                debug_assert!(nt == 1, "FusedQKV is decode (nt==1) only, got nt={nt}");
                let od_total = meta.nqt + 2 * meta.nkt;
                // 1) concat matmul: x × [wq|wk|wv] → q|k|v concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0]),
                    0,
                    self.buf(out_buf),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                // 2) fused bias + rope + KV store in one kernel pass
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                let bias_off =
                    |name: &Option<String>| -> Result<(crate::metal::MetalBuffer, u64), String> {
                        match name {
                            Some(n) => self
                                .state
                                .weight_buf(n)
                                .ok_or_else(|| format!("bias '{n}' not on GPU")),
                            None => Err("fused QKV bias missing".into()),
                        }
                    };
                let bq = bias_off(&meta.bias_q)?;
                let bk = bias_off(&meta.bias_k)?;
                let bv = bias_off(&meta.bias_v)?;
                let (bq_b, bq_o) = bq;
                let (bk_b, bk_o) = bk;
                let (bv_b, bv_o) = bv;
                let pos = {
                    let n = (self.buf(in_bufs[1]).length() as usize) / 4;
                    let p = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(in_bufs[1]).contents().as_ptr() as *const u32,
                            n,
                        )
                    };
                    p[0] as i32
                };

                cb.attn_bias_rope_store(
                    self.buf(out_buf),
                    &bq_b,
                    bq_o,
                    &bk_b,
                    bk_o,
                    &bv_b,
                    bv_o,
                    self.buf(k_id),
                    self.buf(v_id),
                    meta.nqt,
                    meta.nkt,
                    meta.hd,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                    meta.rope_style as i32,
                );
                Ok(())
            }
            Op::FusedQkvNorm { layer } => {
                let meta = match &node.meta {
                    NodeMeta::FusedQkvNorm(m) => m,
                    other => {
                        return Err(format!(
                            "fused_qkv_norm node missing FusedQkvNormMeta: {other:?}"
                        ))
                    }
                };
                let (wb, w_off) = self
                    .state
                    .weight_buf(&meta.qkv_weight)
                    .ok_or_else(|| format!("qkv weight '{}' not on GPU", meta.qkv_weight))?;
                // per-head Q/K RMSNorm weights (llama attn_q_norm / attn_k_norm)
                let norm_off =
                    |name: &Option<String>| -> Result<(crate::metal::MetalBuffer, u64), String> {
                        match name {
                            Some(n) => self
                                .state
                                .weight_buf(n)
                                .ok_or_else(|| format!("norm '{n}' not on GPU")),
                            None => Err("fused QKV norm weight missing".into()),
                        }
                    };
                let (qn_b, qn_o) = norm_off(&meta.q_norm_name)?;
                let (kn_b, kn_o) = norm_off(&meta.k_norm_name)?;
                let nt = node.out_shape[1];
                debug_assert!(nt == 1, "FusedQkvNorm is decode (nt==1) only, got nt={nt}");
                let od_total = meta.nqt + 2 * meta.nkt;
                // 1) concat matmul: x × [wq|wk|wv] → q|k|v concat buffer
                cb.quant_matmul_f32_on_gpu_buf(
                    &wb,
                    w_off,
                    meta.weight_ttype,
                    self.buf(in_bufs[0]),
                    0,
                    self.buf(out_buf),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                // 2) per-head RMSNorm on q/k IN PLACE on the concat buffer
                //    (llama build_norm(Qcur/Kcur, attn_q_norm/attn_k_norm) before
                //    ggml_rope_ext). q section is at byte offset 0; k section at
                //    byte offset nqt*4. Reuse the rms_norm_256 kernel (d=hd rows).
                let off_q = 0u64;
                let off_k = (meta.nqt * 4) as u64;
                let n_q = meta.nh;
                let n_k = meta.nk;
                if crate::metal::rms_norm_256_enabled() {
                    cb.rms_norm_256(
                        self.buf(out_buf),
                        Some(&qn_b),
                        qn_o,
                        self.buf(out_buf),
                        meta.hd,
                        n_q,
                        meta.eps,
                        off_q,
                        off_q,
                    );
                    cb.rms_norm_256(
                        self.buf(out_buf),
                        Some(&kn_b),
                        kn_o,
                        self.buf(out_buf),
                        meta.hd,
                        n_k,
                        meta.eps,
                        off_k,
                        off_k,
                    );
                } else {
                    cb.rms_norm(
                        self.buf(out_buf),
                        Some(&qn_b),
                        qn_o,
                        self.buf(out_buf),
                        meta.hd,
                        n_q,
                        meta.eps,
                        off_q,
                        off_q,
                    );
                    cb.rms_norm(
                        self.buf(out_buf),
                        Some(&kn_b),
                        kn_o,
                        self.buf(out_buf),
                        meta.hd,
                        n_k,
                        meta.eps,
                        off_k,
                        off_k,
                    );
                }
                // 3) no-bias rope + KV store (q in place, k rope+store, v store)
                let pos = {
                    let n = (self.buf(in_bufs[1]).length() as usize) / 4;
                    let p = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(in_bufs[1]).contents().as_ptr() as *const u32,
                            n,
                        )
                    };
                    p[0] as i32
                };
                cb.attn_rope_store(
                    self.buf(out_buf),
                    self.buf(k_id),
                    self.buf(v_id),
                    meta.nqt,
                    meta.nkt,
                    meta.hd,
                    meta.freq_base,
                    meta.freq_scale,
                    pos,
                    meta.rope_style as i32,
                );
                Ok(())
            }
            Op::Scale(_) | Op::Softmax { .. } | Op::FusedBiasRope | Op::BatchMatMul => {
                Err(format!("op {:?} unsupported on Metal (Phase 3)", node.op))
            }
        }
    }

    fn read_host(&self, id: usize) -> Option<&[f32]> {
        let buf = self.pool.get(id)?;
        let len = (buf.length() as usize) / 4;
        Some(unsafe { std::slice::from_raw_parts(buf.contents().as_ptr() as *const f32, len) })
    }

    fn write_host(&mut self, id: usize, data: &[f32]) -> Result<(), String> {
        let buf = self.pool.get(id).ok_or_else(|| format!("no buffer {id}"))?;
        let len = (buf.length() as usize) / 4;
        if len != data.len() {
            return Err(format!(
                "buffer {id}: expected {len} elements, got {}",
                data.len()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                buf.contents().as_ptr() as *mut f32,
                data.len(),
            );
        }
        Ok(())
    }

    fn synchronize(&mut self) {
        self.submit_pending();
    }
}

/// The Metal backend exists only where the trait sees it; helper for the
/// allocator to know whether GPU is available.
pub fn metal_available() -> bool {
    MpsState::get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::alloc::GraphAllocator;
    use crate::graph::backend::Backend;
    use crate::graph::builder::GraphBuilder;
    use crate::graph::scheduler::BackendScheduler;
    use crate::graph::{Backend as Tag, DType};

    fn f32t(name: &str, shape: [i64; 4], data: Vec<f32>) -> crate::tensor::Tensor {
        let mut bytes = Vec::with_capacity(data.len() * 4);
        for x in data {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        let mut t = crate::tensor::Tensor::from_data(crate::tensor::TensorType::F32, &shape, bytes);
        t.name = name.to_string();
        t
    }

    /// GPU graph (silu + add) must match the CPU graph bit-for-bit.
    #[test]
    fn metal_elementwise_matches_cpu() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(backend) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [4, 1, 1, 1], DType::F32);
        let s = gb.silu(x);
        let o = gb.add(s, x);
        gb.output(o);
        let g = gb.build();

        // CPU run
        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input(&g, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
        sched.assign_backends(&mut g.clone(), &ca);
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();

        // Metal run (assign all to Metal)
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
        let splits = sched.split_graph(&g2);
        for (si, sp) in splits.iter().enumerate() {
            eprintln!(
                "[dbg] split {si}: {:?} range {:?} inputs {:?}",
                sp.backend, sp.node_range, sp.inputs
            );
        }
        sched.execute(&g2, &mut alloc).unwrap();
        eprintln!(
            "[dbg] silu out (node 1) = {:?}",
            alloc.copy_to_cpu(1).unwrap()
        );
        eprintln!(
            "[dbg] add out (node 2) = {:?}",
            alloc.copy_to_cpu(2).unwrap()
        );
        let got = alloc.copy_to_cpu(o).unwrap();
        for i in 0..4 {
            assert!(
                (got[i] - expect[i]).abs() < 1e-6,
                "out[{i}] {} vs {}",
                got[i],
                expect[i]
            );
        }
        let _ = backend;
    }

    /// rms_norm on Metal must match CPU within float tolerance.
    #[test]
    fn metal_rmsnorm_matches_cpu() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        // register a norm weight on MPS (name must resolve in weight_buf)
        let wdata: Vec<f32> = (0..8).map(|i| 0.5 + i as f32 * 0.1).collect();
        let mut bytes = Vec::new();
        for x in &wdata {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("nw", &bytes);

        let nw = f32t("nw", [8, 1, 1, 1], wdata);
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [8, 3, 1, 1], DType::F32);
        let r = gb.rms_norm(x, Some(&nw), 1e-5);
        gb.output(r);
        let g = gb.build();

        // CPU
        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.register_weight("nw", nw);
        ca.alloc_graph(&g).unwrap();
        let data: Vec<f32> = (0..24).map(|i| (i as f32 - 12.0) * 0.3).collect();
        ca.fill_input(&g, "x", &data).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, r).unwrap().to_vec();

        // Metal
        let mut sched = BackendScheduler::new();
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &data).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(r).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[rms_norm] max diff {maxd:.3e}");
        assert!(maxd < 1e-4, "rms_norm Metal diverges: {maxd:.3e}");
    }

    /// Cross-backend copies: silu on Metal, input/add on CPU.
    #[test]
    fn metal_cross_backend_copy() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [4, 1, 1, 1], DType::F32);
        let s = gb.silu(x);
        let o = gb.add(s, x);
        gb.output(o);
        let g = gb.build();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input(&g, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();

        // only silu on Metal -> its input (CPU) and output (CPU consumer) cross backends
        let mut g2 = g.clone();
        g2.nodes[0].backend = Some(Tag::CPU);
        g2.nodes[1].backend = Some(Tag::Metal);
        g2.nodes[2].backend = Some(Tag::CPU);
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &[0.5, 1.0, 2.0, -1.0]).unwrap();
        let splits = sched.split_graph(&g2);
        for (si, sp) in splits.iter().enumerate() {
            eprintln!(
                "[dbg] split {si}: {:?} range {:?} inputs {:?}",
                sp.backend, sp.node_range, sp.inputs
            );
        }
        sched.execute(&g2, &mut alloc).unwrap();
        eprintln!(
            "[dbg] silu out (node 1) = {:?}",
            alloc.copy_to_cpu(1).unwrap()
        );
        eprintln!(
            "[dbg] add out (node 2) = {:?}",
            alloc.copy_to_cpu(2).unwrap()
        );
        let got = alloc.copy_to_cpu(o).unwrap();
        for i in 0..4 {
            assert!(
                (got[i] - expect[i]).abs() < 1e-6,
                "cross-backend out[{i}] {} vs {}",
                got[i],
                expect[i]
            );
        }
    }

    /// Real-scale matmul (Q8_0 weight, 896×128 like wk): Metal vs CPU.
    #[test]
    fn metal_matmul_q8_matches_cpu() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let od = 128usize; // output dim
        let inn = 896usize; // input dim // id
                            // random-ish weight [out][in] row-major -> quantize each row to Q8_0
        let wf: Vec<f32> = (0..od * inn)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        let mut wbytes = Vec::new();
        for r in 0..od {
            let row = &wf[r * inn..(r + 1) * inn];
            wbytes.extend_from_slice(&crate::quants::quantize_row_q8_0(row));
        }
        let mut wt = crate::tensor::Tensor::from_data(
            crate::tensor::TensorType::Q8_0,
            &[inn as i64, od as i64, 1, 1],
            wbytes,
        );
        wt.name = "wq8".to_string();
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("wq8", wt.data());

        let nt = 8usize;
        let xd: Vec<f32> = (0..inn * nt)
            .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
            .collect();

        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [inn, nt, 1, 1], DType::F32);
        let m = gb.matmul(x, &wt, None);
        gb.output(m);
        let g = gb.build();

        // manual Q8_0 x f32 reference (weight rows dequantized, f32 activations)
        let mut expect = vec![0.0f32; od * nt];
        {
            let wraw = wt.data();
            let bsz = 34usize; // Q8_0 block: 2 (d) + 32 qs
            for o in 0..od {
                let wrow = &wraw[o * (inn / 32) * bsz..];
                for t in 0..nt {
                    let mut acc = 0.0f32;
                    for b in 0..inn / 32 {
                        let boff = b * bsz;
                        let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                            wrow[boff],
                            wrow[boff + 1],
                        ]));
                        let qs = &wrow[boff + 2..boff + 34];
                        for j in 0..32 {
                            let q = (qs[j] as i8) as f32;
                            acc += q * d * xd[t * inn + b * 32 + j];
                        }
                    }
                    expect[t * od + o] = acc; // token-major [nt][od]
                }
            }
        }

        // Metal
        let mut sched = BackendScheduler::new();
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &xd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(m).unwrap();
        let mut maxd = 0.0f32;
        let mut worst = 0usize;
        let mut nonzero = 0usize;
        for i in 0..got.len() {
            let d = (got[i] - expect[i]).abs();
            if got[i] != 0.0 {
                nonzero += 1;
            }
            if d > maxd {
                maxd = d;
                worst = i;
            }
        }
        eprintln!(
            "[matmul q8] Metal vs manual-Q8x f32 max diff {maxd:.3e} (nonzero {nonzero}/{})",
            got.len()
        );
        assert!(
            maxd < 1e-3,
            "matmul Metal diverges from Q8_0xf32 reference: {maxd:.3e} (worst idx {worst} (t={}, o={}): got {} expect {})",
            worst / od,
            worst % od,
            got[worst],
            expect[worst]
        );
    }

    /// rms_norm at REAL scale (d=896, nt=8, like attn_norm) Metal vs CPU.
    #[test]
    fn metal_rmsnorm_real_scale() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let d = 896usize;
        let nt = 8usize;
        let wdata: Vec<f32> = (0..d).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let mut bytes = Vec::new();
        for x in &wdata {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("nw896", &bytes);
        let nw = f32t("nw896", [d as i64, 1, 1, 1], wdata);
        let xd: Vec<f32> = (0..d * nt)
            .map(|i| ((i * 97) % 200) as f32 / 100.0 - 1.0)
            .collect();

        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [d, nt, 1, 1], DType::F32);
        let r = gb.rms_norm(x, Some(&nw), 1e-6);
        gb.output(r);
        let g = gb.build();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.register_weight("nw896", nw);
        ca.alloc_graph(&g).unwrap();
        ca.fill_input(&g, "x", &xd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, r).unwrap().to_vec();

        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &xd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(r).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[rms_norm 896] max diff {maxd:.3e}");
        assert!(maxd < 1e-4, "rms_norm(896) Metal diverges: {maxd:.3e}");
    }

    /// Cross-backend copy at scale: silu of [896, 8] on Metal.
    #[test]
    fn metal_cross_backend_copy_large() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let d = 896usize;
        let nt = 8usize;
        let xd: Vec<f32> = (0..d * nt)
            .map(|i| ((i * 97) % 200) as f32 / 100.0 - 1.0)
            .collect();
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [d, nt, 1, 1], DType::F32);
        let s = gb.silu(x);
        let o = gb.add(s, x);
        gb.output(o);
        let g = gb.build();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input(&g, "x", &xd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();

        let mut g2 = g.clone();
        g2.nodes[0].backend = Some(Tag::CPU);
        g2.nodes[1].backend = Some(Tag::Metal);
        g2.nodes[2].backend = Some(Tag::CPU);
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &xd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(o).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[cross large] max diff {maxd:.3e}");
        assert!(maxd < 1e-6, "cross-backend large diverges: {maxd:.3e}");
    }

    /// Real pattern: Q8_0 embedding (CPU) -> rms_norm (Metal) with a
    /// cross-backend copy of the embed output in between.
    #[test]
    fn metal_embed_then_rmsnorm_cross_backend() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let ne = 32usize;
        let vocab = 8usize;
        let nt = 4usize;
        // Q8_0 embedding [ne, vocab]
        let ef: Vec<f32> = (0..ne * vocab)
            .map(|i| ((i * 31) % 97) as f32 / 50.0 - 1.0)
            .collect();
        let mut ebytes = Vec::new();
        for r in 0..vocab {
            let row = &ef[r * ne..(r + 1) * ne];
            ebytes.extend_from_slice(&crate::quants::quantize_row_q8_0(row));
        }
        let mut emb = crate::tensor::Tensor::from_data(
            crate::tensor::TensorType::Q8_0,
            &[ne as i64, vocab as i64, 1, 1],
            ebytes,
        );
        emb.name = "embq8".to_string();
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("embq8", emb.data());
        let wdata: Vec<f32> = (0..ne).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();
        let mut wbytes = Vec::new();
        for x in &wdata {
            wbytes.extend_from_slice(&x.to_le_bytes());
        }
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("nwE", &wbytes);
        let nw = f32t("nwE", [ne as i64, 1, 1, 1], wdata);
        let nw2 = nw.clone();
        let ids: Vec<u32> = vec![1, 3, 5, 2];

        let mut gb = GraphBuilder::new();
        let idsn = gb.input("token_ids", [nt, 1, 1, 1], DType::I32);
        let e = gb.embedding(idsn, &emb);
        let r = gb.rms_norm(e, Some(&nw), 1e-5);
        gb.output(r);
        let g = gb.build();

        // all-CPU reference
        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.register_weight("embq8", emb.clone());
        ca.register_weight("nwE", nw);
        ca.alloc_graph(&g).unwrap();
        ca.fill_input_i32(&g, "token_ids", &ids).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, r).unwrap().to_vec();

        // embed CPU + rms_norm Metal
        let mut g2 = g.clone();
        g2.nodes[0].backend = Some(Tag::CPU);
        g2.nodes[1].backend = Some(Tag::CPU);
        g2.nodes[2].backend = Some(Tag::Metal);
        let mut alloc = GraphAllocator::new();
        alloc.register_weight("embq8", emb);
        alloc.register_weight("nwE", nw2);
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input_i32(&g2, "token_ids", &ids).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(r).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[embed+rms] max diff {maxd:.3e}");
        assert!(maxd < 1e-3, "embed->rms cross-backend diverges: {maxd:.3e}");
    }

    /// Multiple Metal nodes alternating with CPU nodes (multi-split sync/copy).
    #[test]
    fn metal_multi_split_alternation() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let d = 64usize;
        let nt = 4usize;
        let wdata: Vec<f32> = (0..d).map(|i| 0.5 + (i % 3) as f32 * 0.2).collect();
        let mut wbytes = Vec::new();
        for x in &wdata {
            wbytes.extend_from_slice(&x.to_le_bytes());
        }
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("nwM", &wbytes);
        let nw = f32t("nwM", [d as i64, 1, 1, 1], wdata);
        let nw2 = nw.clone();
        let xd: Vec<f32> = (0..d * nt)
            .map(|i| ((i * 41) % 199) as f32 / 100.0 - 1.0)
            .collect();

        // x(CPU) -> rms(Metal) -> silu(CPU) -> rms(Metal) -> add(CPU) -> rms(Metal)
        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [d, nt, 1, 1], DType::F32);
        let a = gb.rms_norm(x, Some(&nw), 1e-5);
        let b = gb.silu(a);
        let c = gb.rms_norm(b, Some(&nw), 1e-5);
        let e = gb.add(c, x);
        let f = gb.rms_norm(e, Some(&nw), 1e-5);
        gb.output(f);
        let g = gb.build();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.register_weight("nwM", nw);
        ca.alloc_graph(&g).unwrap();
        ca.fill_input(&g, "x", &xd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, f).unwrap().to_vec();

        let mut g2 = g.clone();
        // 0 CPU, 1 Metal, 2 CPU, 3 Metal, 4 CPU, 5 Metal
        g2.nodes[0].backend = Some(Tag::CPU);
        g2.nodes[1].backend = Some(Tag::Metal);
        g2.nodes[2].backend = Some(Tag::CPU);
        g2.nodes[3].backend = Some(Tag::Metal);
        g2.nodes[4].backend = Some(Tag::CPU);
        g2.nodes[5].backend = Some(Tag::Metal);
        let mut alloc = GraphAllocator::new();
        alloc.register_weight("nwM", nw2);
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &xd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(f).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[multi-split] max diff {maxd:.3e}");
        assert!(maxd < 1e-3, "multi-split alternation diverges: {maxd:.3e}");
    }

    /// Metal KV store + GQA attention vs CPU (F32 inputs, bit-exact check).
    #[test]
    fn metal_attn_kv_matches_cpu() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
        let q = gb.input("q", [8, 2, 1, 1], DType::F32);
        let k = gb.input("k", [8, 2, 1, 1], DType::F32);
        let v = gb.input("v", [8, 2, 1, 1], DType::F32);
        gb.kvcache_store(0, k, v, pos, 16);
        let kv = gb.kvcache_load(0, 8, 16, 2);
        let o = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            crate::graph::ops::AttnMeta {
                layer: 0,
                n_head: 2,
                n_head_kv: 2,
                hd: 4,
                hd_kv: 4,
                nkt: 8,
                scale: 0.5,
            },
        );
        gb.output(o);
        let g = gb.build();

        let qd: Vec<f32> = (0..16)
            .map(|i| ((i * 13) % 29) as f32 / 10.0 - 1.4)
            .collect();
        let kd: Vec<f32> = (0..16)
            .map(|i| ((i * 17) % 31) as f32 / 10.0 - 1.5)
            .collect();
        let vd: Vec<f32> = (0..16)
            .map(|i| ((i * 19) % 37) as f32 / 10.0 - 1.8)
            .collect();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input_i32(&g, "positions", &[0, 1]).unwrap();
        ca.fill_input(&g, "q", &qd).unwrap();
        ca.fill_input(&g, "k", &kd).unwrap();
        ca.fill_input(&g, "v", &vd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();

        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input_i32(&g2, "positions", &[0, 1]).unwrap();
        alloc.fill_input(&g2, "q", &qd).unwrap();
        alloc.fill_input(&g2, "k", &kd).unwrap();
        alloc.fill_input(&g2, "v", &vd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(o).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[attn kv] max diff {maxd:.3e}");
        assert!(maxd < 1e-4, "Metal attention diverges: {maxd:.3e}");
    }

    /// Real Q4_0 matmul (layer-0 wq: [896, 896]) Metal vs manual reference.
    #[test]
    fn metal_matmul_q4_matches_reference() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let od = 896usize;
        let inn = 896usize;
        let nt = 8usize;
        let wf: Vec<f32> = (0..od * inn)
            .map(|i| ((i * 2654435761) % 1000) as f32 / 500.0 - 1.0)
            .collect();
        // quantize to Q4_0: 18 bytes per 32 values (d f16 + 16 nibbles)
        let mut wbytes = Vec::new();
        for r in 0..od {
            let row = &wf[r * inn..(r + 1) * inn];
            for b in 0..inn / 32 {
                let blk = &row[b * 32..b * 32 + 32];
                let mut amax = 0.0f32;
                for &v in blk {
                    amax = amax.max(v.abs());
                }
                let d = if amax == 0.0 { 0.0f32 } else { amax / 127.0 };
                wbytes.extend_from_slice(&half::f16::from_f32(d).to_le_bytes());
                for j in 0..16 {
                    // Q4_0 quantize: q = round(v/d) + 8 in [0,15]. Use i32 for
                    // the +8 offset — v/d can reach ±127 when the row's amax is
                    // near 1 (d = amax/127), so an i8 intermediate overflows
                    // (127+8 > i8::MAX) and panics in debug builds.
                    let q0 = ((blk[j] / d).round() as i32 + 8).clamp(0, 15) as u8;
                    let q1 = ((blk[j + 16] / d).round() as i32 + 8).clamp(0, 15) as u8;
                    wbytes.push(q0 | (q1 << 4));
                }
            }
        }
        let mut wt = crate::tensor::Tensor::from_data(
            crate::tensor::TensorType::Q4_0,
            &[inn as i64, od as i64, 1, 1],
            wbytes,
        );
        wt.name = "wq4".to_string();
        crate::metal::MpsState::get()
            .unwrap()
            .register_weight("wq4", wt.data());

        let xd: Vec<f32> = (0..inn * nt)
            .map(|i| ((i * 1103515245) % 997) as f32 / 500.0 - 1.0)
            .collect();

        // manual Q4_0 x f32 reference
        let mut expect = vec![0.0f32; od * nt];
        {
            let wraw = wt.data();
            for o in 0..od {
                let wrow = &wraw[o * (inn / 32) * 18..];
                for t in 0..nt {
                    let mut acc = 0.0f32;
                    for b in 0..inn / 32 {
                        let boff = b * 18;
                        let d = crate::block::fp16_to_f32(u16::from_le_bytes([
                            wrow[boff],
                            wrow[boff + 1],
                        ]));
                        for j in 0..16 {
                            let byte = wrow[boff + 2 + j];
                            let q0 = ((byte & 0x0F) as i8 - 8) as f32;
                            let q1 = ((byte >> 4) as i8 - 8) as f32;
                            acc += q0 * d * xd[t * inn + b * 32 + j];
                            acc += q1 * d * xd[t * inn + b * 32 + j + 16];
                        }
                    }
                    expect[t * od + o] = acc;
                }
            }
        }

        let mut gb = GraphBuilder::new();
        let x = gb.input("x", [inn, nt, 1, 1], DType::F32);
        let m = gb.matmul(x, &wt, None);
        gb.output(m);
        let g = gb.build();

        let mut sched = BackendScheduler::new();
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input(&g2, "x", &xd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(m).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[matmul q4] Metal vs manual Q4_0xf32 max diff {maxd:.3e}");
        assert!(maxd < 1e-3, "Q4_0 matmul Metal diverges: {maxd:.3e}");
    }

    /// Metal KV store + GQA attention at REAL scale (nh=14, nk=2, hd=64,
    /// nkt=128, nt=30) vs CPU.
    #[test]
    fn metal_attn_kv_real_scale() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let (nh, nk, hd, nkt) = (14usize, 2usize, 64usize, 128usize);
        let nt = 30usize;
        let nqt = nh * hd;
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let q = gb.input("q", [nqt, nt, 1, 1], DType::F32);
        let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
        gb.kvcache_store(0, k, v, pos, 4096);
        let kv = gb.kvcache_load(0, nkt, 4096, nk);
        let o = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            crate::graph::ops::AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk,
                hd,
                hd_kv: nkt / nk,
                nkt,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        gb.output(o);
        let g = gb.build();

        let qd: Vec<f32> = (0..nqt * nt)
            .map(|i| ((i * 13) % 997) as f32 / 400.0 - 1.2)
            .collect();
        let kd: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 17) % 991) as f32 / 400.0 - 1.3)
            .collect();
        let vd: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 19) % 983) as f32 / 400.0 - 1.1)
            .collect();
        let posd: Vec<u32> = (0..nt as u32).collect();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input_i32(&g, "positions", &posd).unwrap();
        ca.fill_input(&g, "q", &qd).unwrap();
        ca.fill_input(&g, "k", &kd).unwrap();
        ca.fill_input(&g, "v", &vd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();

        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input_i32(&g2, "positions", &posd).unwrap();
        alloc.fill_input(&g2, "q", &qd).unwrap();
        alloc.fill_input(&g2, "k", &kd).unwrap();
        alloc.fill_input(&g2, "v", &vd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(o).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[attn kv real] max diff {maxd:.3e}");
        assert!(
            maxd < 1e-3,
            "Metal attention at real scale diverges: {maxd:.3e}"
        );
    }

    /// Metal decode-step attention: nt=1 with 30 already-stored KV rows.
    #[test]
    fn metal_attn_decode_step() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let (nh, nk, hd, nkt) = (14usize, 2usize, 64usize, 128usize);
        let nqt = nh * hd;
        let nkv_prev = 30usize; // KV already filled by the prefill

        // Build a graph that stores 30 tokens (positions 0..29) and 1 decode
        // token (position 30), then attends with nt=1.
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [nkv_prev + 1, 1, 1, 1], DType::I32);
        let q = gb.input("q", [nqt, nkv_prev + 1, 1, 1], DType::F32);
        let k = gb.input("k", [nkt, nkv_prev + 1, 1, 1], DType::F32);
        let v = gb.input("v", [nkt, nkv_prev + 1, 1, 1], DType::F32);
        gb.kvcache_store(0, k, v, pos, 4096);
        let kv = gb.kvcache_load(0, nkt, 4096, nk);
        let o = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            crate::graph::ops::AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk,
                hd,
                hd_kv: nkt / nk,
                nkt,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        gb.output(o);
        let g = gb.build();

        let qd: Vec<f32> = (0..nqt * (nkv_prev + 1))
            .map(|i| ((i * 13) % 997) as f32 / 400.0 - 1.2)
            .collect();
        let kd: Vec<f32> = (0..nkt * (nkv_prev + 1))
            .map(|i| ((i * 17) % 991) as f32 / 400.0 - 1.3)
            .collect();
        let vd: Vec<f32> = (0..nkt * (nkv_prev + 1))
            .map(|i| ((i * 19) % 983) as f32 / 400.0 - 1.1)
            .collect();
        let posd: Vec<u32> = (0..=nkv_prev as u32).collect();

        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input_i32(&g, "positions", &posd).unwrap();
        ca.fill_input(&g, "q", &qd).unwrap();
        ca.fill_input(&g, "k", &kd).unwrap();
        ca.fill_input(&g, "v", &vd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();

        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input_i32(&g2, "positions", &posd).unwrap();
        alloc.fill_input(&g2, "q", &qd).unwrap();
        alloc.fill_input(&g2, "k", &kd).unwrap();
        alloc.fill_input(&g2, "v", &vd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(o).unwrap();
        // compare ONLY the decode row (last token)
        let off = nkv_prev * nqt;
        let mut maxd = 0.0f32;
        for i in off..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[attn decode] decode-row max diff {maxd:.3e}");
        assert!(maxd < 1e-3, "Metal decode attention diverges: {maxd:.3e}");
    }

    /// KV store whose K input is a GPU-computed op (silu) — not host-filled.
    #[test]
    fn metal_store_after_gpu_op() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [2, 1, 1, 1], DType::I32);
        let q = gb.input("q", [8, 2, 1, 1], DType::F32);
        let k = gb.input("k", [8, 2, 1, 1], DType::F32);
        let v = gb.input("v", [8, 2, 1, 1], DType::F32);
        let ks = gb.silu(k); // GPU-computed K input
        gb.kvcache_store(0, ks, v, pos, 16);
        let kv = gb.kvcache_load(0, 8, 16, 2);
        let o = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            crate::graph::ops::AttnMeta {
                layer: 0,
                n_head: 2,
                n_head_kv: 2,
                hd: 4,
                hd_kv: 4,
                nkt: 8,
                scale: 0.5,
            },
        );
        gb.output(o);
        let g = gb.build();
        let qd: Vec<f32> = (0..16)
            .map(|i| ((i * 13) % 29) as f32 / 10.0 - 1.4)
            .collect();
        let kd: Vec<f32> = (0..16)
            .map(|i| ((i * 17) % 31) as f32 / 10.0 - 1.5)
            .collect();
        let vd: Vec<f32> = (0..16)
            .map(|i| ((i * 19) % 37) as f32 / 10.0 - 1.8)
            .collect();
        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input_i32(&g, "positions", &[0, 1]).unwrap();
        ca.fill_input(&g, "q", &qd).unwrap();
        ca.fill_input(&g, "k", &kd).unwrap();
        ca.fill_input(&g, "v", &vd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input_i32(&g2, "positions", &[0, 1]).unwrap();
        alloc.fill_input(&g2, "q", &qd).unwrap();
        alloc.fill_input(&g2, "k", &kd).unwrap();
        alloc.fill_input(&g2, "v", &vd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(o).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[store after gpu op] max diff {maxd:.3e}");
        assert!(maxd < 1e-4, "store-after-gpu-op diverges: {maxd:.3e}");
    }

    /// KV store+attn at REAL dims (nkt=128, n_ctx=32768, nt=30) hand-filled.
    #[test]
    fn metal_store_real_dims() {
        let _g = crate::metal::metal_test_lock();
        crate::metal::MpsState::init();
        let Some(_b) = MetalBackend::new() else {
            eprintln!("MPS unavailable; skipping");
            return;
        };
        let (nh, nk, hd, nkt) = (14usize, 2usize, 64usize, 128usize);
        let nt = 30usize;
        let nqt = nh * hd;
        let mut gb = GraphBuilder::new();
        let pos = gb.input("positions", [nt, 1, 1, 1], DType::I32);
        let q = gb.input("q", [nqt, nt, 1, 1], DType::F32);
        let k = gb.input("k", [nkt, nt, 1, 1], DType::F32);
        let v = gb.input("v", [nkt, nt, 1, 1], DType::F32);
        gb.kvcache_store(0, k, v, pos, 32768);
        let kv = gb.kvcache_load(0, nkt, 32768, nk);
        let o = gb.attn(
            q,
            kv,
            pos,
            crate::graph::ops::AttnMode::Gqa,
            crate::graph::ops::AttnMeta {
                layer: 0,
                n_head: nh,
                n_head_kv: nk,
                hd,
                hd_kv: nkt / nk,
                nkt,
                scale: 1.0 / (hd as f32).sqrt(),
            },
        );
        gb.output(o);
        let g = gb.build();
        let qd: Vec<f32> = (0..nqt * nt)
            .map(|i| ((i * 13) % 997) as f32 / 400.0 - 1.2)
            .collect();
        let kd: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 17) % 991) as f32 / 400.0 - 1.3)
            .collect();
        let vd: Vec<f32> = (0..nkt * nt)
            .map(|i| ((i * 19) % 983) as f32 / 400.0 - 1.1)
            .collect();
        let posd: Vec<u32> = (0..nt as u32).collect();
        let mut sched = BackendScheduler::new();
        let mut ca = GraphAllocator::new();
        ca.alloc_graph(&g).unwrap();
        ca.fill_input_i32(&g, "positions", &posd).unwrap();
        ca.fill_input(&g, "q", &qd).unwrap();
        ca.fill_input(&g, "k", &kd).unwrap();
        ca.fill_input(&g, "v", &vd).unwrap();
        sched.execute(&g, &mut ca).unwrap();
        let expect = ca.get_buffer(&g, o).unwrap().to_vec();
        let mut g2 = g.clone();
        for n in &mut g2.nodes {
            n.backend = Some(Tag::Metal);
        }
        let mut alloc = GraphAllocator::new();
        alloc.enable_metal();
        alloc.alloc_graph(&g2).unwrap();
        alloc.fill_input_i32(&g2, "positions", &posd).unwrap();
        alloc.fill_input(&g2, "q", &qd).unwrap();
        alloc.fill_input(&g2, "k", &kd).unwrap();
        alloc.fill_input(&g2, "v", &vd).unwrap();
        sched.execute(&g2, &mut alloc).unwrap();
        let got = alloc.copy_to_cpu(o).unwrap();
        let mut maxd = 0.0f32;
        for i in 0..got.len() {
            maxd = maxd.max((got[i] - expect[i]).abs());
        }
        eprintln!("[store real dims] max diff {maxd:.3e}");
        assert!(maxd < 1e-3, "store at real dims diverges: {maxd:.3e}");
    }
}
