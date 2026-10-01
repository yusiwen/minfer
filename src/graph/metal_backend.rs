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
use super::{BufRef, CNode, DType};

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

/// F4: the Metal capability matrix, as a free function.
///
/// The registry carries this as [`super::registry::BackendCaps::supports_op`],
/// which cannot take a `&self`; the trait method below forwards to it.
pub fn supports_op(op: &Op, dtype: DType) -> bool {
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
        // D1: Metal's kernels take a buffer and a length, with no element
        // offset, so it can only express an *exact* view (offset 0 == the
        // parent's own buffer, which is what the allocator maps). An offset
        // window would read the wrong bytes, so it is refused here rather
        // than silently mis-computed — the allocator backstops the partial
        // case, which `supports_op` cannot see (the parent's length is not
        // in the op). G5 is where Metal would learn offsets.
        Op::View { offset, .. } => *offset == 0,
        Op::Reshape { .. } | Op::Permute { .. } => true,
        Op::Scale(_) | Op::Softmax { .. } | Op::BatchMatMul => false,
        // Mixed-quant decode QKV epilogue (D3-8 class 2) is CUDA-only; on
        // Metal the graph builder never emits it (qkv_epilogue_ok = false
        // without `--features cuda`), so it is never assigned here.
        Op::QkvBiasRopeStore { .. } => false,
    }
}

/// F4: the fusion-pass capability, as a free function (see [`supports_op`]).
pub fn supports_fused(fused: &FusedOp) -> bool {
    // swiglu_f32 is the only fusion-pass kernel. The bias+rope+store
    // capability is a build-time fused node (FusedQKV/FusedQkvNorm), not a
    // FusionPass target, so it is not advertised here.
    matches!(fused, FusedOp::SwiGLU)
}

/// C8b S5: Metal has no cell-store read path yet (G5), so it refuses **both**
/// explicit window layouts the KV store can hand a node — `attn_span`'s single
/// `[lo, hi)` pair per query and `kv_map`'s `(cell, len)` runs. The trait
/// default is already `false`; this constant is where a reader looks, and
/// `execute_node`'s Attn arm backstops it.
pub const SUPPORTS_ATTN_SPAN: bool = false;

/// C4: Metal addresses f32/f16 KV rows, so it does not read a packed `q8_0`
/// region; [#87] is the work that adds the kernel and flips this.
///
/// [#87]: https://github.com/yusiwen/minfer/issues/87
pub const READS_PACKED_KV: bool = false;

/// F5 ([#58]): registry hook **phase A** of a cross-backend staging copy out of
/// Metal — **declines**: it returns `Ok(false)`, so the allocator's synchronous
/// host round trip handles the pair exactly as it did before F5.
///
/// That is a deliberate, written decision rather than a half-implementation. The
/// async form on Metal is a `MTLBlitCommandEncoder` copy into a staging buffer
/// plus a completion handler or an `MTLEvent`/`MTLSharedEvent` the consumer waits
/// on — a different mechanism from CUDA's — and this ticket was developed on a
/// **Linux** box with no Metal device and no macOS toolchain, so none of it could
/// be compiled, let alone verified bitwise. Shipping un-compilable device code
/// would be the "half-implemented" failure the ticket warns about; declining keeps
/// the pre-F5 behaviour (correct, blocking) and leaves the port as a filed
/// follow-up. The boundary counters therefore report a Metal source's copies as
/// `blocking_host_copies`, which is the honest number on macOS until it lands.
///
/// The CPU→Metal direction is unaffected either way: its device leg is Metal's own
/// `write_host` staging fill.
///
/// [#58]: https://github.com/yusiwen/minfer/issues/58
pub(crate) fn copy_cross(
    _alloc: &mut super::alloc::GraphAllocator,
    _uid: u64,
    _node_id: super::NodeId,
    _dst_backend: super::Backend,
) -> Result<bool, String> {
    Ok(false)
}

/// F5 ([#58]): registry hook **phase B** for a Metal source — a no-op, because
/// [`copy_cross`] declined and the allocator's synchronous path already produced
/// the bytes. See that function for why the Metal port is not in this ticket.
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

/// F4: this backend's registry entry (see `cpu_backend::entry`).
pub fn entry() -> super::registry::BackendEntry {
    use super::registry::{Backend as Handle, BackendCaps, BackendEntry, PRIORITY_METAL};
    BackendEntry {
        handle: Handle::METAL,
        name: "metal",
        priority: PRIORITY_METAL,
        caps: BackendCaps {
            supports_op,
            supports_fused,
            supports_attn_span: SUPPORTS_ATTN_SPAN,
            reads_packed_kv: READS_PACKED_KV,
        },
        pool: |a| a.metal().map(|m| m as &dyn Backend),
        pool_mut: |a| a.metal_mut().map(|m| m as &mut dyn Backend),
        host_read: |a, id| a.metal().and_then(|m| m.read_host(id)).map(|s| s.to_vec()),
        // F5: Metal declines phase A (see `copy_cross` above — no Mac to compile
        // or verify the blit/event port on) and its phase B is therefore a no-op.
        copy_cross,
        await_cross,
        // The MPS device layer holds the process-wide f16 policy (C5 records it
        // so a session written under one width cannot be resumed under another).
        kv_format: |_| {
            if crate::metal::kv_cache_is_f16() {
                super::kvformat::KvFormat::F16
            } else {
                super::kvformat::KvFormat::F32
            }
        },
        enable: |a| a.enable_metal(),
        unavailable: || {
            if metal_available() {
                None
            } else {
                Some("no Metal device, or MPS is unavailable (MINFER_DISABLE_MPS)")
            }
        },
    }
}

/// F4: register the Metal backend (macOS only).
pub fn register(registry: &mut super::registry::Registry) {
    registry.register_entry(entry());
}

impl Backend for MetalBackend {
    fn supports_op(&self, op: &Op, dtype: DType) -> bool {
        supports_op(op, dtype)
    }

    fn supports_attn_span(&self) -> bool {
        SUPPORTS_ATTN_SPAN
    }

    fn supports_fused(&self, fused: &FusedOp) -> bool {
        supports_fused(fused)
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

    fn pool_len(&self) -> usize {
        self.pool.len()
    }

    fn alloc_fresh(&mut self, size: usize) -> usize {
        // never recycled from the free list (see Backend::alloc_fresh)
        self.pool.push(self.state.new_f32_buffer(size));
        self.pool.len() - 1
    }

    fn execute_node(
        &mut self,
        node: &CNode,
        in_bufs: &[BufRef],
        out_buf: BufRef,
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
                if in_bufs[0].id != out_buf.id {
                    self.copy_in(out_buf.id, in_bufs[0].id);
                }
                let n = self.pool[out_buf.id].length() as usize / 4;
                cb.silu_f32(self.buf(out_buf.id), n);
                Ok(())
            }
            Op::Add => {
                cb.add_f32(
                    self.buf(in_bufs[0].id),
                    self.buf(in_bufs[1].id),
                    self.buf(out_buf.id),
                    self.pool[out_buf.id].length() as usize / 4,
                );
                Ok(())
            }
            Op::Mul => {
                cb.mul_f32(
                    self.buf(in_bufs[0].id),
                    self.buf(in_bufs[1].id),
                    self.buf(out_buf.id),
                    self.pool[out_buf.id].length() as usize / 4,
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
                                self.buf(in_bufs[0].id),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf.id),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        } else {
                            cb.rms_norm(
                                self.buf(in_bufs[0].id),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf.id),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        }
                    }
                    None => cb.rms_norm(
                        self.buf(in_bufs[0].id),
                        None,
                        0,
                        self.buf(out_buf.id),
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
                let n = (self.pool[out_buf.id].length() as usize / 4) / d;
                let _ = nh;
                match w {
                    Some((wb, w_off)) => {
                        if crate::metal::rms_norm_256_enabled() {
                            cb.rms_norm_256(
                                self.buf(in_bufs[0].id),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf.id),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        } else {
                            cb.rms_norm(
                                self.buf(in_bufs[0].id),
                                Some(&wb),
                                w_off,
                                self.buf(out_buf.id),
                                d,
                                n,
                                *eps,
                                0,
                                0,
                            );
                        }
                    }
                    None => cb.rms_norm(
                        self.buf(in_bufs[0].id),
                        None,
                        0,
                        self.buf(out_buf.id),
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
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
                    meta.out_dim,
                    meta.in_dim,
                    nt,
                );
                if let Some(bname) = &meta.bias_name {
                    let (bb, b_off) = self
                        .state
                        .weight_buf(bname)
                        .ok_or_else(|| format!("bias '{}' not on GPU", bname))?;
                    cb.add_bias_f32(self.buf(out_buf.id), &bb, b_off, meta.out_dim, nt, 0);
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
                            self.buf(in_bufs[0].id),
                            self.buf(out_buf.id),
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
                            self.buf(in_bufs[0].id),
                            self.buf(in_bufs[1].id),
                            self.buf(out_buf.id),
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
                if in_bufs[0].id != out_buf.id {
                    self.copy_in(out_buf.id, in_bufs[0].id);
                }
                let nt = node.out_shape[1];
                cb.rope_f32(
                    self.buf(out_buf.id),
                    meta.n_head,
                    meta.hd,
                    nt,
                    meta.freq_base,
                    meta.freq_scale,
                    self.buf(in_bufs[1].id),
                    *style as i32,
                    0,
                );
                Ok(())
            }
            Op::SwiGLU => {
                let n = self.pool[out_buf.id].length() as usize / 4;
                cb.swiglu_f32(
                    self.buf(in_bufs[0].id),
                    self.buf(in_bufs[1].id),
                    self.buf(out_buf.id),
                    n,
                );
                Ok(())
            }
            Op::KvcacheStore { layer } => {
                let (k_id, v_id) =
                    kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
                let nkt = node.out_shape[0];
                let nt = (self.pool[in_bufs[0].id].length() as usize / 4) / nkt;
                cb.store_kv(
                    self.buf(in_bufs[0].id),
                    self.buf(k_id),
                    nkt,
                    nt,
                    self.buf(in_bufs[2].id),
                    0,
                );
                cb.store_kv(
                    self.buf(in_bufs[1].id),
                    self.buf(v_id),
                    nkt,
                    nt,
                    self.buf(in_bufs[2].id),
                    0,
                );
                Ok(())
            }
            Op::KvcacheLoad { .. } => Ok(()), // view of the K region
            Op::Attn { explicit_span, .. } => {
                // C8b S5: Metal derives every query's window from `positions` (the
                // pre-E1 form) and has no cell-store read path, so **both** explicit
                // window layouts — `attn_span`'s one `[lo, hi)` pair per query and
                // `kv_map`'s `(cell, len)` runs (C8b S4) — are refused here rather
                // than computed as if they were causal. Assignment already keeps them
                // off this backend (`Backend::supports_attn_span` is false), so this
                // is the backstop that makes a slip loud instead of wrong; G5 is
                // where Metal learns the cell store and this goes away.
                if *explicit_span {
                    return Err(
                        "Metal attention: this node carries an explicit window (attn_span or \
                         kv_map), which Metal does not read yet (G5) — backend assignment \
                         should have kept it on a backend with a cell-store read path"
                            .to_string(),
                    );
                }
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
                let q = self.buf(in_bufs[0].id);
                let o = self.buf(out_buf.id);
                let positions = self.buf(in_bufs[2].id);
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
                // D1: the view is the parent's buffer; nothing to copy.
                if in_bufs[0].id != out_buf.id {
                    return Err(format!(
                        "metal: {} is a view but its output buffer is not its source's (D1 aliasing \
                         missing); refusing to copy silently",
                        node.name
                    ));
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
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
                    od_total,
                    meta.in_dim,
                    nt,
                );
                // 2) swiglu in place: silu(gate rows 0..nf) * up rows nf..2*nf
                //    (llama ggml_swiglu_split); result written back to gate rows
                let n = nt * meta.nf;
                cb.swiglu_f32_off(
                    self.buf(out_buf.id),
                    self.buf(out_buf.id),
                    self.buf(out_buf.id),
                    n,
                    n,
                );
                if std::env::var("MINFER_FFNDEBUG").is_ok() {
                    self.submit_pending();
                    let nb = (self.pool[out_buf.id].length() as usize) / 4;
                    let ob = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(out_buf.id).contents().as_ptr() as *const f32,
                            nb,
                        )
                    };
                    eprintln!(
                        "[ffn-out] FusedFFN out_buf={} len={nb} first4={:?} last4={:?}",
                        out_buf.id,
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
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
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
                    let n = (self.buf(in_bufs[1].id).length() as usize) / 4;
                    let p = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(in_bufs[1].id).contents().as_ptr() as *const u32,
                            n,
                        )
                    };
                    p[0] as i32
                };

                cb.attn_bias_rope_store(
                    self.buf(out_buf.id),
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
                    self.buf(in_bufs[0].id),
                    0,
                    self.buf(out_buf.id),
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
                        self.buf(out_buf.id),
                        Some(&qn_b),
                        qn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_q,
                        meta.eps,
                        off_q,
                        off_q,
                    );
                    cb.rms_norm_256(
                        self.buf(out_buf.id),
                        Some(&kn_b),
                        kn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_k,
                        meta.eps,
                        off_k,
                        off_k,
                    );
                } else {
                    cb.rms_norm(
                        self.buf(out_buf.id),
                        Some(&qn_b),
                        qn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_q,
                        meta.eps,
                        off_q,
                        off_q,
                    );
                    cb.rms_norm(
                        self.buf(out_buf.id),
                        Some(&kn_b),
                        kn_o,
                        self.buf(out_buf.id),
                        meta.hd,
                        n_k,
                        meta.eps,
                        off_k,
                        off_k,
                    );
                }
                // 3) no-bias rope + KV store (q in place, k rope+store, v store)
                let pos = {
                    let n = (self.buf(in_bufs[1].id).length() as usize) / 4;
                    let p = unsafe {
                        std::slice::from_raw_parts(
                            self.buf(in_bufs[1].id).contents().as_ptr() as *const u32,
                            n,
                        )
                    };
                    p[0] as i32
                };
                cb.attn_rope_store(
                    self.buf(out_buf.id),
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
            Op::Scale(_) | Op::Softmax { .. } | Op::BatchMatMul => {
                Err(format!("op {:?} unsupported on Metal (Phase 3)", node.op))
            }
            // CUDA-only mixed-quant decode epilogue; never emitted on the Metal
            // graph (builder sets qkv_epilogue_ok = false without --features
            // cuda), so reaching here is a scheduling invariant violation.
            Op::QkvBiasRopeStore { .. } => {
                Err(format!("op {:?} unsupported on Metal (CUDA-only)", node.op))
            }
        }
    }

    /// C3: the compaction primitive is not ported to Metal, and saying so is the
    /// point — a backend that cannot move cells must fail the compaction rather
    /// than let the allocator renumber runs whose data it did not move (Phase G).
    fn copy_cells(
        &mut self,
        _dst: BufRef,
        _src: BufRef,
        _dst_row: usize,
        _src_row: usize,
        _rows: usize,
        _elems_per_cell: usize,
    ) -> Result<(), String> {
        Err("copy_cells: Metal does not move KV cells yet (Phase G, G5)".to_string())
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
        self.write_host_window(id, 0, data)
    }

    /// E4 S2: a pooled activation buffer is rounded to its size class, so a node
    /// writes its logical window into a buffer that may be longer.
    fn write_host_window(&mut self, id: usize, offset: usize, data: &[f32]) -> Result<(), String> {
        let buf = self.pool.get(id).ok_or_else(|| format!("no buffer {id}"))?;
        let len = (buf.length() as usize) / 4;
        let end = offset.checked_add(data.len()).ok_or_else(|| {
            format!(
                "buffer {id}: offset {offset} + {} elements overflows",
                data.len()
            )
        })?;
        if end > len {
            return Err(format!(
                "buffer {id}: writing {} elements at offset {offset} runs past the {len}-element pool buffer",
                data.len()
            ));
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                data.as_ptr(),
                (buf.contents().as_ptr() as *mut f32).add(offset),
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
mod tests;
