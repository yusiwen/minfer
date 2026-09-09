# 73 · D3-8 — G4 FusedQKV ported to CUDA: both layer classes covered, 14B short-KV breaks through parity (LANDED)

> **Result**: one `attn_bias_rope_store_f32` launch replaces the per-layer 7-launch chain of add_bias×3 + rope×2 + store_kv×2 (math verbatim, bitwise); both layer classes landed — the concat class (`Op::FusedQKV`, 24/48 14B layers) + the mixed-quant class (new `Op::QkvBiasRopeStore`, CUDA-only, 24/48 14B layers) — **decode coverage 48/48**; launches **−310/decode-step (−22.5%**, 22098 → 17130); 14B tg128 23.81 → **24.55 (+3.11%**, isolation +3.15%**), breaking through parity** (24.55 vs llama 24.31, 1.010×), @3254 +1.63% (isolation +2.28%); distance to parity −9.7% → **−7.9%**.
> **Commit**: `3857633` (class 1 concat, 5 files +652) + `a448a4a` (class 2 mixed-quant, 8 files +212). **Date**: 2026-09-08.

## 1. Background — where things stood

While settling rms and bits_to_i32, D3-7's launch census incidentally exposed a structural fact: **the 14B decode qkv chain on CUDA is completely unfused** — per layer rope ×2, KvcacheStore ×2, add_bias ×3, i.e. 7 sub-2µs small kernels × 48 layers ≈ **0.45 ms/step**. And this is not territory where a new design must take risks: the Metal backend has had `Op::FusedQKV` since the graph era (G4) — a concat matmul (loader-registered `blk.{i}.attn_qkv` rows = wq|wk|wv) + one `attn_bias_rope_store` fused pass, with decode logits bit-identical to the unfused path (verified on both 0.5B and 7B). The CUDA backend simply never advertised `Op::FusedQKV`.

So D3-8's (base `/tmp/d3/minfer_pre_d38`, sha1 ccb7df8f, HEAD 81e5d6e) positioning is Stage-3 Tier A: **port a fusion whose semantics were already verified on another backend**, settling the largest item of D3-7's "launch-structure residual (~0.7–1.0 ms)" (the qkv chain, 0.45 ms). The window carried one complication that had to be handled: the machine is ~1.8% **faster** than the D3-7 window (different co-tenant state; anchors 14B tg128 23.81 / @3254 22.04, 7B tg128 50.45 / @1641 48.91) — cross-window comparison would charge that drift to the fusion, so everything ran with the dual basis of **same-window interleaved A/B + same-binary isolation A/B** (post vs post + `MINFER_NO_FUSE_QKV=1`), with isolation as the primary criterion.

A second complication discovered within the session is why this step has two commits: of 14B's 48 layers only **24 layers** have wq|wk|wv in the same quantization type (what the concat class can absorb); the other **24 layers are mixed-quant** (e.g. Q6_K attn_v mixed into Q4_K q/k — D3-7 2b's routing put attn_v onto MMVQ and thereby froze the mixed-quant layer set). The concat matmul requires a single-ttype dispatch, so mixed-quant layers need a second road: three separate matmuls (no bias) + one epilogue. Doing class 1 alone covers 50%, which amounts to leaving half of the 0.45 ms on the table.

## 2. Principle — the GPU mechanism

**What the 7-launch chain's tax is.** The per-layer q/k/v-side work of decode (nt==1) is itself negligible: bias+rope for 5120 q elements, bias+rope/store for 1024 k/v elements each — each kernel executes in 1–2 µs, while each launch's fixed overhead is ~1.3–1.7 µs (the sub-2µs ocean of the D3-1/D3-7 census). 7 kernels × 48 layers = 336 launches/step, of which execution is less than half — the textbook shape of a **launch-bound chain**: merging the launch count converts almost directly into wall-clock savings. The nsys ledger is in §5: bias −144, rope −96, store −96, fused +48 — 7 → 1 per layer.

The per-layer ledger split by class (nt==1): unfused = 3 matmuls + 3 bias + 2 rope + 2 store = **10** launches; class 1 = 1 concat matmul + 1 epilogue = **2** (−8/layer, of which the matmul merge contributes −2); class 2 = 3 matmuls + 1 epilogue = **4** (−6/layer, the three matmuls kept as-is). 48 layers, class 1/2 half each: −8×24 − 6×24 = −336, plus the concat layers' standalone quantize +24 — the same magnitude as the measured −310/step (reconciled item by item against the dispatch table in §5). The reason class 2 does not concat is also in this ledger: repacking mixed-quant weights into one concat plane needs a loader-level repack and extra memory, while the epilogue form touches the weights **zero** — the gain of 6 fewer launches per layer does not require moving 1 byte of weights.

**The fusion's bitwise legitimacy.** The epilogue is not a "reimplementation"; it is a **verbatim transplant**: rope uses `rope_f32`'s neox pairing (j, j+half), the same freq/theta expression, the same cosf/sinf — elementwise bitwise; bias is the same two-operand one-add (`add_bias_f32`'s two operands); the store address (`dst[pos * nkt + j]`) and conversion are verbatim the same as `store_kv_f32`/`store_kv_f16` (`__float2half` is RN, the same rounding as the unfused f16 store's scalar tail). Mathematically this is putting 7 verbatim-identical function bodies into one grid index — so the gates can be bitwise rather than tolerance.

**Why the concat matmul is bitwise.** Class 1 merges 3 matmuls into 1 (14B od 7168 × id 5120, 7B od 4608 × id 3584, Q4_K); the premise that makes this bitwise lives in decode MMVQ's structure: the kernel **serves one row per 256-thread block** and dispatch looks only at (ttype, id, nt) — zero coupling between rows. Concat's row i and the original independent matmul's row i are the same weight bytes × the same activation vector through the same reduction; there is no cross-row reorganization, so concat ⊂ row-wise identity. This argument was explicitly proven by a probe (§4), not left on paper.

**pointer-form: one kernel, two uses.** The q/k/v the kernel receives are **section bases**: the concat class passes the three sections of the concat matmul output (q=base, k=base+nqt, v=base+2·nkt), and the mixed-quant class passes three separate matmul outputs. The data-layout difference is pushed into the caller's pointer arithmetic, and the kernel semantics are singular — this is the key design for covering 100% of layers, and the core reason this is a "port" rather than a "rewrite".

**capture/replay safety.** `pos = positions[0]` is read device-side (nt==1, no host scalar crosses the launch) and replay re-reads the device buffer every step; grid/block are pure shape functions (nqt/2 + nkt/2 + nkt threads, 256 threads per block — 14B is 2560+512+1024 = 4096 threads → 16 blocks, 7B is 2560 threads → 10 blocks) with no nkv/n_past in them — CUDA Graph captures once and replays per frame. The positions i32 conversion is supplied by D3-7's just-landed memo (exactly one conversion per step, stable pointer).

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **One epilogue kernel, two pointer forms** (rather than writing a separate kernel for mixed-quant): there is only one math surface, so the gate needs proving only once; mixed-quant's difference lives entirely in "where the q/k/v pointers point", which belongs to the caller.
- **Class 2 uses a new Op instead of being crammed into `Op::FusedQKV`**: `FusedQKV`'s semantics carry the concat weight name (`FusedQkvMeta.qkv_weight`), which mixed-quant does not have — its inputs are three separate matmul outputs. The new `Op::QkvBiasRopeStore` + `QkvBiasRopeStoreMeta` (= FusedQkvMeta minus the concat weight) keeps dispatch, the scheduler's `kv_pair(layer)` resolution, and the CPU backend's loud-Err set each clean.
- **q's in-place alias rule unchanged**: the builder wires attention to the epilogue node (q's matmul buffer has exactly one consumer), the allocator keeps the §5 alias (the Silu/RoPE family extended + `ensure_kv`), and the backend adds one more layer of insurance — "D2D copy if the alias does not hold" (the RoPE arm's pattern).
- **Class 2 is CUDA-only**: the emission condition is `cfg(feature="cuda")` + CudaState present; Metal keeps the unfused chain for mixed-quant layers (`supports_fused` returns false for the new op) — **bitwise-neutral vs pre-D3-8**. A porting task does not reopen the Metal battlefield.
- **The gate enters the reuse identity**: `nt==1 && gpu && fuse_qkv` (class 2 additionally requires CUDA present) is part of the graph params — `MINFER_NO_FUSE_QKV=1` forces a rebuild, making A/B reliable.

### 3.2 Key code

**Excerpt A · the fused kernel (current tree `src/cuda_kernels.cu`, landed in `3857633`)** — the grid is one linear index: q's rope pairs (nqt/2) + k's rope pairs (nkt/2) + v elements (nkt), one work unit per thread:

```cuda
// src/cuda_kernels.cu (current tree; header comment excerpt: D3-8 CUDA port of Metal's
// kernel_attn_bias_rope_store ... The rope math is VERBATIM rope_f32 (NEOX
// pairing (j, j+half), same freq/theta expression and cosf/sinf —
// bitwise-identical per-element results) ... positions[0] is read
// device-side (nt==1; no host scalar crosses the launch — CUDA Graph
// capture/replay safe). Thread mapping (Metal's): one thread per
// (head, d < hd/2) rope pair for q and k, one thread per v element →
// grid = nqt/2 + nkt/2 + nkt.
__global__ void attn_bias_rope_store_f32(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ bias_q, const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ kv_k, float* __restrict__ kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions, int kv_is_f16
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    const int pos = positions[0];              // device-side: capture-safe

    if (u < qpairs) {
        // q section: bias + rope in place (attention reads q at offset 0)
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int j  = head * hd + d;          // ← the caller already picked the section pointer: concat or separate buffers
        const int j2 = j + half_dim;
        float x0 = q[j]  + bias_q[j];
        float x1 = q[j2] + bias_q[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        q[j]  = x0 * cs - x1 * sn;             // verbatim rope_f32 neox pairing
        q[j2] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        // k section: bias + rope in place + store into the K region
        /* after the same-form rope:
           k[j] = r0; k[j2] = r1;
           ((__half*)kv_k)[(size_t)pos * nkt + j]  = __float2half(r0);  // RN
           ... f32 branch: kv_k[(size_t)pos * nkt + j] = r0; */
    } else {
        // v section: bias + store into the V region
        const int j = u - qpairs - kpairs;
        const float val = v[j] + bias_v[j];
        v[j] = val;
        /* ((__half*)kv_v)[(size_t)pos * nkt + j] = __float2half(val); */
    }
}
```

**Excerpt B · the launcher and the Rust entry (current tree)** — the launcher comment states outright that this is Metal's dispatch_1d shape; the Rust-side `CudaState::attn_bias_rope_store` only forwards pointers:

```cuda
// src/cuda_kernels.cu (current tree)
// D3-8: fused decode QKV epilogue launcher — 256-thread blocks over the
// flat (nqt/2 + nkt/2 + nkt) thread mapping (Metal's dispatch_1d shape).
void launch_attn_bias_rope_store(
    float* q, float* k, float* v,
    const void* bias_q, const void* bias_k, const void* bias_v,
    void* kv_k, void* kv_v,
    int nqt, int nkt, int hd, float freq_base, float freq_scale,
    const int* positions, int kv_is_f16, cudaStream_t stream
) {
    const int total = nqt / 2 + nkt / 2 + nkt;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    attn_bias_rope_store_f32<<<grid, block, 0, stream>>>(q, k, v, ...);
}
```

```rust
// src/cuda.rs (current tree, excerpted comments)
/// D3-8: fused decode QKV epilogue (G4 CUDA port of Metal's
/// `attn_bias_rope_store`). `q`/`k`/`v` are POINTER-FORM section bases:
/// the concat class passes sections of the concat matmul output [q|k|v]
/// (nt==1), the mixed-quant class passes the three separate matmul
/// outputs. Biases added per section, q/k roped in place (math verbatim
/// `rope_f32`), k/v stored into the persistent regions at the same
/// addresses as `store_kv_f32`/`store_kv_f16` (f32 or f16 per `kv_is_f16`).
pub fn attn_bias_rope_store(&self, q, k, v, bias_q, bias_k, bias_v,
                            kv_k, kv_v, nqt, nkt, hd, freq_base,
                            freq_scale, positions, kv_is_f16) {
    unsafe { launch_attn_bias_rope_store(/* pointer forwarding */); }
}
```

**Excerpt C · the builder's two-class dispatch (current tree `src/models/qwen2/graph.rs`)** — the `fuse_qkv` gate + the class 1/class 2/else three arms:

```rust
// src/models/qwen2/graph.rs (current tree)
let fuse_qkv = nt == 1
    && params.cparams.gpu
    && params.cparams.fuse_qkv
    && l.bq.is_some() && l.bk.is_some() && l.bv.is_some();
// class 2 is CUDA-only: on macOS (feature off) the mixed-quant
// layers keep the unfused chain, bitwise-neutral vs pre-D3-8.
#[cfg(feature = "cuda")]
let qkv_epilogue_ok = crate::cuda::CudaState::get().is_some();
#[cfg(not(feature = "cuda"))]
let qkv_epilogue_ok = false;
let (q, kv) = if fuse_qkv && Self::qkv_concat_available(&l.wq, &l.wk, &l.wv) {
    // class 1: one concat matmul (blk.{il}.attn_qkv) + the fused epilogue;
    // q sits at concat offset 0 (nt==1, no stride issue), K/V already
    // written by the fused store
    let qkv = b.fused_qkv(normed, inp_pos, il, FusedQkvMeta { /* .. */ });
    let kv = b.kvcache_load(il, nkt, n_ctx, nk);
    (qkv, kv)
} else if fuse_qkv && qkv_epilogue_ok {
    // D3-8 class 2 (mixed quant types): separate matmuls without
    // bias, then one epilogue pass (bias×3 + rope×2 + store×2 → 1).
    // Attention is wired to the epilogue node so q's matmul buffer
    // has exactly one consumer (in-place alias rule, §5).
    let q = b.matmul(normed, l.wq.as_ref().unwrap(), None);  // no bias
    let k = b.matmul(normed, l.wk.as_ref().unwrap(), None);
    let v = b.matmul(normed, l.wv.as_ref().unwrap(), None);
    let q = b.qkv_bias_rope_store(q, k, v, inp_pos, il,
        QkvBiasRopeStoreMeta { bias_q: .., bias_k: .., bias_v: ..,
            nqt: nh * hd, nkt, hd, freq_base: hp.rope_freq_base,
            freq_scale: hp.rope_freq_scale, rope_style: hp.rope_style,
            kv_elems: nkt * n_ctx });
    let kv = b.kvcache_load(il, nkt, n_ctx, nk);
    (q, kv)
} else { /* unfused chain: 3 matmuls (with bias) + 3 bias + 2 rope + 2 store */ };
```

(The `blk.{i}.attn_qkv` concat weight is registered on the backend registry by the loader — `register_weight("blk.{i}.attn_qkv", ...)`; the 28 lines `3857633` added to the loader are exactly this registration chain.)

**Excerpt D · the backend dispatch arm (current tree `src/graph/cuda_backend.rs`)** — guards + alias insurance + a single launch:

```rust
// src/graph/cuda_backend.rs (current tree; arm header comment excerpt: ... the q matmul buffer
// has exactly one consumer); fall back to a D2D copy if it ever doesn't
// (RoPE arm pattern). Replaces the 7-launch small-kernel tail
// (add_bias×3 + rope×2 + store×2) → 1 launch.
Op::QkvBiasRopeStore { layer } => {
    let meta = /* NodeMeta::QkvBiasRopeStore(m); loud-Err when missing */;
    let nt = node.out_shape[1];
    if nt != 1 {
        return Err(format!("cuda: {}: QkvBiasRopeStore is decode (nt==1) only, got nt={nt}", node.name));
    }
    if !matches!(meta.rope_style, RopeStyle::NonInterleaved) {
        return Err(format!("cuda: {}: qkv epilogue rope style {:#?} not supported \
                            (rope_f32 is neox/non-interleaved only)", node.name, meta.rope_style));
    }
    if meta.hd == 0 || meta.hd % 2 != 0 { /* loud-Err: head dim must be even */ }
    let (k_id, v_id) = kv_pair.ok_or_else(|| format!("KV regions for layer {layer} not allocated"))?;
    // q: in-place (out aliases the q input; copy when it doesn't)
    if in_bufs[0] != out_buf {
        self.copy_d2d(in_bufs[0], out_buf)?;
    }
    /* bias pointer resolution (unregistered → loud-Err); pos = self.positions_i32(in_bufs[3])?
       — supplied by the D3-7 memo, one conversion per step */
    self.state.attn_bias_rope_store(
        self.ptr_of(out_buf)?, self.ptr_of(in_bufs[1])?, self.ptr_of(in_bufs[2])?,
        bq, bk, bv, self.ptr_of(k_id)?, self.ptr_of(v_id)?,
        meta.nqt, meta.nkt, meta.hd, meta.freq_base, meta.freq_scale,
        pos, self.kv_f16,
    );
```

**Excerpt E · the allocator's alias extension (current tree `src/graph/alloc.rs`)** — the epilogue joins the Silu/RoPE family, and `ensure_kv` runs for that layer:

```rust
// src/graph/alloc.rs (current tree)
Op::Silu | Op::RoPE { .. } | Op::QkvBiasRopeStore { .. } => {
    // D3-8: the mixed-quant QKV epilogue also needs the layer's
    // persistent KV regions (it stores k/v like FusedQKV).
    if let Op::QkvBiasRopeStore { layer } = &node.op {
        let kv_elems = match &node.meta {
            NodeMeta::QkvBiasRopeStore(m) => m.kv_elems,
            _ => node.n_elements(),
        };
        self.ensure_kv(*layer, backend, kv_elems);
    }
    // In-place elementwise transforms: alias the input buffer ...
    if last_use[id] > i { /* the output aliases src[0]'s buffer */ }
```

**Excerpt F · the probe test's two pointer forms (current tree `src/cuda.rs`, `cuda_fused_qkv_epilogue_bitwise`)** — the front line of the bitwise gate: on the same concat-matmul output, run the fused form (form 1: section bases = base + offsets) and the separate-buffer form (form 2), and compare byte-for-byte against the unfused 7-launch chain over the q/k/v sections and the KV rows; all combinations of the 14B/7B geometries × f32/f16 KV:

```rust
// src/cuda.rs (current tree, test excerpt)
/// D3-8 probe A: fused `attn_bias_rope_store` vs the unfused chain
/// (add_bias×3 + rope×2 + store_kv×2) on the SAME concat-matmul output —
/// bitwise on the q/k/v sections AND both KV regions, f32 + f16 KV,
/// 14B + 7B shapes, BOTH pointer forms (concat sections AND three
/// separate buffers — the mixed-quant class-2 wiring).
// ---- FUSED form 1: concat buffer, section pointers ----
let base = d_fused as *mut u8;
let (q1, k1, v1) = (
    d_fused,
    unsafe { base.add(nqt * 4) } as *mut std::ffi::c_void,         // k section
    unsafe { base.add((nqt + nkt) * 4) } as *mut std::ffi::c_void, // v section
);
st.attn_bias_rope_store(q1, k1, v1, dbq, dbk, dbv, dk_f, dv_f, ...);
// ---- FUSED form 2: three separate buffers (class-2 shape) ----
let d_q2 = dev_alloc(nqt * 4);
let d_k2 = dev_alloc(nkt * 4);
let d_v2 = dev_alloc(nkt * 4);
/* same kernel, same biases, compared byte-for-byte against the unfused chain */
```

Reading the two commits' change surfaces together: `3857633` (class 1) = `cuda.rs` +389 (wrapper+probe), `cuda_kernels.cu` +113 (kernel+launcher), `cuda_backend.rs` +103 (the FusedQKV arm), `qwen2/graph.rs` +21 (wiring), `loader.rs` +28 (concat weight registration); `a448a4a` (class 2) completes the graph infrastructure: `ops.rs` (Op+Meta), `builder.rs`, `alloc.rs`, `scheduler.rs` (`kv_pair` resolution), `cpu_backend.rs` (the loud-Err set), `json.rs` (dump-graph/trace naming), `cuda_backend.rs`, `qwen2/graph.rs`. The kernel and probe were in place with the class 1 commit, and class 2 reuses the same kernel — that is the landing order of "one math surface, two entry points".

### 3.3 Pitfalls

- **The launch ledger is not the ideal −384.** Beyond the paper ledger of 7 → 1 per layer (336 − 48), two more items must be counted: the 24 concat layers' 3 matmuls → 1 (−48), and the concat matmul's shared-A standalone quantize **+24** — D3-5's MmqCache record window only covers the pre-existing "producer → following matmul group" consumption pairs, and the concat created a new consumption combination the memo window did not catch. The measured total is **−310/decode-step** (16-step trace, −4968 total).
- **The gate script nearly mistook tok/s for divergence.** The only pre/post difference in the greedy battery's raw output was the perf banner's tok/s number — every run's "first divergence at byte ~400" was `1789.6` vs `1788.7 tok/s`. The rule was finalized: **strip timing lines before comparing**.
- **The `node{N}_*` dumps are an instrument limitation, not a gate.** These informational dumps read a whole recycled pool slot, and their content at dump time depends on binary layout: node3 diffs **within the same** binary; node5/8 diff pre-vs-post even under `MINFER_NO_FUSE_QKV=1` (where graph and loader behavior are identical). The real structural gates are the prefill DOT graph byte-identical (topology unchanged) + logits/KV fully byte-identical.
- **The f16 store's rounding mode must be verbatim.** `__float2half` is RN — exactly the same rounding as the unfused f16 store's scalar tail, which is why f16 KV's bitwise holds. Any refactor that "conveniently rewrites the conversion" will crash on this gate.
- **The rope style guard must be loud.** The kernel is a verbatim transplant of the neox (non-interleaved) pairing; silently running another rope style is far more dangerous than a loud-Err — the arm returns an explicit `Err`. Guards of the same kind: nt==1 (the epilogue is decode-only) and hd even (the rope pairing requires it).
- **The mixed-quant layer set is not static.** D3-7 2b routed 14B attn_v into MMVQ, freezing the "11 attn_v layers are Q6_K" layer set; this step's `qkv_concat_available` decides layer by layer at build time rather than hard-coding counts — the gate is the per-layer reconciliation of the census's (D3-7) 24/48 against the measured trace, not a hard-coded expectation.

## 4. Verification

- **(i) Probe tests (defend the kernel math and dispatch equivalence)**: double green — the fused epilogue vs the unfused 7-launch chain bitwise on the q/k/v sections **and** the KV rows (both pointer forms, f32+f16 KV, 14B+7B shapes); concat matmul vs the three separate matmuls bitwise (14B od 7168 × id 5120, 7B od 4608 × id 3584, Q4_K) — the argument "decode MMVQ dispatches row by row ⇒ concat ⊂ row-wise identity" explicitly proven.
- **(ii) Dump gate (defends against end-to-end numeric drift)**: `MINFER_GRAPH_DUMP`, 14B prompt_short + 7B prompt_1k7, −n 4: `logits_{prefill,decode}` + **all** `kv{layer}_*.f32` byte-identical pre-vs-post — **98/98 @14B, 58/58 @7B** (the informational node dumps see 3.3; not used as gates).
- **(iii) Greedy battery (defends against sampler/per-token behavior changes)**: −n 256, 5/5 seeds × both models byte-identical; rp=1.0 identical; the `MINFER_NO_FUSE_QKV=1` control identical; the temp-0.8 sampled control identical — the only pre/post diff in the raw output is the perf banner's tok/s (see 3.3).
- **(iv) Suite 172/0/3** (D3-7's 170 + 2 new probes); prefill undisturbed (pp3254 1833 t/s pre-vs-post — the fusion is gated at build time on nt==1).
- **Coverage audit**: the decode-layer dispatch of `MINFER_GRAPH_TRACE=1` reconciles layer by layer with D3-7's GGUF census — 48/48 @14B (24 concat + 24 epilogue), 28/28 @7B (14 + 14).

## 5. Results

**Mechanism (nsys, 14B @3254, NO_CUDA_GRAPH, the same 16-decode-step trace)**: total kernel launches 22098 → 17130 (**−22.5%**); per decode step ≈ **−310**:

| dispatch item | Δ/step | note |
|---|---|---|
| `add_bias` | −144 | 3 × 48 layers → folded into the epilogue |
| `rope` | −96 | 2 × 48 layers → folded into the epilogue |
| `store_kv` | −96 | 2 × 48 layers → folded into the epilogue |
| `attn_bias_rope_store` | +48 | 1 fused pass per layer |
| q4_K MMVQ | −48 | 24 concat layers: 3 matmuls → 1 |
| standalone quantize | +24 | concat shared-A, a new consumption pair the D3-5 memo window does not cover |

Net wall clock ≈ **−0.9 ms/step**, exceeding D3-7's census projection of 0.45 ms/step for the qkv chain — the difference comes from launch gaps (336 → 48 emission points, each ~1.3–1.7 µs of fixed overhead no longer paid one by one) and class 1's matmul merge (−48); neither of these shows up in the census's "kernel time" basis — only the launch count reconciles.

**Wall clock (interleaved 3-pair medians, every pair clean-separated; isolation = post vs post+NO_FUSE_QKV, same binary)**:

| config | pre_d38 | post | Δ | isolation |
|---|---|---|---|---|
| 14B tg128 | 23.81 | **24.55** | **+3.11%** | +3.15% (24.53 vs 23.78) |
| 14B @3254 | 22.04 | **22.40** | **+1.63%** | +2.28% (22.46 vs 21.96) |
| 7B tg128 | 50.45 | **50.98** | +1.05% | +1.03% (50.92 vs 50.40) |
| 7B @1641 | 48.91 | **49.51** | +1.23% | +1.00% (49.44 vs 48.95) |

Guards all hold with margin (14B tg128 ≥ 22.7, @3254 ≥ 21.5; 7B ≥ 49.0 / ≥ 47.9). @3254's +2.28% (isolation basis) exceeds the +0.8% all-fusion landing bar — class 2's epilogue (the "tail lever") landed in the same commit as class 1, saving a separate tail measurement window.

**Why isolation is the primary criterion.** This window's machine is ~1.8% faster than the D3-7 window (co-tenant state); the pre_d38 anchor and D3-7's post anchor have no comparability whatsoever. Same-binary post vs post+`MINFER_NO_FUSE_QKV=1` differs by only the fusion switch itself under **the same machine condition**, so the drift term is eliminated entirely — which is why both Δ columns in the table must pass the isolation test to count (all four pass: +3.15/+2.28/+1.03/+1.00%).

**Distance to parity (post-D3-8, 14B @3254)**: 22.40 t/s = 44.64 ms/step vs llama 41.12 ms → **−3.52 ms (−7.9%)** (post-D3-7 was −4.44 ms / −9.7%). **14B tg128 breaks through parity this window: 24.55 vs llama 24.31 (1.010×)**; 7B stays ahead (tg128 1.032×, @1641 1.002×). The remaining 14B @3254 list: matmul aggregation ~2.9 ms (the decode-GEMM program), exposed-latency attention ≤0.63 ms (mechanism in doubt), output head 0.44 ms, ffn_down-q6K 0.49 ms, launch-structure residual (the add/add/swiglu tail, dud split ~0.07 ms) — **the qkv-chain item closes here**.

## 6. Lessons

1. **Porting an already-verified fusion is an order of magnitude cheaper than writing a new one.** Metal G4's semantics (verbatim math + bitwise gates) came across as-is, so the risk of the 652-line insertion concentrated on "wiring" rather than "math" — the probe proved concat ⊂ row-wise identity first, and only then did the graph infrastructure (Op/Meta/alloc/scheduler) dare to spread out in one go.
2. **pointer-form decouples "where the data is" from the kernel semantics.** Section bases let the concat and separate-buffer shapes share one kernel — mixed-quant models do not have to give up per-layer reality for the fusion, and coverage went 50% → 100%.
3. **The gate script must defend against itself.** The only "divergence" in the raw output was the perf banner's tok/s — strip the timing lines before comparing, or every gate run chases ghosts.
4. **Informational dumps do not enter gates.** Diagnostic output reading recycled pool slots is inherently binary-layout dependent; gates only trust semantically stable artifacts like logits/KV/DOT, and diffs of diagnostic output are calibrated with "same-binary self-diff" before being interpreted.

---
← 72 · [Index](./README.md) · 74 →
