# 62 · r59 — q4_K W_dsc plane + riders (LANDED, Δ corrected by r59b)

> **Result**: clean basis **+11.1%** (3232.0 → 3590.8 tok/s, finalized by r59b);
> the +26.2% recorded on the spot in a co-tenant window was voided because the
> baseline binary was contaminated (see doc 63). Mechanism level: q4_K bt
> kernel busy 762.59 → 526.82 ms = **−30.9%**, regs 124 → 105, at the cost of
> **+1456 MB** device memory (the W_dsc plane).
> **Commit**: `36a481f` (code, + `15c04ba` docs; the `feb37de` recorded in the
> original text is the unreachable pre-amend copy). **Date**: 2026-09-06
> (Session F, same day as r58).

## 1. Background — where things stood

r58's teardown attribution of the q4_K bt kernel left behind an
evidence-ranked Phase-2 spec, first place going to the **q4_K W_dsc plane** —
"move the scaffolding r56 validated on q6_K onto the other 63% of busy". r58
also proved the converse: pipelining (double buffering/cp.async) is negative
on q4_K because its staging is already a pure copy; but the same attribution
noted that **one genuine ALU residual remains in the staging phase** — the
rank-1 scale decode for every (chunk, od-row) pair. Eliminating residual ALU
(rather than hiding it) is fully compatible with r58's lesson.

On the measurement window: this session ran on a machine carrying a 46 GB
sglang co-tenant (r59b later proved an idle co-tenant produces no tax at all
— the on-the-spot attribution was wrong). This doc explains the mechanism
first; numbers use the clean basis corrected by r59b, with on-the-spot
recorded values labeled as such.

## 2. Principle — the GPU mechanism: turning branchy decoding into a 16 B stream

### 2.1 What the eliminated cost looks like

The q4_K bt kernel's per-k-tile staging must compute, for the B side, the two
coefficients of the rank-1 rescale `(d·sc, −dmin·m)`. The original path
executed, for **every (chunk, od-row) pair**:

```
get_scale_min_k4(c & 7, blk + 4, &sc, &m)   // branchy packed 6-bit scale decode
d    = h2f(*(uint16_t*)blk)                 // f16 → f32 ×2
dmin = h2f(*(uint16_t*)(blk + 2))
dv = d * (float)sc;  mv = -(dmin * (float)m)  // two int→f32 conversion multiplies
```

Scale arithmetic: one staging pass per block tile handles `MMQ_NBJ × KDR` =
128 × 8 = 1024 (chunk, row) pairs, each walking the **branchy**
`get_scale_min_k4` (two different bit-extraction paths, `cc < 4` vs `cc ≥ 4`)
— differing chunk distributions across a warp cause **divergence**; times
2 h2f + 2 multiplies. This is pure staging-phase ALU — exactly the single
exception to r58's "staging is already a pure copy" conclusion.

### 2.2 Planarization: one-time pre-decode at registration

The W_dsc plane moves this decode to **load time**: for each q4_K tensor,
build a chunk-major f32-pair plane:

```
plane bytes = nchunk × od × 8 B = (id/32) × od × 8 = od·id/4 B per tensor
out[(c·od + j)·8 .. +8] = float2(d·sc[c&7], −dmin·m[c&7])
```

chunk-major (`c` outermost) is the key layout choice: the tile the kernel
reads at `kt` is exactly the contiguous rectangle rows `j0..j0+128` × chunks
`c0..c0+7` — stored row-major that is 128 scattered 8 B reads; stored
chunk-major it is **one regular 16 B `gemm_cp16` per row pair** (two float2
are exactly 16 B aligned), flowing straight into the cp.async channel
introduced by r53.

Memory account: the q4_K weights of 7B q4_k_m are actually **5.8 GB** (the
r58 spec's ~1.07 GB estimate used the wrong byte mass of 4.29 GB),
od·id/4 ≈ 5.8 GB / 4 → measured **+1456 MB**. This cost belongs, alongside
r56's q6_K W_dsc (+363 MB) and r53's W_exp (+1.52 GB), to the "plane for
ALU" family; r54's opt-out mode proved such trades are acceptable on a
10-GB-class device — but an exit door must be provided.

### 2.3 Why it is bit-identical

The mma-side rescale demands the plane's coefficients match the in-kernel
computation exactly; three guarantees:

1. **exact f16→f32** (`half::f16::to_f32` ≡ device `__half2float`);
2. **exact u8→f32**;
3. **exactly one IEEE f32 multiply**, one exact negation, and **FMA
   contraction forbidden on both sides** — the moment the host compiler fuses
   `d*s` with a later add into an FMA, bit-identity breaks.

The u8 6-bit scale (q4_K's packed sc/m) differs from q6_K's i8 dsc: q6_K's
dsc is an f16 pair, while q4_K's scale is 6-bit unsigned integers + f16
d/dmin, so the plane stores the **multiplied f32 products** rather than the
raw scales — which is also why it needs only 8 B per pair.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Kernel templated `<KDR, DSC>`**: DSC=true takes the plane stream,
  DSC=false keeps the original scalar decode — two instances of the same
  kernel, degrading per launch when the plane is missing, not an either/or
  at compile time.
- **Geometry-encoded sibling names** `{name}__q4dsc{od}x{id}`: the W_exp
  pattern — the same logical weight name does not collide across geometries,
  and the map keyed by the **original weight's device pointer** hits in O(1).
- **Registration gate mirrors the dispatch gate**: the plane is consumed only
  by the NB-BT kernel, so the registration condition = the kernel's dispatch
  condition (RAW_NB + A_TRANSPOSE) + geometry gates (`id % 256 == 0`, same as
  the kernel's launch gate; `od % 2 == 0` guarantees a row pair's cp.async is
  either fully valid or fully out-of-bounds zero-padded).
- **Failure must be loud**: an alloc/upload failure leaves the map empty →
  the kernel takes DSC=false with a once-per-process eprintln; liveness
  labels distinguish `DSC=f32-plane` / `DSC=in-kernel(dsc=off)` /
  **`DSC=in-kernel(fallback!)`** (r53's lesson: a fast path that degrades
  correctly must carry a visible label — parity cannot see the dispatch
  path).

### 3.2 Key code

Registration side (expand at load + upload + build the map; current-tree
`src/cuda.rs`):

```rust
// src/cuda.rs — expand_q4k_dsc (r59): one-time pre-decode at registration
pub fn expand_q4k_dsc(raw: &[u8], od: usize, id: usize) -> Vec<u8> {
    const Q4KB: usize = 144;
    let nsb = id / 256;
    let nchunk = id / 32;
    let row_len = nsb * Q4KB;
    let mut out = vec![0u8; nchunk * od * 8];      // od·id/4 B, chunk-major
    for j in 0..od {
        let prow = &raw[j * row_len..(j + 1) * row_len];
        for sb in 0..nsb {
            let blk = &prow[sb * Q4KB..sb * Q4KB + Q4KB];
            let d = half::f16::from_bits(u16::from_le_bytes([blk[0], blk[1]])).to_f32();
            let dmin = half::f16::from_bits(u16::from_le_bytes([blk[2], blk[3]])).to_f32();
            let sc = &blk[4..16]; // 12 packed 6-bit scales+mins
            for cc in 0..8usize {
                // host mirror of the device get_scale_min_k4 (cuda_kernels.cu)
                let (s, m) = if cc < 4 {
                    (sc[cc] & 63, sc[cc + 4] & 63)
                } else {
                    ((sc[cc + 4] & 0xF) | ((sc[cc - 4] >> 6) << 4),
                     (sc[cc + 4] >> 4) | ((sc[cc] >> 6) << 4))
                };
                let idx = ((sb * 8 + cc) * od + j) * 8;
                out[idx..idx + 4].copy_from_slice(&(d * (s as f32)).to_bits().to_le_bytes());
                out[idx + 4..idx + 8]
                    .copy_from_slice(&(-(dmin * (m as f32))).to_bits().to_le_bytes());
            }
        }
    }
    out
}
```

Dispatch side (each matmul fetches the plane pointer from the map; a miss
yields `null` → the DSC=false instance):

```rust
// src/cuda.rs — q4_K RAW dispatch arm (r59; the default-on shape after r60)
if type_id == 5 && Self::mmq_gate_on("MINFER_MMQ_RAW") && (id / 32) % 8 == 0 {
    ...
    if nb && at && kd == 8 {
        let (qa8g, sdag) = self.mmq_quantize_transposed(...);   // r34/r49 prepass
        // r59: the q4_K W_dsc f32-pair plane (null on miss ->
        // the DSC=false in-kernel scalar decode instantiation).
        let w_dsc = self.q4k_dsc.lock().unwrap()
            .get(&(wptr as usize)).map(|cp| cp.0)
            .unwrap_or(std::ptr::null_mut());
        nb_ok = qa8g != 0 && sdag != 0
            && launch_mmq_raw_nb_bt_nt(type_id, wptr as *const u8,
                w_dsc as *const u8, qa8g as *const u8, sdag as *const u8,
                out as *mut f32, nt as i32, od as i32, id as i32,
                nchunk, stream, kd) == 1;
```

The kernel-side `DSC=true` branch was already quoted in doc 61: one
`gemm_cp16` per (kdd, mm) streams the 16 B at `W_dsc + ((c0d+kdd)·od + j)·8`
(two float2 — the adjacent two rows of a row pair's four coefficients) into
smem, and `gemm_cp_commit()` commits them in the same group as A/B — no
branch or h2f remains in the staging phase.

Two implementation details worth calling out:

- **The 16 B payload = one pair of od rows**. The loop variable `mm` of
  `nc2 = MMQ_NBJ/2` corresponds to the row pair `(j0+2·mm, j0+2·mm+1)` — one
  16 B cp.async carries, for the same chunk, the `float2(d·sc, −dmin·m)` of
  two adjacent rows. This is why the registration gate demands `od % 2 == 0`:
  a row pair is either entirely valid or entirely out-of-bounds (cp.async's
  src-size qualifier zero-fills the whole pair); there is no half-valid third
  state.
- **The compute side cannot tell the two instances apart**. DSC=true and
  false write the same slots of the same smem array `sds`
  (`sds[kd·NBJ + r]`, one float2 per row), and the mma-side rescale code is
  unchanged word for word — bit-identity therefore holds structurally, not
  because the two sides happen to compute equal values.

### 3.3 riders: moving one-time costs out of the measurement window

The same commit carried in three host-side riders left over from r57
(`prewarm_prefill()`):

1. **Kernel module preload** — `minfer_prewarm_kernels()` sweeps
   `cudaFuncGetAttributes` over the MMQ/FA/fused launch set, forcing the
   fatbin to load outside the measurement window (r58 CUPTI: ~3 ms host stall
   on each side of the first mode-2 swiglu / first bt matmul);
2. **Pinned readback pre-grow** — a 4 MB `cudaHostAlloc` allocated up front
   (otherwise it is the "0.78 ms tail malloc" at the first logits readback);
3. **MmqCache scratch pre-grow** — `buf_q8_prefill`/`buf_qa8_t`/`buf_sda_t`
   sized for a nominal 4096-token prefill at the largest registered nchunk,
   so the first prefill's get_or_grow hits directly (otherwise a surprise
   ~150 MB cudaMalloc lands mid-window). MMQ gating: with MMQ off, the plane
   is dead weight.

## 4. Verification

- **Byte-exactness 0 mismatch**: `expand_q4k_dsc` (the host mirror, including
  the handwritten `get_scale_min_k4` mirror) compared byte-for-byte against
  the device side (the `cuda_q4k_dsc_dense_byte_exact` test, a matrix of
  od×id shapes) — defends against "the plane decoded wrongly but the
  deviation falls inside scale tolerance".
- **parity ×3 + the `MINFER_MMQ_Q4K_DSC=0` contrast**: flip the env var on
  the same binary; both the DSC=true and DSC=false paths must pass parity —
  defends against the one-sided trap of "plane path wrong, contrast path
  green".
- **greedy byte-identity, both paths**: likewise, each path compared against
  the baseline stream.
- **liveness 166×/0**: all 166 launches hit the plane, zero fallbacks —
  defends against "registered the fast path but silently degraded at
  dispatch" (the institutionalized r53 lesson).
- **suite 169/0/3**: defends against cross-shape regressions.

## 5. Results

**On-the-spot record (co-tenant window, basis later corrected by r59b)**:
interleaved 5× ×2 series, baseline 2836.3/2843.2 → new 3574.7/3588.8 =
**+26.1/+26.2%**; at the time the baseline reading below 3219.6 was
attributed to a −12% "co-tenant tax". **r59b proved the baseline binary
itself was the r58 delta build (a −12.5% defect)**; the true clean delta is
**+11.1%** (3232.0 → 3590.8), and the co-tenant-tax story is voided — doc 63
has the correction process and the protocol rule.

Mechanism-level evidence (unaffected by baseline contamination; all
same-binary before/after):

- ncu matched-nt: ffn_down-q4_K **−34.6%** (4.30 → 6.58 G-IMMA/s), gate/up
  **−35%**, regs 124 → 105 (scalar-decode register pressure gone);
- nsys census: q4_K bt busy 762.59 → 526.82 ms = **−30.9%**;
- **but ffn_down is only −5.2%** — r58's premise was half right: the winners
  were gate/up (−37%) and q/o (−18%), the classes with a large staging-phase
  decode ALU/I2F share; ffn_down's deficit is **L2 reuse-shaped** (each launch
  re-reads the 61.6 MB qa8 plane), not decode-shaped — the plane cannot help;
- item 3 (od re-tile) **skipped with evidence**: it needs ≤85 regs, DSC=true
  is 105, estimated ~+0.2%, below the co-tenant noise floor.

Memory: +1456 MB (measured; r58's 1.07 GB estimate erred on q4_K byte mass).
End-of-doc state: default 7B pp3314 ~3581 tok/s = **1.080×** vs llama-bench 3323.29.

## 6. Lessons

1. **The same symptom (a kernel class being slow) can come from opposite
   root causes** — per-kt decode ALU and per-launch A-plane DRAM traffic look
   identical in an attribution table but demand opposite fixes; classify
   "decode-shaped or reuse-shaped" first, then pick the lever.
2. **The baseline binary is a measuring instrument**: when it is not built
   from the code you think it is, every delta it produces is fiction (r59b
   formalized this into a protocol rule).
3. **Memory estimates must use real byte mass**: q4_K is actually 5.8 GB, not
   4.29 GB — a 36% difference in plane cost; cost accounting for plane-type
   changes belongs on the registration code, not on the spec table.
4. **Eliminating ALU beats hiding ALU**: r58 proved hiding (pipelining) is
   negative on a pure-copy host; this doc proves eliminating (planarization)
   is real at the same site — first ask "can this cost be made not to
   exist", then ask "can it be hidden".

---
← 61 · [Index](./README.md) · 63 →
