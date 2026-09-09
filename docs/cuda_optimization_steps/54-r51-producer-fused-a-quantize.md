# 54 · r51 — producer-fused A-quantize mode 1 (LANDED)

> **Result**: the MMQ A-quantize prepass is folded into its producer kernels — `rms_norm_quant_f32_t` / `swiglu_quant_f32_t` write the pad40_t transposed quantized plane in the same pass that produces the f32 output (byte-identical to the standalone prepass). prepass launches **110 → 28** (only wo remains), prepass time **83.0 → 10.1 ms**; the fused swiglu runs 4.09 ms per call vs 5.46 ms for the original pair (−25%); whole prefill **2803.4 → 2856.4 (+1.89%)**, vs-llama 1.18× → 1.16×.
> **Commit**: `cf1ed4b` (code, +421 lines) + `bf0c986` (record). **Date**: 2026-09-06.

## 1. Background — where things stood

With r48 (FAP2 register-resident softmax, +5.6%) and r49 (A-quantize prepass shared-A dedup, +2.32%) landed, the baseline stood at 2797.5 tok/s, vs-llama 1.18×. r47's converged-regime wall decomposition had split the remaining wall into three pieces: q6_K GEMM (15.8% at the time, already driven to 1.13× and near convergence by r38–r41), FA (r48 had cut it to ~4.7%), and a **hidden tax nobody had looked at squarely — the quantize prepass**.

r49's dedup had already cut the prepass from 193 launches to 110 (118.4 → 83.9 ms), but it only removed **redundancy** — quantizing once when the same A is shared by the q/k/v GEMMs. Each of the remaining 110 launches was still a standalone kernel that "reads the f32 activations once, writes the transposed q8 plane once," occupying **~7.4%** of the converged wall. And every one of those 110 reads data **its producer just finished writing**: in the qwen2 prefill graph, every rms_norm / swiglu output **exclusively** feeds the next GEMM group's A input — input-norm → q/k/v, ffn-norm → gate/up, swiglu → down, output-norm → lm_head. The data is freshly written out of L2, then read back by another kernel.

This is the other half of the **layout-transformation locality** lesson r34 (the quantize-transpose prepass, +9.72%) already taught: back then r34 moved the layout transform out of the GEMM kernels into a standalone prepass, winning on "transform once, let the GEMMs reuse it"; but the price was that the transformed data had to be **consumed again**. r49 deleted the repeated transforms; r51 deletes the **distance between the transform and the production** — why not have the producer write the plane while it's at it?

The appeal of this direction is its gain structure: it touches no GEMM, no FA, no hot loop's instruction stream — it just sews two back-to-back kernels into one. Mathematically it is pure reordering — quantize is a pure function of the f32 values, the f32 output is bit-unchanged, the plane is bit-unchanged. All the risk concentrates in the **scheduling window**: the plane must be ready before the consumer arrives, must be keyed correctly, and its tail fill must match the standalone prepass exactly.

## 2. Principle — the GPU mechanism

### 2.1 What the pad40_t plane is and how the GEMM eats it

The A side of MMQ (int8 tensor-core GEMM) does not consume row-major q8_0 but the **transposed + swizzled** layout r34 established: the token dimension is grouped in 64s (`MMQ_A_BLK = 64`, aligned with MMQ's NBI) and padded ("pad40": each token's q8_0 block is 40 B, 64-token blocks aligned). The standalone prepass `quantize_q8_0_pad40_t` handles one (token, chunk) per thread — a chunk is 32 consecutive elements — and emits two planes:

- `yqs [ntb][nchunk][2048]`: the **swizzled** qs bytes for each (64-token block, chunk). The write offset `(((t4*2 + (u>>2)) ^ xswz) << 4) + (u&3)*4` (`t4 = r&3`, `xswz = (r>>2)&7`) is the XOR swizzle finalized in r27 — when the BT kernel reads A in warp slices, this arrangement guarantees the same warp's 16 B blocks land in different bank groups, conflict-free;
- `ysda [ntb][nchunk][256]`: the packed `d|ssum` scale — the f16 `d` (= amax/127) and the int16 `ssum` (the integer sum of the quantized values, used for the min-term correction) packed into 4 B, laid out per r31's q-major region-split.

The BT kernel's A staging is therefore **pure bulk copy** (cp.async after r45): the qa8/sda planes are moved into smem directly by (block, k-tile), and the A side does zero index arithmetic in the hot loop. That is the plane's whole reason to exist — quantize, transpose, and swizzle all leave the hot loop.

### 2.2 The byte-count account

7B @3325 tok. rms-class (d=3584): `nchunk = 3584/32 = 112`, `ntb = ⌈3325/64⌉ = 52` → plane = 52×112×2048 = **11.9 MB** qs + 1.49 MB sda; f32 activations 3325×3584×4 = **47.7 MB**. swiglu-class (nf=18944): `nchunk = 592` → plane = **63.1 MB** qs + 7.9 MB sda; f32 output 3325×18944×4 = **251.9 MB**.

| Producer | production body (DRAM-grade) | what the standalone prepass pays extra | after fusing |
|---|---|---|---|
| rms d=3584 | read x 47.7 + write y 47.7 MB | **another 47.7 MB read** (L2 likely cold by then) + 13.4 MB plane write | the re-read hits L1/L2, near-free on the DRAM side |
| swiglu nf=18944 | read gate+up 503.7 + write dst 251.9 MB | **another 251.9 MB read** + 71 MB plane write | same |

The key mechanism: the fused kernel's phase 2 **re-reads the rows this block just wrote** — swiglu is one block per token row, rms is one block per 8 rows (one warp per row) — so those bytes are 100% still in L1/L2 (a swiglu dst row is 19.5 KB; one block's working set is far smaller than L1/L2 capacity). The standalone prepass instead arrives hundreds of microseconds later, after kernels like the GEMMs and FA have flushed L2, and most likely falls to DRAM. **What is saved is not the byte count, it is the bytes' temperature** — r34's lesson taken to its end: consume the data where it is hottest.

### 2.3 Why it is bit-identical

The fusion changes no floating-point expression: phase 1's rms / silu·mul elementwise formulas are exactly those of the standalone kernels, and the reductions' lane mappings and order are identical → the f32 output is bit-identical; phase 2's quantize body is a **verbatim copy** of `quantize_q8_0_pad40_t`'s code (the same amax grouping — serial amax over each 32-element chunk, the same `rintf`/clamp, the same swizzled writes) → the plane is bit-identical. Quantization is a pure function independent per (token, chunk), so identical input bits + identical code necessarily produce identical output bits — this demotes "did the fusion change the numbers" from a verification problem to a code-review problem. (Contrast r50's lesson: there the **grouping** of f32 reductions changed, which necessarily drifts ULPs; here no grouping changed.)

### 2.4 Where the gain concentrates, and where it doesn't

The phase 2 the fusion adds to each producer is pure added work (L1/L2 reads + plane writes), so the gain = the deleted standalone prepass − the addition. swiglu's production body is large (503.7 MB read + 251.9 MB write) and the standalone down-prepass would have re-read another 251.9 MB → fused 4.09 ms vs 5.46 ms for the pair (**−25%**). rms's production body is small and itself bandwidth-bound (read x + write y ≈ 2×47.7 MB); at d=3584 it is 0.77 ms vs 0.77 ms — **a wash**. This unevenness is itself data: the gain exists exactly where the saved thing is a DRAM re-read.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**Why rms's phase 2 re-reads y instead of staying register-resident.** rms's phase 1 is one warp per row with lanes consuming `d/4` float4s strided; phase 2's quantize body needs one (token, chunk) thread to take an amax over 32 consecutive elements — the two lane→element mappings don't line up. Register residency would require handing values across threads (smem relay + regrouping the amax), i.e. rewriting the quantize body: losing the "verbatim copy" **structural guarantee** of bit-identity, in exchange for saving one L1-hit re-read. The design choice is to keep the verbatim copy: one `__syncthreads()`, then phase 2 re-reads `y` from L1/L2.

**Why one `__syncthreads` is enough.** Phase 2 reads only the rows **this block** wrote: rms's block writes 8 rows and phase 2 quantizes the same 8; swiglu's block writes 1 row and quantizes the same 1. The plane addresses are partitioned by (token-block, chunk) with zero overlap between blocks — intra-block synchronization suffices; there is no cross-block dependency and no grid-level sync needed.

**Why swiglu rejects the register-resident variant (by design).** If swiglu organized phase 1 by the quantize body's thread mapping (thread-per-chunk), each thread would write 32 strided f32s of dst — **non-coalesced f32 stores = 8× sector amplification**, catastrophic on a ~254 MB f32 output stream. So the shape is inverted: phase 1 uses block-per-token-row coalesced float4 silu·mul (same elementwise formula as `swiglu_f32`), and after `__syncthreads()` phase 2's threads re-read this block's row (just written, hot in L1/L2) through the verbatim quantize body. The record also left a hook for the next step: a smem-staged dst or register-resident quantize in phase 2 "could squeeze out another ~0.5–1 ms per launch" — that is r52's mode 2.

**Why the grid covers the 64-padded token count.** Every byte of the transposed plane must be written: the GEMM reads all `ntb×64` rows (out-of-range token rows never write C back, but if the plane's tail rows are not zero-filled, scratch reuse would carry the previous graph's stale bytes into amax/ssum — the result would still be blocked by C's write-back guard, but the plane would no longer be deterministic). The standalone prepass's grid naturally covers the padded total; the fused kernel must **replicate that**: rms's grid = `ntb × (64/8)` blocks of 256 threads, and phase 2 runs the quantize body on padded rows too (`t < n` false → emit zeros); swiglu's grid = `ntb*64` blocks, with padded rows running only the zero fill.

**Why it hangs off r49's MmqCache instead of a new channel.** r49's cache key is (f32 output device pointer, nt, id), and the consumer `prefill_mmq` already consults it. The fused kernel keys the plane under the same key (key = f32 output pointer), so the consumer changes by zero lines. The scratch sizes are exactly those of `mmq_quantize_transposed`, guaranteeing `get_or_grow` returns the same pointer at hit validation — otherwise a grown buffer would make a "hit" read a misaligned plane.

**The gate design**: `MINFER_MMQ_A_FUSE=1` **AND** the full MMQ gate set (RAW/RAW_NB/A_TRANSPOSE/Q6K_NB) + `rows ≥ 16` + `dim % 256 == 0`. The first four prevent "writing a fused plane for a GEMM that would re-quantize natively anyway" (correct but pure waste); `rows ≥ 16` keeps decode (nt==1), the capture window, FusedFFN, and short prefills on the unfused pair; `dim % 256 == 0` satisfies the transposed GEMM's `nchunk % 8` requirement. OOM returns `Err` and the caller falls back to the unfused pair.

### 3.2 Key code

The fused rms kernel (current tree `src/cuda_kernels.cu` lines 840–880; phase 1 shares `rms_norm_f32`'s lane mapping, phase 2 is a verbatim quantize body):

```cuda
__global__ void rms_norm_quant_f32_t(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y,
    uint8_t* __restrict__ yqs,   // [ntb][nchunk][2048] swizzled qs plane
    uint8_t* __restrict__ ysda,  // [ntb][nchunk][256] packed d|ssum
    int d, float eps, int n, int nchunk, int ntb
) {
    // Phase 1: rms_norm — one warp per row, lane mapping and accumulation
    // order identical to rms_norm_f32 (bit-identical output).
    const int row = blockIdx.x * RMSQ_RPB + (threadIdx.x >> 5);
    ...
        ss = warp_reduce_sum(ss);
        float scale = rsqrtf(ss / (float)d + eps);
        ...
            y4[i].x = xv.x * scale * wv.x;      // ← f32 output: bit-identical to rms_norm_f32
            ...
    __syncthreads();
    // Phase 2: quantize this block's rows into the pad40_t plane — the
    // quantize_q8_0_pad40_t body verbatim (one thread per (token, chunk),
    // strided over the block's 8 rows). The grid covers the 64-padded token
    // count, so the padded tail rows are zero-filled exactly like the
    // standalone prepass (deterministic plane regardless of scratch reuse).
```

Phase 2's quantize body (current tree lines 887–916) — verbatim-identical to the standalone prepass, only the data source switched from `x` to the just-written `y`:

```cuda
        float dsc = 0.0f; int ssum = 0; uint32_t packed[8];
        ...
        if (t < n) {
            const float* src = y + (size_t)t * d + b * 32;   // ← the row just written, hot in L1/L2
            float4 sv[8];
            #pragma unroll
            for (int v = 0; v < 8; v++)
                sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
            float am = 0.0f;
            ... amax grouping ...
            dsc = am / 127.0f;
            ...
                int q = int(rintf(e[j] * di));               // ← verbatim-identical to the standalone body
                q = max(-128, min(127, q));
                ...
        }                                                     // t >= n → emit zeros (tail fill)
```

The fused swiglu kernel (current tree lines 943–968) — block-per-row coalesced phase 1 + re-read phase 2:

```cuda
__global__ void swiglu_quant_f32_t(
    const float* __restrict__ gate, const float* __restrict__ up,
    float* __restrict__ dst,
    uint8_t* __restrict__ yqs, uint8_t* __restrict__ ysda,
    int dim, int nt, int nchunk, int ntb
) {
    const int t = blockIdx.x;  // one token row per block (grid = ntb*64)
    if (t < nt) {
        ...
        for (int i = threadIdx.x; i < n4; i += blockDim.x) {   // coalesced float4
            float4 gv = g4[i]; float4 uv = u4[i]; float4 ov;
            ov.x = (gv.x / (1.0f + expf(-gv.x))) * uv.x;       // same formula as swiglu_f32
            ...
            o4[i] = ov;
        }
    }
    __syncthreads();
    for (int b = threadIdx.x; b < nchunk; b += blockDim.x) {   // phase 2 re-reads this row
```

The launcher's grid geometry (current tree lines 3740–3758) — where the padded coverage is settled:

```cuda
void launch_rms_norm_quant_f32_t(...) {
    int grid = ntb * (MMQ_A_BLK / RMSQ_RPB);        // covers all padded rows
    rms_norm_quant_f32_t<<<grid, RMSQ_RPB * WARP, 0, stream>>>(...);
}
void launch_swiglu_quant_f32_t(...) {
    swiglu_quant_f32_t<<<ntb * MMQ_A_BLK, 256, 0, stream>>>(...);   // one block per token
}
```

The host side (current tree `src/cuda.rs` lines 2978–3011) — same scratch, same cache key, `prefill_mmq` untouched:

```rust
pub fn rms_norm_quant(&self, x, w, y, d, n, eps) -> Result<(), String> {
    let nchunk = d / 32;
    let ntb = n.div_ceil(64);
    let qa8 = Self::get_or_grow(&self.buf_qa8_t, ntb * nchunk * 2048);  // same as the standalone prepass
    let sda = Self::get_or_grow(&self.buf_sda_t, ntb * nchunk * 256);
    if qa8.is_null() || sda.is_null() {
        return Err("cuda: rms_norm_quant plane OOM".to_string());       // OOM → fall back
    }
    unsafe {
        launch_rms_norm_quant_f32_t(x as *const f32, w as *const f32,
                                    y as *mut f32, qa8 as *mut u8,
                                    sda as *mut u8, d as i32, eps,
                                    n as i32, nchunk as i32, ntb as i32, stream);
    }
    // register the plane into the r49 MmqCache, key = f32 output pointer;
    // dead_write=false (mode 1 did write the f32)
    self.record_mmq_cache_transposed(y as usize, n, d, qa8 as usize, sda as usize, false);
    Ok(())
}
```

The consumer side's hit validation (current tree `src/cuda.rs` lines 2784–2796) — "the registered plane really got eaten" closes the loop through this half; the physical pointers must match:

```rust
let key = (x as usize, nt as usize, id as usize);
let mut cache = self.mmq_cache.lock().unwrap();
if cache.active && cache.key == key && cache.transposed {
    // get_or_grow may have reallocated on a larger miss: validate the
    // physical pointers so a grown buffer is never reused stale.
    let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8) as usize;
    let sda = Self::get_or_grow(&self.buf_sda_t, need_sda) as usize;
    if qa8 == cache.qa8 && sda == cache.sda {
        return (qa8, sda);          // ← hit: the standalone quantize launch is skipped
    }
}
```

The "before" for contrast: the standalone prepass path that got fused away (current tree `src/cuda.rs` lines 2813–2832) — in mode 0/fallback the consumer still goes through here, launching a standalone `quantize_q8_0_pad40_t` and filling the cache itself; r51's fusion only makes the "miss launch" step **stop happening** after particular producers:

```rust
// (before) r49 cache miss → standalone prepass launch + record
let qa8 = Self::get_or_grow(&self.buf_qa8_t, need_qa8);
let sda = Self::get_or_grow(&self.buf_sda_t, need_sda);
launch_quantize_q8_0_pad40_t(x, qa8 as *mut u8, sda as *mut u8,
                             id, nt, nchunk, ntb, stream);
cache.active = true;
cache.key = key;
cache.transposed = true;
cache.dead_write = false;       // mode 1: the f32 src was written, safe to re-quantize
cache.qa8 = qa8 as usize;
cache.sda = sda as usize;
```

The caller-side gate (current tree `src/graph/cuda_backend.rs` lines 536–538, SwiGLU arm; the RmsNorm arm is isomorphic):

```rust
let dim = node.out_shape[0];
let rows = if dim > 0 { n / dim } else { 0 };
if dim > 0 && n == dim * rows && rows >= 16 && dim % 256 == 0 {
    match self.state.mmq_a_fuse_mode() { 1 => { /* swiglu_quant, fall back to the unfused pair on failure */ } ... }
```

### 3.3 Pitfalls

1. **The temptation and price of register residency**. The "cleverest" shape (phase 1 keeps the silu·mul results in registers and quantizes them directly, saving the re-read) was rejected **at design time**: the thread-per-chunk mapping makes the f32 dst stores non-coalesced (8× sector amplification, unacceptable on a ~254 MB stream), and it breaks the verbatim-copy bit-identity guarantee. Lesson: pick a fusion shape by "which side's store pattern is coalesced," not by "which one saves a read."
2. **Tail fill is part of correctness, not an optimization**. If the padded rows don't get zeros, the plane is non-deterministic under scratch reuse; although C's write-back guard blocks the wrong values, the "plane bit-identical to the standalone prepass" property breaks and the verification distorts with it. The fused kernels replicate the standalone behavior exactly with grid coverage of the padded total + the `t < n` guard — the verifier specifically proves this with poisoned buffers (see §4).
3. **The uneven gain is mechanism, not defect**. fused rms at d=3584 is a 0.77 vs 0.77 wash — rms's production body is small and itself bandwidth-bound, so the fusion only trades in one L1/L2 re-read. Expecting "everywhere +25%" would misjudge the landing as a failure; the real headliner is swiglu→down.
4. **wo was deliberately left out of v1**. The producer of the attention output (wo's A input) is the FA kernel — its output layout is not isomorphic to token rows, so v1 doesn't touch it. The remaining 28 launches correspond exactly to 1 wo per layer, and that arithmetic is itself the proof that "fusion coverage is complete."

## 4. Verification

- **standalone verifier** (nvcc-compiled alone, `valid_r51.cu`): fused kernels vs (standalone producer + standalone prepass) **byte-equal on 11 shapes** — including `nt=3354/129/64/63/1`, `d=896 nt=300`, `d=640 nt=100`, swiglu `nf=18944`, covering both the 7B and 14B geometries. Division of labor among the shape classes: `nt=1/63/64/129` specifically attack the 64-token block **boundaries** (single block, one-short-of-full, exactly full, spanning+tail), `nt=3354` is the real whole-prefill shape, and `d=896/640` test nchunk divisibility at non-primary sizes. The plane buffers are **pre-poisoned** (filled with garbage before running) — any unwritten plane byte (especially the padded tail) surfaces immediately. Guards against: the fusion changing bits, tail fill leaks.
- **parity ×3** (tolerance 1e-4, end-to-end logits): guards against end-to-end math breakage.
- **greedy-32 byte identity**: under mode 1 both the f32 outputs and the planes should be bit-identical to the old path — accumulation order untouched, so the gate should be green, and it is (contrast r50's lesson: that gate catches "regrouping"; there is none here).
- **suite 166/0/3**: the regression surface.
- **A/B interleaved measurement** (same window, same machine, distributions fully separated): guards against machine drift being read as a gain.

## 5. Results

- **prepass**: launches 110 → **28** (only wo remains — its producer is the FA kernel, deliberately untouched in v1); time 83.0 → **10.1 ms**.
- **kernel level** (whole-prefill totals): fused rms 54 × 0.77 ms (vs 0.77 wash), fused swiglu 27 × 4.09 ms (vs 5.46 ms for the original swiglu+quantize pair, **−25%**); net kernel time **−31 ms**.
- **wall clock** (7B @3325 tok): 2803.4 → **2856.4 (+1.89%)**, A/B distributions fully separated; vs-llama **1.18× → 1.16×**. Gain composition: the swiglu→down fusion (−1.37 ms per launch) plus launch-count-halving-scale scheduling savings.
- **the wall after landing** (the signpost for the next round):

  | Residual | time | share of wall |
  |---|---|---|
  | q6_K GEMM | 197.8 ms | 18% |
  | **fused swiglu** | **110.5 ms** | **10%** |
  | FA | 52.9 ms | 4.8% |

  fused swiglu is promoted to a visible residual, and its f32 write + L1/L2 re-read (~251 MB per layer) is exactly r52's mode-2 target.
- suite 166/0/3; greedy byte-identical; parity ×3.

## 6. Lessons

1. **Layout-transformation locality taken to its end**: r34 moved the transform out of the GEMM and won once; r51 moves the transform back into the producer and wins again — two faces of the same coin: **consume the data at its hottest moment and place**.
2. **A verbatim copy is a fusion's bit-identity proof**: making phase 2 a verbatim copy of the fused-away kernel's code turns "did the numbers change" from a verification problem into a review problem; the verifier's job narrows to coverage (poisoned buffers + tail shapes).
3. **Fusion gains concentrate where the production body is large**: reconcile launch by launch (swiglu −25%, rms a wash) and "where mode 2 is worth doing" becomes a reading, not a guess.
4. **AND the gate with the full set**: writing a fused plane for a GEMM that would re-quantize natively is correct waste — the fusion gate must be jointly true with the consumer path's gate set.

← 53-r50-fa-tkv-16 · [Index](./README.md) · 55-r52-skip-write-mode2 →
