# 71 · D3-6 — GQA q-head batched attention: all gates green, still reverted — the 5× L2 re-read was not the residual (REVERTED)

> **Result**: ncu `lts__t_sectors` 2,121,671 → 472,583 (multiple of the analytic 1-pass minimum of 458K **4.63× → 1.03×**) — the 5× K/V re-read was **eliminated exactly as designed**; yet live kernel time was flat: h4w 63.76 → batched 64.79 µs/layer (**+1.6%**, the target had been ≤54 µs). Conclusion (mechanism pinned): the 5× L2 re-read was **already fully latency-hidden in the live regime**; the attention residual (62.1 − 48.9 = 13.2 µs/layer ≈ 0.63 ms/step) is **exposed latency, not traffic** — every bytes-side lever is hereby ruled out. All correctness gates went green **before** the revert: parity ≤1e-4 (gqa=5/7, all shapes), argmax HARD gate (margins 2.187 / 0.557), rp=1.0 greedy byte-identical on both models.
> **Commit**: no code commit (the GQA-batched kernel existed only as `/tmp/d3/patch_2a.py` and was lost with /tmp; what survived is docs commit `92b0712` plus the GQA geometry coverage left in the test table). **Date**: 2026-09-08.

## 1. Background — where things stood

After D3-5 landed the fused-producer decode A quantization (doc 70), the distance ledger of the 14B decode campaign read: @3254 long context 21.68 t/s = 46.12 ms/step, llama.cpp 24.32 t/s = 41.12 ms — **still 5.00 ms short (−10.8%)**. D3-5's follow-up list laid out the known levers and priced each one: attention residual ≈ 0.63 ms, attn_v-q6K padded-f32 tail ≈ 0.28 ms, output head ≈ 0.44 ms, rms/elementwise chain ≈ 1.0 ms — about 2.35 ms combined, 47% of the remaining gap. **The attention residual was the largest single lever**, and the only one that came with "a clear mechanism hypothesis": when D3-4 landed the hybrid rpw dual-kernel dispatch, the judgment it left behind was — the part of attention time (62.1 µs/layer) above the DRAM floor (48.9 µs/layer) comes from the **GQA 5× re-read of K/V**, the so-called "L2 composition".

The provenance of that hypothesis is worth recording, because it decides what this step means. D1's attribution session initially concluded the decode split-attention was **latency-bound** (staging depth was the only kernel that grew with KV); D3-4, when landing the 4-warp h4w body, re-attributed the residual to L2 re-read composition — but that was an inference, not an experiment: at the time there was no control that "cut the re-read". In other words, at the opening of D3-6 **both attributions were still alive, and they made opposite predictions for the same experiment**. Which one was right decided whether Stage 3's attention direction would be "cut traffic" or "cut latency".

The re-read geometry is straightforward: 14B is 40 q heads : 8 kv heads (gqa=5). The h4w split-attention grid is `(ATTN_SPLITS=32, n_head=40)` — **each block serves one q head**. The K/V rows of the same kv head are read once each by 5 mutually independent blocks (different SMs, no shared L1), so L2 sees 5× the sectors. 7B is more extreme: 28:4, gqa=7. D3-4's own words recorded it as "L2 composition" — the residual is composed of re-read traffic.

The hypothesis deserved an immediate lottery ticket, because its payout path is remarkably clean on paper: put all gqa q heads of the same kv head into **one block**, loop the windows reading K/V once, with the five warps each computing their own q — traffic returns to 1× and the math does not move a single bit. The D3-6 session (base `/tmp/d3/minfer_pre_d36`, sha1 92485d3c, HEAD 6085928 window; sglang co-tenant present throughout, same-window interleaved 3× median method, exactly the D3-5 protocol; guards: 7B tg128 ≥ 49.0 / @1641 ≥ 47.9, 14B tg128 ≥ 22.7) built it, all gates went green, and then the combined ncu/nsys evidence vetoed it. **What was vetoed was not code quality but the mechanism itself** — and that is precisely this step's most valuable output: it re-judged Stage 3's attention direction from "cut traffic" to "cut exposed latency".

One session-boundary note: item 2b from the D3-4/D3-1 follow-up (attn_v-q6K → MMVQ routing) did **not** get touched this time (no time); it passed intact to the next session — it is doc 72's 2b.

## 2. Principle — the GPU mechanism

**The byte ledger of the GQA re-read.** The K+V that 14B per-layer decode attention must read at nkv 3254 is: `nkv × n_head_kv × hd × 2 B × 2 (K+V)` = 3254 × 8 × 128 × 2 × 2 ≈ **13.3 MB**. At GB10's ~273 GB/s effective bandwidth, one full read has a DRAM floor ≈ 48.9 µs — matching the measured floor. The h4w grid is `(ATTN_SPLITS, n_head)`, and every q head's block must read its kv head's whole stripe once, so the request count on the L2 side is 5× (the DRAM side stays near 1×, because after the first read the same rows are resident in L2 and the other 4 passes hit L2). ncu's later measurement matched this analysis: **the analytic minimum of one 1-pass ≈ 458K sectors (13.3 MB ÷ 32 B/sector ≈ 416K, the remainder is Q and partial writes), and h4w measured 2,121,671 = 4.63×** (the part short of 5× is the first pass, which goes to DRAM — it does not travel the duplicated L2-sector counting path).

**The cache hierarchy decides at which level the "5× re-read" happens.** L1 is private per SM: the 5 q-head blocks almost certainly land on 5 different SMs, their loads of the same rows cannot see each other, and all of it floods into L2. The batched shape puts the 5 warps into **the same block** (same SM, shared L1): five warps read the same K row in the same window step — the first warp's load brings the row into L1, and the other four hit it directly. That is "L1 catches the sibling warps' same-address loads", and it is the microscopic mechanism by which sectors can drop to 1.03×.

**The batched shape's promise and its price.** Promise: grid `(ATTN_SPLITS, n_kv_heads)`, `32*gqa` threads per block (gqa=5 → 160, gqa=7 → 224), K/V traffic ÷5, paper time should fall from 63.76 µs toward the traffic-dominated limit (projection ≤54 µs). Price: blocks 40 → 8, threads per block 128 → 160/224. Totals reconciliation: h4w's grid (ATTN_SPLITS=32, 40 heads) = **1280 blocks × 4 warps = 5120 warps**; batched (32, 8 kv heads) = **256 blocks × 5 warps = 1280 warps** — total warps ÷4, blocks ÷5. What is resident on an SM is no longer "short 4-warp blocks, many small scheduling units" but "long 5-warp blocks, coarse-grained scheduling": the 5 warps in one block must enter and exit the same window loop together, and any warp's long stripe drags the whole block; conversely, the 5 copies of QK^T/V work for the same kv head share one KV read, and intra-block L1 reuse is the entire source of the gain. The partial buffer is shape-invariant: `ATTN_SPLITS × n_head × pstr × 4 B` = 32 × 40 × 136 × 4 ≈ 696 KB (pstr = (4+hd+3)&!3 = 136), and both shapes write the same [sp][head] layout.

**The critical counterweight: the latency roofline.** Decode attention is a GEMV-class kernel: each warp's time is a serial dependency chain (fetch K row → 8-dim partial dot → subgroup reduction → window max/rescale → fetch V row → FMA accumulate), 32 rows per window, windows back to back, with only one rescale's worth of parallel slack between windows. If that chain's depth decides kernel time and L2 bandwidth has slack anyway, then "reading 5× more" just makes L2 work extra in the shadow of other work — **it never enters the critical path**. That is exactly D1's original attribution (latency-bound); D3-4's "L2 composition" was an over-correction of it. D3-6's experiment design was therefore naturally a **hypothesis trial**: the traffic hypothesis predicts 63.76 → ~45–54 µs, the latency hypothesis predicts flat.

**A contrast that must be thought through first: the serialized profiler sees the opposite regime.** ncu serializes replay by default: cold L2, no concurrent memory-overlap — traffic dominates only in that regime. So ncu showing −28% while nsys live shows +1.6% is not a contradiction: **the two measure two different machines**. Live time is what the wall clock wants; ncu's value is proving that the batched shape "really did cut the traffic", which pins the veto reason at "traffic is not on the critical path" rather than "the batched shape was built wrong".

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **warp = q head, window loop shared.** The batched shape did not rewrite the window math: the per-warp 32-row window pass inside `attn_split_h4w_body` was **extracted verbatim** into a shared device function `h4w_warp_windows`, parameterized per warp (q head, stripe range). K/V base addresses are resolved once by `hk`, so the 5 warps share each window. Verbatim reuse of the window math is the precondition for the gate package not exploding — the parity package only has to prove "the mapping changed, the math did not", and that is the backbone of the bitwise-capable argument.
- **per-warp epilogue, 8/16 butterfly.** h4w has 4 warps co-writing **one** q head's partial (cross-warp LSE merge in the block epilogue, see excerpt C below). In the batched shape each warp's (mx, S) is rescaled online to its final state within the window loop — **warp-local, no cross-warp merge needed**; the only thing left to fold is each warp's 4 row-class oc copies (`oc0..oc3`, one 16-dim slice per 8-lane subgroup), summed with an intra-warp 8/16 butterfly. That is one fewer sync step than h4w's epilogue.
- **Static grid, dispatch slots untouched.** D3-4's self-gating is kept: grid/block contain no nkv (replay-safe), and the kernel self-gates internally on `rpw >= H4W_MIN_RPW`. The batched body and the 4-warp hybrid occupy **the same** rpw≥16 dispatch slot (active for gqa 2..8); at gqa=1 the batched shape degenerates to "one block, one warp" — zero benefit — and gqa>8 pushes the block into the other occupancy class at 288+ threads — both stay on the 4-warp hybrid; the incumbent 1-warp body for rpw<16 stays bitwise.
- **Shapes not chosen**: smem-tile cooperative staging (put KV into shared memory and hand it out to warps) was ruled out at the time — D2's cp.async series has a record of negative results at exactly this granularity. The batched shape's selling point is precisely "move only the mapping, not the math, not the staging": if even that does not gain, the bytes side is hopeless.

### 3.2 Key code

> ⚠️ **Code survival note**: the GQA-batched kernel **never landed as a commit** — it lived as a patch in `/tmp/d3/patch_2a.py`, was reverted after measurement, and /tmp has since been wiped. Excerpts (A)(B)(C)(D)(F)(G) below all come from the **current tree** and show the h4w structure that the batched shape reused verbatim / inherited (window loop = the original text that was extracted into `h4w_warp_windows`; block epilogue = the contrast to the step the batched shape **eliminated**; self-gating = the dispatch slot verbatim); the batched body itself is narrated as a reconstruction from (E)'s test-table comments and the record. (H) is the gate that **survived the revert** — GQA geometry now permanently covers the live h4w kernel.

**Excerpt A · the h4w body's thread mapping and stripe partition (current tree, `attn_split_h4w_body`)** — the batched shape keeps the `lane/w/t/g` semantics and changes only `h = blockIdx.y` (q head) to `h = hk*gqa + w`, and the block thread count to `32*gqa`:

```cuda
// src/cuda_kernels.cu (current tree; the h4w body landed in D3a, promoted in D3-4)
__device__ __forceinline__ void attn_split_h4w_body(
    const float* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, float* __restrict__ partial,
    int nkv, int sp, int h, int nh, int nk, int hd, float scale, int pstr
) {
    const int gqa = nh / nk;
    const int hk = h / gqa;                    // ← the batched shape moves this from "computed per warp"
    const size_t stride_kv = (size_t)nk * hd;  //   up to the block boundary: warp w IS q head hk*gqa+w
    const int lane = threadIdx.x & 31;
    const int w = threadIdx.x >> 5;  // warp id
    const int t = lane & 7;          // 16-dim slice (hd=128 -> 8 slices)
    const int g = lane >> 3;         // row slot within a 4-row pass
    // Same device-side range split as the 1-warp kernel (identical [sp] rows,
    // so the combine sees the same split partitioning).
    const int chunk = (nkv + ATTN_SPLITS - 1) / ATTN_SPLITS;
    const int lo = sp * chunk;
    const int hi = min(nkv, lo + chunk);
    // Balanced contiguous stripes: warp w owns rows [lo + w*rpw, +rpw).
    const int rpw = (chunk + 3) >> 2;
    const int wlo = lo + w * rpw;
    const int wend = min(hi, wlo + rpw);
```

**Excerpt B · the window-loop core (current tree, the part the batched shape reuses verbatim)** — each warp advances its stripe in 32-row windows: 8-lane subgroups produce 4 rows of QK^T, in-window butterfly max, one rescale, probs staged through smem to feed the V accumulation. Note the K row address carries `hk * hd` — **the batched shape is precisely 5 warps concurrently issuing loads to this same address, which is what L1 absorbs**:

```cuda
    // (window loop; probs is a per-warp 32-float stage: __shared__ float probs[H4W_NTHREADS]; float* pw = probs + w * 32;)
    for (int b = wlo; b < wend; b += 32) {
        const int wl = min(32, wend - b); // rows in this window (warp-uniform)
        float kq = -INFINITY;             // this lane's row score
        float mx_new = mx;
        const int np = min(8, wl);
        #pragma unroll
        for (int p = 0; p < 8; p++) {
            if (p >= np) break;
            const int row = b + g * 8 + p;
            float d = 0.0f;
            if (row < wend) {
                const __half* krow = k + row * stride_kv + hk * hd + 16 * t;
                const uint4 ka = *reinterpret_cast<const uint4*>(krow);   // 8 halves/uint4
                const uint4 kb = *reinterpret_cast<const uint4*>(krow + 8);
                d = h4w_dot8(ka, qc0, qc1) + h4w_dot8(kb, qc2, qc3);
            }
            // The subgroup reduce runs for ALL lanes: a shfl_sync with the full
            // mask deadlocks when subgroups diverge on row validity (found by
            // the standalone probe at nkv=3), so the guard selects the score
            // AFTER the reduction.
            float s = h4w_subgroup_sum8(d) * scale;
            if (row >= wend) s = -INFINITY;
            mx_new = fmaxf(mx_new, s);
            if (t == p) kq = s; // lane (g,t) keeps subgroup g's row (g,p)
        }
        #pragma unroll
        for (int off = 8; off < 32; off <<= 1)   // cross-subgroup window max
            mx_new = fmaxf(mx_new, __shfl_xor_sync(0xFFFFFFFFu, mx_new, off));
        const float wsc = expf(mx - mx_new);     // one rescale per 32-row window
        mx = mx_new;
        kq = expf(kq - mx);
        S = S * wsc + kq;
        /* oc0..oc3 scale the same way *= wsc; __syncwarp(); pw[lane] = kq; __syncwarp();
           V accumulation: subgroup g takes row b+4p+g's prob × V row, oc copies accumulate */
    }
```

**Excerpt C · h4w's block epilogue (current tree) — the contrast to the step the batched shape eliminates.** h4w's 4 warps co-write one q head, so it must merge LSE across warps: 4 copies of (mx, S) go into smem, gmax is computed, each warp rescales its own oc, then a stride-128 walk sums and writes the partial. In the batched shape this entire passage disappears — each warp's own (mx, S) is already final, and the partial slots are written per q head:

```cuda
    // ── block epilogue: LSE-merge the 4 warp states, write ONE partial ──
    // (w,g) copy of dim d lands at vkq_s[w*512 + g*128 + d]: lane (g,t) stores
    // dims [16t,+16) at offset w*512 + g*128 + 16t, so the final per-dim sum
    // is a stride-128 walk — bank-conflict-free across threads.
    __shared__ __align__(16) float vkq_s[4 * 512];
    __shared__ float mx_sh[4];
    __shared__ float s_sh[4];
    float Sw = warp_reduce_sum(S);
    if (lane == 0) { mx_sh[w] = mx; s_sh[w] = Sw; }
    __syncthreads();
    const float gmax = fmaxf(fmaxf(mx_sh[0], mx_sh[1]), fmaxf(mx_sh[2], mx_sh[3]));
    const float wsc = expf(mx - gmax); // idle warp: exp(-1e38 - gmax) == 0
    /* oc0..oc3 scale the same way *= wsc; the four float4 copies are written into vkq_s; __syncthreads();
       if (tid < hd): stride-128 walk accumulating the four copies; dst[0]/dst[1] are written by tid==0,
       merging the LSE pair with s_sh[w]*expf(mx_sh[w]-gmax) */
```

**Excerpt D · the self-gating hybrid kernel (current tree, the dispatch slot the batched shape inherits)** — grid/block contain no nkv at all (`positions[0]` is read device-side), so every replay re-gates itself; the batched body occupies this same `rpw >= H4W_MIN_RPW` slot:

```cuda
// src/cuda_kernels.cu (current tree)
#define H4W_NTHREADS 128 // 4 warps per block
#define H4W_MIN_RPW 16   // 4-warp body only when rows/warp amortize the window

__global__ void __launch_bounds__(H4W_NTHREADS, 8)
gqa_attn_split_partial_hybrid(
    const float* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, float* __restrict__ partial,
    const int* positions, int nh, int nk, int hd, float scale, int pstr
) {
    const int nkv = positions[0] + 1;
    const int chunk = (nkv + ATTN_SPLITS - 1) / ATTN_SPLITS;
    if (((chunk + 3) >> 2) < H4W_MIN_RPW) return;   // rpw < 16 -> incumbent body owns
    attn_split_h4w_body(q, k, v, partial, nkv, blockIdx.x, blockIdx.y,
                        nh, nk, hd, scale, pstr);
}
```

**Excerpt E · the batched body itself (reconstructed narration)** — the patch's shape, reconstructed from the record and the test comments: `__global__ void gqa_attn_split_partial_gqa(...)`, `grid = (ATTN_SPLITS, n_kv_heads)`, `block = 32*gqa` threads; `warp w` serves q head `hk*gqa + w` (`hk = blockIdx.y`); each warp calls the extracted `h4w_warp_windows` (excerpt B's loop, with the K/V base `hk*hd` shared by the five warps); the epilogue is per-warp — warp-local (mx, S) writes its partial slot directly, and the 4 row-class oc copies are folded in-warp with the 8/16 butterfly. The rpw self-gating condition matches excerpt D; enabled for gqa 2..8, falling back to the 4-warp hybrid at gqa=1 and >8.

**Excerpt F · Rust-side dispatch (current tree `src/cuda.rs`)** — the fixed-grid partial layout is a contract shared by both kernels (idle splits write `mx=-INF/S=0`, which the combine weights to zero):

```rust
// src/cuda.rs (current tree, excerpted comments)
/// ... in cuda_kernels.cu (fixed grid — the graph-replay capture depends on
/// it; idle splits write an mx=-INF/S=0 partial the combine weights to zero).
pub fn gqa_attn_split(&self, q, k, v, o, positions, nh, nk, hd, scale, f16_kv) {
    let pstr = ((4 + hd + 3) & !3) as i32;
    const ATTN_SPLITS: usize = 32; // mirrors #define ATTN_SPLITS in cuda_kernels.cu
    let need = ATTN_SPLITS * nh * (pstr as usize) * 4;
    let partial = Self::get_or_grow(&self.buf_attn_partial, need);
    // f16_kv ? launch_gqa_attn_split_f16kv(..) : launch_gqa_attn_split_f32kv(..)
    //   -- both launchers enter the hybrid/batched body when rpw>=16
}
```

**Excerpt G · the dual-kernel launcher (current tree, the dispatch structure the batched shape was inserted into)** — the shape of D3-4's self-gating dispatch at the launcher layer: two static launches + combine; the batched shape back then took the hybrid's place in the `rpw>=16` slot (gqa 2..8) and inherited the "dud launch" mechanism unchanged:

```cuda
// src/cuda_kernels.cu (current tree, f16kv launcher; excerpted comments)
// D3-4 L1: hd == 128 (Qwen2.5/Qwen3 decode shapes) dual-kernel
// self-gating dispatch on rpw = ceil(ceil(nkv/ATTN_SPLITS)/4):
// rpw >= H4W_MIN_RPW (nkv >= 1921) -> the 4-warp fattn-vec-style kernel;
// rpw < 16 -> the incumbent 32-thread D2-staged kernel. Both launches are
// static (grid/block nkv-independent), so CUDA-graph capture/replay is
// unaffected; each kernel re-reads positions[0] on every replay and
// exactly one is live for the current nkv (the rpw branch is
// nkv-uniform). The dud launch costs ~1-2 us/layer but keeps the
// small-rpw shapes on the incumbent geometry — running the 1-warp body
// inside 128-thread blocks caps the SM at 12 working warps (1536/128)
// and measured +78% kernel at 7B @1641 (35.4 vs 19.8 us, nsys).
if (hd == 128) {
    gqa_attn_split_partial<__half><<<dim3(ATTN_SPLITS, n_head), 32, 0, stream>>>(/*...*/);
    gqa_attn_split_partial_hybrid<<<dim3(ATTN_SPLITS, n_head), H4W_NTHREADS, 0, stream>>>(/*...*/);
}
gqa_attn_split_combine<<<dim3(1, n_head), hd, 0, stream>>>(partial, o, n_head, hd, pstr);
```

**Excerpt H · the gate that survived the revert (current tree `src/graph/cuda_backend.rs`)** — the two GQA geometries added to the parity test table for 2a were kept intact. Note that the comments still speak in pre-revert language ("drives the GQA-batched kernel"); after the revert these shapes actually cover the **live h4w hybrid** in the nkv≥1921 range — in the master table's words, "these shapes now permanently cover the LIVE h4w kernel too":

```rust
// src/graph/cuda_backend.rs (current tree, split-decode parity test-table entries)
// D3-6 2a: the 14B GQA geometry (40:8, gqa=5) drives the
// GQA-batched kernel (grid (ATTN_SPLITS, 8), 160 threads) on the
// same pos0 sweep — the 1920/1921 boundary picks between the
// bitwise 1-warp incumbent (nkv 1920) and the batched body
// (nkv 1921), and 4094/4095 cover full-window chunk tails.
(40usize, 8usize, 128usize, 4200usize,
 [2usize, 32, 63, 64, 127, 128, 1023, 1919, 1920, 4094, 4095]),
// D3-6 2a: the 7B GQA geometry (28:4, gqa=7 → 224-thread blocks).
// nkv 2808 (pos0 2807) reproduces the in-situ decode-step shape at
// the divergence point seen in the 7B greedy gate.
(28usize, 4usize, 128usize, 4200usize,
 [2usize, 32, 63, 64, 127, 128, 1919, 1920, 2807, 4094, 4095]),
```

(The pos0 table is nkv=pos0+1: a full sweep of 3..4096; 1920/1921 is exactly the rpw 15/16 dispatch boundary; 2808 is the in-situ decode-step shape where the 7B greedy gate hit its knife-edge.)

### 3.3 Pitfalls

- **The full-mask shfl_sync deadlock lesson directly constrained the extraction.** That comment in excerpt B ("a shfl_sync with the full mask deadlocks when subgroups diverge on row validity — found by the standalone probe at nkv=3") was bought with a deadlock back in the D3a era: the subgroup reduction must have ALL lanes participate, and row validity is selected with `-INFINITY` **after** the reduction. The window-loop extraction into `h4w_warp_windows` must not break this — in the batched shape different warps can have different stripe lengths, so the divergence surface is larger than in the original shape.
- **The occupancy structure moved, and it moved a lot.** h4w is `__launch_bounds__(128, 8)` (4 warps/block, targeting 8 blocks/SM); the batched shape is 160 (gqa=5) / 224 (gqa=7) threads with blocks 40→8. The organization of resident warps on the SM is completely different — this is exactly the experimental face of the "wave-structure bound" hypothesis, and also one natural explanation of the +1.6% reading: the L2 wait saved by going 1-pass was eaten back by coarser-grained block scheduling.
- **The epilogue simplification was free — and bought nothing.** h4w's block epilogue must merge LSE across warps (excerpt C's `mx_sh/s_sh` + stride-128 walk); the batched shape's per-warp LSE is already final, so the epilogue is only the in-warp 8/16 butterfly — one fewer `__syncthreads` round, with no live-time gain, again pointing to the bottleneck not being in the epilogue.
- **The batched shape inherited the dud-launch tax.** The dead launch in the dual-kernel dispatch costs ~1–2 µs/layer (excerpt G's comment, verbatim) — the price D3-4 paid to keep small-rpw shapes on the incumbent geometry. The batched shape entered through the same slot mechanism and did not touch that ledger; it is the dud split ~0.070 ms/step item in the D3-7 census (48 layers × ~1.5 µs).
- **The partial-layout contract must not break.** The `partial` slot layout (`[sp][head] × pstr`) and the idle-split semantics (mx=-INF/S=0) are the combine kernel's input contract; the batched shape changed the meaning of the head dimension (the 40 q-head slots stay as they are), and the combine needs zero changes — this is the structural precondition for "moving only the mapping" to hold.

## 4. Verification

All gates ran **before** the revert decision — this revert was a mechanism veto, not a correctness veto:

- **Kernel-level parity (defends against kernel math errors)**: the split-decode parity test was extended with the 14B (40:8, gqa=5) and 7B (28:4, gqa=7) geometries, a full nkv 3..4096 sweep (including the in-situ 2808 divergence-point shape and the rpw 15/16 boundary), plus the D3a outlier calibration (the f16-representable residual class with |q|~57 and V ±144) — all ≤1e-4 green against the CPU reference. The outlier calibration test itself (current tree) — pushing the activation dynamic range to the f16-representable ceiling of |q|~57 / |v|~144, the batched body is still ≤1e-4 vs the CPU reference:

```rust
// src/graph/cuda_backend.rs (current tree, test excerpt)
// D3-6 2a: kernel-level gate of the calibrated tolerance package on
// realistic outlier-scale data (docs/CUDA_OPTIMIZATION.md §2D D3a:
// residual |q|~50, V outliers ±127 — the h4w body measured 6.5e-5
// vs CPU on this class, the incumbent 3.8e-5). The GQA-batched body
// shares the h4w window loop verbatim, so the same ≤1e-4 bound
// applies; 14B geometry (40:8, gqa=5) inside the batched regime
// (nkv 1921 / 4096, f16 KV).
let qs: Vec<f32> = (0..nh * hd)
    .map(|i| (((i * 37) % 19) as f32 / 5.0 - 1.9) * 30.0)  // |q| ~57
    .collect();
let vs: Vec<f32> = (0..nkt)
    .map(|i| (((i * 57) % 11) as f32 / 3.0 - 1.8) * 80.0)  // |v| ~144
    .collect();
assert_close(&format!("attn_split_gqa_batched_outlier(nkv={nkv})"),
             &agot, &aref, 1e-4);
```

These shapes therefore **entered the gate set permanently**: whether or not the batched shape lives, the live h4w kernel is from now on covered by real GQA geometry.

- **Dump gate (defends against end-to-end numeric drift)**: both models, a 2.8K-token prompt (fully inside the batched regime), −n 4: `logits_prefill`/`kv0_prefill`/early-layer KV bitwise (14B kv0–15, 7B kv0–4); `logits_decode` max|Δ| 0.265 (14B) / 0.259 (7B) — the calibrated 0.39-class; **argmax HARD gate green**, margins 2.187 / 0.557 (top-2 is far away, so a flip can only be a mechanistic error). Upper-layer KV decode-side drift = the documented f16-boundary class; the `node{2,3,5,8,11}_decode` and `node{3,5,8}_prefill` pool-slot aliasing diffs reproduce pre-vs-pre (a same-binary self-diff covers all of the pre-vs-post diff — no numeric delta).
- **Greedy + the sampler-vs-kernel attribution gate newly established by this step (defends against misreading a sampler knife-edge as kernel drift)** — this step's durable methodological contribution:
  - Under `--repeat-penalty 1.0` (bare argmax), the −n 256 streams are **byte-identical** on both models — across 512 steps the kernel never flipped a single bare argmax. This is a **strong gate** on kernel numerics: an argmax stream with no sampling machinery in the way is maximally sensitive to kernel numeric noise.
  - Under the default repeat penalty, each run had exactly **one knife-edge flip** (1/256 = 0.4%, far below the 2% line): 14B at step 63, where the bare top-2 probgap is only 0.0167 (logit gap 0.037) and the penalty swapped the order; 7B at step 8, where the post-penalty winner is bare rank 6+ (outside the traced top-5). After the flip the text stays coherent with no repetition degeneracy; the flip's next step is visible in `MINFER_TRACE` as a changed `embed` node input, while steps 0..k−1 align one by one with the baseline's top-5 down to the drift class — the cascade structure is self-consistent.
  - The temp-0.8 seed-7 sampled control is byte-identical on both models (the incumbent regime is symmetric on both sides, so sampling reordering cannot happen).
- **Suite 170/0/3** (including the FA trio and the extended split-decode parity).

## 5. Results (REVERTED + veto mechanism)

The window anchors are the same as D3-5 (14B tg128 23.35 / @3254 21.68; 7B tg128 50.69 / @1641 49.16; guards 7B ≥49.0 / ≥47.9, 14B tg128 ≥22.7). The batched shape does not move the wall clock, so all of these are the post-revert status quo. The combined two-instrument measurement (nsys: NO_CUDA_GRAPH, bench −n 8, mean of the last 384 steps; ncu: 2-capture serialized replay of the same launches):

| kernel | `lts__t_sectors.sum` | vs 1× analytic (458K) | nsys (live) | ncu (serialized) |
|---|---|---|---|---|
| h4w hybrid (5× re-read) | 2,121,671 | 4.63× | 63.76 µs | 150.8 µs |
| GQA-batched (1× read) | 472,583 | **1.03×** | 64.79 µs | 108.7 µs |

- **The traffic hypothesis was falsified**: the batched shape cut L2 sectors to 1.03× of the analytic minimum (L1 caught the sibling warps' same-address loads), yet live time went +1.6% (the projection had been ≤54 µs). **The 5× re-read was already completely hidden by the latency roofline in the live regime** — L2 has bandwidth slack, and the kernel is limited by the latency/dependency chain + wave structure. D1's original attribution (latency-bound) stands; D3-4's "L2 composition" note is corrected.
- **Why +1.6% was enough to veto**: this campaign's bar for a "win" is +1.5% (the landing bar, calibrated at D3-5); this lever's projected gain was −15% to −25% territory (63.76 → ≤54 µs). A measured reverse +1.6% means the mechanism direction is wrong as a whole — not "the gain was eaten by noise" but "the gain does not exist". When a prediction of this magnitude fails, the correct action is to revert and revise the attribution, not to tune and retry.
- **ncu's −28% is another regime**: serialized replay, cold cache, no concurrent overlap — the traffic-dominated world. It proves the batched shape "really did cut traffic", and it proves that this fact is worthless in the live regime.
- **The re-judgment for Stage 3**: the attention residual of 13.2 µs/layer ≈ 0.63 ms/step **cannot be reached by any bytes-side traffic lever**. The next attention lever must cut exposed latency: cross-window prefetch (while preserving occupancy — D3-4 L2's register-pipeline form lost 1:2 on the occupancy cost), or smem-tile cooperative staging (untried, but D2's cp.async negative results make this granularity historically disfavored). **Retry conditions**: a shape that can shorten the dependency-chain depth without reducing resident warps — otherwise the "elimination" of the re-read never converts into time.
- The distance ledger at session close (wall clock untouched): −5.00 ms (−10.8%) unchanged; the known list becomes rms merge ≈1.0 ms + attn_v routing ≈0.28 ms (handed to D3-7) + output head 0.44 ms (mechanism missing) + attention latency lever ≤0.63 ms (mechanism in doubt) — 1.7–2.35 ms combined = 34–47% of the gap, the rest being matmul aggregation + launch-structure slack, forming Stage 3's decision input.

## 6. Lessons

1. **Counter evidence cannot overrule live time.** Traffic ÷4.5 with time flat has only one self-consistent explanation: traffic is not on the critical path; a serialized profiler (ncu) measures the opposite regime of cold cache and no overlap, and its −28% is exactly what the "traffic-dominated world" looks like — only reading the two numbers together pins the mechanism.
2. **gate-green ≠ land.** All correctness gates green only proves the kernel is correct; a performance-mechanism error still gets reverted. The revert decision can (and should) happen after the gates are all green — gates are admission conditions, not landing conditions.
3. **(durable gate methodology) greedy divergence ≠ kernel drift.** The argmax under the default repeat penalty is a knife-edge among up to 64 penalized candidates, and a flip cascades on the next step (a different token enters embed). The clean kernel-numerics gate is **byte-identical rp=1.0 greedy streams**; attribute each flip in the default-penalty stream with a per-step `logits_top` trace to get the bare top-2 probgap — only flips with a large probgap are worth suspecting the kernel.
4. **Keep the test shapes of negative results.** The gqa=5/7 geometries + the 2808 shape + the rpw boundary were kept intact after the revert, going from "the batched shape's gates" to "permanent coverage of the live h4w" — the next person touching attention will not have to rediscover these shapes.

---
← 70 · [Index](./README.md) · 72 →
