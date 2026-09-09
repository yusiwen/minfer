# 13 · r7–r8 raw-byte MMQ kernel + wide tile + FA probe (LANDED (raw) + REVERTED (probe))

> **Result**: `mmq_raw_nt_kernel` (`MINFER_MMQ_RAW=1`, q4_K) landed: 472 vs 441 tok/s
> (+7%), GEMM kernel 20.2 → 23.0 ms same-window comparison, quantize pass 129 → 74 ms;
> the wide tile's first 2124 tok/s was a **phantom** — KD=8 needs 135 KB dynamic smem,
> over the ~99 KB opt-in cap, and both attr-set and launch failed silently with the
> GEMM writing nothing; after adding the cap guard the honest matrix has narrow KD=8
> 472 as the optimum. The FA KV L2-prefetch probe was null (2319–2345 vs 2345),
> reverted.
> **Commit**: `d440d16` (raw kernel), `d9d626a` (quantize rebuild), `a41eac0` (wide
> tile wip), `ef9d5b4` (phantom fix + guard), `87bade0` (FA probe record, probe code
> not kept). **Date**: 2026-09-01.

## 1. Background — where things stood

The r6 spec had just been written (step 12) and Era C's execution contract was in
place: smem holds raw bytes only, staging is pure cp.async, dequant moves into the
mma loop, the `MINFER_MMQ_RAW` gate keeps R1 intact, and the three-stage parity gate
goes first. This step is the spec's first round of execution, plus three adjacent
items carried along:

- **The raw kernel itself** (spec points 1–5): replace R1's word-staging (dequant
  ALU + expanded words resident in smem, serialized ahead of the barrier) with
  raw-byte staging. R1's 441 tok/s trailed the f16 GEMM path's 2318 tok/s by 4.9×,
  and the spec judged the common bottleneck to be the per-32-k-chunk inner-loop
  overhead — raw conversion is the first cut at compressing the inner loop's
  up-front cost.
- **The quantize pass rebuild** (spec point 7 done early): `quantize_q8_0_pad40`
  runs ~129 ms per 7B @2K prefill (380 launches, ~87 GB/s). In the current MMQ wall
  (4.7 s) that is only ~2.7%, so it was destined to measure flat — but it was
  explicitly positioned as **groundwork**: once the raw kernel lands, the MMQ wall
  shrinks several-fold and quantize's share surfaces. Fixing it first avoids
  conflating two variables later.
- **Wide tile + FA probe**: the wide tile is the other face of the spec's
  KD/footprint discussion (a 128-token block halves B traffic); the FA probe was a
  free check of whether the f16 wall's FA item (1.86 ms/layer, ~6%) still had cheap
  gains — `fa_stage_kv_async` is single-buffered, and if the stage stall comes from
  DRAM latency, prefetching the next KV tile early should be able to hide it.

The campaign's position did not change qualitatively after this step — best 472 is
still 4.9× from 2318 — but **elimination** advanced a great deal: the three
hypotheses of B-traffic reduction, doubled reuse, and KV prefetch all went out with
numbers attached, and the remaining hypothesis converged to "inner-loop instruction
economy", leading directly to r9's reference decode.

## 2. Principle — the GPU mechanism

**The raw-byte staging byte arithmetic.** For each (row, 32-k chunk), R1's
word-staging must: read 144 B of Q4KB, do 4 shift/masks, write 4 expanded int words
into smem, decode the scale — then the whole block crosses the barrier. The raw
scheme's ledger is completely different:

- A side (activations, pad40): 40 B per (token, chunk), and **the pad40 layout is
  itself a raw format** — `d`(2B) + `qs`(32B @4) + `ssum`(4B @36); word w of a
  fragment is exactly the raw int8 lane group k∈[4w, 4w+3), with not a single byte
  of conversion needed. Five 8 B `cp.async.ca` complete one chunk's transfer.
- B side (weights, Q4KB): 144 B per (row, 256-k super-block) = nine 16 B cp.async.
  Nibbles unexpanded, un-centered, resident in smem as-is.
- The only "compute" done in the staging section is the per-(row, chunk) scale
  prefetch — a synchronous global read of 3–6 B per unit, written into R1's sds/sdm
  layout. **This is deliberate**: keeping the C-fragment rescale code line-for-line
  identical to R1 leaves fragment mapping as the only parity risk surface.

smem footprint: A 2×8×64×40 = 41 KB + B 2×64×144 = 18.4 KB + scales ≈ 62 KB at
KD=8 → 1 block/SM; KD=4 ≈ 31 KB → 2–3 blocks/SM. r5's lesson (depth vs occupancy is
a function of staging weight) applies here: once the staging section's ALU is zeroed,
both depths are worth re-measuring.

**Dequant at mma time.** Inside the mma loop, one shift/mask per 32-bit word
unpacks the nibbles. In the gaps between `mma.m16n8k32` issue slots the INT32
pipeline is idle, so the unpack ALU overlaps tensor-core issue instead of
serializing. The fragment mapping is deliberately identical to R1's — the int
accumulator and accumulation order are bit-for-bit unchanged — so the parity gate
only has to verify "did the data movement move the right bytes", not the math
again.

**The wide tile's theoretical gain and its hidden premise.** Growing the block from
64×64 to 128×64: grid.y halves and per-output-cell B re-reads halve; meanwhile each
fragment word feeds 2× the mma. This lever's implicit premise is **that the B
re-reads actually happen at DRAM** — as we'll see, that premise is wrong.

**The FA probe's principle.** `fa_prefill_f16kv`'s KV staging is single-buffered:
stage(kt) → compute(kt) → stage(kt+1) in series; if the stage stall comes from DRAM
latency, issuing an L2 prefetch for the next block early could hide it inside
compute. But Qwen2.5-7B is GQA (7 q-heads sharing 1 kv-head): the same K/V rows are
read concurrently by 7 attention blocks, and after the first read brings a row into
L2 the remaining 6 blocks all hit in L2 — the tile is already L2-resident, and the
stage stall was never DRAM-bound.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Whole-super-block constraint**: raw staging moves B in whole 256-k blocks
  (KDR=8 is exactly one super-block), with a launcher-side guard `nb32 % 8 == 0` —
  when the data doesn't form whole blocks the raw path is refused and we fall back
  to R1. The constraint buys zero addressing branches on the B staging side.
- **Scale prefetch into sds/sdm, rescale code untouched**: the C-fragment rescale
  is code that R1's parity has repeatedly validated; reusing it means the raw
  kernel's parity verification focuses on staging and the fragment mapping —
  minimizing "new code volume" is a direct application of the r5 parity-hole
  lesson.
- **The wide tile as a clone of narrow, not a rewrite**: `mmq_raw_wide_nt_kernel`
  is cloned from the parity-proven narrow kernel, changing only the block size
  (128 tokens), the warp mapping (4×2 subs of 32×32), the B fragment count (2→4),
  and `sum[16]→[32]`. The clone strategy makes mapping errors diffable hunk by
  hunk — in hindsight this was exactly the key to localizing the phantom.
- **Why quantize's tree-reduce amax is bitwise-safe**: `max` is exact for any
  association order (float max has no rounding), so replacing the serial
  32-deep fmaxf chain with a 4-way tree leaves amax bit-identical; the `rintf`
  quantization pass is untouched line for line → output bit-identical. The
  performance gain (~87 GB/s → higher, 129 → 74 ms) comes entirely from the
  shorter latency chain and wider stores; numerics untouched.

### 3.2 Key code

**The raw kernel: staging and the double-buffer main loop** (`git show d440d16 --
src/cuda_kernels.cu`, excerpt):

```cuda
// ─── P6: raw-byte MMQ (llama.cpp structure) — q4_K first ─────────────
// smem holds RAW quant bytes; staging is pure cp.async; dequant happens
// in registers inside the mma loop. Only the per-(row, chunk) scales are
// pre-computed at staging (shared by the whole warp; the C-fragment
// rescale needs all 8 fragment rows, and computing them per lane would
// be 4x redundant work in the hot loop).
//   A: pad40 chunk = d(2B) qs(32B @4) ssum(4B @36) — consumed as-is
//      (word w of the fragment = the raw int8 lane group k 4w..4w+3).
//   B: Q4KB=144B super-block per row per 256-k; nibbles unpacked at mma.
// Requires whole super-blocks: nb32 % 8 == 0 (launcher guard).
__device__ __forceinline__ void mmq_cp8(uint8_t* smem_dst, const uint8_t* gsrc,
                                        bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 8 : 0; // src-size 0 => zero-fill the 8B chunk
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}

#define RAW_STAGE(kt, b)                                                       \
    do {                                                                       \
        /* A: (token, chunk) pad40 chunks as 5x 8B cp.async */                 \
        for (int x = threadIdx.x; x < MMQ_BI * KDR * 5; x += blockDim.x) {     \
            int kd = x / (MMQ_BI * 5);                                         \
            int rem = x % (MMQ_BI * 5);                                        \
            int r = rem / 5, u = rem % 5;                                      \
            int c = (kt) * KDR + kd;                                           \
            int tok = i0 + r;                                                  \
            const uint8_t* src =                                               \
                q8x + ((size_t)tok * nb32 + c) * 40 + u * 8;                   \
            uint8_t* dst = qa8 + ((size_t)(b) * KDR + kd) * MMQ_BI * 40        \
                         + (size_t)r * 40 + u * 8;                             \
            mmq_cp8(dst, src, tok < nt && c < nchunk);                         \
        }                                                                      \
        /* B: whole 144B super-block per row as 9x 16B cp.async */             \
        int sb = ((kt) * KDR) >> 3;                                            \
        for (int x = threadIdx.x; x < MMQ_BI * 9; x += blockDim.x) {           \
            int r = x / 9, u = x % 9;                                          \
            int j = j0 + r;                                                    \
            const uint8_t* src =                                               \
                W + (size_t)j * ((size_t)nsb * 144) + (size_t)sb * 144         \
                  + u * 16;                                                    \
            uint8_t* dst = qb8 + ((size_t)(b) * MMQ_BI + r) * 144 + u * 16;    \
            gemm_cp16((__half*)(void*)dst, (const __half*)(const void*)src,    \
                      j < od && sb < nsb);                                     \
        }                                                                      \
        /* per-(row, chunk) scales from global (a few B per unit, L2-hot) */   \
        ...  get_scale_min_k4(sg, blk + 4, &sc, &m);                           \
             sds[...] = h2f(*(const uint16_t*)blk) * (float)sc;                \
             sdm[...] = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);        \
    } while (0)

    RAW_STAGE(0, 0);
    gemm_cp_commit();
    int buf = 0;
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        if (kt + 1 < nktile) RAW_STAGE(kt + 1, buf ^ 1);  // prefetch the next k-tile
        gemm_cp_commit();
        gemm_cp_wait1();          // at most one prefetch group in flight
        __syncthreads();          // scales (plain stores) + landed bytes visible
        for (int kd = 0; kd < KDR; kd++) { /* mma loop: unpack nibbles in registers */ }
    }
```

Against R1: the staging section shrinks from "read 144 B + 4 shift/masks + write
expanded words + decode scale" to "five 8 B + nine 16 B cp.async + a 3–6 B scale
read"; the dequant ALU moves entirely into registers in the mma loop, overlapping
tensor-core issue.

**The quantize pass rebuild** (current tree from `src/cuda_kernels.cu:692`, matching
`d9d626a`'s change; the kernel comment itself records the bitwise justification):

```cuda
__global__ void quantize_q8_0_pad40(
    const float* __restrict__ x, uint8_t* __restrict__ y, int dim, int nt
) {
    ...
    // P6: tree-reduced amax (the serial fmaxf chain was latency-bound)
    // and 16B loads / 4B register-packed stores. Math is bit-identical:
    // max is exact for any association, the rintf pass is unchanged.
    float4 sv[8];
    #pragma unroll
    for (int v = 0; v < 8; v++)
        sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);   // 16B load
    float am = 0.0f;
    #pragma unroll
    for (int v = 0; v < 8; v++)                                   // 4-way tree amax
        am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                             fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
    ...
    uint32_t packed[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        const float* e = &sv[v].x;
        uint32_t p = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int q = int(rintf(e[j] * di));                        // rintf as-is
            q = max(-128, min(127, q));
            p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);         // register packing
            s += q;
        }
        packed[v] = p;
    }
    #pragma unroll
    for (int v = 0; v < 8; v++)
        *reinterpret_cast<uint32_t*>(dst + 4 + 4 * v) = packed[v]; // 4B store ×8
```

**The wide tile's warp mapping** (`git show a41eac0`, narrow's
`wm = warp >> 1, wn = warp & 1` replaced by 4×2 subs):

```cuda
const int wm = warp >> 2, wn = warp & 3;   // wide: 4x2 subs of 32x32
const int i0w = wn * 32;                   // each warp covers 32 od rows
const int j0w = wm * 32;                   // each warp covers 32 token columns
...
float sum[32] = {0.0f};   // [nh][h][l]: 4 B-frags x 2 A-frags x 4 C regs
```

**The smem-cap guard (the phantom's fix)** (`git show ef9d5b4 -- src/cuda_kernels.cu`,
the original r8-era shape; the current tree's launcher has since been rewritten by
r14's 16-chain layout with a recomputed smem budget, but the "check attr-set, check
launch, return 0 on refusal" structure survives to this day):

```cuda
extern "C" int launch_mmq_raw_wide_nt(...) {          // void → int: refusals reportable
    (void)type_id;
    // KD=8 needs 2*8*128*40 + 36.9KB + 16KB = 135KB — over the ~99KB
    // opt-in cap; an over-cap request fails SILENTLY (attr + launch both
    // ignored) and the GEMM silently writes nothing. Guard it: KD=4
    // (86KB) is the only feasible wide depth. Returns 0 when refused so
    // the caller can fall back to the narrow raw kernel.
    if (kd > 4) return 0;
    dim3 grid((nt + 127) / 128, (od + 63) / 64);
    const int smem = 2 * 4 * MMQ_WBI * 40 + 2 * MMQ_WBI * 144
                   + 2 * 2 * 4 * MMQ_WBI * 4;
    cudaError_t e = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(&mmq_raw_wide_nt_kernel<4>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return 0;                                     // attr failed: refuse + clear error
    }
    mmq_raw_wide_nt_kernel<4><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "minfer/cuda: mmq raw wide launch failed: %s\n",
                cudaGetErrorString(e));
        return 0;                                     // launch failed: refuse + report
    }
    return 1;
}
```

The Rust side correspondingly turned "launch succeeded" into an explicit protocol
(`git show ef9d5b4 -- src/cuda.rs`):
`wide_ok = launch_mmq_raw_wide_nt(...) == 1; if !wide_ok { /* fall back to narrow raw */ }`.

### 3.3 Pitfalls

- **The phantom 2124: a double silent failure.** The wide tile's first measurement
  was 2124–2124 tok/s — 4.5× faster than narrow raw's 472 and nearly matching the
  f16 GEMM path's 2318 — numbers so good they were suspicious. The truth: KD=8
  needs 135 KB of dynamic smem, over the ~99 KB opt-in cap; `cudaFuncSetAttribute`
  returned an error code **that nobody checked**, and the kernel launch then failed
  on the smem oversize too, its error swallowed by the subsequent error-clearing
  convention — the GEMM did not write a single byte. The 2124 tok/s wall clock was
  "the sound of no GEMMs running" (verbatim). The mmq_w4k parity failure the wide
  tile hit back at r7 (max diff 82.9, "partial-sum signature") was a fragment of
  the same root cause — over-cap garbage, not a mapping bug: at KD=4 the same
  mapping went parity all-green, proving the mapping itself correct.
- **The semantic chain of silent failure**: `cudaFuncSetAttribute` unchecked → the
  launch error swallowed by the `cudaGetLastError()` clearing convention → the Rust
  side receives "success" → the A/B numbers enter the record. Each link was
  "reasonable" on its own; combined they produce a phantom that can pollute
  decisions. The fixed protocol: the C side explicitly returns 1/0, attr and launch
  are each checked, refusals print to stderr; the Rust side falls back to narrow on
  a 0 return.
- **Hypothesis-cycling without ncu**: ncu was not yet available on this device
  (GB10) — it only worked from r13 — so phantom triage had to run on parity diffs
  and conservation reasoning. The r8 record states the risk explicitly: "further
  MMQ work is hypothesis-cycling" — which directly pushed r9 toward reading
  llama.cpp's source.

## 4. Verification

- **`cuda_prefill_mmq_parity` sweep through the raw path**: the env-routed q4_K
  sub-case runs the raw kernel and compares per-cell against the same host
  reference (defends: staging/mapping moving the wrong bytes).
- **7B greedy token identity vs the f16 path**: the whole generation token-for-token
  identical (defends: accumulation-order drift amplifying into behavioral
  differences over long sequences).
- **suite 169/0** (defends: revert/guard changes introducing a regression).
- **Guard refusal protocol**: over-cap requests return 0 + Rust falls back to narrow
  (defends: phantom recurrence — numbers must be able to prove the kernel actually
  ran).
- **Wide tile KD=4 parity all-green**: the evidence decoupling r7's w4k failure from
  mapping correctness (defends: discarding a correct mapping design by mistaking
  environmental garbage for a mapping bug).

## 5. Results

**Raw kernel (LANDED, `MINFER_MMQ_RAW=1` opt-in)**:

- 7B @2K CLI prefill: **472 vs 441 tok/s (+7%)**, at KD=8; KD=4 = 440 (depth still
  wins after staging ALU zeroed).
- GEMM kernel: 20.2 vs 23.0 ms; quantize pass: 129 → 74 ms (tree amax + packed
  stores, bit-identical).

**Wide tile (honest matrix, all parity-clean, interleaved in the same window)**:

| Shape | tok/s |
|---|---:|
| narrow cp.async KD=8 | **472** |
| wide KD=4 (86 KB, the only feasible wide depth) | 428 |
| narrow KD=4 | 427 |
| R1 word-stage KD=8 | 441 |

Both of the wide tile's theoretical gains were falsified: halving B traffic bought no
time (**L2 absorbs the B re-reads** — B's DRAM traffic was never binding); feeding 2×
mma per fragment word didn't pay either. The constraint-reranking conclusion: the
common bottleneck is the **per-32-k-chunk inner-loop overhead** (the rescale FMA
chain, scale smem reads, sync cadence) — no staging scheme can fix it.

**FA KV L2-prefetch probe (REVERTED, `87bade0`)**: null (2319–2345 vs the 2345
baseline). Veto mechanism: GQA's 7 q-heads/kv-head sharing keeps KV tiles naturally
L2-hot, so the stage stall is not DRAM-bound; `fa_prefill_f16kv` stays at its P5
state (1.86 ms/layer, ~6% of the prefill wall). Retry conditions: only when the KV
tiles' L2 residency is broken (e.g. much longer contexts or less q-head sharing)
does prefetch have latency left to hide.

**MMQ state (as of r8)**: best 472 is still **4.9×** from the f16 GEMM path's 2318;
the default f16 path untouched (2320–2370). The next step's information gap is not
another staging knob but "what does a fast MMQ look like" — leading to r9's
reference decode.

## 6. Lessons

1. **For a suspiciously fast wall-clock number, first prove the kernel actually
   ran**: check both attr-set and launch, report failures explicitly; guarding the
   smem cap is the launcher's duty, not the kernel's.
2. **B-traffic reduction is a dead lever on L2-resident re-reads** — ask at which
   level of the memory hierarchy the re-reads happen before paying a tile-shape
   price for them.
3. **"No speculative changes": a null probe reverts**, even if the change looks
   harmless; harmless ≠ beneficial, and a negative change left in the tree only
   pollutes later attribution.
4. **Do groundwork before it gets hot**: quantize was only 2.7% of the 4.7 s MMQ
   wall, but the moment the raw kernel landed it became a visible share — and in
   the r34 era it grew into one of the main levers.

---

← 12 · [Index](./README.md) · 14 →
