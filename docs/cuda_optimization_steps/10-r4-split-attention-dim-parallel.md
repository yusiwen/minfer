# 10 · R4 — decode split-attention dim-parallel rewrite (LANDED)

> **Result**: the `gqa_attn_split_partial` kernel 148 → 79 µs/layer (nsys); 7B @2K decode 39.2 → 43.2–45.1 tok/s (gap to llama's 44.9: 14% → ~4%); tg128 45.1 → 47.5–47.6 (overtaking llama's 47.1).
> **Commit**: `70f57db`. **Date**: 2026-09-01.

## 1. Background — where things stood

After R2 raised the decode matmuls' weight-stream efficiency (chapter 09), @2K decode stalled at
38.8 tok/s, still ~14% behind llama's 44.9. The nsys attribution was very clean: the gap sat
almost entirely on the split-attention kernel `gqa_attn_split_partial` — about 150 µs per layer,
28 layers ≈ 4.2 ms, one sixth of the ~25.8 ms per step at 7B @2K. And tg128 (short context) had
already reached 45.1 after R2, showing the matmuls were not the problem: **the attention
kernel's time grows linearly with KV length — the @2K gap is exactly it**.

This kernel is the flash-decoding path introduced by 8d (Phase 8's decode attention revision):
at nt==1 each query head has a single query row yet must attend over the whole context's K/V —
28 heads' parallelism alone cannot fill the GPU, so the KV axis is cut into several splits, each
split independently computes an online-softmax partial sum, and a combine kernel merges them.
8d's version fixed the split count at 8: 7B is `28 head × 8 split = 224` single-warp blocks.

The nsys readings of the disease (the record's own numbers):

- ~150 µs per layer against a single **4.3 MB K+V read** (= 2 × nkv 2048 × 4 kv-heads × hd 128 ×
  2 B f16) works out to an **effective stream rate of 28 GB/s** — under 11% of DRAM peak;
- the ACC accumulator is a **runtime-indexed `float4 oc[32]`** living in LOCAL memory (~80
  MB/layer of re-read/rewrite traffic);
- lanes walk rows with 4-byte loads, 64 scattered sectors per row, **12.5% sector efficiency**;
- the whole kernel is only 224 single-warp blocks, occupancy naturally poor.

Worse, the directional evidence: sweeping the split count up from 8, the kernel gets
**monotonically worse** — 148/172/419/609 µs @ SPLITS 8/16/32/64. Doubling parallelism makes it
slower, meaning the bottleneck is not "too few blocks" but **the per-warp geometry itself being
broken**: more splits just copy the same broken geometry more times, and LOCAL memory traffic
floods L1. That is why this step does not "raise SPLITS" — it rewrites the whole dims→lane
mapping.

## 2. Principle — the GPU mechanism

### 2.1 The structure of split-KV decode attention (why it must be two-stage)

Decode attention computes `o = softmax(q·Kᵀ·scale)·V`, with nkv the current context length. At
nt==1, without splitting, the parallel units are only the nh heads — 28 warps on 7B, not enough
to fill GB10 (48 warp slots per SM). Split-KV (flash-decoding) cuts the KV dimension into
`ATTN_SPLITS` pieces:

- **The partial kernel**: each `(split, head)` block runs online softmax over its own row range
  `[lo, hi)`, maintaining the triple `(mx, S, oc)`: `mx` is the max score seen so far, `S` the
  sum of exp weights, `oc` the weighted V accumulator. Each incoming row: `nmx = max(mx, s)`,
  `corr = exp(mx − nmx)`, `S = S·corr + e`, `oc = oc·corr + e·v` — numerically stable, and
  **the full score matrix never needs to materialize**; `(mx, S, oc)` is written into the
  partial buffer.
- **The combine kernel**: first take `gmx = max(mx_sp)` over all splits, then re-weight and sum
  each split's `S` and `oc` by `w_sp = exp(mx_sp − gmx)`, and `o = acc / S`. Mathematically
  equivalent to one softmax over the whole context — the merge weights `exp(mx_sp − gmx)`
  exactly correct the scale differences caused by each split's differing local maxima.

### 2.2 nkv independence: the hard precondition of graph replay

Decode's whole-step compute graph is CUDA-Graph captured and replayed repeatedly (7d). **Grid
dimensions are frozen into the graph at capture and replay cannot change them**; meanwhile nkv
grows every token. So split-attention's grid must be a **static shape** like `dim3(ATTN_SPLITS,
n_head)`, and nkv can only be read at kernel runtime as **data**:

```cuda
const int nkv0 = positions[0] + 1;                    // read from device memory at runtime
const int chunk0 = (nkv0 + ATTN_SPLITS - 1) / ATTN_SPLITS;
```

When the context is short, most splits get no rows (`lo ≥ nkv`) — their loop body never runs,
and they naturally write partial sums of `mx = −INF, S = 0`; on the combine side `w = expf(−INF
− gmx) = 0`, contributing exactly zero. **Idle splits need no special handling at all** — that
is why "static grid + data-side nkv" satisfies both replay safety and numerical correctness.
This structure was set by 8d and R4 keeps it unchanged — what R4 changes is the geometry
**inside** each warp.

### 2.3 The old geometry's three sins

8d's partial kernel assigns each lane **rows** (2 rows per lane); each lane computes the full
128-dim dot product itself and maintains a full 128-dim accumulator:

1. **A LOCAL memory accumulator.** `float4 oc[32]`'s bound `hd4 = hd/4` is a kernel parameter (a
   runtime value), so the compiler cannot fold it into registers → 512 B of local memory
   spill per lane. Online softmax rescales once per batch of rows: all 32 float4s read out of
   local, multiplied by `corr`, written back. nsys works it out to ~80 MB/layer of local
   traffic — **an accumulator that should be registers, faking memory transfers row by row**.
2. **Scattered row traversal.** Each lane walks its row with 4-byte (f16 `__half2`) loads: a 256
   B row should be 8 sectors, measured at 64 sectors touched per row (12.5% efficiency) —
   each DRAM transaction uses 1/8 of its bytes.
3. **A parallelism mismatch.** 224 single-warp blocks; and "raising SPLITS" only gives each
   block fewer rows, making the fixed costs (q load, partial write-out) harder to amortize
   and multiplying the local-traffic copies — the monotonic SPLITS-sweep degradation is this
   mechanism's direct symptom.

### 2.4 The new geometry: lane ↔ dim binding

R4 flips the mapping: **each lane permanently owns 4 dims** (`d0 = lane_id·4`; at hd=128 the 32
lanes exactly cover all dims), and rows are traversed cooperatively by the whole warp:

- The accumulator `oc` is **one float4 register** — zero spill, and the rescale is 4 FMAs;
- Row access via `kv_ld4`: a 16-byte load where 32 lanes each take the row's 4 consecutive dims
  → one coalesced access covers the whole row, sector efficiency restored;
- A single row's dot = this lane's 4-dim partial + a 5-step `__shfl_xor` butterfly → **every
  lane holds the row's complete dot product**, and the softmax state is naturally
  warp-uniform. (In the old design uniformity was free — each lane owned whole rows; the new
  design trades one butterfly per row for all of the local traffic — a trade that always
  wins.)
- Rows are processed **4 at a time**: the 4 rows' K loads are issued in batch first, and the
  serial softmax chain (max → exp → rescale) latency is covered by the later rows' loads.

The arithmetic comparison: old kernel 4.3 MB / 150 µs = 28 GB/s; the new kernel at 79 µs → ~54
GB/s, doubled — the remaining gap is a compute/latency mix, no longer a memory-geometry error.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Lane-owns-dims rather than lane-owns-rows**: shrinking the accumulator from O(hd) to O(1)
  per lane is the root-cause fix; handing row traversal to warp cooperation incidentally makes
  the access pattern coalesced.
- **The dispatch gate `hd % 4 == 0 && hd <= 128`** (the kernel comment's own words: "enforced by
  the dispatch"): the hd=128 7B/14B take the new kernel; shapes not meeting it keep the old
  path — with hd > 128 the dims→lane mapping leaves many idle lanes, not worth it.
- **ATTN_SPLITS 8 → 32**: under the new geometry each split's fixed cost is no longer amplified
  by local traffic, and more splits buy parallelism from 224 → 896 blocks; tg128 (nkv=128,
  only 4 rows per split) benefits too. The **static grid** stays untouched — replay safety.
- **The partial layout follows the geometry**: the old layout was "lane 0 writes out all hd4
  float4s"; the new layout is dim-sliced — each live lane writes its own `oc` to `dst + 4 +
  d0`, **no cross-lane reduction of any kind needed**. The combine side reads by index `p[4 +
  i]` (`i = threadIdx.x`, hd threads).
- **Idle lanes are not culled**: lanes with `d0 ≥ hd` keep participating in the butterfly with
  `q4 = 0` (zero contribution) — one warp-uniform fast path, no branches introduced.

### 3.2 Key code

**Before — the 8d kernel: rows to lanes, LOCAL accumulator, per-item rescale** (commit
`70f57db`'s deleted side, excerpted verbatim):

```cuda
const float4* q4 = reinterpret_cast<const float4*>(q + h * hd);
int hd4 = hd / 4;                       // a runtime value → dynamic indexing
float mx = -INFINITY, S = 0.0f;
float4 oc[32];                          // 512 B/lane → LOCAL memory
#pragma unroll
for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

for (int base = lo; base < hi; base += 64) {   // 2 rows per lane, 64 rows per batch
    ...
    float nmx = fmaxf(mx, bmx);
    float corr = expf(mx - nmx);
    ...
    #pragma unroll
    for (int i = 0; i < hd4; i++) {      // each rescale: 32 items of local read+write
        oc[i].x *= corr; oc[i].y *= corr; oc[i].z *= corr; oc[i].w *= corr;
    }
    S *= corr;
    if (kv0 < hi) {                      // in-row 4-byte load walking (scattered)
        const KV* vrow = v + (size_t)kv0 * stride_kv + hk * hd;
        #pragma unroll
        for (int i = 0; i < hd4; i++) {
            float2 a = kv_ld2<KV>(vrow + i * 4);
            float2 b = kv_ld2<KV>(vrow + i * 4 + 2);
            oc[i].x += e0 * a.x; oc[i].y += e0 * a.y;
            oc[i].z += e0 * b.x; oc[i].w += e0 * b.y;
        }
    }
    ...
}
#pragma unroll                            // at the end: cross-lane reduction, 32 items × 4 dims
for (int i = 0; i < hd4; i++) {
    oc[i].x = warp_reduce_sum(oc[i].x); ...
}
```

**After — the R4 kernel: dims to lanes, register accumulator, a 4-row window** (commit
`70f57db`'s added side; in the current tree this body was later refactored by D2 into
`attn_split_1w_body` with the V loads hoisted into the window too — see chapter 66):

```cuda
// Each lane owns 4 consecutive dims (hd % 4 == 0 and hd <= 128 are
// enforced by the dispatch); lanes with d0 >= hd are idle but keep
// participating in the warp reductions (zero contribution).
int d0 = lane_id * 4;
bool live = d0 < hd;
const float4 q4 = live ? *reinterpret_cast<const float4*>(q + h * hd + d0)
                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

float mx = -INFINITY, S = 0.0f;
float4 oc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);   // one register float4

for (int base = lo; base < hi; base += 4) {
    int nr = min(4, hi - base); // warp-uniform
    // Stage K for the whole batch first; the V addresses are already
    // known, so the compiler hoists those loads above the softmax chain.
    float4 k4[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        k4[j] = (live && j < nr)
            ? kv_ld4<KV>(k + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j >= nr) break; // warp-uniform: all lanes exit together
        // Full-row dot: this lane's 4-dim partial, then a warp reduction
        // so every lane holds the row's complete dot (uniform softmax).
        float d = q4.x * k4[j].x + q4.y * k4[j].y
                + q4.z * k4[j].z + q4.w * k4[j].w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFF, d, off);
        float s = d * scale;
        float nmx = fmaxf(mx, s);
        float corr = expf(mx - nmx);
        float e = expf(s - nmx);
        S = S * corr + e;
        mx = nmx;
        if (live) {
            float4 v4 = kv_ld4<KV>(v + (size_t)(base + j) * stride_kv + hk * hd + d0);
            oc.x = oc.x * corr + e * v4.x;   // rescale+accumulate done in one FMA
            oc.y = oc.y * corr + e * v4.y;
            oc.z = oc.z * corr + e * v4.z;
            oc.w = oc.w * corr + e * v4.w;
        }
    }
}
```

**The partial write-out: from "lane 0 writes all" to dim-sliced** (before → after):

```cuda
// after: no cross-lane reduction; lane 0 writes (mx, S), each live lane writes its own 4 dims
float* dst = partial + ((size_t)sp * nh + h) * pstr;
if (lane_id == 0) {
    dst[0] = mx;
    dst[1] = S;
}
if (live) {
    *reinterpret_cast<float4*>(dst + 4 + d0) = oc; // 16B-aligned via pstr
}
```

**The combine kernel (current tree, identical to the R4 version except `8` → `ATTN_SPLITS`
constant-folding)**:

```cuda
__global__ void gqa_attn_split_combine(
    const float* __restrict__ partial, float* __restrict__ o,
    int nh, int hd, int pstr
) {
    int h = blockIdx.y;
    int i = threadIdx.x; // hd threads
    if (i >= hd) return;
    float gmx = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        gmx = fmaxf(gmx, partial[((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        const float* p = partial + ((size_t)sp * nh + h) * pstr;
        float w = expf(p[0] - gmx);
        S += p[1] * w;
        acc += p[4 + i] * w;             // dim-sliced layout: thread i reads dim i
    }
    o[h * hd + i] = (S > 0.0f) ? acc / S : 0.0f;
}
```

**nkv data-side + static grid** (the current tree's launcher, `src/cuda_kernels.cu`):

```cuda
#define ATTN_SPLITS 32
...
gqa_attn_split_partial<__half><<<dim3(ATTN_SPLITS, n_head), 32, 0, stream>>>(...);
```

The grid `(32, n_head)` is as static at replay as it was at capture; nkv is read device-side
from `positions[0] + 1`, each split takes `chunk = ceil(nkv/32)`, `lo = sp·chunk`, `hi =
min(nkv, lo+chunk)` (the first three lines of `attn_split_1w_body`).

### 3.3 Pitfalls

- **The partial layout and combine must change in lockstep**. When the write side switched from
  "lane 0 writes hd4 float4s" to dim-sliced, the combine's read index changed from `o4[i]` (a
  float4 array) to `p[4 + i]` (a scalar index) — changing either side alone is a silent data
  misalignment; the 16 B alignment comment on `pstr` ("16B-aligned via pstr") is the
  precondition for the float4 direct write.
- **Bigger SPLITS is not better**. Under the new geometry, SPLITS=64 measured 44.7 tok/s, worse
  than 32's 45.1 — once each split's row count halves, the fixed costs (the q load, two
  partial round trips, combine's 32-step loop) start eating the gain. 32 is the measured
  stationary point for this shape family, not a theoretically derived value.
- **`break` must be warp-uniform**. The window loop's `if (j >= nr) break` relies on `nr =
  min(4, hi - base)` being identical across the whole warp — exit on a per-lane condition and
  the butterfly's `__shfl_xor_sync` will not line up its participants. The tail rows at the
  end of a batch are deliberately handled in warp-uniform form.
- **Know the numeric-order change**. The dot product went from "each lane serially over 32
  chunks" to "a 4-dim partial + butterfly tree" — the floating-point summation order changed.
  This is not a bitwise change; correctness is secured by the parity sweep (covering
  SPLITS=32's chunk-boundary shapes), not by byte-for-byte comparison.

## 4. Verification

- **The parity sweep extended to SPLITS=32 chunk boundaries**: nkv takes integer multiples of
  the chunk length and their neighborhoods — defends against split-cutting off-by-one errors
  and empty-split handling mistakes (the idle split's `mx=-INF, S=0` path is only truly
  exercised by boundary shapes).
- **The parity comparison**: the attention output's numerical tolerance check against a
  reference implementation (the CPU path) — defends against sum-order errors introduced by the
  geometry rearrangement being waved through as "precision noise".
- **The full suite**: full regression — defends against the hd gate mis-dispatching other shapes
  (hd≠128).
- **Same-binary interleaved A/B**: medians over the @2K and tg128 windows — the kernel-level 2×
  gain must reproduce on the wall clock, and the direction must agree across both context
  lengths.

## 5. Results

| Metric (7B q4_k_m) | before (8d) | after (R4) | llama.cpp same window |
|---|---:|---:|---:|
| split-attention kernel | 148 µs/layer | **79 µs/layer** | — |
| Effective K+V stream rate | 28 GB/s | ~54 GB/s (4.19 MB / 79 µs) | — |
| @2K decode | 39.2 | **43.2–45.1** | 44.9 (gap 14% → ~4%) |
| tg128 decode | 45.1 | **47.5–47.6** | 47.1 (overtaken) |

SPLITS sensitivity (the record's numbers): old kernel 8/16/32/64 → 148/172/419/609 µs (monotonic
degradation); new kernel SPLITS=64 → 44.7 tok/s vs 32 → 45.1 (no further gain; the stationary
point is 32). The two steps R2+R4 together took 7B decode from 42.2/36.7 (tg128/@2K) to
47.5/43.2–45.1 — short context overtakes llama, @2K enters the ~4% gap zone, and the decode-side
chase of llama.cpp was essentially complete here (the later D series extended it to long
context).

## 6. Lessons

1. **A runtime-indexed array = LOCAL memory**: an accumulator like `float4 oc[32]` that "looks
   like registers" goes to local the moment its bound is a runtime value — 512 B per lane,
   re-read and rewritten row by row; an accumulator's dimension must be compile-time foldable
   into registers.
2. **The correct fix for insufficient parallelism is changing the geometry, not adding copies**:
   the monotonic SPLITS-sweep degradation was already warning "the per-warp geometry is
   broken"; raising the split count just copies the disease.
3. **The lane↔dim binding fixes three things at once**: a register accumulator, coalesced row
   access, and a butterfly reduction trading for warp-uniform softmax — the access pattern is
   a function of the mapping; the same math under a new mapping doubled the kernel.
4. **A decode kernel's grid must be a compile-time constant**; anything that changes per step
   (nkv) goes through device-side data (`positions[0]`) plus idle units writing neutral
   values — the structural constraint for every decode kernel in the CUDA Graph replay era.

---
← [09 · R2 MMVQ weight-streaming](./09-r2-mmvq-weight-streaming.md) · [Index](./README.md) · [11 →](./11-p5-gemm-tiles-fa-rewrite.md)
