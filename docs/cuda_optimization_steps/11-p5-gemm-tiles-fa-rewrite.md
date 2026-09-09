# 11 · P5 — prefill gap session: TM=128 big tiles + FA rewrite (LANDED, with three REVERTED probes)

> **Result**: 7B @2K prefill **1435 → 2340–2370 tok/s** (net +64%); the gap to llama.cpp (3401 @2K) narrows **2.37× → 1.43×**.
> **Commit**: `86ca78c` (P5·0) → `d713e6e` (P5·1) → `725e307` (P5·2) → `fc07c04` (P5·3) landed; `1365c82` (KS=64), `a189837` (TM=256), `b254c22` (AF32) — three negative results reverted. **Date**: 2026-09-01 (single-day session).
> **Record note**: the session-range start `b8568cd` cited by the P5 chapter does not resolve to any commit (explicitly stated in §0 footnote 3); all 7 hashes listed above were verified reachable in practice, and the code excerpts key off them.

## 1. Background — where things stood

After 8p (resident f16 weight cache + dequant folded into the GEMM), 7B @2K prefill
sat at ~1435 tok/s while llama.cpp measured 3401 on the same machine — a **2.37×
gap**. Decode had already approached parity through 8e and R2; prefill had become
the most glaring shortcoming across the entire product line.

R1 (int8 MMQ, doc 08), started on August 31, had proven the quantized route could
reach parity but was still slow at the time (441 tok/s, and not yet profiled); 8m's
wmma f16 GEMM was the serving prefill engine. P5's choice: **break the f16 route
wide open first** — it was both the fastest path available today and the baseline
the coming MMQ campaign would measure itself against. By attribution, the prefill
wall was made of three pieces: GEMM (the biggest), FA-style tiled attention, and
elementwise/convert odds and ends. The session fired six probes in a single day,
each gated by the full suite plus interleaved A/B on the same binary (full-output
diff, never prompt-echo grep), under the rule "whatever clears the bar lands;
negative results revert on the spot with the mechanism preserved."

Where things stall without this step: 2.37× is not a single bug — it is the
superposition of three structural wastes: tile geometry (B-panel re-reads),
tensor-core coverage (FA's P·V was still on the scalar path), and single-element
elementwise kernels. Fixing any one of them alone does not reach parity, so the
day was essentially about turning over every block of the prefill wall and
measuring it.

## 2. Principle — the GPU mechanism

### 2.1 Why a larger N-tile raises tensor-core utilization

The 8m GEMM's output tile is `TN × TM = 64 × TM` (TN=64 nt rows, TM od columns).
Every k-step must move the B panel (a TM-row × KS-k-element weight slice) from L2
into smem, and that B is then reused by all 64 A rows inside the tile. **Going
TM 64 → 128 means the same B bytes serve twice the mma work**:

- **B-panel re-reads per FLOP halve**. Each weight-matrix row is read once per
  k-step and produces 2× the FLOP → the L2 bandwidth needed to sustain the same
  TFLOPS halves;
- **Barriers per FLOP halve**. Double-buffered staging takes one `__syncthreads`
  per k-step; with the tile doubled, each sync amortizes over 2× the mma;
- **Accumulator fragments per warp go 2 → 4** (`fc[2][ODC]`, `ODC = TM/64`): the
  fragment-load-to-mma ratio gets healthier and tensor-core issue between sync
  points goes deeper.

The grid organization amplifies this: `blockIdx.y` is the od tile, so
**neighboring blocks share the same od-tile's B panel** (code comment: 64 rows ×
id f16 ≈ 0.5 MB) — one weight panel in L2 feeds several blocks, and the f16 weight
matrix streams from DRAM roughly once. Kernel time 455 → 302 ms (−34%) is the
direct reading of that mechanism.

Running the ledger for one k-step (KS=32, f16 = 2 B/element):

```
staged into smem: A tile   64 × 32 × 2 =  4 KB
                  B panel  TM × 32 × 2 =  4 KB (TM=64) / 8 KB (TM=128)
FLOP produced:    64 × TM × 32 × 2   = 262K (TM=64) / 524K (TM=128)
smem bytes / KFLOP: ≈ 30.7 (TM=64) → ≈ 23.0 (TM=128)
```

The bytes on both the A and B sides are amortized over 2× the output elements —
per-FLOP smem traffic drops ~25%, barriers per FLOP halve; and the od-tile count
halving means the B panel's number of passes over DRAM halves too. The tensor
core itself did not get faster; what changed is **the fraction of time spent
feeding it**.

### 2.2 What the FA rewrite changed

The FA prefill attention is the online-softmax tiled kernel landed by 8n
(`fa_prefill_f16kv`, Q tile 64 rows × KV tile FA_TKV columns). P5 took two cuts
at it:

1. **P5·0 — put P·V on tensor cores**. QKᵀ was already wmma, but the product of
   the score matrix P with V still went through scalar FMAs — half the FA
   kernel's arithmetic had no tensor core. Changed to `wmma::mma_sync`: P
   (f16, matrix_a) × V (matrix_b, row_major) accumulating into 16×16 f32
   accumulators, 8 accumulator fragments spread over hd=128 per 16-row block.
   **10.06 → 4.24 ms/layer — the FA channel halved**.
2. **P5·3 — parallelize softmax + pad smem rows**. (a) The online softmax
   originally ran as "one 64-row-deep serial chain per warp" pressing on only 2
   of the 8 warps; changed to warp-per-row (8 warps × 8 rows, shuffle
   reduction, −INF seeding) — the serial chain became 8-way parallel; (b) at
   hd=128 the smem row width is exactly 256 B ≡ 0 (mod 32 banks), so **all 8
   rows of every `ldmatrix` land in the same bank group → 8-way conflict**;
   spacing rows +8 halves (272 B) shifts each row by 4 banks. The two cuts
   together took 4.25 → 1.92 ms/layer.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **TM/KS/AF32 all template-parameterized** (`gemm_f16_nt_kernel_t<TM, KS, AF32>`):
  tile size is the "depth vs occupancy" trade-off axis (the three negative
  results in §3.3 are all views of it), and parameterization makes every probe a
  compile-time specialization rather than a runtime branch — a negative result
  only needs the launch selector changed; TM=64 keeps a `MINFER_GEMM_TM=64`
  opt-out.
- **FA's P matrix enters P·V directly as fragments**: at P5·0 P landed in smem
  and was ldmatrix'd from there; later r48 (doc 51) moved softmax into registers
  and converts P in-register into the matrix_a fragment — the current-tree code
  excerpted here is the post-evolution shape, but the tensor-core P·V structure
  itself has not changed since P5·0.
- **Padded stride is the constant `sstr = hd + 8`**: the 272 B row spacing that
  fixes the bank conflict is independent of tile size, so it is hard-coded as a
  kernel constant (still in the current tree).

### 3.2 Key code

**The TM-template GEMM: tile organization and the L2-sharing comment** (comment
above + opening of `gemm_f16_nt_kernel_t` in the current tree):

```cuda
// C[nt, od] = A[nt, id] · B[od, id]^T. 64 x TM output tiles (TM = 64
// baseline, 128 halves the B-panel re-reads through L2 and the per-k-step
// barrier count), k-step 32, double-buffered shared staging, 8 warps (each
// owns 32 nt rows x TM/4 od cols as 2 x TM/64 f32 fragment pairs). f32
// accumulation.
template <int TM, int KS, bool AF32 = false>
__global__ void gemm_f16_nt_kernel_t(...) {
    constexpr int TN = 64;
    constexpr int ODC = TM / 64;  // od 16-col fragments per warp row-half
    ...
    // blockIdx.x = nt tile, blockIdx.y = od tile: consecutive blocks share
    // the same od-tile's B panel (64 rows x id f16, ~0.5MB) in L2, so the
    // f16 weight matrix streams from DRAM ~once instead of nt/64 times.
    int m0 = blockIdx.y * TM;
    int n0 = blockIdx.x * TN;
```

**The inner loop: the 4 mma of fa[4]×fb[2], plus the lesson comment about the
fb[1] offset** (current tree, introduced by `725e307`):

```cuda
// fa[n-half][k-half]; fb[k-half] per od chunk. Both k halves of each
// 32-slice must accumulate (the v1 bug: only the first 16 k's were
// multiplied); fb's k offset is +16 ELEMENTS (one k-half), not +16
// rows.
#pragma unroll
for (int kh = 0; kh < KHC; kh++) {
    wmma::load_matrix_sync(fa[0], &As[buf * TN * KS + wn * 32 * KS + kh * 32], KS);
    wmma::load_matrix_sync(fa[1], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32], KS);
    wmma::load_matrix_sync(fa[2], &As[buf * TN * KS + wn * 32 * KS + kh * 32 + 16], KS);
    wmma::load_matrix_sync(fa[3], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32 + 16], KS);
    #pragma unroll
    for (int oc = 0; oc < ODC; oc++) {
        wmma::load_matrix_sync(fb[0], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32], KS);
        wmma::load_matrix_sync(fb[1], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32 + 16], KS);
        wmma::mma_sync(fc[0][oc], fa[0], fb[0], fc[0][oc]);
        wmma::mma_sync(fc[1][oc], fa[1], fb[0], fc[1][oc]);
        wmma::mma_sync(fc[0][oc], fa[2], fb[1], fc[0][oc]);
        wmma::mma_sync(fc[1][oc], fa[3], fb[1], fc[1][oc]);
    }
}
```

At TM=128, `ODC=2`: between consecutive barriers each of the 8 warps issues
2×2×2 = 8 mma (4 at TM=64) — the sync-overhead-to-tensor-core-work ratio halves
directly.

**FA's padded row stride: the constant behind the bank-conflict fix** (opening of
`fa_prefill_f16kv` in the current tree, introduced by `fc07c04` and in use ever
since):

```cuda
extern __shared__ __align__(256) uint8_t smem[];
// Padded smem row stride: hd=128 halves = 256B ≡ 0 mod 32 banks makes
// every wmma ldmatrix row land on the same bank group (8-way conflict
// per load). +8 halves (272B) shifts each row by 4 banks.
const int sstr = hd + 8;
__half* Qs = reinterpret_cast<__half*>(smem);
__half* Ks = Qs + FA_TQ * sstr;
__half* Vs = Ks + FA_TKV * sstr;
```

**FA's tensor-core P·V** (current-tree shape; the P5·0 prototype went through a
smem relay, r48 moved it to in-register construction):

```cuda
// Build the P@V A-operand (f16 matrix_a) from the scaled fragments IN
// PLACE. matrix_a m16n16k16 row_major and the f32 accumulator use the
// SAME (row,col) layout, so pa.x[i] == fc[cc].x[i] element-wise. Ks in
// QK^T is col_major; V in P@V is row_major (both validated standalone).
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa[FA_TKV / 16];
...
// acc = acc*alpha + P · V. V (B) is row_major from Vs.
#pragma unroll
for (int kk0 = 0; kk0 < FA_TKV; kk0 += 16) {
    #pragma unroll
    for (int ob = 0; ob < 8; ob++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
        wmma::load_matrix_sync(vb, &Vs[kk0 * sstr + ob * 16], sstr);
        wmma::mma_sync(acc[ob], pa[kk0 / 16], vb, acc[ob]);
    }
}
```

**P5·1's two elementwise kernels (the current tree is already the vectorized
form)** — `store_kv_f16` does 4 dims per lane (one `float4` load + two `half2`
stores, scalar fallback for the tail rows):

```cuda
int t = blockIdx.x;
int j = (blockIdx.y * blockDim.x + threadIdx.x) * 4;   // 4 dims per lane
if (t >= nt || j >= nkt) return;
int p = positions[t];
if (j + 3 < nkt) {
    float4 v = *reinterpret_cast<const float4*>(src + (size_t)t * nkt + j);
    __half2* d = reinterpret_cast<__half2*>(dst + (size_t)p * nkt + j);
    d[0] = __floats2half2_rn(v.x, v.y);
    d[1] = __floats2half2_rn(v.z, v.w);
} else { /* tail: scalar loop */ }
```

`convert_f32_f16_kernel` does 8 elements per lane (the kernel comment carries its
own ledger; the "P1" label in the comment is a stale historical numbering — the
P5 record itself also notes the "8p" label collision, see the §5 table).

```cuda
// P1: 8 elements per thread (2x float4 -> 4x half2) instead of one
// scalar element — 8x fewer transactions on the same traffic.
long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
if (base + 7 < n) {
    float4 a = *reinterpret_cast<const float4*>(x + base);
    float4 b = *reinterpret_cast<const float4*>(x + base + 4);
    __half2* o = reinterpret_cast<__half2*>(out + base);
    o[0] = __floats2half2_rn(a.x, a.y); ...
}
```

The single-element kernels' waste is purely at the transaction level: an f32
element is 4 B, so a 32 B sector uses 1/8 of itself; vectorization divides the
transaction count over the same traffic by 4–8 — together these two kernels only
lifted +4%, because they were never at the wall's center of gravity.

**The survival note for the AF32 mirror mechanism** (current tree, `b254c22`'s
mechanism kept along with its own cause of death):

```cuda
// AF32 A staging, mirror scheme: cp.async the F32 k-tile into a smem
// mirror (16B = 4 f32 chunks; async again — the v1 synchronous global
// loads stalled every k-tile and measured -8%), then convert
// smem->smem f32->f16 right before compute. Requires id % 8 == 0.
```

### 3.3 Pitfalls

- **The `fb[1]` offset: +16 elements, not +16 rows**. The first version after
  widening TM fetched the second k-half's B fragment as "+16 rows" — the GEMM
  result was flat-out wrong. B-fragment slices in `ldmatrix`/wmma advance along
  the **k (column)** direction, so `fb[1]`'s offset is k-half = +16 elements;
  `cuda_prefill_f16_gemm_parity` caught it on the spot. The comment still sits
  above the inner loop today (see the excerpt above).
- **Silent fallback on smem overflow**. P5·3's double-buffered working set
  exceeded the 99 KB/block cap, `cudaFuncSetAttribute` failed → the kernel
  **silently** fell back to the legacy path and the whole machine dropped to
  313 tok/s — a performance collapse hidden inside "no error reported". Fix:
  the fallback prints a warning; the padded layout shipped single-buffered
  (69 KB). "Resource limits must fail loudly" became a campaign rule from then
  on (same class as the r8 phantom and r58's fake OOM).
- **`__syncthreads` does not order cp.async**. When the double buffer was cut,
  cp.async's `commit_group`/`wait_group 0` were removed along with it, leaving
  a bare `__syncthreads` to wait on staging — async copies are not constrained
  by the barrier, and parity came out at 0.28. Lesson: `__syncthreads` only
  orders ordinary memory accesses; cp.async must commit/wait group.
- **KS=64 (P5·4, REVERTED)**: the idea of halving per-FLOP barriers again was
  beaten back by occupancy — TM=128+KS=64 needs 56 KB smem (current-tree
  comment verbatim), resident blocks halved, 1464 vs 2345 tok/s (**−38%**).
  The mechanism is kept (`MINFER_GEMM_K64=1` can re-measure it), KS=32 stays
  default.
- **TM=256 (P5·neg, REVERTED)**: a wider tile hit the same wall, −3%; the
  session went ahead and parameterized the thread count anyway (NW: TM≤128 →
  8 warps, TM=256 → 16 warps), and that part of the mechanism survives in the
  current tree.
- **In-kernel f32→f16 A staging (AF32, REVERTED)**: the goal was folding the
  separate convert pass into GEMM staging; end-to-end −8%, and while the
  standalone kernel version had proven correct, the integrated version's parity
  never closed — abandoned (WIP chain `a3b0dcd`/`69c3933`/`aa40ed3`).
  r23 later measured the convert pass at only 6% of the f16 wall — **the upper
  bound never supported this direction**; the lesson "measure the upper bound
  before building" was booked here.

## 4. Verification

- **`cuda_prefill_f16_gemm_parity`**: numeric parity of GEMM output against the
  reference implementation — specifically defends against tile-reshuffle /
  wmma-slicing bugs (it caught the fb[1] bug).
- **Full suite every step + interleaved A/B on the same binary (full-output
  diff)**: defends against "changed A, broke B" and window-drift noise; the P5
  chapter states outright "never prompt-echo grep".
- **Parity check on cp.async ordering**: the 0.28 output difference is the
  direct signal of the removed wait_group — here the parity gate defends
  asynchronous memory ordering, not math.
- **Explicit warning on the smem cap**: after the fix, the fallback path prints
  the actual smem requirement and the failure reason — prevents a repeat of the
  "performance silently collapsed" scenario.

## 5. Results

| Step | Commit | Metric | before → after | Verdict |
|---|---|---|---|---|
| P5·0 FA P·V on wmma | `86ca78c` | FA ms/layer; @2K prefill | 10.06 → **4.24**; +15% | 🟢 |
| P5·1 elementwise vectorization | `d713e6e` | @2K prefill | 1435 → **1493** (+4%) | 🟢 |
| P5·2 TM=128 big tile | `725e307` | @2K prefill; GEMM kernel | 1493 → **2267** (+30%); 455 → 302 ms | 🟢 |
| P5·3 all-warp softmax + padded rows | `fc07c04` | @2K prefill; FA kernel | 2267 → **2365–2371** (+4–5%); 4.25 → 1.92 ms/layer | 🟢 |
| P5·4 KS=64 | `1365c82` | @2K prefill | 1464 vs 2345 (−38%) | 🔴 reverted |
| P5·neg TM=256 | `a189837` | @2K prefill | −3% | 🔴 reverted |
| P5·neg AF32 in-kernel convert | `b254c22` | @2K prefill | −8%, parity hole never closed | 🔴 reverted |

Net effect: **1435 → 2340–2370 tok/s (+64%), vs llama 2.37× → 1.43×**.
Comparison basis: llama.cpp 3401 @2K (ca3d5a3e1 bench build). After P5·3,
`fa_prefill_f16kv` runs 1.92 ms/layer vs llama's 0.79 — FA is still part of the
residual gap, left to the later FAP line (docs 49/51).

Post-P5 nsys budget (per 2K prefill): GEMM ~597 ms (effective ~46 TFLOPS;
llama 455 ms), FA ~54 ms, convert ~56 ms, swiglu ~51 ms — the 1.3× kernel gap
to llama on GEMM plus the misc items together make up the remaining 1.43×.
That budget table then became Era C's (the MMQ campaign's) starting coordinates.

## 6. Lessons

1. **Silent fallbacks are correctness-grade hazards**: resource limits (smem
   caps etc.) must fail loudly and print the actual value — "performance
   suddenly collapsed but nothing errored" always means the fallback path is
   running.
2. **`__syncthreads` does not order cp.async**: ordering of async copies can
   only be established by `commit_group`/`wait_group`; the barrier is
   transparent to them.
3. **Tile widening's benefit boundary sits at "smem doubles → resident blocks
   halve"**: TM 64→128 gave +30%, KS 32→64 gave −38% — the depth-vs-occupancy
   mutual exclusion is a trade-off this campaign rediscovers repeatedly
   (r38/r39/r40 learned it again on MMQ).
4. **Measure the upper bound before writing the kernel**: AF32's −8% and r23's
   6% upper bound show that measuring the target pass's wall-clock share before
   building can veto an entire route up front.

---
← 10 · [Index](./README.md) · 12 →
