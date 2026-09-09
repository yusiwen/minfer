# 14 · r9 llama.cpp MMQ reference decode; shape axis closed (MEAS-ONLY)

> **Result**: the reference source (the `mmq-config-ampere.cuh` family) decode
> complete: 256 threads, occupancy 1 (targeted), SRAM tile I=128 × J≤128, ITER_K=256,
> **synchronous staging (no cp.async)**, float/int accumulator `sum[64]`, 16
> `mma.m16n8k32` per k01 sub-iteration — an instruction model of ~0.018
> inst/MAC/thread against our 0.133. All six shape points measured (all
> parity-clean): narrow cp.async KD=8 **481** is the local optimum, and the shape
> axis closes with evidence; the residual lever points at llama's pre-arranged
> mma-fragment layout.
> **Commit**: `84831d2` (wip: sync-wide variant + shape matrix; that variant was later
> overwritten by r12's 16-chain rewrite and is not kept in the current tree).
> **Date**: 2026-09-01.

## 1. Background — where things stood

The MMQ battle state when r8 closed: the raw-byte kernel had pushed R1's 441 to 472,
but that was still 4.9× from the f16 GEMM path's 2318; the wide tile and B-traffic
hypotheses were out; the FA probe was null. The r8 record wrote down this phase's
real predicament: **ncu was unavailable on this device (GB10)**, and "further MMQ
work is hypothesis-cycling" — with no counters, every lever could only be blind-tested
through the parity gate + interleaved A/B, one hypothesis at a time, extremely costly
and unable to localize "which instruction class is slow".

The only reliable high-density information source at hand was **the opponent's own
source code**: llama.cpp's MMQ runs at ~30 TMAC/s on the same chip (GB10) — f16-GEMM
class. Every one of its design decisions — how big a tile, whether staging uses
cp.async, where scales live, how fragments are arranged — is written in
`ggml-cuda/mmq*.cuh`. r9's choice: **stop blind-testing and read the reference source
as a profile**, producing an instruction-level model of "why it is fast", then verify
one separable hypothesis with a controlled experiment (sync-wide).

Meanwhile there was a methodology gap to close: the shape axis had only two measured
points at the time (narrow 64×64, wide 128×64), and "which shape is optimal" and "why
is it fast" are two independent questions. Without sweeping the shape axis first, any
within-point optimization could be working on the wrong peak. r9 did both.

## 2. Principle — the GPU mechanism

This section is r9's decode output (verified at source-line level, later formalized by
the companion analysis `docs/LLAMA-CPP-MMQ-ANALYSIS.md` §1–§6; line numbers below come
from that document's verification system). Layer by layer:

### 2.1 Configuration and tile geometry

q4_K's instance on NVIDIA (Turing+) is `mul_mat_q<GGML_TYPE_Q4_K, 128, false>`, with
the config-table entry (`mmq-config-ampere.cuh:172`; on GB10 the Blackwell table falls
through to the Ampere table):

```
CASE(GGML_TYPE_Q4_K, 256, 1, 128, 128, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_1,
     MMQ_ITER_K, true, false)
→ nthreads=256, occupancy=1, I=128 (od), J=128 (tokens),
  sram_layout=Q8_1, K_vram=MMQ_ITER_K=256, stream_k=true
```

- **256 threads (8 warps), occupancy 1 is "targeted"** — it is not a failure to fit
  more blocks; the design deliberately budgets smem and registers for 1 block/SM,
  trading tile size for per-block resource headroom.
- Tile 128×128: each block covers 128 od rows × 128 tokens; the warp mapping is
  4 od-groups × 2 warps per group, `rows_per_warp = 32`, `ntx = 2`. Each warp covers
  **64 token columns** (J=128 split across a warp pair) and 32 od rows per warp.
- Accumulator: `sum[J*I/(nwarps*warp_size)] = 128·128/256 = sum[64]` (int32) —
  **64 accumulator slots per thread**, i.e. 16 `tile<16,8,int>` C fragments
  (`ne = 4`).
- Barrier cadence: 4 barriers per 256-k iteration (load_tiles → stage first half of
  y → barrier → vec_dot(k00=0) → barrier → stage second half of y → barrier →
  vec_dot(k00=32) → barrier) = 2 per 128-k — the same cadence as our KD=8.

- **stream-k and fixup**: `stream_k=true` in the config means the host launch
  (`launch_mul_mat_q`, mmq.cuh:1393-1473) chooses between an xy-tiled grid and a
  stream-k grid by `nsm` (SM count): when the wave tail doesn't form whole blocks,
  some blocks take an extra tile and write partial sums to global, and a separate
  **fixup pass** (~34 µs) merges the cross-block fp partial sums in order. This is
  the scheduling tax llama pays for "occupancy=1 but big tiles and few waves" — the
  same territory we later probed in r24 (persistent blocks); llama's answer was "pay
  the fixup", our measurement was "there is no wave tail to flatten".

### 2.2 The q8_1 activation pipeline: a once-per-GEMM standalone quantization kernel

Activations (f32) are quantized into `block_q8_1_mmq` by a standalone kernel,
`quantize_mmq_q8_1` (`quantize.cu:458`, inside the host-side `ggml_cuda_mul_mat_q`,
once per GEMM call, not per block), **before the kernel launches**:

- `block_q8_1_mmq` = a 128-element block (QK8_1_MMQ = 4·QK8_1): a leading 16 B scale
  union (`d4[4]` / `ds4[4]` / `d2s6[8]`) + `int8_t qs[128]`, **sizeof = 144 B**. The
  layout comment says it outright: 128 values chunked, transposed, **each block padded
  by 16 B with the pad reused as the block scale and partial sum** ("d/ssum in pad
  bytes"). Q4_K/Q5_K use the DS4 layout: `half2 ds4[4]`, one 16-bit scale + one
  16-bit partial sum per 32 values (d0,s0,d1,s1,…).
- Inside the kernel the activation tile `tile_y` has row stride
  `MMQ_TILE_Y_K = 36 ints = 144 B` — **the smem row layout equals the global
  `block_q8_1_mmq` layout**: `[scale || qs]`, scale read as `(half2*)y`, the qs plane
  at `y+4`.
- Design implication: the activation operator finishes all "dequant preparation"
  (scale and ssum in place) before entering the kernel, so the in-kernel activation
  operand needs **zero dequantization**; `dsB.y` (the partial sum) is a free input
  for the rank-1 correction term.

### 2.3 Weight staging: the raw-nibble smem layout

`load_tiles_q4_K` (`mmq-load-tiles.cuh:703-812`) stages the 144 B raw Q4KB weights
into `tile_x`, **keeping raw nibbles (one 0..15 nibble per byte), unexpanded into
signed int8 and un-centered**:

```cuda
// mmq-load-tiles.cuh:736-737 — the 0x0F mask isolates nibble by nibble, stored per byte
x_qs[i*sram_stride + 16*(txi/8) + txi%8 + 0] = (qs0 >> 0) & 0x0F0F0F0F;
x_qs[i*sram_stride + 16*(txi/8) + txi%8 + 8] = (qs0 >> 4) & 0x0F0F0F0F;
```

- **dmin is not subtracted at staging** — it is folded into the scale:
  `x_dm[...] = (bxi->dm · make_half2(1.0f, -1.0f)) · make_half2(sc8[l], m8[l])`
  (:772-777), storing `(d·sc, −dmin·m)` in a half2. No `__vsubss4`, no I2F chain
  anywhere in staging.
- smem row stride: `sram_stride = 2·MMQ_TILE_NE_K + 2·MMQ_TILE_NE_K/QI8_1 + 4 =
  64 + 8 + 4 = 76 ints (304 B)`; the **"+4" is a 16 B pad that rotates the bank phase
  of adjacent rows** (defending against ldmatrix column conflicts); the nibble plane
  occupies the first 64 ints and the half2 scale plane follows at offset 64.
- **Staging is synchronous**: plain global→(register)→smem stores ordered by
  `__syncthreads`; the classic MMQ path has **no cp.async / TMA pipeline** — latency
  hiding rests entirely on resident warps (at occupancy=1 that is 8 warps × 32 lanes
  of ILP plus intra-tile reuse).

### 2.4 The compute loop and numerics: raw nibbles into mma, dmin as a rank-1 fold

The compute loop is the generic `ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma`
(`mmq-vec-dot.cuh:369-442`):

- `vec_dot` is called once per 32-k chunk and contains **64 mma**: `j0` 8 steps ×
  `k01` 4 steps × `n` 2 steps (`mma.m16n8k32.row.col.s32.s8.s8.s32`, int32
  accumulate). Because the `sum` index doesn't depend on k01, every sum slot is
  accumulated 4×.
- A fragment: `ldmatrix.m8n8.x4` loads straight off the raw-nibble rows
  (`load_ldmatrix(A[n], x_qs + ...·sram_stride + k0, sram_stride)`) — **ldsm consumes
  the nibble bytes directly**, no pre-expansion needed.
- B fragment: plain LDS (source comment verbatim: "**faster than load_ldmatrix**").
- **The fragment-reuse economics** (the most productive ledger entry of the r9
  decode): each vec_dot loads **8 A fragments** (`tile_A A[ntx][4]`, loaded once
  before the j0 loop and reused across the whole loop) and **32 B fragments** (one
  per (j0, k01), reused across the 2 n iterations); the 64 mma consume them. Per MAC:
  llama issues ~0.125 LDSM per IMMA (r36's accounting), and the A-fragment reads
  amortize another 2× over the 32 od rows a warp covers. By contrast our kernel paid
  ~0.5 shared reads per IMMA — and this ratio difference is **a property of tiling**
  (how much a warp covers, how many mma a fragment feeds), not of staging or
  numerics. r9 accordingly put "raise fragment reuse" on the residual-lever list.
- Per-chunk rescale (fp32): `sum[i] += dmA.x·dsB.x·C.x + dmA.y·dsB.y`, where
  `dmA.x = d·sc` (the weight-quantization product), `dmA.y = −dmin·m`,
  `dsB.x = d` (activation scale), `dsB.y = ssum` (activation partial sum). **The dmin
  correction is a rank-1 term on the (token, od-col) plane, applied at accumulate
  time** — that is the cost structure of "mma eats unsigned nibbles, centering
  deferred to rescale": staging saves ALU in exchange for two FMULs per chunk.
- `get_scale_min_k4` semantics (`ggml-quants.c:880-887`): q4_K's 12 B packed scales
  decode into per-32-value-sub-block `(d_scale, dmin)` — six 6-bit codes + six 4-bit
  combination codes; a value `v ∈ [0,15]` dequantizes to `d·s·v − dmin·m`.

**The q6_K specialization — a different scale flow from q4_K** (noted at r9, cashed
in by Era D's q6_K line): q6_K has its own sram layout (stride also 76 ints, different
composition: `64 + 1 + 4 + 7`), its own `load_tiles_q6_K` and
`vec_dot_q6_K_q8_1_mma`. Two key differences: the nibbles are **fully centered at
staging** (`__vsubss4(ql | qh, 0x20202020)`, subtracting 32 per byte into signed
int8 ∈ [−32,31] — there is no dmin term to defer); and the scales are **split into two
levels** (one float `d` per row + one int8 `sc` per 16-value sub-block), so the
rescale is two-stage: `tmp = (C0·scA0 + C1·scA1)·dB` accumulated per j0 step, finally
`sum += tmp·dA`. One MMQ framework, two numeric flows — which explains why q6_K's mma
kernel could not be "casually" derived from q4_K's, and why (r38) it indeed needed a
standalone BT kernel.

### 2.5 The instruction model: 0.018 vs 0.133

Converting the structure above into warp instructions per MAC: llama's MMQ issues
about **0.018 instructions per thread per MAC**, our raw kernel about **0.133** — a
7× gap. That is the origin of their ~30 TMAC/s against our ~6–8 TMAC/s (r9's verdict:
"the ratio, not tile shape, is their speed"). The ratio's composition is worth taking
apart, because it names which instruction classes "support" rather than "compute":

- **The compute core is full-price**: one int8 mma multiply-add per MAC — this part
  is bit-identical between the two (r13/r20 later confirmed with counters that IMMA/
  FFMA bytes were at parity). The gap is never in the mma itself.
- **The gap is entirely in the support stream**: fragment address arithmetic, scale
  smem reads and I2F/FMUL, nibble-unpack shift/mask, barriers and loop overhead.
  llama compresses it with four mechanisms:
  1. **Fragment reuse** — 8 A-frags reused across 64 mma, 32 B-frags each reused 2×,
     pushing per-MAC load instructions to the minimum;
  2. **The raw-nibble plane halves smem bytes** — the expanded-byte form (our then
     qb8 per-k int8) is ~2× the raw form; on small tiles that is the difference
     between 1 and 2 blocks/SM;
  3. **The half2 scale short path** — rescale goes through 16-bit half2 rather than
     a full fp32 I2F→FMUL chain, only a handful of instructions per chunk;
  4. **Tight indexing** — the 32 od rows × 64 token columns warp shape keeps A
     reads, B reads, and address ALU all at amortized lows.
- The costs it **accepts** are part of the model too: mma consuming unsigned nibbles
  (accumulator carries an unsigned dot product), the dmin correction deferred into a
  rank-1 rescale term, stream-k's fixup pass, and the tile-size/ubatch coupling.

The mechanisms sustaining the low ratio were later confirmed item by item by the
r13/r25 ncu censuses (per-GMAC warp instructions 6.06 vs 10.14 M, the delta all
support instructions) — but **the direction was already set at r9**: the optimization
target changed from "find a faster shape" to "push per-MAC support instructions down
and expose the ILP chains".

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Line-level verification, not impressions**: every conclusion is pinned to source
  line numbers (config table :172, `sum` size mmq.cuh:903, vec_dot main loop
  mmq-vec-dot.cuh:414-440, nibble mask mmq-load-tiles.cuh:736-737, scale fold
  :772-777, the `ne` formula mma.cuh:226-227, C-fragment lane mapping
  mma.cuh:245,262). This discipline later crystallized into the companion analysis
  `docs/LLAMA-CPP-MMQ-ANALYSIS.md`.
- **Readings must be branch-aware**: two earlier misreadings were corrected by the
  source — (1) "C fragment `ne = I·J/64` = 2 regs" is the **AMD MFMA branch**'s
  formula (mma.cuh:107-108); NVIDIA Turing+ is `ne = I·J/32` = 4 regs (formally fixed
  at r11); (2) the scale word had been counted as 1 int, but the DS4 layout is really
  4 ints (`half2 ds4[4]`) — which decides that tile_y's row stride is 144 B and not
  smaller, and in turn the bank phase and ldmatrix behavior.
- **Shape-semantics clarification**: the campaign profile's "nt-512" is not a dispatch
  threshold — it is llama.cpp's ubatch size (`-p 2600` → 512-token ubatch → each
  `mul_mat_q` sees 4 J=128 token tiles); J=128 is chosen **coupled to the 512-token
  ubatch** (`mul_mat_q_switch_J` takes the largest J that fits). That explains why
  "copy the tile shape" is not necessarily meaningful for minfer — our call shapes and
  tile-selection constraints differ.
- **Deriving the instruction model**: dividing tile geometry (per-warp mma count,
  fragment loads, scale operations) by per-warp MACs gives the 0.018 vs 0.133 ratio —
  an ncu-independent "upstream quantity of speed" derivable from source alone.

### 3.2 Key code

The decode produced one testable hypothesis: can "staging style (sync vs cp.async) +
footprint (single vs double buffer → occupancy)" alone explain the gap? r9 ported
**synchronous staging** onto our wide kernel as the control: single buffer, 54 KB,
2 blocks/SM, plain loads + one `__syncthreads` per tile (`git show 84831d2 --
src/cuda_kernels.cu`, excerpt):

```cuda
// Single-buffer sync-staged layout — KD=8 totals 54,272B so TWO blocks
// fit per SM (16 resident warps hide the staging latency):
//   qa8   [KDR][128] x 32B  chunk qs only (d/ssum in sda_q)
//   sda_q [KDR][128] x 8B   (d f16 | ssum i16) packed
//   qb8   [64][144]         B super-blocks (64 od-rows per block tile)
//   sds   [KDR][64] f32, sdm likewise
uint8_t* qa8 = mmq_raw_sh;
uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_WBI * 32);
uint8_t* qb8 = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_WBI * 2);
float* sds = reinterpret_cast<float*>(qb8 + 64 * 144);
float* sdm = sds + KDR * 64;
...
#define RAW_STAGE(kt)                                                          \
    do {                                                                       \
        /* llama.cpp-style synchronous staging, single buffer: plain global    \
         * -> smem loads, one syncthreads orders them. At 128x64 tiles the     \
         * smem is small enough for 2 blocks/SM - latency hiding comes from    \
         * occupancy, not prefetch depth. */                                   \
        for (int x = threadIdx.x; x < MMQ_WBI * KDR * 8; x += blockDim.x) {    \
            int u = x % 8, r = (x / 8) % MMQ_WBI, kd = x / (8 * MMQ_WBI);      \
            int tok = i0 + r, c = (kt) * KDR + kd;                             \
            unsigned v = 0;                                                    \
            if (tok < nt && c < nchunk)                                        \
                v = *(const unsigned*)(q8x + ((size_t)tok * nb32 + c) * 40     \
                                       + 4 + u * 4);                           \
            *(unsigned*)(qa8 + ((size_t)kd * MMQ_WBI + r) * 32 + u * 4) = v;   \
        }                                                                      \
        ...
```

(In the double-buffer version, the corresponding `__syncthreads()` comment also
changed from "scales + landed bytes visible" to "single-buffer stage visible to all
warps".) The experiment held the variables: tile shape and the raw-byte mma math
unchanged, only the staging/footprint/occupancy axis swapped — if it came significantly
closer to narrow cp.async, the gap was on the staging axis; if it still lost, the gap
was in the instruction model.

### 3.3 Pitfalls

- **sync-wide still lost** (462–466 vs 481): porting llama's staging philosophy did
  not pay. That is not an experimental failure — it is precisely the hypothesis
  test's negative output: **the gap is not on the staging axis**.
- **The compact variant's serialization trap**: compact in the shape matrix
  (cp.async 128×64, A side compressed to 32 B qs-only chunks, `sda_q` packed scales)
  measured only 410 — the synchronous load of `sda_q` serialized the A-side reads.
  The bytes the compressed layout saved could not buy back the lost load parallelism.
- **Calibrating the "16 mma/chunk" reading**: the r9 record's "16 mma per warp per
  chunk" counts **per k01 sub-iteration** (8 j0 × 2 n = 16); the full vec_dot (32-k)
  is 64 (× the 4 k01 steps). The companion analysis §6 makes this multiple explicit —
  when reading a reference kernel the "chunk" boundary must be pinned first, or the
  instruction model will be off by 4×.
- **The shelf life of wip code**: `84831d2` is a wip commit; the sync-wide variant
  was overwritten at r12 by the 16-chain rewrite and not kept; this doc's code
  excerpts were verified against that commit's diff.

## 4. Verification

- **sync-wide parity green**: `cuda_prefill_mmq_parity` fully passing (defends: the
  single-buffer + plain-load rework moving the wrong bytes — it has none of the raw
  path's double-buffer semantics to lean on).
- **The six-shape matrix all parity-clean** (@2K, `MINFER_MMQ=1`, interleaved within
  one session) (defends: correctness differences contaminating cross-shape
  comparisons; also machine-state drift — all points measured inside one window).
- **narrow cp.async held at 481 as the control**: every new shape re-tests the anchor
  (defends: baseline drift turning "the new shape is worse" into an artifact).

## 5. Results

**The six-shape matrix (GB10 @2K, all parity-clean)**:

| # | Shape | tok/s | Notes |
|---|---|---:|---|
| 1 | narrow cp.async 64×64 KD=8 | **481** | local optimum (r8's raw kernel) |
| 2 | sync wide 128×64 KD=8 | 464 | llama-style sync staging, single buffer 54 KB, 2 blocks/SM |
| 3 | compact cp.async 128×64 KD=8 | 410 | `sda_q` sync load serialization |
| 4 | wide cp.async 128×64 KD=4 | 428 | r8's honest wide-tile value |
| 5 | narrow cp.async 64×64 KD=4 | 427 | |
| 6 | R1 word-stage 64×64 KD=8 | 441 | the starting point |

- **The shape axis closes with evidence**: the six points cover the main axes of
  {narrow, wide, compact} × {KD=4, KD=8} × {cp.async, sync}, and the optimum is the
  starting point itself (481). The B-DRAM-halving theory is dead (L2 absorbs the
  re-reads); wider tiles pay more in staging/sync than they save.
- **The staging philosophy does not port**: llama is still fastest with synchronous
  staging + occupancy=1 because its **per-MAC instruction stream** is 1/7 of ours —
  it doesn't need prefetch to hide latency, its mma density is high enough. Our
  kernel at 0.133 inst/MAC gets patched by every staging scheme for a problem that is
  really an over-dense instruction stream.
- **Adopted (feeding later steps)**:
  1. **The instruction-model lens** — the optimization target switches from
     "shape/staging" to "instructions per MAC and ILP depth". Direct products:
     r12's 16-chain warp tile + ldmatrix (wide-16 KD=4 1020–1058 vs 441–481, ~2.3×)
     and r14's ldmatrix B-fragments (+18.5% / +23–30%).
  2. **"The pre-arranged mma-fragment layout can be materialized at load/quantize
     time"** — the residual lever r9 named. Mechanism: the fragment ldmatrix consumes
     has an exact word→lane arrangement; re-arranging it live inside the kernel, per
     block, costs exactly the address/convert ALU r9 wanted to eliminate. Moving that
     rearrangement to a **cold path** (weight load time, or the activation-quantize
     prepass) does it once, and the hot loop is left with only ldmatrix itself. The
     A-side version was cashed in at r34 (quantize-transpose prepass, +9.72%, the
     implementation comment explicitly cites llama's `quantize_mmq_q8_1` design); the
     B-side "pre-arranged fragments at load time" was cashed in at r12/r14 via
     ldmatrix + the slot-major smem layout.
  3. **The q8_1-style "once-per-GEMM standalone quantize prepass + self-describing
     block layout"** — became the structural prototype of minfer's pad40 prepass
     lineage (r34 → r51/r52).
- **Vetoed/closed**: synchronous staging (464 < 481, no longer a main axis); the
  compact A layout (410); further peak-hunting on the shape axis (the matrix is
  closed; the local optimum is the current global optimum).
- The r10–r11 follow-up control (the reference inner-loop decomposition ported and
  still flat) squeezed the residual further to the three items ILP depth / ldmatrix /
  tile — that is step 15's content; r9's contribution was fixing the search space:
  "instruction-level structure is the suspect class".

## 6. Lessons

1. **Copy the instruction model, not the shape**: the 0.018 vs 0.133 ratio is the
   speed source; transplanting the opponent's tile/staging shape onto a kernel with a
   different instruction stream measures 464 vs 481.
2. **Sweep the axis first, then work within a point**: a six-point shape matrix
   eliminates a whole family of hypotheses at once — far cheaper, and far more honest,
   than five rounds of incremental optimization on one point.
3. **When the profiler is unavailable, the reference source is the highest-density
   source of fact**: a line-verified instruction model can replace counters in
   localizing the search space (and was later confirmed by the ncu census).
4. **Pin the counting boundaries before reading a kernel** (the chunk/k01/vec_dot
   multiples, the NVIDIA vs AMD `ne` formula branches, the scale word's int count) —
   off by one multiple in the reference reading, off by an order of magnitude in the
   conclusion.

---

← 13 · [Index](./README.md) · 15 →
