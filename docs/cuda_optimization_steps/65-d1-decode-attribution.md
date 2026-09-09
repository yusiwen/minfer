# 65 · D1 decode attribution: split-attention staging depth is the only wall that grows with KV (measurement round, CLOSED)

> **Result**: of 7B q4_k_m decode @1641 KV's 0.91 ms/step wall-clock increment
> vs tg128, **100%** comes from the single kernel `gqa_attn_split_partial`
> (1.98 → 34.1 µs/launch); ncu proves it is **memory-LATENCY-bound** (76.5%
> long_scoreboard), not byte bandwidth and not insufficient parallelism —
> 32 splits/head already supply ample parallelism. The ATTN_SPLITS sweep
> measured a dead end (and is not bitwise-safe); staging-depth-class changes
> were probe-verified bit-identical — this "free-knob list" directly
> authorized D2.
> **Commit**: no repo change (measurement only; artifacts in `/tmp/d1/`,
> ephemeral, key numbers inlined here and in
> [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) §2D). **Date**:
> 2026-09-07.

## 1. Background — where things stood

On 2026-09-07, r60 had just crowned the whole MMQ prefill line: 7B pp3314
~3581 tok/s = **1.080×** vs llama.cpp; the prefill campaign had converged. One
phase of the CUDA campaign remained unconquered: **decode**. The decode state at
the time (master-table D1 row, 7B q4_k_m / DGX Spark GB10):

- **tg128** (KV~1, i.e. the average KV depth across a 128-token generation):
  49.3 tok/s vs llama 49.41 — already at parity;
- **@1641 KV** (generation continuing after a 1641-token prompt): 47.2 tok/s vs
  the llama 49.41-class — **−4.5%** (0.956×).

That combination itself carries attribution information: tg128 flat while @1641
slow means the gap carrier is a computation that **grows with KV depth** — in
decode's kernel list, the only such things are the KV-cache readers (attention)
and KV-dependent elementwise ops. But "it is attention" is not "where attention
is slow": it could be byte bandwidth, parallelism, or dependency-chain latency —
three diagnoses mapping to three entirely different levers (more bandwidth / more
parallelism / restructure staging). D1 set out to separate the three without
changing one line of repo code.

Third, **gate economics**: decode is where bitwise gates are most sensitive (attention's
float summation order reacts to any structural change); writing code before knowing
"which changes are free" most likely lands in tolerance-gate territory, dragging the
session into a parity swamp. Classify the freedoms first, then pick a free one — D1's
methodological bet, later cashed by D2.

## 2. Principle — the GPU mechanism

### 2.1 The kernel under attribution: the shape of decode split-attention

`Op::Attn`'s nt==1 path dispatches to `gqa_attn_split_partial<KV>` (current
tree `src/cuda_kernels.cu`, the 1-warp body in `attn_split_1w_body`):

```cuda
#define ATTN_SPLITS 32

// each block = 1 warp = (1 split, 1 q head);
// lane l owns 4 consecutive dims (hd=128 → 32 lanes × 4 = 128)
int chunk = (nkv + SPLITS - 1) / SPLITS;   // @1641 → 52 rows/split
int lo = sp * chunk;
int hi = min(nkv, lo + chunk);

float4 q4 = /* this lane's 4 q dims */;
float mx = -INFINITY, S = 0.0f;
float4 oc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);   // R4-rewrite product: oc lives in registers

for (int base = lo; base < hi; base += 4) {        // 4 rows per batch
    int nr = min(4, hi - base);                    // warp-uniform
    /* ...per row: 4-dim dot → 5-step shfl butterfly reduction → expf ×2
       → online-softmax update of (mx, S, oc) ... */
}
```

The grid is `(ATTN_SPLITS=32, n_head=28) = 896 single-warp blocks` (7B: 28 q heads,
4 KV heads, GQA 7:1); each block serially online-softmax-scans its split's 52 rows
(@1641), 4 rows per batch. This shape is the product of the R4 dimension-parallel
rewrite (commit `70f57db`) — it replaced the earlier llama-style `LOCAL float4 oc[32]`
accumulator (~80 MB local-memory traffic per layer) with one register `float4` per
lane, lifting @2K decode from 39.2 to 43.2–45.1 tok/s at the time.

### 2.2 Three candidate bottlenecks, three counter criteria

Decode @1641 attention reads K+V once each per layer:

```
per-layer KV bytes = 2 (K+V) × 1641 rows × 512 kv-dim (4 KV heads × hd 128) × 2 B (f16)
                   ≈ 3.36 MB; × 28 layers = 94 MB/step
```

Each candidate bottleneck has its own counter criteria:

| Candidate | Criteria | If it holds, the lever is |
|---|---|---|
| **Byte bandwidth** (byte roofline) | DRAM/L2 byte traffic at the roofline, high SM occupancy | compress KV (f16 KV already done), reduce re-reads |
| **Insufficient parallelism** | waves << 1, SMs mostly idle | increase splits/block count |
| **Memory latency chain** (latency roofline) | bytes × time far below the roofline, but stalls concentrated in long_scoreboard | rework load scheduling (staging) — **no byte change, no parallelism change** |

The criteria's vehicle is ncu's warp-stall sampling and occupancy counters
(the r43-established rule: **attribute a stall to the consumer instruction waiting
on it**). The result of this step was the third of the three, with a precise
quantitative decomposition:

```
in-situ 34.1 µs/launch ≈ 12 µs byte time (@273 GB/s) + ~22 µs exposed latency
```

That is: **two-thirds of the time is neither moving bytes nor computing — it is waiting for loads to return**.

### 2.3 Why a latency chain: a 4-row window buys only 4 rows of load-level parallelism

The mechanism's root is the loop structure (the shape at D1 time): K rows are
staged at batch start, but **each V row's load issues inside the dependency
chain** — the old code comment claimed "V's address is known, the compiler will
hoist the inline V load above the softmax chain", an assertion **falsified by
measurement** (the falsification process is doc 66, which is exactly where D2's
entire gain comes from). In reality each V load queues behind the
`shfl`/`expf` chain, issue point = consume point, so:

```
52 rows serial × ~1 exposed memory latency per row ≈ 22 µs of waiting
4 rows per batch → load-level parallelism capped at 4 rows
```

Reference frame: llama.cpp's decode attention uses the same flash-decoding
skeleton, but consumes a **256-row × 128-dim window with an 8-warp block** —
its latency hiding comes from per-block load depth (dozens of loads in flight
per block), not our per-row chained structure. That previewed two levers of
different magnitude: the small lever = hoisting loads inside the existing
1-warp structure (D2, bitwise-free); the big lever = changing the block/work
mapping (D3a/tolerance-class, doc 68).

### 2.4 Why not parallelism: the ATTN_SPLITS sweep

Intuitively "32 splits not enough → go to 64/128" is the cheapest parallelism lever. Measured dead end, and instructively so:

1. **partial time flat**: split-kernel time unchanged across 32/64/128 —
   the latency chain is per-row; adding blocks does not shorten it;
2. **combine cost 2–3×**: doubling the split count doubles the combine
   reduction's branch count;
3. **not bitwise-safe**: any split-count change reorders float summation
   (outputs differ with ndiff ≈ 3.6e-3, max\|Δ\| ~3e-9 — the r50/r57 class,
   not byte-identical).

Item 3 is D1's most important taxonomic output: it cuts decode attention's change
space into two classes —

- **staging-depth class** (same split ranges, same row order, same per-row ops;
  only load scheduling moves): probe-verified **bit-identical (ndiff=0)** →
  **free knobs**;
- **split/block structure class** (split count, window shape, warp count):
  necessarily reorders summation → must go through tolerance gates (D3a's
  calibration package, doc 68).

D2 was about exhausting the first class's freedoms; the second class did not reopen
until D3a, in the form of a "calibrated tolerance package".

## 3. Implementation (the measurement method)

No code changes in this step; §3 records how the three measurement instruments
were built and their respective pitfalls.

### 3.1 nsys per-kernel census: taking the decode step apart

The attribution's core is an **nsys per-kernel census**: on 7B, take two KV
anchors (tg128 at KV~1 and @1641 post-prompt decode), sample a stretch of
steady decode steps at each, align by kernel name, and compare per launch.
Design points:

- **same-window anchoring**: the two anchors' measurements were interleaved
  within one machine-state window (r59b's lesson — absolute values across
  windows are not comparable, only in-window differences count);
- **the alignment unit is the launch, not the op**: in a decode step the same
  kernel fires once per layer; align, build a "per-kernel per-step total time"
  table, then difference the two anchors;
- **additivity check**: the difference table must add up to the wall-clock
  difference. Measured `gqa_attn_split_partial` 1.98 → 34.1 µs/launch
  (KV 1.64 → 1641), × 28 layers ≈ **+0.90 ms/step, and the two anchors'
  wall-clock difference is 0.91 ms/step** — a single kernel explains 100% of
  the increment; every other kernel (all MMVQ, rms, quantize, combine) is flat.
  The attribution closes, leaving no room for "something else hides elsewhere".

### 3.2 The NCUE probe methodology: nix binaries cannot be ncu-sampled directly → verbatim standalone probe

D1's second instrument is ncu counters (warp-stall sampling, occupancy,
registers), and this GB10 has an unavoidable toolchain reality:

- **a nix-built release binary cannot be directly attached and sampled by ncu**.
  One side is device-counter permission (`ERR_NVGPUCTRPERM`, recorded since the
  R1 chapter): ncu must go through the `sudo -n env LD_LIBRARY_PATH=...` protocol
  (methodology doc no. 77 §2.5; plain sudo strips env vars and ncu silently
  profiles the legacy path — the r56 lesson); the other: the engine runs on
  CUDA-graph capture/replay, and ncu's serialized replay disturbs the wall clock
  and makes it hard to anchor "which logical node is the Nth launch in the replay".

The solution is a **verbatim standalone probe**: extract the
`gqa_attn_split_partial` kernel body unchanged into a standalone `.cu`, compile
with nvcc directly (`-O3 -arch=sm_121a`), reproduce the launch in the probe with
**minfer's real flags/grid/arguments**, and attach ncu to the probe process.
Three disciplines:

1. **verbatim**: the probe kernel is line-identical to the repo kernel — only
   then do probe counter conclusions qualify for extrapolation to in-situ (the
   later D3a/D3-6 probes all followed this);
2. **same flags**: grid, `ATTN_SPLITS`, hd, scale, partial layout all take the
   engine's real values;
3. **nsys owns wall clock, ncu owns structure**: ncu serializes replay, so per-kernel
   times are distorted; use it only for structural readings (occupancy/sectors/stalls);
   all time conclusions come from in-situ nsys (doc no. 77 §2.5's division of labor).

The probe offered two switchable memory modes, a choice that later proved decisive:

- **hot-L2**: small KV run repeatedly, KV resident in L2 — short latency, benefits
  compressed;
- **cold-DRAM**: 28 layers of weights/KV rotating, forced to start from DRAM —
  real decode's cache state.

During D1 the probe incidentally measured a staging variant: hot-L2 mode showed
only **−11%**, a signal that looked "worth doing but not dazzling"; re-measured in
D2 under cold-DRAM, **−42%**. Mode choice underestimated the gain 4× — decode's
KV reads are cold-DRAM-shaped at real scale.

### 3.3 Pitfalls

- **ncu's stall-attribution direction**: PC-sampling books a stall under
  **the consumer instruction waiting on it**, not the producing load (the
  r20/r43 rule). Reading D1's 76.5% long_scoreboard, the attribution target is
  the chain's consumer (expf/oc update), but mechanically you must reason back
  to "the load that was not issued early".
- **Do not over-read wave counts**: 0.78 waves is evidence that "the SM array
  is not full", but the kernel is 896 single-warp blocks; not fitting in one
  wave does not mean parallelism is the bottleneck — the stall distribution
  proves the bottleneck is each warp's own chain.
- **The probe-vs-in-situ gap must have a name**: the gap between probe
  (hot-L2) −11% and the later in-situ nsys −43% decomposes into "cache
  temperature + neighboring kernels' L2 interference" — do not use probe
  absolute times as wall-clock predictions (this calculation became D2's
  three-way evidence chain: probe −42% / nsys −43% / wall clock +2.0%, doc 66).

## 4. Verification (measurement validity)

A measurement round's "gates" are not bitwise/greedy but the credibility of the measurement itself. D1 used four:

- **additivity gate**: the per-kernel difference sum (0.90 ms/step) matches the
  wall-clock difference (0.91 ms/step) — defends against missing or
  double-counted attribution;
- **same-window gate**: all anchors measured interleaved within one machine-state
  window — defends against reading co-tenant drift as a KV effect (r59b-class);
- **verbatim gate**: the probe kernel is line-identical to the repo kernel with
  the same flags — defends against invalid extrapolation of probe conclusions;
- **bitwise classification gate**: the classification of "which changes are
  free" was itself probe-verified — staging-depth class ndiff=0 (verified on
  nkv ∈ {1, 29, 52, 512, 1641}, later extended by D2 to 45/45), split-count
  class ndiff ≈ 3.6e-3 — defends against treating a non-free change as free and
  starting work.

## 5. Results

**Status: MEASURED (measurement only, no repo changes).** The deliverables are three tables:

**① Wall-clock attribution** (7B q4_k_m, GB10, same-window interleaved):

| Anchor | minfer | llama.cpp | Delta |
|---|---|---|---|
| tg128 | 49.3 tok/s | 49.41 | parity |
| @1641 KV | 47.2 tok/s | 49.41-class | **−4.5% (0.956×)** |

100% of the 0.91 ms/step increment comes from `gqa_attn_split_partial` (1.98 →
34.1 µs/launch, the only kernel that grows with KV; combine at 96.0 µs/step etc. are all flat).

**② Kernel diagnosis** (ncu, verbatim standalone probe, minfer flags):
76.5% long_scoreboard stall; all other pipes ≤ 12%; 40 regs; 0.78 waves. Byte
decomposition: 34.1 µs ≈ 12 µs bytes + ~22 µs exposed latency → **memory-LATENCY-bound**.
Consistency check: 12 µs @273 GB/s ≈ 3.3 MB = exactly one layer's 1641 × 512-dim
K+V bytes; moving the full 94 MB of KV per step takes only ~0.34 ms, while attention
spends ~0.95 ms per step — not a byte bottleneck, and not parallelism (32 splits
already give 896 blocks).

**③ Free-knob classification** (probe-verified):

| Change class | bitwise | Verdict |
|---|---|---|
| staging-depth (load scheduling, same ranges/order/ops) | **bit-identical (ndiff=0)** | free → D2 acts |
| ATTN_SPLITS / split-count | ndiff ≈ 3.6e-3, max\|Δ\|~3e-9 | dead end + needs tolerance → sweep measured flat, closed |
| block/work remapping (llama-vec-style multi-warp) | necessarily reorders | left for D3a (tolerance gate, doc 68) |

**Residual micro-lever registry** (targets for later sessions, numbers archived):
`f32_bits_to_i32` (positions repeatedly converted, ~0.5%/step, later collected by
D3-7 2c); the short-KV combine's empty-split reads (~90 µs/step, later vetoed by
the D3b-2 analysis as bitwise-unreachable, see doc 67).

## 6. Lessons

1. **"The only kernel that grows with input scale" is attribution's first cut**:
   two-anchor per-kernel differencing + the additivity check cuts a multi-kernel
   mystery into a single kernel in one stroke — make that cut before mechanisms.
2. **The three diagnoses — latency chain, bytes, parallelism — must be separated
   by counters, not guessed**: their levers are mutually exclusive, and a wrong
   guess costs a whole session (D1 separated them in one stroke with 76.5%
   long_scoreboard + the byte decomposition).
3. **Under nix/sandbox, ncu's entrance is the verbatim standalone probe**: the
   engine binary (especially on CUDA-graph replay) is not a legitimate sampling
   target; the probe must be line-identical with the same flags, and ncu yields
   only structural readings while nsys yields time.
4. **Calibrate the "bitwise-free class" before writing code**: the classification
   that staging-depth is free while split-count is not let D2 land in a day and
   made D3a's tolerance package necessary — a measurement round's most valuable
   output is often not numbers but the freedom map.

---
← 64 · [Index](./README.md) · 66 →
