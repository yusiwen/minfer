# CUDA Inference Path — Optimization Roadmap

> **STATUS (2026-08-31): LIVE ROADMAP.** This was the pre-Phase-7 draft
> (RTX 4080 era, imperative `layer_gpu` path — all since deleted). It is
> now the consolidated CUDA index: current performance (Part I), what the
> Aug 30 session landed (Part II), the remaining plan (Part III), and the
> original roadmap with each item's actual outcome (Part IV).
> Single-sourced implementation records: `docs/CUDA-BACKEND-PLAN.md`
> (Phase 7a–7e) and `docs/CUDA-FOLLOWUP-PLAN.md` (Phase 8, 8a–8p).

## Part I — Current state (DGX Spark GB10, master `1365c82`)

llama.cpp baseline: llama-bench `ca3d5a3e1`, 8 threads, `-ngl 99`, GB10
CUDA. minfer: CLI single-shot, greedy.

| Workload | llama.cpp | minfer | gap |
|---|---:|---:|---:|
| 7B q4_k_m prefill @2K | 3401 | 2340–2370 (P5) | 1.43× |
| 7B decode tg128 | 47.1 | 47.5–47.6 (R4) | ~parity (ahead) |
| 7B decode @2K | 44.9 | 43.2–45.1 (R4) | ~0–4% |
| 0.6B q8_0 prefill @2K | 23909 | 4792 | 5.0× |
| 0.6B decode tg128 | 290 | ~195 | 1.5× |
| 0.5B q4_0 prefill @2K | 30550 | ~3020 | 10× |
| 0.5B decode tg128 | 453 | ~257 | 1.8× |

7B prefill went 30.7 → ~1450 (Aug 30) → 2340–2370 tok/s (P5, Sep 1);
the decode path closed its gap in two steps — R2 (weight streaming,
tg128 42.2 → 45.1) and R4 (split-attention rewrite, tg128 → 47.5, @2K
39.2 → 43.2–45.1 vs llama.cpp's 44.9).

What the engine does now (all in `src/cuda_kernels.cu` + `src/cuda.rs`,
dispatched by `src/graph/cuda_backend.rs`):

- **Weights resident at load**: every matmul weight uploaded once and
  registered by name; Q6_K optionally 224-byte-padded (7e②). No per-op
  H2D/D2H on the hot path (the Part IV ping-pong is gone).
- **Prefill (nt ≥ 16)**: persistent per-weight f16 dequant cache warmed at
  load (8p) → double-buffered cp.async 64×64 wmma f16 GEMM, f32 accum
  (8m②, ~35 TFLOPS measured); FA-style tiled prefill attention (8n).
  A/B: `MINFER_NO_PREFILL_GEMM=1`, `MINFER_NO_W16CACHE=1`,
  `MINFER_NO_FA_PREFILL=1`.
- **Decode (nt == 1)**: per-type MMVQ over q8_0-quantized activations
  (8e/8e②: dp4a, llama.cpp's `MMVQ_PARAMETERS_GB10` launch table), fused
  bias+rope+store, whole-step CUDA-graph capture/replay (7d) — one graph
  launch (~57 µs) per token instead of ~2.7K kernel launches. Logits
  read back through a pinned staging buffer (R3-A2;
  `MINFER_NO_PINNED_READBACK=1` reverts).
  A/B: `MINFER_NO_CUDA_GRAPH=1`.
- A dequant-in-GEMM kernel (`gemm_qb_nt_kernel`, raw quantized B tiles
  unpacked in-register, all 8 types) exists as the memory-lean
  alternative — `MINFER_FUSED_B=1` (8p; slower than the cached f16 path
  on large nt).
- **Graph capture defaults**: decode steps capture automatically;
  repeated identical-nt prefill splits capture automatically after the
  3-run protocol (R3-B — a one-shot CLI prefill never reaches 3 runs and
  pays nothing). `MINFER_NO_PREFILL_CAPTURE=1` opts out;
  `MINFER_CAPTURE_PREFILL=1` (the 8g② opt-in) is redundant but accepted.

## Part II — Landed in the Aug 30 session (`ba3f317`…`761e236`)

- **8m/8m② — wmma prefill GEMM** (`ba3f317`): one 64×64 f16 tensor-core
  GEMM over all 8 quant types replaces per-token weight re-streaming;
  cp.async tile staging added on top. 7B @2K: 30.7 → 1201 tok/s (39×).
- **8n — FA-style prefill attention** (`cb66fca`): 176 → 8.5 ms/layer at
  7B @2K (online softmax, register O accumulator, f16 probs at a 256 B
  stride to avoid a cross-thread score-clobbering race found by a
  standalone harness).
- **8o — decode-start CPU stalls** (`65b686c`): killed a ~635 ms full
  weight re-clone per graph rebuild (`Tensor: Cow::Owned` makes `clone()`
  deep-copy) and a ~920 ms eager concat probe in the decode-graph build.
  First decode step 724 → 35 ms; all decode rates unchanged.
- **8p — persistent f16 weight cache + fused kernel** (`2992f57`):
  dequant once per weight at load instead of every call (288 ms/call on
  7B); 7B @2K prefill → ~1400–1500. The cache is enabled by the loader
  only for models whose quantized matmul weights total ≥ 2 GB
  (`W16_ENABLE_BYTES`) — see the memory etiquette note in Part III.
  The new bitparity test also exposed a latent
  `cudaErrorMisalignedAddress` in `dequant_q5_0_f16` (u32 load at blk+2
  on 22-byte blocks) that would have crashed any Q5_0 prefill GEMM.

### R3 — small-model per-token overhead (Aug 31, `029a9a4`…`761e236`)

Motivation: 0.5B decode is ~4.0 ms/token with ~1.6 ms of GPU floor
(409 MB weights ÷ the 252.7 GB/s probe) — ~2.4 ms of CPU/sync overhead.
Findings came from `MINFER_GRAPH_TRACE` + a DOT dump rather than
assumption:

- **R3-A1 — single-split prefill** (`029a9a4`): the G3 tail-row-reduction
  input `tail_ids` was declared MID-graph (beside its consumers before the
  last layer's FFN), splitting every prefill forward into 4 splits — 2
  extra full-stream syncs + host round-trip copies per forward, and only
  the body split could capture. Declared at the head next to
  token_ids/positions (conditional on `n_out < nt`, as before), prefill is
  one CUDA split. Decode graphs never contained it (already single-split).
  Greedy output bit-identical.
- **R3-A2 — pinned D2H readback** (`a213c89`): the per-step logits readback
  ran a blocking `cudaMemcpy` into a PAGEABLE Vec (driver-internal pinned
  bounce — the H2D side fixed this in 7e⑥, the D2H side never got the same
  treatment), then `forward_graph` cloned the full buffer again although
  the graph output is already exactly n_out×nv. Now: grow-on-demand pinned
  staging buffer + no redundant clone (`MINFER_NO_PINNED_READBACK=1`
  reverts).
- **R3-B — prefill capture defaults ON** (`761e236`): 8g②'s validated
  opt-in becomes automatic (3-run protocol still bounds the cost; A1 made
  the real prefill graph a single split). `MINFER_NO_PREFILL_CAPTURE=1`
  opts out; `MINFER_CAPTURE_PREFILL=1` redundant but accepted.

Bench status: recorded under heavy GPU contention (the shared-box sglang
server was actively serving, ~96% util — interleaved same-binary A/B
showed the pinned path at parity to slightly ahead). Clean pre/post
numbers pending a quiet GPU; decode parity verified via bit-identical
greedy output and the 163-test suite.

### R1 — int8 MMQ prefill GEMM (Aug 31, opt-in, perf pending)

Design (not a verbatim llama.cpp port — its 128×128 warp-tile structure on
the 8p skeleton): a custom kernel on minfer's 64×64×256-thread tile
implementing llama.cpp's MMQ *math*. Activations quantize to q8_0 once per
launch (pad40 blocks, extended with the per-block int sum at offset 36),
weights stay RAW — no f16 dequant pass, no w16 cache, 2-4× less weight
traffic. Tiled `mma.m16n8k32.row.col.s32.s8.s32` (sm_80+; sm_75 keeps the
f16 path), one 32-k int chunk per step; the int C fragment is rescaled per
(token, row, k-block): `sum += da·ds·acc + da·dm·sa`. K-quant nibbles stay
UNSIGNED (q4_K/q5_K: ds=d·sc, dm=−dmin·m); q6_K = k32 chunks with dual
m16n8k16 mmas + separate accumulators rescaled by (d·sc0, d·sc1); q8_0
activations contribute the dm·sa offset term. B staging: 2 threads/row
(q8_0: 4), 8 chunks (256-k) staged per double-buffered dynamic-shared tile
(~94 KB, opt-in via `cudaFuncSetAttribute`; sm_121 reports 100 KB shared/SM
→ 1 block/SM).

Evidence:
- Parity: `cuda_prefill_mmq_parity` — host CPU q8_0-activation reference
  (llama.cpp's dot math) vs GPU, all 8 types × an 8-shape sweep (k depth
  8..112 chunks, 1..56 od-tiles, 1..4 token-tiles, q6_K both layouts):
  max diff < 1e-3 everywhere. Greedy 7B output: MMQ path ≡ f16 path
  token-for-token.
- Perf (7B q4_k_m, 2017-token prefill): MMQ 155 tok/s at sglang ~96% util,
  412 tok/s on a mostly-quiet GPU, vs the f16 w16 path 630-880 / 1460.
  Isolation probes: fixed overhead (quantize + launch) ≈ 0.6 ms — the
  matmul itself runs ~2.9 TMAC/s (llama.cpp's MMQ on the same part: ~24).
- Tuning findings: KD=8 (256-k staging) beat KD=4 (2 blocks/SM) under
  load; per-chunk (32-k) staging exposed the full load latency (2.5
  TMAC/s); ncu cannot sample this device (ERR_NVGPUCTRPERM as root:
  "Unknown Error"), so the remaining gap (≈8×) is unprofiled — likely
  per-warp tile size (llama.cpp: 2× the mma per fragment load) and 4-byte
  staging loads (llama.cpp: coalesced 16B row reads).

Status: committed behind `MINFER_MMQ=1` (default stays the f16 8p path —
MMQ would be a 3.5× prefill regression). Loader skips the w16 warm pass
only when MMQ is actually active.

### P6 raw-byte execution + wide-tile findings (2026-08-31, r7-r8)

Raw kernel landed (`mmq_raw_nt_kernel`, MINFER_MMQ_RAW=1, q4_K): parity
green, 7B token-identical, 472 vs 441 tok/s (+7%), kernel 20.2 vs
23.0 ms. Quantize pass 129 -> 74 ms (tree amax + packed stores).

Wide tile (`mmq_raw_wide_nt_kernel`, 128-token block, 8 warps x 32x32,
MINFER_MMQ_RAW_WIDE=1): mapping PROVEN correct (parity green at KD=4),
BUT the first measured 2124 tok/s was a phantom — KD=8 needs 135 KB
dynamic smem, over the ~99 KB opt-in cap; the attr-set AND launch both
failed silently and the GEMM wrote nothing (fast fake wall time). The
launcher now guards the cap and reports refusal (Rust falls back to the
narrow kernel); KD=4 (86 KB) is the only feasible wide depth.

Constraint re-rank from the r8 matrix (all real, parity-clean):
narrow KD=8 472 > wide KD=4 428 ~= narrow KD=4 427 > R1 441-ish. The
wide tile's halved B traffic bought NOTHING at KD=4 — B DRAM re-reads
are absorbed by L2, so traffic is NOT the binding constraint; and the
2x mma-per-fragment-load did not pay either. The shared bottleneck is
the per-chunk inner-loop overhead (rescale FMA chain + scale smem
reads + syncthreads cadence at 32-k granularity), which no staging
scheme fixes. ncu remains blocked on this device (ERR_NVGPUCTRPERM),
so further MMQ tuning is hypothesis-cycling at ~15 min per iteration.

Where this leaves the goal: the MMQ path (best 472) is still 4.9x off
the f16 GEMM path (2318); making MMQ default requires the inner-loop
restructure nobody has a profile for. The default f16 path is
UNTOUCHED by all P6 work and remains 2320-2370 tok/s.

FA (Step 4): an L2-prefetch of the next KV tile before the
single-buffered stage measured a null delta (2319-2345 vs 2345) — the
GQA sharing (7 q-heads per kv head read the same K/V) already keeps
the tiles L2-hot, and with ncu blocked there is no second hypothesis
worth a build cycle. Reverted; fa_prefill_f16kv stays 1.86 ms/layer
(P5 state), 2.4x behind llama.cpp per layer but only ~6% of the wall.

### P6 r9: the llama.cpp MMQ reference, decoded — and the shape axis closed

Read the actual reference (mmq-config-blackwell.cuh falls through to
mmq-config-ampere.cuh for Q4_K): 256 threads, targeted occupancy 1,
SRAM tile I=128 od-rows x J<=128 tokens, ITER_K=256, SYNCHRONOUS
staging (plain global->smem int loads, no cp.async), float
sum[J*I/threads] accumulators, 16 mma.m16n8k32 per warp per 32-k chunk.
Instruction model: ~0.018 inst/MAC/thread vs our raw kernel's 0.133 —
that ratio, not tile shape alone, is where their ~30 TMAC/s comes from.

Sync-staging applied to our wide kernel (single buffer, 54 KB, 2
blocks/SM): parity green, 462-466 tok/s vs narrow 481 — still loses.
Full shape matrix on GB10 @2K (all parity-clean, MINFER_MMQ=1):

  narrow  cp.async 64x64  KD=8  481   <- local optimum
  sync    wide 128x64     KD=8  464
  compact cp.async 128x64 KD=8  410   (sda_q sync loads serialise)
  wide    cp.async 128x64 KD=4  428
  narrow  cp.async 64x64  KD=4  427
  R1 word-stage 64x64     KD=8  441

B-DRAM-halving is dead on this device (L2 absorbs re-reads); wider
tiles pay more in staging/sync than they save. The shape axis is
exhausted with evidence. The one remaining MMQ lever is llama.cpp's
MMA data layout: pre-arranged mma fragments for the weight side, which
could be produced at WEIGHT-LOAD time by our existing w16 dequant pass
(currently producing f16 we then re-read). That is a load-time format
change plus a kernel rewrite against a new payload layout — a
standalone decision, not a shape tweak.

### P6 r10: the reference inner-loop decomposition, ported — perf flat; delta localized

The q4_K vec_dot in the reference is the UNIVERSAL q8_1 x q8_1 mma
(mmq-vec-dot.cuh): the nibble unpack and the dmin fold happen ONCE per
(row, chunk) inside load_tiles_q4_K, and the compute loop applies
scales as `sum += dmA.x*dsB.x*C + dmA.y*dsB.y` with the weight-side
scales preloaded into registers before the j0 loop.

We ported that decomposition into the sync-wide kernel (dequantize
q4_K to per-k signed int8 in the staging phase — the GGML
sub-block-pair byte interleave handled — plain q8 x q8 mma, no nibble
unpack, no dmv/sa term, rescale halved). Parity green. Measured
462-468 tok/s vs narrow 470: FLAT. Both sit at ~6.4 TMAC/s while the
reference reaches ~30 TMAC/s with the same decomposition, so the
residual is NOT the decomposition either. It is:

1. ILP depth: the reference warp carries 16 independent mma chains per
   chunk (rows_per_warp=32 -> ntx=2 minitiles x J/16 j0-groups) backed
   by 128 accumulator registers (sum[64] + 16 int C-frags); ours has 8
   chains and ~64 accumulator regs. At 1 block/SM the register file
   (255/thread) is half idle — the compiler cannot create chains the
   source does not express. VERIFIED (r11): the I*J/64 ne seen earlier
   is the AMD MFMA branch; the NVIDIA (Turing+) tile is ne = I*J/32 =
   4 regs per m16n8k32 C — so the 16-chain reading stands. Next session
   step one: widen our warp tile to 16 independent mma chains (sum[64],
   16 int C-frags) and re-format the A smem for ldmatrix.
2. ldmatrix A staging: their A fragments load with one LDSM per
   16x32 int8 fragment; ours use 8 LDS.32.
3. Tile 128 od-rows x 128 tokens (vs our 128 x 64).

Redo recipe (the edit session was lost to a post-checkout hook revert
after measurement; measurements above are valid): stage B as per-k
signed int8 ((nib - m), get_scale_min_k4 in staging, byte k at offset
(sg>>1)*32 + k, low/high nibble by sg parity), qb8 = 64 x KDR x 32B,
b-frag words = *(int*)(qb8 + jr*KDR*32 + kd*32 + 4*(lane&3)) [+16],
rescale = sum += da*dsv*C only. Then widen the warp tile to 16 mma
chains (sum[64], clow[64][4] -> needs 2 x ntx minitiles like the
reference) and re-format the A smem for ldmatrix.

### P6 r12: 16-chain warp tile + ldmatrix LANDED — wide MMQ 1.44-2.3x

The redo recipe was executed (block 128 tok x 128 od; each warp owns a
private 16-od-row slice x the full 128-token tile = 8 A-frags x 2
B-frags = 16 independent mma chains per chunk with sum[64] +
clow[8][2][4] live; A fragments load via ldmatrix.m8n8.x4; at KDR=4
the qb8 restage is skipped when two consecutive k-tiles share a
super-block; B staging format and the two-term rescale untouched;
A-row 48B padding tried and reverted — flat). Parity green, suite
169/0, greedy tokens identical to the default path.

Measured (7B @~2K, same session, interleaved): wide-16 KD=4
1020-1058 tok/s vs narrow raw 441-481 in the same window = ~2.3x;
KD=8 973-995. vs the pre-rewrite wide (~719 on a faster machine
state) this is 1.44x; today's machine ran ~38% below the baseline
session, so the same-session ratio is the meaningful number.
MMQ is now ~2.3 TMAC/s-class on q4_K but still ~2.2x below the f16
default GEMM path (2284 tok/s same session) — not yet promotion
material. Residual per the model: the accumulator depth wall is
passed; the next q4_K lever is load-time B-side fragment
pre-formatting (a standalone decision), and the wall is increasingly
dominated by the non-q4_K GEMMs + the quantize pass. Best config:
MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_WIDE=1
MINFER_MMQ_RAW_KD=4 (KD default stays 8, shared with narrow).

P6 re-ranks (2026-08-31, 7B @2K, 2061-token CLI prefill): KD=4 retested
427 vs 438 tok/s (occupancy 1->2 blocks/SM does not pay; staging-depth
amortization dominates — docs finding above confirmed). The bigger
per-warp tile was also tried: 4 warps x 32x32 (mma per fragment word
x2, staging loopified for blockDim generality) measured 399 tok/s AND
broke mmq_w80 parity (zero cells — coverage hole in the hand-mapped
fragments), so it was reverted. Remaining lever per the note above:
wide 16B staging loads + raw-byte smem staging (dequant at mma time),
which is a structural rewrite of mmq_stage_b, not a knob.

Wall-split correction (P6 r5 profile re-read): the 352 ms dequant pass
runs at LOAD time (w16 warm pass), NOT inside the Prefill wall — a
vectorized q4_K dequant measured a null end-to-end delta and was
reverted (the scalar version was already warp-coalesced). The 890 ms
Prefill wall is: GEMM ~600 ms + convert 56 + fa 54 + swiglu 51 (re-
verified at ~257 GB/s = bandwidth peak) + add 19 + host gaps. The GEMM
is ~85% of the wall: it is the only material lever left.

### Wide-tile x-axis closed: 256-token block measured and reverted (2026-09-02)

Hypothesis: widening the wide kernel's token x-tile 128 -> 256 halves
weight re-reads through L2 (then believed the bottleneck, ~21 re-reads
per weight row). Implemented per the worked design — block 256 tok x
64 od, warp = 16 od-rows x one 128-token half (`i0w = (warp&1)*128`,
`j0w = (warp>>1)*16`), all per-warp structures unchanged; launcher
grid `(nt+255)/256 x (od+63)/64`, KD=4 smem 59,392B, KD=8 (102,400B)
refused -> Rust narrow fallback. Parity green on both paths, but perf
REGRESSED: interleaved 3x on the 2600-token 7B prefill = 927/953/945
(median ~942 tok/s) vs 1035-1058 for the 128x128 tile (narrow ~476
unchanged); ncu q-proj: duration 4.584 ms vs 3.619 ms (+27% kernel
time), Memory SOL 73.6% vs 75.7%, Compute 22.1% vs 22.9% — the memory
system ran at the same utilization for 27% longer, i.e. MORE bytes
through L2, not better reuse. Root cause: halving the od tile doubles
the y-block count, and ACTIVATION re-staging (qa8/sda_q, 4,480 B per
token at id=3584) scales with `od/WBJ` while weight bytes per x-tile
visit are only 2,016 B per od-row — A re-reads (~327 MB) were already
~2x the B re-reads (~152 MB) at 128x128, so trading -72 MB of B for
+327 MB of A is strictly negative. The alternative that would cut A
re-reads (16 warps x 128 tokens = 128x256) is impossible: the kernel
needs REG 156-166/thread, so 512 threads cannot be resident (64K
regfile) and occupancy is pinned at 1 block/SM for every tile in this
family. Conclusion: within the 16-chain/sum[64]/256-thread structure,
128x128 is the optimal tile; the token-x-axis and the od-x-axis of
this kernel are both closed by measurement. If L2 traffic is revisited,
the only remaining shape is reusing one staged A across two j-tiles
inside the block (128 tok x 256 od effective; A staged once per k-tile,
B re-expanded per j-tile; same smem/registers, needs an outer j-loop —
untested). Tree reverted to 5c34b4e after measurement.

### j-tile A-reuse inside the block (128tok x 256od, outer jh loop) measured — bar missed, reverted (2026-09-02)

The "only remaining shape" above, implemented and measured (HEAD `fabdefb` +
diff, NOT landed). Block stays 128 tokens wide; od coverage doubled to 256 via
an outer `jh` loop: `grid.y = (od+255)/256`, A (`qa8`+`sda_q`) staged ONCE per
k-tile and reused by both 128-od halves, `qb8`/`sds`/`sdm` one shared set
restaged per (kt, jh) with the jh-shifted row base, `sum[2][64]` accumulator
sets (REG 156-166 -> 175, LOCAL:0, still 1 block/SM), one stage+sync round per
half. The old super-block restage-skip is UNSOUND once `qb8` is shared (the
previous stage always left the other half's rows in the buffer), so B is
re-expanded every (kt, jh) but only the KDR slots the k-tile consumes (KDR/2
pairs = 64B/row at KD=4) — 2x the old per-block B bytes, not 4x.
- Parity: `cuda_prefill_mmq` green on first build at KD=4 and KD=8 default.
- Perf (interleaved 3x, 2600-tok 7B prefill): wide KD=4 1076/1067/1033
  (median ~1067 tok/s) vs narrow 483; baseline band 1035-1050 => ~+2.6%,
  bar (>=1150) missed.
- ncu q-proj (grid (21,14): grid.y halved 28 -> 14 as designed): duration
  3.62 -> 3.55 ms (-2%), Memory(SOL) 75.7 -> 74.0%, Compute 22.9 -> 20.3%,
  issue 0.25 -> 0.22/sched, eligible warps 0.32, active warps 2.00 (1
  block/SM unchanged).
- Root cause: the L2-byte premise does not convert to time. A re-reads DID
  halve as designed (-163MB of ~479MB L2 reads; per-block B materialization
  doubles but blocks halve, so B total is unchanged), yet the kernel is
  LATENCY-bound at 1 block/SM (2 active warps/sched, issue ~0.22): staging
  cost is the per-kt global->smem round trip at 8 resident warps, not L2
  byte volume, and the added second B-stage + barrier round per k-tile gives
  back what removing one A-stage per two compute-units saves.
- Conclusion: the j-tile shape is measured and closed like the x-tile one.
  Further MMQ GEMM gain needs latency hiding — cp.async/TMA double-buffered
  staging, or 2 blocks/SM — not tile-shape traffic arithmetic.
- Tree reverted to `fabdefb` (byte-identical, suite 169/0 re-verified);
  patched kernel preserved at /tmp/cuda_kernels_jh256.cu (script
  /tmp/patch_jh.py).

### cp.async double-buffered raw staging measured — no conversion, reverted (2026-09-02)

The "needs latency hiding" hypothesis above, implemented and measured (HEAD
`82dfc90` + diff, NOT landed). Two-level staging with the raw global->smem
transfers moved to cp.async and prefetched one k-tile ahead; expansions run
smem->smem after the wait. KD=4 (98,304B): raw pad40 A chunks staged 5x 8B
cp.async into a rawa buffer + expanded to qa8/sda_q in smem; raw q4_K
super-blocks 9x 16B cp.async into rawb x2 (ping-pong), expanded to a
partial-slot qb8 `[128][KDR][32]` (only the ktile's KDR slots — odd ktiles
expand the resident sb's high half, halving expansion work) + scales built
from rawb in smem; commit discipline {A(kt+1)},{B(kt+2)} with wait_group 1 on
odd kt leaving the B prefetch in flight. KD=8 (100,352B): same rawb scheme
x1, legacy sync A staging (no rawa budget).
- Parity: `cuda_prefill_mmq` green on first build, KD=4 AND default KD=8.
- Perf (interleaved 3x, 2600-tok 7B prefill): wide KD=4 1030.3/1042.7/1034.1
  (median ~1034 tok/s) vs narrow 472-488; baseline band 1035-1050 => PERF-
  NEUTRAL, bar (>=1150) missed.
- ncu q-proj before -> after: duration 3.62 -> 3.62-3.67 ms (run noise),
  Memory(SOL) 75.7 -> 78.2%, Compute(SM) 22.9 -> 23.8%, issue 0.25 -> 0.25,
  eligible 0.33, active warps/sched 2.00 (1 block/SM, REG 167, smem 98,304B).
- Root cause: the prefetch WORKS — L2 utilization rose (same bytes, better
  overlap) — but wall time did not drop, i.e. the kernel is L2-THROUGHPUT-
  bound (~76-78% SOL), not MLP-starved: the old synchronous staging's 8-warp
  memory-level parallelism already saturates the achievable L2 rate, so
  re-timing identical bytes wins nothing and the extra expansion + barrier
  round gives a little back. This closes the last staging-shape lever: with
  traffic-shape experiments (x-tile, j-tile) and now latency-shape (cp.async
  double-buffering) both measured flat, the remaining MMQ levers are per-MAC
  L2-byte reduction (more k-reuse per byte fetched, stream-k-style
  scheduling) or SM-side instruction efficiency — not staging order.
- Tree reverted to `82dfc90` (byte-identical, cmp-verified); patched kernel
  preserved at /tmp/cuda_kernels_cpasync.cu (script /tmp/patch_cpasync.py).

### P6 r13: counter forensics vs llama.cpp — the gap is the warp-instruction stream, not bytes (2026-09-02)

First working ncu session on this device (metrics gated behind
`sudo -n env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu ncu ...`; note GB20B
has NO `dram__*`/`launch__grid_size`/`l1tex shared-sector` metrics — use
`lts__t_sectors_aperture_device` for memory-side and the CSV "Grid Size"
column for identification; int8 mma counts under
`sm__inst_executed_pipe_tensor_subpipe_imma_op_imma`, NOT the hmma counter,
which reads 0 for BOTH kernels). Profiled: minfer `mmq_raw_wide_nt_kernel<4>`
q-proj (nt 2630, od=id=3584, grid (21,28), 3.632 ms) and llama.cpp
`mul_mat_q<12,128,0>` q-proj-class launches (nt 512 per ubatch, 0.241-0.292
ms, MAC-normalized; llama-bench splits -p 2600 into ubatches of 512 — the
592-block grid belongs to the nt-2600-class GEMMs, q/o at nt 512 run grid
(48,1,1)). Effective MACs used for normalization.

Per-GEMM counter table (q-proj class, per 1e9 effective MACs = GMAC):

| metric (per GMAC)                    | minfer wide KD=4 | llama.cpp mul_mat_q |
|--------------------------------------|------------------|---------------------|
| duration (μs/GMAC)                   | 107.7            | 41.1                |
| warp instructions                    | 10.14 M          | 6.06 M              |
| IMMA tensor ops                      | 2.05 G (=2×MAC)  | 2.00 G (=2×MAC)     |
| shared-load instructions (LDS+LDSM)  | 436.7 K          | 329.5 K             |
| LDS bank conflicts                   | 940.3 K          | 6.5                 |
| LDSM bank conflicts                  | 499.0 K          | 0                   |
| L1 global-load REQUESTS              | 75.3 MB          | 18.1 MB             |
| L2 bytes total                       | 88.0 MB          | 15.2 MB             |
| L2 read sectors                      | 22.0 MB          | 13.7 MB             |
| L2 write sectors                     | 66.1 MB          | 1.45 MB             |
| C-store requests (L1)                | 2.24 MB          | 1.44 MB             |
| stalls/issue-active: longsb/wait/barrier/shortsb | 2.35/0.83/0.62/0.26 | 1.21/0.58/0.18/0.10 |

Key readings:

1. mma work per MAC is IDENTICAL (IMMA = 2.0-2.05 ops/MAC both sides — the
   16-chain structure is now at parity). The 3x is everything AROUND the mma.
2. minfer executes 1.67x more warp instructions per MAC, and duration tracks
   instruction count ~1:1 across every data point measured this session
   (minfer old 10.14 M/GMAC -> 107.7 μs/GMAC; minfer +staging-rework
   13.5 M/GMAC -> 146 μs/GMAC; llama 6.06 M/GMAC -> 41.1 μs/GMAC ≈ 0.10-0.15
   warp-inst/ns on both engines). Per-MAC instruction count is the
   first-order predictor of this kernel class on GB10.
3. The L2-byte story is NOT the binding constraint. minfer's L2 read traffic
   is only 1.6x llama.cpp's per MAC (22.0 vs 13.7 MB/GMAC — A 40B/chunk
   staging + per-chunk scale-header re-reads), and fixing the two biggest
   byte pathologies (below) changed NOTHING in wall time.
4. UNEXPLAINED RESIDUE: minfer shows ~66 MB/GMAC of L2 WRITE sectors
   (2.2 GB per q-proj GEMM, 75% of its L2 traffic) that are NOT C stores
   (requests 2.24 MB/GMAC) and did NOT move when the C-store pattern was
   fixed (2.23 -> 2.18 GB). No source-level writer exists. llama.cpp shows
   ~1.4 MB/GMAC (≈ C size, zero amplification). Either a GB20B lts
   write-counter artifact or a mechanism outside the source model — needs a
   `lts__t_sectors_srcunit_tex_op_write` decomposition before trusting any
   "L2-throughput-bound" conclusion (the P6 r12 cp.async framing relied on
   this SOL number).
5. llama.cpp on GB10 uses the same mma path (IMMA confirmed), 32 od-rows per
   warp (2 minitiles x 128-token half), raw-nibble weight planes in smem
   (65-int padded stride, bank-rotating), q8_1 activations pre-quantized on
   device in the transposed 144B layout with scales in the pad bytes, B
   (token) fragments via plain LDS ("faster than load_ldmatrix"), A-side
   scales preloaded into registers per chunk outside the j0 loop, and
   stream-k scheduling. Instruction-model differences vs ours per 128k chunk:
   their warp covers 2x the od-rows (halving A-fragment loads per MAC), and
   our B-side reads are 16 conflicted LDS.32 vs their 8 LDSM.x4.

Two fixes were implemented and measured (parity-green at KD=4 after fixing a
missing qb8-region resize and a staging race):

- FULL variant (merged 40B single-pass A staging killing the 2B-at-40B-stride
  sector-replay loads, per-super-block scale staging with register unpack of
  all 8 sub-scales, qb8 272B conflict-free stride, float2 C stores):
  L1 load requests 75.3 -> 33.9 MB/GMAC (-55%), L2 reads -14%, BUT
  instructions 10.14 -> 13.48 M/GMAC (+33%) and duration 3.63 -> 4.89 ms
  (+35%) -> 865-871 tok/s (-16%). The staging restructures cost more
  instructions than the bytes they saved.
- MINIMAL variant (qb8 272B stride + float2 C stores only; preserves
  31.8M LDS conflicts killed and halves store instructions): parity green,
  1041-1046 tok/s vs 1023-1043 baseline = PERF-NEUTRAL (+0.3%, noise).

Conclusion: neither L2 bytes, nor store sector efficiency, nor smem bank
conflicts bind this kernel — all three were fixed simultaneously with zero
wall effect. The binding resource is the per-MAC warp-instruction stream
together with issue efficiency (llama.cpp 0.41 issue vs our 0.25 at 2 active
warps/sched both). The quantified next lever is instruction-count reduction
in the per-chunk compute loop itself (fewer, wider smem ops per fragment:
B-fragments as one LDSM from a layout that serves them, scale reads fused
into fewer loads per chunk — WITHOUT adding staging ALU), not another
traffic-shape change. Kernel reverted to HEAD `784786d` (kernel state = `82dfc90`; cmp-verified, suite
169/0); the minimal variant is preserved at
/tmp/cuda_kernels_minfix_variant.cu, the full variant's patch script at
/tmp/patch_fix.py.

### P6 r14: B-fragments via ldmatrix + widened scale loads — wide MMQ 1225 @ KD=4 / 1273 @ KD=8 (LANDED, 2026-09-03)

The r13 lever, implemented exactly: fewer/wider smem ops in the compute
loop, zero staging-ALU growth, zero new global traffic. Three changes to
`mmq_raw_wide_nt_kernel` only (narrow kernel untouched):

1. **B-fragments via ONE `ldmatrix.m8n8.x4`** per (warp, chunk) replacing
   4 LDS.32. qb8 re-tiled SLOT-MAJOR `[8 sg][128 od-row][48B]` — same
   expanded per-k int8 bytes, relayout only. The 48B row stride is 16B-
   aligned (ldmatrix requirement) and gives the 8 matrix rows distinct
   bank phases (12r mod 32), so the LDSM is conflict-free (a 32B stride
   would put all 8 rows on one bank phase = 8-way conflict). The reg_i =
   matrix_i row L/4, bytes (L%4)*4 ldmatrix distribution is exactly the
   mma.m16n8k32 B-operand layout the plain-LDS pattern produced (verified
   standalone vs the LDS pattern before integration). Per-lane address
   parts are loop-invariant; only the uniform sg term moves per chunk.
2. **od-col scales packed float2** (d | dmin*m) at staging (same staging
   store count); compute reads ONE float4 per minitile serving the (j, j+1)
   column pair each C fragment consumes → 8 LDS.32 → 2 LDS.128.
3. **sda_q uint2-tiling** `[KDR][16 g][8 q]` (same 8B/token region size):
   token pair (t, t+8) in one LDS.64 → 16 LDS.32 → 8 LDS.64.

First cut used a uint4 tiling needing KDR*2048B in a KDR*1024B region —
staging overflowed into qb8 and corrupted it (garbage-magnitude parity
diff; found by bisect after fixing a missing j0w term and a uint4 .y/.z
word-offset slip). Smem budgets (launcher recomputed): KD=4 73,728B,
KD=8 98,304B — both inside the ~99KB opt-in cap, 1 block/SM.

Measured (7B @2630 tok, same session, interleaved 3x vs narrow):

| config | baseline (HEAD 5f801fa) | r14 | Δ |
|---|---|---|---|
| wide KD=4 | 1036.9 / 1031.8 / 1011.7 | 1228.0 / 1219.7 / 1224.7 | **+18.5%** |
| narrow (control) | 470.7 / 460.0 / 470.7 | 440.7 / 466.8 / 472.3 | noise |
| wide KD=8 (default) | ~973-995 (r12) | 1269.5 / 1276.9 | **+23-30%** |

ncu (q-proj, nt 2630, grid (21,28), KD=4, per GMAC = 33.78e9 MACs):

| metric | r13 baseline | r14 |
|---|---|---|
| warp instructions (smsp__inst_executed.sum) | 10.14M | **9.45M** (−6.8%) |
| shared-load instructions | 436.7K | **156.0K** (−64%) |
| LDS bank conflicts | 940.3K | **499K** (−47%) |
| duration | 3.632 ms | **2.378 ms** (−34.5%) |

The residual 499K conflicts/GMAC = the A-side LDSM 2-way conflicts (r13's
499.0K unchanged — qa8 keeps its 32B stride; padding it would need +16KB,
pushing KD=8 over the cap). Instructions dropped only 6.8% while duration
dropped 34.5% — most of the win is issue efficiency (smem-op count 44 →
~24 per chunk and their address ALU), confirming r13's issue-stall
reading (longsb 2.35/issue). Gap to llama.cpp narrowed: 70.4 vs their
41.1 μs/GMAC (was 107.7). Parity green at KD=4 and KD=8, suite 166
passed / 3 ignored / 0 failed, greedy-32 token identity vs the default
path (timing lines only). Next lever (not attempted): f32-accumulate
mma.m16n8k32.f32.s8.s8.f32 kills the 64 I2F/chunk rescale (per-chunk
dot ≤ 61,440 < 2^24, so results are bit-identical), and the ~330 rescale
FMUL/FMA/chunk is the remaining instruction hog.

### P6 r15: f32-accumulate s8 mma does not exist; rank-1 term2 rescale — wide MMQ 1295 @ KD=8 (LANDED, 2026-09-03)

r14's named next lever — replacing the int C fragments + I2F rescale with
`mma.m16n8k32.row.col.f32.s8.s8.f32` — is IMPOSSIBLE: ptxas (CUDA 13.0)
rejects the spelling with "Unexpected instruction types specified for
'mma'" on sm_80/90/100/110/120/121 under both PTX ISA 8.8 and 9.0, while
the s32-accumulator control assembles on all of them (/tmp/mma_probe_f32.ptx,
raw-PTX probe). Integer mma has s32-only C/D accumulators; f32 C/D exists
only for f16/bf16/tf32/fp8 inputs, and none of those can carry this data
bit-exactly (q8·scale needs ~18 significand bits vs 11 in f16/tf32) — the
"float-quantized A operand" variant is dead with it.

The planned fallback (accumulate 2 chunks in int before one I2F) is
mathematically INVALID, independent of the 2^21 overflow bound (which was
fine): the rescale coefficients da·dsv and da·dmv differ per 32-k chunk
(`get_scale_min_k4(c & 7, ...)` sub-block scales, per-chunk A d/ssum), so a
merged int dot no longer carries the split needed to apply both chunks'
scales. The error is O(Δscale·dot) ≈ 10²–10³ vs the 1e-3 parity gate for
every possible chunk pairing — the information split dies in the int add.

What DID land, same instruction-stream family: the dmv correction term is
RANK-1 in (token, od-col) — sa is a per-ROW quantity shared by the od-col
pair of each C fragment — so `dma = da·(float)sa` replaces 64
FMUL(da·dmv)/chunk with 16 FMUL(da·sa)/chunk + plain FFMAs (term2 becomes
one FFMA per C value). Per-chunk scale application and the mma/int-C
structure are untouched; numerics stay within ~ulp(|sum|) of the old form
(the new dma rounding is ~1e-6 in the parity fixture).

Measured (7B @2630 tok, interleaved 3x vs narrow, same session):
baseline KD=4 1235.7/1225.8/1226.4, KD=8 1277.1/1268.3/1270.6, narrow
487.7/455.8/487.4 → patched KD=4 1231.2/1232.9/1237.9 (+0.5%, noise),
KD=8 1295.0/1294.5/1295.6 (+1.9%, 3/3 consistent), narrow 473/472/472
(control stable). ncu q-proj (KD=4, grid (21,28)): warp instructions
9.45M → 8.69M/GMAC (−8.1%), duration 2.378 → 2.262 ms (−4.9%). The
sub-1:1 instruction→duration tracking comes with a diagnosis:
SpeedOfLight shows Compute (SM) throughput at 31.5% — the kernel is
stall-bound, not issue-saturated, so further pure ALU-count cuts pay out
sub-linearly. Parity green at KD=4 and KD=8, suite 166/0/3, greedy-32
token identity vs the default path. The rescale floor is now ~346
ops/chunk (64 I2F clow + 16 I2F sa + 16 h2f + 16 FMUL dma + 64 FMUL
da·dsv + 128 FFMA + unpack/LDS); the I2F stream itself is irreducible at
the ISA level. Next levers: the r13-noted warp-tile shape (a warp over
2x od-rows halves A-frag LDSM and per-MAC rescale work — llama.cpp's
structural edge) or prefetch/stall work; the narrow kernel's identical
rank-1 term2 (3 sites) is the same one-line follow-up.

### P6 r16: narrow MMQ gets the r15 rank-1 term2 fold (LANDED, 2026-09-03)

r15's named one-line follow-up, ported to `mmq_raw_nt_kernel` (3 edit
sites, mirroring the wide kernel's r15 pattern exactly): `dma =
da·(float)sa` folded once per row (4 FMUL/chunk) replacing the per-C-value
`I2F(sa)` + `FMUL(da·dmv)` — 8 I2F + 16 FMUL → 4 I2F + 12 FMUL per chunk;
the dsv term and per-chunk scale application untouched.

Measured (7B @2630 tok, narrow KD=8, interleaved 2x vs the HEAD binary):
patched 480.2/480.9 vs baseline 447.0/472.6 tok/s — not worse (+1.7%/+1.8%,
within noise of the r15 control 473/472/472). Parity green at default,
`MINFER_MMQ=1 MINFER_MMQ_RAW=1` (KD=8) and `MINFER_MMQ_RAW_KD=4`; suite
166/0/3. No ncu (the narrow kernel is not the perf path; wide numbers
unchanged).

### MMQ structural rewrite — execution spec (P6 r6, for next session)

Goal: mmq GEMM 6.1 TMAC/s (23 ms per ffn_gu call) -> >=24 (f16-GEMM
parity) or ~30 (llama.cpp parity). Wall impact: at parity the MMQ path
(quantize ~130 + GEMM ~600) deletes the 56 ms convert -> ~2670 tok/s;
at llama.cpp parity the GEMM drops to ~480 ms -> wall ~634 ms -> ~3250
tok/s (goal met).

Design (llama.cpp mmq structure; q4_K first = 79% of MMQ time):
1. smem holds RAW bytes only: A per (token, 32-k chunk) = 40 B pad40
   (2x uint4 qs + uint2 d/ssum); B per (row, 256-k super-block) =
   144 B Q4KB (9x uint4). KD = 8 chunks (256-k) per double buffer.
   Footprint: A 2x8x64x40 = 41 KB + B 2x64x144 = 18.4 KB + scales
   ~ 62 KB -> 1 block/SM at KD=8; KD=4 -> ~31 KB -> 2-3 blocks/SM
   (re-measure both; r5 showed depth beats occupancy for the word-
   staging kernel, raw staging shifts the tradeoff).
2. Staging = pure cp.async 16B chunks (A: 2x16 + 8 B; B: 9x16),
   commit_group per (kt, buf); NO dequant ALU in the staging path.
3. The mma loop dequants IN REGISTERS from smem raw bytes (same math
   as mmq_stage_b TYPE==5: get_scale_min_k4 per (row, sub), nibble
   unpack, __vsubss4) so the ALU overlaps tensor-core issue instead of
   serializing before __syncthreads.
4. Warp tile stays 32(i)x16(j) x 8 warps on 64x64 (the 4-warp 32x32
   variant measured slower AND broke parity — do not retry).
5. Land as mmq_raw_nt_kernel<TYPE> beside mmq_nt_kernel; gate
   MINFER_MMQ_RAW=1 through prefill_mmq so R1 stays intact; q4_K only
   in the first cut (q6_K KSPLIT=2 later).
6. Parity: extend cuda_prefill_mmq_parity with a raw-mode arm (same
   host reference), then the 7B greedy-token-identity check vs the f16
   path, then quiet-window A/B vs R1 MMQ and vs the default f16 path.
7. Quantize pass (129 ms) is follow-up work once the GEMM wins: it is
   convert-bandwidth (~87-102 GB/s) and latency-bound on the serial
   fmaxf chain — tree-reduce amax + register-packed stores, target
   ~60-70 ms.

### R2 — MMVQ weight-streaming rework (Aug 31, DONE)

The 8e decode kernels ran at ~60% of llama.cpp's effective streaming rate
(147 vs ~197 GB/s) because each 32B nibble chunk was read per SUB-BLOCK —
the sibling sub re-reads the same bytes for the other nibble half (2× the
load instructions; L1 absorbed the traffic, the instruction stream did
not) — and q6_K fetched its ql/qh pieces as eight 2-byte loads.

v2 (default when `id % 256 == 0`; q6_K also needs the padded 224B stride;
`MINFER_MMVQ_V1=1` forces the 8e kernels for A/B):
- q4_K/q5_K: one thread per 32-element chunk = a sub-PAIR sharing its
  nibble bytes; the 32B chunk loads once (uint4×2) and serves both subs.
- q6_K: one thread per is-pair (two 16-element subs sharing ql/qh bytes
  and one q8 block); uint4 loads in the padded layout (all 16B-aligned).
- q5_K's qh plane is 32 bytes shared by ALL sub-blocks (byte l = one high
  bit per sub for element l) — bit-indexed, no per-chunk offset.

Numbers (7B q4_k_m, quiet-GPU interleaved A/B, 2 stable runs each):
tg128 42.2 → 45.1 tok/s (+6.9%), decode @2K 36.7 → 38.8 tok/s (+5.7%) —
llama.cpp sits at 47.1 / 44.9. Parity: the per-type decode tests extended
with an id=2560 shape so the suite exercises v2 (the original id=2176
partial-tail shape dispatched to v1, which is exactly how a q6_K
nibble-group bug and a q5_K qh-offset bug slipped past the first run —
the engine-level greedy check caught them); suite 164 passing; greedy
output v1 ≡ v2 token-for-token. Under sglang load the win holds at ~+5-8%
relative.

### R4 — decode split-attention rewrite: dim-parallel lanes (Sep 1, 2026)

The 8d flash-decoding pass was the whole @2K decode gap: nsys showed
`gqa_attn_split_partial` at ~150 us/layer (28 × 150 us = 4.2 ms of the
~25.8 ms 7B @2K step; tg128 carries no such cost) — 28 GB/s effective on
a 4.3 MB K+V read. Three compounding causes: the runtime-indexed
`float4 oc[32]` accumulator lived in LOCAL MEMORY (~80 MB/layer of local
traffic, re-read+re-written on every online-softmax rescale); each lane
walked whole K/V rows with 4-byte loads (64 scattered sector requests
per row, 12.5% sector utilization, L1-bandwidth bound); only 224
single-warp blocks (~4.7 warps/SM). Naively raising the split count made
it monotonically WORSE (148/172/419/609 us for 8/16/32/64 splits — more
resident warps thrash L1 with the local oc arrays).

Rewrite: each lane owns 4 fixed dims (the accumulator is ONE float4 in
registers, zero spill; hd % 4 == 0 && hd <= 128 enforced by the
dispatch), every K/V access is a perfectly-coalesced row instruction,
the row dot is a warp reduction, and rows run in batches of 4 to overlap
the serial online-softmax chain. `ATTN_SPLITS` 8 → 32 (fixed grid stays
capture-safe; idle splits write mx=-INF/S=0 partials the combine weights
to zero; scratch 4x bigger, still < 0.5 MB; combine loop widened).
Kernel: 148 → 79 us/layer. End-to-end (interleaved same-binary A/B,
-n 96): @2K 39.2 → 43.2–45.1 tok/s (llama.cpp 44.9 — gap 14% → ~4%),
tg128 45.1 → 47.5–47.6 (llama.cpp 47.1 — at/above parity). Parity:
`cuda_attn_split_decode_parity` extended to sweep the SPLITS=32 chunk
boundaries (nkv 3..207 × f16/f32 KV). SPLITS=64 measured no better
(44.7 vs 45.1 @2K); 32 kept.

### P5 — prefill gap: GEMM tiles + flash-attn rewrite (Sep 1, `b8568cd`…`1365c82`)

Goal: close the 7B @2K prefill gap (was 1435 vs llama.cpp 3401, 2.37×).
Four steps, each suite-verified and measured with interleaved same-binary
A/B runs on the 2K prompt (full-output diff, never prompt-echo grep):

1. **8p — elementwise vectorization** (+4%, 1435 → 1493): store_kv_f16
   1→4 dims/lane, convert_f32_f16 1→8 elems/lane. The generic 1-elem
   kernels were launch-bound and left 15/16 of every memory transaction
   unused.
2. **8q — 128-wide GEMM tiles** (+30%→2267): `gemm_f16_nt_kernel_t<TM>`
   with TM=128 halves the B-panel L2 re-reads and the barriers per FLOP
   vs TM=64 (455→302 ms kernel time); TM=64 kept via `MINFER_GEMM_TM=64`.
   Isolated with hybrid-kernel bisects after the first version corrupted
   the GEMM: `fb[1]` must offset +16 ELEMENTS (the k-half) not +16 rows
   (the next od row) — caught by `cuda_prefill_f16_gemm_parity`.
3. **8r — FA prefill softmax on all 8 warps** (+3%): warp-per-row online
   softmax (8 warps × 8 rows, lanes own 2 cols, shuffle reductions)
   replaced the 64-deep serial chains on 2 of 8 warps; -INF seeds for
   all-masked rows.
4. **8s — padded smem rows + the two FA bugs** (2267 → 2365–2371): with
   hd=128 every smem row was 256 B ≡ 0 mod 32 banks → 8-way bank
   conflicts on every wmma ldmatrix; +8-half row stride (272 B) shifts
   rows 4 banks. Two bugs on the way: (a) the double-buffered working
   set exceeded GB10's 99KB/block smem cap → `cudaFuncSetAttribute`
   failed → **silent** fallback to the legacy per-token kernel (313
   tok/s); the fallback now prints a one-time warning and the padded
   layout ships single-buffered (69KB); (b) dropping the double buffer
   left cp.async staging without `commit_group`/`wait_group 0` — a bare
   `__syncthreads` does NOT order async copies (0.28 parity diff).
   fa_prefill_f16kv: 4.25 → 1.92 ms/layer (llama.cpp flash-attn 0.79).

5. **k-step-64 (negative result, `1365c82`)**: the GEMM is now <TM, KS>
   with dynamic smem and a chunk-linear staging helper; KS=64 halves the
   barriers per FLOP but measured 1464 vs 2345 tok/s (-38%) — the 56KB
   footprint halves resident blocks. KS=32 stays default; `MINFER_GEMM_K64=1`
   re-tries.

nsys budgets after P5 (per 2K prefill): GEMM ~597 ms (~46 TFLOPS eff,
llama.cpp 455 ms), FA ~54 ms, convert ~56 ms, swiglu ~51 ms. Remaining
levers: GEMM k-warp specialization / mma.ptx, f16-out rms_norm+swiglu
(kills the convert pass), FA KV-split for the last 2.5×.

## Part III — Remaining roadmap

Measured budgets: f16 wmma GEMM ~35 TFLOPS (llama.cpp int8 MMQ ≈ 52
equivalent); MMVQ weight streaming 130–147 GB/s effective vs 252.7 GB/s
read-only probe (93% of the 273 GB/s theoretical); llama.cpp achieves
~197 GB/s on the same decode shape.

- **R1 — int8 MMQ prefill GEMM** (the 7B prefill parity lever): IMPLEMENTED
  2026-08-31, **opt-in** (`MINFER_MMQ=1`), not yet competitive — see the
  Part II R1 record for the design, parity evidence, and the measured
  per-shape numbers that keep the f16 8p path as the default.
- **R2 — MMVQ weight-streaming efficiency**: DONE 2026-08-31 (see the
  Part II R2 record): per-thread chunk mapping + uint4 loads; tg128
  42.2 → 45.1, decode @2K 36.7 → 38.8 (llama.cpp 47.1 / 44.9). The
  remaining tg128/@2K gaps were closed by R4 (see below).
- **R3 — small-model per-token overhead**: LARGELY DONE (Aug 31, see
  Part II). 0.5B decode was ~4.0 ms/token with ~1.6 ms of GPU floor:
  prefill single-split (A1), pinned logits readback + no clone (A2),
  prefill capture automatic (B). Remaining: clean pre/post bench numbers
  on a quiet GPU; the sampler's `apply_top_k` scratch alloc (608 KB/token
  when top_k is enabled — greedy benches never hit it) if non-greedy
  numbers justify it.
- **R4 — decode split-attention efficiency**: DONE 2026-09-01 (see the
  Part II R4 record): dim-parallel lane rewrite of the 8d flash-decoding
  pass (register accumulator, coalesced row loads, ATTN_SPLITS 8 → 32).
  @2K 39.2 → 43.2–45.1 (llama.cpp 44.9), tg128 45.1 → 47.5 (llama.cpp
  47.1). The @2K decode gap is now ~0–4%.
- **Not planned** (revisit with a concrete need): cuBLAS/cublasLt
  (evaluated → closed as 8k; 8m's custom wmma GEMM covered the f16 path),
  VMM pool, multi-GPU, node reordering, Windows, IQ/Q2/Q3 quants.

**Memory etiquette (shared-box constraint, added 2026-08-31):** this box
also runs an sglang 27B server out of the same unified pool. Kernel OOM
under pool exhaustion kills the *other* workload (happened once: the
sglang container died at 23:50 while a 124.8 GB cudaMalloc probe + suite
runs were active). Rules: no raw allocation probes on this box; check
`free -g` before suite runs (the suite transiently reserves up to ~100 GB
of the overcommitted pool); benches stay at single-process 7B scale
(~14 GB) while sglang is serving.

## Part IV — History: the pre-Phase-7 roadmap (2026-08-29) and where it ended

Everything below described the deleted imperative path (`layer_gpu`,
`forward.rs`) on an RTX 4080 Laptop with Qwen2-0.5B Q4_0: prefill 40,
decode 20 tok/s (CPU 18/15). Kept for the record; outcomes annotated.

### Root cause as diagnosed then: per-op CPU↔GPU ping pong

```
CPU path → quantize f32→Q8_0 (CPU) → cudaMemcpy H2D → CUDA kernel → sync → cudaMemcpy D2H → CPU path
```

~6 PCIe round trips × 24 layers ≈ 144 DMA operations per decode step,
2–7 ms of pure overhead. (Correct for that path; Phase 7's resident-weight
graph backend eliminated it structurally.)

### Original P0–P5 and actual outcomes

| Item | Claim then | Outcome |
|---|---|---|
| P0 full-layer GPU offload | add Q4_1/Q8_0 kernels, kill 144 DMAs, 3–4× | Absorbed by Phase 7 graph backend (weights resident, per-op dispatch, split syncs) |
| P1 GPU-side activation quantize | GPU q8_0 kernel unused, 1.2× | Landed as 8c with a measure-first gate (`nt>1 && id≤8192`); the q8_0 path LOSES 63% at 7B ffn_down (weight-bound) — a blind wire would have regressed |
| P2 fused GQA on GPU | wire `gqa_attn_f32`, 1.5× | Landed in Phase 7a/7e (`gqa_attn_f32_f16kv`); prefill attention replaced by FA tiling (8n: 20×); decode attention is 0.12 ms/token at 2K — no longer material |
| P3 cuBLAS for output projection | `cublasSgemm` "leverages tensor cores", 2× | Closed as 8k (not planned). Two errors in the original claim: `cublasSgemm` is FP32 SGEMM — tensor cores require cublasGemmEx with f16/int8; and the need disappeared once 8m's custom wmma GEMM covered large matmuls without the cuBLAS dependency |
| P4 tiled quantized matmul | llama.cpp MMQ "shared-memory tiling with Stream-K decomposition", 1.5× | First attempt judged a negative result, then REVERSED same-day (8e): the real design is integer `__dp4a` dots over q8_0 activations with a per-CC launch table (`MMVQ_PARAMETERS_GB10`: 8 warps, one output row per block) — ported in-tree as the decode MMVQ win (8e/8e②); "Stream-K" was never part of llama.cpp's MMQ. The prefill int8 version of this remains open as R1 |
| P5 CUDA graph for launch overhead | capture decode, 1.2× | Landed as Phase 7d decode capture/replay (one ~57 µs graph launch per token) + opt-in prefill capture (8g②). The original "2,000+ launches (…× ~14 heads)" miscounted — heads don't multiply launches; the true figure is ~95 nodes/layer × 28 layers ≈ 2.7K, same order |

### Implementation order as drawn then

```
P0 (layer_gpu) ─→ P1 (GPU quantize) ─→ P2 (GPU attention)
                                      ↘
                                       P3 (cuBLAS) ─→ P4 (tiled MMQ) ─→ P5 (CUDA Graph)
```

All six landed in some form by 2026-08-31 — none via its original
mechanism except P0's idea. The current forward plan is Part III.
