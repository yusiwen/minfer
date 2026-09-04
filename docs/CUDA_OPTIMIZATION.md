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

### P6 r17: wide warp remap 32 od-rows x 64 tokens — instructions −5.8%, wall NEUTRAL (REVERTED, 2026-09-03)

The r13/r15-noted warp-tile shape lever, implemented exactly as a pure
index remap of `mmq_raw_wide_nt_kernel` (block tile 128×128, staging, qb8
layout, sds/sdm, launcher smem all unchanged): `wn = warp&1` (i0w =
wn*64), `wm = warp>>1` (j0w = wm*32) — each warp owns llama.cpp's
32-od-row × 64-token shape; A-frags h<4 (`i0w + h*16 + lane>>2`), B-frags
nh<4 via TWO ldmatrix.x4 (`j0w + half*16 + …`), 16 chains mma over (nh,h)
both 4, sum[64] budget unchanged (`idx = nh*16 + h*4 + l`); the r15
dma-fold's da/sa uint2 reads shift by i0w, dsv/dmv float4 covers nh<4.
Parity green at KD=4, KD=8 and default on the first build.

Measured (7B @2630 tok, interleaved 3x vs the HEAD binary): KD=4 remap
1231/1250/1257 vs baseline 1246/1230/1239 (medians 1250 vs 1239, +0.9%);
KD=8 1287/1301/1303 vs 1311/1263/1288 (medians 1301 vs 1288, +1.0%) —
noise-band, far under the ≥1350 bar. ncu q-proj (KD=4, grid (21,28)):
warp instructions 8.68 → 8.18 M/GMAC (−5.8%; per-block LDSM.x4 per chunk
72 → 48: A 64→32 as the 8x A-redundancy halves, B 8→16 as B becomes 2x
shared), duration 2.262 → 2.256 ms (−0.3%) — the instruction cut did not
convert to wall time. SpeedOfLight: Compute (SM) 31.5 → 29.75% (still
stall-bound), SM Active Cycles 4.61 → 5.04 M (+9.2%): per warp the 4 A +
2 B LDSMs now feed 16 chains with the B side 2x-shared, and issue
efficiency dropped — the stall profile moved the wrong way, offsetting
the leaner instruction stream. Third independent confirmation of the r15
reading: pure per-MAC instruction cuts in this kernel pay ~0 wall time
while SM% sits at ~30; the binding lever is latency/stall structure
(prefetch, occupancy), not op count.

Kernel reverted to HEAD `708067d` (r16 narrow fold kept; cmp-verified);
the remapped kernel preserved at /tmp/cuda_kernels_r16remap_variant.cu.

### P6 r18: load-time B pre-expansion — staging becomes a bulk copy: wall NEUTRAL (REVERTED, 2026-09-03)

Phase-7 task-1 hypothesis: the wide kernel's per-k-t global->smem B round trip
(raw 144B superblock staged, then ALU-expanded to qb8 [8][128][48]) sits
between compute phases at 8 resident warps and is part of the stall structure
r13/r17 fingered. Fix: materialize the expansion ONCE at weight registration —
(a) EB, per-k int8 per-superblock copy `[sb][od][8][32]` (256B per
(superblock, row), slot bytes element-ordered raw nibbles 0..15 — exactly the
old in-loop expansion's output), laid out so one k-tile's 128 rows x 8 slots
are ONE contiguous 32KB global range; (b) SB, per-(chunk, row) (dv, mv)
float2 pairs `[chunk][od]` replacing the per-chunk 144B header parse +
get_scale_min_k4 unpack. The wide kernel's B + scale staging became pure bulk
copies (2048 16B LDG/STS per superblock, zero dequant ALU); the raw tensor
stayed registered for every other path. One device-side `expand_q4k_kernel`
launch + stream sync per tensor at load; `register_weight_q4k_expanded`
keyed the buffers by raw wptr in a new `mmq_expanded` registry; wide launch
required them, falling back to raw-W narrow staging otherwise. Measured
memory cost on 7B q4_k_m: +5.8 GB device (od·id + 8·od·id/32 bytes; free
26.1 -> ~20 GB of 130.6 — acceptable as predicted, actuals logged per tensor).

Implementation landed parity-green on the first build (8-shape sweep, KD=4 +
KD=8 + default; expanded buffers also registered in the parity fixture so the
wide kernel stayed under test).

Measured (7B @2630 tok, interleaved 3x vs the 1e0dded baseline binary,
same session — NOTE: the box ran ~9% below the r15/r17 session's absolute
numbers, baseline KD=8 1155-1171 vs 1295 then; only the interleaved deltas
are meaningful):

| config | baseline | r18 expanded-B | Δ (medians) |
|---|---|---|---|
| wide KD=8 (default) | 1170.9 / 1164.7 / 1155.5 | 929.8 / 1181.7 / 1175.0 | **+0.9%** (noise-band) |
| wide KD=4 | 1145.7 / 1141.3 / 1130.6 | 1071.6 / 816.2 / 907.5 | **−19% median, high variance** |
| narrow (control) | 463.3 / 429.9 | 444.5 / 441.5 | noise |

ncu q-proj (launch 1, nt 2630, od=id=3584, grid (21,28), per GMAC = 33.78e9):

| metric | baseline KD=4 | r18 KD=4 | baseline KD=8 | r18 KD=8 |
|---|---|---|---|---|
| warp instructions | 8.68 M | 8.65 M (−0.3%) | 8.49 M | 8.46 M (−0.3%) |
| duration | 2.288 ms | 2.335 ms (+2.1%) | 2.155 ms | **2.059 ms (−4.5%)** |
| Compute (SM) | 31.1% | 30.4% | 32.4% | 33.7% |
| Memory Throughput | 44.2% | 36.5% | 46.2% | 43.0% |

Readings:
1. The staging-ALU cut is instruction-neutral (−0.3%): the expansion ALU was
   ~0.1% of the per-block warp stream — the r15/r17 "op cuts pay ~0 at
   SM% 30" rule extends to STAGING op cuts.
2. The latency win is real but small and config-dependent: KD=8 −4.5%
   per-kernel duration converts to +0.9% wall (the GEMM is one of ~200
   launches; only q4_K matmuls benefit). KD=4 REGRESSES +2.1% per-kernel:
   with the restage-once-per-superblock guard, the bulk copy reads 256B/row
   where the ALU path read 144B — B-side staging bytes ~2x — and at KD=4
   that byte excess outweighs the removed latency (also the likely source of
   the erratic KD=4 wall runs).
3. Fourth independent confirmation of the r17 reading: the stall structure
   is NOT meaningfully reduced by removing the B-expansion work between
   compute phases — SM% stays 30-34, still stall-bound. The binding
   latency is elsewhere (A-side staging, scoreboards on the mma chain
   itself), and the quantified ceiling of this lever was ~5% per-kernel.

Bar was >= 1350 tok/s; decisively missed. Reverted to HEAD `1e0dded`
(cmp-verified). The complete variant is preserved for reuse — the EB/SB
materialization machinery is a prerequisite for any future L2-persistence
experiment on expanded weights:
- /tmp/patch_p7_t1.py (anchored patch script: kernel + registry + loaders)
- /tmp/cuda_kernels_p7t1_variant.cu (patched kernel file)
- /tmp/perf_p7t1_ab.sh, /tmp/ncu_p7t1.sh (A/B + ncu harnesses)
- /tmp/minfer_phase7/ncu_r18exp_kd{4,8}.csv, ncu_base_kd{4,8}.csv

### P6 r19: weight L2 residency — __ldg NEUTRAL, L2 persisting window CATASTROPHIC (REVERTED, 2026-09-03)

Phase-7 task-2, on the HEAD (r16-state) wide kernel — the 21x token-tile
re-reads (and cross-launch re-reads within a layer) of the same weight bytes
should hit L2 instead of DRAM. Two levers, one build:

1. **`__ldg` (read-only path) on the wide kernel's weight-side loads** — the
   B-staging superblock `uint4` loads and the scale-header `uint16` loads
   (read-only for the kernel's lifetime; parity-neutral by construction).
2. **`MINFER_MMQ_L2WIN=1`: cudaAccessPolicyWindow** on the raw q4_K weight
   byte range, set per wide launch (hitRatio 1.0, hitProp Persisting,
   missProp Streaming), persisting carveout raised to
   `persistingL2CacheMaxSize` once per process.

Parity green with and without the window (KD=4 + KD=8).

Measured (7B @2630 tok, interleaved 3x, same noisy session — box drifted
−9% vs the r15/r17 absolutes; only intra-session deltas count):

| config | baseline 1e0dded | r19 __ldg | r19 __ldg + L2WIN |
|---|---|---|---|
| wide KD=8 | 780.0 / 1061.7 / 1252.4 | 700.7 / 1278.6 / 1282.3 | **453.6 / 535.4 / 548.2 (−50%)** |
| wide KD=4 | 1195.8 / 1144.4 / 1240.0 | 1201.9 / 1219.7 / 1229.9 (+2.0% med) | **557.6 / 575.4 / 577.2 (−50%)** |

Readings:
1. `__ldg` alone: NEUTRAL (+2.0% KD=4 median, KD=8 lost in box noise, no
   consistent gain). Expected in hindsight — the weight tiles are already
   re-read from L2 within a launch (r13 measured L2 read traffic only 1.6x
   llama.cpp per MAC), and nvcc's const-`__restrict__` handling was likely
   already emitting non-coherent loads; the ncu "L2 Cache Throughput"
   37-46% SOL readings show L2 was not the constraint.
2. The persisting window at hitRatio 1.0 is CATASTROPHIC (−50%, 6/6 runs
   consistent): marking 12.8-34MB per weight persisting fills the carveout
   with weight lines while the C stores (37MB per GEMM), activations, and
   KV traffic lose normal L2 — and every re-mark churns the carveout. The
   access-policy window mechanism WORKS (deterministic effect), just not in
   this direction at this ratio. Untested cheaper variants (out of budget):
   hitRatio ~0.25, windowing only the small q/k/v weights, or
   `cudaCtxResetPersistingL2Cache` between layers.

Bar was >= +5% over the Task-1 state; the only consistent signal was −50%.
Both levers reverted to HEAD `384b3d9` (cmp-verified). Preserved:
- /tmp/patch_p7_t2.py (anchored patch script), /tmp/cuda_kernels_p7t2.cu
  (patched kernel file), /tmp/perf_p7t2_ab.sh, /tmp/perf_p7t2_ab.log.

### P6 r20: residual attribution — the issue-efficiency gap is long-scoreboard exposure in the staging LDG→STS chains, not barrier structure (LANDED split-phase staging, +7.1% KD=4, 2026-09-03)

Session question: WHY is llama.cpp `mul_mat_q` at issue 0.41/sched vs our
0.25 at identical occupancy (2 active warps/sched, 1 block/SM, 8 warps), and
is the cause portable? Method: exact-512-token prompt (deterministic word
text tuned to land on exactly 512 BPE tokens,
/tmp/minfer_phase7/prompt512.locked, 2753 chars), full 15-class
per-issue-active stall set + cycle counters on BOTH q-proj kernels at matched
nt=512, plus minfer nt-2630 for the r13 tie-in. All prior minfer captures
were nt-2630 whole-tensor launches vs llama nt-512 ubatches.

Structural facts established from source (llama.cpp @ ca3d5a3e1, matches the
profiled `mul_mat_q<12,128,0>` = <GGML_TYPE_Q4_K, J=128, fallback=0>):
- 256 threads (8 warps), I=J=128 tiles, `MMQ_ITER_K = 256`, occupancy-1
  launch bounds — same tile shape and occupancy as ours.
- Per 256-k iteration: `load_tiles (W) → stage y-half-1 → barrier →
  vec_dot(0..128) → barrier → stage y-half-2 → barrier → vec_dot(128..256) →
  barrier` = **4 barriers per 256 k = 2 per 128 k — identical barrier
  density to ours** (2/kt of 128 k). Barrier COUNT is at parity; the old
  "their staging is less synchronous" hypothesis is false at source level.
- nt-512 q-proj launch: nty=28 × ntx=4 = 112 tiles, stream-k efficiency
  77.8% < 90% → **grid (48,1,1) persistent blocks + `mul_mat_q_stream_k_fixup`**
  (grid (48,4,1), +34 μs). Ours: grid (4,28) = 112 short blocks, 2.33 waves
  (≈3 × 448 k-cycles/block). Per-tile wall: minfer 211 μs vs llama 113 μs.

Phase-1 table (q-proj class, nt-512, per-issue-active warp ratios; llama =
launch 0 of 8, identical at launch 6):

| metric | minfer <4> | minfer <8> | llama.cpp <12,128> |
|---|---|---|---|
| duration q-proj (μs) | 632.4 | 609.6 | 263.6 |
| issue /cyc/sched | 0.16–0.26 | 0.20 | 0.42 |
| eligible /cyc | 0.22–0.28 | 0.28 | 0.64 |
| warps active /cyc | 2.00 | 2.00 | 2.00 |
| warp inst / tile (k) | 499 | 488 | 356 |
| IMMA mma-inst | 1,605,632 | 1,605,632 | 1,605,632 |
| long_scoreboard | **6.22** | 5.76 | **1.15** |
| wait | 0.63 | 0.57 | 0.58 |
| barrier | 0.26 | 0.17 | 0.19 |
| short_scoreboard | 0.34 | 0.53 | 0.12 |
| not_selected | 0.41 | 0.40 | 0.50 |
| math_pipe_throttle | 0.36 | 0.34 | 0.47 |
| mio_throttle | 0.11 | 0.28 | 0.25 |
| lg_throttle | 0.33 | 0.60 | 0.09 |
| dispatch_stall | 0.29 | 0.28 | 0.31 |
| LDS bank conflicts | 3,211,264 | 3,211,264 | **42** |

Readings:
1. **The gap carrier is long_scoreboard, not barriers.** Named-stall excess
   ≈ +5.2 warps/issue-active, of which longsb = +5.07 (97%); barrier delta is
   +0.07. In warp-time terms: minfer warps spend ~86% of resident cycles
   stalled on global-load latency (6.2 of 7.7 warps-per-issue-active), llama
   ~24% (1.15 of 4.76 — its identity closes exactly: 4.76 = 1 + 4.80).
2. **The r13 stall table was double-distorted**: it compared the PRE-r14
   kernel (before ldmatrix B-frags/r15 rank-1 fold) at nt-2630 against llama
   at nt-512, and it lacked the full stall set. Current kernel at nt-2630:
   issue 0.22, longsb 6.26, barrier 0.40 — barrier was never 0.62-vs-0.18
   caused by structure; the nt-2630 capture's mix of GEMM classes blurred it.
3. **Tensor work is EXACTLY at parity**: IMMA = 1,605,632 on both sides for
   the same q-proj GEMM (mma.m16n8k32 count; 4096 MACs each). The 2.40x
   duration gap = 1.40x instructions-per-tile (r13/r15 residue) × 1.7x
   issue-efficiency.
4. Cycle-counter calibration (r20b): ncu locks 2.14 GHz; smsp active cycles
   1.12M ≈ 2.33 waves × 448k block cycles; issue_active.sum ==
   inst_executed.sum EXACTLY (no dual-issue). A 0.16-vs-0.26 capture
   discrepancy on the SAME kernel was replay-pass clock variance — the
   0.25-vs-0.41 headline from r13 survives (0.26 vs 0.42 clean re-measure).
5. **PC-sampling source attribution** (SourceCounters, 51,032 samples):
   minfer's top stall site is a single `STS [R25], R32` in the A-qs staging
   loop = **16.7% of ALL warp stalls**; the STS/LDG/unpack-ALU staging path
   (STS 23.6% + IMAD 13.5% + SHF 8.8% + LOP3/I2F 6.3%) holds ~half the
   samples, compute (IMMA+FMUL+FFMA) 16%. llama's profile is spread thin
   (top site 6.4%, compute 41%, staging-path ~27%, LDS conflicts ~0).
   Mechanism: the interleaved `LDG.32 → address ALU → STS.32` chains let
   in-order issue stall at the FIRST store of each 4-deep unroll batch —
   one full memory latency (~600 cycles) per batch, ~4-5 batches per kt,
   while the A-tile fetch is also 8-sector-scattered (each 4B load spans
   token stride 4480B → 62.5% sector efficiency, 128 unique sectors per
   warp-kt vs llama's ~60-120 with L1 reuse).

Phase-3, portable fix LANDED (`split-phase A staging` in
`mmq_raw_wide_nt_kernel`): issue ALL A-staging global loads into register
arrays first (`av[KDR*4]`, `dv[]`/`sv[]`), then store to smem — identical
addresses, traffic, and ~identical instruction count; only the dependency
schedule changes. The d/ssum loop folds into the same schedule. Gates:

- parity: `cuda_prefill_mmq` green KD=4 AND default KD=8;
- perf (2630 tok, interleaved 3x vs baseline binary @1e0dded = same wide
  kernel as HEAD): **KD=4 1230.4 → 1317.7 tok/s (+7.1%), KD=8 1275.8 →
  1319.9 (+3.5%)**, narrow control +1.2% (noise). The session's ≥1350 bar
  was NOT met — landed anyway as strictly-positive and reproducible 6/6;
- ncu re-check (nt-512 q-proj): **longsb 6.22 → 2.92 (−53%), duration 632 →
  555 μs (−12.2%), warp-inst 55.9 → 46.8 M (−16%: ptxas schedules the
  register batch leaner)**, IMMA unchanged; the freed stalls moved to
  **lg_throttle 0.33 → 2.38** — the staging is now LSU-queue/request-count
  bound, NOT latency bound;
- suite 166/0/3; greedy 16-token output identical (default vs gated path,
  wide prompt).

Verdict: the stall-gap carrier is proven (long-scoreboard, 97% of the named
excess, sampled inside the staging LDG→STS chains), and the portable fix is
landed. But the wall return on removing 53% of longsb was only
+7.1%/+3.5%: the freed warp-cycles re-saturate on lg_throttle (LSU queue /
request count) — staging latency and staging request count are serially
co-binding, so the latency fix alone converts one bound into the other.
What is NOT the cause (all measured, not assumed): barrier structure
(density already 2/128k on both sides; barrier stalls 0.26-0.39 vs 0.19),
launch shape (stream-k vs 3-wave tiling a wash at these shapes; their fixup
kernel costs +34 μs), occupancy (2.00 warps/sched both), tensor work (IMMA
identical), and memory bytes (r13). Remaining gap, in measured order:
(1) staging request count — the coalesced block-linear A-staging rewrite
(uint4 slices over the KDR×160B contiguous per-token regions, 5-deep
per-thread batches, d/ssum folded in) cuts unique sectors 128 → 80 per
warp-kt (−37%) at 100% efficiency and directly attacks lg_throttle;
requires id % 256 == 0 for 16B-aligned uint4 slices, which the raw-MMQ
dispatch guard `(id / 32) % 8 == 0` already guarantees; (2) the remaining
1.17x instruction surplus per tile (499k → 418k after this patch vs llama
356k) — r13/r15's compute-loop levers, plus extending split-phase to the
B-expansion loop; (3) 3.2M LDS conflicts per launch vs their 42 (the qb8
48B-slot stride — re-tile so the compute-loop LDS.64/128 pattern lands
conflict-free).

Revert anchor: /tmp/cuda_kernels_pre_r20.cu (byte-exact pre-r20 backup,
md5 2504ac93). Patch script: /tmp/patch_r20_stage.py (line-anchored,
count-verified). Captures: /tmp/minfer_phase7/r20_*.csv (+ .ncu-rep PC
samples), prompt512.locked; harnesses /tmp/ncu_r20_matched.sh,
/tmp/ncu_r20b_cycles.sh, /tmp/ncu_r20c_src.sh, /tmp/perf_r20_ab.sh;
parser /tmp/parse_ncu_r20.py; SASS dumps /tmp/minfer_phase7/minfer.sass,
r20_src_{m512,l512}_sass.csv.

### P6 r21: coalesced block-linear A staging — the premise's mechanism CONFIRMED (sectors −29%, lg_throttle −90%) but wall −2%: instruction overhead, not LSU pressure, is the binder (REVERTED, 2026-09-03)

Session question (r20's "remaining gap (1)"): does replacing the wide
kernel's 8-sector-scattered per-(chunk, u-word) A staging with warp-strided
uint4 loads over the contiguous per-token KDR×40B regions (d/ssum folded
out during the store phase, smem addresses unchanged, split-phase schedule
kept) turn the lg_throttle co-binding into wall clock? The coalescing
premise verified clean in source: the pad40 q8 layout is token-major
contiguous per token, and the raw-MMQ dispatch guard `(id / 32) % 8 == 0`
makes the token stride nb32·40 a multiple of 320B, so every per-token
region is 32B-sector aligned → 5·KDR/2 full-sector uint4 slices per token,
lane-strided in 16-token/warp batches (80 sectors/warp-kt at KDR=4),
nb32 % KDR == 0 ⇒ no partial k-tiles ⇒ per-token validity is exactly
`tok < nt`. Implemented as a mmq_raw_wide RAW_STAGE rewrite; parity took
three fixes (per-lane slice count is 5·KDR/4, not 5·KDR/2 — the excess
iterations re-staged the next warp's tokens and warp 7 wrote past qa8 into
sda_q; `idx` must be built from `lane`, not `threadIdx.x`; and the global
token must include i0 — only the sweep's grid.x=2 shape (nt=256) caught
it). A byte-exact CPU simulator of both staging forms (/tmp/sim_r21.py)
confirmed identical qa8/sda_q images across all sweep shapes for KDR=4
and 8.

Measured (2630-tok prompt, interleaved 3x, q-proj class nt-2630 ncu):

| metric | r20 (pre) | r21 (post) | |
|---|---|---|---|
| global-load sectors / launch | 79.35 M | 56.67 M | **−28.6%** (premise holds) |
| lg_throttle /issue-active | 1.42 | 0.14 | **−90%** (lever worked) |
| mio_throttle / math_pipe | 0.37 / 0.39 | 0.00 / 0.08 | gone |
| long_scoreboard | 1.23 | 1.24 | unchanged |
| **wait** | 0.59 | **1.24** | +110% (new binder) |
| short_scoreboard | 0.23 | 0.52 | +126% |
| warp-inst / launch | 245.9 M | 274.1 M | **+11.5%** |
| duration (q-proj) | 1971 μs | 2186 μs | **+10.9%** |
| wall KD=4 / KD=8 | 1312 / 1311 tok/s | 1282 / 1302 | −2.3% / −0.7% |

Reading: the stall MASS is conserved — the throttle classes r20 named
co-binding collapsed exactly as predicted, but the freed issue slots
re-saturated on `wait` (fixed-latency dependency chains) +
short_scoreboard (smem): per 16B slice the new form pays an idx/spt +
s%5 decode (mul-hi chains), a 5-way divergent store branch with byte
extraction from the uint4, and 2×U16 sda stores — where r20's
LDG.32→STS.32 flowed the loaded register straight to smem. +11.5%
warp-inst at this occupancy maps ~1:1 onto +10.9% duration. Conclusion:
**sector efficiency and LSU-queue request count are NOT the wall-clock
binder for the wide kernel's staging at nt-2630; warp-instruction count
is.** r20's "remaining gap" list re-ordered: (1) is dead — the 128→80
sector win is real but buys nothing at this shape; the live levers are
the r13/r15 compute-loop instruction surplus (499k→418k vs llama 356k
per tile) and the 3.2M LDS conflicts. A hybrid that keeps r20's
pass-through LDG.32→STS.32 but folds the separate d/ssum loads into the
qs loads (shared sectors, no byte extraction) is the only remaining
staging idea worth trying, and its ceiling is small (the d/ssum loads
are ~15% of staging requests).

Revert: src/cuda_kernels.cu restored byte-exact to HEAD 9819410
(md5 a037f0c7286abdae41ef0ea73481ff31 = /tmp/cuda_kernels_pre_r21.cu),
parity re-verified green after revert. Artifacts: patch script
/tmp/patch_r21_coalesced_a.py (line-anchored, count-verified), staging
simulator /tmp/sim_r21.py (byte-exact r20-vs-r21 smem image diff), perf
harness /tmp/perf_r21_ab.sh + log /tmp/minfer_phase7/r21_perf_ab.log,
ncu captures /tmp/minfer_phase7/r21_{post,pre,post2,pre2,post_wi,pre_wi}.csv
(+ .logs), parser /tmp/parse_r21_metrics.py.


### P6 r22: conflict attribution + qa8 XOR swizzle LANDED (op_ld 16.86M -> 0, KD=8 +1.4%) / d/ssum stream fold NEGATIVE (REVERTED, 2026-09-03)

Step 0 — attribute the residual conflicts (r21's re-ordered lever list).
There is NO dedicated LDSM conflict metric on GB10 (checked
`--query-metrics`); LDSM conflicts roll into
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`. q-proj
launch, nt-2630, KD=4, grid (21,28), baseline bd6fd4f:

| metric | value |
|---|---|
| op_ld conflicts | **16,859,136** |
| op_st conflicts | 6,498,453 |
| LDSM warp-inst (`sm__inst_executed_op_ldsm`) | 4,741,632 |

The op_ld number is EXACTLY the A-side qa8 ldmatrix 2-way: 4.74M LDSM
x4 inst x 4 phases x 1 extra wavefront per 2-way phase = 16.86M (the
r14-era 499K/GMAC residual). op_st decomposes as the qb8 staging
stores (4 p-lanes per row hit the same bank group: slot offset
6144B*s vanishes mod 128B) + sds float2 stores (2-way, lanes 16 apart)
— stores, not load-path. Lever 1 targets the A-side; op_st left alone.

**Lever 1 (KEPT): XOR-swizzled qa8, zero smem growth.** qa8 keeps its
32B rows but consecutive 4 rows form a 128B super-row and the 16B
granule index is XORed with the super-row index:
`gr(row, h) = ((row & 3) * 2 + h) ^ ((row >> 2) & 7)`,
`byte = (row & ~3) * 32 + gr * 16 + intra * 4` — the same map on the
staging stores and the ldmatrix loads. Rows 4 apart (the 2-way pair)
land on distinct granule phases; every ldmatrix phase gets 8 distinct
bank phases. Verified standalone first (/tmp/ldm_a_swizzle_test.cu,
r14's ldm_b approach): identical A-fragment distribution vs plain LDS
on the logical layout, and ncu on the test kernel shows
524,288 -> 0 op_ld conflicts for the same 131,072 LDSM inst. The
staging STS pattern stays conflict-free under the swizzle (4 consecutive
rows cover all 8 granule phases exactly once). First integrated cut
hoisted lane-only terms per chunk — still −2.4%/−0.9% wall: the
per-ldmatrix XOR/SHL/IADD address ALU ate the freed wavefronts. Final
form precomputes all 8 A-frag offsets per thread ONCE
(`G[g] = g*512 + (lane&12)*32 + ((gr ^ lc ^ ((g&1)<<2)) << 4)`, all
lane-invariant — verified standalone in the same test): each ldmatrix
address is ONE IADD (`p = qat + G[g]`), below baseline's ~3 ops, and
registers DROP below baseline (129/145 vs 141/149, zero spill).

Measured (2630-tok, interleaved 3x same-slot pairs vs /tmp/minfer_pre_r22):
KD=8 (default) 1329.6 vs 1311.8 median = **+1.4%**, 3/3 pairs positive
(1327.0-1330.1 vs 1305.4-1315.3); KD=4 1309.0 vs 1315.3 median = −0.5%
(3/3 marginally negative, inside the ±1% drift band the narrow control
showed) — kept: default path positive, mechanism complete. ncu on the
final build: **op_ld conflicts 16,859,136 -> 0**, op_st 6.47M unchanged
(out of scope), same 4.74M LDSM inst. Parity green KD=4 and KD=8; suite
166/0/3; greedy-32 token identity vs the default path (timing lines
only); smem budgets unchanged (KD=4 73,728B / KD=8 98,304B). The
combined >=1350 bar was NOT reached — per the stop conditions the
positive lever is kept and this entry documents the split outcome.

**Lever 2 (REVERTED): d/ssum fold into the A pass-through** — r21's
named "only remaining staging idea". Implemented as a single coalesced
9-word-per-chunk stream (w=0: d f16|pad, w=1..8: qs; word index =
9*chunk + w, x = tid + i*256) with the w=0 holder grabbing the adjacent
ssum word (+36, one extra 4B load inside the pass-through) and packing
d|ss into sda_q in-register; the separate scattered d/ssum loop is
deleted. Two enumeration variants (per-iteration x/9 magic-div; then a
division-free (+28/+4 mod 9) incremental counter): BOTH measured
−19-20% wall (KD=4 ~1050 vs ~1302, KD=8 ~1039 vs ~1303, 3x interleaved,
parity green). Root cause: the premise "the d/ssum loads are scattered
sector waste" is void — at lane stride nb32*40 they hit the SAME cache
lines the qs pass of the same chunks fetched microseconds earlier, so
they are L1 HITS; folding buys ~no sector traffic while the flat
enumeration pays ALU (mod-9 chains) plus a branchy in-loop sda
store that breaks r20's single deep LDG batch, and the extra register
pressure pushed ptxas to 255 regs + 112B spill at KD=8.
Lever-2's stall-structure lesson repeats r21's: the wide kernel's
staging is instruction/issue-bound, and request-count "savings" that
add ALU lose. The staging book is now closed: the live levers are the
r13/r15 compute-loop instruction surplus and the (now zero) LDSM
conflict headroom is spent.

Artifacts: standalone swizzle test /tmp/ldm_a_swizzle_test{,2}.cu +
ncu script /tmp/ncu_ldmtest.sh
(/tmp/minfer_phase7/ldmtest_conflicts.csv), attribution captures
/tmp/ncu_r22_attr.sh ->
/tmp/minfer_phase7/r22_attr_{pre,swz,final}_kd4.csv (+ .logs), recheck
/tmp/ncu_r22_recheck.sh, perf harness /tmp/perf_r22_ab.sh + logs
/tmp/minfer_phase7/r22_perf_ab.log (combined), r22_swonly_perf_ab.log
(swizzle-only), r22_goff_perf_ab.log (final G-offset form).


### P6 r23 — B-line wall decomposition: the default f16 path's full-graph split (2026-09-03)

Pivot session: with the q4_K MMQ kernel campaign at r22, decompose the
ENTIRE default-path (f16 wmma GEMM) prefill wall to see what is left
OUTSIDE the GEMM work — and where the non-q4_K weight types actually sit.
Method: `nsys profile --trace=cuda` on the 7B q4_k_m 2659-token prefill
(-n 1, default path, HEAD 317c2db), per-launch csv bucketed into op
classes + GEMM-by-weight-type via the deterministic launch sequence
(grid gy = od/128: q/o/down 28, k/v 4, gate/up 148; walk order
q,k,v,o,gate,up,down; weight types from a GGUF metadata census,
/tmp/gguf_types.py). ncu SOL pass on the first layer's q,k,v,o,gate,up,
down + FA + swiglu (48 SMs on GB10). Co-tenant contamination: a 77GB
sglang::scheduler burst cost one run ~74ms of outlier kernels (down L1/L2
22.3/55.6ms, gate 14.1/19.8ms vs the flat ~8.5/10.2ms) — outlier-cleaned
numbers below; interleaved medians for any wall claims. Quiet-window
default path: 2285 tok/s median (3x tight: 2278-2288).

**Graph structure discovery (worth keeping in mind):** the prefill is 27
full layers + layer 27 attention, then an nt=1 TAIL — the last token's
FFN and lm_head run as decode-shaped ops (gather-row, 1-token norms,
MMVQ gate/up/down + 2.5ms q6_K lm_head over 152064 rows). The full
[nt,vocab] logits GEMM does NOT exist in the prefill wall; the q6_K
lm_head (1.45 TMAC = 9.6% of model MACs) is therefore already tail-priced.
193 gemm_f16 launches = 27x7 + 4 (layer 27 = q,k,v,o only), each preceded
1:1 by a convert_f32_f16 (A-side f32->f16).

**Op-class table (profiled run span 1212 ms, GPU busy 1202.7 ms, host
gaps 9.3 ms = 0.8%):**

| op class | ms | % span | ncu limiter | plausible lever | est. wall gain |
|---|---:|---:|---|---|---:|
| GEMM f16 wmma — gate+up (q4_K) | 458 (clean ~425) | 37.8% | **mem SOL 85%** (L1/L2 B-stream of 16-bit weights), SM 38%, occ 50% | MMQ raw-byte B (r6–r22 campaign) | at MMQ parity: ~+25% tok/s |
| GEMM — down (14 q4_K + 13 q6_K) | 334 (clean ~277) | 27.5% | mem 74%, SM 34%; **runs 20% slower per-MAC than gate/up** (10.2 vs 8.5 ms, same 180.5 GMAC, same kernel — cause unexplained; K=18944-deep A loop) | same MMQ; ncu A/B of the asymmetry | +4% if down matches gate/up |
| GEMM — q+o (q4_K) | 91 | 7.5% | latency (SM 17–36%, mem 38–80%, occ 44–49%) | — | small |
| GEMM — k+v (q4_K/q6_K) | 17 | 1.4% | latency (od 512 -> 168 blocks) | — | — |
| attention (FA prefill) | 87 | 7.2% | **occupancy 16.7%** (69.4KB smem -> 1 block/SM); SM 25%, mem 31% — nothing saturated | r23 tried tile-shrink (below): fixed costs ate it; needs llama.cpp-style structure (their FA ~1.2 ms/layer vs our 3.1) | +2–4% |
| convert f32->f16 (A-side) | 73 | 6.0% | DRAM peak (245 GB/s); pure f16-path tax (llama.cpp pays zero — MMQ quantizes in-kernel) | in-kernel convert TRIED (P6: −8%); smem-mirror variant predicted negative (r21/r22 lesson: staging is issue-bound) | ≤ +3% |
| swiglu | 68 | 5.6% | DRAM peak (245 GB/s) — at bandwidth ceiling | fusion into GEMM epilogue (hard) | ≤ +2% |
| add (residual) | 24 | 2.0% | bandwidth | f16-out chain (dtype plumbing) | +1% |
| rms_norm | 23 | 1.9% | 177 GB/s — BELOW the 245 GB/s elementwise peak | block-size/vector tweak | +1% |
| rope + bias + kv store + misc | ~25 | 2.1% | bandwidth | — | — |
| host gaps | 9 | 0.8% | launch overhead | prefill CUDA-graph capture | +0.5% |

**GEMM by weight type:** q6_K-weight GEMMs = 194.9 ms (16.1% of span;
24% of GEMM MACs: 14/28 attn_v, 14/28 ffn_down; lm_head excluded by the
nt=1 tail). Per-MAC there is NO q6_K penalty — same f16 kernel, same
~10.2ms down instances after outlier removal (the load-time w16 dequant
cache erases the type). The "non-q4_K GEMM path" is a non-issue in the
default path; the only type story is the GGUF census: 169 q4_K + 29
q6_K (14 attn_v, 14 ffn_down, output.weight), no q8_0/q5_K in this GGUF.
Per-kernel efficiency: gate 43.4 / up 41.7 / q 41.9 / o 42.0 / down 34.3
(outlier-cleaned) / k+v 32.5 TFLOPS.

**llama.cpp same-model split (nsys, same noisy window — ratios only):**
GEMM 597 ms total (MMQ 39.5 µs/GMAC incl. stream-k fixup) vs our f16
59.5 µs/GMAC + the 73 ms convert tax; A-quantize 48 ms (vs our convert
73); silu 66 (= ours); FA ~34 ms (1.2 ms/layer — 2.5x ours); add 17,
norms 10, rope 6. Their total GPU busy ~805 ms ≈ their quiet-window wall
(2630 tok @3371). The prefill gap IS the GEMM byte-width gap: raw q4_K
B-stream (4.5 bit/w) at 85% SOL beats an f16 B-stream (16 bit/w) at 85%
SOL by ~1.5x per MAC — which is exactly the r6–r22 campaign's premise.
Updated campaign math (2659-tok wall 1160 ms): MMQ at llama.cpp GEMM
parity = quantize ~130 + GEMM 597 = 727 ms vs f16 900 + 73 = 973 ms ->
wall ~914 ms -> ~2900 tok/s (r6's estimate of 2670 was conservative).

**r23 experiment (REVERTED, bar not met): FA_TKV 64->32 occupancy lift.**
Design: FA prefill is smem-capped at 1 block/SM (69.4KB: Q 17.4 + K/V
34.8 + Sf 16.4 + state 0.8). TKV=32 + a corrected launcher formula
(Q at TQ rows, K/V at TKV rows — the old one reserved 3xTQ rows) gives
43,776B -> 2 blocks/SM; the QK^T warp decomposition generalized to
kvw = FA_TKV/2 mma chunks (bit-identical op order at TKV=64).
Gates: cuda_prefill 7/7 green; greedy-32 token identity vs pre-change
binary = timing lines only (a REAL bug was caught on the first cut:
with TKV=32, lanes 16–31 own nonexistent columns and the mask
`kt+c0 < kv_end` alone let them read the next row's scores -> garbage
generation; fixed by adding `c0/c1 < FA_TKV` to the mask — keep in mind
for any future tile reshrink).
Result: ncu occupancy 16.7% -> **32.68%** (2 blocks/SM materialized,
43,776B/block), SM/mem SOL 25.2/30.6 -> 47.8/47.8, kernel duration
3.45 -> 3.22 ms (-6.7% ncu-serialized) — the 2x k-loop fixed costs
(barriers, per-tile alpha rescale, half-wasted softmax lanes) consumed
the latency-hiding gain. Interleaved 3x wall: pre 2284.7 vs post
2277.4 tok/s median = **-0.3%** — far below the +3% bar, REVERTED
(git checkout, binary rebuilt at HEAD).
Consequence: FA's 2.5x/layer gap to llama.cpp is structural (their
flash_attn_ext_f16<128,128,..> keeps 128-wide KV tiles with warp
specialization inside the smem budget); a TKV shrink alone cannot
collect it. Next FA lever, if any: reduce Qs (Q fragments in registers)
to fit double-buffered 64-wide KV staging in <=50KB — same trap the P5
note documented (double-buffer at padded stride = 99KB+).

**Priority queue out of this session:** (1) the MMQ campaign itself —
unchanged, now with a byte-exact wall-impact estimate; (2) the down-GEMM
20% per-MAC asymmetry vs gate/up (cheap ncu A/B, up to +4%); (3) FA
prefill structure (needs a redesign, not a knob); (4) rms_norm bandwidth
(+1%, trivial); everything else is at a hardware ceiling.

Artifacts: /tmp/minfer_phase7/b23_default.nsys-rep (+ trace/kern CSVs),
b23_llamacpp3.nsys-rep (+ b23_lc_* CSVs), b23_ncu_top.csv,
b23_ncu_fa_post.csv, r23_perf_ab.log, r23_tok_{pre,post}.txt;
scripts /tmp/gguf_types.py /tmp/merge_types.py /tmp/agg_b23.py
/tmp/agg_lc.py /tmp/parse_ncu_b23.py /tmp/ncu_b23.sh /tmp/ncu_r23_fa.sh
/tmp/perf_r23_ab.sh /tmp/patch_r23_fa.py /tmp/patch_r23_mask.py;
A/B baseline binary /tmp/minfer_pre_r23 (keep until next B-line session).


### P6 r24: scheduling-structure ladder — tile-order swizzle and persistent blocks both measured-closed/REGRESSIVE (2026-09-04)

The last untried structural family on `mmq_raw_wide_nt_kernel` (the 16-chain
128tok x 128od block: tile shapes, occupancy, staging, op-cuts, L2 residency —
r13–r23 — all measured-closed). Two rungs attempted, in order, both reverted.
NOTE: the box re-measured FASTER than the r13–r23 band — the 7B q4_k_m 2659-tok
prefill (26k prompt, interleaved same-slot 3x medians) sat at **KD=8 ~1385-1391
/ KD=4 ~1366-1373 tok/s** vs the documented ~1330/1318. So the session bar was
interpreted as *relative* (≥ +1.5% over the re-measured baseline) because the
stale "≥1350" absolute is met by the baseline itself; nothing landed (all
deltas were ≤ −0.5% at best).

**Rung 1 — tile-order swizzle (pure in-kernel blockIdx→(x,y) remap, grid
unchanged).** Same binary A/B's the orderings via `MINFER_MMQ_RAW_SCHED`
(passed host→kernel; a bijection on [0,gx)×[0,gy) so each tile is still
computed once, bit-identically). The grid is (21,28)=588 = 48 waves + tail.
| sched | order | KD=8 med | KD=4 med |
|---|---|---:|---:|
| 0 | default x-fastest (B-hot) | 1370.0 | 1362.6 |
| 1 | transposed y-fastest (A-hot) | 1345.0 (−2.3%) | 1307.6 (−4.7%) |
| 2 | G-grouped (GX=3, B-hot within A-groups) | 1344.3 (−2.4%) | 1315.2 (−4.2%) |

sched 0 vs the pre-change binary was within the ±1% noise band. **The default
x-fastest / B-hot order is the best; the A-hot transpose and the grouped
alternate both regress.** A-hot loses because the default already keeps the
small weight panel (B) resident while streaming the (larger, unique) A tile —
A is not reusable within the dispatch window regardless of order, and B-hot
reuse is the one that matters. No L2-lever survives here; consistent with r19
(L2 read traffic only 1.6×) and r21/r22 (the kernel is instruction/issue-bound,
not memory-scheduling-bound).

**Rung 2 — persistent blocks (launch nblocks = num_sms, each block walks a
strided tile list `for (u=blockIdx.x; u<nunits; u+=gridDim.x)`).** The kernel
body was wrapped in this u-loop (so one code path serves both persistent and,
when ustep==nunits, non-persistent — which reproduces the original mapping
exactly). Measured (KD=8, interleaved): baseline 1385.1; u-loop non-persistent
(env PERSIST unset) 1339.5 (**−3.3%**); persistent occ=1 (48 blocks) 1337.8;
persistent occ=2 (96 blocks) 1335.8. ptxas `-Xptxas -v` shows the u-loop and
original kernels are register-identical (115 regs, 0 spill) — so the −3.3% is
the loop-side structure changing the instruction schedule, not register
pressure. **Persistent recovered ~0 over the same-wrap non-persistent** (p1 ≈
p0 within noise), i.e. the claimed "25%-idle 13th wave ≈ 2% wall" tail does NOT
materialize here — the GPU block scheduler pipelines launches smoothly rather
than in lockstep waves, so there is no quantization tail to remove. Persistent
is therefore net regressive: it must carry the loop overhead and buys nothing.

**Rung 3 — full stream-k k-split was NOT attempted** (per the ladder: only if
rungs 1–2 showed promise). They did not, and stream-k would reorder the fp
accumulation (the parity gate passes only under the 1e-3 tolerance, and the
greedy token-identity gate is the killer for any add-reordering variant — the
task's "document as numerically non-portable" branch applies only if it showed
promise first, which it does not).

Conclusion: **this whole scheduling-structure family is measured-closed and
non-helpful for the wide MMQ kernel.** The residual levers remain the
k-loop instruction surplus (499k→418k per tile vs llama 356k, r13/r15/r17) and
the 3.2M op_st conflict mass — the compute/instruction side, not block or
tile scheduling. Artifacts: patch/dev notes /tmp/cuda_kernels_pre_r24.cu,
sched and persist checkpoints /tmp/minfer_r24_sched and /tmp/minfer_r24_persist;
harness /tmp/ab_medians.sh + configs /tmp/ab_cfg*.txt; logs
/tmp/minfer_phase7/r24_{kd4,kd8}_ab.log, r24_kd8_persist_ab.log; prompt
regenerated at /tmp/minfer_phase7/prompt2k.txt (8400B, `gen_prompt.py`).


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
