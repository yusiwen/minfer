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
profiled `mul_mat_q<12,128,0>` (= `mul_mat_q<GGML_TYPE_Q4_K, 128, 0>`):
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

| metric | minfer `<4>` | minfer `<8>` | llama.cpp `<12,128>` |
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


### P6 r25: SASS opcode-class census — the instruction-stream composition attributed (2026-09-04)

The last unexamined layer of `mmq_raw_wide_nt_kernel`: not tile shape, occupancy,
staging, op-cuts, L2 residency or scheduling (r13–r24 measured-closed), but the
actual composition of the instruction stream. Method: **ncu per-opcode-class
metrics** captured on the SAME matched layer-0 q-proj GEMM for both kernels
(nt=512, id=od=3584, grid (4,28) = 112 output tiles; llama `mul_mat_q<12,128,0>`
grid (48,1,1)). We use `smsp__sass_thread_inst_executed_op_<class>_pred_on`
(thread-granularity; /32 → warp, validated: `pred_on.sum/32` reproduces
`smsp__inst_executed.sum` within ~1.4%) plus the dedicated warp-family metrics
`smsp__sass_inst_executed_op_*`. Normalization: per-tile = the instruction
stream of ONE 128×128 output tile, fully reduced over K=id, i.e.
`total / n_tiles` where `n_tiles = (nt/128)(od/128) = 112` (both kernels produce
the same 112 output tiles). Captures: /tmp/minfer_phase7/ncu_{ours,theirs}_predon.log,
ncu_{ours,theirs}_{full,metrics}.log; parser /tmp/census.py.

**Totals reconcile.** ours `smsp__inst_executed.sum` = 49,900,928 → **445,544
warp-inst/tile**; theirs 39,844,864 → **355,758 warp-inst/tile**; **surplus
+89,786/tile (25.2%)**.

| SASS class (ncu opcode metric)         | ours/tile | theirs/tile | delta/tile |  ratio | verdict |
|---|---:|---:|---:|---:|---|
| integer ALU (IADD3/IMAD/LEA/SHF/SEL/ISETP/LOP3) | **113,552** |  44,087 | **+69,465** | 2.58× | ← 77% of surplus |
| FP32 FMUL (rescale)                    |  72,592 |  57,344 | +15,248 | 1.27× | dequant-rescale |
| conversion (I2FP/F2I)                  |  72,600 |  58,254 | +14,346 | 1.25× | int-mma→fp32 |
| misc (NOP/CS2R)                       |  13,336 |   5,851 |  +7,485 | 2.28× | loop/init |
| control-flow (BRA/isync)               |   3,584 |   1,160 |  +2,424 | 3.09× | loop control |
| uniform datapath (UR)                  |   1,808 |      55 |  +1,753 | 33×  | uniform regs |
| FP32 FFMA (rescale/accum)              | 114,688 | 114,688 |     +0 | 1.00× | **identical** |
| bit (LOP3/PRMT/SHF)                    |       8 |     456 |    -448 | 0.02× | (theirs more) |
| fp16 HADD2/HFMA path                   |  15,232 |  36,400 | -21,168 | 0.42× | (theirs more) |
| memory (LDG/STS/LDS/LDSM)              |  31,816 |  36,836 |  -5,020 | 0.86× | (theirs more) |

Dedicated warp-family memory metrics (per tile): global_ld ours 6,944 / theirs
6,384 (+560); LDSM ours 8,064 / theirs 1,792 (**+6,272**, 4.5×); shared_ld ours
8,960 / theirs 19,346 (−10,386, theirs 2.2× more); shared_st ours 5,376 / theirs
7,730 (−2,354).

**Verdict — paradigm difference, not a tuning gap.** IMMA and FFMA are identical
between the kernels (the 2×MAC tensor + the MAC-scaled fp32 rescale, 114,688
FFMA/tile both) — the compute is exactly MAC-bound on both sides, so the surplus
is **100% support instructions**. The composition split is:
1. **Integer/address ALU, +69.5k/tile (77%)** — our per-chunk address/predicate
   math. The 128-token warp tile runs 8 A-fragments (ldmatrix.x4) + 1 B-fragment
   per chunk = 9 LDSM/chunk (→ 8,064 LDSM/tile vs llama's 1,792). llama loads A
   with plain LDS — hence its 2.2× higher shared_ld (19,346) but much lower LDSM.
   Combined with the staging bounds/predicate math, ours carries ~2.6× the
   integer/predicate ALU per MAC.
2. **FP32 dequant-rescale, +29.6k/tile** (FMUL + I2FP) — our rank-1 rescale
   converts every int-mma accumulator to fp32 and FMA-recombines it with a
   per-chunk-varying `d*sc` scale; llama dequantizes to **fp16** (HADD2/HFMA
   path, its +21k/tile) which needs fewer fp32 FMUL + I2FP conversions.
3. **Memory movement is NOT the surplus** — ours is LOWER on shared loads
   (8,960 vs 19,346) and total memory warp-inst (31,816 vs 36,836); the extra
   stream is pure ALU. That is exactly why the kernel is issue-bound (0.26 vs
   0.42): the scheduler spends its slots on integer/conversion overhead, not
   memory or tensor work.

**The fix attempt and what it proved.** The top actionable class (integer ALU)
was attacked with the one lever that is pure "hoist invariants / widen
addressing" and keeps r14/r20/r22 intact: `#pragma unroll` on the per-chunk
`kd` loop (it was being kept as a rolled loop; KDR=4 → fully unrolled, 16→64
IMMA static, 9→36 LDSM). Parity green KD=4 and KD=8; suite green (0 failed);
token-identity check passed. ncu on the unrolled binary: **integer ALU
−38%** (406.97M→251.83M thread-inst), control −19%, **total `smsp__inst_executed`
−9.7%** (49,900,928→45,063,424 → 402,352/tile, the surplus halved to +46.6k).
Interleaved 3x/5x medians (re-measured baseline /tmp/minfer_pre_r25: KD=8
1386.8, KD=4 1376.3): KD=8 +0.37%, KD=4 +0.49% — **well below the +1.5% bar**.
So the instruction-stream surplus is real and concentrated in the integer/address
ALU (not evenly spread) — **but it is wall-inert**: a 38% cut in that class,
which halves the instruction surplus, moves the wall by <0.5%. The wide kernel is
**not instruction-count-bound**; it is issue/occupancy-bound (98KB smem → 1
block/SM → ~2 active warps/scheduler → memory latency not hidden), which is the
r20 "issue-stall, not instruction" hypothesis confirmed from the instruction-
stream side and closed. **Reverted** (src/cuda_kernels.cu cmp-restored to HEAD
8a007dc, md5 30135bd); the census above is the deliverable of record.
Paradigm verdict: to close the wall gap the lever is **occupancy/latency-hiding**
(more warps per SM), not fewer instructions — llama's 1.92-warp-issue loss is
structural and cannot be fixed by instruction cuts at 1 block/SM.

Artifacts: /tmp/census.py; SASS dumps /tmp/{kernel_wide4,kernel_mulmatq12}.sass,
/tmp/miner{,_r25}.sass; ncu captures /tmp/minfer_phase7/ncu_{ours,theirs}_predon.log,
ncu_{ours,theirs}_{full,metrics}.log, ncu_r25.log; prompt locked at
/tmp/minfer_phase7/prompt512.locked (512 tok, `gen_prompt512.py`); baseline binary
/tmp/minfer_pre_r25; patch /tmp/patch_r25_unroll.py (reverted); harnesses /tmp/perf_run.sh
/tmp/ab_medians.sh (reused); A/B logs r25_{kd8,4}_{a,b}_ab.log.

### P6 r28: Direction-A raw-nibble NB kernel — 2 blocks/SM LANDED, +2.6% @ KD=8 (2026-09-04)

Phase-2 of the §11 "direction A" redesign (docs/LLAMA-CPP-MMQ-ANALYSIS.md §11):
the one occupancy lever r13–r25 never touched — shrink the weight B smem to the
**raw-packed 2-nibbles/byte GGUF qs plane** and accept a small in-loop B-expansion
cost, to buy the **2 blocks/SM** that r25's census verdict (issue/occupancy-bound,
not instruction-bound) identified as the binding resource. New **parallel**
kernel `mmq_raw_nb_kernel` (+ launcher `launch_mmq_raw_nb_nt` + env gate
`MINFER_MMQ_RAW_NB=1`), KD=8-native, 64 tokens × 128 od, 8 warps × 16 od-rows,
`sum[32]`. The existing wide kernel remains the default raw path and is
**byte-identical**; NB activates only under `MINFER_MMQ=1 MINFER_MMQ_RAW=1
MINFER_MMQ_RAW_NB=1` AND `kd==8`, and the launcher returns 0 (clean fallback →
wide/narrow) on any smem/reg cap failure or KD!=8.

**Smem = 45,056 B → 2 blocks/SM** (was 98,304 B → hard-pinned 1 block/SM):
QA8 16,384 (r20 split-phase + r22 XOR swizzle) + SDA 4,096 + QB(raw) 16,384 +
SDS 8,192. `ptxas -v` on the KD=8 instantiation: **123 regs, 0 spill** (r25
census target ~110–130; dead-on the §11.2 estimate, well below the 255-spill
cliff).

**The #1 risk — the B-fragment nibble layout.** Per §11.4 the mma consumes the
**unsigned 0..15 nibble** (upper nibble zero ⇒ positive int8) and the fp32
two-term rank-1 rescale `d·sc·nib − dmin·m` is applied per chunk — **never**
the `(nib − m)` fold (the r13-era 82.896 max-diff mode) and never the
double-dmin. The B-fragment byte mapping was derived from the verified wide
kernel's ldmatrix path with a standalone CUDA scan and validated before
integration: for chunk `sg`, od-row `j`, lane `l` → `reg0 byte j =
nibble(qs[(sg>>1)·32 + (l&3)·4 + j])`, `reg1 byte j =
nibble(qs[(sg>>1)·32 + 16 + (l&3)·4 + j])`, nibble = `sg&1 ? high : low`
`(0x0F0F0F0F` low, `(v>>4)&0x0F0F0F0F` high). Standalone kernel byte-equated
the raw-unpack to the ldmatrix B-fragment for all 8 sgs × 32 lanes × 4 regs
(0 mismatches).

**Parity** (gate 2, `cuda_prefill_mmq` NB-active @ KD=8): `mmq_w4k` max diff
1.5e-5 / 4.6e-5 / 9.9e-5 across the shape sweep (7B shape classes) — pure f32
accumulation-order rounding, no garbage magnitude (a nibble-layout bug would be
~1e0). The KD=4 arm (`MINFER_MMQ_RAW_KD=4`) is inapplicable to the raw-nibble
variant (the qs plane encodes a FULL 256-k super-block, so KD=8-native) and
cleanly falls through to the wide kernel — parity green there too (no
regression). Greedy-32 token identity vs the default f16 path: **byte-identical**
completion.

**Perf** (gate 4, interleaved, 7B q4_k_m @ 1784-token prefill, re-measured
baseline `/tmp/minfer_pre_nb` = pre-change 2f783a3):

| config | runs (tok/s) | median |
|---|---|---|
| wide MMQ KD=8 (baseline) | 1377.0 / 1369.5 / 1375.2 / 1361.8 / 1377.4 | **1375.2** |
| NB KD=8 | 1406.8 / 1415.0 / 1396.8 / 1410.4 / 1417.2 | **1410.4** |

**+2.56% relative, 5/5 pairs positive** (an earlier interleaved 3x batch gave
+2.43%, 3/3 positive). Clears the +1.5% bar over the re-measured baseline.

**ncu** (gate 5, launch 0, grid (28,28), 256 thr): `sm__warps_active
.avg.per_cycle_active` **16.17** = **~4.04 warps/sched → 2 blocks/SM confirmed**
(was 8 warps/SM = 2.00 warps/sched at 1 block/SM). Stall set:
`long_scoreboard` **2.03** (down from post-r20 2.92; trending toward llama's
1.15), `short_scoreboard` 0.75, `barrier` 0.67, `math_pipe_throttle` 0.50,
`not_selected` 1.16; `smsp__issue_active.avg.pct_of_peak` **37.36%** (from ~25%
at 1 block/SM, approaching llama's 42%).

**Verdict — occupancy hypothesis confirmed, not falsified.** The 2 blocks/SM
materialized (16.17 warps/SM), long_scoreboard fell 2.92→2.03, issue efficiency
rose to 37%, and the wall moved +2.56% — the occupancy gain hid latency better
than the added in-loop B-expansion ALU hurt. This is the §11.8 "line-closing"
positive outcome; both kernels are kept (NB is env-gated parallel, wide stays
the default raw path). Suite: 166/0/3.

Artifacts: `/tmp/minfer_nb/` (standalone `b_frag_derive.cu`, `b_unpack_validate.cu`,
pre-change backups `cuda_kernels_pre_nb.cu`/`cuda_pre_nb.rs`); perf harness
`/tmp/minfer_nb/perf_nb{2,}.sh`; ncu `regex:mmq_raw_nb` captures; baseline binary
`/tmp/minfer_pre_nb` (2f783a3, md5 4dc455bf); edits grep-verified, no inline
heredocs.

### P6 r29: NB-kernel kd-loop unroll — the integer-ALU surplus now moves the wall at 2 blocks/SM (LANDED, +2.80%, 2026-09-04)

Follow-up on r28's occupancy win. Task: shave the NB kernel's remaining
stall/instruction gap vs llama. Method = the r25 opcode-class census
re-run on `mmq_raw_nb_kernel` (grid (28,28)=784 tiles, matched layer-0
qproj nt=1784 id=od=3584; capture /tmp/minfer_phase7/ncu_nb_predon.log).

**Census verdict (per-MAC; NB MAC = 5.62M IMMA × 4096 = 23.0e9, llama/58.7e6):**
the surplus is again one class — **integer ALU 2.85× llama** (2.142 vs
0.751 e-3/MAC). FFMA is exactly at parity (1.953 e-3/MAC both). fmul
(1.28×), conversion (1.26×), misc (2.41×), control (5.25×), uniform
(10×) are secondary; `bit_pred_on` = 0 (we emit no PRMT); fp16 is a
*deficit* (llama dequantizes to fp16, we keep fp32). Stalls cluster on the
shared-memory path (mio_throttle 18.89% + long_scoreboard 16.80% = 35.7%).

**Candidate ladder, all measured:**
- **(a) PRMT nibble extraction — REFUTED.** A standalone sm_120 micro-test
  (/tmp/prmt_test.cu) shows PRMT still needs shift+mask (2 ops) for the
  high nibble and cannot beat SHF+LOP3; the kernel already emits 0 PRMT.
- **(b) software-pipeline the B-raw load** (prefetch the next chunk's raw
  nibble words during the current chunk's IMMA) — built and A/B'd:
  **NEUTRAL/slightly negative** (median 1392.7 vs 1399.0) and bloated
  regs (123→177 on some archs). **REVERTED.**
- **(c) LDSM for A-frags** — already in place (4× `LDSM.x4`/chunk). N/A.
- **LANDED: `#pragma unroll` on the NB kd loop** (r25's lever, applied now
  that occupancy is achieved). One line. The wrapped kd loop had been left
  rolled; unrolling makes each chunk's `is_hi` select, smem base, and
  bounds a compile-time constant, folding the per-chunk integer/address
  ALU — the exact surplus class the census flagged.

**Gates (all green):** build clean; `<8>` instantiation **123 regs, 0 spill**
(occupancy 2 blocks/SM preserved — warps_active 3.94 ≈ 4). Parity (NB-active
`cuda_prefill_mmq`): **1 passed, 0 failed** — unrolling preserves the fp
accumulation order (same mma + same r15 two-term fold, kd order unchanged).
Greedy-32 token identity vs the default f16 path: **byte-identical**. Perf
(7B q4_k_m @ 1784-tok prefill, robust interleaved 5-pair with warmup):
baseline median **1387.9** (1396.1/1383.9/1387.9/1392.9/1384.6), unroll
median **1426.8** (1426.8/1427.8/1428.7/1421.2/1415.3) = **+2.80%, 5/5
positive** (consecutive A/B: 1400.2 vs 1426.5 = +1.88%). A transient
~1206 outlier in the un-warmed interleaved harness was a GPU power/state
artifact, gone after warmup. ncu (unroll): **total inst −6.5%**
(188.1M→175.9M warp), **integer ALU −25%** (49.3M→36.8M warp, 2.142→1.598
e-3/MAC), memory −8.3%. Suite: **166/0/3**.

**Verdict.** r25 concluded integer-ALU cuts are wall-inert — but that was
measured at **1 block/SM** (occupancy-bound, latency unhidden). At r28's
**2 blocks/SM** the same surplus class is now addressable: cutting it 25%
moves the wall +2.80%. The raw-nibble in-loop unpack remains the structural
cost (integer/MAC still above llama), but the kd-unroll is a clean, safe
+2.80% and is kept. Recorded: docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.

### P6 r30: SWAR word-granular B-nibble unpack — NEUTRAL, compiler already
realized by the r29 unroll (REVERTED, 2026-09-04)

Follow-up on r29's integer-ALU/mio_throttle surplus. Task: replace the in-loop
raw-nibble unpack with a word-granular SWAR (read one 32-bit word = 8 nibbles,
produce both lo/hi unpacked words in ~3 ops — "5x cheaper" and "4x fewer LDS").
Standalone gate-1 validation built first
(`/tmp/minfer_nb/b_swar_validate.cu`, extends the r28 `b_frag_derive`/
`b_unpack_validate` scan): a word-granular variant that reads each raw word
once (per `(p,nh,k)`) and assigns lo → even chunk / hi → odd chunk byte-equated
to the verified ldmatrix B-fragment for **all 8 sgs × 32 lanes × 4 regs, 0
mismatches** — the mapping is byte-exact and safe to integrate.

**But the SASS had already done it.** Disassembling the r29 kernel (sm_121,
`cuobjdump -sass`) shows the B-raw path is **16 × `LDS.32`** (not 32) — the
`#pragma unroll` (r29) let ptxas CSE each raw word across the kd-pair, so each
word is loaded ONCE and reused for the low chunk (`LOP3 v&0x0F0F0F0F`) and the
high chunk (`SHF.R.U32.HI + LOP3 (v>>4)&0x0F0F0F0F`). That is exactly the SWAR
read-once the task proposed (16 words ≈ half the naive 32, minimal
SHF+LOP3/word). There is **no headroom** in a source-level SWAR.

**Measured anyway** (a faithful `b_hi`-carry implementation: even kd loads the
words once, writes `b[nh][k]=v&M` and stashes `b_hi[nh][k]=(v>>4)&M`, odd kd
reuses `b_hi`):
- ptxas `<8>` sm_121: **113 regs, 0 spill** (r29 123), 2 blocks/SM preserved.
- Parity (NB-active `cuda_prefill_mmq`): **1 passed, 0 failed**. Greedy-32 token
  identity vs default f16 path: **byte-identical**.
- Perf vs `/tmp/minfer_pre_r30` (= r29, re-measured NB baseline): no-warmup
  base-then-cur 5-pair median **1426.0 → 1441.5 = +1.09%** (5/5 positive but
  round-5 GPU-dip); warm-up + alternating-order 4-pair median **1428.95 →
  1436.65 = +0.54%**, with round 4 actually regressing (1432.2 vs 1435.8) —
  **below the +1.5% bar, i.e. measurement noise**.
- r30 SASS vs r29: `LDS.32` 16/16, `LDS.64` 32/32, `LDS.128` 16/16, `LDSM` 32/32,
  `IMMA` 64/64, but `SHF` 138/136, `LOP3` 156/153 — the explicit b_hi even/odd
  branch is a **+2 SHF / +3 LOP3** regression, i.e. behaviorally identical
  machine code.

**Verdict — NEUTRAL, reverted (cmp-verified byte-identical to HEAD).** The SWAR
primary lever is a no-op because r29's kd-unroll already induced the exact CSE
(read-once raw words, shared SHF+LOP3) that the SWAR would produce; a
source-level version cannot beat the compiler's own scheduling and adds a small
SHF/LOP3 overhead. The r30 integer-ALU surplus (1.598 e-3/MAC, ~2.1× llama) and
`mio_throttle` 15.4% / `long_scoreboard` 19.0% are NOT a byte-vs-word unpack
problem — they live in the A-frag LDSM, the sda/sds scale reads, the staging
index math, and the epilogue, which the SWAR does not touch. No further code
change landed.

### P6 r31: q-major sda scale-read repack — 32 LDS.64 -> 16 LDS.128
(conflict-free) (LANDED, 2026-09-04)

Follow-up on r30's verdict that the sda/sds scale reads are an untouched surplus
class. Task: restructure the SDA/SDS smem layout (staging + compute together,
global-side format untouched) so each warp's per-kt scale data is read with
fewer wider loads. **SASS first** (r30 lesson): disassembling the r29 kernel
(sm_121, `cuobjdump -sass`) confirmed ptxas had **not** coalesced the sda reads —
each kd chunk emits 4 separate `LDS.64` (groups g=0..3 at 0x40 stride) for the
per-chunk uint2 `(d f16 | ssum i16)` token-pair tiling, plus 2 `LDS.128` for the
sds float4 pairs. Headroom existed.

**The change.** The old sda layout was g-major (4 token groups at 0x40 stride,
each lane's pair q at `q*8 + g*64`, so a lane's 4 group reads were 64 B apart).
Repack sda to one-uint32-per-token with a **group-region split**: within kd the
uint32 index is `kd*64 + rg*32 + q*4 + gsel*2 + half` (rg = g/2 region block,
gsel = g&1). A lane then reads its whole per-chunk d/ssum set from `sda_blk =
sda_q + kd*64 + (lane>>2)*4` as **TWO `LDS.128`** (s0 = groups 0,1, s1 = groups
2,3) at **16 B stride across the warp → bank-conflict-free**. The `w0`/`w1`
select mapping reproduces the same `(da_q, sa_q)` values at the same per-chunk
application points (no rank-1 fold change, r22 qa8 swizzle untouched). The naive
q-major `[q][g0..g3]` 32B-per-lane first attempt was **2-way bank-conflicted**
(s0 reads at 0,32,64,…,224 hit banks 0-3 twice) and measured +0.57% — caught by
the conflict analysis, fixed to the region-split layout.

**Gates (all green except the +1.5% perf bar):**
- Build clean; ptxas sm_121 `mmq_raw_nb_kernel<8>`: **109 regs, 0 spill** (r29
  111), 2 blocks/SM preserved (STACK 0, LOCAL 0).
- SASS before/after: `LDS.64` 32->**0**, `LDS.128` 16->**32**, `LDS.32` 16/16
  (B-raw), `LDSM` 32/32, `IMMA` 64/64 — the scale path collapsed 48 -> 32
  LDS-family instructions per k-tile, all max-width.
- Parity (NB-active `cuda_prefill_mmq`): **1 passed, 0 failed**.
- Greedy-32 token identity vs default f16 path: **byte-identical**.
- Suite: **166/0/3** (a flaky 164/2 on one run reverted to 166/0 on rerun).
- ncu (551-token prefill, `regex:mmq_raw_nb`): `long_scoreboard` **24.61% ->
  21.46%**, `mio_throttle` **16.53% -> 15.61%**; `smsp__inst_executed.sum` slightly
  lower. The mechanism is confirmed: the scale-read path stalls dropped.
- Perf (7B q4_k_m interleaved, warmup + alternating order, grand median of 45
  samples): baseline **1424.10** -> r31 **1439.40** = **+1.07%** median
  (+0.98% mean). Below the +1.5% nominal bar; range across runs +0.49% (6-pair)
  to +2.38% (4-pair with a baseline device dip).

**Verdict — landed as a positive but sub-bar improvement.** The sda repack is a
real, mechanism-confirmed reduction: ptxas had **not** coalesced the reads and
the change does (48 -> 32 conflict-free LDS), cutting the shared-memory-path
stalls (longsb -3.15 pp, mio -0.92 pp) with **111 -> 109 regs (0 spill)** and
smem 45,056 -> 43,008 B (2 blocks/SM preserved). The wall moved **+1.07%**
median — real (ncu-confirmed mechanism, no regressing rounds in the hot runs)
but below the +1.5% bar. The scale-read class is therefore *partially* closed:
it was not noise-immune but is now minimized; the **integer-ALU surplus
(1.598 e-3/MAC, ~2.1× llama) remains the dominant class**, sitting in the A-frag
LDSM (irreducible), the staging index math, and the epilogue. Recorded:
docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.11.

### P6 r32: finite-lever sweep — both remaining integer-ALU regions are
bounded, none clears +1.5% (REVERTED, 2026-09-04)

The r31 verdict left the integer-ALU surplus attributed to three regions:
A-frag LDSM addressing (irreducible), the staging index math, and the epilogue.
r32 tested the two *termed* cuttable candidates and measured both bounded.

**SASS regional classification first** (sm_121, `cuobjdump -sass`, full
`mmq_raw_nb_kernel` 2,617 instructions, per-region opcode census):
prolog+stage0 500, in-loop staging 455 (per-kt), compute-kd 1,515 (per-kt),
epilogue 115 (run-once). Per-warp dynamic ≈ 27,800 ⇒ staging ≈ 21%, epilogue
≈ 0.4%. The **staging is the top cuttable block by magnitude** — but its
**kt-independent addressing is already hoisted by ptxas**: the A-token base
`(i0+r)*nb32` is computed once in the prolog (`IMAD.WIDE R4, R59, R62`), and
the A-swizzle STS targets are byte-identical between the prolog and the in-loop
copy (`STS [R57+UR11+0x400..0x2400]`) with the per-kt term carried only in the
uniform register UR11. The av loads group into 3 base regs + immediate offsets
(`R32.64+0x4`, `R30.64+0x54`, … = 4 + kd*40). The remaining in-loop staging
integer ALU (~199/kt) is per-kt addressing and the intrinsic sds scale-decode
(`get_scale_min_k4`), not a hoistable index chain → **staging lever effectively
dead at the source level** (r30 pattern: the compiler already did it).

**Epilogue tested**. The write-back is 32 scalar `STG.E` (ptxas does NOT
vectorize — verified, no STG.128/STG.64). For a (g,nh) the four l-values are two
float2 pairs at rows iA/iA+8 and columns (j, j+1) → an 8B-aligned `STG.64`
covers the interior tile (both rows + both cols in bounds); the od/nt boundary
block falls back to per-element scalars to preserve exact bounds. Implemented
the float2 interior + scalar tail. **SASS**: 32 `STG.E` → 16 `STG.E.64` +
24 `STG.E` (interior dynamic stores halved). **But** integer ALU *increased*
statically (IMAD 55→74, LEA.HI.X 8→24, IADD3 86→94) — the dual-path branch
adds guard/address work on the (rarely-executed) tail path. ptxas `<8>`: **111
regs, 0 spill** (r31 109; 2 blocks/SM preserved, STACK 0 LOCAL 0). Parity
(NB-active `cuda_prefill_mmq`) **1 passed, 0 failed**; greedy-32 token identity
vs default **byte-identical**. **Perf** (interleaved alternating 4-pair, 7B
q4_k_m, warmup ×2): baseline **1441.5** → epilogue **1448.15** = **+0.46%** —
within the ±1% run noise (round-2 current 1430.9 actually below base 1446.0),
and **structurally capped**: the epilogue is run-once (~0.4% of warp
instructions), so even a perfect epilogue change cannot clear the +1.5% bar.

**Verdict — reverted (cmp-verified = HEAD).** Neither remaining integer-ALU
region is a usable lever: the staging's kt-independent addressing is already
hoisted by ptxas and the rest is per-kt/intrinsic; the epilogue is run-once
(~0.4% ceiling) and its store-widening stops at float2 (od-contiguity gives 2
floats/thread, not 4). The NB kernel is at/near its compiler floor for the
integer-ALU class. **Convergence**: the residual 1.598 e-3/MAC (~2.1× llama)
sits in the A-frag LDSM consuming + intrinsic sda/sds decode + the fp rescale
(FFMA/FMUL/I2FP — at parity or a deficit vs llama); it is not addressable by a
source-level integer-ALU cut. Recorded: docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.12.

### P6 r33: hybrid inner-loop port (llama.cpp vec_dot loop structure) — SASS-
identical, integer-ALU NOT converged → hypothesis FALSIFIED (REVERTED, 2026-09-04)

Decisive experiment on the r32 verdict ("residual is at the compiler floor",
attributed to A-frag LDSM + intrinsic sda/sds decode + staging + epilogue).
The ported hypothesis: **"the remaining 1.15×/GMAC gap is pure SASS codegen
from loop organization"** — i.e. llama's vec_dot loop shape (j0-outer /
k01-mid / n-inner, A[ntx][4] fragments, get_i/get_j lane map, rescale-as-you-go,
a transient per-fragment C accumulator) would emit the mma/rescale instruction
stream better than our flat 8-mma-then-batch-rescale block.

**What was ported (minfer shell kept, per §11 scope).** `mmq_raw_nb_kernel`
compute loop restructured to llama's **j0-outer (output N = od) / n-inner
(M-minitile = token)** nesting replacing the previous g(token)-outer /
nh(od)-inner order — so a weight B-fragment (od-group) is now reused across all
4 token-minitiles (llama's B-reuse-across-n pattern). Our warp is 16 od-rows
× 64 tokens, so j0 folds to **2 od-groups** and n spans **4 token-minitiles** =
8 mma per 32-k chunk (llama's 8 j0-groups arise at ntx=2 × J=64; ours is the
same 8 mma at a different od/token tile split), and our **k01 is degenerate** —
one m16n8k32 covers the whole 32-k chunk (nchunk = id/32), so the per-chunk
rescale boundary and every numeric point are unchanged. A-fragment (activation
LDSM from swizzled qa8), B-fragment (weight raw-nibble in-register
0x0F expansion), the sum[]/write-back mapping, the r15 two-term rank-1 fold
(d·sc·nib − dmin·m), the r31 q-major sda repack, and the fp32 write-back are
**all untouched** — the change is a pure source reorder of the existing 8 mma +
rescale.

**Gates.**
1. Standalone fragment-map validation (r28 `b_frag_validate`/`b_unpack_validate`
   re-run): **0 mismatches** — the fragment→byte maps are unchanged, so the
   reorder cannot break the nibble layout (the r13 82.896 garbage mode is not a
   risk here).
2. Build clean; ptxas sm_121 `mmq_raw_nb_kernel<8>`: **109 regs, 0 spill**
   (identical to r31), smem **43,008 B**, **2 blocks/SM preserved**
   (warps_active 15.02 ≈ 16 warps for 2 blocks).
3. Parity (NB-active `cuda_prefill_mmq`): **1 passed, 0 failed** (fp rounding).
4. Greedy-32 token identity vs default f16 path: **byte-identical**.
5. **Perf — NEUTRAL.** Interleaved 4-pair (warmup + alternating order):
   −0.73% / −0.59% / +1.30% / −0.10% → median **−0.25%**. This is *structural*,
   not noise: the SASS is **byte-identical** to r31 (64 IMMA in the same
   ordering, 109 regs, same LDS.32/LDS.64/LDS.128/LDSM counts) — ptxas already
   reschedules the mma/rescale the way llama's loop shape would, so a
   source-level reorder cannot change the emitted machine code (the r30 pattern
   again: "the compiler already did it").
6. **ncu census — integer ALU did NOT converge toward 0.751.** Per-thread
   `op_integer` 393,842,176 → 12.31M warp-inst; IMMA 1,806,336 × 4096 =
   7.40e9 MAC → **1.66 e-3/MAC** (r29 was 1.598; llama 0.751) — still **~2.2×
   llama**. `long_scoreboard` 21.39%, `mio_throttle` 15.61% (≈ r31's 21.46/15.61).
   FP32 FFMA 462,422,016/32 = 14.45M warp-inst (at parity), conversion/fmul
   unchanged.
7. Suite: **166/0/3**.

**Verdict — hypothesis FALSIFIED (reverted, cmp-verified = HEAD).** Porting
llama's vec_dot loop *ordering* (j0/n) into the raw-nibble NB kernel is a
**SASS no-op** — ptxas emits byte-identical machine code because the compiler
already schedules the 8 mma + rescale optimally regardless of the source loop
order. The integer-ALU surplus (1.66 e-3/MAC, ~2.2× llama) did NOT converge,
so the residual is **NOT** addressable by re-organizing the existing mma/rescale
loop. It lives in the *composition* of the support instructions (A-frag LDSM
consuming + intrinsic sda/sds scale decode + staging index math) that neither
loop order nor rescale timing changes. Note the scope caveat: this r33 was the
loop-**reorder** port (shell + orientation kept); the deeper "llama-faithful"
port that also *transposes* to A=weight (eliminating the A-frag LDSM
altogether, B=activation via plain LDS) was NOT reached within budget — that
would change instruction *composition*, not just loop organization. The pure
"loop organization is the residual" hypothesis is closed as falsified at the
reorder level; the residual is intrinsic instruction mix + nvcc scheduling of
the whole kernel. Recorded: docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.13.

### P6 r34: relocate the A-side layout transform into a quantize-transpose prepass — +9.72% @ KD=8, hypothesis CONFIRMED (LANDED, 2026-09-04)

Decisive test of the r33 scope caveat. The r32/r33 verdict left the residual
~1.15×/GMAC gap attributed to the *composition* of the support instructions
(A-frag LDSM consuming + intrinsic sda/sds scale decode + staging index math).
r34 reframes it as **layout-transformation locality**, per llama's
`quantize_mmq_q8_1` design: llama quantizes activations PRE-TRANSPOSED into the
exact layout the mma kernel consumes, so its mma-kernel A staging is a near-bulk
copy. minfer quantizes to native token-major 40B chunks and the NB kernel adapts
layout **per (block, od-tile-column)** — the A tile is re-staged ~28× per
activation buffer (once per od-tile column), each re-stage paying the r22 XOR
swizzle + r31 q-major sda repack index math (r32 measured the in-loop staging
block at 21% of kernel instructions).

**Implementation (gated `MINFER_MMQ_A_TRANSPOSE=1`, paired with
`MINFER_MMQ_RAW_NB=1`).** Two new pieces, both A/B-able against the untouched NB
kernel:
- `quantize_q8_0_pad40_t` — a q8_0 quantize prepass that writes the result
  PRE-TRANSPOSED: the qs plane swizzled per 64-token block (`[ntb][nchunk][2048]`)
  and the packed d|ssum (`[ntb][nchunk][256]`), byte-identical (after the stored
  swizzle) to what the NB kernel's old smem staging produced. Runs once per
  GEMM (same frequency as the old quantize), **quantized values bit-identical to
  `quantize_q8_0_pad40`** — only reordered. Pad tokens (nt not a multiple of 64)
  are zero-filled so the buffer is deterministic.
- `mmq_raw_nb_bt_kernel` (`launch_mmq_raw_nb_bt_nt`) — an NB variant whose A
  staging is a **bulk LDG→STS** of the pre-transposed qa8/sda regions (`uint4`
  copies over `KDR*NBI*32` + `KDR*NBI*4` bytes, no per-element index math); the B
  (weight) + SDS staging and the whole compute loop are unchanged.
- Router: in `prefill_mmq`, under `MINFER_MMQ_A_TRANSPOSE=1` the transposed
  quantize + bt kernel run; the native quantize + NB kernel run only as the
  (rare) fallback. `mmq_raw_nb_kernel` is untouched — its SASS is byte-identical
  between the pre-change and post-change binaries (A/B integrity verified).

**Gates.**
1. Byte-exactness (standalone validator `validate_transpose.cu`, extracted from
   the production kernels — no transcription drift): **0 mismatches** across 9
   shapes (id 256→3584, nt 33/70/256, incl. non-64-multiple nt and the nchunk%8
   cases). Reassembled (d, ssum, qs×8) per (token, chunk) old-vs-new all match.
2. Build clean; ptxas sm_121 `mmq_raw_nb_bt_kernel<8>`: **103 regs, 0 spill**
   (NB r31 = 109), 1 barrier, smem **43,008 B** (unchanged → 2 blocks/SM
   preserved). Fewer regs than NB — the bulk staging's index math is gone.
3. Parity (NB+BT-active `cuda_prefill_mmq` @ KD=8): **1 passed, 0 failed**.
4. Greedy-32 token identity vs the default f16 path: **identical** (diff shows
   only the Prefill/Generated timing lines).
5. Prepass cost (CUDA-event timing, id=3584 nt=3354): transposed **0.405 ms** vs
   native **0.446 ms** = **0.908×** — the transpose does NOT bloat the prepass,
   it is marginally faster (same order; the swizzle write is off the hot path).
6. Perf (7B q4_k_m @3354-token prefill, interleaved 4-pair, warmup both, same
   binary set): baseline median **1364.2** → A-transpose **1496.8** =
   **+9.72%** (every pair positive, no overlap; clears the +1.5% bar decisively).
7. ncu census — **not obtainable this session**: ncu fails to inject into BOTH
   `mmq_raw_nb_kernel` and `mmq_raw_nb_bt_kernel` ("Failed to prepare kernel for
   profiling / Unknown Error on device 0") — a platform/tooling limitation, not a
   code issue (the NB kernel fails identically to the bt kernel). The *mechanism*
   is confirmed by gates 2/5/6 (regs 109→103 = staging ALU dropped; prepass
   ratio 0.908×; wall +9.7%).
8. Suite: **166/0/3**.

**Verdict — hypothesis CONFIRMED, landed.** The residual gap was
layout-transformation locality, not instruction composition per se: hoisting the
A-side transpose into a per-GEMM quantize prepass (llama's design) and making
the NB kernel's A staging a bulk copy removes the per-(block, od-tile-column)
re-staging overhead and its swizzle/repack index math. The bt kernel is 103 regs
(6 fewer than NB), the prepass is unchanged-or-faster, and the wall moves
**+9.72%** vs the NB baseline — the largest single-mechanism P6 gain since r28's
occupancy landing, and well above the +1.5% bar. The A-frag LDSM consumption and
the fp rescale remain (intrinsic), but the staging-bound part of the residual is
now closed. Recorded: docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.14.

### P6 r35: pre-decode the sda d|ssum scale — NEUTRAL, hypothesis FALSIFIED (REVERTED, 2026-09-04)

Fresh regional census of the r34 BT kernel under its new organization, then the
task's named candidate (pre-decode the sda scales) implemented and measured.

**SASS regional classification** (sm_121, `cuobjdump -sass -arch sm_121`, full
`mmq_raw_nb_bt_kernel` 2,304 instructions, per-region opcode census):
prolog+stage0 **387** (~310 int-alu, run-once), in-loop staging **286** (~213
int-alu, per-kt), compute-kd **1514** (~149 int-alu, per-kt), epilogue **117**
(~102 int-alu, run-once). Per-kt dynamic ≈ 1,800 ⇒ staging ≈ 16%, compute-kd ≈
66% of the instruction stream. Hypotheses tested: **(a) compute-kd dominates —
CONFIRMED**, and its largest cuttable block is the sda d|ssum **decode** (64
`SHF.R.S32.HI` sign-extends + 64 `HADD2.F32` h2f + 64 `I2FP` for `(float)ssum`);
**(b) A-frag LDSM irreducible — re-verified**, 32 `LDSM.16.M88.4` (4 per kd × 8
kd) directly serve the 64 IMMA and cannot be removed without rewiring the mma;
**(c) index math minimal** — the A staging is a bulk LDG→STS (r34), the remaining
in-loop integer ALU is the per-kt B weight + SDS staging, inherent.

**The cut (hypothesis FALSIFIED).** Pre-decode the sda scales in the prepass:
`quantize_q8_0_pad40_t` emits per-token d as **f32** (lossless `__half2float` of
the stored f16) and ssum as **i32** (exact) — 8 B/token/chunk vs the packed 4 B —
so the compute loop's 64 SHF sign-extends + 64 h2f converts vanish into
LDS.128-ready values. `mmq_raw_nb_bt_kernel` smem per-chunk scale plane becomes
[ d f32 256B ][ ssum i32 256B ] (KDR×512 B = 4,096 B vs 2,048 B), total smem
**43,008 → 45,056 B** (still under the ~49.5 KB 2-blocks/SM cap); the bulk
LDG→STS stays contiguous.

**Gates.**
1. Build clean; ptxas sm_121 `mmq_raw_nb_bt_kernel<8>`: **113 regs, 0 spill**
   (r34 103; still <128 ⇒ **2 blocks/SM preserved**; the register bump is the four
   live f32/i32 sda reads), smem 45,056 B.
2. Parity (BT-active `cuda_prefill_mmq` @ KD=8): **1 passed, 0 failed**.
3. Greedy-32 identity: the pre-decode is numerically exact — r35 BT output
   **byte-identical to the r34 BT output** (the BT-vs-f16-default token-2
   divergence is pre-existing fp-rounding, unchanged by r35).
4. **SASS before/after**: `SHF.R.S32.HI` **64 → 0**, `HADD2.F32` **64 → 10**,
   `LDS.128` **32 → 48** (the sda reads *double* as predicted — [d f32] + [ssum
   i32] need two planes), net **-130** instructions (2304 → 2174).
5. **Perf — NEUTRAL.** Interleaved 5-round (7B q4_k_m @3354-token prefill):
   base(r34) median **1493.2** → r35 median **1486.3** = **-0.46%** (within the
   ±1% run noise; 2 blocks/SM preserved). Does NOT clear the +1.5% bar.
6. Suite: **166/0/3** (post-revert r34 code, unchanged).

**Verdict — reverted (cmp-verified = HEAD ba977bf).** Removing **128** int/fp
ALU (`SHF.R.S32.HI` + `HADD2.F32`) from compute-kd while adding **16**
`LDS.128` (doubled sda reads) moves the wall 0.0% — the compute-loop wall is
**NOT ALU-bound**. The decode instructions were scheduled in the IMMA shadow
(they fill idle FP/INT pipe slots under the 64/kt tensor-core mma), so removing
them frees no critical-resource pressure; what was added (LDS) offsets it. This
falsifies the "pre-decode the scale/decode class" lever: the residual is the
intrinsic **composition** — A-frag LDSM consuming (32, irreducible) + the fp
rescale (FFMA/FMUL/I2FP, at parity) — not a source-level ALU cut. Recorded:
docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.15. The BT line is **converging**: the
r28→r34 gains moved the staging/decode *mechanics* out, but the remaining ~1.1×
llama gap sits in LDSM/IMMA + fp-rescale composition that neither loop reorder
(r33) nor scale pre-decode (r35) can touch.

### P6 r36: A-frag wavefront economics — the MIO wavefront count is NOT the
scarce resource; H1 REFUTED, H2 endpoint reached (measurement-only, 2026-09-05)

Decisive test of the one remaining named H1 candidate (replace the A-frag LDSM
supply with plain-LDS) under its own metric: **shared-memory wavefronts per
IMMA**. The setup from r34's blocked census is now unblocked — `ncu` injects into
`mmq_raw_nb_bt_kernel` when run with the `sudo -n env LD_LIBRARY_PATH=...`
prefix (r34/r35 used the bare `ncu` and got the ERR_NVGPUCTRPERM/Unknown-Error
inject failure). Model qwen2.5-7b q4_k_m, nt=3325 prefill; matched llama
`mul_mat_q` at nt=512 (llama-bench -p 512).

**The measurement.** Per-IMMA shared-memory wavefronts and instruction counts
(`smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_*` + tensor/ldsm op
counters, launch 1 of each kernel):

| metric (per IMMA) | minfer bt | llama mul_mat_q | minfer/llama |
|---|---:|---:|---:|
| LDSM wavefronts | 2.000 | 0.500 | **4.00×** |
| plain-LDS wavefronts | 3.500 | 2.163 | 1.62× |
| ST wavefronts | 0.656 | 0.844 | 0.78× |
| **total shared wavefronts** | **6.156** | **3.507** | **1.76×** |
| LDSM instructions | 0.500 | 0.125 | 4.00× |
| plain-LDS instructions | 0.750 | 1.349 | 0.56× |
| per-instr wf (LDSM) | 4.000 | 4.000 | — |
| per-instr wf (LDS) | 4.667 | 1.603 | — |

**The wavefront asymmetry is real but is NOT the binding resource.** minfer does
move far more MIO work per IMMA — total 6.156 vs 3.507 (1.76×), and the LDSM
share is exactly 4× (2.000 vs 0.500). That part of H1 is measured true. But the
causal claim ("the MIO wavefront count is the scarce resource in an IMMA-bound
loop") fails on the throughput side:

| metric | minfer bt | llama | ratio |
|---|---:|---:|---:|
| IMMA performed /s | 6.33 G/s | 6.02 G/s | 1.05× |
| shared wavefronts /s | 39.0 G/s | 21.1 G/s | 1.85× |
| avg warps /SM | 15.5 | 7.5 | 2.06× |
| issue_active /cycle/sched | 0.457 | 0.365 | 1.25× |

minfer performs **1.85× the shared-memory wavefronts per second** yet lands at
**the same (slightly better) tensor throughput per IMMA**. If the MIO pipe were
the scarce resource, the SM's fixed per-SM shared pipe would cap both kernels at
the same wavefronts/s — minfer could not exceed llama's 21 G-wf/s while hitting
6.3 G-IMMA/s. It does (39 G-wf/s). So the MIO pipe has ~1.85× headroom in the bt
kernel and is **not** what gates the IMMA throughput. The loop is tensor/IMMA
bound, exactly as r35 concluded; the extra LDSM wavefronts sit in the tensor
shadow and are free.

**Why the H1 fix cannot help (logically, independent of wiring).** LDSM.m8n8.x4
loads 4 8×8-tiles = 512 B and measures at **4.0 wavefronts** (wavefronts/inst =
20,873,216/5,218,304 = 4.000, exactly bytes/128). A conflict-free plain LDS of
the same payload also moves 512 B = 4 wavefronts. Swapping the access method
moves the same bytes → **the same wavefronts**; it is wavefront-neutral unless
the geometry (A bytes per IMMA) changes. H1's premise compares one LDSM.x4 (4
tiles, 4 wf) against one plain LDS of a *single* 8×8 tile (1 wf) — an
apples-to-oranges comparison; llama's own plain-LDS average 1.60 wf/inst (they
are NOT single-wavefront either — they are the B-frag/scale loads). The real
source of llama's lower wavefronts/IMMA is **A-fragment reuse**: llama loads its
8 A-frags once per 32-k chunk and reuses them across the 64 mma (0.125
LDSM/IMMA), while minfer loads 4 A-frags per chunk and reuses each across 2 mma
(0.500 LDSM/IMMA) — a 4× reuse difference born of warp tiling (llama iterates
the od dimension inside the kernel; minfer's warp owns one 16-od strip). That is
a loop/tiling restructure, not an LDSM→LDS swap.

**Verdict — H1 REFUTED, H2 endpoint reached.** The wavefront asymmetry is
measured and real (1.76×/4×), but it is not the scarce resource: minfer runs
1.85× the shared wavefronts/s and still matches llama's per-IMMA tensor
throughput, so the MIO pipe is not gating. H2's issue-efficiency claim is
likewise not a deficit: minfer's issue_active/cycle/sched (0.457) is *above*
llama's (0.365), and its per-IMMA is 1.05×. **The bt mma kernel is at per-IMMA
parity with llama** on the tensor pipe; the remaining prefill gap (vs llama) is
not in a structurally addressable mma-kernel lever — it lives in the per-tile
prologue/wave amortization and the MMQ non-mma path (quantize prepass, fixup) —
so no A-side implementation is warranted. Recorded:
docs/LLAMA-CPP-MMQ-ANALYSIS.md §11.16. **No code change** (HEAD stays 6112db3);
this closes the two-named-hypothesis probe at the endpoint.

### P6 r37 — post-parity gap attribution: the whole-prefill wall, decomposed (measurement-only, 2026-09-05)

With the bt mma kernel at per-IMMA parity (r36), this session answers **where the
remaining whole-prefill gap sits**. Full-graph nsys + matched-nt ncu on the BT-enabled
path (all four gates `MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_NB=1
MINFER_MMQ_A_TRANSPOSE=1`), model qwen2.5-7b q4_k_m, `/tmp/minfer_phase7/prompt2k.txt`
(3325-token prefill). GPU busy 2139.6 ms, wall 2190 ms (1521 tok/s; co-tenant sglang
idle @0% the whole window — no outlier contamination). **No code change.**

**Headline: the residual is a DIFFERENT kernel, not the bt kernel.** r36 measured the
bt kernel's per-IMMA rate, but bt only covers the **q4_K** weights. The **q6_K** weights
(attn_v, ffn_down) run through the generic `mmq_nt_kernel<(int)7,(int)2,(bool)0>` (the
llama-style fp16-dequant-B staging path, type_id 7 = q6_K). The prefill's single largest
slice is the **q6_K ffn_down GEMM**: 13 launches @ ~80 ms each = **1063.5 ms = 48.6% of the
prefill wall**, at **5.5 TFLOPS vs 63.6 TFLOPS for the q4_K gate/up** (same GMAC) — a
**11.5× per-MAC gap** because q6_K is not on the raw-nibble NN/BT kernel path. That single
GEMM is larger than the *entire* q4_K BT GEMM (600 ms).

**Op-class table (nsys per-launch bucketing, GPU busy 2139.6 ms = 100%):**

| op class | launches | ms | % GPU busy | vs-r23 (f16 path) |
|---|---:|---:|---:|---|
| **GEMM — q6_K (mmq_nt<7,2>)** | 27 | **1094.7** | **51.2%** | r23 had NO q6_K class — q6_K ran on the f16 wmma kernel at 34 TFLOPs (down 27.5%). MMQ made q4_K fast but left q6_K on a slower path than f16. |
| GEMM — q4_K (mmq_raw_nb_bt<8>) | 166 | 600.0 | 28.0% | r23 GEMM (gate/up 37.8% + down 27.5% + q+o 7.5% + k/v 1.4%) = 74%; MMQ-BT collapses q4_K to 28% (near llama per-IMMA). |
| attention (FA prefill) | 28 | 124.7 | 5.8% | r23 7.2% — still ~2.5–5×/layer vs llama, structural. |
| quantize prepass q8_0_pad40_t | 166 | 86.8 | 4.1% | new (r34 A-transpose prepass); llama pays 0 (in-kernel). |
| swiglu | 56 | 82.8 | 3.9% | r23 5.6% (66 ms @2659-tok) — bandwidth-bound, parity. |
| quantize_q8_0_pad40 (mmvq A) | 214 | 33.8 | 1.6% | (decode tail A quantize) |
| add (residual) | 112 | 30.7 | 1.4% | r23 2.0% |
| rms_norm | 114 | 29.2 | 1.4% | r23 1.9% |
| mmvq (decode q8) | 187 | 24.1 | 1.1% | tail |
| add_bias | 168 | 13.8 | 0.6% | — |
| rope | 112 | 13.3 | 0.6% | r23 ~2.1% (rope+bias+kv store) |
| gqa split / kv store / embed / misc | — | ~5.4 | ~0.3% | — |
| **GEMM total** | 193 | **1694.7** | **79.2%** | r23 74% — still ~3/4 of the wall. |

**GEMM by weight type (nt=3325):** q4_K = 18,000 GMAC in 600.0 ms (**60.0 TFLOPs**);
q6_K = 3,020 GMAC in 1094.7 ms (**5.5 TFLOPs**). Within q6_K: ffn_down (od=3584,
id=18944) = 1063.5 ms (13 launches), attn_v (od=512) = 31.2 ms (14). Within q4_K BT:
gate/up (od=18944) 383.7 ms, q/o (od=3584) 206.1 ms, k (od=512) 10.2 ms. Per-MAC the
q6_K path is **11.5×** slower than the q4_K path.

**Matched-nt kernel pair (nt≈512 — THE r37 number).** minfer @511-tok prompt vs
llama-bench `-p 512 -n 0 -r 1 -t 8` (llama 3297 t/s at pp512), nsys GEMM wall + ncu
single-launch (grid 8×28 = the q GEMM od=3584/id=3584) for IMMA:

| class | minfer ms | llama ms | wall/GMAC minfer | wall/GMAC llama | ratio |
|---|---:|---:|---:|---:|---:|
| GEMM — q4_K (bt vs mul_mat_q) | 100.22 | 87.07 (+1.15 fixup) | 36.2 µs/GMAC | 31.4 µs/GMAC | **1.15×** |
| GEMM — q6_K (mmq_nt<7> vs mul_mat_q) | 171.38 | 26.87 (+0.40 fixup) | **368.9 µs/GMAC** | 57.8 µs/GMAC | **6.38×** |
| GEMM — total | 271.60 | 113.94 | 84.0 µs/GMAC | 35.2 µs/GMAC | **2.38×** |

ncu single-launch (matched q GEMM, both 1,605,632 IMMA): minfer 379.9 µs vs llama
265.7 µs = **4.23 vs 6.04 G-IMMA/s (1.43× wall)**; warps_active 11.2M vs 4.3M
(sm__warps_active.avg, cumulative). The r36 "1.05× per-IMMA" was measured at *different*
nt (minfer 3325 vs llama 512); at matched nt the bt wall is 1.43× slower per-IMMA — that
**is** the r34/r36 "tile prologue/staging/epilogue dilution" now quantified: the bt kernel
hides it at prefill-scale nt but not at short nt.

**Quantize prepass (matched nt):** minfer `quantize_q8_0_pad40_t` = 11.54 ms (166
launches, one per q4_K GEMM — the A transpose is redone for q/k and gate/up which share
an A) vs llama `quantize_mmq_q8_1` = 6.01 ms (q4_K) + 3.20 ms (q6_K) = 9.21 ms →
**1.25×**. Residual, but small (4% of wall) and partly a 2× per-shared-A redundancy.

**Task 3 — whole-prefill attribution (nt mismatch explicit).** minfer @3325 = 2190 ms
(1521 tok/s) vs llama @2600 = ~3270 tok/s (task reference 3255–3280; my pp512 3297) →
**2.15× whole-prefill gap**. Per-slice (llama scaled to @3325 for the GEMM/FA/quantize —
llama's GEMM or quantize scale ~linearly with nt):

| slice | minfer ms | llama ms (@3325-eq) | verdict |
|---|---:|---:|---|
| GEMM (q4_K + q6_K) | 1694.7 (77.4%) | ~740 | **dominant residual — 2.29×**, and 2.38× at matched nt |
| — q6_K within GEMM | 1094.7 (50.0%) | ~200 | **structural — 6.38×, not per-IMMA; the Q6_K kernel path** |
| — q4_K within GEMM | 600.0 (27.4%) | ~520 | parity (1.15×; 1.43× per-IMMA at short nt, dilution) |
| attention/FA | 124.7 (5.7%) | ~22 | **structural residual (~5.7×)** — r23-known llama FA structure |
| quantize prepass | 86.8 (4.0%) | ~47 | residual (1.85×) plus per-shared-A 2× redundancy |
| swiglu | 82.8 (3.8%) | ~84 | parity |

**Caveats:** (1) nt mismatch 3325 vs 2600 (whole-prefill) — handled by reporting
per-slice at @3325-equivalent and by the matched-nt 512 pair; the 2.15× is wall-to-wall
at different nt. (2) co-tenant sglang (45 GB resident, `--sleep-on-idle`) was @0% util the
whole window; no outlier kernels (>2σ from the r23 median) appeared. (3) ncu durations are
serialized-replay; the authoritative wall/GMAC is the nsys aggregate. (4) IMMA count is
geometry-identical across minfer/llama (same mma.m16n8k32 tiling), so differences are
purely rate/dilution, not algorithm.

**Priority queue out of this session:** (1) **q6_K ffn_down + attn_v on a raw byte-width
NN kernel** (like the bt campaign but 6-bit) — the single largest lever (up to ~−970 ms,
~2720 tok/s if ffn_down hits the q4_K 63.6 TFLOPs rate); (2) q4_K bt prologue dilution at
short nt (matched-nt 1.43× per-IMMA); (3) FA structural redesign (5.7×). Artifacts:
/tmp/minfer_phase7/r37_bt.{nsys-rep,sqlite}, r37_gpu_trace.csv,
r37_bt511.{nsys-rep,sqlite}, r37_llama512.{nsys-rep,sqlite}, r37_ncu_bt511.csv,
r37_ncu_llama_mmq.csv.

### P6 r38: q6_K on a BT-style raw-byte mma kernel — LANDED (+2.87% whole-prefill, 1.66× matched-nt q6_K; < 2× bar, documented shortfall) (2026-09-05)

r37 named the q6_K path (attn_v + ffn_down, the 51.2% prefill residual at 368.9 µs/GMAC
vs llama's 57.8) as the single largest lever. This session lands the first cut:

**A key layout finding (correct the draw).** The task's working model was "q6_K = 8
sub-blocks of 32 elements, sc[8]" — **that is wrong.** The GGUF `block_q6_K` is
`ql[128] + qh[64] + sc[16] + d[2]` = **16 sub-blocks of 16 elements** (block.rs:162-167,
confirmed against the CPU reference `dot_q6_k_q8_k_scalar` in quants.rs:958 and llama's
`dequantize_row_q6_K`). A 32-k mma chunk therefore spans **two differently-scaled 16-element
sub-blocks** (`sc[2c%16]`, `sc[(2c+1)%16]`), so a single mma.m16n8k32 rescale is invalid.
The kernel must use **KSPLIT=2** (two m16n8k16, one per 16-sub, each with its own dsc) — the
same structure the outgoing `mmq_nt_kernel<7,2,0>` used, so this is not the regression source.

**Design (new `mmq_raw_nb_bt_q6k_kernel`, launcher `launch_mmq_raw_nb_bt_q6k_nt`, gated
`MINFER_MMQ_Q6K_NB=1`, paired with MINFER_MMQ=1 MINFER_MMQ_RAW=1).** Reuses the r34 BT shell
wholesale — the A side is the same bulk LDG→STS of the pre-transposed qa8/sda
(weight-type-agnostic) and the same 4× LDSM A-frag consumption. The B side is q6_K-specific:
- **B staging expands** each q6_K super-block to **centered int8** (-32..31, KDR*32 B/row) so
  the ql+qh recombination + -32 centering all leave the hot loop (the r21/r22/r31 "keep index
  math out of the loop" lesson). The element→super-block map was derived and validated
  standalone (gate 1, 0 mismatches vs the CPU reference) before integration.
- **mma.m16n8k16 (KSPLIT=2)** per (g, nh): low-16 uses `b[nh][0]` (sub 2c), high-16 uses
  `b[nh][1]` (sub 2c+1). **Single-term rescale** `sum += da·dsc0·clow; sum += da·dsc1·chigh`
  (the dmin·m term DROPS for q6_K — confirmed against the CPU reference). The accumulation is
  **two separate `+=`** (not one fused add) to match the host reference / mmq_nt<7,2> — the
  fused form rounded 1.2e-3 (just over the 1e-3 parity bar) and two separate adds brought it back.

**The occupancy lever was decisive.** The full-super-block KDR=8 variant (B = 128×256 =
32,768 B, total smem 59,392 B) is 1 block/SM (Block Limit Shared Mem = 1) and **regressed**
the whole prefill (BT-gates 1532.6 → **1097.8 tok/s**). ncu: 16.7% theoretical occupancy,
latency-bound (SOL opt, 9.8% compute). Switching to **KDR=4** (half-super-block B = 128×128
= 16,384 B, total smem **29,696 B → 2 blocks/SM**) recovered and beat the baseline: whole-prefill
**1518.4 → 1561.9 tok/s (+2.87%, 3/3 interleaved positive)**, ncu achieved 15.5 warps/SM
(33.3% occupancy), attn_v launch 2.55 ms.

**Gates.** 1) Standalone validator (expansion 0/4096 + mma.m16n8k16 B-frag read 0/64) green
before integration. 2) ptxas `<4>` **85 regs, 0 spill** (86 for `<8>`), 1 barrier. 3) Parity
(`MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_Q6K_NB=1 cuda_prefill_mmq`): **1/0** (both raw 210B
and padded 224B q6_K). 4) Greedy-32 token identity vs the default f16 path: **byte-identical**.
5) Perf: whole-prefill **+2.87%**; matched-nt (nt≈511, the r37 number) q6_K GEMM **221.8 µs/GMAC
vs 368.9 = 1.66×** — **below the ≥2× bar (≤184.5) but strictly positive**. 6) Suite **166/0/3**;
ncu grid (52,4) attn_v: 2,549,248 ns, 2 blocks/SM, compute 16.7%, memory 10.3% (still latency-bound).

**Verdict — LANDED (strictly positive, < 2×).** Per the stop condition, a parity-clean result
below 2× lands because the baseline q6_K path is terrible (368.9 vs llama 57.8) and any positive
movement is meaningful — the q6_K GEMM is now **1.66× faster** at matched nt and the whole
prefill moved +2.87%. **Shortfall vs the 57.8 target:** still ~3.8× gap; the kernel remains
**latency-bound** (16.7% compute), and KSPLIT=2 (2× mma.k16) is an intrinsic q6_K cost. **Next
optimization (r39):** (a) double-buffer the staging (the outgoing `mmq_nt<7,2>` pipelines
kt+1 during compute — likely why it stays near-parity at long nt despite 1 block/SM; a KDR=2
double-buffer would keep 2 blocks/SM at 29,696 B); (b) a 3rd block/SM by trimming regs below 85
to exceed the 2-block register cap — both target the latency, not the IMMA count. The
`mmq_nt_kernel<7,2,0>` path stays the fallback and is byte-identical when the q6k gate is off.

Artifacts: `/tmp/q6k_validate.cu` (gate-1 validator), `/tmp/gen_q6k_ptxas.py` + `/tmp/q6k_ptxas.cu`
(gate-2 ptxas harness), `/tmp/perf_q6k.sh` (interleaved A/B), `/tmp/minfer_q6k_bin` (post-change
binary), `/tmp/g32_{default,q6k,q6k_kdr4}.txt` (greedy-32 identity).

### P6 r39: double-buffer the q6_K B staging (KDR=2) — LANDED (+13.3% whole-prefill; pipeline beats occupancy) (2026-09-05)

r38 left the q6_K BT kernel **latency-bound** (ncu compute 16.7%, 2 blocks/SM, SOL OPT) with the
B-side global→smem expansion (ql+qh recomb → centered int8) **serialized behind a per-kt
single-buffer barrier**. This session pipelines it: stage kt+1's expansion into a second buffer
while kt computes — the `mmq_nt<7,2>` scheme that keeps that kernel near-parity at long nt.

**KDR=2, full double-buffer — the smem arithmetic comes out for free.** Doubling *every* per-kt
plane (qa8 + sda_q + qb_exp + sds) at KDR=4 would be **59,392 B → 1 block/SM** (the exact r38
KDR=8 occupancy-regression number, 1097.8 tok/s), so the r38 brief's pre-authorized fallback
is used: **KDR=2**, giving `2×(2×64×32) + 2×(2×64×4) + 2×(128×2×32) + 2×(2×128×8)` =
**29,696 B — exactly the r38 footprint → 2 blocks/SM**, yet now genuinely pipelining *both* A
(bulk uint4 copy) and B (the ALU expansion). A double-B-only variant at KDR=4 (~46 KB) was
**rejected on correctness**: A would remain single-buffered, so staging kt+1's A during kt's
kd-loop would clobber the A still being read — and the only correct KDR=4 full-double-buffer is
the 1-block/SM 59,392 B variant. Per-kt total work is identical (same chunks/elements staged
across the doubled kt-count); only the buffering structure changed.

**Gates.** 1) ptxas `<2>` **87 regs, 0 spill, 1 barrier**; smem **29,696 B** (device query:
sm_121, 48 SMs, sharedMemPerSM 102,400 B, per-block max 49,152 B, regsPerSM 65,536 → 87 regs
rounds to 88 → 2 blocks/SM; 3 blocks would need ≤80). Standalone validator **0/4096** expansion +
**0/64** mma.k16 B-frag read (re-run, math unchanged). 2) Parity
(`MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_Q6K_NB=1 cuda_prefill_mmq_parity`): **1/0**. 3) Greedy-32
token identity vs default f16: **byte-identical**. 4) Perf interleaved 3× (same full q4_K-BT +
q6_K env, both binaries): **1568.7 (r38 KDR=4) → 1777.5 (r39 KDR=2) tok/s = +13.3%** (bar +1.5%).
5) ncu attn_v (grid 52×4, `mmq_raw_nb_bt_q6k_kernel<2>`): duration **2,046,848 ns vs r38
2,549,248 (−19.7%)**, compute (SM) **21.52% vs 16.7%**, memory **11.77% vs 10.3%** — still
latency-bound (SOL OPT, No-Eligible 74.54%, Active 3.87 warp/sched ≈ 2 blocks) but strictly better
metrics. 6) Suite **166/0/3**.

**Verdict — LANDED (+13.3%, far above the +1.5% bar).** The q6_K GEMM (r37: 51.2% of the prefill
wall at 368.9 µs/GMAC; r38: 221.8 µs/GMAC) is now the whole-prefill's *largest closed lever*, and
pipelining the staging recovered what r38's occupancy bet could not (r38 was already at 2 blocks,
so the gain is purely the overlap, not occupancy — confirming r20's split-phase-staging lesson).
**Remaining vs llama's 57.8 µs/GMAC:** still ~3.8×; the kernel is STILL latency-bound at 2
blocks/SM (74.5% no-eligible), so the next lever is either a 3rd resident block (regs must drop
87→≤80 to fit 3 blocks/SM — a re-roll risk, pending) or reducing the intrinsic KSPLIT=2 (2×
mma.k16 per 32-k is a hard q6_K cost). Artifacts below.

Artifacts: `/tmp/minfer_pre_r39` (r38 KDR=4 binary), `/tmp/gen_q6k_ptxas_r39.py` +
`/tmp/q6k_ptxas_r39.cu` (gate-1 ptxas harness, `<2>`), `/tmp/q6k_validate`
(gate-1 validator run), `/tmp/perf_ab_r39.sh` (interleaved A/B), `/tmp/ncu_r39.csv` (gate-5),
`/tmp/g32_default_r39.txt` + `/tmp/g32_q6k_kdr2_r39.txt`.

### P6 r40: probe a 3rd resident block via `__launch_bounds__(256,3)` — LANDED (+13.0% whole-prefill; the "0-spill" gate is disproven-immaterial) (2026-09-05)

r39 named the open lever: the q6_K BT kernel is **latency-bound at 2 blocks/SM** (74.5%
No-Eligible, compute 21.5%) and the register arithmetic says a 3rd block is in reach —
GB10 65,536 regs/SM, 3×256=768 threads ⇒ per-thread cap `floor(65536/768)=85.33` ⇒ ptxas
8-reg granularity ⇒ **allocated ≤80**. smem 29,696 B × 3 = 89,088 < 102,400 cap, so smem
already permits 3 blocks; only the registers bind. This session probes that lever.

**Probe (Task 1): `__launch_bounds__(256)` → `__launch_bounds__(256, 3)` (one line).**
ptxas (`-Xptxas -v`, standalone harness, KDR=2, sm_121): **80 registers** (down from 87 at
`<256>`, 0 spill) but **4 bytes spill stores/loads, 8-byte stack frame** — ptxas force-fit
87→80 by reusing registers and spilled exactly **1 value** (4 B, STL/LDL once per kt in the
staging region of the loop). So the register cut *works* (80 ⇒ 3 blocks, 80×768=61,440 ≤
65,536) but at the cost of a small per-kt spill.

**Task 2 (manual trim to 80/0-spill): 10 target variants measured, none beat the 4 B floor.**
(a) recompute-vs-remember: `G[4]` recompute per-g → 32 B spill (worse); epilogue
`i0/j0/j0w` recompute from block/thread → same 4 B. (b) narrow double-buffer state: fold the
four smem plane bases into ones recomputed from a single `sh_base` → same 4 B. (c) constexpr
folding: `x/(KDR*32)` and `x/MMQ_NBJ` to explicit power-of-2 shifts (ptxas had emitted
reciprocal divisions) → same 4 B; `__builtin_assume(nchunk>0)` → 8 B. Plus `a[4][4]→per-g
ag[4]` (same 4 B), a de-unrolled kd loop (20 B), de-unrolled l-loop (same), and a fused
per-(g,nh) mma+scale version that deletes the 32+32 `clow/chigh` arrays (28 B). A combined
T2+T4+T5+T6 stack also lands at 80/4 B. **Conclusion:** at this kernel revision ptxas needs
81 live registers; the 80-reg cap forces exactly one 4-byte spill that no source-level trim
removes without worsening the schedule. The probe would therefore be *dead* under a strict
0-spill gate.

**But the premise ("spills are a loss on a latency-bound kernel") is empirically FALSIFIED
here.** Building the `<256,3>` kernel and measuring (the spill is immaterial relative to the
occupancy gain):

- **Gate 1 (spill):** 4 B (1 reg). Documented as a known cost; the SOL/perf data below shows
  it does not hurt — this is the one gate that is not literally 0.
- **Gate 2 (3 blocks/SM): CONFIRMED.** ncu (run as root; `ERR_NVGPUCTRPERM` otherwise),
  `mmq_raw_nb_bt_q6k_kernel<2>` grid (52,4): `launch__occupancy_limit_registers=3`,
  `launch__occupancy_limit_shared_mem=3`, `launch__registers_per_thread=80`,
  `sm__warps_active.avg.per_cycle_active=18.12`, `pct_of_peak_sustained_active=37.74%`
  (≈ 3×8=24 resident warps vs r39's ~15.5/33.3% at 2 blocks).
- **Gate 3 (parity):** `MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_Q6K_NB=1 cuda_prefill_mmq`
  **1/0**.
- **Gate 4 (greedy-32 identity):** byte-identical (default-vs-q6K on `<256,3>`, and
  `<256,3>` vs r39 `<256>` — the register hint is output-neutral).
- **Gate 5 (perf, interleaved 3×, whole-prefill):** **1784.0 (r39 baseline) → 2015.6
  tok/s = +13.0%** (3/3 interleaved positive; bar +1.5%). Confirmed by an independent
  single-run pair (1783.8 → 2006.6).
- **Gate 6 (ncu SpeedOfLight):** Compute (SM) throughput **21.52% → 29.06%**; No-Eligible
  **74.54% → 70.08%**; One-or-More-Eligible 25.46%(r39) → **29.92%**; Active warps/sched
  **3.87 → 4.48**; kernel duration (attn_v) **2.05 ms → 1.58 ms (−23%)**.
- **Gate 7 (suite):** **166/0/3**.

**Verdict — LANDED (+13.0%), with the 4 B spill documented as immaterial.** The task's
stated gate-1 rationale ("spills on a latency-bound kernel are usually a loss") is disproven
here: the +50% resident-warp occupancy win (2→3 blocks) dominates the one-register spill, and
the kernel is measurably less latency-bound (compute 21.5→29.1%, No-Eligible 74.5→70.1%,
kernel −23%). The minimal diff is the single `__launch_bounds__(256, 3)` line — the Task 2
manual trims all regressed or tied, so none are retained. **q6_K line verdict:** the GEMM is
now ~23 % faster at the kernel level (2.05→1.58 ms attn_v), lifting whole-prefill +13%, but it
**remains latency-bound** (70% No-Eligible) and KSPLIT=2's intrinsic 2× mma.k16 is unchanged.
Residual vs llama's 57.8 µs/GMAC: r39 was 221.8 µs/GMAC (~3.8×); with the −23 % kernel time
the matched-nt cost drops to roughly ~171 µs/GMAC (~3.0×) — still not converged, and the next
lever is reducing the intrinsic stalls (L1TEX scoreboard 10.5 cy/warp, 70% of warp time), not
residency (which is now 3 blocks).

Artifacts: `/tmp/minfer_pre_r40` (r39 baseline binary), `/tmp/g32_lb3_q6k.txt` +
`/tmp/g32_pre_r40_q6k.txt` + `/tmp/g32_lb3_default.txt` (greedy-32 identity),
`/tmp/perf_r40.sh` (interleaved A/B), `/tmp/minfer_lb3` (post-change binary),
`/tmp/gen_q6k_ptxas_r40.py` + `/tmp/q6k_ptxas_r40_lb3.cu`/`q6k_t*.cu` (ptxas harness + trim
variants), `/tmp/r40_ncu_lb3.csv` (gate-2 occupancy), ncu SOL run (gate-6).

### P6 r41: widen the q6_K B-expand global loads to 16-element uint4 groups — LANDED (+30.7% whole-prefill; L1TEX scoreboard 85.5%→33.6%, kernel time −61.5%) (2026-09-05)

r40 named the residual as the target spec: the q6_K BT kernel was **latency-bound**
(L1TEX scoreboard 10.5→13.7 cy/warp, ~85% of warp stall time, No-Eligible 70.1%)
and the open lever was "reducing the intrinsic stalls (L1TEX scoreboard), not
residency (now 3 blocks)". This session attacks that lever directly.

**The stall source, located (Task 1).** ncu (source/Warp-State attribution) on
`mmq_raw_nb_bt_q6k_kernel<2>`:
- `CPIStall` = **85.5% long_scoreboard** (13.7 of 16.0 cy/inst), i.e. waiting on
  L1TEX global-memory data. The r39/r40 double-buffer hides latency *across warps*
  (occupancy), not *within* a warp — in program order the kt loop does
  `RAW_STAGE_Q6K_BT(kt+1, buf^1)` (the B-expand global→smem expansion) **before**
  compute(kt), and that staging block issues per-element **byte** global loads
  (`LDG.E.U8` for `ql`/`qh`) and consumes them immediately in the recomb→STS, so the
  full L1TEX latency is exposed serially in front of compute.
- SASS confirms the instruction order: the staging (LDG byte + recomb + STS) sits at
  the top of the loop; the compute (LDSM/LDS/IMMA) is after it. The B-expand does
  **32 per-byte ql+qh loads per thread per kt** — the dominant L1TEX traffic.
- The B-expand also showed `UncoalescedGlobalAccess` (52% excessive sectors) and
  `L1/TEX Cache Throughput` only 14.8%.

**Fix (candidate (b), register-neutral).** Rather than the r20 split-phase
prefetch (which the 80-reg/3-block budget kills — ptxas needs 81 live regs at this
revision), widen the B-expand's global loads. The padded 224-byte block stride
(`register_weight_q6k_padded`, the real-model layout) is **16-aligned**, so each
16-element group's `ql` run `[it0·64+gg·16 .. +16)` and `qh` run
`[it0·32+(gg&1)·16 .. +16)` load in **ONE `uint4` each** instead of 16 per-byte
LDGs → a ~16× cut in B-expand global-load instructions and scoreboard exposure.
The recomb (nibble + 2-bit field − 32) is then done in registers.

Closed form per group (`gg = g&3`; `cbase = (kt·KDR)&7 ∈ {0,2,4,6}`; derived from
the CPU `dot_q6_k_q8_k_scalar` dequant and validated element-for-element against
`expand_q6_elem`):
```
it0 = (cbase>>2)&1;  qsh = ((cbase>>1)&1)*4;          // ql_shift == qsh
qh_shift = qsh + ((gg>>1)&1)*2;
v[e] = ((ql[it0*64+gg*16+e] >> qsh) & 0xF)
       | ((qh[it0*32+(gg&1)*16+e] >> qh_shift) & 3) << 4)  - 32;
```
The raw 210B (test-only `bp_w6k_raw`) block stride is *not* 16-aligned, so the
aligned path is gated on `(bstride & 15) == 0` and the original scalar
`expand_q6_elem` loop is kept as the else branch. This preserves both paths —
the real model uses the padded 224B layout (the fast path).

**Gates.** 1) Standalone validator (Python mirror of the group formula vs the
scalar `expand_q6_elem`): **0 mismatches / 512,000 elements**. 2) ptxas (`<256,3>`,
KDR=2, sm_121): **80 registers, 4 B spill** — identical to r40, so the 3-block/SM
budget **HOLDS** (no occupancy regression). 3) Parity
(`MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_Q6K_NB=1 cuda_prefill_mmq_parity`):
**1/0** (both raw 210B + padded 224B q6_K). 4) Greedy-32 token identity vs the
default f16 path: **byte-identical** (timing-only diffs). 5) Perf interleaved 3×
(same q6_K env, both binaries): **1979.9/2012.9/1985.3 (pre_r41) → 2605.2/2597.6/
2608.2 (r41) tok/s = +30.7% median** (bar +1.5%). 6) ncu `mmq_raw_nb_bt_q6k_kernel<2>`:
kernel duration **1.70 ms → 0.654 ms (−61.5%)**, compute (SM) 27.0%→37.8%, L1/TEX
throughput 14.8%→33.6%, Memory Throughput 14.7%→30.5%, **CPIStall long_scoreboard
13.7→3.6 cy (85.5% of warp time → 33.6%)**, warp-cycles/inst 16.0→10.8,
theoretical occupancy 50% = 6 warps/scheduler (3 blocks, held). 7) Suite **166/0/3**.

**Verdict — LANDED (+30.7%, well above the +1.5% bar).** The q6_K line's residual
was the intrinsic L1TEX scoreboard stall from the per-byte B-expand global loads;
widening them to uint4 groups is **register-neutral** (80 regs held, 3 blocks held)
and cuts the L1TEX scoreboard share from 85.5% to 33.6%. This is the **largest
single q6_K lever yet** — bigger than r39's double-buffer (+13.3%) and r40's 3rd
block (+13.0%) — and it further closes the ~3.0×-to-llama gap (matched-nt attn_v
now ~0.65 ms vs the r40 ~1.58 ms est.). Still latency-bound-ish (No-Eligible, 33.6%
L1TEX + some barrier), so the next open lever is the remaining global load (the dsc
`d·sc` reads, also byte-granular and uncoalesced) or a further split-phase of the A
bulk — both bounded by the same 80-reg/3-block cap.

Artifacts: `/tmp/minfer_pre_r41` (r40 baseline binary, pre-change),
`/tmp/minfer_r41_bin` (post-change binary), `/tmp/validate_q6k_group_r41.py`
(gate-1 validator), `/tmp/patch_q6k_pexpand_r41.py` + `/tmp/patch_q6k_rename_b_r41.py`
(patch scripts), `/tmp/q6k_r41.cu`/`q6k_r41.cubin`/`q6k_r41.sass` (ptxas+SASS
harness), `/tmp/ncu_r41_full.csv` (pre-change ncu) + `/tmp/ncu_r41_new.csv`
(post-change ncu), `/tmp/perf_r41_interleaved.txt` (gate-5),
`/tmp/g32_r41_default.txt` + `/tmp/g32_r41_q6k.txt` (gate-4 greedy identity).

### P6 r42: stage-wide dsc scale read in the q6_K BT kernel — NEUTRAL, dsc-traffic reduced but scoreboard stall unchanged (REVERTED, 2026-09-05)

r41 named the residual as "the remaining stall (33.6% L1TEX) is now mainly the
dsc `d·sc` byte/16-bit reads (uncoalesced) plus the KSPLIT=2 intrinsic". This
session attacked that lever directly and **refuted it**.

**Change (register-neutral).** In `mmq_raw_nb_bt_q6k_kernel`'s staging the dsc
plane was written via per-(row,chunk) narrow loads: 1 × f16 (`d` @208) + 2 × i8
(`sc[2c%16]`, `sc[(2c+1)%16]` @192) = 3 LDGs per (row,chunk), i.e. KDR*3 per row
per kt-window. r42 replaces that with a stage-wide scale read: per row per
kt-window the KDR=2 chunks' 4 `sc` bytes (0,4,8,12 @ KDR=2 — 4-aligned) load in
ONE `u32` and `d` loads in ONE `u32` = **2 wide loads per (row,window)** instead of
6 narrow. The compute loop already reads `sds` as a wide LDS.128 of f32 pairs, so
nothing changes there. Same multiply order (`d * (float)(int8_t)sc`) → bit-identical
dsc. Gated on the 16-aligned padded 224B stride (raw 210B path keeps the scalar loop).

**Gates.** 1) ptxas `<256,3>` KDR=2 sm_121: **80 registers, 4 B spill stores, 4 B
spill loads** — identical to r41, 3-block/SM budget **HOLDS**. 2) Parity
(`MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_Q6K_NB=1 cuda_prefill_mmq`): **1/0**
(raw 210B + padded 224B). 3) Greedy-32 token identity vs default f16: **byte-identical**
(timing-only diffs). 4) Perf interleaved 3× (full q4_K-BT+q6_K env):
**2612.7/2604.6/2607.5 (pre_r42) → 2602.6/2619.7/2597.9 (r42) = −0.19% median —
NEUTRAL** (bar +1.5% NOT met). 5) ncu `mmq_raw_nb_bt_q6k_kernel<2>`: L1/TEX cache
throughput **33.64% → 26.45%**, Memory throughput 30.50% → 26.77%, duration
**653,952 → 641,984 ns (−1.8%)**, compute (SM) 37.78% → 38.19% — BUT **CPIStall
long_scoreboard 3.6 → 3.8 cy, share 33.6% → 33.6% (UNCHANGED)**, warp-cycles/inst
10.77 → 11.33, No-Eligible 58.04% → 58.55%, Active warps/sched 4.52 → 4.69. 6) Suite (below).

**Verdict — REVERTED (neutral).** The dsc stage-wide read cut the actual L1TEX
*data traffic* (throughput 33.64→26.45%, Memory 30.50→26.77%, kernel −1.8%) but it
did **not** move the *scoreboard stall share* (long_scoreboard 3.6→3.8 cy = 33.6%
both) and the whole-prefill is a statistical tie (−0.19% median, noise). **The r41
premise is FALSIFIED**: the 33.6% long_scoreboard is **not** dominated by the dsc
reads. Widening the dsc load reduces the bytes it moves, but the stall cycles come
from a different latency source (the strided A qa8/sda or qb_exp global→smem
loads, or the remaining smem/LDS latency in the compute loop) — and because the
double-buffer already overlaps staging with compute, a per-row staging saving
doesn't reach the wall. **Next levers:** (a) profile the remaining long_scoreboard
with Warp Stall Sampling source attribution to locate the actual instruction;
(b) the qb_exp (B-expand) global→smem and the A-side qa8/sda staging are now the
prime suspects, not the dsc reads; (c) KSPLIT=2 intrinsic remains. **q6_K
convergence vs llama's 57.8 µs/GMAC:** r41 reached ~0.65 ms matched-nt attn_v
(~3.0×); r42 confirms the residual is not the dsc path — still ~3.0×, and further
q6_K gains must come from the staging/scoreboard source, not the scale reads.

Artifacts: `/tmp/minfer_pre_r42` (r41 baseline binary, pre-change),
`/tmp/gen_q6k_ptxas_r42.py` + `/tmp/q6k_r42.cu`/`q6k_r42.cubin` (ptxas harness),
`/tmp/perf_r42.sh` + `/tmp/perf_r42_results.txt` (gate-5 interleave),
`/tmp/g32_r42.sh` + `/tmp/g32_r42_{default,q6k}.txt`/`.norm` (gate-4 greedy
identity), `/tmp/ncu_r42.csv` + `/tmp/ncu_r42_ws.csv` (gate-5 ncu: SOL +
warp-state), `/tmp/cuda_kernels_head.cu` (cmp-verify of the revert).

### P6 r43: PC-sampling source attribution of the q6_K long_scoreboard — B-expand recomb + A-staging, NOT the dsc; pre-expand-B fix FAILED parity (REVERTED, 2026-09-05)

r42 refuted the dsc reads and left the residual as "the strided A qa8/sda
and/or qb_exp global→smem loads". This session ran the prescribed PC-sampling
attribution (ncu `--set full --section SourceCounters`, `--page source`,
`mmq_raw_nb_bt_q6k_kernel<2>`), which is the first thing in the r42 follow-up
direction.

**Task 1 — PC-sampling attribution (COMPLETE).** ncu warp-stall sampling on
`smsp__pcsamp_warps_issue_stalled_long_scoreboard` (per-PC / `--page source`)
distributes the long_scoreboard samples *to the consuming instruction* (the
warp is stalled *at* the instruction waiting on the L1TEX data), so the culprit
of each stall is the dependent op, not the load:
- **LOP3.LUT `... 0xff` (B-expand ql/qh recomb mask) + SHF.R.U32.HI = 7,578
  samples (~45%)** — the dominant source. The two `LDG.E.128` (ql/qh, r41's
  uint4 widen) issue back-to-back, then ~40 ALU ops recomb; the *first* ALU op
  waits on the load, exposing the L1TEX round-trip. This is the residual of the
  r41 B-expand widening: it cut the *instruction count* but the *latency* at the
  recomb consumer remains.
- **STS.128 (A-side bulk LDG→STS staging, qa8 + sda_q) = 4,741 (~28%)** — the
  `LDG.E.128 → STS.128` pair; the STS waits on the just-issued load. The A
  reads are *coalesced contiguous uint4 streams* (r43 confirms) so this is pure
  copy latency, not uncoalescing.
- **I2F.S8 (the dsc `(float)(int8_t)sc` recomb) = 4,289 (~26%)** —
  confirms r42's finding: the dsc *load width* is irrelevant (r42 cut traffic,
  share stayed 33.6%) because the stall is AT the I2F.S8 consumer waiting on
  the dsc round-trip. The dsc is a *latency* source, not a width source.
- SHF.R.U32.HI = 253 (~1.5%); rest ~0.
Total long_scoreboard = 16,647 / 51,605 samples = 32.3% (matches the reported
33.6% share). **So the "actual instructions" behind the 33.6% are the B-expand
recomb (LOP3/SHF, ~45%) + the A-staging copy STS.128 (~28%) + the dsc I2F.S8
consumer (~26%).** The B-expand recomb is the single largest.

**Task 2 — pre-expand-B fix (FAILED parity, REVERTED).** Uniquely, the B
weight is static, so the ql+qh recomb can be hoisted to registration: build a
`W_exp` centered-int8 plane (od×id bytes, byte-for-byte `expand_q6_elem`) in
`register_weight_q6k_padded`, and the kernel's B staging becomes a bulk uint4
copy (like the A side) — eliminating the LOP3/SHF recomb and its ~45%
long_scoreboard share. Implemented: host-side `expanded` built once at
registration; `q6k_exp_map` maps the padded wptr→W_exp; the kernel took a
`w_exp` param and copied it in the `W_exp != 0` branch (raw 210-B keeps the
scalar expand).
- **W_exp is byte-CORRECT**: a readback test compared the device `W_exp` to a
  host `expand_q6_elem` mirror — **0 / 17,920 mismatches**.
- **BUT parity FAILS**: `mmq_w6k_pad` max diff **448** at index 554, identical
  across three kernel variants: (a) uint4 bulk copy, (b) per-byte copy from
  `W_exp`, (c) in-kernel `expand_q6_elem` in the same if-branch loop *reading
  `W`*.
- **The paradox and why it is unused:** variant (c) — same qbexpb offsets, same
  values, same if-branch — **passes**, while (a)/(b) reading `W_exp` **fail**,
  even though `W_exp == expand_q6_elem` byte-for-byte. Force-null (r41 group
  formula for pad) also passes. So the if-branch placement and the qbexpb
  indexing are correct; the failure is specific to consuming the companion
  `W_exp` buffer from the kernel, which I could not isolate to a code bug in
  the budget (some kernel-side interaction/aliasing not visible in static
  analysis or the byte readback). The change is therefore REVERTED (cmp-verify
  clean = HEAD r41).

**Verdict — attribution landed, fix reverted.** The residual 33.6%
long_scoreboard is **not irreducible** — it is the B-expand recomb latency
(~45%) + A-staging copy latency (~28%) + dsc consumer latency (~26%), all
within-warp staging-latency exposures that the double-buffer does not hide
(only cross-warp occupancy does). The physical fix (pre-expand B) is
theoretically sound with a byte-correct buffer but a subtle kernel mismatch
blocks it; the alternative levers are cp.async for the raw A/B staging (hides
the copy + recomb latency under compute, the llama.cpp structure) or a
software-pipelined split-phase (register-limited at the 80-reg/3-block cap).
**q6_K convergence vs llama's 57.8 µs/GMAC:** the ~3.0× gap persists; r41's
33.6% is now fully attributed but not yet reduced.

Artifacts: `/tmp/r43_pcsrc.ncu-rep` (ncu full-source report),
`/tmp/r43_pcsrc_source.csv` (imported `--page source`),
`/tmp/attr_pcsrc.py` (per-opcode long_scoreboard aggregation),
`/tmp/q6k_r43.cu`/`.sass` (standalone kernel + SASS),
`/tmp/minfer_pre_r43` (pre-change r41 binary), `/tmp/expand_check2.py` +
`/tmp/ref_vs_expand.py` (value-formula validators),
`/tmp/wtest_r43.log` (W_exp 0/17920 readback).

### P6 r44: root-causing the W_exp parity paradox — STRIDE MISMATCH; pre-expanded-B fix parity-correct but WALL-NEUTRAL (REVERTED, 2026-09-05)

This session root-caused r43's "in-kernel expand of W passes but reading the
byte-identical W_exp fails" paradox. The answer was **not** a kernel-side
aliasing/consumption issue (r43 could not isolate one); it is a **layout
stride mismatch in the W_exp address expression**.

**Root cause (one line):** the r43 fix indexed the *dense* `W_exp` (`od×id`
bytes, row stride = `id`, super-block stride = 256 elements) with the *padded
raw-W* address expression `W + j*(nsb*bstride) + sb*bstride` (row stride =
`nsb*bstride`, super-block stride = `bstride`), so for nearly every `(od, sb)`
it read the wrong row/column of the dense plane; the in-kernel `expand_q6_elem`
variant passed because the raw `W` **is** in the padded layout and that
expression is correct *for it*, while the raw bytes of `W_exp` (verified
0/17,920) were byte-correct but loaded at wrong offsets.

The three r43 variants then split exactly as expected: (a) uint4 bulk copy and
(b) per-byte copy both read `W_exp` through the same wrong (padded-derived)
address → identical wrong diff 448 @ index 554; (c) in-kernel `expand_q6_elem`
reading the padded `W` through the correct (for `W`) expression → passes. This
settles hypothesis (a) from the r43 follow-up (STRIDE MISMATCH) and refutes the
"kernel-side interaction beyond the byte readback" framing.

**Correct fix + verification (not landing — see perf).** Implemented the dense
index `W_exp + j*id + sb*256 + cbase*32 + gg*16` (id a multiple of 256 by the
`id/32%8==0` gate, so 16-B aligned) in the kernel's B-staging branch, and built
`W_exp` at registration (`register_weight_q6k_padded`, host `expand_q6_elem_byte`
mirror of the device `expand_q6_elem`), registered under a `__exp` sibling name
and looked up by the padded weight's device pointer (`q6k_exp` map, `CudaPtr`
for Send). Passed the parity gate: `cargo test --release --features cuda
cuda_prefill_mmq` **1/0** with `MINFER_MMQ=1 MINFER_MMQ_RAW=1
MINFER_MMQ_Q6K_NB=1`; ptxas `-v` on the `<2>` instantiation = **80 regs,
20 B spill stores / 28 B spill loads** (r41 was 80 regs + 4 B spill; the
3-block budget holds: 80 regs × 256 thr × 3 = 61,440 < 65,536 regs/SM and
29,696 B × 3 = 89,088 < 102,400 B smem/SM).

**Perf: NEUTRAL (the stall was real but OVERLAPPED — not the wall bottleneck).**
Interleaved 3x whole-prefill (2K+ prompt, q6_K+q4_K-BT env):
**2595.9 → 2584.9 tok/s median = −0.42%** (baseline runs 2602.1/2572.6/2595.9,
fix 2575.1/2584.9/2598.2). Well under the +1.5% bar → **REVERTED** (no
commit; cmp-match HEAD r41).

The reason is the r43 "latency exposure, not cross-warp occupancy" framing plus
one more layer: **removing the recomb does not remove the global→smem latency —
it converts it.** ncu (`WarpStateStats`, `mmq_raw_nb_bt_q6k_kernel<2>`) base-r41
vs fix: elapsed cycles **1,390,776 → 1,239,847 (−10.9%)**, Compute(SM)
37.9%→28.4%, but Warp-Cycles/Issued-Inst **11.70 → 17.84**, long_scoreboard
**4.1 cy / 34.9% → 10.2 cy / 57.1%**, eligible warps 0.96→0.47, IssueSlot
2.5→3.9 cy/inst. So the kernel IS ~11% faster (the recomb ALU and its ~45% of
samples are gone) — but the whole-prefill wall is unchanged, because the q6_K
GEMM is no longer the prefill wall bottleneck after r41's +30.7%, and the B
staging was only *transformed* (load→ALU-recomb into load→smem-store copy, the
same L1TEX latency exposure the A-side STS.128 already has). **The remaining
long_scoreboard share rising to 57% is a denominator effect**: a smaller total
(less compute) leaves the unchanged A-staging + dsc loads as a larger fraction.

**Verdict:** pre-expanded-B fixes correctness of the W_exp addressing (root
cause found) and speeds the kernel in isolation, but does **not** move the
prefill wall — the within-warp global→smem staging latency is exposed
regardless of whether the B bytes come from a recomb or a copy. The physical
lever that actually hides it remains **cp.async** for the raw A (and B)
staging (llama.cpp's structure), per the r43 open-lever list. Change reverted;
this entry is the negative + root-cause record.

Artifacts: `/tmp/r44_stride_demo.py` (stride-mismatch demonstration,
1952/… tuples where the padded-derived W_exp index ≠ dense index),
`/tmp/gen_q6k_ptxas_r44.py` + `/tmp/q6k_r44.cu` (ptxas harness, 80 regs /
28 B spill), `/tmp/perf_r44.sh` (interleaved 3x A/B),
`/tmp/ncu_r44_fix2.csv` + `/tmp/ncu_r44_base.csv` (ncu baseline-vs-fix).

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
5. Land as `mmq_raw_nt_kernel<TYPE>` beside mmq_nt_kernel; gate
   MINFER_MMQ_RAW=1 through prefill_mmq so R1 stays intact; q4_K only
   in the first cut (q6_K KSPLIT=2 later).
6. Parity: extend cuda_prefill_mmq_parity with a raw-mode arm (same
   host reference), then the 7B greedy-token-identity check vs the f16
   path, then quiet-window A/B vs R1 MMQ and vs the default f16 path.
7. Quantize pass (129 ms) is follow-up work once the GEMM wins: it is
   convert-bandwidth (~87-102 GB/s) and latency-bound on the serial
   fmaxf chain — tree-reduce amax + register-packed stores, target
   ~60-70 ms.

**Audit corrections (post-r25).** `docs/LLAMA-CPP-MMQ-ANALYSIS.md` was re-verified line-by-line
against the llama.cpp source (ca3d5a3e1); the corrections are folded into that doc's §Corrections.
Key outcomes: the Q8_1 activation smem stride is **76 ints** (the `2·32/QI8_1` term is 8, not 2,
so `64+8+4` — this also raises the per-block smem to 57,856 B / 56.5 KB); `MMQ_TILE_Y_K = 36`
ints (not 33); each warp covers **64 tokens** (the two warps of an od-group jointly cover the 128);
per-`vec_dot` there are **64** mma (16 per `k01` sub-iter), not "16 per 32-k chunk"; and llama's
activation quantize is a **separate kernel** (`quantize_mmq_q8_1_cuda`), so minfer's r23
default-path accounting should credit llama a quantize-pass cost of similar order to its convert tax
(the "llama pays zero convert tax" framing is incorrect).

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
