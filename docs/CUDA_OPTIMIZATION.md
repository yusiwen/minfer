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
