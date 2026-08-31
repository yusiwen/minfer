# CUDA Inference Path — Optimization Roadmap

> **STATUS (2026-08-31): LIVE ROADMAP.** This was the pre-Phase-7 draft
> (RTX 4080 era, imperative `layer_gpu` path — all since deleted). It is
> now the consolidated CUDA index: current performance (Part I), what the
> Aug 30 session landed (Part II), the remaining plan (Part III), and the
> original roadmap with each item's actual outcome (Part IV).
> Single-sourced implementation records: `docs/CUDA-BACKEND-PLAN.md`
> (Phase 7a–7e) and `docs/CUDA-FOLLOWUP-PLAN.md` (Phase 8, 8a–8p).

## Part I — Current state (DGX Spark GB10, master `761e236`)

llama.cpp baseline: llama-bench `ca3d5a3e1`, 8 threads, `-ngl 99`, GB10
CUDA. minfer: CLI single-shot, greedy.

| Workload | llama.cpp | minfer | gap |
|---|---:|---:|---:|
| 7B q4_k_m prefill @2K | 3401 | ~1400–1500 (±10% thermal) | 2.3× |
| 7B decode tg128 | 47.1 | 40.5–40.8 | 1.16× |
| 7B decode @2K | 44.9 | ~36 | 1.25× |
| 0.6B q8_0 prefill @2K | 23909 | 4792 | 5.0× |
| 0.6B decode tg128 | 290 | ~195 | 1.5× |
| 0.5B q4_0 prefill @2K | 30550 | ~3020 | 10× |
| 0.5B decode tg128 | 453 | ~257 | 1.8× |

7B prefill went 30.7 → ~1450 tok/s over the Aug 30 session (~47×); decode
never regressed (41.2 baseline vs 40.8 now — the residual gap is pure
weight streaming, see R2).

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

## Part III — Remaining roadmap

Measured budgets: f16 wmma GEMM ~35 TFLOPS (llama.cpp int8 MMQ ≈ 52
equivalent); MMVQ weight streaming 130–147 GB/s effective vs 252.7 GB/s
read-only probe (93% of the 273 GB/s theoretical); llama.cpp achieves
~197 GB/s on the same decode shape.

- **R1 — int8 MMQ prefill GEMM** (the 7B prefill parity lever): quantize
  activations to q8_0 on the fly (`quantize_q8_0_pad40` already exists
  from 8e) and run tiled int8 dp4a/tensor-core GEMM per weight type —
  llama.cpp's MMQ structure. Expected 7B @2K → 3000+ tok/s. The 8p fused
  kernel (raw B bytes in-register) is the natural skeleton.
- **R2 — MMVQ weight-streaming efficiency**: close 147 → ~197 GB/s
  (llama) against the 252.7 GB/s platform probe: access-pattern and
  occupancy work on the per-type decode kernels; lm_head (Q6_K, 444 MB
  per token) is the single biggest row. Expected: decode @2K 36 → ~40,
  tg128 40.8 → ~45.
- **R3 — small-model per-token overhead**: LARGELY DONE (Aug 31, see
  Part II). 0.5B decode was ~4.0 ms/token with ~1.6 ms of GPU floor:
  prefill single-split (A1), pinned logits readback + no clone (A2),
  prefill capture automatic (B). Remaining: clean pre/post bench numbers
  on a quiet GPU; the sampler's `apply_top_k` scratch alloc (608 KB/token
  when top_k is enabled — greedy benches never hit it) if non-greedy
  numbers justify it.
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
