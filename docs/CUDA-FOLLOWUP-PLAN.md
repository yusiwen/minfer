# CUDA Follow-Up Plan (Phase 8)

Status: **planned** (created 2026-08-29, right after Phase 7 completion).
Phase 7 (7a–7e) record: `docs/CUDA-BACKEND-PLAN.md`. This doc consolidates
everything left open: the 7e leftovers, the Phase-7 review findings, and the
former §8 "Out of Scope / Future" list — re-prioritized with measured numbers.

Reference hardware: aarch64 GB10 (SM 12.1, 48 SMs, ~273 GB/s peak, 124 GB).

## 0. Baseline (end of Phase 7, commits `dfa3516`→`d83aa0f`)

| Model | Decode | Notes |
|---|---|---|
| Qwen2.5-0.5B Q4_0 | ~249.6 tok/s | FusedFFN +10% (7e⑤), single-split graph (7e③), graphs ON |
| Qwen3-0.6B Q8_0 | ~190.0 tok/s | FusedFFN +4% |
| Qwen2.5-7B Q4_K_M | ~21.7–26.4 tok/s | 3.1× from 7e② kernel vectorization; **~62% of peak BW** (4.4 GB weights / tok ÷ 26 tok/s ≈ 170 GB/s effective) |
| 7B prefill | 9.18 s / 300 tok | matmul-dominated (7e③) |

Suites: cuda 144/0 (parallel + single), plain 130/0; fmt clean; zero rustc
warnings. CUDA KV cache is **f32-only** (`store_kv_f32`); Metal auto-selects
f16 for the 7B class — that asymmetry is 8b below.

## 8a. Correctness debts (do first, small)

> **Phase 8 review (2026-08-29, independent subagent review of 4fcd0d8..5cbb4ca):
> 11 findings, all fixed in `961f696`** — capture-window abort on execute_node
> errors, replay-vs-open-window guard, kernel weight-READ row guards
> (q4_k/q6_k raw+padded/f32_vec), pos_scratch pool_gen bump, ring-wrap reset
> race, stale padded_weights flag, pinned-ring alloc logging, Metal ffn_gu
> gate (~1.99 GiB on 7B, memory-only), stray 7 MB trace file removed,
> dump/trace/viz fuse_ffn gating aligned with the engine. Suites re-verified
> (cuda 147/0, plain 133/0, fmt clean; 7B/0.5B E2E greedy coherent).

1. **macOS regression run** — the qwen2 FFN-fusion gate was decoupled from
   `fuse_qkv` to `CParams.fuse_ffn` in 7e⑤ (mirroring Qwen3's existing
   intent), and `961f696` additionally gated the Metal `ffn_gu` loader
   registration on the same condition. Metal checks on a Mac: 0.5B + 7B
   greedy text must match the pre-change record; A/B
   `MINFER_NO_FUSE_FFN` (fused vs unfused with FusionPass per the AGENTS.md
   test rule); confirm 0.5B still fuses on Metal (nf=2944 → gate open).
2. **F32-weight GGUF E2E** — 7e④'s F32×F32 kernels have parity coverage but
   no end-to-end model (none cached). Download/quantize one F32-weight model
   and run the standard greedy-coherence + CPU A/B gate.
3. **Rebuild-gate unit test** — assert `MINFER_NO_FUSE_FFN` (and
   `MINFER_NO_FUSE_QKV`) flip `CParams` identity, i.e. `cache::params_match`
   returns false. Cheap; closes the A/B footgun class permanently.

Gate: all existing suites stay green; new gates pass.

**Batch-1 outcome (2026-08-29, commits `789e64b`→`b849601`):**
- 8a③ DONE — `fuse_flags_are_part_of_the_reuse_identity` asserts both flip
  directions through `GraphCache::try_reuse`.
- 8g① DONE — `ComputeGraph::capture_nt_hint()` + decode-only capture gate
  in `graph_replay_step`; `cuda_prefill_shaped_graph_never_captures` runs a
  prefill-shaped (nt=8) graph 4× and asserts zero captures.
- 8a② DONE — and it caught a real CPU bug: `vec_ops::mat_mul_f32` wrote
  token-TRANSPOSED output for nt > 1 (decode nt==1 was accidentally
  correct, hiding it forever; no F32-weight model existed). Verified with a
  byte-exact GGUF→F32 converter (numpy): qwen2.5-0.5B-F32 produced garbage
  on CPU while the same weights kept as Q4_0 ran fine; after the fix, both
  F32 models (0.5B + qwen3-0.6B) produce the identical greedy text on CPU
  and CUDA. Regression test: `f32_matmul_nt2_token_major`.
- 8a① BLOCKED on hardware — no macOS machine in this environment; the
  loader gate + fuse_ffn decoupling remain pending a Mac regression run.

## 8b. KV f16 on CUDA — **DONE 2026-08-29 (`f7b0036`)**

`store_kv_f16` + `gqa_attn_f32_f16kv` (exact structural mirror of the f32
attention: same online softmax, warp reductions, guards; the only delta is
half4 → float4 K/V loads with f32 accumulation). Policy mirrors Metal:
`MINFER_CACHE_TYPE=f16|f32` override, auto f16 when n_layers×n_kv_embd ≥ 8192
(7B class), set at model load and cached per CudaBackend instance.

Gates passed: f16 roundtrip parity vs a half-rounded-KV reference (1e-4,
kernel isolated from quantization noise); 7B 96-token greedy text identical
f16 vs f32; 7B @2K-ctx decode +~11% (swap-order pairs: 10.3/10.2 vs
9.4/9.0 tok/s — the win grows with context). Caveat: `MINFER_GRAPH_DUMP`
reads KV regions as f32 — incompatible with f16 KV (debug dump path only).

## 8c. Prefill Q8_0-activation GEMM — **DONE 2026-08-29 (`69a27c5`, shaped)**

Measure-first verdict (standalone nvcc A/B, quantize included): +38–44% at
id ≤ 8192 (0.5B shapes, 7B attn/qkv/o; activation-heavy), +4.7% at 7B
ffn_gu (weight-bound), **−63% at 7B ffn_down (id=18944)** — the q8_0 kernel
streams weight bytes SLOWER than the f32 kernel when id/od is high, so a
blind wire would have been a large regression. Wired only the winning
region: `nt > 1 && id ≤ 8192` routes Q4_0 prefill through quantize_q8_0 +
q4_0_q8_0_matmul (grow-on-demand scratch; capture-safe because prefill
never captures since 8g①). E2E: 0.5B prefill @3.6K tokens 1005 → 1246 tok/s
(+24%), greedy text unchanged. Parity: `cuda_q4_0_prefill_q8_0_gemm_parity`
(nt>1 vs the kernel's exact math, nt=1 f32 path vs hand dequant);
`cuda_matmul_parity`'s q4_0 arm mirrors activation quantization.

Negative result recorded for the record: the unshaped wire the 7e⑥ bullet
imagined would have REGRESSED 7B-class Q4_0 ffn_down by ~60%. The shape
gate is load-bearing.

## 8d. Attention kernel — **DONE 2026-08-29 (`a5af60f`, split-K flash-decoding)**

nsys capture now works (the 7e② "no kernel data" issue was the report
workflow, not the config): `nsys profile --trace=cuda` + `nsys stats` /
sqlite over `CUPTI_ACTIVITY_KIND_KERNEL`, decode step isolated by gap
segmentation. Result at 7B @2K decode: gqa_attn 48.3% of the step,
q4_k matmul 36.4%, q6_k 13.9% — the single-warp-per-(token, head) kernel
ran 28 warps total (GPU idle) and streamed 2K KV rows serially at
~3 GB/s effective.

Fix: split-K flash-decoding — pass 1 scans SPLITS=8 KV chunks in parallel
(FIXED grid × nh blocks, ranges derived from device-side positions → CUDA
Graph capture stays valid), pass 2 merges (mx, S, oc) partials. Size-stable
state scratch, grown during warmup only. Template over the KV element type
(f16 + f32 layouts). Dispatch nt == 1 only.

Gates: standalone A/B nkv 440 +49% / 2000 +64% / 8000 +80% (maxdiff 1e-8);
parity vs cpu_gqa_attn 1e-4 (empty + partial splits, both KV layouts); 7B
E2E @2K decode 10.1 → 13.7 tok/s (+36%), greedy text identical; @440
neutral (3 pairs within noise). cuda 154/0.

## 8e. MMQ shared-memory tiling for K-quants — **NEGATIVE RESULT 2026-08-29**

The premise did not survive measurement. Standalone A/B (`/tmp/minfer_phase7/
bench8e.cu`, 7e② methodology, L2 DEFEATED by cycling 8× 76 MB weight buffers
≈ 600 MB working set — without this the bench numbers are L2-inflated):

- Current streaming kernel at the 7B FFN shapes: **116 GB/s true streaming**
  (the 7e② "163 GB/s" figure was L2-resident). Cross-check: 7B decode 26.4
  tok/s × 4.4 GB/token = 114 GB/s — decode is ALREADY at the achieved
  streaming bandwidth.
- Variant A (dp4a over q8_0-quantized activations, aligned 40 B block
  layout): ≤ +3% at ffn_gu, −12% at ffn_down (quantize pass not amortized at
  nt=1) — failed the ≥10% gate.
- Variant B (wide blocks: 128 threads, 8 rows/warp, 32 rows/block for more
  loads in flight): −6% — occupancy is not the limiter; the DRAM access
  pattern is.

Conclusion: do NOT wire. The remaining gap to the 273 GB/s marketing number
is the LPDDR5X streaming ceiling (llama.cpp-class access patterns or
cp.async.bulk/TMA would be needed to verify); kernel-side tiling does not
move it. Decode-side bandwidth work is closed unless a fundamentally
different weight layout (e.g. interleaved tiles) is measured to stream
faster.

## 8f. Q5_K kernels — **DONE 2026-08-29 (`b959ec9`, Q5_K + Q5_1)**

The all-or-nothing gate needs a kernel for EVERY matmul weight type; the
0.5B q5_k_m file (ftype Q5_K_M) actually contains **Q5_1** for attn q/k/o,
ffn gate/up and tok_embd, plus Q6_K (ffn_down) and Q8_0 (attn_v, output) —
so Q5_1 was required alongside Q5_K. Both f32-activation matmuls mirror the
Q4_0/Q4_K structures; Q5_K decodes the transposed qh (bit s of byte l) and
deinterleaved qs chunks, with sub-level tail masking for partial last
super-blocks (0.5B id = 896 = 3.5×256; dispatch requires id % 32 == 0).
embed_rows_q5_1/_q5_k cover the q5_1 token embedding. Gates updated in both
qwen2 and qwen3 `weights_on_cuda`.

Gate results: parity test (q5_1 id 64; q5_K id 896 tail, decode-formula
weights over real unpack_q4k_scales, 5e-3); 0.5B q5_k_m E2E — CUDA now
admits the model (was CPU wholesale), greedy output identical to CPU;
cuda 155/0. Q5_0 deferred (no model needs it; add when one does).

## 8g. Prefill capture: gate the accidental path, then productize

Two findings from the Phase 8 completeness audit (2026-08-29):

1. **Immediate hygiene (ships with 8a):** `graph_replay_step` has NO nt
   gate — the scheduler runs the 3-run capture protocol for EVERY CUDA
   split, so a repeated identical-nt prefill (server/slot scenario) would
   silently start capturing a ~437-node graph. Correctness is plausible
   (positions are out-of-window input fills, prefill is single-split since
   7e③, pool churn forces recapture) but untested and benefit-free. Add an
   explicit decode-only capture gate, or a test that turns prefill capture
   into a deliberate feature.
2. **Productization:** capture prefill splits deliberately — after 8c/8d
   (allocator churn at prefill sizes makes capture windows more fragile;
   the launch-overhead win is smaller since big kernels amortize it).

Gate: replay bit-parity harness like 7d's, at pp16 + pp300.

## 8h. Infra / process

1. **Stale docs**: `CUDA_OPTIMIZATION.md` / `CUDA_PROBLEMS.md` — **DONE
   2026-08-29 (`60e9cc1`)**: marked SUPERSEDED with pointers to the current
   plans (absorbed ideas named: cuBLAS → 8k, MMQ tiling → 8e, GPU quantize → 8c).
2. **Optional CUDA CI runner** — device-gated tests skip gracefully today;
   a self-hosted GB10 runner would keep the 144-test suite honest on every
   commit.
3. **Temp files**: the Phase-7 ledger (`/tmp/minfer_phase7/TEMPS.md`) is
   closed; cleanup still awaits the user's decision (no auto-delete).

## 8i. Graph integration test debts — **DONE 2026-08-29 (`60e9cc1`)**

1. **Multi-split capture** — `cuda_multisplit_capture_bit_parity`: CUDA →
   CPU (Softmax) → CUDA graph yields two CUDA splits; both capture
   (captured_count == 2) and replay bit-identical to direct launches.
2. **Multi-turn conversation** — `cuda_conversation_multiturn_reuse`
   (q4_0 0.5B, device): turn-2 incremental (append-only KV + reused decode
   graph) vs turn-2 rehydrated from history (fresh graphs + full re-prefill)
   produce IDENTICAL greedy text. ConversationSpec now derives Clone; note
   that device tests must call `CudaState::init()` themselves (`get()` only
   reads the singleton).
3. **Slot loop** — covered by the same test: both paths run the GraphCache
   prefill→decode alternation the OpenAI server slot uses; a dedicated
   axum-level test remains out of scope (needs a live HTTP harness).

## 8j. cudaGraphExecUpdate (optional)

Parametric exec update instead of full recapture on pool_gen change. Not
worth it today: capture is ms-scale, recapture is rare (pool_gen is stable
across decode steps), and llama.cpp does not use it either. Revisit only if
recapture shows up in a profile.

## 8k. Explicitly not planned (revisit only with a concrete need)

FP16 activations + cuBLAS/cublasLt (large-GEMM path), VMM pool,
multi-GPU + peer copies, `graph_optimize`-style node reordering, Windows,
IQ/Q2/Q3 quant families.

## Priority order

**8a → 8b → 8c → 8d → 8e → 8f → 8i → 8g → 8h → 8j** (8k stays closed).
The 8g① decode-only capture gate ships together with 8a.

Rationale: 8a is correctness debt from Phase 7 itself; 8b is the largest
decode lever with a proven Metal precedent; 8c is a promise to close with a
measure-first gate; 8d/8e chase the remaining 40% bandwidth/overhead headroom
on 7B; 8f widens model coverage; 8g/8h are polish.
