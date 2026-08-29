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

## 8c. Prefill Q8_0-activation GEMM (the 7e⑥ leftover)

The original 7e⑥ bullet bundled two items; only the async-H2D staging half
landed. The unwired half: quantize activations once per prefill
(`launch_quantize_q8_0` exists) and run Q4_0×Q8_0-style GEMMs
(`launch_q4_0_q8_0_matmul` exists) instead of the f32-activation kernels.

- Measure FIRST at 7B prefill shapes (standalone nvcc A/B, the 7e②
  methodology): the f32 kernels already reach decent bandwidth at large nt,
  so the win is unproven — do not wire on faith.
- Wire behind a gate only if the A/B shows a real win; otherwise record the
  negative result in CUDA-BACKEND-PLAN and delete the externs.

Gate: standalone A/B ≥ 10% on the [od, id, nt] shapes that dominate prefill
before any in-tree wiring.

## 8d. Attention kernel (remaining decode overhead)

Attention on CUDA is a hand-rolled f32 path (GQA, kv f32). At short ctx the
weight reads dominate, but attention grows O(ctx) per token while the
quant-matmul kernels now run at ~163 GB/s on FFN shapes.

- Profile first with nsys (the 7e② profiles captured no kernel data — fix
  the capture config or use CUPTI) to get the per-kernel decode split.
- Candidates: fused GQA attention (one kernel for scores+softmax+AV) and/or
  flash-style prefill attention; pair with 8b (f16 KV reads).
- The old `gqa_attn_f32` kernel (pre-Phase-7) may be a starting point.

Gate: node-level parity vs the current path (tolerance class per GPU_SAFETY);
7B decode A/B at ctx 440 and 2K.

## 8e. MMQ shared-memory tiling for K-quants (kernel bandwidth)

The 7e② unit-mapping kernels are streaming; llama.cpp's MMQ uses shared-
memory tiling + better occupancy. 62% of peak BW leaves ~40% on the table —
if tiling pushes the FFN shapes from ~163 to ~200 GB/s, 7B decode goes from
~26 to ~32 tok/s without touching anything else.

- Apply to q4_K/q6_K first (the 7B-critical types), reusing the 7e②
  standalone-A/B + `cuda_kquant_matmul_parity` harness.
- Keep the unit-mapping kernel as fallback (dispatch by shape if tiling wins
  only on large od).

Gate: bit-parity unchanged (same block math), standalone A/B, 7B E2E.

## 8f. Q5_K kernels (lift the all-or-nothing gate)

`q5_k_m` models fall back to CPU wholesale (the weights gate requires every
matmul weight to have a CUDA kernel; Q5_K lacks one). Kernel pattern follows
the q4_K unit mapping (176-byte super-blocks). Also covers Q5_0/Q5_1 if
cheap after Q5_K.

Gate: standalone A/B + parity test mirroring `cuda_kquant_matmul_parity`;
`qwen2.5-0.5b-instruct-q5_k_m` E2E (the negative-test model from 7c).

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

1. **Stale docs**: `CUDA_OPTIMIZATION.md` / `CUDA_PROBLEMS.md` describe the
   pre-Phase-7 imperative path (`layer_gpu`, deleted `forward.rs`) — mark
   superseded with a pointer to CUDA-BACKEND-PLAN (their surviving ideas are
   absorbed here: cuBLAS → 8k, MMQ tiling → 8e, GPU quantize → 8c).
2. **Optional CUDA CI runner** — device-gated tests skip gracefully today;
   a self-hosted GB10 runner would keep the 144-test suite honest on every
   commit.
3. **Temp files**: the Phase-7 ledger (`/tmp/minfer_phase7/TEMPS.md`) is
   closed; cleanup still awaits the user's decision (no auto-delete).

## 8i. Graph integration test debts (CUDA)

Coverage gaps found in the Phase 8 completeness audit (2026-08-29):

1. **Multi-split capture** — per-split capture is supported but the
   bit-parity tests only cover single-split graphs (7e③ made every current
   decode graph single-split; no live exposure today).
2. **Multi-turn conversation on CUDA** — decode graphs are
   n_past-independent, so cross-turn reuse should work structurally, but
   the conversation path (new prefill graph per turn + reused decode graph)
   has no device-level test.
3. **OpenAI server slot loop on CUDA** — the slot's prefill→decode switches
   hit the same GraphCache; no device-level coverage.

Gate: each as a `#[cfg(test)]` device test mirroring the 7d parity harness,
skipping gracefully without a device.

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
