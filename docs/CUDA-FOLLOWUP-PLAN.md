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

## 8e. MMQ / MMVQ for decode K-quants — **VERDICT REVERSED → DONE 2026-08-29 (re-examined per llama.cpp)**

**The 2026-08-29 negative result was wrong on both of its load-bearing claims.**
Re-examination (user request: "8e MMQ tiling 的问题检查一下,参考 llama.cpp 的代码,修复"):

1. **"116 GB/s = the platform's streaming limit" was false.** A pure read-only
   probe (`bwprobe.cu`: grid-stride `__ldcs` uint4 sum over 6×256 MB buffers,
   5 sweeps) measures the GB10 at **252.7 GB/s** read-only (93% of the 273
   GB/s theoretical); D2D memcpy 234.9 GB/s (read+write). The old kernel's
   116 GB/s was its ACHIEVED rate (~46%), not a ceiling — there was ~2.2×
   headroom. (The final bench8e.cu also never actually timed its dp4a
   candidate — the table column was "-" while the "failed the gate" claim was
   already written.)
2. **The structure, not the access pattern, was the limiter.** The in-tree
   kernel runs 2 warps × 4 rows per block (~64 threads), each lane serially
   decoding whole 144 B blocks — only ~28K threads in flight at 7B ffn_down,
   not enough to hide LPDDR latency. llama.cpp's `mul_mat_vec_q` has an
   explicit **GB10 parameter table** (`MMVQ_PARAMETERS_GB10`,
   `GGML_CUDA_CC_DGX_SPARK`): for decode (ncols_dst == 1) it uses ONE output
   row per block with 8 warps (256 threads, `2× generic` when `halve_iters`),
   the row's (block, 32-element sub-block) units spread round-robin over the
   threads, `__dp4a` int dots over q8-quantized activations, and a block-wide
   reduction — ~917K threads in flight at the same shape.

**Fix (ported in-tree, q4_K first):**
- `quantize_q8_0_pad40` — per-token activation quantization into padded 40 B
  blocks ([f16 d][2B pad][32B int8]) so the int8 payload is 4-byte aligned
  for the dp4a reads (the 8e lesson: the unpadded 34 B layout misaligns).
  Scratch: `buf_q8_decode` (nt=1, size-stable per graph, grown during the
  eager warmup runs — capture-safe like `buf_attn_partial`).
- `q4_k_q8_mmvq` — one row per block, 256 threads, units `u = (blk, sub)`,
  per-unit 8×`__dp4a` (dot + Σx for the m-term), value =
  `d8·(s8·d·dot − m8·dm·sx)` (note dm, not d — first draft folded the m-term
  under `d`; caught by the parity test). Sub-block nibble map matches the
  in-tree/llama.cpp q4_K packing: chunk (sub>>1), lo nibbles for even sub,
  hi for odd, element l ↔ byte l. Partial last super-blocks are excluded by
  `nsub = ceil(id/32)` units (same granularity as the q5_K kernel).
- Dispatch: `matmul_f32_ptr_layout`, Q4_K arm — `nt == 1 && id >= 2048 &&
  id % 32 == 0` → mmvq (below id 2048 the win collapses to launch-latency
  noise: 0.5B attn shape measured +2.3%); everything else keeps the f32
  kernel. Prefill (nt > 1) unchanged.

**Measured (bench8e2.cu, L2-defeated 8-buffer cycling, 7B shapes):**

| shape | old kernel | mmvq | Δ |
|---|---|---|---|
| ffn_gu (od 37888 × id 3584) | 0.660 ms = 116 GB/s | 0.373 ms = 205 GB/s | **+77%** |
| ffn_down (od 3584 × id 18944) | 0.338 ms = 113 GB/s | 0.193 ms = 198 GB/s | **+75%** |
| 0.5B attn (896²) | 0.0062 ms | 0.0061 ms | +2.3% (gate excludes) |

**Gates:** parity test `cuda_q4k_decode_mmvq_parity` (od 5120 / id 3584 with
real-magnitude activations incl. near-zero blocks; plus REAL 7B q4_K_M
weights — blk.0 attn_q [3584²], attn_k [3584×512], ffn_gate [3584×18944] —
all bit-exact vs the independent dequant+q8 reference); 7B q4_k_m E2E @2K
greedy text identical to HEAD-base, decode **23.4 → 32.0 tok/s (+37%)**,
200-token × 3 runs each side; 0.5B q5_k_m / Qwen3-0.6B Q8_0 / 7B @440
regressions clean; cuda suite 156/0.

Follow-up candidate (not started): the same structure for **q6_K and q5_K**
decode (llama.cpp's GB10 table covers both; in-tree 8d attention is decode-
only already). The 8e "decode-side bandwidth work is closed" sentence above
is withdrawn — the correct statement is: the f32-activation streaming kernel
structure was the bottleneck; q8+dp4a MMVQ reaches ~200 GB/s of the 252.7
GB/s probe ceiling at the 7B shapes.

## 8e②. q6_K / q5_K decode MMVQ — **DONE 2026-08-30 (`1298cb2`, `1d28235`)**

**R2 (2026-08-31): weight-streaming rework of all three K-quant MMVQ
kernels.** The 8e kernels mapped one thread per 32-element sub-block and
read each 32B nibble chunk per sub (the sibling sub re-reads the same
bytes for the other nibble half — 2× the load instructions) and q6_K used
eight 2-byte loads per 16-byte ql/qh piece. v2 maps one thread to a
32-element chunk (q4_K/q5_K: a sub-pair sharing its nibble bytes — each
weight byte now loads exactly once per row; q6_K: an is-pair sharing
ql/qh bytes), uses uint4 vector loads everywhere the layout allows, and
the q5_K qh plane (32B shared by all subs, bit-indexed) is read without
the bogus per-chunk offset. Dispatch prefers v2 when `id % 256 == 0`
(q6_K additionally requires the padded 224B stride); `MINFER_MMVQ_V1=1`
forces the old kernels for A/B. 7B q4_k_m quiet-GPU A/B: tg128
42.2 → 45.1 tok/s, decode @2K 36.7 → 38.8 tok/s (llama.cpp 47.1 / 44.9);
parity extended to %256 shapes (the original tests' id=2176 exercised
only v1 — the graph-free greedy check caught a q6_K nibble-group bug the
suite missed), full suite 164 passing, greedy output v1 ≡ v2.

**R1 prefill note (2026-08-31)**: the int8 MMQ prefill GEMM landed
opt-in (`MINFER_MMQ=1`); pad40 q8 blocks now carry the per-block int sum
at byte 36 (MMVQ kernels read only d@0/payload@4..36 — unaffected).

The 8e follow-up candidate, ported to both remaining K-quant types with the
same llama.cpp GB10 structure (one row per 256-thread block, sub-block units
round-robin, `__dp4a` over `quantize_q8_0_pad40` activations, 8-warp block
reduce via the shared `mmvq_block_reduce` helper — the q4_K kernel keeps its
inlined copy):

- `q6_k_q8_mmvq` — 16-element units (16 **signed** scales per super-block,
  no min term), `vi = __vsubss4(nib|hi2, 32)` for the −32 bias; weight side
  reads 2-byte halves because the 210 B raw / 224 B padded strides are not
  4-aligned (llama.cpp `get_int_b2` approach); `blk_stride` follows the
  weight registration (loader weights are padded-224, runtime truth).
- `q5_k_q8_mmvq` — q4_K shape plus the q5 high-bit plane: **one qh word per
  nibble word** (`blk + 16 + 4*v`, bit = sub-block index); 176 B stride is
  16-byte aligned so uint32 loads work.
- Dispatch: decode (nt == 1) arms in `matmul_f32_ptr_layout`, shape-gated
  (see below). Prefill and sub-gate shapes keep the f32 kernels.

**Bugs found by the one-hot probe debugger (`dbg56.cu`: x = e_k ⇒ out must
equal w[k], since a one-hot quantizes to exactly 1.0):**
1. q5_K first draft hoisted `qh32` out of the nibble-word loop — elements
   ≥ 4 read qh bytes 16..19 regardless of v (k=40 decoded hibit from
   qh[16] instead of qh[24]; 109/256 one-hot mismatches, and repeated got
   values across different k). Fixed: per-word qh load.
2. q6_K kernel was CORRECT — the failing q6 parity test had its generator
   writing `d` at block offset 0 instead of 208 (`block_q6_K` field order
   is ql[128], qh[64], scales[16], d). The one-hot probe passed on random
   bytes because ref and kernel read the same garbage; the structured test
   (meaningful d) exploded. Fixed the test, not the kernel.

**Shape gate (`1d28235`) — the dispatch crossover is work-size dependent.**
A CUDA-events micro-bench (padded f32 kernel vs MMVQ, 200 reps/shape) shows
the 2-byte weight loads only win once latency is hidden:

| shape | f32 padded | mmvq | ratio |
|---|---|---|---|
| 7B attn_v (od 512 × id 3584) | 8.2 µs | 37.7 µs | **4.6× slower** |
| 0.5B ffn_down (896 × 4864) | 13.0 µs | 38.8 µs | **3.0× slower** |
| od 2048 × id 4864 | 26.8 µs | 44.5 µs | 1.66× slower |
| od 3584 × id 8192 | 95.1 µs | 87.7 µs | 0.92× |
| 7B ffn_down (3584 × 18944) | 507 µs | 333 µs | **0.65× (1.5× faster)** |
| 7B lm_head (152064 × 3584) | 3124 µs | 2200 µs | **0.70× (1.4× faster)** |

Gate: `nt == 1 && od*id >= 24_000_000` (+ `id % 32 == 0` for q6_K); below it
the f32 kernels keep their coalesced-loop win (the earlier `id >= 2048`-only
gate put 7B attn_v and 0.5B ffn_down on the losing side). `MINFER_NO_KQ_MMVQ=1`
forces f32 for A/B. A debugging detour worth recording: 0.5B wall-clock A/B
under a co-resident sglang server (96% GPU util) swung 15–52 tok/s for the
SAME binary — only the per-kernel event timing and the 7B interleaved A/B
were trustworthy.

**Gates:** synthetic parity (full 6-bit / full 0..255 scale-byte coverage,
partial tail super-block id=2176; q5 exercises `get_scale_min_k4` splicing)
via direct decode calls — small shapes intentionally bypass the gate; REAL
weights bit-exact 0.0000 — 7B ffn_down full shape (3584×18944) **through the
graph dispatch** (gate wiring), 7B attn_v + 0.5B ffn_down by direct calls;
7B q4_k_m E2E @2K greedy text identical to base, decode 28.8 → 31.1 tok/s
median (interleaved A/B ×3; on top of 8e's 23.4 → 32.0); Qwen3-0.6B Q8_0 and
0.5B q4_0 bit-identical perf (199.7 / 260.1 tok/s both binaries); cuda suite
158/0. No local model carries Q5_K tensors (the "q5_k_m"-branded 0.5B GGUF
stores Q5_1/Q8_0/Q6_K — confirmed by dump), so q5_K parity rests on the
synthetic plus the shared kernel structure.

Follow-up candidate (not started): llama.cpp's shape-dependent `halve_iters`
(idle-tail rule: double warps when `idle*8 <= iters_wide*2`) — the in-tree
kernels use the fixed 256-thread/1-row layout only.

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

## 8g. Prefill capture — **8g① DONE (8a batch); 8g② DONE 2026-08-29 (`eb24054`); R3-B default flipped ON 2026-08-31 (`761e236`)**

Two findings from the Phase 8 completeness audit (2026-08-29):

1. **Immediate hygiene (shipped with 8a):** `graph_replay_step` has NO nt
   gate — the scheduler runs the 3-run capture protocol for EVERY CUDA
   split, so a repeated identical-nt prefill (server/slot scenario) would
   silently start capturing a ~437-node graph. → Fixed: decode-only
   capture gate (`ComputeGraph::capture_nt_hint()`, ships with 8a).
2. **Productization — DONE (`eb24054`):** prefill capture is now a
   DELIBERATE opt-in (`MINFER_CAPTURE_PREFILL=1` /
   `set_prefill_capture_for_test`) rather than accidental: a repeated
   identical-nt prefill split captures after the same 3-run protocol.
   Gate: `cuda_prefill_capture_bit_parity_pp16_pp300` — captured prefill
   replays bit-identical to direct launches at pp16 AND pp300
   (captured_count == 1 each); real-model smoke unchanged output.
   Default OFF (the 8g① no-capture assertion still holds).
3. **R3-B — default ON (`761e236`, 2026-08-31):** with the pp16/pp300
   parity harness green and R3-A1 making the real prefill graph a single
   split, the gate now defaults ON (3-run protocol still bounds the cost;
   a one-shot CLI prefill never captures). `MINFER_NO_PREFILL_CAPTURE=1`
   opts out; `MINFER_CAPTURE_PREFILL=1` is redundant but accepted. The
   8g① negative test now drives the opt-out via
   `set_prefill_capture_for_test(false)` (env is process-global and the
   suite runs in parallel); `cuda_prefill_capture_defaults_on` pins the
   new default.

## 8h. Infra / process

1. **Stale docs**: `CUDA_OPTIMIZATION.md` / `CUDA_PROBLEMS.md` — **DONE
   2026-08-29 (`60e9cc1`)**: marked SUPERSEDED with pointers to the current
   plans (absorbed ideas named: cuBLAS → 8k, MMQ tiling → 8e, GPU quantize → 8c).
2. **Optional CUDA CI runner** — device-gated tests skip gracefully today;
   a self-hosted GB10 runner would keep the test suite honest on every
   commit. **DEFERRED (2026-08-29):** requires standing self-hosted runner
   infrastructure (a wired-up machine + runner registration) — not
   achievable from a dev session; the 158-test device suite runs green
   locally (cuda 158/0).
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

## 8l. llama.cpp CUDA parity benchmark — DONE 2026-08-30 (`acca28f`)

Cross-benchmark vs llama.cpp `ca3d5a3e1 (10665)`, same GGUF files, GB10,
`-ngl 99 -t 8`, llama-bench `-r 3` (FA 0/1 matrix) + llama-cli cross-check;
minfer side 3 reps `--greedy`. Raw logs: `/tmp/minfer_llamacpp_cmp/`.

**Found + fixed the Q5_K registration gap first**: the CUDA whitelist in
`models/qwen2/loader.rs` (added 7c) was missing `TensorType::Q5_K` (Metal's
list had it). Every Q5_K matmul silently ran on CPU with per-token
GPU↔CPU copies — 0.5B q5_k_m decoded at 51.6 tok/s with the
`CUDA GATE: ... not registered on CUDA` spam, and the 8e② q5_K decode MMVQ
was unreachable on real models. One-line fix → 246.3 tok/s (4.8×).

Decode (tok/s, llama-bench tg128 / minfer -n 256; short context):

| model | llama fa0 | llama fa1 | minfer | gap (fa1) |
|---|---:|---:|---:|---:|
| 0.5B q4_0   | 417.6 | 453.5 | 258.0 | 1.76× |
| 0.5B q5_k_m | 311.0 | 394.6 | 246.3 | 1.60× |
| 0.6B q8_0   | 273.8 | 290.3 | 197.8 | 1.47× |
| 7B q4_k_m   | 46.4  | 47.1  | 41.2  | 1.15× |

7B @2K context: llama.cpp 44.9 (llama-cli) / ~45.1 (bench `-pg 2048,128`
derived) vs minfer 31.3 → 1.43×. Depth penalty llama −5% vs minfer −24% —
the attention/KV path loses ~3 ms/token at 2K.

Prefill (tok/s): llama.cpp pp (MMQ / dequant-GEMM, weights streamed once)
vs minfer **110× (7B q4_k_m: 3401 vs 30.7)**, 69× (0.6B q8_0), ~18–30×
(0.5B). Root cause: minfer's quantized prefill reuses the decode-shaped
kernels with `grid.y = nt` — every token block re-streams the full weight
matrix (7B: ≈4.4 GB × 1920 tok ÷ 62 s ≈ 135 GB/s effective, pure redundant
traffic). The 8c Q8_0-activation GEMM only covers q4_0/q8_0 shapes ≤ 8192
and is itself far from MMQ.

Attribution of the decode gap: at 7B both engines are bandwidth-limited
(llama ≈221 GB/s ≈ 81% of the 273 GB/s peak, minfer ≈193 GB/s ≈ 71%); the
residual 15% + the small-model 1.5–1.8× are per-token overhead (graph
replay is worth +24% on 0.5B: 208 → 258 with `MINFER_NO_CUDA_GRAPH=1`,
launch/sync/sampler CPU time) plus llama.cpp's FA-style single-pass decode
attention vs minfer's multi-pass scores kernel.

Follow-up candidates (ranked by measured impact): ① prefill MMQ-style
tiled int8 GEMM (or dequant→f16 tensor-core GEMM) — closes the 18–110×
prefill gap; ② FA-style decode attention (single KV pass, online softmax)
— the @2K gap and part of the small-model gap; ③ per-token CPU overhead
audit on small models (sampler + sync path).

## 8m. Prefill GEMM: tiled wmma f16 — DONE 2026-08-30 (`ba3f317`, `65b686c` follow-ups)

Root cause of the ~110x prefill gap (7B @2K: 30.7 vs llama 3401 tok/s): the
decode-shaped quant kernels used `grid.y = nt`, re-streaming the whole weight
matrix once per token. Fix: for `nt >= 16 && id % 32 == 0` dequantize the
weight to f16 scratch once per call and run ONE 64x64-tile wmma f16 GEMM
(f32 accum) over all eight supported quant types.

- Kernels (`src/cuda_kernels.cu`): 8 per-type `dequant_*_f16` kernels,
  `convert_f32_f16_kernel`, double-buffered `gemm_f16_nt_kernel`
  (grid.x = nt tile, grid.y = od tile so consecutive blocks share the B
  panel in L2), per-type `block_stride` (Q6_K padded = 224).
- v1 bug: only the first 16 k's of each 32-slice fed the mma (garbage
  prefill) — fixed with fa[4]/fb[2] fragments, 4 mma per k-step.
- 8m-2 (`65b686c`… actually `ba3f317` + the cp.async commit): `cp.async`
  tile staging (arch >= 800; sm_75 keeps the synchronous loader) —
  31 -> 35 TFLOPS, 7B @2K prefill 1082 -> 1204 tok/s.
- Test `cuda_prefill_f16_gemm_parity`: all 8 types + real 7B Q4_K weight;
  the Q6_K "failure" was a test-reference bug (qh field offset +128).
- Env: `MINFER_NO_PREFILL_GEMM=1` reverts.

| model | prefill @2K before | after | llama.cpp |
|---|---|---|---|
| 7B q4_k_m | 30.7 | 1201 | 3401 |
| 0.6B q8_0 | 346 | 4585 | 23909 |
| 0.5B q4_0 | 1721 | 2908 | 30550 |

Remaining gap to llama: their MMQ path quantizes activations to int8 and
uses int8 tensor cores (~52 TFLOPS-equiv); our f16 wmma ceiling measured
~35 TFLOPS. Next lever: fused dequant-in-GEMM (kills the 288 ms dequant +
54 ms convert passes and shrinks B traffic 3.4x, projected ~1600 tok/s),
then an int8 MMQ GEMM for full parity.

## 8n. FA-style tiled prefill attention — DONE 2026-08-30 (`cb66fca`)

nsys on the 7B @2K prefill showed `gqa_attn_f32_f16kv` at 76% of GPU time
(176 ms/layer): one block per (token, head) re-read K per token per head
(~132 GB/layer) with a 128-register accumulator (spills).

New `fa_prefill_f16kv` (hd == 128, nt >= 64; gated by
`MINFER_NO_FA_PREFILL=1`): one block per (64-token q tile, head), K/V tiles
staged in shared memory, QK^T on tensor cores (wmma), online softmax with
probs in f16, O accumulator in per-thread REGISTERS (thread owns a
(row, quadrant) pair — V reads become warp broadcasts; the first
shared-memory O accumulator version hit 8-way bank conflicts, 123 ms/layer).

- Subtle race found by the standalone harness: f16 probs aliasing the f32
  score tile at a 128B row stride clobber OTHER softmax threads' unread
  scores (probs row r lands on scores of rows 2r/2r+1). Fixed with a 256B
  probs stride (row r's probs overlap only Sf row r, already read).
- Result: 176 -> 8.5 ms/layer; 7B @2K prefill contributed ~4.9 s -> 0.24 s.
- Test `cuda_fa_prefill_attention_parity` vs `cpu_gqa_attn` (max err 2.8e-4).

## 8o. One-time decode-start CPU stalls — DONE 2026-08-30 (`65b686c`)

A -n sweep (fixed = total - n x marginal) exposed ~650-920 ms of
pure-CPU time at the prefill->decode graph switch, invisible to GPU
profiling (no kernels, no CUDA API calls):

1. `register_graph_weights` re-cloned every weight tensor on every graph
   rebuild; `Tensor.data: Cow::Owned` makes `t.clone()` a deep copy —
   4.4 GB per rebuild on 7B (~635 ms). Fix: `CpuBackend::register_weight`
   skips already-registered names (weights are immutable after load;
   weights_version guards future changes).
2. The decode-graph build probed FFN concat availability via
   `cuda::concat_rows`, which eagerly REBUILT the concatenated bytes
   (~1.9 GB, ~920 ms) just to call `.is_some()`. Fix: metadata-only
   `cuda::concat_rows_feasible` (the loader already builds + registers the
   concat once at load). Metal path untouched.

Result: first decode step 724 -> 35 ms; every model's decode rate unchanged
(7B 40.6, 0.6B 195, 0.5B q4_0 257 — all within noise of baseline).

## 8p. Prefill GEMM: persistent f16 weights + fused dequant-in-GEMM — DONE 2026-08-30 (`2992f57`)

Two-pass 8m dequantized W into an f16 scratch on EVERY call (288 ms per
7B @2K forward) and the fused-GEMM experiment that replaced it both start
from the same observation: the weight bytes never change, so the f16 form
should exist exactly once per weight, not once per call.

- **Persistent per-weight f16 cache (default)**: `CudaState::w16_cache`
  keyed by the registered device pointer (stable: `register_weight`
  reuses the device copy for same name+size and never frees on replace).
  The loader warms it at LOAD time — a lazy first-call fill put ~8.6 GB of
  cudaMalloc inside the first, timed, prefill and measured WORSE than
  baseline (1165 vs 1201 tok/s). Gated to models whose quantized matmul
  weights total >= 2 GB (`W16_ENABLE_BYTES`): the CUDA test suite keeps
  several loaded models resident on a shared OVERCOMMITTED CUDA pool
  (cudaMemGetInfo free ~= 24 GB of a 130 GB total; reservations succeed
  far past free), and a +1-2 GB cache per loaded fixture made later
  models' weight uploads OOM probabilistically. `MINFER_NO_W16CACHE=1`
  reverts. 7B q4_k_m @2K prefill: 1201 -> ~1400-1495 tok/s.
- **Fused dequant-in-GEMM (opt-in `MINFER_FUSED_B=1`)**:
  `gemm_qb_nt_kernel` dequantizes B tiles in-register from raw quantized
  bytes, same 64x64 wmma tile structure, bit-identical rounding to the
  dequant kernels. Measured SLOWER than the cp.async f16 GEMM on large nt
  (994 vs 1201 on 7B @2K): every nt tile sweep re-dequantizes the whole B
  panel (30 sweeps at nt=1920) and cp.async cannot convert/dequantize, so
  the load pipeline goes synchronous. Kept as the memory-lean alternative
  (no 2 B/element resident copy).
- **Latent bug found by the new bitparity test**:
  `cuda_prefill_fused_b_bitparity` (fused vs legacy bit-equality, all 8
  types x 2 super-block configs) caught cudaErrorMisalignedAddress (716)
  in `dequant_q5_0_f16` — a u32 qh load at blk+2 on 22-byte blocks is
  only 2-byte aligned for odd block indices. Any Q5_0 prefill GEMM would
  have crashed; fixed there and in the new `bqa_q5_0/q5_1` (two u16
  loads). The old parity test never exercised Q5_0/Q5_1.
- Remaining prefill gap to llama.cpp (3401) is the f16 tensor-core
  ceiling (~35 TFLOPS) vs their int8 MMQ (~52) — P2, see 8m.
- **R1 (2026-08-31, opt-in `MINFER_MMQ=1`)**: the int8 MMQ prefill GEMM
  landed as a custom kernel on the 64×64 tile skeleton implementing
  llama.cpp's MMQ math (q8_0 pad40 activations + the block int sum at
  offset 36, raw quantized weights, `mma.m16n8k32` s8, per-(token,row,
  k-block) rescale `da·ds·acc + da·dm·sa`; q6_K = k32 chunks with dual
  m16n8k16 + dual 16-sub rescale). Parity-verified on all 8 types × 8
  shapes vs a host CPU q8_0-activation reference (max diff < 1e-3);
  greedy 7B output identical to the f16 path. NOT default: measured
  ~2.9 TMAC/s per matmul (412 tok/s @2K quiet, 155 under sglang load)
  vs the f16 path's 1460 — the ~8× gap to llama.cpp's MMQ (~24 TMAC/s on
  this part) is unprofiled (ncu cannot sample GB10) and is the next
  lever: bigger per-warp output tiles + coalesced wide staging loads.
  Layout notes: pad40 q8 blocks now carry `int32 sum` at byte 36 (was
  slack — MMVQ decode kernels read only d@0/payload@4..36, unaffected);
  MMQ skips the w16 warm pass when active (`mmq_active()` = cc ≥ 800 &&
  `MINFER_MMQ=1`); staging depth 8×32-k (~94 KB dynamic shared, opt-in)
  measured better than 4×32-k (2 blocks/SM) under load.

## R4. Decode split-attention: dim-parallel lane rewrite — DONE 2026-09-01

8d's flash-decoding pass was latency/local-memory bound: the
runtime-indexed `float4 oc[32]` accumulator lived in local memory
(~80 MB/layer of local traffic, re-touched on every online-softmax
rescale), each lane walked whole K/V rows with 4-byte loads (64 scattered
sector requests per row, 12.5% sector utilization), and 224 single-warp
blocks left ~4.7 warps/SM — ~150 us/layer at 7B @2K, i.e. the entire
@2K-vs-tg128 decode delta. Naively raising the split count made it
monotonically worse (148 → 172 → 419 → 609 us/layer for 8/16/32/64
splits: more resident warps thrash L1 with the local oc arrays).

Rewrite: each lane owns 4 fixed dims (single-float4 register accumulator,
zero spill — `hd % 4 == 0 && hd <= 128` already enforced by the
dispatch), K/V loads are fully-coalesced row instructions, the row dot
is a warp reduction, rows run in batches of 4; `ATTN_SPLITS` 8 → 32
(fixed grid stays capture-safe; idle splits write -INF/0 partials the
combine weights to zero). 148 → 79 us/layer; 7B @2K decode 39.2 →
43.2–45.1 tok/s (llama.cpp 44.9), tg128 45.1 → 47.5 (llama.cpp 47.1).
Parity locked by the extended `cuda_attn_split_decode_parity`
(chunk-boundary nkv sweep × f16/f32 KV).

## 8k. Explicitly not planned (revisit only with a concrete need)

FP16 activations + cuBLAS/cublasLt (large-GEMM path), VMM pool,
multi-GPU + peer copies, `graph_optimize`-style node reordering, Windows,
IQ/Q2/Q3 quant families.

## Priority order

**8a → 8b → 8c → 8d → 8e → 8f → 8i → 8g → 8h → 8j** (8k stays closed).
The 8g① decode-only capture gate ships together with 8a.

Rationale: 8a is correctness debt from Phase 7 itself; 8b is the largest
decode lever with a proven Metal precedent; 8c is a promise to close with a
measure-first gate; 8d/8e chased the bandwidth/overhead headroom on 7B (8e
re-examined 2026-08-29 and reversed into the MMVQ decode win, see above); 8f
widens model coverage; 8g/8h are polish.

## P5 (Sep 1, 2026): prefill gap 2.37× → 1.43×

Landed (details in CUDA_OPTIMIZATION.md §P5): 8p elementwise
vectorization, 8q TM=128 GEMM tiles, 8r all-warp FA softmax, 8s padded
smem rows (fa 4.25 → 1.92 ms/layer) + a one-time warning on the FA
smem-cap fallback; k-step templating with a negative result for KS=64.
New debts: FA smem-optin fallback must stay loud; `MINFER_GEMM_K64=1`
and `MINFER_GEMM_TM=64` are the kept A/B knobs; next levers are f16-out
rms_norm/swiglu (~77 ms) and GEMM mma-level work (597 → 455 ms target).
