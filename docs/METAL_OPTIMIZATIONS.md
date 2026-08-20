# Metal Backend Optimizations

> **Goal**: match llama.cpp (commit `88b47a755`, Apple M4 Pro) performance.
> **Current gap (2026-08-17 same-model, same-parameter A/B)**: decode is 72-88 % of
> llama (0.5B pure GPU 1.1-1.4×; **7B decode now ≈ llama parity, ~19.3 ms/token
> GPU vs 50.5 t/s**, after the q4_K decode matmul port, to-do #7), long prefill ~2.6-2.7× (down
> from 2.8-3.6× via the prefill flash port, `5974eb1`). See [§1 Current state](#1-current-state).
>
> ⚠️ **The §0 progress table is the single source of truth for tracking**; §1-§6
> are the detailed explanations behind it. Update §0 first before changing code.

---

## §0 Progress Overview (single tracking source)

> Legend: `[x]` done (with commit) · `[ ]` to-do · `[—]` decided not to change.
> Commits are from `git log`; a few early items are marked "TODO-trace" (long
> history, to be filled in later).

### ✅ Done

| # | Item | Measured effect | Commit |
|---|---|---|---|
| 1 | Metal backend foundation + 4 correctness fixes (RoPE freq_scale / output_b / softmax max / stack array) | Qwen2-0.5B 130→334 t/s | `2473981` / `26f0e4d` (early; TODO-trace) |
| 2 | Q5_K formula (unsigned) + qh index fix | Q5_K_M CPU/GPU output correct | pre-`3f23560` (TODO-trace) |
| 3 | Q5_1 / Q5_K Metal matmul kernels | Q5_K_M fully GPU | `26f0e4d` / later (TODO-trace) |
| 4 | Q4_0 → f32 activations (aligned with llama Metal, removed Q8_0 quantize) | decode +5-10 % | `ba51f68` |
| 5 | GQA attention `simd_max` divergence fix (partial tiles) | long prefill output correct | `28d4ba2` |
| 6 | GPU-hang safety hardening (bounded wait + dispatch trace + barrier guards) | deadlock → error-exit | `bff73db` |
| 7 | Metal cb/encoder autorelease retain fix | bg-thread cb no longer asserts | `b1256d5` |
| 8 | Fused QKV + FFN gate/up matmuls (decode, nt==1) | decode ~5 % (24 % was a GPU-state artifact, corrected in `26b145b`) | `6f0c847` |
| 9 | KV-parallel split attention (decode, 2-pass) | 200-token 1.56→1.06 s (~32 %) | `b3d4c7a` |
| 10 | Attention float4 acc + adaptive chunks + KV geometric growth | extra ~15 % + long-context fix | `66f4290` |
| 11 | simdgroup GEMM: non-Q4_0 quants (Q8_0/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K) | K_M prefill 300→650 t/s | `c9f865c` / `2c03bd1` |
| 12 | Q4_1 simdgroup GEMM | every quant has a GEMM | `5b914f0` |
| 13 | f16 split attention (`MINFER_CACHE_TYPE=f16`) | f16 decode 1.60→0.95 s | `387d612` |
| 14 | float4 elementwise + parallel RoPE (P6/P7) | 200-token ~0.88→~0.80 s | `ddd3eb0` |
| 15 | CPU sampler speedup (top_k O(n) / top_p sorts only survivors) | sampling ~12.6-14.8→~5.5-6.5 ms/token (2×) | `192378d` |
| 16 | 256-thread RMSNorm + per-kernel profile + KV-growth fixes (chunk cap 32→16, drop sync_kv_to_cpu) | ~3 % decode + long-context ~0.25 ms/token | `a7f21e4` |
| 17 | Parallel prefill attention (3-pass, barrier-free) | pp430 212→144 ms (~32 %); 7B 944→832 ms | `b2c97fd` |
| 18 | `Generated:` pure-decode caliber + dual-caliber bench.sh | measurement credibility fix | `dc66d0d` |
| 19 | Q4_K AVX2 test-reference fix (implementation was already correct) | 29 bin tests green | `266ffb7` |
| 20 | Split-GGUF (7B multi-part) support | 7B loads and runs | `cbba68c` / `34eaf10` |
| 21 | Same-model, same-parameter A/B benchmark doc | gap baseline established | `09d27ae` |
| 22 | **Flash attention port** (llama `kernel_flash_attn_ext_vec`, NSG=1 fixed DK=DV=64) | decode GPU 0.25-1.0 ms/token faster; wall ~10 % (0.5B, byte-identical) | `2e0c8b3` |
| 23 | **Prefill gap root-cause** (2026-08-14): GEMM is NOT the prefill lever | see to-do #4 status | — |
| 24 | **Prefill flash attention port** (llama `kernel_flash_attn_ext_blk`, legacy simdgroup-matrix, NSG=4 fixed DK=DV=64) | prefill GPU ~110→~93 ms (~16 %); f32+f16 byte-identical to classic | `5974eb1` |
| 25 | **hd=128 (7B) prefill flash port** (2026-08-15, §4.3.3 done) | 7B pp310 prefill GPU 1042→**~949 ms (~9 %)**, f32/f16 byte-identical, fixes 7B f16-cache garbage | §4.3.3 |
| 26 | **hd=128 (7B) decode flash port** (2026-08-17, §4.2.2 extend) | 7B decode steady GPU ~51.1 ms vs split ~51.6 ms (pp205), f32+f16+split byte-identical | §4.2.2 |
| 27 | **q4_K decode matmul layout port (7B)** (2026-08-17, to-do #7, §4.2.1) | 7B q4_K dims 70→265 GB/s (attn_q), 18→146 (attn_k), 74→243 GB/s (ffn_g/u); **7B decode steady GPU ~51 → ~19.3 ms/token (~2.6×, now ≈ llama's 50.5 t/s)**; 7B/0.5B byte-identical | to-do #7 |
| 28 | **GEMM partial-tile race + missing Metal `memoryBarrier`** (2026-08-19, §4.3.6): (a) all 8 simdgroup mm kernels lacked the `threadgroup_barrier` BEFORE the partial-tile `temp_str` stores — `temp_str` overlaps sa/sb, so a fast simdgroup overwrites sa/sb while a slow one still reads them → intermittently corrupted last-2-token logits (partial x-tile only); (b) the single prefill encoder had NO `memoryBarrier` between dispatches → RMSNorm write raced QKV read of the reused `bn` buffer (last-2 token slots, huge stale values) | 1.5B/7B first-token nondeterminism (~10-30 % wrong tokens, dump-localized) → **24/24 deterministic, output matches CPU byte-for-byte** | uncommitted |
| 29 | **mm-kernel hot-loop `#pragma unroll`** (2026-08-19, §4.3.9): llama `FOR_UNROLL`s the staging/ik/load/mac loops; minfer's 8 mm kernels had none → added the 6 unroll points (llama-parity set) | 7B pp495 **1438.8 → 1355.6 ms (~5.8 %)**, 0.5B ~6.9 %, 1.5B ~2.4 %; byte-identical + 24/24 determinism | uncommitted |
| 30 | **ik-loop `threadgroup_barrier` → `simdgroup_barrier(mem_none)`** (2026-08-19, §4.3.9 follow-up): .air diff showed the pre-unroll corruption was a rolled-loop compiler artifact, not a memory need; with the unroll in place llama's exact barrier form is now safe | 7B pp495 min **1387.4 → 1370.6 ms (~1.2 %)**; byte-identical (1.5B×24 / 7B×8 / 0.5B×3); removes the last structural mm-kernel difference vs llama | uncommitted |
| 31 | **Phase-0 7B prefill decomposition (2026-08-20, §4.3.10)**: exact 7B MUL_MAT graph mapped (197 GEMMs, 7.000 TFLOP, wk/down/output = q6_K on the q4_k_m; CORRECTS the earlier "all q4_K except ffn_down=q6_K" assumption); llama GPU-busy measured by host timestamps (CB1 83 ms + CB0 964 ms ≈ 1043 ms @ pp495 = 6.71 TF clean window); every remaining factor refuted via an exact-shape replay harness (kernels A/B in-batch 6.21 vs 6.26 TF, grid/smem/buffer-mode/pooling/barriers free, interleave + 2-CB split hurt, weight data no effect, concurrent dispatch no benefit with the per-dispatch barrier) | exact-shape replay (minfer kernel, real 7B shapes, one CB) = **~1126 ms = 6.20 TF**, converging with llama under comparable system load; engine GEMM-only ~1240 ms (residual ~90-115 ms engine-vs-replay, unattributable); gap vs llama stays ~1.25× (consistent with §4.3.9) | — |

### 🔜 To-do (required path to match llama.cpp)

> Principle (2026-08-12): we do NOT accept the current state — whatever
> llama.cpp can achieve, minfer must too. The former "accept the architecture
> floor" verdict is revoked; §4 is the only action path.

| # | Item | Goal | Status |
|---|---|---|---|
| 1 | **GPU trace (minfer + llama): Performance Limiters + per-kernel** | per-phase bottleneck + per-op durations for both sides | **DONE 2026-08-13** (§4.1/§4.1.1): per-kernel + limiter comparison. Early "1.6-3.9×" numbers were trace-semantics artifacts (fused-vs-separate, mixed od); the clean isolation A/B (llama test-backend-ops perf) shows q5_0/q8_0 at parity and **q6_K ffn_down 3× slower** — fixed (this row) |
| 2 | **decode matmul per-call execution** | **q6_K 72→209 GB/s (llama 217); decode 4.27→3.72 ms/tok (~13%)** | **q6_K DONE 2026-08-13** (§4.2.1): ported llama's stride-2/float4 kernel layout; byte-identical + tests green. q5_0/q8_0 already at parity. q4_K next if a K_M model uses it |
| 3 | **flash attention port** | decode **attention** 42.8→~4-6 µs/layer (~7-10×) | **DONE 2026-08-14** (§4.2.2): ported `kernel_flash_attn_ext_vec` (NSG=1, DK=DV=64/NE=2/C=32) as `kernel_flash_attn_ext_f32/_f16`; KV-layout check PASSED (minfer `[nkv][nk*hd]` == llama physical layout — no cache rework). Isolation-verified (`tests/flash_attn_isolation.rs`: cos vs CPU >0.999 for nkv 1..4097 incl. partial/empty chunks; flash-vs-split cos=1.0 through the shared combine), A/B byte-identical (0.5B Q4_K_M f32 + Q4_0 f16 + 7B Q4_K_M), decode GPU 0.25-1.0 ms/token faster (interleaved MINFER_TIMING), wall ~10 %; no long-context regression. Gate: `MINFER_NO_FLASH=1` reverts to split |
| 4 | **prefill flash attention port** (llama `kernel_flash_attn_ext_blk`, legacy `simdgroup_matrix`) + prefill GEMM/small efficiency | prefill 2.3-2.8× → ~1.5× (135 → ~90 ms); GEMM/small 89→~44 ms secondary | **GEMMs RULED OUT 2026-08-14 (§4.3.1)**. Grid-shape probe (3.5-5.4 variance) + barrier/store experiments rule out the GEMM kernels (mem_none ≈ mem_threadgroup ~2-3 % and RACES in minfer). Real pp325 decomposition (0.5B Q4_K_M, MINFER_SKIP_ATTN): **attention 46 ms (34 %)**, everything-else 89 ms. llama pp320 = 47.7 ms total with attention only ~3 ms (6803 vs 6373 t/s `-fa on/off`). llama's prefill attention is `kernel_flash_attn_ext_blk` = **legacy simdgroup-matrix (has_simdgroup_mm, NOT the M5 tensor API)** — single fused kernel vs minfer's 3-pass. **PORT DONE 2026-08-14 (§4.3.2)**: `kernel_flash_attn_blk_f32/_f16` (fixed-shape NSG=4, Q=8, C=64, DK=DV=64, 7168 B shmem, inline causal mask, `kernel_kv_tail_pad` for the partial last block) + host `attn_flash_prefill`. Isolation-verified (`tests/flash_attn_blk_isolation.rs`: cos vs CPU >0.999 across 16 nt/nkv configs incl. partial blocks + GQA, f32+f16, deterministic), A/B **byte-identical to the classic `gqa_attn_f32`** at every layer (f32 AND f16 cache — maxabs 0.0), interleaved MINFER_TIMING prefill GPU **~110→~93 ms (~16 %)**, all 34 bin + 9 isolation tests pass. **Bonus: FIXES the f16-cache prefill 3-pass bug** (the 3-pass `kernel_attn_scores`/`kernel_attn_output` read the f16 KV cache as `float*` → garbage "!!!!!!"; the f16 blk kernel reads half K/V correctly). Default for hd==64 (0.5B/1.5B) and — since the 2026-08-15 hd=128 port — for hd==128 (7B) too. Gate: `MINFER_NO_PREFILL_FLASH=1` reverts to 3-pass. Non-attention 89 vs ~44 ms remains a secondary structural gap — **7B direct per-kernel GEMM A/B 2026-08-18 (§4.3.4): minfer GEMMs at 87-94 % of llama (parity). Phase 0 prefill decomposition 2026-08-18 (§4.3.5): GEMMs are 76 % (0.5B) / 88 % (7B) of prefill; small kernels only 10 % / 4 % → #1 fusion ceiling low. **Phase X 2026-08-18 (§4.3.6): §4.3.4's parity was a test-backend-ops measurement artifact — llama real prefill GEMMs ≈ 6.9 TFLOPS (467 t/s pp466) vs minfer ≈ 5.2 (≈1.33×). Concurrency, fusion, small kernels, dequant type, ik-loop barrier, sb-staging vectorization, and geometry ALL measured/verified — none explains the 1.33×; the gap is not addressable from minfer source (compiler-level only).** |
| 5 | **7B same-model A/B + per-step regression check** (0.5B is the research model; 7B is the user-facing one) | 7B decode/prefill gap vs llama quantified; no 7B regression from each step | **BASELINE 2026-08-14** (§4.4): 7B Q4_K_M pp252: prefill **~240 t/s (52 % of llama 461)**, decode **~18.8 t/s (37 % of llama 50.5)**, steady GPU 50.1-51.3 ms/token. 0.5B sanity: pp252 2010 t/s (33 %), tg32 243 t/s (83 %) — no regression. Baseline recorded for per-step checks |
| 6 | ~~decode small-elementwise efficiency~~ | — | **CLOSED 2026-08-13**: trace shows small-op parity (1.2-2.0 vs 1.3-1.9 µs) — the old 4× claim was subtractive noise (§4.2.3) |
| 7 | **q4_K decode matmul layout port (7B)** — the next decode lever | 7B decode 37 % → closer to llama (steady GPU ~50 → target ~30 ms/token) | **DONE 2026-08-17** (§4.2.1): 7B K_M decode matmuls are **Q4_K-dominated** (attn_q/k/output + ffn_gate/up all Q4_K; Q6_K only output/ffn_down/attn_v). Ported llama's `kernel_mul_mv_q4_K_f32_impl` stride-4/float4 layout into `kernel_q4_k_f32_matmul` (TG(32, nsg=2) dispatch, `sc16`/kmask nibble unpack — the scale/min high/low nibble interleave reproduces llama's `get_scale_min_k4` exactly, verified against llama's dequantizer). Steps: ① isolation probe at 7B dims — old kernel 70/18/74 GB/s (attn_q 3584/3584, attn_k 3584/512, ffn_g/u 18944/3584) → new **265/146/243 GB/s** ② kernel port ③ 7B steady-decode A/B: **~49.7 → ~19.3 ms/token GPU (~2.6×), 17-19 → 45-49 t/s (≈ llama's 50.5)**, git-stash A/B byte-identical ④ 0.5B (no q4_K weights — trivially identical) + 1.5B (q4_K decode + multi path) regression green; all 34 bin tests pass |

> Note: §4.1.1's per-kernel table is superseded by the clean isolation A/B
> (§4.2.1): the trace mixed fused-vs-separate and different od per kernel name.
> The reliable decode gap = q6_K ffn_down kernel saturation (now fixed).

### ❌ Decided not to change (has measured or llama-source evidence)

| # | Item | Evidence |
|---|---|---|
| 1 | 2D `simdgroup_matrix` (mpp tensor) GEMM port | llama disables tensor GEMM on M4 Pro (PARAMETER_AUDIT A) — not llama's advantage |
| 2 | bf16 / f16 intermediate activations | Core convention #1: llama Metal reads f32 activations (only KV → f16) |
| 3 | non-blocking multi-cb | minfer encode already hidden (0.13 ms); `MINFER_SPLIT_CB` measured linear regression |
| 4 | parallel command buffers (A1) | measured regression (1.67/1.08/1.43 s vs serial 0.93 s), reverted |
| 5 | dispatch fusion (store_kv_both / residual_rms_norm) | measured no gain (1.79 vs 1.74 s), reverted |
| 6 | nt==1 matmul rewrite (full-block matvec) | measured at ~200 GB/s bandwidth floor, no gain |
| 7 | Q6_K / Q4_K dequant vectorization | Q5_0 full vectorization only +2.6 % — same low-value class |
| 8 | f16 KV cache as default | 0.5B measured ~3 % slower (dispatch-latency-bound), kept opt-in |

---

## 1. Current state

### 1.1 Same-model, same-parameter A/B (2026-08-14, M4 Pro, identical GGUF)

minfer `--greedy` (pure decode, llama "Generation" caliber); llama.cpp
`llama-bench -b 512 -t 8` (pure eval). Model Qwen2.5-0.5B-Instruct. Prefill
numbers updated after the prefill flash port (`5974eb1`): minfer pp430-435 uses
the `kernel_flash_attn_blk_f32` path (hd==64).

**Q4_K_M** (`qwen2.5-0.5b-instruct-q4_k_m.gguf`):

| Test | llama.cpp | minfer | Gap |
|---|---|---|---|
| prefill 30 tok | 2720 t/s | ~580 t/s (pp35, flash; short-prompt fixed overhead dominates) | — |
| prefill 430 tok | 6909 t/s | ~2530-2620 t/s (pp435, flash) | **2.6-2.7×** |
| decode 128 tok (pure GPU) | 293-299 t/s | ~218 t/s (4.47 ms/tok steady) | **1.3-1.4×** |
| decode, default sampling | 247 t/s | ~197 t/s | **1.25×** |

**Q4_0** (`qwen2.5-0.5b-instruct-q4_0.gguf`):

| Test | llama.cpp | minfer | Gap |
|---|---|---|---|
| prefill 30 tok | 2610 t/s | ~680 t/s (pp35, flash) | — |
| prefill 430 tok | 7449 t/s | ~2770 t/s (pp435, flash) | **2.7×** |
| decode 128 tok | 314-339 t/s | ~279 t/s (3.90 ms/tok steady) | **1.1-1.2×** |

**Reading**:
- **Decode is now 72-88 % of llama** (pure GPU 1.1-1.4×, default sampling 1.25×) —
  driven by rms_norm_256, the chunk-cap/sync fixes, and the per-kernel non-matmul
  profile (was 1.47× before 2026-08-10).
- **Prefill (long) improved from 2.8-3.6× to ~2.6-2.7×** after the prefill flash
  port (§4.3.2): GPU ~164 → ~144 ms at pp435 (~12 %; pp294 ~16 %). The residual
  gap is the non-attention 89 → ~44 ms structural difference (GEMMs + small
  kernels under-occupied, §4.3.1), NOT attention (now ~3-4 ms, llama-like).
- **Short prefill (pp30) is dominated by per-dispatch fixed overhead** (~580-680
  t/s regardless of attention path — flash/3-pass/classic all equal), so pp30 is
  no longer a meaningful attention lever; llama's pp30 is similarly launch-bound.

### 1.2 Per-token GPU decomposition (decode, nt==1, Q4_K_M 0.5B)

| Category | minfer GPU | llama GPU | Evidence |
|---|---|---|---|
| matmul (QKV/O/GU/down/output, ~97 kernels) | **~3.0 ms** (~130 GB/s) | ~3.0 ms (source+params identical) | minfer measured / llama inferred |
| attention (split 2 kernels) | **0.54 ms** | ~0.15-0.2 ms (flash vec 1 kernel) | minfer measured (skip-ATTN) / llama inferred |
| small elementwise (norm/bias/rope/store/add/swiglu, ~300) | **~0.5 ms** | ~0.1-0.3 ms | minfer measured / llama inferred |
| base infra (encode+submit+download) | encode 0.13 + download 0.02-0.03 | ~0.3-0.5 (incl. multi-cb encode) | minfer measured / llama inferred |
| **Total** | **~4.35-4.55 ms/token GPU** | **~3.1-3.3 ms GPU / 3.51 wall** | interleaved A/B |

> **SUPERSEDED 2026-08-13 by the per-kernel trace (§4.1.1)**: the llama-side
> numbers here were *inferred* (no per-op timing existed). The trace shows the
> real picture differs: matmuls are NOT "zero gap" (minfer 1.6-3.9× slower per
> call), and small elementwise is NOT "4× slower" (parity). The Total is right;
> the category split was not.

### 1.3 Whole-pipeline comparison (decode token, nt==1)

| Stage | minfer | llama.cpp | CPU/GPU |
|---|---|---|---|
| Sampling | `sampler.rs` top_k/top_p/temp/repeat-penalty (O(n) + candidate list) | `llama-sampler.cpp` candidate chain (partial_sort) | **CPU** |
| Dispatch encode | `MpsCommandBuffer` set_buffer/set_params ×N, single-threaded | same; multi-cb threads hide encode | **CPU** |
| GPU execution | single cb serial ~483 dispatches (Q4_K_M) | single/multi cb, ~490-530 dispatches | **GPU** |
| Embedding | Q4_0: `kernel_get_rows_q4_0`; others: CPU embed + upload | `ggml_get_rows` → Metal | GPU (or CPU upload) |
| KV store | `kernel_store_kv_f32`/`_f16` (2 dispatches) | `kernel_cpy_f32_f16` ×2 (K,V) | GPU |
| Attention | **2 kernels/layer** (partial + combine) | **1 kernel/layer** (flash vec) | GPU |
| Logits readback | `copy_from_gpu` 607 KB | Metal buffer read | GPU→CPU |

### 1.4 Per-layer kernel sequence (Q4_K_M 0.5B, nt==1)

**minfer — 20/layer** (fused QKV OFF: Q5_0/Q5_0/Q8_0 mixed types, cannot concat):

`RMSNorm → Wq/Wk/Wv 3×matmul → 3×add_bias → 2×RoPE → 2×KV store →
attention split(partial+combine) → Wo matmul → residual → RMSNorm →
fused gate+up matmul → SwiGLU → Ffn_down matmul → residual`

×24 = 480 + output_norm 3 = **483**. Q4_0 model (all Q4_0): fused QKV + BSR active →
**12/layer** ×24 + output 3 + GPU embed 1 = **292**.

**llama.cpp — 17/layer** (flash_attn on):

`RMSNorm → Wq/Wk/Wv 3×matmul → 2×RoPE → 2×KV store(f32→f16) →
flash attention(1 dispatch) → Wo → residual → RMSNorm → gate+up 2×matmul →
SwiGLU → Ffn_down → residual`

×24 = 408 + output 3 + embed 1 ≈ 412 base; graph 822 nodes → **~490-530
dispatches** (f16 cast/cont/reshape are non-no-op nodes).

### 1.5 Early performance milestones (Qwen2-0.5B, historical)

| Phase | Optimization | Decode (short) | Cumulative |
|---|---|---|---|
| Baseline | GPU + 4 correctness fixes | 130 tok/s | 1.0× |
| +2 | Flash Attention + float4 | 151 tok/s | 1.2× |
| +3 | SIMD-parallel attention | 196 tok/s | 1.5× |
| +4 | SIMD-parallel RMSNorm | 334 tok/s | 2.6× |
| +5 | SwiGLU fusion | 312-334 tok/s | 2.5× |

(Early numbers used the blended caliber; since 2026-08-06 `Generated:` is pure decode.)

---

## 2. Gap analysis: verified vs inferred

> **Core finding (2026-08-06 #5, verbatim)**: "Structural" is an **inference, not a
> proven architectural inferiority**. §2.1 is what is strictly VERIFIED; §2.2 is
> the inference after elimination. The decisive per-kernel measurement was
> **completed 2026-08-13** (§4.1.1) — see its per-kernel table, which refuted
> several of the §2.2-§2.4 inferences.

### 2.1 Strictly verified

1. matmul kernel **source** is line-for-line identical (nt==1
   `mul_vec_q_n_f32_impl` / `block_q*_dot_y` translations).
2. dispatch **count** is comparable (~436 vs ~490-530).
3. dispatch **params** match (llama `ggml-metal-impl.h` N_R0/N_SG vs minfer,
   2026-08-06 #6).
4. dispatch count is nearly identical (~484 vs ~490-530); the gap is per-dispatch
   GPU execution time (10.3 µs vs 6.2 µs).
5. prefill GEMM ceiling **~5.4 TFLOPs/s** (`prefill_gemm_throughput_profile`,
   2026-08-11 A1); llama ~7 TFLOPs/s effective.

### 2.2 Closed hypotheses (2026-08-06 #6)

1. **Dispatch params** — DISPROVEN (see 2.1.3).
2. **Attention is the main decode lever** — DISPROVEN: llama `-fa on/off` = 3.64 vs
   3.88 ms (only ~0.25 ms); even flash OFF llama beats minfer.
3. **Multi-cb is a lever** — DISPROVEN: minfer encode is only 0.13 ms;
   `MINFER_SPLIT_CB=N` regresses linearly (0.67 → 0.93/1.23/1.62 s).
4. **CPU side (encode/sampler)** — DISPROVEN: encode 0.13 ms; sampler fixed (2×).
5. **Q5_0 scalar dequant is the matmul bottleneck** — DISPROVEN: full vectorization
   only +2.6 %; ~130 GB/s is nt==1 small-grid structural latency.
6. **Matmul is the prefill bottleneck (before the attention fix)** — 2026-08-11
   proved classic attention ~100 ms (48 %) was; matmuls became the main remainder
   after the fix.

### 2.3 Per-kernel non-matmul profile (2026-08-10, P0)

`metal.rs::tests::non_matmul_bandwidth_profile` (batched-cb, median of 3) — a
single dispatch is dominated by the ~165 µs cb launch+sync floor, so batch dozens
and take the median:

| Kernel | µs/dispatch | Notes |
|---|---|---|
| rms_norm 32t (1 simdgroup) | **13.8** | 7× elementwise — latency-bound |
| rms_norm 256t (8 simdgroups, P1) | **3.7** | ~3.7× faster, bit-identical |
| add_f32 / add_bias / swiglu / rope / store_kv | ~1.6-2.3 | 256-thread elementwise baseline |
| attn_bias_rope_store (BSR) | 3.1 | |
| **attention split pair** (partial+combine, nkv=430) | **44.3** | dominant non-matmul kernel |
| attention classic (single-pass, nkv=430) | **352** | 8× worse than split — confirms the split design |

**Findings**: ① the attention split pair is the dominant non-matmul kernel
(44 µs/layer); only a faithful flash port can cut it. ② the small elementwise tail
(~300 kernels × 2-3 µs) is structurally cheap per-dispatch latency; rms_norm was
the one exception and is now fixed.

### 2.4 Final gap report (2026-08-06, precise decomposition) — SUPERSEDED

| Component | minfer GPU | llama GPU | gap |
|---|---|---|---|
| matmuls | ~3.0 ms (bandwidth-bound, source+params identical) | ~3.0 ms | **~0** |
| **non-matmul** (attention + small + serialization) | **~1.2 ms** | **~0.3 ms** (inferred) | **~0.9 ms** |

**The structural gap is 100 % in non-matmul** — minfer's ~340 small kernels run at
~4× llama's efficiency. (Honest uncertainty: minfer's 1.2 ms is reliable; the
attention-vs-small split has ±0.2 ms noise; llama's 0.3 ms is inferred.)

> **SUPERSEDED 2026-08-13 by §4.1.1's per-kernel trace.** The "~0 gap matmuls"
> and "small ~4×" were subtraction artifacts. Real per-kernel data: matmuls
> ARE the gap (1.6-3.9× per call) and small-op is at parity. This section is
> kept as the historical pinpoint that motivated the trace.

### 2.5 KV-growth component (partially fixed 2026-08-10)

minfer's average decode grows with context (5.05 → 6.7 ms at -n 64→512), llama's
stays flat. Two addressable causes were fixed: attention chunk cap 32→16 (avoid
over-parallelization) + removal of `sync_kv_to_cpu` on the pure-GPU paths (an
O(nkv)/token copy). Interleaved A/B: 4.65-4.76 → 4.50-4.55 ms/token
(~0.2-0.25 ms). The remainder is sub-linear attention KV-read, which llama
amortizes natively via its f16 cache + flash.

---

## 3. Completed optimizations in detail

### 3.1 Correctness fixes (Metal backend foundation)

4 early bugs (all affected output correctness, Qwen2-0.5B):
1. **RoPE freq_scale not applied** (`metal.rs` `rope_f32` got the param +
   `forward.rs` passes `hp.rope_freq_scale`).
2. **output_b not applied** (`output_norm_gpu` adds the bias).
3. **softmax max initialization** (`-INFINITY` instead of 0, prevents NaN).
4. **attention stack array hardcoded** (hd dimension made dynamic).

**Q5_K formula + qh index fix** (2026-07-31, affects CPU + Metal):
- Formula: Q5_0-style signed `dl*(u-16)-ml` was wrong → llama's **unsigned**
  `dl*u-ml`.
- qh high-bit index: `qh[sub*4+pos/8] bit pos%8` was wrong → **`qh[pos] bit sub`**.
- Fixed in `avx2.rs` / `kernel.rs` / `forward.rs` (embed).

**GQA attention `simd_max` divergence fix** (2026-08-01, `28d4ba2`): in a partial
KV tile (`nkv % 32 != 0`), out-of-range lanes exited the loop early → `simd_max(dot)`
ran across divergent lanes with stale registers → corrupted online-softmax running
max → repetition loops. Fix: uniform iteration count + `valid` mask (invalid lanes
`dot=-INF`, `e=0`). Result: prefill logits cos 0.83→0.999. Regression test
`tests/gqa_attn_isolation.rs`.

**GPU-hang safety hardening** (2026-08-03, `bff73db`):
1. `submit()` bounded 10 s wait + `MTLCommandBufferStatus` check + `MINFER_TRACE`
   dispatch trace (GPU fault errors out instead of freezing the machine).
2. attention kernels never return early before the barrier (prevents `nh % nk != 0`
   deadlock).
3. `layer_gpu`/`output_norm_gpu` runtime guards: `nh % nk == 0`, `hd ≤ 256`,
   `id % 32 == 0`; error-exit (`gpu_abort`) on violation.

### 3.2 CPU sampler (2026-08-06)

**Root cause**: `sampler.rs` ran a full-vocab O(n·log n) sort per token + 607 KB copy
(top_k) + a full sort of 151,936 `(usize,f32)` tuples (top_p). llama.cpp uses a
candidate-list chain (`std::partial_sort` O(n·log k) → later samplers operate only
on the ≤k survivors).

**Fix**:
- top_k → `select_nth_unstable_by` (O(n), on a copy to preserve the index→token
  mapping).
- top_p → softmax + sort only the ≤k survivors (falls back to the full-array path
  when >1024 survive).
- temp → skip exp() for masked (-INF) logits.
- `main.rs` moves the logits Vec instead of `logits_all[..].to_vec()` (607 KB/token).

**Measured**: -n 128/256/512 all **~2.0×**; default sampling ~12.6-14.8 →
~5.5-6.5 ms/token; fixed-seed output **byte-identical** (7 sampler tests pass).

### 3.3 Decode optimizations (GPU)

| Item | Effect | Commit |
|---|---|---|
| Fused QKV + FFN gate/up (nt==1 single matmul/group) | ~5 % decode | `6f0c847` |
| KV-parallel split attention (2-pass online-softmax) | ~32 % decode | `b3d4c7a` |
| float4 acc + adaptive chunks + KV geometric growth | extra ~15 % + long context | `66f4290` |
| f16 split attention (partial `_f16`) | f16 1.60→0.95 s | `387d612` |
| float4 elementwise + parallel RoPE (P6/P7) | ~2-3 % | `ddd3eb0` |
| 256-thread RMSNorm (llama `kernel_rms_norm_fuse_impl` port) | ~3-4 % | `a7f21e4` |
| Fused bias+RoPE+KV-store (BSR, 7→1 kernel, nt==1) | ~5 % | `5c106dd` |

**Split-attention design** (`b3d4c7a`):
- Pass 1 `kernel_gqa_attn_partial_f32`/`_f16`: grid (nt, nk, n_chunks); each TG
  computes an online-softmax partial (mx, S, acc) for its KV chunk, same
  tile/barrier/valid-head structure as the classic kernel.
- Pass 2 `kernel_gqa_attn_combine_f32`: grid (nt, nh) merges the partials (pure
  elementwise, no shared mem/barriers).
- `n_chunks = clamp((max_pos+1+31)/32, 1, 16)` (lowered from /16..32 on 2026-08-10;
  `MINFER_ATTN_CHUNKS` overrides). Correctness is invariant to chunk count.

**Fused QKV essentials** (`6f0c847`; the "24 %" figure was a GPU-state artifact,
corrected to ~5 % in `26b145b`): Wq/Wk/Wv and ffn_gate/up are row-major-concatenated
at load (`concat_rows` → `blk.{i}.attn_qkv` / `ffn_gu`) when types + input dim
match. nt==1 runs ONE matmul/group; rope/store/swiglu read sections via `set_buffer`
byte offsets. `MINFER_NO_FUSE_QKV=1` A/B byte-identical;
`gemm_isolation.rs::qkv_row_concat_layout` locks the layout.

**Measurement-trap notes** (avoid repeating):
- Single-dispatch isolation timing is **unreliable** (~165 µs cb launch+sync
  floor) — batch dozens and take the median.
- Each kernel's first timed run has a cold-start/GPU-clock-ramp artifact (~4×) —
  warm first, measure twice.
- Sustained benchmarking thermally throttles the M4 Pro (extreme: all configs
  ~1.3 s) — interleave configs, take min/median.

### 3.4 Prefill optimizations

**simdgroup GEMM (P0/P1, 2026-08-01)**: faithful llama legacy `kernel_mul_mm` port
(64×32 tile, 4 simdgroups × 32 threads, Q4_0 dequant staged into `sa`, f32
activations into `sb`). P0's initial version had 3 bugs (B-staging unclamped rows,
store transpose direction, barrier must be `mem_threadgroup`) → after P1 fixes:
+11 % at 30 tok, +34 % at 70 tok. Dispatched for nt ≥ 16; `MINFER_GEMM=0` falls
back to f32 multi. Isolation test `gemm_isolation.rs` (nt=12/30/32/33).

**Non-Q4_0 GEMMs (2026-08-03, `c9f865c`/`2c03bd1`/`5b914f0`)**: one simdgroup GEMM
per quant — Q8_0/Q5_0/Q5_1 (32-elem blocks) + Q4_K/Q5_K/Q6_K (256-elem
super-blocks). K_M prefill 300→650 t/s; 1.5B Q4_K_M 48→442 t/s (~9×). 8 KB
threadgroup-memory guard. `non_q4_0_gemm_isolation` verifies.

**Parallel prefill attention (2026-08-11, `b2c97fd`)**: classic
`kernel_gqa_attn_f32` is latency-bound at prefill (grid (nt,nk) sequential KV loop,
~24K barriers, ~100 ms = 48 % of prefill, ~25× llama's attention). Replaced by a
3-pass barrier-free design:
1. `kernel_attn_scores`: one 256-thread TG per (t,h) row; each thread computes one
   score.
2. `kernel_softmax_attn`: masked softmax over the kv axis.
3. `kernel_attn_output`: softmax·V sum.

GQA via per-head `hk = h/gqa` (the broadcast-GEMM idea was tried and abandoned — a
2D GEMM can't produce the per-head 3D scores tensor). ⚠️ threadgroup-memory bug:
the softmax's `shmem[tiisg]` writes 32 floats but only 8 were allocated (OOB
corrupted adjacent memory → NaN rows) — fixed to 32×4=128 B; rms_norm_256 had the
same latent bug.

**Measured**: pp430 classic 212 → **144 ms** (attention 100→30 ms); pp30 44→40 ms;
7B pp230 944→832 ms (attention 169→57 ms); 7B decode unchanged. 34 bin + 6
isolation tests pass, end-to-end byte-identical.

### 3.5 KV / long context

- **KV geometric growth** (`66f4290`): `kv_ensure_layer` grows ×2 instead of
  reallocating + copying the whole old KV every token (0.5 ms@KV140 → 4.2 ms@KV2510
  → 0.13 ms). ⚠️ an `old_v` clone typo polluted the V cache (Q4_K_M garbage) — the
  A/B didn't catch it (both paths share the corrupted KV); found against a
  known-good reference.
- **f16 KV opt-in** (`387d612` + `bff73db`): `MINFER_CACHE_TYPE=f16` (default f32).
  `kernel_store_kv_f16` + `kernel_gqa_attn_partial_f16`. 0.5B measured ~3 % slower
  (dispatch-latency-bound), kept opt-in for larger models / long context.
- **Split-GGUF** (`cbba68c`/`34eaf10`): multi-part models (7B `-0000X-of-0000Y`),
  merged tensor index, 7B verified.

---

## 4. Future work (the only path to match llama.cpp)

> 2026-08-12 statement: **we do not accept the current state.** Whatever llama.cpp
> can achieve (decode ~3.1-3.3 ms GPU/token, prefill ~7 TFLOPs/s GEMM) is minfer's
> goal. The following is the evidence-based action path, in order; update the §0
> progress table after each step.

### 4.1 Step 0: GPU trace analysis (DONE 2026-08-13)

**Result in one line**: with Counter Set = Performance Limiters, the Metal
System Trace DOES record per-kernel shader durations (`metal-shader-profiler-intervals`),
giving the first real minfer-vs-llama per-op GPU comparison. The 2026-08-13
early claim "per-kernel durations unavailable" was WRONG — it came from an
`--xpath`/parsing bug (id/ref dedup), now fixed in `scripts/export_trace.sh`.

**Correct xctrace path**: `/usr/bin/xctrace` is a broken stub ("tool not
found"); the real binary is
`/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace`.

**Workflow** (see `scripts/export_trace.sh`, written 2026-08-13):
1. Record in Instruments: template Metal System Trace, **Counter Set →
   Performance Limiters**, Enable Shader Timeline on, Deferred.
2. `scripts/export_trace.sh <trace> [run]` exports + summarizes:
   - `metal-shader-profiler-intervals` → **per-kernel durations** (kernel,
     count, total/avg µs, % GPU work) — the key table
   - `gpu-counter-value` → whole-run + per-phase limiter profile (bottleneck type)
   - `metal-gpu-intervals` → per-forward durations
   - `TRACE_PROC=<proc>` filters per-forward intervals (default `minfer`).
3. Interpretation: percentage-type counters (Limiter/Utilization/Occupancy);
   bandwidth counters (L1/LLC Read Bandwidth) are cumulative — ignore magnitude.

### 4.1.1 Per-kernel comparison — minfer vs llama (decode, M4 Pro, Q4_K_M 0.5B)

> **SUPERSEDED 2026-08-13 by the clean isolation A/B in §4.2.1.** The per-kernel
> numbers below mix fused-vs-separate matmuls and different od per kernel name
> (trace per-step aggregate semantics), so the ratios are NOT reliable. The
> isolation A/B (same od/id/nt, both engines) is authoritative: q5_0/q8_0 at
> parity, q6_K ffn_down 3× slower (fixed). Kept below as the raw trace record.

Both sides recorded with the identical Metal System Trace + Performance
Limiters config. minfer: `-n 128 --greedy` (run2); llama-cli: `-n 128 --temp 0`
(run1). **avg µs per kernel invocation** (the fair metric; step counts differ):

| kernel (minfer) | avg µs | kernel (llama) | avg µs | ratio |
|---|---|---|---|---|
| **q8_0 matmul** | **580.2** | mul_mv_q8_0_f32 | **353.2** | **1.6×** |
| **q6_k matmul** | 37.8 | mul_mv_q6_K_f32 | 15.5 | **2.4×** |
| **q5_0 matmul** | 32.0 | mul_mv_q5_0_f32 | 8.2 | **3.9×** |
| **q4_k matmul** | 31.4 | mul_mv_q4_K_f32 | 9.7 | **3.2×** |
| **attention partial+combine** | 15.2+4.3=19.5 | flash_attn_ext_vec | 5.8 | **3.4×** |
| add_bias | 2.0 | bin_fuse | 1.6 | 1.3× |
| rope | 1.4 | rope_neox | 1.3 | 1.1× |
| rms_norm_256 | 1.3 | rms_norm_mul | 1.9 | **0.7× (minfer faster)** |
| swiglu | 1.2 | swiglu | 1.6 | 0.8× |
| store_kv | 1.3 | set_rows | 1.4 | 0.9× |

**This repoints §4.2/§4.3 priorities — three surprises vs the 2026-08-06 model:**

1. **The decode gap is matmul-dominated, NOT attention/small-op.** minfer's
   nt==1 matmul kernels run 1.6-3.9× slower than llama's on the SAME dims and
   SAME source lineage (§2.1). q8_0 at 580 µs dominates decode GPU time. This
   contradicts the old "matmuls are identical / zero gap" claim (§1.2/§2.4) —
   the kernel source is the same but minfer's EXECUTION is slower per call.
2. **Attention is 3.4×** (19.5 vs 5.8 µs) — confirms the flash port value (§4.2.1),
   but it is now the SECOND lever, not the first.
3. **The small-op 4× gap is REVERSED**: minfer's small kernels are now equal
   or FASTER than llama's (rms_norm 1.3 vs 1.9, swiglu 1.2 vs 1.6). The
   2026-08-06 "small elementwise ~4× slower" (§2.4) is refuted by direct
   measurement — §4.2.2 is downgraded.

**Why matmuls are slower per-call despite identical source**: the limiter
profile shows decode is memory/cache-bound (LLC 64 %, MMU 49 % for BOTH sides;
llama decode LLC 62 %, MMU 49 %). Both engines read the same ~392 MB/token at
~130 GB/s — but llama reads it in **fewer, larger calls** (llama q8_0 = 199
calls for the whole run vs minfer 120; llama's mul_mv_ext/batch handling reads
more per dispatch). The per-call latency gap is the nt==1 grid/launch pattern:
minfer's single-od kernels are dispatched one matmul per call, llama batches
some (mul_mv_ext, R1×2) — see §4.3.

**Bottleneck profile (limiter, from the same traces):**

| Phase | Occupancy Target | LLC Limiter | MMU Limiter | ALU Util |
|---|---|---|---|---|
| minfer prefill | 97.8 % | 9.5 % | 3.7 % | 3.3 % |
| minfer decode | 76.6 % | **64.0 %** | 48.7 % | ~10 % |
| llama prefill | 99.6 % | 4.2 % | 2.3 % | 0.5 % |
| llama decode | 69.4 % | **62.1 %** | 49.3 % | ~10 % |

Both sides: prefill has NO hardware limit (GPU under-occupied — scheduling,
not compute); decode is cache/memory-bound with nearly identical LLC/MMU
pressure. The minfer-vs-llama decode gap is NOT a different bottleneck — it is
minfer's per-call matmul inefficiency under the same memory-bound regime.

**Open question (recorded, low priority)**: whether per-kernel intervals are
recorded for minfer was initially doubted; the export/parse bug made it look
empty. Now confirmed present. No further investigation needed.

### 4.2 decode gap: per-kernel comparison (REVISED 2026-08-13)

**The 2026-08-06 "non-matmul 1.2 ms ≈ 4×" model is REFUTED by the trace.**
§4.1.1's per-kernel measurements show:
- **matmuls are the main decode gap** (1.6-3.9× per-call, q8_0 dominates at
  580 vs 353 µs) — the old §1.2/§2.4 "matmuls identical, zero gap" is wrong;
- **attention is 3.4×** (19.5 vs 5.8 µs) — the flash port stays a real lever;
- **small elementwise is now equal-or-faster than llama** (rms_norm 1.3 vs 1.9,
  swiglu 1.2 vs 1.6) — the 4× small-op claim is refuted (§4.2.2 downgraded).

So the decode work splits into: (A) per-call matmul execution inefficiency
(§4.2.1) and (B) attention 3.4× (§4.2.2). Small-op fusion is closed.

#### 4.2.1 matmul per-call execution (the NEW primary lever)

**Clean isolation A/B (2026-08-13)**: the trace per-kernel table (§4.1.1) was
unreliable (fused-vs-separate, mixed od per kernel name). Replaced by a clean
apples-to-apples isolation harness: minfer `matmul_bandwidth_profile` (batched
cb) vs llama `test-backend-ops perf -b MTL0 -o MUL_MAT` at the SAME od/id/nt:

| matmul | od/id | minfer GB/s | llama GB/s | gap |
|---|---|---|---|---|
| q5_0 | 896/896 | 91 | 97 | ~parity |
| q5_0 (fused GU) | 37888/896 | 155-166 | 211 | moderate |
| **q6_K (ffn_down)** | **896/4864** | **72** | **217** | **3.0×** |
| q8_0 (output) | 151936/896 | 220-251 | 252 | ~parity |

**Root cause (q6_K, the real gap)**: minfer's `kernel_q6_k_f32_matmul` used a
stride-64 super-block loop — for id=4864 (nb=19 super-blocks) only 19 of 64 TG
threads did work (**~30 % utilization**) with scalar (non-vectorized) inner
loops. llama's `kernel_mul_mv_q6_K_f32_impl` uses a stride-2 thread layout
(16 groups × float4 sums) → ~100 % utilization.

**Fix (SHIPPED 2026-08-13)**: ported llama's q6_K kernel layout into minfer
(stride-2 + float4, TG(32, nsg=2) dispatch). Result:
- q6_K isolation: 72 → **209 GB/s** (llama 217)
- decode steady gpu: **4.27 → 3.72 ms/token (~13 %)**
- **full decode A/B (Generated: pure-decode, -n 256 ×5): old cf2705c median
  ~203 t/s → new 525efe1 median ~220 t/s (~+8 %)**
- byte-identical output (git-stash A/B), all tests green.

q5_0 and q8_0 were already at parity — no work there. q4_K is not present in
the 0.5B K_M model weights (Q5_0/Q8_0/Q6_K/Q4_K mixed), so it was not measured
here — but **7B K_M DOES use q4_K as its dominant decode type** (attn_q/k/output
+ ffn_gate/up all q4_K, see §4.4.2), and minfer's `kernel_q4_k_f32_matmul`
(still the old scalar + 256B deinterleave structure) likely carries the same
gap q6_K had. **→ the next decode lever is a q4_K layout port (to-do #7).**

**q4_K PORT DONE 2026-08-17 (to-do #7)** — isolation probe at 7B decode dims
(`matmul_bandwidth_profile`, batched cb, warm):

| matmul | old GB/s | new GB/s |
|---|---|---|
| attn_q (3584/3584) | 70 | **265** |
| attn_k (3584/512) | 18 | **146** |
| ffn_gate/up (18944/3584) | 74 | **243** |

`kernel_q4_k_f32_matmul` rewritten as a faithful transcription of llama's
`kernel_mul_mv_q4_K_f32_impl` (metal.metal:8300): stride-4 super-block loop
(ix=tiisg/8, iq=it/4, ir=it%4), float4 acc over the kmask nibble unpack, `sc16`
scale unpack reproduced without the `thread uint8_t*` view (byte-of-word
extraction). Dispatch TG(32, nsg=2), grid od/4 — same as the q6_K port.
**The `sc16`/kmask unpack was verified against llama's `get_scale_min_k4`
(ggml-quants.c `dequantize_row_q4_K`): it reproduces the exact per-sub-block
scale/min values** (iq=0 → subs {0,1,4,5}, iq=1 → {2,3,6,7}) — the "riskier
nibble interleave" concern was checked, not hand-waved.

**Measured 7B Q4_K_M** (git-stash A/B, byte-identical; 0.5B has no q4_K weights
so trivially identical; 1.5B q4_K decode + multi path coherent):
- 7B steady decode GPU (MINFER_TIMING gpu submit-wait, pp5): **~49.7 → ~19.3
  ms/token (~2.6×)**; wall Generated: **17-19 → 45-49 t/s** (llama 50.5).
- The old 51 ms/token baseline decomposes as q4_K ~35.7 ms (2.57 GB @ 72 GB/s)
  + q6_K ~10 ms + attention/small — exactly the measured numbers; the q4_K
  kernel WAS the 7B decode bottleneck, now everything is at the ~200 GB/s floor.
- All 34 bin tests green.

#### 4.2.2 attention gap confirmed — flash attention port

**Former "dead-end" verdict revoked** (2026-08-12): the 2026-08-06 downgrade to
"dead-end" (~0.3 ms gain, multi-day risk) was made under the accept-floor premise.
Since the goal is to reach llama's level (attention ~0.9 ms of the decode gap),
this is a **required path**, not an option.

**Gap confirmed by isolation (2026-08-13, NOT a trace artifact):**

| metric | minfer | llama | ratio |
|---|---|---|---|
| attention per layer, nkv=430 (isolation) | **42.8 µs** (partial+combine) | **~4-6 µs** (flash vec) | **~7-10×** |

llama flash isolation (`test-backend-ops perf`, f16 KV, nb=1 vec kernel):
kv=4096 → 23.7 µs, kv=8192 → 55.3 µs (≈linear scaling) → extrapolated to
nkv=430 ≈ 2.5-6 µs/layer, matching the trace's 5.8 µs. Attention is **~0.9 ms
of the 3.72 ms decode step (~24 %)** — the largest remaining decode lever after
the q6_K fix.

**Structural root cause** (correcting the earlier "simdgroup_matrix design"
claim): llama's flash does NOT use the hardware matrix engine here — it uses
**`dot(float4,float4)` + `simd_shuffle_down` reductions** for QK^T and PV, which
need **NO threadgroup barrier within a KV tile**. minfer's partial uses scalar
float4-sum accumulation and **2 threadgroup_barrier per 32-row tile**
(28/layer at nkv=430). That barrier + reduction structure is the gap, not the
dot instruction.

> Full line-by-line kernel comparison (design tables, register-vs-shmem
> accumulation, tile geometry, KV-layout indexing, port change surface): **§5.5**.
> Glossary of the variables/functions used throughout this document: **§5.6**.

**Note — `dot()` builtin experiment failed and was reverted (2026-08-13)**:
replacing minfer's hand-written `qv.x+qv.y+qv.z+qv.w` with Metal's `dot()`
builtin errors: `dot` is a reserved Metal function name and the classic
attention kernel (`kernel_gqa_attn_f32/f16`) already uses a **local variable
named `dot`** in the same translation unit. Fully reverted (git diff clean,
tests green); the hand-sum is what the compiler emits for `dot()` anyway, so
the gain would be noise. The real lever is the barrier/reduction structure.

**Note — llama.cpp build environment (out of scope but recorded)**: the llama
repo has a pre-existing ObjC/SDK build issue — `MTL4CommandQueue` needs
macOS 26, but recompiling ggml-metal ObjC with deployment target 26.0 trips
CoreFoundation `-Welaborated-enum-base`. The existing `test-backend-ops` binary
works and has the perf cases used above; **adding NEW llama test cases requires
a build fix that is out of this task's scope**.

**Port decision (options, in order of preference):**

| Option | Description | Gain | Risk |
|---|---|---|---|
| **C. Hybrid (recommended)** | Port llama's float4 QK^T/PV + simd_shuffle_down reduction structure into a single minfer decode-attention kernel, WITHOUT the full function-constant / nwg/nsg specialization system (fixed config for Qwen2 dims) | up to ~7× (42.8 → ~6 µs) | Moderate: new kernel + isolation tests |
| **A. Full port** | Faithful `kernel_flash_attn_ext_vec` (~600 lines, function constants, per-shape pipelines) replacing partial+combine | same as C | High: requires minfer to support function-constant pipeline compilation (runtime selection differs) |
| **B. Optimize partial only** | Fewer barriers / float4 QK^T in the existing split kernel | bounded 1.5-2× | Low |

**Pre-port check required**: minfer's KV cache layout vs llama flash's expected
layout. **CHECKED 2026-08-14 — PASS, no rework needed.** llama's KV cache is
`ggml_new_tensor_3d(type, n_embd_k_gqa, kv_size, n_stream)` with
`n_embd_k_gqa = nk*hd` packed into dim0 — physical layout `[nkv][nk*hd]`,
token stride `nk*hd*elem`. `get_k` (llama-kv-cache.cpp:1368) + `permute(0,2,1,3)`
(llama-graph.cpp:2443) hand the flash kernel `nb11 = nk*hd*elem` (token stride),
`nb12 = hd*elem` (KV-head stride), `ns10 = nb11/nb10 = nk*hd` — exactly
minfer's `k + ki*stride_kv + hk*hd` with `stride_kv = nk*hd`. The flash kernel
can consume minfer's existing KV buffer unchanged; the host passes the stride
args mirroring minfer (`nb10 = elem, nb11 = nk*hd*elem, nb12 = hd*elem,
ns10 = nk*hd`). Full chain in §5.5 D-1 + flash-investigation log Step 6.

**Prefill flash RE-SCOPED 2026-08-14 (was "explicitly deferred")**: the §4.3
re-investigation measured prefill attention at **46 ms of the 135 ms pp325
(34 %)** vs llama's ~3 ms (`kernel_flash_attn_ext_blk`, a SINGLE fused
`simdgroup_matrix` kernel — `has_simdgroup_mm`, NOT the M5 tensor API), so the
earlier "recover only ~25 ms" estimate was wrong (it was based on the stale
30 ms post-parallel-fix attention). This is now the **#1 prefill lever**, ahead
of the GEMM (§4.3.1). Feasibility is high: minfer already ports the vec flash
variant, the KV-layout check passed, and the blk kernel uses the same
`simdgroup_float8x8` primitives as the prefill GEMMs; the per-shape
function-constant system is the main port surface (fixed Qwen2 dims = NSG=1,
DK=DV=64, ncpsg/nqptg constant like the decode port's fixed-shape approach).

**Port SHIPPED 2026-08-14 (option C, decode only)**:
`kernel_flash_attn_ext_f32` / `_f16` in metal.metal — a faithful NSG=1
fixed-shape (DK=DV=64, NE=2, C=32, NL=16) transcription of llama's
`kernel_flash_attn_ext_vec` for the Qwen2 decode dims, chosen to match the
f16_dk64_dv64 instance (function-constant system not needed — the shape is
fixed and guarded on the host). It writes **the same {M, S, O[hd]} partials as
`kernel_gqa_attn_partial_f32`**, so the shared combine kernel merges them
unchanged (the flash kernel's strided C=32-block chunking and the split's
contiguous chunking are interchangeable from the combine's perspective).

- **Dispatch** (`metal.rs::gqa_attn_flash`): grid `(nt, nh, n_chunks)`, 32
  threads, 1024 B shmem (sq4 | ss | so4); f16-cache variant reads the half KV
  directly. Selected in layer_gpu for `nt==1` when `hd==64` (fixed DK/DV);
  `MINFER_NO_FLASH=1` reverts to the split path for A/B.
- **GPU-safety deviations from llama** (documented in the kernel comment): the
  partial-chunk mask is computed **inline per lane** (llama's `sm[]` write→read
  is a cross-lane threadgroup access without a barrier — a race the codebase
  convention forbids); all control flow is `break`-only (no `continue`, no
  early returns) so all 32 lanes reach both `threadgroup_barrier`s; reads clamp
  to `nkv-1` with `-MINF_MAXHALF` masking instead of llama's pad buffer.
- **Verification**: `tests/flash_attn_isolation.rs` — `flash_attn_ext_isolation`
  (cos vs CPU >0.999 for nkv 1..4097 incl. partial tiles, empty chunks,
  n_chunks 1..32, nt 1..2, f32+f16; run-to-run deterministic) +
  `flash_attn_matches_split` (cos=1.0 vs the split path through the shared
  combine). End-to-end A/B (`MINFER_NO_FLASH=1`): **byte-identical** output on
  0.5B Q4_K_M (f32 cache) + Q4_0 (f16 cache) + 7B Q4_K_M.
- **Measured**: interleaved MINFER_TIMING steady-state decode GPU — flash
  **3.55/4.35/4.33** vs split **4.59/4.99/4.51** ms/token at KV≈64 (~0.3-1.0 ms
  faster); KV≈250: 3.65-4.18 vs 3.96-4.40 (still faster, no long-KV
  regression despite the per-head KV re-read for GQA). Wall clock ~10 %
  (0.5B 184-196 vs 168-182 t/s f32; 257-262 vs 230-237 f16). 7B parity
  (weight-read-bound, ~0.2 ms of a 55 ms step). Full detail + llama re-bench in
  the flash-investigation log Step 7.

**hd=128 decode flash PORT 2026-08-17 (7B, to-do #26)**: 7B decode was the last
path still on the split (partial+combine) attention — llama uses a separate
`f16_dk128_dv128` vec instance (`128,128,1`, NE=1, C=32, DK4=DV4=32). Ported as
`kernel_flash_attn_ext_hd128_f32/_f16` (metal.metal, immediately after the
hd=64 kernels): **NE=1** (one float4/lane, no ii loop, DK4=DV4=32, so4 size
NL·DV4=512 B → shmem 1152 B), 32 cc-iterations each with a full
`simd_sum(mqk)` (llama's vec NE==1 branch) instead of the NE=2 shuffle-down
tree, mask computed inline per-lane with `(ic+tx<nkv)?0:-MINF_MAXHALF` (llama's
cross-lane `sm[]` write is the codebase's forbidden race), partial write covers
all 128 dims per lane (`tx*4`). Host (`metal.rs::gqa_attn_flash`): pipeline
selected by `match (kv_cache_is_f16(), hd)` — 4 pipelines
(f32/f16 × hd64/hd128), `flash_attn_enabled` now `hd == 64 || hd == 128`.
Isolation: `tests/flash_attn_isolation.rs` parameterized `Ctx::with_hd`
(nh,nk,hd) — both test fns run hd=64 (14,2,64) AND hd=128 (28,4,128). Verified:
43 tests green; 7B e2e byte-identical for **flash/split × f32/f16** all four
combinations (pp41 + pp205); 0.5B unchanged. Measured pp205 steady decode GPU:
**flash ~51.1 vs split ~51.6 ms/token** (3×interleaved) — the NE=1/32-iter
simd_sum risk point did NOT materialize; parity with slight flash edge (the
split path reads K then V in two kernels + combine; flash reads each once).

#### 4.2.3 small elementwise — CLOSED by the trace (was the "other half")

**Refuted 2026-08-13**: §4.1.1 measured minfer's small elementwise kernels at
1.2-2.0 µs vs llama's 1.3-1.9 µs — **equal or faster, not 4× slower**. The
2026-08-06 "small-op ~4×" claim (§2.4) came from subtractive decomposition
noise, not real per-kernel data. The 256-thread rms_norm, float4 elementwise
(P6/P7), and BSR fusion already brought small ops to parity. No further work
here. The small-op tail in §1.2's 0.5 ms is dispatch latency on tiny kernels
(2-3 µs each), now confirmed present on both sides equally.

### 4.3 prefill GEMM execution efficiency (~5.4 vs ~7 TFLOPs/s)

**Verified**: `prefill_gemm_throughput_profile` (batched-cb, single-dispatch
verified) shows the same kernel varies **3.5→5.4 TFLOPs/s purely by grid shape**
(nt=416→3.5, 448→5.1, 480→5.3, 512→5.2) — grid-row scheduling variance, not a
bc_out bug. The FFN matmuls (od=18944) dominate prefill (~2.8 ms/layer × 24 ≈ the
real pp430 141-157 ms).

**Counter-evidence**: llama uses the same grid (N_R0/N_SG identical) and hits ~7
TFLOPs/s — if true, shape alone cannot explain the gap; the difference is
per-kernel execution (MPS serialization). **So the grid-shape probe has low
expectation**, but it is zero-cost — do it first to rule it out (10-minute scale).

#### 4.3.1 INVESTIGATION RESULT (2026-08-14) — GEMM kernels are NOT the prefill lever

The grid probe + two per-kernel experiments rule out the GEMM kernels:

1. **Grid shape**: not the lever (see above) — llama dispatches the identical
   grid (64×32 tile, (32,4) threads, N_MM defines identical, mm-vs-mv at
   ne11_mm_min=8). Shape alone cannot explain a 1.3× execution gap.
2. **Threadgroup barrier** (minfer's only structural deviation: 6
   `mem_threadgroup` barriers per 2-block ik-loop vs llama's 2 `mem_none`):
   `mem_none` + scalar sb-store measured only **~2-3 %** faster (ffn_up Q5_0
   2757→2705 µs, ffn_up Q4_0 2454→2381 µs) AND **RACES in minfer**
   (nondeterministic garbage output, verified on Q4_K_M). minfer's
   `mem_threadgroup` is a genuine correctness requirement — do not revert.
3. **Vectorized `half2x4` sb store** (llama's float2x4): **RACY in minfer** even
   with `mem_threadgroup` (1/5 runs corrupted). Scalar store stays.

**Real pp325 decomposition** (0.5B Q4_K_M, `MINFER_SKIP_ATTN=1` subtractive,
MINFER_TIMING gpu submit-wait): total **135 ms = attention 46 ms (34 %) +
everything-else 89 ms**. llama pp320 = **47.7 ms total** with attention only
~3 ms (`-fa on` 6803 vs `-fa off` 6373 t/s). **So minfer's attention is ~46 ms
vs llama's ~3 ms (~15×) — the dominant single component.**

**Root cause — llama's prefill attention is a single fused tensor kernel**:
for ne01 ≥ 20 (prefill) llama dispatches `kernel_flash_attn_ext_blk`
(ggml-metal-ops.cpp `ggml_metal_op_flash_attn_ext_use_vec`: `ne01 < 20` → vec,
else blk) which does QK^T + PV via **legacy `simdgroup_matrix` (`float8x8`,
`has_simdgroup_mm` = true on M4 Pro — NOT the M5-gated "tensor API"**, which
llama disables pre-M5 for MUL_MAT only). One dispatch/layer replaces minfer's 3
barrier-free passes (scores + softmax + output).

**Re-scope (to-do #4 revised)**: the prefill lever is a **prefill flash port**
(`kernel_flash_attn_ext_blk`, function-constant nqptg/ncpsg), not GEMM
execution efficiency. Feasibility is high — minfer already ports the decode
vec-variant (to-do #3) and uses the same simdgroup primitives in the prefill
GEMMs; the remaining unknown is the blk KV-tile/pad layout (single-pass needs
the partial-row combine like the decode flash). Expected prefill: 135 → ~90 ms
(~1.5×), still ~2× llama's 48 ms because the **non-attention 89 ms vs llama's
~44 ms** is a secondary structural gap (GEMMs + small kernels, both
under-occupied, no HW limiter per §4.1).

**Follow-up**: after §4.1's trace locates the GEMM gap (if inside the matmul),
pursue a higher-efficiency GEMM structure. 2D `simdgroup_matrix` (mpp tensor)
**already excluded** (llama disables it on M4 Pro, PARAMETER_AUDIT A); bf16 staging
**already excluded** (llama reads f32 activations).

#### 4.3.2 PREFILL FLASH PORT DONE (2026-08-14) — `kernel_flash_attn_blk_f32/_f16`

Faithful fixed-shape transcription of llama's `kernel_flash_attn_ext_blk`
(legacy `simdgroup_matrix`). NSG=4 (llama picks `nsg = ne00 >= 512 ? 8 : 4`; hd=64
→ 4), Q=8 (`OP_FLASH_ATTN_EXT_NQPSG`), C=64 (`OP_FLASH_ATTN_EXT_NCPSG`),
DK=DV=64, 128 threads (32 lanes × 4 simdgroups), shmem 7168 B (sq[512 half] |
so[512 f32] | ss[1024 f32]). Grid `(ceil(nt/8), nh)`; each threadgroup computes
Q=8 query tokens × ALL KV for head h (GQA hk = h/gqa baked into the K/V base).
**Two deliberate GPU-safety deviations from llama**: (1) the causal mask is
computed inline (no mask/block-pad pre-pass kernels); (2) the partial last KV
block (nkv % 64 != 0) is read from a `[2][64][nkt]` tail-pad buffer
(`kernel_kv_tail_pad` copies the last 64 virtual rows from the real cache; padded
rows are zero + masked to -MINF, so they never contribute). Q is **always f32**
(llama reads Q as `float4` regardless of KV type — `llama-graph.cpp:2457-2463`
casts only K/V to f16); the f16 variant switches only the K/V/pad operands to
`half` + `simdgroup_half8x8` K/V tiles.

**Verification**:
- `tests/flash_attn_blk_isolation.rs` (macOS): cos vs a scalar CPU reference
  >0.999 across 16 nt/nkv configs (nkv 1..300 incl. partial blocks, nkv < C,
  multi-threadgroup nt up to 200, GQA=7), f32+f16, run-to-run deterministic.
- A/B vs the long-verified classic `gqa_attn_f32` kernel: **byte-identical**
  (maxabs 0.0) at every layer AND the final logits, for BOTH cache types
  (f32: MINFER_NO_PREFILL_FLASH=1 → 3-pass, MINFER_NO_MATMUL_ATTN=1 → classic;
  f16: classic A/B). 0.5B Q4_K_M, pp140 + 4 decode tokens.
- 34 bin tests + 9 isolation tests pass (includes `metal_pipelines_compile`).

**Performance** (0.5B Q4_K_M, interleaved MINFER_TIMING, pp294): prefill GPU
submit-wait **~110 → ~93 ms (~16 %)**, matching the earlier pp257 wall-clock
trend (2390-2514 vs 1906-2272 t/s). The 135→~90 ms target from §4.3.1 is met.
7B (hd=128) also now uses the flash path — see §4.3.3 (ported 2026-08-15).

**Bonus — fixes a pre-existing f16-cache prefill bug**: the 3-pass
`kernel_attn_scores`/`kernel_attn_output` read the KV buffers as `device const
float *` but with `MINFER_CACHE_TYPE=f16` the cache holds half data → garbage
generation ("!!!!!!"). The f16 blk kernel reads half K/V correctly; the flash
path is now the f16 prefill default for hd==64 and matches the classic f16
output byte-identically.

**Dispatch**: `MINFER_NO_PREFILL_FLASH=1` reverts to the 3-pass for A/B; the
flash path requires hd==64 or hd==128 (fixed DK=DV), else 3-pass. Host side:
`attn_flash_prefill` (metal.rs) grows the pad buffer, runs `kernel_kv_tail_pad`
when nkv % 64 != 0, then the blk kernel (pipeline + shmem selected by hd).

#### 4.3.3 hd=128 (7B) prefill flash feasibility analysis (2026-08-15)

The §4.3.2 port is fixed-shape DK=DV=64 (0.5B/1.5B). The user-facing 7B
(hd=128, nh=28, nk=4, nkt=512) still runs the 3-pass prefill attention. This
section answers whether a hd=128 blk variant is feasible and worth doing.

**llama.cpp reference (verified in source)**:
- 7B prefill uses `kernel_flash_attn_ext_blk` (the `use_vec` gate is
  `ne01 < 20`, i.e. decode-only; ggml-metal-ops.cpp:2526-2533). NSG = 4
  (`ne00 >= 512 ? 8 : 4`, line 2835 — 7B hd=128 → 4, same as 0.5B).
- NQPSG=8 / NCPSG=64 (metal-impl.h:109-110), grid `(ceil(nt/8), nh)`,
  threads `(32, NSG)`.
- shmem formula `FATTN_SMEM` (ops.cpp:2817): hd=128 →
  `8*(128 + 2*128 + 2*2*64) halfs = 10240 B`. hd=64 → 7168 B (matches minfer).

**shmem layout scales cleanly** — the key insight: the `ss` softmax scratch is
`Q*SH` floats (SH=2*C=128, C fixed), NOT `C*hd`, so it does NOT grow with the
head size. minfer hd=128 layout: `sq[Q·DK halfs]=2048 B | so[Q·PV floats]=4096 B
| ss[Q·SH floats]=4096 B` = **10240 B total**, well under the M4 Pro
`max_threadgroup_memory_length = 32768 B` (runtime-queried via a one-off
`probe_device.rs`). ✓

**Constant deltas for a hd=128 variant** (vs the hd=64 kernels at
metal.metal:3534/3717): `DK=DV=128`, `DK4=DV4=32`, `DK8=16`, `PV=128` (`PAD2(128,64)`),
`PV4=32`, `PV8=16`, `NO=PV8/NSG=4` (was 2 → two extra `lo[]` accumulators +
the O-store loop), shmem offsets `so@512`, `ss@1536` (float units), total 2560
floats. Everything else (QK^T loop over DK8/2=8 iters, O+=P·V loop, online
softmax, causal+pad masks, `kernel_kv_tail_pad`, f16 variant, host dispatch)
is unchanged. KV layout + stride already match llama (`[nkv][nk*hd]`, NS10=nkt=512).

**Measured 7B prefill decomposition (pp332, Q4_K_M, MINFER_TIMING GPU
submit-wait, min of 3) vs llama**:

| path | minfer GPU | llama | minfer/llama |
|---|---|---|---|
| 3-pass attention (baseline) | **~1137 ms** | **734 ms** (452 t/s) | **1.55×** |
| no-attention (`MINFER_SKIP_ATTN=1`) | **~1019 ms** | ~730 ms (attn ~4-6 ms) | ~1.40× |
| attention cost (by subtraction) | **~118 ms (10.4 %)** | — | — |

**Verdict — feasible, but LOW leverage at 7B**:
- Attention is only **~10 %** of 7B prefill (vs **~34 %** at 0.5B pp325): the 7B
  GEMMs/FFN dominate and are weight-bound, so a perfect flash port
  (attention ~118 → ~6 ms) yields only **~9.5 % total prefill gain**
  (1137 → ~1029 ms), closing the llama gap from 1.55× to ~1.40×. The remaining
  1.40× is the non-attention structural gap (§4.3.1) — GEMMs + small kernels.
- **BUT the port also fixes a correctness bug**: 7B with `MINFER_CACHE_TYPE=f16`
  currently generates **garbage ("!!!!!!")** — the same f16-cache 3-pass bug as
  0.5B (§4.3.2 bonus), since 7B falls back to 3-pass (hd=128). A hd=128 flash
  f16 variant would make f16 cache correct on the user-facing model. Verified:
  `MINFER_CACHE_TYPE=f16 ./minfer 7B "The capital of France is" -n 12 --greedy`
  → `!!!!!!!!!!!!`.

**PORT SHIPPED 2026-08-15** (decision: implement, on the f16-correctness +
~9.5 % prefill grounds):
- `kernel_flash_attn_blk_hd128_f32/_f16` (metal.metal, after the f16 hd=64
  variant): same structure as the hd=64 kernels with the constant deltas above
  (DK=DV=128, DK4=DV4=32, DK8=16, PV=128/PV4=32/PV8=16, **NO=4**). The O+=P·V
  loop now uses `mv[4]`/`lo[4]` (dim block offsets `ii*8*NSG = 0,32,64,96`) —
  mathematically the llama DV>64 branch (`vs[2]`/`mv[4]`/`NC=(C/8)/2`) but
  re-blocked to minfer's single-`vs` C/8 loop. shmem `so@512`/`ss@1536` (float
  units), total **10240 B** (runtime < 32768 B).
- Host: `pl_flash_attn_blk_hd128{,_f16}` pipelines; `attn_flash_prefill`
  selects pipeline + shmem (10240 vs 7168) by `hd`; `prefill_flash_enabled`
  now allows `hd == 64 || hd == 128`. Pad buffer / `kernel_kv_tail_pad` /
  grid `(ceil(nt/8), nh)` unchanged (C=64 fixed; nkt = nk*hd = 512 at 7B).
- Verified: `tests/flash_attn_blk_isolation.rs` extended to NH=28/NK=4/HD=128
  (same 16 nt/nkv configs — cos vs CPU >0.999, deterministic, f32+f16, 4.05 s);
  end-to-end 7B `blk hd128 ≡ 3-pass ≡ f16` byte-identical output (pp54/62/310),
  **7B f16-cache garbage FIXED** (was "!!!!!!!!!!!!", now identical to f32),
  0.5B unchanged. All 34 bin + 10 isolation tests pass.
- Performance (7B Q4_K_M, pp310, interleaved MINFER_TIMING gpu submit-wait):
  blk hd128 f32 **943-952 ms** (~949) vs 3-pass f32 **1041-1043 ms** → **~9.3 %
  faster**, matching the ~9.5 % §4.3.3 prediction; f16 ~952 ms (parity with
  f32). Decode steady ~50 ms/token unchanged (hd=128 decode still split
  attention — `flash_attn_enabled` remains hd==64-only).

#### 4.3.4 7B prefill GEMM per-kernel A/B (2026-08-18) — GEMMs ruled out at 7B, directly

The §4.3.1 "GEMMs ruled out" verdict was measured at 0.5B dims only. This closes
the gap for the user-facing 7B at the SAME 7B prefill dims (nt=430), same M4 Pro,
same model type (q4_K/q6_K from `minfer info` on `qwen2.5-7b-instruct-q4_k_m`).

**Method**: minfer `prefill_gemm_throughput_profile` (batched cb, warm, median,
TFLOPS added) vs llama `test-backend-ops perf -b MTL0 -o MUL_MAT` at identical
od/id/nt. The llama binary was rebuilt with these exact cases added to
`make_test_cases_perf()` (tests/test-backend-ops.cpp) — the stock binary had no
q4_K/q6_K large-dim MUL_MAT perf cases; the rebuild required two pre-existing
SDK fixes (`-Wno-elaborated-enum-base` for the vDSP `-Welaborated-enum-base`
failure at `-mmacosx-version-min=26.0`, and the `posix_spawn_file_actions_addchdir`
→ `_np` fix in `vendor/sheredom/subprocess.h`).

**Result — minfer 7B prefill GEMMs are at 87-94 % of llama (~parity):**

| 7B matmul | type | od/id | minfer TFLOPS | llama TFLOPS | minfer/llama |
|---|---|---|---|---|---|
| attn_q / attn_output | q4_K | 3584/3584 | 5.55 | 5.94 | **93 %** |
| attn_k | q4_K | 512/3584 | 4.69 | 5.02 | **93 %** |
| ffn_gate/up | q4_K | 18944/3584 | 5.75 | 6.14 | **94 %** |
| attn_v | q6_K | 512/3584 | 4.42 | 5.08 | **87 %** |
| ffn_down | q6_K | 3584/18944 | 5.10 | 5.79 | **88 %** |
| output (lm_head) | q6_K | 152064/3584 | 5.39 | 6.12 | **88 %** |

> Both sides are FLOPs-bound here (weight read once per batch; GB/s is
> meaningless for GEMMs) — TFLOPS is the right metric. The stock binary's
> nearest pre-existing case (m=4096, k=14336, n=512) gives the same
> ~5.8-6.1 TFLOPS range, confirming the added cases are consistent.

**Conclusion**: the 7B prefill ~1.40× residual (§4.3.3) is **NOT GEMM compute** —
the GEMM kernels are ~parity (1.07-1.15×). The GEMM-vs-non-GEMM decomposition
(total 7B prefill GEMM FLOPs ≈ 6.1 TFLOP; at minfer ~5.3 vs llama ~5.9 TFLOPS
that's only ~0.12 s of the measured ~0.29 s no-attn gap) leaves the majority of
the gap in **per-dispatch serialization + small kernels between GEMMs**, matching
the §4.1 limiter profile (prefill has NO hardware limiter — under-occupied,
scheduling-bound). The smallest matmuls (attn_k/v, od=512) show the largest
relative gap (87-93 %), consistent with dispatch-latency-bound small GEMMs. No
new GEMM-kernel work is warranted; the §4.3.1 "secondary structural gap"
conclusion now has direct 7B evidence.

> ⚠️ **2026-08-18 SUPERSEDED by §4.3.6**: this isolation A/B compared
> llama-under-measured isolation (test-backend-ops per-iteration submit/sync
> overhead) against minfer-clean isolation (batched cb). With llama's REAL
> prefill GEMMs at ~6.9 TFLOPS (llama-cli/bench pp466 = 467 t/s) vs minfer's
> ~5.2, the true per-GEMM gap is ~1.33× — the GEMM kernels ARE the lever. The
> isolation "parity" was a measurement artifact.

#### 4.3.5 Phase 0: prefill subtractive decomposition (2026-08-18) — #1 fusion ceiling is low

To size the "small-kernel fusion / dispatch-reduction" lever (proposed §4.3.4
follow-up), the prefill skip gates were extended to nt>1 (`DecodeSkips::active`,
`MINFER_SKIP_MATMULS` / `MINFER_SKIP_SMALL` now honored during prefill — gates
kept in their exact original positions). Subtractive decomposition via
`MINFER_TIMING gpu(submit-wait)`, min of 3 interleaved, pp466 (fixed prompt),
both models Q4_K_M:

| category (derived) | 0.5B pp466 | 7B pp466 |
|---|---|---|
| **total prefill GPU** | **125.7 ms** | **1613.5 ms** |
| attention (flash blk) | 7.2 ms (5.7 %) | 17.5 ms (1.1 %) |
| **layer GEMMs** | **95.2 ms (75.7 %)** | **1424 ms (88.3 %)** |
| **small kernels + their dispatch** | **12.9 ms (10.3 %)** | **65.1 ms (4.0 %)** |
| output GEMM + empty-cb infra | 10.4 ms (8.3 %) | 106.7 ms (6.6 %) |

> Derivation: `attention = base − skip_attn`; `GEMMs = skip_attn − skip_attn+matmul`;
> `small+dispatch = skip_attn+matmul − all-skip`; remainder = output lm_head GEMM
> (not gateable) + the empty-submission floor. The 7B "all-skip" 106.7 ms is
> dominated by the output GEMM (~90 ms at isolation); true empty-cb infra is ~16 ms.

**Reading**:
- **GEMMs dominate prefill at BOTH sizes (76 % / 88 %)**; the flash port cut
  attention to single-digit percentages. The 1.40× gap vs llama cannot come from
  the 4-10 % small-kernel tail.
- **#1 small-kernel fusion ceiling: ≤10 % (0.5B) / ≤4 % (7B)** — and that is the
  upper bound (fusing ALL 12 small kernels/layer); realistic fusion (bias+RoPE,
  residual+RMSNorm, store_kv K+V) recovers at most ~half of it. The 0.5B number
  is just over the 5 % decision gate, but the user-facing 7B is below it.
- **Decision: #1 is NOT the prefill lever — pivot to Phase X.** The gap is the
  real-chain GEMM execution: minfer's real prefill ≈ its own isolation-GEMM sum
  (no loss, but no gain), while llama's real prefill runs *faster* than its
  isolation sum (~8 %) — llama interleaves/pipelines back-to-back GEMMs where
  minfer serializes them. Candidates: prefill concurrency (llama init shows
  `use concurrency = true`; the decode multi-cb regression is a decode-only
  result — prefill is compute/occupancy-bound), a 7B prefill xctrace per-kernel
  comparison (§4.1 method), and per-GEMM occupancy tuning.

> ⚠️ **2026-08-18 CORRECTION (thermal)**: the §4.3.5 table above was taken with a
> *sequential per-config* methodology (3 runs of each config in a row) under GPU
> thermal drift — the M4 Pro throttles to ~1.3-1.4 s floor under sustained load
> (§3.3 measurement trap). The 0.5B numbers in particular are unreliable (the
> "23 ms skip-attn+matmul" vs "120 ms" later). The **authoritative numbers are the
> interleaved re-measure in §4.3.6** (configs cycled per pass, min of 4).

#### 4.3.6 Phase X part 1 (2026-08-18): real-chain GEMM efficiency — the mechanism

**Thermal-corrected interleaved decomposition** (7B Q4_K_M pp466, `MINFER_TIMING`
gpu submit-wait, configs cycled per pass, min of 4, 60 s cooldown first):

| category | 7B pp466 |
|---|---|
| total prefill GPU | **1373.6 ms** |
| attention (flash blk) | 21.7 ms (1.6 %) |
| **layer GEMMs** | **1175.0 ms (85.5 %)** |
| small kernels | 72.9 ms (5.3 %) |
| output GEMM + empty infra | 104.0 ms (7.6 %) |

**The "GEMM↔small transition loss" hypothesis is DISPROVEN**: layer GEMM time is
identical with small kernels interleaved (**1175 ms**) vs back-to-back
(`MINFER_SKIP_SMALL=1`, **1179 ms**). The earlier apparent 256 ms transition
loss was pure thermal drift. **#1 fusion ceiling is finally confirmed LOW**
(small kernels = 5.3 %; even fusing all of them recovers ≤5 %, and the transitions
cost ~0).

**llama concurrency DISPROVEN**: llama's `MTLDispatchTypeConcurrent` (+
dependency-aware `memoryBarrier`, `ggml-metal-device.m:469`/ops.cpp `mem_ranges`)
has **zero effect on llama prefill** — `GGML_METAL_CONCURRENCY_DISABLE=1` vs
default = 465.82 vs 467.17 t/s (0.3 %, noise) at pp466.

**The real gap is in the GEMM kernels themselves — and §4.3.4's isolation A/B was
methodology-flawed**:
- llama real 7B prefill pp466 = **466.9 t/s** (llama-cli AND llama-bench agree,
  ~0.998 s). Total FLOPs 6.59 TFLOP → **llama real ≈ 6.9 TFLOPS effective** —
  *faster* than its own test-backend-ops isolation (5.9-6.1) because that harness
  has a per-iteration submit/sync overhead that systematically under-measures.
- minfer real 7B prefill layer GEMMs = 1175 ms / 6.08 TFLOP = **5.17 TFLOPS** —
  matching minfer's own isolation (batched cb, clean).
- **True per-GEMM gap ≈ 1.33× (6.9 vs 5.2)**, back to the original §4.3.1
  "~5.4 vs ~7" finding. §4.3.4's "parity (87-94 %)" was an artifact of comparing
  llama-under-measured isolation against minfer-clean isolation.
- minfer's `kernel_mul_mm` q4_K (metal.metal:4614) vs llama's legacy
  `kernel_mul_mm` (ggml-metal.metal:10095): SAME 64×32 tile / 128 threads /
  simdgroup_matrix structure. The only structural difference: minfer has **6
  `mem_threadgroup` barriers per 32×64 tile (4 extra inside the ik-loop)** vs
  llama's 2 + cheap `simdgroup_barrier(mem_none)`. But §4.3.1 measured barrier
  removal = only 2-3 % AND races — so barriers are NOT the 1.33× either.

**Open question (the actual remaining lever)**: the ~1.33× per-GEMM execution gap
with structurally-identical kernels. Not concurrency, not fusion, not barriers, not
small kernels, not attention. Remaining suspects to investigate (kernel-level,
GPU-safety-sensitive):
1. **Dequant instruction sequence** — minfer `dequant_q4_k_16` vs llama
   `dequantize_row_q4_K` (different from the mv-kernel's dequantizer; the mm kernel
   dequant may be slower).
2. **sb staging** — minfer scalar `half()` stores vs llama vectorized `S1_2x4`
   (`half2x4`) stores (measured RACY in minfer §4.3.1 — root cause unknown, not
   re-litigated).
3. **Register pressure / residency** — different `temp_a`/sa/sb layout may drop
   resident threadgroups (occupancy), which the batched isolation would partially
   hide but the real chain exposes. Needs a `threads_limited`/occupancy probe.
4. **Grid ordering** — minfer `dispatch((nt+31)/32, (od+63)/64)` vs llama
   `(ne11/nr1, ne01/nr0)` (same shape, both y=od-major).

> **Phase X part 2 (2026-08-18) — suspects 1 & 2 measured and CLOSED, barrier
> confirmed required, geometry confirmed identical**: every source-level element
> of minfer's mm kernels now matches llama. Full record of the experiments:
>
> 1. **Dequant instructions: IDENTICAL.** `dequant_q4_k_16` (metal.metal:716) vs
>    `dequantize_q4_K` (ggml-metal.metal:737) are line-for-line the same
>    (`dl*(q[i]&mask)-ml`, identical q/il/sc/d indexing; only the temp type differs).
> 2. **temp_a float4x4→half4x4** (match llama's `S0_4x4`, suspected register
>    pressure): **no speedup** (7B pp466 ~1374 vs baseline ~1373) and 7B run3
>    output nondeterminism → REVERTED. Register-pressure hypothesis not supported
>    (the MSL compiler apparently allocates the same registers either way).
> 3. **ik-loop barrier → `simdgroup_barrier(mem_none)`** (llama's exact form,
>    4× fewer cross-simdgroup syncs/tile): **CORRUPTS output deterministically**
>    (garbage "!!!!!!!", all runs). The docs' "mem_threadgroup is a genuine
>    correctness requirement" is EMPIRICALLY CONFIRMED. The barrier is the one
>    structural difference from llama but is NOT removable in minfer (root cause
>    of the asymmetry vs llama unestablished; likely a compiler/scheduling
>    interaction with minfer's staging, not a memory-visibility issue per the
>    simdgroup-slice analysis).
> 4. **sb staging scalar→vectorized `half2x4`** (llama's `S1_2x4`, the literal
>    suspect #2): **byte-identical on 7B AND 0.5B** (5/5 runs each, this session;
>    the earlier §4.3.1 "RACY 1/5" claim likely mis-attributed a deterministic
>    output difference to a race — the earlier test compared against the wrong
>    reference) but **ZERO speedup** (interleaved A/B: 1353.4 vs 1352.3 ms) →
>    REVERTED. The staging is not the lever.
> 5. **Geometry: IDENTICAL.** llama's legacy mm pipeline (ggml-metal-device.cpp:704)
>    sets `nr0=64, nr1=32, nsg=4, smem=6144/8192` → dispatch `(nt/32, od/64)` with
>    128 threads = minfer's `gemm_dispatch` exactly (metal.rs:353). Grid order also
>    identical (both y=od-major).
>
> **Net conclusion**: the ~1.33× gap has NO source-level explanation. The only
> remaining lever candidates are MSL-compiler-level (shader machine code / register
> scheduling, comparing the compiled `.air` of both metal.metal files) or a subtle
> execution-environment effect. Not addressable from minfer source. Recommend:
> (a) accept the gap and record it, or (b) if pursued, disassemble both compiled
> shaders and diff the machine code (deep, low-value given the parity of every
> structural element).

#### 4.3.7 GEMM PARTIAL-TILE RACE FIXED + missing Metal `memoryBarrier` (2026-08-19) — the prefill nondeterminism root cause

**Symptom**: 1.5B/7B prefill intermittently produced a wrong first generated token
(~10-30 % of runs; 0.5B was stable). Debug dumps (`debug_dump` build, `-n 1`)
localized the corruption to **exactly the last 2 prefill token slots** of the reused
per-layer `bn` buffer (layer0, huge stale values, rms 18.9 vs clean 0.32), and then
— once that was fixed — to **rows 464/465 of the prefill output logits** (od
141152..141183, i.e. sgitg 3's sub-tile of the LAST partial x-tile).

**Two independent races**:

1. **No `memoryBarrier` between dispatches in the single prefill encoder** (Rust
   `src/metal.rs`). Dispatches in one Metal compute command encoder are ordered but
   write-visibility is NOT guaranteed without an explicit barrier. `bn` is written by
   RMSNorm, read by QKV, rewritten by WO and ffn_down — a reader's last tiles raced
   the writer's first tiles → last-2-token garbage. llama.cpp inserts
   `memoryBarrierWithScope` after every op. **Fix**: `barrier()` (MTLBarrierScopeBuffers)
   at the end of `dispatch_1d/2d/3d`.
2. **Missing `threadgroup_barrier` before the GEMM partial-tile `temp_str` stores**
   (Metal kernels). All 8 simdgroup mm kernels
   (`kernel_q4_0/q4_1/q8_0/q5_0/q5_1/q6_k/q4_k/q5_k_mm_f32`) store the partial tile
   via `temp_str` — a 2048-float region that **overlaps the `sa`/`sb` staging
   buffers** — then copy it out. After the K-loop ends there was NO barrier before
   the `simdgroup_store`s, so a fast simdgroup could overwrite `sa`/`sb` while a slow
   simdgroup still read them for its final MAC → intermittently wrong `mc`
   accumulation → corrupted logits for the partial (last) x-tile. The full-tile path
   stores straight to the output (no `sa`/`sb` reuse) → immune, which is why only the
   last partial tile corrupts. llama.cpp's `mul_mm_t` template has this exact barrier
   (`ggml-metal.metal:10273`) — the ports missed it. **Fix**: added
   `threadgroup_barrier(mem_flags::mem_threadgroup)` before the `temp_str` stores in
   all 8 kernels.

**The "clean" majority output was itself wrong**: pre-fix GPU printed `" Quant"` as
the first token 22/24 times (small corruption flipping argmax to the runner-up) and
garbage (`" an"`, `"라면"`, …) the rest. CPU ground truth is `"The"` — the fix makes
GPU match CPU exactly.

**Verification**:
- 1.5B Q4_K_M: 24/24 first token `"The"` (GPU) == CPU; `-n 8` byte-identical
  ("The transformer language model performs inference steps by").
- 7B Q4_K_M (split GGUF): 24/24 first token `"Your"` (GPU) == CPU.
- 0.5B (Q4_0/Q8_0 mm kernels also patched): unchanged, matches CPU.
- Prefill logits argmax: last rows (488-494) CPU == GPU (row 494 = sampled first
  token); mid-prompt argmax flips (11/495) are expected Q8_0-quantized-activations
  (CPU) vs f32 (GPU) numerical divergence, generation-affecting last row identical.
- `MINFER_GEMM=0` (forces f32-multi output GEMM) + barriers was 24/24 — the
  isolation proof that the residual race lived in the mm GEMM partial-tile path.
- Prefill time unchanged (barrier cost ~0).

#### 4.3.8 Post-fix prefill re-baseline + last cheap levers measured (2026-08-19) — gap accepted

Closes the §4.3.6 "open question" with the last untested cheap levers. All
measurements 7B Q4_K_M pp495 (fixed 495-token prompt), interleaved, min of 4,
60 s cooldown first (same thermal-corrected method as §4.3.6).

| experiment | result | verdict |
|---|---|---|
| **Post-fix baseline** (2026-08-19 barrier fix + kernel barriers): `MINFER_TIMING gpu(submit-wait)` | minfer **1461.8 ms (338.6 t/s)**; llama-cli min **457.5 t/s** (≈1.076 s) → **73.6 %** | no regression from the 08-19 fix (§4.3.6 pp466 ≈ 73 %); gap unchanged ~1.33× per-GEMM |
| **MSL compile options** — llama compiles with `-O3` (CMakeLists.txt:86); minfer runtime default → set `setOptimizationLevel: MTLLibraryOptimizationLevelFast` (msg_send) | byte-identical, but min **1478.0 ms vs 1461.8** (~1.1 % SLOWER) | no benefit; reverted |
| **MTLDispatchTypeConcurrent** (llama's encoder dispatch type; `compute_command_encoder_with_dispatch_type`) | byte-identical + deterministic, but min **1443.6 ms vs 1451.4** (~0.5 %) | noise (matches §4.3.6's llama 0.3 %); reverted |
| **Resource-scoped memoryBarrier** (llama's dependency-aware `memoryBarrierWithResources`) | **not pursued** — the encoder reuses essentially every activation/scratch buffer (buf_hidden/bn/ba/bf/bg/q8/kv/attn_scores) across adjacent dispatches, so dependency-aware ≈ barrier-always; per-dispatch resource tracking adds correctness risk for ~0 expected gain | n/a |

**Net conclusion (final)**: the ~1.33× per-GEMM prefill gap is NOT source-addressable.
Structurally-identical kernels, identical geometry/dequant/staging, compile
optimization level, dispatch type, and barrier regime all measured — none closes it.
Only MSL-compiler register-scheduling / execution-environment differences remain
(§4.3.6). **Gap accepted and recorded**: 7B prefill ~73 % of llama (pp495), decode
at parity (~50 t/s). Future work on prefill would require the tensor-API GEMM
(`mpp::tensor_ops`, llama only uses it for MoE `kernel_mul_mm_id`) as a
beat-llama research direction, not a parity fix.

> **2026-08-19 SUPERSEDED (partially) by §4.3.9**: the compiler-level gap is NOT
> fully source-inaccessible — missing `#pragma unroll` on the mm-kernel hot loops
> recovered ~6 % (7B pp495 1438.8 → 1355.6 ms). The tensor-API note above still
> stands (disabled on M4 by llama's own default; M2 Ultra ~5 % slower).

#### 4.3.9 mm-kernel hot-loop unrolling (2026-08-19) — first measurable compiler-level win

llama's legacy `kernel_mul_mm` forces unrolling with `FOR_UNROLL` (86 uses) on the
staging/ik/load/mac loops; minfer's 8 mm kernels had ZERO `#pragma unroll` and
relied on the MSL optimizer's auto-unroller. Added `#pragma unroll` to the 5 hot
loops of all 8 mm kernels (`kernel_q4_0/q4_1/q8_0/q5_0/q5_1/q6_k/q4_k/q5_k_mm_f32`):
sa-staging (16), sb-staging (8), ik-loop (NK/8=4), ma-load (4), mb-load (2), mac (8)
— matching llama's `FOR_UNROLL` set exactly (llama has no sb-loop; its sb is one
vectorized `S1_2x4` store).

**Verified** (thermal-controlled stash A/B, min of 3, pp495):
- 7B: **1438.8 → 1355.6 ms (~5.8 %)**; 1.5B: 350.9 → 342.6 ms (~2.4 %);
  0.5B: 126.3 → 117.6 ms (~6.9 %).
- Byte-identical on 0.5B/1.5B/7B (sampled tokens + `-n 8` output unchanged);
  1.5B ×24 first-token determinism 24/24.

Reduces the per-GEMM gap from ~1.33× to ~1.25×. The remaining per-GEMM gap is
still not explained at source level (§4.3.6 list); with unrolling done, the next
cheap lever would be eliminating minfer's extra ik-loop `threadgroup_barrier`s
(§4.3.6 suspect, measured ~2-3 %), which the docs previously found non-removable
(deterministic corruption) — the root cause of that asymmetry is still open.

> **2026-08-19 follow-up — ik-loop barrier root cause FOUND via .air diff, and
> the extra barriers REMOVED.** The §4.3.6 corruption verdict was measured on the
> PRE-unroll kernel; it was a **rolled-loop compiler artifact**, not a
> memory-visibility need. Evidence: (a) `.air` disassembly (metal toolchain
> 32023.883) shows the ik-loop `threadgroup_barrier(mem_threadgroup)` →
> `simdgroup_barrier(mem_none)` swap changes ONLY the barrier instruction —
> zero IR scheduling difference; the runtime corruption is below the AIR level.
> (b) `threadgroup_barrier(mem_none)` (threadgroup-wide exec sync, no flush) is
> safe but no faster → the requirement is execution-order, not memory. (c) with
> the `#pragma unroll` from §4.3.9 in place, **`simdgroup_barrier(mem_none)`
> (llama's exact form) is now CORRECT** — 1.5B ×24 + 7B ×8 + 0.5B ×3
> byte-identical/deterministic, 33/34 tests green — and ~1 % faster (7B pp495
> min 1370.6 vs 1387.4 ms). The unrolled straight-line ik-loop schedules each
> simdgroup's 4 iterations contiguously, so the within-simdgroup barrier
> suffices. Applied to all 8 mm kernels: the last structural difference vs llama
> is gone.

> **2026-08-19 — structural-equivalence audit COMPLETE (post unroll+barrier)**.
> minfer's mm GEMM is now verified identical to llama's legacy `kernel_mul_mm` at
> EVERY measurable level, yet the ~1.25× real-chain gap persists:
> - **Source**: ik-loop/load/mac/staging line-identical; dequant identical.
> - **IR**: `.air` disassembly (same standalone toolchain) — kernel bodies
>   `kernel_q4_k_mm_f32` (19 KB) vs `kernel_mul_mm_q4_K_f32` (25 KB); same op mix
>   (1 mac site, 2× 8×8 load, 2× store); only barrier-site counts differ (6 vs 4
>   wg, 2 vs 3 sg).
> - **smem**: 8192 B both (bc_out path).
> - **Dispatch**: llama `(ne11/32, ne01/64, 32×4)` (ops.cpp:2222) == minfer
>   `(nt/32, od/64, 32×4)` (metal.rs:368).
> - **Runtime compile**: both `newLibraryWithSource` with default options
>   (llama adds only preprocessorMacros, irrelevant to the legacy q4_K path).
> - **Wall-clock**: minfer CPU-encode overhead is small (~28 ms, wall 1380 vs GPU
>   1352 ms); llama wall 1069 ms (463 t/s pp495) → **true gap ≥1.25× in the GPU
>   GEMM itself**, not host-side.
> The residual gap is below IR/static visibility (backend machine-code
> scheduling or GPU execution-environment), consistent with §4.3.6. Combined
> unroll (§4.3.9) + barrier (§#30) recover ~6.5 % of the ~1.33×.

#### 4.3.10 Phase-0 7B prefill decomposition (2026-08-20) — exhaustive refutation + exact-shape replay

Closes the §4.3.9 "residual gap" question with a full 7B Q4_K_M pp495 (fixed
495-token prompt) decomposition at the GRAPH level. Every factor was tested
with an exact-shape replay harness (real 7B shapes, minfer kernels, one command
buffer, `-O` Swift — note: Swift `-Onone` fills a 447 MB buffer in ~40 s, which
looked like a GPU hang; always compile the harnesses with `-O`).

**Weight-type map CORRECTED** (from llama's `GGML_METAL_PRINT_GRAPH` dump of all
197 MUL_MAT): 14 layers have `wk` = **q6_K** and `down` = **q6_K** (layers
0,1,2,5,9,11,14,17,20,23,24,25,26,27 / 0,1,2,5,12,14,17,20,23,24,25,26,27), and
**output projection = q6_K [152064×3584]** — NOT "all q4_K except ffn_down=q6_K"
as previously assumed. `wq/wo/wv/gate/up` always q4_K. Total MUL_MAT FLOP =
**7.000 TFLOP** (per-node sum from the graph dump).

**llama GPU-busy (host timestamps, clean window)**: CB1 (main, first 64 nodes)
83 ms + CB0 (worker, remaining 392 nodes) 964 ms ≈ **1043 ms @ pp495 = 6.71 TF**.
This is the target. Under comparable system load (load avg ~3-4) llama-bench
pp495 degrades to 1250-1760 ms and CONVERGES with the replay — the earlier
"llama 1043 vs replay 1126" gap was different load windows.

**Results** (all factors tested):

| experiment | result | verdict |
|---|---|---|
| isolated kernel A/B (single GEMM, fresh CB) | q4K 6.18 vs llama 6.24 TF; q6K equal or minfer-faster (down 5.75 vs 5.98; output **5.93 vs 5.45 — minfer faster**) | kernels equal |
| in-batch kernel A/B (28×6 q4K + output, one CB, distinct weights) | minfer 841 vs llama 833 ms (~1 %) | kernels equal |
| grid / threads / smem | llama `(ne11/32, ne01/64, 32×4)` == minfer `(nt/32, od/64, 32×4)`, 8192 B both | identical |
| buffer storage mode | both StorageModeShared (`newBufferWithBytesNoCopy` llama / `new_buffer` minfer) | identical |
| pooled vs separate weight buffers | 1174 vs 1175 ms | no effect |
| weight DATA (real GGUF vs pseudo-random) | real ~1226 vs random ~1209 ms GEMM-only | no effect |
| barrier per dispatch (replay: `memoryBarrier(scope:.buffers)`) | 1143.2 vs 1143.7 ms | free |
| interleave dummy attn/norm dispatches between GEMMs | 845 vs 840 ms | **hurts** |
| 2-CB split (llama-style parallel encode) | split=14 → 1195 ms, split=27 → 1210 vs 1-CB 1126 ms | **hurts** |
| concurrent vs serial encoder (replay, no barrier) | 1126 vs 1143 ms (and immune to cold-start) | helps only WITHOUT the per-dispatch barrier; with barrier (engine) no benefit |

**Exact-shape replay** (real 7B shapes, all kernels, one CB): **~1126-1143 ms =
6.12-6.20 TF** — the best achievable with these kernels. The engine's own
GEMM-only (`MINFER_SKIP_ATTN=1 MINFER_SKIP_SMALL=1`, ~1240 ms today) is
**~90-115 ms slower than the replay** despite identical kernels/dispatch/barriers
— an engine-vs-harness residual not attributable to any tested factor (encode is
only ~1 ms, so it is GPU-side scheduling, below static visibility, consistent
with §4.3.6/§4.3.9).

**Final standing**: 7B pp495 minfer FULL ~1324-1336 ms vs llama ~1043-1064 ms
(clean), converging under load; the ~1.25× per-GEMM gap is confirmed NOT
source-addressable at any tested level. The concurrent-dispatch-type experiment
was reverted (no benefit with the engine's required per-dispatch barrier).




### 4.4 7B verification and A/B (user-facing model)

All §1 A/B numbers are on the 0.5B research model. The 7B Q4_K_M is the
user-facing model (steady decode ~50 ms/token GPU, prefill pp252 ~240 t/s with the
parallel attention). Add a **7B same-model A/B vs llama** (llama-bench on the
same `qwen2.5-7b-instruct-q4_k_m` split GGUF) to:
- quantify the real 7B gap per the same categories as §1.2 (matmul / attention /
  small / base);
- catch regressions from §4.2/§4.3 that 0.5B A/Bs would miss (7B has different
  dims: nh=28, gqa=7, hd=128, larger FFN — GEMM grid and attention chunk behavior
  differ).
Every completed step must pass: correctness byte-identical on 7B + 0.5B, no 7B
decode/prefill regression vs the pre-change baseline (same-caliber
`MINFER_TIMING` steady-state gpu submit-wait, tok 5+).

#### 4.4.1 7B baseline (2026-08-14) — the pre-change reference

Same model `qwen2.5-7b-instruct-q4_k_m` (2-part split GGUF, 4.36 GiB), same
pp252 prompt, M4 Pro. minfer = `--greedy --no-template`, decode steady = the
MINFER_TIMING `gpu(submit-wait)` tok 5+ segment; llama = llama-bench `-p 252 -n 32 -r 3`.

| metric | minfer | llama | minfer/llama |
|---|---|---|---|
| **prefill pp252** | 205-286 t/s (median ~240) | **460.7 ± 1.1 t/s** | **~52 %** |
| **decode tg32** | 14.5-19.0 t/s (median ~18.8; steady GPU 50.1-51.3 ms/token) | **50.5 ± 0.2 t/s** | **~37 %** |

0.5B sanity on the same commit (no regression): pp252 2009.7 vs llama 6138 t/s
(33 %), tg32 242.8 vs 291.8 t/s (83 %) — matches the §1.1 72-88 % decode band.

> **Key finding: 7B decode gap (~37 % of llama) is much larger than 0.5B
> (~83 %).** 7B has hd=128, nh=28, bigger KV — attention and small-kernel
> overhead dominate more at 7B, so the §4.2.2 flash-attention port is expected
> to help 7B the most. This baseline is the reference for every future
> step's regression check (same-caliber MINFER_TIMING steady-state, tok 5+).
> Note: 7B short-prompt "hi" prefill reads ~41 t/s (health check's 200 t/s
> threshold is 0.5B-calibrated and does NOT apply to 7B short prompts).

> **Post-hoc (2026-08-17)**: the decode flash hd=128 port (§4.2.2) confirmed the
> §4.2.2 "7B parity" prediction — steady decode GPU stays ~51 ms/token (flash
> ~51.1 vs split ~51.6 at pp205, byte-identical), i.e. 7B decode is
> weight-read-bound and the flash port is a correctness/dedup win, not a speed
> win at this step. The 37 % vs llama gap must come from the matmul kernels
> (weight-read-bound limit), not attention — and 7B's decode matmuls are
> **Q4_K-dominated** (not q8_0, which was already at parity): see §4.4.2 /
> to-do #7.

> **RESOLVED 2026-08-17 (to-do #7)**: the q4_K decode matmul port (§4.2.1) moved
> 7B steady decode GPU **~49.7 → ~19.3 ms/token (~2.6×)**, wall 17-19 → 45-49
> t/s — the 7B decode gap vs llama (50.5 t/s) is now **essentially closed**
> (≈95 %). The 51 ms baseline decomposed as q4_K ~35.7 ms + q6_K ~10 ms +
> attention/small; all decode matmuls now run at the ~200 GB/s floor.

#### 4.4.2 7B decode weight-type breakdown (2026-08-17)

`minfer info` on `qwen2.5-7b-instruct-q4_k_m` (2-part): the decode matmuls are
**q4_K-dominated** — `attn_q`/`attn_k`/`attn_output` and `ffn_gate`/`ffn_up`
are all `q4_K`; only `output`/`ffn_down`/`attn_v` are `q6_K` (already fixed in
§4.2.1). So at 7B the decode weight-read profile is the inverse of 0.5B's
(q6_K-heavy): the q6_K fix that shipped 2026-08-13 helps 7B less than it helped
0.5B, and q4_K is the untested, unoptimized decode type. This explains why the
flash decode hd=128 port (§4.2.2) moved 7B steady decode so little (~51.1 vs
~51.6 ms/token — attention was not the 7B decode bottleneck).

**Update 2026-08-17 (to-do #7 DONE)**: the q4_K port (§4.2.1) was the 7B decode
bottleneck. Weight-bytes check: q4_K ~2.57 GB/token (5 matmuls × 28 layers) was
reading at ~72 GB/s ≈ 35.7 ms; at the ~250 GB/s floor it is ~10 ms. Combined
with q6_K (~10 ms) and attention/small, 7B steady decode dropped to ~19.3
ms/token GPU — **at llama parity** (50.5 t/s).

### 4.5 Backfill after each item

After each item completes: update the §0 progress table (check, fill in the commit,
update measured effect) → record the implementation + verification in the relevant
§3/§4 section → update the §1 gap numbers.

---

## 5. Historical appendix (reference only)

> ⚠️ **This appendix is a historical record, for reference only — NOT the basis for
> future plans.** Its architecture conclusions, metric calibers, and kernel
> structures may have been superseded by §1-§4. Future plans follow §0 + §4 only.

### 5.1 Early phase summary (Qwen2-0.5B, 130→334 t/s, 2026-07-27~08-01)

| Phase | Content | Key points |
|---|---|---|
| 1 | 4 correctness bug fixes | RoPE freq_scale, output_b, softmax max, stack array (see §3.1) |
| 2 | Flash Attention (online softmax) + float4 vectorization | KV-parallel chunks + running max/sum fix |
| 3 | SIMD-parallel attention (vec kernel) | 32-lane simd_dot, threadgroup barrier sync |
| 4 | SIMD-parallel RMSNorm | multi-simdgroup reduction + threadgroup buffer |
| 5 | SwiGLU fusion | silu+mul single kernel, saves 1 dispatch/layer |

### 5.2 Early gap-analysis records

- **KV-growth 2.2×** (2026-08-01): f32 KV full re-read vs llama f16 (fix in §2.5).
- **Per-dispatch encode ~24µs vs llama ~7µs** (2026-08-01 conclusion) →
  **REVOKED 2026-08-03**: encode measured at only ~1 ms/step; decode is
  GPU-execution-bound.
- **Q4_0 dual dispatch (quantize+matmul)** (2026-08-01) → fixed: f32 activations
  path (§3.1 #4).

### 5.3 Tested-and-rejected ideas (with commits)

| Idea | Result | Commit |
|---|---|---|
| Parallel command buffers (A1) | regressed (encode already hidden), reverted | `b1256d5` |
| nt==1 matmul full-block matvec rewrite | at bandwidth floor, not integrated | — |
| store_kv_both / residual_rms_norm fusion | no gain, reverted | — |
| naive 1-kernel attention (classic single-pass) | 4.80 vs split 4.15 ms | — |
| f16 KV as default | ~3 % slower (0.5B) | `387d612` (kept opt-in) |
| Q5_0 full vectorization | only +2.6 % matmul | — |
| broadcast-GEMM for prefill attention | 2D GEMM can't produce per-head 3D scores | — |

### 5.4 Stale data tables (do not cite)

The following come from 2026-08-01/03 early measurements with different calibers
(blended t/s vs pure decode; old llama baseline) — **for historical cross-check
only**:
- "Q4_K_M/Q5_K_M prefill 7.3×" (35-token table) — before the non-Q4_0 GEMMs
  existed; now filled in.
- "decode short ~187 / long ~86 t/s" (KV-growth table) — before split attention.
- "pure decode 2.0× / 3.2×" early gap — now 1.1-1.4× (§1.1).

### 5.5 flash attention: llama.cpp vs minfer — detailed kernel comparison (reference for §4.2.2)

> Line-by-line structural comparison behind the §4.2.2 ~7-10× attention gap
> (minfer split attention 42.8 µs/layer vs llama flash ~4-6 µs/layer at
> nkv=430). Source: llama `kernel_flash_attn_ext_vec` (ggml-metal.metal:7218),
> llama dispatch (ggml-metal-ops.cpp:2959), minfer `kernel_gqa_attn_partial_f32`
> (metal.metal:2995), `_f16` (:3127), `kernel_gqa_attn_combine_f32` (:3243).

#### A. Overall design

| | llama `kernel_flash_attn_ext_vec` | minfer `partial + combine` |
|---|---|---|
| kernels/layer | **1** (nwg cross-TG reduce built in) | **2** (partial + combine) |
| per-TG scope | 1 query × 1 head, whole KV loop inside the kernel | 1 query × 1 KV-head × 1 chunk |
| KV parallelism | **32 workgroups each sweep a KV slice** (stride `NWG*NSG*C`), online-softmax partials → temp buffer + a 2nd reduce kernel | `n_chunks` chunks, partials → combine kernel |
| threads/layer (0.5B decode) | grid (1, 14, 32), 32 threads/TG → **14×32 = 448** | grid (1, 2, n_chunks), 32×gqa=224 threads/TG → 448×n_chunks |
| KV read | `half4`/`float4` direct from global (f16 cache), no explicit shmem staging | explicit KV-tile stage into threadgroup shmem per 32-row tile |

> Key: the **cross-TG reduction idea is identical** on both sides (online
> softmax partials + a combine pass). All the difference is **inside a single
> threadgroup**.

#### B. Inside one threadgroup — the core gap

**llama** (dk64 template: `NE=2, NL=16, C=32`, 32 threads = 2 rows × 16 cols):

```
QK^T:  for cc in 0..C/NE:                 // 16 cache columns per simdgroup
         for ii in 0..DK4/NL:             // 16 float4 dots
           mqk[cc] += dot(pk4[..], pq4[..])
         reduce = simd_shuffle_down ×5 + simd_shuffle   // ← NO threadgroup barrier
```

- QK^T accumulates in registers `mqk[]`; the reduction is `simd_shuffle_down`
  (16 lanes of ONE simdgroup merge via pairwise shuffles — **no
  threadgroup barrier needed**).
- softmax update: `M = simd_max`, `S = S*ms + simd_sum(vs)` (same simd reduces).
- PV: `lo[ii] += float4(pv4)*float4(sst)` register accumulate +
  `simd_shuffle_down` reduce.
- **only 2 `simdgroup_barrier` per tile** (QK→softmax, softmax→PV), and
  simdgroup-level (much cheaper than threadgroup-level).
- KV read straight as `half4` from global; relies on HW cache + barrier-free
  pipelining.

**minfer** (partial kernel, 224 threads/TG = 7 simdgroups × 32):

```
per 32-row tile:
  1. all threads stage the KV tile into threadgroup shmem (k_tile4/v_tile4)
  2. threadgroup_barrier                          // ① make data visible
  3. each lane does one row's QK^T:
       dot = 0; for d4: qv = qhead4[d4]*kj4[d4]; dot += qv.x+qv.y+qv.z+qv.w
       batch_mx = simd_max(dot); online-softmax (corr, acc4 *= corr)
  4. threadgroup_barrier                          // ② before reusing shmem
```

Gap sources:
- **After every QK^T reduce and before every PV accumulate llama only needs
  simd-level shuffles; minfer must `threadgroup_barrier`** — its 32 rows are
  processed serially by the same simdgroup's lanes (`for j0` loop, tile_sz=32)
  while different lanes of the 224-thread TG handle different heads (gqa=7).
  Result: **2 global barriers per tile, 28/layer at nkv=430**; each barrier
  stalls the whole TG to the slowest lane.
- llama's 16-column reduce stays inside one simdgroup (32 threads) — **zero
  cross-simdgroup sync**.
- minfer's `acc4` is register-accumulated but serialized by the barrier
  sequence; llama's `mqk/lo` register accumulation pipelines through the
  shuffle chain.

#### C. Why the gap is ~7-10× and not smaller

Of minfer's 42.8 µs/layer (nkv=430), the **28 threadgroup barriers + per-tile
shmem stage-in/out** dominate: a barrier makes all 224 threads wait, while each
32×64 tile does only ~2048 MACs — swamped by sync overhead. llama's simd
reductions drop sync to near zero, and `half4` reads with no shmem staging keep
the memory pipeline unbroken.

#### D. Port (option C) — actual change surface

For Qwen2.5-0.5B (hd=64, nh=14, nk=2, gqa=7, nt==1 decode):

1. **KV-layout check (prerequisite) — DONE 2026-08-14, PASS**: llama's K/V
   physical layout IS `[nkv][nk*hd]` (all KV heads packed in dim0 of the cache
   tensor, token stride `nk*hd`), and the flash kernel reads it with
   `nb11 = nk*hd*elem` (token), `nb12 = hd*elem` (head), `ns10 = nk*hd` — the
   same indexing minfer already uses (`k + ki*stride_kv + hk*hd`,
   `stride_kv = nk*hd`). **No cache rework needed**; the host passes the
   stride args mirroring minfer's layout (`nb10 = elem, nb11 = nk*hd*elem,
   nb12 = hd*elem, ns10 = nk*hd`). Evidence chain in §4.2.2.
 2. **Single-kernel rewrite — DONE 2026-08-14**: fixed `DK=DV=64, NE=2, C=32,
    NSG=1` (0.5B/7B decode dims), dropping llama's function-constant system
    (hardcoded constants; `n_chunks` is the host-tunable grid depth instead of
    llama's fixed NWG). Shmem `sq4 + ss + so4` (no `sm[]` — mask inlined),
    QK^T/PV via float4 + `simd_shuffle_down(8,4,2,1)` + `simd_shuffle(·, NL*ty)`
    broadcast (the reduce routes the full-head sum to lanes 0/16).
 3. **Cross-TG reduce — DONE**: reuse minfer's existing combine kernel — the
    flash kernel writes the same `{M, S, O[hd]}` partials (strided C=32-block
    chunking, interchangeable with the split's contiguous chunking).
 4. **isolation tests — DONE**: `tests/flash_attn_isolation.rs` (scalar CPU ref,
    multi-nkv incl. partial/empty chunks, nt 1-2, f32+f16; flash-vs-split A/B
    through the shared combine) + end-to-end byte-identical A/B
    (`MINFER_NO_FLASH=1`).
 5. **prefill untouched**: nt>1 keeps the current 3-pass parallel attention.

**Risk points**:
- KV layout mismatch was the most likely failure — **CLEARED** (2026-08-14,
  see D-1 above); the layout is compatible as-is.
- Function constants → hardcoded constants is NOT a line-by-line translation;
  shmem offsets must be re-derived (`sgitg*SH` terms are all 0 for NSG=1,
  which simplifies).
- `simd_shuffle_down` lane participation (NE=2 uses only the `NE>1`/`NE>2`
  branches; `simd_shuffle_down(mqk[cc], 16)` folds 16 cols onto lane 0) needs
  careful index alignment.

### 5.6 Glossary — variables and functions used in this document

> Common (both kernels, GGUF/model dims):

| Symbol | Meaning |
|---|---|
| `n_embd` / `ne` | model hidden size (embedding dim), e.g. 896 (0.5B), 3584 (7B) |
| `n_head` / `nh` | number of query heads, e.g. 14 (0.5B), 28 (7B) |
| `n_kv_embd` | KV cache head dim (`hd_kv`), e.g. 128 (0.5B); may differ from `n_embd` |
| `hd` / `hd_kv` | attention head dim: `n_embd/n_head` (64 for 0.5B) and KV head dim |
| `gqa` | group size = `nh/nk` (e.g. 7 for 0.5B) — GQA heads share a KV head |
| `nk` | number of KV heads (`n_head_kv`), e.g. 2 (0.5B) |
| `nkv` | number of KV cache positions (context length so far) |
| `nkt` | total KV cache capacity (max positions) |
| `nt` | number of tokens in the batch (prefill nt>1, decode nt==1) |
| `nqt` | number of query tokens (prefill) |
| `od` | output dim of a matmul (rows of the weight) |
| `id` | input dim of a matmul (cols of the weight) |
| `nf` | FFN intermediate dim (gate/up/down) |
| `positions` | per-token KV position array; `nkv = positions[t] + 1` |
| `max_pos` | largest position used so far |
| `n_chunks` | split-attention chunk count (`MINFER_ATTN_CHUNKS`, adaptive) |
| `Bc` | KV tile size in rows (32) used by the minfer attention kernels |
| `n_layers` / `n_layer` | transformer layer count (24 for 0.5B, 28 for 7B) |
| `Q4_0`/`Q4_K`/`Q5_0`/`Q5_K`/`Q6_K`/`Q8_0` | GGUF weight quant types (see §Quantization in AGENTS.md) |

> llama `ggml` tensor/strided-layout args (used in flash/matmul kernels):

| Symbol | Meaning |
|---|---|
| `ne00..ne33` | tensor dimensions: `ne0x`=dim0(dims of x-th src), `ne1x`=dim1, `ne2x`=dim2, `ne3x`=dim3 |
| `nb10..nb33` | byte stride of each dim for src1 (`nb10`=elem stride in bytes, `nb11`=row/token stride, etc.) |
| `ns10` / `ns20` | `nb11/nb10` and `nb21/nb20` — element count per head/row/token (used as the flash KV inner-loop stride) |
| `ne11` | KV cache length dim (`nkv`) in flash-attn args |
| `ne12`/`ne13` | KV head dim2/dim3 (GQA heads, batch) |
| `nwg` | number of workgroups (32 for flash vec, each sweeps a KV slice) |
| `nsg` | simdgroups per threadgroup (flash vec: 1; blk: 4-8) |
| `NWG`/`NSG` | function-constant copies of `nwg`/`nsg` inside the kernel |
| `NE` | columns per simdgroup in a flash tile (dk64: 2) |
| `NL` | lanes per column = `NW/NE` (dk64: 16), `NW` = 32 (simd width) |
| `C` | flash tile columns (32), `SH = 4*C` shared memory per simdgroup |
| `DK`/`DV` | flash key/value head dims (dk64/dv64 templates; Qwen2.5-0.5B: 64/64) |
| `DK4`/`DV4` | `DK/4`, `DV/4` (float4 element count) |
| `PK`/`PV` | `PAD2(DK,128)`/`PAD2(DV,128)` — padded head dim for shmem |
| `NL`/`NE` | see above (flash vec tile geometry) |

> Metal thread variables:

| Symbol | Meaning |
|---|---|
| `tgpig` | threadgroup position in grid (`.x/.y/.z` = the 3 grid dims) |
| `tiisg` | thread index in simdgroup (0..31) |
| `sgitg` | simdgroup index in threadgroup (0..nsg-1) |
| `simd_max`/`simd_sum` | SIMD (warp) reduction builtins |
| `simd_shuffle_down` | SIMD shuffle reduce (lanes exchange values pairwise) |
| `threadgroup_barrier` | TG-wide memory + execution barrier (all simdgroups) |
| `simdgroup_barrier` | simdgroup-wide barrier (cheaper, single warp) |
| `float4`/`half4` | 4-wide vector types (128-bit / 64-bit) used for SIMD loads |

> minfer kernels (src/metal.metal unless noted):

| Function | Meaning |
|---|---|
| `kernel_gqa_attn_f32/f16` | classic single-pass attention (grid (nt,nk), sequential KV loop) — prefill-past, superseded |
| `kernel_gqa_attn_partial_f32/_f16` | split attention pass 1: online-softmax partials (grid (nt, nk, n_chunks)) |
| `kernel_gqa_attn_combine_f32` | split attention pass 2: merge partials (grid (nt, nh)) |
| `kernel_attn_scores` / `kernel_softmax_attn` / `kernel_attn_output` | 3-pass parallel prefill attention (nt>1, barrier-free) |
| `kernel_store_kv_f32/f16` | KV cache store (f32 or f16 cache) |
| `kernel_q4_0_mm_f32` | Q4_0 prefill GEMM (simdgroup, nt≥16) |
| `kernel_q6_k_f32_matmul` | q6_K matmul (stride-2/float4 layout, decode; was the 3× gap, fixed §4.2.1) |
| `kernel_mul_mv_q6_K_f32_impl` | llama's q6_K kernel whose layout was ported |
| `kernel_rms_norm_fuse_impl` | llama's fused rms_norm kernel pattern (256-thread port `kernel_rms_norm_f32_256`) |
| `kernel_flash_attn_ext_vec` / `_blk` | llama flash kernels: vec = decode (nb<20, simd-shuffle), blk = prefill (simdgroup_matrix) |
| `kernel_cpy_f32_f16` | copy f32→f16 (used for f16 KV cache) |
| `kernel_get_rows_q4_0` | Q4_0 embedding gather |

> minfer env vars:

| Var | Meaning |
|---|---|
| `MINFER_ATTN_CHUNKS` | override split-attention chunk count |
| `MINFER_CACHE_TYPE` | `f16` = f16 KV cache (2 B/elem), default f32 |
| `MINFER_GEMM` | `0` = disable the Q4_0/non-Q4_0 prefill simdgroup GEMMs |
| `MINFER_NO_FUSE_QKV` | `1` = disable fused QKV/FFN-gu decode matmuls |
| `MINFER_SPLIT_CB` | `N` = split the decode into N command buffers |
| `MINFER_TIMING` | `1` = per-category decode GPU timing split |
| `MINFER_TRACE` | `1` = record per-dispatch labels (GPU hang debug) |
