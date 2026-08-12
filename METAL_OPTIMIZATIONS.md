# Metal Backend Optimizations

> **Goal**: match llama.cpp (commit `88b47a755`, Apple M4 Pro) performance.
> **Current gap (2026-08-11 same-model, same-parameter A/B)**: decode is 72-88 % of
> llama (pure GPU 1.1-1.4×), prefill 2.8-3.6×. See [§1 Current state](#1-current-state).
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

### 🔜 To-do (required path to match llama.cpp)

> Principle (2026-08-12): we do NOT accept the current state — whatever
> llama.cpp can achieve, minfer must too. The former "accept the architecture
> floor" verdict is revoked; §4 is the only action path.

| # | Item | Goal | Status |
|---|---|---|---|
| 1 | **Xcode GUI per-kernel GPU trace** (minfer + llama, same workload, decode + prefill) | pinpoint the non-matmul 4× and GEMM ~30 % gap | not started. CLI method proven impossible (§4.1) |
| 2 | **flash attention port** (or an equivalent 1-kernel decode attention) | decode non-matmul 1.2→0.3 ms | gated on trace #1. Former "dead-end" verdict revoked (§4.2) |
| 3 | **prefill GEMM execution efficiency → ~7 TFLOPs/s** (llama level) | prefill 2.3-2.8× → 1× | grid-shape probe first (3.5-5.4 variance, §4.3) |
| 4 | remaining items per trace #1 results | — | TBD |

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

### 1.1 Same-model, same-parameter A/B (2026-08-11, M4 Pro, identical GGUF)

minfer `--greedy` (pure decode, llama "Generation" caliber); llama.cpp
`llama-bench -b 512 -t 8` (pure eval). Model Qwen2.5-0.5B-Instruct.

**Q4_K_M** (`qwen2.5-0.5b-instruct-q4_k_m.gguf`):

| Test | llama.cpp | minfer | Gap |
|---|---|---|---|
| prefill 30 tok | 2720 t/s | ~748 t/s | **3.6×** |
| prefill 430 tok | 6909 t/s | ~2466 t/s | **2.8×** |
| decode 128 tok (pure GPU) | 293-299 t/s | ~218 t/s (4.47 ms/tok steady) | **1.3-1.4×** |
| decode, default sampling | 247 t/s | ~197 t/s | **1.25×** |

**Q4_0** (`qwen2.5-0.5b-instruct-q4_0.gguf`):

| Test | llama.cpp | minfer | Gap |
|---|---|---|---|
| prefill 30 tok | 2610 t/s | ~812 t/s | **3.2×** |
| prefill 430 tok | 7449 t/s | ~2596 t/s | **2.9×** |
| decode 128 tok | 314-339 t/s | ~279 t/s (3.90 ms/tok steady) | **1.1-1.2×** |

**Reading**:
- **Decode is now 72-88 % of llama** (pure GPU 1.1-1.4×, default sampling 1.25×) —
  driven by rms_norm_256, the chunk-cap/sync fixes, and the per-kernel non-matmul
  profile (was 1.47× before 2026-08-10).
- **Prefill remains the main gap (2.8-3.6×)** — after the parallel attention fix
  (100→30 ms), the remainder is matmuls + small kernels (§4.3).

### 1.2 Per-token GPU decomposition (decode, nt==1, Q4_K_M 0.5B)

| Category | minfer GPU | llama GPU | Evidence |
|---|---|---|---|
| matmul (QKV/O/GU/down/output, ~97 kernels) | **~3.0 ms** (~130 GB/s) | ~3.0 ms (source+params identical) | minfer measured / llama inferred |
| attention (split 2 kernels) | **0.54 ms** | ~0.15-0.2 ms (flash vec 1 kernel) | minfer measured (skip-ATTN) / llama inferred |
| small elementwise (norm/bias/rope/store/add/swiglu, ~300) | **~0.5 ms** | ~0.1-0.3 ms | minfer measured / llama inferred |
| base infra (encode+submit+download) | encode 0.13 + download 0.02-0.03 | ~0.3-0.5 (incl. multi-cb encode) | minfer measured / llama inferred |
| **Total** | **~4.35-4.55 ms/token GPU** | **~3.1-3.3 ms GPU / 3.51 wall** | interleaved A/B |

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
> the inference after elimination. The decisive per-kernel measurement (Xcode GUI)
> was **never completed** (§4.1), so any "architecture floor" conclusion is
> provisional — and this project rejects it (see §4).

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

### 2.4 Final gap report (2026-08-06, precise decomposition)

| Component | minfer GPU | llama GPU | gap |
|---|---|---|---|
| matmuls | ~3.0 ms (bandwidth-bound, source+params identical) | ~3.0 ms | **~0** |
| **non-matmul** (attention + small + serialization) | **~1.2 ms** | **~0.3 ms** (inferred) | **~0.9 ms** |

**The structural gap is 100 % in non-matmul** — minfer's ~340 small kernels run at
~4× llama's efficiency. (Honest uncertainty: minfer's 1.2 ms is reliable; the
attention-vs-small split has ±0.2 ms noise; llama's 0.3 ms is inferred.)

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

### 4.1 Step 0 (required): Xcode GUI per-kernel GPU trace

**Why it is required**: every "structural / no low-risk lever" conclusion is based
on elimination + source identity, **never on per-kernel GPU timing** (§2.1 only
verifies down to "per-dispatch GPU time differs"). The CLI method is proven
impossible (2026-08-11 A2):
- `GGML_METAL_CAPTURE_COMPUTE=1` errors with "Capture layer is not inserted"
  (needs Xcode/Instruments GPU capture).
- `xctrace record --template 'Metal System Trace'` captures a .trace, but the CLI
  export is empty — the template has "Shader Timeline: Disabled"; per-kernel
  durations are only visible in the Xcode GUI.

**Action**: capture one decode and one prefill .trace each for minfer and
llama.cpp (same workload) in Xcode/Instruments; compare per-kernel GPU time. This
decides the priority of §4.2/§4.3 (attention-dominated? small-op-dominated?
uniform launch overhead?) and validates whether §1.2's subtractive minfer
decomposition overestimates matmul.

### 4.2 flash attention port (the decode non-matmul 4× gap)

**Former "dead-end" verdict revoked** (2026-08-12): the 2026-08-06 downgrade to
"dead-end" (~0.3 ms gain, multi-day risk) was made under the accept-floor premise.
Since the goal is to reach llama's level (non-matmul 1.2→0.3 ms is the only body of
the structural gap, §2.4), this is a **required path**, not an option.

Current state: minfer's split attention (2 kernels/layer, ~0.54 ms) is structurally
the best non-flash design (classic single-pass is 8× worse). llama's flash is fast
because of its `simdgroup_matrix` design + per-shape function constants, NOT
because it is "1 kernel" (a naive 1-kernel fusion measured 4.80 vs 4.15 ms, slower).

**Candidate**: faithful port of llama `kernel_flash_attn_ext_vec` (~600 lines,
simdgroup matrices, f16 KV, KV-parallel tiles) replacing partial+combine (−24
kernels). Risk: new kernel + isolation tests (follow the established methodology:
isolation test first against a scalar reference, then byte-identical A/B).

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

**Follow-up**: after §4.1's trace locates the GEMM gap (if inside the matmul),
pursue a higher-efficiency GEMM structure. 2D `simdgroup_matrix` (mpp tensor)
**already excluded** (llama disables it on M4 Pro, PARAMETER_AUDIT A); bf16 staging
**already excluded** (llama reads f32 activations).

### 4.4 Backfill after each item

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
