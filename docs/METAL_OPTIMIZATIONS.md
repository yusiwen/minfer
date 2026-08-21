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
| 25 | **hd=128 (7B) prefill flash port** (2026-08-15, §3.4 done) | 7B pp310 prefill GPU 1042→**~949 ms (~9 %)**, f32/f16 byte-identical, fixes 7B f16-cache garbage | §3.4 |
| 26 | **hd=128 (7B) decode flash port** (2026-08-17, §3.3 extend) | 7B decode steady GPU ~51.1 ms vs split ~51.6 ms (pp205), f32+f16+split byte-identical | §3.3 |
| 27 | **q4_K decode matmul layout port (7B)** (2026-08-17, to-do #7, §3.3) | 7B q4_K dims 70→265 GB/s (attn_q), 18→146 (attn_k), 74→243 GB/s (ffn_g/u); **7B decode steady GPU ~51 → ~19.3 ms/token (~2.6×, now ≈ llama's 50.5 t/s)**; 7B/0.5B byte-identical | to-do #7 |
| 28 | **GEMM partial-tile race + missing Metal `memoryBarrier`** (2026-08-19, §3.6): (a) all 8 simdgroup mm kernels lacked the `threadgroup_barrier` BEFORE the partial-tile `temp_str` stores — `temp_str` overlaps sa/sb, so a fast simdgroup overwrites sa/sb while a slow one still reads them → intermittently corrupted last-2-token logits (partial x-tile only); (b) the single prefill encoder had NO `memoryBarrier` between dispatches → RMSNorm write raced QKV read of the reused `bn` buffer (last-2 token slots, huge stale values) | 1.5B/7B first-token nondeterminism (~10-30 % wrong tokens, dump-localized) → **24/24 deterministic, output matches CPU byte-for-byte** | `7253a3b` |
| 29 | **mm-kernel hot-loop `#pragma unroll`** (2026-08-19, §3.4): llama `FOR_UNROLL`s the staging/ik/load/mac loops; minfer's 8 mm kernels had none → added the 6 unroll points (llama-parity set) | 7B pp495 **1438.8 → 1355.6 ms (~5.8 %)**, 0.5B ~6.9 %, 1.5B ~2.4 %; byte-identical + 24/24 determinism | `f3a499d` |
| 30 | **ik-loop `threadgroup_barrier` → `simdgroup_barrier(mem_none)`** (2026-08-19, §3.4 follow-up): .air diff showed the pre-unroll corruption was a rolled-loop compiler artifact, not a memory need; with the unroll in place llama's exact barrier form is now safe | 7B pp495 min **1387.4 → 1370.6 ms (~1.2 %)**; byte-identical (1.5B×24 / 7B×8 / 0.5B×3); removes the last structural mm-kernel difference vs llama | `0e756f3` |
| 31 | **Phase-0 7B prefill decomposition (2026-08-20, §3.6)**: exact 7B MUL_MAT graph mapped (197 GEMMs, 7.000 TFLOP, wk/down/output = q6_K on the q4_k_m; CORRECTS the earlier "all q4_K except ffn_down=q6_K" assumption); llama GPU-busy measured by host timestamps (CB1 83 ms + CB0 964 ms ≈ 1043 ms @ pp495 = 6.71 TF clean window); every remaining factor refuted via an exact-shape replay harness (kernels A/B in-batch 6.21 vs 6.26 TF, grid/smem/buffer-mode/pooling/barriers free, interleave + 2-CB split hurt, weight data no effect, concurrent dispatch no benefit with the per-dispatch barrier) | exact-shape replay (minfer kernel, real 7B shapes, one CB) = **~1126 ms = 6.20 TF**, converging with llama under comparable system load; engine GEMM-only ~1240 ms (residual ~90-115 ms engine-vs-replay, unattributable); gap vs llama stays ~1.25× (consistent with §3.4) | `bd89eab` |
| 32 | **lm_head / final-norm output-rows-only (2026-08-21, §3.7)**: llama computes the final norm + last-layer FFN + lm_head on **n_outputs rows only** (`ggml_get_rows(cur, inp_out_ids)` at qwen2.cpp:106-108; graph dump shows output GEMM `out=[152064 1]` and last-layer gate/down `[.. 1]`); minfer computed `[152064×495]`. **§3.6's "7.000 TFLOP" was WRONG** (assumed output N=495; correct llama total ≈ 6.26 TFLOP — the kernel-level "gap" was mostly this over-count). Fix: `forward()`/`output_norm_gpu`/CUDA all take `n_out`; final rms_norm + lm_head run on the tail n_out rows (n_out=1), logits buffer/download shrink 301 MB → 608 KB | 7B pp495 GPU **~1354 → ~1255 ms** (stable; pre-change noisy 1354-2754), **download ~150 ms → ~0.1 ms** (wall −~150 ms); 0.5B GPU output byte-identical pre/post; 1.5B/7B greedy generation correct | `43989da` |
| 33 | **GPU Q4_K embedding (get_rows) (2026-08-21)**: minfer's `kernel_get_rows_q4_0` was Q4_0-only, so the 7B/1.5B (Q4_K `token_embd.weight`) fell back to CPU scalar dequant + 7 MB `upload_hidden` per prefill (O(nt) wall: ~5-15 ms @ 495 tok, ~100-200 ms @ 8K). Added `kernel_get_rows_q4_k` (reuses the validated `dequant_q4_k_16`, llama `kernel_get_rows_q`-equivalent) + pipeline + type routing in `embed_tokens_gpu` | 7B/1.5B embedding now on GPU; greedy output **byte-identical** pre/post (same seed); `get_rows_q4_k_isolation` test bit-exact vs CPU; no prefill GPU regression | uncommitted |
| 34 | **Last-layer FFN output-rows-only (2026-08-21, §3.7 follow-up)**: llama reduces `cur` + `inpSA` to **n_out rows BEFORE the last layer's FFN** (`ggml_get_rows(cur, inp_out_ids)` + `get_rows(inpSA, inp_out_ids)` at qwen2.cpp:106-108) — the last layer's ffn_norm, gate/up/down, swiglu and BOTH residuals run on 1 row. minfer ran layer-27's entire FFN on all nt. Fix: `layer_gpu` now takes `n_out`/`is_last`; the wo matmul stays on all nt (llama `build_attn` precedes the reduction), then the wo-residual, ffn_norm (byte-offset read of the hidden tail), gate/up/down (dispatched with nt=n_out via the new `x_off` matmul param), swiglu and the final residual all run on the tail n_out rows (`add_f32_off`); CPU path mirrors. minfer's total graph work drops 6.46 → ≈6.26 TFLOP — **now exactly llama's total** | 7B pp499 GPU **~1278 → ~1234 ms (~44 ms, ~3.4 %)** — the doc's ~40 ms estimate confirmed; byte-identical (7B/0.5B GPU greedy + 0.5B CPU fallback, same seeds); decode unchanged (nt==1 path untouched); 33/34 bin tests green (1 pre-existing env-dependent failure: `attn_parallel_realdata_correctness` needs `/tmp/dp3` dumps) | `b6ecbd3` |
| 35 | **Precompiled metallib (2026-08-21, §4.2 to-do #1)**: build.rs compiles `src/metal.metal` → embedded `.metallib` (`xcrun metal -O3` + `metallib`, llama's exact flags; clang module cache redirected via `-fmodules-cache-path` since the default cache dir is unwritable under the build sandbox) → loaded with `newLibraryWithData`; empty-marker fallback to `newLibraryWithSource` when the toolchain is absent. Numerics verified **byte-identical** to the runtime source compile (7B @615/499 + 0.5B greedy, all 4 isolation suites, 33/34 bin tests) — a first apparent divergence was an A/B reference prompt-mixup, not a compiler difference; `-O0` metallib lacks `kernel_q4_1_f32_matmul` (falls back to CPU), so `-O3` only. Runtime hook `MINFER_METALLIB_FILE=<path>` loads an external metallib without rebuilds | 0.5B process wall **~1.32 → ~1.09 s** (warm driver cache; the first-ever run benefits most — no per-process compile), prefill/decode perf unchanged (equal within noise); shader errors now caught at build time | `d09b8db` |
| 36 | **GGUF mmap + zero-copy weights (2026-08-21, §4.2 to-do #2)**: `std::fs::read` (4.4 GB) + per-tensor `extend_from_slice` copy + GPU `new_buffer`+memcpy were THREE full-weight copies (~8.8 GB RAM + 4.4 GB GPU). Now: each part is `mmap`'d (MAP_PRIVATE, zero-dep raw `mmap`/`munmap` FFI, leaked for the process) and `Tensor.data` is a `Cow<'static,[u8]>` **Borrowed** slice of it (zero per-tensor copy; the `output = tok_embd.clone()` weight-tying fallback is now a shallow clone too); the Metal backend wraps each part with ONE page-aligned `newBufferWithBytesNoCopy` (`register_part`) and registers weights as **(buffer, byte offset)** into it — llama's exact design (`ggml_metal_buffer_map` page-aligns; `newBufferWithBytesNoCopy` requires a page-aligned base — per-weight NoCopy at 32-aligned bases was tried first and reads SHIFTED data on the GPU, hence the offset design). `MINFER_WEIGHT_COPY=1` forces the old copy path for A/B | 7B load+prefill wall **~4.3 → ~2.7 s** (warm; first-run 7.2 → 3.3 s); peak RSS **20.9 → 4.7 GB (~4.4×)** — the weights are file-backed pages shared with CPU/GPU; 7B/0.5B greedy byte-identical + CPU fallback identical; prefill/decode perf unchanged (equal within noise); 34/35 bin tests green (1 pre-existing env-dependent) | `d51e8b8` |
| 37 | **f16 KV auto-default (2026-08-21)**: `set_kv_cache_type(n_layers, n_kv_embd)` at model load auto-selects the GPU KV element type — **f16 for the 7B class** (n_layers×n_kv_embd ≥ 8192: KV bandwidth-bound decode), f32 for small models; `MINFER_CACHE_TYPE=f16/f32` overrides. llama always defaults F16 (`llama-context.cpp:3539`); minfer had kept f32 default because 0.5B f16 measured ~3% slower (§0 decided-not #8) — the auto rule applies f16 exactly where it wins and keeps the 0.5B class on f32 | 7B @2K ctx steady decode **f16 ≈ 20.15 ms vs f32 21.13 ms/token (~1 ms, ~5 %)**; 7B greedy byte-identical across auto/f32/f16 (16 tokens); 0.5B untouched (auto→f32); 34/35 bin tests green | this commit  `9099f79` |
| 38 | **GPU get_rows for the remaining embedding types (2026-08-21)**: llama's `kernel_get_rows_q` covers every quant; minfer had Q4_0+Q4_K only (#33), so the 0.5B Q5_0/Q5_1/Q8_0/Q6_K-embedding models (q4_k_m embd=Q5_0, q5_k_m embd=Q5_1, q5_0/q8_0/q6_k models) fell back to CPU scalar dequant + `upload_hidden` per prefill. Added MSL templates `kernel_get_rows_q32` (Q4_1/Q5_0/Q5_1/Q8_0, one thread per 32-elem block, reuses the validated `dequant_*_16` helpers) + `kernel_get_rows_q256` (Q6_K/Q5_K, one thread per 16-elem group, same structure as the Q4_K kernel) + pipeline routing/guards (ne%32 / ne%256) in `embed_tokens_gpu` | `tests/gemm_isolation.rs::get_rows_multi_type_isolation` — all 6 new kernels **bit-exact vs CPU** (rel 0); end-to-end q5_0/q5_k_m/q8_0 GPU == CPU greedy (same seed); 0.5B q4_k_m (embd Q5_0) now embeds on GPU (was CPU fallback); 1.5B Q4_K / 7B Q4_K unchanged; 34/35 bin tests green. **Note**: the 0.5B q6_k model shows a PRE-EXISTING GPU-vs-CPU greedy divergence (reproduced on the pre-#38 binary — not from this change; its 896-dim Q6_K embd is ne%256≠0 → CPU fallback) — flagged for later investigation | this commit |
| 39 | **GPU warm-up read + merged embed (2026-08-21)**: (a) the mmap loader (#36) REGRESSED cold-start prefill — the FIRST GPU access to file-backed (mmap) pages costs ~44 ms of one-time page/TLB setup per process (0.5B pp1 wall 20 → 56 ms vs the copy path; a CPU-side madvise/touch does NOT fix it — the cost is the GPU's own access). Fix: `register_part` now dispatches a dummy `kernel_warmup_read` over the whole part buffer at model load (outside the CLI's Total timing; llama-bench numbers are equally warm). (b) the embed is now dispatched into the MAIN command buffer (llama builds `ggml_get_rows` into the main graph — one submit instead of two). | 0.5B pp1 wall **56 → 14 ms**, pp31 prefill **~430 → ~950 t/s** (llama 2686, gap 6× → 2.4×); 7B pp31 **~190 → ~247 t/s** (llama 328, gap 1.7× → 1.3×), pp499 GPU 1215 → **1142 ms (~6 %)**, first-decode −~50 ms; cost: +~0.2 s 7B load (the read is amortized into load), 0.5B/7B outputs byte-identical, 34/35 bin tests green | this commit |
| 40 | **Small-batch prefill matmul threshold (2026-08-21)**: minfer dispatched the simdgroup GEMMs only for nt ≥ 16 (chosen on the 0.5B in P0); nt∈[9,15] fell to the `_multi` kernels which serialize t INSIDE the threadgroup — measured **7B pp12: 16.6 t/s vs llama 130 (~7.8×)**, 0.5B pp12 ~3.9×. Fix: adaptive rule `nt ≥ 2 && (od ≥ 2048 || nt ≥ 9)` — GEMM for all nt ≥ 9 (llama `ne11_mm_min=8` → MM for ne11>8) AND for nt∈[2,8] with large od (7B class: od≥3584 — GEMM ≈ llama's `kernel_mul_mv_ext` there, measured pp4 7B 34 vs llama 61); small-od 0.5B matmuls keep the multi at [2,8] (measured better). The nt∈[2,8] small-od gap vs llama's ext kernel (0.5B pp4 141 vs 583) remains open — low value (2-8 token prompts), deferred | **7B pp12 16.6 → ~124 t/s (≈llama 130, parity)**, pp4 15 → 34 t/s; 0.5B pp12 278 → ~500 t/s, pp31 unchanged (~950); byte-identical (7B/0.5B greedy), pp499 parity, 34/35 bin tests green | this commit |

**Completed optimizations in detail** (every done / decided-not item):
[§3.1 Correctness fixes](#31-correctness-fixes-metal-backend-foundation) ·
[§3.2 CPU sampler](#32-cpu-sampler-2026-08-06) ·
[§3.3 Decode optimizations](#33-decode-optimizations-gpu) ·
[§3.4 Prefill optimizations](#34-prefill-optimizations) ·
[§3.5 KV / long context](#35-kv-long-context) ·
[§3.6 Prefill GEMM gap investigation (resolved)](#36-prefill-gemm-gap-investigation-resolved-2026-08-14-08-20-decided-not-to-change)

### 🔜 To-do (required path to match llama.cpp)

> Principle (2026-08-12): we do NOT accept the current state — whatever
> llama.cpp can achieve, minfer must too. The former "accept the architecture
> floor" verdict is revoked; §4 is the only action path.

| # | Item | Goal | Status |
|---|---|---|---|
| 1 | **GPU trace (minfer + llama): Performance Limiters + per-kernel** | per-phase bottleneck + per-op durations for both sides | **DONE 2026-08-13** (§3.3): per-kernel + limiter comparison. Early "1.6-3.9×" numbers were trace-semantics artifacts (fused-vs-separate, mixed od); the clean isolation A/B (llama test-backend-ops perf) shows q5_0/q8_0 at parity and **q6_K ffn_down 3× slower** — fixed (this row) |
| 2 | **decode matmul per-call execution** | **q6_K 72→209 GB/s (llama 217); decode 4.27→3.72 ms/tok (~13%)** | **q6_K DONE 2026-08-13** (§3.3): ported llama's stride-2/float4 kernel layout; byte-identical + tests green. q5_0/q8_0 already at parity. q4_K next if a K_M model uses it |
| 3 | **flash attention port** | decode **attention** 42.8→~4-6 µs/layer (~7-10×) | **DONE 2026-08-14** (§3.3): ported `kernel_flash_attn_ext_vec` (NSG=1, DK=DV=64/NE=2/C=32) as `kernel_flash_attn_ext_f32/_f16`; KV-layout check PASSED (minfer `[nkv][nk*hd]` == llama physical layout — no cache rework). Isolation-verified (`tests/flash_attn_isolation.rs`: cos vs CPU >0.999 for nkv 1..4097 incl. partial/empty chunks; flash-vs-split cos=1.0 through the shared combine), A/B byte-identical (0.5B Q4_K_M f32 + Q4_0 f16 + 7B Q4_K_M), decode GPU 0.25-1.0 ms/token faster (interleaved MINFER_TIMING), wall ~10 %; no long-context regression. Gate: `MINFER_NO_FLASH=1` reverts to split |
| 4 | **prefill flash attention port** (llama `kernel_flash_attn_ext_blk`, legacy `simdgroup_matrix`) + prefill GEMM/small efficiency | prefill 2.3-2.8× → ~1.5× (135 → ~90 ms); GEMM/small 89→~44 ms secondary | **GEMMs RULED OUT 2026-08-14 (§3.6)**. Grid-shape probe (3.5-5.4 variance) + barrier/store experiments rule out the GEMM kernels (mem_none ≈ mem_threadgroup ~2-3 % and RACES in minfer). Real pp325 decomposition (0.5B Q4_K_M, MINFER_SKIP_ATTN): **attention 46 ms (34 %)**, everything-else 89 ms. llama pp320 = 47.7 ms total with attention only ~3 ms (6803 vs 6373 t/s `-fa on/off`). llama's prefill attention is `kernel_flash_attn_ext_blk` = **legacy simdgroup-matrix (has_simdgroup_mm, NOT the M5 tensor API)** — single fused kernel vs minfer's 3-pass. **PORT DONE 2026-08-14 (§3.4)**: `kernel_flash_attn_blk_f32/_f16` (fixed-shape NSG=4, Q=8, C=64, DK=DV=64, 7168 B shmem, inline causal mask, `kernel_kv_tail_pad` for the partial last block) + host `attn_flash_prefill`. Isolation-verified (`tests/flash_attn_blk_isolation.rs`: cos vs CPU >0.999 across 16 nt/nkv configs incl. partial blocks + GQA, f32+f16, deterministic), A/B **byte-identical to the classic `gqa_attn_f32`** at every layer (f32 AND f16 cache — maxabs 0.0), interleaved MINFER_TIMING prefill GPU **~110→~93 ms (~16 %)**, all 34 bin + 9 isolation tests pass. **Bonus: FIXES the f16-cache prefill 3-pass bug** (the 3-pass `kernel_attn_scores`/`kernel_attn_output` read the f16 KV cache as `float*` → garbage "!!!!!!"; the f16 blk kernel reads half K/V correctly). Default for hd==64 (0.5B/1.5B) and — since the 2026-08-15 hd=128 port — for hd==128 (7B) too. Gate: `MINFER_NO_PREFILL_FLASH=1` reverts to 3-pass. Non-attention 89 vs ~44 ms remains a secondary structural gap — **7B direct per-kernel GEMM A/B 2026-08-18 (§3.6): minfer GEMMs at 87-94 % of llama (parity). Phase 0 prefill decomposition 2026-08-18 (§3.6): GEMMs are 76 % (0.5B) / 88 % (7B) of prefill; small kernels only 10 % / 4 % → #1 fusion ceiling low. **Phase X 2026-08-18 (§3.6): §4.3.4's parity was a test-backend-ops measurement artifact — llama real prefill GEMMs ≈ 6.9 TFLOPS (467 t/s pp466) vs minfer ≈ 5.2 (≈1.33×). Concurrency, fusion, small kernels, dequant type, ik-loop barrier, sb-staging vectorization, and geometry ALL measured/verified — none explains the 1.33×; the gap is not addressable from minfer source (compiler-level only).** **FINAL 2026-08-20 (§3.6)**: exact 7B MUL_MAT graph mapped (197 GEMMs, 7.000 TFLOP; wk/down/output = q6_K — CORRECTS the earlier assumption), llama GPU-busy by host timestamps ≈ 1043 ms @ pp495 (6.71 TF, clean window), exact-shape replay (real 7B shapes, minfer kernels, one CB) ≈ **1126 ms = 6.20 TF**. Isolated AND in-batch kernel A/B equal (6.21 vs 6.26 TF); grid/smem/buffer-mode/pooling/weight-data/barriers all free; GEMM interleave + 2-CB split hurt; concurrent dispatch no benefit with the engine's required per-dispatch barrier. Residual engine-vs-replay ~90-115 ms unattributable (GPU-side scheduling, below static visibility). Under comparable system load (avg 3-4) llama-bench pp495 degrades to 1250-1760 ms and **converges with the replay**. **CLOSED: attention port DONE, GEMM/small gap confirmed NOT source-addressable at every tested level — only the tensor-API GEMM (`mpp::tensor_ops`, llama disables on M4) remains as a beat-llama research direction, not a parity fix.** **2026-08-21 CORRECTION (§3.7 / #32)**: the '7.000 TFLOP / 6.71 TF' figures were WRONG — the output GEMM is N=1 (not N=495), so llama ≈ 6.26 TFLOP ≈ 6.0 TF ≈ the replay. The measured 'gap' was mostly minfer's full-nt lm_head over-count (≈539 GFLOP + 301 MB download), now FIXED (output-rows-only): 7B pp495 GPU ~1354 → ~1255 ms + download −~150 ms. Residual ≈1.2× remains not source-addressable (kernel/graph level).** |
| 5 | **7B same-model A/B + per-step regression check** (0.5B is the research model; 7B is the user-facing one) | 7B decode/prefill gap vs llama quantified; no 7B regression from each step | **BASELINE 2026-08-14** (§1.6): 7B Q4_K_M pp252: prefill **~240 t/s (52 % of llama 461)**, decode **~18.8 t/s (37 % of llama 50.5)**, steady GPU 50.1-51.3 ms/token. 0.5B sanity: pp252 2010 t/s (33 %), tg32 243 t/s (83 %) — no regression. **CURRENT 2026-08-21**: decode at parity (≈50 t/s, #27); prefill **pp495 ~1255 ms vs llama ~1043-1064 ms (~82 %, §3.6/§3.7)** after the lm_head output-rows-only fix (#32: GPU −~100 ms + download −~150 ms). Regression checks run per step; all green |
| 6 | ~~decode small-elementwise efficiency~~ | — | **CLOSED 2026-08-13**: trace shows small-op parity (1.2-2.0 vs 1.3-1.9 µs) — the old 4× claim was subtractive noise (§3.3) |
| 7 | **q4_K decode matmul layout port (7B)** — the next decode lever | 7B decode 37 % → closer to llama (steady GPU ~50 → target ~30 ms/token) | **DONE 2026-08-17** (§3.3): 7B K_M decode matmuls are **Q4_K-dominated** (attn_q/k/output + ffn_gate/up all Q4_K; Q6_K only output/ffn_down/attn_v). Ported llama's `kernel_mul_mv_q4_K_f32_impl` stride-4/float4 layout into `kernel_q4_k_f32_matmul` (TG(32, nsg=2) dispatch, `sc16`/kmask nibble unpack — the scale/min high/low nibble interleave reproduces llama's `get_scale_min_k4` exactly, verified against llama's dequantizer). Steps: ① isolation probe at 7B dims — old kernel 70/18/74 GB/s (attn_q 3584/3584, attn_k 3584/512, ffn_g/u 18944/3584) → new **265/146/243 GB/s** ② kernel port ③ 7B steady-decode A/B: **~49.7 → ~19.3 ms/token GPU (~2.6×), 17-19 → 45-49 t/s (≈ llama's 50.5)**, git-stash A/B byte-identical ④ 0.5B (no q4_K weights — trivially identical) + 1.5B (q4_K decode + multi path) regression green; all 34 bin tests pass |

> Note: §3.3's per-kernel table is superseded by the clean isolation A/B
> (§3.3): the trace mixed fused-vs-separate and different od per kernel name.
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
| 8 | f16 KV cache as default | 0.5B measured ~3 % slower (dispatch-latency-bound) → **REPLACED by auto-select (#37)**: f16 for the 7B class (KV bandwidth-bound), f32 for small models — the 0.5B-side of this decision still stands |
| 9 | **MTLDispatchTypeConcurrent encoder** (2026-08-20, §3.6) | helps only WITHOUT the per-dispatch barrier (replay 1126 vs 1143 ms); with the engine's required `memoryBarrier` it is noise (FULL 1331-1370 vs serial 1336) — reverted |
| 10 | **GEMM-interleave / 2-CB split** (2026-08-20, §3.6) | dummy attn/norm dispatches between GEMMs: 845 vs 840 ms (hurts); llama-style split CB1+CB0: 1195-1210 vs 1126 ms (hurts) |
| 11 | **Prefill GEMM kernel experiments** (2026-08-14, §3.6) | grid-shape probe (3.5-5.4 TF variance) not the lever; `mem_none` ik-loop barrier + vectorized sb-store both **RACY in minfer** (deterministic corruption) — the `mem_threadgroup` barrier is a genuine correctness requirement |

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
| prefill 30 tok | 2610 t/s | ~950 t/s (pp31, #39; was ~430) | ~2.4× |
| prefill 430 tok | 7449 t/s | ~2770 t/s (pp435, flash) | **2.7×** |
| decode 128 tok | 314-339 t/s | ~279 t/s (3.90 ms/tok steady) | **1.1-1.2×** |

**Reading**:
- **Decode is now 72-88 % of llama** (pure GPU 1.1-1.4×, default sampling 1.25×) —
  driven by rms_norm_256, the chunk-cap/sync fixes, and the per-kernel non-matmul
  profile (was 1.47× before 2026-08-10).
- **Prefill (long) improved from 2.8-3.6× to ~2.6-2.7×** after the prefill flash
  port (§3.4): GPU ~164 → ~144 ms at pp435 (~12 %; pp294 ~16 %). The residual
  gap is the non-attention 89 → ~44 ms structural difference (GEMMs + small
  kernels under-occupied, §3.6), NOT attention (now ~3-4 ms, llama-like).
- **Short prefill (pp30)**: dominated by per-dispatch fixed overhead (~950 t/s
  at pp31 after #39 — the mmap first-access TLB cost is now absorbed at load;
  was ~430 t/s), so pp30 is no longer a meaningful attention lever; llama's
  pp30 is similarly launch-bound (0.5B 2686, 7B 328 t/s — minfer now 2.4×/1.3×).

> **Post-#28-32 (2026-08-21)**: the §1.1 0.5B numbers remain the reference —
> the GEMM-partial-tile barrier fix (#28) was a correctness fix (no regression),
> the unroll/barrier changes (#29/#30) improved 0.5B prefill ~7-9 % on top, and
> the lm_head output-rows-only fix (#32, §3.7) is a prefill/lm_head change that
> does not alter the 0.5B decode reference here. No re-measurement has
> invalidated any §1.1 figure. The user-facing **7B** state is the current
> focus — see §1.6.

### 1.2 Per-token GPU decomposition (decode, nt==1, Q4_K_M 0.5B)

| Category | minfer GPU | llama GPU | Evidence |
|---|---|---|---|
| matmul (QKV/O/GU/down/output, ~97 kernels) | **~3.0 ms** (~130 GB/s) | ~3.0 ms (source+params identical) | minfer measured / llama inferred |
| attention (split 2 kernels) | **0.54 ms** | ~0.15-0.2 ms (flash vec 1 kernel) | minfer measured (skip-ATTN) / llama inferred |
| small elementwise (norm/bias/rope/store/add/swiglu, ~300) | **~0.5 ms** | ~0.1-0.3 ms | minfer measured / llama inferred |
| base infra (encode+submit+download) | encode 0.13 + download 0.02-0.03 | ~0.3-0.5 (incl. multi-cb encode) | minfer measured / llama inferred |
| **Total** | **~4.35-4.55 ms/token GPU** | **~3.1-3.3 ms GPU / 3.51 wall** | interleaved A/B |

> **SUPERSEDED 2026-08-13 by the per-kernel trace (§3.3)**: the llama-side
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

### 1.6 Current 7B state (user-facing model, Q4_K_M, 2026-08-21)

The 7B (`qwen2.5-7b-instruct-q4_k_m`, split GGUF) is the user-facing model; this
is the current standing after the decode q4_K port (#27), the correctness/unroll/
barrier work (#28-30), the Phase-0 prefill decomposition (#31, §3.6), the
lm_head output-rows-only fix (#32, §3.7), and the GPU Q4_K embedding port (#33).

| Metric | minfer | llama.cpp | Gap |
|---|---|---|---|
| **decode** (steady GPU) | ~19.3 ms/token (45-49 t/s) | ~50.5 t/s (19.8 ms/tok) | **≈95 % (parity)** |
| **prefill pp495 GPU** | **~1234 ms @ pp499** (was ~1278 pre-#34; ≈5.07 TF @ ≈6.26 TFLOP) | ~1043-1064 ms (≈6.0 TF @ 6.26 TFLOP) | **~86 % (~1.16-1.18×)** |
| **prefill pp495 logits download** | **~0.1 ms** (608 KB, was ~150 ms/301 MB) | blit, 608 KB | parity |
| exact-shape replay (pure GEMM, one CB) | ~1126 ms (6.20 TF) | — | converges with llama under comparable load |

- **Decode is at parity** (essentially closed by #27's q4_K decode matmul port).
- **Prefill**: the big lm_head over-count (minfer computed `[152064×495]` logits,
  llama computes `[152064×1]` after `get_rows(inp_out_ids)`) is FIXED (#32) —
  GPU ~1354 → ~1255 ms, download ~150 → ~0.1 ms. The corrected FLOP accounting
  (llama ≈ 6.26 TFLOP, NOT 7.0) shows llama ≈ 6.0 TF ≈ the replay's 6.2 TF.
- The residual ~1.2× prefill gap is kernel/graph-level, below static visibility
  (GPU backend machine-code scheduling / execution environment) and **accepted**
  (§3.6). The only remaining research direction is the tensor-API GEMM
  (`mpp::tensor_ops`) — llama disables it on M4 by default, so it is a
  beat-llama option, not a parity fix. The ~40 ms follow-up (last-layer FFN on
  output rows only, §3.7) is now **DONE (#34)** — minfer's total graph work
  (≈6.26 TFLOP) exactly matches llama's, and the pp495 gap closed 1.2× → ~1.16-1.18×.

---

## 2. Gap analysis: verified vs inferred

> **Core finding (2026-08-06 #5, verbatim)**: "Structural" is an **inference, not a
> proven architectural inferiority**. §2.1 is what is strictly VERIFIED; §2.2 is
> the inference after elimination. The decisive per-kernel measurement was
> **completed 2026-08-13** (§3.3) — see its per-kernel table, which refuted
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

> **SUPERSEDED 2026-08-13 by §3.3's per-kernel trace.** The "~0 gap matmuls"
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

**q6_K decode matmul layout port (2026-08-13, §3.3)** — the decode gap was
matmul-dominated, not attention/small-op (per-kernel trace §3.3). Clean
isolation A/B (minfer `matmul_bandwidth_profile` vs llama `test-backend-ops
perf -b MTL0 -o MUL_MAT`, same od/id/nt) showed **q6_K ffn_down (896/4864) 72 →
217 GB/s (3.0× gap)** — minfer's stride-64 super-block loop left only 19/64 TG
threads busy (~30 % util) with scalar inner loops. Fix: ported llama's
`kernel_mul_mv_q6_K_f32_impl` stride-2 + float4 layout (TG(32, nsg=2) dispatch):
- q6_K isolation 72 → **209 GB/s** (llama 217); decode steady GPU **4.27 → 3.72
  ms/token (~13 %)**; wall ~203 → ~220 t/s (Generated: pure-decode).
- byte-identical (git-stash A/B), all tests green. q5_0/q8_0 already at parity.

**q4_K decode matmul layout port (2026-08-17, to-do #7, §3.3)** — 7B K_M
decode is Q4_K-dominated (attn_q/k/output + ffn_gate/up all q4_K; Q6_K only
output/ffn_down/attn_v) and carried the same gap q6_K had. Rewrote
`kernel_q4_k_f32_matmul` as a faithful `kernel_mul_mv_q4_K_f32_impl`
transcription: stride-4 super-block loop, float4 acc over the kmask nibble
unpack, `sc16` scale unpack (verified byte-exact vs llama's `get_scale_min_k4`,
ggml-quants.c `dequantize_row_q4_K`), TG(32, nsg=2), grid od/4:
- 7B isolation at decode dims: attn_q (3584/3584) 70 → **265 GB/s**, attn_k
  (3584/512) 18 → **146 GB/s**, ffn_gate/up (18944/3584) 74 → **243 GB/s**.
- **7B steady decode GPU ~49.7 → ~19.3 ms/token (~2.6×)**; wall 17-19 → 45-49
  t/s (llama 50.5). 0.5B/1.5B regression green, all 34 bin tests pass.

**flash decode attention port (2026-08-14 hd=64 + 2026-08-17 hd=128, §3.3)** —
isolation confirmed attention at **42.8 µs/layer (partial+combine) vs llama's
~4-6 µs (flash vec) = ~7-10×**, ~0.9 ms of the 3.72 ms decode step (~24 %).
Root cause: llama's flash uses `dot(float4,float4)` + `simd_shuffle_down`
reductions with **no threadgroup barrier within a KV tile**; minfer's split had
2 `threadgroup_barrier` per 32-row tile. Ported llama's `kernel_flash_attn_ext_vec`
as `kernel_flash_attn_ext_f32/_f16` (NSG=1, DK=DV=64, NE=2, C=32, NL=16,
fixed-shape for Qwen2, writes the same {M,S,O} partials so the shared combine
merges unchanged). KV-layout check PASSED (minfer `[nkv][nk*hd]` == llama
physical layout — no cache rework). GPU-safety deviations: inline per-lane
partial-chunk mask (llama's cross-lane `sm[]` write is a race), break-only
control flow so all 32 lanes reach both barriers, clamped reads + `-MINF`
masking instead of a pad buffer. Verified: `tests/flash_attn_isolation.rs`
(cos vs CPU >0.999 for nkv 1..4097, cos=1.0 vs split through the shared
combine), A/B byte-identical (0.5B Q4_K_M f32 + Q4_0 f16 + 7B Q4_K_M), decode
GPU ~0.3-1.0 ms/token faster, wall ~10 %. **hd=128 decode flash (7B,
2026-08-17)**: separate `kernel_flash_attn_ext_hd128_f32/_f16` (NE=1, DK4=DV4=32,
full `simd_sum(mqk)` per cc-iteration); 7B pp205 steady decode GPU flash ~51.1
vs split ~51.6 ms/token (weight-read-bound; attention was NOT the 7B decode
bottleneck — the q4_K port below was).

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

**Prefill flash attention port (2026-08-14, §3.4, `5974eb1`)** — llama's
prefill attention is a SINGLE fused `kernel_flash_attn_ext_blk`
(`simdgroup_matrix`, NSG=4, Q=8, C=64) vs minfer's 3-pass (scores + softmax +
output); measured minfer attention 46 ms of the 135 ms pp325 (34 %) vs llama's
~3 ms. Ported `kernel_flash_attn_blk_f32/_f16` (fixed-shape NSG=4, DK=DV=64,
7168 B shmem, inline causal mask, `kernel_kv_tail_pad` for the partial last KV
block). Isolation-verified (`tests/flash_attn_blk_isolation.rs`: cos vs CPU
>0.999 across 16 nt/nkv configs incl. partial blocks + GQA, f32+f16,
deterministic), A/B **byte-identical to the classic `gqa_attn_f32`** at every
layer (f32 AND f16 cache — maxabs 0.0), interleaved MINFER_TIMING prefill GPU
**~110 → ~93 ms (~16 %)**. **Bonus: fixes the f16-cache prefill 3-pass bug**
(the 3-pass kernels read the f16 KV cache as `float*` → garbage "!!!!!!"; the
f16 blk kernel reads half K/V correctly). Default for hd==64 (0.5B/1.5B).
Gate: `MINFER_NO_PREFILL_FLASH=1`.

**hd=128 (7B) prefill flash port (2026-08-15, §3.4, `89ac39a`)** — 7B pp310
prefill GPU 1042 → **~949 ms (~9 %)**, f32/f16 byte-identical, fixes 7B
f16-cache garbage. Default for hd==128 (7B) too.

**GEMM partial-tile race + missing Metal `memoryBarrier` fix (2026-08-19,
§3.4, `7253a3b`)** — (a) all 8 simdgroup mm kernels lacked the
`threadgroup_barrier` BEFORE the partial-tile `temp_str` stores (`temp_str`
overlaps sa/sb, so a fast simdgroup overwrites them while a slow one still
reads → intermittently corrupted last-2-token logits, partial x-tile only);
(b) the single prefill encoder had NO `memoryBarrier` between dispatches →
RMSNorm write raced the QKV read of the reused `bn` buffer (last-2 token slots,
huge stale values). Result: 1.5B/7B first-token nondeterminism (~10-30 % wrong
tokens) → **24/24 deterministic, output matches CPU byte-for-byte**.

**mm-kernel hot-loop `#pragma unroll` (2026-08-19, §3.4, `f3a499d`)** —
llama `FOR_UNROLL`s the staging/ik/load/mac loops; minfer's 8 mm kernels had
none. Added the 6 unroll points (llama-parity set): 7B pp495 **1438.8 →
1355.6 ms (~5.8 %)**, 0.5B ~6.9 %, 1.5B ~2.4 %; byte-identical + 24/24
determinism.

**ik-loop `threadgroup_barrier` → `simdgroup_barrier(mem_none)` (2026-08-19,
§3.4 follow-up, `0e756f3`)** — the pre-unroll corruption was a rolled-loop
compiler artifact, not a memory-visibility need (.air diff: the barrier swap
changes ONLY the barrier instruction — zero IR scheduling difference). With the
unroll in place llama's exact barrier form is now safe: 7B pp495 min
**1387.4 → 1370.6 ms (~1.2 %)**, byte-identical (1.5B×24 / 7B×8 / 0.5B×3).
Removes the last structural mm-kernel difference vs llama.

### 3.5 KV / long context

- **KV geometric growth** (`66f4290`): `kv_ensure_layer` grows ×2 instead of
  reallocating + copying the whole old KV every token (0.5 ms@KV140 → 4.2 ms@KV2510
  → 0.13 ms). ⚠️ an `old_v` clone typo polluted the V cache (Q4_K_M garbage) — the
  A/B didn't catch it (both paths share the corrupted KV); found against a
  known-good reference.
- **f16 KV auto-default** (`387d612` + `bff73db` + #37): `MINFER_CACHE_TYPE=f16/f32`
  overrides; unset → `set_kv_cache_type` at load picks f16 for the 7B class
  (n_layers×n_kv_embd ≥ 8192) and f32 for small models (0.5B measured ~3 % slower
  on f16 — dispatch-latency-bound). `kernel_store_kv_f16` + the `_f16` flash /
  partial kernels serve the f16 path.
- **Split-GGUF** (`cbba68c`/`34eaf10`): multi-part models (7B `-0000X-of-0000Y`),
  merged tensor index, 7B verified.

### 3.6 Prefill GEMM gap investigation — RESOLVED (2026-08-14 → 08-20, decided not to change)

The prefill GEMM throughput gap was investigated exhaustively and is now
**closed as "not source-addressable"** (the mm kernels are proven identical to
llama's). Record of the investigation (was §4.3.1-§4.3.10):

| Step | Finding | Verdict |
|---|---|---|
| §4.3.1 (08-14): grid-shape probe + kernel experiments | GEMM efficiency varies 3.5→5.4 TF purely by grid shape; `mem_none` barrier + vectorized sb-store either no-gain or **RACY in minfer** (deterministic corruption) | GEMM kernels ruled out as the prefill lever |
| §4.3.4 (08-18): 7B direct per-kernel GEMM A/B | minfer GEMMs at **87-94 % of llama** (test-backend-ops) | GEMMs at parity (isolation) |
| §4.3.5 (08-18): Phase 0 subtractive decomposition | GEMMs = 76 % (0.5B) / 88 % (7B) of prefill; small kernels only 10 % / 4 % | #1 fusion ceiling low |
| §4.3.6 (08-18): real-chain mechanism | §4.3.4's parity was a test-backend-ops artifact — llama real prefill GEMMs ≈ 6.9 TF vs minfer ≈ 5.2 (≈1.33×); concurrency/fusion/small-kernels/dequant/barrier/vectorization/geometry all measured — none explains 1.33× | gap not addressable from minfer source |
| §4.3.7 (08-19): partial-tile race + missing `memoryBarrier` | FIXED (→ §3.4) | correctness |
| §4.3.8 (08-19): re-baseline + last cheap levers | MSL `-O3` no benefit; concurrent dispatch ~0.5 % (noise); dependency-aware barrier ≈ barrier-always | gap accepted |
| §4.3.9 (08-19): `#pragma unroll` + ik-loop barrier | ~5.8 % + ~1.2 % recoveries (→ §3.4); structural-equivalence audit: source/IR/.air/smem/dispatch/runtime-compile ALL identical | last structural mm-kernel difference gone |
| §4.3.10 (08-20): Phase-0 7B decomposition + exact-shape replay | 7B MUL_MAT graph = 197 GEMMs / 7.000 TFLOP (wk/down/output q6_K — CORRECTS earlier assumption); llama GPU-busy ≈ 1043 ms @ pp495 (6.71 TF, host timestamps); exact-shape replay (real shapes, one CB) ≈ **1126 ms = 6.20 TF**; isolated + in-batch kernel A/B EQUAL (6.21 vs 6.26 TF); grid/smem/buffer-mode/pooling/weight-data/barriers free; GEMM-interleave + 2-CB split HURT; concurrent dispatch no benefit with the per-dispatch barrier; engine-vs-replay residual ~90-115 ms unattributable (GPU-side scheduling, below static visibility) | **CLOSED**: gap confirmed not source-addressable at every tested level |

**Residual status (pre-#32, 2026-08-20)**: 7B pp495 minfer FULL ~1324-1336 ms
vs llama ~1043-1064 ms (~73 %, ≈1.25-1.3×), converging under comparable system
load (llama-bench degrades to 1250-1760 ms at load avg 3-4). **Updated 2026-08-21
by #32 (§3.7)**: after the lm_head output-rows-only fix minfer is ~1255 ms GPU
(≈82 %, ≈1.2×) with download ~0.1 ms — see §1.6 / §3.7. The only remaining
research direction is the **tensor-API GEMM** (`mpp::tensor_ops`) — llama
disables it on M4 by default, so it is a beat-llama option, not a parity fix
(decided-not, §0).

> **⚠️ 2026-08-21 CORRECTION — the §3.6 FLOP accounting was WRONG.** The
> "7.000 TFLOP" total assumed the output (lm_head) GEMM runs on all N=495
> tokens. The graph dump actually shows `out=[152064 1 1 1]` for the output and
> the last layer's gate/up/down on N=1 — llama's `ggml_get_rows(cur,
> inp_out_ids)` (qwen2.cpp:106-108) reduces to **n_outputs rows after the last
> attention**, so the last layer's FFN + final norm + lm_head all run on N=1.
> Correct llama total ≈ **6.26 TFLOP** → llama efficiency ≈ 6.0 TF (NOT 6.71),
> essentially equal to the replay's 6.2 TF. **minfer was computing the full-nt
> output GEMM `[152064×495]` (≈539 GFLOP waste) + final norm on all rows + a
> 301 MB logits download — a large fraction of the measured "gap" was this
> over-count, not the kernels.** Fixed 2026-08-21 (§3.7 / changelog #32).

### 3.7 lm_head / final-norm output-rows-only (2026-08-21, changelog #32)

**Discovery** (from the `LLAMA_METAL_E2E.md` reference): llama's graph reduces
the hidden state to `n_outputs` rows right after the last layer's attention
(`ggml_get_rows(cur, inp_out_ids)` + `get_rows(inpSA, inp_out_ids)`,
`src/models/qwen2.cpp:106-108`), so the last layer's FFN, the final norm and the
lm_head all run on **1 row** for a single-sequence prefill (graph dump: output
GEMM `out=[152064 1 1 1]`, last-layer gate/down `[.. 1 ..]`). minfer computed
the output projection over **all nt tokens** (`[nv×nt]`) and downloaded all of
it.

**Fix**: `forward()` / `output_norm_gpu` (Metal + CUDA) now take `n_out`
(number of output rows = the LAST n_out tokens, single-sequence row-major
`[nt][ne]` hidden). The final `rms_norm` + output GEMM + bias + logits buffer +
download all operate on `n_out` rows (n_out=1 for the minfer CLI). Host:
`src/models/qwen2/forward.rs` (GPU `output_norm_gpu(…, n_out, …)`, CUDA path,
CPU fallback slices `hidden[(nt-n_out)*ne..]`), `src/metal.rs:2012` +
`rms_norm(.., off)` byte-offset, `src/main.rs` passes `n_out=1`.

**Measured** (7B Q4_K_M pp495, interleaved A/B same window):
- GPU submit-wait: pre ~1354 (noisy 1354-2754) → post **1253-1271 ms** (stable)
- logits download: ~150 ms (301 MB) → **~0.1 ms** (608 KB)
- 0.5B GPU generation **byte-identical** pre/post (same seed, same text); 1.5B /
  7B greedy generation correct.
- Corrected efficiency: llama ≈ 6.0 TF @ 6.26 TFLOP; minfer post-#32 ≈
  6.46 TFLOP @ ~1.255 s ≈ 5.15 TF — the remaining ~1.2× was the §3.6
  kernel/graph residual plus the **last layer's FFN still on all nt** (llama
  reduces before it) ≈ ~40 ms follow-up potential.

**Follow-up — DONE 2026-08-21 (#34)**: the last layer's FFN now runs on the
output rows only, exactly matching llama's `get_rows(inp_out_ids)` reduction
(qwen2.cpp:106-108 reduces `cur` + `inpSA` BEFORE the last layer's FFN). Fix in
`layer_gpu` (`metal.rs`): the wo matmul stays on all nt (llama `build_attn`
precedes the reduction), then the wo-residual, ffn_norm (byte-offset read of the
hidden tail), gate/up/down (dispatched with nt=n_out via the new `x_off` matmul
param) and the final residual all run on the tail n_out rows (`add_f32_off`);
the CPU path in `forward.rs` mirrors (gate/up/down on n_out rows, residuals on
the hidden tail slice).
- **Measured** (7B Q4_K_M pp499, interleaved A/B same window): GPU submit-wait
  **~1278 → ~1234 ms (~44 ms, ~3.4 %)** — the ~40 ms estimate confirmed.
- minfer's total graph work drops **6.46 → ≈6.26 TFLOP — now exactly llama's
  total**; efficiency ≈ 5.07 TF @ 6.26 TFLOP (llama ≈ 6.0 TF) → gap ~1.16-1.18×.
- Correctness: 7B/0.5B GPU greedy **byte-identical** pre/post (same seeds) +
  0.5B CPU fallback identical; decode untouched (nt==1 ⇒ ffn_nt==nt, the reduced
  path never triggers); 33/34 bin tests green (`attn_parallel_realdata_correctness`
  fails on missing `/tmp/dp3` dumps — pre-existing environment dependency).


---

## 4. Roadmap status — all items resolved (2026-08-20)

> The 2026-08-12 principle — "we do not accept the current state; whatever
> llama.cpp can achieve, minfer must too" — drove the work below. Every roadmap
> item is now **resolved** (done, or decided-not-to-change with measured /
> source evidence). The detailed records live in §3 ("Completed optimizations
> in detail", incl. the §3.6 investigation record) — see the §3 links in §0's
> Progress Overview; decided-not items are in §0's "Decided not to change"
> table. This chapter holds the remaining research + operational reference,
> plus the **cold-start to-dos added 2026-08-21 (§4.2)** (a separate axis from
> the steady-state gap — not part of the §0 "match llama.cpp" table).

### 4.1 Remaining research (not a parity fix)

The only direction left to beat llama on prefill is the **tensor-API GEMM**
(`mpp::tensor_ops`, `kernel_mul_mm_id`) — llama disables it on M4 Pro by
default (PARAMETER_AUDIT A), so it is a research exploration, not a parity
requirement. No other prefill path remains open (§3.6 / §0 decided #1).

**2026-08-21 (#32, §3.7)**: the last-layer FFN output-rows-only follow-up
(llama's `get_rows(inp_out_ids)` shrinks the graph before layer-27's FFN, closing
the minfer-vs-llama prefill work difference 6.46 vs 6.26 TFLOP) is now **DONE
(#34)** — no prefill work difference remains; minfer's total equals llama's.

### 4.2 Cold-start optimization to-dos (2026-08-21)

**Observed**: 7B run twice in a row → the second run ≈ 2× faster (Total
1.46 s → 0.77 s). Note the CLI's `Total` timing starts AFTER model load
(`main.rs:313 infer_start`), so the 2× is in the **inference phases**
(prefill + decode), not the load. Root causes:

| Factor | Effect | Evidence |
|---|---|---|
| **GPU weight-buffer cold state (run 1)** | the 5.2 GB weights are freshly CPU-memcpy'd into Shared buffers at load; the GPU's first read hits cold MMU/TLB + page residency → slower prefill + decode on run 1 | decode 32.6 t/s (~84 GB/s) run 1 vs 46.4 t/s (~119 GB/s) run 2 |
| **GPU clock ramp** | first GPU burst after idle starts below max clock | secondary |
| **Model-load wall (not in `Total`, but real wall time)** | 4.4 GB `std::fs::read` (gguf.rs:1711,1736) + Metal shader source compile (`newLibraryWithSource`, metal.rs:1120) — run 2 mitigated by the OS page cache + the Metal driver's on-disk shader cache | run 1 load visibly slow, run 2 ~free |

Warm steady-state = run 2's numbers (pp30 ~0.17 s, decode ~46 t/s). Not a bug.

| # | Item | Current | Approach (llama parity) | Expected | Blocker / Risk |
|---|---|---|---|---|---|
| 1 | **Precompiled metallib** | ~~every process compiles `metal.metal` from source via `newLibraryWithSource`~~ → **DONE 2026-08-21 (#35)** | build-time `metal` compiler → embed a `.metallib` → load with `newLibraryWithData` | remove the per-invocation shader compile — 0.5B process wall 1.32 → **1.09 s** (warm; the first-ever run benefits most) | the standalone Metal toolchain (`xcrun metal`) IS installed; build.rs falls back to source compile when absent — no numerics risk (byte-identical, verified) |
| 2 | **GGUF mmap + zero-copy weight buffers** | ~~`std::fs::read` the whole file into a Vec, then copy each weight into a Shared GPU buffer (2 passes)~~ → **DONE 2026-08-21 (#36)** | mmap the GGUF; `newBufferWithBytesNoCopy` over the mapped data (llama `ggml-metal-device.m:1668`) | remove the 4.4 GB copy pass + the 4.4 GB intermediate Vec; the GPU reads the mapped file pages directly — 7B load wall 4.3→**2.7 s**, peak RSS 20.9→**4.7 GB** | `newBufferWithBytesNoCopy` requires a page-aligned base → ONE buffer per mmap'd part + per-weight (buffer, offset), exactly llama's design. **Cold-start regression (fixed #39)**: the first GPU access to file-backed pages cost ~44 ms per process (short prompts) — absorbed at load by a dummy GPU warm-up read (`kernel_warmup_read`) |
| 3 | **Persistent / server mode** (note) | one-shot CLI: every invocation re-loads the model | keep the process alive and reuse the loaded model (llama-server pattern) | removes reload for repeated calls — the definitive fix for the observed run-1/run-2 gap | out of the current CLI scope |

> Item 1 removes a genuinely avoidable per-invocation cost; item 2 is mostly a
> memory + one-copy-pass win (the 4.4 GB disk read is unavoidable either way).
> Item 3 is the only way to make *every* invocation fast, at the cost of a
> daemon. Together they target the cold-start axis only — steady-state
> inference is unaffected (already at the §1.6 numbers).

### 4.3 Reference: GPU profiling tooling (done, operational)

- `xctrace`: `/usr/bin/xctrace` is a broken stub ("tool not found"); the real
  binary is `/Applications/Xcode.app/Contents/Developer/usr/bin/xctrace`.
- Record in Instruments: Metal System Trace, **Counter Set → Performance
  Limiters**, Enable Shader Timeline on, Deferred.
- `scripts/export_trace.sh <trace> [run]`: exports per-kernel durations
  (`metal-shader-profiler-intervals`), the limiter profile
  (`gpu-counter-value`), and per-forward intervals (`metal-gpu-intervals`);
  `TRACE_PROC=<proc>` filters per-forward intervals.
- Interpretation: percentage counters (Limiter/Utilization/Occupancy) are
  ratios; bandwidth counters (L1/LLC Read Bandwidth) are cumulative — ignore
  magnitude.

### 4.4 Process (was §4.5 backfill)

After each item completes: update the §0 progress table (check, fill in the
commit, update measured effect) → record the implementation + verification in
the relevant §3 section → update the §1 gap numbers.

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

### 5.5 flash attention: llama.cpp vs minfer — detailed kernel comparison (reference for §3.3)

> Line-by-line structural comparison behind the §3.3 ~7-10× attention gap
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
   nb12 = hd*elem, ns10 = nk*hd`). Evidence chain in §3.3.
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
| `kernel_q6_k_f32_matmul` | q6_K matmul (stride-2/float4 layout, decode; was the 3× gap, fixed §3.3) |
| `kernel_mul_mv_q6_K_f32_impl` | llama's q6_K kernel whose layout was ported |
| `kernel_rms_norm_fuse_impl` | llama's fused rms_norm kernel pattern (256-thread port `kernel_rms_norm_f32_256`) |
| `kernel_flash_attn_ext_vec` / `_blk` | llama flash kernels: vec = decode (nb<20, simd-shuffle), blk = prefill (simdgroup_matrix) |
| `kernel_cpy_f32_f16` | copy f32→f16 (used for f16 KV cache) |
| `kernel_get_rows_q4_0` | Q4_0 embedding gather |

> minfer env vars:

| Var | Meaning |
|---|---|
| `MINFER_ATTN_CHUNKS` | override split-attention chunk count |
| `MINFER_CACHE_TYPE` | `f16`/`f32` KV cache override; unset → auto (7B class f16, small f32, #37) |
| `MINFER_GEMM` | `0` = disable the Q4_0/non-Q4_0 prefill simdgroup GEMMs |
| `MINFER_NO_FUSE_QKV` | `1` = disable fused QKV/FFN-gu decode matmuls |
| `MINFER_SPLIT_CB` | `N` = split the decode into N command buffers |
| `MINFER_TIMING` | `1` = per-category decode GPU timing split |
| `MINFER_TRACE` | `1` = record per-dispatch labels (GPU hang debug) |
