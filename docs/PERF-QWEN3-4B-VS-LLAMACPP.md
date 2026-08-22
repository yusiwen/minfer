# Qwen3-4B Q4_K_M Performance: minfer vs llama.cpp

Analysis of where minfer (compute-graph path) stands against llama.cpp on the
same GGUF file, and *why* — measured on the M4 Pro dev machine, 2026-08.

## Setup

- **Hardware**: Apple M4 Pro (10 P + 4 E CPU cores, 20-core GPU, ~273 GB/s memory bandwidth).
- **Model**: `Qwen3-4B-Instruct-Q4_K_M.gguf` (2.32 GiB, 4.02 B params, 36 layers,
  hd=128 decoupled, GQA 8 kv-heads, kv dim 1024, `qwen3.context_length = 40960`).
  The **same file** drives both engines.
- **llama.cpp**: build `749f688fc (10547)`, `GGML_METAL=ON`; measured with
  `llama-bench` (`-p <n> -n <m> -ngl 99|0 -r 3..5`) and cross-checked with `llama-cli`.
- **minfer**: `target/release`, graph path; `MINFER_TIMING=1` (sample/forward split)
  and the `MINFER_OP_PROFILE=1` op profiler (committed with this analysis —
  per-op host-encode table on the first submit, one line per later submit).
- **Protocol**: every number measured alone (no concurrent benchmark processes —
  concurrent CPU load visibly corrupts Metal numbers, see §6).

## Headline numbers

| Scenario | minfer | llama.cpp | Ratio |
|---|---|---|---|
| **Metal decode** (tg, tok/s) | **75.4–75.9** | **78.6–79.7** | **~95 %** |
| **Metal prefill, steady-state** (marginal, tok/s) | ~800–1000 | ~900 | ≈ parity |
| Metal prefill, first request, 241 tok | 552 ms pre-fix → **~370 ms post-fix** (default n_ctx 4096) | 313 ms | 1.76× → ~1.2× wall |
| **CPU decode** (tok/s) | **1.1** | **61–68** (8–10 threads) | **~60×** |
| **CPU prefill** (tok/s) | ~1.1 | ~211 | **~190×** |

Post-fix note: the first-request gap was a KV-sizing policy issue (§2), now
fixed — the 241-token first request drops from 552 ms to ~370 ms
(241×1.15 + ~106 ms instead of + ~289 ms).

Bottom line: **on the GPU there is no meaningful gap** — decode is ~95 % of
llama and prefill is at parity in steady state; the visible first-request
difference is a KV-size/initialization policy issue, not a kernel issue. The
**real gap is CPU-only**: minfer is single-threaded and uses scalar (non-NEON)
dot products. Details below.

## 1. Metal decode: no gap (75.9 vs 79.7 tok/s)

Per-token decomposition of minfer's `forward()` (13.09 ms/token wall):

| Component | Time | Share |
|---|---|---|
| GPU execution (submit wait) | ~12.5 ms | ~98 % |
| Host encode of ~250 kernel dispatches | ~0.19 ms | ~1.5 % |
| Sampling | 0.08 ms | 0.6 % |
| Logits download + bookkeeping | remainder | — |

- Both engines are **memory-bandwidth bound**: every decode token reads the full
  2.49 GB of quantized weights. 2.49 GB / 12.5 ms ≈ **199 GB/s**, which is
  ~85–90 % of the M4 Pro's *practical* peak (~220–230 GB/s of the 273 GB/s spec).
  There is nothing left on the table at the kernel level: minfer's
  `kernel_q4_k_f32_matmul` is a faithful port of llama's
  `kernel_mul_mv_q4_K_f32_impl` (NR0=2 / NSG=2, stride-4 super-block thread
  layout) — same work, same dispatch.
- The residual ~4 % is measurement structure (llama-bench excludes logits
  download/sampling bookkeeping differently) plus micro-differences: KV cache
  **f32 vs llama's f16**, and per-token host overhead.
- `MINFER_OP_PROFILE` confirms nothing pathological on the host side: over a
  full prefill + 8 decode submits, MatMul encode totals 0.585 ms, every other op
  sub-ms; the decode GPU time is stable at 12.3–13.0 ms/submit.

## 2. Metal prefill: parity in steady state; the first request pays a KV-size tax

Both engines use **simdgroup GEMMs** for prefill matmuls: minfer's 64×32-tile
`kernel_q4_k_mm_f32` (f32 activations) vs llama's 64×64 `kernel_mul_mm_q4_K_f32`
(f16 activations). Both land at ~75–80 % of the GPU's fp32 peak — measured
marginal throughput:

| Prompt | minfer first-submit GPU (pre-fix) | llama (llama-bench) |
|---|---|---|
| pp16 / 5 tok | ~283 ms (5 tok) | 238 tok/s (67 ms) |
| pp120 / 121 tok | 390 ms | — |
| pp240 / 241 tok | 552 ms | 771 tok/s (313 ms) |
| pp480 / 481 tok | 851 ms | 821 tok/s (585 ms) |
| marginal (slope) | **~1.0–1.25 ms/tok** | **~1.1–1.2 ms/tok** |

→ steady-state prefill ≈ **800–1000 tok/s (minfer) vs ~900 tok/s (llama)**.

### The first-request tax (the 425-vs-771 wall-clock gap) — FIXED

minfer's single-shot CLI used to size the **KV regions at `max_seq_len =
40960`** (`Qwen3Graph::forward` passed `model.hparams.max_seq_len` straight
through, and `GraphParams.cparams.n_ctx` sizes the persistent per-layer KV
regions):

```
36 layers × 2 regions × 40960 × 1024 × 4 B (f32) = 12.1 GB of Metal shared buffers
```

The **first** GPU command buffer then paid ~275 ms — a one-time Metal
driver cost that scales with the total KV buffer bytes (measured first-submit
GPU time vs n_ctx — same process, `--cnv`):

| n_ctx | first-submit GPU time |
|---|---|
| 2048 (0.6 GB) | 87 ms |
| 4096 (1.2 GB) | 106 ms |
| 40960 (12.1 GB) | 289 ms |

After the first submit the cost never recurs (decode submits are 12.3–13.0 ms
each; a second, larger prefill in the same process — conv turn 2 — shows no
fixed cost either). Note the cost is **not** memory commitment: peak RSS is
identical (~2.1 GB) at n_ctx 4096 and 40960 — it is the Metal driver's
first-use setup (buffer VA/registration) for the oversized regions, so it is
strictly a one-time latency + address-space tax, never a resident-memory one.

llama.cpp avoids showing it because (a) its KV cache is sized by `n_ctx`
(default 4096 → ~1 GB) and (b) it runs a **warmup pass at model load** that
absorbs the one-time cost before the first real request.

Secondary consequence (pre-fix): the single-shot process held **12.1 GB** of f32
KV buffers it never used (a 10-token prompt), while `--cnv` / `--n-ctx`
correctly used the CLI value (conv turn-1 at n_ctx 4096: 106 ms; at 40960:
289 ms — the difference was entirely this sizing).

**Fix (this commit)**: `ModelDef::forward` now takes `n_ctx`; the single-shot
CLI passes `--n-ctx` (default 4096) and both `Qwen2Graph::forward` /
`Qwen3Graph::forward` clamp it to the model's `max_seq_len` (llama.cpp clamps
the same way). Result: default first-submit 289 ms → **~106–109 ms**, the
12.1 GB address-space allocation disappears, and the 5-token prefill wall
drops from 0.31 s to 0.11 s. Greedy output is byte-identical (KV *content*
is unchanged — only buffer sizes). `--n-ctx` now also documents as applying to
single-shot / `--cnv`, not just the server.

## 3. CPU: the real gap — ~60× decode, ~190× prefill

minfer CPU decode: **1.1 tok/s** (936 ms/token, 99.4 % in `forward()`).
llama.cpp CPU decode: **61–68 tok/s** (best at `-t 8`; `-t 10` → 60.8;
`-t 14` *including the 4 E-cores* collapses to 15.9 tok/s — E-cores hurt).

The gap decomposes into two compounding causes, both structural:

1. **Single-threaded (×~10)**: minfer's CPU backend has no threading
   (no rayon, no thread pool — only the HTTP server spawns a worker).
   llama.cpp uses 8–10 P-core threads.
2. **Scalar dot products on ARM (×~5.7 per core)**: minfer's quantized dot
   kernels are AVX2 (x86) with a **scalar fallback** — on aarch64 there is no
   SIMD path at all. llama.cpp uses NEON dot kernels (q4_K × q8_1 with
   int16 accumulate + vectorized activation quantization).

Per-core effective bandwidth: llama 166 GB/s ÷ 10 threads ≈ 16.6 GB/s/core vs
minfer 2.66 GB/s/core → 5.7× per core. 10 × 5.7 ≈ 57× ≈ the measured ~60×.

CPU prefill has the same per-token cost as decode (no cross-token reuse in the
matmul kernels), so minfer ~1.1 tok/s vs llama 211 tok/s (pp240) — ~190×.

## 4. Secondary: KV cache f32 vs f16 (long-context decode)

Decode attention re-reads the whole KV: 36 × 4 kv-heads × 128 hd × 2 (K+V) ×
4 B × C = 147 KB × C per token. At 191 GB/s:

| Context | minfer (f32) | llama (f16) | decode slowdown vs llama |
|---|---|---|---|
| 2048 | +1.6 ms/tok | +0.8 ms/tok | ~6 % |
| 4096 | +3.2 ms/tok | +1.6 ms/tok | ~13 % |
| 8192 | +6.3 ms/tok | +3.2 ms/tok | ~24 % |

Also halves the KV read traffic and the §2 first-submit tax. (llama.cpp's
default `kv_cache_type` is f16; minfer stores f32.)

## 5. Recommendations (ordered by measured impact)

1. ~~Fix single-shot KV sizing~~ **DONE (this commit)** — `ModelDef::forward`
   takes `n_ctx`; single-shot passes `--n-ctx` (default 4096), clamped to the
   model's `max_seq_len`. First-request latency 289 → ~106 ms, 12.1 GB
   address-space allocation gone; greedy output byte-identical.
2. **CPU backend: NEON dot products + threading** (the only big gap): aarch64
   SIMD kernels (q4_K/q8_1 style) + a small thread pool would take CPU decode
   from 1.1 toward 10–60 tok/s. Per-core scalar→NEON is worth ~5.7×, threading
   another ~10× (diminishing; E-core awareness needed, see §3).
3. **KV cache f16/bf16**: halves KV traffic and the §2 first-submit tax; worth
   ~6–24 % on long-context decode.
4. **Prefill marginal (optional)**: f16-activation GEMM or a 64×64 tile to close
   the last ~10 %; at current parity this is low priority.

## 6. Methodology notes / pitfalls seen

- **Never benchmark two engines concurrently**: an earlier llama-bench Metal run
  executed while the CPU benchmark was running measured 27.5 tok/s (vs 79.7
  alone) with 17 % variance. Metal decode still needs CPU cores to encode
  ~250 kernel dispatches per token.
- llama-bench's backend column lists *available* devices ("MTL,BLAS") even for
  `-ngl 0` — the `-ngl` value is the authoritative offload control.
- llama-cli perf lines: "Prompt: … t/s" = prefill, "Generation: … t/s" = decode.
- minfer's first prefill in a fresh process includes the KV-size first-submit
  tax (§2; post-fix ~106 ms at default n_ctx 4096); `--cnv` (n_ctx 4096) is the
  clean way to measure steady-state prefill, or subtract the measured tax from
  the single-shot total.
- Raw instrumentation: `MINFER_OP_PROFILE=1` (op-encode table + per-submit GPU
  time), `MINFER_TIMING=1` (sample vs forward per token).

## Appendix: raw measurements

| What | Value |
|---|---|
| minfer Metal decode, n=64 | 75.9 tok/s (forward 13.09 ms/tok, GPU 12.5) |
| minfer Metal decode, n=16 | 75.4 tok/s |
| minfer CPU decode, n=64 | 1.1 tok/s (forward 936 ms/tok) |
| llama-bench Metal tg64 | 79.73 ± 0.14 tok/s |
| llama-cli Metal Generation, n=48 | 78.6 tok/s |
| llama-bench CPU tg32/64 (t=10) | 60.74 ± 6.4 / 66.71 ± 3.2 tok/s |
| llama-cli CPU Generation (t=8 / 10 / 14) | 68.4 / 60.8 / 15.9 tok/s |
| llama-bench Metal pp16/240/480 | 238 / 771 / 821 tok/s |
| llama-bench CPU pp240 | 211 tok/s |
| minfer prefill GPU (first submit, pre-fix) | 5 tok: 283 ms · 121: 390 ms · 241: 552 ms · 481: 851 ms |
| minfer prefill GPU vs n_ctx (10 tok, pre-fix) | 2048: 87 ms · 4096: 106 ms · 40960: 289 ms |
| minfer first-submit, post-fix (n_ctx 4096 / 2048 / 8192 / 65536→clamped) | 109 / 99 / 127 / 308 ms |
| minfer first-submit RSS (n_ctx 4096 vs 40960) | 2.14 vs 2.12 GB — tax is not resident memory |
| minfer conv turn-2 delta prefill (n_ctx 4096) | 16 tok in 12.3 ms (no tax) |
| minfer 5-token prefill wall (pre → post fix) | 0.31 s → 0.11 s |
| greedy 64-token output, pre vs post fix | byte-identical |
