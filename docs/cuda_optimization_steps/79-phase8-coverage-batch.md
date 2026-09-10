# 79 · Phase-8 coverage & first-measurement batch — 8b KV f16, 8c shaped Q8_0 GEMM, 8d split-K attention, 8f Q5_K kernels, 8l llama parity baseline, 8q Q5_0 (LANDED)

> **Result**: KV f16 +~11% 7B @2K decode (8b); shaped Q8_0-activation prefill
> GEMM +24% 0.5B prefill with a −63% negative that proved the shape gate
> load-bearing (8c); split-K flash-decoding +36% 7B @2K decode (8d, later
> superseded by R4); Q5_K/Q5_1 kernels completed the CUDA model gate (8f); the
> 8l parity benchmark set the llama.cpp baseline (decode 1.15–1.76×, prefill
> 18–110× gaps) that directed 8m–8p and the whole MMQ campaign; 8q enabled
> Q5_0 models (0.5B q4_k_m: 148.7 → ~1200 prefill / 56.9 → ~306 decode tok/s).
> **Commits**: `f7b0036` (8b), `69a27c5` (8c), `a5af60f` (8d), `b959ec9` (8f),
> `acca28f` (8l), `9f419f9` (8q, squashed). **Dates**: 2026-08-29 → 2026-09-08.

## 1. Background — where things stood

After the correctness batch (doc 78), Phase 8 chased two things in parallel:
decode/prefill performance headroom, and model coverage (a weight type the
CUDA gate rejects means a whole model silently falls back to CPU). This doc
records the six items that neither produced a master-table "campaign" of their
own nor fit the perf-step docs 02–06 — the coverage work and the first honest
measurements against llama.cpp. The 8l benchmark deserves special weight: its
decode and prefill gap tables are the origin of the 8m–8p prefill line, the
R-series, and the r5–r60 MMQ campaign.

## 2. Principle — the GPU mechanism (per item, in brief)

- **8b — KV f16 halves the attention/KV byte traffic.** K/V rows are read once
  per decode step per head; storing them as f16 instead of f32 halves those
  bytes while the accumulation stays f32. The win grows with context (more KV
  rows to stream).
- **8c — activation quantization only pays when the matmul is
  activation-heavy.** q8_0×int8 dots trade per-element dequant for a smaller
  byte stream; at id ≤ 8192 (attn/qkv/o shapes) the A-side savings dominate;
  at 7B ffn_down (id=18944) the shape is weight-bound and the q8_0 kernel
  streams weight bytes SLOWER than the f32 kernel — hence a shape gate.
- **8d — split-K fills the machine.** The incumbent kernel ran one warp per
  (token, head): 28 warps total at 7B @2K (the GPU is idle), streaming 2K KV
  rows serially at ~3 GB/s effective. Splitting the KV scan into SPLITS=8
  chunks scanned in parallel (fixed grid → capture-safe) raises warps-in-flight
  ~8×; pass 2 merges (mx, S, oc) partials.
- **8f — the all-or-nothing gate needs EVERY matmul weight type.** The CUDA
  participation gate admits a model only if every matmul weight has a kernel;
  the 0.5B "q5_k_m" file actually stores Q5_1 + Q6_K + Q8_0, so Q5_1 kernels
  were required alongside Q5_K.
- **8l — measurement, not mechanism.** Cross-benchmarking against llama.cpp on
  the same GGUF files turns "we feel slower" into per-shape gap numbers, which
  is what ranks the follow-up levers.
- **8q — the CUDA alignment contract is not advisory.** A
  `reinterpret_cast<const uint32_t*>` load of a word at byte offset 2 of a
  22-byte block is only 2-byte aligned for even block indices — on GB10
  unified memory it faults nondeterministically depending on page-mapping
  state (small-shape parity tests passed; the real 93.6 MB tok_embd faulted).

## 3. Implementation

### 3.1 8b — KV f16 on CUDA (`f7b0036`)

`store_kv_f16` + `gqa_attn_f32_f16kv` — an exact structural mirror of the f32
attention kernel: same online softmax, same warp reductions, same guards; the
only delta is half4 → float4 K/V loads with f32 accumulation. Policy mirrors
Metal: `MINFER_CACHE_TYPE=f16|f32` override, auto f16 when
`n_layers × n_kv_embd ≥ 8192` (the 7B class), decided at model load and cached
per `CudaBackend` instance. Caveat on record: `MINFER_GRAPH_DUMP` reads KV
regions as f32, so it is incompatible with f16 KV (debug dump path only).

### 3.2 8c — prefill Q8_0-activation GEMM, SHAPED (`69a27c5`)

Measure-first verdict (standalone nvcc A/B, quantization included):
+38–44% at id ≤ 8192 (0.5B shapes, 7B attn/qkv/o — activation-heavy),
+4.7% at 7B ffn_gu (weight-bound), **−63% at 7B ffn_down (id=18944)**. A blind
wire of the 7e⑥ idea would have REGRESSED 7B-class Q4_0 ffn_down by ~60%.
Wired only the winning region: `nt > 1 && id ≤ 8192` routes Q4_0 prefill
through `quantize_q8_0` + `q4_0_q8_0_matmul` (grow-on-demand scratch;
capture-safe because prefill never captures since 8g①, doc 78). E2E: 0.5B
prefill @3.6K tokens 1005 → 1246 tok/s (+24%), greedy text unchanged.

### 3.3 8d — split-K flash-decoding decode attention (`a5af60f`)

nsys capture was fixed first (the 7e② "no kernel data" issue was the report
workflow, not the config): `nsys profile --trace=cuda` + `nsys stats` / sqlite
over `CUPTI_ACTIVITY_KIND_KERNEL`, decode step isolated by gap segmentation.
Attribution at 7B @2K decode: gqa_attn 48.3% of the step, q4_k matmul 36.4%,
q6_k 13.9%. Fix: split-K flash-decoding — pass 1 scans SPLITS=8 KV chunks in
parallel (FIXED grid × nh blocks, ranges derived from device-side positions so
CUDA Graph capture stays valid), pass 2 merges (mx, S, oc) partials. Size-
stable state scratch, grown during warmup only; templated over the KV element
type (f16 + f32 layouts); dispatched at nt == 1 only. Superseded later by R4's
dim-parallel lane rewrite (doc 10), which removed the local-memory accumulator
8d still paid for.

### 3.4 8f — Q5_K + Q5_1 kernels (`b959ec9`)

Both f32-activation matmuls mirror the Q4_0/Q4_K structures; Q5_K decodes the
transposed qh (bit s of byte l) and deinterleaved qs chunks, with sub-level
tail masking for partial last super-blocks (0.5B id = 896 = 3.5×256; dispatch
requires `id % 32 == 0`). `embed_rows_q5_1/_q5_k` cover the q5_1 token
embedding. Gates updated in BOTH qwen2 and qwen3 `weights_on_cuda`. Q5_0 was
deferred at the time (no cached model needed it) — until 8q.

### 3.5 8l — the llama.cpp parity benchmark (`acca28f`)

Cross-benchmark vs llama.cpp `ca3d5a3e1` (build 10665), same GGUF files, GB10,
`-ngl 99 -t 8`, llama-bench `-r 3` (FA 0/1 matrix) + llama-cli cross-check;
minfer side 3 reps `--greedy`.

**Found + fixed the Q5_K registration gap first**: the CUDA whitelist in
`models/qwen2/loader.rs` (added in 7c) was missing `TensorType::Q5_K` (Metal's
list had it). Every Q5_K matmul silently ran on CPU with per-token GPU↔CPU
copies — 0.5B q5_k_m decoded at 51.6 tok/s behind `CUDA GATE: ...` spam. A
one-line fix → 246.3 tok/s (4.8×).

| model | llama fa0 | llama fa1 | minfer | gap (fa1) |
|---|---:|---:|---:|---:|
| 0.5B q4_0   | 417.6 | 453.5 | 258.0 | 1.76× |
| 0.5B q5_k_m | 311.0 | 394.6 | 246.3 | 1.60× |
| 0.6B q8_0   | 273.8 | 290.3 | 197.8 | 1.47× |
| 7B q4_k_m   | 46.4  | 47.1  | 41.2  | 1.15× |

7B @2K context: llama 44.9 (llama-cli) / ~45.1 (bench) vs minfer 31.3 → 1.43×.
Depth penalty: llama −5% vs minfer −24% — the attention/KV path loses
~3 ms/token at 2K. **Prefill gap: 110×** (7B q4_k_m: 3401 vs 30.7 tok/s), 69×
(0.6B q8_0), ~18–30× (0.5B). Root cause: minfer's quantized prefill reused the
decode-shaped kernels with `grid.y = nt` — every token block re-streamed the
full weight matrix (7B: ≈4.4 GB × 1920 tok ÷ 62 s ≈ 135 GB/s of pure
redundant traffic). Decode-gap attribution: both engines bandwidth-limited at
7B (llama ≈221 GB/s ≈ 81% of the 273 GB/s peak, minfer ≈193 ≈ 71%); the
residual 15% + the small-model 1.5–1.8× are per-token overhead (graph replay
is worth +24% on 0.5B: 208 → 258 with `MINFER_NO_CUDA_GRAPH=1`) plus llama's
FA-style single-pass decode attention vs minfer's multi-pass scores kernel.
The ranked follow-ups became the later campaigns: ① prefill tiled int8 GEMM
(→ R1 + 8m–8p + r5–r60), ② FA-style decode attention (→ 8d/R4/D3a), ③
per-token CPU overhead audit (→ R3).

### 3.6 8q — Q5_0 CUDA enablement + the misaligned-load fix (`9f419f9`)

0.5B q4_k_m GGUFs carry Q5_0 weights (token_embd, per-layer attn_q/k/v/o,
ffn_gate/up; attn_v q8_0, ffn_down q6_K, output q8_0), but the CUDA
participation whitelist (`models/{qwen2,qwen3}/graph.rs`) predated Q5_0, so
the whole model silently fell back to CPU (148.7 tok/s prefill) behind
per-token `CUDA GATE: a matmul weight has an unsupported type` spam.

Enablement (type-symmetric, no dispatch changes): gate whitelists extended
with Q5_0 via `matmul_t_ok`/`embed_t_ok` helpers + a diagnostic that names the
offending weight, type, and reason; `cuda.rs` gains the `embed_rows_on_gpu`
arm (type_id 6) and a `launch_q5_0_f32_matmul` dispatch arm; `cuda_kernels.cu`
gains `embed_rows_q5_0` and `q5_0_f32_matmul` (f32-activation legacy
structure, warp-per-4-rows). Prefill GEMM/MMQ reuse the existing
type-agnostic path. **The fault that mattered**: the first E2E run died in a
sticky `cudaErrorMisalignedAddress` (716) cascade — allocs/launches/syncs
failing from the first prefill matmul onward, then a null-device-pointer
panic at decode. Root cause: the Q5_0 block is 22 bytes, so the `qh` word at
block offset 2 is NOT 4-byte aligned for even block indices. Fix: both
kernels load `qh` as two 2-byte-aligned `uint16_t` loads; no shared kernel
touched. (Same root-cause family as 8p's latent `dequant_q5_0_f16` fix — doc
05.)

## 4. Verification

- **8b**: f16 roundtrip parity vs a half-rounded-KV reference (1e-4, kernel
  isolated from quantization noise); 7B 96-token greedy text identical f16 vs
  f32. cuda 154/0 after 8d.
- **8c**: `cuda_q4_0_prefill_q8_0_gemm_parity` (nt>1 vs the kernel's exact
  math; nt=1 f32 path vs hand dequant); `cuda_matmul_parity`'s q4_0 arm mirrors
  activation quantization; greedy text unchanged.
- **8d**: standalone A/B nkv 440 +49% / 2000 +64% / 8000 +80% (maxdiff 1e-8);
  parity vs `cpu_gqa_attn` 1e-4 (empty + partial splits, both KV layouts);
  7B E2E @2K greedy identical; @440 neutral (3 pairs within noise).
- **8f**: parity test (q5_1 id 64; q5_K id 896 tail, decode-formula weights
  over real `unpack_q4k_scales`, 5e-3); 0.5B q5_k_m E2E greedy identical to
  CPU; cuda 155/0.
- **8l**: llama-bench `-r 3` × FA 0/1 + llama-cli cross-check on the llama
  side; minfer 3 reps `--greedy`; same GGUF files both sides.
- **8q**: `cuda_q5_0_realshape_isolation` (new device test — embed
  shape-bisect + legacy/f16/MMQ/decode matmuls at the model's exact shapes,
  with a real `state.sync()` after each step because `Backend::synchronize`
  does not wait on the stream outside capture windows); full suite 173;
  E2E clean 3/3 runs.

## 5. Results

| item | headline number |
|---|---|
| 8b KV f16 | 7B @2K decode +~11% (swap-order pairs 10.3/10.2 vs 9.4/9.0 tok/s); win grows with context |
| 8c shaped Q8_0 GEMM | 0.5B prefill @3.6K 1005 → 1246 tok/s (+24%); −63% negative at 7B ffn_down kept out by the gate |
| 8d split-K attention | 7B E2E @2K decode 10.1 → 13.7 tok/s (+36%); @440 neutral |
| 8f Q5_K/Q5_1 | 0.5B q5_k_m admitted to CUDA (was CPU wholesale), greedy identical |
| 8l parity baseline | decode 1.15–1.76×, prefill 18–110× vs llama — the campaign's target sheet |
| 8q Q5_0 | 0.5B q4_k_m prefill 148.7 → ~1200 tok/s, decode 56.9 → ~306 tok/s (CPU fallback eliminated) |

## 6. Lessons

1. **Measure first, then wire (8c)**: the shape gate is load-bearing — the
   same kernel is +38–44% at one shape and −63% at another.
2. **Coverage is performance (8f/8q)**: a missing weight-type kernel silently
   costs a whole model its GPU; the "CUDA GATE" spam is the tell.
3. **Benchmark the competitor before optimizing (8l)**: per-shape gap numbers
   rank levers better than any profile of your own code.
4. **Respect the alignment contract (8q)**: `uint32_t` loads on non-4-aligned
   offsets are illegal and fault nondeterministically on unified memory —
   small-shape tests passing proves nothing; test the REAL shapes.
5. **Attention structure evolves in generations (8d → R4 → D3a/D3-4)**: 8d's
   split-K removed the warp-count bottleneck but kept a local-memory
   accumulator; each rewrite removed the bottleneck the previous one created.

← 78 · [Index](./README.md)
