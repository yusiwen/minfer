# 82 · small-M dispatch fix — multi-token MMVQ + token-looped legacy kernels (LANDED)

> **Result**: 7B q4_k_m batched decode nt=3 105.9 → 29.4 ms (3.60×), nt=8 279.3 → 48.6 ms (5.75×); weight traffic is now independent of nt (marginal cost 34.4 → 4.3 ms/token); batched forwards are bitwise-equal to serial nt=1 runs on all 8 quant types. The pre-registered amortization bar (2.5× at nt=3) was missed at 1.87× — a cost-model error, dissected in §5.
> **Commit**: `65ef00f` (code + docs) **Date**: 2026-09-11

## 1. Background — where things stood

Doc 81 closed the D5 speculative-decoding campaign by its pre-registered gate: the
D5-1a instrument (`minfer specverify`) measured the D5-0 gate end-to-end and found
C_T(3) = 106.0 ms against the ≤ 22.1 ms the speculative economics required. The
root cause was identified and recorded as the **small-M dispatch hole**: in
`matmul_f32_ptr_layout` (src/cuda.rs), `nt >= 16` took the tiled MMQ GEMM and
`nt == 1` took the MMVQ decode kernels, so **nt = 2–15 fell through to the legacy
f32 kernels**, whose launch grid uses the token count as `grid.y` — one full
weight re-stream per token, measured at ~34.9 ms/token flat on the 7B (4.36 GiB /
125 GB/s ≈ 34.9).

The hole matters beyond speculative decoding: every short-prompt prefill (CLI,
server), every multi-turn incremental prefill (`--cnv` delta prefill is a stream
of nt ≈ 1–15 batches), and any future multi-token feature (beam search, parallel
sampling, MTP (Multi-Token Prediction)) pays the per-token weight re-stream. This
step fixes the hole itself. It is explicitly **not** a D5 reopening: the doc 81
gate arithmetic is final, and §5 shows the D5 gate still fails after the fix.

A second, subtler motivation: the engine invariant "a batched step never costs
more than the equivalent serial steps" was false for every quant type. Pre-fix
amortization (nt·C_T(1)/C_T(nt)) at nt = 3 was 0.52× on the 7B — batching
**lost** to serial by ~2×.

## 2. Principle — the GPU mechanism

In the old decode kernels the token index was a grid dimension:

```c
dim3 grid(od_blocks, nt, 1);   // every (row-block, token) pair = one block
```

Each block streams its weight rows from DRAM, dots them against ONE activation
row, and exits. The weight bytes are re-read nt times. At 7B shapes the re-read
is a DRAM-level restream: the per-pass weight set (4.36 GiB) is far beyond any
cache, so nt = 3 costs 3 × 4.36 GiB ≈ 56 GB ≈ 3 × 18.5 ms of pure weight
traffic at the ~238 GB/s the decode kernels sustain.

llama.cpp's `mul_mat_vec_q` (recorded in LLAMA-CPP-MMQ-ANALYSIS.md §12) uses the
opposite structure: the token loop lives **inside the block**, so each weight
byte is loaded once and dotted against up to 8 activation rows (`tmp[ncols_dst]
[rows_per_cuda_block]`). The same idea applies at two levels here:

- **Register-level reuse** (K-quant MMVQ multi kernels): the weight bytes a
  thread loads per sub-block stay in registers while an inner `for t` loop dots
  them against nt token rows. Weight traffic = nt = 1 exactly; the dequant
  arithmetic (nibble extraction, scale fetch) also amortizes.
- **L1-level reuse** (legacy f32 kernels + the 8c q4_0 q8-GEMM): the per-token
  body is wrapped in an in-block `for t` loop, so the block's weight rows are
  re-read across tokens from L1 (a 7B ffn_down row group is 4 rows ≈ 42 KB,
  trivially L1-resident). DRAM still sees one stream.

Why not the other candidate? Padding nt ≤ 8 batches into the tiled GEMM's M-tile
(gate 16 → 2) streams weights once but at the GEMM's ~1/3.1 weight-stream
efficiency: 3 × 18.33 / 56.7 = 0.97× — it computes *more slowly than running the
three tokens serially*. Only the token-loop structure wins, because MMVQ's dp4a
(int8 dot) path makes extra tokens nearly free on the bandwidth axis.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

Two mechanisms, split by kernel family:

- **K-quants (q4_K / q5_K / q6_K) — new `_multi` kernels for nt ∈ [2, 8]**,
  appended beside the v1/v2 decode kernels. The single-token kernels (including
  the D3b/D4-4 pf/pf_dpl latency variants) are untouched: the nt = 1 decode hot
  path and its capture graphs keep their exact code, occupancy tuning, and
  bitwise behavior. The multi kernels hold a fixed `float acc[8]` with a
  uniform `t < nt` guard under `#pragma unroll`, so every accumulator index
  stays compile-time and nothing spills to local memory.
- **Legacy f32 kernels (q4_0 / q4_1 / q5_0 / q5_1 / q8_0 / f32-vec) and the 8c
  q4_0×q8-GEMM — in-place token wrap.** These kernels serve nt = 1 for the
  non-K-quant types, so adding separate multi variants would double the
  maintenance for no benefit; instead the per-token body moved inside a
  `for (t = 0; t < nt; ++t)` loop and `grid.y` collapsed to 1. At nt = 1 the
  loop runs once with the exact original op order — bitwise — and the launch
  parameters (including nt) are baked per captured graph, so capture semantics
  are unchanged.

Dispatch changes in `matmul_f32_ptr_layout`:

- `nt >= 16` GEMM gate → **`nt >= 9`** (batches of 9–15 take the tiled GEMM;
  padding waste is acceptable there and the MMQ M-tile floor makes smaller
  batches uneconomical — 0.97× per §2).
- q4_K / q5_K / q6_K arms: `nt == 1` → existing MMVQ (shape gates preserved);
  `2 ≤ nt ≤ 8` → the new multi kernels. The nt == 1 shape gates (id ≥ 2048 /
  od·id ≥ 24M / ≥ 4M) do **not** apply at nt ≥ 2: a weights-once kernel beats a
  per-token kernel at any shape once the token loop amortizes the uncoalesced
  load latency.
- The non-K-quant arms are unchanged in Rust — their kernels now handle
  multi-token internally.

Out of scope, unchanged: `Op::FusedQKV`/`FusedFFN` fusion (stays `nt == 1`
gated; epilogue fusion into MMVQ is a later optimization), and CUDA graph
capture for nt > 1 (eager nt ≤ 8 costs only +1.2 ms, measured in doc 80). The
q8 activation scratch needed **no** change: `decode_quantize_native` was already
nt-parameterized (`need = nt·(id/32)·40`), and `quantize_q8_0_pad40` already
quantizes all nt rows per launch.

### 3.2 Key code

The multi-token MMVQ core (q4_K v1 variant; the v2/v5/v6 variants follow the
same shape):

```c
float acc[8] = {0.0f, ..., 0.0f};            // fixed lanes, compile-time indices
for (int u = threadIdx.x; u < nsub; u += 256) {
    // ... weight-block loads + scale dequant: ONCE per sub-block ...
    const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 16 + (sub >> 1) * 32);
    #pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < nt) {                         // uniform guard, no divergence
            const uint8_t* x8 = acts8 + ((size_t)t * nsub + (size_t)u) * Q8PB;
            // ... dp4a dot of qw against token t's q8 sub-block ...
            acc[t] += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
        }
    }
}
mmvq_block_reduce_multi(acc, output, od, nt);  // per-token shuffle + warp_sums reduce
```

The dispatch arm (q4_K; q5_K/q6_K mirror it with their env gates):

```rust
if nt == 1 && id >= 2048 && id % 32 == 0 {
    self.q4_k_decode_mmvq(wptr, x, out, od, id, nt);        // unchanged hot path
} else if nt >= 2 && nt <= 8 && id % 32 == 0 {
    self.q4_k_decode_mmvq_multi(wptr, x, out, od, id, nt);  // Step 82
} else {
    launch!(launch_q4_k_f32_matmul)                          // nt == 1 small shapes only
}
```

### 3.3 Pitfalls

- **C++ name mangling**: the new launchers were appended at the end of
  `cuda_kernels.cu`, outside the file's `extern "C" { ... }` launcher blocks —
  nvcc emitted `_Z25launch_...` symbols and Rust's `extern "C"` declarations
  failed to link. Fixed by wrapping the new launcher section in its own
  `extern "C" { }`.
- **q6_K v2 loop bound**: the q4_K/q5_K v2 kernels iterate 8 nibble words
  (`ws[8]`), but the q6_K v2 kernel iterates **4** (`qls[8]`/`qhs[8]` word
  pairs: `qls[v]` with `qls[v + 4]`). The first multi-token build transcribed
  `v < 8` for q6_K — out-of-bounds reads and double-counted elements. The
  bitwise probe caught it on the first run (§4).
- **Parity-test references**: two existing tests drove K-quant matmuls at
  nt = 3 with an f32-activation reference — valid when nt = 3 hit the legacy
  f32 kernel, stale after the dispatch change. Their references now use the
  pad40 q8 round-trip (the semantics of the kernels the dispatch actually
  selects), with the mmvq tests' 1e-2 relative tolerance.

## 4. Verification

- **Bitwise probe** (`cuda_multi_token_matmul_bitwise`, new): one batched
  nt = 3 forward vs three nt = 1 forwards over identical weight bytes and
  activations, compared by exact f32 bit patterns across 12 dispatch cases —
  q4_K v2 (id 3584) and v1 (id 3904, partial tail super-block), q5_K v2/v1 at
  od·id ≥ 24M (so nt = 1 also rides MMVQ and the comparison stays
  same-family), q6_K padded-224B and raw-210B, q8_0, q4_0 with id > 8192,
  q4_1, q5_0, q5_1, f32. This defends against any per-(row, token) op-order
  change introduced by the token loops — and it caught the q6_K loop-bound
  bug immediately. The 8c q4_0×q8-GEMM arm (nt > 1 only, so it has no same-
  family nt = 1 sibling) is checked against an independent host dequant
  reference at the standard 1e-2 relative tolerance.
- **Updated parity tests**: `cuda_kquant_matmul_parity` and
  `cuda_q5_matmul_parity` (q8-activation references, §3.3).
- **Full suite**: 175 passed / 0 failed (`cargo test --release --features cuda`).
- **specverify pre/post matrix**: 4 models × nt 1..8, `-p 512 -r 40`, two
  passes (forward + reversed) per point; the 7B also has the doc 81 pre-fix
  curve as an independent anchor (C_T(1) 18.33/18.35, C_T(3) 106.0/105.9 — the
  two pre-fix builds agree within noise).
- **Bench regression** (tg128 / pp512, §5): defends the untouched nt = 1 decode
  path and the retimed ≥ 9 GEMM gate.

## 5. Results

7B q4_k_m (`specverify -p 512 -r 40`, median of 2 × 40 reps, ms per batched
decode step):

| nt | pre-fix | post-fix | speedup | amort. pre | amort. post |
|---:|--------:|---------:|--------:|-----------:|------------:|
| 1  | 18.36   | 18.35    | 1.00×   | 1.00×      | 1.00×       |
| 2  | 72.78   | 27.08    | 2.69×   | 0.50×      | 1.36×       |
| 3  | 105.91  | 29.44    | 3.60×   | 0.52×      | 1.87×       |
| 4  | 140.48  | 32.70    | 4.30×   | 0.52×      | 2.24×       |
| 5  | 174.25  | 36.47    | 4.78×   | 0.53×      | 2.52×       |
| 6  | 207.77  | 41.44    | 5.01×   | 0.53×      | 2.66×       |
| 7  | 244.74  | 45.22    | 5.41×   | 0.53×      | 2.84×       |
| 8  | 279.27  | 48.57    | 5.75×   | 0.53×      | 3.02×       |

The marginal cost per extra token collapsed from ~34.4 ms (a full weight
re-stream) to ~4.3 ms — that residue is real per-token work (GQA (Grouped-Query
Attention) reads, q8 activation quantize, the nt × dp4a compute), no longer
bandwidth. The batching invariant holds for every nt ≥ 3 (amortization > 1×,
and 1.36× even at nt = 2).

The three smaller models gain less, and the mechanism is instructive:

| model | nt=3 pre→post (ms) | speedup | amort.(3) pre→post |
|---|---|---:|---|
| Qwen2.5-0.5B q4_0 | 7.29 → 7.14 | 1.02× | 1.17× → 1.19× |
| Qwen2.5-0.5B q5_k_m | 8.54 → 7.18 | 1.19× | 1.06× → 1.27× |
| Qwen3-0.6B q8_0 | 12.22 → 12.44 | 0.98× | 1.00× → 1.00× |

For these models the whole per-layer weight matrix already fits in L2, so the
old `grid.y = nt` re-reads were L2-buffered, not DRAM restreams — the bug was a
cache-tax, not a bandwidth bug, and the fix removes only the L2 → SM re-reads.
The catastrophic regime is specific to large models, where the per-pass weight
set exceeds L2 by orders of magnitude.

**Pre-registered bar: MISSED, root cause recorded.** The plan bar was
amortization(nt=3) ≥ 2.5× (predicted ~2.9×); measured 1.87×. The bar was
mis-derived: it took C_T(3) ≈ C_T(1) + ε, treating the per-token marginal as
negligible, when the correct model is C_T(nt) ≈ weights (≈ 15 ms) +
nt × token-work (≈ 4.3 ms). The 2.5× figure was in fact the doc 81 D5
spec-economics threshold (C_T(3) ≤ 22.1 ms) smuggled in as a fix bar. The
fix's actual goal — weight traffic independent of nt — is fully achieved, and
the pre-fix amortization regression (0.52×) is gone. D5 stays closed:
C_T(3) = 29.4 ms > 22.1 ms required. (⚠️ see the doc 81 §4.3 errata, same
day: the 22.1 threshold traced to a cost model and an external anchor that
were both later invalidated — corrected economics put post-fix minfer at
≈1.42× for d=2 on the 14B; the closure verdict is under campaign review.)

Regression: pp512 4075.96 ± 52.96 → 4072.79 ± 62.43 t/s (−0.08%, noise; the
retimed ≥ 9 GEMM gate does not touch nt ≥ 16 prefill), tg128 51.86 ± 2.93 →
53.60 ± 0.03 t/s (the nt = 1 path is untouched; the pre-fix spread was run
noise).

Artifacts: `/tmp/d82/*.json`, `/tmp/d82/minfer-pre`, `/tmp/d82/minfer-post`
(ephemeral; key numbers inlined above).

## 6. Lessons

- A pre-registered bar must be derived from the mechanism it tests: the
  amortization ceiling is C_T(1)/marginal, and the marginal includes attention,
  quantize, and dp4a compute — not just weight bytes.
- `grid.y = token` is a DRAM bug only when the per-pass weight set exceeds L2;
  below that it degrades into a small L2 re-read tax, and the same fix gains
  almost nothing.
- A fixed-size accumulator array with a uniform `t < nt` guard under
  `#pragma unroll` keeps all indexing compile-time — llama.cpp's
  `mul_mat_vec_q` structure transfers directly, and nt = 1 can stay on the
  original hand-tuned kernels.
- Bitwise batched-vs-serial probes are the sharpest guard for token-loop
  refactors: they catch a wrong loop bound (q6_K's `v < 4`, not `v < 8`) on the
  first run, where tolerance-based tests would pass.

← 81 · [Index](./README.md)
