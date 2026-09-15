# 04 · Reading minfer's kernels II — from GEMV to tiled GEMM

> **Part**: Part 3b — the matmul ladder. **Prereq**: [chapter 03](03-kernels-elementwise.md)
> (the dispatch chain, quant block layout, index formulas — this chapter uses all three).
> **Code**: `src/cuda_kernels.cu`, `src/graph/cuda_backend.rs` — all `file:line` citations
> verified against the tree at writing time (the function name is the stable address, the
> line number a convenience).

## 1. Background — where this sits

Chapter 03 taught you the reading method on the easiest kernels in the file.
This chapter reads the family that earns the GPU its keep — the **matmuls** —
the heart of the tutorial for a simple reason: in a transformer forward
pass, the matmuls are where nearly all the floating-point work and nearly
all the weight traffic happen. Every other kernel exists to feed them or
clean up after them.

The chapter is a ladder with five rungs, in the order the code itself climbed
during the optimization campaign:

1. a **scalar GEMV** (General Matrix-Vector multiply — one output = one
   dot product of a weight row with the activation vector);
2. a **vectorized GEMV**, same math, 16-byte loads;
3. the **quantized decode path** — MMVQ (matrix-vector quantized), integer
   dots over packed 4-bit weights, which is what a real decode step runs;
4. the **tiled f16 GEMM** (General Matrix-Multiply) for prefill — tiling in
   three layers, `__syncthreads()` between them, `wmma` tensor-core
   fragments inside;
5. a **pointer to the int8 MMQ** (matrix-matrix quantized) prefill GEMM —
   the current default — whose anatomy lives in the reference docs.

One word before we start, because it organizes everything: **`nt`** is the
number of token rows the matmul processes — 1 during decode, hundreds during
prefill. Almost every dispatch decision you are about to see is a function
of `nt`, and §4 turns that into arithmetic: at `nt == 1` the matmul
degenerates into a GEMV that is memory-bound no matter how you code it; at
large `nt` each weight byte is reused so many times that the same silicon
becomes compute-bound. The ladder exists because no single kernel wins at
both ends.

## 2. Principle — the concepts

### 2.1 The matmul corner of the kernel inventory

The forensics protocol starts with enumeration. Chapter 03 counted **89
`__global__` kernels** in `src/cuda_kernels.cu` (8,386 lines — re-run the
grep to confirm before citing):

```bash
grep -n '__global__' src/cuda_kernels.cu
```

Of those 89, **42 are matmul-family kernels** (names matching
`matmul`/`mmvq`/`gemm`/`mmq`). You navigate them by regime:

| Family | Examples (first hit line) | What it does | Where taught |
|---|---|---|---|
| f32-activation matvec | `f32_f32_matmul_vec` :2177, `f32_f32_matmul_scalar` :2233, `q4_0_f32_matmul` :124, `q6_k_f32_matmul_padded` :1917 | dot products against f32 activations, per weight type | **this chapter** (§3.1–3.2) |
| Decode MMVQ (int8 dots) | `q4_k_q8_mmvq` :1254, `q6_k_q8_mmvq` :1339, `q4_0_q8_mmvq` :8091, `q8_0_p32_q8_mmvq` :8292 (+ `_v2`/`_multi`/`_pf` variants) | one weight row per 256-thread block, `__dp4a` over quantized activations | **this chapter** (§3.3–3.4) |
| Prefill GEMM (f16) | `gemm_f16_nt_kernel_t` :4787, `gemm_qb_nt_kernel` :5350 | tiled tensor-core GEMM over dequantized f16 weights | **this chapter** (§3.5) |
| Prefill MMQ (int8) | `mmq_nt_kernel` :5663, `mmq_raw_nb_kernel` :6376, `mmq_raw_nb_bt_kernel` :6656, `mmq_raw_nb_bt_q6k_kernel` :6976, `mmq_ksplit_reduce_kernel` :7316 | tiled int8 tensor-core GEMM, raw weight bytes | **this chapter** (§3.6 — pointer only) |
| Activation quantize | `quantize_q8_0_pad40` :733, `quantize_q8_0_pad40_t` :794, `quantize_q8_0` :2252 | f32 activations → 40-byte q8 blocks for the quantized paths | **this chapter** (§3.3–3.6) |

Three things the table does not show:

1. **The `nt` regimes share one dispatch function.** Rust-side, one place
   decides GEMV-vs-GEMM: `CudaState::matmul_f32_ptr_layout`
   (`cuda.rs:2725`). Every MatMul-shaped node goes through it — the
   plain `Op::MatMul` arm (`src/graph/cuda_backend.rs:931`) *and* the
   decode-fused `Op::FusedQKV` concat matmul (`cuda_backend.rs:877`).
2. **The v2/multi/pf suffixes are variants, not new algorithms.**
   `q4_k_q8_mmvq_v2` (:1447) is the same dp4a structure reorganized for
   16-byte weight loads; `_multi` adds an in-block token loop for `nt` 2–8;
   `_pf` pipelines tall rows. Read one, you have read the family's skeleton.
3. **Two quantize kernels serve two layouts.** `quantize_q8_0_pad40` (:733)
   writes token-major q8 blocks (the GEMV/MMVQ layout);
   `quantize_q8_0_pad40_t` (:794) writes the *transposed, swizzled* layout
   the MMQ GEMM stages (§3.6). Same math — max, scale, round — different
   destination addresses.

### 2.2 GEMV: the shape decode asks for

During decode, `nt == 1`: the activation side of the matmul is a single
vector `y[id]`, and the op is `out[r] = Σ_i W[r][i] · y[i]` for
`r = 0 .. od-1`, with the weight matrix in minfer's memory convention —
metadata `[in, out]`, memory row-major `[out][in]` (AGENTS rule 4) — so
output `r` reads exactly weight row `r`, which is `id` contiguous values.
The two layout facts from chapter 03 are load-bearing here: activations are
token-major `[nt][id]`, so at `nt == 1` the vector `y` is one contiguous run
that every thread reads (cached, DRAM pays once); and weight rows are
contiguous, so a thread or block assigned one output row streams memory
linearly — TECH-PRIMER §5.2's coalescing story applies for free.

The thread-mapping question ("what does one thread do?") has two natural
answers for a GEMV, and minfer ships both:

- **one thread per output element** — the thread owns `out[r]` and loops
  the dot product serially. Simple, and the weight reads are perfectly
  coalesced across threads (consecutive `r` = consecutive rows), but each
  thread does `id` dependent multiply-adds with no help from its neighbors.
- **one block per output row** — 256 threads split the row's `id` elements,
  each computing a partial sum, then a two-stage reduction (warp shuffle,
  then shared memory across warps) produces the output. Shorter serial
  chains — and the real prize: it is the shape the *quantized* decode
  kernels need, because one 256-thread block can also own the decoding of a
  whole row's packed blocks (§3.4).

### 2.3 Why GEMM wants tiles: the reuse arithmetic

At prefill, `nt` is the prompt length (say 512). The GEMM is

```
C[nt][od] = A[nt][id] · B[od][id]ᵀ
```

and the naive loop order ("for each output element, dot one row with one
column") re-reads the same weight bytes once per token: `nt` × the whole
matrix. The fix is **tiling** — process the output in small rectangles so
every tile of `B` loaded once serves *all* the tokens in the tile of `A`.
Count it for the 0.5B down-projection `[od=896, id=4864]` at `nt = 512`
(dims: `docs/inference_e2e_walkthrough/05-graph-builder-ir.md:162`):

- **without reuse**: each weight byte is read `nt` times → 2.45 MB (Q4_0) ×
  512 ≈ 1.25 GB of traffic for one layer, one forward.
- **with a 64-token tile**: each weight byte is read `nt / 64 = 8` times
  (once per token-tile) → ≈ 19.6 MB.

That ratio — traffic divided by `ceil(nt / tile)` — is the entire economic
argument for the prefill GEMM's complexity, and why `fa_prefill_f16kv`
(chapter 05) is shaped the same way: 64 query tokens share one K/V stream.

### 2.4 Tiling in three layers

"Tile" appears at three scales in a high-performance GEMM; name them now,
because chapter 06's technique catalog assumes you can:

1. **Block tile** — which rectangle of `C` one *block* owns. Set by the
   grid: `blockIdx.x` picks the token-tile, `blockIdx.y` the output-tile.
2. **Shared-memory staging** — the block cooperatively copies its A-tile and
   B-tile from global memory into shared memory (the on-chip scratchpad from
   chapter 01), because every thread will re-read those tiles many times.
   `__syncthreads()` — the barrier making every thread wait until all copies
   are done — separates "staging" from "compute".
3. **Register tile** — each *thread* accumulates a small sub-rectangle of
   the block tile in registers (or in `wmma` *fragments* — the tensor-core
   register layout, defined in chapter 05 §3.1). Registers are the only
   place an accumulator can live without paying memory traffic.

```text
C[nt][od]                       one BLOCK TILE = 64 tokens × TM outputs
 ┌────────────┬────────────┐    ┌──────────────────────────┐
 │ blk (0,0)  │ blk (0,1)  │    │  shared memory:          │
 ├────────────┼────────────┤    │   As = A-tile  64×KS f16 │  ← staged once,
 │ blk (1,0)  │ blk (1,1)  │    │   Bs = B-tile  TM×KS f16 │     reused by all
 └────────────┴────────────┘    │  registers:              │  64·TM threads
   blockIdx.x  blockIdx.y       │   fc[j][oc] fragments    │  ← accumulated
                                └──────────────────────────┘     per thread
```

The k-dimension (`id`) is *not* tiled in the grid — it is walked in chunks
of `KS` inside the block, double-buffered (stage the next chunk while the
current one computes). That loop nest is what §3.5 reads.

### 2.5 The quantized families: MMVQ vs MMQ

Both quantized families follow the same algebra — walkthrough doc 10 §2.4
if the pairing is not reflexive yet — but they sit at opposite ends of the
`nt` axis:

- **MMVQ** (decode, `nt == 1`): quantize the single activation row to int8
  once (a 40-byte-per-block "pad40" layout), then each weight block's packed
  nibbles form an integer dot with the int8 activation values via `__dp4a` —
  the SIMT instruction computing a 4-way 8-bit integer dot in one op; scales
  fold in at the end. One block per weight row, streaming the row once.
  Nothing is reusable at `nt == 1`, so the kernel's only job is converting
  bandwidth into outputs efficiently.
- **MMQ** (prefill, `nt ≥ 9`): the same int8 idea arranged as a *tiled GEMM*
  on the int8 **tensor cores** — `mma.m16n8k32.s8` multiplies 16×32 by
  32×8 int8 tiles in hardware. Weights are staged raw (never dequantized);
  activations quantized once per call in a prepass.

The reason decode does not want the tensor-core path: `mma.m16n8k32`
computes a 16-row output tile, so at `nt == 1` 15 of those 16 rows are
padding — the instruction's throughput is wasted, and the binding resource
is weight bytes anyway (walkthrough doc 15 §2.3). Prefill does not want MMVQ
because of the §2.3 table: one block per row re-reads every weight byte `nt`
times.

## 3. In minfer's code

### 3.1 `f32_f32_matmul_scalar` — the scalar GEMV

The plainest matmul in the file — the shape every later kernel improves on.
The kernel, `src/cuda_kernels.cu:2233`:

```c
// General-case fallback: one thread per (token, output) pair, scalar dot.
__global__ void f32_f32_matmul_scalar(
    const float* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nt * od) return;
    int t = (int)(idx / od), r = (int)(idx % od);
    const float* wr = weights + (size_t)r * id;
    const float* y = acts + (size_t)t * id;
    float acc = 0.0f;
    for (int i = 0; i < id; i++) acc += wr[i] * y[i];
    output[idx] = acc;
}
```

Line by line:

- **`idx` in `long long`, and the ceil-div guard** — the flat thread index
  over `nt * od` outputs, widened defensively (chapter 03's overflow habit);
  `if (idx >= nt * od) return;` is the usual grid-tail guard.
- **`t = idx / od, r = idx % od`** — decode the flat index back into
  (token, output). One thread owns one output element: the first of §2.2's
  two mappings.
- **`wr = weights + r * id`** — the weight row. Row-major `[out][in]` means
  output `r`'s inputs are `weights[r*id .. r*id+id]`, contiguous. Two
  adjacent threads (`r`, `r+1`) therefore read two adjacent rows — coalesced
  at the warp level.
- **`y = acts + t * id`** — the token's activation row. At `nt == 1` every
  thread reads the *same* `y` bytes; they hit L1/L2 and cost DRAM once.
- **the loop, and the store** — a serial scalar dot: `id` dependent
  multiply-adds, one accumulator, no unrolling, no vector loads. This is the
  baseline the rest of the ladder exists to beat. The store address *is*
  `idx`: `od` outputs per token are contiguous, so flat index = address.

The launcher (`src/cuda_kernels.cu:3676`) is where the ladder's first fork
lives — `id % 8 == 0` (the vector kernel's float4 alignment requirement)
routes to `f32_f32_matmul_vec` with `grid = ceil(od/8)`, 64-thread blocks;
otherwise the scalar kernel launches with the familiar
`grid = ceil(nt*od/256)` ceil-div. This fork is the *whole* F32 dispatch —
no quant gates, no MMVQ — and the scalar kernel is its general-case
fallback. For the quantized types, each has a sibling with the same GEMV
shape (`q4_0_f32_matmul` :124 reads packed nibbles but keeps f32
activations; `q6_k_f32_matmul_padded` :1917 is the padded-stride variant),
so the *shape* of this kernel is the GEMV shape for the whole
f32-activation family.

**Bytes moved — the number that decides everything.** Qwen2.5-0.5B has
`n_embd = 896` and FFN width 4864 (`docs/QWEN2-SUPPORT.md:79`;
`docs/inference_e2e_walkthrough/05-graph-builder-ir.md:162`). Two layers,
f32 weights, one decode token:

- attention `wo` `[896 out, 896 in]`: weights = 896·896·4 B = **3.21 MB**;
  activations 3.6 KB; outputs 3.6 KB. FLOPs = 2·896·896 ≈ 1.61 MFLOP.
- `ffn_down` `[896 out, 4864 in]`: weights = 896·4864·4 B = **17.4 MB**;
  FLOPs = 2·896·4864 ≈ 8.72 MFLOP.

At the GB10's documented ~273 GB/s (`docs/GLOSSARY.md:127`), `ffn_down`
cannot finish faster than 17.4 MB ÷ 273 GB/s ≈ **63.7 µs** — and the math it
must do (8.7 MFLOP) is ~136 GFLOP/s at that pace, *nothing* for a GPU. That
asymmetry is the whole story: a GEMV's arithmetic intensity — FLOPs per byte
moved — is fixed by the shape, not by how cleverly you code it, and at
2 FLOP per 4-byte element it is **0.5 FLOP/byte**. You cannot optimize a
scalar GEMV into being compute-bound; you can only (a) move fewer bytes —
quantize the weights (§3.4) — or (b) reuse bytes — add tokens (§3.5).

**CPU counterpart.** Walkthrough doc 10
([`10-cpu-matmul-kernels.md`](../inference_e2e_walkthrough/10-cpu-matmul-kernels.md))
is the CPU mirror of this whole ladder: the dequant-at-load bandwidth
argument (§2.1), why the activations go int8 too (§2.2), and the AVX2/NEON
implementations (§3). The GPU kernel above is the same scalar reference
executed once per thread instead of once per core — and doc 10's §2.1 is
the same ledger §4 keeps with device numbers: weights dominate, so their
bytes are the budget.

### 3.2 `f32_f32_matmul_vec` — the vectorized GEMV

Same math, three mechanical changes. The kernel header states the mapping
(`src/cuda_kernels.cu:2173`):

```c
// ─── F32 × F32 matmul (7e④) ───────────────────────────────────
// Same unit lane mapping as the q4_K kernel: lanes own (row, 256-elem
// chunk) pairs, float4 loads on both operands. Requires id % 8 == 0 for
// the aligned float4 loads; the scalar kernel covers the general case.
```

First change — **the constants and the mapping** (`cuda_kernels.cu:2183`):

```c
    const int NR0 = 4;
    const int NSG = 2;
    const int CHK = 256;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (r0 >= od) return;

    int nch = (id + CHK - 1) / CHK;
    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
        const float* y = acts + (size_t)t * id;
```

One 64-thread block (2 warps — the launcher's `block(64)`) owns `NSG × NR0 =
8` output rows: warp `w` takes rows `r0 .. r0+3`, and each *lane* (one of the
warp's 32 threads) will own a `(row, 256-element chunk)` pair. The grid is
`ceil(od / 8)` blocks — output-parallel only, no `grid.y = nt`.

Second change — **the inner loop is a unit-lane loop with float4 loads**
(`cuda_kernels.cu:2205`):

```c
        for (int u = lane_id; u < nch * NR0; u += WARP) {
            int ic = u % nch, rr = u / nch;
            const float* wr = weights + (size_t)(r0 + rr) * id + ic * CHK;
            const float* yc = y + ic * CHK;
            int len = min(CHK, id - ic * CHK);
            float p = 0.0f;
            // the unit's lane streams the WHOLE chunk (8 floats per pass)
            for (int i = 0; i < len; i += 8) {
                float4 a0 = *reinterpret_cast<const float4*>(wr + i);
                float4 a1 = *reinterpret_cast<const float4*>(wr + i + 4);
                float4 b0 = *reinterpret_cast<const float4*>(yc + i);
                float4 b1 = *reinterpret_cast<const float4*>(yc + i + 4);
                p += a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w
                   + a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
            }
            acc[rr] += p;
        }
```

- **`u = lane_id; u += WARP`** — the round-robin unit loop from §2.2's
  second mapping, one level down: a *warp* owns a row, its lanes split the
  row's 256-element chunks. For `id = 4864`: `nch = 19` chunks, 4 rows per
  warp → 76 units over 32 lanes ≈ 2.4 passes.
- **`float4`** — a 16-byte vector type; `reinterpret_cast<const float4*>`
  loads four consecutive f32 in **one 16-byte transaction** instead of four
  4-byte ones — hence the launcher's `id % 8 == 0` gate: each pass consumes
  two float4 pairs (8 elements), so misaligned `id` would fault.
- **`acc[rr] += p`** — each lane keeps one partial per row (`float acc[4]`,
  zeroed at :2201); chunk partials accumulate per lane, not globally.

Third change — **the reduction and the token loop** (`cuda_kernels.cu:2223-2227`):
per row, `warp_reduce_sum` — the butterfly shuffle from chapter 02 — folds
the 32 lane-partials into lane 0, which stores `output[t * od + r0 + rr]`.
The `for t` loop wraps *everything*, so one launch serves every token; the
comment calls out the bitwise constraint (the `nt == 1` order is exactly
preserved) and the reason it exists — the pre-Step-82 shape (`grid.y = nt`)
re-streamed the whole weight matrix once per token (campaign record:
`docs/CUDA_OPTIMIZATION.md` §0 row 82).

**Why 16-byte transactions matter (and why it is still slow).** DRAM talks
in bursts, not bytes: a 4-byte load still costs a 32-byte sector, and a
warp's scattered 4-byte loads can cost up to 8× the useful traffic. The
float4 load aligns each lane's request to 16 bytes, so a warp's 32 lanes
touch 32 × 16 = 512 contiguous bytes — fully dense, minimum transactions
(TECH-PRIMER §5.2's coalescing rule, at the widest general-purpose width).
But notice what did *not* change: the weight bytes per output, and therefore
the 0.5 FLOP/byte intensity. Vectorization makes the memory system run at
its best; it cannot change the shape's budget — on Qwen2.5-0.5B in f32,
`ffn_down` decode is still a ≥ 63.7 µs layer. The byte count only drops when
the weights themselves shrink — rung 3.

**CPU counterpart.** Doc 10 §3 does the same promotion in Rust: scalar loop
→ `#[target_feature(enable = "avx2")]` chunks with `_mm256_fmadd_ps` —
float4 ↔ 256-bit SIMD vectors, the warp reduction ↔ a horizontal add. Same
gain class, same ceiling.

### 3.3 The dispatch — one function decides the whole ladder

Here is the section to bookmark: every MatMul-shaped node reaches the same
match arm, and the arm hands everything to one function. The backend arm
(`src/graph/cuda_backend.rs:931`, abridged to the calls that matter):

```rust
Op::MatMul { transpose_b } => {
    if *transpose_b { return Err(...); }           // weights arrive row-major [out][in]
    ...
    // quant kernels address whole 32-element blocks … (id % 32 == 0 gate)
    if meta.weight_ttype != crate::tensor::TensorType::F32 && id % 32 != 0 {
        return Err(...);                           // GPU_SAFETY: Err, not fallback
    }
    ...
    self.state.matmul_f32_ptr_layout(
        wptr, meta.weight_ttype, self.ptr_of(in_bufs[0])?,
        self.ptr_of(out_buf)?, od, id, nt,
        self.state.is_weight_padded(&meta.weight_name),
    )?;
    if let Some(bname) = &meta.bias_name { ... }   // add_bias_f32, ch-03's epilogue
    Ok(())
}
```

The name parses as: matmul with **f32 activations** (`_f32`), raw
**pointers** (`_ptr` — no tensor objects cross the FFI), and a **layout**
flag (`_layout` — whether a Q6_K weight was registered with the padded
224-byte stride). The decision tree inside (`cuda.rs:2725`) has three
tiers:

**Tier 1 — prefill GEMM** (`cuda.rs:2754`): `nt >= 9` **and** `id % 32 == 0`
**and** a supported quant type → a tiled GEMM — `prefill_mmq` (int8,
default) or `prefill_gemm_f16` (the f16 escape, §3.5) depending on
`mmq_active()` (`cuda.rs:3027`: compute capability ≥ 8.0 and `MINFER_MMQ`
not `0`).

**Tier 2 — decode/small-batch per-type kernels**: everything else falls
through to a `match ttype` with per-type shape gates. The Q4_0 arm
(`cuda.rs:2836`):

```rust
} else if nt == 1 && id >= 2048 && id % 32 == 0 && !Self::no_q40_mmvq() {
    // doc 103: decode MMVQ … claims shapes that previously ran the f32 kernel
    self.q4_0_decode_mmvq(wptr, x, out, od, id, nt);
    Ok(())
} else if nt >= 2 && nt <= 8 && id > 8192 && id % 32 == 0 && !Self::no_q40_mmvq() {
    self.q4_0_decode_mmvq_multi(wptr, x, out, od, id, nt);
    Ok(())
} else {
    launch!(launch_q4_0_f32_matmul)
}
```

and the K-quant arms have the same skeleton with their own measured gates
(`cuda.rs:2886`–2870): Q4_K decode MMVQ at `id >= 2048`, Q5_K at
`od*id >= 24_000_000`, Q6_K at `od*id >= 4_000_000` — each a documented
crossover where the dp4a structure starts beating the f32-activation kernel,
each with an env opt-out for A/B. **Tier 3 — the F32 fallback**:
`launch_f32_f32_matmul` → vec or scalar by the `id % 8` fork of §3.1.

The one-line answer to "which kernel does decode's Matmul dispatch to?":

- **decode (`nt == 1`)**: `Op::MatMul` (`cuda_backend.rs:931`) →
  `matmul_f32_ptr_layout` (`cuda.rs:2725`) → per-type MMVQ — for Q4_0 with
  `id ≥ 2048`: `q4_0_decode_mmvq` (`cuda.rs:4708`) → `launch_q4_0_q8_mmvq`
  (`cuda_kernels.cu:8248`) → **`q4_0_q8_mmvq`** (`cuda_kernels.cu:8091`),
  after `decode_quantize_native` (`cuda.rs:3235`) has produced (or memoized,
  the MmqCache) the pad40 q8 activation plane via `quantize_q8_0_pad40`.
- **prefill (`nt ≥ 9`)**: the same arm → `mmq_active()` → `prefill_mmq`
  (`cuda.rs:3501`) → for Q4_K: transposed-A prepass
  `quantize_q8_0_pad40_t` (`cuda_kernels.cu:794`) then
  `launch_mmq_raw_nb_bt_nt` (`cuda_kernels.cu:7328`) →
  **`mmq_raw_nb_bt_kernel`** (`cuda_kernels.cu:6656`); Q6_K has its own BT
  kernel (:6976); older/fallback shapes land on `mmq_nt_kernel` (:5663).

**The walkthrough's "MMVQ decode", verified honestly.** The e2e walkthrough's
master table (doc 15 §2.2) says decode runs "per-type MMVQ + f32-activation
kernels" — exactly what the code says, but the *shape gates* decide which
matmuls take MMVQ on your model. On **Qwen2.5-0.5B** (`id = 896` for every
attention projection, gate and up; `id = 4864` only for `ffn_down`):
`ffn_down` (id 4864 ≥ 2048) → **MMVQ** (`q4_0_q8_mmvq` on a Q4_0 model);
everything else — QKV projections, `wo`, gate, up, `lm_head` (id 896) —
misses the gate and runs the **f32-activation kernel** `q4_0_f32_matmul`
(:124). On Qwen2.5-7B (`id = 3584` everywhere), every decode matmul clears
the gate and the whole step is MMVQ — plus the v2 variants, since
`mmvq_v2(id)` (`cuda.rs:5229`) additionally requires `id % 256 == 0`
(3584 = 256·14 ✓). The gate is not an oversight: the arms' comments record
the measured crossovers (small-`id` MMVQ loses — the uncoalesced nibble
loads dominate when rows are short, `cuda.rs:2916-2920`). The reading habit
this tutorial keeps hammering: **the master table gives the structure; the
gates give your model's truth.**

One more dispatch consumer: the decode fused path. Chapter 05 documented
`Op::FusedQKV` (`cuda_backend.rs:843`) as concat-matmul then
`attn_bias_rope_store`; the concat matmul inside it is the *same*
`matmul_f32_ptr_layout` call (`cuda_backend.rs:877`), so the fused node and
the plain `Op::MatMul` node make identical kernel choices at identical
shapes — the fusion is in the epilogue, not the matvec.

### 3.4 `q4_0_q8_mmvq` — the real decode path

Now the kernel your 0.5B Q4_0 `ffn_down` actually runs at `nt == 1`. First
its ingredients, then the code.

**The activation plane.** `decode_quantize_native` (`cuda.rs:3235`)
quantizes the one f32 activation row into the **pad40** layout — 40 bytes per
32-element block: 2-byte f16 scale, 2 bytes of padding, 32 int8 values at
offset 4, and a 4-byte int32 sum at offset 36 (`cuda_kernels.cu:729-732`
documents the layout; the sum feeds the *MMQ* min-term correction and is
"invisible" to MMVQ). The writer kernel is `quantize_q8_0_pad40` (:733),
one thread per block, tree-reduced amax — chapter 03's quantize family,
already read.

**The kernel** — `src/cuda_kernels.cu:8091`:

```c
__global__ void __launch_bounds__(256) q4_0_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nb = id >> 5; // dispatch gate: id % 32 == 0
    const int row_stride = nb * Q4B;
    const uint8_t* x8row = acts8 + (size_t)t * nb * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nb; u += 256) {
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)u * Q4B;
        const float d4 = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const uint8_t* x8b = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            // 18-B stride: payload is 2B-aligned only — two u16 halves/word
            const uint32_t w =
                (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v) |
                ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * v + 2) << 16);
            const uint32_t lo = w & 0x0F0F0F0F;           // elements 4v..4v+3
            const uint32_t hi = (w >> 4) & 0x0F0F0F0F;    // elements 16+4v..
            dot = __dp4a((int)lo, (int)xw[v], dot);
            dot = __dp4a((int)hi, (int)xw[v + 4], dot);
            sx  = __dp4a(0x01010101, (int)xw[v], sx);
            sx  = __dp4a(0x01010101, (int)xw[v + 4], sx);
        }
        // q4_0 value = (nibble - 8) * d  →  Σ = d * (dot - 8 * sx)
        acc += d8 * d4 * (float)(dot - 8 * sx);
    }
    mmvq_block_reduce(acc, output, od, t);
}
```

**What one thread processes: a slice of one weight row — 32-element blocks,
round-robin.** The launcher (`cuda_kernels.cu:8248`) is `grid(od, nt)` × 256
threads, so block `row` owns output element `out[t][row]` and its 256
threads split the row's `nb = id/32` quant blocks (`u = threadIdx.x; u +=
256`). For `ffn_down` 0.5B: `nb = 152`, so each thread handles exactly one
block and 104 threads idle — a tail you accept because the structure is
per-row on purpose. Line by line:

- **`row_stride = nb * Q4B`** — `Q4B` is 18 (`cuda_kernels.cu:10`), chapter
  03's Q4_0 block. One thread's block pointer is `row·2736 + u·18` — a
  stride-18 walk, the row streamed linearly.
- **`d4`, `d8`** — the two f16 scales: the weight block's and the
  activation block's (`Q8PB = 40` stride, `cuda_kernels.cu:679`).
- **the 2-byte-loads-as-u32 trick** — Q4_0's 18-byte stride guarantees only
  2-byte alignment (family comment at :8085-8086), so a direct `uint32_t`
  load would be a misaligned-access fault on some devices; the kernel
  assembles each 32-bit word from two `uint16_t` loads. "It compiles" is not
  "it is defined".
- **`lo` / `hi` masks** — the chapter-03 nibble convention as SIMD: one u32
  word holds four packed nibbles; `& 0x0F0F0F0F` extracts elements `4v..4v+3`
  (low nibbles), `>> 4` the elements `16+4v..` (high nibbles). Two `__dp4a`s
  per word compute 4-way int8 dots against the matching int8 activation words
  — eight multiply-adds in four instructions, in integer silicon.
- **`sx`** — the sum of the activation int8 values (dp4a of `0x01010101` × x
  accumulates x's bytes). Why: the stored nibble is `round(v/d) + 8`, so the
  true dot is `Σ (nibble−8)·x = dot − 8·sx` — the −8 correction for free,
  the same algebra as the CPU's `dot_q4_0_q8_0` (walkthrough doc 10 §2.4).
- **`acc += d8 * d4 * (dot - 8*sx)`** — both scales fold once per block, in
  f32. The integer accumulation is *exact*; the only rounding in the row is
  the two quantizations, which already happened.
- **`mmvq_block_reduce`** (`cuda_kernels.cu:1310`) — the two-stage reduction
  of §2.2's second mapping: 5 warp shuffles, `warp_sums[8]` in shared memory,
  `__syncthreads()`, thread 0 adds and stores `output[t*od + row]`.

**The K-quant sibling.** `q4_k_q8_mmvq` (:1254) has the same skeleton —
`grid(od, nt)`, 256 threads, round-robin units, dp4a, shared reduce — with
the Q4_K super-block decode inside the unit loop (:1268-1292):
`get_scale_min_k4` unpacking per-sub-block scale/min nibbles (:1274), one
dp4a pair per 4 bytes of nibbles (:1284-1290), and the two-term correction
`d8 · (s8·d·dot − m8·dm·sx)` (:1292) because Q4_K stores a per-sub-block
min. The v2 variant (:1447) reorganizes the same math for 16-byte `uint4`
weight loads (the R2 "weight-streaming" rework,
`docs/cuda_optimization_steps/09-r2-mmvq-weight-streaming.md`), and the
`_multi` variants (:7682+) wrap the unit loop in `for t` — the nt 2–8
regime. The differences are load widths and loop nesting, never dot algebra.

**Why int8 dots at all — the intensity arithmetic.** Rung 3's payoff on
§3.1's budget: same `ffn_down` layer, now Q4_0. Weight bytes: 896 rows ×
152 blocks × 18 B = **2.45 MB** (was 17.4 MB — 7.1× less traffic);
intensity: 2 FLOP per 0.5625-byte element = **3.56 FLOP/weight-byte** (was
0.5; `docs/GLOSSARY.md:126` rounds this to "~1 MAC per weight-byte →
bandwidth-bound"). The floor drops from 63.7 µs to 2.45 MB ÷ 273 GB/s ≈
**9.0 µs** — still memory-bound (you never escape that at `nt == 1`), but
7× lower. And the compute to fill that bandwidth — 8.7 MFLOP in 9 µs — is
trivial, which is precisely why integer dp4a silicon is *enough*.

**CPU counterpart.** Doc 10 §2.4 states the pairing rule and §3.2
implements it: `dot_q4_0_q8_0` consumes Q8_0 activation blocks with the
identical `(nibble−8)` correction and scale folding. The implementations
agree because the parity gates (chapter 06) compare them to the f64
reference — note the one *intentional* divergence: the CPU quantizes
activations to Q8_0's 34-byte blocks; the GPU here uses the 40-byte pad40
layout.

### 3.5 `gemm_f16_nt_kernel_t` — the prefill GEMM, tiled in three layers

The f16 prefill path is the tutorial's payoff for §2.4's theory: here is
where block tiles, shared-memory staging, and register tiles actually live.
It is also the *escape* path today (the int8 MMQ of §3.6 is the default),
but it is the right one to read first — smaller, and every idea transfers.

**How to get there.** `MINFER_MMQ=0` routes prefill to `prefill_gemm_f16`
(`cuda.rs:3844`), which obtains the weight as f16 (from the persistent
per-weight f16 cache — `w16_get`, `cuda.rs:3907`, dequantized once by
chapter 03's `dequant_q*_f16` — or by dequantizing into scratch on this
call), converts the f32 activations once (`launch_convert_f16`), and
launches the GEMM (`launch_gemm_f16`, `cuda_kernels.cu:5075`).

**The contract.** `C[nt, od] = A[nt, id] · B[od, id]ᵀ`. The header comment
is the design in six lines (`cuda_kernels.cu:4780-4785`):

```c
// C[nt, od] = A[nt, id] · B[od, id]^T. 64 x TM output tiles (TM = 64
// baseline, 128 halves the B-panel re-reads through L2 and the per-k-step
// barrier count), k-step 32, double-buffered shared staging, 8 warps (each
// owns 32 nt rows x TM/4 od cols as 2 x TM/64 f32 fragment pairs). f32
// accumulation. Tails: nt/od masked at store, k-tail zero-filled (id % 8
// == 0 keeps the uint4 chunk loads aligned).
```

**Layer 1 — block tiles.** The grid is
`dim3 grid((nt + 63) / 64, (od + TM_ - 1) / TM_)` (`cuda_kernels.cu:5098`,
`TM_ = 128` default per the `MINFER_GEMM_TM` selection at :5079-5086). Block
`(bx, by)` owns output rows `n0 = bx·64` (tokens) × `m0 = by·TM` (outputs).
The comment at :4809-4811 records why the *token* axis is `grid.x`:
consecutive blocks share one B panel (TM weight rows × id), so the L2 serves
the weight stream across blocks — the weight matrix streams from DRAM ~once
per forward instead of once per token-tile.

**Layer 2 — shared-memory staging, double-buffered.** The setup
(`cuda_kernels.cu:4796-4802`) carves one dynamic shared-memory allocation
into `As` (2 × 64×KS f16 — two buffers), `Bs` (2 × TM×KS f16), and `Cs` (a
per-warp staging area for the store). The k-loop (`cuda_kernels.cu:4877`):

```c
    for (int k = 0; k < id; k += KS, buf ^= 1) {
#if __CUDA_ARCH__ >= 800
        if (k + KS < id) {
            gemm_stage_ab<TM, KS, TN, AF32>(A, B, As, Bs, Am, buf ^ 1, n0,
                                            m0, k + KS, nt, od, id, tid);
            gemm_cp_commit();
            // wait until the CURRENT tile landed (one group may stay in flight)
            gemm_cp_wait1();
        } else {
            gemm_cp_wait0();
        }
        __syncthreads();
```

Read it as a pipeline, not a loop: at k-step `j`, the block issues `cp.async`
copies (the asynchronous global→shared copy path, TECH-PRIMER §5.3) for the
*next* tile into buffer `buf^1` while it still computes on buffer `buf`;
`gemm_cp_wait1` waits until only the current tile's copy group is
outstanding, and `__syncthreads()` makes the whole block's math wait for the
whole block's staging. `KS = 32` by default (`:5087-5094` — the KS=64
variant halves barrier count but its 56 KB shared appetite halves resident
blocks on GB10, measured −38%; the comment prices it). This is the
double-buffered staging of §2.4, verbatim.

**Layer 3 — register tiles as wmma fragments.** Each warp owns a 32-token ×
TM-output rectangle (warp `w`: `wm = w >> 1` picks the od chunk, `wn = w & 1`
the 32-row token half, :4805-4808). The compute step (`cuda_kernels.cu:4954`):

```c
#pragma unroll
        for (int kh = 0; kh < KHC; kh++) {
            wmma::load_matrix_sync(fa[0], &As[buf * TN * KS + wn * 32 * KS + kh * 32], KS);
            wmma::load_matrix_sync(fa[1], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32], KS);
            wmma::load_matrix_sync(fa[2], &As[buf * TN * KS + wn * 32 * KS + kh * 32 + 16], KS);
            wmma::load_matrix_sync(fa[3], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32 + 16], KS);
#pragma unroll
            for (int oc = 0; oc < ODC; oc++) {
                wmma::load_matrix_sync(fb[0], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32], KS);
                wmma::load_matrix_sync(fb[1], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32 + 16], KS);
                wmma::mma_sync(fc[0][oc], fa[0], fb[0], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[1], fb[0], fc[1][oc]);
                wmma::mma_sync(fc[0][oc], fa[2], fb[1], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[3], fb[1], fc[1][oc]);
            }
        }
```

Chapter 05 defined fragments and `mma_sync`; here note the *shape* of the
nest: per 32-wide k-slice, four A-fragments (two 16-token rows × two 16-wide
k-halves) multiply two B-fragments each, accumulating into `fc[j][oc]` —
registers for the *entire* k-loop. The trailing comment at :4950-4953 is a
fossil of a real bug ("the v1 bug: only the first 16 k's were multiplied") —
both k-halves must accumulate; fragment indexing bugs do not crash, they
silently halve your dot products (the parity gates catch them, chapter 06).

**The store.** After the k-loop, each warp spills its fragments through
`Cs` (shared) and writes out with bounds masks (`cuda_kernels.cu:4973-4986`):
`store_matrix_sync` lands the 16×16 fragment in shared memory, then lanes
copy the 256 values to global `C[n * od + m]` for in-range `(n, m)` — how
the kernel handles the ragged tail of a 30-token prompt without a second
code path. This is also why §4's table has a prefill row that looks nothing
like the decode rows: with 64 tokens per tile, each staged B byte is
consumed by 64 tokens' worth of fragments *before* the next k-tile is staged
— the reuse arithmetic of §2.3 made silicon.

**Bank conflicts, in one paragraph.** Shared memory is not one wide port:
the hardware splits it into 32 banks (lanes) of 4 bytes each, and a warp's
access is fast only when its 32 addresses hit 32 *distinct* banks; when
several lanes address the same bank — a **bank conflict** — the accesses
serialize into as many passes as there are colliding lanes. The classic
trigger is a shared row stride that is an exact multiple of the bank count
(32 floats = 128 bytes): every row's column 0 lands in bank 0, so a
column-wise read across rows collapses to one bank. The standard fix is
padding the stride by one bank's width — exactly what chapter 05's attention
kernel does with `sstr = hd + 8` (`cuda_kernels.cu:4168-4171`, its comment
is a worked example worth rereading now that you know the term). This GEMM
sidesteps the issue differently: its hot shared reads are `wmma::
load_matrix_sync` calls, and the fragment-load hardware handles the layout.
Background: TECH-PRIMER §5 (the memory-hierarchy table and coalescing rules
these bank arguments extend).

**CPU counterpart.** Doc 10 §3 is the honest mirror: the CPU cannot afford
per-thread fragments, so it tiles across cache lines and SIMD registers, and
its "shared memory" is L1/L2 with hardware coherence doing what
`__syncthreads()` does here. Layer 2 is where the architectures genuinely
diverge.

### 3.6 The int8 MMQ prefill — a pointer, not a tour

The default prefill path (`nt ≥ 9`, `mmq_active()`, `MINFER_MMQ` unset) is
the campaign's flagship: the int8 MMQ GEMM, promoted default-on at r60 after
measuring **1.080× vs llama.cpp on 7B Q4_K_M** (`cuda.rs:3006-3011`,
`docs/CUDA_OPTIMIZATION.md` P6). Its anatomy is a whole reference doc, and
the tutorial's policy is to link, not re-explain — but you should recognize
its pieces in a profile:

- **Activation prepass**: `quantize_q8_0_pad40_t` (`cuda_kernels.cu:794`)
  quantizes f32 activations to int8 *and writes them pre-transposed and
  swizzled* into the exact layout the GEMM stages (`cuda.rs:3556`;
  llama.cpp's `quantize_mmq_q8_1` design — "byte-identical … only
  reordered", :782-790).
- **The GEMM**: `mmq_raw_nb_bt_kernel` (`cuda_kernels.cu:6656`) — raw
  quantized weight bytes staged per tile, decoded in registers next to the
  `mma.m16n8k32.s8` instruction, per-k-block scale rescale, f32
  accumulation. The q4_K route enters at `cuda.rs:3714`
  (`launch_mmq_raw_nb_bt_nt`, `cuda_kernels.cu:7328`); Q6_K has its own BT
  kernel (:6976);
  non-BT-consumable shapes fall back to `mmq_nt_kernel` (:5663).
- **Split-K**: when the grid is M-starved (small `nt`), doc 92's auto
  ksplit (`cuda.rs:3694-3705`) slices the k-range across `grid.z` and
  `mmq_ksplit_reduce_kernel` (:7316) adds the partials — the same split-K
  family as decode attention (chapter 05 §2.4).

```c
template <int KDR, bool DSC>
__global__ void __launch_bounds__(256) mmq_raw_nb_bt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ W_dsc,
    const uint8_t* __restrict__ qa8g,
    const uint8_t* __restrict__ sdag, float* __restrict__ C,
    int nt, int od, int id, int nchunk,
    float* __restrict__ Cpart, int ksplit
) {
```

Nine lines on purpose: the parameters tell the story (raw weights `W`, a
pre-decoded scale plane `W_dsc`, the swizzled activation planes
`qa8g`/`sdag`, a partial-output buffer `Cpart` for split-K). The deep read —
staging, swizzles, fragments, the r34→r60 lever history — lives in
[`docs/LLAMA-CPP-MMQ-ANALYSIS.md`](../LLAMA-CPP-MMQ-ANALYSIS.md) (the
llama.cpp reference kernel, instruction by instruction) and
[`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) (+ its per-step
records) for how minfer's grew against it.

## 4. Performance intuition

### 4.1 The arithmetic-intensity table — decode vs prefill

Arithmetic intensity (**AI**) is FLOPs performed per byte of memory traffic
(`docs/GLOSSARY.md:126`). For a matmul it is decided by the shape, and the
deciding variable is `nt` — how many token rows share each weight byte. The
numbers below are this chapter's own byte/FLOP arithmetic, for one
Qwen2.5-0.5B `ffn_down` layer `[od=896, id=4864]` in Q4_0 (dims:
`docs/QWEN2-SUPPORT.md:79`, walkthrough 05):

| Path | Kernel | Weight bytes moved | AI (FLOP / byte) | What bounds it |
|---|---|---|---|---|
| decode, f32 weights | `f32_f32_matmul_{scalar,vec}` | 17.4 MB | **0.5** | weight bytes, by a landslide |
| decode, Q4_0 (MMVQ) | `q4_0_q8_mmvq` | **2.45 MB** (+ ~6 KB q8 activations) | **≈ 3.6** | still weight bytes — 7.1× lower wall |
| decode, any type (`nt == 1`) | — | one weight stream | ~1 MAC/weight-byte (`docs/GLOSSARY.md:126`) | bandwidth, always |
| prefill, `nt = 512` (MMQ) | `mmq_raw_nb_bt_kernel` | 2.45 MB weights + 3.1 MB pad40 activations + 1.8 MB output ≈ **7.4 MB** | **≈ 600** | math (tensor-core) throughput |

The derivation of the last row: FLOPs = 2·512·896·4864 ≈ 4.46 GFLOP; MMQ
bytes ≈ 512·152·40 B (activations) + 2.45 MB (weights) + 512·896·4 B
(output) ≈ 7.4 MB; 4.46e9 / 7.4e6 ≈ 600. Between the first and last row the
intensity swings by three orders of magnitude — and that swing, not any
kernel's cleverness, is what the dispatch gate `nt >= 9` (`cuda.rs:2754`)
reacts to.

Two consequences worth internalizing:

- **Decode's wall only moves when the bytes do.** At `nt == 1` no code
  change can raise AI; the levers are fewer bytes (quantization: 7.1× here)
  or fewer launches around the same bytes (chapter 05's fusion + CUDA
  Graph). Hence decode wins like "MMVQ +74–77% at 7B shapes"
  (`cuda.rs:2887-2889`, the 8e② record) vs prefill wins like "the whole
  kernel replaced".
- **Prefill's wall only moves when the math does.** At AI ≈ 600 the traffic
  is amortized; what limits the GEMM is MACs per second — why the MMQ
  campaign is a story of inner-loop structure, not byte counts.

### 4.2 The roofline, in one paragraph

The **roofline model** prices any kernel as
`time ≥ max(FLOPs / peak-FLOPs, bytes / peak-BW)` — the larger term wins
(`docs/GLOSSARY.md:125`). On GB10 the bandwidth term uses the documented
~273 GB/s unified LPDDR5x figure (`docs/GLOSSARY.md:127`; chapter 01's toy
measured ~225–229 GB/s of it, `01-gpu-mental-model.md:227`), and the
glossary's one-line classification is this chapter's summary: **GB10 decode
is memory-bound, prefill compute-bound** (`docs/GLOSSARY.md:124`). Check it
against the table: decode Q4_0 at AI ≈ 3.6 tops out near 273 GB/s × 3.6 ≈
1 TFLOP/s *if every byte were useful* — the compute side is far above that,
so bytes bind at every quant level. Prefill at AI ≈ 600 would need ~160
TFLOP/s to stay bandwidth-limited — the other side of the crossover — so
math binds instead. The budget sentence: **at `nt == 1` you buy bandwidth;
at `nt ≥ 9` you buy MACs.**

### 4.3 What would make each rung slow

- **Scalar GEMV, badly gridded.** One thread per output is fine — the
  coalescing is free — but launch it with a 2-D grid over `nt` (the
  pre-Step-82 shape, §3.2's comment) and every token re-streams every byte.
- **Vec GEMV, misaligned.** Drop the `id % 8` gate and the float4 loads
  fault or split into two transactions at the row tails.
- **MMVQ, below its gate.** The uncoalesced 18-byte (2-byte-aligned) weight
  walks only pay off because dp4a turns them into 8 MACs per load; on short
  rows (`id < 2048`, or the 24M/4M-element K-quant floors) the f32 kernels'
  wide coalesced loads win — that is what the per-arm gates *are*
  (`cuda.rs:2842`, `2820-2824`, `2855`).
- **Prefill GEMM, mis-tiled.** A tile that underfills the machine (TM=64 at
  huge `od`, the `MINFER_GEMM_TM` A/B) or a k-step whose shared appetite
  halves occupancy (KS=64's −38%, `cuda_kernels.cu:5087-5094`) trades the
  §2.3 reuse away.
- **Anywhere: the silent 15/16.** Force the tensor-core GEMM onto `nt == 1`
  and 15 of every 16 mma rows are padding (walkthrough 15 §2.3) — no
  profiler shows an error, just a tok/s number that never improves.

## 5. Try it / Observe

Build once (nvcc chain and arch pitfalls: [`docs/BUILD.md`](../BUILD.md); the
GB10's nvcc is not on every shell's `PATH`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
cargo build --release --features cuda
```

Run the bench — `-p 512` prefill tokens (MMQ territory), `-n 64` decode
tokens (MMVQ territory) — flags per [`docs/USAGE.md`](../USAGE.md)
(`bench [-p N] [-n N] [-r N] [-o md|csv|json]`); any Q4_0/Q4_K_M model
works, and `minfer` auto-downloads given an HF name (`docs/USAGE.md:16`):

```bash
./target/release/minfer bench -p 512 -n 64 -r 3 hf:Qwen/Qwen2-0.5B-GGUF:qwen2-0.5b-q4_0.gguf
```

A/B the chapter's decode rung — on a Q4_0 model `MINFER_NO_Q40_MMVQ=1`
reverts `ffn_down` (the one matmul that clears the `id ≥ 2048` gate, §3.3)
from MMVQ to the f32-activation kernel; `MINFER_MMVQ_V1=1` swaps the v2
weight-streaming variants for the 8e originals on K-quant models:

```bash
MINFER_NO_Q40_MMVQ=1 ./target/release/minfer bench -p 512 -n 64 -r 3 <model.gguf>
MINFER_MMQ=0 ./target/release/minfer bench -p 512 -n 64 -r 3 <model.gguf>   # prefill: f16 GEMM (§3.5) instead of int8 MMQ
```

What to look for: the first A/B moves *decode* tok/s (a 7× byte-budget
change on one matmul per layer — small but visible; measured decode deltas
for this class: `docs/CUDA_OPTIMIZATION.md` §0 rows 103/8e); the second
moves *prefill* pp tok/s (the f16 path streams ~3.6× the weight bytes,
§4.1's table). To see which kernels your model actually dispatched, record
`MINFER_TRACE=/tmp/t.json` on a run (trace/viz flow: chapter 03 §5).

## 6. Cross-references

- **[05 · Reading minfer's kernels III](05-kernels-attention-host.md)** —
  next: attention kernels + the Rust host layer, including the `Op::FusedQKV`
  decode tail whose concat matmul runs through §3.3's dispatch.
- **[03 · Reading minfer's kernels I](03-kernels-elementwise.md)** —
  previous: the dispatch chain, the Q4_0 block layout, and the dequant
  kernels that feed §3.5's f16 cache.
- **[Walkthrough 10 · CPU matmul kernels](../inference_e2e_walkthrough/10-cpu-matmul-kernels.md)**
  — the CPU twin: the pairing rule (§2.2, §2.4), the dequant-at-load
  argument (§2.1), the SIMD implementations (§3).
- **[Walkthrough 15 · CUDA backend](../inference_e2e_walkthrough/15-cuda-backend.md)**
  — the dispatch tiers of §3.3 at engine level + the MMQ campaign table.
- **[`docs/LLAMA-CPP-MMQ-ANALYSIS.md`](../LLAMA-CPP-MMQ-ANALYSIS.md)** —
  the reference MMQ kernel, instruction by instruction; the depth §3.6 skips.
- **[`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md)** (+ its
  `docs/cuda_optimization_steps/` records) — measured history of every rung:
  8e (MMVQ), R2 (v2), Step 82 (small-M dispatch), R1→r60 (MMQ promotion).
- **[`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md)** — §5 (memory
  hierarchy, coalescing — the bank-conflict background), §6.2 (matmul
  family map at reference depth).
- **[`docs/GLOSSARY.md`](../GLOSSARY.md)** — AI, roofline, the 273 GB/s
  figure, "decode memory-bound / prefill compute-bound".

← [03 · Reading minfer's kernels I](03-kernels-elementwise.md) · [Index](./README.md) · [05 · Reading minfer's kernels III](05-kernels-attention-host.md)
