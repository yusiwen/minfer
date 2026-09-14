# 05 · Reading minfer's kernels III — attention and the host side

> **Part**: Part 3c — attention kernels + the Rust host layer. **Prereq**: chapters 03–04
> (the matmul ladder, tiling, dispatch — and why decode is memory-bound).
> **Code**: `src/cuda_kernels.cu` (`fa_prefill_f16kv`:4157, `attn_bias_rope_store_f32`:2590),
> `src/graph/cuda_backend.rs` (the `Backend` trait implementation),
> `src/graph/scheduler.rs` (splits, cross-backend copies, replay trigger).

## 1. Background — where this sits

Chapters 03 and 04 read the matmul kernels: how a quantized weight row becomes a
dot product, how tiles map onto blocks, and how the prefill GEMM (General
Matrix-Multiply) differs from the decode matvec. This chapter reads the hardest
device code in the repo — the two attention kernels — and then crosses the
language boundary into the Rust layer that decides which kernel launches, with
which pointers, in which order.

Two things make attention special compared to the matmuls you have already read:

- **It is not a fixed-shape problem.** A matmul's shape comes from the model and
  the token count. Attention's inner loop length is `positions[t] + 1` — the
  number of keys accumulated so far — and that is *data on the GPU*, not a host
  integer. Both kernels in this chapter read the `positions` array on the device
  to find their work. This is minfer's graph rule 1 — "KV positions are data, not
  structure" (`AGENTS.md:78`) — doing real work inside a kernel.
- **It has a serial dependency the matmuls do not have.** Softmax (the
  exponentiate-and-normalize that turns scores into weights) needs the *largest
  score of the whole row* before any output can be finalized. The online softmax
  restructures that dependency into a rescaling loop; §2 teaches it from zero
  with a two-chunk worked example before we touch the real kernel.

The host half of the chapter follows one decode step through the Rust stack:
the `execute_node` dispatch, the device buffer pool, the fused tail that writes
K/V (key/value) into the cache, and the CUDA Graph (record-once, replay-many)
machinery that removes per-kernel launch overhead. Where a design decision has
history — the Phase-3 host-copy bug, the all-weights gate — we cite the record
instead of retelling it.

## 2. Principle — softmax needs the whole row; online softmax pretends it doesn't

### 2.1 The problem with plain softmax

Attention computes, for every query row `t` and every key position `p ≤ t`, a
score, then converts each row into probabilities with softmax and uses them to
average the value rows:

```
s[t][p] = (q[t] · k[p]) / sqrt(hd)
a[t][p] = exp(s[t][p]) / Σ_p' exp(s[t][p'])
o[t]    = Σ_p a[t][p] · v[p]
```

The CPU code can do this literally: materialize the whole score row, take a max,
exponentiate, sum, divide (walkthrough 11 does this in scalar Rust). On the GPU
that "whole row first" step is the problem: at a 2K-token context a single head
row is 2048 scores, and a naive kernel would have to store the whole score
matrix somewhere — huge write+read traffic in global memory, or shared memory
it does not fit in. We want to process the keys in *chunks* with a small
fixed-size state, the way streaming code reads a huge file. The obstacle:
`exp(s - max)` needs the max over the entire row before the first exponentiate
is correct.

### 2.2 The online softmax: running max, running sum, rescaled output

The fix (from the FlashAttention paper — a technique the
[`CUDA-TECH-PRIMER`](../CUDA-TECH-PRIMER.md) §6.3 lists for the prefill
kernel) is to carry three small state variables per row while walking the keys:

- `m` — the largest score seen so far (the running max);
- `l` — the sum of `exp(s - m)` over the keys seen so far (the running sum,
  always computed against the *current* max);
- `o` — the accumulated weighted value sum `Σ exp(s - m) · v`, again against the
  current max.

For each new chunk of scores you compute the new row max `m_new`, then notice
that everything accumulated so far was scaled by `exp(· - m_old)` while it now
needs to be scaled by `exp(· - m_new)`. The correction factor is a single scalar:

```
α = exp(m_old - m_new)
o ← o · α + Σ_new exp(s - m_new) · v
l ← l · α + Σ_new exp(s - m_new)
```

At the end, `o / l` is exactly the softmax-weighted average — the same number
the one-shot formula would have produced, up to floating-point rounding. The
max is never "global"; it is eventually-consistent, and every step that used a
soon-to-be-outdated max gets multiplied back into line.

### 2.3 A two-chunk worked example (actual arithmetic)

One query row, eight keys, processed in two chunks of four. Scores (already
divided by `sqrt(hd)`):

```
chunk 1: s = [1, 2, 4, 3]      chunk 2: s = [0, 5, 2, 1]
```

Value rows: `v₁..v₄ = 1, 2, 3, 4` and `v₅..v₈ = 5, 6, 7, 8`. (Numbers chosen so
the arithmetic stays readable; all figures rounded to 4 decimals — sums are
evaluated at full precision and then rounded, so re-multiplying the printed
operands may differ in the last digit. The online and one-shot totals agree
exactly before rounding.)

**Reference (one-shot) pass.** Global max = 5. Exponentials
`exp(s − 5)`:

```
e⁻⁴    e⁻³    e⁻¹    e⁻²  |  e⁻⁵    e⁰    e⁻³    e⁻⁴
0.0183 0.0498 0.3679 0.1353  0.0067 1.0000 0.0498 0.0183
```

Denominator `l = 1.6462`. Numerator
`o = 0.0183·1 + 0.0498·2 + 0.3679·3 + 0.1353·4 + 0.0067·5 + 1·6 + 0.0498·7 + 0.0183·8 = 8.2916`.
Final output `8.2916 / 1.6462 = 5.037`.

**Online pass, chunk 1.** Max so far `m = 4`. Running sum against 4:

```
l = e¹⁻⁴ + e²⁻⁴ + e⁴⁻⁴ + e³⁻⁴ = 0.0498 + 0.1353 + 1.0000 + 0.3679 = 1.5530
o = 0.0498·1 + 0.1353·2 + 1.0000·3 + 0.3679·4            = 4.7920
```

**Online pass, chunk 2.** The new chunk contains the true max, 5, so
`m_new = 5` and the correction factor is
`α = exp(m_old − m_new) = exp(4 − 5) = 0.3679`. Rescale everything carried from
chunk 1, then add the new chunk computed against 5:

```
o ← 4.7920 · 0.3679 + (0.0067·5 + 1·6 + 0.0498·7 + 0.0183·8)
  = 1.7629         + 6.5287
  = 8.2916
l ← 1.5530 · 0.3679 + (0.0067 + 1.0000 + 0.0498 + 0.0183)
  = 0.5713         + 1.0748
  = 1.6462
```

Both running totals land exactly on the one-shot values. Notice *why* the
rescale is legitimate: the accumulated `o` was a sum of `exp(s − 4)·v` terms, and
multiplying by `exp(4 − 5)` rewrites each term as `exp(s − 5)·v`. The max is only
a *shared shift* that keeps the exponentials out of overflow/underflow territory;
shifting it is algebra, not approximation.

Two details that matter when you read the real kernel:

- The **first chunk is special**: `m` starts at `−∞` and `α` would be
  `exp(−∞ − m_new) = 0`, so the code simply skips the rescale on the first tile
  (see the `fresh0`/`fresh1` flags at `cuda_kernels.cu:4280-4285`).
- **Masked keys must contribute exactly nothing**: causality forbids attending
  to future positions, so a masked score is forced to `0.0` *after* the max
  reduction, not merely given a tiny weight — otherwise `l` would be polluted and
  the normalization would be subtly wrong (this is the `(gcol[q] <= qpos0)`
  guard at `cuda_kernels.cu:4290-4293`).

### 2.4 RoPE in two sentences, and why the tail is fused

**RoPE (Rotary Position Embedding)** encodes a token's position by *rotating*
each adjacent pair of the head vector — element `j` with element `j + hd/2` —
through an angle proportional to the absolute position. Because a rotation of
`m` followed by the inverse rotation of `n` cancels to `m − n`, a query at
position `m` dotted with a key at position `n` depends only on the relative
distance `m − n`, which is exactly the invariance attention wants. (The full
math and the CPU implementation: walkthrough 11 §2.3 — not repeated here.)

**Fusion** means merging several small kernels into one so the intermediate
data never leaves the chip and the launch count drops. In the decode graph,
three tiny steps follow the QKV matmuls — add the projection bias, apply RoPE
to `q` and `k`, scatter `k`/`v` into the cache — each a few microseconds of
work, each a kernel launch. §3.2 reads the kernel that does all of them in one
pass.

## 3. In minfer's code

### 3.1 `fa_prefill_f16kv` — flash-attention-style prefill attention

**The contract.** One CUDA thread block (a group of threads that runs on one
SM — Streaming Multiprocessor — and can share an on-chip scratchpad called
*shared memory*) owns a **64-token tile of queries** for **one head**, and
streams the whole key/value history for that head through a 32-key tile:
`grid.x = ceil(nt / 64)` query token tiles, `grid.y = nh` heads,
128 threads = 4 warps (a warp is 32 threads that execute in lockstep)
(`cuda_kernels.cu:4416`, the launcher). K/V come from the persistent cache as
`__half` (16-bit float — the f16 KV cache from chapter 04's bandwidth story);
q and the output `o` are f32. The kernel is gated to `hd == 128` models — see
the dispatch note at the end of this section.

**Why this shape at all.** The header comment above the kernel
(`cuda_kernels.cu:4096-4102`) is the honest cost accounting of the kernel it
replaced:

```c
// The legacy gqa_attn_f32_f16kv launches one block per (token, head): K is
// re-read per token per head (7B @2K: ~132 GB per layer) and the hd-wide
// accumulator lives in registers (float4 oc[32] = 128 regs → spills). It
// measured 176 ms per layer (76% of the whole 2K prefill). This kernel
// tiles the q dimension: one block per (64-token q tile, head), K/V tiles
// staged in shared memory, QK^T on tensor cores, online softmax with the
// O accumulator in shared memory. K traffic drops to ~0.8 GB per layer.
```

One block per (token, head) re-reads every K row once *per query token*; tiling
64 queries together amortizes each K row across 64 consumers — chapter 03's GEMM
tiling reasoning, applied to attention.

**The tile constants and shared layout** (`cuda_kernels.cu:4112-4113` and
`4167-4174`): `FA_TQ = 64` query rows, `FA_TKV = 32` key columns per iteration;
shared memory holds the q tile, the K tile, and the V tile, all as `__half` with
a padded row stride:

```c
extern __shared__ __align__(256) uint8_t smem[];
// Padded smem row stride: hd=128 halves = 256B ≡ 0 mod 32 banks makes
// every wmma ldmatrix row land on the same bank group (8-way conflict
// per load). +8 halves (272B) shifts each row by 4 banks.
const int sstr = hd + 8;
__half* Qs = reinterpret_cast<__half*>(smem);
__half* Ks = Qs + FA_TQ * sstr;
__half* Vs = Ks + FA_TKV * sstr;
```

The comment is a miniature lesson in **bank conflicts**: shared memory is banked
in 32 lanes, and when every row is exactly 256 bytes wide, the same column of
consecutive rows lands in the same bank, so a multi-row access serializes. The
16-byte padding shifts each row off the hot banks — no algorithm change, just an
address formula.

**QKᵀ on tensor cores.** `wmma` (Warp Matrix Multiply-Accumulate, the CUDA
API for tensor-core matmul) multiplies 16×16×16 matrix fragments; a *fragment*
is the per-lane register layout of a piece of a matrix. The Q·Kᵀ product of a
16-query-row block against the 32-column K tile is accumulated in fragment
registers (`cuda_kernels.cu:4233-4246`):

```c
wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[FA_TKV / 16];
for (int cc = 0; cc < FA_TKV / 16; cc++) wmma::fill_fragment(fc[cc], 0.0f);
for (int d = 0; d < hd; d += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[FA_TKV / 16];
    for (int cc = 0; cc < FA_TKV / 16; cc++)
        wmma::load_matrix_sync(fb[cc], &Ks[cc * 16 * sstr + d], sstr);
    wmma::load_matrix_sync(fa, &Qs[wm * 16 * sstr + d], sstr);
    for (int cc = 0; cc < FA_TKV / 16; cc++)
        wmma::mma_sync(fc[cc], fa, fb[cc], fc[cc]);
}
```

Line by line: `fc[cc]` accumulates this warp's 16 query rows against K-tile
columns `[16·cc, 16·cc+16)`; the loop over `d` walks the head dimension 16
elements at a time (the K dimension of this GEMM), loading an A-fragment
(16 query rows × 16 dims, row-major) and two B-fragments (16 dims × 16 key
columns, col-major — `Kᵀ`'s layout falls out of storing `K` row-major) per
step. The `scale` factor was already folded into q at load
(`cuda_kernels.cu:4188`), so the scores need no second pass.

**The online softmax, fragment-resident.** Now the section-2 machinery, but the
"row" lives in tensor-core accumulator registers. For an m16n16 f32 accumulator
each lane holds 8 elements — two fragment rows (`r0`, `r0+8`) × four column
groups — and four lanes (`l = 0..3`) share one row, so a row max is a local
loop plus a 2-step butterfly shuffle (`__shfl_xor_sync` exchanges a register
between lanes) (`cuda_kernels.cu:4265-4279`):

```c
float mnew0 = -INFINITY, mnew1 = -INFINITY;
for (int q = 0; q < FA_TKV / 16 * 4; q++) {
    // valid = causal (kv <= query pos) AND within the stored KV range
    // (rows >= kv_end are zero-staged and must NOT contribute).
    bool v0 = (gcol[q] <= qpos0) && (gcol[q] < kv_end);
    bool v1 = (gcol[q] <= qpos1) && (gcol[q] < kv_end);
    if (v0) mnew0 = fmaxf(mnew0, sm[q]);
    if (v1) mnew1 = fmaxf(mnew1, sm1_[q]);
}
for (int off = 1; off <= 2; off <<= 1) {
    mnew0 = fmaxf(mnew0, __shfl_xor_sync(0xffffffffu, mnew0, off));
    mnew1 = fmaxf(mnew1, __shfl_xor_sync(0xffffffffu, mnew1, off));
}
```

Note the two validity conditions on every element: **causal masking**
(`gcol[q] <= qpos0` — positions are read from device memory at
`cuda_kernels.cu:4213-4214`) and **tile-range masking** (`gcol[q] < kv_end` —
the last KV tile is zero-filled beyond the history end, and zeros must not
enter the max). Then the classic triple — fresh flags for the first tile,
exponentiate against the new max, running sums —
(`cuda_kernels.cu:4280-4303`):

```c
const int fresh0 = (m0 == -INFINITY);
float a0 = fresh0 ? 0.0f : __expf(m0 - mnew0);   // α, the rescale factor
if (mnew0 == -INFINITY) a0 = 1.0f;               // fully-masked tile: no-op
for (int q = 0; q < FA_TKV / 16 * 4; q++) {
    p0[q] = (((gcol[q] <= qpos0) && (gcol[q] < kv_end)))
                ? __expf(sm[q] - mnew0) : 0.0f;  // masked ⇒ exactly 0
    sum0 += p0[q];
}
for (int off = 1; off <= 2; off <<= 1)             // 4-lane sum butterfly
    sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
if (mnew0 != -INFINITY) m0 = mnew0;
l0 = l0 * a0 + sum0;                               // l ← l·α + new sum
```

**The O rescale in code.** The output accumulator is 8 fragments (the
64-dim-per-16-row-block V product, `hd/16 = 8` blocks). Each fragment's 8
elements interleave the two fragment rows, so the per-row α is applied by
multiplying the right lanes of every fragment (`cuda_kernels.cu:4307-4315`):

```c
// rescale O fragments by the per-row alpha (x[0,1,4,5] -> row r0,
// x[2,3,6,7] -> row r1 — the m16n16 f32 accumulator layout).
for (int ob = 0; ob < 8; ob++) {
    acc[ob].x[0] *= aa0; acc[ob].x[1] *= aa0;
    acc[ob].x[2] *= aa1; acc[ob].x[3] *= aa1;
    acc[ob].x[4] *= aa0; acc[ob].x[5] *= aa0;
    acc[ob].x[6] *= aa1; acc[ob].x[7] *= aa1;
}
```

This is §2.2's `o ← o·α`, 64 rows at a time, entirely in registers. Then P (the
probabilities — the scaled scores) is packed into an f16 A-fragment *in place*
(`cuda_kernels.cu:4317-4333`, exploiting that the m16n16 f32 accumulator and the
m16k16 f16 row-major A-fragment use the same element-per-lane layout), and the
P·V product is accumulated (`cuda_kernels.cu:4334-4343`). The comment at
`4317-4320` records the layout facts that make the round trip free — the kind of
thing you verify once with a standalone fragment-layout test, then trust.

**Write-out and K/V staging.** After the KV loop, one final normalize `acc / l`
and the fragments go to global memory; the last, partially-filled query tile
stages through shared memory so out-of-range rows can be skipped
(`cuda_kernels.cu:4347-4385`); rows whose `l` is 0 (fully masked) stay 0. The
K/V tiles themselves arrive via `fa_stage_kv_async` (`cuda_kernels.cu:4119-4155`):
16-byte `cp.async` transfers — an asynchronous copy that lands in shared memory
without passing through registers — which zero-fill rows beyond `kv_end` by
capping the copy size (`sz = full ? 16 : 0`, `cuda_kernels.cu:4131`); the
pre-sm80 fallback does plain synchronous vector loads. Zero-filling lets the
softmax treat out-of-range keys uniformly and exclude them with the one
`gcol < kv_end` test instead of a second control path.

**Which models take this path.** The host wrapper
`gqa_attn_f16kv` (`src/cuda.rs:4390`) gates it:

```rust
// 8n: prefill (nt >= 64) runs the FA-style tiled attention. ...
if nt >= 2 && hd == 128 && !Self::no_fa_prefill() {
    let rc = unsafe { launch_fa_prefill_f16kv(...) };
    if rc == 0 { return; }
}
```

(`src/cuda.rs:4418-4436`.) Three conditions, each with a reason: `nt >= 2`
(doc 86 lowered the gate from `nt >= 16` because the kernel masks causally from
the positions array, so verify-shaped short batches are safe); `hd == 128`
(FA_HQ is hard-wired to `hd/4 = 32`); and `MINFER_NO_FA_PREFILL=1`
(`src/cuda.rs:4769-4770`) as the A/B escape hatch. If the shared-memory opt-in
fails at launch (`cuda_kernels.cu:4398-4413`) the launcher returns `−1`, prints
one loud warning, and the wrapper falls back to the legacy per-token kernel —
the one visible fallback in the attention path, and it is *announced*, not
silent. Note for Qwen2.5-0.5B specifically: its head dim is 64
(`docs/QWEN2-SUPPORT.md:79`), so 0.5B prefill runs the legacy
`gqa_attn_f32_f16kv` kernel; `fa_prefill_f16kv` serves the hd=128 classes
(Qwen2.5-7B, Qwen3-4B…). The CPU counterpart — the same online softmax in
scalar Rust — is walkthrough 11 §3.2's attention arms.

### 3.2 `attn_bias_rope_store_f32` — the fused decode tail

**The contract.** Decode processes one token (`nt == 1`). After the QKV
projection matmuls, three small jobs remain before attention can run: add the
attention biases (if the model has them), rotate q and k by RoPE, and write k/v
into the persistent KV cache. The unfused graph spent **seven launches** on
these (`add_bias` ×3, `rope` ×2, `store_kv` ×2 — the count in the kernel's
header comment, `cuda_kernels.cu:2571-2574`; TECH-PRIMER §6.4 prices the whole
campaign at "−310 launches/step"). This kernel is one launch that does all of
it, ending with K/V in exactly the layout the next kernel reads.

**Thread mapping.** The launcher (`cuda_kernels.cu:3997-4014`) is a flat 1-D
grid of 256-thread blocks over `total = nqt/2 + nkt/2 + nkt` — one thread per
*RoPE pair* for q (`nqt/2`), one per RoPE pair for k (`nkt/2`), one per element
for v (`nkt`); `nqt = nh·hd` and `nkt = nk·hd` are the q and k section widths
of the (single) token's QKV output. Each thread branches on which section its
linear id `u` falls in — three sections, one kernel
(`cuda_kernels.cu:2590-2610`):

```c
__global__ void attn_bias_rope_store_f32(
    float* __restrict__ q, float* __restrict__ k, float* __restrict__ v,
    const float* __restrict__ bias_q,          // …bias_k, bias_v likewise
    float* __restrict__ kv_k,                  // persistent K region (kv_v too)
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions,
    int kv_is_f16
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    const int pos = positions[0];          // nt==1: read on device
```

`positions[0]` is read *on the device* — the comment at `2588-2589` calls this
out: "no host scalar crosses the launch — CUDA Graph capture/replay safe". A
captured graph freezes its kernel arguments; a host-side `n_past` integer would
be baked in and wrong on every replay. Device-side data is re-read every replay.

**Section 1 — q: bias + RoPE in place** (`cuda_kernels.cu:2612-2625`):

```c
if (u < qpairs) {
    // q section: bias + rope in place (attention reads q at offset 0)
    const int head = u / half_dim;
    const int d    = u % half_dim;
    const int base = head * hd;
    const int j  = base + d;
    const int j2 = j + half_dim;
    float x0 = q[j]  + bias_q[j];
    float x1 = q[j2] + bias_q[j2];
    float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
    float theta = pos * freq;
    float cs = cosf(theta), sn = sinf(theta);
    q[j]  = x0 * cs - x1 * sn;
    q[j2] = x0 * sn + x1 * cs;
}
```

This is verbatim `rope_f32` (`cuda_kernels.cu:2503-2526`) with the bias add
folded into the loads — same NEOX pairing `(j, j + hd/2)`, same frequency
expression, same `cosf/sinf`. "Verbatim" is a hard requirement: the fused
kernel had to be *bit-identical* to the seven-kernel chain it replaced (the
header comment, `cuda_kernels.cu:2580-2585`, lists each correspondence) —
fusion is only free when the answer does not change. The A/B proof lives in
the parity tests (`cuda_backend.rs:3952` exercises the f16 round trip end to
end).

**Section 2 — k: bias + RoPE + store into the cache** (`cuda_kernels.cu:2626-2649`).
The first twelve lines are the q-section math with `bias_k`/`k` swapped — same
pairing, same frequency, same rotation. The new part is what happens after the
rotation: the rotated values are written *both* back to the k buffer *and* into
the persistent K region:

```c
    const float r0 = x0 * cs - x1 * sn;
    const float r1 = x0 * sn + x1 * cs;
    k[j]  = r0;
    k[j2] = r1;
    if (kv_is_f16) {
        ((__half*)kv_k)[(size_t)pos * nkt + j]  = __float2half(r0);
        ((__half*)kv_k)[(size_t)pos * nkt + j2] = __float2half(r1);
    } else {
        kv_k[(size_t)pos * nkt + j]  = r0;
        kv_k[(size_t)pos * nkt + j2] = r1;
    }
}
```

The store lines are the whole KV-cache story in miniature:
`kv_k[(size_t)pos * nkt + j]` — the cache is a flat `[position][nkt]` array,
and the thread computes the scatter address itself from the device-side
`positions[0]`. The same line exists in the standalone `store_kv_f32`
(`cuda_kernels.cu:2539`); the f16 branch converts on store with `__float2half`
(round-to-nearest), the identical conversion the unfused `store_kv_f16` path
uses — again for bit-identity.

**Section 3 — v: bias + store** (`cuda_kernels.cu:2650-2660`): v gets no RoPE
(only q and k are rotated), so its threads add the bias and store one element
each, same `pos * nkt + j` addressing into the V region.

**Why one kernel instead of three (really seven).** Two independent reasons,
and they compound:

1. **Launch overhead.** Every kernel launch has a fixed CPU-side cost, and the
   decode step is a chain of hundreds of small kernels (§4 does the arithmetic);
   TECH-PRIMER §6.4 puts decode chains in the "launch-overhead-bound" regime
   ("2 µs/graph-gap scale", `docs/CUDA-TECH-PRIMER.md:300-302`). Three launches
   replaced by one saves two gaps *per layer per token*, plus the L2
   (layer-2 cache on the GPU) round-trips of writing q/k/v out and reading them
   back.
2. **Producer–consumer locality.** The k section writes the rotated values
   *once*, directly to the cache address `gqa_attn_split` (the next kernel in
   the layer) will read. In the unfused chain k makes three trips through the
   memory hierarchy — written by `rope_f32`, read by `store_kv_f16`, read again
   by attention — for data that never needed to leave the L2. Chapter 04's
   "decode is memory-bound" framing is why this matters even for a few KB.

The graph-level counterpart of this kernel is `Op::FusedQKV` — AGENTS rule 7:
"Decode fusions: `Op::FusedQKV` (concat matmul + bias/rope/store)…"
(`AGENTS.md:84`), with the mechanics in TECH-PRIMER §6.4
(`docs/CUDA-TECH-PRIMER.md:294-298`). Section 3.4 shows the Rust arm that
launches it.

**Two ways to call the same kernel.** The `q/k/v` parameters are *pointer-form
section bases*, which lets one kernel serve both decode layer classes
(`cuda_kernels.cu:2574-2578`): the **concat class** points all three into one
concatenated matmul output (`q = base`, `k = base + nqt`, `v = base + 2·nkt`;
the Rust arm does this pointer arithmetic at `cuda_backend.rs:902-911`), and
the **mixed-quant class** (e.g. a model where `attn_v` is Q6_K and cannot join
the concat) points them at three separate matmul outputs
(`Op::QkvBiasRopeStore`, arm at `cuda_backend.rs:778-842`). One device kernel,
two graph topologies, zero duplicated math.

### 3.3 KV in device memory — where the cache actually lives

Chapter 03 followed weights; the KV cache is the other half of GPU-resident
state, and it has a different owner. Walkthrough 07 §2.5 explains the
*allocator* side; here we look at where those regions physically sit when the
CUDA backend runs, and what the layout buys the kernels of §3.1–3.2.

**Ownership and lifetime.** Each layer owns exactly two persistent regions,
created on first use on the layer's assigned backend
(`src/graph/alloc.rs:384-392`):

```rust
fn ensure_kv(&mut self, layer: usize, backend: Backend, size: usize) -> [BufRef; 2] {
    if let Some(&pair) = self.kv.get(&layer) {
        return pair;
    }
    let k = self.alloc_persistent(&format!("kv.{layer}.k"), backend, size);
    let v = self.alloc_persistent(&format!("kv.{layer}.v"), backend, size);
    self.kv.insert(layer, [k, v]);
    [k, v]
}
```

`alloc_persistent` (`alloc.rs:395-403`) routes through the same pool allocator
as everything else — on CUDA that is a `cudaMalloc` held in the backend's
buffer pool (§3.4) — and registers the buffer as *never freed*. Because the
allocator lives in `GraphCache` (AGENTS rule 2, `AGENTS.md:79`), the regions
survive graph rebuilds and hold their contents across decode steps. That is
the whole cache: two ordinary device buffers per layer that nobody is allowed
to recycle.

**Size and layout.** The size comes from the graph builder:
`kv_elems: nkt * n_ctx` (`src/models/qwen2/graph.rs:128`), where
`nkt = n_head_kv · hd` (the `n_kv_embd` dimension) and `n_ctx` is the
capacity. So the brief question — "[n_past][kv_heads*head_dim]?" — resolves
like this in the store code:

```
region capacity : [n_ctx][nkt]          (nkt = n_head_kv * hd)
element (p, j)  : region[p * nkt + j]   p = absolute position, j = kv dim
```

The store address `dst[positions[t] * nkt + j]` (`cuda_kernels.cu:2539`,
`2644`) indexes by the **absolute position** of the token. `n_past` never
appears in the layout — it is only ever *how many leading rows are valid*, and
that count lives in the `positions` array on the device. That is precisely
AGENTS rule 1, "KV positions are data, not structure" (`AGENTS.md:78`): the
graph topology is identical at position 0 and position 2000, and the kernels
discover the valid range by reading `positions[t] + 1`
(`cuda_backend.rs:1100-1102` has the comment; `fa_prefill_f16kv` computes
`kv_end = positions[last_t] + 1` at `cuda_kernels.cu:4193-4194`). Two payoffs
we have already met: the decode graph can be allocated once and replayed
(§3.5), and attention never needs a host round trip to learn where the history
ends.

```
  decode step, one layer (all CUDA, no host copies)
  rms_norm ──concat matmul──▶ [q | k | v] (pool buf)
                                 │ attn_bias_rope_store_f32:
                                 │  bias+RoPE q,k; K,V scattered at row `pos`
                                 ▼
  ┌─────────────────────┐  ┌─────────────────────┐
  │ kv.{layer}.k region │  │ kv.{layer}.v region │  persistent (never freed)
  │ [n_ctx][nkt] f16    │  │ [n_ctx][nkt] f16    │
  └─────────┬───────────┘  └──────────┬──────────┘
            │ gqa_attn_split reads rows 0..pos
            ▼
      attention output ──▶ wo matmul ──▶ next layer
```

**f16 KV: half the bytes, same addresses.** The `kv_f16` flag
(`cuda_backend.rs:20-26`) is fixed at backend construction from the process
policy (`kv_cache_is_f16`, `src/cuda.rs:1390`, set by the loader at
`src/models/qwen2/loader.rs:347`). When it is on, every store converts to
`__half` and every attention read converts back; §3.2's kernel shows both
sides of that. The `store_kv_f16` header comment states the trade
(`cuda_kernels.cu:2542-2549`): "halves attention read bandwidth", and
`src/cuda.rs:5128-5130` adds the fine print — *the region stays f32-sized; the
f16 view uses the first half of the bytes*: allocation does not shrink, the
bytes written per store and read per attention call do (§4 does the
arithmetic). The correctness story for the f16 round trip is test
`cuda_kv_f16_roundtrip_attn` (`cuda_backend.rs:3952`).

**Why attention can read the regions directly.** A `KvcacheLoad` node is not a
copy — its output buffer *is* the K region (`alloc.rs:226-230` maps the node to
`pair[0]`; the CUDA arm comments "out_buf IS the region — no kernel" at
`cuda_backend.rs:428-430`). So the whole path — matmul, fused tail, cache,
attention, next layer — touches pool device memory and crosses no host
boundary. The one structural guard on that layout: attention requires
`hd == hd_kv` and `nkt == n_head_kv · hd` (the kernels stride KV rows by
`nkt`), and violations return `Err`, not a workaround
(`cuda_backend.rs:1071-1083`; the same guard has a GPU_SAFETY audit entry,
`docs/GPU_SAFETY.md:83`).

### 3.4 The Rust host side — `cuda_backend.rs` as a `Backend`

Everything device-side so far was launched by a Rust struct implementing the
`Backend` trait (`src/graph/backend.rs`: capability query, buffer pool,
`execute_node`, host read/write, `synchronize`). Walkthrough 15 gives the full
tour; this section reads the four parts a contributor actually touches.

**Struct state** (`cuda_backend.rs:20-72`). One field per responsibility:

```rust
pub struct CudaBackend {
    state: &'static crate::cuda::CudaState,  // device handle + the one stream
    kv_f16: bool,                            // KV element policy (§3.3)
    pool: Vec<CudaBuf>,                      // every live device buffer
    free: Vec<usize>,                        // byte-length free list into `pool`
    pool_gen: u64,                           // bumped on every (re)allocation
    ...
    graph_execs: Vec<CapturedGraph>,         // captured CUDA Graphs (§3.5)
    capturing: Option<(u64, (usize, usize))>,// open capture window, if any
    graphs_mode: GraphMode,                  // Enabled / Disabled
    ...
}
```

`CudaState` (`src/cuda.rs`) is the process-wide singleton: device init, the
single stream (the GPU's FIFO execution queue — work enqueued earlier finishes
earlier), the weight registry, and `sync()`. There is deliberately no second
stream: "the stream is the program order" stays true. `pool_gen` looks like
bookkeeping but is load-bearing for §3.5: a captured graph bakes in device
*pointers*, so any pool churn invalidates every capture, and `pool_gen` is the
generation counter that detects that.

**`execute_node`: the dispatch.** The trait method
(`cuda_backend.rs:1380-1401`) is a thin wrapper: it calls
`execute_node_inner` and, if the node failed *while a capture window was
open*, aborts the window first (`cuda_backend.rs:1390-1397` — a doomed window
must never be closed into a cached graph). The real dispatch is one big match
at `cuda_backend.rs:427`:

```rust
match &node.op {
    // Inputs are host-filled by the allocator; KvcacheLoad is a view
    // of the persistent K region (out_buf IS the region — no kernel).
    Op::Input | Op::KvcacheLoad { .. } => Ok(()),
    ...
```

Three representative arms (all cites `src/graph/cuda_backend.rs`):

- `Op::FusedQKV` (`843-930`) — the decode tail of §3.2: the concat matmul
  (`matmul_f32_ptr_layout`, `877-886`, writing `[q|k|v]` into the output
  buffer), then the fused epilogue (`attn_bias_rope_store`, `912-928`) with
  pointer-form section bases computed at `902-911`. KV region pointers come
  from the scheduler's `kv_pair` (§3.3); guards reject `nt != 1`, non-neox
  RoPE, odd `hd` — each `Err` with the offending values (`850-868`).
- `Op::Attn` (`1063-1177`) — the shape dispatcher of §3.1/§3.2: invariants
  first (`1071-1095`), then `nt == 1` → `gqa_attn_split` (split-K
  flash-decoding; the 8d comment at `1103-1107` records why: the single-warp
  kernel left the GPU idle, 48% of the 7B decode step per nsys), `2..=16` →
  the batched variant (`1130-1145`), `nt > 16` → the prefill kernels
  (`1146-1175`), where `gqa_attn_f16kv` internally routes to
  `fa_prefill_f16kv` (`src/cuda.rs:4418`).
- `Op::KvcacheStore` (`1028-1061`) — the unfused prefill store: verifies the
  output buffer *is* the K region (`1031-1035`), derives `nt` from the element
  count, converts positions device-side, and launches
  `store_kv_f16`/`store_kv_f32` once per region.

Every arm ends `Ok(())` or returns `Err(String)`; there is no third outcome.
The match's fallthrough makes the policy explicit (`cuda_backend.rs:1179-1181`):

```rust
op => Err(format!(
    "cuda: op {op:?} has no kernel (stays on the CPU backend per supports_op)"
)),
```

**Buffer pool lifecycle.** The pool is a `Vec<CudaBuf>` (raw pointer + byte
length) with a free list of indices:

```rust
fn alloc_buffer(&mut self, size: usize) -> usize {
    let _sg = self.stream_guard(); // cudaMalloc syncs the device
    let bytes = size * 4;
    if let Some(pos) = self.free.iter().position(|&id| self.pool[id].bytes == bytes) {
        let id = self.free.remove(pos);
        self.pool_gen += 1;            // pointers may have changed hands
        return id;
    }
    let ptr = <crate::cuda::CudaState>::cuda_malloc(bytes); // null on OOM (logged)
    self.pool.push(CudaBuf { ptr, bytes });
    self.pool_gen += 1;
    self.pool.len() - 1
}
```

(`cuda_backend.rs:1333-1353`.) Three conventions to notice. *Exact-size
reuse*: the free list matches byte length, so a recycled buffer is always big
enough. *`pool_gen` on every path*: fresh alloc or reuse, both bump it, because
both can change the node→pointer mapping a captured graph depends on. *No
panics under the lock*: OOM surfaces later as a clean `Err` from `ptr_of`,
because this method may be holding the process-wide stream lock and a panic
would poison it for every other user (`cuda_backend.rs:1345-1348`).
`free_buffer` never calls `cudaFree` — it recycles
(`cuda_backend.rs:1355-1362`), which is what lets the persistent KV regions
and the per-step scratch share one arena; `alloc_fresh`
(`cuda_backend.rs:1364-1378`) bypasses the free list when a buffer's *id* is
still referenced elsewhere (cross-backend staging, walkthrough 07 §2.7). Drop
(`cuda_backend.rs:358-381`) frees the pool, the positions scratch, and every
captured exec. The ownership rule wrapping all of this is AGENTS rule 8:
"Backends own their buffer pools; the allocator is the single owner"
(`AGENTS.md:85`) — the `GraphAllocator` decides *which* buffer a node gets and
when it dies; the backend only manages device memory behind those decisions.

**`read_host` / `write_host` — and the copy rule.** The asymmetry is the
lesson:

```rust
fn read_host(&self, _id: usize) -> Option<&[f32]> {
    // A staged D2H transfer cannot return a borrowed slice (this method
    // takes &self; the host staging buffer would escape its guard). Use
    // `copy_to_host` via alloc.rs's copy_to_cpu CUDA arm instead.
    None
}
```
(`cuda_backend.rs:1403-1408`.) Reading device memory back to the host is
*always* an explicit, syncing `copy_to_host` (`cuda_backend.rs:320-333`:
`state.sync()` then a pinned-staging readback); `write_host`
(`cuda_backend.rs:1410-1426`) is the input-fill path — a pinned-staged *async*
H2D copy, safe because same-stream ordering means later kernels see the data.
The rule behind the asymmetry — **never host-copy a GPU-pending buffer** — is
AGENTS rule 5 (`AGENTS.md:82`), written in the blood of Phase 3. In three
sentences: a per-node host readback inside a split whose command buffer was
still open read *stale* (not-yet-written) data, which surfaced as an all-zero
KV region and garbled output (`docs/COMPUTE-GRAPH-DESIGN.md:905-907`). The fix
was not "sync more" but structural — the in-place aliasing rule plus a single
sanctioned copy point at split boundaries — so the bug class has nowhere to
reappear. The GPU_SAFETY audit generalizes the lesson: any change to shared
mutable GPU state must be validated against a known-good reference, not just
an A/B of two paths over the same corrupted state
(`docs/GPU_SAFETY.md:151-156`).

**`synchronize` and the bounded-wait rule.**

```rust
fn synchronize(&mut self) {
    if self.capturing.is_none() {
        let _sg = self.stream_guard();
    }
    self.state.clear_mmq_cache();   // memos are one-execution-window scoped
    self.pos_memo = None;
    self.close_capture_or_sync();   // closes an open capture, or plain sync
}
```

(`cuda_backend.rs:1428-1440`.) It is deliberately the *only* place a split
boundary waits: memos expire, an open capture window closes here, and the
actual wait is `CudaState::sync` (`src/cuda.rs:2303-2312`) —
`cudaGetLastError` checked, then `cudaStreamSynchronize`, and its error code
checked. That is the CUDA expression of the GPU-safety rule "synchronize() is
the one choke point: stream-ordered work is waited with a bounded loop and the
status is checked" (TECH-PRIMER §7, `docs/CUDA-TECH-PRIMER.md:317-318`; the
rules themselves are `docs/GPU_SAFETY.md`). The scheduler calls this at every
backend boundary via `alloc.sync_backend` — where §3.5 picks up.

### 3.5 CUDA Graph capture/replay and cross-backend splits

**The problem.** A decode step is a few hundred small kernel launches (§4
counts them), each paying a CPU-side cost — TECH-PRIMER §8's one-liner:
"per-launch CPU overhead (~2–7 µs) is pure tax"
(`docs/CUDA-TECH-PRIMER.md:322-323`). **CUDA Graphs** (record a sequence of
launches once, then submit them all with a single replay call) remove most of
that tax without changing the kernels. The scheduler asks the CUDA backend,
before executing a split, whether it wants to replay a capture
(`src/graph/scheduler.rs:194-213`; the ask itself is one line,
`c.graph_replay(graph.uid, split.node_range, …)` at `scheduler.rs:201`). On
the backend, `graph_replay_step` (`cuda_backend.rs:184-251`) runs a three-run
protocol: the first two executions of a (graph uid, node range) go through
normal per-node launches (`graph_runs` counter, `cuda_backend.rs:222-226`); on
the third, the backend opens a *capture window* (`graph_begin_capture`,
holding the process-wide stream lock so no other backend's work is recorded
into the graph, `cuda_backend.rs:237-248`) — from then until `synchronize`,
every kernel the dispatch enqueues is *recorded*, not executed. At the
boundary, `close_capture_or_sync` (`cuda_backend.rs:268-306`) instantiates
the recorded graph, launches it once, and caches the exec; every later step
replays the whole split as **one** `graph_launch_exec` call
(`cuda_backend.rs:210-216`). N per-node launches collapse into one.

That sounds fragile — it would be, if anything the kernels read could change
between steps. Two invariants hold it up. First, **positions are data**
(§3.3): kernel arguments (pointers, dims) are identical every step; only
buffer *contents* change, and those are rewritten before replay — TECH-PRIMER
§8's "why replay is safe in minfer's design"
(`docs/CUDA-TECH-PRIMER.md:338-343`). Second, **pool generations**: any
buffer (re)allocation bumps `pool_gen`, and a replay whose captured `pool_gen`
differs is destroyed and re-captured (`cuda_backend.rs:205-210`).

**`MINFER_NO_CUDA_GRAPH=1` is the A/B revert.** It forces
`GraphMode::Disabled` at construction (`cuda_backend.rs:105-109`), which makes
`graph_replay_step` return `false` (`cuda_backend.rs:190-192`) — every step
runs the plain per-node launch path. It is also the *recovery* switch: any
capture/replay failure disables graphs for the rest of the session with a loud
message saying exactly that (`cuda_backend.rs:217-219`, `291-296`). TECH-PRIMER
§8 calls it "the A/B control used by every graph-adjacent step doc"
(`docs/CUDA-TECH-PRIMER.md:336-337`). A related hard rule: nothing inside a
capture window may sync — a debug readback corrupts the capture, the 7e②
"faster but wrong" incident (`docs/GPU_SAFETY.md:206`) — which is why
trace/viz capture disables replay in the scheduler (`scheduler.rs:190-197`).

**The split/copy story at backend boundaries.** On a mixed graph — or any
graph where consecutive nodes landed on different backends — the scheduler
partitions nodes into contiguous same-backend `Split`s (`split_graph`,
`scheduler.rs:73-120`) and executes each with the same boundary protocol
(`scheduler.rs:177-188`):

```rust
if let Some(pb) = prev_backend {
    if pb != split.backend {
        // 1. flush the previous backend's async work
        alloc.sync_backend(pb);
        // 1b. staged Metal/CUDA captures are valid now — read back
        flush_metal_captures(graph, alloc, &mut staged, trace_on, live_on);
        flush_cuda_captures(graph, alloc, &mut cuda_caps, trace_on, live_on);
        // 2. copy this split's inputs across backends
        for &inp in &split.inputs {
            alloc.copy_across(inp, split.backend)?;
        }
    }
}   // then execute the split's nodes (§3.4's dispatch walk)
```

Sync, then copy, then execute — the *only* sanctioned cross-backend copy in
the system, which is how rule 5's "never host-copy a GPU-pending buffer"
survives contact with multi-backend graphs (walkthrough 08 §2.4 calls this
the split protocol). On an all-CUDA model there is exactly one split, the
boundary work vanishes, and the loop reduces to the replay check plus the
dispatch walk (`scheduler.rs:214-292`; the `BackendTag::Cuda` execute arm is
`scheduler.rs:286-289`). A node whose buffer is on a different backend than
its split is an assignment/alloc bug and returns `Err` with both backends
named (`scheduler.rs:234-241`).

**The gate: all weights registered, or `Err` — never silent.** The kernels of
§3.1–3.2 only exist for the quant types the backend implements. minfer's
answer to "what if a weight has an unsupported type" is to decide *at build
time*, all-or-nothing: CUDA participation requires a device **and** every
weight registered with a kernel-supported type
(`src/models/qwen2/graph.rs:433-438`,
`cuda_on = … && Self::weights_on_cuda(model)`). `weights_on_cuda`
(`src/models/qwen2/graph.rs:688-800`) walks every tensor — embedding,
per-layer wq/wk/wv/wo, gate/up/down, norms, biases — and on failure prints the
exact loser: `"CUDA GATE: weight '{}' (type {:?}) has no CUDA kernel or is not
registered"` (`src/models/qwen2/graph.rs:790-797`). That either routes the
whole model to CPU (loudly, at build time, recorded in `CParams.gpu`) or
admits the graph as fully-GPU. What is *forbidden* is the third option:
discovering mid-run that a kernel is missing and quietly falling back. If a
weight lookup still fails inside `execute_node`, it is
`Err` naming the weight (`cuda_backend.rs:869-874`); an unhandled op is `Err`
(`cuda_backend.rs:1179-1181`); a kernel-invariant violation is `Err` with the
actual values (`cuda_backend.rs:1071-1095`). AGENTS states the contract once:
"kernel-invariant violations return `Err` from `execute_node` — never a
silent CPU fallback; backend assignment is decided at build time"
(`AGENTS.md:72`); TECH-PRIMER §7 repeats it
(`docs/CUDA-TECH-PRIMER.md:312-314`); the design record explains why — silent
fallbacks make performance and correctness bugs indistinguishable
(`docs/COMPUTE-GRAPH-DESIGN.md:909-916`).

## 4. Performance intuition

**Launch overhead, decoded into numbers.** Count the kernels one decode step
launches, directly off the dispatch table of §3.4, for **Qwen2.5-0.5B** (24
layers, 14 query heads / 2 KV heads, `hd = 64`, `n_kv_embd = 128` —
`docs/QWEN2-SUPPORT.md:79`) with the default decode fusions on:

| per layer | launches |
|---|---|
| `RmsNorm` ×2 (attn_norm, ffn_norm) | 2 |
| `FusedQKV` — QKV matvec + `attn_bias_rope_store_f32` | 2 |
| `Attn` (nt = 1) — `gqa_attn_split_partial` + `_combine` | 2 |
| `MatMul` ×2 — wo, down | 2 |
| `FusedFFN` — gate+up concat matvec + in-place swiglu | 2 |
| `Add` ×2 (residuals); `KvcacheLoad` is a view (§3.3) | 2 |
| **per layer** | **12** |

24 layers × 12 = 288, plus embedding gather, the (memoized, §3.4) positions
conversion, final norm, and lm_head ≈ **292 launches per token** — counted
from the dispatch table, not measured; `nsys stats` (§5) shows the real
number for your quant and gate combination. Price it: TECH-PRIMER §8's
measured band for per-launch CPU overhead is ~2–7 µs
(`docs/CUDA-TECH-PRIMER.md:322-323`), so the eager path spends roughly
**0.6–2.0 ms per token just launching kernels** — before the GPU has done any
work. A captured step replays all of it with one launch call. The repo has a
measured anchor for this class of win: the positions-conversion memo (§3.4)
eliminated re-conversions that cost "240 launches/step … ~0.28 ms of pure
launch overhead" at a 14B decode (`cuda_backend.rs:37-43`) — about 1.2 µs per
launch, right in TECH-PRIMER's band. §3.2's fusion is the same arithmetic at
graph level — the 7-launch QKV tail becomes 1 ("−310 launches/step" across a
whole model, `docs/CUDA-TECH-PRIMER.md:294-298`) — and the dispatch notes
price even one wasted launch at "~1-2 us/layer" (`cuda_kernels.cu:4044-4045`).

**f16 KV bytes per token per layer.** With `nkt = n_head_kv · hd`, each region
stores `nkt` elements per position. Qwen2.5-0.5B: `nkt = 2·64 = 128` elements
→ one f32 K row is 512 B, K + V together **1 KB per token per layer** (the
walkthrough's number: 24 KB/token across 24 layers,
`docs/inference_e2e_walkthrough/09-prefill-forward-path.md:253`). With f16 KV
each row is 256 B → **512 B per token per layer, 12 KB/token** model-wide.
Decode attention at context length `p` reads `2 · p` such rows per layer, so
the halving directly halves the attention kernel's KV traffic; at Qwen3-4B
scale (`n_kv_embd = 1024`, 36 layers — 288 KB per position in f32,
`docs/inference_e2e_walkthrough/11-attention-vecops-kv.md:71`) that is ~144 KB
per position *touched*, though the regions stay f32-sized in allocation
(§3.3). The flip side is precision: K/V are rounded to f16 on store and every
downstream kernel reads the rounded values — which is why the parity tests
compare against the *f16-rounded* reference, not f32
(`cuda_backend.rs:4095`, `4831-4841`).

**Prefill attention: the tiling win in one number.** The legacy per-(token,
head) kernel re-read the K history once per query token per head: at 7B @2K
that was ~132 GB of K traffic *per layer*, 176 ms, 76% of the whole 2K
prefill. `fa_prefill_f16kv` amortizes each K row across a 64-query tile and
stages K/V once per 32-key chunk: **~0.8 GB per layer** — about 165× less
traffic (`cuda_kernels.cu:4096-4102`). The grid at those shapes is small and
regular: `ceil(2048/64) = 32` query tiles × 28 heads = 896 blocks of 128
threads, each asking for `((64 + 2·32) · 136 · 2) = 34,816 B ≈ 34.8 KB` of
dynamic shared memory (`cuda_kernels.cu:4395`), raised via
`cudaFuncSetAttribute` (`cuda_kernels.cu:4398-4414`). What makes it *slow*, by
construction: an `hd ≠ 128` model silently takes the legacy path (0.5B does
exactly this, §3.1); a device that refuses the shared-memory opt-in falls back
with one printed warning and a "~50× slower" attention
(`cuda_kernels.cu:4402-4412`); and an unpadded shared-memory stride would
re-introduce the 8-way bank conflicts the `sstr = hd + 8` line exists to
prevent (`cuda_kernels.cu:4168-4171`).

## 5. Try it / Observe

Build once (the nvcc chain is chapter 02's / `docs/BUILD.md`; the GB10's nvcc
is not on every shell's `PATH`):

```bash
export PATH=/usr/local/cuda/bin:$PATH
cargo build --release --features cuda

# A/B the graph replay (the biggest decode lever, §3.5) — same command, env on/off:
./target/release/minfer bench -p 512 -n 128 -r 3 <model.gguf> -o md
MINFER_NO_CUDA_GRAPH=1 ./target/release/minfer bench -p 512 -n 128 -r 3 <model.gguf> -o md

# A/B the attention and fusion paths (each rebuilds the graph; AGENTS rule 7):
MINFER_NO_FA_PREFILL=1 ./target/release/minfer bench -p 512 -n 0 -r 2 <model.gguf>   # hd=128 models only
MINFER_NO_FUSE_QKV=1  ./target/release/minfer bench -p 128 -n 64 -r 3 <model.gguf>   # decode tail, §3.2
```

Expect the replayed runs to win on decode tok/s (the launch tax of §4); expect
**identical greedy output** — replay is bit-parity-gated
(`cuda_graph_replay_bit_parity`, `cuda_backend.rs:6234`).

**Per-node timing and values:** `MINFER_TRACE` records every node's real
output stats (decode steps included; KV nodes skipped) for the viz page — see
`viz/README.md` ("Real trace", `viz/README.md:49-52`):

```bash
MINFER_TRACE=/tmp/t.json ./target/release/minfer <model.gguf> "Hello!" -n 5
# the trace shows the fused nodes (fused_qkv, qkv_bias_rope_store) sitting
# where seven nodes used to be; to see the launch tax itself, compare:
nsys profile -o /tmp/decode --force-overwrite true ./target/release/minfer <model.gguf> "hi" -n 32
nsys stats --report cuda_gpu_kern_sum /tmp/decode.nsys-report
```

**Parity tests** (no model file needed — synthetic weights, real GPU):

```bash
cargo test --release --features cuda cuda_fa_prefill_attention_parity -- --nocapture
cargo test --release --features cuda cuda_kv_f16_roundtrip_attn     -- --nocapture
cargo test --release --features cuda cuda_graph_replay_bit_parity   -- --nocapture
```

## 6. Cross-references

- **06** — next: optimization + verification, where the launch/bandwidth
  arithmetic of §4 becomes a toolkit.
- **04** — previous: the decode matvec and why decode is memory-bound (the
  premise §3.2's fusion argument stands on).
- `docs/inference_e2e_walkthrough/11-attention-vecops-kv.md` — the attention /
  RoPE / KV *math* and CPU implementations this chapter deliberately does not
  repeat (§2.1, §2.3 especially).
- `docs/inference_e2e_walkthrough/07-allocator-liveness-kv.md` — why the KV
  regions are persistent and who frees them (§2.5).
- `docs/inference_e2e_walkthrough/08-scheduler-execute.md` — the split protocol
  and error contract from the scheduler's side (§2.4, §2.5).
- `docs/inference_e2e_walkthrough/15-cuda-backend.md` — the end-to-end CUDA
  backend tour (the "what"; this chapter is the "how, line by line").
- `docs/CUDA-BACKEND-DESIGN.md` — design goals + the phase-by-phase
  implementation record (§5) behind every "Phase 7d"-style comment quoted here.
- `docs/CUDA-TECH-PRIMER.md` §7 (synchronization discipline) and §8 (CUDA
  Graphs) — the reference-depth versions of §3.4–3.5.
- `docs/GPU_SAFETY.md` + `AGENTS.md` — the safety rules cited throughout
  (rules 1, 5, 7, 8; the Err-never-fallback contract).
- `docs/CUDA_OPTIMIZATION.md` (+ `docs/cuda_optimization_steps/`) — the
  measured campaign history for every number quoted from a step doc.

← [04 · Reading minfer's kernels II](04-kernels-matmul.md) · [Index](./README.md) · [06 · Optimization methods](06-optimization-methods.md) →
