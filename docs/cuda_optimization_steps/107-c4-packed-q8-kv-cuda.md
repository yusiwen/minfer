# 107 · C4 packed Q8_0 KV on CUDA — the fused decode epilogue and the FA prefill (#144)

> **Result**: Qwen3-0.6B Q8_0 `pp2048` 564.5 → **8231.1 tok/s** (14.58x; the f16 arm is 8540.8, so
> the packed/f16 factor went **14.7x → 1.038x**); Qwen2.5-0.5B q4_0 `tg128` q8_0 161.5 → **170.1 tok/s**
> (1.053x, the f16 gap 1.470x → 1.393x — the ticket's "~1.18x" was the f16-weight arm's cut and did
> *not* reproduce on q4_0 weights).
> **Commit**: `<this PR>` (the code in `src/cuda_kernels.cu`, `src/cuda.rs`,
> `src/graph/{cuda_backend,alloc,ops}.rs`, `src/models/qwen{2,3}/graph.rs`). **Date**: 2026-09-26.

## 1. Background — where things stood

C4 (ticket [#42](https://github.com/yusiwen/minfer/issues/42)) gave the engine a **packed Q8_0 KV
cache**: a cell is a whole number of 32-element Q8_0 blocks (34 bytes each: one f16 scale plus 32
int8 quants) rounded up to whole f32 words. Its CUDA half, S2b
([#87](https://github.com/yusiwen/minfer/issues/87)), landed the `KV_LAYOUT_F32/F16/Q8_0` tag, a
byte-addressed `kv_row` + `kv4<LAYOUT>` load, a `store_kv_q8_0` that reproduces the CPU quantizer
byte for byte, and the registry's `reads_packed_kv = true` for CUDA. The result was correct and
**3.76x smaller than f32**, but three paths were deliberately off their tuned route and the S2b
record said so:

| model / config | f16 | q8_0 | factor |
|---|---|---|---|
| Qwen2.5-0.5B q4_0 `tg128` | 239.76 | 162.46 | 1.48x slower |
| the same f16 with `MINFER_NO_FUSE_QKV=1` | 203.58 | — | the packed kernel's own cost is 1.25x |
| Qwen2.5-0.5B q4_0 `pp2048` | 2693.54 | 2171.85 | 1.24x slower |
| Qwen3-0.6B Q8_0 (hd 128) `pp2048` | 8604.56 | 562.76 | **15.3x slower** |
| Qwen3-0.6B Q8_0 (hd 128) `tg128` | 137.85 | 122.21 | 1.13x slower |

The three cuts were:

1. **No fused decode QKV epilogue.** `Op::FusedQKV` / `Op::QkvBiasRopeStore`'s epilogue writes one
   K/V element per thread. A Q8_0 block's scale is `amax/127` over all 32 of its elements, so a
   per-element store has no whole block to quantize — the builders' `layer_gpu` gate carried
   `&& !packed` and a Q8_0 decode ran the unfused chain (`add_bias`×3 + `rope`×2 + `store_kv`×2).
2. **`kv4<Q8_0>` costs four int8 converts and four multiplies** where `kv4<F16>` costs two
   `__half2` converts. That is the residual 1.25x the S2b table attributes to "the packed kernel's
   own cost".
3. **No packed FA prefill.** `fa_prefill_f16kv` stages K/V into shared memory **as f16** with
   16-byte `cp.async` chunks and then runs a tensor-core QK^T; a packed cell cannot be `cp.async`'d
   because its quants must be scaled first. The dispatch therefore sent a packed prefill to the
   general layout-tagged kernel (one block per (token, head), K re-read per token per head) — the
   same kernel the FA rewrite was built to replace, at 15x.

This ticket takes cuts 1 and 3. Cut 2 (a `dp4a` packed K dot) is a **numerics change** — it
accumulates `int` dot products against a per-head-quantized query — and needs its own accuracy
statement and a re-measured real-model tolerance, so it is filed separately rather than assumed.

## 2. Principle — the GPU mechanism

**Cut 1: the launch count, not the arithmetic.** On GB10 a decode step of the 0.5B is 24 layers ×
~95 nodes; the unfused QKV chain is 7 launches/layer where the fused one is 1 (the concat matmul is
the same work as the three it replaces, and the packed epilogue does the same bias/rope/quantize
work the chain did, just in one grid). A CUDA-graph replay still pays a per-launch gap of a couple
of microseconds, so 6 launches/layer × 24 layers is ~0.3 ms/token — which is exactly the 0.32 ms
the two arms differ by at 161 vs 170 tok/s. The fused packed epilogue is therefore expected to buy
the **launch overhead**, and nothing about it makes the quantizer cheaper: the block-owning
mapping has one thread per (head, 32-element K block), and because the rope pair `(d, d + hd/2)`
straddles blocks, each thread recomputes its own pair's `powf`/`cosf`/`sinf`. For `hd = 64` that is
2x the transcendentals of the f32/f16 mapping — real work, but ~1% of a 5.9 ms step.

**Cut 3: staging is the only layout-dependent step.** The FA kernel's cost structure is fixed by
its tile, not by what the cache stores: a 64-query × 128-dim tile per (q-tile, head), K/V tiles of
32 rows staged into padded shared memory, QK^T on `wmma` tensor cores, a register-resident online
softmax, and P·V on tensor cores. Q8_0 K/V can feed exactly that pipeline if the *staging loop*
dequantizes: 32 halves per K/V row are 16 bytes per 8 elements, one `uint4` smem store, and the
8-element group never straddles a block because a head base is 32-element aligned. The dequantized
tile is f16, so the packed prefill lands in the **same precision class as the f16 FA route**, which
is why the packed/f16 gap after the change is the tile's own cost rather than a new numerics
question. The general layout-tagged kernel stays the documented fallback for a device whose
shared-memory opt-in fails.

## 3. Implementation

### 3.1 Design choices

- **A separate packed epilogue kernel, not a template branch.** The K/V thread mapping is
  fundamentally different (one thread per block, not per element), so `attn_bias_rope_store_q8_0`
  is its own kernel; the f32/f16 instruction stream is untouched.
- **Neither K nor V is written back.** Both fused classes leave those buffers dead (attention reads
  the packed region), and a block-owning thread cannot write `k` in place without racing the thread
  that reads its pair partner. The observable output — the packed region's bytes — is the unfused
  chain's.
- **One quantizer, two callers.** `q8_0_quantize_block(x, cell)` was factored out of
  `store_kv_q8_0`; the epilogue calls the same function, so a CPU-parity fix cannot land in one and
  miss the other.
- **The fused metas carry the cell width.** `FusedQkvMeta` / `QkvBiasRopeStoreMeta` /
  `FusedQkvNormMeta` gained `row_elems` (stamped by the model builders from
  `CParams::kv_format`), and the allocator sizes the region from it. Before #144 only
  `KvcacheMeta` had it, because a packed cache never built a fused node.
- **`fa_prefill_f16kv` grew a `LAYOUT` template parameter** rather than a second kernel: the body
  after staging is layout-blind, so one body + a per-layout staging function is the whole
  difference. `MINFER_NO_FA_PREFILL=1` still forces the general kernel for both layouts, which is
  the same-binary A/B control.
- **An observation counter at the chokepoint.** `gqa_attn_kv_prefill` bumps
  `testfail::note_checked("cuda_fa_prefill_q8_0")` after a successful packed FA launch, so the gate
  can distinguish "the packed FA route ran" from "some correct attention kernel ran".

### 3.2 Key code

The packed K block, from `src/cuda_kernels.cu` (the block offset is the line the new gate caught):

```cpp
const int b = u - qpairs;
const int blk = b % (hd / Q8_0_BLOCK_ELEMS);
const int head = b / (hd / Q8_0_BLOCK_ELEMS);
float x[Q8_0_BLOCK_ELEMS];
#pragma unroll
for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
    // `d` is the element's index inside the head, so the block's own
    // offset must be added ...
    const int d = blk * Q8_0_BLOCK_ELEMS + i;
    const int dd = (d < half_dim) ? d : d - half_dim;
    const int ja = head * hd + dd;
    const int jb = ja + half_dim;
    float x0 = k[ja] + bias_k[ja];
    float x1 = k[jb] + bias_k[jb];
    float freq = freq_scale / powf(freq_base, (2.0f * dd) / hd);
    float theta = pos * freq;
    float cs = cosf(theta), sn = sinf(theta);
    x[i] = (d < half_dim) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
}
q8_0_quantize_block(x, kv_k + (size_t)row * row_bytes + (size_t)b * Q8_0_BLOCK_BYTES);
```

The packed staging, the only layout-dependent part of the FA prefill:

```cpp
if (LAYOUT == KV_LAYOUT_Q8_0) {
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    for (int c = tid; c < FA_TKV * hd / 8; c += nthreads) {
        int r = (c * 8) / hd, d = (c * 8) % hd;
        const int p = kt + r;
        const int row = kv_cell<MAP>(bound, mt, 0, p);
        if (p < kv_end) {
            kv8_q8_0(Ks + r * sstr + d, kv_row(kv_kbase, row, row_bytes), hk * hd + d);
            kv8_q8_0(Vs + r * sstr + d, kv_row(kv_vbase, row, row_bytes), hk * hd + d);
        } else {
            *reinterpret_cast<uint4*>(Ks + r * sstr + d) = z4;
            *reinterpret_cast<uint4*>(Vs + r * sstr + d) = z4;
        }
    }
    return;
}
```

### 3.3 Pitfalls

- **The block offset.** The first version of the K loop wrote `const int d = i;` (the *block-local*
  index), so every block of a head computed the head's first 32 values. The new byte-exact gate
  failed immediately at `pos = 0` with both byte arrays printed; the fix is the `blk * 32 + i`
  above. This is the concrete reason the gate's first arm is byte-exact against the CPU quantizer.
- **The region-width mismatch.** The first packed fused graph died at load:
  `KV region for layer 0: the node declares 34 words per cell but the q8_0 layout packs one cell of
  512 elements into 136`. The fused node handed the allocator its *logical* width; with `row_elems`
  in the meta it hands the packed one, and `ensure_kv`'s cross-check holds.
- **In-place q.** The rope rewrites `q` in place, so an A/B test that reuses the same buffer must
  rewrite it before the second arm — the first version of the new epilogue gate compared a
  doubly-roped `q` and "found" a 2.0 delta that was the test's fault.
- **`cp.async` cannot transform.** The f16 staging is 16 bytes of `cp.async` per 8 elements; the
  packed arm must load, scale, convert and store, so it is synchronous. The FA kernel's
  `cp.async.commit_group/wait_group` pair after the call is harmless on that arm.

## 4. Verification

- `cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer` (**new**): K and V byte-exact against the CPU
  quantizer at `pos = 0` (where the rope is the identity permutation), a value check on the roped q
  and a dequantized-K check at `pos = 7` (the device's `cosf`/`sinf` may differ from the host's by
  an ulp, which can flip a quant on a boundary).
- `cuda_q8_0_fa_prefill_attention_parity` (**new**): a 100-token, hd-128 prefill against the CPU
  attention over the *same packed bytes* dequantized on the host — max err **4.8e-4** against the
  5e-3 class `cuda_fa_prefill_attention_parity` pins for f16 (2.8e-4) — plus the `note_checked`
  observation arm.
- The five named Q8_0 gates stay green: `cuda_q8_0_store_matches_the_cpu_quantizer`,
  `cuda_kv_q8_0_roundtrip_attn`, `cuda_q8_0_kv_cell_move_strides_by_row_bytes`, the Q8_0 arm of
  `cuda_map_window_matches_the_span_over_the_same_rows`, and the device arm of
  `a_packed_kv_cache_answers_like_the_f32_one`.
- Mutations (gate contract rule 3): `d = i` → the byte arm red; `elem >> 4` in `kv8_q8_0` → the FA
  parity red (max err 3.33); skipping the packed FA launch → the parity arm **green** (5.8e-5) and
  the observation arm red.
- `MINFER_TEST_ISSUE162=1` drives all 124 audited `<<<` sites (six new) green; the
  `tests/fixtures/cuda_launch_sites.tsv` fixture was regenerated.

## 5. Results

GB10 sm_121, `cargo build --release --features cuda`, `minfer bench -p 2048 -n 128 --n-ctx 4096
-o json`, `MINFER_CACHE_TYPE` pinned, **5 interleaved rounds, medians**, 2026-09-26. Baseline is a
separate binary built from master `85c712e`.

| arm | pp2048 | tg128 |
|---|---|---|
| baseline (master) 0.5B q4_0 q8_0 | 2139.10 | 161.49 |
| baseline 0.5B q4_0 f16 | 2626.25 | 237.37 |
| baseline 0.5B q4_0 f16 `MINFER_NO_FUSE_QKV=1` | 2647.15 | 200.80 |
| baseline Qwen3-0.6B Q8_0 q8_0 | 564.47 | 122.57 |
| baseline Qwen3-0.6B Q8_0 f16 | 8323.24 | 136.73 |
| **new** 0.5B q4_0 q8_0 (fused, item 1) | 2144.18 | **170.11** |
| new 0.5B q4_0 q8_0 `MINFER_NO_FUSE_QKV=1` | 2136.85 | 161.63 |
| new Qwen3-0.6B Q8_0 q8_0 (FA packed, item 3) | **8231.05** | 122.47 |
| new Qwen3-0.6B Q8_0 q8_0 `MINFER_NO_FA_PREFILL=1` | 564.57 | 122.26 |
| new Qwen3-0.6B Q8_0 f16 | 8540.83 | 136.53 |

- **Item 1, 1.053x** (same-binary fused/unfused 1.0525x). The bar named before measuring was 1.15x,
  taken from the ticket's "~1.18x"; that number is the *f16-weight* arm's cut
  (237.37/200.80 = 1.182x) and did not reproduce on the q4_0 packed arm. The f16 gap narrowed
  1.470x → 1.393x.
- **Item 3, 14.58x** (8231.05 vs 564.57 same-binary; 14.58x vs the baseline binary). The bar named
  first was ≥ 0.5x of the f16 arm; the measured ratio is **0.964x** — the packed/f16 factor is
  **1.038x** where it was 14.7x.
- **No regression**: the 0.5B `pp2048` (hd 64, where FA does not apply) moved 2139.10 → 2144.18
  (1.002x, noise).

## 6. Lessons

1. **A bar borrowed from another weight type is not a bar for this arm.** The fusion cut is
   launch-overhead; its *relative* size depends on what fraction of the step the surrounding
   matmuls take, and f16 weights make that fraction different from q4_0 weights. Name the bar on
   the arm being changed.
2. **A byte-exact reference finds the block-offset bug in one run.** A tolerance gate would have
   passed the wrong K values (they are still "plausible roped numbers"); comparing to the CPU
   quantizer's bytes did not.
3. **An observation arm is what separates "correct" from "routed".** With the packed FA launch
   removed the parity arm's error *improved* (5.8e-5 vs 4.8e-4) — a mode-vs-reference gate cannot
   see a silent fallback to a *more* accurate path.
4. **A new node kind must carry the packed cell width.** The fused metas had no `row_elems` because
   only the store kind could ever run packed; the first packed fused graph found that out at load,
   with the region refusing rather than corrupting.

← [106 · query-formula gates](./106-query-formula-gates.md) · [Index](./README.md)
