# 108 · #186: the dp4a packed Q8_0 K dot on CUDA (LANDED)

> **Result**: the packed Q8_0 decode K dot now accumulates in `int` (`__dp4a`) against
> a per-block-quantized query. Qwen2.5-0.5B q4_0 `tg128` **171.95 → 193.30 tok/s**
> (1.124x, medians of 5 interleaved same-binary rounds on GB10 sm_121, 2026-09-27),
> which narrows the packed/f16 gap **1.396x → 1.242x**; Qwen3-0.6B Q8_0 (hd 128)
> `tg128` **123.98 → 136.66** (1.102x), i.e. the packed decode now *matches* the f16
> arm (136.73). The isolated kernel delta (same 1-warp geometry, load only) at hd 64
> falls **+96% → +40%**, and the real-path partial kernel **67.4 → 39.9 µs** per
> layer/token.
> **Commit**: `<this PR>` (`src/cuda_kernels.cu`, `src/cuda.rs`,
> `src/graph/cuda_backend.rs`, `tests/fixtures/cuda_launch_sites.tsv`).
> **Date**: 2026-09-27.

## 1. Background — where things stood

[#144](./107-c4-packed-q8-kv-cuda.md) landed the packed fused decode epilogue and the
packed FA prefill. What it deliberately left was the third item, a **dp4a packed K
dot**, because quantizing the query into the accumulation is a numerics change with
its own accuracy statement (filed as [#186](https://github.com/yusiwen/minfer/issues/186)).
The residual it named was **1.393x** on Qwen2.5-0.5B q4_0 `tg128` (170.11 q8_0 vs
236.97 f16), and [#144] says the residual lives in **two** places: the 1-warp split-K
decode body's `rpw_gate = 0` **plus** the `kv4<KV_LAYOUT_Q8_0>` load. A dp4a rewrite
can only touch the second.

That split is the whole trap of the ticket: a measurement that attributes the entire
q8_0-vs-f16 kernel gap to the load over-claims, because at hd 128 the f16 arm takes
the 4-warp hybrid body while the packed arm is stuck on the 1-warp one. The
measurement therefore had to be structured so the two arms share **one** geometry,
with the load as the only variable. At **hd 64** (the 0.5B's shape) they do:
`launch_gqa_attn_split_f16kv` and `launch_gqa_attn_split_q8_0` both take the `hd != 128`
branch, both launch `gqa_attn_split_partial<LAYOUT, true, false>` with `rpw_gate = 0`
on the same `(32, n_head)` grid of 32-thread blocks. The only difference in the
instantiation is `kv4<LAYOUT>`.

**The bar, named before measuring: the load-attributable share of the Q8_0 decode step
must be ≥ 10%.** The reasoning: (a) the whole q8_0 → f16 residual is 1.393x, and only
one of its two named components is addressable, so the recoverable share is a strict
subset; (b) a kernel that made the load *free* would recover at most
`share / 1.393`, and dp4a removes only the convert+multiply part of the K side — not
the memory traffic, not the block-scale read, and it *adds* a query quantize; (c)
[#144]'s landed item 1 was worth 5.3% on this arm, the measured floor for "worth
landing", and a dp4a rewrite that cannot clear 5% of the step is not worth a numerics
change. 10% of the step for the *entire* load delta is the conservative upper-bound
test: if the whole delta is under it, no variant can be worth landing.

**The bar cleared: 20.3%.** `nsys` per-kernel durations on the real decode path (node
tracing, so CUDA-graph replays are expanded) at 0.5B/n_ctx 4096/tg128: the incumbent
packed partial kernel is **67424 ns** median per layer/token (15360 instances =
24 layers x 128 tokens x 5 passes) against the f16 arm's **18528 ns** — a
**1.182 ms/token** delta of a **5.818 ms/token** step. At hd 64 there is no geometry
difference to confuse it with, so the whole delta is the load.

## 2. Principle — the GPU mechanism

**The packed load is instruction/latency-bound, not bandwidth-bound.** The packed arm
reads *half* the bytes of the f16 arm and is still ~2-3.6x slower per kernel, so the
cost is not traffic. Per 4 elements `kv4<KV_LAYOUT_Q8_0>` does: form the block address
(`elem >> 5` and a `* 34`), load the f16 scale, convert it to f32, load four `int8`
quants, convert each to f32, multiply each by the scale, then four FMAs for the dot
with the (f32) query. `kv4<KV_LAYOUT_F16>` does two 16-byte-loads worth of `__half2`,
two `__half2`→f32 converts and the same four FMAs. The packed arm pays roughly four
extra converts plus four extra multiplies per four elements, on a long dependency
chain (scale load → cvt → mul → FMA) that a 1-warp block cannot hide.

**The integer form deletes exactly that.** Quantize the query into the *same* Q8_0
format once per (head, 32-element block): `qi = round(q_i / d_q)` with
`d_q = amax/127`. The stored K quants are already `k_i = round(k_i / d_k)`, so
`sum_i qi * k_i` is an `int` dot that `__dp4a` computes four lanes at a time, and the
true f32 dot is recovered by multiplying the accumulated `int` by `d_q * d_k` — **once
per block**, not once per element. K is never converted to float. V still is: V
accumulates in f32, so only the K side changes. This is exactly the CPU's
`dot_q8_0_q8_0` shape, which is why the numerics class is the CPU packed class.

**The block scale is a per-lane value inside a row.** A lane owns four consecutive dims
and a 32-element block spans eight lanes, so the eight lanes that share a block reduce
their `amax` with three `__shfl_xor_sync` offsets (4/2/1 stay inside the 8-lane group;
`hd % 32 == 0` is `ensure_kv`'s packed-width invariant). Each lane then carries its own
`qscale * kd[j]`, so the multiply happens **before** the existing float warp reduction,
not after it.

**The packing cost is the risk, and it is real.** A Q8_0 block is `f16 d; i8 qs[32]`
(34 bytes), so `blk + 2 + (elem & 31)` is not 4-byte aligned (`34 * k` alternates
parity) and the four quants cannot be one 32-bit load — they are four byte loads plus
three shifts and three ORs. That is why this was a measurement, not an argument: the
packing could have eaten the entire saving. It did not.

## 3. Implementation

### 3.1 Design choices

- **A template parameter, not a second body.** `attn_split_1w_body<LAYOUT, CAUSAL,
  MAP>` gained `bool Q8DP4A = false`; the K staging and the dot are the only branches
  it guards. The f32/f16 instruction streams and the `gqa_attn_split_partial_bt`
  (verify) instantiation are untouched (they pass `false`). One source, so the
  incumbent and the int dot cannot drift.
- **The query quantize lives in the kernel, before the row loop.** A separate prepass
  kernel would need its own launch per layer (~1-2 µs x 24) and a graph node; the
  in-kernel form costs three shuffles and four `rintf` per lane, amortized over the
  whole split's rows. It also keeps the decode launch count unchanged, which matters:
  the decode graph is captured, and a new node is a topology change.
- **The arm is per instance of the launcher, selected at the call site.** The `.cu`
  launcher branches on an `int dp4a` argument; `cuda::q8_kv_dp4a_enabled()` resolves
  `MINFER_NO_DP4A_Q8_KV` once per process and `gqa_attn_split_once` passes it. This is
  the same discipline as the per-engine `kv_layout` tag: the answer is fixed before a
  captured graph can observe a change.
- **An observation counter at the chokepoint.** The Rust call site bumps
  `testfail::note_checked("cuda_q8_kv_dp4a")` only when it launched the dp4a arm, so a
  gate can prove *which* arm ran rather than reading the launcher's own report.

### 3.2 Key code

The packed load, from `src/cuda_kernels.cu`:

```cpp
__device__ __forceinline__ void kv4_q8_0_packed(const char* row, int elem, int& q, float& d) {
    const unsigned char* blk =
        reinterpret_cast<const unsigned char*>(row) + (size_t)(elem >> 5) * Q8_0_BLOCK_BYTES;
    d = __half2float(*reinterpret_cast<const __half*>(blk));
    const signed char* p = reinterpret_cast<const signed char*>(blk + 2) + (elem & 31);
    q = ((int)(unsigned char)p[0]) | ((int)(unsigned char)p[1] << 8) |
        ((int)(unsigned char)p[2] << 16) | ((int)(unsigned char)p[3] << 24);
}
```

The query quantize, once per lane per split, with the 8-lane `amax` reduction:

```cpp
if (Q8DP4A) {
    float amax = live ? fmaxf(fmaxf(fabsf(q4.x), fabsf(q4.y)), fmaxf(fabsf(q4.z), fabsf(q4.w)))
                      : 0.0f;
    #pragma unroll
    for (int off = 4; off > 0; off >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, off));
    qscale = amax / 127.0f;
    const float qid = (qscale != 0.0f) ? (1.0f / qscale) : 0.0f;
    if (live) { /* pack round(q_i * qid) into qk */ }
}
```

…and the dot itself, replacing the four-term f32 expression:

```cpp
d = (float)__dp4a(qk, ki4[j], 0) * (qscale * kd4[j]);
```

### 3.3 Pitfalls

- **A one-key arm is vacuous.** The first version of the new gate arm ran the decode
  at `positions = [0]`, so the window held a single cell, the softmax had one key and
  the K score cancelled out of the output entirely. Mutating the block base
  (`elem >> 5` → `elem >> 4`) still passed. The mutation is what found it — the gate
  now stores **two** cells and reads them through an explicit `[0, 2)` span.
- **The u32-bits indirection in the fixture.** `positions`/`cells`/`span` buffers hold
  i32 values as `f32::from_bits`, so a literal `[0, 2]` must be written through the
  bits helper, not as `0.0`/`2.0`.
- **The max |Δlogit| did not move — because the prefill dominates it.** The real-model
  gate's worst delta is at its `nt = 512` prefill step, which the decode kernel does
  not touch. The *decode* steps' deltas did move (1.622 → 1.538 at step 1, and the
  argmax max 0.596 → 0.553), and nsys shows the decode kernel name changing with the
  control; that is the evidence the arm ran (see §4).

## 4. Verification

Provenance for every number: **GB10 sm_121, CUDA 13.0, driver 580.178.04,
2026-09-27**, `cargo build --release --features cuda` at this worktree.

**The isolated probe** (temporary, removed before the commit; the source is the
`kv4_q8_0_packed`/`Q8DP4A` code above driven through `cuda::gqa_attn_split`):
9 interleaved rounds x 200 launches, medians, µs per call (partial + combine). The
f16 region holds the **dequantized bytes of the Q8_0 cells**, so the two arms read the
same K/V and differ only in the accessor:

| shape | f16 | q8_0 incumbent | q8_0 dp4a | incumbent delta | dp4a delta |
|---|---|---|---|---|---|
| 0.5B hd64 nh14 nk2 nkv2049 | 20.485 | 40.176 | 28.683 | +96.1% | +40.0% |
| 0.5B hd64 nh14 nk2 nkv2176 | 20.496 | 40.996 | 28.703 | +100.0% | +40.0% |
| Qwen3 hd128 nh16 nk2 nkv1024 | 16.379 | 22.559 | 18.448 | +37.7% | +12.6% |
| Qwen3 hd128 nh16 nk2 nkv2176 | 21.546 | 40.955 | 28.703 | +90.1% | +31.5% |

At hd 64 both layouts take the identical 1-warp body, so the delta is the load alone;
dp4a removes **56-58% of it**. (The probe's absolute times are smaller than the real
path's because the probe re-reads one layer's region back to back and keeps it L2-hot;
the real decode interleaves 24 layers and streams the weights. That is why the real
path is measured with nsys as well.)

**nsys, real decode path** (`MINFER_CACHE_TYPE=q8_0 nsys profile --trace=cuda
--cuda-graph-trace=node -o ... ./target/release/minfer bench -p 2048 -n 128 --n-ctx
4096 <0.5B q4_0>`), 15360 instances = 24 layers x 128 tokens x 5 passes:

| kernel | arm | med ns | total s | per token ms |
|---|---|---|---|---|
| `gqa_attn_split_partial<2,1,0,false>` | incumbent | 67424 | 1.0466 | 1.635 |
| `gqa_attn_split_partial<2,1,0,true>` | dp4a | 39904 | 0.6224 | 0.973 |
| `gqa_attn_split_partial<1,1,0>` | f16 | 18528 | 0.2900 | 0.453 |

The load-attributable delta was 1.182 ms/token (20.3% of the 5.818 ms step); dp4a
removes 0.663 ms of it, leaving 0.519 ms.

**ncu, one kernel instance per arm** (`sudo /usr/local/cuda-13.0/bin/ncu --metrics
gpu__time_duration.sum,sm__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read.sum
--cache-control none -k regex:gqa_attn_split_partial -c 24 <bench>`, **collected as
root, the module parameter `RmProfilingAdminOnly` unchanged** — the recipe is
`CUDA-BACKEND-DESIGN.md` §7.2a). These are replayed/serialized durations, not the
nsys step times; read the *relations*:

| arm | duration µs | L1 load sectors | L2 read sectors | instructions |
|---|---|---|---|---|
| q8_0 incumbent `<2,1,0,0>` | 120.67 | 792904 | 59886 | 3242470 |
| q8_0 dp4a `<2,1,0,1>` | 89.82 | 792904 | 61142 | 2747486 |
| f16 `<1,1,0,0>` | 30.59 | 463008 | 107441 | 2436294 |

dp4a removes **494984 instructions (−15.3%)** and leaves the L1 load-sector count
**exactly unchanged** (792904 in both). The packed arm's L1 load sectors are
**1.71x** f16's while its L2 read sectors are **0.57x** f16's: a Q8_0 cell is 34-byte
blocks, so `kv4<Q8_0>` issues four separate byte loads plus a scattered f16 scale
load per four elements, each its own L1 request, and the L1 cannot merge them the
way it merges the f16 arm's two 8-byte `__half2` loads. **That is what the residual
1.242x is — an L1 request-count cost of the block layout, not DRAM traffic and no
longer the arithmetic.** It is also why dp4a recovers 56% of the load delta and not
all of it: it deletes the converts and multiplies, not the requests. (The same
34-byte-stride L1TEX mechanism is what doc 104 found for the q8_0 *weight* kernels.)

**Same-binary A/B** (`minfer bench -p 2048 -n 128 --n-ctx 4096 -o json`, control
`MINFER_NO_DP4A_Q8_KV=1`, 5 interleaved matched rounds, medians):

| model / config | incumbent | dp4a | ratio |
|---|---|---|---|
| 0.5B q4_0 `tg128` | 171.95 | **193.30** | **1.124x** |
| 0.5B q4_0 `pp2048` | 2149.22 | 2150.37 | flat (no regression) |
| Qwen3-0.6B Q8_0 `tg128` | 123.98 | **136.66** | **1.102x** |

The f16 arm measured in the same round is **240.03** on the 0.5B, so the packed/f16
residual went **1.396x → 1.242x**; on Qwen3-0.6B the f16 arm is 136.73, i.e. dp4a
closes the hd-128 packed decode gap. (The 236.97 in the ticket is a different round's
number and is not mixed into this comparison.)

**The tolerance class was re-measured, not assumed.**
`a_packed_kv_cache_answers_like_the_f32_one` (0.5B, CUDA arm, 9 steps, 37.82 spread)
reads max |Δlogit| **2.479504** and at the argmax **0.552662**, greedy 9/9 for dp4a;
the incumbent arm in the same binary reads **2.479504 / 0.596050 / 9-of-9**. Both are
inside the pinned class (tail ≤ 4.0, argmax ≤ 1.0), and the worst delta is unchanged
because it is the prefill step's (see §3.3).

**The gate provably reaches the dp4a kernel.** `nsys` on the real-model gate itself,
with and without the control, names the kernel that ran:

```
dp4a:     gqa_attn_split_partial<(int)2, (bool)1, (bool)0, (bool)1>  384 instances, 1980 ns
incumbent gqa_attn_split_partial<(int)2, (bool)1, (bool)0, (bool)0>  384 instances, 2326 ns
```

so the tolerance numbers above are not "both arms ran the old path".

**Gates.** `cuda_kv_q8_0_roundtrip_attn` grew a third arm: the decode (`nt = 1`) kernel
over **two** cells through an explicit span, with an exactly Q8_0-representable query
(`amax = 1`, values in `{-1, 0, 1}`) so the int dot's answer equals the f32 kernel's
over the dequantized cells, plus the observation-counter arm. Measured max |Δ| = 0.
The named Q8_0 gates stay green: `cuda_q8_0_store_matches_the_cpu_quantizer`,
`cuda_kv_q8_0_roundtrip_attn`, `cuda_q8_0_kv_cell_move_strides_by_row_bytes`, the Q8_0
arm of `cuda_map_window_matches_the_span_over_the_same_rows`,
`a_packed_kv_cache_answers_like_the_f32_one`,
`cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer`,
`cuda_q8_0_fa_prefill_attention_parity`.

**Mutations (gate contract rule 3).** With the corrected two-cell arm, changing
`kv4_q8_0_packed`'s block base from `elem >> 5` to `elem >> 4` turns the arm red:

```
the dp4a decode K dot is not the f32 kernel over the dequantized cells with an
exactly-representable query: max |Δ| = 0.35126442
test result: FAILED. 0 passed; 1 failed
```

(The counter arm's mutation is the control itself: `MINFER_NO_DP4A_Q8_KV=1` makes
`gqa_attn_split_q8_0` launch the `(bool)0` instantiation, the chokepoint is not
bumped, and the counter assertion fails. The A/B runs above use exactly that lever.)

## 5. Results

- **0.5B q4_0 `tg128` 171.95 → 193.30 tok/s (1.124x)**, medians of 5 interleaved
  same-binary rounds; the packed/f16 gap 1.396x → 1.242x.
- **Qwen3-0.6B Q8_0 `tg128` 123.98 → 136.66 (1.102x)** — parity with its own f16 arm
  (136.73).
- **`pp2048` flat** on both models (the decode kernel is not on the prefill path at
  hd 64; hd-128 prefill uses the FA kernel, which dp4a does not touch).
- The isolated load delta at hd 64 is cut by ~56-58%, and the real-path partial kernel
  by 41% (67.4 → 39.9 µs).
- **The residual is filed, not just named.** The ncu attribution in §4 leaves a
  concrete remainder — the packed arm's **1.71x** L1 load-sector count against f16's,
  with 0.57x its L2 sectors — and it is tracked as
  [#202](https://github.com/yusiwen/minfer/issues/202) with those four numbers, the exact
  kernel names and shape, the mechanism, and the candidate layouts (split/aligned quant
  plane, 36-byte block, 8-element loads). This is the piece between "the packed cache is
  a memory win" and "the packed cache is decode-useful".
- **Prefill / verify left alone, deliberately.** The `nt > 1` layout-tagged kernel
  (`gqa_attn_f32<...>`) also reads `kv4<Q8_0>`, and it measured **1.32x** slower than
  its f16 arm on the 0.5B prefill attention (32347 vs 24544 ns average, same nsys run).
  But that is a different kernel with a different query shape (per token **and** per
  head, no shared block scale), it is not the residual the ticket targets (the ticket's
  1.393x is `tg128`), and the bar was named on the decode step. Taking it would be a
  separate measurement; it is named here rather than assumed.

## 6. Lessons

1. **The mutation is what tells you the gate is vacuous.** The first decode arm looked
   correct — it ran the dp4a kernel, compared against the dequantized cells and passed
   with max |Δ| = 0 — but with one key the score it was testing could not reach the
   output. Only breaking the implementation exposed it. A green arm with a perfect
   bound is not evidence that the arm tests the thing.
2. **"The two arms read the same" is not "the arms are the same path".** The tolerance
   gate's *max* |Δlogit| was bit-identical between arms because the max is at the
   prefill step. The arm-ran evidence has to come from somewhere the arm changed: the
   per-step deltas, the argmax, the `nsys` kernel name, or the observation counter.
3. **A bar on the addressable share, not on the whole gap, keeps the decision honest.**
   The whole q8_0-vs-f16 kernel gap is geometry + load; only the load is addressable.
   Measuring at a shape where the two share one geometry is what made the 20.3% a
   load number rather than a geometry number.
4. **A 34-byte block layout is hostile to a packed load.** The three shifts and three
   ORs that assemble four int8 into one `__dp4a` operand are pure overhead a 4-byte
   aligned layout would not pay. That the win survives them is a property of this
   kernel's instruction/ latency bound, not a general rule.
5. **Deleting instructions is not the same as deleting requests.** dp4a removed 15.3%
   of the executed instructions and left the L1 load sectors untouched (792904 in
   both arms); the residual gap to f16 is 1.71x the L1 load sectors from the block
   layout, with *fewer* L2 sectors than f16. A counter set that had only gone after
   instruction count would have predicted a larger win than the hardware gave — the
   ncu pair is what names the remaining term.

← [107 · C4 packed Q8_0 KV on CUDA](./107-c4-packed-q8-kv-cuda.md) · [Index](./README.md)
