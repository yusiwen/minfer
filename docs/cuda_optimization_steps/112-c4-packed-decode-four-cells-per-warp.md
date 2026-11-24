# 112 · #847: the packed Q8_0 decode passes f16 — four KV cells per warp at hd 64 (LANDED)

> **Result**: on Qwen2.5-0.5B q4_0 at hd 64 (`bench -p 2048 -n 128 -r 3 --n-ctx 4096`,
> 3 alternating rounds, medians), q8_0 KV decode reads **243.35 tok/s** against f16's
> **236.18** — the packed/f16 ratio **1.0139x → 0.9705x**, i.e. the packed cache is now
> **3.0% faster** than f16 at the same step (4.2341 → 4.1093 ms/token) instead of 1.4%
> slower. The lever is E2's, applied once more: the hd-64 packed split kernel went from two
> KV cells per warp (two 16-lane groups) to **four cells per warp (four 8-lane groups, 8 dims
> per lane)**, so a warp's serial cell iterations halve again (8.5 → ~4.25) and the K
> reduction drops from 4 shuffles to 3. Kernel medians (nsys, 48 launches):
> `gqa_attn_split_partial` **20.65 → 13.34 µs/layer**, `gqa_attn_split_combine` 2.94 → 2.94
> (untouched). The win grows with context: at 8k the decode ratio goes **1.044x → 0.906x**
> (one round). The pre-registered bar (partial ≤17.0 µs/layer, step ≤1.005x, tolerance
> class) is cleared on both rungs. The 4-way merge is **not** byte-identical to the 2-cell
> arm (max |Δlogit| 0.48 whole-vector), but each arm sits **2.48 / 2.50** from f32 and the
> greedy continuation still agrees 9/9 — the association moves a result inside the
> quantization noise, not the accuracy class. hd 128, f16/f32 and the prefill body are
> untouched (`hd == 64` dispatch); prefill stays 1.232x at 2k (#845).
> **Date**: 2026-11-18. **Box**: dgxspark (aarch64, GB10 sm_121, driver 580.178.04,
> CUDA 13.0). **Base**: `d4818f9` (+ this change).

## 1. Where it stood

Record [111](./111-c4-packed-decode-epilogue-and-two-cell.md) left the packed decode at
**1.0139x** f16 and named the remainder exactly: the partial kernel at **20.65 µs/layer**
against the f16 arm's **18.27**, with the combine at parity (2.94 vs 2.82). One microsecond
per layer is 24 µs per step; the residual step gap was 58 µs of 4 191, so the whole
remaining cost was that 2.38 µs/layer of split-kernel time.

The kernel is **not** bandwidth-bound, and the arithmetic says so: the launch is
`dim3(32, 14)` = **448 one-warp blocks over 48 SMs** — 9.3 warps/SM, ~19% of the 48 warp
slots, the occupancy record 110 measured — and the work per layer at that grid is on the
order of a microsecond of warp instructions against a 20.65 µs wall. What is left is the
latency of each cell's serial chain, and the operand that moved it before was *cells in
flight per warp*.

## 2. The lever, and why this one

Two ways to raise the concurrency:

- **More splits** (`ATTN_SPLITS` 32 → 64/128). It multiplies the partials buffer by the
  same factor, and `gqa_attn_split_combine` walks the splits **serially** (`for sp in
  0..ATTN_SPLITS`, one thread per output dim). At 32 splits it is already 2.94 µs; at 128
  it would be ~12 µs of new cost against ~10 µs saved. Rejected as the first rung.
- **More cells per warp**, which is what E2 did (38.53 → 20.58 µs/layer for a 1.87x on the
  kernel) and what this record does again: the warp's 32 lanes split into **four 8-lane
  groups**, each owning one cell range of the split, each lane holding **8 dims**.

The same lever, one step further, and the combine is not touched at all.

## 3. The change

`attn_split_1w_body_q8_4cell` (`src/cuda/kernels/attention_decode.cu`) replaces the 2-cell
body behind the same `Q8DP4A && Q8WIDE` + `hd == 64` dispatch, so f16/f32, the bt prefill
body and hd 128 keep their kernels:

- `g = lane_id >> 3`, `li = lane_id & 7`, `d0 = li * 8`; the split's cell range divides by
  four (`quart = (chunk + 3) / 4`) instead of two.
- The K dot is **two `dp4a`** per lane (dims `d0..d0+3`, `d0+4..d0+7`) summed in `int` —
  exact — then one multiply by `qscale * kd`, then an **8-wide** reduction (3 `shfl_xor`
  instead of 4).
- The query quantization is unchanged **by construction**, not by luck: a Q8_0 block is
  still the same 32 dims, now four lanes instead of eight, so `amax` covers the same 32
  values and every dim's quant byte is identical. Only the lane assignment moved.
- The four groups' `(mx, S, oc)` merge pairwise (offset 8, then 16) with the split-combine
  formula, the same guard against `expf(-INF - -INF)` the 2-cell body uses.
- The body is **parameterized on `MAP`**, because a set-valued window names its rows through
  `kv_cell_left`/`kv_cell` instead of `row0 + i`. The first cut was non-MAP only (the
  dispatch gate was `!MAP`) and that asymmetry is exactly what
  `graph::cuda_backend::tests::attn_window::cuda_map_window_matches_the_span_over_the_same_rows`
  caught: the span path took the 4-cell body while the map path stayed on the 1-warp body,
  and the two disagree by **1.2e-7** (one fp32 ULP) on 6 of the q8_0 cases. The gate is right
  to demand bitwise there — same bytes, same rows, one kernel — so the body took the map
  walk rather than the gate taking a tolerance.

## 4. Measured

Kernel medians (nsys, 48 launches = 24 layers × 2 passes, `tg32` window, `q8_0`):

| kernel | 2-cell | 4-cell |
|---|---|---|
| `gqa_attn_split_partial` | 20.65 µs/layer | **13.34 µs/layer** |
| `gqa_attn_split_combine` | 2.94 µs | 2.94 µs |

Step, `bench -p 2048 -n 128 -r 3 --n-ctx 4096`, 3 alternating rounds (per-round tok/s in
brackets, median quoted):

| arm | tok/s | vs f16 |
|---|---|---|
| q8_0, 2-cell | **235.18** (235.68 / 235.18 / 234.09) | 1.0137x |
| q8_0, 4-cell | **243.35** (243.35 / 243.65 / 242.65) | **0.9705x** |
| f16 | **236.18** (236.18 / 236.79 / 236.02) | 1.0000x |

4.2341 → 4.1093 ms per token: **−125 µs/step**, of which the kernel accounts for
24 × 7.31 = 175 µs (the rest is session-to-session clock drift, which the interleaved rounds
carry). An earlier 3-round set, taken before the `MAP` generalization, read **246.02** vs
**238.41** = **0.9691x** — the same ratio on a ~1% faster session, which is the reason the
ratio, not the absolute tok/s, is the claim.

Prefill is unchanged: `pp2048` 2 163.1 tok/s against f16's 2 664.6, i.e. **1.232x** (inside
the 1.224–1.244 spread records 107/111 quote).

**At 8k** (`bench -p 8192 -n 32 -r 1 --n-ctx 16384`, one round): decode **174.33 → 201.03**
tok/s against f16's 182.05, i.e. **1.044x → 0.906x**; prefill 562.5 → 562.5 against 794.9
(1.413x). The gain grows with the cell count, which is the signature of latency hiding
rather than bandwidth.

## 5. Numerics

The 4-way merge is a different association again, and unlike E2 it is **not** byte-identical
to its predecessor: on "Count from one to five." the 8 decode-step logits differ from the
2-cell arm by at most **0.4759** whole-vector (every one of the 151 936 floats moves), with
the argmax agreeing at every step.

That is the quantization noise, not a new error class. Against the f32 arm the tolerance
gate reads **2.4795** max |Δlogit| of a 37.821 spread, **0.5201** at the reference argmax
and **9/9** greedy agreement (the CPU arm: 3.0289 / 0.6460 / 9/9) — the 2-cell arm measured
2.50 / 0.60 in record 111. So E2's byte-identity was a property of *that* merge landing on
the same bits, not a property of the technique; record 111's lesson 4 ("changes the
numerics is a hypothesis until measured") gains a corollary: so is *unchanged* numerics.

## 6. Verification

- `cargo build --release --features cuda` — **pass**, every `nvcc`-accepted gencode target
  (`sm_75…sm_121` + `compute_121` PTX; CUDA 13 accepts no sm_70/72).
- `scripts/cuda_test.sh` — **866 passed / 0 failed / 62 ignored** (lib 652/0/21, bin
  168/0/35, integration 46/0/6).
- `scripts/real_model_gates.sh` with `MINFER_TEST_MODEL` = the 0.5B q4_0 and
  `MINFER_BATCH_TEST_MODEL` = Qwen3-0.6B Q8_0 — **35 passed / 0 failed / 0 ignored** (168
  filtered).
- The packed-cache tolerance gate (`a_packed_kv_cache_answers_like_the_f32_one`,
  `--ignored`) — **pass**; numbers in §5.
- `graph::cuda_backend::tests::attn_window::cuda_map_window_matches_the_span_over_the_same_rows`
  — **pass**, i.e. the map and span window encodings agree **bitwise** on the same rows
  through the new body (§3).
- Greedy continuations on 4/4 prompts × 24 tokens match the 2-cell arm; "Count from one to
  five." and "Why is the sky blue?" read coherently.

## 7. Scope and leftovers

- **hd 128** is untouched by this change and stays the geometry-confounded pair of
  [#848](https://github.com/yusiwen/minfer/issues/848).
- **Prefill at hd 64** is unchanged: 1.232x at 2k, 1.413x at 8k —
  [#845](https://github.com/yusiwen/minfer/issues/845).
- **Metal** had its own residual, 1.20x / 1.10x — closed in [113](./113-c4-packed-decode-metal-fused-epilogue.md)
  ([#844](https://github.com/yusiwen/minfer/issues/844)).
- The split count is untouched, so the combine kernel is still the serial-over-splits shape;
  raising it is only worth revisiting if the partial kernel drops another large step.
- **The two levers [#847](https://github.com/yusiwen/minfer/issues/847) was written around are
  not needed for hd 64 any more.** An int8 V accumulator via `dp4a` would have changed the
  numerics and the aligned 36 B / `p32` block would have changed the format; the residual
  closed on *occupancy*, with the cell bytes and the V arithmetic untouched. Lever 1 stays
  the answer if a longer context ever brings the residual back; lever 2 stays Metal's
  ([#844](https://github.com/yusiwen/minfer/issues/844)), where the load shape is the binder.
- **Headroom, priced** ([the ticket's analysis](https://github.com/yusiwen/minfer/issues/847#issuecomment-6975335468)):
  the packed attention reads 99.4 MB/step in 391 µs = **254 GB/s** against the GB10's ~273 GB/s
  peak, so it sits at the memory floor; the f16 arm's 370 GB/s is the L2 reuse GQA's 7:1 head
  sharing gives it and the packed arm does not get. Free attention would put the ratio at
  0.878, f16's *effective* bandwidth at **0.942**, and the two remaining routes are spent
  (eight cells per warp spills registers, more splits pays the serial combine where it saves).
  Realistic ceiling here: **0.955–0.960**. The other 88% of the step is work both arms share,
  running at ~100 GB/s — 36% of peak — i.e. a decode-occupancy campaign, not this ticket.
- [#847](https://github.com/yusiwen/minfer/issues/847) closes with this record: the hd-64
  decode residual is closed, not merely re-stated.

## 8. Lessons

1. **The same lever has a second notch.** E2 halved the serial iterations and got 1.87x;
   halving them again got 1.55x. The decay is visible (13.34 vs the 10.3 a clean 2x would
   give) and a third notch would not pay — the fixed cost is per-warp startup, not per cell.
2. **Occupancy here comes from the grid, not from the bytes.** 448 blocks over 48 SMs is 19%
   of the machine; a kernel whose stall table says "latency" cannot be fixed by moving fewer
   bytes, which is why #202's load levers were null and why the cell count per warp was the
   thing that worked.
3. **A body applied to one window encoding is a body that lies about the other.** The
   map/span gate is only cheap to keep while both encodings run the *same* kernel; the
   moment one instantiation diverges the paths disagree in the last bit, and a tolerance
   there would have hidden the asymmetry instead of naming it. Run the gate before the
   benchmark, not after.
4. **Byte-identity is a property of a merge, not a technique.** E2 measured 0 of 151 936
   floats moving; the same idea one notch further moves all of them and still lands in the
   same class. Quote the class, treat identity as a bonus.

← [111 · #633/#843: the packed Q8_0 decode residual falls 1.239x → 1.0139x](./111-c4-packed-decode-epilogue-and-two-cell.md) · [Index](./index.md)
