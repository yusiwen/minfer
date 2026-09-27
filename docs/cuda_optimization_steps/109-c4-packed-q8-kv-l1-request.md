# 109 · #202: the packed Q8_0 KV cell's L1 request count — cut 42.5%, the decode step did not move (LANDED — counter-only)

> **Result**: the packed decode's four 8-bit K/V quant loads became two 16-bit
> loads (no layout change). The packed/f16 **L1 load-sector ratio falls 1.71x →
> 0.98x** (792 904 → 455 840 sectors; f16 463 008) and the kernel sheds 2.2% of its
> instructions. The **same-binary `tg128` bar named before measuring (>= 2%) was
> not cleared: +0.27%** (193.13 → 193.65 tok/s, medians of 5 interleaved rounds),
> and the kernel itself is only 1.7% faster (nsys). The L1 request count is
> therefore **not** what the packed decode's 1.23x residual is bound by; the
> counter moves exactly as #202 predicted and the time does not follow.
> **Commit**: `<this PR>` (`src/cuda_kernels.cu`, `src/cuda.rs`,
> `tests/fixtures/cuda_launch_sites.tsv`). **Date**: 2026-09-27.

## 1. Background — where things stood

[#186](https://github.com/yusiwen/minfer/issues/186) (record [108](./108-c4-dp4a-packed-q8-kv-cuda.md))
landed the dp4a packed Q8_0 K dot and cut the packed/f16 decode gap from 1.396x to
1.242x. Its ncu attribution pinned the remainder on one counter: at hd 64 / nh 14 /
nk 2 / nkv 2049-2176 the packed `gqa_attn_split_partial` issued **1.71x** the f16
arm's L1 load sectors (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`, 792 904 vs
463 008) while reading **0.57x** its L2 sectors (61 142 vs 107 441). dp4a removed
494 984 instructions and left the L1 count **bit-identical**, so the count was
attributed to the **layout**, not the arithmetic:

> A Q8_0 block is `f16 d; i8 qs[32]` = 34 bytes ... the 4-element group the dot
> needs lives at `blk + 2 + (elem & 31)`, which has **no 4-byte alignment
> guarantee** (`34 * k` alternates parity), so `kv4<Q8_0>` cannot do one 32-bit
> load: it issues four separate byte loads, plus a fifth, scattered f16 scale load
> per group.

That attribution is [issue #202](https://github.com/yusiwen/minfer/issues/202),
which names three candidate fixes in the order the mechanism suggests: split the
scale plane from the quants and align the quants (doc 104's
[`q80-p32-split-plane`](./104-q80-p32-split-plane.md) pattern applied to the KV
cell), widen the block to 36 bytes with the quants at offset 4, or **load the
quants 8 at a time where the group is 8-element aligned — which needs no layout
change at all**. The ticket says explicitly: if the third clears the bar, prefer
it, because a layout change drags the CPU store/read path, the `copy_cells`
stride, `map_q8_0_cells`, the FA-prefill staging and the **C5 session format**
with it.

This step takes the no-layout candidate, measures it, and reports what the
measurement says: the counter moves further than the ticket asked for, and the
throughput does not move at all.

## 2. Principle — the GPU mechanism, and the part of it that was wrong

**The mechanism, restated as a load-instruction budget.** The packed decode body
stages four rows of K and V per iteration; per lane and per row it needs four
quants and one block scale from each of K and V. The incumbent accessor
(`kv4<KV_LAYOUT_Q8_0>`, and `kv4_q8_0_packed` for the dp4a K side) spends **five**
load instructions per group: one 16-bit scale plus four `signed char`. The f16 arm
spends two (`__half2` pairs) for the same four elements, and its scale is implicit.
That is the 10-vs-4 instructions-per-lane-row gap whose sector footprint is the
1.71x.

**The candidate, and why it is exact.** The four bytes are at
`blk + 2 + (elem & 31)`, which is 4-byte aligned only when the block index is odd
(`34k mod 4 = 2k mod 4`). But it is **always 2-byte aligned**: `34k` is even,
`2 + 4m` is even. Two `unsigned short` loads therefore fetch the same four bytes
in half the requests, and no layout change, no padding and no alignment promise
are needed:

| load arm | instructions per 4-element group | bytes moved |
|---|---|---|
| incumbent K (dp4a) | 1 x u16 (scale) + 4 x s8 = **5** | 6 |
| incumbent V (convert) | 1 x u16 (scale) + 4 x s8 = **5** | 6 |
| wide K/V (this step) | 1 x u16 (scale) + 2 x u16 = **3** | 6 |

The bytes are identical and the assembled `int` is identical, so this is an
access-pattern change, not a numerics change. Ten load instructions per lane-row
become six.

**The part the arithmetic missed.** Reducing the *request count* is only worth
time if the kernel was bound by L1 request throughput. The arithmetic above says
the packed accessor issues ~2.5x f16's load instructions; it does not say the
kernel waits on them. #186's own numbers already hinted at the alternative: the
packed kernel was **2.15x** f16's time (39.9 vs 18.5 us) while its **instruction
count was only 1.13x** and its **L2 traffic 0.57x**. A 2.15x duration from ~1.1x
instructions and *fewer* bytes is a **latency** signature — a long
load -> convert -> multiply chain in a single-warp block — and latency is exactly
what neither the request count nor the sector count measures. This step is the
experiment that separates the two: change the requests, hold everything else
(geometry, arithmetic, bytes, numerics) fixed, and read both counters.

## 3. Implementation

### 3.1 Design choices (why this shape and not a layout change)

- **No layout moves.** The candidate is bit-exact and touches one accessor. The
  alternatives both change the 34-byte block: a split plane changes where the
  blocks live, a 36-byte block changes `KvFormat::Q8_0.row_elems`, the CPU
  `pack_q8_0_cell_into`/`accumulate_q8_0_row`/`map_q8_0_cells`, the `copy_cells`
  stride, the FA-prefill staging `kv8_q8_0` and the **C5 session container**
  (a version bump plus a refusal for older packed files, and a redefinition of what
  `cuda_q8_0_store_matches_the_cpu_quantizer` asserts). The measurement below shows
  none of that would have bought throughput, so the small blast radius is also the
  correct one — ex post.
- **A compile-time arm, selected once per process.** `bool Q8WIDE` joins
  `attn_split_1w_body`/`gqa_attn_split_partial` as a template parameter, and
  `cuda::q8_kv_wide_enabled()` resolves `MINFER_NO_Q8_KV_WIDE=1` once per process
  and passes it as a value — the same captured-graph discipline #186 used for
  `dp4a` (a flag that flipped mid-process would select a different kernel than the
  one the captured exec recorded). The `wide` arm is only meaningful on the dp4a
  arm (the convert arm's K goes through `kv4<Q8_0, false>` for V and the
  `__dp4a`/convert split for K), so the launcher gates it as `dp4a && wide`.
- **One source per arm, not two bodies.** `q8_0_load4_bytes` / `q8_0_load4_wide`
  are two small device functions behind `q8_0_load4<WIDE>`, and
  `kv4_q8_0_impl<WIDE>` is the one dequant body. The incumbent arm's instruction
  stream is unchanged (`kv4<Q8_0, false>` is the old specialization verbatim), so
  the control is the pre-change accessor, not a re-derivation of it.
- **Three new launch sites, driven.** The three window modes (map / span / causal)
  get a `wide` instantiation each
  (`launch:gqa_attn_split_q8_0__{map,span,causal}_wide`). The #162 audit requires
  each `<<<` to name its own site and read its own error, and
  `cuda_issue162_...` drives every site, so the driver gained a third arm and
  `tests/fixtures/cuda_launch_sites.tsv` was regenerated.

### 3.2 Key code

The two accessors, from `src/cuda_kernels.cu`:

```cpp
// The incumbent: four separate `signed char` accesses.
__device__ __forceinline__ void q8_0_load4_bytes(
    const unsigned char* blk, int off, int& q, float& d) {
    d = __half2float(*reinterpret_cast<const __half*>(blk));
    const signed char* p = reinterpret_cast<const signed char*>(blk + 2) + off;
    q = ((int)(unsigned char)p[0]) | ((int)(unsigned char)p[1] << 8) |
        ((int)(unsigned char)p[2] << 16) | ((int)(unsigned char)p[3] << 24);
}

// #202: the same four bytes in HALF the load instructions. `34 * k + 2 + 4m` is
// even for every block index `k` and every 4-element-aligned offset `4m`, so a
// group is always 2-byte aligned even though it is only 4-byte aligned when `k`
// is odd.
__device__ __forceinline__ void q8_0_load4_wide(
    const unsigned char* blk, int off, int& q, float& d) {
    d = __half2float(*reinterpret_cast<const __half*>(blk));
    const unsigned char* p = blk + 2 + off;
    const unsigned lo = *reinterpret_cast<const unsigned short*>(p);
    const unsigned hi = *reinterpret_cast<const unsigned short*>(p + 2);
    q = (int)(lo | (hi << 16));
}
```

and the body picks them with `q8_0_load4<Q8WIDE>` for both K and V:

```cpp
            if (Q8DP4A) {
                if (use) kv4_q8_0_packed<Q8WIDE>(
                    kv_row(k, cell[j], row_bytes), hk * hd + d0, ki4[j], kd4[j]);
                ...
            } else {
                k4[j] = use ? kv4<LAYOUT, Q8WIDE>(kv_row(k, cell[j], row_bytes), hk * hd + d0)
                            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
            v4[j] = use ? kv4<LAYOUT, Q8WIDE>(kv_row(v, cell[j], row_bytes), hk * hd + d0)
                        : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
```

### 3.3 Pitfalls

- **The vec4 V load is not the cheap one.** It is tempting to assume the V side is
  a single `float4`; it is not — `kv4<KV_LAYOUT_Q8_0>` is the same five-instruction
  accessor for V, so the wide arm has to cover both K and V to halve the budget
  (10 → 6). Applying it to K alone would have left V's byte loads in place.
- **The `kv4` template needed a second parameter, not a second function.**
  `kv4<LAYOUT>` is an explicit-specialization set; a partial specialization for
  `Q8_0` was not available, so the F32/F16 specializations are now
  `kv4<*, false>` and `Q8_0` has `false`/`true` specializations over the one
  `kv4_q8_0_impl<WIDE>` body. The f16 arm's SASS is unchanged.
- **A kernel-name regex has to be anchored.** `ncu -k regex:gqa_attn_split_partial`
  also matches `..._combine` and `..._bt`; the collection uses
  `regex:gqa_attn_split_partial$`, and ncu's available-kernel list shows the base
  name (no template arguments) — a regex that includes `<` silently profiles
  nothing and prints an "Available Kernels" list instead.

## 4. Verification

Provenance for every number: **GB10 sm_121, CUDA 13.0, driver 580.178.04,
2026-09-27**, `cargo build --release --features cuda` at this worktree (base
5f3f26e), model
`/home/yusiwen/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf`.

**The bar, named before measuring.** (a) the packed/f16 L1 load-sector ratio must
fall from 1.71x to **<= 1.30x**; (b) the same-binary `tg128` median must beat the
#186 dp4a baseline **193.30 tok/s by >= 2%** (>= 197.2), with `pp2048` within 2% of
the control. (b) was justified as the load-attributable decode residual after #186
(0.519 ms of a ~5.2 ms/token step, 10%) times the ~40% of load instructions this
removes.

**ncu, one kernel instance per arm**
(`sudo /usr/local/cuda-13.0/bin/ncu --metrics
gpu__time_duration.sum,sm__inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,lts__t_sectors_op_read.sum
--cache-control none -k 'regex:gqa_attn_split_partial$' -c 24 <bench> -p 2048 -n 128
--n-ctx 4096`, **collected as root, the module parameter `RmProfilingAdminOnly`
unchanged at `1`** — `cat /proc/driver/nvidia/params`; the recipe is
[`CUDA-BACKEND-DESIGN.md`](../CUDA-BACKEND-DESIGN.md) §7.2a). The values are
constant across all 24 replays for the two counters that matter; the durations are
replay-serialized and are **not** quoted as absolutes — only relations.

| arm | kernel | instructions | L1 load sectors | L2 read sectors |
|---|---|---|---|---|
| incumbent Q8_0 (from #202) | `<2,1,0,0>` | 3 242 470 | 792 904 | 59 886 |
| dp4a byte (control, `MINFER_NO_Q8_KV_WIDE=1`) | `<2,1,0,1,0>` | 2 747 486 | 792 904 | 59 842 |
| **dp4a wide (this step)** | `<2,1,0,1,1>` | **2 687 888** | **455 840** | 59 815 |
| f16 | `<1,1,0,0,0>` | 2 436 294 | 463 008 | 106 057 |

The control reproduces #202's dp4a row **exactly** (2 747 486 / 792 904), which is
what makes the comparison same-binary. The wide arm removes **59 598 instructions
(−2.17%)** and **337 064 L1 load sectors (−42.5%)**, so

- packed/f16 L1 ratio: **792 904 / 463 008 = 1.712x before, 455 840 / 463 008 =
  0.9845x after** — the packed arm now issues *fewer* L1 load sectors than f16;
- packed/f16 L2 ratio is unchanged (0.565x before, 0.564x after): the same bytes
  are read, only the request shape changed.

Bar (a) is met with margin.

**nsys, the real decode path** (`MINFER_CACHE_TYPE=q8_0 nsys profile --trace=cuda
--cuda-graph-trace=node ... bench -p 2048 -n 128 --n-ctx 4096`, 15 360 instances =
24 layers x 128 tokens x 5 passes):

| arm | kernel | avg ns | med ns | stddev ns |
|---|---|---|---|---|
| control | `<2,1,0,1,0>` | 41 053.5 | 40 320 | 7 276 |
| wide | `<2,1,0,1,1>` | 40 353.3 | 39 744 | 6 484 |

The kernel is **1.7% faster on average, 1.4% on the median** — real, same-sign in
both statistics, and far smaller than the 42.5% request cut. 15 360 launches x
~700 ns saved = ~10.7 ms over the traced run, which is ~0.3% of the 128 x 3-token
tg timing — i.e. it exactly predicts the whole-step result below.

**Same-binary A/B** (`minfer bench -p 2048 -n 128 --n-ctx 4096 -o json`, 5
interleaved matched rounds, medians; `tg128` tok/s):

| round | control | wide | f16 |
|---|---|---|---|
| 1 | 193.48 | 193.74 | 239.00 |
| 2 | 193.68 | 193.65 | 239.00 |
| 3 | 193.13 | 193.54 | 237.51 |
| 4 | 193.03 | 193.73 | 238.20 |
| 5 | 192.77 | 193.59 | 238.58 |
| **median** | **193.13** | **193.65** | **238.58** |

`wide/control = 1.0027` (**+0.27%**), `pp2048` medians 2 137.44 vs 2 146.97 (flat,
inside the round-to-round spread). **Bar (b) was not cleared.** The f16 arm in the
same binary is 238.58, so the packed/f16 ratio moved **1.235x → 1.232x**:
statistically unmoved.

**The named gates stay green.** The full CUDA suite is
`scripts/cuda_test.sh` (serial, `--test-threads=1`); the Q8_0 device set is
`cuda_q8_0_store_matches_the_cpu_quantizer`, `cuda_kv_q8_0_roundtrip_attn`,
`cuda_q8_0_kv_cell_move_strides_by_row_bytes`, the Q8_0 arm of
`cuda_map_window_matches_the_span_over_the_same_rows`,
`a_packed_kv_cache_answers_like_the_f32_one`,
`cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer` and
`cuda_q8_0_fa_prefill_attention_parity` (counts in §5).

**The tolerance class was re-measured, not assumed.**
`a_packed_kv_cache_answers_like_the_f32_one` (0.5B, CUDA arm) — measured numbers in
§5; the wide arm is byte-identical to the byte-load arm by construction and the
gate compares each arm against the **dequantized cell**, not against the other
accessor.

**Mutation evidence (gate contract rule 3).** Swapping the two 16-bit halves in
`q8_0_load4_wide` (`q = (int)(hi | (lo << 16));`) turns
`cuda_kv_q8_0_roundtrip_attn` red on the dp4a decode arm:

```
the dp4a decode K dot is not the f32 kernel over the dequantized cells with an
exactly-representable query: max |Δ| = 2.5326836
test result: FAILED. 0 passed; 1 failed
```

(The `nt = 3` arm above it stays green because it runs the general
`gqa_attn_f32<Q8_0>` kernel, which still takes the byte form — so the failing arm
is exactly the one the mutation touches, not a shared path.)

## 5. Results

- **Primary acceptance (#202): MET.** The packed/f16 L1 load-sector ratio is
  **1.712x → 0.9845x** (792 904 → 455 840; f16 463 008), and the packed arm's
  instruction count is 2.17% lower. This is the acceptance criterion the ticket
  states, and it is met without a layout change, a session-format bump or any
  CPU-side edit.
- **Secondary acceptance (the same-binary A/B bar, >= 2%): NOT MET.** `tg128`
  193.13 → 193.65 (**+0.27%**), and the kernel is 1.4-1.7% faster. The measured
  result is a **partial refutation**: the L1 request count is *not* what the packed
  decode's 1.23x residual is bound by.
- **What the residual is not.** Not L2 traffic (0.564x f16, and unchanged), not
  instruction count (1.10x f16 after this step, from 1.13x), and — as of this
  step — not L1 load sectors (0.98x f16). The remaining candidate is the
  **latency of the per-lane chain**: a separate per-block scale load, a `cvt`, and
  the V-side dequant multiply sit on the critical path of a single-warp block that
  stages only four rows, where the f16 arm's two `__half2` loads and
  half2-to-float2 converts are shorter and more regular. That is a hypothesis for
  a follow-up ncu *stall* attribution, not a claim made here.
- **No layout change was taken, and the measurement says none was warranted.**
  The p32 split plane and the 36-byte block both change the CPU store/read path,
  the `copy_cells` stride, `map_q8_0_cells`, the FA-prefill staging and the C5
  session version — and the cheapest candidate already drives the L1 counter to
  0.98x without moving throughput. A layout change could not have been justified by
  this measurement.
- **Real-model counts and tolerance class** (GB10 sm_121, `FEATURES=cuda
  scripts/real_model_gates.sh`, both configurations, 2026-09-27): §5.1.

### 5.1 Suite and tolerance records

- CUDA unit, GB10 sm_121, `scripts/cuda_test.sh`, 2026-09-27: **562 / 0 / 42**
  (serial; + 7 / 0 / 0 and 3 / 0 / 6 integration). The row was refreshed here from
  #196's 548 / 0 / 39: #140's K-quant encoder tests (7 passed / 1 ignored) and
  #142's bf16 writer + CPU-path tests (7 passed / 2 ignored) account for the whole
  +14 / +3; **this step adds no test**, so it does not move the row.
- CUDA real-model set (0.5B), GB10 sm_121, `FEATURES=cuda
  scripts/real_model_gates.sh`, 2026-09-27: **42 / 0** (was 39 / 0).
- CUDA real-model set (Qwen3-0.6B), same box and command with
  `MINFER_BATCH_TEST_MODEL=.../Qwen3-0.6B-Q8_0.gguf`, 2026-09-27: **42 / 0**.
- `compute-sanitizer --tool memcheck` over the CUDA unit suite, GB10 sm_121,
  `scripts/cuda_test.sh` under the sanitizer, 2026-09-27: **0 API errors**
  (562 / 0 / 42).
- `a_packed_kv_cache_answers_like_the_f32_one`, CUDA arm, 0.5B, 2026-09-27
  (`cargo test --release --features cuda a_packed_kv_cache_answers_like_the_f32_one
  -- --ignored --nocapture --test-threads=1`): max |Δlogit| **2.479504** of a
  37.821205 spread, |Δ| at the reference argmax **0.5526619** (worst step 0.5526619,
  first step 0.27636337), greedy **9/9** — **bit-identical to #186's recorded
  values**, which is what the byte-identical accessor predicts. Inside the pinned
  class (tail <= 4.0, argmax <= 1.0). The CPU arm in the same run reads 3.0289202 /
  0.64604187 / 9-of-9.

## 6. Lessons

1. **A counter can move exactly as predicted and buy nothing.** The L1 load-sector
   count fell 42.5% in the direction #202's mechanism demanded, and the decode
   step moved +0.27%. The mechanism was right about the accessor and wrong about
   the bottleneck: a sector count is a *work* measure, and a kernel can be bound by
   *latency* with a small, fixed dependency chain.
2. **Do not infer a bottleneck from a ratio of counters.** #186's own table
   contained the warning: 2.15x duration from 1.13x instructions and 0.57x bytes.
   That is a latency signature, and the right next instrument is a stall/memory-latency
   attribution, not another request-count reduction.
3. **Name the throughput bar before measuring, even when the ticket's stated
   acceptance is a counter.** The counter criterion here is met outright; only the
   pre-registered throughput bar exposes that the ticket's *purpose* ("the
   difference between 'the packed cache is a memory win' and '... decode-useful'")
   is not advanced. Landing with that stated is the honest outcome; widening the bar
   after the fact would have hidden it.
4. **The cheapest candidate can also be the correct one ex post.** The no-layout
   option was preferred on blast-radius grounds before measuring; the measurement
   then showed the layout alternatives would have moved the same counter and the
   same (zero) throughput, at a much higher cost.

← [108 · #186: the dp4a packed Q8_0 K dot on CUDA](./108-c4-dp4a-packed-q8-kv-cuda.md) · [Index](./README.md)
