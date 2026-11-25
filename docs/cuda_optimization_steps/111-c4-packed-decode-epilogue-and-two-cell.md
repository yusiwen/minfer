# 111 · #633/#843: the packed Q8_0 decode residual falls 1.239x → 1.0139x — a 4-thread epilogue and two cells per warp (LANDED)

> **Result**: on Qwen2.5-0.5B q4_0 at hd 64 (`bench -p 2048 -n 128 -r 3 --n-ctx 4096`),
> q8_0 KV decode went **193.20 → 235.34 tok/s** against f16's 238.62, i.e. the packed/f16
> ratio **1.239x → 1.0139x**; the per-token gap fell **0.998 → 0.058 ms** (94%). Two levers:
> (1) the packed QKV epilogue's K/V sections were 4 threads each running a 32-element serial
> loop — **21.6 → 1.92 µs/layer**, bitwise; (2) the hd-64 packed split kernel ran two KV cells
> per warp with 16-lane groups — attention **38.53 → 20.58 µs/layer**. The 8 decode-step
> logits are byte-identical (0 of 151 936 floats differ per step) and 4/4 prompts × 24 greedy
> tokens match. The aligned-block option (#633 option 1) is rejected for this residual with
> record 104 + #202's arithmetic; its better-justified home is Metal (#844). Prefill remains
> 1.224x at 2k (#845).
> **Bar**: the bar lives in the tickets, not in this record.
> [#633](https://github.com/yusiwen/minfer/issues/633) (opened 2026-10-20) fixed the floor at
> parity with f16 — the split kernel at f16's own 18.4 µs/layer, a 2.1x cut — and required
> that any smaller bar be "stated first with its step-level arithmetic". The figures this
> change was held to are the ones [#843](https://github.com/yusiwen/minfer/issues/843) carries
> (epilogue ≤3.0 µs/layer with 123/123 + 65/65 bit-identical KV dumps, step ≤1.13x from E1
> alone, two-lever step ≤1.05x), and [#843](https://github.com/yusiwen/minfer/issues/843) was
> opened on 2026-11-18, *after* the 2026-11-17 measurement: for E1 those are a **post-hoc**
> target, and #633's parity floor is the only bar that predates the work. Written down on
> 2026-11-25 at [#842](https://github.com/yusiwen/minfer/issues/842)'s book hand-off, which
> asked for the provenance; this record should have carried it from the start. **MET** on every
> count: 1.92 µs/layer against ≤3.0, 1.0139x against ≤1.05x (and 1.1226x from E1 alone against
> ≤1.13x).
> **Date**: 2026-11-18. **Box**: dgxspark (aarch64, GB10 sm_121, driver 580.178.04, CUDA 13.0,
> ncu 2025.3.1). **Base**: `e844591` (+ this PR).

## 1. Where it stood

[#212](./110-c4-pack-q8-decode-stall-attribution.md) closed the *attribution* of the packed
decode residual: the split kernel was 2.13x f16's duration on 1.10x its instructions, the named
stall was long-scoreboard (10.09 vs 3.03 cycles/issued instruction), and both load-side levers
were null. It recorded 1.230x and handed the shape levers to [#633]. #633 named option 1 (a
4-byte-aligned block) and option 2 (more MLP: more splits, or two rows per warp with 16 lanes
each), priced both, and left them unbuilt.

What neither measured was the rest of the step. #212's nsys regex covered
`gqa_attn_split_partial$` only; the packed QKV epilogue was outside it.

## 2. The measurement

A/B on `e844591`, 3 alternating rounds per arm (9 samples), medians: f16 **239.37**, q8_0
**193.20** tok/s → **1.239x** (record 110's 1.230x reproduced). Whole-step nsys
(`--trace=cuda --cuda-graph-trace=node`, decode steps 8–128, kernel sums per step):

| per step | f16 | q8_0 | delta |
|---|---|---|---|
| step (ms) | 4.178 | 5.176 | **0.998** |
| kernels total (ms) | 2.711 | 3.690 | 0.979 |
| attention `gqa_attn_split_partial` | 0.451 | 0.952 | **0.501** |
| packed epilogue `attn_bias_rope_store_q8_0` | 0.041 | 0.518 | **0.477** |
| all other kernels | 2.219 | 2.220 | 0.001 |
| gap (step − kernels) | 1.467 | 1.486 | 0.019 |

The 0.47 ms "rest" was the packed epilogue: 2,880 launches (24/step) at **21.6 µs median**,
against the f16 epilogue's **1.7 µs**. Its q section's 448 threads finish in ~1.7 µs; the kernel
lasts as long as its K/V tail — **4 K threads + 4 V threads**, each running a 32-element serial
loop (loads, `powf`/`cosf`/`sinf` for K, bias + quantize for V).

## 3. E1 — the epilogue (#843)

**Fix.** One warp per K/V block: 32 lanes compute the block's 32 elements (one roped element
each), a warp max gives `amax`, then the same `q8_0_quantize_block` math runs per lane. The q
section is untouched. The epilogues are `attn_bias_rope_store_q8_0` (Qwen2) and
`attn_rope_store_q8_0` (Qwen3's no-bias arm) in `src/cuda/kernels/kv_store.cu`.

**Why the bytes are unchanged.** `amax` is a max over the same 32 values (order-independent),
the f16 scale and the per-element `rintf`/clamp are unchanged, and each lane writes its own
quant byte. `__shfl_xor_sync` with a full mask + `width = 16` sums within a 16-lane group, so
the grouping is the only structural requirement; the K/V section starts `qpairs` rounded up to
32 so every group is one hardware warp.

**Measured.** epilogue median **1.92 µs/layer** (avg 2.13, 768 launches), step **1.1226x**
(q8_0 213.35 vs f16 239.51). Bitwise: **123/123** dump files on the 0.5B, **65/65** on
Qwen3-0.6B, generated tokens identical.

## 4. E2 — two cells per warp (#633 option 2)

**Fix.** At hd 64 the packed split body left lanes 16–31 masked off. Group A (lanes 0–15) takes
the first half of the split's cell range, group B (16–31) the second; each lane still owns 4
dims. The dot reduction is 16-wide (`width = 16`), and the two halves' `(mx, S, oc)` are merged
with the split-combine formula (a group with no cells stays `mx = -INF, S = 0`; an invalid row
is masked out of the online softmax rather than branched, so the warp stays converged — the
first implementation produced `expf(-INF - -INF) = NaN` for an empty half and was caught by the
logit comparison before it could land). The dispatch is compile-time
`Q8DP4A && Q8WIDE && !MAP` plus a runtime `hd == 64`, so f16/f32, the bt prefill kernel and hd
128 are untouched.

**Measured.** attention median **20.58 µs/layer** (from 38.53), step **1.0139x** (q8_0 235.34 vs
f16 238.62). The merge is a different association, so the change is tolerance-class in
principle; on the measured cases the 8 decode-step logits are **byte-identical** to E1's
(0/151 936 per step), the prefill logits identical, 0.6B unchanged, and 4/4 prompts × 24 greedy
tokens match.

## 5. Option 1, rejected for this residual

A 4-byte-aligned block (36 B or the `p32` plane) buys a 47% load-instruction cut, but #202 cut
the packed kernel's L1 sectors by 74% and its load instructions by 66% for **+0.27%** on
`tg128`; record 104 rejected the padded-36B form for q8_0 weights (+6% traffic, no gain). The
packed kernel's cost is the per-lane V dequant chain, not the load count. The format lever's
better-justified home is Metal, where the native packed decode is **LSU-bound** (five 2-byte
loads vs one `half4`), tracked in #844.

## 6. Verification

- `cargo build --release --features cuda` — **pass**, every `nvcc`-accepted gencode target
  (`sm_75…sm_121` + `compute_121` PTX; CUDA 13 accepts no sm_70/72). The code uses
  `__shfl_xor_sync` width sub-sections, `if constexpr` and `__dp4a`; the full-arch build is the
  compile gate.
- `scripts/cuda_test.sh` — **866 passed / 0 failed / 62 ignored** (lib 652/0/21, bin
  168/0/35, integration 46/0/6).
- `scripts/real_model_gates.sh` with `MINFER_TEST_MODEL` = the 0.5B q4_0 and
  `MINFER_BATCH_TEST_MODEL` = Qwen3-0.6B Q8_0 — **35 passed / 0 failed / 0 ignored** (168
  filtered).
- The packed-cache tolerance gate
  (`models::qwen2::graph::tests::a_packed_kv_cache_answers_like_the_f32_one`, `--ignored`) —
  **pass**; the E2 tree also measured the 8 decode-step logits byte-identical to E1's.
- `pp2048` q8_0 2 177 vs 2 174 before E1, inside noise; decode 1.0139x.

## 7. Scope and leftovers

- **Prefill at hd 64** is unchanged: 1.229x at 2k, 1.46x at 8k (one round) — #845.
- **Metal** had its own residual, 1.20x (hd 64) / 1.10x (hd 128) — closed in record 113, where the premise
  'LSU-bound' was refuted: 82 % of it was the lost QKV fusion.
- **At 8k** packed decode reads 1.045x (175.35 vs 183.27 tok/s, one round): the dequant work
  scales with the token count, so the memory win does not become a speed win with context.
- The CPU path is not affected: its K dot is SIMD (`dot_q8_0_q8_0`), V is the scalar
  `accumulate_q8_0_row`, and it showed no such residual in an indicative A/B.

## 8. Lessons

1. **An attribution is not a decomposition.** #212's `19% share × 2.13x` explained the step
   ratio by inference; summing the step's kernels found half the gap in a kernel it never
   profiled. A residual claim should name the kernel table, not a share.
2. **A 1% estimate can be 9x wrong.** Record 107 priced the epilogue's transcendentals at ~1% of
   the step; the measured cost was 9%, because the estimate counted the extra `powf`s and not
   the 4-thread serial chain that exposes the loads.
3. **The cheap half was a defect, not a design trade.** The epilogue's one-thread-per-block
   mapping was chosen to share the CPU quantizer, but the warp form is byte-identical; the
   mapping did not have to be serial.
4. **"Changes the numerics" is a hypothesis until measured.** #633 predicted the 16-lane merge
   would be tolerance-class; on the measured cases the logits are byte-identical. The gate
   stays a tolerance battery, but the risk was overestimated.

← [110 · #212: the packed Q8_0 decode residual is a long-scoreboard chain](./110-c4-pack-q8-decode-stall-attribution.md) · [Index](./index.md)
