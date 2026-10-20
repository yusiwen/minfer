# 110 · #212: the packed Q8_0 decode residual is a long-scoreboard chain, and no load-shape change moves it (MEASURED — LEVER REFUTED, CODE REVERTED)

> **Result**: at hd 64 / nh 14 / nk 2 / nkv 2049-2176 the packed decode split kernel
> is **2.13x** the f16 kernel's duration (nsys median 39 232 vs 18 400 ns) while
> issuing only **1.10x** its instructions. The named stall is
> **long-scoreboard**: **10.09** vs **3.03** cycles per issued instruction on the
> same-geometry f16 arm (medians of 3, ncu). The load shape does not set it:
> the byte-load control issues **1.66x** the load instructions and **1.74x** the L1
> sectors for the same bytes and runs **+2.4% slower** (nsys), and a
> deferred-conversion staging form — loads kept raw, conversions moved into the
> consumer — is **bit-identical** (123/123 dump files), **+0.09%** on `tg128` in its
> runtime-gated form (5 interleaved rounds), and **−2.98%** when rebuilt clean
> without the gating branch, its kernel **+17.7%** slower (nsys median 46 592 vs
> 39 584 ns). Both levers were measured against a bar named first; the staging arm
> was then **reverted**, because it buys nothing and the clean form is a
> regression. What remains is the per-lane
> chain itself inside a 1-warp block at ~9.3 warps/SM, and closing it needs a
> numerics- or format-changing reshape, which is a different ticket's decision.
> **Date**: 2026-10-20. **Box**: dgxspark (aarch64, GB10 sm_121, driver 580.178.04,
> CUDA 13.0, ncu 2025.3.1). **Base**: `cc4f26e` (+ this record).

## 1. Where things stood

[#202](https://github.com/yusiwen/minfer/issues/202) (record [109](./109-c4-packed-q8-kv-l1-request.md))
cut the packed arm's L1 load sectors 1.71x -> 0.98x of f16's and bought
**+0.27%** on `tg128`: a counter moved the whole way, the step did not. That left
the residual — 1.23x packed/f16 on the whole decode step at hd 64 — attributed to
nothing, with one hypothesis on record: the per-lane critical path (a separate
per-block scale load, a `cvt`, and on V four scale multiplies) inside a single-warp
block that stages four rows, against f16's two `__half2` loads and
`__half22float2`. This record measures that hypothesis at the stall level, then
tests the two layout-preserving levers it implies.

The step ratio is pinned by the kernel: the split kernel is ~0.96 ms of the packed
step's ~5.18 ms (19%), so a kernel-level 2.13x explains essentially the whole
1.23x step ratio. Everything below is therefore about the kernel.

**Same binary, same shape, both arms.** `q8_0` runs
`gqa_attn_split_partial<(int)2,(bool)1,(bool)0,(bool)1,(bool)1>` (Q8_0 / causal /
non-map / dp4a / wide) and `f16` runs `<1,1,0,0,0>` — the *same*
`attn_split_1w_body`, one warp per block, four rows staged per iteration, 16 of 32
lanes live at hd 64. The 4-warp hybrid (D3-4 L1) is gated `hd == 128` **and** f16
KV, so it is not in this comparison; a re-run at hd 128 for contrast would compare
the packed 1-warp kernel against a 4-warp f16 one and measure geometry, not the
chain.

## 2. The attribution

Provenance for every number: **GB10 sm_121, CUDA 13.0, driver 580.178.04,
2026-10-20**, `cargo build --release --features cuda` at `cc4f26e`, model
`~/.cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf`,
`bench -p 2048 -n 128 --n-ctx 4096`.

**nsys, the real decode path** (`nsys profile --trace=cuda --cuda-graph-trace=node
--stats=true`, 9 216 instances = 24 layers x 128 tokens x 3 passes; medians of the
per-launch distribution, which is what the step actually pays):

| arm | kernel | avg ns | **median ns** |
|---|---|---|---|
| `q8_0` wide + dp4a | `<2,1,0,1,1>` | 39 789 | **39 232** |
| `q8_0` byte + dp4a (`MINFER_NO_Q8_KV_WIDE=1`) | `<2,1,0,1,0>` | 40 876 | 40 192 |
| `f16` | `<1,1,0,0,0>` | 18 861 | **18 400** |

**ncu, one kernel instance per arm** (`sudo ncu --kernel-name
'regex:gqa_attn_split_partial$' --launch-count 1 --metrics <list> --csv`; counters
are deterministic across runs, stall ratios are medians of 3 and vary < 10%
between runs, and **durations are replay-relative only** — the same arm reads
75-93 us across runs, so no absolute duration is quoted from ncu):

| metric | `q8_0` wide | `q8_0` byte | `f16` |
|---|---|---|---|
| instructions | 2 687 888 | 2 747 486 | 2 436 294 |
| global load instr | 173 012 | 287 756 | 120 960 |
| L1 load sectors | 455 840 | 792 904 | 463 008 |
| sectors / request | 2.63 | 2.76 | 4.00 |
| L1 hit rate | 86.5% | 86.5% | 76.7% |
| L2 read sectors | 61 110 | — | 107 443 |
| **long-scoreboard** (cyc/issued inst) | **10.09** | **10.01** | **3.03** |
| short-scoreboard | 2.09 | 2.01 | 2.22 |
| wait | 1.63 | 1.65 | 1.26 |
| not-selected | 0.19 | 0.23 | 0.27 |
| warps active | 18.8% | — | 19.3% |
| issue active | 15.0% | — | 29.7% |
| warp cycles per issued inst | 15.60 | — | 7.76 |

Read the two tables together:

- **The stall is long-scoreboard, and it is the whole gap.** The packed arm burns
  15.60 warp cycles per issued instruction to f16's 7.76 (2.01x) on 1.10x the
  instructions — the duration ratio is latency, not work. Long-scoreboard alone
  accounts for 10.09 of the 15.60; every other named reason is at or below f16's
  (short-scoreboard 2.09 vs 2.22, not-selected 0.19 vs 0.27). Both arms sit at the
  same occupancy (~19% of 48 warp slots: 448 blocks of one warp over 48 SMs =
  9.3 warps/SM), so occupancy is not the discriminator either.
- **The load shape is not the binder, again and harder.** The byte control issues
  **66% more** load instructions and **74% more** L1 sectors for the same bytes —
  and its long-scoreboard is **10.01**, i.e. identical, and it is 2.4% *slower* on
  the real path today (record 109 measured 1.4-1.7% the same way; both are parity
  within this box's spread). A stall that does not move when the request count
  moves 74% is not a request-count stall.
- **The chain is.** Per row the packed lane does: scale load -> `cvt`; two `u16`
  loads -> shift/or assembly; `dp4a`; one scale multiply; five `shfl_xor`; then V's
  four `(int8 -> float) * scale` before its `fma`. f16 does two `__half2` loads ->
  `cvt` -> `fma`, and shares the same five shuffles. The arithmetic control points
  the same way: dropping `dp4a` (the convert arm, `MINFER_NO_DP4A_Q8_KV=1`) keeps
  the load shape and *adds* conversions — 3 242 470 instructions, long-scoreboard
  **21.37**, duration +55% over the wide arm in the same ncu run set. Longer chain,
  longer kernel; different load shape, same kernel.

**Step level, same binary, 3 interleaved rounds, medians**
(`minfer bench -p 2048 -n 128 -r 3 --n-ctx 4096 -o json`):

| test | f16 | q8_0 | ratio |
|---|---|---|---|
| `tg128` | 237.18 | **192.76** | **1.230x** |
| `pp2048` | 2 671.95 | 2 148.68 | 1.244x |

## 3. The lever that was tried, and why it was reverted

**Hypothesis.** The body stages four rows before the first softmax step (D2) so
that all of a batch's loads overlap. But `kv4_q8_0_packed` *converts inside the
accessor*, and the SASS shows what that does: one row's ~6 `LDG.E.U16`, then
`I2F.S8`/`I2F.S16` on those results, then the next row's loads — the load phase
ends after every row, so the batch never overlaps. Keeping the staging loop
load-only (quants as an `int`, block scale as its raw 16 bits) and converting in
the consumer should expose one latency per four rows instead of four.

**Bar, named before measuring** (`/tmp/212/bar.md`, written before the A/B): (a)
the kernel's median nsys duration must fall **>= 10%**; (b) `tg128` must beat the
same-binary control by **>= +2%**, `pp2048` inside 2%. (The control's env lever,
`MINFER_NO_Q8_KV_STAGE=1`, existed only for as long as the arm did and was removed
with it — it is not a shipping knob, and `check_env_vars` has no entry for it.)
Justification: 19% kernel share x 10% = ~+2% step; less is inside this box's
round-to-round spread (~1%).

**Implementation.** `q8_0_load4_bits` / `kv4_q8_0_raw` return the raw words;
`attn_split_1w_body` gained a runtime `q8stage` that keeps K and V staged raw and
defers `__half2float` plus the V dequant to the consumer. A runtime switch, not a
template one, so the audited `<<<` site count is unchanged (135) and the arm stays
a same-binary A/B. The conversion is the same `__half2float` of the same bits, so
the arithmetic is bit-identical by construction.

**Bitwise gate: passed.** `MINFER_GRAPH_DUMP` over `-n 8 --greedy --no-template`
with the staged arm and with the control: **123 / 123 dump files identical**
(`diff -rq` over `logits_*`, `node*`, `kv*layer{k,v}`), same generated tokens.

**Result in the gated form: null.**

| metric | control (off) | staged (on) |
|---|---|---|
| `tg128` median, 5 interleaved rounds | 191.03 | **191.21** (**+0.09%**) |
| `pp2048` median | 2 157.79 | 2 148.91 |
| instructions executed | 3 037 902 | 2 951 396 |
| long-scoreboard | 10.06 | 9.60 |
| ncu duration (same run set) | 93.2 us | 92.4 us |

Bar (a) is not assessable from this pair — the runtime branch taxes **both** arms
(the kernel grows to 1 032 instructions, the control path alone executes 3 037 902
against the pre-change 2 687 888), so the two durations sit under the same overhead
— and bar (b) missed at +0.09% against +2%.

**The clean rebuild — the number the gated arm could not give.** After the revert,
the same staging was rebuilt **without** the runtime switch (staged form
unconditional, kernel **776** instructions) and measured against the master binary
in one session, interleaved:

| metric | master (728 instr) | staged, no branch (776 instr) | delta |
|---|---|---|---|
| nsys kernel avg / median, 9 216 instances | 40 317 / **39 584** ns | 47 373 / **46 592** ns | **+17.5% / +17.7%** |
| `tg128` median, 3 interleaved rounds | **193.83** | 188.06 | **−2.98%** |
| `pp2048` median | 2 156.96 | 2 173.22 | +0.8% |
| bitwise dumps (both binaries) | — | 123 / 123 identical | — |

So the deferral is not neutral: it makes the kernel **~18% slower**, and the
step-level cost (~3%) is exactly the 19% kernel share times 18%. The earlier +0.09%
could not show this because the branch overhead sat on both arms.

**Why it is a regression, not a shorter chain.** The staging keeps four rows' raw
words live (four `int` + four `u16` for K, the same again for V, on top of the
existing `float4` staging) and moves every conversion *after* the whole load phase,
so the `cvt`/int8 work no longer overlaps the loads that produce it — the dependent
chain gets longer, and the register pressure rises. The incumbent's "convert right
after each load" is what lets the next row's loads issue while the previous row's
values are being built. That is the opposite of what the hypothesis assumed, and it
is the third independent confirmation that this kernel's cost is the chain, not the
loads.

**Reverted.** Both forms are gone: `q8_0_load4_bits`, `kv4_q8_0_raw`, the `q8stage`
argument and `q8_kv_stage_enabled` were removed after the measurement; the tree
difference of this record is documentation only.

## 4. What is left, and why neither is a kernel tweak

1. **A block layout whose quants are 4-byte aligned.** A 36-byte block
   (`f16 d; pad; i8 qs[32]`) or the `p32` split plane lets one group's four quants
   be a single `LDG.32` and moves the scale off the quant line. This is #202's
   priced-and-deferred alternative, and this record does **not** overturn the
   deferral — it strengthens it: the byte/wide pair shows that a 74% sector cut and
   a 66% load-instruction cut together buy ~2%, so a 3-load -> 2-load change inside
   the same chain would not pay for the blast radius (CPU store/read, `copy_cells`
   stride, `map_q8_0_cells`, FA-prefill staging, the C5 session version +
   `FLAG_PACKED`, and Metal's own `q8_0` readers).
2. **Geometry.** 448 warps (32 splits x 14 heads) over 48 SMs, 16 of 32 lanes live
   at hd 64, one warp per block. More splits, or two rows per warp (16 lanes each),
   would raise memory-level parallelism — and both change the numerics: the split
   count is baked into the partial buffer and the combine order, and a 16-lane
   reduction changes the butterfly width. That is a tolerance-class decision with a
   greedy-identity battery, not a scheduling change.

Both options are handed off as [#633](https://github.com/yusiwen/minfer/issues/633), with the
prices above and the acceptance each would have to carry, so this ticket can close on the
attribution alone.

So the "packed cache is decode-useful" question closes with numbers: on CUDA the
packed cell is a **memory** win (3.76x smaller than f32, 1.88x smaller than f16)
that costs **1.23x decode / 1.24x prefill at hd 64** and **1.01x / 1.04x at
hd 128**; the residual is a latency chain that no load-shape change moves, and
closing it requires a layout or geometry reshape that changes the file contract or
the tolerance class.

## 5. Verification

- **Bitwise (the staging arm, before it was reverted)**: 123/123 dump files
  identical between the control and the staged arm, and again between the clean
  rebuild and the master binary; generated tokens identical; `diff -rq` clean.
- **The clean rebuild**: `cargo build --release --features cuda` twice (staging,
  then master after `git checkout --`), each binary copied out and measured in the
  same session — `nsys` twice per binary (`bench -p 2048 -n 128 -r 1`), 3
  interleaved `bench` rounds per binary.
- **ncu**: medians of 3 per arm, `--kernel-name 'regex:gqa_attn_split_partial$'`
  (anchored — the unanchored regex also matches `_bt`, `_hybrid` and `_combine`;
  record 109's pitfall), `sudo` because `RmProfilingAdminOnly = 1` on this box.
- **nsys**: 9 216 instances per arm, one run each, `--cuda-graph-trace=node`.
- **Step level**: 3 interleaved rounds per arm, medians; the two arms' own spreads
  are ~0.5% (f16 236.94-237.90, q8_0 192.67-192.84), so the 1.230x ratio is well
  outside them.
- **Suite**: this change adds no code and no test. The current recorded rows stand
  unchanged — CUDA unit **697 / 0 / 51** (`scripts/cuda_test.sh`, dgxspark,
  2026-10-20), 0.5B real-model set **51 / 0**, Qwen3-0.6B real-model set **51 / 0**
  (`docs/TEST-BASELINES.md`).

## 6. Lessons

1. **A stall that survives a 74% sector cut and a 66% load-instruction cut is not a
   load stall.** #202 refuted the sector count; this step refutes the request count
   and the scheduling together. The only counter that moved the duration in this
   family is *arithmetic* (the convert arm's +55%).
2. **"Stages four rows" is a comment, not a guarantee.** The accessor's inline
   conversion ended the load phase after every row; the SASS was the only place
   that was visible. Read the SASS before trusting a staging comment — and note
   that deferring the conversion is not a fix either: rebuilt clean it costs
   **+18% kernel**, because the conversion stops overlapping the next row's loads.
3. **Measure the arm you would ship.** The staged form was bit-identical and
   neutral, but the runtime branch made the *shipping* path 13% more instructions.
   A same-binary A/B measures the arms; the binary's own code growth is a second
   cost that only a revert makes visible (SASS 728 -> 1 032).
4. **A refutation is the deliverable when the two cheap levers are spent.** The
   remaining options change the file contract or the tolerance class; recording
   their price is what lets the next reader decide instead of re-deriving.

**Scratch** (raw JSON, ncu CSVs, nsys reports, the dump pairs, the bar file): `/tmp/212/`.

← [109 · #202: the packed Q8_0 KV cell's L1 request count](./109-c4-packed-q8-kv-l1-request.md) · [Index](./index.md)
