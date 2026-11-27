# 115 · #845: the hd-64 prefill takes the FA body — 4.32x, and the packed arm reads 1.010x (LANDED)

> **Result**: hd-64 `pp2048` f16 **2 668.4 → 11 534.3** tok/s (**4.323x**), packed q8_0
> **2 188.7 → 11 417.5** (**5.216x**), so the packed/f16 prefill factor goes **1.2191x → 1.0102x**.
> `pp8192`: f16 780.5 → **10 364.1** (13.28x), q8_0 548.8 → **9 985.1** (18.20x), factor
> **1.4223x → 1.0380x**. Decode is untouched (hd-64 `tg128` packed/f16 **0.9693x** — the packed
> arm is 3.1% *faster*, record 114's 0.9712x). The attention kernel drops from **72.7% / 78.5%**
> of the hd-64 prompt's GPU time to **2.0% / 2.5%**.
> **Bar**: [#845](https://github.com/yusiwen/minfer/issues/845)'s acceptance, filed 2026-11-18 —
> `pp2048` at hd 64 **<=1.05x** f16 and no decode regression. Pre-registered prediction and
> decision rule: the 2026-11-27 comment
> ([#845 comment](https://github.com/yusiwen/minfer/issues/845#issuecomment-7211665245)) —
> 1.00-1.03x and ~4x. **MET** on both cells (1.0102x / 1.0380x, 4.32x / 5.22x).
> **Commit**: `<this PR>`. **Date**: 2026-11-27. **Box**: `dgxspark` (aarch64, GB10 sm_121,
> driver 580.178.04, CUDA 13.0, nsight-systems 2025.3.2).

## 1. Background — where things stood

[#845](https://github.com/yusiwen/minfer/issues/845) was the last performance cell of the packed
Q8_0 cache's "smaller and no slower" story: at head dimension (hd) 64 — Qwen2.5-0.5B and the other
small classes — a Q8_0 key-value (KV) cache answered a 2 048-token prompt at **1.224x** the f16
arm's time and at 8k at **1.46x**.

Record [114](./114-c4-packed-q8-kv-hd128-column.md) then showed the two prefill penalties are
*different defects*, not one. At hd 128 **both** arms take the flash-attention (FA) prefill and the
packed one pays **+44.8%** per launch for staging its blocks into the f16 tile — that is
[#858](https://github.com/yusiwen/minfer/issues/858). At hd 64 **neither** arm took it: the dispatch
gate was `hd == 128`, so both ran the legacy kernel, one block per (token, head), which re-reads
every K row once per query token (record
[03](./03-fa-tiled-prefill-attention-8n.md): 176 ms/layer, 76% of a 2K prefill). The packed arm paid
one extra per-element dequant on top.

A kernel trace on the tree this step starts from (`33ea56c`, Qwen2.5-0.5B q4_0, `bench -p 2048
-n 32 -r 1 --n-ctx 4096`) priced it:

| KV | kernel | share of GPU time | median/launch | x 24 layers |
|---|---|---|---|---|
| f16 | `gqa_attn_f32_f16kv<(bool)1,(bool)0>` | **72.7%** | 24.98 ms | 600 ms |
| q8_0 | `gqa_attn_f32<(int)2,(bool)1,(bool)0>` | **78.1%** | 32.34 ms | 776 ms |

`pp2048` whole prompt: 2 636 / 2 146 tok/s under the profiler = 777 / 953 ms. The two attention
kernels' 176 ms difference is the whole prefill delta; the matmul side is format-blind. The missing
piece was not a kernel — it was a gate and one template parameter.

## 2. Principle — one head dimension, two shapes

**Why the body was hd-128-only.** `fa_prefill_kv` stages K/V tiles into `__half` shared memory and
runs QK^T and P·V on `wmma` (Warp Matrix Multiply-Accumulate) tensor cores. Everything in it is
already general over the head dimension — the smem row stride (`hd + 8` halves), the per-token
strides (`nh * hd`, `nk * hd`), the QK^T loop (`for d = 0; d < hd; d += 16`) — except the **output
fragment count**: the O accumulator is one 16x16x16 f32 fragment per 16 output columns, so `acc[8]`
and five `ob < 8` loops spelled `hd = 128` into the body. That is a template parameter, not a second
kernel.

**Why lifting it closes the ratio, and not just the absolute number.** Both arms were on the legacy
body at hd 64, so both paid the per-token re-read; the packed one additionally paid a per-element
dequant, which is why its launch was **1.295x** the f16 one (32.34 / 24.98 ms). The FA body removes
the re-read for both, and it moves the packed arm's dequant *into the tile staging* — once per
staged element rather than once per query-key pair. The packed launch is still **1.289x** the f16
one (227.9 / 176.8 µs), i.e. the same relative tax, but it now applies to **2.0%** of the prompt
instead of 78%: 24 layers x 51.2 µs = **1.23 ms of a 179 ms prompt**, 0.7%. That is why the packed
factor falls to 1.01-1.04x while neither arm loses anything — the absolute win is the ticket's
real prize.

**Why the shared-memory opt-in disappeared.** The pre-#845 launcher issued a
`cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem)` and returned `-1`
(one printed warning, legacy fallback) when it failed. `smem` is
`(FA_TQ + 2*FA_TKV) * (hd + 8) * 2` — **18 432 B at hd 64, 34 816 B at hd 128**, both below the
48 KiB *default* dynamic-smem cap, so the call never did anything for either shape and the failure
arm could not trigger. It is gone; an HD above 128 (unreachable: the backend's guard caps `hd` at
128) would have to go back through `minfer_smem_optin` (ADR-0034).

## 3. Implementation

- **`fa_prefill_kv` gains `int HD`** as a template parameter (`<CAUSAL, MAP, LAYOUT, HD>`), and the
  runtime `hd` argument is gone — the body binds `constexpr int hd = HD` and
  `constexpr int NOB = HD/16`, so the two cannot disagree. `acc[8]` becomes `acc[NOB]` and the five
  `ob < 8` loops become `ob < NOB`.
- **Twelve instantiations, twelve launch sites**: (hd 64, hd 128) x (f16, packed q8_0) x (causal,
  span, map). The hd-128 sites keep their pre-#845 `launch:` names; the new ones are `*_hd64`. The
  #162 audit fixture is regenerated (136 -> 142 sites) and the device driver
  (`src/cuda/issue162_tests/sites.rs`) drives all twelve, each into a real
  `cudaErrorInvalidValue`.
- **The gate** (`src/cuda/methods/attention.rs`) becomes
  `nt >= 2 && (hd == 64 || hd == 128) && !MINFER_NO_FA_PREFILL && layout != KV_LAYOUT_F32` — the
  same pair Metal's `prefill_flash_enabled` accepts. `MINFER_NO_FA_PREFILL=1` is the same-binary
  "before" arm, which is what the tables below measure.
- **Pre-warm** covers all twelve (load-time, #223).
- **Gates**: `cuda_fa_prefill_attention_parity` and `cuda_q8_0_fa_prefill_attention_parity` now loop
  over both head dims, and each asserts the *route* through a chokepoint counter
  (`cuda_fa_prefill_f16` is new; `cuda_fa_prefill_q8_0` existed since #144) — the CPU-reference
  comparison alone cannot pin the body, see §4.3.
- **`cuda_kv_q8_0_roundtrip_attn`** (the packed store -> attention round trip, `nt = 3`, hd 64)
  changed shape: its graph arm now takes the FA staging, so it compares at the f16-tile class, and
  its **exact** claim (`kv4<Q8_0>` reconstructs exactly `d * q[i]`) moved to a second arm that
  drives the general layout-tagged kernel *directly* on the same bytes — that kernel is still the
  documented fallback (`MINFER_NO_FA_PREFILL=1`, or an f32 cache) and it measured **0** difference
  against the f32 regions.
- **The verify band changes body for a packed cache at hd 64** (`1 < nt <= 16`): from
  `gqa_attn_f32<(int)2,...>` to FA. That is the behaviour hd 128 has had since #144, and no
  greedy-identity contract rests on it — a packed cache refuses speculative decoding outright
  (`spec::SpecEngine::new`).
- **E1b's tile precondition was checked, not assumed.** `fa_prefill_kv` tiles 64 query rows against
  one window, so a tile must not span two sequences. At hd 64 the legacy kernel was per token and
  did not care; the constraint is already enforced on this path by `Batch::check` (contiguity) and,
  for the server, by `prefill_batch_ok()` returning `false` whenever a CUDA device exists
  (`crates/minfer-cli/src/server/batch.rs`) — so a CUDA prefill forward is always one sequence.
- **`#860` is fixed in passing**: the kernel's header comment claimed the staging was double
  buffered; it is synchronous (one Ks/Vs pair, `cp.async.wait_group 0` before the tile is read) and
  the packed arm does not `cp.async` at all. The comment now says so.

## 4. Verification

### 4.1 The suites

- `scripts/cuda_test.sh` (CUDA build, `dgxspark`, serial): **867 passed / 0 failed / 62 ignored**
  across the workspace targets (#858 quoted 866/0/62 for the same wrapper; the +1 is a test landed
  since). The two parity tests and the round-trip gate are members of that count — no test was
  added, both parity gates were parameterised instead.
- `FEATURES=cuda scripts/real_model_gates.sh` (serial, 0.5B configuration plus
  `MINFER_BATCH_TEST_MODEL=Qwen3-0.6B-Q8_0`): **35 passed / 0 failed / 0 ignored**. It includes the
  chunked-vs-unchunked prefill equality and the server's batch-equals-serial gates, both of which
  now run a hd-64 FA prefill whenever a CUDA device is present.
- `python3 scripts/check_cuda_launch_returns.py`: every one of the **142** `<<<` sites reads its own
  error; `--check-fixture` green against the regenerated `tests/fixtures/cuda_launch_sites.tsv`.

### 4.2 Numerics — the class the bar asked for

The staging rounds K/V through `__half`, so the claim is a tolerance class, exactly as the ticket
anticipated. Kernel-level, against the CPU reference over the same KV bytes (5e-3 bound):

| parity gate | hd 64 | hd 128 |
|---|---|---|
| `cuda_fa_prefill_attention_parity` (f16) | max abs delta **3.44e-4** | 2.78e-4 |
| `cuda_q8_0_fa_prefill_attention_parity` (packed) | max abs delta **6.72e-4** | 4.80e-4 |

Model level, `a_packed_kv_cache_answers_like_the_f32_one` on the CUDA arm (the packed engine's
prefill now takes FA at hd 64): **greedy 9/9** steps agree with the f32 engine, max |Δlogit|
**2.366** of a 37.821 spread, max |Δ| at the reference argmax 0.559; the packed-row-shift twin reads
**8/8** greedy. These are the numbers that test reported before this change as well — the FA body
does not move the format-level class.

### 4.3 The gates can fail (mutation evidence)

Reverting the dispatch condition to `hd == 128` and re-running the two parity gates turns both red
on their **route** assertion (`the f16 prefill at hd 64 did not reach the FA launch` /
`the packed prefill did not reach the FA launch`). The parity arms themselves *pass* in that mutant —
at a **10x tighter** max |Δ| (3.5e-5 f16, 3.0e-5 packed), because the legacy body never rounds the
cache to f16. That is the reason both gates now carry the counter: a 5e-3 CPU comparison cannot tell
the two bodies apart, so the observation half is what makes them route gates (gate contract rule 3).

## 5. Results

### 5.1 The bench matrix

`minfer bench -p 2048 -n 128 -r 3 --n-ctx 4096`, three **alternating** rounds (the four
(kv, body) combinations in a different order each round), medians of the per-round `ts_mean`; the
two models, one command, one tree:

| cell | f16 | q8_0 | packed/f16 |
|---|---|---|---|
| **hd 64 `pp2048`**, FA (default) | **11 534.3** [11 482.5, 11 534.3, 11 831.9] | **11 417.5** [11 238.8, 11 417.5, 11 460.7] | **1.0102x** |
| **hd 64 `pp2048`**, legacy body | 2 668.4 [2 460.7, 2 668.4, 2 671.7] | 2 188.7 [2 176.6, 2 188.7, 2 197.3] | 1.2191x |
| **speedup** | **4.323x** | **5.216x** | |
| **hd 64 `pp8192`**, FA (default) | **10 364.1** [10 189.5, 10 364.1, 10 406.7] | **9 985.1** [9 914.1, 9 985.1, 10 118.0] | **1.0380x** |
| **hd 64 `pp8192`**, legacy body | 780.5 [777.6, 780.5, 794.0] | 548.8 [547.8, 548.8, 553.5] | 1.4223x |
| **speedup** | **13.28x** | **18.20x** | |
| **hd 128 `pp2048`** (control, route unchanged) | 8 724.1 [8 553.4, 8 724.1, 8 814.9] | 8 286.8 [8 274.4, 8 286.8, 8 289.5] | 1.0528x |
| hd 128 `pp2048`, legacy body | 597.5 [594.3, 597.5, 600.1] | 565.6 [565.4, 565.6, 568.9] | 1.0564x |

Decode, same rounds: hd 64 `tg128` f16 **233.83** / q8_0 **241.27** (packed 3.1% faster,
f16/packed **0.9693x** — record 114 read 0.9712x); hd 128 `tg128` f16 153.33 / q8_0 152.69
(**1.0042x**). No arm of this change touches `nt == 1`, and the numbers say so.

Two caveats belong beside the table:

- **The 8k row is one rep per round**, like record 114's; its legacy arm is quadratic (a 2 048-token
  prompt at 780 tok/s against 8 192 tokens at the same 780 tok/s is not a linear reading — the
  legacy body re-reads K per query token, so its cost grows with the prompt and the measured
  speedup grows with it).
- **The hd-128 cell is a control, not this ticket's measurement.** It is the unchanged route (both
  arms already took FA), and it read 1.0528x here against record 114's 1.0796x and 1.0269x in an
  earlier concurrent run of the same binary — the packed/f16 *prefill ratio* moves by +-2.5 pp
  between sessions while the routing does not, because the cross-arm weight-matmul noise is the same
  size as the effect (record 114's §3.3). The stable quantity is the per-launch kernel delta
  (§5.2), not the ratio; #858 should quote the trace, and a bar on this cell needs a same-session
  control.

### 5.2 Where the time went

nsys medians per launch, hd 64, same 2 048-token prompt, both bodies of each layout:

| | legacy body | FA body |
|---|---|---|
| f16 attention | `gqa_attn_f32_f16kv<(bool)1,(bool)0>` **24.98 ms**, 72.7% of GPU time | `fa_prefill_kv<1,0,1,64>` **176.75 µs**, **2.0%** |
| packed attention | `gqa_attn_f32<(int)2,(bool)1,(bool)0>` **32.34 ms**, 78.1% | `fa_prefill_kv<1,0,2,64>` **227.92 µs**, **2.5%** |
| packed/f16 per launch | 1.295x | **1.289x** |
| `mmq_nt_kernel` share | 17.6% / 14.7% | **65.7% / 65.8%** |

The prompt is now weight-matmul bound. Attention costs 24 x 176.75 µs = **4.24 ms** of the 178.7 ms
prompt (2.4%), and the packed arm's staging adds 24 x 51.2 µs = **1.23 ms** (0.7%) — against the
1.0% the bench measures, the residual being the cross-arm matmul noise the shares already show.
For comparison, the same body at hd 128 costs 460.56 us/launch for f16 (record 114), so hd 64 lands
at 38% of the hd-128 launch — the tile has half the output columns and two KV heads instead of
eight, and the prediction in the pre-registration (~200 µs) was within 12%.

## 6. Lessons

1. **A gate that names a shape is worth measuring before designing a kernel.** The hd-128-only gate
   made a 4-18x prefill win look like an unsolvable 1.22x ratio; the fix was one template parameter
   and one condition. When a ticket says "no FA prefill at X", check whether the body is
   shape-specific or merely *gated* — the difference is a day of work.
2. **The ratio and the absolute number can disagree about how big a change is.** The bar (<=1.05x)
   and the prize (4.3x) lived in the same patch; a review that only asked "did the ratio move"
   would have rejected it at 1.0102x as "barely better than 1.2191x". State both, and name which
   one the ticket's acceptance actually tests.
3. **A tolerance gate cannot pin a route.** The CPU-reference parity passes on the legacy body at
   3.5e-5 — 10x *tighter* than the FA tile's 3.4e-4. Only the chokepoint counter distinguishes the
   two, and the mutation run is what demonstrates it (#144's rule, applied to f16).
4. **Check the precondition a new shape inherits, not the one it had.** E1b's "a query tile must not
   span two sequences" was already enforced for hd 128 by composition (`Batch::check`,
   `prefill_batch_ok()`); the hd-64 lift inherits the guarantee, and the record should name where it
   comes from rather than assuming it.
5. **A stale comment is a design input.** #860's "double buffered" claim would have told #858 that
   the staging is already overlapped; it is not, and the packed arm cannot `cp.async` at all. Read
   the loop, not the header.

<- [114 · #848: the hd-128 column measured — and the four ratios #842 quotes](./114-c4-packed-q8-kv-hd128-column.md) · [Index](./index.md)
