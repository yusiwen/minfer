# 66 · D2: explicit K+V register staging (LANDED, +2.0% @1641) and the cp.async negative result

> **Result**: 7B decode @1641 KV **47.2 → 48.2 tok/s (+2.0%)**,
> `gqa_attn_split_partial` kernel 34.1 → **19.4 µs/launch (−43%)**; probe
> (cold-DRAM) −42% / nsys −43% / wall clock +2.0% three-way consistent;
> **bitwise-identical** (greedy-32/256 byte-identical streams, suite 169/0/3).
> tg128 flat (vs llama: 0.956× → **0.975×**, gap −4.5% → −2.4%).
> Attached: the 5 variants measured to death in the D1/D2 windows (NR=8
> register window, cp.async smem pipelines ×3, pair lookahead) — all
> bitwise-safe, all slower than the landed form; veto mechanism inside.
> **Commit**: `0730c15`. **Date**: 2026-09-07.

## 1. Background — where things stood

D1 (doc 65) had attributed 100% of 7B decode @1641's 0.91 ms/step wall-clock
gap to `gqa_attn_split_partial`, with the diagnosis: 34.1 µs/launch ≈ 12 µs
bytes + ~22 µs of **exposed memory latency** — each V row's load issues
inside the online-softmax dependency chain, and the 4-row batch window
provides only 4 rows of load-level parallelism. D1 had also calibrated the
freedoms: **staging-depth-class changes (same split ranges, same row order,
same per-row ops; only load scheduling moves) are probe-verified
bit-identical**; split-count / block-structure classes necessarily reorder
float summation and need tolerance gates.

D2's task was thus defined very narrowly: **without touching any arithmetic,
move the K and V row loads out of the dependency chain**. This was the only
unused item on D1's "free knob" list. D1's probe had given a conservative
signal (−11% for a staging variant in hot-L2 mode); re-measured under
cold-DRAM it was −42% outright — decode's KV reads are cold-DRAM-shaped at
real scale, so this lever is 4× the hot-L2 signal.

One obstacle had to be cleared first, and it was dramatic in its own right:
an old comment lying in the kernel claimed "V's addresses are known, the
compiler will hoist the inline V loads above the softmax chain". **That
comment is wrong** — it is both the entire source of this optimization's
gain and a good lesson: trusting a comment is no substitute for dumping the
SASS once.

## 2. Principle — the GPU mechanism

### 2.1 The old form's latency chain

The pre-D2 execution structure for each 4-row batch:

```
batch start:  K rows ×4 loads (staged, outside the chain)
loop:         for each row j:
                dot (4 mults) → 5-step shfl butterfly → 2 × expf  ← serial dependency chain
                → V row load   ← inline, issue point = consume point, queued behind the chain
                → oc accumulate update ← consumes V
```

The key is that the `shfl`/`expf` chain is a **loop-carried dependency**
(mx/S updated row by row); the compiler's load reordering does not dare (or
did not) cross it to hoist the V load: the V load's issue is deferred to
near its consume point, and the instruction before the consume point is
`expf`. So every row pays one load latency (hundreds of ns on GB10's
cold-DRAM path), 52 rows × one each ≈ 20+ µs of pure waiting — exactly
matching the ~22 µs exposed latency D1 decomposed.

"Why doesn't the compiler hoist it itself" is worth recording: the inline
load sits inside an `if (live)` branch, and with a large loop body and a
tight register budget, ptxas's scheduling window is insufficient to lift it
to batch start; this is not a compiler bug but an **ownership-of-gain
problem** — only the code's author knows "this batch's V can all issue
before the chain", and that knowledge must be written explicitly into the
source.

### 2.2 The new form: loads back-to-back

D2 stages **all of each batch's K and V** into registers before the first
softmax step:

```
batch start:  K rows ×4 loads + V rows ×4 loads  ← 16 LDGs back-to-back, all outside the chain
loop:         for each row j: dot → shfl → expf → oc update (consuming the in-register v4[j])
```

The batch-start back-to-back loads went from 4 (K only) to 8 (K+V) — the V
loads no longer hang row by row on their consume points but issue early
outside the chain alongside K, and the load latency overlaps the serial
chain — overlapping not just this batch but the previous batch's residual
chain. The SASS is the most direct evidence:

> 16 `LDG.E.CONSTANT` clustered at 0x7a0–0x9f0; the chain's first
> `SHFL.BFLY`/`MUFU.EX2` at 0xde0.

### 2.3 The arithmetic: why the wall clock gains only +2.0%

The kernel −43% but the wall clock only +2.0% — that ratio is dictated by
the decode step's anatomy:

```
kernel savings: 28 layers × (34.1 − 19.4) µs ≈ 0.41 ms/step
step time @1641: ≈21 ms (47.2 tok/s)
0.41 / 21 ≈ +2.0%  ✓
```

attention @1641 is ~4.5% of 7B's decode step (28 × 34.1 µs ≈ 0.95 ms / step
~21 ms; the same anatomy measured on 14B in D3-1 shows an even lower
short-KV share: 0.75%) — D2 cut 43% of it, landing on the wall as **+2.0%**
(0.41 ms / 21 ms). This is also the conversion rate that recurs throughout
the later decode campaign: kernel-level big wins ⇒ small wall-clock wins;
only moving multiple kernels together (the D3-5/D3-7/D3-8/D4-4 bundles)
stacks up to the +5% class.

Register cost (ptxas `-Xptxas -v`): 40 → 52–58 regs (f16) / 72 (f32),
STACK/LOCAL **0**, occupancy unchanged (capped at 24 blocks/SM) — 52–58
regs did not push anything out of the SM; the gain is net.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Registers, not shared memory**: each K/V row is only 16 B per lane per
  row (f16 KV 8 B/lane/matrix); smem staging would add an LDS round-trip and
  a barrier; registers are the zero-cost home for a load. The
  negative-results table (§5) confirms the smem routes are all slower.
- **Window = the existing 4-row batch (NR=4)**: the batch size is already a
  warp-uniform control-flow boundary; widening to 8 rows (NR=8) measured
  slower — in-flight load count hits a wall (§5).
- **Scheduling only, no arithmetic**: same batch rows, same row order, same
  per-row op sequence — float summation order bit-for-bit unchanged; that is
  the entire content of bitwise-identical by construction, and the boundary
  D1's taxonomy authorized.

### 3.2 Key code

before (the diff of commit `0730c15`, the "-" side, i.e. the D1-era shape):

```cuda
for (int base = lo; base < hi; base += 4) {
    int nr = min(4, hi - base); // warp-uniform
    // Stage K for the whole batch first; the V addresses are already
    // known, so the compiler hoists those loads above the softmax chain.
    float4 k4[4];                                   // ← the old comment's claim, which is wrong
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        k4[j] = (live && j < nr)
            ? kv_ld4<KV>(k + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        ...
        S = S * corr + e;
        mx = nmx;
        if (live) {
            // ← inline V load: issue point = consume point, queued behind the shfl/expf chain
            float4 v4 = kv_ld4<KV>(v + (size_t)(base + j) * stride_kv + hk * hd + d0);
            oc.x = oc.x * corr + e * v4.x;
            ...
        }
    }
}
```

after (the diff "+" side; the current tree carries it in `attn_split_1w_body`,
`src/cuda_kernels.cu:2830`):

```cuda
for (int base = lo; base < hi; base += 4) {
    int nr = min(4, hi - base); // warp-uniform
    // D2: stage BOTH K and V for the whole 4-row window before the first
    // softmax step. All 8 row loads then issue back-to-back and their
    // latency overlaps the serial chain; the old form relied on the
    // compiler hoisting the inline V loads, which it does not do across
    // the shfl/softmax dependency chain (D1 probe: −11% hot-L2, D2 probe:
    // −42% cold-DRAM vs inline V; bitwise-identical — same rows, same
    // order, same per-row ops, only the load scheduling changes).
    float4 k4[4], v4[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {           // before the chain: 8 row loads back-to-back
        k4[j] = (live && j < nr)
            ? kv_ld4<KV>(k + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        v4[j] = (live && j < nr)
            ? kv_ld4<KV>(v + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j >= nr) break; // warp-uniform: all lanes exit together
        float d = q4.x * k4[j].x + q4.y * k4[j].y
                + q4.z * k4[j].z + q4.w * k4[j].w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFF, d, off);   // the serial chain, unchanged
        float s = d * scale;
        float nmx = fmaxf(mx, s);
        float corr = expf(mx - nmx);
        float e = expf(s - nmx);
        S = S * corr + e;
        mx = nmx;
        if (live) {
            float4 vv = v4[j];                  // ← consume the register; no load issued
            oc.x = oc.x * corr + e * vv.x;
            oc.y = oc.y * corr + e * vv.y;
            oc.z = oc.z * corr + e * vv.z;
            oc.w = oc.w * corr + e * vv.w;
        }
    }
}
```

Side by side: the loop body (shfl/expf/oc update) is untouched character for
character; the only structural change is the `v4` array filled at batch
start and `v4[j]` replacing the inline load in the row loop. The
`kv_ld4<KV>` template (f32 reads float4 directly / f16 converts two half2
into a float4, `src/cuda_kernels.cu:2784`) is also untouched.

Archival note: this code was later lifted verbatim into the device function
`attn_split_1w_body` at D3-4 L1 (doc 69), sharing one source with the
4-warp hybrid kernel — the comment states "math is byte-identical to the
pre-refactor kernel". The semantics quoted in this doc are those of the
current tree's `attn_split_1w_body`.

### 3.3 Pitfalls

- **The comment lied**: the old comment asserted the compiler would hoist
  the inline V load. Measurement falsified it — in the SASS of the inline
  form the V loads crowd next to their consume points, not at batch start.
  Lesson: **before optimizing, `cuobjdump -sass`; do not trust comments**
  (doc no. 77's SASS-first rule).
- **NR=8's in-flight ceiling**: widening the window from 4 to 8 rows to hide
  one more latency layer measured 22.7 vs 19.9 µs — slower; per-lane
  in-flight load count (and register pressure) has a hard cap, and deepening
  the window is not free (§5).
- **The probe's wait-group bug (r59b-class)**: the first cp.async probe
  reported several pipeline variants DIVERGENT and the rest "coincidentally
  OK". The root cause was in the probe, not the concept:
  `cp.async.wait_group <STAGES-1>` only guarantees the oldest group's
  completion when **exactly STAGES groups are in flight**; when the loop
  tail has fewer than STAGES iterations it must be `wait_group 0`. Any
  cp.async ring over a runtime trip count needs this branchy wait. Unfixed,
  the negative-results table would have been archived under the wrong reason
  — "concept error" instead of "granularity error".

## 4. Verification

All gates green (what each defends):

- **probe memcmp gate**: the `gqa_attn_split_partial` partial buffers
  compared byte-for-byte, **ndiff=0, 45/45** @ nkv ∈ {1, 29, 52, 512, 1641},
  across all candidate variants — defends against a staging change quietly
  touching arithmetic;
- **greedy stream gate**: greedy-32 and greedy-256 token streams vs the
  pre-change binary **byte-identical** — defends against end-to-end
  cumulative drift and sampler interactions;
- **parity suite**: the trio (`cuda_prefill_mmq_parity`,
  `cuda_prefill_capture_bit_parity_pp16_pp300`,
  `cuda_fa_prefill_attention_parity`) + `cuda_attn_split_decode_parity` +
  q4k/q6k decode MMVQ parity — defends against collateral damage outside
  attention (prefill/capture/MMVQ);
- **suite**: **169/0/3** — defends against regressions elsewhere in the repo;
- **three-way consistency**: probe −42% / in-situ nsys −43% / wall clock
  +2.0% — the kernel-level numbers of two independent instruments agree, and
  the wall clock lands on the prediction via §2.2's anatomy — defends
  against pseudo-gains that "look good on only one instrument".

## 5. Results

| Evidence | baseline | D2 | Δ |
|---|---|---|---|
| probe, cold-DRAM 28-layer rotation, nkv=1641 | 34.6 µs | 19.9 µs | **−42%** |
| nsys in-situ per-launch µs @1641 | 34.1 | 19.4 | **−43%** |
| decode `-n 128` @1641 (3× interleaved median) | 47.2 tok/s | **48.2 tok/s** | **+2.0%** |
| decode tg128 (KV~1) | 49.4 | 49.4 | flat |
| combine kernel | 96.0 µs/step | 95.7 µs/step | flat |
| ptxas | 40 regs (f16) | 52–58 regs (f16), 72 (f32), STACK/LOCAL 0 | occupancy unchanged (24 blocks/SM cap-bound) |

vs llama.cpp: tg@1641 47.2/49.41 = 0.956× → **48.2/49.41 = 0.975×** (gap
−4.5% → −2.4%); tg128 stays at parity. 7B decode thus enters llama's ±2.5%
band, and later sessions (the D3 series) kept grinding from this base.

### 5.1 Negative results (do not retry blindly)

All bitwise-safe (after fixing §3.3's probe wait-group bug), all measured
**slower than the landed form's 19.9 µs** in the same cold-DRAM mode:

| Variant | cold-DRAM µs | vs 19.9 |
|---|---|---|
| register staging NR=8 (window doubled in depth) | 22.7 | slower |
| cp.async smem K-only pipeline NR=4 S=2 / S=3 | 23.4 / 22.4 | slower |
| cp.async smem K-only pipeline NR=8 S=2 | 30.9 | far slower |
| cp.async smem K+V pipeline NR=4 S=2 | 21.0 | slower |
| pair-unrolled register lookahead (95 regs) | 22.1 | slower |

**Veto mechanism** (under what conditions a retry is worthwhile):

- **LDGSTS granularity**: f16 KV is only 8 B per lane per row — `cp.async`'s
  `LDGSTS.E.BYPASS.128` vocabulary collapses at this granularity, and the
  smem route forces one LDS round-trip; the direct LDG-into-register route
  pays neither tax. **cp.async remains the right tool — but only where the
  staging tile is ≥16 B/lane** (the MMQ kernels; r45/r53/r56 all rely on
  it); decode attention is not in that class unless a KV-layout change (e.g.
  dpl, doc 76) widens the row granularity.
- **Window depth**: NR=8 hits the hard cap of in-flight loads/register
  budget, not "not enough hiding". Unless the SM microarchitecture
  generation changes (more in-flight loads per warp), this direction is not
  worth retrying.

### 5.2 Residuals and what came after

The landed kernel's latency decomposition: 19.4 µs ≈ 12 µs bytes + **~7 µs
for the serial chain itself** — the chain became the floor. Going further
requires a block/work-mapping change (a llama-vec-style multi-warp 256-row
cooperative rewrite), unreachable under byte-identity and requiring
tolerance gates (D3a did it and was reverted for rpw pathology, doc 68;
D3-4 ultimately recovered it at long KV via hybrid dispatch, doc 69). The
two micro-levers `f32_bits_to_i32` (~0.5%/step) and the combine's
empty-split reads (~90 µs/step) were left for later (D3-7 2c / D3b-2).

## 6. Lessons

1. **SASS first, comments second**: a "the compiler will do it for me"
   comment let the inline V load pay 22 µs/launch of latency for nothing;
   the first step of any load-scheduling optimization is always dumping the
   SASS to see the issue points.
2. **Calibrate the bitwise-free class with a probe first, then maximize
   it**: D1's classification (staging-depth free) + D2's execution (8 loads
   back-to-back) form the standard rhythm of "measurement authorizes →
   narrow change → three-way verified"; a −43% kernel required no parity
   price at all.
3. **Archive negative results with their mechanisms**: the 5 variants died
   of LDGSTS granularity (8 B/lane) and the in-flight hard cap, not
   "cp.async is bad" — write the mechanism down, and the next layout change
   reveals which premise moved and whether a retry pays.
4. **Instrument bugs forge concept errors**: one wait_group tail-semantics
   bug nearly wrote "cp.async doesn't work" into the archive; instruments
   must prove themselves first (r59b's baseline contamination and this doc's
   probe divergence are the same species).

---
← 65 · [Index](./README.md) · 67 →
