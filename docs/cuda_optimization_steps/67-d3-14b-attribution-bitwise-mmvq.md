# 67 · D3: 14B decode attribution (D3-1) + the D3b bitwise MMVQ triple (1b LANDED; 1a/1c REVERTED)

> **Result**: D3-1 measurement round — 14B decode's wall-clock gap (43.84 vs
> llama 41.14 ms/step) is **not attention** (at short KV attention is only
> 0.75% of the step): about half is three MMVQ stragglers (attn_v-q6K
> **134.8 GB/s** on padded-f32, ffn_down-q6K **198.9**, the output head
> **200.1**, vs the 220–225 GB/s class, totaling ≈ **+1.2 ms/step**), the
> other half is the elementwise/launch chain (97 rms + 265 quantize + ~700
> sub-2 µs launches); the whole step is **93.4% weight streaming**, launch
> gaps already better than llama. D3b (bar: ≥ +0.4% at the target site,
> bitwise-gated): **1b** `q6_k_q8_mmvq_v2_pf` pipelining LANDED (14B tg128
> +0.44%, @3254 +0.24%; 7B +3.0%/+2.7% — later voided by D4-2, see §5);
> **1a** both routes REVERTED (MMVQ routing is not bitwise; the NSG 2→1
> remap is bitwise-green but the kernel **+9.6%**); **1c** 160-thread blocks
> REVERTED (the gain mechanism does not exist on this SM).
> **Commit**: `f1825b5` (1b); 1a/1c were reverted after measurement, no repo
> trace (patches in `/tmp/d3/`, ephemeral). **Date**: 2026-09-07.

## 1. Background — where things stood

D2 (doc 66) ground 7B decode @1641 to 48.2 tok/s (0.975× vs llama) with
tg128 at parity — 7B was inside llama's ±2.5% circle. The campaign's next
small model was **Qwen2.5-14B q4_k_m**: 48 layers, hidden 5120, GQA 40:8,
decode tg128 ~22.8 tok/s vs llama 24.31 (0.94×), worse at long KV. 14B's
weight volume is ~2× 7B's, and decode streams the whole weight set every
step, so **per-kernel effective GB/s** becomes decode's first-class metric
for the first time — gaps that in the 7B era were explained by launch counts
and attention shape become, at 14B, "which matmuls fail to reach the stream
rate they should".

Meanwhile D3a (the 4-warp fattn-vec rewrite, doc 68) was opening another
front: structural attention rewrites must go through tolerance gates. D3's
session design therefore split into two tracks — D3-1 attribution + D3b
doing only bitwise levers (this doc), with attention structure left to
D3a/D3-4 (docs 68/69). Bar pre-registered: a lever survives only if measured
**≥ +0.4%** at its target site (the kernel/anchor it claims to save),
otherwise revert.

## 2. Principle — the GPU mechanism

### 2.1 The anatomy of the 14B decode step

The quantitative anatomy from D3-1's nsys census (method in §3.1):

- per step **43.84 ms** (llama.cpp 41.14, gap 2.70 ms / 6.6%);
- **93.4% of kernel time is weight streaming** (the MMVQ/MMQ family passing
  all weights through DRAM) — decode is essentially a "per-token pass over
  the full weight set"; the launch-gaps item is already better than llama
  (the engine's CUDA-graph replay wins on per-launch overhead — not a gap
  source);
- at short KV attention is only **0.75%** of the step — the wall-clock gap
  has nothing to do with it;
- the gap's composition: **~half = three MMVQ stragglers running at low
  stream rates** (§2.2), totaling ≈ +1.2 ms/step; **~the other half = the
  elementwise/launch chain** (97 rms + 265 quantize + ~700 sub-2 µs launches
  — this chain was later collected in three batches by D3-5/D3-7/D3-8, docs
  70/72/73).

An important contrast item: **attention is the only kernel in the 14B step
that grows with KV** (at nkv=3254, +3.33 ms/step, 67% of the DRAM byte
floor, long_scoreboard-bound) — it matters, but its lever lives at long KV
and needs tolerance gates, outside this doc's bitwise track.

### 2.2 The three stragglers' arithmetic

A decode matmul's effective stream rate = weight bytes / kernel time. The
14B census computed this table per kernel at the tg128 and @3254 anchors;
three fell below the "220–225 GB/s class":

| Kernel | Shape (od × id) | Dispatch path | GB/s | Shortfall |
|---|---|---|---|---|
| attn_v-q6K | 1024 × 5120 (5.24M elem, 11 layers) | **padded-f32** (the 24M gate excludes it) | **134.8** | −40% |
| ffn_down-q6K | 5120 × 13824 (id 13824 → npair 432) | MMVQ v2 | **198.9** | −10% |
| output head lm_head-q6K | 152064 × 5120 (npair 160) | MMVQ v2 | **200.1** | −10% |

attn_v's byte arithmetic: per-layer weights = 1024 rows × (5120/256 = 20
super-blocks) × 224 B (padded stride) = 4.59 MB; 134.8 GB/s ⇒ ~34 µs/launch,
matching nsys's 36.4 µs. The three together ≈ +1.2 ms/step — exactly half
the gap.

### 2.3 MMVQ vs padded-f32: why the 24M gate keeps attn_v out

The 8e era measured decode MMVQ's shape crossover on device (archived in the
`src/cuda.rs:2589` comment): below `od*id < ~24M` padded-f32 wins (od 512 →
MMVQ 4.5× slower, od 896 → 3.0×, 2048×4864 → 1.66×); at large shapes MMVQ
wins (7B ffn_down 1.5×, lm_head 1.4×). Mechanism:

- **small od ⇒ only 1–2 units per thread** (a unit = half of 32 elements,
  q6_K's dp4a dot unit): q5/q6's nibble byte reads are inherently scattered
  (unit granularity 16–64 B); with too few units those scattered loads'
  latency is fully exposed; the padded-f32 kernel is a block dequant + FMA
  loop with coalesced reads and is actually faster at small shapes;
- **large od ⇒ many units per thread**: dp4a's arithmetic-density advantage
  + uint4 widened reads (the R2 rework) outweigh the scattered-load
  disadvantage.

14B attn_v is od 1024 × id 5120 = **5.24M** < 24M ⇒ excluded from MMVQ by
the gate, landing on the padded-f32 kernel — yet it is exactly one of the
kernels that most deserves MMVQ at that shape (D3-7 2b later lowered the
gate to 4M and rescued it, doc 72; this doc's 1a handles "what could be done
in the code of that moment").

### 2.4 v2's second-serial-unit problem (pf's mechanism)

`q6_k_q8_mmvq_v2` is a u-loop form: each thread `for (u = tid; u < npair;
u += 256)` accumulates unit by unit. With npair ≤ 256 each thread has one
unit — no problem; at **npair > 256** (14B ffn_down: id 13824 → npair 432)
threads 0..175 get a second unit — and that second unit's **weight + q8
loads issue only after the first unit's accumulation**, serially hanging off
the critical path. The 198.9 vs 220-class gap is mostly this exposed load
latency.

The fix is of D2's lineage: **move only load scheduling, not arithmetic** — issue both units' loads before any accumulation (bitwise-free, staging-depth class).

## 3. Implementation

### 3.1 The D3-1 method: nsys census + per-kernel GB/s

The same rig as D1 (doc 65) with two upgrades: the anchors become 14B's
tg128 + @3254; the census table annotates every matmul kernel with
**effective GB/s** (weight bytes / nsys time), turning "who is below class
level" into a visible ranking rather than a feeling. The full report lives
in `/tmp/d3/D3_FINDINGS.md` (ephemeral); the key numbers are inlined in the
master table's D3b rows and this doc.

### 3.2 D3b-1b: q6_k_q8_mmvq_v2_pf, LANDED (`f1825b5`)

before — v2's u-loop (`src/cuda_kernels.cu:1536`, current tree, untouched by 1b):

```cuda
float acc = 0.0f;
for (int u = threadIdx.x; u < npair; u += 256) {   // ← when npair>256 the second
    const int kbx = u >> 3, pair = u & 7;          //    pass's loads issue only
    const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * blk_stride;
    ...                                             //    after the first pass accumulates
    acc += d8 * sc0 * d * (float)dot0 + d8 * sc1 * d * (float)dot1;
}
```

after — the pf form splits the unit into load/acc halves, both loads back-to-back:

```cuda
__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2_pf(
    const uint8_t* __restrict__ weights, const uint8_t* __restrict__ acts8,
    float* __restrict__ output, int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * blk_stride;
    const int npair = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;
    const uint8_t* wrow = weights + (size_t)row * row_stride;

    float acc = 0.0f;
    const int u0 = threadIdx.x;
    const int u1 = u0 + 256;                 // the same thread's second unit
    if (u0 < npair) {
        Q6kUnitRegs r0, r1;
        q6k_unit_load(u0, wrow, x8row, blk_stride, &r0);      // load #1
        const bool two = u1 < npair;       // npair > blockDim (guaranteed by the dispatch gate)
        if (two) q6k_unit_load(u1, wrow, x8row, blk_stride, &r1); // load #2 immediately after
        q6k_unit_acc(&r0, acc);                                // only then accumulate
        if (two) q6k_unit_acc(&r1, acc);
    }
    mmvq_block_reduce(acc, output, od, t);
}
```

`q6k_unit_load`/`q6k_unit_acc` (`src/cuda_kernels.cu:1590/1611`) mechanically
split the v2 loop body: the load half gathers uint4 ql/qh + scales + the q8
row pointers into the register group `Q6kUnitRegs`; the acc half is the
original dp4a tree. **The bitwise argument** (written twice, in the file
comment and the commit): same thread→unit mapping (u = tid, tid+256), same
per-unit dp4a tree, per-unit accumulation statements **character-identical**
(same FMA contraction shape:
`acc += d8 * sc0 * d * dot0 + d8 * sc1 * d * dot1`), ascending u order, same
block reduction — the only thing moved is load scheduling, i.e. D2's
staging-depth precedent reused on MMVQ.

Dispatch gate: id > 8192 (14B ffn_down's npair 432 was the only target shape
at the time) goes to pf, everything else to the v2 loop. **Note**: the gate
had no upper bound — the consequence of that omission is §5's D4-2
CORRECTION; the current tree (`src/cuda.rs:4293`) has
`id > 8192 && id <= 16384`.

### 3.3 D3b-1a: rescuing attn_v from padded-f32, both routes dead

Goal: rescue attn_v's 134.8 GB/s (1024 × 5120) from the padded-f32 kernel.

**(a) MMVQ routing (lower the 24M gate) — not bitwise, out immediately.**
MMVQ quantizes activations to q8 and then dp4a; the padded kernel consumes
f32 directly. **Different accumulation semantics ⇒ a zero-byte gate is
impossible**; any routing change must go through D3a's tolerance session.
Within the bitwise track this road is closed (D3-7 2b later took the
tolerance track and landed, doc 72).

**(b) NSG 2→1 row→warp remap — bitwise-green, measured a loss.** The padded
kernel (`q6_k_q8_mmvq_padded` family, `NR0=2` rows/warp, `NSG=2` warps/block):

```cuda
const int NR0 = 2;                  // 2 rows per warp
const int NSG = 2;                  // 2 warps per block (block = 4 rows)
int r0 = (blockIdx.x * NSG + warp_id) * NR0;
```

The remap: kernel+launcher change NSG 2→1 together, rows still 2/warp, warp
count doubled (as recorded); each warp still streams the full activation row
for its 2 rows (f32 id 5120 = **20 KB**), so y re-read traffic grows **1:1
with the warp count**. Result: all bitwise gates green, but the nsys
same-window kernel went **36.4 → 39.9 µs (+9.6%)**, wall tg128 **−0.74%** /
@3254 **−0.33%** → reverted.

Veto mechanism: at this shape the padded kernel is **not warp-starved**
(134.8 GB/s is an L2/latency composition problem, not a parallelism deficit);
within byte-identity the padded kernel's only free knob is rows-per-warp,
and **2 is already the sweet spot**.

### 3.4 D3b-1c: dynamic output-head block size — the mechanism does not exist

lm_head npair = 160 (id 5120) means **96 threads of a 256-thread block
idle**. The change: launch 160-thread blocks (5 warps, rounded to warp
count), with `mmvq_block_reduce` bounded by the actual warp count (idle
threads contribute only exact +0.0 terms ⇒ bitwise-safe; the reducer
`src/cuda_kernels.cu:1269` has a fixed 8-slot `warp_sums[8]`, extra slots
zero for a narrow block). Bitwise all green, but 14B wall **+0.04% / +0.09%**
— noise level.

Veto mechanism (an arithmetic veto, not measurement uncertainty): GB10's SM
thread limit is **1536**. 6 × 256-thread blocks already fill the 1536 thread
slots (960 live); 9 × 160 = 1440 live — **the gain mechanism of "eliminating
idle threads" does not exist on this SM**: the live-thread increment a narrow
block buys (1440 vs 960) is bounded neither by thread slots nor by occupancy
(at this shape the blocks are far below the SM's block cap). 7B @1641's
+0.26% (SEP) was real but sub-bar (npair 112 → 144 idle, a smaller magnitude).

### 3.5 Pitfalls

- **dump-gate pool-slot aliasing**: informational node dumps like
  `node{3,5,8}_prefill.f32` read recycled pool slots, and node→slot identity
  **drifts with binary layout** — in 1a's contrast they appeared as
  "contents byte-equal but slots swapped". From then on the bitwise gate
  checks only logits (both phases)/kv/decode-node files; same-size prefill
  node slot differences are recorded as aliasing, not value differences.
  (This artifact was cited repeatedly in D3-5/D3-8 as the "documented
  class".)
- **Dispatch gate with only a lower bound**: 1b's `id > 8192` had no upper
  bound, letting 7B ffn_down (npair 592) silently drop units — every bitwise
  gate compared v2_pf-to-v2_pf binaries and **can never see work the
  dispatch drops** (details and cost in §5). This is the campaign's most
  expensive pitfall, dug out and fixed by D4-2 (doc 74).

## 4. Verification

1b's (the only landed item) gate chain, and what each gate defends:

- **dump gate**: 114/114 `MINFER_GRAPH_DUMP` files vs the pre-change binary
  byte-identical (logits prefill+decode, kv0, all 48 layers' KV, node dumps)
  — defends against a kernel scheduling change moving any value;
- **greedy gate**: the `-n 256` token stream byte-identical — defends
  against end-to-end cumulative drift;
- **suite**: **169/0/3** (+ the FA trio, split-decode parity,
  replay-bit-parity by name) — defends against collateral damage to
  attention/prefill/capture;
- **the by-construction argument**: same unit mapping, same dp4a tree,
  character-identical accumulation statements, ascending u, same reduction —
  defends against the fluke of "gates green but mechanism wrong";
- **window anchoring**: 14B tg128 22.80–23.11 (D3-1 window 22.81), @3254
  21.01–21.11 (21.25); 7B tg128 48.03 (guard 49.4, co-tenant window; 49.47
  after 1b clears the guard), @1641 46.66 (guard 48.2); pp512 guard unmoved
  (prefill unchanged, spot-check 2055 tok/s) — defends against co-tenant
  drift disguised as gain (the r59b rule).

1a(b) and 1c passed the same bitwise gates (all green) and were then vetoed
on the wall clock — **the bitwise gates prove "not broken", the wall clock
proves "not better"**; neither suffices alone.

## 5. Results

| Lever | Status | Kernel | Wall (interleaved 3× median) | Verdict |
|---|---|---|---|---|
| **1b** v2_pf (npair>256 dual-unit loads hoisted) | **LANDED** `f1825b5` | — | **14B** tg128 22.80→22.90 (**+0.44%**), @3254 21.01→21.06 (**+0.24%**, SEP); **7B** tg128 48.03→49.47 (+3.0%, min-new > max-base), @1641 46.66→47.94 (+2.7%, SEP) | landed; 7B numbers voided, see CORRECTION below |
| **1a**(a) MMVQ routing (lower the 24M gate) | REVERTED | — | — | not bitwise (q8 vs f32 accumulation semantics), needs a tolerance session → landed separately as D3-7 2b |
| **1a**(b) NSG 2→1 remap | REVERTED | 36.4→**39.9 µs** (+9.6%) | tg128 −0.74%, @3254 −0.33% | y re-reads grow 1:1 with warps; the padded kernel is not warp-starved |
| **1c** 160-thread blocks | REVERTED | — | 14B +0.04%/+0.09%; 7B @1641 +0.26% (sub-bar) | 1536 threads/SM: the gain mechanism does not exist |
| **D3b-2** short-KV combine skip | vetoed by analysis (not implemented) | — | — | see below |

(SEP = strict separation in same-window interleaved pairing: min-new >
max-base, paired intervals non-overlapping — a stronger verdict than
medians, defined by the r59b window protocol, see doc 77.)

**D3b-2 (vetoed by analysis, the door closed before code)**: the
single-split path is bitwise-equal to 32-split partial+combine only at
nkv=1; a real decode step's combine merges ~nkv/chunk live partials with
`exp(mx−gmx)` rescaling, **reordering the float summation relative to the
serial online-softmax chain** (D1 measured split-count reordering:
ndiff ≈ 3.6e-3, max\|Δ\| ~3e-9). And the split grid is frozen by CUDA-graph
replay capture (the kernel reads positions at runtime; branching at capture
time would lock the captured step's shape for the whole session). The
bitwise-reachable residual — combine early-exit on empty splits (contributing
exact +0.0) — is worth ≤ ~15 µs/step, below every bar; the real ~166 µs/step
prize needs tolerance gates.

**D4-2 CORRECTION (2026-09-09, must be read together with this doc)**: **1b's
two 7B legs are void**. v2_pf's dispatch gate `id > 8192` had no upper
bound; 7B ffn_down (id 18944 → **npair 592**) entered and each thread's
`u1 = tid+256` covers only up to 511 — **units 512..591 (80 of 592 = 13.5%
of the down-q6K work) were never computed**. 7B's +3.0%/+2.7% was mostly
"13.5% less work computed", not pipelining gain; and since all bitwise gates
compare v2_pf-to-v2_pf binaries, **structurally the dropped units are
invisible**. The fix (`b31084c`, doc 74) added the upper bound
`id ≤ 16384` (npair ≤ 512) to the gate, with higher shapes taking the v2
loop form; after re-anchoring 7B returned to its honest baseline. **The
pipelining gain itself is the 14B-scale +0.2–0.4% class — the two 14B legs
(+0.44%/+0.24%) are unaffected and stand**.

Window anchors (this session's interleaved pre-binary medians): 14B tg128 22.80–23.11, @3254 21.01–21.11; 7B tg128 48.03 (guard 49.4), @1641 46.66 (guard 48.2); pp512 guard unmoved.

## 6. Lessons

1. **decode's attribution unit is GB/s, not tok/s**: in a step that is 93.4%
   weight streaming, "which matmuls sit below the 220–225 GB/s class" is an
   executable question list; the three-straggler list directly generated the
   1a/1b/1c levers and the later D3-7 2b landing.
2. **Admit the bitwise track's boundary in advance**: routing-class changes
   (q8 vs f32) are out of bounds by nature; remap-class changes (y re-reads
   1:1) are in bounds by nature but the mechanism may not exist — compute
   the mechanism first (warp count, y bytes, SM thread slots), then write
   code; 1c could have been vetoed by the 1536-thread arithmetic before any
   code, but was instead vetoed only after writing and measuring.
3. **Dispatch gates need bounds on both sides; bitwise gates cannot see
   dropped work**: 1b's unbounded-upper gate let 7B decode run for two days
   missing 13.5% of its down-proj work with every gate green — tests can
   only prove "two binaries did the same thing", not "dispatch made the
   kernel do all the work" (D4-2's core lesson and this doc's most expensive
   line).
4. **Analytical vetoes are also output**: D3b-2 used D1's reordering evidence
   + the replay-freeze mechanism to close a seemingly ~166 µs/step prize to
   ≤15 µs, saving an entire tolerance session — run the bitwise-feasibility
   math before writing code.

---
← 66 · [Index](./README.md) · 68 →
