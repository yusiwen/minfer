# 45 · r42 — q6_K stage-wide dsc scale reads (REVERTED)

> **Result**: whole-prefill 2607.5 → 2602.6 tok/s (**−0.19%, within the noise band**); L1TEX throughput 33.64 → 26.45% (the dsc data volume really did fall), but the `long_scoreboard` share **did not move at 33.6%** (3.6 → 3.8 cy) → the premise was falsified, reverted. r41's "the residual is mainly dsc reads" attribution does not hold for the dsc path.
> **Commit**: `a1421e6` (**record-only commit** — the code change was reverted; the tree is at the r41 shape, cmp-match with HEAD). **Date**: 2026-09-05.

> **Code provenance note**: r42's code change is gone from the tree, and the record commit `a1421e6` contains docs only (verified with `--stat`: only `docs/CUDA_OPTIMIZATION.md` +52 and `docs/LLAMA-CPP-MMQ-ANALYSIS.md` +27). Per STYLE hard rule 0, this doc takes its evidence from narration + current-tree code (the post-revert shape, which is also the pre-change shape); the trialed change's shape is described from the record's text.

## 1. Background — where things stood

r41's uint4 group loads cut the B-expand fetch width 16-fold, `long_scoreboard` fell from 85.5% to 33.6%, and whole-prefill gained
+30.7% in one step. The winner handily left behind the next map: r41's record explicitly spelled out its own residual's composition — "the remaining stall (33.6% L1TEX) is now mainly the dsc
`d·sc` byte/16-bit reads (uncoalesced) plus the KSPLIT=2 intrinsic".

To r42 this attribution looked almost like a success that could be copied directly: widening B-expand's narrow loads won, so why wouldn't widening dsc's narrow loads win? The dsc reads' shape is
**three narrow LDGs per (row, chunk)**: the f16 `d` sits at in-block offset 208 (1×u16), the two sub-block scales `sc[2c%16]`, `sc[(2c+1)%16]` start at offset 192 (2×u8); a KDR=2 kt window is
**6 narrow loads per row**. Fewer than B-expand's 32, but the same "narrow, scattered, immediately consumed" shape — and the scale reads are mixed into the staging loop, each one a potential seed
of a scoreboard event.

Where this step's absence would stall: the 33.6% `long_scoreboard` is the next visible ceiling; if it really is driven by dsc's narrow loads, one more widening of the same kind should knock it
down. r42's value is not in the gain but in **using one clean experiment to veto this shortcut**, forcing r43's instrumentation upgrade.

**How reasonable the hypothesis was at the time.** From the layout, the dsc reads are even more "scattered" than B-expand: B-expand's ql/qh are contiguous runs within a block (r41's uint4 merges
at least within a group), while dsc's three reads sit at three separate in-block offsets (192+s0, 192+s0+1, 208); across rows, adjacent od rows j and j+1 have block bases `nsb·224` B apart — the
two rows' dsc reads never land in the same sector. So the intuition "these narrow reads are manufacturing L1TEX pressure" is entirely sound — after r42 we know it holds in the **throughput**
sense, not the **critical-path latency** sense. The intuition was not wrong; what was wrong was treating throughput pressure as the stall source.

## 2. Principle — the GPU mechanism

**The trialed change (as described by the record; not on the tree).** Widen the dsc reads per stage window: within a KDR=2 window, the two chunks' 4 scale bytes merge into **one u32** (4 contiguous
B from `blk+192+s0`; s0 is even and the block base is 16-aligned, so 4 B alignment holds) and `d` merges into **one u32** — each (row, window) drops from 6 narrow loads to **2 wide loads** (~3×
instruction cut on this path); the multiplication order is unchanged and `dsc = d·sc` is bit-identical; the same `(bstride & 15) == 0` alignment gate.

**The hypothesis it tests, and why the hypothesis could be wrong.** r41's win mechanism was twofold: it cut the number of loads, and those loads were **serially consumed** (recomb→STS follows
right behind), so the L1TEX round-trip latency could not be hidden. The dsc reads satisfy the first half (many, narrow loads) but not the equivalent form of the second half — if the dsc loads'
results are not eaten immediately by a **tightly dependent** instruction, then each load's latency is already covered by other warps or by subsequent independent instructions, and swapping 6
narrow loads for 2 wide ones just **moves fewer bytes and issues a few instructions that were never blocking**.

**The divide between the two metrics (this doc's core concept).**

- **L1TEX throughput %**: a data-plane metric — the byte volume the L1TEX pipe moves per unit time as a fraction of peak. Many narrow loads, scattered bytes → a high number.
- **`long_scoreboard` (CPIStall)**: a latency-plane metric — cycles a warp spins because some instruction's input dependency (an earlier load's result) is unmet. It cares only about
  **unhidden round trips on the critical path**.

r41 happened to move both at once (what it cut was a serially consumed chain), which made people assume they always move together. r42's experiment design happens to pull them apart: bytes down,
throughput down, stall unmoved — proving that on the dsc path **only the data plane moved**. Where was the latency hiding? The record's refined answer was later nailed down by r43: at the
**consumer end** (`I2F.S8` waiting on the `d·sc` result), which widening the load end can never reach.

**The arithmetic of bytes vs transactions (why "traffic falls" ≠ "latency falls").** The data the dsc path must move is fixed: per (row, chunk) = 1×u16 `d` + 2×u8 `sc` ≈ 4
B; per block per kt (128 rows × 2 chunks) = 1,024 B — **not one byte more or less** before or after widening. What changes is the **transaction structure**: in the narrow shape these are 768
`LDG.E.U8/U16` (most landing on different bytes of the same 32 B sector → low sector utilization, many instructions); in the wide shape 256 u32s → transactions ÷3, sectors fall with them, and
pipe utilization (throughput %) drops from 33.64% to 26.45%. **But the L1TEX round-trip latency behind each load is not one cycle shorter** — if those latencies were never consumed by the
critical path anyway, the saved transaction overhead shows up only as a −1.8% kernel duration, not in the stall. That is the complete mechanism of "traffic down, stall unchanged".

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Inherit r41's stage-level granularity**: one dispatch per thread per kt window, dsc's two wide loads folded into the same staging macro, no new synchronization, no new buffers — the variables
  converge to "only the load width changed", which is what makes the experiment clean.
- **Why widen only to (row, window) granularity and not merge across rows**: dsc's mergeable bytes exist only **within one row's block** (the sc region from 192 + the d at 208, ≤16 B apart);
  scales across od rows are `nsb·224` B apart, and "merging" them is not widening but gather — swapping one contiguous-segment load for several scattered-address accesses only increases
  transactions. So (row, window) is the only legal widening granularity under this layout; r42's shape is not design conservatism, it is geometry.
- **Bit equivalence first**: the multiplication order of scales and `d` keeps the per-chunk original order (`d·sc0` first, then `d·sc1`); the widening changes fetching, not arithmetic, and dsc
  and the final output stay bit-identical — the verification gates (parity/greedy) need no exemptions.
- **The same alignment gate**: reuse `(bstride & 15) == 0`; the raw 210-B test path keeps narrow reads.

### 3.2 Key code

The trialed change is not on the tree (reverted; see the note at the top). What the tree keeps is **the narrow-read shape of the dsc reads** — the object r42 worked on, and the post-revert status
quo (after r53 it survives in the `RAW_STAGE_Q6K_BT` macro as the fallback when the `W_dsc` plane is absent):

```cuda
// src/cuda_kernels.cu — the dsc section of RAW_STAGE_Q6K_BT (narrow-read fallback arm)
// per (row, chunk): 1×u16 d (offset 208) + 2×u8 sc (from offset 192) = 3 narrow LDGs
for (int x = threadIdx.x; x < MMQ_NBJ * KDR; x += blockDim.x) {
    const int r = x % MMQ_NBJ, kd = x / MMQ_NBJ;
    const int j = j0 + r, c = (kt) * KDR + kd;
    float dsc0 = 0.0f, dsc1 = 0.0f;
    if (j < od && c < nchunk) {
        const uint8_t* blk = W + (size_t)j * ((size_t)nsb * bstride)
            + (size_t)(c >> 3) * bstride;
        const float d = h2f(*(const uint16_t*)(blk + 208));       // f16 d: narrow read 1
        const int s0 = 2 * (c & 7);
        dsc0 = d * (float)(int8_t)blk[192 + s0];                  // sc even: narrow read 2 (→ I2F.S8)
        dsc1 = d * (float)(int8_t)blk[192 + s0 + 1];              // sc odd: narrow read 3 (→ I2F.S8)
    }
    sdsb[(size_t)kd * MMQ_NBJ + r] = make_float2(dsc0, dsc1);     // consumption: I2F right behind the load
}
```

Of the three narrow reads, the two `int8_t→float` conversions are exactly r43's PC-sampling-named `I2F.S8` consumers — **the stall is booked to those two conversions, not to the loads**. r42
worked the load end, which is prescribing at the wrong address.

What the real consumer-end fix looks like — r56's `W_dsc` f32 plane (precomputed `d·sc` at registration; the live shape on the tree):

```cuda
// src/cuda_kernels.cu — the dsc section's W_dsc arm (r56): I2F and the scale reads leave the hot loop together
if (W_dsc != nullptr) {
    const int nc2 = MMQ_NBJ / 2; /* 16-B groups per kd (2 float2s) */
    for (int g = threadIdx.x; g < KDR * nc2; g += blockDim.x) {
        const int kdd = g / nc2, m = g % nc2;
        const int j = j0 + 2 * m;
        const bool full = (j + 1 < od);
        gemm_cp16(                                                     // cp.async: also made asynchronous along the way
            (__half*)(void*)(sdsb + (size_t)kdd * MMQ_NBJ + 2 * m),
            (const __half*)(const void*)(W_dsc
                + ((size_t)(c0d + kdd) * (size_t)od + (size_t)j) * 8),
            full);
    }
}
```

The contrast makes the lever-class difference visible: r42 swapped the **fetch shape** (6 narrow → 2 wide, I2F still there); r56 swapped the **work's location** (`d·sc` computed at registration,
leaving one contiguous 16 B copy in the kernel, I2F gone entirely). The latter landed at +2.35% — which also proves in reverse that r42's veto was not "this path is hopeless" but "this lever
class is hopeless".

### 3.3 Pitfalls

- **No pit at the correctness gate**: parity 1/0 (raw+padded) and greedy-32 byte-identical passed first try — the arithmetic never moved, only the fetch shape did. This step's "pitfalls" are all
  in **diagnosis**.
- **The "the metric improved" trap**: ncu showed L1TEX throughput 33.64 → 26.45%, Memory 30.50 → 26.77%, kernel duration
  −1.8% — three numbers all "getting better". But watching only throughput or duration would misread −1.8% as "right direction, insufficient force", leading to doubling down on this path. **When
  the constraining metrics (stall share 33.6%, wall −0.19%) don't move, all the improvement metrics moved for nothing**.
- **No-Eligible 58.0 → 58.6% is also a signal**: the issue-port wait share did not fall, further corroborating that the bottleneck is not issue-side volume but waiting on the dependency chain.

## 4. Verification

- **Parity dump (raw + padded) 1/0**: kernel output bit-identical to the CPU reference — defends against fetch/reassembly errors introduced by the widened loads (the byte-order trap when
  reassembling scale bytes into a u32).
- **greedy-32 byte-identical**: the end-to-end decoded token stream unchanged — defends against the cumulative class of errors where the numbers "look right" but the decode trajectory drifts.
- **ncu L1TEX throughput / Memory throughput
  before/after**: confirms the mechanism really moved (bytes/transactions really did fall) — rules out the mundane explanation "the change never took effect", guaranteeing what is vetoed is the
  **hypothesis**, not the **implementation**.
- **ncu CPIStall `long_scoreboard` before/after**: 3.6 → 3.8 cy, share 33.6% → 33.6% — **the key veto evidence**: the target symptom did not move.
- **No-Eligible before/after**: 58.0% → 58.6% — the issue-port wait share did not fall, corroborating the bottleneck is not on the issue side.
- **whole-prefill A/B**: 2607.5 → 2602.6 (−0.19%), same-window interleaved median — far below the +1.5% landing bar, the wall-level final word.

## 5. Results

| Metric | before (r41 tree) | after (r42 trial) |
|---|---|---|
| whole-prefill | 2607.5 | 2602.6 (**−0.19%**, noise; bar +1.5%) |
| kernel duration | — | −1.8% |
| L1TEX throughput | 33.64% | 26.45% |
| Memory throughput | 30.50% | 26.77% |
| `long_scoreboard` | 3.6 cy / 33.6% | **3.8 cy / 33.6% (unmoved)** |
| No-Eligible | 58.0% | 58.6% |

**Veto mechanism**: the change drove its direct mechanism (dsc byte volume, fetch count) fully into place, yet both the attributed target symptom (33.6% `long_scoreboard`) and the final gain
(wall) stayed unmoved — the hypothesis "the width of dsc's narrow loads drives the residual stall" was cleanly falsified. Keeping a zero-gain width branch would only add a SASS-comparison burden
to every later step, so the whole segment was reverted and the tree keeps the r41 shape.

**Metric behavior of r41 vs r42 side by side** (same kernel, same metrics, two back-to-back rounds):

| Metric | r41 (uint4 B-expand, +30.7%) | r42 (wide dsc reads, −0.19%) |
|---|---|---|
| L1TEX throughput | 14.8% → 33.6% (↑: the pipe really got busy) | 33.64% → 26.45% (↓: transactions really got fewer) |
| `long_scoreboard` | 85.5% → 33.6% (**↓ moved with it**) | 33.6% → 33.6% (**did not move at all**) |
| compute | 27.0% → 37.8% | — (no improvement) |
| whole-prefill | +30.7% | −0.19% |

r41's widening hit a **serially consumed chain**, so the two metrics had to move together; r42's widening hit **transaction overhead that was not on the critical path**, so only the throughput
plane moved. Read side by side, the decoupling of "throughput improved" from "wall improved" is plain — this comparison was the most persuasive attribution evidence before r43.

**Under what future conditions a retry is worthwhile**: only when the next profile shows `long_scoreboard` booked to dsc's **load instructions** (not their consumer ALUs) does widening become a
candidate again. r43's PC-sampling then delivered the final verdict: the dsc-side stall is booked to the `I2F.S8` consumer (26% of the residual) — a **latency source, not a width source**. The fix
that finally landed therefore changed lever class: r56's `W_dsc` f32 plane precomputes `d·sc` at registration, moving the consumer end out of the hot loop together with the scale reads.

## 6. Lessons

1. **Cutting bytes is not cutting latency**: the stall is booked to the **consuming instruction**; profiles must chase down the dependency's downstream op, not stop at the load.
2. **Throughput-class and stall-class metrics can move in opposite directions**: r41 making them move together is the special case (a serially consumed chain), not the rule; every time, list the
   constraining metric separately and check whether it moved.
3. **A zero-gain but mechanism-complete experiment is an instrument**: r42's −0.19% bought the conclusion "dsc width is innocent" and directly named the next tool — warp-stall source-level
   sampling (r43).

---
← [44 · r41 q6_K B-expand uint4 widen](44-r41-q6k-bexpand-uint4.md) · [Index](./README.md) · [46 · r43 PC-sampling attribution](46-r43-pc-sampling-attribution.md) →
