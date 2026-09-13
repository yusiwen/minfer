# 99 · Fast-verify P0/P1 — the stall root cause corrected (three times), one bitwise micro-fix landed, the knob's EV re-priced down

**Status**: ✅ P0/P1 LANDED (2026-09-15). The 方案 B campaign opened with a
re-classification pass that **corrected doc 98's tolerance-class blanket** and
then, through four measured interventions, located the BT kernels' true
bottleneck. One bitwise-safe fix landed (B-plane XOR swizzle, −4.3% at kernel
instance level); three scheduling-side levers measured NULL; the profile now
pins the binding constraint as **register-file-capped occupancy (16
warps/SM) × L1TEX latency**, which neither bitwise nor tolerance-class
scheduling levers can move. The `MINFER_FAST_VERIFY` knob's pre-registered EV
(C_T(9) 73 → ~40–50 ms) is **not supported** by the profile and is re-priced
down to single-digit percent; the knob is **not built**.

## 1. P0 re-classification: doc 98 was too pessimistic

doc 98 closed the line claiming the remaining levers "reorder the fp32
accumulation → tolerance-class by construction". Re-reading the kernel's
accumulation structure falsifies that:

- the per-tile int mma chain (`mmq_mma_k32`, s32) is **exact integer
  arithmetic — order-free**;
- the only order-carrying step is the **per-kd fp32 fold** into `sum[]`
  (`sum[idx] += da·dsv·clow + dma·dmv`), executed in ascending kd order;
- anything that preserves that fold — load scheduling, fragment prefetch,
  tile-geometry changes, operand-placement changes — is **bitwise-safe**.

The doc-91 line-27 criterion ("moves data only → bitwise identical") already
said this; doc 98 over-generalized. Consequence: the bitwise-safe lever set is
wider than doc 98 claimed, so P1 tested its members directly.

## 2. Four interventions, one survivor

| lever | class | result (q4_K BT avg/med µs, d=8 run) |
|---|---|---|
| baseline (doc 96/98) | — | 187.6 / 142.4 |
| cp.async double-buffer staging (doc 98, reverted) | bitwise | 188.5 / 146.1 — NULL |
| compute fragment prefetch (kd+1 A/B issued before kd's mma) | bitwise | 186.7 / 144.1 — NULL |
| non-volatile int mma (ptxas scheduling freedom) | bitwise | 186.4 / 144.4 — NULL |
| **B-plane XOR swizzle (16-B granule, row-XOR)** | bitwise | **184.2 / 142.5 — kept** |

The swizzle mirrors the A-plane's r22 trick on `qb_raw`: rows are 128 B = the
full 32-bank span, so unswizzled rows alias the same bank group and the
compute's 8-row nibble reads conflicted 8-way (ncu: **40% excessive shared
wavefronts** → **eliminated**, Est. 30% claim → realized −4.3% at instance
level, −1.8% avg kernel, e2e noise). Bitwise gate: 4/4 prompts byte-identical
to the doc-98 baseline; suite 187/0/3; C_T ladder and pp512 unchanged.

**Debug note**: the first swizzle build produced garbage from a classic XOR
error — the read side mapped logical chunks `p*2` and `p*2+1` as
`(p*2)^ph` and `(p*2)^ph + 1`, but XOR does not distribute over the +1; each
chunk index must be XORed separately. One run of the output-sanity gate caught
it (see doc 98 §2 — same lesson, second instance).

## 3. The root cause, corrected three times

1. **doc 92**: "per-tile staging serialization" — falsified by doc 98's
   cp.async null (staging latency was already hidden or irrelevant).
2. **doc 96 Phase 0**: "Short Scoreboard = smem stalls" — falsified by this
   campaign's pc-sampling: the stall is **L1TEX scoreboard** (global/local
   loads), 36–40% of warp-issue cycles; the shared side is now clean.
3. **doc 98**: "remaining levers are tolerance-class" — falsified by §1; the
   bitwise-safe set was wider, but its members measure NULL anyway.

The verified structure: `sum[32] + clow[32] + fragments` pin the kernel at
~127 registers/thread → **512 threads/SM = 16 warps** (register-file ceiling,
occupancy-proof against every partition tried). At 16 warps the staging's
in-flight bytes (~8 KB/SM) cannot cover the L2-served load latency (~30–50 KB
needed), so the kernel idles at ~25–27% memory SOL / ~16% compute SOL —
**latency-bound with a hard occupancy ceiling**. Deeper per-warp pipelines,
freer scheduling, and conflict-free smem all get absorbed; block-level
parallelism (doc 92's k-split, +11.7 ms) is the only lever that ever moved
this kernel, and it is already harvested.

## 4. Verdict on the fast-verify knob (方案 B P2 — not built)

The knob's premise was that tolerance-class levers (bigger k-tiles, ldmatrix
B, deeper mma pipelining) reach C_T(9) ≈ 40–50 ms. The profile says those
levers all act on **scheduling**, and scheduling is measured dead here; the
binding constraint (occupancy × latency) is structural. The one remaining
priced lever is **weight repacking to 32B-aligned row strides** (q4_K 144 B
stride → 27% excessive global sectors): worth ~7–10% of this kernel's time at
best in a latency-bound regime, across a wide reader surface (MMVQ/MMQ/BT all
re-addressed). Re-priced EV does not justify the knob's non-bitwise contract.

**The performance line stays closed** — now with the mechanism actually
understood instead of assumed. doc 95's adaptive-d configuration remains the
production spec operating point.

## 5. Disposition

- Landed: the B-plane XOR swizzle (both `mmq_raw_nb_kernel` and
  `mmq_raw_nb_bt_kernel`; the wide kernel does not read `qb_raw`).
- Reverted: fragment prefetch and non-volatile mma (measured NULL, doc-90
  discipline).
- Not built: `MINFER_FAST_VERIFY`. Measurement artifacts in `/tmp/d96p1/`.
