# 33 · r30 — SWAR unpack: the compiler already did it (REVERTED)

> **Result**: the word-granular SWAR B-nibble unpack measured +1.09% (un-warmed) / +0.54% (warmed) —
> noise level, below the +1.5% bar; SASS reconciliation +2 SHF/+3 LOP3, everything else identical class
> by class = behaviorally equivalent machine code. **Reverted (cmp-verified = HEAD)**. r29's kd-unroll
> had already induced ptxas to perform exactly this CSE — "read the SASS before writing the lever"
> thereby became this campaign's standard up-front gate.
> **Commit**: `0071b31` (record commit, docs-only — the code experiment never
> entered the tree). **Date**: 2026-09-04.

## 1. Background — where things stood

After r29 took +2.80% with a single `#pragma unroll`, the NB kernel's integer-ALU surplus versus llama
still stood at ~2.1×/MAC (1.598 vs 0.751 e-3/MAC — what remained after r29 cut 25%, from a starting
2.85×) — only the top of the instruction-stream mountain had been shaved off. In the MMQ analysis
doc's (§11) task list sat a candidate recorded since Direction A was chartered — **Task 2: replace the
per-chunk raw-nibble unpack with a word-granular SWAR** — and its paper ledger was tempting: "one
32-bit word holds 8 nibbles; read once, ~3 bit ops produce both lo and hi copies; LDS count cut by
more than half".

Before occupancy was fixed (the r25 era) this candidate was never prioritized: under 1 block/SM both
the mio/longsb stalls and the instruction surplus hid inside idle issue slots. Now occupancy had
doubled and issue slots were contended, so "issue half as many shared loads" once again looked like
real money. r30's plan was therefore the textbook three steps: **standalone byte equivalence → kernel
integration → interleaved A/B**.

But this step added one new gate to the campaign: **SASS-first** — before touching the integration,
`cuobjdump -sass` the r29 kernel and take the B-raw path apart. That gate is r30's methodological
legacy to the rest of the campaign: r32 (staging addressing already hoisted out of the loop by ptxas)
and r33 (hybrid inner loop whose ported SASS is byte-identical) both reused the same veto pattern.

One more time skew needs spelling out: Task 2's original ledger in the §11 task list was written
against the **r28-era kernel shape** — the 8 chunks each loaded their own words, 32 LDS.32 per lane
per k-tile; word-granular SWAR reads once and produces two copies, saving more than half on paper (the
proposal's own words: "4× fewer LDS"). Once r29 landed, that ledger's implicit premise — "the chunks'
loads are independent of each other" — had already been eliminated by the compiler. The lever itself
did not change; what changed is the thing it was meant to optimize. All of r30's work was quantifying
this skew.

## 2. Principle — the GPU mechanism

**SWAR** (SIMD Within A Register): use whole-word integer bit ops to process several sub-fields of a
word in parallel — here, read one 32-bit raw word (8 4-bit nibbles) and use shift+mask to produce the
low-half and high-half nibbles simultaneously, instead of each chunk loading and extracting on its
own.

First the paper ledger. The NB kernel has `KDR=8` chunks per k-tile; each chunk per lane unpacks 2
B-fragments (the nh=0/1 od rows) × 2 32-bit words:

- **Per-chunk independent loads (naive ledger)**: 8 kd × 2 nh × 2 words = **32
LDS.32** / lane / k-tile.
- The key structural fact: chunks `2p` and `2p+1` unpack from the **same pair of
words** (same `p = sg>>1`; only `is_hi = sg&1` differs — the low nibble goes to one chunk, the high
nibble to the other). There are only **16 unique words**.

Word-granular SWAR's selling point is compressing those 32 loads into 16 "read-once, produce-two". But
r29's kd-unroll happened to lay the 8 iterations out statically, letting ptxas see through the
"adjacent chunks share a word" fact for the first time — **it had already performed this CSE**. SASS
evidence (SM121, `cuobjdump -sass`):

- The B-raw path is already **16 × LDS.32 = read-once**: each unique word is
loaded exactly once;
- The lo use is a plain `LOP3 v&M`; the hi use is `SHF.R.U32.HI + LOP3` —
instruction-for-instruction isomorphic to handwritten SWAR's "read once, shift+mask to produce lo/hi".

In other words, the mechanism the SWAR proposal promised (halved loads + minimal bit ops) **is already
in the binary**. A source-level SWAR rewrite would, at best, make ptxas re-discover the same schedule
(SASS unchanged); at worst it would add explicit carry variables and reordered instructions, pushing
register pressure and issue count up. It cannot beat "what the compiler already emits" unless ptxas's
schedule happens to be suboptimal. r30's value was turning that sentence into a measured fact and
hardening it into a rule.

The extraction arithmetic is priced identically on both sides, which is also why the SASS
reconciliation was doomed to show only a ±few-instruction residual. Per unique word: the lo use costs
1 LOP3 (`v & M`), the hi use 1 SHF + 1 LOP3 (`(v >> 4) & M`); the two unique words per kd-pair total
2×LOP3 + 2×(SHF+LOP3), and the handwritten SWAR version is exactly the same. Load side: both versions
issue 16 LDS.32 (8 unique word-pairs × 2 words). Bank behavior is unchanged too — same address set,
same conflict distribution. **The only degree of freedom between the two versions is instruction
scheduling order**, and scheduling is precisely ptxas's job.

Why the CSE only appeared after r29: CSE requires the compiler to see both uses of the same word
within one visible scope. In the per-chunk loop the two uses belong to def-use chains of different
iterations, and ptxas does not merge across iterations; once unrolling spreads the 8 iterations into
one basic block, the redundancy is directly exposed and the merge is routine dataflow analysis. Put
differently, **SWAR's benefit was always a free byproduct of r29's unroll** — the proposal simply did
not realize it had already landed.

Why shared-instruction count is worth chasing separately in this kernel: LDS/LDSM go through the MIO
queue, serialized with the LSU issue slots and the shared-memory bank ports; the post-r29 stall
profile (mio_throttle 18.89%) shows the MIO side genuinely backing up. But "which class of shared
instruction is backing up" must be attributed class by class — the residual list r30 left at revert
time (A-frag LDSM, sda/sds scale reads, staging indexing, epilogue) is the output of exactly that
attribution. After the CSE, the B-unpack class is neither the largest family nor a
further-compressible one, and its MIO-side suspicion is hereby cleared.

## 3. Implementation

### 3.1 Design choices (gate order: verify the map first, then the machine code)

The experiment advanced through three gates; a failed earlier gate stops entry into the next:

1. **Standalone byte equivalence** (`/tmp/minfer_nb/b_swar_validate.cu`): the
word-granular variant was cross-checked against the validated ldmatrix B-fragment reference, sweeping
all 8 sg × 32 lanes × 4 registers — **0 mismatches**. This gate defends against layout errors
introduced by "re-deriving the lane/word→fragment byte mapping" (r28's top risk was exactly this).
2. **SASS-first**: disassemble the r29 kernel's SASS and count the B-raw path's
instruction classes. This gate runs **before** integration — §2's conclusion comes from here. Strictly
speaking, once this gate passed the experiment's fate was sealed: to win, source-level SWAR's SASS
would have to issue fewer instructions than "the CSE the compiler already did", which is mechanically
impossible. Operationally, the SASS gate is a reproducible procedure: for an SM121 target, `cuobjdump
-sass` the NB kernel, count by opcode class (LDS/LDS.64/LDS.128/LDSM/IMMA/SHF/LOP3…), and reconcile
against the expected list — here 16 LDS.32 (B-raw read-once), 64 IMMA, a minimal SHF/LOP3 set. Only a
mismatched list leaves room for a source-level lever; when every line matches, the experiment can be
judged a loss before integration.
3. **Faithful integration measurement**: since the SASS had already ruled, the
measurement was still run — in the "faithful b_hi-carry" form (even kd reads the raw words and
produces/stages lo/hi; odd kd reuses them), keeping extraction and consumption semantics
point-for-point identical, eliminating any claim that "what was measured was not the proposal itself".

### 3.2 Key code

The SWAR code never entered the tree (cmp-verified after the revert), so there is no commit to cite;
as the contrast, what fell back to the current tree is the B-unpack it tried to replace
(`src/cuda_kernels.cu`, the post-r29 unrolled form — note each chunk branches only on `is_hi`, and the
word pair is shared within a kd-pair):

```cuda
// src/cuda_kernels.cu — mmq_raw_nb_kernel, B fragments (current tree)
{
    const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
    const unsigned M = 0x0F0F0F0Fu;
    #pragma unroll
    for (int nh = 0; nh < 2; nh++) {
        const int jj = j0w + nh * 8 + (lane >> 2);
        const uint8_t* qs = qb_raw + (size_t)jj * 128;
        const uint32_t* q0 = (const uint32_t*)(qs + p * 32 + lm3 * 4);
        const uint32_t* q1 = (const uint32_t*)(qs + p * 32 + 16 + lm3 * 4);
        uint32_t v0 = *q0, v1 = *q1;                      // ← CSE'd within the kd-pair
        b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));  // LOP3 / SHF+LOP3
        b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
    }
}
```

The SWAR proposal's source shape (illustrative, not repository code — reconstructed from the
b_hi-carry scheme recorded in §11.10): turn "each chunk reads its own" into explicit cross-chunk word
carrying —

```cuda
// illustrative (never in the tree): even chunks read the words and produce lo/hi, odd chunks reuse b_hi directly
uint32_t v0 = *q0, v1 = *q1;          // once per kd-pair only
uint32_t b_lo0 = v0 & M, b_hi0 = (v0 >> 4) & M;   // carried to the next kd
b[nh][0] = is_hi ? b_hi0 : b_lo0;     // consumption site unchanged
```

— which is exactly the shape ptxas had already generated after r29's unroll; writing it out explicitly
only extends the carry variables' live ranges.

### 3.3 Pitfalls

- **Gate 1 is necessary but not sufficient**: byte equivalence verifies that the
**mapping is correct** (the fragment bytes land where they should); it has no say on "is it faster".
r30's lesson is not "SWAR was wrong" but "correctness verification ≠ benefit verification" — the SASS
sits between them.
- **The paper LDS ledger's implicit premise**: "save half the LDS" assumes each
load happens independently; after r29's unroll that premise was already dead. A lever's benefit model
must be bound to the **current compilation artifact**, not to the kernel shape as it stood when the
proposal was written.
- **Why the faithful form matters**: the pre-revert measurement used a
b_hi-carry semantically identical to the proposal, leaving no footing for the "you measured something
else" objection — which is what gives the revert its full force.
- **The measurement gate was not skipped after the SASS gate ruled**: "the SASS
says equivalent" and "the wall clock says no difference" are two independent pieces of evidence, and
this campaign wants both: the SASS reconciliation proves the machine code is behaviorally equivalent,
the A/B measurement proves the wall clock really does not move. A revert that skips the measurement
gate leaves an open case in the "it was actually 0.1% better" scenario.

## 4. Verification

- **Standalone byte equivalence**: full sweep of 8 sg × 32 lanes × 4 regs, 0
mismatches — defends against mapping/layout errors (the r28-class risk; unrelated to benefit).
- **SASS opcode reconciliation** (`cuobjdump -sass`, SM121): LDS/LDS.64/LDS.128/
LDSM = 16/32/16/32, identical item by item; 64 IMMA identical; the only deltas **+2 SHF, +3 LOP3**;
113 regs / 0 spill — proves the two versions are behaviorally equivalent machine code, so any measured
difference can only be noise. Incidentally: the SWAR version's 113 regs is *fewer* than the
incumbent's 123 — registers were never this step's constraint axis, the instruction stream was; "fewer
registers" cannot rescue a SASS-equivalent lever.
- **Parity + greedy-32**: the integrated build's cross-check and greedy output
all green — confirms "what was measured was the equivalent".
- **Interleaved A/B (4-pair, alternating order, round-4 regression)**: un-warmed
+1.09% / warmed +0.54% — both inside the noise band, below the +1.5% bar. Alternating order (AB-BA
rotation) controls for bias in the measurement order itself (clock-ramp/thermal-drift directionality);
"round-4 regression" means the sequence regressed by the fourth round, further showing the signal had
no stable direction — a real +2.80% (r29) is monotonically identifiable from the first pair.
- **Revert gate**: cmp-verified — after the revert the working tree is
byte-identical to HEAD, ruling out an incomplete revert.

The four gates' division of labor, collapsed into one table:

| Gate | What it defends against | r30 outcome |
|---|---|---|
| Byte equivalence | Mapping/layout errors (the r28-class risk) | 0 mismatch, pass |
| SASS reconciliation | A do-nothing lever isomorphic to the compilation artifact | +2 SHF/+3 LOP3, ruled out |
| Integration correctness (parity + greedy) | Semantic drift introduced by integration | all green |
| Interleaved A/B | Phantom wall-clock gains | +0.54%/+1.09%, noise |

The ruling was made by the second gate; the last two turned it into "a revert backed by measurement"
rather than "an armchair abandonment".

## 5. Results (REVERTED: the veto mechanism)

Measured: **+1.09% (un-warmed) / +0.54% (warmed, alternating-order 4-pair median)** — noise level; the
SASS reconciliation proves behaviorally equivalent machine code. After the revert = HEAD.

**Veto mechanism**: when the SASS shows the compiler already emits the instruction stream a proposal
promises (here: read-once 16 × LDS.32 + minimal SHF/LOP3), a source-level equivalent rewrite can only
add overhead or tread water — there is no "better source shape" left for the compiler to discover,
because the endpoint is already occupied. **Retry conditions** (any one being met makes a retry
worthwhile; the common thread is that the trigger is observable in the SASS):

1. A toolchain upgrade changes ptxas's scheduling strategy — more than 16
LDS.32 reappear on the B-raw path in the SASS (the CSE disappears);
2. A kernel-shape change (unroll removed, kd-pair structure rearranged) makes
"same word, double use" invisible again;
3. A genuinely non-isomorphic alternative path appears (e.g. an ldmatrix B side)
and changes the reconciliation baseline.

Even then the working order remains: re-read the SASS first, then decide whether to rewrite the source
— the criterion is the compilation artifact, not the proposal document.

**Residual attribution (this step's real output)**: at the §11.10 revert, the NB kernel's remaining
surplus classes were pinned down — the int-ALU surplus of 1.598 e-3/MAC and the shared-path stalls
(mio_throttle 15.4%, long_scoreboard 19.0%) come from **A-frag LDSM, sda/sds scale reads, staging
index arithmetic, and the epilogue** — and **not** from byte-vs-word unpacking. The next item on that
list became r31's sda scale-read repack.

## 6. Lessons

1. **Read the SASS before writing the lever**: a source-level equivalent rewrite
that is isomorphic to machine code the compiler already emits can only add overhead (the r30 pattern —
reproduced by r32 and r33 in succession; three same-pattern vetoes).
2. **Byte equivalence verifies the map, not the gold**: 0 mismatches says "runs
correctly"; between it and "runs fast" stands the compilation artifact.
3. **Bind the benefit model to the current compilation artifact**: when the
paper ledger's implicit premise (independent loads) was destroyed by the previous landed step (r29's
unroll), the lever had to be re-valued — it could not be advanced on the old books.
4. **A REVERTED step's residual attribution is the next step's signpost**: r30's
class list fed directly into r31.

← [32-r29-nb-kd-loop-unroll](32-r29-nb-kd-loop-unroll.md) · [Index](./README.md) · [34-r31-qmajor-sda-repack](34-r31-qmajor-sda-repack.md) →


