# 36 · r33 — Hybrid inner-loop port: SASS fully identical, hypothesis falsified (REVERTED)

> **Result**: porting the **shape** of llama.cpp's `j0`-outer/`n`-inner inner loop into
> `mmq_raw_nb_kernel` (only the loop enumeration order changes — no layout, math, or shell
> changes): every emission-relevant gate untouched — SASS **byte-identical** to r31 (64 IMMA
> in the same order, 109 regs, same LDS/LDSM counts), parity 1/0, greedy byte-identical,
> interleaved 4-pair median **−0.25%**. **The "remaining 1.15×/GMAC residual = loop-
> organization-induced SASS codegen" hypothesis is falsified**; the census integer-ALU
> 1.66 e-3/MAC (llama 0.751) did not converge toward llama, and the line closes here.
> **Commit**: `697ef04` (docs-only record commit; the ported code was completed, measured,
> and reverted in the local tree — never landed as a code commit). **Date**: 2026-09-04.

## 1. Background — where things stood

r32 (doc 35) sealed off the "instruction count" axis entirely: staging addressing already hoisted by ptxas (source-level lever dead), epilogue structurally capped, and the remaining integer-ALU surplus = A-frag LDSM + intrinsic sda/sds decode + fp rescale, all "compiler floor". On the books, the NB kernel's per-GMAC warp instruction stream still ran **1.15×** above llama.cpp — a number with its own shrinking history: at r13's counter forensics the gap was 10.14 vs 6.06 M/GMAC (**1.67×**); r28's NB kernel (2 blocks/SM) and r29's unroll squeezed it to 1.15×. Every **conceivable, source-level cuttable instruction** had been tried or proven not to exist.

One surviving explanatory framework remained — the campaign's last open "soft" hypothesis: **is this 1.15×/GMAC residual purely loop organization — the SASS codegen difference that the loop's nesting shape induces?** ptxas's scheduler takes as input not just the instruction set but the loop's nesting shape — perhaps llama's inner loop enumerates the mmas and rescales in some particular order that lets ptxas build a tighter pipeline (different issue gaps, different register-liveness peaks, different stall landings); perhaps we only need to write the loop in llama's shape and the machine code will "converge" toward it.

The hypothesis has a history worth recording. r10 (doc 15) ported llama's **math decomposition** — when nibbles unpack, when dmin folds in, when scales multiply — and still measured 462–468 vs 470 tok/s: **the same decomposition, still 5× slower**; the residual was not in the math organization. r30 issued a method warning: the SWAR unpack was written before anyone noticed the SASS-level ptxas had long CSE'd it — **look at the machine code before acting**. r33 tests the next organization level down: no math change, only the **issue order of the mmas and rescales** (the loop's enumeration shape). If this level is falsified too, the "codegen" class of hypotheses has exactly one exit left — changing the instruction **composition** itself (r33 explicitly records it as a scope caveat, handed to r34).

Archival situation identical to r32: `git show 697ef04 --stat` contains only `docs/CUDA_OPTIMIZATION.md` +69 lines and `docs/LLAMA-CPP-MMQ-ANALYSIS.md` +47 lines — the ported code was measured and reverted; the record commit is the only carrier.

## 2. Principle — the GPU mechanism

**Two loop shapes, one DAG.** The computation inside a 32-k chunk is fixed: 4 A-frags (activations, each 16 tokens × 32k) × 2 B-frags (weights, each 8 od × 32k) = **8 mma.m16n8k32**, and each mma's accumulator is consumed by the same chunk's two-term rank-1 rescale. Draw the chunk's 8 mmas + 8 rescale groups as a dependency graph: the nodes are fully determined by the chunk geometry, and the only edges are "mma → its own rescale" — **both sources enumerate the same graph**:

- **minfer shape (g-outer × nh-inner)**: load 4 A-frags, load 2 B-frags, then `for g { for nh { mma(A[g], B[nh]) } }` — A-frag loads are amortized across the nh loop, B-frag loads across the g loop.
- **llama shape (j0-outer / k01 / n-inner)**: `for j0 { load B; for k01 { for n { mma(B, A[n]); rescale } } }` — **the B-frag (weight) load is hoisted outside the n loop**, one weight fragment reused by all `n` token-minitiles (llama's B-reuse-across-n).

Fragment-load counts, mma counts, and rescale application points are identical — the only difference is "which load is written at which nesting level in the source". **Once fully unrolled, source order is merely an enumeration order of the DAG, not a scheduling constraint**: `#pragma unroll` flattens the loop bodies into straight-line code, and ptxas's scheduler reshuffles the same DAG freely. r33's entire bet was that "the same DAG, enumerated differently, might schedule differently".

**The hypothesis's steelman — what source order can change in theory.** Stated fairly, it is not baseless: before unrolling, source order genuinely determines (a) the issue timing of loads relative to mmas (stall landings), (b) each fragment's register liveness interval (pressure peaks), (c) which load already sits outside the inner loop at the source level (approximate manual hoisting). All three are real constraints in **unrolled-less** source — the bet is that they still constrain ptxas **after** unrolling. r33 is the controlled experiment for that bet.

The hypothesis earns serious treatment because it makes two observable predictions: (1) if source order affects codegen, the ported SASS must change (instruction order, register allocation, or sync points — at least one); (2) if the SASS is unchanged, the wall-clock difference must be zero. The predictions are mutually exclusive and both cheap to test. r33's design puts both on the table and lets the SASS diff referee.

In method lineage, r33 is the last step of the "SASS forensics" three-step ladder: r30's first SASS-first (look at what the compiler already generates before writing a lever), r32's region census (where instructions live, how often they execute), r33 promoting identity itself to a criterion (**byte-identical = hypothesis dead**). All three steps share one tool (`cuobjdump -sass`) — getting cheaper and more lethal each step.

**Geometry conversion: where the 8 mmas come from.** llama's warp tile is 32 od × 64 tokens: `rows_per_warp / tile_C::I = 2` M-minitiles (`ntx = 2`), with j0 stepping by `ntx × tile_C::J`. Our warp tile is **16 od × 64 tokens** (NB kernel: 8 warps × 16 od = MMQ_NBJ 128) — under the same `m16n8k32` / `tile<16,8,int>` lane map, the port folds into **j0 = 2 od-groups × n = 4 token-minitiles = 8 mmas per 32-k chunk** — the same 8, only the enumeration order changed. The **64 IMMA** in static SASS = the KDR=8 kd expansion × 8 mmas per chunk — "64 IMMA in the same order" refers to exactly this batch's issue order.

A note on why the hypothesis tempts: llama's tile amortizes the weight B-frag load across all token-minitiles, which looks like a free locality win — the next day r36 quantified with wavefront counting that llama's fragment-load rate is **0.125 LDSM/IMMA (ours 0.5)**, so the gap really is in fragment reuse. But that is a **tiling property** (32 od rows/warp vs 16), not a loop-order property — r33 proves with SASS identity that under the same tile, enumeration order changes no load count.

**k01 degenerates → parity by construction.** llama's k01 loop steps by `QI8_1` (the 8-k sub-blocks inside one q8_1 block) because their one m16n8k32 consumes 32-k while one q8_1 tile holds several sub-blocks. In our geometry **k01 is degenerate**: one m16n8k32 covers exactly the whole 32-k chunk (`nchunk = id/32`), so the k01 loop has a single iteration. The per-32-k-chunk rescale boundaries therefore do not move, and **the fp accumulation order never changes** — numerical equivalence is not a measured gamble but constructed. It also means any SASS difference could come **only** from scheduling, never from numerical reordering — a single-variable experiment, cleanly rare.

**SASS identity = the definition of falsification.** This experiment has a logical shortcut: identical SASS ⇒ identical cycle behavior ⇒ wall-clock difference necessarily 0. So the correct experimental order is **diff the SASS first, then decide whether to run performance** — byte-identical machine code cannot produce a different wall clock; running A/B merely adds a formal number for the record. r33 is the campaign's first use of "SASS identity" as an independent gate that can terminate an experiment early.

## 3. Implementation

### 3.1 Design choices (why a "hybrid" port)

**"Hybrid" means: loop order only, shell fully kept.** The port keeps the kernel signature, the 64×128 block geometry, KD=8, all smem layouts (qa8 / the r31 q-major sda repack / qb-raw / sds), r20 split-phase staging, the r22 XOR swizzle, the launcher and `MINFER_MMQ_RAW_NB` gate, r15's two-term fp32 rank-1 rescale semantics, and the fp32 write-back. The **only** change is the compute loop's enumeration shape. This makes any SASS difference attributable solely to loop order — attributional singularity matters more than "porting more like llama".

**The fragment mapping table is validated before compiling.** The easiest mistake in an enumeration-order port is not scheduling but miscopying one line of the (g, nh, l) → (i, j) output-unit mapping — parity would catch it, but how it catches (a few ulps vs large offsets) wastes half a day. The port first tabulated, for the new enumeration, which C accumulator each mma writes and which output grid point each accumulator maps to, then checked them one by one against the original: **fragment maps validated, 0 mismatches**. Only after that does the k01-degenerate constructive parity argument truly close.

**Not porting the operand orientation is deliberate.** In llama's mma, **A = weights, B = activations** (their activation fragment goes through `load_generic`, a trivial LDS — the source comment's own words: "faster than load_ldmatrix"; only the weight fragment earns ldmatrix). minfer is **A = activations (ldmatrix), B = weights (raw-nibble register unpack)**. Flipping the orientation too would change the instruction **composition** (eliminating the A-frag LDSM class) — no longer a "loop organization" experiment. That half-step is explicitly recorded as a scope caveat, left for r34's narrow slice.

**Write the equivalence proof first, then the code.** The k01-degeneracy argument was written before the implementation: parity is guaranteed by construction, and the experiment's output space has only two points — "SASS same / different".

### 3.2 Key code

minfer's original compute loop (survived the revert in the current tree; each chunk's 8 independent mma chains, with A-frag loads and B-frag unpack):

```cuda
// src/cuda_kernels.cu:6344-6388 (mmq_raw_nb_kernel — g-outer × nh-inner, the before of r33)
// A fragments: 4 independent 16-token groups (T=64), r22 G[].
int a[4][4], b[2][2];
#pragma unroll
for (int g = 0; g < 4; g++) {                       // ← A (activations): loaded once via ldmatrix
    const uint8_t* p = qat + G[g];
    unsigned r0_, r1_, r2_, r3_;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
        : "r"((unsigned)__cvta_generic_to_shared(p)));
    a[g][0] = (int)r0_; a[g][1] = (int)r1_;
    a[g][2] = (int)r2_; a[g][3] = (int)r3_;
}
// B fragments: raw-nibble in-loop unpack (weight-side register unpack, 2 frags per chunk)
{
    const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
    const unsigned M = 0x0F0F0F0Fu;
    #pragma unroll
    for (int nh = 0; nh < 2; nh++) {
        const int jj = j0w + nh * 8 + (lane >> 2);
        const uint8_t* qs = qb_raw + (size_t)jj * 128;
        const uint32_t* q0 = (const uint32_t*)(qs + p * 32 + lm3 * 4);
        const uint32_t* q1 = (const uint32_t*)(qs + p * 32 + 16 + lm3 * 4);
        uint32_t v0 = *q0, v1 = *q1;
        b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));
        b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
    }
}
…
// 8 independent mma chains per thread per chunk (4 A-frags x 2 B-frags),
// all C fragments live simultaneously.
#pragma unroll
for (int g = 0; g < 4; g++)                         // ← outer loop over A frags
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)                  // ← inner loop over B frags
        mmq_mma_k32(clow[g][nh], a[g], b[nh]);      // 8 mmas, then the same chunk's rescale
```

llama's reference shape (the source the port was checked against, `mmq-vec-dot.cuh:408-437`, current upstream tree):

```cuda
// llama.cpp ggml/src/ggml-cuda/mmq-vec-dot.cuh:408-437 (the q8_1×q8_1 mma branch)
#pragma unroll
for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {         // ← outer loop over od-groups (weights)
#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        tile_B   B;
        float2 dsB[tile_C::ne/2];
        load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01,
                     MMQ_TILE_Y_K);                     // ← B frag hoisted outside the n loop (trivial LDS)
#pragma unroll
        for (int n = 0; n < ntx; ++n) {                 // ← inner loop over token-minitiles
            tile_C C;
            mma(C, A[n][k01/QI8_1], B);
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] +=
                    dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                sum[(j0/tile_C::J + n)*tile_C::ne + l] +=
                    dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;   // ← two-term rank-1 rescale
            }
        }
    }
}
```

The other big shell piece kept as-is is the rescale section — r15's two-term rank-1 fold (the `d*sc` main term + the `-dmin*m` rank-1 term) and r31's sda reads (two LDS.128s per warp) survive unchanged in the port, because they define the fp-accumulation numerical contract:

```cuda
// src/cuda_kernels.cu:6399-6427 (mmq_raw_nb_kernel's sda reads + rescale — kept shell, excerpt)
// r31: Q-major sda repack — one uint32 per token; the group-region
// split (g/2 region block, q*16B stride) makes each warp LDS.128
// read 8 unique 16B words at 16B stride = bank-conflict-free.
const uint32_t* sda_blk = sda_q + (size_t)kd * MMQ_NBI
                          + (size_t)(lane >> 2) * 4;
const uint4 s0 = *(const uint4*)(sda_blk);
const uint4 s1 = *(const uint4*)(sda_blk + 32);
#pragma unroll
for (int g = 0; g < 4; g++) {
    …
    da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));      // d (f16 half-word)
    sa_q[0] = (int)(short)(w0 >> 16);                  // ssum (i16 half-word)
    …
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        #pragma unroll
        for (int l = 0; l < 4; l++) {
            …
            sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];  // main term d*sc
            sum[idx] += dma[l >> 1] * dmv[nh][l & 1];                 // rank-1 term -dmin*m (r15)
        }
}
```

The port (never committed) replaced minfer's `for g { for nh }` with the `for j0 (2 od-groups) { for n (4 token-minitiles) }` above: the B-frag load hoisted outside the n level, the A-frag load still outermost (once per chunk, unchanged), the rescale fold copied verbatim in the `sum[…] += …` two-term form. All load counts, mma counts, and rescale application points map one-to-one to the before (the §3.1 mapping-table check targets exactly this). The two enumerations side by side:

```text
before (minfer):     load a[0..4]; load b[0..2];
                     for g in 0..4 { for nh in 0..2 { mma(a[g], b[nh]); rescale(g,nh) } }

ported (llama shape): load a[0..4];                     // ← still outermost, once per chunk
                      for j0 in 0..2 {                  // od-groups (weights)
                          load b[j0];                   // ← hoisted outside the n loop
                          for n in 0..4 { mma(a[n], b[j0]); rescale(j0,n) }   // token-minitiles
                      }
```

Both are 8 mmas, 8 rescale groups, 6 fragment loads per chunk — **isomorphic DAG, different enumeration order**. ("Hybrid" gets its name here: llama's loop shape + all of minfer's shell.)

### 3.3 Pitfalls

- **A SASS diff must be same-version, same-flags.** Byte-identity is meaningful only under the same ptxas and the same compile options; compiling each side separately and diffing `cuobjdump -sass` means any flag drift manufactures fake differences.
- **Mapping-table errors and scheduling differences are two diseases — do not use one medicine.** The biggest risk in an enumeration-order port is a miscopied (g,nh,l)→(i,j) mapping (a parity disease), not scheduling (a perf disease); only the 0-mismatch mapping check gives the SASS comparison its "pure scheduling" interpretive authority.
- **Do not read "the census did not move" as "the experiment was botched".** After the port the integer-ALU census read 1.66 e-3/MAC, unmoved — that is not a sign the port failed; it is positive evidence that "source order does not affect instruction composition". The census measures composition; r33 manipulated order.
- **Take the logical shortcut first.** Running the 4-pair A/B before looking at the SASS wastes a round of machine time; the SASS identity check alone condemns the experiment in ~10 minutes. Fixing "diff first, then measure" as an order is this experiment's real methodological output.

## 4. Verification

- **SASS byte-identity gate (the experiment's main gate)**: `cuobjdump -sass` comparing the port against the r31 baseline — defends against both the reverse illusion "thought we changed scheduling, actually didn't" and the false negative "thought we didn't change, actually did". Result: **byte-identical** — 64 IMMA in the same order, same LDS/LDSM counts.
- **Fragment mapping-table check (0 mismatches)**: defends against the subtlest output-unit misalignment an enumeration-order port can produce — the precondition for the SASS comparison's "pure scheduling" authority.
- **ptxas resource audit**: `mmq_raw_nb_kernel<8>` 109 regs / 0 spill, smem 43,008 B, 2 blocks/SM — all identical to r31, ruling out the bypass "loop reordering changed register allocation".
- **Parity 1/0 + greedy-32 byte-identity**: confirms the k01-degenerate constructive-equivalence argument holds on real hardware.
- **Interleaved 4-pair A/B**: −0.73% / −0.59% / +1.30% / −0.10%, median **−0.25%** — completes the record for the "wall clock unchanged" ruling.
- **Instruction census**: integer-ALU 1.66 e-3/MAC (llama 0.751), unmoved before and after the port — quantifies the "no convergence toward llama".

The gates' execution order is itself part of the conclusion: SASS diff (~10 minutes) → resource audit (as before) → parity/greedy (constructive confirmation) → A/B (the formal number) → census (quantifying non-convergence) — the cheaper the gate, the earlier it runs; the first gate delivered the verdict and the remaining four were purely record-keeping.

## 5. Results

**Hypothesis falsified, line closed.** With the loop shape swapped to llama's j0-outer/n-inner, ptxas produced SASS **byte-identical** to r31: the compiler schedules the unrolled 8-mma + rescale stream the same way whether the source enumerates g-outer/nh-inner or j0-outer/n-inner. §2's three steelman points (issue timing, liveness intervals, manual hoisting) all evaporate after unrolling — they were always ptxas's scheduling degrees of freedom, not source-level constraints. Byte-identical SASS physically cannot give a different wall clock; the measured median −0.25% (inside the noise band) is just the footnote on that logical necessity. Ported code reverted, cmp-verified = HEAD.

**The verdict, stated fully** (the §11.13 wording): the 1.15×/GMAC residual is **not** loop-organization codegen; it is an **inherent difference in instruction composition** (A-frag LDSM consumption + intrinsic sda/sds scale decode + staging index math — the regions r32 attributed) plus **nvcc/ptxas's scheduling of the whole kernel**, and source-level reordering touches neither. To change ptxas's output, the DAG itself must change.

**The verdict's later footnotes**: the next day, r36 added a mechanism footnote to the residual with wavefront counting — the MIO pipe is not scarce, and llama's real advantage is the **A-frag reuse rate** (0.125 vs 0.5 LDSM/IMMA, a tiling property) — refining but not overturning r33's conclusion; and r34 picked the "change the composition" direction out of the scope caveat and landed +9.72% (doc 37). The door r33 closed, r34 dismantled around the frame and walked through.

**The experiment's cost-benefit.** The entire cost of this falsification: one local loop rewrite (after the mapping check), two compiles + a SASS diff, one 4-pair A/B — bought the closure of the entire "loop organization" axis, plus a cheaply re-runnable gate (after any future ptxas version change, one SASS diff suffices). Against the routes it eliminated (continuing trial and error along loop shape, each round a full port + full verification), this is one of the campaign's best cost-performance "negative results".

**Veto mechanism and retry conditions**: this hypothesis's retry conditions are written precisely in the scope caveat — only ports that **change instruction composition** can move the wall: flip the orientation to A=weights (eliminating the per-tile activation A-frag LDSM, weights moving to trivial LDS), or move the A-side layout transform out of the kernel. The former was out of reach within the then-current budget; the latter was realized by r34 as a narrow slice. If a future ptxas version changes behavior, the re-check costs one SASS diff — the cheap re-inspection entrance this experiment left behind.

## 6. Lessons

1. **A loop-organization hypothesis can be falsified by SASS identity before any performance measurement**: byte-identical machine code is definitional evidence of "hypothesis dead" — diff the SASS first, then decide whether to spend machine time.
2. **After `#pragma unroll`, source order is only an enumeration order of the DAG**: issue timing, register liveness, and load hoisting are real constraints at the source level and scheduling degrees of freedom after unrolling — to change ptxas's output, change the DAG first.
3. **Split a large residual into separately falsifiable sub-hypotheses**: "codegen difference" hides three components — composition, order, scheduling; r32 sealed composition, r33 sealed order, and the remaining scheduling component can only be bypassed by changing the problem itself (r34).
4. **The mapping table precedes the compiler**: in a loop-order experiment, the output-unit mapping check (0 mismatches) is the precondition for the SASS comparison to mean anything.
5. **The scope caveat is a negative result's will**: the sentence in r33's record about "the deeper direction, and why it was not taken" became r34's work order directly — writing down clearly what was not done has more long-term value than one more measured number.

← 35-r32-finite-lever-sweep · [Index](./README.md) · 37-r34-quantize-transpose-prepass →
