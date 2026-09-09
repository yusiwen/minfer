# 61 · r58 — q4_K BT spec + cp.async-db2 transplant (REVERTED)

> **Result**: 7B pp3314-eq same-window paired A/B: transplanted 2819.3 vs baseline
> 3227.6 tok/s = **−12.6%** → reverted. The transplant itself was functionally
> clean (greedy-32 byte-identical, parity ×3 all green) — it lost on the
> performance model, not on correctness. The campaign's first case of
> "mechanism transplant succeeded, value formula failed".
> **Commit**: no repo change (code reverted without leaving a commit; the
> recorded commit `093ae41` is docs only). **Date**: 2026-09-06 (Session F).

## 1. Background — where things stood

By the time r57 was reverted, the q6_K line had already "closed its door": the
r53 W_exp + cp.async B staging bundle had pushed 7B prefill to 3176.9 tok/s,
the r56 A-side bundle (A cp.async + W_dsc plane) pushed it further to 3212.5,
the r54 opt-out gate confirmed the plane was worth its memory, and the r55
roofline audit showed swiglu already at 89% of the bandwidth roof with nothing
to harvest from a prefill CUDA-Graph either — the campaign was formally
declared CONVERGED at r55, after which r56 still dug +2.35% out of the
"already closed" pile. r57 (FA KV double buffer) broke byte identity and was
reverted.

At this point vs-llama stood at 1.035× (3212.5 vs the 3324.42 clean-machine
anchor). The last region never structurally examined was the **q4_K bt
kernel** — the raw-nibble NB kernel shaped back in the r28 era, then widened
in tile. r47's converged-regime breakdown had given it a respectable number:
the q4_K GEMM sat only 1.06× behind llama, which at first glance read as
"already at parity". But 1.06× spread over the whole prefill is still several
hundred milliseconds, and r56 had just proven the q6_K staging mechanism
(cp.async + pre-expanded plane) transplants as-is onto another family.

Session F's plan therefore had two steps: **Phase 1a** was pure measurement —
take the q4_K bt kernel apart and attribute where the 1.06× actually lived;
**Phase 2** would rank the levers by that attribution. The conclusion of the
first step spawned this transplant experiment: since the r39+r53+r56 staging
pipeline had scored three consecutive hits of +13.3/+5.03/+2.35% on q6_K,
porting it to q4_K bt looked like a "free" checklist item.

That "looked like" is exactly where this document's lesson lives.

## 2. Principle — the GPU mechanism: the pipeline value formula

### 2.1 The Phase-1a structural attribution

The baseline was re-measured first: 3219.6 median (consistent with the 3212.5
recorded at the r56 landing — the window was healthy). Then a kernel-busy
breakdown of the landed configuration:

- Whole-prefill kernel busy = 984.2 ms, of which **q4_K bt = 622.4 ms
  (63.2%)**, 166 launches in four classes:

| Class | Throughput | Relative state |
|---|---|---|
| gate/up (ffn_gu) | 7.51 G-IMMA/s | best |
| q/o (attn_q, attn_o) | 7.46 G-IMMA/s | best |
| ffn_down | 5.94 G-IMMA/s | **21% below its own steady state** |
| k/v (attn_k, attn_v) | 5.05 G-IMMA/s | **27.8% ceil-wave loss** |

- matched-nt ncu against llama: we run 7.5 vs llama's ~6–6.5 G-IMMA/s (on the
  major classes) — **the mma loop itself is not behind; it is ahead**. The
  1.06× gap does not live in the IMMA.

Conclusion: the q4_K residual lives in three places — (1) per-kt staging
exposure in the ffn_down class (each k-tile's staging phase tops out with a
stretch of unmasked global latency); (2) ceil-wave tail quantization (27.8%
in the k/v class, ≈ 26.1 ms ≈ 2.6% of the whole prefill); (3) launch gaps.

### 2.2 The transplant's value formula

The top delta r58 chose was to move the q6_K three-piece set over: KDR=2
double buffering + cp.async A/sds staging + sb-parity cp.async B window
(measured 46,080 B smem, 122 regs, LDGSTS appearing in the SASS). The
instinct that it "should earn" came from the q6_K track record. But putting
the two transplants side by side in the same formula exposes where the
instinct went wrong:

```
Transplant value ≈ exposed latency hidden − pipeline cost
Exposed latency ≈ f(how expensive the staging it replaces is)  ← the two families differ wildly here
Pipeline cost ≈ higher barrier density + lookahead depth × issue slots + smem/register pressure
```

**The q6_K transplant (won)**: before r41, B staging was the ql+qh
recombination (recomb ALU) + per-byte LDG + the dsc I2F conversion — the
staging phase itself was long and expensive. KDR=2 double buffering let that
expensive work for kt+1 overlap kt's compute, hiding a whole stretch of
genuine ALU/latency cost in compute's shadow; r53/r56 then replaced the
remaining pure copies with cp.async, hiding the LDG scoreboard latency.
**Only an expensive thing-being-replaced leaves something to hide.**

**The q4_K bt side (lost)**: look at the staging macros of
`mmq_raw_nb_bt_kernel` in the current tree (after r34's quantize-transpose
prepass, the A side is already a pure `uint4` bulk copy; the B-side qs is
also a pure `uint4` copy):

```cuda
// src/cuda_kernels.cu — RAW_STAGE_NB_BT of mmq_raw_nb_bt_kernel (current tree,
// the shape after r59; at r58 there was only this "pure copy" staging, without
// the DSC branch below)
/* ---- A: bulk LDG->STS of the pre-transposed qa8/sda (no math) ----*/
{
    const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16; off += blockDim.x)
        ((uint4*)(qa8))[off] = ((const uint4*)(qa8g + qbase))[off];      // 16B bulk copy
    const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16; off += blockDim.x)
        ((uint4*)(sda_q))[off] = ((const uint4*)(sdag + sbase))[off];    // 16B bulk copy
}
/* ---- B: bulk raw qs super-block copy (r18-style, no staging ALU) */
{
    const int sb = ((kt) * KDR) >> 3;
    for (int off = threadIdx.x; off < MMQ_NBJ * 8; off += blockDim.x) {
        const int jj = off >> 3, c8 = off & 7;
        const int j = j0 + jj;
        uint4 v = make_uint4(0, 0, 0, 0);
        if (j < od && sb < nsb)
            v = *(const uint4*)(W + (size_t)j * ((size_t)nsb * 144)
                + (size_t)sb * 144 + 16 + (size_t)c8 * 16);
        *(uint4*)(qb_raw + (size_t)jj * 128 + (size_t)c8 * 16) = v;
    }
}
```

No recomb, no I2F, no per-byte LDG — the A-side layout transform moved into
the prepass at r34, and the B side was confirmed a pure copy in the r18
attempt. **The staging phase itself was already near its floor.**
Injecting KDR=2 double buffering at this point buys:

1. **4× the barrier density**: at KDR=8 you synchronize once per 8 chunks;
   KDR=2 makes it once per 2 chunks — sync cost ×4;
2. **1-deep lookahead**: double buffering can only run one tile ahead, and on
   GB10 the global→smem round-trip latency is on the order of ~600–900 cyc;
   the copy time of one 64-token × KDR=2 tile is far shorter than that
   latency, so the lookahead cannot fill the latency hole at all;
3. **No register/ALU savings at all**: q6_K's double buffering incidentally
   moved the recomb's register round-trip out of the compute phase; q4_K
   never had that round-trip to move.

The first term of the formula ≈ 0, the second is strictly positive — **net
value negative**. −12.6% is the measured size of that negative value (much
larger than the expected "small negative", because the 4× barrier density
also disturbed the issue schedule that had been tracking well).

This is the mirror image of the r45 lesson: r45 said "a non-bottleneck kernel
getting faster does not make the wall faster" (horizontal: swap in another
kernel); r58 says "**a mechanism's cost depends on the granularity of what
it replaces**" (vertical: the same mechanism hosted on different costs).
Together they form the complete transplant criterion: **quantify the host's
staging-cost structure first, then decide whether to transplant.**

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

The transplant copied the q6_K landing version wholesale (r39's KDR=2 +
r53's cp.async B window + r56's cp.async A), adapting only the q4_K layout:

- q6_K's B-side cp.async source was the W_exp dense plane; q4_K has no W_exp,
  so the B side remained the 144 B super-block raw qs copy, but switched to
  cp.async (sb-parity rotating window);
- the q4_K dsc (scale) side kept the per-(chunk, row) `get_scale_min_k4`
  decode (the W_dsc plane was then still item 1 of the r58 Phase-2 spec, not
  yet implemented);
- smem budget: two copies of A (qa8+sda) + two B windows = 46,080 B, 122
  regs — 0 spill, resident block count unchanged.

"Build the minimal transplant first, then measure" was itself the right call
(all gates green, one clean measurement); the mistake was skipping §2.2's
arithmetic.

### 3.2 Key code

What the transplanted pipeline looks like (q6_K side, current tree): the
transplanted source vanished with the revert, but its host mechanism lives on
in the q6_K kernel — the staging macros of current-tree `mmq_raw_nb_bt_q6k_kernel`
are exactly the set moved to q4_K (the comments show the three mechanisms layered):

```cuda
// src/cuda_kernels.cu — mmq_raw_nb_bt_q6k_kernel (the r39+r53+r56 combined shape)
// r39: DOUBLE-BUFFERED staging — two copies of every per-kt plane so kt+1's
// global->smem expansion (the ql+qh recomb) overlaps kt's compute, hiding the
// B-staging latency that left r38 latency-bound.
uint8_t*  qa8    = mmq_q6k_sh;                       // [2][KDR*NBI*32]
uint8_t*  sda_q  = qa8 + 2 * KDR * MMQ_NBI * 32;     // [2][KDR*NBI*4]
uint8_t*  qb_exp = sda_q + 2 * KDR * MMQ_NBI * 4;    // [2][NBJ*KDR*32]
float2*   sds    = ...;                              // [2][KDR*NBJ]
...
/* ---- A: r56 cp.async bulk staging of the pre-transposed qa8/sda --*/
/* (r45's mechanism on top of r53: the sync LDG->STS exposed its global
 * latency at the top of every staging phase; cp.async hands it to the
 * async unit and the group wait below hides it under the previous
 * tile's compute. ...) */
for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16; off += blockDim.x)
    gemm_cp16((__half*)(void*)(qa8b + (size_t)off * 16),
              (const __half*)(const void*)(qa8g + qbase + (size_t)off * 16),
              true);
/* ---- B: KDR*32-chunk super-block window ... r53 bundle: the ql+qh recomb
 * + -32 centering ran ONCE at registration (expand_q6k_dense -> W_exp),
 * so the staging is a pure cp.async bulk copy from W_exp — no recomb ALU,
 * no register round-trip, no ql/qh reads ... */
```

Note that the premise under which this code works on q6_K is precisely that
**q6_K's staging was once expensive** (recomb + I2F), so "hiding it" had real
value; once r53/r56 eliminated the expensive part too, the q6_K side's own
cp.async gains had already converged into the bundle's +5.03/+2.35%. Moving
the same mechanism onto q4_K, whose staging was already cheap, makes the
gain term vanish outright.

### 3.3 Pitfalls

Three bugs the gates stopped: the transplant itself was well built — all three bugs were caught before landing:

1. **Word/byte pointer confusion (staging side)**: the sda plane's stepping
   used a `uint32_t*` as if it were a byte pointer, jumping 4 B per step —
   the sda plane was trampled at 4× stride. The parity dump exposed it.
2. **The same confusion (compute side)**: the consumer side committed it
   again — the r44-class stride mismatch of "bytes correct, offsets
   misaligned". This time the **greedy-32 byte stream** called it first
   (output diverges from some token onward).
3. **Wrong B-window rotation period**: the B window is a whole super-block
   (144 B, covering 8 chunks), while at KDR=2 a super-block completes only
   every 4 kt — the window must rotate on **super-block parity (every 4
   kt)**, not on kt. Rotating on kt means buffer 1 is never written and the
   compute side reads the previous round's stale data. Most insidious: **the
   nt=13 dump looked clean** — pure stale-node luck (the reused smem happened
   to still hold the correct data). Only after fixing the rotation period did
   everything truly go green.

Pitfall 3 is the generic disease of double-buffer changes: **the rotation
period must follow "how many kt one copy of data covers", not kt itself**.
When KDR changes, every kt-periodic implicit assumption must be re-audited.

## 4. Verification

- **greedy-32 byte stream**: defends against "wrong math but small deviation"
  — it was the first to catch both pointer-confusion bugs; after the fixes,
  IDENTICAL.
- **parity dump ×3** (against the CPU reference): defends against
  "systematically wrong values that greedy happens not to trip on" — it
  exposed bug 1; after the fix, ×3 all green.
- **nt=13 small-shape dump**: specifically defends against tail/small-shape
  boundary errors — with the stale-node analysis it flushed out bug 3.
- **Same-window paired A/B**: the baseline was re-measured first at 3219.6
  (consistent with the landed record), ensuring the −12.6% reading was clean
  (no r59b-style baseline contamination — re-measured on the spot).

After all functional gates went green the performance was still −12.6%,
which is what made the revert legitimate: not "done wrong", but "even done
right it should not be done".

## 5. Results

- Transplant vs baseline: **2819.3 vs 3227.6 tok/s = −12.6%** (same-window
  paired; baseline window median 3219.6).
- Functional state: greedy-32 IDENTICAL, parity ×3 green, 0 spill, LDGSTS
  present in the SASS — the mechanism transplant itself fully succeeded.
- **Veto mechanism**: the second term of the value formula (4× barrier
  density + 1-deep lookahead that cannot cover a 600–900 cyc latency + zero
  ALU savings) is strictly negative, and the first term ≈ 0 (q4_K staging
  was already a pure copy). After the revert the baseline returned to 3219.6.
- **Retry conditions**: double buffering/cp.async re-enters the candidate
  list only if q4_K bt's staging becomes expensive again (e.g. new
  staging-phase ALU is introduced, or the A-side layout transform is forced
  back into the kernel) — r59's W_dsc plane went the opposite direction
  (eliminating the remaining staging ALU rather than hiding it).

The Phase-2 spec was produced ranked by attribution (handed to r59):
(1) **q4_K W_dsc plane** (r56's scaffolding moved onto the other 63% of
busy); (2) wave re-tiling (+0.3–0.8%, byte-identical); (3) fused ffn_gu
concat; (4) riders (host-side small items like prewarm/pre-grow).

## 6. Lessons

1. **The cost of a mechanism depends on the granularity of the thing it
   replaces — such a mechanism is not free; it is AMORTIZATION-BOUND**. The
   mirror of r45: before transplanting, first compute how much "exposed
   latency to hide" remains on the host.
2. **A double buffer's value = whether lookahead depth × tile copy time can
   cover the memory round-trip latency**; a 1-deep lookahead is worth zero
   against a ~600–900 cyc latency, while the barrier density is a real 4×.
3. **The rotation period of a staging window/buffer follows the data's
   coverage span (super-block), not kt**; changing KDR requires re-auditing
   every kt-periodic assumption.
4. **"The dump looks clean" is not the same as "no stale data was read"** —
   stale-node luck can let a wrong configuration pass a small-shape dump;
   attack rotation/reuse bugs with shapes spanning multiple rotations.

---
← 60 · [Index](./README.md) · 62 →
