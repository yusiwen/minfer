# 0061. The Q6_K dot accumulates in eight lanes, and the scalar kernel implements that order

- Status: Accepted
- Date: 2026-10-29
- Issues: #693, #691

## Context

ADR-0010 set the identity bar: a kernel change is A/B'd bitwise against the existing path, and the
x86 K-quant dots are bitwise-identical to a scalar reference ([#56](https://github.com/yusiwen/minfer/issues/56)).
The gate compares each SIMD kernel against the scalar, so the *order* of the float arithmetic is
whatever the scalar kernel implements. The reference is the scalar, not "the natural per-element
loop".

For Q4_K and Q5_K that order costs almost nothing, because their per-subblock scales are folded in
`i32` and only the superblock-level folds remain — `sumf -= dmin·mterm; sumf += d·total`, two
dependent `f32` operations per superblock. Q6_K is different: its scale is per group, sixteen
groups per superblock, so the scalar folds `sumf += d · scale_g · sum_g` sixteen times. That is a
sixteen-deep chain of 4-cycle adds per superblock — about 64 cycles, ~20 ns at 3.2 GHz — and every
SIMD kernel had to reproduce it exactly.

[#689](https://github.com/yusiwen/minfer/issues/689) measured that chain as the floor: the landed
AVX2 kernel ran 312.8 ns for a 4096-element row (19.7 ns/superblock, one row = sixteen
superblocks) and the restructured AVX-512 arm could not reach the 15% bar because the chain, not
the integer work, was the limit. [#693](https://github.com/yusiwen/minfer/issues/693)'s step-1
probe kept the same integer work and replaced the sixteen folds with the reference
implementation's shape — eight `i32` lanes, a per-lane `f32` accumulator across the row, the −32
offset corrected once per superblock from the q8_K bsums — and measured 167.9 ns, **1.86×** the
landed kernel on the same box, same buffers, interleaved rounds.

The constraint was never "the scalar's per-group loop is the only legitimate order". The gate's
content is agreement: every kernel reproduces the reference bitwise. Redefining the reference to
the lane order keeps that property and removes the chain.

## Decision

For the Q6_K × Q8_K dot the reference order is defined as follows. One superblock is 256 elements
in sixteen groups of sixteen (`g`), with per-group `i8` scales `s[g]` and per-group activation sums
`b[g]` (the q8_K bsums). `q6[...]` below is the element's 0..63 code; the −32 offset is step 2.

1. Eight `i32` lanes `L[0..8]`, zero at the start of the superblock. For chunk `c = 0..7` (groups
   `2c` and `2c+1`, 32 consecutive elements) and sub-position `j = 0..3`:

   ```
   L[j]   += s[2c]   · Σ_{e<4} q6[32c + 4j + e]      · q8[32c + 4j + e]
   L[4+j] += s[2c+1] · Σ_{e<4} q6[32c + 16 + 4j + e] · q8[32c + 16 + 4j + e]
   ```

2. Correction, once per lane before the float step:

   ```
   L[k] -= 32 · (s[2k]·b[2k] + s[2k+1]·b[2k+1])
   ```

3. With `d = f16(q6.d) · f16(q8.d)` and one `f32` accumulator per lane, held across the whole row:

   ```
   acc[k] += d * (L[k] as f32)        // a multiply and an add, not an FMA
   ```

4. The row result is the fixed tree

   ```
   ((acc[0]+acc[1]) + (acc[2]+acc[3])) + ((acc[4]+acc[5]) + (acc[6]+acc[7]))
   ```

The integer steps are exact, so a kernel may accumulate them in any order; steps 3 and 4 fix the
float order. `dot_q6_k_q8_k_scalar` implements exactly this, and the AVX2 and AVX-512 kernels
reproduce it bitwise. NEON keeps its named tolerance — it was never in the x86 bitwise set. Q4_K
and Q5_K are unchanged: their order is already two folds per superblock, and 2 is not a chain
worth moving.

## Alternatives considered

- **Keep the per-group order.** Rejected by measurement: it costs 1.86× of Q6_K dot time, and the
  chain is the reason the AVX-512 kernel could not be brought above the AVX2 one.
- **Keep the integer lanes but compare with a tolerance instead of bitwise.** Rejected: it would
  delete the exact gate that makes an ISA-level change safe, for a difference that is entirely
  under our control.
- **Accumulate in f64 (or Kahan) so the order stops mattering.** Rejected: slower on the same
  kernels, and it still would not be bitwise against the scalar the gate compares with.

## Consequences

- The model's logits change in the last bits. Greedy-text gates are expected to be unchanged; any
  pinned numeric vector is re-derived in the landing PR. Suite counts do not change.
- The x86 bitwise property is preserved: the scalar is now the lane order, and the three x86 gates
  (AVX2, AVX-512, dispatch) keep asserting SIMD == scalar.
- The chain that made Q6_K's AVX-512 kernel lose to AVX2 ([#691](https://github.com/yusiwen/minfer/issues/691))
  is gone, so the dispatch question is re-measured on the landing PR instead of guessed.
- A later Q6_K change that alters this order is a new ADR, not a refactor.
