# 20 · r15 — f32-accumulate mma probe (dead end) + rank-1 term2 rescale (LANDED)

> **Result**: wide-kernel MMQ KD=8 **1271 → 1295 tok/s (+1.9%, 3/3 consistent)**; ncu per-GMAC warp instructions **9.45 → 8.69 M (−8.1%)**, q-proj kernel 2.378 → 2.262 ms (−4.9%). At the same time a seemingly tempting ISA route was closed: **integer mma has no f32-accumulator form** (proven by a ptxas probe).
> **Commit**: `b999e9a`. **Date**: 2026-09-03.

## 1. Background — where things stood

On 2026-09-03 the q4_K MMQ campaign (Era C) reached round 15. The campaign target had not changed since r6: lift the MMQ GEMM from 6.1 TMAC/s to ≥24 (f16-path equivalent) or ~30 (llama.cpp equivalent), turning the whole quantized prefill path around. The previous rounds' chain of results:

- **r12** (16-chain warp tile + ldmatrix A fragments): punctured the ILP-depth wall, wide kernel KD=4 reaching 1020–1058 tok/s, ~2.3× faster than narrow (441–481) — the largest single-step structural gain to that point.
- **x-tile / j-tile / cp.async-db** (the staging-shape family): all flat or reverted; the verdict was that the staging-order axis is closed and "the only remaining levers are per-MAC instruction efficiency".
- **r13** (ncu counter forensics): the first hard evidence — **per-GMAC warp instruction count is this kernel class's first-order predictor**. minfer 10.14 M/GMAC vs llama.cpp's 6.06 M (1.67×), with duration tracking instruction count linearly at ~0.10–0.15 warp-inst/ns across all data points of both engines. It also eliminated three "suspects": L2 bytes, store efficiency, bank conflicts — all three fixed simultaneously, the wall unmoved.
- **r14** (B fragments on a single ldmatrix.x4 + widened scale reads): down to **9.45 M/GMAC**, wide kernel KD=4 1225 / KD=8 1273 tok/s, kernel 3.632 → 2.378 ms. A big win for the "fewer/wider smem ops" lever class.

The epilogue itself has a genealogy worth telling: the two-term rescale's math passed unchanged from the R1 word-level kernel (`mmq_nt_kernel`, 2026-08-31) all the way into the raw kernel — the comment above the wide kernel's epilogue at r15 time reads verbatim "rescale: identical math/layout to the R1 kernel". That is, **this epilogue is the oldest living code in the campaign**: in the R1 era it was the price of correctness; r12/r14's structural rewrites bypassed it twice (touching the tile, touching fragment loads) but nobody touched its instruction composition. r13's ledger pushed it onto the stage: llama's 6.06 M/GMAC contains no such mass of per-value floating-point operations — llama uses float accumulators (floating-point mma) and never needs an I2F boundary at all, while we must pay the translation tax for an s32 accumulator.

At r15's opening the instruction-stream ledger read: llama 6.06, us 9.45, a 3.39 M/GMAC difference. r14's record named the next lever: **`mma...f32.s8.s8.f32`** — an f32-accumulating integer mma. The idea's appeal is the entire reason the rescale epilogue exists: the integer mma's accumulator is **s32**, so every C value must pass through `I2F` (int→float conversion) before multiplying two scale terms. If the accumulator were f32 to begin with, that int→float boundary disappears entirely and the epilogue's instruction stream could collapse by a large block.

r15's first move was therefore an **ISA probe**: does this instruction even exist? The answer is no (see §2), and one ptxas call falsified it — part of this step's methodological value: **before designing around an instruction, spend a few minutes verifying the instruction exists**.

With the probe dead, the same math spawned two alternatives: merging the integer dot products of 2 chunks before one I2F (vetoed algebraically, see §2.3), and the ultimately landed **rank-1 fold** — it does not reduce I2F, but cuts the floating-point multiplication stream in term2 by 3/4.

## 2. Principle — the GPU mechanism

### 2.1 Where the two-term rescale comes from

MMQ's (int8 tensor-core quantized GEMM) numeric structure: A (activations) quantized to q8, each 32-element block carrying scale `da` and the **in-block integer sum** `sa` (the q8_1 format's d/s fields, computed for free by the quantize prepass); B (weights, q4_K) with each 32-sub-block having effective scale `dsv = d·sc` and negative bias term `dmv = −dmin·m` (expanded into float2 at staging).

The s8 mma (`mma.m16n8k32`, B side fed unsigned nibble values v ∈ [0,15] directly — exactly representable in s8) computes the integer dot product:

```
S_ij = Σ_k a_int[i,k] · v[j,k]
```

while the true product is `b = d·sc·v − dmin·m`, so every C value needs two correction terms (the "two-term rescale"):

```
true(i,j) = Σ_chunks [ da_i · dsv_j · S_ij   ← term1: the main term
                     + dmv_j · (da_i · sa_i) ] ← term2: the correction term
```

term1 multiplies the mma's integer result (I2F mandatory); term2 multiplies the A block sums. The key observation is term2's **structure**:

```
term2(i,j) = dmv_j · (da_i · sa_i) = (row vector da·sa) ⊗ (column vector dmv)
```

a **rank-1 matrix** (outer product) on the (token, od-col) plane. The row-side factor `da_i·sa_i` is identical for all columns of a row; the col-side factor `dmv_j` is identical for all rows of a column.

By contrast term1 is **not** rank-1: `S_ij` in `da_i · dsv_j · S_ij` is the mma's per-(i,j) output — a full-rank matrix. term2 is foldable precisely because it contains no mma result, being only the product of two per-row/per-column coefficient sets. "Ask the correction term's rank first, then decide how to evaluate it" is this step's reusable criterion.

Where the two coefficient sets originate in the data path (both computed once at quantization, expanded at staging; the GEMM hot loop only multiplies):

- **col-side dsv/dmv**: expanded per sub-block into float2 at B staging (around `src/cuda_kernels.cu:6024` in the current tree):

```cuda
float d = h2f(*(const uint16_t*)blk);        // super-block scale d (f16)
float dmin = h2f(*(const uint16_t*)(blk + 2)); // super-block dmin (f16)
// …sc/m are the sub-block's 6-bit indices (get_scale_min_k4 table semantics):
dv = d * (float)sc;                          // effective scale  → dsv
mv = -(dmin * (float)m);                     // negative bias    → dmv
sds[(size_t)kd * MMQ_WBJ + r] = make_float2(dv, mv);
```

- **row-side da/sa**: the A quantize prepass computes the in-block integer sum for free while writing each q8 block (around `src/cuda_kernels.cu:5301`: `da[r] = d; sa[r] = s;`) — the q8_1 format's s field. The GEMM kernel just reads it from staging, with no extra pass.

### 2.2 Where the old code wasted

Before the fold, term2 was evaluated in full **for every C value**:

```cuda
sum[idx] += da * dmv[nh][l & 1] * sa;   // old: 2 FMUL per C value
```

The wide kernel has 64 C values per thread per chunk (8 A-frags × 2 B-frags × 4 C regs), but the row side's `da·sa` takes only 2 distinct values (the token pair each A-frag covers). That is, **48 of the 64 FMULs repeat the same product** — the commit message's count "64 FMUL(da·dmv)/chunk → 16 FMUL/chunk" is exactly this ledger: fold the row-side product to once per row (8 g × 2 rows = 16), leaving each C value a single `dma·dmv`, which the compiler issues as an **FFMA** (fused multiply-add into the sum accumulator).

Per-chunk, per-thread op ledger for the term2 part (term1 and the I2F are unchanged, hence excluded from the delta):

| | Old (r14 shape) | New (r15 fold) | Δ |
|---|---|---|---|
| FMUL (row side da·sa or dma) | 64 (one `da·dmv` per C value, then × `sa`) | 16 (one `dma = da·sa` per row) | −48 |
| FMUL/FMA (col side × dmv into sum) | 64 FMUL + 64 FADD | 64 FFMA | fused, no standalone FADD |
| I2F (sa to float) | repeated per value (CSE-able) | exactly 16 (inside dma) | structured |

The three components of `sum[idx] += dma[l >> 1] * dmv[nh][l & 1]` — reading dma, reading dmv, FFMA into sum — were all going to happen in the epilogue anyway; the fold saves only those 48 redundant FMULs.

### 2.3 The two rejected alternatives (falsified on paper, before implementation)

**Route A: f32-accumulating integer mma.** If `mma...f32.s8.s8.f32` existed, the I2F boundary vanishes. Probe result: **ptxas (CUDA 13.0, PTX 8.8/9.0) reports "Unexpected instruction types" for every target sm_80 through sm_121**; the control group's s32-accumulator form (`.s32.s8.s8.s32`) assembles clean across the board. The conclusion is not a syntax problem but an ISA fact: **integer mma comes only with an s32 accumulator**.

Could one fall back to a floating-point mma with f16/tf32/fp8 operands and f32 accumulation to emulate it? No — for the product to pass the parity gate (1e-3), the operands must carry the `q8·scale` product **bit-exactly**. int8's 8-bit magnitude times an f16 scale's 11 significant bits needs ~18 significant bits in the product; f16 has only 11, tf32 only 11, fp8 only 3–4. Every operand format loses precision before the multiply, and the parity gate fails.

**Route B: 2-chunk integer merge.** Add the integer dot products of two adjacent 32-k chunks in s32 first, then do **one** I2F, hoping to halve the I2F count. Overflow-wise fully feasible (per chunk |S| ≤ 32·127·15 ≈ 2^16, two merged chunks are far from 2^31 — the meaning of the record's "regardless of the 2^21 bound": range was never the obstacle). **But algebraically void**: the rescale coefficients differ per chunk (`get_scale_min_k4(c&7)` yields per-sub-block d/dmin, and the A side's d/ssum also varies per block); the merged S = S₀+S₁ can only be multiplied by **one pair** of coefficients, and the information needed to split back into two chunks is already lost. The commit message's conclusion: under any pairing, the error is **O(1e2)** — 5 orders of magnitude worse than the 1e-3 parity gate.

**Why llama.cpp doesn't pay this tax**: the key reading from r9's reference decode — llama's MMQ uses **float sum accumulators** (16 mma chains, each warp accumulating floating-point sums directly per chunk). It performs the same two-term rescale but has no I2F boundary: the mma emits floats and the rescale is a pure FFMA chain. Our instruction-count chase keeps running into this structural difference: **the accumulator bit-width/bandwidth the integer mma saves must be bought back in the epilogue with I2F+rescale**. r15's probe formally archived "can this tax be waived" as: no.

### 2.4 Why the fold is numerically safe

The new code's only numeric change is **one multiplication's association order**:

```
old: (da · dmv) · sa     new: (da · sa) · dmv
```

Three reals, two floating-point multiplies, only the parentheses exchanged — the difference is at ~ulp(|sum|) scale (the commit message's verbatim "numerics within ~ulp(|sum|)"), two orders of magnitude of margin against the 1e-3 gate. Note this is not "harmless enough to skip verification": floating-point multiply is commutative but not associative, and such changes still pass every gate (see §4) — it is just that the error class here is foreseeable and explainable.

### 2.5 Why this magnitude clears the +1.5% bar

r13's 1:1 tracking law (duration ≈ instruction count / 0.10–0.15 warp-inst/ns) predicts: an instruction stream −8.1% should buy back kernel time of the same order. The kernel measured −4.9% — **sub-linear**, because SpeedOfLight Compute (SM) is only 31.5%: the kernel is stall-bound, issue is not the bottleneck, and only part of the removed instructions actually shortened the critical path. This is a boundary fact delivered alongside the step: **in this regime, ALU-class cuts' payout decays from 1:1** (r17 would push this rule to its extreme).

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Probe first**. Verifying the f32-accumulate mma costs one raw-PTX file + one ptxas call; writing the kernel first and discovering the instruction doesn't exist wastes a full implement+verify round. In hindsight the commit message's structure — probe verdict in the first paragraph, the landing after — is the record form of "falsify first, design second".
- **The veto happened on paper**. The 2-chunk merge needed no code to reject: per-chunk-varying coefficients is a fact visible to the eye in the source (`get_scale_min_k4(c&7)` called per sub-block inside the unrolled loop), and one algebraic argument closes it.
- **The landed choice is "the cheapest cut within the same instruction-stream family"**. The original target (eliminating the I2F boundary) died at the ISA level, but the motive behind it — shrinking the epilogue's floating-point op stream — still stood. The rank-1 fold touches no I2F, no term1, no layout; it only re-orders term2's multiplication tree — the smallest-surface, cleanest-numeric cut available.
- **Wide kernel only**. The wide kernel (`mmq_raw_wide_nt_kernel`) is the performance path (1225–1273 vs narrow's 441–481); narrow stays as the control, ported one round later (r16, next doc).

### 3.2 Key code

The change is 10 lines of the wide kernel's epilogue in `src/cuda_kernels.cu` (the `b999e9a` diff, +8/−2). Before/after:

```cuda
// ── BEFORE (r14 state): term2 evaluated in full for every C value ──────────────
#pragma unroll
for (int nh = 0; nh < 2; nh++)
    #pragma unroll
    for (int l = 0; l < 4; l++) {
        const float da = da_q[l >> 1];
        const float sa = (float)sa_q[l >> 1];        // ← repeated per value (CSE-able)
        const int idx = (g * 2 + nh) * 4 + l;
        sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];  // term1
        sum[idx] += da * dmv[nh][l & 1] * sa;        // ← the source of 64 FMUL/chunk
    }

// ── AFTER (r15): the row-side product folded to once per row ─────────────────
// r15: the dmv correction term is rank-1 in (token, od-col) —
// the row-side product da*sa is shared by the od-col pair of
// each C fragment, so fold it once per row (16 FMUL/chunk)
// instead of once per C value (64 FMUL/chunk). The dsv term
// and the per-chunk scale application are unchanged.
const float dma[2] = { da_q[0] * (float)sa_q[0],
                       da_q[1] * (float)sa_q[1] };   // ← once per g (16/chunk)
#pragma unroll
for (int nh = 0; nh < 2; nh++)
    #pragma unroll
    for (int l = 0; l < 4; l++) {
        const float da = da_q[l >> 1];
        const int idx = (g * 2 + nh) * 4 + l;
        sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];  // term1 unchanged
        sum[idx] += dma[l >> 1] * dmv[nh][l & 1];    // ← one FFMA per C value
    }
```

The two subscripts deserve a read: `dma[l >> 1]` indexes by **row** (l=0,1 share row 0; l=2,3 share row 1 — in m16n8k32's C-fragment layout, a thread's 2×2 accumulators share rows within a column pair), and `dmv[nh][l & 1]` indexes by **column pair**. The fold's entire gain comes from "sharing within a row", so the row/col indexing must not be swapped (see §3.3).

Its location in the current tree: `src/cuda_kernels.cu:6136` (the wide kernel `mmq_raw_wide_nt_kernel`'s per-chunk epilogue, with da/sa read from the uint2-packed staging in one LDS.64):

```cuda
// token pair (t, t+8) in one LDS.64 (uint2 tiling)
const uint2 pk2 = *(const uint2*)(sda_q
    + (size_t)kd * MMQ_WBI * 2 + g * 16 + (lane >> 2) * 2);
da_q[0] = h2f((unsigned short)(pk2.x & 0xFFFF));
sa_q[0] = (int)(short)(pk2.x >> 16);
da_q[1] = h2f((unsigned short)(pk2.y & 0xFFFF));
sa_q[1] = (int)(short)(pk2.y >> 16);
const float dma[2] = { da_q[0] * (float)sa_q[0],
                       da_q[1] * (float)sa_q[1] };
```

`sa` itself comes from the A quantize prepass's free computation (the q8_1 block's s field, around `src/cuda_kernels.cu:5301`'s `da[r] = d; sa[r] = s;`) — term2's row-side data was in place at quantization time; the GEMM side merely consumes it.

### 3.3 Pitfalls

- **Row/col index confusion is the only high-risk point**. `dma` by `l >> 1`, `dmv` by `l & 1` — written swapped, it compiles clean, term1 stays correct, and only term2 is wrong; the output deviation drowns in the scale magnitude but is findable, at the cost of localization time. m16n8k32's C-fragment layout (4 C values per thread = 2 rows × 2 columns) is this fold's spatial precondition — draw the layout clearly before touching the epilogue.
- **Don't bet on the compiler's CSE**. The old code wrote `(float)sa_q[l >> 1]` inside the value loop; ptxas could theoretically hoist it. r15's approach makes the sharing explicit — the `dma` array is constructed outside the loop. Lesson: redundancy left for the compiler to eliminate is "maybe saved"; redundancy eliminated structurally is "definitely saved".
- **Probes should use raw PTX, not inline asm**. Inline asm's error paths tangle in constraint parsing; feeding a raw `.ptx` file straight to ptxas yields a pure instruction-type verdict ("Unexpected instruction types"), and swapping the accumulator type in the same file runs the control group — the difference between two calls is the conclusion.

## 4. Verification

- **Parity gate (KD=4 + KD=8)**: against the CPU reference implementation at 1e-3 tolerance — defends against term2 indexing errors and re-association drift beyond expectation; the ulp-scale error class argued in §2.4 is backed by this gate.
- **Suite 166/0/3**: the full test suite (166 pass / 0 fail / 3 skipped) — defends against collateral damage beyond the GEMM.
- **greedy-32 token identity**: 32 greedy-decoded tokens compared token-for-token against the default path — defends against "numbers pass but semantics drift".
- **ncu counter A/B** (q-proj GEMM: inst/GMAC + duration + SM%) — defends against "the wall clock moved but nobody knows why": this step captured −8.1% inst and −4.9% duration together, and the ratio is itself evidence of the stall-bound regime.
- **Interleaved 3× same-window A/B** (KD=8 3/3 consistent) — defends against machine drift polluting the delta (r59b's lesson, by now internalized as procedure).

## 5. Results

| Metric | before (r14) | after (r15) | Δ |
|---|---|---|---|
| Wide kernel KD=8 whole-prefill | 1271 tok/s | **1295 tok/s** | +1.9% (3/3 consistent) |
| Wide kernel KD=4 | ~1226 tok/s | ~1233 tok/s | +0.5% (noise band) |
| Narrow kernel control | — | stable | — |
| ncu warp-inst / GMAC (q-proj) | 9.45 M | **8.69 M** | −8.1% |
| Kernel duration | 2.378 ms | **2.262 ms** | −4.9% |
| SpeedOfLight Compute (SM) | — | 31.5% | stall-bound evidence |

The instruction stream's campaign trajectory: r13 baseline 10.14 M/GMAC → r14 9.45 → r15 **8.69** (llama.cpp 6.06; the control is llama-bench @ `ca3d5a3e1`'s same-shape q-proj kernel). This round also drew a floor: **the rescale epilogue is down to ~346 ops/chunk, and the I2F stream is ISA-irreducible** — as long as the s32 accumulator is the only form, one I2F per C value is the fixed tax of integer GEMM.

Reading these numbers in the campaign's coordinates at the time: wide KD=8's 1295 tok/s was still ~1.8× from the same-session f16 default path (2284–2370), and the per-GMAC duration gap measured at r14 was 70.4 vs llama's 41.1 µs/GMAC; r15 cut 0.76 M out of the 3.39 M instruction gap, picking the per-MAC instruction-efficiency main line's low-hanging fruit clean. The next lever after r15's ALU cuts was named as warp-tile shape (2× od-rows per warp) or prefetch/stall structural work — the former is r17's story (reverted), the latter cashed in at r20 (+7.1%).

## 6. Lessons

1. **Build the ISA boundary with a raw-PTX probe, then design around it** — an instruction that doesn't exist (or that ptxas rejects) should never get the chance to grow into a kernel.
2. **Look for rank-1 (outer-product) structure in rescale/correction terms** — a per-value-evaluated correction term that separates by row/column pays FMULs per row instead of per value.
3. **Floating-point re-association is an ulp-scale numeric change**: it can land, but it needs an explicit argument + the full parity gate; "just moving parentheses" is not a reason to skip.
4. **When SM% is low, ALU cuts pay out sub-linearly** (−8.1% inst → −4.9% duration): the instruction stream is still the correct lever, but diminishing returns have begun — this signal foreshadows r17's lesson.

---

← 19 · [Index](./README.md) · 21 →
