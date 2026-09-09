# 16 · r12 — 16-chain warp tile + ldmatrix (LANDED)

> **Result**: after the q4_K MMQ wide kernel's warp-tile rearrangement, 7B @2K same-window interleaved measurement gives **wide-16 KD=4 1020–1058 vs narrow 441–481 tok/s ≈ 2.3×** (KD=8 973–995; vs the pre-rewrite wide ~719 = 1.44×) — the "accumulator-depth wall" r10 located is flattened in one stroke, the largest structural landing of P6 before r34.
> **Commit**: `774a116`. **Date**: 2026-09-02.

## 1. Background — where things stood

The r9–r11 elimination ladder had narrowed the q4_K MMQ suspects to instruction-level structure. r9 measured the entire shape matrix: narrow cp.async KD=8 **481** was the local optimum, all six tile/staging shapes parity-clean — the shape axis closed — and read out llama.cpp's instruction model, **~0.018 inst/MAC/thread vs our 0.133**. r10 ported llama's inner-loop math decomposition verbatim: 462–468 vs narrow's 470, motionless — the math decomposition was eliminated and the residual locked to three items: **ILP chain depth** (llama 16 independent mma chains + 128 accumulator registers; ours 8 chains / ~64 registers), **ldmatrix A staging** (1 LDSM vs 8 per-lane LDS.32 per fragment), and **tile 128×128**. r11 verified against llama.cpp's `mma.cuh` that `ne = I·J/32` (NVIDIA Turing+ branch; `I·J/64` is the AMD MFMA branch), the "16 chains" reading stood, and r12's plan had its prerequisite proof.

At this point MMQ was stuck at 441–481 tok/s (opt-in `MINFER_MMQ=1`) against the same-window f16 default path's 2284 — a 5× gap. r6's target (≥24 TMAC/s for f16 parity, ~30 for llama parity) meant prefill could go from ~2340 to ~2670/3250. MMQ was the only part of the prefill wall still beyond 5×.

The direct host of the rewrite was r8's wide kernel `mmq_raw_wide_nt_kernel`: 128-token block, 8 warps × 32×32 per-warp tile, best cross-machine-state result ~719 tok/s (the source of the 1.44× comparison in r12's results). Its margin over narrow (481) came from halving B traffic — but r8 had already established "L2 absorbs the B re-reads", so the wide tile's gains stopped there; **the shared bottleneck is the per-chunk inner-loop overhead** (r8's words). r12 cut on that bottleneck instead of moving the tile again.

One process note worth recording: r10's code died to the post-checkout hook, and in r12's session the hook was absent (the nix shell had no rusty-hook binary); commits explicitly bypassed it plus a make-up fmt-equivalence check. The 16-chain plan could not afford to be lost twice.

r12 executed r10's redo recipe but **touched only the first two items**: double the chain depth + ldmatrix for the A fragments; the B-side staging format and the two-term rescale were not changed by a single character. The cut was deliberately narrow — which is exactly why the 2.3× number could later be attributed cleanly.

## 2. Principle — the GPU mechanism

**The warp tile's geometry.** The new block tile is 128 tokens × 128 od with 8 warps. Each warp claims **one private 16-od-row slice × the entire 128-token tile**:

- A fragments (token axis): 128 tok ÷ 16 tok per m16n8k32 fragment = **8 A fragments**;
- B fragments (od axis): 16 od ÷ 8 rows per fragment = **2 B fragments**;
- per chunk (32-k): 8 × 2 = **16 mma**.

The key is "who shares whom": A fragments are reused by all 16 chains in the warp (the token axis is the warp's common axis), while B fragments are warp-private (the od axis is the warp's private axis). Against the old shape — 2 A fragments × 4 B fragments = 8 chains per warp (`clow[4][2][4]`) — the new shape moves the entire token axis into a single warp, doubling the chain count.

**The accumulator register ledger.** 16 chains cost registers: `clow[8][2][4]` = 64 int C-fragment registers + `sum[64]` = 64 float accumulators, **128 accumulator-class registers live simultaneously** — exactly the llama depth r10/r11 recorded. This budget is only payable at 1 block/SM: 256 threads/block against a 64K register/SM file ≈ a hard cap of ~256 per thread, and 128 accumulators plus operands and indices barely fit. That is why llama's config says "target occupancy 1" — not a flaw, a budget choice: **when no second block exists to fill the mma latency holes, chain depth IS the issue window**. r5–r6 had already measured that occupancy 1→2 does not rescue this kernel (depth inverted against occupancy); r12 maxed out the depth instead.

**ldmatrix.** `ldmatrix.sync.aligned.m8n8.x4.shared.b16` is one instruction fetching four 8×8 b16 matrices (512 B) from shared memory, distributed to the 32 lanes' 4 registers in the standard mma A-operand layout — exactly one m16n8k32 A fragment's worth. The per-lane alternative has each lane compute 4 addresses and issue 4 narrow loads (r10's accounting: 1 LDSM vs 8 LDS.32 per fragment). What is saved is more than instruction count: the address ALU disappears too, and the load granularity goes from 4 B to 16 B/lane.

**KDR=4 restage-skip.** B's raw bytes are organized in 256-k super-blocks; at KDR=4 two adjacent k-tiles fall inside one super-block, so the expanded `qb8` can survive across the k-tile barrier — restaging happens only when `((kt·KDR) & 7) == 0`, halving B-side movement. Free bandwidth, but not the main line: the main line is chain depth.

**Why KD=4 overtakes KD=8.** Intuition says doubling kd depth should amortize per-MAC staging overhead, but the smem ledger says otherwise: at the 128×128 tile, KD=8's dynamic smem is markedly thicker, and r8's phantom lesson (wide KD=8 once hit the ~99 KB cap) means every depth point must pass the launcher's explicit cap check. KD=4 with restage-skip already halves B movement, and depth's small amortization loses to the tighter smem budget and shallower residency — measured KD=4 (1020–1058) > KD=8 (973–995). This is also restage-skip's value: **it lets the shallow depth stop paying full price for B movement for the first time**.

**The issue-window arithmetic.** 1 block/SM, 8 warps/SM (4 schedulers × 2 warps each): the schedulers spend most cycles waiting on mma results. The old kernel issues 8 mma per thread per chunk; the new one 16 — each scheduler pass can accumulate twice the pending mma. No second block's warp exists to fill the holes (r5–r6 measured occupancy 1→2 not helping), so **chain depth is the only source of issue window**. This is the concrete form of "depth inverted against occupancy" in this kernel: r5–r6 stopped at KD depth; r12 pushed the same principle onto the accumulator chains.

**A coverage reconciliation.** Verifying the tile is self-consistent from another angle: per chunk per warp MAC = 16 mma × 16×8×32 = 65,536, and the warp's private slice is 16 od × 128 tok × 32 k = 65,536 — exact. Block level: 8 warps × 16 od = 128 od ✓, every warp covers all 128 tokens ✓. This reconciliation is not decoration: 16-chain rearrangements most easily produce coverage holes where some output rows are computed twice and others missed (the r5–r6 4-warp 32×32 hand mapping leaked exactly this way — zero cells, parity blew up immediately). r12 passing parity on the first try says the (A fragment g, B fragment nh) → (token, od) mapping is a complete bijection.

**The cost accounting for leaving B alone.** r12 deliberately kept B fragments on "per-lane narrow reads + unpack nibbles at mma time": 2 B fragments per chunk × 2 32-bit reads each + unpack shift/mask. That overhead is not small, but it is a **fixed quantity per chunk**, not multiplied by the chain count — whereas the A side's chain structure **multiplies onto every mma**. Double the chains first, clean up B later; in the reverse order (both sides at once) the 2.3× could not be attributed. r14 validated this ordering two weeks later: switching B fragments to ldmatrix took another +18.5%, showing B did still have meat — but that was the next cut.

**Why ~2× was the expectation.** 1 block/SM, 8 warps/SM: old kernel 8 mma per thread per chunk with a shallow issue window; new kernel 16. Doubling chain depth lets each scheduler's mma issue port fit twice the work between dependency stalls. The measured 2.3× slightly exceeds expectation, indicating LDSM's load efficiency also contributed.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **The warp takes "16 od × all 128 tok" instead of a 2D sub-block**: putting the whole token axis in the warp is the source of chain depth (8 A fragments); cutting od into 8 slices is the source of the warp count. The reverse (4-warp 32×32 sub-blocks) cannot raise the chain count — r12 tried it on the spot, negative.
- **The B side untouched**: r10's lesson is one variable at a time. B's unpack still happens in registers at mma time (ldmatrix B fragments only at r14, pre-expansion tried at r18), so the 2.3× can only come from the A side's two changes.
- **KDR=4, not 8**: KD=8's wide shape has larger smem, closer to the 99 KB cap (r8's phantom lesson), while restage-skip stops KD=4's B movement from being penalized — measured KD=4 (1020–1058) > KD=8 (973–995).
- **Scales loaded per group for the accumulators**: `da_q/sa_q` changed to per-token-group reads to preserve accumulator registers (see excerpt B's comment) — 128 accumulators already fill the register file, so the rescale side's intermediates must yield.

### 3.2 Key code

**Excerpt A · the inner loop before/after in the r12 diff (`git show 774a116 -- src/cuda_kernels.cu`, segment by segment)** — the whole secret of 8-chains-to-16 is in these 20 lines:

```cuda
// ---- BEFORE (r8/r9 shape): 2 A frags x 4 B frags = 8 chains, A frags per-lane narrow loads ----
int a[2][4], b[4][2];
int clow[4][2][4], chigh[4][2][4];
#pragma unroll
for (int h = 0; h < 2; h++) {                    // 2 A fragments
    const int r0 = i0w + h * 16 + (lane >> 2);
    const uint8_t* p0 = qat + (size_t)r0 * 32;
    a[h][0] = *(const int*)(p0 + 4 * (lane & 3)); // each lane computes its own address:
    a[h][1] = *(const int*)(p1 + 4 * (lane & 3)); // 4 LDS.32 per fragment
    a[h][2] = *(const int*)(p0 + 4 * ((lane & 3) + 4));
    a[h][3] = *(const int*)(p1 + 4 * ((lane & 3) + 4));
}
...
for (int nh = 0; nh < 4; nh++)                    // 4 B frags x 2 A frags
    for (int h = 0; h < 2; h++)
        mmq_mma_k32(clow[nh][h], a[h], b[nh]);    // 8 chains

// ---- AFTER (r12): 8 A frags x 2 B frags = 16 chains, A frags via ldmatrix ----
// A fragments: 8 independent 16-token groups tile the full
// 128-token row, one ldmatrix.x4 per group (16 rows x 32B:
// lanes 0-7 -> rows 0-7 byte 0, 8-15 -> rows 8-15 byte 0,
// 16-23 -> rows 0-7 byte 16, 24-31 -> rows 8-15 byte 16 —
// the standard m16n8k32 A-fragment distribution).
int a[8][4], b[2][2];
int clow[8][2][4];
#pragma unroll
for (int g = 0; g < 8; g++) {                     // 8 A fragments, 1 LDSM each
    const uint8_t* p = qat
        + (size_t)(g * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * 32
        + ((lane >> 4) & 1) * 16;
    unsigned r0_, r1_, r2_, r3_;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
        : "r"((unsigned)__cvta_generic_to_shared(p)));
    a[g][0] = (int)r0_; a[g][1] = (int)r1_;
    a[g][2] = (int)r2_; a[g][3] = (int)r3_;
}
// B fragments: 2 minitiles of the warp's private 16 od-rows;
// unpack the raw nibbles in registers (staging format unchanged).
...
// 16 independent mma chains per thread per chunk, all C
// fragments live simultaneously (llama.cpp accumulator depth).
#pragma unroll
for (int g = 0; g < 8; g++)                       // 8 A fragments
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)                // x 2 B fragments
        mmq_mma_k32(clow[g][nh], a[g], b[nh]);    // 16 chains, C fragments live throughout
```

Three correspondences to note: `b[4][2] → b[2][2]` (B fragments 4→2, the warp owns 16 rows outright); `clow[4][2][4] → clow[8][2][4]` (accumulators organized by A fragment); and the comment's "llama.cpp accumulator depth" directly cites r10's finding.

**Excerpt A′ · the block tile itself widening (the r12 diff's staging-layout header)** — od 64 → 128 is the spatial precondition for doubling chain depth (only then can each warp hold the full token row × its private 16 rows):

```cuda
// BEFORE (r8/r9 wide): od tile 64
-    //   sds   [KDR][64] f32, sdm likewise
-    float* sdm = sds + KDR * 64;
// AFTER (r12): od tile 128
+    //   sds   [KDR][128] f32, sdm likewise
+    float* sdm = sds + KDR * MMQ_WBJ;        // MMQ_WBJ: 64 → 128
...
+    float sum[64] = {0.0f};   // [g][nh][l]: 8 A-frags x 2 B-frags x 4 C regs
```

`MMQ_WBI` (tokens, 128) unchanged, `MMQ_WBJ` (od) doubled: the smem scale plane doubles with it, buying each warp a private od slice instead of a shared one. The `sum[64]` comment is the ledger itself — 8 × 2 × 4.

**Excerpt B · the float accumulators and register yielding (r12 diff)** — `sum[64]` is the other half of chain depth; the rescale side switches to per-group loads:

```cuda
float sum[64] = {0.0f};   // [g][nh][l]: 8 A-frags x 2 B-frags x 4 C regs
...
// rescale: identical math/layout to the R1 kernel; A-side
// d/ssum come straight from the raw chunk. da/sa load per
// token-group to keep registers for the accumulators.
float dsv[2][8], dmv[2][8];   // BEFORE: float dsv[4][8], dmv[4][8]
```

The two-term rescale (`dsv` weight scale, `dmv` dmin term) keeps math identical to the r7–r8 kernel — the diff only shrinks the arrays' first dimension from 4 to 2 (B fragment count halved).

**Excerpt B′ · the accumulator fold epilogue (r12 diff)** — the line folding the int C fragments back into float `sum`; the index changes from (nh, h) to (g, nh), the multiplication structure untouched:

```cuda
// BEFORE:
sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
// AFTER:
sum[idx] += da * dsv[nh][jj] * (float)clow[g][nh][l];
```

`dsv[nh][jj]` is the warp-private 16 rows' weight scale (`jj` walks the od columns), `da` the A-side token scale — per chunk, 64 int C values fold into 64 float accumulators through the two scale terms. **This chain is not one of the 16 mma chains** (it reads already-finished C fragments), so doubling chain depth does not disturb it; its register footprint (`dsv/dmv`, 32 floats) does compete with the accumulators for budget — the origin of §3.1's "scales loaded per group".

**Excerpt C · KDR=4 restage-skip (introduced by r12; still at lines 5983–5986 of the current tree)** — the expanded B bytes survive across k-tiles:

```cuda
/* At KDR=4 two consecutive k-tiles share the super-block and the
 * expanded qb8 persists across the kt barrier - restage only when
 * this k-tile starts a new super-block. */
if (((kt) * KDR & 7) == 0) {
    // ... B staging: global read of qb8 raw bytes + register unpack written to smem
}
```

(At r12 the B fragments still unpacked nibbles at mma time; the "register unpack written to smem" above evolved at r14 into pre-expansion + the slot-major 48B layout — see below.)

**Excerpt D · this kernel's direct descendants (current tree, `mmq_raw_wide_nt_kernel`)** — r12's skeleton lives as-is, with B fragments also on ldmatrix (r14) and addresses switched to precomputed XOR-swizzle offsets (r22):

```cuda
// src/cuda_kernels.cu (current tree): r14's B-fragment ldmatrix + r22's G[g] swizzle
int a[8][4], b[2][2];
int clow[8][2][4];
#pragma unroll
for (int g = 0; g < 8; g++) {
    const uint8_t* p = qat + G[g];          // r22: XOR-swizzled granule offset
    unsigned r0_, r1_, r2_, r3_;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                 "{%0,%1,%2,%3}, [%4];\n" ...);
}
// B fragments: ONE ldmatrix.x4 serves both 8-od-row
// minitiles (matrices 0/1 = od-rows 0-7 at k-halves 0/1,
// matrices 2/3 = od-rows 8-15). reg_i of lane L = matrix_i row
// L/4, bytes (L%4)*4 — the exact mma.m16n8k32 B-operand
// distribution the plain LDS pattern produced.
{
    const uint8_t* rb8 = qb8 + (size_t)sg * (MMQ_WBJ * MMQ_WBQ)
        + (size_t)(j0w + (lane >> 4) * 8 + (lane & 7)) * MMQ_WBQ
        + (size_t)((lane >> 3) & 1) * 16;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 " ...);  // 1 LDSM
}
#pragma unroll
for (int g = 0; g < 8; g++)
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        mmq_mma_k32(clow[g][nh], a[g], b[nh]);   // 16 chains, unchanged to this day
```

r12 is not the endpoint but the skeleton: r14 (B-fragment ldmatrix, another +18.5%), r20 (split-phase A staging), and r22 (qa8 XOR swizzle) all grew on this 16-chain shape; the `mmq_raw_nb_bt` kernel promoted today (r28+) inherits the same "chain depth × register budget" ledger.

### 3.3 Pitfalls

- **The 4-warp 32×32 variant is negative**: chain depth does not double automatically with warp count — 4 warps × 32×32 sub-blocks still leaves 8 chains per warp, and it fragments the token axis so A fragments lose reuse. Rejected by measurement.
- **48B padding of A rows measured flat**: the A-side row padding for ldmatrix bank phase measured flat (48B on the B side is r14's business; not worth it on A at the time) — reverted; changes are not kept for "looking tidier".
- **Registers maxed out**: 128 accumulators + 32 A-fragment registers exhaust the budget, so `da_q/sa_q` must be re-read per group rather than resident throughout — the hidden tax paid for chain depth.
- **smem cap checked point by point**: doubling the od tile doubles the scale plane; KD=8 totals 98,304 B and KD=4 73,728 B — both inside the ~99 KB opt-in cap, with the launcher explicitly checking attr/launch results (r7's phantom lesson institutionalized: over-cap must be refused loudly, never silently fall back).
- **Hook bypass**: rusty-hook was missing from this session's nix shell and the post-checkout hook was bypassed (r10's code had just been swallowed by it); a make-up fmt-equivalence check + suite 169/0 were run at commit time. The process lesson is the mirror of r10's: a hook can swallow code (r10) or be absent itself (r12) — both states need a manual-verification backstop.

## 4. Verification

- **Parity gate (`cuda_prefill_mmq`)**: old and new kernels element-wise identical — defends against the 16-chain rearrangement changing summation order (each chain's k order is unchanged, so bitwise was the theory; measured green). The mechanism deserves spelling out: the 16 chains each accumulate their own (A fragment × B fragment) pairs, and **the summation order at block level is isomorphic to the old kernel's** — only issue timing changed, not the numeric path. So the parity expectation is identity, not tolerance; a 1e-3-scale drift would mean the rearrangement accidentally changed the reduction structure and must be investigated as a bug.
- **Suite 169/0**: whole-model regression — defends against the wide branch working only for q4_K and breaking other types.
- **Greedy token identity**: byte-compared token stream against the default path — defends against any end-to-end drift. This layer backstops the kernel parity: parity covers a single kernel, greedy covers the whole forward.
- **Same-window interleaved A/B**: wide-16 vs narrow interleaved in one session window — defends against cross-session machine drift (the campaign's hard rule).
- **Default-path isolation**: four env gates (`MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_WIDE=1 MINFER_MMQ_RAW_KD=4` to reach the new path); the f16 default path unaffected.

## 5. Results

| Comparison | Number (7B @2K, same-window interleaved) | Ratio |
|---|---|---|
| wide-16 KD=4 | **1020–1058 tok/s** | vs narrow 441–481 ≈ **2.3×** |
| wide-16 KD=8 | 973–995 tok/s | slightly below KD=4 |
| vs pre-rewrite wide | ~719 tok/s (cross machine states) | **1.44×** |
| vs same-window f16 default | 2284 tok/s | still **~2.2×** behind — below the promotion threshold |

parity green, suite 169/0, greedy token identity. Best config: `MINFER_MMQ=1 MINFER_MMQ_RAW=1 MINFER_MMQ_RAW_WIDE=1 MINFER_MMQ_RAW_KD=4`.

**Why not a promotion.** The 2.3× is against our own old kernel; against the f16 default GEMM path (same window 2284) it is still 2.2× behind, and the f16 path was the default production behavior. Flipping a path that is still 2.2× slower to default would be pure regression — so r12's landing semantics are "the biggest step on the MMQ line", not "a step for the engine's default behavior". MMQ's default-on had to wait for r60 (after post-parity and the q6_K/FA/prepass lines converged). The commit message's "not promotion material yet" means exactly this, and it pre-records the next lever (load-time B-side fragment pre-formatting).

Campaign coordinates: this is the largest structural landing of P6 before r34 (quantize-transpose prepass, +9.72%); of r10's redo recipe (16 chains + ldmatrix + 128×128 tile), two items were executed here and the third (128×128 tile) was achieved along with this shape's 128×128 block tile. The commit also records the next step: load-time B-side fragment pre-formatting — the line that became r14's B-fragment ldmatrix / r18's pre-expansion experiments. The residual 2.2× to f16 was handed to r13's counter forensics (per-MAC warp instruction stream 10.14 vs 6.06 M).

**The basis of the numbers.** Every ratio in this doc names its comparison target, because the machine state drifted across r12's two days (master table footnote 2: within the r12–r25 window, adjacent sessions on this box drifted −9% to +38%): **2.3× is wide-16 vs narrow inside one narrow same-window band** (1020–1058 vs 441–481, mutually comparable); **1.44× is cross-machine-state against the pre-rewrite wide's ~719** (weak control, corroborating only); the KD=8 vs KD=4 ordering (973–995 vs 1020–1058) is likewise a narrow-band reading. Copy the basis along with the number, or 2.3× and 1.44× will be mistaken for two measurements of one quantity.

## 6. Lessons

1. **Write the hardware's required ILP explicitly — the compiler cannot invent chains the source never declared**: 8 chains and 16 chains are both "correct" at the PTX level, but only the latter fills the mma issue ports; chain depth is a source-level contract, not a compiler optimization knob.
2. **Chain depth's price is registers, and registers' budget is occupancy**: 128 accumulators are only payable at 1 block/SM — do the ledger first, then choose the shape; occupancy and depth are inverted in this kernel class (r5–r6 measured it long before).
3. **One-variable slicing makes ratios attributable**: with the B side and rescale untouched by a single character, the 2.3× lands cleanly on "chain depth + A-fragment LDSM", and r14's follow-on stacking had a stable baseline.
4. **Ratios must carry their basis**: 2.3× (in-band vs narrow) and 1.44× (cross-state vs old wide) are both correct; mixing them misleads — draw the interleaved window's boundary first, then the ratio means something.

← 15 · [Index](./README.md) · 17 →
