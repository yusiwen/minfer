# 74 · D4-2 — B0 latent correctness fix + all bitwise occupancy/prefetch axes closed (LANDED)

> **Result**: the q6_K MMVQ v2_pf dispatch had only a lower bound (`id > 8192`) and no upper bound, while the kernel processes only two units per thread (npair ≤ 512) — 7B ffn_down (id 18944, npair 592) had silently dropped 80/592 = 13.5% of the down-projection work on every decode step since D3b-1b; first-step logits deviated by as much as 4.79 yet argmax survived, and every prior gate compared v2_pf against v2_pf, so nobody saw it. The fix (`b31084c`) gives the pipelined route an upper bound `id ≤ 16384`; taller rows take the v2 loop form (identical per-unit arithmetic and ascending accumulation order); the 7B anchors were **re-anchored 50.67 → 49.30 / 49.84 → 48.49 (−2.7%, the honest price of correctness)**, 14B bitwise-unchanged. The same session closed Lever A arithmetically (llama L2 prefetch: the mechanism is inert on our decode shapes, pre-build veto) and Lever B (B1 `__launch_bounds__` register squeeze, B2 160-thread right-sizing, both killed with mechanism), with zero perf regressions.
> **Commit**: `b31084c` (B0) + docs commit. **Date**: 2026-09-09.

## 1. Background — where things stood

The map D4-1's census (`/tmp/d4/D4_DESIGN.md`) drew for decode was: minfer's decode GEMV aggregation is already **ahead** of llama by 0.85 ms — the q4_K-class kernels run at 84% of DRAM peak, and the q6_K class still carries a small occupancy deficit; the real bulk is the attention structure (read at the time as ~2.53 ms/step) plus odds and ends of tail. D4-2's budget was about 3 hours, with the discipline carried over from r44/r46: every lever gets its mechanism arithmetic written before deciding whether to build, and every rejection must leave reproducible numbers behind.

Spreading the census numbers out: one 14B @3254 step is roughly 3.32 ms attention + ~3.6 ms decode GEMV (of which the three q6_K shapes are the bulk) + the remaining tail, against llama's same-window full step of ~5.3 ms — so "decode GEMV is already ahead" is the standing premise, and D4-2's three lines each targeted one thing: attention (left for D4-3; this doc only hands over the decomposition-update brief in §5), q6_K occupancy (Lever B), and a mechanism checklist of "what else have we not copied" (Lever A — the only decode mechanism in llama's kernels that we lack is L2 prefetch).

Of the session's three target classes, B0 was originally just routine audit: check whether each decode kernel's dispatch gate actually matches the shape set it covers. The motivation for this audit came from the previous phase's experience — the decode dispatch layer had branches bolted on repeatedly through the D3 series (D3-7's attn_v MMVQ routing, D3b-1b's tall-row pipelined branch, FusedQKV's two layer classes), and every added branch is one more round of "shape set × kernel coverage" alignment risk. When D3b-1b landed `q6_k_q8_mmvq_v2_pf`, its gate read `id > 8192`: the intent was "tall rows (npair > 256, i.e. id > 8192) take the pipelined form", but the pipelined kernel's coverage ceiling is 512 units, and nothing forced the gate's upper bound to align with it.

The design doc `/tmp/d4/D4_DESIGN.md` organized the session into four levers: **B0** (the correctness audit's product), **A** (porting the L2 weight prefetch from llama's GB10 kernels — it was the only mechanism on the llama side we lacked at the D4-1 reconciliation), **B** (the bitwise occupancy axis of q6_K decode: register budget, pipeline depth, block geometry), **C** (PDL, recorded as follow-up). Every lever had pass/veto criteria pre-registered, and the A/B axes only admitted **bitwise** changes — any loosening of the numeric path was excluded from this session (that belongs to "tolerance-gated redesign", see §5's 0.42 ms true-deficit conclusion).

The consequence on 7B is concrete: Qwen2.5-7B has 10 q6_K layers of ffn_down (id 18944 → npair 592). With 592 > 512, units 512..591 are touched by no thread — on every decode step, on every such row, 13.5% of the down-projection dot product simply does not exist. And 14B's corresponding shape is id 13824 → npair 432 ≤ 512, which was always correct. So of the "7B +3.0%/+2.7%" gain D3b-1b recorded at the time, the vast majority was actually **the skipped work**, not pipelining (§1's correction note already voided D3b-1b's 7B half-row; 14B's +0.44%/+0.24% is the true magnitude of pipelining itself).

The dropped part has one more layer of stealth: 592 = 512 + 80, and 80 is exactly the number of units a 256-thread round can cover (80 ≤ 256) — that is, the loss happens on the **third logical iteration**, while the kernel's first two rounds (0..255, 256..511) are perfectly normal; any validation that only spot-checks the front half of a row sees flawless data. The bug's shape gives it natural immunity to "sampling validation"; only a full-coverage gate (complete logits against a reference) can expose it.

Why could this bug lie dormant for an entire D3b-1b→D4-2 cycle? Two reasons compound. First, **argmax survives**: dropping 13.5% of the down-projection is a fixed systematic gap (the same 80 units dropped every step); the logits are pulled off wholesale, but top-1 never flips on the tested prompts — the max|Δlogit| 4.79 damage is visible only in the dump, and at the generated-text level it "looks normal". Second, **every gate compared v2_pf against v2_pf**: D3b-1b's bitwise gates (114/114 dump memcmp, greedy byte-for-byte) compared the pre-fix and post-fix v2_pf binaries — the bug existed on both sides of the comparison, so the memcmp was always equal. v1 (the fully-covering loop kernel), as the semantic reference, was never pulled into 7B's decode gates.

That is the origin of this doc's headline-level lesson: in a campaign "optimizing correctness", the first job is confirming the engine computes the right thing.

## 2. Principle — the GPU mechanism

**The v2 unit geometry and where the 512 bound comes from.** Inside q6_K's 256-element super-block there are 8 is-pairs (each pair = two 16-element sub-blocks sharing the same set of ql/qh bytes and each carrying its own 2-bit scale). The v2 kernel's mapping is **one thread per pair** (32 elements): `npair = id/32`, and a 256-thread block covers 256 pairs per round. The v2 loop form iterates `for (u = threadIdx.x; u < npair; u += 256)` and covers any npair; v2_pf (pipelined) **unrolls this loop to exactly two iterations** — `u0 = tid`, `u1 = tid + 256` — with both units' weight and activation loads issued before accumulation, moving the second unit's weight-load latency off the critical path (this is the source of D3b-1b's +0.2–0.4%). The price of the two-way unroll is registers: v2_pf is 48 regs/thread vs v2's 40 — the register budget buys exactly this dual-unit pipeline. So v2_pf covers npair ≤ 512, and **this ceiling is decided by the kernel's structure, not a dispatch parameter**.

7B ffn_down's id 18944 → npair 592. `u1 = tid + 256` reaches at most 255 + 256 = 511; units 512..591 fall outside every thread's `two` test and are loaded and accumulated by no one, ever. 80/592 = 13.5% — which matches the −2.7% the 7B anchors dropped after re-anchoring (down-q6K is about 20% of 7B's per-step weight stream; 13.5% × 20% ≈ 2.7% — the arithmetic is self-consistent).

**Lever A: why llama's L2 weight prefetch is inert on our side.** llama's `mmvq` kernel issues an L2 prefetch for "the next round's weight blocks" inside its K loop; the prefetch distance is 2 loop iterations = 2·bpi 256-element blocks, where `blocks_per_iter = vdr·nwarps·32/qi`: from `vecdotq.cuh`, `VDR_Q4_K_Q8_1_MMVQ=2` and `QI4_K=16` (and `VDR_Q6_K_Q8_1_MMVQ=1`, `QI6_K=8`) both give **8 threads per 256-element block**; on the GB10 decode path ncols_dst=1 and nwarps=4, so bpi = 4·nwarps = 16 and the prefetch distance is 32 blocks. The prefetch actually fires only when `bpr > 2·bpi = 32` (when a row has ≤ 32 blocks, the "32 blocks ahead" address is already outside the row and the path does nothing): in 14B **only ffn_down** (bpr = 13824/256 = 54) qualifies; all id-5120 shapes (gu/qkv/q/o, lm_head, attn_v, bpr = 20) never prefetch.

Our decode MMVQ form, meanwhile, is **one thread per unit, 256 threads per round**: npair is 80 (gu/qkv/qo), 216 (down-q4K), 160 (lm_head/attn_v) by shape, and v2_pf's 432 unrolls into u0/u1 — **every decode shape's K loop is exactly 1 iteration**. What llama's prefetch needs is "a distance 2 rounds out"; we have no "2 rounds out" to prefetch into. Then look at the mechanism's own payoff: where llama's prefetch actually engages (ffn_down-q4K) it runs 224.1 GB/s, **below** our prefetch-less 228.6 GB/s. Both families sit at 75–84% of GB10's DRAM peak — in the bandwidth-saturated regime, L2 prefetch produces no new bandwidth; it only hides latency inside the overlap window, and the 8-threads-per-block MMVQ form already hides latency through multi-block parallelism. Conclusion: closed pre-build, no build, no commit (the D4-1 §3 precedent).

A clarification of "8 threads per 256-element block", because it is the root of the two families' shape difference: llama's MMVQ has 8 threads jointly consume one 256-element super-block (`qi` 32-element chunks × `vdr` register pairs per thread, e.g. Q4_K's QI=16, VDR=2 → 2×16 nibbles per thread), 256 threads eat 32 blocks per round, and the loop iterates as many rounds as the row has blocks — ffn_down's bpr=54 gives 54/32 ≈ 1.7 rounds, so a loop body exists and prefetch has something to attach to. Our v2 form, in reverse, presses a whole pair (32 elements) into one thread's register set (4×uint4 ql/qh + q8 slots), and 256 threads eat 256 pairs per round = an entire row with id ≤ 8192 — the loop body is flattened to iteration count 1 on most shapes. These are two legitimate form choices: llama uses shallow threads × deep loops (prefetch has a target), we use wide threads × zero loops (no prefetch needed), and D4-1's measurement already ruled the latter not slower.

**Why the bandwidth-saturated regime kills both the A and B axes at once.** The number 75–84% of DRAM peak is the denominator of the whole session: decode GEMV is a pure streaming workload (one 32-element dot product per weight byte, arithmetic intensity ~0.03 FLOP/B); piling more resident warps into the SM merely issues more concurrent load requests — while bandwidth has slack this lifts throughput (that is how the q4_K class reached 228.6 GB/s), but with only 16–25% headroom left, an occupancy increment's real effect is a longer request queue, latency hidden, throughput unmoved. That is the unified explanation of why B1 (+1 block probed at −0.75%) and B2 (+3 blocks probed at +1.5~3%) both fail to pay, and it is the same coin as Lever A's conclusion "prefetch produces no bandwidth in the saturated regime". Conversely, what is a real lever in this regime is a layout that still sits below 93% effective byte rate (q6_K's 224B stride streams out 14 dead bytes per block) — it cuts dead bytes out of the request stream directly, needing no extra concurrency; D4-4 picked up that thread (the dpl in doc 76).

**Lever B: the occupancy axis's mechanism ledger.** Fresh ncu (2025.3.1, sudo recipe, 14B @3254) confirmed v2_pf holds 48 regs/thread → **Block-Limit-Registers 5 blocks/SM**, below Block-Limit-Warps 6 (sibling control kernels at 40/39 regs both reach 6 blocks). Three candidate directions: (B1) use `__launch_bounds__(256,6)` to force the compiler down to 40 registers, freeing room for a 6th block; (B1c) the reverse — hand npair-432 to the v2 loop (1 unit/thread), trading 6 blocks × 1-unit-MLP against 5 blocks × 2-unit-MLP; (B2) right-size 256 threads to 160 (zero idle warps for the npair-160 shapes, 9 resident blocks × 5 active warps vs 6 × 5). Each one's mechanism expectation is "occupancy +1 block → more warps to hide latency", but each must pass both the bitwise and the wall-clock gates — occupancy itself is not the goal.

B1c's matchup deserves expansion, because it is the only "free" control of the three (both kernels already exist, both already bitwise-gated). 5 blocks × 2-unit-MLP: each resident warp holds two units' weights and activations already loaded into registers, the accumulation chain being the two serial segments "dot(u0) → dot(u1)", with the second segment's load latency already hidden by the pipeline; 48 registers → 5 blocks per SM. 6 blocks × 1-unit-MLP: each warp looks at one unit per round, and load latency must be hidden by **cross-block warp switching**; 40 registers → 6 blocks per SM. In theory the latter has 20% more resident warps (30 → 36); measured, v2_pf wins/ties — inside a 256-thread × 4-warp decode block, the per-block reduction (`mmvq_block_reduce`'s shfl chain) and tail-wave effects eat the gain of "2 more blocks", while the 2-unit pipeline hides latency more efficiently inside a block. That is direct evidence that "register-capped one block below the warp cap" is not a free lunch: that one block's price is 48 registers of pipeline funding, and neither cutting it (B1) nor bypassing it (B1c's motive) pays it off.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**B0's fix shape: an upper bound + rerouted landing, not a generalized kernel.** Three options: (a) add a loop to v2_pf so any npair is covered — this cancels the unroll and makes short rows also pay the 48-register price; (b) split by npair at dispatch — npair ≤ 512 goes to v2_pf, everything else to the v2 loop — zero kernel changes, since the v2 loop is already the full-coverage form for any npair; (c) write a third "generalized pipelined" kernel. (b) was chosen: the v2 loop's per-unit arithmetic is identical to v2_pf's and the accumulation order is the same ascending u — exactly the bitwise invariant D3b-1b established — so the rerouted taller rows need no new correctness proof. The bound is written as `id ≤ 16384` (= 512 units × 32), strictly equivalent to npair ≤ 512.

One design detail worth recording: why the gate's deciding quantity is `id` and not `npair`. At the dispatch point `id` is a call argument (ready-made on the host side), while `npair = id/32` is a derived quantity inside the kernel — recomputing npair in the host gate and comparing is semantically equivalent to writing `id ≤ 16384` directly, but the latter presses the conversion into a constant, with the conversion noted in a comment. What really matters is not the notation but the **alignment obligation**: between the kernel's coverage ceiling (512 units) and the gate's ceiling (16384) there must be a comment-level conversion chain, so that anyone changing one end in the future sees the other end in the same screen of code. B0's root cause was precisely that this chain was missing in D3b-1b — the kernel comment said "npair > blockDim here (dispatch-gated)", pushing the obligation onto the gate, and the gate wrote only half of it.

**The choice of verification anchor: the `-n 1` first-step dump.** After the fix, a comparison point is needed that simultaneously decides "7B is fixed" and "14B was not touched". Full-run dump cross-binary comparison was rejected (reason in 3.3); the first step (prompt-only context, bit-identical inputs on both sides) is the only clean point; the 7B reference is the fully-covering v1 kernel — for v2_pf it is a semantic reference, not a bitwise one (see 3.3's rounding class), so the criterion is "the delta falls in the v1-vs-v2 rounding class and argmax matches", not "the delta is zero".

**The pre-build discipline for Levers A/B.** A was vetoed with prefetch-distance arithmetic before any code was written; B2 should likewise have been vetoed before writing code — its prior art (D3b-1c, D3-5 1b) was sitting in the §0 master table; that process lesson is recorded in 3.3.

### 3.2 Key code

**Excerpt A · the before/after of the B0 fix (`git show b31084c -- src/cuda.rs`, the diff hunk in full)** — the change is one condition line plus a comment, and the comment pins the bug's complete mechanism at the dispatch point:

```diff
--- a/src/cuda.rs
+++ b/src/cuda.rs
@@ -4170,7 +4170,16 @@ impl CudaState {
         unsafe {
             if Self::mmvq_v2(id) && blk_stride_padded {
                 // v2's uint4 ql/qh loads need the padded 224B stride
-                if id > 8192 {
+                // D4-2 B0 correctness fix: v2_pf processes exactly TWO units
+                // per thread (u = tid, tid+256 → npair ≤ 512), but the old
+                // gate (id > 8192) had no upper bound — npair=592 shapes
+                // (7B ffn_down id 18944, 10 q6_K layers) silently dropped
+                // units 512..591, corrupting 7B decode since D3b-1b. Guard
+                // the upper bound; taller rows take the v2 loop form, which
+                // walks any npair with identical per-unit arithmetic and the
+                // same ascending-u accumulation order (bitwise for every
+                // npair ≤ 512 shape, which keep the pipelined kernel).
+                if id > 8192 && id <= 16384 {
```

The current tree has since stacked B1c's `MINFER_Q6K_PF` A/B switch at the same spot (pipelined stays the default, "0" forces the v2 loop, consistent in semantics with r60's opt-out family):

```rust
// src/cuda.rs (current tree, the v2 dispatch section of q6_k_decode_mmvq)
if id > 8192
    && id <= 16384
    && !std::env::var("MINFER_Q6K_PF").map_or(false, |v| v == "0")
{
    // D3b-1b: tall rows (npair > 256, e.g. ffn_down id 13824)
    // run the pipelined variant (bitwise-identical, loads for
    // both serial units issue up front).
    launch_q6_k_q8_mmvq_v2_pf(wptr as *const u8, /* … */ 224, stream);
} else {
    launch_q6_k_q8_mmvq_v2(wptr as *const u8, /* … */ 224, stream);
}
```

**Excerpt B · the v2_pf kernel in full (`src/cuda_kernels.cu` 1635–1661)** — the bug's physical evidence sits in the comment on line 1655: the `two` test covers "is u1 out of range", but for npair > 512 shapes the u0/u1 two slots **cannot hold** a third unit to begin with; the kernel does not defend itself and relies entirely on the dispatch gate:

```cuda
// src/cuda_kernels.cu (current tree; b31084c did not touch this kernel)
__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2_pf(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * blk_stride;
    const int npair = id >> 5;                     // id/32: one is-pair per thread
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;
    const uint8_t* wrow = weights + (size_t)row * row_stride;

    float acc = 0.0f;
    const int u0 = threadIdx.x;                    // 0..255
    const int u1 = u0 + 256;                       // 256..511 ← where the upper bound comes from
    if (u0 < npair) {
        Q6kUnitRegs r0, r1;
        q6k_unit_load(u0, wrow, x8row, blk_stride, &r0);
        const bool two = u1 < npair; // npair > blockDim here (dispatch-gated)
        if (two) q6k_unit_load(u1, wrow, x8row, blk_stride, &r1);
        q6k_unit_acc(&r0, acc);                    // both units' loads issue first
        if (two) q6k_unit_acc(&r1, acc);           //  accumulation after = the pipeline itself
    }
    mmvq_block_reduce(acc, output, od, t);
}
```

For 7B ffn_down: `npair = 592`, `u0 < npair` is true for all 256 threads, and `two = (u1 < 592)` is true for all of 0..255 as well — the two slots load units 0..511 and accumulate/reduce as usual, **units 512..591 vanish silently**, with no out-of-bounds access, no NaN, no signal of any kind to make the old gate suspicious.

**Excerpt C · the new landing for taller rows: the v2 loop form (`src/cuda_kernels.cu` 1536–1573 excerpt)** — the full-coverage form for any npair, ascending-u accumulation; post-fix 7B ffn_down runs exactly here:

```cuda
// src/cuda_kernels.cu (current tree, q6_k_q8_mmvq_v2 inner loop excerpt)
float acc = 0.0f;
for (int u = threadIdx.x; u < npair; u += 256) {   // full coverage of any npair
    const int kbx = u >> 3, pair = u & 7;
    const int s0 = 2 * pair, s1 = 2 * pair + 1;
    const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * blk_stride;
    const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    const float sc0 = (float)(int8_t)blk[192 + s0];
    const float sc1 = (float)(int8_t)blk[192 + s1];
    const int chunk = pair >> 2, g = pair & 3;
    // padded 224B stride ⇒ every ql/qh piece is 16B aligned
    const uint4 qla = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32);
    const uint4 qlb = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32 + 16);
    const uint4 qha = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32);
    const uint4 qhb = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32 + 16);
    /* ... nibble unpack + 2-bit high-bit assembly, dp4a per v element ... */
    acc += d8 * sc0 * d * (float)dot0 + d8 * sc1 * d * (float)dot1;
}
mmvq_block_reduce(acc, output, od, t);
```

Its per-unit statements are verbatim the same lineage as v2_pf's (the comment on `q6k_unit_acc`'s accumulation statement reads, in the original: "textually identical to the v2 accumulation statement") — that is the grounds on which the reroute needs no new correctness proof.

**Excerpt D · the pipeline's funding structure (`src/cuda_kernels.cu` 1578–1601 excerpt)** — where the 48 registers come from: both units' complete weight slots (4×uint4 ql/qh ×2 copies each + the scale/d scalars ×2 copies) are all held in registers before accumulation begins; this funding is exactly what B1 wanted to cut:

```cuda
// src/cuda_kernels.cu (current tree; introduced in D3b-1b)
// Same mapping and arithmetic as q6_k_q8_mmvq_v2; both units' loads issue
// before either accumulates so the second unit's weight latency leaves the
// critical path. Bitwise-identical (see the file comment).
struct Q6kUnitRegs {
    uint4 qla, qlb, qha, qhb;      // one unit's ql/qh: 4×16B = 16 regs
    const uint32_t* xw;            // pointer to the q8 activation's 4B slot
    uint32_t shift;
    int g;
    float d, sc0, sc1, d8;
};                                 // ≈ 21 regs/unit × 2 copies + reduction state ≈ 48

__device__ __forceinline__ void q6k_unit_load(
    int u, const uint8_t* __restrict__ wrow, const uint8_t* __restrict__ x8row,
    int blk_stride, Q6kUnitRegs* r
) {
    const int kbx = u >> 3, pair = u & 7;
    const uint8_t* blk = wrow + (size_t)kbx * blk_stride;
    r->d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    r->sc0 = (float)(int8_t)blk[192 + 2 * pair];
    /* ... the uint4 loads of sc1 / ql / qh, all issued before accumulation ... */
```

### 3.3 Pitfalls

- **Cross-binary full-run dump comparison failed twice, invalid both times.** After generated tokens diverge, the two runs feed different contexts into subsequent steps — from then on every logits/KV difference is cascade noise, measurable at 1e15–1e18-magnitude "deltas", unrelated to compute corruption; additionally the persistent KV region's **unwritten tail slots** hold pool garbage whose layout is binary-dependent. The only clean comparable point is the first decode step (`-n 1`: prompt-only context, identical inputs on both sides). This rule later became the D series' standard instrument (collected in the doc 77 methodology).
- **v1 is not a bitwise reference for v2.** In the fix validation, 7B post-fix vs the v1 reference showed max|Δlogit| = 0.254, alarming at first glance — but 14B's own v2-vs-v1 is also 0.231: this is the v1-vs-v2 rounding class that has existed since the R2 era (argmax preserved), a **pre-existing difference**, not an incomplete fix. The adjudication gate must distinguish "fixed down to the reference's rounding class" from "fixed to the reference".
- **B1's spill shape.** `__launch_bounds__(256,6)` squeezed v2_pf from 48 to 40 regs while generating STACK 40 (10 spill words) — the forced-occupancy ledger is not "8 fewer registers" but "more spill traffic in the hot loop". Seeing REG 40 + STACK 40 at the SASS level should already be the stop signal.
- **Prior-art search should happen before writing code.** B2 (160-thread right-sizing) is verbatim the same lineage as D3b-1c's and D3-5 1b's conclusions ("9 blocks×160 live threads ≈ 6×256 allocated — the idle-thread gain does not exist"); those rows had been sitting in the §0 master table for a long time, and re-measuring's only value was upgrading "neutral" to "measured worse, with per-kernel control". Process correction: grep the master table's mechanism keywords before writing a kernel.

## 4. Verification

- **14B pre-vs-post full dump gate**: `MINFER_GRAPH_DUMP` 107/107 gate files byte-identical (the 7 node{N} diffs = the documented pool-slot instrument class), plus greedy per-token byte-identical — defends against the fix-only tree producing any behavioral drift on the npair ≤ 512 paths (this fix does not touch those dispatches).
- **Why the pool-slot instrument class can be exempted**: those 7 `node{N}` files have pool slot numbers in their names, and the allocator's slot assignment order may differ between binaries (the same-named buffer lands in a different slot → different filename, same content). The adjudication protocol is "the other 107 files byte-identical + the node{N} diffs reproduce pre-vs-pre" — i.e. self-compare the unchanged binary once, and the same 7 node{N} diffs appear, proving it is instrument noise rather than behavior. This protocol was reused verbatim in D4-4 (doc 76).
- **7B first-step logits gate (-n 1)**: post-fix vs the fully-covering v1 reference max|Δ| 0.254 with identical argmax (= 14B's own v2-vs-v1 rounding class of 0.231); pre-bug vs the v1 reference max|Δ| 4.72 — the same gate pins both the bug's magnitude and the fix's completeness.
- **7B greedy stream alignment**: post-fix, 7B greedy follows the v1 reference stream token by token — defends against the intermediate state of "the dump is right, generation still drifts".
- **suite 173/0/3**: full-model regression, defending against the dispatch change rippling into other quantization paths (the gate change lives in the dispatch layer and should only affect the q6_K tall-row branch, but running the full suite is baseline discipline).
- **7B's "semantic reference" gate**: the fix introduced a new dispatch landing (the v2 loop taking over npair-592), a landing that had never run on 7B decode before — so beyond the dump gate, 7B greedy's token-by-token alignment against the v1 reference stream covers the "new landing × real generation loop" combination.
- **B1/B2's control gates**: B2 first passed the bitwise 98/98 gate files, then used nsys for per-kernel isolated measurement, with the v2_pf/q4_K rows as the ±0.6% control group — defends against reading window drift as a kernel effect.

## 5. Results

**The fix's own wall clock (pre = the `b31084c` binary, post = the final tree; interleaved 3× medians)** — a fix-only tree should be identical, and it measured so:

| config | pre (B0) | post (final) | Δ | note |
|---|---:|---:|---:|---|
| 14B tg128 | 24.13 | 24.01 | ≈ 0 (noise; pair-3 straddle) | fix-only tree behaviorally identical, as designed |
| 14B @3254 | 22.12 | 22.42 | ≈ 0 (noise) | guards hold (≥ 24.0 / ≥ 21.9) |
| 7B tg128 | 49.21 | 49.29 | +0.2% | re-anchor below |
| 7B @1641 | 48.41 | 48.50 | +0.2% | guards hold |

**7B re-anchor (the buggy pre-D4-2 binary vs the post-fix B0 binary, 3× interleaved medians)**: tg128 **50.67 → 49.30 (−2.7%)**, @1641 **49.84 → 48.49 (−2.7%)** — both shapes exactly −2.7%, the honest price of "starting to count the 13.5% of down-q6K work" (down-q6K ≈ 20% of 7B's per-step stream). The old 7B guards (49.0/47.9) were built on a work-dropping kernel and are voided: the new guards are **tg128 ≥ 48.3 / @1641 ≥ 47.5**. 14B cross-check: the pre/post medians interleave within window noise (the binary path is bitwise-identical).

Both shapes, same-day window, same fix, exactly equal drops (−2.7%/−2.7%) — that is not coincidence but the signature of arithmetic self-consistency: the fix's only behavioral change is letting ffn_down-q6K's 10 layers count the 13.5% of missing dot products back in, a workload that is constant per decode step, so the decode wall clock — insensitive to KV length — withdraws by the same proportion. Had the two shapes dropped by clearly different amounts, that would instead signal something else mixed into the measurement. The guard re-anchoring rule is recorded alongside: **correct-over-fast** — the guard floor is re-set to follow the "correct engine", never keeping a work-dropping path to preserve an old number; the old guard's provenance (which measurement, which binary set it) goes into the docs so the next re-anchor can trace it.

**Same-window vs-llama (llama-bench `ca3d5a3e1`, 2026-09-09 window, aligned to the -n 128 anchors)**:

| config | minfer (post-D4-2) | llama | ratio |
|---|---:|---:|---:|
| 14B tg128 (KV≈0) | 24.01 | 24.14 ± 0.02 | **0.995× (parity)** |
| 14B @3254 | 22.42 | 24.14 (tg128 @ -p 3254) | **0.928×** |
| 14B pp3254 | ~1816–1859 | 1634 ± 112 | **1.11–1.14×** |
| 7B tg128 (KV≈0) | 49.30 | 47.65 ± 0.09 | **1.035×** |
| 7B @1641 | 48.49 | 47.69 ± 0.01 | **1.017×** |

Note: D4-1's 0.921× headline compared minfer tg64-exclusive against llama tg8, and still inside an sglang co-tenanted window — aligned to the tg128 anchors, 14B short-KV is parity, and 14B long-KV's 0.928× is the real gap (the attention structure item, handled by D4-3). 7B's ratios are on the **correct engine** (the pre-fix 7B numbers were flattered by the dropped work).

The operational meaning of this correction deserves to be spelled out: 0.921× (tg64-exclusive/tg8) and 0.995× (tg128/tg128) differ by 7 points with no code change at all — only the anchor alignment and the co-tenanted window changed. Decode A/B comparisons must (1) use the same -n anchor on both sides, (2) use the same window or explicitly state the window drift magnitude, (3) compare median to median. After D4-2 these three became the standard for decode comparisons (collected in the doc 77 methodology).

**The three levers' closing numbers**:

- **Lever A (llama L2 prefetch)**: closed pre-build. Mechanism ledger: the prefetch distance 2·bpi = 32 blocks only engages for rows with bpr > 32 (in 14B only ffn_down, bpr 54); our kernel runs exactly 1 K iteration per decode shape, with no "2 rounds out" to prefetch into; where llama's own prefetch engages it runs 224.1 GB/s < our prefetch-less 228.6 GB/s; both families sit at 75–84% of GB10's DRAM peak. No build, no commit.
- **B1 (`__launch_bounds__(256,6)`)**: SASS REG 48→40 + STACK 40 (10 spill words); probe tg128 +0.25%, @3254 **−0.75%** (22.54→22.37 medians) → **KILLED** — the spill cost exceeds the 6th block's gain.
- **B1c (rerouting npair-432 to the v2 loop, `MINFER_Q6K_PF=0`)**: 6 blocks × 1-unit-MLP against 5 blocks × 2-unit-MLP — v2_pf wins/ties (tg128 24.08 vs 24.02/24.08; @3254 median 22.28 vs 22.16, one noisy pair each) → **v2_pf kept by default**; the env is kept as a documented opt-out.
- **B2 (160-thread right-sizing, = the D3b-1c re-measurement)**: bitwise 98/98, but nsys per-kernel: lm_head 3243.7 → 3292.7 µs (**+1.51%**), attn_v 24.41 → 25.15 µs (**+3.04%**), v2_pf/q4_K rows ±0.6% (control group) → **KILLED**.

The B line's stopping rule (pre-registered in the brief): q6_K's occupancy gap **cannot be closed within this kernel family under the bitwise constraint**. The limiter record stands as originally judged: v2_pf holds 48 regs for the dual-unit pipeline, pinned by registers one block below the warp cap; cutting registers (B1) pays spill, cutting the pipeline (the B1c control) loses intra-block latency hiding, changing block geometry (B2) measures worse — all three roads were walked to the end, and the geometry already sits at the measured optimum. The remaining 0.42 ms of true deficit can only be reached by tolerance-gated redesign (numeric slack traded for layout freedom) or the attention structure lever, neither of which belongs to this session.

**@3254 gap decomposition update (the brief handed to D4-3)**: (1) q6_K's "205 GB/s" census rate counts 210-byte blocks while the padded stride actually streams 224 bytes — the true DRAM rates are ≈ 218.9 (ffn_down) / 221.6 (lm_head) / 189.5 (attn_v), against the q4_K class's 228.6, making the true q6_K-vs-class deficit ≈ **0.42 ms/step**, not 1.1–1.2; (2) the bitwise occupancy axis is fully closed (above), and realizing this 0.42 ms requires a tolerance-gated redesign; (3) the ~2.53 ms attention structure item is untouched and remains D4-3's prize (later corrected by D4-3 to ~1.6 ms — the anchor had stood on a llama-bench artifact); (4) 7B decode numbers before `b31084c` are incomparable with anything after it.

**Lever C (PDL)**: recorded as follow-up, not built this session — the interaction risk with CUDA-graph capture needs its own session (llama attaches `cudaGridDependencySynchronize` + programmatic stream serialization to every decode kernel; the target is the ~0.3 ms launch-gap residual the graphs left behind). This item was formally closed in D4-4 L2 (see doc 76).

The ledger at session close: one correctness fix landed (7B is trustworthy from here on), four levers closed with measured mechanism (A pre-build, B1/B2 killed, B1c control archived), zero perf regressions, and the corrected @3254 gap decomposition handed to D4-3 — plus one process correction (master-table prior-art search moved up front) and one instrument rule (`-n 1` first-step dump) entering the methodology library. This is exactly the expected output shape of a "Tier B" session: not every line gains performance, but every remaining millisecond has an owner.

## 6. Lessons

1. **A dispatch gate is a correctness surface, not a performance surface.** Any dispatch of the form "the kernel only covers ≤ X" must have a dual upper bound; the coverage assertion belongs in the kernel comment with the host gate mechanically aligned to it (`npair ≤ 512 ⇔ id ≤ 16384`), or the next new shape is the next 7B.
2. **The `-n 1` first-step dump is the only clean cross-binary comparable point.** Dump differences after generation diverges = cascade noise + unwritten KV tail-slot garbage; a 1e15–1e18 "delta" is not corruption.
3. **After a correctness fix, anchors must be re-set.** A guard set on a wrong engine is a negative asset (the old 7B guards flattered dropped work as speed); the −2.7% re-anchor is correctness's price and belongs in the record, not hidden.
4. **Grep the master table's mechanism keywords before writing kernel code.** B2's prior art sat in the table for two sessions; pre-build arithmetic (Lever A's approach) is an order of magnitude cheaper than post-build measurement.

← 73 · [Index](./README.md) · 75 →
