# 60 · r57 — FA KV staging double buffering (REVERTED)

> **Result**: both attempts fail the greedy-32 byte-identity gate — attempt 1 forked at token 19 (the fix uncovered a real bug in the attempt code: the tail-tile's stage→global O copy was wrapped in a warp guard); attempt 2, bug fixed, **still forked**, with the residual being r50-class inherent rounding drift: **FA_TQ is also a tile size**, so the "r50's caution doesn't apply" premise was falsified. REVERTED per the two-attempt stop rule; the FA kernel keeps the FA_TQ=64 / FA_TKV=32 / per-tile synchronous drain shape. Session E's basket net gain is left to item 2 alone (r56, +2.35%).
> **Commit**: `c3268cc` (record-only commit). **Date**: 2026-09-06.
> *Code provenance note: the code attempts never entered the repository (a revert destroys them), so `git show` cannot reference the changes themselves; this doc's code excerpts are from the current tree (= the post-revert state), and the attempted shape is reconstructed by narration.*

## 1. Background — where things stood

After r48 (FAP2) moved the FA prefill kernel's softmax into registers, the FA kernel went 5.16 → 2.12 ms (2.43×) and whole prefill +5.6%; r55's convergence statement had FA at 5.3% of the wall, the largest single residual outside the GEMMs. Session E's basket lined up five items: item 1 = FA KV staging double buffering (this doc), item 2 = the q6_K A-side bundle (r56, LANDED), items 3/4/5 = rms_nw roofline, host-stall root-cause, and tail pre-grow (all unreached — budget exhausted). FA went first because the q6_K line had just used cp.async + double buffering (KDR=2) to hide the "staging wait" inside compute (the r39/r53/r56 trio), and the same trick looked like it could transplant directly onto FA's KV movement.

The FA kernel's KV supply shape at the time (current tree, i.e. the in-service post-revert form): each KV tile iteration issues cp.async at its top but **immediately follows with a `wait_group 0` synchronous drain** — DRAM latency is exposed once per KV tile iteration. This is exactly the target shape of the "remove the wait" (WAIT) class of lever as r45/r56 defined it:

- **in service**: stage(kt) → commit → wait 0 → syncthreads → compute. The serial k-loop pays the full latency once per tile. The tile count also lives here: using the pp3314 anchor as an example, each (q-block, head)'s KV chain passes ~104 tiles (3314/32), and each tile's serial head = issue + DRAM latency + barrier — that is the pool "hide the latency" wants to eat.
- **the attempt**: a prologue fetches tile 0 first; inside the loop the issue changes to `kt+FA_TKV` into `buf^1` with `wait_group 1` — kt+1's copies fly under kt's QK^T/softmax/P·V.

But there is a prior record that must be routed around: **r50** (the FA_TKV 32→16 occupancy experiment) — it pushed to 3 blocks/SM, but the occupancy gain was eaten by the doubled per-tile sync/softmax overhead (−0.5%/−0.01%), and it **lost greedy byte identity**, concluding "FA_TKV reduction is a dead lever that also breaks byte-identity." Earlier still, r46 (FAP1) took FA_TKV 64→32 + S/P row padding to kernel −11% but only +0.27% whole-prefill (FA was not the wall-critical path then), REVERTED — the FA line was only truly closed by r48 (FAP2) register softmax. r57's premise was written explicitly: **FA_TKV stays 32** (KV tile boundaries unmoved → each row's KV-column accumulation grouping and online-softmax rescale points unmoved), and only FA_TQ shrinks 64→48 to free shared memory for a second K/V buffer — "r50's caution is only about the KV tile and does not apply to TQ." This doc is the record of that premise being falsified.

The smem arithmetic (the design's starting point, checked item by item): in service, `(FA_TQ + 2×FA_TKV) × (hd+8) × 2 = (64+64)×136×2 = 34,816 B`; the attempt's composition is Q at 48 rows `48×136×2 = 13,056 B` + two copies each of K/V `4 × 32×136×2 = 34,816 B`, totaling **47,872 B**, targeting 2 blocks/SM. FA_TQ=48 also carries a by-product: 128 threads = 4 warps and 48 rows = 3 sixteen-row blocks, so **warp 3 no longer owns Q rows** — it is assigned to staging full-time.

## 2. Principle — the GPU mechanism

### 2.1 What double buffering buys in this class of kernel

The q6_K NB-BT kernel's KDR=2 double buffering (r39) is a campaign-verified template: each kt has two staging planes, kt+1's global→smem copy overlaps kt's mma compute, and `wait_group 1` waits only for "the previous group" to land (cp.async groups complete in issue order — a semantic guarantee). Its record on q6_K: 1568.7 → 1777.5 tok/s (**+13.3%**), attn_v kernel −19.7% — on the premise that staging latency was genuinely on q6_K's critical path then. The gain mechanism lifts the global-read latency (hundreds of cycles) off the serial chain and hands it to the async units. FA's KV supply is structurally isomorphic: the KV tile stream feeds the online-softmax main loop, and the main loop does three compute segments per tile (QK^T, softmax, P·V) — enough to shelter one copy. r57's bet was "isomorphic ⇒ same gain."

### 2.2 The in-service form's byte and latency account

The current-tree kernel's (`src/cuda_kernels.cu`) tile constants and smem layout:

```cuda
#define FA_TQ 64
#define FA_TKV 32
// Shared layout (dynamic, ~35 KB — opt-in via cudaFuncSetAttribute):
//   Qs [64*hd] f16   q tile (scale folded in, f16 for the tensor-core QK^T)
//   Ks [FA_TKV*hd] f16   K tile      Vs [FA_TKV*hd] f16  V tile
```

The staging helper is 16 B cp.async (introduced in P5.3, `fc07c04`), with out-of-range rows zero-filled via src-size 0:

```cuda
__device__ __forceinline__ void fa_stage_kv_async(
    const __half* __restrict__ k, const __half* __restrict__ v,
    __half* Ks, __half* Vs, int kt, int kv_end, ...) {
#if __CUDA_ARCH__ >= 800
    for (int c = tid; c < FA_TKV * hd / 8; c += nthreads) {
        int r = (c * 8) / hd, d = (c * 8) % hd;
        int p = kt + r;
        bool full = p < kv_end;
        unsigned kd = (unsigned)__cvta_generic_to_shared(Ks + r * sstr + d);
        ...
        int sz = full ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(kd),
                     "l"(k + (size_t)p * stride_kv + hk * hd + d), "r"(sz));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(vd),
                     "l"(v + (size_t)p * stride_kv + hk * hd + d), "r"(sz));
    }
```

But the main loop's call pattern is single-buffered + fully synchronous drain — cp.async merely changes the copy's issuer, and not a single beat of latency is hidden:

```cuda
for (int kt = 0; kt < kv_end; kt += FA_TKV) {
    // stage K/V tile (padded stride, zero-filled beyond kv_end)
    fa_stage_kv_async(k, v, Ks, Vs, kt, kv_end, hk, hd, stride_kv, sstr, tid, 128);
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n");
    asm volatile("cp.async.wait_group 0;\n");   // ← drains immediately: zero overlap
#endif
    __syncthreads();
    // S = Q·K^T via wmma ... in service: the full DRAM latency exposed once at the top of every tile
```

The launcher-side smem account (current tree `src/cuda_kernels.cu`) — the `34,816 B` is computed here, and the r57 attempt changed exactly this one expression:

```cuda
int launch_fa_prefill_f16kv(...) {
    // Qs + Ks + Vs only (S/P no longer go through shared memory). sstr = hd+8
    // padding; Ks/Vs are FA_TKV rows (the r46 launcher's 3*FA_TQ bug is gone).
    size_t smem = ((size_t)FA_TQ + 2 * FA_TKV) * (hd + 8) * 2;
    //   = (64 + 64) × 136 × 2 = 34,816 B (in service)
    //   attempt = (48 + 4×32) × 136 × 2 = 47,872 B (Q shrunk rows + K/V double buffering)
```

(A note in passing: the comment block above the helper says "Overlapped with the previous tile's … via double buffering," which does not match this wait-0 call site — it is a stale historical comment; in an audit, trust the call site.)

### 2.3 Why byte-identity "should" hold (and where it actually broke)

The per-row float accumulation chain depends on two things: **the KV tile boundaries** (which fix the online-softmax m/l rescale points) and **each row's KV-column-to-lane grouping** (which fixes the summation's float associativity order). The former deserves a sentence of expansion: at the end of every KV tile, online-softmax performs `m_new = max(m_old, tile_max)` and an O rescale by `exp(m_old − m_new)` — every time a KV tile boundary moves, the rescale points move and the float chain changes shape; that is why r50 lost identity the moment it touched TKV. The premise assumed both of these are determined by FA_TKV alone, and that TQ only changes "which warp owns which 16 rows" — pure scheduling, outside the float chain. The measured outcome (attempt 2 still forks) overturns it: **any change to tile geometry perturbs the summation grouping somewhere in the float chain**, TQ included; the specific bit path was not chased to the bottom — the two-attempt stop rule triggered first. From r50 (TKV) to r57 (TQ), two independent samples support the conclusion: for this kernel, **the tile constants are part of the numeric contract**.

The conclusion also has a campaign-level corroboration: D3-4's tolerance calibration measured the end-to-end magnitude of **any** accumulation-order change at max|Δlogits| 0.38/0.39 (14B/7B) — that is, if a tile change really moved the float chain, the output would drift systematically at this magnitude and the greedy stream would fork sooner or later; there is no mild version where "the tile changed but only the last bit wobbles." r57's fork shape (the greedy-32 stream going wholly off after token 19) matches this magnitude exactly.

### 2.4 The attempted shape (reconstructed by narration)

A four-part pipelined structure, each item paired with the in-service form's change point:

1. **prologue**: before entering the KV loop, stage tile 0 into `buf 0` (`commit + wait_group 0 + syncthreads`) — the first data is in place, and the loop body no longer has the "wait, then compute" serial head.
2. **in-loop issue**: each iteration issues the K/V copies of `kt + FA_TKV` into `buf^1` (double-buffer alternation), `commit`ed as a group.
3. **the wait**: `wait_group 1` — kt's group has landed (groups complete in issue order) and kt+1's copies stay in flight.
4. **warp division of labor**: FA_TQ=48 → 3 sixteen-row blocks; warps 0–2 own Q rows and do QK^T/softmax/P·V, while **warp 3 does staging full-time** (issue and zero-fill) — otherwise it idles through the whole loop.

Total smem 47,872 B (§1's arithmetic), targeting 2 blocks/SM. The numeric expectation: FA_TKV=32 unmoved → per-row accumulation order unmoved → byte-identical — an expectation that died in §2.3.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Touch TQ only, not TKV**: quarantining r50's prior record outside "the KV tile boundary," the presumed float-chain entry point — this isolation assumption is exactly what got falsified, but it was the only design that could preserve byte-identity at the time.
- **The two-attempt stop rule applies in advance**: a standing campaign discipline — at most two attempts per lever; if the second still hits a red gate, revert. It guards against the "so close" sunk-cost spiral; r57 is the rule's textbook execution (attempt 1 fixed a real bug, attempt 2 falsified the premise, stop).
- **Gates first, performance last**: greedy-32 byte identity is the first gate, before any performance measurement — until the correctness gate is green, performance numbers are meaningless (this doc ends with no wall numbers precisely because the gate never went green). This was also the campaign's consistent gate order before the D series.

### 3.2 Key code

See §2.2 — the two excerpts of the in-service form (= the post-revert status quo) are "what the revert returned to." The attempted change's shape is narrated in §2.4; its diff has no commit to cite (see the code provenance note at the top).

### 3.3 Pitfalls

- **A real bug (in the attempt code, not in the in-service kernel)**: after attempt 1 forked, a section-by-section hunt found the tail-tile's stage→global O copy written inside a warp guard — tail rows outside the guard would **never be written out**, and greedy-32 forked at token 19. This bug explains all of attempt 1's forking, and the fix itself was correct; it was not an incumbent bug (the in-service kernel never forked).
- **The premise falsified**: with the real bug fixed, attempt 2 still forked — the residual is §2.3's inherent rounding drift, independent of implementation quality. This kind of drift "cannot be fixed"; only the numeric contract can be changed.
- **Rebuild md5 is unusable as revert verification**: the rebuilt binary's md5 after the revert differs from pre-attempt — nvcc builds are not bit-deterministic; whether a revert is clean is judged by behavior (the greedy stream + perf sanity), not by md5.

## 4. Verification

- **greedy-32 byte identity**: the first gate and the fatal one — attempt 1 used it to catch the warp-guard bug, attempt 2 used it to falsify TQ-independence. Guards against: any float-chain perturbation that "shouldn't change in theory."
- **perf sanity (post-revert)**: 3222.4 ≈ the landed median 3212.5, and the greedy stream byte-identical to the record — guards against "an unclean revert."
- **the two-attempt stop**: stop at the second red gate; no third TQ/buffering combination is chased — guards against sunk-cost-driven endless debugging.

## 5. Results

| Item | result |
|---|---|
| attempt 1 | greedy-32 forked at token 19 → traced to a tail-tile O copy warp-guard bug in the attempt code (a real bug, fixed) |
| attempt 2 | **still forked** after the bug fix → r50-class inherent rounding drift; FA_TQ is also a tile size |
| disposition | **REVERTED** per the two-attempt stop rule; no wall numbers (the correctness gate never went green, so performance measurement would be meaningless) |
| revert verification | rebuild md5 differs (expected — builds are not bit-deterministic); the greedy stream matches the record; perf sanity 3222.4 ≈ 3212.5 |
| basket settlement | Session E's net gain rests on item 2 alone (r56, +2.35%); items 3/4/5 (rms_nw roofline, host stalls, tail pre-grow) unreached — budget exhausted |

**The veto mechanism**: this lever's value proposition was "a pure-overlap gain with zero numeric risk," and that proposition rests on "TQ stays out of the float chain" — falsified outright by attempt 2. With byte-identity as a hard gate, the mechanism is **unconditionally unlandable**.

**Retry conditions**: double buffering becomes discussable again only when some FA change **deliberately gives up** byte-identity — at that point it passes the tolerance gates as part of that change, not as an independent "free" lever. The tolerance gate's concrete shape already has a calibrated precedent in the campaign (D3-4 h4w): an end-to-end max|Δlogits| 0.30–0.39-class tolerance (the inherent magnitude of any accumulation-order change), a hard argmax gate, greedy flips attributed one by one (sampler knife-edge vs kernel drift), and a temp-0.8 control group. In other words, "accepting non-identical output" is not a relaxation of verification but an exchange for a more expensive yet executable verification.

## 6. Lessons

1. **r50's caution generalized into a law: any FA tile-size change (TKV or TQ) breaks byte-identity** — tile geometry is the float summation grouping itself; every tile constant is part of the numeric contract.
2. **A "free knob" is only a hypothesis until it has been shown to stay out of the accumulation chain**: before building the mechanism, spend ten minutes verifying the invariance premise with one dump gate; r57 ran the order backwards and paid two full implementation rounds.
3. **Failed attempts also pay rent**: the tail-tile O copy's warp-guard bug would have remained a landmine without this hunt — the red gate did not run in vain. The negative conclusion "FA_TQ is also a tile size" itself entered the campaign record, and it blocks all future "touching only TQ should be fine" variant proposals.
4. **Judge a revert by behavior, not md5**: nvcc builds are not bit-deterministic; the greedy stream + perf sanity are the evidence of "a clean revert."

---
← 59 · [Index](./README.md) · 61 →
