# 48 · r45 — cp.async for the q6_K A-side staging: mechanism confirmed, wall-neutral (REVERTED)

> **Result**: the A-side bulk LDG→STS replaced with explicit PTX `cp.async.cg.shared.global` (16 B, group-count pipeline): kernel Duration 659,680 → 592,448 ns (**−10.2%**), long_scoreboard absolute volume 4.38 → 3.57 cy (**−18.4%**), Compute(SM) 37.4 → 42.1%, registers 80 / LOCAL 0 (r41's 4 B spill vanished along the way) — but whole-prefill 2604.5 → 2595.6 tok/s (**−0.34%**, noise, below the +1.5% bar) → **REVERTED**. The q6_K line declares convergence: after r41 this GEMM is no longer the prefill bottleneck, and a faster kernel cannot reach the wall. SASS forensics trap: cp.async in SASS is **`LDGSTS.E.BYPASS.128`**, not the literal `CP.ASYNC` — the gate must grep `LDGSTS`.
> **Commit**: `9825ffd`. **Date**: 2026-09-05.

> **Provenance note**: r45's code change was reverted after measurement (cmp-match HEAD r41), and `9825ffd` is a docs-only commit whose diff contains no code. The excerpts here come from the **current tree's** `src/cuda_kernels.cu`: `gemm_cp16` and the A-side cp.async staging are the final landed form of r45's mechanism (r53 packaged the B side, r56 installed the same mechanism back on the A side, with comments explicitly signing "r45's mechanism"), and the group-count pipeline main loop is unchanged word for word.

## 1. Background — where things stood

r44 had just dissolved the W_exp stride mismatch while also delivering a costlier conclusion: pre-expanding B (deleting the recomb ALU) went parity all-green and kernel −10.9%, but whole-prefill −0.42% — after r41 the q6_K GEMM is no longer the wall. MMQ-analysis §11.24's closing wrote that pre-expansion is a "correctness fix for the W_exp addressing but a dead end for wall perf", and that **the lever that can actually hide B's global→smem staging latency inside compute is cp.async (llama.cpp's structure)**. r45's implementation cashes in that lever. Where does its absence stall? Of the attribution table's two biggest items (recomb 45% + A-staging 28%), r44 addressed only the former; the A side's 28% is inherent to the synchronous LDG→STS chain — as long as staging is "load into registers, then write smem", this latency must queue in the warp's instruction stream every tile. And whether the q6_K line still has a second half (the stacked gains of bundle form) depends entirely on whether both mechanisms can be proven effective — r45 is the other half of that proof.

### 1.1 Why the A side first, and the timeline

Three reasons to start with the A side. First, **the A side is the pure bulk-copy candidate**: after r34 the BT route moved activation quantization into the prepass, so the `qa8`/`sda` the kernel reads are pre-transposed q8 planes and staging is a straight per-16 B transport with no per-element transform — cp.async is a natural drop-in. The B side was still stuck on the recomb (ql+qh unpack); its pure-copy form needed W_exp's fix (r44's dense indexing) as a premise. Second, in r43's PC-sampling attribution the A-side staging STS.128 held 28% of stall samples, the second-biggest source, worth a shot. Third, this doubles as a **cheap convergence test**: if hiding the attribution's second-biggest latency speeds the kernel ~10% and the wall still does not move, then "the q6_K line has converged" is not a conjecture but an empirical fact, and the budget should pivot wholesale.

Background numbers: after r41 whole-prefill ~2600 tok/s (vs-llama 1.27×), the q6_K kernel matched-nt ~0.59 ms, and the campaign bar is +1.5% (whole-prefill). r44 had already set an iron rule: kernel-level wins do not automatically convert to wall clock on the converged line. r45 would either become the counterexample or nail the rule down.

The timeline is itself a signal: r44 and r45 were two consecutive rounds of the same evening (docs commits at 21:38 and 21:59); half an hour after r44's §11.24 closing wrote "cp.async is the real lever", r45's implementation began — attribution, root cause, and mechanism all completed in one day, the fastest hypothesis→verification loop of the entire campaign.

## 2. Principle — the GPU mechanism

### 2.1 The synchronous chain vs the asynchronous copy

The synchronous staging instruction sequence is `LDG.128` (global → register) followed by `STS.128` (register → smem). Between the two instructions sits the full L1TEX/DRAM round-trip latency: the warp issues the LDG and then stall-waits on the STS's data dependency (long_scoreboard), the register serving only as the latency's **display stand**. r44 already proved: swapping the latency's consumer from the recomb ALU to the STS copy leaves the exposure equivalent.

A small account: on the synchronous chain, every 16 B moved costs two instructions (LDG+STS) plus one stall-wait; cp.async is one instruction, zero stall-waits. Halving the instruction count is only the secondary gain (r25 proved instruction-count cuts are wall-inert at 1 block/SM); **the main gain is the stall-wait leaving the warp's critical path** — not one byte moved changed, what changed is what the warp waits for. This explains why r45's kernel −10.2% is worth more than r25-class pure instruction cuts (~0%): it moves the wait structure, not the instruction total.

`cp.async.cg.shared.global [dst], [src], 16` is a different physical path: Ampere's asynchronous copy unit (SASS-level `LDGSTS`) writes the 16 bytes **from global directly into smem, bypassing the register file**. The issuing warp does not wait for the data to land; completion is accounted per commit-group — `cp.async.commit_group` bundles all previously issued copies into one group, and `cp.async.wait_group N` waits until at most N groups are outstanding. The interleave of "transport" and "compute" thus changes from an accident of instruction scheduling into an explicit contract: issue copies → do other compute → wait on groups → barrier → consume.

Three qualifier/semantics details deserve their own listing (r53/r56 both reused this primitive set):

- **`.cg` (cache-global)**: the copy goes through L2, bypassing L1. Staging data is consumed once per byte (once it enters the mma it is done), so caching in L1 is pure waste; `.ca` (cache-all) is for repeatedly-read data.
- **The src-size 4th operand**: in `cp.async.cg [dst],[src],16,sz`, `sz ∈ [0,16]` controls the bytes actually moved, with the remainder **zero-filled**. r45's A-side planes are always full (the prepass zero-fills the padded rows), so it passes `full=true`'s 16; r53's B-side W_exp tail rows (beyond od) rely entirely on `sz=0` zero-fill — the same instruction serves both "pure copy" and "copy with zero-fill" semantics.
- **Groups are ordered**: commit_group completes in issue order (in-order group completion), which is the premise for reasoning precisely that "at wait_group 1 exactly the previous group has landed and the newest is still in flight".

One alternatives-check on "why group counting": among the synchronization granularities available inside a kernel, `__syncthreads` is a block-wide full stop (back to synchronous semantics), spin-on-flag needs `threadfence` + atomic polling (one extra global round trip per tile, with an expensive correctness argument), and CUDA events cannot be used inside a kernel — **commit/wait_group is the only in-kernel asynchronous wait primitive that is both cheap and precisely reasonable about**. Group counting's "coarseness" (you cannot wait for one specific group, only for "N groups remaining") is exactly digested by double buffering's "alternate consumption" structure: the only wait shape ever needed is "the previous group landed, the newest is in flight".

### 2.2 The arithmetic of the group-count pipeline

The kernel already has r39's double buffering (KDR=2, two staging buffers per kt) and r40's 3 blocks/SM. r45 turns the wait into **group counting**:

- Each main-loop round first issues the copies for **kt+1** into buffer `buf^1` and commits;
- `wait_group 1` — allows the newest group (kt+1's) to still be in flight, requiring only that **the previous group (kt's) has landed**, exactly covering the buffer this round consumes;
- The last tile has no "next group" to lean on, so `wait_group 0` drains all outstanding groups;
- One `__syncthreads()` after the wait: cp.async completion visibility is **per-thread**, and smem writes must pass a block-level barrier to be visible to other threads in the block.

Using the first few rounds to unroll the pipeline (group numbers in commit order; the pre-loop already staged tile0→buf0 and committed group 1):

```
prologue:  stage(tile0 → buf0) … commit group1 … barrier
kt=0,buf0: stage(tile1 → buf1)=group2 → wait1 ⇒ group1 landed (group2 in flight) → barrier → compute buf0
kt=1,buf1: stage(tile2 → buf0)=group3 → wait1 ⇒ group2 landed (group3 in flight) → barrier → compute buf1
kt=2,buf0: stage(tile3 → buf1)=group4 → wait1 ⇒ group3 landed (group4 in flight) → barrier → compute buf0
last:      no next tile               → wait0 ⇒ all landed              → barrier → compute the last buffer
```

There is exactly one invariant: **when wait1 returns, the copy group of the buffer this round consumes has necessarily landed, and the groups still in flight write only the other buffer**. In-order group completion guarantees this correspondence without any timing assumptions — precisely why group counting is cheaper than "drain every round" and safer than "hand-rolled events".

The latency is thus pushed into the compute window: kt's copies are already in flight while kt−1's mma chain executes, and the warp's pre-consumption stall-wait shrinks to the group's tail residual. r20's split-phase lesson (long_scoreboard's carrier is the LDG→STS chain) finally has a hardware-level solution in this structure — r20 relied on splitting load and consume into two phases to let ptxas overlap them; cp.async simply removes "transport" from the warp's instruction stream.

### 2.3 The alignment premise and the A planes' luck

cp.async's 16 B form requires both source and destination 16B-aligned. The A side's two planes satisfy this naturally, and the geometry reads straight out of the staging loop: the `qa8` segment moves `KDR·NBI·32` bytes per (tile, kt) window (32 bytes of q8 per chunk per row), the `sda` segment moves `KDR·NBI·4` bytes (4 bytes of scale pair per chunk per row), and both have whole numbers of 16 B chunks as the per-thread copy granularity — no straddling partial chunks, so `full=true` always holds (the prepass-produced planes are laid out on a 16 B grid with padded rows zero-filled). The B side could not do this yet (the recomb is per-element ALU with no contiguous-16 B copy semantics), which is the physical basis for "A first, B later"; B's pure-copy form waited for r53 to complete it with `expand_q6k_dense`'s dense plane (that path also needs cp.async's **src-size zero-fill** semantics for tail rows beyond od — see §2.1's qualifier list).

### 2.4 The relationship with r44: two orthogonal mechanisms

Put r44/r45 on one mechanism map and they address two different facets of the same physical latency: r44 deletes **extra work on the compute side** (the recomb ALU), r45 deletes **exposure time on the wait side** (the LDG→STS stall-wait). The mechanisms neither depend on nor cancel each other — the embryo of r53's later basket thesis: "mechanisms that overlap in traffic but not in mechanism compose". "Traffic overlap" (both reduce the same L1TEX round trips) with "mechanism non-overlap" (one deletes ALU, one hides waits) can stack because deleting work reduces the **number of instructions to issue** while hiding waits reduces the **cycles instructions stall** — they act on the instruction stream's numerator and denominator respectively, and tightening both does not cancel out. The counterexample is applying the same mechanism twice (r42's further dsc widening had no gain): for a wait that no longer exists, a second "hide" has no object. At the time no one could prove wall-clock composability (each was wall-neutral solo), but r45's design already consciously preserved composability with r44: swapping the A side to cp.async does not touch the B side's structure, so when the B side later becomes a pure copy (awaiting W_exp) the two naturally fit together.

## 3. Implementation

### 3.1 Design choices: A-side solo + group-count waits, budget untouched

The change is deliberately narrow: only the qa8/sda two segments of the A side inside the `RAW_STAGE_Q6K_BT` macro and the main loop's wait structure; the B-side recomb (r41 form), the KDR=2 double buffering, and the 3-block residency budget are all untouched. The wait uses group counting (`wait1` in-loop, `wait0` in the last round) rather than `wait_group 0` every round — the latter regresses to synchronous semantics and the pipeline would be for nothing. r39-era buffer WAR barrier (consumers must finish reading before overwrite) is kept: cp.async only changes "which execution unit writes smem" and grants no exemption from write-after-read hazards.

Since the code was ultimately reverted, the diff's coverage is recorded here as narration (the mechanism skeleton is visible verbatim in the current tree's r53/r56 landed versions, see §3.2): (1) four new device functions `gemm_cp16`/`gemm_cp_commit`/`gemm_cp_wait1`/`gemm_cp_wait0` (inside the Ampere guard); (2) the qa8/sda segments of the `RAW_STAGE_Q6K_BT` macro changed from `reinterpret_cast<const uint4*>` loads + STS.128 to per-16 B-chunk `gemm_cp16` (B side untouched); (3) the main loop changed the double-buffer switch point from "stage, then synchronously wait" to "stage, then group-count wait". The rejected alternatives: `wait_group 0` every round (back to synchronous); cp.async on the B side too (impossible — the recomb is per-element ALU with no copyable contiguous-16 B semantics); the `__pipeline_memcpy_async` intrinsic (a pitfall, see §3.3#1).

### 3.2 Key code

**The explicit PTX copy primitives (current tree `src/cuda_kernels.cu` 4483–4491, introduced in r45 and in use since)**:

```cuda
__device__ __forceinline__ void gemm_cp16(__half* smem_dst, const __half* gsrc, bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 16 : 0; // src-size 0 => zero-fill the 16B chunk
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}
__device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void gemm_cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void gemm_cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
```

Three details: `__cvta_generic_to_shared` converts a generic pointer into a 32-bit shared-window address (required by PTX's shared state space); the 4th operand is the **src-size** qualifier, and at `sz=0` the hardware zero-fills the whole 16 B chunk (r53 uses this for tail rows beyond od); the `.cg` qualifier goes through L2 bypassing L1 (staging data is consumed once, so L1 caching is meaningless).

**The A-side staging loop (current tree, inside the RAW_STAGE macro; the comment signs r45)**:

```cuda
/* ---- A: r56 cp.async bulk staging of the pre-transposed qa8/sda --*/
/* (r45's mechanism on top of r53: the sync LDG->STS exposed its      */
/*  global latency at the top of every staging phase; cp.async hands  */
/*  it to the async unit and the group wait below hides it under the  */
/*  previous tile's compute. Bytes identical - the plane is always    */
/*  full: the prepass zero-fills the padded rows.)                    */
{
    const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16; off += blockDim.x)
        gemm_cp16((__half*)(void*)(qa8b + (size_t)off * 16),
                  (const __half*)(const void*)(qa8g + qbase + (size_t)off * 16),
                  true);
    const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16; off += blockDim.x)
        gemm_cp16((__half*)(void*)(sdaqb + (size_t)off * 16),
                  (const __half*)(const void*)(sdag + sbase + (size_t)off * 16),
                  true);
}
```

**The group-count pipeline main loop (current tree 6905–6924)**:

```cuda
RAW_STAGE_Q6K_BT(0, 0);          // tile 0 synchronously preloaded (group already committed)
__syncthreads();

int buf = 0;
for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
    // stage kt+1 into the OTHER buffer while reading buffer buf
    if (kt + 1 < nktile) {
        RAW_STAGE_Q6K_BT(kt + 1, buf ^ 1);
        // two groups pending (kt's and kt+1's); wait until only kt+1's
        // remains — group(kt) has landed, buf^1's copies stay in flight
        gemm_cp_wait1();
    } else {
        gemm_cp_wait0();  // last tile: drain every outstanding group
    }
    __syncthreads();  // cross-thread visibility of the kt buffer's async copies
    // ... kt's mma compute consumes buffer buf; buffer WAR is guaranteed by the existing barrier ...
}
```

The only difference from r45's shape at the time: the B side was still r41's recomb (the `EXP=false` branch), dsc was still a scalar read, and the commit sat at the end of RAW_STAGE — the mechanism skeleton (group counting + the visibility barrier + WAR preservation) matches the above word for word.

The assembly relationship of the three excerpts: `gemm_cp16`/`commit`/`wait` are the primitive layer (16 B semantics + group accounting); the RAW_STAGE macro's A segment is the issue layer of "whole numbers of chunks per thread" — it returns as soon as it issues, never waiting for data; the main loop is the scheduling layer — using group counts to express "wait for whose copies before consuming whom". All three layers are independently testable (SASS counts, byte equivalence inside the macro, the loop invariant), and r45's verification gates were built along exactly these three layers.

### 3.3 Pitfalls

1. **`__pipeline_memcpy_async` silently falls back.** The first version used the CUDA intrinsic: the compiler could not prove a generic `uint8_t*` (byte pointer) is 16B-aligned, so it **did not emit cp.async and silently fell back to LDG+STS** — it compiled, was semantically correct, and performed identically, the only tell being 0 copy instructions in the SASS. A textbook case of "intrinsic ≠ hardware mechanism": a mechanism must be accepted via SASS, not via compiling. Only the explicit PTX (copy size 16 as a compile-time constant) produced LDGSTS.
2. **The SASS mnemonic trap: cp.async is `LDGSTS.E.BYPASS.128`.** Grep for `CP.ASYNC` and you conclude wrongly that nothing was issued. The gate is always `cuobjdump -sass | grep LDGSTS` (24 hits in the `<2>` instantiation). This entered the 77-verification-methodology forensics checklist.
3. **An unexpected dividend in the register account.** Once cp.async removed the register way-station, `REG 80 / LOCAL 0` — the 4 B spill that r40-era's "10 hand-trim variants could not squeeze out" was gone. Register pressure determined by the staging structure is harder than any register-level micro-tuning.
4. **The visibility barrier can be neither omitted nor duplicated.** The `__syncthreads()` after each round's wait is a cross-thread visibility requirement (cp.async completion is per-thread semantics); the barrier the buffer WAR depends on is a different one — the two barriers have different duties, and merging or deleting either produces intermittent dirty data.
5. **The N in `wait_group N` is "the number of groups allowed to remain unfinished", not "wait for the first N groups".** `wait_group 1` = wait until at most 1 group is outstanding (i.e. everything before the second-to-last has landed); reading the semantics backwards turns the pipeline half-synchronous or leaves dangling reads. The correctness argument for group-count code must land on §2.2's trace table, not on intuition.

## 4. Verification

| Gate | Reading | What it defends against |
|---|---|---|
| SASS gate: `cuobjdump -sass` grep `LDGSTS` | **24 hits** in the `<2>` instantiation | "the intrinsic silently fell back" — confirms the physical mechanism is really running rather than the compiler downgrading; this doc's most important gate |
| parity `cuda_prefill_mmq` (GPU vs CPU reference) | **1/0** | data misalignment introduced by swapping the staging mechanism (the copied bytes are bit-equivalent to LDG+STS: the planes are always full, no partial chunks) |
| greedy-32 byte comparison | byte-identical | any drift on the numeric path propagating into sampling |
| suite | 166/0/3 | the whole-engine regression net |
| ptxas registers/spill | REG 80 / LOCAL 0 | the 3-block residency budget (r40's +13% depends on it), plus checking cp.async's expected register dividend |
| ncu base-vs-r45 (interleaved, same window) | Duration −10.2%, longsb absolute −18.4%, Compute(SM) 37.42→42.13% | three-line self-consistency at the mechanism layer: faster, fewer stall-waits, higher compute share — the exclusion of "just measurement noise" |
| whole-prefill interleaved 3× | 2604.5 → 2595.6 (−0.34%) | machine drift faking a trend; the verdict "wall-neutral, bar not met" |

Together, ncu's mechanism-layer readings are the complete signature of cp.async working: Elapsed Cycles 1,408,834 → 1,261,636 (−10.4%) and Warp-Cycles/Issued-Inst 11.55 → 10.60 improving in the same direction — the contrast with r44's "cycles fall but per-instruction stall rises" (latency changing form): cp.async really did hide the wait inside compute, not just reshuffle the instruction mix.

## 5. Results

| Layer | before (r41/r44 baseline) | after (r45) | Verdict |
|---|---|---|---|
| SASS | 0 cp.async (intrinsic fallback) | LDGSTS ×24 | mechanism confirmed |
| kernel Duration | 659,680 ns | 592,448 ns (−10.2%) | mechanism took effect |
| long_scoreboard (absolute) | 4.38 cy | 3.57 cy (−18.4%) | the wait really was hidden |
| Compute(SM) | 37.42% | 42.13% | compute share rose |
| registers/spill | 80 / 4 B | 80 / 0 B | unexpected dividend |
| whole-prefill (interleaved 3×) | 2604.5 tok/s | 2595.6 (−0.34%) | **wall-neutral, bar not met** |

Measurement protocol as in r44: same session window, the same pair of binaries (base / r45) interleaved 3 rounds of whole-prefill taking the median; ncu is matched-nt sampling at the same shape. This window's ~2600 tok/s absolute value is meaningful only within this window. vs-llama stays at 1.27×.

**Veto mechanism (why reverted)**: the +1.5% bar was not met, and −0.34% has no directionality even. Merged with r44 into a complete convergence proof: **r44 deleted work (the recomb ALU, −10.9% cycles) and the wall did not move; r45 hid waits (cp.async, −10.2% duration, −18% longsb) and the wall did not move either** — the two biggest stall sources (45% + 28%) were each handled by a correct mechanism, and whole-prefill did not move an inch. Only one conclusion remains: after r41 the q6_K GEMM is not the wall, and the remaining ~3.0× vs-llama gap of the matched-nt kernel (against 57.8 µs/GMAC) no longer drives whole-prefill. The q6_K line declares **CONVERGED**; further kernel-level tuning (cp.async B, split-phase, pre-expansion) is expected wall-neutral in solo form. Reverted by discipline (cmp-match HEAD).

The immediate consequence of the convergence declaration is worth recording: it is not a negative "we're done" but an authorization for budget migration. r46 (23:00 that day) turned to audit FA, and r47 redid the converged-domain wall decomposition the next day (q6_K 1094.7 → 196.4 ms, falling from 51.2% to 15.8%; FA rose to the #1 structural residual) — both rounds' project proposals cite r44/r45's dual-mechanism convergence evidence directly. Had these two rounds' wall-neutrality been vaguely recorded as "the attempts didn't work", the later budget allocation would have had no basis; precisely because the veto mechanism spelled out "deleting work didn't work + hiding waits didn't work ⇒ the slice is not the wall", the conclusion could be safely extrapolated.

**Under what future conditions a retry is worthwhile**: when the A-side wait becomes the critical path again. r56 cashed this in precisely: after r53's package (W_exp pure copy + cp.async B staging, +5.03%) landed, the wall's composition moved — "A-side staging STS + dsc consumption" became the q6_K line's remaining bulk, and r56 installed r45's mechanism back on the A side verbatim. r56's readings:

| Layer | after r53 → r56 | Verdict |
|---|---|---|
| ffn_down kernel (ncu) | 12.76 → 12.01 ms (−5.9%) | the A-side cp.async took effect |
| attn_v kernel | −4.2% | same |
| whole-prefill | 3138.6 → 3212.5 (**+2.35%**, distributions separated) | r45's mechanism touched the wall for the first time |
| SASS | both instantiations (`<2,true>` and `<2,false>`) contain LDGSTS | the mechanism fully in place |

r45's own words were "**Not dead, WAITING**", and r53/r56 proved the second half. Together with doc 47, this doc forms the campaign's most important pair of counter-intuitive samples: a mechanism's value is not a property of the mechanism; it is a property of the wall.

## 6. Lessons

1. **A mechanism can be correct, confirmed, and simultaneously worthless** — wall-clock value depends on what remains on the critical path, not on how elegant the mechanism is; a vetoed mechanism must enter the record carrying its "when to retry" conditions.
2. **An intrinsic is not a mechanism**: the lesson of `__pipeline_memcpy_async` silently falling back to LDG+STS is that any "I used hardware feature X" claim must carry SASS-level evidence.
3. **Grep SASS with the real mnemonics**: cp.async is called `LDGSTS.E.BYPASS.128` in SASS — get the search keyword wrong and every conclusion is wrong.
4. **Pin a convergence verdict with two independent mechanism lines**: deleting work does not move the wall + hiding waits does not move the wall is far stronger than a single piece of evidence; only then is the post-convergence budget migration (q6_K → FA/prepass) justified.

← 47-r44-wexp-stride-mismatch · [Index](./README.md) · 49-r46-fap1-fa-audit →
