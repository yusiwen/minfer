# 43 · r40 — `__launch_bounds__(256,3)` third resident block (LANDED)

> **Result**: 7B whole-prefill 1784.0 → 2015.6 tok/s (+13.0%, 3/3 interleaved pairs + one independent pair); attn_v q6_K kernel 2.05 → 1.58 ms (−23%).
> **Commit**: `65ecef7`. **Date**: 2026-09-05.

## 1. Background — where things stood

r39's double buffering freed the q6_K kernel from "serial staging" (+13.3%), but the kernel was still latency-bound: compute share 21.5%, No-Eligible (the share of cycles with no eligible warp at the issue slots) 74.5%, active warps/sched 3.87. The SM was waiting on latency, and with only 2 resident blocks on the SM = 16 schedulable warps, the supply of latency cover was capped by occupancy.

Of occupancy's three constraints, the q6_K kernel's state after r39 landed was:

- **smem**: 29,696 B/block × 3 = 89,088 B < 102,400 B (GB10's per-SM opt-in ceiling) — **smem had allowed a 3rd block all along**;
- **registers**: r39 recorded 87 regs/0 spill. 87 × 768 threads (3 blocks × 256) = 66,816 > 65,536 (the per-SM register file) — **registers pinned residency at 2**;
- **block size/other**: 256 threads × 3 = 768 ≤ the per-SM thread ceiling; no obstacle.

Conclusion: a pure register-budget problem. Squeeze per-thread registers from 87 to within 80 and the 3rd block fits. The audit of the three constraints:

| Constraint | 2 blocks/SM (status quo) | 3 blocks/SM (target) | Verdict |
|---|---|---|---|
| smem | 29,696 × 2 = 59,392 B | 29,696 × 3 = 89,088 < 102,400 | allowed long ago |
| Registers | 87 × 512 = 44,544 ≤ 65,536 | 87 × 768 = 66,816 > 65,536 | **the sole veto** |
| Threads | 512 ≤ per-SM ceiling | 768 ≤ per-SM ceiling | allowed |

The occupancy ladder is one this campaign has climbed repeatedly: r28 used smem (45,056 B) to buy q4_K 2 blocks/SM (+2.56%); r39 cashed in q6_K's latency hiding at 2 blocks/SM; r40 is registers' turn — the same scale, weighed a third time.

The question "can a 3rd block be reached by trimming regs across the 85.33 line" had in fact been on record since r38 landed (the tail of r38's commit message carried the todo "whether a 3rd block is reachable by trimming regs < 85"); r39's record promoted it to the next lever (r40's commit message verbatim: "Adds the r39-named lever"); after r39 every condition was ripe, and r40 cashed it in with **one hint line**. Read as three points in time, this is the documentation system working normally: **phenomenon (85 regs blocks residency) → on record (r38 todo) → promoted to lever (r39) → cashed in (r40)** — skipped steps usually happen when the phenomenon never got written down.

The only obstacle was psychological: squeezing to 80 regs means spilling (registers overflowing to local memory), and "0 spill" had been the admission convention for new kernels throughout the campaign (r28 123 regs/0 spill, r38 85 regs/0 spill, r39 87 regs/0 spill — every doc's ptxas audit treated 0 spill as a green metric). r40's real work was not writing that line of code; it was proving experimentally that **4 B of spill is immaterial**, demoting the convention from "rule" to "heuristic".

The contrast in cost structure is also worth recording: **the implementation is a one-time single line; the forensics is ten variant builds plus a round of ncu**. For this class of "one-line lever" in the campaign, the doc is usually a hundred times the size of the code — because all the transferable knowledge is in the evidence chain, not in that line of code.

## 2. Principle — the GPU mechanism

### 2.1 The register arithmetic: where 80 comes from

The second parameter of `__launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)` is a contract handed to ptxas: **guarantee at least 3 blocks of 256 threads can be resident simultaneously**. ptxas back-derives the per-thread register ceiling from it:

```
per-SM register file        = 65,536 32-bit registers
3 blocks × 256 threads      = 768 threads
65,536 / 768                = 85.33 → rounded down to the 8-registers-per-thread allocation granularity → ceiling 80
```

87 regs × 768 = 66,816 exceeds the register file, so ptxas's natural allocation only allows 2 blocks; capped at 80, the 1 extra live value (ptxas actually needs 81) has to spill. **4 B spill = 1 32-bit value** going to local memory (thread-local, backed by L1/L2), adding one store/load pair per pass through the staging path.

Both directions must be checked for the occupancy verdict to stand: at 2 blocks, 87 × 512 = 44,544 is far within 65,536 — 87 regs is fine in itself; the problem only appears in the 768-thread multiplication. The allocation granularity of 8 comes from an implementation detail: ptxas allocates registers in whole warp (32-thread) segments in units of 8/thread; the division result 85.33 can never be granted, and the nearest grantable step is 80.

smem side-check: 29,696 × 3 = 89,088 ≤ 102,400; smem does not block. So the launch occupancy limit moves from (regs=2, smem=3) to (regs=3, smem=3) — the first time both limiters read 3 together.

**The raw evidence form of ptxas `-Xptxas -v`**: compile with `-Xptxas -v` and ptxas prints one resource line per entry. The corresponding lines from the two builds (output format verbatim, numbers from the record):

```
ptxas info : Compiling entry function '...mmq_raw_nb_bt_q6k_kernel<2>'
ptxas info : Used 87 registers, 0 bytes spill stores, 0 bytes spill loads      // r39 build
ptxas info : Used 80 registers, 4 bytes spill stores, 4 bytes spill loads      // r40 build (after the hint)
```

"4 bytes spill stores/loads" is exactly that 1 32-bit live value's store/load pair, 4 B each — matching §2.1's "one extra store/load pair" word for word. That line is the entire compile-time consequence of r40's entire code change.

### 2.2 Why 4 B of spill is immaterial here

The default judgment "spill is a loss" comes from compute-bound kernel intuition: spill's local accesses insert into the hot loop and steal issue slots. But this kernel's measured state is **No-Eligible 74.5%** (ncu issue statistics: the share of issue-eligible cycles in which no warp is eligible, i.e. the idle rate of an SM that "wants to issue but has nothing to issue") — for three quarters of the cycles the SM wants to issue and has no warp to issue; that is insufficient latency cover, not insufficient issue bandwidth. The two sides of the scale:

- **Gain side**: resident warps 16 → 24 (theoretical +50%). The total schedulable latency tolerance grows linearly with resident warps, and r39 had just left a large amount of LDG/expansion latency sitting in the pipeline, which 8 more warps are exactly positioned to absorb.
- **Cost side**: 1 spill slot = one pair of 4 B local accesses per iteration on the staging path, L1-hit territory, and thoroughly buried under the parallelism of 24 warps.

The scale's verdict rests on measurement, not reasoning: +13.0% wall, kernel −23%. **The "0 spill gate" is empirically falsified here** — it defends against "a spill avalanche caused by runaway register pressure", not "any spill is guilty".

To make the gain-side mechanism explicit: a latency-bound kernel eats occupancy because **memory requests in flight = resident warps × pending loads per warp**. r39's double buffering made each warp pend more loads (the expanded-B LDG stream), but only 16 warps on the SM were available to cushion that latency; the 3rd block raises the latency-cushioning warps to 24, which is what moves No-Eligible from 74.5% to 70.1%. The cost-side spill goes to local memory: on first spill ptxas allocates a fixed per-thread local slot at launch, after which it is just an `STL`/`LDL` pair for that 1 live value — 4 B, L1-hit territory, and located on the staging path (already the latency-dominated segment, where one more access pair is fully absorbed by the parallelism of 24 warps).

Reading the issue-slot economics: No-Eligible 70% means that of every 10 cycles, only about 3 per SM have a warp eligible to issue. The two roads to more throughput are (a) make each issue do more (ILP/wider loads — r41's route) or (b) reduce the no-warp-eligible cycles (more resident warps — this doc's route). The two roads have different gain ceilings: this doc lowered No-Eligible by only 4.4 percentage points yet took +13.0% — because the added issues concentrate in the stall segments of the critical path; that disproportion is a typical reading for a latency-dominated kernel.

### 2.3 The relay with r39/r41

r39 raised compute share to 21.5% but No-Eligible stayed 74.5%; r40 lowers No-Eligible to 70.1% and raises compute to 29.06% — once the occupancy line's gain cashed in, the residual bottleneck is the L1TEX scoreboard (measured at the r40 point: 10.5 cy/warp, ~70% of warp time; r41 attributes it precisely to 32 per-byte `LDG.E.U8` and presses it further to 3.6 cy). The occupancy and load-width lines each own a segment and do not substitute for each other.

### 2.4 Why stop at 3: the 4th block's ceiling

r40 also ruled out the next step while it was there. 4 blocks/SM requires: smem 29,696 × 4 = 118,784 B > 102,400 B — **smem vetoes outright**; even with enough smem, the register ceiling would drop to 65,536/1024 = 64/thread (64 at the granularity of 8), a further 16 down from 80, and the spill surface would roll from 1 slot into a sheet. So 3 is this kernel's natural residency endpoint on GB10 — r40 got there in one step, with no "squeeze one more block" tail.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

1. **Hint, not hand-trimming**: let ptxas decide at the 80-cap which value to give up (it has whole-kernel live-range data); the human sets the contract and does not meddle in the allocation. Measured: ptxas's choice was 1 4 B spill.
2. **Sweep hand-trimmed variants before accepting spill**: to confirm 4 B was not the lazy option, a Task-2 10-variant register-trimming sweep was run (listed verbatim in the commit message: G-recomputed, smem-base fold, de-unroll, assume, div→shift, per-g A-frag, epilogue recompute, fused mma+scale, etc.) — **all landed at 80 regs/4 B spill or worse**. Conclusion: this ptxas needs 81 live registers, and the 4 B spill is a hard floor, not a matter of effort.
3. **Don't touch KDR/double buffering/dispatch gate**: r40 is fully orthogonal to r38/r39's mechanisms — a one-line change, output-neutral.

### 3.2 Key code

The entire code change is one line in `src/cuda_kernels.cu` (the complete code diff of `65ecef7`):

```cuda
 template <int KDR>
-__global__ void __launch_bounds__(256) mmq_raw_nb_bt_q6k_kernel(
+__global__ void __launch_bounds__(256, 3) mmq_raw_nb_bt_q6k_kernel(
     const uint8_t* __restrict__ W, const uint8_t* __restrict__ qa8g,
     const uint8_t* __restrict__ sdag, float* __restrict__ C,
     int nt, int od, int id, int nchunk, int bstride
```

`__launch_bounds__`'s first parameter (256) locks the thread count; the second (3) is the minimum-resident-blocks contract. The current tree (`cuda_kernels.cu:6701`) keeps this line as-is, and r53 additionally gave the template an `EXP` parameter (`<KDR, bool>`); the hint never changed again.

**ptxas `-Xptxas -v` forensics** is this doc's "implementation" centerpiece. Compile with `-Xptxas -v`, read each kernel's register/spill report line; the key numbers of the two builds:

| Build | regs/thread | spill | Residency ceiling (regs side) |
|---|---|---|---|
| r39 (no hint) | 87 | 0 | 2 blocks (87×768 > 65,536) |
| r40 (hint (256,3)) | 80 | 1 slot = 4 B | 3 blocks (80×768 = 61,440 ≤ 65,536) |

The same-table audit of the 10 hand-trimmed variants all came out ≥4 B spill — the evidence chain for the 4 B floor is complete (numbers from the r40 commit message and master row 54). The variants named in the commit message and each one's register-saving hypothesis:

| Variant | Hypothesis | Result |
|---|---|---|
| G-recomputed | recompute the 4 ldmatrix offsets each time, saving `G[4]` | 80/4B |
| smem-base fold | fold the smem base-address arithmetic | 80/4B |
| de-unroll | un-roll the kd `#pragma unroll` to shorten live ranges | 80/4B (or worse) |
| assume | alignment/aliasing assumptions to help ptxas narrow | 80/4B |
| div→shift | replace division with shifts to save temporaries | 80/4B |
| per-g A-frag | shrink the A-fragment register footprint | 80/4B |
| epilogue recompute | recompute epilogue addresses | 80/4B |
| fused mma+scale | fuse mma and the rescale path | 80/4B |

(The sweep was 10 variants in total; the table lists the 8 named in the commit message.) Conclusion: at the 80-cap, this ptxas needs 81 live registers no matter how things are arranged — the 4 B spill is a **hard floor**, not a matter of effort. accept-the-spill went from compromise to evidence-backed decision.

The hint's survival through later evolution is also on record: after the current tree's r53 added the `EXP` boolean to the template, instantiations remained `mmq_raw_nb_bt_q6k_kernel<KDR, true/false>` and the prewarm lines are `mmq_raw_nb_bt_q6k_kernel<2, true>` / `<2, false>` (`cuda_kernels.cu` lines 7140-7141); `__launch_bounds__(256, 3)` was never reverted. A forward rule worth setting down: **launch_bounds is a per-instantiation contract** — every time the template gains an instantiation (`<KDR, bool>` is one → two), the `-Xptxas -v` audit must be rerun for that instance; the hint only constrains ptxas's allocation target and does not guarantee a new instance can still reach 80 regs at 0/small spill. When r53 landed, this was exactly one of the items needing reconfirmation beyond "r41's uint4 widen held 80/4B".

**The ncu Occupancy section's residency evidence** (two launch-limit lines from r40's verification record; §5 has the achieved values):

```
Block Limit Registers                3      ← 80 regs × 768 threads = 61,440 ≤ 65,536
Block Limit Shared Mem               3      ← 29,696 × 3 = 89,088 ≤ 102,400
Block Limit Warps                    6      ← 48-warp ceiling / 8 warps per block
```

The minimum of the three Block Limits sets the residency ceiling: before, min(2, 3, 6) = 2; after, min(3, 3, 6) = 3 — registers went from "single veto" to a joint decider tied with smem, which is precisely the hint's semantic goal.

### 3.3 Pitfalls

1. **The inertia of the 0-spill convention**: three consecutive steps (r28/r38/r39) recorded 0 spill as green in their audit tables, nearly mistaking a heuristic for an admission rule. Half of r40's contribution is the number; the other half is writing down this convention's scope of validity.
2. **The register granularity trap**: 65,536/768 = 85.33 — plan for 85 and ptxas still cannot grant it; the allocation granularity is 8, so the effective ceiling is 80. When planning occupancy, round down to the granularity, not to the division result.
3. **smem allows ≠ can be resident**: when r39 landed, the smem side already fit 3 blocks (89,088 < 102,400), but the launch occupancy limit's regs=2 cast the single veto. Check occupancy bottlenecks item by item; never infer from "smem unchanged".

## 4. Verification

- **parity 1/0**: logits deviation within the gate (the campaign's parity gate: run the same prompt against the baseline binary and compare max |Δlogits| ≤ 1e-3) — a register hint touches no arithmetic path (defends against numerical regressions introduced by collateral changes like "touched code while squeezing registers").
- **greedy-32 byte-identical**: the commit message states "register hint is output-neutral" — the same arithmetic with only the resource allocation changed (defends against argmax knife-edge flips).
- **ptxas resource audit**: 80 regs/1×4 B spill, and the launch occupancy limit reads regs=3, smem=3 — direct evidence the hint took effect (defends against "hint written but ptxas ignored it").
- **ncu occupancy recheck**: 3 blocks/SM confirmed — achieved 18.12 warps/SM = 37.74% (against the 48-warp ceiling; theoretical 24 = 50%), active warps/sched 3.87 → 4.48 (defends against "3 blocks in theory but never 3 in practice"). The gap between achieved 18.12 and theoretical 24 is the normal loss of scheduling tails and divergence (wave tails, uneven block progress), not evidence the 3rd block failed to be resident — the launch limit and active warps counters corroborate each other: **the ceiling really reached 3, and the average activation count rose with it from the 16 magnitude to 18+** (under the old shape the average cannot pass the theoretical ceiling of 16).
- **A/B interleaved 3/3 + one independent pair**: same-window, same-binary pairs, all positive (defends against machine-drift fake deltas).
- **suite 166/0/3**: full regression (defends against collateral damage to others).

## 5. Results

| Metric | before → after | Note |
|---|---|---|
| whole-prefill (7B, same-window A/B median) | 1784.0 → 2015.6 tok/s (**+13.0%**, 3/3 + one independent pair) | master row 54 |
| attn_v q6_K kernel | 2.05 → 1.58 ms (**−23%**) | the half-more of residency cashed in |
| compute share | 21.52% → 29.06% | issue slots keep backfilling |
| No-Eligible | 74.5% → 70.1% | latency cover improved but still the main bottleneck |
| active warps/sched | 3.87 → 4.48 | ncu scheduler view |
| occupancy (achieved) | 18.12 warps/SM = 37.74% (3 blocks confirmed) | theoretical 24 warps = 50% |
| matched-nt q6_K | ~221.8 → ~171 µs/GMAC (est.) | still ~3.0× vs llama's 57.8 |
| vs llama.cpp (3325-eq anchor) | 1.87× → 1.65× | the engine got faster, so the relative multiple falls |

Three readings:

1. **One line for 13%**: the change is one constant in a template parameter position; the gain comes from §2.2's scale — +50% theoretical resident warps against 1 4 B spill slot. Occupancy-class levers can have a very high price/performance ratio on latency-dominated kernels.
2. **Three consecutive steps compounding**: r38 (new kernel, +2.87%) → r39 (overlap, +13.3%) → r40 (residency, +13.0%), the q6_K line pressed from 368.9 µs/GMAC down to ~171 and whole-prefill pushed from the 1518 band past 2000. Each step's "Results" section hands the next step its state (the 16.7% → 21.5% compute curve, No-Eligible 74.5%). The kernel −23% vs wall +13.0% ratio is also self-consistent: in r37's attribution the q6_K GEMM was about half the wall clock, so cutting the wall-critical segment 23% amortizes to +11–13% over the whole wall — the two layers of numbers cross-check each other.
3. **The residual is named**: No-Eligible still 70.1%, L1TEX scoreboard 10.5 cy/warp ≈ 70% of warp time — the next lever is load width (r41's uint4 B-expand cuts that number to 3.6 cy and whole-prefill gains another +30.7%), not residency (3 blocks reached; no cheap smem/register headroom remains).
4. **Unit-cost trajectory**: matched-nt q6_K ~171 µs/GMAC, still ~3.0× vs llama's 57.8 (r40 commit message verbatim), but the internal decomposition has changed — the commit message also records "the residual is KSPLIT=2's intrinsic 2x mma.k16 plus memory-latency stall, not residency": the residency line is exhausted, and half the remaining gap is the doubled mma structural cost q6_K's 16-sub-block layout imposes, half is load latency. r41 attacks the latter.

**Baseline drift note**: r39 landed reporting 1777.5; this doc's in-window baseline is 1784.0 — absolute values from different session windows of the same code are not comparable (§0 table-reading convention); deltas are always taken from same-window A/B (+13.0%, 3/3 plus one independent pair), never subtracted across windows.

**q6_K line postscript**: this doc is the third step of the four-step arc (r38 skeleton → r39 overlap → **r40 residency** → r41 load width); after r41 the q6_K line had 1.27× left vs llama, and the line closed in the r45–r53 cp.async bundle. This doc's `__launch_bounds__(256, 3)` is the only kernel parameter in the four steps never touched again — once the residency contract is set right, all later evolution (template parameters, cp.async, pre-expanded planes) happens inside its resource frame.

## 6. Lessons

1. **0 spill is a heuristic, not a law**: measure spill's actual cost first, then decide how much register pressure to pay to avoid it; on a latency-bound kernel, a 4 B spill slot is no match for +50% resident warps.
2. **Check occupancy's three constraints item by item**: smem, registers (capped after rounding up to the 8 granularity), and thread count each veto independently; "smem fits" does not imply "the block gets in".
3. **Let ptxas be the allocator and the human the auditor**: hint sets the contract, -Xptxas -v produces the evidence, the variant sweep sets the floor — the correct division of labor for register trimming.
4. **Before buying residency, confirm the kernel really is latency-dominated**: the No-Eligible/active-warps counters are the grounds for buying occupancy; on a kernel whose compute is already saturated, buying residency only dilutes each warp's smem/L1 quota.

---
[← 42 · r39 q6_K KDR=2 double-buffer](42-r39-q6k-kdr2-double-buffer.md) · [Index](./README.md) · [44 · r41 q6_K B-expand uint4 widen](44-r41-q6k-bexpand-uint4.md) →
