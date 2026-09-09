# 22 · r17 — Wide warp remap 32od × 64tok (REVERTED)

> **Result**: instruction stream **8.68 → 8.18 M inst/GMAC (−5.8%, LDSM.x4 72 → 48/block-chunk)**, but wall clock **KD=4 1250 vs 1239 / KD=8 1301 vs 1288 (+0.9%/+1.0%, noise band)** — SM% 31.5 → 29.75 and SM Active Cycles +9.2%: the drop in issue efficiency ate the entire instruction reduction. **Third independent confirmation: in the SM% ~30 stall-bound regime, pure per-MAC instruction reduction buys ~0 wall clock.**
> **Commit**: `9d09a81` (record only; kernel code reverted to `708067d` and cmp-verified, the remap variant lived only in `/tmp` and was never committed). **Date**: 2026-09-03.

## 1. Background — where things stood

r15 had landed the rank-1 fold that very morning (wide kernel 1295 tok/s @KD=8, 8.69 M inst/GMAC), and its record pointed the next lever at two directions: **warp-tile shape (2× od-rows per warp)** or prefetch/stall-structure work. r17 chose the former — the next stop on the "per-MAC instruction efficiency" main line that r13 (counter forensics) and r15 had both pointed to.

The chain of reasons for picking warp shape runs like this:

1. **r9 had noted a shape difference while decoding the llama.cpp reference**: llama's Q4_K MMQ config is 256 threads, tile I=128 × J≤128, 16 mma chains per warp per chunk — the chain count matches ours (post-r12), but **the warp's split differs**: llama gives each warp **32 od-rows × 64 tokens**, ours is **16 od-rows × 128 tokens**.
2. **r13's ledger**: ours 10.14 → 8.69 M inst/GMAC after r15, llama 6.06 M — still 2.6 M apart. Within the instruction stream, the A/B fragments' smem reads (LDSM) are one of the biggest items, and the LDSM count is set directly by warp shape (how many chains reuse each fragment).
3. **r14's precedent**: routing B fragments through a single ldmatrix.x4 (fewer smem ops) had bought +18.5–30% — "fewer/wider smem ops" was a proven paying lever class at the time, and warp remapping is another entrance into that same lever class.

**Hypothesis**: change the warp shape to llama's 32od × 64tok; the A-fragment group count drops from 8 to 4 (each warp covers only 64 tokens) so A-side LDSM halves; B-side rises to 2 ldmatrix.x4 covering 32 od-rows; total LDSM per block-chunk falls from 72 to 48 (−33%), and by r13's 1:1 tracking law this should buy back a few percent of wall clock. The session's pre-registered bar: **KD=8 ≥ 1350 tok/s** (baseline 1288–1295, i.e. requiring +4% or more).

Two background readings in the execution environment were, in hindsight, both warnings: the wide kernel at KD=8 has 98,304 B of smem (single-buffer synchronous staging) — **1 block/SM**, with latency hiding resting entirely on resident-warp depth rather than prefetch; and the issue-efficiency gap vs llama had been measured in r20's matched-nt comparison: **0.26 vs 0.42 issue/sched (same occupancy)** — 3 of every 4 issue slots waiting. The instruction stream was a "thin yet fat" problem, but the binding constraint was the waiting.

In hindsight, the hypothesis erred by applying r13's tracking law outside its domain of validity — r13 itself had written the qualifier in its lessons: "per-MAC instruction count is the first-order predictor **but see r25: only while issue is stall-bound**" (r25 had not happened yet, but the SM% 31.5 reading already had the premise sitting right there). r17's value lies in nailing down that boundary with one clean full-remap experiment.

## 2. Principle — the GPU mechanism

### 2.1 The ledger of the two warp shapes

| | Before (r12–r16 shape) | After (llama-shape remap) |
|---|---|---|
| warp owns | 128 tok × 16 od | **64 tok × 32 od** |
| warp grid | 8 warps each in one 16-od slot (`j0w = warp*16`) | `wn=warp&1 → i0w=wn*64`, `wm=warp>>1 → j0w=wm*32` |
| chain grid (per chunk per thread) | 8 A-frags × 2 B-frags = 16 chains | 4 A-frags × 4 B-frags = 16 chains |
| `sum[]` budget | 16 chains × 4 C regs = 64 floats (unchanged) | 64 floats (unchanged) |
| LDSM per thread per chunk | A 8 + B 1 = **9** | A 4 + B 2 = **6** |
| LDSM.x4 per block-chunk | 8 warps × 9 = **72** | 8 warps × 6 = **48** |

The unchanged chain count is the clever part of this remap: 4×4 and 8×2 are both 16 independent mma chains, so the register budget (`sum[64]`), staging, qb8/sds smem layout, and launcher all stay untouched — **a pure index rearrangement**.

### 2.2 The ignored side: B-fragment amortization halves

The LDSM ledger saves "read counts", but the reuse rate changes:

| Reuse structure (per chunk, per warp) | Before 8×2 grid | After 4×4 grid |
|---|---|---|
| chains reusing each B fragment | **8** | **4** (amortization halves) |
| chains reusing each A fragment | 2 | 4 (amortization doubles) |
| block-level A-side LDSM | 8 warps × 8 = 64 | 8 warps × 4 = 32 |
| block-level B-side LDSM | 8 warps × 1 = 8 | 8 warps × 2 = 16 (and the same 32-od span is read redundantly once each by the two warps wn=0/1) |
| total | **72** | **48** |

- **Before**: each B fragment (1 LDSM.x4) is reused by **8** A chains; each A fragment by 2 B chains.
- **After**: each B fragment is reused by only **4** A chains; each A fragment by 4 B chains.

More hidden is the block-level redundancy: after the remap the same 32-od span is ldmatrix'd once each by **two warps** (`wn=0/1`) (B-side LDSM per block rises from 8 to 16), while A-side drops from 64 to 32 — the total does go 72 → 48, but the saving is on the **better-amortized side** (A) and the addition on the **worse-amortized side** (B). The commit message's mechanism sentence "the per-warp B-side went 2x-shared" says exactly this: B fragments' per-read amortization fell from 8 chains to 4.

### 2.3 Why −5.8% instructions cannot buy back wall clock: the stall-bound accounting

r13's tracking law (duration ≈ instruction count / 0.10–0.15 warp-inst/ns) carries an implicit premise: **issue is the constraint**. The kernel's true state at this point is SpeedOfLight Compute (SM) ≈ 31.5% — only three-tenths of each SM's issue slots are doing work, the rest are waiting (r20 later localized it: waiting on the A-staging LDG→STS chain's long_scoreboard, a ~600-cycle full-batch stall per 4-deep batch). In a regime where issue is far from saturated:

- Cutting instructions that do not stand on the critical path → wall clock does not move (the critical path is stall, not issue width);
- Worse: the remap makes the stall structure **worse** — B-side amortization halves and two warps redundantly read the same B span, so the waits on the MIO/smem dependency chain thicken.

Three ncu readings piece together the full causal chain: instructions −5.8% (the stream got thinner), **SM% 31.5 → 29.75** (issue efficiency actually dropped), **SM Active Cycles 4.61 → 5.04 M (+9.2%)** (the same work, but the SM spent more active cycles waiting) → duration −0.3% (net effect ≈ 0). This is not "measurement noise masking the gain" — it is **the gain being mechanically cancelled**.

### 2.4 Why this veto has general value

This is the **third independent confirmation** of the same rule, and the three lever classes all differed: x-tile/j-tile/cp.async-db cut traffic shape (L2 bytes), r17 cut instruction count — two different dimensions of "getting thinner", both zeroing out at SM% ~30. The rule therefore upgraded from "some class of change doesn't work" to "**this regime's constraint is latency/stall structure, not resource consumption volume**". The rule's positive form was later cashed by r20: without changing a byte of traffic or removing a single instruction, only re-ordering the LDG→STS dependency timing, +7.1%.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Pure index remap, not one staging line touched**. Staging buffers, qb8 slot layout, sds scale, launcher all stay — the experiment answers only "how much is the warp-shape variable alone worth", mixing in no other degrees of freedom. This keeps the veto conclusion clean: what was measured is the shape's own causality.
- **The unchanged sum[64] budget is both a constraint and a moat**. The 16-chain × 4-reg accumulator structure is the ILP depth validated in r12; a remap that changed the chain count (say 4×4 → 2×8) would have introduced a second variable. Keeping 16 chains keeps the experiment single-variable.
- **Parity-first build discipline**: full index rearrangement fails most easily in the C fragment's lane→(row, col) mapping; this experiment's first build came out parity-all-green (KD=4 + KD=8 + default), showing that m16n8k32's fragment layout math holds under both shapes — itself a validation of the layout understanding.
- **Variant kept in /tmp, not committed**. After the revert the remap code had no commit value (the mechanism was vetoed), but the complete recipe went into the commit message (the five `wn/wm/i0w/j0w` expressions + h<4/nh<4 + 2×ldmatrix.x4) so it can be rebuilt from the record if ever needed. This is "even a veto must be reproducible" handling.

### 3.2 Key code

**Before shape** (the r17-era tree, `151fa97`'s `mmq_raw_wide_nt_kernel`; the warp mapping of the time was one 16-od slot per warp, `j0w = warp*16`, over all 128 tokens). A fragments: 8 groups, one ldmatrix.x4 each —

```cuda
int clow[8][2][4];
#pragma unroll
for (int g = 0; g < 8; g++) {                 // 8 groups of 16 tokens = 128 tok
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
```

B fragments: **one** ldmatrix.x4 serves both 8-od-row minitiles at once (matrices 0/1 = the two k halves of od rows 0–7, matrices 2/3 = od rows 8–15); the register distribution is exactly mma.m16n8k32's B-operand layout —

```cuda
{
    const uint8_t* rb8 = qb8
        + (size_t)sg * (MMQ_WBJ * MMQ_WBQ)
        + (size_t)(j0w + (lane >> 4) * 8 + (lane & 7)) * MMQ_WBQ
        + (size_t)((lane >> 3) & 1) * 16;
    unsigned b0_, b1_, b2_, b3_;
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{%0,%1,%2,%3}, [%4];\n"
        : "=r"(b0_), "=r"(b1_), "=r"(b2_), "=r"(b3_)
        : "r"((unsigned)__cvta_generic_to_shared(rb8)));
    b[0][0] = (int)b0_; b[0][1] = (int)b1_;
    b[1][0] = (int)b2_; b[1][1] = (int)b3_;
}
#pragma unroll
for (int g = 0; g < 8; g++)                   // 8×2 = 16 chains
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        mmq_mma_k32(clow[g][nh], a[g], b[nh]);
```

**After shape** (never committed; rebuilt from the `9d09a81` record): the warp mapping becomes `wn=warp&1 → i0w=wn*64`, `wm=warp>>1 → j0w=wm*32`; the A-fragment loop `g<8` → **`h<4`** (64 tokens per warp = 4 groups); the B fragment single ldmatrix.x4 → **two** (covering 32 od-rows, `nh<4`); the mma chain grid 8×2 → **4×4** (still 16 chains, `sum[64]` unchanged); staging/qb8/sds/launcher untouched line for line.

> Forensics note: per STYLE rule 0, this doc's git budget was spent on locating the before-shape code; the after-shape code exists in no commit (the variant lived only at `/tmp/cuda_kernels_r16remap_variant.cu`, gone after the machine rebooted), so its shape is given per the commit message's item-by-item description — no "pseudocode" excerpt is dressed up as a real source.

### 3.3 Pitfalls

- **First build was already parity-all-green** — a "pit not stepped in" worth recording: full index remaps fail most easily in the lane→(row,col) fragment mapping (the same minefield as the r15/r16 fold series); under the 4×4 grid both the A-fragment and B-fragment lane distributions changed and both were derived correctly. The precondition: writing out on paper first what each of m16n8k32's registers should hold.
- **A pre-registered bar is what gives the veto teeth**. Had the bar been set after seeing 1250/1301, +0.9% could easily have been defended into "the direction is right, keep it". This round's bar (KD=8 ≥ 1350) was registered before measurement, and the noise-band reading triggered the revert directly.
- **Reverts must be cmp-verified**: after reverting the kernel to `708067d`, a byte-for-byte comparison against the pre-experiment state (the commit message's own words: "cmp-verified") — preventing a "revert" from becoming "yet another unverified change".

## 4. Verification

- **Parity gate (KD=4 + KD=8 + default, three configs)**: after the full remap the output still matches the CPU reference — defends against lane-mapping errors; passed on the first build.
- **Interleaved 3× A/B vs the HEAD binary**: KD=4 1250 vs 1239, KD=8 1301 vs 1288 — defends against machine drift; the readings landed in the noise band (+0.9%/+1.0%), and against the pre-registered bar ≥1350 the verdict is a loss.
- **ncu counter A/B (q-proj KD=4)**: inst/GMAC, LDSM counts, duration, SM%, SM Active Cycles — this round ncu was not "a nice-to-have" but the **veto instrument**: the wall clock only says "no effect"; the counters explain "why no effect" (the −5.8% instruction cut was cancelled by the issue-efficiency drop).

## 5. Results

| Metric | before (r16 tree) | after (remap) | Verdict |
|---|---|---|---|
| KD=4 whole-prefill | 1239 | 1250 | +0.9%, noise band |
| KD=8 whole-prefill | 1288 | 1301 | +1.0%, noise band; bar ≥1350 missed |
| ncu inst / GMAC | 8.68 M | **8.18 M** | −5.8% (the stream really got thinner) |
| LDSM.x4 / block-chunk | 72 | **48** | −33% |
| ncu duration | 2.262 ms | 2.256 ms | −0.3% (the wall does not move) |
| SpeedOfLight Compute (SM) | 31.5% | **29.75%** | issue efficiency dropped instead |
| SM Active Cycles | 4.61 M | **5.04 M** | +9.2% (more waiting) |

**Veto mechanism** (why it was reverted, and what the evidence chain is):

1. The wall clock +0.9%/+1.0% falls in the noise band, 4 percentage points short of the pre-registered bar (KD=8 ≥ 1350, requiring +4%+);
2. Mechanically this is not "not yet tuned": the −5.8% instruction cut was fully cashed in the issue stream, but SM% fell 1.75 points instead and Active Cycles rose +9.2% — B-fragment amortization halved plus dual-warp redundant reads of the same B span made the stall structure worse, cancelling the thinner stream;
3. This is the third independent confirmation that "at SM% ~30, per-MAC instruction cuts buy ~0 wall clock" (the first two: the traffic-shape cuts of x-tile/j-tile, and the staging family); the rule upgraded to a regime-level constraint;
4. Retry conditions: **only after the stall structure is fixed and SM% rises materially** can the −5.8% instruction cut cash out at ~1:1 — and r20 (split-phase A staging, +7.1%) is precisely the correct fix that repaired the stall structure first; after r28 the NB kernel re-defined the occupancy regime at 2 blocks/SM, the wide kernel was no longer the performance path, and this shape experiment was never re-run under the new regime.

**Revert execution**: the kernel was restored to `708067d` (r16's fold kept) and cmp-verified; the remap variant lived only in `/tmp` (never committed), with the recipe fully recorded in `9d09a81`'s commit message.

**The second half of this rule's story** (later verification within the same campaign, showing the "retry conditions" are not lip service): r25's SASS census found that 100% of the stream's excess was supporting instructions (int ALU 77%), and the wall clock stayed lazy after cuts — the same regime conclusion as r17; while **r29, after the NB kernel landed at 2 blocks/SM, measured int ALU −25% / inst −6.5% buying a real +2.80% wall clock** — once occupancy rose (the SM%-degraded premise removed), instruction cuts' payout recovered to near 1:1. r17's veto is therefore "wrong timing" rather than "wrong direction forever": in the 1 block/SM stall-bound regime it is zero; in the 2 blocks/SM regime it is real.

## 6. Lessons

1. **An instruction cut's payout is premised on SM%**: when issue is far from saturated (~30%), cuts land where they don't stand on the critical path and the wall clock does not budge — read SpeedOfLight first, then pick the lever class.
2. **"Getting thinner" has two dimensions (bytes, instructions), and both zero out in the stall-bound regime**; the third axis that can move the wall clock is **the stall structure itself** (dependency timing, staging phases) — r20's +7.1% was the first cash-out of that axis.
3. **Shape is not a separable degree of freedom**: llama's warp shape is embedded in its own staging depth, ITER_K, and accumulator structure; porting the shape without the surroundings yields not "llama's shape" but a "mismatch".
4. **A veto experiment's entire value is in the mechanism readings**: the wall clock says "useless"; ncu explains "why useless and when it is worth another try" — otherwise the revert is just a waste.

---
← [21 · r16 narrow-kernel rank-1 fold](./21-r16-narrow-kernel-rank1-fold.md) · [Index](./README.md) · [23 · r18 load-time B pre-expansion](./23-r18-load-time-b-preexpansion.md) →
