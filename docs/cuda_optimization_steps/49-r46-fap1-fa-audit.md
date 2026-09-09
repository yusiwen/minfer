# 49 · r46 (FAP1) — FA audit + occupancy/bank-conflict levers: kernel −11% but wall-neutral (REVERTED)

> **Result**: the audit of `fa_prefill_f16kv` overturned the latent assumption "FA is a scalar kernel" — it **has long been wmma m16n16k16 + online softmax**; the real problem is occupancy starvation (69.38 KB smem/block → 1 block/SM, 16.64% occ, ncu estimated speedup headroom 68.7%) plus the bank-conflicted S/P smem round trip (row stride ≡ 0 mod 32 banks, MIO scoreboard 36%). Levers: FA_TKV 64→32 (~43.8 KB → 2 blocks/SM) + S/P row padding (+8 f32) + incidentally fixing a hidden launcher smem over-request (`3*FA_TQ` is correct only when `FA_TQ==FA_TKV`). Result: FA kernel 5.16 → 4.58 ms (**−11%**), 2 blocks/SM, SM busy 40% — whole-prefill only **+0.27%** (< the +1.5% bar; cutting 16 ms from the 124.7 ms FA slice is noise-level) → **REVERTED**. Named its successor FAP2: register-resident softmax, deleting the S/P round trip entirely (r48 landed +5.6%).
> **Commit**: `a186f51`. **Date**: 2026-09-05.

> **Provenance note**: r46's code change was reverted after measurement, and `a186f51` is a docs-only commit. The r46-era FA kernel excerpts here (FA_TKV=64, the Sf/Pf smem round trip, the old launcher formula) come from the **historical tree before r48 landed** (a bounded line range of `git show d38744d~1:src/cuda_kernels.cu`) — the last time this code existed in the repo; the current-tree excerpts (the FA_TKV=32 define, the +8 padding comments) are the parts that survived r46/r48, comments unchanged verbatim.

## 1. Background — where things stood

### 1.1 A stale map

On the afternoon of 2026-09-05, the q6_K line had swallowed two wall-neutral results in a row: r44 (pre-expanding B deletes the recomb, kernel −10.9%, wall −0.42%) and r45 (cp.async hides the A-side wait, kernel −10.2%, wall −0.34%). Two independent mechanism lines proved the same thing: **after r41 the q6_K GEMM is no longer the prefill wall's bottleneck**. That is good news (a line converged) and bad news (where does the next shot go?). The r37-era wall decomposition (q6_K GEMM at 51.2%) was stale — continuing to allocate budget by the old map would spend the whole campaign on irrelevant slices.

FA (prefill flash-attention, `fa_prefill_f16kv`) was the biggest question mark hanging in the archive. Its timeline: 8n (`cb66fca`) landed the tiled FA taking attention from 176 ms/layer to 8.5 ms/layer; P5·0 (`86ca78c`) put P·V on the tensor cores (10.06 → 4.24 ms/layer); P5·3 (`fc07c04`) fixed the K/V staging's 8-way ldmatrix conflict. FA had not been touched since, and the r23-era reading was "FA's 2.5×/layer gap is structural (llama keeps 128-wide KV tiles)" — a judgment frozen in the f16-path era, and after the MMQ redesign (r28–r41) turned the whole wall's composition over, no one had re-measured FA's relative position.

r23's judgment also carried an unverified assumption worth auditing today: the comparison object back when FA was called "structurally slow" was llama.cpp's 128-wide KV tile geometry; but "to what extent our FA already uses tensor cores" had itself not been fully audited since P5·0. If the audit found large scalar stretches in the kernel, the FAP was a 20×-class rewrite opportunity; if it found an already all-tensor-core structure, the opportunity was elsewhere.

One easily overlooked piece of context on the timeline: by 2026-09-05 three rounds in a row had been reverted (r43 parity FAIL, r44 wall-neutral, r45 wall-neutral), all on q6_K. r46 was the day's first move **leaving q6_K** — executing exactly the budget migration r44/r45's convergence declaration authorized. From this moment the campaign's narrative switched from "the depth of one GEMM line" to "the sweep of the whole wall"; r47's whole-wall decomposition and the FA/prepass line of r48–r52 are all downstream of this switch.

### 1.2 FAP1's project shape: audit first

So FAP1's first step was not writing code but **auditing**: what is this kernel actually now? The audit also had to answer a second question: what share of the wall does the FA slice hold now, and is the lever's ceiling enough to clear the bar? The two answers together decide whether FAP1 "acts" or "books and moves on". In hindsight, this project order saved the session — of the audit's three findings, the first (the structure is innocent) closed an imagined rewrite, the third (the slice's ceiling) vetoed the day's lever, and the second (the mechanism list of occupancy + conflicts) was inherited verbatim by FAP2 two months later. Each finding paid for itself, but only together did they constitute a complete decision.

## 2. Principle — the GPU mechanism

### 2.0 The audit method: three evidence lines

The audit ran along three parallel lines, cross-checking each other:

1. **Structure line (read the code)**: read `fa_prefill_f16kv` section by section, annotating each compute stage's execution unit (wmma / ALU / smem round trip) — answering "what is it".
2. **Resource line (do the accounting)**: derive each block's byte count from the kernel's smem layout declarations, divide by the device's smem/warp budget to derive blocks/SM and the occupancy ceiling — answering "is it starved".
3. **Behavior line (ncu)**: occupancy counters, estimated speedup, stall composition (MIO scoreboard etc.) — answering "which class of resource is the bottleneck on".

The three lines converge in §2.1–2.3: structure innocent, resources starved, behavior consistent.

### 2.1 Audit finding one: structure innocent — wmma + online softmax are already there

The structure checklist confirmed by reading the kernel section by section (historical tree, FA_TKV=64 era):

- **QK^T**: `wmma::mma_sync` (m16n16k16, f16 A/B, f32 accumulate), each warp holding several 16×16 accumulator fragments (`fc[0]`/`fc[1]` in the excerpt below cover one 32-column group), looping over hd in steps of 16 — the Q row blocks × K column blocks run entirely on the tensor cores.
- **online softmax**: per-KV-tile `m`/`l`/`alpha` state (block-shared arrays `msh`/`lsh`/`alpha`, one entry per row) + exponential rescaling — the standard flash-attention shape, not a "one-shot whole-softmax".
- **P·V**: P enters wmma as an f16 A-operand from `Pf` via `load_matrix_sync`, V is the B-operand — the second tensor-core stage.

The "FA is a scalar kernel" assumption is falsified — **a rewrite-class opportunity does not exist**. That conclusion alone paid for the audit: it shut down the "FAP = a big rewrite" fantasy and pointed the campaign at small, precise levers. The audit also supplied a reference coordinate: llama.cpp's fattn keeps 128-wide KV tiles (the source of r23's old judgment), while we run 64 — narrower tiles mean worse amortization of per-tile fixed overhead, a second structural disadvantage beyond §2.2's occupancy starvation, but one that cannot be fixed by "widen the tile" (the r24/P5·neg lesson: wider tiles are worse at 1 block/SM); it can only be discussed after occupancy is solved.

### 2.2 Audit finding two: occupancy starvation — the 69.38 KB arithmetic

The kernel's smem ledger (FA_TQ=64, FA_TKV=64, hd=128, staging row stride = hd+8 = 136 halves):

| Plane | Size | Bytes |
|---|---|---|
| Qs (q tile, f16) | 64×136×2 | 17,408 B |
| Ks (K tile, f16) | 64×136×2 | 17,408 B |
| Vs (V tile, f16) | 64×136×2 | 17,408 B |
| Sf/Pf (aliased reuse: S f32 → P f16) | 64×64×4 | 16,384 B |
| msh/lsh/alpha (online softmax state) | 3×64×4 | 768 B |
| **Total** | | **69,376 B ≈ 69.38 KB** |

69.38 KB/block → each SM fits only 1 block (2 blocks need 138.8 KB, over the budget ceiling); 256 threads = 8 warps, which against 48 warp slots is **16.64% occupancy**. ncu's estimated-speedup reading: **68.7%** — the SM spends most of its time without enough resident warps to fill the latency.

Why 16.7% is especially lethal for this kernel: FA's main loop is the serial chain "stage KV tile → QK^T → online softmax → P·V", and every segment of the chain has dependency stall-waits (staging waits on global, softmax waits on smem, mma waits on operands). Occupancy's job is to let **other warps' instructions** fill those stall holes. 1 block/SM means the holes can only be filled by the same block's 7 sibling warps, which share the same serial chain's rhythm (aligned to the same tile boundaries); 2 blocks/SM introduces a second block in a different phase, with a very different hole-filling probability. FA's KV tile loop runs dozens of iterations (nt/FA_TKV rounds), and each round's stall residual × dozens of rounds is the main source of ncu's 68.7% estimated headroom. For a kernel that passes dozens of KV tiles per layer, double buffering is the only source of overlap — and double buffering can only hide so much depth; occupancy is its missing amplifier.

### 2.3 Audit finding three: the S/P round trip's bank conflict — 256 B ≡ 0 mod 128 B

S/P makes a full round trip through shared memory: QK^T's wmma accumulator `store_matrix_sync` into `Sf` (f32) → softmax reads Sf, writes `Pf` (f16, aliasing the same smem) → P·V's `load_matrix_sync` reads back from Pf. The problem is the row stride: `Sf`'s row stride = FA_TKV f32 = 64×4 = **256 B**; `Pf`'s row stride = `FA_PSTR` = FA_TKV×2 halves = **256 B**. 32 banks × 4 B = 128 B per bank cycle, and 256 B ≡ 0 (mod 128 B) — **every row starts on the same bank group**.

Unroll one concrete access: P·V's `load_matrix_sync(pa, &Pf[...], FA_PSTR)` takes 8 rows × 16 halves at once, and every ldmatrix row lands on a start address ≡ 0 (mod 256 B) — i.e. all 8 rows hit the same bank group and the access serializes into 8 beats. The softmax's row rotation (one warp handling rr = warp, warp+8, warp+16... in turn) also re-calibrates to the same bank group on every row change. This is **the same disease** P5·3 fixed on K/V staging ("256 B rows ≡ 0 mod 32 banks = 8-way ldmatrix conflicts"; the fix then was +8 half row padding, making the stride 272 B ≡ 16 mod 128 B) — this time growing on S/P. Corroborating evidence: MIO scoreboard holds **36%** of stalls — the smem round trip's queue pressure; a tax of "8 extra beats per round trip per tile × dozens of tiles per layer" lands squarely on the MIO pipe.

### 2.4 The lever arithmetic: FA_TKV 64→32 + row padding

- **Occupancy**: after halving FA_TKV, Ks/Vs each fall from 17,408 B to 8,704 B and Sf/Pf from 16,384 B to 8,192 B: total Qs 17,408 + Ks 8,704 + Vs 8,704 + Sf/Pf 8,192 + state 768 = **43,776 B ≈ 43.8 KB** → 2 blocks per SM (87.6 KB) → 16 warps / 48 slots = **33.3% occupancy**, doubled.
- **Bank conflicts**: S/P rows each padded +8 f32, the stride becoming 288 B ≡ 32 (mod 128 B) — cross-row accesses land on staggered bank groups.
- **Cost forecast**: halving FA_TKV doubles the KV tile count per layer (nt/FA_TKV rounds), and every tile's softmax state updates (the three shared-memory arrays m/l/alpha read/written), masking, and tile-boundary synchronization all double — this cost was outweighed at the time by "occupancy doubled" (net −11%), and r50 (FA_TKV 32→16) later proved it to be this lever class's intrinsic tax rate: the occupancy gain is sublinear (each added block yields fewer new hole-filling opportunities) while the per-tile cost is linear (tile count × fixed cost), so the two curves must cross. Parked here for now; r50 measures the intersection.

The two smem ledgers, before and after the lever, side by side (hd=128, sstr=136 halves):

| Plane | FA_TKV=64 (before) | FA_TKV=32 (after) |
|---|---|---|
| Qs | 64×136×2 = 17,408 B | 17,408 B (unchanged) |
| Ks | 64×136×2 = 17,408 B | 32×136×2 = 8,704 B |
| Vs | 64×136×2 = 17,408 B | 32×136×2 = 8,704 B |
| Sf/Pf | 64×64×4 = 16,384 B | 64×32×4 = 8,192 B |
| msh/lsh/alpha | 768 B | 768 B |
| **Total** | **69,376 B → 1 block/SM** | **43,776 B → 2 blocks/SM** |

Note Qs unchanged and Ks/Vs/Sf halved linearly — this is the occupancy lever's "subtraction" essence: it adds nothing, only presses the request under the threshold, unlocking the SM's second set of resources (the second block's warp slots).
- **Why the padding is +8 f32**: the Sf rows (f32) and Pf rows (f16, aliasing the same smem) each gain 8 elements of their own width — the Sf row stride becomes `(FA_TKV+8)×4 = 288 B ≡ 32 (mod 128 B)`, and Pf's likewise counted in halves. 288 and 128's greatest common divisor is 32, so cross-row accesses stagger by 8 banks — no longer the 0-offset full conflict, nor the exactly-halved (128 B) 2-way. +8 is the same dose P5·3 validated on staging (272 B ≡ 16 mod 128 B, the half version), not an arbitrary odd number.

### 2.5 The hidden killer: the launcher's smem over-request

The launcher's requested dynamic smem sets the occupancy ceiling. The old formula:

```
smem = 3 * FA_TQ * (hd + 8) * 2  +  FA_TQ * FA_TKV * 4  +  3 * FA_TQ * 4
```

The first term assumes all three planes Q/K/V have FA_TQ rows — **true only when `FA_TQ == FA_TKV`** (then 3×64 = 64+2×64, exactly equal). With FA_TKV changed to 32, the first term's real requirement is `(FA_TQ + 2*FA_TKV)*(hd+8)*2 = 34,816 B`, yet the formula still requests `52,224 B` (17.4 KB too many, effectively counting Vs as a full FA_TQ-sized plane); adding Sf/Pf and the state, the total request is 61,184 B ≈ 61.2 KB → 2×61.2 = 122.4 KB **still over the ceiling** → still 1 block/SM. That is: without fixing this line, all of FA_TKV=32's occupancy gain **silently zeroes out**, while functionality, parity, and even kernel time can all look "normal" — occupancy goes from 1 block to 1 block, with no error whatsoever. This is the doc's most valuable pit (see §3.3).

The word "hidden" deserves unpacking too: this bug was **correct** during the years FA_TQ == FA_TKV (the two expressions are identical), so it survived three rounds of changes (8n → P5·0 → P5·3) unharmed. A latent bug's signature is "the equivalence holds under the current parameters" — the first asymmetric parameter (r46 was this campaign's first experiment with asymmetric FA tiles) detonates it. Writing `3 * FA_TQ` or `(FA_TQ + 2*FA_TKV)` made no observable difference while the equality held; the difference only shows when the first symmetry-breaking change arrives.

### 2.6 The wall-clock ceiling: why this lever was doomed to fall below the bar

The FA slice was about **124.7 ms** of the wall at the time (r47's subsequent precise attribution: 125.8 ms = 10.2% of 1239 ms GPU busy). The lever's mechanism ceiling is kernel −11% ≈ **−16 ms**, against a ~1.27 s whole-prefill wall a **~1.3% nominal ceiling** — even fully cashed in and serially added to the wall, it is already under the +1.5% bar. The measured +0.27% (within the noise band) says even this much barely surfaced — the A/B noise band is of ±2% magnitude, and +0.27% is indistinguishable from 0.

Re-doing the arithmetic of "why even the nominal ceiling is this small": the lever's 11% magnitude looks decent, but it acts on a slice holding only 10% of the wall, so the product is 1.1–1.3%. **Lever value = slice share × lever magnitude**; once multiplied, any small factor can kill the product — this arithmetic later became the standard back-of-envelope before project approval (r55's swiglu roofline bound and r47's FAP2 estimate 2× → −4.9% use the same formula). The structure of the conclusion matters: **the mechanism holds (kernel −11%), the lever is too weak (slice × magnitude = ceiling < bar)** — what is vetoed is "this lever × this timing", not "the FA slice" (a distinction r47 immediately corrects, see §5).

The contrast of "same mechanism, different wall" is also worth recording: before 8n, attention (the LOCAL-memory-accumulation version) was 76% of whole-prefill, and a kernel improvement of the same magnitude in that era was a +8%-class wall event; by r46's day it was worth +0.27%. A mechanism's value is not conserved — every time the wall's composition changes, every historical lever's value must be re-priced. That is the arithmetic essence of "wall decompositions have a shelf life" (r47).

## 3. Implementation

### 3.1 Design choices: an audited small lever, not a rewrite

The audit's conclusions directly shaped the implementation: the structure (wmma + online softmax) stays untouched; only three quantities move — `FA_TKV` 64→32 (one `#define`, everything else symbolic), the S/P row stride +8 f32 (two stride constants), and the launcher's smem formula (one line). Each is a minimal diff that can be reverted independently.

"Everything else symbolic" deserves emphasis: every tile-related quantity in the kernel (columns per tile, softmax's column ownership, mask boundaries, the grid's token tiling) must be derived from FA_TQ/FA_TKV rather than hardcoded — r46's diff is small because the FA code has maintained this discipline since 8n; conversely, any hardcoded constant silently breaks when tiles change (r50 runs into this from another direction: the tail tile's O write-out reuses smem as a 64×128 f32 buffer, and that 32 KB is an implicit constant no symbol covers, so the launcher must take the max to be safe). There is one more reason not to rewrite: P5·0/P5·3's records already prove this FA skeleton's tile geometry is a measured local optimum; tearing it down has no audit evidence behind it.

### 3.2 Key code

First draw the kernel's per-tile execution flow, marking where each of r46's three levers lands (the +8 padding on K/V staging is pre-existing from P5·3; the S/P padding and TKV are new in r46):

```
per block (one 64-token q tile × one head):
  stage Q → Qs [FA_TQ rows]                     ← +8 padded row stride (P5·3)
  for kt in 0..nt step FA_TKV:                  ← r46: step 64→32 (lever 1)
    stage K/V → Ks/Vs [FA_TKV rows]             ← +8 padded row stride (P5·3)
    QK^T (wmma) → fc fragments
    store fc → Sf [row stride = FA_TKV f32]     ← r46: +8 f32 padding (lever 2)
    softmax: read Sf / update m,l,alpha / write Pf   ← the S/P smem round trip (deleted in r48)
    P·V (wmma): load Pf [row stride = FA_PSTR]  ← likewise protected by the padding
    rescale O
  writeout O
launcher: smem formula                          ← r46: 3*FA_TQ → real row counts (lever 3)
```

**The r46-era smem layout and S/P round trip (historical tree `d38744d~1`, pre-r48)** — the definitions and plane layout:

```cuda
#define FA_TQ 64
#define FA_TKV 64                                    // ← r46's lever: change to 32
#define FA_PSTR (FA_TKV * 2) // probs row stride in halves (256B): probs row r
                             // aliases only Sf row r's first half, already read
                             // by the same thread — no cross-thread race
...
__half* Ks = Qs + FA_TQ * sstr;
__half* Vs = Ks + FA_TKV * sstr;
float*  Sf = reinterpret_cast<float*>(Vs + FA_TKV * sstr);
__half* Pf = reinterpret_cast<__half*>(Sf);        // alias: probs after softmax
float*  msh  = reinterpret_cast<float*>(Sf + FA_TQ * FA_TKV);
float*  lsh  = msh + FA_TQ;
float*  alpha = lsh + FA_TQ;
```

S's landing and the softmax round trip (`store_matrix_sync` out to Sf → read Sf for max/sum → write Pf; row stride `FA_TKV` f32 = 256 B, i.e. §2.3's conflict source). Also read the online softmax's state structure along the way: `msh`/`lsh`/`alpha` are block-shared arrays, one entry per row — m/l updates, the cross-tile alpha rescaling, and the final O scaling all pass through them, once per KV tile; the S/P smem round trip stacks on every step of this state machine. In the whole "warp-per-row" loop, one row's 64 columns are split 2 per lane across 32 lanes, with max/sum reduced via a `__shfl_xor` tree — that part is clean design; the problem is only the smem transport before and after it:

```cuda
wmma::store_matrix_sync(&Sf[swm * 16 * FA_TKV + swk * 32], fc[0], FA_TKV, wmma::mem_row_major);
wmma::store_matrix_sync(&Sf[swm * 16 * FA_TKV + swk * 32 + 16], fc[1], FA_TKV, wmma::mem_row_major);
...
for (int rr = warp; rr < FA_TQ; rr += 8) {         // one WARP per row
    int c0 = lane * 2, c1 = c0 + 1;
    float s0 = v0 ? Sf[rr * FA_TKV + c0] : -INFINITY;   // ← round-trip stop 2: read Sf
    ...
    Pf[rr * FA_PSTR + c0] = __float2half(p0);           // ← round-trip stop 3: write Pf
    Pf[rr * FA_PSTR + c1] = __float2half(p1);
}
// round-trip stop 4 (P·V): ldmatrix takes 8×16 from Pf at row stride FA_PSTR — 256B ≡ 0,
// 8 rows in the same bank group = 8-way conflict:
wmma::load_matrix_sync(pa, &Pf[wm * 16 * FA_PSTR + kk0], FA_PSTR);
```

**The old launcher formula and its hidden over-request (same historical tree)**:

```cuda
// old: the first term counts K/V as FA_TQ rows too — correct only when FA_TQ==FA_TKV
size_t smem = (size_t)3 * FA_TQ * (hd + 8) * 2
            + (size_t)FA_TQ * FA_TKV * 4 + 3 * FA_TQ * 4;
// r46's fix (at the time): the staging term counts the planes' real rows —
//   (FA_TQ + 2*FA_TKV)*(hd + 8)*2 + FA_TQ*FA_TKV*4 + 3*FA_TQ*4
//   → at TKV=32 the request goes 61,184 → 43,776 B, unlocking 2 blocks/SM.
```

**The r46 legacy surviving in the current tree (`fa_prefill_f16kv` and the launcher, the FAP2 version)** — `FA_TKV 32` is exactly the value r46 introduced; the launcher's smem formula has been changed to count the planes' real rows, and the current tree's comment signs this fix directly:

```cuda
#define FA_TQ 64
#define FA_TKV 32                          // ← r46's 64→32 survives to this day
...
// Padded smem row stride: hd=128 halves = 256B ≡ 0 mod 32 banks makes
// every wmma ldmatrix row land on the same bank group (8-way conflict
// per load). +8 halves (272B) shifts each row by 4 banks.
const int sstr = hd + 8;
```

```cuda
int launch_fa_prefill_f16kv(...) {
    // Qs + Ks + Vs only (S/P no longer go through shared memory). sstr = hd+8
    // padding; Ks/Vs are FA_TKV rows (the r46 launcher's 3*FA_TQ bug is gone).
    size_t smem = ((size_t)FA_TQ + 2 * FA_TKV) * (hd + 8) * 2;
    ...
    dim3 grid((nt + FA_TQ - 1) / FA_TQ, nh, 1);
    fa_prefill_f16kv<<<grid, 128, smem, stream>>>(...);   // 256→128 threads is r48's change
}
```

The S/P round trip itself was deleted entirely in r48 (S/P no longer enter smem, which is why the current launcher has no Sf/Pf term at all), but the "row stride +8" padding idea lives on as the staging padding (`sstr = hd+8`) — the same 272 B rule that P5·3 introduced, r46 reused on S/P, and r48 kept on the K/V/Q staging.

### 3.3 Pitfalls

1. **The launcher smem over-request is the lever's silent killer.** `3*FA_TQ` happens to equal `(FA_TQ+2*FA_TKV)` when `FA_TQ==FA_TKV`, harmless for years; the first asymmetric tile change makes it over-request by 17.4 KB, pressing 2 blocks/SM back to 1 — **no error, no functional anomaly, kernel time nearly unchanged**; the only observation point is the occupancy counters. For any "change tile size to buy occupancy" experiment, step one is finding the line where the launcher formula's equality with the real plane geometry fails.
2. **The 256 B ≡ 0 mod 128 B bank rule recurs.** P5·3 fixed the staging rows, and r46 hit the same disease on the S/P rows; only r48's outright deletion of the S/P round trip cured it. Any smem layout whose "row stride is a multiple of the warp-visible width" must first pass this congruence check.
3. **The fixed-cost tax of halving tiles exists from day one.** FA_TKV halved → tile count doubled → per-tile softmax state updates/barrier counts doubled. In r46 the occupancy doubling covered it (net −11%); in r50 (32→16) it overtook (wall-neutral) — two segments of the same tax-rate curve, and r46+r50 together closed the entire "shrink the FA tile" axis.
4. **The audit precedes the lever.** Acting on the assumption and "rewriting FA as tensor core" outright would have burned the whole session on a kernel that was already tensor-core. Half an hour of code reading saved a directional error.
5. **The post-lever utilization reading must point to the next step.** After the lever landed, SM busy was only 40% — even at 2 blocks/SM, the SM still had long idle stretches (staging latency and the softmax serial chain remained). At the time this reading was not a "not good enough" setback but FAP2's signpost: the remaining problem was not in scheduling geometry (occupancy doubled, bar unmet) but in the data path (the S/P round trip itself) — delete it, and busy and the wall move together (r48's FAP2 version is exactly what improved SM busy and the wall simultaneously).

## 4. Verification

| Gate | Reading | What it defends against |
|---|---|---|
| Structure audit's three readings | wmma/online-softmax in place; smem ledger 69.38 KB; ncu occ 16.64% + est speedup 68.7% | "investing in the wrong kernel" — the audit is itself the evidence chain, and every later number checks back against §2's arithmetic |
| Measured occupancy | **2 blocks/SM** after FA_TKV=32 + the launcher fix | §3.3#1's silent over-request — occupancy must be measured, never inferred from "the code changed" |
| Kernel timing | 5.16 → 4.58 ms (**−11%**), SM busy 40% | rules out "the mechanism never took effect" |
| whole-prefill interleaved A/B | **+0.27%** (< the +1.5% bar) | machine drift faking a trend; same-window same-binary interleave, the verdict "lever too weak" |

Note: r46's record lists no parity/greedy-specific gate (the lever is a pure scheduling-geometry change; the numeric path is untouched); r50 of the same month added this lesson after the fact — **any FA_TKV change alters the online softmax's accumulation order** (the length of each tile's m/l/alpha rescaling chain changes), so a strict greedy byte-identity gate is inherently unsatisfiable for this class of change (the FA campaign's new gate note, hit again in r57). r46 happened just before that gate note was discovered; after r50, the verification baseline for FA-class changes became "the parity gate + tolerance-style comparison", with the byte-identity gate reserved for changes that do not touch accumulation order.

## 5. Results

| Layer | before (r45's reverted state) | after (r46) | Verdict |
|---|---|---|---|
| Structure audit | "FA might be scalar" (unverified assumption) | wmma m16n16k16 + online softmax in place | no rewrite-class opportunity |
| smem/occupancy | 69.38 KB → 1 block/SM (16.64% occ) | 43.8 KB → 2 blocks/SM (33.3% occ) | lever took effect |
| S/P conflict | row stride 256 B ≡ 0 (MIO 36%) | +8 f32 padding staggered the banks | mechanism took effect |
| FA kernel | 5.16 ms | 4.58 ms (−11%) | mechanism took effect |
| whole-prefill | baseline | **+0.27%** (nominal ceiling ~1.3%) | **< the +1.5% bar** |

Measurement protocol as in r44/r45: same-window same-binary interleaved A/B; kernel-level is matched-nt nsys/ncu sampling. The A/B noise band is of ±2% magnitude, and +0.27% is statistically indistinguishable from 0 — this is both the evidence for "below the bar" and a reminder that any <1% wall reading alone cannot ground a conclusion; it must be triangulated with mechanism-layer readings (kernel timing, occupancy, SM busy). vs-llama stays at 1.27×.

**Veto mechanism (why reverted)**: +0.27% is far below the +1.5% bar. The arithmetic had already sealed it: FA slice ~124.7 ms × lever magnitude 11% ≈ −16 ms, a ~1.3% nominal ceiling against the ~1.27 s wall — **even perfectly cashed in, it cannot clear the bar**, and the measured value merely confirmed the ceiling was not even reached. r46's written conclusion at the time was "FA is not the wall-critical path in the converged-GEMM domain"; reverted by discipline (code not kept, mechanism filed).

**The post-hoc correction and retry conditions (this doc's most important turn)**: the immediately following r47 redid the converged-domain wall decomposition with fresh nsys, **overturning the "FA is not wall-relevant" half of that sentence** — FA = 125.8 ms = 10.2% of wall = **5.72× vs-llama, the whole engine's #1 structural residual**. r46 and r47 do not contradict; they veto/endorse different propositions: r46 proves "**this lever** (the −11% occupancy/conflict fix) cannot reach the bar on this slice"; r47 proves "**this slice** deserves a bigger lever". FAP2 was thus chartered: register-resident softmax (delete the S/P smem round trip entirely rather than pad it), estimated 2× → −63 ms = −4.9% wall — r48 cashed it: FA kernel 5.16 → 2.12 ms (2.43×), whole-prefill 2603.5 → 2749.9 (**+5.6%**), FA's wall share 10.2% → ~4.7%. r46's mechanism findings (occupancy starvation + the S/P round trip) were inherited verbatim by r48; only the solution upgraded from "optimize the round trip" to "eliminate the round trip".

Two more rounds on the same lever axis closed the "shrink the tile" axis for good:

- **r50 (FA_TKV 32→16)**: 3 blocks/SM achieved, but the wall −0.5%/−0.01% — the occupancy gain was canceled by the doubled per-tile synchronization/softmax overhead (the fixed-cost tax forecast in §2.4 overtaking), and greedy-32 byte-identity was lost (inherent to FA_TKV changes). Merged with r46 into the conclusion: **shrinking the FA tile is a dead lever** (both points, r46 + r50, negative).
- **r57 (FA KV staging double buffering)**: FA_TQ 48 + double-buffered K/V; both attempts lost greedy-32 identity at token 19 — r50's "the FA tile is a size" notice striking again.

With that, the FAP lever map settled: beyond the occupancy axis (r46/r50/r57, dead), the only big lever left was eliminating the S/P round trip itself (r48, +5.6%) — exactly the value of r46's audit-produced mechanism list: it marked "where the blood is" correctly; the day's bandage was merely too small.

One more note on the campaign-wide impact: after r48 landed FA held ~4.7% of the wall, and r47's decomposition simultaneously named the A-quantize prepass's shared-A dedup (r49, +2.32%) and producer folding (r51/r52, +1.89%/+5.45%) — FAP1's same-day "book it and move on" decision ultimately cashed in all the mechanism information it preserved, in the form of "FAP2 + the prepass line". r46's round's direct output list:

| # | Output | Form | Downstream cash-in |
|---|---|---|---|
| 1 | The structural conclusion "FA is already wmma + online softmax" | audit record | spared a directional rewrite |
| 2 | The mechanism list of occupancy starvation + the S/P round trip | audit record | r48's FAP2 accepted it wholesale |
| 3 | The launcher smem formula fix (staging term counts real rows) | code (survived) | permanent; signed in the current tree's comments |
| 4 | `FA_TKV = 32` | code (survived) | permanent; r48/r50's baseline |
| 5 | FAP2 chartered (2× → −4.9% wall estimate) | record | r48 cashed +5.6% |

Judged as "REVERTED", this is the directory's densest-output doc: of five outputs, four survived across rounds; the code revert only discarded "the occupancy lever itself" — the one thing already proven insufficient at the time.

## 6. Lessons

1. **Positive mechanism / negative wall clock is a real outcome class**: record the mechanism and re-rank the slice, rather than discarding both — "kernel −11% but wall +0.27%" hides the next step's map.
2. **A wall-neutral measurement vetoes "this lever", not "this slice"**: judging a slice's value requires a fresh whole-wall decomposition (r47), never extrapolation from the old ledger or a single lever's result.
3. **Check the launcher before any occupancy-class lever**: the smem request formula's equality with the real plane geometry holds only for certain tile combinations; change the tile without checking the launcher and the gain silently zeroes out with no error at all.
4. **Audit before coding**: half an hour of structural audit closed an imagined 20× rewrite and pointed the budget at the FAP2 that actually worked.

← 48-r45-cpasync-q6k-a-staging · [Index](./README.md) · 50-r47-converged-wall-decomposition →
