# 17 · x-tile / j-tile / cp.async-db — the staging-shape family, closed (REVERTED / closed)

> **Result**: on top of r12's landed state (baseline band 1035–1058 tok/s), three staging-shape experiments were vetoed in a row: x-tile (256 tok × 64 od) ~942 (**−9%**, kernel +27%); j-tile (128 tok × 256 od, A reuse) ~1067 (**+2.6%**, bar 1150 not met); cp.async-db (double-buffered async staging) ~1034 (**neutral**, L2 SOL 75.7 → 78.2%). After three angles measured flat, the verdict: **the staging-order axis is closed**; the remaining levers are per-MAC L2 bytes and SM-side instruction efficiency — not traffic shape.
> **Commit**: `f061cb8` / `4993804` / `784786d` — **all three docs-only negative-result records**; the code snapshots lived in `/tmp` (j-tile noted as `/tmp/cuda_kernels_jh256.cu`) and no longer exist. **Date**: 2026-09-02 (all three same day).

> ⚠️ **Code evidence note**: none of this doc's three changes has a verifiable code commit. The excerpts below come in two kinds: those marked "current tree" are genuinely surviving code — the r12 kernel shape (`774a116`, the common host of all three experiments) and its descendants today; fragments marked "reconstructed" are rebuilt from the record's text, **not surviving code**.

## 1. Background — where things stood

r12 had landed the 16-chain warp tile that same morning: MMQ jumped from 441–481 to 1020–1058 tok/s (≈2.3×). But the same-window f16 default path was at 2284 — still 2.2× away, and r6's parity target (≥24 TMAC/s) unmet. r10/r11 had narrowed the suspects from "math" to "instruction-level structure", and r12 executed two of the three items (chain depth, A-fragment LDSM); the next natural hypothesis was the third structural class: **traffic shape** — the A-vs-B re-read ratio in L2, staging depth and order.

The hypothesis had concrete numbers behind it. At the 128×128 baseline tile, the A side's activation re-reads already outweighed B's weight re-reads: A streams 4,480 B per token over all K (40 B/chunk × 112 chunks, id=3584), re-moved once for every y-block (od-direction tile); the weight side touches 2,016 B per row per visit (144 B/super-block × 14 super-blocks). Cumulative over the whole prefill: **A re-reads 327 MB vs B re-reads 152 MB — A is already 2× B**. Intuition says that is unbalanced: why not tune the tile shape toward "read less A"?

And r12's fresh profile handed the hypothesis a knife: the 128×128 baseline's q-proj kernel already ran Memory SOL at **75.7%** — near the ceiling; meanwhile the j-tile experiment's pre-run measured issue **0.22** and active warps per scheduler **2.00**. Together these two readings are this doc's drama: the memory subsystem is nearly saturated (high SOL) but the SM's issue rate is very low (low issue) — **byte bottleneck or latency bottleneck?** There was no criterion yet. The three experiments are a differential diagnosis of exactly this question, one from each direction.

So three shapes were measured in one day, each moving one axis: **x-tile** (widen the token dimension to 256, indirectly squeezing od tile to 64), **j-tile** (merge od to 256 so A moves once and is reused by both), and **cp.async-db** (keep the tile, swap synchronous staging for a cp.async double-buffered async pipeline). The three represent the two directions of "make A cheaper" and one direction of "make the waiting disappear".

## 2. Principle — the GPU mechanism

**Why x-tile's ledger is negative.** The block tile goes 128 tok × 128 od → 256 tok × 64 od: doubling the token dimension halves the x-block count; halving od doubles the y-block count. Two traffic streams follow:

- **A re-reads ∝ y-block count**: every block stages the whole A tile. y-blocks double → A traffic doubles: 327 MB → ~654 MB (**+327 MB**);
- **B re-reads ∝ x-block count**: each block touches its own 64/128 weight rows. x-blocks halve → B traffic halves: 152 MB → ~76 MB (**−72 MB**).

Paying +327 MB of A to buy −72 MB of B, when A is already the 2× majority — **strictly negative**. The reverse, 128 tok × 256 od (widen both to keep the product), is geometrically feasible but 512 threads/block cannot stay resident at REG 156–166 — the register wall seals that road. The ncu evidence matches the ledger: q-proj kernel 4.584 vs 3.619 ms (**+27%**), Memory SOL 73.6% vs 75.7%, Compute 22.1% vs 22.9% — utilization barely moved; the same work took 27% more time.

**j-tile: bytes saved, time not.** 128 tok × 256 od + an outer jh loop: A stages once per k-tile and is reused by both 128-od halves — A re-reads measured halved (**−163 MB L2**). Geometrically, merging block od 128 → 256 halves the y-block count and A's move count with it; inside the block, an outer loop computes od half 0 first, then half 1, the two halves sharing the same staged A and accumulator structure (`sum[2][64]`, two copies). The price is one extra round of B-stage + `__syncthreads` per k-tile. The kernel's state: **1 block/SM, 2.00 active warps per scheduler, issue 0.22** — a classic latency bottleneck: too few warps, so the saved L2 bytes have no queue pressure to relieve, while the added stage+barrier round lands directly on the critical path. Result ~1067 (+2.6%), far below the 1150 bar (ncu side: q-proj kernel 3.62 → 3.55 ms, only −2%; Memory SOL 75.7 → 74.0% — it went down, from waiting one more barrier round). One engineering trap besides: the shared `qb8`'s restage-skip is **unsound** in this shape — the previous round's stage always holds the other half's rows; partial-slot expansion (KDR/2 pairs) pushes per-block B bytes back to 2× — blocked from both ends.

**cp.async-db: prefetch works, but the bottleneck isn't prefetch.** Two-level staging: raw bytes go out one k-tile early via `cp.async`, expansion happens smem→smem. L2 SOL rose 75.7% → **78.2%** — async movement does work, waiting does shrink. But the kernel is **L2-throughput-bound** (bytes/s at the ceiling), not MLP-starved (insufficient concurrency): sending the same bytes to L2 earlier leaves the throughput ceiling unchanged, wall clock neutral (~1034, baseline band 1035–1050).

**The mechanism behind the family verdict.** Together the three experiments covered traffic shape's three degrees of freedom: od split ratio (x-tile, negative), A reuse (j-tile, bytes saved but not time), async depth (cp.async-db, prefetch works but no bottleneck to solve). All three roads lead to the same reading: at 1 block/SM this kernel is bound by L2 throughput and the latency structure — **rearranging the same bytes produces no time**.

**How to read the counters: a two-metric cheat sheet.** Memory SOL (Speed of Light) is ncu's percentage of measured traffic over the hardware's theoretical peak — 75.7% means the memory subsystem is doing useful work three quarters of the time; high does not mean "more traffic still helps", because the bottleneck may be rotating elsewhere. issue (issued warp instructions per cycle per scheduler) is the utilization of the SM's issue ports — 0.22 means the four schedulers issue nothing most cycles, warps waiting on something (memory, barriers, dependency chains). **Reading "high SOL + low issue" together: the SM is waiting on memory, but memory is also nearly full — adding more bytes (x-tile's A, j-tile's second stage round) makes both sides worse together; making the same bytes arrive earlier (cp.async-db) improves waiting, not throughput.** That is the differential logic the three experiments' data combine into.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **All three experiments share the r12 host**: everything sits on `mmq_raw_wide_nt_kernel` (KD=4, baseline band 1035–1058), one geometry/scheduling parameter changed at a time — so the negative results corroborate each other instead of polluting each other.
- **x-tile chose 256×64 over 128×256**: the latter's 512 threads cannot stay resident at REG 156–166 (the record rejects it explicitly); 256×64 keeps 256 threads and the per-warp structure (warp = 16 od rows × 128-token half) — the minimal testable variant.
- **j-tile's bar pre-calibrated at 1150** (≈ baseline +9%): A halving was a profiled fact; the bar answers "is the structural complication worth it" — +2.6% misses the bar, withdraw, no feelings involved.
- **cp.async-db leaves the tile alone**: it isolates the single "async depth" variable, same mechanism as the narrow kernel's existing cp.async double buffer (excerpt C).

### 3.2 Key code

**Excerpt A · the host's staging shape (current tree, r12's synchronous-staging comment kept verbatim)** — the part all three experiments touch:

```cuda
// src/cuda_kernels.cu (current tree, the RAW_STAGE macro header of mmq_raw_wide_nt_kernel)
#define RAW_STAGE(kt)                                                          \
    do {                                                                       \
        /* llama.cpp-style synchronous staging, single buffer: plain global    \
         * -> smem loads, one syncthreads orders them. */                      \
        {  /* from r20 this evolves into split-phase: fire a batch of          \
             * independent LDGs into registers first, then STS them —         \
             * a later story; at r12 it was LDG->STS interleaved one by one */\
        ...
    } while (0)
```

**Excerpt B · the geometry the experiments changed (current tree launcher, real code)** — the two divisors of `grid` are exactly what x-tile/j-tile modified:

```cuda
// src/cuda_kernels.cu (current tree, launch_mmq_raw_wide_nt)
// 16-chain layout: 128-token x 128-od block tile. ...
dim3 grid((nt + 127) / 128, (od + 127) / 128);   // ← x-tile: (nt+255)/256, (od+63)/64
                                                 // ← j-tile: (od+255)/256 + in-block jh loop
if (kd <= 4) {
    const int smem = 4 * MMQ_WBI * 32 + 4 * MMQ_WBI * 8
                   + 8 * MMQ_WBJ * MMQ_WBQ + 2 * 4 * MMQ_WBJ * 4;
    cudaError_t e = cudaFuncSetAttribute(..., cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (e != cudaSuccess) { cudaGetLastError(); return 0; }   // over-smem: explicit refusal
    mmq_raw_wide_nt_kernel<4><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
}
```

With the token dimension pulled to 256, x-tile's KD=8 smem request goes over the cap — the launcher refuses explicitly per the table above and falls back to the narrow path (record verbatim: "KD=8 refused at the launcher -> narrow fallback"). This is the rule instituted after r7's phantom actually working: **resource caps must fail loudly**.

**Excerpt C · the mechanism cp.async-db wanted to port (current tree narrow kernel, real code)** — the narrow kernel has had a double-buffered cp.async pipeline since r7–r8; the db experiment carried it into the wide kernel:

```cuda
// src/cuda_kernels.cu (current tree, narrow kernel main loop — the db experiment's mechanism blueprint)
__device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void gemm_cp_wait1()  { asm volatile("cp.async.wait_group 1;\n"); }

RAW_STAGE(0, 0);
gemm_cp_commit();
int buf = 0;
for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
    if (kt + 1 < nktile) RAW_STAGE(kt + 1, buf ^ 1);  // prefetch the next k-tile
    gemm_cp_commit();
    gemm_cp_wait1();          // at most the prefetch group pending
    __syncthreads();          // scales (plain stores) + landed bytes visible
    ...
}
```

**Reconstructed fragments (flagged: not surviving code, rebuilt from the record)** — the three variants' shape differences:

```cuda
// [reconstructed] x-tile: block constants become 256 tok x 64 od, warp structure unchanged
// #define MMQ_WBI 256
// #define MMQ_WBJ 64
//   warp = 16 od-rows x 128-token half; KD=8 refused at the launcher by the smem cap → narrow fallback
//   result: A traffic ∝ y-blocks doubled (+327 MB), B traffic ∝ x-blocks halved (−72 MB) → ~942 tok/s

// [reconstructed] j-tile: od 256, outer jh loop lets two 128-od halves reuse the same staged A
// for (int jh = 0; jh < 2; jh++) {
//     /* A staging once per k-tile (hoisted out of jh); sum[2][64], two accumulator copies */
//     /* B re-expanded per (kt, jh) — the second stage + barrier round lands on the critical path */
// }
//   result: A re-reads −163 MB, wall clock ~1067 (+2.6%) < bar 1150

// [reconstructed] cp.async-db: wide kernel switched to double buffer — cp.async raw bytes one k-tile early, expansion smem→smem
//   result: L2 SOL 75.7 → 78.2%, wall clock ~1034 = neutral (baseline band 1035–1050)
```

### 3.3 Pitfalls

- **x-tile's KD=8 over-cap did not become a phantom**: the launcher refused explicitly and fell back to narrow — r7's silent-failure lesson (the 2124 phantom) institutionalized and cashed in for the first time, with the fallback path covered by the parity gate.
- **j-tile's restage-skip is unsound**: when the shared `qb8` is reused across jh, "the previous round's stage holds the other half's rows" voids the skip condition; partial-slot expansion pushes B bytes back to 2× — reuse and skip are mutually exclusive in this shape.
- **cp.async-db "reverted before landing"**: the neutral result was withdrawn on the spot, never even entering the tree behind an opt-in gate — the standard treatment of negative results: into the docs, leaving no live complexity behind.
- **All three lived in /tmp**: experiment code written, measured, and deleted the same day; only docs remain in the repo. r10's hook incident and this /tmp convention are two faces of one process rule: **the only vehicle for a negative result is the record**.

## 4. Verification

- **Parity gate**: all three variants parity green (x-tile both paths — the main path and the narrow fallback — pass) — defends against geometry rearrangement changing outputs.
- **Same-window interleaved A/B**: x-tile 927/953/945 interleaved three times vs the baseline band 1035–1058; j-tile median 1067 vs 1035–1050; cp.async-db ~1034 — all interleaved in the same window, defending against machine drift.
- **ncu evidence**: x-tile's duration/Memory-SOL/Compute triple proves "same utilization, longer"; j-tile's issue 0.22 / 2.00 warps-per-scheduler proves the latency bottleneck; cp.async-db's L2 SOL 78.2% proves prefetch worked.
- **Default-path isolation**: experiments sat behind opt-in env gates; the f16 default and the narrow raw path unaffected (narrow ~476 unchanged in the same window).
- **The verification gap on the code side must be recorded honestly**: the three experiments' code existed only in that day's /tmp working tree, with no re-verifiable commit — this doc's "verification" reproduces the measurement evidence chain (parity + A/B + ncu), not the code itself. That x-tile's narrow fallback path was covered by the parity gate matters especially: the fallback was not "untested", it was a second, tested path.

## 5. Results

| Experiment | Shape | Wall clock (7B @2K, same-window interleaved) | Δ | Key evidence | Disposition |
|---|---|---|---|---|---|
| x-tile | 256 tok × 64 od | ~942 (927/953/945) | **−9%** | kernel +27%, SOL flat; A +327 MB vs B −72 MB | REVERTED |
| j-tile | 128 tok × 256 od + jh loop | ~1067 | **+2.6%** (bar 1150 unmet) | A re-reads −163 MB; issue 0.22, 2.00 warps/sched | REVERTED |
| cp.async-db | double-buffered async staging | ~1034 | **neutral** (band 1035–1050) | L2 SOL 75.7 → 78.2% | REVERTED |

Veto mechanisms (each worth archiving): x-tile lost on the **ledger** (A is already the 2× majority, and it added to A); j-tile lost on **bottleneck mismatch** (under a latency bottleneck, byte savings don't cash); cp.async-db lost on **bottleneck conservation** (the L2 throughput ceiling doesn't move, so sending the same bytes earlier buys nothing).

**Retry conditions (STYLE rule: a REVERTED doc must state under what future conditions a retry is worthwhile).** None of this family's three vetoes is a permanent judgment — they are judgments **under the kernel's current state**; change the state and the ledger changes:

- **x-tile**: when A re-reads are no longer the 2× majority (e.g. after r34's quantize-transpose prepass turns A into a one-shot pre-transposed plane, or a shape where A resides in L2), the "+A buys −B" directionality inverts and the wide token dimension is worth re-measuring;
- **j-tile**: when the kernel escapes the 1-block/SM latency bottleneck (occupancy raised enough to cover the stage+barrier rounds, like the NB kernel's 2 blocks/SM after r28), the −163 MB of A bytes get a chance to cash into time;
- **cp.async-db**: when the kernel is genuinely MLP-starved (issue even lower, SOL also low, staging latency exposed on the critical path) — r53/r56 found exactly that state on the q6_K BT kernel, which is when the cp.async bundle finally landed (+5.03%/+2.35%).

In other words: **what this family closes is "rearranging traffic on the 128×128 16-chain kernel", not traffic optimization itself**. The same mechanisms revived once the bottleneck class changed — the most important hidden thread between this doc and the Era D docs that follow.

The family-closing verdict: after all three degrees of freedom (od split, A reuse, async depth) were measured, the staging-order axis formally closed. The next lever had to change "bytes per MAC" or "SM-side instruction efficiency" — and the immediately following r13 (counter forensics: 10.14 vs llama's 6.06 M warp instructions per GMAC) and r14 (B-fragment ldmatrix, +18.5%) walked exactly those two directions.

## 6. Lessons

1. **Compute the re-read ledger before tuning the tile**: A side 4,480 B/token × y-block count, B side 2,016 B/row × x-block count — the 327 vs 152 MB imbalance sits right there, and "symmetrically widen the token dimension" necessarily adds to the majority.
2. **Byte savings only cash under a byte bottleneck**: the 1-block/SM latency bottleneck (issue 0.22) ate j-tile's −163 MB; the L2 throughput ceiling ate cp.async-db's prefetch — confirm the bottleneck class before choosing the lever.
3. **Negative results need a complete evidence chain too**: each of the three carries parity + interleaved A/B + an ncu triple; only then does a family verdict stand. Code in /tmp, records in docs — this campaign's fixed archival form for negative results.

← 16 · [Index](./README.md) · 18 →
