# 15 · r10–r11 — reference inner-loop decomposition port; ILP reading verification (REVERTED / MEAS-ONLY)

> **Result**: porting llama.cpp's q8_1×q8_1 generic decomposition verbatim into our MMQ kernel: parity green, but 462–468 vs narrow's 470 tok/s (@2K) — **the same decomposition still runs ~6.4 TMAC/s** (llama ~30): the gap is not in the math. r10 pinned the residual to exactly three items — ILP depth / ldmatrix / tile — which became r12's execution recipe. r11 verified the tile `ne = I·J/32` (4 registers per m16n8k32 C fragment), establishing the "16 independent mma chains" reading.
> **Commit**: `025a69f` (r10, docs-only), `83e5580` + `6f29e65` (r11, docs-only). **All three are docs commits — the kernel code changes were swallowed by a post-checkout hook revert after measurement and never became code commits.** **Date**: 2026-09-01.

## 1. Background — where things stood

By the end of r9 the q4_K MMQ campaign stood at the "shape axis exhausted" node. R1 (`40e97c9`) had built the int8 tensor-core prefill GEMM — parity-clean but only ~6.1 TMAC/s; r7–r8's raw-byte kernel (the `d440d16` family) swapped staging to bare bytes and unpacked nibbles in registers, reaching 472 tok/s; after r8's wide tile (128-token block) landed, r9 finished the whole shape matrix: narrow cp.async KD=8 **481** (local optimum) > sync-wide 128×64 464 > compact 410 > wide KD=4 428 > narrow KD=4 427 > R1 441. All six shapes parity-clean: **under the current inner-loop structure, tile/staging shape is no longer a lever**.

Two details from this trajectory belong in the record before r10. One: R1's first measurement was 155 (co-tenant) / 412 (quiet), and 441 was only calibrated in the r7–r8 window — the same opt-in path swings 2.7× across machine states, the direct reason the campaign later made "same-window interleaved A/B" a hard rule. Two: r8's wide tile's first 2124 tok/s was a phantom of silent smem over-cap (attr-set and launch both failing silently, the GEMM writing nothing); after the guard the honest number was 428 — "suspiciously fast" readings should be checked against resource caps, a reflex r10's measurements would need again. The six-point matrix in one sentence: **every shape and staging variant sits in the narrow 410–481 band while llama runs ~30 TMAC/s** — the difference is structure, not parameters.

r9's real output was a completed read of llama.cpp's MMQ reference implementation (the `mmq-config-ampere.cuh` Q4_K branch): 256 threads, targeted occupancy 1, SRAM tile I=128 od rows × J≤128 tokens, ITER_K=256, synchronous staging, **16 `mma.m16n8k32` per warp per 32-k chunk**. The instruction model reconciles to **llama ~0.018 inst/MAC/thread vs our 0.133** — a 7× gap. That, not a magical tile shape, is the source of their ~30 TMAC/s.

r6's campaign target was MMQ from 6.1 TMAC/s to ≥24 (f16 parity) or ~30 (llama parity): f16 parity deletes the 56 ms convert pass (whole-prefill ~2670 tok/s), llama parity gives ~3250. We were still 5× from the f16 default path (same window 2320–2370) and every shape point had been tried — without eliminating the "different math decomposition" hypothesis first, all subsequent kernel-side work would rest on a wrong foundation.

r10's problem statement is thus concrete: **is our kernel doing "the same math" as llama's?** The q4_K dot product can be organized in different orders — when nibbles unpack, when dmin folds in, when scales multiply, whether the B side stays per-k signed int8 — and every organization changes the instruction stream. r9 reconciled "instructions per MAC" but not "what each instruction is". r10 ported llama's inner-loop decomposition item by item, made both sides identical in math organization, and measured the remaining gap.

This step also has a special archival situation: r10's kernel edits were reverted by the repo's post-checkout hook after that day's measurement, and **the code never became a commit**. What survives is only docs commit `025a69f`, carrying the full redo recipe and all measurement numbers. r11's two commits are likewise docs-only: first declaring r10's 16-chain claim unverified by code (`83e5580`), then verifying it against llama.cpp's `mma.cuh` (`6f29e65`). Both of this doc's "results" are measurement records, not code records.

## 2. Principle — the GPU mechanism

**mma chains and ILP.** `mma.sync.aligned.m16n8k32` (int8, s32 accumulator) has fixed instruction latency. In a warp, if the next mma's accumulator depends on the previous one's result the two mma serialize; if 16 mma each write their own accumulator (16 **independent chains**), the warp issues all 16 back-to-back before returning to consume the first result — the issue-window depth is the chain count. That is what "chain count ≈ ILP depth" means.

Comparing the chain structures arithmetically. llama: SRAM tile 128 od × ≤128 tokens, 256 threads, 16 m16n8k32 per warp per 32-k chunk, backed by **128 accumulator registers** (int C fragments + float sum). Ours (r8 wide shape): 8 warps × 32×32 per-warp tile, 2×4 = **8 chains** per chunk, ~64 accumulator registers. Half the chains means an issue window half as deep; and occupancy is 1 block/SM in both kernels — **no other warp exists to fill the mma latency holes, so chain depth IS throughput**. llama dares to spend 128 accumulator registers precisely because it targets occupancy 1 and its register budget (255/thread at 256 threads) cannot be exhausted.

**C-fragment register count: `ne = I·J/32`.** Each m16n8k32's C operand is 4 int registers per thread (16×8 output / 32 lanes = 4). The constant r11 had to verify: is llama.cpp's `tile<16,8,int>::ne` 2 or 4? If `ne = I·J/64 = 2`, one `tile::mma()` issues two hardware mma, 16 calls could be 32 chains or another structure, and r10's reading is void; if `ne = 4`, the 16 chains are confirmed and r12's "16-chain warp tile" plan stands. Verdict: `I·J/64` is the **AMD MFMA branch** (one MFMA covers a larger tile, fewer registers per thread); the NVIDIA Turing+ branch is `I·J/32`, 16×8 → 4.

**ldmatrix vs per-lane loads.** The A fragment (m16n8k32's A operand, 4 ints = 16 bytes per thread) can come from shared memory in one `ldmatrix.m8n8.x4` (1 LDSM instruction, hardware-distributed in mma layout), or each lane can compute its own addresses and issue 4 `LDS.32` (8 load instructions per fragment pair). llama uses LDSM; we were on per-lane loads. Not just instruction count — LDSM's address generation reuses one set of matrix coordinates, so ALU overhead is smaller too.

Breaking that cost down. In the per-lane form each lane first computes "where my 4 ints are": row from `lane >> 2`, column from `lane & 3`, times the chunk's row pitch — 4 integer ALU plus 4 narrow loads per lane per fragment, and the narrow loads can still hit shared-memory bank conflicts. LDSM replaces the whole set with one instruction: the lane supplies a base address (distributed within the warp by convention), the hardware does the rest. r10 listed this as the second residual item; r12 cashed it in, and r14 later applied the same weapon to the B fragments.

**Why the decomposition itself matters.** The q4_K super-block dot product has several equivalent forms: unpack timing (all at staging vs item-by-item at mma), dmin fold timing (once per (row, chunk) vs repeatedly per accumulator), B-side form (keep nibbles vs pre-convert to signed int8). Different organizations, different instruction mixes. Only after r10 aligned all these axes with llama and performance did not move an inch did the elimination carry force.

Spelling out llama's decomposition. q4_K's scale is two-stage: the super-block scale `d` (f16) and the per-32-k-chunk correction `dmin`. llama has staging unpack the nibbles into per-k signed int8 and pre-fold the dmin term in one pass, so the compute side does only two FMAs per accumulator — one for the weight scale `dsv`, one for the correction `dmv`. Our raw-byte kernel of the time (excerpt B's shape) left the nibble unpack at mma time item by item, mixing SHF/AND-class integer ALU into the compute-side stream. The MAC totals of the two organizations are exactly equal; what differs is **the composition of the support instructions** — r10's question: if that composition is also made identical, does the 5× gap survive?

**Why the 462–468 band.** In throughput terms: the r9 window's 481 tok/s corresponds to MMQ ~6.4 TMAC/s, and r10's 462–468 lands in the same band — the port changed no throughput number. Meanwhile llama's same decomposition runs ~30 TMAC/s: **the same math, 5× the machine time**. This "flat" is more informative than any +x%: it crosses the whole "math" line off the residual list.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

r10 changed **only the math organization**, no shape or staging parameter: the host is the sync-wide kernel r9 had just measured (128×64 tile, single buffer 54 KB, 2 blocks/SM, plain loads + one `__syncthreads` per tile) — the shape matrix's point closest to narrow (462–466 vs 481), so the post-change delta is directly comparable. Textbook **controlled variables**: change the tile to llama's 128×128 at the same time and any gain is unattributable.

A second reason for sync-wide: it is the only point in r9's matrix that is "structurally close to llama (synchronous staging, no cp.async pipeline) yet not performance-competitive". r9's "copy the instruction model, not the shape" is purest here — staging was already synchronized, so the remaining differences concentrate in the inner loop, exactly what r10 wanted to isolate.

The ported decomposition (aligned item-by-item with llama's `vec_dot_q4_K_q8_1`): 1. **nibble unpack + dmin fold move into staging**, once per (row, chunk) — the compute side never sees a nibble; 2. **the compute side only applies pre-loaded scales** (two FMAs per accumulator, weight-side scale registers pre-loaded); 3. **the B side stays per-k signed int8** (unpacked signed bytes go straight into the mma).

Point 3 in full: q4_K's nibbles are unsigned 4-bit, and feeding them straight into the `s8` mma reads the high nibble with the wrong sign; llama's decomposition splits the two nibbles into separate signed byte streams at staging (low nibble `& 0x0F`, high nibble `>> 4`), the dmin correction carrying the unsigned-offset compensation. Our old kernel did the same work at mma time (excerpt B's `(sg & 1) ? ((n0 >> 4) & 0x0F0F0F0Fu) : (n0 & 0x0F0F0F0Fu)`) — **redone for every fragment of every chunk**; after the port the three-line expression leaves the compute loop and appears once in staging.

### 3.2 Key code

> ⚠️ **Code survival note**: the sync-wide kernel r10 actually edited **is no longer in the current tree** (hook revert, no code commit). The excerpts come from two verifiable locations: (a) the current tree's `mmq_raw_nt_kernel` inner loop — the narrow sibling kernel that survived r12 (plus a rank-1 fold in r16), same structural lineage as the r10 era, shown as the "pre-port" chain-structure class; (b) llama.cpp's `mma.cuh` — r11's verification target. r10's decomposition itself is reconstructed from context per the redo recipe; r12's commit (doc 16) is its true landed form.

**Excerpt A · the mma chain unit (current tree, `mmq_mma_k32`)** — the hardware contract r11 verified is, in our code, this one wrapper: one instruction, 4 C registers accumulated in place:

```cuda
// src/cuda_kernels.cu (current tree)
__device__ __forceinline__ void mmq_mma_k32(int* d, const int* a, const int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])   // 4 C registers = I·J/32
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
```

Each call consumes one m16n8k32: 16×8×32 MAC / 32 lanes. Chain count = the number of calls per chunk whose accumulators do not overlap.

**Excerpt B · the surviving representative of the pre-port chain structure (current tree narrow kernel inner loop)** — note the A fragments are per-lane groups of 4 narrow loads (the "8 LDS.32 per fragment pair" class), accumulators grouped by (B fragment × A fragment) into 4 `int[4]`, 4 mma per chunk:

```cuda
// src/cuda_kernels.cu (current tree, mmq_raw_nt_kernel after r16; same structural class as the r10 era)
for (int kd = 0; kd < KDR; kd++) {
    ...
    // A fragments: int8 lane words straight out of the raw chunk.
    int a[2][4], b[2][2];
    int clow[2][2][4], chigh[2][2][4];
    #pragma unroll
    for (int h = 0; h < 2; h++) {
        const int r0 = i0w + h * 16 + (lane >> 2);
        const uint8_t* p0 = qat + (size_t)r0 * 40 + 4;
        a[h][0] = *(const int*)(p0 + 4 * (lane & 3));        // per-lane narrow loads:
        a[h][1] = *(const int*)(p1 + 4 * (lane & 3));        // 4 LDS.32 per A fragment
        a[h][2] = *(const int*)(p0 + 4 * ((lane & 3) + 4));  // (llama's counterpart is
        a[h][3] = *(const int*)(p1 + 4 * ((lane & 3) + 4));  //  1 ldmatrix.x4)
    }
    // B fragments: unpack the raw nibbles in registers.
    #pragma unroll
    for (int nh = 0; nh < 2; nh++) {
        uint32_t n0 = *(const uint32_t*)(rb8 + ...);         // nibbles unpacked at mma time
        b[nh][0] = (int)((sg & 1) ? ((n0 >> 4) & 0x0F0F0F0Fu)
                                  : (n0 & 0x0F0F0F0Fu));
    }
    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            mmq_mma_k32(clow[nh][h], a[h], b[nh]);           // 4 independent chains/chunk
    ...
}
```

r10's port replaced this segment's B side: unpack and dmin fold move into staging (once per (row, chunk)); the compute-side `b[]` becomes the staged signed int8 directly, scales applied as two pre-loaded FMAs. **The reconstructed shape (rebuilt per the redo recipe, not surviving code)**: staging produces `int8 b8[MMQ_BI][32]` (per-k signed) plus the pre-folded `dsv/dmv`, and the compute loop body shrinks to "4 mma + 2×8 FMA". Its only difference from excerpt B is the instruction mix — exactly the variable r10 isolated.

**Excerpt B′ · the half of the port left untouched: the two-term rescale (current tree narrow kernel)** — however the decomposition is organized, every accumulator must multiply the weight scale (`dsv` term) and the dmin correction (`dmv` term); the port moved the B side into staging but kept this scale-application chain, aligned item-by-item with llama's counterpart:

```cuda
// src/cuda_kernels.cu (current tree, post-r16 state; same structure in the r10 era)
// rescale: identical math/layout to the R1 kernel; A-side
// d/ssum come straight from the raw chunk.
float da_q[4];
int sa_q[4];
#pragma unroll
for (int t4 = 0; t4 < 4; t4++) {
    const uint8_t* at = qat + (size_t)(i0w + (lane >> 2) + t4 * 8) * 40;
    da_q[t4] = h2f(*(const uint16_t*)at);            // A-side super-block scale d (f16)
    sa_q[t4] = (int)*(const uint32_t*)(at + 36);     // A-side ssum (i32)
}
const float dma[4] = { da_q[0] * (float)sa_q[0], ... };
```

The A-side `d/ssum` reads straight from the bare chunk (bytes 0–1 of the 40 B/chunk layout are d, bytes 36–39 are ssum) — same lineage as llama's q8_1 A side, and the part the port need not touch: **r10 aligned only the decomposition's organization; the scale semantics were already identical on both sides**.

**Excerpt C · r11's verification target (llama.cpp `mma.cuh`)** — the two `ne` branches; `I·J/64` is the AMD one:

```cpp
// llama.cpp ggml/src/ggml-cuda/mma.cuh (reference repo, the evidence lines r11 read)
#if defined(AMD_MFMA_AVAILABLE)
        static constexpr int ne = I * J / 64;      // ← the branch r10 first misread (MFMA)
        T x[ne] = {0};
...
#if defined(VOLTA_MMA_AVAILABLE)
        static constexpr int ne = I * J / WARP_SIZE; // NVIDIA half-precision branch: 16·8/32 = 4
        half2 x[ne] = {{0.0f, 0.0f}};
```

The int8 tile's NVIDIA branch (`tile<I,J,int>`) is `ne = I·J/32`, likewise **4** for tile<16,8> — strictly matching excerpt A's 4 C registers. 16 `tile::mma()` calls = 16 hardware mma = 16 independent chains. r10's reading stands.

### 3.3 Pitfalls

- **The post-checkout hook revert swallowed the working tree**. The day's kernel edits were restored by the hook after measurement completed; the code entered no commit. The salvage was writing the redo recipe and all numbers into docs commit `025a69f`. The lesson is procedural: the first action when a measurement session ends is to commit (even docs-only) — code can be redone, numbers cannot.
- **Measurement validity must be argued separately**. The hook revert happened **after** measurement, and the interleaved A/B numbers were recorded into the commit message on the spot — so "the code is gone" does not mean "the numbers are suspect". But it is a question that must be answered explicitly: had the revert come before measurement, this doc would be void. The master table's Status column states exactly this: "REVERTED (edits lost to a post-checkout hook; **measurements valid**)".
- **The `ne` reading: wrong first, verified later**. r10's record justified "16 chains" via the reference's `tile<16,8,int>` — but the `ne = I·J/64 = 2` it read contradicted the 4-register C fragment. r11's first commit (`83e5580`) **publicly suspended** the contradiction instead of glossing over it; the second (`6f29e65`) located the `#if defined(AMD_MFMA_AVAILABLE)` branch and resolved it. A compile-time-constant branch selection decided whether the entire r12 plan stood.

## 4. Verification

- **Parity gate (greedy token identity)**: the port changes only instruction organization, not the numeric path, so output must be token-for-token identical — defends against "optimization changed the math". r10: parity green.
- **Suite (169/0)**: whole-model regression — defends against the port breaking non-q4_K paths (the decomposition entered only the sync-wide kernel, but the suite runs everything).
- **Same-window interleaved A/B**: 462–468 vs narrow's 470 interleaved within one session window, medians taken — defends against cross-session machine drift (forerunner of the r59b lesson, already consciously practiced here).
- **Default-path isolation**: the `MINFER_MMQ=1` opt-in gate keeps the f16 default path untouched — defends against experiments polluting production.
- **r11's verification is static, but still verification**: the `ne` reading runs no performance gate; it reconciles against the hardware ISA contract — excerpt A's `{%0,%1,%2,%3}` 4-register C operand is our code-side evidence, the `mma.cuh` branch the reference-side evidence, and both must yield the same number (4). Defends against deriving a plan from a wrong hardware model.

A note on this doc's special status: a normal step verifies **surviving code** (byte-exact dumps, re-runnable greedy streams); this doc verifies a **measurement record** — the numbers and redo recipe inside three docs commits. The code cannot be re-verified (the hook restored it), which is exactly why every number went into the commit message: the record is the archive. This doc's code has **no** surviving verification target: all three commits are docs-only; excerpts A/B's line references were checked against the current tree and excerpt C against the reference repo; r10's own code shape rests only on the redo recipe's text (flagged in §3.2).

## 5. Results

**r10 (decomposition port, REVERTED / measurement valid)**: sync-wide + llama decomposition = **462–468 tok/s** @2K, control narrow 470 — delta inside the noise band, **~6.4 TMAC/s on both sides**, while llama's same decomposition runs ~30 TMAC/s. Veto mechanism: this was a **controlled experiment** whose "no difference" outcome is itself the conclusion — the math decomposition is eliminated and the residual converges to three items: 1. **ILP depth** — llama 16 independent mma chains + 128 accumulator registers, ours 8 chains / ~64 registers; 2. **ldmatrix A staging** — 1 LDSM vs 8 LDS.32 per fragment pair; 3. **tile 128×128** (llama) vs 128×64 (ours).

The revert itself carried no numeric cost (the code never survived anyway); the "under what future conditions is a retry worthwhile" answer is that r12 executed the redo recipe directly — no second r10 needed.

**r11 (MEAS-ONLY)**: no performance numbers. The output is two definitive conclusions: `ne = I·J/32 = 4` (the NVIDIA Turing+ branch), and the 16-chain reading stands. Of r10's three residuals, (1) and (2) both depend on this reading — r11 is r12's **prerequisite proof**, not an optional footnote.

Campaign-level significance: r9 eliminated the shape axis, r10 the math axis; the one suspect class their eliminations converge on — **instruction-level structure (ILP depth, load width)** — is exactly r12's target. r12's subsequent 2.3× is that reasoning chain cashed in.

Matching the redo recipe against r12's execution item by item shows the complete "handoff" the r10 record produced:

| r10's residual | r12's execution |
|---|---|
| ILP depth: 8 chains / ~64 accumulator registers | 16-chain warp tile: `clow[8][2][4]` + `sum[64]` = 128 accumulator-class registers |
| ldmatrix A staging: 8 LDS.32 per fragment pair | A fragments loaded per group by `ldmatrix.m8n8.x4` (8 groups covering the 128-token rows) |
| Tile 128×128 | block tile 128 tokens × 128 od (the host had been 128×64) |

The third item especially shows r10/r11's value: without r11 pinning `ne = 4`, "16 chains" is an unverified mental calculation, and r12 would not have dared fill the register budget (128 accumulators) to the brim.

## 6. Lessons

1. **When the same math runs 5× slower, the difference is not in the math**: align the decompositions with one controlled experiment first (even if the outcome is "flat") to converge the suspect class to instruction-level structure — far cheaper than intuition-driven kernel tinkering.
2. **A negative result's only vehicle is the record**: code can be swallowed by a hook, but interleaved A/B numbers written into a docs commit survive; commit the record first when a measurement session ends.
3. **Read ISA details against the correct backend branch**: `I·J/64` (AMD MFMA) and `I·J/32` (NVIDIA) differ by 2×, and that decides whether "16 chains" — the whole next-step plan — stands; publicly suspending the contradiction and then verifying beats glossing forward.

← 14 · [Index](./README.md) · 16 →
