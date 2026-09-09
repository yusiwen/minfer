# 31 · r28 — Direction-A raw-nibble NB kernel, 2 blocks/SM (LANDED)

> **Result**: the companion kernel `mmq_raw_nb_kernel` (64 tok × 128 od,
> KD=8-native, smem 45,056 B → **2 blocks/SM**; the wide kernel is 98,304 B → 1):
> whole-prefill median **1375.2 → 1410.4 (+2.56%, 5/5 positive; the earlier batch
> +2.43%, 3/3)**, clearing the +1.5% bar set by r24; parity 1.5e-5..9.9e-5 (pure
> f32 rounding); greedy-32 byte-identical; ncu:
> `sm__warps_active.avg.per_cycle_active` **16.17 ≈ 4.04 warps/sched** (2
> blocks/SM confirmed), long_scoreboard 2.92 → 2.03, issue_active 25 → 37.36%;
> **123 regs / 0 spill**. r25's verdict inverted: occupancy is the bound
> resource — and it can be bought back with smem.
> **Commit**: `0957a08` (design doc `2f783a3`). **Date**: 2026-09-04.

> **Code provenance note (STYLE rule 0)**: the kernel/launcher/dispatch excerpts
> come from the **current tree** (Grep + bounded Reads). The current tree = the
> r28 landed form + two later small changes, both flagged in place below:
> ① the `#pragma unroll` on the kd loop (landed in r29); ② sda_q repacked to one
> uint32 per token (r31, smem 45,056 → 43,008 B). The r28-era dispatch gate is
> taken from a narrow-range diff of `git show 0957a08 -- src/cuda.rs`.

## 1. Background — where things stood

r24 had closed the scheduling-structure family and r25's census delivered the
paradigm verdict: the wide kernel is **issue/occupancy-bound** — 98,304 B smem →
1 block/SM → 8 warps/SM → ~2 warps/sched; with both resident warps stalled
together on long_scoreboard the issue slots spin empty (issue 0.25 vs llama 0.42
at warps_active 2.00 on both sides). That verdict in turn pushed a never-touched
lever onto the stage: **occupancy itself can be bought back — paid for in smem**.

The largest slice of those 98,304 B is exactly the B (weight) plane: qb8's
**expanded** per-k int8 storage (1 byte per nibble, slot-major 48 B slots) takes
8 × 128 × 48 = **49,152 B — exactly half**. llama's MMQ, by contrast, keeps the
weights as **raw nibbles** (2 nibbles/byte, `x_qs[...] = (qs0>>0) & 0x0F0F0F0F`,
unpacked only at use time). Direction A's bet took shape from this: swap the B
smem plane for a raw-packed qs plane, accept the small ALU cost of unpacking
inside the inner loop, and squeeze the per-block budget to ≤ ~49.5 KB → 2
blocks/SM → ~4 warps/sched. r25 had already proven "−38% instruction cut,
<0.5% wall-clock", so "+a few percent of instructions" should also be wall-inert
**provided occupancy lands** — **both ends are functions of occupancy; this bet
is placed on occupancy**.

The danger in this step was numerics: nibble-layout mapping is the breeding
ground of the r13-era "82.896 max-diff" parity pattern (wrong nibble position,
wrong sign extension, wrong dmin fold — any one of them produces garbage diffs
at the 1e0 scale). The design doc (`2f783a3`, §11.5 risk table #1) listed
"wrong B-fragment nibble layout" as the top risk and mandated **standalone
verification before integration**.

## 2. Principle — the GPU mechanism

### 2.1 The occupancy arithmetic: why 45,056 B is exactly 2 blocks/SM

On consumer-class parts like GB10 the per-SM shared-memory budget is on the
order of ~99-100 KB (r7's measured "~99KB opt-in cap" is the same ballpark; the
design doc §11.1 rule of thumb: **per-block ≤ 49.5 KB ⇒ 2 blocks/SM**).
Candidate geometries (KD=8, KDR=8):

| Geometry (T×O) | QA8 | SDA | QB_exp | SDS | exp total | QB_raw | **raw total** | blocks/SM (exp / raw) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **64×128** | 16,384 | 4,096 | 49,152 | 8,192 | 77,824 | 16,384 | **45,056** | 1 / **2** |
| 128×64 | 32,768 | 8,192 | 24,576 | 4,096 | 69,632 | 8,192 | **53,248** | 1 / 1 |
| 64×64 | 16,384 | 4,096 | 24,576 | 4,096 | 49,152 | 8,192 | 32,768 | 2 / 2–3 |

The wide kernel 128×128 + QB_exp = 98,304 B → 1 block. **64×128 + raw** is the
only shape satisfying both constraints at once: (a) 2 blocks/SM; (b) **the od
tile stays 128**. The latter is dictated by the A/B re-read arithmetic (x-tile
row 26: A 327 MB vs B 152 MB = 2.1:1) — A re-reads scale with `od/O`, B re-reads
with `nt/T`; the dominant stream is A, so O cannot shrink: 64×64 saves more smem
but doubles `od/O`, driving the dominant stream 327 → 654 MB — strictly worse;
64×128 merely sacrifices the smaller stream (B re-reads 152 → 304 MB, already
absorbed by L2).

### 2.2 Why raw nibble halves the plane, and where the cost lands

- **Halving**: qb8 expanded = 1 byte per nibble (48 B slot/row); a raw qs plane
  = 128 B per od row holding 256 nibbles (2/byte) → B plane 49,152 → 16,384 B.
  This is the **inverse trade** of llama's scheme: it unpacks at stage time
  (x_qs masking), we **mask at use time** (`& 0x0F`) — only that way does the
  smem saving materialize.
- **The cost** (quantified with the r25 census, §11.3): the census proved minfer
  was already using the cheaper ldmatrix B path (LDSM 8,064/tile vs theirs
  1,792). Option 2 moves B back to plain LDS + unpack: ~30-60 extra ALU per
  (warp, chunk); the hedge is that the A side drops from 8 LDSM per chunk to 4
  (T 64 → 4 token groups). Net instructions **+3-6%** — an order of magnitude
  smaller than the −38% cut r25 proved wall-inert, **contingent on the occupancy
  lever firing**.
- **Warp shape** (§11.2): 8 warps × 16 od-rows = 128 od; each warp exclusively
  owns 16 od rows and reads all 64 tokens; mma m16n8k32 maps m=token, n=od;
  8 independent mma chains per 32-k chunk (4 A-frags × 2 B-frags), `sum[32]`
  (down from 64); register estimate ~110-130 (avoiding r22 Lever-2's 255-spill
  cliff).

### 2.3 The numerics contract: unsigned nibble + rank-1 two-term rescale

mma consumes the **unsigned 0..15 nibble** directly as the int8 B operand (the
high nibble is zero ⇒ always positive); the integer accumulator holds
`C_int = Σ_k nib(k)·act(k)`; the per-chunk fp32 fold is the exact r15 two-term
form — `sum += da·dsv·C_int + dma·dmv`, where `dsv = d·sc` and `dmv = −dmin·m`.
This is precisely the dequant form of `d·s·nib − dmin·m`. **A centered fold of
the shape `(nib − m)` is forbidden anywhere**: q4_K's dmin offset is multiplied
by the sub-block scale (`−dmin·m`); a fixed subtraction is a mathematical error
— the root of the 82.896-diff pattern. The fp32 write-back epilogue stores
`C[i·od + j] = sum[...]` directly; no f16 anywhere on the mma→store path.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **A companion kernel, not a wide-kernel replacement**: NB is gated by
  `MINFER_MMQ_RAW_NB=1`; the wide kernel stays byte-identical and remains the
  default raw path — A/B and rollback are free, the risk ceiling is "zero".
- **KD=8-native**: the raw qs plane encodes the **full 256-k super-block**;
  KD=4 is meaningless for it → for KD≠8 the launcher cleanly returns 0 (fall
  back to wide/narrow), never ambiguous.
- **Standalone verification before integration**: the B-fragment nibble mapping
  was derived from the wide kernel's validated ldmatrix path, then given a
  standalone byte-equivalence test (8 sgs × 32 lanes × 4 regs, 0 mismatches) —
  the top risk was cleared before integration.
- **Keep the proven machinery**: r20 split-phase A staging, the r22 XOR
  swizzle, and the r15 two-term rescale are kept as-is — NB changes only
  "kernel shape + B representation", stacking no new variables.
- **Pre-registered kill criteria** (§11.7): parity fails three fixes in a row,
  or occupancy arrives (ncu reads ~4 warps/sched) but wall-clock does not move
  (falsify and close Direction A), or KD=8 register spill proves incurable —
  any hit pulls the plug; no "let's try again" gray zone.

### 3.2 Key code

**Kernel signature and smem map** (`src/cuda_kernels.cu:6199-6233`; the
Q-major sda_q layout in the comment is the post-r31 form; at r28 it was
uint2-per-token, 4,096 B):

```cuda
__global__ void __launch_bounds__(256) mmq_raw_nb_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
    extern __shared__ uint8_t mmq_nb_sh[];
    // Single-buffer sync-staged, KD=8 totals 43,008 B -> 2 blocks/SM:
    //   qa8     [KDR][64][32]   chunk q8 planes (r22 XOR swizzle, r20 split)
    //   sda_q   [KDR][64]         (d f16 | ssum i16) packed, one uint32 per
    //                             token, Q-MAJOR: ...
    //   qb_raw  [128][128]        raw GGUF qs plane (2 nibbles/byte, full
    //                             super-block per od-row)
    //   sds     [KDR][128] float2 (d | dmin*m): r15 rank-1 rescale terms
    uint8_t* qa8 = mmq_nb_sh;
    uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_NBI * 32);
    uint8_t* qb_raw = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_NBI);
    float2* sds = reinterpret_cast<float2*>(qb_raw + MMQ_NBJ * 128);
```

**A-side staging: r20 split-phase + r22 XOR swizzle kept as-is**
(`src/cuda_kernels.cu`, inside RAW_STAGE_NB; excerpt of the LDG batch and the
swizzled store):

```cuda
            unsigned av[KDR * 2];   /* 8 words x 64 tok x KDR / 256 thr */
            _Pragma("unroll")
            for (int i = 0; i < KDR * 2; ++i) {          /* LDGs issued first, batched */
                const int x = threadIdx.x + i * 256;
                const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),
                          kd = x / (8 * MMQ_NBI);        /* KDR=8, 8*NBI=512 */
                const int tok = i0 + r, c = (kt) * KDR + kd;
                unsigned v = 0;
                if (tok < nt && c < nchunk)
                    v = *(const unsigned*)(q8x
                        + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);
                av[i] = v;
            }
            ...
                const int R = kd * MMQ_NBI + r;           /* r22 XOR swizzle store */
                *(unsigned*)(qa8 + (size_t)(R & ~3) * 32
                    + (size_t)(((((R & 3) << 1) + (u >> 2))
                                ^ ((R >> 2) & 7)) << 4)
                    + (size_t)(u & 3) * 4) = av[i];
```

**B-side staging: the raw qs plane as a pure bulk copy (zero staging ALU) +
the SDS two-term form** (`src/cuda_kernels.cu`, excerpt):

```cuda
        /* ---- B: bulk raw qs super-block copy (r18-style, no staging ALU) */
        for (int off = threadIdx.x; off < MMQ_NBJ * 8; off += blockDim.x) {
            const int jj = off >> 3, c8 = off & 7;
            const int j = j0 + jj;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (j < od && sb < nsb)
                v = *(const uint4*)(W + (size_t)j * ((size_t)nsb * 144)
                    + (size_t)sb * 144 + 16 + (size_t)c8 * 16);
            *(uint4*)(qb_raw + (size_t)jj * 128 + (size_t)c8 * 16) = v;
        }
        /* ---- B: SDS per-(chunk, od-row) rank-1 rescale terms ---- */
        ...
                dv = d * (float)sc;            /* d·sc    */
                mv = -(dmin * (float)m);       /* −dmin·m: the two-term form, never folded */
            sds[(size_t)kd * MMQ_NBJ + r] = make_float2(dv, mv);
```

**The core: the inner-loop raw-nibble B-fragment unpack**
(`src/cuda_kernels.cu:6358-6374` — where all of this step's risk and all of its
payoff live):

```cuda
            // B fragments: raw-nibble in-loop unpack (validated == wide kernel
            // ldmatrix). reg0 = qs[(sg>>1)*32 + (l&3)*4 + 0..3],
            //            reg1 = qs[(sg>>1)*32 + 16 + (l&3)*4 + 0..3].
            {
                const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
                const unsigned M = 0x0F0F0F0Fu;
                #pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    const int jj = j0w + nh * 8 + (lane >> 2);
                    const uint8_t* qs = qb_raw + (size_t)jj * 128;
                    const uint32_t* q0 = (const uint32_t*)(qs + p * 32 + lm3 * 4);
                    const uint32_t* q1 = (const uint32_t*)(qs + p * 32 + 16 + lm3 * 4);
                    uint32_t v0 = *q0, v1 = *q1;
                    b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));
                    b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
                }
            }
```

Key points: `sg & 1` picks the high/low nibble (one 32-bit word packs 8
nibbles, half low and half high); after masking there is **no sign extension**
— the unsigned 0..15 value is used as int8, and centering is left entirely to
SDS's two-term rescale. `is_hi`/`p`/`lm3` are all loop-invariant within the kd
loop; r29's unroll (the outer `#pragma unroll` on the next line, current-tree
line 6337) folds them into compile-time constants.

**8 independent mma chains + the two-term rescale fold** (`src/cuda_kernels.cu`,
excerpt):

```cuda
            #pragma unroll
            for (int g = 0; g < 4; g++)          /* 4 A-frags x 2 B-frags, */
                #pragma unroll                    /* all C fragments live at once */
                for (int nh = 0; nh < 2; nh++)
                    mmq_mma_k32(clow[g][nh], a[g], b[nh]);
            ...
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[l >> 1];
                        const int idx = (g * 2 + nh) * 4 + l;
                        sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];
                        sum[idx] += dma[l >> 1] * dmv[nh][l & 1];
                    }   /* sum += da·dsv·C_int + dma·dmv — the r15 two-term form, verbatim */
```

**fp32 epilogue** (`src/cuda_kernels.cu:6434-6444`):

```cuda
    for (int g = 0; g < 4; g++)
        for (int nh = 0; nh < 2; nh++)
            for (int l = 0; l < 4; l++) {
                const int i = i0 + g * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];
            }
```

**Launcher: smem arithmetic + every guard "cleanly returns 0"**
(`src/cuda_kernels.cu:7164-7193`; 43,008 is the post-r31 number; at r28 it was
45,056 = sda_q 4,096 B):

```cuda
extern "C" int launch_mmq_raw_nb_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id;
    // 64-token x 128-od block tile, KD=8 native. Raw qs plane + single-buffer
    // staging = 43,008 B (r31 q-major sda repack shrinks sda_q 4,096 ->
    // 2,048 B) >>> 2 blocks/SM on GB10. KD!=8 is inapplicable to the
    // raw-nibble variant (the qs plane encodes a FULL 256-k super-block), so
    // clean-fallback (return 0) to the wide kernel. smem/reg guards return 0
    // on any cap failure (never silently launch over cap).
    if (kd != 8) return 0;
    const int smem = 8 * MMQ_NBI * 32   // qa8
                   + 8 * MMQ_NBI * 4    // sda_q (one uint32 per token)
                   + MMQ_NBJ * 128      // qb_raw
                   + 8 * MMQ_NBJ * 8;   // sds (float2 = 8B)
    dim3 grid((nt + MMQ_NBI - 1) / MMQ_NBI, (od + MMQ_NBJ - 1) / MMQ_NBJ);
    cudaFuncSetAttribute(reinterpret_cast<const void*>(&mmq_raw_nb_kernel<8>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { cudaGetLastError(); return 0; }
    mmq_raw_nb_kernel<8><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, ...); return 0; }
    return 1;
}
```

**Dispatch gate: r28's before/after** (narrow range of
`git show 0957a08 -- src/cuda.rs`):

```rust
// BEFORE: the RAW path had only the wide/narrow tiers
let wide_ok = wide && launch_mmq_raw_wide_nt(...) == 1;
if !wide_ok { launch_mmq_raw_nt(...); }

// AFTER: NB is inserted at the front; on failure each tier falls back cleanly (return 0 = did not run, not an error)
let nb = std::env::var("MINFER_MMQ_RAW_NB").as_deref() == Ok("1");
let nb_debug = std::env::var("MINFER_MMQ_RAW_NB_DEBUG").as_deref() == Ok("1");
// Direction-A NB raw-nibble kernel is KD=8-native; it activates
// only under the full MMQ gate set (MINFER_MMQ=1 + MINFER_MMQ_RAW=1 here).
// launcher returns 0 on KD!=8 or smem/reg cap failure
// -> clean fallback to the wide/narrow raw path below.
let nb_ok = nb && launch_mmq_raw_nb_nt(...) == 1;
if nb_ok && nb_debug { eprintln!("minfer/cuda: mmq raw NB kernel active (KD=8)"); }
if !nb_ok { /* wide → narrow, unchanged */ }
```

(This segment has since evolved further in the current tree: r34's A-transpose
prepass path now sits at the front, with plain NB as its fallback arm —
`src/cuda.rs:3352-3380`; the NB-BT variant carries the default path.)

### 3.3 Pitfalls

- **The top risk was dismantled up front**: the nibble-layout mapping was not
  "tested in passing" inside the kernel — the mapping was derived from the wide
  kernel's validated ldmatrix B path, given a standalone byte-equivalence check
  (full comparison over 8 sgs × 32 lanes × 4 regs, 0 mismatches), and only then
  integrated. r13's 82.896 pattern (layout error → 1e0 garbage diff) was
  stopped before integration.
- **The register cliff**: the inner-loop B-unpack temps + `sum[32]` risked
  pushing ptxas to 255 regs + spill (r22 Lever-2's failure mode). Landed at
  123 regs / 0 spill — occupancy was not bitten back by registers.
- **The semantics of the KD=4 arm**: the raw qs plane = full 256-k
  super-block; KD=4 does not apply; the parity matrix's KD=4 arm falls back
  cleanly to the wide kernel. This must be written down, otherwise someone will
  misread "the KD=4 numbers = wide kernel" as an NB result.
- **The ghost of silent failure lives on**: r7's phantom-2124 lesson is baked
  into the launcher's shape — the attr result is checked, over-cap never
  launches silently, a returned 0 triggers explicit fallback, plus the
  `MINFER_MMQ_RAW_NB_DEBUG=1` liveness label ("NB kernel active").

## 4. Verification

Six Phase-2 gates (scheduled by the design doc §11.6, all green) plus one
up-front gate:

- **Standalone B-unpack byte equivalence (before integration)**: full
  comparison against the wide kernel's ldmatrix B fragments (8 sgs × 32 lanes ×
  4 regs, 0 mismatches) — defends against nibble-layout mapping errors (the
  source of the 82.896 pattern).
- **ptxas register gate**: `-Xptxas -v` reports 123 regs / 0 spill — defends
  against register spill silently killing the 2-blocks/SM goal itself.
- **Parity**: NB-active `cuda_prefill_mmq` max diff **1.5e-5..9.9e-5** — pure
  f32 rounding magnitude; a layout bug would be ~1e0 (the 82.896 pattern), so
  the magnitude directly identifies the defect class. The KD=4 arm falls back
  cleanly and parity is green.
- **greedy-32 identity**: byte-identical to the default f16 path — defends
  against graph-level/numerics-level breakage.
- **Perf (gate 4, the first formal application of the r24 bar)**: interleaved
  3× median, NB 1410.4 vs re-measured baseline 1375.2 = **+2.56% (5/5
  positive)** ≥ +1.5%; the headline number requires distribution separation.
- **ncu occupancy gate (gate 5)**: `sm__warps_active.avg.per_cycle_active`
  **16.17 = ~4.04 warps/sched → 2 blocks/SM**; long_scoreboard 2.92 → 2.03
  (closing on llama's 1.15); issue_active 25 → 37.36% — the bet's mechanism
  demonstrably happened. This is the pre-registered kill criterion used in
  reverse: had occupancy arrived while wall-clock stayed flat, the entire
  Direction-A line would close.
- **Suite 166/0/3** — defends against regressions.

## 5. Results

- **Wall-clock**: 1375.2 → **1410.4** (+2.56%, 5/5 positive; the earlier batch
  +2.43%, 3/3) — the first clean crossing of the +1.5% landing bar set by r24.
- **Kernel level**: smem 98,304 → 45,056 B; ~2 → **~4.04 warps/sched**;
  long_scoreboard 2.92 → 2.03; issue_active 25 → 37.36%. 123 regs / 0 spill.
  The two kernels coexist; the wide kernel stays the default, NB is
  gate-selected.
- **The verdict inverted**: r25 said "cutting instructions does not move the
  wall"; r28 says "the lever that moves the wall is occupancy, the purchase
  price is smem (the B representation), and the change is +3-6% instructions —
  and that instruction increment, exactly as the mirror image of r25's verdict
  predicted, is wall-inert." Methodology doc #77 fixed this arc into a
  transferable rule: **"Buy occupancy, then cut instructions
  (r13→r25→r28/r29): at 1 block/SM the instruction surplus is real but
  wall-inert; buy occupancy first, and the same instruction cuts start paying
  out (+2.6/+2.8%)."**
- **The arc closes (foreshadowing r29)**: the same kd-unroll lever,
  +0.37/+0.49% at 1 block/SM in r25 (reverted), +2.80% on NB's 2 blocks/SM in
  r29 (landed) — occupancy unlocks instruction cuts, not the other way around.
- The NB kernel has been the chassis of the q4_K line ever since: r29 (unroll)
  and r31 (sda repack) land directly on top of it; r34's A-transpose prepass
  and the r58/r59 BT variants evolve along its geometry (in the current tree
  the default path is carried by NB-BT, with plain NB as the fallback arm).

## 6. Lessons

1. **Occupancy and instruction cuts are an ordering relationship**: at 1
   block/SM, buy occupancy first; any surgery on the instruction stream waits
   until occupancy is up — in the wrong order, a good lever measures as noise.
2. **smem is the occupancy currency on consumer-class parts**: squeeze the
   budget from 98 KB to ≤49.5 KB and blocks/SM doubles; what you squeeze is
   decided by the A/B re-read ratio (2.1:1) — shrink the token dimension, keep
   the od dimension, sacrifice the smaller stream.
3. **Dismantle layout-mapping-class risk up front with standalone byte
   equivalence**: derive the new mapping from a validated path, run the full
   comparison (0 mismatches), then integrate — an order of magnitude cheaper
   than "debugging a 1e0 diff" inside end-to-end parity.
4. **Pre-registered kill criteria let even a null result wind the line down**:
   "occupancy arrives but the wall does not move ⇒ close Direction A" was
   written before the build, so the measured negative did not trigger wobbly
   retries.

---
← [30 · r25 SASS opcode census](30-r25-sass-opcode-census.md) · [Index](./README.md) · [32 · r29 NB kd-loop unroll](32-r29-nb-kd-loop-unroll.md) →
