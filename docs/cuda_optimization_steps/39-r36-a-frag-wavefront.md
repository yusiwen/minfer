# 39 · r36 — A-frag wavefront economics: H1 falsified (MEAS-ONLY, no code change)

> **Result**: ncu injected into the bt kernel successfully for the first time: minfer
> **6.156 vs llama 3.507** shared wavefronts/IMMA (1.76×), with the LDSM share exactly
> **4.00×** — but minfer simultaneously runs **1.85× the wavefronts/s** (39.0 vs 21.1 G/s)
> and a level IMMA rate (**6.33 vs 6.02 G-IMMA/s**). If MIO were truly scarce, both
> kernels would be capped by the same wf/s ceiling — the causal chain breaks, H1 (the
> LDSM→plain-LDS swap) is falsified; H2 (issue-slot deficit) does not exist either. **No
> code change**; the bt kernel is at per-IMMA parity with llama.
> **Commit**: `f44fc44` (docs-only; HEAD untouched). **Date**: 2026-09-05.

## 1. Background — where things stood

r35 had struck the entire "decode ALU" instruction class off the wall-clock lever list:
−130 SASS instructions bought 0.0% wall clock, the mechanism being that those
instructions hide in the IMMA's shadow. With that, only one **named candidate** in the
r34 kernel's census had never been tested under its own metric: **the A-frag (activation
fragment) LDSM supply**.

The candidate's provenance: r32/r33/r35 repeatedly described the residual as "the
inherent combination of A-frag LDSM consumption + fp rescale". And llama.cpp's MMQ
kernel offers a glaring contrast — its B-frags (weights) go through `load_generic`
(plain `LDS`), the source comment saying outright *"faster than load_ldmatrix"*, while
minfer's A-frags all go through `ldmatrix.sync.aligned.m8n8.x4`. Hence this round's
**H1**: *swapping minfer's A-frag supply from LDSM to plain-LDS (load_generic-style)
should reduce shared-memory wavefronts and thereby speed up this IMMA-bound loop* — with
the implicit premise that **MIO (the shared-memory wavefront channel) is a scarce
resource**. Standing beside it, **H2** was the era's other bottleneck theory: *the loop
is issue-slot-deficit (insufficient per-cycle issue rate) bound*.

This round also had a debt to repay: r34/r35's ncu censuses were both missing due to
platform injection failures (r34 landed on the regs/prepass/wall triple of indirect
evidence). For the wavefront argument to stand, the kernel's real counters had to be
obtained first.

## 2. Principle — the GPU mechanism

**What a wavefront is.** The L1TEX data pipe services shared-memory accesses in
wavefronts: one warp-level instruction splits into several 128 B service cycles. A
conflict-free 32-lane LDS is about 1 wavefront; one `ldmatrix.m8n8.x4` moves 4 8×8 b16
tiles = 4×128 B = **512 B = 4.0 wavefronts** (the record's self-consistency check:
20,873,216 wf / 5,218,304 inst = 4.0, matching bit for bit).

**Metric definitions.** The canonical ratio for wavefront pressure is

```
wavefronts/IMMA = Σ smsp__sass_l1tex_data_pipe_lsu_wavefronts_mem_shared_op_{ldsm,ld,st}
                  ÷ smsp__inst_executed_pipe_tensor_subpipe_imma
```

The denominator is IMMA (not thread instructions) because this loop's "output" is the
tensor-core multiply-accumulate; wavefronts are the input serving it.

**The scarcity test (this round's methodological core).** "Consumes more of a resource
per unit of output" and "is limited by that resource" are two different propositions.
The MIO pipe's service rate is a fixed per-SM value — if it were scarce, **any** kernel
would be capped by the same wavefronts/s ceiling, and whoever hits the line first stops.
So the criterion lives on the throughput side:

```
wavefronts/s = (wavefronts/IMMA) × (IMMA/s)
```

If minfer's wf/IMMA is 1.76× while the IMMA rate is level, then wf/s must be 1.76× —
running 1.85× above llama's operating point without slowing down means the ceiling is
still far away and MIO has plenty of headroom. This "throughput accounting" (work/op →
work/s → compare against the ceiling) is the entire mechanism by which this round
falsifies H1.

**Why "changing the access style" cannot save wavefronts.** H1's intuition comes from
llama's plain-LDS being "faster". But an LDSM.x4 moves 512 B = 4 wf, and a conflict-free
plain-LDS over the same 512 B is still 4 wf — **same bytes → same wavefronts**, unless
the geometry changes. H1's original argument compared LDSM.x4 (4 wf) against "a single
tile's plain-LDS" (1 wf) — apples to oranges; llama's own plain-LDS averages 1.60
wf/inst too (B-frags via `load_generic`, equally not single-wavefront).

**llama's real advantage: the A-frag reuse rate.** Per 32-k chunk, llama loads 8 A-frags
that serve 64 mmas (**0.125 LDSM/IMMA**); minfer loads 4 A-frags, each serving only 2
mmas (**0.500 LDSM/IMMA**) — the 4× reuse gap comes from the warp division of labor:
llama's warp iterates od inside the kernel (the same A-frag spans 8 od-tile columns),
while minfer's warp owns a single 16-od narrow strip (`j0w = warp * 16`), so each A-frag
naturally spans fewer mmas. **Reuse is a tiling property, not an access-style property**
— the root cause no LDS swap can fix.

## 3. Implementation

### 3.1 Design choices (why measure instead of write)

- **Unlock injection first, numbers later.** r34/r35's ncu failure was fixed in r36's
  first step: running ncu under the `sudo -n env LD_LIBRARY_PATH=...` prefix made
  injection succeed; without the prefix the driver returns `ERR_NVGPUCTRPERM` /
  *"Unknown Error on device 0"* — exactly the wall r34/r35 hit. Every later P6
  attribution round reuses this prefix.
- **Try H1 under its own metric.** The earlier rounds' lessons (r30: SASS first;
  r32/r33: the compiler has often already done what you intended) all point at the same
  principle — a named candidate must first prove a gap on **the counter where it could
  win**, then prove the gap is the bottleneck; only passing both gates earns code. H1
  passed the first gate (wf/IMMA 1.76×) and died on the spot at the second (throughput
  cap).
- **Contrast configuration**: minfer = qwen2.5-7b q4_k_m, nt=3325 prefill (the bt path);
  llama = `mul_mat_q`, llama-bench `-p 512`, launch 1 each. **The nt values were not
  paired** — the hazard did not surface this round; r37's matched-nt re-test exposed it
  (see §5's correction note).

### 3.2 Key code

The accused A-frag supply (era-tree `mmq_raw_nb_bt_kernel`; r36 changed no code, and the
current tree has evolved past r59 — this is the round's form):

```cuda
const int j0w = warp * 16;          // the warp owns a single 16-od narrow strip ← the structural root of low reuse
…
const unsigned l12m = (unsigned)(lane & 12) * 32;
const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                         ^ ((lane >> 2) & 3)) << 4;
unsigned G[4];                       // r22: precomputed 8 A-frag byte offsets (XOR swizzle)
#pragma unroll
for (int g = 0; g < 4; g++)
    G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);
…
for (int kt = 0; kt < nktile; ++kt) {
    …
    int a[4][4], b[2][2];
    #pragma unroll
    for (int g = 0; g < 4; g++) {            // 4 ldmatrix.x4 per chunk
        const uint8_t* p = qat + G[g];
        unsigned r0_, r1_, r2_, r3_;
        asm volatile(
            "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
            "{%0,%1,%2,%3}, [%4];\n"
            : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
            : "r"((unsigned)__cvta_generic_to_shared(p)));
        a[g][0] = (int)r0_; a[g][1] = (int)r1_;
        a[g][2] = (int)r2_; a[g][3] = (int)r3_;
    }
    /* 8 mmas per chunk: mmq_mma_k32(clow[g][nh], a[g], b[nh]), g=0..3, nh=0..1
       → 4 LDSM / 8 IMMA = 0.500 LDSM/IMMA; each a[g] serves exactly nh's 2 mmas */
```

Arithmetic re-check: 4 `ldmatrix.x4`/chunk × 4 wf = 16 wf over 8 mmas → **2.000
LDSM-wf/IMMA**, matching the ncu reading bit for bit; on llama's side 0.125
LDSM-inst/IMMA × 4 = 0.500 wf/IMMA. The entire 4× gap is derivable from the warp
division of labor — H1's "access style" narrative has no footing in the code.

### 3.3 Pitfalls

- **ERR_NVGPUCTRPERM**: performance counters need driver-level permission; plain
  injection fails outright with *"Unknown Error on device 0"*. The `sudo -n env
  LD_LIBRARY_PATH=...` prefix is the machine-reproducible fix (permissions + the
  injection library path, fixed together).
- **H1's original argument was apples-to-oranges**: LDSM.x4 (4 tiles, 4 wf) versus a
  single-tile plain-LDS (1 wf) — payloads differ 4×, so the comparison is meaningless; a
  fair comparison must lock the bytes.
- **The unpaired-nt measurement planted a landmine**: minfer@3325 vs llama@512 mixed two
  nt scales. This round's per-IMMA conclusion (1.05×) stands, but r37 revealed it holds
  only at prefill-scale nt — at short nt the tile-prologue amortization degrades bt's
  wall/IMMA to 1.43×. Cross-kernel comparisons must always lock nt.

## 4. Verification

- **Metric self-consistency check**: minfer LDSM inst 0.500/IMMA × 4 wf = 2.000
  LDSM-wf/IMMA, matching the counter reading bit for bit — rules out a counter piped to
  the wrong pipe.
- **launch 1 each**: a single launch, no re-entry — rules out CUDA Graph / prewarm
  polluting the counts.
- **Throughput and ratio cross-validation**: 6.156 wf/IMMA × 6.33 G-IMMA/s = 38.97 ≈
  39.0 G wf/s — two independent sources (the ratio and the throughput) close.
- **issue_active cross-check of H2**: 0.457 vs 0.365 (minfer higher) — the issue-slot
  deficit does not exist either; both bottleneck theories cleared in one round.

## 5. Results

**Wavefronts per IMMA** (minfer bt @nt=3325 vs llama `mul_mat_q` @nt=512):

| per-IMMA | minfer bt | llama | ratio |
|---|---:|---:|---:|
| LDSM wf | 2.000 | 0.500 | 4.00× |
| LDS wf | 3.500 | 2.163 | 1.62× |
| ST wf | 0.656 | 0.844 | 0.78× |
| **total wf** | **6.156** | **3.507** | **1.76×** |
| LDSM inst | 0.500 | 0.125 | 4.00× |
| LDS inst | 0.750 | 1.349 | 0.56× |

**The throughput side (the three-line ledger that falsifies H1)**:

| | minfer bt | llama | ratio |
|---|---:|---:|---:|
| wavefronts/s | **39.0 G/s** | 21.1 G/s | **1.85×** |
| IMMA/s | 6.33 G/s | 6.02 G/s | 1.05× (level) |
| issue_active/cyc/sched | 0.457 | 0.365 | minfer higher |

**The ruling**: minfer sustains its per-IMMA tensor rate while running 1.85× the
wavefronts/s — MIO is ~1.85× away from its cap and **is not a scarce resource**; the
extra wavefronts hide in the tensor shadow (the same mechanism as r35's ALU shadow — a
second consecutive round of "delete/swap supply → no wall-clock response"). H1
falsified; H2 at its endpoint (no deficit). **The bt mma kernel is at per-IMMA parity
with llama**, and the remaining prefill gap lies outside addressable mma-structure
levers (per-tile prologue/wave amortization, the quantize prepass, fixup). **No code
change; HEAD untouched.**

**The r37 correction note (a boundary condition for later readers)**: the two sides of
the table above ran different nt (3325 vs 512). r37's matched-nt re-test showed bt's
**wall**/IMMA at llama's 1.43× when nt≈511 — per-IMMA parity holds only at prefill-scale
nt; short nt is diluted by the prologue. Always cite this round's "per-IMMA parity"
conclusion with that nt clause attached.

## 6. Lessons

1. **The throughput-accounting method**: a high work/op ≠ being limited by the resource
   behind that op; compute work/s first, then compare against the resource's fixed-rate
   ceiling — "uses a lot" and "is stuck on it" differ by exactly one capping test.
2. **H1-class access-style swaps die of byte equality**: LDSM.x4 and a conflict-free
   plain-LDS over the same payload are both 512 B = 4 wavefronts; an access-style swap
   has wavefront effects only when the geometry changes.
3. **A-frag reuse is a tiling property**: the 0.125 vs 0.500 LDSM/IMMA gap comes from
   whether the warp iterates od inside the kernel (llama) or each guards a 16-od narrow
   strip (minfer) — moving it requires re-cutting the warp division of labor, not
   swapping a load instruction.
4. **After two consecutive same-mechanism falsifications (r35's ALU shadow, r36's
   wavefront shadow), the "find a faster supply inside the bt kernel" line can close as
   a whole** — the wall-clock gap must live outside the kernel or in other kernels,
   which is exactly where r37's whole-wall attribution starts.

← [38-r35-scale-predecode](38-r35-scale-predecode.md) · [Index](./README.md) · [40-r37-post-parity-attribution](40-r37-post-parity-attribution.md) →

