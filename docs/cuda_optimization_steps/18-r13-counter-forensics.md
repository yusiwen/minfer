# 18 · r13 — counter forensics against llama.cpp (MEAS-ONLY, closed)

> **Result**: the first working ncu session split "we are ~2.6× slower than llama" down to
> hardware-counter level — the per-GMAC warp instruction stream 10.14 M vs 6.06 M (1.67×) is
> the gap's carrier, while IMMA tensor work equals exactly 2×MAC on both sides (mma
> structure irrelevant); two patches purpose-built to fix "bytes/conflicts/store efficiency"
> (FULL −16%, MINIMAL +0.3% noise) disproved all three hypotheses at once by counter-
> evidence.
> **Commit**: `5ca037d` (docs-only; the two kernel patches were restored after measurement
> as cmp-verified, code not committed). **Date**: 2026-09-03.

## 1. Background — where things stood

r12 had landed the 16-chain warp tile + ldmatrix A fragments, jumping the wide kernel from
441–481 to 1020–1058 tok/s (~2.3×), the largest step of P6's structural rewrite. But on the
vs-llama ruler: the same q-proj-class GEMM runs **107.7 µs/GMAC** on minfer wide KD=4 versus
**41.1 µs/GMAC** on llama.cpp — the remaining gap is still ~2.6×, and nobody could say where
it "lives" in the hardware.

The candidate explanations were a long list, each intuitively plausible: is our L2 read
traffic larger? Too many shared-memory bank conflicts? Low store sector efficiency on the C
matrix? A staging barrier structure that is too synchronous? r9–r10 had already given a
"paper" answer — llama's instruction model is ~0.018 inst/MAC/thread versus our 0.133 — but
that was a ratio **derived** from source structure, not **read** off the hardware. Paper
reasoning indicates direction but carries no evidentiary force: it cannot distinguish "more
instructions but hidden by latency" from "more instructions directly lengthening execution".

An earlier step had measured and closed the whole staging-shape family: x-tile (256-token
wide block) −9% reverted, j-tile (A reuse) +2.6% below bar reverted, cp.async double buffer
neutral reverted. All three were "traffic shape" levers, all ineffective. The empty result
was itself a signal — if bytes and access shape are not the binding resource, the only
remaining candidate on the binding-resource list is the SM-side per-MAC instruction stream.
But "only remaining" is not "confirmed".

That is r13's positioning: **attribute the 2.6× gap with first-hand ncu counters, then
disprove with two patches designed as mutually exclusive explanations**. If "byte theory" is
right, the FULL patch that cuts L1 requests by 55% should win; if it loses, byte theory is
out. Without this step, every lever after r14 would be guessing. One engineering constraint
came attached: this was the campaign's **first working ncu session** — getting ncu running
on this GB10 machine and finding the right counter names was itself a pitfalls lesson (see
§3.3).

## 2. Principle — the GPU mechanism

**Why normalize per-GMAC.** The two binaries differ in tile shape, kernel count, and launch
shape; comparing absolute counts is meaningless. Dividing every counter by the GEMM's total
MACs (normalized to "per 10⁹ multiply-accumulates") yields **hardware resources consumed per
unit of math work** — a measure of the kernel's "instructional structure", decoupled from
problem size. Both engines do the same amount of math (the IMMA counts prove this below), so
the entire difference is structural.

**duration's linear law is this case's instrument of judgment.** The execution time of the
SM-side warp instruction stream is approximately

```
duration ≈ warp_inst / (issue_rate × SM × scheduler × clock)
```

The converted slope measured in this session, on **both engines and all data points**, is
the same order (~0.10–0.15 warp-inst/ns, the record's basis), i.e. duration and instruction
count are nearly collinear on the scatter plot. llama's issue efficiency is 0.41 vs our 0.25
— so the 1.67× instruction difference multiplied by the issue difference explains the ~2.6×
time difference exactly. Conversely: if instruction counts were close with a 2.6× time gap,
the answer would live in the stalls; if the IMMA counts differed by 1.67×, the answer would
live in mma structure. The counter matrix turns these three hypotheses into three decidable
readings.

**GB10's counter availability (ncu device id GB20B) sets the forensics checklist.**
- No `dram__*` metric family — the DRAM-level byte/SOL view is unreadable on this device;
  the deepest memory view is the L2 sector counter `lts__t_sectors_aperture_device`.
- int8 tensor instructions do **not** enter the `hmma` (fp16 mma) counters — it reads 0 for
  an int8 kernel; the IMMA sub-pipeline counters must be read instead.
- Available and used here: warp instruction execution counts, IMMA tensor operations, shared
  load/store bank-conflict counts, L2 read/write sectors, L1 request counts, kernel
  duration.

**The FULL/MINIMAL counter-evidence design.** The two patches map onto "byte theory"'s three
sub-hypotheses: FULL = merged A staging (cuts L1 requests/L2 reads) + per-superblock
register scale staging + qb8 row pitch 256→272 B (kills bank conflicts) + float2-ized C
stores (raises store sector efficiency); MINIMAL keeps only the last two. If any of the
three hypotheses were the binding resource, MINIMAL should show at least visible gains and
FULL should be clearly positive. Both measured nothing — all three hypotheses out
simultaneously.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**Why the q-proj-class GEMM as the alignment target.** It is a regularly-shaped near-square
GEMM, both engines run it through the same int8 mma path, and its share of the prefill wall
is stable — a natural "same problem" control. The llama-side alignment leans on llama-
bench's chunking behavior: `-p 2600` is cut into a **nt=512 ubatch**, so llama's q-proj-
class launch runs as grid (48,1,1) at 0.24–0.29 ms each — a real launch ncu can sample
directly, comparable in size to ours, not a synthetic micro-benchmark.

**Why two patches rather than one.** One patch mixing four sub-changes means a win cannot be
attributed and a loss cannot be blamed. FULL/MINIMAL's set difference (merged staging,
register scales) alone carries "byte theory"'s strong hypothesis, and the shared part (272 B
row pitch, float2 C stores) carries the "conflicts/store efficiency" hypothesis. The two
outcomes together (FULL slower, MINIMAL flat) carry more information than any single mixed
patch.

**Why restore immediately after measuring.** Neither patch reached the bar and both
mechanisms were already disproven; keeping them would only pollute the next round's A/B
"baseline" (a lesson re-validated bloodily at r59b — a stale baseline nurtured a fake +26.2%
reading for two weeks). Restoration was confirmed byte-for-byte against HEAD with cmp, and
the variant code was archived in /tmp for later reference.

### 3.2 Key code

r13 sampled the wide kernel's compute loop **as it then stood**. That state has no
standalone commit (restored after measurement); the only citable real-code source is the
**deletion side** of r14's commit `c64cd99`. First, the B-fragment read pattern the counters
measured — 4 scalar LDS.32 per (warp, chunk):

```cuda
// git show c64cd99 deletion side (the r13-era tree; r14 replaced this segment with ldmatrix.x4)
// B fragments: 2 minitiles of the warp's private 16 od-rows;
// the staged bytes are already per-k int8 in element order
// (slot sg of the row), so both words are plain smem loads.
#pragma unroll
for (int nh = 0; nh < 2; nh++) {
    const int jr = j0w + nh * 8 + (lane >> 2);          // this minitile's od row
    const uint8_t* rb8 = qb8 + (size_t)jr * 256 + sg * 32;
    b[nh][0] = *(const int*)(rb8 + 4 * (lane & 3));     // LDS.32
    b[nh][1] = *(const int*)(rb8 + 16 + 4 * (lane & 3));// LDS.32
}
```

The scale and A-side d/ssum reads are likewise narrow scalar streams — the per-chunk shared-
read instruction count was r14's target (record basis: scales 8 LDS.32 per minitile → 2
LDS.128; sda_q 16 LDS.32 per chunk → 8 LDS.64):

```cuda
// also c64cd99's deletion side: the od-column scales fully enumerated, 8 elements each way
// (after ptxas DCE, 8 LDS.32 actually execute per lane per nh)
float dsv[2][8], dmv[2][8];
#pragma unroll
for (int nh = 0; nh < 2; nh++)
    #pragma unroll
    for (int jj = 0; jj < 8; jj++) {
        dsv[nh][jj] = sdst[j0w + nh * 8 + jj];   // sds: one f32 per row
        dmv[nh][jj] = sdmt[j0w + nh * 8 + jj];   // sdm: separate plane
    }
...
// A-side d/ssum: two scalar reads per (chunk, g); the token pair (t, t+8) sits 16B apart
#pragma unroll
for (int t4 = 0; t4 < 2; t4++) {
    const unsigned pk = *(const unsigned*)(sda_q
        + (size_t)kd * MMQ_WBI * 2
          + (g * 16 + (lane >> 2) + t4 * 8) * 2);
    da_q[t4] = h2f((unsigned short)(pk & 0xFFFF));
    sa_q[t4] = (int)(short)(pk >> 16);
}
```

**Honestly flagged**: the FULL/MINIMAL patch implementations existed only in the r13
session's working tree, restored after measurement and never committed (`5ca037d` is docs-
only), so per this directory's conventions no code excerpt can be provided here; their
composition is the four-item list in §2 (merged A staging, per-superblock register scale
staging, qb8 row pitch 256→272 B, float2 C stores), with numbers in §5.

### 3.3 Pitfalls

1. **ncu's launch form**. The invocation that finally worked on this machine is `sudo -n env
   LD_LIBRARY_PATH=... ncu ...` — sudo strips the user environment, and the CUDA toolchain's
   shared-library paths must be carried through explicitly with `env`, or ncu's injection
   fails.
2. **The entire `dram__*` family is absent**. GB20B has no DRAM counters; a session hunting
   "decisive DRAM-traffic evidence" comes back empty-handed. Forensics must land on
   `lts__t_sectors_aperture_device` (L2 sectors, split by aperture).
3. **int8 mma is not in the hmma counters**. hmma reads 0 for both kernels — unfamiliarity
   would misread it as "the tensor cores are idle". int8 mma volume lives under the IMMA
   sub-pipeline counters.
4. **The 66 MB/GMAC L2 write-sector mystery**. The minfer side showed ~66 MB/GMAC of L2
   write sectors with no corresponding writer findable in the source; the float2 C-store
   patch (which explicitly changes the store shape) had zero effect on it. Suspended
   unexplained at the time, later classified as a counter artifact / denominator effect.
   Lesson: before concluding, verify an unattributed counter reading with a patch that
   "should change it if fixed".

## 4. Verification

- **Parity gate (once per patch)**: both FULL and MINIMAL passed bitwise parity before being
  allowed into A/B — this session is a counter-evidence design; had a patch computed
  wrongly, "slower" would no longer point at a mechanism conclusion (defends: misreading an
  implementation bug as a mechanism veto).
- **Restoration check**: after measurement the kernel was cmp'd byte-for-byte against HEAD
  and confirmed identical (defends: a half-restored state polluting every later A/B's
  baseline).
- **suite 169/0**: full regression green after restoration (defends: collateral damage from
  the restoration).
- **Counter cross-consistency**: IMMA equals 2×MAC on both sides (proving the two control
  kernels really do the same math, legitimizing the normalization); FULL's inst +33% →
  duration +35% lands on §2's linear law (proving the instruction→time chain is trustworthy,
  not a sampling flutter).
- **llama-side launch authenticity**: the sample target is llama-bench's real nt=512 ubatch
  launch (grid (48,1,1), 0.24–0.29 ms), not a self-made benchmark (defends: synthetic and
  real launches being structurally incomparable).

## 5. Results

**The forensics table (q-proj class, per-GMAC normalized)**:

| Counter | minfer wide KD=4 | llama.cpp | Ratio |
|---|---|---|---|
| warp instructions | 10.14 M | 6.06 M | **1.67×** |
| IMMA tensor ops | 2.0–2.05 G (= 2×MAC) | 2.0–2.05 G (= 2×MAC) | **parity** |
| LDS bank conflicts (read+write) | 940.3 K + 499.0 K | 6.5 + 0 | ~100×+ (see below) |
| L2 read sectors | 22.0 MB | 13.7 MB | 1.6× |
| L2 write sectors | 66.1 MB | 1.45 MB | 45× (no source, later judged artifact) |
| duration / GMAC | 107.7 µs | 41.1 µs | 2.6× |

**The two patches' measurements** (interleaved A/B in the same session, baseline 1023–1043
tok/s):

| Patch | Content | Counter change | Result |
|---|---|---|---|
| FULL | merged A staging + register scale staging + qb8 272B pitch + float2 C stores | L1 requests **−55%**, L2 reads −14%, but inst 10.14 → 13.48 M/GMAC (**+33%**) | 865–871 tok/s, **−16%** |
| MINIMAL | only qb8 272B pitch + float2 C stores | the corresponding conflict/store metrics improved | 1041–1046 tok/s, **+0.3% (noise)** |

**Attribution conclusion**: L2 bytes, store sector efficiency, bank conflicts — the three
hypotheses disproven **simultaneously** (all three fixed at once, wall clock zero effect,
FULL even slower from instruction bloat). The gap's carrier is **the per-MAC warp
instruction stream × issue efficiency**: 1.67× instruction difference × issue 0.25 vs 0.41 ≈
the 2.6× time difference, closing against the linear law. r13 thereby pointed the campaign's
next lever at "cut shared-read instruction count in the compute loop without adding any
staging ALU or global traffic" — exactly the design input for r14 (B fragments via ldmatrix
+ widened scale reads, +18.5%/+23–30%).

**Veto mechanism (this step is MEAS-ONLY)**: the step's "product" is the attribution
conclusion, not code; the two patches were restored (cmp-verified) for missing the bar and
being mechanically disproven, with the command-level evidence chain recorded in `5ca037d`.
The future retry condition for FULL-class "byte theory" levers: when the kernel leaves the
stall-bound regime (r25's correction: the instruction stream is the first-order predictor,
but only under issue-stall constraints), or when L2 bandwidth itself becomes the bottleneck.

## 6. Lessons

1. **The per-MAC instruction stream is this kernel class's (GB10, stall-bound regime) first-
   order predictor** — bytes, sector efficiency, and bank conflicts can all be fixed
   simultaneously while the wall clock does not move.
2. **Attribution needs a counter-evidence patch**: a patch that "removes hypothesis X"
   measuring zero effect beats ten pages of reasoning; the preconditions are a parity-clean
   patch and one that genuinely hits the hypothesis's target (FULL's L1 requests −55% proved
   that).
3. **Establish which counters the device exposes before designing the forensics**: GB20B has
   no `dram__*`, and int8 mma sits in IMMA not hmma — get the checklist wrong and the
   session is wasted.
4. **Flag unattributed counter readings (the 66 MB/GMAC L2 write sectors) instead of
   concluding**, and only trust them after a "fixing the write shape should change it" patch
   verifies.

---

← 17 · [Index](./README.md) · 19 →
