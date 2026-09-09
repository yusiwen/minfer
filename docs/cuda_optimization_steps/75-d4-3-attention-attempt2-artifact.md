# 75 · D4-3 — attention structure rewrite attempt 2 NO-GO + D4-1's llama target was a llama-bench artifact (CLOSED)

> **Result**: the split-attention kernel `vec_attn` aligned to llama's fattn-vec geometry (five axes pb × threads × rows × minb × STG, a 690-row sweep, 0 correctness skips) best result **41.07 µs @14B/@3254 (bar ≤~32, 1.64× off the current dispatch kernel), 16.22 µs @7B/@1641 (bar ≤~10.35, 1.53×)** — folded into in-situ wall clock that is +1.6–1.8% < the +2% integration bar, **NO-GO; the attention line closes with a "measurement correction"**. The session's real output is the artifact identification: ncu proves llama-bench's own decode `fattn-vec` (grid (1,2,40)) loads a constant 5,427,200 B at any context ≈ one 128-row KV iteration per block = **of 3255 KV rows only 256 covered (7.9%)**, while llama-cli's decode (grid (1,7,40)) loads 53.2/142.7 MB scaling with context (corroborated by a mid-context recall A/B). Honest llama decode attention ≈ 2.0–2.1 TB/s ≈ 1.7 ms/step — minfer's 3.32 ms is a **~1.9× gap, not 4.9×**; 14B @3254's true gap shrinks from 13.4% to ~10% (attention ~3.5% of it).
> **Commit**: `ce01048` (docs-only, no repo code change). **Date**: 2026-09-09.

## 1. Background — where things stood

The brief handed over at D4-2's close said: in one 14B @3254 step the attention structure accounts for ~2.53 ms, the largest single piece of the remaining gap. That number's provenance is D4-1's trace: llama's decode attention at 14B/@3254 recorded **6.94 µs split + 7.20 µs combine = 14.02 µs/layer** per layer, 48 layers ≈ 0.67 ms/step; minfer's split+combine measured 3.32 ms/step. The reading at the time was "llama uses some structure we lack", and the byte ledger was spread out: 14B @3254's KV reads ≈ 48 layers × 66.8 MB per step (K+V f16, 3255 rows × 40 heads × 128 dims × 2 B × 2 tables), read inside 0.333 ms → **~9.6 TB/s effective L2 read rate** — an order of magnitude above our attention kernels' 1.0–1.4 TB/s. The prize was estimated at 2.3–2.5 ms/step.

That reading already had a contradiction buried in it: 9.6 TB/s exceeds GB10's L2 fabric capability. But D4-1's conclusion kept it when written into the brief, because "llama did it" was the trace's direct output. D4-3's task was therefore twofold: (1) build a llama-geometry kernel per the brief and probe it against a pre-registered GO bar; (2) if it cannot be built, explain where llama's 9.6 TB/s came from — "our kernel structure is inadequate" and "the target itself was wrong" are two completely different follow-up roads.

The session ran probe-first with a hard go/no-go: **write the probe first, register the bar first, run the sweep first; any integration only after a GO**. The GO bar was pre-registered as: a kernel with llama's structure (pb splits × windows × subgroup lanes, Q resident in registers, K/V read straight from global, probabilities broadcast with shfl — no smem touched in the hot loop) must simultaneously reach ≥2× the current dispatch kernel on **both 14B/@3254 (kernel-total ≤ ~32 µs) and 7B/@1641 (≤ ~10.35 µs)**; the integration bar was separately set at +2% wall clock (2% of a ~40 ms 14B step ≈ 0.8 ms). Both bars were written down before any numbers ran — that is why this doc can close cleanly on a "NO-GO".

## 2. Principle — the GPU mechanism

**The current dispatch kernel's shape (the probe's incumbent baseline).** minfer's decode attention is D3-4's hybrid rpw dual-kernel split: `gqa_attn_split_partial_hybrid` (4-warp form) or `gqa_attn_split_partial` (1-warp form) slices KV by `ATTN_SPLITS`, each block claims a segment and advances an online softmax row by row (each lane claims 4 consecutive dims, the full-row dot reduced with warp shfl), partials written to global; `gqa_attn_split_combine` merges. 14B/@3254 measures 63.8 + 3.6 = 67.4 µs (nsys per-kernel), 7B/@1641 is 20.7 + 4.1 = 24.8 µs. The probe copied this kernel family's device bodies **verbatim** into `/tmp/d4/probe_attn2.cu` as the baseline (`attn_split_1w_body` / `gqa_attn_split_partial_hybrid` / `gqa_attn_split_combine`), guaranteeing the sweep's control is the same math.

**llama fattn-vec's geometry vs ours.** llama's `flash_attn_ext_vec` (in ncu: `flash_attn_ext_vec<128,1,F16,F16,false>`) is organized as `pb` (parallel blocks) × KV windows: each block claims a window of KV (R rows), within the window K/V are read directly from global, Q is resident in registers, inter-row probabilities are broadcast with warp shfl, and window boundaries write partial results back for the combine to merge. Its block count = pb × n_head_kv (grid shaped (1, 7, 40): y dim 7 splits, z dim 40 KV heads), while we were fixed at 2 splits at the time. For a low-occupancy kernel at ≤2 blocks/SM, **performance is decided by the SASS's load batching**: in the natural load→dot→shfl interleaved loop, the next LDG waits for the current softmax dependency chain to finish before issuing; staging the whole window's K/V upfront (a run of back-to-back LDGs) lets the load latencies overlap each other — this difference measured 1.0 vs 1.7 TB/s in this probe (see §5).

**Why 9.6 TB/s is a reductio, not a target.** GB10's L2 bandwidth is on the order of ~6-7 TB/s and DRAM ~1.9-2.3 TB/s peak (both families' decode kernels sit in the measured 75–84% of DRAM peak = the 1.4–1.9 TB/s-class interval). If llama truly read 66.8 MB × 48 layers = 3.2 GB from L2 inside 0.333 ms, it would need 9.6 TB/s — beyond the fabric, physically impossible. Conversely, if it read only the bytes the trace's split kernel touched (5.43 MB × 48 = 260 MB ÷ 0.333 ms = 0.78 TB/s), it lands exactly in the latency-bound class all our decode kernels occupy. D4-3's core hypothesis test sits right here: **the premise of the 9.6 TB/s ledger (that the traced kernels cover all the KV) must be directly verified or refuted by ncu's byte counts**.

**The artifact's mechanism hypothesis (the part left undecided).** llama-bench's host side picks `flash_attn_ext_vec`'s pb as 2 (grid (1,2,40)), and its split/window loop runs only about one 128-row iteration at all three contexts KV≈1024/2474/3255 — i.e. the bench path's kernel actually covers only the first 256 KV rows. Why `ca3d5a3e1`'s occupancy loop picks pb=2 in that window while llama-cli's identical loop picks pb=7 — the record explicitly states this was **not pinned down** ("the observed facts above are unambiguous and reproducible; the binary is `b10665-ca3d5a3e1` per its own banner"). This doc only pins down the reproducible byte facts and leaves the host-side selection logic for upstream to check.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

**Probe structure: verbatim incumbent copies + a family of parameterized kernels.** `/tmp/d4/probe_attn2.cu` (nvcc -O3 -arch=sm_121a) holds three kinds of things: (1) the incumbent bodies copied verbatim from `src/cuda_kernels.cu` (the baseline's credibility comes from "zero rewriting"); (2) the llama-geometry template kernel `vec_attn<T,R,MINB,STG>` (T = pb, R = window rows, MINB = minimum blocks per window, STG = the load-scheduling axis); (3) a double-precision CPU reference. The five-axis sweep: pb ∈ {2,4,8,16,32}, threads ∈ {32,64,128}, R ∈ {32,64,128}, minb ∈ {1,4,8}, STG ∈ {0,1,2} (0 = chunked staging every 8 rows; 1 = whole-window K+V staging; 2 = STG1 + next-window K prefetch).

**The timing protocol aligned to decode's real shape**: batch-of-32, min-of-8, with two rotating L2-hot KV copies (simulating decode's KV-resident-in-L2 state). The correctness gate **0.05 abs + adversarial outliers** (q injection ±57 every 911 rows, v injection ±138 every 1543 rows) — the outlier injections specifically guard against the fake green of "softmax masking large errors on random data". All 690 sweep rows passed the gate, 0 correctness skips.

**Code survival note for the probe**: `/tmp/d4/probe_attn2.cu` and the sweep raw data `/tmp/d4/d43_sweep5.csv` are session artifacts, not in the repo; the probe's GO/NO-GO verdict and all key numbers were finalized in `ce01048`'s docs commit (the same treatment as the r10/r11 precedent: a measurement session's first disk write is docs). The 3.2 excerpts below fall into two classes by content — the incumbent excerpts come from the current tree (the probe is verbatim identical to it), and the `vec_attn` form is reconstructed from the record.

### 3.2 Key code

**Excerpt A · the incumbent's hot loop (`src/cuda_kernels.cu` 2803–2872 excerpt, the probe baseline = verbatim same lineage)** — note the D2 comment: staging the 4-row window's K+V before computing was a load-scheduling correction bought at −11%/−42% by the D1/D2 probes; `vec_attn`'s STG axis pushes exactly this idea to whole-window granularity:

```cuda
// src/cuda_kernels.cu (current tree, attn_split_1w_body hot loop excerpt)
for (int base = lo; base < hi; base += 4) {
    int nr = min(4, hi - base); // warp-uniform
    // D2: stage BOTH K and V for the whole 4-row window before the first
    // softmax step. All 8 row loads then issue back-to-back and their
    // latency overlaps the serial chain; the old form relied on the
    // compiler hoisting the inline V loads, which it does not do across
    // the shfl/softmax dependency chain (D1 probe: −11% hot-L2, D2 probe:
    // −42% cold-DRAM vs inline V; bitwise-identical — same rows, same
    // order, same per-row ops, only the load scheduling changes).
    float4 k4[4], v4[4];
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        k4[j] = (live && j < nr)
            ? kv_ld4<KV>(k + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        v4[j] = (live && j < nr)
            ? kv_ld4<KV>(v + (size_t)(base + j) * stride_kv + hk * hd + d0)
            : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < 4; j++) {
        if (j >= nr) break; // warp-uniform: all lanes exit together
        float d = q4.x * k4[j].x + q4.y * k4[j].y + q4.z * k4[j].z + q4.w * k4[j].w;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            d += __shfl_xor_sync(0xFFFFFFFF, d, off);   // full-row dot reduction
        float s = d * scale;
        float nmx = fmaxf(mx, s);                        // online softmax step
        float corr = expf(mx - nmx);
        float e = expf(s - nmx);
        S = S * corr + e;
        mx = nmx;
        if (live) {
            float4 vv = v4[j];
            oc.x = oc.x * corr + e * vv.x;  /* ... oc.y/z/w same form ... */
        }
    }
}
```

**Excerpt B · the hybrid dual-kernel's self-gating dispatch (`src/cuda_kernels.cu` 2894–2908)** — the sweep's third baseline (h4w) and the "current dispatch kernel" 67.4/24.8 µs both measure through this path; the gate's warp-uniform condition is the key to replay safety:

```cuda
// src/cuda_kernels.cu (current tree, gqa_attn_split_partial dispatch arm excerpt)
// D3-4 L1 dual-kernel dispatch: when rpw_gate > 0 and the 4-warp kernel
// owns this nkv (rpw >= rpw_gate), exit before touching anything — the
// hybrid kernel writes the partial. The branch is nkv-uniform across the
// whole grid (positions[0] is launch-wide), so this stays replay-safe,
// and for every nkv the incumbent path takes, the arithmetic in the body
// is unchanged (bitwise; dump-memcmp gated).
if (rpw_gate > 0) {
    const int nkv0 = positions[0] + 1;
    const int chunk0 = (nkv0 + ATTN_SPLITS - 1) / ATTN_SPLITS;
    if (((chunk0 + 3) >> 2) >= rpw_gate) return;
}
attn_split_1w_body<KV>(q, k, v, partial, positions[0] + 1,
                       blockIdx.x, blockIdx.y, nh, nk, hd, scale, pstr,
                       threadIdx.x);
```

**Excerpt C · `vec_attn`'s reconstructed form (reconstructed from the probe record, not surviving code)** — the skeleton of the llama geometry: a block claims (split, window), Q resident in registers, online softmax row by row after STG=1's whole-window staging; the difference from excerpt A is only the row-grouping granularity and the staging depth, the math being the same online softmax:

```cuda
// /tmp/d4/probe_attn2.cu (session artifact; form reconstructed from the record)
template <int T_PB, int R_WIN, int MINB, int STG>
__global__ void vec_attn(const float* q, const f16* k, const f16* v,
                         float* partial, int nkv, /* … */) {
    // per block: 1 KV head × 1 window (R rows); Q lives in registers, K/V
    // read straight from global.
    // STG=0: chunked staging every CS=8 rows; STG=1: the whole window's
    // K+V staged in one go; STG=2: STG1 + next-window K prefetch.
    float4 kreg[R_WIN], vreg[R_WIN];
    if (STG >= 1) {
        #pragma unroll
        for (int r = 0; r < R_WIN; r++) {          // whole-window batched LDG —
            kreg[r] = ld_kv(k, base + r);           //  issued back-to-back, latencies overlap
            vreg[r] = ld_kv(v, base + r);
        }
    }
    for (int r = 0; r < R_WIN; r++) {
        float d = warp_dot(q, STG >= 1 ? kreg[r] : ld_kv(k, base + r));
        // shfl reduction → online softmax → P accumulates into oc (same form as excerpt A)
    }
    // write the partial at the window tail; the combine merges (same combine as the incumbent)
}
```

### 3.3 Pitfalls

- **Both probe bugs were caught by the correctness gate, not "explained away" by the numbers**: (1) the subgroup row-map missed the warp stripe offset — every 32nd row's dot was misaligned; (2) one refactor dropped the chunked V-pass's initial staging — the STG=0 variant was slow and wrong overall. Both manifested as correctness FAILs (the outlier-injection design happened to amplify them), and the swEEP produced no "sick numbers". That is the value of the probe-first process: the correctness gate hangs inside the probe, so a bad kernel never survives to reach the main tree.
- **ncu byte counts must be read via the LDG instruction composition**, not the total alone: the bench-path kernel issues 44 LDGs per warp ≈ K 16 + V 16 + Q 4 + mask 8 — exactly one 128-row KV iteration's worth. The reason "byte-identical at KV 1024/2474/3255" holds is that the three captures' LDG counts and address patterns are completely identical; if the total byte count does not change while the context changes 3×, that is far stronger artifact evidence than "slow".
- **Artifact identification takes three-way agreement** to be final: ncu byte counts (5.43 MB constant vs 53.2/142.7 MB scaling with ctx) + behavioral A/B (mid-context passphrase recall) + reductio (9.6 TB/s beyond the fabric). Any single piece still leaves room for reinterpretation.

## 4. Verification

- **Probe correctness gate (0.05 abs + adversarial outliers + double CPU ref)**: defends against the sweep numbers coming from a mis-computing kernel; 690 rows, 0 skips.
- **Baseline same-lineage check**: the incumbent bodies were copied verbatim from the current tree — defends against a fake GO/NO-GO caused by "the probe baseline is faster/slower than the real kernel".
- **The artifact identification's ncu evidence**: the bench decode `flash_attn_ext_vec<128,1,F16,F16,false>` shows SASS global-load bytes **5,427,200 B bit-identical** across the KV≈1024/2474/3255 captures, LDG/warp = 44; llama-cli decode (grid (1,7,40)) loads 53.2 MB at KV≈2477 (`-c 2610`) (≈ the 50.7 MB full-coverage arithmetic) and 142.7 MB at KV≈4869 (`-c 0`) — scaling with context.
- **Behavioral A/B (recall gate)**: the mid-context passphrase (KESTREL-5150, ~row 1200/2470) is answered correctly by llama-cli under both `-c 0` and the bench-like `-c 2610` — proving the CLI path's attention really reads the full context; llama-bench's tg output never performs any recall check, so it cannot notice on its own that the bench path drops 92% of the KV.
- **Reductio cross-check**: taking the trace reading as full coverage, 48 layers × 66.8 MB ÷ 0.333 ms = 9.6 TB/s L2 — beyond the fabric, impossible; taking ncu's measured bytes, 5.43 MB × 48 ÷ 0.333 ms = 0.78 TB/s — the same class as the other latency-bound kernels. Both directions rule out "9.6 TB/s is real".

## 5. Results

**Sweep results (all passing the correctness gate; split+combine totals, µs)**:

| shape | best vec geometry | split | combine | total | probe baseline (1w/h4w) | in-situ dispatch today | GO bar |
|---|---|---:|---:|---:|---|---:|---|
| 14B @3254 | pb=4 T=128 R=64 STG=1 | 38.89 | 2.17 | **41.07** | 49.74 / 55.26 | 67.4 (63.8+3.6) | ≤~32 → **FAIL (1.64×)** |
| 7B @1641 | pb=16 T=32 R=32 STG=1 | 12.17 | 4.04 | **16.22** | 20.45 / 32.78 | 24.8 (20.7+4.1) | ≤~10.35 → **FAIL (1.53×)** |
| 7B @512 | pb=4 T=128 R=32 STG=1 | 8.04 | 2.12 | **10.16** | 11.61 / 24.59 | ~11.8 | — |

The geometry plateau is flat (pb 2–8 within 1.5 µs on 14B), and the STG axis enters noise on the plateau — **once loading is batched per window, more scheduling depth buys nothing at this block-count level**. vec_attn is indeed ~17–21% faster than the incumbent 1w baseline, but still 1.5–1.6× short of the bar.

**The GO/NO-GO arithmetic (veto mechanism)**: the probe→in-situ conversion factor was calibrated on the h4w baseline (probe 63.8 / in-situ 51.2 = 1.25): 14B 41.07 ÷ 1.25 ≈ 48.5 + 3.6 combine ≈ 52 µs → 0.72–0.82 ms saved per step ≈ **+1.6–1.8% wall clock < the +2% integration bar**; 7B ≈ +0.7%. Both bars fail → **no integration, the line closes**. The honest remaining headroom (14B ~1.6 ms/step) requires ~2.1 TB/s effective rate — llama's own measured rate — corresponding to pb=7 (280-block)-class geometry plus a staged-load recipe beyond the sweep plateau; that is a new session with a new bar (≤25–32 µs @14B), not a continuation of D4-3.

**THE ARTIFACT — the correction of D4-1's target (this doc's archival centerpiece)**:

| capture | grid | SASS global-load bytes | LDG/warp | KV rows covered |
|---|---|---|---|---|
| **llama-bench** decode, KV≈1024 / 2474 / 3255 | (1,**2**,40) | **5,427,200 — bit-identical across all three** | 44 ≈ one 128-row iteration (K 16 + V 16 + Q 4 + mask 8) | 2 splits × 128 = **256/3255 (7.9%)** |
| **llama-cli** decode, KV≈2477 (`-c 2610`) | (1,**7**,40) | **53.2 MB** (≈ the 50.7 MB full-coverage arithmetic) | — | full |
| **llama-cli** decode, KV≈4869 (`-c 0`) | (1,**7**,40) | **142.7 MB**, scaling with ctx | — | full |

The bench path kernel's load traffic is **context-independent** (KV 1k or 3.3k are both 68 KB per block), i.e. its 2-split grid runs only ~one 128-row iteration per block and covers only the first 256 rows; the CLI path picks 7 splits and full coverage. Recalibrated: llama's honest full-context decode fattn-vec @14B runs at **≈2.0–2.1 TB/s effective** (53.2 MB/60 µs-ncu ≈ 25 µs live @2477; 142.7/171.5 ≈ 70 µs live @4869), i.e. ~33–36 µs per layer including combine at @3254 ≈ **1.7 ms/step, not 0.69**. minfer's 3.32 ms/step is a **~1.9× gap, not 4.9×**; 14B @3254's honest wall-clock gap to llama shrinks from 13.4% to ~10%, of which attention is ~3.5%.

**Verdict: the D-series attention line closes with a "measurement correction".** D4-1's census itself was not wrong (those two kernels really did run 48×6.94 + 48×7.20 µs — the ledger balances); the error was reading those two kernels as "full-context attention". The follow-up road is rewritten accordingly: (1) 14B decode's truly recoverable space against llama is ~10% of wall clock, of which attention is only ~3.5% and needs ~2.1 TB/s to reach llama's level; (2) future attention sessions start from (1,7,40)-class geometry + explicit load staging, with the bar set at ≤25–32 µs @14B; (3) no minfer-side code was changed by this session — `ce01048` is docs-only, and the probe numbers and the artifact verdict are the entire output.

## 6. Lessons

1. **Never quote llama-bench's long-context tg rates as attention targets.** The bench path's attention kernel can skip most of the KV and still produce a tg number (bench never validates recall); to check an attention target, use a llama-cli recall A/B or ncu byte counts.
2. **The reductio of "the target is impossible" should be done as early as possible.** 9.6 TB/s beyond the fabric was a contradiction computable at the brief stage; doing the dimensional check before building the kernel would have saved half of this session's kernel sweep.
3. **A pre-registered bar is the NO-GO's talisman.** 41.07 µs was the sweep's best number; without a bar, "39% faster" would have tempted an integration doomed to fall below the integration line. Bar written first, numbers run after — only then can a NO-GO close cleanly.
4. **Artifact identification needs a three-way evidence chain**: byte counts (ncu) + behavioral validation (recall A/B) + a physical ceiling (reductio) — any single piece of evidence can be "reinterpreted"; only three-way agreement is worth rewriting a conclusion over.

← 74 · [Index](./README.md) · 76 →
