# 58 · r55 — swiglu roofline audit + one-shot prefill CUDA-Graph: both closed on the record (CLOSED)

> **Result**: both low-risk levers are closed **before any code was written** by measurement — fused-swiglu already runs at 242 GB/s ≈ 89% of the 273 GB/s spec roofline, so even a perfect kernel's ceiling gain is only +0.74% (< the +1.5% bar); the reclaimable part of one-shot prefill capture (the recurring gaps) is ≤ 0.1% (the rest is one-time host stalls and capture-ILLEGAL mid-window `cudaMalloc`s). The campaign verdict: **CONVERGED** (3181 vs 3324.4 = 1.05×).
> **Commit**: `83c3c67` (record-only commit — zero tree changes, baseline binary cmp-verified). **Date**: 2026-09-06.

## 1. Background — where things stood

r53 landed the q6_K B-side bundle (+5.03%), r54 fitted the exit valve onto the 1.52 GB plane, and whole prefill stood at 3181 tok/s, vs-llama 1.05×. With the q6_K line, the FA line, and the A-quantize line all closed out, Session D — as the "basket final round" — took inventory of the remaining low-risk levers and picked two targets that looked most within reach:

1. **Speeding up fused-swiglu again**. r51/r52 had already fused silu+mul into the producer (mode 1 writes the f32 + q8 plane; mode 2 skips even the f32, producing q8 in registers). This kernel is an already-optimized object, not a beginner's job: prepass launches went 193 → 110 (r49's shared-A dedup) → 28 (r51's fusion), prepass time 118.4 → 83.9 → 10.1 ms; mode-2 then took fused swiglu from 110.5 → 64.1 ms (−42%). The proposal argued there was still something to mine — after all, what remains is "pure bandwidth" work, and everyone wants to squeeze it once more.
2. **CUDA-Graph capture for the one-shot prefill**. R3-B's 3-run protocol auto-captures repeated same-nt prefills (the server/multi-turn paths benefit); but a CLI one-shot prefill never reaches a 3rd run and has always run bare. nsys shows ~7.84 ms of idle inside the window — "capturing those gaps away" sounds like free money.

The campaign had an iron rule in force at this point: **the +1.5% bar**. It is not arbitrary: the A/B interleaved measurement window noise is ±2% (co-tenant drift, machine state), and anything below it can neither be proven nor reproduced — every lever landed since r37 sits above it, and every REVERTED "right direction but too small" lever (r44/r45 alone at −0.34%/−0.42%) sits below it. r55 changed the working method: **derive the ceiling first, then decide whether to write code**. This round's output is not code but two veto records with numbers in them — "documented skip" is this campaign's formal disposition category: the numbers, the veto mechanism, and the retry conditions go into the record, the lever is terminated, and it will not come back next session as an "obviously doable" low-hanging fruit.

Baseline sanity first: in a window with a co-tenant running, the measurement read 3144.4–3151.4 (against 3181 on a quiet machine, recorded in footnote 2), and the binary cmp-matched HEAD — confirming the measurement object had not drifted, and all readings were taken on the same object.

## 2. Principle — the GPU mechanism

### 2.1 The roofline-bound-before-coding method

A bandwidth-bound kernel's time lower bound is:

```
t_min = bytes_min / BW_ceiling
```

GB10's trap is that **ncu has no `dram__*` counters** — the "measure the DRAM bytes" road does not exist, so `bytes_min` must be **analytically derived** from the access pattern, with ncu's sector counts used only to prove the access pattern (width, coalescing), never to count bytes. `BW_ceiling` takes the spec value of 273 GB/s. That choice biases the bound in the right, conservative direction: real achievable bandwidth ≤ spec, so the true headroom can only be smaller than computed. The inference chain then closes: **if even a spec-perfect kernel cannot clear the bar, no implementation can** — that is the "bound kills the lever" logic, and it is far cheaper than "implement, then measure": the whole chain is one ncu run plus some arithmetic.

### 2.2 The swiglu traffic audit: why the proposal's estimate was 2× low

`swiglu_quant_nw_f32_t` (r52's mode-2 kernel) does, per element: read gate's f32, read up's f32, compute silu·mul, quantize into the q8 plane. Its access shape is hard-coded in the kernel — one token row per block, one `float4` per lane:

```cuda
const float4* g4 = reinterpret_cast<const float4*>(gate + (size_t)t * dim);
const float4* u4 = reinterpret_cast<const float4*>(up + (size_t)t * dim);
float4 gv = g4[f];
float4 uv = u4[f];
v0 = (gv.x / (1.0f + expf(-gv.x))) * uv.x;   // silu(g)*u, per float4
```

ncu's verdict has two parts:

- **The read side is f32, full stop**. Measured read traffic is 502.2 MB = exactly `2 × nt × dim × 4 B` — that equation is itself the proof: if the reads were f16 or q8, the bytes would be half or a quarter. The proposal's traffic estimate counted a narrower width and came in **a full 2× low**; implement with that estimate and you only discover the bar was never reachable after finishing.
- **Vectorization is already maxed**. `sectors/request = 16`, and arithmetic fits it exactly: one warp request = 32 lanes × 16 B (`float4`) = 512 B = 16 × 32 B sectors. Every byte is already inside a used transfer; there is no "switch to vectorized reads" headroom to mine.

The write side's minimum bytes derive too: mode 2 writes no f32 output (r52's core), only the q8 plane and the sda plane — `36 B` per 32-element block (32 int8s + the f16 dsc) plus each block's sda share, totaling ≈ 70.9 MB. So the 573.1 MB composition is checkable: **read 502.2 (the proposal estimated half of that) + write ≈ 70.9**.

Total minimum DRAM traffic 573.1 MB, measured duration 2.367 ms → **242 GB/s = 89% of roofline**. 94.6% occupancy, with a stall profile of a latency-bound pure stream — this is what "an already-maximized stream" looks like: no coalescible accesses, no raisable occupancy; the remaining 11% is the latency gap inherent near DRAM, not something the kernel's shape can claw back. So the ceiling gain:

```
573.1 MB / 273 GB/s = 2.099 ms (ideal)
2.367 − 2.099 = 0.268 ms per launch
× all swiglu launches ≈ 7.2 ms ≈ +0.74% whole-prefill  <  +1.5% bar
```

**Even a perfect kernel falls 0.76 percentage points short** — the skip is computed, not felt.

The method's applicability boundary is also written down: this bound is only tight for **bandwidth-bound** kernels. If ncu shows low occupancy, or stalls stuck on ALU or synchronization (rather than long_scoreboard-class memory stalls), the kernel is latency/compute-bound, the byte lower bound no longer represents achievable time, and you must switch to occupancy/dependency-chain analysis. swiglu's 94.6% occupancy + pure-stream stall profile falls exactly inside the bound's domain — part of "read the profile first, then pick the analysis tool."

### 2.3 The one-shot capture account: three ingredients inside the 7.84 ms idle

First the capture mechanism: stream capture records the kernels/copies issued in sequence on a stream inside the window as a graph, and one `cudaGraphLaunch` replays all of it — at replay the host no longer walks the driver per launch. It removes the **per-launch host overhead and gaps**, and pays off more the more times the same graph is replayed. Decode lives on it precisely because the same decode graph replays hundreds of times; R3-B's 3-run protocol was likewise designed for "the same nt prefill appearing repeatedly" (capture on the 3rd occurrence, cost amortized by subsequent replays). A one-shot prefill occurs exactly once, so nothing gets amortized.

What it cannot remove, and cannot digest, nsys splits the 7.84 ms idle into three classes:

| Ingredient | magnitude | can capture reclaim it? |
|---|---|---|
| two one-time ~3 ms host stalls (minfer's own host code, not inside CUDA APIs) | ~6 ms | no — they live in host logic outside the capture window; replay is irrelevant to them, and they happen only once |
| mid-window `cudaMalloc` | 0.78 ms | no — and it is **capture-ILLEGAL**: synchronous allocation and other potentially-implicitly-synchronizing APIs are forbidden inside a stream-capture window, so the graph cannot even be built; legalizing it means changing the allocator (the pre-grow/pre-warm route) |
| recurring inter-launch gaps | ~0.1% | **yes** — the only thing replay can remove |

The reclaimable part is ≤ 0.1%, an order of magnitude below the bar. The capture mechanism itself is not wrong — what's wrong is aiming it at a scenario whose replay count is 1.

The comparison can be made concrete: the decode campaign later quantified the pool replay can save — the graph-gap pool is on the order of **2 µs/launch** (D4-4's PDL probe went after exactly it, ultimately abandoned over the co-residency tax). Reasoning backwards from that number to the one-shot prefill: even if all recurring gaps were reclaimable, their absolute size lives in that 2 µs/launch pool — consistent with nsys's ~0.1%.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **The bar registers first; the numbers speak after**: the +1.5% is fixed in writing before measurement; once the two levers' bounds come out, the conclusions follow automatically, with no sunk cost of "build it and see if it's good." This is isomorphic to another standing campaign discipline: the "pre-registered bars" of many later rounds (e.g. D4-3's ≤~32 µs bar) are the same practice.
- **Attribution freshness first**: r47 had just overturned r37's attribution table (q6_K fell from 51.2% to 15.4% — decompositions go stale as levers land). r55's audit ran on the converged shape (the tree after r53/r54), so neither the object nor the numbers were stale; this is itself a transferable practice — any roofline/attribution audit should first ask "whose engine version is this decomposition of."
- **Skips get recorded too**: the measured numbers, the veto mechanism, and the conditions worth a retry all go into the record. A documented skip's value is terminating a lever — this doc's §5 retry conditions are written for exactly that.
- **Zero tree changes**: this round touches no source file, and the baseline binary is cmp-verified against HEAD — every measurement is taken on a clean object; `83c3c67` contains only the record.

### 3.2 Key code

The audited mode-2 swiglu kernel (`src/cuda_kernels.cu`) — note it is already "fused all the way": reads `float4`, writes packed q8, no f32 round trip (the f32 write is r51's mode 1; r52's mode 2 saved that too):

```cuda
__global__ void swiglu_quant_nw_f32_t(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    uint8_t* __restrict__ yqs,   // [ntb][nchunk][2048]
    uint8_t* __restrict__ ysda,  // [ntb][nchunk][256]
    int dim, int nt, int nchunk, int ntb
) {
    const int t = blockIdx.x;  // one token row per block (grid = ntb*64)
    ...
        if (active) {
            const float4* g4 = reinterpret_cast<const float4*>(gate + (size_t)t * dim);
            const float4* u4 = reinterpret_cast<const float4*>(up + (size_t)t * dim);
            float4 gv = g4[f];
            float4 uv = u4[f];
            v0 = (gv.x / (1.0f + expf(-gv.x))) * uv.x;   // silu(g)*u, per float4
            ...
        }
        float am = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
        // 8-lane group reduce ... dsc = am/127 ... rintf quantization into the packed word
```

The question the roofline audit asks: how else could a "perfect kernel" change it? Item by item: the read bytes don't change (the f32 inputs are what they are; the width is not this kernel's choice); the write bytes don't change (the q8 plane is the downstream MMQ's fixed input format); the `float4` accesses already push sectors/request to 16; occupancy is already 94.6%. **No bytes left to change means no time left to mine.**

The capture-side window primitive (`src/cuda.rs`) — the audit confirms it is per-stream, and any host-side sync/allocation inside the window is illegal:

```rust
pub fn graph_begin_capture(&self) -> bool {
    let stream = self.stream();
    let err = unsafe { cudaStreamBeginCapture(stream, 1) };
    ...
}
```

That 0.78 ms `cudaMalloc` in the nsys trace lands while this window is open — the problem is not a slow kernel, it is that the window **cannot legally open**. To eat that bite, the allocation must first become pre-grow (r59's rider later took that road), not by forcing capture on top.

"Prefill never enters a capture window" is an explicit invariant in the current tree (`src/cuda.rs`, at the f16 GEMM dispatch):

```rust
// Prefill never enters a CUDA Graph capture window (8g①
// decode-only gate), so the on-demand scratch grow is safe.
if nt > 1 && id <= 8192 {
    let q8 = Self::get_or_grow(
        &self.buf_q8_prefill,
        nt * (id / 32) * Q8B,
    );
    ...
```

The 3-run protocol and its supporting conventions also live in `prefill_mmq`'s doc comment (same file, current tree):

```rust
/// R1: int8 MMQ prefill GEMM — quantize activations to q8_0 (pad40
/// blocks; ...) ... The q8 scratch follows the same
/// grow-on-demand lifecycle as the f16 path's buf_f16_x: the 3-run
/// capture protocol sizes it before the capture window opens.
pub fn prefill_mmq(
```

That comment also explains why an on-demand scratch grow is allowed on the prefill path — capture-illegal allocations only matter inside a capture window, and prefill never enters one; the 3-run protocol sizes the buffers **before** the window opens. What r55's "one-shot prefill capture" proposal was really asking: should this 8g① invariant be **overturned** for the CLI one-shot scenario? With the reclaimable part ≤ 0.1%, the answer is no — and overturning it would first require fixing every grow point inside the window (each one a potential capture-illegal allocation).

### 3.3 Pitfalls

- **The proposal's traffic estimate was 2× low**: it estimated the read side at a narrow width. One ncu run, and the identity `502.2 = 2 × nt × dim × 4` sentenced it outright. Had it been implemented first and measured later, a full round of implementation + debugging would have been wasted on an unwinnable position — and implementers tend to find reasons to keep investing after a "so close to the bar" result.
- **GB10 has no `dram__*` counters**: "measured bytes" does not exist; only analytic derivation + ncu sector counts as corroboration. This is a platform limitation — a bound must state its derivation chain explicitly, otherwise the bound itself cannot be re-checked.
- **Capture legality only becomes visible by walking nsys ms by ms**: the mid-window `cudaMalloc` is perfectly legal in code review (the allocator belongs there) and illegal only in the capture context; "capture-illegal" is a runtime fact no type system will catch.
- **Baseline sanity in a co-tenant window**: 3144.4–3151.4 vs 3181 on a quiet machine — confirm the object hasn't drifted before the numbers mean anything (footnote 2's accounting).

## 4. Verification

- **ncu sectors/request = 16**: proves the accesses are maximally vectorized — guards against the illusion that "vectorization headroom remains."
- **byte-identity cross-check**: `502.2 MB ≡ 2 × nt × dim × 4 B` — guards against misjudging the read width (exactly where the proposal's estimate went wrong). Same on the write side: 573.1 − 502.2 = 70.9 MB matches the q8/sda plane arithmetic — guards against a missing term in the total traffic.
- **baseline cmp check**: the binary matches HEAD — guards against measuring the wrong object.
- **nsys ingredient classification**: every idle ms tagged one-time / recurring / capture-illegal — guards against the linear extrapolation that "all 7.84 ms could be captured away."
- **pre-registered bar**: the +1.5% fixed before measurement — guards against moving the goalposts after the fact.

## 5. Results

Both levers are closed; the numbers:

| Lever | measured | ceiling gain | verdict |
|---|---|---|---|
| fused-swiglu rewrite | 573.1 MB / 2.367 ms = 242 GB/s = **89% of roofline** (94.6% occ, a latency-limited pure stream) | ≤ **+0.74%** (0.268 ms × all launches) | skip: even a perfect kernel is < the +1.5% bar |
| one-shot prefill capture | idle 7.84 ms = one-time host stalls ~6 ms + `cudaMalloc` 0.78 ms (capture-illegal) + recurring gaps ~0.1% | ≤ **+0.1%** | skip: the reclaimable part is an order of magnitude below the bar |

**Residual leads on the record** (left for later sessions): rms_nw at 153 GB/s = 56% of roofline (ideal +0.98%); the host stalls deserve a root-cause pass; tail pre-grow +0.1%. Note that rms_nw, this "last lead," has an ideal gain of +0.98% that is itself below the +1.5% bar — done alone it also fails the gate, and it is only worth being a member of some future bundle. That is the full meaning of the convergence verdict: not just "no lever ≥ bar," but "even the remaining leads' ceilings summed cannot constitute an independent next step."

**The campaign convergence statement** (this round's core output): whole-prefill tightened from r37's 2.15× to **3181 vs 3324.4 = 1.05×** (r53/r54). The wall decomposition and each component's closing state:

| Wall component | share | state |
|---|---|---|
| q4_K GEMM | 63.2% | closed — unless the q8_1 GEMM-prologue fusion, a step-function, is attempted |
| q6_K GEMM | 15.4% | closed |
| fused producers | 8.8% | swiglu closed this round; rms is the last lead (56% of roofline) |
| FA | 5.3% | the 2.43× is taken |
| host | ~1% | stall root-cause is a recorded pass |

**No identified lever is ≥ +1.5%; the next tier is the step-function q8_1 prologue fusion (llama.cpp's route of folding activation quantization into the GEMM main loop). Campaign verdict: CONVERGED** (under the current gate set — r56/r59 later reopened it with the bundling mechanism, see docs 59 and 62).

Retry conditions (otherwise these two skips would be re-proposed forever): swiglu — only if the q8_1 prologue fusion changes its traffic pattern (the read side stops being independent f32 streams) or the bar is lowered below +0.74%; capture — only if the allocator is capture-legalized (pre-grow) AND the host stall is root-caused into the launch path, and even then the ceiling remains the recurring gaps ≈ 0.1%.

## 6. Lessons

1. **Roofline-bound-before-coding**: derive the byte lower bound first, then decide whether to write code — if a perfect kernel cannot clear the bar, no implementation can. Both levers cost one ncu run, not one implementation cycle.
2. **A proposal's traffic estimate must be audited by ncu first**: the swiglu proposal underestimated 2× (narrow-width read side); the cost of an estimate error surviving an entire implementation cycle far exceeds one audit.
3. **Capture's gain accounting is the recurring part**: one-time host stalls and capture-illegal operations are not in the reclaimable set; for a replay-count-of-1 scenario, capture's ceiling is the inter-launch gaps.
4. **A documented skip is a deliverable**: numbers + veto mechanism + retry conditions in the record — only then is a lever truly closed.

---
← 57 · [Index](./README.md) · 59 →
