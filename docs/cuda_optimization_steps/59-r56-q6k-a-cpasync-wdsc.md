# 59 · r56 — q6_K A-side bundle: A cp.async + W_dsc f32 plane (LANDED)

> **Result**: whole prefill 3138.6 → 3212.5 tok/s (**+2.35%**, vs-llama 1.035×); ffn_down kernel 12.76 → 12.01 ms (−5.9%), attn_v −4.2%; both the `<2,true>` and `<2,false>` instantiations reach 80 regs / 0 stack with LDGSTS in the SASS; the cost is the +363.2 MB W_dsc plane. parity ×3, the greedy 453-character stream byte-identical, liveness 27×/0.
> **Commit**: `4cf7c74` (code) + `29084de` (record). **Date**: 2026-09-06.

## 1. Background — where things stood

r53 closed out the q6_K B side as a bundle (W_exp plane + cp.async, +5.03%), and r54 fitted the exit valve. Session E's basket list had two items left: item 1 was FA KV staging double buffering (next doc, REVERTED), and item 2 is this doc — q6_K's two remaining residuals.

Both residuals had been named in r43's PC-sampling attribution. At that time the q6_K kernel's stall profile was three pieces: **B-expand recombination 45% + A-side staging STS 28% + dsc I2F consumption 26%**. r44/r45 killed the 45% piece (each wall-neutral alone), and r53 bundled them into the wall. So after r53, the record carried §11.32's verdict verbatim: "A-side staging STS (~28%) + the dsc I2F consumer (~26%) — a W_dsc f32 plane would be the symmetric next bundle member, and an A-side cp.async redo could now compose".

Two terms need expanding:

- **The A side**: each kt, the NB-BT kernel must move the activation side's qa8 (the pad40 transposed q8 plane) and sda (the d/ssum scale plane) into shared memory. In the r53 shape, B was already cp.async while A was still synchronous LDG→STS — the global-read latency at the top of every staging phase stood exposed (r45 quantified this mechanism's kernel-level payoff back then: kernel −10.2%, long_scoreboard stall −18%, but the wall only −0.34% — the q6_K GEMM was no longer the wall at the time).
- **The dsc consumer**: every (chunk,row) reads the int8 scale at `blk[192+s0]` and the f16 d at `blk[208]` from the raw weight block, converts via I2F, then multiplies — 26% of r43's stall mass sat on the I2F. r42 tried a stage-level scale read and got NEUTRAL: cutting dsc bytes cannot cut dsc latency; the stall is on the consumer side.

r45's history is this doc's foreshadowing: A-side cp.async was tried once alone, was REVERTED, on the grounds that "the q6_K GEMM is no longer the bottleneck — a faster kernel not on the wall can never reach the wall." The bundle thesis predicted: once B becomes a pure copy and the wall moves onto the A-side wait and dsc consumption, **the same mechanism turns positive again**. r56 is that prediction's verification. The three mechanisms' "provenance" and "what they remove" line up into a table:

| Mechanism | what it removes | quantitative evidence |
|---|---|---|
| r44 W_exp dense plane | the B recombination's WORK | kernel −10.9% (wall −0.42% at the time) |
| r45 A-side cp.async | the staging's WAIT | kernel −10.2%, longsb −18% (wall −0.34% at the time) |
| r53 B bundle | both of the above together | whole prefill +5.03% |
| r56 A cp.async + W_dsc | the remaining A WAIT + dsc I2F WORK | whole prefill +2.35% (this doc) |

## 2. Principle — the GPU mechanism

### 2.1 The pipeline structure after the bundle

After r56, each kt's staging becomes **one cp.async stream**: all four planes — A's qa8, A's sda, B's W_exp, and dsc — are issued with explicit-PTX `gemm_cp16`, and `gemm_cp_commit()` moves to the **end** of the RAW_STAGE macro — exactly one commit group per kt binds the four planes together. The waits in the main loop are unconditional:

- middle tiles: `gemm_cp_wait1()` — two groups in flight (kt's and kt+1's); wait until only kt+1's remains, kt's copies (in-order group completion) have landed, and kt+1's copies keep flying under kt's compute;
- the last tile: `gemm_cp_wait0()` — drain every group.

The unconditional wait is a correctness requirement: group completion is in issue order, and any data-conditional skipping of the wait would fork the group ordering between the two EXP states.

### 2.2 The W_dsc plane: turning the 26% I2F stall into one 16 B copy

At registration, precompute each (chunk c, row j) dsc pair: `out[(c*od + j)*8 .. +8] = float2(d·sc[2(c&7)], d·sc[2(c&7)+1])`. The **chunk-major** layout is the key — the kernel's per-kt staging window is exactly "the contiguous MMQ_NBJ rows of chunk c0+kd," and under chunk-major those bytes are contiguous, so one `gemm_cp16` per pair (2 float2 = 16 B) moves them. The raw path's bill: per (row, chunk), 2 non-adjacent byte reads (int8 scale) + 1 f16 read + 3 I2F + 2 f32 multiplies; the plane path: 1 vectorized 16 B copy, zero ALU.

Plane size: `nchunk × od × 8 B = (id/32) · od · 8 = od · id / 4`. The q6_K tensors of 7B q4_k_m total **363.2 MB** — a quarter of r53's 1.52 GB, consistent with the arithmetic "one f32 pair per 4 B bytes."

Why chunk-major is the layout's right answer: the kernel's per-kt staging window is "rows `j0 .. j0+MMQ_NBJ` of chunk `c0+kd`." If the plane were row-major (`plane[j*nchunk + c]`), adjacent rows of the same chunk would sit `nchunk × 8` bytes apart in memory — one copy per row, impossible to coalesce into 16 B. Chunk-major (`plane[(c*od + j)*8]`) puts a chunk's rows together: row j's 8 bytes are immediately followed by row j+1's 8 bytes — **one pair per row, two pairs per chunk** — exactly composing the 16 B cp.async transfer unit. The layout follows the consumption window; this is the same design law repeatedly verified since r31 (the q-major sda repack).

### 2.3 Why bit-identity is constructed

The f32 stored in the plane must be bit-identical to the f32 the r41 scalar path computes on the fly, otherwise parity stops being a formality and becomes a bet. The construction guarantee comes from three points: f16→f32 is exact (`half::f16::to_f32`, i.e. `__half2float`); i8→f32 is exact; **exactly one f32 multiply, with FMA contraction forbidden on both sides** — `d * sc0` is a single multiply, and any contraction or reordering moves the last bit. The kernel side changed only "where this product comes from," not "how it is computed."

### 2.4 The even-od gate

The kernel issues 16 B chunks by **row pairs** and zero-fills whole pairs: a pair is either fully inside od or fully outside it (src-size 0 zero-fill). A concrete boundary: at od = 4N the last pair is rows (4N−2, 4N−1) and `full = (j+1 < od)` holds for both; at od = 4N+1 the last pair is (4N, 4N+1), where row 4N+1 does not exist — if it were still issued, row 4N's scale would be written "as a pair" with wrong content or overwritten by the zero fill, and the scale lost would be exactly that real odd row. Hence the registration gate adds `od % 2 == 0`, and odd tensors map-miss back to the r41 scalar path. The zero-fill granularity must equal the data structure's granularity (here = the row pair) — another form of the r44-class stride accident, blocked at registration time.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **One round bundles two mechanisms**, not two rounds: r53 already proved that changes "overlapping in traffic, non-overlapping in mechanism" can add (r44's "remove the work" + r45's "remove the wait"); A-cp.async (remove the wait) and W_dsc (remove the work) are exactly that thesis's next pair.
- **All the scaffolding is reused from r53**: the geometry-encoded sibling name (`{name}__dsc{od}x{id}`, guarding against a same-name different-shape stale plane being silently reused), the map keyed by the padded weight's device pointer, and on allocation failure an empty map + a loud eprintln once per process.
- **Riding r54's gate**: W_dsc registration hangs under `Q6K_NB && Q6K_EXP != "0" && id%256==0 (+ od%2==0)` — `EXP=0` opts out of **all** plane memory in one move (1.52 GB + 363 MB), so r54's "exit valve" semantics are not quietly broken by the new plane.
- **A null plane = the r41 scalar path**, byte-identical: raw weights, allocation failure, and odd od all land here.

### 3.2 Key code

The cp.async primitives and group ops (`src/cuda_kernels.cu`) — the src-size qualifier is the zero-fill mechanism:

```cuda
__device__ __forceinline__ void gemm_cp16(__half* smem_dst, const __half* gsrc, bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 16 : 0; // src-size 0 => zero-fill the 16B chunk
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}
__device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void gemm_cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void gemm_cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
```

The A side's cp.async conversion (inside the RAW_STAGE_Q6K_BT macro; r53's synchronous LDG→STS replaced; bytes unchanged — the prepass already fills the padded rows):

```cuda
/* ---- A: r56 cp.async bulk staging of the pre-transposed qa8/sda --*/
/* (r45's mechanism on top of r53: the sync LDG->STS exposed its     */
/* global latency at the top of every staging phase; cp.async hands  */
/* it to the async unit and the group wait below hides it under the  */
/* previous tile's compute. Bytes identical - the plane is always    */
/* full: the prepass zero-fills the padded rows.)                    */
{
    const size_t qbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_QASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 32) / 16; off += blockDim.x)
        gemm_cp16((__half*)(void*)(qa8b + (size_t)off * 16),
                  (const __half*)(const void*)(qa8g + qbase + (size_t)off * 16), true);
    const size_t sbase = ((size_t)blockIdx.x * nchunk + (size_t)(kt) * KDR) * MMQ_A_SDASZ;
    for (int off = threadIdx.x; off < (KDR * MMQ_NBI * 4) / 16; off += blockDim.x)
        gemm_cp16((__half*)(void*)(sdaqb + (size_t)off * 16),
                  (const __half*)(const void*)(sdag + sbase + (size_t)off * 16), true);
}
```

The dsc pairs' 16 B streamed copy (same macro; the scalar else branch keeps r41 verbatim):

```cuda
/* ---- B: dsc pair (d*sc[2c%16], d*sc[(2c+1)%16]) per (chunk,row) ----*/
/* r56: W_dsc f32 plane (registration-time precompute; chunk-major    */
/* layout plane[c*od + j] = float2(d*sc0, d*sc1)) turns the scalar    */
/* blk[192+..]/blk[208] loads + I2F (a leading r43 residual stall     */
/* post-r53) into a contiguous 16-B cp.async stream inside the same   */
/* per-kt commit group. Null plane = the r41 scalar path.             */
{
    const int c0d = (kt) * KDR;
    if (W_dsc != nullptr) {
        const int nc2 = MMQ_NBJ / 2; /* 16-B chunks (2 float2) per kd */
        for (int g = threadIdx.x; g < KDR * nc2; g += blockDim.x) {
            const int kdd = g / nc2, m = g % nc2;
            const int j = j0 + 2 * m;
            /* od even (registration gate) => a pair is either fully */
            /* valid or fully beyond od (src-size zero-fill).        */
            const bool full = (j + 1 < od);
            gemm_cp16((__half*)(void*)(sdsb + (size_t)kdd * MMQ_NBJ + 2 * m),
                      (const __half*)(const void*)(W_dsc
                          + ((size_t)(c0d + kdd) * (size_t)od + (size_t)j) * 8),
                      full);
        }
    } else { /* r41 scalar: d = h2f(blk[208]); dsc = d * (i8)blk[192+s0] ... */ }
}
...
gemm_cp_commit();   /* r56: commit moved to the END of RAW_STAGE — one group per kt covers A+B+dsc */
```

The main loop's wait structure (middle tiles wait1 / last tile wait0):

```cuda
RAW_STAGE_Q6K_BT(0, 0);
__syncthreads();
int buf = 0;
for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
    if (kt + 1 < nktile) {
        RAW_STAGE_Q6K_BT(kt + 1, buf ^ 1);
        /* two groups pending; wait until only kt+1's remains — group(kt)
         * (r56: A + dsc too, not just B) has landed, buf^1's copies stay
         * in flight under kt's compute (in-order group completion). */
        gemm_cp_wait1();
    } else {
        gemm_cp_wait0();  // last tile: drain every outstanding group
    }
    __syncthreads();  /* r53/r56: cross-thread visibility of the kt buffer */
```

The host-side `expand_q6k_dsc` (`src/cuda.rs`) — the comment's "bit-identical by construction" three elements are §2.3:

```rust
/// Output: `nchunk * od * 8` bytes, `out[(c*od + j)*8..+8]` =
/// float2(d*sc[2(c&7)], d*sc[2(c&7)+1]) — chunk-major so the kernel's
/// per-kt staging ... is a pure 16-B cp.async stream. Bit-identical to
/// the in-kernel r41 scalar computation: exact f16->f32 (half::f16, =
/// __half2float), exact i8->f32, one IEEE f32 multiply, no FMA
/// contraction on either side.
pub fn expand_q6k_dsc(padded: &[u8], od: usize, id: usize) -> Vec<u8> {
    ...
    let d = half::f16::from_bits(d_bits).to_f32();
    for cc in 0..8usize {
        let s0 = 2 * cc;
        let sc0 = blk[192 + s0] as i8 as f32;
        let sc1 = blk[192 + s0 + 1] as i8 as f32;
        let idx = ((sb * 8 + cc) * od + j) * 8;
        out[idx..idx + 4].copy_from_slice(&(d * sc0).to_bits().to_le_bytes());
        out[idx + 4..idx + 8].copy_from_slice(&(d * sc1).to_bits().to_le_bytes());
    }
```

The liveness label's A-side extension (current tree `src/cuda.rs`) — r56 also adds the A/dsc paths to the label r53 erected ("a fallback-correct fast path needs a visible label, parity cannot see it"); the census in §4 reads this composite output:

```rust
// r56: name the A/dsc staging paths too (liveness
// check per the r53 lesson — a fallback-correct fast
// path needs a visible label, parity cannot see it).
let a = if !w_dsc.is_null() {
    "A=cp.async DSC=f32-plane"
} else {
    "A=cp.async DSC=scalar"
};
eprintln!(
    "minfer/cuda: mmq raw NB-BT q6_K kernel active \
     (r56 {}, r54 B={}, r39 KDR=2 double-buffer, \
     A-transpose)",
    a, b
);
```

### 3.3 Pitfalls

- **`sudo -n` strips environment variables (this doc's most retellable pit)**: running the profile elevated via plain `sudo -n ncu ...` has sudo strip the user environment by default — gate variables like `MINFER_MMQ_Q6K_NB` all vanish, and ncu **silently profiles the legacy path**: the numbers "look normal" but measure a kernel that is not even running. Symptom chain: the filter was set to the NB-BT q6_K kernel name → the profiled data's shape looked unfamiliar → open the ncu report's **"Available Kernels" list and match the filter term: zero hits** — the NB-BT q6_K kernel is not in the list at all, meaning the gate was off. Fix: `sudo -E`, inline the env in the command (`sudo MINFER_MMQ_Q6K_NB=1 ncu ...`), or just run as the same user.
- **r53's 24 B stack disappears**: after r56 both instantiations land at 80 regs / 0 stack — planarizing the dsc incidentally unloads the scalar path's register pressure inside `<2,false>`; for the first time the two states' occupancy budgets match exactly.

- The "false regression" of suite 167/1: `cuda_conversation_multiturn_reuse` failed — `git stash` bisection verified it **fails the same way on a clean HEAD**: pre-existing, environment-sensitive, unrelated to this change. Not chased.
- A co-tenant outlier in the A/B distribution: with that point excluded, the distributions fully separate; without excluding it, the median gets dragged flat — the campaign's "distribution separation, not mean comparison" accounting saves the day again.

## 4. Verification

- **`cuda_q6k_dsc_dense_byte_exact`**: independent scalar mirror vs the host expander + device-plane read-back, 3 shapes, 0 mismatches — guards against plane byte misalignment (the r44-class stride error).
- **parity ×3**: guards against numeric path changes (per §2.3 this should be a construction guarantee; a routine confirmation).
- **greedy 453-character stream byte-identical**: guards against accumulated drift the parity fixtures cannot cover.
- **liveness census 27× `A=cp.async DSC=f32-plane, B=W_exp-cp.async`, 0 fallbacks**: guards against "parity/greedy all green but the fast path never ran" (r53's original lesson; r56 adds the A/dsc paths to the label too).
- **regs/stack and SASS check: both instantiations 80 regs / 0 stack, LDGSTS present**: guards against the explicit-PTX trap — if LDGSTS is missing from the SASS, the compiler degraded cp.async into a synchronous copy.
- **ncu kernel level**: ffn_down 12.76 → 12.01 ms (−5.9%), attn_v −4.2% — guards against attributing wall noise to kernel improvement.

## 5. Results

| Metric | before → after | Δ |
|---|---|---|
| whole prefill (distributions separated, one co-tenant outlier excluded) | 3138.6 → **3212.5 tok/s** | **+2.35%** |
| vs-llama (the 3324.4 anchor) | 1.05× → **1.035×** | — |
| ffn_down kernel | 12.76 → 12.01 ms | −5.9% |
| attn_v kernel | — | −4.2% |
| device memory | — | +363.2 MB (W_dsc = od·id/4) |
| registers/stack | r53's `<2,true>` 80/24B → both states **80 / 0** | — |
| suite | 167/1 (the only failure = `cuda_conversation_multiturn_reuse`, fails the same way on a clean HEAD, bisect-verified) | — |

The bundle thesis's second score: r45's mechanism **landed exactly as predicted** once the wall moved over — with B a pure copy, the A-side wait and dsc consumption are the wall, and dismantling both together moves the wall. Worth recording this score's "timing recipe": r45 (`9825ffd`, row 45) → (the wall moves) → r56 (`4cf7c74`), with the ten rounds r46–r55 in between; the mechanism was not forgotten in the interim, it was just waiting for the attribution to refresh. This differs from "abandoning a failed direction" — what should be abandoned is the **hypothesis** ("the A-side wait is now worth dismantling"), and what should be kept is the **verified mechanism** (cp.async staging itself).

This score also draws the boundary: basket logic is not a universal transplant template — the immediately following r57 (FA KV double buffering) and r58 (porting cp.async-db2 to q4_K) were both REVERTED (docs 60 and 61); a mechanism's value depends on what it replaces.

## 6. Lessons

1. **After the wall moves, yesterday's wall-neutral mechanism becomes today's positive gain** — a reverted lever is worth retrying after the attribution refreshes, provided it is re-attributed (the r45 → r56 arc followed "look at where the wall is" throughout).
2. **A registration-time plane must be "bit-identical by construction"**: exact conversions, exactly one multiply, FMA contraction forbidden on both sides — bit-level consistency backstopped by tests will not survive compiler version changes.
3. **`sudo` stripping the env is profiling's silent poison**: running ncu elevated strips the gate variables along with everything else, making you profile the legacy path while believing you're testing the new kernel; a zero-hit ncu "Available Kernels" list is its fingerprint.
4. **A paired-staging validity gate must align pair-wise** (od even): the zero-fill granularity must equal the data structure's granularity; a half-valid boundary row silently loses data.

---
← 58 · [Index](./README.md) · 60 →
