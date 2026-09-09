# 51 · r48 (FAP2) — register-resident softmax: deleting the S/P smem round trip outright (LANDED)

> **Result**: the S/P shared-memory round trip in `fa_prefill_f16kv` is deleted outright — softmax runs directly on the QK^T wmma accumulator fragments, and P is built in registers, in place, as the matrix_a operand of P·V. FA kernel **5.16 → 2.12 ms (2.43×**, meeting the ≤2.6 ms target), whole-prefill **2603.5 → 2749.9 tok/s (+5.6%)**, FA share of the wall **10.2% → ~4.7%**, vs-llama 1.27× → 1.21×. ncu: MIO/shared-scoreboard stalls vanish (top stall becomes global K/V loads), shared wavefronts 44% → 21%, 124 regs / 0 spill. Gates: parity ×3 green, greedy-32 byte-identical, suite 166/0/3.
> **Commit**: `d38744d` (code, `src/cuda_kernels.cu` only) + `7e2ee62` (record). **Date**: 2026-09-06.

## 1. Background — where things stood

r46 (FAP1) is the direct predecessor of this step. The audit confirmed that the FA prefill kernel (`fa_prefill_f16kv`, introduced in the 8n era, reworked through the P5.0/P5.3 rounds) **already was** wmma m16n16k16 + online softmax, not a scalar implementation; the real defects were two: 69.38 KB of smem → 1 block/SM (16.64% occupancy), and a bank-conflicted **S/P smem round trip** (S written to smem, softmax read out, P written back, P·V read back again, with row strides landing in the same bank group). FAP1's lever was conservative: FA_TKV 64→32 + S/P row padding + a launcher over-allocation fix — kernel 5.16 → 4.58 ms (−11%), the mechanism was entirely positive, but **the wall gained only +0.27%**, below the +1.5% bar, REVERTED. FAP2 survived as a name: "register-resident softmax, eliminate the S/P round trip for good."

Then r47's converged-regime wall decomposition changed FA's situation: once the q4_K/q6_K GEMMs both reached parity (1.06×/1.13×), FA at 125.8 ms = **5.72× and 10.2% of GPU busy time** surfaced as the #1 structural residual — r46's "FA slice is smaller / gets overlapped" hypothesis was falsified. r47 used that same audit to run the budget: if deleting the S/P round trip buys a ~2× kernel improvement, that is −63 ms = **−4.9% of the wall**, clearing the bar three times over. FAP2 moved up from "mechanically worth doing" to "wall-level top priority."

In other words, the task facing r48 was very concrete: no longer another occupancy patch on the old structure, but tearing down the structural assumption that "S must pass through shared memory" itself. In the old structure softmax cooperated across warps on a per-row basis, so the data had to land in smem; to make it register-resident, the geometric relationship between warps and data had to change first.

## 2. Principle — the GPU mechanism

**Why S/P went through smem — a geometry problem, not an implementation problem.** wmma accumulator fragments live only in a single warp's registers. The old geometry was 256 threads (8 warps), warp mapping `wm = warp>>1` (four 16-row q blocks) × `wn = warp&1` (the left/right halves of the FA_TKV=64 columns): one q block's S was computed **by two warps cooperating**, while online softmax needs the max/sum over the **whole row** — the data had to be handed across warps, and smem was the only channel. The per-KV-tile round-trip bill: S store (64×64 f32 = 16 KB) + softmax read (16 KB) + P store-back (64×64 f16 = 8 KB) + ldmatrix read for P·V (8 warps × 4 k-steps × 512 B = 16 KB) ≈ **56 KB/tile of smem traffic**, plus small round trips for the three state arrays msh/lsh/alpha, and the store-side row stride ≡ 0 mod 32 banks (8-way conflict). r46 tried to treat the conflict with padding and measured only −11% kernel — **the round trip itself was still there**.

**The new geometry: one warp owns a full row block.** r48 changes the tile to 4 warps (128 threads) × FA_TQ=64 rows × FA_TKV=32 columns: each warp exclusively owns one 16-row q block × **all 32 KV columns**. The QK^T wmma accumulator `fc[2]` (two 16×16 fragments) is now warp-private — softmax can stay in registers, and so can P.

**4-lane row groups and the butterfly.** Documented layout of the m16n16k16 f32 accumulator: lane L holds rows `L>>2` and `(L>>2)+8`, columns `2·(L&3)`, `2·(L&3)+1` (and the +8 offsets). So all 32 columns of one row land in exactly **4 lanes** (`l = 0..3`). The in-row max/sum reduction needs only a two-step butterfly with `__shfl_xor` offsets 1 and 2 — the old implementation was a full-warp 32-lane five-step reduction (offsets 16..1). Each lane maintains the `m/l` state for two rows (`m0/m1/l0/l1`), fully register-resident; the three smem arrays `msh/lsh/alpha` disappear entirely.

**The f32 accumulator ↔ f16 matrix_a lane-map identity.** The A operand of P·V is an f16 `matrix_a` fragment (m16n16k16 row_major), and its element→`x[i]` mapping is **exactly the same** as the f32 accumulator fragment's (the PTX ISA specifies the same per-lane layout for both fragment kinds). This identity (flagged unit-validated in the commit message) means P needs no sts/lds/shuffle at all: write `__float2half(p)` straight into `pa[cc].x[i]` and the same register becomes the A operand of the next wmma in place. The S→P "round trip" degenerates into an in-register type conversion.

**The register ledger: why 124 regs fit.** Per-warp resident fragment state (approximate accounting): O accumulator `acc[8]` = 8 × 8 = 64 f32; QK^T accumulator `fc[2]` = 16 f32; P operand `pa[2]` = 16 f16 (8 32-bit registers); softmax linearized arrays `sm/sm1_/gcol` = 8+8+8 = 24; `m/l` state and alpha 8 more; ~120 live registers in total — matching the measured **124 regs / 0 spill**. The same ledger explains two design boundaries: why each warp can only take FA_TKV=32 (`fc`/`pa`/`sm` all grow linearly with FA_TKV; 64 would need +60 regs and would certainly spill), and why doubling the O accumulator from the old `acc[4]` to `acc[8]` did not blow the budget — the old implementation gave each warp only the wn half-width (64 output columns), the new one gives each warp **all 128 output columns** of its full row block; what the width costs is paid for by deleting the S/P smem round trip, and the net ledger is favorable.

**The major-ness trap.** The B operand of QK^T is K^T (k = head dim, n = KV position), and K is physically stored row-major as `[kv][hd]` — which for wmma is exactly a **col_major B** (`B[k][n] = ptr[n·ldm + k]`); the B operand of P·V is V itself (k = KV position, n = head dim), where the same physical layout must be a **row_major B**. The two consumers read the same tile along two different axes, so their major-ness flags are necessarily opposite. During development both were written as row_major: QK^T actually computed Q·K (untransposed) and the parity max err blew up from the 1e-4 class **to 0.54**, recovering only when K was set back to col_major.

**Barrier and occupancy reconciliation.** The per-KV-tile `__syncthreads` count drops from 4 to 3 (the one after QK^T, which existed to serve softmax's cross-warp reads of Sf, now has no readers). smem 69.38 → 34.82 KB = `(FA_TQ + 2·FA_TKV)·(hd+8)·2` = 128·136·2 = 34,816 B, blocks go 1 → 2/SM. Note that warp-level occupancy did not change: before, 1 block × 8 warps; now, 2 blocks × 4 warps — 8 warps/SM either way. **The gain comes from two places**: the disappearance of the MIO/shared-scoreboard round trip (the 36% item in ncu goes straight to zero), and the phase interleaving of two independent blocks (while one block waits at a barrier the other can issue; in the single-block era a barrier idled the whole SM). r46 got −11% from occupancy alone (2 blocks × 8 warps); r48 got 2.43× by "deleting work" — a clean contrast.

**The mechanism fingerprint in ncu.** shared wavefronts fell from 44% to 21% — the remaining 21% is the smem write/read of Q/K/V staging itself (the part that was always supposed to be there); the vanished half corroborates the zeroing of the MIO/shared-scoreboard stalls: **the S/P round-trip traffic evaporated at the counter level, it was not hidden in some other bucket**. The top stall becoming L1TEX scoreboard (global K/V loads) says the bottleneck has ceded to where data genuinely moves — a kernel whose top stall changes from "traffic it built itself" to "inputs it must move" is the healthy signature of a structural change.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Full-row warp tile (4 warps) instead of keeping 8 warps**: a cross-warp distributed-fragment softmax (lane reductions spanning warps, P reassembled via shuffles) is theoretically possible, but it would trade the 4-lane butterfly for a mixed smem/shuffle protocol between warps, and the complexity returns to the starting point. 4 warps × 16 rows = 64 rows = FA_TQ — the geometry closes.
- **FA_TKV=32 is part of the new geometry**, not just an occupancy knob: each warp's accumulator is `acc[8]` (covering all output columns of hd=128; the old implementation gave each warp only the half-width `acc[4]`), and a 124-reg / 0-spill budget exactly fits; any larger FA_TKV doubles the fragment count of `fc[]`/`pa[]` and blows the registers; any smaller was the direction r50 later falsified.
- **128 threads remain enough for staging**: per-tile K/V staging = 2 × FA_TKV × hd × 2 B = 16 KB, spread over 128 threads that is 128 B/thread = 8 uint4 cp.async per thread — issue width is not lacking; and once softmax is register-resident, the extra 4 warps have no work to do on S and only widen the barrier-arrival wait surface.
- **Staging stays cp.async + padded stride (`sstr = hd+8`)**, and the launcher over-allocation r46 fixed is rebuilt with the new formula. The pre-sm80 build target (sm_75) keeps the synchronous staging fallback path, and its loop stride is parameterized on `nthreads` the same way (see the sync branch of `fa_stage_kv_async` in the current tree) — the same class of trap must be plugged on both paths.
- **The tail tile's O write-back reuses smem**: the Qs/Ks/Vs region is idle after the KV loop, and 64×128 f32 = 32 KB < the 34.8 KB budget (this reuse later became r50's launcher lesson: two smem users must take the max).

A term-by-term comparison of old vs new geometry (this table is the diff's "semantic summary"):

| Dimension | Old (pre-r48) | New (r48) |
|---|---|---|
| Threads / warps | 256 / 8 | 128 / 4 |
| Warp mapping | `wm=warp>>1` × `wn=warp&1` (half column block) | `wm=warp` (full row block × all columns) |
| Where S lives | accumulator → **Sf smem** (f32, 64×64) | **accumulator fragment `fc[]`** (registers throughout) |
| Softmax shape | one row per warp, 32-lane 5-step reduction | 4-lane row groups, 2-step butterfly, 2 rows per lane |
| Where P lives | **Pf smem** (f16, FA_PSTR) → ldmatrix read back | **matrix_a fragment `pa[]`** (built in place) |
| m/l/alpha | msh/lsh/alpha smem arrays | registers `m0/m1/l0/l1` |
| O accumulator | `acc[4]` (half width, 64 columns per warp) | `acc[8]` (full width, 128 columns per warp) |
| Barriers per tile | 4 | 3 |
| smem layout | Qs+Ks+Vs+Sf(+Pf alias)+msh/lsh/alpha | **Qs+Ks+Vs** (34,816 B) |

### 3.2 Key code

**Old structure (`d38744d^`, 8 warps splitting the columns in half):** each warp computes only a 16×32 half of the column block, and S must land in Sf via `store_matrix_sync`:

```cuda
const int warp = tid >> 5; // 0..7
const int wm = warp >> 1;  // q 16-block: 4
const int wn = warp & 1;   // 64-dim chunk: 2
...
    wmma::mma_sync(fc[0], fa, fb[0], fc[0]);
    wmma::mma_sync(fc[1], fa, fb[1], fc[1]);
    }
    // S lands in smem: row stride FA_TKV=64 f32 = 256B ≡ 0 mod 32 banks (the conflict source)
    wmma::store_matrix_sync(&Sf[swm * 16 * FA_TKV + swk * 32],      fc[0], FA_TKV, wmma::mem_row_major);
    wmma::store_matrix_sync(&Sf[swm * 16 * FA_TKV + swk * 32 + 16], fc[1], FA_TKV, wmma::mem_row_major);
}
__syncthreads();   // softmax is a cross-warp consumer; all of S must be in place
```

The old softmax was "one row per warp" (8 warps sweeping the 64 rows in turn), each lane holding 2 columns with a full-warp five-step reduction; P was written back to Pf and m/l/alpha written back to smem:

```cuda
for (int rr = warp; rr < FA_TQ; rr += 8) {   // one warp per row, 8 rows/warp
    int c0 = lane * 2, c1 = c0 + 1;
    float s0 = v0 ? Sf[rr * FA_TKV + c0] : -INFINITY;   // read S back
    float m_new = fmaxf(s0, s1);
    for (int off = 16; off > 0; off >>= 1)              // full-warp 5-step reduction
        m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffffu, m_new, off));
    float m_old = msh[rr];                              // state in smem too
    a = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
    Pf[rr * FA_PSTR + c0] = __float2half(p0);           // P written back to smem
    ...
    if (lane == 0) { lsh[rr] = lsh[rr] * a + sum; msh[rr] = m_new; }
    if (lane == 0) alpha[rr] = a;                       // alpha broadcast back to smem
}
__syncthreads();   // P·V's ldmatrix readers live in other warps
```

After that, P·V also had to `load_matrix_sync` Pf back from smem — one trip there, one trip back; S and P were each moved twice.

**New structure (current tree, `src/cuda_kernels.cu`; the kernel is unchanged since `d38744d`):** the warp mapping and all state move into registers — each lane derives the two rows and column group it holds from the fragment layout:

```cuda
// FAP2: 4 warps (128 threads), warp wm owns a full 16-query-row block x
// all FA_TKV KV columns. S lives in wmma accumulators and is softmaxed in
// place; P is converted f32->f16 into the A fragment of P@V — the S/P
// shared round trip and its bank conflicts are gone entirely.
const int warp = tid >> 5;     // 0..3
const int wm  = warp;          // 16-query-row block
const int lane = tid & 31;
const int l  = lane & 3;       // 4-lane row group
const int r0 = lane >> 2;      // fragment local row 0 (0..7)
const int r1 = r0 + 8;         // fragment local row 1
const int c0 = 2 * l;          // cols (2l, 2l+1, 2l+8, 2l+9) per fragment
...
wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];  // full hd=128 output
float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;   // state in registers
```

QK^T: K stays a **col_major** B (= the K^T semantics), and S stays directly in the accumulator fragment `fc[]`:

```cuda
wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[FA_TKV / 16];
for (int d = 0; d < hd; d += 16) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[FA_TKV / 16];
    for (int cc = 0; cc < FA_TKV / 16; cc++)
        wmma::load_matrix_sync(fb[cc], &Ks[cc * 16 * sstr + d], sstr);
    wmma::load_matrix_sync(fa, &Qs[wm * 16 * sstr + d], sstr);
    for (int cc = 0; cc < FA_TKV / 16; cc++)
        wmma::mma_sync(fc[cc], fa, fb[cc], fc[cc]);   // S stays resident in fc
}
__syncthreads();   // the old structure had its 2nd barrier here (cross-warp Sf reads) — no longer needed
```

Online softmax on the fragments: first linearize `fc[].x[i]` into `(fragment, quad)` arrays and record each element's global column number (masking has to be re-evaluated per element); the in-row reduction uses only the 4-lane butterfly:

```cuda
float sm[FA_TKV / 16 * 4], sm1_[FA_TKV / 16 * 4];
int gcol[FA_TKV / 16 * 4];
for (int cc = 0; cc < FA_TKV / 16; cc++) {          // x[0,1,4,5]→row r0
    sm[quad+0] = fc[cc].x[0]; ... sm1_[quad+3] = fc[cc].x[7];
    gcol[quad+0] = kt + cc * 16 + c0;  ...           // global KV column numbers
}
for (int q = 0; q < FA_TKV / 16 * 4; q++) {          // causal + range masking
    bool v0 = (gcol[q] <= qpos0) && (gcol[q] < kv_end);
    if (v0) mnew0 = fmaxf(mnew0, sm[q]); ...
}
for (int off = 1; off <= 2; off <<= 1) {             // 4-lane row group, 2 steps
    mnew0 = fmaxf(mnew0, __shfl_xor_sync(0xffffffffu, mnew0, off));
    mnew1 = fmaxf(mnew1, __shfl_xor_sync(0xffffffffu, mnew1, off));
}
const int fresh0 = (m0 == -INFINITY);
float a0 = fresh0 ? 0.0f : __expf(m0 - mnew0);
if (mnew0 == -INFINITY) a0 = 1.0f;                   // fully masked tile: state untouched
...
for (int q = ...) { p0[q] = valid ? __expf(sm[q] - mnew0) : 0.0f; sum0 += p0[q]; }
for (int off = 1; off <= 2; off <<= 1) sum0 += __shfl_xor_sync(..., sum0, off);
if (mnew0 != -INFINITY) m0 = mnew0;
l0 = l0 * a0 + sum0;                                 // m/l update, no smem
```

Then the pivotal move of the whole piece: **P is built in place as the A operand of P·V**. The f32 accumulator and the f16 row_major matrix_a have identical per-lane layouts, so `pa.x[i] = __float2half(p)` writes element by element; the O accumulator is rescaled by per-row alpha (same layout: `x[0,1,4,5]`→row r0, `x[2,3,6,7]`→row r1); V for P·V is a **row_major** B:

```cuda
// rescale O by per-row alpha (m16n16 f32 accumulator layout)
for (int ob = 0; ob < 8; ob++) {
    acc[ob].x[0] *= aa0; acc[ob].x[1] *= aa0;   // row r0
    acc[ob].x[2] *= aa1; acc[ob].x[3] *= aa1;   // row r1
    acc[ob].x[4] *= aa0; acc[ob].x[5] *= aa0;
    acc[ob].x[6] *= aa1; acc[ob].x[7] *= aa1;
}
// Build the P@V A-operand IN PLACE: matrix_a m16n16k16 row_major and the
// f32 accumulator use the SAME (row,col) layout → pa.x[i] = f(p_i), element by element.
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa[FA_TKV / 16];
for (int cc = 0; cc < FA_TKV / 16; cc++) {
    pa[cc].x[0] = __float2half(p0[quad + 0]);
    pa[cc].x[1] = __float2half(p0[quad + 1]);
    pa[cc].x[2] = __float2half(p1[quad + 0]);   // x[2,3] is row r1
    ...
}
// acc = acc*alpha + P·V; V (B) is row_major — K in QK^T is col_major,
// the two operands' major-ness is opposite (both row_major ⇒ both matmuls wrong).
for (int kk0 = 0; kk0 < FA_TKV; kk0 += 16)
    for (int ob = 0; ob < 8; ob++) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
        wmma::load_matrix_sync(vb, &Vs[kk0 * sstr + ob * 16], sstr);
        wmma::mma_sync(acc[ob], pa[kk0 / 16], vb, acc[ob]);
    }
__syncthreads();   // 3 barriers per tile (was 4)
```

Launcher: the smem formula shrinks from "Qs+Ks+Vs+Sf/Pf" to pure staging, and the r46 fix's explicit opt-in and failure fallback are preserved (NOT silent):

```cuda
// Qs + Ks + Vs only (S/P no longer go through shared memory). sstr = hd+8
// padding; Ks/Vs are FA_TKV rows (the r46 launcher's 3*FA_TQ bug is gone).
size_t smem = ((size_t)FA_TQ + 2 * FA_TKV) * (hd + 8) * 2;   // 128*136*2 = 34,816 B
...cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, (int)smem);  // failure → fall back to the old kernel
dim3 grid((nt + FA_TQ - 1) / FA_TQ, nh, 1);
fa_prefill_f16kv<<<grid, 128, smem, stream>>>(...);          // 256 → 128 threads
```

### 3.3 Pitfalls

1. **The double major-ness mistake** (detailed in §2): the B operands of QK^T and P·V share the same physical layout but are semantic transposes of each other, so the flags must be one col_major and one row_major. The symptom is extremely misleading: the kernel "runs", and parity max err goes from the 1e-4 class to 0.54 — not a precision regression but wrong math. After the fix, both lane maps were validated standalone before integration.
2. **Halving the threads vs hardcoded strides**: the staging loop's stride was originally written for 256 threads; changing the launch to 128 without parameterizing the stride (`fa_stage_kv_async(..., tid, 128)`, Q loads `i += 128`) means cp.async moves only **half** of each K/V tile and the other half is uninitialized smem — silently wrong data. r46's launcher over-allocation was the same class of "parameters didn't follow the geometry" trap.
3. **Fragment masks must be recomputed per element**: in the old row loop a lane's column numbers were contiguous `c0/c1`; in the new layout one lane's 32 columns are scattered as `(2l, 2l+1, 2l+8, 2l+9)` per fragment, so the causal mask and the `kv_end` range mask must be evaluated element by element on each `gcol[q]`, and a fully masked tile keeps the −INF state (alpha=1, P=0, l untouched) — the semantics were aligned branch by branch with the old implementation, and greedy-32 byte-identical confirms it.

## 4. Verification

| Gate | Numbers | What it defends against |
|---|---|---|
| `cuda_fa_prefill_attention_parity` (`src/graph/cuda_backend.rs`) | 1/0 green | point-by-point comparison of a small GQA graph (nh=4, nk=2, hd=128, nt=100, f16 KV) against the reference — pins both lane maps (the accumulator rescale mapping + P's matrix_a identity) and the major-ness combination, catching "it runs but the math is wrong" (the major-ness bug's 0.54 is exactly what it caught) |
| `cuda_prefill` | 7/0 green | whole-graph prefill numeric regression, catching the kernel swap changing downstream nodes (rms/rope/FFN chains) in a real layer sequence |
| `cuda_prefill_mmq_parity` | 1/0 green | numerics when the MMQ GEMM path coexists with FA in the same graph, catching breakage from attention-side changes coupling into the GEMM side |
| greedy-32 byte-identical | byte-for-byte | catches ULP-level reordering drifting into the sampler; r48's rewrite preserved it (r50/r57 later proved FA tile-size changes do not always preserve it — see doc 53) |
| suite | 166/0/3 | full regression (CPU/Metal/graph paths), catching CUDA-layer changes leaking into shared code |
| ncu reconciliation | MIO stall zeroed, shared wavefronts 44→21%, 124 regs/0 spill | catches "faster but mechanism unexplained" — confirms the gain really comes from deleting the S/P round trip, and that 5.16 → 2.12 ms clears the preset ≤2.6 ms target line |

## 5. Results

| Level | before | after | Δ |
|---|---|---|---|
| FA kernel (3325-tok prefill, ncu) | 5.16 ms | **2.12 ms** | **2.43×** |
| whole-prefill (same-window interleaved ×3, median) | 2603.5 tok/s | **2749.9 tok/s** | **+5.6%** |
| FA share of the wall | 10.2% | **~4.7%** | halved |
| vs-llama (3325-eq anchor) | 1.27× | 1.21× | −0.06× |
| smem / block | 69.38 KB (1 block/SM) | 34.82 KB (2 blocks/SM) | −50% |
| barriers per tile / smem S/P traffic | 4 / ~56 KB | 3 / **0** | round trip deleted |

r47's prediction was 2× → −63 ms (−4.9% of the wall); measured, FA saved ~68 ms and the wall gained +5.6% — the prediction landed and slightly exceeded. whole-prefill's +5.6% is far past the +1.5% bar, and this is the campaign's first case of a structural ">2× residual" being removed wholesale.

The same S/P defect, two fixes — the gap between the final numbers is exactly the gap between "working around" and "deleting":

| Route | smem | blocks/SM | S/P round trip | kernel | wall |
|---|---|---|---|---|---|
| r46 (FAP1): FA_TKV 64→32 + row padding | 69.38 → 43.8 KB | 1 → 2 | kept (conflict reduced) | 5.16 → 4.58 ms (−11%) | +0.27% (REVERTED) |
| r48 (FAP2): full-row warp tile + register softmax | 69.38 → 34.82 KB | 1 → 2 | **deleted** | 5.16 → **2.12 ms (2.43×)** | **+5.6% (LANDED)** |

Follow-on coordinates: r49 trimmed the prepass further (+2.32%); the FA line was not touched again until r50/r57 — both times it hit the greedy-identity wall on tile size, and r48's geometry remains FA's stable chassis (r50's launcher lesson — the 32 KB of tail-tile O write-back smem reuse must be counted in the launcher's max — is the sequel to the last item in this doc's §3.1).

## 6. Lessons

1. **Deleting a round trip beats optimizing around it**: r46's padding on the S/P round trip bought kernel −11%; r48 deleting the round trip outright is 2.43×. Rounds where the mechanism is positive but the wall is negative (r46) are usually patches on a structure that should have been deleted.
2. **Register residency is a geometric property**: whether softmax can stay in registers depends on whether a warp exclusively owns complete data rows — change the warp×tile geometry first, then talk about "keeping data out of memory." Keeping the data layout and only swapping load/store instructions never yields this magnitude.
3. **Judge wmma operand major-ness per matmul**: the same physical memory is a col_major B in QK^T and a row_major B in P·V — the two consumption directions are mutual transposes. Validate lane maps standalone before integrating; "the kernel runs" is not evidence, parity numbers are.
4. **Halving threads requires auditing every loop unrolled by thread count**: constants that "follow the launch geometry" like staging strides, once hardcoded, turn a thread reduction into silently dropping half the data.

---
← [50 · r47 converged-regime wall decomposition](./50-r47-converged-wall-decomposition.md) · [Index](./README.md) · [52 →](./52-r49-a-quantize-shared-dedup.md)
