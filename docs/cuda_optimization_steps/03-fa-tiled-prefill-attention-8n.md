# 03 · 8n — FA-style tiled prefill attention (LANDED)

> **Result**: 7B q4_k_m @2K prefill attention **176 → 8.5 ms/layer (20×)**; K traffic **~132 GB → ~0.8 GB/layer**. This step broke through the entire post-8m prefill wall — the 1082 tok/s baseline in the 8m② commit window owes most of itself to this.
> **Commit**: `cb66fca`. **Date**: 2026-08-30 (same day as 8m: landed 30 minutes after 8m and 20 minutes before 8m②).

## 1. Background — where things stood

Once 8m swapped the prefill GEMM for tiled wmma, the wall's composition flipped instantly: attention became the overwhelming majority. The old kernel `gqa_attn_f32_f16kv` measured **176 ms/layer** at 7B @2K — **76%** of the entire 2K prefill wall (28 layers × 176 ms ≈ 4.9 s, against a whole wall of ~7.0 s at 294 tok/s). The GEMM was already running at 31 TFLOPS; continuing to optimize it would have been the wrong next target — the 76% had to fall first.

The old kernel's disease is the same one decode attention later cured (R4): **one block per (token, head)**. One block per q-head per token, and that block must read K in full — every byte of K is re-read nt × (q heads per kv head) times, ~132 GB/layer at 7B @2K, all burned on the L2/DRAM transport side. Meanwhile the hd-wide output accumulator `float4 oc[32]` occupies 128 registers and triggers spills — the same disease as the "LOCAL-memory accumulator = ~80 MB/layer of local traffic" entry in R4's later decode table.

The fix held no suspense: FlashAttention had already established "tile the q dimension + online softmax" as the standard shape. minfer's KV cache has stored f16 since the 7e series, so Q/K/V can feed tensor cores directly for QK^T. 8n's job was to land that shape on CUDA: **one block per (64-token q tile, head)**, K/V entering shared memory tile by tile, S = Q·K^T on wmma, softmax in online form, the O accumulator resident in registers.

## 2. Principle — the GPU mechanism

### 2.1 Online softmax

Naive attention must finish computing the **entire row** of scores before softmax (it needs the full-row max and full-row sum), which means the S matrix materializes at full size. Online softmax makes it incremental: KV is processed tile by tile, and each row keeps three pieces of state — running max `m`, running sum `l`, output accumulator `O`. When processing tile k:

1. Find the new max within the tile: `m_new = max(m_old, max(S_tile))`;
2. Rescale factor `alpha = exp(m_old − m_new)`; multiply the old O by alpha and the old l by alpha;
3. In-tile probabilities `p = exp(S − m_new)`, accumulated into O and l.

After all tiles are processed, `O / l` is the correct softmax-weighted output. Mathematically an identity transformation; the price is one O rescale per tile — in exchange, O and S both need only tile-sized storage and K/V is consumed as a stream.

### 2.2 The byte ledger: 132 GB → 0.8 GB

Old shape: every (token, head) block reads all of K/V → ~132 GB per layer. For scale: reading the full 2048×2048 K matrix (hd=128, f16) once is 2048×2048×128×2 B ≈ 1 GB — the old kernel's re-read factor is exactly the nt × heads order of magnitude, with V doubling it again. Tiled shape: a block's 64 tokens share the same K/V tiles (reused 64 times once in shared memory), so K/V's effective read volume is diluted by the q-tile width; with blocks for the same kv-head hitting L2 against each other, the measured figure lands at **~0.8 GB/layer**. That 20× transport difference is the main source of 176 → 8.5 ms — at this scale attention is a bandwidth war just like the GEMM.

### 2.3 Register O and the thread geometry

Inside a block, 256 threads = 64 rows × 4 quadrants: each thread owns the f32 accumulator `acc[32]` for **one row's 32 dims** (a quarter of hd=128). Keeping O in registers pays twice: the alpha rescale is a pure register operation (no reading O back from shared memory, scaling, writing back), and every V read in P·V is an intra-warp broadcast. The code comment, verbatim: *Keeping O in registers (instead of shared) makes every P·V V-read a warp-wide broadcast and the alpha reads conflict-free.*

QK^T uses wmma: 8 warps split the 64×64 S matrix by `wm = warp>>1` (4 q 16-blocks) × `wk = warp&1` (2 kv 32-blocks), and each warp runs `mma_sync` over the hd loop (f16 inputs, f32 accumulation). P·V is still a scalar FMA loop in this step (P stored f16, V f16, acc f32) — moving P·V onto tensor cores was a separate later step, P5·0 (10.06 → 4.24 ms/layer).

### 2.4 f16 probs and the 256 B stride

The post-softmax probabilities P are stored f16, directly **aliasing** the score matrix `Sf`'s memory — scores are dead data once softmax has consumed them. This alias is where a 64×64 buffer is saved from the 97 KB smem budget, and it is also this step's only correctness mine (see §3.3): P's row stride must be **256 B** (`FA_PSTR = FA_TKV*2` halves), so row r's probabilities overlap only the first half of row r's own scores.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Grid `(nt/64 q tiles, heads)`, 256 threads, dynamic smem ~65 KB** (actual allocation 66,304 B: Qs/Ks/Vs at 64×128×2 B = 16 KB each, Sf 64×64×4 B = 16 KB, m/l/alpha 768 B). Over the static limit, so it follows 8m's procedure with the `cudaFuncSetAttribute` opt-in. (The layout table in the kernel's header comment says "~97 KB" and lists a row `O [64*hd] f32` — a leftover from the draft period when O still lived in shared memory: the actual code moved O into registers and the launcher's allocation expression has no O. Small drift of this kind between comment and code is itself worth recording.)
- **The Q tile lands as f16 with the scale folded into Q**: `q * scale` is computed once at load, so QK^T's inner loop no longer multiplies by scale every step — what is saved is scalar work inside the wmma loop; the price is that Q's f16 rounding happens early (numerically absorbed into the 5e-3 tolerance gate).
- **S/P alias**: saves a 64×64 buffer; correctness is secured by the program-order argument for the 256 B stride (§3.3).
- **GQA slicing**: q head `h` reads only the K/V slice of kv head `h/gqa` (`stride_kv = nk*hd`), not all of K — 7B is 28:4, and each kv head's K/V is shared among its 7 q heads' tile blocks.
- **`kv_end = positions[last_t] + 1`**: KV positions are data, not structure (graph rule §1) — the kernel reads the position of the q tile's last token from the `positions` input, and every KV row beyond that bound is skipped.
- **Gated on `hd == 128`** (the Qwen2.5/2.5-7B shape); other head dims take the old path.
- Masked positions are handled inside the data flow: the score comparison `kv_g <= qpos`; a fully-masked tile writes 0 probabilities and keeps the softmax state untouched.

### 3.2 Key code

All excerpts below come from the original `fa_prefill_f16kv` introduced by `cb66fca` (in the current tree this kernel has since evolved through the P5·0/P5·3/FAP2 series into the wmma P·V + register-softmax version, but this chapter's tile geometry, online-softmax state machine, and race argument survive unchanged to this day).

**The smem layout and tile constants**:

```cuda
// Shared layout (dynamic, ~97 KB — opt-in via cudaFuncSetAttribute):
//   Qs [64*hd] f16   q tile (scale folded in, f16 for the tensor-core QK^T)
//   Ks [64*hd] f16   K tile           Vs [64*hd] f16  V tile
//   S  [64*64]       f32 scores, aliased as f16 probs after the row softmax
//   m/l/alpha [64] f32 per-row online-softmax state
#define FA_TQ 64
#define FA_TKV 64
#define FA_PSTR (FA_TKV * 2) // probs row stride in halves (256B): probs row r
                             // aliases only Sf row r's first half, already read
                             // by the same thread — no cross-thread race
#define FA_HQ 32 // hd/4 dims per accumulator thread (kernel is gated to hd == 128)

__global__ void fa_prefill_f16kv(
    const float* __restrict__ q, const __half* __restrict__ k,
    const __half* __restrict__ v, float* __restrict__ o,
    const int* __restrict__ positions,
    int nh, int nk, int hd, float scale, int nt
) {
    extern __shared__ __align__(256) uint8_t smem[];
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + FA_TQ * hd;
    __half* Vs = Ks + FA_TKV * hd;
    float* Sf = reinterpret_cast<float*>(Vs + FA_TKV * hd);
    __half* Pf = reinterpret_cast<__half*>(Sf); // alias: probs after softmax
    float* msh = reinterpret_cast<float*>(Sf + FA_TQ * FA_TKV);
    float* lsh = msh + FA_TQ;
    float* alpha = lsh + FA_TQ;
```

**Q tile load (scale folded in + tail zeroing) and K/V tile staging (16B/lane, out-of-bounds zero-fill)**:

```cuda
    // load q tile (scale folded in) as f16
    for (int i = tid; i < FA_TQ * hd; i += 256) {
        int r = i / hd, d = i % hd;
        int t = tq0 + r;
        float qv = (t < nt) ? q[(size_t)t * ne_q + h * hd + d] * scale : 0.0f;
        Qs[i] = __float2half(qv);
    }
    if (tid < FA_TQ) {           // per-row online-softmax state init
        msh[tid] = -INFINITY;
        lsh[tid] = 0.0f;
    }
    …
    // stage K/V tile (16B per lane; rows beyond kv_end zero-filled)
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    for (int i = tid * 8; i < FA_TKV * hd; i += 2048) {
        int r = i / hd, d = i % hd;
        int p = kt + r;
        if (p < kv_end) {
            kk4 = *reinterpret_cast<const uint4*>(&k[(size_t)p * stride_kv + hk * hd + d]);
            vv4 = *reinterpret_cast<const uint4*>(&v[(size_t)p * stride_kv + hk * hd + d]);
        } else {
            kk4 = z4;  vv4 = z4;   // zero-fill out-of-bounds KV rows → S=0, backstopped again by the mask logic
        }
        *reinterpret_cast<uint4*>(&Ks[i]) = kk4;
        *reinterpret_cast<uint4*>(&Vs[i]) = vv4;
    }
    __syncthreads();
```

**Register O and thread ownership**:

```cuda
    // Per-thread output accumulator: thread owns (row, quadrant) with
    // row = tid & 63, quadrant = tid >> 6 (FA_HQ dims each). Keeping O in
    // registers (instead of shared) makes every P·V V-read a warp-wide
    // broadcast and the alpha reads conflict-free.
    float acc[FA_HQ];
#pragma unroll
    for (int dd = 0; dd < FA_HQ; dd++) acc[dd] = 0.0f;
    __syncthreads();

    const int last_t = min(nt - 1, tq0 + FA_TQ - 1);
    const int kv_end = positions[last_t] + 1;   // KV positions are data
    const int arow = tid & (FA_TQ - 1);
    const int aquad = tid >> 6; // 0..3
```

**QK^T on the tensor core** (each warp computes one 16×32 sub-block of S):

```cuda
        using namespace nvcuda;
        int warp = tid >> 5;       // 0..7
        int wm = warp >> 1;        // q 16-block: 4
        int wk = warp & 1;         // kv 32-block: 2
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2];
        for (int d = 0; d < hd; d += 16) {
            wmma::load_matrix_sync(fa, &Qs[wm * 16 * hd + d], hd);
            wmma::load_matrix_sync(fb[0], &Ks[wk * 32 * hd + d], hd);
            wmma::load_matrix_sync(fb[1], &Ks[(wk * 32 + 16) * hd + d], hd);
            wmma::mma_sync(fc[0], fa, fb[0], fc[0]);
            wmma::mma_sync(fc[1], fa, fb[1], fc[1]);
        }
        wmma::store_matrix_sync(&Sf[wm*16*FA_TKV + wk*32],      fc[0], FA_TKV, wmma::mem_row_major);
        wmma::store_matrix_sync(&Sf[wm*16*FA_TKV + wk*32 + 16], fc[1], FA_TKV, wmma::mem_row_major);
```

**The online softmax in full** — including the verbatim race comment (this chapter's lesson carrier):

```cuda
        // online softmax per row (thread = row): probs land in Pf (f16)
        if (tid < FA_TQ) {
            int r = tid;
            int qpos = (tq0 + r < nt) ? positions[tq0 + r] : -1;
            float m_old = msh[r], m_new = m_old;
            for (int kk = 0; kk < FA_TKV; kk++) {
                int kv_g = kt + kk;
                if (kv_g <= qpos && kv_g < kv_end) {
                    float s = Sf[r * FA_TKV + kk];
                    if (s > m_new) m_new = s;
                }
            }
            float a = 1.0f;
            // Pf rows use a 256B stride: row r's probs overlap ONLY Sf row r's
            // first half, which this same thread has already read (each read
            // precedes its clobbering write in program order). A 128B stride
            // would race: probs for row r land on scores of rows 2r/2r+1 that
            // other softmax threads have not read yet.
            if (m_new == -INFINITY) {
                // nothing valid in this tile: keep state, zero probs
                for (int kk = 0; kk < FA_TKV; kk++) Pf[r * FA_PSTR + kk] = __float2half(0.0f);
            } else {
                a = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                float sum = 0.0f;
                for (int kk = 0; kk < FA_TKV; kk++) {
                    int kv_g = kt + kk;
                    float p = 0.0f;
                    if (kv_g <= qpos && kv_g < kv_end)
                        p = __expf(Sf[r * FA_TKV + kk] - m_new);
                    Pf[r * FA_PSTR + kk] = __float2half(p);
                    sum += p;
                }
                lsh[r] = lsh[r] * a + sum;
            }
            alpha[r] = a;  msh[r] = m_new;
        }
        __syncthreads();
```

**Rescale + P·V + epilogue**:

```cuda
        // rescale the accumulator by alpha, then add P · V
        float ar = alpha[arow];
#pragma unroll
        for (int dd = 0; dd < FA_HQ; dd++) acc[dd] *= ar;
        for (int kk = 0; kk < FA_TKV; kk++) {
            float p = __half2float(Pf[arow * FA_PSTR + kk]);
            if (p != 0.0f) {
                const __half* vrow = &Vs[kk * hd + aquad * FA_HQ];
#pragma unroll
                for (int dd = 0; dd < FA_HQ; dd++) acc[dd] += p * __half2float(vrow[dd]);
            }
        }
        __syncthreads();
    }

    // write out: acc / l — rows with l == 0 stay 0 (fully masked)
    if (tq0 + arow < nt) {
        float inv = (lsh[arow] > 0.0f) ? 1.0f / lsh[arow] : 0.0f;
        float* orow = &o[(size_t)(tq0 + arow) * ne_q + h * hd + aquad * FA_HQ];
#pragma unroll
        for (int dd = 0; dd < FA_HQ; dd++) orow[dd] = acc[dd] * inv;
    }
```

### 3.3 Pitfalls

- **The score-clobber race (this step's #1 pitfall).** After Pf aliases Sf, P's writes run concurrently with reads of not-yet-consumed S. Under the naive 128 B stride, row r's probabilities land on the first half of `Sf` rows 2r/2r+1 — rows belonging to **other** softmax threads whose reads have not happened yet in program order: a race. The 256 B stride (`FA_PSTR = FA_TKV*2`) confines row r's probabilities to the first half of row r's own scores, which the same thread has already finished reading — **every read precedes its clobbering write in the same thread's program order**, so no race exists. This bug was not found by the graph parity test: the end-to-end logits comparison can pass under lucky scheduling; it was the **standalone harness** (a test rig that drives the kernel repeatedly, independent of graph execution, and compares outputs) that exposed the cross-thread race class.
- **The numeric path of a fully-masked tile**: when `m_new == -INFINITY` (no valid KV position in the tile) the old state must be kept and probabilities written as 0 — taking the normal branch would let `__expf(-INF − -INF)` = NaN propagate down through l/O.
- **The 97 KB dynamic smem opt-in**: when the attribute set fails, call `cudaGetLastError()` first to clear the error before returning — leaving it set would poison the subsequent stream (the same lesson as 8m's capture poisoning).
- **P's f16 rounding**: S stays f32 throughout and the softmax output converts to f16 — this is where the 5e-3 tolerance gate comes from (P·V reads f16 back), and it is also the numerical contract that had to be preserved when P5·0 later moved P·V onto tensor cores.

## 4. Verification

- **The `fa_prefill_f16kv` parity test** (`src/graph/cuda_backend.rs:4306`): seeded pseudo-random q/k/v, run through the **real graph nodes** (`kvcache_store` lands the KV, then the attention node executes), reference = `cpu_gqa_attn` computed on the same f16-rounded K/V, gate `assert_close(..., 5e-3)`. Defends against: kernel numeric errors, mask errors, GQA slicing errors.
- **The standalone harness**: a driver independent of graph execution that runs the kernel repeatedly and compares outputs. Defends against: scheduling-dependent cross-thread races — the class the graph parity test cannot catch (this step's race is exactly what it caught).
- **E2E greedy equality + whole-prefill timing**: defends against assembly errors and confirms the wall-clock gain.
- Masked / n_past-grown positions: the parity test's positions sequence covers non-zero starting points. Defends against: KV position handling errors (the data-fied positions of graph rule §1).

## 5. Results

- **Kernel level**: 176 → **8.5 ms/layer** (7B @2K, 20×); K traffic ~132 GB → **~0.8 GB/layer**.
- **Wall-clock level**: attention's share of the wall fell from 76% to single digits. Cross-check: at 8m's landing, 294 tok/s ⇒ whole wall 6.97 s, of which attention 4.93 s; after the fix the wall ≈ 2.04 + 0.24 ≈ 2.3 s ⇒ an estimate of ~900 tok/s. The 8m② window's measured baseline was 1082, and after cp.async **1204 tok/s** (against llama-bench 3401 @2K, ~2.8×) — the gap between estimate and measurement belongs to machine state and small same-window fixes; the orders of magnitude agree.
- Later evolution (each has its own chapter; not expanded here): P5·0 moved P·V onto tensor cores (10.06 → 4.24 ms/layer); P5·3 added padding to kill ldmatrix bank conflicts; r48 (FAP2) moved softmax wholesale into registers (5.16 → 2.12 ms/layer); r50/r57 nailed down the boundaries for changing FA tile sizes. The skeleton 8n built — the 64-token q tile, the online-softmax state machine, GQA kv slicing, register O — all continues in today's FA kernel.

## 6. Lessons

1. **Prefill attention's root disease is the same as decode's: per-(token,head) K/V re-reads** — tiling the q dimension is the only correct cure, and the gain comes from the byte ledger (÷64 reuse + L2), not from smarter FLOPs.
2. **Every byte saved by aliasing needs a program-order argument attached**: the P/S alias's correctness depends on the stride making each "clobber" land only on data the same thread has already read; 128 B's "save half" is a race.
3. **The standalone harness and graph parity are complementary gates**: a scheduling-dependent race can hide behind lucky scheduling in end-to-end comparisons; only controlled repeated runs expose it reliably.
4. **Moving the O accumulator into registers was a two-step walk**: 8n first made O register-resident (the alpha rescale with zero smem round trips), while softmax itself only entered registers at FAP2 — change one data path at a time, each step measurable.

---
← [02](./02-wmma-f16-prefill-gemm-8m.md) · [Index](./README.md) · [04 →](./04-decode-start-stall-8o.md)
