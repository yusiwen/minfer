# 02 · 8m/8m② — Tiled wmma f16 prefill GEMM (LANDED)

> **Result**: 7B q4_k_m @2K prefill **30.7 → 294** (8m) **→ 1204 tok/s** (8m②), **39×** for the whole row; GEMM kernel **31 → 35 TFLOPS**. Against llama-bench's 3401 @2K, the gap converged from ~110× to ~2.8×.
> **Commit**: `ba3f317` (8m: tiled wmma f16 GEMM), `cdc6599` (8m②: cp.async tile staging). **Date**: 2026-08-30 (both commits the same day, 50 minutes apart).

## 1. Background — where things stood

After Phase 7 raised the CUDA backend's skeleton (the previous chapter), the engine had its first working path, but 7B @2K prefill was only 30.7 tok/s. The bottleneck diagnosis was already clear at that point: **prefill was running kernels shaped for decode**. The decode kernels (the per-op matmuls that predate MMVQ) have a grid shape of `grid.y = nt` — one row of blocks per token, and every block re-reads the whole weight matrix (or its slice of output rows) from VRAM. At nt==1 that is a fair price; at nt=2048 the weight stream is re-read 2048 times.

Arithmetic shows how severe the mismatch is: 7B q4_k_m weights ~4.4 GB, GB10 DRAM roofline 273 GB/s — ideally one pass of the weights over DRAM takes ~16 ms. But 30.7 tok/s means the 2048-token prefill runs 66.7 s — the extra tens of seconds are almost entirely "the same bytes re-entering L2/SM over and over".

Two roads were open for the fix at the time:

1. **Keep the quantized format** and build an int8 tensor-core GEMM (the llama.cpp MMQ route) — numerical correctness demands a q8 activation pipeline and per-type dedicated kernels: a large engineering effort;
2. **Dequantize the weights to f16 first** and let one standard f16 tensor-core GEMM swallow the entire prefill — dequant logic is isolated per type into separate small kernels, the GEMM body is a single one, and the wmma API is directly usable.

8m chose route 2. The reason is the leverage structure: route 2 covers all 8 quantization types with one GEMM, breaks through the wall first, and establishes the f16 baseline; route 1 (R1's int8 MMQ) was later built as its own effort and became the new default at r60 — but that is a later story, and its existence in no way negates the f16 baseline's value: `MINFER_MMQ=0` remains the escape hatch to this day.

## 2. Principle — the GPU mechanism

### 2.1 wmma 16×16×16 fragments

NVIDIA's warp-level matrix API (`nvcuda::wmma`) wraps one 16×16×16 multiply-accumulate as a warp-cooperative operation. Three objects:

- `fragment<matrix_a, 16,16,16, __half, row_major>`: a 16×16 sub-block of A, held in pieces by the warp's 32 lanes (which lane holds which element is **opaque** — the root of a pitfall hit later);
- `fragment<matrix_b, 16,16,16, __half, col_major/row_major>`: a 16×16 sub-block of B; the layout flag decides the addressing interpretation when loading from shared memory;
- `fragment<accumulator, 16,16,16, float>`: the 16×16 result block of C, accumulated in f32.

`load_matrix_sync` loads a fragment from shared memory (with a leading-dimension stride), `mma_sync(acc, a, b, acc)` performs the multiply-accumulate (underneath are HMMA tensor-core instructions: f16 inputs, f32 accumulation), and `store_matrix_sync` writes back. f16 inputs give each tensor-core instruction several times the throughput of the contemporary f32 path, while f32 accumulation preserves numerical accuracy — that is why "dequant to f16" can harvest the hardware dividend.

### 2.2 Tile geometry: why 64×64

The 8m② baseline shape: output tile **TN=64 (nt direction) × TM=64 (od direction)**, k-step width KS=32, 256 threads = 8 warps. Each warp owns one output sub-block of 32 rows (nt) × TM/4 columns (od), i.e. 2×(TM/64) pairs of 16×16 accumulator fragments.

Shared memory budget (KS=32, TM=64): A panel 2×64×32 halves (double buffer) = 8 KB, B panel 2×64×32 halves = 8 KB, C scratch 8 warps × 256 f32 = 8 KB, **24 KB total** — inside the 48 KB static limit, so multiple blocks can be resident per SM. Two opposite directions were later nailed down by measurement (the launcher comments keep the record): KS=64 grows dynamic smem to 56 KB and halves resident blocks, **−38%** (depth traded for inverted occupancy); TM=256 is wider but hits the same wall, **−3%**.

The weight-traffic arithmetic is the whole point of this step. The grid is arranged as `blockIdx.x = nt tile, blockIdx.y = od tile`, with **blocks consecutive in blockIdx.x sharing the same od-tile's B panel** (64 rows × id columns f16, ~0.5 MB f16 for 7B's ffn_gu): this panel streams from DRAM into L2 once, and all nt/64 nt-tile blocks then hit it from L2. Compared with the decode-shaped kernels' "re-read all weights per token", weight DRAM traffic drops from nt× to **~1×**.

The overall FLOP ledger: one 7B @2K prefill's matmul total ≈ 2 × 6.9e9 × 2048 ≈ 2.8×10¹³ FLOP. At 31 TFLOPS (8m) the GEMM takes ~0.9 s — the difference against the 2048/294 ≈ 7.0 s wall at 294 tok/s is made up of attention (still 176 ms/layer then, the next chapter's protagonist), the dequant pass, and elementwise work.

### 2.3 The cost and payoff of dequant-to-f16-then-GEMM

Payoff: one GEMM covers 8 types; all type differences are isolated into per-type dequant kernels (each a plain "read block → compute f16 → write row" loop); tensor cores at full strength.

The cost: one matmul call passes the data three times — quantized weights read in (~4.4 GB for the whole 7B model), f16 written out (~13.6 GB), and the GEMM reads the f16 back (~13.6 GB). In the 8m shape the dequant **re-runs on every call** (the scratch buffer `buf_f16_w` is rewritten per call) — 8p's record prices this pass at **288 ms/call** (7B). This directly spawned 8p's two successors: a persistent f16 cache dequantized once at load time (gated at ≥2 GB, `W16_ENABLE_BYTES`), and the more memory-frugal dequant-in-GEMM fused kernel (`MINFER_FUSED_B=1`). In the 8m era this cost was knowingly paid: it is still far smaller than the 2048× weight re-read it replaces.

### 2.4 8m②'s cp.async

In the synchronous-staging shape, every warp must wait for the global→shared round trip to complete before it can start computing at every 32-k step (31 TFLOPS stalls right there). `cp.async.cg.shared.global` (sm_80+) turns 16 B chunk copies into async operations: the main loop issues the fetch for tile k+32 while computing tile k, and `commit_group`/`wait_group 1` maintains the double-buffer rhythm of "one group in flight, one group ready" → **35 TFLOPS**.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **One dequant kernel per type + one shared GEMM** (dispatch on type_id 0–7) rather than 8 copies of the GEMM. The GEMM only ever sees `__half* B` and never knows the source format.
- **Gate: `nt >= 16` takes this path, and `id % 32 == 0`** — the latter is both a requirement of the block math (all GGUF types use 32-element base blocks) and guarantees the 16 B alignment of the uint4 tile loads; real models' ids (3584/5120/13824…) satisfy it naturally.
- **The 64×64×32 tile** (see §2.2's arithmetic and the two REVERTED counterexamples).
- **C lands through shared memory**: accumulator fragments `store_matrix_sync` into the per-warp `Cs`, then a lane loop writes global with the nt/od tail masks — the tail masking lives in exactly one place.
- **The dynamic smem opt-in mechanism**: TM=128/KS=64 instances need 56 KB, over the 48 KB static limit, so a `cudaFuncSetAttribute` path is mandatory (this became one of §3.3's pitfalls).

### 3.2 Key code

**Type isolation on the dequant side.** A simple-type sample (`src/cuda_kernels.cu`, with the 8m section-header comment excerpted alongside):

```cuda
// ─── 8m: prefill dequant-to-f16 + wmma HGEMM ────────────────────────────
// Prefill (nt >= 16) routes quantized matmuls through ONE tiled
// tensor-core GEMM instead of the decode-shaped kernels whose
// grid.y = nt re-streamed the whole weight matrix once per token
// (7B q4_k_m @2K: 30.7 tok/s vs llama.cpp MMQ 3401). Weights are
// dequantized to f16 once per call into a scratch buffer, activations
// converted to f16, then C[nt, od] = A[nt, id] · B[od, id]^T via
// 16x16x16 wmma with f32 accumulation. Gated on id % 32 == 0.

__global__ void dequant_q4_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 18;          // Q4_0 block = 2B scale + 16B nibbles
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const uint8_t* q = blk + 2;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    // minfer Q4_0 stores round(v/d) + 8 (same -8 offset as the matmuls).
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i]      = __float2half(d * (float(q[i] & 0x0F) - 8.0f));
        o[i + 16] = __float2half(d * (float(q[i] >> 4) - 8.0f));
    }
}
```

All K-quant complexity hides inside the dequant — a q6_K 16-element sub-block sample (the `block_stride` parameter handles both the 7e② 224B padded and the original 210B registration layouts, something the GEMM never needs to know):

```cuda
__global__ void dequant_q6_k_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out,
    int od, int id, int block_stride
) {
    int nsub = id / 16; // 16-element units, 16 per 256 super-block
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nsub) return;
    int row = (int)(g / nsub), s = (int)(g % nsub);
    int sp = s / 16, sub = s % 16;
    const uint8_t* blk = w + ((long long)row * (id / 256) + sp) * block_stride;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    const uint8_t* ql = blk;              // 128B low nibbles
    const uint8_t* qh = blk + 128;        // 64B high 2 bits
    const int8_t* sc = (const int8_t*)(blk + 192);  // 16 per-block scales
    // …ql/qh/sc interleaved addressing (n/tt/gq decomposition); o written at the in-row sp*256 + … offset
    float dsc = d * float(sc[sc_idx]);
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int nib = (tt < 2) ? (ql[ql_off + r] & 0x0F) : (ql[ql_off + r] >> 4);
        int q2 = (qh[qh_off + r] >> (tt * 2)) & 3;
        o[r] = __float2half(dsc * float((nib | (q2 << 4)) - 32));
    }
}
```

**The GEMM body's skeleton** (`gemm_f16_nt_kernel_t`, current tree; at writing time the tree already contains the P5 series' TM=128 default and the AF32 variant — the 8m② baseline is TM=64/KS=32):

```cuda
template <int TM, int KS, bool AF32 = false>
__global__ void gemm_f16_nt_kernel_t(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int nt, int od, int id
) {
    using namespace nvcuda;
    constexpr int TN = 64;
    constexpr int ODC = TM / 64;  // od 16-col fragments per warp row-half
    // …dynamic smem layout: As(2×TN×KS) + Bs(2×TM×KS) + Cs(NW×256 f32)
    const int NW = blockDim.x >> 5;   // warps: 8 for TM<=128, 16 for TM=256
    int warp = tid >> 5;
    int wm = warp >> 1;               // od chunk of this warp
    int wn = warp & 1;                // nt sub-tile: 2 x 32 rows
    // blockIdx.x = nt tile, blockIdx.y = od tile: consecutive blocks share
    // the same od-tile's B panel (64 rows x id f16, ~0.5MB) in L2, so the
    // f16 weight matrix streams from DRAM ~once instead of nt/64 times.
    int m0 = blockIdx.y * TM;
    int n0 = blockIdx.x * TN;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][ODC];
    // …fill_fragment(fc, 0)
```

**The wmma inner loop** — note the v1 bug lesson carved in place in the comments:

```cuda
        // fa[n-half][k-half]; fb[k-half] per od chunk. Both k halves of each
        // 32-slice must accumulate (the v1 bug: only the first 16 k's were
        // multiplied); fb's k offset is +16 ELEMENTS (one k-half), not +16
        // rows.
#pragma unroll
        for (int kh = 0; kh < KHC; kh++) {
            wmma::load_matrix_sync(fa[0], &As[buf*TN*KS + wn*32*KS + kh*32], KS);
            wmma::load_matrix_sync(fa[1], &As[buf*TN*KS + (wn*32+16)*KS + kh*32], KS);
            wmma::load_matrix_sync(fa[2], &As[buf*TN*KS + wn*32*KS + kh*32 + 16], KS);
            wmma::load_matrix_sync(fa[3], &As[buf*TN*KS + (wn*32+16)*KS + kh*32 + 16], KS);
#pragma unroll
            for (int oc = 0; oc < ODC; oc++) {
                wmma::load_matrix_sync(fb[0], &Bs[buf*TM*KS + (ob+oc*16)*KS + kh*32], KS);
                wmma::load_matrix_sync(fb[1], &Bs[buf*TM*KS + (ob+oc*16)*KS + kh*32 + 16], KS);
                wmma::mma_sync(fc[0][oc], fa[0], fb[0], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[1], fb[0], fc[1][oc]);
                wmma::mma_sync(fc[0][oc], fa[2], fb[1], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[3], fb[1], fc[1][oc]);
            }
        }
        __syncthreads();
```

Structure: the warp sub-block of 32 rows × TM columns decomposes into 2 n-halves (16 rows each) × 2 k-halves × ODC od chunks; `fa[0..4]` is loaded once and reused by all od chunks — A-fragment reuse is the first dividend the tile shape pays.

**8m②'s cp.async double-buffer staging**:

```cuda
// 8m②: cp.async global→shared staging (sm_80+). The synchronous load
// stalled every warp on the L2 round trip each 32-k step (~31 TFLOPS
// measured); async copies overlap the k+32 tile fetch with the k compute.
__device__ __forceinline__ void gemm_cp16(__half* smem_dst, const __half* gsrc, bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 16 : 0; // src-size 0 => zero-fill the 16B chunk
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}
__device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void gemm_cp_wait1()  { asm volatile("cp.async.wait_group 1;\n"); }

// P4: stage the A (TN rows) and B (TM rows) k-tiles [k0, k0+KS) into the
// double-buffered dynamic-smem regions. …
template <int TM, int KS, int TN, bool AF32 = false>
__device__ __forceinline__ void gemm_stage_ab(/* … */) {
    for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
        int r = (c * 8) / KS, d = (c * 8) % KS;
        gemm_cp16(As + bbuf*TN*KS + r*KS + d, A + (long long)n*id + k0 + d,
                  n < nt && k0 + d < id);      // zero-fill out-of-bounds chunks (src-size 0)
    }
    for (int c = tid; c < TM * KS / 8; c += blockDim.x) { /* same for the B panel */ }
}
```

The rhythm in the main loop (issue async for the next tile, wait ready for the current tile, one `__syncthreads` aligns the whole block):

```cuda
    for (int k = 0; k < id; k += KS, buf ^= 1) {
        if (k + KS < id) {
            gemm_stage_ab<TM, KS, TN>(A, B, As, Bs, Am, buf ^ 1, n0, m0, k + KS, …);
            gemm_cp_commit();
            gemm_cp_wait1();   // wait until the CURRENT tile landed (one group in flight)
        } else {
            gemm_cp_wait0();
        }
        __syncthreads();
        // …the wmma inner loop consumes buf…
```

Out-of-bounds handling hides in `gemm_cp16`'s `full` parameter: a cp.async with `src-size 0` is a hardware zero-fill — no branches writing zeros needed for any nt/od/k tail.

**The launcher and the timeline of tile parameters** (`launch_gemm_f16`; the comments double as tombstones for three later-REVERTED directions):

```cuda
void launch_gemm_f16(const __half* a, const __half* b, float* c,
                     int nt, int od, int id, cudaStream_t stream, bool af32) {
    // P2: od-tile width (128 default = halved B re-reads; MINFER_GEMM_TM=64
    // reverts to the 8m② baseline for A/B).
    static int tm = -1;
    if (tm < 0) { /* MINFER_GEMM_TM: 64/128/256, default 128 */ }
    // KS = staged k-width per tile. KS=64 halves the barriers per FLOP but
    // measured -38% (56KB dynamic smem halves resident blocks on GB10);
    // KS=32 (8m2 baseline) stays the default. MINFER_GEMM_K64=1 re-tries 64.
    static int ks = -1;
    if (ks < 0) { ks = getenv("MINFER_GEMM_K64") ? 64 : 32; }
    const size_t dyn_smem = (size_t)(2*64*ks + 2*tm*ks) * 2 + 8 * 256 * 4;
    // …GEMM_LAUNCH: grid((nt+63)/64, (od+TM_-1)/TM_), 256 threads, dyn_smem
```

**The Rust-side routing** (the non-fused branch of `cuda.rs`'s `prefill_gemm_f16_inner`; the current tree already contains 8p's persistent cache `w16_get` — in the 8m era every call went straight through `get_or_grow(&self.buf_f16_w)` + `launch_dequant_f16`):

```rust
        let w16 = match self.w16_get(wptr, type_id, od, id, block_stride) {
            Some(p) => p, // persistent copy, dequant already done (8p)
            None => {
                let w16 = Self::get_or_grow(&self.buf_f16_w, od * id * 2);
                unsafe {
                    launch_dequant_f16(type_id, wptr as *const u8, w16, …, stream);
                }
                w16
            }
        };
        // …then launch_gemm_f16(x16, w16, out, nt, od, id, stream, false)
```

### 3.3 Pitfalls

- **The wmma fragment addressing unit is elements, not rows.** The v1 kernel multiplied only the first 16-k half of each 32-k slice; `fb`'s second k-half offset was written as +16 rows instead of +16 elements. The fragment's interpretation of layout is completely opaque, and this class of error shows up numerically as "results systematically too small / misaligned" — the parity test catches it immediately.
- **Setting the >48 KB dynamic smem attribute silently fails during graph capture** and poisons the first captured launch. The fix: `gemm_prefill_smem_init()` calls `cudaFuncSetAttribute` **eagerly** on all pre-registered GEMM instances at stream creation, never leaving it to a running capture window.
- **Q5_0's 22-byte blocks are naturally 2-byte aligned**: a `u32 load at blk+2` on an even block is `cudaErrorMisalignedAddress` (716). The mine was planted in 8m and only detonated in 8p's bitparity test; today's code assembles qh from two u16s, with the comment written right in the dequant kernel.
- **Templates cannot have C linkage**: the `extern "C"` block must open and close around the templated GEMM (paired comment markers exist in the `.cu`), otherwise nvcc name-mangling conflicts arise.
- **cp.async chunk alignment**: `id % 8 == 0` guarantees a 16 B chunk never straddles a boundary (naturally covered by the outer id % 32 gate).
- The three REVERTED boundaries that later nailed down the tile shape (KS=64 −38%, TM=256 −3%, in-kernel f32→f16 A conversion −8%) all remain in the launcher as comments — every axis of the tile was tried at a more "aggressive" value, and 64×64×32 is this kernel's local optimum on GB10.

## 4. Verification

- **`cuda_prefill_f16_gemm_parity`** (Rust-side test, current tree `src/graph/cuda_backend.rs:4457`): for each quantization type it generates **random but valid** block bytes (small d/dmin so the f16 scratch never overflows), and the reference is computed on the Rust side as an f32 matmul over the dequant of **the exact same bytes** — it tests kernel-vs-reference parity, independent of quantization quality. Tail shapes od=70, nt=33 (id fixed at 256, since real tensors' id is always %32==0), all 8 types covered one by one, ending with a real 7B Q4_K shape check (skipped when no dump is available, keeping the suite hermetic). Defends against dequant addressing errors and fragment assembly errors.
- **E2E greedy equality**: logits comparison of CUDA graph output vs CPU graph output + greedy text identical token by token (the standard gate established in 7b/7c). Defends against "each kernel is right, the assembly is wrong".
- **The bitparity test** (introduced in the 8p era, feeding back into this step): the dequantized f16 is compared bit-for-bit against the CPU reference — it detonated §3.3's Q5_0 alignment mine. Defends against silent errors that happen to stay inside tolerance.
- Per-kernel-change standalone nvcc A/B compile verification has been the convention since 7e②; both of this step's commits followed it.

## 5. Results

- **8m (`ba3f317`)**: 7B @2K prefill **30.7 → 294 tok/s** (9.6×) — the full dividend of replacing per-token weight re-reads with one tiled wmma GEMM.
- **8m② (`cdc6599`)**: cp.async double buffering took the GEMM kernel from **31 → 35 TFLOPS**; the whole-prefill A/B inside that commit's window was **1082 → 1204 tok/s** (the climb from 294 to 1082 came mostly from the same-day 8n attention fix — see the next chapter — and the decode start fix 8o; the row number is the window value after 8m② landed).
- **The whole row (master table row 2)**: 30.7 → 294 → 1204 tok/s, **39×**; vs llama-bench 3401 @2K ≈ **2.8×**.
- Later: 8p's load-time f16 weight cache pushed the whole row to ~1400–1500 tok/s; P5·2 (TM=128) +1.5×; R1 MMQ finally became the default at r60 — the f16 GEMM was demoted from workhorse to the `MINFER_MMQ=0` escape hatch, but the entire P5-era GEMM optimization stack (TM=128, KS=32, cp.async staging) was trained on this f16 path.

## 6. Lessons

1. **The first-principles problem of prefill GEMM is "the weight stream crosses DRAM only once"**: tiling plus letting consecutive blocks share one B panel turns nt re-reads into ~1 — this single structural fix was worth 9.6×.
2. **Dequant-to-f16 is the highest-leverage shape for the starting phase**: 8 quantization types isolated into 8 plain dequant kernels, one tensor-core GEMM eating all prefill; the cost (the 288 ms/call dequant pass) was later clawed back step by step via the load-time cache / dequant-in-GEMM.
3. **The wmma fragment addressing unit is elements, not rows** — the `+16 elements` vs `+16 rows` mistake is the fragment API's #1 trap.
4. **Kernels needing >48 KB smem must opt in before capture**; setting the attribute inside a running capture window silently fails and poisons the first captured launch.

---
← [01](./01-phase7-cuda-backend.md) · [Index](./README.md) · [03 →](./03-fa-tiled-prefill-attention-8n.md)
