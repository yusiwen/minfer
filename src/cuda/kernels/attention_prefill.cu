// src/cuda/kernels/attention_prefill.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"

extern "C" {

// ─── 8n: FA-style prefill attention (staged KV) ─────────────────────────
// The legacy gqa_attn_f32_f16kv launches one block per (token, head): K is
// re-read per token per head (7B @2K: ~132 GB per layer) and the hd-wide
// accumulator lives in registers (float4 oc[32] = 128 regs → spills). It
// measured 176 ms per layer (76% of the whole 2K prefill). This kernel
// tiles the q dimension: one block per (64-token q tile, head), K/V tiles
// staged in shared memory, QK^T on tensor cores, online softmax with the
// O accumulator in shared memory. K traffic drops to ~0.8 GB per layer.
//
// The tile is f16 whatever the cache holds: `LAYOUT` picks the staging —
// `KV_LAYOUT_F16` copies halves with 16-byte `cp.async` chunks, and #144 item 3's
// `KV_LAYOUT_Q8_0` dequantizes each packed 32-element block into the same tile
// (one 16-byte smem store per 8 elements). Everything after staging — the
// tensor-core QK^T, the fragment-resident softmax and P·V — is layout-blind.
//
// Shared layout (dynamic, ~35 KB — opt-in via cudaFuncSetAttribute):
//   Qs [64*hd] f16   q tile (scale folded in, f16 for the tensor-core QK^T)
//   Ks [FA_TKV*hd] f16   K tile      Vs [FA_TKV*hd] f16  V tile
// S and P live entirely in wmma accumulator fragments (FAP2 register-resident
// softmax — NO Sf/Pf shared round trip, no m/l/alpha shared arrays). Each warp
// owns a full 16-query-row block x all FA_TKV KV columns, so the online softmax
// (per-row max/sum on the fragments) and the P·V contraction (build the f16
// A-operand from the scaled fragments in place) are both warp-local.
#define FA_TQ 64
#define FA_TKV 32

// P3: async K/V tile staging (16B cp.async chunks; rows beyond kv_end
// zero-filled). Overlapped with the previous tile's QK^T/softmax/P·V via
// double buffering — the synchronous staging version paid the full DRAM
// latency once per KV tile inside the block's serial k-loop.
}

template <bool MAP, int LAYOUT>
__device__ __forceinline__ void fa_stage_kv_async(
    const void* __restrict__ kv_kbase, const void* __restrict__ kv_vbase,
    __half* Ks, __half* Vs, const int* __restrict__ bound, int mt,
    int kt, int kv_end,
    int hk, int hd, int stride_kv, int sstr, int tid, int nthreads,
    size_t row_bytes
) {
    // C8b S4: `p` is a linear window index, which the contiguous modes take as the
    // arena row itself and the map mode resolves through the runs. `mt` is the
    // tile's widest window, so its run list names every index this tile stages
    // (row past `kv_end` are stored zero-length by the size-0 cp.async below, so
    // their address is never read).
    if (LAYOUT == KV_LAYOUT_Q8_0) {
        // #144 item 3: a packed cell cannot be `cp.async`'d — a Q8_0 block's 32
        // quants must be dequantized before they can be a tensor-core operand —
        // so the packed staging is a synchronous load/scale/convert of 8
        // elements (one 16 B smem store) at a time. The rows past `kv_end` are
        // zero-filled through the same store, exactly like the f16 arm's
        // zero-length cp.async. `row_bytes` is the packed cell width
        // (`KvFormat::Q8_0.row_bytes(nkt)`), the same byte stride every other
        // packed kernel addresses a cell with.
        const uint4 z4 = make_uint4(0, 0, 0, 0);
        for (int c = tid; c < FA_TKV * hd / 8; c += nthreads) {
            int r = (c * 8) / hd, d = (c * 8) % hd;
            const int p = kt + r;
            const int row = kv_cell<MAP>(bound, mt, 0, p);
            if (p < kv_end) {
                kv8_q8_0(Ks + r * sstr + d, kv_row(kv_kbase, row, row_bytes), hk * hd + d);
                kv8_q8_0(Vs + r * sstr + d, kv_row(kv_vbase, row, row_bytes), hk * hd + d);
            } else {
                *reinterpret_cast<uint4*>(Ks + r * sstr + d) = z4;
                *reinterpret_cast<uint4*>(Vs + r * sstr + d) = z4;
            }
        }
        return;
    }
    const __half* __restrict__ k = reinterpret_cast<const __half*>(kv_kbase);
    const __half* __restrict__ v = reinterpret_cast<const __half*>(kv_vbase);
#if __CUDA_ARCH__ >= 800
    for (int c = tid; c < FA_TKV * hd / 8; c += nthreads) {
        int r = (c * 8) / hd, d = (c * 8) % hd;
        const int p = kt + r;
        bool full = p < kv_end;
        const int row = kv_cell<MAP>(bound, mt, 0, p);
        unsigned kd = (unsigned)__cvta_generic_to_shared(Ks + r * sstr + d);
        unsigned vd = (unsigned)__cvta_generic_to_shared(Vs + r * sstr + d);
        int sz = full ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(kd),
                     "l"(k + (size_t)row * stride_kv + hk * hd + d), "r"(sz));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(vd),
                     "l"(v + (size_t)row * stride_kv + hk * hd + d), "r"(sz));
    }
#else
    // pre-sm80: synchronous staging (sm_75 stays a build target)
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    for (int i = tid * 8; i < FA_TKV * hd; i += nthreads * 8) {
        int r = i / hd, d = i % hd;
        const int p = kt + r;
        const int row = kv_cell<MAP>(bound, mt, 0, p);
        uint4 kk4, vv4;
        if (p < kv_end) {
            kk4 = *reinterpret_cast<const uint4*>(&k[(size_t)row * stride_kv + hk * hd + d]);
            vv4 = *reinterpret_cast<const uint4*>(&v[(size_t)row * stride_kv + hk * hd + d]);
        } else {
            kk4 = z4;
            vv4 = z4;
        }
        *reinterpret_cast<uint4*>(&Ks[r * sstr + d]) = kk4;
        *reinterpret_cast<uint4*>(&Vs[r * sstr + d]) = vv4;
    }
#endif
}

template <bool CAUSAL, bool MAP, int LAYOUT>
__global__ void fa_prefill_kv(
    const float* __restrict__ q,
    const void* __restrict__ k,
    const void* __restrict__ v,
    float* __restrict__ o,
    const int* __restrict__ bound,
    int nh, int nk, int hd,
    float scale,
    int nt,
    size_t row_bytes
) {
    extern __shared__ __align__(256) uint8_t smem[];
    // Padded smem row stride: hd=128 halves = 256B ≡ 0 mod 32 banks makes
    // every wmma ldmatrix row land on the same bank group (8-way conflict
    // per load). +8 halves (272B) shifts each row by 4 banks.
    const int sstr = hd + 8;
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + FA_TQ * sstr;
    __half* Vs = Ks + FA_TKV * sstr;

    const int tq0 = blockIdx.x * FA_TQ;
    const int h = blockIdx.y;
    const int gqa = nh / nk;
    const int hk = h / gqa;
    const int ne_q = nh * hd;
    const int stride_kv = nk * hd;
    const int tid = threadIdx.x; // 128

    // load q tile (scale folded in) as f16
    for (int i = tid; i < FA_TQ * hd; i += 128) {
        int r = i / hd, d = i % hd;
        int t = tq0 + r;
        float qv = (t < nt) ? q[(size_t)t * ne_q + h * hd + d] * scale : 0.0f;
        Qs[r * sstr + d] = __float2half(qv);
    }
    __syncthreads();

    const int last_t = min(nt - 1, tq0 + FA_TQ - 1);
    // E1b: the staged extent is tile-wide, so the window is taken tile-wide too
    // (min lo, max hi) and the per-row mask trims it exactly. A query tile must
    // therefore not span two sequences — E2's composition keeps a sequence's
    // tokens contiguous, and the CPU path carries no such constraint.
    // CAUSAL is the pre-E1 expression, unchanged.
    int kv_end, win_lo, mt = tq0;
    if (MAP) {
        // C8b S4: a map window's key set is a *prefix* of the sequence's address
        // space, so the tile's staged extent is its largest window (`win_lo` 0) and
        // the per-row limit below is that window's row count. `mt` is the query
        // that owns the widest one: every other query's run list is a prefix of it
        // (a query tile must not span two sequences — E1b's precondition, which the
        // span path already relies on for its tile-wide min/max).
        kv_end = 0;
        win_lo = 0;
        for (int t = tq0; t <= last_t; t++) {
            const int n = attn_map_nkv(bound, t);
            if (n > kv_end) {
                kv_end = n;
                mt = t;
            }
        }
    } else if (CAUSAL) {
        kv_end = bound[last_t] + 1;
        win_lo = 0;
    } else {
        kv_end = 0;
        win_lo = bound[tq0];
        for (int t = tq0; t <= last_t; t++) {
            kv_end = max(kv_end, bound[nt + t]);
            win_lo = min(win_lo, bound[t]);
        }
    }
    const bool tile_full = (tq0 + FA_TQ <= nt); // all 64 O rows in-bounds

    // FAP2 decomposition: 4 warps (128 threads), warp wm owns a full 16-query-row
    // block x all FA_TKV KV columns. S lives in registers (wmma accumulators) and
    // is softmaxed in place; P (the scaled S) is converted f32->f16 into the A
    // fragment of P@V, so the S/P shared round trip and its bank conflicts are
    // gone entirely. m/l/alpha are register-resident (per lane, 2 rows).
    using namespace nvcuda;
    const int warp = tid >> 5; // 0..3
    const int wm = warp;       // 16-query-row block
    const int lane = tid & 31;
    const int l = lane & 3;
    const int r0 = lane >> 2;      // fragment local row 0 (0..7)
    const int r1 = r0 + 8;         // fragment local row 1
    const int row0 = wm * 16 + r0; // block-local query row
    const int row1 = wm * 16 + r1;
    const int c0 = 2 * l;          // fragment col group (2l, 2l+1, 2l+8, 2l+9)
    const int t0 = tq0 + row0, t1 = tq0 + row1;
    // E1b: the per-row limit is *exclusive*. Causal (`bound` is `positions`)
    // keeps the pre-E1 expression `positions[t] + 1`; with an explicit span
    // (`bound` is `[lo, hi)` pairs) the limit is the window's `hi = bound[nt+t]`.
    // Using `bound[t]` there read the window's `lo`, so every row kept only the
    // `lo` column: prefill of any non-zero-start sequence silently attended to a
    // single row (token 0 looked right because its window *is* that row).
    // C8b S4: a map's limit is its window's *row count* — the mask below compares
    // `gcol` against the packed index space the staging walks, not an absolute cell.
    const int qlim0 =
        (t0 < nt) ? (MAP ? attn_map_nkv(bound, t0) : (CAUSAL ? bound[t0] + 1 : bound[nt + t0])) : 0;
    const int qlim1 =
        (t1 < nt) ? (MAP ? attn_map_nkv(bound, t1) : (CAUSAL ? bound[t1] + 1 : bound[nt + t1])) : 0;

    // O accumulator: P@V over hd=128 per 16-row block -> 8 x 16x16 fragments.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[8];
#pragma unroll
    for (int ob = 0; ob < 8; ob++) wmma::fill_fragment(acc[ob], 0.0f);
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    const int kt0 = CAUSAL ? 0 : (win_lo / FA_TKV) * FA_TKV;
    for (int kt = kt0; kt < kv_end; kt += FA_TKV) {
        // stage K/V tile (padded stride, zero-filled beyond kv_end)
        fa_stage_kv_async<MAP, LAYOUT>(k, v, Ks, Vs, bound, mt, kt, kv_end, hk, hd, stride_kv,
                                       sstr, tid, 128, row_bytes);
#if __CUDA_ARCH__ >= 800
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
#endif
        __syncthreads();

        // S = Q · K^T via wmma (reduction over hd), this warp's full row block.
        // fc[0] = kv cols [0,16), fc[1] = [16, FA_TKV) of this tile.
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[FA_TKV / 16];
#pragma unroll
        for (int cc = 0; cc < FA_TKV / 16; cc++) wmma::fill_fragment(fc[cc], 0.0f);
        for (int d = 0; d < hd; d += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[FA_TKV / 16];
#pragma unroll
            for (int cc = 0; cc < FA_TKV / 16; cc++)
                wmma::load_matrix_sync(fb[cc], &Ks[cc * 16 * sstr + d], sstr);
            wmma::load_matrix_sync(fa, &Qs[wm * 16 * sstr + d], sstr);
#pragma unroll
            for (int cc = 0; cc < FA_TKV / 16; cc++)
                wmma::mma_sync(fc[cc], fa, fb[cc], fc[cc]);
        }
        __syncthreads();

        // Fragment-resident online softmax. Each lane holds fragment rows r0/r1
        // at cols c0, c0+1, c0+8, c0+9 in every fc[cc]; the 4 lanes (l=0..3)
        // sharing a row hold all FA_TKV columns. Within-lane reduce over the
        // fragments, then __shfl_xor over offsets 1,2 (the 4-lane group).
        float sm[FA_TKV / 16 * 4], sm1_[FA_TKV / 16 * 4]; // per (cc, quad)
        int gcol[FA_TKV / 16 * 4];
#pragma unroll
        for (int cc = 0; cc < FA_TKV / 16; cc++) {
            int quad = cc * 4;
            sm[quad + 0] = fc[cc].x[0];  sm[quad + 1] = fc[cc].x[1];
            sm[quad + 2] = fc[cc].x[4];  sm[quad + 3] = fc[cc].x[5];
            sm1_[quad + 0] = fc[cc].x[2]; sm1_[quad + 1] = fc[cc].x[3];
            sm1_[quad + 2] = fc[cc].x[6]; sm1_[quad + 3] = fc[cc].x[7];
            gcol[quad + 0] = kt + cc * 16 + c0;     gcol[quad + 1] = kt + cc * 16 + c0 + 1;
            gcol[quad + 2] = kt + cc * 16 + c0 + 8; gcol[quad + 3] = kt + cc * 16 + c0 + 9;
        }
        float mnew0 = -INFINITY, mnew1 = -INFINITY;
#pragma unroll
        for (int q = 0; q < FA_TKV / 16 * 4; q++) {
            // valid = causal (kv <= query pos) AND within the stored KV range
            // (rows >= kv_end are zero-staged and must NOT contribute).
            bool v0 = (gcol[q] < qlim0) && (gcol[q] < kv_end) && (CAUSAL || gcol[q] >= win_lo);
            bool v1 = (gcol[q] < qlim1) && (gcol[q] < kv_end) && (CAUSAL || gcol[q] >= win_lo);
            if (v0) mnew0 = fmaxf(mnew0, sm[q]);
            if (v1) mnew1 = fmaxf(mnew1, sm1_[q]);
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            mnew0 = fmaxf(mnew0, __shfl_xor_sync(0xffffffffu, mnew0, off));
            mnew1 = fmaxf(mnew1, __shfl_xor_sync(0xffffffffu, mnew1, off));
        }
        const int fresh0 = (m0 == -INFINITY);
        const int fresh1 = (m1 == -INFINITY);
        float a0 = fresh0 ? 0.0f : __expf(m0 - mnew0);
        float a1 = fresh1 ? 0.0f : __expf(m1 - mnew1);
        if (mnew0 == -INFINITY) a0 = 1.0f;
        if (mnew1 == -INFINITY) a1 = 1.0f;
        float p0[FA_TKV / 16 * 4], p1[FA_TKV / 16 * 4];
        float sum0 = 0.0f, sum1 = 0.0f;
#pragma unroll
        for (int q = 0; q < FA_TKV / 16 * 4; q++) {
            p0[q] = ((gcol[q] < qlim0) && (gcol[q] < kv_end) && (CAUSAL || gcol[q] >= win_lo))
                        ? __expf(sm[q] - mnew0) : 0.0f;
            p1[q] = ((gcol[q] < qlim1) && (gcol[q] < kv_end) && (CAUSAL || gcol[q] >= win_lo))
                        ? __expf(sm1_[q] - mnew1) : 0.0f;
            sum0 += p0[q]; sum1 += p1[q];
        }
#pragma unroll
        for (int off = 1; off <= 2; off <<= 1) {
            sum0 += __shfl_xor_sync(0xffffffffu, sum0, off);
            sum1 += __shfl_xor_sync(0xffffffffu, sum1, off);
        }
        if (mnew0 != -INFINITY) m0 = mnew0;
        if (mnew1 != -INFINITY) m1 = mnew1;
        l0 = l0 * a0 + sum0; l1 = l1 * a1 + sum1;
        const float aa0 = (mnew0 == -INFINITY) ? 1.0f : a0;
        const float aa1 = (mnew1 == -INFINITY) ? 1.0f : a1;

        // rescale O fragments by the per-row alpha (x[0,1,4,5] -> row r0,
        // x[2,3,6,7] -> row r1 — the m16n16 f32 accumulator layout).
#pragma unroll
        for (int ob = 0; ob < 8; ob++) {
            acc[ob].x[0] *= aa0; acc[ob].x[1] *= aa0;
            acc[ob].x[2] *= aa1; acc[ob].x[3] *= aa1;
            acc[ob].x[4] *= aa0; acc[ob].x[5] *= aa0;
            acc[ob].x[6] *= aa1; acc[ob].x[7] *= aa1;
        }

        // Build the P@V A-operand (f16 matrix_a) from the scaled fragments IN
        // PLACE. matrix_a m16n16k16 row_major and the f32 accumulator use the
        // SAME (row,col) layout, so pa.x[i] == fc[cc].x[i] element-wise. Ks in
        // QK^T is col_major; V in P@V is row_major (both validated standalone).
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa[FA_TKV / 16];
#pragma unroll
        for (int cc = 0; cc < FA_TKV / 16; cc++) {
            int quad = cc * 4;
            pa[cc].x[0] = __float2half(p0[quad + 0]);
            pa[cc].x[1] = __float2half(p0[quad + 1]);
            pa[cc].x[2] = __float2half(p1[quad + 0]);
            pa[cc].x[3] = __float2half(p1[quad + 1]);
            pa[cc].x[4] = __float2half(p0[quad + 2]);
            pa[cc].x[5] = __float2half(p0[quad + 3]);
            pa[cc].x[6] = __float2half(p1[quad + 2]);
            pa[cc].x[7] = __float2half(p1[quad + 3]);
        }
        // acc = acc*alpha + P · V. V (B) is row_major from Vs.
#pragma unroll
        for (int kk0 = 0; kk0 < FA_TKV; kk0 += 16) {
#pragma unroll
            for (int ob = 0; ob < 8; ob++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
                wmma::load_matrix_sync(vb, &Vs[kk0 * sstr + ob * 16], sstr);
                wmma::mma_sync(acc[ob], pa[kk0 / 16], vb, acc[ob]);
            }
        }
        __syncthreads();
    }

    // write out: acc / l — rows with l == 0 stay 0 (fully masked). Full tiles
    // store the fragments straight to global o; the tail tile stages through
    // shared memory so rows t >= nt can be skipped.
    if (tile_full) {
        float i0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
        float i1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
#pragma unroll
        for (int ob = 0; ob < 8; ob++) {
            acc[ob].x[0] *= i0; acc[ob].x[1] *= i0;
            acc[ob].x[2] *= i1; acc[ob].x[3] *= i1;
            acc[ob].x[4] *= i0; acc[ob].x[5] *= i0;
            acc[ob].x[6] *= i1; acc[ob].x[7] *= i1;
            wmma::store_matrix_sync(
                &o[(size_t)(tq0 + wm * 16) * ne_q + h * hd + ob * 16],
                acc[ob], ne_q, wmma::mem_row_major);
        }
    } else {
        // Qs/Ks/Vs regions are free after the KV loop: contiguous staging for
        // the 64x128 f32 O tile (32 KB < the 34.8 KB smem budget).
        float* stage = reinterpret_cast<float*>(smem);
        float i0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
        float i1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
#pragma unroll
        for (int ob = 0; ob < 8; ob++) {
            acc[ob].x[0] *= i0; acc[ob].x[1] *= i0;
            acc[ob].x[2] *= i1; acc[ob].x[3] *= i1;
            acc[ob].x[4] *= i0; acc[ob].x[5] *= i0;
            acc[ob].x[6] *= i1; acc[ob].x[7] *= i1;
            wmma::store_matrix_sync(
                &stage[(wm * 16) * hd + ob * 16],
                acc[ob], hd, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < FA_TQ * hd; idx += 128) {
            int r = idx / hd, c = idx % hd;
            int t = tq0 + r;
            if (t < nt) o[(size_t)t * ne_q + h * hd + c] = stage[idx];
        }
    }
}
extern "C" {

int launch_fa_prefill_kv(
    const float* q, const void* k, const void* v, float* o,
    const int* bound, int mode, int nh, int nk, int hd, float scale, int nt,
    int layout, size_t row_bytes,
    cudaStream_t stream
) {
    // Qs + Ks + Vs only (S/P no longer go through shared memory). sstr = hd+8
    // padding; Ks/Vs are FA_TKV rows (the r46 launcher's 3*FA_TQ bug is gone).
    size_t smem = ((size_t)FA_TQ + 2 * FA_TKV) * (hd + 8) * 2;
    static size_t attr_smem = 0;
    if (smem > attr_smem) {
        // The opt-in is per *function*, so every instantiation the dispatch below
        // can pick needs its own — a single call (the pre-C8b S4 form) left one of
        // the two window modes without it. #144 adds the packed instantiations:
        // the smem size is layout-independent (the packed cell is dequantized
        // into the same f16 tile), so one attribute covers whichever layout runs.
        cudaError_t e = cudaSuccess;
        for (int m = ATTN_WIN_CAUSAL; m <= ATTN_WIN_MAP && e == cudaSuccess; m++) {
            const void* f = m == ATTN_WIN_MAP
                                ? (const void*)&fa_prefill_kv<false, true, KV_LAYOUT_F16>
                            : m == ATTN_WIN_SPAN
                                ? (const void*)&fa_prefill_kv<false, false, KV_LAYOUT_F16>
                                : (const void*)&fa_prefill_kv<true, false, KV_LAYOUT_F16>;
            e = cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
            if (e != cudaSuccess) break;
            const void* p = m == ATTN_WIN_MAP
                                ? (const void*)&fa_prefill_kv<false, true, KV_LAYOUT_Q8_0>
                            : m == ATTN_WIN_SPAN
                                ? (const void*)&fa_prefill_kv<false, false, KV_LAYOUT_Q8_0>
                                : (const void*)&fa_prefill_kv<true, false, KV_LAYOUT_Q8_0>;
            e = cudaFuncSetAttribute(p, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        }
        if (e != cudaSuccess) {
            cudaGetLastError(); // clear the error so it cannot poison the stream
            // NOT silent: the caller falls back to the legacy per-token
            // attention kernel (~50x slower) — this must be visible.
            static int warned = 0;
            if (!warned) {
                warned = 1;
                fprintf(stderr,
                        "minfer/cuda: fa_prefill_kv smem %zu B exceeds the device "
                        "limit; falling back to the legacy attention kernel\n",
                        smem);
            }
            return -1;
        }
        attr_smem = smem;
    }
    dim3 grid((nt + FA_TQ - 1) / FA_TQ, nh, 1);
    // #162: a failed launch here is a DOCUMENTED fallback, not a sticky error —
    // the Rust caller falls back to the legacy per-token attention kernel when
    // this returns non-zero (the `-1` smem arm above is the same contract).
    bool launched;
    if (layout == KV_LAYOUT_Q8_0) {
        if (mode == ATTN_WIN_MAP) {
            minfer_launch_prelude("launch:fa_prefill_kv__q8_0_map",
                                  "fa_prefill_kv<false,true,KV_LAYOUT_Q8_0>");
            fa_prefill_kv<false, true, KV_LAYOUT_Q8_0><<<grid, minfer_launch_block("launch:fa_prefill_kv__q8_0_map", 128), smem, stream>>>(q, k, v, o, bound, nh, nk, hd,
                                                                       scale, nt, row_bytes);
            launched = minfer_launch_ok_opt("launch:fa_prefill_kv__q8_0_map",
                                            "fa_prefill_kv<false,true,KV_LAYOUT_Q8_0>");
        } else if (mode == ATTN_WIN_SPAN) {
            minfer_launch_prelude("launch:fa_prefill_kv__q8_0_span",
                                  "fa_prefill_kv<false,false,KV_LAYOUT_Q8_0>");
            fa_prefill_kv<false, false, KV_LAYOUT_Q8_0><<<grid, minfer_launch_block("launch:fa_prefill_kv__q8_0_span", 128), smem, stream>>>(q, k, v, o, bound, nh, nk, hd,
                                                                        scale, nt, row_bytes);
            launched = minfer_launch_ok_opt("launch:fa_prefill_kv__q8_0_span",
                                            "fa_prefill_kv<false,false,KV_LAYOUT_Q8_0>");
        } else {
            minfer_launch_prelude("launch:fa_prefill_kv__q8_0_causal",
                                  "fa_prefill_kv<true,false,KV_LAYOUT_Q8_0>");
            fa_prefill_kv<true, false, KV_LAYOUT_Q8_0><<<grid, minfer_launch_block("launch:fa_prefill_kv__q8_0_causal", 128), smem, stream>>>(q, k, v, o, bound, nh, nk, hd,
                                                                       scale, nt, row_bytes);
            launched = minfer_launch_ok_opt("launch:fa_prefill_kv__q8_0_causal",
                                            "fa_prefill_kv<true,false,KV_LAYOUT_Q8_0>");
        }
        return launched ? 0 : -1;
    }
    if (mode == ATTN_WIN_MAP) {
        minfer_launch_prelude("launch:fa_prefill_kv__f16_map",
                              "fa_prefill_kv<false,true,KV_LAYOUT_F16>");
        fa_prefill_kv<false, true, KV_LAYOUT_F16><<<grid, minfer_launch_block("launch:fa_prefill_kv__f16_map", 128), smem, stream>>>(q, k, v, o, bound, nh, nk, hd,
                                                                   scale, nt, row_bytes);
        launched = minfer_launch_ok_opt("launch:fa_prefill_kv__f16_map",
                                        "fa_prefill_kv<false,true,KV_LAYOUT_F16>");
    } else if (mode == ATTN_WIN_SPAN) {
        minfer_launch_prelude("launch:fa_prefill_kv__f16_span",
                              "fa_prefill_kv<false,false,KV_LAYOUT_F16>");
        fa_prefill_kv<false, false, KV_LAYOUT_F16><<<grid, minfer_launch_block("launch:fa_prefill_kv__f16_span", 128), smem, stream>>>(q, k, v, o, bound, nh, nk, hd,
                                                                    scale, nt, row_bytes);
        launched = minfer_launch_ok_opt("launch:fa_prefill_kv__f16_span",
                                        "fa_prefill_kv<false,false,KV_LAYOUT_F16>");
    } else {
        minfer_launch_prelude("launch:fa_prefill_kv__f16_causal",
                              "fa_prefill_kv<true,false,KV_LAYOUT_F16>");
        fa_prefill_kv<true, false, KV_LAYOUT_F16><<<grid, minfer_launch_block("launch:fa_prefill_kv__f16_causal", 128), smem, stream>>>(q, k, v, o, bound, nh, nk, hd,
                                                                   scale, nt, row_bytes);
        launched = minfer_launch_ok_opt("launch:fa_prefill_kv__f16_causal",
                                        "fa_prefill_kv<true,false,KV_LAYOUT_F16>");
    }
    return launched ? 0 : -1;
}
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// This file owns the template instantiations, so the address-taking must
// happen here (a cross-TU template reference is nvcc #20280-D and can fail
// to link).
extern "C" void minfer_prewarm_attention_prefill_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, (fa_prefill_kv<true, false, KV_LAYOUT_F16>));
    MINFER_PREWARM_ONE(a, (fa_prefill_kv<false, false, KV_LAYOUT_F16>));
    MINFER_PREWARM_ONE(a, (fa_prefill_kv<false, true, KV_LAYOUT_F16>));
    MINFER_PREWARM_ONE(a, (fa_prefill_kv<true, false, KV_LAYOUT_Q8_0>));
    MINFER_PREWARM_ONE(a, (fa_prefill_kv<false, false, KV_LAYOUT_Q8_0>));
    MINFER_PREWARM_ONE(a, (fa_prefill_kv<false, true, KV_LAYOUT_Q8_0>));
}
