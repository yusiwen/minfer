// CUDA kernel sources for minfer — the shared header.
//
// Every device helper, macro and host declaration that more than one
// translation unit of `src/cuda/kernels/` needs lives here.  Two rules make it
// safe to include from many `.cu` files compiled in one whole-program nvcc
// invocation (no `-rdc`):
//
//   * a device helper is `static __device__ __forceinline__`, so each TU gets
//     its own copy and there is no multiple-definition error (guard probe,
//     issue #263 blueprint §6 fact 1); and
//   * the `minfer_launch_*` / `minfer_smem_optin` family is only *declared*
//     here — `guard.cu` owns the definitions and the sticky #147/#162 state,
//     because a duplicated owner would silently read 0 after a real failing
//     launch (blueprint §6 fact 4).
//
// No `<<<>>>` may live in this header: `scripts/check_cuda_launch_returns.py`
// asserts that, and it is what keeps "every launch site reads its own error"
// complete.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cstdint>
#include <mma.h>

// The #223 pre-warm's one attribute query per kernel instantiation.  `attrs` is
// the caller's `cudaFuncAttributes`; a missing instantiation is not an error —
// it only means that path was never compiled in.
#define MINFER_PREWARM_ONE(attrs, k)                                           \
    do {                                                                       \
        if (cudaFuncGetAttributes(&(attrs),                                    \
                                  reinterpret_cast<const void*>(&(k)))         \
            != cudaSuccess) {                                                  \
            cudaGetLastError();                                                \
        }                                                                      \
    } while (0)

// ─── Block size constants (must match src/block.rs) ───────────
#define Q4B  18   // sizeof(BlockQ4_0): half d + uchar qs[16]
#define Q41B 20   // sizeof(BlockQ4_1): half d + half m + uchar qs[16]
#define Q8B  34   // sizeof(BlockQ8_0): half d + char qs[32]
#define Q4KB 144  // sizeof(BlockQ4_K)
#define Q5KB 176  // sizeof(BlockQ5_K)
#define Q6KB 210  // sizeof(BlockQ6_K)
#define WARP 32

// ─── #162: every `<<<>>>` reads its own launch error ─────────────────────────
// The #147 checked-launch helpers are *defined* in `guard.cu`, the single owner of
// the #147/#162 state, and declared here for every family translation unit.
// They live OUTSIDE any `extern "C" {` block so their C++ linkage matches the
// definitions; a duplicate owner would read an empty sticky after a real failing
// launch, which is exactly the silent failure #162 was filed about.
//
//   minfer_launch_ok      — a REQUIRED launch: names the site and the kernel
//                           instantiation, clears the latch it named, and records
//                           a sticky failure that `CudaBackend::execute_node`
//                           (ONE Rust-side check, not 65 signature changes)
//                           turns into an `Err`, so no consumer ever reads a
//                           stale output.
//   minfer_launch_ok_opt  — a launch on a path with a DOCUMENTED fallback (the
//                           MMQ fast paths, the fa-prefill smem fallback, and the
//                           int-returning launchers whose Rust caller already
//                           makes the decision): names the site, clears the
//                           latch, and does NOT set the sticky.
//   minfer_launch_block — the launch geometry with the #162 injection lever: at
//                         an armed site the block dim becomes deliberately
//                         illegal, so the launch fails for real with a real latch
//                         (`cudaErrorInvalidValue`, probed on sm_121) and the
//                         kernel never runs. Data, not 65 bespoke mechanisms.
void minfer_launch_prelude(const char* site, const char* kernel_name);
bool minfer_launch_ok(const char* site, const char* kernel_name);
bool minfer_launch_ok_opt(const char* site, const char* kernel_name);
size_t minfer_launch_smem(const char* site, size_t smem);
dim3 minfer_launch_block(const char* site, dim3 block);
dim3 minfer_launch_block(const char* site, unsigned block);

// #263: the rest of the one-owner family `guard.cu` defines. `minfer_smem_optin`
// is the single place a `cudaFuncSetAttribute` return value is read (called by
// the MMQ/GEMM launchers' opt-ins), `minfer_optin_limit` is the device's
// queried `cudaDevAttrMaxSharedMemoryPerBlockOptin`, and `minfer_test_call_fails`
// is the `MINFER_TEST_CALL_FAIL` matcher `gemm_smem_optin` shares.
bool minfer_smem_optin(const char* site, const char* kernel_name, const void* fn, int bytes);
int minfer_optin_limit(void);
bool minfer_test_call_fails(const char* site);

// ─── Helper: warp-level sum reduction ─────────────────────────
static __device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ─── Helper: fp16 → f32 (using CUDA intrinsics) ──────────────
static __device__ __forceinline__ float h2f(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ─── Helper: bf16 → f32 (#208) ───────────────────────────────
// bf16 is f32's **top** 16 bits: the decode is a left shift of the raw word
// with the low 16 bits cleared, and it is exact for every value (no rounding,
// no bias, and a NaN payload survives as a NaN). This is the same expression
// the CPU path uses (`crate::block::bf16_to_f32` == `f32::from_bits(bits << 16)`),
// which is what makes the kernel-level exactness gate a bitwise comparison
// rather than a tolerance.
static __device__ __forceinline__ float b2f(uint16_t b) {
    return __uint_as_float(static_cast<unsigned>(b) << 16);
}

// ─── Helper: unpack Q4_K 6-bit scale and min ────────────────
// Q4_K stores 16 × 6-bit values (8 scales + 8 mins) packed into 12 bytes.
// This mirrors Metal's get_scale_min_k4 and Rust block.rs::unpack_q4k_scales.

static __device__ __forceinline__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]   >> 6) << 4);
    }
}

// i.e. +74–77% per matmul; at id < 2048 the win collapses to noise
// (launch-latency bound), so dispatch gates on id >= 2048.
//
// Activation layout: padded 40-byte q8_0 blocks — [f16 d][2B pad][32B int8]
// — so the int8 payload is 4-byte aligned for the uint32/dp4a reads. The
// scratch (buf_q8_decode) is size-stable per graph (id fixed), grown during
// the warmup runs, never inside a capture window.
#define Q8PB 40

// --- D3-5 1a: fused-producer decode A-quantize ------------------------------
// The prefill analogue is rms_norm_quant_f32_t (r51). At decode (nt==1) an
// activation row feeds ONE MMVQ matmul group, and the standalone
// quantize_q8_0_pad40 launch in front of every matmul is pure launch +
// global-round-trip overhead (D3-1: ~265 launches = 0.46 ms/step). Fusing the
// quantize into the PRODUCER (rms_norm / swiglu) removes the launch while
// keeping every MMVQ consumer byte-identical: the epilogue below is the
// standalone kernel's per-32-block body VERBATIM (max is exact for any
// association; the rintf/clamp pass is elementwise), so the pad40 q8 bytes
// are bit-identical by construction. MINFER_NO_DECODE_A_FUSE=1 reverts to the
// standalone pair (A/B gate).
static __device__ __forceinline__ void quantize_pad40_block(
    const float* __restrict__ src, uint8_t* __restrict__ dst
) {
    float4 sv[8];
    #pragma unroll
    for (int v = 0; v < 8; v++)
        sv[v] = *reinterpret_cast<const float4*>(src + 4 * v);
    float am = 0.0f;
    #pragma unroll
    for (int v = 0; v < 8; v++)
        am = fmaxf(am, fmaxf(fmaxf(fabsf(sv[v].x), fabsf(sv[v].y)),
                             fmaxf(fabsf(sv[v].z), fabsf(sv[v].w))));
    float d = am / 127.0f;
    float di = (d != 0.0f) ? 1.0f / d : 0.0f;
    *reinterpret_cast<__half*>(dst) = __float2half(d);
    int s = 0;
    uint32_t packed[8];
    #pragma unroll
    for (int v = 0; v < 8; v++) {
        const float* e = &sv[v].x;
        uint32_t p = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int q = int(rintf(e[j] * di));
            q = max(-128, min(127, q));
            p |= (uint32_t)(uint8_t)(int8_t)q << (8 * j);
            s += q;
        }
        packed[v] = p;
    }
    #pragma unroll
    for (int v = 0; v < 8; v++)
        *reinterpret_cast<uint32_t*>(dst + 4 + 4 * v) = packed[v];
    *reinterpret_cast<uint32_t*>(dst + 36) = uint32_t(s);
}

#define MMQ_A_BLK 64          // tokens per A block == MMQ_NBI
#define MMQ_A_QASZ (MMQ_A_BLK * 32)   // 2048 B: one block's swizzled qs plane
#define MMQ_A_SDASZ (MMQ_A_BLK * 4)   // 256 B: one block's packed d|ssum


// rms rows per block (one warp per row, the rms_norm_f32 mapping).
#define RMSQ_RPB 8


// 8e follow-up: the warp+block reduction shared by the q5_K/q6_K MMVQ
// kernels (same shape as the inline one in q4_k_q8_mmvq).
static __device__ __forceinline__ void mmvq_block_reduce(
    float acc, float* __restrict__ output, int od, int t
) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) v += warp_sums[k];
        output[(size_t)t * od + (size_t)blockIdx.x] = v;
    }
}

// ─── C4 S2b: the KV layout tag and the one load idiom ────────────────────────
//
// A packed Q8_0 cell cannot be addressed as a typed element array, so every KV
// address from here on is formed in BYTES: `kv_row` gives a cell's first byte and
// `kv4<LAYOUT>` loads four elements out of a row. The three tags are a host
// contract — the Rust side stores the same 0/1/2 codes (the `KvFormat`
// discriminants) and every launcher takes the tag as an `int`, so a packed region
// can never be handed to a kernel that would address it as f32 rows.
//
// F32 and F16 are the pre-C4 addresses verbatim (`float4`; two `__half2`), so the
// instruction streams of those two instantiations are unchanged. Q8_0 reads block
// `elem/32`'s f16 scale plus four quants at `2 + elem%32`: a 4-element group never
// straddles a block, because a KV head's base is `hd`-aligned and `hd % 32 == 0`
// (`ensure_kv`'s packed-width check enforces exactly that).
#define KV_LAYOUT_F32 0
#define KV_LAYOUT_F16 1
#define KV_LAYOUT_Q8_0 2

/// Elements per Q8_0 block and bytes per Q8_0 block (`block::BlockQ8_0`: one f16
/// scale followed by 32 int8 quants).
#define Q8_0_BLOCK_ELEMS 32
#define Q8_0_BLOCK_BYTES 34

static __device__ __forceinline__ const char* kv_row(const void* base, long long cell, size_t row_bytes) {
    return reinterpret_cast<const char*>(base) + (size_t)cell * row_bytes;
}

// `WIDE` (issue #202) selects how a four-element Q8_0 group's quants are loaded:
// false is the incumbent four byte loads, true is two 16-bit loads. It has no
// meaning for F32/F16, whose loads are already one / two vector loads, so only
// the `kv4<*, false>` specialisations exist for them.
template <int LAYOUT, bool WIDE = false>
__device__ __forceinline__ float4 kv4(const char* row, int elem);

template <>
__device__ __forceinline__ float4 kv4<KV_LAYOUT_F32, false>(const char* row, int elem) {
    return *reinterpret_cast<const float4*>(row + (size_t)elem * 4);
}

template <>
__device__ __forceinline__ float4 kv4<KV_LAYOUT_F16, false>(const char* row, int elem) {
    const __half* p = reinterpret_cast<const __half*>(row) + elem;
    __half2 a = *reinterpret_cast<const __half2*>(p);
    __half2 b = *reinterpret_cast<const __half2*>(p + 2);
    float2 x = __half22float2(a);
    float2 y = __half22float2(b);
    return make_float4(x.x, x.y, y.x, y.y);
}

// The four quants of one 4-element group, as an `int` in little-endian byte
// order, plus the group's Q8_0 block scale as a float.
//
// The incumbent load: four separate `signed char` accesses. A Q8_0 block is
// `f16 d; i8 qs[32]` (34 B), so `blk + 2 + (elem & 31)` has no 4-byte alignment
// guarantee (`34 * k` alternates parity) and no single 32-bit load can replace
// them. #202 measured this as the L1 request-count cost of the packed cell: at
// hd 64 the packed decode arm issues 1.71x the f16 arm's load sectors while
// reading 0.57x its L2 sectors.
static __device__ __forceinline__ void q8_0_load4_bytes(
    const unsigned char* blk, int off, int& q, float& d) {
    d = __half2float(*reinterpret_cast<const __half*>(blk));
    const signed char* p = reinterpret_cast<const signed char*>(blk + 2) + off;
    q = ((int)(unsigned char)p[0]) | ((int)(unsigned char)p[1] << 8) |
        ((int)(unsigned char)p[2] << 16) | ((int)(unsigned char)p[3] << 24);
}

// #202: the same four bytes in HALF the load instructions. `34 * k + 2 + 4m` is
// even for every block index `k` and every 4-element-aligned offset `4m` (`34k`
// is even, `2 + 4m` is even), so a group is always 2-byte aligned even though it
// is only 4-byte aligned when `k` is odd. Two `unsigned short` loads therefore
// fetch the four bytes with two L1 requests instead of four, and the two halves
// are recombined exactly as the byte loads were. The values are bit-identical to
// `q8_0_load4_bytes`, so this is an access-pattern change, not a numerics change.
static __device__ __forceinline__ void q8_0_load4_wide(
    const unsigned char* blk, int off, int& q, float& d) {
    d = __half2float(*reinterpret_cast<const __half*>(blk));
    const unsigned char* p = blk + 2 + off;
    const unsigned lo = *reinterpret_cast<const unsigned short*>(p);
    const unsigned hi = *reinterpret_cast<const unsigned short*>(p + 2);
    q = (int)(lo | (hi << 16));
}

template <bool WIDE>
__device__ __forceinline__ void q8_0_load4(
    const unsigned char* blk, int off, int& q, float& d) {
    if (WIDE) q8_0_load4_wide(blk, off, q, d);
    else q8_0_load4_bytes(blk, off, q, d);
}

template <bool WIDE>
__device__ __forceinline__ float4 kv4_q8_0_impl(const char* row, int elem) {
    const unsigned char* blk =
        reinterpret_cast<const unsigned char*>(row) + (size_t)(elem >> 5) * Q8_0_BLOCK_BYTES;
    int q;
    float d;
    q8_0_load4<WIDE>(blk, elem & 31, q, d);
    return make_float4(d * (float)(signed char)(q & 0xff),
                       d * (float)(signed char)((q >> 8) & 0xff),
                       d * (float)(signed char)((q >> 16) & 0xff),
                       d * (float)(signed char)((q >> 24) & 0xff));
}

template <>
__device__ __forceinline__ float4 kv4<KV_LAYOUT_Q8_0, false>(const char* row, int elem) {
    return kv4_q8_0_impl<false>(row, elem);
}

template <>
__device__ __forceinline__ float4 kv4<KV_LAYOUT_Q8_0, true>(const char* row, int elem) {
    return kv4_q8_0_impl<true>(row, elem);
}

// #186: the same four quants as `kv4<KV_LAYOUT_Q8_0>` but left in `int8`, packed
// little-endian into one `int` for `__dp4a`, plus the block's f16 scale as a
// float. #202's `WIDE` picks the two-16-bit-load form above.
template <bool WIDE = false>
__device__ __forceinline__ void kv4_q8_0_packed(const char* row, int elem, int& q, float& d) {
    const unsigned char* blk =
        reinterpret_cast<const unsigned char*>(row) + (size_t)(elem >> 5) * Q8_0_BLOCK_BYTES;
    q8_0_load4<WIDE>(blk, elem & 31, q, d);
}

// #144: dequantize EIGHT consecutive elements of one packed Q8_0 KV cell into
// eight halves (one 16-byte tensor-core staging slot). `elem` must be a multiple
// of 8 and every head base is 32-element aligned (`ensure_kv`), so the group
// never straddles a Q8_0 block — the same precondition `kv4<Q8_0>` relies on.
// The dequant is `kv4<Q8_0>`'s, element for element (block scale times the signed
// quant), rounded to half because the FA path's smem tile is f16.
static __device__ __forceinline__ void kv8_q8_0(__half* dst, const char* row, int elem) {
    const unsigned char* blk =
        reinterpret_cast<const unsigned char*>(row) + (size_t)(elem >> 5) * Q8_0_BLOCK_BYTES;
    const float d = __half2float(*reinterpret_cast<const __half*>(blk));
    const signed char* q = reinterpret_cast<const signed char*>(blk + 2) + (elem & 31);
    __half2* h = reinterpret_cast<__half2*>(dst);
    #pragma unroll
    for (int i = 0; i < 4; i++)
        h[i] = __floats2half2_rn(d * (float)q[2 * i], d * (float)q[2 * i + 1]);
}


// helper: convert a half4 (hd is a multiple of 4) to float4
static __device__ __forceinline__ float4 h4_to_f4(const __half* p) {
    float2 a = __half22float2(*reinterpret_cast<const __half2*>(p));
    float2 b = __half22float2(*reinterpret_cast<const __half2*>(p + 2));
    return make_float4(a.x, a.y, b.x, b.y);
}

// ─── E1: the per-query attention window ───────────────────────────────────────
// `CAUSAL` is the pre-E1 behaviour, still what every single-sequence caller uses:
// the bound is `positions[t] + 1` and rows start at 0. The windowed
// instantiation (E1b, reached only when the node is `Attn { explicit_span: true }`)
// reads the `[lo, hi)` pair the KV store's ownership resolved, with `lo` at
// `bound[t]` and `hi` at `bound[nt + t]`.
//
// `CAUSAL` is a template parameter, not a runtime flag, so the causal kernels
// compile to exactly the instructions they did before E1b — no extra index
// arithmetic in the hot loop, which the SASS diff of the two revisions checks.
template <bool CAUSAL>
__device__ __forceinline__ void attn_window(
    const int* __restrict__ bound, int t, int nt, int& row0, int& nkv) {
    if (CAUSAL) {
        row0 = 0;
        nkv = bound[t] + 1;
    } else {
        row0 = bound[t];
        nkv = bound[nt + t] - bound[t];
    }
}

// ─── C8b S4: the `kv_map` window ──────────────────────────────────────────────
// A sequence that reads a prefix in place has a window that is **not** one
// contiguous range: `[0, r)` lives in the donor's cells and `[r, p]` in its own
// run. `kv_map` carries it as a zero-padded list of `(cell, len)` runs, and the
// `MAP` template flag below makes the kernels name their rows through that list.
//
// `MAP` is a template parameter for the same reason `CAUSAL` is (E1b): the
// causal instantiation must compile to exactly the pre-E1 instructions, so no
// runtime branch may enter the row walk. A map window's key set is a **prefix**
// of the sequence's address space — every run before the query's own, plus its
// own row — so every kernel's existing `index < limit` mask stays valid; only
// the row *address* and the limit's value change. The window mode is the input's
// size, so it is a build-time property (C8b S2's departure 2).
#define KV_MAP_MAX_SPANS 4 // mirrors kvcache::KV_MAP_MAX_SPANS
#define ATTN_WIN_CAUSAL 0  // `positions`: rows [0, positions[t] + 1)
#define ATTN_WIN_SPAN 1    // `attn_span`: one [lo, hi) pair per query
#define ATTN_WIN_MAP 2     // `kv_map`: (cell, len) runs per query

// The map window's row count for query `t` (every run before its own, plus its
// own row).
static __device__ __forceinline__ int attn_map_nkv(const int* __restrict__ bound, int t) {
    const int* m = bound + (size_t)t * KV_MAP_MAX_SPANS * 2;
    int n = 0;
    #pragma unroll
    for (int r = 0; r < KV_MAP_MAX_SPANS; r++) n += m[r * 2 + 1];
    return n;
}

// The per-query window: `row0`/`nkv` for the contiguous modes, the row count for
// the map mode (`row0` is then unused, because rows resolve through `kv_cell`).
template <bool CAUSAL, bool MAP>
__device__ __forceinline__ void attn_extent(
    const int* __restrict__ bound, int t, int nt, int& row0, int& nkv) {
    if (MAP) {
        row0 = 0;
        nkv = attn_map_nkv(bound, t);
    } else {
        attn_window<CAUSAL>(bound, t, nt, row0, nkv);
    }
}

// The arena row that linear window index `i` names. The contiguous modes are the
// resolved base plus the index; the map mode walks the runs, which the compiler
// unrolls — a sharing sequence has two, so the first answers nearly every index.
// `left` (map only) is how many rows that run still holds from `i` on, which is
// what lets a 4-row batch resolve once and then add (see the split body): the
// walk is per batch, not per row.
template <bool MAP>
__device__ __forceinline__ int kv_cell_left(
    const int* __restrict__ bound, int qt, int row0, int i, int& left) {
    if (!MAP) {
        left = 4; // >= any batch the split body stages
        return row0 + i;
    }
    const int* m = bound + (size_t)qt * KV_MAP_MAX_SPANS * 2;
    int off = i;
    #pragma unroll
    for (int r = 0; r < KV_MAP_MAX_SPANS; r++) {
        const int len = m[r * 2 + 1];
        if (off < len) {
            left = len - off;
            return m[r * 2] + off;
        }
        off -= len;
    }
    left = 0;
    return m[0]; // unreachable: the runs' lengths sum to nkv
}

template <bool MAP>
__device__ __forceinline__ int kv_cell(
    const int* __restrict__ bound, int qt, int row0, int i) {
    int left;
    return kv_cell_left<MAP>(bound, qt, row0, i, left);
}

#if __CUDA_ARCH__ >= 800
static __device__ __forceinline__ void gemm_cp16(__half* smem_dst, const __half* gsrc, bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 16 : 0; // src-size 0 => zero-fill the 16B chunk
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}
static __device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
static __device__ __forceinline__ void gemm_cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
static __device__ __forceinline__ void gemm_cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }
#endif // __CUDA_ARCH__ >= 800

#define MMQ_BI 64   // tokens (i) per block tile
#define MMQ_BJ 64   // od rows (j) per block tile
#define MMQ_WS 9    // shared words per tile row: 8 data + 1 bank-conflict pad
#define MMQ_KD 8    // 32-k chunks staged per buffer (256-k, llama.cpp-style);


static __device__ __forceinline__ void mmq_mma_k32(int* d, const int* a, const int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void mmq_mma_k16(int* d, const int* a, int b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

static constexpr int MMQ_NBI = 64;   // tokens (i) per block tile
static constexpr int MMQ_NBJ = 128;  // od rows (j) per block tile

static __device__ __forceinline__ void mmvq_block_reduce_multi(
    const float* acc /* [8] */, float* __restrict__ output, int od, int nt, int t0
) {
    __shared__ float warp_sums[8];
    for (int t = 0; t < nt; ++t) {
        float a = acc[t];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) a += __shfl_xor_sync(0xFFFFFFFF, a, off);
        if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = a;
        __syncthreads();
        if (threadIdx.x == 0) {
            float v = 0.0f;
            #pragma unroll
            for (int k = 0; k < 8; k++) v += warp_sums[k];
            output[(size_t)(t0 + t) * od + (size_t)blockIdx.x] = v;
        }
        __syncthreads(); // warp_sums is rewritten by the next iteration
    }
}
