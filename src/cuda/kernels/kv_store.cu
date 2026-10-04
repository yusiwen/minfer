// src/cuda/kernels/kv_store.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


// ─── KV cache store: scatter nt rows into persistent cache ───

__global__ void store_kv_f32(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int nkt, int nt,
    const int* positions
) {
    int t = blockIdx.x;
    int j = blockIdx.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = src[t * nkt + j];
}

// 8b: f16 KV variant — stores f32 rows as half into the same persistent
// region viewed as half (2 bytes/elem); halves attention read bandwidth.
// P1: one thread converts 4 dims (float4 read -> 2x __half2 store). The
// original one-thread-per-element grid (nt x nkt of SINGLE-THREAD blocks)
// measured ~7 GB/s on the 7B @2K prefill (1.05 M blocks of 1 thread);
// this shape moves the same bytes with 128-thread blocks and vector loads.
// nkt is a multiple of 4 on every CUDA f16-KV path (nkt = nk * hd, hd % 4
// == 0 enforced by the dispatch); the scalar tail keeps odd shapes safe.
__global__ void store_kv_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int nkt, int nt,
    const int* positions
) {
    int t = blockIdx.x;
    int j = (blockIdx.y * blockDim.x + threadIdx.x) * 4;
    if (t >= nt || j >= nkt) return;
    int p = positions[t];
    if (j + 3 < nkt) {
        float4 v = *reinterpret_cast<const float4*>(src + (size_t)t * nkt + j);
        __half2* d = reinterpret_cast<__half2*>(dst + (size_t)p * nkt + j);
        d[0] = __floats2half2_rn(v.x, v.y);
        d[1] = __floats2half2_rn(v.z, v.w);
    } else {
        for (int i = j; i < nkt; i++)
            dst[(size_t)p * nkt + i] = __float2half(src[(size_t)t * nkt + i]);
    }
}

// Quantize ONE block (32 f32 values) into one packed Q8_0 block at `cell`.
// This is the single device-side statement of the C4 S2b quantizer — `d =
// amax/127` stored as an f16 with round-to-nearest-even and every quant
// `rintf(x/d)` clamped to the i8 range — shared by the KV store and, since #144
// item 1, the packed fused decode epilogue. `x` may live in registers or in
// global memory; the loop reads each element once.
__device__ __forceinline__ void q8_0_quantize_block(const float* x, unsigned char* cell) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    *reinterpret_cast<__half*>(cell) = __float2half_rn(d);
    signed char* q = reinterpret_cast<signed char*>(cell + 2);
    #pragma unroll
    for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
        float v = rintf(x[i] * id);
        v = fminf(127.0f, fmaxf(-128.0f, v));
        q[i] = (signed char)(int)v;
    }
}

// C4 S2b: quantize nt f32 rows into packed Q8_0 cells. One thread per
// (row, 32-element block); `row_bytes` is the packed cell's byte width
// (`KvFormat::Q8_0.row_bytes(nkt)` = ceil(nkt/32*34 / 4) * 4 words).
//
// The quantizer is the CPU's, step for step (`quants::quantize_row_q8_0_into`,
// whose aarch64 path is the scalar loop): `d = amax/127` stored as an f16 with
// round-to-nearest-even, and each quant `rintf(x/d)` clamped to the i8 range.
// `rintf` is round-to-nearest-even under the default rounding mode, which is the
// `round_ties_even` the CPU uses — so both backends write the same bytes for the
// same row, and a CPU/device Q8_0 comparison is a layout check, not a tolerance.
__global__ void store_kv_q8_0(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int nkt, int nt, size_t row_bytes,
    const int* positions
) {
    const int t = blockIdx.x;
    const int nblk = nkt / Q8_0_BLOCK_ELEMS;
    const int blk = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= nt || blk >= nblk) return;
    const int p = positions[t];
    const float* x = src + (size_t)t * nkt + (size_t)blk * Q8_0_BLOCK_ELEMS;
    q8_0_quantize_block(x, dst + (size_t)p * row_bytes + (size_t)blk * Q8_0_BLOCK_BYTES);
}

// ─── Fused decode QKV epilogue: bias-add + RoPE + KV-store (nt==1) ───
// D3-8: CUDA port of Metal's kernel_attn_bias_rope_store (G4 FusedQKV). One
// kernel replaces the 7-launch unfused chain (add_bias×3 + rope×2 +
// store_kv×2). q/k/v are POINTER-FORM section bases so one kernel serves
// both decode-QKV layer classes: the concat class (wq|wk|wv same ttype)
// points them INTO the concat matmul output (q=base, k=base+nqt,
// v=base+2*nkt) and the mixed-quant class (e.g. Q6_K attn_v, no concat
// matmul) points them at the three separate matmul outputs. Applies the
// per-section bias, RoPEs q and k IN PLACE, and stores k/v into the
// persistent KV regions. The rope math is VERBATIM rope_f32 (NEOX pairing
// (j, j+half), same freq/theta expression and cosf/sinf — bitwise-identical
// per-element results), the bias add is verbatim add_bias_f32 (same two
// operands, one add), and the store addresses/conversions are verbatim
// store_kv_f32 / store_kv_f16 (dst[pos * nkt + j]; __float2half is RN, the
// same conversion the unfused f16 store's scalar tail uses) — bit-identical
// outputs, fewer launches. Thread mapping (Metal's): one thread per
// (head, d < hd/2) rope pair for q and k, one thread per v element →
// grid = nqt/2 + nkt/2 + nkt. positions[0] is read device-side (nt==1; no
// host scalar crosses the launch — CUDA Graph capture/replay safe).
__global__ void attn_bias_rope_store_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ kv_k,
    float* __restrict__ kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions,
    const int* cells,
    int kv_is_f16
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    // C6: `positions[0]` is the token's index within its sequence (what RoPE
    // rotates by); `cells[0]` is the allocator-resolved KV row. They differ
    // whenever the run does not start at cell 0 (multi-slot server), so the
    // store below addresses rows by `row`, never by `pos`.
    const int pos = positions[0];
    const int row = cells[0];

    if (u < qpairs) {
        // q section: bias + rope in place (attention reads q at offset 0)
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int j  = base + d;
        const int j2 = j + half_dim;
        float x0 = q[j]  + bias_q[j];
        float x1 = q[j2] + bias_q[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        q[j]  = x0 * cs - x1 * sn;
        q[j2] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        // k section: bias + rope in place + store into the K region
        const int u2   = u - qpairs;
        const int head = u2 / half_dim;
        const int d    = u2 % half_dim;
        const int base = head * hd;
        const int j  = base + d;
        const int j2 = j + half_dim;
        float x0 = k[j]  + bias_k[j];
        float x1 = k[j2] + bias_k[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        const float r0 = x0 * cs - x1 * sn;
        const float r1 = x0 * sn + x1 * cs;
        k[j]  = r0;
        k[j2] = r1;
        if (kv_is_f16) {
            ((__half*)kv_k)[(size_t)row * nkt + j]  = __float2half(r0);
            ((__half*)kv_k)[(size_t)row * nkt + j2] = __float2half(r1);
        } else {
            kv_k[(size_t)row * nkt + j]  = r0;
            kv_k[(size_t)row * nkt + j2] = r1;
        }
    } else {
        // v section: bias + store into the V region
        const int j = u - qpairs - kpairs;
        const float val = v[j] + bias_v[j];
        v[j] = val;
        if (kv_is_f16) {
            ((__half*)kv_v)[(size_t)row * nkt + j] = __float2half(val);
        } else {
            kv_v[(size_t)row * nkt + j] = val;
        }
    }
}

// ─── #144 item 1: the PACKED arm of the fused decode epilogue ────────────────
// The f32/f16 kernel above writes one K/V element per thread, which a packed
// cell cannot accept: a Q8_0 block's scale needs all 32 of its elements before
// any of them can be quantized. This arm keeps the same q section (bias + rope
// in place, one thread per pair) and re-maps the K and V sections to one thread
// per (head, 32-element block): the thread computes the block's 32 values
// itself and hands them to `q8_0_quantize_block`, the store's own quantizer, so
// the bytes it writes are the unfused chain's (`add_bias`+`rope`+`store_kv_q8_0`)
// verbatim.
//
// K's 32 roped values are computed from the *unroped* k row plus the rope pair
// partner (element d <-> d + hd/2 within the head, the neox pairing
// `attn_bias_rope_store_f32` uses). Neither K nor V is written back: both fused
// classes leave those buffers dead (attention reads the packed region) and a
// block-owning thread cannot write `k` in place without racing the thread that
// reads its pair partner. The observable output — the packed region's bytes — is
// the unfused chain's.
__global__ void attn_bias_rope_store_q8_0(
    float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    unsigned char* __restrict__ kv_k,
    unsigned char* __restrict__ kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions,
    const int* cells,
    size_t row_bytes
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kblks = nkt / Q8_0_BLOCK_ELEMS;
    const int total = qpairs + 2 * kblks;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    // C6: `positions[0]` is the rope angle's sequence-relative index and
    // `cells[0]` the allocator-resolved row the packed cell is written at.
    const int pos = positions[0];
    const int row = cells[0];

    if (u < qpairs) {
        // q section: bias + rope in place (verbatim attn_bias_rope_store_f32)
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int j  = base + d;
        const int j2 = j + half_dim;
        float x0 = q[j]  + bias_q[j];
        float x1 = q[j2] + bias_q[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        q[j]  = x0 * cs - x1 * sn;
        q[j2] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kblks) {
        // K section: one (head, block). `b` is also the block's index inside the
        // packed row (blocks are laid out in flat element order).
        const int b = u - qpairs;
        const int blk = b % (hd / Q8_0_BLOCK_ELEMS);
        const int head = b / (hd / Q8_0_BLOCK_ELEMS);
        float x[Q8_0_BLOCK_ELEMS];
        #pragma unroll
        for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
            // `d` is the element's index inside the head, so the block's own
            // offset must be added — without it every block of a head would
            // compute the head's first 32 values (caught by
            // cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer's byte arm).
            const int d = blk * Q8_0_BLOCK_ELEMS + i;
            const int dd = (d < half_dim) ? d : d - half_dim;
            const int ja = head * hd + dd;
            const int jb = ja + half_dim;
            float x0 = k[ja] + bias_k[ja];
            float x1 = k[jb] + bias_k[jb];
            float freq = freq_scale / powf(freq_base, (2.0f * dd) / hd);
            float theta = pos * freq;
            float cs = cosf(theta), sn = sinf(theta);
            x[i] = (d < half_dim) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
        }
        q8_0_quantize_block(x, kv_k + (size_t)row * row_bytes + (size_t)b * Q8_0_BLOCK_BYTES);
    } else {
        // V section: one block, bias + quantize. The bias is folded into the
        // quantized value only: like the K section, the V buffer itself is dead
        // in both fused classes (attention reads the packed region), and leaving
        // it unwritten lets `v` stay `const`.
        const int b = u - qpairs - kblks;
        const int base = b * Q8_0_BLOCK_ELEMS;
        float x[Q8_0_BLOCK_ELEMS];
        #pragma unroll
        for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) x[i] = v[base + i] + bias_v[base + i];
        q8_0_quantize_block(x, kv_v + (size_t)row * row_bytes + (size_t)b * Q8_0_BLOCK_BYTES);
    }
}
extern "C" {

void launch_store_kv_f32(
    const float* src, float* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 grid(nt, nkt, 1);
    minfer_launch_prelude("launch:store_kv_f32", "store_kv_f32");
    store_kv_f32<<<grid, minfer_launch_block("launch:store_kv_f32", dim3(1, 1, 1)), 0, stream>>>(src, dst, nkt, nt, positions);
    minfer_launch_ok("launch:store_kv_f32", "store_kv_f32");
}

void launch_store_kv_f16(
    const float* src, void* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 block(128, 1, 1);
    dim3 grid(nt, (nkt / 4 + 127) / 128, 1);
    minfer_launch_prelude("launch:store_kv_f16", "store_kv_f16");
    store_kv_f16<<<grid, minfer_launch_block("launch:store_kv_f16", block), 0, stream>>>(src, (__half*)dst, nkt, nt, positions);
    minfer_launch_ok("launch:store_kv_f16", "store_kv_f16");
}

// C4 S2b: the packed store. One thread per (row, 32-element block); `row_bytes`
// is the packed cell's byte width, which the host takes from
// `KvFormat::Q8_0.row_bytes(nkt)`. `nkt` must be a multiple of 32 — `ensure_kv`'s
// `check_width` is what refuses anything else, so the grid arithmetic is exact.
void launch_store_kv_q8_0(
    const float* src, void* dst, int nkt, int nt, size_t row_bytes,
    const int* positions, cudaStream_t stream
) {
    const int nblk = nkt / Q8_0_BLOCK_ELEMS;
    dim3 block(64, 1, 1);
    dim3 grid(nt, (nblk + 63) / 64, 1);
    minfer_launch_prelude("launch:store_kv_q8_0", "store_kv_q8_0");
    store_kv_q8_0<<<grid, minfer_launch_block("launch:store_kv_q8_0", block), 0, stream>>>(
        src, (unsigned char*)dst, nkt, nt, row_bytes, positions);
    minfer_launch_ok("launch:store_kv_q8_0", "store_kv_q8_0");
}

// D3-8: fused decode QKV epilogue launcher — 256-thread blocks over the
// flat (nqt/2 + nkt/2 + nkt) thread mapping (Metal's dispatch_1d shape).
// C4 S2b: F32/F16 only. A packed cache never builds the fused epilogue (the
// builders' `layer_gpu` gate gains `&& !packed`), and the host wrapper refuses
// `layout == Q8_0` loudly rather than let the per-element store address a packed
// cell it has no whole block to quantize.
void launch_attn_bias_rope_store(
    float* q, float* k, float* v,
    const void* bias_q, const void* bias_k, const void* bias_v,
    void* kv_k, void* kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions, const int* cells, int kv_is_f16,
    cudaStream_t stream
) {
    const int total = nqt / 2 + nkt / 2 + nkt;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    minfer_launch_prelude("launch:attn_bias_rope_store", "attn_bias_rope_store_f32");
    attn_bias_rope_store_f32<<<grid, minfer_launch_block("launch:attn_bias_rope_store", block), 0, stream>>>(
        q, k, v,
        (const float*)bias_q, (const float*)bias_k, (const float*)bias_v,
        (float*)kv_k, (float*)kv_v,
        nqt, nkt, hd, freq_base, freq_scale, positions, cells, kv_is_f16);
    minfer_launch_ok("launch:attn_bias_rope_store", "attn_bias_rope_store_f32");
}

// #144 item 1: the packed arm's launcher. Same 256-thread blocks over the
// (nqt/2 qpairs + nkt/32 K blocks + nkt/32 V blocks) thread mapping; `row_bytes`
// is the packed cell's byte width, the same number `store_kv_q8_0` is given.
void launch_attn_bias_rope_store_q8_0(
    float* q, const float* k, const float* v,
    const void* bias_q, const void* bias_k, const void* bias_v,
    void* kv_k, void* kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions, const int* cells, size_t row_bytes,
    cudaStream_t stream
) {
    const int total = nqt / 2 + 2 * (nkt / Q8_0_BLOCK_ELEMS);
    const int block = 256;
    const int grid = (total + block - 1) / block;
    minfer_launch_prelude("launch:attn_bias_rope_store_q8_0", "attn_bias_rope_store_q8_0");
    attn_bias_rope_store_q8_0<<<grid, minfer_launch_block("launch:attn_bias_rope_store_q8_0", block), 0, stream>>>(
        q, k, v,
        (const float*)bias_q, (const float*)bias_k, (const float*)bias_v,
        (unsigned char*)kv_k, (unsigned char*)kv_v,
        nqt, nkt, hd, freq_base, freq_scale, positions, cells, row_bytes);
    minfer_launch_ok("launch:attn_bias_rope_store_q8_0", "attn_bias_rope_store_q8_0");
}
}

// ─── C3/C7b: move KV rows within one arena (arena compaction) ───────────────
// One block walks the rows one at a time with a barrier between them, **in the
// direction the overlap requires**: ascending when the run slides down, descending
// when it slides up (C7b, where growing a run pushes the runs above it up). Either
// way the row a write could clobber has already been copied. Overlapping is the
// normal case — a compaction slides a run into the gap next to it — and
// `cudaMemcpyAsync` device-to-device is documented undefined for overlapping
// ranges, which is exactly why this is a kernel and not a memcpy: no staging
// buffer, no second pass.
__global__ void kv_move_rows(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int dst_row, int src_row, int rows, int elems
) {
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const bool down = dst_row <= src_row;
    for (int k = 0; k < rows; k++) {
        const int r = down ? k : (rows - 1 - k);
        const float* s = src + ((size_t)src_row + (size_t)r) * (size_t)elems;
        float* d = dst + ((size_t)dst_row + (size_t)r) * (size_t)elems;
        for (int i = tid; i < elems; i += nth) d[i] = s[i];
        // Every thread must finish this row before any thread touches the next one:
        // a later write can land on a row that is still being read.
        __syncthreads();
    }
}

// Returns 0 on success, non-zero when the contract is violated or the launch
// itself failed (the caller turns that into an `Err`, never a silent no-op).
// A documented `_opt` site (#162): the int return is the decision, so the launch
// failure is named at the site and cleared, and no sticky is set.
extern "C" int launch_kv_move_rows(
    float* dst, const float* src,
    int dst_row, int src_row, int rows, int elems,
    cudaStream_t stream
) {
    if (rows <= 0 || elems <= 0) return 0;
    if (dst_row < 0 || src_row < 0) return 1;
    minfer_launch_prelude("launch:kv_move_rows", "kv_move_rows");
    kv_move_rows<<<1, minfer_launch_block("launch:kv_move_rows", 256), 0, stream>>>(dst, src, dst_row, src_row, rows, elems);
    return minfer_launch_ok_opt("launch:kv_move_rows", "kv_move_rows") ? 0 : 1;
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// These kernels are plain `__global__` functions, but their module is loaded
// by the pre-warm, so the family keeps its own registration entry.
extern "C" void minfer_prewarm_kv_store_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, store_kv_f16);
}
