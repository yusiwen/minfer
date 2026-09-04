// CUDA kernels for minfer — Q4_0 matmul + element-wise ops.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <mma.h>
#include <cstdint>

// ─── Block size constants (must match src/block.rs) ───────────
#define Q4B  18   // sizeof(BlockQ4_0): half d + uchar qs[16]
#define Q41B 20   // sizeof(BlockQ4_1): half d + half m + uchar qs[16]
#define Q8B  34   // sizeof(BlockQ8_0): half d + char qs[32]
#define Q4KB 144  // sizeof(BlockQ4_K)
#define Q5KB 176  // sizeof(BlockQ5_K)
#define Q6KB 210  // sizeof(BlockQ6_K)
#define WARP 32

// ─── Helper: warp-level sum reduction ─────────────────────────
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ─── Helper: fp16 → f32 (using CUDA intrinsics) ──────────────
__device__ float h2f(uint16_t h) {
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

// ─── Q4_0 × Q8_0 matrix multiplication (bit-exact with CPU) ──
// Thread block: 64 threads (2 warps × 32 lanes)
// Each warp computes NR0=4 consecutive output rows
// Grid: x = ceil(od / (NR0*NSG)), y = nt

__global__ void q4_0_q8_0_matmul(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nb = id / 32;
    int q4s = nb * Q4B;
    int q8s = nb * Q8B;

    const uint8_t* xr = acts + t * q8s;

    float sumf[NR0];
    #pragma unroll
    for (int row = 0; row < NR0; row++) sumf[row] = 0.0f;

    // Each lane handles every WARP-th block
    for (int b = lane_id; b < nb; b += WARP) {
        // Q8_0 block
        float d8 = h2f(*reinterpret_cast<const uint16_t*>(xr + b * Q8B));
        const int8_t* xq = reinterpret_cast<const int8_t*>(xr + b * Q8B + 2);

        for (int row = 0; row < NR0; row++) {
            int o = r0 + row;
            if (o >= od) break;

            const uint8_t* wr = weights + o * q4s;
            float d4 = h2f(*reinterpret_cast<const uint16_t*>(wr + b * Q4B));
            const uint8_t* wq = wr + b * Q4B + 2;

            int bs = 0;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint8_t byte = wq[j];
                bs += (int(byte & 0x0F) - 8) * int(xq[j])
                    + (int(byte >> 4) - 8) * int(xq[j + 16]);
            }
            sumf[row] += float(bs) * d4 * d8;
        }
    }

    // Warp-level reduction and write
    for (int row = 0; row < NR0; row++) {
        int o = r0 + row;
        if (o < od) {
            float total = warp_reduce_sum(sumf[row]);
            if (lane_id == 0) {
                output[t * od + o] = total;
            }
        }
    }
}

// ─── Q4_0 × f32 matrix multiplication ─────────────────────────
// Thread block: 64 threads (2 warps), each warp computes 4 rows
// Grid: x = ceil(od / 8), y = nt

__device__ float block_q4_0_dot_y(const uint8_t* block, float sumy, const float* yl, int il) {
    float d = h2f(*reinterpret_cast<const uint16_t*>(block));
    const uint16_t* qs = reinterpret_cast<const uint16_t*>(block + 2) + il / 2;
    float acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
    #pragma unroll
    for (int i = 0; i < 8; i += 2) {
        uint16_t v = qs[i / 2];
        acc0 += yl[i + 0] * float(v & 0x000F);
        acc1 += yl[i + 1] * float(v & 0x0F00);
        acc2 += yl[i + 8] * float(v & 0x00F0);
        acc3 += yl[i + 9] * float(v & 0xF000);
    }
    return d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
}

__global__ void q4_0_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int QK = 32;
    const int NW = 32;
    const int NQ = 16;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt) return;

    int nb = id / QK;
    int q4s = nb * Q4B;

    const uint8_t* ax0 = weights + (r0 + 0) * q4s;
    const uint8_t* ax1 = weights + (r0 + 1) * q4s;
    const uint8_t* ax2 = weights + (r0 + 2) * q4s;
    const uint8_t* ax3 = weights + (r0 + 3) * q4s;
    const float* y = acts + t * id;

    int ix = lane_id / (NW / NQ);
    int il = (lane_id % (NW / NQ)) * 8;

    float sumf0 = 0, sumf1 = 0, sumf2 = 0, sumf3 = 0;
    float yl[16];
    const float* yb = y + ix * QK + il;

    for (int ib = ix; ib < nb; ib += NQ) {
        float sumy0 = 0, sumy1 = 0;
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sy = sumy0 + sumy1;
        if (r0 + 0 < od) sumf0 += block_q4_0_dot_y(ax0 + ib * Q4B, sy, yl, il);
        if (r0 + 1 < od) sumf1 += block_q4_0_dot_y(ax1 + ib * Q4B, sy, yl, il);
        if (r0 + 2 < od) sumf2 += block_q4_0_dot_y(ax2 + ib * Q4B, sy, yl, il);
        if (r0 + 3 < od) sumf3 += block_q4_0_dot_y(ax3 + ib * Q4B, sy, yl, il);
        yb += QK * NQ;
    }

    sumf0 = warp_reduce_sum(sumf0);
    sumf1 = warp_reduce_sum(sumf1);
    sumf2 = warp_reduce_sum(sumf2);
    sumf3 = warp_reduce_sum(sumf3);
    if (lane_id == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
        if (r0 + 2 < od) output[t * od + r0 + 2] = sumf2;
        if (r0 + 3 < od) output[t * od + r0 + 3] = sumf3;
    }
}

// ─── Q8_0 × f32 matrix multiplication ─────────────────────────
// Each block: fp16 d + 32 × int8 qs. Dot product: d * sum(qs[i] * x[i]).
// Thread block: 64 threads (2 warps), each warp computes 4 rows
// Grid: x = ceil(od / 8), y = nt

__global__ void q8_0_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int QK = 32;
    const int QK4 = QK / 4;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nb = id / QK;
    int ws = nb * Q8B;
    const float* y = acts + t * id;

    float sumf[NR0] = {0};

    for (int row = 0; row < NR0 && r0 + row < od; row++) {
        const uint8_t* wr = weights + (r0 + row) * ws;
        float sum = 0.0f;

        for (int b = lane_id; b < nb; b += WARP) {
            float d8 = h2f(*reinterpret_cast<const uint16_t*>(wr + b * Q8B));
            const int8_t* qs = reinterpret_cast<const int8_t*>(wr + b * Q8B + 2);
            const float4* x4 = reinterpret_cast<const float4*>(y + b * QK);

            float bs = 0.0f;
            #pragma unroll
            for (int i = 0; i < QK4; i++) {
                float4 xv = x4[i];
                bs += float(qs[i*4 + 0]) * xv.x
                    + float(qs[i*4 + 1]) * xv.y
                    + float(qs[i*4 + 2]) * xv.z
                    + float(qs[i*4 + 3]) * xv.w;
            }
            sum += bs * d8;
        }
        sumf[row] = sum;
    }

    for (int row = 0; row < NR0 && r0 + row < od; row++) {
        sumf[row] = warp_reduce_sum(sumf[row]);
        if (lane_id == 0) {
            output[t * od + r0 + row] = sumf[row];
        }
    }
}

// ─── Q4_1 × f32 matrix multiplication ─────────────────────────
// Q4_1 block: fp16 d (scale), fp16 m (min), 16 packed nibble bytes (32 elts).
// val_i = nibble_i * d + m  →  dot = d * sum(nibble_i * x_i) + m * sum(x_i).
// Thread block: 64 threads (2 warps), each warp computes 4 rows.
// Grid: x = ceil(od / 8), y = nt.

__global__ void q4_1_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int QK = 32;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nb = id / QK;
    int ws = nb * Q41B;
    const float* y = acts + t * id;

    float sumf[NR0] = {0};

    for (int row = 0; row < NR0 && r0 + row < od; row++) {
        const uint8_t* wr = weights + (r0 + row) * ws;
        float sum = 0.0f;

        for (int b = lane_id; b < nb; b += WARP) {
            const uint8_t* block = wr + b * Q41B;
            float d = h2f(*reinterpret_cast<const uint16_t*>(block));
            float m = h2f(*reinterpret_cast<const uint16_t*>(block + 2));
            const uint8_t* qs = block + 4;
            const float* xb = y + b * QK;

            float sumx = 0.0f;
            float sumq = 0.0f;

            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint8_t byte = qs[j];
                float x0 = xb[j];
                float x1 = xb[j + 16];
                sumx += x0 + x1;
                sumq += float(byte & 0x0F) * x0 + float(byte >> 4) * x1;
            }
            sum += sumq * d + sumx * m;
        }
        sumf[row] = sum;
    }

    for (int row = 0; row < NR0 && r0 + row < od; row++) {
        sumf[row] = warp_reduce_sum(sumf[row]);
        if (lane_id == 0) {
            output[t * od + r0 + row] = sumf[row];
        }
    }
}

// ─── Q5_1 × f32 matrix multiplication ─────────────────────────
// Q5_1 block: 24 bytes / 32 elements — f16 d, f16 m, u32 qh (bit j ↔ elem j,
// bit j+16 ↔ elem j+16), 16 bytes qs (byte j: low nibble = elem j, high =
// elem j+16). Value = d * unsigned_5bit + m (NO −16 offset — Q5_1 has a min).
// Structure mirrors q4_0_f32_matmul (4 rows/warp, 2 warps/block, lanes
// stride 32-element blocks).
__global__ void q5_1_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (t >= nt || r0 >= od) return;

    int nb = id / 32;
    int row_stride = nb * 24;
    const float* y = acts + (size_t)t * id;

    float acc[NR0];
    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) acc[rr] = 0.0f;

    for (int b = lane_id; b < nb; b += WARP) {
        const float* xb = y + b * 32;
        float sumx = 0.0f;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            float4 xv = *reinterpret_cast<const float4*>(xb + v * 4);
            sumx += xv.x + xv.y + xv.z + xv.w;
        }
        #pragma unroll
        for (int rr = 0; rr < NR0; rr++) {
            int o = r0 + rr;
            if (o >= od) break;
            const uint8_t* blk = weights + (size_t)o * row_stride + b * 24;
            float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
            float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
            uint32_t qh = *reinterpret_cast<const uint32_t*>(blk + 4);
            const uint8_t* qs = blk + 8;
            float sdot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                float u_lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1);
                float u_hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1);
                sdot += u_lo * xb[j] + u_hi * xb[j + 16];
            }
            acc[rr] += d * sdot + m * sumx;
        }
    }

    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) {
        int o = r0 + rr;
        if (o < od) {
            float v = warp_reduce_sum(acc[rr]);
            if (lane_id == 0) output[t * od + o] = v;
        }
    }
}

// forward declaration (defined with the Q4_K section below)
__device__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m);

// ─── Q5_K × f32 matrix multiplication ─────────────────────────
// Q5_K super-block: 176 bytes / 256 elements — f16 d, f16 dmin, scales[12]
// (same 6-bit packing as Q4_K), qh[32] (bit s of byte l = the >16 bit of
// element (sub s, pos l) — TRANSPOSED vs the nibble order), qs[128] (4
// chunks of 32 bytes; chunk ci: low nibbles = sub 2ci, high = sub 2ci+1;
// byte l ↔ element l of the sub). Value = d·s[sub]·(nib + 16·bit) −
// dmin·m[sub], unsigned (no −16).
// Tail handling: id % 32 == 0 is required (dispatch guard); a partial last
// super-block masks whole 32-element sub-blocks — neither the weights'
// padding nibbles nor the next token's activations are touched.
__global__ void q5_k_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int QKK = 256;
    const int NR0 = 4;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (t >= nt || r0 >= od) return;

    int nbe = (id + QKK - 1) / QKK;
    int row_stride = nbe * 176;
    const float* y = acts + (size_t)t * id;

    float acc[NR0];
    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) acc[rr] = 0.0f;

    for (int u = lane_id; u < nbe * NR0; u += WARP) {
        int ib = u % nbe;
        int rr = u / nbe;
        const uint8_t* blk = weights + (size_t)(r0 + rr) * row_stride + ib * 176;
        float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const uint8_t* sc = blk + 4;
        const uint8_t* qh = blk + 16;
        const uint8_t* qs = blk + 48;
        const float* yb = y + (size_t)ib * QKK;

        uint8_t sc_s[8], sc_m[8];
        #pragma unroll
        for (int j = 0; j < 8; j++) get_scale_min_k4(j, sc, &sc_s[j], &sc_m[j]);

        // valid sub-blocks in this super-block (tail masking, id % 32 == 0)
        int valid = min(8, (id - ib * QKK + 31) / 32);

        float partial = 0.0f;
        for (int sub = 0; sub < valid; sub++) {
            int ci = sub >> 1;
            int hi = sub & 1;
            const uint8_t* q4 = qs + ci * 32;
            const float* xs = yb + sub * 32;
            float sdot = 0.0f, sx = 0.0f;
            #pragma unroll
            for (int l = 0; l < 32; l++) {
                float nib = hi ? float(q4[l] >> 4) : float(q4[l] & 0x0F);
                float w = nib + 16.0f * float((qh[l] >> sub) & 1);
                sdot += w * xs[l];
                sx += xs[l];
            }
            partial += d * float(sc_s[sub]) * sdot - dm * float(sc_m[sub]) * sx;
        }
        acc[rr] += partial;
    }

    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) {
        float v = warp_reduce_sum(acc[rr]);
        if (lane_id == 0 && r0 + rr < od) output[t * od + r0 + rr] = v;
    }
}

// ─── Helper: unpack Q4_K 6-bit scale and min ────────────────
// Q4_K stores 16 × 6-bit values (8 scales + 8 mins) packed into 12 bytes.
// This mirrors Metal's get_scale_min_k4 and Rust block.rs::unpack_q4k_scales.

__device__ void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4)  | ((q[j]   >> 6) << 4);
    }
}

// ─── Q4_K × f32 matrix multiplication ─────────────────────────
// Q4_K super-block: 256 elements, 8 sub-blocks × 32.
// Block (144 bytes): fp16 d, fp16 dmin, uchar scales[12], uchar qs[128].
// Dequant: val = d * scale[sub] * nibble - dmin * min[sub].
// Q4_K nibble layout (llama.cpp format): byte j low nibble = elem j,
// byte j high nibble = elem j+16 (within sub-block).
// NR0=2 rows per warp, NSG=2 warps per block (4 rows per block).
// Grid: x = ceil(od / 4), y = nt.

// ─── Q4_K × f32 matrix multiplication (7e②: vectorized + unit mapping) ──
// Each lane owns (row, super-block) pairs — all 32 lanes stay busy even
// when nbe < 32 (the 7B FFN shapes have nbe = 14, which idled 18/32 lanes
// in the lane-per-block layout). Weight loads are uint4 (Q4KB = 144 is
// 16-byte aligned), activations float4; measured 3× the pair-layout
// kernel on the 7B shapes (~163 GB/s vs ~42 GB/s).
__global__ void q4_k_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int QKK = 256;
    const int NR0 = 4;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nbe = (id + QKK - 1) / QKK;
    int row_stride = nbe * Q4KB;
    const float* y = acts + t * id;

    float acc[NR0];
    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) acc[rr] = 0.0f;

    for (int u = lane_id; u < nbe * NR0; u += WARP) {
        int ib = u % nbe;
        int rr = u / nbe;
        const uint8_t* blk = weights + (r0 + rr) * row_stride + ib * Q4KB;

        float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const uint8_t* sc = blk + 4;
        const uint8_t* qs = blk + 16;
        const float* yb = y + ib * QKK;

        uint8_t sc_s[8], sc_m[8];
        for (int j = 0; j < 8; j++) get_scale_min_k4(j, sc, &sc_s[j], &sc_m[j]);

        float partial = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const float* yl = yb + j * 64;
            float slo = 0.0f, shi = 0.0f, syl = 0.0f, syh = 0.0f;
            #pragma unroll
            for (int u2 = 0; u2 < 2; u2++) {
                uint4 q = *reinterpret_cast<const uint4*>(qs + j * 32 + u2 * 16);
                const uint8_t* b = reinterpret_cast<const uint8_t*>(&q);
                const float* ylo = yl + u2 * 16;
                const float* yhi = yl + 32 + u2 * 16;
                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    float4 ya = *reinterpret_cast<const float4*>(ylo + v * 4);
                    float4 yb4 = *reinterpret_cast<const float4*>(yhi + v * 4);
                    slo += float(b[v*4+0] & 0x0F) * ya.x + float(b[v*4+1] & 0x0F) * ya.y
                         + float(b[v*4+2] & 0x0F) * ya.z + float(b[v*4+3] & 0x0F) * ya.w;
                    shi += float(b[v*4+0] >> 4) * yb4.x + float(b[v*4+1] >> 4) * yb4.y
                         + float(b[v*4+2] >> 4) * yb4.z + float(b[v*4+3] >> 4) * yb4.w;
                    syl += ya.x + ya.y + ya.z + ya.w;
                    syh += yb4.x + yb4.y + yb4.z + yb4.w;
                }
            }
            partial += d * (float(sc_s[2 * j]) * slo + float(sc_s[2 * j + 1]) * shi)
                     - dm * (float(sc_m[2 * j]) * syl + float(sc_m[2 * j + 1]) * syh);
        }
        acc[rr] += partial;
    }

    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) {
        float v = warp_reduce_sum(acc[rr]);
        if (lane_id == 0 && r0 + rr < od) output[t * od + r0 + rr] = v;
    }
}

// ─── 8e-reversal: llama.cpp MMVQ structure for DECODE (nt == 1) ───────────
// The original 8e verdict ("116 GB/s = the platform's streaming limit") was
// wrong: a plain read-only kernel does 252.7 GB/s on GB10 (93% of the 273
// GB/s theoretical). The f32-activation kernel achieved only ~46% because it
// runs 2 warps x 4 rows with each lane serially processing whole 144B blocks
// (~28K threads in flight at 7B ffn_down) — not enough parallelism to hide
// LPDDR latency. llama.cpp's mul_mat_vec_q uses ONE output row per block
// with the row's (block, 32-element sub-block) units spread round-robin over
// 256 threads (8 warps), int dp4a dots over q8-quantized activations, and a
// block-wide reduction — ~917K threads in flight at the same shape.
// Measured (bench8e2, L2-defeated, 7B shapes): 194–207 GB/s vs 112–117,
// i.e. +74–77% per matmul; at id < 2048 the win collapses to noise
// (launch-latency bound), so dispatch gates on id >= 2048.
//
// Activation layout: padded 40-byte q8_0 blocks — [f16 d][2B pad][32B int8]
// — so the int8 payload is 4-byte aligned for the uint32/dp4a reads. The
// scratch (buf_q8_decode) is size-stable per graph (id fixed), grown during
// the warmup runs, never inside a capture window.
#define Q8PB 40

// 40B layout: 2B f16 d, 2B pad, 32B int8 payload (offset 4), 4B i32 sum of the
// quantized values (offset 36 — the pad40 slack). The sum feeds the MMQ prefill
// GEMM's min-term correction (llama.cpp's q8_1 "s"); the MMVQ decode kernels
// only read d and the payload, so the extra word is invisible to them.
__global__ void quantize_q8_0_pad40(
    const float* __restrict__ x,
    uint8_t* __restrict__ y,
    int dim, int nt
) {
    int nb = dim / 32;
    int total = nt * nb;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;
    int t = tid / nb;
    int b = tid % nb;
    const float* src = x + (size_t)t * dim + b * 32;
    uint8_t* dst = y + ((size_t)t * nb + b) * Q8PB;
    // P6: tree-reduced amax (the serial fmaxf chain was latency-bound)
    // and 16B loads / 4B register-packed stores. Math is bit-identical:
    // max is exact for any association, the rintf pass is unchanged.
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

__global__ void __launch_bounds__(256) q4_k_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = (id + 255) / 256;
    const int row_stride = nbe * Q4KB;
    const int nsub = (id + 31) / 32; // ceil — partial tail super-blocks excluded
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + blk_i * Q4KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        // sub-block nibbles: chunk (sub>>1) of 32B, lo nibbles for even sub,
        // hi for odd; element l of the sub-block ↔ byte l.
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 16 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t w = qw[v];
            const int n = lo ? (int)(w & 0x0F0F0F0F) : (int)((w >> 4) & 0x0F0F0F0F);
            const int xa = (int)xw[v]; // q8 block covers exactly this sub-block
            dot = __dp4a(n, xa, dot);
            sx  = __dp4a(0x01010101, xa, sx);
        }
        // value = d*s*nib − dm*m (dm is the block's own dmin, not d)
        acc += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xFFFFFFFF, acc, off);
    __shared__ float warp_sums[8];
    if ((threadIdx.x & 31) == 0) warp_sums[threadIdx.x >> 5] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; k++) v += warp_sums[k];
        output[(size_t)t * od + row] = v;
    }
}

// 8e follow-up: the warp+block reduction shared by the q5_K/q6_K MMVQ
// kernels (same shape as the inline one in q4_k_q8_mmvq).
__device__ __forceinline__ void mmvq_block_reduce(
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

// 8e follow-up: decode (nt == 1) Q6_K MMVQ — same llama.cpp GB10 structure
// as q4_k_q8_mmvq (one row per 256-thread block, sub-block units round-robin
// across lanes, dp4a over q8 activations). Q6_K sub-blocks are 16 elements
// (16 signed 6-bit scales per 256-element super-block, no min term), so the
// unit is half of a 32-element q8 activation block and the per-unit dot is
// 4 dp4a. Element l of sub-block s (l in [0,16), s in [0,16)):
//   chunk = s/8, group g = (s/2)%4, half is = s%2
//   ql byte  = chunk*64 + (g%2)*32 + is*16 + l   (lo nibble for g<2, hi else)
//   qh byte  = 128 + chunk*32 + is*16 + l        (2-bit pair g per element)
//   scale    = (int8) blk[192 + s]; value = d * scale * (q6 - 32)
// q6_K block strides are 210B raw / 224B padded (7e② repack) — both even but
// not 4-aligned, so the weight side reads 2-byte halves (llama.cpp
// get_int_b2 style); the 256B-aligned base keeps every access 2B-aligned.
__global__ void __launch_bounds__(256) q6_k_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = (id + 255) >> 8;
    const int row_stride = nbe * blk_stride;
    const int nsub = (id + 15) >> 4; // ceil — partial tail super-blocks excluded
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 4, s = u & 15;
        const int chunk = s >> 3, g = (s >> 1) & 3, is = s & 1;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)blk_i * blk_stride;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const float sc = (float)(int8_t)blk[192 + s];
        const uint8_t* ql = blk + chunk * 64 + (g & 1) * 32 + is * 16;
        const uint8_t* qh = blk + 128 + chunk * 32 + is * 16;
        const uint8_t* x8 = x8row + (size_t)(u >> 1) * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4) + (u & 1) * 4;
        int dot = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const uint32_t wl = (uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * v) |
                                ((uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * v + 2) << 16);
            const uint32_t wh = (uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * v) |
                                ((uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * v + 2) << 16);
            const uint32_t nib = (g < 2) ? (wl & 0x0F0F0F0F) : ((wl >> 4) & 0x0F0F0F0F);
            const uint32_t hi = ((wh >> (2 * g)) & 0x03030303) << 4;
            // q6 nibble+high pair is 0..63; subtract 32 per byte (in-range,
            // never saturates) to get the signed value for dp4a
            const int vi = __vsubss4((int)(nib | hi), 0x20202020);
            dot = __dp4a(vi, (int)xw[v], dot);
        }
        acc += d8 * sc * d * (float)dot;
    }

    mmvq_block_reduce(acc, output, od, t);
}

// 8e follow-up: decode (nt == 1) Q5_K MMVQ — the q4_k_q8_mmvq structure with
// the q5 high-bit plane folded in. Sub-blocks are 32 elements (scales/mins
// packed like q4_K via get_scale_min_k4). Element l of sub-block s:
//   nibble byte = 48 + (s/2)*32 + l   (lo nibble for even s, hi for odd)
//   high bit    = (qh[l] >> s) & 1    (qh plane at byte 16, 1 bit per element)
//   value = d*s5*(nib | bit<<4) − dm*m5  → acc += d8*(s8*d*dot − m8*dm*sx)
// 176B block stride is 16-byte aligned, so the weight side uses uint32 loads.
__global__ void __launch_bounds__(256) q5_k_q8_mmvq(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = (id + 255) >> 8;
    const int row_stride = nbe * Q5KB;
    const int nsub = (id + 31) >> 5; // ceil — partial tail super-blocks excluded
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < nsub; u += 256) {
        const int blk_i = u >> 3, sub = u & 7;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)blk_i * Q5KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        uint8_t s8, m8;
        get_scale_min_k4(sub, blk + 4, &s8, &m8);
        const uint32_t* qw = reinterpret_cast<const uint32_t*>(blk + 48 + (sub >> 1) * 32);
        const bool lo = (sub & 1) == 0;
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot = 0, sx = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t w = qw[v];
            // qh word v holds the high bits of elements 4v..4v+3 (byte l of
            // the qh plane, bit `sub`) — one word per nibble word
            const uint32_t qh32 = *reinterpret_cast<const uint32_t*>(blk + 16 + 4 * v);
            const uint32_t nib = lo ? (w & 0x0F0F0F0F) : ((w >> 4) & 0x0F0F0F0F);
            const uint32_t hi = ((qh32 >> sub) & 0x01010101) << 4;
            const int xa = (int)xw[v];
            dot = __dp4a((int)(nib | hi), xa, dot);
            sx  = __dp4a(0x01010101, xa, sx);
        }
        // value = d*s*(nib|bit) − dm*m (dm is the block's own dmin, not d)
        acc += d8 * ((float)s8 * (float)d * (float)dot - (float)m8 * (float)dm * (float)sx);
    }

    mmvq_block_reduce(acc, output, od, t);
}

// R2: weight-streaming rework of the K-quant MMVQ kernels. The 8e kernels
// read each 32B nibble chunk per SUB-BLOCK (the sibling sub re-reads the
// same bytes for the other nibble half — 2× the load instructions, L1
// absorbed) and q6_K used eight 2-byte loads per 16-byte ql/qh piece. The
// v2 kernels map one thread to a 32-element CHUNK (q4_K/q5_K: a sub-pair
// sharing its nibble bytes; q6_K: an is-pair sharing ql/qh bytes), so each
// weight byte is loaded exactly once per row and every access in the
// padded (224B-stride) q6_K layout is 4-byte aligned. Dispatch prefers v2
// when id % 256 == 0 (full super-blocks); MINFER_MMVQ_V1=1 forces the old
// kernels for A/B.
__global__ void __launch_bounds__(256) q4_k_q8_mmvq_v2(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * Q4KB;
    const int npair = id >> 6;         // 64-element chunks (sub-pairs)
    const int nsub = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 2, c = u & 3;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * Q4KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const int s0 = 2 * c, s1 = 2 * c + 1;
        uint8_t s8a, m8a, s8b, m8b;
        get_scale_min_k4(s0, blk + 4, &s8a, &m8a);
        get_scale_min_k4(s1, blk + 4, &s8b, &m8b);
        // one 32B nibble chunk: lo nibbles = sub s0's 32 elements, hi = sub s1's
        // (16B-aligned: 144·kbx + 16 + 32·c ≡ 0 mod 16)
        const uint4 w0 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32);
        const uint4 w1 = *reinterpret_cast<const uint4*>(blk + 16 + c * 32 + 16);
        const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const uint8_t* x8a = x8row + (size_t)(kbx * 8 + s0) * Q8PB;
        const uint8_t* x8b = x8row + (size_t)(kbx * 8 + s1) * Q8PB;
        const float d8a = h2f(*reinterpret_cast<const uint16_t*>(x8a));
        const float d8b = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xa = reinterpret_cast<const uint32_t*>(x8a + 4);
        const uint32_t* xb = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dota = 0, sxa = 0, dotb = 0, sxb = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t wv = ws[v];
            const int xa_v = (int)xa[v], xb_v = (int)xb[v];
            dota = __dp4a((int)(wv & 0x0F0F0F0F), xa_v, dota);
            sxa  = __dp4a(0x01010101, xa_v, sxa);
            dotb = __dp4a((int)((wv >> 4) & 0x0F0F0F0F), xb_v, dotb);
            sxb  = __dp4a(0x01010101, xb_v, sxb);
        }
        acc += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
             + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
    }

    mmvq_block_reduce(acc, output, od, t);
}

__global__ void __launch_bounds__(256) q5_k_q8_mmvq_v2(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * Q5KB;
    const int npair = id >> 6;
    const int nsub = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * nsub * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 2, c = u & 3;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * Q5KB;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
        const float dm = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
        const int s0 = 2 * c, s1 = 2 * c + 1;
        uint8_t s8a, m8a, s8b, m8b;
        get_scale_min_k4(s0, blk + 4, &s8a, &m8a);
        get_scale_min_k4(s1, blk + 4, &s8b, &m8b);
        const uint4 w0 = *reinterpret_cast<const uint4*>(blk + 48 + c * 32);
        const uint4 w1 = *reinterpret_cast<const uint4*>(blk + 48 + c * 32 + 16);
        // the qh plane is 32 bytes SHARED by all 8 sub-blocks (byte l holds
        // one high bit per sub for element l) — every chunk reads the same
        // bytes, only the bit index (s0/s1) differs
        const uint4 h0 = *reinterpret_cast<const uint4*>(blk + 16);
        const uint4 h1 = *reinterpret_cast<const uint4*>(blk + 16 + 16);
        const uint32_t ws[8] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w};
        const uint32_t hs[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
        const uint8_t* x8a = x8row + (size_t)(kbx * 8 + s0) * Q8PB;
        const uint8_t* x8b = x8row + (size_t)(kbx * 8 + s1) * Q8PB;
        const float d8a = h2f(*reinterpret_cast<const uint16_t*>(x8a));
        const float d8b = h2f(*reinterpret_cast<const uint16_t*>(x8b));
        const uint32_t* xa = reinterpret_cast<const uint32_t*>(x8a + 4);
        const uint32_t* xb = reinterpret_cast<const uint32_t*>(x8b + 4);
        int dota = 0, sxa = 0, dotb = 0, sxb = 0;
        #pragma unroll
        for (int v = 0; v < 8; v++) {
            const uint32_t wv = ws[v];
            // qh byte l holds one high bit per sub for element l: bit s of
            // the bytes covering this chunk's elements
            const uint32_t qhv = hs[v];
            const uint32_t hia = (((qhv >> s0) & 0x01010101u) << 4);
            const uint32_t hib = (((qhv >> s1) & 0x01010101u) << 4);
            const int xa_v = (int)xa[v], xb_v = (int)xb[v];
            dota = __dp4a((int)((wv & 0x0F0F0F0F) | hia), xa_v, dota);
            sxa  = __dp4a(0x01010101, xa_v, sxa);
            dotb = __dp4a((int)(((wv >> 4) & 0x0F0F0F0F) | hib), xb_v, dotb);
            sxb  = __dp4a(0x01010101, xb_v, sxb);
        }
        acc += d8a * ((float)s8a * d * (float)dota - (float)m8a * dm * (float)sxa)
             + d8b * ((float)s8b * d * (float)dotb - (float)m8b * dm * (float)sxb);
    }

    mmvq_block_reduce(acc, output, od, t);
}

// q6_K v2: one thread per 32-element is-pair (two 16-element sub-blocks
// sharing their ql/qh bytes and one q8 block). Requires the padded 224B
// block stride so every ql/qh access is 4-byte aligned (u32 loads).
__global__ void __launch_bounds__(256) q6_k_q8_mmvq_v2(
    const uint8_t* __restrict__ weights,
    const uint8_t* __restrict__ acts8,
    float* __restrict__ output,
    int od, int id, int nt, int blk_stride
) {
    const int row = blockIdx.x;
    const int t = blockIdx.y;
    const int nbe = id >> 8;
    const int row_stride = nbe * blk_stride;
    const int npair = id >> 5;
    const uint8_t* x8row = acts8 + (size_t)t * (id >> 5) * Q8PB;

    float acc = 0.0f;
    for (int u = threadIdx.x; u < npair; u += 256) {
        const int kbx = u >> 3, pair = u & 7;
        const int s0 = 2 * pair, s1 = 2 * pair + 1;
        const uint8_t* blk = weights + (size_t)row * row_stride + (size_t)kbx * blk_stride;
        const float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
        const float sc0 = (float)(int8_t)blk[192 + s0];
        const float sc1 = (float)(int8_t)blk[192 + s1];
        // v1 mapping with s = 2*pair + half: chunk = s>>3 = pair>>2,
        // g = (s>>1)&3 = pair&3, is = s&1 = half (the pair's two subs share
        // chunk/g; only the 16-byte is-half differs)
        const int chunk = pair >> 2, g = pair & 3;
        // padded 224B stride ⇒ every ql/qh piece is 16B aligned
        const uint4 qla = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32);
        const uint4 qlb = *reinterpret_cast<const uint4*>(blk + chunk * 64 + (g & 1) * 32 + 16);
        const uint4 qha = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32);
        const uint4 qhb = *reinterpret_cast<const uint4*>(blk + 128 + chunk * 32 + 16);
        const uint32_t qls[8] = {qla.x, qla.y, qla.z, qla.w, qlb.x, qlb.y, qlb.z, qlb.w};
        const uint32_t qhs[8] = {qha.x, qha.y, qha.z, qha.w, qhb.x, qhb.y, qhb.z, qhb.w};
        const uint32_t shift = 2 * g;
        const uint8_t* x8 = x8row + (size_t)u * Q8PB;
        const float d8 = h2f(*reinterpret_cast<const uint16_t*>(x8));
        const uint32_t* xw = reinterpret_cast<const uint32_t*>(x8 + 4);
        int dot0 = 0, dot1 = 0;
        #pragma unroll
        for (int v = 0; v < 4; v++) {
            const uint32_t wl0 = qls[v], wl1 = qls[v + 4];
            const uint32_t wh0 = qhs[v], wh1 = qhs[v + 4];
            const uint32_t nib0 = (g < 2) ? (wl0 & 0x0F0F0F0F) : ((wl0 >> 4) & 0x0F0F0F0F);
            const uint32_t nib1 = (g < 2) ? (wl1 & 0x0F0F0F0F) : ((wl1 >> 4) & 0x0F0F0F0F);
            const uint32_t hi0 = ((wh0 >> shift) & 0x03030303) << 4;
            const uint32_t hi1 = ((wh1 >> shift) & 0x03030303) << 4;
            const int vi0 = __vsubss4((int)(nib0 | hi0), 0x20202020);
            const int vi1 = __vsubss4((int)(nib1 | hi1), 0x20202020);
            dot0 = __dp4a(vi0, (int)xw[v], dot0);
            dot1 = __dp4a(vi1, (int)xw[v + 4], dot1);
        }
        acc += d8 * sc0 * d * (float)dot0 + d8 * sc1 * d * (float)dot1;
    }

    mmvq_block_reduce(acc, output, od, t);
}

__global__ void q6_k_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int QKK = 256;
    const int NR0 = 2;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nbe = (id + QKK - 1) / QKK;
    int row_stride = nbe * Q6KB;

    const uint8_t* w0 = weights + (r0 + 0) * row_stride;
    // NR0 = 2 with odd od: the last group's row 1 does not exist — alias it
    // to row 0 (in-bounds); the write guard discards the sum (7e review)
    const bool row1_ok = (r0 + 1) < od;
    const uint8_t* w1 = row1_ok ? weights + (r0 + 1) * row_stride : w0;
    const float* y = acts + t * id;

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (int ib = lane_id; ib < nbe; ib += WARP) {
        const uint8_t* blk0 = w0 + ib * Q6KB;
        const uint8_t* blk1 = w1 + ib * Q6KB;

        float bd0 = h2f(*reinterpret_cast<const uint16_t*>(blk0 + 208));
        float bd1 = h2f(*reinterpret_cast<const uint16_t*>(blk1 + 208));

        const uint8_t* ql0 = blk0;
        const uint8_t* ql1 = blk1;
        const uint8_t* qh0 = blk0 + 128;
        const uint8_t* qh1 = blk1 + 128;
        const int8_t* sc0 = (const int8_t*)(blk0 + 192);
        const int8_t* sc1 = (const int8_t*)(blk1 + 192);
        const float* yb = y + ib * QKK;

        // 7e②: the y side (the hot cache-resident stream) loads float4
        // groups instead of 32-float-strided scalars; ql/qh stay per-byte —
        // Q6KB = 210 is not 16-byte aligned, so vector weight loads would
        // need a repacked layout (possible follow-up).
        for (int n = 0; n < 2; n++) {
            #pragma unroll
            for (int g = 0; g < 2; g++) {
                // l stays in [0,32): two 16-element groups per half; the four
                // terms ys[0/32/64/96] use scales sc[n*8 + g + {0, 2, 4, 6}]
                // (si = l/16 + n*8 in the scalar formulation, is = g).
                const float* yb_n = yb + n * 128 + g * 16;

                float p00 = 0.0f, p01 = 0.0f, p02 = 0.0f, p03 = 0.0f;
                float p10 = 0.0f, p11 = 0.0f, p12 = 0.0f, p13 = 0.0f;

                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    const int l = g * 16 + v * 4;
                    float4 ys0 = *reinterpret_cast<const float4*>(yb_n + 0);
                    float4 ys1 = *reinterpret_cast<const float4*>(yb_n + 32);
                    float4 ys2 = *reinterpret_cast<const float4*>(yb_n + 64);
                    float4 ys3 = *reinterpret_cast<const float4*>(yb_n + 96);

                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const int b = l + r;
                        int qh0_b = qh0[b];
                        int qh1_b = qh1[b];
                        int q0_0 = ((int)(ql0[b] & 0xF) | ((qh0_b & 3) << 4)) - 32;
                        int q1_0 = ((int)(ql1[b] & 0xF) | ((qh1_b & 3) << 4)) - 32;
                        int q0_1 = ((int)(ql0[b + 32] & 0xF) | (((qh0_b >> 2) & 3) << 4)) - 32;
                        int q1_1 = ((int)(ql1[b + 32] & 0xF) | (((qh1_b >> 2) & 3) << 4)) - 32;
                        int q0_2 = ((int)(ql0[b] >> 4) | (((qh0_b >> 4) & 3) << 4)) - 32;
                        int q1_2 = ((int)(ql1[b] >> 4) | (((qh1_b >> 4) & 3) << 4)) - 32;
                        int q0_3 = ((int)(ql0[b + 32] >> 4) | (((qh0_b >> 6) & 3) << 4)) - 32;
                        int q1_3 = ((int)(ql1[b + 32] >> 4) | (((qh1_b >> 6) & 3) << 4)) - 32;

                        // component index is r (the element within the
                        // float4), not v — the v selector was the 7e② bug
                        // (3/4 of the y values were never read).
                        const float* c0 = reinterpret_cast<const float*>(&ys0);
                        const float* c1 = reinterpret_cast<const float*>(&ys1);
                        const float* c2 = reinterpret_cast<const float*>(&ys2);
                        const float* c3 = reinterpret_cast<const float*>(&ys3);
                        const float y0 = c0[r];
                        const float y1 = c1[r];
                        const float y2 = c2[r];
                        const float y3 = c3[r];
                        p00 += float(q0_0) * y0;
                        p01 += float(q0_1) * y1;
                        p02 += float(q0_2) * y2;
                        p03 += float(q0_3) * y3;
                        p10 += float(q1_0) * y0;
                        p11 += float(q1_1) * y1;
                        p12 += float(q1_2) * y2;
                        p13 += float(q1_3) * y3;
                    }
                    yb_n += 4;
                }
                int si = n * 8 + g;
                sumf0 += bd0 * (float(sc0[si + 0]) * p00 + float(sc0[si + 2]) * p01
                              + float(sc0[si + 4]) * p02 + float(sc0[si + 6]) * p03);
                sumf1 += bd1 * (float(sc1[si + 0]) * p10 + float(sc1[si + 2]) * p11
                              + float(sc1[si + 4]) * p12 + float(sc1[si + 6]) * p13);
            }
            ql0 += 64; ql1 += 64;
            qh0 += 32; qh1 += 32;
        }
    }

    sumf0 = warp_reduce_sum(sumf0);
    sumf1 = warp_reduce_sum(sumf1);
    if (lane_id == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
    }
}

// ─── Q6_K × f32 matmul, PADDED weight layout (7e②) ────────────
// Registered via register_weight_q6k_padded: each 210-byte block lives in a
// 224-byte slot (224 = 14×16), so every block — and the ql/qh/scales fields
// inside it — is 16-byte aligned and the weight stream uses uint4 loads.
// This is the 7B decode bottleneck fix: Q6_K (ffn_down + output.weight) is
// ~45% of the q4_K_M weight traffic and the 210-byte stride previously
// forced 1-byte-per-instruction reads.
__global__ void q6_k_f32_matmul_padded(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int QKK = 256;
    const int NR0 = 2;
    const int NSG = 2;
    const int Q6KPB = 224;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (t >= nt || r0 >= od) return;

    int nbe = (id + QKK - 1) / QKK;
    int row_stride = nbe * Q6KPB;

    const uint8_t* w0 = weights + (r0 + 0) * row_stride;
    // NR0 = 2 with odd od: the last group's row 1 does not exist — alias it
    // to row 0 (in-bounds); the write guard discards the sum (7e review)
    const bool row1_ok = (r0 + 1) < od;
    const uint8_t* w1 = row1_ok ? weights + (r0 + 1) * row_stride : w0;
    const float* y = acts + t * id;

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (int ib = lane_id; ib < nbe; ib += WARP) {
        const uint8_t* blk0 = w0 + ib * Q6KPB;
        const uint8_t* blk1 = w1 + ib * Q6KPB;

        float bd0 = h2f(*reinterpret_cast<const uint16_t*>(blk0 + 208));
        float bd1 = h2f(*reinterpret_cast<const uint16_t*>(blk1 + 208));

        const uint8_t* ql0 = blk0;
        const uint8_t* ql1 = blk1;
        const uint8_t* qh0 = blk0 + 128;
        const uint8_t* qh1 = blk1 + 128;
        const int8_t* sc0 = (const int8_t*)(blk0 + 192);
        const int8_t* sc1 = (const int8_t*)(blk1 + 192);
        const float* yb = y + ib * QKK;

        for (int n = 0; n < 2; n++) {
            #pragma unroll
            for (int g = 0; g < 2; g++) {
                // 16 elements per group (l = g*16 .. g*16+15); the four terms
                // ys[0/32/64/96] use scales sc[n*8 + g + {0, 2, 4, 6}].
                const float* yb_n = yb + n * 128 + g * 16;

                float p00 = 0.0f, p01 = 0.0f, p02 = 0.0f, p03 = 0.0f;
                float p10 = 0.0f, p11 = 0.0f, p12 = 0.0f, p13 = 0.0f;

                #pragma unroll
                for (int v = 0; v < 4; v++) {
                    // 16 weight bytes per group per source = one uint4 each
                    uint4 ql0a = *reinterpret_cast<const uint4*>(ql0 + n * 64 + g * 16);
                    uint4 ql0b = *reinterpret_cast<const uint4*>(ql0 + n * 64 + 32 + g * 16);
                    uint4 ql1a = *reinterpret_cast<const uint4*>(ql1 + n * 64 + g * 16);
                    uint4 ql1b = *reinterpret_cast<const uint4*>(ql1 + n * 64 + 32 + g * 16);
                    uint4 qh0a = *reinterpret_cast<const uint4*>(qh0 + n * 32 + g * 16);
                    uint4 qh1a = *reinterpret_cast<const uint4*>(qh1 + n * 32 + g * 16);
                    const uint8_t* a0 = reinterpret_cast<const uint8_t*>(&ql0a);
                    const uint8_t* b0 = reinterpret_cast<const uint8_t*>(&ql0b);
                    const uint8_t* a1 = reinterpret_cast<const uint8_t*>(&ql1a);
                    const uint8_t* b1 = reinterpret_cast<const uint8_t*>(&ql1b);
                    const uint8_t* h0 = reinterpret_cast<const uint8_t*>(&qh0a);
                    const uint8_t* h1 = reinterpret_cast<const uint8_t*>(&qh1a);

                    float4 ys0 = *reinterpret_cast<const float4*>(yb_n + 0);
                    float4 ys1 = *reinterpret_cast<const float4*>(yb_n + 32);
                    float4 ys2 = *reinterpret_cast<const float4*>(yb_n + 64);
                    float4 ys3 = *reinterpret_cast<const float4*>(yb_n + 96);
                    const float* c0 = reinterpret_cast<const float*>(&ys0);
                    const float* c1 = reinterpret_cast<const float*>(&ys1);
                    const float* c2 = reinterpret_cast<const float*>(&ys2);
                    const float* c3 = reinterpret_cast<const float*>(&ys3);

                    #pragma unroll
                    for (int r = 0; r < 4; r++) {
                        const int j = v * 4 + r;
                        int h0b = h0[j];
                        int h1b = h1[j];
                        int q0_0 = ((int)(a0[j] & 0xF) | ((h0b & 3) << 4)) - 32;
                        int q1_0 = ((int)(a1[j] & 0xF) | ((h1b & 3) << 4)) - 32;
                        int q0_1 = ((int)(b0[j] & 0xF) | (((h0b >> 2) & 3) << 4)) - 32;
                        int q1_1 = ((int)(b1[j] & 0xF) | (((h1b >> 2) & 3) << 4)) - 32;
                        int q0_2 = ((int)(a0[j] >> 4) | (((h0b >> 4) & 3) << 4)) - 32;
                        int q1_2 = ((int)(a1[j] >> 4) | (((h1b >> 4) & 3) << 4)) - 32;
                        int q0_3 = ((int)(b0[j] >> 4) | (((h0b >> 6) & 3) << 4)) - 32;
                        int q1_3 = ((int)(b1[j] >> 4) | (((h1b >> 6) & 3) << 4)) - 32;

                        p00 += float(q0_0) * c0[r];
                        p01 += float(q0_1) * c1[r];
                        p02 += float(q0_2) * c2[r];
                        p03 += float(q0_3) * c3[r];
                        p10 += float(q1_0) * c0[r];
                        p11 += float(q1_1) * c1[r];
                        p12 += float(q1_2) * c2[r];
                        p13 += float(q1_3) * c3[r];
                    }
                    yb_n += 4;
                }
                int si = n * 8 + g;
                sumf0 += bd0 * (float(sc0[si + 0]) * p00 + float(sc0[si + 2]) * p01
                              + float(sc0[si + 4]) * p02 + float(sc0[si + 6]) * p03);
                sumf1 += bd1 * (float(sc1[si + 0]) * p10 + float(sc1[si + 2]) * p11
                              + float(sc1[si + 4]) * p12 + float(sc1[si + 6]) * p13);
            }
        }
    }

    sumf0 = warp_reduce_sum(sumf0);
    sumf1 = warp_reduce_sum(sumf1);
    if (lane_id == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
    }
}

// ─── Row gather / embedding (7e③) ─────────────────────────────
// Embedding = gather + dequantize weight rows on device (removes the CPU
// round trips around the prefill's embed and G3 tail get_rows). ids are
// I32-as-f32 bit patterns (exact for |v| < 2^24), read via __float2int_rn.
// The generic f32 gather (get_rows: out[t*n+i] = x[ids[t]*n+i]) shares the
// f32 kernel.

__global__ void gather_rows_f32(
    const float* __restrict__ src,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n, int nt
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)nt * n;
    if (idx >= total) return;
    int t = (int)(idx / n);
    int i = (int)(idx % n);
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    out[idx] = src[(long long)id * n + i];
}

__global__ void embed_rows_q8_0(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    const int BS = 34; // f16 d + 32 int8
    int nb = n_embd / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nb) return;
    int t = tid / nb, b = tid % nb;
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    const uint8_t* blk = w + ((long long)id * nb + b) * BS;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const int8_t* q = (const int8_t*)(blk + 2);
    float* o = out + (long long)t * n_embd + b * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) o[i] = d * float(q[i]);
}

__global__ void embed_rows_q4_0(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    const int BS = 18; // f16 d + 16 nibble bytes
    int nb = n_embd / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nb) return;
    int t = tid / nb, b = tid % nb;
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    const uint8_t* blk = w + ((long long)id * nb + b) * BS;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const uint8_t* q = blk + 2;
    float* o = out + (long long)t * n_embd + b * 32;
    // element j = LOW nibble of byte j; element j+16 = HIGH nibble.
    // minfer Q4_0 stores round(v/d) + 8 (same -8 offset as the matmuls and
    // the CPU embed path).
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i] = d * (float(q[i] & 0x0F) - 8.0f);
        o[i + 16] = d * (float(q[i] >> 4) - 8.0f);
    }
}

__global__ void embed_rows_q4_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    int nsp = n_embd / 256; // super-blocks per row
    int nsub = nsp * 8;     // 32-element sub-blocks per row
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nsub) return;
    int t = tid / nsub, s = tid % nsub;
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    int sp = s / 8, sub = s % 8;
    const uint8_t* blk = w + ((long long)id * nsp + sp) * Q4KB;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    // sub-block s covers elements [32s..32s+31]: chunk j = s/2, LOW nibbles
    // for even s, HIGH for odd (scale index s).
    int j = sub / 2, half = sub % 2;
    const uint8_t* q = blk + 16 + j * 32;
    float* o = out + (long long)t * n_embd + s * 32;
    float ds = d * float(scb), dmm = dmin * float(mb);
    #pragma unroll
    for (int l = 0; l < 32; l++) {
        float nib = half ? float(q[l] >> 4) : float(q[l] & 0x0F);
        o[l] = ds * nib - dmm;
    }
}

// Q6_K: one thread per 16-element sub-block (16 per 256 super-block).
// block_stride = 210 (raw GGUF) or 224 (padded registration).
__global__ void embed_rows_q6_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt, int block_stride
) {
    int nsp = n_embd / 256;
    int nsub = nsp * 16;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nsub) return;
    int t = tid / nsub, s = tid % nsub;
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    int sp = s / 16, sub = s % 16;
    const uint8_t* blk = w + ((long long)id * nsp + sp) * block_stride;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    // sub s = n*8 + tt*2 + g: element base n*128 + tt*32 + g*16
    int n = sub / 8, rem = sub % 8, tt = rem / 2, g = rem % 2;
    int ql_off = n * 64 + (tt % 2) * 32 + g * 16;
    int qh_off = n * 32 + g * 16;
    int sc_idx = n * 8 + tt * 2 + g;
    float* o = out + (long long)t * n_embd + sp * 256 + n * 128 + tt * 32 + g * 16;
    float dsc = d * float(sc[sc_idx]);
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int nib = (tt < 2) ? (ql[ql_off + r] & 0x0F) : (ql[ql_off + r] >> 4);
        int q2 = (qh[qh_off + r] >> (tt * 2)) & 3;
        o[r] = dsc * float((nib | (q2 << 4)) - 32);
    }
}

// ─── F32 × F32 matmul (7e④) ───────────────────────────────────
// Same unit lane mapping as the q4_K kernel: lanes own (row, 256-elem
// chunk) pairs, float4 loads on both operands. Requires id % 8 == 0 for
// the aligned float4 loads; the scalar kernel covers the general case.
__global__ void f32_f32_matmul_vec(
    const float* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int CHK = 256;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int t = blockIdx.y;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (t >= nt || r0 >= od) return;

    int nch = (id + CHK - 1) / CHK;
    const float* y = acts + (size_t)t * id;

    float acc[NR0];
    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) acc[rr] = 0.0f;

    for (int u = lane_id; u < nch * NR0; u += WARP) {
        int ic = u % nch, rr = u / nch;
        const float* wr = weights + (size_t)(r0 + rr) * id + ic * CHK;
        const float* yc = y + ic * CHK;
        int len = min(CHK, id - ic * CHK);
        float p = 0.0f;
        // the unit's lane streams the WHOLE chunk (8 floats per pass)
        for (int i = 0; i < len; i += 8) {
            float4 a0 = *reinterpret_cast<const float4*>(wr + i);
            float4 a1 = *reinterpret_cast<const float4*>(wr + i + 4);
            float4 b0 = *reinterpret_cast<const float4*>(yc + i);
            float4 b1 = *reinterpret_cast<const float4*>(yc + i + 4);
            p += a0.x * b0.x + a0.y * b0.y + a0.z * b0.z + a0.w * b0.w
               + a1.x * b1.x + a1.y * b1.y + a1.z * b1.z + a1.w * b1.w;
        }
        acc[rr] += p;
    }

    #pragma unroll
    for (int rr = 0; rr < NR0; rr++) {
        float v = warp_reduce_sum(acc[rr]);
        if (lane_id == 0 && r0 + rr < od) output[(size_t)t * od + r0 + rr] = v;
    }
}

// General-case fallback: one thread per (token, output) pair, scalar dot.
__global__ void f32_f32_matmul_scalar(
    const float* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nt * od) return;
    int t = (int)(idx / od), r = (int)(idx % od);
    const float* wr = weights + (size_t)r * id;
    const float* y = acts + (size_t)t * id;
    float acc = 0.0f;
    for (int i = 0; i < id; i++) acc += wr[i] * y[i];
    output[idx] = acc;
}

// ─── Quantize f32 → Q8_0 (1 thread per 32-element block) ─────
// Matches CPU scalar path: half delta + 32 signed int8 values

__global__ void quantize_q8_0(
    const float* __restrict__ x,
    uint8_t* __restrict__ y,
    int dim, int nt
) {
    int nb = dim / 32;
    int total = nt * nb;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total) return;

    int t = tid / nb;
    int b = tid % nb;

    const float* src = x + t * dim + b * 32;
    uint8_t* dst = y + (t * nb + b) * Q8B;

    float am = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(src[j]));
    float d = am / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    *reinterpret_cast<__half*>(dst) = __float2half(d);

    for (int j = 0; j < 32; j++) {
        int q = int(rintf(src[j] * id));
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        dst[2 + j] = uint8_t(int8_t(q));
    }
}

// ─── RMSNorm (32 threads per row, no shared memory) ──────────
// y[t][i] = x[t][i] * rsqrt(mean(x[t]²) + eps) * w[i]

__global__ void rms_norm_f32(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    int d, float eps, int n
) {
    int row = blockIdx.x;
    if (row >= n) return;

    int tid = threadIdx.x;
    int d4 = d / 4;

    const float4* x4 = reinterpret_cast<const float4*>(x + row * d);

    float ss = 0.0f;
    for (int i = tid; i < d4; i += WARP) {
        float4 v = x4[i];
        ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    ss = warp_reduce_sum(ss);

    float scale = rsqrtf(ss / (float)d + eps);

    float4* y4 = reinterpret_cast<float4*>(y + row * d);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int i = tid; i < d4; i += WARP) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        y4[i].x = xv.x * scale * wv.x;
        y4[i].y = xv.y * scale * wv.y;
        y4[i].z = xv.z * scale * wv.z;
        y4[i].w = xv.w * scale * wv.w;
    }
}

// ─── Add bias: y[t][i] += b[i] ───────────────────────────────

__global__ void add_bias_f32(
    float* __restrict__ y,
    const float* __restrict__ b,
    int d
) {
    int t = blockIdx.x, i = threadIdx.x + blockIdx.y * blockDim.x;
    if (i >= d) return;
    y[t * d + i] += b[i];
}

// ─── Element-wise add: z = x + y ─────────────────────────────

__global__ void add_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    z[tid] = x[tid] + y[tid];
}

// ─── Element-wise multiply: z = x * y ────────────────────────

__global__ void mul_f32(
    const float* __restrict__ x,
    const float* __restrict__ y,
    float* __restrict__ z,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    z[tid] = x[tid] * y[tid];
}

// ─── SiLU in-place: y = y / (1 + exp(-y)) ────────────────────

__global__ void silu_f32(float* y, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float v = y[tid];
    y[tid] = v / (1.0f + expf(-v));
}

// ─── SwiGLU fused: dst = silu(gate) * up ─────────────────────

// 7e⑤: in-place split swiglu over one buffer — buf[i] = silu(buf[i]) *
// buf[off + i] (the fused FFN concat matmul output: gate rows 0..nf, up
// rows nf..2*nf; results written back into the gate rows).
__global__ void swiglu_f32_off(float* __restrict__ buf, int n, int off) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float g = buf[tid];
    buf[tid] = (g / (1.0f + expf(-g))) * buf[off + tid];
}

__global__ void swiglu_f32(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    float* __restrict__ dst,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    float g = gate[tid];
    dst[tid] = (g / (1.0f + expf(-g))) * up[tid];
}

// ─── I32 input decode: positions/token ids arrive as f32::from_bits(v)
// bit patterns (graph convention, alloc.rs fill_input_i32) while the rope /
// store / attention kernels read raw int32. One elementwise pass
// reinterprets the bits into a scratch buffer — fully device-side, so the
// per-layer path needs no host sync (and stays CUDA-Graph-replayable).

__global__ void f32_bits_to_i32(
    const float* __restrict__ src,
    int* __restrict__ dst,
    int n
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n) return;
    dst[tid] = __float_as_int(src[tid]);
}

// ─── RoPE (NEOX-style, in-place) ─────────────────────────────
// x layout: [nt][n_head][n_dims] — pairs (x[i], x[i+half])
// NEOX-style: pairs (x[i], x[i+hd/2]) for each head

__global__ void rope_f32(
    float* x,
    int n_head, int n_dims, int nt,
    float freq_base, float freq_scale,
    const int* positions
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= n_head) return;

    int half = n_dims / 2;
    int base = (t * n_head + h) * n_dims;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float freq = freq_scale / powf(freq_base, (2.0f * i) / n_dims);
        float theta = positions[t] * freq;
        float cs = cosf(theta), sn = sinf(theta);
        int j = base + i;
        int j2 = j + half;
        float x0 = x[j], x1 = x[j2];
        x[j]  = x0 * cs - x1 * sn;
        x[j2] = x0 * sn + x1 * cs;
    }
}

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

// helper: convert a half4 (hd is a multiple of 4) to float4
__device__ __forceinline__ float4 h4_to_f4(const __half* p) {
    float2 a = __half22float2(*reinterpret_cast<const __half2*>(p));
    float2 b = __half22float2(*reinterpret_cast<const __half2*>(p + 2));
    return make_float4(a.x, a.y, b.x, b.y);
}

// ─── GQA Attention (online softmax, 32 threads/head/token) ───
// q/k/v/o layout: [nt][nh][hd]; k/v stored as [nkv][nk][hd]

// 8b: GQA attention over an f16 KV cache — exact structural mirror of
// gqa_attn_f32 (same online softmax, same reductions); the ONLY difference
// is the K/V load mechanics: half4 → float4 conversions, f32 accumulation
// everywhere (Metal pl_gqa_attn_f16 precision class).
__global__ void gqa_attn_f32_f16kv(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ o,
    const int* positions,
    int nh, int nk, int hd,
    float scale, int nt
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= nh) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int hk = h / gqa;
    int ne_q = nh * hd;
    int stride_kv = nk * hd;

    const float* qhead = q + t * ne_q + h * hd;
    float* ohead = o + t * ne_q + h * hd;

    int tid = threadIdx.x;
    int hd4 = hd / 4;
    const float4* q4 = reinterpret_cast<const float4*>(qhead);

    // Online softmax with persistent accumulators
    const int NE = 2;
    const int C = WARP * NE;

    float mx = -INFINITY;
    float S = 0.0f;
    float4 oc[32];
    #pragma unroll
    for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

    for (int batch = 0; batch < nkv; batch += C) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = batch + tid * NE;
        int kv1 = kv0 + 1;

        if (kv0 < nkv) {
            const __half* krow = k + (size_t)kv0 * stride_kv + hk * hd;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = h4_to_f4(krow + i * 4);
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s0 = d * scale;
        }
        if (kv1 < nkv) {
            const __half* krow = k + (size_t)kv1 * stride_kv + hk * hd;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = h4_to_f4(krow + i * 4);
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s1 = d * scale;
        }

        float batch_mx = fmaxf(s0, s1);
        // Warp-level max reduction
        for (int off = 16; off > 0; off >>= 1)
            batch_mx = fmaxf(batch_mx, __shfl_xor_sync(0xFFFFFFFF, batch_mx, off));
        float new_mx = fmaxf(mx, batch_mx);
        float corr = expf(mx - new_mx);

        float e0 = expf(s0 - new_mx);
        float e1 = expf(s1 - new_mx);

        #pragma unroll
        for (int i = 0; i < hd4; i++) oc[i].x *= corr, oc[i].y *= corr, oc[i].z *= corr, oc[i].w *= corr;
        S *= corr;

        if (kv0 < nkv) {
            const __half* vrow = v + (size_t)kv0 * stride_kv + hk * hd;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = h4_to_f4(vrow + i * 4);
                oc[i].x += e0 * vv.x; oc[i].y += e0 * vv.y;
                oc[i].z += e0 * vv.z; oc[i].w += e0 * vv.w;
            }
        }
        if (kv1 < nkv) {
            const __half* vrow = v + (size_t)kv1 * stride_kv + hk * hd;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = h4_to_f4(vrow + i * 4);
                oc[i].x += e1 * vv.x; oc[i].y += e1 * vv.y;
                oc[i].z += e1 * vv.z; oc[i].w += e1 * vv.w;
            }
        }
        S += e0 + e1;
        mx = new_mx;
    }

    // Warp-level reduction of S and oc
    S = warp_reduce_sum(S);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        oc[i].x = warp_reduce_sum(oc[i].x);
        oc[i].y = warp_reduce_sum(oc[i].y);
        oc[i].z = warp_reduce_sum(oc[i].z);
        oc[i].w = warp_reduce_sum(oc[i].w);
    }

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float4* o4 = reinterpret_cast<float4*>(ohead);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        o4[i].x = oc[i].x * inv;
        o4[i].y = oc[i].y * inv;
        o4[i].z = oc[i].z * inv;
        o4[i].w = oc[i].w * inv;
    }
}

// ─── 8d: split-K decode attention (flash-decoding) ───────────
// The single-warp-per-(token,head) online-softmax kernel leaves the GPU idle
// at nt == 1 (28 warps total at 7B) — nsys showed 48% of the 7B decode step
// at 2K ctx. Pass 1 scans SPLITS KV chunks in parallel (fixed grid, device-
// side range split so CUDA Graph capture stays valid); pass 2 merges the
// partial (mx, S, oc) results. Partial layout: [SPLITS][nh][pstr] floats,
// pstr = (4+hd+3)&~3 — oc starts at float offset 4 so the float4 writes are
// 16-byte aligned. Scratch is a fixed-size state buffer (nh/hd are graph
// constants), grown during warmup — never inside a capture window.
//
// R-followup rewrite (dim-parallel lanes). nsys on the previous version
// showed: (a) the hd-wide float4 oc[32] accumulator is runtime-indexed
// (hd is a kernel argument) so it lives in LOCAL MEMORY — ~80 MB of local
// traffic per layer, re-read+re-written on every online-softmax rescale;
// (b) each lane walked whole K/V rows with 4-byte loads — 64 scattered
// sector requests per row (12.5% sector utilization), L1-bandwidth bound;
// (c) only 224 single-warp blocks (~4.7 warps/SM). Net: ~150 us/layer at
// 7B @2K (~28 GB/s effective on a 4.3 MB K+V read) — the entire @2K
// decode gap to llama.cpp (28 x 151 us = 4.2 ms of a ~25.8 ms step) — and
// it got MONOTONICALLY worse with more splits (148 -> 172 -> 419 -> 609 us
// for 8/16/32/64) because more resident warps thrash L1 with the local
// oc arrays. Now each lane owns 4 fixed dims: the accumulator is ONE
// float4 in registers (zero spill, hd <= 128 enforced by the dispatch),
// every K/V access is a perfectly-coalesced row instruction, and the row
// dot is a warp reduction. Rows run in batches of 4 to overlap the serial
// online-softmax chain. Idle splits still write an mx=-INF/S=0 partial
// that the combine weights to zero; the [SPLITS][nh][pstr] layout is
// unchanged (combine untouched).

#define ATTN_SPLITS 32

template <typename KV>
__device__ __forceinline__ float4 kv_ld4(const KV* p);

template <>
__device__ __forceinline__ float4 kv_ld4<float>(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

template <>
__device__ __forceinline__ float4 kv_ld4<__half>(const __half* p) {
    __half2 a = *reinterpret_cast<const __half2*>(p);
    __half2 b = *reinterpret_cast<const __half2*>(p + 2);
    float2 x = __half22float2(a);
    float2 y = __half22float2(b);
    return make_float4(x.x, x.y, y.x, y.y);
}

template <typename KV>
__global__ void gqa_attn_split_partial(
    const float* __restrict__ q,
    const KV* __restrict__ k,
    const KV* __restrict__ v,
    float* __restrict__ partial,
    const int* positions,
    int nh, int nk, int hd, float scale, int pstr
) {
    const int SPLITS = ATTN_SPLITS;
    int sp = blockIdx.x;
    int h = blockIdx.y;
    int nkv = positions[0] + 1;
    int chunk = (nkv + SPLITS - 1) / SPLITS;
    int lo = sp * chunk;
    int hi = min(nkv, lo + chunk);

    int lane_id = threadIdx.x;
    int gqa = nh / nk;
    int hk = h / gqa;
    int stride_kv = nk * hd;
    // Each lane owns 4 consecutive dims (hd % 4 == 0 and hd <= 128 are
    // enforced by the dispatch); lanes with d0 >= hd are idle but keep
    // participating in the warp reductions (zero contribution).
    int d0 = lane_id * 4;
    bool live = d0 < hd;
    const float4 q4 = live ? *reinterpret_cast<const float4*>(q + h * hd + d0)
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    float mx = -INFINITY, S = 0.0f;
    float4 oc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    for (int base = lo; base < hi; base += 4) {
        int nr = min(4, hi - base); // warp-uniform
        // Stage K for the whole batch first; the V addresses are already
        // known, so the compiler hoists those loads above the softmax chain.
        float4 k4[4];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            k4[j] = (live && j < nr)
                ? kv_ld4<KV>(k + (size_t)(base + j) * stride_kv + hk * hd + d0)
                : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            if (j >= nr) break; // warp-uniform: all lanes exit together
            // Full-row dot: this lane's 4-dim partial, then a warp reduction
            // so every lane holds the row's complete dot (uniform softmax).
            float d = q4.x * k4[j].x + q4.y * k4[j].y + q4.z * k4[j].z + q4.w * k4[j].w;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                d += __shfl_xor_sync(0xFFFFFFFF, d, off);
            float s = d * scale;
            float nmx = fmaxf(mx, s);
            float corr = expf(mx - nmx);
            float e = expf(s - nmx);
            S = S * corr + e;
            mx = nmx;
            if (live) {
                float4 v4 = kv_ld4<KV>(v + (size_t)(base + j) * stride_kv + hk * hd + d0);
                oc.x = oc.x * corr + e * v4.x;
                oc.y = oc.y * corr + e * v4.y;
                oc.z = oc.z * corr + e * v4.z;
                oc.w = oc.w * corr + e * v4.w;
            }
        }
    }

    // No cross-lane reduction needed: each lane owns distinct dims.
    float* dst = partial + ((size_t)sp * nh + h) * pstr;
    if (lane_id == 0) {
        dst[0] = mx;
        dst[1] = S;
    }
    if (live) {
        *reinterpret_cast<float4*>(dst + 4 + d0) = oc; // 16B-aligned via pstr
    }
}

__global__ void gqa_attn_split_combine(
    const float* __restrict__ partial,
    float* __restrict__ o,
    int nh, int hd, int pstr
) {
    int h = blockIdx.y;
    int i = threadIdx.x; // hd threads
    if (i >= hd) return;
    float gmx = -INFINITY;
    for (int sp = 0; sp < ATTN_SPLITS; sp++)
        gmx = fmaxf(gmx, partial[((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < ATTN_SPLITS; sp++) {
        const float* p = partial + ((size_t)sp * nh + h) * pstr;
        float w = expf(p[0] - gmx);
        S += p[1] * w;
        acc += p[4 + i] * w;
    }
    o[h * hd + i] = (S > 0.0f) ? acc / S : 0.0f;
}

__global__ void gqa_attn_f32(
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    float* __restrict__ o,
    const int* positions,
    int nh, int nk, int hd,
    float scale, int nt
) {
    int t = blockIdx.x;
    int h = blockIdx.y;
    if (t >= nt || h >= nh) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int hk = h / gqa;
    int ne_q = nh * hd;
    int stride_kv = nk * hd;

    const float* qhead = q + t * ne_q + h * hd;
    float* ohead = o + t * ne_q + h * hd;

    int tid = threadIdx.x;
    int hd4 = hd / 4;
    const float4* q4 = reinterpret_cast<const float4*>(qhead);

    // Online softmax with persistent accumulators
    const int NE = 2;
    const int C = WARP * NE;

    float mx = -INFINITY;
    float S = 0.0f;
    float4 oc[32];
    #pragma unroll
    for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

    for (int batch = 0; batch < nkv; batch += C) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = batch + tid * NE;
        int kv1 = kv0 + 1;

        if (kv0 < nkv) {
            const float4* k4 = reinterpret_cast<const float4*>(k + kv0 * stride_kv + hk * hd);
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = k4[i];
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s0 = d * scale;
        }
        if (kv1 < nkv) {
            const float4* k4 = reinterpret_cast<const float4*>(k + kv1 * stride_kv + hk * hd);
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i], kvv = k4[i];
                d += qv.x * kvv.x + qv.y * kvv.y + qv.z * kvv.z + qv.w * kvv.w;
            }
            s1 = d * scale;
        }

        float batch_mx = fmaxf(s0, s1);
        // Warp-level max reduction
        for (int off = 16; off > 0; off >>= 1)
            batch_mx = fmaxf(batch_mx, __shfl_xor_sync(0xFFFFFFFF, batch_mx, off));
        float new_mx = fmaxf(mx, batch_mx);
        float corr = expf(mx - new_mx);

        float e0 = expf(s0 - new_mx);
        float e1 = expf(s1 - new_mx);

        #pragma unroll
        for (int i = 0; i < hd4; i++) oc[i].x *= corr, oc[i].y *= corr, oc[i].z *= corr, oc[i].w *= corr;
        S *= corr;

        if (kv0 < nkv) {
            const float4* v4 = reinterpret_cast<const float4*>(v + kv0 * stride_kv + hk * hd);
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = v4[i];
                oc[i].x += e0 * vv.x; oc[i].y += e0 * vv.y;
                oc[i].z += e0 * vv.z; oc[i].w += e0 * vv.w;
            }
        }
        if (kv1 < nkv) {
            const float4* v4 = reinterpret_cast<const float4*>(v + kv1 * stride_kv + hk * hd);
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 vv = v4[i];
                oc[i].x += e1 * vv.x; oc[i].y += e1 * vv.y;
                oc[i].z += e1 * vv.z; oc[i].w += e1 * vv.w;
            }
        }
        S += e0 + e1;
        mx = new_mx;
    }

    // Warp-level reduction of S and oc
    S = warp_reduce_sum(S);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        oc[i].x = warp_reduce_sum(oc[i].x);
        oc[i].y = warp_reduce_sum(oc[i].y);
        oc[i].z = warp_reduce_sum(oc[i].z);
        oc[i].w = warp_reduce_sum(oc[i].w);
    }

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    float4* o4 = reinterpret_cast<float4*>(ohead);
    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        o4[i].x = oc[i].x * inv;
        o4[i].y = oc[i].y * inv;
        o4[i].z = oc[i].z * inv;
        o4[i].w = oc[i].w * inv;
    }
}

// ====================================================================
// extern "C" launch wrappers (called from Rust via FFI)
// ====================================================================

extern "C" {

void launch_q4_0_q8_0_matmul(
    const uint8_t* weights, const uint8_t* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q4_0_q8_0_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q4_0_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q4_0_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q8_0_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q8_0_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q4_1_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q4_1_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q6_k_f32_matmul_padded(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 2, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q6_k_f32_matmul_padded<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_swiglu_f32_off(
    float* buf, int n, int off, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    swiglu_f32_off<<<grid, block, 0, stream>>>(buf, n, off);
}

void launch_gather_rows_f32(
    const float* src, const float* ids, float* out,
    int n, int nt, cudaStream_t stream
) {
    long long total = (long long)nt * n;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    gather_rows_f32<<<(int)grid, block, 0, stream>>>(src, ids, out, n, nt);
}

// Q5_1: one thread per 32-element block (24-byte blocks).
__global__ void embed_rows_q5_1(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    int nb = n_embd / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nb) return;
    int t = tid / nb, b = tid % nb;
    int id = __float_as_int(ids[t]);
    const uint8_t* blk = w + ((long long)id * nb + b) * 24;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint32_t qh = *reinterpret_cast<const uint32_t*>(blk + 4);
    const uint8_t* qs = blk + 8;
    float* o = out + (long long)t * n_embd + b * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float u_lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1);
        float u_hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1);
        o[j] = d * u_lo + m;
        o[j + 16] = d * u_hi + m;
    }
}

// Q5_K: one thread per 32-element sub-block (8 per 176-byte super-block);
// masked at n_embd for partial tail super-blocks (id % 32 == 0 layouts).
__global__ void embed_rows_q5_k(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    int nsp = (n_embd + 255) / 256;
    int nsub = (n_embd + 31) / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nsub) return;
    int t = tid / nsub, sidx = tid % nsub;
    int id = __float_as_int(ids[t]);
    int sp = sidx / 8, sub = sidx % 8;
    const uint8_t* blk = w + ((long long)id * nsp + sp) * 176;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    int ci = sub >> 1, hi = sub & 1;
    const uint8_t* q4 = blk + 48 + ci * 32;
    const uint8_t* qh = blk + 16;
    float ds = d * float(scb), dmm = dmin * float(mb);
    float* o = out + (long long)t * n_embd + sidx * 32;
    int rem = n_embd - sidx * 32;
    int lim = rem < 32 ? rem : 32;
    for (int l = 0; l < lim; l++) {
        float nib = hi ? float(q4[l] >> 4) : float(q4[l] & 0x0F);
        float wv = nib + 16.0f * float((qh[l] >> sub) & 1);
        o[l] = ds * wv - dmm;
    }
}

void launch_embed_rows(
    const uint8_t* w, const float* ids, float* out,
    int n_embd, int nt, int type_id, int block_stride, cudaStream_t stream
) {
    int block = 256;
    long long total = 0;
    if (type_id == 0 || type_id == 1) { // q8_0 / q4_0: one thread per 32-group
        total = (long long)nt * (n_embd / 32);
    } else if (type_id == 2) {          // q4_K: one per 32-element sub-block
        total = (long long)nt * (n_embd / 256) * 8;
    } else if (type_id == 4) {          // q5_1: one per 32-element block
        total = (long long)nt * (n_embd / 32);
    } else if (type_id == 5) {          // q5_K: one per 32-element sub-block
        total = (long long)nt * ((n_embd + 31) / 32);
    } else {                            // q6_K: one per 16-element sub-block
        total = (long long)nt * (n_embd / 256) * 16;
    }
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    switch (type_id) {
        case 0: embed_rows_q8_0<<<(int)grid, block, 0, stream>>>(w, ids, out, n_embd, nt); break;
        case 1: embed_rows_q4_0<<<(int)grid, block, 0, stream>>>(w, ids, out, n_embd, nt); break;
        case 2: embed_rows_q4_k<<<(int)grid, block, 0, stream>>>(w, ids, out, n_embd, nt); break;
        case 4: embed_rows_q5_1<<<(int)grid, block, 0, stream>>>(w, ids, out, n_embd, nt); break;
        case 5: embed_rows_q5_k<<<(int)grid, block, 0, stream>>>(w, ids, out, n_embd, nt); break;
        default: embed_rows_q6_k<<<(int)grid, block, 0, stream>>>(w, ids, out, n_embd, nt, block_stride); break;
    }
}

void launch_f32_f32_matmul(
    const float* w, const float* x, float* out,
    int od, int id, int nt, cudaStream_t stream
) {
    if (id % 8 == 0) {
        dim3 grid((od + 7) / 8, nt), block(64);
        f32_f32_matmul_vec<<<grid, block, 0, stream>>>(w, x, out, od, id, nt);
    } else {
        long long total = (long long)nt * od;
        int block = 256;
        long long grid = (total + block - 1) / block;
        if (grid > 2147483647LL) grid = 2147483647LL;
        f32_f32_matmul_scalar<<<(int)grid, block, 0, stream>>>(w, x, out, od, id, nt);
    }
}

void launch_q4_k_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 2, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q4_k_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_quantize_q8_0_pad40(
    const float* x, uint8_t* y, int dim, int nt, cudaStream_t stream
) {
    int nb = dim / 32;
    long long total = (long long)nt * nb;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    quantize_q8_0_pad40<<<(int)grid, block, 0, stream>>>(x, y, dim, nt);
}

void launch_q4_k_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    q4_k_q8_mmvq<<<grid, 256, 0, stream>>>(weights, acts8, output, od, id, nt);
}

void launch_q6_k_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, nt);
    q6_k_q8_mmvq<<<grid, 256, 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
}

void launch_q5_k_q8_mmvq(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    q5_k_q8_mmvq<<<grid, 256, 0, stream>>>(weights, acts8, output, od, id, nt);
}

// R2: weight-streaming rework (see the v2 kernel comments). Same signatures
// as the v1 launchers so dispatch can A/B via MINFER_MMVQ_V1=1.
void launch_q4_k_q8_mmvq_v2(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    q4_k_q8_mmvq_v2<<<grid, 256, 0, stream>>>(weights, acts8, output, od, id, nt);
}

void launch_q6_k_q8_mmvq_v2(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, int blk_stride, cudaStream_t stream
) {
    dim3 grid(od, nt);
    q6_k_q8_mmvq_v2<<<grid, 256, 0, stream>>>(weights, acts8, output, od, id, nt, blk_stride);
}

void launch_q5_k_q8_mmvq_v2(
    const uint8_t* weights, const uint8_t* acts8, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    dim3 grid(od, nt);
    q5_k_q8_mmvq_v2<<<grid, 256, 0, stream>>>(weights, acts8, output, od, id, nt);
}

void launch_q5_1_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q5_1_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q5_k_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q5_k_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_q6_k_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 2, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    q6_k_f32_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
}

void launch_quantize_q8_0(
    const float* x, uint8_t* y, int dim, int nt, cudaStream_t stream
) {
    int nb = dim / 32;
    int total = nt * nb;
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((total + block_sz - 1) / block_sz, 1, 1);
    quantize_q8_0<<<grid, block, 0, stream>>>(x, y, dim, nt);
}

void launch_rms_norm_f32(
    const float* x, const float* w, float* y,
    int d, float eps, int n, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1);
    dim3 grid(n, 1, 1);
    rms_norm_f32<<<grid, block, 0, stream>>>(x, w, y, d, eps, n);
}

void launch_add_bias_f32(
    float* y, const float* b, int d, int n, cudaStream_t stream
) {
    dim3 block(64, 1, 1); // 64 threads in x, grid y handles dim remainder
    dim3 grid(n, (d + 63) / 64, 1);
    add_bias_f32<<<grid, block, 0, stream>>>(y, b, d);
}

void launch_add_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    add_f32<<<grid, block, 0, stream>>>(x, y, z, n);
}

void launch_mul_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    mul_f32<<<grid, block, 0, stream>>>(x, y, z, n);
}

void launch_silu_f32(float* y, int n, cudaStream_t stream) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    silu_f32<<<grid, block, 0, stream>>>(y, n);
}

void launch_swiglu_f32(
    const float* gate, const float* up, float* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    swiglu_f32<<<grid, block, 0, stream>>>(gate, up, dst, n);
}

void launch_f32_bits_to_i32(
    const float* src, int* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    f32_bits_to_i32<<<grid, block, 0, stream>>>(src, dst, n);
}

void launch_rope_f32(
    float* x, int n_head, int n_dims, int nt,
    float freq_base, float freq_scale,
    const int* positions, cudaStream_t stream
) {
    int block_sz = 64; // threads per head dimension
    dim3 block(block_sz, 1, 1);
    dim3 grid(nt, n_head, 1);
    rope_f32<<<grid, block, 0, stream>>>(x, n_head, n_dims, nt, freq_base, freq_scale, positions);
}

void launch_store_kv_f32(
    const float* src, float* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 grid(nt, nkt, 1);
    store_kv_f32<<<grid, dim3(1, 1, 1), 0, stream>>>(src, dst, nkt, nt, positions);
}

void launch_store_kv_f16(
    const float* src, void* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 block(128, 1, 1);
    dim3 grid(nt, (nkt / 4 + 127) / 128, 1);
    store_kv_f16<<<grid, block, 0, stream>>>(src, (__half*)dst, nkt, nt, positions);
}

void launch_gqa_attn_f32_f16kv(
    const float* q, const void* k, const void* v, float* o,
    const int* positions,
    int n_head, int n_head_kv, int hd,
    float scale, int nt, cudaStream_t stream
) {
    int block_sz = 32; // one warp per (token, head)
    dim3 block(block_sz, 1, 1);
    dim3 grid(nt, n_head, 1);
    gqa_attn_f32_f16kv<<<grid, block, 0, stream>>>(
        q, (__half*)k, (__half*)v, o, positions,
        n_head, n_head_kv, hd, scale, nt
    );
}

void launch_gqa_attn_split_f16kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* positions,
    int n_head, int n_head_kv, int hd,
    float scale, int pstr, cudaStream_t stream
) {
    gqa_attn_split_partial<__half><<<dim3(ATTN_SPLITS, n_head), 32, 0, stream>>>(
        q, (const __half*)k, (const __half*)v, partial, positions,
        n_head, n_head_kv, hd, scale, pstr
    );
    gqa_attn_split_combine<<<dim3(1, n_head), hd, 0, stream>>>(
        partial, o, n_head, hd, pstr
    );
}

void launch_gqa_attn_split_f32kv(
    const float* q, const void* k, const void* v, float* o,
    float* partial, const int* positions,
    int n_head, int n_head_kv, int hd,
    float scale, int pstr, cudaStream_t stream
) {
    gqa_attn_split_partial<float><<<dim3(ATTN_SPLITS, n_head), 32, 0, stream>>>(
        q, (const float*)k, (const float*)v, partial, positions,
        n_head, n_head_kv, hd, scale, pstr
    );
    gqa_attn_split_combine<<<dim3(1, n_head), hd, 0, stream>>>(
        partial, o, n_head, hd, pstr
    );
}

void launch_gqa_attn_f32(
    const float* q, const float* k, const float* v, float* o,
    const int* positions, int nh, int nk, int hd,
    float scale, int nt, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1); // 32 threads per block (1 warp)
    dim3 grid(nt, nh, 1);
    gqa_attn_f32<<<grid, block, 0, stream>>>(q, k, v, o, positions, nh, nk, hd, scale, nt);
}

// ─── 8n: FA-style prefill attention (f16 KV) ────────────────────────────
// The legacy gqa_attn_f32_f16kv launches one block per (token, head): K is
// re-read per token per head (7B @2K: ~132 GB per layer) and the hd-wide
// accumulator lives in registers (float4 oc[32] = 128 regs → spills). It
// measured 176 ms per layer (76% of the whole 2K prefill). This kernel
// tiles the q dimension: one block per (64-token q tile, head), K/V tiles
// staged in shared memory, QK^T on tensor cores, online softmax with the
// O accumulator in shared memory. K traffic drops to ~0.8 GB per layer.
//
// Shared layout (dynamic, ~65 KB — opt-in via cudaFuncSetAttribute):
//   Qs [64*hd] f16   q tile (scale folded in, f16 for the tensor-core QK^T)
//   Ks [64*hd] f16   K tile           Vs [64*hd] f16  V tile
//   S  [64*64]       f32 scores, aliased as f16 probs after the row softmax
//   m/l/alpha [64] f32 per-row online-softmax state
// Both matmuls run on tensor cores: QK^T computes S into shared memory, and
// P·V accumulates into per-warp 16x16 f32 fragments that persist across KV
// tiles (scaled in place by the per-row alpha — see the loop body).
#define FA_TQ 64
#define FA_TKV 64
#define FA_PSTR (FA_TKV * 2) // probs row stride in halves (256B): probs row r aliases only Sf row r's first half, already read by the same thread — no cross-thread race

// P3: async K/V tile staging (16B cp.async chunks; rows beyond kv_end
// zero-filled). Overlapped with the previous tile's QK^T/softmax/P·V via
// double buffering — the synchronous staging version paid the full DRAM
// latency once per KV tile inside the block's serial k-loop.
__device__ __forceinline__ void fa_stage_kv_async(
    const __half* __restrict__ k, const __half* __restrict__ v,
    __half* Ks, __half* Vs, int kt, int kv_end,
    int hk, int hd, int stride_kv, int sstr, int tid
) {
#if __CUDA_ARCH__ >= 800
    for (int c = tid; c < FA_TKV * hd / 8; c += 256) {
        int r = (c * 8) / hd, d = (c * 8) % hd;
        int p = kt + r;
        bool full = p < kv_end;
        unsigned kd = (unsigned)__cvta_generic_to_shared(Ks + r * sstr + d);
        unsigned vd = (unsigned)__cvta_generic_to_shared(Vs + r * sstr + d);
        int sz = full ? 16 : 0;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(kd),
                     "l"(k + (size_t)p * stride_kv + hk * hd + d), "r"(sz));
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(vd),
                     "l"(v + (size_t)p * stride_kv + hk * hd + d), "r"(sz));
    }
#else
    // pre-sm80: synchronous staging (sm_75 stays a build target)
    const uint4 z4 = make_uint4(0, 0, 0, 0);
    for (int i = tid * 8; i < FA_TKV * hd; i += 2048) {
        int r = i / hd, d = i % hd;
        int p = kt + r;
        uint4 kk4, vv4;
        if (p < kv_end) {
            kk4 = *reinterpret_cast<const uint4*>(&k[(size_t)p * stride_kv + hk * hd + d]);
            vv4 = *reinterpret_cast<const uint4*>(&v[(size_t)p * stride_kv + hk * hd + d]);
        } else {
            kk4 = z4;
            vv4 = z4;
        }
        *reinterpret_cast<uint4*>(&Ks[r * sstr + d]) = kk4;
        *reinterpret_cast<uint4*>(&Vs[r * sstr + d]) = vv4;
    }
#endif
}

__global__ void fa_prefill_f16kv(
    const float* __restrict__ q,
    const __half* __restrict__ k,
    const __half* __restrict__ v,
    float* __restrict__ o,
    const int* __restrict__ positions,
    int nh, int nk, int hd,
    float scale,
    int nt
) {
    extern __shared__ __align__(256) uint8_t smem[];
    // Padded smem row stride: hd=128 halves = 256B ≡ 0 mod 32 banks makes
    // every wmma ldmatrix row land on the same bank group (8-way conflict
    // per load). +8 halves (272B) shifts each row by 4 banks.
    const int sstr = hd + 8;
    __half* Qs = reinterpret_cast<__half*>(smem);
    // Single-buffered K/V tiles: double buffering at the padded stride
    // (5 x 64 x 136 x 2B) exceeds GB10's 99KB/block smem cap and silently
    // falls back to the legacy attention kernel — padding wins more than
    // the staging overlap, so the overlap is the feature that goes.
    __half* Ks = Qs + FA_TQ * sstr;
    __half* Vs = Ks + FA_TKV * sstr;
    float* Sf = reinterpret_cast<float*>(Vs + FA_TKV * sstr);
    __half* Pf = reinterpret_cast<__half*>(Sf); // alias: probs after softmax
    float* msh = reinterpret_cast<float*>(Sf + FA_TQ * FA_TKV);
    float* lsh = msh + FA_TQ;
    float* alpha = lsh + FA_TQ;

    const int tq0 = blockIdx.x * FA_TQ;
    const int h = blockIdx.y;
    const int gqa = nh / nk;
    const int hk = h / gqa;
    const int ne_q = nh * hd;
    const int stride_kv = nk * hd;
    const int tid = threadIdx.x; // 256

    // load q tile (scale folded in) as f16
    for (int i = tid; i < FA_TQ * hd; i += 256) {
        int r = i / hd, d = i % hd;
        int t = tq0 + r;
        float qv = (t < nt) ? q[(size_t)t * ne_q + h * hd + d] * scale : 0.0f;
        Qs[r * sstr + d] = __float2half(qv);
    }
    if (tid < FA_TQ) {
        msh[tid] = -INFINITY;
        lsh[tid] = 0.0f;
    }
    // Output accumulator: P·V runs on tensor cores. Warp w = (wm, wn), with
    // wm = w>>1 (16-row chunk of the 64-row q tile) and wn = w&1 (64-dim
    // chunk of hd=128), keeps four 16x16 f32 fragments that persist across
    // KV tiles; the per-row online-softmax rescale (alpha) multiplies the
    // fragment lanes directly (m16n16 f32 accumulator layout: lane L holds
    // fragment rows L>>2 and (L>>2)+8). The previous scalar P·V loop read
    // Pf[arow*FA_PSTR + kk] with all 32 lanes landing on the SAME shared
    // bank (row stride 256B ≡ 0 mod 32 banks) — a 32-way conflict per kk —
    // and ran the whole P·V on CUDA cores; it dominated the kernel.
    using namespace nvcuda;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4];
#pragma unroll
    for (int nc = 0; nc < 4; nc++) wmma::fill_fragment(acc[nc], 0.0f);
    __syncthreads();

    const int last_t = min(nt - 1, tq0 + FA_TQ - 1);
    const int kv_end = positions[last_t] + 1;
    const bool tile_full = (tq0 + FA_TQ <= nt); // all 64 O rows in-bounds

    const int warp = tid >> 5; // 0..7
    const int wm = warp >> 1;  // q 16-block: 4
    const int wn = warp & 1;   // 64-dim chunk: 2
    const int lane = tid & 31;
    const int ar0 = wm * 16 + (lane >> 2); // alpha/l row for fragment row 0
    const int ar1 = ar0 + 8;               // fragment row 1

    for (int kt = 0; kt < kv_end; kt += FA_TKV) {
        // stage K/V tile (padded stride; synchronous — see the smem note)
        fa_stage_kv_async(k, v, Ks, Vs, kt, kv_end, hk, hd, stride_kv, sstr, tid);
#if __CUDA_ARCH__ >= 800
        // the copies are ASYNC: a bare __syncthreads does NOT order them —
        // commit + wait_group 0 makes the tile visible before QK^T reads it
        asm volatile("cp.async.commit_group;\n");
        asm volatile("cp.async.wait_group 0;\n");
#endif
        __syncthreads();

        // S = Q · K^T via wmma (reduction over hd)
        {
            int swm = warp >> 1;
            int swk = warp & 1;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2];
            wmma::fill_fragment(fc[0], 0.0f);
            wmma::fill_fragment(fc[1], 0.0f);
            for (int d = 0; d < hd; d += 16) {
                wmma::load_matrix_sync(fa, &Qs[swm * 16 * sstr + d], sstr);
                wmma::load_matrix_sync(fb[0], &Ks[swk * 32 * sstr + d], sstr);
                wmma::load_matrix_sync(fb[1], &Ks[(swk * 32 + 16) * sstr + d], sstr);
                wmma::mma_sync(fc[0], fa, fb[0], fc[0]);
                wmma::mma_sync(fc[1], fa, fb[1], fc[1]);
            }
            wmma::store_matrix_sync(&Sf[swm * 16 * FA_TKV + swk * 32], fc[0], FA_TKV, wmma::mem_row_major);
            wmma::store_matrix_sync(&Sf[swm * 16 * FA_TKV + swk * 32 + 16], fc[1], FA_TKV, wmma::mem_row_major);
        }
        __syncthreads();

        // online softmax — one WARP per row (8 warps cover the 64 rows, 8
        // rows per warp across k-tiles): each lane owns 2 of the 64 columns
        // and warp shuffles reduce max/sum, replacing the previous
        // thread-per-row serial scans (64-deep dependent chains on 2 of 8
        // warps — the kernel's dominant serial cost). Pf rows keep the 256B
        // stride: lane L's probs clobber exactly the two Sf floats lane L
        // itself read (per-lane program order, same invariant as before).
        {
            for (int rr = warp; rr < FA_TQ; rr += 8) {
                int t = tq0 + rr;
                int qpos = (t < nt) ? positions[t] : -1;
                int c0 = lane * 2, c1 = c0 + 1;
                bool v0 = kt + c0 <= qpos && kt + c0 < kv_end;
                bool v1 = kt + c1 <= qpos && kt + c1 < kv_end;
                // -INF seed: an all-masked row must keep the -INF state (a 0.0
                // seed would corrupt m/alpha for tiles with nothing valid).
                float s0 = v0 ? Sf[rr * FA_TKV + c0] : -INFINITY;
                float s1 = v1 ? Sf[rr * FA_TKV + c1] : -INFINITY;
                float m_new = fmaxf(s0, s1);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    m_new = fmaxf(m_new, __shfl_xor_sync(0xffffffffu, m_new, off));
                float m_old = msh[rr];
                float a = 1.0f;
                if (m_new == -INFINITY) {
                    // nothing valid in this tile: keep state, zero probs
                    Pf[rr * FA_PSTR + c0] = __float2half(0.0f);
                    Pf[rr * FA_PSTR + c1] = __float2half(0.0f);
                } else {
                    a = (m_old == -INFINITY) ? 0.0f : __expf(m_old - m_new);
                    float p0 = v0 ? __expf(s0 - m_new) : 0.0f;
                    float p1 = v1 ? __expf(s1 - m_new) : 0.0f;
                    Pf[rr * FA_PSTR + c0] = __float2half(p0);
                    Pf[rr * FA_PSTR + c1] = __float2half(p1);
                    float sum = p0 + p1;
#pragma unroll
                    for (int off = 16; off > 0; off >>= 1)
                        sum += __shfl_xor_sync(0xffffffffu, sum, off);
                    if (lane == 0) {
                        lsh[rr] = lsh[rr] * a + sum;
                        msh[rr] = m_new;
                    }
                }
                if (lane == 0) alpha[rr] = a;
            }
        }
        __syncthreads();

        // acc = acc * alpha + P · V on tensor cores. Masked-out probs are
        // already zero in Pf, and K/V rows beyond kv_end are zero-staged, so
        // the full 64-wide k-reduction is safe. The per-lane row scale uses
        // the documented m16n16 f32 accumulator layout (x[0,1,4,5] -> row
        // lane>>2, x[2,3,6,7] -> row (lane>>2)+8); the parity test locks it.
        float a0 = alpha[ar0];
        float a1 = alpha[ar1];
#pragma unroll
        for (int nc = 0; nc < 4; nc++) {
            acc[nc].x[0] *= a0; acc[nc].x[1] *= a0;
            acc[nc].x[2] *= a1; acc[nc].x[3] *= a1;
            acc[nc].x[4] *= a0; acc[nc].x[5] *= a0;
            acc[nc].x[6] *= a1; acc[nc].x[7] *= a1;
        }
        {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vb;
#pragma unroll
            for (int kk0 = 0; kk0 < FA_TKV; kk0 += 16) {
                wmma::load_matrix_sync(pa, &Pf[wm * 16 * FA_PSTR + kk0], FA_PSTR);
#pragma unroll
                for (int nc = 0; nc < 4; nc++) {
                    wmma::load_matrix_sync(vb, &Vs[kk0 * sstr + wn * 64 + nc * 16], sstr);
                    wmma::mma_sync(acc[nc], pa, vb, acc[nc]);
                }
            }
        }
        __syncthreads();
    }

    // write out: acc / l — rows with l == 0 stay 0 (fully masked). Full
    // tiles store the fragments straight to global o (every row/col offset
    // is 16-float aligned); the tail tile stages through shared memory so
    // rows t >= nt can be skipped.
    if (tile_full) {
        float i0 = (lsh[ar0] > 0.0f) ? 1.0f / lsh[ar0] : 0.0f;
        float i1 = (lsh[ar1] > 0.0f) ? 1.0f / lsh[ar1] : 0.0f;
#pragma unroll
        for (int nc = 0; nc < 4; nc++) {
            acc[nc].x[0] *= i0; acc[nc].x[1] *= i0;
            acc[nc].x[2] *= i1; acc[nc].x[3] *= i1;
            acc[nc].x[4] *= i0; acc[nc].x[5] *= i0;
            acc[nc].x[6] *= i1; acc[nc].x[7] *= i1;
            wmma::store_matrix_sync(
                &o[(size_t)(tq0 + wm * 16) * ne_q + h * hd + wn * 64 + nc * 16],
                acc[nc], ne_q, wmma::mem_row_major);
        }
    } else {
        // Qs/Ks/Vs regions are free after the KV loop: 48 KB contiguous
        // staging for the 64x128 f32 O tile.
        float* stage = reinterpret_cast<float*>(smem);
        float i0 = (lsh[ar0] > 0.0f) ? 1.0f / lsh[ar0] : 0.0f;
        float i1 = (lsh[ar1] > 0.0f) ? 1.0f / lsh[ar1] : 0.0f;
#pragma unroll
        for (int nc = 0; nc < 4; nc++) {
            acc[nc].x[0] *= i0; acc[nc].x[1] *= i0;
            acc[nc].x[2] *= i1; acc[nc].x[3] *= i1;
            acc[nc].x[4] *= i0; acc[nc].x[5] *= i0;
            acc[nc].x[6] *= i1; acc[nc].x[7] *= i1;
            wmma::store_matrix_sync(
                &stage[(wm * 16) * hd + wn * 64 + nc * 16],
                acc[nc], hd, wmma::mem_row_major);
        }
        __syncthreads();
        for (int idx = tid; idx < FA_TQ * hd; idx += 256) {
            int r = idx / hd, c = idx % hd;
            int t = tq0 + r;
            if (t < nt) o[(size_t)t * ne_q + h * hd + c] = stage[idx];
        }
    }
}

int launch_fa_prefill_f16kv(
    const float* q, const __half* k, const __half* v, float* o,
    const int* positions, int nh, int nk, int hd, float scale, int nt,
    cudaStream_t stream
) {
    size_t smem = (size_t)3 * FA_TQ * (hd + 8) * 2 + (size_t)FA_TQ * FA_TKV * 4 + 3 * FA_TQ * 4; // Qs/Ks/Vs padded (+8 halves per row)
    static size_t attr_smem = 0;
    if (smem > attr_smem) {
        cudaError_t e = cudaFuncSetAttribute(
            fa_prefill_f16kv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) {
            cudaGetLastError(); // clear the error so it cannot poison the stream
            // NOT silent: the caller falls back to the legacy per-token
            // attention kernel (~50x slower) — this must be visible.
            static int warned = 0;
            if (!warned) {
                warned = 1;
                fprintf(stderr,
                        "minfer/cuda: fa_prefill_f16kv smem %zu B exceeds the device "
                        "limit; falling back to the legacy attention kernel\n",
                        smem);
            }
            return -1;
        }
        attr_smem = smem;
    }
    dim3 grid((nt + FA_TQ - 1) / FA_TQ, nh, 1);
    fa_prefill_f16kv<<<grid, 256, smem, stream>>>(q, k, v, o, positions, nh, nk, hd, scale, nt);
    return 0;
}

// ─── 8m: prefill dequant-to-f16 + wmma HGEMM ────────────────────────────
// Prefill (nt >= 16) routes quantized matmuls through ONE tiled
// tensor-core GEMM instead of the decode-shaped kernels whose
// grid.y = nt re-streamed the whole weight matrix once per token
// (7B q4_k_m @2K: 30.7 tok/s vs llama.cpp MMQ 3401). Weights are
// dequantized to f16 once per call into a scratch buffer, activations
// converted to f16, then C[nt, od] = A[nt, id] · B[od, id]^T via
// 16x16x16 wmma with f32 accumulation. Gated on id % 32 == 0 (block
// math + 16B-aligned uint4 tile loads).

// type_id mapping for launch_dequant_f16 (Rust side passes it):
// 0=q8_0 1=q4_0 2=q4_1 3=q5_0 4=q5_1 5=q4_K 6=q5_K 7=q6_K

__global__ void dequant_q8_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 34;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const int8_t* q = (const int8_t*)(blk + 2);
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int i = 0; i < 32; i++) o[i] = __float2half(d * float(q[i]));
}

__global__ void dequant_q4_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 18;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    const uint8_t* q = blk + 2;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    // minfer Q4_0 stores round(v/d) + 8 (same -8 offset as the matmuls).
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i] = __float2half(d * (float(q[i] & 0x0F) - 8.0f));
        o[i + 16] = __float2half(d * (float(q[i] >> 4) - 8.0f));
    }
}

__global__ void dequant_q4_1_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 20;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    const uint8_t* q = blk + 4;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        o[i] = __float2half(d * float(q[i] & 0x0F) + m);
        o[i + 16] = __float2half(d * float(q[i] >> 4) + m);
    }
}

__global__ void dequant_q5_0_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 22;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // 22-byte blocks are only 2-byte aligned: assemble qh from two u16
    // loads — a u32 load at blk+2 misaligns for even g
    // (cudaErrorMisalignedAddress 716; latent until 8p's bitparity test
    // exercised Q5_0 prefill GEMM for the first time).
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    const uint8_t* qs = blk + 6;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1) - 16.0f;
        float hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1) - 16.0f;
        o[j] = __float2half(d * lo);
        o[j + 16] = __float2half(d * hi);
    }
}

__global__ void dequant_q5_1_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nb = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nb) return;
    int row = (int)(g / nb);
    const uint8_t* blk = w + g * 24;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint32_t qh = *reinterpret_cast<const uint32_t*>(blk + 4);
    const uint8_t* qs = blk + 8;
    __half* o = out + (long long)row * id + (int)(g % nb) * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1);
        float hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1);
        o[j] = __float2half(d * lo + m);
        o[j + 16] = __float2half(d * hi + m);
    }
}

__global__ void dequant_q4_k_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nsub = id / 32; // 8 sub-blocks per 256 super-block
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nsub) return;
    int row = (int)(g / nsub), s = (int)(g % nsub);
    int sp = s / 8, sub = s % 8;
    const uint8_t* blk = w + ((long long)row * (id / 256) + sp) * Q4KB;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    int j = sub / 2, half = sub % 2;
    const uint8_t* q = blk + 16 + j * 32;
    __half* o = out + (long long)row * id + s * 32;
    float ds = d * float(scb), dmm = dmin * float(mb);
    #pragma unroll
    for (int l = 0; l < 32; l++) {
        float nib = half ? float(q[l] >> 4) : float(q[l] & 0x0F);
        o[l] = __float2half(ds * nib - dmm);
    }
}

__global__ void dequant_q5_k_f16(
    const uint8_t* __restrict__ w, __half* __restrict__ out, int od, int id
) {
    int nsub = id / 32;
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long)od * nsub) return;
    int row = (int)(g / nsub), sidx = (int)(g % nsub);
    int sp = sidx / 8, sub = sidx % 8;
    const uint8_t* blk = w + ((long long)row * (id / 256) + sp) * Q5KB;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    int ci = sub >> 1, hi = sub & 1;
    const uint8_t* q4 = blk + 48 + ci * 32;
    const uint8_t* qh = blk + 16;
    __half* o = out + (long long)row * id + sidx * 32;
    float ds = d * float(scb), dmm = dmin * float(mb);
    #pragma unroll
    for (int l = 0; l < 32; l++) {
        float nib = hi ? float(q4[l] >> 4) : float(q4[l] & 0x0F);
        float wv = nib + 16.0f * float((qh[l] >> sub) & 1);
        o[l] = __float2half(ds * wv - dmm);
    }
}

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
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int n = sub / 8, rem = sub % 8, tt = rem / 2, gq = rem % 2;
    int ql_off = n * 64 + (tt % 2) * 32 + gq * 16;
    int qh_off = n * 32 + gq * 16;
    int sc_idx = n * 8 + tt * 2 + gq;
    __half* o = out + (long long)row * id + sp * 256 + n * 128 + tt * 32 + gq * 16;
    float dsc = d * float(sc[sc_idx]);
    #pragma unroll
    for (int r = 0; r < 16; r++) {
        int nib = (tt < 2) ? (ql[ql_off + r] & 0x0F) : (ql[ql_off + r] >> 4);
        int q2 = (qh[qh_off + r] >> (tt * 2)) & 3;
        o[r] = __float2half(dsc * float((nib | (q2 << 4)) - 32));
    }
}

// f32 activations -> f16 (one kernel, elementwise).
__global__ void convert_f32_f16_kernel(
    const float* __restrict__ x, __half* __restrict__ out, long long n
) {
    // P1: 8 elements per thread (2x float4 -> 4x half2) instead of one
    // scalar element — 8x fewer transactions on the same traffic.
    long long base = ((long long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (base + 7 < n) {
        float4 a = *reinterpret_cast<const float4*>(x + base);
        float4 b = *reinterpret_cast<const float4*>(x + base + 4);
        __half2* o = reinterpret_cast<__half2*>(out + base);
        o[0] = __floats2half2_rn(a.x, a.y);
        o[1] = __floats2half2_rn(a.z, a.w);
        o[2] = __floats2half2_rn(b.x, b.y);
        o[3] = __floats2half2_rn(b.z, b.w);
    } else if (base < n) {
        for (long long i = base; i < n; i++) out[i] = __float2half(x[i]);
    }
}

// 8m②: cp.async global→shared staging (sm_80+). The synchronous load
// stalled every warp on the L2 round trip each 32-k step (~31 TFLOPS
// measured); async copies overlap the k+32 tile fetch with the k compute.
__device__ __forceinline__ void gemm_load_tile_sync(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* As, __half* Bs,
    int n0, int m0, int k0, int nt, int od, int id
) {
    int r = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id;
    int n = n0 + r;
    if (n < nt && k_ok) {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) =
            *reinterpret_cast<const uint4*>(A + (long long)n * id + k0 + c4);
    } else {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
    int m = m0 + r;
    if (m < od && k_ok) {
        *reinterpret_cast<uint4*>(Bs + r * 32 + c4) =
            *reinterpret_cast<const uint4*>(B + (long long)m * id + k0 + c4);
    } else {
        *reinterpret_cast<uint4*>(Bs + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
}

#if __CUDA_ARCH__ >= 800
__device__ __forceinline__ void gemm_cp16(__half* smem_dst, const __half* gsrc, bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 16 : 0; // src-size 0 => zero-fill the 16B chunk
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}
__device__ __forceinline__ void gemm_cp_commit() { asm volatile("cp.async.commit_group;\n"); }
__device__ __forceinline__ void gemm_cp_wait1() { asm volatile("cp.async.wait_group 1;\n"); }
__device__ __forceinline__ void gemm_cp_wait0() { asm volatile("cp.async.wait_group 0;\n"); }

__device__ __forceinline__ void gemm_load_tile_async(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* As, __half* Bs,
    int n0, int m0, int k0, int nt, int od, int id
) {
    // 256 threads: 64 rows x 4 aligned 16B chunks (8 halves each).
    int r = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id; // id % 8 == 0 gate keeps chunks inside the row
    int n = n0 + r;
    gemm_cp16(As + r * 32 + c4, A + (long long)n * id + k0 + c4, n < nt && k_ok);
    int m = m0 + r;
    gemm_cp16(Bs + r * 32 + c4, B + (long long)m * id + k0 + c4, m < od && k_ok);
}

// P2: B-panel loader for TM-row od tiles (TM > 64 needs a second pass of
// 64 rows; A always stages TN = 64 rows).
__device__ __forceinline__ void gemm_load_b_async(
    const __half* __restrict__ B, __half* Bs,
    int m0, int k0, int od, int id, int tm
) {
    int r0 = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id;
    for (int rep = 0; rep < tm / 64; rep++) {
        int r = r0 + rep * 64;
        int m = m0 + r;
        gemm_cp16(Bs + r * 32 + c4, B + (long long)m * id + k0 + c4, m < od && k_ok);
    }
}

// synchronous B loader for pre-sm80 builds lives AFTER the sm_80 guard
// (sm_75 is still a build target and compiles the fallback paths).
#endif // __CUDA_ARCH__ >= 800

__device__ __forceinline__ void gemm_load_b_sync(
    const __half* __restrict__ B, __half* Bs,
    int m0, int k0, int od, int id, int tm
) {
    int r0 = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    bool k_ok = k0 + c4 < id;
    for (int rep = 0; rep < tm / 64; rep++) {
        int r = r0 + rep * 64;
        int m = m0 + r;
        if (m < od && k_ok) {
            *reinterpret_cast<uint4*>(Bs + r * 32 + c4) =
                *reinterpret_cast<const uint4*>(B + (long long)m * id + k0 + c4);
        } else {
            *reinterpret_cast<uint4*>(Bs + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
        }
    }
}

// sync B loader lives outside the sm_80 guard: the pre-sm80 fallback paths
// of gemm_f16_nt_kernel_t reference it (sm_75 is still a build target).

// A template cannot have C linkage: pause the extern "C" block around the
// templated GEMM.
} // extern "C"

// AF32 A staging, mirror scheme: cp.async the F32 k-tile into a smem
// mirror (16B = 4 f32 chunks; async again — the v1 synchronous global
// loads stalled every k-tile and measured -8%), then convert
// smem->smem f32->f16 right before compute. Requires id % 8 == 0.
template <int KS>
__device__ __forceinline__ void gemm_mirror_a32(
    float* Am, int bbuf, const float* A32, int TN,
    int n0, int k0, int nt, int id, int tid
) {
#if __CUDA_ARCH__ >= 800
    for (int c = tid; c < TN * KS / 4; c += blockDim.x) {
        int r = (c * 4) / KS, d = (c * 4) % KS;
        int n = n0 + r;
        gemm_cp16(reinterpret_cast<__half*>(Am + bbuf * TN * KS + r * KS + d),
                  reinterpret_cast<const __half*>(A32 + (long long)n * id + k0 + d),
                  n < nt && k0 + d < id);
    }
#endif // pre-sm80 callers use the inline synchronous staging instead
}

// P4: stage the A (TN rows) and B (TM rows) k-tiles [k0, k0+KS) into the
// double-buffered dynamic-smem regions. Chunk-linear mapping: chunk c
// covers 8 consecutive halves, r = c*8/KS, d = c*8%KS (KS % 8 == 0).
// AF32: A arrives as f32 activations and converts on stage (P6), so the
// separate convert_f32_f16 pass disappears.
template <int TM, int KS, int TN, bool AF32 = false>
__device__ __forceinline__ void gemm_stage_ab(
    const __half* __restrict__ A, const __half* __restrict__ B,
    __half* As, __half* Bs, float* Am, int bbuf,
    int n0, int m0, int k0, int nt, int od, int id, int tid
) {
#if __CUDA_ARCH__ >= 800
    if (AF32) {
        gemm_mirror_a32<KS>(Am, bbuf, reinterpret_cast<const float*>(A), TN,
                            n0, k0, nt, id, tid);
    } else {
        for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
            int r = (c * 8) / KS, d = (c * 8) % KS;
            int n = n0 + r;
            gemm_cp16(As + bbuf * TN * KS + r * KS + d,
                      A + (long long)n * id + k0 + d, n < nt && k0 + d < id);
        }
    }
    for (int c = tid; c < TM * KS / 8; c += blockDim.x) {
        int r = (c * 8) / KS, d = (c * 8) % KS;
        int m = m0 + r;
        gemm_cp16(Bs + bbuf * TM * KS + r * KS + d,
                  B + (long long)m * id + k0 + d, m < od && k0 + d < id);
    }
#endif // pre-sm80 callers use the inline synchronous staging instead
}

// C[nt, od] = A[nt, id] · B[od, id]^T. 64 x TM output tiles (TM = 64
// baseline, 128 halves the B-panel re-reads through L2 and the per-k-step
// barrier count), k-step 32, double-buffered shared staging, 8 warps (each
// owns 32 nt rows x TM/4 od cols as 2 x TM/64 f32 fragment pairs). f32
// accumulation. Tails: nt/od masked at store, k-tail zero-filled (id % 8
// == 0 keeps the uint4 chunk loads aligned).
template <int TM, int KS, bool AF32 = false>
__global__ void gemm_f16_nt_kernel_t(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int nt, int od, int id
) {
    using namespace nvcuda;
    constexpr int TN = 64;
    constexpr int ODC = TM / 64;  // od 16-col fragments per warp row-half
    constexpr int KHC = KS / 32;  // k-half PAIRS per staged tile (each loop iteration consumes fa[0]+fa[2] = 2 k-halves)
    // dynamic smem: TM=128 + KS=64 needs 56KB — over the 48KB static cap
    extern __shared__ __align__(16) uint8_t smem_raw[];
    __half* As = reinterpret_cast<__half*>(smem_raw); // 2 x TN*KS halves
    // AF32 adds a 2 x TN*KS f32 mirror right after As (cp.async target)
    float* Am = reinterpret_cast<float*>(As + 2 * TN * KS);
    __half* Bs = reinterpret_cast<__half*>(
        smem_raw + 2 * TN * KS * 2 + (AF32 ? 2 * TN * KS * 4 : 0)); // 2 x TM*KS
    float* Cs = reinterpret_cast<float*>(Bs + 2 * TM * KS); // NW x 256 f32

    const int tid = threadIdx.x;
    const int NW = blockDim.x >> 5;   // warps: 8 for TM<=128, 16 for TM=256
    int warp = tid >> 5;
    int wm = warp >> 1;               // od chunk of this warp (TM/(NW/2) cols)
    int wn = warp & 1;                // nt sub-tile: 2 x 32 rows
    // blockIdx.x = nt tile, blockIdx.y = od tile: consecutive blocks share
    // the same od-tile's B panel (64 rows x id f16, ~0.5MB) in L2, so the
    // f16 weight matrix streams from DRAM ~once instead of nt/64 times.
    int m0 = blockIdx.y * TM;
    int n0 = blockIdx.x * TN;
    const int ob = wm * (TM / (NW >> 1)); // od row base of this warp's chunk

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2][ODC];
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int oc = 0; oc < ODC; oc++) wmma::fill_fragment(fc[j][oc], 0.0f);

    int buf = 0;
#if __CUDA_ARCH__ >= 800
    // stage A (64 rows) + B (TM rows) for k = 0
    gemm_stage_ab<TM, KS, TN, AF32>(A, B, As, Bs, Am, 0, n0, m0, 0, nt, od, id, tid);
    gemm_cp_commit();
#else
    {
        if (AF32) { // pre-sm80: synchronous global->smem convert (no cp.async)
            for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                int r = (c * 8) / KS, d = (c * 8) % KS;
                int n = n0 + r;
                uint4 h = make_uint4(0u, 0u, 0u, 0u);
                if (n < nt && d < id) {
                    const float4 f0 = *reinterpret_cast<const float4*>(
                        reinterpret_cast<const float*>(A) + (long long)n * id + d);
                    const float4 f1 = *reinterpret_cast<const float4*>(
                        reinterpret_cast<const float*>(A) + (long long)n * id + d + 4);
                    __half2 p0 = __floats2half2_rn(f0.x, f0.y);
                    __half2 p1 = __floats2half2_rn(f0.z, f0.w);
                    __half2 p2 = __floats2half2_rn(f1.x, f1.y);
                    __half2 p3 = __floats2half2_rn(f1.z, f1.w);
                    h = make_uint4(*reinterpret_cast<unsigned*>(&p0),
                                   *reinterpret_cast<unsigned*>(&p1),
                                   *reinterpret_cast<unsigned*>(&p2),
                                   *reinterpret_cast<unsigned*>(&p3));
                }
                *reinterpret_cast<uint4*>(As + r * KS + d) = h;
            }
        } else {
            for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                int r = (c * 8) / KS, d = (c * 8) % KS;
                int n = n0 + r;
                if (n < nt && d < id) {
                    *reinterpret_cast<uint4*>(As + r * KS + d) =
                        *reinterpret_cast<const uint4*>(A + (long long)n * id + d);
                } else {
                    *reinterpret_cast<uint4*>(As + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        for (int c = tid; c < TM * KS / 8; c += blockDim.x) {
            int r = (c * 8) / KS, d = (c * 8) % KS;
            int m = m0 + r;
            if (m < od && d < id) {
                *reinterpret_cast<uint4*>(Bs + r * KS + d) =
                    *reinterpret_cast<const uint4*>(B + (long long)m * id + d);
            } else {
                *reinterpret_cast<uint4*>(Bs + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
            }
        }
        __syncthreads();
    }
#endif
    for (int k = 0; k < id; k += KS, buf ^= 1) {
#if __CUDA_ARCH__ >= 800
        if (k + KS < id) {
            gemm_stage_ab<TM, KS, TN, AF32>(A, B, As, Bs, Am, buf ^ 1, n0,
                                            m0, k + KS, nt, od, id, tid);
            gemm_cp_commit();
            // wait until the CURRENT tile landed (one group may stay in flight)
            gemm_cp_wait1();
        } else {
            gemm_cp_wait0();
        }
        __syncthreads();
#else
        if (k + KS < id) {
            if (AF32) { // pre-sm80: synchronous convert into the next buffer
                for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                    int r = (c * 8) / KS, d = (c * 8) % KS;
                    int n = n0 + r;
                    uint4 h = make_uint4(0u, 0u, 0u, 0u);
                    if (n < nt && k + KS + d < id) {
                        const float4 f0 = *reinterpret_cast<const float4*>(
                            reinterpret_cast<const float*>(A) + (long long)n * id + k + KS + d);
                        const float4 f1 = *reinterpret_cast<const float4*>(
                            reinterpret_cast<const float*>(A) + (long long)n * id + k + KS + d + 4);
                        __half2 p0 = __floats2half2_rn(f0.x, f0.y);
                        __half2 p1 = __floats2half2_rn(f0.z, f0.w);
                        __half2 p2 = __floats2half2_rn(f1.x, f1.y);
                        __half2 p3 = __floats2half2_rn(f1.z, f1.w);
                        h = make_uint4(*reinterpret_cast<unsigned*>(&p0),
                                       *reinterpret_cast<unsigned*>(&p1),
                                       *reinterpret_cast<unsigned*>(&p2),
                                       *reinterpret_cast<unsigned*>(&p3));
                    }
                    *reinterpret_cast<uint4*>(As + (buf ^ 1) * TN * KS + r * KS + d) = h;
                }
            } else {
                for (int c = tid; c < TN * KS / 8; c += blockDim.x) {
                    int r = (c * 8) / KS, d = (c * 8) % KS;
                    int n = n0 + r;
                    if (n < nt && k + KS + d < id) {
                        *reinterpret_cast<uint4*>(As + (buf ^ 1) * TN * KS + r * KS + d) =
                            *reinterpret_cast<const uint4*>(A + (long long)n * id + k + KS + d);
                    } else {
                        *reinterpret_cast<uint4*>(As + (buf ^ 1) * TN * KS + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
                    }
                }
            }
            for (int c = tid; c < TM * KS / 8; c += blockDim.x) {
                int r = (c * 8) / KS, d = (c * 8) % KS;
                int m = m0 + r;
                if (m < od && k + KS + d < id) {
                    *reinterpret_cast<uint4*>(Bs + (buf ^ 1) * TM * KS + r * KS + d) =
                        *reinterpret_cast<const uint4*>(B + (long long)m * id + k + KS + d);
                } else {
                    *reinterpret_cast<uint4*>(Bs + (buf ^ 1) * TM * KS + r * KS + d) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
        }
        __syncthreads();
#endif
        if (AF32) {
            // mirror[buf] landed with the wait above: convert smem->smem
            for (int c = tid; c < TN * KS / 4; c += blockDim.x) {
                int r = (c * 4) / KS, d = (c * 4) % KS;
                float4 f = *reinterpret_cast<float4*>(
                    &Am[buf * TN * KS + r * KS + d]);
                __half2 p0 = __floats2half2_rn(f.x, f.y);
                __half2 p1 = __floats2half2_rn(f.z, f.w);
                *reinterpret_cast<__half2*>(As + buf * TN * KS + r * KS + d) = p0;
                *reinterpret_cast<__half2*>(As + buf * TN * KS + r * KS + d + 2) = p1;
            }
            __syncthreads();
        }
        // fa[n-half][k-half]; fb[k-half] per od chunk. Both k halves of each
        // 32-slice must accumulate (the v1 bug: only the first 16 k's were
        // multiplied); fb's k offset is +16 ELEMENTS (one k-half), not +16
        // rows.
#pragma unroll
        for (int kh = 0; kh < KHC; kh++) {
            wmma::load_matrix_sync(fa[0], &As[buf * TN * KS + wn * 32 * KS + kh * 32], KS);
            wmma::load_matrix_sync(fa[1], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32], KS);
            wmma::load_matrix_sync(fa[2], &As[buf * TN * KS + wn * 32 * KS + kh * 32 + 16], KS);
            wmma::load_matrix_sync(fa[3], &As[buf * TN * KS + (wn * 32 + 16) * KS + kh * 32 + 16], KS);
#pragma unroll
            for (int oc = 0; oc < ODC; oc++) {
                wmma::load_matrix_sync(fb[0], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32], KS);
                wmma::load_matrix_sync(fb[1], &Bs[buf * TM * KS + (ob + oc * 16) * KS + kh * 32 + 16], KS);
                wmma::mma_sync(fc[0][oc], fa[0], fb[0], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[1], fb[0], fc[1][oc]);
                wmma::mma_sync(fc[0][oc], fa[2], fb[1], fc[0][oc]);
                wmma::mma_sync(fc[1][oc], fa[3], fb[1], fc[1][oc]);
            }
        }
        __syncthreads();
    }

    int lane = threadIdx.x & 31;
#pragma unroll
    for (int j = 0; j < 2; j++) {
#pragma unroll
        for (int oc = 0; oc < ODC; oc++) {
            wmma::store_matrix_sync(Cs + warp * 256, fc[j][oc], 16, wmma::mem_row_major);
            int nb = n0 + wn * 32 + j * 16, mb = m0 + ob + oc * 16;
            for (int e = lane; e < 256; e += 32) {
                int n = nb + (e >> 4), m = mb + (e & 15);
                if (n < nt && m < od)
                    C[(long long)n * od + m] = Cs[warp * 256 + (e >> 4) * 16 + (e & 15)];
            }
        }
    }
}

// one-time cudaFuncSetAttribute for dynamic smem above the 48KB static cap
template <typename K>
static void gemm_smem_optin(K kernel, size_t bytes) {
    static size_t done = 0;
    if (bytes > 48 * 1024 && bytes > done) {
        cudaError_t e = cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
        if (e != cudaSuccess) {
            cudaGetLastError(); // do not poison the stream; the launch errors below
            fprintf(stderr, "minfer/cuda: gemm smem optin %zu B failed: %s\n", bytes,
                    cudaGetErrorString(e));
            return;
        }
        done = bytes;
    }
}

// Set every prefill-GEMM instantiation's dynamic-smem attribute EAGERLY:
// the AF32 variants need >48KB, and an attribute set lazily during CUDA
// graph capture fails silently and poisons the first captured launch.
extern "C" void gemm_prefill_smem_init() {
    const int tms[3] = {64, 128, 256};
    for (int i = 0; i < 3; i++) {
        int tm = tms[i], threads = (tm >= 256) ? 512 : 256, nw = threads >> 5;
        for (int ks = 32; ks <= 64; ks += 32) {
            size_t base = (size_t)(2 * 64 * ks + 2 * tm * ks) * 2 + (size_t)nw * 256 * 4;
            for (int af32 = 0; af32 <= 1; af32++) {
                size_t bytes = base + (af32 ? (size_t)2 * 64 * ks * 4 : 0);
                if (bytes > 48 * 1024) {
#define OPTIN(TM2, KS2, AF2)                                                       \
    cudaFuncSetAttribute(gemm_f16_nt_kernel_t<TM2, KS2, AF2>,                      \
                         cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes)
                    if (tm == 64 && ks == 32 && af32) OPTIN(64, 32, true);
                    else if (tm == 64 && ks == 32) OPTIN(64, 32, false);
                    else if (tm == 64 && ks == 64 && af32) OPTIN(64, 64, true);
                    else if (tm == 64 && ks == 64) OPTIN(64, 64, false);
                    else if (tm == 128 && ks == 32 && af32) OPTIN(128, 32, true);
                    else if (tm == 128 && ks == 32) OPTIN(128, 32, false);
                    else if (tm == 128 && ks == 64 && af32) OPTIN(128, 64, true);
                    else if (tm == 128 && ks == 64) OPTIN(128, 64, false);
                    else if (tm == 256 && ks == 32 && af32) OPTIN(256, 32, true);
                    else if (tm == 256 && ks == 32) OPTIN(256, 32, false);
                    else if (tm == 256 && ks == 64 && af32) OPTIN(256, 64, true);
                    else if (tm == 256 && ks == 64) OPTIN(256, 64, false);
                    cudaGetLastError(); // oversize requests (TM=256+KS=64) are fine to skip
#undef OPTIN
                }
            }
        }
    }
}

extern "C" {

void launch_dequant_f16(
    int type_id, const uint8_t* w, __half* out,
    int od, int id, int block_stride, cudaStream_t stream
) {
    int block = 256;
    long long total;
    switch (type_id) {
        case 7: total = (long long)od * (id / 16); break;            // q6_K
        default: total = (long long)od * (id / 32); break;           // all others
    }
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    switch (type_id) {
        case 0: dequant_q8_0_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        case 1: dequant_q4_0_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        case 2: dequant_q4_1_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        case 3: dequant_q5_0_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        case 4: dequant_q5_1_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        case 5: dequant_q4_k_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        case 6: dequant_q5_k_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id); break;
        default: dequant_q6_k_f16<<<(int)grid, block, 0, stream>>>(w, out, od, id, block_stride); break;
    }
}

void launch_convert_f16(
    const float* x, __half* out, long long n, cudaStream_t stream
) {
    long long grid = (n / 8 + 255) / 256;
    if (grid > 2147483647LL) grid = 2147483647LL;
    convert_f32_f16_kernel<<<(int)grid, 256, 0, stream>>>(x, out, n);
}

void launch_gemm_f16(
    const __half* a, const __half* b, float* c,
    int nt, int od, int id, cudaStream_t stream, bool af32
) {
    // P2: od-tile width (128 default = halved B re-reads; MINFER_GEMM_TM=64
    // reverts to the 8m② baseline for A/B).
    static int tm = -1;
    if (tm < 0) {
        const char* e = getenv("MINFER_GEMM_TM");
        int v = e ? atoi(e) : 128;
        tm = (v <= 64) ? 64 : (v >= 256) ? 256 : 128;
    }
    // KS = staged k-width per tile. KS=64 halves the barriers per FLOP but
    // measured -38% (56KB dynamic smem halves resident blocks on GB10);
    // KS=32 (8m2 baseline) stays the default. MINFER_GEMM_K64=1 re-tries 64.
    static int ks = -1;
    if (ks < 0) {
        const char* e = getenv("MINFER_GEMM_K64");
        ks = (e && atoi(e)) ? 64 : 32;
    }
    const size_t dyn_smem = (size_t)(2 * 64 * ks + 2 * tm * ks) * 2 + 8 * 256 * 4;
#define GEMM_LAUNCH(TM_, KS_)                                                      \
    do {                                                                           \
        dim3 grid((nt + 63) / 64, (od + TM_ - 1) / TM_);                           \
        if (af32) {                                                                \
            gemm_smem_optin(gemm_f16_nt_kernel_t<TM_, KS_, true>, dyn_smem);       \
            gemm_f16_nt_kernel_t<TM_, KS_, true>                                   \
                <<<grid, 256, dyn_smem, stream>>>(a, b, c, nt, od, id);            \
        } else {                                                                   \
            gemm_smem_optin(gemm_f16_nt_kernel_t<TM_, KS_, false>, dyn_smem);      \
            gemm_f16_nt_kernel_t<TM_, KS_, false>                                  \
                <<<grid, 256, dyn_smem, stream>>>(a, b, c, nt, od, id);            \
        }                                                                          \
    } while (0)
    if (tm >= 128) {
        if (ks >= 64)
            GEMM_LAUNCH(128, 64);
        else
            GEMM_LAUNCH(128, 32);
    } else {
        if (ks >= 64)
            GEMM_LAUNCH(64, 64);
        else
            GEMM_LAUNCH(64, 32);
    }
#undef GEMM_LAUNCH
}

// P6: A arrives as f32 activations; converts inside the kernel on stage.
void launch_gemm_f32a(
    const float* a, const __half* b, float* c,
    int nt, int od, int id, cudaStream_t stream
) {
    launch_gemm_f16(reinterpret_cast<const __half*>(a), b, c, nt, od, id, stream, true);
    // P6 debug: surface the async launch error (A32-in-capture still fails)
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess)
        fprintf(stderr, "minfer/cuda: af32 gemm launch failed: %s\n",
                cudaGetErrorString(e));
    if (getenv("MINFER_A32_SYNC")) {
        e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess)
            fprintf(stderr, "minfer/cuda: af32 gemm ASYNC FAULT nt=%d od=%d id=%d: %s\n",
                    nt, od, id, cudaGetErrorString(e));
    }
}


// ─── 8p: fused dequant-in-GEMM ────────────────────────────────────────────
// 8m's two-pass path dequantized W to an f16 scratch (288 ms on 7B @2K)
// before every GEMM and streamed that f16 matrix back from DRAM. This
// kernel keeps the identical 64x64 wmma tile structure but dequantizes B
// tiles in-register from the RAW quantized bytes (a Q4_K 64-row k-tile
// reads ~1.5 KB of quantized data + headers instead of 4 KB of f16 — and
// the separate dequant pass disappears). A still comes from the f16
// activation scratch (launch_convert_f16 stays; a fused f32→f16 A load is
// future work). Each bqa_* mirrors its dequant_*_f16 kernel's element math
// and __float2half rounding EXACTLY, so fused and two-pass results are
// bit-identical (asserted in cuda_prefill_fused_b_bitparity).
//
// Requires id % 256 == 0 (host-side gate): the 8-element runs never
// straddle a 32-sub-block, and K-quant super-block boundaries stay aligned.

__device__ __forceinline__ void bqa_q8_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 34) + (long long)(e0 >> 5) * 34;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++)
        dst[l] = __float2half(d * float((int8_t)blk[2 + b + l]));
}

__device__ __forceinline__ void bqa_q4_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 18) + (long long)(e0 >> 5) * 18;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    int b = e0 & 31;
    // minfer Q4_0 stores round(v/d) + 8 (same -8 offset as the matmuls);
    // element e uses byte blk[2 + (e & 15)]: lo nibble for e < 16.
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[2 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        dst[l] = __float2half(d * (nib - 8.0f));
    }
}

__device__ __forceinline__ void bqa_q4_1(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 20) + (long long)(e0 >> 5) * 20;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[4 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        dst[l] = __float2half(d * nib + m);
    }
}

__device__ __forceinline__ void bqa_q5_0(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 22) + (long long)(e0 >> 5) * 22;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // 22-byte blocks are only 2-byte aligned: assemble qh from two u16
    // loads (a plain u32 load at blk+2 misaligns for even block indices —
    // cudaErrorMisalignedAddress, caught by cuda_prefill_fused_b_bitparity).
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    int b = e0 & 31;
    // element e: nibble qs[e & 15] (lo for e < 16), high bit qh >> e.
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[6 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        float v = nib + 16.0f * float((qh >> e) & 1) - 16.0f;
        dst[l] = __float2half(d * v);
    }
}

__device__ __forceinline__ void bqa_q5_1(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 5) * 24) + (long long)(e0 >> 5) * 24;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6) << 16);
    int b = e0 & 31;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int e = b + l;
        uint8_t byte = blk[8 + (e & 15)];
        float nib = (e < 16) ? float(byte & 0x0F) : float(byte >> 4);
        float v = nib + 16.0f * float((qh >> e) & 1);
        dst[l] = __float2half(d * v + m);
    }
}

__device__ __forceinline__ void bqa_q4_k(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 8) * 144) + (long long)(e0 >> 8) * 144;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    int eb = e0 & 255;
    int sub = eb >> 5, l0 = eb & 31;
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    float ds = d * float(scb), dmm = dmin * float(mb);
    int half = sub & 1;
    const uint8_t* q = blk + 16 + (sub >> 1) * 32;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int bi = l0 + l;
        float nib = half ? float(q[bi] >> 4) : float(q[bi] & 0x0F);
        dst[l] = __float2half(ds * nib - dmm);
    }
}

__device__ __forceinline__ void bqa_q5_k(
    const uint8_t* w, int row, int id, int e0, __half* dst
) {
    const uint8_t* blk = w + (long long)row * ((id >> 8) * 176) + (long long)(e0 >> 8) * 176;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float dmin = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    int eb = e0 & 255;
    int sub = eb >> 5, l0 = eb & 31;
    uint8_t scb, mb;
    get_scale_min_k4(sub, blk + 4, &scb, &mb);
    float ds = d * float(scb), dmm = dmin * float(mb);
    int ci = sub >> 1, hi = sub & 1;
    const uint8_t* q4 = blk + 48 + ci * 32;
    const uint8_t* qh = blk + 16;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int bi = l0 + l;
        float nib = hi ? float(q4[bi] >> 4) : float(q4[bi] & 0x0F);
        float wv = nib + 16.0f * float((qh[bi] >> sub) & 1);
        dst[l] = __float2half(ds * wv - dmm);
    }
}

__device__ __forceinline__ void bqa_q6_k(
    const uint8_t* w, int row, int id, int e0, int bstride, __half* dst
) {
    // bstride: 210 raw or 224 padded (7e② repack keeps intra-block layout).
    const uint8_t* blk = w + (long long)row * ((id >> 8) * bstride) + (long long)(e0 >> 8) * bstride;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk + 208));
    const uint8_t* ql = blk;
    const uint8_t* qh = blk + 128;
    const int8_t* sc = (const int8_t*)(blk + 192);
    int eb = e0 & 255;
    int n = eb >> 7, rem = eb & 127, tt = rem >> 5, gq = (rem >> 4) & 1;
    int ql_off = n * 64 + (tt & 1) * 32 + gq * 16;
    int qh_off = n * 32 + gq * 16;
    float dsc = d * float(sc[n * 8 + tt * 2 + gq]);
    int r0 = eb & 15;
    #pragma unroll
    for (int l = 0; l < 8; l++) {
        int r = r0 + l;
        int nib = (tt < 2) ? (ql[ql_off + r] & 0x0F) : (ql[ql_off + r] >> 4);
        int q2 = (qh[qh_off + r] >> (tt * 2)) & 3;
        dst[l] = __float2half(dsc * float((nib | (q2 << 4)) - 32));
    }
}

// One 64-row B tile k-slice: thread = (row, 8-element chunk); the B side
// dequantizes raw bytes in-register, the A side vector-loads f16.
__device__ __forceinline__ void gemm_qb_load_tile(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    __half* As, __half* Bs,
    int n0, int m0, int k0, int nt, int od, int id,
    int type_id, int q6_stride
) {
    int r = threadIdx.x >> 2, c4 = (threadIdx.x & 3) * 8;
    int n = n0 + r;
    if (n < nt) {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) =
            *reinterpret_cast<const uint4*>(A + (long long)n * id + k0 + c4);
    } else {
        *reinterpret_cast<uint4*>(As + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
    int m = m0 + r;
    if (m < od) {
        int e0 = k0 + c4;
        __half* dst = Bs + r * 32 + c4;
        switch (type_id) {
            case 0: bqa_q8_0(W, m, id, e0, dst); break;
            case 1: bqa_q4_0(W, m, id, e0, dst); break;
            case 2: bqa_q4_1(W, m, id, e0, dst); break;
            case 3: bqa_q5_0(W, m, id, e0, dst); break;
            case 4: bqa_q5_1(W, m, id, e0, dst); break;
            case 5: bqa_q4_k(W, m, id, e0, dst); break;
            case 6: bqa_q5_k(W, m, id, e0, dst); break;
            default: bqa_q6_k(W, m, id, e0, q6_stride, dst); break;
        }
    } else {
        *reinterpret_cast<uint4*>(Bs + r * 32 + c4) = make_uint4(0u, 0u, 0u, 0u);
    }
}

// C[nt, od] = A[nt, id] · dequant(W[od, id])^T — same tile/warp structure,
// fragment layout, and store masking as gemm_f16_nt_kernel; only the B
// staging differs (raw bytes dequantized in-register). Synchronous
// double-buffered loads: cp.async cannot convert or dequantize.
__global__ void gemm_qb_nt_kernel(
    const __half* __restrict__ A, const uint8_t* __restrict__ W,
    float* __restrict__ C, int nt, int od, int id,
    int type_id, int q6_stride
) {
    using namespace nvcuda;
    __shared__ __half As[2][64 * 32];
    __shared__ __half Bs[2][64 * 32];
    __shared__ float Cs[8][16 * 16];

    int warp = threadIdx.x >> 5;
    int wm = warp >> 1;
    int wn = warp & 1;
    int m0 = blockIdx.y * 64;
    int n0 = blockIdx.x * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2];
    wmma::fill_fragment(fc[0], 0.0f);
    wmma::fill_fragment(fc[1], 0.0f);

    int buf = 0;
    gemm_qb_load_tile(A, W, As[0], Bs[0], n0, m0, 0, nt, od, id, type_id, q6_stride);
    __syncthreads();
    for (int k = 0; k < id; k += 32, buf ^= 1) {
        if (k + 32 < id)
            gemm_qb_load_tile(A, W, As[buf ^ 1], Bs[buf ^ 1], n0, m0, k + 32,
                              nt, od, id, type_id, q6_stride);
        // fa: [n-block 0/1] x [k-half 0/1]; fb: [k-half 0/1] (same as 8m).
        wmma::load_matrix_sync(fa[0], &As[buf][wn * 32 * 32], 32);
        wmma::load_matrix_sync(fa[1], &As[buf][(wn * 32 + 16) * 32], 32);
        wmma::load_matrix_sync(fa[2], &As[buf][wn * 32 * 32 + 16], 32);
        wmma::load_matrix_sync(fa[3], &As[buf][(wn * 32 + 16) * 32 + 16], 32);
        wmma::load_matrix_sync(fb[0], &Bs[buf][wm * 16 * 32], 32);
        wmma::load_matrix_sync(fb[1], &Bs[buf][wm * 16 * 32 + 16], 32);
        wmma::mma_sync(fc[0], fa[0], fb[0], fc[0]);
        wmma::mma_sync(fc[1], fa[1], fb[0], fc[1]);
        wmma::mma_sync(fc[0], fa[2], fb[1], fc[0]);
        wmma::mma_sync(fc[1], fa[3], fb[1], fc[1]);
        __syncthreads();
    }

    int lane = threadIdx.x & 31;
    #pragma unroll
    for (int j = 0; j < 2; j++) {
        wmma::store_matrix_sync(Cs[warp], fc[j], 16, wmma::mem_row_major);
        int nb = n0 + wn * 32 + j * 16, mb = m0 + wm * 16;
        for (int e = lane; e < 256; e += 32) {
            int n = nb + (e >> 4), m = mb + (e & 15);
            if (n < nt && m < od)
                C[(long long)n * od + m] = Cs[warp][(e >> 4) * 16 + (e & 15)];
        }
    }
}

void launch_gemm_qb_nt(
    const __half* a, const uint8_t* w, float* c,
    int nt, int od, int id, int type_id, int q6_stride, cudaStream_t stream
) {
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    gemm_qb_nt_kernel<<<grid, 256, 0, stream>>>(a, w, c, nt, od, id, type_id, q6_stride);
}

} // extern "C"


// ─── R1: int8 MMQ prefill GEMM ────────────────────────────────────────────
// llama.cpp's MMQ math structure on minfer's 8p tile skeleton. Activations
// are quantized to q8_0 once per launch (pad40 blocks; the kernel writes the
// per-block int sum into the 4 slack bytes at offset 36), weights stay RAW —
// no f16 dequant pass, no w16 cache. A tiled mma.m16n8k32 (s8) GEMM
// accumulates one 32-k int chunk per step; the int C fragment is rescaled
// per (token, row, k-block) with the block-scale products and, for
// min-carrying types, a rank-1 offset term (weight min × activation block
// sum) — exactly llama.cpp's per-block correction, so results sit within
// f32 rounding of the CPU q8_0-activation dot path.
//
// K-quants keep their nibbles UNSIGNED in the int GEMM and carry the min
// term separately (llama.cpp's unpack_scales trick — the nibble grid is
// 0..15/0..31, so the "integer part" is non-negative and the scale/offset
// pair (d·s, −dmin·m) is applied per 32-k sub-block). q6_K's scales live on
// 16-element sub-blocks: each 32-k chunk spans two of them, so the chunk
// runs as TWO m16n8k16 mmas (low/high k halves) with separate int
// accumulators, rescaled by (d·sc0, d·sc1) — k32 staging/sync cadence, full
// mma throughput (a per-16 chunk loop measured 4× slower end-to-end: q6_K
// carries the 7B model's ffn_down + lm_head).
//
// Fragment layouts follow the PTX ISA mma.m16n8k32/m16n8k16 row.col
// documentation (A: row = groupID, k = tig·4 quads; B: col = groupID;
// C: row = groupID + 8·(l>>1), col = tig·2 + (l&1) — the C mapping is the
// production-proven llama.cpp tile_C get_i/get_j).
//
// Block tile 64 tokens × 64 rows, warp tile 32 tokens × 16 rows (the 8m/8p
// wm/wn warp mapping — consecutive blocks share one od-tile's weight panel
// in L2). Double-buffered shared tiles, synchronous staging (cp.async can't
// unpack quantized bytes): per k-chunk the A side moves 40B/token of q8 and
// the B side 16–34B/row of raw weight bytes — 2–4× less DRAM traffic than
// the f16 cache path, from the tensor-core int8 pipe. B staging uses two
// threads per row (one 16B half each) so consecutive threads read
// consecutive bytes instead of striding across rows.

#define MMQ_BI 64   // tokens (i) per block tile
#define MMQ_BJ 64   // od rows (j) per block tile
#define MMQ_WS 9    // shared words per tile row: 8 data + 1 bank-conflict pad
#define MMQ_KD 8    // 32-k chunks staged per buffer (256-k, llama.cpp-style);
                // ~94KB smem/block, llama.cpp ITER_K-style; measured faster than
                // KD=4 (2 blocks/SM) under load — revisit on a quiet GPU

// A-side q8 staging: thread t handles token row t>>2 (words t&3 and 4+(t&3)).
__device__ __forceinline__ void mmq_stage_a(
    const uint8_t* __restrict__ q8x, int i0, int nt, int nb32, int kb,
    int* qa, float* da, int* sa
) {
    int r = threadIdx.x >> 2, q = threadIdx.x & 3;
    int tok = i0 + r;
    int w0 = 0, w1 = 0; float d = 0.0f; int s = 0;
    if (tok < nt) {
        const uint8_t* blk = q8x + ((size_t)tok * nb32 + kb) * 40;
        w0 = *(const uint32_t*)(blk + 4 + 4 * q);
        w1 = *(const uint32_t*)(blk + 20 + 4 * q);
        if (q == 0) {
            d = h2f(*(const uint16_t*)blk);
            s = (int)*(const uint32_t*)(blk + 36);
        }
    }
    qa[r * MMQ_WS + q] = w0;
    qa[r * MMQ_WS + 4 + q] = w1;
    if (q == 0) { da[r] = d; sa[r] = s; }
}

// B-side raw-weight staging: 128 threads → 2 per weight row (thread half
// t&1 covers words 4·half..4·half+3 of the 32-k chunk); q8_0 uses 4 threads
// per row. Headers (scale/offset) come from the half-0 threads. Type math
// mirrors the bqa_* dequantizers / MMVQ kernels exactly.
template <int TYPE>
__device__ __forceinline__ void mmq_stage_b(
    const uint8_t* __restrict__ W, int j0, int od, int nb32, int c, int bstride,
    int* qb, float* ds, float* dm, float* ds1
) {
    int t = threadIdx.x;
    int r = t >> 1, half = t & 1;          // 2 threads per row (q8_0: 4)
    if constexpr (TYPE == 0) {
        r = t >> 2, half = t & 3;
    } else {
        if (t >= 128) return;              // 2 threads × 64 rows
    }
    int j = j0 + r;
    if (j >= od) {
        if constexpr (TYPE == 0) {
            #pragma unroll
            for (int w = half; w < 8; w += 4) qb[r * MMQ_WS + w] = 0;
            if (half == 0) { ds[r] = 0.0f; }
        } else {
            #pragma unroll
            for (int w = 4 * half; w < 4 * half + 4; w++) qb[r * MMQ_WS + w] = 0;
            if (half == 0) { ds[r] = 0.0f; dm[r] = 0.0f; if (TYPE == 7) ds1[r] = 0.0f; }
        }
        return;
    }
    if constexpr (TYPE == 0) {          // q8_0: 34B blocks, signed payload
        const uint8_t* blk = W + (size_t)j * (nb32 * 34) + (size_t)c * 34;
        if (half == 0) ds[r] = h2f(*(const uint16_t*)blk);
        #pragma unroll
        for (int w = half; w < 8; w += 4)
            qb[r * MMQ_WS + w] =
                (int)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * w)
                | ((int)*reinterpret_cast<const uint16_t*>(blk + 4 + 4 * w) << 16);
    } else if constexpr (TYPE == 1) {   // q4_0: value = d·(nib − 8)
        const uint8_t* blk = W + (size_t)j * (nb32 * 18) + (size_t)c * 18;
        if (half == 0) ds[r] = h2f(*(const uint16_t*)blk);
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            qb[r * MMQ_WS + w] = __vsubss4((int)nib, 0x08080808);
        }
    } else if constexpr (TYPE == 2) {   // q4_1: value = d·nib + m
        const uint8_t* blk = W + (size_t)j * (nb32 * 20) + (size_t)c * 20;
        if (half == 0) {
            ds[r] = h2f(*(const uint16_t*)blk);
            dm[r] = h2f(*(const uint16_t*)(blk + 2));
        }
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            qb[r * MMQ_WS + w] = (int)nib;
        }
    } else if constexpr (TYPE == 3) {   // q5_0: value = d·(nib + 16·bit − 16)
        const uint8_t* blk = W + (size_t)j * (nb32 * 22) + (size_t)c * 22;
        uint32_t qh;
        if (half == 0) {
            ds[r] = h2f(*(const uint16_t*)blk);
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2);
        } else {
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4);
        }
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 8 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            uint32_t hi = ((((qh >> (4 * wb + 0)) & 1u) << 0) | (((qh >> (4 * wb + 1)) & 1u) << 8)
                        | (((qh >> (4 * wb + 2)) & 1u) << 16) | (((qh >> (4 * wb + 3)) & 1u) << 24)) << 4;
            qb[r * MMQ_WS + w] = __vsubss4((int)(nib | hi), 0x10101010);
        }
    } else if constexpr (TYPE == 4) {   // q5_1: value = d·(nib + 16·bit) + m
        const uint8_t* blk = W + (size_t)j * (nb32 * 24) + (size_t)c * 24;
        uint32_t qh;
        if (half == 0) {
            ds[r] = h2f(*(const uint16_t*)blk);
            dm[r] = h2f(*(const uint16_t*)(blk + 2));
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4);
        } else {
            qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 6);
        }
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t N = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 8 + 4 * wb)
                      | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 10 + 4 * wb) << 16);
            uint32_t nib = (w >= 4) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            uint32_t hi = ((((qh >> (4 * wb + 0)) & 1u) << 0) | (((qh >> (4 * wb + 1)) & 1u) << 8)
                        | (((qh >> (4 * wb + 2)) & 1u) << 16) | (((qh >> (4 * wb + 3)) & 1u) << 24)) << 4;
            qb[r * MMQ_WS + w] = (int)(nib | hi);
        }
    } else if constexpr (TYPE == 5) {   // q4_K: value = d·s·nib − dmin·m (nib unsigned)
        int sb = c >> 3, s = c & 7;
        const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * 144) + (size_t)sb * 144;
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            uint32_t N = *(const uint32_t*)(blk + 16 + (s >> 1) * 32 + 4 * w);
            uint32_t nib = (s & 1) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            qb[r * MMQ_WS + w] = (int)nib;
        }
        if (half == 0) {
            uint8_t sc, m;
            get_scale_min_k4(s, blk + 4, &sc, &m);
            ds[r] = h2f(*(const uint16_t*)blk) * (float)sc;
            dm[r] = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);
        }
    } else if constexpr (TYPE == 6) {   // q5_K: q5 high-bit plane over the q4_K layout
        int sb = c >> 3, s = c & 7;
        const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * 176) + (size_t)sb * 176;
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            uint32_t N = *(const uint32_t*)(blk + 48 + (s >> 1) * 32 + 4 * w);
            uint32_t nib = (s & 1) ? ((N >> 4) & 0x0F0F0F0Fu) : (N & 0x0F0F0F0Fu);
            uint32_t QH = *(const uint32_t*)(blk + 16 + 4 * w);
            uint32_t hi = ((QH >> s) & 0x01010101u) << 4;
            qb[r * MMQ_WS + w] = (int)(nib | hi);
        }
        if (half == 0) {
            uint8_t sc, m;
            get_scale_min_k4(s, blk + 4, &sc, &m);
            ds[r] = h2f(*(const uint16_t*)blk) * (float)sc;
            dm[r] = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);
        }
    } else {                            // q6_K: value = d·sc·(q6 − 32), 16-elem sub-scales
        // one 32-k chunk = two subs: s0 = 2c (words 0..3), s1 = 2c+1 (words 4..7);
        // s wraps at 16 into the next super-block (sb = c>>3)
        int sb = c >> 3;
        const uint8_t* blk = W + (size_t)j * ((nb32 >> 3) * bstride) + (size_t)sb * bstride;
        if (half == 0) {
            float d = h2f(*(const uint16_t*)(blk + 208));
            ds[r]  = d * (float)(int8_t)blk[192 + (2 * c) % 16];
            ds1[r] = d * (float)(int8_t)blk[192 + (2 * c + 1) % 16];
        }
        int s = (2 * c + half) % 16;
        int chunk = s >> 3, g = (s >> 1) & 3, is = s & 1;
        const uint8_t* ql = blk + chunk * 64 + (g & 1) * 32 + is * 16;
        const uint8_t* qh = blk + 128 + chunk * 32 + is * 16;
        #pragma unroll
        for (int w = 4 * half; w < 4 * half + 4; w++) {
            int wb = w & 3;
            uint32_t QL = (uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * wb)
                       | ((uint32_t)*reinterpret_cast<const uint16_t*>(ql + 4 * wb + 2) << 16);
            uint32_t QH = (uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * wb)
                       | ((uint32_t)*reinterpret_cast<const uint16_t*>(qh + 4 * wb + 2) << 16);
            uint32_t nib = (g < 2) ? (QL & 0x0F0F0F0Fu) : ((QL >> 4) & 0x0F0F0F0Fu);
            uint32_t hi = ((QH >> (2 * g)) & 0x03030303u) << 4;
            qb[r * MMQ_WS + w] = __vsubss4((int)(nib | hi), 0x20202020);
        }
    }
}

__device__ __forceinline__ void mmq_mma_k32(int* d, const int* a, const int* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void mmq_mma_k16(int* d, const int* a, int b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
}

// TYPE: 0 q8_0, 1 q4_0, 2 q4_1, 3 q5_0, 4 q5_1, 5 q4_K, 6 q5_K, 7 q6_K.
// KSPLIT: 1 = one m16n8k32 per chunk (types 0-6); 2 = two m16n8k16 with
// separate int accumulators (q6_K's per-16 sub-scales). HAS_OFF = the
// min-carrying types (rank-1 offset term in the rescale).
template <int TYPE, int KSPLIT, bool HAS_OFF>
__global__ void __launch_bounds__(256) mmq_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id, int bstride
) {
#if __CUDA_ARCH__ >= 800
    // mma.m16n8k32 (s8) needs sm_80+; sm_75 keeps the f16 w16-cache path.
    // Dynamic shared (llama.cpp-style 256-k staging depth): staging ONE
    // 32-k chunk per sync exposed the full global-load latency every
    // chunk — 2.5 TMAC/s on GB10. 8 chunks per buffer amortize it 8×.
    extern __shared__ int mmq_sh[];
    int* qa = mmq_sh;                                   // [2][KD][BI*WS]
    int* qb = qa + 2 * MMQ_KD * (MMQ_BI * MMQ_WS);      // [2][KD][BJ*WS]
    int* ssa = qb + 2 * MMQ_KD * (MMQ_BJ * MMQ_WS);     // [2][KD][BI]
    float* sda = reinterpret_cast<float*>(ssa + 2 * MMQ_KD * MMQ_BI);
    float* sds = sda + 2 * MMQ_KD * MMQ_BI;             // [2][KD][BJ]
    float* sds1 = sds + 2 * MMQ_KD * MMQ_BI;            // q6_K: second 16-sub
    float* sdm = sds1 + 2 * MMQ_KD * MMQ_BI;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;   // same warp mapping as 8m/8p
    const int i0 = blockIdx.x * MMQ_BI;
    const int j0 = blockIdx.y * MMQ_BJ;
    const int i0w = wn * 32;
    const int j0w = wm * 16;
    const int nb32 = id >> 5;
    const int nchunk = id >> 5;
    const int nktile = (nchunk + MMQ_KD - 1) / MMQ_KD;

    float sum[16] = {0.0f};   // [nh][h][l]: 2 B-frags × 2 A-frags × 4 C regs

#define MMQ_STAGE_TILE(kt, b)                                                   \
    for (int kd = 0; kd < MMQ_KD; kd++) {                                       \
        const int c = (kt) * MMQ_KD + kd;                                       \
        if (c < nchunk) {                                                       \
            mmq_stage_a(q8x, i0, nt, nb32, c,                                   \
                        qa + ((b) * MMQ_KD + kd) * (MMQ_BI * MMQ_WS),           \
                        sda + ((b) * MMQ_KD + kd) * MMQ_BI,                     \
                        ssa + ((b) * MMQ_KD + kd) * MMQ_BI);                    \
            mmq_stage_b<TYPE>(W, j0, od, nb32, c, bstride,                      \
                              qb + ((b) * MMQ_KD + kd) * (MMQ_BJ * MMQ_WS),     \
                              sds + ((b) * MMQ_KD + kd) * MMQ_BJ,               \
                              sdm + ((b) * MMQ_KD + kd) * MMQ_BJ,               \
                              sds1 + ((b) * MMQ_KD + kd) * MMQ_BJ);             \
        }                                                                       \
    }

    MMQ_STAGE_TILE(0, 0)
    __syncthreads();

    int buf = 0;
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        if (kt + 1 < nktile) MMQ_STAGE_TILE(kt + 1, buf ^ 1)

        for (int kd = 0; kd < MMQ_KD; kd++) {
            const int c = kt * MMQ_KD + kd;
            if (c >= nchunk) break;
            const int* qat = qa + (buf * MMQ_KD + kd) * (MMQ_BI * MMQ_WS);
            const int* qbt = qb + (buf * MMQ_KD + kd) * (MMQ_BJ * MMQ_WS);
            const float* sdat = sda + (buf * MMQ_KD + kd) * MMQ_BI;
            const int* ssat = ssa + (buf * MMQ_KD + kd) * MMQ_BI;
            const float* sdst = sds + (buf * MMQ_KD + kd) * MMQ_BJ;
            const float* sds1t = sds1 + (buf * MMQ_KD + kd) * MMQ_BJ;
            const float* sdmt = sdm + (buf * MMQ_KD + kd) * MMQ_BJ;

            // A fragments: 2× 16-token halves; B fragments: 2× 8-row halves.
            // Fragment word layout is the k32 one for every type; KSPLIT=2
            // only splits the mma into low/high k halves (words 0..3 / 4..7).
            int a[2][4], b[2][2];
            int clow[2][2][4], chigh[2][2][4];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r0 = (i0w + h * 16 + (lane >> 2)) * MMQ_WS + (lane & 3);
                const int r1 = (i0w + h * 16 + 8 + (lane >> 2)) * MMQ_WS + (lane & 3);
                a[h][0] = qat[r0];
                a[h][1] = qat[r1];
                a[h][2] = qat[r0 + 4];
                a[h][3] = qat[r1 + 4];
            }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const int rb = (j0w + nh * 8 + (lane >> 2)) * MMQ_WS + (lane & 3);
                b[nh][0] = qbt[rb];
                b[nh][1] = qbt[rb + 4];
            }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) { clow[nh][h][l] = 0; chigh[nh][h][l] = 0; }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++) {
                    if constexpr (KSPLIT == 1) {
                        mmq_mma_k32(clow[nh][h], a[h], b[nh]);
                    } else {
                        mmq_mma_k16(clow[nh][h], a[h], b[nh][0]);
                        mmq_mma_k16(chigh[nh][h], a[h] + 2, b[nh][1]);
                    }
                }

            // per-(token, row, k-block) rescale: value = ds·int (+ dm·sa), all × da
            const float da_q[4] = {
                sdat[i0w +  lane / 4], sdat[i0w + 8 + lane / 4],
                sdat[i0w + 16 + lane / 4], sdat[i0w + 24 + lane / 4],
            };
            const float sa_q[4] = {
                (float)ssat[i0w +  lane / 4], (float)ssat[i0w + 8 + lane / 4],
                (float)ssat[i0w + 16 + lane / 4], (float)ssat[i0w + 24 + lane / 4],
            };
            float dsv[2][8], dsv1[2][8], dmv[2][8];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    dsv[nh][jj] = sdst[j0w + nh * 8 + jj];
                    dsv1[nh][jj] = sds1t[j0w + nh * 8 + jj];
                    dmv[nh][jj] = sdmt[j0w + nh * 8 + jj];
                }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        // C element l: row = (l>>1)*8 + lane/4, col = (lane&3)*2 + (l&1)
                        const float da = da_q[h * 2 + (l >> 1)];
                        const float sa = sa_q[h * 2 + (l >> 1)];
                        const int jj = (lane & 3) * 2 + (l & 1);
                        const int idx = nh * 8 + h * 4 + l;
                        if constexpr (KSPLIT == 1) {
                            sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
                            if (HAS_OFF) sum[idx] += da * dmv[nh][jj] * sa;
                        } else {
                            sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
                            sum[idx] += da * dsv1[nh][jj] * (float)chigh[nh][h][l];
                        }
                    }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + i0w + h * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[nh * 8 + h * 4 + l];
            }
#endif // __CUDA_ARCH__ >= 800
}


// ─── P6: raw-byte MMQ (llama.cpp structure) — q4_K first ─────────────
// smem holds RAW quant bytes; staging is pure cp.async; dequant happens
// in registers inside the mma loop. Only the per-(row, chunk) scales are
// pre-computed at staging (shared by the whole warp; the C-fragment
// rescale needs all 8 fragment rows, and computing them per lane would
// be 4x redundant work in the hot loop).
//   A: pad40 chunk = d(2B) qs(32B @4) ssum(4B @36) — consumed as-is
//      (word w of the fragment = the raw int8 lane group k 4w..4w+3).
//   B: Q4KB=144B super-block per row per 256-k; nibbles unpacked at mma.
// Requires whole super-blocks: nb32 % 8 == 0 (launcher guard).
__device__ __forceinline__ void mmq_cp8(uint8_t* smem_dst, const uint8_t* gsrc,
                                        bool full) {
    unsigned d = (unsigned)__cvta_generic_to_shared(smem_dst);
    int sz = full ? 8 : 0; // src-size 0 => zero-fill the 8B chunk
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d),
                 "l"(gsrc), "r"(sz));
}

template <int KDR>
__global__ void __launch_bounds__(256) mmq_raw_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_raw_sh[];
    uint8_t* qa8 = mmq_raw_sh;                                    // [2][KDR][BI][40]
    uint8_t* qb8 = qa8 + 2 * KDR * MMQ_BI * 40;                   // [2][BI][144]
    float* sds = reinterpret_cast<float*>(qb8 + 2 * MMQ_BI * 144);// [2][KDR][BI]
    float* sdm = sds + 2 * KDR * MMQ_BI;                          // [2][KDR][BI]

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int wm = warp >> 1, wn = warp & 1;   // same warp mapping as R1
    const int i0 = blockIdx.x * MMQ_BI;
    const int j0 = blockIdx.y * MMQ_BJ;
    const int i0w = wn * 32;
    const int j0w = wm * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[16] = {0.0f};   // [nh][h][l]: 2 B-frags x 2 A-frags x 4 C regs

#define RAW_STAGE(kt, b)                                                       \
    do {                                                                       \
        /* A: (token, chunk) pad40 chunks as 5x 8B cp.async */                 \
        for (int x = threadIdx.x; x < MMQ_BI * KDR * 5; x += blockDim.x) {     \
            int kd = x / (MMQ_BI * 5);                                         \
            int rem = x % (MMQ_BI * 5);                                        \
            int r = rem / 5, u = rem % 5;                                      \
            int c = (kt) * KDR + kd;                                           \
            int tok = i0 + r;                                                  \
            const uint8_t* src =                                               \
                q8x + ((size_t)tok * nb32 + c) * 40 + u * 8;                   \
            uint8_t* dst = qa8 + ((size_t)(b) * KDR + kd) * MMQ_BI * 40        \
                         + (size_t)r * 40 + u * 8;                             \
            mmq_cp8(dst, src, tok < nt && c < nchunk);                         \
        }                                                                      \
        /* B: whole 144B super-block per row as 9x 16B cp.async */             \
        int sb = ((kt) * KDR) >> 3;                                            \
        for (int x = threadIdx.x; x < MMQ_BI * 9; x += blockDim.x) {           \
            int r = x / 9, u = x % 9;                                          \
            int j = j0 + r;                                                    \
            const uint8_t* src =                                               \
                W + (size_t)j * ((size_t)nsb * 144) + (size_t)sb * 144         \
                  + u * 16;                                                    \
            uint8_t* dst = qb8 + ((size_t)(b) * MMQ_BI + r) * 144 + u * 16;    \
            gemm_cp16((__half*)(void*)dst, (const __half*)(const void*)src,    \
                      j < od && sb < nsb);                                     \
        }                                                                      \
        /* per-(row, chunk) scales from global (a few B per unit, L2-hot) */   \
        for (int x = threadIdx.x; x < MMQ_BI * KDR; x += blockDim.x) {         \
            int kd = x / MMQ_BI, r = x % MMQ_BI;                               \
            int c = (kt) * KDR + kd;                                           \
            int j = j0 + r;                                                    \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                int sg = c & 7;                                                \
                const uint8_t* blk =                                           \
                    W + (size_t)j * ((size_t)nsb * 144)                        \
                      + (size_t)(c >> 3) * 144;                                \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(sg, blk + 4, &sc, &m);                        \
                dv = h2f(*(const uint16_t*)blk) * (float)sc;                   \
                mv = -(h2f(*(const uint16_t*)(blk + 2)) * (float)m);           \
            }                                                                  \
            sds[((b) * KDR + kd) * MMQ_BI + r] = dv;                           \
            sdm[((b) * KDR + kd) * MMQ_BI + r] = mv;                           \
        }                                                                      \
    } while (0)

    RAW_STAGE(0, 0);
    gemm_cp_commit();

    int buf = 0;
    for (int kt = 0; kt < nktile; ++kt, buf ^= 1) {
        if (kt + 1 < nktile) RAW_STAGE(kt + 1, buf ^ 1);
        gemm_cp_commit();
        gemm_cp_wait1();          // at most the prefetch group pending
        __syncthreads();          // scales (plain stores) + landed bytes visible

        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)(buf * KDR + kd) * MMQ_BI * 40;
            const float* sdst = sds + (buf * KDR + kd) * MMQ_BI;
            const float* sdmt = sdm + (buf * KDR + kd) * MMQ_BI;

            // A fragments: int8 lane words straight out of the raw chunk.
            int a[2][4], b[2][2];
            int clow[2][2][4], chigh[2][2][4];
            #pragma unroll
            for (int h = 0; h < 2; h++) {
                const int r0 = i0w + h * 16 + (lane >> 2);
                const int r1 = r0 + 8;
                const uint8_t* p0 = qat + (size_t)r0 * 40 + 4;
                const uint8_t* p1 = qat + (size_t)r1 * 40 + 4;
                a[h][0] = *(const int*)(p0 + 4 * (lane & 3));
                a[h][1] = *(const int*)(p1 + 4 * (lane & 3));
                a[h][2] = *(const int*)(p0 + 4 * ((lane & 3) + 4));
                a[h][3] = *(const int*)(p1 + 4 * ((lane & 3) + 4));
            }
            // B fragments: unpack the raw nibbles in registers.
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const int jr = j0w + nh * 8 + (lane >> 2);
                const uint8_t* rb8 = qb8 + (size_t)(buf * MMQ_BI + jr) * 144;
                uint32_t n0 = *(const uint32_t*)(rb8 + 16 + (sg >> 1) * 32
                                                + 4 * (lane & 3));
                uint32_t n1 = *(const uint32_t*)(rb8 + 16 + (sg >> 1) * 32
                                                + 4 * ((lane & 3) + 4));
                b[nh][0] = (int)((sg & 1) ? ((n0 >> 4) & 0x0F0F0F0Fu)
                                          : (n0 & 0x0F0F0F0Fu));
                b[nh][1] = (int)((sg & 1) ? ((n1 >> 4) & 0x0F0F0F0Fu)
                                          : (n1 & 0x0F0F0F0Fu));
            }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) { clow[nh][h][l] = 0; chigh[nh][h][l] = 0; }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    mmq_mma_k32(clow[nh][h], a[h], b[nh]);

            // rescale: identical math/layout to the R1 kernel; A-side
            // d/ssum come straight from the raw chunk.
            float da_q[4];
            int sa_q[4];
            #pragma unroll
            for (int t4 = 0; t4 < 4; t4++) {
                const uint8_t* at = qat + (size_t)(i0w + (lane >> 2) + t4 * 8) * 40;
                da_q[t4] = h2f(*(const uint16_t*)at);
                sa_q[t4] = (int)*(const uint32_t*)(at + 36);
            }
            // r15: the dmv correction term is rank-1 in (token, od-col) —
            // the row-side product da*sa is shared by the od-col pair of
            // each C fragment, so fold it once per row (4 FMUL/chunk)
            // instead of once per C value (8 FMUL/chunk). The dsv term and
            // the per-chunk scale application are unchanged.
            const float dma[4] = { da_q[0] * (float)sa_q[0],
                                   da_q[1] * (float)sa_q[1],
                                   da_q[2] * (float)sa_q[2],
                                   da_q[3] * (float)sa_q[3] };
            float dsv[2][8], dmv[2][8];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int jj = 0; jj < 8; jj++) {
                    dsv[nh][jj] = sdst[j0w + nh * 8 + jj];
                    dmv[nh][jj] = sdmt[j0w + nh * 8 + jj];
                }
            #pragma unroll
            for (int nh = 0; nh < 2; nh++)
                #pragma unroll
                for (int h = 0; h < 2; h++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[h * 2 + (l >> 1)];
                        const int jj = (lane & 3) * 2 + (l & 1);
                        const int idx = nh * 8 + h * 4 + l;
                        sum[idx] += da * dsv[nh][jj] * (float)clow[nh][h][l];
                        sum[idx] += dma[h * 2 + (l >> 1)] * dmv[nh][jj];
                    }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int nh = 0; nh < 2; nh++)
        #pragma unroll
        for (int h = 0; h < 2; h++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + i0w + h * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[nh * 8 + h * 4 + l];
            }
#endif // __CUDA_ARCH__ >= 800
}

template <int KDR>
// (wide clone: 128-token x 128-od block tile; each warp owns 16 od-rows and
// issues 16 independent mma chains per 32-k chunk = 2 B-frags (8 od-rows
// each) x 8 A-frags (16 tokens each) — llama.cpp MMQ accumulator depth.)
#undef RAW_STAGE
__global__ void __launch_bounds__(256) mmq_raw_wide_nt_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
#define MMQ_WBI 128
#define MMQ_WBJ 128
#define MMQ_WBQ 48  // padded per-(sg,row) qb8 slot: 16B-aligned, 12r mod 32
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_raw_sh[];
    // Single-buffer sync-staged layout — KD=8 totals 98,304B (1 block/SM;
    // KD=4 is 73,728B; resident warps hide the staging latency):
    //   qa8   [KDR][128] x 32B  chunk qs only (d/ssum in sda_q). r22: the
    //                         16B granules are XOR-swizzled inside 128B
    //                         super-rows (4 rows each): granule
    //                         ((row&3)*2 + h) ^ ((row>>2)&7) — the 32B row
    //                         stride puts ldmatrix rows 4 apart on the same
    //                         bank phase (2-way conflict); the swizzle gives
    //                         every ldmatrix phase 8 distinct phases, zero
    //                         smem growth.
    //   sda_q [KDR][128] x 8B   (d f16 | ssum i16) packed, uint2-tiling
    //                           [KDR][16 g][8 q]: one LDS.64 serves the
    //                           token pair (t, t+8) a C fragment needs
    //   qb8   [8][128][48]      B sub-blocks pre-expanded to per-k int8,
    //                           SLOT-MAJOR (sg-major); 48B row stride puts
    //                           every ldmatrix row on a distinct bank phase
    //                           (raw nibbles 0..15; 128 od-rows per tile)
    //   sds   [KDR][128] float2 (d | dmin*m): one float4 load serves the
    //                           (j, j+1) od-col pair per minitile
    uint8_t* qa8 = mmq_raw_sh;
    uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_WBI * 32);
    uint8_t* qb8 = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_WBI * 2);
    float2* sds = reinterpret_cast<float2*>(qb8 + 8 * MMQ_WBJ * MMQ_WBQ);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Each warp owns a private 16-od-row slice and reads the FULL 128-token
    // tile: B fragments become warp-exclusive, A fragments are warp-shared.
    const int i0 = blockIdx.x * MMQ_WBI;
    const int j0 = blockIdx.y * MMQ_WBJ;
    const int j0w = warp * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[64] = {0.0f};   // [g][nh][l]: 8 A-frags x 2 B-frags x 4 C regs

#define RAW_STAGE(kt)                                                          \
    do {                                                                       \
        /* llama.cpp-style synchronous staging, single buffer: plain global    \
         * -> smem loads, one syncthreads orders them. At 128x64 tiles the     \
         * smem is small enough for 2 blocks/SM - latency hiding comes from    \
         * occupancy, not prefetch depth. */                                   \
        /* r20: split-phase A staging. The old interleaved LDG->STS chains    \
         * stalled the warp at the first store of every 4-deep unroll batch   \
         * (PC-sampled: the leading STS held 16.7% of all warp stalls = one   \
         * full memory latency per batch, ~4 batches per kt). Issue ALL       \
         * global loads into registers first - one deep independent LDG batch \
         * per warp per kt - then store to smem. Identical addresses, traffic \
         * and instruction count; only the dependency schedule changes. */    \
        /* r22: r20's split-phase A staging is kept verbatim; only the qa8    \
         * store address gained the XOR swizzle. The d/ssum stream fold      \
         * (single 9-word-per-chunk pass) was tried and REVERTED: the old    \
         * scattered d/ssum loads are L1 hits (the qs pass of the same       \
         * chunks has the lines resident), so folding saves little sector    \
         * traffic while the flat enumeration costs ALU + branchy batches    \
         * (-19% wall, see docs r22). */                                     \
        {                                                                      \
            unsigned av[KDR * 4];          /* qs words: 128*KDR*8 / 256 thr */ \
            unsigned short dv[KDR / 2];    /* d f16 words: 128*KDR / 256 */    \
            unsigned sv[KDR / 2];          /* ssum words */                    \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 4; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),             \
                          kd = x >> 10;    /* x/(8*MMQ_WBI), 8*128 = 1024 */   \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned v = 0;                                                \
                if (tok < nt && c < nchunk)                                    \
                    v = *(const unsigned*)(q8x                                 \
                        + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);          \
                av[i] = v;                                                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR / 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_WBI - 1), kd = x >> 7;  /* x/MMQ_WBI */ \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned short d16 = 0;                                        \
                unsigned ss = 0;                                               \
                if (tok < nt && c < nchunk) {                                  \
                    const uint8_t* src = q8x                                   \
                        + ((size_t)tok * nb32 + c) * 40;                       \
                    d16 = *(const unsigned short*)src;                         \
                    ss = (unsigned)(short)*(const int*)(src + 36);             \
                }                                                              \
                dv[i] = d16;                                                   \
                sv[i] = ss;                                                    \
            }                                                                              _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 4; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_WBI - 1),             \
                          kd = x >> 10;                                        \
                const int R = kd * MMQ_WBI + r;                                \
                *(unsigned*)(qa8 + (size_t)(R & ~3) * 32                       \
                    + (size_t)(((((R & 3) << 1) + (u >> 2))                    \
                                ^ ((R >> 2) & 7)) << 4)                        \
                    + (size_t)(u & 3) * 4) = av[i];                            \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR / 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_WBI - 1), kd = x >> 7;                  \
                *(unsigned*)(sda_q + ((size_t)kd * MMQ_WBI + (r >> 4) * 8      \
                                      + (r & 7)) * 2 + ((r >> 3) & 1)) =       \
                    dv[i] | (sv[i] << 16);                                     \
            }                                                                  \
        }                                                                      \
        /* B: expand ONE 256-k super-block to per-k int8 AT STAGING - raw     \
         * nibble values 0..15 (folding (nib - m) here would skew the dmin    \
         * term, which the epilogue scales by dmin*m, not d*sc; the two-term  \
         * dsv/dmv rescale stays). Pair p covers sub-blocks 2p (low           \
         * nibbles) and 2p+1 (high nibbles) over qs bytes p*32..p*32+31;      \
         * slot bytes are element-ordered so compute reads plain words;       \
         * slots live at [sg][row][48B] (sg-major) so one ldmatrix.x4 loads   \
         * a whole 16-od-row x 32-k B fragment per warp-chunk.                \
         * At KDR=4 two consecutive k-tiles share the super-block and the     \
         * expanded qb8 persists across the kt barrier - restage only when    \
         * this k-tile starts a new super-block. */                           \
        if (((kt) * KDR & 7) == 0) {                                           \
        for (int x = threadIdx.x; x < MMQ_WBJ * 4; x += blockDim.x) {          \
            const int r = x >> 2, p = x & 3;                                   \
            const int j = j0 + r, sb = ((kt) * KDR) >> 3;                      \
            uint4 v0 = make_uint4(0, 0, 0, 0), v1 = make_uint4(0, 0, 0, 0);    \
            if (j < od && sb < nsb) {                                          \
                const uint8_t* src =                                           \
                    W + (size_t)j * ((size_t)nsb * 144)                        \
                      + (size_t)sb * 144 + 16 + p * 32;                        \
                v0 = *(const uint4*)(src);                                     \
                v1 = *(const uint4*)(src + 16);                                \
            }                                                                  \
            const unsigned M = 0x0F0F0F0Fu;                                    \
            uint8_t* dst = qb8 + (size_t)(p * 2) * (MMQ_WBJ * MMQ_WBQ)         \
                         + (size_t)r * MMQ_WBQ;                                \
            *(uint4*)(dst)      = make_uint4(v0.x & M, v0.y & M,               \
                                             v0.z & M, v0.w & M);              \
            *(uint4*)(dst + 16) = make_uint4(v1.x & M, v1.y & M,               \
                                             v1.z & M, v1.w & M);              \
            uint8_t* dst1 = dst + MMQ_WBJ * MMQ_WBQ;                           \
            *(uint4*)(dst1)     = make_uint4((v0.x >> 4) & M,                  \
                                             (v0.y >> 4) & M,                  \
                                             (v0.z >> 4) & M,                  \
                                             (v0.w >> 4) & M);                 \
            *(uint4*)(dst1 + 16) = make_uint4((v1.x >> 4) & M,                 \
                                              (v1.y >> 4) & M,                 \
                                              (v1.z >> 4) & M,                 \
                                              (v1.w >> 4) & M);                \
        }                                                                      \
        }                                                                      \
        for (int x = threadIdx.x; x < MMQ_WBJ * KDR; x += blockDim.x) {        \
            int r = x % MMQ_WBJ, kd = x / MMQ_WBJ;                             \
            int j = j0 + r, c = (kt) * KDR + kd;                               \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                const uint8_t* blk =                                           \
                    W + (size_t)j * ((size_t)nsb * 144)                        \
                      + (size_t)(c >> 3) * 144;                                \
                float d = h2f(*(const uint16_t*)blk);                          \
                float dmin = h2f(*(const uint16_t*)(blk + 2));                 \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(c & 7, blk + 4, &sc, &m);                     \
                dv = d * (float)sc;                                            \
                mv = -(dmin * (float)m);                                       \
            }                                                                  \
            sds[(size_t)kd * MMQ_WBJ + r] = make_float2(dv, mv);               \
        }                                                                      \
    } while (0)

    RAW_STAGE(0);

    // r22: precomputed swizzled A-frag byte offsets. The 8 ldmatrix
    // addresses per chunk are lane-invariant except for the kd base:
    // addr = qat + G[g], G[g] = g*512 + (lane&12)*32
    //      + (((lane&3)*2 + (lane>>4&1)) ^ (lane>>2&3) ^ ((g&1)*4)) << 4.
    // One IADD per ldmatrix (below baseline's 3), and the granule XOR
    // gives every ldmatrix phase 8 distinct bank phases (the 32B row
    // stride is 2-way conflicted).
    const unsigned l12m = (unsigned)(lane & 12) * 32;
    const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                             ^ ((lane >> 2) & 3)) << 4;
    unsigned G[8];
    #pragma unroll
    for (int g = 0; g < 8; g++)
        G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);

    int buf = 0; (void)buf;
    for (int kt = 0; kt < nktile; ++kt) {
        if (kt > 0) RAW_STAGE(kt);
        __syncthreads();          // single-buffer stage visible to all warps

        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)kd * MMQ_WBI * 32;

            // A fragments: 8 independent 16-token groups tile the full
            // 128-token row, one ldmatrix.x4 per group (16 rows x 32B:
            // lanes 0-7 -> rows 0-7 byte 0, 8-15 -> rows 8-15 byte 0,
            // 16-23 -> rows 0-7 byte 16, 24-31 -> rows 8-15 byte 16 —
            // the standard m16n8k32 A-fragment distribution).
            // r22: addresses are the precomputed swizzled offsets G[g]
            // (see above) — XOR-swizzled granule index, same map the
            // staging stores use.
            int a[8][4], b[2][2];
            int clow[8][2][4];
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                const uint8_t* p = qat + G[g];
                unsigned r0_, r1_, r2_, r3_;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                    : "r"((unsigned)__cvta_generic_to_shared(p)));
                a[g][0] = (int)r0_; a[g][1] = (int)r1_;
                a[g][2] = (int)r2_; a[g][3] = (int)r3_;
            }
            // B fragments: ONE ldmatrix.x4 serves both 8-od-row
            // minitiles (matrices 0/1 = od-rows 0-7 at k-halves 0/1,
            // matrices 2/3 = od-rows 8-15). reg_i of lane L = matrix_i row
            // L/4, bytes (L%4)*4 — the exact mma.m16n8k32 B-operand
            // distribution the plain LDS pattern produced. Per-lane address
            // parts are loop-invariant; only the sg term moves per chunk.
            {
                const uint8_t* rb8 = qb8
                    + (size_t)sg * (MMQ_WBJ * MMQ_WBQ)
                    + (size_t)(j0w + (lane >> 4) * 8 + (lane & 7)) * MMQ_WBQ
                    + (size_t)((lane >> 3) & 1) * 16;
                unsigned b0_, b1_, b2_, b3_;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(b0_), "=r"(b1_), "=r"(b2_), "=r"(b3_)
                    : "r"((unsigned)__cvta_generic_to_shared(rb8)));
                b[0][0] = (int)b0_; b[0][1] = (int)b1_;
                b[1][0] = (int)b2_; b[1][1] = (int)b3_;
            }
            #pragma unroll
            for (int g = 0; g < 8; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) clow[g][nh][l] = 0;
            // 16 independent mma chains per thread per chunk, all C
            // fragments live simultaneously (llama.cpp accumulator depth).
            #pragma unroll
            for (int g = 0; g < 8; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    mmq_mma_k32(clow[g][nh], a[g], b[nh]);

            // rescale: identical math/layout to the R1 kernel; A-side
            // d/ssum come straight from the raw chunk. da/sa load per
            // token-group to keep registers for the accumulators.
            // od-col scales: one float4 per minitile serves the (j, j+1)
            // column pair the C fragment consumes (float2-packed at staging).
            float dsv[2][2], dmv[2][2];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const float4 sc4 = *(const float4*)(sds
                    + (size_t)kd * MMQ_WBJ + j0w + nh * 8 + (lane & 3) * 2);
                dsv[nh][0] = sc4.x; dsv[nh][1] = sc4.z;
                dmv[nh][0] = sc4.y; dmv[nh][1] = sc4.w;
            }
            #pragma unroll
            for (int g = 0; g < 8; g++) {
                float da_q[2];
                int sa_q[2];
                // token pair (t, t+8) in one LDS.64 (uint2 tiling)
                const uint2 pk2 = *(const uint2*)(sda_q
                    + (size_t)kd * MMQ_WBI * 2 + g * 16 + (lane >> 2) * 2);
                da_q[0] = h2f((unsigned short)(pk2.x & 0xFFFF));
                sa_q[0] = (int)(short)(pk2.x >> 16);
                da_q[1] = h2f((unsigned short)(pk2.y & 0xFFFF));
                sa_q[1] = (int)(short)(pk2.y >> 16);
                // r15: the dmv correction term is rank-1 in (token, od-col) —
                // the row-side product da*sa is shared by the od-col pair of
                // each C fragment, so fold it once per row (16 FMUL/chunk)
                // instead of once per C value (64 FMUL/chunk). The dsv term
                // and the per-chunk scale application are unchanged.
                const float dma[2] = { da_q[0] * (float)sa_q[0],
                                       da_q[1] * (float)sa_q[1] };
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[l >> 1];
                        const int idx = (g * 2 + nh) * 4 + l;
                        sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];
                        sum[idx] += dma[l >> 1] * dmv[nh][l & 1];
                    }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < 8; g++)
        #pragma unroll
        for (int nh = 0; nh < 2; nh++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + g * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];
            }
#endif // __CUDA_ARCH__ >= 800
}


// ─── mmq RAW-NIBBLE kernel (Direction-A "NB" variant, docs §11) ───────────
// 64 tokens x 128 od, KD=8 native (one full 256-k super-block per k-tile in
// the raw qs plane), 8 warps x 16 od-rows, sum[32]. The weight B tile is the
// RAW 2-nibbles/byte qs plane (O*128 = 16,384 B), so the block totals
// 45,056 B -> 2 blocks/SM on GB10. B fragments are assembled in-loop by
// reading the packed nibbles (NO staging-ALU expansion, NO ldmatrix for B).
// Numerics are exact to the wide kernel: the mma consumes the UNSIGNED 0..15
// nibble as the int8 B operand, and the fp32 two-term rank-1 rescale
// (d*sc*nib - dmin*m) is applied per chunk — never the (nib - m) fold.
//
// The A path is byte-identical to mmq_raw_wide_nt_kernel (r20 split-phase
// staging + r22 XOR swizzle, 32B rows) but over T=64 (4 A-frag groups).
// The B unpack was validated standalone against the wide kernel's
// ldmatrix B-fragment: for chunk sg, od-row jj, lane l:
//   reg0 byte j = nibble(sg&1 ? hi : lo) of qs[(sg>>1)*32 + (l&3)*4 + j]
//   reg1 byte j = nibble(sg&1 ? hi : lo) of qs[(sg>>1)*32 + 16 + (l&3)*4 + j]
// (0x0F0F0F0F low, (v>>4)&0x0F0F0F0F high; upper nibble zero => positive int8).
static constexpr int MMQ_NBI = 64;   // tokens (i) per block tile
static constexpr int MMQ_NBJ = 128;  // od rows (j) per block tile
template <int KDR>
__global__ void __launch_bounds__(256) mmq_raw_nb_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ q8x,
    float* __restrict__ C, int nt, int od, int id
) {
#if __CUDA_ARCH__ >= 800
    extern __shared__ uint8_t mmq_nb_sh[];
    // Single-buffer sync-staged, KD=8 totals 43,008 B -> 2 blocks/SM:
    //   qa8     [KDR][64][32]   chunk q8 planes (r22 XOR swizzle, r20 split)
    //   sda_q   [KDR][64]         (d f16 | ssum i16) packed, one uint32 per
    //                             token, Q-MAJOR: within kd the 4 token groups
    //                             for a given lane's pair q are CONTIGUOUS
    //                             (uint32 idx = kd*64 + q*8 + g*2 + half), so a
    //                             lane reads its whole per-chunk d/ssum set with
    //                             TWO LDS.128 (32 LDS.64 -> 16 LDS.128 / k-tile).
    //   qb_raw  [128][128]        raw GGUF qs plane (2 nibbles/byte, full
    //                             super-block per od-row)
    //   sds     [KDR][128] float2 (d | dmin*m): r15 rank-1 rescale terms
    uint8_t* qa8 = mmq_nb_sh;
    uint32_t* sda_q = reinterpret_cast<uint32_t*>(qa8 + KDR * MMQ_NBI * 32);
    uint8_t* qb_raw = reinterpret_cast<uint8_t*>(sda_q + KDR * MMQ_NBI);
    float2* sds = reinterpret_cast<float2*>(qb_raw + MMQ_NBJ * 128);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // Each warp owns a private 16-od-row slice and reads the FULL 64-token
    // tile (B fragments warp-exclusive, A fragments warp-shared).
    const int i0 = blockIdx.x * MMQ_NBI;
    const int j0 = blockIdx.y * MMQ_NBJ;
    const int j0w = warp * 16;
    const int nb32 = id >> 5;
    const int nchunk = nb32;
    const int nsb = nb32 >> 3;
    const int nktile = (nchunk + KDR - 1) / KDR;

    float sum[32] = {0.0f};   // [g=4][nh=2][l=4] C accumulators (fp32)

#define RAW_STAGE_NB(kt)                                                      \
    do {                                                                       \
        /* ---- A: r20 split-phase (LDG batch then STS) + r22 XOR swizzle ----*/\
        {                                                                      \
            unsigned av[KDR * 2];   /* 8 words x 64 tok x KDR / 256 thr */     \
            unsigned short dv[2];   /* KDR * NBI/256 = 2 f16 d words */        \
            unsigned sv[2];                                                     \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),             \
                          kd = x / (8 * MMQ_NBI);   /* KDR=8, 8*NBI = 512 */  \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned v = 0;                                                \
                if (tok < nt && c < nchunk)                                    \
                    v = *(const unsigned*)(q8x                                 \
                        + ((size_t)tok * nb32 + c) * 40 + 4 + u * 4);          \
                av[i] = v;                                                     \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 2; ++i) {                                      \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_NBI - 1), kd = x / MMQ_NBI;  /* x/NBI */ \
                const int tok = i0 + r, c = (kt) * KDR + kd;                   \
                unsigned short d16 = 0; unsigned ss = 0;                       \
                if (tok < nt && c < nchunk) {                                  \
                    const uint8_t* src = q8x + ((size_t)tok * nb32 + c) * 40;  \
                    d16 = *(const unsigned short*)src;                         \
                    ss = (unsigned)(short)*(const int*)(src + 36);             \
                }                                                              \
                dv[i] = d16; sv[i] = ss;                                       \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < KDR * 2; ++i) {                                \
                const int x = threadIdx.x + i * 256;                           \
                const int u = x & 7, r = (x >> 3) & (MMQ_NBI - 1),             \
                          kd = x / (8 * MMQ_NBI);                               \
                const int R = kd * MMQ_NBI + r;                                \
                *(unsigned*)(qa8 + (size_t)(R & ~3) * 32                       \
                    + (size_t)(((((R & 3) << 1) + (u >> 2))                    \
                                ^ ((R >> 2) & 7)) << 4)                        \
                    + (size_t)(u & 3) * 4) = av[i];                            \
            }                                                                  \
            _Pragma("unroll")                                                  \
            for (int i = 0; i < 2; ++i) {                                      \
                const int x = threadIdx.x + i * 256;                           \
                const int r = x & (MMQ_NBI - 1), kd = x / MMQ_NBI;             \
                const int g = r >> 4, t = r & 15, q = t & 7, half = t >> 3;    \
                const int rg = g >> 1, gsel = g & 1;                           \
                /* conflict-free: region=g/2 block, q*16B stride, gsel*8B */  \
                *(unsigned*)(sda_q + (size_t)kd * MMQ_NBI                       \
                              + rg * 32 + q * 4 + gsel * 2 + half) =           \
                    dv[i] | (sv[i] << 16);                                     \
            }                                                                  \
        }                                                                      \
        /* ---- B: bulk raw qs super-block copy (r18-style, no staging ALU) */ \
        {                                                                      \
            const int sb = ((kt) * KDR) >> 3;                                  \
            for (int off = threadIdx.x; off < MMQ_NBJ * 8; off += blockDim.x) {\
                const int jj = off >> 3, c8 = off & 7;                         \
                const int j = j0 + jj;                                         \
                uint4 v = make_uint4(0, 0, 0, 0);                              \
                if (j < od && sb < nsb)                                        \
                    v = *(const uint4*)(W + (size_t)j * ((size_t)nsb * 144)    \
                        + (size_t)sb * 144 + 16 + (size_t)c8 * 16);            \
                *(uint4*)(qb_raw + (size_t)jj * 128 + (size_t)c8 * 16) = v;    \
            }                                                                  \
        }                                                                      \
        /* ---- B: SDS per-(chunk, od-row) rank-1 rescale terms ---- */        \
        for (int x = threadIdx.x; x < MMQ_NBJ * KDR; x += blockDim.x) {        \
            const int r = x % MMQ_NBJ, kd = x / MMQ_NBJ;                       \
            const int j = j0 + r, c = (kt) * KDR + kd;                         \
            float dv = 0.0f, mv = 0.0f;                                        \
            if (j < od && c < nchunk) {                                        \
                const uint8_t* blk = W + (size_t)j * ((size_t)nsb * 144)       \
                    + (size_t)(c >> 3) * 144;                                  \
                const float d = h2f(*(const uint16_t*)blk);                    \
                const float dmin = h2f(*(const uint16_t*)(blk + 2));           \
                uint8_t sc, m;                                                 \
                get_scale_min_k4(c & 7, blk + 4, &sc, &m);                     \
                dv = d * (float)sc;                                            \
                mv = -(dmin * (float)m);                                       \
            }                                                                  \
            sds[(size_t)kd * MMQ_NBJ + r] = make_float2(dv, mv);               \
        }                                                                      \
    } while (0)

    // r22: precomputed swizzled A-frag byte offsets (identical to wide kernel).
    const unsigned l12m = (unsigned)(lane & 12) * 32;
    const unsigned grc = (unsigned)(((lane & 3) << 1) + ((lane >> 4) & 1)
                             ^ ((lane >> 2) & 3)) << 4;
    unsigned G[4];   // only 4 token groups at T=64
    #pragma unroll
    for (int g = 0; g < 4; g++)
        G[g] = (unsigned)g * 512 + l12m + ((g & 1) ? (grc ^ 64u) : grc);

    RAW_STAGE_NB(0);

    for (int kt = 0; kt < nktile; ++kt) {
        if (kt > 0) RAW_STAGE_NB(kt);
        __syncthreads();

        #pragma unroll
        for (int kd = 0; kd < KDR; kd++) {
            const int c = kt * KDR + kd;
            if (c >= nchunk) break;
            const int sg = c & 7;
            const uint8_t* qat = qa8 + (size_t)kd * MMQ_NBI * 32;

            // A fragments: 4 independent 16-token groups (T=64), r22 G[].
            int a[4][4], b[2][2];
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                const uint8_t* p = qat + G[g];
                unsigned r0_, r1_, r2_, r3_;
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0,%1,%2,%3}, [%4];\n"
                    : "=r"(r0_), "=r"(r1_), "=r"(r2_), "=r"(r3_)
                    : "r"((unsigned)__cvta_generic_to_shared(p)));
                a[g][0] = (int)r0_; a[g][1] = (int)r1_;
                a[g][2] = (int)r2_; a[g][3] = (int)r3_;
            }
            // B fragments: raw-nibble in-loop unpack (validated == wide kernel
            // ldmatrix). reg0 = qs[(sg>>1)*32 + (l&3)*4 + 0..3],
            //            reg1 = qs[(sg>>1)*32 + 16 + (l&3)*4 + 0..3].
            {
                const int p = sg >> 1, is_hi = sg & 1, lm3 = lane & 3;
                const unsigned M = 0x0F0F0F0Fu;
                #pragma unroll
                for (int nh = 0; nh < 2; nh++) {
                    const int jj = j0w + nh * 8 + (lane >> 2);
                    const uint8_t* qs = qb_raw + (size_t)jj * 128;
                    const uint32_t* q0 = (const uint32_t*)(qs + p * 32 + lm3 * 4);
                    const uint32_t* q1 = (const uint32_t*)(qs + p * 32 + 16 + lm3 * 4);
                    uint32_t v0 = *q0, v1 = *q1;
                    b[nh][0] = (int)(is_hi ? ((v0 >> 4) & M) : (v0 & M));
                    b[nh][1] = (int)(is_hi ? ((v1 >> 4) & M) : (v1 & M));
                }
            }
            int clow[4][2][4];
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) clow[g][nh][l] = 0;
            // 8 independent mma chains per thread per chunk (4 A-frags x 2
            // B-frags), all C fragments live simultaneously.
            #pragma unroll
            for (int g = 0; g < 4; g++)
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    mmq_mma_k32(clow[g][nh], a[g], b[nh]);

            // rescale: exact r15 two-term rank-1 fold (d*sc, -dmin*m).
            float dsv[2][2], dmv[2][2];
            #pragma unroll
            for (int nh = 0; nh < 2; nh++) {
                const float4 sc4 = *(const float4*)(sds
                    + (size_t)kd * MMQ_NBJ + j0w + nh * 8 + (lane & 3) * 2);
                dsv[nh][0] = sc4.x; dsv[nh][1] = sc4.z;
                dmv[nh][0] = sc4.y; dmv[nh][1] = sc4.w;
            }
            // r31: Q-major sda repack — one uint32 per token; the group-region
            // split (g/2 region block, q*16B stride) makes each warp LDS.128
            // read 8 unique 16B words at 16B stride = bank-conflict-free.
            // s0 = groups 0,1, s1 = groups 2,3 (TWO LDS.128 instead of the old
            // four LDS.64). Same values, same per-chunk application points.
            const uint32_t* sda_blk = sda_q + (size_t)kd * MMQ_NBI
                                      + (size_t)(lane >> 2) * 4;
            const uint4 s0 = *(const uint4*)(sda_blk);
            const uint4 s1 = *(const uint4*)(sda_blk + 32);
            #pragma unroll
            for (int g = 0; g < 4; g++) {
                float da_q[2];
                int sa_q[2];
                const unsigned w0 = g == 0 ? s0.x : (g == 1 ? s0.z : (g == 2 ? s1.x : s1.z));
                const unsigned w1 = g == 0 ? s0.y : (g == 1 ? s0.w : (g == 2 ? s1.y : s1.w));
                da_q[0] = h2f((unsigned short)(w0 & 0xFFFF));
                sa_q[0] = (int)(short)(w0 >> 16);
                da_q[1] = h2f((unsigned short)(w1 & 0xFFFF));
                sa_q[1] = (int)(short)(w1 >> 16);
                const float dma[2] = { da_q[0] * (float)sa_q[0],
                                       da_q[1] * (float)sa_q[1] };
                #pragma unroll
                for (int nh = 0; nh < 2; nh++)
                    #pragma unroll
                    for (int l = 0; l < 4; l++) {
                        const float da = da_q[l >> 1];
                        const int idx = (g * 2 + nh) * 4 + l;
                        sum[idx] += da * dsv[nh][l & 1] * (float)clow[g][nh][l];
                        sum[idx] += dma[l >> 1] * dmv[nh][l & 1];
                    }
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int g = 0; g < 4; g++)
        #pragma unroll
        for (int nh = 0; nh < 2; nh++)
            #pragma unroll
            for (int l = 0; l < 4; l++) {
                const int i = i0 + g * 16 + (l >> 1) * 8 + (lane >> 2);
                const int j = j0 + j0w + nh * 8 + (lane & 3) * 2 + (l & 1);
                if (i < nt && j < od)
                    C[(size_t)i * od + j] = sum[(g * 2 + nh) * 4 + l];
            }
#endif // __CUDA_ARCH__ >= 800
}

extern "C" int launch_mmq_raw_nb_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id;
    // 64-token x 128-od block tile, KD=8 native. Raw qs plane + single-buffer
    // staging = 43,008 B (r31 q-major sda repack shrinks sda_q 4,096 ->
    // 2,048 B) >>> 2 blocks/SM on GB10. KD!=8 is inapplicable to the
    // raw-nibble variant (the qs plane encodes a FULL 256-k super-block), so
    // clean-fallback (return 0) to the wide kernel. smem/reg guards return 0
    // on any cap failure (never silently launch over cap).
    if (kd != 8) return 0;
    const int smem = 8 * MMQ_NBI * 32   // qa8
                   + 8 * MMQ_NBI * 4    // sda_q (one uint32 per token)
                   + MMQ_NBJ * 128      // qb_raw
                   + 8 * MMQ_NBJ * 8;   // sds (float2 = 8B)
    dim3 grid((nt + MMQ_NBI - 1) / MMQ_NBI, (od + MMQ_NBJ - 1) / MMQ_NBJ);
    cudaFuncSetAttribute(reinterpret_cast<const void*>(&mmq_raw_nb_kernel<8>),
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { cudaGetLastError(); return 0; }
    mmq_raw_nb_kernel<8><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "minfer/cuda: mmq raw NB launch failed: %s\n",
                cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" int launch_mmq_raw_wide_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id;
    // 16-chain layout: 128-token x 128-od block tile. r14: qb8 slot-major
    // 48B stride (ldmatrix-for-B) + packed scales. KD=8 totals 98,304B and
    // KD=4 73,728B — both inside the ~99KB opt-in cap, 1 block/SM. The
    // attr/launch results are checked: an over-cap request used to fail
    // SILENTLY (r7 phantom 2124).
    dim3 grid((nt + 127) / 128, (od + 127) / 128);
    if (kd <= 4) {
        const int smem = 4 * MMQ_WBI * 32 + 4 * MMQ_WBI * 8
                       + 8 * MMQ_WBJ * MMQ_WBQ + 2 * 4 * MMQ_WBJ * 4;
        cudaError_t e = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(&mmq_raw_wide_nt_kernel<4>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) { cudaGetLastError(); return 0; }
        mmq_raw_wide_nt_kernel<4><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    } else {
        const int smem = 8 * MMQ_WBI * 32 + 8 * MMQ_WBI * 8
                       + 8 * MMQ_WBJ * MMQ_WBQ + 2 * 8 * MMQ_WBJ * 4;
        cudaError_t e = cudaFuncSetAttribute(
            reinterpret_cast<const void*>(&mmq_raw_wide_nt_kernel<8>),
            cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) { cudaGetLastError(); return 0; }
        mmq_raw_wide_nt_kernel<8><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    }
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        fprintf(stderr, "minfer/cuda: mmq raw wide launch failed: %s\n",
                cudaGetErrorString(e));
        return 0;
    }
    return 1;
}

extern "C" void launch_mmq_raw_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, cudaStream_t stream, int kd
) {
    (void)type_id; // q4_K only in the first cut
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    if (kd <= 4) {
        const int smem = 2 * 4 * MMQ_BI * 40 + 2 * MMQ_BI * 144
                         + 2 * 2 * 4 * MMQ_BI * 4;
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&mmq_raw_nt_kernel<4>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mmq_raw_nt_kernel<4><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    } else {
        const int smem = 2 * 8 * MMQ_BI * 40 + 2 * MMQ_BI * 144
                         + 2 * 2 * 8 * MMQ_BI * 4;
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&mmq_raw_nt_kernel<8>),
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        mmq_raw_nt_kernel<8><<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id);
    }
}

extern "C" int cuda_shared_per_sm() {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
    return v;
}

extern "C" int cuda_shared_per_block_optin() {
    int v = 0;
    cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    return v;
}

extern "C" void launch_mmq_nt(
    int type_id, const uint8_t* w, const uint8_t* q8, float* c,
    int nt, int od, int id, int q6_stride, cudaStream_t stream
) {
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    // dynamic shared: qa+qb tiles, sda/sds/sds1/sdm (float) + ssa (int)
    const int smem =
        (2 * MMQ_KD * (MMQ_BI * MMQ_WS) + 2 * MMQ_KD * (MMQ_BJ * MMQ_WS)
         + 2 * MMQ_KD * MMQ_BI + 4 * 2 * MMQ_KD * MMQ_BI)
        * 4;
#define MMQ_LAUNCH(KERN)                                                       \
    do {                                                                       \
        cudaFuncSetAttribute(reinterpret_cast<const void*>(&KERN),             \
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem); \
        KERN<<<grid, 256, smem, stream>>>(w, q8, c, nt, od, id, q6_stride);    \
    } while (0)
    switch (type_id) {
        case 0: MMQ_LAUNCH((mmq_nt_kernel<0, 1, false>)); break;
        case 1: MMQ_LAUNCH((mmq_nt_kernel<1, 1, false>)); break;
        case 2: MMQ_LAUNCH((mmq_nt_kernel<2, 1, true>)); break;
        case 3: MMQ_LAUNCH((mmq_nt_kernel<3, 1, false>)); break;
        case 4: MMQ_LAUNCH((mmq_nt_kernel<4, 1, true>)); break;
        case 5: MMQ_LAUNCH((mmq_nt_kernel<5, 1, true>)); break;
        case 6: MMQ_LAUNCH((mmq_nt_kernel<6, 1, true>)); break;
        default: MMQ_LAUNCH((mmq_nt_kernel<7, 2, false>)); break;
    }
#undef MMQ_LAUNCH
}
