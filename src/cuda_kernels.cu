// CUDA kernels for minfer — Q4_0 matmul + element-wise ops.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
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
    float am = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; j++) am = fmaxf(am, fabsf(src[j]));
    float d = am / 127.0f;
    float di = (d != 0.0f) ? 1.0f / d : 0.0f;
    *reinterpret_cast<__half*>(dst) = __float2half(d);
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        int q = int(rintf(src[j] * di));
        q = max(-128, min(127, q));
        dst[4 + j] = uint8_t(int8_t(q));
    }
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
__global__ void store_kv_f16(
    const float* __restrict__ src,
    __half* __restrict__ dst,
    int nkt, int nt,
    const int* positions
) {
    int t = blockIdx.x;
    int j = blockIdx.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = __float2half(src[t * nkt + j]);
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

template <typename KV>
__device__ __forceinline__ float2 kv_ld2(const KV* p);

template <>
__device__ __forceinline__ float2 kv_ld2<float>(const float* p) {
    return make_float2(p[0], p[1]);
}

template <>
__device__ __forceinline__ float2 kv_ld2<__half>(const __half* p) {
    return __half22float2(*reinterpret_cast<const __half2*>(p));
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
    const int SPLITS = 8;
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
    const float4* q4 = reinterpret_cast<const float4*>(q + h * hd);
    int hd4 = hd / 4;

    float mx = -INFINITY, S = 0.0f;
    float4 oc[32];
    #pragma unroll
    for (int i = 0; i < hd4; i++) oc[i] = make_float4(0, 0, 0, 0);

    for (int base = lo; base < hi; base += 64) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = base + lane_id * 2;
        int kv1 = kv0 + 1;
        if (kv0 < hi) {
            const KV* krow = k + (size_t)kv0 * stride_kv + hk * hd;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i];
                float2 a = kv_ld2<KV>(krow + i * 4);
                float2 b = kv_ld2<KV>(krow + i * 4 + 2);
                d += qv.x * a.x + qv.y * a.y + qv.z * b.x + qv.w * b.y;
            }
            s0 = d * scale;
        }
        if (kv1 < hi) {
            const KV* krow = k + (size_t)kv1 * stride_kv + hk * hd;
            float d = 0.0f;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float4 qv = q4[i];
                float2 a = kv_ld2<KV>(krow + i * 4);
                float2 b = kv_ld2<KV>(krow + i * 4 + 2);
                d += qv.x * a.x + qv.y * a.y + qv.z * b.x + qv.w * b.y;
            }
            s1 = d * scale;
        }
        float bmx = fmaxf(s0, s1);
        for (int off = 16; off > 0; off >>= 1)
            bmx = fmaxf(bmx, __shfl_xor_sync(0xFFFFFFFF, bmx, off));
        float nmx = fmaxf(mx, bmx);
        float corr = expf(mx - nmx);
        float e0 = (kv0 < hi) ? expf(s0 - nmx) : 0.0f;
        float e1 = (kv1 < hi) ? expf(s1 - nmx) : 0.0f;
        #pragma unroll
        for (int i = 0; i < hd4; i++) {
            oc[i].x *= corr; oc[i].y *= corr; oc[i].z *= corr; oc[i].w *= corr;
        }
        S *= corr;
        if (kv0 < hi) {
            const KV* vrow = v + (size_t)kv0 * stride_kv + hk * hd;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float2 a = kv_ld2<KV>(vrow + i * 4);
                float2 b = kv_ld2<KV>(vrow + i * 4 + 2);
                oc[i].x += e0 * a.x; oc[i].y += e0 * a.y;
                oc[i].z += e0 * b.x; oc[i].w += e0 * b.y;
            }
        }
        if (kv1 < hi) {
            const KV* vrow = v + (size_t)kv1 * stride_kv + hk * hd;
            #pragma unroll
            for (int i = 0; i < hd4; i++) {
                float2 a = kv_ld2<KV>(vrow + i * 4);
                float2 b = kv_ld2<KV>(vrow + i * 4 + 2);
                oc[i].x += e1 * a.x; oc[i].y += e1 * a.y;
                oc[i].z += e1 * b.x; oc[i].w += e1 * b.y;
            }
        }
        S += e0 + e1;
        mx = nmx;
    }

    #pragma unroll
    for (int i = 0; i < hd4; i++) {
        oc[i].x = warp_reduce_sum(oc[i].x);
        oc[i].y = warp_reduce_sum(oc[i].y);
        oc[i].z = warp_reduce_sum(oc[i].z);
        oc[i].w = warp_reduce_sum(oc[i].w);
    }
    S = warp_reduce_sum(S);

    if (lane_id == 0) {
        float* dst = partial + ((size_t)sp * nh + h) * pstr;
        dst[0] = mx;
        dst[1] = S;
        float4* o4 = reinterpret_cast<float4*>(dst + 4); // 16B-aligned by pstr
        #pragma unroll
        for (int i = 0; i < hd4; i++) o4[i] = oc[i];
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
    for (int sp = 0; sp < 8; sp++)
        gmx = fmaxf(gmx, partial[((size_t)sp * nh + h) * pstr]);
    float S = 0.0f, acc = 0.0f;
    for (int sp = 0; sp < 8; sp++) {
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
    dim3 grid(nt, nkt, 1);
    store_kv_f16<<<grid, dim3(1, 1, 1), 0, stream>>>(src, (__half*)dst, nkt, nt, positions);
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
    gqa_attn_split_partial<__half><<<dim3(8, n_head), 32, 0, stream>>>(
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
    gqa_attn_split_partial<float><<<dim3(8, n_head), 32, 0, stream>>>(
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
// Shared layout (dynamic, ~97 KB — opt-in via cudaFuncSetAttribute):
//   Qs [64*hd] f16   q tile (scale folded in, f16 for the tensor-core QK^T)
//   Ks [64*hd] f16   K tile           Vs [64*hd] f16  V tile
//   S  [64*64]       f32 scores, aliased as f16 probs after the row softmax
//   O  [64*hd] f32   output accumulator (rescaled by alpha per row)
//   m/l/alpha [64] f32 per-row online-softmax state
#define FA_TQ 64
#define FA_TKV 64
#define FA_PSTR (FA_TKV * 2) // probs row stride in halves (256B): probs row r aliases only Sf row r's first half, already read by the same thread — no cross-thread race
#define FA_HQ 32 // hd/4 dims per accumulator thread (kernel is gated to hd == 128)

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
    __half* Qs = reinterpret_cast<__half*>(smem);
    __half* Ks = Qs + FA_TQ * hd;
    __half* Vs = Ks + FA_TKV * hd;
    float* Sf = reinterpret_cast<float*>(Vs + FA_TKV * hd);
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
        Qs[i] = __float2half(qv);
    }
    if (tid < FA_TQ) {
        msh[tid] = -INFINITY;
        lsh[tid] = 0.0f;
    }
    // Per-thread output accumulator: thread owns (row, quadrant) with
    // row = tid & 63, quadrant = tid >> 6 (FA_HQ dims each). Keeping O in
    // registers (instead of shared) makes every P·V V-read a warp-wide
    // broadcast and the alpha reads conflict-free.
    float acc[FA_HQ];
#pragma unroll
    for (int dd = 0; dd < FA_HQ; dd++) acc[dd] = 0.0f;
    __syncthreads();

    const int last_t = min(nt - 1, tq0 + FA_TQ - 1);
    const int kv_end = positions[last_t] + 1;

    const int arow = tid & (FA_TQ - 1);
    const int aquad = tid >> 6; // 0..3

    for (int kt = 0; kt < kv_end; kt += FA_TKV) {
        // stage K/V tile (16B per lane; rows beyond kv_end zero-filled)
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
            *reinterpret_cast<uint4*>(&Ks[i]) = kk4;
            *reinterpret_cast<uint4*>(&Vs[i]) = vv4;
        }
        __syncthreads();

        // S = Q · K^T via wmma (reduction over hd)
        {
            using namespace nvcuda;
            int warp = tid >> 5;       // 0..7
            int wm = warp >> 1;        // q 16-block: 4
            int wk = warp & 1;         // kv 32-block: 2
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa;
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2];
            wmma::fill_fragment(fc[0], 0.0f);
            wmma::fill_fragment(fc[1], 0.0f);
            for (int d = 0; d < hd; d += 16) {
                wmma::load_matrix_sync(fa, &Qs[wm * 16 * hd + d], hd);
                wmma::load_matrix_sync(fb[0], &Ks[wk * 32 * hd + d], hd);
                wmma::load_matrix_sync(fb[1], &Ks[(wk * 32 + 16) * hd + d], hd);
                wmma::mma_sync(fc[0], fa, fb[0], fc[0]);
                wmma::mma_sync(fc[1], fa, fb[1], fc[1]);
            }
            wmma::store_matrix_sync(&Sf[wm * 16 * FA_TKV + wk * 32], fc[0], FA_TKV, wmma::mem_row_major);
            wmma::store_matrix_sync(&Sf[wm * 16 * FA_TKV + wk * 32 + 16], fc[1], FA_TKV, wmma::mem_row_major);
        }
        __syncthreads();

        // online softmax per row (thread = row): probs land in Pf (f16)
        if (tid < FA_TQ) {
            int r = tid;
            int t = tq0 + r;
            int qpos = (t < nt) ? positions[t] : -1;
            float m_old = msh[r];
            float m_new = m_old;
            for (int kk = 0; kk < FA_TKV; kk++) {
                int kv_g = kt + kk;
                if (kv_g <= qpos && kv_g < kv_end) {
                    float s = Sf[r * FA_TKV + kk];
                    if (s > m_new) m_new = s;
                }
            }
            float a = 1.0f;
            float l_new = lsh[r];
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
                    if (kv_g <= qpos && kv_g < kv_end) {
                        p = __expf(Sf[r * FA_TKV + kk] - m_new);
                    }
                    Pf[r * FA_PSTR + kk] = __float2half(p);
                    sum += p;
                }
                l_new = lsh[r] * a + sum;
            }
            alpha[r] = a;
            msh[r] = m_new;
            lsh[r] = l_new;
        }
        __syncthreads();

        // rescale the accumulator by alpha, then add P · V
        float ar = alpha[arow];
#pragma unroll
        for (int dd = 0; dd < FA_HQ; dd++) acc[dd] *= ar;
        for (int kk = 0; kk < FA_TKV; kk++) {
            float p = __half2float(Pf[arow * FA_PSTR + kk]);
            if (p != 0.0f) {
                const __half* vrow = &Vs[kk * hd + aquad * FA_HQ];
#pragma unroll
                for (int dd = 0; dd < FA_HQ; dd++) {
                    acc[dd] += p * __half2float(vrow[dd]);
                }
            }
        }
        __syncthreads();
    }

    // write out: acc / l — rows with l == 0 stay 0 (fully masked)
    int t = tq0 + arow;
    if (t < nt) {
        float l = lsh[arow];
        float inv = (l > 0.0f) ? 1.0f / l : 0.0f;
        float* orow = &o[(size_t)t * ne_q + h * hd + aquad * FA_HQ];
#pragma unroll
        for (int dd = 0; dd < FA_HQ; dd++) orow[dd] = acc[dd] * inv;
    }
}

int launch_fa_prefill_f16kv(
    const float* q, const __half* k, const __half* v, float* o,
    const int* positions, int nh, int nk, int hd, float scale, int nt,
    cudaStream_t stream
) {
    size_t smem = (size_t)3 * FA_TQ * hd * 2 + (size_t)FA_TQ * FA_TKV * 4 + 3 * FA_TQ * 4;
    static size_t attr_smem = 0;
    if (smem > attr_smem) {
        cudaError_t e = cudaFuncSetAttribute(
            fa_prefill_f16kv, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        if (e != cudaSuccess) {
            cudaGetLastError(); // clear the error so it cannot poison the stream
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
    uint32_t qh = *reinterpret_cast<const uint32_t*>(blk + 2);
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
    long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g < n) out[g] = __float2half(x[g]);
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
#endif // __CUDA_ARCH__ >= 800

// C[nt, od] = A[nt, id] · B[od, id]^T. 64x64 output tiles, k-step 32,
// double-buffered shared staging, 8 warps (each owns a 16 nt x 16 od
// fragment pair). f32 accumulation. Tails: nt/od masked at store, k-tail
// zero-filled (id % 8 == 0 keeps the uint4 chunk loads aligned).
__global__ void gemm_f16_nt_kernel(
    const __half* __restrict__ A, const __half* __restrict__ B,
    float* __restrict__ C, int nt, int od, int id
) {
    using namespace nvcuda;
    __shared__ __half As[2][64 * 32];
    __shared__ __half Bs[2][64 * 32];
    __shared__ float Cs[8][16 * 16]; // per-warp store staging

    int warp = threadIdx.x >> 5;      // 0..7
    int wm = warp >> 1;               // od sub-tile: 4 x 16 rows
    int wn = warp & 1;                // nt sub-tile: 2 x 32 rows
    // blockIdx.x = nt tile, blockIdx.y = od tile: consecutive blocks share
    // the same od-tile's B panel (64 rows x id f16, ~0.5MB) in L2, so the
    // f16 weight matrix streams from DRAM ~once instead of nt/64 times.
    int m0 = blockIdx.y * 64;
    int n0 = blockIdx.x * 64;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> fa[4];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> fb[2];
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc[2];
    wmma::fill_fragment(fc[0], 0.0f);
    wmma::fill_fragment(fc[1], 0.0f);

    int buf = 0;
#if __CUDA_ARCH__ >= 800
    gemm_load_tile_async(A, B, As[0], Bs[0], n0, m0, 0, nt, od, id);
    gemm_cp_commit();
#else
    gemm_load_tile_sync(A, B, As[0], Bs[0], n0, m0, 0, nt, od, id);
    __syncthreads();
#endif
    for (int k = 0; k < id; k += 32, buf ^= 1) {
#if __CUDA_ARCH__ >= 800
        if (k + 32 < id) {
            gemm_load_tile_async(A, B, As[buf ^ 1], Bs[buf ^ 1], n0, m0, k + 32, nt, od, id);
            gemm_cp_commit();
        }
        // wait until the CURRENT tile landed (one group may stay in flight)
        if (k + 32 < id)
            gemm_cp_wait1();
        else
            gemm_cp_wait0();
        __syncthreads();
#else
        if (k + 32 < id)
            gemm_load_tile_sync(A, B, As[buf ^ 1], Bs[buf ^ 1], n0, m0, k + 32, nt, od, id);
        __syncthreads();
#endif
        // fa: [n-block 0/1] x [k-half 0/1]; fb: [k-half 0/1] — both k halves
        // of the 32-wide slice must be accumulated (the v1 bug: only the
        // first 16 k's were multiplied).
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
    long long grid = (n + 255) / 256;
    if (grid > 2147483647LL) grid = 2147483647LL;
    convert_f32_f16_kernel<<<(int)grid, 256, 0, stream>>>(x, out, n);
}

void launch_gemm_f16(
    const __half* a, const __half* b, float* c,
    int nt, int od, int id, cudaStream_t stream
) {
    dim3 grid((nt + 63) / 64, (od + 63) / 64);
    gemm_f16_nt_kernel<<<grid, 256, 0, stream>>>(a, b, c, nt, od, id);
}

} // extern "C"
