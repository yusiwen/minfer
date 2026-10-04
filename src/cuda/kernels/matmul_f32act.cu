// src/cuda/kernels/matmul_f32act.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


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
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (r0 >= od) return;

    int nb = id / 32;
    int q4s = nb * Q4B;
    int q8s = nb * Q8B;

    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
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
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;


    int nb = id / QK;
    int q4s = nb * Q4B;

    const uint8_t* ax0 = weights + (r0 + 0) * q4s;
    const uint8_t* ax1 = weights + (r0 + 1) * q4s;
    const uint8_t* ax2 = weights + (r0 + 2) * q4s;
    const uint8_t* ax3 = weights + (r0 + 3) * q4s;
    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
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
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (r0 >= od) return;

    int nb = id / QK;
    int ws = nb * Q8B;
    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
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
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;

    if (r0 >= od) return;

    int nb = id / QK;
    int ws = nb * Q41B;
    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
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

}

// ─── Q5_1 × f32 matrix multiplication ─────────────────────────
// Q5_1 block: 24 bytes / 32 elements — f16 d, f16 m, u32 qh (bit j ↔ elem j,
// bit j+16 ↔ elem j+16), 16 bytes qs (byte j: low nibble = elem j, high =
// elem j+16). Value = d * unsigned_5bit + m (NO −16 offset — Q5_1 has a min).
// Structure mirrors q5_1_f32_matmul (4 rows/warp, 2 warps/block, lanes
// stride 32-element blocks). Q5_0: 22B = f16 d + u32 qh + 16 nibble bytes;
// value = nibble + 16*high_bit − 16 (no per-block min, unlike Q5_1).
__global__ void q5_0_f32_matmul(
    const uint8_t* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (r0 >= od) return;

    int nb = id / 32;
    int row_stride = nb * 22;
    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
        const float* y = acts + (size_t)t * id;

        float acc[NR0];
        #pragma unroll
        for (int rr = 0; rr < NR0; rr++) acc[rr] = 0.0f;

        for (int b = lane_id; b < nb; b += WARP) {
            const float* xb = y + b * 32;
            #pragma unroll
            for (int rr = 0; rr < NR0; rr++) {
                int o = r0 + rr;
                if (o >= od) break;
                const uint8_t* blk = weights + (size_t)o * row_stride + b * 22;
                float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
                // qh at block offset 2 is not 4-byte aligned (22B stride) — two
                // aligned u16 loads; misaligned u32 faults nondeterministically
                // on GB10 unified memory (err 716).
                uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                            | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
                const uint8_t* qs = blk + 6;
                float sdot = 0.0f;
                #pragma unroll
                for (int j = 0; j < 16; j++) {
                    float u_lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1) - 16.0f;
                    float u_hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1) - 16.0f;
                    sdot += u_lo * xb[j] + u_hi * xb[j + 16];
                }
                acc[rr] += d * sdot;
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

}

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
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (r0 >= od) return;

    int nb = id / 32;
    int row_stride = nb * 24;
    // Step 82: the token dimension lives in this in-block loop, not in the
    // launch grid (grid.y used to be nt = one full weight re-stream per
    // token). The weight bytes for the block's rows are re-read across
    // tokens from L1, so DRAM sees one weight stream per block; nt==1
    // keeps the exact single-token op order (bitwise).
    for (int t = 0; t < nt; ++t) {
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

}

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
extern "C" {

void launch_q4_0_q8_0_matmul(
    const uint8_t* weights, const uint8_t* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), 1, 1);
    minfer_launch_prelude("launch:q4_0_q8_0_matmul", "q4_0_q8_0_matmul");
    q4_0_q8_0_matmul<<<grid, minfer_launch_block("launch:q4_0_q8_0_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q4_0_q8_0_matmul", "q4_0_q8_0_matmul");
}

void launch_q4_0_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), 1, 1);
    minfer_launch_prelude("launch:q4_0_f32_matmul", "q4_0_f32_matmul");
    q4_0_f32_matmul<<<grid, minfer_launch_block("launch:q4_0_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q4_0_f32_matmul", "q4_0_f32_matmul");
}

void launch_q8_0_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), 1, 1);
    minfer_launch_prelude("launch:q8_0_f32_matmul", "q8_0_f32_matmul");
    q8_0_f32_matmul<<<grid, minfer_launch_block("launch:q8_0_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q8_0_f32_matmul", "q8_0_f32_matmul");
}

void launch_q4_1_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), 1, 1);
    minfer_launch_prelude("launch:q4_1_f32_matmul", "q4_1_f32_matmul");
    q4_1_f32_matmul<<<grid, minfer_launch_block("launch:q4_1_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q4_1_f32_matmul", "q4_1_f32_matmul");
}

void launch_q4_k_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 2, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    minfer_launch_prelude("launch:q4_k_f32_matmul", "q4_k_f32_matmul");
    q4_k_f32_matmul<<<grid, minfer_launch_block("launch:q4_k_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q4_k_f32_matmul", "q4_k_f32_matmul");
}

void launch_q5_1_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), 1, 1);
    minfer_launch_prelude("launch:q5_1_f32_matmul", "q5_1_f32_matmul");
    q5_1_f32_matmul<<<grid, minfer_launch_block("launch:q5_1_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q5_1_f32_matmul", "q5_1_f32_matmul");
}

void launch_q5_0_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), 1, 1);
    minfer_launch_prelude("launch:q5_0_f32_matmul", "q5_0_f32_matmul");
    q5_0_f32_matmul<<<grid, minfer_launch_block("launch:q5_0_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q5_0_f32_matmul", "q5_0_f32_matmul");
}

void launch_q5_k_f32_matmul(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 4, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    minfer_launch_prelude("launch:q5_k_f32_matmul", "q5_k_f32_matmul");
    q5_k_f32_matmul<<<grid, minfer_launch_block("launch:q5_k_f32_matmul", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q5_k_f32_matmul", "q5_k_f32_matmul");
}
}
