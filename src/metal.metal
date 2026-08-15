// Metal shaders for minfer — Q4_0 matmul + element-wise ops.

#include <metal_stdlib>
using namespace metal;

constant int Q4B = 18;
constant int Q41B = 20;
constant int Q5B = 22;

// Shared kernel launch parameters
constant short NW_Q = 32;
constant short NQ_Q = 16;
constant short QK   = 32;

// ─── Q4_0 × Q8_0 matrix multiplication (bit-exact with CPU) ───

constant int Q8B = 34;

kernel void kernel_q4_0_q8_0_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const uchar  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tid   [[thread_position_in_threadgroup]]
) {
    // Layout: 64 threads = 2 simdgroups × 32 lanes.
    // Each simdgroup computes NR0=4 consecutive output rows.
    // Each threadgroup therefore computes 8 output rows for one token.
    const int NR0 = 4;
    const int NSG = 2;
    const int NW  = 32;

    const int tiisg = (int)tid.x % NW;   // lane in simdgroup
    const int sgitg = (int)tid.x / NW;   // simdgroup in threadgroup
    const int t     = (int)tgpig.y;      // token index
    const int r0    = ((int)tgpig.x * NSG + sgitg) * NR0; // base output row

    if (t >= p[2] || r0 >= p[0]) return;

    const int nb  = p[1] / 32;
    const int q4s = nb * Q4B;
    const int q8s = nb * Q8B;

    device const uchar * xr = acts + t * q8s;

    float sumf[NR0];
    for (int row = 0; row < NR0; row++) sumf[row] = 0.0f;

    // Each lane handles every NW-th block, computing its 4 rows in lockstep.
    for (int b = tiisg; b < nb; b += NW) {
        // Q8_0 block is shared across the 4 rows handled by this simdgroup.
        device const half * xb = (device const half *)(xr + b * Q8B);
        float d8 = float(xb[0]);
        device const char * xq = (device const char *)(xb + 1);

        for (int row = 0; row < NR0; row++) {
            int o = r0 + row;
            if (o >= p[0]) break;

            device const uchar * wr = weights + o * q4s;
            device const half * wb = (device const half *)(wr + b * Q4B);
            float d4 = float(wb[0]);
            device const uchar * wq = (device const uchar *)(wb + 1);

            int bs = 0;
            for (int j = 0; j < 16; j++) {
                uchar byte = wq[j];
                bs += (int(byte & 0x0F) - 8) * int(xq[j])
                    + (int(byte >> 4) - 8) * int(xq[j + 16]);
            }
            sumf[row] += float(bs) * d4 * d8;
        }
    }

    // Reduce each row across the simdgroup and write.
    for (int row = 0; row < NR0; row++) {
        int o = r0 + row;
        if (o < p[0]) {
            float total = simd_sum(sumf[row]);
            if (tiisg == 0) {
                output[t * p[0] + o] = total;
            }
        }
    }
}

// ─── Q4_0 × Q8_0 prefill (multi-token) ──────────────────────
// Same layout as kernel_q4_0_q8_0_matmul but loops over all
// tokens within each threadgroup, reusing the weight rows.
// Grid: x = ceil(od/8), y = 1. TG = 64 threads.

kernel void kernel_q4_0_q8_0_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const uchar  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tid   [[thread_position_in_threadgroup]]
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int NW  = 32;

    const int tiisg = (int)tid.x % NW;
    const int sgitg = (int)tid.x / NW;
    const int r0    = ((int)tgpig.x * NSG + sgitg) * NR0;

    if (r0 >= p[0]) return;

    const int nb  = p[1] / 32;
    const int q4s = nb * Q4B;
    const int q8s = nb * Q8B;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * xr = acts + t * q8s;
        float sumf[NR0];
        for (int row = 0; row < NR0; row++) sumf[row] = 0.0f;

        for (int b = tiisg; b < nb; b += NW) {
            device const half * xb = (device const half *)(xr + b * Q8B);
            float d8 = float(xb[0]);
            device const char * xq = (device const char *)(xb + 1);

            for (int row = 0; row < NR0; row++) {
                int o = r0 + row;
                if (o >= p[0]) break;

                device const uchar * wr = weights + o * q4s;
                device const half * wb = (device const half *)(wr + b * Q4B);
                float d4 = float(wb[0]);
                device const uchar * wq = (device const uchar *)(wb + 1);

                int bs = 0;
                for (int j = 0; j < 16; j++) {
                    uchar byte = wq[j];
                    bs += (int(byte & 0x0F) - 8) * int(xq[j])
                        + (int(byte >> 4) - 8) * int(xq[j + 16]);
                }
                sumf[row] += float(bs) * d4 * d8;
            }
        }

        for (int row = 0; row < NR0; row++) {
            int o = r0 + row;
            if (o < p[0]) {
                float total = simd_sum(sumf[row]);
                if (tiisg == 0) output[t * p[0] + o] = total;
            }
        }
    }
}

inline float block_q5_0_dot_y(device const uchar * block, float sumy, thread float * yl, int il) {
    device const half   * hptr = (device const half *)block;
    // Q5_0: d(2B) + qh(4B) + qs(16B) = 22B. hptr+3 = skip d+qh → start of qs.
    device const ushort * qs   = (device const ushort *)(hptr + 3) + il / 2;
    float d = float(hptr[0]);
    // Read qh byte-by-byte: offset 2 is 2-byte aligned (unaligned uint32_t is UB in Metal)
    uint32_t qh = (uint32_t)block[2] | ((uint32_t)block[3] << 8) | ((uint32_t)block[4] << 16) | ((uint32_t)block[5] << 24);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        ushort v = qs[i / 2];
        acc0 += yl[i + 0] * ((v & 0x000F) | ((qh >> (i + 0 + il       )) << 4  & 0x00010));
        acc1 += yl[i + 1] * ((v & 0x0F00) | ((qh >> (i + 1 + il       )) << 12 & 0x01000));
        acc2 += yl[i + 8] * ((v & 0x00F0) | ((qh >> (i + 0 + il + 16)) << 8  & 0x00100));
        acc3 += yl[i + 9] * ((v & 0xF000) | ((qh >> (i + 1 + il + 16)) << 16 & 0x10000));
    }
    return d * (sumy * -16.0f + acc0 + acc1 + acc2 + acc3);
}

kernel void kernel_q5_0_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= p[2]) return;

    const int q5s = nb * Q5B;
    device const uchar * ax0 = weights + (r0 + 0) * q5s;
    device const uchar * ax1 = weights + (r0 + 1) * q5s;
    device const uchar * ax2 = weights + (r0 + 2) * q5s;
    device const uchar * ax3 = weights + (r0 + 3) * q5s;
    device const float  * y  = acts + t * p[1];

    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
    float yl[16];
    device const float * yb = y + ix * QK + il;

    for (int ib = ix; ib < nb; ib += NQ_Q) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sy = sumy0 + sumy1;
        if (r0 + 0 < p[0]) sumf0 += block_q5_0_dot_y(ax0 + ib * Q5B, sy, yl, il);
        if (r0 + 1 < p[0]) sumf1 += block_q5_0_dot_y(ax1 + ib * Q5B, sy, yl, il);
        if (r0 + 2 < p[0]) sumf2 += block_q5_0_dot_y(ax2 + ib * Q5B, sy, yl, il);
        if (r0 + 3 < p[0]) sumf3 += block_q5_0_dot_y(ax3 + ib * Q5B, sy, yl, il);
        yb += QK * NQ_Q;
    }

    sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
    sumf2 = simd_sum(sumf2); sumf3 = simd_sum(sumf3);
    if (tiisg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
        if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
    }
}

kernel void kernel_q5_0_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    const int q5s = nb * Q5B;
    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * ax0 = weights + (r0 + 0) * q5s;
        device const uchar * ax1 = weights + (r0 + 1) * q5s;
        device const uchar * ax2 = weights + (r0 + 2) * q5s;
        device const uchar * ax3 = weights + (r0 + 3) * q5s;
        device const float  * y  = acts + t * p[1];

        float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
        float yl[16];
        device const float * yb = y + ix * QK + il;

        for (int ib = ix; ib < nb; ib += NQ_Q) {
            float sumy0 = 0.0f, sumy1 = 0.0f;
            for (short i = 0; i < 8; i += 2) {
                sumy0 += yb[i + 0] + yb[i + 1];
                yl[i + 0] = yb[i + 0];
                yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
                sumy1 += yb[i + 16] + yb[i + 17];
                yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
                yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
            }
            float sy = sumy0 + sumy1;
            if (r0 + 0 < p[0]) sumf0 += block_q5_0_dot_y(ax0 + ib * Q5B, sy, yl, il);
            if (r0 + 1 < p[0]) sumf1 += block_q5_0_dot_y(ax1 + ib * Q5B, sy, yl, il);
            if (r0 + 2 < p[0]) sumf2 += block_q5_0_dot_y(ax2 + ib * Q5B, sy, yl, il);
            if (r0 + 3 < p[0]) sumf3 += block_q5_0_dot_y(ax3 + ib * Q5B, sy, yl, il);
            yb += QK * NQ_Q;
        }

        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        sumf2 = simd_sum(sumf2); sumf3 = simd_sum(sumf3);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
            if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
            if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
        }
    }
}

inline float block_q4_0_dot_y(device const uchar * block, float sumy, thread float * yl, int il) {
    device const half   * hptr = (device const half *)block;
    device const ushort * qs   = (device const ushort *)(hptr + 1) + il / 2;
    float d = float(hptr[0]);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        ushort v = qs[i / 2];
        acc0 += yl[i + 0] * float(v & 0x000F);
        acc1 += yl[i + 1] * float(v & 0x0F00);
        acc2 += yl[i + 8] * float(v & 0x00F0);
        acc3 += yl[i + 9] * float(v & 0xF000);
    }
    return d * (sumy * -8.0f + acc0 + acc1 + acc2 + acc3);
}

kernel void kernel_q4_0_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= p[2]) return;

    const int q4s = nb * Q4B;
    device const uchar * ax0 = weights + (r0 + 0) * q4s;
    device const uchar * ax1 = weights + (r0 + 1) * q4s;
    device const uchar * ax2 = weights + (r0 + 2) * q4s;
    device const uchar * ax3 = weights + (r0 + 3) * q4s;
    device const float  * y  = acts + t * p[1];

    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
    float yl[16];
    device const float * yb = y + ix * QK + il;

    for (int ib = ix; ib < nb; ib += NQ_Q) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sy = sumy0 + sumy1;
        if (r0 + 0 < p[0]) sumf0 += block_q4_0_dot_y(ax0 + ib * Q4B, sy, yl, il);
        if (r0 + 1 < p[0]) sumf1 += block_q4_0_dot_y(ax1 + ib * Q4B, sy, yl, il);
        if (r0 + 2 < p[0]) sumf2 += block_q4_0_dot_y(ax2 + ib * Q4B, sy, yl, il);
        if (r0 + 3 < p[0]) sumf3 += block_q4_0_dot_y(ax3 + ib * Q4B, sy, yl, il);
        yb += QK * NQ_Q;
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    sumf2 = simd_sum(sumf2);
    sumf3 = simd_sum(sumf3);
    if (tiisg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
        if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
    }
}

// ─── Q5_1 × f32 matrix multiplication ────────────────────────
// Q5_1: d(f16,2) + m(f16,2) + qh(u32,4) + qs(u8,16) = 24 bytes/32 elem.
// weight = d * unsigned_5bit + m
// qh at offset 4 (4-byte aligned — safe uint32 read)

inline float block_q5_1_dot_y(device const uchar * block, float sumy, thread float * yl, int il) {
    device const half   * hptr = (device const half *)block;
    // Q5_1: d(2B) + m(2B) + qh(4B) + qs(16B) = 24B. hptr+4 = skip d+m+qh → start of qs.
    device const ushort * qs   = (device const ushort *)(hptr + 4) + il / 2;
    float d = float(hptr[0]);
    float m = float(hptr[1]);

    // Read qh byte-by-byte (unaligned access is UB on ARM/Metal)
    uint32_t qh = (uint32_t)block[4] | ((uint32_t)block[5] << 8) | ((uint32_t)block[6] << 16) | ((uint32_t)block[7] << 24);

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        ushort v = qs[i / 2];
        acc0 += yl[i + 0] * ((v & 0x000F) | ((qh >> (i + 0 + il       )) << 4  & 0x00010));
        acc1 += yl[i + 1] * ((v & 0x0F00) | ((qh >> (i + 1 + il       )) << 12 & 0x01000));
        acc2 += yl[i + 8] * ((v & 0x00F0) | ((qh >> (i + 0 + il + 16)) << 8  & 0x00100));
        acc3 += yl[i + 9] * ((v & 0xF000) | ((qh >> (i + 1 + il + 16)) << 16 & 0x10000));
    }
    return d * (acc0 + acc1 + acc2 + acc3) + sumy * m;
}

kernel void kernel_q5_1_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= p[2]) return;

    const int q5s = nb * 24;
    device const uchar * ax0 = weights + (r0 + 0) * q5s;
    device const uchar * ax1 = weights + (r0 + 1) * q5s;
    device const uchar * ax2 = weights + (r0 + 2) * q5s;
    device const uchar * ax3 = weights + (r0 + 3) * q5s;
    device const float  * y  = acts + t * p[1];

    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
    float yl[16];
    device const float * yb = y + ix * QK + il;

    for (int ib = ix; ib < nb; ib += NQ_Q) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sy = sumy0 + sumy1;
        if (r0 + 0 < p[0]) sumf0 += block_q5_1_dot_y(ax0 + ib * 24, sy, yl, il);
        if (r0 + 1 < p[0]) sumf1 += block_q5_1_dot_y(ax1 + ib * 24, sy, yl, il);
        if (r0 + 2 < p[0]) sumf2 += block_q5_1_dot_y(ax2 + ib * 24, sy, yl, il);
        if (r0 + 3 < p[0]) sumf3 += block_q5_1_dot_y(ax3 + ib * 24, sy, yl, il);
        yb += QK * NQ_Q;
    }

    sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
    sumf2 = simd_sum(sumf2); sumf3 = simd_sum(sumf3);
    if (tiisg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
        if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
    }
}

kernel void kernel_q5_1_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    const int q5s = nb * 24;
    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * ax0 = weights + (r0 + 0) * q5s;
        device const uchar * ax1 = weights + (r0 + 1) * q5s;
        device const uchar * ax2 = weights + (r0 + 2) * q5s;
        device const uchar * ax3 = weights + (r0 + 3) * q5s;
        device const float  * y  = acts + t * p[1];

        float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
        float yl[16];
        device const float * yb = y + ix * QK + il;

        for (int ib = ix; ib < nb; ib += NQ_Q) {
            float sumy0 = 0.0f, sumy1 = 0.0f;
            for (short i = 0; i < 8; i += 2) {
                sumy0 += yb[i + 0] + yb[i + 1];
                yl[i + 0] = yb[i + 0];
                yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
                sumy1 += yb[i + 16] + yb[i + 17];
                yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
                yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
            }
            float sy = sumy0 + sumy1;
            if (r0 + 0 < p[0]) sumf0 += block_q5_1_dot_y(ax0 + ib * 24, sy, yl, il);
            if (r0 + 1 < p[0]) sumf1 += block_q5_1_dot_y(ax1 + ib * 24, sy, yl, il);
            if (r0 + 2 < p[0]) sumf2 += block_q5_1_dot_y(ax2 + ib * 24, sy, yl, il);
            if (r0 + 3 < p[0]) sumf3 += block_q5_1_dot_y(ax3 + ib * 24, sy, yl, il);
            yb += QK * NQ_Q;
        }

        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        sumf2 = simd_sum(sumf2); sumf3 = simd_sum(sumf3);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
            if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
            if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
        }
    }
}

// ─── Q4_0 × f32 prefill (multi-token) ───────────────────────
// Same as kernel_q4_0_f32_matmul but loops over all tokens
// within each threadgroup. Grid: x = ceil(od / (NR0*NSG)), y = 1.

kernel void kernel_q4_0_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    const int q4s = nb * Q4B;
    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * ax0 = weights + (r0 + 0) * q4s;
        device const uchar * ax1 = weights + (r0 + 1) * q4s;
        device const uchar * ax2 = weights + (r0 + 2) * q4s;
        device const uchar * ax3 = weights + (r0 + 3) * q4s;
        device const float  * y  = acts + t * p[1];

        float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
        float yl[16];
        device const float * yb = y + ix * QK + il;

        for (int ib = ix; ib < nb; ib += NQ_Q) {
            float sumy0 = 0.0f, sumy1 = 0.0f;
            for (short i = 0; i < 8; i += 2) {
                sumy0 += yb[i + 0] + yb[i + 1];
                yl[i + 0] = yb[i + 0];
                yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
                sumy1 += yb[i + 16] + yb[i + 17];
                yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
                yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
            }
            float sy = sumy0 + sumy1;
            if (r0 + 0 < p[0]) sumf0 += block_q4_0_dot_y(ax0 + ib * Q4B, sy, yl, il);
            if (r0 + 1 < p[0]) sumf1 += block_q4_0_dot_y(ax1 + ib * Q4B, sy, yl, il);
            if (r0 + 2 < p[0]) sumf2 += block_q4_0_dot_y(ax2 + ib * Q4B, sy, yl, il);
            if (r0 + 3 < p[0]) sumf3 += block_q4_0_dot_y(ax3 + ib * Q4B, sy, yl, il);
            yb += QK * NQ_Q;
        }

        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        sumf2 = simd_sum(sumf2); sumf3 = simd_sum(sumf3);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
            if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
            if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
        }
    }
}

// ─── Q4_0 × f32 GEMM (simdgroup_matrix, prefill nt > 1) ─────// Faithful port of llama.cpp's kernel_mul_mm_q4_0_f32 (legacy simdgroup path):
//   - dequantize_q4_0: uint16 reads + float4x4 SIMD
//   - A staged transposed into sa; B staged via float2x4 vector stores
//   - simdgroup_half8x8 inputs -> simdgroup_float8x8 accumulators
//   - mc += mb × ma (llama's exact order for the transposed-A layout)
// A = weights (od × id Q4_0), B = acts (nt × id f32), C = out (nt × od).
// M = od, K = id, N = nt. Threadgroup 128 threads (4 simdgroups), 64×32 tile.
// Grid: x = ceil(nt/32), y = ceil(od/64). smem = 8192 B (sa/sb + bc_out temp).

// Dequant 16 elements of a Q4_0 block into a float4x4, matching llama's layout.
inline void dequant_q4_0_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const half   * dh  = (device const half *)blkp;
    device const ushort * qs  = (device const ushort *)(dh + 1);
    const float d  = float(dh[0]);
    const float d1 = il ? (d / 16.0f) : d;
    const float d2 = d1 / 256.0f;
    const float md = -8.0f * d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = ushort(mask0 << 8);
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * float(qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * float(qs[i] & mask1) + md;
    }
    reg = reg_f;
}

// ─── Dequant helpers for the non-Q4_0 GEMM kernels (faithful llama.cpp ports) ─
// Each produces a float4x4 of 16 f32 for one 32-element block half (il=0/1),
// except Q6_K which produces a 32-element sub-block of a 256-element super-block
// (il=0..7). Layout matches llama's block_* structs / GGUF tensor bytes.

static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q) {
    return j < 4 ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

// Q4_1: d(half,2) + m(half,2) + qs(u16*8,16) = 20 B / 32 elems. Unsigned + m.
inline void dequant_q4_1_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint16_t * qs = (device const uint16_t *)(blkp + 4);
    const float d  = float(*(device const half *)blkp);
    const float m  = float(*(device const half *)(blkp + 2));
    const float d1 = il ? (d / 16.0f) : d;
    const float d2 = d1 / 256.0f;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = ushort(mask0 << 8);
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = (float(qs[i] & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = (float(qs[i] & mask1) * d2) + m;
    }
    reg = reg_f;
}

// Q5_0: d(half,2) + qh(u32,4) + qs(u16*8,16) = 22 B / 32 elems. Signed (val - 16).
inline void dequant_q5_0_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint16_t * qs = (device const uint16_t *)(blkp + 6);
    const float d  = float(*(device const half *)blkp);
    const float md = -16.0f * d;
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = (uint32_t)blkp[2] | ((uint32_t)blkp[3] << 8) | ((uint32_t)blkp[4] << 16) | ((uint32_t)blkp[5] << 24);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * (float)x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * (float)x1 + md;
    }
    reg = reg_f;
}

// Q5_1: d(half,2) + m(half,2) + qh(u32,4) + qs(u16*8,16) = 24 B / 32 elems. Unsigned + m.
inline void dequant_q5_1_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint16_t * qs = (device const uint16_t *)(blkp + 8);
    const float d = float(*(device const half *)blkp);
    const float m = float(*(device const half *)(blkp + 2));
    const ushort mask = il ? 0x00F0 : 0x000F;
    const uint32_t qh = (uint32_t)blkp[4] | ((uint32_t)blkp[5] << 8) | ((uint32_t)blkp[6] << 16) | ((uint32_t)blkp[7] << 24);
    const int x_mv = il ? 4 : 0;
    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ? 0 : 4;
    float4x4 reg_f;
    for (int i = 0; i < 8; i++) {
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i)) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;
        const int32_t x0 = ((((qs[i]) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);
        reg_f[i/2][2*(i%2) + 0] = d * (float)x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * (float)x1 + m;
    }
    reg = reg_f;
}

// Q8_0: d(half,2) + qs(int8*32,32) = 34 B / 32 elems.
inline void dequant_q8_0_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const int8_t * qs = (device const int8_t *)(blkp + 2);
    const float d = float(*(device const half *)blkp);
    float4x4 reg_f;
    for (int i = 0; i < 16; i++) {
        reg_f[i/4][i%4] = (float)qs[i + 16*il] * d;
    }
    reg = reg_f;
}

// Q6_K: d(half,2) LAST + ql(u8,128) + qh(u8,64) + scales(i8,16) = 210 B / 256 elems.
// il = 0..7 = which 32-element sub-block of the super-block.
inline void dequant_q6_k_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    const float d_all = float(*(device const half *)(blkp + 208));
    device const uint16_t * ql = (device const uint16_t *)blkp;
    device const uint16_t * qh = (device const uint16_t *)(blkp + 128);
    device const int8_t * scales = (device const int8_t *)(blkp + 192);

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = (float)scales[(il%2) + 2*((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    float4x4 reg_f;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg_f[i][0] = dl0 * ((float)(q & 0xFF))       - ml;
        reg_f[i][1] = dl1 * ((float)(q & 0xFF00))     - ml;
        reg_f[i][2] = dl2 * ((float)(q & 0xFF0000))   - ml;
        reg_f[i][3] = dl3 * ((float)(q & 0xFF000000)) - ml;
    }
    reg = reg_f;
}

// Q4_K: d(2) + dmin(2) + scales(12) + qs(128) = 144 B / 256 elems. il = 0..15
// (16-element il-halves). val = dl * nibble - ml, dl = d*sc, ml = dmin*scm.
inline void dequant_q4_k_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uchar * q = blkp + 16;   // qs
    const short is = (il/4) * 2;
    q = q + (il/4) * 32 + 16 * (il&1);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, blkp + 4);  // scales
    const float d   = il < 2 ? float(*(device const half *)blkp) : float(*(device const half *)blkp) / 16.0f;
    const float min = float(*(device const half *)(blkp + 2));
    const float dl = d * float(sc[0]);
    const float ml = min * float(sc[1]);
    const ushort mask = il < 2 ? 0x0F : 0xF0;
    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        reg_f[i/4][i%4] = dl * float(q[i] & mask) - ml;
    }
    reg = reg_f;
}

// Q5_K: d(2) + dmin(2) + scales(12) + qh(32) + qs(128) = 176 B / 256 elems.
// il = 0..15; qh byte = sub-block high bits. val = dl*(nibble + qh_bit*16|256) - ml.
inline void dequant_q5_k_16(device const uchar * blkp, short il, thread float4x4 & reg) {
    device const uint8_t * q  = blkp + 48;   // qs
    device const uint8_t * qh = blkp + 16;   // qh
    const short is = (il/4) * 2;
    q  = q + 32 * (il/4) + 16 * (il&1);
    qh = qh + 16 * (il&1);
    const uint8_t ul = 1 << (il/2);
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2(is, il/2, blkp + 4);
    const float d   = il < 2 ? float(*(device const half *)blkp) : float(*(device const half *)blkp) / 16.0f;
    const float min = float(*(device const half *)(blkp + 2));
    const float dl = d * float(sc[0]);
    const float ml = min * float(sc[1]);
    const ushort mask  = il < 2 ? 0x0F : 0xF0;
    const float qh_val = il < 2 ? 16.0f : 256.0f;
    float4x4 reg_f;
    for (int i = 0; i < 16; ++i) {
        reg_f[i/4][i%4] = dl * (float(q[i] & mask) + (qh[i] & ul ? qh_val : 0.0f)) - ml;
    }
    reg = reg_f;
}

kernel void kernel_q4_0_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;   // NK/16
    constexpr int NL1 = 4;   // NK/8

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    // clamp thread row/col so the staging pointer stays in bounds
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // zero the staging tiles so out-of-range rows (partial tiles) stay 0,
    // not stale/NaN threadgroup memory from previous dispatches
    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        // === Stage A: dequant Q4_0 weights into sa (llama transposed layout) ===
        thread float4x4 temp_a;
        dequant_q4_0_16(weights + (r0 + lr0) * nblk * Q4B + (loop_k/32) * Q4B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        // === Stage B: f32 activations into sb (scalar, equivalent to llama's float2x4 store) ===
        // llama writes to the TRUE (unclamped) sb position, reading from the clamped row.
        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;                 // K chunk
        const short by = (tiitg/NL1)/8;               // N group (raw, fills OOB rows w/ clamp data)
        const short bly = (tiitg/NL1)%8;              // N sub   (raw)
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Matrix multiply (4 K sub-tiles of 8) ===
        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // === Store C (M×N) to output (N×M = [p[2]][p[0]]) ===
    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        // full tile: direct transposed store
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        // partial tile (bc_out): per-simdgroup temp_str + float4 copy
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q4_1 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure as kernel_q4_0_mm_f32; Q4_1: d(2) + m(2) + qs(16) = 20 B.
kernel void kernel_q4_1_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q41B = 20;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q4_1_16(weights + (r0 + lr0) * nblk * Q41B + (loop_k/32) * Q41B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q8_0 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure as kernel_q4_0_mm_f32; Q8_0: d(half,2) + qs(int8*32,32) = 34 B.
kernel void kernel_q8_0_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q8B = 34;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q8_0_16(weights + (r0 + lr0) * nblk * Q8B + (loop_k/32) * Q8B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q5_0 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure; Q5_0: d(2) + qh(4) + qs(16) = 22 B. Signed (val - 16).
kernel void kernel_q5_0_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q5B = 22;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q5_0_16(weights + (r0 + lr0) * nblk * Q5B + (loop_k/32) * Q5B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q5_1 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure; Q5_1: d(2) + m(2) + qh(4) + qs(16) = 24 B. Unsigned + m.
kernel void kernel_q5_1_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q51B = 24;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q5_1_16(weights + (r0 + lr0) * nblk * Q51B + (loop_k/32) * Q51B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q6_K × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same 64×32-tile structure as Q4_0, but the weights are 256-element super-blocks
// (Q6KB=210). Each 32-elem K step spans 2 "il halves" of the super-block (il =
// (loop_k%256)/16 + il0, il0 = 0/1), dequantized by dequant_q6_k_16.
kernel void kernel_q6_k_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q6KB = 210;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 256;   // super-blocks per row

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        // super-block at loop_k/256; the 32-elem step spans 2 il-halves
        dequant_q6_k_16(weights + (r0 + lr0) * nblk * Q6KB + (loop_k/256) * Q6KB,
                        ((loop_k % 256) / 16) + il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q4_1 × f32 matrix multiplication (simdgroup-cooperative) ──
// Same structure as Q4_0 but with (d, m, qs) block layout. Dequant: val = q * d + m.

inline float block_q4_1_dot_y(device const uchar * block, float sumy, thread float * yl, int il) {
    device const half   * hptr = (device const half *)block;
    device const ushort * qs   = (device const ushort *)(hptr + 2) + il / 2;
    float d = float(hptr[0]);
    float m = float(hptr[1]);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int i = 0; i < 8; i += 2) {
        ushort v = qs[i / 2];
        acc0 += yl[i + 0] * float(v & 0x000F);
        acc1 += yl[i + 1] * float(v & 0x0F00);
        acc2 += yl[i + 8] * float(v & 0x00F0);
        acc3 += yl[i + 9] * float(v & 0xF000);
    }
    return d * (acc0 + acc1 + acc2 + acc3) + sumy * m;
}

kernel void kernel_q4_1_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = p[1] / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= p[2]) return;

    const int q41s = nb * Q41B;
    device const uchar * ax0 = weights + (r0 + 0) * q41s;
    device const uchar * ax1 = weights + (r0 + 1) * q41s;
    device const uchar * ax2 = weights + (r0 + 2) * q41s;
    device const uchar * ax3 = weights + (r0 + 3) * q41s;
    device const float  * y  = acts + t * p[1];

    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
    float yl[16];
    device const float * yb = y + ix * QK + il;

    for (int ib = ix; ib < nb; ib += NQ_Q) {
        float sumy0 = 0.0f, sumy1 = 0.0f;
        for (short i = 0; i < 8; i += 2) {
            sumy0 += yb[i + 0] + yb[i + 1];
            yl[i + 0] = yb[i + 0];
            yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
            sumy1 += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
            yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
        }
        float sy = sumy0 + sumy1;
        if (r0 + 0 < p[0]) sumf0 += block_q4_1_dot_y(ax0 + ib * Q41B, sy, yl, il);
        if (r0 + 1 < p[0]) sumf1 += block_q4_1_dot_y(ax1 + ib * Q41B, sy, yl, il);
        if (r0 + 2 < p[0]) sumf2 += block_q4_1_dot_y(ax2 + ib * Q41B, sy, yl, il);
        if (r0 + 3 < p[0]) sumf3 += block_q4_1_dot_y(ax3 + ib * Q41B, sy, yl, il);
        yb += QK * NQ_Q;
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    sumf2 = simd_sum(sumf2);
    sumf3 = simd_sum(sumf3);
    if (tiisg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
        if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
    }
}

// ─── Q4_1 × f32 prefill (multi-token) ───────────────────────

kernel void kernel_q4_1_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4; const short NSG = 2;
    const int nb = p[1] / QK;
    const int r0 = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int q41s = nb * Q41B;
    const short ix = (short)tiisg / (NW_Q / NQ_Q);
    const short il = ((short)tiisg % (NW_Q / NQ_Q)) * 8;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * ax0 = weights + (r0 + 0) * q41s;
        device const uchar * ax1 = weights + (r0 + 1) * q41s;
        device const uchar * ax2 = weights + (r0 + 2) * q41s;
        device const uchar * ax3 = weights + (r0 + 3) * q41s;
        device const float  * y  = acts + t * p[1];
        float sumf0 = 0.0f, sumf1 = 0.0f, sumf2 = 0.0f, sumf3 = 0.0f;
        float yl[16]; device const float * yb = y + ix * QK + il;
        for (int ib = ix; ib < nb; ib += NQ_Q) {
            float sumy0 = 0.0f, sumy1 = 0.0f;
            for (short i = 0; i < 8; i += 2) {
                sumy0 += yb[i + 0] + yb[i + 1];
                yl[i + 0] = yb[i + 0]; yl[i + 1] = yb[i + 1] * (1.0f / 256.0f);
                sumy1 += yb[i + 16] + yb[i + 17];
                yl[i + 8] = yb[i + 16] * (1.0f / 16.0f);
                yl[i + 9] = yb[i + 17] * (1.0f / 4096.0f);
            }
            float sy = sumy0 + sumy1;
            if (r0 + 0 < p[0]) sumf0 += block_q4_1_dot_y(ax0 + ib * Q41B, sy, yl, il);
            if (r0 + 1 < p[0]) sumf1 += block_q4_1_dot_y(ax1 + ib * Q41B, sy, yl, il);
            if (r0 + 2 < p[0]) sumf2 += block_q4_1_dot_y(ax2 + ib * Q41B, sy, yl, il);
            if (r0 + 3 < p[0]) sumf3 += block_q4_1_dot_y(ax3 + ib * Q41B, sy, yl, il);
            yb += QK * NQ_Q;
        }
        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        sumf2 = simd_sum(sumf2); sumf3 = simd_sum(sumf3);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
            if (r0 + 2 < p[0]) output[t * p[0] + r0 + 2] = sumf2;
            if (r0 + 3 < p[0]) output[t * p[0] + r0 + 3] = sumf3;
        }
    }
}

inline void get_scale_min_k4(int j, device const uchar * q, thread uchar & d, thread uchar & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >> 4)  | ((q[j]   >> 6) << 4);
    }
}

// ─── Q5_K × f32 matrix multiplication (simdgroup-cooperative) ──
// Q5_K super-block: 256 elements = 8 sub-blocks × 32.
// Block layout (176 bytes): half d, half dmin, uchar scales[12], uchar qh[32], uchar qs[128].
// Dequant: val = d * scale[sub] * u - dmin * min[sub], u = 5-bit unsigned.
// qh layout: element (sub s, pos p) high bit = qh[p] bit s.
// qs layout: byte (s/2)*32 + p — sub s even -> lo nibble, s odd -> hi nibble.
// NR0=2 rows per simdgroup, NSG=2 simdgroups per threadgroup => 64 threads.
// Grid: x = ceil(od / (NR0*NSG)), y = nt, TG = (64, 1, 1).

kernel void kernel_q5_k_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256;
    const int Q5KB = 176;
    const short NR0 = 2;
    const short NSG = 2;
    const short NW  = 32;

    const int nbe = p[1] / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= p[2]) return;

    const int row_stride = nbe * Q5KB;
    device const uchar * w0 = weights + (r0 + 0) * row_stride;
    device const uchar * w1 = weights + (r0 + 1) * row_stride;
    device const float  * y  = acts + t * p[1];

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (int ib = (int)tiisg; ib < nbe; ib += NW) {
        device const uchar * blk0 = w0 + ib * Q5KB;
        device const uchar * blk1 = w1 + ib * Q5KB;

        float bd0  = float(*(device const half *)(blk0 + 0));
        float bm0  = float(*(device const half *)(blk0 + 2));
        float bd1  = float(*(device const half *)(blk1 + 0));
        float bm1  = float(*(device const half *)(blk1 + 2));
        device const uchar * sc0 = blk0 + 4;
        device const uchar * sc1 = blk1 + 4;
        device const uchar * qh0 = blk0 + 16;
        device const uchar * qh1 = blk1 + 16;
        device const uchar * qs0 = blk0 + 48;
        device const uchar * qs1 = blk1 + 48;
        device const float * yb = y + ib * QKK;

        uchar sc0_s[8], sc0_m[8], sc1_s[8], sc1_m[8];
        for (int j = 0; j < 8; j++) {
            get_scale_min_k4(j, sc0, sc0_s[j], sc0_m[j]);
            get_scale_min_k4(j, sc1, sc1_s[j], sc1_m[j]);
        }

        // Deinterleave qs nibbles: 4 chunks of 32 bytes, each covering 2 subblocks
        uchar nb0[256], nb1[256];
        for (int ci = 0; ci < 4; ci++) {
            device const uchar * ch0 = qs0 + ci * 32;
            device const uchar * ch1 = qs1 + ci * 32;
            for (int l = 0; l < 32; l++) {
                nb0[(2*ci)*32 + l] = ch0[l] & 0x0F;
                nb0[(2*ci+1)*32 + l] = ch0[l] >> 4;
                nb1[(2*ci)*32 + l] = ch1[l] & 0x0F;
                nb1[(2*ci+1)*32 + l] = ch1[l] >> 4;
            }
        }

        for (int s = 0; s < 8; s++) {
            float dsc0 = bd0 * sc0_s[s]; float dmn0 = bm0 * sc0_m[s];
            float dsc1 = bd1 * sc1_s[s]; float dmn1 = bm1 * sc1_m[s];

            device const float * ys = yb + s * 32;

            float acc0 = 0.0f, acc1 = 0.0f, sumy = 0.0f;
            for (int k = 0; k < 32; k++) {
                // high bit: element (sub s, pos k) -> qh[k] bit s
                int u0 = nb0[s*32 + k] | (((qh0[k] >> s) & 1) << 4);
                int u1 = nb1[s*32 + k] | (((qh1[k] >> s) & 1) << 4);
                float yv = ys[k];
                acc0 += (float)u0 * yv;
                acc1 += (float)u1 * yv;
                sumy += yv;
            }
            sumf0 += dsc0 * acc0 - dmn0 * sumy;
            sumf1 += dsc1 * acc1 - dmn1 * sumy;
        }
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    if (tiisg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
    }
}

// ─── Q5_K × f32 prefill (multi-token) ───────────────────────

kernel void kernel_q5_k_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256;
    const int Q5KB = 176;
    const short NR0 = 2;
    const short NSG = 2;
    const short NW  = 32;
    const int nbe = p[1] / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int row_stride = nbe * Q5KB;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * w0 = weights + (r0 + 0) * row_stride;
        device const uchar * w1 = weights + (r0 + 1) * row_stride;
        device const float  * y  = acts + t * p[1];
        float sumf0 = 0.0f, sumf1 = 0.0f;
        for (int ib = (int)tiisg; ib < nbe; ib += NW) {
            device const uchar * blk0 = w0 + ib * Q5KB;
            device const uchar * blk1 = w1 + ib * Q5KB;
            float bd0 = float(*(device const half *)(blk0 + 0));
            float bm0 = float(*(device const half *)(blk0 + 2));
            float bd1 = float(*(device const half *)(blk1 + 0));
            float bm1 = float(*(device const half *)(blk1 + 2));
            device const uchar * sc0 = blk0 + 4; device const uchar * sc1 = blk1 + 4;
            device const uchar * qh0 = blk0 + 16; device const uchar * qh1 = blk1 + 16;
            device const uchar * qs0 = blk0 + 48; device const uchar * qs1 = blk1 + 48;
            device const float * yb = y + ib * QKK;
            uchar sc0_s[8], sc0_m[8], sc1_s[8], sc1_m[8];
            for (int j = 0; j < 8; j++) {
                get_scale_min_k4(j, sc0, sc0_s[j], sc0_m[j]);
                get_scale_min_k4(j, sc1, sc1_s[j], sc1_m[j]);
            }
            uchar nb0[256], nb1[256];
            for (int ci = 0; ci < 4; ci++) {
                device const uchar * ch0 = qs0 + ci * 32;
                device const uchar * ch1 = qs1 + ci * 32;
                for (int l = 0; l < 32; l++) {
                    nb0[(2*ci)*32 + l] = ch0[l] & 0x0F;
                    nb0[(2*ci+1)*32 + l] = ch0[l] >> 4;
                    nb1[(2*ci)*32 + l] = ch1[l] & 0x0F;
                    nb1[(2*ci+1)*32 + l] = ch1[l] >> 4;
                }
            }
            for (int s = 0; s < 8; s++) {
                float dsc0 = bd0 * sc0_s[s]; float dmn0 = bm0 * sc0_m[s];
                float dsc1 = bd1 * sc1_s[s]; float dmn1 = bm1 * sc1_m[s];
                device const float * ys = yb + s * 32;
                float acc0 = 0.0f, acc1 = 0.0f, sumy = 0.0f;
                for (int k = 0; k < 32; k++) {
                    int u0 = nb0[s*32 + k] | (((qh0[k] >> s) & 1) << 4);
                    int u1 = nb1[s*32 + k] | (((qh1[k] >> s) & 1) << 4);
                    float yv = ys[k];
                    acc0 += (float)u0 * yv;
                    acc1 += (float)u1 * yv;
                    sumy += yv;
                }
                sumf0 += dsc0 * acc0 - dmn0 * sumy;
                sumf1 += dsc1 * acc1 - dmn1 * sumy;
            }
        }
        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        }
    }
}

// ─── Q4_K × f32 matrix multiplication (simdgroup-cooperative) ──
// Q4_K super-block: 256 elements = 8 sub-blocks × 32.
// Block layout (144 bytes): half d, half dmin, uchar scales[12], uchar qs[128].
// Dequant: val = d * scale[sub] * nibble - dmin * min[sub].
// NR0=2 rows per simdgroup, NSG=2 simdgroups per threadgroup => 64 threads.
// Grid: x = ceil(od / (NR0*NSG)), y = nt, TG = (64, 1, 1).

kernel void kernel_q4_k_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256;
    const int Q4KB = 144;
    const short NR0 = 2;
    const short NSG = 2;
    const short NW  = 32;

    const int nbe = p[1] / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= p[2]) return;

    const int row_stride = nbe * Q4KB;
    device const uchar * w0 = weights + (r0 + 0) * row_stride;
    device const uchar * w1 = weights + (r0 + 1) * row_stride;
    device const float  * y  = acts + t * p[1];

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (int ib = (int)tiisg; ib < nbe; ib += NW) {
        device const uchar * blk0 = w0 + ib * Q4KB;
        device const uchar * blk1 = w1 + ib * Q4KB;

        float bd0  = float(*(device const half *)(blk0 + 0));
        float bm0  = float(*(device const half *)(blk0 + 2));
        float bd1  = float(*(device const half *)(blk1 + 0));
        float bm1  = float(*(device const half *)(blk1 + 2));
        device const uchar * sc0 = blk0 + 4;
        device const uchar * sc1 = blk1 + 4;
        device const uchar * qs0 = blk0 + 16;
        device const uchar * qs1 = blk1 + 16;
        device const float * yb = y + ib * QKK;

         uchar sc0_s[8], sc0_m[8], sc1_s[8], sc1_m[8];
         for (int j = 0; j < 8; j++) {
             get_scale_min_k4(j, sc0, sc0_s[j], sc0_m[j]);
             get_scale_min_k4(j, sc1, sc1_s[j], sc1_m[j]);
         }

         // Deinterleave qs nibbles: 4 chunks of 32 bytes, each covering 2 subblocks
         // chunk[l] lo nibble → sub 2*chunk, elem l
         // chunk[l] hi nibble → sub 2*chunk+1, elem l
         uchar nb0[256], nb1[256];
         for (int ci = 0; ci < 4; ci++) {
             device const uchar * ch0 = qs0 + ci * 32;
             device const uchar * ch1 = qs1 + ci * 32;
             for (int l = 0; l < 32; l++) {
                 nb0[(2*ci)*32 + l] = ch0[l] & 0x0F;
                 nb0[(2*ci+1)*32 + l] = ch0[l] >> 4;
                 nb1[(2*ci)*32 + l] = ch1[l] & 0x0F;
                 nb1[(2*ci+1)*32 + l] = ch1[l] >> 4;
             }
         }

         for (int s = 0; s < 8; s++) {
             float dsc0 = bd0 * sc0_s[s]; float dmn0 = bm0 * sc0_m[s];
             float dsc1 = bd1 * sc1_s[s]; float dmn1 = bm1 * sc1_m[s];

             device const float  * ys = yb + s * 32;

             float acc0 = 0.0f, acc1 = 0.0f, sumy = 0.0f;
             for (int k = 0; k < 32; k++) {
                 float yv = ys[k];
                 acc0 += (float)nb0[s * 32 + k] * yv;
                 acc1 += (float)nb1[s * 32 + k] * yv;
                 sumy += yv;
             }
             sumf0 += dsc0 * acc0 - dmn0 * sumy;
             sumf1 += dsc1 * acc1 - dmn1 * sumy;
         }
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    if (tiisg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
    }
}

// ─── Q4_K × f32 prefill (multi-token) ───────────────────────

kernel void kernel_q4_k_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256; const int Q4KB = 144;
    const short NR0 = 2; const short NSG = 2; const short NW = 32;
    const int nbe = p[1] / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int row_stride = nbe * Q4KB;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * w0 = weights + (r0 + 0) * row_stride;
        device const uchar * w1 = weights + (r0 + 1) * row_stride;
        device const float  * y  = acts + t * p[1];
        float sumf0 = 0.0f, sumf1 = 0.0f;
        for (int ib = (int)tiisg; ib < nbe; ib += NW) {
            device const uchar * blk0 = w0 + ib * Q4KB;
            device const uchar * blk1 = w1 + ib * Q4KB;
            float bd0 = float(*(device const half *)(blk0 + 0));
            float bm0 = float(*(device const half *)(blk0 + 2));
            float bd1 = float(*(device const half *)(blk1 + 0));
            float bm1 = float(*(device const half *)(blk1 + 2));
            device const uchar * sc0 = blk0 + 4; device const uchar * sc1 = blk1 + 4;
             device const uchar * qs0 = blk0 + 16; device const uchar * qs1 = blk1 + 16;
             device const float * yb = y + ib * QKK;
             uchar sc0_s[8], sc0_m[8], sc1_s[8], sc1_m[8];
             for (int j = 0; j < 8; j++) {
                 get_scale_min_k4(j, sc0, sc0_s[j], sc0_m[j]);
                 get_scale_min_k4(j, sc1, sc1_s[j], sc1_m[j]);
             }
             // Deinterleave qs nibbles: 4 chunks of 32 bytes, each covering 2 subblocks
             uchar nb0[256], nb1[256];
             for (int ci = 0; ci < 4; ci++) {
                 device const uchar * ch0 = qs0 + ci * 32;
                 device const uchar * ch1 = qs1 + ci * 32;
                 for (int l = 0; l < 32; l++) {
                     nb0[(2*ci)*32 + l] = ch0[l] & 0x0F;
                     nb0[(2*ci+1)*32 + l] = ch0[l] >> 4;
                     nb1[(2*ci)*32 + l] = ch1[l] & 0x0F;
                     nb1[(2*ci+1)*32 + l] = ch1[l] >> 4;
                 }
             }
             for (int s = 0; s < 8; s++) {
                 float dsc0 = bd0 * sc0_s[s]; float dmn0 = bm0 * sc0_m[s];
                 float dsc1 = bd1 * sc1_s[s]; float dmn1 = bm1 * sc1_m[s];
                 device const float  * ys = yb + s * 32;
                 float acc0 = 0.0f, acc1 = 0.0f, sumy = 0.0f;
                 for (int k = 0; k < 32; k++) {
                     float yv = ys[k];
                     acc0 += (float)nb0[s * 32 + k] * yv;
                     acc1 += (float)nb1[s * 32 + k] * yv;
                     sumy += yv;
                 }
                 sumf0 += dsc0 * acc0 - dmn0 * sumy;
                 sumf1 += dsc1 * acc1 - dmn1 * sumy;
             }
        }
        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        }
    }
}

// ─── Q6_K × f32 matrix multiplication (simdgroup-cooperative) ──
// Q6_K super-block: 256 elements = 16 sub-blocks × 16.
// Block layout (210 bytes): uchar ql[128], uchar qh[64], char scales[16], half d.
// Dequant: val = d * scales[sub] * ((low4 | (high2 << 4)) - 32).
// NR0=2, NSG=2, TG=64. Grid: x = ceil(od/4), y = nt.

kernel void kernel_q6_k_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Faithful port of llama's kernel_mul_mv_q6_K_f32_impl (stride-2 thread
    // layout, float4 sums). Dispatch: TG(32, nsg=2), grid_x = od/(nr0*nsg).
    constexpr short NR0 = 2;
    constexpr short NSG = 2;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = p[1] / 256;                       // super-blocks per row
    const int r0 = (int)tgpig.x;                     // grid tile (not row!)
    const int t  = (int)tgpig.y;
    if (t >= p[2]) return;

    const int first_row = (r0 * NSG + (int)sgitg) * NR0;

    const int row_stride = nb * 210;                 // Q6_K block bytes
    device const uchar * x0 = weights + first_row * row_stride;
    device const float * yy = acts + t * p[1];

    float sumf[NR0] = { 0.0f, 0.0f };

    float yl[16];

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;                       // 0 or 1
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    for (int i = ix; i < nb; i += 2) {
        device const uchar * blk = x0 + i * 210;
        device const uchar * q1 = blk + q_offset_l;
        device const uchar * q2 = q1 + 32;
        device const uchar * qh = blk + 128 + q_offset_h;
        device const int8_t  * sc = (device const int8_t *)(blk + 192) + is;
        device const half   * dh = (device const half *)(blk + 208);

        device const float * y = yy + i * 256 + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < NR0; ++row) {
            float4 sums = { 0.0f, 0.0f, 0.0f, 0.0f };

            for (short l = 0; l < 4; ++l) {
                sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
            }

            sumf[row] += float(dh[0]) * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
                                       + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));

            q1 += row_stride;
            q2 += row_stride;
            qh += row_stride;
            sc += row_stride;
            dh += row_stride / 2;
        }
    }

    for (int row = 0; row < NR0 && first_row + row < p[0]; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            output[t * p[0] + first_row + row] = sum_all;
        }
    }
}

// ─── Q6_K × f32 prefill (multi-token) ───────────────────────

kernel void kernel_q6_k_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256; const int Q6KB = 210;
    const short NR0 = 2; const short NSG = 2; const short NW = 32;
    const int nbe = p[1] / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int row_stride = nbe * Q6KB;

    for (int t = 0; t < p[2]; t++) {
        device const uchar * w0 = weights + (r0 + 0) * row_stride;
        device const uchar * w1 = weights + (r0 + 1) * row_stride;
        device const float  * y  = acts + t * p[1];
        float sumf0 = 0.0f, sumf1 = 0.0f;
        for (int ib = (int)tiisg; ib < nbe; ib += NW) {
            device const uchar * blk0 = w0 + ib * Q6KB;
            device const uchar * blk1 = w1 + ib * Q6KB;
            float bd0 = float(*(device const half *)(blk0 + 208));
            float bd1 = float(*(device const half *)(blk1 + 208));
            device const uchar * ql0 = blk0; device const uchar * ql1 = blk1;
            device const uchar * qh0 = blk0 + 128; device const uchar * qh1 = blk1 + 128;
            device const char  * sc0 = (device const char *)(blk0 + 192);
            device const char  * sc1 = (device const char *)(blk1 + 192);
            device const float * yb = y + ib * QKK;
            for (int n = 0; n < 2; n++) {
                for (int l = 0; l < 32; l++) {
                    int is = l / 16;
                    device const float * ys = yb + n * 128 + l;
                    int q0_0 = ((int)(ql0[l] & 0xF) | (((int)(qh0[l] >> 0) & 3) << 4)) - 32;
                    int q1_0 = ((int)(ql1[l] & 0xF) | (((int)(qh1[l] >> 0) & 3) << 4)) - 32;
                    int q0_1 = ((int)(ql0[l + 32] & 0xF) | (((int)(qh0[l] >> 2) & 3) << 4)) - 32;
                    int q1_1 = ((int)(ql1[l + 32] & 0xF) | (((int)(qh1[l] >> 2) & 3) << 4)) - 32;
                    int q0_2 = ((int)(ql0[l] >> 4) | (((int)(qh0[l] >> 4) & 3) << 4)) - 32;
                    int q1_2 = ((int)(ql1[l] >> 4) | (((int)(qh1[l] >> 4) & 3) << 4)) - 32;
                    int q0_3 = ((int)(ql0[l + 32] >> 4) | (((int)(qh0[l] >> 6) & 3) << 4)) - 32;
                    int q1_3 = ((int)(ql1[l + 32] >> 4) | (((int)(qh1[l] >> 6) & 3) << 4)) - 32;
                    int si = is + n * 8;
                    sumf0 += bd0 * float(sc0[si + 0]) * ys[0]  * float(q0_0)
                           + bd0 * float(sc0[si + 2]) * ys[32] * float(q0_1)
                           + bd0 * float(sc0[si + 4]) * ys[64] * float(q0_2)
                           + bd0 * float(sc0[si + 6]) * ys[96] * float(q0_3);
                    sumf1 += bd1 * float(sc1[si + 0]) * ys[0]  * float(q1_0)
                           + bd1 * float(sc1[si + 2]) * ys[32] * float(q1_1)
                           + bd1 * float(sc1[si + 4]) * ys[64] * float(q1_2)
                           + bd1 * float(sc1[si + 6]) * ys[96] * float(q1_3);
                }
                ql0 += 64; ql1 += 64;
                qh0 += 32; qh1 += 32;
            }
        }
        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        if (tiisg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = sumf0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = sumf1;
        }
    }
}

// ─── Q8_0 × f32 matrix multiplication (simdgroup-cooperative) ──
// Direct translation of llama.cpp kernel_mul_mv_q8_0_f32_impl.
// NR0=2 rows per simdgroup, NSG=4 simdgroups per threadgroup => 128 threads.
// Grid: x = ceil(od / NR0), y = nt, TG = (32, NSG, 1).
// All simdgroups cooperate on the same NR0 rows, partitioning the input dim.

kernel void kernel_q8_0_f32_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float  * shmem     [[threadgroup(0)]]
) {
    const short NR0 = 2;
    const short NSG = 4;
    const short NW  = 32;
    const short NQ  = 8;

    const int nb = p[1] / QK;
    const int r0 = (int)tgpig.x * NR0;
    const int t  = (int)tgpig.y;
    if (t >= p[2] || r0 >= p[0]) return;

    const int q8s = nb * Q8B;
    device const float * y = acts + t * p[1];

    device const uchar * ax0 = weights + (r0 + 0) * q8s;
    device const uchar * ax1 = weights + (r0 + 1) * q8s;

    const short ix = tiisg / (NW / NQ);          // 0..7
    const short il = tiisg % (NW / NQ);          // 0..3
    const int ib0 = sgitg * NQ + ix;

    threadgroup float * sh0 = shmem + 0 * NW;
    threadgroup float * sh1 = shmem + 1 * NW;
    sh0[tiisg] = 0.0f;
    sh1[tiisg] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sumf0 = 0.0f, sumf1 = 0.0f;
    device const float * yb = y + ib0 * QK + il * NQ;

    for (int ib = ib0; ib < nb; ib += NSG * NQ) {
        float yl[NQ];
        for (short i = 0; i < NQ; ++i) yl[i] = yb[i];

        device const char * qs0 = ((device const char *)((device const half *)(ax0 + ib * Q8B) + 1)) + il * NQ;
        device const char * qs1 = ((device const char *)((device const half *)(ax1 + ib * Q8B) + 1)) + il * NQ;

        float sumq0 = 0.0f, sumq1 = 0.0f;
        for (short i = 0; i < NQ; ++i) {
            sumq0 += qs0[i] * yl[i];
            sumq1 += qs1[i] * yl[i];
        }

        sumf0 += sumq0 * float(((device const half *)(ax0 + ib * Q8B))[0]);
        sumf1 += sumq1 * float(((device const half *)(ax1 + ib * Q8B))[0]);

        yb += NSG * NQ * QK;
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);

    if (tiisg == 0) {
        sh0[sgitg] = sumf0;
        sh1[sgitg] = sumf1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float tot0 = simd_sum(sh0[tiisg]);
    float tot1 = simd_sum(sh1[tiisg]);
    if (tiisg == 0 && sgitg == 0) {
        if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = tot0;
        if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = tot1;
    }
}

// ─── Q8_0 × f32 prefill (multi-token) ───────────────────────

kernel void kernel_q8_0_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float  * shmem     [[threadgroup(0)]]
) {
    const short NR0 = 2; const short NSG = 4; const short NW = 32; const short NQ = 8;
    const int nb = p[1] / QK;
    const int r0 = (int)tgpig.x * NR0;
    const int q8s = nb * Q8B;
    const short ix = tiisg / (NW / NQ);
    const short il = tiisg % (NW / NQ);
    const int ib0 = sgitg * NQ + ix;
    threadgroup float * sh0 = shmem + 0 * NW;
    threadgroup float * sh1 = shmem + 1 * NW;

    device const uchar * ax0 = weights + (r0 + 0) * q8s;
    device const uchar * ax1 = weights + (r0 + 1) * q8s;

    for (int t = 0; t < p[2]; t++) {
        device const float * y = acts + t * p[1];
        sh0[tiisg] = 0.0f; sh1[tiisg] = 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sumf0 = 0.0f, sumf1 = 0.0f;
        device const float * yb = y + ib0 * QK + il * NQ;
        for (int ib = ib0; ib < nb; ib += NSG * NQ) {
            float yl[NQ];
            for (short i = 0; i < NQ; ++i) yl[i] = yb[i];
            device const char * qs0 = ((device const char *)((device const half *)(ax0 + ib * Q8B) + 1)) + il * NQ;
            device const char * qs1 = ((device const char *)((device const half *)(ax1 + ib * Q8B) + 1)) + il * NQ;
            float sumq0 = 0.0f, sumq1 = 0.0f;
            for (short i = 0; i < NQ; ++i) { sumq0 += qs0[i] * yl[i]; sumq1 += qs1[i] * yl[i]; }
            sumf0 += sumq0 * float(((device const half *)(ax0 + ib * Q8B))[0]);
            sumf1 += sumq1 * float(((device const half *)(ax1 + ib * Q8B))[0]);
            yb += NSG * NQ * QK;
        }
        sumf0 = simd_sum(sumf0); sumf1 = simd_sum(sumf1);
        if (tiisg == 0) { sh0[sgitg] = sumf0; sh1[sgitg] = sumf1; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float tot0 = simd_sum(sh0[tiisg]);
        float tot1 = simd_sum(sh1[tiisg]);
        if (tiisg == 0 && sgitg == 0) {
            if (r0 + 0 < p[0]) output[t * p[0] + r0 + 0] = tot0;
            if (r0 + 1 < p[0]) output[t * p[0] + r0 + 1] = tot1;
        }
    }
}

// ─── Get rows (embedding lookup, Q4_0 → f32) ────────────────
// Reads rows from a quantized embedding table and dequantizes to f32.
// weights: [n_vocab][nb * Q4B], ids: [nt], dst: [nt][ne].

kernel void kernel_get_rows_q4_0(
    device const uchar  * weights [[buffer(0)]],
    device const int    * ids     [[buffer(1)]],
    device       float  * dst     [[buffer(2)]],
    constant    int     & ne      [[buffer(3)]],
    constant    int     & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int nb = ne / 32;
    int total = nt * nb;
    int idx = (int)tid;
    if (idx >= total) return;

    int t = idx / nb;
    int b = idx % nb;
    int token_id = ids[t];

    int off = (token_id * nb + b) * Q4B;
    device const half * hptr = (device const half *)(weights + off);
    float d4 = float(hptr[0]);
    device const uchar * qs = weights + off + 2;

    int base = t * ne + b * 32;
    for (int j = 0; j < 16; j++) {
        uchar byte = qs[j];
        dst[base + j]      = float(int(byte & 0x0F) - 8) * d4;
        dst[base + j + 16] = float(int(byte >> 4) - 8) * d4;
    }
}

// ─── RMSNorm (1 threadgroup per row, 32 threads) ─────────────
// Parallel sum-of-squares via simd_sum (single simdgroup, no shared memory).
// y[t][i] = x[t][i] * rsqrt(mean(x[t]²) + eps) * w[i]

kernel void kernel_rms_norm_f32(
    device const float * x       [[buffer(0)]],
    device const float * w       [[buffer(1)]],
    device       float * y       [[buffer(2)]],
    constant    int    & d       [[buffer(3)]],
    constant    float  & eps     [[buffer(4)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]],
    uint3 ntg   [[threads_per_threadgroup]]
) {
    int row = tgpig.x;
    int d4 = d / 4;

    device const float4 * x4 = (device const float4 *)(x + row * d);

    float ss = 0.0f;
    for (int i = tpitg.x; i < d4; i += 32) {
        ss += dot(x4[i], x4[i]);
    }
    int rem = d - d4 * 4;
    if (tpitg.x == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        for (int i = 0; i < rem; i++) ss += x_tail[i] * x_tail[i];
    }
    ss = simd_sum(ss);

    float scale = 1.0f / sqrt(ss / (float)d + eps);

    device float4 * y4 = (device float4 *)(y + row * d);
    device const float4 * w4 = (device const float4 *)w;
    for (int i = tpitg.x; i < d4; i += 32) {
        y4[i] = x4[i] * scale * w4[i];
    }
    if (tpitg.x == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        device       float * y_tail = y + row * d + d4 * 4;
        device const float * w_tail = w + d4 * 4;
        for (int i = 0; i < rem; i++) y_tail[i] = x_tail[i] * scale * w_tail[i];
    }
}

// ─── Add bias ────────────────────────────────────────────────
// y[t][i] += b[i]

// ─── RMSNorm, multi-simdgroup (256-thread) variant ───────────
// Faithful llama.cpp transcription (kernel_rms_norm_fuse_impl): the threadgroup
// is 256 threads (8 simdgroups). Per-simdgroup partial sums are reduced through
// a small threadgroup buffer with TWO threadgroup barriers. The 32-thread
// single-simdgroup kernel above was measured at ~7x the per-dispatch cost of
// the 256-thread elementwise kernels (P0 profile 2026-08-10) — a single simdgroup
// cannot hide DRAM latency for one 896-element row. Dispatch nt = min(d/4, 256).
kernel void kernel_rms_norm_f32_256(
    device const float * x       [[buffer(0)]],
    device const float * w       [[buffer(1)]],
    device       float * y       [[buffer(2)]],
    constant    int    & d       [[buffer(3)]],
    constant    float  & eps     [[buffer(4)]],
    threadgroup float * shmem [[threadgroup(0)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]]
) {
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ntg = 256; // dispatched threads per threadgroup (8 simdgroups)
    int row = tgpig.x;
    int d4 = d / 4;
    device const float4 * x4 = (device const float4 *)(x + row * d);
    float ss = 0.0f;
    for (int i = 32 * sgitg + tiisg; i < d4; i += ntg) {
        ss += dot(x4[i], x4[i]);
    }
    int rem = d - d4 * 4;
    if (tiisg == 0 && sgitg == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        for (int i = 0; i < rem; i++) ss += x_tail[i] * x_tail[i];
    }
    ss = simd_sum(ss);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ss = shmem[tiisg];
    ss = simd_sum(ss);

    float scale = 1.0f / sqrt(ss / (float)d + eps);

    device float4 * y4 = (device float4 *)(y + row * d);
    device const float4 * w4 = (device const float4 *)w;
    for (int i = 32 * sgitg + tiisg; i < d4; i += ntg) {
        y4[i] = x4[i] * scale * w4[i];
    }
    if (tiisg == 0 && sgitg == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        device       float * y_tail = y + row * d + d4 * 4;
        device const float * w_tail = w + d4 * 4;
        for (int i = 0; i < rem; i++) y_tail[i] = x_tail[i] * scale * w_tail[i];
    }
}


 kernel void kernel_add_bias_f32(
     device       float * y [[buffer(0)]],
     device const float * b [[buffer(1)]],
     constant    int    & d [[buffer(2)]],
     uint2 tid [[thread_position_in_grid]]
 ) {
     const int t = tid.x, i4 = tid.y;
     const int i = i4 * 4;
     if (i + 3 < d) {
         *(device float4 *)(y + t * d + i) += *(device const float4 *)(b + i);
     } else {
         for (int k = i; k < d; k++) y[t * d + k] += b[k];
     }
 }

// ─── Element-wise add (float4) ───────────────────────────────
// z = x + y; 4 elements per thread, scalar tail for n % 4 != 0.

kernel void kernel_add_f32(
    device const float * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device       float * z [[buffer(2)]],
    constant    int    & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        *(device float4 *)(z + 4*t) = *(device const float4 *)(x + 4*t) + *(device const float4 *)(y + 4*t);
    } else {
        for (int k = 4*t; k < n; k++) z[k] = x[k] + y[k];
    }
}

// ─── Element-wise multiply (float4) ──────────────────────────
// z[t] = x[t] * y[t]; 4 elements per thread.

kernel void kernel_mul_f32(
    device const float * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device       float * z [[buffer(2)]],
    constant    int    & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        *(device float4 *)(z + 4*t) = *(device const float4 *)(x + 4*t) * *(device const float4 *)(y + 4*t);
    } else {
        for (int k = 4*t; k < n; k++) z[k] = x[k] * y[k];
    }
}

// ─── SiLU (in-place, float4) ─────────────────────────────────
// y[i] = y[i] / (1 + exp(-y[i])); 4 elements per thread.

kernel void kernel_silu_f32(
    device float * y [[buffer(0)]],
    constant int & n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        float4 v = *(device float4 *)(y + 4*t);
        float4 r;
        r.x = v.x / (1.0f + exp(-v.x));
        r.y = v.y / (1.0f + exp(-v.y));
        r.z = v.z / (1.0f + exp(-v.z));
        r.w = v.w / (1.0f + exp(-v.w));
        *(device float4 *)(y + 4*t) = r;
    } else {
        for (int k = 4*t; k < n; k++) {
            float v = y[k];
            y[k] = v / (1.0f + exp(-v));
        }
    }
}

// ─── SwiGLU (fused SiLU + Mul, float4) ───────────────────────
// dst[i] = silu(gate[i]) * up[i]; 4 elements per thread.

kernel void kernel_swiglu_f32(
    device const float * gate [[buffer(0)]],
    device const float * up   [[buffer(1)]],
    device       float * dst  [[buffer(2)]],
    constant    int    & n    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        float4 g = *(device const float4 *)(gate + 4*t);
        float4 u = *(device const float4 *)(up + 4*t);
        float4 r;
        r.x = (g.x / (1.0f + exp(-g.x))) * u.x;
        r.y = (g.y / (1.0f + exp(-g.y))) * u.y;
        r.z = (g.z / (1.0f + exp(-g.z))) * u.z;
        r.w = (g.w / (1.0f + exp(-g.w))) * u.w;
        *(device float4 *)(dst + 4*t) = r;
    } else {
        for (int k = 4*t; k < n; k++) {
            float g = gate[k];
            dst[k] = (g / (1.0f + exp(-g))) * up[k];
        }
    }
}

// ─── RoPE (in-place) ─────────────────────────────────────────
// Applies rotary positional embedding to Q and K.
// x layout: [nt][n_head][n_dims]

// ─── Parallel prefill attention (P1 2026-08-11) ──────────────
// The classic kernel_gqa_attn_f32 at prefill is latency-bound: grid (nt,nk),
// each threadgroup loops the KV sequentially with 2 barriers/tile (~24K
// barriers at nt=430) → measured ~100 ms (48% of prefill, ~25x llama's
// attention). This 3-pass replacement is fully parallel (no threadgroup
// barriers):
//   pass 1 kernel_attn_scores:  scores[t][h][kv] = dot(q[t][h][0..hd], k[kv][hk*hd..]) * scale
//   pass 2 kernel_softmax_attn: masked softmax over kv per (t,h) row (in-place)
//   pass 3 kernel_attn_output:  out[t][h][0..hd] = Σ_kv softmax[t][h][kv] * v[kv][hk*hd..]
// GQA: each query head h uses KV group hk = h/gqa.

// pass 1: scores. Grid: (nt*nh) threadgroups of 256 threads — one threadgroup
// per (t,h) row, threads split across the nkv scores. Each thread computes one
// score = dot(q[t][h][0..hd], k[kv][hk*hd..]) * scale.
kernel void kernel_attn_scores(
    device const float * q    [[buffer(0)]],  // [nt][nh*hd]
    device const float * k    [[buffer(1)]],  // [nkv][nkt]
    device       float * scores [[buffer(2)]], // [nt][nh][nkv]
    constant    int    & nh    [[buffer(3)]],
    constant    int    & hd    [[buffer(4)]],
    constant    int    & nkv   [[buffer(5)]],
    constant    int    & nt    [[buffer(6)]],
    constant    int    & gqa   [[buffer(7)]],
    constant    int    & nkt   [[buffer(8)]],
    constant    float  & scale [[buffer(9)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {
    const int th = (int)tgpig.x;   // t*nh+h
    const int t = th / nh;
    const int h = th % nh;
    if (t >= nt) return;
    const int hk = h / gqa;
    device const float * qh = q + t * nh * hd + h * hd;
    // thread i computes scores[th][i] (i in 0..nkv, 256 threads)
    const int kv = (int)tpitg.x;
    if (kv >= nkv) return;
    device const float * kh = k + kv * nkt + hk * hd;
    float s = 0.0f;
    for (int d = 0; d < hd; d++) s += qh[d] * kh[d];
    scores[th * nkv + kv] = s * scale;
}

// pass 3: out[t][h][0..hd] = Σ_kv softmax[t][h][kv] * v[kv][hk*hd..hd].
// Grid: (nt*nh) threadgroups of 256 threads — one per (t,h), threads split
// across the hd output dims (hd<=256 for Qwen).
kernel void kernel_attn_output(
    device const float * scores [[buffer(0)]],  // [nt][nh][nkv] (softmaxed)
    device const float * v      [[buffer(1)]],  // [nkv][nkt]
    device       float * out    [[buffer(2)]],  // [nt][nh*hd]
    constant    int    & nh    [[buffer(3)]],
    constant    int    & hd    [[buffer(4)]],
    constant    int    & nkv   [[buffer(5)]],
    constant    int    & nt    [[buffer(6)]],
    constant    int    & gqa   [[buffer(7)]],
    constant    int    & nkt   [[buffer(8)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint3 tpitg [[thread_position_in_threadgroup]]
) {
    const int th = (int)tgpig.x;   // t*nh+h
    const int t = th / nh;
    const int h = th % nh;
    if (t >= nt) return;
    const int hk = h / gqa;
    const int d = (int)tpitg.x;   // 0..hd-1
    if (d >= hd) return;
    device const float * sc = scores + th * nkv;
    device const float * vh = v + hk * hd + d;
    float acc = 0.0f;
    for (int kv = 0; kv < nkv; kv++) acc += sc[kv] * vh[kv * nkt];
    out[t * nh * hd + h * hd + d] = acc;
}

kernel void kernel_softmax_attn(
    device       float * scores [[buffer(0)]],   // [nt*nh][nkv] (scores already scaled)
    constant    int    * positions [[buffer(1)]],
    constant    int    & nkv    [[buffer(2)]],
    constant    int    & nt     [[buffer(3)]],
    constant    int    & nh     [[buffer(4)]],
    threadgroup float * shmem  [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int ntg = 256; // dispatched threads per threadgroup (8 simdgroups)
    const int th = (int)tgpig.x;   // row = t*nh + h
    const int t = th / nh;
    const int vl = positions[t] + 1;   // valid KV length for this token
    device float * row = scores + th * nkv;

    if (sgitg == 0) { shmem[tiisg] = -INFINITY; }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 1: max over valid positions (scores already scaled)
    float m = -INFINITY;
    for (int i = 32*sgitg + tiisg; i < vl; i += ntg) {
        m = max(m, row[i]);
    }
    m = simd_max(m);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) { shmem[sgitg] = m; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m = shmem[tiisg];
    m = simd_max(m);

    // pass 2: exp(sum) and write normalized + masked
    // (re-init shmem to 0 — the max pass left per-simdgroup maxes in it)
    if (sgitg == 0) { shmem[tiisg] = 0.0f; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.0f;
    for (int i = 32*sgitg + tiisg; i < vl; i += ntg) {
        float e = exp(row[i] - m);
        row[i] = e;
        s += e;
    }
    s = simd_sum(s);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) { shmem[sgitg] = s; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    s = shmem[tiisg];
    s = simd_sum(s);

    const float inv = 1.0f / s;
    for (int i = 32*sgitg + tiisg; i < nkv; i += ntg) {
        if (i < vl) row[i] *= inv;
        else row[i] = 0.0f;
    }
}

kernel void kernel_rope_f32(
    device float * x [[buffer(0)]],
    constant int & n_head [[buffer(1)]],
    constant int & n_dims [[buffer(2)]],
    constant int & nt [[buffer(3)]],
    constant float & freq_base [[buffer(4)]],
    constant float & freq_scale [[buffer(5)]],
    constant int * positions [[buffer(6)]],
    constant int & rope_style [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]   // (dim, head, token)
) {
    int half_dim = n_dims / 2;
    int d = tid.x;       // 0..half_dim-1
    int h = tid.y;       // 0..n_head-1
    int t = tid.z;       // 0..nt-1
    if (t >= nt || h >= n_head || d >= half_dim) return;
    int base = (t * n_head + h) * n_dims;
    float freq = freq_scale / pow(freq_base, (2.0 * d) / n_dims);
    float theta = positions[t] * freq;
    float cs = cos(theta), sn = sin(theta);
    int i0, i1;
    if (rope_style == 1) {
        i0 = base + 2 * d;
        i1 = base + 2 * d + 1;
    } else {
        i0 = base + d;
        i1 = base + d + half_dim;
    }
    float x0 = x[i0], x1 = x[i1];
    x[i0] = x0 * cs - x1 * sn;
    x[i1] = x0 * sn + x1 * cs;
}

// ─── KV cache store ──────────────────────────────────────────
// Scatters nt new K/V rows into the persistent KV cache at positions[].

 kernel void kernel_store_kv_f32(
    device const float * src [[buffer(0)]],
    device       float * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int j = tid.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = src[t * nkt + j];
}

// F16 KV cache variant: stores f32 K/V rows into a half cache (2 bytes/elem),
// halving attention memory bandwidth (matches llama.cpp's F16 cache).
kernel void kernel_store_kv_f16(
    device const float * src [[buffer(0)]],
    device       half  * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int j = tid.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = half(src[t * nkt + j]);
}

// ─── Fused bias-add + RoPE + KV-store (nt==1 decode) ──────────
// One kernel replaces add_bias×3 + rope×2 + store_kv×2 (7 dispatches → 1).
// bqkv layout (nt==1): [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt].
// Applies the per-section bias, RoPE to the q/k sections, and stores k/v into
// the KV cache (f32 or f16 per kv_is_f16). Bit-identical to the 7 separate
// kernels: bias then rope on the same values, same store addresses.
// Thread mapping: one thread per (head, d<half_dim) rope pair for q and k,
// one thread per v element → grid = nqt/2 + nkt/2 + nkt.

kernel void kernel_attn_bias_rope_store(
    device       float * bqkv [[buffer(0)]],
    device const float * bias_q [[buffer(1)]],
    device const float * bias_k [[buffer(2)]],
    device const float * bias_v [[buffer(3)]],
    device        void * kv_k [[buffer(4)]],
    device        void * kv_v [[buffer(5)]],
    constant        int & nqt [[buffer(6)]],
    constant        int & nkt [[buffer(7)]],
    constant        int & hd  [[buffer(8)]],
    constant      float & freq_base [[buffer(9)]],
    constant      float & freq_scale [[buffer(10)]],
    constant        int & pos [[buffer(11)]],
    constant        int & rope_style [[buffer(12)]],
    constant        int & kv_is_f16 [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = (int)tid;
    if (u >= total) return;

    if (u < qpairs) {
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int i0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int i1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        float x0 = bqkv[i0] + bias_q[i0];
        float x1 = bqkv[i1] + bias_q[i1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        bqkv[i0] = x0 * cs - x1 * sn;
        bqkv[i1] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        const int u2   = u - qpairs;
        const int head = u2 / half_dim;
        const int d    = u2 % half_dim;
        const int base = head * hd;
        const int j0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int j1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        const int k0 = nqt + j0, k1 = nqt + j1;
        float x0 = bqkv[k0] + bias_k[j0];
        float x1 = bqkv[k1] + bias_k[j1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        float r0 = x0 * cs - x1 * sn;
        float r1 = x0 * sn + x1 * cs;
        bqkv[k0] = r0; bqkv[k1] = r1;
        if (kv_is_f16) {
            ((device half *)kv_k)[pos * nkt + j0] = half(r0);
            ((device half *)kv_k)[pos * nkt + j1] = half(r1);
        } else {
            ((device float *)kv_k)[pos * nkt + j0] = r0;
            ((device float *)kv_k)[pos * nkt + j1] = r1;
        }
    } else {
        const int j  = u - qpairs - kpairs;
        const int vi = nqt + nkt + j;
        float v = bqkv[vi] + bias_v[j];
        bqkv[vi] = v;
        if (kv_is_f16) {
            ((device half *)kv_v)[pos * nkt + j] = half(v);
        } else {
            ((device float *)kv_v)[pos * nkt + j] = v;
        }
    }
}

// ─── Flash Attention (GQA, online softmax, tiled K/V) ─────
// One threadgroup per (token, KV_head). Each simdgroup processes one
// query head. K and V tiles loaded into threadgroup-shared memory
// and reused by all query heads in the GQA group.
// Grid: (nt, nk), TG: (32, gqa) where gqa = nh / nk.

kernel void kernel_gqa_attn_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t  = (int)tgpig.x;
    int hk = (int)tgpig.y;
    if (t >= nt || hk >= nk) return;

    int nkv  = positions[t] + 1;
    int gqa  = nh / nk;
    // Do NOT return early for heads beyond nh: all simdgroups in the
    // threadgroup must reach every threadgroup_barrier below, or the GPU
    // deadlocks when nh % nk != 0. Invalid heads run the loop with a dummy
    // head index but skip the output write.
    int  h0         = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h          = valid_head ? h0 : 0;

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;

    device const float * qhead = q + t * stride_q + h * hd;
    device       float * ohead = o + t * stride_q + h * hd;

    threadgroup float * k_tile = shmem;
    threadgroup float * v_tile = shmem + Bc * hd;

    float mx = -INFINITY;
    float S = 0.0f;
    float acc[256];
    for (int i = 0; i < hd; i++) acc[i] = 0.0f;

    int n_tiles = (nkv + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int kv_start = tile_idx * Bc;
        int tile_sz  = min(Bc, nkv - kv_start);

        int total = tile_sz * hd;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total; i += tgsz) {
            int ki = kv_start + i / hd;
            int di = i % hd;
            k_tile[i] = k[ki * stride_kv + hk * hd + di];
            v_tile[i] = v[ki * stride_kv + hk * hd + di];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            // All 32 lanes of the simdgroup execute the SAME iteration count so
            // simd_max below never runs across divergent lanes (a divergent
            // simd_max includes stale register values from exited lanes, which
            // corrupts the online-softmax running max for partial tiles).
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float * kj = k_tile + j * hd;
                dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += qhead[d] * kj[d];
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d = 0; d < hd; d++) acc[d] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float * vj = v_tile + j * hd;
                for (int d = 0; d < hd; d++) acc[d] += e * vj[d];
                S += e;
            }
            mx = new_mx;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);
    for (int d = 0; d < hd; d++) acc[d] = simd_sum(acc[d]);

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    if (valid_head) {
        for (int d = tiisg; d < hd; d += 32) {
            ohead[d] = acc[d] * inv;
        }
    }
}

// F16 KV cache variant of kernel_gqa_attn_f32: K/V are read from a half cache
// (2 bytes/elem, matching llama.cpp) and converted to f32 when staged into
// threadgroup tiles. Enabled via MINFER_CACHE_TYPE=f16 (default is f32).
kernel void kernel_gqa_attn_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * o        [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t  = (int)tgpig.x;
    int hk = (int)tgpig.y;
    if (t >= nt || hk >= nk) return;

    int nkv  = positions[t] + 1;
    int gqa  = nh / nk;
    // Do NOT return early for heads beyond nh: all simdgroups in the
    // threadgroup must reach every threadgroup_barrier below, or the GPU
    // deadlocks when nh % nk != 0. Invalid heads run the loop with a dummy
    // head index but skip the output write.
    int  h0         = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h          = valid_head ? h0 : 0;

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;

    device const float * qhead = q + t * stride_q + h * hd;
    device       float * ohead = o + t * stride_q + h * hd;

    threadgroup float * k_tile = shmem;
    threadgroup float * v_tile = shmem + Bc * hd;

    float mx = -INFINITY;
    float S = 0.0f;
    float acc[256];
    for (int i = 0; i < hd; i++) acc[i] = 0.0f;

    int n_tiles = (nkv + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int kv_start = tile_idx * Bc;
        int tile_sz  = min(Bc, nkv - kv_start);

        int total = tile_sz * hd;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total; i += tgsz) {
            int ki = kv_start + i / hd;
            int di = i % hd;
            k_tile[i] = float(k[ki * stride_kv + hk * hd + di]);
            v_tile[i] = float(v[ki * stride_kv + hk * hd + di]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float * kj = k_tile + j * hd;
                dot = 0.0f;
                for (int d = 0; d < hd; d++) dot += qhead[d] * kj[d];
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d = 0; d < hd; d++) acc[d] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float * vj = v_tile + j * hd;
                for (int d = 0; d < hd; d++) acc[d] += e * vj[d];
                S += e;
            }
            mx = new_mx;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);
    for (int d = 0; d < hd; d++) acc[d] = simd_sum(acc[d]);

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    if (valid_head) {
        for (int d = tiisg; d < hd; d += 32) {
            ohead[d] = acc[d] * inv;
        }
    }
}

// ─── KV-parallel split attention (nt==1 decode) ──────────────
// The decode bottleneck (measured ~48% of per-token time, grows with KV): for
// nt==1 the classic kernel uses a grid of only (1, nk) threadgroups that loop
// the KV tiles SEQUENTIALLY (latency-bound, GPU underutilized). This two-pass
// split parallelizes the KV dimension:
//   pass 1  kernel_gqa_attn_partial_f32: grid (nt, nk, n_chunks) — each TG
//           computes an online-softmax PARTIAL (mx, S, acc[hd]) for its KV
//           chunk [c*cs, min(nkv,(c+1)*cs)). Same tile/barrier structure as the
//           classic kernel, but each TG loops only its chunk.
//   pass 2  kernel_gqa_attn_combine_f32: grid (nt, nh) — reads the n_chunks
//           partials, merges with the standard max/exp/l-sum, writes output.
// GPU safety: pass 1 preserves the uniform-loop + valid-head + no-early-return
// patterns; pass 2 is a pure elementwise kernel (no shared memory, no barriers).

kernel void kernel_gqa_attn_partial_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t     = (int)tgpig.x;
    int hk    = (int)tgpig.y;
    int chunk = (int)tgpig.z;
    if (t >= nt || hk >= nk) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int h0  = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h  = valid_head ? h0 : 0;

    // chunk bounds: chunk c covers [c*cs, min(nkv,(c+1)*cs)), cs = ceil(nkv/P).
    // Empty chunks (kv_start >= nkv) produce an empty partial (mx=-INF, S=0,
    // acc=0) that the combine ignores via exp(-INF - m) == 0.
    int cs = (nkv + n_chunks - 1) / n_chunks;
    int kv_start = chunk * cs;
    int kv_end   = min(nkv, kv_start + cs);

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;
    int hd4       = hd / 4;   // hd % 4 == 0 is guarded upstream (layer_gpu)

    device const float4 * qhead4 = (device const float4 *)(q + t * stride_q + h * hd);

    threadgroup float4 * k_tile4 = (threadgroup float4 *)shmem;
    threadgroup float4 * v_tile4 = (threadgroup float4 *)(shmem + Bc * hd);

    float mx = -INFINITY;
    float S = 0.0f;
    // Vectorized float4 accumulator: hd<=256 => at most 64 float4s. Kept small
    // (64 floats for hd=64) so the compiler can keep it in REGISTERS — the
    // scalar dynamic-indexed float acc[256] landed in per-thread LOCAL memory,
    // which was the long-context attention bottleneck (per-thread serial DRAM
    // RMWs that don't improve with more parallelism).
    float4 acc4[64];
    for (int d4 = 0; d4 < hd4; d4++) acc4[d4] = 0.0f;

    int n_tiles = (kv_end - kv_start + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int ks = kv_start + tile_idx * Bc;
        int tile_sz = min(Bc, kv_end - ks);

        int total4 = tile_sz * hd4;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total4; i += tgsz) {
            int ki = ks + i / hd4;
            int di = i % hd4;
            device const float4 * k4 = (device const float4 *)(k + ki * stride_kv + hk * hd);
            device const float4 * v4 = (device const float4 *)(v + ki * stride_kv + hk * hd);
            k_tile4[i] = k4[di];
            v_tile4[i] = v4[di];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float4 * kj4 = k_tile4 + j * hd4;
                dot = 0.0f;
                for (int d4 = 0; d4 < hd4; d4++) {
                    float4 qv = qhead4[d4] * kj4[d4];
                    dot += qv.x + qv.y + qv.z + qv.w;
                }
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d4 = 0; d4 < hd4; d4++) acc4[d4] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float4 * vj4 = v_tile4 + j * hd4;
                for (int d4 = 0; d4 < hd4; d4++) acc4[d4] += e * vj4[d4];
                S += e;
            }
            mx = new_mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);

    // partial layout: [t][h][chunk] = {mx, S, acc[hd]}  (contiguous per chunk)
    int pbase = ((t * nh + h) * n_chunks + chunk) * (2 + hd);
    if (valid_head) {
        if (tiisg == 0) {
            partial[pbase + 0] = mx;
            partial[pbase + 1] = S;
        }
        // UNIFORM d loop (all 32 lanes step together, so simd_sum reduces the
        // SAME component across lanes) — a per-lane divergent loop over the
        // acc4 elements would make simd_sum reduce mismatched values.
        for (int d = 0; d < hd; d++) {
            float4 a4 = acc4[d / 4];
            float val;
            switch (d % 4) {
                case 0: val = simd_sum(a4.x); break;
                case 1: val = simd_sum(a4.y); break;
                case 2: val = simd_sum(a4.z); break;
                default: val = simd_sum(a4.w); break;
            }
            if (tiisg == 0) partial[pbase + 2 + d] = val;
        }
    }
}

// F16 KV cache variant of kernel_gqa_attn_partial_f32: K/V read from a half
// cache (2 bytes/elem) and converted to f32 (float4) when staged into the
// threadgroup tiles. The partials + combine are f32, so kernel_gqa_attn_combine_f32
// is shared. Enabled via MINFER_CACHE_TYPE=f16.
kernel void kernel_gqa_attn_partial_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    const int Bc = 32;
    int t     = (int)tgpig.x;
    int hk    = (int)tgpig.y;
    int chunk = (int)tgpig.z;
    if (t >= nt || hk >= nk) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int h0  = hk * gqa + (int)sgitg;
    bool valid_head = (h0 < nh);
    int  h  = valid_head ? h0 : 0;

    int cs = (nkv + n_chunks - 1) / n_chunks;
    int kv_start = chunk * cs;
    int kv_end   = min(nkv, kv_start + cs);

    int stride_q  = nh * hd;
    int stride_kv = nk * hd;
    int hd4       = hd / 4;

    device const float4 * qhead4 = (device const float4 *)(q + t * stride_q + h * hd);

    threadgroup float4 * k_tile4 = (threadgroup float4 *)shmem;
    threadgroup float4 * v_tile4 = (threadgroup float4 *)(shmem + Bc * hd);

    float mx = -INFINITY;
    float S = 0.0f;
    float4 acc4[64];
    for (int d4 = 0; d4 < hd4; d4++) acc4[d4] = 0.0f;

    int n_tiles = (kv_end - kv_start + Bc - 1) / Bc;
    for (int tile_idx = 0; tile_idx < n_tiles; tile_idx++) {
        int ks = kv_start + tile_idx * Bc;
        int tile_sz = min(Bc, kv_end - ks);

        int total4 = tile_sz * hd4;
        int tgsz = 32 * gqa;
        for (int i = tiisg + (int)sgitg * 32; i < total4; i += tgsz) {
            int ki = ks + i / hd4;
            int di = i % hd4;
            device const half4 * k4 = (device const half4 *)(k + ki * stride_kv + hk * hd);
            device const half4 * v4 = (device const half4 *)(v + ki * stride_kv + hk * hd);
            k_tile4[i] = float4(k4[di]);
            v_tile4[i] = float4(v4[di]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int j0 = 0; j0 < tile_sz; j0 += 32) {
            const int j = j0 + (int)tiisg;
            const bool valid = (j < tile_sz);
            float dot = -INFINITY;
            if (valid) {
                threadgroup float4 * kj4 = k_tile4 + j * hd4;
                dot = 0.0f;
                for (int d4 = 0; d4 < hd4; d4++) {
                    float4 qv = qhead4[d4] * kj4[d4];
                    dot += qv.x + qv.y + qv.z + qv.w;
                }
                dot *= scale;
            }

            float batch_mx = simd_max(dot);
            float new_mx = max(mx, batch_mx);
            float corr = exp(mx - new_mx);
            for (int d4 = 0; d4 < hd4; d4++) acc4[d4] *= corr;
            S *= corr;
            float e = valid ? exp(dot - new_mx) : 0.0f;
            if (valid) {
                threadgroup float4 * vj4 = v_tile4 + j * hd4;
                for (int d4 = 0; d4 < hd4; d4++) acc4[d4] += e * vj4[d4];
                S += e;
            }
            mx = new_mx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    S = simd_sum(S);

    int pbase = ((t * nh + h) * n_chunks + chunk) * (2 + hd);
    if (valid_head) {
        if (tiisg == 0) {
            partial[pbase + 0] = mx;
            partial[pbase + 1] = S;
        }
        for (int d = 0; d < hd; d++) {
            float4 a4 = acc4[d / 4];
            float val;
            switch (d % 4) {
                case 0: val = simd_sum(a4.x); break;
                case 1: val = simd_sum(a4.y); break;
                case 2: val = simd_sum(a4.z); break;
                default: val = simd_sum(a4.w); break;
            }
            if (tiisg == 0) partial[pbase + 2 + d] = val;
        }
    }
}

// ─── Flash attention (decode nt==1) — port of llama kernel_flash_attn_ext_vec
// (option C): single-simdgroup fixed-shape port, DK=DV=64, NE=2, C=32,
// NWG=n_chunks, NSG=1. Each threadgroup computes an online-softmax PARTIAL
// {M, S, O[hd]} over the strided KV chunks {iwg, iwg+n_chunks, ...}×C — the
// SAME partial format as kernel_gqa_attn_partial_f32, so
// kernel_gqa_attn_combine_f32 merges them unchanged.
//
// GPU-safety (deadlock discipline):
//  - No per-lane early returns. `if (ic >= nkv) break` depends only on
//    tgpig/lane-independent values → all 32 lanes break together.
//  - No `continue`. Out-of-range KV lanes are masked to -MINF_MAXHALF (inline,
//    lane-local) so exp() yields ~0; the read is clamped to nkv-1 (in-bounds,
//    value ignored).
//  - All lanes reach every threadgroup_barrier. NSG=1 fixed: no cross-simdgroup
//    reduce, no threadgroup_barrier in the reduce phase (llama's `r` loop runs
//    only for NSG>1).
//  - The uniform d-loop for the acc reduction is NOT needed here — each lane
//    writes its own so4[tiisg] float4 slot (ty==0 lanes, 16 slots = hd/4).
//  - hd==64 is required (DK/DV fixed); host layer_gpu guards hd==64 &&
//    hd%4==0 before dispatching (else falls back to the split-attention path).

kernel void kernel_flash_attn_ext_f32(
    device const float * q        [[buffer(0)]],
    device const float * k        [[buffer(1)]],
    device const float * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int DK  = 64;
    constexpr int DV  = 64;
    constexpr int NE  = 2;
    constexpr int C   = 32;
    constexpr int NW  = 32;
    constexpr int NL  = NW / NE;   // 16
    constexpr int DK4 = DK / 4;    // 16
    constexpr int DV4 = DV / 4;    // 16
    constexpr float MINF_MAXHALF = 65504.0f; // half max; -MINF_MAXHALF ~= -INF mask

    const int t   = (int)tgpig.x;
    const int h   = (int)tgpig.y;
    const int iwg = (int)tgpig.z;
    if (t >= nt || h >= nh) return;

    const int nkv = positions[t] + 1;
    const int gqa = nh / nk;
    const int hk  = h / gqa;
    const int stride_kv  = nk * hd;         // f32 elements per token row
    const int stride_kv4 = stride_kv / 4;   // float4 per token row
    const int hk4        = hk * hd / 4;     // head offset in float4

    const int tx = (int)tiisg % NL;   // 0..NL-1 (DK4 dim)
    const int ty = (int)tiisg / NL;   // 0..NE-1 (token lane)

    // shmem layout (f32): sq4[DK4 float4] | ss[C] | so4[NW float4]
    // (no sm[] array: the partial-chunk mask is computed inline below so every
    //  lane reads/writes only its own registers — no cross-lane threadgroup
    //  access outside the two barrier-protected ss[] handoffs)
    threadgroup float4 * sq4 = (threadgroup float4 *)shmem;
    threadgroup float  * ss  = shmem + DK4 * 4;
    threadgroup float4 * so4 = (threadgroup float4 *)(ss + C);

    // load Q head into shared memory (DK4 float4)
    device const float4 * q4 = (device const float4 *)(q + t * (nh * hd) + h * hd);
    for (int i = (int)tiisg; i < DK4; i += NW) sq4[i] = q4[i];
    // zero ss and this lane's O slot
    for (int i = (int)tiisg; i < C; i += NW) ss[i] = 0.0f;
    so4[tiisg] = (float4)0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    // KV chunk loop: chunks iwg, iwg+n_chunks, ... each of C tokens
    for (int ic0 = iwg; ; ic0 += n_chunks) {
        int ic = ic0 * C;
        if (ic >= nkv) break;

        // Q*K^T
        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; ++cc) {
            int token = ic + NE * cc + ty;
            if (token >= nkv) token = nkv - 1; // clamped read, value masked out
            device const float4 * pk = (device const float4 *)
                (k + token * stride_kv) + hk4 + tx;
            float4 qv = sq4[tx] * pk[0];
            mqk[cc] = qv.x + qv.y + qv.z + qv.w;
            // simdgroup reduce over the DK4 lanes (tx): full-head dot
            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
            // broadcast the reduced value from lane NL*ty
            mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
        }
        // store scaled score (+ partial-chunk mask) in ss[2*tx+ty] == token
        // ic+2tx+ty; out-of-range lanes get -MINF_MAXHALF (exp() ~= 0 contribution).
        // Mask is computed inline (lane-local) so no threadgroup memory race.
        ss[NE * tx + ty] = mqk[tx] * scale
                         + ((ic + NE * tx + ty < nkv) ? 0.0f : -MINF_MAXHALF);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // online softmax
        {
            const float m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) so4[tiisg] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // O = O + (Q*K^T)*V
        {
            float4 lo = (float4)0.0f;
            for (int cc = 0; cc < C / NE; ++cc) {
                int token = ic + NE * cc + ty;
                if (token >= nkv) token = nkv - 1; // clamped read, value masked out
                device const float4 * pv4 = (device const float4 *)
                    (v + token * stride_kv) + hk4 + tx;
                lo += pv4[0] * ss[NE * cc + ty];
            }
            // merge the NE=2 ty lanes (token ic+2cc and ic+2cc+1)
            lo += simd_shuffle_down(lo, 16);
            if (ty == 0) so4[tiisg] += lo;
        }
    }

    // write partial (same layout as partial_f32): {M, S, O[hd]} per (t,h,iwg)
    int pbase = ((t * nh + h) * n_chunks + iwg) * (2 + hd);
    if (tiisg == 0) {
        partial[pbase + 0] = M;
        partial[pbase + 1] = S;
    }
    if (ty == 0) {
        float4 acc = so4[tiisg];
        partial[pbase + 2 + tx * 4 + 0] = acc.x;
        partial[pbase + 2 + tx * 4 + 1] = acc.y;
        partial[pbase + 2 + tx * 4 + 2] = acc.z;
        partial[pbase + 2 + tx * 4 + 3] = acc.w;
    }
}

// F16 KV cache variant of kernel_flash_attn_ext_f32: K/V read from a half cache
// (2 bytes/elem) and converted to f32 when dotted. Partials + combine are f32
// and shared with the f32 variant (kernel_gqa_attn_combine_f32).
kernel void kernel_flash_attn_ext_f16(
    device const float * q        [[buffer(0)]],
    device const half  * k        [[buffer(1)]],
    device const half  * v        [[buffer(2)]],
    device       float * partial  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh        [[buffer(5)]],
    constant    int    & nk        [[buffer(6)]],
    constant    int    & hd        [[buffer(7)]],
    constant    float  & scale     [[buffer(8)]],
    constant    int    & nt        [[buffer(9)]],
    constant    int    & n_chunks  [[buffer(10)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int DK  = 64;
    constexpr int DV  = 64;
    constexpr int NE  = 2;
    constexpr int C   = 32;
    constexpr int NW  = 32;
    constexpr int NL  = NW / NE;
    constexpr int DK4 = DK / 4;
    constexpr int DV4 = DV / 4;
    constexpr float MINF_MAXHALF = 65504.0f;

    const int t   = (int)tgpig.x;
    const int h   = (int)tgpig.y;
    const int iwg = (int)tgpig.z;
    if (t >= nt || h >= nh) return;

    const int nkv = positions[t] + 1;
    const int gqa = nh / nk;
    const int hk  = h / gqa;
    const int stride_kv  = nk * hd;         // half elements per token row
    const int stride_kv4 = stride_kv / 4;   // half4 per token row
    const int hk4        = hk * hd / 4;

    const int tx = (int)tiisg % NL;
    const int ty = (int)tiisg / NL;

    threadgroup float4 * sq4 = (threadgroup float4 *)shmem;
    threadgroup float  * ss  = shmem + DK4 * 4;
    threadgroup float4 * so4 = (threadgroup float4 *)(ss + C);

    device const float4 * q4 = (device const float4 *)(q + t * (nh * hd) + h * hd);
    for (int i = (int)tiisg; i < DK4; i += NW) sq4[i] = q4[i];
    for (int i = (int)tiisg; i < C; i += NW) ss[i] = 0.0f;
    so4[tiisg] = (float4)0.0f;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float M = -INFINITY;
    float S = 0.0f;

    for (int ic0 = iwg; ; ic0 += n_chunks) {
        int ic = ic0 * C;
        if (ic >= nkv) break;

        float mqk[C / NE];
        for (int cc = 0; cc < C / NE; ++cc) {
            int token = ic + NE * cc + ty;
            if (token >= nkv) token = nkv - 1;
            device const half4 * pk = (device const half4 *)
                (k + token * stride_kv) + hk4 + tx;
            float4 kv4 = float4(pk[0]);
            float4 qv = sq4[tx] * kv4;
            mqk[cc] = qv.x + qv.y + qv.z + qv.w;
            mqk[cc] += simd_shuffle_down(mqk[cc],  8);
            mqk[cc] += simd_shuffle_down(mqk[cc],  4);
            mqk[cc] += simd_shuffle_down(mqk[cc],  2);
            mqk[cc] += simd_shuffle_down(mqk[cc],  1);
            mqk[cc] = simd_shuffle(mqk[cc], NL * ty);
        }
        ss[NE * tx + ty] = mqk[tx] * scale
                         + ((ic + NE * tx + ty < nkv) ? 0.0f : -MINF_MAXHALF);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            const float m = M;
            const float s = ss[tiisg];
            M = simd_max(max(M, s));
            const float ms = exp(m - M);
            const float vs = exp(s - M);
            S = S * ms + simd_sum(vs);
            ss[tiisg] = vs;
            if (ty == 0) so4[tiisg] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        {
            float4 lo = (float4)0.0f;
            for (int cc = 0; cc < C / NE; ++cc) {
                int token = ic + NE * cc + ty;
                if (token >= nkv) token = nkv - 1;
                device const half4 * pv4 = (device const half4 *)
                    (v + token * stride_kv) + hk4 + tx;
                lo += float4(pv4[0]) * ss[NE * cc + ty];
            }
            lo += simd_shuffle_down(lo, 16);
            if (ty == 0) so4[tiisg] += lo;
        }
    }

    int pbase = ((t * nh + h) * n_chunks + iwg) * (2 + hd);
    if (tiisg == 0) {
        partial[pbase + 0] = M;
        partial[pbase + 1] = S;
    }
    if (ty == 0) {
        float4 acc = so4[tiisg];
        partial[pbase + 2 + tx * 4 + 0] = acc.x;
        partial[pbase + 2 + tx * 4 + 1] = acc.y;
        partial[pbase + 2 + tx * 4 + 2] = acc.z;
        partial[pbase + 2 + tx * 4 + 3] = acc.w;
    }
}

// ─── Flash attention for prefill (nt > 1) — port of llama kernel_flash_attn_ext_blk
// (the legacy simdgroup_matrix flash, docs/METAL_OPTIMIZATIONS.md §4.3.1). Fixed-shape:
// NSG=4, Q=8, C=64, DK=DV=64. Grid (ceil(nt/8), nh), 128 threads (32 lanes x 4
// simdgroups). Each threadgroup computes Q=8 query tokens x ALL KV for head h
// (GQA head hk = h/gqa is baked into the K/V base). Faithful llama transcription
// with two GPU-safety deviations: the causal mask is computed inline (no
// mask/pad pre-pass kernels), and the PARTIAL last KV block (nkv % 64 != 0) is
// read from the [2][64][nkt] tail-pad buffer filled by kernel_kv_tail_pad
// (padded rows are zero + masked to -MINF, so they never contribute).
//
// shmem (7168 B): sq[512 half] | so[512 f32] | ss[1024 f32]
kernel void kernel_flash_attn_blk_f32(
    device const float * q         [[buffer(0)]],
    device const float * k         [[buffer(1)]],
    device const float * v         [[buffer(2)]],
    device const float * pad       [[buffer(3)]],   // [2][64][nkt] K-tail then V-tail
    device       float * out       [[buffer(4)]],
    constant    int    * positions [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int Q   = 8;
    constexpr int C   = 64;
    constexpr int NSG = 4;
    constexpr int DK  = 64;
    constexpr int DV  = 64;
    constexpr int NW  = 32;
    constexpr int NQ  = Q / NSG;      // 2
    constexpr int SH  = 2 * C;        // 128
    constexpr int DK4 = DK / 4;       // 16
    constexpr int DV4 = DV / 4;       // 16
    constexpr int DK8 = DK / 8;       // 8
    constexpr int PV  = 64;           // PAD2(DV, 64)
    constexpr int PV4 = PV / 4;       // 16
    constexpr int PV8 = PV / 8;       // 8
    constexpr int NC  = (C / 8) / NSG; // 2
    constexpr int NO  = PV8 / NSG;    // 2
    constexpr float MINF = 65504.0f;

    const int iq1 = (int)tgpig.x * Q;
    const int iq2 = (int)tgpig.y;
    const int nblk = (nkv + C - 1) / C;

    const int nqt  = nh * hd;
    const int nkt  = nk * hd;
    const int hk   = iq2 / (nh / nk);
    const int hoff = hk * hd;

    const int tx = (int)tiisg;

    // shmem layout (bytes): sq[0..1024) | so[1024..3072) | ss[3072..7168)
    threadgroup half  * sq = (threadgroup half  *)shmem;
    threadgroup float * so = (threadgroup float *)(shmem + 256);
    threadgroup float * ss = (threadgroup float *)(shmem + 768);

    threadgroup half4  * sq4 = (threadgroup half4  *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;
    threadgroup float2 * ss2 = (threadgroup float2 *)ss;

    // load Q heads into shared memory (each simdgroup loads NQ queries)
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *)(q + (iq1 + j) * nqt + iq2 * hd);
        for (int i = tx; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (iq1 + j < nt) ? half4(q4[i]) : (half4)0.0f;
        }
    }

    // zero so + ss
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] = (float4)0.0f;
        for (int i = tx; i < SH; i += NW) ss[j * SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (short jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -FLT_MAX / 2; }

    for (int ic0 = 0; ic0 < nblk; ++ic0) {
        const int ic = ic0 * C;
        const bool partial = (ic + C > nkv);
        const int pos0 = partial ? (nkv - C) : ic;
        // K/V source: direct cache rows (K at ic*nkt + head hoff) or the tail pad.
        device const float * ksrc = partial ? (pad + hoff) : (k + ic * nkt + hoff);
        device const float * vsrc = partial ? (pad + C * nkt + hoff) : (v + ic * nkt + hoff);

        // ── Q*K^T ──
        {
            threadgroup const half  * pq = sq;
            threadgroup       float * ps = ss + sgitg * 8;
            device     const float * pk = ksrc + sgitg * (8 * nkt);
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8 mq[2];
                simdgroup_float8x8 mk[2];
                #pragma unroll
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
                    simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);
                    simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * nkt);
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax (causal + pad mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = (iq1 + j < nt) ? positions[iq1 + j] : (int)nkv - 1;
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = pos0 + 2 * tx;
            s2[0] += (kpos0 >= 0 && kpos0 <= qpos) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= 0 && kpos0 + 1 <= qpos) ? 0.0f : -MINF;
            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * (SH / 2) + tx] = vs2;
            for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P * V ──
        {
            simdgroup_float8x8 lo[NO];
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            {
                device const float * pv = vsrc + 8 * sgitg;   // dim offset 8*sgitg
                for (short cc = 0; cc < C / 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    simdgroup_float8x8 mv[2];
                    simdgroup_load(mv[0], pv + 0 * NSG, nkt, 0, false);
                    simdgroup_load(mv[1], pv + 8 * NSG, nkt, 0, false);
                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                    pv += 8 * nkt;
                }
            }
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── store to global ──
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// f16-K/V variant (MINFER_CACHE_TYPE=f16): reads the half K/V cache and tail pad.
// Q is ALWAYS f32 (llama reads Q as float4 regardless of the KV cache type — the
// graph only casts K/V to f16, llama-graph.cpp:2457-2463), so this kernel keeps
// the f32 Q input and only switches the global K/V/pad operands + the simdgroup
// K/V tile types to half8x8.
kernel void kernel_flash_attn_blk_f16(
    device const float * q         [[buffer(0)]],
    device const half *  k         [[buffer(1)]],
    device const half *  v         [[buffer(2)]],
    device const half *  pad       [[buffer(3)]],
    device       float * out      [[buffer(4)]],
    constant    int    * positions [[buffer(5)]],
    constant    int    & nh        [[buffer(6)]],
    constant    int    & nk        [[buffer(7)]],
    constant    int    & hd        [[buffer(8)]],
    constant    float  & scale     [[buffer(9)]],
    constant    int    & nt        [[buffer(10)]],
    constant    int    & nkv       [[buffer(11)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]],
    threadgroup float * shmem [[threadgroup(0)]]
) {
    constexpr int Q   = 8;
    constexpr int C   = 64;
    constexpr int NSG = 4;
    constexpr int DK  = 64;
    constexpr int DV  = 64;
    constexpr int NW  = 32;
    constexpr int NQ  = Q / NSG;
    constexpr int SH  = 2 * C;
    constexpr int DK4 = DK / 4;
    constexpr int DV4 = DV / 4;
    constexpr int DK8 = DK / 8;
    constexpr int PV  = 64;
    constexpr int PV4 = PV / 4;
    constexpr int PV8 = PV / 8;
    constexpr int NC  = (C / 8) / NSG;
    constexpr int NO  = PV8 / NSG;
    constexpr float MINF = 65504.0f;

    const int iq1 = (int)tgpig.x * Q;
    const int iq2 = (int)tgpig.y;
    const int nblk = (nkv + C - 1) / C;

    const int nqt  = nh * hd;
    const int nkt  = nk * hd;
    const int hk   = iq2 / (nh / nk);
    const int hoff = hk * hd;

    const int tx = (int)tiisg;

    threadgroup half  * sq = (threadgroup half  *)shmem;
    threadgroup float * so = (threadgroup float *)(shmem + 256);
    threadgroup float * ss = (threadgroup float *)(shmem + 768);

    threadgroup half4  * sq4 = (threadgroup half4  *)sq;
    threadgroup float4 * so4 = (threadgroup float4 *)so;
    threadgroup float2 * ss2 = (threadgroup float2 *)ss;

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        device const float4 * q4 = (device const float4 *)(q + (iq1 + j) * nqt + iq2 * hd);
        for (int i = tx; i < DK4; i += NW) {
            sq4[j * DK4 + i] = (iq1 + j < nt) ? half4(q4[i]) : (half4)0.0f;
        }
    }

    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] = (float4)0.0f;
        for (int i = tx; i < SH; i += NW) ss[j * SH + i] = 0.0f;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    float S[NQ];
    float M[NQ];
    for (short jj = 0; jj < NQ; ++jj) { S[jj] = 0.0f; M[jj] = -FLT_MAX / 2; }

    for (int ic0 = 0; ic0 < nblk; ++ic0) {
        const int ic = ic0 * C;
        const bool partial = (ic + C > nkv);
        const int pos0 = partial ? (nkv - C) : ic;
        device const half * ksrc = partial ? (pad + hoff) : (k + ic * nkt + hoff);
        device const half * vsrc = partial ? (pad + C * nkt + hoff) : (v + ic * nkt + hoff);

        // ── Q*K^T ──
        {
            threadgroup const half  * pq = sq;
            threadgroup       float * ps = ss + sgitg * 8;
            device     const half  * pk = ksrc + sgitg * (8 * nkt);
            for (short cc = 0; cc < NC; ++cc) {
                simdgroup_float8x8 mqk = make_filled_simdgroup_matrix<float, 8>(0.0f);
                simdgroup_half8x8 mq[2];
                simdgroup_half8x8 mk[2];
                #pragma unroll
                for (short i = 0; i < DK8 / 2; ++i) {
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_load(mq[0], pq + 0 * 8 + 16 * i, DK);
                    simdgroup_load(mq[1], pq + 1 * 8 + 16 * i, DK);
                    simdgroup_load(mk[0], pk + 0 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_load(mk[1], pk + 1 * 8 + 16 * i, nkt, 0, true);
                    simdgroup_barrier(mem_flags::mem_none);
                    simdgroup_multiply_accumulate(mqk, mq[0], mk[0], mqk);
                    simdgroup_multiply_accumulate(mqk, mq[1], mk[1], mqk);
                }
                simdgroup_store(mqk, ps, SH, 0, false);
                pk += 8 * (NSG * nkt);
                ps += 8 * NSG;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── online softmax (causal + pad mask inline) ──
        for (short jj = 0; jj < NQ; ++jj) {
            const short j = jj * NSG + sgitg;
            const int qpos = (iq1 + j < nt) ? positions[iq1 + j] : (int)nkv - 1;
            const float m = M[jj];
            float2 s2 = ss2[j * (SH / 2) + tx] * scale;
            const int kpos0 = pos0 + 2 * tx;
            s2[0] += (kpos0 >= 0 && kpos0 <= qpos) ? 0.0f : -MINF;
            s2[1] += (kpos0 + 1 >= 0 && kpos0 + 1 <= qpos) ? 0.0f : -MINF;
            M[jj] = simd_max(max(M[jj], max(s2[0], s2[1])));
            const float ms = exp(m - M[jj]);
            const float2 vs2 = exp(s2 - M[jj]);
            S[jj] = S[jj] * ms + simd_sum(vs2[0] + vs2[1]);
            ss2[j * (SH / 2) + tx] = vs2;
            for (int i = tx; i < PV4; i += NW) so4[j * PV4 + i] *= ms;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ── O += P * V ──
        {
            simdgroup_float8x8 lo[NO];
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_load(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
            {
                device const half * pv = vsrc + 8 * sgitg;
                for (short cc = 0; cc < C / 8; ++cc) {
                    simdgroup_float8x8 vs;
                    simdgroup_load(vs, ss + 8 * cc, SH, 0, false);
                    simdgroup_half8x8 mv[2];
                    simdgroup_load(mv[0], pv + 0 * NSG, nkt, 0, false);
                    simdgroup_load(mv[1], pv + 8 * NSG, nkt, 0, false);
                    simdgroup_multiply_accumulate(lo[0], vs, mv[0], lo[0]);
                    simdgroup_multiply_accumulate(lo[1], vs, mv[1], lo[1]);
                    pv += 8 * nkt;
                }
            }
            {
                threadgroup float * sot = so + 8 * sgitg;
                for (short ii = 0; ii < NO; ++ii) {
                    simdgroup_store(lo[ii], sot, PV, 0, false);
                    sot += 8 * NSG;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ── store to global ──
    for (short jj = 0; jj < NQ; ++jj) {
        const short j = jj * NSG + sgitg;
        if (iq1 + j >= nt) break;
        device float4 * dst4 = (device float4 *)(out + (iq1 + j) * nqt + iq2 * hd);
        const float inv = S[jj] == 0.0f ? 0.0f : 1.0f / S[jj];
        for (int i = tx; i < PV4; i += NW) dst4[i] = so4[j * PV4 + i] * inv;
    }
}

// Copy the last partial KV block (nkv % 64 != 0) into the [2][64][nkt] flash-prefill
// tail pad: virtual rows [nkv-64, nkv) from the real cache (head offset hoff, both
// K and V), rows outside [0, nkv) zeroed — the causal+pad mask hides them. Grid
// (nkt, 64), one thread per (dim, virtual row). Handles f32 (e=4) or f16 (e=2)
// cache via the `f16` flag.
kernel void kernel_kv_tail_pad(
    device const char * ksrc  [[buffer(0)]],
    device const char * vsrc  [[buffer(1)]],
    device       char * pad   [[buffer(2)]],
    constant    int    & nkv   [[buffer(3)]],
    constant    int    & nkt   [[buffer(4)]],
    constant    int    & f16   [[buffer(5)]],
    uint2  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]]
) {
    const int d   = (int)tgpig.x;
    const int t   = (int)tgpig.y;
    const int e   = f16 ? 2 : 4;
    const int pos = nkv - 64 + t;
    const bool valid = pos >= 0 && pos < nkv;
    const int dst = (t * nkt + d) * e;
    if (valid) {
        const int src = (pos * nkt + d) * e;
        for (int b = 0; b < e; ++b) {
            pad[dst + b] = ksrc[src + b];
            pad[64 * nkt * e + dst + b] = vsrc[src + b];
        }
    } else {
        for (int b = 0; b < e; ++b) {
            pad[dst + b] = 0;
            pad[64 * nkt * e + dst + b] = 0;
        }
    }
}

kernel void kernel_gqa_attn_combine_f32(
    device const float * partial [[buffer(0)]],
    device       float * o       [[buffer(1)]],
    constant    int    & nh       [[buffer(2)]],
    constant    int    & hd       [[buffer(3)]],
    constant    int    & nt       [[buffer(4)]],
    constant    int    & n_chunks [[buffer(5)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]]
) {
    int t = (int)tgpig.x;
    int h = (int)tgpig.y;
    if (t >= nt || h >= nh) return;

    int pbase = (t * nh + h) * n_chunks * (2 + hd);

    // merged running max over the partials
    float m = -INFINITY;
    for (int c = 0; c < n_chunks; c++) {
        m = max(m, partial[pbase + c * (2 + hd) + 0]);
    }
    if (m == -INFINITY) {
        // no partial had data (nkv==0) — not reachable for real heads, but
        // write zeros rather than NaN (exp(-INF - -INF)) for GPU safety.
        device float * ohead = o + t * (nh * hd) + h * hd;
        for (int d = tiisg; d < hd; d += 32) ohead[d] = 0.0f;
        return;
    }

    float l = 0.0f;
    float acc[256];
    for (int d = 0; d < hd; d++) acc[d] = 0.0f;
    for (int c = 0; c < n_chunks; c++) {
        int cbase = pbase + c * (2 + hd);
        float e = exp(partial[cbase + 0] - m);
        l += partial[cbase + 1] * e;
        for (int d = tiisg; d < hd; d += 32) {
            acc[d] += partial[cbase + 2 + d] * e;
        }
    }

    float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
    device float * ohead = o + t * (nh * hd) + h * hd;
    for (int d = tiisg; d < hd; d += 32) {
        ohead[d] = acc[d] * inv;
    }
}

// ─── Q4_K × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same 256-elem super-block structure as Q6_K; Q4_K: 144 B/super-block.
kernel void kernel_q4_k_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q4KB = 144;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 256;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q4_k_16(weights + (r0 + lr0) * nblk * Q4KB + (loop_k/256) * Q4KB,
                        ((loop_k % 256) / 16) + il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q5_K × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same 256-elem super-block structure; Q5_K: 176 B/super-block.
kernel void kernel_q5_k_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q5KB = 176;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 256;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q5_k_16(weights + (r0 + lr0) * nblk * Q5KB + (loop_k/256) * Q5KB,
                        ((loop_k % 256) / 16) + il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        for (short ik = 0; ik < NK/8; ik++) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}
