
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

