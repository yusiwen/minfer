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

