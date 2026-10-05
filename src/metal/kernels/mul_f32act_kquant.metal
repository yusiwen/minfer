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
    // Faithful port of llama's kernel_mul_mv_q4_K_f32_impl (stride-4 super-block
    // thread layout, float4 sums, kmask nibble/scale unpack). Dispatch:
    // TG(32, nsg=2), grid_x = od/(nr0*nsg) — same as the q6_K port.
    constexpr short NR0 = 2;
    constexpr short NSG = 2;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const int nb = p[1] / 256;                       // super-blocks per row
    const int r0 = (int)tgpig.x;                     // grid tile (not row!)
    const int t  = (int)tgpig.y;
    if (t >= p[2]) return;

    const int first_row = (r0 * NSG + (int)sgitg) * NR0;

    const int row_stride = nb * 144;                 // Q4_K block bytes
    device const uchar * x0 = weights + first_row * row_stride;
    device const float * yy = acts + t * p[1];

    float sumf[NR0] = { 0.0f, 0.0f };

    const short ix = tiisg / 8;                      // 0...3
    const short it = tiisg % 8;                      // 0...7
    const short iq = it / 4;                         // 0 or 1
    const short ir = it % 4;                         // 0...3

    for (int ib = ix; ib < nb; ib += 4) {
        device const uchar * blk = x0 + ib * 144;
        device const ushort * sc = (device const ushort *)(blk + 4) + iq;
        device const ushort * q1 = (device const ushort *)(blk + 16) + 16 * iq + 4 * ir;
        device const half   * dh = (device const half *)(blk + 0);

        device const float * y4 = yy + ib * 256 + 64 * iq + 8 * ir;

        float yl[16];
        float yh[16];
        float4 sumy = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (short i = 0; i < 8; ++i) {
            yl[i +  0] = y4[i +  0]; sumy[0] += yl[i +  0];
            yl[i +  8] = y4[i + 32]; sumy[1] += yl[i +  8];
            yh[i +  0] = y4[i +128]; sumy[2] += yh[i +  0];
            yh[i +  8] = y4[i +160]; sumy[3] += yh[i +  8];
        }

        for (short row = 0; row < NR0; ++row) {
            // llama's sc16/sc8 unpack — byte view of the 4 masked uint16 words.
            const ushort sc16_0 = (ushort)(sc[0] & kmask1);
            const ushort sc16_1 = (ushort)(sc[2] & kmask1);
            const ushort sc16_2 = (ushort)(((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2));
            const ushort sc16_3 = (ushort)(((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2));

            const float sc8_0 = (float)(sc16_0 & 0xFF);
            const float sc8_1 = (float)(sc16_0 >> 8);
            const float sc8_2 = (float)(sc16_1 & 0xFF);
            const float sc8_3 = (float)(sc16_1 >> 8);
            const float sc8_4 = (float)(sc16_2 & 0xFF);
            const float sc8_5 = (float)(sc16_2 >> 8);
            const float sc8_6 = (float)(sc16_3 & 0xFF);
            const float sc8_7 = (float)(sc16_3 >> 8);

            device const ushort * q2 = q1 + 32;

            float4 acc1 = { 0.0f, 0.0f, 0.0f, 0.0f };
            float4 acc2 = { 0.0f, 0.0f, 0.0f, 0.0f };

            for (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (float)(q1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (float)(q1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (float)(q1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (float)(q1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (float)(q2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (float)(q2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (float)(q2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (float)(q2[i] & 0xF000);
            }

            sumf[row] += float(dh[0]) * ((acc1[0] + (1.0f/256.0f) * acc1[1]) * sc8_0 +
                                         (acc1[2] + (1.0f/256.0f) * acc1[3]) * sc8_1 * (1.0f/16.0f) +
                                         (acc2[0] + (1.0f/256.0f) * acc2[1]) * sc8_4 +
                                         (acc2[2] + (1.0f/256.0f) * acc2[3]) * sc8_5 * (1.0f/16.0f)) -
                        float(dh[1]) * (sumy[0] * sc8_2 + sumy[1] * sc8_3 + sumy[2] * sc8_6 + sumy[3] * sc8_7);

            q1 += row_stride / 2;
            sc += row_stride / 2;
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
        // RACE FIX (Qwen3 GPU nondeterminism): without a trailing barrier the
        // next iteration's `sh0[tiisg] = 0.0f` zeroing (top of the loop) can
        // overtake a slow thread still reading `sh0[tiisg]` in the simd_sum
        // above — it then reduces over zeros and writes a wrong (often 0)
        // output element. All threads must finish their shmem reads before
        // anyone re-zeroes the buffer. Only this kernel cooperates across
        // simdgroups via shmem inside a t-loop, so only it needs the barrier.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

