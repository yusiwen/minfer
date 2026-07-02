// Metal shaders for minfer — Q4_0 matmul + element-wise ops.
// Translated from: llama.cpp/ggml/src/ggml-metal/ggml-metal.metal

#include <metal_stdlib>
using namespace metal;

constant int Q4B = 18;

// ─── Q4_0 × Q8_0 matrix multiplication (bit-exact with CPU) ───

constant int Q8B = 34;

kernel void kernel_q4_0_q8_0_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const uchar  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
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

    if (t >= nt || r0 >= od) return;

    const int nb  = id / 32;
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
            if (o >= od) break;

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
        if (o < od) {
            float total = simd_sum(sumf[row]);
            if (tiisg == 0) {
                output[t * od + o] = total;
            }
        }
    }
}

// ─── Q4_0 × f32 matrix multiplication (simdgroup-cooperative) ──
// Direct translation of llama.cpp mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>.
// Threadgroup layout: NSG=2 simdgroups × NW=32 lanes = 64 threads.
// Each simdgroup handles NR0=4 consecutive output rows.
// Grid: x = ceil(od / (NR0*NSG)), y = nt, TG = 64 threads.

constant short NW_Q = 32;
constant short NQ_Q = 16;
constant short QK   = 32;
constant int   Q41B = 20;

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
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = id / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= nt) return;

    const int q4s = nb * Q4B;
    device const uchar * ax0 = weights + (r0 + 0) * q4s;
    device const uchar * ax1 = weights + (r0 + 1) * q4s;
    device const uchar * ax2 = weights + (r0 + 2) * q4s;
    device const uchar * ax3 = weights + (r0 + 3) * q4s;
    device const float  * y  = acts + t * id;

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
        if (r0 + 0 < od) sumf0 += block_q4_0_dot_y(ax0 + ib * Q4B, sy, yl, il);
        if (r0 + 1 < od) sumf1 += block_q4_0_dot_y(ax1 + ib * Q4B, sy, yl, il);
        if (r0 + 2 < od) sumf2 += block_q4_0_dot_y(ax2 + ib * Q4B, sy, yl, il);
        if (r0 + 3 < od) sumf3 += block_q4_0_dot_y(ax3 + ib * Q4B, sy, yl, il);
        yb += QK * NQ_Q;
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    sumf2 = simd_sum(sumf2);
    sumf3 = simd_sum(sumf3);
    if (tiisg == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
        if (r0 + 2 < od) output[t * od + r0 + 2] = sumf2;
        if (r0 + 3 < od) output[t * od + r0 + 3] = sumf3;
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
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int nb  = id / QK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= nt) return;

    const int q41s = nb * Q41B;
    device const uchar * ax0 = weights + (r0 + 0) * q41s;
    device const uchar * ax1 = weights + (r0 + 1) * q41s;
    device const uchar * ax2 = weights + (r0 + 2) * q41s;
    device const uchar * ax3 = weights + (r0 + 3) * q41s;
    device const float  * y  = acts + t * id;

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
        if (r0 + 0 < od) sumf0 += block_q4_1_dot_y(ax0 + ib * Q41B, sy, yl, il);
        if (r0 + 1 < od) sumf1 += block_q4_1_dot_y(ax1 + ib * Q41B, sy, yl, il);
        if (r0 + 2 < od) sumf2 += block_q4_1_dot_y(ax2 + ib * Q41B, sy, yl, il);
        if (r0 + 3 < od) sumf3 += block_q4_1_dot_y(ax3 + ib * Q41B, sy, yl, il);
        yb += QK * NQ_Q;
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    sumf2 = simd_sum(sumf2);
    sumf3 = simd_sum(sumf3);
    if (tiisg == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
        if (r0 + 2 < od) output[t * od + r0 + 2] = sumf2;
        if (r0 + 3 < od) output[t * od + r0 + 3] = sumf3;
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
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256;
    const int Q4KB = 144;
    const short NR0 = 2;
    const short NSG = 2;
    const short NW  = 32;

    const int nbe = id / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= nt) return;

    const int row_stride = nbe * Q4KB;
    device const uchar * w0 = weights + (r0 + 0) * row_stride;
    device const uchar * w1 = weights + (r0 + 1) * row_stride;
    device const float  * y  = acts + t * id;

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

        for (int s = 0; s < 8; s++) {
            float dsc0 = bd0 * sc0_s[s]; float dmn0 = bm0 * sc0_m[s];
            float dsc1 = bd1 * sc1_s[s]; float dmn1 = bm1 * sc1_m[s];

            // llama.cpp Q4_K nibble format: byte j low nibble = elem j, byte j high nibble = elem j+16
            device const uchar * qb0 = qs0 + s * 16;
            device const uchar * qb1 = qs1 + s * 16;
            device const float  * ys = yb + s * 32;

            float acc0 = 0.0f, acc1 = 0.0f, sumy = 0.0f;
            for (int j = 0; j < 16; j++) {
                uchar b0 = qb0[j];
                uchar b1 = qb1[j];
                float y_lo = ys[j];
                float y_hi = ys[j + 16];
                acc0 += float(b0 & 0x0F) * y_lo + float(b0 >> 4) * y_hi;
                acc1 += float(b1 & 0x0F) * y_lo + float(b1 >> 4) * y_hi;
                sumy += y_lo + y_hi;
            }
            sumf0 += dsc0 * acc0 - dmn0 * sumy;
            sumf1 += dsc1 * acc1 - dmn1 * sumy;
        }
    }

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    if (tiisg == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
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
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const int QKK = 256;
    const int Q6KB = 210;
    const short NR0 = 2;
    const short NSG = 2;
    const short NW  = 32;

    const int nbe = id / QKK;
    const int r0  = ((int)tgpig.x * NSG + (int)sgitg) * NR0;
    const int t   = (int)tgpig.y;
    if (t >= nt) return;

    const int row_stride = nbe * Q6KB;
    device const uchar * w0 = weights + (r0 + 0) * row_stride;
    device const uchar * w1 = weights + (r0 + 1) * row_stride;
    device const float  * y  = acts + t * id;

    float sumf0 = 0.0f, sumf1 = 0.0f;

    for (int ib = (int)tiisg; ib < nbe; ib += NW) {
        device const uchar * blk0 = w0 + ib * Q6KB;
        device const uchar * blk1 = w1 + ib * Q6KB;

        float bd0 = float(*(device const half *)(blk0 + 208));
        float bd1 = float(*(device const half *)(blk1 + 208));
        device const uchar * ql0 = blk0;
        device const uchar * ql1 = blk1;
        device const uchar * qh0 = blk0 + 128;
        device const uchar * qh1 = blk1 + 128;
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

    sumf0 = simd_sum(sumf0);
    sumf1 = simd_sum(sumf1);
    if (tiisg == 0) {
        if (r0 + 0 < od) output[t * od + r0 + 0] = sumf0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = sumf1;
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
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup float  * shmem     [[threadgroup(0)]]
) {
    const short NR0 = 2;
    const short NSG = 4;
    const short NW  = 32;
    const short NQ  = 8;

    const int nb = id / QK;
    const int r0 = (int)tgpig.x * NR0;
    const int t  = (int)tgpig.y;
    if (t >= nt || r0 >= od) return;

    const int q8s = nb * Q8B;
    device const float * y = acts + t * id;

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
        if (r0 + 0 < od) output[t * od + r0 + 0] = tot0;
        if (r0 + 1 < od) output[t * od + r0 + 1] = tot1;
    }
}

// ─── Quantize f32 → Q8_0 (1 thread per 32-element block) ─────
// Matches CPU scalar path: half delta + 32 signed int8 values.

kernel void kernel_quantize_q8_0(
    device const float * x   [[buffer(0)]],
    device       uchar * y   [[buffer(1)]],
    constant    int    & dim [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    int nb = dim / 32;
    int total = nt * nb;
    if ((int)tid >= total) return;

    int t = (int)tid / nb;
    int b = (int)tid % nb;

    device const float * src = x + t * dim + b * 32;
    device       uchar * dst = y + (t * nb + b) * Q8B;

    float am = 0.0f;
    for (int j = 0; j < 32; j++) am = fmax(am, fabs(src[j]));
    float d  = am / 127.0f;
    float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    device half * dptr = (device half *)dst;
    *dptr = half(d);

    for (int j = 0; j < 32; j++) {
        int q = int(round(src[j] * id));
        q = clamp(q, -128, 127);
        dst[2 + j] = uchar(q);
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
    ss = simd_sum(ss);

    float scale = 1.0f / sqrt(ss / (float)d + eps);

    device float4 * y4 = (device float4 *)(y + row * d);
    device const float4 * w4 = (device const float4 *)w;
    for (int i = tpitg.x; i < d4; i += 32) {
        y4[i] = x4[i] * scale * w4[i];
    }
}

// ─── Add bias ────────────────────────────────────────────────
// y[t][i] += b[i]

kernel void kernel_add_bias_f32(
    device       float * y [[buffer(0)]],
    device const float * b [[buffer(1)]],
    constant    int    & d [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int t = tid.x, i = tid.y;
    if (i >= d) return;
    y[t * d + i] += b[i];
}

// ─── Element-wise add ────────────────────────────────────────
// z = x + y

kernel void kernel_add_f32(
    device const float * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device       float * z [[buffer(2)]],
    constant    int    & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n) return;
    z[tid] = x[tid] + y[tid];
}

// ─── Element-wise multiply ───────────────────────────────────
// z[t] = x[t] * y[t]

kernel void kernel_mul_f32(
    device const float * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device       float * z [[buffer(2)]],
    constant    int    & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n) return;
    z[tid] = x[tid] * y[tid];
}

// ─── SiLU (in-place) ─────────────────────────────────────────
// y[i] = y[i] / (1 + exp(-y[i]))

kernel void kernel_silu_f32(
    device float * y [[buffer(0)]],
    constant int & n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n) return;
    float v = y[tid];
    y[tid] = v / (1.0 + exp(-v));
}

// ─── SwiGLU (fused SiLU + Mul) ───────────────────────────────
// dst[i] = silu(gate[i]) * up[i]

kernel void kernel_swiglu_f32(
    device const float * gate [[buffer(0)]],
    device const float * up   [[buffer(1)]],
    device       float * dst  [[buffer(2)]],
    constant    int    & n    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if ((int)tid >= n) return;
    float g = gate[tid];
    dst[tid] = (g / (1.0f + exp(-g))) * up[tid];
}

// ─── RoPE (in-place) ─────────────────────────────────────────
// Applies rotary positional embedding to Q and K.
// x layout: [nt][n_head][n_dims]

kernel void kernel_rope_f32(
    device float * x [[buffer(0)]],
    constant int & n_head [[buffer(1)]],
    constant int & n_dims [[buffer(2)]],
    constant int & nt [[buffer(3)]],
    constant float & freq_base [[buffer(4)]],
    constant float & freq_scale [[buffer(5)]],
    constant int * positions [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int h = tid.y;
    if (t >= nt || h >= n_head) return;
    int half_dim = n_dims / 2;
    int base = (t * n_head + h) * n_dims;
    for (int i = 0; i < half_dim; i++) {
        float freq = freq_scale / pow(freq_base, (2.0 * i) / n_dims);
        float theta = positions[t] * freq;
        float cs = cos(theta), sn = sin(theta);
        int j = base + i;
        int j2 = j + half_dim;
        float x0 = x[j], x1 = x[j2];
        x[j]  = x0 * cs - x1 * sn;
        x[j2] = x0 * sn + x1 * cs;
    }
}

// ─── GQA Attention ───────────────────────────────────────────
// Computes grouped-query attention for one or more tokens.
// q/k/v/o layout: [nt][nh][hd] for q/o, [nkv][nk][hd] for k/v.

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

kernel void kernel_gqa_attn_f32(
    device const float * q [[buffer(0)]],
    device const float * k [[buffer(1)]],
    device const float * v [[buffer(2)]],
    device       float * o [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    int    & nh [[buffer(5)]],
    constant    int    & nk [[buffer(6)]],
    constant    int    & hd [[buffer(7)]],
    constant    float  & scale [[buffer(8)]],
    constant    int    & nt [[buffer(9)]],
    uint2 tg_id [[threadgroup_position_in_grid]],
    uint tiisg [[thread_index_in_simdgroup]]
) {
    int t = tg_id.x;
    int h = tg_id.y;
    if (t >= nt || h >= nh) return;

    int nkv = positions[t] + 1;
    int gqa = nh / nk;
    int hk = h / gqa;
    int ne_q = nh * hd;
    int stride_kv = nk * hd;

    device const float * qhead = q + t * ne_q + h * hd;
    device const float * khead = k + hk * hd;
    device const float * vhead = v + hk * hd;
    device       float * ohead = o + t * ne_q + h * hd;

    int hd4 = hd / 4;
    device const float4 * q4 = (device const float4 *)qhead;

    const int NE = 2;
    const int C = 32 * NE;

    float mx = -INFINITY;
    float S = 0.0f;
    float4 oc[32];
    for (int i = 0; i < hd4; i++) oc[i] = (float4)0.0f;

    for (int batch = 0; batch < nkv; batch += C) {
        float s0 = -INFINITY, s1 = -INFINITY;
        int kv0 = batch + tiisg * NE;
        int kv1 = kv0 + 1;

        if (kv0 < nkv) {
            device const float4 * k4 = (device const float4 *)(khead + kv0 * stride_kv);
            float d = 0.0f;
            for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);
            s0 = d * scale;
        }
        if (kv1 < nkv) {
            device const float4 * k4 = (device const float4 *)(khead + kv1 * stride_kv);
            float d = 0.0f;
            for (int i = 0; i < hd4; i++) d += dot(q4[i], k4[i]);
            s1 = d * scale;
        }

        float batch_mx = simd_max(max(s0, s1));
        float new_mx = max(mx, batch_mx);
        float corr = exp(mx - new_mx);
        float e0 = exp(s0 - new_mx);
        float e1 = exp(s1 - new_mx);

        for (int i = 0; i < hd4; i++) oc[i] *= corr;
        S *= corr;

        if (kv0 < nkv) {
            device const float4 * v4 = (device const float4 *)(vhead + kv0 * stride_kv);
            for (int i = 0; i < hd4; i++) oc[i] += e0 * v4[i];
        }
        if (kv1 < nkv) {
            device const float4 * v4 = (device const float4 *)(vhead + kv1 * stride_kv);
            for (int i = 0; i < hd4; i++) oc[i] += e1 * v4[i];
        }
        S += e0;
        S += e1;

        mx = new_mx;
    }

    S = simd_sum(S);
    for (int i = 0; i < hd4; i++) oc[i] = simd_sum(oc[i]);

    float inv = (S > 0.0f) ? (1.0f / S) : 0.0f;
    device float4 * o4 = (device float4 *)ohead;
    for (int i = 0; i < hd4; i++) o4[i] = oc[i] * inv;
}
