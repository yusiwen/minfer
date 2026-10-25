// ─── Q4_K × f32 matrix multiplication (simdgroup-cooperative) ──
// Q4_K super-block: 256 elements = 8 sub-blocks × 32.
// Block layout (144 bytes): half d, half dmin, uchar scales[12], uchar qs[128].
// Dequant: val = d * scale[sub] * nibble - dmin * min[sub].
// NR0=2 rows per simdgroup, NSG=2 simdgroups per threadgroup => 64 threads.
// Grid: x = ceil(od / (NR0*NSG)), y = nt, TG = (64, 1, 1).

template <short NSG>
kernel void kernel_q4_k_f32_matmul_nsg(
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
    // TG(32, NSG), grid_x = ceil(od/(NR0*NSG)) — same as the q6_K port. NSG is a
    // template parameter since #673's 0.186 ms target: 2 simdgroups (64 threads) per
    // threadgroup is 8x less memory-level parallelism than the reference's small-batch
    // kernel, and both sit at ~23 % of the bandwidth roofline.
    constexpr short NR0 = 2;

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

// ─── Q4_K × f32 prefill (multi-token, weights read once) ─────
//
// The verify band (`1 < nt <= GEMM_MIN_NT`) runs on this kernel, so two things
// are load-bearing and both are structural:
//
// * **Per-row bitwise with `kernel_q4_k_f32_matmul`.** The lane mapping, the
//   block stride (4), the activation loads, the `acc1`/`acc2` float4 sums and the
//   final scale/min expression are the single-row kernel's, verbatim. The only
//   difference is where the token loop sits: `sumf[row][u]` separates the
//   accumulators, so token `u`'s `+=` sequence is the single-row kernel's, in the
//   same order (blocks ascending). That is what makes a spec-verify round's
//   numbers the decode path's numbers (ADR-0010, ADR-0017) — and
//   `small_batch_matvec_is_per_row_exact_for_every_quant` gates it.
// * **One weight pass per round.** The token loop is INSIDE the block loop: the
//   block's scales/nibbles are decoded once and reused for every token, which is
//   the whole point of batching a verify round (a per-token pass re-streams the
//   model's weights). `MAXT` bounds the accumulators, so `nt` stays unbounded
//   (`MINFER_GEMM=0` also sends long prefills here, in chunks of `MAXT`).
//
// Grid: x = ceil(od / (NR0*NSG)), y = 1, TG = (32, NSG, 1) — unchanged.

kernel void kernel_q4_k_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NR0 = 2;
    constexpr short NSG = 2;
    /// Tokens accumulated per weight pass; `nt` itself is unbounded. 4 keeps the
    /// accumulator set (2 rows x 4) inside the register file: at 8 the compiler
    /// spilled them and the kernel cost ~1.6 full weight passes per extra token,
    /// which is worse than the GEMM it is meant to beat.
    constexpr short MAXT = 4;

    constexpr ushort kmask1 = 0x3f3f;
    constexpr ushort kmask2 = 0x0f0f;
    constexpr ushort kmask3 = 0xc0c0;

    const int nb = p[1] / 256;
    const int row_stride = nb * 144;
    const int r0 = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    const short ix = tiisg / 8;                      // 0...3
    const short it = tiisg % 8;                      // 0...7
    const short iq = it / 4;                         // 0 or 1
    const short ir = it % 4;                         // 0...3

    // A threadgroup's last tile can start past the output rows (od % 4 != 0), and
    // only the guard on the store would catch it — but row 1's pointers are used
    // unconditionally below, so both are clamped here instead (#431: the previous
    // body read past the weight tensor for such a tile).
    if (r0 >= p[0]) return;
    const bool has1 = (r0 + 1) < p[0];

    device const uchar * wrow[NR0];
    wrow[0] = weights + (r0 + 0) * row_stride;
    wrow[1] = has1 ? weights + (r0 + 1) * row_stride : wrow[0];

    for (int t0 = 0; t0 < p[2]; t0 += MAXT) {
        const int tn = min((int)MAXT, p[2] - t0);
        float sumf[NR0][MAXT];
        for (short row = 0; row < NR0; ++row) {
            for (short u = 0; u < MAXT; ++u) sumf[row][u] = 0.0f;
        }

        for (int ib = ix; ib < nb; ib += 4) {
            // Decode this block's weight fields once, for both rows.
            ushort q1r[NR0][4];
            ushort q2r[NR0][4];
            ushort sc16[NR0][4];
            float  dh0r[NR0];
            float  dh1r[NR0];
            for (short row = 0; row < NR0; ++row) {
                device const uchar * blk = wrow[row] + ib * 144;
                device const ushort * sc = (device const ushort *)(blk + 4) + iq;
                device const ushort * q1 = (device const ushort *)(blk + 16) + 16 * iq + 4 * ir;
                device const half   * dh = (device const half *)(blk + 0);

                const ushort sc16_0 = (ushort)(sc[0] & kmask1);
                const ushort sc16_1 = (ushort)(sc[2] & kmask1);
                const ushort sc16_2 = (ushort)(((sc[4] >> 0) & kmask2) | ((sc[0] & kmask3) >> 2));
                const ushort sc16_3 = (ushort)(((sc[4] >> 4) & kmask2) | ((sc[2] & kmask3) >> 2));

                sc16[row][0] = sc16_0;
                sc16[row][1] = sc16_1;
                sc16[row][2] = sc16_2;
                sc16[row][3] = sc16_3;

                for (short i = 0; i < 4; ++i) {
                    q1r[row][i] = q1[i];
                    q2r[row][i] = q1[32 + i];
                }
                dh0r[row] = float(dh[0]);
                dh1r[row] = float(dh[1]);
            }

            for (int u = 0; u < tn; ++u) {
                // The single-row kernel's activation slice, verbatim.
                device const float * y4 = acts + (t0 + u) * p[1] + ib * 256 + 64 * iq + 8 * ir;

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
                    float4 acc1 = { 0.0f, 0.0f, 0.0f, 0.0f };
                    float4 acc2 = { 0.0f, 0.0f, 0.0f, 0.0f };

                    for (short i = 0; i < 4; ++i) {
                        acc1[0] += yl[2*i + 0] * (float)(q1r[row][i] & 0x000F);
                        acc1[1] += yl[2*i + 1] * (float)(q1r[row][i] & 0x0F00);
                        acc1[2] += yl[2*i + 8] * (float)(q1r[row][i] & 0x00F0);
                        acc1[3] += yl[2*i + 9] * (float)(q1r[row][i] & 0xF000);
                        acc2[0] += yh[2*i + 0] * (float)(q2r[row][i] & 0x000F);
                        acc2[1] += yh[2*i + 1] * (float)(q2r[row][i] & 0x0F00);
                        acc2[2] += yh[2*i + 8] * (float)(q2r[row][i] & 0x00F0);
                        acc2[3] += yh[2*i + 9] * (float)(q2r[row][i] & 0xF000);
                    }

                    // The scale bytes are derived here rather than pre-expanded:
                    // the values are identical (same masks), and four `ushort`
                    // words instead of eight floats keeps the per-block state
                    // small enough to stay in registers.
                    const float sc8_0 = (float)(sc16[row][0] & 0xFF);
                    const float sc8_1 = (float)(sc16[row][0] >> 8);
                    const float sc8_2 = (float)(sc16[row][1] & 0xFF);
                    const float sc8_3 = (float)(sc16[row][1] >> 8);
                    const float sc8_4 = (float)(sc16[row][2] & 0xFF);
                    const float sc8_5 = (float)(sc16[row][2] >> 8);
                    const float sc8_6 = (float)(sc16[row][3] & 0xFF);
                    const float sc8_7 = (float)(sc16[row][3] >> 8);

                    sumf[row][u] +=
                        dh0r[row] * ((acc1[0] + (1.0f/256.0f) * acc1[1]) * sc8_0 +
                                     (acc1[2] + (1.0f/256.0f) * acc1[3]) * sc8_1 * (1.0f/16.0f) +
                                     (acc2[0] + (1.0f/256.0f) * acc2[1]) * sc8_4 +
                                     (acc2[2] + (1.0f/256.0f) * acc2[3]) * sc8_5 * (1.0f/16.0f)) -
                        dh1r[row] * (sumy[0] * sc8_2 + sumy[1] * sc8_3 +
                                     sumy[2] * sc8_6 + sumy[3] * sc8_7);
                }
            }
        }

        for (int u = 0; u < tn; ++u) {
            const float s0 = simd_sum(sumf[0][u]);
            const float s1 = simd_sum(sumf[1][u]);
            if (tiisg == 0) {
                output[(t0 + u) * p[0] + r0 + 0] = s0;
                if (has1) output[(t0 + u) * p[0] + r0 + 1] = s1;
            }
        }
    }
}

// ─── Q6_K dot: ONE arithmetic, two consumers ─────────────────
//
// The decode kernel (`nt == 1`) and the verify band (`1 < nt <= GEMM_MIN_NT`)
// must produce bit-identical values for the same (row, position), because a
// speculative round's identity contract is "the verify rows are the decode
// path's own numbers" (ADR-0010, ADR-0017). Sharing the *source* of the dot —
// rather than repeating its expression — is what makes that structural: Metal is
// free to contract and reassociate float expressions, and a verbatim copy in a
// second kernel compiles differently. Measured (M4 Pro, Q6_K, full-mantissa
// data): a copied expression disagreed by ~1 ULP in 72% of the outputs, while
// the same function inlined in both kernels is exact.
inline float q6k_block_dot(
    device const uchar * q1,
    device const uchar * q2,
    device const uchar * qh,
    device const int8_t * sc,
    float d,
    thread const float * yl
) {
    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    float4 sums = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (short l = 0; l < 4; ++l) {
        sums[0] += yl[4*l + 0] * ((int8_t)((q1[l] & 0xF) | ((qh[l] & kmask1) << 4)) - 32);
        sums[1] += yl[4*l + 1] * ((int8_t)((q2[l] & 0xF) | ((qh[l] & kmask2) << 2)) - 32);
        sums[2] += yl[4*l + 2] * ((int8_t)((q1[l]  >> 4) | ((qh[l] & kmask3) << 0)) - 32);
        sums[3] += yl[4*l + 3] * ((int8_t)((q2[l]  >> 4) | ((qh[l] & kmask4) >> 2)) - 32);
    }
    return d * (sums[0] * float(sc[0]) + sums[1] * float(sc[2])
              + sums[2] * float(sc[4]) + sums[3] * float(sc[6]));
}

// ─── Q6_K × f32 matrix multiplication (simdgroup-cooperative) ──
// Q6_K super-block: 256 elements = 16 sub-blocks × 16.
// Block layout (210 bytes): uchar ql[128], uchar qh[64], char scales[16], half d.
// Dequant: val = d * scales[sub] * ((low4 | (high2 << 4)) - 32).
// NR0=2, NSG=2, TG=64. Grid: x = ceil(od/4), y = nt.

template <short NSG>
kernel void kernel_q6_k_f32_matmul_nsg(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    // Faithful port of llama's kernel_mul_mv_q6_K_f32_impl (stride-2 thread
    // layout, float4 sums). Dispatch: TG(32, NSG), grid_x = ceil(od/(NR0*NSG)).
    constexpr short NR0 = 2;

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
            sumf[row] += q6k_block_dot(q1, q2, qh, sc, float(dh[0]), yl);

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

// ─── Q6_K × f32 prefill (multi-token, weights read once) ─────
//
// Same two structural requirements as the Q4_K sibling above: the single-row
// kernel's lane mapping (`tid`/`ix`/`ip`/`il`), stride (2) and `sums[0..3]`
// expression are kept verbatim so each token's accumulation is the decode path's
// bit for bit, while the token loop sits INSIDE the block loop so one pass over
// the weights serves the whole verify round. Grid unchanged:
// x = ceil(od / (NR0*NSG)), y = 1, TG = (32, NSG, 1).

kernel void kernel_q6_k_f32_matmul_multi(
    device const uchar  * weights  [[buffer(0)]],
    device const float  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     * p        [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    constexpr short NR0 = 2;
    constexpr short NSG = 2;
    constexpr short MAXT = 4;   // see the Q4_K sibling's note on register pressure

    const int nb = p[1] / 256;
    const int row_stride = nb * 210;
    const int r0 = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    const short tid = tiisg / 2;
    const short ix  = tiisg % 2;
    const short ip  = tid / 8;                       // 0 or 1
    const short il  = tid % 8;
    const short l0  = 4 * il;
    const short is  = 8 * ip + l0 / 16;

    const short y_offset   = 128 * ip + l0;
    const short q_offset_l =  64 * ip + l0;
    const short q_offset_h =  32 * ip + l0;

    if (r0 >= p[0]) return;
    const bool has1 = (r0 + 1) < p[0];

    device const uchar * wrow[NR0];
    wrow[0] = weights + (r0 + 0) * row_stride;
    wrow[1] = has1 ? weights + (r0 + 1) * row_stride : wrow[0];

    for (int t0 = 0; t0 < p[2]; t0 += MAXT) {
        const int tn = min((int)MAXT, p[2] - t0);
        float sumf[NR0][MAXT];
        for (short row = 0; row < NR0; ++row) {
            for (short u = 0; u < MAXT; ++u) sumf[row][u] = 0.0f;
        }

        for (int i = ix; i < nb; i += 2) {
            // This block's weight pointers, once for both rows.
            device const uchar * q1p[NR0];
            device const uchar * q2p[NR0];
            device const uchar * qhp[NR0];
            device const int8_t * scp[NR0];
            float dh0r[NR0];
            for (short row = 0; row < NR0; ++row) {
                device const uchar * blk = wrow[row] + i * 210;
                q1p[row] = blk + q_offset_l;
                q2p[row] = blk + q_offset_l + 32;
                qhp[row] = blk + 128 + q_offset_h;
                scp[row] = (device const int8_t *)(blk + 192) + is;
                dh0r[row] = float(*(device const half *)(blk + 208));
            }

            for (int u = 0; u < tn; ++u) {
                // The single-row kernel's activation slice, verbatim.
                device const float * y = acts + (t0 + u) * p[1] + i * 256 + y_offset;
                float yl[16];
                for (short l = 0; l < 4; ++l) {
                    yl[4*l + 0] = y[l +  0];
                    yl[4*l + 1] = y[l + 32];
                    yl[4*l + 2] = y[l + 64];
                    yl[4*l + 3] = y[l + 96];
                }

                // The SAME dot the decode kernel calls — see q6k_block_dot.
                for (short row = 0; row < NR0; ++row) {
                    sumf[row][u] += q6k_block_dot(q1p[row], q2p[row], qhp[row],
                                                  scp[row], dh0r[row], yl);
                }
            }
        }

        for (int u = 0; u < tn; ++u) {
            const float s0 = simd_sum(sumf[0][u]);
            const float s1 = simd_sum(sumf[1][u]);
            if (tiisg == 0) {
                output[(t0 + u) * p[0] + r0 + 0] = s0;
                if (has1) output[(t0 + u) * p[0] + r0 + 1] = s1;
            }
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

// #673: the shipped arm (NSG=2, 64 threads) and the wide variant (NSG=8, 256 threads).
// Each output row is still summed by one simdgroup with the unchanged lane layout, so
// the two are bitwise identical — the existing gates check that.
template [[host_name("kernel_q4_k_f32_matmul")]]      kernel void kernel_q4_k_f32_matmul_nsg<2>(device const uchar *, device const float *, device float *, constant int *, uint3, ushort, ushort);
template [[host_name("kernel_q4_k_f32_matmul_nsg8")]] kernel void kernel_q4_k_f32_matmul_nsg<8>(device const uchar *, device const float *, device float *, constant int *, uint3, ushort, ushort);
template [[host_name("kernel_q6_k_f32_matmul")]]      kernel void kernel_q6_k_f32_matmul_nsg<2>(device const uchar *, device const float *, device float *, constant int *, uint3, ushort, ushort);
template [[host_name("kernel_q6_k_f32_matmul_nsg8")]] kernel void kernel_q6_k_f32_matmul_nsg<8>(device const uchar *, device const float *, device float *, constant int *, uint3, ushort, ushort);
