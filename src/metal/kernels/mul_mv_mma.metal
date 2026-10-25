// ─── few-row MMA matmul: 2..16 activation rows ──────────────────────────────
// Port of llama.cpp's `kernel_mul_mv_mma_gen` (kernels/mul_mv_mma.metal:504-...),
// the family that covers Q4_K and Q6_K when the activation has 2..16 rows. That is
// exactly the window where minfer runs a plain matvec today and llama does not
// (#673): its `mul_mat` asks `use_mma` *before* `use_mm` / `mul_mv_ext` / `mul_mv`
// (ggml-metal-ops.cpp:2606), so a speculative verify at `nt = 3` gets an 8-row tile
// kernel while minfer re-reads the weights once per token.
//
// Shape of one threadgroup: 8*NT output rows × 8*RT activation rows, NSG simdgroups
// splitting K into 64-weight chunks and reducing through threadgroup memory.
//   A = weights, dequantized per 16-weight chunk into an 8x8 f16 fragment
//   B = activations (f32), loaded into f16 fragment tiles
//   C = A×B accumulated in f32 simdgroup matrices
//
// p[] slots (the engine passes ints; see `metal/ops.rs`):
//   0 K (input dim)   1 od (output rows)   2 nt (activation rows)
//   3 output row stride (elements)         4 weight row stride (bytes)
//   5 activation row stride (elements)     6 NSG
//   7 block size in bytes
//
// Grid: x = ceil(od / (8*NT)), y = ceil(nt / (8*RT)), z = 1; TG = (32, NSG).
// Threadgroup memory: NSG * NT * RT * 256 bytes (one 8x8 float tile per partial).

constant constexpr ushort mma_f16_1024_bits = 0x6400;

// The halves 1024 + q for q < 1024 are exact normal values, unlike the subnormal
// q*2^-24 that Metal may flush to zero (llama's norm trick, mul_mv_mma.metal:62).
inline half2 mma_1024_plus(ushort2 q) {
    return as_type<half2>(q | mma_f16_1024_bits);
}

// The A-fragment row and the first B-fragment column lane l holds in an 8x8
// simdgroup matrix (llama's lane maps, mul_mv_mma.metal:23-24).
inline short mma_lane_fm(ushort l) { return ((l / 4) & 4) + ((l / 2) % 4); }
inline short mma_lane_fn(ushort l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

// Reduces the NSG partial sums of an 8*NT x 8*RT tile and writes it to `out`.
template <short NT, short RT>
inline void mma_store(
        thread float (&acc)[RT][NT][2],
        device float * out,
        threadgroup float * red,
        int i01, int i11, int od, int nt, int out_stride, short NSG,
        ushort tiisg, ushort sgitg) {
    constexpr short LANES = 32;
    #pragma unroll
    for (short rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (short t = 0; t < NT; ++t) {
            red[((sgitg * RT + rt) * NT + t) * 64 + 2 * tiisg + 0] = acc[rt][t][0];
            red[((sgitg * RT + rt) * NT + t) * 64 + 2 * tiisg + 1] = acc[rt][t][1];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short idx = sgitg * LANES + tiisg; idx < RT * NT * 64; idx += NSG * LANES) {
        float sum = 0.0f;
        for (short sg = 0; sg < NSG; ++sg) {
            sum += red[sg * (RT * NT * 64) + idx];
        }

        const short rt = idx / (NT * 64);
        const short t  = (idx / 64) % NT;
        const short l  = (idx % 64) / 2;
        const short e  = idx % 2;

        const int r0 = i01 + 8 * t + mma_lane_fm(l);
        const int r1 = i11 + 8 * rt + mma_lane_fn(l) + e;
        if (r0 < od && r1 < nt) {
            out[(long) r1 * out_stride + r0] = sum;
        }
    }
}

// The body both K-quant kernels share; `deq` is a 16-weight dequantizer returning a
// float4x4, the same helper the GEMM family already uses (kernels/dequantize.h).
// The dequantizer as a *compile-time* selector, not a function pointer: a pointer
// template parameter is not inlined reliably, and the first measurement of this arm
// (2.5x slower than the matvec at nt=3) is consistent with the call never being
// flattened. KIND 0 = Q4_K, 1 = Q6_K.
template <short NT, short RT, short NSG, int KIND>
inline void mma_qk_body(
        device const uchar * weights,
        device const float * acts,
        device float * out,
        constant int * p,
        threadgroup float * red,
        uint3 tgpig, ushort tiisg, ushort sgitg) {
    // NSG is a *compile-time* constant: as a runtime value it made the reduction
    // loop and the K-loop stride unrollable, which is llama's arrangement too
    // (`FC_mul_mv_mma_nsg` is a function constant).
    constexpr short NL = 16;   // 16-weight chunks per K-quant super-block

    const int K = p[0], od = p[1], nt = p[2], out_stride = p[3];
    const int w_row = p[4], a_row = p[5];

    const short fm = mma_lane_fm(tiisg);
    const short fn = mma_lane_fn(tiisg);
    const int i01 = (int) tgpig.x * (8 * NT);
    const int i11 = (int) tgpig.y * (8 * RT);

    // A lane's row for tile t, clamped to the last row (llama's src0_row).
    device const uchar * x[NT];
    #pragma unroll
    for (short t = 0; t < NT; ++t) {
        const int r = min(i01 + 8 * t + fm, od - 1);
        x[t] = weights + (long) r * w_row;
    }

    // B: lane j = fm reads chunk values 16*(fm/2) + 8*(fm%2) .. +7 (llama's src1_row).
    device const float4 * y[RT][2];
    #pragma unroll
    for (short rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (short e = 0; e < 2; ++e) {
            const int r = min(i11 + 8 * rt + fn + e, nt - 1);
            y[rt][e] = (device const float4 *) (acts + (long) r * a_row) + 4 * (fm / 2) + 2 * (fm % 2);
        }
    }

    simdgroup_float8x8 mc[RT][NT];
    #pragma unroll
    for (short rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (short t = 0; t < NT; ++t) {
            mc[rt][t] = make_filled_simdgroup_matrix<float, 8>(0.0f);
        }
    }

    const int nch = K / 64;    // 64-weight chunks in the row

    for (int g = sgitg; g < nch; g += NSG) {
        simdgroup_float8x8 mb[RT][8];
        #pragma unroll
        for (short rt = 0; rt < RT; ++rt) {
            const float4 a0 = y[rt][0][16 * g + 0];
            const float4 a1 = y[rt][0][16 * g + 1];
            const float4 b0 = y[rt][1][16 * g + 0];
            const float4 b1 = y[rt][1][16 * g + 1];
            #pragma unroll
            for (short s = 0; s < 4; ++s) {
                mb[rt][s    ].thread_elements()[0] = a0[s];
                mb[rt][s    ].thread_elements()[1] = b0[s];
                mb[rt][s + 4].thread_elements()[0] = a1[s];
                mb[rt][s + 4].thread_elements()[1] = b1[s];
            }
        }

        const int ci = 4 * g + fn / 2;   // 16-weight chunk index within the row

        #pragma unroll
        for (short t = 0; t < NT; ++t) {
            float4x4 w;
            device const uchar * blk = x[t] + (long) (ci / NL) * p[7];
            if (KIND == 0) {
                dequant_q4_k_16(blk, ci % NL, w);
            } else {
                dequant_q6_k_16(blk, ci % NL, w);
            }
            #pragma unroll
            for (short s = 0; s < 8; ++s) {
                // The A fragment is **half**, not float: Apple's matrix unit runs an
                // f16 fragment several times faster than an f32 one, and the weights
                // are 4-6 bit to begin with, so the f32 fragment bought no accuracy.
                // This is llama's split too — its specialized `blk` kernels use
                // `simdgroup_half8x8 ma` with a float B and float accumulators
                // (mul_mv_mma.metal:283-290); the first version of this kernel copied
                // the generic path's f32 A and lost the bandwidth the tile had saved.
                simdgroup_half8x8 ma;
                ma.thread_elements()[0] = (half) w[s / 4][s % 4];
                ma.thread_elements()[1] = (half) w[s / 4 + 2][s % 4];
                #pragma unroll
                for (short rt = 0; rt < RT; ++rt) {
                    simdgroup_multiply_accumulate(mc[rt][t], ma, mb[rt][s], mc[rt][t]);
                }
            }
        }
    }

    float acc[RT][NT][2];
    #pragma unroll
    for (short rt = 0; rt < RT; ++rt) {
        #pragma unroll
        for (short t = 0; t < NT; ++t) {
            acc[rt][t][0] = mc[rt][t].thread_elements()[0];
            acc[rt][t][1] = mc[rt][t].thread_elements()[1];
        }
    }

    mma_store<NT, RT>(acc, out, red, i01, i11, od, nt, out_stride, NSG, tiisg, sgitg);
}

// NSG is baked per instantiation, like llama's `FC_mul_mv_mma_nsg` function constant.
// Only the values the tiling can pick are instantiated: 16 for od <= 6144 (every
// layer of a 7B/14B) and 8 above it (the 152064-row output head).
#define MINFER_MMA_ONE(pname, NT, RT, NSG, KIND)                                          \
kernel void pname(                                                                        \
        device const uchar * weights [[buffer(0)]], device const float * acts [[buffer(1)]], \
        device float * out [[buffer(2)]], constant int * p [[buffer(3)]],                  \
        uint3 tgpig [[threadgroup_position_in_grid]],                                      \
        ushort tiisg [[thread_index_in_simdgroup]],                                        \
        ushort sgitg [[simdgroup_index_in_threadgroup]],                                   \
        threadgroup float * red [[threadgroup(0)]]) {                                      \
    mma_qk_body<NT, RT, NSG, KIND>(weights, acts, out, p, red, tgpig, tiisg, sgitg);        \
}

MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg8_nt1_rt1, 1, 1, 8, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg8_nt1_rt2, 1, 2, 8, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg8_nt2_rt1, 2, 1, 8, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg8_nt2_rt2, 2, 2, 8, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg8_nt4_rt1, 4, 1, 8, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg8_nt4_rt2, 4, 2, 8, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg16_nt1_rt1, 1, 1, 16, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg16_nt1_rt2, 1, 2, 16, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg16_nt2_rt1, 2, 1, 16, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg16_nt2_rt2, 2, 2, 16, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg16_nt4_rt1, 4, 1, 16, 0)
MINFER_MMA_ONE(kernel_q4_k_f32_mma_nsg16_nt4_rt2, 4, 2, 16, 0)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg8_nt1_rt1, 1, 1, 8, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg8_nt1_rt2, 1, 2, 8, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg8_nt2_rt1, 2, 1, 8, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg8_nt2_rt2, 2, 2, 8, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg8_nt4_rt1, 4, 1, 8, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg8_nt4_rt2, 4, 2, 8, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg16_nt1_rt1, 1, 1, 16, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg16_nt1_rt2, 1, 2, 16, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg16_nt2_rt1, 2, 1, 16, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg16_nt2_rt2, 2, 2, 16, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg16_nt4_rt1, 4, 1, 16, 1)
MINFER_MMA_ONE(kernel_q6_k_f32_mma_nsg16_nt4_rt2, 4, 2, 16, 1)
#undef MINFER_MMA_ONE
