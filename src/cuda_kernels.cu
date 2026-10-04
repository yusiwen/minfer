// CUDA kernels for minfer — the shrinking remainder of the single
// pre-#263 translation unit.
//
// Each stage of #263 moved one kernel family into src/cuda/kernels/;
// this file is deleted when it is empty (stage 6).
#include "cuda/kernels/common.cuh"

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

// #141: f16 token-embedding row gather. An f16 GGUF stores token_embd.weight
// as f16, so the embed op needs a device path in the same type — a registered
// matmul-only f16 would still drop the whole model to the CPU through the
// all-or-nothing `weights_on_cuda` gate. One thread per output element.
__global__ void embed_rows_f16(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    long long tid = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (long long)nt * n_embd) return;
    int t = (int)(tid / n_embd), i = (int)(tid % n_embd);
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    const __half* row = reinterpret_cast<const __half*>(w + (size_t)id * n_embd * 2);
    out[tid] = __half2float(row[i]);
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
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (r0 >= od) return;

    int nch = (id + CHK - 1) / CHK;
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

// ─── F16 × F32 matmul (#141) ──────────────────────────────────
// f16 weight matmul for the device backends. The weights stay 2 bytes on the
// device (that is the point of an f16 GGUF: half the weight stream of F32), so
// this does NOT dequantize to an f32 buffer at registration; it converts each
// half to float in-register. Same NR0/NSG unit mapping and token-in-block loop
// as f32_f32_matmul_vec, so the token dimension is not in the launch grid.
// half2 loads need id even; the vec kernel is used when id % 8 == 0 and the
// scalar kernel covers the general case.
__global__ void f16_f32_matmul_vec(
    const __half* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    const int NR0 = 4;
    const int NSG = 2;
    const int CHK = 256;

    int warp_id = threadIdx.x / WARP;
    int lane_id = threadIdx.x % WARP;
    int r0 = (blockIdx.x * NSG + warp_id) * NR0;
    if (r0 >= od) return;

    int nch = (id + CHK - 1) / CHK;
    for (int t = 0; t < nt; ++t) {
        const float* y = acts + (size_t)t * id;

        float acc[NR0];
        #pragma unroll
        for (int rr = 0; rr < NR0; rr++) acc[rr] = 0.0f;

        for (int u = lane_id; u < nch * NR0; u += WARP) {
            int ic = u % nch, rr = u / nch;
            const __half* wr = weights + (size_t)(r0 + rr) * id + ic * CHK;
            const float* yc = y + ic * CHK;
            int len = min(CHK, id - ic * CHK);
            float p = 0.0f;
            // the unit's lane streams the WHOLE chunk (8 halves per pass)
            for (int i = 0; i < len; i += 8) {
                const __half2* wp = reinterpret_cast<const __half2*>(wr + i);
                float2 f0 = __half22float2(wp[0]);
                float2 f1 = __half22float2(wp[1]);
                float2 f2 = __half22float2(wp[2]);
                float2 f3 = __half22float2(wp[3]);
                float4 b0 = *reinterpret_cast<const float4*>(yc + i);
                float4 b1 = *reinterpret_cast<const float4*>(yc + i + 4);
                p += f0.x * b0.x + f0.y * b0.y + f1.x * b0.z + f1.y * b0.w
                   + f2.x * b1.x + f2.y * b1.y + f3.x * b1.z + f3.y * b1.w;
            }
            acc[rr] += p;
        }

        #pragma unroll
        for (int rr = 0; rr < NR0; rr++) {
            float v = warp_reduce_sum(acc[rr]);
            if (lane_id == 0 && r0 + rr < od) output[(size_t)t * od + r0 + rr] = v;
        }
    }
}

__global__ void f16_f32_matmul_scalar(
    const __half* __restrict__ weights,
    const float* __restrict__ acts,
    float* __restrict__ output,
    int od, int id, int nt
) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)nt * od) return;
    int t = (int)(idx / od), r = (int)(idx % od);
    const __half* wr = weights + (size_t)r * id;
    const float* y = acts + (size_t)t * id;
    float acc = 0.0f;
    for (int i = 0; i < id; i++) acc += __half2float(wr[i]) * y[i];
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


// D3-5 1a: decode fused rms_norm + pad40 q8 epilogue. The rms body is
// rms_norm_f32 verbatim (bit-identical f32 y); the epilogue re-reads the row
// this block just wrote (L1-hot after __syncthreads) and runs the standalone
// per-block quantize body verbatim. Same launch geometry as rms_norm_f32
// (grid n, WARP threads).
__global__ void __launch_bounds__(128) rms_norm_quant_pad40(
    const float* __restrict__ x,
    const float* __restrict__ w,
    float* __restrict__ y,
    uint8_t* __restrict__ q8,
    int d, float eps, int n
) {
    int row = blockIdx.x;
    if (row >= n) return;

    int tid = threadIdx.x;
    int d4 = d / 4;

    // D3-7 2c: wide-block geometry (launch picks 32 or 128 threads). The
    // reduction is bitwise-preserved: lanes 0..31 keep the exact 32-thread
    // form's element->lane mapping, serial per-lane accumulation order and
    // warp_reduce_sum tree; the unroll only deepens load pipelining. scale
    // reaches the whole block through shared memory. The write and quantize
    // loops are per-element / per-32-block independent, so their wider
    // thread mapping cannot change any output bit.
    __shared__ float s_scale;
    float scale;
    if (tid < WARP) {
        const float4* x4 = reinterpret_cast<const float4*>(x + row * d);

        float ss = 0.0f;
        #pragma unroll 8
        for (int i = tid; i < d4; i += WARP) {
            float4 v = x4[i];
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = warp_reduce_sum(ss);

        scale = rsqrtf(ss / (float)d + eps);
        if (tid == 0) s_scale = scale;
    }
    __syncthreads();
    scale = s_scale;

    float4* y4 = reinterpret_cast<float4*>(y + row * d);
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4w = reinterpret_cast<const float4*>(x + row * d);
    for (int i = tid; i < d4; i += blockDim.x) {
        float4 wv = w4[i];
        float4 xv = x4w[i];
        y4[i].x = xv.x * scale * wv.x;
        y4[i].y = xv.y * scale * wv.y;
        y4[i].z = xv.z * scale * wv.z;
        y4[i].w = xv.w * scale * wv.w;
    }

    // epilogue: whole block arrives (all threads share `row`), then each
    // thread quantizes blocks tid, tid+blockDim.x, ... of its own row.
    __syncthreads();
    int nb = d / 32;
    const float* src = y + (size_t)row * d;
    uint8_t* dst = q8 + (size_t)row * nb * Q8PB;
    for (int b = tid; b < nb; b += blockDim.x)
        quantize_pad40_block(src + (size_t)b * 32, dst + (size_t)b * Q8PB);
}

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


// D3-5 1a: decode fused swiglu + pad40 q8 epilogue. Body = swiglu_f32_off
// verbatim (guarded, no early return — every thread reaches the barrier).
// Block bx wrote output elements [bx*256, bx*256+256) = quant blocks
// bx*8 .. bx*8+7, so 8 threads per block re-read them (L1-hot) and quantize;
// across the grid this is the same thread-count as the standalone kernel
// (one thread per 32-block). REQUIRES the 256-thread launch geometry.
__global__ void swiglu_quant_pad40(
    float* __restrict__ buf,
    uint8_t* __restrict__ q8,
    int n, int off
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float g = buf[tid];
        buf[tid] = (g / (1.0f + expf(-g))) * buf[off + tid];
    }
    __syncthreads();
    int b = blockIdx.x * 8 + (int)threadIdx.x;
    if (threadIdx.x < 8 && b < (n >> 5))
        quantize_pad40_block(buf + (size_t)b * 32, q8 + (size_t)b * Q8PB);
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

// Quantize ONE block (32 f32 values) into one packed Q8_0 block at `cell`.
// This is the single device-side statement of the C4 S2b quantizer — `d =
// amax/127` stored as an f16 with round-to-nearest-even and every quant
// `rintf(x/d)` clamped to the i8 range — shared by the KV store and, since #144
// item 1, the packed fused decode epilogue. `x` may live in registers or in
// global memory; the loop reads each element once.
__device__ __forceinline__ void q8_0_quantize_block(const float* x, unsigned char* cell) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) amax = fmaxf(amax, fabsf(x[i]));
    const float d = amax / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    *reinterpret_cast<__half*>(cell) = __float2half_rn(d);
    signed char* q = reinterpret_cast<signed char*>(cell + 2);
    #pragma unroll
    for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
        float v = rintf(x[i] * id);
        v = fminf(127.0f, fmaxf(-128.0f, v));
        q[i] = (signed char)(int)v;
    }
}

// C4 S2b: quantize nt f32 rows into packed Q8_0 cells. One thread per
// (row, 32-element block); `row_bytes` is the packed cell's byte width
// (`KvFormat::Q8_0.row_bytes(nkt)` = ceil(nkt/32*34 / 4) * 4 words).
//
// The quantizer is the CPU's, step for step (`quants::quantize_row_q8_0_into`,
// whose aarch64 path is the scalar loop): `d = amax/127` stored as an f16 with
// round-to-nearest-even, and each quant `rintf(x/d)` clamped to the i8 range.
// `rintf` is round-to-nearest-even under the default rounding mode, which is the
// `round_ties_even` the CPU uses — so both backends write the same bytes for the
// same row, and a CPU/device Q8_0 comparison is a layout check, not a tolerance.
__global__ void store_kv_q8_0(
    const float* __restrict__ src,
    unsigned char* __restrict__ dst,
    int nkt, int nt, size_t row_bytes,
    const int* positions
) {
    const int t = blockIdx.x;
    const int nblk = nkt / Q8_0_BLOCK_ELEMS;
    const int blk = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= nt || blk >= nblk) return;
    const int p = positions[t];
    const float* x = src + (size_t)t * nkt + (size_t)blk * Q8_0_BLOCK_ELEMS;
    q8_0_quantize_block(x, dst + (size_t)p * row_bytes + (size_t)blk * Q8_0_BLOCK_BYTES);
}

// ─── Fused decode QKV epilogue: bias-add + RoPE + KV-store (nt==1) ───
// D3-8: CUDA port of Metal's kernel_attn_bias_rope_store (G4 FusedQKV). One
// kernel replaces the 7-launch unfused chain (add_bias×3 + rope×2 +
// store_kv×2). q/k/v are POINTER-FORM section bases so one kernel serves
// both decode-QKV layer classes: the concat class (wq|wk|wv same ttype)
// points them INTO the concat matmul output (q=base, k=base+nqt,
// v=base+2*nkt) and the mixed-quant class (e.g. Q6_K attn_v, no concat
// matmul) points them at the three separate matmul outputs. Applies the
// per-section bias, RoPEs q and k IN PLACE, and stores k/v into the
// persistent KV regions. The rope math is VERBATIM rope_f32 (NEOX pairing
// (j, j+half), same freq/theta expression and cosf/sinf — bitwise-identical
// per-element results), the bias add is verbatim add_bias_f32 (same two
// operands, one add), and the store addresses/conversions are verbatim
// store_kv_f32 / store_kv_f16 (dst[pos * nkt + j]; __float2half is RN, the
// same conversion the unfused f16 store's scalar tail uses) — bit-identical
// outputs, fewer launches. Thread mapping (Metal's): one thread per
// (head, d < hd/2) rope pair for q and k, one thread per v element →
// grid = nqt/2 + nkt/2 + nkt. positions[0] is read device-side (nt==1; no
// host scalar crosses the launch — CUDA Graph capture/replay safe).
__global__ void attn_bias_rope_store_f32(
    float* __restrict__ q,
    float* __restrict__ k,
    float* __restrict__ v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    float* __restrict__ kv_k,
    float* __restrict__ kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions,
    const int* cells,
    int kv_is_f16
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    // C6: `positions[0]` is the token's index within its sequence (what RoPE
    // rotates by); `cells[0]` is the allocator-resolved KV row. They differ
    // whenever the run does not start at cell 0 (multi-slot server), so the
    // store below addresses rows by `row`, never by `pos`.
    const int pos = positions[0];
    const int row = cells[0];

    if (u < qpairs) {
        // q section: bias + rope in place (attention reads q at offset 0)
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int j  = base + d;
        const int j2 = j + half_dim;
        float x0 = q[j]  + bias_q[j];
        float x1 = q[j2] + bias_q[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        q[j]  = x0 * cs - x1 * sn;
        q[j2] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        // k section: bias + rope in place + store into the K region
        const int u2   = u - qpairs;
        const int head = u2 / half_dim;
        const int d    = u2 % half_dim;
        const int base = head * hd;
        const int j  = base + d;
        const int j2 = j + half_dim;
        float x0 = k[j]  + bias_k[j];
        float x1 = k[j2] + bias_k[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        const float r0 = x0 * cs - x1 * sn;
        const float r1 = x0 * sn + x1 * cs;
        k[j]  = r0;
        k[j2] = r1;
        if (kv_is_f16) {
            ((__half*)kv_k)[(size_t)row * nkt + j]  = __float2half(r0);
            ((__half*)kv_k)[(size_t)row * nkt + j2] = __float2half(r1);
        } else {
            kv_k[(size_t)row * nkt + j]  = r0;
            kv_k[(size_t)row * nkt + j2] = r1;
        }
    } else {
        // v section: bias + store into the V region
        const int j = u - qpairs - kpairs;
        const float val = v[j] + bias_v[j];
        v[j] = val;
        if (kv_is_f16) {
            ((__half*)kv_v)[(size_t)row * nkt + j] = __float2half(val);
        } else {
            kv_v[(size_t)row * nkt + j] = val;
        }
    }
}

// ─── #144 item 1: the PACKED arm of the fused decode epilogue ────────────────
// The f32/f16 kernel above writes one K/V element per thread, which a packed
// cell cannot accept: a Q8_0 block's scale needs all 32 of its elements before
// any of them can be quantized. This arm keeps the same q section (bias + rope
// in place, one thread per pair) and re-maps the K and V sections to one thread
// per (head, 32-element block): the thread computes the block's 32 values
// itself and hands them to `q8_0_quantize_block`, the store's own quantizer, so
// the bytes it writes are the unfused chain's (`add_bias`+`rope`+`store_kv_q8_0`)
// verbatim.
//
// K's 32 roped values are computed from the *unroped* k row plus the rope pair
// partner (element d <-> d + hd/2 within the head, the neox pairing
// `attn_bias_rope_store_f32` uses). Neither K nor V is written back: both fused
// classes leave those buffers dead (attention reads the packed region) and a
// block-owning thread cannot write `k` in place without racing the thread that
// reads its pair partner. The observable output — the packed region's bytes — is
// the unfused chain's.
__global__ void attn_bias_rope_store_q8_0(
    float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ bias_q,
    const float* __restrict__ bias_k,
    const float* __restrict__ bias_v,
    unsigned char* __restrict__ kv_k,
    unsigned char* __restrict__ kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions,
    const int* cells,
    size_t row_bytes
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kblks = nkt / Q8_0_BLOCK_ELEMS;
    const int total = qpairs + 2 * kblks;
    const int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= total) return;
    // C6: `positions[0]` is the rope angle's sequence-relative index and
    // `cells[0]` the allocator-resolved row the packed cell is written at.
    const int pos = positions[0];
    const int row = cells[0];

    if (u < qpairs) {
        // q section: bias + rope in place (verbatim attn_bias_rope_store_f32)
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int j  = base + d;
        const int j2 = j + half_dim;
        float x0 = q[j]  + bias_q[j];
        float x1 = q[j2] + bias_q[j2];
        float freq = freq_scale / powf(freq_base, (2.0f * d) / hd);
        float theta = pos * freq;
        float cs = cosf(theta), sn = sinf(theta);
        q[j]  = x0 * cs - x1 * sn;
        q[j2] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kblks) {
        // K section: one (head, block). `b` is also the block's index inside the
        // packed row (blocks are laid out in flat element order).
        const int b = u - qpairs;
        const int blk = b % (hd / Q8_0_BLOCK_ELEMS);
        const int head = b / (hd / Q8_0_BLOCK_ELEMS);
        float x[Q8_0_BLOCK_ELEMS];
        #pragma unroll
        for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) {
            // `d` is the element's index inside the head, so the block's own
            // offset must be added — without it every block of a head would
            // compute the head's first 32 values (caught by
            // cuda_q8_0_fused_epilogue_matches_the_cpu_quantizer's byte arm).
            const int d = blk * Q8_0_BLOCK_ELEMS + i;
            const int dd = (d < half_dim) ? d : d - half_dim;
            const int ja = head * hd + dd;
            const int jb = ja + half_dim;
            float x0 = k[ja] + bias_k[ja];
            float x1 = k[jb] + bias_k[jb];
            float freq = freq_scale / powf(freq_base, (2.0f * dd) / hd);
            float theta = pos * freq;
            float cs = cosf(theta), sn = sinf(theta);
            x[i] = (d < half_dim) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
        }
        q8_0_quantize_block(x, kv_k + (size_t)row * row_bytes + (size_t)b * Q8_0_BLOCK_BYTES);
    } else {
        // V section: one block, bias + quantize. The bias is folded into the
        // quantized value only: like the K section, the V buffer itself is dead
        // in both fused classes (attention reads the packed region), and leaving
        // it unwritten lets `v` stay `const`.
        const int b = u - qpairs - kblks;
        const int base = b * Q8_0_BLOCK_ELEMS;
        float x[Q8_0_BLOCK_ELEMS];
        #pragma unroll
        for (int i = 0; i < Q8_0_BLOCK_ELEMS; i++) x[i] = v[base + i] + bias_v[base + i];
        q8_0_quantize_block(x, kv_v + (size_t)row * row_bytes + (size_t)b * Q8_0_BLOCK_BYTES);
    }
}
extern "C" {

void launch_q6_k_f32_matmul_padded(
    const uint8_t* weights, const float* acts, float* output,
    int od, int id, int nt, cudaStream_t stream
) {
    const int NR0 = 2, NSG = 2;
    dim3 block(64, 1, 1);
    dim3 grid((od + NR0 * NSG - 1) / (NR0 * NSG), nt, 1);
    minfer_launch_prelude("launch:q6_k_f32_matmul_padded", "q6_k_f32_matmul_padded");
    q6_k_f32_matmul_padded<<<grid, minfer_launch_block("launch:q6_k_f32_matmul_padded", block), 0, stream>>>(weights, acts, output, od, id, nt);
    minfer_launch_ok("launch:q6_k_f32_matmul_padded", "q6_k_f32_matmul_padded");
}

void launch_swiglu_f32_off(
    float* buf, int n, int off, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    minfer_launch_prelude("launch:swiglu_f32_off", "swiglu_f32_off");
    swiglu_f32_off<<<grid, minfer_launch_block("launch:swiglu_f32_off", block), 0, stream>>>(buf, n, off);
    minfer_launch_ok("launch:swiglu_f32_off", "swiglu_f32_off");
}


// D3-5 1a: decode fused swiglu + pad40 q8 epilogue. The 256-thread block size
// is part of the epilogue's block->quant-block mapping (8 blocks per 256
// elements) — do not change it without changing the kernel.
void launch_swiglu_quant_pad40(
    float* buf, uint8_t* q8, int n, int off, cudaStream_t stream
) {
    int block = 256;
    int grid = (n + block - 1) / block;
    minfer_launch_prelude("launch:swiglu_quant_pad40", "swiglu_quant_pad40");
    swiglu_quant_pad40<<<grid, minfer_launch_block("launch:swiglu_quant_pad40", block), 0, stream>>>(buf, q8, n, off);
    minfer_launch_ok("launch:swiglu_quant_pad40", "swiglu_quant_pad40");
}

void launch_gather_rows_f32(
    const float* src, const float* ids, float* out,
    int n, int nt, cudaStream_t stream
) {
    long long total = (long long)nt * n;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    minfer_launch_prelude("launch:gather_rows_f32", "gather_rows_f32");
    gather_rows_f32<<<(int)grid, minfer_launch_block("launch:gather_rows_f32", block), 0, stream>>>(src, ids, out, n, nt);
    minfer_launch_ok("launch:gather_rows_f32", "gather_rows_f32");
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

// Q5_0: one thread per 32-element block (22B = f16 d + u32 qh + 16 qs bytes);
// value = d * (nibble + 16*high_bit - 16) — matches quants.rs dot_q5_0_q8_0
// and kernel.rs's CPU embed path.
__global__ void embed_rows_q5_0(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    const int BS = 22;
    int nb = n_embd / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nb) return;
    int t = tid / nb, b = tid % nb;
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    const uint8_t* blk = w + ((long long)id * nb + b) * BS;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    // qh sits at block offset 2 — NOT 4-byte aligned for even block indices
    // (22-byte stride). Two 2-byte-aligned u16 loads instead of one misaligned
    // u32: misaligned u32 loads fault nondeterministically on GB10 unified
    // memory (err 716) depending on page-mapping state.
    uint32_t qh = (uint32_t)*reinterpret_cast<const uint16_t*>(blk + 2)
                | ((uint32_t)*reinterpret_cast<const uint16_t*>(blk + 4) << 16);
    const uint8_t* qs = blk + 6;
    float* o = out + (long long)t * n_embd + b * 32;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        float v_lo = float(qs[j] & 0x0F) + 16.0f * float((qh >> j) & 1) - 16.0f;
        float v_hi = float(qs[j] >> 4) + 16.0f * float((qh >> (j + 16)) & 1) - 16.0f;
        o[j] = d * v_lo;
        o[j + 16] = d * v_hi;
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

// Q4_1: one thread per 32-element block (20B = f16 d + f16 m + 16 qs
// bytes); value = d * nibble + m (unsigned nibbles, no centering) —
// matches quants.rs dot_q4_1_q8_0 and kernel.rs's CPU embed path. No
// high-bit plane; the 20B stride keeps blk+4 4-byte-aligned for every
// row, so the nibble bytes are read as four ALIGNED u32s (GB10
// misaligned-load rule, see the q5_0 note).
__global__ void embed_rows_q4_1(
    const uint8_t* __restrict__ w,
    const float* __restrict__ ids,
    float* __restrict__ out,
    int n_embd, int nt
) {
    const int BS = 20;
    int nb = n_embd / 32;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nt * nb) return;
    int t = tid / nb, b = tid % nb;
    int id = __float_as_int(ids[t]); // I32-as-f32 bit pattern (graph rule §4)
    const uint8_t* blk = w + ((long long)id * nb + b) * BS;
    float d = h2f(*reinterpret_cast<const uint16_t*>(blk));
    float m = h2f(*reinterpret_cast<const uint16_t*>(blk + 2));
    const uint32_t* qs = reinterpret_cast<const uint32_t*>(blk + 4);
    float* o = out + (long long)t * n_embd + b * 32;
    #pragma unroll
    for (int v = 0; v < 4; v++) {
        const uint32_t q = qs[v];
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            o[v * 4 + k] = d * float((q >> (8 * k)) & 0x0F) + m;
            o[v * 4 + k + 16] = d * float((q >> (8 * k + 4)) & 0x0F) + m;
        }
    }
}

void launch_embed_rows(
    const uint8_t* w, const float* ids, float* out,
    int n_embd, int nt, int type_id, int block_stride, cudaStream_t stream
) {
    int block = 256;
    long long total = 0;
    if (type_id == 0 || type_id == 1 || type_id == 7) { // q8_0 / q4_0 / q4_1: one thread per 32-group
        total = (long long)nt * (n_embd / 32);
    } else if (type_id == 2) {          // q4_K: one per 32-element sub-block
        total = (long long)nt * (n_embd / 256) * 8;
    } else if (type_id == 4) {          // q5_1: one per 32-element block
        total = (long long)nt * (n_embd / 32);
    } else if (type_id == 6) {          // q5_0: one per 32-element block
        total = (long long)nt * (n_embd / 32);
    } else if (type_id == 5) {          // q5_K: one per 32-element sub-block
        total = (long long)nt * ((n_embd + 31) / 32);
    } else {                            // q6_K: one per 16-element sub-block
        total = (long long)nt * (n_embd / 256) * 16;
    }
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    switch (type_id) {
        case 0:
            minfer_launch_prelude("launch:embed_rows__q8_0", "embed_rows_q8_0");
            embed_rows_q8_0<<<(int)grid, minfer_launch_block("launch:embed_rows__q8_0", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q8_0", "embed_rows_q8_0");
            break;
        case 1:
            minfer_launch_prelude("launch:embed_rows__q4_0", "embed_rows_q4_0");
            embed_rows_q4_0<<<(int)grid, minfer_launch_block("launch:embed_rows__q4_0", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q4_0", "embed_rows_q4_0");
            break;
        case 2:
            minfer_launch_prelude("launch:embed_rows__q4_k", "embed_rows_q4_k");
            embed_rows_q4_k<<<(int)grid, minfer_launch_block("launch:embed_rows__q4_k", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q4_k", "embed_rows_q4_k");
            break;
        case 7:
            minfer_launch_prelude("launch:embed_rows__q4_1", "embed_rows_q4_1");
            embed_rows_q4_1<<<(int)grid, minfer_launch_block("launch:embed_rows__q4_1", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q4_1", "embed_rows_q4_1");
            break;
        case 4:
            minfer_launch_prelude("launch:embed_rows__q5_1", "embed_rows_q5_1");
            embed_rows_q5_1<<<(int)grid, minfer_launch_block("launch:embed_rows__q5_1", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q5_1", "embed_rows_q5_1");
            break;
        case 5:
            minfer_launch_prelude("launch:embed_rows__q5_k", "embed_rows_q5_k");
            embed_rows_q5_k<<<(int)grid, minfer_launch_block("launch:embed_rows__q5_k", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q5_k", "embed_rows_q5_k");
            break;
        case 6:
            minfer_launch_prelude("launch:embed_rows__q5_0", "embed_rows_q5_0");
            embed_rows_q5_0<<<(int)grid, minfer_launch_block("launch:embed_rows__q5_0", block), 0, stream>>>(w, ids, out, n_embd, nt);
            minfer_launch_ok("launch:embed_rows__q5_0", "embed_rows_q5_0");
            break;
        default:
            minfer_launch_prelude("launch:embed_rows__q6_k", "embed_rows_q6_k");
            embed_rows_q6_k<<<(int)grid, minfer_launch_block("launch:embed_rows__q6_k", block), 0, stream>>>(w, ids, out, n_embd, nt, block_stride);
            minfer_launch_ok("launch:embed_rows__q6_k", "embed_rows_q6_k");
            break;
    }
}

// #141: f16 embedding gather — one thread per output element (no block math).
// Checked like `launch_f16_f32_matmul` (see the note there): 0 = success.
int launch_embed_rows_f16(
    const uint8_t* w, const float* ids, float* out,
    int n_embd, int nt, cudaStream_t stream
) {
    const char* site = "launch:embed_rows_f16";
    const char* kn = "embed_rows_f16";
    minfer_launch_prelude(site, kn);
    int block = 256;
    long long total = (long long)nt * n_embd;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    embed_rows_f16<<<(int)grid, minfer_launch_block(site, block), 0, stream>>>(w, ids, out, n_embd, nt);
    return minfer_launch_ok(site, kn) ? 0 : 1;
}

void launch_f32_f32_matmul(
    const float* w, const float* x, float* out,
    int od, int id, int nt, cudaStream_t stream
) {
    if (id % 8 == 0) {
        dim3 grid((od + 7) / 8, 1), block(64);
        minfer_launch_prelude("launch:f32_f32_matmul__vec", "f32_f32_matmul_vec");
        f32_f32_matmul_vec<<<grid, minfer_launch_block("launch:f32_f32_matmul__vec", block), 0, stream>>>(w, x, out, od, id, nt);
        minfer_launch_ok("launch:f32_f32_matmul__vec", "f32_f32_matmul_vec");
    } else {
        long long total = (long long)nt * od;
        int block = 256;
        long long grid = (total + block - 1) / block;
        if (grid > 2147483647LL) grid = 2147483647LL;
        minfer_launch_prelude("launch:f32_f32_matmul__scalar", "f32_f32_matmul_scalar");
        f32_f32_matmul_scalar<<<(int)grid, minfer_launch_block("launch:f32_f32_matmul__scalar", block), 0, stream>>>(w, x, out, od, id, nt);
        minfer_launch_ok("launch:f32_f32_matmul__scalar", "f32_f32_matmul_scalar");
    }
}

// #141: f16 weights × f32 activations. Same shape split as the f32 launcher —
// the vector kernel needs id % 8 == 0 (aligned half2 / float4 loads), the
// scalar kernel covers every id.
//
// Returns 0 on success, non-zero when the launch itself failed. #141 is a new
// kernel, so it reads its own `cudaGetLastError` at the site through the #147
// helpers instead of joining the 65 unchecked launchers recorded in #162: the
// failure is named where it is made and the Rust caller turns it into an `Err`
// (never a silent fallback).

int launch_f16_f32_matmul(
    const uint8_t* w, const float* x, float* out,
    int od, int id, int nt, cudaStream_t stream
) {
    const __half* wh = reinterpret_cast<const __half*>(w);
    if (id % 8 == 0) {
        const char* site = "launch:f16_f32_matmul_vec";
        const char* kn = "f16_f32_matmul_vec";
        minfer_launch_prelude(site, kn);
        dim3 grid((od + 7) / 8, 1), block(64);
        f16_f32_matmul_vec<<<grid, minfer_launch_block(site, block), 0, stream>>>(wh, x, out, od, id, nt);
        return minfer_launch_ok(site, kn) ? 0 : 1;
    }
    const char* site = "launch:f16_f32_matmul_scalar";
    const char* kn = "f16_f32_matmul_scalar";
    minfer_launch_prelude(site, kn);
    long long total = (long long)nt * od;
    int block = 256;
    long long grid = (total + block - 1) / block;
    if (grid > 2147483647LL) grid = 2147483647LL;
    f16_f32_matmul_scalar<<<(int)grid, minfer_launch_block(site, block), 0, stream>>>(wh, x, out, od, id, nt);
    return minfer_launch_ok(site, kn) ? 0 : 1;
}

void launch_quantize_q8_0(
    const float* x, uint8_t* y, int dim, int nt, cudaStream_t stream
) {
    int nb = dim / 32;
    int total = nt * nb;
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((total + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:quantize_q8_0", "quantize_q8_0");
    quantize_q8_0<<<grid, minfer_launch_block("launch:quantize_q8_0", block), 0, stream>>>(x, y, dim, nt);
    minfer_launch_ok("launch:quantize_q8_0", "quantize_q8_0");
}

void launch_rms_norm_f32(
    const float* x, const float* w, float* y,
    int d, float eps, int n, cudaStream_t stream
) {
    dim3 block(WARP, 1, 1);
    dim3 grid(n, 1, 1);
    minfer_launch_prelude("launch:rms_norm_f32", "rms_norm_f32");
    rms_norm_f32<<<grid, minfer_launch_block("launch:rms_norm_f32", block), 0, stream>>>(x, w, y, d, eps, n);
    minfer_launch_ok("launch:rms_norm_f32", "rms_norm_f32");
}


// D3-5 1a: decode fused rms_norm + pad40 q8 epilogue (n==1 decode producers).
void launch_rms_norm_quant_pad40(
    const float* x, const float* w, float* y, uint8_t* q8,
    int d, float eps, int n, cudaStream_t stream
) {
    // D3-7 2c: 128-thread wide block (was WARP). The body is
    // blockDim.x-relative and bitwise-identical at either geometry; 128
    // threads cut the per-row write/quantize latency chains 4x (census:
    // 9.4 -> target ~4 us at hidden 5120, 94.6 launches/decode-step).
    minfer_launch_prelude("launch:rms_norm_quant_pad40", "rms_norm_quant_pad40");
    rms_norm_quant_pad40<<<n, minfer_launch_block("launch:rms_norm_quant_pad40", 128), 0, stream>>>(x, w, y, q8, d, eps, n);
    minfer_launch_ok("launch:rms_norm_quant_pad40", "rms_norm_quant_pad40");
}

void launch_add_bias_f32(
    float* y, const float* b, int d, int n, cudaStream_t stream
) {
    dim3 block(64, 1, 1); // 64 threads in x, grid y handles dim remainder
    dim3 grid(n, (d + 63) / 64, 1);
    minfer_launch_prelude("launch:add_bias_f32", "add_bias_f32");
    add_bias_f32<<<grid, minfer_launch_block("launch:add_bias_f32", block), 0, stream>>>(y, b, d);
    minfer_launch_ok("launch:add_bias_f32", "add_bias_f32");
}

void launch_add_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:add_f32", "add_f32");
    add_f32<<<grid, minfer_launch_block("launch:add_f32", block), 0, stream>>>(x, y, z, n);
    minfer_launch_ok("launch:add_f32", "add_f32");
}

void launch_mul_f32(
    const float* x, const float* y, float* z, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:mul_f32", "mul_f32");
    mul_f32<<<grid, minfer_launch_block("launch:mul_f32", block), 0, stream>>>(x, y, z, n);
    minfer_launch_ok("launch:mul_f32", "mul_f32");
}

void launch_silu_f32(float* y, int n, cudaStream_t stream) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:silu_f32", "silu_f32");
    silu_f32<<<grid, minfer_launch_block("launch:silu_f32", block), 0, stream>>>(y, n);
    minfer_launch_ok("launch:silu_f32", "silu_f32");
}

void launch_swiglu_f32(
    const float* gate, const float* up, float* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:swiglu_f32", "swiglu_f32");
    swiglu_f32<<<grid, minfer_launch_block("launch:swiglu_f32", block), 0, stream>>>(gate, up, dst, n);
    minfer_launch_ok("launch:swiglu_f32", "swiglu_f32");
}

void launch_f32_bits_to_i32(
    const float* src, int* dst, int n, cudaStream_t stream
) {
    int block_sz = 256;
    dim3 block(block_sz, 1, 1);
    dim3 grid((n + block_sz - 1) / block_sz, 1, 1);
    minfer_launch_prelude("launch:f32_bits_to_i32", "f32_bits_to_i32");
    f32_bits_to_i32<<<grid, minfer_launch_block("launch:f32_bits_to_i32", block), 0, stream>>>(src, dst, n);
    minfer_launch_ok("launch:f32_bits_to_i32", "f32_bits_to_i32");
}

void launch_rope_f32(
    float* x, int n_head, int n_dims, int nt,
    float freq_base, float freq_scale,
    const int* positions, cudaStream_t stream
) {
    int block_sz = 64; // threads per head dimension
    dim3 block(block_sz, 1, 1);
    dim3 grid(nt, n_head, 1);
    minfer_launch_prelude("launch:rope_f32", "rope_f32");
    rope_f32<<<grid, minfer_launch_block("launch:rope_f32", block), 0, stream>>>(x, n_head, n_dims, nt, freq_base, freq_scale, positions);
    minfer_launch_ok("launch:rope_f32", "rope_f32");
}

void launch_store_kv_f32(
    const float* src, float* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 grid(nt, nkt, 1);
    minfer_launch_prelude("launch:store_kv_f32", "store_kv_f32");
    store_kv_f32<<<grid, minfer_launch_block("launch:store_kv_f32", dim3(1, 1, 1)), 0, stream>>>(src, dst, nkt, nt, positions);
    minfer_launch_ok("launch:store_kv_f32", "store_kv_f32");
}

void launch_store_kv_f16(
    const float* src, void* dst, int nkt, int nt,
    const int* positions, cudaStream_t stream
) {
    dim3 block(128, 1, 1);
    dim3 grid(nt, (nkt / 4 + 127) / 128, 1);
    minfer_launch_prelude("launch:store_kv_f16", "store_kv_f16");
    store_kv_f16<<<grid, minfer_launch_block("launch:store_kv_f16", block), 0, stream>>>(src, (__half*)dst, nkt, nt, positions);
    minfer_launch_ok("launch:store_kv_f16", "store_kv_f16");
}

// C4 S2b: the packed store. One thread per (row, 32-element block); `row_bytes`
// is the packed cell's byte width, which the host takes from
// `KvFormat::Q8_0.row_bytes(nkt)`. `nkt` must be a multiple of 32 — `ensure_kv`'s
// `check_width` is what refuses anything else, so the grid arithmetic is exact.
void launch_store_kv_q8_0(
    const float* src, void* dst, int nkt, int nt, size_t row_bytes,
    const int* positions, cudaStream_t stream
) {
    const int nblk = nkt / Q8_0_BLOCK_ELEMS;
    dim3 block(64, 1, 1);
    dim3 grid(nt, (nblk + 63) / 64, 1);
    minfer_launch_prelude("launch:store_kv_q8_0", "store_kv_q8_0");
    store_kv_q8_0<<<grid, minfer_launch_block("launch:store_kv_q8_0", block), 0, stream>>>(
        src, (unsigned char*)dst, nkt, nt, row_bytes, positions);
    minfer_launch_ok("launch:store_kv_q8_0", "store_kv_q8_0");
}

// D3-8: fused decode QKV epilogue launcher — 256-thread blocks over the
// flat (nqt/2 + nkt/2 + nkt) thread mapping (Metal's dispatch_1d shape).
// C4 S2b: F32/F16 only. A packed cache never builds the fused epilogue (the
// builders' `layer_gpu` gate gains `&& !packed`), and the host wrapper refuses
// `layout == Q8_0` loudly rather than let the per-element store address a packed
// cell it has no whole block to quantize.
void launch_attn_bias_rope_store(
    float* q, float* k, float* v,
    const void* bias_q, const void* bias_k, const void* bias_v,
    void* kv_k, void* kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions, const int* cells, int kv_is_f16,
    cudaStream_t stream
) {
    const int total = nqt / 2 + nkt / 2 + nkt;
    const int block = 256;
    const int grid = (total + block - 1) / block;
    minfer_launch_prelude("launch:attn_bias_rope_store", "attn_bias_rope_store_f32");
    attn_bias_rope_store_f32<<<grid, minfer_launch_block("launch:attn_bias_rope_store", block), 0, stream>>>(
        q, k, v,
        (const float*)bias_q, (const float*)bias_k, (const float*)bias_v,
        (float*)kv_k, (float*)kv_v,
        nqt, nkt, hd, freq_base, freq_scale, positions, cells, kv_is_f16);
    minfer_launch_ok("launch:attn_bias_rope_store", "attn_bias_rope_store_f32");
}

// #144 item 1: the packed arm's launcher. Same 256-thread blocks over the
// (nqt/2 qpairs + nkt/32 K blocks + nkt/32 V blocks) thread mapping; `row_bytes`
// is the packed cell's byte width, the same number `store_kv_q8_0` is given.
void launch_attn_bias_rope_store_q8_0(
    float* q, const float* k, const float* v,
    const void* bias_q, const void* bias_k, const void* bias_v,
    void* kv_k, void* kv_v,
    int nqt, int nkt, int hd,
    float freq_base, float freq_scale,
    const int* positions, const int* cells, size_t row_bytes,
    cudaStream_t stream
) {
    const int total = nqt / 2 + 2 * (nkt / Q8_0_BLOCK_ELEMS);
    const int block = 256;
    const int grid = (total + block - 1) / block;
    minfer_launch_prelude("launch:attn_bias_rope_store_q8_0", "attn_bias_rope_store_q8_0");
    attn_bias_rope_store_q8_0<<<grid, minfer_launch_block("launch:attn_bias_rope_store_q8_0", block), 0, stream>>>(
        q, k, v,
        (const float*)bias_q, (const float*)bias_k, (const float*)bias_v,
        (unsigned char*)kv_k, (unsigned char*)kv_v,
        nqt, nkt, hd, freq_base, freq_scale, positions, cells, row_bytes);
    minfer_launch_ok("launch:attn_bias_rope_store_q8_0", "attn_bias_rope_store_q8_0");
}
} // extern "C" (C8b S4: fa_stage_kv_async became a template, which cannot have
  // C linkage — the launchers above keep theirs; the FA kernel and its launcher
  // below open a block of their own)
extern "C" {

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

// ─── prefill-GEMM dynamic shared memory: one formula, checked opt-ins ────
//
// The dynamic-smem requirement of one `gemm_f16_nt_kernel_t` instantiation is
// the kernel's own byte layout (see its definition): the As tile (2*TN*KS
// halves), the AF32 f32 mirror Am (2*TN*KS floats, AF32 only), the Bs tile
// (2*TM*KS halves) and the Cs store (NW*256 floats). TN = 64 and NW =
// blockDim.x/32 = 8 at every launch site (256 threads). The launcher
// (`launch_gemm_f16`) and the production opt-in (`gemm_smem_optin` below, called
// from the launcher's `GEMM_ONE`) MUST read the same number, so this formula is
// the single source. (Issue #145, history: the **eager** sweep removed by #218
// used a stale copy of this formula that assumed a 512-thread TM=256 launch and
// always added the AF32 mirror, so it asked for 131072 B for
// `gemm_f16_nt_kernel_t<256,64,true>` — more than the device's
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` (101376 B on GB10/sm_121). The
// rejected `cudaFuncSetAttribute` return value was never read, and the latched
// `cudaErrorInvalidValue` later surfaced as a phantom kernel-launch error.)
static size_t gemm_dynamic_smem_bytes(int tm, int ks, bool af32) {
    const size_t tn = 64, nw = 8;
    return (2 * tn * ks + 2 * (size_t)tm * ks) * 2
           + (af32 ? (size_t)2 * tn * ks * 4 : 0)
           + nw * 256 * 4;
}

// The launchable prefill-GEMM set, listed **once**. The fatbin lookup below, the
// production pre-warm entry (#223) and the #218 test seam all expand this list,
// so an instantiation that is compiled cannot be silently missed by a caller's
// own copy — the class of drift the #145/#218 records are about.
#define MINFER_GEMM_OPTIN_SET(X)                                               \
    X(64, 32, false) X(64, 32, true)                                           \
    X(64, 64, false) X(64, 64, true)                                           \
    X(128, 32, false) X(128, 32, true)                                         \
    X(128, 64, false) X(128, 64, true)                                         \
    X(256, 32, false) X(256, 32, true)                                         \
    X(256, 64, false) X(256, 64, true)

// The compiled `gemm_f16_nt_kernel_t<tm, ks, af32>` instantiation, or null for
// a combination that is not in the fatbin.
#define GEMM_FN_FOR(TM, KS, AF)                                                \
    if (tm == (TM) && ks == (KS) && af32 == (AF))                              \
        return reinterpret_cast<const void*>(&gemm_f16_nt_kernel_t<TM, KS, AF>);
static const void* gemm_f16_fn_for(int tm, int ks, bool af32) {
    MINFER_GEMM_OPTIN_SET(GEMM_FN_FOR)
    return nullptr;
}
#undef GEMM_FN_FOR

// #218: how many times a prefill-GEMM dynamic-smem opt-in was attempted while
// the launch stream was inside a capture window. The design invariant is that
// this **never** happens: the attribute is decided on the first (uncaptured)
// launch and cached per instantiation, so a capture-window launch re-reads the
// cached answer. `cuda_prefill_smem_optin_is_never_set_inside_a_capture_window`
// asserts the counter stays zero; `MINFER_TEST_CAPTURE_WARMUP=1` (the
// documented test seam) drives capture onto the first run and makes it fire.
static int g_gemm_smem_in_capture = 0;
extern "C" int gemm_smem_optin_in_capture_count() { return g_gemm_smem_in_capture; }

// The **production** opt-in for one prefill-GEMM instantiation, decided through
// the shared #147 helper. The launcher runs per layer per prefill, so an
// admitted or refused answer is recorded per instantiation and never re-asked
// on the hot path. Returns true when the >48 KiB dynamic smem is admitted and
// the launch may proceed.
//
// The template parameters are **the instantiation's own** (`TM`, `KS`, `AF32`),
// not a deduced function-pointer type: every `gemm_f16_nt_kernel_t` shares one
// signature, so a `template <typename K>` cache here would be a *per-signature*
// cache and the first admitted instantiation would answer for all the others
// (the #218 coverage gate found exactly that: `<64,64,true>` at 73728 B cached
// "admitted" for `<128,64,false>` at 57344 B, which was never set). `static
// state` must stay inside a per-`(TM,KS,AF32)` instantiation.
//
// #223 restored the **eager pre-warm**: `CudaState::try_new` drives this same
// production function once per process, for every launchable `(tm, ks, af32)`,
// from the earliest point in the process (context creation) where no stream can
// hold a capture window yet — so "the attribute is set outside any window" holds
// by construction, not by inference. This function is still the lazy per-launch
// opt-in too (defence in depth): it caches the admitted/refused answer per
// instantiation, so a launch whose pre-warm was skipped by the control env
// (`MINFER_NO_GEMM_PREWARM=1`) or by a regression still decides on its first
// uncaptured launch. The two remaining emergent mechanisms — capture opens in
// `cudaStreamCaptureModeThreadLocal` and only from the **third** run of a
// `(uid, range)` key (both recorded next to the capture trigger in
// `graph/cuda_backend.rs`) — are now defence in depth behind the pre-warm, not
// the primary guarantee. `stream` is used only to observe (and count) the
// never-case; the pre-warm passes 0 because no window can exist yet.
template <int TM, int KS, bool AF32>
static bool gemm_smem_optin(const char* site, const char* kernel_name, size_t bytes,
                            cudaStream_t stream) {
    static int state = 0;  // 0 = undecided, 1 = admitted, -1 = refused
    const bool injected = minfer_test_call_fails(site);
    // The 48 KiB default cap admits it — unless the test knob asks this site to
    // fail, because then the failure path is what is under test.
    if (bytes <= 48 * 1024 && !injected) return true;
    if (state != 0 && !injected) return state == 1;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (stream != 0 && cudaStreamIsCapturing(stream, &cs) == cudaSuccess
        && cs != cudaStreamCaptureStatusNone) {
        // The #218 gate asserts this stays zero: the attribute is set before the
        // window opens, never inside it.
        g_gemm_smem_in_capture++;
    } else {
        cudaGetLastError();  // a failed query must not latch
    }
    const bool ok = minfer_smem_optin(
        site, kernel_name,
        reinterpret_cast<const void*>(gemm_f16_nt_kernel_t<TM, KS, AF32>), (int)bytes);
    if (!injected) state = ok ? 1 : -1;  // a test answer is never cached
    return ok;
}

// Introspection for the Rust gates. `gemm_smem_need` is the single-source
// formula; `opted_in` is the device's own answer read back through
// `cudaFuncGetAttributes`; `limit` is the queried device opt-in limit.
extern "C" int gemm_prefill_smem_limit() { return minfer_optin_limit(); }
extern "C" size_t gemm_smem_need(int tm, int ks, int af32) {
    return gemm_dynamic_smem_bytes(tm, ks, af32 != 0);
}
extern "C" int gemm_smem_opted_in(int tm, int ks, int af32) {
    const void* fn = gemm_f16_fn_for(tm, ks, af32 != 0);
    if (fn == nullptr) return 0;
    cudaFuncAttributes a;
    if (cudaFuncGetAttributes(&a, fn) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return a.maxDynamicSharedSizeBytes
                   >= (int)gemm_dynamic_smem_bytes(tm, ks, af32 != 0)
               ? 1
               : 0;
}

// Drive the production per-instantiation opt-in for one compiled combination.
// Returns 1 when admitted, 0 when refused (named by `minfer_smem_optin`), -1
// when the combination is not in the fatbin. Both the pre-warm (#223) and the
// test seam below go through here, so they read the same dispatch and the same
// cache.
static int gemm_smem_optin_decision(int tm, int ks, bool a, size_t bytes,
                                    const char* site, cudaStream_t stream) {
#define GEMM_OPTIN_CASE(TM, KS, AF)                                            \
    if (tm == (TM) && ks == (KS) && a == (AF))                                 \
        return gemm_smem_optin<TM, KS, AF>(                                    \
                   site, "gemm_f16_nt_kernel_t<" #TM "," #KS "," #AF ">",      \
                   bytes, stream)                                              \
                   ? 1                                                         \
                   : 0;
    MINFER_GEMM_OPTIN_SET(GEMM_OPTIN_CASE)
#undef GEMM_OPTIN_CASE
    return -1;
}

// #223: the **production** eager pre-warm entry, called once per process from
// `CudaState::try_new` for every combination the Rust side enumerates. It drives
// `gemm_smem_optin`, so the attribute is set through the one production path and
// the per-instantiation cache the launcher later reads is already populated.
//
// Outcomes:  1 = the attribute is in force (set now, or already cached);
//            0 = refused — `minfer_smem_optin` already named the instantiation,
//                the requested bytes, the call, the device limit and
//                `cudaGetErrorName`, and cleared the latch;
//           -1 = the combination is not in the fatbin (the caller skips it);
//           -2 = **skipped deliberately**: the request exceeds the device's own
//                `cudaDevAttrMaxSharedMemoryPerBlockOptin`, so the attribute was
//                never called — the call could only return
//                `cudaErrorInvalidValue` and the instantiation cannot launch on
//                this device at all (`minfer_smem_optin` printed the reason).
// Not gated on `stream`: there is none at context creation, and the whole point
// of the placement is that no capture window can exist.
extern "C" int gemm_prefill_smem_prewarm_one(int tm, int ks, int af32) {
    const bool a = af32 != 0;
    const size_t bytes = gemm_dynamic_smem_bytes(tm, ks, a);
    const int decision = gemm_smem_optin_decision(tm, ks, a, bytes, "prewarm:gemm_f16", 0);
    if (decision != 0) return decision;  // 1 admitted, -1 not compiled
    // Refused. Re-derive *why* so the caller can report a deliberate skip
    // separately from a real attribute failure; the report itself (with the
    // formula, the limit and the error name) came from `minfer_smem_optin`.
    const int limit = minfer_optin_limit();
    return (limit > 0 && bytes > (size_t)limit) ? -2 : 0;
}

// #218 test seam: drive the **production** `gemm_smem_optin` (template, cache
// and all) for one compiled instantiation, so the #145-derived coverage gate
// reads the decision production makes rather than a mirror of it. Returns 1
// when admitted, 0 when refused/skipped, -1 when the combination is not in the
// fatbin. Test-only: production reaches the same function through
// `launch_gemm_f16`'s `GEMM_ONE` and through `gemm_prefill_smem_prewarm_one`.
extern "C" int gemm_prefill_smem_optin_one_for_test(int tm, int ks, int af32, void* stream) {
    const bool a = af32 != 0;
    const size_t bytes = gemm_dynamic_smem_bytes(tm, ks, a);
    return gemm_smem_optin_decision(tm, ks, a, bytes, "test:gemm_smem_optin",
                                    reinterpret_cast<cudaStream_t>(stream));
}

// Test injection (issue #145): latch a REAL `cudaErrorInvalidValue` by asking
// for more dynamic shared memory than the device admits — the exact pre-fix
// init request — and deliberately NOT clear it, so the Rust gate can prove
// `CudaState::sync` reports a latched error as latched (and clears it) instead
// of blaming the kernel that just ran. Never called by production code.
extern "C" int cuda_test_latch_oversized_smem() {
    int dev = 0, limit = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
    cudaError_t e = cudaFuncSetAttribute(
        reinterpret_cast<const void*>(&gemm_f16_nt_kernel_t<128, 32, false>),
        cudaFuncAttributeMaxDynamicSharedMemorySize, limit + 4096);
    return (int)e;  // left latched on purpose
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
        case 0:
            minfer_launch_prelude("launch:dequant_f16__q8_0", "dequant_q8_0_f16");
            dequant_q8_0_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q8_0", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q8_0", "dequant_q8_0_f16");
            break;
        case 1:
            minfer_launch_prelude("launch:dequant_f16__q4_0", "dequant_q4_0_f16");
            dequant_q4_0_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q4_0", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q4_0", "dequant_q4_0_f16");
            break;
        case 2:
            minfer_launch_prelude("launch:dequant_f16__q4_1", "dequant_q4_1_f16");
            dequant_q4_1_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q4_1", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q4_1", "dequant_q4_1_f16");
            break;
        case 3:
            minfer_launch_prelude("launch:dequant_f16__q5_0", "dequant_q5_0_f16");
            dequant_q5_0_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q5_0", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q5_0", "dequant_q5_0_f16");
            break;
        case 4:
            minfer_launch_prelude("launch:dequant_f16__q5_1", "dequant_q5_1_f16");
            dequant_q5_1_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q5_1", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q5_1", "dequant_q5_1_f16");
            break;
        case 5:
            minfer_launch_prelude("launch:dequant_f16__q4_k", "dequant_q4_k_f16");
            dequant_q4_k_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q4_k", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q4_k", "dequant_q4_k_f16");
            break;
        case 6:
            minfer_launch_prelude("launch:dequant_f16__q5_k", "dequant_q5_k_f16");
            dequant_q5_k_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q5_k", block), 0, stream>>>(w, out, od, id);
            minfer_launch_ok("launch:dequant_f16__q5_k", "dequant_q5_k_f16");
            break;
        default:
            minfer_launch_prelude("launch:dequant_f16__q6_k", "dequant_q6_k_f16");
            dequant_q6_k_f16<<<(int)grid, minfer_launch_block("launch:dequant_f16__q6_k", block), 0, stream>>>(w, out, od, id, block_stride);
            minfer_launch_ok("launch:dequant_f16__q6_k", "dequant_q6_k_f16");
            break;
    }
}

void launch_convert_f16(
    const float* x, __half* out, long long n, cudaStream_t stream
) {
    long long grid = (n / 8 + 255) / 256;
    if (grid > 2147483647LL) grid = 2147483647LL;
    minfer_launch_prelude("launch:convert_f16", "convert_f32_f16_kernel");
    convert_f32_f16_kernel<<<(int)grid, minfer_launch_block("launch:convert_f16", 256), 0, stream>>>(x, out, n);
    minfer_launch_ok("launch:convert_f16", "convert_f32_f16_kernel");
}

// #147: returns 1 when the launch was issued and accepted, 0 when the
// dynamic-smem opt-in failed or the launch itself returned an error — either
// way the call is named here and the caller (`prefill_gemm_f16_inner`) turns
// the 0 into an `Err`, never a silent launch into a checked error.
int launch_gemm_f16(
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
    // #145: the single-source formula (AF32-aware). The old inline copy dropped
    // the AF32 f32 mirror, so `launch_gemm_f32a` declared 16384 B less than the
    // kernel's own `Bs`/`Cs` offsets need at KS=32 — an out-of-declaration
    // shared-memory access that "worked" only because the block's smem happened
    // to be carved where nothing else wrote.
    const size_t dyn_smem = gemm_dynamic_smem_bytes(tm, ks, af32);
    // #147: one site token per (af32) variant: the opt-in and the launch are
    // separate checks with separate injections.
    const char* const attr_site = af32 ? "attr:gemm_f16_a32" : "attr:gemm_f16_f16";
    const char* const launch_site = af32 ? "launch:gemm_f16_a32" : "launch:gemm_f16_f16";
    int ok = 0;
#define GEMM_ONE(TM_, KS_, AF_)                                                        \
    do {                                                                               \
        const char* const nm = "gemm_f16_nt_kernel_t<" #TM_ "," #KS_ "," #AF_ ">";     \
        if (gemm_smem_optin<TM_, KS_, AF_>(attr_site, nm, dyn_smem, stream)) {        \
            minfer_launch_prelude(launch_site, nm);                                    \
            gemm_f16_nt_kernel_t<TM_, KS_, AF_>                                        \
                <<<grid, 256, minfer_launch_smem(launch_site, dyn_smem), stream>>>(    \
                    a, b, c, nt, od, id);                                              \
            ok = minfer_launch_ok(launch_site, nm) ? 1 : 0;                            \
        }                                                                              \
    } while (0)
#define GEMM_LAUNCH(TM_, KS_)                                                          \
    do {                                                                               \
        dim3 grid((nt + 63) / 64, (od + TM_ - 1) / TM_);                               \
        if (af32) {                                                                    \
            GEMM_ONE(TM_, KS_, true);                                                  \
        } else {                                                                       \
            GEMM_ONE(TM_, KS_, false);                                                 \
        }                                                                              \
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
#undef GEMM_ONE
    return ok;
}

// P6: A arrives as f32 activations; converts inside the kernel on stage.
// #147: returns `launch_gemm_f16`'s own result (1 = launched and accepted) so
// the af32 path's failed launch is checked by its caller too.
int launch_gemm_f32a(
    const float* a, const __half* b, float* c,
    int nt, int od, int id, cudaStream_t stream
) {
    const int ok =
        launch_gemm_f16(reinterpret_cast<const __half*>(a), b, c, nt, od, id, stream, true);
    // The launch error (and the opt-in failure) is named inside
    // `launch_gemm_f16` now (#147); this stays for the opt-in synchronous
    // fault probe.
    if (getenv("MINFER_A32_SYNC") && ok) {
        cudaError_t e = cudaStreamSynchronize(stream);
        if (e != cudaSuccess) {
            fprintf(stderr, "minfer/cuda: af32 gemm ASYNC FAULT nt=%d od=%d id=%d: %s\n",
                    nt, od, id, cudaGetErrorString(e));
            cudaGetLastError();  // named here, cleared here
        }
    }
    return ok;
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
    minfer_launch_prelude("launch:gemm_qb_nt", "gemm_qb_nt_kernel");
    gemm_qb_nt_kernel<<<grid, minfer_launch_block("launch:gemm_qb_nt", 256), 0, stream>>>(a, w, c, nt, od, id, type_id, q6_stride);
    minfer_launch_ok("launch:gemm_qb_nt", "gemm_qb_nt_kernel");
}

} // extern "C"
// r59 rider (r57 items 4+5): force the fatbin module load + first-touch
// attribute queries at REGISTRATION time — cudaFuncGetAttributes on the
// launch set loads the module, moving the ~3 ms first-launch host stalls
// (r58 CUPTI: bracketing the first mode-2 swiglu / first bt matmul) out of
// the measured prefill window. Attribute errors are ignored (a missing
// instantiation only means that path was never compiled in).
//
// #263: decomposed into one `minfer_prewarm_<family>_kernels()` per
// translation unit. A *template* address taken across TUs is nvcc
// warning #20280-D and can fail to link, so each family registers its own;
// the dispatcher only makes plain host calls.

// Per-family pre-warm entries, one per translation unit that owns
// kernels the fatbin module must load.
extern "C" void minfer_prewarm_mmq_nb_kernels(void);
extern "C" void minfer_prewarm_mmq_raw_kernels(void);
extern "C" void minfer_prewarm_mmq_bt_q6k_kernels(void);
extern "C" void minfer_prewarm_attention_prefill_kernels(void);
extern "C" void minfer_prewarm_attention_decode_kernels(void);
extern "C" void minfer_prewarm_mmvq_aquant_kernels(void);
extern "C" void minfer_prewarm_mmvq_skipwrite_kernels(void);
extern "C" void minfer_prewarm_kernels(void) {
    cudaFuncAttributes a;
    // ops_elementwise
    MINFER_PREWARM_ONE(a, swiglu_f32_off);
    MINFER_PREWARM_ONE(a, rms_norm_f32);
    MINFER_PREWARM_ONE(a, rope_f32);
    // kv_store
    MINFER_PREWARM_ONE(a, store_kv_f16);
    // ops_misc
    MINFER_PREWARM_ONE(a, f32_f32_matmul_vec);
    MINFER_PREWARM_ONE(a, f16_f32_matmul_vec);
    MINFER_PREWARM_ONE(a, embed_rows_f16);
    minfer_prewarm_mmq_nb_kernels();
    minfer_prewarm_mmq_raw_kernels();
    minfer_prewarm_mmq_bt_q6k_kernels();
    minfer_prewarm_attention_prefill_kernels();
    minfer_prewarm_attention_decode_kernels();
    minfer_prewarm_mmvq_aquant_kernels();
    minfer_prewarm_mmvq_skipwrite_kernels();
}

extern "C" {
} // extern "C"

// ─── C3/C7b: move KV rows within one arena (arena compaction) ───────────────
// One block walks the rows one at a time with a barrier between them, **in the
// direction the overlap requires**: ascending when the run slides down, descending
// when it slides up (C7b, where growing a run pushes the runs above it up). Either
// way the row a write could clobber has already been copied. Overlapping is the
// normal case — a compaction slides a run into the gap next to it — and
// `cudaMemcpyAsync` device-to-device is documented undefined for overlapping
// ranges, which is exactly why this is a kernel and not a memcpy: no staging
// buffer, no second pass.
__global__ void kv_move_rows(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int dst_row, int src_row, int rows, int elems
) {
    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    const bool down = dst_row <= src_row;
    for (int k = 0; k < rows; k++) {
        const int r = down ? k : (rows - 1 - k);
        const float* s = src + ((size_t)src_row + (size_t)r) * (size_t)elems;
        float* d = dst + ((size_t)dst_row + (size_t)r) * (size_t)elems;
        for (int i = tid; i < elems; i += nth) d[i] = s[i];
        // Every thread must finish this row before any thread touches the next one:
        // a later write can land on a row that is still being read.
        __syncthreads();
    }
}

// Returns 0 on success, non-zero when the contract is violated or the launch
// itself failed (the caller turns that into an `Err`, never a silent no-op).
// A documented `_opt` site (#162): the int return is the decision, so the launch
// failure is named at the site and cleared, and no sticky is set.
extern "C" int launch_kv_move_rows(
    float* dst, const float* src,
    int dst_row, int src_row, int rows, int elems,
    cudaStream_t stream
) {
    if (rows <= 0 || elems <= 0) return 0;
    if (dst_row < 0 || src_row < 0) return 1;
    minfer_launch_prelude("launch:kv_move_rows", "kv_move_rows");
    kv_move_rows<<<1, minfer_launch_block("launch:kv_move_rows", 256), 0, stream>>>(dst, src, dst_row, src_row, rows, elems);
    return minfer_launch_ok_opt("launch:kv_move_rows", "kv_move_rows") ? 0 : 1;
}
