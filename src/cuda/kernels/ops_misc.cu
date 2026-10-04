// src/cuda/kernels/ops_misc.cu — CUDA kernel source for minfer (issue #263 step 2).
//
// Includes the shared declarations/helpers from `common.cuh`; the
// launchers below live in the translation unit that instantiates the
// kernels they launch (no `-rdc`, no cross-TU template reference).
#include "common.cuh"


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
}


// ─── #223 pre-warm entry for this translation unit ───────────────────────────
// These kernels are plain `__global__` functions, but their module is loaded
// by the pre-warm, so the family keeps its own registration entry.
extern "C" void minfer_prewarm_ops_misc_kernels(void) {
    cudaFuncAttributes a;
    MINFER_PREWARM_ONE(a, f32_f32_matmul_vec);
    MINFER_PREWARM_ONE(a, f16_f32_matmul_vec);
    MINFER_PREWARM_ONE(a, embed_rows_f16);
}
