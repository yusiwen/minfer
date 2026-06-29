// Metal shader for minfer — quantized matrix multiplication (Q4_0 × Q8_0).
// Translated from: llama.cpp/ggml/src/ggml-metal/ggml-metal.metal
//   (kernel_mul_mv_q4_0_f32, simplified for minfer's row-by-row dispatch)
//
// Device capabilities required:
//   MTLGPUFamilyApple7 (simdgroup_reduction for future optimizations)
//
// Each thread computes one output element: out[t * od + o] = dot(w_row[o], x_row[t])
// where w is Q4_0 quantized, x is Q8_0 quantized.

#include <metal_stdlib>
using namespace metal;

// Block size constants (matching kernel.rs / avx2.rs)
constant int Q4B = 18;   // sizeof(BlockQ4_0): half d + uchar[16] qs
constant int Q8B = 34;   // sizeof(BlockQ8_0): half d + char[32] qs

/// Q4_0 × Q8_0 matrix multiplication kernel.
/// Each thread handles one (token, output) pair.
///
/// Buffer layout:
///   [0] weights: [layer][output_dim][Q4_0 blocks...] — contiguous Q4_0 rows
///   [1] acts:    [token][Q8_0 blocks...]              — contiguous Q8_0 rows
///   [2] output:  [token][output_dim]                  — f32 result
kernel void kernel_q4_0_q8_0_matmul(
    device const uchar  * weights  [[buffer(0)]],
    device const uchar  * acts     [[buffer(1)]],
    device       float  * output   [[buffer(2)]],
    constant    int     & od       [[buffer(3)]],
    constant    int     & id       [[buffer(4)]],
    constant    int     & nt       [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    const int t = tid.x;
    const int o = tid.y;
    if (t >= nt || o >= od) { return; }

    const int nb = id / 32;                     // number of Q8_0 blocks per row
    const int q4_stride = nb * Q4B;             // bytes per Q4_0 row
    const int q8_stride = nb * Q8B;             // bytes per Q8_0 row

    device const uchar * wrow = weights + o * q4_stride;
    device const uchar * xrow = acts    + t * q8_stride;

    float sum = 0.0f;
    for (int b = 0; b < nb; b++) {
        // Q4_0 block: [d: half][qs: uchar[16]] = 18 bytes
        device const half * wblk = (device const half *)(wrow + b * Q4B);
        float d4 = float(wblk[0]);
        device const uchar * wq = (device const uchar *)(wblk + 1);

        // Q8_0 block: [d: half][qs: char[32]] = 34 bytes
        device const half * xblk = (device const half *)(xrow + b * Q8B);
        float d8 = float(xblk[0]);
        device const char * xq = (device const char *)(xblk + 1);

        // Dot product: Q4_0 nibble decode (signed, centered by -8) × Q8_0 i8
        // Q4_0 interleaved: qs[j] = {lo=v_j, hi=v_{j+16}}
        float block_sum = 0.0f;
        for (int j = 0; j < 16; j++) {
            uchar byte = wq[j];
            int q4_lo = int(byte & 0x0F) - 8;
            int q4_hi = int(byte >> 4) - 8;
            block_sum += float(q4_lo * int(xq[j]) + q4_hi * int(xq[j + 16]));
        }
        sum += block_sum * d4 * d8;
    }
    output[t * od + o] = sum;
}
