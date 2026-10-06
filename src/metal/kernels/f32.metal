// ─── F32 weight matmul ──────────────────────────────────────
//
// The f32 twin of `kernel_f16_f32_matmul` (`f16.metal`). A GGUF with
// `general.file_type = F32` (`minfer convert --outtype f32`, `llama-quantize …
// F32`) stores its 2-D weights as f32, and the loader registers them raw, so the
// matmul dispatch needs a kernel that reads `device const float *` weight rows.
// CUDA has the same arm (`launch_f32_f32_matmul`, `src/cuda/kernels/ops_misc.cu`);
// before #317 Metal silently fell through to the Q4_0 kernel, which reads the
// f32 bytes as Q4_0 blocks (a zero f16 scale) and writes zeros.
//
// Same geometry as the f16 kernel: one 64-thread threadgroup covers NR0*NSG = 8
// output rows, the token dimension loops inside so a prefill re-streams a weight
// row once per threadgroup. Weights are [od][id] row-major; acts are [nt][id];
// output is [nt][od].
kernel void kernel_f32_f32_matmul(
    device const float * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant     int   * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiisg [[thread_index_in_simdgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]]
) {
    const short NR0 = 4;
    const short NSG = 2;
    const int od = p[0];
    const int id = p[1];
    const int nt = p[2];
    const int r0 = ((int)tgpig.x * NSG + (int)sgitg) * NR0;

    for (int t = 0; t < nt; t++) {
        device const float * y = acts + (size_t)t * id;
        float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
        for (int i = (int)tiisg; i < id; i += 32) {
            float a = y[i];
            if (r0 + 0 < od) acc0 += weights[(size_t)(r0 + 0) * id + i] * a;
            if (r0 + 1 < od) acc1 += weights[(size_t)(r0 + 1) * id + i] * a;
            if (r0 + 2 < od) acc2 += weights[(size_t)(r0 + 2) * id + i] * a;
            if (r0 + 3 < od) acc3 += weights[(size_t)(r0 + 3) * id + i] * a;
        }
        float s0 = simd_sum(acc0);
        float s1 = simd_sum(acc1);
        float s2 = simd_sum(acc2);
        float s3 = simd_sum(acc3);
        if (tiisg == 0) {
            if (r0 + 0 < od) output[(size_t)t * od + r0 + 0] = s0;
            if (r0 + 1 < od) output[(size_t)t * od + r0 + 1] = s1;
            if (r0 + 2 < od) output[(size_t)t * od + r0 + 2] = s2;
            if (r0 + 3 < od) output[(size_t)t * od + r0 + 3] = s3;
        }
    }
}
