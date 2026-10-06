// ─── F16 weight kernels (#164) ──────────────────────────────
//
// An f16 GGUF stores 2-D tensors as half (2 B/element) and 1-D norms/biases as
// f32 (the file contract `minfer convert --outtype f16` writes). The weights
// stay 2 B/element on the device — converting them to f32 at registration would
// throw away the half-width weight stream the format exists for — so these
// kernels load `half` and promote each value in-register. This is the Metal twin
// of CUDA's #141 kernels (`f16_f32_matmul_vec`/`_scalar`, `embed_rows_f16`).
//
// A second 2 B/element dtype (bf16, #208) slots in the same way: a new
// `kernel_bf16_f32_matmul` + `kernel_get_rows_bf16` selected by the F16 arm's
// sibling in `quant_matmul_f32_on_gpu_buf` and `embed_tokens_gpu`.

// f16 weight × f32 activation matmul. One threadgroup covers NR0*NSG output
// rows; each simdgroup owns NR0 of them. Grid is (ceil(od / (NR0*NSG)), 1) and
// the token dimension loops inside the threadgroup (like CUDA's f16 matmul), so
// a prefill re-streams the weight rows once per threadgroup instead of once per
// (token, threadgroup). Weights are [od][id] row-major; acts are [nt][id];
// output is [nt][od].
kernel void kernel_f16_f32_matmul(
    device const half  * weights [[buffer(0)]],
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
            if (r0 + 0 < od) acc0 += float(weights[(size_t)(r0 + 0) * id + i]) * a;
            if (r0 + 1 < od) acc1 += float(weights[(size_t)(r0 + 1) * id + i]) * a;
            if (r0 + 2 < od) acc2 += float(weights[(size_t)(r0 + 2) * id + i]) * a;
            if (r0 + 3 < od) acc3 += float(weights[(size_t)(r0 + 3) * id + i]) * a;
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

// f16 token-embedding row gather: one thread per output element. `weights` is
// [n_vocab][ne] half, `ids` [nt] i32 (rule §4: I32 stored as an f32 bit pattern,
// so the `int` load reads the right value), `dst` [nt][ne] f32.
kernel void kernel_get_rows_f16(
    device const half  * weights [[buffer(0)]],
    device const int   * ids     [[buffer(1)]],
    device       float * dst     [[buffer(2)]],
    constant    int    & ne      [[buffer(3)]],
    constant    int    & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = nt * ne;
    int idx = (int)tid;
    if (idx >= total) return;
    int t = idx / ne;
    int i = idx % ne;
    int token_id = ids[t];
    dst[idx] = float(weights[(size_t)token_id * ne + i]);
}
