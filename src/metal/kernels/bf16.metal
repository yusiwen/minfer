// ─── BF16 weight kernels (#208) ─────────────────────────────
//
// bf16 is f32's top 16 bits, so the decode is a left shift — `as_type<float>`
// of `bits << 16`, **exact** for every value including NaNs. That is the same
// promotion `crate::block::bf16_to_f32` / `common.cuh::b2f` perform, and the
// Metal twin of CUDA's #208 kernels (`bf16_f32_matmul_vec`/`_scalar`,
// `embed_rows_bf16`).
//
// **Its own kernel, not a dtype flag on the f16 one** (the argument #208's CUDA
// half settled): bf16 and f16 are different 2 B/element layouts, so a shared
// kernel would need a per-element branch in the hottest device kernel to choose
// between `as_type<float>(bits << 16)` and `float(half)`. Two kernels keep the
// inner loops branch-free. The geometry is the f16 pair's exactly.
//
// A bf16 GGUF keeps its 2-D weights 2 B/element on the device (no registration
// copy); 1-D norms/biases stay f32 — the file contract `minfer convert
// --outtype bf16` writes.

// bf16 weight × f32 activation matmul. One threadgroup covers NR0*NSG output
// rows; each simdgroup owns NR0 of them. Grid is (ceil(od / (NR0*NSG)), 1) and
// the token dimension loops inside the threadgroup (like the f16 kernel — and
// CUDA's), so a prefill re-streams the weight rows once per threadgroup instead
// of once per (token, threadgroup). Weights are [od][id] row-major bf16
// (`ushort` words); acts are [nt][id] f32; output is [nt][od] f32.
kernel void kernel_bf16_f32_matmul(
    device const ushort * weights [[buffer(0)]],
    device const float  * acts    [[buffer(1)]],
    device       float  * output  [[buffer(2)]],
    constant     int    * p       [[buffer(3)]],
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
            if (r0 + 0 < od) acc0 += as_type<float>(((uint)weights[(size_t)(r0 + 0) * id + i]) << 16) * a;
            if (r0 + 1 < od) acc1 += as_type<float>(((uint)weights[(size_t)(r0 + 1) * id + i]) << 16) * a;
            if (r0 + 2 < od) acc2 += as_type<float>(((uint)weights[(size_t)(r0 + 2) * id + i]) << 16) * a;
            if (r0 + 3 < od) acc3 += as_type<float>(((uint)weights[(size_t)(r0 + 3) * id + i]) << 16) * a;
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

// bf16 token-embedding row gather: one thread per output element. `weights` is
// [n_vocab][ne] bf16 (`ushort`), `ids` [nt] i32 (rule §4: I32 stored as an f32
// bit pattern, so the `int` load reads the right value), `dst` [nt][ne] f32.
kernel void kernel_get_rows_bf16(
    device const ushort * weights [[buffer(0)]],
    device const int    * ids     [[buffer(1)]],
    device       float  * dst     [[buffer(2)]],
    constant     int    & ne      [[buffer(3)]],
    constant     int    & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int total = nt * ne;
    int idx = (int)tid;
    if (idx >= total) return;
    int t = idx / ne;
    int i = idx % ne;
    int token_id = ids[t];
    dst[idx] = as_type<float>(((uint)weights[(size_t)token_id * ne + i]) << 16);
}
