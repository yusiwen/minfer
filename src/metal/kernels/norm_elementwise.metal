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
    int rem = d - d4 * 4;
    if (tpitg.x == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        for (int i = 0; i < rem; i++) ss += x_tail[i] * x_tail[i];
    }
    ss = simd_sum(ss);

    float scale = 1.0f / sqrt(ss / (float)d + eps);

    device float4 * y4 = (device float4 *)(y + row * d);
    device const float4 * w4 = (device const float4 *)w;
    for (int i = tpitg.x; i < d4; i += 32) {
        y4[i] = x4[i] * scale * w4[i];
    }
    if (tpitg.x == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        device       float * y_tail = y + row * d + d4 * 4;
        device const float * w_tail = w + d4 * 4;
        for (int i = 0; i < rem; i++) y_tail[i] = x_tail[i] * scale * w_tail[i];
    }
}

// ─── Add bias ────────────────────────────────────────────────
// y[t][i] += b[i]

// ─── RMSNorm, multi-simdgroup (256-thread) variant ───────────
// Faithful llama.cpp transcription (kernel_rms_norm_fuse_impl): the threadgroup
// is 256 threads (8 simdgroups). Per-simdgroup partial sums are reduced through
// a small threadgroup buffer with TWO threadgroup barriers. The 32-thread
// single-simdgroup kernel above was measured at ~7x the per-dispatch cost of
// the 256-thread elementwise kernels (P0 profile 2026-08-10) — a single simdgroup
// cannot hide DRAM latency for one 896-element row. Dispatch nt = min(d/4, 256).
kernel void kernel_rms_norm_f32_256(
    device const float * x       [[buffer(0)]],
    device const float * w       [[buffer(1)]],
    device       float * y       [[buffer(2)]],
    constant    int    & d       [[buffer(3)]],
    constant    float  & eps     [[buffer(4)]],
    threadgroup float * shmem [[threadgroup(0)]],
    uint3  tgpig   [[threadgroup_position_in_grid]],
    ushort tiisg   [[thread_index_in_simdgroup]],
    ushort sgitg   [[simdgroup_index_in_threadgroup]]
) {
    if (sgitg == 0) {
        shmem[tiisg] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int ntg = 256; // dispatched threads per threadgroup (8 simdgroups)
    int row = tgpig.x;
    int d4 = d / 4;
    device const float4 * x4 = (device const float4 *)(x + row * d);
    float ss = 0.0f;
    for (int i = 32 * sgitg + tiisg; i < d4; i += ntg) {
        ss += dot(x4[i], x4[i]);
    }
    int rem = d - d4 * 4;
    if (tiisg == 0 && sgitg == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        for (int i = 0; i < rem; i++) ss += x_tail[i] * x_tail[i];
    }
    ss = simd_sum(ss);

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    ss = shmem[tiisg];
    ss = simd_sum(ss);

    float scale = 1.0f / sqrt(ss / (float)d + eps);

    device float4 * y4 = (device float4 *)(y + row * d);
    device const float4 * w4 = (device const float4 *)w;
    for (int i = 32 * sgitg + tiisg; i < d4; i += ntg) {
        y4[i] = x4[i] * scale * w4[i];
    }
    if (tiisg == 0 && sgitg == 0) {
        device const float * x_tail = x + row * d + d4 * 4;
        device       float * y_tail = y + row * d + d4 * 4;
        device const float * w_tail = w + d4 * 4;
        for (int i = 0; i < rem; i++) y_tail[i] = x_tail[i] * scale * w_tail[i];
    }
}


 kernel void kernel_add_bias_f32(
     device       float * y [[buffer(0)]],
     device const float * b [[buffer(1)]],
     constant    int    & d [[buffer(2)]],
     uint2 tid [[thread_position_in_grid]]
 ) {
     const int t = tid.x, i4 = tid.y;
     const int i = i4 * 4;
     if (i + 3 < d) {
         *(device float4 *)(y + t * d + i) += *(device const float4 *)(b + i);
     } else {
         for (int k = i; k < d; k++) y[t * d + k] += b[k];
     }
 }

// ─── Element-wise add (float4) ───────────────────────────────
// z = x + y; 4 elements per thread, scalar tail for n % 4 != 0.

kernel void kernel_add_f32(
    device const float * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device       float * z [[buffer(2)]],
    constant    int    & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        *(device float4 *)(z + 4*t) = *(device const float4 *)(x + 4*t) + *(device const float4 *)(y + 4*t);
    } else {
        for (int k = 4*t; k < n; k++) z[k] = x[k] + y[k];
    }
}

// ─── Element-wise multiply (float4) ──────────────────────────
// z[t] = x[t] * y[t]; 4 elements per thread.

kernel void kernel_mul_f32(
    device const float * x [[buffer(0)]],
    device const float * y [[buffer(1)]],
    device       float * z [[buffer(2)]],
    constant    int    & n [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        *(device float4 *)(z + 4*t) = *(device const float4 *)(x + 4*t) * *(device const float4 *)(y + 4*t);
    } else {
        for (int k = 4*t; k < n; k++) z[k] = x[k] * y[k];
    }
}

// ─── SiLU (in-place, float4) ─────────────────────────────────
// y[i] = y[i] / (1 + exp(-y[i])); 4 elements per thread.

kernel void kernel_silu_f32(
    device float * y [[buffer(0)]],
    constant int & n [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        float4 v = *(device float4 *)(y + 4*t);
        float4 r;
        r.x = v.x / (1.0f + exp(-v.x));
        r.y = v.y / (1.0f + exp(-v.y));
        r.z = v.z / (1.0f + exp(-v.z));
        r.w = v.w / (1.0f + exp(-v.w));
        *(device float4 *)(y + 4*t) = r;
    } else {
        for (int k = 4*t; k < n; k++) {
            float v = y[k];
            y[k] = v / (1.0f + exp(-v));
        }
    }
}

// ─── SwiGLU (fused SiLU + Mul, float4) ───────────────────────
// dst[i] = silu(gate[i]) * up[i]; 4 elements per thread.

kernel void kernel_swiglu_f32(
    device const float * gate [[buffer(0)]],
    device const float * up   [[buffer(1)]],
    device       float * dst  [[buffer(2)]],
    constant    int    & n    [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    const int n4 = n >> 2;
    int t = (int)tid;
    if (t < n4) {
        float4 g = *(device const float4 *)(gate + 4*t);
        float4 u = *(device const float4 *)(up + 4*t);
        float4 r;
        r.x = (g.x / (1.0f + exp(-g.x))) * u.x;
        r.y = (g.y / (1.0f + exp(-g.y))) * u.y;
        r.z = (g.z / (1.0f + exp(-g.z))) * u.z;
        r.w = (g.w / (1.0f + exp(-g.w))) * u.w;
        *(device float4 *)(dst + 4*t) = r;
    } else {
        for (int k = 4*t; k < n; k++) {
            float g = gate[k];
            dst[k] = (g / (1.0f + exp(-g))) * up[k];
        }
    }
}

