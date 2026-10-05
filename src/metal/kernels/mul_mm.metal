// ─── Q4_0 × f32 GEMM (simdgroup_matrix, prefill nt > 1) ─────// Faithful port of llama.cpp's kernel_mul_mm_q4_0_f32 (legacy simdgroup path):
//   - dequantize_q4_0: uint16 reads + float4x4 SIMD
//   - A staged transposed into sa; B staged via float2x4 vector stores
//   - simdgroup_half8x8 inputs -> simdgroup_float8x8 accumulators
//   - mc += mb × ma (llama's exact order for the transposed-A layout)
// A = weights (od × id Q4_0), B = acts (nt × id f32), C = out (nt × od).
// M = od, K = id, N = nt. Threadgroup 128 threads (4 simdgroups), 64×32 tile.
// Grid: x = ceil(nt/32), y = ceil(od/64). smem = 8192 B (sa/sb + bc_out temp).

kernel void kernel_q4_0_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;   // NK/16
    constexpr int NL1 = 4;   // NK/8

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    // clamp thread row/col so the staging pointer stays in bounds
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    // zero the staging tiles so out-of-range rows (partial tiles) stay 0,
    // not stale/NaN threadgroup memory from previous dispatches
    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        // === Stage A: dequant Q4_0 weights into sa (llama transposed layout) ===
        thread float4x4 temp_a;
        dequant_q4_0_16(weights + (r0 + lr0) * nblk * Q4B + (loop_k/32) * Q4B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        // === Stage B: f32 activations into sb (scalar, equivalent to llama's float2x4 store) ===
        // llama writes to the TRUE (unclamped) sb position, reading from the clamped row.
        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;                 // K chunk
        const short by = (tiitg/NL1)/8;               // N group (raw, fills OOB rows w/ clamp data)
        const short bly = (tiitg/NL1)%8;              // N sub   (raw)
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        #pragma unroll
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === Matrix multiply (4 K sub-tiles of 8) ===
        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        #pragma unroll
        for (short ik = 0; ik < NK/8; ik++) {
            // llama.cpp parity (ggml-metal.metal:10239): the ik-loop only READS
            // sa/sb (staging writes are visible after the barrier above). With the
            // unrolled loop a within-simdgroup barrier suffices — the pre-unroll
            // deterministic corruption (docs §4.3.6) was a rolled-loop compiler
            // artifact, not a memory-visibility need.
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    // === Store C (M×N) to output (N×M = [p[2]][p[0]]) ===
    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        // full tile: direct transposed store
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        // partial tile (bc_out): per-simdgroup temp_str + float4 copy
        // ensure every simdgroup finished reading sa/sb before temp_str overwrites it (llama.cpp parity)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q4_1 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure as kernel_q4_0_mm_f32; Q4_1: d(2) + m(2) + qs(16) = 20 B.
kernel void kernel_q4_1_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q41B = 20;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q4_1_16(weights + (r0 + lr0) * nblk * Q41B + (loop_k/32) * Q41B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        #pragma unroll
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        #pragma unroll
        for (short ik = 0; ik < NK/8; ik++) {
            // llama.cpp parity (ggml-metal.metal:10239): the ik-loop only READS
            // sa/sb (staging writes are visible after the barrier above). With the
            // unrolled loop a within-simdgroup barrier suffices — the pre-unroll
            // deterministic corruption (docs §4.3.6) was a rolled-loop compiler
            // artifact, not a memory-visibility need.
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        // ensure every simdgroup finished reading sa/sb before temp_str overwrites it (llama.cpp parity)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

// ─── Q8_0 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure as kernel_q4_0_mm_f32; Q8_0: d(half,2) + qs(int8*32,32) = 34 B.
kernel void kernel_q8_0_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q8B = 34;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 32;

    const int r0 = (int)tgpig.y * NR0;
    const int r1 = (int)tgpig.x * NR1;

    const short nr0 = (M - r0 < NR0) ? (M - r0) : NR0;
    const short nr1 = (N - r1 < NR1) ? (N - r1) : NR1;

    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1;
    const short lr1 = ((short)tiitg/NL1) < nr1 ? ((short)tiitg/NL1) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    threadgroup half * sa = (threadgroup half *)shmem;             // 64×32 f16 = 4096 B
    threadgroup half * sb = (threadgroup half *)(shmem + 4096);    // 32×32 f16 = 2048 B

    simdgroup_half8x8 ma[4], mb[2];
    simdgroup_float8x8 mc[8];
    for (short i = 0; i < 8; i++) mc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);

    for (short i = tiitg; i < 32*32; i += 128) sb[i] = 0.0f;
    for (short i = tiitg; i < 64*32; i += 128) sa[i] = 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int loop_k = 0; loop_k < K; loop_k += NK) {
        thread float4x4 temp_a;
        dequant_q8_0_16(weights + (r0 + lr0) * nblk * Q8B + (loop_k/32) * Q8B, il0, temp_a);

        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma unroll
        for (short i = 0; i < 16; i++) {
            const short sx = 2*il0 + i/8;
            const short sy = lr0/8;
            const short lx = lr0%8;
            const short ly = i%8;
            const short ib = 8*sx + sy;
            sa[64*ib + 8*ly + lx] = half(temp_a[i/4][i%4]);
        }

        const short iy = 8*(tiitg % NL1);
        const short bx = tiitg % NL1;
        const short by = (tiitg/NL1)/8;
        const short bly = (tiitg/NL1)%8;
        const short bib = 4*bx + by;
        device const float * y = acts + (r1 + lr1)*p[1] + loop_k + iy;
        #pragma unroll
        for (short i = 0; i < 8; i++) {
            sb[64*bib + 8*bly + i] = half(y[i]);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half * lsma = sa + 4*64*(sgitg%2);
        threadgroup const half * lsmb = sb + 2*64*(sgitg/2);

        #pragma unroll
        for (short ik = 0; ik < NK/8; ik++) {
            // llama.cpp parity (ggml-metal.metal:10239): the ik-loop only READS
            // sa/sb (staging writes are visible after the barrier above). With the
            // unrolled loop a within-simdgroup barrier suffices — the pre-unroll
            // deterministic corruption (docs §4.3.6) was a rolled-loop compiler
            // artifact, not a memory-visibility need.
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 4; i++) simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 2; i++) simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
            simdgroup_barrier(mem_flags::mem_none);
            #pragma unroll
            for (short i = 0; i < 8; i++) {
                simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
            }
            lsma += 8*64;
            lsmb += 4*64;
        }
    }

    if (r0 + NR0 <= M && r1 + NR1 <= N) {
        device float * C = output + (r1 + 16*(sgitg >> 1))*p[0] + (r0 + 32*(sgitg & 1));
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i/4)*p[0] + 8*(i%4), p[0], 0, false);
        }
    } else {
        // ensure every simdgroup finished reading sa/sb before temp_str overwrites it (llama.cpp parity)
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;
        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sgitg == 0) {
            for (int j = (int)tiitg; j < nr1; j += NR1) {
                device float  * D  = output + r0 + (r1 + j)*p[0];
                device float4 * D4 = (device float4 *)D;
                threadgroup float  * C  = temp_str + j*NR0;
                threadgroup float4 * C4 = (threadgroup float4 *)C;
                int i = 0;
                for (; i < nr0/4; i++) *(D4 + i) = *(C4 + i);
                i *= 4;
                for (; i < nr0; i++) *(D + i) = *(C + i);
            }
        }
    }
}

