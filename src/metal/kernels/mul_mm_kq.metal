// ─── Q5_0 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure; Q5_0: d(2) + qh(4) + qs(16) = 22 B. Signed (val - 16).
kernel void kernel_q5_0_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q5B = 22;
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
        dequant_q5_0_16(weights + (r0 + lr0) * nblk * Q5B + (loop_k/32) * Q5B, il0, temp_a);

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

// ─── Q5_1 × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same structure; Q5_1: d(2) + m(2) + qh(4) + qs(16) = 24 B. Unsigned + m.
kernel void kernel_q5_1_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q51B = 24;
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
        dequant_q5_1_16(weights + (r0 + lr0) * nblk * Q51B + (loop_k/32) * Q51B, il0, temp_a);

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

// ─── Q6_K × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same 64×32-tile structure as Q4_0, but the weights are 256-element super-blocks
// (Q6KB=210). Each 32-elem K step spans 2 "il halves" of the super-block (il =
// (loop_k%256)/16 + il0, il0 = 0/1), dequantized by dequant_q6_k_16.
kernel void kernel_q6_k_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q6KB = 210;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 256;   // super-blocks per row

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
        // super-block at loop_k/256; the 32-elem step spans 2 il-halves
        dequant_q6_k_16(weights + (r0 + lr0) * nblk * Q6KB + (loop_k/256) * Q6KB,
                        ((loop_k % 256) / 16) + il0, temp_a);

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

// ─── Q4_K × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same 256-elem super-block structure as Q6_K; Q4_K: 144 B/super-block.
kernel void kernel_q4_k_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q4KB = 144;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 256;

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
        dequant_q4_k_16(weights + (r0 + lr0) * nblk * Q4KB + (loop_k/256) * Q4KB,
                        ((loop_k % 256) / 16) + il0, temp_a);

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

// ─── Q5_K × f32 GEMM (simdgroup-cooperative, prefill nt>=16) ──
// Same 256-elem super-block structure; Q5_K: 176 B/super-block.
kernel void kernel_q5_k_mm_f32(
    device const uchar * weights [[buffer(0)]],
    device const float * acts    [[buffer(1)]],
    device       float * output  [[buffer(2)]],
    constant    int    * p       [[buffer(3)]],
    uint3  tgpig [[threadgroup_position_in_grid]],
    ushort tiitg [[thread_index_in_threadgroup]],
    ushort sgitg [[simdgroup_index_in_threadgroup]],
    threadgroup char * shmem [[threadgroup(0)]]
) {
    constexpr int Q5KB = 176;
    constexpr int NR0 = 64;
    constexpr int NR1 = 32;
    constexpr int NK  = 32;
    constexpr int NL0 = 2;
    constexpr int NL1 = 4;

    const int M = p[0], K = p[1], N = p[2];
    const int nblk = K / 256;

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
        dequant_q5_k_16(weights + (r0 + lr0) * nblk * Q5KB + (loop_k/256) * Q5KB,
                        ((loop_k % 256) / 16) + il0, temp_a);

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
