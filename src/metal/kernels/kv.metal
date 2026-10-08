// ─── KV cache store ──────────────────────────────────────────
// Scatters nt new K/V rows into the persistent KV cache at positions[].

 kernel void kernel_store_kv_f32(
    device const float * src [[buffer(0)]],
    device       float * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int j = tid.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = src[t * nkt + j];
}

// F16 KV cache variant: stores f32 K/V rows into a half cache (2 bytes/elem),
// halving attention memory bandwidth (matches llama.cpp's F16 cache).
kernel void kernel_store_kv_f16(
    device const float * src [[buffer(0)]],
    device       half  * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int j = tid.y;
    if (t >= nt || j >= nkt) return;
    dst[positions[t] * nkt + j] = half(src[t * nkt + j]);
}

// C4 S2b Metal twin (issue #310): quantize nt f32 rows into packed Q8_0 cells
// of a persistent KV region. One thread per (row, 32-element block). The
// quantizer is the CPU's (`quants::quantize_row_q8_0_into`, whose aarch64 path
// is `quantize_scalar`): `d = amax/127`, the f16 scale written round-to-nearest-
// even, and each quant `rint(x/d)` clamped to the i8 range — `rint` is
// round-to-nearest-even under Metal's default rounding mode, the same
// `round_ties_even` the CPU uses, so both backends write the same bytes for the
// same row. `row_bytes` is the packed cell's word-padded byte width
// (`KvFormat::Q8_0.row_bytes(nkt)`), which is what makes a cell move a plain
// word copy. This is the only packed store: CUDA's `store_kv_q8_0` is its
// counterpart.
kernel void kernel_store_kv_q8_0(
    device const float * src [[buffer(0)]],
    device       uchar * dst [[buffer(1)]],
    constant    int    & nkt [[buffer(2)]],
    constant    int    & nt  [[buffer(3)]],
    constant    int    * positions [[buffer(4)]],
    constant    uint   & row_bytes [[buffer(5)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t   = tid.x;
    int blk = tid.y;
    int nblk = nkt / 32;
    if (t >= nt || blk >= nblk) return;
    int p = positions[t];
    device const float * x = src + (size_t)t * nkt + (size_t)blk * 32;
    float am = 0.0f;
    for (int i = 0; i < 32; i++) am = max(am, fabs(x[i]));
    const float d = am / 127.0f;
    const float id = (d != 0.0f) ? (1.0f / d) : 0.0f;
    device uchar * cell = dst + (size_t)p * (size_t)row_bytes + (size_t)blk * 34;
    *(device half *)cell = half(d);
    device char * q = (device char *)(cell + 2);
    for (int i = 0; i < 32; i++) {
        q[i] = (char)(int)clamp(rint(x[i] * id), -128.0f, 127.0f);
    }
}

// ─── Fused bias-add + RoPE + KV-store (nt==1 decode) ──────────
// One kernel replaces add_bias×3 + rope×2 + store_kv×2 (7 dispatches → 1).
// bqkv layout (nt==1): [q: 0..nqt][k: nqt..nqt+nkt][v: nqt+nkt..nqt+2nkt].
// Applies the per-section bias, RoPE to the q/k sections, and stores k/v into
// the KV cache (f32 or f16 per kv_is_f16). Bit-identical to the 7 separate
// kernels: bias then rope on the same values, same store addresses.
// Thread mapping: one thread per (head, d<half_dim) rope pair for q and k,
// one thread per v element → grid = nqt/2 + nkt/2 + nkt.

kernel void kernel_attn_bias_rope_store(
    device       float * bqkv [[buffer(0)]],
    device const float * bias_q [[buffer(1)]],
    device const float * bias_k [[buffer(2)]],
    device const float * bias_v [[buffer(3)]],
    device        void * kv_k [[buffer(4)]],
    device        void * kv_v [[buffer(5)]],
    constant        int & nqt [[buffer(6)]],
    constant        int & nkt [[buffer(7)]],
    constant        int & hd  [[buffer(8)]],
    constant      float & freq_base [[buffer(9)]],
    constant      float & freq_scale [[buffer(10)]],
    constant        int & pos [[buffer(11)]],
    constant        int & rope_style [[buffer(12)]],
    constant        int & kv_is_f16 [[buffer(13)]],
    uint tid [[thread_position_in_grid]]
) {
    const int half_dim = hd / 2;
    const int qpairs = nqt / 2;
    const int kpairs = nkt / 2;
    const int total = qpairs + kpairs + nkt;
    const int u = (int)tid;
    if (u >= total) return;

    if (u < qpairs) {
        const int head = u / half_dim;
        const int d    = u % half_dim;
        const int base = head * hd;
        const int i0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int i1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        float x0 = bqkv[i0] + bias_q[i0];
        float x1 = bqkv[i1] + bias_q[i1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        bqkv[i0] = x0 * cs - x1 * sn;
        bqkv[i1] = x0 * sn + x1 * cs;
    } else if (u < qpairs + kpairs) {
        const int u2   = u - qpairs;
        const int head = u2 / half_dim;
        const int d    = u2 % half_dim;
        const int base = head * hd;
        const int j0 = (rope_style == 1) ? (base + 2 * d)     : (base + d);
        const int j1 = (rope_style == 1) ? (base + 2 * d + 1) : (base + d + half_dim);
        const int k0 = nqt + j0, k1 = nqt + j1;
        float x0 = bqkv[k0] + bias_k[j0];
        float x1 = bqkv[k1] + bias_k[j1];
        const float freq = freq_scale / pow(freq_base, (2.0 * d) / hd);
        const float theta = pos * freq;
        const float cs = cos(theta), sn = sin(theta);
        float r0 = x0 * cs - x1 * sn;
        float r1 = x0 * sn + x1 * cs;
        bqkv[k0] = r0; bqkv[k1] = r1;
        if (kv_is_f16) {
            ((device half *)kv_k)[pos * nkt + j0] = half(r0);
            ((device half *)kv_k)[pos * nkt + j1] = half(r1);
        } else {
            ((device float *)kv_k)[pos * nkt + j0] = r0;
            ((device float *)kv_k)[pos * nkt + j1] = r1;
        }
    } else {
        const int j  = u - qpairs - kpairs;
        const int vi = nqt + nkt + j;
        float v = bqkv[vi] + bias_v[j];
        bqkv[vi] = v;
        if (kv_is_f16) {
            ((device half *)kv_v)[pos * nkt + j] = half(v);
        } else {
            ((device float *)kv_v)[pos * nkt + j] = v;
        }
    }
}

