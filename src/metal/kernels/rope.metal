// ─── RoPE (in-place) ─────────────────────────────────────────
// Applies rotary positional embedding to Q and K.
// x layout: [nt][n_head][n_dims]

kernel void kernel_rope_f32(
    device float * x [[buffer(0)]],
    constant int & n_head [[buffer(1)]],
    constant int & n_dims [[buffer(2)]],
    constant int & nt [[buffer(3)]],
    constant float & freq_base [[buffer(4)]],
    constant float & freq_scale [[buffer(5)]],
    constant int * positions [[buffer(6)]],
    constant int & rope_style [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]   // (dim, head, token)
) {
    int half_dim = n_dims / 2;
    int d = tid.x;       // 0..half_dim-1
    int h = tid.y;       // 0..n_head-1
    int t = tid.z;       // 0..nt-1
    if (t >= nt || h >= n_head || d >= half_dim) return;
    int base = (t * n_head + h) * n_dims;
    float freq = freq_scale / pow(freq_base, (2.0 * d) / n_dims);
    float theta = positions[t] * freq;
    float cs = cos(theta), sn = sin(theta);
    int i0, i1;
    if (rope_style == 1) {
        i0 = base + 2 * d;
        i1 = base + 2 * d + 1;
    } else {
        i0 = base + d;
        i1 = base + d + half_dim;
    }
    float x0 = x[i0], x1 = x[i1];
    x[i0] = x0 * cs - x1 * sn;
    x[i1] = x0 * sn + x1 * cs;
}

