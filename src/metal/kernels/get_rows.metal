// ─── Get rows (embedding lookup, Q4_0 → f32) ────────────────
// Reads rows from a quantized embedding table and dequantizes to f32.
// weights: [n_vocab][nb * Q4B], ids: [nt], dst: [nt][ne].

kernel void kernel_get_rows_q4_0(
    device const uchar  * weights [[buffer(0)]],
    device const int    * ids     [[buffer(1)]],
    device       float  * dst     [[buffer(2)]],
    constant    int     & ne      [[buffer(3)]],
    constant    int     & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int nb = ne / 32;
    int total = nt * nb;
    int idx = (int)tid;
    if (idx >= total) return;

    int t = idx / nb;
    int b = idx % nb;
    int token_id = ids[t];

    int off = (token_id * nb + b) * Q4B;
    device const half * hptr = (device const half *)(weights + off);
    float d4 = float(hptr[0]);
    device const uchar * qs = weights + off + 2;

    int base = t * ne + b * 32;
    for (int j = 0; j < 16; j++) {
        uchar byte = qs[j];
        dst[base + j]      = float(int(byte & 0x0F) - 8) * d4;
        dst[base + j + 16] = float(int(byte >> 4) - 8) * d4;
    }
}

// ─── GET_ROWS Q4_K: embedding lookup for Q4_K tables (256-elem super-blocks).
// One thread per (token, 16-element group); reuses dequant_q4_k_16 (the same
// helper the Q4_K GEMM kernels use, validated byte-identical to llama's
// dequantize_row_q4_K). Matches llama's kernel_get_rows_q<block_q4_K, ...>.
// Host guards ne % 256 == 0 (super-block alignment).
kernel void kernel_get_rows_q4_k(
    device const uchar  * weights [[buffer(0)]],
    device const int    * ids     [[buffer(1)]],
    device       float  * dst     [[buffer(2)]],
    constant    int     & ne      [[buffer(3)]],
    constant    int     & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    constexpr int Q4KB = 144;
    int nsb = (ne / 256) * 16;   // 16-element groups per token row
    int total = nt * nsb;
    int idx = (int)tid;
    if (idx >= total) return;

    int t = idx / nsb;
    int g = idx % nsb;
    int token_id = ids[t];

    int super = g / 16;
    int il = g % 16;
    int off = (token_id * (ne / 256) + super) * Q4KB;

    float4x4 reg;
    dequant_q4_k_16(weights + off, il, reg);

    int base = t * ne + g * 16;
    for (int j = 0; j < 4; j++) {
        dst[base + j*4 + 0] = reg[j][0];
        dst[base + j*4 + 1] = reg[j][1];
        dst[base + j*4 + 2] = reg[j][2];
        dst[base + j*4 + 3] = reg[j][3];
    }
}

// ─── GET_ROWS for the remaining embedding types (2026-08-21) ─────
// llama's kernel_get_rows_q covers every quant type; minfer had Q4_0 + Q4_K
// only, so Q5_0/Q5_1/Q8_0/Q6_K/Q5_K-embedding models fell back to CPU dequant
// + full upload_hidden per prefill. These are templates over the validated
// dequant_*_16 helpers (the same ones the GEMM/matmul kernels use):
//   - 32-elem block types (Q4_1/Q5_0/Q5_1/Q8_0): one thread per block, two
//     16-elem halves (il=0,1) — same structure as kernel_get_rows_q4_0
//   - 256-elem super-block types (Q6_K/Q5_K): one thread per 16-elem group
//     (il=0..15) — same structure as kernel_get_rows_q4_k
// Host guards: ne % 32 == 0 (block types) / ne % 256 == 0 (super-block types).

template<int BS, void (*dequant16)(const device uchar *, short, thread float4x4 &)>
kernel void kernel_get_rows_q32(
    device const uchar  * weights [[buffer(0)]],
    device const int    * ids     [[buffer(1)]],
    device       float  * dst     [[buffer(2)]],
    constant    int     & ne      [[buffer(3)]],
    constant    int     & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int nb = ne / 32;
    int idx = (int)tid;
    if (idx >= nt * nb) return;
    int t = idx / nb;
    int b = idx % nb;
    int off = (ids[t] * nb + b) * BS;
    float4x4 r0, r1;
    dequant16(weights + off, 0, r0);
    dequant16(weights + off, 1, r1);
    int base = t * ne + b * 32;
    for (int j = 0; j < 4; j++) {
        dst[base + j*4 + 0] = r0[j][0];
        dst[base + j*4 + 1] = r0[j][1];
        dst[base + j*4 + 2] = r0[j][2];
        dst[base + j*4 + 3] = r0[j][3];
        dst[base + 16 + j*4 + 0] = r1[j][0];
        dst[base + 16 + j*4 + 1] = r1[j][1];
        dst[base + 16 + j*4 + 2] = r1[j][2];
        dst[base + 16 + j*4 + 3] = r1[j][3];
    }
}

template<int BS, void (*dequant16)(const device uchar *, short, thread float4x4 &)>
kernel void kernel_get_rows_q256(
    device const uchar  * weights [[buffer(0)]],
    device const int    * ids     [[buffer(1)]],
    device       float  * dst     [[buffer(2)]],
    constant    int     & ne      [[buffer(3)]],
    constant    int     & nt      [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    int nsb = (ne / 256) * 16;   // 16-element groups per token row
    int idx = (int)tid;
    if (idx >= nt * nsb) return;
    int t = idx / nsb;
    int g = idx % nsb;
    int super = g / 16;
    int il = g % 16;
    int off = (ids[t] * (ne / 256) + super) * BS;
    float4x4 reg;
    dequant16(weights + off, il, reg);
    int base = t * ne + g * 16;
    for (int j = 0; j < 4; j++) {
        dst[base + j*4 + 0] = reg[j][0];
        dst[base + j*4 + 1] = reg[j][1];
        dst[base + j*4 + 2] = reg[j][2];
        dst[base + j*4 + 3] = reg[j][3];
    }
}

typedef decltype(kernel_get_rows_q32<20, dequant_q4_1_16>) get_rows_q32_t;
typedef decltype(kernel_get_rows_q256<210, dequant_q6_k_16>) get_rows_q256_t;

template [[host_name("kernel_get_rows_q4_1")]] kernel get_rows_q32_t kernel_get_rows_q32<20, dequant_q4_1_16>;
template [[host_name("kernel_get_rows_q5_0")]] kernel get_rows_q32_t kernel_get_rows_q32<22, dequant_q5_0_16>;
template [[host_name("kernel_get_rows_q5_1")]] kernel get_rows_q32_t kernel_get_rows_q32<24, dequant_q5_1_16>;
template [[host_name("kernel_get_rows_q8_0")]] kernel get_rows_q32_t kernel_get_rows_q32<34, dequant_q8_0_16>;
template [[host_name("kernel_get_rows_q6_k")]] kernel get_rows_q256_t kernel_get_rows_q256<210, dequant_q6_k_16>;
template [[host_name("kernel_get_rows_q5_k")]] kernel get_rows_q256_t kernel_get_rows_q256<176, dequant_q5_k_16>;

// Generic f32 row selection (llama ggml_get_rows for f32 data): the graph's
// n_out tail-row reduction. One thread per (row, element):
//   out[t*ne + i] = x[ids[t]*ne + i]
kernel void kernel_get_rows_f32(
    device const float  * x    [[buffer(0)]],
    device const int    * ids  [[buffer(1)]],
    device       float  * out  [[buffer(2)]],
    constant    int     & ne   [[buffer(3)]],
    uint2 tid [[thread_position_in_grid]]
) {
    int t = tid.x;
    int i = tid.y;
    int id = ids[t];
    out[t * ne + i] = x[id * ne + i];
}

// ─── GPU warm-up read (2026-08-21, METAL_OPTIMIZATIONS #39) ─────
// Dummy full-buffer read at model load: the FIRST GPU access to file-backed
// (mmap) pages costs ~44 ms of one-time page/TLB setup per process; running a
// read of every page HERE moves that cost out of the first prefill (llama-bench
// numbers are equally warm). Each thread reads one uchar4 (faults its page);
// only thread 0 writes a single byte to the small output buffer.
kernel void kernel_warmup_read(
    device const uchar * src [[buffer(0)]],
    device       uchar * out [[buffer(1)]],
    uint tid [[thread_position_in_grid]]
) {
    device const uchar4 * p = (device const uchar4 *)src + tid;
    if (tid == 0) out[0] = p->x;
}

