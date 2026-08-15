// test_03_single_layer.cu
// Capture a complete single-layer forward pass (24 kernel launches) in a CUDA graph.
// Pinpoints the exact kernel that causes capture failure (error 901).
//
// Model: Qwen2-0.5B parameters (Q4_0 weights)
//   NE=896  NQT=896  NKT=128  NF=4864  NH=14  NK=2  HD=64  NT=1

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>

// Block sizes from cuda_kernels.cu
#define Q4B  18
#define Q8B  34

// ─── launch wrappers (defined in cuda_kernels.cu) ──────────────
extern "C" {
void launch_q4_0_q8_0_matmul(const uint8_t* w, const uint8_t* a, float* o, int od, int id, int nt, cudaStream_t s);
void launch_quantize_q8_0(const float* x, uint8_t* y, int dim, int nt, cudaStream_t s);
void launch_rms_norm_f32(const float* x, const float* w, float* y, int d, float eps, int n, cudaStream_t s);
void launch_add_bias_f32(float* y, const float* b, int d, int n, cudaStream_t s);
void launch_add_f32(const float* x, const float* y, float* z, int n, cudaStream_t s);
void launch_swiglu_f32(const float* gate, const float* up, float* dst, int n, cudaStream_t s);
void launch_rope_f32(float* x, int n_head, int n_dims, int nt, float fb, float fs, const int* pos, cudaStream_t s);
void launch_store_kv_f32(const float* src, float* dst, int nkt, int nt, const int* pos, cudaStream_t s);
void launch_gqa_attn_f32(const float* q, const float* k, const float* v, float* o, const int* pos, int nh, int nk, int hd, float scale, int nt, cudaStream_t s);
}

// ─── weight helper: allocate Q4_0 buffer filled with zeros ─────
static uint8_t* alloc_q4_0(int out_dim, int in_dim) {
    // Q4_0: each block encodes 32 input values into 18 bytes
    // To produce zero output: set the scale (first 2 bytes) to 0.0f
    int blocks_per_col = in_dim / 32;
    int total_blocks = out_dim * blocks_per_col;
    size_t size = (size_t)total_blocks * Q4B;
    uint8_t* buf;
    cudaMalloc(&buf, size);
    cudaMemset(buf, 0, size);
    // Zero scale → all outputs are zero (correct for testing)
    return buf;
}

// ─── helper: print a kernel launch result ───────────────────────
#define CHECK_KERNEL(n, name) do { \
    cudaError_t _e = cudaGetLastError(); \
    printf("  [%02d/24] %-24s err=%d\n", n, name, _e); \
    if (_e != 0 && !first_error) { first_error = n; first_error_name = name; } \
} while(0)

int main() {
    const int NE  = 896,  NQT = 896,  NKT = 128,  NF  = 4864;
    const int NH  = 14,   NK  = 2,    HD  = 64,   NT  = 1;

    // ─── allocate device buffers ────────────────────────────────
    float *hidden, *bn, *bq, *bk, *bv, *ba, *bf, *bg;
    uint8_t *q8_bn, *q8_ba;
    uint8_t *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;
    float *norm_attn, *norm_ffn, *bq_bias, *bk_bias, *bv_bias;
    float *kv_k, *kv_v;
    int *d_positions;

    cudaMalloc(&hidden,    NT * NE  * sizeof(float));
    cudaMalloc(&bn,        NT * NE  * sizeof(float));
    cudaMalloc(&bq,        NT * NQT * sizeof(float));
    cudaMalloc(&bk,        NT * NKT * sizeof(float));
    cudaMalloc(&bv,        NT * NKT * sizeof(float));
    cudaMalloc(&ba,        NT * NE  * sizeof(float));
    cudaMalloc(&bf,        NT * NF  * sizeof(float));
    cudaMalloc(&bg,        NT * NF  * sizeof(float));

    int q8_bn_blocks = NE / 32;
    int q8_ba_blocks = NF / 32;  // NF > NE
    cudaMalloc(&q8_bn,     NT * q8_bn_blocks * Q8B);
    cudaMalloc(&q8_ba,     NT * q8_ba_blocks * Q8B);

    wq     = alloc_q4_0(NQT, NE);
    wk     = alloc_q4_0(NKT, NE);
    wv     = alloc_q4_0(NKT, NE);
    wo     = alloc_q4_0(NE, NE);
    w_gate = alloc_q4_0(NF, NE);
    w_up   = alloc_q4_0(NF, NE);
    w_down = alloc_q4_0(NE, NF);

    cudaMalloc(&norm_attn, NE  * sizeof(float));
    cudaMalloc(&norm_ffn,  NE  * sizeof(float));
    cudaMalloc(&bq_bias,   NQT * sizeof(float));
    cudaMalloc(&bk_bias,   NKT * sizeof(float));
    cudaMalloc(&bv_bias,   NKT * sizeof(float));
    cudaMemset(norm_attn, 0, NE  * sizeof(float));
    cudaMemset(norm_ffn,  0, NE  * sizeof(float));
    cudaMemset(bq_bias,   0, NQT * sizeof(float));
    cudaMemset(bk_bias,   0, NKT * sizeof(float));
    cudaMemset(bv_bias,   0, NKT * sizeof(float));

    // KV cache: allocate generous space (use position 0 only)
    cudaMalloc(&kv_k, 1024 * NKT * sizeof(float));
    cudaMalloc(&kv_v, 1024 * NKT * sizeof(float));
    cudaMemset(kv_k, 0, 1024 * NKT * sizeof(float));
    cudaMemset(kv_v, 0, 1024 * NKT * sizeof(float));
    cudaMalloc(&d_positions, sizeof(int));
    int pos0 = 0;
    cudaMemcpy(d_positions, &pos0, sizeof(int), cudaMemcpyHostToDevice);

    // ─── stream ──────────────────────────────────────────────────
    cudaStream_t s;
    cudaStreamCreate(&s);

    int first_error = 0;
    const char* first_error_name = nullptr;

    for (int mode = 2; mode >= 0; mode--) {
        const char* mode_names[] = {"Global", "ThreadLocal", "Relaxed"};
        printf("\n=== test_03 single-layer mode=%d (%s) ===\n", mode, mode_names[mode]);

        first_error = 0;
        first_error_name = nullptr;

        cudaError_t err = cudaStreamBeginCapture(s, (cudaStreamCaptureMode)mode);
        printf("  begin capture: %d\n", err);
        if (err != 0) { printf("  SKIP\n"); continue; }

        // ─── kernel sequence (mirrors layer_gpu) ─────────────────

        // 01: rms_norm(attn)  — grid(NT), block(32)
        launch_rms_norm_f32(hidden, norm_attn, bn, NE, 1e-5f, NT, s);
        CHECK_KERNEL(1, "rms_norm(attn)");

        // 02: quantize_q8_0
        launch_quantize_q8_0(bn, q8_bn, NE, NT, s);
        CHECK_KERNEL(2, "quantize_q8_0(attn)");

        // 03: matmul WQ
        launch_q4_0_q8_0_matmul(wq, q8_bn, bq, NQT, NE, NT, s);
        CHECK_KERNEL(3, "matmul(WQ)");

        // 04: add_bias bq
        launch_add_bias_f32(bq, bq_bias, NQT, NT, s);
        CHECK_KERNEL(4, "add_bias(bq)");

        // 05: matmul WK
        launch_q4_0_q8_0_matmul(wk, q8_bn, bk, NKT, NE, NT, s);
        CHECK_KERNEL(5, "matmul(WK)");

        // 06: add_bias bk
        launch_add_bias_f32(bk, bk_bias, NKT, NT, s);
        CHECK_KERNEL(6, "add_bias(bk)");

        // 07: matmul WV
        launch_q4_0_q8_0_matmul(wv, q8_bn, bv, NKT, NE, NT, s);
        CHECK_KERNEL(7, "matmul(WV)");

        // 08: add_bias bv
        launch_add_bias_f32(bv, bv_bias, NKT, NT, s);
        CHECK_KERNEL(8, "add_bias(bv)");

        // 09: rope Q — grid(NT, NH), block(64)
        launch_rope_f32(bq, NH, HD, NT, 10000.0f, 1.0f, d_positions, s);
        CHECK_KERNEL(9, "rope(Q)");

        // 10: rope K — grid(NT, NK), block(64)
        launch_rope_f32(bk, NK, HD, NT, 10000.0f, 1.0f, d_positions, s);
        CHECK_KERNEL(10, "rope(K)");

        // 11: store_kv K — grid(NT, NKT), block(1)
        launch_store_kv_f32(bk, kv_k, NKT, NT, d_positions, s);
        CHECK_KERNEL(11, "store_kv(K)");

        // 12: store_kv V
        launch_store_kv_f32(bv, kv_v, NKT, NT, d_positions, s);
        CHECK_KERNEL(12, "store_kv(V)");

        // 13: gqa_attn — grid(NT, NH), block(32)
        float scale = 1.0f / sqrtf((float)HD);
        launch_gqa_attn_f32(bq, kv_k, kv_v, ba, d_positions, NH, NK, HD, scale, NT, s);
        CHECK_KERNEL(13, "gqa_attn");

        // 14: quantize_q8_0 for WO
        launch_quantize_q8_0(ba, q8_ba, NE, NT, s);
        CHECK_KERNEL(14, "quantize_q8_0(wo)");

        // 15: matmul WO
        launch_q4_0_q8_0_matmul(wo, q8_ba, bn, NE, NE, NT, s);
        CHECK_KERNEL(15, "matmul(WO)");

        // 16: add residual (attn)
        launch_add_f32(hidden, bn, hidden, NT * NE, s);
        CHECK_KERNEL(16, "add(residual attn)");

        // 17: rms_norm(ffn)
        launch_rms_norm_f32(hidden, norm_ffn, ba, NE, 1e-5f, NT, s);
        CHECK_KERNEL(17, "rms_norm(ffn)");

        // 18: quantize_q8_0 for FFN
        launch_quantize_q8_0(ba, q8_ba, NE, NT, s);
        CHECK_KERNEL(18, "quantize_q8_0(ffn)");

        // 19: matmul Gate
        launch_q4_0_q8_0_matmul(w_gate, q8_ba, bg, NF, NE, NT, s);
        CHECK_KERNEL(19, "matmul(Gate)");

        // 20: matmul Up
        launch_q4_0_q8_0_matmul(w_up, q8_ba, bf, NF, NE, NT, s);
        CHECK_KERNEL(20, "matmul(Up)");

        // 21: swiglu (silu(gate) * up)
        launch_swiglu_f32(bg, bf, bg, NT * NF, s);
        CHECK_KERNEL(21, "swiglu");

        // 22: quantize_q8_0 for Down
        launch_quantize_q8_0(bg, q8_ba, NF, NT, s);
        CHECK_KERNEL(22, "quantize_q8_0(down)");

        // 23: matmul Down
        launch_q4_0_q8_0_matmul(w_down, q8_ba, bn, NE, NF, NT, s);
        CHECK_KERNEL(23, "matmul(Down)");

        // 24: add residual (ffn)
        launch_add_f32(hidden, bn, hidden, NT * NE, s);
        CHECK_KERNEL(24, "add(residual ffn)");

        // ─── end capture ─────────────────────────────────────────
        if (first_error != 0) {
            printf("\n  *** FIRST capture error at kernel #%d (%s) ***\n", first_error, first_error_name);
            cudaStreamEndCapture(s, nullptr);  // abort capture
            cudaGetLastError();                // consume stale error
            continue;
        }

        cudaGraph_t g = nullptr;
        err = cudaStreamEndCapture(s, &g);
        printf("  end capture: err=%d graph=%p\n", err, (void*)g);
        if (err != 0 || g == nullptr) {
            printf("  FAIL (end capture returned error)\n");
            cudaGetLastError();
            continue;
        }

        // ─── instantiate ─────────────────────────────────────────
        cudaGraphExec_t ge = nullptr;
        err = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        printf("  instantiate: %d\n", err);
        if (err != 0) {
            printf("  FAIL (instantiate)\n");
            cudaGraphDestroy(g);
            continue;
        }

        // ─── launch ──────────────────────────────────────────────
        err = cudaGraphLaunch(ge, s);
        printf("  launch: %d\n", err);
        err = cudaStreamSynchronize(s);
        printf("  sync: %d\n", err);
        if (err == 0) {
            printf("  PASS — single layer graph capture OK\n");
        } else {
            printf("  FAIL — sync error after graph launch\n");
        }

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }

    // ─── cleanup ─────────────────────────────────────────────────
    cudaStreamDestroy(s);
    cudaFree(hidden); cudaFree(bn);  cudaFree(bq);  cudaFree(bk);
    cudaFree(bv);     cudaFree(ba);  cudaFree(bf);  cudaFree(bg);
    cudaFree(q8_bn);  cudaFree(q8_ba);
    cudaFree(wq);  cudaFree(wk);  cudaFree(wv);  cudaFree(wo);
    cudaFree(w_gate); cudaFree(w_up); cudaFree(w_down);
    cudaFree(norm_attn); cudaFree(norm_ffn);
    cudaFree(bq_bias); cudaFree(bk_bias); cudaFree(bv_bias);
    cudaFree(kv_k); cudaFree(kv_v); cudaFree(d_positions);

    return 0;
}
