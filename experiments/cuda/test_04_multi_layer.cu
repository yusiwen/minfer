// test_04_multi_layer.cu
// Test CUDA graph capture with N layers (argv[1]).
// Binary-search to find how many layers the graph can handle before capture fails.
//
// Usage: ./test_04 [N]
//   N = number of layers to capture (default: 2)

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define Q4B 18
#define Q8B 34

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

static uint8_t* alloc_q4_0(int out_dim, int in_dim) {
    int blocks = out_dim * (in_dim / 32);
    size_t sz = (size_t)blocks * Q4B;
    uint8_t* buf;
    cudaMalloc(&buf, sz);
    cudaMemset(buf, 0, sz);
    return buf;
}

static void run_one_layer(cudaStream_t s, int layer, int* step,
    float* hidden, float* bn, float* bq, float* bk, float* bv, float* ba,
    float* bf, float* bg, uint8_t* q8_bn, uint8_t* q8_ba,
    uint8_t* wq, uint8_t* wk, uint8_t* wv, uint8_t* wo,
    uint8_t* w_gate, uint8_t* w_up, uint8_t* w_down,
    float* norm_attn, float* norm_ffn,
    float* bbq, float* bbk, float* bbv,
    float* kv_k, float* kv_v, int* dpos,
    int NE, int NQT, int NKT, int NF, int NH, int NK, int HD, int NT,
    int* first_error, const char** first_err_name)
{
    auto chk = [&](const char* name) {
        cudaError_t e = cudaGetLastError();
        (*step)++;
        printf("  L%d [%02d] %-24s err=%d\n", layer, *step, name, e);
        if (e != 0 && *first_error == 0) { *first_error = *step; *first_err_name = name; }
    };

    float scale = 1.0f / sqrtf((float)HD);

    launch_rms_norm_f32(hidden, norm_attn, bn, NE, 1e-5f, NT, s);       chk("rms_norm(attn)");
    launch_quantize_q8_0(bn, q8_bn, NE, NT, s);                         chk("quantize_q8_0(attn)");
    launch_q4_0_q8_0_matmul(wq, q8_bn, bq, NQT, NE, NT, s);             chk("matmul(WQ)");
    launch_add_bias_f32(bq, bbq, NQT, NT, s);                            chk("add_bias(bq)");
    launch_q4_0_q8_0_matmul(wk, q8_bn, bk, NKT, NE, NT, s);             chk("matmul(WK)");
    launch_add_bias_f32(bk, bbk, NKT, NT, s);                            chk("add_bias(bk)");
    launch_q4_0_q8_0_matmul(wv, q8_bn, bv, NKT, NE, NT, s);             chk("matmul(WV)");
    launch_add_bias_f32(bv, bbv, NKT, NT, s);                            chk("add_bias(bv)");
    launch_rope_f32(bq, NH, HD, NT, 10000.0f, 1.0f, dpos, s);          chk("rope(Q)");
    launch_rope_f32(bk, NK, HD, NT, 10000.0f, 1.0f, dpos, s);          chk("rope(K)");
    launch_store_kv_f32(bk, kv_k, NKT, NT, dpos, s);                    chk("store_kv(K)");
    launch_store_kv_f32(bv, kv_v, NKT, NT, dpos, s);                    chk("store_kv(V)");
    launch_gqa_attn_f32(bq, kv_k, kv_v, ba, dpos, NH, NK, HD, scale, NT, s); chk("gqa_attn");
    launch_quantize_q8_0(ba, q8_ba, NE, NT, s);                         chk("quantize_q8_0(wo)");
    launch_q4_0_q8_0_matmul(wo, q8_ba, bn, NE, NE, NT, s);              chk("matmul(WO)");
    launch_add_f32(hidden, bn, hidden, NT * NE, s);                     chk("add(residual attn)");
    launch_rms_norm_f32(hidden, norm_ffn, ba, NE, 1e-5f, NT, s);       chk("rms_norm(ffn)");
    launch_quantize_q8_0(ba, q8_ba, NE, NT, s);                         chk("quantize_q8_0(ffn)");
    launch_q4_0_q8_0_matmul(w_gate, q8_ba, bg, NF, NE, NT, s);          chk("matmul(Gate)");
    launch_q4_0_q8_0_matmul(w_up, q8_ba, bf, NF, NE, NT, s);            chk("matmul(Up)");
    launch_swiglu_f32(bg, bf, bg, NT * NF, s);                          chk("swiglu");
    launch_quantize_q8_0(bg, q8_ba, NF, NT, s);                         chk("quantize_q8_0(down)");
    launch_q4_0_q8_0_matmul(w_down, q8_ba, bn, NE, NF, NT, s);          chk("matmul(Down)");
    launch_add_f32(hidden, bn, hidden, NT * NE, s);                     chk("add(residual ffn)");
}

int main(int argc, char** argv) {
    int N_LAYERS = argc > 1 ? atoi(argv[1]) : 2;
    printf("=== test_04 multi-layer (N=%d layers, %d kernels) ===\n", N_LAYERS, N_LAYERS * 24);

    const int NE = 896, NQT = 896, NKT = 128, NF = 4864;
    const int NH = 14, NK = 2, HD = 64, NT = 1;

    // Shared buffers (reused across layers)
    float *hidden, *bn, *bq, *bk, *bv, *ba, *bf, *bg;
    uint8_t *q8_bn, *q8_ba;
    cudaMalloc(&hidden, NT * NE * sizeof(float));
    cudaMalloc(&bn,     NT * NE * sizeof(float));
    cudaMalloc(&bq,     NT * NQT * sizeof(float));
    cudaMalloc(&bk,     NT * NKT * sizeof(float));
    cudaMalloc(&bv,     NT * NKT * sizeof(float));
    cudaMalloc(&ba,     NT * NE * sizeof(float));
    cudaMalloc(&bf,     NT * NF * sizeof(float));
    cudaMalloc(&bg,     NT * NF * sizeof(float));
    cudaMalloc(&q8_bn,  NT * (NE / 32) * Q8B);
    cudaMalloc(&q8_ba,  NT * (NF / 32) * Q8B);

    // Shared weights (reused across layers)
    uint8_t *wq, *wk, *wv, *wo, *w_gate, *w_up, *w_down;
    wq     = alloc_q4_0(NQT, NE);
    wk     = alloc_q4_0(NKT, NE);
    wv     = alloc_q4_0(NKT, NE);
    wo     = alloc_q4_0(NE, NE);
    w_gate = alloc_q4_0(NF, NE);
    w_up   = alloc_q4_0(NF, NE);
    w_down = alloc_q4_0(NE, NF);

    float *norm_attn, *norm_ffn, *bbq, *bbk, *bbv;
    cudaMalloc(&norm_attn, NE * sizeof(float));
    cudaMalloc(&norm_ffn,  NE * sizeof(float));
    cudaMalloc(&bbq, NQT * sizeof(float));
    cudaMalloc(&bbk, NKT * sizeof(float));
    cudaMalloc(&bbv, NKT * sizeof(float));
    cudaMemset(norm_attn, 0, NE * sizeof(float));
    cudaMemset(norm_ffn,  0, NE * sizeof(float));
    cudaMemset(bbq, 0, NQT * sizeof(float));
    cudaMemset(bbk, 0, NKT * sizeof(float));
    cudaMemset(bbv, 0, NKT * sizeof(float));

    // Per-layer KV caches
    float** kv_k_arr = new float*[N_LAYERS];
    float** kv_v_arr = new float*[N_LAYERS];
    for (int i = 0; i < N_LAYERS; i++) {
        cudaMalloc(&kv_k_arr[i], 1024 * NKT * sizeof(float));
        cudaMalloc(&kv_v_arr[i], 1024 * NKT * sizeof(float));
        cudaMemset(kv_k_arr[i], 0, 1024 * NKT * sizeof(float));
        cudaMemset(kv_v_arr[i], 0, 1024 * NKT * sizeof(float));
    }

    int* dpos;
    cudaMalloc(&dpos, sizeof(int));
    int p0 = 0;
    cudaMemcpy(dpos, &p0, sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t s;
    cudaStreamCreate(&s);

    // Test modes: try Relaxed (2) first, then ThreadLocal (1)
    int modes[] = {2, 1, 0};
    for (int m = 0; m < 3; m++) {
        int mode = modes[m];
        const char* mn[] = {"Global", "ThreadLocal", "Relaxed"};
        printf("\n--- mode=%d (%s) ---\n", mode, mn[mode]);

        int step = 0, first_error = 0;
        const char* first_err_name = nullptr;

        cudaError_t err = cudaStreamBeginCapture(s, (cudaStreamCaptureMode)mode);
        printf("  begin capture: %d\n", err);
        if (err != 0) { printf("  SKIP\n"); continue; }

        for (int il = 0; il < N_LAYERS; il++) {
            run_one_layer(s, il, &step,
                hidden, bn, bq, bk, bv, ba, bf, bg, q8_bn, q8_ba,
                wq, wk, wv, wo, w_gate, w_up, w_down,
                norm_attn, norm_ffn, bbq, bbk, bbv,
                kv_k_arr[il], kv_v_arr[il], dpos,
                NE, NQT, NKT, NF, NH, NK, HD, NT,
                &first_error, &first_err_name);
            if (first_error) break;
        }

        if (first_error) {
            printf("\n  *** FIRST capture error at step %d (%s) in layer %d ***\n",
                   first_error, first_err_name, (first_error - 1) / 24);
            cudaStreamEndCapture(s, nullptr);
            cudaGetLastError();
            continue;
        }

        cudaGraph_t g = nullptr;
        err = cudaStreamEndCapture(s, &g);
        printf("  end capture: err=%d\n", err);
        if (err != 0 || g == nullptr) {
            printf("  FAIL (end capture)\n");
            cudaGetLastError();
            continue;
        }

        cudaGraphExec_t ge = nullptr;
        err = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        printf("  instantiate: %d\n", err);
        if (err != 0) { printf("  FAIL\n"); cudaGraphDestroy(g); continue; }

        err = cudaGraphLaunch(ge, s);
        printf("  launch: %d\n", err);
        err = cudaStreamSynchronize(s);
        printf("  sync: %d\n", err);
        printf("  %s\n", err == 0 ? "PASS" : "FAIL");

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }

    // Cleanup
    cudaStreamDestroy(s);
    cudaFree(hidden); cudaFree(bn); cudaFree(bq); cudaFree(bk);
    cudaFree(bv); cudaFree(ba); cudaFree(bf); cudaFree(bg);
    cudaFree(q8_bn); cudaFree(q8_ba);
    cudaFree(wq); cudaFree(wk); cudaFree(wv); cudaFree(wo);
    cudaFree(w_gate); cudaFree(w_up); cudaFree(w_down);
    cudaFree(norm_attn); cudaFree(norm_ffn);
    cudaFree(bbq); cudaFree(bbk); cudaFree(bbv);
    for (int i = 0; i < N_LAYERS; i++) { cudaFree(kv_k_arr[i]); cudaFree(kv_v_arr[i]); }
    delete[] kv_k_arr; delete[] kv_v_arr;
    cudaFree(dpos);
    return 0;
}
