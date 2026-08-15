// test_02_rms_norm_single.cu
// Verify rms_norm_f32 kernel (from cuda_kernels.cu) can be graph-captured.
// Uses the launch wrapper from cuda_kernels.cu, not a local definition.

#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

// Declare launch wrapper from cuda_kernels.cu (linked at compile time)
extern "C" void launch_rms_norm_f32(
    const float* x, const float* w, float* y,
    int d, float eps, int n, cudaStream_t stream);

int main() {
    const int NE = 896, NT = 1;
    float *dx, *dw, *dy;
    cudaMalloc(&dx, NT * NE * sizeof(float));
    cudaMalloc(&dw, NE * sizeof(float));
    cudaMalloc(&dy, NT * NE * sizeof(float));
    cudaMemset(dx, 0, NT * NE * sizeof(float));
    cudaMemset(dw, 0, NE * sizeof(float));
    cudaMemset(dy, 0, NT * NE * sizeof(float));

    cudaStream_t s;
    cudaStreamCreate(&s);

    int passed = 0, failed = 0;
    for (int mode = 0; mode <= 2; mode++) {
        const char* mode_names[] = {"Global", "ThreadLocal", "Relaxed"};
        printf("=== test_02 rms_norm mode=%d (%s) ===\n", mode, mode_names[mode]);

        cudaError_t err = cudaStreamBeginCapture(s, (cudaStreamCaptureMode)mode);
        printf("  begin capture: %d\n", err);
        if (err != 0) { printf("  SKIP\n"); continue; }

        launch_rms_norm_f32(dx, dw, dy, NE, 1e-5f, NT, s);
        cudaError_t le = cudaGetLastError();
        printf("  after kernel: GetLastError=%d\n", le);
        if (le != 0) { printf("  FAIL\n"); failed++; cudaStreamEndCapture(s, nullptr); continue; }

        cudaGraph_t g = nullptr;
        err = cudaStreamEndCapture(s, &g);
        printf("  end capture: err=%d\n", err);
        if (err != 0 || g == nullptr) { printf("  FAIL\n"); failed++; continue; }

        cudaGraphExec_t ge = nullptr;
        err = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        printf("  instantiate: %d\n", err);
        if (err != 0) { printf("  FAIL\n"); cudaGraphDestroy(g); failed++; continue; }

        err = cudaGraphLaunch(ge, s);
        printf("  launch: %d\n", err);
        err = cudaStreamSynchronize(s);
        printf("  sync: %d\n", err);
        if (err == 0) { printf("  PASS\n"); passed++; } else { printf("  FAIL\n"); failed++; }

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }

    cudaStreamDestroy(s);
    cudaFree(dx); cudaFree(dw); cudaFree(dy);

    printf("\ntest_02: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
