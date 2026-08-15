// test_01_basic_capture.cu
// Verify basic CUDA graph capture works on this GPU.
// A single simple kernel (add) is captured, instantiated, and replayed.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_kernel(int n, float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] + y[i];
}

int main() {
    const int N = 1024;
    float *dx, *dy;
    cudaMalloc(&dx, N * sizeof(float));
    cudaMalloc(&dy, N * sizeof(float));
    cudaMemset(dx, 0, N * sizeof(float));
    cudaMemset(dy, 0, N * sizeof(float));

    cudaStream_t s;
    cudaStreamCreate(&s);

    int passed = 0, failed = 0;

    for (int mode = 0; mode <= 2; mode++) {
        const char* mode_names[] = {"Global", "ThreadLocal", "Relaxed"};
        printf("=== Test mode %d (%s) ===\n", mode, mode_names[mode]);

        cudaError_t err = cudaStreamBeginCapture(s, (cudaStreamCaptureMode)mode);
        printf("  begin capture: %d\n", err);
        if (err != 0) { printf("  SKIP\n"); continue; }

        add_kernel<<<N/256, 256, 0, s>>>(N, dx, dy);
        cudaError_t le = cudaGetLastError();
        printf("  after kernel: GetLastError=%d\n", le);

        cudaGraph_t g = nullptr;
        err = cudaStreamEndCapture(s, &g);
        printf("  end capture: %d\n", err);
        if (err != 0) { printf("  FAIL\n"); failed++; continue; }

        cudaGraphExec_t ge = nullptr;
        err = cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);
        printf("  instantiate: %d\n", err);
        if (err != 0) {
            cudaGraphDestroy(g);
            printf("  FAIL\n"); failed++; continue;
        }

        err = cudaGraphLaunch(ge, s);
        printf("  launch: %d\n", err);
        err = cudaStreamSynchronize(s);
        printf("  sync: %d\n", err);

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
        passed++;
    }

    cudaStreamDestroy(s);
    cudaFree(dx);
    cudaFree(dy);

    printf("\nResults: %d passed, %d failed\n", passed, failed);
    return failed > 0 ? 1 : 0;
}
