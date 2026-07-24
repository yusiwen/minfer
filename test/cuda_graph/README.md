# CUDA Graph Capture Debugging Test Suite

Isolate why `cudaStreamBeginCapture` → `cudaStreamEndCapture` → `cudaGraphInstantiate`
causes stream corruption (error 900/901) in the full 24-layer decode loop.

## Test Files

| # | File | Purpose | Compile | Result |
|---|------|---------|----------|--------|
| 01 | `test_01_basic_capture.cu` | Verify RTX 2080 Ti supports CUDA graph for a simple `add` kernel | `nvcc -o /tmp/t01 test_01_basic_capture.cu -arch=sm_75 && /tmp/t01` | ✅ PASS (3/3 modes) |
| 02 | `test_02_rms_norm_single.cu` | Verify `rms_norm_f32` kernel alone is graph-capture-safe | `nvcc -o /tmp/t02 test_02_rms_norm_single.cu ../../src/cuda_kernels.cu -I/usr/local/cuda/include -arch=sm_75 && /tmp/t02` | ✅ PASS (3/3 modes) |
| 03 | `test_03_single_layer.cu` | Capture **single** layer_gpu (24 kernel launches) | `nvcc -o /tmp/t03 test_03_single_layer.cu ../../src/cuda_kernels.cu -I/usr/local/cuda/include -arch=sm_75 && /tmp/t03` | ✅ PASS (3/3 modes) |
| 04 | `test_04_multi_layer.cu` | Capture **N layers** (argv[1]), binary-search threshold | `nvcc -o /tmp/t04 test_04_multi_layer.cu ../../src/cuda_kernels.cu -I/usr/local/cuda/include -arch=sm_75 && /tmp/t04 24` | ✅ PASS up to 24 layers (576 kernels, 3/3 modes) |
| 05 | `test_05_with_output_projection.cu` | Capture **N layers + output projection** (151936×896 matmul, 73 MB weight) | `nvcc -o /tmp/t05 test_05_with_output_projection.cu ../../src/cuda_kernels.cu -I/usr/local/cuda/include -arch=sm_75 && /tmp/t05 24` | ✅ PASS (Relaxed mode) |

## Conclusion

**All C++ standalone tests pass.** The CUDA graph capture infrastructure (kernels, launch wrappers,
capture API calls) works correctly for the complete 24-layer + output-projection pipeline.
The failure (error 901/900 in minfer) is **not caused by any kernel or kernel combination**.

### Root Cause Hypothesis

The problem must be in the **Rust → CUDA integration layer**, specifically one of:

1. **Stale CUDA error state** from the prefill step (cudaGetLastError not consumed before
   cudaStreamBeginCapture)
2. **FFI parameter marshaling** — Rust passes kernel arguments differently from C++ (e.g.,
   incorrect pointer types, alignment issues that only manifest during graph instantiation)
3. **Stream state corruption** — the stream carries residual state from a previous forward
   pass that conflicts with capture mode
4. **`get_or_grow` / `cudaMalloc` during capture** — if any buffer resize triggers
   `cudaMalloc` while the stream is in capture mode

### Next Step

Build a minimal **Rust integration test** that replicates forward.rs's capture flow but
with hardcoded layers (no model loading), to confirm the issue is in the FFI path.

## Environment

```
GPU:     NVIDIA GeForce RTX 2080 Ti (SM 7.5, Turing)
Driver:  13.0 (from nvidia-smi)
Toolkit: 12.0 (nvcc)
```

## Reference

- minfer CUDA kernels: `../../src/cuda_kernels.cu`
- minfer CUDA backend: `../../src/cuda.rs`
- minfer forward pass: `../../src/models/qwen2/forward.rs`
