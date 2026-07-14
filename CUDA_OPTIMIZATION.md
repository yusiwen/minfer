# CUDA Inference Path — Optimization Roadmap

Current performance: Prefill 40 tok/s, Decode 20 tok/s (RTX 4080 Laptop GPU, Qwen2-0.5B Q4_0)

Comparison: CPU baseline = Prefill 18 tok/s, Decode 15 tok/s → ~2.2x prefill, ~1.3x decode

Llama.cpp typically achieves 5-10x over CPU on the same hardware for 0.5B models.

## Root Cause: Per-Op CPU↔GPU Ping Pong

The current implementation dispatches from `kernel.rs` at the **individual matmul** level:

```
CPU path → quantize f32→Q8_0 (CPU) → cudaMemcpy H2D → CUDA kernel → sync → cudaMemcpy D2H → CPU path
```

For a single decode token, each layer triggers:
- Q/K/V projection (3 matmuls in batch)
- WO projection (1 matmul)
- Gate/Up projection (2 matmuls in batch)
- Down projection (1 matmul)

That's ~6 PCIe round trips × 24 layers = **144 DMA operations per decode step**. Each sync+memcpy adds 10-50µs latency, totaling 2-7ms of pure overhead. The actual GPU compute for a 1-token matmul is only ~0.1ms.

## Optimization Plan

### P0: Enable Full-Layer GPU Offload (High Priority)

The Qwen2-0.5B model has **mixed quantization types**: most weights are Q4_0, but `ffn_down.weight` varies per layer (some Q4_0, some Q4_1) and norm/embedding weights are F32/Q8_0.

**Required**: Add Q4_1 × f32 and Q8_0 × f32 matmul kernels to `cuda_kernels.cu`, and update `CudaState::layer_gpu()` to handle these types.

**Net effect**: Eliminates all 144 per-step DMA transfers → single `upload_hidden` + `download_logits`.
**Estimated speedup**: 3-4x (decode → 60-80 tok/s)

### P1: GPU-Side Activation Quantization

Currently, `f32 → Q8_0` quantization runs on CPU (`avx2::quantize_row_q8_0_buf`). The CUDA `quantize_q8_0` kernel already exists but is unused in `layer_gpu()`.

**Required**: Remove CPU quantize call in `layer_gpu()`; the GPU kernel runs on the f32 buffer directly.
**Estimated speedup**: 1.2x

### P2: Fused Grouped-Query Attention on GPU

The `gqa_attn_f32` CUDA kernel already exists. Currently, attention runs on CPU via `vec_dot_f32` loops (`forward.rs:gqa_attn()`).

**Required**: Wire `gqa_attn_f32` into `layer_gpu()` instead of CPU fallback.
**Estimated speedup**: 1.5x (especially for long-context decode)

### P3: cuBLAS for Large Output Projection

The output projection (`output.weight` / `token_embd.weight`) is a large matrix: `[151936 × 896]` (Q8_0). A dense cuBLAS `cublasSgemm` would leverage tensor cores.

**Required**: Link against `cublas` and dispatch large matmuls to cuBLAS when src is f32.
**Estimated speedup**: 2x (for output projection specifically)

### P4: Quantized MatMul with Shared Memory Tiling

The current Q4_0×Q8_0 kernel uses a simple warp-cooperative reduction (`NR0=4` rows per warp). Llama.cpp's MMQ uses shared-memory tiling with Stream-K decomposition, which improves occupancy and memory bandwidth utilization.

**Required**: Rewrite matmul kernel to tile Q4_0 weights into shared memory, using `__ldg()` for activation reads.
**Estimated speedup**: 1.5x

### P5: CUDA Graph for Kernel Launch Overhead

Each kernel launch has ~5-10µs overhead. For 2,000+ kernel launches per decode step (24 layers × ~6 ops × ~14 heads), this adds up.

**Required**: After warmup, capture the entire decode graph with `cudaGraphInstantiate`/`cudaGraphLaunch`.
**Estimated speedup**: 1.2x

## Implementation Order

```
P0 (layer_gpu) ─→ P1 (GPU quantize) ─→ P2 (GPU attention)
                                     ↘
                                      P3 (cuBLAS) ─→ P4 (tiled MMQ) ─→ P5 (CUDA Graph)
```

P0 alone eliminates the dominant bottleneck and should bring performance to within 2-3x of llama.cpp.
