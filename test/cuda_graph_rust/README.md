# Rust FFI Integration Tests for CUDA Graph

Tests to isolate why minfer's Rust → FFI → CUDA graph capture fails
while equivalent C++ standalone tests pass.

## Files

| # | File | Purpose | Run |
|---|------|---------|-----|
| 06 | `examples/cuda_graph_diag.rs` | Minimal Rust → FFI → CUDA graph capture (no model) | `cargo run --example cuda_graph_diag` |
| 07 | (planned) | Stream state verification after prefill | — |
| 08 | (planned) | Buffer growth (`get_or_grow`) during capture | — |

## Usage

```bash
# Build and run the diagnostic tool
cargo run --example cuda_graph_diag

# With debug output
MINFER_CUDA_DEBUG=1 cargo run --example cuda_graph_diag

# Run only on specific GPU
CUDA_VISIBLE_DEVICES=1 cargo run --example cuda_graph_diag
```

## Linking

The example links against the same libraries as the main minfer binary:
- `cuda_kernels.a` (built by `build.rs` from `src/cuda_kernels.cu`)
- `libcudart.so` (CUDA runtime)
- `libstdc++.so` (C++ standard library, required by `cuda_kernels.cu`)
