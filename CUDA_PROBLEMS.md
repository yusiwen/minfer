# CUDA 推理路径问题分析

## 修复日志

| 日期 | 问题 | 修复 |
|------|------|------|
| 2026-07-21 | #1 P0 严重 | `use_gpu` 检查所有层 Q4_0；`layer_gpu` 失败时回退 CPU 而不是 `return vec![]` |
| 2026-07-21 | #2 P0 严重 | 添加 `[features] cuda = []` 到 Cargo.toml |
| 2026-07-21 | #3 P1 | 新增 `q8_0_f32_matmul` CUDA kernel；`output_norm_gpu` 支持 Q8_0 输出权重；`matmul_on_gpu`/`quant_matmul_f32` 支持 Q8_0 |
| 2026-07-21 | #7 P2 性能 | CUDA Graph：首次 decode 步骤录制 graph，后续 replay；单次 `cudaGraphLaunch` 替代 ~360 次 kernel 启动 |
| 2026-07-21 | #4 P1 重要 | `init_kv_cache()` 预分配 KV cache 到 `n_ctx`，消除 O(n²) 增长 |
| 2026-07-21 | #5 P1 重要 | `sync()` 添加 `cudaStreamSynchronize` 错误检查；`layer_gpu` 结尾添加 `cuda_kernel_check` |
| 2026-07-21 | #6 P2 低危 | `rms_norm` 无权重时使用 `expect` 而不是读输出缓冲区 UB |

## 仍待修复

*所有已知 CUDA 推理问题已修复。*

---

## 一、CUDA 推理路径总览

minfer 的 CUDA 推理有**两条路径**，选择逻辑在 `src/models/qwen2/forward.rs` 中：

```
forward() 入口
  │
  ├─ embed_tokens() (CPU, f32)
  │
  ├─ [use_gpu 检查] ── 检查 layer 0 所有权重是否 Q4_0 且在 GPU 上
  │     │
  │     ├── true  → 全层 GPU 卸载路径 (layer_gpu)
  │     │   ├─ upload_hidden()    上传 hidden → GPU
  │     │   ├─ upload_positions() 上传位置 → GPU
  │     │   ├─ for each layer: cuda.layer_gpu()  ── 全层 GPU (零拷贝)
  │     │   ├─ cuda.output_norm_gpu()            ── 输出投影 (GPU)
  │     │   └─ sync() + download_logits() ── 下载 logits
  │     │
  │     └── false → CPU + 按算子 GPU 派发路径 (kernel.rs dispatch)
  │         ├─ for each layer:
  │         │   ├─ RMSNorm (CPU, AVX2)
  │         │   ├─ quant_matmul_f32_batch() → CPU量化→上传→GPU matmul→sync→下载
  │         │   │    (每个 matmul 一輪 PCIe 往返！每层 ~6 次往返)
  │         │   ├─ RoPE (CPU)
  │         │   ├─ GQA attention (CPU)
  │         │   └─ SwiGLU (CPU)
  │         └─ RMSNorm + output matmul (CPU/GPU dispatch)
```

### 与 Metal 路径的关键差异

| 方面 | Metal | CUDA |
|------|-------|------|
| use_gpu 检查 | 不检查量化类型 | 要求 layer 0 全 Q4_0 |
| layer_gpu 量化支持 | Q4_0/Q4_1/Q4_K/Q6_K | 仅 Q4_0 |
| output_norm_gpu 量化支持 | Q4_0/Q4_1/Q8_0/Q4_K/Q6_K | 仅 Q4_0 |
| 命令缓冲 | MpsCommandBuffer 批量提交 | 每个 kernel 直接 `<<<>>>` 启动 |
| KV cache 增长 | 使用共享内存 ptr copy | 同步 cudaMemcpy |
| 错误检查 | Metal API 返回 NSError | 无检查 |

---

## 二、与 llama.cpp 的逻辑差异

### 1. 计算图 vs 直接执行

| | minfer | llama.cpp |
|---|---|---|
| 执行模型 | 每步直接调用 kernel | ggml 计算图 DAG → 后端调度 |
| 中间数据位置 | 在 GPU 缓冲区上（全层路径） | 图分配器管理，全 GPU |
| kernel 融合 | 无 | RMSNorm+Mul, 多个 ADD 融合 |
| MatMul 调度 | 固定 Q4_0×Q8_0 | mmvq/mmq/mmvf/mmf 按 batch size 自动选择 |

### 2. KV Cache 管理

| | minfer | llama.cpp |
|---|---|---|
| 分配策略 | 每次增量增长 1 槽位 | 初始化预分配 n_ctx 大小 |
| 增长成本 | O(n) 同步 cudaMemcpy 每次 | 无增长（一次性分配） |
| GPU 侧持久化 | ✅ 是（全层路径） | ✅ 是 |
| 非连续位置支持 | ❌ 假设连续（gqa_attn 读取），store_kv 散列写 | ✅ ggml_set_rows + causal mask |

### 3. 后端抽象

| | minfer | llama.cpp |
|---|---|---|
| 调度层 | `kernel.rs` 运行时分发到 Metal/CUDA/CPU | `ggml_backend_sched` 图拆分 + 跨后端拷贝 |
| 权重加载 | 加载时注册到 GPU backend | `select_weight_buft` 按 device 能力自动分配 |
| 多后端并存 | 不支持（Metal 或 CUDA，二选一） | 支持（CPU+CUDA 混合，auto-split） |

### 4. 量化类型支持

| | minfer CUDA | llama.cpp |
|---|---|---|
| layer_gpu 路径 | Q4_0 仅此 | 30+ 类型 |
| kernel.rs 派发路径 | Q4_0/Q4_1/Q4_K/Q6_K/Q8_0（matmul 部分用 GPU，其他用 CPU） | 全 GPU |

### 5. Kernel 启动与错误处理

| | minfer | llama.cpp |
|---|---|---|
| 错误检查 | 无 | `CUDA_CHECK` 宏覆盖所有 kernel 启动和 API 调用 |
| 批量提交 | 无（每 kernel 单独启动） | CUDA Graph 录制 + 回放 |
| Stream 管理 | 单 stream | 多 stream（并发区域 fork/join） |

---

## 三、已发现的问题

### 问题 1（P0 严重）：`layer_gpu()` false → `return vec![]` 中断推理

**位置**: `src/models/qwen2/forward.rs:119` + `src/cuda.rs:652-656`

**根因**: CUDA 的 `layer_gpu()` 仅支持 Q4_0 权重的全层 GPU 计算。当模型中存在 Q4_1/Q4_K/Q6_K 权重的层时（即使是混合量化模型），`layer_gpu()` 返回 false，forward 函数直接返回空 Vec：

```rust
// forward.rs:119
if !cuda.layer_gpu(il, l, ...) {
    eprintln!("layer_gpu returned false at layer {}", il);
    cuda.sync();
    return vec![];  // ← 返回空向量，推理中断
}
```

```rust
// cuda.rs:652-656
let all_q4_0 = wq.ttype == TensorType::Q4_0 && wk.ttype == TensorType::Q4_0
    && wv.ttype == TensorType::Q4_0 && wo.ttype == TensorType::Q4_0
    && ffn_gate.ttype == TensorType::Q4_0 && ffn_up.ttype == TensorType::Q4_0
    && ffn_down.ttype == TensorType::Q4_0;
if !all_q4_0 { return false; }
```

**对比 Metal**: Metal 的 `layer_gpu()` 支持 Q4_0/Q4_1 和 Q4_K/Q6_K 的分组处理（同组内不混用），并且有 `attn_all_q4/attn_any_q4k` 等条件判断来选择不同的 matmul 内核。

**后果**: 混合量化模型（如某些 Qwen2-0.5B 变体混合了 Q4_0 和 Q4_1）的推理完全空白，无输出。

**修复方案**:
- 当 `layer_gpu()` 返回 false 时，不要返回 `vec![]`，而是：
  1. `cuda.download_hidden(&mut hidden)` — 下载当前 GPU hidden 状态到 CPU
  2. 将对应层的计算切换到 CPU 路径 (`run_cpu = true`)，或使用 `kernel.rs` 派发
  3. 继续后续层的 GPU 路径（如果能重新上传 hidden）

---

### 问题 2（P0 严重）：Cargo.toml 缺少 `[features]` 定义

**位置**: `Cargo.toml` 全文件

**根因**: 代码中多处使用了 `#[cfg(feature = "cuda")]`（`cuda.rs`, `cuda_kernels.cu`, `kernel.rs`, `forward.rs`, `loader.rs`），但 `Cargo.toml` 没有声明此 feature：

```toml
# 当前 Cargo.toml — 没有 [features] 段
[dependencies]
rand = "0.8"
...
```

**后果**: 无法使用 `cargo build --features cuda` 编译 CUDA 支持。所有 `#[cfg(feature = "cuda")]` 代码被完全排除，GPU 推理路径不可用。

**修复方案**: 在 `Cargo.toml` 中添加：
```toml
[features]
cuda = []
```

---

### 问题 3（P1 重要）：`output_norm_gpu` 仅支持 Q4_0 输出权重

**位置**: `src/cuda.rs:749`

```rust
if output.ttype != TensorType::Q4_0 { return false; }
```

**对比 Metal**: Metal 支持 Q4_0/Q4_1/Q8_0/Q4_K/Q6_K：
```rust
if output.ttype != TensorType::Q4_0 && output.ttype != TensorType::Q4_1
    && output.ttype != TensorType::Q8_0
    && output.ttype != TensorType::Q4_K && output.ttype != TensorType::Q6_K {
    return false;
}
```

**后果**: 如果模型输出权重是 Q8_0 或其他类型，`output_norm_gpu` 返回 false，fallback 路径依赖正确下载 hidden 到 CPU 重新计算 RMSNorm + matmul。但 `download_hidden` 读取的是 `buf_hidden`，此缓冲区在 `output_norm_gpu` 内部被部分 RMSNorm 写入过（即使函数返回 false），可能导致数据不一致。

**修复方案**: 添加 Q4_K/Q8_0 的 f32 matmul kernel（`q4_k_f32_matmul`, `q8_0_f32_matmul`），并扩展 `output_norm_gpu` 的条件判断。

---

### 问题 4（P1 重要）：KV Cache O(n²) 增量增长 + 同步 cudaMemcpy

**位置**: `src/cuda.rs:397-424`

```rust
fn kv_ensure_layer(&self, il: usize, max_nkv: usize, nkt: usize) {
    let need = max_nkv * nkt * 4;      // 每步增长 1
    let old_size = szvec[il] * nkt * 4; // 上一次的大小
    if old_size >= need { return; }
    // 每次增长都做同步 cudaMemcpy 复制全部旧数据
    self.copy_device_to_device(kvec[il].0, new_k, old_size);
    self.copy_device_to_device(vvec[il].0, new_v, old_size);
}
```

**对比 llama.cpp**: KV cache 在初始化时一次性预分配 `n_ctx * dim` 大小，从不增长，零复制开销。

**后果**: 生成 n 个 token 的总复制量 ≈ `L × n² × dim × 4 × 0.5`（L=24 层, dim=128）。500 token 约 1.5 GB 的同步 `cudaMemcpy`，严重拖慢解码速度。`copy_device_to_device` 是同步的 `cudaMemcpy`（非 `cudaMemcpyAsync`），每次都阻塞 CPU。

**修复方案**: 
- 在 `CudaState::try_new()` 或首次使用时预分配 `n_ctx` 大小的 KV cache。
- 或者至少批量增长（每次 ×2 而不是 +1），并在增长时使用 `cudaMemcpyAsync`。

---

### 问题 5（P1 重要）：缺少 CUDA Kernel 错误检查

**位置**: `src/cuda_kernels.cu` 所有 launch wrapper

**根因**: 所有 kernel launch `<<<grid, block, 0, stream>>>` 后都没有错误检查：

```c
// cuda_kernels.cu — 典型模式，没有错误
q4_0_q8_0_matmul<<<grid, block, 0, stream>>>(weights, acts, output, od, id, nt);
```

**对比 llama.cpp**: 每个 kernel launch 后都有 `CUDA_CHECK(cudaGetLastError())`，关键同步点有 `CUDA_CHECK(cudaStreamSynchronize(stream))`。

**后果**: 如果 kernel 配置非法（grid/block 尺寸超限）、shared memory 不足、或设备异常，kernel 静默失败，输出包含未初始化的垃圾数据。

**修复方案**:
- 在每个 Rust FFI kernel wrapper（`extern "C" { fn launch_* }`）调用后添加 `cuda_check(cudaGetLastError(), "kernel_name")`。
- 在 `cudaStreamSynchronize` 后也检查错误状态。

---

### 问题 6（P2 低危）：RMSNorm 无权重时的缓冲区别名

**位置**: `src/cuda.rs:494`

```rust
pub fn rms_norm(&self, x: *mut std::ffi::c_void, w: Option<*mut std::ffi::c_void>,
        y: *mut std::ffi::c_void, d: usize, n: usize, eps: f32) {
    let wptr = w.unwrap_or(y); // 无权重时将 y（输出）当作权重读取
    ...
}
```

**根因**: RMSNorm kernel 读取 `w[i]` 做 `output[i] = x[i] * scale * w[i]`。当 `w = None` 时，`wptr = y`，即读取未初始化的输出缓冲区。这是未定义行为。

**触发条件**: 在 Qwen2 模型中不会触发（所有 RMSNorm 层都有权重），但其他架构可能没有。

**修复方案**: 当 `w` 为 None 时，将权重设置为全 1 的 buffer，或 kernel 内跳过权重乘法：
```rust
let wptr = w.unwrap_or_else(|| {
    // 确保 w_buf 已初始化为全 1.0
    let buf = Self::get_or_grow(&self.buf_w_ones, (d * 4) as usize);
    // ... 初始化 ...
    buf
});
```

---

### 问题 7（P2 性能）：缺少 CUDA Graph / Command Buffer 批处理

**位置**: `src/cuda.rs` 全局

**根因**: 每个 CUDA kernel 直接通过 `<<<>>>` 启动到 stream，没有 Metal 的 `MpsCommandBuffer` 等效机制。

**对比 llama.cpp**: 使用 CUDA Graph 录制整个 decode 图，降低 kernel 启动开销。

**后果**: 24 层 × ~15 个 kernel/层 ≈ 360 次 kernel 启动，每次 ~5-10µs，总计 ~2-3ms 纯启动延迟。对 batch=1 解码，kernel 执行时间可能小于启动延迟。

**修复方案**:
- 短期: 实现类似 Metal 的 `CudaCommandBuffer` 抽象，将 kernel 编码到 graph 中，最后统一提交。
- 长期: 使用 CUDA Graph API (`cudaGraphInstantiate`/`cudaGraphLaunch`) 录制并回放 decode 阶段的计算图。

---

## 四、与 llama.cpp 的性能差异根因

根据 `CUDA_OPTIMIZATION.md` 的基准数据（RTX 4080 Laptop, Qwen2-0.5B Q4_0）：

| 阶段 | minfer CUDA | minfer CPU | llama.cpp CUDA | 差距 |
|------|-------------|------------|-----------------|------|
| Prefill tok/s | 40 | 18 | ~200+ (估计) | ~5x |
| Decode tok/s | 20 | 15 | ~100+ (估计) | ~5x |
| Speedup vs CPU | 2.2x / 1.3x | 1x | 5-10x | 3-5x 差距 |

主要瓶颈（由重到轻）：
1. **问题 4** — KV cache O(n²) 增长 + 同步 cudaMemcpy（解码主耗时）
2. **问题 7** — 无 kernel 批处理，每个 kernel 单独启动（360+ 次/步）
3. **MatMul 内核无 batch 自适应** — 对 nt=1 的 decode 仍用大 grid 的 tile kernel，而非 llama.cpp 的 mmvq 向量内核
4. **问题 5** — 可能存在的静默 kernel 重试（驱动层面）

---

## 五、修复优先级

### P0（阻塞性 — 推理无法正常工作）
- [ ] **#1**: 修复 `layer_gpu` false → `return vec![]` — 添加优雅回退
- [ ] **#2**: 添加 `[features] cuda = []` 到 `Cargo.toml`

### P1（重要 — 功能/性能严重受损）
- [ ] **#4**: 预分配 KV cache 到 `n_ctx` 大小
- [ ] **#5**: 添加 CUDA kernel 错误检查
- [ ] **#3**: 扩展 `output_norm_gpu` 量化类型支持

### P2（优化 — 性能提升）
- [ ] **#6**: 修复 RMSNorm 无权重时的缓冲区别名
- [ ] **#7**: 实现 CUDA Graph / CudaCommandBuffer
- [ ] 添加 mmvq 向量 matmul 内核（针对 batch=1 的解码场景）
