# Metal Backend Optimizations

> **目标**：与 llama.cpp (commit `88b47a755`，Apple M4 Pro) 的性能对齐。
> **当前差距（2026-08-11 同模型同参数 A/B）**：decode 72-88 % of llama（纯 GPU
> 1.1-1.4×），prefill 2.8-3.6×。详见 [§1 现状](#1-现状与差距)。
>
> ⚠️ **§0 进度总览表是本文件的唯一进度跟踪依据**；§1-§6 是表格内容的展开说明。
> 修改代码前请先更新 §0。

---

## §0 进度总览（唯一跟踪依据）

> 标记约定：`[x]` 已完成（含实现 commit）· `[ ]` 待办 · `[—]` 已判定不改。
> commit hash 以 `git log` 为准；少数早期条目标"待追溯"（历史较长，后续补齐）。

### ✅ 已完成

| # | 工作项 | 实测效果 | Commit |
|---|---|---|---|
| 1 | Metal 后端基础 + 4 个正确性修复（RoPE freq_scale / output_b / softmax max / 栈数组） | Qwen2-0.5B 130→334 t/s | `2473981` / `26f0e4d`（早期，待追溯细化） |
| 2 | Q5_K 公式（unsigned）+ qh 索引修复 | Q5_K_M CPU/GPU 输出正确 | `3f23560` 前（待追溯） |
| 3 | Q5_1 / Q5_K Metal matmul kernels | Q5_K_M 全 GPU 可跑 | `26f0e4d` / 后续（待追溯） |
| 4 | Q4_0 → f32 activations（对齐 llama Metal，删 Q8_0 量化） | decode +5-10 % | `ba51f68` |
| 5 | GQA attention `simd_max` 发散修复（partial tiles） | 长 prefill 输出正确 | `28d4ba2` |
| 6 | GPU-hang 安全加固（bounded wait + dispatch trace + 屏障守卫） | 死锁→报错退出 | `bff73db` |
| 7 | Metal cb/encoder autorelease retain 修复 | 后台线程 cb 不再 assert | `b1256d5` |
| 8 | Fused QKV + FFN gate/up matmuls（decode, nt==1） | decode ~5 %（24 % 是 GPU 状态假象，`26b145b` 修正） | `6f0c847` |
| 9 | KV-parallel split attention（decode, 2-pass） | 200-token 1.56→1.06 s（~32 %） | `b3d4c7a` |
| 10 | Attention float4 acc + 自适应 chunks + KV 几何增长 | 额外 ~15 % + 长上下文修复 | `66f4290` |
| 11 | simdgroup GEMM：非 Q4_0 quants（Q8_0/Q5_0/Q5_1/Q4_K/Q5_K/Q6_K） | K_M prefill 300→650 t/s | `c9f865c` / `2c03bd1` |
| 12 | Q4_1 simdgroup GEMM | 全部 quant 都有 GEMM | `5b914f0` |
| 13 | f16 split attention（`MINFER_CACHE_TYPE=f16`） | f16 decode 1.60→0.95 s | `387d612` |
| 14 | float4 elementwise + 并行 RoPE（P6/P7） | 200-token ~0.88→~0.80 s | `ddd3eb0` |
| 15 | CPU sampler 加速（top_k O(n) / top_p 只排幸存者） | 采样 ~12.6-14.8→~5.5-6.5 ms/token（2×） | `192378d` |
| 16 | 256-thread RMSNorm + 逐 kernel 剖析 + KV-growth 修复（chunk cap 32→16、去 sync_kv_to_cpu） | ~3 % decode + 长上下文 ~0.25 ms/token | `a7f21e4` |
| 17 | 并行 prefill attention（3-pass，barrier-free） | pp430 212→144 ms（~32 %）；7B 944→832 ms | `b2c97fd` |
| 18 | `Generated:` 纯 decode 口径 + 双口径 bench.sh | 测量可信度修复 | `dc66d0d` |
| 19 | Q4_K AVX2 测试引用修复（实现本就正确） | 29 bin 测试全绿 | `266ffb7` |
| 20 | Split-GGUF（7B 多分卷）支持 | 7B 可加载推理 | `cbba68c` / `34eaf10` |
| 21 | 同模型同参数 A/B 基准文档 | 差距基准确立 | `09d27ae` |

### 🔜 待办（对齐 llama.cpp 性能的必要路径）

> 原则（2026-08-12）：不接受现状——llama.cpp 能达到的性能 minfer 也要达到。
> 原先的 "accept the architecture floor" 结论已撤销；§4 是唯一行动路径。

| # | 工作项 | 目标 | 当前状态 |
|---|---|---|---|
| 1 | **Xcode GUI 逐 kernel GPU trace**（minfer + llama，同 workload，decode + prefill） | 定位非 matmul 4× 与 GEMM ~30 % 差距的精确位置 | 未开始。CLI 方法已证不可行（§4.1） |
| 2 | **flash attention 移植**（或等效的 1-kernel decode attention） | decode 非 matmul 1.2→0.3 ms | 由 trace 决定。原 "dead-end" 判定撤销（§4.2） |
| 3 | **prefill GEMM 执行效率 → ~7 TFLOPs/s**（llama 水平） | prefill 2.3-2.8× → 1× | grid-shape 探查先行（3.5-5.4 方差，§4.3） |
| 4 | 其余按 trace #1 结果补充 | — | 待定 |

### ❌ 已判定不改（有实测或 llama 源码一致性依据）

| # | 工作项 | 判定依据 |
|---|---|---|
| 1 | 2D `simdgroup_matrix`（mpp tensor）GEMM 移植 | llama 在 M4 Pro 禁用 tensor GEMM（PARAMETER_AUDIT A）——不是 llama 的优势来源 |
| 2 | bf16 / f16 中间 activations | Core convention #1：llama Metal 读 f32 activations（仅 KV → f16） |
| 3 | 非阻塞 multi-cb | minfer encode 已隐藏（0.13 ms）；实测 `MINFER_SPLIT_CB` 线性回归 |
| 4 | 并行 command buffer（A1） | 实测回归（1.67/1.08/1.43 s vs 串行 0.93 s），已 revert |
| 5 | dispatch fusion（store_kv_both / residual_rms_norm） | 实测无增益（1.79 vs 1.74 s），已 revert |
| 6 | nt==1 matmul 重写（全 block matvec） | 实测已达 ~200 GB/s 带宽地板，无收益 |
| 7 | Q6_K / Q4_K dequant 向量化 | Q5_0 全量向量化仅 +2.6 % —— 同类低收益 |
| 8 | f16 KV 缓存改默认 | 0.5B 实测 ~3 % 更慢（dispatch-latency-bound），保持 opt-in |

---

## 1. 现状与差距

### 1.1 同模型、同参数 A/B（2026-08-11，M4 Pro，相同 GGUF）

minfer `--greedy`（纯 decode，llama "Generation" 口径）；llama.cpp `llama-bench
-b 512 -t 8`（纯 eval）。模型 Qwen2.5-0.5B-Instruct。

**Q4_K_M**（`qwen2.5-0.5b-instruct-q4_k_m.gguf`）：

| Test | llama.cpp | minfer | Gap |
|---|---|---|---|
| prefill 30 tok | 2720 t/s | ~748 t/s | **3.6×** |
| prefill 430 tok | 6909 t/s | ~2466 t/s | **2.8×** |
| decode 128 tok（纯 GPU） | 293-299 t/s | ~218 t/s（4.47 ms/tok 稳态） | **1.3-1.4×** |
| decode，默认采样 | 247 t/s | ~197 t/s | **1.25×** |

**Q4_0**（`qwen2.5-0.5b-instruct-q4_0.gguf`）：

| Test | llama.cpp | minfer | Gap |
|---|---|---|---|
| prefill 30 tok | 2610 t/s | ~812 t/s | **3.2×** |
| prefill 430 tok | 7449 t/s | ~2596 t/s | **2.9×** |
| decode 128 tok | 314-339 t/s | ~279 t/s（3.90 ms/tok 稳态） | **1.1-1.2×** |

**读数**：
- **Decode 已是 llama 的 72-88 %**（纯 GPU 1.1-1.4×，默认采样 1.25×）——由
  rms_norm_256、chunk-cap/sync 修复、逐 kernel 剖析共同推动（2026-08-10 前是 1.47×）。
- **Prefill 仍是主要差距（2.8-3.6×）**——并行 attention 修复后（100→30 ms），
  剩余差距在 matmuls + small kernels（§4.3）。

### 1.2 逐 token GPU 分解（decode, nt==1, Q4_K_M 0.5B）

| Category | minfer GPU | llama GPU | 证据 |
|---|---|---|---|
| matmul（QKV/O/GU/down/output，~97 kernels） | **~3.0 ms**（~130 GB/s） | ~3.0 ms（源码+参数一致） | minfer 实测 / llama 推断 |
| attention（split 2 kernels） | **0.54 ms** | ~0.15-0.2 ms（flash vec 1 kernel） | minfer 实测（skip-ATTN）/ llama 推断 |
| small elementwise（norm/bias/rope/store/add/swiglu，~300） | **~0.5 ms** | ~0.1-0.3 ms | minfer 实测 / llama 推断 |
| base infra（encode+submit+download） | encode 0.13 + download 0.02-0.03 | ~0.3-0.5（含 multi-cb encode） | minfer 实测 / llama 推断 |
| **Total** | **~4.35-4.55 ms/token GPU** | **~3.1-3.3 ms GPU / 3.51 wall** | 交错 A/B |

### 1.3 整条流水线对比（decode token, nt==1）

| Stage | minfer | llama.cpp | CPU/GPU |
|---|---|---|---|
| Sampling | `sampler.rs` top_k/top_p/temp/repeat-penalty（O(n) + 候选链） | `llama-sampler.cpp` 候选链（partial_sort） | **CPU** |
| Dispatch encode | `MpsCommandBuffer` set_buffer/set_params ×N，单线程 | 相同；multi-cb 线程隐藏 encode | **CPU** |
| GPU execution | 单 cb 串行 ~483 dispatches（Q4_K_M） | 单/多 cb，~490-530 dispatches | **GPU** |
| Embedding | Q4_0: `kernel_get_rows_q4_0`；其它: CPU embed + upload | `ggml_get_rows` → Metal | GPU（或 CPU upload） |
| KV store | `kernel_store_kv_f32`/`_f16`（2 dispatches） | `kernel_cpy_f32_f16` ×2（K,V） | GPU |
| Attention | **2 kernels/layer**（partial + combine） | **1 kernel/layer**（flash vec） | GPU |
| Logits readback | `copy_from_gpu` 607 KB | Metal buffer read | GPU→CPU |

### 1.4 每层 kernel 序列（Q4_K_M 0.5B, nt==1）

**minfer — 20/layer**（fused QKV OFF：Q5_0/Q5_0/Q8_0 混合类型，无法 concat）：

`RMSNorm → Wq/Wk/Wv 3×matmul → 3×add_bias → 2×RoPE → 2×KV store →
attention split(partial+combine) → Wo matmul → residual → RMSNorm →
fused gate+up matmul → SwiGLU → Ffn_down matmul → residual`

×24 = 480 + output_norm 3 = **483**。Q4_0 模型（全 Q4_0）：fused QKV + BSR 生效 →
**12/layer** ×24 + output 3 + GPU embed 1 = **292**。

**llama.cpp — 17/layer**（flash_attn on）：

`RMSNorm → Wq/Wk/Wv 3×matmul → 2×RoPE → 2×KV store(f32→f16) →
flash attention(1 dispatch) → Wo → residual → RMSNorm → gate+up 2×matmul →
SwiGLU → Ffn_down → residual`

×24 = 408 + output 3 + embed 1 ≈ 412 base；graph 822 nodes → **~490-530
dispatches**（f16 cast/cont/reshape 非 no-op 节点）。

### 1.5 早期性能里程碑（Qwen2-0.5B，历史记录）

| Phase | Optimization | Decode (short) | Cumulative |
|---|---|---|---|
| Baseline | GPU + 4 正确性修复 | 130 tok/s | 1.0× |
| +2 | Flash Attention + float4 | 151 tok/s | 1.2× |
| +3 | SIMD-parallel attention | 196 tok/s | 1.5× |
| +4 | SIMD-parallel RMSNorm | 334 tok/s | 2.6× |
| +5 | SwiGLU fusion | 312-334 tok/s | 2.5× |

（早期数字为混合口径；2026-08-06 后 `Generated:` 已是纯 decode 口径。）

---

## 2. 差距分析：已证实 vs 推断

> **核心结论（2026-08-06 #5，原文）**："Structural" is an **inference, not a
> proven architectural inferiority**。§2.1 是严格 VERIFIED 的部分，§2.2 是排除
> 后的推断。逐 kernel 的判定性测量（Xcode GUI）**从未完成**（§4.1），所以任何
> "架构上限"结论都是暂定——本项目不接受该结论（见 §4）。

### 2.1 严格已证实

1. matmul kernel **源码**逐行一致（nt==1 `mul_vec_q_n_f32_impl` /
   `block_q*_dot_y` 翻译）。
2. dispatch **数量**可比（~436 vs ~490-530）。
3. dispatch **参数**一致（`ggml-metal-impl.h` N_R0/N_SG 与 minfer 逐项匹配，
   2026-08-06 #6）。
4. dispatch **次数**几乎一致（~484 vs ~490-530），差距在每 dispatch 的 GPU 执行
   时长（10.3 µs vs 6.2 µs）。
5. prefill GEMM 天花板 **~5.4 TFLOPs/s**（`prefill_gemm_throughput_profile`，
   2026-08-11 A1）；llama 有效 ~7 TFLOPs/s。

### 2.2 已关闭的假设（2026-08-06 #6）

1. **Dispatch 参数** — DISPROVEN（见 2.1.3）。
2. **Attention 是 decode 主杠杆** — DISPROVEN：llama `-fa on/off` = 3.64 vs
   3.88 ms（只省 ~0.25 ms）；即使 flash OFF llama 也快于 minfer。
3. **Multi-cb 是杠杆** — DISPROVEN：minfer encode 仅 0.13 ms，`MINFER_SPLIT_CB=N`
   线性回归（0.67 → 0.93/1.23/1.62 s）。
4. **CPU 侧（encode/sampler）** — DISPROVEN：encode 0.13 ms；sampler 已修（2×）。
5. **Q5_0 标量 dequant 是 matmul 瓶颈** — DISPROVEN：全量向量化仅 +2.6 %；
   matmuls ~130 GB/s 是 nt==1 小 grid 结构性延迟。
6. **Matmul 是 prefill 差距的主因（attention 修复前）** — 2026-08-11 证实 classic
   attention ~100 ms（48 %）才是；修复后 matmuls 成为剩余主项。

### 2.3 非 matmul 逐 kernel 剖析（2026-08-10, P0）

`metal.rs::tests::non_matmul_bandwidth_profile`（batched-cb，median of 3）——
单 dispatch 被 ~165 µs cb launch+sync 地板主导，必须批量测中位数：

| Kernel | µs/dispatch | Notes |
|---|---|---|
| rms_norm 32t（1 simdgroup） | **13.8** | 7× elementwise——latency-bound |
| rms_norm 256t（8 simdgroups，P1） | **3.7** | ~3.7× 更快，bit-identical |
| add_f32 / add_bias / swiglu / rope / store_kv | ~1.6-2.3 | 256-thread elementwise 基线 |
| attn_bias_rope_store（BSR） | 3.1 | |
| **attention split pair**（partial+combine, nkv=430） | **44.3** | 主导非 matmul kernel |
| attention classic（single-pass, nkv=430） | **352** | 比 split 差 8×——确认 split 设计 |

**发现**：① attention split pair 是主导非 matmul kernel（44 µs/layer），只有忠实
flash port 能砍；② small elementwise tail（~300 kernels × 2-3 µs）是结构性便宜
的每-dispatch 延迟，rms_norm 是唯一例外且已修。

### 2.4 Final gap report（2026-08-06 精确分解）

| Component | minfer GPU | llama GPU | gap |
|---|---|---|---|
| matmuls | ~3.0 ms（bandwidth-bound，源码+参数一致） | ~3.0 ms | **~0** |
| **非 matmul**（attention + small + serialization） | **~1.2 ms** | **~0.3 ms**（推断） | **~0.9 ms** |

**结构差距 100 % 在非 matmul**——minfer ~340 个小 kernel 以 llama 的 ~4× 效率运行。
（诚实不确定度：minfer 的 1.2 ms 可靠；attention-vs-small 细分有 ±0.2 ms 噪声；
llama 的 0.3 ms 是推断。）

### 2.5 KV-growth 组件（2026-08-10 部分修复）

minfer 平均 decode 随上下文增长（5.05 → 6.7 ms at -n 64→512），llama 持平。两个
可修复原因已处理：attention chunk cap 32→16（避免过度并行）+ 移除纯 GPU 路径的
`sync_kv_to_cpu`（每 token O(nkv) 拷贝）。交错 A/B：4.65-4.76 → 4.50-4.55 ms/token
（~0.2-0.25 ms）。剩余是 sub-linear attention KV-read，llama 靠 f16 cache + flash
原生摊销。

---

## 3. 已完成优化详解

### 3.1 正确性修复（Metal 后端基础）

4 个早期 bug（均影响输出正确性，Qwen2-0.5B）：
1. **RoPE freq_scale 未应用**（`metal.rs` `rope_f32` 加参数 + `forward.rs` 传
   `hp.rope_freq_scale`）。
2. **output_b 未应用**（`output_norm_gpu` 加 bias）。
3. **softmax max 初始化**（用 `-INFINITY` 而非 0，防 NaN）。
4. **attention 栈数组硬编码**（hd 维度动态化）。

**Q5_K 公式 + qh 索引修复**（2026-07-31，影响 CPU + Metal）：
- 公式：Q5_0 风格 signed `dl*(u-16)-ml` 错 → llama 的 **unsigned** `dl*u-ml`。
- qh 高位索引：`qh[sub*4+pos/8] bit pos%8` 错 → **`qh[pos] bit sub`**。
- 影响 `avx2.rs` / `kernel.rs` / `forward.rs`（embed）三处。

**GQA attention `simd_max` 发散修复**（2026-08-01, `28d4ba2`）：
partial KV tile（`nkv % 32 != 0`）中越界 lane 提前退出循环 → `simd_max(dot)` 跨发散
lane 读到 stale 寄存器 → online-softmax running max 损坏 → 重复循环输出。
修复：统一迭代次数 + `valid` 掩码（无效 lane `dot=-INF`, `e=0`）。结果：prefill
logits cos 0.83→0.999，回归测试 `tests/gqa_attn_isolation.rs`。

**GPU-hang 安全加固**（2026-08-03, `bff73db`）：
1. `submit()` bounded 10 s wait + `MTLCommandBufferStatus` 检查 + `MINFER_TRACE`
   dispatch trace（GPU fault 报错退出而非冻结整机）。
2. attention kernel 不在 barrier 前提前 return（防 `nh % nk != 0` 死锁）。
3. `layer_gpu`/`output_norm_gpu` 运行时守卫：`nh % nk == 0`、`hd ≤ 256`、
   `id % 32 == 0`，违反即报错退出（`gpu_abort`）。

### 3.2 CPU 采样器（2026-08-06）

**根因**：`sampler.rs` 每 token 全词表 O(n·log n) 排序 + 607 KB 拷贝（top_k）+
151,936 `(usize,f32)` 全排序（top_p）。llama.cpp 是候选链（`std::partial_sort`
O(n·log k) → 后序 sampler 只处理 ≤k 幸存者）。

**修复**：
- top_k → `select_nth_unstable_by`（O(n)，在副本上跑保持 index→token 映射）。
- top_p → 只对 ≤k 幸存者 softmax + 排序（>1024 时回退全数组路径）。
- temp → 跳过 masked（-INF）logits 的 exp()。
- `main.rs` 用 move 替代 `logits_all[..].to_vec()`（607 KB/token）。

**实测**：-n 128/256/512 均 **~2.0×**；默认采样 ~12.6-14.8 → ~5.5-6.5 ms/token；
固定种子输出 **byte-identical**（7 个 sampler 测试全过）。

### 3.3 Decode 优化（GPU）

| 工作项 | 效果 | commit |
|---|---|---|
| Fused QKV + FFN gate/up（nt==1 单 matmul/组） | ~5 % decode | `6f0c847` |
| KV-parallel split attention（2-pass online-softmax） | ~32 % decode | `b3d4c7a` |
| float4 acc + 自适应 chunks + KV 几何增长 | 额外 ~15 % + 长上下文 | `66f4290` |
| f16 split attention（partial `_f16`） | f16 1.60→0.95 s | `387d612` |
| float4 elementwise + 并行 RoPE（P6/P7） | ~2-3 % | `ddd3eb0` |
| 256-thread RMSNorm（llama `kernel_rms_norm_fuse_impl` 移植） | ~3-4 % | `a7f21e4` |
| Fused bias+RoPE+KV-store（BSR，7→1 kernel, nt==1） | ~5 % | `5c106dd` |

**Split attention 设计要点**（`b3d4c7a`）：
- Pass 1 `kernel_gqa_attn_partial_f32`/`_f16`：grid (nt, nk, n_chunks)，每 TG 对其
  KV chunk 做 online-softmax partial (mx, S, acc)，结构同 classic kernel。
- Pass 2 `kernel_gqa_attn_combine_f32`：grid (nt, nh) 合并 partials（纯 elementwise，
  无 shared mem/barrier）。
- `n_chunks = clamp((max_pos+1+31)/32, 1, 16)`（2026-08-10 从 /16..32 下调，
  `MINFER_ATTN_CHUNKS` 可覆盖）。正确性不随 chunk 数变化。

**Fused QKV 要点**（`6f0c847`，25 % 数字是 GPU 状态假象，`26b145b` 修正为 ~5 %）：
Wq/Wk/Wv 与 ffn_gate/up 在加载时**行主序 concat**（`concat_rows` → `blk.{i}.attn_qkv`
/ `ffn_gu`），类型+输入维一致时生效。nt==1 只跑 1 个 matmul/组，rope/store/swiglu
用 `set_buffer` 字节偏移读段。`MINFER_NO_FUSE_QKV=1` A/B byte-identical；
`gemm_isolation.rs::qkv_row_concat_layout` 锁定布局。

**测量陷阱记录**（防重复踩坑）：
- 单 dispatch 隔离计时**不可靠**（~165 µs cb launch+sync 地板）——批量数十次取中位。
- 每个 kernel 首次计时有 cold-start/GPU-clock-ramp 假象（~4×）——先 warm 再测两次。
- 持续压测会热节流 M4 Pro（极端时所有配置 ~1.3 s）——交错配置、取 min/median。

### 3.4 Prefill 优化

**simdgroup GEMM（P0/P1，2026-08-01）**：忠实移植 llama legacy `kernel_mul_mm`
（64×32 tile，4 simdgroup × 32 线程，Q4_0 dequant 入 `sa`，f32 activations 入 `sb`）。
P0 初版 3 bug（B-staging 未 clamp 的行、store transpose 方向、barrier 必须
`mem_threadgroup`）→ P1 修正后 30 tok +11 %、70 tok +34 %。nt ≥ 16 dispatch；
`MINFER_GEMM=0` 回退 f32 multi。隔离测试 `gemm_isolation.rs`（nt=12/30/32/33）。

**非 Q4_0 GEMM（2026-08-03，`c9f865c`/`2c03bd1`/`5b914f0`）**：每个 quant 一个
simdgroup GEMM——Q8_0/Q5_0/Q5_1（32-elem 块）+ Q4_K/Q5_K/Q6_K（256-elem 超块）。
K_M prefill 300→650 t/s，1.5B Q4_K_M 48→442 t/s（~9×）。8 KB threadgroup-memory
守卫。`non_q4_0_gemm_isolation` 验证。

**并行 prefill attention（2026-08-11，`b2c97fd`）**：classic `kernel_gqa_attn_f32`
在 prefill 是 latency-bound（grid (nt,nk) 顺序 KV 循环，~24K barriers，~100 ms =
48 % of prefill，~25× llama）。3-pass barrier-free 替代：
1. `kernel_attn_scores`：每 (t,h) 一个 256-thread TG，每线程算一个 score。
2. `kernel_softmax_attn`：按 kv 轴 masked softmax。
3. `kernel_attn_output`：softmax·V 求和。

GQA 用 per-head `hk = h/gqa`（broadcast-GEMM 尝试后放弃——2D GEMM 无法产出 per-head
3D scores）。⚠️ threadgroup-memory bug：softmax 的 `shmem[tiisg]` 写 32 floats 但只
分配 8 个（OOB 污染邻近内存 → NaN 行）——改 32×4=128 B；rms_norm_256 同 bug 修复。

**实测**：pp430 classic 212 → **144 ms**（attention 100→30 ms）；pp30 44→40 ms；
7B pp230 944→832 ms（attention 169→57 ms）；7B decode 不变。34 bin + 6 isolation
全过，end-to-end byte-identical。

### 3.5 KV / 长上下文

- **KV 几何增长**（`66f4290`）：`kv_ensure_layer` ×2 增长，替代每 token 重分配+拷贝
  整个旧 KV（0.5 ms@KV140 → 4.2 ms@KV2510 → 0.13 ms）。⚠️ 期间 `old_v` 克隆 typo
  污染 V cache → Q4_K_M 垃圾（A/B 测不出——两路径共享同一坏 KV，靠 known-good 参考
  发现）。
- **f16 KV opt-in**（`387d612` + `bff73db`）：`MINFER_CACHE_TYPE=f16`（默认 f32）。
  `kernel_store_kv_f16` + `kernel_gqa_attn_partial_f16`。0.5B 实测 ~3 % 更慢
  （dispatch-latency-bound），保留 opt-in 供大模型/长上下文。
- **split-GGUF**（`cbba68c`/`34eaf10`）：多分卷模型（7B `-0000X-of-0000Y`），
  merged tensor index，7B 加载验证通过。

---

## 4. 未来工作（对齐 llama.cpp 的唯一路径）

> 2026-08-12 声明：**不接受现状**。llama.cpp 能达到的性能（decode ~3.1-3.3 ms
> GPU/token、prefill ~7 TFLOPs/s GEMM）就是 minfer 的目标。以下是有依据的行动路径，
> 按顺序执行；每步完成后更新 §0 进度表。

### 4.1 第 0 步（必须）：Xcode GUI 逐 kernel GPU trace

**为什么必须**：所有 "structural / 无低风险杠杆" 结论都基于排除法 + 源码一致性
推断，**从未有逐 kernel GPU 时长佐证**（§2.1 只证实到"per-dispatch GPU 时间不同"）。
CLI 方法已被证不可行（2026-08-11 A2）：
- `GGML_METAL_CAPTURE_COMPUTE=1` 报 "Capture layer is not inserted"（需 Xcode/
  Instruments GPU capture）。
- `xctrace record --template 'Metal System Trace'` 能抓 .trace，但 CLI export 为空
  ——模板 "Shader Timeline: Disabled"，逐 kernel 时长只能 Xcode GUI 看。

**操作**：在 Xcode/Instruments 中对 minfer 和 llama.cpp（同 workload）各抓一份
decode 与 prefill 的 .trace，逐 kernel 对比 GPU 时长。这决定 §4.2/§4.3 的优先级
（attention 主导？small-op 主导？uniform launch 开销？），也验证 §1.2 中 minfer
subtractive 分解是否高估 matmul。

### 4.2 flash attention 移植（decode 非 matmul 4× 差距）

**原 "dead-end" 判定撤销**（2026-08-12）：2026-08-06 以 "~0.3 ms 收益、多日高风险"
将其降级为 dead-end，但那是基于 **accept-floor 的前提**。既然目标是达到 llama 水平
（非 matmul 1.2→0.3 ms 是结构差距的唯一主体，§2.4），这是**必做路径**，不是可选项。

现状：minfer 的 split attention（2 kernels/layer，~0.54 ms）在结构上已是最优的
非-flash 设计（classic single-pass 8× 更差）。llama 的 flash 快在
`simdgroup_matrix` 设计 + 每形状 function constants，非 "1 kernel" 本身（naive
1-kernel 融合实测 4.80 vs 4.15 ms，更慢）。

**候选实现**：忠实移植 llama `kernel_flash_attn_ext_vec`（~600 行，simdgroup
matrices，f16 KV，KV-parallel tiles）替换 partial+combine（−24 kernels）。
风险：新 kernel + 隔离测试（遵循既有方法论：先 isolation 测试确定性与标量参考
一致，再 byte-identical A/B）。

### 4.3 prefill GEMM 执行效率（~5.4 vs ~7 TFLOPs/s）

**已证实**：`prefill_gemm_throughput_profile`（batched-cb，单 dispatch 验证）显示
同 kernel 纯 grid 形状改变吞吐 **3.5→5.4 TFLOPs/s**（nt=416→3.5, 448→5.1,
480→5.3, 512→5.2）——grid-row 调度方差，非 bc_out bug。FFN matmuls（od=18944）
主导 prefill（~2.8 ms/layer × 24 ≈ 真实 pp430 141-157 ms）。

**反证**：llama 用相同 grid（N_R0/N_SG 一致）达 ~7 TFLOPs/s——若属实，形状本身
无法解释差距，差异在逐 kernel 执行（MPS 序列化）。**因此 grid-shape 探查期望低**，
但零成本、先做以排除（10 分钟级别）。

**后续**：依据 §4.1 trace 定位 GEMM 差距（若在 matmul 内）后，考虑更高执行效率的
GEMM 结构。2D `simdgroup_matrix`（mpp tensor）**已排除**（llama 在 M4 Pro 禁用，
PARAMETER_AUDIT A）；bf16 staging **已排除**（llama 读 f32 activations）。

### 4.4 完成后的回填

每项完成后：更新 §0 进度表（勾选、填 commit、更新实测效果）→ 在对应 §3/§4 章节
记录实现与验证 → 同步 §1 差距数值。

---

## 5. 历史附录（仅参考）

> ⚠️ **本附录是历史记录，只作参考，不作为后续计划的依据。**
> 其中的架构结论、数值口径、kernel 结构可能已被 §1-§4 取代。
> 后续计划一律以 §0 进度表 + §4 为准。

### 5.1 早期 Phase 摘要（Qwen2-0.5B，130→334 t/s，2026-07-27~08-01）

| Phase | 内容 | 关键点 |
|---|---|---|
| 1 | 4 个正确性 bug 修复 | RoPE freq_scale、output_b、softmax max、栈数组（详见 §3.1） |
| 2 | Flash Attention（online softmax）+ float4 向量化 | KV 并行 chunk + running max/sum 修正 |
| 3 | SIMD-parallel attention（vec kernel） | 32-lane simd_dot、threadgroup barrier 同步 |
| 4 | SIMD-parallel RMSNorm | 多 simdgroup 归约 + threadgroup buffer |
| 5 | SwiGLU fusion | silu+mul 单 kernel，省 1 dispatch/层 |

### 5.2 早期差距分析记录

- **KV-growth 2.2×**（2026-08-01）：f32 KV 全量重读 vs llama f16（详见 §2.5 修复）。
- **Per-dispatch encode ~24µs vs llama ~7µs**（2026-08-01 结论）→ **2026-08-03 已
  撤销**：encode 实测仅 ~1 ms/step，decode 是 GPU 执行受限。
- **Q4_0 双 dispatch（quantize+matmul）**（2026-08-01）→ 已修：f32 activations
  路径（§3.1 #4）。

### 5.3 已测试并否定的想法（含 commit）

| 想法 | 结果 | commit |
|---|---|---|
| 并行 command buffer（A1） | 回归（encode 已隐藏），revert | `b1256d5` |
| nt==1 matmul 全 block matvec 重写 | 已达带宽地板，未集成 | — |
| store_kv_both / residual_rms_norm fusion | 无增益，revert | — |
| naive 1-kernel attention（classic 单 pass） | 4.80 vs split 4.15 ms | — |
| f16 KV 改默认 | ~3 % 更慢（0.5B） | `387d612`（保留 opt-in） |
| Q5_0 全量向量化 | 仅 +2.6 % matmul | — |
| broadcast-GEMM 做 prefill attention | 2D GEMM 无法产出 per-head 3D scores | — |

### 5.4 过时数据表（勿引用）

以下来自 2026-08-01/03 的早期测量，口径与当前不同（混合 t/s vs 纯 decode；
旧 llama baseline），**仅供对照历史**：
- "Q4_K_M/Q5_K_M prefill 7.3×"（35-token 表）——当时无非 Q4_0 GEMM，现已补齐。
- "decode short ~187 / long ~86 t/s"（KV-growth 表）——split attention 前。
- "纯 decode 2.0× / 3.2×" 早期差距——现在 1.1-1.4×（§1.1）。
