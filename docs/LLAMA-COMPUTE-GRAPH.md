# llama.cpp Compute Graph Design Analysis

This document analyzes the compute graph architecture design of llama.cpp and its role in end-to-end inference.
Based on llama.cpp source code (2026-08 version).

---

## 1. Core Data Structures

### 1.1 `ggml_cgraph` — The Compute Graph Itself

Defined in `ggml/src/ggml-impl.h:329`:

```cpp
struct ggml_cgraph {
    int size;             // maximum number of nodes/leafs/grads/grad_accs
    int n_nodes;          // number of operator nodes currently in use
    int n_leafs;          // number of constant leaf nodes (weights, etc.)
    ggml_tensor ** nodes; // mutable tensors (operator nodes), topologically ordered
    ggml_tensor ** leafs; // constant tensors (immutable data)
    ggml_tensor ** grads; // gradients (for training, nullptr during inference)
    ggml_tensor ** grad_accs;
    int32_t * use_counts;
    ggml_hash_set visited_hash_set;
    enum ggml_cgraph_eval_order order; // LEFT_TO_RIGHT or RIGHT_TO_LEFT
    uint64_t uid;  // graph identifier, used for reuse detection (0 means not set)
};
```

**Key design**: `nodes` is a **topologically ordered** operator sequence. Each `ggml_tensor` node's `src[]` array points to predecessor nodes, forming a DAG. During execution, forward computation proceeds in `nodes[0..n_nodes-1]` order. `uid` is a graph identifier for recognizing identical graph topologies across calls — e.g. the CUDA backend keys its CUDA Graph cache on `uid` (0 means unset/ignored).

### 1.2 `llm_graph_context` — Graph Builder Base Class

Defined in `src/llama-graph.h:950`, this is the base class for all model graph construction:

```cpp
struct llm_graph_context {
    const llm_arch arch;
    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;
    // ... model dimension parameters (n_embd, n_layer, n_head, n_rot, ...)

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;
    const llama_memory_context_i * mctx; // KV cache memory context

    ggml_context * ctx0; // ggml memory pool (nodes are allocated here)
    ggml_cgraph  * gf;  // the graph to be filled

    llm_graph_result * res; // output result container

    // common builder methods
    ggml_tensor * build_inp_embd(...);
    ggml_tensor * build_norm(...);
    ggml_tensor * build_lora_mm(...);
    ggml_tensor * build_qkv(...);
    // ... etc.
};
```

Each concrete model (e.g., `llama_model_llama`, `llama_model_qwen2`) inherits from this class and implements the specific forward logic in its `graph::graph()` constructor.

### 1.3 `llm_graph_result` — Graph Execution Result Container

Defined in `src/llama-graph.h:859`:

```cpp
class llm_graph_result {
    ggml_tensor * t_inp_tokens;  // input token ids
    ggml_tensor * t_logits;      // output logits
    ggml_tensor * t_embd;        // hidden state (for embedding extraction)
    ggml_tensor * t_embd_pooled; // pooled embedding
    ggml_tensor * t_h_nextn;     // hidden state for MTP/NextN

    std::vector<ggml_tensor *> t_layer_inp; // per-layer input (for speculative decoding)

    std::vector<llm_graph_input_ptr> inputs;  // input tensor set
    std::vector<llm_graph_fused_node> fused_nodes; // fused nodes

    ggml_context_ptr ctx_compute;
    ggml_cgraph * gf;
    int64_t max_nodes;
};
```

**Graph reuse detection**: `llm_graph_result::can_reuse(params)`（声明于 `src/llama-graph.h:888`，比较逻辑在 `llm_graph_params::allow_reuse()`，`src/llama-graph.h:738`）比较新参数与上一步参数是否拓扑等价：`arch`、`gtype`、`cvec`、`loras`、`cparams` 标志位（`embeddings`/`causal_attn`/`nextn_layer_offset`）、ubatch 结构（`n_tokens`/`n_seq_tokens`/`n_seqs`/`equal_seqs`，以及 `equal_seqs` 拆分时的 seq id 集合）、`n_outputs`、samplers 集合（含其输出张量绑定）。**关键不变式：图拓扑是这些参数的确定性函数**——参数等价则拓扑必然相同。因此复用时不重建图、不调 `ggml_backend_sched_alloc_graph()`，仅更新输入张量数据（`res->set_inputs(&ubatch)`）。注意 `n_past` **不参与**比较：KV cache 位置是数据（每步填充的 idx/position 输入张量），不是图结构。

---

## 2. End-to-End Inference Flow

### 2.1 Call Chain

```
llama_decode(batch)
  └─ llama_context::decode(batch)
       ├─ balloc->init(batch)          // initialize batch allocator
       ├─ while (has ubatch):
       │    └─ process_ubatch(ubatch, gtype, mctx)
       │         ├─ 1. mctx->apply()                    // apply KV cache memory context
       │         ├─ 2. check graph reuse
       │         │     if (res->can_reuse(gparams)):
       │         │         reuse previous graph directly
       │         │     else:
       │         │         model.build_graph(gparams)     // build ggml_cgraph
       │         │           └─ dispatch → llama_model_xxx::build_arch_graph()
       │         │                 └─ construct graph object
       │         │         ggml_backend_sched_alloc_graph(sched, gf)
       │         ├─ 3. res->set_inputs(&ubatch)          // copy input data to GPU tensors
       │         └─ 4. graph_compute(gf)                 // execute computation
       │              └─ ggml_backend_sched_graph_compute_async(sched, gf)
       ├─ extract logits / embeddings
       └─ sample next token
```

### 2.2 Graph Construction Process

Using a standard Transformer decoder as an example (e.g., `src/models/llama.cpp`):

```cpp
// 1. input embedding
inpL = build_inp_embd(model.tok_embd);  // token ids → embedding

// 2. per-layer processing
for (int il = 0; il < n_layer; ++il) {
    inpSA = inpL;  // residual connection save point

    // Pre-norm
    cur = build_norm(inpL, model.layers[il].attn_norm, nullptr, LLM_NORM_RMS, il);

    // Q/K/V projection (may include LoRA)
    auto [Qcur, Kcur, Vcur] = build_qkv(layer, cur, ...);

    // RoPE positional encoding
    Qcur = ggml_rope_ext(ctx0, Qcur, inp_pos, nullptr, n_rot, ...);
    Kcur = ggml_rope_ext(ctx0, Kcur, inp_pos, nullptr, n_rot, ...);

    // KV Cache write + Attention（KV 写入与 wo 投影均在 build_attn 内部完成）
    cur = build_attn(inp_attn, layer.wo, layer.wo_b, layer.wo_s,
                     Qcur, Kcur, Vcur, nullptr, nullptr, nullptr, kq_scale, il);

    // residual connection
    inpL = ggml_add(ctx0, inpSA, cur);

    // FFN (SwiGLU: silu(gate(x)) * up(x))
    cur = build_norm(inpL, layer.ffn_norm, nullptr, LLM_NORM_RMS, il);
    gate = build_lora_mm(layer.ffn_gate, cur);
    up   = build_lora_mm(layer.ffn_up,   cur);
    cur  = ggml_mul(ctx0, ggml_silu(ctx0, gate), up);
    cur  = build_lora_mm(layer.ffn_down, cur);
    inpL = ggml_add(ctx0, inpL, cur);  // residual
}

// 3. output layer
cur = build_norm(inpL, model.output_norm, nullptr, LLM_NORM_RMS, -1);
cur = build_lora_mm(model.output, cur);  // lm_head
ggml_build_forward_expand(gf, cur);       // register into graph
res->t_logits = cur;
```

Each `build_xxx()` call creates a ggml tensor node on `ctx0`, automatically establishing `src[]` dependency relationships. Finally, `ggml_build_forward_expand()` recursively adds the result node and all its predecessors into `gf->nodes[]`.

### 2.3 Backend Scheduler Execution

```
ggml_backend_sched_graph_compute_async(sched, gf)
  └─ ggml_backend_sched_split_graph(sched, gf)  // if needed
       ├─ Pass 1: assign backend for each node (prefer GPU)
       ├─ Pass 2: expand GPU coverage up/down to reduce cross-backend copies
       └─ Pass 3: split by backend assignment into splits (contiguous subgraphs)
  └─ for each split:
       ├─ sync previous split (ensure previous split has completed)
       ├─ copy inputs to split's backend (insert copy nodes for cross-backend transfers)
       ├─ ggml_backend_graph_compute(backend, subgraph)
       └─ copy outputs to next split's backend
```

---

## 3. Backend Scheduler Multi-Backend Dispatch

### 3.1 `split_graph()` — Graph Partitioning Algorithm

Defined in `ggml/src/ggml-backend.cpp:1055`.

**Core idea**: Split the entire `ggml_cgraph` into several contiguous subgraphs (splits) by backend assignment, with each split executed on the same backend.

实际算法是 **5 个 pass**（不是简单的"三趟扫描"）：

1. **Pass 1 — 初始赋值**：遍历所有叶与节点，对无显式 backend 指定的张量按 `backend_id_from_cur()`（即其数据所在 buffer 的 backend，典型是权重所在 GPU）赋值，不覆盖用户指定。

2. **Pass 2 — 扩展赋值**：共 **4 个子趟**（顺序：expand GPU down → expand GPU up → expand rest down → expand rest up）。前两趟只扩展非 CPU 的 GPU backend（`cur_backend_id == n_backends - 1` 即 CPU 时清零，跳过）；后两趟把剩余未赋值节点扩展给当前 backend（含 CPU）。结果：**CPU 仅在权重在 CPU、或 GPU 之间存在 CPU-only op 时被使用**；不支持的 op 留空待后续处理。

3. **Pass 3 — 升级 + 兜底**：对已赋值节点，若存在"buffer type 相同且优先级更高"的 backend 支持该 op 且所有 src 兼容，则升级（例如 BLAS/CPU 共享 host buffer type 时可把 CPU 节点升到 BLAS）；对仍未赋值的节点，选"支持最多已赋值输入"的 backend。

4. **Pass 4 — src/view 补齐**：view 节点与其 view_src 同 backend；剩余未赋值 src 继承 dst 的 backend；仍空则选第一个支持的 backend（`GGML_ASSERT` 保证必有——因此必须存在 CPU 兜底）。

5. **Pass 5 — 切分**：遍历 `nodes[]`，除"相邻节点 backend 变化"外，**以下情况也会开启新 split**：当前节点的权重 src（`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`）位于不同且不兼容的 backend（此时可复用上一 split 的显存）；split 的输入/输出张量数量超限。生成 `splits[]` 序列，并记录每个 split 需要跨 backend 拷贝的输入/输出张量。

### 3.2 `graph_compute()` — Per-Split Execution

```cpp
for (int i = 0; i < n_splits; i++) {
    split = &splits[i];

    // 1. Ensure previous split has completed (buffer may be reused)
    if (prev_backend != split->backend) {
        ggml_backend_synchronize(prev_backend);
    }

    // 2. Copy input tensors to current split's backend
    for (input : split->inputs) {
        ggml_backend_tensor_copy(input, split_backend);
    }

    // 3. Execute current split's subgraph
    ggml_backend_graph_compute(split_backend, split->cgraph);

    // 4. Copy output tensors to next split's backend
    //    (via events for async operation, does not block current backend)
}
```

实现细节（`ggml_backend_sched_compute_splits()`，`ggml-backend.cpp:1594`）：

- 前一个 split 的同步通过 `ggml_backend_event` 完成（`event_synchronize`/`event_wait`），没有 event 时才退化为 `ggml_backend_synchronize`。
- 用户输入张量（`GGML_TENSOR_FLAG_INPUT`）必须**立即**同步拷贝，防止用户在拷贝完成前改写数据。
- **MoE 权重优化**：当 split 的首节点是 `MUL_MAT_ID`（MoE 专家 matmul）且输入权重在 host buffer 时，会读取 expert id 张量、只把本次用到的连续 expert 块拷到 GPU（含尾部 padding 防 NaN），显著减少跨后端拷贝量。

### 3.3 Multi-Backend Scenario Example

```
Graph:  [Embedding(CPU)] → [MatMul(GPU)] → [RMSNorm(CPU)] → [MatMul(GPU)] → [LMHead(CPU)]

Split 1: Embedding         (CPU)
Split 2: MatMul            (GPU)
Split 3: RMSNorm           (CPU)  — GPU does not support this op
Split 4: MatMul            (GPU)
Split 5: LMHead            (CPU)
```

Tensor copy nodes are automatically inserted between splits. Async transfer via `ggml_backend_event` does not block GPU computation.

---

## 4. Graph Reuse Optimization

### 4.1 Reuse Conditions

复用判定由 `llm_graph_params::allow_reuse()`（`src/llama-graph.h:738`）实现，`llm_graph_result::can_reuse()`（`src/llama-graph.h:888`）调用它。比较项：

- `arch` — model architecture
- `gtype` — graph type (decode/prefill/MTP draft, etc.)
- `cvec` / `loras` / `cross` — 适配器指针
- `cparams.embeddings` / `cparams.causal_attn` / `cparams.nextn_layer_offset` 等
- `ubatch` 结构：`n_tokens`、`n_seq_tokens`、`n_seqs`、`n_seqs_unq`、`equal_seqs`、token/embd 输入形态；当 `equal_seqs` 拆分时还逐一比较各 seq 的 `seq_id`
- `n_outputs` — 输出 token 数
- `samplers` — sampler 集合（及存在 sampler 时的 `output[i]`/`seq_id[i][0]` 绑定）

**`n_past`（KV cache 位置）不参与比较**——它只影响输入数据（idx/position 张量的值），不影响图拓扑。

### 4.2 Reuse Benefits

- **Skip graph reconstruction**: Avoid `build_graph()` rebuilding the DAG
- **Skip memory allocation**: `ggml_backend_sched_alloc_graph()` is an expensive operation
- **Retain buffers**: GPU buffers are not released, reused directly

For the decode phase (n_tokens=1 each time), the graph topology is nearly unchanged, resulting in very high reuse rates.

---

## 5. Key Graph Construction Patterns

### 5.1 `build_inp_embd()` — Input Embedding

```
token_ids [n_tokens] → ggml_get_rows(embeddings) → inpL [n_embd, n_tokens]
```

When the input is embeddings rather than token ids (e.g., multimodal), `ubatch.embd` is used directly.

### 5.2 `build_norm()` — Layer Normalization

Supports both RMSNorm and LayerNorm, distinguished by the `LLM_NORM_RMS` / `LLM_NORM` enum.

### 5.3 `build_qkv()` — Q/K/V Projection

Combines the Q, K, V projections into one or more matrix multiplications (支持融合 wqkv 与分离 wq/wk/wv 两种路径), returning a struct (定义于 `src/llama-graph.h:937`):

```cpp
struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};
```

### 5.4 `build_lora_mm()` — Matrix Multiplication with LoRA

实际签名（`src/llama-graph.h:1006`，权重在前）：`build_lora_mm(w, cur, w_s = nullptr)`：

```cpp
ggml_tensor * build_lora_mm(ggml_tensor * w, ggml_tensor * cur,
                            ggml_tensor * w_s = nullptr) {
    res = ggml_mul_mat(w, cur);        // w @ cur
    if (w_s) res = ggml_mul(res, w_s); // per-tensor scale
    for (lora : *loras) {
        ab_cur = lora.b @ (lora.a @ cur);  // B @ (A @ cur)
        res = ggml_add(res, ggml_scale(ab_cur, scale));
    }
}
```

LoRA adapters are statically unrolled during graph construction, adding no runtime branching. 注意：LoRA 展开与否会影响 `loras` 参数进而影响 `allow_reuse()` 的复用判定（切换 LoRA 适配器会改变图拓扑，触发重建）。

### 5.5 `build_attn()` — Attention

Supports multiple attention modes:
- **MHA** (Multi-Head Attention)
- **GQA** (Grouped-Query Attention)
- **MLA** (Multi-head Latent Attention, DeepSeek)
- **SWA** (Sliding Window Attention)
- **Flash Attention** (when `cparams.flash_attn` is enabled)

### 5.6 KV Cache Interaction

```
inp_attn = build_attn_inp_kv()      // create KV cache input descriptor（注册 idx/mask 等输入张量）
...
cur = build_attn(inp_attn, wo, ..., Qcur, Kcur, Vcur, kq_scale, il)
        ├─ ggml_build_forward_expand(Qcur/Kcur/Vcur)   // 先展开，防止被重排
        ├─ mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il)     // ★ 写 KV cache（在 build_attn 内部）
        ├─ k = mctx_cur->get_k(ctx0, il)               // 读历史 K（cache 张量的 view）
        ├─ v = ggml_view_4d(ctx0, k, ...)              // V 是 K 张量尾部视图（KV 合存时）
        └─ build_attn_mha(q, k, v, kq_mask, ...)       // attention
```

注意：**不存在 `inp_attn->set_input_kv()` 接口**。KV 写入是 `build_attn()` 内部经 `mctx_cur->cpy_k()` 完成的图节点（写入索引来自 `llm_graph_input_attn_kv::self_k_idxs`，每步由 `set_input(ubatch)` 填充）；K/V 读取是 KV cache 张量的视图（`get_k()`），因此**图拓扑与 `n_past` 无关**——位置只存在于输入张量的值中，这正是图复用能在 decode 每步生效的原因。

KV cache memory management is abstracted by the `llama_memory_context_i` interface, supporting multiple implementations (standard cache, ISWA, DSA, MTP, etc.).

---

## 6. Comparison with minfer

| Dimension | llama.cpp | minfer |
|-----------|-----------|--------|
| **Execution model** | Declarative DAG, build graph first then execute | Imperative forward, compute while building |
| **Memory management** | Backend scheduler auto-allocates + reuses | Manual tensor lifecycle management |
| **Multi-backend** | Auto split + async copy | Manual GPU dispatch (`metal.rs`) |
| **Operator fusion** | 构图期由模型代码/后端内核层完成（`LLM_FUSED_OP_FLASH_ATTN` 等；CUDA 后端用 `ggml_can_fuse` 在 kernel 层融合 op 序列；CUDA Graph 捕获以 `uid` 为键），scheduler **没有**通用融合 pass | 已有手工融合（GPU: swiglu/attn_bias_rope_store/融合 qkv+gu kernel；CPU: 批量 matmul） |
| **Graph reuse** | `can_reuse()` skips reconstruction | Recompute every step |
| **Complexity** | ~3800 lines graph framework + ~100-200 lines per model | No independent graph layer, direct implementation in forward.rs |
| **Flexibility** | New models only need to inherit `llm_graph_context` | New models require writing complete forward |
| **Debugging capability** | `ggml_graph_dump_dot()` exports DOT graph | No graph structure, difficult to visualize globally |

---

## 7. Design Summary

llama.cpp's compute graph architecture is one of its **core competitive advantages**:

1. **Graph-execution separation**: Building `ggml_cgraph` is a pure CPU operation, fast and side-effect-free; execution is uniformly dispatched by the backend scheduler
2. **Multi-backend transparency**: Model code is unaware of specific hardware; the scheduler automatically handles GPU offload and data transfer
3. **Graph reuse**: For fixed batch size decode scenarios, skipping reconstruction significantly reduces latency
4. **Extensibility**: Each model architecture only needs to inherit `llm_graph_context` and implement graph construction, reusing all infrastructure
5. **Debuggability**: DOT format export, callback mechanism, and tensor naming facilitate tracing

The tradeoff is higher code complexity, but this architecture makes llama.cpp a true **inference runtime** (rather than a simple matrix multiplication library), capable of efficiently supporting hardware configurations from single CPU to multi-GPU.

---

## Reference Files

| File | Content |
|------|---------|
| `ggml/src/ggml-impl.h:329` | `ggml_cgraph` struct definition |
| `ggml/src/ggml.c` | Graph operation implementations (`ggml_new_graph`, `ggml_build_forward_expand`, `ggml_graph_dump_dot`) |
| `ggml/src/ggml-backend.cpp:1055` | `ggml_backend_sched_split_graph()` graph partitioning (5 passes) |
| `ggml/src/ggml-backend.cpp:1594` | `ggml_backend_sched_compute_splits()` per-split execution (含 MoE 专家部分拷贝) |
| `ggml/src/ggml-backend.cpp:1961` | `ggml_backend_sched_graph_compute_async()` entry |
| `ggml/include/ggml-backend.h` | Backend scheduler API documentation |
| `src/llama-graph.h:738` | `llm_graph_params::allow_reuse()` 复用判定 |
| `src/llama-graph.h:859` | `llm_graph_result` result container |
| `src/llama-graph.h:888` | `llm_graph_result::can_reuse()` |
| `src/llama-graph.h:950` | `llm_graph_context` graph builder base class |
| `src/llama-graph.cpp` | `set_input()`/`build_attn()`(含 KV 写入 `cpy_k`)/`build_lora_mm()` 实现 |
| `src/llama-context.cpp:1325` | `process_ubatch()` end-to-end flow |
| `src/llama-context.cpp:2475` | `graph_compute()` execution entry point |
| `src/models/llama.cpp` | LLaMA model graph construction example |
| `src/models/qwen2.cpp:53` | Qwen2 model graph constructor（`llama_model_qwen2::graph::graph`） |
| `src/models/models.h` | All model architecture declarations (nested `struct graph : public llm_graph_context`) |
