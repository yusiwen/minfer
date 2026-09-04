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

**Graph reuse detection**: `llm_graph_result::can_reuse(params)` (declared at `src/llama-graph.h:888`; the comparison logic is in `llm_graph_params::allow_reuse()`, `src/llama-graph.h:738`) compares whether the new params are topologically equivalent to the previous step's params: `arch`, `gtype`, `cvec`, `loras`, `cparams` flag bits (`embeddings`/`causal_attn`/`nextn_layer_offset`), ubatch structure (`n_tokens`/`n_seq_tokens`/`n_seqs`/`equal_seqs`, and the seq id set when `equal_seqs` is split), `n_outputs`, sampler set (including its output tensor bindings). **Key invariant: the graph topology is a deterministic function of these params** —— if the params are equivalent, the topology is necessarily identical. Therefore, on reuse, the graph is not rebuilt and `ggml_backend_sched_alloc_graph()` is not called, only the input tensor data is updated (`res->set_inputs(&ubatch)`). Note that `n_past` **does not participate** in the comparison: the KV cache position is data (the idx/position input tensors filled at each step), not graph structure.

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

    // KV Cache write + Attention (KV write and wo projection are both done inside build_attn)
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

The actual algorithm is **5 passes** (not a simple "three-pass scan"):

1. **Pass 1 — Initial assignment**: iterate all leaves and nodes; for tensors without an explicit backend, assign by `backend_id_from_cur()` (i.e. the backend of the buffer holding the data, typically the GPU where weights live), without overriding user-specified assignments.

2. **Pass 2 — Assignment expansion**: a total of **4 sub-passes** (in order: expand GPU down → expand GPU up → expand rest down → expand rest up). The first two only expand non-CPU GPU backends (cleared and skipped when `cur_backend_id == n_backends - 1`, i.e. CPU); the last two expand the remaining unassigned nodes to the current backend (including CPU). Result: **CPU is used only when weights are on CPU, or there is a CPU-only op between GPUs**; unsupported ops are left empty for later handling.

3. **Pass 3 — Upgrade + fallback**: for already-assigned nodes, if a backend with the "same buffer type and higher priority" supports the op and all srcs are compatible, upgrade (e.g. when BLAS/CPU share the host buffer type, a CPU node can be upgraded to BLAS); for still-unassigned nodes, choose the backend that "supports the most already-assigned inputs".

4. **Pass 4 — src/view completion**: a view node shares its backend with its view_src; remaining unassigned srcs inherit the dst's backend; if still empty, choose the first supporting backend (`GGML_ASSERT` guarantees one exists —— therefore a CPU fallback must exist).

5. **Pass 5 — Splitting**: iterate `nodes[]`; apart from "adjacent nodes' backend changes", **the following cases also start a new split**: the current node's weight src (`GGML_BACKEND_BUFFER_USAGE_WEIGHTS`) is on a different and incompatible backend (in which case the previous split's GPU memory can be reused); a split's input/output tensor count exceeds the limit. This produces the `splits[]` sequence, and records the input/output tensors each split needs to copy across backends.

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

Implementation details (`ggml_backend_sched_compute_splits()`, `ggml-backend.cpp:1594`):

- Synchronization with the previous split is done via `ggml_backend_event` (`event_synchronize`/`event_wait`), degrading to `ggml_backend_synchronize` only when no event exists.
- User input tensors (`GGML_TENSOR_FLAG_INPUT`) must be copied **immediately and synchronously**, to prevent the user from modifying data before the copy completes.
- **MoE weight optimization**: when the split's first node is `MUL_MAT_ID` (MoE expert matmul) and the input weights are in a host buffer, it reads the expert id tensor and copies only the contiguous expert blocks used this time to the GPU (including trailing padding to prevent NaN), significantly reducing cross-backend copy volume.

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

The reuse decision is implemented by `llm_graph_params::allow_reuse()` (`src/llama-graph.h:738`), called by `llm_graph_result::can_reuse()` (`src/llama-graph.h:888`). Comparison items:

- `arch` — model architecture
- `gtype` — graph type (decode/prefill/MTP draft, etc.)
- `cvec` / `loras` / `cross` — adapter pointers
- `cparams.embeddings` / `cparams.causal_attn` / `cparams.nextn_layer_offset` etc.
- `ubatch` structure: `n_tokens`, `n_seq_tokens`, `n_seqs`, `n_seqs_unq`, `equal_seqs`, token/embd input shapes; when `equal_seqs` is split, it also compares each seq's `seq_id` one by one
- `n_outputs` — the number of output tokens
- `samplers` — sampler set (and the `output[i]`/`seq_id[i][0]` bindings when a sampler is present)

**`n_past` (the KV cache position) does not participate in the comparison** —— it only affects the input data (the values of the idx/position tensors), not the graph topology.

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

Combines the Q, K, V projections into one or more matrix multiplications (supports both fused wqkv and separate wq/wk/wv paths), returning a struct (defined at `src/llama-graph.h:937`):

```cpp
struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};
```

### 5.4 `build_lora_mm()` — Matrix Multiplication with LoRA

Actual signature (`src/llama-graph.h:1006`, weights first): `build_lora_mm(w, cur, w_s = nullptr)`:

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

LoRA adapters are statically unrolled during graph construction, adding no runtime branching. Note: whether LoRA is unrolled affects the `loras` param and thus the `allow_reuse()` reuse decision (switching a LoRA adapter changes the graph topology and triggers a rebuild).

### 5.5 `build_attn()` — Attention

Supports multiple attention modes:
- **MHA** (Multi-Head Attention)
- **GQA** (Grouped-Query Attention)
- **MLA** (Multi-head Latent Attention, DeepSeek)
- **SWA** (Sliding Window Attention)
- **Flash Attention** (when `cparams.flash_attn` is enabled)

### 5.6 KV Cache Interaction

```
inp_attn = build_attn_inp_kv()      // create KV cache input descriptor (register idx/mask etc. input tensors)
...
cur = build_attn(inp_attn, wo, ..., Qcur, Kcur, Vcur, kq_scale, il)
        ├─ ggml_build_forward_expand(Qcur/Kcur/Vcur)   // expand first, prevent reordering
        ├─ mctx_cur->cpy_k(ctx0, Kcur, k_idxs, il)     // ★ write KV cache (done inside build_attn)
        ├─ k = mctx_cur->get_k(ctx0, il)               // read history K (a view of the cache tensor)
        ├─ v = ggml_view_4d(ctx0, k, ...)              // V is a tail view of the K tensor (when KV is stored together)
        └─ build_attn_mha(q, k, v, kq_mask, ...)       // attention
```

Note: **the `inp_attn->set_input_kv()` interface does not exist**. The KV write is a graph node done inside `build_attn()` via `mctx_cur->cpy_k()` (the write indices come from `llm_graph_input_attn_kv::self_k_idxs`, filled each step by `set_input(ubatch)`); the K/V read is a view of the KV cache tensor (`get_k()`), so **the graph topology is independent of `n_past`** —— the position exists only in the values of the input tensors, which is exactly why graph reuse works at every decode step.

KV cache memory management is abstracted by the `llama_memory_context_i` interface, supporting multiple implementations (standard cache, ISWA, DSA, MTP, etc.).

---

## 6. Comparison with minfer

| Dimension | llama.cpp | minfer |
|-----------|-----------|--------|
| **Execution model** | Declarative DAG, build graph first then execute | Imperative forward, compute while building |
| **Memory management** | Backend scheduler auto-allocates + reuses | Manual tensor lifecycle management |
| **Multi-backend** | Auto split + async copy | Manual GPU dispatch (`metal.rs`) |
| **Operator fusion** | Done at graph-construction time by the model code / backend kernel layer (`LLM_FUSED_OP_FLASH_ATTN` etc.; the CUDA backend uses `ggml_can_fuse` to fuse op sequences at the kernel layer; CUDA Graph capture is keyed on `uid`), and the scheduler has **no** general fusion pass | Already has manual fusion (GPU: swiglu/attn_bias_rope_store/fused qkv+gu kernel; CPU: batched matmul) |
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
| `ggml/src/ggml-backend.cpp:1594` | `ggml_backend_sched_compute_splits()` per-split execution (including MoE expert partial copy) |
| `ggml/src/ggml-backend.cpp:1961` | `ggml_backend_sched_graph_compute_async()` entry |
| `ggml/include/ggml-backend.h` | Backend scheduler API documentation |
| `src/llama-graph.h:738` | `llm_graph_params::allow_reuse()` reuse decision |
| `src/llama-graph.h:859` | `llm_graph_result` result container |
| `src/llama-graph.h:888` | `llm_graph_result::can_reuse()` |
| `src/llama-graph.h:950` | `llm_graph_context` graph builder base class |
| `src/llama-graph.cpp` | `set_input()`/`build_attn()` (including KV write `cpy_k`)/`build_lora_mm()` implementation |
| `src/llama-context.cpp:1325` | `process_ubatch()` end-to-end flow |
| `src/llama-context.cpp:2475` | `graph_compute()` execution entry point |
| `src/models/llama.cpp` | LLaMA model graph construction example |
| `src/models/qwen2.cpp:53` | Qwen2 model graph constructor (`llama_model_qwen2::graph::graph`) |
| `src/models/models.h` | All model architecture declarations (nested `struct graph : public llm_graph_context`) |
