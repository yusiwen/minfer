# CLI 多轮对话改造方案 (CLI Multi-Turn Conversation Plan)

**Status:** Design Document + implementation record
**Date:** 2026-08-22
**参考代码:** llama.cpp `tools/cli/cli-context.cpp`（现行 client-server CLI）、legacy `examples/main/main.cpp`（独立进程交互循环，commit `1d36b3670` 前）、`common/chat.h` / `common/chat.cpp`（模板机制）

> **Revision 1 (2026-08-22) — Phase 0 implemented.** `template::format_single` + `FormattedDelta`
> 落地（§5.3），6 个单测全绿：差分正确性（assistant 内容不重复喂回）、尾部换行补偿、
> 无换行不补偿、空前缀、前缀失败兜底（reverse 模板）、无模板 ChatML fallback、null-content
> 历史。token 级一致性断言延后到 L1.5 KV 等价测试（需真实 tokenizer/模型，§8.4）。

---

## 目录

1. [概述](#1-概述)
2. [目标与范围](#2-目标与范围)
3. [llama.cpp 参考分析](#3-llamacpp-参考分析)
4. [minfer 现状与可复用件](#4-minfer-现状与可复用件)
5. [目标架构](#5-目标架构)
6. [与 llama.cpp 的对照](#6-与-llamacpp-的对照)
7. [分阶段实现计划](#7-分阶段实现计划)
8. [测试策略](#8-测试策略)
9. [开放问题与决策点](#9-开放问题与决策点)

---

## 1. 概述

minfer 的 CLI（`src/main.rs`）目前是**单发**模式：一个 prompt → 渲染单条 user 消息 → prefill → 解码 → 退出。
Server 路径（`src/server/`）已经支持多消息渲染与完整解码循环，但 CLI 无法连续对话。

本文档参考 llama.cpp 的两代 CLI 设计，为 minfer CLI 设计多轮对话（conversation mode）改造方案。
核心策略是**追加式 KV + 增量模板渲染**（llama.cpp legacy 路线）：整个对话累积在 KV 缓存中，
每回合只对新消息做 prefill/decode，而不是像现行 llama.cpp CLI 那样每回合全量重发历史。

---

## 2. 目标与范围

### 目标

- `minfer <model> --cnv` 进入多轮对话：`> ` 提示符循环，模型回答后等待下一条输入
- 每回合只计算增量（O(回合新增 token)），长会话不退化
- 多轮历史经 chat 模板正确渲染（复用 `template::render_messages`）
- 完整继承现有采样/停止/UTF-8 安全输出能力（与 server 一致）

### 范围（MVP）

| 项 | 说明 |
|---|---|
| 多轮对话循环 | `--cnv`，含首条消息预置（`--cnv "首条"`） |
| 增量模板渲染 | `template::format_single`（差分法，对应 `common_chat_format_single`） |
| EOG / EOT 处理 | 模型自停 + 未达 EOG 时补 EOT，保持 KV 与模板 canonical 输出一致 |
| 命令 | `/exit` `/clear` `/regen` `/help` |
| 上下文溢出 | 截断历史 + 全量重灌（MVP）；KV 平移列为后续 |
| 单回合模式 | `-st/--single-turn`（脚本/测试用） |
| `--system` 系统提示词 | 对话模式新增 |

### 非目标（MVP）

- KV 上下文平移（llama.cpp `seq_rm`/`seq_add`）——见 §5.7 未来项
- 会话持久化 `--session`（KV state 序列化）——llama.cpp 有，minfer 列为 Phase 3 可选
- raw 交互模式（无模板时的 `--in-prefix`/`--in-suffix`/`-r` reverse-prompt 交互）——Phase 4 可选
- 多行粘贴 UI、`/read` `/glob` 文件注入、彩色 UI 细化

---

## 3. llama.cpp 参考分析

llama.cpp 有两代 CLI，设计取舍不同：

### 3.1 两代 CLI 对比

| | Legacy `examples/main/main.cpp`（独立进程） | 现行 `tools/cli/cli-context.cpp`（client-server） |
|---|---|---|
| 会话状态 | `chat_msgs: vector<common_chat_msg>` 内存历史；KV 缓存累积全部对话 | `messages: json` 数组；每回合把**全量历史**发给 server |
| 每回合计算量 | O(新增 delta)：增量模板渲染 + 只对新 token prefill/decode | O(全历史)：server 每请求重新 prefill（服务端另有 prompt-cache 优化） |
| 模板 | `common_chat_format_single`（**差分法**，见 §3.2） | 全量 `common_chat_templates_apply` |
| 回合终止 | EOG token / reverse-prompt（`-r`） | EOG / stop strings |
| 中断 | 第一次 Ctrl+C 打断生成（`need_insert_eot`），第二次退出 | Ctrl+C 停止生成 / 退出 |
| 会话持久化 | `--prompt-cache`：token 流 + KV state 落盘，加载时前缀匹配跳过重算 | 无 |
| 上下文溢出 | KV context shift（`llama_kv_self_seq_rm/add`，保留 `n_keep`） | 服务端 slot 处理 |
| 命令 | 无 slash 命令（Ctrl+C / EOF） | `/exit` `/regen` `/clear` `/read` `/glob` `/image` |
| 启动 | `-cnv` 有模板时自动开启；`-st` 单回合 | `--single-turn` |

### 3.2 增量渲染（差分法）——核心技巧

`common_chat_format_single(tmpls, past_msg, new_msg, add_ass)`（`common/chat.cpp:653`）：

```
fmt_past = apply(messages = past_msg,        add_generation_prompt = false)
fmt_new  = apply(messages = past_msg + new,  add_generation_prompt = add_ass)
delta    = fmt_new[fmt_past.len() ..]        // 假定 fmt_new 以 fmt_past 为前缀
if add_ass && fmt_past 以 '\n' 结尾: delta = "\n" + delta   // 尾部换行补偿
```

要点：

- **差分代替全量**：assistant 已生成的内容在 `fmt_past` 前缀里，diff 后不会重复喂给模型。
  每回合 KV 里追加的只是"新 user 消息 + generation prompt"。
- **尾部换行补偿**：模板在每条消息后输出 `\n`，它属于前缀尾部，被 diff 吃掉，
  但 canonical 文本需要它（EOG 之后、下一个 `<|im_start|>` 之前），所以手动补回。
  注意模型生成的 EOG（如 `<|im_end|>`）之后**不会**自己生成这个 `\n`。
- **前缀假设**：`fmt_new` 必须以 `fmt_past` 开头——对确定性模板（无 `now`/随机）恒成立。

### 3.3 回合循环（legacy main.cpp 状态机）

```
初始: chat_add_and_format("system", -sys) + chat_add_and_format("user", -p)
      → 全量渲染 → prefill 进 KV（位置 0..n）
解码: assistant_ss 累积生成文本
      EOG token 命中 → chat_add_and_format("assistant", assistant_ss) 记入历史
                      → is_interacting = true → LOG("\n> ") → 读输入
输入: user_inp = chat_add_and_format("user", buffer)   // 返回 delta
      if need_insert_eot: 先插入 EOT token             // 仅 Ctrl+C 打断时
      tokenize(delta) 追加到 embd_inp → 从现有 KV 继续
      assistant_ss 清空
溢出: n_past + embd.size() >= n_ctx → seq_rm + seq_add 平移（保留 n_keep，丢一半）
```

关键细节（`need_insert_eot`，main.cpp:42/70/910）：用户生成中途按 Ctrl+C 打断时置位；
下回合输入前把 EOT（无 EOT 用 EOS）写进 KV，保证历史与模板输出一致。

### 3.4 现行 client-server CLI 的借鉴点

`cli-context.cpp` 的 `messages` 数组 + 回合循环本身很简洁，值得借鉴的是 **UX 层**：

- slash 命令：`/exit` `/regen`（删最后一条 assistant 消息后重发）、`/clear`（清空并重放 system）
- `--single-turn`：`-st` 等价物
- Ctrl+C：生成中第一次打断、空闲时退出（`cli.cpp` 的 signal_handler）
- banner：命令列表提示（`/exit or Ctrl+C`、`/regen`、`/clear`…）

> 注意：现行 llama.cpp CLI 的 `/regen` 之所以简单，是因为每回合全量重发历史、服务端重新 prefill。
> minfer 走追加式 KV 时 `/regen` 需要 KV 回滚（见 §5.6）——这正是本方案与现行 llama.cpp CLI
> 的关键差异点。

---

## 4. minfer 现状与可复用件

### 4.1 现状（`src/main.rs`）

```
读 prompt（参数或 stdin 一行） → get_chat_template(GGUF) → render_template(单 user 消息)
→ encode → model.forward() prefill → 解码循环（penalties → top-k → top-p → temp）
→ stop 串字节级匹配 → 打印 → 退出
```

单发流程与多轮对话只差"循环 + 历史 + 增量渲染"，解码循环本体可直接复用。

### 4.2 可复用件（已实现，均有测试）

| 组件 | 位置 | 用途 |
|---|---|---|
| `render_messages(template, messages, add_gen_prompt, bos)` | `src/template.rs` | 多消息渲染 + ChatML fallback（server 在用）——差分法的两个渲染原语 |
| `forward_cached(model, tokens, positions, n_out, n_ctx, &mut GraphCache)` | `src/models/qwen2/graph.rs:276` | 位置寻址、KV 区域跨重建持久、显式 n_ctx——**追加式 KV 的基础** |
| `GraphCache::new()` | `src/graph/cache.rs` | 每会话独立 cache（server `slot.rs` 同款用法） |
| `sampler::{sample_with_penalties, recent_window, match_stop_suffix}` | `src/sampler.rs` | 采样链 / 惩罚窗口 / stop 串后缀匹配 |
| `Tokenizer::{encode, decode_bytes, complete_utf8_prefix_len}` | `src/tokenizer.rs` | 字节级解码，UTF-8 安全输出 |
| `is_stop_token`（eos / im_end） | `src/main.rs:530` | EOG 检测 |
| server 解码循环模式 | `src/server/chat.rs` | penalties/stop/UTF-8 处理的参考实现（rev-5 修正：stop 匹配跑**全量**字节流） |

### 4.3 缺口

1. **CLI 用 `forward()`（进程全局 cache + max_seq_len）**——会话模式必须换成
   `forward_cached(..., params.n_ctx, &mut conv.cache)`：一是遵守 `--n-ctx`，
   二是内存（max_seq_len 的 f32 KV 对 0.5B 是 ~3 GB，n_ctx=4096 是 ~0.4 GB）。
2. **无 `format_single`（差分渲染）**——`render_messages` 是全量原语，缺增量封装。
3. **无会话状态 / 交互循环 / slash 命令**。
4. **无 EOT 插入、溢出处理、`/regen` 回滚**。

---

## 5. 目标架构

### 5.1 总体策略：追加式 KV + 增量渲染（llama.cpp legacy 路线）

选择理由：

1. minfer 图路径的 KV 区域**按位置寻址且跨图重建持久**（`forward_cached`），追加天然成立，
   不需要任何图结构改动（llama.cpp legacy 的 n_past 自由与 minfer 的"位置是数据"同构）。
2. 每回合只算增量：长会话成本 O(回合增量)，而不是每回合 O(全历史) 重 prefill。
3. minfer CLI 是独立单进程（无 server 层），与 legacy `main.cpp` 架构一致，
   而现行 llama.cpp CLI 的"全量重发"是因为它本来就是 server 客户端。

### 5.2 数据结构（新模块 `src/conversation.rs`）

```rust
/// 一次多轮对话会话。整个会话的 token 流累积在 KV 区域中。
pub struct Conversation {
    /// 已记录消息（role, content）。与 server 的 ChatMessage 同构；null content 支持。
    pub messages: Vec<(String, Option<String>)>,
    /// 会话私有 cache：KV 区域 + 图分配器（n_ctx = params.n_ctx，不是 max_seq_len）。
    pub cache: GraphCache,
    /// KV 中完整的 token 流（prefill delta + 生成 + 手动插入的 EOT）。
    /// 用于：惩罚窗口 seeding、/regen 回滚、一致性校验（debug）。
    pub stream_tokens: Vec<u32>,
    /// 下一个写入位置（== stream_tokens.len() 的强不变量）。
    pub current_pos: usize,
    /// 当前 assistant 回合的起始位置（/regen 回滚点 = 本回合 delta 第一个 token 的位置）。
    pub turn_pos: usize,
    pub rng: StdRng,
    /// 惩罚窗口；每回合开始时从 stream_tokens 尾部取 64（llama.cpp repeat_last_n）。
    pub prev_tokens: Vec<u32>,
    /// stop 串匹配用完整生成字节流（server rev-5 语义：全流后缀匹配）。
    pub pending_bytes: Vec<u8>,
    /// 跨 token 的半截 UTF-8 字符（避免 U+FFFD）。
    pub incomplete_utf8: Vec<u8>,
    /// 模板 EOG：im_end，缺省用 eos（与 is_stop_token 一致）。
    pub eot_token: u32,
    /// 上回合未以 EOG 结束 → 下回合输入前先写 EOT。
    pub need_insert_eot: bool,
    /// tokenizer.chat_template；None → ChatML fallback 渲染。
    pub template: Option<String>,
    pub bos_text: String,
}
```

### 5.3 增量渲染：`template::format_single`

`src/template.rs` 新增，完全对应 `common_chat_format_single`：

```rust
pub fn format_single(
    template: &str,
    messages: &[(String, Option<String>)],   // 已记录历史
    new_msg: (String, Option<String>),
    add_generation_prompt: bool,
    bos_token: &str,
) -> String {
    let fmt_past = if messages.is_empty() { String::new() } else {
        render_messages(template, messages, false, bos_token)
    };
    let mut all = messages.to_vec();
    all.push(new_msg);
    let fmt_new = render_messages(template, &all, add_generation_prompt, bos_token);
    let mut out = String::new();
    if add_generation_prompt && !fmt_past.is_empty() && fmt_past.ends_with('\n') {
        out.push('\n');                       // 尾部换行补偿（§3.2）
    }
    if fmt_new.starts_with(&fmt_past) {
        out.push_str(&fmt_new[fmt_past.len()..]);
    } else {
        // 前缀失败（模板非确定性）→ 返回全量并置位，调用方走全量重灌兜底
        out = fmt_new;
        // 通过返回值或 Result 通知 prefix_mismatch
    }
    out
}
```

### 5.4 KV 一致性不变量（本方案正确性的基石）

> **不变量**：任意时刻，KV 中位置 `0..current_pos` 的内容 ==
> `tokenize(render(messages, add_generation_prompt = false))` 的 token 前缀。

达成方式：

1. 每回合 delta 由差分法得到（其 token 化内容 == canonical 文本的新增部分）；
2. 回合未以 EOG 结束 → 下回合前先写 EOT（`forward([eot], [current_pos])`），
   保证 assistant 消息在 KV 中以模板规定的 `<|im_end|>` 结尾；
3. 已知偏差：BPE 跨边界合并可能使 `tokenize(fmt_past) + tokenize(delta) != tokenize(fmt_new)`
   （llama.cpp 同款限制）。对 ChatML / Llama3 模板实测通常满足；测试中 assert，生产中容忍
   （模型对个别边界 token 差异鲁棒，llama.cpp 即如此）；
4. 兜底：若 `format_single` 前缀失败或 `debug_assert` 一致性被破坏 → 全量重灌
   （重置 cache → 渲染全历史 → 全量 prefill），功能不丢、只损失一次 prefill 时间。

> 该不变量同时让未来功能变简单：溢出截断（§5.7）、`/regen` 回滚、会话重放（§5.8）
> 都只需"重渲染同一消息列表"，token 流必然与 KV 对齐。

### 5.5 回合生命周期

```
┌─────────────────────────────────────────────────────────────┐
│ 初始化（--cnv）                                              │
│  system（--system 或模板默认）+ 首条 user → render_messages  │
│  → tokenize → prefill(位置 0..n) → current_pos = n          │
│  无首条 prompt → 直接进入等待输入                              │
└─────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────┐
│ 用户回合                                                      │
│ 1. 读输入（readline；空行跳过；/cmd 分发）                     │
│ 2. if need_insert_eot: forward([eot],[current_pos])          │
│    current_pos += 1; stream_tokens.push(eot)                 │
│ 3. delta = format_single(tmpl, messages, ("user", input),    │
│                          add_gen_prompt=true, bos)           │
│ 4. 溢出检查: current_pos + len(tokenize(delta)) > n_ctx      │
│    → §5.7 截断 + 重灌                                        │
│ 5. prev_tokens = recent_window(stream_tokens, 64)            │
│    turn_pos = current_pos                                    │
│ 6. prefill delta（位置 [current_pos..]）                      │
│ 7. 解码循环（同现 main.rs：penalties → top-k → top-p → temp；│
│    stop 串全流字节级匹配；UTF-8 安全输出；EOG 检测）            │
│    生成文本按字节累积 → String（assistant_text）               │
│ 8. 结束原因分支：                                            │
│    a) EOG 命中   → messages.push(("assistant", text))        │
│                    need_insert_eot = false                   │
│    b) stop 串 / n_predict 耗尽 / Ctrl+C 打断                 │
│                  → messages.push(("assistant", text))        │
│                    need_insert_eot = true   // 未达 EOG      │
│ 9. 打印 "\n> "，回到 1                                       │
└─────────────────────────────────────────────────────────────┘
```

解码循环直接沿用现有 `main.rs` 循环的骨架，改动点：

- `forward()` → `forward_cached(..., params.n_ctx, &mut conv.cache)`；
- 初始 `positions` 从 `current_pos` 起（不是 0）；
- `prev_tokens` 每回合重灌（`recent_window(stream_tokens, 64)`），解码中随生成继续维护；
- assistant 文本字节累积（`decode_bytes` + `incomplete_utf8`），回合结束转 String 入历史。

### 5.6 命令与 UX

| 命令 | 行为 |
|---|---|
| `/exit` `/quit` | 退出（EOF / Ctrl+D 同） |
| `/clear` | 清空历史与 KV：`messages` 清空（保留 system）、换新 `GraphCache`、`stream_tokens`/`current_pos` 归零 |
| `/regen` | 重生成最后一条 assistant 回答：pop 最后一条 assistant 消息 → `current_pos = turn_pos`（位置寻址区域直接回退指针，**无需任何数据搬移**——区域后半段是陈旧数据但永远不会被读）→ 重新走本回合（step 5 起） |
| `/help` | 帮助 |
| （Phase 3）`/save <file>` `/load <file>` | 会话持久化（§5.8） |

UX（对齐 llama.cpp）：

- 提示符 `> `（assistant 回合结束打印 `\n> `）；
- 颜色：assistant 输出 / 用户输入 / 提示符三色，`--color auto`（tty 检测）；
- Ctrl+C：生成中第一次 → 打断（`need_insert_eot = true`）；空闲时 → 退出（exit 130）；
- `-mli/--multiline-input`：多行输入（MVP 可简化为空行提交；`\` 续行属细化项）。

> `/regen` 回滚的正确性依赖 §5.4 不变量：KV 回退到 `turn_pos` 后，重放同一个 user 消息
> delta 的 token 流与先前完全一致（内容相同 → 相同位置写入相同 KV 区域），模型视角与
> 上次完全一致，只是采样重新开始。

### 5.7 上下文溢出

**MVP（推荐）：截断 + 全量重灌**

1. 保留 system（若有）；从 `messages` 头部丢弃最旧的非 system 回合，
   直到 `渲染剩余历史 + 新 delta` 的 token 数 ≤ n_ctx；
2. 换新 `GraphCache`、`stream_tokens` 清空、`current_pos = 0`；
3. `render_messages(剩余历史, add_gen_prompt=false)` 全量 prefill；
4. 继续正常回合；打印警告 `<<context full: dropped oldest N turns>>`。

成本：一次 O(剩余历史) prefill（0.5B @ n_ctx=4096 秒级），MVP 可接受。

**未来：KV 上下文平移（llama.cpp `seq_rm`/`seq_add`）**
minfer 中位置是数据、KV 区域是每层连续 f32 数组 → 平移 = 每层区域 memmove + `current_pos`
调整，**无图结构改动**。但需要给 allocator 加"区域平移"原语（Phase 4 可选），并注意
与 `CParams.n_ctx` 边界检查的配合。

**兜底**：截断后仍放不下（单条消息超长）→ 报错退出。

### 5.8 会话持久化（Phase 3 可选）

- `--session <file>`：`messages` 以 JSON 落盘（`serde_json`，已是依赖；格式与 server
  `ChatMessage` 兼容：`[{"role": ..., "content": ...}, ...]`）；
- 加载后**全量重灌**（不序列化 KV state——llama.cpp 存 token 流 + KV state 可秒恢复，
  minfer MVP 不做；因 §5.4 不变量，重灌结果与继续会话一致，只是多一次 prefill）；
- llama.cpp legacy 的"加载会话 + token 前缀匹配跳过重算"列为未来项。

### 5.9 CLI 参数（新增）

```
--cnv, --conversation      启用多轮对话（默认 off，保持单发用法兼容）
-st,  --single-turn        单回合后退出（脚本 / 测试）
--system <STR>             系统提示词（对话模式；默认用模板缺省或 ChatML 缺省）
-mli, --multiline-input    多行输入（空行提交）
--color [on|off|auto]      输出颜色（默认 auto = tty 检测）
--session <FILE>           会话存取（Phase 3）
（Phase 4 可选） -r/--reverse-prompt、--in-prefix、--in-suffix（raw 交互）
```

- `--cnv` 与 `--no-template` 互斥：对话模式必须走模板或 ChatML fallback；
  `--cnv --no-template` → 报错（raw 交互模式是 Phase 4）。
- 默认值决策：llama.cpp 是"有模板自动开启"（`COMMON_CONVERSATION_MODE_AUTO`）；
  minfer **保持显式 `--cnv`**，避免破坏 README 中 `minfer <model> "hello"` 的单发契约
  （auto 作为备选决策点，见 §9）。
- 顺带（Phase 0 可选清理）：单发路径也切到 `forward_cached(..., params.n_ctx, ...)`，
  KV 内存从 `max_seq_len` 降到 `n_ctx`（0.5B：~3 GB → ~0.4 GB）；位置 < n_ctx 时 logits
  不变，安全。

---

## 6. 与 llama.cpp 的对照

| llama.cpp | minfer 方案 | 差异说明 |
|---|---|---|
| `common_chat_format_single` 差分渲染 | `template::format_single`（§5.3） | 逐行对应：两次 render + diff + 尾部 `\n` 补偿 |
| `chat_add_and_format` + `embd_inp` 追加 | `Conversation::stream_tokens` + `forward_cached` 追加 | 同构：KV 持续累积 |
| EOG → 记 assistant 历史 → `\n> ` 读输入 | 同（§5.5） | — |
| `need_insert_eot`（仅 Ctrl+C） | **统一**：任何未达 EOG 的结束都置位（§5.5-b） | 比 llama.cpp 更严格，保持 §5.4 不变量；代价是一次 EOT decode |
| `common_sampler_reset` + prompt 消费喂采样器 | `prev_tokens = recent_window(stream_tokens, 64)` 每回合重灌 | 等价效果，更简单 |
| KV context shift（`seq_rm/add`） | MVP：截断 + 重灌；未来：区域 memmove（§5.7） | 实现差异见 §5.7 |
| `--prompt-cache`（token + KV state） | Phase 3：仅 messages JSON + 重灌（§5.8） | KV 序列化后续项 |
| 现行 CLI 的 `/exit /regen /clear` | 同 + `/help`（§5.6） | `/regen` 走 KV 回退指针，见 §5.6 注 |
| `-cnv` 自动开启 | `--cnv` 显式开启（§5.9） | 向后兼容 |
| `-st/--single-turn` | `-st` | 同 |

---

## 7. 分阶段实现计划

### Phase 0：模板差分渲染（`src/template.rs`）

**Status: implemented (rev 1).** `format_single(template: Option<&str>, messages, new_msg,
add_generation_prompt, bos_token) -> FormattedDelta{text, prefix_matched}`；`template=None`
走 ChatML fallback。测试见 §8.8（6 个单测，`cargo test format_single`）。

- 新增 `format_single`（§5.3）：差分 + 尾部 `\n` 补偿 + 前缀失败通知。
- （可选清理）单发路径切 `forward_cached(..., params.n_ctx, ...)`。
- 测试：
  - ChatML / Llama3 风格模板：`format_single(past, ("user", x))` 的 diff 正确；
  - 补偿逻辑：past 以 `\n` 结尾 vs 不以；
  - 前缀失败兜底：非确定性模板返回全量 + 标记；
  - token 级断言：`tokenize(fmt_past) + tokenize(delta) == tokenize(fmt_new)`（对标准模板）→ **延后到 L1.5 KV 等价测试**。

### Phase 1：会话核心（新模块 `src/conversation.rs`）

- `Conversation` 结构（§5.2）+ 回合循环（§5.5）。
- **回合循环必须面向 `Engine` 抽象实现**（`trait Engine { fn forward(&mut self, tokens, positions, n_out) -> Vec<f32> }`，真实实现包 `forward_cached`）——这是 L1 mock 测试的前提（§8.2）。
- EOG / EOT 处理、`need_insert_eot` 统一插入。
- `prev_tokens` 每回合重灌；`stream_tokens` 全程维护。
- 测试：
  - 状态机单测（mock 无模型）：EOT 插入时机、messages 累积、turn_pos 记录；
  - 一致性单测：构造 token 流模拟，验证 §5.4 不变量；
  - 真实模型集成冒烟：stdin 管道喂 2–3 回合（Qwen2.5-0.5B），验证无 U+FFFD、
    stop 串截断、EOG 结束、位置单调不越界。

### Phase 2：CLI 接线与 UX（`src/main.rs`）

- 参数：`--cnv` `-st` `--system` `-mli` `--color`；`--cnv` 与 `--no-template` 互斥校验。
- slash 命令：`/exit` `/clear` `/regen` `/help`。
- 提示符与颜色（tty 检测）；Ctrl+C 打断（依赖信号处理——见 §9 决策点 3）。
- **flush 纪律**：打印 `> ` 提示符与每段输出后必须 flush stdout——管道化测试的前置条件（否则 L2 死锁，见 §8.3）。
- 测试：脚本化 stdin 会话 golden 测试（greedy + 固定 seed）；
  `/regen` 后回答变化（seed 不同）且不崩；`/clear` 后位置归零。

### Phase 3：上下文溢出 + 会话持久化

- 截断 + 重灌（§5.7）；`--session` 存取（§5.8）。
- 测试：
  - 溢出截断纯函数（丢最旧回合直到可容纳、保留 system）；
  - 溢出后继续对话无 KV 越界 panic；
  - session round-trip：save → load → 继续对话，行为与不落盘一致。

### Phase 4（可选）：增强

- KV 上下文平移（allocator 区域 memmove）；
- raw 交互模式（`-r`/`--in-prefix`/`--in-suffix`）；
- `/read` `/glob` 文件注入；
- server / CLI 解码循环统一（抽 `src/generate.rs` 共享，server 后续适配）。

---

## 8. 测试策略

> **llama.cpp 自身并不自动化测试交互模式**：CI（`ci/run.sh`）只对 CLI 跑 `-no-cnv` 一发冒烟
> （`-n 64 --ignore-eos -p "..."` + `time`），会话/交互功能靠手工验收（legacy `main.cpp` 与现行
> `tools/cli` 均如此）。因此"交互会话怎么测"是 minfer 超出 llama.cpp 的部分，本节从零设计。

### 8.1 分层测试架构

| 层 | 名称 | 内容 | 依赖 |
|---|---|---|---|
| L0 | 纯单测 | `format_single` 差分/换行补偿/前缀兜底、命令解析、溢出截断纯函数 | 无模型、无 IO |
| L1 | 状态机单测 | mock `Engine` 驱动 `Conversation` 回合循环，逐字段断言状态（§8.2） | 无模型 |
| L1.5 | KV 等价测试 | 追加式会话 vs 一次性全量重渲染，比较层 0 KV 区域内容（§8.4） | 真实模型 + cache 内部访问 |
| L2 | 进程级集成 | 真实二进制 + 管道 stdin 脚本化输入，断言输出与退出码（§8.3） | 真实模型 |
| L3 | golden 快照 | 固定后端 + greedy + seed，逐字节对比（§8.3） | 固定模型/量化 |
| L4 | 交叉验证 | CLI（追加式）vs server（全量重渲染）行为一致（§8.4） | 真实模型 |
| L5 | 边界/鲁棒性 | 非 UTF-8、`\r\n`、超长行、小 `n_ctx` 截断、互斥参数报错（§8.6） | 真实模型 |
| L6 | 信号测试 | SIGINT 打断/退出语义（unix）（§8.5） | dev-dep `libc` |
| L7 | PTY 测试 | tty 相关：`--color auto`、`-mli`、提示符同步（§8.5） | `script` 或 dev-dep `portable-pty` |
| L8 | 性能冒烟 | 2–3 回合会话 CI 内完成 + 增量性断言（§8.4） | 真实模型 |

> 运行期"5 个 crate"约束只约束**运行时依赖**；dev-dependencies（`libc`、可选 `portable-pty`）
> 不受限——测试基础设施不应绑架二进制。

### 8.2 L1：状态机单测（Engine 抽象）

回合循环必须面向 `Engine` 抽象实现（Phase 1 要求，§7），而不是直接调用 `forward_cached`：

```rust
pub trait Engine {
    /// 返回 n_out*nv logits；真实实现包 Qwen2Graph::forward_cached。
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32>;
}
```

单测注入可编程 mock（构造器给定 logits 序列 / token 输出序列）：

| Mock 行为 | 覆盖路径 |
|---|---|
| 恒返回 EOG 的 logits | 正常回合结束：messages 追加 assistant、`need_insert_eot=false`、回到输入 |
| 固定 token 序列后 EOG | stop 串截断、跨 token UTF-8（`incomplete_utf8`）、assistant 文本字节累积 |
| 一直非 EOG 直到 `n_predict` 耗尽 | `need_insert_eot=true`、下回合 EOT 先写（§5.4 不变量） |
| `/regen` 场景 | `turn_pos` 回退 + `stream_tokens` 截断 + 同一 user delta 重放 |
| `/clear` 场景 | 状态归零、KV 换新 |

每步断言 `stream_tokens` / `current_pos` / `turn_pos` / `messages` 的变化——
**不依赖真实模型即可全覆盖状态机**，直接验证 §5.4 不变量（`stream_tokens` 即 KV 内容的
宿主侧镜像）。

### 8.3 L2/L3：进程级集成 + golden（核心答案）

用 `std::process::Command` 启动编译产物，stdin 走管道写入**脚本化会话**，stdout/stderr 读至 EOF：

```rust
// tests/conversation_cli.rs
let mut child = Command::new(env!("CARGO_BIN_EXE_minfer"))
    .args(["--cnv", "--greedy", "--seed", "42", "--color", "off",
           "--n-ctx", "512", MODEL])
    .env("MINFER_DISABLE_MPS", "1")          // 固定 CPU 后端（golden 后端相关）
    .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
    .spawn().unwrap();
// 写入脚本化输入（可一次性写完，也可回合间 sleep 模拟节奏）：
//   "hi\nwhat is 2+2?\n/exit\n"
// 读输出到 EOF → 断言：含 2 段 assistant 输出、无 U+FFFD、exit code 0。
```

要点：

- **确定性**：`--greedy --seed 42`（temp=0 → argmax，确定）。**后端必须固定**——CPU 与 Metal
  logits 差 ~1e1（AGENTS.md 记录），greedy 可能选到不同 token；golden 默认按 CPU 路径生成
  （`MINFER_DISABLE_MPS=1`），Metal 只跑非 golden 冒烟（非空、无 U+FFFD）。
- **golden 快照**：`tests/fixtures/conversation/*.stdout` 逐字节对比；bless 脚本
  （`UPDATE_SNAPSHOTS=1 cargo test --test conversation_cli`）受控更新。golden 对模型文件与量化
  敏感——CI 固定用 `download::resolve` 缓存的 qwen2.5-0.5b-instruct-q4_0（与现有冒烟一致）。
- **防死锁/防挂死**：交互程序必须遵守 flush 纪律（§7 Phase 2）——打印 `> ` 后不 flush 会让
  管道测试互相等待。测试侧用 `try_wait` + 截止时间循环（std-only，不引 `wait-timeout`），
  超时 `kill()` + 断言失败。
- **一次性写完 vs 节奏驱动**：确定性输出不受输入节奏影响，大多数断言用例一次性写完输入即可；
  需要"等 `> ` 提示符出现再发下一行"的用例走 L7（§8.5）。

### 8.4 L1.5 / L4 / L8：三个针对性正确性测试

**KV 等价测试（最强正确性断言）**：同一多回合历史跑两条路径——
(a) 追加式：逐回合增量 prefill/decode；(b) 一次性：`render(messages, false)` 全量 prefill。
比较层 0 的 K/V 区域内容（in-module 测试经 allocator 宿主访问，或 `MINFER_GRAPH_DUMP`
写出的 `kv0_prefill.f32`）。
期望 bit-identical 或 ≤1e-6：同一 token 同位置的 K/V 分别由 prefill 图与 decode 图算出，
图结构差异（fused QKV 等）已验证不改变数值（AGENTS.md：fused/unfused decode bit-identical）。
**这是把 §5.4 不变量变成可执行断言的直接方式**——比比较最终文本强得多。

**增量性断言（防 O(n²) 回归）**：对话模式每回合 prefill 在 stderr 打印 token 数
（"Prefill: N tokens"）；测试断言第 K 回合的 N == 该回合 delta token 数，且 ≪ 全历史长度——
直接锁死"每回合只算增量"，防止未来某天被改成全量重渲染（行为正确但性能回退）。

**与 server 交叉验证**：同一 2 回合历史，CLI（追加式）vs server（全量重渲染，
`/v1/chat/completions`），greedy 下断言 finish_reason 一致、输出非空、无 U+FFFD。
行为级而非 bit 级（批处理/图差异），文档已声明。

### 8.5 L6 / L7：信号与 PTY

- **SIGINT（`#[cfg(unix)]`）**：dev-dep `libc`，`kill(pid, SIGINT)`。
  用例：
  - 生成中第一次 SIGINT → 不退出、回到 `> `、下一回合先插 EOT（行为断言：能继续、回答正常、不崩）；
  - 空闲时第二次 SIGINT → exit 130。
- **PTY**：`--color auto` / `-mli` / 提示符同步依赖 isatty。两个选项：
  (a) 系统 `script -q /dev/null <cmd>` 给子进程分配 pty（零依赖，macOS/Linux 自带）；
  (b) dev-dep `portable-pty`（纯 Rust）写小型 harness。
  MVP 只覆盖 2–3 个用例：颜色 ANSI 码出现、`-mli` 空行提交、提示符同步下发输入。

### 8.6 L5：边界 / 鲁棒性用例矩阵

| 场景 | 输入 | 断言 |
|---|---|---|
| 空行跳过 | `\nhi\n` | 不产生空回合 |
| EOF 退出 | 首回合即关闭 stdin（Ctrl+D） | exit 0 |
| `\r\n` 行尾 | `hi\r\n/exit\r\n` | 正常（trim） |
| 非 UTF-8 字节 | `\xff\xfe\n` | 不 panic、按字节处理或明确报错 |
| 超长行 / 中文 / emoji | 多行混合 | 输出无 U+FFFD、无 panic |
| `/regen` 无历史 | 首回合 `/regen` | 提示"无可重生成"，继续 |
| `/clear` 后继续 | 2 回合后 `/clear` + 新回合 | 位置归零、历史清空、仍能回答 |
| 小 `n_ctx` 截断 | `--n-ctx 64` + 长首条 + 后续回合 | 触发截断警告、无 KV 越界 panic、可继续 |
| `--cnv --no-template` | — | 明确报错退出（互斥） |
| `-st` 单回合 | `-st` + 1 条输入 | 一回合后退出 |
| `--system` | `--system "..."` | 首条渲染含 system（golden 可验证） |

### 8.7 手工验收清单（自动化覆盖不到）

真实终端观感：提示符/换行/颜色渲染、终端内 Ctrl+C 观感、多行粘贴、`\` 续行、生成中按键
回显。作为发布前 checklist，不进 CI。

### 8.8 汇总

| 层级 | 内容 |
|---|---|
| L0 单测（template） | `format_single` 差分 / 换行补偿 / 前缀兜底 / token 级一致性（§7 Phase 0） |
| L1 单测（conversation） | `Engine` mock：EOT、turn_pos、regen 回退、截断纯函数（§8.2） |
| L1.5 KV 等价 | 追加式 vs 全量重渲染，层 0 KV 比较（§8.4） |
| L2/L3 集成 | 管道 stdin 多回合冒烟 + golden 快照（§8.3） |
| L4 交叉验证 | CLI vs server 行为一致（§8.4） |
| L5 边界矩阵 | §8.6 |
| L6/L7 信号与 PTY | SIGINT 语义、tty 用例（§8.5） |
| 回归 | 单发模式行为不变（README 用法）；`cargo test` 全绿 |
| 溢出续聊 / `--session` round-trip | Phase 3 追加（行为断言） |

---

## 9. 开放问题与决策点

1. **`--cnv` 显式开启 vs 有模板自动开启**（llama.cpp 默认 auto）。
   推荐：显式开启（MVP），文档注明未来可加 auto。破坏性小、与现有用法不冲突。
2. **溢出策略**：截断 + 重灌（推荐，MVP）vs KV 平移 vs 直接报错。
   推荐截断 + 重灌：实现简单、行为可预期；平移留给 Phase 4。
3. **Ctrl+C 打断**是否入 MVP：需要信号处理（项目当前 5 个 crate 无 ctrlc/libc；
   新增依赖或 `#[cfg(unix)]` unsafe signal handler）。
   推荐：MVP 先不做打断（EOF 退出即可），打断语义 Phase 2 末尾补。
4. **解码循环是否与 server 共享**（抽 `src/generate.rs`）：MVP 复制现循环骨架（低风险），
   统一重构放 Phase 4。
5. **会话持久化是否入范围**：推荐 Phase 3 可选（只存 messages JSON，不存 KV）。
6. **无模板时 `--cnv` 的行为**：ChatML fallback 渲染（推荐，恒可用）vs 报错。
7. **EOT 插入策略**：统一"未达 EOG 即插"（推荐，保证不变量）vs 仅 Ctrl+C（llama.cpp 原样）。
