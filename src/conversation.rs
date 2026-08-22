//! CLI 多轮对话会话核心（docs/CLI-CONVERSATION-PLAN.md）。
//!
//! 策略：**追加式 KV + 增量模板渲染**（llama.cpp legacy `examples/main` 路线）。
//! 整个会话的 token 流累积在 KV 区域中（位置寻址、跨图重建持久），每回合只对
//! 新消息的 delta 做 prefill/decode——长会话成本 O(回合增量)，而非 O(全历史)。
//!
//! 关键抽象：
//! - [`Engine`]：推理后端（`forward` + `reset_cache`）。真实实现 [`GraphEngine`]
//!   持有 `ModelDef` 引用 + 会话私有 `GraphCache` + `n_ctx`；mock 实现使状态机
//!   可在无模型环境下被完全测试（§8.2 L1）。
//! - [`TokenCodec`]：字节级编码/解码（`Tokenizer` 的 trait 化），mock 友好。
//! - [`Conversation`]：只持有逻辑状态（消息、token 流镜像、采样器窗口、回滚点），
//!   **不含 cache**——cache 归 `GraphEngine`，避免 `&mut self` 与 `&mut cache`
//!   的借用冲突，也让 `Conversation` 完全可测。
//!
//! KV 一致性不变量（§5.4）：每回合 EOT 插入之后、delta 追加之前，
//! `stream_tokens`（KV 的宿主侧镜像）== `tokenize(render(messages, false))`
//! 的 token 前缀（可能缺末尾模板换行）。`/regen` 回滚（`turn_pos` 指针回退）与
//! 全量重灌都依赖它。

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::graph::cache::GraphCache;
use crate::models::ModelDef;
use crate::sampler;
use crate::template::{self, format_single};
use crate::tokenizer::Tokenizer;

/// 推理后端抽象：L1 mock 测试的前提（§8.2）。
pub trait Engine {
    /// 返回 n_out*nv logits（n_out=1 时即最后/唯一 token 的 logits）。
    /// 真实实现包 `ModelDef::forward_graph_cached`。
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32>;
    /// 丢弃全部 KV 状态（全量重灌 / `/clear` 时调用）。
    fn reset_cache(&mut self);
}

/// 字节级编码/解码抽象：`Tokenizer` 的 trait 化（mock 友好）。
pub trait TokenCodec {
    fn encode(&self, text: &str) -> Vec<u32>;
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8>;
}

impl TokenCodec for Tokenizer {
    fn encode(&self, text: &str) -> Vec<u32> {
        Tokenizer::encode(self, text)
    }
    fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
        Tokenizer::decode_bytes(self, ids)
    }
}

/// 真实推理引擎：模型引用 + 会话私有 `GraphCache` + `n_ctx`。
/// cache 在此（而非 `Conversation` 内），保证 Engine 可被 mock 替换（§8.2）。
pub struct GraphEngine<'a> {
    model: &'a dyn ModelDef,
    cache: GraphCache,
    n_ctx: usize,
}

impl<'a> GraphEngine<'a> {
    pub fn new(model: &'a dyn ModelDef, n_ctx: usize) -> Self {
        Self { model, cache: GraphCache::new(), n_ctx }
    }
}

impl Engine for GraphEngine<'_> {
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32> {
        self.model
            .forward_graph_cached(tokens, positions, n_out, self.n_ctx, &mut self.cache)
    }
    fn reset_cache(&mut self) {
        self.cache = GraphCache::new();
    }
}

/// 会话构造参数。
pub struct ConversationSpec {
    /// `tokenizer.chat_template`；None → ChatML fallback 渲染。
    pub template: Option<String>,
    pub bos_text: String,
    /// EOG 集合（eos + im_end）。
    pub eog: Vec<u32>,
    /// 未达 EOG 结束时插入的 token（im_end，缺省 eos）。
    pub eot: u32,
    pub seed: u64,
    pub n_ctx: usize,
    /// `--system` 提示词（作为首条 system 消息）。
    pub system_prompt: Option<String>,
}

/// 每回合采样/停止参数（从 CLI 的 GenParams 映射而来）。
pub struct TurnParams {
    pub n_predict: usize,
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub stop_strings: Vec<String>,
}

/// 一回合生成结果。
// text / stopped_by_* 由流式 emit 路径输出，字段保留为结构化结果（测试用）。
#[derive(Debug)]
pub struct TurnOutcome {
    /// assistant 生成文本（stop 串截断后；不含 EOG）。
    #[allow(dead_code)]
    pub text: String,
    #[allow(dead_code)]
    pub stopped_by_eog: bool,
    #[allow(dead_code)]
    pub stopped_by_string: bool,
    /// n_predict 耗尽或上下文填满。
    pub hit_n_predict: bool,
    /// 本回合 prefill 的 token 数（增量性断言用；fallback 路径为全量长度）。
    pub prefill_tokens: usize,
    pub tokens_generated: usize,
    /// 上下文溢出时丢弃的旧回合数（0 = 未截断，§5.7）。
    pub dropped_turns: usize,
}

#[derive(Debug)]
pub enum ConvError {
    NothingToRegen,
    ContextFull { needed: usize, available: usize },
    EmptyInput,
}

impl std::fmt::Display for ConvError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ConvError::NothingToRegen => write!(f, "no assistant reply to regenerate"),
            ConvError::ContextFull { needed, available } => write!(
                f,
                "context full: need {needed} tokens, have {available} (--n-ctx)"
            ),
            ConvError::EmptyInput => write!(f, "input tokenizes to nothing"),
        }
    }
}

/// 一次多轮对话会话（逻辑状态；推理状态在 `Engine` 中）。
pub struct Conversation {
    /// 已记录消息（role, content）。与 server 的 ChatMessage 同构。
    pub messages: Vec<(String, Option<String>)>,
    /// KV 中完整 token 流的宿主侧镜像（prefill delta + 生成 + 手动 EOT）。
    pub stream_tokens: Vec<u32>,
    /// 下一个写入位置（== stream_tokens.len() 的强不变量）。
    pub current_pos: usize,
    /// 当前回合 delta 的起始位置（`/regen` 回滚点；0 = 需全量重渲染）。
    pub turn_pos: usize,
    pub rng: StdRng,
    /// 惩罚窗口；每回合开始时从 stream_tokens 尾部取 64。
    pub prev_tokens: Vec<u32>,
    /// EOG 集合。
    pub eog: Vec<u32>,
    /// 未达 EOG 时插入的 token。
    pub eot: u32,
    /// 上回合未以 EOG 结束 → 下回合输入前先写 EOT（§5.4 不变量）。
    pub need_insert_eot: bool,
    pub template: Option<String>,
    pub bos_text: String,
    pub n_ctx: usize,
}

const REPEAT_LAST_N: usize = 64;

impl Conversation {
    pub fn new(spec: ConversationSpec) -> Self {
        let messages = spec
            .system_prompt
            .map(|s| ("system".to_string(), Some(s)))
            .into_iter()
            .collect();
        Self {
            messages,
            stream_tokens: Vec::new(),
            current_pos: 0,
            turn_pos: 0,
            rng: StdRng::seed_from_u64(spec.seed),
            prev_tokens: Vec::new(),
            eog: spec.eog,
            eot: spec.eot,
            need_insert_eot: false,
            template: spec.template,
            bos_text: spec.bos_text,
            n_ctx: spec.n_ctx,
        }
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    /// 全量渲染当前 messages（含/不含 generation prompt）。
    fn render_full(&self, add_generation_prompt: bool) -> String {
        match &self.template {
            Some(t) => template::render_messages(t, &self.messages, add_generation_prompt, &self.bos_text),
            None => template::fallback_chatml_messages(&self.messages, add_generation_prompt),
        }
    }

    /// 重置 cache 并从零 prefill 给定 token 流（全量重灌路径）。
    /// 返回 prefill 的最后 token logits（首个采样的输入）。
    fn rehydrate_full(&mut self, engine: &mut dyn Engine, tokens: &[u32]) -> Vec<f32> {
        engine.reset_cache();
        self.stream_tokens.clear();
        self.current_pos = 0;
        self.turn_pos = 0;
        if tokens.is_empty() {
            return Vec::new();
        }
        let positions: Vec<usize> = (0..tokens.len()).collect();
        let logits = engine.forward(tokens, &positions, 1);
        self.stream_tokens.extend_from_slice(tokens);
        self.current_pos = tokens.len();
        logits
    }

    /// 启动会话：`Some(首条输入)` → 全量渲染 + prefill + 生成；None → 等待输入。
    /// (入口保留为完整 API；CLI 用 `user_turn` 走统一路径，`start` 由测试覆盖。)
    #[allow(dead_code)]
    pub fn start(
        &mut self,
        first_input: Option<&str>,
        decoder: &dyn TokenCodec,
        cfg: &TurnParams,
        engine: &mut dyn Engine,
        emit: &mut dyn FnMut(&[u8]),
    ) -> Result<Option<TurnOutcome>, ConvError> {
        let Some(input) = first_input else {
            return Ok(None);
        };
        self.messages.push(("user".to_string(), Some(input.to_string())));
        let full = self.render_full(true);
        let toks = decoder.encode(&full);
        if toks.is_empty() {
            return Err(ConvError::EmptyInput);
        }
        let logits = self.rehydrate_full(engine, &toks);
        self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
        let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
        out.prefill_tokens = toks.len();
        Ok(Some(out))
    }

    /// 追加一条 user 消息并生成 assistant 回复。
    pub fn user_turn(
        &mut self,
        input: &str,
        decoder: &dyn TokenCodec,
        cfg: &TurnParams,
        engine: &mut dyn Engine,
        emit: &mut dyn FnMut(&[u8]),
    ) -> Result<TurnOutcome, ConvError> {
        // 1. 上回合未达 EOG → 先补 EOT，保持 KV 与模板 canonical 输出一致（§5.4）。
        if self.need_insert_eot {
            let _ = engine.forward(&[self.eot], &[self.current_pos], 1);
            self.stream_tokens.push(self.eot);
            self.current_pos += 1;
        }

        // 2. 增量渲染（差分法，§5.3）。
        let delta = format_single(
            self.template.as_deref(),
            &self.messages,
            ("user".to_string(), Some(input.to_string())),
            true,
            &self.bos_text,
        );
        let delta_toks = decoder.encode(&delta.text);
        if delta_toks.is_empty() {
            return Err(ConvError::EmptyInput);
        }

        // 3. 前缀失败（模板非确定性）→ 全量重灌兜底（§5.4）。
        if !delta.prefix_matched {
            self.messages.push(("user".to_string(), Some(input.to_string())));
            let full = self.render_full(true);
            let toks = decoder.encode(&full);
            if toks.is_empty() {
                return Err(ConvError::EmptyInput);
            }
            let logits = self.rehydrate_full(engine, &toks);
            self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
            let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
            out.prefill_tokens = toks.len();
            return Ok(out);
        }

        // 4. 上下文溢出：丢弃最旧的非 system 回合 + 全量重灌（§5.7）。
        if self.current_pos + delta_toks.len() > self.n_ctx {
            self.messages.push(("user".to_string(), Some(input.to_string())));
            let dropped = self.drop_oldest_turns_until_fits(decoder);
            if dropped == 0 {
                // 只剩 [system?, 最后一条 user] 仍放不下：单条消息超长，报错。
                // 回滚：弹出刚推入的 user 消息，并复位 EOT 标记（若本回合已写 EOT，
                // 它已正确关闭上一回合，不应在下一回合重复插入）。
                self.messages.pop();
                self.need_insert_eot = false;
                return Err(ConvError::ContextFull {
                    needed: self.current_pos + delta_toks.len(),
                    available: self.n_ctx,
                });
            }
            let full = self.render_full(true);
            let toks = decoder.encode(&full);
            let logits = self.rehydrate_full(engine, &toks);
            self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
            let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
            out.prefill_tokens = toks.len();
            out.dropped_turns = dropped;
            return Ok(out);
        }

        // 5. 记录 user 消息、设置回滚点、prefill delta。
        self.messages.push(("user".to_string(), Some(input.to_string())));
        self.turn_pos = self.current_pos;
        let positions: Vec<usize> = (self.current_pos..self.current_pos + delta_toks.len()).collect();
        let logits = engine.forward(&delta_toks, &positions, 1);
        self.stream_tokens.extend_from_slice(&delta_toks);
        self.current_pos += delta_toks.len();
        // 惩罚窗口在 delta 追加**之后**重灌（llama.cpp 把 prompt token 喂进采样器窗口）。
        self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);

        let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
        out.prefill_tokens = delta_toks.len();
        Ok(out)
    }

    /// 重生成最后一条 assistant 回复：回滚到 `turn_pos`（位置寻址区域直接回退
    /// 指针，区域后半段是陈旧数据但永远不会被读）→ 重放最后一条 user 消息。
    pub fn regen_turn(
        &mut self,
        decoder: &dyn TokenCodec,
        cfg: &TurnParams,
        engine: &mut dyn Engine,
        emit: &mut dyn FnMut(&[u8]),
    ) -> Result<TurnOutcome, ConvError> {
        if self.messages.last().map(|m| m.0.as_str()) != Some("assistant") {
            return Err(ConvError::NothingToRegen);
        }
        self.messages.pop();
        self.stream_tokens.truncate(self.turn_pos);
        self.current_pos = self.turn_pos;
        self.need_insert_eot = false;

        let Some((_, Some(last_user))) = self.messages.last().cloned() else {
            return Err(ConvError::NothingToRegen);
        };

        // 首回合（或全量重灌后的回合）turn_pos == 0：回滚后 KV 为空，需全量渲染；
        // 其余回合只需重放该 user 消息的 delta。
        let toks = if self.turn_pos == 0 {
            let full = self.render_full(true);
            decoder.encode(&full)
        } else {
            let delta = format_single(
                self.template.as_deref(),
                &self.messages[..self.messages.len() - 1],
                ("user".to_string(), Some(last_user)),
                true,
                &self.bos_text,
            );
            decoder.encode(&delta.text)
        };
        if toks.is_empty() {
            return Err(ConvError::EmptyInput);
        }
        if self.current_pos + toks.len() > self.n_ctx {
            return Err(ConvError::ContextFull {
                needed: self.current_pos + toks.len(),
                available: self.n_ctx,
            });
        }

        let positions: Vec<usize> = (self.current_pos..self.current_pos + toks.len()).collect();
        let logits = engine.forward(&toks, &positions, 1);
        self.stream_tokens.extend_from_slice(&toks);
        self.current_pos += toks.len();
        // 惩罚窗口在 delta 追加**之后**重灌（与 user_turn 一致）。
        self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);

        let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
        out.prefill_tokens = toks.len();
        Ok(out)
    }

    /// `/clear`：清空历史与采样状态（保留 system 与 rng）。
    /// 注意：调用方必须同时 `engine.reset_cache()`。
    pub fn clear(&mut self) {
        let system = self.messages.iter().find(|m| m.0 == "system").cloned();
        self.messages = system.into_iter().collect();
        self.stream_tokens.clear();
        self.current_pos = 0;
        self.turn_pos = 0;
        self.prev_tokens.clear();
        self.need_insert_eot = false;
    }

    /// 上下文溢出：丢弃最旧的非 system 回合（user + 其 assistant 对），直到
    /// `render(messages, true)` 的 token 数 ≤ n_ctx。返回丢弃的回合数；
    /// 只剩 [system?, 最后一条 user] 仍放不下时返回 0（调用方报错）。
    fn drop_oldest_turns_until_fits(&mut self, decoder: &dyn TokenCodec) -> usize {
        let mut dropped = 0;
        loop {
            let full = self.render_full(true);
            if decoder.encode(&full).len() <= self.n_ctx {
                return dropped;
            }
            let Some(idx) = self.oldest_droppable_turn_start() else {
                return dropped;
            };
            self.messages.remove(idx);
            if idx < self.messages.len() && self.messages[idx].0 == "assistant" {
                self.messages.remove(idx);
            }
            dropped += 1;
        }
    }

    /// 最旧可丢弃回合的起点：最后一条 user 消息之前第一个 user 消息的位置。
    fn oldest_droppable_turn_start(&self) -> Option<usize> {
        let last_user = self.messages.iter().rposition(|m| m.0 == "user")?;
        (0..last_user).find(|&i| self.messages[i].0 == "user")
    }

    /// `--session`：messages 序列化为 OpenAI 风格 JSON 数组
    /// （与 server 的 ChatMessage 同构：`[{"role","content"},...]`，§5.8）。
    pub fn messages_to_json(&self) -> String {
        let arr: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        serde_json::to_string_pretty(&arr).unwrap_or_default()
    }

    /// 从 JSON 数组解析 messages（`--session` 加载；非法输入返回 None）。
    pub fn messages_from_json(json: &str) -> Option<Vec<(String, Option<String>)>> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let arr = v.as_array()?;
        arr.iter()
            .map(|m| {
                let role = m.get("role")?.as_str()?.to_string();
                let content = m.get("content").and_then(|c| c.as_str()).map(str::to_string);
                Some((role, content))
            })
            .collect()
    }

    /// 载入历史并**全量重灌** KV（§5.8：不序列化 KV state——因 §5.4 不变量，
    /// 重灌结果与继续会话一致，只是多一次 prefill）。
    pub fn load_history(
        &mut self,
        messages: Vec<(String, Option<String>)>,
        decoder: &dyn TokenCodec,
        engine: &mut dyn Engine,
    ) {
        self.messages = messages;
        self.need_insert_eot = false;
        self.turn_pos = 0;
        let full = self.render_full(false);
        let toks = decoder.encode(&full);
        let _ = self.rehydrate_full(engine, &toks);
    }

    /// 解码循环：从 `logits`（prefill 最后 token）开始逐 token 采样/解码，
    /// 直到 EOG / stop 串 / n_predict / 上下文填满。
    fn generate_assistant_with_logits(
        &mut self,
        decoder: &dyn TokenCodec,
        cfg: &TurnParams,
        engine: &mut dyn Engine,
        emit: &mut dyn FnMut(&[u8]),
        mut logits: Vec<f32>,
    ) -> Result<TurnOutcome, ConvError> {
        let stop_bytes: Vec<Vec<u8>> = cfg
            .stop_strings
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();
        // 全部生成字节（stop 匹配 + assistant 文本累积）；emitted 跟踪已 emit 字节。
        let mut full: Vec<u8> = Vec::new();
        let mut emitted = 0usize;
        let mut n_gen = 0usize;
        let mut stopped_by_eog = false;
        let mut stopped_by_string = false;
        let mut hit_n_predict = false;

        loop {
            if n_gen >= cfg.n_predict {
                hit_n_predict = true;
                break;
            }
            // KV 写位置必须 < n_ctx（越界会写坏区域）。
            if self.current_pos >= self.n_ctx {
                hit_n_predict = true;
                break;
            }
            let sampled = sampler::sample_with_penalties(
                &mut logits,
                cfg.temp,
                cfg.top_k,
                cfg.top_p,
                cfg.repeat_penalty,
                cfg.frequency_penalty,
                cfg.presence_penalty,
                &self.prev_tokens,
                &mut self.rng,
            );
            if self.is_eog(sampled.token_id) {
                // EOG 必须写入 KV：canonical 渲染在 assistant 消息后带 EOG 标记
                // （§5.4），llama.cpp 也是先解码 EOG 再停止。不写会破坏不变量。
                stopped_by_eog = true;
                self.prev_tokens.push(sampled.token_id);
                if self.prev_tokens.len() > REPEAT_LAST_N {
                    self.prev_tokens.drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
                }
                self.stream_tokens.push(sampled.token_id);
                let _ = engine.forward(&[sampled.token_id], &[self.current_pos], 1);
                self.current_pos += 1;
                break;
            }
            n_gen += 1;
            full.extend_from_slice(&decoder.decode_bytes(&[sampled.token_id]));
            // stop 串匹配跑**全量**字节流（llama.cpp 同款；跨 token 的 stop 也能命中）。
            // stop 串不属于 canonical 文本 → 其 token 不写入 KV/窗口（与 llama.cpp
            // 保留在 KV 的做法不同，但保持 §5.4 不变量严格成立）。
            if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                stopped_by_string = true;
                full.truncate(cut); // 记录文本不含 stop 串
                break;
            }
            self.prev_tokens.push(sampled.token_id);
            if self.prev_tokens.len() > REPEAT_LAST_N {
                self.prev_tokens.drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
            }
            self.stream_tokens.push(sampled.token_id);
            // 只 emit 完整 UTF-8 前缀（跨 token 半字符 holdback，避免 U+FFFD）。
            let complete = emitted + crate::tokenizer::complete_utf8_prefix_len(&full[emitted..]);
            if complete > emitted {
                emit(&full[emitted..complete]);
                emitted = complete;
            }
            logits = engine.forward(&[sampled.token_id], &[self.current_pos], 1);
            self.current_pos += 1;
        }
        if emitted < full.len() {
            emit(&full[emitted..]);
        }

        let text = String::from_utf8(full.clone())
            .unwrap_or_else(|_| String::from_utf8_lossy(&full).into_owned());
        self.messages.push(("assistant".to_string(), Some(text.clone())));
        // 未达 EOG → 下回合输入前补 EOT（§5.4）。
        self.need_insert_eot = !stopped_by_eog;

        Ok(TurnOutcome {
            text,
            stopped_by_eog,
            stopped_by_string,
            hit_n_predict,
            prefill_tokens: 0, // 调用方填写
            tokens_generated: n_gen,
            dropped_turns: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    const EOS: u32 = 2;
    const IM_END: u32 = 7;

    /// 可编程 mock 引擎：每次 forward 弹出 program 中的下一个 token id，
    /// 返回在该 id 上打尖峰的 logits（temp=0 → greedy 必选它）。
    /// program 必须覆盖每次 forward 调用（含 EOT 插入与 delta prefill）。
    struct MockEngine {
        program: VecDeque<u32>,
        calls: Vec<(Vec<u32>, Vec<usize>, usize)>,
        resets: usize,
        vocab: usize,
    }

    impl MockEngine {
        fn new(program: Vec<u32>) -> Self {
            Self { program: program.into(), calls: Vec::new(), resets: 0, vocab: 4096 }
        }
        fn call_tokens(&self) -> Vec<u32> {
            self.calls.iter().flat_map(|c| c.0.clone()).collect()
        }
    }

    impl Engine for MockEngine {
        fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32> {
            self.calls.push((tokens.to_vec(), positions.to_vec(), n_out));
            let id = self.program.pop_front().unwrap_or(EOS);
            let mut logits = vec![0.0f32; self.vocab];
            logits[id as usize] = 100.0;
            logits
        }
        fn reset_cache(&mut self) {
            self.resets += 1;
        }
    }

    /// 假 codec：与真实 tokenizer 同语义——模板特殊标记是**单个** token id
    /// （`<|im_end|>` = IM_END，`<|im_start|>` = 7000），其余按字节编码。
    /// 这样 `tokenize(render(...))` 的 canonical 形式与 KV 中的 EOG/EOT 单 token 对齐，
    /// §5.4 不变量才能在 token 级被断言（字节式 codec 会把 `<|im_end|>` 拆成 10 个 token）。
    struct FakeCodec;

    impl FakeCodec {
        const IM_START: u32 = 7000;
    }

    impl TokenCodec for FakeCodec {
        fn encode(&self, text: &str) -> Vec<u32> {
            let mut out = Vec::new();
            let mut rest = text;
            while !rest.is_empty() {
                if let Some(r) = rest.strip_prefix("<|im_end|>") {
                    out.push(IM_END);
                    rest = r;
                } else if let Some(r) = rest.strip_prefix("<|im_start|>") {
                    out.push(Self::IM_START);
                    rest = r;
                } else {
                    let b = rest.as_bytes()[0];
                    out.push(b as u32);
                    rest = &rest[1..];
                }
            }
            out
        }
        fn decode_bytes(&self, ids: &[u32]) -> Vec<u8> {
            let mut out = Vec::new();
            for &id in ids {
                match id {
                    IM_END => out.extend_from_slice(b"<|im_end|>"),
                    Self::IM_START => out.extend_from_slice(b"<|im_start|>"),
                    b => out.push(b as u8),
                }
            }
            out
        }
    }

    fn cfg() -> TurnParams {
        TurnParams {
            n_predict: 512,
            temp: 0.0, // greedy：mock 尖峰必被选中
            top_k: 4096,
            top_p: 1.0,
            repeat_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_strings: Vec::new(),
        }
    }

    fn spec(n_ctx: usize) -> ConversationSpec {
        ConversationSpec {
            template: None, // ChatML fallback
            bos_text: String::new(),
            eog: vec![EOS, IM_END],
            eot: IM_END,
            seed: 42,
            n_ctx,
            system_prompt: None,
        }
    }

    fn conv(n_ctx: usize) -> Conversation {
        Conversation::new(spec(n_ctx))
    }

    fn noop_emit() -> impl FnMut(&[u8]) {
        |_| {}
    }

    /// canonical 渲染（ChatML fallback，无 generation prompt）的字节级 token 化。
    fn canonical(messages: &[(String, Option<String>)]) -> Vec<u32> {
        FakeCodec.encode(&template::fallback_chatml_messages(messages, false))
    }

    fn fallback_full(messages: &[(String, Option<String>)]) -> Vec<u32> {
        FakeCodec.encode(&template::fallback_chatml_messages(messages, true))
    }

    #[test]
    fn start_no_input_waits() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![]);
        let out = c.start(None, &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert!(out.is_none());
        assert!(c.messages.is_empty());
        assert_eq!(c.current_pos, 0);
    }

    #[test]
    fn first_turn_full_render_and_eog() {
        let mut c = conv(512);
        // 'H','i', EOG, EOG-decode 占位
        let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS]);
        let mut emitted: Vec<u8> = Vec::new();
        let out = c
            .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut |b| emitted.extend_from_slice(b))
            .unwrap()
            .unwrap();
        assert!(out.stopped_by_eog);
        assert_eq!(out.text, "Hi");
        // 首回合全量渲染（含 generation prompt）+ 生成 + EOG（§5.4：EOG 入 KV）
        let full = fallback_full(&[("user".into(), Some("hi".into()))]);
        assert_eq!(out.prefill_tokens, full.len());
        assert_eq!(c.stream_tokens, [&full[..], &[72, 105, IM_END]].concat());
        assert_eq!(c.messages, vec![
            ("user".into(), Some("hi".into())),
            ("assistant".into(), Some("Hi".into())),
        ]);
        assert!(!c.need_insert_eot);
        assert_eq!(emitted, b"Hi");
    }

    #[test]
    fn second_turn_appends_only_delta() {
        let mut c = conv(512);
        // t1: EOG, EOG-decode 占位；t2: EOG, EOG-decode 占位
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        let len_after_t1 = c.stream_tokens.len();
        let t2 = c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert!(t2.stopped_by_eog);
        // delta = "\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"（含换行补偿）
        let delta = FakeCodec.encode("\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(t2.prefill_tokens, delta.len());
        // 增量性：只追加 delta + 本回合 EOG（立即命中，无生成 token）
        assert_eq!(c.stream_tokens.len(), len_after_t1 + delta.len() + 1);
        assert_eq!(&c.stream_tokens[len_after_t1..len_after_t1 + delta.len()], &delta[..]);
        assert_eq!(c.stream_tokens.last(), Some(&IM_END));
        // turn_pos 指向 t2 delta 起点（= t1 结束后位置）
        assert_eq!(c.turn_pos, len_after_t1);
        // assistant 消息已记录（本轮立即 EOG → 空文本）
        assert_eq!(c.messages.last().unwrap().1.as_deref(), Some(""));
    }

    #[test]
    fn stop_string_truncates_and_sets_eot() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
        let mut tp = cfg();
        tp.stop_strings = vec!["!".to_string()];
        let out = c.user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit()).unwrap();
        assert!(out.stopped_by_string);
        assert!(!out.stopped_by_eog);
        assert_eq!(out.text, "Hi");
        assert!(c.need_insert_eot, "stop 串终止 → 下回合需补 EOT");

        // 下回合先插 EOT（engine 收到 [eot] 调用），再走 delta。
        let mut eng2 = MockEngine::new(vec![999, IM_END]); // 999 = EOT 插入的 dummy
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit()).unwrap();
        assert_eq!(eng2.calls[0].0, vec![IM_END], "first call must insert EOT");
        assert_eq!(eng2.calls[0].1, vec![c.turn_pos - 1], "EOT 写在 delta 之前的位置");
        // EOT 后、delta 前：stream 前缀 == canonical 前缀（缺末尾模板换行）
        let canon = canonical(&[
            ("user".into(), Some("hi".into())),
            ("assistant".into(), Some("Hi".into())),
        ]);
        assert_eq!(c.stream_tokens[..canon.len() - 1], canon[..canon.len() - 1]);
    }

    #[test]
    fn n_predict_exhaustion_sets_eot() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![72, 105, 33, 34]);
        let mut tp = cfg();
        tp.n_predict = 2;
        let out = c.user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit()).unwrap();
        assert!(out.hit_n_predict);
        assert_eq!(out.text, "Hi");
        assert_eq!(out.tokens_generated, 2);
        assert!(c.need_insert_eot);
    }

    #[test]
    fn regen_rolls_back_and_regenerates() {
        let mut c = conv(512);
        // t1: EOG,占位；t2: 'P', EOG,占位；regen: 'W', EOG,占位
        let mut eng = MockEngine::new(vec![IM_END, EOS, 80, IM_END, EOS, 87, IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert_eq!(c.messages.last().unwrap().0, "assistant");
        let turn2_start = c.turn_pos;

        // regen t2：回滚到 turn2_start，重放 t2 delta，生成 'W'
        let mut eng2 = MockEngine::new(vec![87, IM_END, EOS]); // 'W', EOG, 占位
        let out = c.regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit()).unwrap();
        assert_eq!(out.text, "W");
        assert!(out.stopped_by_eog);
        // messages: user hi, assistant "", user Q, assistant W
        assert_eq!(c.messages.len(), 4);
        assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("W"));
        // t2 旧内容 'P' 已从流中移除
        assert!(!c.stream_tokens.contains(&80));
        assert_eq!(c.turn_pos, turn2_start);
    }

    #[test]
    fn regen_first_turn_uses_full_render() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert_eq!(c.turn_pos, 0);

        // regen t1：turn_pos == 0 → 全量渲染（engine 第一次调用必须是全量，不是后缀）
        let mut eng2 = MockEngine::new(vec![88, IM_END, EOS]); // 'X', EOG, 占位
        let out = c.regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit()).unwrap();
        assert_eq!(out.text, "X");
        let full = fallback_full(&[("user".into(), Some("hi".into()))]);
        assert_eq!(eng2.calls[0].0, full, "first call must be the full first-turn render");
    }

    #[test]
    fn regen_without_assistant_errors() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![]);
        let err = c.regen_turn(&FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap_err();
        assert!(matches!(err, ConvError::NothingToRegen));
    }

    #[test]
    fn clear_resets_state_keeps_system() {
        let mut s = spec(512);
        s.system_prompt = Some("Be nice.".into());
        let mut c = Conversation::new(s);
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        c.clear();
        assert_eq!(c.messages, vec![("system".into(), Some("Be nice.".into()))]);
        assert!(c.stream_tokens.is_empty());
        assert_eq!(c.current_pos, 0);
        assert_eq!(c.turn_pos, 0);
        assert!(!c.need_insert_eot);
    }

    #[test]
    fn context_full_errors_before_prefill() {
        let mut c = conv(32); // 很小的 n_ctx
        let mut eng = MockEngine::new(vec![IM_END]);
        let long_input = "x".repeat(64);
        let err = c
            .user_turn(&long_input, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap_err();
        assert!(matches!(err, ConvError::ContextFull { .. }));
        // 状态未污染：没有任何 forward 调用
        assert!(eng.calls.is_empty());
        assert!(c.messages.is_empty());
    }

    #[test]
    fn context_fill_during_decode_stops_cleanly() {
        // fallback([user hi], true) 的 token 数为 21（特殊标记各 1 token）
        let mut c = conv(30);
        // 首回合 delta 占 21；解码一路填到 n_ctx=30
        let mut eng = MockEngine::new(vec![72, 105, 33, 34, 35, 36, 37, 38, 39, 40, IM_END]);
        let mut tp = cfg();
        tp.n_predict = 100;
        let out = c.user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit()).unwrap();
        assert!(out.hit_n_predict, "上下文填满应像 n_predict 耗尽一样干净停止");
        assert!(c.current_pos <= 30);
        assert!(c.need_insert_eot);
    }

    #[test]
    fn prev_tokens_reseeded_per_turn() {
        let mut c = conv(512);
        // t1: 'H','i', EOG, 占位；t2: EOG, 占位
        let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS, IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        // t1 结束后 prev_tokens = 最后 64 个 stream token（含 delta + 生成 + EOG）
        assert_eq!(c.prev_tokens.len(), c.stream_tokens.len().min(REPEAT_LAST_N));
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert_eq!(c.prev_tokens.len(), c.stream_tokens.len().min(REPEAT_LAST_N));
    }

    #[test]
    fn eot_inserted_before_first_user_turn_after_no_eog() {
        // start 之后首回合直接以 stop 串结束 → 第二个 user_turn 先插 EOT
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
        let mut tp = cfg();
        tp.stop_strings = vec!["!".to_string()];
        c.start(Some("hi"), &FakeCodec, &tp, &mut eng, &mut noop_emit()).unwrap();
        assert!(c.need_insert_eot);

        let mut eng2 = MockEngine::new(vec![999, IM_END]);
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit()).unwrap();
        assert_eq!(eng2.calls[0].0, vec![IM_END]);
    }

    // === Phase 3：溢出截断 + 会话持久化 ===

    #[test]
    fn overflow_truncates_oldest_turns_and_rehydrates() {
        // n_ctx=50：turn1(22) + turn2(44) 都放得下；turn3 的 delta 会让
        // current_pos + delta > 50 → 丢弃最旧的 user+assistant 对并全量重灌。
        let mut c = conv(50);
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert_eq!(c.stream_tokens.len(), 22, "t1: 21-token render + EOG");
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        assert_eq!(c.stream_tokens.len(), 44, "t2: +21 delta + EOG");

        // t3 触发溢出：重灌（engine.reset）+ 丢 1 回合 + 全量渲染
        let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
        let out = c.user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit()).unwrap();
        assert_eq!(out.dropped_turns, 1, "oldest turn must be dropped");
        assert_eq!(eng3.resets, 1, "rehydrate must reset the engine cache");
        // 消息：最旧 [user hi, assistant] 对已被丢弃
        assert_eq!(
            c.messages,
            vec![
                ("user".into(), Some("Q".into())),
                ("assistant".into(), Some("".into())),
                ("user".into(), Some("X".into())),
                ("assistant".into(), Some("".into())),
            ]
        );
        // KV = 重灌后的全量渲染（turn_pos 归零）+ EOG
        let canon = FakeCodec.encode(&template::fallback_chatml_messages(
            &[("user".into(), Some("Q".into())), ("assistant".into(), Some("".into()))],
            false,
        ));
        assert_eq!(c.turn_pos, 0);
        let full = FakeCodec.encode(&template::fallback_chatml_messages(
            &[
                ("user".into(), Some("Q".into())),
                ("assistant".into(), Some("".into())),
                ("user".into(), Some("X".into())),
            ],
            true,
        ));
        assert_eq!(c.stream_tokens, [&full[..], &[IM_END]].concat());
        assert_eq!(c.stream_tokens.len(), canon.len() + full.len() - canon.len() + 1);
    }

    #[test]
    fn overflow_single_message_still_errors() {
        let mut c = conv(20); // 小于单条消息的渲染长度
        let mut eng = MockEngine::new(vec![]);
        let err = c
            .user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap_err();
        assert!(matches!(err, ConvError::ContextFull { .. }));
    }

    #[test]
    fn session_json_round_trip() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        let json = c.messages_to_json();
        let parsed = Conversation::messages_from_json(&json).expect("parse");
        assert_eq!(parsed, c.messages);
        // null content 保留（OpenAI 风格对象格式）
        let with_null = vec![("user".into(), Some("hi".into())), ("assistant".into(), None)];
        let with_null_json = serde_json::json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": null },
        ])
        .to_string();
        let j2 = Conversation::messages_from_json(&with_null_json).unwrap();
        assert_eq!(j2, with_null);
        // 非法输入 → None
        assert!(Conversation::messages_from_json("not json").is_none());
    }

    #[test]
    fn load_history_rehydrates_full_render() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit()).unwrap();
        let saved = c.messages_to_json();

        // 新会话载入历史 → 全量重灌（render(messages, false)）
        let mut c2 = conv(512);
        let mut eng2 = MockEngine::new(vec![IM_END, EOS]);
        let msgs = Conversation::messages_from_json(&saved).unwrap();
        c2.load_history(msgs, &FakeCodec, &mut eng2);
        assert_eq!(c2.messages, c.messages);
        assert_eq!(c2.current_pos, c2.stream_tokens.len());
        let canon = canonical(&c.messages);
        assert_eq!(c2.stream_tokens, canon, "KV = canonical render of loaded history");
        assert_eq!(c2.turn_pos, 0);

        // 载入后继续对话（增量 delta 从重灌后的 KV 续写）
        let out = c2.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit()).unwrap();
        assert!(out.stopped_by_eog);
        assert_eq!(c2.messages.len(), 4);
        assert!(c2.current_pos > c2.stream_tokens.len().saturating_sub(1));
        assert_eq!(c2.current_pos, c2.stream_tokens.len());
    }

    /// 真实模型 2 回合冒烟（L2 的一部分；默认 ignore，与现有 realdata 测试一致）：
    ///   cargo test --bin minfer conversation_real_model_smoke -- --ignored
    /// 依赖本地缓存的 Qwen2.5-0.5B q4_0（无则跳过）。
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn conversation_real_model_smoke() {
        fn cached_qwen05_q4_0() -> Option<std::path::PathBuf> {
            let home = std::env::var_os("HOME")?;
            let mut p = std::path::PathBuf::from(home);
            p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
            p.exists().then_some(p)
        }
        let Some(path) = cached_qwen05_q4_0() else {
            eprintln!("0.5B q4_0 not cached; skipping conversation smoke");
            return;
        };
        let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
        let model = crate::models::load_model(&gguf).expect("load model");
        let tok = crate::tokenizer::Tokenizer::load(&gguf.parts[0].ctx);
        let ctx = &gguf.parts[0].ctx;
        let template = ctx
            .kv
            .iter()
            .find(|kv| kv.key == "tokenizer.chat_template")
            .map(|kv| kv.get_val_str(0).to_string());
        let special = model.special_tokens();
        let bos_text = tok
            .id_to_token
            .get(tok.bos_token as usize)
            .cloned()
            .unwrap_or_default();

        let spec = ConversationSpec {
            template,
            bos_text,
            eog: {
                let mut v = vec![special.eos];
                if let Some(im) = special.im_end {
                    v.push(im);
                }
                v
            },
            eot: special.im_end.unwrap_or(special.eos),
            seed: 42,
            n_ctx: 512,
            system_prompt: None,
        };
        let mut conv = Conversation::new(spec);
        let mut engine = GraphEngine::new(&*model, 512);
        let tp = TurnParams {
            n_predict: 16, // 短回答即可（0.5B 通常几十 token 内 EOG）
            temp: 0.0,     // greedy：确定、快
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_strings: Vec::new(),
        };
        let mut emitted: Vec<u8> = Vec::new();
        let t1 = conv
            .start(Some("hi"), &tok, &tp, &mut engine, &mut |b| emitted.extend_from_slice(b))
            .unwrap()
            .expect("first turn ran");
        let t2 = conv
            .user_turn("what is 2+2?", &tok, &tp, &mut engine, &mut |b| emitted.extend_from_slice(b))
            .unwrap();
        eprintln!("t1 text: {:?}", t1.text);
        eprintln!("t2 text: {:?}", t2.text);
        assert!(!t1.text.is_empty(), "turn 1 must answer");
        assert!(!t2.text.is_empty(), "turn 2 must answer");
        assert!(!t1.text.contains('\u{FFFD}') && !t2.text.contains('\u{FFFD}'), "no U+FFFD");
        // 增量性：t2 的 delta prefill 必须远小于 t1 的全量 prefill
        assert!(
            t2.prefill_tokens < t1.prefill_tokens,
            "t2 delta ({}) must be < t1 full render ({})",
            t2.prefill_tokens,
            t1.prefill_tokens
        );
        // 强不变量：current_pos == stream_tokens.len()（KV 镜像一致）
        assert_eq!(conv.current_pos, conv.stream_tokens.len());
        assert_eq!(conv.messages.len(), 4, "user, assistant, user, assistant");
        // EOG 后无需补 EOT（模型自停）
        assert!(!conv.need_insert_eot, "greedy 0.5B usually EOGs; if this trips, inspect output");
    }
}
