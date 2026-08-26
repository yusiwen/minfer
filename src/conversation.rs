//! Core of the CLI multi-turn conversation session (docs/CLI-CONVERSATION-PLAN.md).
//!
//! Strategy: **append-only KV + incremental template rendering** (the llama.cpp legacy `examples/main` route).
//! The session's token stream accumulates in the KV region (position-addressed, persistent across graph rebuilds); each turn only
//! prefills/decodes the delta of the new message — long sessions cost O(turn delta) instead of O(full history).
//!
//! Key abstractions:
//! - [`Engine`]: the inference backend (`forward` + `reset_cache`). The real implementation [`GraphEngine`]
//!   holds a `ModelDef` reference + a session-private `GraphCache` + `n_ctx`; the mock implementation lets the state machine
//!   be fully tested without a model (§8.2 L1).
//! - [`TokenCodec`]: byte-level encode/decode (trait-ified `Tokenizer`), mock-friendly.
//! - [`Conversation`]: holds only logical state (messages, the token-stream mirror, the sampler window, the rollback point),
//!   **no cache** — the cache belongs to `GraphEngine`, avoiding the `&mut self` vs `&mut cache`
//!   borrow conflict, and keeps `Conversation` fully testable.
//!
//! KV consistency invariant (§5.4): after each turn's EOT insertion and before the delta append,
//! `stream_tokens` (the host-side mirror of the KV) == a token prefix of `tokenize(render(messages, false))`
//! (possibly missing the trailing template newline). `/regen` rollback (`turn_pos` pointer rewind) and
//! full re-render both rely on it.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::graph::cache::GraphCache;
use crate::models::ModelDef;
use crate::sampler;
use crate::template::{self, format_single};
use crate::tokenizer::Tokenizer;

/// Inference backend abstraction: the prerequisite for L1 mock testing (§8.2).
pub trait Engine {
    /// Returns n_out*nv logits (when n_out=1, the logits of the last/only token).
    /// The real implementation wraps `ModelDef::forward_graph_cached`.
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32>;
    /// Drops all KV state (called on full re-render / `/clear`).
    fn reset_cache(&mut self);
}

/// Byte-level encode/decode abstraction: trait-ified `Tokenizer` (mock-friendly).
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

/// The real inference engine: a model reference + a session-private `GraphCache` + `n_ctx`.
/// The cache lives here (not in `Conversation`), so the Engine can be swapped for a mock (§8.2).
pub struct GraphEngine<'a> {
    model: &'a dyn ModelDef,
    cache: GraphCache,
    n_ctx: usize,
}

impl<'a> GraphEngine<'a> {
    pub fn new(model: &'a dyn ModelDef, n_ctx: usize) -> Self {
        Self {
            model,
            cache: GraphCache::new(),
            n_ctx,
        }
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

/// Session construction parameters.
pub struct ConversationSpec {
    /// `tokenizer.chat_template`; None → ChatML fallback rendering.
    pub template: Option<String>,
    pub bos_text: String,
    /// The EOG set (eos + im_end).
    pub eog: Vec<u32>,
    /// Token inserted when a turn ends without EOG (im_end, defaulting to eos).
    pub eot: u32,
    pub seed: u64,
    pub n_ctx: usize,
    /// The `--system` prompt (used as the first system message).
    pub system_prompt: Option<String>,
}

/// Per-turn sampling/stopping parameters (mapped from the CLI's GenParams).
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

/// The result of one generation turn.
// text / stopped_by_* are emitted by the streaming emit path; the fields are kept as structured results (for tests).
#[derive(Debug)]
pub struct TurnOutcome {
    /// The assistant-generated text (after stop-string truncation; without the EOG).
    #[allow(dead_code)]
    pub text: String,
    #[allow(dead_code)]
    pub stopped_by_eog: bool,
    #[allow(dead_code)]
    pub stopped_by_string: bool,
    /// n_predict exhausted or the context is full.
    pub hit_n_predict: bool,
    /// Number of tokens prefilled this turn (for incrementality asserts; full length on the fallback path).
    pub prefill_tokens: usize,
    pub tokens_generated: usize,
    /// Old turns dropped on context overflow (0 = not truncated, §5.7).
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

/// A multi-turn conversation session (logical state; inference state lives in `Engine`).
pub struct Conversation {
    /// Recorded messages (role, content). Isomorphic with the server's ChatMessage.
    pub messages: Vec<(String, Option<String>)>,
    /// Host-side mirror of the full token stream in the KV (prefill delta + generated + manual EOT).
    pub stream_tokens: Vec<u32>,
    /// The next write position (strong invariant: == stream_tokens.len()).
    pub current_pos: usize,
    /// Start position of the current turn's delta (`/regen` rollback point; 0 = needs a full re-render).
    pub turn_pos: usize,
    pub rng: StdRng,
    /// Penalty window; the last 64 stream_tokens taken at the start of each turn.
    pub prev_tokens: Vec<u32>,
    /// The EOG set.
    pub eog: Vec<u32>,
    /// Token inserted when a turn does not reach EOG.
    pub eot: u32,
    /// The previous turn did not end with EOG → write EOT before the next turn's input (§5.4 invariant).
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

    /// Fully renders the current messages (with/without the generation prompt).
    fn render_full(&self, add_generation_prompt: bool) -> String {
        match &self.template {
            Some(t) => {
                template::render_messages(t, &self.messages, add_generation_prompt, &self.bos_text)
            }
            None => template::fallback_chatml_messages(&self.messages, add_generation_prompt),
        }
    }

    /// Resets the cache and prefills the given token stream from scratch (the full re-render path).
    /// Returns the logits of the last prefill token (the input for the first sample).
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

    /// Starts the session: `Some(first input)` → full render + prefill + generate; None → wait for input.
    /// (Entry kept as the full API; the CLI goes through the unified `user_turn` path, `start` is covered by tests.)
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
        self.messages
            .push(("user".to_string(), Some(input.to_string())));
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

    /// Appends a user message and generates the assistant reply.
    pub fn user_turn(
        &mut self,
        input: &str,
        decoder: &dyn TokenCodec,
        cfg: &TurnParams,
        engine: &mut dyn Engine,
        emit: &mut dyn FnMut(&[u8]),
    ) -> Result<TurnOutcome, ConvError> {
        // 1. Previous turn did not reach EOG → insert EOT first, keeping the KV consistent with the template's canonical output (§5.4).
        if self.need_insert_eot {
            let _ = engine.forward(&[self.eot], &[self.current_pos], 1);
            self.stream_tokens.push(self.eot);
            self.current_pos += 1;
        }

        // 2. Incremental rendering (diff-based, §5.3).
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

        // 3. Prefix mismatch (non-deterministic template) → full re-render fallback (§5.4).
        if !delta.prefix_matched {
            self.messages
                .push(("user".to_string(), Some(input.to_string())));
            let full = self.render_full(true);
            let toks = decoder.encode(&full);
            if toks.is_empty() {
                return Err(ConvError::EmptyInput);
            }
            let logits = self.rehydrate_full(engine, &toks);
            self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
            let mut out =
                self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
            out.prefill_tokens = toks.len();
            return Ok(out);
        }

        // 4. Context overflow: drop the oldest non-system turn + full re-render (§5.7).
        if self.current_pos + delta_toks.len() > self.n_ctx {
            self.messages
                .push(("user".to_string(), Some(input.to_string())));
            let dropped = self.drop_oldest_turns_until_fits(decoder);
            if dropped == 0 {
                // Only [system?, last user] left and it still does not fit: a single message is too long, error out.
                // Roll back: pop the just-pushed user message and reset the EOT flag (if EOT was written this turn,
                // it already closed the previous turn correctly and must not be inserted again next turn).
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
            let mut out =
                self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
            out.prefill_tokens = toks.len();
            out.dropped_turns = dropped;
            return Ok(out);
        }

        // 5. Record the user message, set the rollback point, prefill the delta.
        self.messages
            .push(("user".to_string(), Some(input.to_string())));
        self.turn_pos = self.current_pos;
        let positions: Vec<usize> =
            (self.current_pos..self.current_pos + delta_toks.len()).collect();
        let logits = engine.forward(&delta_toks, &positions, 1);
        self.stream_tokens.extend_from_slice(&delta_toks);
        self.current_pos += delta_toks.len();
        // The penalty window is reseeded **after** the delta is appended (llama.cpp feeds prompt tokens into the sampler window).
        self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);

        let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
        out.prefill_tokens = delta_toks.len();
        Ok(out)
    }

    /// Regenerates the last assistant reply: roll back to `turn_pos` (the position-addressed region just rewinds
    /// the pointer; the tail of the region is stale but never read) → replay the last user message.
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

        // First turn (or a turn after a full re-render) has turn_pos == 0: after the rollback the KV is empty, so a full render is needed;
        // other turns only need to replay that user message's delta.
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
        // The penalty window is reseeded **after** the delta is appended (same as user_turn).
        self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);

        let mut out = self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
        out.prefill_tokens = toks.len();
        Ok(out)
    }

    /// `/clear`: clears history and sampling state (keeps system and rng).
    /// Note: the caller must also call `engine.reset_cache()`.
    pub fn clear(&mut self) {
        let system = self.messages.iter().find(|m| m.0 == "system").cloned();
        self.messages = system.into_iter().collect();
        self.stream_tokens.clear();
        self.current_pos = 0;
        self.turn_pos = 0;
        self.prev_tokens.clear();
        self.need_insert_eot = false;
    }

    /// Context overflow: drop the oldest non-system turns (a user + its assistant pair) until
    /// the token count of `render(messages, true)` is ≤ n_ctx. Returns the number of dropped turns;
    /// returns 0 when only [system?, last user] is left and it still does not fit (the caller errors out).
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

    /// Start of the oldest droppable turn: the position of the first user message before the last user message.
    fn oldest_droppable_turn_start(&self) -> Option<usize> {
        let last_user = self.messages.iter().rposition(|m| m.0 == "user")?;
        (0..last_user).find(|&i| self.messages[i].0 == "user")
    }

    /// `--session`: serializes messages as an OpenAI-style JSON array
    /// (isomorphic with the server's ChatMessage: `[{"role","content"},...]`, §5.8).
    pub fn messages_to_json(&self) -> String {
        let arr: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        serde_json::to_string_pretty(&arr).unwrap_or_default()
    }

    /// Parses messages from a JSON array (`--session` loading; invalid input returns None).
    pub fn messages_from_json(json: &str) -> Option<Vec<(String, Option<String>)>> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        let arr = v.as_array()?;
        arr.iter()
            .map(|m| {
                let role = m.get("role")?.as_str()?.to_string();
                let content = m
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(str::to_string);
                Some((role, content))
            })
            .collect()
    }

    /// Loads history and **fully re-renders** the KV (§5.8: KV state is not serialized — thanks to the §5.4 invariant,
    /// the re-render matches continuing the session, just with one extra prefill).
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

    /// Decode loop: sample/decode token by token starting from `logits` (the last prefill token),
    /// until EOG / a stop string / n_predict / the context is full.
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
        // All generated bytes (stop matching + assistant text accumulation); emitted tracks the bytes already emitted.
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
            // The KV write position must be < n_ctx (out-of-bounds writes would corrupt the region).
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
                // The EOG must be written to the KV: the canonical render carries the EOG marker after the assistant message
                // (§5.4), and llama.cpp also decodes the EOG before stopping. Not writing it would break the invariant.
                stopped_by_eog = true;
                self.prev_tokens.push(sampled.token_id);
                if self.prev_tokens.len() > REPEAT_LAST_N {
                    self.prev_tokens
                        .drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
                }
                self.stream_tokens.push(sampled.token_id);
                let _ = engine.forward(&[sampled.token_id], &[self.current_pos], 1);
                self.current_pos += 1;
                break;
            }
            n_gen += 1;
            full.extend_from_slice(&decoder.decode_bytes(&[sampled.token_id]));
            // Stop-string matching runs over the **full** byte stream (same as llama.cpp; a stop spanning tokens still hits).
            // Stop strings are not part of the canonical text → their tokens are not written to the KV/window (unlike llama.cpp,
            // which keeps them in the KV, but this keeps the §5.4 invariant strictly true).
            if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                stopped_by_string = true;
                full.truncate(cut); // the recorded text excludes the stop string
                break;
            }
            self.prev_tokens.push(sampled.token_id);
            if self.prev_tokens.len() > REPEAT_LAST_N {
                self.prev_tokens
                    .drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
            }
            self.stream_tokens.push(sampled.token_id);
            // Emit only complete UTF-8 prefixes (holdback of half-characters across tokens, avoiding U+FFFD).
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
        self.messages
            .push(("assistant".to_string(), Some(text.clone())));
        // Did not reach EOG → insert EOT before the next turn's input (§5.4).
        self.need_insert_eot = !stopped_by_eog;

        Ok(TurnOutcome {
            text,
            stopped_by_eog,
            stopped_by_string,
            hit_n_predict,
            prefill_tokens: 0, // filled in by the caller
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

    /// Programmable mock engine: each forward pops the next token id from the program,
    /// returning logits spiked at that id (temp=0 → greedy is forced to pick it).
    /// The program must cover every forward call (including EOT insertion and delta prefill).
    struct MockEngine {
        program: VecDeque<u32>,
        calls: Vec<(Vec<u32>, Vec<usize>, usize)>,
        resets: usize,
        vocab: usize,
    }

    impl MockEngine {
        fn new(program: Vec<u32>) -> Self {
            Self {
                program: program.into(),
                calls: Vec::new(),
                resets: 0,
                vocab: 4096,
            }
        }
        fn call_tokens(&self) -> Vec<u32> {
            self.calls.iter().flat_map(|c| c.0.clone()).collect()
        }
    }

    impl Engine for MockEngine {
        fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32> {
            self.calls
                .push((tokens.to_vec(), positions.to_vec(), n_out));
            let id = self.program.pop_front().unwrap_or(EOS);
            let mut logits = vec![0.0f32; self.vocab];
            logits[id as usize] = 100.0;
            logits
        }
        fn reset_cache(&mut self) {
            self.resets += 1;
        }
    }

    /// Fake codec: same semantics as the real tokenizer — template special markers are **single** token ids
    /// (`<|im_end|>` = IM_END, `<|im_start|>` = 7000), everything else is encoded byte-wise.
    /// This aligns the canonical form of `tokenize(render(...))` with the single-token EOG/EOT in the KV,
    /// so the §5.4 invariant can be asserted at the token level (a byte-wise codec would split `<|im_end|>` into 10 tokens).
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
            temp: 0.0, // greedy: the mock spike is always picked
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

    /// Byte-level tokenization of the canonical render (ChatML fallback, no generation prompt).
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
        let out = c
            .start(None, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert!(out.is_none());
        assert!(c.messages.is_empty());
        assert_eq!(c.current_pos, 0);
    }

    #[test]
    fn first_turn_full_render_and_eog() {
        let mut c = conv(512);
        // 'H','i', EOG, EOG-decode placeholder
        let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS]);
        let mut emitted: Vec<u8> = Vec::new();
        let out = c
            .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut |b| {
                emitted.extend_from_slice(b)
            })
            .unwrap()
            .unwrap();
        assert!(out.stopped_by_eog);
        assert_eq!(out.text, "Hi");
        // First turn: full render (with generation prompt) + generation + EOG (§5.4: EOG enters the KV)
        let full = fallback_full(&[("user".into(), Some("hi".into()))]);
        assert_eq!(out.prefill_tokens, full.len());
        assert_eq!(c.stream_tokens, [&full[..], &[72, 105, IM_END]].concat());
        assert_eq!(
            c.messages,
            vec![
                ("user".into(), Some("hi".into())),
                ("assistant".into(), Some("Hi".into())),
            ]
        );
        assert!(!c.need_insert_eot);
        assert_eq!(emitted, b"Hi");
    }

    #[test]
    fn second_turn_appends_only_delta() {
        let mut c = conv(512);
        // t1: EOG, EOG-decode placeholder; t2: EOG, EOG-decode placeholder
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        let len_after_t1 = c.stream_tokens.len();
        let t2 = c
            .user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert!(t2.stopped_by_eog);
        // delta = "\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n" (with newline compensation)
        let delta = FakeCodec.encode("\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n");
        assert_eq!(t2.prefill_tokens, delta.len());
        // Incrementality: only the delta + this turn's EOG are appended (hit immediately, no generated tokens)
        assert_eq!(c.stream_tokens.len(), len_after_t1 + delta.len() + 1);
        assert_eq!(
            &c.stream_tokens[len_after_t1..len_after_t1 + delta.len()],
            &delta[..]
        );
        assert_eq!(c.stream_tokens.last(), Some(&IM_END));
        // turn_pos points to the start of t2's delta (= the position after t1)
        assert_eq!(c.turn_pos, len_after_t1);
        // The assistant message is recorded (immediate EOG this turn → empty text)
        assert_eq!(c.messages.last().unwrap().1.as_deref(), Some(""));
    }

    #[test]
    fn stop_string_truncates_and_sets_eot() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
        let mut tp = cfg();
        tp.stop_strings = vec!["!".to_string()];
        let out = c
            .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
            .unwrap();
        assert!(out.stopped_by_string);
        assert!(!out.stopped_by_eog);
        assert_eq!(out.text, "Hi");
        assert!(
            c.need_insert_eot,
            "stop-string termination → EOT needed before the next turn"
        );

        // The next turn inserts EOT first (the engine receives the [eot] call), then runs the delta.
        let mut eng2 = MockEngine::new(vec![999, IM_END]); // 999 = dummy for the EOT insertion
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
            .unwrap();
        assert_eq!(eng2.calls[0].0, vec![IM_END], "first call must insert EOT");
        assert_eq!(
            eng2.calls[0].1,
            vec![c.turn_pos - 1],
            "EOT written at the position before the delta"
        );
        // After EOT, before delta: the stream prefix == the canonical prefix (missing the trailing template newline)
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
        let out = c
            .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
            .unwrap();
        assert!(out.hit_n_predict);
        assert_eq!(out.text, "Hi");
        assert_eq!(out.tokens_generated, 2);
        assert!(c.need_insert_eot);
    }

    #[test]
    fn regen_rolls_back_and_regenerates() {
        let mut c = conv(512);
        // t1: EOG, placeholder; t2: 'P', EOG, placeholder; regen: 'W', EOG, placeholder
        let mut eng = MockEngine::new(vec![IM_END, EOS, 80, IM_END, EOS, 87, IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert_eq!(c.messages.last().unwrap().0, "assistant");
        let turn2_start = c.turn_pos;

        // regen t2: roll back to turn2_start, replay t2's delta, generate 'W'
        let mut eng2 = MockEngine::new(vec![87, IM_END, EOS]); // 'W', EOG, placeholder
        let out = c
            .regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
            .unwrap();
        assert_eq!(out.text, "W");
        assert!(out.stopped_by_eog);
        // messages: user hi, assistant "", user Q, assistant W
        assert_eq!(c.messages.len(), 4);
        assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("W"));
        // t2's old content 'P' has been removed from the stream
        assert!(!c.stream_tokens.contains(&80));
        assert_eq!(c.turn_pos, turn2_start);
    }

    #[test]
    fn regen_first_turn_uses_full_render() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert_eq!(c.turn_pos, 0);

        // regen t1: turn_pos == 0 → full render (the engine's first call must be the full render, not a suffix)
        let mut eng2 = MockEngine::new(vec![88, IM_END, EOS]); // 'X', EOG, placeholder
        let out = c
            .regen_turn(&FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
            .unwrap();
        assert_eq!(out.text, "X");
        let full = fallback_full(&[("user".into(), Some("hi".into()))]);
        assert_eq!(
            eng2.calls[0].0, full,
            "first call must be the full first-turn render"
        );
    }

    #[test]
    fn regen_without_assistant_errors() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![]);
        let err = c
            .regen_turn(&FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap_err();
        assert!(matches!(err, ConvError::NothingToRegen));
    }

    #[test]
    fn clear_resets_state_keeps_system() {
        let mut s = spec(512);
        s.system_prompt = Some("Be nice.".into());
        let mut c = Conversation::new(s);
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        c.clear();
        assert_eq!(c.messages, vec![("system".into(), Some("Be nice.".into()))]);
        assert!(c.stream_tokens.is_empty());
        assert_eq!(c.current_pos, 0);
        assert_eq!(c.turn_pos, 0);
        assert!(!c.need_insert_eot);
    }

    #[test]
    fn context_full_errors_before_prefill() {
        let mut c = conv(32); // a very small n_ctx
        let mut eng = MockEngine::new(vec![IM_END]);
        let long_input = "x".repeat(64);
        let err = c
            .user_turn(&long_input, &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap_err();
        assert!(matches!(err, ConvError::ContextFull { .. }));
        // State is not polluted: no forward calls at all
        assert!(eng.calls.is_empty());
        assert!(c.messages.is_empty());
    }

    #[test]
    fn context_fill_during_decode_stops_cleanly() {
        // fallback([user hi], true) tokenizes to 21 tokens (special markers are 1 token each)
        let mut c = conv(30);
        // The first turn's delta takes 21; decoding fills all the way up to n_ctx=30
        let mut eng = MockEngine::new(vec![72, 105, 33, 34, 35, 36, 37, 38, 39, 40, IM_END]);
        let mut tp = cfg();
        tp.n_predict = 100;
        let out = c
            .user_turn("hi", &FakeCodec, &tp, &mut eng, &mut noop_emit())
            .unwrap();
        assert!(
            out.hit_n_predict,
            "context full should stop as cleanly as n_predict exhaustion"
        );
        assert!(c.current_pos <= 30);
        assert!(c.need_insert_eot);
    }

    #[test]
    fn prev_tokens_reseeded_per_turn() {
        let mut c = conv(512);
        // t1: 'H','i', EOG, placeholder; t2: EOG, placeholder
        let mut eng = MockEngine::new(vec![72, 105, IM_END, EOS, IM_END, EOS]);
        c.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        // After t1, prev_tokens = the last 64 stream tokens (including delta + generated + EOG)
        assert_eq!(
            c.prev_tokens.len(),
            c.stream_tokens.len().min(REPEAT_LAST_N)
        );
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert_eq!(
            c.prev_tokens.len(),
            c.stream_tokens.len().min(REPEAT_LAST_N)
        );
    }

    #[test]
    fn eot_inserted_before_first_user_turn_after_no_eog() {
        // After start, the first turn ends via a stop string → the second user_turn inserts EOT first
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![72, 105, 33]); // 'H','i','!'
        let mut tp = cfg();
        tp.stop_strings = vec!["!".to_string()];
        c.start(Some("hi"), &FakeCodec, &tp, &mut eng, &mut noop_emit())
            .unwrap();
        assert!(c.need_insert_eot);

        let mut eng2 = MockEngine::new(vec![999, IM_END]);
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
            .unwrap();
        assert_eq!(eng2.calls[0].0, vec![IM_END]);
    }

    // === Phase 3: overflow truncation + session persistence ===

    #[test]
    fn overflow_truncates_oldest_turns_and_rehydrates() {
        // n_ctx=50: turn1(22) + turn2(44) both fit; turn3's delta makes
        // current_pos + delta > 50 → drop the oldest user+assistant pair and fully re-render.
        let mut c = conv(50);
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert_eq!(c.stream_tokens.len(), 22, "t1: 21-token render + EOG");
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert_eq!(c.stream_tokens.len(), 44, "t2: +21 delta + EOG");

        // t3 triggers overflow: re-render (engine.reset) + drop 1 turn + full render
        let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
        let out = c
            .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
            .unwrap();
        assert_eq!(out.dropped_turns, 1, "oldest turn must be dropped");
        assert_eq!(eng3.resets, 1, "rehydrate must reset the engine cache");
        // Messages: the oldest [user hi, assistant] pair has been dropped
        assert_eq!(
            c.messages,
            vec![
                ("user".into(), Some("Q".into())),
                ("assistant".into(), Some("".into())),
                ("user".into(), Some("X".into())),
                ("assistant".into(), Some("".into())),
            ]
        );
        // KV = the full render after re-render (turn_pos reset to zero) + EOG
        let canon = FakeCodec.encode(&template::fallback_chatml_messages(
            &[
                ("user".into(), Some("Q".into())),
                ("assistant".into(), Some("".into())),
            ],
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
        assert_eq!(
            c.stream_tokens.len(),
            canon.len() + full.len() - canon.len() + 1
        );
    }

    #[test]
    fn overflow_single_message_still_errors() {
        let mut c = conv(20); // smaller than a single message's render length
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
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        let json = c.messages_to_json();
        let parsed = Conversation::messages_from_json(&json).expect("parse");
        assert_eq!(parsed, c.messages);
        // null content is preserved (OpenAI-style object format)
        let with_null = vec![
            ("user".into(), Some("hi".into())),
            ("assistant".into(), None),
        ];
        let with_null_json = serde_json::json!([
            { "role": "user", "content": "hi" },
            { "role": "assistant", "content": null },
        ])
        .to_string();
        let j2 = Conversation::messages_from_json(&with_null_json).unwrap();
        assert_eq!(j2, with_null);
        // Invalid input → None
        assert!(Conversation::messages_from_json("not json").is_none());
    }

    #[test]
    fn load_history_rehydrates_full_render() {
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        let saved = c.messages_to_json();

        // A new session loads the history → full re-render (render(messages, false))
        let mut c2 = conv(512);
        let mut eng2 = MockEngine::new(vec![IM_END, EOS]);
        let msgs = Conversation::messages_from_json(&saved).unwrap();
        c2.load_history(msgs, &FakeCodec, &mut eng2);
        assert_eq!(c2.messages, c.messages);
        assert_eq!(c2.current_pos, c2.stream_tokens.len());
        let canon = canonical(&c.messages);
        assert_eq!(
            c2.stream_tokens, canon,
            "KV = canonical render of loaded history"
        );
        assert_eq!(c2.turn_pos, 0);

        // Continue the conversation after loading (the incremental delta continues from the re-rendered KV)
        let out = c2
            .user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
            .unwrap();
        assert!(out.stopped_by_eog);
        assert_eq!(c2.messages.len(), 4);
        assert!(c2.current_pos > c2.stream_tokens.len().saturating_sub(1));
        assert_eq!(c2.current_pos, c2.stream_tokens.len());
    }

    /// Real-model 2-turn smoke test (part of L2; ignored by default, consistent with the existing realdata tests):
    ///   cargo test --bin minfer conversation_real_model_smoke -- --ignored
    /// Requires the locally cached Qwen2.5-0.5B q4_0 (skips if absent).
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
            n_predict: 16, // a short reply suffices (0.5B usually EOGs within a few dozen tokens)
            temp: 0.0,     // greedy: deterministic and fast
            top_k: 40,
            top_p: 0.95,
            repeat_penalty: 1.1,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            stop_strings: Vec::new(),
        };
        let mut emitted: Vec<u8> = Vec::new();
        let t1 = conv
            .start(Some("hi"), &tok, &tp, &mut engine, &mut |b| {
                emitted.extend_from_slice(b)
            })
            .unwrap()
            .expect("first turn ran");
        let t2 = conv
            .user_turn("what is 2+2?", &tok, &tp, &mut engine, &mut |b| {
                emitted.extend_from_slice(b)
            })
            .unwrap();
        eprintln!("t1 text: {:?}", t1.text);
        eprintln!("t2 text: {:?}", t2.text);
        assert!(!t1.text.is_empty(), "turn 1 must answer");
        assert!(!t2.text.is_empty(), "turn 2 must answer");
        assert!(
            !t1.text.contains('\u{FFFD}') && !t2.text.contains('\u{FFFD}'),
            "no U+FFFD"
        );
        // Incrementality: t2's delta prefill must be much smaller than t1's full prefill
        assert!(
            t2.prefill_tokens < t1.prefill_tokens,
            "t2 delta ({}) must be < t1 full render ({})",
            t2.prefill_tokens,
            t1.prefill_tokens
        );
        // Strong invariant: current_pos == stream_tokens.len() (the KV mirror is consistent)
        assert_eq!(conv.current_pos, conv.stream_tokens.len());
        assert_eq!(conv.messages.len(), 4, "user, assistant, user, assistant");
        // No EOT needed after EOG (the model stops on its own)
        assert!(
            !conv.need_insert_eot,
            "greedy 0.5B usually EOGs; if this trips, inspect output"
        );
    }
}
