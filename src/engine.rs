//! The engine facade (#763 S6a): a loaded model, and a session that runs turns.
//!
//! This is the L2 tier ADR-0079 describes. `Engine` owns everything a model needs
//! to answer (the mapping, the model, the tokenizer, the chat template); a
//! [`Session`] borrows it and owns the per-conversation state (the KV cache and
//! the message history). A host creates one `Engine` and as many sessions as it
//! has conversations.
//!
//! **Status:** loading, `chat` and the raw `complete` continuation are here. The
//! CLI's single-shot path calls the free [`generate`] because it threads a
//! `SpecEngine`; `Engine::generate` is the embedder's entry — same loop, with the
//! engine owning the run's KV cache (#799). The CLI's conversation mode still
//! drives `conversation::Conversation` directly (#800).

use std::path::Path;
use std::sync::Mutex;

pub use crate::cancel::CancelToken;
pub use generate::{generate, GenerateOutcome, GenerateParams, SpecStatsOut};

use crate::conversation::{self, ConvError, Conversation, GraphEngine, TurnOutcome, TurnParams};
use crate::gguf::{GgufLoadError, GgufModel};
use crate::graph::cache::GraphCache;
use crate::graph::offload::OffloadRequest;
use crate::models::{ModelDef, ModelLoadError, SpecialTokens};
use crate::sampler::MirostatState;
use crate::template::{self, TemplateError};
use crate::tokenizer::Tokenizer;

/// Why an [`Engine`] could not be built.
#[derive(Debug)]
pub enum EngineError {
    /// The file did not parse as GGUF.
    Gguf(GgufLoadError),
    /// The architecture is unsupported, or a weight is missing.
    Model(ModelLoadError),
    /// The tokenizer metadata is unsupported or incomplete.
    Tokenizer(String),
    /// The chat template cannot be rendered.
    Template(TemplateError),
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EngineError::Gguf(e) => write!(f, "{e}"),
            EngineError::Model(e) => write!(f, "{e}"),
            EngineError::Tokenizer(e) => write!(f, "tokenizer: {e}"),
            EngineError::Template(e) => write!(f, "chat template: {}", e.message()),
        }
    }
}

impl std::error::Error for EngineError {}

impl From<GgufLoadError> for EngineError {
    fn from(e: GgufLoadError) -> Self {
        EngineError::Gguf(e)
    }
}

impl From<ModelLoadError> for EngineError {
    fn from(e: ModelLoadError) -> Self {
        EngineError::Model(e)
    }
}

/// A loaded model and everything needed to run a turn on it.
pub struct Engine {
    gguf: GgufModel,
    model: Box<dyn ModelDef>,
    tokenizer: Tokenizer,
    template: Option<String>,
    /// The ChatML fallback was chosen (no template in the file, none supplied).
    chatml_notice: bool,
    /// The single-shot path's KV state, owned **per engine** ([#799]).
    ///
    /// A `GraphCache`'s KV regions are persistent and sized once, so two models
    /// or two `n_ctx` values must never share one. Sessions do not come through
    /// here — each owns its own cache through `GraphEngine`.
    ///
    /// [#799]: https://github.com/yusiwen/minfer/issues/799
    generate_cache: Mutex<GenerateCache>,
}

/// The single-shot path's cache and the context size it was built for.
///
/// A generation carries no sequence state, so its KV is a pure optimization:
/// when the context size changes, dropping the cache and rebuilding costs one
/// graph build and keeps the facade total instead of panicking in the
/// allocator, which correctly refuses to resize live regions ([#799]).
///
/// [#799]: https://github.com/yusiwen/minfer/issues/799
struct GenerateCache {
    n_ctx: Option<usize>,
    cache: GraphCache,
}

impl GenerateCache {
    fn new() -> Self {
        Self {
            n_ctx: None,
            cache: GraphCache::new(),
        }
    }

    /// The cache sized for `n_ctx`, rebuilt when the size changed.
    fn for_ctx(&mut self, n_ctx: usize) -> &mut GraphCache {
        if self.n_ctx != Some(n_ctx) {
            self.cache = GraphCache::new();
            self.n_ctx = Some(n_ctx);
        }
        &mut self.cache
    }
}

impl Engine {
    /// Load with the process configuration ([`crate::config`]) and no template
    /// override — what a host that has not configured anything gets.
    pub fn load(path: &Path) -> Result<Self, EngineError> {
        Self::load_with(path, OffloadRequest::Default, None, None)
    }

    /// Load with an explicit offload request, KV cache type and chat-template
    /// override. `None` falls back to the process configuration, so the CLI's
    /// `--gpu-layers` / `MINFER_CACHE_TYPE` / `--chat-template` keep working.
    pub fn load_with(
        path: &Path,
        offload: OffloadRequest,
        cache_type: Option<&str>,
        template_override: Option<&str>,
    ) -> Result<Self, EngineError> {
        let gguf = crate::gguf::load_gguf_model(path)?;
        Self::from_gguf(gguf, offload, cache_type, template_override)
    }

    /// Load a model the host already holds in memory (#797) — received over the
    /// network, embedded, or read from a volume the host decrypted itself.
    ///
    /// The buffer is not copied: every weight tensor is a range of it, the same
    /// contract a path load has with its mapping. A split model is refused by
    /// name (its siblings are resolved beside the file, and a bytes load has no
    /// directory). The rest of the load is identical, so a continuation or a turn
    /// behaves exactly as it does on a path load.
    pub fn load_from_bytes(bytes: Vec<u8>) -> Result<Self, EngineError> {
        let gguf = crate::gguf::load_gguf_model_from_bytes(bytes)?;
        Self::from_gguf(gguf, OffloadRequest::Default, None, None)
    }

    /// The shared tail of both loaders: config resolution, the model, the
    /// tokenizer and the validated template. Everything above this is a
    /// difference in where the bytes came from.
    fn from_gguf(
        gguf: crate::gguf::GgufModel,
        offload: OffloadRequest,
        cache_type: Option<&str>,
        template_override: Option<&str>,
    ) -> Result<Self, EngineError> {
        let cache = cache_type
            .map(str::to_string)
            .or_else(|| crate::config::get().kv_cache_type.clone());
        let model = crate::models::load_model_configured(&gguf, "", offload, cache.as_deref())?;
        let tokenizer = Tokenizer::load(&gguf.parts[0].ctx).map_err(EngineError::Tokenizer)?;
        // One resolver: the override beats the GGUF's copy, and whichever wins is
        // validated before the first turn (`template::startup_template`).
        let startup = template::startup_template(
            template::from_gguf(gguf.parts[0].data()).as_deref(),
            template_override,
        )
        .map_err(EngineError::Template)?;
        Ok(Self {
            gguf,
            model,
            tokenizer,
            template: startup.text,
            chatml_notice: startup.announce_chatml,
            generate_cache: Mutex::new(GenerateCache::new()),
        })
    }

    /// The loaded GGUF index (metadata, tensor table, the mapping).
    pub fn gguf(&self) -> &GgufModel {
        &self.gguf
    }
    pub fn model(&self) -> &dyn ModelDef {
        self.model.as_ref()
    }
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
    /// The model's chat template, when it has one (else the ChatML fallback).
    pub fn template(&self) -> Option<&str> {
        self.template.as_deref()
    }
    /// Whether the ChatML fallback is in force (announce it once, as the CLI does).
    pub fn chatml_notice(&self) -> bool {
        self.chatml_notice
    }
    pub fn special_tokens(&self) -> SpecialTokens {
        self.model.special_tokens()
    }
    /// One single-shot generation on this engine (no session state): prefill
    /// `params.tokens`, decode up to `params.n_predict`, stream deltas through
    /// `emit`. The caller owns tokenization and prompt rendering.
    ///
    /// The engine owns the run's KV cache, so repeated runs on one engine build
    /// their graph once; a changed `n_ctx` rebuilds it. Two engines never share
    /// one, and two concurrent runs on the *same* engine serialize on it —
    /// sessions are the independent-concurrency path.
    pub fn generate(
        &self,
        params: &GenerateParams<'_>,
        emit: &mut dyn FnMut(&[u8]),
        on_prefill: &mut dyn FnMut(&[f32], f64),
    ) -> GenerateOutcome {
        let mut state = self.generate_cache.lock().unwrap();
        // The same clamp `generate` applies, so the cache is keyed by the size
        // the regions are actually built for.
        let ctx = params
            .n_ctx
            .max(params.tokens.len())
            .min(self.model.max_seq_len());
        generate(
            self.model.as_ref(),
            &self.tokenizer,
            None,
            params,
            state.for_ctx(ctx),
            emit,
            on_prefill,
        )
    }

    /// A new conversation on this engine. The session owns its KV cache and its
    /// message history; the engine is shared and never mutated by a turn.
    pub fn session(&self, cfg: SessionConfig) -> Session<'_> {
        let special = self.special_tokens();
        let mut eog = vec![special.eos];
        if let Some(im) = special.im_end {
            eog.push(im);
        }
        let eot = special.im_end.unwrap_or(special.eos);
        let bos_text = self
            .tokenizer
            .id_to_token
            .get(self.tokenizer.bos_token as usize)
            .cloned()
            .unwrap_or_default();
        let spec = conversation::ConversationSpec {
            template: self.template.clone(),
            bos_text,
            eog,
            eot,
            seed: cfg.seed,
            n_ctx: cfg.n_ctx,
            mirostat_tau: cfg.mirostat_tau,
            system_prompt: cfg.system_prompt,
        };
        Session {
            engine: self,
            conversation: Conversation::new(spec),
            graph: GraphEngine::new(self.model.as_ref(), cfg.n_ctx),
        }
    }
}

/// What a new session needs to know that the engine cannot answer.
#[derive(Debug, Clone)]
pub struct SessionConfig {
    /// Context size (the KV arena and the causal bound).
    pub n_ctx: usize,
    pub seed: u64,
    /// Initial mirostat `mu` (`2 * tau`); the session owns it across turns.
    pub mirostat_tau: f32,
    /// The system prompt, when the host has one.
    pub system_prompt: Option<String>,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self {
            n_ctx: 4096,
            seed: 42,
            mirostat_tau: 5.0,
            system_prompt: None,
        }
    }
}

/// One conversation: the KV cache plus the message history.
pub struct Session<'a> {
    engine: &'a Engine,
    conversation: Conversation,
    graph: GraphEngine<'a>,
}

impl<'a> Session<'a> {
    /// Run one user turn and stream the answer through `emit` as bytes (a delta
    /// is not guaranteed to be a complete UTF-8 sequence; concatenate before
    /// decoding).
    pub fn chat(
        &mut self,
        input: &str,
        params: &TurnParams,
        emit: &mut dyn FnMut(&[u8]),
    ) -> Result<TurnOutcome, ConvError> {
        let Session {
            engine,
            conversation,
            graph,
        } = self;
        conversation.user_turn(input, &engine.tokenizer, params, graph, emit)
    }

    /// One **raw continuation** on this session's KV (#795): `tokens` are
    /// prefilled at the cursor — no template render, no history entry — and the
    /// run decodes like a turn, streaming through `emit`.
    ///
    /// Only the new tokens are prefilled, so a long prompt can be fed in
    /// pieces; `TurnOutcome::prefill_tokens` counts them. The continuation is
    /// not part of the history: the next [`chat`](Self::chat) rebuilds the KV
    /// from the messages, so a continuation never leaks into a later turn.
    pub fn complete(
        &mut self,
        tokens: &[u32],
        params: &TurnParams,
        emit: &mut dyn FnMut(&[u8]),
    ) -> Result<TurnOutcome, ConvError> {
        let Session {
            engine,
            conversation,
            graph,
        } = self;
        conversation.complete_tokens(tokens, &engine.tokenizer, params, graph, emit)
    }

    /// Tokens written to this session's KV (prompt, generated tokens and the
    /// EOT that closes a turn): the position the next prefill starts at.
    pub fn n_past(&self) -> usize {
        self.conversation.current_pos
    }

    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }
    pub fn conversation_mut(&mut self) -> &mut Conversation {
        &mut self.conversation
    }
    pub fn graph_mut(&mut self) -> &mut GraphEngine<'a> {
        &mut self.graph
    }
    pub fn engine(&self) -> &'a Engine {
        self.engine
    }
}

/// The session's mirostat state is per conversation; this is the helper a host
/// needs when it configures a sampler pipeline itself.
pub fn mirostat_state(tau: f32) -> MirostatState {
    MirostatState::new(tau)
}

mod generate;

#[cfg(test)]
mod tests;
