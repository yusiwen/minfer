//! The engine facade (#763 S6a): a loaded model, and a session that runs turns.
//!
//! This is the L2 tier ADR-0079 describes. `Engine` owns everything a model needs
//! to answer (the mapping, the model, the tokenizer, the chat template); a
//! [`Session`] borrows it and owns the per-conversation state (the KV cache and
//! the message history). A host creates one `Engine` and as many sessions as it
//! has conversations.
//!
//! **Status:** S6a covers loading and `chat`. The raw `complete` entry point and
//! the CLI's single-shot path move here in S6b; the CLI's conversation mode is
//! this facade's first consumer once wired (next increment).

use std::path::Path;

pub use generate::{generate, CancelToken, GenerateOutcome, GenerateParams, SpecStatsOut};

use crate::conversation::{self, ConvError, Conversation, GraphEngine, TurnOutcome, TurnParams};
use crate::gguf::{GgufLoadError, GgufModel};
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
    pub fn generate(
        &self,
        params: &GenerateParams<'_>,
        emit: &mut dyn FnMut(&[u8]),
        on_prefill: &mut dyn FnMut(&[f32], f64),
    ) -> GenerateOutcome {
        generate(
            self.model.as_ref(),
            &self.tokenizer,
            None,
            params,
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
