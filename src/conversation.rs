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
/// What a KV restore handed back (C5 S2): the host state the file carried, the rows it
/// restored, and its size for the caller's log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvRestore {
    pub host: Vec<u8>,
    pub written: usize,
    pub bytes: u64,
}

pub trait Engine {
    /// Returns n_out*nv logits (when n_out=1, the logits of the last/only token).
    /// The real implementation wraps `ModelDef::forward_graph_cached`.
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32>;
    /// Drops all KV state (called on full re-render / `/clear`).
    fn reset_cache(&mut self);
    /// Phase C / C2: remove the KV rows of `[start, start + len)` and re-base the
    /// rows after them by `-len`, re-roping their K, so a token that was at
    /// position `p >= start + len` becomes addressable at `p - len`. Returns the
    /// new written-row count.
    ///
    /// `Err` when this engine has no KV to remove rows from (mock/plain engines,
    /// a fresh cache) or when the backend cannot physically move rows (Metal is
    /// Phase G). The caller then falls back to the exact drop-and-re-render path,
    /// so an unavailable shift costs time and is logged — it is never silent and
    /// never corrupts the session.
    fn kv_rm(&mut self, _start: usize, _len: usize) -> Result<usize, String> {
        Err("this engine has no KV rows to remove".to_string())
    }
    /// C5 S2: write this engine's KV arena to `path`, carrying `host` — the opaque host
    /// state those rows belong to — inside the container. Returns the bytes written.
    ///
    /// `Err` when this engine has no arena to save (mock/plain engines), or when it
    /// carries state the container cannot describe yet (a speculative draft). The caller
    /// logs the reason and keeps going: a session that cannot be snapshotted still works,
    /// it just re-seeds next time.
    fn kv_save(&mut self, _path: &std::path::Path, _host: &[u8]) -> Result<u64, String> {
        Err("this engine has no KV arena to save".to_string())
    }
    /// C5 S2: restore this engine's KV arena from `path` and hand back the host state the
    /// file carries. The arena is left untouched when the file is refused (a failed load
    /// is a no-op), so the caller can fall back to re-seeding.
    fn kv_load(&mut self, _path: &std::path::Path) -> Result<KvRestore, String> {
        Err("this engine has no KV arena to load".to_string())
    }
    /// doc 97: whether speculative rounds are available (mock/plain engines: no).
    fn has_spec(&self) -> bool {
        false
    }
    /// doc 97: one speculative round — draft `d` tokens, verify at nt=d+1,
    /// return the accepted prefix + bonus (the emitted token batch). The
    /// round writes the KV rows for the seed and every batch token except
    /// the LAST one (it becomes the next round's seed). Returns None when
    /// the engine carries no draft.
    fn spec_round(
        &mut self,
        _seed: u32,
        _pos: usize,
        _s: &crate::spec::SpecSampler,
        _prev_tokens: &mut Vec<u32>,
        _rng: &mut StdRng,
    ) -> Option<Vec<u32>> {
        None
    }
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
        // #153: stamp the engine's KV format before the first forward — a `--session`
        // resume reads the session header's element type through the registry's
        // `kv_format` hook (the allocator's stamp), and a per-engine format must be
        // installed by the caller that knows the engine, not read from a global.
        let mut cache = GraphCache::new();
        cache.alloc().set_kv_format(model.kv_format());
        Self {
            model,
            cache,
            n_ctx,
        }
    }
    /// doc 97: the wrapped model (spec rounds drive it directly).
    pub fn model(&self) -> &'a dyn ModelDef {
        self.model
    }
    /// doc 97: the session KV cache (the spec verify writes the same regions).
    pub fn cache_mut(&mut self) -> &mut GraphCache {
        &mut self.cache
    }
    /// doc 97: the session context size (the round's KV horizon).
    pub fn n_ctx(&self) -> usize {
        self.n_ctx
    }
}

/// doc 97: a `GraphEngine` carrying an optional speculative draft
/// (`--cnv --spec-draft <model>`). `forward`/`reset_cache` delegate; the
/// spec round drives the same target model + KV cache the plain path uses,
/// so spec and sequential share one KV state (doc 94's identity contract).
pub struct SpecAwareEngine<'a> {
    inner: GraphEngine<'a>,
    spec: Option<crate::spec::SpecEngine>,
    sparams: crate::spec::SpecSampler,
}

impl<'a> SpecAwareEngine<'a> {
    pub fn new(
        inner: GraphEngine<'a>,
        spec: Option<crate::spec::SpecEngine>,
        sparams: crate::spec::SpecSampler,
    ) -> Self {
        Self {
            inner,
            spec,
            sparams,
        }
    }
}

impl Engine for SpecAwareEngine<'_> {
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32> {
        self.inner.forward(tokens, positions, n_out)
    }
    fn reset_cache(&mut self) {
        self.inner.reset_cache();
        // the draft KV must rewind with the target (positions are absolute)
        if let Some(sp) = self.spec.as_mut() {
            sp.reset_draft();
        }
    }
    fn kv_rm(&mut self, start: usize, len: usize) -> Result<usize, String> {
        // The draft carries its own KV indexed by the same absolute positions;
        // moving the target's rows without moving the draft's would desync the
        // two, so a speculative session re-renders instead (the caller logs it).
        if self.spec.is_some() {
            return Err("context shift with a speculative draft is not wired".to_string());
        }
        self.inner.kv_rm(start, len)
    }
    fn kv_save(&mut self, path: &std::path::Path, host: &[u8]) -> Result<u64, String> {
        // The draft keeps its own KV indexed by the same absolute positions; saving the
        // target's rows without the draft's would resume into a desynchronized pair, so a
        // speculative session refuses to snapshot (the caller re-seeds instead).
        if self.spec.is_some() {
            return Err(
                "KV session: a speculative draft is not part of the container yet — this \
                 session re-seeds instead of resuming"
                    .to_string(),
            );
        }
        self.inner.kv_save(path, host)
    }
    fn kv_load(&mut self, path: &std::path::Path) -> Result<KvRestore, String> {
        if self.spec.is_some() {
            return Err(
                "KV session: a speculative draft is not part of the container yet — this \
                 session re-seeds instead of resuming"
                    .to_string(),
            );
        }
        self.inner.kv_load(path)
    }
    fn has_spec(&self) -> bool {
        self.spec.is_some()
    }
    fn spec_round(
        &mut self,
        seed: u32,
        pos: usize,
        _s: &crate::spec::SpecSampler,
        prev_tokens: &mut Vec<u32>,
        rng: &mut StdRng,
    ) -> Option<Vec<u32>> {
        let model = self.inner.model();
        let n_ctx = self.inner.n_ctx();
        let sparams = &self.sparams;
        let cache = self.inner.cache_mut();
        let sp = self.spec.as_mut()?;
        Some(sp.round(model, cache, seed, pos, n_ctx, sparams, prev_tokens, rng))
    }
}

impl Engine for GraphEngine<'_> {
    fn forward(&mut self, tokens: &[u32], positions: &[usize], n_out: usize) -> Vec<f32> {
        self.model
            .forward_graph_cached(tokens, positions, n_out, self.n_ctx, &mut self.cache)
    }
    fn reset_cache(&mut self) {
        // #153: a fresh cache must carry the engine's format from the start, exactly
        // like `new` (a rebuild must not leave it at the F32 default).
        self.cache = GraphCache::new();
        self.cache.alloc().set_kv_format(self.model.kv_format());
    }
    fn kv_rm(&mut self, start: usize, len: usize) -> Result<usize, String> {
        // The physical removal is host-side by design (memmove + re-rope through
        // the existing `copy_kv_to_cpu` / `write_host` pair), so it needs no
        // backend kernel — but a backend that cannot hand its KV to the host
        // fails here and the caller re-renders.
        let (freq_base, freq_scale) = self.model.rope_params();
        let rope = crate::graph::kvcache::KvRope {
            freq_base,
            freq_scale,
            n_head_kv: self.model.n_head_kv(),
            hd: self.model.n_embd_head(),
            style: self.model.rope_style(),
        };
        self.cache.alloc().kv_rm(start, len, &rope)
    }

    fn kv_save(&mut self, path: &std::path::Path, host: &[u8]) -> Result<u64, String> {
        let report = self.cache.alloc().kv_save_with_host(path, host)?;
        Ok(report.bytes)
    }

    fn kv_load(&mut self, path: &std::path::Path) -> Result<KvRestore, String> {
        let expect = crate::graph::kvsession::expect_for(self.model, self.n_ctx);
        let (host, report) = self.cache.alloc().kv_load_with_host(path, &expect)?;
        Ok(KvRestore {
            host,
            written: report.written,
            bytes: report.bytes,
        })
    }
}

/// Session construction parameters.
#[derive(Clone)]
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
    /// F3 (#48): initial mirostat `mu` is `2 * tau`; the session owns the state
    /// so it survives across turns (each turn's `TurnParams` only carries the
    /// configuration).
    pub mirostat_tau: f32,
    /// The `--system` prompt (used as the first system message).
    pub system_prompt: Option<String>,
}

/// Per-turn sampling/stopping parameters (mapped from the CLI's GenParams).
pub struct TurnParams {
    pub n_predict: usize,
    /// The whole F3 sampler configuration (#48): the six pre-F3 knobs plus
    /// min-p / typical / XTC / DRY / mirostat / logit bias.
    pub sampler: sampler::SamplerConfig,
    pub stop_strings: Vec<String>,
}

/// doc 97: how a speculative batch ended (drives the position bookkeeping).
#[derive(Debug)]
enum BreakKind {
    None,
    Eog,
    Stop,
    Cap,
}

/// The result of one generation turn.
///
/// `text` / `stopped_by_eog` / `stopped_by_string` are emitted by the streaming
/// emit path as locals; the *fields* are read only by tests
/// (`conversation::tests`, `grammar::tests`, `server::batch::tests`), so each
/// carries the `not(test)` allowance rather than a bare one (`[#243]`'s rule,
/// applied in `[#244]`). A caller that wants the structured result instead of the
/// streamed deltas would read them in production.
///
/// [#243]: https://github.com/yusiwen/minfer/issues/243
/// [#244]: https://github.com/yusiwen/minfer/issues/244
#[derive(Debug)]
pub struct TurnOutcome {
    /// The assistant-generated text (after stop-string truncation; without the EOG).
    #[cfg_attr(not(test), allow(dead_code))]
    pub text: String,
    /// Read by `conversation::tests`; the streaming path uses the local.
    #[cfg_attr(not(test), allow(dead_code))]
    pub stopped_by_eog: bool,
    /// Read by `conversation::tests`; the streaming path uses the local.
    #[cfg_attr(not(test), allow(dead_code))]
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
    ContextFull {
        needed: usize,
        available: usize,
    },
    EmptyInput,
    /// F3 (#48): mirostat's per-step `mu` has no home in a verify round, so the
    /// combination is refused loudly instead of silently sampling without it.
    MirostatWithSpec,
    /// F2 (#47): the grammar engine refused a step (no allowed token, a token it
    /// rejects). The turn stops; the emitted text is a valid prefix.
    Grammar(String),
    /// F7 (#50): the model's chat template cannot be rendered (an unsupported
    /// construct, a syntax error). The turn is refused loudly — the engine never
    /// substitutes a generic ChatML prompt.
    Template(String),
}

impl From<template::TemplateError> for ConvError {
    fn from(e: template::TemplateError) -> Self {
        ConvError::Template(e.message())
    }
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
            ConvError::MirostatWithSpec => write!(
                f,
                "mirostat and speculative decoding cannot be combined (mirostat's per-step \
                 state has no home in a verify round)"
            ),
            ConvError::Grammar(msg) => write!(f, "{msg}"),
            ConvError::Template(msg) => write!(f, "{msg}"),
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
    /// F3 (#48): mirostat's running surprise budget, session-scoped (the KV
    /// snapshot does not carry it — neither does it carry `rng`; a resumed
    /// session restarts `mu` at `2 * tau`, which affects sampling only).
    pub mirostat: sampler::MirostatState,
    /// F2 (#47): the grammar automaton's position. Reset at the start of every
    /// assistant turn, so each turn produces a fresh instance of the schema
    /// (a persistent state would be stuck at "accepting" after turn 1).
    pub grammar_state: Option<crate::grammar::GrammarState>,
}

const REPEAT_LAST_N: usize = 64;

/// Version of the [`ConversationSnapshot`] JSON. A snapshot this build does not
/// understand is *not* applied (the caller re-seeds) — the same refuse-not-guess rule the
/// KV container's own version follows.
pub const SNAPSHOT_VERSION: u32 = 1;

/// The host state that belongs to a KV session (C5 S2), as stored in the container's
/// opaque host section. Everything here is host bookkeeping; the model's rows live in
/// the container's payload.
#[derive(Debug, Clone, PartialEq)]
pub struct ConversationSnapshot {
    pub messages: Vec<(String, Option<String>)>,
    pub stream_tokens: Vec<u32>,
    pub current_pos: usize,
    pub turn_pos: usize,
    pub prev_tokens: Vec<u32>,
    pub need_insert_eot: bool,
}

/// Renders a message list the way a session does: the model's own chat template
/// when it has one, the ChatML fallback otherwise.
///
/// F7 (#50): a template that cannot be rendered is a refusal, not a fallback.
fn render_messages_with(
    template: Option<&str>,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_text: &str,
) -> Result<String, template::TemplateError> {
    template::render_messages_opt(template, messages, add_generation_prompt, bos_text)
}

/// Start of the oldest droppable turn: the index of the first user message before
/// the last user message. `None` when only the system prompt and the current user
/// message are left.
fn oldest_droppable_turn_start(messages: &[(String, Option<String>)]) -> Option<usize> {
    let last_user = messages.iter().rposition(|m| m.0 == "user")?;
    (0..last_user).find(|&i| messages[i].0 == "user")
}

/// C2's context shift is on by default (llama.cpp's server does the same);
/// `MINFER_NO_CONTEXT_SHIFT=1` forces the exact drop-and-re-render overflow path.
fn context_shift_enabled() -> bool {
    !std::env::var("MINFER_NO_CONTEXT_SHIFT").map_or(false, |v| v == "1")
}

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
            mirostat: sampler::MirostatState::new(spec.mirostat_tau),
            grammar_state: None,
        }
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    /// Fully renders the current messages (with/without the generation prompt).
    ///
    /// F7 (#50): the template is validated when the session is created, but a
    /// refusal here is still a turn error — never a generic ChatML prompt.
    fn render_full(&self, add_generation_prompt: bool) -> Result<String, ConvError> {
        render_messages_with(
            self.template.as_deref(),
            &self.messages,
            add_generation_prompt,
            &self.bos_text,
        )
        .map_err(|e| ConvError::Template(e.message()))
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
    /// Test-only (#238): driven by `conversation::tests::first_turn_full_render_and_eog` and the `#[ignore]`d `models::qwen2::graph::tail_tests::cuda_conversation_multiturn_reuse`; `#[cfg(test)]` keeps it out of production builds.
    #[cfg(test)]
    pub(crate) fn start(
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
        let full = self.render_full(true)?;
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

    /// doc 97: speculative decode turn — mirrors `generate_assistant_with_logits`
    /// token-for-token (the identity contract) while the Engine's spec round
    /// commits 1..=d+1 tokens per call. Position bookkeeping: `current_pos`
    /// is the slot of the newest committed-but-unwritten token (the round's
    /// seed); the round writes the seed's row plus every batch row except the
    /// last, so after a full batch `current_pos += batch.len()`. At any
    /// turn-ending break the newest committed token is forwarded explicitly
    /// if still unwritten, keeping the §5.4 KV-consistency across turns
    /// (stop-string tokens are never committed; their spec-written rows are
    /// stale and get overwritten before they are ever read).
    fn generate_assistant_spec(
        &mut self,
        decoder: &dyn TokenCodec,
        cfg: &TurnParams,
        engine: &mut dyn Engine,
        emit: &mut dyn FnMut(&[u8]),
        mut logits: Vec<f32>,
    ) -> Result<TurnOutcome, ConvError> {
        let stop_refs: Vec<&[u8]> = Vec::new();
        let _ = stop_refs; // replaced below (borrow of cfg)
        let stop_bytes: Vec<Vec<u8>> = cfg
            .stop_strings
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect();
        let stop_refs: Vec<&[u8]> = stop_bytes.iter().map(|v| v.as_slice()).collect();
        let mut full: Vec<u8> = Vec::new();
        let mut emitted = 0usize;
        let mut n_gen = 0usize;
        let mut stopped_by_eog = false;
        let mut stopped_by_string = false;
        let mut hit_n_predict = false;
        // The seed: sampled from the entry logits exactly like the plain
        // path's first iteration (G1 token identity starts at token 0).
        self.grammar_state = cfg.sampler.grammar.as_ref().map(|g| g.state());
        let sampled = match sampler::sample_with_config_grammar(
            &mut logits,
            &cfg.sampler,
            &self.prev_tokens,
            &mut self.mirostat,
            &mut self.grammar_state,
            &mut self.rng,
        ) {
            Ok(s) => s,
            Err(e) => return Err(ConvError::Grammar(e.to_string())),
        };
        let mut seed = sampled.token_id;
        // Commit the seed (its KV row is written by the first round).
        if self.is_eog(seed) {
            stopped_by_eog = true;
            self.prev_tokens.push(seed);
            if self.prev_tokens.len() > REPEAT_LAST_N {
                self.prev_tokens
                    .drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
            }
            self.stream_tokens.push(seed);
            let _ = engine.forward(&[seed], &[self.current_pos], 1);
            self.current_pos += 1;
        } else {
            n_gen += 1;
            full.extend_from_slice(&decoder.decode_bytes(&[seed]));
            if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                stopped_by_string = true;
                full.truncate(cut);
            } else {
                self.prev_tokens.push(seed);
                if self.prev_tokens.len() > REPEAT_LAST_N {
                    self.prev_tokens
                        .drain(0..self.prev_tokens.len() - REPEAT_LAST_N);
                }
                self.stream_tokens.push(seed);
                let complete =
                    emitted + crate::tokenizer::complete_utf8_prefix_len(&full[emitted..]);
                if complete > emitted {
                    emit(&full[emitted..complete]);
                    emitted = complete;
                }
            }
        }
        loop {
            // Loop-top exits: the newest committed token (the seed) sits at
            // current_pos UNWRITTEN — flush it (one nt=1 forward) so the
            // §5.4 KV-consistency holds across turns.
            if n_gen >= cfg.n_predict {
                hit_n_predict = true;
                let _ = engine.forward(&[seed], &[self.current_pos], 1);
                self.current_pos += 1;
                break;
            }
            if self.current_pos >= self.n_ctx {
                hit_n_predict = true;
                let _ = engine.forward(&[seed], &[self.current_pos], 1);
                self.current_pos += 1;
                break;
            }
            if stopped_by_eog || stopped_by_string {
                break;
            }
            let toks = engine
                .spec_round(
                    seed,
                    self.current_pos,
                    &crate::spec::SpecSampler {
                        cfg: cfg.sampler.clone(),
                    },
                    &mut self.prev_tokens,
                    &mut self.rng,
                )
                .expect("spec engine present (has_spec)");
            let mut consumed = 0usize;
            let mut break_kind = BreakKind::None;
            for (i, &tok) in toks.iter().enumerate() {
                if n_gen >= cfg.n_predict {
                    hit_n_predict = true;
                    break_kind = BreakKind::Cap;
                    break;
                }
                if self.is_eog(tok) {
                    stopped_by_eog = true;
                    // prev_tokens: the round's accept loop already pushed it.
                    self.stream_tokens.push(tok);
                    // The batch's last token is the unwritten seed-slot: write
                    // it explicitly (§5.4, mirroring the plain loop's EOG).
                    if i + 1 == toks.len() {
                        let _ = engine.forward(&[tok], &[self.current_pos + 1 + i], 1);
                    }
                    // The EOG is not counted in tokens_generated (the plain
                    // loop breaks before its `n_gen += 1` — mirror exactly).
                    self.current_pos += 1 + i + 1;
                    break_kind = BreakKind::Eog;
                    break;
                }
                n_gen += 1;
                full.extend_from_slice(&decoder.decode_bytes(&[tok]));
                if let Some(cut) = sampler::match_stop_suffix(&full, &stop_refs) {
                    stopped_by_string = true;
                    full.truncate(cut);
                    // toks[i] is NOT committed (stop strings are not part of
                    // the canonical text); committed slots = seed + toks[0..i].
                    self.current_pos += 1 + i;
                    break_kind = BreakKind::Stop;
                    break;
                }
                // prev_tokens: the round's accept loop already pushed every
                // emitted token (main's consumption loop relies on the same
                // fact — pushing again would duplicate window entries and
                // skew the repeat penalty).
                self.stream_tokens.push(tok);
                let complete =
                    emitted + crate::tokenizer::complete_utf8_prefix_len(&full[emitted..]);
                if complete > emitted {
                    emit(&full[emitted..complete]);
                    emitted = complete;
                }
                consumed = i + 1;
            }
            match break_kind {
                BreakKind::None => {
                    // Full batch committed: toks[len-1] is the new seed
                    // (committed, its row written by the next round).
                    seed = toks[toks.len() - 1];
                    self.current_pos += toks.len();
                }
                BreakKind::Eog => break,
                BreakKind::Stop => break,
                BreakKind::Cap => {
                    // toks[0..consumed] committed and written; the round's
                    // would-be seed was never committed — nothing unwritten.
                    self.current_pos += consumed;
                    break;
                }
            }
        }
        if emitted < full.len() {
            emit(&full[emitted..]);
        }

        let text = String::from_utf8(full.clone())
            .unwrap_or_else(|_| String::from_utf8_lossy(&full).into_owned());
        self.messages
            .push(("assistant".to_string(), Some(text.clone())));
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
        //
        // (C2) When the context is already full the EOT cannot be written yet:
        // the overflow path below makes room first, and the token — which belongs
        // at the *end* of the stream in both cases — commutes with the removal.
        let mut eot_written = false;
        if self.need_insert_eot && self.current_pos < self.n_ctx {
            let _ = engine.forward(&[self.eot], &[self.current_pos], 1);
            self.stream_tokens.push(self.eot);
            self.current_pos += 1;
            self.need_insert_eot = false;
            eot_written = true;
        }

        // 2. Incremental rendering (diff-based, §5.3).
        let delta = format_single(
            self.template.as_deref(),
            &self.messages,
            ("user".to_string(), Some(input.to_string())),
            true,
            &self.bos_text,
        )?;
        let delta_toks = decoder.encode(&delta.text);
        if delta_toks.is_empty() {
            return Err(ConvError::EmptyInput);
        }

        // 3. Prefix mismatch (non-deterministic template) → full re-render fallback (§5.4).
        if !delta.prefix_matched {
            self.messages
                .push(("user".to_string(), Some(input.to_string())));
            let full = self.render_full(true)?;
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

        // 4. Context overflow: drop the oldest non-system turns (§5.7).
        //
        // C2: when the engine can remove KV rows, the retained turns keep their
        // rows — the dropped turn's token region is removed physically and the
        // rows after it are re-based (K re-roped) — so the turn costs one delta
        // prefill instead of a re-prefill of the whole retained conversation.
        // The retained rows keep the values they were computed with, i.e. the
        // influence of the dropped turns: that is C2's named tolerance class
        // (`docs/ARCHITECTURE-EXECUTION-PLAN.md` §5). `MINFER_NO_CONTEXT_SHIFT=1`
        // forces the exact drop-and-re-render path instead.
        let eot_budget = usize::from(self.need_insert_eot);
        if self.current_pos + eot_budget + delta_toks.len() > self.n_ctx {
            self.messages
                .push(("user".to_string(), Some(input.to_string())));
            let (first, dropped_msg) = self.plan_overflow_drop(decoder)?;
            if dropped_msg == 0 {
                // Only [system?, last user] left and it still does not fit: a single message is too long, error out.
                // Roll back: pop the just-pushed user message and reset the EOT flag (if EOT was written this turn,
                // it already closed the previous turn correctly and must not be inserted again next turn).
                self.messages.pop();
                // If the EOT was written this turn it already closed the previous
                // turn; otherwise it is still pending for the next attempt.
                self.need_insert_eot = !eot_written;
                return Err(ConvError::ContextFull {
                    needed: self.current_pos + eot_budget + delta_toks.len(),
                    available: self.n_ctx,
                });
            }
            // Both boundaries are verified against the stream before anything is
            // touched; an unverifiable one (an exotic template, a backend that
            // cannot move rows) falls back to the exact re-render.
            let region = self.overflow_region(first, dropped_msg, decoder)?;
            self.messages.drain(first..first + dropped_msg);
            let full = self.render_full(true)?;
            let toks = decoder.encode(&full);

            let mut used_shift = false;
            let mut shifted_logits = None;
            if context_shift_enabled() {
                match region {
                    Some((start, len))
                        if len > 0
                            && start + len <= self.current_pos
                            && self.current_pos - len + eot_budget + delta_toks.len()
                                <= self.n_ctx =>
                    {
                        match engine.kv_rm(start, len) {
                            Ok(n) => {
                                // `n` is the engine's written-row count. It can
                                // exceed the stream length because a `/regen`
                                // rollback leaves stale rows behind, so the only
                                // invariant to hold it to is that nothing that
                                // was addressable got lost.
                                debug_assert!(
                                    n + len >= self.current_pos,
                                    "the removal dropped addressable rows ({n} + {len} < {})",
                                    self.current_pos
                                );
                                self.stream_tokens.drain(start..start + len);
                                self.current_pos = self.stream_tokens.len();
                                // The pending EOT belongs at the end of the
                                // stream, so it survives the removal and lands at
                                // its shifted position.
                                if self.need_insert_eot {
                                    let _ = engine.forward(&[self.eot], &[self.current_pos], 1);
                                    self.stream_tokens.push(self.eot);
                                    self.current_pos += 1;
                                    self.need_insert_eot = false;
                                }
                                self.turn_pos = self.current_pos;
                                let positions: Vec<usize> =
                                    (self.current_pos..self.current_pos + delta_toks.len()).collect();
                                let logits = engine.forward(&delta_toks, &positions, 1);
                                self.stream_tokens.extend_from_slice(&delta_toks);
                                self.current_pos += delta_toks.len();
                                used_shift = true;
                                shifted_logits = Some(logits);
                                eprintln!(
                                    "[conversation] context shift: dropped {len} KV rows at {start} \
                                     ({dropped_msg} messages), prefill {} tokens instead of {}",
                                    delta_toks.len(),
                                    toks.len()
                                );
                            }
                            Err(e) => eprintln!(
                                "[conversation] context shift unavailable ({e}); re-rendering the retained turns"
                            ),
                        }
                    }
                    // The dropped region could not be located in the KV stream
                    // (or the result would not fit): re-render, which is exact.
                    _ => eprintln!(
                        "[conversation] context shift not applicable (region {region:?} against {} \
                         written rows); re-rendering the retained turns",
                        self.current_pos
                    ),
                }
            }
            let logits = match shifted_logits {
                Some(l) => l,
                None => {
                    // The re-rendered prompt carries the assistant's <|im_end|>,
                    // so a pending EOT is already covered by it.
                    self.need_insert_eot = false;
                    self.rehydrate_full(engine, &toks)
                }
            };
            self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
            let mut out =
                self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
            out.prefill_tokens = if used_shift {
                delta_toks.len()
            } else {
                toks.len()
            };
            out.dropped_turns = dropped_msg / 2;
            return Ok(out);
        }

        // 5. Record the user message, set the rollback point, prefill the delta.
        self.messages
            .push(("user".to_string(), Some(input.to_string())));
        // The very first turn has no KV yet, and `delta_toks` covers only the new
        // message: whatever already sits in `messages` (the `--system` prompt) has
        // to be prefilled too, or the stream would not be the canonical render
        // (§5.4) — and the system prompt would silently never reach the model.
        if self.current_pos == 0 {
            let full = self.render_full(true)?;
            let toks = decoder.encode(&full);
            if toks.is_empty() {
                self.messages.pop();
                return Err(ConvError::EmptyInput);
            }
            let logits = self.rehydrate_full(engine, &toks);
            self.prev_tokens = sampler::recent_window(&self.stream_tokens, REPEAT_LAST_N);
            let mut out =
                self.generate_assistant_with_logits(decoder, cfg, engine, emit, logits)?;
            out.prefill_tokens = toks.len();
            return Ok(out);
        }
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
            let full = self.render_full(true)?;
            decoder.encode(&full)
        } else {
            let delta = format_single(
                self.template.as_deref(),
                &self.messages[..self.messages.len() - 1],
                ("user".to_string(), Some(last_user)),
                true,
                &self.bos_text,
            )?;
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

    /// Context overflow plan: drop the oldest non-system turns (a user + its
    /// assistant pair) until the token count of `render(messages, true)` fits
    /// `n_ctx`. Returns the index of the first message to drop and how many
    /// consecutive messages that is; `(0, 0)` when only `[system?, last user]` is
    /// left and it still does not fit (the caller reports `ContextFull`).
    ///
    /// Planned against a copy of the message list, so the caller can still
    /// resolve the dropped region's token span in the KV *before* the messages
    /// change.
    fn plan_overflow_drop(&self, decoder: &dyn TokenCodec) -> Result<(usize, usize), ConvError> {
        let mut msgs = self.messages.clone();
        // Indices in the *original* message list: `removed` messages before the
        // current drop have already gone, and drops always take the oldest turn,
        // so `idx + removed` maps the working index back.
        let mut first = usize::MAX;
        let mut end = 0usize;
        let mut removed = 0usize;
        loop {
            let full = render_messages_with(self.template.as_deref(), &msgs, true, &self.bos_text)?;
            if decoder.encode(&full).len() <= self.n_ctx {
                break;
            }
            let Some(idx) = oldest_droppable_turn_start(&msgs) else {
                break;
            };
            msgs.remove(idx);
            let mut n = 1;
            if idx < msgs.len() && msgs[idx].0 == "assistant" {
                msgs.remove(idx);
                n = 2;
            }
            if first == usize::MAX {
                first = idx + removed;
            }
            removed += n;
            end = idx + removed;
        }
        Ok(if first == usize::MAX {
            (0, 0)
        } else {
            (first, end - first)
        })
    }

    /// Token region `[start, start + len)` of the KV stream that messages
    /// `[first, first + count)` occupy, or `None` when the region cannot be
    /// verified against the stream (see [`Conversation::stream_boundary`]).
    fn overflow_region(
        &self,
        first: usize,
        count: usize,
        decoder: &dyn TokenCodec,
    ) -> Result<Option<(usize, usize)>, ConvError> {
        if count == 0 {
            return Ok(None);
        }
        let (Some(start), Some(end)) = (
            self.stream_boundary(first, decoder)?,
            self.stream_boundary(first + count, decoder)?,
        ) else {
            return Ok(None);
        };
        Ok((end > start).then_some((start, end - start)))
    }

    /// Token offset at which message `upto` starts in the KV stream, verified:
    /// `messages[..upto]` must render to a token *prefix* of the stream.
    ///
    /// The stream is the canonical render (§5.4), but it was built from
    /// incremental deltas, so a boundary is only usable when re-encoding the
    /// render of the messages before it reproduces the stream's first tokens
    /// exactly. A template whose deltas do not line up that way — or a session
    /// state where they do not — yields `None`, and the caller falls back to the
    /// exact re-render path instead of shifting the wrong rows.
    fn stream_boundary(
        &self,
        upto: usize,
        decoder: &dyn TokenCodec,
    ) -> Result<Option<usize>, ConvError> {
        if upto == 0 {
            return Ok(Some(0));
        }
        if upto > self.messages.len() {
            return Ok(None);
        }
        let text = render_messages_with(
            self.template.as_deref(),
            &self.messages[..upto],
            false,
            &self.bos_text,
        )?;
        let toks = decoder.encode(&text);
        Ok(self.stream_tokens.starts_with(&toks).then_some(toks.len()))
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

    /// Serialize [`Self::snapshot`] for the container's opaque host section. JSON for
    /// the same reason the history is: it is the host's own bookkeeping, it is small, and
    /// a human inspecting a session file can read it. The `version` field is the
    /// refuse-not-guess marker a second format change needs.
    pub fn snapshot_to_json(&self) -> String {
        let msgs: Vec<serde_json::Value> = self
            .messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect();
        let v = serde_json::json!({
            "version": SNAPSHOT_VERSION,
            "messages": msgs,
            "stream_tokens": self.stream_tokens,
            "current_pos": self.current_pos,
            "turn_pos": self.turn_pos,
            "prev_tokens": self.prev_tokens,
            "need_insert_eot": self.need_insert_eot,
        });
        serde_json::to_string(&v).unwrap_or_default()
    }

    /// Parse a host section written by [`Self::snapshot_to_json`]. `None` on anything
    /// that is not a snapshot this build understands — the caller falls back to
    /// re-seeding, loudly.
    pub fn snapshot_from_json(json: &str) -> Option<ConversationSnapshot> {
        let v: serde_json::Value = serde_json::from_str(json).ok()?;
        if v.get("version")?.as_u64()? != SNAPSHOT_VERSION as u64 {
            return None;
        }
        let messages = v
            .get("messages")?
            .as_array()?
            .iter()
            .map(|m| {
                let role = m.get("role")?.as_str()?.to_string();
                let content = m
                    .get("content")
                    .and_then(|c| c.as_str())
                    .map(str::to_string);
                Some((role, content))
            })
            .collect::<Option<Vec<_>>>()?;
        let tokens = |key: &str| -> Option<Vec<u32>> {
            v.get(key)?
                .as_array()?
                .iter()
                .map(|x| x.as_u64().map(|n| n as u32))
                .collect()
        };
        Some(ConversationSnapshot {
            messages,
            stream_tokens: tokens("stream_tokens")?,
            current_pos: v.get("current_pos")?.as_u64()? as usize,
            turn_pos: v.get("turn_pos")?.as_u64()? as usize,
            prev_tokens: tokens("prev_tokens")?,
            need_insert_eot: v.get("need_insert_eot")?.as_bool()?,
        })
    }

    /// Apply a snapshot restored from a KV container (C5 S2). The KV rows are already in
    /// the engine; this puts the host back in the state they were written for, so the
    /// next turn prefills only its own delta.
    ///
    /// Validated before it is applied: a snapshot that contradicts the invariant
    /// `current_pos == stream_tokens.len()`, or that claims positions this conversation's
    /// `n_ctx` cannot hold, is refused — the caller then re-seeds, which is always safe.
    pub fn restore_snapshot(&mut self, snap: &ConversationSnapshot) -> Result<(), String> {
        if snap.current_pos != snap.stream_tokens.len() {
            return Err(format!(
                "session snapshot: {} written positions for {} stream tokens (the host mirror \
                 is inconsistent)",
                snap.current_pos,
                snap.stream_tokens.len()
            ));
        }
        if snap.current_pos > self.n_ctx {
            return Err(format!(
                "session snapshot: {} written positions do not fit this run's n_ctx {}",
                snap.current_pos, self.n_ctx
            ));
        }
        if snap.turn_pos > snap.current_pos {
            return Err(format!(
                "session snapshot: turn_pos {} is past the {} written positions",
                snap.turn_pos, snap.current_pos
            ));
        }
        self.messages = snap.messages.clone();
        self.stream_tokens = snap.stream_tokens.clone();
        self.current_pos = snap.current_pos;
        self.turn_pos = snap.turn_pos;
        self.prev_tokens = snap.prev_tokens.clone();
        self.need_insert_eot = snap.need_insert_eot;
        Ok(())
    }

    /// Loads history and **fully re-renders** the KV (§5.8: KV state is not serialized — thanks to the §5.4 invariant,
    /// the re-render matches continuing the session, just with one extra prefill).
    pub fn load_history(
        &mut self,
        messages: Vec<(String, Option<String>)>,
        decoder: &dyn TokenCodec,
        engine: &mut dyn Engine,
    ) -> Result<(), ConvError> {
        self.messages = messages;
        self.need_insert_eot = false;
        self.turn_pos = 0;
        let full = self.render_full(false)?;
        let toks = decoder.encode(&full);
        let _ = self.rehydrate_full(engine, &toks);
        Ok(())
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
        // doc 97: speculative rounds when the engine carries a draft. The
        // spec loop is a sibling of the plain loop (not a branch inside it)
        // because a round commits a BATCH of pre-sampled tokens — the
        // per-token machinery (EOG / stop strings / penalty window / UTF-8
        // holdback) is mirrored token-for-token, so the emitted stream is
        // byte-identical (doc 94/95 identity contract).
        if engine.has_spec() {
            // F3 (#48): mirostat's per-step state has no home in a verify round;
            // refuse rather than sample without it (`SpecSampler` carries no mu).
            if cfg.sampler.mirostat != sampler::MirostatMode::Off {
                return Err(ConvError::MirostatWithSpec);
            }
            return self.generate_assistant_spec(decoder, cfg, engine, emit, logits);
        }
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
        // F2: a fresh automaton position for this turn (see the field's note).
        self.grammar_state = cfg.sampler.grammar.as_ref().map(|g| g.state());

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
            let sampled = match sampler::sample_with_config_grammar(
                &mut logits,
                &cfg.sampler,
                &self.prev_tokens,
                &mut self.mirostat,
                &mut self.grammar_state,
                &mut self.rng,
            ) {
                Ok(s) => s,
                Err(e) => return Err(ConvError::Grammar(e.to_string())),
            };
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
mod tests;
