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
        Self {
            model,
            cache: GraphCache::new(),
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
        self.cache = GraphCache::new();
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
    ContextFull {
        needed: usize,
        available: usize,
    },
    EmptyInput,
    /// F3 (#48): mirostat's per-step `mu` has no home in a verify round, so the
    /// combination is refused loudly instead of silently sampling without it.
    MirostatWithSpec,
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
fn render_messages_with(
    template: Option<&str>,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_text: &str,
) -> String {
    match template {
        Some(t) => template::render_messages(t, messages, add_generation_prompt, bos_text),
        None => template::fallback_chatml_messages(messages, add_generation_prompt),
    }
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
        }
    }

    pub fn is_eog(&self, id: u32) -> bool {
        self.eog.contains(&id)
    }

    /// Fully renders the current messages (with/without the generation prompt).
    fn render_full(&self, add_generation_prompt: bool) -> String {
        render_messages_with(
            self.template.as_deref(),
            &self.messages,
            add_generation_prompt,
            &self.bos_text,
        )
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
        let sampled = sampler::sample_with_config(
            &mut logits,
            &cfg.sampler,
            &self.prev_tokens,
            &mut self.mirostat,
            &mut self.rng,
        );
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
            let (first, dropped_msg) = self.plan_overflow_drop(decoder);
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
            let region = self.overflow_region(first, dropped_msg, decoder);
            self.messages.drain(first..first + dropped_msg);
            let full = self.render_full(true);
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
            let full = self.render_full(true);
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

    /// Context overflow plan: drop the oldest non-system turns (a user + its
    /// assistant pair) until the token count of `render(messages, true)` fits
    /// `n_ctx`. Returns the index of the first message to drop and how many
    /// consecutive messages that is; `(0, 0)` when only `[system?, last user]` is
    /// left and it still does not fit (the caller reports `ContextFull`).
    ///
    /// Planned against a copy of the message list, so the caller can still
    /// resolve the dropped region's token span in the KV *before* the messages
    /// change.
    fn plan_overflow_drop(&self, decoder: &dyn TokenCodec) -> (usize, usize) {
        let mut msgs = self.messages.clone();
        // Indices in the *original* message list: `removed` messages before the
        // current drop have already gone, and drops always take the oldest turn,
        // so `idx + removed` maps the working index back.
        let mut first = usize::MAX;
        let mut end = 0usize;
        let mut removed = 0usize;
        loop {
            let full = render_messages_with(self.template.as_deref(), &msgs, true, &self.bos_text);
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
        if first == usize::MAX {
            (0, 0)
        } else {
            (first, end - first)
        }
    }

    /// Token region `[start, start + len)` of the KV stream that messages
    /// `[first, first + count)` occupy, or `None` when the region cannot be
    /// verified against the stream (see [`Conversation::stream_boundary`]).
    fn overflow_region(
        &self,
        first: usize,
        count: usize,
        decoder: &dyn TokenCodec,
    ) -> Option<(usize, usize)> {
        if count == 0 {
            return None;
        }
        let start = self.stream_boundary(first, decoder)?;
        let end = self.stream_boundary(first + count, decoder)?;
        (end > start).then_some((start, end - start))
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
    fn stream_boundary(&self, upto: usize, decoder: &dyn TokenCodec) -> Option<usize> {
        if upto == 0 {
            return Some(0);
        }
        if upto > self.messages.len() {
            return None;
        }
        let text = render_messages_with(
            self.template.as_deref(),
            &self.messages[..upto],
            false,
            &self.bos_text,
        );
        let toks = decoder.encode(&text);
        self.stream_tokens.starts_with(&toks).then_some(toks.len())
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

    /// The host state a KV session belongs to (C5 S2).
    ///
    /// The container carries the KV *rows*; this is everything the host needs to continue
    /// the conversation those rows were written for — which is what lets a resume prefill
    /// **nothing**. The message list is part of it (not just the token bookkeeping), so a
    /// resumed session can render its next turn's delta from the same history the KV was
    /// built from.
    pub fn snapshot(&self) -> ConversationSnapshot {
        ConversationSnapshot {
            messages: self.messages.clone(),
            stream_tokens: self.stream_tokens.clone(),
            current_pos: self.current_pos,
            turn_pos: self.turn_pos,
            prev_tokens: self.prev_tokens.clone(),
            need_insert_eot: self.need_insert_eot,
        }
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
            let sampled = sampler::sample_with_config(
                &mut logits,
                &cfg.sampler,
                &self.prev_tokens,
                &mut self.mirostat,
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
        /// doc 97: when non-empty the engine carries a speculative draft;
        /// each spec_round pops one pre-accepted batch.
        spec_batches: VecDeque<Vec<u32>>,
        /// C2: when true the mock accepts KV row removals and records them;
        /// false is the plain engine (and GraphEngine without a shiftable
        /// backend), which makes the conversation re-render instead.
        shiftable: bool,
        /// C2: virtual written rows, so `kv_rm` can report the new count.
        rows: usize,
        shifts: Vec<(usize, usize)>,
    }

    impl MockEngine {
        fn new(program: Vec<u32>) -> Self {
            Self {
                program: program.into(),
                calls: Vec::new(),
                resets: 0,
                vocab: 4096,
                spec_batches: VecDeque::new(),
                shiftable: false,
                rows: 0,
                shifts: Vec::new(),
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
            if let Some(&p) = positions.last() {
                self.rows = self.rows.max(p + 1);
            }
            let id = self.program.pop_front().unwrap_or(EOS);
            let mut logits = vec![0.0f32; self.vocab];
            logits[id as usize] = 100.0;
            logits
        }
        fn reset_cache(&mut self) {
            self.resets += 1;
            self.rows = 0;
        }
        fn kv_rm(&mut self, start: usize, len: usize) -> Result<usize, String> {
            if !self.shiftable {
                return Err("mock engine without a KV".to_string());
            }
            assert!(start + len <= self.rows, "removal past the written rows");
            self.shifts.push((start, len));
            self.rows -= len;
            Ok(self.rows)
        }
        fn has_spec(&self) -> bool {
            !self.spec_batches.is_empty()
        }
        fn spec_round(
            &mut self,
            _seed: u32,
            _pos: usize,
            _s: &crate::spec::SpecSampler,
            _prev_tokens: &mut Vec<u32>,
            _rng: &mut StdRng,
        ) -> Option<Vec<u32>> {
            Some(self.spec_batches.pop_front().expect("batch queued"))
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
            // greedy: the mock spike is always picked
            sampler: sampler::SamplerConfig {
                temp: 0.0,
                top_k: 4096,
                top_p: 1.0,
                repeat_penalty: 1.0,
                ..sampler::SamplerConfig::default()
            },
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
            mirostat_tau: 5.0,
            system_prompt: None,
        }
    }

    fn conv(n_ctx: usize) -> Conversation {
        Conversation::new(spec(n_ctx))
    }

    /// The snapshot JSON with its `version` field replaced (the refuse-not-guess test).
    fn json_with_version(json: &str, version: u32) -> String {
        let mut v: serde_json::Value = serde_json::from_str(json).unwrap();
        v["version"] = serde_json::json!(version);
        v.to_string()
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

    /// C5 S2's host-state half: a conversation restored from a session snapshot starts
    /// with **nothing prefilled** and its next turn issues exactly the engine calls the
    /// in-memory run's does (delta prefill + decodes — never a re-render of the history).
    /// The KV bytes that make that legitimate are the C5 container's own gate.
    #[test]
    fn a_resumed_snapshot_prefills_nothing_and_continues_alike() {
        // The in-memory run: one turn, then the snapshot `--session` would write on exit.
        // The mock pops one program token per `forward` — including prefill calls — so the
        // program below is the one that makes turn 2 sample '!' on both sides.
        let mut base = conv(512);
        let mut eng = MockEngine::new(vec![72, 105, EOS, 33, 33, EOS]); // 'H','i',EOG,(EOG write),'!',EOG
        base.start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        let snap = base.snapshot();
        let json = base.snapshot_to_json();
        let parsed = Conversation::snapshot_from_json(&json).expect("the snapshot round-trips");
        assert_eq!(parsed, snap, "JSON must carry the snapshot losslessly");

        // The resumed run: a fresh conversation and a fresh (empty) engine, as a second
        // `--session` process would have.
        let mut resumed = conv(512);
        let mut eng2 = MockEngine::new(vec![33, EOS]);
        resumed.restore_snapshot(&parsed).expect("restore");
        assert!(
            eng2.calls.is_empty() && eng2.resets == 0,
            "restoring a snapshot must not touch the engine at all"
        );
        assert_eq!(resumed.messages, base.messages);
        assert_eq!(resumed.stream_tokens, base.stream_tokens);
        assert_eq!(resumed.current_pos, base.current_pos);
        assert_eq!(resumed.prev_tokens, base.prev_tokens);
        assert_eq!(resumed.need_insert_eot, base.need_insert_eot);

        // The next turn on both: same input, same engine calls, same output.
        let before = eng.calls.len();
        let out_base = base
            .user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        let base_calls: Vec<(Vec<u32>, Vec<usize>)> = eng.calls[before..]
            .iter()
            .map(|(t, p, _)| (t.clone(), p.clone()))
            .collect();
        let out_resumed = resumed
            .user_turn("Q", &FakeCodec, &cfg(), &mut eng2, &mut noop_emit())
            .unwrap();
        let resumed_calls: Vec<(Vec<u32>, Vec<usize>)> = eng2
            .calls
            .iter()
            .map(|(t, p, _)| (t.clone(), p.clone()))
            .collect();
        assert_eq!(
            base_calls, resumed_calls,
            "a resumed turn must issue the same forwards, at the same positions"
        );
        assert_eq!(out_base.text, out_resumed.text, "greedy tokens must agree");
        assert_eq!(out_resumed.text, "!");
        // And the first call is the turn's own delta, not the history re-rendered.
        let full_render = fallback_full(&base.messages);
        assert!(
            resumed_calls[0].0.len() < full_render.len(),
            "the resumed turn prefilled {} token(s) — a re-render would be {}",
            resumed_calls[0].0.len(),
            full_render.len()
        );
        assert_eq!(resumed.messages, base.messages);
        assert_eq!(resumed.stream_tokens, base.stream_tokens);
        assert_eq!(resumed.current_pos, base.current_pos);
    }

    /// C5 S2: a snapshot that contradicts the host mirror is refused *before* it is
    /// applied — the caller then re-seeds, which is always safe.
    #[test]
    fn a_snapshot_that_contradicts_the_host_mirror_is_refused() {
        let mut c = conv(512);
        let good = ConversationSnapshot {
            messages: vec![("user".into(), Some("hi".into()))],
            stream_tokens: vec![1, 2, 3],
            current_pos: 3,
            turn_pos: 0,
            prev_tokens: vec![2, 3],
            need_insert_eot: false,
        };
        c.restore_snapshot(&good)
            .expect("a consistent snapshot applies");

        // current_pos must equal the token stream's length.
        let mut bad = good.clone();
        bad.current_pos = 4;
        let err = c.restore_snapshot(&bad).unwrap_err();
        assert!(err.contains("host mirror"), "{err}");
        // ... and must fit this run's n_ctx.
        let mut bad = good.clone();
        bad.stream_tokens = vec![1; 600];
        bad.current_pos = 600;
        let err = c.restore_snapshot(&bad).unwrap_err();
        assert!(err.contains("n_ctx"), "{err}");
        // ... and turn_pos may not run past it.
        let mut bad = good.clone();
        bad.turn_pos = 9;
        let err = c.restore_snapshot(&bad).unwrap_err();
        assert!(err.contains("turn_pos"), "{err}");

        // A JSON this build does not understand is `None`, never a guess.
        assert!(Conversation::snapshot_from_json("{}").is_none());
        assert!(Conversation::snapshot_from_json("not json").is_none());
        let bumped = json_with_version(&c.snapshot_to_json(), SNAPSHOT_VERSION + 1);
        assert!(Conversation::snapshot_from_json(&bumped).is_none());
        // The refused snapshots above must not have touched the conversation.
        assert_eq!(c.current_pos, 3);
        assert_eq!(c.stream_tokens, vec![1, 2, 3]);
    }

    /// C5 S2: an engine with no arena (the mocks, and any backend that cannot hand its KV
    /// to the host) refuses both calls, so the CLI falls back to re-seeding instead of
    /// pretending the session was resumed.
    #[test]
    fn an_engine_without_a_kv_refuses_the_session_calls() {
        let mut eng = MockEngine::new(vec![]);
        let path = std::path::Path::new("/tmp/minfer-c5s2-not-a-kv-file");
        assert!(eng.kv_save(path, b"{}").is_err());
        assert!(eng.kv_load(path).is_err());
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
    fn spec_rounds_commit_batches_and_stop_at_eog() {
        // doc 97: the spec loop must mirror the plain loop's stream/window
        // semantics. The mock's spec batches replace the sampling: the seed
        // comes from the prefill spike, then each round commits one batch.
        let mut c = conv(512);
        // program[0] spikes the prefill logits → the turn's seed ('z');
        // the spares back the mock's per-forward logits afterwards.
        let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS, EOS]);
        // The prefill consumes program[0] (its spike seeds the turn); the
        // remaining spikes back the mock's per-call logits if a plain forward
        // ever runs (it should not, beyond the EOG slot write).
        eng.spec_batches = vec![
            vec![b'a' as u32, b'b' as u32, b'c' as u32], // round 1 batch
            vec![IM_END],                                // round 2: EOG in-batch
        ]
        .into();
        let out = c
            .start(Some("hi"), &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap()
            .unwrap();
        // The seed ('z', sampled from the prefill spike) is the first
        // committed token; the batch follows it.
        assert_eq!(out.text, "zabc");
        assert!(out.stopped_by_eog);
        assert!(!out.stopped_by_string);
        // The EOG is not counted (plain-loop mirror).
        assert_eq!(out.tokens_generated, 4);
        // The assistant message holds the batch text; the stream holds the
        // committed tokens plus the EOG.
        assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("zabc"));
        assert!(c.stream_tokens.contains(&(b'a' as u32)));
        assert!(c.stream_tokens.contains(&IM_END));
        // need_insert_eot stays false: the turn reached EOG.
        assert!(!c.need_insert_eot);
    }

    #[test]
    fn spec_round_mid_batch_stop_string_truncates() {
        // The stop string lands mid-batch: committed tokens stop at the cut,
        // the stop tokens are not part of the canonical text.
        let mut cfgv = cfg();
        cfgv.stop_strings = vec!["bc".to_string()];
        let mut c = conv(512);
        let mut eng = MockEngine::new(vec![b'z' as u32, EOS, EOS]);
        eng.spec_batches = vec![vec![b'a' as u32, b'b' as u32, b'c' as u32, b'd' as u32]].into();
        let out = c
            .start(Some("hi"), &FakeCodec, &cfgv, &mut eng, &mut noop_emit())
            .unwrap()
            .unwrap();
        // The cut lands mid-batch (at 'c'): committed text stops before the
        // stop string, the batch tail ('d') is discarded uncommitted. The
        // tokens through the stop-completing one are counted (the plain loop
        // increments n_gen before its stop check — mirror exactly).
        assert_eq!(out.text, "za");
        assert!(out.stopped_by_string);
        assert!(!out.stopped_by_eog);
        assert_eq!(out.tokens_generated, 4);
        assert_eq!(c.messages.last().unwrap().1.as_deref(), Some("za"));
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

    /// C2: with a shiftable engine, an overflowing turn removes the dropped
    /// turn's rows from the KV instead of re-rendering, so it prefills only its
    /// own delta — and the resulting stream is still the canonical render of the
    /// new message list (the §5.4 invariant), which is what makes the shortcut
    /// safe to take.
    #[test]
    fn overflow_shift_removes_the_turn_and_prefills_only_the_delta() {
        let mut c = conv(50);
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        assert_eq!(c.stream_tokens.len(), 44);

        let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
        eng3.shiftable = true;
        eng3.rows = c.current_pos; // the stub mirrors the session's written rows
        let out = c
            .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
            .unwrap();

        assert_eq!(out.dropped_turns, 1, "the oldest turn is dropped");
        assert_eq!(eng3.resets, 0, "a shift must not reset the cache");
        assert_eq!(eng3.shifts.len(), 1, "exactly one KV removal");
        let (start, len) = eng3.shifts[0];
        // "hi" and its reply are gone; Q's turn is still at the stream head.
        assert_eq!(start, 0, "Q's turn starts at the stream head");
        assert_eq!(
            len, 23,
            "the dropped turn's span, including its trailing newline"
        );
        // The overflow turn prefills its own delta, not the whole render.
        // (`c.messages` now ends with the new reply; the canonical prompt is the
        // messages as they were when the turn's prefill ran.)
        let full = fallback_full(&c.messages[..c.messages.len() - 1]);
        let delta = format_single(
            None,
            &c.messages[..c.messages.len() - 2],
            ("user".to_string(), Some("X".to_string())),
            true,
            "",
        );
        assert_eq!(
            out.prefill_tokens,
            FakeCodec.encode(&delta.text).len(),
            "only the overflow turn's own delta is prefilled"
        );
        assert!(
            out.prefill_tokens < full.len(),
            "the shift must not re-prefill the render ({} vs {})",
            out.prefill_tokens,
            full.len()
        );

        // The §5.4 invariant survives the removal: the stream is exactly the
        // canonical render of the final message list.
        assert_eq!(c.messages.len(), 4);
        assert_eq!(c.stream_tokens, [&full[..], &[IM_END]].concat());
        assert_eq!(c.current_pos, c.stream_tokens.len());
    }

    /// C2: a system prompt is *not* part of the removed region — the drop starts
    /// after it, so the shift is a middle removal (`start > 0`), and the system
    /// prompt's rows are not even re-roped.
    #[test]
    fn overflow_shift_keeps_the_system_prompt_in_place() {
        let mut spec = spec(60);
        spec.system_prompt = Some("sys".to_string());
        let mut c = Conversation::new(spec);
        // The system prompt is part of the first turn's delta; the boundary check
        // must still resolve it (ChatML renders it as its own block).
        let mut eng = MockEngine::new(vec![IM_END, EOS, IM_END, EOS]);
        eng.shiftable = true;
        c.user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        c.user_turn("Q", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();
        let sys_tokens = canonical(&[("system".into(), Some("sys".into()))]);
        let mut eng3 = MockEngine::new(vec![IM_END, EOS]);
        eng3.shiftable = true;
        eng3.rows = c.current_pos; // the stub mirrors the session's written rows
        let out = c
            .user_turn("X", &FakeCodec, &cfg(), &mut eng3, &mut noop_emit())
            .unwrap();
        assert_eq!(out.dropped_turns, 1);
        assert_eq!(eng3.shifts.len(), 1, "the shift must fire");
        let (start, _) = eng3.shifts[0];
        assert_eq!(
            start,
            sys_tokens.len(),
            "the removed region starts right after the system prompt"
        );
        assert_eq!(
            &c.stream_tokens[..sys_tokens.len()],
            &sys_tokens[..],
            "the system prompt's tokens stay at the head"
        );
        assert_eq!(
            c.messages[0],
            ("system".into(), Some("sys".into())),
            "the system prompt survives the drop"
        );
    }

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

    /// The very first `user_turn` has no KV yet, so it must prefill the whole
    /// render — including anything already in `messages`, i.e. the `--system`
    /// prompt. Before C2 the delta was prefilled on its own, which silently
    /// dropped the system prompt from the KV.
    #[test]
    fn first_user_turn_prefills_the_system_prompt() {
        let mut sp = spec(128);
        sp.system_prompt = Some("be brief".to_string());
        let mut c = Conversation::new(sp);
        let mut eng = MockEngine::new(vec![IM_END]);
        let out = c
            .user_turn("hi", &FakeCodec, &cfg(), &mut eng, &mut noop_emit())
            .unwrap();

        let full = fallback_full(&[
            ("system".into(), Some("be brief".into())),
            ("user".into(), Some("hi".into())),
        ]);
        assert_eq!(
            c.stream_tokens,
            [&full[..], &[IM_END]].concat(),
            "the first turn's KV must be the canonical render + EOG"
        );
        assert_eq!(out.prefill_tokens, full.len());
        let sys = canonical(&[("system".into(), Some("be brief".into()))]);
        assert_eq!(
            &c.stream_tokens[..sys.len()],
            &sys[..],
            "the system prompt must reach the KV"
        );
    }

    /// The locally cached Qwen2.5-0.5B q4_0 the real-model tests run against.
    fn cached_qwen05_q4_0() -> Option<std::path::PathBuf> {
        let home = std::env::var_os("HOME")?;
        let mut p = std::path::PathBuf::from(home);
        p.push(".cache/minfer/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_0.gguf");
        p.exists().then_some(p)
    }

    /// Real-model 2-turn smoke test (part of L2; ignored by default, consistent with the existing realdata tests):
    ///   cargo test --bin minfer conversation_real_model_smoke -- --ignored
    /// Requires the locally cached Qwen2.5-0.5B q4_0 (skips if absent).
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn conversation_real_model_smoke() {
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
            mirostat_tau: 5.0,
            system_prompt: None,
        };
        let mut conv = Conversation::new(spec);
        let mut engine = GraphEngine::new(&*model, 512);
        let tp = TurnParams {
            n_predict: 16, // a short reply suffices (0.5B usually EOGs within a few dozen tokens)
            sampler: sampler::SamplerConfig {
                temp: 0.0, // greedy: deterministic and fast
                top_k: 40,
                top_p: 0.95,
                repeat_penalty: 1.1,
                ..sampler::SamplerConfig::default()
            },
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
        // The model's own stop behaviour is **not** the engine's contract: on these
        // prompts the greedy 0.5B runs to the 16-token cap instead of emitting EOG
        // (t1 stops mid-sentence), so `need_insert_eot` is true. That is a fact about
        // a small model, not a defect — both stop branches are pinned by the
        // scripted-engine tests `first_turn_full_render_and_eog` (EOG -> no EOT) and
        // `n_predict_exhaustion_sets_eot` (cap -> EOT). What this real-model run must
        // hold is the rule that ties the flag to the state the *next* turn reads:
        // the EOT is owed exactly when the stream does not end on an EOG.
        let last = *conv.stream_tokens.last().expect("the session wrote tokens");
        eprintln!(
            "[smoke] t1 stopped_by_eog={} t2 stopped_by_eog={} last_stream_token={last} \
             need_insert_eot={}",
            t1.stopped_by_eog, t2.stopped_by_eog, conv.need_insert_eot
        );
        assert_eq!(
            conv.need_insert_eot,
            !conv.eog.contains(&last),
            "need_insert_eot ({}) must mirror whether the stream ends on an EOG (last stream \
             token {last}, eog {:?})",
            conv.need_insert_eot,
            conv.eog
        );
    }

    /// C2 real-model measurement: with a small context the conversation must
    /// overflow, and the overflowing turn must *shift* the KV window — prefilling
    /// only its own delta — instead of re-prefilling the retained render.
    ///
    ///   cargo test --release --bin minfer context_shift_real_model -- --ignored --nocapture
    ///
    /// This is the check that the shift is actually reachable on a real model:
    /// the token boundaries are re-derived from the chat template and verified
    /// against the KV stream, which a byte-level test codec cannot exercise. The
    /// printed numbers are the ones recorded in the execution plan's C2 record;
    /// the assertion is that the shift fires, not that its logits match a fresh
    /// prefill (they cannot — that is C2's tolerance class).
    #[test]
    #[ignore = "requires the cached 0.5B model (~/.cache/minfer/models)"]
    fn context_shift_real_model_measurement() {
        let Some(path) = cached_qwen05_q4_0() else {
            eprintln!("0.5B q4_0 not cached; skipping the context-shift measurement");
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

        // A context this small overflows after a handful of short turns; the
        // system prompt makes the retained prefix non-empty, which is the
        // conversation case the shift exists for.
        let n_ctx = 192;
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
            n_ctx,
            mirostat_tau: 5.0,
            system_prompt: Some("You are a terse assistant: answer in one short sentence.".into()),
        };
        let mut conv = Conversation::new(spec);
        let mut engine = GraphEngine::new(&*model, n_ctx);
        let tp = TurnParams {
            n_predict: 8, // keep the run short; a few tokens per reply still overflow n_ctx
            sampler: sampler::SamplerConfig {
                temp: 0.0,
                top_k: 40,
                top_p: 0.95,
                repeat_penalty: 1.1,
                ..sampler::SamplerConfig::default()
            },
            stop_strings: Vec::new(),
        };
        let prompts = [
            "The capital of France is",
            "The capital of Japan is",
            "The capital of Italy is",
            "The capital of Spain is",
            "The capital of Egypt is",
            "The capital of Peru is",
            "The capital of Kenya is",
            "The capital of Norway is",
            "The capital of Sweden is",
            "The capital of Greece is",
            "The capital of Poland is",
            "The capital of Portugal is",
        ];
        let mut shifted = 0usize;
        let mut rehydrated = 0usize;
        for (i, p) in prompts.iter().enumerate() {
            let before = conv.stream_tokens.len();
            let out = conv
                .user_turn(p, &tok, &tp, &mut engine, &mut |_| {})
                .expect("turn");
            eprintln!(
                "[c2] real-model turn {i}: prefill {} tokens (stream {before} -> {}), \
                 dropped {} turn(s), reply {:?}",
                out.prefill_tokens,
                conv.stream_tokens.len(),
                out.dropped_turns,
                out.text
            );
            if out.dropped_turns > 0 {
                if out.prefill_tokens < 40 {
                    shifted += 1;
                } else {
                    rehydrated += 1;
                }
            }
            assert_eq!(conv.current_pos, conv.stream_tokens.len());
            assert!(
                conv.stream_tokens.len() <= n_ctx,
                "the KV must stay inside n_ctx"
            );
        }
        assert_eq!(
            rehydrated, 0,
            "an overflow must shift the window, not re-prefill the retained render"
        );
        assert!(shifted > 0, "no turn overflowed n_ctx = {n_ctx}");
        eprintln!(
            "[c2] context shift fired on {shifted} overflow(s); no full re-render was needed"
        );
    }
}
