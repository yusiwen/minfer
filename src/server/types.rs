//! OpenAI-compatible request/response types, sampling-parameter resolution and
//! validation (OPENAI-CHAT-API-PLAN.md §Data Structures, §Error Handling).
//!
//! Defaults deliberately match llama.cpp / the minfer CLI (temperature 0.8,
//! top_p 0.95, top_k 40, repeat_penalty 1.1, max_tokens = no limit) rather
//! than the OpenAI spec — see the plan's "Request/Response Format" section.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// === Request Types ===

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    /// The client's own message objects (#613): role/content plus any extra key
    /// (`tool_calls`, `name`, a vendor extension), key order preserved.
    pub messages: Vec<crate::template::ToolJson>,
    pub model: Option<String>,
    pub stream: Option<bool>,
    pub max_tokens: Option<i64>, // default -1 = no limit (llama.cpp n_predict)
    pub temperature: Option<f32>,
    pub top_k: Option<u32>,
    pub top_p: Option<f32>,
    pub repeat_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
    #[serde(alias = "stop_sequences")]
    pub stop: Option<StopCondition>,
    pub seed: Option<u64>,
    // === F3 sampler set (#48). All optional; omitting them leaves the
    // pre-F3 chain bit-identical. ===
    pub min_p: Option<f32>,
    pub typical_p: Option<f32>,
    pub xtc_probability: Option<f32>,
    pub xtc_threshold: Option<f32>,
    pub dry_multiplier: Option<f32>,
    pub dry_base: Option<f32>,
    pub dry_allowed_length: Option<usize>,
    pub dry_penalty_last_n: Option<usize>,
    /// DRY restart sequences. Both spellings are accepted and are never guessed
    /// from each other's characters (#118): an array of **strings** is
    /// llama.cpp's OpenAI-compatible form and is resolved against the vocabulary
    /// per request, while the array of **token-id arrays** is minfer's original
    /// form and is used as given. An empty array is a `400` in either shape.
    pub dry_sequence_breakers: Option<DrySequenceBreakers>,
    /// 0 = off, 1 = mirostat v1, 2 = mirostat v2.
    pub mirostat: Option<i64>,
    pub mirostat_tau: Option<f32>,
    pub mirostat_eta: Option<f32>,
    pub mirostat_m: Option<usize>,
    /// OpenAI's `logit_bias`: `{"<token_id>": bias}`. A key that is not a token
    /// id, or an id outside the vocabulary, is a `400`.
    pub logit_bias: Option<HashMap<String, f32>>,
    /// F2 (#47): OpenAI structured output. `{"type":"text"}` (or absent) leaves
    /// sampling unconstrained, `{"type":"json_object"}` admits any JSON value,
    /// and `{"type":"json_schema","json_schema":{"name":…,"schema":{…}}}` admits
    /// the compiled schema.
    pub response_format: Option<ResponseFormat>,
    /// F2: llama.cpp-style GBNF extension field. Mutually exclusive with a
    /// non-text `response_format` (one grammar per request, never a precedence
    /// rule).
    pub grammar: Option<String>,
    // === Tool calling (#597). Absent means the template's `tools` is `none`,
    // which is the pinned behaviour every byte-for-byte gate renders. ===
    /// OpenAI `tools`: the request's own array of tool definitions, key order
    /// and unknown keys included, because the template echoes each entry into
    /// the prompt (`src/template/tool_json.rs`).
    pub tools: Option<Vec<crate::template::ToolJson>>,
    /// OpenAI `tool_choice`. See [`ChatCompletionRequest::prompt_tools`] for
    /// which modes this server can honour, and what the others mean.
    pub tool_choice: Option<ToolChoice>,
}

/// OpenAI `tool_choice`, in the two shapes the API defines.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(untagged)]
pub enum ToolChoice {
    /// `"auto"` / `"none"` / `"required"`.
    Mode(ToolChoiceMode),
    /// `{"type":"function","function":{"name":"…"}}` — a forced call.
    Function(ForcedToolChoice),
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ToolChoiceMode {
    Auto,
    None,
    Required,
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct ForcedToolChoice {
    /// OpenAI spells this `"function"`; kept so an unknown type is a `400` with
    /// the value named rather than a deserialization failure with no context.
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ForcedToolName,
}

/// What `tool_choice` asks of the generation (#614).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolChoiceIntent {
    /// No constraint; the model decides.
    Auto,
    /// Render without tools.
    None,
    /// The model must call one of the offered tools.
    Required,
    /// The model must call this tool.
    Forced(String),
}

impl ToolChoiceIntent {
    /// `Some(None)` = required, `Some(Some(name))` = forced, `None` = free.
    pub fn constraint(&self) -> Option<Option<&str>> {
        match self {
            Self::Required => Some(None),
            Self::Forced(name) => Some(Some(name.as_str())),
            Self::Auto | Self::None => None,
        }
    }
}

#[derive(Debug, Deserialize, PartialEq, Eq)]
pub struct ForcedToolName {
    pub name: String,
}

/// OpenAI `response_format` (the subset this server implements).
#[derive(Debug, Deserialize, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseFormat {
    /// No constraint.
    Text,
    /// Any single JSON value (`root ::= j-value`).
    JsonObject,
    /// A JSON Schema.
    JsonSchema { json_schema: JsonSchemaSpec },
}

#[derive(Debug, Deserialize, Serialize)]
pub struct JsonSchemaSpec {
    /// OpenAI requires a name; it does not change the accepted language.
    #[serde(default)]
    pub name: Option<String>,
    /// The schema itself. `strict` is accepted and ignored (this compiler is
    /// always strict about what it supports).
    pub schema: serde_json::Value,
    #[serde(default)]
    pub strict: Option<bool>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopCondition {
    String(String),
    Array(Vec<String>),
}

/// The two `dry_sequence_breakers` shapes, told apart by JSON type — never by
/// looking at a string's characters (#118).
///
/// * [`Self::Strings`] — `["\n", ":"]`, llama.cpp's OpenAI-compatible shape.
///   Each breaker is matched through the vocabulary when the request is handled
///   (the tokenizer is per server, not per request), and a breaker that matches
///   nothing is a `400`.
/// * [`Self::Ids`] — `[[198],[13,2]]`, minfer's original shape: the token
///   sequences are already resolved, so no tokenizer is involved.
///
/// A mixed array (`["\n", [198]]`) matches neither variant and fails to
/// deserialize, which the request path reports as an invalid body.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum DrySequenceBreakers {
    /// Breaker strings (llama.cpp's form).
    Strings(Vec<String>),
    /// Resolved token-id sequences (minfer's form).
    Ids(Vec<Vec<u32>>),
}

/// Concrete sampling parameters, resolved once per request.
#[derive(Clone, Debug)]
pub struct SamplingParams {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    // === F3 sampler set (#48) ===
    pub min_p: f32,
    pub typical_p: f32,
    pub xtc_probability: f32,
    pub xtc_threshold: f32,
    pub dry_multiplier: f32,
    pub dry_base: f32,
    pub dry_allowed_length: usize,
    pub dry_penalty_last_n: usize,
    pub dry_breakers: Vec<Vec<u32>>,
    /// #118: the `dry_sequence_breakers` **string** form, kept unresolved until
    /// [`Self::resolve_dry_breakers`] runs where the tokenizer lives. Empty when
    /// the request used the token-id form or omitted the field.
    pub dry_breaker_strings: Vec<String>,
    pub mirostat: crate::sampler::MirostatMode,
    pub mirostat_tau: f32,
    pub mirostat_eta: f32,
    pub mirostat_m: usize,
    pub logit_bias: Vec<(u32, f32)>,
    /// F2 (#47): the raw grammar request, compiled against the vocabulary by
    /// [`Self::compile_grammar`] once the tokenizer is available. `None` for an
    /// unconstrained request, so the default path is bit-identical.
    pub grammar_source: Option<crate::grammar::GrammarSource>,
    /// F2: the compiled grammar, filled in by [`Self::compile_grammar`].
    pub grammar: Option<std::sync::Arc<crate::grammar::Grammar>>,
    pub seed: u64,
    pub stop_strings: Vec<String>,
    pub max_tokens: i64,
}

impl SamplingParams {
    /// The sampler configuration these parameters describe — the single place
    /// the server maps its request fields onto the sampler's config.
    pub fn sampler_config(&self) -> crate::sampler::SamplerConfig {
        crate::sampler::SamplerConfig {
            temp: self.temp,
            top_k: self.top_k,
            top_p: self.top_p,
            repeat_penalty: self.repeat_penalty,
            frequency_penalty: self.frequency_penalty,
            presence_penalty: self.presence_penalty,
            min_p: self.min_p,
            typical_p: self.typical_p,
            dry_multiplier: self.dry_multiplier,
            dry_base: self.dry_base,
            dry_allowed_length: self.dry_allowed_length,
            dry_penalty_last_n: self.dry_penalty_last_n,
            dry_breakers: self.dry_breakers.clone(),
            xtc_probability: self.xtc_probability,
            xtc_threshold: self.xtc_threshold,
            mirostat: self.mirostat,
            mirostat_tau: self.mirostat_tau,
            mirostat_eta: self.mirostat_eta,
            mirostat_m: self.mirostat_m,
            logit_bias: self.logit_bias.clone(),
            grammar: self.grammar.clone(),
        }
    }

    /// #118: resolve the request's `dry_sequence_breakers` **string** form into
    /// token sequences, now that the vocabulary is available.
    ///
    /// Called on the handler side, like [`Self::compile_grammar`], so a breaker
    /// that matches no token (or exceeds `Tokenizer`'s length limits) is a `400`
    /// before the job is queued. The token-id form is already resolved and needs
    /// no tokenizer; a request that used it returns here immediately. Sequences
    /// are deduplicated against the token-id ones, so a request that sends both
    /// shapes never doubles a breaker.
    pub fn resolve_dry_breakers(
        &mut self,
        tokenizer: &crate::tokenizer::Tokenizer,
    ) -> Result<(), ApiError> {
        if self.dry_breaker_strings.is_empty() {
            return Ok(());
        }
        let seqs = tokenizer
            .resolve_sequence_breakers(&self.dry_breaker_strings)
            .map_err(ApiError::invalid_request)?;
        for seq in seqs {
            if !self.dry_breakers.contains(&seq) {
                self.dry_breakers.push(seq);
            }
        }
        Ok(())
    }

    /// F2 (#47): compile the request's grammar/schema against the vocabulary.
    /// Called on the handler side (where the tokenizer lives) before the job is
    /// queued, so an unsupported construct is a `400` and never reaches a slot.
    pub fn compile_grammar(
        &mut self,
        tokenizer: &crate::tokenizer::Tokenizer,
        special: &crate::models::SpecialTokens,
    ) -> Result<(), ApiError> {
        let Some(src) = self.grammar_source.clone() else {
            return Ok(());
        };
        let g = crate::grammar::compile_source(&src, tokenizer, special)
            .map_err(ApiError::invalid_request)?;
        self.grammar = Some(std::sync::Arc::new(g));
        Ok(())
    }

    /// Refuse a nonsensical sampler configuration with a `400` — the same strict
    /// validation the CLI runs, plus the vocabulary check for `logit_bias` ids
    /// (the vocabulary is known per server, not per request).
    pub fn validate(&self, n_vocab: usize) -> Result<(), ApiError> {
        let cfg = self.sampler_config();
        cfg.validate().map_err(ApiError::invalid_request)?;
        cfg.validate_logit_bias(n_vocab)
            .map_err(ApiError::invalid_request)
    }
}

// llama.cpp / minfer CLI defaults (common/common.h).
pub const DEFAULT_TEMP: f32 = 0.8;
pub const DEFAULT_TOP_K: usize = 40;
pub const DEFAULT_TOP_P: f32 = 0.95;
pub const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub const MAX_TOKENS_UNLIMITED: i64 = -1;

const VALID_ROLES: [&str; 5] = ["system", "user", "assistant", "tool", "developer"];

impl ChatCompletionRequest {
    /// Parse + validate a raw JSON body. Returns a 400 `invalid_request_error`
    /// on malformed JSON, missing/empty messages, unknown roles, or non-string
    /// content (multimodal arrays are rejected per the plan's Non-Goals).
    pub fn parse(body: &[u8]) -> Result<Self, ApiError> {
        let req: ChatCompletionRequest = serde_json::from_slice(body)
            .map_err(|e| ApiError::invalid_request(format!("invalid request body: {e}")))?;
        if req.messages.is_empty() {
            return Err(ApiError::invalid_request(
                "messages must not be empty".to_string(),
            ));
        }
        for (index, message) in req.messages.iter().enumerate() {
            validate_message(index, message)?;
        }
        if let Some(tools) = &req.tools {
            // "No tools" is the absence of the field, not an empty array: an
            // empty array is a client bug that would otherwise render as a
            // tools branch with nothing in it (#597).
            if tools.is_empty() {
                return Err(ApiError::invalid_request(
                    "tools must not be empty; omit the field to send none".to_string(),
                ));
            }
            for (index, tool) in tools.iter().enumerate() {
                validate_tool(index, tool)?;
            }
        }
        if req.tool_choice.is_some() && req.tools.is_none() {
            return Err(ApiError::invalid_request(
                "`tool_choice` requires `tools`".to_string(),
            ));
        }
        Ok(req)
    }

    /// The `tools` this request puts in the prompt, or `None` (#597).
    ///
    /// `tool_choice: "none"` renders without them; every other mode renders them,
    /// including `"required"` and a forced function, which are enforced by a
    /// sampling grammar (#614, [`Self::resolve_tool_choice`]).
    pub fn prompt_tools(&self) -> Result<Option<&[crate::template::ToolJson]>, ApiError> {
        let tools = self.tools.as_deref();
        match &self.tool_choice {
            Some(ToolChoice::Mode(ToolChoiceMode::None)) => Ok(None),
            _ => Ok(tools),
        }
    }

    /// What `tool_choice` asks for, resolved against the offered tools (#614).
    ///
    /// `"required"` and a forced function become a **sampling constraint**: the
    /// handler builds a GBNF for the model's `<tool_call>` syntax, so the call the
    /// parser then extracts is the one the client demanded. A forced name that is
    /// not offered, or a forced-choice object whose `type` is not `"function"`, is
    /// a `400` naming it — never a silently ignored field.
    pub fn resolve_tool_choice(&self) -> Result<ToolChoiceIntent, ApiError> {
        use crate::template::ToolJson;
        match &self.tool_choice {
            None | Some(ToolChoice::Mode(ToolChoiceMode::Auto)) => Ok(ToolChoiceIntent::Auto),
            Some(ToolChoice::Mode(ToolChoiceMode::None)) => Ok(ToolChoiceIntent::None),
            Some(ToolChoice::Mode(ToolChoiceMode::Required)) => Ok(ToolChoiceIntent::Required),
            Some(ToolChoice::Function(forced)) => {
                if forced.kind != "function" {
                    return Err(ApiError::invalid_request(format!(
                        "tool_choice.type is '{}'; this server implements 'function' only",
                        forced.kind
                    )));
                }
                let name = forced.function.name.trim();
                if name.is_empty() {
                    return Err(ApiError::invalid_request(
                        "tool_choice.function.name must be a non-empty string".to_string(),
                    ));
                }
                let offered = self.tools.as_deref().unwrap_or(&[]);
                let found = offered.iter().any(|tool| {
                    tool.get("function")
                        .and_then(|function| function.get("name"))
                        .and_then(ToolJson::as_str)
                        == Some(name)
                });
                if !found {
                    return Err(ApiError::invalid_request(format!(
                        "tool_choice forces '{name}', which is not in `tools`"
                    )));
                }
                Ok(ToolChoiceIntent::Forced(name.to_string()))
            }
        }
    }

    /// Resolve request options into concrete sampling params. `rng_seed` is the
    /// default seed when the request omits `seed` (random, not the CLI's 42).
    ///
    /// Fallible from F3 on: an unknown `mirostat` mode or a `logit_bias` key
    /// that is not a token id is a `400` here, never a silently ignored option.
    /// Range validation runs in [`SamplingParams::validate`], which also needs
    /// the vocabulary size.
    pub fn resolve(&self, rng_seed: u64) -> Result<SamplingParams, ApiError> {
        let stop_strings = match &self.stop {
            None => Vec::new(),
            Some(StopCondition::String(s)) => vec![s.clone()],
            Some(StopCondition::Array(v)) => v.clone(),
        };
        let mirostat = match self.mirostat {
            None => crate::sampler::MirostatMode::Off,
            Some(v) => crate::sampler::MirostatMode::parse(v).map_err(ApiError::invalid_request)?,
        };
        let mut logit_bias = Vec::new();
        if let Some(m) = &self.logit_bias {
            for (key, bias) in m {
                let id = key.parse::<u32>().map_err(|_| {
                    ApiError::invalid_request(format!(
                        "logit_bias key '{key}' is not a token id (expected a decimal integer)"
                    ))
                })?;
                logit_bias.push((id, *bias));
            }
            logit_bias.sort_by_key(|(id, _)| *id);
        }
        // F2 (#47): one grammar per request. `grammar` and a non-text
        // `response_format` together are a 400 — never a silent precedence rule.
        if self.grammar.is_some()
            && !matches!(self.response_format, None | Some(ResponseFormat::Text))
        {
            return Err(ApiError::invalid_request(
                "`grammar` and a non-text `response_format` are mutually exclusive; send one"
                    .to_string(),
            ));
        }
        let grammar_source = match (&self.grammar, &self.response_format) {
            (Some(g), _) => Some(crate::grammar::GrammarSource::Gbnf(g.clone())),
            (None, Some(ResponseFormat::JsonObject)) => {
                Some(crate::grammar::GrammarSource::AnyJson)
            }
            (None, Some(ResponseFormat::JsonSchema { json_schema })) => Some(
                crate::grammar::GrammarSource::Json(json_schema.schema.clone()),
            ),
            (None, Some(ResponseFormat::Text)) | (None, None) => None,
        };
        // #118: the two `dry_sequence_breakers` shapes, told apart by JSON type.
        // The string form stays unresolved until `resolve_dry_breakers` runs
        // where the tokenizer lives. An explicitly empty array is a `400` in
        // either shape (llama.cpp refuses it too) — "no breakers" is expressed
        // by omitting the field.
        let (dry_breakers, dry_breaker_strings) = match &self.dry_sequence_breakers {
            None => (Vec::new(), Vec::new()),
            Some(DrySequenceBreakers::Ids(v)) => {
                if v.is_empty() {
                    return Err(ApiError::invalid_request(
                        "dry_sequence_breakers must not be empty; omit the field to use none"
                            .to_string(),
                    ));
                }
                (v.clone(), Vec::new())
            }
            Some(DrySequenceBreakers::Strings(v)) => {
                if v.is_empty() {
                    return Err(ApiError::invalid_request(
                        "dry_sequence_breakers must not be empty; omit the field to use none"
                            .to_string(),
                    ));
                }
                (Vec::new(), v.clone())
            }
        };
        Ok(SamplingParams {
            temp: self.temperature.unwrap_or(DEFAULT_TEMP),
            top_k: self.top_k.map(|k| k as usize).unwrap_or(DEFAULT_TOP_K),
            top_p: self.top_p.unwrap_or(DEFAULT_TOP_P),
            repeat_penalty: self.repeat_penalty.unwrap_or(DEFAULT_REPEAT_PENALTY),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            min_p: self.min_p.unwrap_or(0.0),
            typical_p: self.typical_p.unwrap_or(1.0),
            xtc_probability: self.xtc_probability.unwrap_or(0.0),
            xtc_threshold: self.xtc_threshold.unwrap_or(0.5),
            dry_multiplier: self.dry_multiplier.unwrap_or(0.0),
            dry_base: self.dry_base.unwrap_or(1.75),
            dry_allowed_length: self.dry_allowed_length.unwrap_or(2),
            dry_penalty_last_n: self.dry_penalty_last_n.unwrap_or(64),
            dry_breakers,
            dry_breaker_strings,
            mirostat,
            mirostat_tau: self.mirostat_tau.unwrap_or(5.0),
            mirostat_eta: self.mirostat_eta.unwrap_or(0.1),
            mirostat_m: self.mirostat_m.unwrap_or(100),
            logit_bias,
            grammar_source,
            grammar: None,
            seed: self.seed.unwrap_or(rng_seed),
            stop_strings,
            max_tokens: self.max_tokens.unwrap_or(MAX_TOKENS_UNLIMITED),
        })
    }
}

/// One request message (#613).
///
/// The object is carried verbatim into the template, so only the two fields the
/// engine itself reads are validated: `role` (a known role) and `content` (a
/// string, or `null`/absent for a tool-call turn). Every other key — a client's
/// `tool_calls`, `name`, a vendor extension — passes through untouched; a template
/// that echoes the message sees exactly what the client sent.
fn validate_message(index: usize, message: &crate::template::ToolJson) -> Result<(), ApiError> {
    use crate::template::ToolJson;
    if message.as_object().is_none() {
        return Err(ApiError::invalid_request(format!(
            "messages[{index}] must be an object with `role` and `content`"
        )));
    }
    match message.get("role").and_then(ToolJson::as_str) {
        Some(role) if VALID_ROLES.contains(&role) => {}
        Some(role) => {
            return Err(ApiError::invalid_request(format!(
                "unknown message role '{role}' at messages[{index}]"
            )))
        }
        None => {
            return Err(ApiError::invalid_request(format!(
                "messages[{index}] has no string `role`"
            )))
        }
    }
    if let Some(content) = message.get("content") {
        match content {
            ToolJson::Null | ToolJson::String(_) => {}
            // content is a string or null. An array (multimodal) is a documented
            // non-goal: a clear 400 naming the message, never a serde panic.
            _ => {
                return Err(ApiError::invalid_request(format!(
                    "messages[{index}].content must be a string or null (multimodal content \
                     arrays are not supported)"
                )))
            }
        }
    }
    // A replayed assistant turn (#613). "No calls" is the absence of the field, not
    // an empty array — the same rule `tools: []` follows, and the shape the Qwen2.5
    // template's `not message.tool_calls` branch would otherwise see as one.
    if let Some(tool_calls) = message.get("tool_calls") {
        let ToolJson::Array(calls) = tool_calls else {
            return Err(ApiError::invalid_request(format!(
                "messages[{index}].tool_calls must be an array"
            )));
        };
        if calls.is_empty() {
            return Err(ApiError::invalid_request(format!(
                "messages[{index}].tool_calls must not be empty; omit the field to send none"
            )));
        }
        for (call_index, call) in calls.iter().enumerate() {
            validate_tool_call(index, call_index, call)?;
        }
    }
    Ok(())
}

/// One replayed tool call on an `assistant` message (#613).
///
/// The OpenAI shape is `{id, type: "function", function: {name, arguments}}`, and
/// `arguments` is a JSON **string** — the template echoes it through `tojson`, so
/// an object would render different bytes. Anything else is a `400` naming the
/// message and the entry, never a silently dropped call.
fn validate_tool_call(
    message: usize,
    index: usize,
    call: &crate::template::ToolJson,
) -> Result<(), ApiError> {
    use crate::template::ToolJson;
    let at = format!("messages[{message}].tool_calls[{index}]");
    if call.as_object().is_none() {
        return Err(ApiError::invalid_request(format!(
            "{at} must be an object with `id`, `type` and `function`"
        )));
    }
    match call.get("id").and_then(ToolJson::as_str) {
        Some(id) if !id.trim().is_empty() => {}
        _ => {
            return Err(ApiError::invalid_request(format!(
                "{at}.id must be a non-empty string"
            )))
        }
    }
    match call.get("type").and_then(ToolJson::as_str) {
        Some("function") => {}
        Some(other) => {
            return Err(ApiError::invalid_request(format!(
                "{at}.type is '{other}'; this server implements 'function' only"
            )))
        }
        None => {
            return Err(ApiError::invalid_request(format!(
                "{at} has no string `type` (expected \"function\")"
            )))
        }
    }
    let Some(function) = call.get("function") else {
        return Err(ApiError::invalid_request(format!(
            "{at} has no `function` object"
        )));
    };
    if function.as_object().is_none() {
        return Err(ApiError::invalid_request(format!(
            "{at}.function must be an object"
        )));
    }
    match function.get("name").and_then(ToolJson::as_str) {
        Some(name) if !name.trim().is_empty() => {}
        _ => {
            return Err(ApiError::invalid_request(format!(
                "{at}.function.name must be a non-empty string"
            )))
        }
    }
    match function.get("arguments") {
        Some(ToolJson::String(_)) => {}
        _ => {
            return Err(ApiError::invalid_request(format!(
                "{at}.function.arguments must be a string: OpenAI sends the call's \
                 arguments as JSON text and the template echoes them through `tojson`"
            )))
        }
    }
    Ok(())
}

/// One OpenAI tool definition: `{"type":"function","function":{"name":…}}`
/// (#597).
///
/// Only the structure the prompt depends on is validated. The `function` object
/// is passed through verbatim — an unknown key (OpenAI's `strict`, a vendor
/// extension) must survive into the prompt, so this is not a typed subset.
fn validate_tool(index: usize, tool: &crate::template::ToolJson) -> Result<(), ApiError> {
    use crate::template::ToolJson;
    if tool.as_object().is_none() {
        return Err(ApiError::invalid_request(format!(
            "tools[{index}] must be an object with `type` and `function`"
        )));
    }
    match tool.get("type").and_then(ToolJson::as_str) {
        Some("function") => {}
        Some(other) => {
            return Err(ApiError::invalid_request(format!(
                "tools[{index}].type is '{other}'; this server implements 'function' only"
            )))
        }
        None => {
            return Err(ApiError::invalid_request(format!(
                "tools[{index}] has no string `type` (expected \"function\")"
            )))
        }
    }
    let Some(function) = tool.get("function") else {
        return Err(ApiError::invalid_request(format!(
            "tools[{index}] has no `function` object"
        )));
    };
    if function.as_object().is_none() {
        return Err(ApiError::invalid_request(format!(
            "tools[{index}].function must be an object"
        )));
    }
    match function.get("name").and_then(ToolJson::as_str) {
        Some(name) if !name.trim().is_empty() => {}
        _ => {
            return Err(ApiError::invalid_request(format!(
                "tools[{index}].function.name must be a non-empty string"
            )))
        }
    }
    Ok(())
}

// === Response Types ===
#[derive(Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: &'static str,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Serialize)]
pub struct Choice {
    pub index: u32,
    pub message: ResponseMessage,
    pub finish_reason: String,
}

#[derive(Serialize)]
pub struct ResponseMessage {
    pub role: &'static str,
    pub content: String,
    /// Present only when the completion carried a tool-call block (#614).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ResponseToolCall>>,
}

/// One OpenAI tool call in a response.
#[derive(Serialize)]
pub struct ResponseToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: ResponseFunction,
}

#[derive(Serialize)]
pub struct ResponseFunction {
    pub name: String,
    pub arguments: String,
}

impl ResponseToolCall {
    /// The response shape of a parsed call. The model does not write an id, so
    /// one is synthesised here — a client needs it for the follow-up `tool`
    /// message's `tool_call_id` (llama.cpp leaves it out; OpenAI requires it).
    pub fn new(call: &crate::server::toolcall::ToolCall) -> Self {
        Self {
            id: format!("call_{}", uuid::Uuid::new_v4().simple()),
            kind: "function",
            function: ResponseFunction {
                name: call.name.clone(),
                arguments: call.arguments.clone(),
            },
        }
    }
}

#[derive(Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

/// Build a non-streaming chat completion response.
pub fn build_response(
    id: &str,
    model: &str,
    created: i64,
    text: String,
    finish_reason: &str,
    prompt_tokens: usize,
    completion_tokens: usize,
    tool_calls: Option<Vec<ResponseToolCall>>,
) -> ChatCompletionResponse {
    ChatCompletionResponse {
        id: id.to_string(),
        object: "chat.completion",
        created,
        model: model.to_string(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant",
                content: text,
                tool_calls,
            },
            finish_reason: finish_reason.to_string(),
        }],
        usage: Usage {
            prompt_tokens: prompt_tokens as u32,
            completion_tokens: completion_tokens as u32,
            total_tokens: (prompt_tokens + completion_tokens) as u32,
        },
    }
}

// === Streaming chunk builders ===
// Each returns the JSON `data:` payload for one SSE event (the caller prefixes
// "data: " and appends "\n\n", or the chunk is used verbatim in a Sse<Event>).

/// First chunk: role only, null content.
pub fn chunk_role(id: &str, model: &str, created: i64) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": null}, "finish_reason": null}],
    })
    .to_string()
}

/// Content chunk.
pub fn chunk_content(id: &str, model: &str, created: i64, text: &str) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": null}],
    })
    .to_string()
}

/// First chunk of a streamed tool call: index, id, type and the function name;
/// the arguments follow in [`chunk_tool_call_arguments`] fragments.
pub fn chunk_tool_call_start(
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    call_id: &str,
    name: &str,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{
                "index": index,
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": ""},
            }]},
            "finish_reason": null,
        }],
    })
    .to_string()
}

/// One arguments fragment of a streamed tool call.
pub fn chunk_tool_call_arguments(
    id: &str,
    model: &str,
    created: i64,
    index: usize,
    fragment: &str,
) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {"tool_calls": [{"index": index, "function": {"arguments": fragment}}]},
            "finish_reason": null,
        }],
    })
    .to_string()
}

/// The SSE payloads for one parser event: a content chunk, or a call's start and
/// its arguments fragment. The streaming handler and its differential test both
/// go through this one mapping.
pub fn chunks_for_tool_event(
    id: &str,
    model: &str,
    created: i64,
    call_index: &mut usize,
    event: &crate::server::toolcall::Event,
) -> Vec<String> {
    match event {
        crate::server::toolcall::Event::Content(text) => {
            vec![chunk_content(id, model, created, text)]
        }
        crate::server::toolcall::Event::Call(call) => {
            let call = ResponseToolCall::new(call);
            let mut out = vec![chunk_tool_call_start(
                id,
                model,
                created,
                *call_index,
                &call.id,
                &call.function.name,
            )];
            if !call.function.arguments.is_empty() {
                out.push(chunk_tool_call_arguments(
                    id,
                    model,
                    created,
                    *call_index,
                    &call.function.arguments,
                ));
            }
            *call_index += 1;
            out
        }
    }
}

/// Final chunk: empty delta + finish_reason.
pub fn chunk_finish(id: &str, model: &str, created: i64, reason: &str) -> String {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": reason}],
    })
    .to_string()
}

// === Error Types ===

#[derive(Debug, Clone)]
pub struct ApiError {
    pub status: u16,
    pub message: String,
    pub error_type: &'static str,
}

impl ApiError {
    pub fn invalid_request(msg: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: msg.into(),
            error_type: "invalid_request_error",
        }
    }
    pub fn exceed_context(msg: impl Into<String>) -> Self {
        Self {
            status: 400,
            message: msg.into(),
            error_type: "exceed_context_size_error",
        }
    }
    pub fn unavailable(msg: impl Into<String>) -> Self {
        Self {
            status: 503,
            message: msg.into(),
            error_type: "unavailable_error",
        }
    }
    pub fn server(msg: impl Into<String>) -> Self {
        Self {
            status: 500,
            message: msg.into(),
            error_type: "server_error",
        }
    }
    pub fn json(&self) -> String {
        serde_json::json!({
            "error": {
                "message": self.message,
                "type": self.error_type,
                "code": self.status,
            }
        })
        .to_string()
    }
}

#[cfg(test)]
mod tests;
