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
    pub messages: Vec<ChatMessage>,
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
    /// DRY restart sequences as token-id sequences (llama.cpp's string form
    /// needs a tokenizer port — see the F3 record and #48).
    pub dry_sequence_breakers: Option<Vec<Vec<u32>>>,
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
pub struct ChatMessage {
    pub role: String,
    pub content: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
pub enum StopCondition {
    String(String),
    Array(Vec<String>),
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
        for m in &req.messages {
            if !VALID_ROLES.contains(&m.role.as_str()) {
                return Err(ApiError::invalid_request(format!(
                    "unknown message role '{}'",
                    m.role
                )));
            }
            // content: Option<String> — an array (multimodal) fails serde with
            // a clear error; we surface it as invalid_request_error.
        }
        Ok(req)
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
            dry_breakers: self.dry_sequence_breakers.clone().unwrap_or_default(),
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

#[derive(Debug)]
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
mod tests {
    use super::*;

    #[test]
    fn parse_valid_request() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_k":20,"stop":["\n\n","User:"]}"#;
        let req = ChatCompletionRequest::parse(body).expect("parse");
        assert_eq!(req.messages.len(), 1);
        assert_eq!(req.temperature, Some(0.7));
        assert_eq!(req.top_k, Some(20));
        match &req.stop {
            Some(StopCondition::Array(v)) => assert_eq!(v.len(), 2),
            _ => panic!("stop should be array"),
        }
    }

    #[test]
    fn parse_stop_sequences_alias() {
        // Anthropic alias "stop_sequences" maps onto `stop`
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"stop_sequences":["a","b"]}"#;
        let req = ChatCompletionRequest::parse(body).expect("parse");
        match &req.stop {
            Some(StopCondition::Array(v)) => assert_eq!(v, &["a".to_string(), "b".to_string()]),
            _ => panic!("stop_sequences alias must populate stop"),
        }
    }

    #[test]
    fn parse_rejects_malformed_json() {
        let err = ChatCompletionRequest::parse(br#"{"messages": ["#).unwrap_err();
        assert_eq!(err.status, 400);
        assert_eq!(err.error_type, "invalid_request_error");
    }

    #[test]
    fn parse_rejects_empty_messages() {
        let err = ChatCompletionRequest::parse(br#"{"messages":[]}"#).unwrap_err();
        assert_eq!(err.status, 400);
    }

    #[test]
    fn parse_rejects_unknown_role() {
        let err =
            ChatCompletionRequest::parse(br#"{"messages":[{"role":"pirate","content":"yo"}]}"#)
                .unwrap_err();
        assert_eq!(err.status, 400);
        assert!(err.message.contains("pirate"));
    }

    #[test]
    fn parse_rejects_array_content() {
        // multimodal content arrays are a documented non-goal -> 400, not a serde panic
        let err = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":[{"type":"text","text":"hi"}]}]}"#,
        )
        .unwrap_err();
        assert_eq!(err.status, 400);
    }

    #[test]
    fn resolve_applies_defaults() {
        let req = ChatCompletionRequest::parse(br#"{"messages":[{"role":"user","content":"hi"}]}"#)
            .unwrap();
        let p = req.resolve(1234).unwrap();
        assert_eq!(p.temp, DEFAULT_TEMP);
        assert_eq!(p.top_k, DEFAULT_TOP_K);
        assert_eq!(p.top_p, DEFAULT_TOP_P);
        assert_eq!(p.repeat_penalty, DEFAULT_REPEAT_PENALTY);
        assert_eq!(p.frequency_penalty, 0.0);
        assert_eq!(p.presence_penalty, 0.0);
        assert_eq!(p.max_tokens, MAX_TOKENS_UNLIMITED);
        assert_eq!(
            p.seed, 1234,
            "request seed missing -> caller-provided random default"
        );
        assert!(p.stop_strings.is_empty());
        // F3: every new knob's default is a no-op.
        assert_eq!(p.min_p, 0.0);
        assert_eq!(p.typical_p, 1.0);
        assert_eq!(p.xtc_probability, 0.0);
        assert_eq!(p.dry_multiplier, 0.0);
        assert_eq!(p.mirostat, crate::sampler::MirostatMode::Off);
        assert!(p.logit_bias.is_empty());
    }

    #[test]
    fn resolve_honors_explicit_values() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.1,"max_tokens":50,"seed":7,"stop":"EOF","frequency_penalty":0.5,"presence_penalty":0.25}"#;
        let req = ChatCompletionRequest::parse(body).unwrap();
        let p = req.resolve(1234).unwrap();
        assert_eq!(p.temp, 0.1);
        assert_eq!(p.max_tokens, 50);
        assert_eq!(p.seed, 7);
        assert_eq!(p.stop_strings, vec!["EOF".to_string()]);
        assert_eq!(p.frequency_penalty, 0.5);
        assert_eq!(p.presence_penalty, 0.25);
    }

    #[test]
    fn error_json_format_matches_llama_cpp() {
        let e = ApiError::exceed_context("prompt too long");
        let v: serde_json::Value = serde_json::from_str(&e.json()).unwrap();
        assert_eq!(v["error"]["code"], 400);
        assert_eq!(v["error"]["type"], "exceed_context_size_error");
        assert_eq!(v["error"]["message"], "prompt too long");
    }

    #[test]
    fn response_json_shape() {
        let r = build_response("chatcmpl-x", "qwen", 1, "hi".into(), "stop", 3, 1);
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["object"], "chat.completion");
        assert_eq!(v["choices"][0]["message"]["role"], "assistant");
        assert_eq!(v["choices"][0]["finish_reason"], "stop");
        assert_eq!(v["usage"]["total_tokens"], 4);
    }

    #[test]
    fn chunk_builders() {
        let r = chunk_role("id", "m", 1);
        let v: serde_json::Value = serde_json::from_str(&r).unwrap();
        assert_eq!(v["object"], "chat.completion.chunk");
        assert_eq!(v["choices"][0]["delta"]["role"], "assistant");
        assert!(v["choices"][0]["delta"]["content"].is_null());

        let c = chunk_content("id", "m", 1, "text");
        let v: serde_json::Value = serde_json::from_str(&c).unwrap();
        assert_eq!(v["choices"][0]["delta"]["content"], "text");

        let f = chunk_finish("id", "m", 1, "length");
        let v: serde_json::Value = serde_json::from_str(&f).unwrap();
        assert_eq!(v["choices"][0]["finish_reason"], "length");
        assert!(v["choices"][0]["delta"].as_object().unwrap().is_empty());
    }

    #[test]
    fn resolve_rejects_a_bad_mirostat_mode_or_logit_bias_key() {
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"mirostat":3}"#,
        )
        .unwrap();
        let err = req.resolve(0).unwrap_err();
        assert_eq!(err.status, 400);
        assert!(err.message.contains("mirostat"), "{}", err.message);

        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"logit_bias":{"not-an-id":1.0}}"#,
        )
        .unwrap();
        let err = req.resolve(0).unwrap_err();
        assert_eq!(err.status, 400);
        assert!(err.message.contains("logit_bias"), "{}", err.message);
    }

    #[test]
    fn resolve_and_validate_the_f3_surface() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"min_p":0.05,"typical_p":0.9,
            "xtc_probability":0.5,"xtc_threshold":0.1,"dry_multiplier":0.8,"dry_base":1.75,
            "dry_allowed_length":3,"dry_penalty_last_n":32,"dry_sequence_breakers":[[198],[13,2]],
            "mirostat":2,"mirostat_tau":4.0,"mirostat_eta":0.2,"mirostat_m":50,
            "logit_bias":{"198":-2.0,"42":1.5}}"#;
        let req = ChatCompletionRequest::parse(body).unwrap();
        let p = req.resolve(0).unwrap();
        assert_eq!(p.min_p, 0.05);
        assert_eq!(p.typical_p, 0.9);
        assert_eq!(p.dry_breakers, vec![vec![198u32], vec![13, 2]]);
        assert_eq!(p.mirostat, crate::sampler::MirostatMode::V2);
        // The map is resolved in a deterministic (sorted) order.
        assert_eq!(p.logit_bias, vec![(42u32, 1.5), (198u32, -2.0)]);
        assert!(p.validate(1000).is_ok());

        // Out-of-vocab id and an out-of-range value are 400s.
        let err = p.validate(100).unwrap_err();
        assert_eq!(err.status, 400);
        assert!(err.message.contains("outside the vocabulary"));
        let bad = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"min_p":2.0}"#,
        )
        .unwrap()
        .resolve(0)
        .unwrap();
        assert!(bad.validate(1000).is_err());
    }

    // === F2 (#47): the structured-output surface ============================

    #[test]
    fn resolve_maps_response_format_and_the_grammar_extension() {
        // Absent / text -> unconstrained.
        let req = ChatCompletionRequest::parse(br#"{"messages":[{"role":"user","content":"hi"}]}"#)
            .unwrap();
        assert!(req.resolve(0).unwrap().grammar_source.is_none());
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}"#,
        )
        .unwrap();
        assert!(req.resolve(0).unwrap().grammar_source.is_none());

        // json_object -> any JSON value.
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
        )
        .unwrap();
        match req.resolve(0).unwrap().grammar_source {
            Some(crate::grammar::GrammarSource::AnyJson) => {}
            other => panic!("expected AnyJson, got {other:?}"),
        }

        // json_schema -> the schema, name and strict are accepted and ignored.
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "response_format":{"type":"json_schema","json_schema":
                   {"name":"person","strict":true,"schema":{"type":"object"}}}}"#,
        )
        .unwrap();
        match req.resolve(0).unwrap().grammar_source {
            Some(crate::grammar::GrammarSource::Json(v)) => {
                assert_eq!(v["type"], "object");
            }
            other => panic!("expected Json, got {other:?}"),
        }

        // The GBNF extension field.
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"grammar":"root ::= \"a\""}"#,
        )
        .unwrap();
        match req.resolve(0).unwrap().grammar_source {
            Some(crate::grammar::GrammarSource::Gbnf(g)) => assert_eq!(g, "root ::= \"a\""),
            other => panic!("expected Gbnf, got {other:?}"),
        }
    }

    #[test]
    fn resolve_refuses_two_grammars_or_an_unknown_response_type() {
        // `grammar` + a non-text `response_format`: a 400, never a precedence rule.
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"grammar":"root ::= .",
                 "response_format":{"type":"json_object"}}"#,
        )
        .unwrap();
        let err = req.resolve(0).unwrap_err();
        assert_eq!(err.status, 400);
        assert!(
            err.message.contains("mutually exclusive"),
            "{}",
            err.message
        );

        // An unknown response_format type is refused by serde -> 400.
        let err = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"xml"}}"#,
        )
        .unwrap_err();
        assert_eq!(err.status, 400);

        // A json_schema entry without a schema is a 400.
        let err = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "response_format":{"type":"json_schema","json_schema":{"name":"x"}}}"#,
        )
        .unwrap_err();
        assert_eq!(err.status, 400);
    }

    #[test]
    fn compile_grammar_turns_an_unsupported_schema_into_a_400() {
        let tok = crate::tokenizer::Tokenizer::empty();
        let special = crate::models::SpecialTokens {
            eos: 0,
            im_end: None,
        };

        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "response_format":{"type":"json_schema","json_schema":
                   {"name":"x","schema":{"type":"string","pattern":"^a"}}}}"#,
        )
        .unwrap();
        let mut p = req.resolve(0).unwrap();
        assert!(
            p.grammar.is_none(),
            "uncompiled until the vocabulary is known"
        );
        let err = p.compile_grammar(&tok, &special).unwrap_err();
        assert_eq!(err.status, 400);
        assert!(err.message.contains("pattern"), "{}", err.message);
        assert!(
            p.grammar.is_none(),
            "a refused schema leaves no grammar behind"
        );

        // A supported schema compiles and lands in the sampler config.
        let req = ChatCompletionRequest::parse(
            br#"{"messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#,
        )
        .unwrap();
        let mut p = req.resolve(0).unwrap();
        p.compile_grammar(&tok, &special).expect("compiles");
        assert!(p.sampler_config().grammar.is_some());
    }
}
