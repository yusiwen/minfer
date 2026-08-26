//! OpenAI-compatible request/response types, sampling-parameter resolution and
//! validation (OPENAI-CHAT-API-PLAN.md §Data Structures, §Error Handling).
//!
//! Defaults deliberately match llama.cpp / the minfer CLI (temperature 0.8,
//! top_p 0.95, top_k 40, repeat_penalty 1.1, max_tokens = no limit) rather
//! than the OpenAI spec — see the plan's "Request/Response Format" section.

use serde::{Deserialize, Serialize};

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
pub struct SamplingParams {
    pub temp: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repeat_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: u64,
    pub stop_strings: Vec<String>,
    pub max_tokens: i64,
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
    pub fn resolve(&self, rng_seed: u64) -> SamplingParams {
        let stop_strings = match &self.stop {
            None => Vec::new(),
            Some(StopCondition::String(s)) => vec![s.clone()],
            Some(StopCondition::Array(v)) => v.clone(),
        };
        SamplingParams {
            temp: self.temperature.unwrap_or(DEFAULT_TEMP),
            top_k: self.top_k.map(|k| k as usize).unwrap_or(DEFAULT_TOP_K),
            top_p: self.top_p.unwrap_or(DEFAULT_TOP_P),
            repeat_penalty: self.repeat_penalty.unwrap_or(DEFAULT_REPEAT_PENALTY),
            frequency_penalty: self.frequency_penalty.unwrap_or(0.0),
            presence_penalty: self.presence_penalty.unwrap_or(0.0),
            seed: self.seed.unwrap_or(rng_seed),
            stop_strings,
            max_tokens: self.max_tokens.unwrap_or(MAX_TOKENS_UNLIMITED),
        }
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
        let p = req.resolve(1234);
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
    }

    #[test]
    fn resolve_honors_explicit_values() {
        let body = br#"{"messages":[{"role":"user","content":"hi"}],"temperature":0.1,"max_tokens":50,"seed":7,"stop":"EOF","frequency_penalty":0.5,"presence_penalty":0.25}"#;
        let req = ChatCompletionRequest::parse(body).unwrap();
        let p = req.resolve(1234);
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
}
