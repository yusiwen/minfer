//! Tool-call extraction (#614): a `<tool_call>` block in the model's completion
//! becomes OpenAI `tool_calls`, never ordinary content.
//!
//! The parser is the render path's mirror image. ADR-0062 carries a request
//! message's `tool_calls` into the prompt; this module reads them back out of the
//! completion, so a client can replay the round trip. Two properties shape it:
//!
//! * **Incremental.** The generation loop emits text in arbitrary UTF-8 chunks, so
//!   the opening tag can be split across chunks. A suffix that could still become
//!   `<tool_call>` is held back, and the block body is parsed only once the JSON
//!   value is complete — the closing tag is *never* found by `str::find`, because
//!   the arguments may legitimately contain that literal inside a string.
//! * **The arguments are the model's bytes.** A JSON object is carried as its own
//!   source slice (whitespace and key order included); a JSON *string* is decoded,
//!   because OpenAI sends the call's arguments as text and some templates render
//!   them through `tojson`. This is the rule transformers' parser and OpenAI's
//!   shape both support.
//!
//! A block that opens and never closes is an error — a named refusal upstream —
//! never a half-call in `content`.

use serde::Deserialize;
use serde_json::value::RawValue;

/// The opening tag of the Qwen2/Qwen3 (and Hermes) tool-call format.
pub const OPEN_TAG: &str = "<tool_call>";
/// Its closing tag.
pub const CLOSE_TAG: &str = "</tool_call>";

/// One call parsed out of a completion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    /// The function name the model wrote.
    pub name: String,
    /// The arguments as JSON text: the block's own bytes for an object/array, the
    /// decoded contents for a JSON string.
    pub arguments: String,
}

/// What the parser emits, in completion order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Event {
    /// Text outside a call block.
    Content(String),
    /// One complete call block.
    Call(ToolCall),
}

/// A completion that cannot be turned into a message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    detail: String,
}

impl ParseError {
    fn new(detail: impl Into<String>) -> Self {
        Self {
            detail: detail.into(),
        }
    }

    /// The client-facing wording.
    pub fn message(&self) -> String {
        format!(
            "the model's output ended inside a tool-call block or carried a malformed one: {}",
            self.detail
        )
    }
}

/// The incremental parser. Feed it the completion's chunks, then call
/// [`Self::finish`]; the events are returned in order.
#[derive(Debug, Default)]
pub struct Parser {
    /// Unconsumed text: a content tail that could still be a tag prefix or a
    /// trailing whitespace run, or a block body that is not complete yet.
    buf: String,
    /// Inside a `<tool_call>` block: the body (JSON + closing tag) is pending.
    in_block: bool,
    /// Just closed a block: leading whitespace of the next content segment is a
    /// separator, not content.
    after_call: bool,
    /// A malformed block seen mid-stream; surfaced by [`Self::finish`].
    error: Option<ParseError>,
}

impl Parser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one chunk; returns the events that are final now.
    pub fn feed(&mut self, text: &str) -> Vec<Event> {
        if self.error.is_some() {
            return Vec::new();
        }
        self.buf.push_str(text);
        let mut out = Vec::new();
        if let Err(e) = self.pump(&mut out, false) {
            self.error = Some(e);
        }
        out
    }

    /// Flush: the completion is over. An open block is an error.
    pub fn finish(mut self) -> Result<Vec<Event>, ParseError> {
        if let Some(e) = self.error.take() {
            return Err(e);
        }
        let mut out = Vec::new();
        self.pump(&mut out, true)?;
        Ok(out)
    }

    fn pump(&mut self, out: &mut Vec<Event>, eos: bool) -> Result<(), ParseError> {
        loop {
            if self.in_block {
                if !self.block_step(out, eos)? {
                    return Ok(());
                }
                continue;
            }
            if !self.content_step(out, eos) {
                return Ok(());
            }
        }
    }

    /// Consume as much content as is final. Returns false when the parser is
    /// waiting for more bytes (or at EOS, when nothing is left).
    fn content_step(&mut self, out: &mut Vec<Event>, eos: bool) -> bool {
        if let Some(at) = self.buf.find(OPEN_TAG) {
            let mut before = &self.buf[..at];
            // Whitespace between two calls separates them; it is not content.
            if self.after_call {
                before = before.trim_start();
            }
            // The whitespace a template writes before the block is formatting,
            // not content (llama.cpp's and transformers' parsers agree here).
            let before = before.trim_end();
            if !before.is_empty() {
                out.push(Event::Content(before.to_string()));
            }
            self.buf.drain(..at + OPEN_TAG.len());
            self.in_block = true;
            self.after_call = false;
            return true;
        }

        // No tag in the buffer: hold a suffix that could still become one, plus a
        // trailing whitespace run (it may turn out to sit before a block).
        let hold = if eos {
            0
        } else {
            // Hold a suffix that could still become the opening tag, plus the
            // whitespace before it: that run is a separator if the tag completes,
            // and only content if it does not.
            let tag_prefix = longest_suffix_prefix(&self.buf, OPEN_TAG);
            let before = &self.buf[..self.buf.len() - tag_prefix];
            let whitespace = trailing_whitespace_len(before);
            let hold = if self.after_call && self.buf.chars().all(char::is_whitespace) {
                // After a call, a whitespace-only run is a separator until text
                // proves otherwise.
                self.buf.len()
            } else {
                tag_prefix + whitespace
            };
            hold
        };
        let cut = self.buf.len() - hold;
        if cut > 0 {
            let mut text = &self.buf[..cut];
            if self.after_call {
                text = text.trim_start();
            }
            if !text.is_empty() {
                out.push(Event::Content(text.to_string()));
            }
            self.buf.drain(..cut);
            self.after_call = false;
        }
        false
    }

    /// Consume one block body. Returns false when more bytes are needed.
    fn block_step(&mut self, out: &mut Vec<Event>, eos: bool) -> Result<bool, ParseError> {
        let start = self.buf.len() - self.buf.trim_start().len();
        if start == self.buf.len() {
            // Only whitespace (or nothing) after the opening tag.
            if eos {
                return Err(ParseError::new("the block opened but holds no call"));
            }
            return Ok(false);
        }
        match raw_value_len(&self.buf[start..]) {
            Ok(len) => {
                let after = start + len;
                let rest = &self.buf[after..];
                let ws = rest.len() - rest.trim_start().len();
                let after_ws = &rest[ws..];
                if after_ws.starts_with(CLOSE_TAG) {
                    let json = &self.buf[start..after];
                    let call = parse_block(json)?;
                    out.push(Event::Call(call));
                    let consumed = after + ws + CLOSE_TAG.len();
                    self.buf.drain(..consumed);
                    self.in_block = false;
                    self.after_call = true;
                    return Ok(true);
                }
                if after_ws.is_empty() || CLOSE_TAG.starts_with(after_ws) {
                    if eos {
                        // The JSON value is complete and nothing but (a prefix of)
                        // the closing tag follows: the model stopped at EOS after a
                        // usable call. transformers closes the region implicitly
                        // here too; only *incomplete arguments* are truncated.
                        let json = &self.buf[start..after];
                        let call = parse_block(json)?;
                        out.push(Event::Call(call));
                        self.buf.clear();
                        self.in_block = false;
                        self.after_call = true;
                        return Ok(true);
                    }
                    return Ok(false);
                }
                return Err(ParseError::new(format!(
                    "expected `</tool_call>` after the call's JSON, found {:?}",
                    truncate_at(after_ws, 40)
                )));
            }
            Err(RawStatus::Eof) => {
                if eos {
                    return Err(ParseError::new(
                        "the completion ended inside the call's JSON",
                    ));
                }
                Ok(false)
            }
            Err(RawStatus::Bad(detail)) => Err(ParseError::new(format!(
                "the call's JSON does not parse: {detail} (near {:?})",
                truncate_at(&self.buf[start..], 160)
            ))),
        }
    }
}

/// The GBNF for `tool_choice: "required"` (one of `names`) or a forced function.
///
/// It forces the model's own tool-call syntax: the `<tool_call>` wrapper, a
/// `{"name": …, "arguments": …}` object with the name fixed or enumerated, and a
/// JSON object for the arguments (reusing the schema compiler's generic `j-object`
/// rules). The result is the block this module parses, so extraction and the
/// constraint agree by construction.
///
/// Only the `<tool_call>` format can be constrained this way; a template that
/// emits another family's syntax is refused by the caller instead.
pub fn forced_call_grammar(names: &[&str], forced: Option<&str>) -> String {
    let mut grammar = String::new();
    for (name, rhs) in crate::grammar::GENERIC_RULES {
        grammar.push_str(name);
        grammar.push_str(" ::= ");
        grammar.push_str(rhs);
        grammar.push('\n');
    }
    let name_rule = match forced {
        Some(name) => gbnf_literal(name),
        None => names
            .iter()
            .map(|name| gbnf_literal(name))
            .collect::<Vec<_>>()
            .join(" | "),
    };
    grammar.push_str("root ::= ");
    grammar.push_str(&gbnf_literal("<tool_call>\n{\"name\": \""));
    grammar.push(' ');
    grammar.push_str(&name_rule);
    grammar.push(' ');
    grammar.push_str(&gbnf_literal("\", \"arguments\": "));
    grammar.push_str("j-object");
    grammar.push(' ');
    grammar.push_str(&gbnf_literal("}\n</tool_call>"));
    grammar.push('\n');
    grammar
}

/// One string as a GBNF literal: quotes, backslashes and control characters are
/// escaped the way the grammar parser expects them.
fn gbnf_literal(text: &str) -> String {
    let mut out = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

/// Parse a whole completion in one call (the non-streaming path).
pub fn parse_completion(text: &str) -> Result<(String, Vec<ToolCall>), ParseError> {
    let mut parser = Parser::new();
    let mut events = parser.feed(text);
    events.extend(parser.finish()?);
    let mut content = String::new();
    let mut calls = Vec::new();
    for event in events {
        match event {
            Event::Content(text) => content.push_str(&text),
            Event::Call(call) => calls.push(call),
        }
    }
    Ok((content, calls))
}

#[derive(Deserialize)]
struct Block<'a> {
    name: String,
    #[serde(borrow)]
    arguments: Option<&'a RawValue>,
}

fn parse_block(json: &str) -> Result<ToolCall, ParseError> {
    let block: Block = serde_json::from_str(json)
        .map_err(|e| ParseError::new(format!("the call object does not parse: {e}")))?;
    if block.name.trim().is_empty() {
        return Err(ParseError::new("the call's `name` is empty"));
    }
    let raw = block
        .arguments
        .ok_or_else(|| ParseError::new("the call has no `arguments`"))?;
    let text = raw.get();
    // A string argument is the call's text (OpenAI's shape); anything else is
    // carried as the bytes the model wrote.
    let arguments = if text.starts_with('"') {
        serde_json::from_str::<String>(text).map_err(|e| {
            ParseError::new(format!("the call's arguments string does not decode: {e}"))
        })?
    } else {
        text.to_string()
    };
    Ok(ToolCall {
        name: block.name,
        arguments,
    })
}

enum RawStatus {
    Eof,
    Bad(String),
}

/// The length of the JSON value at the start of `s`, or why it is not complete.
fn raw_value_len(s: &str) -> Result<usize, RawStatus> {
    let de = serde_json::Deserializer::from_str(s);
    let mut values = de.into_iter::<&RawValue>();
    match values.next() {
        Some(Ok(_)) => Ok(values.byte_offset()),
        Some(Err(e)) if e.is_eof() => Err(RawStatus::Eof),
        Some(Err(e)) => Err(RawStatus::Bad(e.to_string())),
        None => Err(RawStatus::Eof),
    }
}

/// The length of the longest suffix of `text` that is a proper prefix of `tag`.
fn longest_suffix_prefix(text: &str, tag: &str) -> usize {
    let bytes = text.as_bytes();
    let tag = tag.as_bytes();
    let max = tag.len().saturating_sub(1).min(bytes.len());
    (1..=max)
        .rev()
        .find(|&k| bytes.ends_with(&tag[..k]))
        .unwrap_or(0)
}

fn trailing_whitespace_len(text: &str) -> usize {
    text.len() - text.trim_end().len()
}

fn truncate_at(text: &str, max: usize) -> &str {
    match text.char_indices().nth(max) {
        Some((i, _)) => &text[..i],
        None => text,
    }
}

#[cfg(test)]
mod tests;
