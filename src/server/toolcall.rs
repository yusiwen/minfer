//! Tool-call extraction (#614, #699): a call block in the model's completion
//! becomes OpenAI `tool_calls`, never ordinary content.
//!
//! The parser is the render path's mirror image. ADR-0062 carries a request
//! message's `tool_calls` into the prompt; this module reads them back out of the
//! completion, so a client can replay the round trip. The per-family syntax lives
//! in `format` (ADR-0066): one row says what opens and closes a call, which object
//! keys carry the name and the arguments, and what a `tool_choice` grammar wraps
//! around them. Two properties shape the parser:
//!
//! * **Incremental.** The generation loop emits text in arbitrary UTF-8 chunks, so
//!   the opening anchor can be split across chunks. A suffix that could still
//!   become it is held back, and the block body is parsed only once the JSON value
//!   is complete — the closing anchor is *never* found by `str::find`, because the
//!   arguments may legitimately contain that literal inside a string.
//! * **The arguments are the model's bytes.** A JSON object is carried as its own
//!   source slice (whitespace and key order included); a JSON *string* is decoded,
//!   because OpenAI sends the call's arguments as text and some templates render
//!   them through `tojson`. This is the rule transformers' parser and OpenAI's
//!   shape both support.
//!
//! A block that opens and never closes is an error — a named refusal upstream —
//! never a half-call in `content`.

pub(crate) mod format;

use std::collections::HashMap;

use serde_json::value::RawValue;

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
#[derive(Debug)]
pub struct Parser {
    /// The format row this completion is parsed against (ADR-0066).
    format: &'static format::ToolFormat,
    /// Unconsumed text: a content tail that could still be a tag prefix or a
    /// trailing whitespace run, or a block body that is not complete yet.
    buf: String,
    /// Inside a call block: the body (JSON + closing anchor) is pending.
    in_block: bool,
    /// Just closed a block: leading whitespace of the next content segment is a
    /// separator, not content.
    after_call: bool,
    /// A malformed block seen mid-stream; surfaced by [`Self::finish`].
    error: Option<ParseError>,
}

impl Parser {
    pub fn new(format: &'static format::ToolFormat) -> Self {
        Self {
            format,
            buf: String::new(),
            in_block: false,
            after_call: false,
            error: None,
        }
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
        let open = self.format.open;
        if let Some(at) = self.buf.find(open) {
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
            self.buf.drain(..at + open.len());
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
            let tag_prefix = longest_suffix_prefix(&self.buf, open);
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
        if matches!(self.format.shape, format::BlockShape::XmlFunction { .. }) {
            return self.xml_step(out, eos);
        }
        if matches!(self.format.shape, format::BlockShape::NameThenArgs { .. }) {
            return self.name_args_step(out, eos);
        }
        let len = match raw_value_len(&self.buf[start..]) {
            Ok(len) => len,
            Err(RawStatus::Eof) => {
                if eos {
                    return Err(ParseError::new(
                        "the completion ended inside the call's JSON",
                    ));
                }
                return Ok(false);
            }
            Err(RawStatus::Bad(detail)) => {
                return Err(ParseError::new(format!(
                    "the call's JSON does not parse: {detail} (near {:?})",
                    truncate_at(&self.buf[start..], 160)
                )))
            }
        };
        let after = start + len;
        match self.format.shape {
            format::BlockShape::ObjectThenClose(close) => {
                let rest = &self.buf[after..];
                let ws = rest.len() - rest.trim_start().len();
                let after_ws = &rest[ws..];
                if after_ws.starts_with(close) {
                    let json = &self.buf[start..after];
                    let call = parse_block(json, self.format)?;
                    out.push(Event::Call(call));
                    let consumed = after + ws + close.len();
                    self.buf.drain(..consumed);
                    self.in_block = false;
                    self.after_call = true;
                    return Ok(true);
                }
                if after_ws.is_empty() || close.starts_with(after_ws) {
                    if eos {
                        // The JSON value is complete and nothing but (a prefix of)
                        // the closing anchor follows: the model stopped at EOS after
                        // a usable call. transformers closes the region implicitly
                        // here too; only *incomplete arguments* are truncated.
                        let json = &self.buf[start..after];
                        let call = parse_block(json, self.format)?;
                        out.push(Event::Call(call));
                        self.buf.clear();
                        self.in_block = false;
                        self.after_call = true;
                        return Ok(true);
                    }
                    return Ok(false);
                }
                Err(ParseError::new(format!(
                    "expected `{close}` after the call's JSON, found {:?}",
                    truncate_at(after_ws, 40)
                )))
            }
            format::BlockShape::ArrayAtEos => {
                let json = &self.buf[start..after];
                let calls = parse_call_array(json, self.format)?;
                out.extend(calls.into_iter().map(Event::Call));
                self.buf.drain(..after);
                self.in_block = false;
                self.after_call = true;
                Ok(true)
            }
            format::BlockShape::ObjectUntilNextOpen { separator } => {
                let json = &self.buf[start..after];
                let call = parse_block(json, self.format)?;
                let rest = &self.buf[after..];
                let ws = rest.len() - rest.trim_start().len();
                let mut at = after + ws;
                if let Some(after_sep) = self.buf[at..].strip_prefix(separator) {
                    let ws2 = after_sep.len() - after_sep.trim_start().len();
                    at += separator.len() + ws2;
                }
                let next = &self.buf[at..];
                let open = self.format.open;
                if next.starts_with(open) {
                    out.push(Event::Call(call));
                    self.buf.drain(..at + open.len());
                    // The next object starts now: stay in the block.
                    return Ok(true);
                }
                if next.is_empty() || open.starts_with(next) {
                    if eos {
                        // A complete call with nothing but a separator or a partial
                        // next open after it: the model stopped at EOS.
                        out.push(Event::Call(call));
                        self.buf.clear();
                        self.in_block = false;
                        self.after_call = true;
                        return Ok(true);
                    }
                    return Ok(false);
                }
                Err(ParseError::new(format!(
                    "expected `{separator}` or `{open}` after the call's JSON, found {:?}",
                    truncate_at(next, 40)
                )))
            }
            format::BlockShape::XmlFunction { .. } => {
                unreachable!("the XML shape is dispatched before the JSON body parse")
            }
            format::BlockShape::NameThenArgs { .. } => {
                unreachable!("the name-in-anchor shape is dispatched before the JSON body parse")
            }
        }
    }

    /// Consume one Qwen3.5 XML block (#708).
    ///
    /// The body is `<function=NAME>` followed by repeated
    /// `<parameter=KEY>VALUE</parameter>` entries, then `</function>` and
    /// `</tool_call>`. Values are text and are serialized back to JSON text for
    /// OpenAI's `arguments`; a complete function with no block close at EOS is a
    /// usable call (the same implicit-close rule the JSON rows use), while a
    /// truncated parameter is a refusal.
    fn xml_step(&mut self, out: &mut Vec<Event>, eos: bool) -> Result<bool, ParseError> {
        let format::BlockShape::XmlFunction {
            function_close,
            block_close,
        } = self.format.shape
        else {
            unreachable!("xml_step is called only for the XML shape")
        };
        let start = self.buf.len() - self.buf.trim_start().len();
        let body = &self.buf[start..];
        let Some(after_open) = body.strip_prefix("<function=") else {
            if eos {
                return Err(ParseError::new("the block opened but holds no function"));
            }
            if "<function=".starts_with(body) {
                return Ok(false);
            }
            return Err(ParseError::new(format!(
                "expected `<function=` after the call block, found {:?}",
                truncate_at(body, 40)
            )));
        };
        let Some(name_end) = after_open.find('>') else {
            if eos {
                return Err(ParseError::new(
                    "the completion ended inside the function's name",
                ));
            }
            return Ok(false);
        };
        let name = after_open[..name_end].trim();
        if name.is_empty() {
            return Err(ParseError::new("the call's `name` is empty"));
        }
        let mut rest = &after_open[name_end + 1..];
        let mut arguments: serde_json::Map<String, serde_json::Value> = serde_json::Map::new();
        loop {
            let ws = rest.trim_start();
            if let Some(after) = ws.strip_prefix(function_close) {
                let tail = after.trim_start();
                if tail.starts_with(block_close) {
                    let call = xml_call(name, &arguments)?;
                    out.push(Event::Call(call));
                    let consumed = start + (body.len() - tail.len()) + block_close.len();
                    self.buf.drain(..consumed);
                    self.in_block = false;
                    self.after_call = true;
                    return Ok(true);
                }
                if tail.is_empty() || block_close.starts_with(tail) {
                    if eos {
                        let call = xml_call(name, &arguments)?;
                        out.push(Event::Call(call));
                        self.buf.clear();
                        self.in_block = false;
                        self.after_call = true;
                        return Ok(true);
                    }
                    return Ok(false);
                }
                return Err(ParseError::new(format!(
                    "expected `{block_close}` after `{function_close}`, found {:?}",
                    truncate_at(tail, 40)
                )));
            }
            if let Some(after) = ws.strip_prefix("<parameter=") {
                let Some(key_end) = after.find('>') else {
                    if eos {
                        return Err(ParseError::new(
                            "the completion ended inside a parameter's name",
                        ));
                    }
                    return Ok(false);
                };
                let key = after[..key_end].trim();
                if key.is_empty() {
                    return Err(ParseError::new("a parameter has an empty name"));
                }
                let after_key = &after[key_end + 1..];
                let Some(value_end) = after_key.find("</parameter>") else {
                    if eos {
                        return Err(ParseError::new(
                            "the completion ended inside a parameter's value",
                        ));
                    }
                    return Ok(false);
                };
                arguments.insert(
                    key.to_string(),
                    serde_json::Value::String(after_key[..value_end].trim().to_string()),
                );
                rest = &after_key[value_end + "</parameter>".len()..];
                continue;
            }
            if ws.is_empty() || "<parameter=".starts_with(ws) || function_close.starts_with(ws) {
                if eos {
                    return Err(ParseError::new(
                        "the completion ended inside the call's parameters",
                    ));
                }
                return Ok(false);
            }
            return Err(ParseError::new(format!(
                "expected `<parameter=` or `{function_close}` after the call's name, found {:?}",
                truncate_at(ws, 40)
            )));
        }
    }

    /// Consume one Ministral 3 `[TOOL_CALLS]name[ARGS]{…}` block (#709).
    ///
    /// The name is the text before `[ARGS]`; the JSON value after it is the
    /// arguments object itself. One call per block; the next `[TOOL_CALLS]` or EOS
    /// ends it. The arguments keep the model's own bytes, like the JSON rows.
    fn name_args_step(&mut self, out: &mut Vec<Event>, eos: bool) -> Result<bool, ParseError> {
        let format::BlockShape::NameThenArgs { args_marker } = self.format.shape else {
            unreachable!("name_args_step is called only for the name-in-anchor shape")
        };
        let start = self.buf.len() - self.buf.trim_start().len();
        let body = &self.buf[start..];
        let Some(args_at) = body.find(args_marker) else {
            if eos {
                return Err(ParseError::new(
                    "the block opened but holds no call (expected `[ARGS]`)",
                ));
            }
            return Ok(false);
        };
        let name = body[..args_at].trim();
        if name.is_empty() {
            return Err(ParseError::new("the call's `name` is empty"));
        }
        let after_marker = start + args_at + args_marker.len();
        let ws = self.buf[after_marker..].len() - self.buf[after_marker..].trim_start().len();
        let value_start = after_marker + ws;
        let len = match raw_value_len(&self.buf[value_start..]) {
            Ok(len) => len,
            Err(RawStatus::Eof) => {
                if eos {
                    return Err(ParseError::new(
                        "the completion ended inside the call's arguments",
                    ));
                }
                return Ok(false);
            }
            Err(RawStatus::Bad(detail)) => {
                return Err(ParseError::new(format!(
                    "the call's arguments do not parse: {detail} (near {:?})",
                    truncate_at(&self.buf[value_start..], 160)
                )))
            }
        };
        let value_end = value_start + len;
        let rest = &self.buf[value_end..];
        let ws2 = rest.len() - rest.trim_start().len();
        let next = &rest[ws2..];
        let open = self.format.open;
        if next.starts_with(open) {
            let call = call_from_arguments(name, &self.buf[value_start..value_end])?;
            out.push(Event::Call(call));
            self.buf.drain(..value_end + ws2 + open.len());
            // The next call starts now: stay in the block.
            return Ok(true);
        }
        if next.is_empty() || open.starts_with(next) {
            if eos {
                let call = call_from_arguments(name, &self.buf[value_start..value_end])?;
                out.push(Event::Call(call));
                self.buf.clear();
                self.in_block = false;
                self.after_call = true;
                return Ok(true);
            }
            return Ok(false);
        }
        Err(ParseError::new(format!(
            "expected `{open}` after the call's arguments, found {:?}",
            truncate_at(next, 40)
        )))
    }
}

/// The GBNF for `tool_choice: "required"` (one of `names`) or a forced function.
///
/// It forces the model's own tool-call syntax from the format row (ADR-0066): the
/// row's literals wrap a `name` fixed or enumerated at `ToolFormat::name_key`
/// and a JSON object for the arguments (reusing the schema compiler's generic
/// `j-object` rules). The result is the block this module parses, so extraction
/// and the constraint agree by construction.
///
/// `None` means the template's format is recognised but has no grammar shape yet;
/// the caller answers a named `400` rather than accepting an unconstrained
/// generation.
pub fn forced_call_grammar(
    format: &format::ToolFormat,
    names: &[&str],
    forced: Option<&str>,
) -> Option<String> {
    let shape = format.grammar?;
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
    grammar.push_str(&gbnf_literal(shape.open));
    grammar.push(' ');
    grammar.push_str(&name_rule);
    grammar.push(' ');
    grammar.push_str(&gbnf_literal(shape.between));
    grammar.push_str("j-object");
    if !shape.close.is_empty() {
        grammar.push(' ');
        grammar.push_str(&gbnf_literal(shape.close));
    }
    grammar.push('\n');
    Some(grammar)
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
pub fn parse_completion(
    text: &str,
    format: &'static format::ToolFormat,
) -> Result<(String, Vec<ToolCall>), ParseError> {
    let mut parser = Parser::new(format);
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

/// Read one call object, picking the row's name and arguments keys.
fn parse_block(json: &str, format: &format::ToolFormat) -> Result<ToolCall, ParseError> {
    let object: HashMap<String, &RawValue> = serde_json::from_str(json)
        .map_err(|e| ParseError::new(format!("the call object does not parse: {e}")))?;
    call_from_object(&object, format)
}

/// Read a block that holds a JSON array of call objects (Mistral Nemo).
///
/// The array is the block; an empty array is a block with no call, which is a
/// refusal rather than a silent no-op.
fn parse_call_array(json: &str, format: &format::ToolFormat) -> Result<Vec<ToolCall>, ParseError> {
    let objects: Vec<HashMap<String, &RawValue>> = serde_json::from_str(json).map_err(|e| {
        ParseError::new(format!(
            "the call block is not an array of call objects: {e}"
        ))
    })?;
    if objects.is_empty() {
        return Err(ParseError::new("the call block holds no call"));
    }
    objects
        .iter()
        .map(|object| call_from_object(object, format))
        .collect()
}

/// Take the name and arguments out of one call object.
///
/// The object is a map of raw values, so the arguments stay the model's own
/// bytes (ADR-0063 decision 3); a JSON string argument is decoded because OpenAI
/// sends the call's arguments as text.
fn call_from_object(
    object: &HashMap<String, &RawValue>,
    format: &format::ToolFormat,
) -> Result<ToolCall, ParseError> {
    let name = match object.get(format.name_key) {
        Some(raw) => serde_json::from_str::<String>(raw.get()).map_err(|e| {
            ParseError::new(format!(
                "the call's `{}` is not a string: {e}",
                format.name_key
            ))
        })?,
        None => {
            return Err(ParseError::new(format!(
                "the call has no `{}`",
                format.name_key
            )))
        }
    };
    if name.trim().is_empty() {
        return Err(ParseError::new("the call's `name` is empty"));
    }
    let raw = object
        .get(format.args_key)
        .ok_or_else(|| ParseError::new(format!("the call has no `{}`", format.args_key)))?;
    let text = raw.get();
    // `null` is the absent arm, the way `Option<&RawValue>` read it before the
    // format table moved the key name into a row.
    if text == "null" {
        return Err(ParseError::new(format!(
            "the call has no `{}`",
            format.args_key
        )));
    }
    // A string argument is the call's text (OpenAI's shape); anything else is
    // carried as the bytes the model wrote.
    let arguments = if text.starts_with('"') {
        serde_json::from_str::<String>(text).map_err(|e| {
            ParseError::new(format!("the call's arguments string does not decode: {e}"))
        })?
    } else {
        text.to_string()
    };
    Ok(ToolCall { name, arguments })
}

/// Build a call whose name came from the anchor and whose arguments are the JSON
/// value after `[ARGS]` (Ministral 3).
///
/// The object keeps the model's bytes; a JSON string is decoded, the same rule
/// the wrapper rows use for their `arguments` field.
fn call_from_arguments(name: &str, json: &str) -> Result<ToolCall, ParseError> {
    let text = json.trim();
    if text == "null" {
        return Err(ParseError::new("the call has no arguments"));
    }
    let arguments = if text.starts_with('"') {
        serde_json::from_str::<String>(text).map_err(|e| {
            ParseError::new(format!("the call's arguments string does not decode: {e}"))
        })?
    } else {
        text.to_string()
    };
    Ok(ToolCall {
        name: name.to_string(),
        arguments,
    })
}

/// Serialize a Qwen3.5 parameter map to OpenAI's arguments string.
///
/// The XML values are text, so the object is built in memory and serialized as
/// JSON; unlike the JSON rows, these are not the model's own bytes.
fn xml_call(
    name: &str,
    arguments: &serde_json::Map<String, serde_json::Value>,
) -> Result<ToolCall, ParseError> {
    let arguments = serde_json::to_string(arguments)
        .map_err(|e| ParseError::new(format!("the call's arguments do not serialize: {e}")))?;
    Ok(ToolCall {
        name: name.to_string(),
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
