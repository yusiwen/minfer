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
        if matches!(self.format.shape, format::BlockShape::CallThenJson { .. }) {
            return self.call_then_json_step(out, eos);
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
            format::BlockShape::CallThenJson { .. } => {
                unreachable!("the Gemma shape is dispatched before the JSON body parse")
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

    /// Consume one Gemma 4 `<|tool_call>call:NAME{…}<tool_call|>` block (#710).
    ///
    /// The body is lenient JSON (unquoted keys, `<|"|>` strings); it is read into
    /// a `serde_json::Value` and serialized to JSON text for OpenAI's
    /// `arguments`. A complete body at EOS without the close is a usable call,
    /// the same implicit-close rule the other rows use.
    fn call_then_json_step(&mut self, out: &mut Vec<Event>, eos: bool) -> Result<bool, ParseError> {
        let format::BlockShape::CallThenJson { name_prefix, close } = self.format.shape else {
            unreachable!("call_then_json_step is called only for the Gemma shape")
        };
        let start = self.buf.len() - self.buf.trim_start().len();
        let body = &self.buf[start..];
        let Some(after_prefix) = body.strip_prefix(name_prefix) else {
            if eos {
                return Err(ParseError::new("the block opened but holds no call"));
            }
            if name_prefix.starts_with(body) {
                return Ok(false);
            }
            return Err(ParseError::new(format!(
                "expected `{name_prefix}` after the call block, found {:?}",
                truncate_at(body, 40)
            )));
        };
        let Some(brace) = after_prefix.find('{') else {
            if eos {
                return Err(ParseError::new("the call has no `{` body"));
            }
            return Ok(false);
        };
        let name = after_prefix[..brace].trim();
        if name.is_empty() {
            return Err(ParseError::new("the call's `name` is empty"));
        }
        let value_start = start + name_prefix.len() + brace;
        let (value, len) = match gemma_json(&self.buf[value_start..]) {
            Ok(pair) => pair,
            Err(RawStatus::Eof) => {
                if eos {
                    return Err(ParseError::new(
                        "the completion ended inside the call's body",
                    ));
                }
                return Ok(false);
            }
            Err(RawStatus::Bad(detail)) => {
                return Err(ParseError::new(format!(
                    "the call's body does not parse: {detail} (near {:?})",
                    truncate_at(&self.buf[value_start..], 160)
                )))
            }
        };
        let value_end = value_start + len;
        let rest = &self.buf[value_end..];
        let ws = rest.len() - rest.trim_start().len();
        let tail = &rest[ws..];
        if tail.starts_with(close) {
            let call = call_from_value(name, &value)?;
            out.push(Event::Call(call));
            self.buf.drain(..value_end + ws + close.len());
            self.in_block = false;
            self.after_call = true;
            return Ok(true);
        }
        if tail.is_empty() || close.starts_with(tail) {
            if eos {
                let call = call_from_value(name, &value)?;
                out.push(Event::Call(call));
                self.buf.clear();
                self.in_block = false;
                self.after_call = true;
                return Ok(true);
            }
            return Ok(false);
        }
        Err(ParseError::new(format!(
            "expected `{close}` after the call's body, found {:?}",
            truncate_at(tail, 40)
        )))
    }
}

/// The GBNF for `tool_choice: "required"` (every offered tool) or a forced function.
///
/// It forces the model's own tool-call syntax from the format row (ADR-0066) and
/// the selected tool's `parameters` schema (ADR-0070): each selected tool is one
/// root alternative — the row's open/between/close literals around a literal name
/// and a schema-compiled `args-i` rule. Qwen3.5's XML shape has its own
/// flat-scalar `<parameter=…>` builder instead (ADR-0071). The result is the block
/// this module parses, so extraction and the constraint agree by construction.
///
/// `Ok(None)` means the template's format is recognised but has no grammar shape
/// yet; the caller answers a named `400`. `Err` is a schema that cannot be
/// compiled (also a named `400`), never a fallback to an unconstrained object.
pub fn forced_call_grammar(
    format: &format::ToolFormat,
    tools: &[minfer::template::ToolJson],
    forced: Option<&str>,
) -> Result<Option<String>, String> {
    let Some(shape) = format.grammar else {
        return Ok(None);
    };
    let selected = selected_tools(tools, forced)?;
    let mut grammar = String::new();
    push_generic_rules(&mut grammar);
    let mut alternatives = Vec::new();
    match shape {
        format::GrammarShape::Json {
            open,
            between,
            close,
        } => {
            for (index, (name, schema)) in selected.iter().enumerate() {
                let args_rule = format!("args-{index}");
                let (rules, body) = minfer::grammar::compile_schema_rules(schema, &args_rule)?;
                push_rules(&mut grammar, &rules);
                push_rule(&mut grammar, &args_rule, &body);
                let mut alternative = format!(
                    "{} {} {} {}",
                    gbnf_literal(open),
                    gbnf_literal(name),
                    gbnf_literal(between),
                    args_rule
                );
                if !close.is_empty() {
                    alternative.push(' ');
                    alternative.push_str(&gbnf_literal(close));
                }
                alternatives.push(alternative);
            }
        }
        format::GrammarShape::Xml { .. } => {
            for (name, schema) in &selected {
                alternatives.push(xml_call_alternative(shape, format.open, name, schema)?);
            }
        }
        format::GrammarShape::Gemma { .. } => {
            push_gemma_rules(&mut grammar);
            for (index, (name, schema)) in selected.iter().enumerate() {
                let (rules, args) = gemma_schema_rules(schema, &format!("gemma-{index}"))?;
                push_rules(&mut grammar, &rules);
                let format::GrammarShape::Gemma { name_prefix, close } = shape else {
                    unreachable!("the Gemma arm matched the Gemma shape")
                };
                let mut alternative = format!(
                    "{} {} {} {}",
                    gbnf_literal(format.open),
                    gbnf_literal(name_prefix),
                    gbnf_literal(name),
                    args
                );
                alternative.push(' ');
                alternative.push_str(&gbnf_literal(close));
                alternatives.push(alternative);
            }
        }
    }
    grammar.push_str("root ::= ");
    grammar.push_str(&alternatives.join("\n    | "));
    grammar.push('\n');
    Ok(Some(grammar))
}

fn selected_tools<'a>(
    tools: &'a [minfer::template::ToolJson],
    forced: Option<&str>,
) -> Result<Vec<(&'a str, serde_json::Value)>, String> {
    let mut selected: Vec<(&str, serde_json::Value)> = Vec::new();
    for tool in tools {
        let Some(function) = tool.get("function") else {
            continue;
        };
        let Some(name) = function.get("name").and_then(|name| name.as_str()) else {
            continue;
        };
        if forced.is_some_and(|forced| forced != name) {
            continue;
        }
        let parameters = match function.get("parameters") {
            Some(parameters) => serde_json::to_value(parameters)
                .map_err(|e| format!("tool_choice: `{name}`'s parameters do not serialize: {e}"))?,
            // A tool without a schema still gets an object, not an unconstrained
            // JSON value: the arguments are an object by OpenAI's shape.
            None => serde_json::json!({"type": "object"}),
        };
        selected.push((name, parameters));
    }
    if selected.is_empty() {
        return Err("tool_choice found no offered tool to constrain".to_string());
    }
    Ok(selected)
}

fn push_generic_rules(grammar: &mut String) {
    for (name, rhs) in minfer::grammar::GENERIC_RULES {
        push_rule(grammar, name, rhs);
    }
}

fn push_rules(grammar: &mut String, rules: &[(String, String)]) {
    for (name, rhs) in rules {
        push_rule(grammar, name, rhs);
    }
}

fn push_rule(grammar: &mut String, name: &str, body: &str) {
    grammar.push_str(name);
    grammar.push_str(" ::= ");
    grammar.push_str(body);
    grammar.push('\n');
}

/// One root alternative for Qwen3.5's XML block: the row's open literal, then
/// `<function=NAME>`, the schema's parameters, `</function>` and `</tool_call>`.
fn xml_call_alternative(
    shape: format::GrammarShape,
    block_open: &str,
    name: &str,
    schema: &serde_json::Value,
) -> Result<String, String> {
    let format::GrammarShape::Xml {
        function_open,
        function_close,
        block_close,
        parameter_open,
        parameter_close,
    } = shape
    else {
        unreachable!("xml_call_alternative is called only for the XML shape")
    };
    let entries = xml_parameters(schema)?;
    let mut alternative = format!(
        "{} j-ws {} {} \">\"",
        gbnf_literal(block_open),
        gbnf_literal(function_open),
        gbnf_literal(name)
    );
    for (key, value) in entries {
        alternative.push_str(" j-ws ");
        alternative.push_str(&gbnf_literal(&format!("{parameter_open}{key}>")));
        alternative.push('(');
        alternative.push_str(&value);
        alternative.push(')');
        alternative.push_str(&gbnf_literal(parameter_close));
    }
    alternative.push_str(" j-ws ");
    alternative.push_str(&gbnf_literal(function_close));
    alternative.push_str(" j-ws ");
    alternative.push_str(&gbnf_literal(block_close));
    Ok(alternative)
}

/// The object keywords the XML parameter grammar understands; annotations are
/// ignored, and every language-changing keyword not here is a named refusal.
const XML_OBJECT_KEYWORDS: &[&str] = &[
    "type",
    "properties",
    "required",
    "additionalProperties",
    "description",
    "title",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

/// The value keywords the XML parameter grammar understands. A value's JSON
/// type is not representable (the parser stores text), only its text form.
const XML_VALUE_KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "description",
    "title",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
];

fn xml_parameters(schema: &serde_json::Value) -> Result<Vec<(String, String)>, String> {
    let object = match schema {
        serde_json::Value::Bool(true) => return Ok(Vec::new()),
        serde_json::Value::Bool(false) => {
            return Err("json-schema: `false` admits no value and cannot be compiled".to_string())
        }
        serde_json::Value::Object(object) => object,
        other => {
            return Err(format!(
                "json-schema: a tool parameters schema must be an object or a boolean, got {}",
                minfer::grammar::json_type_name(other)
            ))
        }
    };
    for key in object.keys() {
        if !XML_OBJECT_KEYWORDS.contains(&key.as_str()) {
            return Err(format!(
                "json-schema: `{key}` cannot be expressed in the XML parameter grammar"
            ));
        }
    }
    match object.get("type") {
        None => {}
        Some(serde_json::Value::String(ty)) if ty == "object" => {}
        Some(serde_json::Value::String(ty)) => {
            return Err(format!(
                "json-schema: the XML tool parameters must be an object schema, got `{ty}`"
            ))
        }
        Some(_) => {
            return Err("json-schema: `type` must be a string".to_string());
        }
    }
    let required = match object.get("required") {
        None => Vec::new(),
        Some(serde_json::Value::Array(names)) => {
            let mut out = Vec::new();
            for name in names {
                match name {
                    serde_json::Value::String(name) => out.push(name.as_str()),
                    other => {
                        return Err(format!(
                            "json-schema: `required` entries must be strings, got {}",
                            minfer::grammar::json_type_name(other)
                        ))
                    }
                }
            }
            out
        }
        Some(other) => {
            return Err(format!(
                "json-schema: `required` must be an array, got {}",
                minfer::grammar::json_type_name(other)
            ))
        }
    };
    let properties = match object.get("properties") {
        None => return Ok(Vec::new()),
        Some(serde_json::Value::Object(properties)) => properties,
        Some(other) => {
            return Err(format!(
                "json-schema: `properties` must be an object, got {}",
                minfer::grammar::json_type_name(other)
            ))
        }
    };
    for name in &required {
        if !properties.contains_key(*name) {
            return Err(format!(
                "json-schema: `required` names `{name}`, which is not declared in `properties`"
            ));
        }
    }
    let mut entries = Vec::new();
    for (name, property) in properties {
        if name.contains('<') || name.contains('>') {
            return Err(format!(
                "json-schema: the parameter name `{name}` cannot be placed in an XML tag"
            ));
        }
        if name.trim() != name {
            return Err(format!(
                "json-schema: the parameter name `{name}` has leading or trailing whitespace"
            ));
        }
        entries.push((name.clone(), xml_value_rule(property)?));
    }
    Ok(entries)
}

fn xml_value_rule(schema: &serde_json::Value) -> Result<String, String> {
    let object = match schema {
        serde_json::Value::Bool(true) => return Ok("[^<]*".to_string()),
        serde_json::Value::Bool(false) => {
            return Err("json-schema: `false` admits no value and cannot be compiled".to_string())
        }
        serde_json::Value::Object(object) => object,
        other => {
            return Err(format!(
                "json-schema: a parameter schema must be an object or a boolean, got {}",
                minfer::grammar::json_type_name(other)
            ))
        }
    };
    for key in object.keys() {
        if !XML_VALUE_KEYWORDS.contains(&key.as_str()) {
            return Err(format!(
                "json-schema: `{key}` cannot be expressed in the XML parameter grammar"
            ));
        }
    }
    if let Some(values) = object.get("enum") {
        let values = values
            .as_array()
            .ok_or_else(|| "json-schema: `enum` must be an array of values".to_string())?;
        return xml_enum_rule(values);
    }
    if let Some(value) = object.get("const") {
        return xml_scalar_text(value);
    }
    match object.get("type") {
        None => Ok("[^<]*".to_string()),
        Some(serde_json::Value::String(ty)) => match ty.as_str() {
            "string" => Ok("[^<]*".to_string()),
            "integer" => Ok("j-integer".to_string()),
            "number" => Ok("j-number".to_string()),
            "boolean" => Ok("\"true\" | \"false\"".to_string()),
            "object" | "array" | "null" => Err(format!(
                "json-schema: an XML parameter value cannot be a `{ty}` (the parser stores text)"
            )),
            other => Err(format!("json-schema: unknown type `{other}`")),
        },
        Some(_) => Err("json-schema: `type` must be a string".to_string()),
    }
}

fn xml_enum_rule(values: &[serde_json::Value]) -> Result<String, String> {
    if values.is_empty() {
        return Err("json-schema: `enum` must contain at least one value".to_string());
    }
    let mut alternatives = Vec::new();
    for value in values {
        alternatives.push(xml_scalar_text(value)?);
    }
    Ok(alternatives.join(" | "))
}

fn xml_scalar_text(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(text) => {
            if text.contains('<') {
                return Err(
                    "json-schema: a string containing `<` cannot be represented in the XML \
                     parameter text"
                        .to_string(),
                );
            }
            Ok(gbnf_literal(text))
        }
        serde_json::Value::Number(number) => Ok(gbnf_literal(&number.to_string())),
        serde_json::Value::Bool(value) => Ok(gbnf_literal(if *value { "true" } else { "false" })),
        serde_json::Value::Null => {
            Err("json-schema: `null` cannot be represented in the XML parameter text".to_string())
        }
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => Err(
            "json-schema: an object/array value cannot be represented in the XML parameter text"
                .to_string(),
        ),
    }
}

/// The Gemma dialect's global rules: `<|"|>` strings and the bare-key/object
/// forms the parser's `GemmaJson` reads.
fn push_gemma_rules(grammar: &mut String) {
    let delim = gbnf_literal("<|\"|>");
    push_rule(grammar, "g-string", &format!("{delim} [^<]* {delim}"));
    push_rule(grammar, "g-key", "[A-Za-z0-9_]+ | g-string");
    push_rule(
        grammar,
        "g-object",
        "\"{\" j-ws \"}\" | \"{\" j-ws g-key j-ws \":\" j-ws g-value \
         (j-ws \",\" j-ws g-key j-ws \":\" j-ws g-value)* j-ws \"}\"",
    );
    push_rule(
        grammar,
        "g-array",
        "\"[\" j-ws \"]\" | \"[\" j-ws g-value (j-ws \",\" j-ws g-value)* j-ws \"]\"",
    );
    push_rule(
        grammar,
        "g-value",
        "g-object | g-array | g-string | j-number | \"true\" | \"false\" | \"null\"",
    );
}

/// The schema keywords the Gemma dialect understands; annotations are ignored,
/// and every other keyword is a named refusal.
const GEMMA_KEYWORDS: &[&str] = &[
    "type",
    "enum",
    "const",
    "anyOf",
    "oneOf",
    "properties",
    "required",
    "additionalProperties",
    "items",
    "description",
    "title",
    "default",
    "examples",
    "deprecated",
    "readOnly",
    "writeOnly",
    "$schema",
];

fn is_gemma_annotation(key: &str) -> bool {
    matches!(
        key,
        "description"
            | "title"
            | "default"
            | "examples"
            | "deprecated"
            | "readOnly"
            | "writeOnly"
            | "$schema"
    )
}

struct GemmaCompiler {
    rules: Vec<(String, String)>,
    counter: usize,
    prefix: String,
}

impl GemmaCompiler {
    fn fresh(&mut self, hint: &str) -> String {
        self.counter += 1;
        let clean: String = hint
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                    c
                } else {
                    '-'
                }
            })
            .collect();
        format!("{}-{clean}-{}", self.prefix, self.counter)
    }

    fn rule(&mut self, hint: &str, body: String) -> String {
        let name = self.fresh(hint);
        self.rules.push((name.clone(), body));
        name
    }

    /// The tool's `parameters` must be an object schema; the body is the object.
    fn root(&mut self, schema: &serde_json::Value) -> Result<String, String> {
        match schema {
            serde_json::Value::Bool(true) => Ok("g-object".to_string()),
            serde_json::Value::Bool(false) => {
                Err("json-schema: `false` admits no value and cannot be compiled".to_string())
            }
            serde_json::Value::Object(object) => {
                for key in object.keys() {
                    if !GEMMA_KEYWORDS.contains(&key.as_str()) {
                        return Err(format!(
                            "json-schema: `{key}` cannot be expressed in the Gemma grammar"
                        ));
                    }
                }
                if ["anyOf", "oneOf", "enum", "const", "items"]
                    .iter()
                    .any(|key| object.contains_key(*key))
                {
                    return Err(
                        "json-schema: the tool parameters must be an object schema".to_string()
                    );
                }
                match object.get("type") {
                    None => {
                        if ["properties", "required", "additionalProperties"]
                            .iter()
                            .any(|key| object.contains_key(*key))
                        {
                            self.value(schema, "args")
                        } else {
                            Ok("g-object".to_string())
                        }
                    }
                    Some(serde_json::Value::String(ty)) if ty == "object" => {
                        self.value(schema, "args")
                    }
                    Some(serde_json::Value::String(ty)) => Err(format!(
                        "json-schema: the tool parameters must be an object schema, got `{ty}`"
                    )),
                    Some(_) => Err("json-schema: `type` must be a string".to_string()),
                }
            }
            other => Err(format!(
                "json-schema: a schema must be an object or a boolean, got {}",
                minfer::grammar::json_type_name(other)
            )),
        }
    }

    fn value(&mut self, schema: &serde_json::Value, hint: &str) -> Result<String, String> {
        let object = match schema {
            serde_json::Value::Bool(true) => return Ok("g-value".to_string()),
            serde_json::Value::Bool(false) => {
                return Err(
                    "json-schema: `false` admits no value and cannot be compiled".to_string(),
                )
            }
            serde_json::Value::Object(object) => object,
            other => {
                return Err(format!(
                    "json-schema: a schema must be an object or a boolean, got {}",
                    minfer::grammar::json_type_name(other)
                ))
            }
        };
        for key in object.keys() {
            if !GEMMA_KEYWORDS.contains(&key.as_str()) {
                return Err(format!(
                    "json-schema: `{key}` cannot be expressed in the Gemma grammar"
                ));
            }
        }
        if object.contains_key("enum") || object.contains_key("const") {
            if object.contains_key("enum") && object.contains_key("const") {
                return Err("json-schema: `enum` and `const` cannot be combined".to_string());
            }
            if object.keys().any(|key| {
                !is_gemma_annotation(key.as_str())
                    && key.as_str() != "type"
                    && key.as_str() != "enum"
                    && key.as_str() != "const"
            }) {
                return Err(
                    "json-schema: `enum`/`const` combined with other constraints cannot be \
                     expressed"
                        .to_string(),
                );
            }
            if let Some(values) = object.get("enum") {
                let values = values
                    .as_array()
                    .ok_or_else(|| "json-schema: `enum` must be an array of values".to_string())?;
                if values.is_empty() {
                    return Err("json-schema: `enum` must contain at least one value".to_string());
                }
                let mut alternatives = Vec::new();
                for value in values {
                    alternatives.push(gemma_literal(value)?);
                }
                return Ok(alternatives.join(" | "));
            }
            return gemma_literal(object.get("const").expect("the const arm"));
        }
        if object.contains_key("anyOf") || object.contains_key("oneOf") {
            if object.contains_key("anyOf") && object.contains_key("oneOf") {
                return Err("json-schema: `anyOf` and `oneOf` cannot be combined".to_string());
            }
            let key = if object.contains_key("anyOf") {
                "anyOf"
            } else {
                "oneOf"
            };
            if object
                .keys()
                .any(|other| !is_gemma_annotation(other.as_str()) && other.as_str() != key)
            {
                return Err(
                    "json-schema: a union combined with other constraints cannot be expressed"
                        .to_string(),
                );
            }
            let branches = object
                .get(key)
                .and_then(|branches| branches.as_array())
                .ok_or_else(|| "json-schema: `anyOf`/`oneOf` must be an array".to_string())?;
            if branches.is_empty() {
                return Err("json-schema: `anyOf`/`oneOf` must not be empty".to_string());
            }
            let mut alternatives = Vec::new();
            for (index, branch) in branches.iter().enumerate() {
                alternatives.push(self.value(branch, &format!("{hint}-branch-{index}"))?);
            }
            return Ok(alternatives.join(" | "));
        }
        let ty = match object.get("type") {
            None => None,
            Some(serde_json::Value::String(ty)) => Some(ty.as_str()),
            Some(_) => return Err("json-schema: `type` must be a string".to_string()),
        };
        let has_object = ["properties", "required", "additionalProperties"]
            .iter()
            .any(|key| object.contains_key(*key));
        let has_array = object.contains_key("items");
        if let Some(ty) = ty {
            if ty != "object" && has_object {
                return Err(format!(
                    "json-schema: object keywords present but `type` is `{ty}`"
                ));
            }
            if ty != "array" && has_array {
                return Err(format!(
                    "json-schema: array keywords present but `type` is `{ty}`"
                ));
            }
        }
        match ty {
            None => {
                if has_object {
                    self.object(object, hint)
                } else if has_array {
                    self.array(object, hint)
                } else {
                    Ok("g-value".to_string())
                }
            }
            Some("object") => self.object(object, hint),
            Some("array") => self.array(object, hint),
            Some("string") => Ok("g-string".to_string()),
            Some("integer") => Ok("j-integer".to_string()),
            Some("number") => Ok("j-number".to_string()),
            Some("boolean") => Ok("\"true\" | \"false\"".to_string()),
            Some("null") => Ok("\"null\"".to_string()),
            Some(other) => Err(format!("json-schema: unknown type `{other}`")),
        }
    }

    fn object(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        hint: &str,
    ) -> Result<String, String> {
        let properties = match object.get("properties") {
            None => None,
            Some(serde_json::Value::Object(properties)) => Some(properties),
            Some(other) => {
                return Err(format!(
                    "json-schema: `properties` must be an object, got {}",
                    minfer::grammar::json_type_name(other)
                ))
            }
        };
        let required = match object.get("required") {
            None => Vec::new(),
            Some(serde_json::Value::Array(names)) => {
                let mut out = Vec::new();
                for name in names {
                    match name {
                        serde_json::Value::String(name) => out.push(name.as_str()),
                        other => {
                            return Err(format!(
                                "json-schema: `required` entries must be strings, got {}",
                                minfer::grammar::json_type_name(other)
                            ))
                        }
                    }
                }
                out
            }
            Some(other) => {
                return Err(format!(
                    "json-schema: `required` must be an array, got {}",
                    minfer::grammar::json_type_name(other)
                ))
            }
        };
        let mut entries = Vec::new();
        if let Some(properties) = properties {
            for name in &required {
                if !properties.contains_key(*name) {
                    return Err(format!(
                        "json-schema: `required` names `{name}`, which is not declared in \
                         `properties`"
                    ));
                }
            }
            for (name, property) in properties {
                let key = gemma_key(name)?;
                let value = self.value(property, &format!("{hint}-{name}"))?;
                entries.push(format!("{key} j-ws \":\" j-ws ({value})"));
            }
        } else if !required.is_empty() {
            return Err(
                "json-schema: `required` names a property but `properties` is absent".to_string(),
            );
        }
        let body = if entries.is_empty() {
            "\"{\" j-ws \"}\"".to_string()
        } else {
            format!(
                "\"{{\" j-ws {} j-ws \"}}\"",
                entries.join(" j-ws \",\" j-ws ")
            )
        };
        Ok(self.rule(&format!("{hint}-object"), body))
    }

    fn array(
        &mut self,
        object: &serde_json::Map<String, serde_json::Value>,
        hint: &str,
    ) -> Result<String, String> {
        let body = match object.get("items") {
            None => "g-array".to_string(),
            Some(items) => {
                let value = self.value(items, &format!("{hint}-item"))?;
                format!(
                    "\"[\" j-ws \"]\" | \"[\" j-ws ({value}) (j-ws \",\" j-ws ({value}))* \
                     j-ws \"]\""
                )
            }
        };
        Ok(self.rule(&format!("{hint}-array"), body))
    }
}

fn gemma_schema_rules(
    schema: &serde_json::Value,
    prefix: &str,
) -> Result<(Vec<(String, String)>, String), String> {
    let mut compiler = GemmaCompiler {
        rules: Vec::new(),
        counter: 0,
        prefix: prefix.to_string(),
    };
    let root = compiler.root(schema)?;
    Ok((compiler.rules, root))
}

/// A property name as the dialect writes it: bare when it is an identifier, a
/// `<|"|>` string otherwise. A name containing `<` cannot be written safely.
fn gemma_key(name: &str) -> Result<String, String> {
    if name.contains('<') {
        return Err(format!(
            "json-schema: the property name `{name}` cannot be placed in a Gemma string"
        ));
    }
    if !name.is_empty() && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
        Ok(gbnf_literal(name))
    } else {
        Ok(gbnf_literal(&format!("<|\"|>{name}<|\"|>")))
    }
}

fn gemma_literal(value: &serde_json::Value) -> Result<String, String> {
    match value {
        serde_json::Value::String(text) => {
            if text.contains('<') {
                return Err(
                    "json-schema: a string containing `<` cannot be represented in a Gemma \
                     string"
                        .to_string(),
                );
            }
            Ok(gbnf_literal(&format!("<|\"|>{text}<|\"|>")))
        }
        serde_json::Value::Number(number) => Ok(gbnf_literal(&number.to_string())),
        serde_json::Value::Bool(value) => Ok(gbnf_literal(if *value { "true" } else { "false" })),
        serde_json::Value::Null => Ok(gbnf_literal("null")),
        serde_json::Value::Object(_) | serde_json::Value::Array(_) => Err(
            "json-schema: an object/array enum value cannot be represented as a Gemma literal"
                .to_string(),
        ),
    }
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

/// Serialize a lenient Gemma body to OpenAI's arguments string.
fn call_from_value(name: &str, value: &serde_json::Value) -> Result<ToolCall, ParseError> {
    let arguments = serde_json::to_string(value)
        .map_err(|e| ParseError::new(format!("the call's arguments do not serialize: {e}")))?;
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

/// A cursor over Gemma 4's lenient JSON body (#710).
///
/// The dialect has unquoted keys and `<|"|>` string delimiters; numbers,
/// booleans, null, arrays and nested objects are JSON's. A partial token at the
/// end of the buffer returns [`RawStatus::Eof`], so the streaming parser holds
/// rather than refusing a call that is still arriving.
struct GemmaJson<'a> {
    text: &'a str,
    at: usize,
}

impl<'a> GemmaJson<'a> {
    fn skip_ws(&mut self) {
        while let Some(b) = self.text.as_bytes().get(self.at) {
            if b.is_ascii_whitespace() {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn value(&mut self) -> Result<serde_json::Value, RawStatus> {
        self.skip_ws();
        let rest = &self.text[self.at..];
        if rest.starts_with("<|\"|>") || rest.starts_with('"') {
            return Ok(serde_json::Value::String(self.string()?));
        }
        if !rest.is_empty() && "<|\"|>".starts_with(rest) {
            return Err(RawStatus::Eof);
        }
        match rest.as_bytes().first().copied() {
            None => Err(RawStatus::Eof),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => {
                for (literal, value) in [
                    ("true", serde_json::Value::Bool(true)),
                    ("false", serde_json::Value::Bool(false)),
                    ("null", serde_json::Value::Null),
                ] {
                    if rest.starts_with(literal) {
                        self.at += literal.len();
                        return Ok(value);
                    }
                    if literal.starts_with(rest) {
                        return Err(RawStatus::Eof);
                    }
                }
                Err(RawStatus::Bad("expected a value".to_string()))
            }
        }
    }

    fn string(&mut self) -> Result<String, RawStatus> {
        if self.text[self.at..].starts_with("<|\"|>") {
            self.at += 5;
            let start = self.at;
            let Some(rel) = self.text[start..].find("<|\"|>") else {
                return Err(RawStatus::Eof);
            };
            let value = self.text[start..start + rel].to_string();
            self.at = start + rel + 5;
            return Ok(value);
        }
        if self.text.as_bytes().get(self.at) != Some(&b'"') {
            return Err(RawStatus::Bad("expected a string".to_string()));
        }
        let start = self.at;
        let mut escaped = false;
        let mut at = start + 1;
        while at < self.text.len() {
            let b = self.text.as_bytes()[at];
            if escaped {
                escaped = false;
            } else if b == b'\\' {
                escaped = true;
            } else if b == b'"' {
                let slice = &self.text[start..=at];
                let value = serde_json::from_str::<String>(slice)
                    .map_err(|e| RawStatus::Bad(e.to_string()))?;
                self.at = at + 1;
                return Ok(value);
            }
            at += 1;
        }
        Err(RawStatus::Eof)
    }

    fn key(&mut self) -> Result<String, RawStatus> {
        self.skip_ws();
        if self.text[self.at..].starts_with("<|\"|>")
            || self.text.as_bytes().get(self.at) == Some(&b'"')
        {
            return self.string();
        }
        let rest = &self.text[self.at..];
        if !rest.is_empty() && "<|\"|>".starts_with(rest) {
            return Err(RawStatus::Eof);
        }
        let start = self.at;
        while let Some(b) = self.text.as_bytes().get(self.at) {
            if b.is_ascii_alphanumeric() || *b == b'_' {
                self.at += 1;
            } else {
                break;
            }
        }
        if self.at == start {
            if self.at >= self.text.len() {
                return Err(RawStatus::Eof);
            }
            return Err(RawStatus::Bad("expected a key".to_string()));
        }
        Ok(self.text[start..self.at].to_string())
    }

    fn object(&mut self) -> Result<serde_json::Value, RawStatus> {
        self.at += 1; // `{`
        let mut map = serde_json::Map::new();
        loop {
            self.skip_ws();
            match self.text.as_bytes().get(self.at) {
                None => return Err(RawStatus::Eof),
                Some(b'}') => {
                    self.at += 1;
                    return Ok(serde_json::Value::Object(map));
                }
                Some(_) => {}
            }
            let key = self.key()?;
            self.skip_ws();
            if self.text.as_bytes().get(self.at) != Some(&b':') {
                if self.at >= self.text.len() {
                    return Err(RawStatus::Eof);
                }
                return Err(RawStatus::Bad("expected `:` after a key".to_string()));
            }
            self.at += 1;
            let value = self.value()?;
            map.insert(key, value);
            self.skip_ws();
            match self.text.as_bytes().get(self.at) {
                None => return Err(RawStatus::Eof),
                Some(b',') => self.at += 1,
                Some(b'}') => {
                    self.at += 1;
                    return Ok(serde_json::Value::Object(map));
                }
                Some(_) => return Err(RawStatus::Bad("expected `,` or `}`".to_string())),
            }
        }
    }

    fn array(&mut self) -> Result<serde_json::Value, RawStatus> {
        self.at += 1; // `[`
        let mut values = Vec::new();
        loop {
            self.skip_ws();
            match self.text.as_bytes().get(self.at) {
                None => return Err(RawStatus::Eof),
                Some(b']') => {
                    self.at += 1;
                    return Ok(serde_json::Value::Array(values));
                }
                Some(_) => {}
            }
            values.push(self.value()?);
            self.skip_ws();
            match self.text.as_bytes().get(self.at) {
                None => return Err(RawStatus::Eof),
                Some(b',') => self.at += 1,
                Some(b']') => {
                    self.at += 1;
                    return Ok(serde_json::Value::Array(values));
                }
                Some(_) => return Err(RawStatus::Bad("expected `,` or `]`".to_string())),
            }
        }
    }

    fn number(&mut self) -> Result<serde_json::Value, RawStatus> {
        let start = self.at;
        while let Some(b) = self.text.as_bytes().get(self.at) {
            if b.is_ascii_digit() || matches!(b, b'-' | b'+' | b'.' | b'e' | b'E') {
                self.at += 1;
            } else {
                break;
            }
        }
        if self.at == start {
            if self.at >= self.text.len() {
                return Err(RawStatus::Eof);
            }
            return Err(RawStatus::Bad("expected a number".to_string()));
        }
        let slice = &self.text[start..self.at];
        if self.at >= self.text.len()
            && (slice.ends_with('-')
                || slice.ends_with('+')
                || slice.ends_with('e')
                || slice.ends_with('E')
                || slice.ends_with('.'))
        {
            return Err(RawStatus::Eof);
        }
        let number = serde_json::from_str::<serde_json::Number>(slice)
            .map_err(|e| RawStatus::Bad(e.to_string()))?;
        Ok(serde_json::Value::Number(number))
    }
}

fn gemma_json(text: &str) -> Result<(serde_json::Value, usize), RawStatus> {
    let mut parser = GemmaJson { text, at: 0 };
    let value = parser.value()?;
    Ok((value, parser.at))
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
