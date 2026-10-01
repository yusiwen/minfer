// Chat template rendering using minijinja
// Reads tokenizer.chat_template from GGUF, renders with message context
//
// Design record: docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md (F7, #50).
//
// Two contracts matter here:
//
//  1. A template that **cannot be rendered** is a loud error naming the
//     construct — never a silent generic ChatML prompt. The old behaviour
//     (warn + fall back) silently dropped Qwen3's think-block extraction and
//     tool-call formatting because its template uses Python string-method
//     syntax that minijinja cannot run (docs/QWEN3-SUPPORT-PLAN.md §5 gotcha
//     #9).
//  2. Python `str` methods that transformers chat templates use are provided
//     through minijinja's own extension point
//     (`Environment::set_unknown_method_callback`) with CPython semantics, so
//     the *published* template renders instead of being refused.

use minijinja::value::{Value, ValueKind};
use minijinja::{context, Environment, Error, ErrorKind, State};
use std::fmt;

/// The Python `str` methods the unknown-method hook implements. Listed in the
/// refusal message so an unsupported template tells the reader what *is*
/// available.
const SUPPORTED_STR_METHODS: &[&str] = &[
    "capitalize",
    "count",
    "endswith",
    "find",
    "join",
    "lower",
    "lstrip",
    "replace",
    "rfind",
    "rsplit",
    "rstrip",
    "split",
    "startswith",
    "strip",
    "title",
    "upper",
];

/// A chat template that minfer refuses to render.
///
/// This is deliberately **not** converted into a generic prompt: the caller
/// decides what a refusal means (CLI: exit; server: startup refusal or HTTP
/// 400; session: abort the turn).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateError {
    /// What kind of construct failed, e.g. `unsupported template construct`.
    pub kind: &'static str,
    /// minijinja's own description (it names the method / the syntax problem).
    pub detail: String,
    /// Line in the template, when minijinja reported one.
    pub line: Option<usize>,
}

impl TemplateError {
    fn classify(err: &Error) -> Self {
        let kind = match err.kind() {
            ErrorKind::UnknownMethod => "unsupported template construct",
            ErrorKind::SyntaxError => "syntax error",
            _ => "render error",
        };
        let detail = err
            .detail()
            .map(str::to_string)
            .unwrap_or_else(|| err.to_string());
        Self {
            kind,
            detail,
            line: err.line(),
        }
    }

    /// The full, user-facing refusal. Names the construct, the template line
    /// and the engine's alternative, so nothing is silent.
    pub fn message(&self) -> String {
        let at = match self.line {
            Some(n) => format!(" (template line {n})"),
            None => String::new(),
        };
        format!(
            "chat template error — {}: {}{}; minfer refuses to fall back to a generic ChatML \
             prompt. Supported Python str methods: {}",
            self.kind,
            self.detail,
            at,
            SUPPORTED_STR_METHODS.join(", "),
        )
    }
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message())
    }
}

impl std::error::Error for TemplateError {}

// ---------------------------------------------------------------------------
// Python `str` semantics (the unknown-method hook)
// ---------------------------------------------------------------------------

/// One argument of a Python string method, restricted to the shapes the chat
/// templates in the wild use.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PyArg {
    Str(String),
    Int(i64),
    None_,
    /// The fixture encoding of CPython's tuple argument (`startswith(("a","b"))`).
    StrList(Vec<String>),
}

/// A Python string method result.
#[derive(Debug, Clone, PartialEq, Eq)]
enum PyValue {
    Str(String),
    Int(i64),
    Bool(bool),
    List(Vec<String>),
}

impl PyArg {
    fn from_value(v: &Value) -> Result<Self, String> {
        match v.kind() {
            ValueKind::String => Ok(PyArg::Str(v.as_str().unwrap_or_default().to_string())),
            ValueKind::None | ValueKind::Undefined => Ok(PyArg::None_),
            ValueKind::Bool | ValueKind::Number => v
                .as_i64()
                .map(PyArg::Int)
                .ok_or_else(|| format!("argument {v} is not an integer")),
            ValueKind::Seq => {
                let mut out = Vec::new();
                for item in v.try_iter().map_err(|e| e.to_string())? {
                    out.push(
                        item.as_str()
                            .ok_or_else(|| {
                                format!("sequence argument contains {item}, not a string")
                            })?
                            .to_string(),
                    );
                }
                Ok(PyArg::StrList(out))
            }
            other => Err(format!("unsupported argument of type {other}")),
        }
    }
}

impl PyValue {
    fn into_value(self) -> Value {
        match self {
            PyValue::Str(s) => Value::from(s),
            PyValue::Int(i) => Value::from(i),
            PyValue::Bool(b) => Value::from(b),
            PyValue::List(v) => Value::from(v),
        }
    }
}

/// The character set argument of `strip`/`lstrip`/`rstrip`, or `None` for
/// Python's whitespace default.
fn py_char_set(args: &[PyArg], idx: usize) -> Result<Option<Vec<char>>, String> {
    match args.get(idx) {
        None | Some(PyArg::None_) => Ok(None),
        Some(PyArg::Str(s)) => Ok(Some(s.chars().collect())),
        Some(other) => Err(format!("expected a string, got {other:?}")),
    }
}

fn py_trim(recv: &str, set: Option<&[char]>, left: bool, right: bool) -> String {
    let matches = |c: char| match set {
        Some(set) => set.contains(&c),
        None => c.is_whitespace(),
    };
    let mut out = recv;
    if left {
        out = out.trim_start_matches(matches);
    }
    if right {
        out = out.trim_end_matches(matches);
    }
    out.to_string()
}

/// CPython `str.split` / `str.rsplit`.
fn py_split(recv: &str, args: &[PyArg], from_right: bool) -> Result<PyValue, String> {
    let sep = match args.first() {
        None | Some(PyArg::None_) => None,
        Some(PyArg::Str(s)) => Some(s.as_str()),
        Some(other) => return Err(format!("separator must be a string, got {other:?}")),
    };
    let maxsplit = match args.get(1) {
        None => -1,
        Some(PyArg::Int(i)) => *i,
        Some(other) => return Err(format!("maxsplit must be an integer, got {other:?}")),
    };
    let Some(sep) = sep else {
        // Whitespace mode: runs of whitespace separate, no empty elements, and
        // leading/trailing whitespace is ignored.
        if maxsplit == 0 {
            return Ok(PyValue::List(vec![py_trim(recv, None, true, true)]));
        }
        let mut rest = recv;
        let mut out = Vec::new();
        let mut splits: i64 = 0;
        loop {
            rest = rest.trim_start();
            if rest.is_empty() {
                break;
            }
            if maxsplit >= 0 && splits == maxsplit {
                out.push(rest.to_string());
                break;
            }
            match rest.find(char::is_whitespace) {
                Some(i) => {
                    out.push(rest[..i].to_string());
                    rest = &rest[i..];
                    splits += 1;
                }
                None => {
                    out.push(rest.to_string());
                    break;
                }
            }
        }
        return Ok(PyValue::List(out));
    };
    if sep.is_empty() {
        return Err("empty separator".to_string());
    }
    let parts: Vec<String> = if maxsplit < 0 {
        recv.split(sep).map(str::to_string).collect()
    } else if maxsplit == 0 {
        vec![recv.to_string()]
    } else {
        let n = maxsplit as usize + 1;
        if from_right {
            let mut v: Vec<String> = recv.rsplitn(n, sep).map(str::to_string).collect();
            v.reverse();
            v
        } else {
            recv.splitn(n, sep).map(str::to_string).collect()
        }
    };
    Ok(PyValue::List(parts))
}

/// CPython `str.title`: a letter is upper-cased when it follows a non-cased
/// character, lower-cased otherwise.
fn py_title(recv: &str) -> String {
    let mut out = String::with_capacity(recv.len());
    let mut prev_cased = false;
    for c in recv.chars() {
        if prev_cased {
            out.extend(c.to_lowercase());
        } else {
            out.extend(c.to_uppercase());
        }
        prev_cased = c.is_alphanumeric();
    }
    out
}

/// CPython `str.find`/`str.rfind`: the character index, not the byte offset.
fn py_index(recv: &str, sub: &str, from_right: bool) -> Option<i64> {
    let idx = if from_right {
        recv.rfind(sub)
    } else {
        recv.find(sub)
    };
    idx.map(|i| recv[..i].chars().count() as i64)
}

/// A Python `str` method with CPython semantics.
///
/// `None` means "not a method this engine implements" (the caller turns that
/// into a refusal); `Err(..)` is a Python-level argument error.
fn py_str_method(name: &str, recv: &str, args: &[PyArg]) -> Option<Result<PyValue, String>> {
    match name {
        "strip" | "lstrip" | "rstrip" | "split" | "rsplit" | "startswith" | "endswith"
        | "replace" | "lower" | "upper" | "title" | "capitalize" | "join" | "find" | "rfind"
        | "count" => {}
        _ => return None,
    }
    Some(impl_str_method(name, recv, args))
}

/// The supported subset's implementation (the name is checked by the caller).
fn impl_str_method(name: &str, recv: &str, args: &[PyArg]) -> Result<PyValue, String> {
    let no_args = |args: &[PyArg]| -> Result<(), String> {
        if args.is_empty() {
            Ok(())
        } else {
            Err(format!("{name}() takes no arguments"))
        }
    };
    let sub_arg = |args: &[PyArg]| -> Result<String, String> {
        match args.first() {
            Some(PyArg::Str(s)) => Ok(s.clone()),
            Some(other) => Err(format!("expected a string argument, got {other:?}")),
            None => Err(format!("{name}() needs an argument")),
        }
    };
    match name {
        "strip" | "lstrip" | "rstrip" => {
            let set = py_char_set(args, 0)?;
            let left = name != "rstrip";
            let right = name != "lstrip";
            Ok(PyValue::Str(py_trim(recv, set.as_deref(), left, right)))
        }
        "split" => py_split(recv, args, false),
        "rsplit" => py_split(recv, args, true),
        "startswith" | "endswith" => {
            let prefixes: Vec<String> = match args.first() {
                Some(PyArg::Str(s)) => vec![s.clone()],
                Some(PyArg::StrList(v)) => v.clone(),
                Some(other) => return Err(format!("expected a string or a tuple, got {other:?}")),
                None => return Err(format!("{name}() needs an argument")),
            };
            if args.len() > 1 {
                return Err(format!(
                    "{name}(start[, end]) is not implemented; only the prefix form is"
                ));
            }
            let hit = prefixes.iter().any(|p| {
                if name == "startswith" {
                    recv.starts_with(p)
                } else {
                    recv.ends_with(p)
                }
            });
            Ok(PyValue::Bool(hit))
        }
        "replace" => {
            if args.len() < 2 {
                return Err("replace() needs old and new".to_string());
            }
            let old = sub_arg(&args[0..1])?;
            let new = sub_arg(&args[1..2])?;
            let count = match args.get(2) {
                None => -1,
                Some(PyArg::Int(i)) => *i,
                Some(other) => return Err(format!("count must be an integer, got {other:?}")),
            };
            if count < 0 {
                Ok(PyValue::Str(recv.replace(&old, &new)))
            } else {
                Ok(PyValue::Str(recv.replacen(&old, &new, count as usize)))
            }
        }
        "lower" => {
            no_args(args)?;
            Ok(PyValue::Str(recv.to_lowercase()))
        }
        "upper" => {
            no_args(args)?;
            Ok(PyValue::Str(recv.to_uppercase()))
        }
        "title" => {
            no_args(args)?;
            Ok(PyValue::Str(py_title(recv)))
        }
        "capitalize" => {
            no_args(args)?;
            let mut chars = recv.chars();
            match chars.next() {
                Some(first) => {
                    let mut s: String = first.to_uppercase().collect();
                    s.extend(chars.flat_map(char::to_lowercase));
                    Ok(PyValue::Str(s))
                }
                None => Ok(PyValue::Str(String::new())),
            }
        }
        "join" => {
            let items = match args.first() {
                Some(PyArg::StrList(v)) => v.clone(),
                // CPython joins a *string* argument character by character.
                Some(PyArg::Str(s)) => s.chars().map(|c| c.to_string()).collect(),
                Some(other) => return Err(format!("join() expects a sequence, got {other:?}")),
                None => return Err("join() needs an argument".to_string()),
            };
            Ok(PyValue::Str(items.join(recv)))
        }
        "find" | "rfind" => {
            let sub = sub_arg(args)?;
            if args.len() > 1 {
                return Err(format!("{name}(sub, start[, end]) is not implemented"));
            }
            Ok(PyValue::Int(
                py_index(recv, &sub, name == "rfind").unwrap_or(-1),
            ))
        }
        "count" => {
            let sub = sub_arg(args)?;
            if sub.is_empty() {
                return Ok(PyValue::Int(recv.chars().count() as i64 + 1));
            }
            Ok(PyValue::Int(recv.matches(&sub).count() as i64))
        }
        other => unreachable!("unarity-checked method {other}"),
    }
}

/// The unknown-method hook: Python string methods on string values, everything
/// else refused by name.
fn unknown_method(
    _state: &State,
    value: &Value,
    method: &str,
    args: &[Value],
) -> Result<Value, Error> {
    if value.kind() != ValueKind::String {
        return Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("{} has no method named {method}", value.kind()),
        ));
    }
    let recv = value.as_str().unwrap_or_default();
    let py_args = match args
        .iter()
        .map(PyArg::from_value)
        .collect::<Result<Vec<_>, _>>()
    {
        Ok(a) => a,
        Err(msg) => return Err(Error::new(ErrorKind::InvalidOperation, msg)),
    };
    match py_str_method(method, recv, &py_args) {
        Some(Ok(v)) => Ok(v.into_value()),
        Some(Err(msg)) => Err(Error::new(ErrorKind::InvalidOperation, msg)),
        None => Err(Error::new(
            ErrorKind::UnknownMethod,
            format!("unsupported Python str method `{method}`"),
        )),
    }
}

/// The engine's template environment. Every render goes through this so the
/// Python-compatibility hook cannot be forgotten, and `raise_exception` (used
/// by many templates for their own refusals) surfaces as a real error.
fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        // A template's deliberate refusal is not a fallback trigger either.
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    env
}

/// Compile the template and render a canary conversation through it.
///
/// This is the load-time gate: the CLI and `serve` call it once so an
/// unrenderable template is refused *before* any request, with the construct
/// named. The canary includes a `<think>` assistant turn because Qwen3's
/// `str.split`/`lstrip`/`rstrip` calls only execute on that branch.
pub fn validate(template: &str) -> Result<(), TemplateError> {
    let canary: Vec<(String, Option<String>)> = vec![
        ("system".into(), Some("You are a helpful assistant.".into())),
        ("user".into(), Some("hi".into())),
        (
            "assistant".into(),
            Some("<think>\nchecking\n</think>\n\nhello".into()),
        ),
        ("user".into(), Some("again".into())),
    ];
    render_messages(template, &canary, true, "")?;
    // Qwen3's `str.split`/`lstrip`/`rstrip` only execute when an assistant turn
    // is replayed *after* the last user query, so the canary covers that too.
    let replay: Vec<(String, Option<String>)> = vec![
        ("user".into(), Some("hi".into())),
        (
            "assistant".into(),
            Some("<think>\nchecking\n</think>\n\nhello".into()),
        ),
    ];
    render_messages(template, &replay, false, "").map(|_| ())
}

/// Render a chat template with a full message list (server path,
/// OPENAI-CHAT-API-PLAN.md). `messages` are `(role, content)` pairs.
///
/// Returns a refusal — never a generic prompt — when the template cannot be
/// compiled or rendered. `add_generation_prompt` appends the assistant header
/// when the template supports it; `bos_token` is exposed to the template
/// context (and `tools`, matching transformers, is `none`).
pub fn render_messages(
    template: &str,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<String, TemplateError> {
    let mut env = environment();
    if let Err(e) = env.add_template("chat", template) {
        return Err(TemplateError::classify(&e));
    }
    let tmpl = env
        .get_template("chat")
        .map_err(|e| TemplateError::classify(&e))?;

    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| {
            serde_json::json!({
                "role": role,
                "content": content,
            })
        })
        .collect();

    tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        // transformers passes `tools=None`; `none` keeps `{% if tools %}` and
        // `tools is defined` behaving the way jinja2 does.
        tools => Value::from(()),
    })
    .map_err(|e| TemplateError::classify(&e))
}

/// Render with an optional template: `None` (the GGUF carries no
/// `tokenizer.chat_template` at all) uses the ChatML fallback; `Some(t)` renders
/// the model's own template and propagates a refusal.
pub fn render_messages_opt(
    template: Option<&str>,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<String, TemplateError> {
    match template {
        Some(t) => render_messages(t, messages, add_generation_prompt, bos_token),
        None => Ok(fallback_chatml_messages(messages, add_generation_prompt)),
    }
}

/// Diff rendering result (the minfer version of `common_chat_format_single`).
pub struct FormattedDelta {
    /// Appended text: the part of `fmt_new` after stripping the `fmt_past` prefix (including trailing-newline compensation).
    pub text: String,
    /// Whether `fmt_new` starts with the `fmt_past` prefix; false means the template is
    /// non-deterministic (e.g. depends on external state) → caller must fall back to a full re-render (§5.4).
    pub prefix_matched: bool,
}

/// Incremental rendering: given the recorded history and the new message, return the text delta that only needs to be appended to KV.
///
/// Mirrors llama.cpp `common_chat_format_single` (common/chat.cpp:653): renders
/// twice (with/without the new message) and diffs, avoiding re-feeding generated
/// history to the model. If `fmt_past` ends with `\n`, prepend `\n` — that newline
/// is the prefix tail, eaten by the diff, but not emitted after the model's EOG; canonical text needs it (§3.2/§5.4).
///
/// When `template` is None, render with the ChatML fallback (`fallback_chatml_messages`).
pub fn format_single(
    template: Option<&str>,
    messages: &[(String, Option<String>)],
    new_msg: (String, Option<String>),
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<FormattedDelta, TemplateError> {
    let render = |msgs: &[(String, Option<String>)], add_gen: bool| {
        render_messages_opt(template, msgs, add_gen, bos_token)
    };
    let fmt_past = if messages.is_empty() {
        String::new()
    } else {
        render(messages, false)?
    };
    let mut all = messages.to_vec();
    all.push(new_msg);
    let fmt_new = render(&all, add_generation_prompt)?;

    let mut out = String::new();
    // Trailing-newline compensation: when fmt_past ends with '\n', the newline is
    // the prefix tail, eaten by the diff, but required by canonical text (after EOG, before next message).
    if add_generation_prompt && !fmt_past.is_empty() && fmt_past.ends_with('\n') {
        out.push('\n');
    }
    if fmt_new.starts_with(&fmt_past) {
        out.push_str(&fmt_new[fmt_past.len()..]);
        Ok(FormattedDelta {
            text: out,
            prefix_matched: true,
        })
    } else {
        // Prefix mismatch: non-deterministic template → return the full text; the caller falls back to a full re-render.
        Ok(FormattedDelta {
            text: fmt_new,
            prefix_matched: false,
        })
    }
}

/// Render a chat template with minijinja (single user message, CLI path).
/// Refuses loudly if the template cannot be rendered.
pub fn render_template(
    template: &str,
    user_input: &str,
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<String, TemplateError> {
    let messages = vec![("user".to_string(), Some(user_input.to_string()))];
    render_messages(template, &messages, add_generation_prompt, bos_token)
}

/// Fallback: simple ChatML format over ALL messages (server path). Every
/// message keeps its role/content; null content (assistant tool-call turn)
/// emits the role marker only.
///
/// Used **only** for a GGUF with no `tokenizer.chat_template`: there is no
/// model-specific format to lose, and the load path prints a notice.
pub(crate) fn fallback_chatml_messages(
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
) -> String {
    let mut r = String::new();
    for (role, content) in messages {
        match content {
            Some(c) => r.push_str(&format!("<|im_start|>{}\n{}<|im_end|>\n", role, c)),
            None => r.push_str(&format!("<|im_start|>{}\n<|im_end|>\n", role)),
        }
    }
    if add_generation_prompt {
        r.push_str("<|im_start|>assistant\n");
    }
    r
}

#[cfg(test)]
mod tests;
