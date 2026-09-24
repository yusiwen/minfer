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

    /// Fixture comparison form (the JSON encoding of the CPython result).
    fn to_json(self) -> serde_json::Value {
        match self {
            PyValue::Str(s) => serde_json::Value::String(s),
            PyValue::Int(i) => serde_json::Value::from(i),
            PyValue::Bool(b) => serde_json::Value::from(b),
            PyValue::List(v) => {
                serde_json::Value::Array(v.into_iter().map(serde_json::Value::String).collect())
            }
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
mod tests {
    use super::*;

    fn msgs() -> Vec<(String, Option<String>)> {
        vec![
            ("system".into(), Some("You are helpful.".into())),
            ("user".into(), Some("Hi".into())),
            ("assistant".into(), Some("Hello!".into())),
        ]
    }

    fn m(role: &str, content: &str) -> (String, Option<String>) {
        (role.to_string(), Some(content.to_string()))
    }

    #[test]
    fn render_messages_with_jinja_template() {
        let tmpl = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
        let out = render_messages(tmpl, &msgs(), true, "<|endoftext|>").unwrap();
        assert_eq!(
            out,
            "<system>You are helpful.</system><user>Hi</user><assistant>Hello!</assistant><assistant>"
        );
    }

    #[test]
    fn render_messages_null_content_emits_role_only() {
        // null content must not panic and must preserve the role marker.
        // (How the template renders a null is template-dependent — the contract
        // is that all messages are passed through.)
        let tmpl = "{% for m in messages %}[{{ m['role'] }}:{{ m['content'] }}]{% endfor %}";
        let mut msgs = msgs();
        msgs.push(("assistant".into(), None)); // tool-call turn
        let out = render_messages(tmpl, &msgs, false, "").unwrap();
        assert!(out.starts_with("[system:You are helpful.]"), "got: {out}");
        assert!(out.contains("[user:Hi]"), "got: {out}");
        assert!(out.contains("[assistant:Hello!]"), "got: {out}");
        assert!(
            out.contains("[assistant:"),
            "null-content role marker present: {out}"
        );
    }

    // === The loud-refusal contract (F7 #50, design §2.3) ===
    //
    // Before F7 both of these rendered a generic ChatML prompt and only printed
    // a warning. The gate is that they are errors which *name* the construct;
    // the mutation check (removing the unknown-method hook) makes the second
    // test fail, and reverting makes it pass.

    #[test]
    fn invalid_template_is_a_loud_error() {
        let err = render_messages("{{ bad", &msgs(), true, "").unwrap_err();
        let msg = err.message();
        assert_eq!(err.kind, "syntax error");
        assert!(
            msg.contains("refuses to fall back"),
            "refusal must say so: {msg}"
        );
        assert!(
            !msg.contains("<|im_start|>"),
            "the refusal must not contain a rendered prompt: {msg}"
        );
        eprintln!("refusal (syntax error):\n{msg}");
    }

    #[test]
    fn unsupported_str_method_is_a_loud_error_naming_the_construct() {
        let tmpl = "{% for m in messages %}{{ m['content'].splitlines() }}{% endfor %}";
        let err = render_messages(tmpl, &msgs(), false, "").unwrap_err();
        let msg = err.message();
        assert!(
            msg.contains("splitlines"),
            "the refusal must name the construct: {msg}"
        );
        assert!(
            msg.contains("refuses to fall back"),
            "refusal must say so: {msg}"
        );
        assert_eq!(
            err.line,
            Some(1),
            "the template line must be reported: {msg}"
        );
        // Shown by `cargo test -- --nocapture`; this is the user-facing text.
        eprintln!("refusal (unsupported method):\n{msg}");
    }

    #[test]
    fn template_raise_exception_is_a_refusal_not_a_fallback() {
        let tmpl = "{{ raise_exception('this template refuses a null system prompt') }}";
        let err = render_messages(tmpl, &msgs(), false, "").unwrap_err();
        assert!(
            err.message().contains("refuses a null system prompt"),
            "the template's own refusal text must survive: {}",
            err.message()
        );
    }

    #[test]
    fn validate_refuses_an_unrenderable_template_before_any_request() {
        assert!(validate("{{ messages[0]['content'] }}").is_ok());
        let err = validate("{% for m in messages %}{{ m['content'].splitlines() }}{% endfor %}")
            .unwrap_err();
        assert!(err.message().contains("splitlines"), "{}", err.message());
    }

    #[test]
    fn chatml_fallback_applies_only_when_there_is_no_template() {
        // Some(t) that cannot render -> refusal (tested above); None -> ChatML.
        let out = render_messages_opt(None, &msgs(), true, "").unwrap();
        assert_eq!(
            out,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
        // A null content keeps the role marker and is not stringified.
        let mut msgs = msgs();
        msgs.push(("assistant".into(), None));
        let out = render_messages_opt(None, &msgs, false, "").unwrap();
        assert!(out.contains("<|im_start|>assistant\n<|im_end|>"), "{out}");
        assert!(!out.contains("None"), "null must not be stringified: {out}");
    }

    #[test]
    fn render_template_single_user_unchanged() {
        // CLI path still renders a single user message via the template
        let tmpl =
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:";
        let out = render_template(tmpl, "hello", true, "").unwrap();
        assert_eq!(out, "user: hello\nassistant:");
    }

    // === format_single (incremental diff rendering, CLI-CONVERSATION-PLAN.md §5.3) ===

    /// Qwen2.5-style ChatML template (a newline after each message, ending with an assistant header).
    const QWEN_CHATML: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    #[test]
    fn format_single_diffs_only_new_user_message() {
        // History [system, user, assistant] + new user → delta contains only the new message
        // and the assistant header; it must not repeat the already-generated assistant content.
        let past = vec![
            m("system", "You are helpful."),
            m("user", "hi"),
            m("assistant", "Hello!"),
        ];
        let d = format_single(
            Some(QWEN_CHATML),
            &past,
            m("user", "what is 2+2?"),
            true,
            "",
        )
        .unwrap();
        assert!(
            d.prefix_matched,
            "prefix must match for a deterministic template"
        );
        assert_eq!(
            d.text,
            "\n<|im_start|>user\nwhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
        // Invariant: KV prefix + delta == fmt_new (canonical full rendering).
        // KV prefix = fmt_past minus the newline the template emits after the last message;
        // the model doesn't emit that newline after EOG, so compensation prepends it to delta.
        let fmt_past = render_messages(QWEN_CHATML, &past, false, "").unwrap();
        let kv_prefix = fmt_past.strip_suffix('\n').unwrap_or(&fmt_past);
        let mut all = past.clone();
        all.push(m("user", "what is 2+2?"));
        let fmt_new = render_messages(QWEN_CHATML, &all, true, "").unwrap();
        assert_eq!(format!("{kv_prefix}{}", d.text), fmt_new);
    }

    #[test]
    fn format_single_no_trailing_newline_no_compensation() {
        // Template produces no trailing '\n' → no compensation.
        let tmpl = "{% for mm in messages %}[{{ mm['role'] }}:{{ mm['content'] }}]{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
        let past = vec![m("user", "hi"), m("assistant", "Hello!")];
        let d = format_single(Some(tmpl), &past, m("user", "Q"), true, "").unwrap();
        assert!(d.prefix_matched);
        assert_eq!(d.text, "[user:Q]<assistant>");
    }

    #[test]
    fn format_single_empty_history_returns_full_new() {
        let d = format_single(Some(QWEN_CHATML), &[], m("user", "first"), true, "").unwrap();
        assert!(d.prefix_matched);
        let expect = render_messages(QWEN_CHATML, &[m("user", "first")], true, "").unwrap();
        assert_eq!(d.text, expect);
    }

    #[test]
    fn format_single_prefix_mismatch_falls_back_to_full() {
        // Non-deterministic template (reverse) → fmt_new doesn't start with fmt_past → full text + flag.
        let tmpl = "{% for mm in messages|reverse %}[{{ mm['role'] }}]{% endfor %}";
        let past = vec![m("user", "hi"), m("assistant", "Hello!")];
        let d = format_single(Some(tmpl), &past, m("user", "Q"), false, "").unwrap();
        assert!(!d.prefix_matched);
        let mut all = past.clone();
        all.push(m("user", "Q"));
        let expect = render_messages(tmpl, &all, false, "").unwrap();
        assert_eq!(d.text, expect, "mismatch must return the full re-render");
    }

    #[test]
    fn format_single_fallback_without_template() {
        // template=None → rendered via the ChatML fallback; same prefix/compensation semantics.
        let past = vec![m("user", "hi"), m("assistant", "Hello!")];
        let d = format_single(None, &past, m("user", "Q"), true, "").unwrap();
        assert!(d.prefix_matched);
        assert_eq!(
            d.text,
            "\n<|im_start|>user\nQ<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn format_single_null_content_history() {
        // An assistant message with content=None (tool-call turn) must not break the diff.
        let tmpl = "{% for mm in messages %}[{{ mm['role'] }}]{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
        let past = vec![
            m("user", "hi"),
            ("assistant".to_string(), None),
            m("user", "again"),
        ];
        let d = format_single(Some(tmpl), &past, m("assistant", "ok"), false, "").unwrap();
        assert!(d.prefix_matched);
        assert_eq!(d.text, "[assistant]");
    }

    // === Python str-method semantics, checked against CPython ================
    //
    // The expectation table is generated by CPython itself
    // (/tmp/f7-make-fixtures.py; see the fixture's provenance) so the engine is
    // never its own reference.

    const PYSTR_JSON: &str = include_str!("../tests/fixtures/chat/pystr_semantics.json");

    fn py_arg(v: &serde_json::Value) -> PyArg {
        match v {
            serde_json::Value::String(s) => PyArg::Str(s.clone()),
            serde_json::Value::Number(n) => PyArg::Int(n.as_i64().unwrap()),
            serde_json::Value::Null => PyArg::None_,
            serde_json::Value::Array(a) => {
                PyArg::StrList(a.iter().map(|x| x.as_str().unwrap().to_string()).collect())
            }
            other => panic!("unsupported fixture argument {other}"),
        }
    }

    #[test]
    fn python_str_methods_match_cpython() {
        let fx: serde_json::Value = serde_json::from_str(PYSTR_JSON).expect("pystr fixture");
        let entries = fx["entries"].as_array().expect("entries");
        assert!(entries.len() >= 24, "fixture shrank: {}", entries.len());
        for e in entries {
            let method = e["method"].as_str().unwrap();
            let recv = e["receiver"].as_str().unwrap();
            let args: Vec<PyArg> = e["args"].as_array().unwrap().iter().map(py_arg).collect();
            let got = py_str_method(method, recv, &args)
                .unwrap_or_else(|| panic!("{method} not implemented"))
                .unwrap_or_else(|e| panic!("{method}({recv:?}) errored: {e}"));
            assert_eq!(
                got.to_json(),
                e["expected"],
                "{recv:?}.{method}({args:?}) must match CPython"
            );
        }
    }

    #[test]
    fn unknown_str_method_is_reported_as_unsupported() {
        assert!(py_str_method("splitlines", "a\nb", &[]).is_none());
        assert!(py_str_method("encode", "x", &[]).is_none());
    }

    #[test]
    fn python_split_edge_cases() {
        // Empty separator is a Python ValueError, not a silent split.
        assert!(py_str_method("split", "abc", &[PyArg::Str(String::new())])
            .unwrap()
            .is_err());
        // maxsplit = 0 returns the whole string.
        assert_eq!(
            py_str_method("split", "a b c", &[PyArg::None_, PyArg::Int(0)])
                .unwrap()
                .unwrap()
                .to_json(),
            serde_json::json!(["a b c"])
        );
    }

    // === Reference renderings, per model (the ticket's first acceptance) =====
    //
    // Generated by transformers 5.17.0 from the model's own `chat_template` in
    // its `tokenizer_config.json` (see each fixture's provenance). The engine
    // renders the *same template text* here; the real-model gate in
    // conversation.rs additionally renders the template the GGUF artifact
    // carries and requires the same bytes.

    const CHAT_FIXTURES: &[(&str, &str)] = &[
        (
            "qwen2.5-0.5b-instruct",
            include_str!("../tests/fixtures/chat/qwen2.5-0.5b-instruct.json"),
        ),
        (
            "qwen2.5-7b-instruct",
            include_str!("../tests/fixtures/chat/qwen2.5-7b-instruct.json"),
        ),
        (
            "qwen2.5-14b-instruct",
            include_str!("../tests/fixtures/chat/qwen2.5-14b-instruct.json"),
        ),
        (
            "qwen3-0.6b",
            include_str!("../tests/fixtures/chat/qwen3-0.6b.json"),
        ),
    ];

    #[test]
    fn model_templates_render_byte_for_byte() {
        for (slug, raw) in CHAT_FIXTURES {
            let fx: serde_json::Value =
                serde_json::from_str(raw).unwrap_or_else(|e| panic!("{slug} fixture: {e}"));
            // Provenance must stay attached to the fixture.
            for key in ["model", "revision", "template_sha256", "template"] {
                assert!(!fx[key].is_null(), "{slug}: fixture lacks {key}");
            }
            let prov = &fx["provenance"];
            assert!(
                prov["reference"].as_str().unwrap().contains("transformers"),
                "{slug}: reference generator must be named"
            );
            assert!(
                prov["template_source"]
                    .as_str()
                    .unwrap()
                    .contains("tokenizer_config.json"),
                "{slug}: template source must be named"
            );
            assert!(!prov["command"].as_str().unwrap().is_empty());
            assert!(!prov["date"].as_str().unwrap().is_empty());

            let template = fx["template"].as_str().unwrap();
            let bos = fx["bos_token"].as_str().unwrap_or("");
            let cases = fx["cases"].as_array().unwrap();
            assert!(cases.len() >= 6, "{slug}: fixture shrank");
            let mut names = Vec::new();
            for case in cases {
                let name = case["name"].as_str().unwrap();
                names.push(name.to_string());
                let msgs: Vec<(String, Option<String>)> = case["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| {
                        (
                            m["role"].as_str().unwrap().to_string(),
                            m["content"].as_str().map(str::to_string),
                        )
                    })
                    .collect();
                let add_gen = case["add_generation_prompt"].as_bool().unwrap();
                let expected = case["expected"].as_str().unwrap();
                let got = render_messages(template, &msgs, add_gen, bos)
                    .unwrap_or_else(|e| panic!("{slug}/{name} refused: {}", e.message()));
                assert_eq!(
                    got, expected,
                    "{slug}/{name}: rendered prompt differs from the transformers reference\n\
                     got:      {got:?}\n\
                     expected: {expected:?}"
                );
            }
            // The ticket asks explicitly for a multi-turn conversation, a system
            // message and a generation prompt: require all three in every fixture.
            assert!(names.iter().any(|n| n == "multi_turn"), "{slug}: {names:?}");
            assert!(
                names.iter().any(|n| n == "system_and_user"),
                "{slug}: {names:?}"
            );
            assert!(
                names.iter().any(|n| n.contains("generation_prompt")),
                "{slug}: {names:?}"
            );
        }
    }

    #[test]
    fn qwen3_reasoning_turn_is_extracted_like_transformers() {
        // The construct that motivated F7: the template's own split/lstrip/rstrip
        // calls must produce transformers' output, not a ChatML prompt.
        let fx: serde_json::Value = serde_json::from_str(CHAT_FIXTURES[3].1).unwrap();
        let case = fx["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == "reasoning_assistant_replay")
            .expect("fixture case");
        let expected = case["expected"].as_str().unwrap();
        assert!(
            expected.contains("<think>"),
            "the reference must exercise the think block: {expected:?}"
        );
        let msgs: Vec<(String, Option<String>)> = case["messages"]
            .as_array()
            .unwrap()
            .iter()
            .map(|m| {
                (
                    m["role"].as_str().unwrap().to_string(),
                    m["content"].as_str().map(str::to_string),
                )
            })
            .collect();
        let got = render_messages(
            fx["template"].as_str().unwrap(),
            &msgs,
            case["add_generation_prompt"].as_bool().unwrap(),
            fx["bos_token"].as_str().unwrap_or(""),
        )
        .unwrap();
        assert_eq!(got, expected);
        assert!(
            expected.contains("\n  Let me count"),
            "the reference must show the normalized reasoning block: {expected:?}"
        );
    }

    // === The template the GGUF artifact carries (real-model gate) ===========
    //
    // The CI gate above renders the *published* `tokenizer_config.json`
    // template. This gate renders the copy inside the GGUF the engine actually
    // loads and requires the same bytes. For Qwen2.5 and Qwen3 those two texts
    // differ (the converter rewrote a tool-call string and, for Qwen3, an older
    // template revision with an equivalent `range(...)` loop instead of
    // `messages[::-1]`), so the assertion is on the rendered bytes, which is
    // what the ticket asks for.

    #[test]
    #[ignore = "requires the cached GGUF models (~/.cache/minfer/models)"]
    fn gguf_template_renders_like_the_reference() {
        for (slug, raw) in CHAT_FIXTURES {
            let fx: serde_json::Value = serde_json::from_str(raw).unwrap();
            let rel = fx["gguf"].as_str().unwrap_or_default();
            let Some(path) = expand_home(rel) else {
                eprintln!("{slug}: no gguf path in the fixture");
                continue;
            };
            if !path.exists() {
                eprintln!("{slug}: {} not cached; skipping", path.display());
                continue;
            }
            let gguf = crate::gguf::load_gguf_model(&path).expect("parse GGUF");
            let template = gguf.parts[0]
                .ctx
                .kv
                .iter()
                .find(|kv| kv.key == "tokenizer.chat_template")
                .map(|kv| kv.get_val_str(0).to_string())
                .unwrap_or_else(|| panic!("{slug}: the GGUF carries no chat template"));
            let bos = fx["bos_token"].as_str().unwrap_or("");
            let cases = fx["cases"].as_array().unwrap();
            for case in cases {
                let msgs: Vec<(String, Option<String>)> = case["messages"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|m| {
                        (
                            m["role"].as_str().unwrap().to_string(),
                            m["content"].as_str().map(str::to_string),
                        )
                    })
                    .collect();
                let expected = case["expected"].as_str().unwrap();
                let got = render_messages(
                    &template,
                    &msgs,
                    case["add_generation_prompt"].as_bool().unwrap(),
                    bos,
                )
                .unwrap_or_else(|e| panic!("{slug}: GGUF template refused: {}", e.message()));
                assert_eq!(
                    got,
                    expected,
                    "{slug}/{}: the GGUF's own template renders differently from the reference",
                    case["name"].as_str().unwrap()
                );
            }
            eprintln!(
                "{slug}: {} cases render byte for byte from the GGUF template",
                cases.len()
            );
        }
    }

    fn expand_home(path: &str) -> Option<std::path::PathBuf> {
        if path.is_empty() {
            return None;
        }
        match path.strip_prefix("~/") {
            Some(rest) => Some(std::path::PathBuf::from(std::env::var_os("HOME")?).join(rest)),
            None => Some(std::path::PathBuf::from(path)),
        }
    }
}
