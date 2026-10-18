// Chat template rendering using minijinja
// Reads tokenizer.chat_template from GGUF, renders with message context
//
// Design record: docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md (F7, #50).
//
// Three contracts matter here:
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
//  3. **One resolver decides which template is used** (`resolve_template`): a
//     `--chat-template <FILE>` override beats the GGUF's copy, and whichever
//     wins is validated there, so no entry point can render text another one
//     refused. Time reaches a template through one injected clock, never a
//     wall-clock read inside a render (ADR-0057).

use minijinja::value::{Value, ValueKind};
use minijinja::{context, Environment, Error, ErrorKind, State};
use std::fmt;
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

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
    "ljust",
    "lstrip",
    "replace",
    "rfind",
    "rjust",
    "rsplit",
    "rstrip",
    "split",
    "splitlines",
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

    /// A `--chat-template <FILE>` that cannot be used: unreadable, empty, or
    /// (through `classify`) unrenderable. The path is part of the text because
    /// the user named it and nothing else identifies which file failed.
    fn file(path: &Path, detail: impl Into<String>) -> Self {
        Self {
            kind: "template file",
            detail: format!("{}: {}", path.display(), detail.into()),
            line: None,
        }
    }

    /// The full, user-facing refusal. Names the construct, the template line
    /// and the engine's alternative, so nothing is silent.
    pub fn message(&self) -> String {
        if self.kind == "template file" {
            return format!("chat template error — {}: {}", self.kind, self.detail);
        }
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

/// CPython `str.splitlines`: split at every line boundary, and a trailing
/// boundary does not produce a trailing empty element.
///
/// The boundary set is CPython's, not `char::is_whitespace`: `\n`, `\r`,
/// `\r\n`, `\v`, `\f`, `\x1c`, `\x1d`, `\x1e`, `\x85`, `\u{2028}`, `\u{2029}`.
fn py_splitlines(recv: &str, keepends: bool) -> Vec<String> {
    let is_boundary = |c: char| {
        matches!(
            c,
            '\n' | '\r'
                | '\u{0b}'
                | '\u{0c}'
                | '\u{1c}'
                | '\u{1d}'
                | '\u{1e}'
                | '\u{85}'
                | '\u{2028}'
                | '\u{2029}'
        )
    };
    let mut out = Vec::new();
    let mut chars = recv.char_indices().peekable();
    let mut start = 0;
    while let Some((i, c)) = chars.next() {
        if !is_boundary(c) {
            continue;
        }
        let mut end = i + c.len_utf8();
        if c == '\r' {
            // `\r\n` is one boundary, not two: consume the `\n`.
            if let Some(&(j, '\n')) = chars.peek() {
                chars.next();
                end = j + 1;
            }
        }
        out.push(if keepends {
            recv[start..end].to_string()
        } else {
            recv[start..i].to_string()
        });
        start = end;
    }
    if start < recv.len() {
        out.push(recv[start..].to_string());
    }
    out
}

/// CPython `str.ljust` / `str.rjust`: pad to `width` with one fill character.
fn py_justify(recv: &str, args: &[PyArg], pad_right: bool) -> Result<PyValue, String> {
    let width = match args.first() {
        Some(PyArg::Int(i)) => *i,
        Some(other) => return Err(format!("width must be an integer, got {other:?}")),
        None => return Err("a width is required".to_string()),
    };
    let fill = match args.get(1) {
        None => ' ',
        Some(PyArg::Str(s)) => {
            let mut cs = s.chars();
            match (cs.next(), cs.next()) {
                (Some(c), None) => c,
                _ => {
                    return Err(
                        "the fill character must be a string of exactly one character".to_string(),
                    )
                }
            }
        }
        Some(other) => {
            return Err(format!(
                "the fill character must be a string, got {other:?}"
            ))
        }
    };
    let len = recv.chars().count() as i64;
    if width <= len {
        return Ok(PyValue::Str(recv.to_string()));
    }
    let pad: String = std::iter::repeat(fill)
        .take((width - len) as usize)
        .collect();
    Ok(PyValue::Str(if pad_right {
        format!("{recv}{pad}")
    } else {
        format!("{pad}{recv}")
    }))
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
        | "count" | "splitlines" | "ljust" | "rjust" => {}
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
        "splitlines" => {
            let keepends = match args.first() {
                None | Some(PyArg::None_) => false,
                Some(PyArg::Int(i)) => *i != 0,
                Some(other) => return Err(format!("keepends must be a boolean, got {other:?}")),
            };
            if args.len() > 1 {
                return Err("splitlines() takes at most one argument".to_string());
            }
            Ok(PyValue::List(py_splitlines(recv, keepends)))
        }
        "ljust" => py_justify(recv, args, true),
        "rjust" => py_justify(recv, args, false),
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

/// A UTC civil timestamp — the value a template's `strftime_now` formats.
///
/// It is *injected* rather than read inside a render, so a render is
/// reproducible in a test and an unknown directive cannot silently become a
/// literal (ADR-0057).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CivilTime {
    year: i32,
    month: u32,
    day: u32,
    hour: u32,
    minute: u32,
    second: u32,
}

/// Unix seconds → UTC civil date/time (Howard Hinnant's `civil_from_days`,
/// proleptic Gregorian, valid far past any timestamp a chat template prints).
pub(crate) fn civil_from_unix(secs: i64) -> CivilTime {
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let hour = (rem / 3_600) as u32;
    let minute = ((rem % 3_600) / 60) as u32;
    let second = (rem % 60) as u32;

    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097); // [0, 146096]
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11], March-based
    let day = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let month = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let year = (yoe + era * 400 + i64::from(month <= 2)) as i32;

    CivilTime {
        year,
        month,
        day,
        hour,
        minute,
        second,
    }
}

/// The production clock: UTC now, second resolution.
fn utc_now() -> CivilTime {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    civil_from_unix(secs)
}

/// The `strftime` directive subset `strftime_now` implements. The full CPython
/// set is deliberately not claimed: an unknown directive is refused by name, the
/// same policy the `str` methods follow.
const SUPPORTED_STRFTIME: &str = "%Y %y %m %d %H %M %S %F %T %%";

pub(crate) fn format_civil(t: CivilTime, fmt: &str) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = fmt.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        let Some(d) = chars.next() else {
            return Err("a format may not end with a bare `%`".to_string());
        };
        match d {
            'Y' => out.push_str(&format!("{:04}", t.year)),
            'y' => out.push_str(&format!("{:02}", t.year.rem_euclid(100))),
            'm' => out.push_str(&format!("{:02}", t.month)),
            'd' => out.push_str(&format!("{:02}", t.day)),
            'H' => out.push_str(&format!("{:02}", t.hour)),
            'M' => out.push_str(&format!("{:02}", t.minute)),
            'S' => out.push_str(&format!("{:02}", t.second)),
            'F' => out.push_str(&format!("{:04}-{:02}-{:02}", t.year, t.month, t.day)),
            'T' => out.push_str(&format!("{:02}:{:02}:{:02}", t.hour, t.minute, t.second)),
            '%' => out.push('%'),
            other => {
                return Err(format!(
                    "unsupported strftime_now directive `%{other}`; supported: {SUPPORTED_STRFTIME}"
                ))
            }
        }
    }
    Ok(out)
}

/// The environment with an explicit clock: the only way time enters a render.
fn environment_with_clock(clock: fn() -> CivilTime) -> Environment<'static> {
    let mut env = Environment::new();
    env.set_unknown_method_callback(unknown_method);
    env.add_function("raise_exception", |msg: String| -> Result<Value, Error> {
        // A template's deliberate refusal is not a fallback trigger either.
        Err(Error::new(ErrorKind::InvalidOperation, msg))
    });
    env.add_function(
        "strftime_now",
        move |fmt: String| -> Result<String, Error> {
            format_civil(clock(), &fmt).map_err(|m| Error::new(ErrorKind::InvalidOperation, m))
        },
    );
    env
}

/// The single place that decides which template a run uses.
///
/// A `--chat-template <FILE>` override wins over the GGUF's copy; whichever is
/// chosen is validated here, so every entry point refuses the same text at the
/// same moment. `Ok(None)` is the documented ChatML case (no template anywhere)
/// — the caller announces it, it is never silent.
pub fn resolve_template(
    embedded: Option<&str>,
    file: Option<&Path>,
) -> Result<Option<String>, TemplateError> {
    let chosen = match file {
        Some(path) => {
            let text = std::fs::read_to_string(path)
                .map_err(|e| TemplateError::file(path, e.to_string()))?;
            if text.trim().is_empty() {
                return Err(TemplateError::file(path, "the template file is empty"));
            }
            Some(text)
        }
        None => embedded.map(str::to_string),
    };
    if let Some(text) = chosen.as_deref() {
        if let Err(e) = validate(text) {
            // A file's refusal names the file; an embedded one already names the
            // model's template, so it is passed through unchanged.
            return Err(match file {
                Some(path) => {
                    let at = e
                        .line
                        .map(|n| format!(" (template line {n})"))
                        .unwrap_or_default();
                    TemplateError::file(path, format!("{}{}", e.detail, at))
                }
                None => e,
            });
        }
    }
    Ok(chosen)
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
    render_messages_with_clock(
        utc_now,
        template,
        messages,
        add_generation_prompt,
        bos_token,
    )
}

/// `render_messages` with an explicit clock (the injection point): the
/// environment is built from the clock, so nothing inside a render reads time.
fn render_messages_with_clock(
    clock: fn() -> CivilTime,
    template: &str,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<String, TemplateError> {
    let mut env = environment_with_clock(clock);
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
    render_messages_opt_in(
        utc_now,
        template,
        messages,
        add_generation_prompt,
        bos_token,
    )
}

/// `render_messages_opt` with an explicit clock, so a render is reproducible.
fn render_messages_opt_in(
    clock: fn() -> CivilTime,
    template: Option<&str>,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<String, TemplateError> {
    match template {
        Some(t) => render_messages_with_clock(clock, t, messages, add_generation_prompt, bos_token),
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
    format_single_clocked(
        utc_now,
        template,
        messages,
        new_msg,
        add_generation_prompt,
        bos_token,
    )
}

/// `format_single` with an explicit clock: a clock-dependent template renders
/// differently on the two passes, so the diff reports `prefix_matched = false`
/// and the caller does a full re-render (§5.4).
fn format_single_clocked(
    clock: fn() -> CivilTime,
    template: Option<&str>,
    messages: &[(String, Option<String>)],
    new_msg: (String, Option<String>),
    add_generation_prompt: bool,
    bos_token: &str,
) -> Result<FormattedDelta, TemplateError> {
    let render = |msgs: &[(String, Option<String>)], add_gen: bool| {
        render_messages_opt_in(clock, template, msgs, add_gen, bos_token)
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

/// The single-turn CLI prompt, plus whether the caller must announce the
/// no-template case.
pub struct SingleTurnPrompt {
    pub text: String,
    /// True when neither the GGUF nor a file supplied a template: the caller
    /// prints the one-line ChatML notice.
    pub chatml_notice: bool,
}

/// The single-turn prompt: `--chat-template <FILE>`, else the GGUF's copy, else
/// the announced ChatML fallback.
///
/// This is the **only** place the single-turn path chooses between a rendered
/// template and ChatML, and the notice flag is returned with the text it
/// describes — so a notice can never claim a prompt the render did not produce
/// (#596). `--no-template` bypasses this entirely (the caller keeps the raw
/// prompt) and is refused alongside `--chat-template`.
pub fn render_single_turn(
    embedded: Option<&str>,
    file: Option<&Path>,
    prompt: &str,
    bos_token: &str,
) -> Result<SingleTurnPrompt, TemplateError> {
    match resolve_template(embedded, file)? {
        Some(template) => Ok(SingleTurnPrompt {
            text: render_template(&template, prompt, true, bos_token)?,
            chatml_notice: false,
        }),
        None => {
            let messages = vec![("user".to_string(), Some(prompt.to_string()))];
            Ok(SingleTurnPrompt {
                text: render_messages_opt(None, &messages, true, bos_token)?,
                chatml_notice: true,
            })
        }
    }
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
