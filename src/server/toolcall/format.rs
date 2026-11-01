//! The tool-call format table (#699, ADR-0066): one row per chat-template
//! family.
//!
//! #614 shipped the `<tool_call>` format as three constants inside the parser.
//! Other families differ in ways a constant cannot carry: the literal that opens
//! a call, the field names inside the call object (Llama 3.1 writes
//! `parameters`, not `arguments`), whether calls repeat, and the wrapper a
//! `tool_choice` grammar needs. This module is the single place those facts
//! live; the parser and the grammar builder both read the same row, so what the
//! grammar produces is what the parser extracts (ADR-0063 decision 7).
//!
//! Detection is a literal scan of the chat template's source, not of a rendered
//! prompt: the template is the only per-model description the GGUF carries, and
//! it is the thing that decides which syntax the model was told to write. Jinja
//! comments are removed first — a format that appears only in a comment is
//! documentation, not something the template emits. Rows are ordered most
//! specific first, and the first row whose anchors all appear wins.
//!
//! Only the Qwen2/Qwen3 row exists today; the family rows land as data (ADR-0066
//! records the shape) rather than as parser branches.

use std::borrow::Cow;

/// One family's call syntax.
#[derive(Debug, Clone, Copy)]
pub struct ToolFormat {
    /// Stable name, used by logs, errors and tests.
    pub id: &'static str,
    /// Every literal must appear in the template's source (comments removed)
    /// for this row to be picked.
    pub anchors: &'static [&'static str],
    /// The literal that opens a call block.
    pub open: &'static str,
    /// The literal that closes a call block.
    pub close: &'static str,
    /// The call object's name field.
    pub name_key: &'static str,
    /// The call object's arguments field.
    pub args_key: &'static str,
    /// The `tool_choice` grammar shape, or `None` when the format is recognised
    /// but cannot be constrained yet. A `None` row is a named `400` for
    /// `tool_choice`, never an unconstrained generation.
    pub grammar: Option<GrammarShape>,
}

/// The literal pieces a `tool_choice` GBNF wraps around the name rule and the
/// schema compiler's `j-object`.
///
/// `open` ends right before the name, `between` sits between the name and the
/// arguments, and `close` follows the arguments. The Qwen row's literals include
/// the newline and the key outside the object, because that is the block the
/// parser extracts.
#[derive(Debug, Clone, Copy)]
pub struct GrammarShape {
    pub open: &'static str,
    pub between: &'static str,
    pub close: &'static str,
}

/// Qwen2/Qwen3 (and Hermes): the format #614 shipped, unchanged.
pub static QWEN: ToolFormat = ToolFormat {
    id: "qwen2",
    anchors: &["<tool_call>"],
    open: "<tool_call>",
    close: "</tool_call>",
    name_key: "name",
    args_key: "arguments",
    grammar: Some(GrammarShape {
        open: "<tool_call>\n{\"name\": \"",
        between: "\", \"arguments\": ",
        close: "}\n</tool_call>",
    }),
};

/// The table, most specific first.
static FORMATS: &[&ToolFormat] = &[&QWEN];

/// Pick the row whose anchors all appear in the template's source.
///
/// An unrecognised template answers `None`; parsing then falls back to
/// [`QWEN`] (the pre-table behaviour), while `tool_choice` refuses by name.
pub fn detect(template: &str) -> Option<&'static ToolFormat> {
    let source = without_jinja_comments(template);
    FORMATS
        .iter()
        .copied()
        .find(|format| format.anchors.iter().all(|anchor| source.contains(anchor)))
}

/// `template` with every `{# ... #}` removed.
///
/// An unterminated comment drops the rest: the template compiler refuses such a
/// template at startup, so this is only the defensive arm.
fn without_jinja_comments(template: &str) -> Cow<'_, str> {
    if !template.contains("{#") {
        return Cow::Borrowed(template);
    }
    let mut out = String::with_capacity(template.len());
    let mut rest = template;
    while let Some(start) = rest.find("{#") {
        out.push_str(&rest[..start]);
        let after = &rest[start + 2..];
        let Some(end) = after.find("#}") else {
            return Cow::Owned(out);
        };
        rest = &after[end + 2..];
    }
    out.push_str(rest);
    Cow::Owned(out)
}

#[cfg(test)]
mod tests;
