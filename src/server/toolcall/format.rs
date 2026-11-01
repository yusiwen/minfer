//! The tool-call format table (#699, ADR-0066): one row per chat-template
//! family.
//!
//! #614 shipped the `<tool_call>` format as three constants inside the parser.
//! Other families differ in ways a constant cannot carry: the literal that opens
//! a call, the field names inside the call object (Llama 3.1 writes
//! `parameters`, not `arguments`), whether a block holds one call or an array of
//! them, and the wrapper a `tool_choice` grammar needs. This module is the single
//! place those facts live; the parser and the grammar builder both read the same
//! row, so what the grammar produces is what the parser extracts (ADR-0063
//! decision 7).
//!
//! Detection is a literal scan of the chat template's source, not of a rendered
//! prompt: the template is the only per-model description the GGUF carries, and
//! it is the thing that decides which syntax the model was told to write. Jinja
//! comments are removed first — a format that appears only in a comment is
//! documentation, not something the template emits. Rows are ordered most
//! specific first, and the first row whose anchors all appear (and none of whose
//! exclusions appear) wins.

use std::borrow::Cow;

/// What a block's body is, and what ends it.
#[derive(Debug, Clone, Copy)]
pub enum BlockShape {
    /// One JSON object, then this literal.
    ObjectThenClose(&'static str),
    /// One JSON array of call objects; the generation's end (EOS) closes it.
    ArrayAtEos,
    /// One JSON object per call; the next open, an optional `separator`, or EOS
    /// closes it.
    ObjectUntilNextOpen { separator: &'static str },
    /// One call per block, with the name in the open anchor: the text before
    /// `args_marker` is the name, and the JSON value after it is the arguments.
    /// The next open or EOS closes the block.
    NameThenArgs { args_marker: &'static str },
    /// Qwen3.5's XML body: `<function=NAME>` followed by repeated
    /// `<parameter=KEY>VALUE</parameter>`, then `function_close` and
    /// `block_close`. The values are text, not JSON, so the parser serializes
    /// the collected object back to JSON text.
    XmlFunction {
        function_close: &'static str,
        block_close: &'static str,
    },
}

/// One family's call syntax.
#[derive(Debug, Clone, Copy)]
pub struct ToolFormat {
    /// Stable name, used by logs, errors and tests.
    pub id: &'static str,
    /// Every literal must appear in the template's source (comments removed)
    /// for this row to be picked.
    pub anchors: &'static [&'static str],
    /// None of these literals may appear in the template's source. This is how a
    /// row for one member of a family excludes a sibling with the same prefix
    /// (Mistral Nemo's array excludes the `[ARGS]` form, which has its own row).
    pub without: &'static [&'static str],
    /// The literal that opens a call block.
    pub open: &'static str,
    /// The block's body and its closing rule.
    pub shape: BlockShape,
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
/// arguments, and `close` follows the arguments. An **empty** `close` means the
/// arguments end the block (Ministral 3's `[ARGS]` form). The Qwen row's
/// literals include the newline and the key outside the object, because that is
/// the block the parser extracts; the other rows do the same for their wrappers.
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
    without: &[],
    open: "<tool_call>",
    shape: BlockShape::ObjectThenClose("</tool_call>"),
    name_key: "name",
    args_key: "arguments",
    grammar: Some(GrammarShape {
        open: "<tool_call>\n{\"name\": \"",
        between: "\", \"arguments\": ",
        close: "}\n</tool_call>",
    }),
};

/// Mistral Nemo / Magistral: one `[TOOL_CALLS]` block holding a JSON array.
///
/// The `[ARGS]` sibling (`[TOOL_CALLS]name[ARGS]{…}`, Ministral 3 / #709) shares
/// the open literal, so this row excludes a template that mentions `[ARGS]`.
pub static MISTRAL_NEMO: ToolFormat = ToolFormat {
    id: "mistral_nemo",
    anchors: &["[TOOL_CALLS]"],
    without: &["[ARGS]"],
    open: "[TOOL_CALLS]",
    shape: BlockShape::ArrayAtEos,
    name_key: "name",
    args_key: "arguments",
    grammar: Some(GrammarShape {
        open: "[TOOL_CALLS] [{\"name\": \"",
        between: "\", \"arguments\": ",
        close: "}]",
    }),
};

/// Ministral 3: `[TOOL_CALLS]name[ARGS]{…}` (#709).
///
/// The name is the text between the open and `[ARGS]`; the JSON value after it is
/// the arguments object itself (not a `{"name","arguments"}` wrapper). One call
/// per block; the next `[TOOL_CALLS]` or EOS ends it. The row sits before Mistral
/// Nemo, whose array form shares the open but has no `[ARGS]`.
pub static MISTRAL3: ToolFormat = ToolFormat {
    id: "ministral3",
    anchors: &["[TOOL_CALLS]", "[ARGS]"],
    without: &[],
    open: "[TOOL_CALLS]",
    shape: BlockShape::NameThenArgs {
        args_marker: "[ARGS]",
    },
    // The JSON keys are unused by the name-in-anchor shape; the fields stay for
    // the uniform row description.
    name_key: "name",
    args_key: "arguments",
    grammar: Some(GrammarShape {
        open: "[TOOL_CALLS]",
        between: "[ARGS]",
        close: "",
    }),
};

/// Llama 3.1: `<|python_tag|>` then one JSON object per call, `parameters` for
/// the arguments, and an optional `;` between calls.
///
/// A completion that omits the tag is content, not a call: the row is anchored on
/// the tag (llama.cpp's parser scans the JSON instead; the divergence is recorded
/// in the fixture).
pub static LLAMA31: ToolFormat = ToolFormat {
    id: "llama31",
    anchors: &["<|python_tag|>"],
    without: &[],
    open: "<|python_tag|>",
    shape: BlockShape::ObjectUntilNextOpen { separator: ";" },
    name_key: "name",
    args_key: "parameters",
    grammar: Some(GrammarShape {
        open: "<|python_tag|>{\"name\": \"",
        between: "\", \"parameters\": ",
        close: "}",
    }),
};

/// Qwen3.5: an XML-ish body inside the `<tool_call>` wrapper (#708).
///
/// The body is `<function=NAME>` followed by `<parameter=KEY>VALUE</parameter>`
/// entries; the values are text, and the parser serializes the collected object
/// to JSON text for OpenAI's `arguments` string. The row is parse-only:
/// `tool_choice` sees `grammar: None` and answers a named 400, because an XML
/// grammar over the tool's JSON schema is a second grammar compiler.
///
/// transformers 5.17.0's released `qwen3_5` fallback and llama.cpp's
/// Qwen3-Coder/3.5 parser both wrap the markup this way.
pub static QWEN35: ToolFormat = ToolFormat {
    id: "qwen3_5",
    anchors: &["<tool_call>", "<function="],
    without: &[],
    open: "<tool_call>",
    shape: BlockShape::XmlFunction {
        function_close: "</function>",
        block_close: "</tool_call>",
    },
    // The JSON keys are unused by the XML shape; the fields stay because a row
    // is one uniform description, and a future JSON-shaped sibling may share it.
    name_key: "name",
    args_key: "arguments",
    grammar: None,
};

/// The table, most specific first. `QWEN35` must precede `QWEN`: both open with
/// `<tool_call>`, and only the XML row's anchor list includes `<function=`.
/// `MISTRAL3` must precede `MISTRAL_NEMO`: the array row excludes `[ARGS]`, and
/// this one requires it.
static FORMATS: &[&ToolFormat] = &[&QWEN35, &MISTRAL3, &MISTRAL_NEMO, &LLAMA31, &QWEN];

/// Pick the row whose anchors all appear (and whose exclusions do not) in the
/// template's source.
///
/// An unrecognised template answers `None`; parsing then falls back to
/// [`QWEN`] (the pre-table behaviour), while `tool_choice` refuses by name.
pub fn detect(template: &str) -> Option<&'static ToolFormat> {
    let source = without_jinja_comments(template);
    FORMATS.iter().copied().find(|format| {
        format.anchors.iter().all(|anchor| source.contains(anchor))
            && !format.without.iter().any(|anchor| source.contains(anchor))
    })
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
