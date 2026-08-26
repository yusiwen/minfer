// Chat template rendering using minijinja
// Reads tokenizer.chat_template from GGUF, renders with message context

use minijinja::{context, Environment};

const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// Render a chat template with a full message list (server path,
/// OPENAI-CHAT-API-PLAN.md). `messages` are `(role, content)` pairs.
/// Falls back to multi-message ChatML if the template is missing/invalid or
/// rendering fails. `add_generation_prompt` appends the assistant header when
/// the template supports it; `bos_token` is exposed to the template context.
pub fn render_messages(
    template: &str,
    messages: &[(String, Option<String>)],
    add_generation_prompt: bool,
    bos_token: &str,
) -> String {
    let mut env = Environment::new();

    // Register the template
    if env.add_template("chat", template).is_err() {
        eprintln!("Warning: invalid chat template, falling back to ChatML");
        return fallback_chatml_messages(messages, add_generation_prompt);
    }
    let tmpl = match env.get_template("chat") {
        Ok(t) => t,
        Err(_) => return fallback_chatml_messages(messages, add_generation_prompt),
    };

    let msgs: Vec<serde_json::Value> = messages
        .iter()
        .map(|(role, content)| {
            serde_json::json!({
                "role": role,
                "content": content,
            })
        })
        .collect();

    let result = tmpl.render(context! {
        messages => msgs,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        tools => minijinja::Value::UNDEFINED,
    });

    match result {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "Warning: chat template rendering failed ({}), falling back to ChatML",
                e
            );
            fallback_chatml_messages(messages, add_generation_prompt)
        }
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
) -> FormattedDelta {
    let render = |msgs: &[(String, Option<String>)], add_gen: bool| match template {
        Some(t) => render_messages(t, msgs, add_gen, bos_token),
        None => fallback_chatml_messages(msgs, add_gen),
    };
    let fmt_past = if messages.is_empty() {
        String::new()
    } else {
        render(messages, false)
    };
    let mut all = messages.to_vec();
    all.push(new_msg);
    let fmt_new = render(&all, add_generation_prompt);

    let mut out = String::new();
    // Trailing-newline compensation: when fmt_past ends with '\n', the newline is
    // the prefix tail, eaten by the diff, but required by canonical text (after EOG, before next message).
    if add_generation_prompt && !fmt_past.is_empty() && fmt_past.ends_with('\n') {
        out.push('\n');
    }
    if fmt_new.starts_with(&fmt_past) {
        out.push_str(&fmt_new[fmt_past.len()..]);
        FormattedDelta {
            text: out,
            prefix_matched: true,
        }
    } else {
        // Prefix mismatch: non-deterministic template → return the full text; the caller falls back to a full re-render.
        FormattedDelta {
            text: fmt_new,
            prefix_matched: false,
        }
    }
}

/// Render a chat template with minijinja (single user message, CLI path).
/// Falls back to simple ChatML if the template cannot be rendered.
pub fn render_template(
    template: &str,
    user_input: &str,
    add_generation_prompt: bool,
    bos_token: &str,
) -> String {
    let mut env = Environment::new();

    // Register the template
    if env.add_template("chat", template).is_err() {
        eprintln!("Warning: invalid chat template, falling back to ChatML");
        return fallback_chatml(user_input, add_generation_prompt);
    }
    let tmpl = match env.get_template("chat") {
        Ok(t) => t,
        Err(_) => return fallback_chatml(user_input, add_generation_prompt),
    };

    let messages = vec![serde_json::json!({
        "role": "user",
        "content": user_input,
    })];

    let result = tmpl.render(context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        tools => minijinja::Value::UNDEFINED,
    });

    match result {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "Warning: chat template rendering failed ({}), falling back to ChatML",
                e
            );
            fallback_chatml(user_input, add_generation_prompt)
        }
    }
}

/// Fallback: simple ChatML format over ALL messages (server path). Every
/// message keeps its role/content; null content (assistant tool-call turn)
/// emits the role marker only.
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

/// Fallback: simple ChatML format (CLI path, single user message)
fn fallback_chatml(user_input: &str, add_generation_prompt: bool) -> String {
    let mut r = format!(
        "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n",
        DEFAULT_SYSTEM, user_input,
    );
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

    #[test]
    fn render_messages_with_jinja_template() {
        let tmpl = "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}</{{ m['role'] }}>{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
        let out = render_messages(tmpl, &msgs(), true, "<|endoftext|>");
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
        let mut m = msgs();
        m.push(("assistant".into(), None)); // tool-call turn
        let out = render_messages(tmpl, &m, false, "");
        assert!(out.starts_with("[system:You are helpful.]"), "got: {out}");
        assert!(out.contains("[user:Hi]"), "got: {out}");
        assert!(out.contains("[assistant:Hello!]"), "got: {out}");
        assert!(
            out.contains("[assistant:"),
            "null-content role marker present: {out}"
        );
    }

    #[test]
    fn render_messages_fallback_preserves_all_roles() {
        // invalid template -> ChatML fallback keeps system/assistant messages
        let out = render_messages("{{ bad", &msgs(), true, "");
        assert_eq!(
            out,
            "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello!<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    #[test]
    fn render_messages_fallback_null_content() {
        let mut m = msgs();
        m.push(("assistant".into(), None));
        let out = render_messages("{{ bad", &m, false, "");
        assert!(
            out.contains("<|im_start|>assistant\n<|im_end|>"),
            "null content role marker"
        );
        assert!(!out.contains("None"), "null must not be stringified");
    }

    #[test]
    fn render_template_single_user_unchanged() {
        // CLI path still renders a single user message via the template
        let tmpl =
            "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:";
        let out = render_template(tmpl, "hello", true, "");
        assert_eq!(out, "user: hello\nassistant:");
    }

    // === format_single (incremental diff rendering, CLI-CONVERSATION-PLAN.md §5.3) ===

    /// Qwen2.5-style ChatML template (a newline after each message, ending with an assistant header).
    const QWEN_CHATML: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    fn m(role: &str, content: &str) -> (String, Option<String>) {
        (role.to_string(), Some(content.to_string()))
    }

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
        );
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
        let fmt_past = render_messages(QWEN_CHATML, &past, false, "");
        let kv_prefix = fmt_past.strip_suffix('\n').unwrap_or(&fmt_past);
        let mut all = past.clone();
        all.push(m("user", "what is 2+2?"));
        let fmt_new = render_messages(QWEN_CHATML, &all, true, "");
        assert_eq!(format!("{kv_prefix}{}", d.text), fmt_new);
    }

    #[test]
    fn format_single_no_trailing_newline_no_compensation() {
        // Template produces no trailing '\n' → no compensation.
        let tmpl = "{% for mm in messages %}[{{ mm['role'] }}:{{ mm['content'] }}]{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}";
        let past = vec![m("user", "hi"), m("assistant", "Hello!")];
        let d = format_single(Some(tmpl), &past, m("user", "Q"), true, "");
        assert!(d.prefix_matched);
        assert_eq!(d.text, "[user:Q]<assistant>");
    }

    #[test]
    fn format_single_empty_history_returns_full_new() {
        let d = format_single(Some(QWEN_CHATML), &[], m("user", "first"), true, "");
        assert!(d.prefix_matched);
        let expect = render_messages(QWEN_CHATML, &[m("user", "first")], true, "");
        assert_eq!(d.text, expect);
    }

    #[test]
    fn format_single_prefix_mismatch_falls_back_to_full() {
        // Non-deterministic template (reverse) → fmt_new doesn't start with fmt_past → full text + flag.
        let tmpl = "{% for mm in messages|reverse %}[{{ mm['role'] }}]{% endfor %}";
        let past = vec![m("user", "hi"), m("assistant", "Hello!")];
        let d = format_single(Some(tmpl), &past, m("user", "Q"), false, "");
        assert!(!d.prefix_matched);
        let mut all = past.clone();
        all.push(m("user", "Q"));
        let expect = render_messages(tmpl, &all, false, "");
        assert_eq!(d.text, expect, "mismatch must return the full re-render");
    }

    #[test]
    fn format_single_fallback_without_template() {
        // template=None → rendered via the ChatML fallback; same prefix/compensation semantics.
        let past = vec![m("user", "hi"), m("assistant", "Hello!")];
        let d = format_single(None, &past, m("user", "Q"), true, "");
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
        let d = format_single(Some(tmpl), &past, m("assistant", "ok"), false, "");
        assert!(d.prefix_matched);
        assert_eq!(d.text, "[assistant]");
    }
}
