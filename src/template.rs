// Chat template rendering using minijinja
// Reads tokenizer.chat_template from GGUF, renders with message context

use minijinja::{Environment, context};

const DEFAULT_QWEN_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
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
            eprintln!("Warning: chat template rendering failed ({}), falling back to ChatML", e);
            fallback_chatml_messages(messages, add_generation_prompt)
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

    let messages = vec![
        serde_json::json!({
            "role": "user",
            "content": user_input,
        }),
    ];

    let result = tmpl.render(context! {
        messages => messages,
        add_generation_prompt => add_generation_prompt,
        bos_token => bos_token,
        tools => minijinja::Value::UNDEFINED,
    });

    match result {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Warning: chat template rendering failed ({}), falling back to ChatML", e);
            fallback_chatml(user_input, add_generation_prompt)
        }
    }
}

/// Fallback: simple ChatML format over ALL messages (server path). Every
/// message keeps its role/content; null content (assistant tool-call turn)
/// emits the role marker only.
fn fallback_chatml_messages(
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
        DEFAULT_SYSTEM,
        user_input,
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
        assert!(out.contains("[assistant:"), "null-content role marker present: {out}");
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
        assert!(out.contains("<|im_start|>assistant\n<|im_end|>"), "null content role marker");
        assert!(!out.contains("None"), "null must not be stringified");
    }

    #[test]
    fn render_template_single_user_unchanged() {
        // CLI path still renders a single user message via the template
        let tmpl = "{% for m in messages %}{{ m['role'] }}: {{ m['content'] }}\n{% endfor %}assistant:";
        let out = render_template(tmpl, "hello", true, "");
        assert_eq!(out, "user: hello\nassistant:");
    }
}
