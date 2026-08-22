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

/// 差分渲染结果（`common_chat_format_single` 的 minfer 版）。
pub struct FormattedDelta {
    /// 追加文本：`fmt_new` 去掉 `fmt_past` 前缀后的部分（含尾部换行补偿）。
    pub text: String,
    /// `fmt_new` 是否以 `fmt_past` 为前缀。false 表示模板非确定性（如依赖
    /// 外部状态），调用方必须放弃增量、走全量重灌兜底（§5.4）。
    pub prefix_matched: bool,
}

/// 增量渲染：给定已记录历史与新消息，返回"只需追加进 KV"的文本增量。
///
/// 对应 llama.cpp `common_chat_format_single`（common/chat.cpp:653）：
/// 两次渲染（有/无新消息）取差分，避免把已生成的历史重复喂给模型；
/// 若 `fmt_past` 以 `\n` 结尾则前补 `\n`——该换行属于前缀尾部，被 diff 吃掉，
/// 但模型生成的 EOG 之后不会自带它，而 canonical 文本需要它（§3.2/§5.4）。
///
/// `template` 为 None 时用 ChatML fallback 渲染（`fallback_chatml_messages`）。
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
    // 尾部换行补偿：fmt_past 以 '\n' 结尾时，该换行属于前缀尾部，diff 会丢掉它，
    // 但 canonical 文本（模型 EOG 之后、下一条消息之前）需要它。
    if add_generation_prompt && !fmt_past.is_empty() && fmt_past.ends_with('\n') {
        out.push('\n');
    }
    if fmt_new.starts_with(&fmt_past) {
        out.push_str(&fmt_new[fmt_past.len()..]);
        FormattedDelta { text: out, prefix_matched: true }
    } else {
        // 前缀失败：模板非确定性 → 返回全量，调用方走全量重灌兜底。
        FormattedDelta { text: fmt_new, prefix_matched: false }
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

    // === format_single（增量差分渲染，CLI-CONVERSATION-PLAN.md §5.3）===

    /// Qwen2.5 风格 ChatML 模板（每条消息后跟一个换行，结尾有 assistant 头）。
    const QWEN_CHATML: &str = "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n{% endif %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}";

    fn m(role: &str, content: &str) -> (String, Option<String>) {
        (role.to_string(), Some(content.to_string()))
    }

    #[test]
    fn format_single_diffs_only_new_user_message() {
        // 历史 [system, user, assistant] + 新 user → delta 只含新消息与 assistant 头，
        // 不得重复包含已生成的 assistant 内容。
        let past = vec![
            m("system", "You are helpful."),
            m("user", "hi"),
            m("assistant", "Hello!"),
        ];
        let d = format_single(Some(QWEN_CHATML), &past, m("user", "what is 2+2?"), true, "");
        assert!(d.prefix_matched, "prefix must match for a deterministic template");
        assert_eq!(
            d.text,
            "\n<|im_start|>user\nwhat is 2+2?<|im_end|>\n<|im_start|>assistant\n"
        );
        // 不变量：KV 前缀 + delta == fmt_new（canonical 全量渲染）。
        // KV 前缀 = fmt_past 去掉模板在最后一条消息后输出的换行——模型生成 EOG 后
        // 不会自带该换行，补偿逻辑正是把它加回 delta 开头。
        let fmt_past = render_messages(QWEN_CHATML, &past, false, "");
        let kv_prefix = fmt_past.strip_suffix('\n').unwrap_or(&fmt_past);
        let mut all = past.clone();
        all.push(m("user", "what is 2+2?"));
        let fmt_new = render_messages(QWEN_CHATML, &all, true, "");
        assert_eq!(format!("{kv_prefix}{}", d.text), fmt_new);
    }

    #[test]
    fn format_single_no_trailing_newline_no_compensation() {
        // 模板不产生尾部 '\n' → 不做补偿。
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
        // 非确定性模板（reverse）→ fmt_new 不以 fmt_past 为前缀 → 全量 + 标记。
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
        // template=None → ChatML fallback 渲染，同样满足前缀/补偿语义。
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
        // assistant 消息 content=None（tool-call 回合）不破坏差分。
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
