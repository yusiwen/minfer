// Chat template rendering using minijinja
// Reads tokenizer.chat_template from GGUF, renders with message context

use minijinja::{Environment, context};

const DEFAULT_QWEN_SYSTEM: &str = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";
const DEFAULT_SYSTEM: &str = "You are a helpful assistant.";

/// Render a chat template with minijinja.
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

/// Fallback: simple ChatML format
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
