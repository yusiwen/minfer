// Chat template formatting for Qwen2 ChatML style
// Reads tokenizer.chat_template from GGUF, reformats as simple ChatML

/// Render a chat template: wraps user input with ChatML format
/// For Qwen2: <|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
/// Also handles other common templates by extracting message format
pub fn render_template(template: &str, user_input: &str, _add_generation_prompt: bool) -> String {
    // Detect ChatML template (Qwen2, DeepSeek, etc.)
    if template.contains("<|im_start|>") {
        return format!(
            "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            user_input
        );
    }

    // Detect Llama 3 template
    if template.contains("<|begin_of_text|>") || template.contains("<|start_header_id|>") {
        return format!(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            user_input
        );
    }

    // Detect Mistral template
    if template.contains("[INST]") {
        return format!("<s>[INST] {} [/INST]", user_input);
    }

    // Fallback: use template directly with simple {input} substitution
    let result = template.replace("{input}", user_input)
        .replace("{{input}}", user_input)
        .replace("{{ message['content'] }}", user_input)
        .replace("{content}", user_input);

    if result != template {
        return result;
    }

    // Last resort: just return the user input as-is
    user_input.to_string()
}
