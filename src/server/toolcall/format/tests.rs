//! Unit tests for the tool-call format table (#699, ADR-0066). See the parent
//! module for the contract; the family rows' fixtures live with the parser's.

use super::*;

fn detected(template: &str) -> Option<&'static str> {
    detect(template).map(|format| format.id)
}

#[test]
fn the_qwen_row_is_detected_from_the_template_source() {
    assert_eq!(detected("<tool_call>"), Some("qwen2"));
    assert_eq!(
        detected("{% if tools %}<tool_call>{% endif %}"),
        Some("qwen2")
    );
}

#[test]
fn a_jinja_comment_does_not_count_as_a_rendered_format() {
    assert_eq!(detected("{# <tool_call> #}"), None);
    assert_eq!(
        detected("{# <tool_call> #}{% if false %}<tool_call>{% endif %}"),
        Some("qwen2"),
        "the comment is gone, the unreachable branch's literal is still source"
    );
}

#[test]
fn the_mistral_array_row_is_detected_and_the_args_form_is_not() {
    assert_eq!(
        detected("[TOOL_CALLS] [{\"name\": \"get_weather\""),
        Some("mistral_nemo")
    );
    assert_eq!(
        detected("[TOOL_CALLS]get_weather[ARGS]{\"city\": \"Paris\"}"),
        None,
        "the name-in-anchor [ARGS] form is #709, not the array row"
    );
}

#[test]
fn the_llama_row_is_detected_from_the_python_tag() {
    assert_eq!(
        detected("<|python_tag|>{\"name\": \"get_weather\", \"parameters\": {}}"),
        Some("llama31")
    );
}

#[test]
fn a_template_without_a_known_anchor_has_no_row() {
    assert_eq!(
        detected("{% for m in messages %}{{ m.content }}{% endfor %}"),
        None
    );
    assert_eq!(detected(""), None);
}

#[test]
fn an_unterminated_comment_drops_the_rest() {
    assert_eq!(without_jinja_comments("a{# b").as_ref(), "a");
    assert_eq!(without_jinja_comments("a{# b #}c").as_ref(), "ac");
    assert_eq!(without_jinja_comments("plain").as_ref(), "plain");
}
