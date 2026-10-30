#!/usr/bin/env python3
"""Regenerate `tests/fixtures/chat/parse_qwen2.5.json` (#614).

The engine is never its own reference. Every `expected` message in this fixture is
what **transformers 5.17.0** parses out of the same completion, through
`transformers.utils.chat_parsing.parse_response` and the Qwen2/Qwen3 response
template that ships with it (`transformers/cli/serving/utils.py`,
`_RESPONSE_TEMPLATE_FALLBACKS`), with the same tool list.

`parse_response` returns the call's arguments as a JSON value (a dict for an object,
a string when the block carried a string), which is exactly the two arms minfer's
parser distinguishes: an object is carried as the model's own bytes, a string is
decoded. The Rust gate compares the parsed values (and, for the prose cases, the
content) — it never compares this engine's output with itself.

The fixture also asserts, at generation time, that every parsed call renders back
through the committed Qwen2.5 template into a `<tool_call>` block: the parse is a
call the model's own template can replay (the round trip ADR-0062 opened).

Usage:
    uv run --with transformers==5.17.0 python scripts/gen_tool_call_parse_fixture.py
    uv run --with transformers==5.17.0 python scripts/gen_tool_call_parse_fixture.py --check

`--check` re-parses every case (excluding the `date` field) and compares. It needs
`transformers`, which the devShell does not provide, so it is a manual regeneration
tool; the committed bytes are what CI gates.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tests/fixtures/chat/qwen2.5-0.5b-instruct.json"
FIXTURE = ROOT / "tests/fixtures/chat/parse_qwen2.5.json"

# The Qwen2/Qwen3 response template transformers ships as a fallback for tokenizers
# without one. Copied verbatim (including the comments' reasoning) so the fixture's
# parser is the released one, not our paraphrase.
SPEC = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|im_start|>assistant\n",
    "fields": {
        "thinking": {"open": "<think>", "close": "</think>", "content": "text"},
        "tool_calls": {
            "open_pattern": r"\s*<tool_call>",
            "close": "</tool_call>",
            "repeats": True,
            "content": "json",
            "transform": {"type": "function", "function": "{content}"},
        },
        "content": {
            "close_pattern": r"\s*(?:<\|im_end\|>|<\|endoftext\|>|<\|eot_id\|>)",
            "content": "text",
        },
    },
}

GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string"},
            },
            "required": ["city"],
        },
    },
}

PREFIX = "<|im_start|>assistant\n"

CALL = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>'

CASES = [
    {"name": "single_call", "completion": CALL},
    {
        "name": "two_calls",
        "completion": CALL + "\n" + CALL,
    },
    {
        "name": "content_before_the_call",
        "completion": "Let me check.\n" + CALL,
    },
    {
        "name": "content_after_the_call",
        "completion": CALL + "\nDone.",
    },
    {
        "name": "string_arguments",
        "completion": '<tool_call>\n{"name": "get_weather", "arguments": "{\\"city\\": \\"Paris\\"}"}\n</tool_call>',
    },
    {
        "name": "unicode_and_html_in_the_arguments",
        "completion": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "检索 <b> & \'c\'"}}\n</tool_call>',
    },
    {
        # The closing tag as data: a splitter that looks for the literal fails here.
        # transformers' region scan *does* fail on it, so this case records a
        # deliberate divergence: minfer finds the JSON value's end and is not
        # confused by the literal. See the Rust gate.
        "name": "closing_tag_inside_the_arguments",
        "diverges": "transformers splits the region at the first `</tool_call>` and refuses the incomplete JSON; minfer scans the JSON value and accepts it",
        "completion": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "</tool_call>"}}\n</tool_call>',
    },
    {
        "name": "extra_keys_in_the_call_object",
        "completion": '<tool_call>\n{"id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
    },
    {"name": "prose", "completion": "Hello, world!\n"},
    {
        "name": "truncated_json",
        "completion": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Par',
    },
    {
        "name": "truncated_close",
        "completion": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}',
    },
    {"name": "empty_block", "completion": "<tool_call>"},
]


def render_reconstruction(call: dict, template: str) -> str:
    """Render the parsed call back through the model's template."""
    from transformers.utils.chat_template_utils import render_jinja_template

    messages = [
        {"role": "user", "content": "Weather in Paris?"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [call],
        },
    ]
    rendered, _ = render_jinja_template(
        conversations=[messages],
        tools=[GET_WEATHER],
        chat_template=template,
        add_generation_prompt=False,
        bos_token=None,
    )
    return rendered[0]


def build() -> dict:
    from transformers.utils.chat_parsing import parse_response

    reference = json.loads(REFERENCE.read_text())
    template = reference["template"]

    cases = []
    for case in CASES:
        try:
            message = parse_response(
                case["completion"], SPEC, prefix=PREFIX, tools=[GET_WEATHER]
            )
        except ValueError as exc:
            cases.append({**case, "error": str(exc)})
            continue
        calls = message.get("tool_calls") or []
        for call in calls:
            rendered = render_reconstruction(call, template)
            if "<tool_call>" not in rendered or call["function"]["name"] not in rendered:
                sys.exit(
                    f"case {case['name']}: the parsed call does not render back through "
                    f"the template: {rendered!r}"
                )
        cases.append(
            {
                **case,
                "expected": {
                    "content": message.get("content"),
                    "tool_calls": [
                        {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        }
                        for call in calls
                    ],
                },
            }
        )

    return {
        "model": reference["model"],
        "revision": reference["revision"],
        "response_template": SPEC,
        "prefix": PREFIX,
        "tools": [GET_WEATHER],
        "provenance": {
            "reference": (
                "HuggingFaceTB transformers 5.17.0 "
                "transformers.utils.chat_parsing.parse_response with the Qwen2/Qwen3 "
                "response-template fallback shipped in transformers/cli/serving/utils.py "
                "(_RESPONSE_TEMPLATE_FALLBACKS)"
            ),
            "template_source": (
                "tests/fixtures/chat/qwen2.5-0.5b-instruct.json (the same template the "
                "tools and replay fixtures render)"
            ),
            "command": "uv run --with transformers==5.17.0 python scripts/gen_tool_call_parse_fixture.py",
            "date": "2026-10-30",
        },
        "cases": cases,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="compare instead of rewriting")
    args = ap.parse_args()

    built = build()
    if args.check:
        if not FIXTURE.exists():
            print(f"{FIXTURE} is missing", file=sys.stderr)
            return 1
        current = json.loads(FIXTURE.read_text())
        for doc in (current, built):
            doc.get("provenance", {}).pop("date", None)
        if current != built:
            print(f"{FIXTURE} is stale; rerun without --check", file=sys.stderr)
            return 1
        print(f"{FIXTURE} matches the transformers reference ({len(built['cases'])} cases)")
        return 0

    FIXTURE.write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE} ({len(built['cases'])} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
