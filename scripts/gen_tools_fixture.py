#!/usr/bin/env python3
"""Regenerate `tests/fixtures/chat/tools_qwen2.5-0.5b-instruct.json` (#597).

The engine is never its own reference: every `expected` string in the fixture is
what **transformers 5.17.0** renders for the same `messages` + `tools`, using
`render_jinja_template` — the exact function `AutoTokenizer.apply_chat_template`
calls (`tokenization_utils_base.py`). The template text is copied from the
already-committed `qwen2.5-0.5b-instruct.json` and its sha256 is verified, so the
tools reference and the tools-free reference are provably the same template.

The fixture is embedded at compile time by `src/template/tests.rs`, so a change
here is a visible diff in a Rust gate — never an invisible test helper.

Usage:
    uv run --with transformers==5.17.0 python scripts/gen_tools_fixture.py
    uv run --with transformers==5.17.0 python scripts/gen_tools_fixture.py --check

`--check` re-renders every case and compares, excluding the `date` field (which
records when the fixture was regenerated) — the same rule `gen_chat_fixtures.py`
follows. It needs `transformers`, which the flake's devShell does not provide, so
it is a manual regeneration tool and not part of the CI doc gates; the committed
bytes are what CI gates.

Adding a case means adding it to `CASES` here, regenerating, and re-running the
Rust gate — the gate asserts that the fixture is non-trivial (every case carries
tools, the bytes contain the `<tools>` block, and at least one case exercises a
non-ASCII and an HTML-character description).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
REFERENCE = ROOT / "tests/fixtures/chat/qwen2.5-0.5b-instruct.json"
FIXTURE = ROOT / "tests/fixtures/chat/tools_qwen2.5-0.5b-instruct.json"

# A get-weather tool in the OpenAI shape. The key order is the client's
# (`type` first): a `BTreeMap`-backed JSON value would sort it, which is the
# whole reason `src/template/tool_json.rs` exists.
GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The city name."},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["city"],
        },
    },
}

# Two tools, nesting, an array bound, a fractional default, and a description
# that carries `<`, `>`, `&`, `'` and non-ASCII text: CPython's `json.dumps`
# leaves all of those alone (`ensure_ascii=False`, no HTML escaping), so a
# renderer that escapes them fails here.
SEARCH_AND_CONVERT = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search <docs> & notes for a query; unicode stays raw: 检索 ünïcode.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to look for."},
                    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
                    "score": {"type": "number", "default": 0.5},
                    "include_archived": {"type": "boolean", "default": False},
                    "context": {
                        "type": "object",
                        "properties": {
                            "before": {"type": "integer", "minimum": 0},
                            "after": {"type": "integer", "minimum": 0},
                        },
                        "additionalProperties": False,
                    },
                },
                "required": ["query"],
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Convert an amount between currencies.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number"},
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                },
                "required": ["amount", "from", "to"],
            },
        },
    },
]

CASES = [
    {
        "name": "single_function",
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": "What is the weather in Paris?"},
        ],
        "tools": [GET_WEATHER],
        "add_generation_prompt": True,
    },
    {
        "name": "two_functions_nested_schema",
        "messages": [
            {"role": "user", "content": "Search the docs for batching, then convert 12.5 EUR."}
        ],
        "tools": SEARCH_AND_CONVERT,
        "add_generation_prompt": True,
    },
    {
        # No system message: the template supplies its own default inside the
        # tools branch, which is a different code path from case 1.
        "name": "no_system_message_no_generation_prompt",
        "messages": [{"role": "user", "content": "Which tools do you have?"}],
        "tools": [GET_WEATHER],
        "add_generation_prompt": False,
    },
    {
        # A replayed tool *round-trip* (#613): the assistant turn carries
        # `tool_calls`, so it renders through the template's replay branch instead
        # of its plain-content branch. `content` is null — the shape OpenAI
        # clients send — and `arguments` is a JSON string, which the template
        # echoes through `tojson`.
        "name": "assistant_tool_call_replay",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_weather_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": '{"temperature": 18, "unit": "celsius"}'},
        ],
        "tools": [GET_WEATHER],
        "add_generation_prompt": True,
    },
    {
        # A replayed tool result. The messages carry only `role`/`content` — the
        # same shape minfer's request type has — so the assistant turn renders
        # through the template's plain-content branch (it has no `tool_calls`)
        # and the tool turn through its `<tool_response>` branch.
        "name": "tool_response_replay",
        "messages": [
            {"role": "user", "content": "Weather in Paris?"},
            {"role": "assistant", "content": "Let me check."},
            {"role": "tool", "content": '{"temperature": 18, "unit": "celsius"}'},
        ],
        "tools": [GET_WEATHER],
        "add_generation_prompt": True,
    },
]


def render(case: dict, template: str) -> str:
    # Imported here so `--help` works without transformers installed.
    from transformers.utils.chat_template_utils import render_jinja_template

    rendered, _generation_indices = render_jinja_template(
        conversations=[case["messages"]],
        tools=case["tools"],
        chat_template=template,
        add_generation_prompt=case["add_generation_prompt"],
        bos_token=None,
    )
    return rendered[0]


def build() -> dict:
    reference = json.loads(REFERENCE.read_text())
    template = reference["template"]
    digest = hashlib.sha256(template.encode()).hexdigest()
    if digest != reference["template_sha256"]:
        sys.exit(
            f"{REFERENCE} is internally inconsistent: template sha256 is {digest}, "
            f"the fixture records {reference['template_sha256']}"
        )

    cases = []
    for case in CASES:
        out = render(case, template)
        if "<tools>" not in out or "</tools>" not in out:
            sys.exit(f"case {case['name']}: the reference did not render a tools block: {out!r}")
        cases.append({**case, "expected": out})

    return {
        "model": reference["model"],
        "revision": reference["revision"],
        "template_sha256": digest,
        "template": template,
        "bos_token": None,
        "provenance": {
            "reference": (
                "HuggingFaceTB transformers 5.17.0 "
                "transformers.utils.chat_template_utils.render_jinja_template — the function "
                "AutoTokenizer.apply_chat_template calls (tokenization_utils_base.py)"
            ),
            "template_source": (
                f"copied from tests/fixtures/chat/qwen2.5-0.5b-instruct.json, sha256 {digest}"
            ),
            "command": (
                "uv run --with transformers==5.17.0 python scripts/gen_tools_fixture.py"
            ),
            "date": "2026-10-19",
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
