#!/usr/bin/env python3
"""Regenerate the Mistral Nemo and Llama 3.1 tool-call parse fixtures (#699).

The engine is never its own reference. Each `expected` message is what
**transformers 5.17.0** parses out of the same completion, through
`transformers.utils.chat_parsing.parse_response` and an authored
`response_template` for the family. transformers ships no Mistral or Llama
response template, so — unlike `parse_qwen2.5.json` — the template here is ours;
llama.cpp's `common_chat_parse_mistral_nemo` / `common_chat_parse_llama_3_1` is
the second oracle for the accepted and rejected shapes. The fixture records both
facts in `provenance`, so a reader can tell a released reference from an
authored one.

The two fixtures exist so the engine's parser can be compared case for case:

    tests/fixtures/chat/parse_mistral_nemo.json
    tests/fixtures/chat/parse_llama31.json

Usage:
    uv run --with transformers==5.17.0 python scripts/gen_tool_call_family_fixtures.py
    uv run --with transformers==5.17.0 python scripts/gen_tool_call_family_fixtures.py --check

`--check` re-parses every case (excluding the `date` field) and compares. It
needs `transformers`, which the devShell does not provide, so it is a manual
regeneration tool; the committed bytes are what CI gates.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CHAT = ROOT / "tests/fixtures/chat"

GET_WEATHER = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}

# Mistral Nemo / Magistral: one `[TOOL_CALLS]` block holding a JSON array. The
# leading `\s*` in the content close keeps the space before the marker out of the
# prose; the array is the block, so there is no close literal.
MISTRAL_SPEC = {
    "defaults": {"role": "assistant"},
    "start_anchor": "[/INST]",
    "fields": {
        "tool_calls": {
            "open": "[TOOL_CALLS]",
            "content": "json",
            "transform_each": True,
            "transform": {
                "type": "function",
                "function": {"name": "{name}", "arguments": "{arguments}"},
            },
        },
        "content": {"close": "[TOOL_CALLS]", "content": "text"},
    },
}
MISTRAL_PREFIX = "[INST] Weather in Paris? [/INST]"
MISTRAL_CASES = [
    (
        "single_call",
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    ),
    (
        "two_calls",
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}, '
        '{"name": "get_weather", "arguments": {"city": "Lyon"}}]',
    ),
    (
        "content_before_the_call",
        'Let me check. [TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    ),
    (
        "extra_id_key",
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}, '
        '"id": "abc123456"}]',
    ),
    (
        "string_arguments",
        '[TOOL_CALLS] [{"name": "f", "arguments": "{\\"city\\": \\"Paris\\"}"}]',
    ),
    (
        "truncated_array",
        '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Par"',
    ),
    ("prose", "Hello, world!"),
]

# Llama 3.1: `<|python_tag|>` then one JSON object per call, `parameters` for the
# arguments, and an optional `;`/newline between calls. The close is a lookahead
# at the next tag or EOS, so the `;` is consumed without entering the body.
LLAMA_SPEC = {
    "defaults": {"role": "assistant"},
    "start_anchor": "<|start_header_id|>assistant<|end_header_id|>\n\n",
    "fields": {
        "tool_calls": {
            "open": "<|python_tag|>",
            "close_pattern": r"\s*;?\s*(?=<\|python_tag\|>|<\|eot_id\|>|\Z)",
            "content": "json",
            "repeats": True,
            "transform": {
                "type": "function",
                "function": {"name": "{content.name}", "arguments": "{content.parameters}"},
            },
        },
        "content": {"close": "<|python_tag|>", "content": "text"},
    },
}
LLAMA_PREFIX = (
    "<|start_header_id|>user<|end_header_id|>\n\nWeather in Paris?<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)
LLAMA_CASES = [
    (
        "single_call",
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}',
    ),
    (
        "two_calls",
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}\n'
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Lyon"}}',
    ),
    (
        "two_calls_semicolon",
        '<|python_tag|>{"name": "a", "parameters": {}};'
        '<|python_tag|>{"name": "b", "parameters": {"x": 1}}',
    ),
    (
        "content_before_the_call",
        'Let me check. <|python_tag|>{"name": "get_weather", "parameters": {"city": "Paris"}}',
    ),
    (
        "type_key",
        '<|python_tag|>{"type": "function", "name": "get_weather", '
        '"parameters": {"city": "Paris"}}',
    ),
    (
        "truncated_object",
        '<|python_tag|>{"name": "get_weather", "parameters": {"city": "Par"',
    ),
    (
        "without_the_python_tag",
        '{"name": "get_weather", "parameters": {"city": "Paris"}}',
    ),
    ("prose", "Hello, world!"),
]


def build(spec: dict, prefix: str, cases: list[tuple[str, str]]) -> list[dict]:
    from transformers.utils.chat_parsing import parse_response

    out = []
    for name, completion in cases:
        try:
            message = parse_response(completion, spec, prefix=prefix, tools=[GET_WEATHER])
        except ValueError as exc:
            out.append({"name": name, "completion": completion, "error": str(exc)})
            continue
        calls = message.get("tool_calls") or []
        out.append(
            {
                "name": name,
                "completion": completion,
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
    return out


def document(family: str, spec: dict, prefix: str, cases: list[dict], reference: str) -> dict:
    return {
        "format": family,
        "response_template": spec,
        "prefix": prefix,
        "tools": [GET_WEATHER],
        "provenance": {
            "reference": (
                "HuggingFaceTB transformers 5.17.0 "
                "transformers.utils.chat_parsing.parse_response with an authored "
                f"response_template for {reference}"
            ),
            "template_source": (
                "authored from the family's published call syntax; llama.cpp's "
                "parser is the second oracle"
            ),
            "command": (
                "uv run --with transformers==5.17.0 python "
                f"scripts/gen_tool_call_family_fixtures.py --family {family}"
            ),
            "date": "2026-11-01",
        },
        "cases": cases,
    }


def targets() -> dict[str, tuple[pathlib.Path, dict]]:
    return {
        "mistral_nemo": (
            CHAT / "parse_mistral_nemo.json",
            document(
                "mistral_nemo",
                MISTRAL_SPEC,
                MISTRAL_PREFIX,
                build(MISTRAL_SPEC, MISTRAL_PREFIX, MISTRAL_CASES),
                "Mistral Nemo / Magistral `[TOOL_CALLS] [{…}]` "
                "(llama.cpp `common_chat_parse_mistral_nemo`)",
            ),
        ),
        "llama31": (
            CHAT / "parse_llama31.json",
            document(
                "llama31",
                LLAMA_SPEC,
                LLAMA_PREFIX,
                build(LLAMA_SPEC, LLAMA_PREFIX, LLAMA_CASES),
                'Llama 3.1 `{"name":…, "parameters":…}` '
                "(llama.cpp `common_chat_parse_llama_3_1`)",
            ),
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--family",
        choices=sorted(targets()),
        help="one family; the default checks or rewrites both",
    )
    ap.add_argument("--check", action="store_true", help="compare instead of rewriting")
    args = ap.parse_args()

    chosen = targets()
    if args.family:
        chosen = {args.family: chosen[args.family]}
    failures = 0
    for family, (path, built) in chosen.items():
        if args.check:
            if not path.exists():
                print(f"{path} is missing", file=sys.stderr)
                failures += 1
                continue
            current = json.loads(path.read_text())
            for doc in (current, built):
                doc.get("provenance", {}).pop("date", None)
            if current != built:
                print(f"{path} is stale; rerun without --check", file=sys.stderr)
                failures += 1
                continue
            print(f"{path} matches the transformers reference ({len(built['cases'])} cases)")
        else:
            path.write_text(json.dumps(built, indent=2, ensure_ascii=False) + "\n")
            print(f"wrote {path} ({len(built['cases'])} cases)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
