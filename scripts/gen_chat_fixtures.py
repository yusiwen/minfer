#!/usr/bin/env python3
"""Regenerate the CPython-authoritative chat fixtures.

Two fixtures, one authority (CPython, never this engine):

* `tests/fixtures/chat/pystr_semantics.json` — the Python `str` methods the
  chat templates call (the `SUPPORTED_STR_METHODS` hook).
* `tests/fixtures/chat/tojson_cpython.json` — the `tojson` filter (#597), which
  transformers binds to `json.dumps(..., ensure_ascii=False)`. The separators,
  the unescaped `<`/`>`/`&`/`'`, the raw non-ASCII and the key order are all
  CPython's, recorded here so the Rust filter is compared against CPython rather
  than against a hand-written expectation.

The engine is never its own reference: every expected value is what CPython
evaluates for the same input. Both fixtures are embedded at compile time by
`src/template/tests.rs`, so a change here is a visible diff in a Rust gate —
never an invisible test helper.

Usage:
    python3 scripts/gen_chat_fixtures.py            # rewrite both fixtures
    python3 scripts/gen_chat_fixtures.py --check    # compare, exit 1 on drift

`--check` compares the `entries` array and the `generator` provenance string of
both fixtures. The `date` field records when a fixture was last regenerated; it
is informational and deliberately excluded from the comparison, otherwise the
check would fail every day after a regeneration (issue #133).

Adding a `str` method means adding its cases here, regenerating, and widening
`SUPPORTED_STR_METHODS` in `src/template.rs` — the Rust gate asserts the two
sets are equal, so an implemented method without a case fails.
"""

from __future__ import annotations

import json
import pathlib
import platform
import sys

CHAT = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/chat"
FIXTURE = CHAT / "pystr_semantics.json"
TOJSON_FIXTURE = CHAT / "tojson_cpython.json"

# (method, receiver, args) — args are the JSON forms the Rust harness understands:
# a string, an integer, `null`, or a list of strings.
CASES: list[tuple[str, str, list]] = [
    # --- strip / lstrip / rstrip: the optional argument is a character SET ---
    ("strip", "\n\nhello\n", []),
    ("strip", "xxhelloxx", ["x"]),
    ("strip", " \t hello \t ", []),
    ("lstrip", "\n\nhello\n\n", ["\n"]),
    ("lstrip", "abchello", ["abc"]),
    ("lstrip", "hello", ["xyz"]),
    ("rstrip", "hello\n\n", ["\n"]),
    ("rstrip", "helloabcba", ["abc"]),
    # --- split / rsplit: separator and maxsplit ---
    ("split", "a</think>b</think>c", ["</think>"]),
    ("split", "a</think>b</think>c", ["</think>", 1]),
    ("split", "</think>leading", ["</think>"]),
    ("split", "trailing</think>", ["</think>"]),
    ("split", "a b  c\td\n", []),
    ("rsplit", "a.b.c", [".", 1]),
    # --- startswith / endswith: a string or a tuple of strings ---
    ("startswith", "hello world", ["hello"]),
    ("startswith", "hello world", [["x", "he"]]),
    ("endswith", "hello world", ["world"]),
    ("endswith", "hello", ["x"]),
    # --- replace: count omitted, positive, longer pattern ---
    ("replace", "a-b-c", ["-", "+"]),
    ("replace", "aaa", ["a", "b", 2]),
    ("replace", "aaaa", ["aa", "b"]),
    # --- case folding ---
    ("lower", "MiXeD", []),
    ("upper", "MiXeD", []),
    ("title", "hello world", []),
    ("capitalize", "hello world", []),
    # --- join: over a string and over a sequence ---
    ("join", "-", ["abc"]),
    ("join", ", ", [["a", "b", "c"]]),
    # --- find / rfind / count: character indexes, not byte offsets ---
    ("find", "hello world", ["world"]),
    ("find", "hello world", ["xyz"]),
    ("rfind", "a.b.c", ["."]),
    ("count", "banana", ["an"]),
    # --- splitlines: CPython's boundary set, with and without keepends ---
    ("splitlines", "a\nb\r\nc", []),
    ("splitlines", "a\nb\r\nc", [1]),
    ("splitlines", "trailing\n", []),
    ("splitlines", "a\rb\u2028c", []),
    ("splitlines", "x\u000bv\u000cf\u001cg\u001dh\u001ei\u0085j", []),
    ("splitlines", "", []),
    # --- ljust / rjust: width and one-character fill, unicode and wide input ---
    ("ljust", "ab", [5]),
    ("ljust", "ab", [5, "-"]),
    ("ljust", "abc", [2]),
    ("ljust", "", [3, "."]),
    ("rjust", "ab", [5]),
    ("rjust", "ab", [5, "0"]),
    ("rjust", "abc", [1]),
    ("rjust", "\u65e5\u672c", [5, "x"]),
]


# (value, indent) — the `tojson` filter's input and CPython's `json.dumps`
# arguments. `indent` is `None` for the compact form. Key order in these literals
# is the expected output order: `json.dumps` preserves insertion order, and the
# Rust gate deserializes the fixture in document order.
TOJSON_CASES: list[tuple[object, int | None]] = [
    # OpenAI tool shape: the separators are `", "` / `": "`, not the compact form.
    ({"type": "function", "function": {"name": "get_weather", "parameters": {"unit": "c"}}}, None),
    # Every value kind, nested; numbers keep their spelling (`0.5`, `-2`).
    ({"a": 1, "b": [1, 2, 3], "c": {"d": True, "e": False, "f": None}, "g": 0.5, "h": -2}, None),
    # CPython's `ensure_ascii=False` and Jinja's HTML escaping are both absent:
    # `<`, `>`, `&`, `'`, `"` and non-ASCII stay as written.
    ({"s": "<docs> & 'quotes' \"double\" / 检索 ünïcode"}, None),
    # The JSON control escapes.
    ({"k": "\n\t\r\b\f\u0000"}, None),
    # Empty containers, and arrays of containers.
    ({"nested": [[], {}, [{"x": [1]}], []]}, None),
    # Non-object roots.
    ("plain string", None),
    (42, None),
    (None, None),
    # The pretty form (a template can ask for an indent).
    ({"a": [1, {"b": "c"}], "d": {}}, 2),
]


def generate_tojson() -> list[dict]:
    entries = []
    for value, indent in TOJSON_CASES:
        entries.append(
            {
                "value": value,
                "indent": indent,
                "expected": json.dumps(value, ensure_ascii=False, indent=indent),
            }
        )
    return entries


def build_tojson(entries: list[dict], date: str) -> dict:
    return {
        "provenance": {
            "reference": "CPython json.dumps(value, ensure_ascii=False, indent=indent)",
            "generator": f"CPython {platform.python_version()} json.dumps",
            "command": "scripts/gen_chat_fixtures.py",
            "date": date,
        },
        "entries": entries,
    }


def build(entries: list[dict], date: str) -> dict:
    return {
        "provenance": {
            "generator": f"CPython {platform.python_version()} builtin str methods",
            "command": "scripts/gen_chat_fixtures.py",
            "date": date,
        },
        "entries": entries,
    }


def generate() -> list[dict]:
    entries = []
    for method, receiver, args in CASES:
        # A JSON list is CPython's tuple argument (startswith(("a", "b"))).
        call_args = [tuple(a) if isinstance(a, list) else a for a in args]
        expected = getattr(receiver, method)(*call_args)
        entries.append(
            {
                "method": method,
                "receiver": receiver,
                "args": args,
                "expected": expected,
            }
        )
    return entries


def main(argv: list[str]) -> int:
    check = "--check" in argv[1:]

    if check:
        if not FIXTURE.exists():
            print(f"missing fixture: {FIXTURE}", file=sys.stderr)
            return 1
        committed = json.loads(FIXTURE.read_text())
        fresh = generate()
        if committed.get("entries") != fresh:
            print(
                "fixture drift: the committed entries are not what CPython evaluates "
                "for this script's CASES (run the script without --check)",
                file=sys.stderr,
            )
            for i, (a, b) in enumerate(zip(committed.get("entries", []), fresh)):
                if a != b:
                    print(f"  first difference at entry {i}:\n    committed: {a}\n    fresh:     {b}", file=sys.stderr)
                    break
            if len(committed.get("entries", [])) != len(fresh):
                print(
                    f"  committed {len(committed.get('entries', []))} entries, "
                    f"fresh {len(fresh)}",
                    file=sys.stderr,
                )
            return 1
        committed_generator = committed.get("provenance", {}).get("generator")
        fresh_generator = f"CPython {platform.python_version()} builtin str methods"
        if committed_generator != fresh_generator:
            print(
                f"generator drift: fixture says {committed_generator!r}, this interpreter is "
                f"{fresh_generator!r}",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {len(fresh)} entries match CPython {platform.python_version()}")

        if not TOJSON_FIXTURE.exists():
            print(f"missing fixture: {TOJSON_FIXTURE}", file=sys.stderr)
            return 1
        committed_tojson = json.loads(TOJSON_FIXTURE.read_text())
        fresh_tojson = generate_tojson()
        if committed_tojson.get("entries") != fresh_tojson:
            print(
                "fixture drift: the committed tojson entries are not what CPython "
                "json.dumps evaluates (run the script without --check)",
                file=sys.stderr,
            )
            for i, (a, b) in enumerate(zip(committed_tojson.get("entries", []), fresh_tojson)):
                if a != b:
                    print(f"  first difference at entry {i}:\n    committed: {a!r}\n    fresh:     {b!r}", file=sys.stderr)
                    break
            return 1
        generator = committed_tojson.get("provenance", {}).get("generator")
        fresh_generator_tojson = f"CPython {platform.python_version()} json.dumps"
        if generator != fresh_generator_tojson:
            print(
                f"generator drift: fixture says {generator!r}, this interpreter is "
                f"{fresh_generator_tojson!r}",
                file=sys.stderr,
            )
            return 1
        print(f"ok: {len(fresh_tojson)} tojson entries match CPython {platform.python_version()}")
        return 0

    date = FIXTURE.exists() and json.loads(FIXTURE.read_text()).get("provenance", {}).get("date")
    if not date:
        import datetime

        date = datetime.date.today().isoformat()
    doc = build(generate(), date)
    FIXTURE.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE} ({len(doc['entries'])} entries)")

    tojson_doc = build_tojson(generate_tojson(), date)
    TOJSON_FIXTURE.write_text(json.dumps(tojson_doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {TOJSON_FIXTURE} ({len(tojson_doc['entries'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
