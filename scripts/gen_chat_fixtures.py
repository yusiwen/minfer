#!/usr/bin/env python3
"""Regenerate `tests/fixtures/chat/pystr_semantics.json` from CPython itself.

The engine is never its own reference: every expected value in the fixture is
what CPython evaluates for the same receiver and arguments. The fixture is
embedded at compile time by `src/template/tests.rs`, so a change here is a
visible diff in a Rust gate — never an invisible test helper.

Usage:
    python3 scripts/gen_chat_fixtures.py            # rewrite the fixture
    python3 scripts/gen_chat_fixtures.py --check    # compare, exit 1 on drift

`--check` compares the `entries` array and the `generator` provenance string.
The `date` field records when the fixture was last regenerated; it is
informational and deliberately excluded from the comparison, otherwise the
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

FIXTURE = pathlib.Path(__file__).resolve().parent.parent / "tests/fixtures/chat/pystr_semantics.json"

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
        return 0

    date = FIXTURE.exists() and json.loads(FIXTURE.read_text()).get("provenance", {}).get("date")
    if not date:
        import datetime

        date = datetime.date.today().isoformat()
    doc = build(generate(), date)
    FIXTURE.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {FIXTURE} ({len(doc['entries'])} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
