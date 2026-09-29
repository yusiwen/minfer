#!/usr/bin/env python3
"""Audit the source-layout rules `AGENTS.md` states.

Two rules, both cheap to check and both easy to break silently:

1. **A test module is a file, never an inline block.** `#[cfg(test)] mod tests;`
   plus `<module>/tests.rs`. An inline `#[cfg(test)] mod tests { … }` in a
   production file is what the PR that introduced this checker moved out, one
   module at a time; nothing in rustc objects to it coming back, so this does.

2. **Every `.rs` under `src/` is declared.** A file that no `mod` declaration
   names is never compiled — and when that file holds tests, they silently do
   not run while `cargo test` and CI stay green. This is the dangerous half:
   rule 1 makes it easy to create `foo/tests.rs` and forget the `mod tests;` in
   `foo.rs`.

`#[path = "..."]` moves a module's file somewhere the declaration walk below
does not look, so a file carrying one is reported rather than guessed at.

Usage:

    python3 scripts/check_source_layout.py [--root src]
    python3 scripts/check_source_layout.py --selftest

Exit codes: 0 clean, 1 violations, 2 usage or I/O error. The checker is pure and
offline: stdlib only, no build, no network.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

#: Where the modules live, and the crate root the declaration walk starts from.
DEFAULT_ROOT = "src"
CRATE_ROOT = "main.rs"

#: `mod name;` — a declaration that reaches a file. `pub` / `pub(crate)`
#: prefixes count. Attributes are deliberately not matched: a `#[cfg(...)]`-gated
#: declaration still names the file, and reachability here asks "does anything
#: declare it", not "does it compile in this configuration".
MOD_DECL = re.compile(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?mod[ \t]+(\w+)[ \t]*;", re.M)

#: An inline test module — the shape rule 1 forbids. The wanted shape ends in a
#: semicolon, so the `{` is what makes this a violation. Whitespace (including a
#: newline) may separate the attribute from `mod`: both
#: `#[cfg(test)] mod tests {` and the two-line form are the same shape.
INLINE_TEST_MOD = re.compile(
    r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*mod[ \t]+(\w+)[ \t]*\{"
)

#: `#[path = "..."]` chooses a module's file by attribute.
PATH_ATTR = re.compile(r"#\[path[ \t]*=")


def code_lines(text: str) -> str:
    """`text` with whole-line `//` comments blanked, line numbers preserved.

    Both rules are regexes over source text, so a line that only *mentions* one
    of the shapes in prose must not count. (`/* */` blocks are not handled: the
    tree has none, and a checker that guesses is worse than one that says what
    it does not know.)
    """
    return "\n".join("" if ln.lstrip().startswith("//") else ln for ln in text.splitlines())


def module_dir(path: str) -> str:
    """The directory Rust looks in for `path`'s child modules.

    The crate root and a directory module (`mod.rs`, `lib.rs`) hold their
    children beside themselves; any other file `foo.rs` holds them in `foo/`.
    """
    d, b = os.path.split(path)
    return d if b in ("main.rs", "mod.rs", "lib.rs") else path[:-3]


def source_files(root: str) -> list[str]:
    out: list[str] = []
    for r, _, fs in os.walk(root):
        out += [os.path.join(r, f) for f in fs if f.endswith(".rs")]
    return sorted(out)


def check(root: str = DEFAULT_ROOT) -> list[str]:
    """Every layout violation under `root`, as `path:line: message`."""
    files = source_files(root)
    if not files:
        return [f"{root}: no .rs files"]
    known = set(files)
    problems: list[str] = []

    for f in files:
        code = code_lines(open(f, encoding="utf-8").read())
        for m in INLINE_TEST_MOD.finditer(code):
            line = code[: m.start()].count("\n") + 1
            problems.append(
                f"{f}:{line}: inline `#[cfg(test)] mod {m.group(1)} {{` — move it to "
                f"{module_dir(f)}/{m.group(1)}.rs and declare "
                f"`#[cfg(test)] mod {m.group(1)};` (AGENTS.md, Layout)"
            )
        for m in PATH_ATTR.finditer(code):
            line = code[: m.start()].count("\n") + 1
            problems.append(
                f"{f}:{line}: `#[path = ...]` — this checker cannot resolve a module "
                f"whose file is chosen by an attribute; declare it by name instead"
            )

    crate_root = os.path.join(root, CRATE_ROOT)
    if crate_root not in known:
        return problems + [f"{crate_root}: crate root missing"]

    seen: set[str] = set()
    stack = [crate_root]
    while stack:
        f = stack.pop()
        if f in seen:
            continue
        seen.add(f)
        d = module_dir(f)
        for name in MOD_DECL.findall(code_lines(open(f, encoding="utf-8").read())):
            for cand in (f"{d}/{name}.rs", f"{d}/{name}/mod.rs"):
                if cand in known:
                    stack.append(cand)
                    break

    for f in files:
        if f not in seen:
            problems.append(
                f"{f}: not reachable from {crate_root} — no `mod` declaration names it, "
                f"so it is never compiled (tests in it would silently not run)"
            )
    return problems


def selftest() -> int:
    """Pin the checker against the shapes it must and must not flag.

    Each case is `(name, tree, expected_problem_count, substrings)`: the count is
    asserted as well as the wording, so a case cannot pass by reporting the right
    words for the wrong number of problems.
    """
    cases: list[tuple[str, dict[str, str], int, list[str]]] = [
        (
            "clean tree: file modules, `pub mod`, a directory module",
            {
                "main.rs": "mod a;\nmod b;\nfn main() {}\n",
                "a.rs": "pub fn f() {}\n#[cfg(test)]\nmod tests;\n",
                "a/tests.rs": "#[test]\nfn t() {}\n",
                "b/mod.rs": "pub mod c;\n#[cfg(test)]\nmod tests;\n",
                "b/c.rs": "pub fn g() {}\n",
                "b/tests.rs": "#[test]\nfn t() {}\n",
            },
            0,
            [],
        ),
        (
            "an inline `#[cfg(test)] mod tests {` is a violation",
            {
                "main.rs": "mod a;\nfn main() {}\n",
                "a.rs": "pub fn f() {}\n#[cfg(test)]\nmod tests {\n    #[test]\n    fn t() {}\n}\n",
            },
            1,
            ["a.rs:2", "inline `#[cfg(test)] mod tests {`", "a/tests.rs"],
        ),
        (
            "a one-line inline test module is the same violation",
            {
                "main.rs": "mod a;\nfn main() {}\n",
                "a.rs": "pub fn f() {}\n#[cfg(test)] mod tests { #[test] fn t() {} }\n",
            },
            1,
            ["a.rs:2", "inline `#[cfg(test)] mod tests {`"],
        ),
        (
            "an undeclared file is a violation (its tests would not run)",
            {
                "main.rs": "mod a;\nfn main() {}\n",
                "a.rs": "pub fn f() {}\n",
                "a/tests.rs": "#[test]\nfn t() {}\n",
            },
            1,
            ["a/tests.rs", "not reachable from"],
        ),
        (
            "a `mod` named only in a comment declares nothing",
            {
                "main.rs": "mod a;\nfn main() {}\n",
                "a.rs": "// See `mod helper;` in the old tree.\npub fn f() {}\n",
                "a/helper.rs": "pub fn h() {}\n",
            },
            1,
            ["a/helper.rs: not reachable from"],
        ),
        (
            "`#[path = ...]` is reported, and its file is then unsurprising-orphan",
            {
                "main.rs": "#[path = \"other.rs\"]\nmod a;\nfn main() {}\n",
                "other.rs": "pub fn f() {}\n",
            },
            2,
            ["main.rs:1: `#[path = ...]`", "other.rs: not reachable from"],
        ),
    ]

    failures: list[str] = []
    for name, tree, want_count, want in cases:
        with tempfile.TemporaryDirectory() as d:
            root = os.path.join(d, "src")
            os.makedirs(root)
            for rel, text in tree.items():
                p = os.path.join(root, rel)
                os.makedirs(os.path.dirname(p), exist_ok=True)
                with open(p, "w", encoding="utf-8") as fh:
                    fh.write(text)
            got = check(root)
        if len(got) != want_count:
            failures.append(
                f"{name}: expected {want_count} problem(s), got {len(got)}: {got}"
            )
            continue
        for w in want:
            if not any(w in g for g in got):
                failures.append(f"{name}: no problem mentions {w!r}: {got}")

    if failures:
        for f in failures:
            print(f"selftest: {f}", file=sys.stderr)
        return 1
    print(f"selftest: {len(cases)} cases pass")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=DEFAULT_ROOT, help=f"source root (default {DEFAULT_ROOT})")
    ap.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()
    if not os.path.isdir(args.root):
        print(f"check_source_layout: {args.root}: not a directory", file=sys.stderr)
        return 2

    problems = check(args.root)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(f"check_source_layout: {len(problems)} violation(s)", file=sys.stderr)
        return 1
    print(
        f"check_source_layout: {args.root} obeys the layout rules "
        f"(test modules are files, and every .rs is declared)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
