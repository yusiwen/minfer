#!/usr/bin/env python3
"""Keep `AGENTS.md` small enough to be an index rather than a document.

`AGENTS.md` is loaded into every agent session as workspace instructions, and the
workspace imposes a byte budget on the whole instruction set. An over-long
`AGENTS.md` therefore does not merely read badly: it pushes the *other*
instruction files out of the session entirely. It had reached 72,692 B, at which
point the global `~/.dsh/AGENTS.md` — the long-form parallel-worktree rules
`scripts/agent_worktree.sh` points at — was dropped from the session by the
budget, unnoticed.

The cap enforces the file's own rule, stated at its line 7: *"This file is the
always-loaded index. Deep dives live in `docs/` — don't duplicate them here."*
The heavy blocks belong to their topic documents (`docs/KV-CACHE-DESIGN.md`,
`docs/GATE-CONTRACT.md`, `docs/SUPPORT-MATRIX.md`, …), and `AGENTS.md` links to
them.

A cap is a ratchet, not a target. Raising `CAP` is allowed, but it is a decision:
raise it in the same commit as the growth and say in the commit message which
section grew and why it cannot live in `docs/`. Prefer moving the block.

Usage:

    python3 scripts/check_agents_size.py [--root .] [--list]
    python3 scripts/check_agents_size.py --selftest

Exit codes: 0 within the cap, 1 over the cap, 2 usage or I/O error. The checker
is pure and offline: stdlib only, no build, no network.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

# The ratchet. 24 KiB leaves the file room to be a router (routing, commands,
# numbered invariants one line each, a short doc index) without room to grow back
# into a document. Measured at 24,516 B when this landed (issue #434 S3), so the
# headroom is deliberately thin: a *section* of new prose fails the gate, and the
# two sanctioned responses are to move a block to its topic doc (preferred) or to
# raise this cap in the commit that needs it. A cap with slack would let the file
# drift back one paragraph at a time, which is how it reached 72,692 B.
CAP = 24_576

HEADING = re.compile(r"^## (?!#)")


def sections(text: str) -> list[tuple[str, int]]:
    """Every `##` section as `(title, bytes)`, plus the preamble.

    Bytes are counted per line as written, including the terminating newline, so
    the sections sum to the file size.
    """
    lines = text.split("\n")
    out: list[tuple[str, int]] = []
    title = "(preamble)"
    total = 0
    for i, line in enumerate(lines):
        if HEADING.match(line):
            out.append((title, total))
            title, total = line[3:].strip(), 0
        # the split leaves a trailing "" for a file ending in a newline; only the
        # lines before it are terminated, so the sections sum to the file size
        total += len(line.encode("utf-8")) + (1 if i < len(lines) - 1 else 0)
    out.append((title, total))
    return out


def report(text: str, path: Path) -> list[str]:
    """Every problem with `path`'s size, plus the per-section accounting."""
    problems: list[str] = []
    size = len(text.encode("utf-8"))
    problems.append(f"check_agents_size: {path} is {size} B (cap {CAP} B)")
    ranked = sorted(sections(text), key=lambda item: item[1], reverse=True)
    for title, n in ranked:
        if n:
            problems.append(f"  {n:>7} B  {100 * n / size:5.1f}%  {title}")
    return problems


def check(root: Path) -> int:
    path = root / "AGENTS.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"check_agents_size: cannot read {path}: {exc}", file=sys.stderr)
        return 2
    size = len(text.encode("utf-8"))
    lines = report(text, path)
    biggest = max(sections(text), key=lambda item: item[1])
    if size > CAP:
        for line in lines:
            print(line)
        print(
            f"check_agents_size: {path} is {size - CAP} B over the cap — move the largest "
            f"section ({biggest[0]!r}, {biggest[1]} B) to the doc that owns it and link it, "
            f"or raise CAP in this commit and say why"
        )
        return 1
    print(
        f"check_agents_size: {path} is {size} B (cap {CAP} B, {CAP - size} B headroom); "
        f"largest section {biggest[0]!r} at {biggest[1]} B"
    )
    return 0


def selftest() -> int:
    """The checker's own pass/fail cases, in-process."""
    cases: list[tuple[str, bool, str]] = []
    small = "# Title\n\n## Layout\n\nsmall\n"
    big = "# Title\n\n## Layout\n\n" + ("x" * (CAP + 10)) + "\n"
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "AGENTS.md").write_text(small, encoding="utf-8")
        cases.append(("a file within the cap passes", check(root) == 0, ""))
        (root / "AGENTS.md").write_text(big, encoding="utf-8")
        cases.append(("a file over the cap fails", check(root) == 1, ""))
    # the section walk must attribute every byte, including the preamble
    parsed = sum(n for _, n in sections(small))
    cases.append(("the sections sum to the file size", parsed == len(small.encode()), f"{parsed}"))
    ranked = sorted(sections(big), key=lambda item: item[1], reverse=True)
    cases.append(("the largest section is named", ranked[0][0] == "Layout", ranked[0][0]))
    with tempfile.TemporaryDirectory() as tmp:
        cases.append(("a missing file is an I/O error", check(Path(tmp)) == 2, ""))
    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_agents_size --selftest: {len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Keep AGENTS.md small enough to be an index.")
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument("--list", action="store_true", help="print the per-section report and exit 0")
    ap.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    root = Path(args.root)
    if args.list:
        path = root / "AGENTS.md"
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"check_agents_size: cannot read {path}: {exc}", file=sys.stderr)
            return 2
        for line in report(text, path):
            print(line)
        return 0
    return check(root)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
