#!/usr/bin/env python3
"""Fail on a hand-copied source listing in a live document.

A fenced block that opens with a comment naming a repository source file is a **transcription
claim**: the page says "this is `src/…`, as it was when I copied it". #219 and #645 both found the
same rot — a copy that had stopped being true — and the doc gates cannot read a code block the way
they read a `path:NNN` anchor.

The fix for a claim is to stop copying: `{{#rustdoc_include path:anchor}}` renders the source itself
(mdBook 0.4.37; `{{#rustdoc_include}}` rather than `{{#include}}` because `check_figures.py` claims
every plain include for figures). This checker enforces the direction of travel:

- a claim in a document with no ledger entry fails — a new copy;
- more claims than the entry records fails — a new copy in a migrated page;
- **fewer** claims than the entry records fails too — a conversion must lower the number in the
  same commit, so the population can only shrink;
- an entry whose document carries no claim fails — the ledger may not rot.

The ledger is `scripts/doc-transcriptions.toml`, beside this checker (ADR-0023), one `[[doc]]` entry
per page with its current count and a reason. It is a migration list, not a permission: every entry
should disappear as its page is converted.

Usage:

    python3 scripts/check_doc_code_copies.py [--root .] [--ledger FILE] [--list]
    python3 scripts/check_doc_code_copies.py --selftest

Exit codes: 0 clean, 1 a violation, 2 usage or I/O error. Pure and offline: stdlib only.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import re
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import check_doc_line_anchors as anchors
import docset

LEDGER = "scripts/doc-transcriptions.toml"

#: A fenced block's delimiter.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: The provenance claim: the first non-blank line, optionally behind a comment marker, naming a
#: repository source path (`// src/models/qwen2/loader.rs:318 (\`load\`)`). A path that does not
#: resolve in this tree is another tree's citation and is ignored.
CLAIM = re.compile(
    r"^\s*(?://+[!/]?\s*|#\s*|--\s*)?`?"
    r"(?P<path>[\w./-]+\.(?:rs|cu|cuh|metal|py|sh|toml|yml|yaml))"
    r"(?::\d+(?:-\d+)?)?"
)


class LedgerError(Exception):
    """The ledger is missing, unreadable or malformed — usage/I/O, never a verdict."""


@dataclass
class Claim:
    doc: str
    line: int
    path: str

    def where(self) -> str:
        return f"{self.doc}:{self.line}"


@dataclass
class Report:
    claims: list[Claim] = field(default_factory=list)
    unlisted: list[str] = field(default_factory=list)
    #: `(doc, recorded, actual)` — both directions are violations (the ratchet).
    mismatched: list[tuple[str, int, int]] = field(default_factory=list)
    stale: list[tuple[str, int]] = field(default_factory=list)

    def problems(self) -> int:
        return len(self.unlisted) + len(self.mismatched) + len(self.stale)


def repo_paths(root: Path) -> list[str]:
    """Repository-relative file paths, pruned like the other checkers."""
    paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in (".git", ".worktrees", "target", "book"):
            continue
        paths.append(rel.as_posix())
    return paths


def resolves(token: str, paths: list[str]) -> bool:
    """Whether `token` names a file of this tree (exact, by suffix, or by unique basename)."""
    if token in paths:
        return True
    if "/" in token:
        return any(path.endswith("/" + token) for path in paths)
    return sum(1 for path in paths if path.rsplit("/", 1)[-1] == token) == 1


def load_ledger(path: Path) -> dict[str, tuple[int, str]]:
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LedgerError(f"cannot read {path}: {error.strerror or error}") from error
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LedgerError(f"{path}: {error}") from error
    entries: dict[str, tuple[int, str]] = {}
    for position, entry in enumerate(data.get("doc", []), start=1):
        doc = str(entry.get("path", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        count = entry.get("count")
        if not doc or not reason or not isinstance(count, int) or count < 1:
            raise LedgerError(
                f"{path}: entry #{position} needs a non-empty `path`, a `count` >= 1 and a `reason`"
            )
        entries[doc] = (count, reason)
    return entries


def scan(root: Path, files: list[Path], known: list[str]) -> list[Claim]:
    claims: list[Claim] = []
    for path in files:
        rel = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(rel, pattern) for pattern in anchors.FROZEN):
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        fence: str | None = None
        opening = 0
        body: list[str] = []
        for index, text in enumerate(lines + ["```"]):
            match = FENCE.match(text)
            if fence is None and match is not None:
                fence = match.group(1)
                opening = index + 2
                body = []
                continue
            if fence is not None and match is not None and match.group(1)[0] == fence[0]:
                fence = None
                content = [line for line in body if line.strip()]
                if content and "{{#" not in content[0]:
                    claim = CLAIM.match(content[0].strip())
                    if claim is not None and resolves(claim.group("path"), known):
                        claims.append(Claim(doc=rel, line=opening, path=claim.group("path")))
                continue
            if fence is not None:
                body.append(text)
    return claims


def check(root: Path, ledger: Path, report_out: Report | None = None) -> int:
    try:
        entries = load_ledger(ledger)
    except LedgerError as error:
        print(f"check_doc_code_copies: {error}", file=sys.stderr)
        return 2
    try:
        files = docset.scanned(root)
    except docset.ScopeError as error:
        print(f"check_doc_code_copies: {error}", file=sys.stderr)
        return 1
    claims = scan(root, files, repo_paths(root))
    actual: dict[str, int] = {}
    for claim in claims:
        actual[claim.doc] = actual.get(claim.doc, 0) + 1
    report = Report(claims=claims)
    for doc, count in actual.items():
        if doc not in entries:
            report.unlisted.append(doc)
        elif entries[doc][0] != count:
            report.mismatched.append((doc, entries[doc][0], count))
    for doc, (count, _reason) in entries.items():
        if doc not in actual:
            report.stale.append((doc, count))
    if report_out is not None:
        report_out.claims = report.claims
        report_out.unlisted = report.unlisted
        report_out.mismatched = report.mismatched
        report_out.stale = report.stale
    return 1 if report.problems() else 0


def print_report(report: Report) -> None:
    for doc in report.unlisted:
        print(f"UNLISTED  {doc}: carries a transcription claim and has no ledger entry")
    for doc, recorded, real in report.mismatched:
        if real > recorded:
            print(f"NEW CLAIM {doc}: {real} claim(s), ledger records {recorded} — include the listing")
        else:
            print(f"LOWER IT  {doc}: {real} claim(s), ledger records {recorded} — the count shrank")
    for doc, recorded in report.stale:
        print(f"STALE     {doc}: ledger records {recorded}, the document carries none")
    print(
        f"check_doc_code_copies: {len(report.claims)} claim(s) in "
        f"{len({claim.doc for claim in report.claims})} document(s), "
        f"{len(report.unlisted)} unlisted, {len(report.mismatched)} mismatched, "
        f"{len(report.stale)} stale"
    )


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool]] = []
    LEDGER_OK = '[[doc]]\npath = "docs/a.md"\ncount = 1\nreason = "a legacy copy"\n'

    def run(docs: dict[str, str], ledger: str) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in {**{"src/x.rs": "fn x() {}\n"}, **docs}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
            path = root / LEDGER
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ledger, encoding="utf-8")
            return check(root, path)

    COPY = "```rust\n// src/x.rs\nfn x() {}\n```\n"
    cases.append(("a claim in an unlisted document fails", run({"docs/a.md": COPY}, "") == 1))
    cases.append(("a ledgered claim passes", run({"docs/a.md": COPY}, LEDGER_OK) == 0))
    cases.append(
        (
            "a second claim in a ledgered document fails",
            run({"docs/a.md": COPY + COPY}, LEDGER_OK) == 1,
        )
    )
    cases.append(
        (
            "converting one without lowering the count fails",
            run(
                {"docs/a.md": COPY},
                '[[doc]]\npath = "docs/a.md"\ncount = 2\nreason = "two legacy copies"\n',
            )
            == 1,
        )
    )
    cases.append(
        (
            "an include is not a claim",
            run(
                {"docs/a.md": "```rust\n{{#rustdoc_include ../src/x.rs:x}}\n```\n"},
                "",
            )
            == 0,
        )
    )
    cases.append(
        (
            "another tree's path is not a claim",
            run({"docs/a.md": "```cuda\n// ggml-cuda.cu\n__global__ void k();\n```\n"}, "") == 0,
        )
    )
    cases.append(
        (
            "a ledger entry that covers nothing fails",
            run({"docs/a.md": "```rust\nfn y() {}\n```\n"}, LEDGER_OK) == 1,
        )
    )
    with contextlib.redirect_stderr(io.StringIO()):
        # The checker prints the ledger error it returns; the case asserts the code.
        malformed = run({"docs/a.md": COPY}, '[[doc]]\npath = "docs/a.md"\ncount = 0\nreason = ""\n')
    cases.append(("a malformed entry is a usage error", malformed == 2))
    failures = [name for name, ok in cases if not ok]
    for name in failures:
        print(f"SELFTEST FAIL  {name}", file=sys.stderr)
    if failures:
        print(
            f"check_doc_code_copies --selftest: {len(failures)} case(s) failed", file=sys.stderr
        )
        return 1
    print("check_doc_code_copies --selftest: every case passes (the checker bites)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="repository root (default: this tree)")
    parser.add_argument(
        "--ledger",
        default=None,
        help=f"the migration ledger, relative to --root (default: {LEDGER})",
    )
    parser.add_argument("--list", action="store_true", help="print every claim")
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    if not (root / "docs").is_dir():
        print(f"check_doc_code_copies: no docs/ under {root}", file=sys.stderr)
        return 2
    ledger = Path(args.ledger) if args.ledger else root / LEDGER
    if not ledger.is_absolute():
        ledger = root / ledger

    report = Report()
    code = check(root, ledger, report)
    if args.list:
        for claim in report.claims:
            print(f"{claim.where()} ← {claim.path}")
    print_report(report)
    if code == 1:
        print(
            "check_doc_code_copies: bind the listing with {{#rustdoc_include}} (an ANCHOR block in "
            "the source), or lower the ledger count for a conversion (docs/GATE-CONTRACT.md §Counts)",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
