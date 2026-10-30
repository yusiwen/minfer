#!/usr/bin/env python3
"""Fail on a derived count in a live document that does not name its measurement point.

`docs/GATE-CONTRACT.md` §"Counts" states the rule: a live document describes structure,
and a count that is the point of a record carries the revision it measured. No gate can
check that a number is *right* — this one checks that the sentence says which tree it is
right about.

The evidence that a prose convention is not enough is one ticket pair. #219 corrected a
batch of these numbers on 2026-10-04; #636 found three of them wrong again sixteen days
later (the always-loaded index was 27% low, `src/cuda.rs` two lines short, the backend
170 lines short, and a test count from before the suite split). The doc gates could not
see any of it: `check_doc_line_anchors.py` proves a `path:NNN` resolves, never that the
number beside it is current.

What counts as a claim:

  - a repository path next to a line count, in either order
    (`` `src/cuda.rs` (1,014 lines) ``, `` 2,351 lines in `cuda_backend.rs` ``);
  - a bare size of the tree — `` 70,797 lines of Rust ``, `` 44,105 LOC ``,
    `` ~55k LOC ``;
  - a line count of 1,000 or more that names no path at all — `` across 17
    translation units, 10,326 lines ``, `` the 4,546-line `impl` `` (#700). Three
    digits stay out deliberately: "the kernel is 5 lines" is prose about a sample,
    not a measurement of the tree;
  - a count of test functions — `` 39 test functions ``.

The arms are tried in order and the first match names the claim, so a count that
does sit beside a path keeps the more specific "path + line count" label.

A claim passes when its own line or either neighbour carries a **stamp**: a commit
(`` at `8c14286` ``, `` at commit 8c14286 ``, or a bare backticked 7–40-char hex) or an
ISO date (the shape the dated records use). A fenced block is a sample, not a claim.

A document that is deliberately a record is listed in `scripts/doc-counts.toml` beside
this checker (ADR-0023) with a glob and a reason. The ledger is a ratchet in both
directions: a claim no stamp and no entry cover is a failure, and an entry that exempts
nothing is a failure too — it must go in the commit that fixes the last claim it covers.

`--report` prints the census and exits 0: that is the adoption mode, and the mode a
reviewer runs before re-stamping anything.

Usage:

    python3 scripts/check_doc_counts.py [--root .] [--ledger FILE] [--report] [--list]
    python3 scripts/check_doc_counts.py --selftest

Exit codes: 0 clean, 1 a violation, 2 usage or I/O error. Pure and offline: stdlib only,
no build, no network.
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

import docset

LEDGER = "scripts/doc-counts.toml"

#: A repository-owned path token, the same extension set `check_doc_line_anchors.py` owns.
PATH_TOKEN = r"[\w./-]+\.(?:rs|cu|cuh|metal|toml|py|md|sh|yml|yaml)"

#: The claim shapes. Name first, so a failure says *what kind* of number it saw. The
#: two path+count arms are separate because either order is written in the docs, and
#: a single regex with a lookbehind would miss the wrapped line cases.
CLAIM_PATTERNS = (
    (
        "path + line count",
        re.compile(rf"{PATH_TOKEN}[^\n]{{0,60}}?\b\d[\d,]*\s+lines\b"),
    ),
    (
        "line count + path",
        re.compile(rf"\b\d[\d,]*\s+lines\b[^\n]{{0,60}}?{PATH_TOKEN}"),
    ),
    ("tree size", re.compile(r"\b\d[\d,]*\s+(?:lines of (?:Rust|tests)|LOC)\b")),
    ("~Nk size", re.compile(r"~\s?\d+(?:\.\d+)?k\s+(?:lines|LOC)\b")),
    (
        # #700: a line count of 1,000 or more is a size of the tree wherever it is written.
        # Beside a *file* count instead of a path ("across 17 translation units,
        # 10,326 lines") neither path arm can see it, which is how four stale claims
        # in the CUDA tutorial survived #641's gate. The hyphenated compound ("the
        # 4,546-line `impl`") is the same claim one word later. Three digits stay out:
        # "the kernel is 5 lines" is prose about a sample, not a measurement.
        "line count, no path",
        re.compile(r"\b(?:[1-9]\d{0,2}(?:,\d{3})+|[1-9]\d{3,})(?:\s+lines|-line)\b"),
    ),
    ("test-function count", re.compile(r"\b\d+\s+test functions?\b")),
)

#: What makes a claim a snapshot: it names the revision (or at least the date) it was
#: measured against. A bare backticked hex is enough — that is how the docs cite a
#: revision — and so is an ISO date, which the dated records use instead of a SHA.
STAMP_PATTERNS = (
    re.compile(r"\bat commit\s+`?[0-9a-f]{7,40}\b"),
    re.compile(r"\bat\s+`[0-9a-f]{7,40}`"),
    re.compile(r"`[0-9a-f]{7,40}`"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
)

#: A fenced block delimiter, read the way the other doc gates read it: its content is a
#: sample of code or output, not a claim about the tree.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: How far into a document its own measurement stamp may sit. The walkthrough pages open
#: with a `Code:` block that ends `lines verified at commit <sha>`, and a `Sizes:` line
#: added in the same place says the same about the counts; a document that declares its
#: revision at the top is a snapshot of that revision, exactly like a ledgered record.
DOC_STAMP_WINDOW = 30


class LedgerError(Exception):
    """The ledger is missing, unreadable or malformed — a usage/I/O error, never a verdict."""


@dataclass
class Claim:
    """One derived count, and the ledger entry that exempts it (if any)."""

    doc: str
    line: int
    kind: str
    text: str
    exempt: str | None = None

    def where(self) -> str:
        return f"{self.doc}:{self.line}"


@dataclass
class Report:
    claims: list[Claim] = field(default_factory=list)
    #: Every claim the scan saw, stamped or not — the census needs both halves.
    seen: int = 0
    #: Entries that exempted no claim: the ledger may not rot.
    stale: list[tuple[str, str]] = field(default_factory=list)

    def live(self) -> list[Claim]:
        return [c for c in self.claims if c.exempt is None]


def load_ledger(path: Path) -> list[tuple[str, str]]:
    """The `[[exclude]]` entries: `(glob, reason)`, both required and non-empty."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LedgerError(f"cannot read {path}: {error.strerror or error}") from error
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LedgerError(f"{path}: {error}") from error
    entries: list[tuple[str, str]] = []
    for position, entry in enumerate(data.get("exclude", []), start=1):
        glob = str(entry.get("glob", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        if not glob or not reason:
            raise LedgerError(
                f"{path}: exclude #{position} needs a non-empty `glob` and `reason`"
            )
        entries.append((glob, reason))
    return entries


def stamped(lines: list[str], index: int) -> bool:
    """Whether the claim at `lines[index]` names its measurement point (or a neighbour does)."""
    window = "\n".join(lines[max(0, index - 1) : index + 2])
    return any(pattern.search(window) for pattern in STAMP_PATTERNS)


def scan(root: Path, files: list[Path], entries: list[tuple[str, str]]) -> Report:
    """Every claim in `files`, with its ledger exemption; plus the stale entries."""
    report = Report()
    covered = {glob: False for glob, _reason in entries}
    for path in files:
        rel = path.relative_to(root).as_posix()
        exempt = next(
            (reason for glob, reason in entries if fnmatch.fnmatch(rel, glob)), None
        )
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        header = "\n".join(lines[:DOC_STAMP_WINDOW])
        doc_exempt = (
            "the document declares its revision in its header"
            if any(pattern.search(header) for pattern in STAMP_PATTERNS)
            else None
        )
        fence: str | None = None
        for index, text in enumerate(lines):
            opener = FENCE.match(text)
            if fence is not None:
                if opener is not None and opener.group(1)[0] == fence[0]:
                    fence = None
                continue
            if opener is not None:
                fence = opener.group(1)
                continue
            kind = next(
                (name for name, pattern in CLAIM_PATTERNS if pattern.search(text)), None
            )
            if kind is None:
                continue
            report.seen += 1
            if stamped(lines, index):
                continue
            claim = Claim(
                doc=rel,
                line=index + 1,
                kind=kind,
                text=text.strip(),
                exempt=exempt or doc_exempt,
            )
            report.claims.append(claim)
            if exempt is not None:
                glob = next(g for g, reason in entries if reason == exempt)
                covered[glob] = True
    for glob, reason in entries:
        if not covered[glob]:
            report.stale.append((glob, reason))
    return report


def check(
    root: Path,
    ledger: Path,
    files: list[Path] | None = None,
    report_out: Report | None = None,
) -> int:
    """The whole check; returns the exit code (0 clean, 1 violations, 2 ledger error)."""
    try:
        entries = load_ledger(ledger)
    except LedgerError as error:
        print(f"check_doc_counts: {error}", file=sys.stderr)
        return 2
    if files is None:
        files = docset.scanned(root)
    report = scan(root, files, entries)
    if report_out is not None:
        report_out.claims = report.claims
        report_out.seen = report.seen
        report_out.stale = report.stale
    return 1 if report.live() or report.stale else 0


def print_report(report: Report) -> None:
    for claim in report.live():
        print(f"{claim.where()} [{claim.kind}] {claim.text[:120]}")
    for glob, reason in report.stale:
        print(
            f"STALE LEDGER  {glob}: exempts no claim any more — remove the entry "
            f"(reason was: {reason})",
            file=sys.stderr,
        )
    print(
        f"check_doc_counts: {report.seen} claim(s) seen, "
        f"{len(report.live())} without a stamp, {len(report.stale)} stale ledger entry/entries"
    )


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool, str]] = []
    EX = '[[exclude]]\nglob = "docs/record/*.md"\nreason = "a dated record"\n'

    def run(docs: dict[str, str], ledger: str = "") -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, body in docs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
            path = root / LEDGER
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(ledger, encoding="utf-8")
            return check(
                root,
                path,
                files=[root / name for name in docs],
            )

    cases.append(
        (
            "an unstamped path + line count fails",
            run({"docs/a.md": "`src/cuda.rs` (1,014 lines) — the state.\n"}) == 1,
            "",
        )
    )
    cases.append(
        (
            "a line count stamped at a commit passes",
            run({"docs/a.md": "`src/cuda.rs` (1,016 lines at `38633fb`) — the state.\n"}) == 0,
            "",
        )
    )
    cases.append(
        (
            "a line count stamped with a date passes",
            run({"docs/a.md": "The 2026-10-04 audit measured 2,351 lines there.\n"}) == 0,
            "",
        )
    )
    cases.append(
        (
            "a tree-size claim fails",
            run({"docs/a.md": "The tree is ~55k LOC of Rust.\n"}) == 1,
            "",
        )
    )
    cases.append(
        (
            "a line count that names no path fails",
            run({"docs/a.md": "It spans 17 translation units, 10,326 lines.\n"}) == 1,
            "",
        )
    )
    cases.append(
        (
            "the same claim stamped on its own line passes",
            run({"docs/a.md": "It spans 17 translation units, 10,326 lines at `f26e7e1`.\n"})
            == 0,
            "",
        )
    )
    cases.append(
        (
            "a stamp on the neighbouring line covers it",
            run({"docs/a.md": "Measured at\n`f26e7e1`: 17 translation units, 10,326 lines.\n"})
            == 0,
            "",
        )
    )
    cases.append(
        (
            "the hyphenated form is the same claim",
            run({"docs/a.md": "`src/cuda.rs` loses its 4,546-line `impl`.\n"}) == 1,
            "",
        )
    )
    cases.append(
        (
            "a line count below 1,000 is not a claim",
            run({"docs/a.md": "The kernel is 5 lines; the file is 842 lines.\n"}) == 0,
            "",
        )
    )
    cases.append(
        (
            "the pre-#637 tutorial sentence fails",
            run(
                {
                    "docs/a.md": "Today that prints **99 `__global__ void` definitions** — plus "
                    "six prose mentions —\nacross **17 files** (10,147 lines). Nobody memorizes "
                    "99 entries; you navigate by\nfamily.\n"
                }
            )
            == 1,
            "",
        )
    )
    cases.append(
        (
            "a test-function count fails",
            run({"docs/a.md": "It carries 39 test functions.\n"}) == 1,
            "",
        )
    )
    cases.append(
        (
            "a structural fact is not a claim",
            run({"docs/a.md": "`common.cuh` + 17 translation units, one per family.\n"}) == 0,
            "",
        )
    )
    cases.append(
        (
            "a fenced sample is not a claim",
            run({"docs/a.md": "```\n`src/cuda.rs` (1,014 lines)\n```\n"}) == 0,
            "",
        )
    )
    cases.append(
        (
            "a document that declares its revision in its header is exempt",
            run(
                {
                    "docs/a.md": "> **Sizes**: measured at `c252e24`.\n\n"
                    "`src/cuda.rs` (1,016 lines).\n"
                }
            )
            == 0,
            "",
        )
    )
    cases.append(
        (
            "a ledgered document is exempt",
            run(
                {"docs/record/day.md": "`src/cuda.rs` was 1,014 lines.\n"},
                EX,
            )
            == 0,
            "",
        )
    )
    cases.append(
        (
            "a ledger entry that exempts nothing fails",
            run({"docs/a.md": "`src/cuda.rs` (1,016 lines at `38633fb`).\n"}, EX) == 1,
            "",
        )
    )
    with contextlib.redirect_stderr(io.StringIO()):
        # The checker prints the ledger error it returns; the case asserts the code.
        malformed = run(
            {"docs/a.md": "nothing to see\n"},
            '[[exclude]]\nglob = "docs/a.md"\nreason = ""\n',
        )
    cases.append(
        ("a ledger entry without a reason is a usage error", malformed == 2, "")
    )
    failures = [name for name, ok, _ in cases if not ok]
    for name in failures:
        print(f"SELFTEST FAIL  {name}", file=sys.stderr)
    if failures:
        print(f"check_doc_counts --selftest: {len(failures)} case(s) failed", file=sys.stderr)
        return 1
    print("check_doc_counts --selftest: every case passes (the checker bites)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="repository root (default: this tree)")
    parser.add_argument(
        "--ledger",
        default=None,
        help=f"the exclusion ledger, relative to --root (default: {LEDGER})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the census and exit 0 (adoption mode; the ledger ratchet still prints)",
    )
    parser.add_argument("--list", action="store_true", help="print every claim, exempt ones too")
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    if not (root / "docs").is_dir():
        print(f"check_doc_counts: no docs/ under {root}", file=sys.stderr)
        return 2
    ledger = Path(args.ledger) if args.ledger else root / LEDGER
    if not ledger.is_absolute():
        ledger = root / ledger

    try:
        docset.announce(root)
    except docset.ScopeError as error:
        print(f"check_doc_counts: {error}", file=sys.stderr)
        return 1

    report = Report()
    code = check(root, ledger, report_out=report)
    if args.list:
        for claim in report.claims:
            tag = f" [ledger: {claim.exempt}]" if claim.exempt else ""
            print(f"{claim.where()} [{claim.kind}]{tag} {claim.text[:120]}")
    print_report(report)
    if args.report:
        return 0
    if code == 1:
        print(
            "check_doc_counts: state structure, or stamp the claim at the revision it "
            "measured (see docs/GATE-CONTRACT.md §Counts)",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
