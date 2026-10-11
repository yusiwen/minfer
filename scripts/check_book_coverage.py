#!/usr/bin/env python3
"""Fail on a `docs/` document that the book cannot publish, or publish by accident.

`docs/SUMMARY.md` is the book's table of contents, and mdBook treats it as the *only*
list of chapters: a `docs/**.md` file it does not name is not published — no page, no
copied source — and `mdbook build` still exits 0. The other direction is just as
quiet: a SUMMARY entry whose file does not exist is rendered as an empty **draft**
page, so a renamed document leaves a blank chapter behind and nothing says so. Both
were measured against the pinned mdBook 0.4.37 (#515).

`scripts/check_docs_links.py` cannot see either: the link, and the file it names,
both exist in the checkout. `scripts/check_book_links.py` sees the *consequence* for
a document the book links to, but a document nothing links to is invisible to it —
this checker is about the file set, not about the links.

A document that is deliberately not a chapter is listed in
`scripts/book-coverage.toml` beside this checker (ADR-0023) with a glob and a reason.
The ledger is a ratchet: a new `docs/**.md` that no chapter and no entry covers is a
failure, and an entry that no longer covers a file is a failure too. `docs/SUMMARY.md`
itself is one of those entries — mdBook consumes it and never publishes it, which is
a structural reason rather than a decision anybody has to revisit.

Usage:

    python3 scripts/check_book_coverage.py [--root .] [--ledger FILE] [--list]
    python3 scripts/check_book_coverage.py --selftest

Exit codes: 0 clean, 1 a violation, 2 usage or I/O error. Pure and offline: stdlib
only, no build, no network.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import shutil
import sys
import tempfile
import tomllib
from pathlib import Path

#: A chapter link in `SUMMARY.md`. Part headings (`- [Part I]()`) carry no target and
#: are skipped, which is why an empty match is not a violation.
CHAPTER_LINK = re.compile(r"\]\(([^)\s]+)\)")
LEDGER = "scripts/book-coverage.toml"


class Report:
    """What the scan found, so the checker can print it and the selftest assert it."""

    def __init__(self) -> None:
        self.chapters: set[str] = set()
        self.missing: list[str] = []  # SUMMARY entries with no file
        self.outside: list[str] = []  # SUMMARY entries outside the book's src
        self.docs: list[str] = []  # every docs/**.md
        self.unlisted: list[str] = []  # docs/**.md that is not a chapter


def scan(root: Path, src: str) -> Report:
    """Read `SUMMARY.md` and the `src/**.md` file set under `root`."""
    report = Report()
    summary = root / src / "SUMMARY.md"
    if not summary.is_file():
        raise FileNotFoundError(f"{summary} is missing — the book has no table of contents")
    text = summary.read_text(encoding="utf-8")
    src_dir = (root / src).resolve()
    for match in CHAPTER_LINK.finditer(text):
        target = match.group(1).split("#", 1)[0]
        if not target or not target.endswith(".md"):
            continue  # a part heading, an image, or an external link
        resolved = (summary.parent / target).resolve()
        rel = resolved.relative_to(root).as_posix() if resolved.is_relative_to(root) else target
        if not resolved.is_relative_to(src_dir):
            report.outside.append(rel)
        elif not resolved.is_file():
            report.missing.append(rel)
        else:
            report.chapters.add(rel)
    report.docs = sorted(
        p.relative_to(root).as_posix() for p in (root / src).rglob("*.md")
    )
    report.unlisted = [d for d in report.docs if d not in report.chapters]
    return report


def load_ledger(path: Path) -> list[tuple[str, str]]:
    """`(glob, reason)` for every excluded document."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: list[tuple[str, str]] = []
    for entry in data.get("exclude", []):
        pattern = entry.get("glob", "")
        reason = entry.get("reason", "")
        if not pattern or not reason:
            raise ValueError(f"{path}: an entry needs both `glob` and `reason`")
        out.append((pattern, reason))
    return out


def check(root: Path, ledger_path: Path, list_all: bool = False) -> int:
    """Print the verdict; return the exit code."""
    if not ledger_path.is_file():
        print(f"check_book_coverage: {ledger_path} is missing", file=sys.stderr)
        return 2
    try:
        ledger = load_ledger(ledger_path)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"check_book_coverage: {exc}", file=sys.stderr)
        return 2
    src = source_root(root)
    try:
        report = scan(root, src)
    except FileNotFoundError as exc:
        print(f"check_book_coverage: {exc}", file=sys.stderr)
        return 2

    patterns = [g for g, _ in ledger]
    uncovered = [
        d for d in report.unlisted if not any(fnmatch.fnmatchcase(d, p) for p in patterns)
    ]
    stale = [
        (pattern, reason)
        for pattern, reason in ledger
        if not any(fnmatch.fnmatchcase(d, pattern) for d in report.unlisted)
    ]

    if list_all:
        for doc in report.unlisted:
            cover = next((p for p in patterns if fnmatch.fnmatchcase(doc, p)), "(uncovered)")
            print(f"OUTSIDE {doc}  [{cover}]")
    for doc in uncovered:
        print(f"ORPHAN DOC  {doc}", file=sys.stderr)
        print(
            f"            not a chapter and no entry in {ledger_path} covers it — "
            f"mdBook publishes no page for it and `mdbook build` still exits 0",
            file=sys.stderr,
        )
    for doc in report.missing:
        print(f"MISSING CHAPTER  {src}/SUMMARY.md: {doc}", file=sys.stderr)
        print(
            "                 the file does not exist, so mdBook renders an empty draft "
            "page under this title and still exits 0",
            file=sys.stderr,
        )
    for doc in report.outside:
        print(f"OUTSIDE SRC  {src}/SUMMARY.md: {doc}", file=sys.stderr)
        print(
            f"             a chapter must live under the book's `{src}/` — mdBook copies "
            "only what is inside its source directory",
            file=sys.stderr,
        )
    for pattern, reason in stale:
        print(
            f"STALE EXCLUDE  {ledger_path}: `{pattern}` covers nothing any more",
            file=sys.stderr,
        )
        print(f"               (reason was: {reason})", file=sys.stderr)

    problems = len(uncovered) + len(report.missing) + len(report.outside) + len(stale)
    if problems:
        print(
            f"check_book_coverage: {len(uncovered)} uncovered document(s), "
            f"{len(report.missing)} missing chapter(s), {len(report.outside)} chapter(s) "
            f"outside `{src}/`, {len(stale)} stale ledger entry(ies)",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_book_coverage: {len(report.chapters)} chapters, "
        f"{len(report.unlisted)} document(s) outside them all covered by "
        f"{len(ledger)} ledger entries, 0 uncovered, 0 stale"
    )
    return 0


def source_root(root: Path) -> str:
    """The book's own `src` directory, from `book.toml` (default `docs`)."""
    config = root / "book.toml"
    if config.is_file():
        try:
            with config.open("rb") as fh:
                src = tomllib.load(fh).get("book", {}).get("src")
            if isinstance(src, str) and src:
                return src
        except tomllib.TOMLDecodeError:
            pass
    return "docs"


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool, str]] = []
    # Every fixture writes `docs/SUMMARY.md`, which is never a chapter, so the default
    # ledger carries the structural entry the real one also has.
    EX = '[[exclude]]\nglob = "docs/SUMMARY.md"\nreason = "the table of contents"\n'
    B_EXCLUDE = '[[exclude]]\nglob = "docs/b.md"\nreason = "a document under test"\n'

    def fixture(tmp: str, summary: str, docs: dict[str, str], ledger: str) -> tuple[Path, Path]:
        root = Path(tmp)
        shutil.rmtree(root, ignore_errors=True)  # cases share one temp dir
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / "docs" / "SUMMARY.md").write_text(summary, encoding="utf-8")
        for name, body in docs.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        path = root / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ledger, encoding="utf-8")
        return root, path

    def run(tmp: str, summary: str, docs: dict[str, str], ledger: str = EX) -> int:
        root, path = fixture(tmp, summary, docs, ledger)
        return check(root, path)

    with tempfile.TemporaryDirectory() as tmp:
        summary = "# Summary\n\n- [A](./a.md)\n"
        docs = {"docs/a.md": "# A\n"}
        cases.append(("every document is a chapter", run(tmp, summary, docs) == 0, ""))
        docs["docs/b.md"] = "# B\n"
        cases.append(
            ("a document that is not a chapter fails", run(tmp, summary, docs) == 1, "")
        )
        cases.append(
            (
                "an excluded document passes",
                run(tmp, summary, docs, EX + B_EXCLUDE) == 0,
                "",
            )
        )
        cases.append(
            (
                "an exclude that covers nothing fails",
                run(tmp, summary, {"docs/a.md": "# A\n"}, EX + B_EXCLUDE) == 1,
                "",
            )
        )
        cases.append(
            (
                "a chapter whose file is missing fails",
                run(
                    tmp,
                    "# Summary\n\n- [Gone](./gone.md)\n",
                    docs,
                    EX + B_EXCLUDE,
                )
                == 1,
                "",
            )
        )
        cases.append(
            (
                "a part heading with no target is not a chapter",
                run(tmp, "# Summary\n\n- [Part I]()\n- [A](./a.md)\n", {"docs/a.md": "# A\n"}) == 0,
                "",
            )
        )
        cases.append(
            (
                "a chapter outside the book's src fails",
                run(tmp, "# Summary\n\n- [Up](../README.md)\n", {"docs/a.md": "# A\n", "README.md": "# R\n"})
                == 1,
                "",
            )
        )
        cases.append(
            (
                "an entry without a reason is a usage error",
                run(
                    tmp,
                    summary,
                    docs,
                    EX + '[[exclude]]\nglob = "docs/b.md"\nreason = ""\n',
                )
                == 2,
                "",
            )
        )
        # the SUMMARY.md shape: mdBook consumes it, so it is never a chapter itself
        cases.append(
            (
                "SUMMARY.md is unlisted unless the ledger says why",
                run(tmp, summary, {"docs/SUMMARY.md": "# Summary\n", "docs/a.md": "# A\n"}, "") == 1,
                "",
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases.append(
            (
                "a tree without SUMMARY.md is an I/O error",
                check(root, root / LEDGER) == 2,
                "",
            )
        )
    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_book_coverage --selftest: {len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="the repository root (default: the repo this script lives in)",
    )
    parser.add_argument(
        "--ledger",
        default=LEDGER,
        help=f"the exclusion ledger, relative to --root (default: {LEDGER})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every document outside the chapter list, with the entry that covers it",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the built-in pass/fail cases in a temp tree and exit non-zero on a failure",
    )
    args = parser.parse_args(argv[1:])
    if args.selftest:
        return selftest()
    root = args.root.resolve()
    return check(root, root / args.ledger, args.list)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
