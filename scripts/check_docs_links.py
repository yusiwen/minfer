#!/usr/bin/env python3
"""Fail on a relative markdown link whose target does not exist.

The execution plan's documentation steps gate on a "docs build", but `mdbook build`
only *renders*: a relative link to a file that was renamed or never written becomes a
dead link in the HTML without failing anything. This script is the missing half — it
resolves every relative link in the repo's markdown against the filesystem and names
the offending file, line and target (ticket #63).

What it checks: `[text](target)` and `![alt](target)` inline links, plus
reference-style definitions (`[label]: target`), in every `docs/**/*.md`,
`AGENTS.md`, `README.md` and `viz/README.md`. Fenced code blocks are skipped — they
hold examples, not links.

**What this checker cannot see, and why it is not enough.** It resolves a target
against *the checkout*, so it is blind to everything `mdbook build` decides: a
chapter named `README.md` is published as `index.html` while the link text is
rewritten to `README.html`, and a document that is not in `docs/SUMMARY.md` is not
published at all — both were true, and green here, for 300 links on the site (see
`scripts/check_book_links.py`, which reads the rendered `book/` instead).

What it deliberately does not check:

- **anchors** (`page.md#section`): the target file must exist, the fragment is not
  resolved against its headings (that needs a slugifier per renderer; the rendered
  ids are checked by `scripts/check_book_links.py`);
- **site-absolute paths** (`/foo`): mdBook renders those against the site root, not
  this checkout;
- **external URLs**: anything with a scheme (`https:`, `mailto:`, …) is out of scope.

Usage:

    python3 scripts/check_docs_links.py [--root .]
    python3 scripts/check_docs_links.py --selftest

Exit codes: 0 clean, 1 a link does not resolve, 2 usage or I/O error. Pure and
offline: stdlib only, no build, no network.
"""

from __future__ import annotations

import argparse
import re
import sys

import docset
import tempfile
import urllib.parse
from pathlib import Path

# Inline `[text](target)` / `![alt](target)` — the target stops at the first
# whitespace (a title) or the closing paren.
INLINE = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?")
# Reference definitions: `[label]: target`, at the start of a line.
REFDEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^\s>]+)>?")
# A fence: ``` or ~~~, optionally indented, optionally with an info string.
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
# Inline code spans, including the double-backtick form. Text inside them is *code*
# (`ne[1](attn_q)` is a shape expression, not a link), so it is removed before the
# link patterns run.
CODE_SPAN = re.compile(r"(`+)(.+?)\1", re.DOTALL)
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")

def markdown_files(root: Path) -> list[Path]:
    """Every markdown file the gate covers, in a stable order.

    The set is discoverable — every tracked `*.md` — minus the files `scripts/docset.py`'s
    ledger excuses with a reason (#522). It used to be `docs/**` plus a hand-kept list of
    three outside files, which is how a dead link in `.github/`, `viz/` or at the root
    stayed unread.
    """
    return docset.scanned(root)


def without_code_spans(text: str) -> str:
    """`text` with every inline code span blanked, newlines preserved.

    Spans can cross a line break (a sentence wrapping inside backticks is normal in
    these docs), so this runs on the whole file rather than line by line — a
    per-line regex leaves the tail of a wrapped span looking like a link
    (`ne[1](attn_q)` is a shape expression, not a target). Blanking keeps the line
    numbering intact for the error messages.
    """

    def blank(match: re.Match) -> str:
        return "\n" * match.group(0).count("\n")

    return CODE_SPAN.sub(blank, text)


def targets_in(path: Path):
    """Yield `(line_number, target)` for every link target in `path`."""
    fence: str | None = None
    cleaned = without_code_spans(path.read_text(encoding="utf-8"))
    for n, raw in enumerate(cleaned.splitlines(), start=1):
        m = FENCE.match(raw)
        if m:
            marker = m.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        for match in INLINE.finditer(raw):
            yield n, match.group(1)
        match = REFDEF.match(raw)
        if match:
            yield n, match.group(1)


def resolvable(source: Path, target: str) -> bool:
    """Whether `target`, written inside `source`, names something that exists."""
    if not target or target.startswith("#"):
        return True  # in-page anchor: the file itself is the target
    if SCHEME.match(target) or target.startswith("/"):
        return True  # external URL or site-absolute path — out of scope
    path = urllib.parse.unquote(target.split("#", 1)[0])
    if not path:
        return True
    resolved = (source.parent / path).resolve()
    if resolved.exists():
        return True
    # `dir/` (or a bare name) that exists as a directory holding an index page.
    for index in ("README.md", "index.md"):
        if (resolved / index).is_file():
            return True
    return False


def scan(root: Path, files: list[Path] | None = None) -> tuple[list[str], int, int]:
    """`(missing, checked, file_count)` for every covered file under `root`."""
    if files is None:
        files = markdown_files(root)
    checked = 0
    missing: list[str] = []
    for path in files:
        rel = path.relative_to(root)
        for line, target in targets_in(path):
            if not target or target.startswith("#") or SCHEME.match(target):
                continue
            if target.startswith("/"):
                continue
            checked += 1
            if not resolvable(path, target):
                missing.append(f"{rel}:{line}: {target}")
    return missing, checked, len(files)


def check(root: Path, files: list[Path] | None = None) -> int:
    """Print the verdict for `root`; return the exit code."""
    if not (root / "docs").is_dir():
        print(f"check_docs_links: no docs/ under {root}", file=sys.stderr)
        return 2
    missing, checked, count = scan(root, files)
    if not count:
        print(f"check_docs_links: no markdown found under {root}", file=sys.stderr)
        return 2
    for line in missing:
        print(f"BROKEN LINK  {line}", file=sys.stderr)
    if missing:
        print(
            f"check_docs_links: {len(missing)} of {checked} relative links do not "
            f"resolve (in {count} markdown files)",
            file=sys.stderr,
        )
        return 1
    print(f"check_docs_links: {checked} relative links resolve in {count} markdown files")
    return 0


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "target.md").write_text("# Target\n", encoding="utf-8")
        (root / "docs" / "sub").mkdir()
        (root / "docs" / "sub" / "README.md").write_text("# Sub\n", encoding="utf-8")

        def run(body: str) -> int:
            (root / "docs" / "page.md").write_text(body, encoding="utf-8")
            return check(root)

        cases.append(("a link to an existing file passes", run("see [t](./target.md)\n") == 0, ""))
        cases.append(
            ("a link to a missing file fails", run("see [gone](./nope.md)\n") == 1, "")
        )
        cases.append(
            (
                "a link inside a fenced code block is ignored",
                run("```\n[gone](./nope.md)\n```\n") == 0,
                "",
            )
        )
        cases.append(
            (
                "a link inside an inline code span is ignored",
                run("the shape `ne[1](attn_q)` is not a link\n") == 0,
                "",
            )
        )
        cases.append(
            (
                "a reference definition with a missing target fails",
                run("see [t][gone]\n\n[gone]: ./nope.md\n") == 1,
                "",
            )
        )
        cases.append(
            (
                "an external URL is out of scope",
                run("see [x](https://example.com/nope)\n") == 0,
                "",
            )
        )
        cases.append(("a site-absolute path is out of scope", run("see [x](/nope)\n") == 0, ""))
        cases.append(("an in-page anchor stands alone", run("see [x](#section)\n") == 0, ""))
        cases.append(
            (
                "`dir/` resolves through its README.md",
                run("see [sub](./sub/)\n") == 0,
                "",
            )
        )
        # The boundary this checker owns: `README.md` *exists in the checkout* even
        # though mdBook publishes it as `index.html`, so the link above is green
        # here and dead on the site. `check_book_links.py` owns that half.
        cases.append(
            (
                "a README.md target passes here (the book checker owns it)",
                run("see [sub](./sub/README.md)\n") == 0,
                "",
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        cases.append(("a tree without docs/ is an I/O error", check(Path(tmp)) == 2, ""))
    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_docs_links --selftest: {len(cases) - failed}/{len(cases)} cases pass")
    # 12. The scope is discoverable (#522): markdown outside `docs/` is read too.
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "docs" / "target.md").write_text("# Target\n", encoding="utf-8")
        (root / ".github").mkdir()
        (root / ".github" / "note.md").write_text("[dead](./nope.md)\n", encoding="utf-8")
        cases.append(("a dead link outside docs/ fails", check(root) == 1, ""))
        (root / ".github" / "note.md").write_text("[live](../docs/target.md)\n", encoding="utf-8")
        cases.append(("a link outside docs/ resolves against the root", check(root) == 0, ""))

    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_docs_links --selftest: {len(cases) - failed}/{len(cases)} cases pass")
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
        "--selftest",
        action="store_true",
        help="run the built-in pass/fail cases in a temp tree and exit non-zero on a failure",
    )
    args = parser.parse_args(argv[1:])
    if args.selftest:
        return selftest()
    root = args.root.resolve()
    try:
        scope = docset.announce(root)
    except docset.ScopeError as error:
        print(f"check_docs_links: {error}", file=sys.stderr)
        return 1
    return check(root, files=scope.scanned)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
