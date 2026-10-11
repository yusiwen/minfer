#!/usr/bin/env python3
"""Fail on a link in the *rendered* book that a reader cannot follow.

`check-docs` builds the site with `scripts/build_book.sh` and then ran
`scripts/check_docs_links.py`, which resolves a link against **the checkout**. Every
step of the build was therefore green while the published book carried 315 dead
links over 81 distinct targets, because mdBook — not the checkout — decides what a
link points at:

1. **A chapter named `README.md` is published as `index.html`**, but the `.md`→`.html`
   link rewrite keeps the basename, so every "back to the index" link 404s. Measured
   in an isolated two-file book; on the site,
   `…/cuda_tutorial/README.html` is 404 while `…/cuda_tutorial/index.html` is 200.
2. **A document that is not in `docs/SUMMARY.md` is not published at all** — mdBook
   renders no page for it and copies no `.md` — while a relative link to it stays a
   relative link. `mdbook build` exits 0 either way: a source file with no SUMMARY
   entry is dropped silently, and a SUMMARY entry with no file is silently rendered
   as an empty *draft* page.

So this checker reads the output tree instead of the source: every relative `href`
(and `src`) in every rendered page must resolve to a file in `book/`, and every
`#fragment` must be an `id=` in the page it points at. It is the half
`check_docs_links.py` cannot be, and its docstring says so.

What it deliberately does not check: external URLs (anything with a scheme),
site-absolute paths (`/…`, which mdBook resolves against the site root, not this
checkout), and `book/print.html` (a generated concatenation of every chapter, so its
links are the same ones checked in their own pages).

**The ledger.** Known-dead links are listed in `scripts/book-links.toml` beside this
checker (ADR-0023), each `target` a glob over the book-relative target and each with
a reason. The ledger is a ratchet: a dead link no entry covers is a failure, and an
entry that no longer covers a dead link is a failure too, so it can only shrink. It
exists so that a fix landed in slices — [#516](https://github.com/yusiwen/minfer/issues/516)
re-points the links — does not have to land in the same PR as the gate that must
catch the next one.

Usage:

    python3 scripts/check_book_links.py [--root .] [--book book] [--ledger FILE]
    python3 scripts/check_book_links.py --selftest

Exit codes: 0 clean, 1 a dead link or a stale ledger entry, 2 usage or I/O error
(`book/` not built, ledger missing). Pure and offline: stdlib only, no build, no
network.
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import re
import sys
import tempfile
import tomllib
import urllib.parse
from pathlib import Path

#: `href` and `src` are both checked: mdBook copies every non-`.md` source file, so a
#: missing image or asset is a real 404 too.
ATTR = re.compile(r'(?:href|src)="([^"]*)"')
#: Every `id=` in a page — the anchors `#fragment` may point at.
IDS = re.compile(r'id="([^"]+)"')
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*:")
#: A generated concatenation of all chapters; its links are checked in their pages.
SKIP_PAGES = frozenset({"print.html"})
LEDGER = "scripts/book-links.toml"


class Problem:
    """One link a reader cannot follow."""

    __slots__ = ("page", "href", "key", "detail", "source")

    def __init__(self, page: str, href: str, key: str, detail: str, source: str) -> None:
        self.page = page
        self.href = href
        self.key = key
        self.detail = detail
        self.source = source


def pages(book: Path) -> list[Path]:
    """Every rendered page, in a stable order."""
    return sorted(p for p in book.rglob("*.html") if p.name not in SKIP_PAGES)


def links(page: Path):
    """Every `(href, line_number)` in `page`."""
    text = page.read_text(encoding="utf-8", errors="replace")
    for n, line in enumerate(text.splitlines(), start=1):
        for match in ATTR.finditer(line):
            yield match.group(1), n


def ids_of(path: Path, cache: dict[Path, set[str]]) -> set[str]:
    """The `id=` set of `path`, memoised for the run."""
    if path not in cache:
        cache[path] = set(
            IDS.findall(path.read_text(encoding="utf-8", errors="replace"))
        )
    return cache[path]


def source_doc(root: Path, resolved: Path, book: Path, src: str) -> str:
    """The markdown a rendered path came from, or `?`.

    `book/x/NN-y.html` is `docs/x/NN-y.md`; a chapter rendered as `index.html` is the
    `README.md` beside it. Both roots come from the book's own config (`book.toml`'s
    `src` and the `--book` directory), so a renamed build dir does not silently turn
    every hint into `?`. The mapping is only used for the error message.
    """
    try:
        rel = resolved.relative_to(book)
    except ValueError:
        return "?"
    doc = Path(src, *rel.parts).with_suffix(".md")
    if doc.name == "index.md":
        doc = doc.with_name("README.md")
    return doc.as_posix()


def classify(root: Path, book: Path, src: str, resolved: Path, target: str) -> str:
    """Why the target does not resolve — the sentence after the arrow."""
    doc = source_doc(root, resolved, book, src)
    doc_path = root / doc
    sibling = resolved.with_name("index.html")
    if os.path.basename(target) == "README.html" and sibling.is_file():
        return (
            "mdBook publishes a `README.md` chapter as `index.html`, so this link "
            f"needs `index.html` (which exists at {sibling.relative_to(root).as_posix()})"
        )
    if doc != "?" and doc_path.is_file():
        return (
            f"the document exists at {doc} but is not a chapter: `{src}/SUMMARY.md` "
            "does not list it, so mdBook publishes no page for it"
        )
    if not resolved.is_relative_to(book):
        return f"the target is outside the book's source tree (`{src}/`)"
    return f"no such file in the built book ({resolved.relative_to(root).as_posix()})"


def problems_in(book: Path, root: Path, src: str = "docs") -> list[Problem]:
    """Every dead link or missing fragment in the built book."""
    found: list[Problem] = []
    cache: dict[Path, set[str]] = {}
    for page in pages(book):
        page_rel = page.relative_to(book).as_posix()
        for href, _line in links(page):
            if not href or SCHEME.match(href) or href.startswith("//") or href.startswith("/"):
                continue
            path, _, fragment = href.partition("#")
            path = urllib.parse.unquote(path)
            if path:
                key = os.path.normpath(os.path.join(os.path.dirname(page_rel), path))
                resolved = (book / key).resolve()
            else:
                key = page_rel  # in-page anchor: the page is the target
                resolved = page
            doc = source_doc(root, resolved, book, src)
            if not resolved.exists():
                found.append(
                    Problem(page_rel, href, key, classify(root, book, src, resolved, path), doc)
                )
                continue
            # The fragment is compared **decoded**: mdBook percent-encodes a
            # non-ASCII href while the `id=` it generates stays raw (`#1-概述`
            # becomes `#1-%E6%A6%82%E8%BF%B0`), and a browser matches the decoded
            # fragment against the id. Comparing the raw strings reported nine
            # working links in `docs/CLI-CONVERSATION-PLAN.md` as dead.
            decoded = urllib.parse.unquote(fragment)
            if fragment and resolved.suffix == ".html" and decoded not in ids_of(resolved, cache):
                found.append(
                    Problem(
                        page_rel,
                        href,
                        f"{key}#{decoded}",
                        f"the page has no `id=\"{decoded}\"` (mdBook slugs a heading into its id)",
                        doc,
                    )
                )
    return found


def load_ledger(path: Path) -> list[tuple[str, str]]:
    """`(target_glob, reason)` for every known-dead entry."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    entries = data.get("known_dead", [])
    out: list[tuple[str, str]] = []
    for entry in entries:
        target = entry.get("target", "")
        reason = entry.get("reason", "")
        if not target or not reason:
            raise ValueError(f"{path}: an entry needs both `target` and `reason`")
        out.append((target, reason))
    return out


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


def check(root: Path, book_dir: Path, ledger_path: Path) -> int:
    """Print the verdict; return the exit code."""
    if not book_dir.is_dir():
        print(
            f"check_book_links: {book_dir} is not built — run scripts/build_book.sh first",
            file=sys.stderr,
        )
        return 2
    if not ledger_path.is_file():
        print(f"check_book_links: {ledger_path} is missing", file=sys.stderr)
        return 2
    try:
        ledger = load_ledger(ledger_path)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"check_book_links: {exc}", file=sys.stderr)
        return 2

    found = problems_in(book_dir, root, source_root(root))
    patterns = [p for p, _ in ledger]
    unexplained = [
        p for p in found if not any(fnmatch.fnmatchcase(p.key, pat) for pat in patterns)
    ]
    stale = [
        (pat, reason)
        for pat, reason in ledger
        if not any(fnmatch.fnmatchcase(p.key, pat) for p in found)
    ]

    for problem in unexplained:
        print(f"DEAD LINK  {book_dir.name}/{problem.page}  href={problem.href}", file=sys.stderr)
        print(f"           from {problem.source} — {problem.detail}", file=sys.stderr)
    for pattern, reason in stale:
        print(f"STALE LEDGER  {ledger_path}: `{pattern}` covers nothing any more", file=sys.stderr)
        print(f"              (reason was: {reason})", file=sys.stderr)

    distinct = len({p.key for p in found})
    page_count = len(pages(book_dir))
    if unexplained or stale:
        print(
            f"check_book_links: {len(unexplained)} unexplained dead link(s) "
            f"({len(found)} dead in all, {distinct} distinct targets, {page_count} pages) "
            f"and {len(stale)} stale ledger entry(ies)",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_book_links: {len(found)} known-dead link(s) ({distinct} distinct targets) "
        f"all matched by {len(ledger)} ledger entries, 0 unexplained, in {page_count} pages"
    )
    return 0


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        book = root / "book"
        (book / "sub").mkdir(parents=True)
        ledger = root / LEDGER
        ledger.parent.mkdir(parents=True)

        def write_ledger(body: str) -> None:
            ledger.write_text(body, encoding="utf-8")

        def run(page: str, ledger_body: str = "", extra: dict[str, str] | None = None) -> int:
            for name, text in (extra or {}).items():
                target = (book / name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(text, encoding="utf-8")
            (book / "page.html").write_text(page, encoding="utf-8")
            write_ledger(ledger_body)
            return check(root, book, ledger)

        cases.append(
            (
                "a link to a rendered page passes",
                run('<a href="sub/index.html">x</a>', "", {"sub/index.html": "<h1 id=\"s\">s</h1>"})
                == 0,
                "",
            )
        )
        cases.append(("a link to a missing page fails", run('<a href="nope.html">x</a>') == 1, ""))
        cases.append(
            (
                "a README.html link is named as the rewrite trap",
                run(
                    '<a href="sub/README.html">x</a>',
                    "",
                    {"sub/index.html": "<h1>s</h1>"},
                )
                == 1,
                "",
            )
        )
        cases.append(
            (
                "a covered dead link passes",
                run('<a href="nope.html">x</a>', '[[known_dead]]\ntarget = "nope.html"\nreason = "t"\n')
                == 0,
                "",
            )
        )
        cases.append(
            (
                "a ledger entry that covers nothing fails",
                run('<a href="sub/index.html">x</a>', '[[known_dead]]\ntarget = "nope.html"\nreason = "t"\n')
                == 1,
                "",
            )
        )
        cases.append(
            (
                "a missing fragment fails",
                run('<a href="sub/index.html#gone">x</a>', "", {"sub/index.html": "<h1 id=\"here\">s</h1>"})
                == 1,
                "",
            )
        )
        cases.append(
            (
                "an in-page fragment is checked against its own page",
                run('<h1 id="top">t</h1>\n<a href="#gone">x</a>') == 1,
                "",
            )
        )
        cases.append(
            (
                "a heading id present in the page passes",
                run('<h1 id="top">t</h1>\n<a href="#top">x</a>') == 0,
                "",
            )
        )
        cases.append(
            (
                "a percent-encoded fragment matches its raw id",
                run(
                    '<a href="sub/index.html#caf%C3%A9">x</a>',
                    "",
                    {"sub/index.html": '<h1 id="café">s</h1>'},
                )
                == 0,
                "",
            )
        )
        cases.append(
            (
                "an external URL is out of scope",
                run('<a href="https://example.com/nope">x</a>') == 0,
                "",
            )
        )
        cases.append(
            (
                "a site-absolute path is out of scope",
                run('<a href="/nope">x</a>') == 0,
                "",
            )
        )
        cases.append(
            (
                "a ledger entry without a reason is a usage error",
                run('<a href="nope.html">x</a>', '[[known_dead]]\ntarget = "nope.html"\nreason = ""\n')
                == 2,
                "",
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases.append(
            (
                "a tree that was never built is an I/O error",
                check(root, root / "book", root / LEDGER) == 2,
                "",
            )
        )
        (root / "book").mkdir()
        cases.append(
            (
                "a missing ledger is an I/O error",
                check(root, root / "book", root / LEDGER) == 2,
                "",
            )
        )
    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_book_links --selftest: {len(cases) - failed}/{len(cases)} cases pass")
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
        "--book",
        default="book",
        help="the built book, relative to --root (default: book)",
    )
    parser.add_argument(
        "--ledger",
        default=LEDGER,
        help=f"the known-dead ledger, relative to --root (default: {LEDGER})",
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
    return check(root, root / args.book, root / args.ledger)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
