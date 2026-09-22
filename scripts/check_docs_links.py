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

What it deliberately does not check:

- **anchors** (`page.md#section`): the target file must exist, the fragment is not
  resolved against its headings (that needs a slugifier per renderer);
- **site-absolute paths** (`/foo`): mdBook renders those against the site root, not
  this checkout;
- **external URLs**: anything with a scheme (`https:`, `mailto:`, …) is out of scope.

Usage: `python3 scripts/check_docs_links.py [repo-root]` (default: the repo this
script lives in). Exits 0 when every relative target resolves, 1 otherwise.
"""

from __future__ import annotations

import re
import sys
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
    """Every file the gate covers, in a stable order."""
    files = sorted((root / "docs").rglob("*.md"))
    for extra in ("AGENTS.md", "README.md", "viz/README.md"):
        p = root / extra
        if p.is_file():
            files.append(p)
    return sorted(set(files))


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


def resolvable(source: Path, target: str, root: Path) -> bool:
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


def main(argv: list[str]) -> int:
    root = Path(argv[1]).resolve() if len(argv) > 1 else Path(__file__).resolve().parent.parent
    files = markdown_files(root)
    if not files:
        print(f"check_docs_links: no markdown found under {root}", file=sys.stderr)
        return 1

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
            if not resolvable(path, target, root):
                missing.append(f"{rel}:{line}: {target}")
    for line in missing:
        print(f"BROKEN LINK  {line}", file=sys.stderr)
    if missing:
        print(
            f"check_docs_links: {len(missing)} of {checked} relative links do not "
            f"resolve (in {len(files)} markdown files)",
            file=sys.stderr,
        )
        return 1
    print(f"check_docs_links: {checked} relative links resolve in {len(files)} markdown files")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
