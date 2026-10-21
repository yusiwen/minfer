#!/usr/bin/env python3
"""List the live documents that cite the files a diff changes.

The three doc gates this repository runs answer "does the citation resolve?", "did
its target move?" and "does its number name a revision?". None of them answers the
question a reviewer has to ask about prose: **does the sentence still describe the
code?** That question has no mechanical test — measured for #327, 99 of the 144 live
anchors carrying an adjacent non-symbol token do not contain that token inside their
own range, and the tokens are ordinary prose, so a rule over them would judge the
neighbouring sentence.

What *can* be automated is the candidate list. Every anchor is already resolved by
`check_doc_line_anchors.py`, so the reverse index — changed file → the paragraphs that
cite it — falls out of the same scan. This script prints that list for a diff and
nothing else: it never fails on "impact" (there is no such verdict), and a range it
cannot read is an error rather than an empty list, because "no output" must never be
mistaken for "no impact".

The PR body carries the answer: `scripts/classify_changes.py` emits `doc_cited`, and
`scripts/check_pr_body.py` then requires a filled `## Doc impact` section (the ninth
heading, conditional on that flag). `N/A — <reason>` is a filled section.

Usage:

    python3 scripts/doc_impact.py --base origin/master [--head HEAD] [--root .]

Exit codes: 0 the list was printed, 2 the range could not be read. Pure and offline:
stdlib plus `git diff`, no network.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import check_doc_line_anchors as anchors

HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


class ImpactError(Exception):
    """The range could not be read — usage or I/O, never a verdict."""


def changed_paths(root: Path, base: str, head: str) -> list[str]:
    """The paths the range touches, repository-relative (`git diff --name-only`)."""
    proc = subprocess.run(
        ["git", "-C", str(root), "diff", "--name-only", f"{base}...{head}"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        raise ImpactError(
            f"git diff {base}...{head}: {detail[-1] if detail else 'git failed'}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def heading_above(root: Path, doc: str, line: int) -> str:
    """The nearest markdown heading above `line` in `doc`, for the reader's context."""
    lines = (root / doc).read_text(encoding="utf-8", errors="replace").splitlines()
    for index in range(min(line, len(lines)) - 1, -1, -1):
        match = HEADING.match(lines[index])
        if match is not None:
            return match.group(2).strip()
    return "(no heading)"


def impact(root: Path, base: str, head: str) -> tuple[list[str], list[anchors.Anchor]]:
    """`(changed paths, anchors into them)`, sorted by document then line."""
    changed = changed_paths(root, base, head)
    report = anchors.Checker(root).run()
    wanted = set(changed)
    hits = [anchor for anchor in report.anchors if anchor.target in wanted]
    hits.sort(key=lambda anchor: (anchor.doc, anchor.line))
    return sorted(changed), hits


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", required=True, help="the base revision to diff against")
    parser.add_argument("--head", default="HEAD", help="the head revision (default: HEAD)")
    parser.add_argument("--root", default=None, help="repository root (default: this tree)")
    args = parser.parse_args(argv[1:])

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    try:
        changed, hits = impact(root, args.base, args.head)
    except ImpactError as error:
        print(f"doc_impact: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"doc_impact: {error}", file=sys.stderr)
        return 2

    docs = sorted({anchor.doc for anchor in hits})
    print(
        f"doc_impact: {len(changed)} changed file(s), {len(hits)} citation(s) in "
        f"{len(docs)} document(s)"
    )
    for anchor in hits:
        heading = heading_above(root, anchor.doc, anchor.line)
        symbol = f" `{anchor.symbol}`" if anchor.symbol else ""
        print(f"  {anchor.where()} — §{heading} → {anchor.target}{symbol}")
    if not hits:
        print("  none — no live document cites a file this range changes")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
