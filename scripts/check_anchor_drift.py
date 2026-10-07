#!/usr/bin/env python3
"""Fail a PR whose `path:NNN` anchor points at text a line-shifting commit moved.

`scripts/check_doc_line_anchors.py` (#266) proves three things about an anchor: the
file exists, the line is in range, and — for a backticked symbol — that the symbol is
still in the file the anchor names.  It **cannot see a moving target**: a commit that
adds or removes lines in an anchored file shifts every anchor into it, the anchors
still resolve, and the checker reports green.  That is not hypothetical; it happened
twice in one round on 2026-10-07 and both were caught by a hand-written old->new line
map instead of by a gate ([#329] / [PR #338]: 7 anchors off by one in
``docs/inference_e2e_walkthrough/07-…`` and ``14-…``; [#299] / [PR #343]: 69 anchors
off by 1-3 across eight docs).

Why a **sibling** script rather than ``check_doc_line_anchors.py --against <rev>``:
the existing checker is a pure filesystem audit — ``Checker(root)`` walks one tree and
its ``--selftest`` builds synthetic trees in ``tempfile`` directories, with no
repository at all.  The drift mode needs two trees (the base revision and the working
tree), a ``git diff`` and a ``git archive``, and it must not perturb the existing
checker's ``--root`` / ``--list`` / ``--strict-symbols`` surface or its verdicts.  A
sibling imports it as a module, so the anchor grammar (``ANCHOR``), the path
resolution (``Checker``), the symbol rule and the ``FROZEN`` policy are literally the
same objects — there is no second notion of "live anchor" to keep in sync.

How it works.  ``git diff -U0 <rev>...<head>`` is reduced to a per-file hunk list, hence
an old->new line map.  The anchors are read from **both** revisions: the base revision's
docs come from ``git archive <rev>``, the head side's from the working tree (or from
``git archive <head>`` under ``--head <rev>``, which makes a "the head before the
re-pointing commit" reproduction a one-liner).  An anchor is paired with the
base-revision anchor on the same doc line (*normalised* — the ``:NNN`` / ``:NNN-MMM``
numbers replaced — so a re-pointed line still pairs), and then:

``stale``
    the base anchor's lines map forward to different numbers than the working tree's
    anchor carries: the doc was not re-pointed.  Reported as
    ``<doc>:<line> -> <target>:<old> (now <new>)`` and **fails** (exit 1).

``ambiguous``
    a cited *endpoint* has no image in the map, because the range deleted that line:
    no arithmetic can say what the anchor should now cite.  Reported, never guessed;
    promoted to a failure by ``--strict``.  This is the only undecidable shape — a
    surviving line always maps to a surviving line, so a line of an *added* block can
    never be the image of an anchor read from the base revision, and a block added or
    removed *between* two surviving endpoints is decidable (the endpoints map, and the
    mapped numbers are the answer — [#299]'s re-point extended exactly such ranges).

``correct`` / ``unchanged`` / ``frozen`` / ``not compared``
    pass.  ``not compared`` covers a base anchor whose doc line was rewritten beyond
    the numbers, whose cited span changed shape, or whose target path resolves
    differently at the head tree (a rename or a deletion — its own boundary, below).

The ``FROZEN`` map of ``check_doc_line_anchors.py`` exempts a doc here too, by the
same lookup: a record of a past revision is not re-pointed, so its anchors are never
drift.

Boundary (what this does not catch).
  - A rename or a deletion is *not* mapped through the diff's hunk list: an anchor
    into a renamed/deleted path is reported ``ambiguous`` rather than followed.  A doc
    that still names a deleted tree-shaped path is already the existing checker's
    ``missing``.
  - A base anchor whose doc line was rewritten in the range is ``not compared``: the
    author touched that line, and pairing it would be a guess.  It is counted and
    listed by ``--list`` so "why did this pass?" has an answer.
  - Only a same-shape pair is compared (range to range, single to single), because a
    changed extent is an author edit, not arithmetic.
  - The working tree is the head side by default, so an uncommitted doc edit is checked
    against the committed diff.  Run it on a clean tree (CI does), or pass ``--head
    <rev>`` and nothing local is read at all.

Usage::

    python3 scripts/check_anchor_drift.py [--root DIR] [--list] [--strict] <rev>
    python3 scripts/check_anchor_drift.py [--head REV] <rev>   # revision to revision
    python3 scripts/check_anchor_drift.py --selftest

Exit codes: 0 clean, 1 stale anchors (or ambiguous under ``--strict``),
2 usage or I/O error.

[#299]: https://github.com/yusiwen/minfer/issues/299
[#329]: https://github.com/yusiwen/minfer/issues/329
[PR #338]: https://github.com/yusiwen/minfer/pull/338
[PR #343]: https://github.com/yusiwen/minfer/pull/343
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# The anchor grammar, the resolution, the symbol rule and FROZEN are the existing
# checker's, imported rather than restated (see the module docstring).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_doc_line_anchors as anchors  # noqa: E402


class DriftError(Exception):
    """A usage or I/O failure — never a drift verdict."""


# ---------------------------------------------------------------------------
# The old->new line map: `git diff -U0` hunks, and what they say about a line.
# ---------------------------------------------------------------------------

#: `@@ -old,count +new,count @@`; a count of 1 is elided by git, a count of 0 is not.
HUNK = re.compile(
    r"^@@ -(?P<os>\d+)(?:,(?P<oc>\d+))? \+(?P<ns>\d+)(?:,(?P<nc>\d+))? @@"
)

RENAME_FROM = "rename from "
RENAME_TO = "rename to "


@dataclass(frozen=True)
class Hunk:
    """One `-U0` hunk: `old_count` old lines replaced by `new_count` new ones."""

    old_start: int
    old_count: int
    new_start: int
    new_count: int

    @property
    def delta(self) -> int:
        return self.new_count - self.old_count

    @property
    def first_shifted_old(self) -> int:
        """The first *old* line this hunk's delta applies to.

        A pure insertion (`old_count == 0`) is anchored *after* ``old_start``, so the
        shift starts at ``old_start + 1``; a replacement shifts from the line after
        its last removed one.
        """
        return self.old_start + max(self.old_count, 1)

    def removes(self, old_line: int) -> bool:
        """Whether this hunk deleted ``old_line`` (which therefore has no new line)."""
        return self.old_count > 0 and self.old_start <= old_line < self.old_start + self.old_count

    def touches(self, lo: int, hi: int) -> bool:
        """Whether a *removed* block falls anywhere inside the cited range ``[lo, hi]``."""
        if self.old_count == 0:
            return False
        return self.old_start <= hi and self.old_start + self.old_count - 1 >= lo


@dataclass
class FileDiff:
    """One file's hunk list, keyed by its path in the head tree."""

    path: str
    hunks: list[Hunk] = field(default_factory=list)

    def shift(self, old_line: int) -> int:
        """The cumulative line delta at ``old_line``."""
        return sum(h.delta for h in self.hunks if h.first_shifted_old <= old_line)

    def old_to_new(self, old_line: int) -> int | None:
        """Where the base revision's ``old_line`` now is, or None if it was removed."""
        for hunk in self.hunks:
            if hunk.removes(old_line):
                return None
        return old_line + self.shift(old_line)


@dataclass
class DiffMap:
    """The parsed range: touched files, renames and deletions."""

    files: dict[str, FileDiff] = field(default_factory=dict)
    renames: dict[str, str] = field(default_factory=dict)
    deleted: set[str] = field(default_factory=set)


def parse_diff(text: str) -> DiffMap:
    """Reduce a `git diff -U0` transcript to its hunks, renames and deletions."""
    out = DiffMap()
    current: FileDiff | None = None
    old_path: str | None = None
    rename_from: str | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current, old_path, rename_from = None, None, None
        elif line.startswith("--- "):
            p = line[4:].strip()
            old_path = None if p == "/dev/null" else _strip_prefix(p)
        elif line.startswith("+++ "):
            p = line[4:].strip()
            if p == "/dev/null":
                if old_path:
                    out.deleted.add(old_path)
                current = None
            else:
                path = _strip_prefix(p)
                current = FileDiff(path=path)
                out.files[path] = current
                if rename_from:
                    out.renames[rename_from] = path
        elif line.startswith(RENAME_FROM):
            rename_from = line[len(RENAME_FROM) :].strip()
        elif line.startswith(RENAME_TO):
            continue
        else:
            match = HUNK.match(line)
            if match and current is not None:
                current.hunks.append(
                    Hunk(
                        old_start=int(match.group("os")),
                        old_count=int(match.group("oc") or 1),
                        new_start=int(match.group("ns")),
                        new_count=int(match.group("nc") or 1),
                    )
                )
    return out


def _strip_prefix(path: str) -> str:
    return path[2:] if path[:2] in ("a/", "b/") else path


# ---------------------------------------------------------------------------
# The verdicts.
# ---------------------------------------------------------------------------

STALE = "stale"
AMBIGUOUS = "ambiguous"
CORRECT = "correct"
UNCHANGED = "unchanged"
NOT_COMPARED = "not-compared"

#: Verdicts the existing checker already fails on; the drift mode leaves them to it.
ALREADY_FAILED = ("missing", "out-of-range", "range-order", "symbol-moved")


@dataclass
class Verdict:
    """One paired anchor and what the line map says about it."""

    doc: str
    line: int
    target: str
    old: str
    new: str | None
    kind: str
    detail: str = ""

    def where(self) -> str:
        return f"{self.doc}:{self.line}"

    def sentence(self) -> str:
        """The one-line form the ticket fixes: `doc:line -> target:old (now new)`."""
        if self.kind == STALE:
            return f"{self.where()} → {self.target}:{self.old} (now {self.new})"
        if self.kind == AMBIGUOUS:
            return f"{self.where()} → {self.target}:{self.old} — {self.detail}"
        return f"{self.where()} → {self.target}:{self.old} [{self.kind}] {self.detail}".rstrip()


@dataclass
class DriftReport:
    verdicts: list[Verdict] = field(default_factory=list)

    def of(self, kind: str) -> list[Verdict]:
        return [v for v in self.verdicts if v.kind == kind]

    @property
    def stale(self) -> list[Verdict]:
        return self.of(STALE)

    @property
    def ambiguous(self) -> list[Verdict]:
        return self.of(AMBIGUOUS)

    @property
    def compared(self) -> int:
        return sum(1 for v in self.verdicts if v.kind in (STALE, AMBIGUOUS, CORRECT, UNCHANGED))

    def worst(self, strict: bool) -> int:
        if self.stale:
            return 1
        return 1 if strict and self.ambiguous else 0


# ---------------------------------------------------------------------------
# The check.
# ---------------------------------------------------------------------------


def git(root: Path, *args: str) -> bytes:
    """Run git in ``root``; a non-zero exit is an I/O error, never a verdict."""
    proc = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.decode("utf-8", "replace").strip() or f"git {args[0]} failed"
        raise DriftError(f"git {' '.join(args[:2])}: {detail}")
    return proc.stdout


def extract_rev(root: Path, rev: str) -> Path:
    """Materialise ``rev``'s tree in a fresh temp directory (never a git worktree).

    A worktree would register itself under ``.git/worktrees`` and leave state behind on
    a crash; ``git archive`` writes no git metadata at all.
    """
    tmp = tempfile.mkdtemp(prefix="anchor-drift-")
    data = git(root, "archive", "--format=tar", rev)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as tar:
        tar.extractall(tmp, filter="data")
    return Path(tmp)


def normalise(line: str) -> str:
    """One doc line with every anchor's numbers replaced — the pairing key.

    A re-pointing commit changes only the digits, so the normalised line is the same
    on both sides of the range; that is exactly the proof [PR #343]'s re-point used
    (`":NNN" -> ":N"` multisets identical), applied per line instead of per file.
    """

    def repl(match: re.Match[str]) -> str:
        tail = "-N" if match.group("end") else ""
        return f"{match.group('path')}:N{tail}"

    return anchors.ANCHOR.sub(repl, line)


def live(as_: list[anchors.Anchor]) -> list[anchors.Anchor]:
    """The anchors the drift mode can judge: resolved, not frozen, not already failed."""
    return [a for a in as_ if a.target is not None and a.verdict not in ("frozen", *ALREADY_FAILED)]


def _doc_lines(root: Path, rel: str) -> list[str]:
    return (root / rel).read_text(encoding="utf-8", errors="replace").splitlines()


def _shown(a: anchors.Anchor) -> str:
    return f"{a.start}-{a.end}" if a.end is not None else str(a.start)


def _pair(
    base: list[anchors.Anchor],
    head: list[anchors.Anchor],
    base_lines: list[str],
    head_lines: list[str],
) -> tuple[list[tuple[anchors.Anchor, anchors.Anchor]], list[anchors.Anchor]]:
    """Pair base anchors with the working tree's anchors; return the pairs and the rest.

    Primary key: the doc line with its numbers normalised.  Fallback, for a doc line
    reworded around an unchanged anchor: the written path, the symbol, the span width
    and the shape must all agree, and the signature must be unique on both sides.
    """
    base_by_key: dict[str, list[anchors.Anchor]] = {}
    for a in base:
        base_by_key.setdefault(normalise(base_lines[a.line - 1]), []).append(a)
    head_by_key: dict[str, list[anchors.Anchor]] = {}
    for a in head:
        head_by_key.setdefault(normalise(head_lines[a.line - 1]), []).append(a)

    pairs: list[tuple[anchors.Anchor, anchors.Anchor]] = []
    paired_base: set[int] = set()
    paired_head: set[int] = set()
    for key, group in base_by_key.items():
        for b, h in zip(group, head_by_key.get(key, [])):
            # A span whose extent changed is an author edit, not arithmetic: leave it.
            if (b.end is None) != (h.end is None):
                continue
            pairs.append((b, h))
            paired_base.add(id(b))
            paired_head.add(id(h))

    unpaired = [b for b in base if id(b) not in paired_base]
    free_head = [h for h in head if id(h) not in paired_head]

    def signature(a: anchors.Anchor) -> tuple[str, str | None, bool, int | None]:
        return (a.path, a.symbol, a.end is not None, None if a.end is None else a.end - a.start)

    for sig in {signature(b) for b in unpaired} & {signature(h) for h in free_head}:
        bs = [b for b in unpaired if signature(b) == sig]
        hs = [h for h in free_head if signature(h) == sig]
        if len(bs) == 1 and len(hs) == 1:
            pairs.append((bs[0], hs[0]))
            unpaired.remove(bs[0])
            free_head.remove(hs[0])
    return pairs, unpaired


def check(
    root: Path,
    rev: str,
    frozen: dict[str, str] | None = None,
    head: str | None = None,
) -> DriftReport:
    """Compare the head tree's anchors with ``rev``'s over ``git diff <rev>...<head>``.

    ``head=None`` reads the *working tree* (the CI case, and the one that works before
    a commit); ``head=<rev>`` materialises that revision with ``git archive`` and reads
    the anchors from it, so a "the head before the re-pointing commit" reproduction
    needs no checkout.
    """
    git(root, "rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}")
    if head is not None:
        git(root, "rev-parse", "--verify", "--quiet", f"{head}^{{commit}}")
    diff = parse_diff(
        git(
            root,
            "diff",
            "-U0",
            "--no-color",
            "--no-ext-diff",
            f"{rev}...{head or 'HEAD'}",
        ).decode("utf-8", "replace")
    )

    base_root = extract_rev(root, rev)
    head_root = Path(root) if head is None else extract_rev(root, head)
    try:
        base_checker = anchors.Checker(base_root, frozen)
        head_checker = anchors.Checker(head_root, frozen)
        base_by_doc: dict[str, list[anchors.Anchor]] = {}
        for a in live(base_checker.run().anchors):
            base_by_doc.setdefault(a.doc, []).append(a)
        head_by_doc: dict[str, list[anchors.Anchor]] = {}
        for a in live(head_checker.run().anchors):
            head_by_doc.setdefault(a.doc, []).append(a)

        report = DriftReport()
        for doc in sorted(base_by_doc):
            if not (head_root / doc).is_file():
                continue  # the doc itself is gone from the head tree
            if base_checker.frozen_reason(doc) is not None:
                continue  # FROZEN, by the existing checker's own lookup
            pairs, unpaired = _pair(
                sorted(base_by_doc[doc], key=lambda a: a.line),
                sorted(head_by_doc.get(doc, []), key=lambda a: a.line),
                _doc_lines(base_root, doc),
                _doc_lines(head_root, doc),
            )
            for b, h in pairs:
                report.verdicts.append(_judge(diff, b, h))
            for b in unpaired:
                report.verdicts.append(
                    Verdict(
                        doc=doc,
                        line=b.line,
                        target=b.target or b.path,
                        old=_shown(b),
                        new=None,
                        kind=NOT_COMPARED,
                        detail="the doc line changed beyond the anchor numbers",
                    )
                )
        report.verdicts.sort(key=lambda v: (v.doc, v.line))
        return report
    finally:
        shutil.rmtree(base_root, ignore_errors=True)
        if head is not None:
            shutil.rmtree(head_root, ignore_errors=True)


def _judge(diff: DiffMap, b: anchors.Anchor, h: anchors.Anchor) -> Verdict:
    """One paired anchor: map the base numbers forward and compare with the tree's."""
    target = b.target or b.path
    verdict = Verdict(
        doc=h.doc,
        line=h.line,
        target=target,
        old=_shown(b),
        new=None,
        kind=CORRECT,
    )
    if h.target != b.target:
        verdict.kind = NOT_COMPARED
        verdict.detail = f"the anchor resolves to {h.target} at the head tree, not {b.target}"
        return verdict
    if target in diff.renames:
        verdict.kind = AMBIGUOUS
        verdict.detail = (
            f"the anchored file was renamed to {diff.renames[target]} in this range — "
            f"a human must re-point the anchor"
        )
        return verdict
    if target in diff.deleted:
        verdict.kind = AMBIGUOUS
        verdict.detail = "the anchored file was deleted in this range — a human decides the citation"
        return verdict

    fd = diff.files.get(target)
    if fd is None or not fd.hunks:
        verdict.kind = UNCHANGED
        return verdict

    # The only undecidable shape: a cited *endpoint* has no image in the range, i.e. the
    # range itself was rewritten away.  A block added or removed *between* two surviving
    # endpoints is decidable — the endpoints map, and the mapped numbers are the answer;
    # [#299]'s re-point extended exactly such ranges (`backend.rs:36-42` -> `36-45`), so
    # calling them ambiguous would report the fix as unresolved.
    mapped_start = fd.old_to_new(b.start)
    mapped_end = fd.old_to_new(b.end) if b.end is not None else None
    if mapped_start is None or (b.end is not None and mapped_end is None):
        verdict.kind = AMBIGUOUS
        verdict.detail = "a cited line was removed by this range — a human must re-point it"
        return verdict
    verdict.new = str(mapped_start) if mapped_end is None else f"{mapped_start}-{mapped_end}"
    if (mapped_start, mapped_end) == (h.start, h.end):
        verdict.kind = CORRECT
        verdict.detail = "already re-pointed"
        return verdict
    verdict.kind = STALE
    if any(hunk.touches(b.start, b.end if b.end is not None else b.start) for hunk in fd.hunks):
        verdict.detail = f"the tree now has this text at {verdict.new} (a cited range grew or shrank)"
    else:
        verdict.detail = f"the tree now has this text at {verdict.new}"
    return verdict


def print_report(report: DriftReport, rev: str, head: str, listing: bool) -> None:
    """Silent on success unless ``--list``; the failure rows carry the whole claim."""
    if listing:
        for verdict in report.verdicts:
            print(verdict.sentence())
        print(
            f"check_anchor_drift: {len(report.verdicts)} anchors over {rev}...{head} · "
            f"{report.compared} compared · {len(report.stale)} stale · "
            f"{len(report.ambiguous)} ambiguous · {len(report.of(NOT_COMPARED))} not compared"
        )
    for verdict in report.stale:
        print(f"STALE ANCHOR  {verdict.sentence()}", file=sys.stderr)
    for verdict in report.ambiguous:
        print(f"AMBIGUOUS     {verdict.sentence()}", file=sys.stderr)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("rev", nargs="?", help="the base revision to diff the head tree against")
    parser.add_argument(
        "--root",
        default=None,
        help="repository root (default: the tree this script lives in)",
    )
    parser.add_argument(
        "--head",
        default=None,
        help="read the head-side anchors from this revision instead of the working tree",
    )
    parser.add_argument("--list", action="store_true", help="print every compared anchor")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="promote the ambiguous (human-must-judge) bucket to a failure",
    )
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args(argv[1:])

    if args.selftest:
        return selftest()
    if not args.rev:
        parser.print_usage(sys.stderr)
        print("check_anchor_drift: a base revision is required", file=sys.stderr)
        return 2

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    if not (root / "docs").is_dir():
        print(f"check_anchor_drift: no docs/ under {root}", file=sys.stderr)
        return 2
    try:
        report = check(root, args.rev, head=args.head)
    except DriftError as error:
        print(f"check_anchor_drift: {error}", file=sys.stderr)
        return 2
    print_report(report, args.rev, args.head or "the working tree", args.list)
    code = report.worst(args.strict)
    if code == 1:
        print(
            f"check_anchor_drift: {len(report.stale)} anchor(s) point at shifted text — "
            f"re-point them by the mapped delta, or cite a symbol instead of a bare range",
            file=sys.stderr,
        )
    return code


# ---------------------------------------------------------------------------
# --selftest: each case builds a real two-commit repository, because the thing
# under test is a diff.  A case that stops failing is a regression in the
# checker; one that starts failing is a regression in the rule.
# ---------------------------------------------------------------------------

#: The base tree.  ``src/thing.rs`` is numbered so a line's identity is its number
#: and an insertion after line 10 shifts every line after it by three.
BASE_FILES = {
    "src/thing.rs": "".join(f"// thing line {i}\n" for i in range(1, 41)),
    "src/rewritten.rs": "".join(f"// rewritten line {i}\n" for i in range(1, 41)),
    "src/other.rs": "".join(f"// other line {i}\n" for i in range(1, 11)),
    "docs/stale.md": "The block `target_symbol` (`src/thing.rs:20-22`) is described here.\n",
    "docs/repointed.md": "See (`src/thing.rs:20-22`) for the re-pointed case.\n",
    "docs/ambiguous.md": "See (`src/rewritten.rs:20-22`) for the rewritten block.\n",
    "docs/unchanged.md": "See (`src/other.rs:3`) for the untouched file.\n",
    "docs/frozen/old.md": "See (`src/thing.rs:20-22`) in a record of a past revision.\n",
}

#: The frozen pattern the selftest passes instead of the real ``FROZEN`` map.
SELFTEST_FROZEN = {"docs/frozen/*.md": "a record of a past revision"}


def _head_files() -> dict[str, str]:
    """The working tree: thing.rs grows by three lines, rewritten.rs rewrites 20-22."""
    thing = "".join(f"// thing line {i}\n" for i in range(1, 11))
    thing += "// new line a\n// new line b\n// new line c\n"
    thing += "".join(f"// thing line {i}\n" for i in range(11, 41))
    rewritten = "".join(f"// rewritten line {i}\n" for i in range(1, 20))
    rewritten += "// the whole cited block was replaced\n"
    rewritten += "".join(f"// rewritten line {i}\n" for i in range(23, 41))
    return {
        "src/thing.rs": thing,
        "src/rewritten.rs": rewritten,
        "docs/repointed.md": "See (`src/thing.rs:23-25`) for the re-pointed case.\n",
    }


@contextlib.contextmanager
def _hermetic_git():
    """Commit deterministically and unsigned, ignoring the maintainer's git config."""
    keys = (
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_SYSTEM",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
        "GIT_COMMITTER_DATE",
    )
    saved = {key: os.environ.get(key) for key in keys}
    os.environ.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_AUTHOR_NAME": "anchor-drift selftest",
            "GIT_AUTHOR_EMAIL": "selftest@example.invalid",
            "GIT_AUTHOR_DATE": "2026-01-01T00:00:00 +0000",
            "GIT_COMMITTER_NAME": "anchor-drift selftest",
            "GIT_COMMITTER_EMAIL": "selftest@example.invalid",
            "GIT_COMMITTER_DATE": "2026-01-01T00:00:00 +0000",
        }
    )
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _write(root: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def _commit(root: Path, message: str) -> str:
    subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-q", "-m", message],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    return subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _build_case_repo(root: Path) -> str:
    """A two-commit repository; returns the base revision's sha."""
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True)
    _write(root, BASE_FILES)
    base = _commit(root, "base")
    _write(root, _head_files())
    _commit(root, "head")
    return base


def selftest() -> int:
    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    with _hermetic_git(), tempfile.TemporaryDirectory(prefix="anchor-drift-selftest-") as tmp:
        root = Path(tmp)
        base = _build_case_repo(root)
        report = check(root, base, frozen=SELFTEST_FROZEN)

        def kinds(doc: str) -> list[str]:
            return [v.kind for v in report.verdicts if v.doc == doc]

        # 1. The stale case: the number did not move with the text, so it must fail —
        #    and the row must name the mapped numbers.
        expect(kinds("docs/stale.md") == [STALE], "a shifted anchor must be stale")
        stale = report.stale[0] if report.stale else None
        expect(stale is not None and stale.target == "src/thing.rs", "the stale row names the file")
        expect(stale is not None and stale.old == "20-22", "the stale row names the written range")
        expect(stale is not None and stale.new == "23-25", "the stale row names the mapped range")
        expect(
            stale is not None and stale.sentence().endswith("src/thing.rs:20-22 (now 23-25)"),
            "the stale row is `doc:line -> target:old (now new)`",
        )

        # 2. The same move, re-pointed in the doc: correct, and not a failure.
        expect(kinds("docs/repointed.md") == [CORRECT], "a re-pointed anchor must pass")

        # 3. A rewritten block: the map cannot decide, so it is reported, not failed.
        expect(kinds("docs/ambiguous.md") == [AMBIGUOUS], "a rewritten block must be ambiguous")
        ambiguous_only = DriftReport(verdicts=[v for v in report.verdicts if v.kind == AMBIGUOUS])
        expect(not ambiguous_only.worst(False), "ambiguous alone must not fail the default run")
        expect(ambiguous_only.worst(True) == 1, "--strict must promote ambiguous to a failure")

        # 4. A frozen record is exempt by the same policy, and is never a verdict.
        expect(
            not [v for v in report.verdicts if v.doc.startswith("docs/frozen/")],
            "a frozen record must not be judged",
        )

        # 5. A file the range does not touch passes silently.
        expect(kinds("docs/unchanged.md") == [UNCHANGED], "an untouched target must pass")

        # 6. Exit codes and the silent pass, end to end.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expect(
                main(["check_anchor_drift.py", "--root", str(root), base]) == 1,
                "exit 1 when an anchor is stale",
            )
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ) as err:
            code = main(["check_anchor_drift.py", "--root", str(root), "HEAD"])
        expect(code == 0, "exit 0 on an empty range")
        expect(err.getvalue() == "", "an empty range must print nothing")

        # 7. The policy is the lever: freeze the moving target's citation and the same
        #    range goes green, with the ambiguous bucket still only a report.
        frozen = {**SELFTEST_FROZEN, "docs/stale.md": "a record of a past revision"}
        quiet = check(root, base, frozen=frozen)
        expect(not quiet.stale, "freezing the doc must clear its stale anchor")
        expect(quiet.worst(False) == 0, "the frozen run must be green")
        expect(bool(quiet.ambiguous), "the ambiguous bucket survives the freeze")
        expect(quiet.worst(True) == 1, "--strict promotes the ambiguous bucket to a failure")

        # 8. A bad revision is an I/O error, not a verdict.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            expect(
                main(["check_anchor_drift.py", "--root", str(root), "no-such-rev"]) == 2,
                "an unknown revision must exit 2",
            )

    if failures:
        for failure in failures:
            print(f"SELFTEST FAIL  {failure}", file=sys.stderr)
        print(f"check_anchor_drift --selftest: {len(failures)} case(s) failed", file=sys.stderr)
        return 1
    print("check_anchor_drift --selftest: every case passes (the checker bites)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
