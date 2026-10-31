#!/usr/bin/env python3
"""Fail on a commit hash a document cites that neither resolves nor says it is history.

`docs/GATE-CONTRACT.md` §"Citations" states the rule: a backticked hash is a locator, and a
document that keeps a hash which cannot resolve must say so where a machine can read it.
Neither of the two locator gates can see one — `check_docs_links.py` reads link targets,
`check_doc_line_anchors.py` reads `path:NNN` — so an unresolvable hash in a table cell or a
sentence is invisible to every other gate ([#443], [#683], [#684]).

The class is not hypothetical. Measured on 2026-10-31 (`master` `24626ad`): 934 backticked
hash-like tokens, 115 unresolvable in 48 distinct hashes — 41 pre-rewrite labels in one
record document (72 occurrences), six llama.cpp hashes in 17 documents (41 occurrences), and
one pre-rewrite label already documented as unresolvable (`b8568cd`, 2 occurrences). Every
one of them is legitimate, which is why the gate needs three classes rather than a bare probe:

  - **resolves** — the token names a commit **HEAD can reach**, not merely an object the
    clone happens to hold. The distinction is measured, not theoretical: an amend or a rebase
    leaves the discarded commit in the object database, so `git cat-file -t` resolves it
    locally while a fresh clone cannot — 14 citations in 10 hashes on this tree. A hash that
    resolves to a non-commit, or to a commit HEAD cannot reach, is a defect unless excused.
  - **ledgered** — `scripts/doc-hashes.toml` pins the hash with a `kind` (`foreign` or
    `pre-rewrite`), the repository it comes from, and a reason. One entry per hash, never per
    line: a line number drifts on any edit above it ([ADR-0027]).
  - **marked** — the document declares, above a block, that the block's unresolvable hashes are
    kept as history:

        <!-- doc-hashes: history -->

    The marker covers **the block that follows it** — the next run of non-blank lines, which is
    one paragraph, one table or one record entry — and it is invisible in the rendered book. It
    excuses an **unresolvable** hash only: a hash that resolves is a locator and no marker can
    exempt it. Block scope is the point, not a detail: #685 had to narrow
    `METAL_OPTIMIZATIONS.md`'s prose caveat because that file interleaves pre-rewrite labels with
    current-history commits, and a document-wide marker would let a mistyped current hash hide
    anywhere in it (ADR-0065). Prose inside a list item cannot carry a marker without splitting
    the list, so a label there belongs in the ledger instead.

Everything else fails, naming the file, line and hash. Both exemptions ratchet, the
`adr-citations.toml` shape: an entry no document cites fails, an entry whose hash now resolves
fails (it is not history any more), an unknown `kind` or an empty reason fails, and a marker
that covers no unresolvable hash fails.

A tree the gate cannot judge is a failure, not a pass: no git, not a work tree, or a
`.git/shallow` clone — where an absent object may simply not have been fetched — exits 2.

Usage:

    python3 scripts/check_doc_hashes.py [--root .] [--ledger FILE] [--report] [--list]
    python3 scripts/check_doc_hashes.py --selftest

Exit codes: 0 clean, 1 a violation, 2 usage or I/O error (including "cannot decide"). Pure and
offline apart from `git`, which answers which objects exist.

[#443]: https://github.com/yusiwen/minfer/issues/443
[#683]: https://github.com/yusiwen/minfer/issues/683
[#684]: https://github.com/yusiwen/minfer/issues/684
[ADR-0027]: https://github.com/yusiwen/minfer/blob/master/docs/adr/0027-a-citation-is-dated-too.md
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import docset

LEDGER = "scripts/doc-hashes.toml"

#: A backticked short hash. The bound is the corpus's own convention: 7 is the shortest
#: abbreviation git will render, 40 is a full object name.
TOKEN = re.compile(r"`([0-9a-f]{7,40})`")

#: A fenced block is a sample of code or output, not a claim about the tree (the same reading
#: `check_doc_counts.py` gives it).
FENCE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")

#: The document marker, on a line of its own. The class vocabulary is deliberately one word:
#: a second class is a decision, not a habit.
MARKER = re.compile(r"^\s*<!--\s*doc-hashes:\s*(.*?)\s*-->\s*$")
MARKER_CLASSES = frozenset({"history"})

#: What a ledger entry may call itself. `foreign` is another repository's history; `pre-rewrite`
#: is this repository's, lost to a history rewrite.
KINDS = frozenset({"foreign", "pre-rewrite"})


class LedgerError(Exception):
    """The ledger is missing, unreadable or malformed — a usage error, never a verdict."""


class GitError(Exception):
    """The tree cannot answer whether an object exists — a usage error, never a pass."""


@dataclass
class Exempt:
    """One pinned hash: what it is, where it comes from, and why it stays."""

    hash: str
    kind: str
    source: str
    reason: str


@dataclass
class Citation:
    """One backticked hash, its line, and what the gate decided about it."""

    doc: str
    line: int
    hash: str
    text: str
    state: str = ""  # commit | ledger: <source> | marker | a failure reason
    why: str = ""


@dataclass
class Marker:
    """One `<!-- doc-hashes: history -->` line and the block of lines it excuses (1-based)."""

    doc: str
    line: int
    start: int
    end: int  # exclusive
    covered: int = 0

    def covers(self, line: int) -> bool:
        return self.start <= line < self.end


@dataclass
class Report:
    citations: list[Citation] = field(default_factory=list)
    failures: list[Citation] = field(default_factory=list)
    stale_ledger: list[tuple[Exempt, str]] = field(default_factory=list)
    stale_markers: list[tuple[str, int]] = field(default_factory=list)
    seen: int = 0
    resolved: int = 0
    unresolved: int = 0

    def clean(self) -> bool:
        return not (self.failures or self.stale_ledger or self.stale_markers)


def load_ledger(path: Path) -> list[Exempt]:
    """The `[[exempt]]` entries, each complete and unique, or a usage error."""
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as error:
        raise LedgerError(f"cannot read {path}: {error.strerror or error}") from error
    try:
        data = tomllib.loads(raw)
    except tomllib.TOMLDecodeError as error:
        raise LedgerError(f"{path}: {error}") from error
    entries: list[Exempt] = []
    seen: set[str] = set()
    for position, entry in enumerate(data.get("exempt", []), start=1):
        pinned = str(entry.get("hash", "")).strip().lower()
        kind = str(entry.get("kind", "")).strip()
        source = str(entry.get("source", "")).strip()
        reason = str(entry.get("reason", "")).strip()
        where = f"{path}: exempt #{position}"
        if not re.fullmatch(r"[0-9a-f]{7,40}", pinned):
            raise LedgerError(f"{where}: `hash` must be 7-40 lower-case hex characters")
        if kind not in KINDS:
            raise LedgerError(
                f"{where}: `kind` must be one of {', '.join(sorted(KINDS))} (got {kind!r})"
            )
        if not source:
            raise LedgerError(f"{where}: `source` must name where the hash comes from")
        if not reason:
            raise LedgerError(f"{where}: `reason` must be non-empty")
        if pinned in seen:
            raise LedgerError(f"{where}: `{pinned}` is pinned twice")
        seen.add(pinned)
        entries.append(Exempt(pinned, kind, source, reason))
    return entries


def git(root: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    """Run one git command against `root`, turning a missing binary into a usage error."""
    try:
        return subprocess.run(
            ["git", "-C", str(root), *args],
            input=stdin,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise GitError("`git` is not on PATH, so no hash can be resolved") from error


def require_usable(root: Path) -> None:
    """Refuse a tree that cannot answer whether an object exists.

    A shallow clone is the dangerous one: it has history, just not all of it, so every object
    above the graft point reads as missing and the gate would fail on hashes that are real.
    """
    inside = git(root, "rev-parse", "--is-inside-work-tree")
    if inside.returncode != 0 or inside.stdout.strip() != "true":
        detail = inside.stderr.strip() or "no answer"
        raise GitError(f"{root} is not a git work tree ({detail})")
    shallow = git(root, "rev-parse", "--git-path", "shallow")
    if shallow.returncode != 0:
        raise GitError(f"cannot locate the shallow marker under {root}: {shallow.stderr.strip()}")
    marker = Path(shallow.stdout.strip())
    if not marker.is_absolute():
        marker = root / marker
    if marker.exists():
        raise GitError(
            f"{root} is a shallow clone ({marker}) — an absent object may simply never have "
            "been fetched; fetch the full history (`fetch-depth: 0`) and re-run"
        )


def reachable(root: Path) -> set[str]:
    """Every commit HEAD can reach, by full object name.

    Existence is not the test, and that is measured rather than assumed: a clone that has done
    an amend or a rebase keeps the discarded commits as unreachable objects, so `git cat-file -t`
    resolves them while a fresh clone does not. On this tree the difference was 14 citations in
    10 hashes — the amended-away twins the CUDA records already call unresolvable — which a
    local-only probe silently admitted. A citation is a locator in the history at this revision,
    the same rule `check_status.py` applies to its baseline commits.
    """
    listing = git(root, "rev-list", "HEAD")
    if listing.returncode != 0:
        raise GitError(f"`git rev-list HEAD` failed under {root}: {listing.stderr.strip()}")
    return set(listing.stdout.split())


def resolve(root: Path, hashes: list[str]) -> dict[str, tuple[str | None, str]]:
    """`{hash: (status, why)}` for every distinct token, in two git processes.

    One `--batch-check` rather than one process per token, plus one `rev-list`: the work is a
    few hundred tokens today, so the gate stays bounded by the corpus, not by the runner
    (gate-contract rule 4).

    Status is `commit` (reachable from HEAD), the object's own type when it exists but is not a
    commit, `unreachable` when it is a commit HEAD cannot reach, or `None` when git has no such
    object at all. The last two are defeated only by the ledger or a document marker.
    """
    if not hashes:
        return {}
    batch = git(root, "cat-file", "--batch-check", stdin="\n".join(hashes) + "\n")
    if batch.returncode != 0:
        raise GitError(f"`git cat-file --batch-check` failed: {batch.stderr.strip()}")
    live = reachable(root)
    out: dict[str, tuple[str | None, str]] = {}
    lines = batch.stdout.splitlines()
    if len(lines) != len(hashes):
        raise GitError(
            f"`git cat-file --batch-check` answered {len(lines)} line(s) for "
            f"{len(hashes)} object name(s)"
        )
    for name, line in zip(hashes, lines):
        parts = line.split()
        # "<sha> <type> <size>" when the object exists; "<input> missing" (and "<input>
        # ambiguous" for a prefix git cannot pick) otherwise.
        if len(parts) >= 3 and parts[1] in {"commit", "tree", "blob", "tag"}:
            if parts[1] != "commit":
                out[name] = (parts[1], "")
            elif parts[0] in live:
                out[name] = ("commit", "")
            else:
                out[name] = (
                    "unreachable",
                    "a commit HEAD cannot reach — an amended-away twin, or a branch never merged",
                )
        else:
            out[name] = (None, " ".join(parts[1:]) or "missing")
    return out


def scan(root: Path, files: list[Path], entries: list[Exempt], report: Report) -> None:
    """Read every document, decide every hash, and fill `report` (failures, classes, ratchets)."""
    by_hash = {entry.hash: entry for entry in entries}
    covered = {entry.hash: False for entry in entries}
    pending: list[Citation] = []
    markers: dict[str, list[Marker]] = {}

    for path in files:
        rel = path.relative_to(root).as_posix()
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        fence: str | None = None
        declared: list[int] = []  # 0-based marker lines, outside fences
        for index, text in enumerate(lines):
            opener = FENCE.match(text)
            if fence is not None:
                if opener is not None and opener.group(1)[0] == fence[0]:
                    fence = None
                continue
            if opener is not None:
                fence = opener.group(1)
                continue
            found = MARKER.match(text)
            if found is not None:
                name = found.group(1).strip()
                if name not in MARKER_CLASSES:
                    report.failures.append(
                        Citation(
                            rel,
                            index + 1,
                            "",
                            text.strip(),
                            why=(
                                f"unknown marker class {name!r}; the vocabulary is "
                                f"{', '.join(sorted(MARKER_CLASSES))}"
                            ),
                        )
                    )
                else:
                    declared.append(index)
                continue
            for token in TOKEN.finditer(text):
                pending.append(Citation(rel, index + 1, token.group(1), text.strip()))

        # A marker covers the block that follows it: the next run of non-blank lines, once any
        # further marker lines have been skipped. A marker with no block behind it declares
        # nothing and is itself a failure.
        for index in declared:
            start = index + 1
            while start < len(lines) and (
                not lines[start].strip() or MARKER.match(lines[start])
            ):
                start += 1
            if start >= len(lines):
                report.failures.append(
                    Citation(
                        rel,
                        index + 1,
                        "",
                        lines[index].strip(),
                        why="covers no block — a marker must sit above the block it excuses",
                    )
                )
                continue
            end = start
            while end < len(lines) and lines[end].strip():
                end += 1
            markers.setdefault(rel, []).append(Marker(rel, index + 1, start + 1, end + 1))

    # The ledger's own hashes go into the same batch: "the entry resolves now" and "nothing
    # cites it" are different findings, and only a resolution can tell them apart.
    state = resolve(root, sorted({citation.hash for citation in pending} | set(by_hash)))

    for citation in pending:
        report.citations.append(citation)
        report.seen += 1
        kind, why = state[citation.hash]
        if kind == "commit":
            report.resolved += 1
            citation.state = "commit"
            continue
        report.unresolved += 1
        if kind not in (None, "unreachable"):
            citation.why = f"resolves to a {kind}, not a commit"
            citation.state = citation.why
            report.failures.append(citation)
            continue
        detail = "unresolvable" if kind is None else f"unresolvable ({why})"
        entry = by_hash.get(citation.hash)
        if entry is not None:
            covered[entry.hash] = True
            citation.state = f"ledger: {entry.source}"
            continue
        marker = next(
            (m for m in markers.get(citation.doc, []) if m.covers(citation.line)), None
        )
        if marker is not None:
            marker.covered += 1
            citation.state = "marker"
            continue
        citation.why = (
            f"{detail} and unexcused — pin it in {LEDGER}, or put "
            "`<!-- doc-hashes: history -->` above the block"
        )
        citation.state = citation.why
        report.failures.append(citation)

    for entry in entries:
        if covered[entry.hash]:
            continue
        why = (
            "its hash resolves to a commit now, so it is not history"
            if state.get(entry.hash, (None, ""))[0] == "commit"
            else "no document cites it any more"
        )
        report.stale_ledger.append((entry, why))

    for doc, found in markers.items():
        for marker in found:
            if not marker.covered:
                report.stale_markers.append((doc, marker.line))


def check(
    root: Path,
    ledger: Path,
    files: list[Path] | None = None,
    report_out: Report | None = None,
) -> int:
    """The whole check; returns the exit code (0 clean, 1 violations, 2 usage)."""
    try:
        entries = load_ledger(ledger)
        require_usable(root)
        if files is None:
            files = docset.scanned(root)
        report = Report()
        scan(root, files, entries, report)
    except (LedgerError, GitError) as error:
        print(f"check_doc_hashes: {error}", file=sys.stderr)
        return 2
    if report_out is not None:
        report_out.citations = report.citations
        report_out.failures = report.failures
        report_out.stale_ledger = report.stale_ledger
        report_out.stale_markers = report.stale_markers
        report_out.seen = report.seen
        report_out.resolved = report.resolved
        report_out.unresolved = report.unresolved
    return 0 if report.clean() else 1


def print_report(report: Report) -> None:
    for citation in report.failures:
        where = f"{citation.doc}:{citation.line}"
        name = f" `{citation.hash}`" if citation.hash else ""
        print(f"{where}{name}  {citation.why}")
        if citation.text:
            print(f"    {citation.text[:160]}")
    for entry, why in report.stale_ledger:
        print(
            f"STALE LEDGER  hash {entry.hash}: {why} — remove the entry "
            f"(reason was: {entry.reason})",
            file=sys.stderr,
        )
    for rel, line in report.stale_markers:
        print(
            f"STALE MARKER  {rel}:{line}: covers no unresolvable hash — remove it",
            file=sys.stderr,
        )
    print(
        f"check_doc_hashes: {report.seen} hash token(s) seen, "
        f"{report.resolved} resolve, {report.unresolved} unresolvable, "
        f"{len(report.failures)} unexcused, {len(report.stale_ledger)} stale ledger "
        f"entry/entries, {len(report.stale_markers)} stale marker(s)"
    )


def selftest() -> int:
    """The checker's own pass/fail cases, in a throwaway repository."""
    cases: list[tuple[str, bool]] = []
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        env = {
            **os.environ,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_AUTHOR_NAME": "selftest",
            "GIT_AUTHOR_EMAIL": "selftest@example.com",
            "GIT_COMMITTER_NAME": "selftest",
            "GIT_COMMITTER_EMAIL": "selftest@example.com",
        }

        def sh(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                input=stdin,
                capture_output=True,
                text=True,
                env=env,
            )

        sh("init", "-q", "-b", "main")
        (root / "seed.txt").write_text("seed\n", encoding="utf-8")
        sh("add", "-f", "seed.txt")
        sh("commit", "-q", "-m", "seed", "--no-gpg-sign")
        head = sh("rev-parse", "HEAD").stdout.strip()
        short = head[:7]
        blob = sh("hash-object", "-w", "--stdin", stdin="blob\n").stdout.strip()
        # A commit HEAD cannot reach but the object database still holds — the amended-away
        # twin a local clone resolves and a fresh clone does not.
        sh("checkout", "-q", "-b", "twin")
        sh("commit", "-q", "--allow-empty", "-m", "twin", "--no-gpg-sign")
        twin = sh("rev-parse", "HEAD").stdout.strip()
        sh("checkout", "-q", "main")
        sh("branch", "-q", "-D", "twin")
        led = root / LEDGER
        led.parent.mkdir(parents=True, exist_ok=True)

        def run(docs: dict[str, str], ledger: str = "") -> int:
            for name, body in docs.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(body, encoding="utf-8")
            led.write_text(ledger, encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                return check(root, led, files=[root / name for name in docs])

        entry = (
            '[[exempt]]\n\nhash = "deadbeef"\nkind = "foreign"\nsource = "llama.cpp"\n'
            'reason = "a foreign repository\'s history"\n'
        )

        cases.append(("a resolvable hash passes", run({"docs/a.md": f"`{short}`\n"}) == 0))
        cases.append(("an unresolvable hash fails", run({"docs/a.md": "`deadbeef`\n"}) == 1))
        cases.append(
            (
                "the marker excuses the block below it",
                run({"docs/a.md": "<!-- doc-hashes: history -->\n\n`deadbeef`\n"}) == 0,
            )
        )
        cases.append(
            (
                "the marker does not reach the block after that",
                run(
                    {
                        "docs/a.md": "<!-- doc-hashes: history -->\n\n"
                        "`deadbeef`\n\nA second paragraph with `cafebabe`.\n"
                    }
                )
                == 1,
            )
        )
        cases.append(
            (
                "a marker that covers nothing fails",
                run(
                    {
                        "docs/a.md": "<!-- doc-hashes: history -->\n\n"
                        f"`{short}`\n"
                    }
                )
                == 1,
            )
        )
        cases.append(
            (
                "a marker with no block behind it fails",
                run({"docs/a.md": f"`{short}`\n\n<!-- doc-hashes: history -->\n"}) == 1,
            )
        )
        cases.append(
            (
                "a ledger entry excuses it",
                run({"docs/a.md": "`deadbeef`\n"}, entry) == 0,
            )
        )
        cases.append(
            (
                "a ledger entry no document cites fails",
                run({"docs/a.md": f"`{short}`\n"}, entry) == 1,
            )
        )
        cases.append(
            (
                "a ledger entry whose hash resolves fails",
                run(
                    {"docs/a.md": f"`{short}`\n"},
                    '[[exempt]]\n\nhash = "%s"\nkind = "foreign"\nsource = "x"\n'
                    'reason = "y"\n' % short,
                )
                == 1,
            )
        )
        cases.append(
            (
                "a marked document keeps a resolvable hash as a locator",
                run(
                    {
                        "docs/a.md": "<!-- doc-hashes: history -->\n\n"
                        f"`{short}` at `deadbeef`\n"
                    }
                )
                == 0,
            )
        )
        cases.append(
            (
                "an unknown marker class fails",
                run({"docs/a.md": "<!-- doc-hashes: ancient -->\n\n`deadbeef`\n"}) == 1,
            )
        )
        cases.append(
            (
                "two markers cover two blocks",
                run(
                    {
                        "docs/a.md": "<!-- doc-hashes: history -->\n\n`deadbeef`\n\n"
                        "<!-- doc-hashes: history -->\n\n`cafebabe`\n"
                    }
                )
                == 0,
            )
        )
        cases.append(
            (
                "a fenced block is not a citation",
                run({"docs/a.md": "```\n`deadbeef`\n```\n"}) == 0,
            )
        )
        cases.append(
            (
                "a marker inside a fenced block declares nothing",
                run(
                    {
                        "docs/a.md": "```html\n<!-- doc-hashes: history -->\n```\n\n"
                        "`deadbeef`\n"
                    }
                )
                == 1,
            )
        )
        cases.append(
            (
                "an object that is not a commit fails",
                run({"docs/a.md": f"`{blob[:9]}`\n"}) == 1,
            )
        )
        cases.append(
            (
                "a commit HEAD cannot reach fails",
                run({"docs/a.md": f"`{twin[:9]}`\n"}) == 1,
            )
        )
        cases.append(
            (
                "an unknown kind is a usage error",
                run(
                    {"docs/a.md": "`deadbeef`\n"},
                    entry.replace('"foreign"', '"alien"'),
                )
                == 2,
            )
        )
        cases.append(
            (
                "an entry without a reason is a usage error",
                run(
                    {"docs/a.md": "`deadbeef`\n"},
                    '[[exempt]]\n\nhash = "deadbeef"\nkind = "foreign"\n'
                    'source = "llama.cpp"\nreason = "   "\n',
                )
                == 2,
            )
        )
        with tempfile.TemporaryDirectory() as plain:
            doc = Path(plain) / "docs" / "a.md"
            doc.parent.mkdir(parents=True)
            doc.write_text("`deadbeef`\n", encoding="utf-8")
            led.write_text("", encoding="utf-8")
            with contextlib.redirect_stderr(io.StringIO()):
                cases.append(
                    ("a directory that is not a git tree is a usage error", check(Path(plain), led) == 2)
                )
        shallow = root / ".git" / "shallow"
        shallow.write_text(head + "\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()):
            cases.append(
                ("a shallow clone is a usage error", check(root, led, files=[root / "seed.txt"]) == 2)
            )
        shallow.unlink()
        (root / "docs" / "b.md").write_text("`deadbeef`\n", encoding="utf-8")
        led.write_text("", encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()):
            cases.append(
                (
                    "--report prints the census and exits 0",
                    main(["check_doc_hashes.py", "--root", str(root), "--report"]) == 0,
                )
            )

    failures = [name for name, ok in cases if not ok]
    for name in failures:
        print(f"SELFTEST FAIL  {name}", file=sys.stderr)
    if failures:
        print(f"check_doc_hashes --selftest: {len(failures)} case(s) failed", file=sys.stderr)
        return 1
    print("check_doc_hashes --selftest: every case passes (the gate bites)")
    return 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=None, help="repository root (default: this tree)")
    parser.add_argument(
        "--ledger",
        default=None,
        help=f"the exemption ledger, relative to --root (default: {LEDGER})",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the census and exit 0 (adoption mode; failures and stale entries still print)",
    )
    parser.add_argument("--list", action="store_true", help="print every hash and its class")
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    if not (root / "docs").is_dir():
        print(f"check_doc_hashes: no docs/ under {root}", file=sys.stderr)
        return 2
    ledger = Path(args.ledger) if args.ledger else root / LEDGER
    if not ledger.is_absolute():
        ledger = root / ledger

    try:
        docset.announce(root)
    except docset.ScopeError as error:
        print(f"check_doc_hashes: {error}", file=sys.stderr)
        return 1

    report = Report()
    code = check(root, ledger, report_out=report)
    if code == 2:
        # A ledger that cannot be read, or a tree that cannot answer, is not a verdict:
        # printing a census of "0 seen" would read as a pass.
        return 2
    if args.list:
        for citation in report.citations:
            print(f"{citation.doc}:{citation.line}  `{citation.hash}`  {citation.state}")
    print_report(report)
    if args.report:
        return 0
    if code == 1:
        sys.stdout.flush()
        print(
            "check_doc_hashes: a hash is a locator — resolve it, pin it in "
            f"{LEDGER}, or put `<!-- doc-hashes: history -->` above the block "
            "(see docs/GATE-CONTRACT.md §Citations)",
            file=sys.stderr,
        )
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
