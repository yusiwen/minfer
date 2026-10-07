#!/usr/bin/env python3
"""Fail on a `path:NNN` line anchor in the documentation that no longer points anywhere.

`scripts/check_docs_links.py` resolves markdown link *targets*, so a moved line rots
silently: the link still resolves, the number it names does not. Measured on master
`6b6d94f`, the docs carry ~1 900 `path:NNN` tokens, ~1 350 of them into files of this
repository — 400 into the four backend files (`src/cuda.rs`, `src/cuda_kernels.cu`,
`src/metal.rs`, `src/metal.metal`) that the source-layout campaign moves (issue #261,
`docs/SOURCE-LAYOUT-PLAN.md`). This script is the missing half (issue #266).

What it checks, for every anchor found in `docs/**/*.md` and `AGENTS.md`:

``A`` **the path resolves.** A written path is matched against the tree by exact
repository-relative path, then by path suffix, then by unique basename (`cuda.rs` →
`src/cuda.rs`). A path that resolves to *more than one* file (`` `tests.rs:32` ``)
is reported as ``ambiguous`` and skipped — guessing would be worse than abstaining.
A path that resolves to nothing is judged by one rule, and only one:

  - written with a directory separator, a repository-owned extension and a
    first component that exists at the repository root (`src/gone.rs`) → **missing**
    (a violation: the tree is the unit here, and this is a claim about the tree);
  - anything else (`ggml-cuda.cu`, `common/common.h`, `mmq.cuh`) → ``external``,
    because the docs legitimately cite the llama.cpp reference tree by basename and
    share its extensions. External anchors are counted and `--list` shows them; the
    honest limit is written down under "Boundary" below.

``B`` **the number is a real line.** `NNN` must be within the file, and for a range
`NNN-MMM` both ends must be, with `NNN <= MMM`.

``C`` **the named symbol is mentioned in the file the anchor points at** — a heuristic,
spelled out here because it is one. When the anchor's own span is adjacent to a
backticked identifier (nothing between them but `` ` `` `(` `)` `,` `;` `:` `.`
whitespace, at most 6 characters), that identifier is the anchor's *named symbol*.
A trailing ``()`` is part of the token, not a disqualifier: `` `metal_available()` ``
names the symbol `` `metal_available` `` (issue #339), because the docs write a
function with its call parentheses whenever the sentence is about calling it.
The pass condition is the symbol appearing within ``SYMBOL_WINDOW`` lines either side
of the anchor line. A miss is classified by where the symbol actually is, which is
what keeps the heuristic honest rather than loud:

  - nowhere in the target file but somewhere under `src/` → **symbol-moved**: the
    anchor names a repository item that lives in another file. This is the
    extracted-`tests.rs` rot (`metal_pipelines_compile` moved to
    `src/metal/tests.rs`) and it *fails*.
  - in the target file, but outside the window → ``symbol-far``: reported, not a
    failure. Prose routinely names a neighbouring item ("the wrapper `x` … its
    caller `y` (`file.rs:123`)"), so a distance is a smell, not a proof.
  - nowhere under `src/` at all → ``symbol-foreign``: reported, not a failure. Most
    of these are C / CUDA / Metal / llama.cpp API names (`cudaFuncSetAttribute`,
    `MTLDispatchTypeConcurrent`) or a kernel named in a comment; a checker that
    cannot tell an FFI name from a renamed Rust item must not fail on absence.

Both non-failing classes are listed by `--list`, printed in the summary, and promoted
to failures by `--strict-symbols` — so "why did this pass?" always has an answer. Rule
E's range miss joins them on the same terms.

``D`` **the continuation is range-checked too.** A backticked span whose whole content
is `:NNN` or `:NNN-MMM` — a second range in the file the line already cited — attaches
to the nearest *preceding* `path.ext:NNN` match on the same line and is judged against
**that** file: the same resolution (rule A) and the same range test (rule B). A line
with no preceding anchor, or a continuation that precedes every anchor on its line,
stays silent; a continuation whose path anchor is itself external, ambiguous or frozen
is classified the same way. This closes issue #355: the bare form was invisible, so a
re-point that fixed the visible `path:NNN` left its continuation behind.

``E`` **the range holds the symbol too** — a *note*, counted and listed, not a failure
(issue #339). Rule C's window is deliberately loose (±25 lines), so an anchor like
`` `metal_available()` (`metal_backend.rs:1054-1056`) `` passes while the definition is
700 lines away. Rule E therefore asks the narrower question the citation actually
claims — does the identifier occur **inside** `start-end` (for a bare `path:NNN`, on
that one line)? It is a *note* and not a failure because of what that rule costs
today: measured on `7991218`, **79** non-frozen anchors carry an adjacent backticked
symbol whose identifier is not inside their own range (**84** once a trailing `()` is
stripped; 46 of the 84 cite a `NNN-MMM` range, 38 a single line, and 81 of the 84 hold
the identifier in the target file at all, which is what makes them range misses). A
rule that failed on that population would have to re-point every one of them in the
PR that adds it — and 26 of the 84 live in walkthrough docs that state the revision
their lines were verified against (`lines verified at commit e7fa0da`), where a
re-point would falsify the record. Two of the three anchors that motivated the rule
show why the in-range test is not the whole answer anyway: `:256`'s `execute_node`
anchor cites `316-1020`, which **does** contain the definition at 843 — containment
cannot see a range that starts 527 lines early — and `:586`'s `synchronize` anchor has
no adjacent symbol at all, so no adjacency rule can see it. Those two were re-pointed
by hand.
A *call-site* citation is not a miss: if the range names the symbol at a call, the
identifier is inside the range and the anchor passes. The convention this note
enforces is written in ``docs/GATE-CONTRACT.md`` §"Prose anchors"; the sweep of the
80 is tracked by [#336] and [#356] alongside the bare ranges.

**Frozen records.** A frozen-file set (the ``GRANDFATHERED_BARE`` pattern of
`scripts/check_dead_code_annotations.py`) exempts the historical records whose anchors
were written against a revision that has since moved: rewriting them would falsify a
measurement, so the record keeps its text and the checker abstains. Every entry carries
a one-line reason, and the set is a ratchet *downwards*: an entry that no longer covers
a file carrying a single anchor fails (``STALE FREEZE``), like a grandfather key that no
longer covers a bare `allow`. An entry whose files all resolve today is printed as
``UNUSED FREEZE`` but does not fail — the exemption is a policy for a record of a past
revision, not a claim that the record is broken today. The exemption covers
continuations exactly as it covers path anchors: a frozen record's bare `:NNN` is
`frozen`, never a violation.

**Boundary (what this does not catch).** A bare basename that names nothing in this
tree is treated as an external citation, so now that the campaign's Step 2 has retired
`src/cuda_kernels.cu` ([#263](https://github.com/yusiwen/minfer/issues/263)), the *bare*
`cuda_kernels.cu:NNN` form passes while the `src/cuda_kernels.cu:NNN` form fails. That is the deliberate cost of not guessing
between this tree and llama.cpp's, whose files share the `.cu`/`.cuh` extensions;
converting the live anchors to symbol anchors (ticket #266 ¶c) is what removes them.

A bare range whose *content* has drifted is invisible here by construction, and no
rule closes that gap: measured for ticket #327, 99 of the 144 live anchors that
carry an adjacent backticked **non-symbol** token do not contain that token inside
their own range — the tokens are code expressions and fragments of a neighbouring
sentence, not the claim. The convention that replaces the missing rule — a prose
anchor names a **section heading** next to its range, and a symbol anchor is
preferred where one exists — is written down in ``docs/GATE-CONTRACT.md``
§"Prose anchors: name the section, not the line range".

Rule D's **window is the doc line, and it is one-directional**. Measured on `f754ee3`
with the issue's criterion (a backticked span whose content is exactly `:NNN` /
`:NNN-MMM`, on a doc line that carries a `path.ext:NNN` match), the docs carry **58**
such spans: **38** resolve, **13** are out of range (2 of them in a frozen record) and
**6** follow no anchor on their line (`:1752`, `:1849` in
`docs/ARCHITECTURE-EXECUTION-PLAN.md`; `:1289-1321` twice in
`docs/ARCHITECTURE-ROADMAP.md`; `:2045-2052`, `:2054-2063` in
`docs/inference_e2e_walkthrough/14-metal-backend.md`), while one
(`docs/inference_e2e_walkthrough/14-metal-backend.md:499`) inherits its path anchor's
`ambiguous` verdict. Those seven stay silent deliberately: in each, the file is named
in the *sentence before* the number, or the path anchor is on the same line but *after*
it, and attaching the number to the nearest anchor in either direction would be a
guess with a wrong answer available (the roadmap pair belongs to a file named on the
previous line, not to the `metal_backend.rs` anchor that follows it). The window is
stated once in ``docs/GATE-CONTRACT.md`` §"A bare `:NNN` continuation attaches to the
anchor before it"; widening it is the follow-up the boundary names.

Usage::

    python3 scripts/check_doc_line_anchors.py [--root DIR] [--list] [--strict-symbols]
    python3 scripts/check_doc_line_anchors.py --selftest

Exit codes: 0 clean, 1 violations, 2 usage or I/O error.

See ``docs/SOURCE-LAYOUT-PLAN.md`` §6 for the policy this implements.
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import io
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

#: Extensions this repository owns as *source*. A path that does not resolve is only
#: judged under rule A when it ends in one of these — `.cpp`/`.h`/`.c` are llama.cpp's
#: tree, never this one.
REPO_EXTENSIONS = (
    "rs",
    "cu",
    "cuh",
    "metal",
    "toml",
    "py",
    "sh",
    "yml",
    "yaml",
    "md",
)

#: `path:NNN`, `path:NNN-MMM`, with an optional whitespace-free path. The leading
#: `(?<![\w./-])` keeps a match from starting mid-token (`docs/x.md:12` in
#: `a/docs/x.md:12` is one anchor, not two).
ANCHOR = re.compile(
    r"(?<![\w./-])(?P<path>[\w][\w./-]*\.(?:" + "|".join(REPO_EXTENSIONS) + r"))"
    r":(?P<start>\d+)(?:-(?P<end>\d+))?"
)

#: One backticked span.
CODE_SPAN = re.compile(r"`([^`]+)`")

#: A **bare continuation**: a backticked span whose whole content is `:NNN` or
#: `:NNN-MMM`, i.e. a second range in the file the line already cited. The attachment
#: rule is stated once in `docs/GATE-CONTRACT.md` §"A bare `:NNN` continuation attaches
#: to the anchor before it": the continuation attaches to the **nearest preceding**
#: `path.ext:NNN` match on the **same doc line**, and a line with no such match — or a
#: continuation that comes before every anchor on its line — stays silent, because
#: there is no file to attribute the number to. The `ANCHOR` grammar requires the
#: path, which is why the continuation was invisible before issue #355.
CONTINUATION = re.compile(r"^:(\d+)(?:-(\d+))?$")

#: A symbol-shaped token: a bare identifier or a `Type::item` path, starting with a
#: letter (never `_foo`) and looking like code rather than prose — it must carry an
#: underscore, a `::`, an inner capital or be a `CONSTANT`. That last clause is what
#: keeps the English words these docs wrap in backticks (`` `returns` ``, `` `and` ``)
#: from being read as symbols and failed for "moving".
SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*(?:::[A-Za-z][A-Za-z0-9_]*)*$")

#: The call parentheses of a function citation: `` `metal_available()` `` names the
#: symbol `metal_available` (issue #339), and the identifier — not the call — is what
#: the checker greps for.
CALL_SUFFIX = re.compile(r"\(\)$")


def symbol_base(token: str) -> str:
    """The identifier a backticked token names: `f()`, `Type::f`, `f` → `f`."""
    return CALL_SUFFIX.sub("", token.strip()).split("::")[-1]


def looks_like_symbol(token: str) -> bool:
    """Whether a backticked token is a code symbol rather than a prose word."""
    token = CALL_SUFFIX.sub("", token.strip())
    if not SYMBOL.match(token) or "." in token:
        return False
    base = token.split("::")[-1]
    if "_" in token or "::" in token:
        return True
    if re.search(r"[A-Z]", base[1:]):
        return True
    return base.isupper() and len(base) >= 3


#: Lines either side of the anchor in which the named symbol must appear (rule C).
SYMBOL_WINDOW = 25

#: Characters allowed between the symbol and the anchor for the two to be adjacent.
ADJACENT_GAP = 6
GAP_FILLER = re.compile(r"^[\s`()\[\]*,;:.–—-]*$")

#: The frozen historical records, ``pattern`` → one-line reason. Rewriting these would
#: falsify a measurement taken against the revision named in the record
#: (`docs/SOURCE-LAYOUT-PLAN.md` §6.2). The set is a ratchet *downwards*: an entry
#: that no longer covers a file carrying a single anchor fails (``stale_frozen``,
#: exit 1), and an entry whose anchors all happen to resolve today is reported as
#: ``unused_frozen`` — the reason for freezing is the policy for a record of a past
#: revision, not the accident that today's line numbers still line up.
FROZEN = {
    "docs/cuda_optimization_steps/*.md": (
        "per-step CUDA optimisation records: each states the tree revision it was "
        "measured against, and its anchors are into that revision's "
        "src/cuda_kernels.cu / src/cuda.rs before the Step 1/2 split"
    ),
    "docs/QWEN2.5-*.md": (
        "dated Qwen2.5 debugging records; their anchors locate code as it stood on "
        "the day of the session, not the current tree"
    ),
    "docs/KNOWN-CPU-ISSUES-*.md": (
        "dated CPU-issue record written against a pre-tests.rs-extraction revision"
    ),
    "docs/CPU_OPTIMIZATIONS.md": (
        "declared layered record: its §Status paragraph states that everything above "
        "§Conclusion is a 2026-07-01 snapshot whose src/models/qwen2/forward.rs:NNN "
        "references name a file deleted in the compute-graph Phase 6"
    ),
    "docs/PARAMETER_AUDIT.md": (
        "2026-08-06 parameter audit against a named llama.cpp commit and the "
        "then-current metal.rs; SOURCE-LAYOUT-PLAN §6.2 names its older tables "
        "as a frozen historical record"
    ),
    "docs/ARCHITECTURE-EXECUTION-PLAN.md": (
        "per-ticket history: each entry quotes the issue body, its acceptance "
        "criteria and panic transcripts verbatim from the revision they were "
        "written against (SOURCE-LAYOUT-PLAN §6.2)"
    ),
    "docs/METAL-OBJC2-MIGRATION-PLAN.md": (
        "completed objc 0.2 → objc2 migration record (its Status line reads "
        "Implemented, 2026-08-25): the two `Cargo.toml:38-43` anchors name the "
        "`[lints.rust] unexpected_cfgs` block that commit 6a382a3 deleted, so they "
        "cite the pre-execution file and cannot be re-pointed — the block is gone, "
        "and no revision of Cargo.toml ever held it at 38-43 — and the five "
        "`metal_backend.rs:NNN` anchors name that file's 2026-08-25 revision, which "
        "~20 commits have moved since (issue #344)"
    ),
}

#: Directories that exist at the repository root today. Rule A only judges a
#: non-resolving path whose first component is one of these.
ROOT_DIRS = ("src", "docs", "scripts", "tests", "viz", "benches", "experiments")


@dataclass
class Anchor:
    """One `path:NNN` token, its resolution and the verdict on it.

    ``continuation`` marks a bare `:NNN` span: ``path`` is then the *written* path of
    the anchor it attached to, ``start``/``end`` are its own numbers, and the verdict
    is judged against that anchor's resolved target (rule D).
    """

    doc: str
    line: int
    path: str
    start: int
    end: int | None
    target: str | None = None
    symbol: str | None = None
    continuation: bool = False
    verdict: str = "ok"
    detail: str = ""
    #: Rule E (#339): the first line of the target file that holds the named symbol,
    #: when that identifier is **not** inside the cited range. ``None`` when the
    #: anchor carries no symbol, when the symbol is inside the range, or when it is
    #: not in the file at all (that case is ``symbol-moved`` / ``symbol-foreign``
    #: already). Excluded from ``compare`` so equality is unchanged by judging.
    range_miss: int | None = field(default=None, compare=False)

    def where(self) -> str:
        return f"{self.doc}:{self.line}"

    def shown(self) -> str:
        rng = f"{self.start}-{self.end}" if self.end is not None else f"{self.start}"
        return f":{rng}" if self.continuation else f"{self.path}:{rng}"

    def is_violation(self, strict_symbols: bool = False) -> bool:
        if self.verdict in ("missing", "out-of-range", "range-order", "symbol-moved"):
            return True
        return strict_symbols and (
            self.verdict in ("symbol-far", "symbol-foreign") or self.range_miss is not None
        )


@dataclass
class Report:
    anchors: list[Anchor] = field(default_factory=list)
    stale_frozen: list[tuple[str, str]] = field(default_factory=list)
    unused_frozen: list[tuple[str, str]] = field(default_factory=list)
    #: Bare continuations that attach to no preceding anchor on their line: the
    #: documented silent class (rule D), counted so "why did this pass?" has an answer.
    unattached: int = 0

    def violations(self, strict_symbols: bool = False) -> list[Anchor]:
        return [a for a in self.anchors if a.is_violation(strict_symbols)]

    def count(self, verdict: str) -> int:
        return sum(1 for a in self.anchors if a.verdict == verdict)


def doc_files(root: Path) -> list[Path]:
    """Every file the gate covers, in a stable order."""
    files = sorted((root / "docs").rglob("*.md"))
    agents = root / "AGENTS.md"
    if agents.is_file():
        files.append(agents)
    return sorted(files)


def build_index(root: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Repository-relative file paths plus a basename index for rule A."""
    paths: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if rel.parts and rel.parts[0] in (".git", ".worktrees", "target"):
            continue
        paths.append(rel.as_posix())
    by_basename: dict[str, list[str]] = {}
    for rel in paths:
        by_basename.setdefault(Path(rel).name, []).append(rel)
    return sorted(paths), by_basename


def resolve(path: str, root: Path, paths: list[str], by_basename: dict[str, list[str]]) -> list[str]:
    """Every repository file ``path`` could name, in the tree's spelling."""
    if (root / path).is_file():
        return [path]
    suffix = "/" + path
    matches = [rel for rel in paths if rel == path or rel.endswith(suffix)]
    if matches:
        return matches
    return by_basename.get(Path(path).name, [])


def is_missing_path(path: str, root: Path) -> bool:
    """Rule A's one judged case: a tree-shaped claim that names no file."""
    if "/" not in path:
        return False
    head, _, tail = path.partition("/")
    if head not in ROOT_DIRS:
        return False
    if not (root / head).is_dir():
        return False
    return tail != ""


def symbol_of(line: str, span: tuple[int, int]) -> str | None:
    """The backticked symbol adjacent to the anchor at ``span``, if any.

    "Adjacent" is a documented rule, not a guess: only fillers (punctuation and
    whitespace) may sit between the two spans, at most ``ADJACENT_GAP`` characters,
    so a table cell's other columns or an earlier sentence never donate a symbol.
    """
    before, after = line[: span[0]], line[span[1] :]
    for match in reversed(list(CODE_SPAN.finditer(before))):
        token = match.group(1).strip()
        gap = before[match.end() :]
        if len(gap) > ADJACENT_GAP or not GAP_FILLER.match(gap):
            break
        if looks_like_symbol(token):
            return token
        break
    for match in CODE_SPAN.finditer(after):
        token = match.group(1).strip()
        gap = after[: match.start()]
        if len(gap) > ADJACENT_GAP or not GAP_FILLER.match(gap):
            break
        if looks_like_symbol(token):
            return token
        break
    return None


class Checker:
    """One run over one tree: resolve, range-check and symbol-check every anchor."""

    def __init__(self, root: Path, frozen: dict[str, str] | None = None) -> None:
        self.root = root
        self.paths, self.by_basename = build_index(root)
        self.frozen = dict(FROZEN if frozen is None else frozen)
        self._lines: dict[str, list[str]] = {}
        self._src_text: str | None = None
        self._unattached = 0

    # -- helpers ---------------------------------------------------------------

    def lines_of(self, rel: str) -> list[str]:
        if rel not in self._lines:
            self._lines[rel] = (
                (self.root / rel).read_text(encoding="utf-8", errors="replace").splitlines()
            )
        return self._lines[rel]

    def src_text(self) -> str:
        """Every `src/**` file, concatenated — the "does this symbol exist here" probe."""
        if self._src_text is None:
            chunks = []
            for rel in self.paths:
                if rel.startswith("src/"):
                    chunks.append(
                        (self.root / rel).read_text(encoding="utf-8", errors="replace")
                    )
            self._src_text = "\n".join(chunks)
        return self._src_text

    def frozen_reason(self, doc: str) -> str | None:
        for pattern, reason in self.frozen.items():
            if fnmatch.fnmatch(doc, pattern):
                return reason
        return None

    def anchors_on_line(self, rel: str, number: int, text: str) -> list[tuple[Anchor, tuple[int, int] | None]]:
        """The anchors one doc line carries, in written order, with a symbol span.

        A `path.ext:NNN` match is an anchor with its own span (rule C reads the
        backticked token next to it). A backticked span whose whole content is
        `:NNN` / `:NNN-MMM` is a bare continuation (rule D): it attaches to the
        nearest *preceding* path anchor on this line and is judged against that
        anchor's written path. A continuation with no such anchor is dropped — the
        documented silent class — and counted on the report.
        """
        matches = list(ANCHOR.finditer(text))
        entries: list[tuple[int, Anchor, tuple[int, int] | None]] = []
        for match in matches:
            entries.append(
                (
                    match.start(),
                    Anchor(
                        doc=rel,
                        line=number,
                        path=match.group("path"),
                        start=int(match.group("start")),
                        end=int(match.group("end")) if match.group("end") else None,
                    ),
                    match.span(),
                )
            )
        for span in CODE_SPAN.finditer(text):
            bare = CONTINUATION.match(span.group(1).strip())
            if bare is None:
                continue
            preceding = [m for m in matches if m.start() < span.start()]
            if not preceding:
                self._unattached += 1
                continue
            entries.append(
                (
                    span.start(),
                    Anchor(
                        doc=rel,
                        line=number,
                        path=preceding[-1].group("path"),
                        start=int(bare.group(1)),
                        end=int(bare.group(2)) if bare.group(2) else None,
                        continuation=True,
                    ),
                    None,
                )
            )
        entries.sort(key=lambda entry: entry[0])
        return [(anchor, span) for _, anchor, span in entries]

    # -- the check -------------------------------------------------------------

    def run(self, files: list[Path] | None = None) -> Report:
        report = Report()
        self._unattached = 0
        # pattern → [has an anchor, has a violation], aggregated over every file it covers.
        ratchet: dict[str, list[bool]] = {pattern: [False, False] for pattern in self.frozen}
        for path in files if files is not None else doc_files(self.root):
            rel = path.relative_to(self.root).as_posix()
            reason = self.frozen_reason(rel)
            lines = path.read_text(encoding="utf-8").splitlines()
            here: list[Anchor] = []
            for number, text in enumerate(lines, start=1):
                for anchor, span in self.anchors_on_line(rel, number, text):
                    if span is not None:
                        anchor.symbol = symbol_of(text, span)
                    self._judge(anchor)
                    here.append(anchor)
            if reason is not None:
                pattern = next(p for p in self.frozen if fnmatch.fnmatch(rel, p))
                if here:
                    ratchet[pattern][0] = True
                if any(a.is_violation(False) for a in here):
                    ratchet[pattern][1] = True
                for anchor in here:
                    anchor.verdict = "frozen"
                    anchor.detail = reason
                    # The exemption covers the whole anchor: a record frozen against a
                    # past revision is not a rule-E miss either.
                    anchor.range_miss = None
            report.anchors.extend(here)
        report.unattached = self._unattached
        # The ratchet. An entry whose files no longer carry a single anchor names a
        # record that is gone or emptied, so the exemption must go with it. An entry
        # whose anchors all happen to resolve today is *reported* (``unused_frozen``)
        # but not failed: the reason for freezing is the policy for a record of a past
        # revision, not the accident that today's line numbers still line up.
        for pattern, (has_anchor, has_violation) in ratchet.items():
            reason = self.frozen[pattern]
            if not has_anchor:
                report.stale_frozen.append((pattern, reason))
            elif not has_violation:
                report.unused_frozen.append((pattern, reason))
        return report

    def _judge(self, anchor: Anchor) -> None:
        candidates = resolve(anchor.path, self.root, self.paths, self.by_basename)
        if not candidates:
            if is_missing_path(anchor.path, self.root):
                anchor.verdict = "missing"
                anchor.detail = "no file of this tree has that path"
            else:
                anchor.verdict = "external"
                anchor.detail = "not a path of this tree (another tree's citation)"
            return
        if len(candidates) > 1:
            anchor.verdict = "ambiguous"
            anchor.detail = "basename matches " + ", ".join(
                sorted(candidates)[:4]
            ) + (" …" if len(candidates) > 4 else "")
            return
        target = candidates[0]
        anchor.target = target
        lines = self.lines_of(target)
        if anchor.end is not None and anchor.end < anchor.start:
            anchor.verdict = "range-order"
            anchor.detail = f"{anchor.start}-{anchor.end} is reversed"
            return
        if anchor.start > len(lines) or (anchor.end is not None and anchor.end > len(lines)):
            anchor.verdict = "out-of-range"
            anchor.detail = f"{target} has {len(lines)} lines"
            return
        if not anchor.symbol:
            anchor.verdict = "ok"
            return
        base = symbol_base(anchor.symbol)
        pattern = r"\b" + re.escape(base) + r"\b"
        # Rule E (#339): the citation claims the range holds the symbol, so ask that
        # question first — the window below is looser (±SYMBOL_WINDOW lines from the
        # *start*) and is a note rather than the claim.
        end = anchor.end if anchor.end is not None else anchor.start
        if not re.search(pattern, "\n".join(lines[anchor.start - 1 : end])):
            for number, text in enumerate(lines, start=1):
                if re.search(pattern, text):
                    anchor.range_miss = number
                    break
        lo = max(0, anchor.start - 1 - SYMBOL_WINDOW)
        hi = min(len(lines), anchor.start + SYMBOL_WINDOW)
        window = "\n".join(lines[lo:hi])
        if re.search(pattern, window):
            anchor.verdict = "ok"
            anchor.detail = f"`{anchor.symbol}` within ±{SYMBOL_WINDOW} lines"
        elif re.search(pattern, "\n".join(lines)):
            anchor.verdict = "symbol-far"
            anchor.detail = (
                f"`{anchor.symbol}` is in {target} but outside ±{SYMBOL_WINDOW} lines"
            )
        elif re.search(pattern, self.src_text()):
            anchor.verdict = "symbol-moved"
            anchor.detail = f"`{anchor.symbol}` is not in {target} (it lives elsewhere in src/)"
        else:
            anchor.verdict = "symbol-foreign"
            anchor.detail = f"`{anchor.symbol}` is not a `src/` item (FFI or kernel name?)"


def print_list(report: Report) -> None:
    """`file:line → target` rows, verdict last, for every anchor."""
    for anchor in report.anchors:
        target = anchor.target or anchor.path
        rng = f"{anchor.start}-{anchor.end}" if anchor.end is not None else str(anchor.start)
        symbol = f" `{anchor.symbol}`" if anchor.symbol else ""
        bare = " (bare continuation)" if anchor.continuation else ""
        miss = (
            f" [range-miss: `{anchor.symbol}` is not inside {anchor.shown()}, "
            f"it is at {target}:{anchor.range_miss}]"
            if anchor.range_miss is not None
            else ""
        )
        print(f"{anchor.where()} → {target}:{rng} [{anchor.verdict}]{symbol}{bare}{miss}")


def summarize(report: Report, strict_symbols: bool) -> None:
    total = len(report.anchors)
    frozen = report.count("frozen")
    external = report.count("external")
    ambiguous = report.count("ambiguous")
    checked = total - frozen - external - ambiguous
    notes = report.count("symbol-far") + report.count("symbol-foreign")
    misses = sum(1 for a in report.anchors if a.range_miss is not None)
    continuations = sum(1 for a in report.anchors if a.continuation)
    print(
        f"check_doc_line_anchors: {total} anchors ({continuations} bare continuations, "
        f"{report.unattached} unattached) · {frozen} frozen · {external} external · "
        f"{ambiguous} ambiguous · {checked} checked · {notes} symbol notes · "
        f"{misses} range misses"
    )
    for pattern, reason in report.stale_frozen:
        print(
            f"STALE FREEZE  {pattern}: no file it covers carries an anchor any more — "
            f"remove the entry from FROZEN (reason was: {reason})",
            file=sys.stderr,
        )
    for pattern, _reason in report.unused_frozen:
        print(
            f"UNUSED FREEZE {pattern}: every anchor it covers resolves today; the "
            f"exemption is policy, not a current failure — drop it when the record is "
            f"rewritten",
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        default=None,
        help="repository root (default: the tree this script lives in)",
    )
    parser.add_argument("--list", action="store_true", help="print every anchor's verdict")
    parser.add_argument(
        "--strict-symbols",
        action="store_true",
        help=(
            "promote symbol-far / symbol-foreign notes, and a rule-E range miss "
            "(the cited range does not hold the named symbol), to failures"
        ),
    )
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args(argv[1:])

    if args.selftest:
        return selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    if not (root / "docs").is_dir():
        print(f"check_doc_line_anchors: no docs/ under {root}", file=sys.stderr)
        return 2

    report = Checker(root).run()
    if args.list:
        print_list(report)
    summarize(report, args.strict_symbols)

    violations = report.violations(args.strict_symbols)
    for anchor in violations:
        # A bare rule-E miss has no verdict of its own (the anchor resolves and the
        # window passes), so it is labelled by what it is; an anchor that already
        # violates something keeps its verdict and gains the range-miss detail.
        label = anchor.verdict if anchor.verdict != "ok" else "range-miss"
        miss = (
            f" — `{anchor.symbol}` is not inside {anchor.shown()}; it is at "
            f"{anchor.target}:{anchor.range_miss}"
            if anchor.range_miss is not None
            else ""
        )
        print(
            f"{label.upper():13s} {anchor.where()}: {anchor.shown()} "
            f"→ {anchor.target or '?'} — {anchor.detail}{miss}",
            file=sys.stderr,
        )
    if violations or report.stale_frozen:
        print(
            f"check_doc_line_anchors: {len(violations)} bad anchors"
            + (f", {len(report.stale_frozen)} stale frozen entries" if report.stale_frozen else "")
            + " — anchor to a symbol where the number was only a locator, or fix the number",
            file=sys.stderr,
        )
        return 1
    return 0


# ---------------------------------------------------------------------------
# --selftest:  one pass case and one case per way the checker must bite. Each
# case builds its own tree: a case that stops failing is a regression in the
# checker, and a case that starts failing is a regression in the rule.
# ---------------------------------------------------------------------------

#: The prose of ``src/thing.rs`` is line-numbered by construction: ``target_symbol``
#: sits on line 21, the ``tail_*`` fillers run from line 23, and ``tail_30`` (line 52)
#: is deliberately more than ``SYMBOL_WINDOW`` lines away from it — the same anchor
#: passes with ``target_symbol`` and is only a note with ``tail_30``.
FIXTURE_FILES = {
    "src/thing.rs": "\n".join(
        ["//! thing"]
        + [f"fn filler_{i}() {{}}" for i in range(1, 20)]
        + ["pub fn target_symbol() {}", ""]
        + [f"fn tail_{i}() {{}}" for i in range(1, 40)]
    )
    + "\n",
    "src/other.rs": "pub fn moved_symbol() {}\n",
    "src/nested/tests.rs": "fn alpha() {}\n",
    "src/elsewhere/tests.rs": "fn beta() {}\n",
    # Rule E (#339): the definition sits on line 1 and is *called* on line 33, so
    # the two citations below are 32 lines apart — outside ``SYMBOL_WINDOW`` from
    # each other, which is what separates "the range holds the symbol" (the call
    # site, line 33) from "the window saw it nearby" (the definition, line 1).
    "src/caller.rs": "\n".join(
        ["pub fn target_symbol() -> u32 { 0 }"]
        + [f"// filler {i}" for i in range(1, 31)]
        + ["pub fn caller() -> u32 {", "    target_symbol()", "}"]
    )
    + "\n",
    "docs/ok.md": "The function `target_symbol` (`src/thing.rs:21`) is here.\n",
    "docs/external.md": "See `ggml-cuda.cu:12` and `mmq.cuh:9`.\n",
    "docs/ambiguous.md": "Both are named `tests.rs:1`.\n",
    "docs/far.md": "Nothing near `tail_30` (`src/thing.rs:21`).\n",
}

#: The anchor line every case reuses, so only the symbol or the number changes.
GOOD_ANCHOR = "`src/thing.rs:21`"


def _run_case(extra: dict[str, str], frozen: dict[str, str] | None = None) -> Report:
    """Check a fresh tree of ``FIXTURE_FILES`` + ``extra`` and return its report."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, text in {**FIXTURE_FILES, **extra}.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        return Checker(root, frozen or {}).run()


def _run_main(
    extra: dict[str, str], *flags: str, frozen: dict[str, str] | None = None
) -> tuple[int, str]:
    """Run ``main`` over a fresh fixture tree; return its exit code and stderr."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for rel, text in {**FIXTURE_FILES, **extra}.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        saved = globals()["FROZEN"]
        globals()["FROZEN"] = frozen or {}
        err = io.StringIO()
        try:
            # The cases assert exit codes and messages, not chatter.
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                code = main(["check_doc_line_anchors.py", "--root", str(root), *flags])
        finally:
            globals()["FROZEN"] = saved
        return code, err.getvalue()


def _exit_code(extra: dict[str, str], *flags: str, frozen: dict[str, str] | None = None) -> int:
    return _run_main(extra, *flags, frozen=frozen)[0]


def _stderr_of(extra: dict[str, str], *flags: str, frozen: dict[str, str] | None = None) -> str:
    return _run_main(extra, *flags, frozen=frozen)[1]


def _verdicts(report: Report, doc: str) -> list[str]:
    return [a.verdict for a in report.anchors if a.doc == doc]


def _range_miss(report: Report, doc: str) -> int | None:
    """The single anchor's rule-E finding on ``doc``, if it carries one."""
    rows = [a for a in report.anchors if a.doc == doc]
    return rows[0].range_miss if rows else None


def selftest() -> int:
    failures: list[str] = []

    def expect(condition: bool, message: str) -> None:
        if not condition:
            failures.append(message)

    # 1. The clean case, and the three ways an anchor is *not* judged.
    report = _run_case({})
    expect(_verdicts(report, "docs/ok.md") == ["ok"], "a real line with a near symbol must pass")
    expect(
        _verdicts(report, "docs/external.md") == ["external", "external"],
        "another tree's citations must be reported external, not checked",
    )
    expect(
        _verdicts(report, "docs/ambiguous.md") == ["ambiguous"],
        "an ambiguous basename must be reported, not guessed",
    )
    expect(
        _verdicts(report, "docs/far.md") == ["symbol-far"],
        "a symbol in the file but outside the window must be a note",
    )
    expect(not report.violations(False), "no fixture anchor is a violation by default")

    # 2. The synthetic stale anchor: out of range, both ends of a range, reversed.
    report = _run_case({"docs/bad.md": f"See `src/thing.rs:999`.\n"})
    expect(_verdicts(report, "docs/bad.md") == ["out-of-range"], "an out-of-range line must fail")
    report = _run_case({"docs/bad.md": "See `src/thing.rs:10-999`.\n"})
    expect(
        _verdicts(report, "docs/bad.md") == ["out-of-range"],
        "an out-of-range range end must fail",
    )
    report = _run_case({"docs/bad.md": "See `src/thing.rs:30-3`.\n"})
    expect(_verdicts(report, "docs/bad.md") == ["range-order"], "a reversed range must fail")

    # 3. A path that claims to be one of this tree's files and is not.
    report = _run_case({"docs/bad.md": "See `src/gone.rs:3`.\n"})
    expect(
        _verdicts(report, "docs/bad.md") == ["missing"],
        "a tree-shaped path that names no file must fail",
    )

    # 4. The named symbol left the file the anchor points at.
    report = _run_case({"docs/bad.md": f"Nothing here names `moved_symbol` ({GOOD_ANCHOR}).\n"})
    expect(
        _verdicts(report, "docs/bad.md") == ["symbol-moved"],
        "a symbol that moved out of the file must fail",
    )
    report = _run_case({"docs/note.md": f"`cudaFuncSetAttribute` ({GOOD_ANCHOR}).\n"})
    expect(
        _verdicts(report, "docs/note.md") == ["symbol-foreign"],
        "a symbol that is no src/ item at all must be a note",
    )

    # 5. The window is what separates a pass from a note: one anchor line, two symbols.
    report = _run_case({"docs/win.md": f"`target_symbol` ({GOOD_ANCHOR}).\n"})
    expect(_verdicts(report, "docs/win.md") == ["ok"], "a symbol inside the window must pass")
    report = _run_case({"docs/win.md": f"`tail_30` ({GOOD_ANCHOR}).\n"})
    expect(
        _verdicts(report, "docs/win.md") == ["symbol-far"],
        "the same anchor with a distant symbol must flip ok -> symbol-far",
    )

    # 6. Freezing: a record's bad anchor is skipped, but an exemption must be *needed*.
    frozen = {"docs/frozen/*.md": "a record of a past revision"}
    report = _run_case({"docs/frozen/old.md": "See `src/thing.rs:999`.\n"}, frozen)
    expect(_verdicts(report, "docs/frozen/old.md") == ["frozen"], "a frozen anchor must be skipped")
    expect(not report.stale_frozen, "a frozen entry that still covers a violation is not stale")
    expect(not report.violations(False), "a frozen violation must not fail the run")
    report = _run_case({"docs/frozen/clean.md": f"`target_symbol` ({GOOD_ANCHOR}).\n"}, frozen)
    expect(bool(report.unused_frozen), "a frozen entry whose anchors all resolve must be reported")
    expect(not report.stale_frozen, "an entry that still covers files is not stale")
    report = _run_case({}, {"docs/absent/*.md": "a record that no longer exists"})
    expect(bool(report.stale_frozen), "a frozen entry that covers no file must fail as stale")
    expect(
        _exit_code({}, frozen={"docs/absent/*.md": "gone"}) == 1,
        "a stale frozen entry must fail the run",
    )

    # 7. The bare `:NNN` continuation (#355). It attaches to the nearest preceding
    #    `path.ext:NNN` on the same line and is range-checked against that file.
    report = _run_case({"docs/cont.md": f"See {GOOD_ANCHOR}, tail `:30`.\n"})
    expect(
        _verdicts(report, "docs/cont.md") == ["ok", "ok"],
        "a continuation in range passes against the file its anchor names",
    )
    report = _run_case({"docs/cont.md": f"See {GOOD_ANCHOR}, tail `:999`.\n"})
    expect(
        _verdicts(report, "docs/cont.md") == ["ok", "out-of-range"],
        "an out-of-range continuation must fail",
    )
    cont = [a for a in report.anchors if a.doc == "docs/cont.md" and a.continuation]
    expect(len(cont) == 1, "the bare span is reported as one continuation")
    expect(cont and cont[0].shown() == ":999", "the continuation reports its bare written form")
    expect(
        cont and cont[0].target == "src/thing.rs",
        "the continuation is judged against the preceding anchor's resolved file",
    )
    expect(
        _exit_code({"docs/cont.md": f"See {GOOD_ANCHOR}, tail `:999`.\n"}) == 1,
        "a bad continuation fails the run",
    )

    # 7b. The window is the doc line, and it is one-directional: the nearest anchor
    #     *before* the span on the same line, never one on the next line and never one
    #     after it (that anchor names another file).
    report = _run_case(
        {"docs/edge.md": f"{GOOD_ANCHOR} on one line.\n\nA bare `:999` on the next line.\n"}
    )
    expect(
        _verdicts(report, "docs/edge.md") == ["ok"],
        "a continuation on the next line is outside the window and stays silent",
    )
    expect(report.unattached == 1, "the silent span is counted, so the boundary is visible")
    report = _run_case({"docs/edge.md": f"A bare `:999` precedes {GOOD_ANCHOR} on its line.\n"})
    expect(
        _verdicts(report, "docs/edge.md") == ["ok"],
        "a continuation before every anchor on its line stays silent",
    )
    expect(report.unattached == 1, "the preceding-only count is reported")
    report = _run_case({"docs/edge.md": "A bare `:999` stands alone.\n"})
    expect(_verdicts(report, "docs/edge.md") == [], "a line with no anchor carries no anchor")
    expect(report.unattached == 1, "a line with no anchor still counts its silent span")

    # 7c. Two path candidates on one line: each continuation follows the *nearest*
    #     preceding anchor, so the second one is judged against two-line `other.rs`
    #     (out of range) and not against the 61-line `thing.rs` (which would pass).
    report = _run_case(
        {"docs/two.md": "`src/thing.rs:21` then `:30`; `src/other.rs:1` then `:2`.\n"}
    )
    expect(
        _verdicts(report, "docs/two.md") == ["ok", "ok", "ok", "out-of-range"],
        "each continuation attaches to its own nearest preceding anchor",
    )

    # 7d. The frozen-plan case: a per-ticket record keeps its text, so its bare
    #     continuations are exempt by the same lookup as its path anchors — and they
    #     count as violations for the ratchet, so the exemption is not reported unused.
    plan = {"docs/ARCHITECTURE-EXECUTION-PLAN.md": "per-ticket history of a past revision"}
    report = _run_case(
        {"docs/ARCHITECTURE-EXECUTION-PLAN.md": f"See {GOOD_ANCHOR}, `:999`.\n"}, plan
    )
    expect(
        _verdicts(report, "docs/ARCHITECTURE-EXECUTION-PLAN.md") == ["frozen", "frozen"],
        "a continuation in a frozen record is exempt like its path anchor",
    )
    expect(not report.violations(False), "a frozen continuation must not fail the run")
    expect(not report.stale_frozen, "the entry still covers anchors, so it is not stale")
    expect(not report.unused_frozen, "the frozen entry covers a real continuation violation")
    expect(
        _exit_code(
            {"docs/ARCHITECTURE-EXECUTION-PLAN.md": f"See {GOOD_ANCHOR}, `:999`.\n"},
            frozen=plan,
        )
        == 0,
        "the real frozen pattern makes a bad continuation pass",
    )

    # 8. Rule E (#339): the cited range must hold the symbol the sentence names.
    #    The window test is ±25 lines around the *start*, so an anchor can pass it
    #    and still cite a range 660 lines away from the definition — which is what
    #    the three walkthrough anchors did. The call-site case is the control: a
    #    citation of a *call* is legitimate and must not be reported.
    call_site = {"docs/range.md": "`target_symbol` (`src/caller.rs:33`).\n"}
    report = _run_case(call_site)
    expect(
        _verdicts(report, "docs/range.md") == ["ok"],
        "a range that holds a *call* of the symbol passes (rule E control case)",
    )
    expect(
        _range_miss(report, "docs/range.md") is None,
        "the call-site control case must not be reported as a range miss",
    )
    offence = {"docs/range.md": "`target_symbol` (`src/caller.rs:20`).\n"}
    report = _run_case(offence)
    expect(
        _verdicts(report, "docs/range.md") == ["ok"],
        "the window test alone still calls a symbol 19 lines away `ok`...",
    )
    expect(
        _range_miss(report, "docs/range.md") == 1,
        "... but rule E reports that line 20 does not hold it (the definition is line 1)",
    )
    expect(not report.violations(False), "a range miss is a note, not a default failure")
    expect(bool(report.violations(True)), "--strict-symbols promotes a range miss to a failure")
    expect(
        _exit_code(offence) == 0,
        "the plain run stays green over a range miss (the population is 79 on master)",
    )
    expect(
        _exit_code(offence, "--strict-symbols") == 1,
        "a range miss fails the run under --strict-symbols",
    )
    stderr = _stderr_of(offence, "--strict-symbols")
    expect("docs/range.md:1" in stderr, "the failure names the doc line")
    expect(
        "src/caller.rs:1" in stderr,
        "the failure names the target file and where the symbol actually is",
    )
    # The `()` half: `metal_available()` is a symbol, so the anchor is checked at
    # all — without the strip it is `ok` with no symbol note (the #339 row :526).
    report = _run_case({"docs/range.md": "`target_symbol()` (`src/caller.rs:20`).\n"})
    expect(
        _range_miss(report, "docs/range.md") == 1,
        "a trailing `()` is stripped before the symbol test, so the token is checked",
    )
    expect(
        not [
            a
            for a in _run_case(
                {"docs/range.md": "`target_symbol()` (`src/caller.rs:33`).\n"}
            ).anchors
            if a.doc == "docs/range.md" and a.is_violation(True)
        ],
        "the `()` form passes when its range holds the call",
    )
    # A single-line anchor is a range of one: the identifier must be on that line.
    report = _run_case({"docs/range.md": "`target_symbol` (`src/thing.rs:23`).\n"})
    expect(
        _range_miss(report, "docs/range.md") == 21,
        "a one-line anchor whose line does not hold the symbol is a range miss",
    )

    # 9. The exit codes, end to end.
    expect(
        _exit_code({"docs/bad.md": "See `src/thing.rs:999`.\n"}) == 1,
        "exit 1 when an anchor is bad",
    )
    expect(_exit_code({}) == 0, "exit 0 when every anchor resolves")
    expect(_exit_code({}, "--strict-symbols") == 1, "--strict-symbols promotes a note to a failure")
    expect(_exit_code({}, "--list") == 0, "--list changes no verdict")

    if failures:
        for failure in failures:
            print(f"SELFTEST FAIL  {failure}", file=sys.stderr)
        print(
            f"check_doc_line_anchors --selftest: {len(failures)} case(s) failed", file=sys.stderr
        )
        return 1
    print("check_doc_line_anchors --selftest: every case passes (the checker bites)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
