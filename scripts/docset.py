"""Which markdown a documentation gate covers, and the reasons some of it is out of scope.

The doc gates used to take their file set from a hard-coded glob — `docs/**` plus
`AGENTS.md` — which is how four anchors citing the *deleted* `src/models/qwen2/forward.rs`
survived in `experiments/cuda/CUDA_GRAPH_PROGRESS.md`, one directory above the gate
([#522](https://github.com/yusiwen/minfer/issues/522)): documentation outside that glob
was never read, not even to say "out of scope".

So the set is now discoverable: **every** `*.md` in the tree, minus an explicit ledger
(`scripts/markdown-scope.toml`) whose entries each carry a reason, per
[ADR-0023](https://github.com/yusiwen/minfer/blob/master/docs/adr/0023-gates-earn-their-keep.md)
— reasoned exclusions beside their checker, and a ratchet in both directions:

- a file whose path matches no ledger pattern is **scanned**;
- a ledger pattern that matches no file is a **problem** (the ledger may not rot);
- when the tree is a git checkout, the walked total must equal `git ls-files '*.md'`'s
  count, so neither the walk's ignore list nor a ledger entry can silently swallow a
  tracked document.

The walk prunes the build and scratch trees (`book/` is the mdBook output, `target/` the
Rust build dir) — the git comparison is what proves the prune list is not hiding
anything. A tree without git (the base revision `check_anchor_drift.py` extracts with
`git archive`) simply skips that comparison.

Used by `check_doc_line_anchors.py` (and through it `check_anchor_drift.py`) and
`check_docs_links.py`, so one file set answers "which markdown do the doc gates own".
"""

from __future__ import annotations

import fnmatch
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

LEDGER = "scripts/markdown-scope.toml"
#: Directories no gate reads: build output, scratch trees and VCS metadata. Pruned, and
#: then checked against `git ls-files` — see the module docstring.
IGNORED_DIRS = frozenset(
    {
        ".git",
        ".worktrees",
        "target",
        "book",
        "node_modules",
        ".venv",
        ".cache",
        "dist",
        ".ccc_config",  # the ccc semantic-search cache (a HuggingFace snapshot tree)
    }
)


class ScopeError(Exception):
    """The scope itself is broken (an unreadable ledger, or a pattern matching nothing)."""


@dataclass
class Scope:
    """The files a gate reads, the ones a ledger entry excuses, and the totals."""

    scanned: list[Path] = field(default_factory=list)
    excluded: list[tuple[str, str]] = field(default_factory=list)
    total: int = 0
    tracked: int | None = None  # `git ls-files '*.md' | wc -l`, when git can answer
    missing: list[str] = field(default_factory=list)  # tracked, but the walk did not find it
    untracked: int = 0  # walked, but not tracked yet (a scratch note is still scanned)


def walk_markdown(root: Path) -> list[Path]:
    """Every `*.md` under `root`, in a stable order, with the build trees pruned."""
    found: list[Path] = []
    for path in sorted(root.rglob("*.md")):
        if IGNORED_DIRS.intersection(path.relative_to(root).parts[:-1]):
            continue
        found.append(path)
    return found


def tracked_markdown(root: Path) -> set[str] | None:
    """`git ls-files '*.md'` for `root`, or `None` when git cannot answer."""
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "*.md"],
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {name for name in out.decode("utf-8", "replace").split("\0") if name}


def load_ledger(path: Path) -> dict[str, str]:
    """`{pattern: reason}` from the ledger's `[exclude]` table."""
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: dict[str, str] = {}
    for pattern, reason in data.get("exclude", {}).items():
        if not reason or not str(reason).strip():
            raise ScopeError(f"{path}: `{pattern}` needs a non-empty reason")
        out[pattern] = str(reason)
    return out


def select(root: Path, ledger_path: Path | None = None) -> Scope:
    """Partition `root`'s markdown into scanned and excluded; raise `ScopeError` on a broken ledger."""
    root = Path(root)
    ledger = load_ledger(ledger_path if ledger_path is not None else root / LEDGER)
    scope = Scope()
    used: set[str] = set()
    for path in walk_markdown(root):
        rel = path.relative_to(root).as_posix()
        pattern = next((p for p in sorted(ledger) if fnmatch.fnmatch(rel, p)), None)
        if pattern is None:
            scope.scanned.append(path)
        else:
            used.add(pattern)
            scope.excluded.append((rel, ledger[pattern]))
    scope.total = len(scope.scanned) + len(scope.excluded)
    stale = [p for p in ledger if p not in used]
    if stale:
        raise ScopeError(
            f"{LEDGER}: {len(stale)} pattern(s) match no markdown file — the ledger may not rot: "
            + ", ".join(f"`{p}`" for p in sorted(stale))
        )
    tracked = tracked_markdown(root)
    if tracked is not None:
        scope.tracked = len(tracked)
        walked = {p.relative_to(root).as_posix() for p in scope.scanned}
        walked |= {rel for rel, _ in scope.excluded}
        scope.missing = sorted(tracked - walked)
        scope.untracked = len(walked - tracked)
    return scope


def summary(scope: Scope) -> str:
    """The one-line census a checker prints: scanned + excluded = total (vs. tracked)."""
    line = (
        f"markdown scope: {len(scope.scanned)} scanned + {len(scope.excluded)} excluded "
        f"= {scope.total} file(s)"
    )
    if scope.tracked is None:
        return line + " (no git checkout; the tracked count is not compared)"
    if scope.missing:
        raise ScopeError(
            f"{len(scope.missing)} tracked markdown file(s) are neither scanned nor excluded — "
            "a pruned directory is hiding them: " + ", ".join(scope.missing[:5])
        )
    if scope.untracked:
        return line + f"; every one of git's {scope.tracked} tracked `*.md` is covered (+{scope.untracked} untracked)"
    return line + f", which is exactly git's {scope.tracked} tracked `*.md`"


def scanned(root: Path) -> list[Path]:
    """The files a gate reads for `root`, silently — the discovery path.

    Printing belongs to the checkers' entry points (`announce`), not here: the selftests
    run the checkers in-process and assert that a clean run prints nothing.
    """
    return select(root).scanned


def announce(root: Path) -> Scope:
    """`select` for a checker's `main`, printing the census and raising on a broken scope."""
    scope = select(root)
    print(summary(scope))
    return scope


def exclusions(root: Path) -> dict[str, str]:
    """`{repo-relative path: reason}` for the documents this root's ledger excuses.

    `check_anchor_drift.py` needs it to tell "the document left the gates" from "the
    document's anchors were deleted": without this, a ledger entry added by the very
    commit under review would read as seven removed anchors.
    """
    return {rel: reason for rel, reason in select(root).excluded}


def selftest() -> int:
    """The scope layer's own cases: discovery, the ledger, and both ratchets."""
    cases: list[tuple[str, bool]] = []

    def build(tmp: str, files: dict[str, str], ledger: str = "") -> Path:
        root = Path(tmp)
        for rel, text in files.items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        if ledger:
            (root / LEDGER).parent.mkdir(parents=True, exist_ok=True)
            (root / LEDGER).write_text(ledger, encoding="utf-8")
        return root

    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a", "notes.md": "n", "experiments/x.md": "x"})
        scope = select(root)
        cases.append(("every markdown is scanned by default", len(scope.scanned) == 3))
        cases.append(("an unknown tree is not excluded", not scope.excluded))

    with tempfile.TemporaryDirectory() as tmp:
        root = build(
            tmp,
            {"docs/a.md": "a", "experiments/old/x.md": "x"},
            '[exclude]\n"experiments/**" = "a record"\n',
        )
        scope = select(root)
        cases.append(
            ("a ledger pattern excludes what it matches", [r for r, _ in scope.excluded] == ["experiments/old/x.md"])
        )
        cases.append(("`**` crosses a directory separator", len(scope.scanned) == 1))

    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a"}, '[exclude]\n"viz/none.md" = "stale"\n')
        try:
            select(root)
            cases.append(("a pattern matching nothing fails", False))
        except ScopeError:
            cases.append(("a pattern matching nothing fails", True))
    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a"}, '[exclude]\n"docs/a.md" = "  "\n')
        try:
            select(root)
            cases.append(("an empty reason fails", False))
        except ScopeError:
            cases.append(("an empty reason fails", True))

    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a", "target/gen.md": "g", "book/out.md": "b", ".ccc_config/c.md": "c"})
        cases.append(("the build and cache trees are pruned", len(select(root).scanned) == 1))

    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a"})
        cases.append(("without git the tracked count is not compared", select(root).tracked is None))

    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a", "target/hidden.md": "h"})
        subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
        # `-f`: the host's global gitignore likely lists `target/`, and the point of the
        # case is a *tracked* file inside a pruned directory.
        subprocess.run(["git", "-C", tmp, "add", "-f", "docs/a.md", "target/hidden.md"], check=True)
        scope = select(root)
        cases.append(
            ("a pruned directory may not hide a tracked file", scope.missing == ["target/hidden.md"])
        )
        try:
            summary(scope)
            cases.append(("and the summary raises for it", False))
        except ScopeError:
            cases.append(("and the summary raises for it", True))

    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a"})
        subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
        subprocess.run(["git", "-C", tmp, "add", "docs/a.md"], check=True)
        cases.append(
            (
                "a clean tree agrees with git and says so",
                "exactly git's 1 tracked" in summary(select(root)),
            )
        )
    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, {"docs/a.md": "a", "scratch.md": "s"})
        subprocess.run(["git", "-C", tmp, "init", "-q"], check=True)
        subprocess.run(["git", "-C", tmp, "add", "docs/a.md"], check=True)
        cases.append(
            (
                "an untracked scratch note is scanned, not a failure",
                "+1 untracked" in summary(select(root)),
            )
        )

    failed = sum(1 for _, ok in cases if not ok)
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"docset --selftest: {len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent)
    parser.add_argument("--list", action="store_true", help="print the scanned and excluded files")
    parser.add_argument("--selftest", action="store_true", help="run the scope layer's own cases")
    args = parser.parse_args(argv[1:])
    if args.selftest:
        return selftest()
    try:
        scope = select(args.root.resolve())
        print(summary(scope))
    except ScopeError as error:
        print(f"docset: {error}", file=sys.stderr)
        return 1
    if args.list:
        for path in scope.scanned:
            print(f"SCANNED   {path.relative_to(args.root.resolve()).as_posix()}")
        for rel, reason in scope.excluded:
            print(f"EXCLUDED  {rel} — {reason}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
