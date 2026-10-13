#!/usr/bin/env python3
"""Classify a change set so `ci.yml`'s code jobs can skip a docs-only change.

Every `ci.yml` run used to compile the crate for three jobs whatever changed. The
measurement behind this classifier (13 completed runs, 2026-10-09): `build-linux-cuda`
median **12.3 min**, `build-macos` 1.9 min, `check-docs` 1.3 min, `test-linux-cpu`
1.1 min, `check-viz` / `lint-workflows` / `check-pr-body` about 0.1 min each — while
104 of the last 200 non-merge commits changed documentation only, and 8 of the last
15 runs were on `docs/*` branches.

**Why a job-level `if:` and not `paths:`/`paths-ignore:`.** A workflow skipped by
path filtering leaves its required checks in `Pending` forever, so the PR cannot
merge. A *skipped job* reports `Success` and satisfies a required check. The
workflow therefore classifies the diff once, in its `changes` job, and every code
job carries a job-level guard.

Classes and the one rule that makes them safe:

- `docs` — `docs/**`, `README.md`, `AGENTS.md`, `LICENSE`. Read by the doc gates,
  which run unconditionally (`check-docs`), so a change confined to these needs no
  other job.
- `plan` — `scripts/status.toml`, read by `check_status.py` in `check-docs` and by
  nothing else.
- `baselines` — `scripts/test-baselines.toml`, read by `check_baselines.py --check`
  in `check-docs` **and** by its `--check-live` in `test-linux-cpu`, so that one job
  must still run.
- `rust` — everything else, including every path this file has not been taught.
**The unknown path is `rust`.** A path this classifier does not recognise is a
reason to run the jobs, never to skip them — and so is a range it cannot read (an
empty `--base`, the all-zero `before` git reports for a first push, a revision that
does not resolve, or an empty diff, which is not evidence that nothing changed).

Modes::

    python3 scripts/classify_changes.py --base origin/master --head HEAD
    python3 scripts/classify_changes.py --selftest

Stdout is only the GitHub-output lines (`rust=…`, `baselines=…`), so the workflow can
`>> "$GITHUB_OUTPUT"` it; the human summary, and the reason for a fail-safe, go to
stderr.

Pure and offline apart from the one `git diff` the mode needs; stdlib only.

Exit codes: 0 the classification was printed, 2 usage.

"""

from __future__ import annotations

import argparse
import contextlib
import io
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

#: Paths only the documentation gates read.
DOCS_PREFIXES = ("docs/",)
DOCS_FILES = ("README.md", "AGENTS.md", "LICENSE")

#: The `docs/` paths a **code** job reads, so the prefix above must not claim them.
#: Empty, and that is the point: it is the list the audit in the module docstring
#: produces, and ADR-0024's remedy (move the record beside its readers) is what
#: keeps it empty. #459 moved the F6 fixture manifest to `tests/fixtures/` and #461
#: moved the dead-code baseline to `scripts/`, each deleting its entry here.
RUST_DOC_FILES: tuple[str, ...] = ()

#: The plan's ledger: read by `check_status.py` inside `check-docs`, by nothing else.
PLAN_LEDGER = "scripts/status.toml"

#: The suite ledger: read by `check_baselines.py --check` inside `check-docs` *and*
#: by its `--check-live` inside `test-linux-cpu`, which must therefore still run.
BASELINES_LEDGER = "scripts/test-baselines.toml"

#: The all-zero SHA git reports as `before` for a branch's first push.
ZERO_SHA = re.compile(r"\A0+\Z")

#: The honest limit `--help` must state (also asserted by `--selftest`).
HELP_LIMIT = (
    "What it proves: which paths a range changed, and how they classify. What it cannot "
    "prove: that a path it has never been taught is safe to skip — so an unclassified "
    "path, an empty range and an unreadable one all count as `rust`, and the jobs run."
)

#: The counts `classify` returns — the four *classes of path* the job guards read.
CLASSES = ("docs", "plan", "baselines", "rust")

#: The bindings: which ADR governs which code surface (#541). Validated by
#: `scripts/check_adr.py` (which owns the ledger's ratchets); read here so the PR-time
#: question can **name the decision** instead of asking "is this an ADR matter?".
#:
#: A binding is a `path` (a file) or a `symbol` (everywhere the token occurs under
#: `src/`, so it survives a file move). Both kinds make the same claim — "this surface's
#: shape is fixed by that ADR" — and the reader is deliberately permissive: a malformed
#: ledger yields no bindings rather than an exception, because this script's contract is
#: to classify, and `check-docs` is where a broken ledger fails.
GOVERNS_LEDGER = "scripts/adr-governs.toml"

#: A **new** design document is a new contract, so it is a decision surface too. Only an
#: *added* one: editing a design doc is how the contract is kept true, which is routine.
DESIGN_DOC = re.compile(r"\Adocs/[A-Z0-9][A-Za-z0-9-]*-DESIGN\.md\Z")


def classify(paths: list[str]) -> dict[str, int]:
    """`{class: count}` for `paths`; anything unrecognised is `rust`."""
    counts = {name: 0 for name in CLASSES}
    for raw in paths:
        path = raw.strip()
        if not path:
            continue
        while path.startswith("./"):
            path = path[2:]
        if path == BASELINES_LEDGER:
            counts["baselines"] += 1
        elif path == PLAN_LEDGER:
            counts["plan"] += 1
        elif path in RUST_DOC_FILES:
            counts["rust"] += 1
        elif path in DOCS_FILES or path.startswith(DOCS_PREFIXES):
            counts["docs"] += 1
        else:
            counts["rust"] += 1
    return counts


def load_governs(root: Path) -> list[dict[str, str]]:
    """The ADR↔surface bindings, read permissively — `check_adr.py` validates the file."""
    path = root / GOVERNS_LEDGER
    if not path.is_file():
        return []
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return []
    entries = data.get("governs", [])
    if not isinstance(entries, list):
        return []
    return [e for e in entries if isinstance(e, dict)]


#: A `symbol` binding names an identifier's **definition sites**, not the places the
#: name is mentioned (#541): a knob or an FFI constant that this tree does not define is
#: bound by the file that implements the mechanism instead. `check_adr.py` enforces the
#: same rule when it validates the ledger.
SYMBOL_DEFINES = (
    r"^\s*(?:pub(?:\([a-z]+\))?\s+)?(?:unsafe\s+)?(?:async\s+)?"
    r"(?:fn|struct|enum|trait|const|static|type|impl|mod)\s+{symbol}\b"
)


def symbol_files(root: Path, symbol: str) -> set[str]:
    """Every `src/**/*.rs` that **defines** `symbol`, repository-relative."""
    pattern = re.compile(SYMBOL_DEFINES.format(symbol=re.escape(symbol)), re.MULTILINE)
    return {
        p.relative_to(root).as_posix()
        for p in (root / "src").rglob("*.rs")
        if pattern.search(p.read_text(encoding="utf-8", errors="replace"))
    }


def governed(root: Path, paths: list[str]) -> list[tuple[str, str, str]]:
    """`[(adr, shown surface, what)]` for the bindings this range touches, in ledger order.

    A `path` binding matches the changed file itself; a `symbol` binding matches any file
    that holds the symbol *at head*, which is what lets the binding survive a move.
    """
    changed = {p.strip().lstrip("./") for p in paths}
    out: list[tuple[str, str, str]] = []
    cache: dict[str, set[str]] = {}
    for entry in load_governs(root):
        adr, what = str(entry.get("adr", "")), str(entry.get("what", ""))
        target = entry.get("path")
        if target:
            if str(target) in changed:
                out.append((adr, str(target), what))
        elif entry.get("symbol"):
            symbol = str(entry["symbol"])
            holders = cache.setdefault(symbol, symbol_files(root, symbol))
            hit = sorted(holders & changed)
            if hit:
                out.append((adr, f"`{symbol}` (in {hit[0]})", what))
    return out


def decide(
    counts: dict[str, int], added: tuple[str, ...] = (), touched: int = 0
) -> dict[str, bool]:
    """The three predicates: the Rust jobs, the suite job's reason, and the ADR question."""
    return {
        "rust": counts["rust"] > 0,
        "baselines": counts["baselines"] > 0,
        "decision_surface": touched > 0 or any(DESIGN_DOC.match(p) for p in added),
    }


def decision_reasons(touched: list[tuple[str, str, str]], added: tuple[str, ...] = ()) -> list[str]:
    """The decisions a range has to answer for, one line each — the classifier's evidence."""
    out = [f"ADR-{adr} governs {shown} — {what}" for adr, shown, what in touched]
    out += [f"{path} (new) — a new design contract" for path in sorted(added) if DESIGN_DOC.match(path)]
    return out


def git_paths(root: Path, base: str, head: str) -> tuple[list[str] | None, str]:
    """`([paths], "")` or `(None, reason)` — never an exception, never a guess.

    A three-dot range is the primitive that stays correct on both event kinds: on a
    `pull_request` it is the merge base against `origin/<base_ref>`, and on a push it
    degenerates to the pushed commits. `--no-renames` keeps a rename's old path in the
    list instead of letting git collapse it away.
    """
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=ACDMRTUXB",
            f"{base}...{head}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip().splitlines()
        last = detail[-1] if detail else f"exit {proc.returncode}"
        return None, f"git diff {base}...{head} failed ({last})"
    return [line for line in proc.stdout.splitlines() if line.strip()], ""


def git_added(root: Path, base: str, head: str) -> list[str]:
    """The paths the range **added** — the one status the decision question needs.

    A *new* design document is a new contract; editing one is not. Read separately from
    ``git_paths`` so the classifier's main call stays a `--name-only` list.
    """
    proc = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--name-only",
            "--no-renames",
            "--diff-filter=A",
            f"{base}...{head}",
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return []
    return [line for line in proc.stdout.splitlines() if line.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="classify_changes.py",
        description=(
            "Classify the changed paths of a range into docs / plan / baselines / rust, "
            "and print the two job predicates `ci.yml`'s code jobs need. " + HELP_LIMIT
        ),
        epilog=(
            "The workflow runs this in its `changes` job with --base origin/<base_ref> "
            "(pull_request) or the push's `before` SHA, and each code job carries a "
            "job-level `if:` — never a workflow-level `paths:` filter, which would leave "
            "the required checks Pending."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base", metavar="REV", help="the range's base revision")
    parser.add_argument(
        "--head", metavar="REV", default="HEAD", help="the range's head (default HEAD)"
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the built-in pass/fail cases and exit non-zero on a failure",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="the repository root (default: the repo this script lives in)",
    )
    return parser


def emit(decision: dict[str, bool]) -> None:
    """The GitHub-output lines, and only those, on stdout."""
    print(f"rust={'true' if decision['rust'] else 'false'}")
    print(f"baselines={'true' if decision['baselines'] else 'false'}")
    print(f"decision_surface={'true' if decision['decision_surface'] else 'false'}")


# --------------------------------------------------------------------------
# --selftest
# --------------------------------------------------------------------------

#: `(name, paths, rust, baselines)` for `classify` + `decide`; the pair of booleans
#: is what the workflow's job guards read.
#:
#: The two ledger paths are written as **literals**, never as `PLAN_LEDGER` /
#: `BASELINES_LEDGER`. A case built from the constant moves with it — including the
#: `./`-normalisation one — so a misspelled constant passes every case while the real
#: file silently changes class (#466). Written out here, renaming a ledger fails a
#: named case until this table is updated deliberately.
#: `(name, changed paths, added paths, expected ADR or None)` for the governance binding.
#: Driven through `governed()` against the **real** ledger and the **real** tree, because
#: that is the claim under test: "a change here answers to that decision". A rename that
#: breaks a `path` binding or a move that breaks a `symbol` binding fails `check_adr.py`,
#: which owns the ledger's ratchets; these cases pin the *matching* semantics.
GOVERNANCE_CASES = (
    ("a backend-id change is governed by ADR-0011", ["src/graph/registry.rs"], [], "0011"),
    ("a KV-format change is governed by ADR-0006", ["src/graph/kvformat.rs"], [], "0006"),
    ("a quantizer change is governed by ADR-0020", ["src/quantize.rs"], [], "0020"),
    ("a template change is governed by ADR-0019", ["src/template.rs"], [], "0019"),
    ("a dependency change is governed by ADR-0007", ["Cargo.toml"], [], "0007"),
    ("a KV-session change is governed by ADR-0011 and ADR-0014", ["src/graph/kvsession.rs"], [], "0011"),
    ("a symbol binding matches a file the symbol lives in", ["src/graph/backend.rs"], [], "0009"),
    ("a routine source change is governed by nothing", ["src/main.rs"], [], None),
    ("a documentation change is governed by nothing", ["docs/USAGE.md"], [], None),
    ("a test-file change alone is governed by nothing", ["src/rendering_tests.rs"], [], None),
    ("a *new* design document asks the question", [], ["docs/FOO-DESIGN.md"], None),
)

FIXTURE_CASES = (
    ("a docs chapter alone", ["docs/ARCHITECTURE.md"], False, False),
    ("README.md and AGENTS.md", ["README.md", "AGENTS.md"], False, False),
    ("the LICENSE", ["LICENSE"], False, False),
    (
        "the plan ledger alone is scripts/status.toml",
        ["scripts/status.toml"],
        False,
        False,
    ),
    (
        "the suite ledger alone is scripts/test-baselines.toml",
        ["scripts/test-baselines.toml"],
        False,
        True,
    ),
    (
        "the suite ledger beside its prose",
        ["scripts/test-baselines.toml", "docs/TEST-BASELINES.md"],
        False,
        True,
    ),
    (
        "a checker's own record, beside it (ADR-0024)",
        ["scripts/dead-code-baseline.toml"],
        True,
        False,
    ),
    ("one source file", ["src/graph/alloc.rs"], True, False),
    ("the build configuration", ["Cargo.toml", "Cargo.lock", "build.rs"], True, False),
    ("a checker that is not a ledger", ["scripts/check_status.py"], True, False),
    ("the workflow itself", [".github/workflows/ci.yml"], True, False),
    ("a viz sample", ["viz/samples/qwen2.json"], True, False),
    ("an unknown root file", ["Makefile"], True, False),
    ("a deleted docs file", ["docs/old-chapter.md"], False, False),
    (
        "./-prefixed paths are normalised",
        ["./docs/x.md", "./README.md", "./scripts/status.toml"],
        False,
        False,
    ),
    ("no paths at all", [], False, False),
)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in `cwd` with a hermetic identity and no signing/hooks."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.email=classify@example.invalid",
            "-c",
            "user.name=classify",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def _git_checked(cwd: Path, *args: str) -> str:
    proc = _git(cwd, *args)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def run_selftest() -> int:
    """Run the built-in pass/fail cases and return 0 when every one behaves."""
    results: list[tuple[str, bool, str]] = []

    def record(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    for name, paths, want_rust, want_baselines in FIXTURE_CASES:
        decision = decide(classify(paths))
        record(
            name,
            decision["rust"] == want_rust and decision["baselines"] == want_baselines,
            f"got {decision}, wanted rust={want_rust} baselines={want_baselines}",
        )

    # The ratchet behind the exception list: it is empty because the audit found no
    # `docs/` path a code job reads, and adding one has to be a deliberate edit that
    # changes this case too (with its justification in the module docstring).
    record(
        "no docs/ path is carried as a code-job exception",
        RUST_DOC_FILES == (),
        f"RUST_DOC_FILES={RUST_DOC_FILES!r}",
    )

    # The range itself, end to end, in a hermetic repo: a docs commit, then a source
    # commit, then the three fail-safe ranges.
    with tempfile.TemporaryDirectory(prefix="classify_changes_selftest_") as tmp:
        root = Path(tmp)
        _git_checked(root, "init", "-q")
        _git_checked(root, "commit", "-q", "--allow-empty", "-m", "c1")
        c1 = _git_checked(root, "rev-parse", "HEAD")
        (root / "docs").mkdir()
        (root / "docs" / "chapter.md").write_text("# chapter\n", encoding="utf-8")
        _git_checked(root, "add", "-A")
        _git_checked(root, "commit", "-q", "-m", "c2 docs")
        c2 = _git_checked(root, "rev-parse", "HEAD")
        (root / "src").mkdir()
        (root / "src" / "lib.rs").write_text("pub fn f() {}\n", encoding="utf-8")
        _git_checked(root, "add", "-A")
        _git_checked(root, "commit", "-q", "-m", "c3 source")
        c3 = _git_checked(root, "rev-parse", "HEAD")

        paths, reason = git_paths(root, c1, c2)
        record(
            "a docs commit classifies as docs",
            paths == ["docs/chapter.md"] and decide(classify(paths))["rust"] is False,
            f"paths={paths} reason={reason}",
        )
        paths, reason = git_paths(root, c1, c3)
        record(
            "a range ending in a source commit classifies as rust",
            paths is not None and decide(classify(paths))["rust"] is True,
            f"paths={paths} reason={reason}",
        )
        paths, reason = git_paths(root, "does-not-exist", c3)
        record(
            "an unresolvable revision refuses to classify",
            paths is None and "failed" in reason,
            f"paths={paths} reason={reason}",
        )

        def run_main(*args: str) -> str:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                main(["classify_changes.py", "--root", str(root), *args])
            return out.getvalue()

        record(
            "the all-zero before-SHA is a fail-safe: every code job runs",
            "rust=true" in run_main("--base", "0" * 40, "--head", c3),
            "",
        )
        record(
            "an empty base is a fail-safe: every code job runs",
            "rust=true" in run_main("--base", "", "--head", c3),
            "",
        )
        record(
            "an empty diff is a fail-safe, not evidence that nothing changed",
            "rust=true" in run_main("--base", c3, "--head", c3),
            "",
        )
        record(
            "a docs range prints all three predicates false",
            run_main("--base", c1, "--head", c2)
            == "rust=false\nbaselines=false\ndecision_surface=false\nadr_question=\n",
            run_main("--base", c1, "--head", c2),
        )

    # The governance bindings, one case per shape of match: which ADR the range answers
    # to, and that `decision_surface` follows it. Driven against the real ledger and the
    # real tree — that is the claim under test. The ledger's own ratchets (a path must
    # exist, a symbol must occur, a superseded ADR fails) belong to `check_adr.py`, which
    # owns the file; a broken ledger yields no bindings here rather than an exception.
    root = Path.cwd()
    if (root / "src").is_dir():
        empty = {name: 0 for name in CLASSES}
        for name, paths, added, expected in GOVERNANCE_CASES:
            found = governed(root, list(paths))
            adrs = sorted({adr for adr, _shown, _what in found})
            surface = decide(empty, tuple(added), len(found))["decision_surface"]
            want = bool(expected) or bool(added)
            ok = surface == want and (expected in adrs if expected else not adrs)
            record(
                name,
                ok,
                f"expected {expected or 'no binding'} (decision_surface={want}), "
                f"got {adrs or 'none'} (decision_surface={surface})",
            )
    else:
        print("SKIP  the governance cases (not run from a checkout)")

    # `--help` states the honest limit.
    record("--help states the honest limit", HELP_LIMIT in build_parser().description, "")

    failures = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1
            print(f"      got: {detail or 'no detail'}")
    total = len(results)
    print(f"classify_changes selftest: {total - failures}/{total} cases pass")
    return 1 if failures else 0


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    if args.selftest:
        if args.base or args.head != "HEAD":
            parser.error("--selftest does not combine with --base/--head")
        return run_selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    base = (args.base or "").strip()
    head = (args.head or "").strip()

    paths: list[str] | None = None
    reason = ""
    if not base:
        reason = "the base revision is empty"
    elif ZERO_SHA.match(base):
        reason = f"the base is the all-zero SHA ({base}) — a new branch or a first push"
    elif not head:
        reason = "the head revision is empty"
    else:
        paths, reason = git_paths(root, base, head)
        if paths is not None and not paths:
            paths, reason = None, f"the range {base}...{head} is empty"

    if paths is None:
        print(f"classify_changes: {reason} — running every code job", file=sys.stderr)
        # The fail-safe asks the ADR question too: a range nobody could read is not
        # evidence that no decision was taken.
        emit({"rust": True, "baselines": False, "decision_surface": True})
        print("adr_question=")
        return 0

    counts = classify(paths)
    added = tuple(git_added(root, base, head))
    touched = governed(root, paths)
    decision = decide(counts, added, len(touched))
    reasons = decision_reasons(touched, added)
    summary = ", ".join(f"{name}={counts[name]}" for name in CLASSES)
    print(
        f"classify_changes: {len(paths)} changed path(s) over {base}...{head} ({summary}) "
        f"→ rust={'true' if decision['rust'] else 'false'} "
        f"baselines={'true' if decision['baselines'] else 'false'} "
        f"decision_surface={'true' if decision['decision_surface'] else 'false'}",
        file=sys.stderr,
    )
    for line in reasons:
        print(f"classify_changes: ADR question — {line}", file=sys.stderr)
    emit(decision)
    # The answer, on one GITHUB_OUTPUT-safe line, so `check-pr-body` can quote it in the
    # message it fails with — the point of the binding is that the message names the ADR.
    print("adr_question=" + " | ".join(reasons))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
