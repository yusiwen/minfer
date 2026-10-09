#!/usr/bin/env python3
"""Check that the execution plan's status prose agrees with its machine-readable source.

`scripts/status.toml` is the one source for the plan's phase counters, its `next:`
sentence and its baseline / `refreshed against` commit ids. Those were hand-edited
prose once and drifted (the preamble read `Phase C 6/8` two lines above its own
"complete (8/8)", and the `next:` sentence still called E4/E5 upcoming long after
they landed — issue #94). Every prose target carries its own `file` + `regex` in
the manifest, so this checker holds no second copy of any value and can report
*the file, the line and both values* when they disagree.

Modes:

- ``--check`` (the default) reads `scripts/status.toml` and asserts the prose
  agrees: the phase counters and completion words, the `next:` sentence and the two
  commit ids — and that both ids are ancestors of ``HEAD``
  (``git merge-base --is-ancestor``, so "refreshed against master = X" cannot point
  at a rewritten-away commit).
- ``--selftest`` runs the pass/fail cases in a hermetic temp git repo (a good tree,
  a mutated phase counter, a mutated source value, a missing commit id, a
  non-ancestor commit, a `[[phase]]`-less manifest and a foreign `kind`), so CI
  exercises the logic on every PR without touching the real records.

The suite measurements are **not** in this ledger and not checked here: they live
in `scripts/test-baselines.toml`, are validated against `docs/TEST-BASELINES.md` by
`scripts/check_baselines.py`, and its `--check-live` compares one row against a real
`cargo test` log. The split exists because the two halves have different prose
targets and different consumers; the placement rule is recorded in
`docs/adr/0023-the-machine-ledgers-live-beside-their-checkers.md`. That sibling
imports this module's shared plumbing (``read_text``, ``line_of``, ``label_of``,
``has_problem``, ``require_kind``, ``usage_error``) rather than restating it — the
pattern `scripts/check_anchor_drift.py` uses for `check_doc_line_anchors.py`.

**What it proves, and what it cannot.** ``--help`` states the limit in full: this
checker can prove that the prose *agrees with the source*; it cannot prove that a
measurement was real. The plan's ``§11`` sequencing diagram's per-ticket check
marks are deliberately **out of scope** — they are not derivable from
``(done, total)``.

Usage::

    python3 scripts/check_status.py --check
    python3 scripts/check_status.py --selftest

Pure and offline: stdlib only (``tomllib``), no network. ``git`` is invoked only
for the ancestor test and by ``--selftest``'s fixture repo.

Exit codes: 0 the check passes, 1 the check fails, 2 usage or I/O error.

"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

#: The manifest this checker owns, relative to `--root`. Its sibling
#: `scripts/check_baselines.py` owns `scripts/test-baselines.toml`.
MANIFEST = "scripts/status.toml"

#: The `kind` this checker owns. A manifest declaring another kind is a different
#: ledger handed to the wrong checker, and `require_kind` refuses it by name.
KIND = "plan"

#: The top-level scalars a `plan` manifest must carry. A missing one is a hard
#: failure: a ledger that lost its declaration must not read as "nothing to check".
REQUIRED_SCALARS = ("baseline_commit", "refreshed_against")

#: The honest limit `--help` must state (also asserted by `--selftest`).
HELP_LIMIT = (
    "What it proves: the plan's prose agrees with scripts/status.toml. What it cannot "
    "prove: that a measurement was real. The plan's §11 sequencing diagram's per-ticket "
    "check marks are deliberately out of scope — they are not derivable from (done, total)."
)

#: Fields a phase regex may capture and the manifest stores.
PHASE_FIELDS = ("state", "done", "total")


# ---------------------------------------------------------------------------
# Shared plumbing. `scripts/check_baselines.py` imports these rather than
# restating them (the `scripts/check_anchor_drift.py` pattern), so the two
# ledgers stay separate while their prose-comparison mechanics stay one
# implementation.
# ---------------------------------------------------------------------------


def line_of(text: str, offset: int) -> int:
    """The 1-based line in `text` holding byte/character `offset`."""
    return text.count("\n", 0, offset) + 1


def read_text(path: Path) -> str:
    """Read `path` as UTF-8, or raise `OSError` (the caller reports it)."""
    return path.read_text(encoding="utf-8")


def label_of(root: Path, path: Path) -> str:
    """`path` relative to `root` when it is under it, else its absolute form."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def has_problem(problems: list[str], *needles: str) -> bool:
    """Whether some problem message carries every one of `needles`."""
    return any(all(needle in problem for needle in needles) for problem in problems)


def require_kind(
    manifest: dict, label: str, kind: str, problems: list[str]
) -> bool:
    """Refuse a manifest whose `kind` is not `kind`; True when it matched.

    A missing or foreign `kind` is a hard failure, never a reason to skip the
    sections it implies: "the ledger lost its declaration" must not read as
    "there is nothing to check".
    """
    got = manifest.get("kind")
    if got == kind:
        return True
    problems.append(
        f"{label}: kind is {str(got)!r}, not {kind!r} — a different ledger was "
        f"handed to this checker"
    )
    return False


def usage_error(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def is_ancestor(root: Path, commit: str) -> tuple[bool, int, str]:
    """`(ok, returncode, stderr)` for `git merge-base --is-ancestor COMMIT HEAD`."""
    proc = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.returncode, proc.stderr.strip()


def compare_scalars(
    root: Path, manifest: dict, label: str, problems: list[str]
) -> dict[str, tuple[str, int]]:
    """Compare each `[prose.<field>]` `value` group against the manifest scalar.

    Returns `{field: (value, line)}` for the fields that matched, so the caller
    can reuse the line in an ancestry failure.
    """
    seen: dict[str, tuple[str, int]] = {}
    for field, target in manifest.get("prose", {}).items():
        want = manifest.get(field)
        if want is None:
            problems.append(
                f"{label}: [prose.{field}] has no matching top-level {field!r} value"
            )
            continue
        text = read_text(root / target["file"])
        match = re.search(target["regex"], text, re.MULTILINE)
        if match is None or "value" not in match.groupdict():
            problems.append(
                f'{target["file"]}: no prose match for {field!r} '
                f"(the ledger says {str(want)!r})"
            )
            continue
        line = line_of(text, match.start("value"))
        got = match.group("value")
        if got != want:
            problems.append(
                f'{target["file"]}:{line}: {field}: prose says {got!r}, '
                f"the ledger says {str(want)!r}"
            )
        else:
            seen[field] = (got, line)
    return seen


def compare_phases(
    root: Path, manifest: dict, label: str, problems: list[str]
) -> int:
    """Compare every `[[phase]]` row against its own prose regex. Returns rows checked."""
    checked = 0
    for phase in manifest.get("phase", []):
        pid = phase["id"]
        text = read_text(root / phase["file"])
        match = re.search(phase["regex"], text, re.MULTILINE)
        if match is None:
            problems.append(
                f'{phase["file"]}: no prose match for phase {pid} '
                f"(state {phase['state']!r}, {phase['done']}/{phase['total']})"
            )
            continue
        checked += 1
        for field in PHASE_FIELDS:
            if field not in match.groupdict():
                continue
            got = match.group(field)
            want = phase[field]
            got_norm: object = got.strip()
            want_norm: object = want
            if isinstance(want, int):
                try:
                    got_norm = int(got)
                except ValueError:
                    problems.append(
                        f'{phase["file"]}: phase {pid} {field}: prose says {got!r}, '
                        f"the ledger says {str(want)!r}"
                    )
                    continue
            if got_norm != want_norm:
                line = line_of(text, match.start(field))
                problems.append(
                    f'{phase["file"]}:{line}: phase {pid} {field}: prose says {got!r}, '
                    f"the ledger says {str(want)!r}"
                )
    return checked


def check_manifest(root: Path, status_path: Path) -> tuple[list[str], dict]:
    """Every prose/source problem for `status_path` under `root`, plus a summary.

    A manifest of the wrong `kind` short-circuits: the one problem names it, rather
    than burying it under "this other ledger has no baseline_commit" noise.
    """
    label = label_of(root, status_path)
    manifest = tomllib.loads(read_text(status_path))
    problems: list[str] = []
    if not require_kind(manifest, label, KIND, problems):
        return problems, {"phases": 0, "scalars": 0}

    for field in REQUIRED_SCALARS:
        if not manifest.get(field):
            problems.append(f"{label}: {field} is missing")
    if not manifest.get("phase"):
        problems.append(
            f"{label}: no [[phase]] rows — a truncated ledger must not pass"
        )

    scalars = compare_scalars(root, manifest, label, problems)
    phases = compare_phases(root, manifest, label, problems)

    for field in REQUIRED_SCALARS:
        commit = manifest.get(field)
        if not commit:
            continue
        ok, code, message = is_ancestor(root, commit)
        if ok:
            continue
        where = manifest.get("prose", {}).get(field, {}).get("file", label)
        line = scalars.get(field, (None, None))[1]
        location = f"{where}:{line}" if line else where
        detail = f" ({message})" if message else ""
        problems.append(
            f"{location}: {field} = {commit} is not an ancestor of HEAD "
            f"(git merge-base --is-ancestor exited {code}){detail}"
        )

    return problems, {"phases": phases, "scalars": len(scalars)}


# --------------------------------------------------------------------------
# --selftest: a hermetic fixture repo, so CI runs the logic without a read of
# the real records. The suite-count half's cases live in check_baselines.py.
# --------------------------------------------------------------------------

# The fixture is deliberately a *subset* of the real manifest/schema (two phases)
# with its own values; it exercises the code paths, not the real numbers. Its
# regexes are written exactly as they appear in the TOML.
FIXTURE_PLAN = r"""# Fixture plan

**Status:** Phase A **complete** (9/9, 2026-09-01); Phase G **scheduled** (0/7) — later.

**Next: the Metal round (on a Mac).**

**Baseline:** `HEAD = __BASE__` (2026-09-01). This status was refreshed against `master =
__REFRESH__` (2026-09-02); it is refreshed with every PR.
"""

FIXTURE_STATUS = r"""schema = 1
kind = "plan"
baseline_commit = "__BASE__"
refreshed_against = "__REFRESH__"
next = "the Metal round (on a Mac)"

[prose.baseline_commit]
file = "docs/ARCHITECTURE-EXECUTION-PLAN.md"
regex = 'HEAD = (?P<value>[0-9a-f]{7,40})'

[prose.refreshed_against]
file = "docs/ARCHITECTURE-EXECUTION-PLAN.md"
regex = 'master =\s*(?P<value>[0-9a-f]{7,40})'

[prose.next]
file = "docs/ARCHITECTURE-EXECUTION-PLAN.md"
regex = 'Next: (?P<value>[^\n]+?)\.\*\*'

[[phase]]
id = "A"
done = 9
total = 9
state = "complete"
file = "docs/ARCHITECTURE-EXECUTION-PLAN.md"
regex = 'Phase A\s+\*\*(?P<state>[a-z ]+)\*\* \((?P<done>[0-9]+)/(?P<total>[0-9]+)'

[[phase]]
id = "G"
done = 0
total = 7
state = "scheduled"
file = "docs/ARCHITECTURE-EXECUTION-PLAN.md"
regex = 'Phase G\s+\*\*(?P<state>[a-z ]+)\*\* \((?P<done>[0-9]+)/(?P<total>[0-9]+)'
"""


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    """Run git in `cwd` with a hermetic identity and no signing/hooks."""
    return subprocess.run(
        [
            "git",
            "-C",
            str(cwd),
            "-c",
            "user.email=check-status@example.invalid",
            "-c",
            "user.name=check-status",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "core.hooksPath=/dev/null",
            *args,
        ],
        capture_output=True,
        text=True,
    )


def _git_checked(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    proc = _git(cwd, *args)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc


def build_fixture(root: Path) -> tuple[str, str, str]:
    """A two-branch git repo whose manifest passes; returns `(c1, c2, c3)`.

    `c1` is the ancestor both commits descend from, `c2` is `HEAD`, and `c3` is
    on a sibling branch — the non-ancestor the selftest needs.
    """
    root.mkdir(parents=True, exist_ok=True)
    _git_checked(root, "init", "-q")
    _git_checked(root, "commit", "-q", "--allow-empty", "-m", "c1")
    c1 = _git_checked(root, "rev-parse", "HEAD").stdout.strip()
    write_fixture(root, c1)
    _git_checked(root, "add", "-A")
    _git_checked(root, "commit", "-q", "-m", "c2")
    c2 = _git_checked(root, "rev-parse", "HEAD").stdout.strip()
    branch = _git_checked(root, "symbolic-ref", "--short", "HEAD").stdout.strip()
    _git_checked(root, "checkout", "-q", "-b", "other", c1)
    _git_checked(root, "commit", "-q", "--allow-empty", "-m", "c3")
    c3 = _git_checked(root, "rev-parse", "HEAD").stdout.strip()
    _git_checked(root, "checkout", "-q", branch)
    return c1, c2, c3


def default_plan(base: str, refresh: str | None = None) -> str:
    """The passing fixture plan, baselined at `base` and refreshed at `refresh`."""
    return FIXTURE_PLAN.replace("__BASE__", base).replace("__REFRESH__", refresh or base)


def default_status(base: str, refresh: str | None = None) -> str:
    """The passing fixture manifest for `default_plan`."""
    return FIXTURE_STATUS.replace("__BASE__", base).replace(
        "__REFRESH__", refresh or base
    )


def write_fixture(
    root: Path,
    base: str,
    plan: str | None = None,
    status: str | None = None,
    refresh: str | None = None,
) -> None:
    """Write the fixture tree; `None` keeps the passing default for that file."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "ARCHITECTURE-EXECUTION-PLAN.md").write_text(
        plan if plan is not None else default_plan(base, refresh), encoding="utf-8"
    )
    (root / "scripts" / "status.toml").write_text(
        status if status is not None else default_status(base, refresh), encoding="utf-8"
    )


def run_selftest() -> int:
    """Run the built-in pass/fail cases and return 0 when every one behaves."""
    with tempfile.TemporaryDirectory(prefix="check_status_selftest_") as tmp:
        root = Path(tmp)
        c1, _c2, c3 = build_fixture(root)
        status = root / "scripts" / "status.toml"
        results: list[tuple[str, bool, str]] = []

        def record(name: str, ok: bool, detail: str = "") -> None:
            results.append((name, ok, detail))

        # 1. The passing tree.
        write_fixture(root, c1)
        problems, _summary = check_manifest(root, status)
        record("a tree whose prose agrees passes", not problems, str(problems))

        # 2. A mutated phase counter in the plan.
        write_fixture(root, c1, plan=default_plan(c1).replace("9/9", "10/9"))
        problems, _ = check_manifest(root, status)
        record(
            "a mutated plan phase counter fails with file, line and both values",
            has_problem(problems, "ARCHITECTURE-EXECUTION-PLAN.md", "phase A done", "'10'", "'9'"),
            str(problems),
        )

        # 3. The other direction: the source edited, the prose left alone.
        write_fixture(root, c1, status=default_status(c1).replace("done = 9", "done = 8", 1))
        problems, _ = check_manifest(root, status)
        record(
            "a mutated scripts/status.toml value fails against the prose",
            has_problem(problems, "phase A done", "'9'", "'8'"),
            str(problems),
        )

        # 4. A missing commit id must not read as "nothing to check".
        write_fixture(
            root, c1, status=default_status(c1).replace('baseline_commit = "' + c1 + '"', "")
        )
        problems, _ = check_manifest(root, status)
        record(
            "a missing baseline_commit fails by name",
            has_problem(problems, "scripts/status.toml", "baseline_commit is missing"),
            str(problems),
        )

        # 5. A [[phase]]-less manifest must not pass vacuously.
        write_fixture(
            root,
            c1,
            status="schema = 1\nkind = \"plan\"\nbaseline_commit = \"%s\"\n"
            "refreshed_against = \"%s\"\n" % (c1, c1),
        )
        problems, _ = check_manifest(root, status)
        record(
            "a manifest with no [[phase]] rows fails",
            has_problem(problems, "no [[phase]] rows"),
            str(problems),
        )

        # 6. The sibling ledger handed to this checker: one problem, named.
        write_fixture(root, c1, status=default_status(c1).replace('kind = "plan"', 'kind = "counts"'))
        problems, _ = check_manifest(root, status)
        record(
            "the counts ledger is refused by name, and alone",
            has_problem(problems, "kind is 'counts', not 'plan'") and len(problems) == 1,
            str(problems),
        )

        # 7. A non-ancestor commit fails the ancestry test and *nothing else*, so
        #    the control differs only in the property under test (gate contract
        #    rule 2): the prose and the source both name the sibling-branch `c3`.
        write_fixture(root, c1, plan=default_plan(c1, c3), status=default_status(c1, c3))
        problems, _ = check_manifest(root, status)
        record(
            "a non-ancestor commit fails the ancestry test",
            has_problem(problems, "refreshed_against", "is not an ancestor of HEAD")
            and len(problems) == 1,
            str(problems),
        )

        # 8. `--help` states the honest limit.
        record(
            "--help states the honest limit",
            HELP_LIMIT in build_parser().description,
            "",
        )

    failures = 0
    for name, ok, detail in results:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1
            print(f"      got: {detail or 'no problems'}")
    total = len(results)
    print(f"check_status selftest: {total - failures}/{total} cases pass")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_status.py",
        description=(
            "Check that the execution plan's status prose agrees with "
            "scripts/status.toml (the phase counters, the next: sentence and the "
            "baseline/refreshed commit ids). The suite counts are the sibling "
            "ledger scripts/test-baselines.toml, checked by check_baselines.py. "
            + HELP_LIMIT
        ),
        epilog=(
            "Every prose target's file and regex live in scripts/status.toml, so "
            "this script holds no second copy of any value. Edit the source, not "
            "the counter."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert the prose agrees with the source (the default mode)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the built-in pass/fail cases in a temp repo and exit non-zero on a failure",
    )
    parser.add_argument(
        "--root",
        metavar="DIR",
        help="the repository root (default: the repo this script lives in)",
    )
    parser.add_argument(
        "--status",
        metavar="FILE",
        help=f"the manifest (default: {MANIFEST} under --root)",
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    if args.selftest:
        if args.check or args.status:
            parser.error("--selftest does not combine with --check/--status")
        return run_selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    status = Path(args.status).resolve() if args.status else root / MANIFEST

    try:
        problems, summary = check_manifest(root, status)
    except OSError as exc:
        usage_error(f"check_status: cannot read {status}: {exc}")
    except tomllib.TOMLDecodeError as exc:
        usage_error(f"check_status: {status} is not valid TOML: {exc}")

    if problems:
        for problem in problems:
            print(f"check_status: {problem}", file=sys.stderr)
        print(
            f"check_status: the prose disagrees with {status} "
            f"({len(problems)} problem(s))",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_status: prose agrees with {status} "
        f"({summary['phases']} phases, {summary['scalars']} scalars); "
        f"baseline/refreshed commits are ancestors of HEAD"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
