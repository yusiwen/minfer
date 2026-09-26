#!/usr/bin/env python3
"""Check that the status prose agrees with its machine-readable source.

The execution plan's phase counters and its `next:` sentence, its baseline and
`refreshed against` commit ids, and the live suite counts recorded in `AGENTS.md`
used to be hand-edited prose. They drifted (the preamble once read `Phase C 6/8`
two lines above its own "complete (8/8)"; the `next:` sentence still called E4/E5
upcoming long after they landed; `AGENTS.md`'s CUDA count trailed the real run by
a week). This script makes them **derived facts**: `docs/status.toml` is the one
source, and every prose target carries its own `file` + `regex` (with named
groups), so the checker holds no second copy of any value and can report *the
file, the line and both values* when they disagree (issue #94).

Modes:

- ``--check`` (the default) reads `docs/status.toml` and asserts the prose agrees:
  the plan's phase counters and completion words, its `next:` sentence, its two
  commit ids — and that both ids are ancestors of ``HEAD``
  (``git merge-base --is-ancestor``, so "refreshed against master = X" cannot point
  at a rewritten-away commit) — plus `AGENTS.md`'s count rows (numbers, box and
  date).
- ``--check-live LOG --box NAME`` parses a `cargo test` log into its unit and
  integration totals and compares the manifest rows for that box. Hardening that
  is not optional: a box with no manifest rows fails loudly; a log with no
  ``Running unittests`` or no ``Running tests/`` block fails (truncated/empty);
  a non-zero ``failed`` count fails; ANSI escapes are stripped before matching;
  and each ``test result:`` block is attributed to the nearest preceding
  ``Running`` header.
- ``--selftest`` runs the pass/fail cases in a hermetic temp git repo (a good
  tree, a mutated phase counter, a mutated count, a mutated source value, a
  missing box, a truncated log and a non-ancestor commit), so CI exercises the
  logic on every PR without a live cargo run.

**What it proves, and what it cannot.** ``--help`` states the limit in full: this
checker can prove that the prose *agrees with the source* and that a counts row's
numbers *matched a cargo run that was parsed*; it cannot prove a measurement was
real. The ``§11`` sequencing diagram's per-ticket check marks are deliberately
**out of scope** — they are not derivable from ``(done, total)``.

Usage::

    python3 scripts/check_status.py --check
    python3 scripts/check_status.py --check-live /tmp/cargo-test.log --box x86_64-ci
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

#: ANSI SGR escapes: the CI log carries them *inside* the `Running` header
#: (`\x1b[1m\x1b[92m     Running\x1b[0m tests/...`), so they are stripped before
#: any matching.
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
#: `Running unittests src/main.rs (…)` / `Running tests/foo.rs (…)`.
RUNNING_RE = re.compile(r"Running (unittests|tests/)")
#: `   Doc-tests minfer` — a later block without the `Running` prefix; a result
#: under it must not be attributed to the last integration binary.
DOCTEST_RE = re.compile(r"^\s*Doc-tests\b")
#: `test result: ok. 455 passed; 0 failed; 33 ignored; …` (or `FAILED.`).
RESULT_RE = re.compile(
    r"test result: (?:ok|FAILED)\. ([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored"
)

#: The honest limit `--help` must state (also asserted by `--selftest`).
HELP_LIMIT = (
    "What it proves: the prose agrees with docs/status.toml, and (in --check-live) "
    "that a counts row's numbers matched a cargo log that was parsed. What it cannot "
    "prove: that a measurement was real. The §11 sequencing diagram's per-ticket check "
    "marks are deliberately out of scope — they are not derivable from (done, total)."
)

#: Fields a phase regex may capture and the manifest stores.
PHASE_FIELDS = ("state", "done", "total")


def line_of(text: str, offset: int) -> int:
    """The 1-based line in `text` holding byte/character `offset`."""
    return text.count("\n", 0, offset) + 1


def read_text(path: Path) -> str:
    """Read `path` as UTF-8, or raise `OSError` (the caller reports it)."""
    return path.read_text(encoding="utf-8")


def is_ancestor(root: Path, commit: str) -> tuple[bool, int, str]:
    """`(ok, returncode, stderr)` for `git merge-base --is-ancestor COMMIT HEAD`."""
    proc = subprocess.run(
        ["git", "-C", str(root), "merge-base", "--is-ancestor", commit, "HEAD"],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, proc.returncode, proc.stderr.strip()


def compare_scalars(
    root: Path, manifest: dict, problems: list[str]
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
                f"docs/status.toml: [prose.{field}] has no matching top-level {field!r} value"
            )
            continue
        text = read_text(root / target["file"])
        match = re.search(target["regex"], text, re.MULTILINE)
        if match is None or "value" not in match.groupdict():
            problems.append(
                f'{target["file"]}: no prose match for {field!r} '
                f'(docs/status.toml says {str(want)!r})'
            )
            continue
        line = line_of(text, match.start("value"))
        got = match.group("value")
        if got != want:
            problems.append(
                f'{target["file"]}:{line}: {field}: prose says {got!r}, '
                f"docs/status.toml says {str(want)!r}"
            )
        else:
            seen[field] = (got, line)
    return seen


def compare_phases(root: Path, manifest: dict, problems: list[str]) -> int:
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
                        f"docs/status.toml says {str(want)!r}"
                    )
                    continue
            if got_norm != want_norm:
                line = line_of(text, match.start(field))
                problems.append(
                    f'{phase["file"]}:{line}: phase {pid} {field}: prose says {got!r}, '
                    f"docs/status.toml says {str(want)!r}"
                )
    return checked


def compare_counts(root: Path, manifest: dict, problems: list[str]) -> int:
    """Compare every `[[counts]]` row against its own prose regex.

    A row's regex is searched for all matches; the match whose `box` group equals
    the row's box is used, so two rows sharing a command (`cargo test --release`
    on two boxes) still bind to distinct bullets. Returns rows checked.
    """
    checked = 0
    for row in manifest.get("counts", []):
        ident = f'{row["key"]} ({row["box"]})'
        text = read_text(root / row["file"])
        match = None
        for candidate in re.finditer(row["regex"], text, re.MULTILINE):
            groups = candidate.groupdict()
            if "box" not in groups or groups["box"].strip() == row["box"]:
                match = candidate
                break
        if match is None:
            problems.append(
                f'{row["file"]}: no prose match for {ident} '
                f"(box {row['box']!r}) — the counter was edited or the box renamed"
            )
            continue
        checked += 1
        for field in match.groupdict():
            if field == "box" or field not in row:
                continue
            got = match.group(field)
            want = row[field]
            got_norm: object = got.strip()
            if isinstance(want, int):
                try:
                    got_norm = int(got)
                except ValueError:
                    pass
            if got_norm != want:
                line = line_of(text, match.start(field))
                problems.append(
                    f'{row["file"]}:{line}: {ident} {field}: prose says {got!r}, '
                    f"docs/status.toml says {str(want)!r}"
                )
    return checked


def check_manifest(root: Path, status_path: Path) -> tuple[list[str], dict]:
    """Every prose/source problem for `status_path` under `root`, plus a summary."""
    manifest = tomllib.loads(status_path.read_text(encoding="utf-8"))
    problems: list[str] = []
    scalars = compare_scalars(root, manifest, problems)
    phases = compare_phases(root, manifest, problems)
    counts = compare_counts(root, manifest, problems)

    for field in ("baseline_commit", "refreshed_against"):
        commit = manifest.get(field)
        if not commit:
            problems.append(f"docs/status.toml: {field} is missing")
            continue
        ok, code, message = is_ancestor(root, commit)
        if ok:
            continue
        where = manifest.get("prose", {}).get(field, {}).get("file", "docs/status.toml")
        line = scalars.get(field, (None, None))[1]
        location = f"{where}:{line}" if line else where
        detail = f" ({message})" if message else ""
        problems.append(
            f"{location}: {field} = {commit} is not an ancestor of HEAD "
            f"(git merge-base --is-ancestor exited {code}){detail}"
        )

    live = [row for row in manifest.get("counts", []) if row.get("live_check")]
    summary = {
        "phases": phases,
        "counts": counts,
        "live": live,
        "recorded": len(manifest.get("counts", [])) - len(live),
        "scalars": len(scalars),
    }
    return problems, summary


def parse_cargo_log(text: str) -> tuple[dict[str, list[int]], set[str]]:
    """`({suite: [passed, failed, ignored]}, suites_seen)` for a `cargo test` log."""
    cleaned = ANSI_RE.sub("", text)
    totals: dict[str, list[int]] = {"unit": [0, 0, 0], "integration": [0, 0, 0]}
    seen: set[str] = set()
    bucket: str | None = None
    for raw in cleaned.splitlines():
        running = RUNNING_RE.search(raw)
        if running:
            bucket = "unit" if running.group(1) == "unittests" else "integration"
            seen.add(bucket)
            continue
        if DOCTEST_RE.match(raw):
            bucket = "other"
            continue
        result = RESULT_RE.search(raw)
        if result and bucket in totals:
            for index in range(3):
                totals[bucket][index] += int(result.group(index + 1))
    return totals, seen


def check_live(status_path: Path, log_path: Path, box: str) -> list[str]:
    """Compare the manifest rows for `box` against a parsed cargo log."""
    problems: list[str] = []
    manifest = tomllib.loads(status_path.read_text(encoding="utf-8"))
    rows = [row for row in manifest.get("counts", []) if row["box"] == box]
    if not rows:
        problems.append(
            f"no manifest rows for box {box!r} — refusing a vacuous pass "
            f"(known boxes: {sorted({r['box'] for r in manifest.get('counts', [])})})"
        )
        return problems
    cargo_rows = [row for row in rows if row.get("suite") == "unit"]
    if not cargo_rows:
        problems.append(
            f"no cargo-test rows for box {box!r} — refusing a vacuous pass"
        )
        return problems
    try:
        text = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        problems.append(f"cannot read the cargo log {log_path}: {exc}")
        return problems

    totals, seen = parse_cargo_log(text)
    if "unit" not in seen:
        problems.append(
            f'{log_path}: no `Running unittests` block — the log is truncated or empty'
        )
    if "integration" not in seen:
        problems.append(
            f'{log_path}: no `Running tests/` block — the log is truncated or empty'
        )
    unit = totals["unit"]
    integration = totals["integration"]
    if unit[1] or integration[1]:
        problems.append(
            f"{log_path}: the run is red (unit {unit[1]} failed, "
            f"integration {integration[1]} failed) — refusing to compare it"
        )
    for row in cargo_rows:
        ident = f'{row["key"]} ({box})'
        for field, got in (("passed", unit[0]), ("failed", unit[1]), ("ignored", unit[2])):
            if field in row and row[field] != got:
                problems.append(
                    f'{row["file"]}: {ident} {field}: '
                    f"docs/status.toml says {row[field]}, the run says {got}"
                )
        if "integration_passed" in row:
            for field, got in (
                ("integration_passed", integration[0]),
                ("integration_failed", integration[1]),
                ("integration_ignored", integration[2]),
            ):
                if field in row and row[field] != got:
                    problems.append(
                        f'{row["file"]}: {ident} {field}: '
                        f"docs/status.toml says {row[field]}, the run says {got}"
                    )
    return problems


# --------------------------------------------------------------------------
# --selftest: a hermetic fixture repo, so CI runs the logic without a cargo run
# --------------------------------------------------------------------------

# The fixture is deliberately a *subset* of the real manifest/schema (two phases,
# two count rows) with its own values; it exercises the code paths, not the real
# numbers. Its regexes are written exactly as they appear in the TOML.
FIXTURE_PLAN = r"""# Fixture plan

**Status:** Phase A **complete** (9/9, 2026-09-01); Phase G **scheduled** (0/7) — later.

**Next: the Metal round (on a Mac).**

**Baseline:** `HEAD = __BASE__` (2026-09-01). This status was refreshed against `master =
__REFRESH__` (2026-09-02); it is refreshed with every PR.
"""

FIXTURE_AGENTS = r"""# Fixture agents

  - CPU unit, box `aarch64 (this box)`, `cargo test --release`, 2026-09-01: **457 passed / 0 failed / 3 ignored** unit + **10 / 0 / 6** integration.
  - CPU unit, box `x86_64 (CI runner)`, `cargo test --release`, 2026-09-01: **455 passed / 0 failed / 3 ignored** unit + **10 / 0 / 6** integration.
"""

_FIXTURE_COUNTS_RE = (
    r'CPU unit, box `(?P<box>[^`]+)`, `cargo test --release`, '
    r'(?P<date>[0-9]{4}-[0-9]{2}-[0-9]{2}): '
    r'\*\*(?P<passed>[0-9]+) passed / (?P<failed>[0-9]+) failed / (?P<ignored>[0-9]+) ignored\*\* unit \+ '
    r'\*\*(?P<integration_passed>[0-9]+) / (?P<integration_failed>[0-9]+) / (?P<integration_ignored>[0-9]+)\*\* integration'
)

FIXTURE_TOML = (
    r"""schema = 1
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

[[counts]]
key = "cpu-unit"
box = "aarch64 (this box)"
suite = "unit"
command = "cargo test --release"
date = "2026-09-01"
passed = 457
failed = 0
ignored = 3
integration_passed = 10
integration_failed = 0
integration_ignored = 6
live_check = false
file = "AGENTS.md"
regex = '__COUNTS_RE__'

[[counts]]
key = "cpu-unit"
box = "x86_64 (CI runner)"
suite = "unit"
command = "cargo test --release"
date = "2026-09-01"
passed = 455
failed = 0
ignored = 3
integration_passed = 10
integration_failed = 0
integration_ignored = 6
live_check = true
file = "AGENTS.md"
regex = '__COUNTS_RE__'
"""
).replace("__COUNTS_RE__", _FIXTURE_COUNTS_RE)

FIXTURE_LOG = (
    "\x1b[1m\x1b[92m     Running\x1b[0m unittests src/main.rs (target/release/deps/minfer-abc)\n"
    "\nrunning 460 tests\n"
    "test result: ok. 455 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 1.00s\n"
    "\n"
    "\x1b[1m\x1b[92m     Running\x1b[0m tests/conversation_cli.rs (target/release/deps/conversation_cli-abc)\n"
    "\nrunning 16 tests\n"
    "test result: ok. 10 passed; 0 failed; 6 ignored; 0 measured; 0 filtered out; finished in 0.05s\n"
)

FIXTURE_LOG_TRUNCATED = (
    "\x1b[1m\x1b[92m     Running\x1b[0m unittests src/main.rs (target/release/deps/minfer-abc)\n"
    "test result: ok. 455 passed; 0 failed; 3 ignored; 0 measured; 0 filtered out; finished in 1.00s\n"
)


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


def default_toml(base: str, refresh: str | None = None) -> str:
    """The passing fixture manifest for `default_plan`."""
    return FIXTURE_TOML.replace("__BASE__", base).replace("__REFRESH__", refresh or base)


def write_fixture(
    root: Path,
    base: str,
    plan: str | None = None,
    agents: str | None = None,
    toml: str | None = None,
    refresh: str | None = None,
) -> None:
    """Write the fixture tree; `None` keeps the passing default for that file."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "ARCHITECTURE-EXECUTION-PLAN.md").write_text(
        plan if plan is not None else default_plan(base, refresh), encoding="utf-8"
    )
    (root / "AGENTS.md").write_text(
        agents if agents is not None else FIXTURE_AGENTS, encoding="utf-8"
    )
    (root / "docs" / "status.toml").write_text(
        toml if toml is not None else default_toml(base, refresh), encoding="utf-8"
    )


def _has(problems: list[str], *needles: str) -> bool:
    return any(all(needle in problem for needle in needles) for problem in problems)


def run_selftest() -> int:
    """Run the built-in pass/fail cases and return 0 when every one behaves."""
    with tempfile.TemporaryDirectory(prefix="check_status_selftest_") as tmp:
        root = Path(tmp)
        c1, _c2, c3 = build_fixture(root)
        status = root / "docs" / "status.toml"
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
            _has(problems, "ARCHITECTURE-EXECUTION-PLAN.md", "phase A done", "'10'", "'9'"),
            str(problems),
        )

        # 3. A mutated AGENTS.md count.
        write_fixture(root, c1, agents=FIXTURE_AGENTS.replace("**455 passed", "**999 passed"))
        problems, _ = check_manifest(root, status)
        record(
            "a mutated AGENTS.md count fails with both values",
            _has(problems, "AGENTS.md", "x86_64 (CI runner)", "'999'", "'455'"),
            str(problems),
        )

        # 4. The other direction: the source edited, the prose left alone.
        write_fixture(root, c1, toml=default_toml(c1).replace("done = 9", "done = 8", 1))
        problems, _ = check_manifest(root, status)
        record(
            "a mutated docs/status.toml value fails against the prose",
            _has(problems, "phase A done", "'9'", "'8'"),
            str(problems),
        )

        # 5. A box with no manifest rows (a fabricated box) must not pass vacuously.
        write_fixture(root, c1)
        problems = check_live(status, root / "cargo.log", "GB10 sm_121")
        record(
            "a box with no manifest rows fails loudly",
            _has(problems, "no manifest rows for box"),
            str(problems),
        )

        # 6. A truncated log (no `Running tests/` block) must not pass.
        log = root / "cargo.log"
        log.write_text(FIXTURE_LOG_TRUNCATED, encoding="utf-8")
        problems = check_live(status, log, "x86_64 (CI runner)")
        record(
            "a truncated log fails on the missing integration block",
            _has(problems, "no `Running tests/` block"),
            str(problems),
        )

        # 7. A non-ancestor commit fails the ancestry test and *nothing else*, so
        #    the control differs only in the property under test (gate contract
        #    rule 2): the prose and the source both name the sibling-branch `c3`.
        write_fixture(root, c1, plan=default_plan(c1, c3), toml=default_toml(c1, c3))
        problems, _ = check_manifest(root, status)
        record(
            "a non-ancestor commit fails the ancestry test",
            _has(problems, "refreshed_against", "is not an ancestor of HEAD")
            and len(problems) == 1,
            str(problems),
        )

        # 8. The positive live case, then a live mismatch.
        write_fixture(root, c1)
        log.write_text(FIXTURE_LOG, encoding="utf-8")
        problems = check_live(status, log, "x86_64 (CI runner)")
        record("a matching cargo log passes --check-live", not problems, str(problems))
        write_fixture(root, c1, toml=default_toml(c1).replace("passed = 455", "passed = 999"))
        problems = check_live(status, log, "x86_64 (CI runner)")
        record(
            "a live count mismatch fails",
            _has(problems, "passed", "999", "455"),
            str(problems),
        )

        # 9. `--help` states the honest limit.
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
            "Check that the status prose agrees with docs/status.toml (phase "
            "counters, the next: sentence, the baseline/refreshed commit ids and "
            "AGENTS.md's suite counts), and optionally that the counts matched a "
            "cargo log. " + HELP_LIMIT
        ),
        epilog=(
            "Every prose target's file and regex live in docs/status.toml, so this "
            "script holds no second copy of any value. Edit the source, not the counter."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="assert the prose agrees with the source (the default mode)",
    )
    parser.add_argument(
        "--check-live",
        metavar="CARGO-LOG",
        help="parse a `cargo test` log and compare the rows for --box",
    )
    parser.add_argument(
        "--box",
        metavar="NAME",
        help="the box whose rows --check-live compares (e.g. `x86_64 (CI runner)`)",
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
        help="the manifest (default: docs/status.toml under --root)",
    )
    return parser


def usage_error(message: str) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(2)


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])

    if args.selftest:
        if args.check or args.check_live or args.box:
            parser.error("--selftest does not combine with --check/--check-live/--box")
        return run_selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    status = Path(args.status).resolve() if args.status else root / "docs" / "status.toml"

    if args.check_live:
        if not args.box:
            parser.error("--check-live requires --box")
        try:
            problems = check_live(status, Path(args.check_live), args.box)
        except OSError as exc:
            usage_error(f"check_status: cannot read {status}: {exc}")
        if problems:
            for problem in problems:
                print(f"check_status: {problem}", file=sys.stderr)
            print(
                f"check_status: --check-live failed for box {args.box!r} "
                f"({len(problems)} problem(s))",
                file=sys.stderr,
            )
            return 1
        print(f"check_status: the {args.box!r} rows match {args.check_live}")
        return 0

    if args.box:
        parser.error("--box is only meaningful with --check-live")

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
        f"({summary['phases']} phases, {summary['counts']} count rows: "
        f"{len(summary['live'])} live-checkable, {summary['recorded']} recorded "
        f"measurements); baseline/refreshed commits are ancestors of HEAD"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
