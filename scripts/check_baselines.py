#!/usr/bin/env python3
"""Check that the suite measurements agree with their machine-readable source.

`scripts/test-baselines.toml` is the one source for every suite number quoted in
`docs/TEST-BASELINES.md`. Those numbers were hand-edited prose once and drifted
(the plan's CUDA count trailed the real run by a week — issue #94). Every row
carries its own `file` + `regex` in the manifest, so this checker holds no second
copy of any value and can report *the file, the line and both values* when they
disagree.

Modes:

- ``--check`` (the default) reads `scripts/test-baselines.toml` and asserts the
  prose agrees: each row's numbers, box and date, matched to its own bullet. A
  recorded row may also carry a **projection** (``projection_key`` /
  ``projection_box`` / ``projection_base_passed``): when the CPU row it shares a
  test binary with has moved since it was measured, ``--check`` **prints** (never
  fails on) the projected value, so a stale recorded row cannot look current
  without a hint beside it (#207). The relation is inexact — the device-gated
  tests move independently — which is why it is a hint and not a comparison.
- ``--check-live LOG --box NAME`` parses a `cargo test` log into its unit and
  integration totals and compares the manifest rows for that box. Hardening that
  is not optional: a box with no manifest rows fails loudly; a log with no
  ``Running unittests`` or no ``Running tests/`` block fails (truncated/empty);
  a non-zero ``failed`` count fails; ANSI escapes are stripped before matching;
  and each ``test result:`` block is attributed to the nearest preceding
  ``Running`` header. This is the one mode with a live consumer: the
  `test-linux-cpu` job runs it against the `x86_64 (CI runner)` row.
- ``--selftest`` runs the pass/fail cases in a hermetic temp tree (a good tree, a
  mutated prose count, a mutated row, a `[[counts]]`-less manifest, a foreign
  `kind`, a box with no rows and a truncated log), so CI exercises the logic on
  every PR without a live cargo run.

The plan's counters and `next:` sentence live in `scripts/status.toml` and are
checked by `scripts/check_status.py`; the two ledgers are separate because their
prose targets and consumers are (see
`docs/adr/0023-the-machine-ledgers-live-beside-their-checkers.md`). The shared
plumbing — ``read_text``, ``line_of``, ``label_of``, ``has_problem``,
``require_kind``, ``usage_error`` — is imported from that module rather than
restated here, the same pattern `scripts/check_anchor_drift.py` uses for
`check_doc_line_anchors.py`.

**What it proves, and what it cannot.** ``--help`` states the limit in full: this
checker can prove that the prose *agrees with the source* and that a row's
numbers *matched a cargo run that was parsed*; it cannot prove a measurement was
real.

Usage::

    python3 scripts/check_baselines.py --check
    python3 scripts/check_baselines.py --check-live /tmp/cargo-test.log --box 'x86_64 (CI runner)'
    python3 scripts/check_baselines.py --selftest

Pure and offline: stdlib only (``tomllib``), no network, no git.

Exit codes: 0 the check passes, 1 the check fails, 2 usage or I/O error.

"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_status as status  # noqa: E402  (the shared prose-comparison plumbing)

#: The manifest this checker owns, relative to `--root`. The plan's counters and
#: `next:` sentence are the sibling `scripts/status.toml`, checked by
#: `scripts/check_status.py`.
MANIFEST = "scripts/test-baselines.toml"

#: The `kind` this checker owns. A manifest declaring another kind is a different
#: ledger handed to the wrong checker, and `status.require_kind` refuses it by name.
KIND = "counts"

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
    "What it proves: the prose agrees with scripts/test-baselines.toml, and (in "
    "--check-live) that a counts row's numbers matched a cargo log that was parsed. "
    "What it cannot prove: that a measurement was real."
)


def projection_hints(manifest: dict) -> list[str]:
    """Non-failing hints for a recorded row a CPU-side change has moved (#207).

    The CUDA rows are recorded measurements refreshed only on a device run, but
    they count the **same test binary plus the device-gated tests**: a
    feature-independent test added by a CPU-only ticket moves both. A row may
    therefore carry `projection_key` / `projection_box` (the CPU row that shares
    its binary) and `projection_base_passed` (that row's `passed` value at the
    moment this row was measured), and this function projects
    `passed + (source_passed - base)`.

    It is deliberately a **hint, never a check**: the device-gated tests move
    independently of the CPU row, so the relation is inexact. The point is that a
    stale row cannot look current without a projection beside it.
    """
    rows = {
        (row.get("key"), row.get("box")): row for row in manifest.get("counts", [])
    }
    hints: list[str] = []
    for row in manifest.get("counts", []):
        if "projection_key" not in row:
            continue
        source = rows.get((row["projection_key"], row.get("projection_box")))
        if source is None:
            continue
        base = int(row["projection_base_passed"])
        delta = int(source["passed"]) - base
        if delta == 0:
            continue
        hints.append(
            f'{row["key"]} ({row["box"]}) records {row["passed"]} passed but the '
            f'{source["key"]} ({source["box"]}) row it was measured with has moved '
            f'{delta:+d} since then, so a device run would report about '
            f'{int(row["passed"]) + delta} passed — refresh the row on the next '
            f"device run (#207)"
        )
    return hints


def compare_counts(
    root: Path, manifest: dict, label: str, problems: list[str]
) -> int:
    """Compare every `[[counts]]` row against its own prose regex.

    A row's regex is searched for all matches; the match whose `box` group equals
    the row's box is used, so two rows sharing a command (`cargo test --release`
    on two boxes) still bind to distinct bullets. Returns rows checked.
    """
    checked = 0
    for row in manifest.get("counts", []):
        ident = f'{row["key"]} ({row["box"]})'
        text = status.read_text(root / row["file"])
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
                line = status.line_of(text, match.start(field))
                problems.append(
                    f'{row["file"]}:{line}: {ident} {field}: prose says {got!r}, '
                    f"the ledger says {str(want)!r}"
                )
    return checked


def check_manifest(root: Path, manifest_path: Path) -> tuple[list[str], dict]:
    """Every prose/source problem for `manifest_path` under `root`, plus a summary.

    A manifest of the wrong `kind` short-circuits: the one problem names it, rather
    than burying it under "this other ledger has no baseline_commit" noise.
    """
    label = status.label_of(root, manifest_path)
    manifest = tomllib.loads(status.read_text(manifest_path))
    problems: list[str] = []
    if not status.require_kind(manifest, label, KIND, problems):
        return problems, {"counts": 0, "live": [], "recorded": 0, "hints": []}
    rows = manifest.get("counts", [])
    if not rows:
        problems.append(
            f"{label}: no [[counts]] rows — a truncated ledger must not pass"
        )
    counts = compare_counts(root, manifest, label, problems)
    live = [row for row in rows if row.get("live_check")]
    summary = {
        "counts": counts,
        "live": live,
        "recorded": len(rows) - len(live),
        "hints": projection_hints(manifest),
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


def check_live(
    manifest_path: Path, log_path: Path, box: str, label: str
) -> list[str]:
    """Compare the manifest rows for `box` against a parsed cargo log."""
    problems: list[str] = []
    manifest = tomllib.loads(status.read_text(manifest_path))
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
                    f"{label} says {row[field]}, the run says {got}"
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
                        f"{label} says {row[field]}, the run says {got}"
                    )
    return problems


# --------------------------------------------------------------------------
# --selftest: a hermetic fixture tree. No git here — this ledger carries no
# commit ids, so nothing needs a repository (the plan half's fixture builds one).
# --------------------------------------------------------------------------

# The fixture is deliberately a *subset* of the real manifest/schema (two count
# rows) with its own values; it exercises the code paths, not the real numbers.
# Its regex is written exactly as it appears in the TOML.
FIXTURE_BASELINES = r"""# Fixture baselines

  - CPU unit, box `dgxspark (aarch64, GB10 sm_121)`, `cargo test --release`, 2026-09-01: **457 passed / 0 failed / 3 ignored** unit + **10 / 0 / 6** integration.
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
kind = "counts"

[[counts]]
key = "cpu-unit"
box = "dgxspark (aarch64, GB10 sm_121)"
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
file = "docs/TEST-BASELINES.md"
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
file = "docs/TEST-BASELINES.md"
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


def write_fixture(
    root: Path, toml: str | None = None, prose: str | None = None
) -> None:
    """Write the fixture tree; `None` keeps the passing default for that file."""
    (root / "docs").mkdir(parents=True, exist_ok=True)
    (root / "scripts").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "TEST-BASELINES.md").write_text(
        prose if prose is not None else FIXTURE_BASELINES, encoding="utf-8"
    )
    (root / "scripts" / "test-baselines.toml").write_text(
        toml if toml is not None else FIXTURE_TOML, encoding="utf-8"
    )


def run_selftest() -> int:
    """Run the built-in pass/fail cases and return 0 when every one behaves."""
    with tempfile.TemporaryDirectory(prefix="check_baselines_selftest_") as tmp:
        root = Path(tmp)
        manifest = root / "scripts" / "test-baselines.toml"
        label = "scripts/test-baselines.toml"
        results: list[tuple[str, bool, str]] = []

        def record(name: str, ok: bool, detail: str = "") -> None:
            results.append((name, ok, detail))

        # 1. The passing tree.
        write_fixture(root)
        problems, _summary = check_manifest(root, manifest)
        record("a tree whose prose agrees passes", not problems, str(problems))

        # 2. A mutated prose count.
        write_fixture(root, prose=FIXTURE_BASELINES.replace("**455 passed", "**999 passed"))
        problems, _ = check_manifest(root, manifest)
        record(
            "a mutated prose count fails with both values",
            status.has_problem(problems, "TEST-BASELINES.md", "x86_64 (CI runner)", "'999'", "'455'"),
            str(problems),
        )

        # 3. The other direction: the source edited, the prose left alone.
        write_fixture(root, toml=FIXTURE_TOML.replace("passed = 455", "passed = 999"))
        problems, _ = check_manifest(root, manifest)
        record(
            "a mutated scripts/test-baselines.toml value fails against the prose",
            status.has_problem(problems, "TEST-BASELINES.md", "x86_64 (CI runner)", "'455'", "'999'"),
            str(problems),
        )

        # 4. A manifest with no rows must not pass vacuously.
        write_fixture(root, toml='schema = 1\nkind = "counts"\n')
        problems, _ = check_manifest(root, manifest)
        record(
            "a manifest with no [[counts]] rows fails",
            status.has_problem(problems, "no [[counts]] rows"),
            str(problems),
        )

        # 5. The plan ledger handed to this checker: one problem, named.
        write_fixture(root, toml=FIXTURE_TOML.replace('kind = "counts"', 'kind = "plan"'))
        problems, _ = check_manifest(root, manifest)
        record(
            "the plan ledger is refused by name, and alone",
            status.has_problem(problems, "kind is 'plan', not 'counts'") and len(problems) == 1,
            str(problems),
        )

        # 6. A box with no manifest rows (a fabricated box) must not pass vacuously.
        write_fixture(root)
        problems = check_live(manifest, root / "cargo.log", "GB10 sm_121", label)
        record(
            "a box with no manifest rows fails loudly",
            status.has_problem(problems, "no manifest rows for box"),
            str(problems),
        )

        # 7. A truncated log (no `Running tests/` block) must not pass.
        log = root / "cargo.log"
        log.write_text(FIXTURE_LOG_TRUNCATED, encoding="utf-8")
        problems = check_live(manifest, log, "x86_64 (CI runner)", label)
        record(
            "a truncated log fails on the missing integration block",
            status.has_problem(problems, "no `Running tests/` block"),
            str(problems),
        )

        # 8. The positive live case, then a live mismatch.
        write_fixture(root)
        log.write_text(FIXTURE_LOG, encoding="utf-8")
        problems = check_live(manifest, log, "x86_64 (CI runner)", label)
        record("a matching cargo log passes --check-live", not problems, str(problems))
        write_fixture(root, toml=FIXTURE_TOML.replace("passed = 455", "passed = 999"))
        problems = check_live(manifest, log, "x86_64 (CI runner)", label)
        record(
            "a live count mismatch fails",
            status.has_problem(problems, "passed", "999", "455"),
            str(problems),
        )

        # 9. #207: a recorded row whose CPU twin has moved prints a *non-failing*
        #    projection hint; a current one prints none. The hint is a projection,
        #    not a comparison, so the pair is the mutation evidence: same code,
        #    only the base value differs.
        moved = {
            "counts": [
                {"key": "cpu-unit", "box": "b", "passed": 462},
                {
                    "key": "cuda-unit",
                    "box": "g",
                    "passed": 548,
                    "projection_key": "cpu-unit",
                    "projection_box": "b",
                    "projection_base_passed": 455,
                },
            ]
        }
        moved_hints = projection_hints(moved)
        record(
            "a moved CPU twin prints a non-failing projection hint (#207)",
            len(moved_hints) == 1 and "548" in moved_hints[0] and "555" in moved_hints[0],
            str(moved_hints),
        )
        current = {
            "counts": [
                {"key": "cpu-unit", "box": "b", "passed": 462},
                {
                    "key": "cuda-unit",
                    "box": "g",
                    "passed": 555,
                    "projection_key": "cpu-unit",
                    "projection_box": "b",
                    "projection_base_passed": 462,
                },
            ]
        }
        record(
            "a recorded row refreshed together with its projection prints no hint",
            projection_hints(current) == [],
            str(projection_hints(current)),
        )

        # 10. `--help` states the honest limit.
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
    print(f"check_baselines selftest: {total - failures}/{total} cases pass")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_baselines.py",
        description=(
            "Check that the suite measurements in docs/TEST-BASELINES.md agree with "
            "scripts/test-baselines.toml, and optionally that the live row matched a "
            "cargo log. The plan's counters and next: sentence are the sibling ledger "
            "scripts/status.toml, checked by check_status.py. " + HELP_LIMIT
        ),
        epilog=(
            "Every row's file and regex live in scripts/test-baselines.toml, so this "
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
        help="run the built-in pass/fail cases in a temp tree and exit non-zero on a failure",
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
        if args.check or args.check_live or args.box or args.status:
            parser.error(
                "--selftest does not combine with --check/--check-live/--box/--status"
            )
        return run_selftest()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parent.parent
    manifest = Path(args.status).resolve() if args.status else root / MANIFEST
    label = status.label_of(root, manifest)

    if args.check_live:
        if not args.box:
            parser.error("--check-live requires --box")
        try:
            problems = check_live(manifest, Path(args.check_live), args.box, label)
        except OSError as exc:
            status.usage_error(f"check_baselines: cannot read {manifest}: {exc}")
        if problems:
            for problem in problems:
                print(f"check_baselines: {problem}", file=sys.stderr)
            print(
                f"check_baselines: --check-live failed for box {args.box!r} "
                f"({len(problems)} problem(s))",
                file=sys.stderr,
            )
            return 1
        print(f"check_baselines: the {args.box!r} rows match {args.check_live}")
        return 0

    if args.box:
        parser.error("--box is only meaningful with --check-live")

    try:
        problems, summary = check_manifest(root, manifest)
    except OSError as exc:
        status.usage_error(f"check_baselines: cannot read {manifest}: {exc}")
    except tomllib.TOMLDecodeError as exc:
        status.usage_error(f"check_baselines: {manifest} is not valid TOML: {exc}")

    if problems:
        for problem in problems:
            print(f"check_baselines: {problem}", file=sys.stderr)
        print(
            f"check_baselines: the prose disagrees with {manifest} "
            f"({len(problems)} problem(s))",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_baselines: prose agrees with {manifest} "
        f"({summary['counts']} count rows: {len(summary['live'])} live-checkable, "
        f"{summary['recorded']} recorded measurements)"
    )
    # #207: a recorded row whose CPU twin has moved since it was measured is
    # *printed*, never failed on — the relation is inexact.
    for hint in summary["hints"]:
        print(f"check_baselines: projection hint: {hint}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
