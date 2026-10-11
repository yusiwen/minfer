#!/usr/bin/env python3
"""Fail on a production `MINFER_*` env var that no document names.

#521 was filed because three knobs changed behaviour and were invisible: `-r`'s
warmup was described as "1 untimed warmup each" long after it became a
steady-clock budget (`MINFER_BENCH_WARMUP_MS`), and `MINFER_N_BATCH` — read and
printed at *every* server start, default 2048 — had no home outside the plan and
the roadmap. A knob a user cannot find is a knob that does not exist; the failure
is silent in both directions, because nothing enumerates what the engine reads.

So this checker enumerates it. It walks `src/**/*.rs`, takes every
`std::env::var("MINFER_…")` / `var_os` literal, and skips the test files by the
repository's own layout conventions (`tests.rs`, `<module>/tests/…`, `*_tests.rs`,
`*_test.rs` — the rule `check_source_layout.py` enforces). Each remaining name
must be **documented** — named in `docs/USAGE.md`, or in a live document under
`docs/` (plus `AGENTS.md` and `README.md`) — or listed in
`scripts/env-vars.toml` beside this checker with a reason.

What counts as *documented* excludes the dated records (`docs/adr/`,
`docs/cuda_optimization_steps/`, the execution plan, the roadmap, the baselines
and the three `*_OPTIMIZATIONS.md` pages): a name that appears only there is not
findable by a reader who wants to turn a knob, which is exactly why the ledger
exists. The alternative — requiring `docs/USAGE.md` for all 84 of today's names —
would need ~65 curated entries, because most are per-kernel A/B switches whose
home is a design doc; that stronger form is
[#531](https://github.com/yusiwen/minfer/issues/531).

The ledger is a ratchet: an entry whose name is documented fails (the exemption is
unnecessary), an entry whose name is no longer read fails, and a reason is
required. The checker's own cases cover both directions and the enumeration rule.

Usage:

    python3 scripts/check_env_vars.py [--root .] [--list]
    python3 scripts/check_env_vars.py --selftest

Exit codes: 0 clean, 1 a violation, 2 usage or I/O error. Pure and offline: stdlib
only, no build, no network.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
import tomllib
from pathlib import Path

LEDGER = "scripts/env-vars.toml"
USAGE = "docs/USAGE.md"
#: A `MINFER_…` name passed to `env::var`/`env::var_os`.
READ = re.compile(r'env::var(?:_os)?\(\s*"(MINFER_[A-Z0-9_]+)"')
#: The dated-record documents: history, not a place a reader looks up a knob. Each
#: entry is a path fragment; the ADR corpus and the campaign records are whole trees.
RECORDS = (
    "docs/adr/",
    "docs/cuda_optimization_steps/",
    "docs/ARCHITECTURE-EXECUTION-PLAN.md",  # the per-ticket history
    "docs/ARCHITECTURE-ROADMAP.md",  # the plan document
    "docs/TEST-BASELINES.md",  # the suite ledger
    "docs/CUDA_OPTIMIZATION.md",  # the campaign record
    "docs/METAL_OPTIMIZATIONS.md",  # the campaign record
    "docs/CPU_OPTIMIZATIONS.md",  # the campaign record
)
EXTRA_DOCS = ("AGENTS.md", "README.md")


def is_test_file(path: Path) -> bool:
    """The layout rule's test shapes: `tests.rs`, `<module>/tests/…`, `*_tests.rs`."""
    parts, name = path.parts[1:], path.name
    if name == "tests.rs" or name.endswith(("_tests.rs", "_test.rs")):
        return True
    if "tests" in parts:
        return True
    return any(segment.endswith(("_tests", "_test")) for segment in parts[:-1])


def production_reads(root: Path) -> dict[str, list[str]]:
    """`{name: [file, …]}` for every `MINFER_*` read outside the test files."""
    found: dict[str, list[str]] = {}
    for path in sorted((root / "src").rglob("*.rs")):
        if is_test_file(path.relative_to(root)):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in READ.finditer(text):
            found.setdefault(match.group(1), []).append(path.relative_to(root).as_posix())
    return found


def documents(root: Path) -> dict[str, str]:
    """The live documents: `{path: text}` for docs minus the records, plus the root pair."""
    docs: dict[str, str] = {}
    for path in sorted((root / "docs").rglob("*.md")):
        rel = path.relative_to(root).as_posix()
        if any(record in rel for record in RECORDS):
            continue
        docs[rel] = path.read_text(encoding="utf-8", errors="replace")
    for extra in EXTRA_DOCS:
        p = root / extra
        if p.is_file():
            docs[extra] = p.read_text(encoding="utf-8", errors="replace")
    return docs


def load_ledger(path: Path) -> dict[str, str]:
    """`{name: reason}` for every internal-only knob."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    out: dict[str, str] = {}
    for name, reason in data.get("internal", {}).items():
        if not reason or not str(reason).strip():
            raise ValueError(f"{path}: `{name}` needs a non-empty reason")
        out[name] = str(reason)
    return out


def check(root: Path, list_all: bool = False) -> int:
    """Print the verdict for `root`; return the exit code."""
    ledger_path = root / LEDGER
    if not ledger_path.is_file():
        print(f"check_env_vars: {ledger_path} is missing", file=sys.stderr)
        return 2
    try:
        ledger = load_ledger(ledger_path)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"check_env_vars: {exc}", file=sys.stderr)
        return 2
    if not (root / "src").is_dir():
        print(f"check_env_vars: no src/ under {root}", file=sys.stderr)
        return 2

    reads = production_reads(root)
    docs = documents(root)
    usage = docs.get(USAGE, "")
    problems: list[str] = []
    where: dict[str, str] = {}
    in_usage = 0
    documented = 0
    for name in sorted(reads):
        if name in usage:
            where[name] = USAGE
            in_usage += 1
            continue
        home = next((path for path, text in docs.items() if name in text), None)
        if home:
            where[name] = home
            documented += 1
            continue
        if name not in ledger:
            problems.append(
                f"{LEDGER}: `{name}` is read in production ({', '.join(sorted(set(reads[name])))}) "
                "but no live document names it — document it, or add it to the ledger with a reason"
            )
        else:
            where[name] = "(internal)"

    for name, reason in sorted(ledger.items()):
        if name in where and where[name] != "(internal)":
            problems.append(
                f"{LEDGER}: `{name}` is documented at {where[name]}; the exemption is unnecessary"
            )
        elif name not in reads:
            problems.append(f"{LEDGER}: `{name}` is no longer read anywhere in src/")

    if list_all:
        for name in sorted(reads):
            print(f"{name:34s} {where.get(name, '(undocumented)')}")

    for message in problems:
        print(f"ENV VAR  {message}", file=sys.stderr)
    if problems:
        print(
            f"check_env_vars: {len(problems)} problem(s) over {len(reads)} production "
            f"`MINFER_*` read(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_env_vars: {len(reads)} production `MINFER_*` read(s) — {in_usage} in {USAGE}, "
        f"{documented} in another live document, {len(ledger)} internal with a reason"
    )
    return 0


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool, str]] = []
    LEDGER_BODY = '[internal]\n{name} = "{reason}"\n'

    def tree(tmp: str, read: str, usage: str = "", design: str = "", record: str = "",
             ledger: str = "") -> Path:
        root = Path(tmp)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "engine.rs").write_text(f'let x = env::var("{read}");\n', encoding="utf-8")
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / USAGE).write_text(usage, encoding="utf-8")
        (root / "docs" / "design.md").write_text(design, encoding="utf-8")
        (root / "docs" / "adr").mkdir(exist_ok=True)
        (root / "docs" / "adr" / "0001-x.md").write_text(record, encoding="utf-8")
        path = root / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ledger, encoding="utf-8")
        return root

    def run(**kw) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            return check(tree(tmp, **kw))

    V = "MINFER_TEST_KNOB"
    cases.append(("a name in the manual passes", run(read=V, usage=f"`{V}=1` turns it on") == 0, ""))
    cases.append(("a name in a live design doc passes", run(read=V, design=f"`{V}` gates it") == 0, ""))
    cases.append(
        (
            "a name only in a record needs a ledger entry",
            run(read=V, record=f"`{V}` was measured") == 1,
            "",
        )
    )
    cases.append(
        (
            "that same name passes once ledgered",
            run(read=V, record=f"`{V}` was measured", ledger=LEDGER_BODY.format(name=V, reason="an A/B switch")) == 0,
            "",
        )
    )
    cases.append(
        (
            "a name no document names fails",
            run(read=V) == 1,
            "",
        )
    )
    cases.append(
        (
            "a ledger entry that is documented fails",
            run(read=V, usage=f"`{V}` is in the manual", ledger=LEDGER_BODY.format(name=V, reason="reason")) == 1,
            "",
        )
    )
    cases.append(
        (
            "a ledger entry that is no longer read fails",
            run(read="MINFER_OTHER", usage="`MINFER_OTHER` is in the manual", ledger=LEDGER_BODY.format(name=V, reason="reason")) == 1,
            "",
        )
    )
    cases.append(
        (
            "an empty reason is a usage error",
            run(read=V, ledger=f'[internal]\n{V} = ""\n') == 2,
            "",
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "src" / "tests.rs").write_text('env::var("MINFER_ONLY_IN_TESTS");\n', encoding="utf-8")
        (root / "docs").mkdir()
        (root / "docs" / "design.md").write_text("nothing", encoding="utf-8")
        ledger = root / LEDGER
        ledger.parent.mkdir(parents=True, exist_ok=True)
        ledger.write_text("[internal]\n", encoding="utf-8")
        cases.append(("a test-file read is not enumerated", check(root) == 0, ""))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "docs").mkdir()
        cases.append(("a missing ledger is an I/O error", check(root) == 2, ""))
    with tempfile.TemporaryDirectory() as tmp:
        root = tree(tmp, "MINFER_VAR_OS", design="`MINFER_VAR_OS` gates it")
        (root / "src" / "engine.rs").write_text('env::var_os("MINFER_VAR_OS");\n', encoding="utf-8")
        cases.append(("`env::var_os` counts", check(root) == 0, ""))
    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_env_vars --selftest: {len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="the repository root (default: the repo this script lives in)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="print every production `MINFER_*` read and where it is documented",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the built-in pass/fail cases in a temp tree and exit non-zero on a failure",
    )
    args = parser.parse_args(argv[1:])
    if args.selftest:
        return selftest()
    return check(args.root.resolve(), args.list)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
