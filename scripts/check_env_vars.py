#!/usr/bin/env python3
"""Require a documentation home for every production `MINFER_*` env var.

#521 was filed because three knobs changed behaviour and were invisible, and the gate it
added settled for the *weak* form: a name that some live document mentions somewhere
passes. This is the strong form that ticket left open (#531): every production read has a
**named, verified** home.

`scripts/env-vars.toml` is the index. Each knob is in exactly one table:

- `[documented] name = "<file>"` — the checker verifies the file exists, is a *live*
  document (not one of the dated records listed below; those are history, not a place a
  reader looks up a knob), and actually names the knob. So "this file documents it" is
  checked, not asserted.
- `[internal] name = "<reason>"` — no live document names it, and the reason says what it
  does and where its only record is. The checker fails when a live document *does* name a
  knob the ledger calls internal: the exemption has to stay necessary.

The enumeration is the read itself: `std::env::var("MINFER_…")` / `var_os` in
`src/**/*.rs`, skipping the test files by the layout rule `check_source_layout.py`
already holds (`tests.rs`, `<module>/tests/…`, `*_tests.rs`, `*_test.rs`). Both tables
ratchet — an entry whose knob is no longer read fails, a `[documented]` home that does not
contain the name fails — and a knob in neither table fails with the file that reads it,
which is the #521 class exactly (`MINFER_N_BATCH` was read and printed at every server
start while nothing a user reads named it).

The ledger also states its own totals (`# Measured <date>: N production reads — D
documented, I internal`), and the checker compares that line with what it just counted
rather than trusting it — the `check_baselines.py` pattern, where the prose carries the
numbers and the code is the source of truth. The line is *required*: a missing one fails,
so the comparison cannot be dropped by deleting a comment. It existed because the line
had trailed the code by one knob from #531 to #628 (the totals were the only place the
counts lived, and nothing recomputed them).

Usage:

    python3 scripts/check_env_vars.py [--root .] [--list]
    python3 scripts/check_env_vars.py --selftest

Exit codes: 0 clean, 1 a violation, 2 usage or I/O error. Pure and offline: stdlib only,
no build, no network.
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
#: The ledger's own totals line. Its numbers are prose; the code is the source of truth,
#: so they are compared (the `check_baselines.py` pattern), and a missing line fails.
TOTALS = re.compile(
    r"^#\s*Measured\s+(?P<date>\d{4}-\d{2}-\d{2}):\s*"
    r"(?P<reads>\d+)\s+production\s+reads?\s*—\s*"
    r"(?P<documented>\d+)\s+documented,\s*(?P<internal>\d+)\s+internal\b",
    re.MULTILINE,
)
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


def load_ledger(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """`({name: documentation home}, {name: reason})` from the ledger's two tables."""
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    documented: dict[str, str] = {}
    internal: dict[str, str] = {}
    for name, home in data.get("documented", {}).items():
        if not home or not str(home).strip():
            raise ValueError(f"{path}: `{name}` needs a documentation home")
        documented[name] = str(home)
    for name, reason in data.get("internal", {}).items():
        if not reason or not str(reason).strip():
            raise ValueError(f"{path}: `{name}` needs a non-empty reason")
        internal[name] = str(reason)
    both = sorted(set(documented) & set(internal))
    if both:
        raise ValueError(f"{path}: {', '.join(both)} is in both tables")
    return documented, internal


def check(root: Path, list_all: bool = False) -> int:
    """Print the verdict for `root`; return the exit code."""
    ledger_path = root / LEDGER
    if not ledger_path.is_file():
        print(f"check_env_vars: {ledger_path} is missing", file=sys.stderr)
        return 2
    try:
        documented, internal = load_ledger(ledger_path)
    except (ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"check_env_vars: {exc}", file=sys.stderr)
        return 2
    if not (root / "src").is_dir():
        print(f"check_env_vars: no src/ under {root}", file=sys.stderr)
        return 2

    reads = production_reads(root)
    live = documents(root)
    problems: list[str] = []
    # The counts the ledger's prose line must agree with: per *read* knob, so a stale
    # table entry cannot make the header look right while a knob has no home.
    counted = (
        len(reads),
        sum(1 for name in reads if name in documented),
        sum(1 for name in reads if name in internal),
    )

    # The ledger's totals line: compared, not trusted (the `check_baselines.py` pattern).
    # Required, so deleting the comment cannot delete the check (#628).
    ledger_text = ledger_path.read_text(encoding="utf-8", errors="replace")
    totals = TOTALS.search(ledger_text)
    if totals is None:
        plural = "" if counted[0] == 1 else "s"
        problems.append(
            f"{LEDGER}: no totals line — state what the checker counted, on a comment line: "
            f"`# Measured <date>: {counted[0]} production read{plural} — {counted[1]} "
            f"documented, {counted[2]} internal.`"
        )
    else:
        stated = (int(totals["reads"]), int(totals["documented"]), int(totals["internal"]))
        if stated != counted:
            line = ledger_text.count("\n", 0, totals.start()) + 1
            problems.append(
                f"{LEDGER}:{line}: the totals line says {stated[0]} production reads — "
                f"{stated[1]} documented, {stated[2]} internal, but the code reads "
                f"{counted[0]} — {counted[1]} documented, {counted[2]} internal; update "
                "the line"
            )

    # Every knob's home, and the file that reads it — the two facts a reader wants.
    where: dict[str, str] = {}
    for name in sorted(reads):
        source = ", ".join(sorted(set(reads[name])))
        if name in documented:
            home = documented[name]
            where[name] = home
            path = root / home
            if any(record in home for record in RECORDS):
                problems.append(
                    f"{LEDGER}: `{name}` names `{home}`, which is a dated record, not a "
                    "document a reader looks up a knob in"
                )
            elif not path.is_file():
                problems.append(f"{LEDGER}: `{name}` names `{home}`, which does not exist")
            elif name not in path.read_text(encoding="utf-8", errors="replace"):
                problems.append(
                    f"{LEDGER}: `{name}` claims `{home}`, but that file does not name it"
                )
        elif name in internal:
            where[name] = "(internal)"
            home_names = [d for d, text in live.items() if name in text]
            if home_names:
                problems.append(
                    f"{LEDGER}: `{name}` is marked internal, but {home_names[0]} documents it "
                    "— move it to [documented]"
                )
        else:
            problems.append(
                f"{LEDGER}: `{name}` is read in production ({source}) and has no entry — "
                "add it to [documented] with the file that documents it, or to [internal] "
                "with a reason"
            )

    for name in sorted(set(documented) | set(internal)):
        if name not in reads:
            problems.append(f"{LEDGER}: `{name}` is no longer read anywhere in src/")

    if list_all:
        for name in sorted(reads):
            print(f"{name:34s} {where.get(name, '(no entry)')}")

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
        f"check_env_vars: {counted[0]} production `MINFER_*` read(s), every one with a home "
        f"— {counted[1]} in a named document, {counted[2]} internal with a reason"
    )
    return 0


def selftest() -> int:
    """The index's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool]] = []
    V = "MINFER_TEST_KNOB"
    DESIGN = "The knob `MINFER_TEST_KNOB` gates the thing."

    def build(tmp: str, read: str, ledger: str, design: str = "", usage: str = "",
              record: str = "", files: dict[str, str] | None = None) -> Path:
        root = Path(tmp)
        (root / "src").mkdir(parents=True, exist_ok=True)
        (root / "src" / "engine.rs").write_text(f'let x = env::var("{read}");\n', encoding="utf-8")
        (root / "docs").mkdir(parents=True, exist_ok=True)
        (root / USAGE).write_text(usage, encoding="utf-8")
        (root / "docs" / "design.md").write_text(design, encoding="utf-8")
        (root / "docs" / "adr").mkdir(exist_ok=True)
        (root / "docs" / "adr" / "0001-x.md").write_text(record, encoding="utf-8")
        for rel, text in (files or {}).items():
            path = root / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        path = root / LEDGER
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(ledger, encoding="utf-8")
        return root

    def run(ledger: str, header_text: str | None = None, **kw) -> int:
        """Run the checker over a temp tree; the totals line defaults to the right one."""
        read = kw.get("read", "")
        data = tomllib.loads(ledger) if ledger.strip() else {}
        if header_text is None:
            header_text = header(
                1,
                1 if read in data.get("documented", {}) else 0,
                1 if read in data.get("internal", {}) else 0,
            )
        with tempfile.TemporaryDirectory() as tmp:
            return check(build(tmp, ledger=header_text + ledger, **kw))

    def header(reads: int, documented: int, internal: int) -> str:
        """The totals line a tree with these counts must carry."""
        plural = "" if reads == 1 else "s"
        return (
            f"# Measured 2026-10-20: {reads} production read{plural} — "
            f"{documented} documented, {internal} internal.\n\n"
        )

    def tab(kind: str, name: str, value: str) -> str:
        return f'[{kind}]\n{name} = "{value}"\n'

    cases.append(
        ("a documented knob passes", run(tab("documented", V, "docs/design.md"), read=V, design=DESIGN) == 0)
    )
    cases.append(
        ("the manual is a valid home", run(tab("documented", V, USAGE), read=V, usage=DESIGN) == 0)
    )
    cases.append(
        (
            "a home that does not name the knob fails",
            run(tab("documented", V, "docs/design.md"), read=V, design="something else") == 1,
        )
    )
    cases.append(
        (
            "a home that does not exist fails",
            run(tab("documented", V, "docs/missing.md"), read=V) == 1,
        )
    )
    cases.append(
        (
            "a dated record is not a home",
            run(tab("documented", V, "docs/adr/0001-x.md"), read=V, record=DESIGN) == 1,
        )
    )
    cases.append(
        ("an internal knob with a reason passes", run(tab("internal", V, "an A/B switch"), read=V) == 0)
    )
    cases.append(
        (
            "an internal knob whose name a live doc has fails",
            run(tab("internal", V, "reason"), read=V, design=DESIGN) == 1,
        )
    )
    cases.append(
        (
            "an internal knob mentioned only by a record passes",
            run(tab("internal", V, "reason"), read=V, record=DESIGN) == 0,
        )
    )
    cases.append(("a knob with no entry fails", run("", read=V) == 1))
    cases.append(
        (
            "a knob in both tables is a usage error",
            run(tab("documented", V, "docs/design.md") + tab("internal", V, "x"), read=V, design=DESIGN) == 2,
        )
    )
    cases.append(
        (
            "an entry whose knob is no longer read fails",
            run(tab("internal", "MINFER_GONE", "reason"), read="MINFER_OTHER") == 1,
        )
    )
    cases.append(("an empty reason is a usage error", run(tab("internal", V, ""), read=V) == 2))
    cases.append(("an empty home is a usage error", run(tab("documented", V, ""), read=V) == 2))
    cases.append(
        (
            "a totals line that matches the counts passes",
            run(tab("documented", V, "docs/design.md"), read=V, design=DESIGN) == 0,
        )
    )
    cases.append(
        (
            "a totals line one read off fails",
            run(
                tab("documented", V, "docs/design.md"),
                header_text=header(2, 1, 0),
                read=V,
                design=DESIGN,
            )
            == 1,
        )
    )
    cases.append(
        (
            "a ledger without a totals line fails",
            run(
                tab("documented", V, "docs/design.md"),
                header_text="",
                read=V,
                design=DESIGN,
            )
            == 1,
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, read="MINFER_ONLY_IN_TESTS", ledger="")
        (root / "src" / "engine.rs").write_text("// no reads here\n", encoding="utf-8")
        (root / "src" / "tests.rs").write_text('env::var("MINFER_ONLY_IN_TESTS");\n', encoding="utf-8")
        (root / LEDGER).write_text(header(0, 0, 0), encoding="utf-8")
        cases.append(("a test-file read is not enumerated", check(root) == 0))
    with tempfile.TemporaryDirectory() as tmp:
        root = build(tmp, read="MINFER_OTHER", ledger="")
        (root / "src" / "engine.rs").write_text('env::var_os("MINFER_VAR_OS");\n', encoding="utf-8")
        (root / LEDGER).write_text(header(1, 0, 0), encoding="utf-8")
        cases.append(("`env::var_os` needs an entry too", check(root) == 1))
        (root / LEDGER).write_text(
            header(1, 0, 1) + tab("internal", "MINFER_VAR_OS", "reason"), encoding="utf-8"
        )
        cases.append(("and passes once the entry exists", check(root) == 0))
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src").mkdir()
        (root / "docs").mkdir()
        cases.append(("a missing ledger is an I/O error", check(root) == 2))

    failed = sum(1 for _, ok in cases if not ok)
    for name, ok in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
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
