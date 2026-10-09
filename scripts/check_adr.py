#!/usr/bin/env python3
"""Check the shape of `docs/adr/` — the decisions' one home, and a frozen one.

An ADR records a decision, the alternatives it beat, its consequences and the date
it was taken. The whole value of the corpus rests on two properties this checker
enforces mechanically, because both are cheap to break by hand and expensive to
notice:

1. **Numbering is dense and stable.** `NNNN-slug.md`, four digits, numbering from
   `0001` with no gaps and no reuse. A number is a citation target that other
   documents hold, so it is never renumbered and never reissued.
2. **`Superseded` is a two-way, forward-only link.** An ADR whose `Status` says
   `Superseded by ADR-0015` requires `0015` to exist, to have a *higher* number,
   and to name it back in `Supersedes:`. That back-link is what makes "the
   decision changed" visible from the newer record as well as the older one.

The boundary the corpus itself rests on is not checkable here and is stated in
`docs/adr/README.md`: an ADR holds the decision and its rationale, never the
current contract and never a measurement. Those live in the design docs (mutable)
and in `docs/status.toml` (the machine source). A checker can force a shape, not a
habit.

Usage:

    python3 scripts/check_adr.py [--root .]
    python3 scripts/check_adr.py --selftest

Exit codes: 0 clean, 1 violations, 2 usage or I/O error. Pure and offline:
stdlib only, no build, no network.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

NAME = re.compile(r"^(\d{4})-([a-z0-9][a-z0-9-]*)\.md$")
TITLE = re.compile(r"^#\s+(\d{4})\.\s+\S")
FIELD = re.compile(r"^-\s+([A-Za-z][A-Za-z ]*):\s*(.*?)\s*$")
STATUS = re.compile(r"^(Proposed|Accepted|Rejected|Superseded by ADR-(\d{4}))$")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REF = re.compile(r"ADR-(\d{4})")
REQUIRED_SECTIONS = ("## Context", "## Decision", "## Alternatives considered", "## Consequences")


class Adr:
    def __init__(self, path: Path, text: str) -> None:
        self.path = path
        self.text = text
        self.name = path.name
        m = NAME.match(self.name)
        self.file_number = m.group(1) if m else None
        self.title_number = None
        tm = TITLE.match(text)
        if tm:
            self.title_number = tm.group(1)
        self.fields: dict[str, str] = {}
        for line in text.split("\n"):
            fm = FIELD.match(line)
            if fm:
                self.fields.setdefault(fm.group(1).strip().lower(), fm.group(2))
        self.status = self.fields.get("status", "")
        self.date = self.fields.get("date", "")
        sm = STATUS.match(self.status)
        self.superseded_by = sm.group(2) if (sm and sm.group(2)) else None
        self.supersedes = REF.findall(self.fields.get("supersedes", ""))


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def check(root: Path) -> int:
    adr_dir = root / "docs" / "adr"
    if not adr_dir.is_dir():
        print(f"check_adr: no {adr_dir} directory", file=sys.stderr)
        return 2
    problems: list[str] = []

    files = sorted(p for p in adr_dir.iterdir() if p.is_file() and p.suffix == ".md")
    adrs: list[Adr] = []
    for p in files:
        if p.name == "README.md":
            continue
        adr = Adr(p, read(p))
        if not adr.file_number:
            problems.append(
                f"{p.relative_to(root)}: filename must be NNNN-slug.md "
                f"(four digits, lower-case slug), e.g. 0007-kv-format-is-a-gate.md"
            )
            continue
        adrs.append(adr)
    adrs.sort(key=lambda a: a.file_number or "")

    seen: dict[str, Adr] = {}
    for adr in adrs:
        number = adr.file_number or ""
        rel = adr.path.relative_to(root)
        if number in seen:
            problems.append(f"{rel}: number {number} is already used by {seen[number].name}")
        seen[number] = adr
        if adr.title_number != number:
            problems.append(
                f"{rel}: title must be `# {number}. <title>` — the heading says "
                f"{adr.title_number!r}, the filename says {number!r}"
            )
        if not adr.status:
            problems.append(f"{rel}: missing `- Status:` (Proposed|Accepted|Rejected|Superseded by ADR-NNNN)")
        elif not STATUS.match(adr.status):
            problems.append(f"{rel}: Status {adr.status!r} is not one of the four allowed forms")
        if not adr.date:
            problems.append(f"{rel}: missing `- Date: YYYY-MM-DD`")
        elif not DATE.match(adr.date):
            problems.append(f"{rel}: Date {adr.date!r} is not YYYY-MM-DD")
        for section in REQUIRED_SECTIONS:
            if not any(line.startswith(section) for line in adr.text.split("\n")):
                problems.append(f"{rel}: missing required section {section!r}")

    # numbering is dense from 0001
    expected = [f"{i:04d}" for i in range(1, len(adrs) + 1)]
    actual = [a.file_number for a in adrs]
    if actual != expected:
        problems.append(
            f"docs/adr: numbering must be dense from 0001 — found {', '.join(actual) or 'none'}, "
            f"expected {', '.join(expected)}"
        )

    # supersession is forward-only and two-way
    for adr in adrs:
        rel = adr.path.relative_to(root)
        if adr.superseded_by:
            target = seen.get(adr.superseded_by)
            if target is None:
                problems.append(f"{rel}: Superseded by ADR-{adr.superseded_by}, which does not exist")
            else:
                if adr.superseded_by <= (adr.file_number or ""):
                    problems.append(
                        f"{rel}: Superseded by ADR-{adr.superseded_by} must name a *higher* "
                        f"number than its own {adr.file_number}"
                    )
                if (adr.file_number or "") not in target.supersedes:
                    problems.append(
                        f"{rel}: ADR-{adr.superseded_by} must name it back with "
                        f"`- Supersedes: ADR-{adr.file_number}` — supersession is visible from both ends"
                    )
        for sup in adr.supersedes:
            other = seen.get(sup)
            if other is None:
                problems.append(f"{rel}: Supersedes ADR-{sup}, which does not exist")
                continue
            if sup >= (adr.file_number or ""):
                problems.append(f"{rel}: Supersedes ADR-{sup} must be a *lower* number")
            if other.superseded_by != adr.file_number:
                problems.append(
                    f"{rel}: Supersedes ADR-{sup}, but that ADR's Status does not say "
                    f"`Superseded by ADR-{adr.file_number}`"
                )

    # the index lists every ADR, and renumbers nothing that does not exist
    index = adr_dir / "README.md"
    if not index.is_file():
        problems.append("docs/adr/README.md: missing — the corpus needs an index")
    else:
        itext = read(index)
        for adr in adrs:
            if adr.name not in itext:
                problems.append(f"docs/adr/README.md: does not list {adr.name}")
        for ref in set(REF.findall(itext)):
            if ref not in seen:
                problems.append(f"docs/adr/README.md: names ADR-{ref}, which does not exist")
        summary = root / "docs" / "SUMMARY.md"
        if not summary.is_file():
            problems.append("docs/SUMMARY.md: missing, so the index cannot be a chapter")
        elif "./adr/README.md" not in read(summary):
            problems.append(
                "docs/SUMMARY.md: does not reference `./adr/README.md` — an index outside the "
                "book is an index nobody reads"
            )

    if problems:
        for p in problems:
            print(f"check_adr: {p}")
        print(f"check_adr: {len(problems)} problem(s) across {len(adrs)} ADR(s)")
        return 1
    print(f"check_adr: {len(adrs)} ADR(s) numbered 0001-{adrs[-1].file_number if adrs else '????'}, "
          f"indexed, supersession consistent")
    return 0


FIXTURE_README = """# Architecture Decision Records

| # | Date | Decision | Status |
|---|---|---|---|
| [0001](./0001-first-decision.md) | 2026-01-01 | The first decision | Accepted |
"""

FIXTURE_ADR = """# 0001. The first decision

- Status: Accepted
- Date: 2026-01-01

## Context

ctx

## Decision

decided

## Alternatives considered

none recorded

## Consequences

cost
"""


def _write(root: Path, name: str, text: str) -> None:
    (root / "docs" / "adr").mkdir(parents=True, exist_ok=True)
    (root / "docs" / "adr" / name).write_text(text, encoding="utf-8")


def _fixture(root: Path, adrs: dict[str, str], readme: str = FIXTURE_README,
             summary: str = "# Summary\n\n- [ADR index](./adr/README.md)\n") -> None:
    for name, text in adrs.items():
        _write(root, name, text)
    _write(root, "README.md", readme)
    (root / "docs" / "SUMMARY.md").write_text(summary, encoding="utf-8")


def selftest() -> int:
    cases: list[tuple[str, bool, str]] = []

    def expect(name: str, ok: bool, note: str = "") -> None:
        cases.append((name, ok, note))

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        _fixture(root, {"0001-first-decision.md": FIXTURE_ADR})
        expect("a well-formed single-ADR corpus passes", check(root) == 0)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        _fixture(root, {"1-first-decision.md": FIXTURE_ADR})
        expect("a filename without four digits fails", check(root) == 1)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        second = FIXTURE_ADR.replace("# 0001.", "# 0003.").replace("0001", "0003")
        _fixture(root, {"0001-first-decision.md": FIXTURE_ADR, "0003-third.md": second},
                 readme=FIXTURE_README + "| [0003](./0003-third.md) | 2026-01-02 | Third | Accepted |\n")
        expect("a gap in the numbering fails", check(root) == 1)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        bad = FIXTURE_ADR.replace("- Status: Accepted", "- Status: Maybe")
        _fixture(root, {"0001-first-decision.md": bad})
        expect("an unknown Status fails", check(root) == 1)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        old = FIXTURE_ADR.replace("- Status: Accepted", "- Status: Superseded by ADR-0002")
        new = (FIXTURE_ADR.replace("# 0001.", "# 0002.").replace("0001-first", "0002-second")
               .replace("- Status: Accepted", "- Status: Accepted\n- Supersedes: ADR-0001"))
        _fixture(root, {"0001-first-decision.md": old, "0002-second.md": new},
                 readme=FIXTURE_README + "| [0002](./0002-second.md) | 2026-01-02 | Second | Accepted |\n")
        expect("a two-way Superseded link passes", check(root) == 0)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        # the newer ADR forgets to name the older one back
        old = FIXTURE_ADR.replace("- Status: Accepted", "- Status: Superseded by ADR-0002")
        new = FIXTURE_ADR.replace("# 0001.", "# 0002.").replace("0001-first", "0002-second")
        _fixture(root, {"0001-first-decision.md": old, "0002-second.md": new},
                 readme=FIXTURE_README + "| [0002](./0002-second.md) | 2026-01-02 | Second | Accepted |\n")
        expect("a one-way Superseded link fails", check(root) == 1)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        _fixture(root, {"0001-first-decision.md": FIXTURE_ADR}, readme="# Architecture Decision Records\n")
        expect("a missing index entry fails", check(root) == 1)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        _fixture(root, {"0001-first-decision.md": FIXTURE_ADR}, summary="# Summary\n\n- [Intro](./intro.md)\n")
        expect("an index outside SUMMARY.md fails", check(root) == 1)

    with tempfile.TemporaryDirectory(prefix="check_adr_selftest_") as tmp:
        root = Path(tmp)
        _fixture(root, {"0001-first-decision.md": FIXTURE_ADR.replace("## Consequences", "## Notes")})
        expect("a missing required section fails", check(root) == 1)

    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_adr --selftest: {len(cases) - failed}/{len(cases)} cases pass")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Check the shape of docs/adr/.")
    ap.add_argument("--root", default=".", help="repository root (default: .)")
    ap.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    return check(Path(args.root))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
