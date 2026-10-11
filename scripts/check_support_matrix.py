#!/usr/bin/env python3
"""Keep `docs/SUPPORT-MATRIX.md`'s AVX2 column true to the kernels that exist.

The matrix is the one place a reader looks up "does this quant run here", and its
AVX2 (Advanced Vector Extensions 2) column rotted silently: it marked **Q5_0** as
SIMD while `src/quants/` had no AVX2 kernel for it — the row beside it (Q4_1) and
below it (Q5_1) were already ❌, and nothing re-read the cell (#519). A
capability matrix is exactly the kind of claim that goes stale without a test: it
is prose about code, and the code moved.

So this checker derives the AVX2 set from the source and compares it with the
column, in **both** directions:

- a quant row marked ✅ whose kernel does not exist fails (the Q5_0 case);
- a quant row marked ❌ while its kernel does exist fails (a stale ❌).

Derivation: a `dot_*` function counts as an AVX2 kernel when it lives in
`src/quants/avx2.rs` (the K-quant dots, named without a suffix) or when its name
ends in `_avx2` (the Q4_0/Q8_0 dots, which sit beside their NEON and scalar
arms). The quant type is read out of the symbol (`dot_q4_0_q8_0_avx2` → `q4_0`,
`dot_q4_k_q8_k` → `q4_k`), so a renamed kernel cannot silently drop out of the
derivation — it turns into a ❌-cell mismatch instead.

What it deliberately does not check: the `F16`/`BF16`/`F32` rows. Their AVX2 cell
means a vectorized *weight dot or cast* (`src/vec_ops/`), not a `dot_*` quant
kernel, so the derivation above does not describe them; the `CPU`, `CUDA GPU` and
`Metal GPU` columns are out of scope entirely — a device kernel cannot be proved
to exist from a symbol name, and the CUDA/Metal columns are gate-verified on
hardware instead (`docs/TEST-BASELINES.md`).

Usage:

    python3 scripts/check_support_matrix.py [--root .] [--list]
    python3 scripts/check_support_matrix.py --selftest

Exit codes: 0 clean, 1 a mismatch, 2 usage or I/O error. Pure and offline: stdlib
only, no build, no network.
"""

from __future__ import annotations

import argparse
import re
import sys
import tempfile
from pathlib import Path

MATRIX = "docs/SUPPORT-MATRIX.md"
#: The five columns a capability row must have, in order, after `Type|Bits|Block`.
COLUMNS = ("CPU", "AVX2", "CUDA GPU", "Metal GPU")
#: A `dot_*` definition, with the quant type it implements.
SYMBOL = re.compile(r"\bfn\s+(dot_(q[0-9]+_[0-9a-z]+)_q8_[0-9a-z_]+)")
#: A matrix row: `| **Q4_0** | 4 | 18 B / 32 val | ✅ | ✅ | ✅ | ✅ |`.
ROW = re.compile(r"^\|\s*\*\*([A-Za-z0-9_]+)\*\*\s*\|(.*)\|\s*$")
CHECK = "✅"
CROSS = "❌"


class Matrix:
    """The parsed table: `{type: [cell, …]}` plus the header's column order."""

    def __init__(self) -> None:
        self.header: list[str] = []
        self.rows: dict[str, list[str]] = {}


def parse_matrix(path: Path) -> Matrix:
    """Read the matrix's first table; raise `ValueError` on a shape it cannot read."""
    matrix = Matrix()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            if matrix.rows:
                break  # the table ended
            continue
        row = ROW.match(line)
        if row:
            cells = [c.strip() for c in row.group(2).split("|")]
            matrix.rows[row.group(1)] = cells
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if "AVX2" in cells:
            matrix.header = cells
    if not matrix.header:
        raise ValueError(f"{path}: no header row with an `AVX2` column")
    for column in COLUMNS:
        if column not in matrix.header:
            raise ValueError(f"{path}: header is missing the `{column}` column: {matrix.header}")
    if not matrix.rows:
        raise ValueError(f"{path}: no capability rows")
    return matrix


def avx2_kernels(root: Path) -> dict[str, str]:
    """`{quant_type: "file:line-ish symbol"}` for every AVX2 quant dot in `src/quants/`."""
    found: dict[str, str] = {}
    for path in sorted((root / "src" / "quants").glob("*.rs")):
        text = path.read_text(encoding="utf-8")
        for match in SYMBOL.finditer(text):
            name, quant = match.group(1), match.group(2)
            # The K-quant dots live in the avx2 module and carry no suffix; the
            # Q4_0/Q8_0 ones sit beside their NEON arm and say `_avx2`.
            if path.name != "avx2.rs" and not name.endswith("_avx2"):
                continue
            found.setdefault(quant, f"{path.name}:{name}")
    return found


def check(root: Path, list_all: bool = False) -> int:
    """Print the verdict for `root`; return the exit code."""
    path = root / MATRIX
    if not path.is_file():
        print(f"check_support_matrix: {path} is missing", file=sys.stderr)
        return 2
    try:
        matrix = parse_matrix(path)
        kernels = avx2_kernels(root)
    except (ValueError, OSError) as exc:
        print(f"check_support_matrix: {exc}", file=sys.stderr)
        return 2

    index = matrix.header.index("AVX2")
    # The parsed row excludes the `Type` cell (it is the regex's group 1), so a
    # header position maps to `cells` at one less.
    offset = 1
    problems: list[str] = []
    claimed: set[str] = set()
    quant_rows = 0
    for name, cells in matrix.rows.items():
        quant = name.lower()
        if not re.fullmatch(r"q[0-9]+_[0-9a-z]+", quant):
            if list_all:
                print(f"SKIP  {name}: not a quant row (its AVX2 cell means a vectorized dot/cast)")
            continue
        quant_rows += 1
        if len(cells) <= index - 1:
            problems.append(f"{MATRIX}: row `{name}` has {len(cells)} cells, the table needs more")
            continue
        raw = cells[index - offset]
        # A cell may carry a footnote marker (`✅⁹`); the mark is its first character.
        cell = raw[:1] if raw[:1] in (CHECK, CROSS) else raw
        has_kernel = quant in kernels
        if has_kernel:
            claimed.add(quant)
        if list_all:
            print(
                f"{'OK ' if (cell == CHECK) == has_kernel else 'BAD'}  {name}: "
                f"AVX2={raw} kernel={kernels.get(quant, '-')}"
            )
        if cell == CHECK and not has_kernel:
            problems.append(
                f"{MATRIX}: `{name}` claims AVX2 but `src/quants/` has no `dot_*` kernel for it"
            )
        elif cell == CROSS and has_kernel:
            problems.append(
                f"{MATRIX}: `{name}` says no AVX2, but {kernels[quant]} exists"
            )
        elif cell not in (CHECK, CROSS):
            problems.append(f"{MATRIX}: `{name}` has an unreadable AVX2 cell `{raw}`")

    for quant, where in sorted(kernels.items()):
        if quant not in claimed:
            problems.append(
                f"src/quants/{where} has no row in {MATRIX} claiming AVX2 for `{quant}`"
            )

    for message in problems:
        print(f"MATRIX MISMATCH  {message}", file=sys.stderr)
    if problems:
        print(
            f"check_support_matrix: {len(problems)} mismatch(es) over {quant_rows} quant rows "
            f"and {len(kernels)} AVX2 kernel(s)",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_support_matrix: {quant_rows} quant rows agree with the {len(kernels)} AVX2 "
        f"kernel(s) in src/quants/ ({', '.join(sorted(kernels))})"
    )
    return 0


def selftest() -> int:
    """The checker's own pass/fail cases, in a temp tree."""
    cases: list[tuple[str, bool, str]] = []
    HEADER = "| Type | Bits | Block | CPU | AVX2 | CUDA GPU | Metal GPU |\n"
    RULE = "|------|------|-------|:---:|:----:|:--------:|:---------:|\n"

    def row(name: str, avx2: str) -> str:
        return f"| **{name}** | 4 | 18 B / 32 val | ✅ | {avx2} | ✅ | ✅ |\n"

    def run(rows: str, source: str, filename: str = "avx2.rs") -> int:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "docs").mkdir(parents=True, exist_ok=True)
            (root / MATRIX).write_text(HEADER + RULE + rows, encoding="utf-8")
            quants = root / "src" / "quants"
            quants.mkdir(parents=True, exist_ok=True)
            (quants / filename).write_text(source, encoding="utf-8")
            return check(root)

    KQ = "pub(super) unsafe fn dot_q4_k_q8_k(a: &[u8]) -> f32 { 0.0 }\n"
    Q4 = "unsafe fn dot_q4_0_q8_0_avx2(a: &[u8]) -> f32 { 0.0 }\n"

    cases.append(
        (
            "a row whose kernel exists passes",
            run(row("Q4_K", "✅"), KQ) == 0,
            "",
        )
    )
    cases.append(
        (
            "a row claiming AVX2 with no kernel fails",
            run(row("Q5_0", "✅"), KQ) == 1,
            "",
        )
    )
    cases.append(
        (
            "a row denying AVX2 while the kernel exists fails",
            run(row("Q4_K", "❌"), KQ) == 1,
            "",
        )
    )
    cases.append(
        (
            "a suffix-named kernel outside avx2.rs counts",
            run(row("Q4_0", "✅"), Q4, "dot_q4_0.rs") == 0,
            "",
        )
    )
    cases.append(
        (
            "a suffix-less kernel outside avx2.rs does not count",
            run(row("Q4_0", "❌"), Q4.replace("_avx2", "_neon"), "dot_q4_0.rs") == 0,
            "",
        )
    )
    cases.append(
        (
            "a kernel with no row claiming it fails",
            run(row("Q8_0", "❌"), Q4) == 1,
            "",
        )
    )
    cases.append(
        (
            "an unreadable cell fails",
            run(row("Q4_K", "maybe"), KQ) == 1,
            "",
        )
    )
    cases.append(
        (
            "a non-quant row is skipped",
            run(row("Q4_K", "✅") + row("F16", "✅"), KQ) == 0,
            "",
        )
    )
    cases.append(
        (
            "a footnote on a ✅ cell is still a tick",
            run(row("Q4_K", "✅⁹"), KQ) == 0,
            "",
        )
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases.append(("a tree without the matrix is an I/O error", check(root) == 2, ""))
        (root / "docs").mkdir()
        (root / MATRIX).write_text("| Type | Bits | Block | CPU |\n", encoding="utf-8")
        cases.append(
            (
                "a matrix without the AVX2 column is a usage error",
                check(root) == 2,
                "",
            )
        )
    failed = 0
    for name, ok, note in cases:
        print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({note})" if note and not ok else ""))
        failed += 0 if ok else 1
    print(f"check_support_matrix --selftest: {len(cases) - failed}/{len(cases)} cases pass")
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
        help="print every quant row's verdict and the kernel it derives",
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
