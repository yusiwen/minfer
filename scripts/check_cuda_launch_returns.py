#!/usr/bin/env python3
"""Audit every CUDA ``<<<>>>`` in ``src/cuda_kernels.cu`` for a named launch read.

Issue #162, sibling of the source-level checks in ``check_docs_links.py`` and
``check_status.py``. A ``<<<>>>`` has no return value: its error is read with the
*immediately following* ``cudaGetLastError``. Before #162, 104 sites in this file
enqueued a kernel and never read the error, so a failed launch latched and
surfaced later at ``CudaState::sync`` with no indication of which launch produced
it.

This check requires, for **every** ``<<<`` occurrence in the file:

1. a ``minfer_launch_prelude("<site>", ...)`` call at the same site, before the
   launch — it reports a pre-existing latch as *not* this launch's;
2. a ``minfer_launch_ok("<site>", ...)`` (required) or
   ``minfer_launch_ok_opt("<site>", ...)`` (documented fallback) call after the
   launch's statement, naming the **same** ``<site>``; and
3. a ``<site>`` that starts with ``launch:``.

The site token is the join key: a read that names a different site, or a missing
read, is reported by line and by what was expected.

**Honest scope.** This is a source audit: it proves the read is *written*, not
that it runs, and it cannot see whether a launch produced the value the code
asserts. It also cannot verify the *injection lever* (the ``minfer_launch_block``
wrap) — that is what the device gate
``cuda_issue162_every_launch_site_names_itself_and_leaves_no_latch`` checks by
actually driving each armed site into a real failing launch. The two halves are
deliberately separate: the audit catches a site that lost its read, the device
gate catches a site that cannot be made to fail.

Exit status: 0 when every site passes (the offending list is empty), 1 otherwise.
``--list`` prints ``line<TAB>owner<TAB>site<TAB>kernel`` for every site;
``--fixture PATH`` writes that list and ``--check-fixture PATH`` compares it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = REPO / "src" / "cuda_kernels.cu"


def code_only(text: str):
    """``(code, blanked)`` with comments and string/char bodies handled.

    ``code`` has comments blanked but string **contents preserved**, so the
    ``"launch:…"`` site tokens (and ``extern "C"``) stay readable. ``blanked``
    additionally blanks string/char bodies, and is what site *detection* uses: a
    ``<<<`` inside a literal is not a launch. Offsets and line breaks are
    identical in both, so line/column arithmetic is exact.

    ``<<<>>>`` appears in ordinary comments in this file (the #147/#162 notes), and
    a source audit fooled by a comment would pass on a file with no read at all.
    """
    code = list(text)
    blanked = list(text)
    i, n = 0, len(text)
    state = "code"
    while i < n:
        c = text[i]
        nxt = text[i + 1] if i + 1 < n else ""
        if state == "code":
            if c == "/" and nxt == "/":
                state = "line_comment"
                code[i] = blanked[i] = " "
                code[i + 1] = blanked[i + 1] = " "
                i += 2
                continue
            if c == "/" and nxt == "*":
                state = "block_comment"
                code[i] = blanked[i] = " "
                code[i + 1] = blanked[i + 1] = " "
                i += 2
                continue
            if c == '"':
                state = "string"
                i += 1
                continue
            if c == "'":
                state = "char"
                i += 1
                continue
        elif state == "line_comment":
            if c == "\n":
                state = "code"
            else:
                code[i] = blanked[i] = " "
        elif state == "block_comment":
            if c == "*" and nxt == "/":
                code[i] = blanked[i] = " "
                code[i + 1] = blanked[i + 1] = " "
                i += 2
                state = "code"
                continue
            if c != "\n":
                code[i] = blanked[i] = " "
        elif state in ("string", "char"):
            if c == "\\":
                blanked[i] = " "
                if nxt and nxt != "\n":
                    blanked[i + 1] = " "
                i += 2
                continue
            if (state == "string" and c == '"') or (state == "char" and c == "'"):
                state = "code"
            elif c != "\n":
                blanked[i] = " "
        i += 1
    return "".join(code), "".join(blanked)


FUNC_RE = re.compile(
    r'^(?:static\s+)?(?:extern\s+"C"\s+)?(?:template\s*<[^>]*>\s*)?'
    r"(?:void|int|bool|size_t)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
)
MACRO_RE = re.compile(r"^\s*#define\s+([A-Za-z_][A-Za-z0-9_]*)")


def owners(code_lines: list[str]) -> list[tuple[int, int, str]]:
    """``(start_line, end_line, name)`` for every function and macro body."""
    found: list[tuple[int, int, str]] = []
    i = 0
    n = len(code_lines)
    while i < n:
        line = code_lines[i]
        m = MACRO_RE.match(line)
        if m:
            # the body runs while each line's code text ends with a backslash
            j = i
            while j < n and code_lines[j].rstrip().endswith("\\"):
                j += 1
            found.append((i, j, m.group(1)))
            i = j + 1
            continue
        if line and not line[0].isspace():
            m = FUNC_RE.match(line)
            if m and not line.lstrip().startswith(("//", "/*")):
                j = i + 1
                while j < n and code_lines[j] != "}":
                    j += 1
                found.append((i, min(j, n - 1), m.group(1)))
                i = j + 1
                continue
        i += 1
    return found


def owner_of(found, line: int) -> str:
    best = None
    for start, end, name in found:
        if start <= line <= end:
            # innermost: the latest start wins (a macro body inside a function)
            if best is None or start >= best[0]:
                best = (start, end, name)
    return best[2] if best else "?"


def kernel_fragment(code_lines: list[str], site_line: int, site_col: int) -> str:
    """The kernel identifier the launch names (the text before ``<<<``).

    Empty when the text before ``<<<`` is not an identifier — a macro parameter
    (``(KERN)``) or a kernel name carried on the macro line above (``GEMM_ONE``).
    An empty fragment means "the gate cannot statically require a substring"; the
    runtime gate still asserts the message names *something*.
    """
    text = code_lines[site_line][:site_col].strip()
    text = re.sub(r"^(?:case\s+\w+:|default:)\s*", "", text)
    ident = re.match(r"([A-Za-z_][A-Za-z0-9_:]*)", text)
    return ident.group(1) if ident else ""


def statement_end(code_lines: list[str], line: int, col: int) -> int:
    """Index of the line carrying the ``;`` closing the ``<<<>>>`` statement."""
    depth = 0
    started = False
    i = line
    pos = col
    while i < len(code_lines):
        text = code_lines[i]
        while pos < len(text):
            c = text[pos]
            if c == "(":
                depth += 1
                started = True
            elif c == ")":
                depth -= 1
            elif c == ";" and started and depth == 0:
                return i
            pos += 1
        i += 1
        pos = 0
    raise ValueError("unterminated launch statement at line %d" % (line + 1))


def just_after_launch(code_lines, line, col):
    """``(line, col)`` just past the ``>>>`` closing the launch config.

    The config is usually one line, but the MMQ launchers wrap it across two
    (``<<<grid, 256,`` / ``minfer_launch_smem(…), stream>>>``), so scan forward.
    """
    i = line
    while i < len(code_lines):
        end = code_lines[i].find(">>>", col if i == line else 0)
        if end >= 0:
            return i, end + 3
        i += 1
    raise ValueError("no >>> after the launch on line %d" % (line + 1))


def site_string(text: str, prefix: str, window: tuple[int, int] | None = None,
                before: int | None = None) -> str | None:
    """The site argument of a helper call, resolving a simple variable.

    Most sites pass the literal, but #141/#147 declared ``const char* const site =
    "launch:…"`` (and ``launch_site = af32 ? "…" : "…"``), so an identifier is
    resolved through its assignment inside ``window`` (the enclosing function,
    widened past a macro body). ``site`` is a reused variable name, so the window
    is what keeps the resolution honest. A variable the audit cannot resolve is
    reported, never silently accepted.
    """
    m = re.search(re.escape(prefix) + r"\(([^,)]+)", text)
    if not m:
        return None
    arg = m.group(1).strip()
    lit = re.fullmatch(r'"(launch:[^"]*)"', arg)
    if lit:
        return lit.group(1)
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", arg):
        return _resolve_variable(arg, window, before)
    return None


def _resolve_variable(
    name: str, window: tuple[int, int] | None, before: int | None
) -> str | None:
    """The *nearest preceding* `name = "launch:…"` in the window.

    `site` is declared once per branch (the f16 matmul's vec and scalar arms both
    use the name), so the nearest assignment before the site is the only correct
    answer; a first-match search would report the vec token for the scalar arm.
    """
    lo, hi = window if window else (0, len(_RESOLVE_LINES))
    hi = min(hi, before if before is not None else hi)
    for i in range(hi, lo - 1, -1):
        if i >= len(_RESOLVE_LINES):
            continue
        m = _RESOLVE_RE.search(_RESOLVE_LINES[i])
        if m and m.group(1) == name:
            return m.group(2)
    return None


_RESOLVE_LINES: list[str] = []
_RESOLVE_RE = re.compile(r"[\s*>]\b([A-Za-z_][A-Za-z0-9_]*)\s*=[^;]*?\"(launch:[^\"]*)\"")


def set_resolve_source(code: str) -> None:
    global _RESOLVE_LINES
    _RESOLVE_LINES = code.split("\n")


def resolve_window(found, lineno: int) -> tuple[int, int]:
    """The line window a site's variables may be declared in.

    Normally the enclosing function. Inside a macro body (``GEMM_ONE`` holds the
    launch; ``launch_site`` is declared in ``launch_gemm_f16``) widen it to the
    enclosing function's start.
    """
    own = None
    for start, end, _name in found:
        if start <= lineno <= end and (own is None or start >= own[0]):
            own = (start, end)
    if own is None:
        return (0, len(_RESOLVE_LINES))
    outer = None
    for start, end, _name in found:
        if start < own[0] and end >= own[1] and (outer is None or start >= outer[0]):
            outer = (start, end)
    if outer is not None:
        return (outer[0], own[1])
    return own


def audit(source: Path):
    raw = source.read_text()
    code, blanked = code_only(raw)
    code_lines = code.split("\n")
    blank_lines = blanked.split("\n")
    raw_lines = raw.split("\n")
    set_resolve_source(code)
    found = owners(code_lines)
    rows = []
    problems = []
    line_start = 0
    for lineno, line in enumerate(raw_lines):
        base = line_start
        line_start += len(line) + 1
        col = 0
        while True:
            idx = line.find("<<<", col)
            if idx < 0:
                break
            col = idx + 3
            # ignore a commented-out or string-literal occurrence
            if blanked[base + idx : base + idx + 3] != "<<<":
                continue
            owner = owner_of(found, lineno)
            window = resolve_window(found, lineno)
            end_line = statement_end(code_lines, lineno, idx)
            # site token: nearest preceding prelude inside the same owner
            # The site's own prelude: the nearest one before it. A block with two
            # launches (the MMQ if/else, #147) shares one prelude, which is what
            # "before the launch" asks for.
            pre = None
            for back in range(lineno, window[0] - 1, -1):
                seg = code_lines[back]
                if back == lineno:
                    seg = seg[:idx]
                tok = site_string(seg, "minfer_launch_prelude", window, lineno)
                if tok:
                    pre = tok
                    break
                # a preceding read means the previous site's block ended: this
                # site has no prelude of its own (do not borrow the previous
                # site's, which would misreport the failure as a token mismatch)
                if re.search(r"minfer_launch_ok(?:_opt)?\(", seg):
                    break
            frag = kernel_fragment(code_lines, lineno, idx)
            rows.append((lineno + 1, owner, pre or "", frag))
            # the injection lever: the launch geometry must route through a
            # helper, so arming the site can drive a REAL failing launch. The
            # lever is the block (`minfer_launch_block`, every ordinary site) or
            # the dynamic smem (`minfer_launch_smem`, the #147 smem launchers).
            cfg_start = blanked.find("<<<", base + idx)
            cfg_end = blanked.find(">>>", cfg_start)
            cfg = code[cfg_start : cfg_end + 3]
            if not ("minfer_launch_block(" in cfg or "minfer_launch_smem(" in cfg):
                problems.append(
                    "%d: the <<< config carries no minfer_launch_block/smem lever, "
                    "so arming the site cannot make the launch fail for real "
                    "(owner %s)" % (lineno + 1, owner)
                )
            if not pre:
                problems.append(
                    "%d: no minfer_launch_prelude(\"launch:…\") before the <<< "
                    "(owner %s)" % (lineno + 1, owner)
                )
                continue
            # the read after the statement, in the same owner
            ok_tok = None
            gt_line, gt_col = just_after_launch(code_lines, lineno, idx)
            for fwd in range(end_line, len(code_lines)):
                seg = code_lines[fwd]
                if fwd == gt_line:
                    seg = seg[gt_col:]
                m = re.search(r"minfer_launch_ok(?:_opt)?\(", seg)
                if m:
                    ok_tok = site_string(seg[m.start() :], "minfer_launch_ok", window, lineno)
                    if ok_tok is None:
                        ok_tok = site_string(seg[m.start() :], "minfer_launch_ok_opt", window, lineno)
                    break
            if ok_tok is None:
                problems.append(
                    "%d: the <<< at this site has no minfer_launch_ok/_opt read "
                    "after it (owner %s, site %s)" % (lineno + 1, owner, pre)
                )
                continue
            if ok_tok != pre:
                problems.append(
                    "%d: the read names %s but the launch site is %s (owner %s)"
                    % (lineno + 1, ok_tok, pre, owner)
                )
    return rows, problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--list", action="store_true", help="print line/owner/site/kernel")
    ap.add_argument("--fixture", type=Path, help="write the site list to PATH")
    ap.add_argument("--check-fixture", type=Path, help="compare against PATH")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    rows, problems = audit(args.source)
    if args.list or args.fixture:
        for lineno, owner, site, frag in rows:
            print("%d\t%s\t%s\t%s" % (lineno, owner, site, frag))
    if args.fixture:
        args.fixture.write_text(
            "\n".join("%d\t%s\t%s\t%s" % r for r in rows) + "\n"
        )
    if args.check_fixture:
        want = [
            tuple(l.split("\t"))
            for l in args.check_fixture.read_text().splitlines()
            if l.strip()
        ]
        # The line column is documentation for the PR table, not identity: a
        # comment above a site shifts every line and must not turn the gate red.
        # (owner, site, kernel-fragment) is what the device gate compares.
        want = [w[1:4] for w in want if len(w) == 4]
        have = [(b, c, d) for _a, b, c, d in rows]
        if want != have:
            for i, (w, h) in enumerate(zip(want, have)):
                if w != h:
                    print(
                        "fixture mismatch at row %d: fixture %r, source %r"
                        % (i + 1, w, h)
                    )
            if len(want) != len(have):
                print(
                    "fixture has %d sites, source has %d" % (len(want), len(have))
                )
            return 1
    if problems:
        print(
            "cuda launch-return audit: %d unchecked site(s) in %s"
            % (len(problems), args.source)
        )
        for p in problems:
            print("  " + p)
        return 1
    print(
        "cuda launch-return audit: every one of the %d <<< sites reads its own "
        "error through minfer_launch_ok/_opt and names a launch: site"
        % len(rows)
    )
    return 0


def selftest() -> int:
    """Pin the parser against the shapes it must not be fooled by."""
    cases = [
        # a launch whose only "read" is in a comment must fail
        ("// x <<<1,1>>>\nvoid f(){ g<<<1,1>>>(); }\n", 1),
        # a launch with a prelude, a lever and a matching read must pass
        (
            'static void h(const char* s);\nvoid f(){\n'
            '  minfer_launch_prelude("launch:x", "g");\n'
            '  g<<<1, minfer_launch_block("launch:x", 1), 0, s>>>();\n'
            '  minfer_launch_ok("launch:x", "g");\n}\n',
            0,
        ),
        # a read that names the wrong site must fail
        (
            'static void h(const char* s);\nvoid f(){\n'
            '  minfer_launch_prelude("launch:x", "g");\n'
            '  g<<<1, minfer_launch_block("launch:x", 1), 0, s>>>();\n'
            '  minfer_launch_ok("launch:y", "g");\n}\n',
            1,
        ),
        # a site with a read but no injection lever must fail (arming it could
        # not drive a real failing launch)
        (
            'static void h(const char* s);\nvoid f(){\n'
            '  minfer_launch_prelude("launch:x", "g");\n'
            '  g<<<1, 1, 0, s>>>();\n'
            '  minfer_launch_ok("launch:x", "g");\n}\n',
            1,
        ),
        # a macro body site is found
        (
            'static void h(const char* s);\nvoid f(){\n'
            '  #define M() do { minfer_launch_prelude("launch:m", "g"); '
            'g<<<1, minfer_launch_block("launch:m", 1), 0, s>>>(); '
            'minfer_launch_ok("launch:m", "g"); } while (0)\n}\n',
            0,
        ),
    ]
    import tempfile

    for body, want in cases:
        with tempfile.NamedTemporaryFile("w", suffix=".cu", delete=False) as fh:
            fh.write(body)
            path = Path(fh.name)
        rows, problems = audit(path)
        got = 1 if problems else 0
        if got != want:
            print("selftest failed: want %d got %d for:\n%s\n%s" % (want, got, body, problems))
            path.unlink()
            return 1
        path.unlink()
    print("check_cuda_launch_returns.py selftest: %d cases pass" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main())
