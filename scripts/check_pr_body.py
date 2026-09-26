#!/usr/bin/env python3
"""Require a PR body to carry the two gate sections, filled.

`docs/GATE-CONTRACT.md` states the five rules a gate must satisfy. Two of them
live in the PR body as prose: the numeric **bar named before measuring** (rules
3 and 5) and the **mutation evidence** showing the gate can fail (rule 3).
Prose cannot be checked for truth, but it can be checked for *presence*: this
script fails a body whose required headings are missing, or whose two gate
sections are empty or left at the template placeholder.

**The honest limit.** This check can force the sentence to exist; it can never
force it to be true. A body that names a bar and pastes a transcript still has
to be read by a human — presence is all a script can see, and pretending
otherwise would be exactly the "passed for the wrong reason" failure
`docs/GATE-CONTRACT.md` warns about (issue #175).

Rules, kept simple and documented:

- **Headings.** The seven required headings are matched as ATX headings at
  level two or three (`##` / `###`), case-insensitively, with surrounding
  whitespace, trailing `#`s and surrounding emphasis/backticks ignored. The
  section body runs to the next heading at the same or a shallower level.
- **Fences.** A heading inside a fenced code block (``` or ~~~) does not count:
  a body that merely *quotes* the template must not pass. The fence rule is the
  common simple one — the opening marker's character closes the block; no
  CommonMark length/nesting subtleties.
- **Filled vs empty.** HTML comments are stripped first, so a section that
  still holds only the template's `<!-- guidance -->` is empty. A section whose
  every remaining non-blank line starts with the `FILL-ME:` placeholder prefix
  is likewise unfilled. `N/A — <reason>` (also the ASCII hyphen form) is a
  **filled** section: a stated non-applicability is honest, an empty section is
  not.
- **Non-gate sections.** Only presence is required for What changed / Why /
  Verification / Honest scope / Follow-ups; only the two gate sections are
  checked for content.

Usage:

    python3 scripts/check_pr_body.py <body-file>          # or `-` for stdin
    python3 scripts/check_pr_body.py --body-env VAR       # body from $VAR
    python3 scripts/check_pr_body.py --selftest           # built-in cases

CI passes the body through `env:` and uses `--body-env`, never a shell
interpolation of `${{ github.event.pull_request.body }}` — a PR body is
attacker-controlled text and interpolating it into a `run:` command is a
script-injection hazard. The check is pure and offline: stdlib only, no API
token, no network, so it works on a fork PR.

Exit codes: 0 the body passes, 1 the body fails, 2 usage or I/O error.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

# The seven headings, in template order. A missing one is named by its exact
# heading text, so the CI failure points at the line to add.
REQUIRED_HEADINGS = (
    "What changed",
    "Why",
    "Verification",
    "Bar named before measuring",
    "Mutation evidence",
    "Honest scope",
    "Follow-ups",
)

# Only these two are checked for content (the "gate sections").
GATE_HEADINGS = (
    "Bar named before measuring",
    "Mutation evidence",
)

# The template's placeholder prefix. A section whose every non-blank line starts
# with it is unfilled — the "placeholder left in place" failure.
PLACEHOLDER_PREFIX = "fill-me:"

#: The limit that `--help` must state (also asserted by `--selftest`).
HELP_LIMIT = (
    "This check can force the sentence to exist; it can never force it to be true."
)

FENCE_RE = re.compile(r"^\s{0,3}(`{3,}|~{3,})")
HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
WHITESPACE_RE = re.compile(r"\s+")

REQUIRED_NORMS = {h.casefold(): h for h in REQUIRED_HEADINGS}
GATE_NORMS = {h.casefold() for h in GATE_HEADINGS}


def normalize_heading(text: str) -> str:
    """A heading text reduced to its comparable form.

    Case, surrounding whitespace, a trailing colon and surrounding emphasis or
    backticks are all ignored; internal whitespace is collapsed.
    """
    t = text.strip().strip("*_`").strip()
    t = t.rstrip(":").strip()
    return WHITESPACE_RE.sub(" ", t).casefold()


def parse_headings(lines: list[str]):
    """Yield `(line_index, level, text)` for ATX headings outside code fences."""
    fence: str | None = None
    for i, line in enumerate(lines):
        fence_match = FENCE_RE.match(line)
        if fence_match:
            marker = fence_match.group(1)[0]
            if fence is None:
                fence = marker
            elif fence == marker:
                fence = None
            continue
        if fence is not None:
            continue
        heading_match = HEADING_RE.match(line)
        if heading_match:
            yield i, len(heading_match.group(1)), heading_match.group(2)


def sections(body: str) -> dict[str, tuple[int, str]]:
    """Map each required heading's normalized text to `(line_number, body_text)`.

    The first occurrence wins. A section body runs from the line after its
    heading to the next heading at the same or a shallower level (or EOF); a
    deeper heading stays inside it.
    """
    lines = body.splitlines()
    marks = list(parse_headings(lines))
    found: dict[str, tuple[int, str]] = {}
    for pos, (index, level, text) in enumerate(marks):
        if level not in (2, 3):
            continue  # only ## / ### identify a required section
        norm = normalize_heading(text)
        if norm not in REQUIRED_NORMS or norm in found:
            continue
        end = len(lines)
        for later_index, later_level, _ in marks[pos + 1 :]:
            if later_level <= level:
                end = later_index
                break
        found[norm] = (index + 1, "\n".join(lines[index + 1 : end]))
    return found


def section_state(text: str) -> str:
    """`"empty"`, `"placeholder"` or `"filled"` for a gate section's body."""
    without_comments = COMMENT_RE.sub("", text)
    content = [line.strip() for line in without_comments.splitlines() if line.strip()]
    if not content:
        return "empty"
    if all(line.casefold().startswith(PLACEHOLDER_PREFIX) for line in content):
        return "placeholder"
    return "filled"


def check_body(body: str) -> list[str]:
    """Every problem with `body`, in template order; empty means it passes."""
    found = sections(body)
    problems: list[str] = []
    for position, heading in enumerate(REQUIRED_HEADINGS, start=1):
        if heading.casefold() in found:
            continue
        previous = REQUIRED_HEADINGS[position - 2] if position > 1 else None
        where = f"required heading {position} of {len(REQUIRED_HEADINGS)}"
        if previous is not None:
            where += f', expected after "## {previous}"'
        else:
            where += " and must come first"
        problems.append(f'missing required heading "## {heading}" ({where})')

    for heading in GATE_HEADINGS:
        entry = found.get(heading.casefold())
        if entry is None:
            continue  # already reported as missing
        line, text = entry
        state = section_state(text)
        if state == "empty":
            problems.append(
                f'"## {heading}" (line {line}) is empty — state the fact, or '
                f'"N/A — <reason>" if it genuinely does not apply'
            )
        elif state == "placeholder":
            problems.append(
                f'"## {heading}" (line {line}) still holds only the template '
                f'placeholder ("{PLACEHOLDER_PREFIX} ...")'
            )
    return problems


def usage_error(message: str) -> None:
    """Print `message` and exit 2 (usage or I/O, distinct from a body failure)."""
    print(message, file=sys.stderr)
    raise SystemExit(2)


def read_body(args: argparse.Namespace) -> str:
    """The body named by the CLI; exits 2 on a usage or I/O error."""
    if args.body_env and args.file is not None:
        usage_error("check_pr_body: give either a body file or --body-env, not both")
    if not args.body_env and args.file is None:
        usage_error("check_pr_body: give a body file (or `-` for stdin) or --body-env VAR")
    if args.body_env:
        if args.body_env not in os.environ:
            usage_error(f"check_pr_body: environment variable {args.body_env!r} is not set")
        return os.environ[args.body_env]
    if args.file == "-":
        return sys.stdin.read()
    try:
        return open(args.file, encoding="utf-8").read()
    except OSError as exc:
        usage_error(f"check_pr_body: cannot read {args.file!r}: {exc}")
        raise  # unreachable; keeps type checkers happy


def complete_body(
    bar: str = "the checker exits 0 on a template-shaped body and names the heading on each negative",
    mutation: str = "deleted `## Mutation evidence` -> exit 1, message names it",
) -> str:
    """A body that passes: every heading present and both gate sections filled."""
    return (
        "## What changed\n"
        "- added the checker\n"
        "## Why\n"
        "presence is checkable\n"
        "## Verification\n"
        "`python3 scripts/check_pr_body.py --selftest` -> every case passes\n"
        f"## Bar named before measuring\n{bar}\n"
        f"## Mutation evidence\n{mutation}\n"
        "## Honest scope\n"
        "presence, not truth\n"
        "## Follow-ups\n"
        "N/A — none\n"
    )


def selftest_cases() -> list[tuple[str, str, bool, str | None]]:
    """`(name, body, should_pass, expected_substring)` for every built-in case."""
    missing = complete_body().replace("## Mutation evidence\n", "")
    placeholder = complete_body(bar="FILL-ME: the numeric bar, named before measuring")
    empty_gate = complete_body(mutation="<!-- guidance only -->")
    n_a = complete_body(
        bar="N/A — this change introduces no measured number",
        mutation="N/A - covered by the checker's own selftest",
    )
    heading_variants = (
        "### what changed ###\n"
        "- x\n"
        "### WHY\n"
        "y\n"
        "### Verification:\n"
        "z\n"
        "### BAR NAMED BEFORE MEASURING\n"
        "the bar\n"
        "### mutation evidence\n"
        "the mutation\n"
        "### Honest scope\n"
        "n/a\n"
        "### Follow-ups\n"
        "N/A — none\n"
    )
    fenced_gates = (
        "## What changed\n- x\n## Why\ny\n## Verification\nz\n"
        "The template quoted below must not satisfy the check:\n"
        "```markdown\n"
        "## Bar named before measuring\n"
        "the bar\n"
        "## Mutation evidence\n"
        "the mutation\n"
        "```\n"
        "## Honest scope\nn/a\n## Follow-ups\nN/A — none\n"
    )
    # The whole template, quoted inside a fence: no real heading exists.
    template = (
        "```markdown\n"
        "## What changed\nplaceholder\n## Why\nplaceholder\n## Verification\nplaceholder\n"
        "## Bar named before measuring\nFILL-ME: the numeric bar\n"
        "## Mutation evidence\nFILL-ME: the mutation\n"
        "## Honest scope\nplaceholder\n## Follow-ups\nplaceholder\n"
        "```\n"
    )
    return [
        ("complete body passes", complete_body(), True, None),
        (
            "missing `## Mutation evidence` fails and names it",
            missing,
            False,
            "Mutation evidence",
        ),
        (
            "placeholder left in place fails",
            placeholder,
            False,
            "Bar named before measuring",
        ),
        ("empty gate section fails", empty_gate, False, "Mutation evidence"),
        ("`N/A — reason` passes", n_a, True, None),
        (
            "gate headings quoted inside a fence fail",
            fenced_gates,
            False,
            "Bar named before measuring",
        ),
        (
            "the whole template quoted inside a fence fails",
            template,
            False,
            "What changed",
        ),
        ("`###`/case/trailing-`#` headings pass", heading_variants, True, None),
    ]


def run_selftest() -> int:
    failures = 0
    for name, body, should_pass, expected in selftest_cases():
        problems = check_body(body)
        passed = not problems
        ok = passed == should_pass
        if ok and expected is not None:
            ok = any(expected in problem for problem in problems)
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            failures += 1
            print(f"      expected {'pass' if should_pass else f'a failure naming {expected!r}'}")
            print(f"      got: {problems or 'pass'}")
    # The help text must state the limit (issue #175 acceptance).
    help_ok = HELP_LIMIT in build_parser().description
    print(f"{'PASS' if help_ok else 'FAIL'}  --help states the presence/truth limit")
    if not help_ok:
        failures += 1
    total = len(selftest_cases()) + 1
    print(f"check_pr_body selftest: {total - failures}/{total} cases pass")
    return 1 if failures else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="check_pr_body.py",
        description=(
            "Require a PR body's headings and its two filled gate sections "
            "(docs/GATE-CONTRACT.md, issue #175). " + HELP_LIMIT
        ),
        epilog=(
            "Check only that the sentences are present — never that they are true. "
            "A section stating `N/A — <reason>` counts as filled."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="path to a file holding the PR body (`-` reads stdin)",
    )
    parser.add_argument(
        "--body-env",
        metavar="VAR",
        help="read the body from environment variable VAR (the CI-safe form)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run the built-in pass/fail cases in-process and exit non-zero on a failure",
    )
    return parser


def main(argv: list[str]) -> int:
    parser = build_parser()
    args = parser.parse_args(argv[1:])
    if args.selftest:
        if args.file is not None or args.body_env:
            parser.error("--selftest does not take a body")
        return run_selftest()

    body = read_body(args)
    problems = check_body(body)
    if problems:
        for problem in problems:
            print(f"check_pr_body: {problem}", file=sys.stderr)
        print(
            f"check_pr_body: the PR body fails the gate-shape check "
            f"({len(problems)} problem(s)); see .github/PULL_REQUEST_TEMPLATE.md",
            file=sys.stderr,
        )
        return 1
    print(
        f"check_pr_body: all {len(REQUIRED_HEADINGS)} required headings present and both "
        f"gate sections filled (presence only, not truth)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
