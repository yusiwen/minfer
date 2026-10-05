#!/usr/bin/env python3
"""Audit the *shape* of every `allow(dead_code)` annotation (issue #254, Layer 1).

Sibling of ``scripts/check_dead_code_oracle.py`` (Layer 2), which is the liveness
half: it strips every annotation and compares rustc's own dead-code set against
``docs/dead-code-baseline.toml``. This half is cheap — no build, no rustc — and
**it checks the annotation's shape, not liveness.** It cannot see whether an item
is really dead; it sees only that the annotation is spelled in a form whose
meaning is checkable, and that a human wrote down *why*. Saying that plainly is
what keeps it honest: a green run here is not evidence that no code is dead.

The rules, from AGENTS.md Core Convention 5 and the #238–#244 series:

``R1`` **A bare ``#[allow(dead_code)]`` is rejected.** The only acceptable item-level
forms are ``#[cfg_attr(<cfg>, allow(dead_code))]`` — naming the *configuration in
which the item is unused* (``not(test)`` for a test-only reader,
``not(feature = "cuda")`` for a device-only one) — or scoping the item
``#[cfg(test)]``, which needs no annotation at all. A bare ``allow`` is legal
only for the sites grandfathered below, which are dead in **every** compilable
configuration, so a ``cfg_attr`` naming one configuration would be a false
statement about the other (the [#243] rule). Each such site carries a reason on
the item (``R3``) and names its owner ticket or the row that constructs it.

``R2`` **A ``cfg_attr`` annotation must name a ``cfg``.** ``cfg_attr(<cfg>, …)``
whose ``<cfg>`` is empty or is not a configuration predicate is rejected —
``#[cfg_attr(dead_code, allow(dead_code))]`` names the lint, not a configuration.

``R3`` **The annotated item carries a reason.** The contiguous comment block
directly above the annotation (through any other attributes in the same run) must
contain a *reason marker*: a ticket reference (``#254``, ``[#244]``), a named
consumer or configuration (``tokenizer::tests``, ``not(test)``, ``[`select`]``),
or an explicit deferred-use phrase (``deferred``, ``reserved``, ``would
construct``, ``kept pending``, …). This is a *presence* test: it cannot judge
whether the reason is true, and it is deliberately not a prose grader.

``R4`` **Point at the preferred form.** An item that is only ever reached from
``#[cfg(test)]`` code belongs under ``#[cfg(test)]`` (or in the module's
``tests.rs``), not behind an ``allow``; the rejection message says so.

**Grandfathering.** The 11 bare sites that exist on the ticket's base commit are
listed in ``GRANDFATHERED_BARE``, keyed ``<path>:<item>``. This is a ratchet on
new code, not a bulk rewrite. The list is also a ratchet *downwards*: an entry
that no longer covers a bare site (the annotation was tightened, or the item was
deleted) is a failure, so fixing a site forces the entry out.

Usage::

    python3 scripts/check_dead_code_annotations.py [--root src]
    python3 scripts/check_dead_code_annotations.py --selftest

Exit codes: 0 clean, 1 violations, 2 usage or I/O error.

[#243]: https://github.com/yusiwen/minfer/issues/243
[#244]: https://github.com/yusiwen/minfer/issues/244
[#254]: https://github.com/yusiwen/minfer/issues/254
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile

#: Where the annotations live, relative to the repository root.
DEFAULT_ROOT = "src"

#: `#[allow(<lints>)]` / `#![allow(<lints>)]` — the bare form. The `lints` group
#: is `[^()]*` on purpose: the spellings in this tree are flat lists, and a
#: nested one must be reported as unrecognised rather than mis-read.
BARE_ALLOW = re.compile(
    r"^[ \t]*#(?P<inner>!?)\[[ \t]*allow\((?P<lints>[^()]*)\)[ \t]*\](?P<tail>[ \t]*//.*)?$"
)

#: `#[cfg_attr(<cfg>, allow(<lints>))]` / inner form. `<cfg>` is greedy so a
#: comma *inside* the cfg (`not(any(feature = "cuda", test))`) is not mistaken
#: for the separator.
CFG_ATTR_ALLOW = re.compile(
    r"^[ \t]*#(?P<inner>!?)\[[ \t]*cfg_attr\((?P<cfg>.+),[ \t]*allow\((?P<lints>[^()]*)\)"
    r"[ \t]*\)[ \t]*\](?P<tail>[ \t]*//.*)?$"
)

#: Any attribute line, used to walk an attribute run and to spot a spelling this
#: checker does not understand.
ATTRIBUTE = re.compile(r"^[ \t]*#!?\[")

DEAD_CODE = re.compile(r"\bdead_code\b")

#: Configuration predicate names `R2` accepts. A custom `--cfg` is added here
#: deliberately: a checker that guesses is worse than one that asks to be taught.
CFG_PREDICATES = {
    "all",
    "any",
    "debug_assertions",
    "doc",
    "doctest",
    "false",
    "feature",
    "fuzzing",
    "miri",
    "not",
    "panic",
    "proc_macro",
    "relocation_model",
    "sanitize",
    "target_abi",
    "target_arch",
    "target_endian",
    "target_env",
    "target_family",
    "target_feature",
    "target_has_atomic",
    "target_os",
    "target_pointer_width",
    "target_thread_local",
    "target_vendor",
    "test",
    "true",
    "ub_checks",
    "unix",
    "windows",
}

#: A reason marker inside the comment block (`R3`). Presence, not truth: a ticket
#: reference, a backticked name/cfg/type, or one of the deferred-use phrases the
#: #244 convention uses to say what would construct or read the item.
REASON_MARKERS = (
    re.compile(r"#\d+"),
    re.compile(r"`[^`]+`"),
    re.compile(
        r"\b(?:tests?|gates?|fixtures?|kernels?|callers?|readers?|deferred|reserved|"
        r"retained|pending|would construct|would read|no caller|no production|"
        r"compiled out|parity|unjudged)\b",
        re.IGNORECASE,
    ),
)

#: ``<path>:<item>`` for the bare ``allow(dead_code)`` sites that predate this
#: ratchet. Every one is dead in **every** compilable configuration, so a
#: ``cfg_attr`` would name a configuration that does not exist; the reason on the
#: item says what would construct or read it. Tighten one and its entry here must
#: go — a stale entry is a failure (the ratchet only turns one way).
#:
#: The two former `src/metal.rs` entries were judged on a Mac by
#: [#255]: `matmul_on_gpu_buf` is `#[cfg(test)]` (its only readers are
#: `src/metal/tests.rs`) and the `impl MpsState` blanket was deleted (the macOS
#: non-test oracle reports no dead member).
GRANDFATHERED_BARE = {
    "src/convert.rs:order",
    "src/cuda/ffi_runtime.rs:cudaStreamWaitEvent",
    "src/cuda/methods/events.rs:stream_wait_event",
    "src/device_tier.rs:Amd",
    "src/device_tier.rs:Mthreads",
    "src/device_tier.rs:Apple",
    "src/graph/ops.rs:Mha",
    "src/tokenizer.rs:id_to_score",
    "src/tokenizer.rs:id_to_type",
}

#: The item a line names, best effort. Ordered: `impl` and the keyword forms are
#: unambiguous, a field has a `:`, a variant is a bare CamelCase name.
ITEM_NAME = (
    (re.compile(r"^\s*impl\s*(?:<[^>]*>\s*)?([A-Za-z_]\w*)"), "impl"),
    (re.compile(r"\bfn\s+([A-Za-z_]\w*)"), "fn"),
    (re.compile(r"\b(?:struct|enum|union|trait|type|const|static)\s+([A-Za-z_]\w*)"), "item"),
    (re.compile(r"^\s*(?:pub(?:\s*\([^)]*\))?\s+)?([A-Za-z_]\w*)\s*:"), "field"),
    (re.compile(r"^\s*([A-Z][A-Za-z0-9_]*)\s*[,({=]"), "variant"),
)


def item_name(lines: list[str], index: int) -> str | None:
    """The name of the item the attribute at ``index`` decorates."""
    j = index + 1
    while j < len(lines):
        s = lines[j].strip()
        if not s or s.startswith("//") or ATTRIBUTE.match(lines[j]):
            j += 1
            continue
        for rx, _kind in ITEM_NAME:
            m = rx.search(s)
            if m:
                return m.group(1)
        return None
    return None


def reason_comment(lines: list[str], index: int) -> str:
    """The comment block that documents the attribute at ``index``.

    Walks up through any other attributes first (``#[cfg(test)]`` above the
    annotation is still the same item), then collects the contiguous ``//`` block.
    """
    j = index - 1
    while j >= 0 and ATTRIBUTE.match(lines[j]):
        j -= 1
    block: list[str] = []
    while j >= 0 and lines[j].lstrip().startswith("//"):
        block.append(lines[j].strip())
        j -= 1
    block.reverse()
    tail = ""
    m = BARE_ALLOW.match(lines[index]) or CFG_ATTR_ALLOW.match(lines[index])
    if m and m.group("tail"):
        tail = m.group("tail")
    return " ".join(block) + " " + tail


def has_reason(text: str) -> bool:
    return any(rx.search(text) for rx in REASON_MARKERS)


def cfg_names_a_configuration(cfg: str) -> bool:
    """``R2``: is ``cfg`` a configuration predicate rather than empty/a literal?"""
    cfg = cfg.strip()
    if not cfg:
        return False
    m = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", cfg)
    return bool(m) and m.group(1) in CFG_PREDICATES


def check_file(
    path: str,
    display: str,
    grandfather: set[str],
    problems: list[str],
    covered: set[str],
) -> None:
    lines = open(path, encoding="utf-8").read().splitlines()
    for i, line in enumerate(lines):
        if not DEAD_CODE.search(line):
            continue
        bare = BARE_ALLOW.match(line)
        cfgattr = CFG_ATTR_ALLOW.match(line)
        if not bare and not cfgattr:
            if ATTRIBUTE.match(line) and "allow" in line:
                problems.append(
                    f"{display}:{i + 1}: unrecognised `dead_code` annotation — this checker "
                    f"judges only `#[allow(...)]` and `#[cfg_attr(<cfg>, allow(...))]` on one "
                    f"line; rewrite it in one of those shapes"
                )
            continue
        name = item_name(lines, i)
        key = f"{display}:{name}" if name else None
        if bare:
            if DEAD_CODE.search(bare.group("lints")) is None:
                continue
            if key not in grandfather:
                problems.append(
                    f"{display}:{i + 1}: bare `#[allow(dead_code)]`"
                    + (f" on `{name}`" if name else "")
                    + " — name the configuration in which the item is unused, with the note: "
                    "`#[cfg_attr(not(test), allow(dead_code))]` (test-only reader), "
                    "`#[cfg_attr(not(feature = \"cuda\"), allow(dead_code))]` (device-only), or "
                    "move the item under `#[cfg(test)]` / into `tests.rs` (no annotation then); "
                    "if it is genuinely dead in every configuration, add it to "
                    "GRANDFATHERED_BARE with a reason (#254)"
                )
            elif key:
                covered.add(key)
        else:
            cfg = cfgattr.group("cfg")
            if not cfg_names_a_configuration(cfg):
                problems.append(
                    f"{display}:{i + 1}: `cfg_attr({cfg.strip()}, allow(dead_code))` does not name a "
                    "configuration — the first argument must be a cfg predicate (`not(test)`, "
                    "`not(feature = \"cuda\")`, `any(...)`, `target_os = ...`) (#243, #254)"
                )
            if DEAD_CODE.search(cfgattr.group("lints")) is None:
                continue
        note = reason_comment(lines, i)
        if not has_reason(note):
            problems.append(
                f"{display}:{i + 1}: no reason for the `dead_code` annotation"
                + (f" on `{name}`" if name else "")
                + " — the comment block above it must name the consumer (`tokenizer::tests`, "
                "`not(feature = \"cuda\")`), cite the owning ticket (`#254`), or say what would "
                "construct/read the item (#244, #254)"
            )


def check(
    root: str = DEFAULT_ROOT,
    grandfather: set[str] | None = None,
    report_stale: bool = True,
) -> list[str]:
    """Every violation under ``root``, as ``path:line: message``."""
    if grandfather is None:
        grandfather = GRANDFATHERED_BARE
    files: list[str] = []
    for r, _dirs, fs in os.walk(root):
        files += [os.path.join(r, f) for f in fs if f.endswith(".rs")]
    files.sort()
    if not files:
        return [f"{root}: no .rs files"]
    problems: list[str] = []
    covered: set[str] = set()
    base = os.path.dirname(root.rstrip("/")) or "."
    for f in files:
        check_file(f, os.path.relpath(f, base).replace(os.sep, "/"), grandfather, problems, covered)
    if report_stale:
        for key in sorted(grandfather - covered):
            problems.append(
                f"{key}: GRANDFATHERED_BARE entry no longer covers a bare `allow(dead_code)` — the "
                f"site was tightened or deleted; drop the entry (the ratchet shrinks) (#254)"
            )
    return problems


def selftest() -> int:
    """Pin the checker against the shapes it must accept and reject."""
    # (name, source, expected problem count). The grandfather list is injected so
    # the ratchet's own behaviour is a case, not an assumption.
    cases = [
        (
            "bare allow is rejected",
            "// #1 says so\n#[allow(dead_code)]\nfn a() {}\n",
            1,
        ),
        (
            "bare allow on a grandfathered item is accepted",
            "// Kept pending #138; a caller in the device path would use it.\n#[allow(dead_code)]\nfn kept() {}\n",
            0,
        ),
        (
            "cfg_attr naming not(test) with a reason is accepted",
            "// Read by `thing::tests` only.\n#[cfg_attr(not(test), allow(dead_code))]\nfn b() {}\n",
            0,
        ),
        (
            "cfg_attr with no comment is rejected",
            "#[cfg_attr(not(test), allow(dead_code))]\nfn c() {}\n",
            1,
        ),
        (
            "cfg_attr whose reason is not contiguous is rejected",
            "// Read by `thing::tests`.\n\n#[cfg_attr(not(test), allow(dead_code))]\nfn c2() {}\n",
            1,
        ),
        (
            "cfg_attr with a comment that states no reason is rejected",
            "// A function.\n#[cfg_attr(not(test), allow(dead_code))]\nfn c3() {}\n",
            1,
        ),
        (
            "cfg_attr must name a cfg",
            "// Deferred: a `select` caller would read it.\n#[cfg_attr(dead_code, allow(dead_code))]\nfn d() {}\n",
            1,
        ),
        (
            "cfg_attr with an empty cfg is rejected",
            "// Deferred: nothing constructs it yet.\n#[cfg_attr(, allow(dead_code))]\nfn d2() {}\n",
            1,
        ),
        (
            "a combined allow list with dead_code is still bare",
            "// Reason: #1.\n#[allow(unused, dead_code)]\nfn e() {}\n",
            1,
        ),
        (
            "an allow list without dead_code is ignored",
            "// Reason: #1.\n#[allow(unused)]\nfn f() {}\n",
            0,
        ),
        (
            "inner cfg_attr with a module-doc reason is accepted",
            "//! Kept for the CUDA path (#167); the `cuda` feature builds it in.\n"
            "#![cfg_attr(not(feature = \"cuda\"), allow(dead_code))]\nfn g() {}\n",
            0,
        ),
        (
            "the reason may come from the attribute's own trailing comment",
            "#[cfg_attr(not(test), allow(dead_code))] // read by `thing::tests`\nfn h() {}\n",
            0,
        ),
        (
            "a multi-line cfg predicate still names the configuration",
            "// Deferred: `not(test)` because `thing::tests` constructs it.\n"
            "#[cfg_attr(not(any(feature = \"cuda\", test)), allow(dead_code))]\n"
            "enum I { A, B }\n",
            0,
        ),
        (
            "a cfg_attr on a variant with a reason is accepted",
            "// Reserved: a device row keyed by the `0x1000000` offset.\n"
            "#[cfg_attr(not(test), allow(dead_code))]\n"
            "A,\n",
            0,
        ),
    ]
    failures = 0
    for name, body, want in cases:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "src")
            os.makedirs(src)
            with open(os.path.join(src, "fixture.rs"), "w", encoding="utf-8") as fh:
                fh.write(body)
            got = len(check(src, grandfather={"src/fixture.rs:kept"}, report_stale=False))
        if got != want:
            print(f"selftest FAIL: {name}: want {want} problem(s), got {got}")
            failures += 1
    # The ratchet's other direction: a grandfather entry with no bare site left is
    # a failure, so a tightened/deleted site forces its entry out.
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "src")
        os.makedirs(src)
        with open(os.path.join(src, "fixture.rs"), "w", encoding="utf-8") as fh:
            fh.write("// Reason: #1.\n#[cfg_attr(not(test), allow(dead_code))]\nfn gone() {}\n")
        got = len(check(src, grandfather={"src/fixture.rs:gone"}, report_stale=True))
    if got != 1:
        print(f"selftest FAIL: a stale grandfather entry fails: want 1 problem, got {got}")
        failures += 1
    if failures:
        print(f"check_dead_code_annotations.py selftest: {failures} case(s) failed")
        return 1
    print(f"check_dead_code_annotations.py selftest: {len(cases)} cases pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT, help="source root to audit")
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not os.path.isdir(args.root):
        print(f"{args.root}: not a directory", file=sys.stderr)
        return 2
    problems = check(args.root)
    for p in problems:
        print(p)
    if problems:
        print(
            f"check_dead_code_annotations.py: {len(problems)} annotation-shape problem(s). "
            "This is a shape check, not a liveness check (#254); the liveness half is "
            "scripts/check_dead_code_oracle.py.",
            file=sys.stderr,
        )
        return 1
    print(
        "check_dead_code_annotations.py: annotation shapes clean "
        f"({len(GRANDFATHERED_BARE)} grandfathered bare site(s)) — shape only, not liveness (#254)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
