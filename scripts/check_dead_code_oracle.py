#!/usr/bin/env python3
"""The stripped-oracle ratchet against newly hidden dead code (issue #254, Layer 2).

Rustc treats an `#[allow(dead_code)]` item as a **liveness root**: the annotation
hides the item *and its whole call chain*. A single stray `allow` can therefore
hide a large dead subgraph while `deny(warnings)` stays quiet, which is why the
#227/#238–#244 cleanup measured liveness with the annotations *removed* rather
than with a name census. This script productises that oracle:

1. Copy the tree to a scratch directory (the checked-out tree is never modified)
   and replace **every** `allow(dead_code)` — bare, `cfg_attr`, combined lists and
   the inner `#![...]` form — with a line-preserving marker comment, so line
   numbers and spans stay comparable. A spelling this script cannot strip (a
   multi-line attribute, a `clippy::dead_code` path) is an error, never a silent
   skip; if zero annotations are stripped that is an error too.
2. Run ``cargo check --release [--features cuda] --message-format=json`` over the
   copy with ``RUSTFLAGS=--cap-lints=warn``. The cap matters: the crate's
   ``#![cfg_attr(not(test), deny(warnings))]`` would turn the lint into an error
   and can truncate the pass.
3. Take **every `src/` span** of every `dead_code` diagnostic — reading only the
   primary span undercounts by ~2.6× — and derive each item's ``(name, kind)``
   from the diagnostic's message (``fields `a`, `b` … are never read`` is two
   `field` items; ``variants `X`, `Y` …`` is two `variant` items).
4. Compare the set against ``docs/dead-code-baseline.toml``. **An addition fails**
   and is printed with its file:line and a paste-ready manifest entry; a removal
   is informational (the manifest only needs to shrink when convenient). Renames
   and moves are invisible on purpose: the key is ``(name, kind)``, and the entry's
   ``file`` is documentation, reported as a note when it moves.

**Architecture.** The baseline is the **union** of what the host and the x86_64 CI
runner see: the dead set can differ by ``cfg(target_arch)`` (NEON/AVX2-gated code),
so an item present on only one of the two is still one manifest entry, and the
other architecture reports it as a removal (informational). ``--target`` runs the
check against another triple without moving the manifest — that is how the
difference is measured locally; the authoritative comparison is the CI job, which
runs this script with no ``--target``.

**Target directory.** The scratch copy shares the calling tree's ``target/`` by
default (``--target-dir``), so the run pays for the crate, not for every
dependency. Only the local crate recompiles; the copy is deleted afterwards unless
``--keep`` is given.

**Blind spots, stated rather than discovered.**
* macOS is not compiled here (or on the Linux CI runner): a macOS-only module is
  invisible, and a cross-platform item whose only caller sits in a
  ``#[cfg(target_os = "macos")]`` test block *looks* dead here while it is live
  there (``ModelDef::forward_graph`` was exactly this). ``build-macos`` only
  type-checks, and the manifest carries ``macos = "unjudged"`` for that reason.
* The oracle is a ``cargo check``: "live" is a compile-time reference, not runtime
  reachability.
* ``--features debug_dump`` and ``cuda_static`` are not covered.

Usage::

    python3 scripts/check_dead_code_oracle.py --config cpu
    python3 scripts/check_dead_code_oracle.py --config cuda
    python3 scripts/check_dead_code_oracle.py --selftest
    python3 scripts/check_dead_code_oracle.py --config cpu --print-toml   # seed/update

Exit codes: 0 clean, 1 new dead item(s), 2 infrastructure error (unrecognised
annotation, cargo failure, an empty capture, a malformed manifest).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO / "docs" / "dead-code-baseline.toml"

#: The configurations the manifest carries, and the cargo feature set each one
#: compiles. `cuda` is maximal: `src/cuda.rs` and `src/graph/cuda_backend.rs` exist
#: only under the feature, so items they would have constructed are dead without
#: it.
CONFIGS = {
    "cpu": None,
    "cuda": "cuda",
}

#: Kinds the manifest may name, i.e. the message verbs rustc uses for dead code.
KINDS = ("fn", "struct", "enum", "union", "trait", "type", "const", "static", "field", "variant")

#: `#[allow(<lints>)]` / `#![allow(<lints>)]`.
BARE_ALLOW = re.compile(
    r"^(?P<indent>[ \t]*)#(?P<inner>!?)\[[ \t]*allow\((?P<lints>[^()]*)\)[ \t]*\](?P<tail>[ \t]*//.*)?$"
)
#: `#[cfg_attr(<cfg>, allow(<lints>))]` / inner form. `<cfg>` is greedy so a comma
#: inside the cfg (`not(any(feature = "cuda", test))`) is not the separator.
CFG_ATTR_ALLOW = re.compile(
    r"^(?P<indent>[ \t]*)#(?P<inner>!?)\[[ \t]*cfg_attr\((?P<cfg>.+),[ \t]*"
    r"allow\((?P<lints>[^()]*)\)[ \t]*\)[ \t]*\](?P<tail>[ \t]*//.*)?$"
)
ATTRIBUTE = re.compile(r"^[ \t]*#!?\[")
DEAD_CODE = re.compile(r"\bdead_code\b")
MARKER = "// minfer-dead-code-oracle: stripped"

#: Message verb -> kind, in first-match order.
KIND_BY_VERB = (
    (re.compile(r"\bfields?\b"), "field"),
    (re.compile(r"\bvariants?\b"), "variant"),
    (re.compile(r"\bconstants?\b"), "const"),
    (re.compile(r"\bstatics?\b"), "static"),
    (re.compile(r"\b(?:functions?|methods?)\b"), "fn"),
    (re.compile(r"\bstructs?\b"), "struct"),
    (re.compile(r"\benums?\b"), "enum"),
    (re.compile(r"\btraits?\b"), "trait"),
    (re.compile(r"\bunions?\b"), "union"),
    (re.compile(r"\btype aliases?\b"), "type"),
)
BACKTICK = re.compile(r"`([^`]+)`")


class OracleError(RuntimeError):
    """An infrastructure problem: the run proves nothing, so exit 2."""


def drop_dead_code(lints: str) -> str:
    """`lints` with `dead_code` removed, keeping the other lints in order."""
    parts = [p.strip() for p in lints.split(",") if p.strip()]
    for p in parts:
        if p != "dead_code" and "dead_code" in p:
            raise OracleError(f"unrecognised lint path `{p}` in an allow list")
    return ", ".join(p for p in parts if p != "dead_code")


def strip_source(text: str, where: str) -> tuple[str, list[int]]:
    """`text` with every `allow(dead_code)` replaced, line numbers preserved.

    Returns the new text and the 1-based lines that carried an annotation. A
    `dead_code` mention in a comment is prose and is left alone; one in an
    attribute this function does not understand is an error.
    """
    out: list[str] = []
    stripped: list[int] = []
    for i, raw in enumerate(text.splitlines(keepends=True), 1):
        line = raw.rstrip("\n")
        for rx, rebuild in (
            (CFG_ATTR_ALLOW, lambda m, rest: f'{m.group("indent")}#{m.group("inner")}[cfg_attr({m.group("cfg")}, allow({rest}))]'),
            (BARE_ALLOW, lambda m, rest: f'{m.group("indent")}#{m.group("inner")}[allow({rest})]'),
        ):
            m = rx.match(line)
            if not m or not DEAD_CODE.search(m.group("lints")):
                continue
            rest = drop_dead_code(m.group("lints"))
            tail = m.group("tail") or ""
            out.append((rebuild(m, rest) + tail if rest else f'{m.group("indent")}{MARKER}') + "\n")
            stripped.append(i)
            break
        else:
            if DEAD_CODE.search(line) and ATTRIBUTE.match(line):
                raise OracleError(
                    f"{where}:{i}: unrecognised `allow(dead_code)` spelling — this oracle strips "
                    f"only single-line `#[allow(...)]` / `#[cfg_attr(<cfg>, allow(...))]` "
                    f"attributes; rewrite it or teach strip_source() the shape"
                )
            out.append(raw)
    text_out = "".join(out)
    # Belt and braces: nothing that is code may still mention the lint. A
    # remaining one means a shape the loop above did not see (a multi-line list,
    # a macro), and a silently under-stripped oracle under-reports dead code.
    leftover = [
        (i, ln)
        for i, ln in enumerate(text_out.splitlines(), 1)
        if DEAD_CODE.search(ln) and not ln.lstrip().startswith("//")
    ]
    if leftover:
        i, ln = leftover[0]
        raise OracleError(
            f"{where}:{i}: `dead_code` survived the strip: {ln.strip()!r} — the oracle cannot "
            f"trust an under-stripped tree"
        )
    return text_out, stripped


def strip_tree(root: Path) -> int:
    """Strip every `.rs` under `root`; return the annotation count."""
    total = 0
    for path in sorted(root.rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        new, lines = strip_source(text, str(path))
        if lines:
            path.write_text(new, encoding="utf-8")
            total += len(lines)
    return total


def copy_tree(src: Path, dst: Path) -> None:
    """Copy `src` to `dst`, skipping build output and other worktrees."""
    shutil.copytree(
        src,
        dst,
        symlinks=True,
        ignore=shutil.ignore_patterns(
            "target", ".git", ".worktrees", "node_modules", ".direnv", "*.gguf", "book"
        ),
    )


def src_relative(file_name: str) -> str | None:
    """`file_name` as a `src/...` path, or None when it is not crate source."""
    fn = file_name.replace("\\", "/")
    if "/src/" in fn:
        return "src/" + fn.split("/src/", 1)[1]
    if fn.startswith("src/"):
        return fn
    return None


def items_of(message: str) -> list[tuple[str, str]]:
    """``(name, kind)`` for every item a dead-code message names."""
    names = BACKTICK.findall(message)
    if not names:
        return []
    kind = "item"
    for rx, k in KIND_BY_VERB:
        if rx.search(message):
            kind = k
            break
    return [(n, kind) for n in names]


def parse_capture(text: str) -> tuple[list[dict], set[tuple[str, int]], dict[tuple[str, str], dict]]:
    """`(diagnostics, sites, item_rows)` from a `--message-format=json` capture.

    ``sites`` is every `src/` span of every `dead_code` diagnostic (the primary
    span alone undercounts); ``item_rows`` maps ``(name, kind)`` to the first
    ``{file, line, message}`` that named it.
    """
    diags: list[dict] = []
    sites: set[tuple[str, int]] = set()
    items: dict[tuple[str, str], dict] = {}
    for raw in text.splitlines():
        try:
            o = json.loads(raw)
        except ValueError:
            continue
        if o.get("reason") != "compiler-message":
            continue
        m = o.get("message") or {}
        if (m.get("code") or {}).get("code") != "dead_code":
            continue
        spans = []
        for s in m.get("spans", []):
            rel = src_relative(s.get("file_name", ""))
            if rel is None:
                continue
            spans.append({"file": rel, "line": s["line_start"], "primary": bool(s.get("is_primary"))})
            sites.add((rel, s["line_start"]))
        if not spans:
            continue
        diags.append({"message": m.get("message", ""), "spans": spans})
        primary = next((s for s in spans if s["primary"]), spans[0])
        for name, kind in items_of(m.get("message", "")):
            items.setdefault(
                (name, kind),
                {"file": primary["file"], "line": primary["line"], "message": m.get("message", "")},
            )
    return diags, sites, items


def load_baseline(path: Path, allow_missing: bool = False) -> dict[str, list[dict]]:
    """The manifest, validated: every entry names an item, a kind and a reason.

    ``allow_missing`` is the seeding path: with ``--print-toml`` the manifest does
    not exist yet, and an empty one is the correct starting point.
    """
    if not path.is_file():
        if allow_missing:
            return {config: [] for config in CONFIGS}
        raise OracleError(f"{path}: manifest missing")
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise OracleError(f"{path}: not valid TOML: {e}") from e
    for config in CONFIGS:
        entries = data.get(config, [])
        if not isinstance(entries, list):
            raise OracleError(f"{path}: [[{config}]] must be an array of tables")
        seen: set[tuple[str, str]] = set()
        for e in entries:
            for field in ("name", "kind", "reason"):
                if not isinstance(e.get(field), str) or not e[field].strip():
                    raise OracleError(f"{path}: [[{config}]] entry {e!r} is missing `{field}`")
            if e["kind"] not in KINDS:
                raise OracleError(f"{path}: [[{config}]] `{e['name']}` has unknown kind `{e['kind']}`")
            key = (e["name"], e["kind"])
            if key in seen:
                raise OracleError(f"{path}: [[{config}]] lists `{e['name']}` ({e['kind']}) twice")
            seen.add(key)
    return data


def toml_string(s: str) -> str:
    """`s` as the body of a TOML basic string (the paste-ready entries must parse)."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def toml_entry(config: str, name: str, kind: str, file: str, reason: str) -> str:
    return (
        f"[[{config}]]\n"
        f'name = "{toml_string(name)}"\n'
        f'kind = "{toml_string(kind)}"\n'
        f'file = "{toml_string(file)}"\n'
        f'reason = "{toml_string(reason)}"\n'
    )


def check(
    tree: Path,
    config: str,
    baseline_path: Path,
    scratch_root: Path | None,
    target: str | None,
    target_dir: Path | None,
    capture_path: Path | None,
    keep: bool,
    print_toml: bool,
) -> int:
    features = CONFIGS[config]
    made_root = scratch_root is None
    root = Path(tempfile.mkdtemp(prefix="minfer-dead-code-oracle-")) if made_root else scratch_root
    assert root is not None
    work = root / "tree"
    if work.exists():
        raise OracleError(f"{work}: already exists — pass a fresh --scratch directory")
    root.mkdir(parents=True, exist_ok=True)
    try:
        copy_tree(tree, work)
        stripped = strip_tree(work / "src")
        if stripped == 0:
            raise OracleError(
                "no `allow(dead_code)` was stripped — the tree or the strip patterns changed; "
                "an oracle that strips nothing would pass on any dead code"
            )
        cmd = ["cargo", "check", "--release"]
        if features:
            cmd += ["--features", features]
        if target:
            cmd += ["--target", target]
        cmd += ["--message-format=json"]
        env = dict(os.environ)
        env["RUSTFLAGS"] = (env.get("RUSTFLAGS", "").strip() + " --cap-lints=warn").strip()
        if target_dir is not None:
            env["CARGO_TARGET_DIR"] = str(target_dir)
        print(
            f"check_dead_code_oracle.py [{config}]: stripped {stripped} annotation(s); "
            f"cargo {' '.join(cmd[1:])}"
        )
        proc = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True)
        if capture_path is not None:
            capture_path.parent.mkdir(parents=True, exist_ok=True)
            capture_path.write_text(proc.stdout, encoding="utf-8")
            print(f"  capture -> {capture_path}")
        if proc.returncode != 0:
            tail = "\n".join(proc.stderr.strip().splitlines()[-25:])
            raise OracleError(f"cargo {' '.join(cmd[1:])} failed (rc={proc.returncode}):\n{tail}")
        diags, sites, items = parse_capture(proc.stdout)
        if not diags:
            raise OracleError(
                "the stripped check reported no `dead_code` diagnostic at all — with every "
                "annotation stripped that cannot be true; the capture is suspect"
            )
        print(
            f"  dead_code diagnostics: {len(diags)}   src/ spans (sites): {len(sites)}   "
            f"items: {len(items)}"
        )
        baseline = load_baseline(baseline_path, allow_missing=print_toml)
        entries = baseline.get(config, [])
        base_keys = {(e["name"], e["kind"]) for e in entries}
        added = sorted(set(items) - base_keys)
        removed = sorted(base_keys - set(items))
        moved = [
            (name, kind, items[(name, kind)]["file"], e.get("file"))
            for e in entries
            for name, kind in [(e["name"], e["kind"])]
            if (name, kind) in items
            and e.get("file")
            and e["file"] != items[(name, kind)]["file"]
        ]
        if print_toml:
            print(f"\n# --- {config}: {len(items)} item(s) in the stripped oracle ---")
            by_key = {(e["name"], e["kind"]): e for e in entries}
            for name, kind in sorted(items):
                row = items[(name, kind)]
                prev = by_key.get((name, kind))
                reason = (
                    prev["reason"]
                    if prev
                    else "TODO: why is this dead item retained, and what would construct/read it? (#254)"
                )
                print(toml_entry(config, name, kind, row["file"], reason))
        for name, kind in removed:
            was = next((e.get("file", "?") for e in entries if (e["name"], e["kind"]) == (name, kind)), "?")
            print(f"  note: `{name}` ({kind}) is no longer dead here (was {was}) — drop the entry when convenient")
        for name, kind, now, then in moved:
            print(f"  note: `{name}` ({kind}) moved {then} -> {now} — update the entry's `file`")
        if added:
            print(
                f"\ncheck_dead_code_oracle.py [{config}]: {len(added)} NEW dead item(s) are not in "
                f"{baseline_path}:",
                file=sys.stderr,
            )
            for name, kind in added:
                row = items[(name, kind)]
                print(f"  {name} ({kind}) at {row['file']}:{row['line']}", file=sys.stderr)
                print(f"     {row['message']}", file=sys.stderr)
                print(
                    toml_entry(
                        config,
                        name,
                        kind,
                        row["file"],
                        "TODO: why is this retained, and what would construct/read it? (#254)",
                    ),
                    file=sys.stderr,
                )
            print(
                "  An addition is a gate failure: either make the item live, delete it, or add the "
                "entry above with a reason — the manifest is the record of what is deliberately "
                "retained (#254).",
                file=sys.stderr,
            )
            return 1
        print(
            f"  baseline [{config}]: {len(entries)} entry/entries — 0 addition(s), "
            f"{len(removed)} removal(s); PASS"
        )
        return 0
    finally:
        if made_root and not keep:
            shutil.rmtree(root, ignore_errors=True)


def selftest() -> int:
    """Pin the strip, the item derivation and the comparison against their shapes."""
    failures = 0

    def expect(name: str, got, want) -> None:
        nonlocal failures
        if got != want:
            print(f"selftest FAIL: {name}: want {want!r}, got {got!r}")
            failures += 1

    strip_cases = [
        ("bare", "#[allow(dead_code)]\n", f"{MARKER}\n", 1),
        ("bare inner", "#![allow(dead_code)]\n", f"{MARKER}\n", 1),
        ("combined keeps the others", "#[allow(unused, dead_code)]\n", "#[allow(unused)]\n", 1),
        ("combined order", "#[allow(dead_code, unused)]\n", "#[allow(unused)]\n", 1),
        ("cfg_attr", "#[cfg_attr(not(test), allow(dead_code))]\n", f"{MARKER}\n", 1),
        (
            "cfg_attr combined",
            '#[cfg_attr(not(feature = "cuda"), allow(unused, dead_code))]\n',
            '#[cfg_attr(not(feature = "cuda"), allow(unused))]\n',
            1,
        ),
        ("an unrelated allow is left alone", "#[allow(unused)]\n", "#[allow(unused)]\n", 0),
        ("prose is left alone", "// mentions allow(dead_code) in prose\n", "// mentions allow(dead_code) in prose\n", 0),
        ("line numbers are preserved", "fn a() {}\n#[allow(dead_code)]\nfn b() {}\n", f"fn a() {{}}\n{MARKER}\nfn b() {{}}\n", 1),
        (
            "a trailing comment rides along on a combined list",
            "#[allow(unused, dead_code)] // kept\n",
            "#[allow(unused)] // kept\n",
            1,
        ),
    ]
    for name, body, want_text, want_count in strip_cases:
        got_text, got_lines = strip_source(body, "fixture.rs")
        expect(f"strip: {name}", (got_text, len(got_lines)), (want_text, want_count))
    for name, body in [
        ("multi-line allow", "#[allow(\n    dead_code\n)]\n"),
        ("clippy path", "#[allow(clippy::dead_code)]\n"),
    ]:
        try:
            strip_source(body, "fixture.rs")
            expect(f"strip must refuse: {name}", "no error", "OracleError")
        except OracleError:
            pass

    expect(
        "fields are two items",
        sorted(items_of("fields `text`, `stopped_by_eog`, and `stopped_by_string` are never read")),
        [("stopped_by_eog", "field"), ("stopped_by_string", "field"), ("text", "field")],
    )
    expect(
        "variants keep their kind",
        items_of("variants `Scale`, `Softmax`, and `Reshape` are never constructed"),
        [("Scale", "variant"), ("Softmax", "variant"), ("Reshape", "variant")],
    )
    expect(
        "methods are fn",
        items_of("methods `as_any` and `forward_graph` are never used"),
        [("as_any", "fn"), ("forward_graph", "fn")],
    )
    expect("struct", items_of("struct `DeviceKey` is never constructed"), [("DeviceKey", "struct")])
    expect("constant", items_of("constant `TIERS` is never used"), [("TIERS", "const")])

    capture = "\n".join(
        [
            json.dumps({"reason": "compiler-message", "message": {"code": {"code": "dead_code"},
                "message": "function `never_called` is never used",
                "spans": [{"file_name": "src/a.rs", "line_start": 4, "is_primary": True},
                          {"file_name": "src/a.rs", "line_start": 9, "is_primary": False}]}}),
            json.dumps({"reason": "compiler-message", "message": {"code": {"code": "unused"},
                "message": "unused variable: `x`", "spans": []}}),
            json.dumps({"reason": "build-finished", "success": True}),
        ]
    )
    diags, sites, items = parse_capture(capture)
    expect("one diagnostic", len(diags), 1)
    expect("every src span is a site", sorted(sites), [("src/a.rs", 4), ("src/a.rs", 9)])
    expect("item key", sorted(items), [("never_called", "fn")])
    expect("unknown reason lines are ignored", parse_capture("not json\n")[0], [])

    if failures:
        print(f"check_dead_code_oracle.py selftest: {failures} case(s) failed")
        return 1
    print("check_dead_code_oracle.py selftest: strip, item and capture cases pass")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", choices=sorted(CONFIGS), help="the configuration to check")
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE), help="the manifest to compare against")
    parser.add_argument("--tree", default=str(REPO), help="the tree to copy and strip")
    parser.add_argument("--scratch", default=None, help="scratch root (must not already hold `tree/`)")
    parser.add_argument("--target", default=None, help="rustc target triple (architecture validation only)")
    parser.add_argument("--target-dir", default=None, help="shared target dir (default: <tree>/target)")
    parser.add_argument("--capture", default=None, help="write the raw JSON capture here")
    parser.add_argument("--keep", action="store_true", help="keep the scratch tree")
    parser.add_argument("--print-toml", action="store_true", help="print the manifest entries for this run")
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if not args.config:
        parser.error("--config is required (or use --selftest)")
    tree = Path(args.tree).resolve()
    if not tree.is_dir():
        print(f"{tree}: not a directory", file=sys.stderr)
        return 2
    target_dir = Path(args.target_dir).resolve() if args.target_dir else tree / "target"
    try:
        return check(
            tree=tree,
            config=args.config,
            baseline_path=Path(args.baseline).resolve(),
            scratch_root=Path(args.scratch).resolve() if args.scratch else None,
            target=args.target,
            target_dir=target_dir,
            capture_path=Path(args.capture).resolve() if args.capture else None,
            keep=args.keep,
            print_toml=args.print_toml,
        )
    except OracleError as e:
        print(f"check_dead_code_oracle.py: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
