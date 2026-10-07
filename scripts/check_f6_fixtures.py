#!/usr/bin/env python3
"""Audit the F6 fixture manifest, and the cached fixtures it describes (issue #205).

The F6 byte-parity gates compare `minfer quantize` output against files under
``~/.cache/minfer/f6-src/`` — an f16/f32 source and a ``llama-quantize``
reference per target. Those files are inputs the gates do not produce, so until
this checker nothing noticed when the cache held *something else*: a stale or
replaced reference was compared against silently and the gate stayed green.

``docs/f6-fixtures.json`` is the record: one entry per **artifact content
identity** — ``path``, ``bytes``, ``sha256`` (or a recorded ``sha256_prefix``
where the 2026-10-07 table truncated it), the exact ``producer`` command, the
producer's identity (``minfer_commit`` for a minfer-produced file, or
``llamacpp_commit`` + ``compiler`` + ``ffp_contract`` + ``cflags`` for a
llama-produced one), ``date`` and an **absolute** box label (gate contract rule
5). A path with more than one entry is a *recorded divergence*, and every such
path must be explained by a ``divergence_notes`` entry — the two 2026-10-07
divergences are why the manifest exists (a minfer producer-version difference on
the f16 source, and a ``-ffp-contract`` difference on the ``llama-quantize``
reference).

What it checks, ``--check`` (no cache needed, so CI runs it):

``S1`` the manifest parses and carries ``schema``, ``cache_root``, ``entries``
and ``divergence_notes``.

``S2`` every entry's shape: a relative ``path``, exactly one of ``sha256``
(64 lowercase hex) or ``sha256_prefix`` (8), a positive ``bytes`` when a full
digest is claimed, a non-empty ``producer``, a known ``producer_kind``, a
``YYYY-MM-DD`` ``date``, an absolute ``box`` label (a relative one — ``this
box``, ``local``, … — is rejected; GATE-CONTRACT rule 5), the identity fields
its kind requires, a ``producer_note`` whenever an identity field says
``unrecorded`` (a gap must be explained, never silent), and a ``digest_note``
whenever the digest is a truncated prefix.

``S3`` a ``sha256_prefix`` never contradicts a full digest for the same path —
a prefix that *is* a prefix of the recorded full digest would be a soft
duplicate, not a divergence.

``S4`` every path with more than one entry appears in a ``divergence_notes``
entry, and every path a note names exists. An unexplained divergence is the bug
this manifest is supposed to make impossible.

``S5`` the manifest and the tree agree: every ``~/.cache/minfer/f6-src/…``
fixture spelled in ``src/**.rs`` is an entry (a ``.gguf`` path is an entry
verbatim; a directory must prefix some entry), and every ``.gguf`` path a
``producer`` command names is an entry. A gate that grows a new fixture, or a
recipe that gains a step, cannot silently leave the record behind.

What it checks, ``--verify`` (needs the cache; the developer's and the F6 gate's
form): ``S1``–``S5``, then per entry ``bytes`` + ``sha256`` against the file,
and every file under the cache root that no entry names is reported as
``extra``. A full-digest mismatch is a **failure** naming the file, the expected
digest(s) and the actual one; a prefix-only entry accepts a matching prefix with
a ``WEAK`` note (and ``--strict-digests`` turns that note into a failure), and a
prefix mismatch is a failure like any other. ``--file PATH`` verifies one file —
the form the issue's tamper check uses.

**Honest limits.** This checks *content*, not truth: a manifest entry whose
recorded digest is wrong is accepted, because nothing else recorded those bytes.
The ``--verify`` half needs the cache, so CI (which has none) covers only
``--check`` and ``--selftest``; the F6 gates verify the fixtures they resolve
through ``src/tooling/tests/f6_fixtures.rs``, which is where a tampered cache
stops a run that CI cannot reach. A prefix-only entry can be defeated by a
1-in-2**32 collision — re-capturing the full digest is the fix, and the 11
entries that need it say so.

Usage::

    python3 scripts/check_f6_fixtures.py --check                    # CI: manifest only
    python3 scripts/check_f6_fixtures.py --verify                   # + the cached files
    python3 scripts/check_f6_fixtures.py --verify --require-all     # every entry present
    python3 scripts/check_f6_fixtures.py --file /tmp/x.gguf         # one file
    python3 scripts/check_f6_fixtures.py --selftest                 # the checker's own cases

Exit codes: 0 clean, 1 violations, 2 usage or I/O error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

#: The checked-in manifest, relative to the repository root.
MANIFEST = "docs/f6-fixtures.json"

#: The persistent fixture cache the manifest's `cache_root` names (the recipe in
#: `docs/GGUF-TOOLING.md` §4.2.1). Overridable for a verification run against a
#: copy, which is how the tamper case is demonstrated without touching the cache.
DEFAULT_CACHE = "~/.cache/minfer/f6-src"

#: Producer kinds, and the identity fields each requires. The field set is the
#: point of the manifest: `minfer_commit` is what settles the f16-source
#: divergence and the llama.cpp triple is what settles the q4_0 one.
REQUIRED_BY_KIND = {
    "minfer": ("minfer_commit",),
    "llama-quantize": ("llamacpp_commit", "compiler", "ffp_contract", "cflags"),
    "hf-download": ("source",),
}

#: Fields a `llama-quantize` entry may claim for `-ffp-contract`. Only these two
#: are meaningful: the flag is either on (GCC's default) or off.
FFP_CONTRACTS = ("fast", "off")

#: Keys an entry may carry — an allowlist, so a typo (`minfercommit`) is a
#: failure instead of a silently absent identity.
ENTRY_KEYS = {
    "path", "bytes", "sha256", "sha256_prefix", "producer", "producer_kind",
    "producer_note", "digest_note", "date", "box", "also_on",
    "minfer_commit", "llamacpp_commit", "llamacpp_binary", "compiler",
    "ffp_contract", "cflags", "source",
}

#: A `box` label is absolute or it is a lie the next agent cannot read (gate
#: contract rule 5). These are the relative spellings that have bitten before.
RELATIVE_BOX = (
    re.compile(r"\bthis box\b", re.I),
    re.compile(r"\blocal(host| machine)?\b", re.I),
    re.compile(r"\bhere\b", re.I),
    re.compile(r"\bmy machine\b", re.I),
)

SHA256 = re.compile(r"^[0-9a-f]{64}$")
SHA256_PREFIX = re.compile(r"^[0-9a-f]{8}$")
DATE = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}$")
#: The fixture root the source tree spells its defaults under.
SOURCE_FIXTURE = re.compile(r"f6-src/([A-Za-z0-9._/-]+)")
#: `unrecorded`, `not recorded`, `unknown` — an identity field that admits a gap.
UNRECORDED = re.compile(r"unrecord|not recorded|unknown", re.I)


# --------------------------------------------------------------------------- #
# S1-S5: the manifest alone
# --------------------------------------------------------------------------- #

def load_manifest(path: Path) -> tuple[dict | None, list[str]]:
    """Parse the manifest; ``(manifest, problems)``."""
    if not path.is_file():
        return None, [f"{path}: no such manifest (the F6 fixture record is missing)"]
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"{path}: does not parse as JSON: {exc}"]
    if not isinstance(doc, dict):
        return None, [f"{path}: the manifest must be a JSON object"]
    problems = []
    for key in ("schema", "cache_root", "entries", "divergence_notes"):
        if key not in doc:
            problems.append(f"{path}: missing top-level {key!r}")
    if doc.get("schema") != 1:
        problems.append(f"{path}: schema must be 1, got {doc.get('schema')!r}")
    if not isinstance(doc.get("entries"), list) or not doc.get("entries"):
        problems.append(f"{path}: entries must be a non-empty list")
    if not isinstance(doc.get("divergence_notes"), list):
        problems.append(f"{path}: divergence_notes must be a list")
    return doc, problems


def check_entry(path: Path, i: int, e: dict, problems: list[str]) -> None:
    """``S2`` — one entry's shape. ``where`` prefixes every message."""
    where = f"{path}: entries[{i}]"
    if not isinstance(e, dict):
        problems.append(f"{where}: not an object")
        return
    unknown = sorted(set(e) - ENTRY_KEYS)
    if unknown:
        problems.append(f"{where}: unknown field(s) {unknown} (a typo is not an identity)")
    p = e.get("path")
    if not isinstance(p, str) or not p:
        problems.append(f"{where}: path must be a non-empty string")
        return
    where = f"{path}: {p}"
    if p.startswith("/") or ".." in p.split("/") or "\\" in p:
        problems.append(f"{where}: path must be relative to cache_root and may not escape it")
    full = e.get("sha256")
    prefix = e.get("sha256_prefix")
    if (full is None) == (prefix is None):
        problems.append(f"{where}: exactly one of sha256 / sha256_prefix is required")
    if full is not None and not (isinstance(full, str) and SHA256.match(full)):
        problems.append(f"{where}: sha256 must be 64 lowercase hex, got {full!r}")
    if prefix is not None and not (isinstance(prefix, str) and SHA256_PREFIX.match(prefix)):
        problems.append(f"{where}: sha256_prefix must be 8 lowercase hex, got {prefix!r}")
    if prefix is not None and not e.get("digest_note"):
        problems.append(
            f"{where}: a truncated digest needs a digest_note saying why the full one is missing"
        )
    if full is not None:
        b = e.get("bytes")
        if not isinstance(b, int) or b <= 0:
            problems.append(f"{where}: bytes must be a positive integer with a full digest")
    elif "bytes" in e and (not isinstance(e["bytes"], int) or e["bytes"] <= 0):
        problems.append(f"{where}: bytes, when present, must be a positive integer")
    for key in ("producer", "producer_kind", "date", "box"):
        if not isinstance(e.get(key), str) or not e[key].strip():
            problems.append(f"{where}: {key} must be a non-empty string")
    kind = e.get("producer_kind")
    if kind not in REQUIRED_BY_KIND:
        problems.append(
            f"{where}: producer_kind {kind!r} is not one of {sorted(REQUIRED_BY_KIND)}"
        )
        return
    gap = False
    for key in REQUIRED_BY_KIND[kind]:
        v = e.get(key)
        if not isinstance(v, str) or not v.strip():
            problems.append(f"{where}: producer_kind {kind!r} requires {key}")
        elif UNRECORDED.search(v):
            gap = True
    if gap and not e.get("producer_note"):
        problems.append(
            f"{where}: an identity field says the producer is unrecorded, so a producer_note "
            "must say what is missing"
        )
    if kind == "llama-quantize" and e.get("ffp_contract") not in FFP_CONTRACTS:
        problems.append(
            f"{where}: ffp_contract must be one of {list(FFP_CONTRACTS)}, got "
            f"{e.get('ffp_contract')!r} (the flag this whole record turns on)"
        )
    if not DATE.match(str(e.get("date", ""))):
        problems.append(f"{where}: date must be YYYY-MM-DD, got {e.get('date')!r}")
    box = str(e.get("box", ""))
    for rx in RELATIVE_BOX:
        if rx.search(box):
            problems.append(
                f"{where}: box {box!r} is a relative label; rule 5 of docs/GATE-CONTRACT.md "
                "requires an absolute one (`<host> (<OS>, <arch>)`)"
            )
    also = e.get("also_on", [])
    if not isinstance(also, list):
        problems.append(f"{where}: also_on must be a list")
    else:
        for j, a in enumerate(also):
            if not isinstance(a, dict) or not a.get("box") or not DATE.match(str(a.get("date", ""))):
                problems.append(f"{where}: also_on[{j}] needs a box and a YYYY-MM-DD date")
            elif any(rx.search(str(a["box"])) for rx in RELATIVE_BOX):
                problems.append(f"{where}: also_on[{j}] box {a['box']!r} is a relative label")


def check_manifest(path: Path, doc: dict) -> list[str]:
    """``S2``-``S5``: every invariant the manifest alone can carry."""
    problems: list[str] = []
    entries = doc.get("entries") or []
    if not isinstance(entries, list):
        return [f"{path}: entries must be a list"]
    for i, e in enumerate(entries):
        check_entry(path, i, e, problems)
    # S3 + duplicate identities.
    by_path: dict[str, list[dict]] = {}
    for e in entries:
        if isinstance(e, dict) and isinstance(e.get("path"), str):
            by_path.setdefault(e["path"], []).append(e)
    for p, es in sorted(by_path.items()):
        seen = set()
        for e in es:
            ident = e.get("sha256") or e.get("sha256_prefix")
            if ident in seen:
                problems.append(f"{path}: {p}: duplicate identity {ident}")
            seen.add(ident)
        fulls = [e["sha256"] for e in es if e.get("sha256")]
        for e in es:
            pre = e.get("sha256_prefix")
            if pre and any(f.startswith(pre) for f in fulls):
                problems.append(
                    f"{path}: {p}: sha256_prefix {pre!r} is a prefix of the recorded full "
                    "digest — that is the same file, not a divergence"
                )
    # S4: every divergence explained, every explained path real.
    notes = doc.get("divergence_notes") or []
    explained: set[str] = set()
    for i, n in enumerate(notes):
        if not isinstance(n, dict):
            problems.append(f"{path}: divergence_notes[{i}] is not an object")
            continue
        for key in ("cause", "paths", "explanation"):
            if not n.get(key):
                problems.append(f"{path}: divergence_notes[{i}] needs a non-empty {key}")
        for p in n.get("paths") or []:
            explained.add(p)
            if p not in by_path:
                problems.append(f"{path}: divergence_notes[{i}] names {p!r}, which no entry has")
    for p, es in sorted(by_path.items()):
        if len(es) > 1 and p not in explained:
            problems.append(
                f"{path}: {p} has {len(es)} recorded contents but no divergence_notes entry — "
                "an unexplained divergence is the failure this manifest exists to prevent"
            )
    return problems


def check_source_references(root: Path, doc: dict, problems: list[str]) -> None:
    """``S5``: the tree's fixture paths and the manifest's producers agree."""
    known = {e["path"] for e in doc.get("entries", []) if isinstance(e, dict) and e.get("path")}
    src = root / "src"
    if not src.is_dir():
        problems.append(f"{src}: no src/ to cross-check against")
        return
    for f in sorted(src.rglob("*.rs")):
        text = f.read_text(encoding="utf-8", errors="replace")
        for m in SOURCE_FIXTURE.finditer(text):
            rel = m.group(1).rstrip("/")
            if rel in known:
                continue
            if rel.endswith(".gguf"):
                problems.append(
                    f"{f.relative_to(root)}: fixture {rel!r} is used by a gate but has no "
                    "manifest entry (docs/f6-fixtures.json)"
                )
            elif not any(k.startswith(rel + "/") for k in known):
                problems.append(
                    f"{f.relative_to(root)}: directory {rel!r} holds no manifest entry"
                )
    for e in doc.get("entries", []):
        if not isinstance(e, dict):
            continue
        for m in SOURCE_FIXTURE.finditer(str(e.get("producer", ""))):
            rel = m.group(1).rstrip("/")
            if rel.endswith(".gguf") and rel not in known:
                problems.append(
                    f"{MANIFEST}: {e.get('path')}: its producer names {rel!r}, which no entry "
                    "records — the recipe and the record disagree"
                )


# --------------------------------------------------------------------------- #
# --verify: the bytes
# --------------------------------------------------------------------------- #

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def expected(entries: list[dict]) -> str:
    """The recorded digests for one path, as a message fragment."""
    out = []
    for e in entries:
        if e.get("sha256"):
            out.append(f"{e['sha256']} ({e['box']}, {e['date']})")
        else:
            out.append(f"{e['sha256_prefix']}… (truncated: {e['box']}, {e['date']})")
    return "; ".join(out)


def verify_file(
    path: Path,
    entries: list[dict],
    problems: list[str],
    weak: list[str],
    missing: list[str],
    strict: bool,
) -> str | None:
    """Verify one file against its entries; the actual digest, or ``None`` if absent.

    A full-digest match wins outright. Failing that, a matching ``sha256_prefix``
    is a ``WEAK`` acceptance — the digest the record actually holds — and
    ``--strict-digests`` promotes that note to a failure. Nothing matches: the
    message names the file, every recorded digest and the actual one.
    """
    if not path.is_file():
        # A cache is per box: the Mac's entries are legitimately absent here. Only
        # --require-all turns that into a failure.
        missing.append(entries[0]["path"])
        return None
    actual = sha256_file(path)
    size = path.stat().st_size
    for e in entries:
        if e.get("sha256") and e["sha256"] == actual:
            if "bytes" in e and e["bytes"] != size:
                problems.append(
                    f"{path}: sha256 matches but bytes is {size}, the entry says {e['bytes']}"
                )
            return actual
    sized = [e for e in entries if e.get("bytes") is not None]
    if sized and all(e["bytes"] != size for e in sized):
        problems.append(
            f"{path}: {size} bytes; the record has {[e['bytes'] for e in sized]} "
            f"({expected(entries)})"
        )
        return actual
    for e in entries:
        pre = e.get("sha256_prefix")
        if pre and not actual.startswith(pre):
            continue
        if pre:
            note = f"{path}: sha256 {actual} matches only the recorded prefix {pre}…"
            if strict:
                problems.append(f"{note} (--strict-digests)")
            else:
                weak.append(note)
            return actual
    problems.append(
        f"{path}: sha256 {actual} is not the content the record describes "
        f"(recorded: {expected(entries)}; actual: {actual})"
    )
    return actual


def resolve(path: Path, by_path: dict[str, list[dict]], root: Path) -> list[dict]:
    """The entries that describe ``path``: by cache-relative path, else basename."""
    try:
        rel = path.resolve().relative_to(root.resolve()).as_posix()
        if rel in by_path:
            return by_path[rel]
    except (OSError, ValueError):
        pass
    # A copy outside the cache (the tamper check's `/tmp` copy) is recognised by
    # its basename. The `--file` form is explicit, so a basename collision is the
    # caller's to notice — the whole-cache form never guesses this way.
    return [e for es in by_path.values() for e in es
            if e["path"].rsplit("/", 1)[-1] == path.name]


def verify_cache(
    root: Path, doc: dict, only: Path | None, strict: bool, verbose: bool = True
) -> tuple[list[str], list[str]]:
    """``(problems, missing)`` — ``missing`` is a per-box report, not a failure."""
    problems: list[str] = []
    weak: list[str] = []
    missing: list[str] = []
    entries = [e for e in doc.get("entries", []) if isinstance(e, dict)]
    by_path: dict[str, list[dict]] = {}
    for e in entries:
        by_path.setdefault(e["path"], []).append(e)
    if only is not None:
        if not only.is_file():
            return [f"{only}: no such file"], []
        if not resolve(only, by_path, root):
            return [f"{only}: no manifest entry describes this file (basename or cache path)"], []
        verify_file(only, resolve(only, by_path, root), problems, weak, missing, strict)
    else:
        for p, es in sorted(by_path.items()):
            verify_file(root / p, es, problems, weak, missing, strict)
        if root.is_dir():
            known = set(by_path)
            extras = []
            for f in sorted(root.rglob("*")):
                if not f.is_file():
                    continue
                rel = f.relative_to(root).as_posix()
                if rel not in known:
                    extras.append(rel)
            if extras and verbose:
                print(
                    f"check_f6_fixtures: {len(extras)} file(s) under {root} are not in "
                    f"{MANIFEST} (reported, not a failure — a gate cannot resolve them): "
                    f"{', '.join(extras[:6])}{' …' if len(extras) > 6 else ''}"
                )
            if extras and not verbose and strict:
                problems.append(
                    f"{root}: --strict-extra: {len(extras)} unrecorded file(s): "
                    f"{', '.join(extras[:6])}"
                )
        else:
            problems.append(f"{root}: the fixture cache does not exist")
    if verbose:
        for w in weak:
            print(f"check_f6_fixtures: WEAK {w}")
        if missing:
            print(
                f"check_f6_fixtures: {len(missing)} recorded content(s) are not on this box "
                f"(a cache is per box; --require-all turns this into a failure): "
                f"{', '.join(missing[:6])}{' …' if len(missing) > 6 else ''}"
            )
    return problems, missing


# --------------------------------------------------------------------------- #
# --selftest
# --------------------------------------------------------------------------- #

def _tiny_manifest(root: Path, sha: str, **over) -> dict:
    e = {
        "path": "ref/tiny.gguf",
        "bytes": 4,
        "sha256": sha,
        "producer": "llama-quantize src.gguf ref/tiny.gguf q4_0",
        "producer_kind": "llama-quantize",
        "llamacpp_commit": "deadbeef",
        "compiler": "gcc 13.3.0",
        "ffp_contract": "fast",
        "cflags": "-O3 -DNDEBUG",
        "date": "2026-10-07",
        "box": "dgxspark (aarch64, GB10 sm_121)",
    }
    e.update(over)
    return {
        "schema": 1,
        "cache_root": "~/.cache/minfer/f6-src",
        "divergence_notes": [],
        "entries": [e],
    }


def selftest() -> int:
    """The checker's own cases, including the tamper case the issue asks for."""
    cases = 0
    failures = 0

    def record(name: str, ok: bool, detail: str = "") -> None:
        nonlocal cases, failures
        cases += 1
        if not ok:
            failures += 1
            print(f"selftest FAIL: {name}{': ' + detail if detail else ''}")

    tmp = Path(tempfile.mkdtemp(prefix="f6-fixtures-selftest-"))
    try:
        root = tmp / "cache"
        (root / "ref").mkdir(parents=True)
        payload = b"tiny"
        (root / "ref" / "tiny.gguf").write_bytes(payload)
        good = hashlib.sha256(payload).hexdigest()
        manifest_path = tmp / "f6-fixtures.json"

        doc = _tiny_manifest(root, good)
        manifest_path.write_text(json.dumps(doc), encoding="utf-8")
        problems = check_manifest(manifest_path, doc)
        record("a well-formed manifest has no structural problems", not problems, str(problems))

        problems, _ = verify_cache(root, doc, None, strict=False, verbose=False)
        record("the recorded file verifies", not problems, str(problems))

        # The tamper case: one byte flipped in a copy must be named with both digests.
        (root / "ref" / "tiny.gguf").write_bytes(b"tinv")
        actual = hashlib.sha256(b"tinv").hexdigest()
        problems, _ = verify_cache(root, doc, None, strict=False, verbose=False)
        named = problems and good in problems[0] and actual in problems[0]
        record("a one-byte tamper fails naming the file and both digests", bool(named), str(problems))
        record("the tamper message names the file", bool(problems) and "tiny.gguf" in problems[0])

        # An unknown file under the root is reported (and is not a failure).
        (root / "ref" / "tiny.gguf").write_bytes(payload)
        (root / "ref" / "unknown.gguf").write_bytes(b"x")
        problems, _ = verify_cache(root, doc, None, strict=False, verbose=False)
        record("an unrecorded file under the root does not fail the verify", not problems, str(problems))

        # A prefix-only entry: weak by default, a failure under --strict-digests.
        pre = _tiny_manifest(root, None)
        del pre["entries"][0]["sha256"]
        pre["entries"][0]["sha256_prefix"] = good[:8]
        pre["entries"][0]["digest_note"] = "recorded as a prefix only"
        problems, _ = verify_cache(root, pre, None, strict=False, verbose=False)
        record("a matching prefix is accepted weakly", not problems, str(problems))
        problems, _ = verify_cache(root, pre, None, strict=True, verbose=False)
        record("--strict-digests promotes that weak note to a failure", bool(problems))
        pre["entries"][0]["sha256_prefix"] = "0" * 8
        problems, _ = verify_cache(root, pre, None, strict=False, verbose=False)
        record("a prefix mismatch fails", bool(problems))

        # --file resolves by basename, which is how a /tmp tamper copy is checked.
        copy = tmp / "elsewhere" / "tiny.gguf"
        copy.parent.mkdir()
        copy.write_bytes(b"tinv")
        problems, _ = verify_cache(root, doc, copy, strict=False, verbose=False)
        record("--file on a tampered copy outside the root fails", bool(problems), str(problems))

        # Structure: each rejection the manifest must produce.
        bad = _tiny_manifest(root, good)
        del bad["entries"][0]["compiler"]
        manifest_path.write_text(json.dumps(bad), encoding="utf-8")
        record("a missing identity field is a structural problem",
               bool(check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, "abc")
        record("a short digest is a structural problem", bool(check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, box="this box")
        record("a relative box label is a structural problem",
               bool(check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, ffp_contract="on")
        record("an unknown -ffp-contract value is a structural problem",
               bool(check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, minfer_commit="unrecorded")
        bad["entries"][0]["producer_kind"] = "minfer"
        record("an unrecorded producer without a note is a structural problem",
               bool(check_manifest(manifest_path, bad)))
        bad["entries"][0]["producer_note"] = "not recorded at the time"
        record("... and is accepted once the note explains it",
               not check_manifest(manifest_path, bad))
        bad = _tiny_manifest(root, good)
        soft = dict(bad["entries"][0])
        del soft["sha256"]
        soft["sha256_prefix"] = good[:8]
        soft["digest_note"] = "recorded as a prefix only"
        bad["entries"].append(soft)
        record("a prefix that is a prefix of the full digest for the same path is rejected",
               any("not a divergence" in p for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good)
        bad["entries"].append(dict(bad["entries"][0], sha256="f" * 64))
        record("two contents for one path need a divergence note",
               any("divergence_notes" in p for p in check_manifest(manifest_path, bad)))
        bad["divergence_notes"] = [
            {"cause": "a build flag", "paths": ["ref/tiny.gguf"], "explanation": "why"}
        ]
        record("... and pass once the note explains them",
               not [p for p in check_manifest(manifest_path, bad) if "divergence" in p])
        bad["divergence_notes"][0]["paths"] = ["ref/ghost.gguf"]
        record("a note naming an unknown path is rejected",
               any("no entry has" in p for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good)
        bad["entries"][0]["minfercommit"] = "typo"
        record("an unknown field is rejected", bool(check_manifest(manifest_path, bad)))
        record("a missing manifest is a problem",
               bool(load_manifest(tmp / "nope.json")[1]))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if failures:
        print(f"check_f6_fixtures.py selftest: {failures} case(s) failed")
        return 1
    print(f"check_f6_fixtures.py selftest: {cases} cases pass (the tamper case names both digests)")
    return 0


# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", default=None, help=f"the manifest (default {MANIFEST})")
    parser.add_argument("--root", default=None, help=f"the fixture cache (default {DEFAULT_CACHE})")
    parser.add_argument("--file", default=None, help="verify one file instead of the whole cache")
    parser.add_argument("--verify", action="store_true", help="also check the cached bytes")
    parser.add_argument("--require-all", action="store_true",
                        help="with --verify: every entry must be present in the cache")
    parser.add_argument("--strict-digests", action="store_true",
                        help="with --verify: a truncated (prefix-only) digest is a failure")
    parser.add_argument("--strict-extra", action="store_true",
                        help="with --verify: an unrecorded file under the root is a failure")
    parser.add_argument("--check", action="store_true", help="manifest structure only (the default)")
    parser.add_argument("--selftest", action="store_true", help="run the checker's own cases")
    args = parser.parse_args()
    if args.selftest:
        return selftest()

    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest_path = Path(args.manifest) if args.manifest else root / MANIFEST
    doc, problems = load_manifest(manifest_path)
    if doc is None:
        for p in problems:
            print(p, file=sys.stderr)
        return 1
    problems += check_manifest(manifest_path, doc)
    check_source_references(root, doc, problems)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        print(f"check_f6_fixtures.py: {len(problems)} manifest problem(s)", file=sys.stderr)
        return 1

    cache = Path(os.path.expanduser(args.root or doc.get("cache_root") or DEFAULT_CACHE))
    if args.file:
        args.verify = True
    if args.verify:
        before = len(problems)
        if not cache.is_dir() and not args.file:
            print(f"check_f6_fixtures.py: {cache} does not exist; nothing to verify "
                  f"(the manifest's {len(doc['entries'])} entries are the record)")
            return 0
        found_problems, missing = verify_cache(
            cache, doc, Path(args.file) if args.file else None, args.strict_digests
        )
        problems += found_problems
        if args.require_all and missing:
            problems.append(
                f"{cache}: --require-all: {len(missing)} recorded fixture(s) absent: "
                f"{', '.join(missing[:6])}"
            )
        for p in problems[before:]:
            print(p, file=sys.stderr)
        if len(problems) > before:
            print(f"check_f6_fixtures.py: {len(problems) - before} fixture problem(s) under {cache}",
                  file=sys.stderr)
            return 1
        print(f"check_f6_fixtures.py: verified {len(doc['entries'])} recorded content(s) "
              f"against {cache} — every present file is the one the record describes")
        return 0

    print(f"check_f6_fixtures.py: {len(doc['entries'])} entries, "
          f"{len(doc['divergence_notes'])} explained divergence(s) — structure only, "
          "pass --verify on a box with the fixture cache to check the bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
