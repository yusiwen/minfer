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

``S7`` every ``producer`` that *is* a command runs the program its
``producer_kind`` names and **writes the entry's own path**: a ``minfer``
producer whose argv is a ``llama-quantize`` invocation, or one that writes
somewhere else under the cache root, is a record that cannot be re-run. A
``producer`` that is not a command (a bare ``unrecorded …`` note) must carry a
``producer_note``. That is the shape ``--regenerate`` relies on, and unlike the
rest of the manifest it needs no bytes — so CI pins it.

``S6`` every path whose record includes a ``llama-quantize`` entry has **exactly
one** entry marked ``authoritative_reference``, it is a ``llama-quantize`` entry
built with ``-ffp-contract=fast``, and the field appears nowhere else. That entry
is *the reference the byte-parity claim is asserted against* (docs/
GGUF-TOOLING.md §4.2): a ``ref/…`` path has two recorded contents because the two
boxes' ``llama-quantize`` builds differ, and the claim was measured against the
dgxspark GCC one. Without the mark, a mismatch cannot say whether the gate is
looking at the reference the claim is about or at a different compiler's build of
the same source — which is the whole of issue #349.

What it checks, ``--verify`` (needs the cache; the developer's and the F6 gate's
form): ``S1``–``S6``, then per entry ``bytes`` + ``sha256`` against the file,
and every file under the cache root that no entry names is reported as
``extra``. A full-digest mismatch is a **failure** naming the file, the expected
digest(s) and the actual one; a prefix-only entry accepts a matching prefix with
a ``WEAK`` note (and ``--strict-digests`` turns that note into a failure), and a
prefix mismatch is a failure like any other. ``--file PATH`` verifies one file —
the form the issue's tamper check uses.

**Honest limits.** This checks *content*, not truth: a manifest entry whose
recorded digest is wrong is accepted, because nothing else recorded those bytes.
The ``--verify`` half needs the cache, so CI (which has none) covers only
``--check``, ``--selftest`` and ``--regenerate --dry-run``; the F6 gates verify
the fixtures they resolve through ``src/tooling/tests/f6_fixtures.rs``, which is
where a tampered cache stops a run that CI cannot reach. A prefix-only entry can
be defeated by a 1-in-2**32 collision — re-capturing the full digest is the fix,
and the 11 entries that need it say so.

``--regenerate`` (issue #345) makes the recipe runnable instead of hand-run:
rather than ``--verify`` merely *checking* the cached bytes against the record, it
re-runs the recorded ``producer`` command for the selected entries and re-records
the content identity (``sha256``, ``bytes``, ``date``) from what the run wrote.
The mode is a **verification gate first**:

* the producer program must be present here — the recorded command's program
  (``minfer``, resolved from ``PATH`` and then ``./target/release/minfer``;
  ``curl`` for an ``hf-download`` entry) is looked up before anything runs, and a
  command that cannot run is **refused by name**, never silently replaced by
  another producer and never run with a substituted binary;
* **regenerating a ``llama-quantize`` reference is out of scope** — the
  byte-parity claim is about one specific compiler's build (§4.2.1), and this mode
  refuses such an entry naming the ``llamacpp_binary`` it records (and whether
  that binary exists here). It must be re-run on the box that records it, with
  that build;
* the run's content must **match the record**: a produced digest that equals an
  entry's full ``sha256`` re-records that entry (the date of the reproducing run;
  nothing else changes) and is idempotent — a second run the same day writes
  nothing. A produced digest that matches **no** recorded content for the path is
  a **finding**, not an update (exit 1): the record's producer no longer
  reproduces the record. When the entry names a producer identity (``minfer_commit``)
  that differs from what runs here the run is **refused** naming both commits and
  both digests, because the run is not the producer the entry names — that is the
  cross-box/cross-commit case, and recording it needs the identity field moved by
  hand first;
* it writes **only** ``sha256``/``bytes``/``date``: the ``authoritative_reference``
  mark, the producer command and every identity field are left exactly as they
  were, so the mode can neither invent provenance nor move the content the
  byte-parity claim is asserted against. Every write is printed, and the manifest
  is re-validated (``S1``-``S7``) after the write;
* the existing cache file is moved aside (``<path>.regen-before``, a rename, so no
  second copy is made) before the producer runs; a finding, a refusal after a run
  or a producer failure restores it and keeps the rejected content at
  ``<path>.regen-rejected``, so a rejected run cannot destroy a fixture the gates
  need.

``--dry-run`` classifies every selected entry without running anything — ``RUN``
with the ``~``-expanded command, or ``REFUSED`` with the reason — which is what CI
runs: a manifest whose command has the wrong shape, or whose producer is a
program this mode does not know, fails it.

Usage::

    python3 scripts/check_f6_fixtures.py --check                    # CI: manifest only
    python3 scripts/check_f6_fixtures.py --verify                   # + the cached files
    python3 scripts/check_f6_fixtures.py --verify --require-all     # every entry present
    python3 scripts/check_f6_fixtures.py --file /tmp/x.gguf         # one file
    python3 scripts/check_f6_fixtures.py --selftest                 # the checker's own cases

    # #354: audit another record — --manifest, or MINFER_F6_MANIFEST when it is
    # set (the manifest-side twin of MINFER_F6_CACHE). An empty override is a
    # usage error, and --regenerate refuses the variable outright.
    MINFER_F6_MANIFEST=/tmp/f6-fixtures.json python3 scripts/check_f6_fixtures.py --check
    python3 scripts/check_f6_fixtures.py --check --manifest /tmp/f6-fixtures.json

    # #345: re-run a recorded producer and re-record the content identity
    python3 scripts/check_f6_fixtures.py --regenerate --dry-run     # CI: classify, run nothing
    python3 scripts/check_f6_fixtures.py --regenerate --only qwen2.5-0.5b-instruct-f32.gguf
    python3 scripts/check_f6_fixtures.py --regenerate --box 'dgxspark (aarch64, GB10 sm_121)'

Exit codes: 0 clean, 1 violations (a --regenerate finding, or a producer failure),
2 usage or I/O error, 3 refused (a --regenerate target whose producer cannot run
here, or a --strict-runnable dry run that has one).
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import hashlib
import io
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

#: The checked-in manifest, relative to the repository root.
MANIFEST = "docs/f6-fixtures.json"

#: The manifest's own relocation (issue #354), the manifest-side twin of the
#: cache override below: it selects which record is audited. ``--manifest`` wins
#: over it; ``--regenerate`` refuses it, because re-recording into a manifest
#: chosen by the environment is the accident the override must not create. A
#: value that is set but empty is refused rather than read as "unset" — auditing
#: the tracked record while the operator believes another was selected is the
#: wrong-file failure the override exists to prevent.
MANIFEST_ENV = "MINFER_F6_MANIFEST"

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

#: The program each `producer_kind`'s recorded command must run (S7). The
#: basename is compared, so `minfer` and `./target/release/minfer` are the same
#: producer while a `minfer`-kind entry that records a `llama-quantize` command is
#: rejected. `--regenerate` resolves the same three programs.
PRODUCER_PROGRAM = {
    "minfer": ("minfer",),
    "llama-quantize": ("llama-quantize",),
    "hf-download": ("curl",),
}

#: The subcommands a recorded `minfer` producer may name (`--regenerate` runs the
#: command verbatim; this list only lets the argument parser find the output path).
MINFER_SUBCOMMANDS = ("convert", "quantize", "split")

#: Keys an entry may carry — an allowlist, so a typo (`minfercommit`) is a
#: failure instead of a silently absent identity.
ENTRY_KEYS = {
    "path", "bytes", "sha256", "sha256_prefix", "producer", "producer_kind",
    "producer_note", "digest_note", "date", "box", "also_on",
    "minfer_commit", "llamacpp_commit", "llamacpp_binary", "compiler",
    "ffp_contract", "cflags", "source", "authoritative_reference",
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

def resolve_manifest(arg: str | None, env: str | None, root: Path) -> tuple[Path, str, list[str]]:
    """The manifest to audit, and where the choice came from (issue #354).

    ``--manifest`` > ``MINFER_F6_MANIFEST`` > the tracked ``docs/f6-fixtures.json``.
    Returns ``(path, source, problems)`` with ``source`` one of ``"argv"``,
    ``"env"``, ``"default"``. An environment value that is set but blank is a
    usage problem, never a silent fall back to the tracked record: the whole point
    of the override is that the checker audits the record it was pointed at, or
    says why it cannot.
    """
    if arg:
        return Path(arg), "argv", []
    if env is not None:
        if not env.strip():
            return root / MANIFEST, "env", [
                f"{MANIFEST_ENV} is set but empty: an empty override is refused, not read as "
                f"unset. Auditing {MANIFEST} while the operator believes another record was "
                f"selected is the wrong-file failure this override exists to prevent (issue "
                f"#354). Unset {MANIFEST_ENV}, or name a path."
            ]
        return Path(env), "env", []
    return root / MANIFEST, "default", []


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
    if "authoritative_reference" in e:
        if not isinstance(e["authoritative_reference"], bool):
            problems.append(
                f"{where}: authoritative_reference must be true or false, got "
                f"{e['authoritative_reference']!r}"
            )
        elif kind != "llama-quantize":
            problems.append(
                f"{where}: authoritative_reference is only meaningful on a llama-quantize "
                f"entry (the byte-parity claim is about that build), not {kind!r}"
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


def split_command(text: str) -> list[str] | None:
    """The argv of a recorded producer, or ``None`` when the text is not a command.

    ``None`` is the honest answer for ``unrecorded (byte-identical to …)``: an
    absent command is a gap the record has to admit, not one to guess at.
    """
    try:
        argv = shlex.split(text)
    except ValueError:
        return None
    return argv or None


def output_path(argv: list[str]) -> str | None:
    """The path a recorded producer command writes, or ``None`` if unreadable.

    The rules are per program, because the recorded shapes are
    ``minfer convert|quantize|split IN OUT [--flag VALUE]`` (the subcommand is not
    a filename), ``llama-quantize [--pure] IN OUT TYPE`` and ``curl [-sL] -o OUT
    URL``. A shape the rules do not cover returns ``None`` — never a guess.
    """
    prog = os.path.basename(argv[0])
    args = argv[1:]
    if prog == "curl":
        for i, a in enumerate(args):
            if a == "-o" and i + 1 < len(args):
                return args[i + 1]
        return None
    if prog == "minfer":
        if not args or args[0] not in MINFER_SUBCOMMANDS:
            return None
        positional: list[str] = []
        skip_value = False
        for a in args[1:]:
            if skip_value:
                skip_value = False
                continue
            if a.startswith("--"):
                # `--flag=value` carries its own value; `--flag value` does not.
                skip_value = "=" not in a
                continue
            if a.startswith("-"):
                continue
            positional.append(a)
        return positional[1] if len(positional) > 1 else None
    if prog == "llama-quantize":
        positional = [a for a in args if not a.startswith("-")]
        return positional[1] if len(positional) > 1 else None
    return None


def expand_path_token(token: str, old_root: Path | None, new_root: Path | None) -> str:
    """``~``-expand a recorded path, relocating the cache root when asked.

    ``MINFER_F6_CACHE``/``--root`` relocate the cache for an experiment (the
    docstring of the checker says so), so a command that names the recorded cache
    root has to follow it — and only that prefix, never a bare path rewrite.
    """
    p = os.path.expanduser(token)
    if old_root is not None and new_root is not None and old_root != new_root:
        op, np_ = str(old_root), str(new_root)
        if p == op:
            return np_
        if p.startswith(op + os.sep):
            return np_ + p[len(op):]
    return p


def check_command_shape(
    manifest: Path, i: int, e: dict, cache_root: str | None, problems: list[str]
) -> None:
    """``S7``: the producer command runs the kind's program and writes the entry."""
    where = f"{manifest}: entries[{i}]"
    p = e.get("path")
    if not isinstance(p, str) or not p:
        return
    where = f"{manifest}: {p}"
    argv = split_command(str(e.get("producer", "")))
    out = output_path(argv) if argv else None
    if out is None:
        if not e.get("producer_note"):
            problems.append(
                f"{where}: the producer is not a runnable command and carries no producer_note "
                "— a record that cannot be re-run has to say why"
            )
        return
    prog = os.path.basename(argv[0])
    allowed = PRODUCER_PROGRAM.get(e.get("producer_kind"), ())
    if prog not in allowed:
        problems.append(
            f"{where}: producer_kind {e.get('producer_kind')!r} runs {prog!r}; the recorded "
            f"command must run {' or '.join(allowed)} (S7)"
        )
    if not isinstance(cache_root, str) or not cache_root:
        return
    expected = os.path.normpath(os.path.join(os.path.expanduser(cache_root), p))
    actual = os.path.normpath(os.path.expanduser(out))
    if actual != expected:
        problems.append(
            f"{where}: the recorded command writes {out!r}, not the entry's own path — the "
            "record and the producer disagree about the output (S7)"
        )


def check_manifest(path: Path, doc: dict) -> list[str]:
    """``S2``-``S5``: every invariant the manifest alone can carry."""
    problems: list[str] = []
    entries = doc.get("entries") or []
    if not isinstance(entries, list):
        return [f"{path}: entries must be a list"]
    for i, e in enumerate(entries):
        check_entry(path, i, e, problems)
        if isinstance(e, dict):
            check_command_shape(path, i, e, doc.get("cache_root"), problems)
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
    # S6: exactly one authoritative reference per path a llama-quantize record
    # describes (issue #349). The byte-parity claim is asserted against the build
    # that one entry names; without it, a mismatch cannot be read.
    for p, es in sorted(by_path.items()):
        marks = [e for e in es if e.get("authoritative_reference") is True]
        if not marks:
            if any(e.get("producer_kind") == "llama-quantize" for e in es):
                problems.append(
                    f"{path}: {p} has a llama-quantize record but no entry marked "
                    "authoritative_reference — the byte-parity claim does not record which "
                    "compiler's build it is asserted against (issue #349)"
                )
            continue
        if len(marks) > 1:
            problems.append(
                f"{path}: {p} marks {len(marks)} entries authoritative_reference; exactly one "
                "content is the reference the byte-parity claim is about"
            )
        for e in marks:
            if e.get("producer_kind") != "llama-quantize":
                problems.append(
                    f"{path}: {p}: the authoritative_reference entry is "
                    f"{e.get('producer_kind')!r}, not a llama-quantize build"
                )
            if e.get("ffp_contract") != "fast":
                problems.append(
                    f"{path}: {p}: the authoritative_reference entry is a "
                    f"-ffp-contract={e.get('ffp_contract')!r} build; the byte-parity claim is "
                    "asserted against the contracting build (docs/GGUF-TOOLING.md §4.2)"
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
# --regenerate: re-run a recorded producer and re-record the content identity
# --------------------------------------------------------------------------- #

def resolve_program(program: str, repo_root: Path) -> str | None:
    """The executable a recorded command names, or ``None`` when it is not here.

    A program with a path separator is resolved against the repository root (that
    is where ``./target/release/minfer`` lives), a bare one against ``PATH``, and
    ``minfer`` falls back to the repo's own release binary. There is deliberately
    **no fallback between producers**: a missing ``llama-quantize`` is a refusal,
    never a reason to run ``minfer`` instead.
    """
    raw = os.path.expanduser(program)
    if os.sep in raw:
        cand = Path(raw) if os.path.isabs(raw) else repo_root / raw
        return str(cand) if cand.is_file() and os.access(cand, os.X_OK) else None
    found = shutil.which(raw)
    if found:
        return found
    if raw == "minfer":
        cand = repo_root / "target" / "release" / "minfer"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return None


def program_candidates(program: str, repo_root: Path) -> str:
    """Where :func:`resolve_program` looked — the refusal names the missing binary."""
    raw = os.path.expanduser(program)
    if os.sep in raw:
        cand = Path(raw) if os.path.isabs(raw) else repo_root / raw
        return f"looked for {cand}"
    if raw == "minfer":
        return f"looked for `minfer` on PATH and at {repo_root / 'target' / 'release' / 'minfer'}"
    return f"looked for `{raw}` on PATH"


def running_minfer_commit(repo_root: Path, env: dict) -> str | None:
    """The minfer commit that runs here: ``MINFER_F6_COMMIT``, else the tree's HEAD."""
    override = env.get("MINFER_F6_COMMIT")
    if override:
        return override.strip() or None
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    out = proc.stdout.strip()
    return out if proc.returncode == 0 and out else None


def recorded_commit(entry: dict) -> tuple[str | None, str]:
    """``(sha or None, the raw field)`` for an entry's recorded minfer commit.

    ``None`` means the entry admits the gap (``unrecorded (2026-09-27 producer)``):
    there is no identity to check the run against, which is why a differing
    content is then an unexcused finding rather than a producer-version one.
    """
    raw = str(entry.get("minfer_commit", ""))
    if not raw or UNRECORDED.search(raw):
        return None, raw
    m = re.search(r"[0-9a-f]{7,40}", raw)
    return (m.group(0) if m else None), raw


def commits_agree(a: str, b: str) -> bool:
    return a.startswith(b) or b.startswith(a)


def restore_fixture(out_path: Path, backup: Path, rejected: Path) -> None:
    """Put the pre-run content back, keeping the rejected bytes beside it."""
    if out_path.exists():
        if rejected.exists():
            rejected.unlink()
        os.replace(out_path, rejected)
    if backup.exists():
        os.replace(backup, out_path)


def regenerate(
    doc: dict,
    manifest_path: Path,
    cache_root: Path,
    repo_root: Path,
    only: str | None = None,
    box: str | None = None,
    dry_run: bool = False,
    strict_runnable: bool = False,
    running_commit: str | None = None,
    today: str | None = None,
) -> int:
    """``--regenerate``: re-run the recorded producers and re-record what they wrote.

    Returns the process exit code (0 clean, 1 a finding or a producer failure,
    2 usage/I/O, 3 refused). Every write is printed; ``authoritative_reference``,
    the producer command and every identity field are never touched.
    """
    entries = [e for e in doc.get("entries", []) if isinstance(e, dict)]
    if running_commit is None:
        running_commit = running_minfer_commit(repo_root, os.environ)
    if today is None:
        today = datetime.date.today().isoformat()
    record_root = Path(os.path.expanduser(doc.get("cache_root") or str(cache_root)))

    selected: list[int] = []
    for i, e in enumerate(entries):
        if only is not None:
            want_path, _, want_box = only.partition("@")
            if e.get("path") != want_path:
                continue
            if want_box and e.get("box") != want_box:
                continue
        if box is not None:
            boxes = [e.get("box")] + [a.get("box") for a in e.get("also_on", [])
                                      if isinstance(a, dict)]
            if box not in boxes:
                continue
        selected.append(i)
    if not selected:
        wanted = f"--only {only}" if only else f"--box {box}"
        print(f"check_f6_fixtures.py: {wanted} matches no manifest entry", file=sys.stderr)
        return 2

    # One producer command per path: the entries of a recorded divergence share
    # the command and can only be told apart by the content the run produced.
    groups: dict[str, list[int]] = {}
    for i in selected:
        groups.setdefault(entries[i]["path"], []).append(i)

    runs = refusals = findings = changes = 0
    for path, idxs in groups.items():
        first = entries[idxs[0]]
        argv = split_command(str(first.get("producer", "")))
        if not argv or output_path(argv) is None:
            refusals += 1
            print(f"REFUSED  {path}: the producer is not a recorded command "
                  f"({first.get('producer')!r}) — there is nothing to run here")
            continue
        prog = os.path.basename(argv[0])
        kind = first.get("producer_kind")
        if prog not in PRODUCER_PROGRAM.get(kind, ()):
            refusals += 1
            print(f"REFUSED  {path}: producer_kind {kind!r} records a {prog!r} command "
                  "(S7: the record and the producer disagree)")
            continue
        if any(not entries[i].get("sha256") for i in idxs):
            refusals += 1
            print(f"REFUSED  {path}: an entry records only a truncated digest "
                  "(sha256_prefix), so the run cannot be verified against it — re-capture the "
                  "full digest first")
            continue
        if kind == "minfer":
            blocked = False
            for i in idxs:
                sha, raw = recorded_commit(entries[i])
                if sha and running_commit and not commits_agree(sha, running_commit):
                    refusals += 1
                    blocked = True
                    print(
                        f"REFUSED  {path} [{entries[i].get('box')}]: the entry names minfer "
                        f"commit {raw!r} but {running_commit} is what runs here — this run is not "
                        "the producer the entry names, so its bytes cannot be recorded against "
                        f"it; re-record the identity by hand first, or select the content you can "
                        f"verify with --only '{path}@{entries[i].get('box')}'"
                    )
            if blocked:
                continue
        if kind == "llama-quantize":
            binary = os.path.expanduser(str(first.get("llamacpp_binary") or prog))
            refusals += 1
            here = "is present here" if resolve_program(binary, repo_root) else "is not present here"
            print(f"REFUSED  {path}: the producer is a llama.cpp build ({binary}), which {here} — "
                  f"recorded on {first.get('box')}; regenerating a llama.cpp reference is out of "
                  "scope (docs/GGUF-TOOLING.md §4.2.2), because the byte-parity claim is about that "
                  "one compiler's build: re-run it on that box and record the digest by hand")
            continue

        resolved = resolve_program(argv[0], repo_root)
        if resolved is None:
            refusals += 1
            print(f"REFUSED  {path}: the producer program {argv[0]!r} is not present here "
                  f"({program_candidates(argv[0], repo_root)}) — it cannot run on this box; build "
                  f"it (`cargo build --release`) or run the producer on {first.get('box')}")
            continue

        run_argv = [resolved] + [expand_path_token(a, record_root, cache_root) for a in argv[1:]]
        out_path = Path(expand_path_token(output_path(argv), record_root, cache_root))
        backup = out_path.with_name(out_path.name + ".regen-before")
        rejected = out_path.with_name(out_path.name + ".regen-rejected")
        if dry_run:
            runs += 1
            print(f"RUN      {path} [{first.get('box')}]\n         "
                  f"{' '.join(shlex.quote(a) for a in run_argv)}")
            continue

        had_file = out_path.is_file()
        runs += 1
        if had_file:
            try:
                if backup.exists():
                    backup.unlink()
                os.replace(out_path, backup)
            except OSError as exc:
                print(f"check_f6_fixtures.py: {out_path}: cannot move the fixture aside: {exc}",
                      file=sys.stderr)
                return 2
        print(f"RUN      {path} [{first.get('box')}]\n         "
              f"{' '.join(shlex.quote(a) for a in run_argv)}")
        try:
            proc = subprocess.run(run_argv, cwd=str(repo_root), env=os.environ.copy(),
                                  capture_output=True, text=True)
        except OSError as exc:
            proc = None
            tail = [str(exc)]
        if proc is None or proc.returncode != 0:
            code = "?" if proc is None else str(proc.returncode)
            tail = tail if proc is None else \
                ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-3:]
            findings += 1
            print(f"FAIL     {path}: the recorded producer exited {code}: "
                  f"{' | '.join(tail) if tail else '(no output)'}")
            restore_fixture(out_path, backup, rejected)
            continue

        if not out_path.is_file():
            findings += 1
            print(f"FAIL     {path}: the producer exited 0 but wrote no {out_path}")
            restore_fixture(out_path, backup, rejected)
            continue
        actual = sha256_file(out_path)
        size = out_path.stat().st_size
        matched = [i for i in idxs if entries[i].get("sha256") == actual]
        if not matched:
            elsewhere = [e for e in entries
                         if e.get("path") == path and e.get("sha256") == actual]
            findings += 1
            print(f"FINDING  {path} [{first.get('box')}]: the producer named by the record "
                  f"produced different content — {actual} ({size} B), against the record's "
                  f"{expected([entries[i] for i in idxs])}")
            if elsewhere:
                print(f"         the run reproduced the content recorded for "
                      f"{elsewhere[0].get('box')} ({elsewhere[0].get('date')}) instead — a "
                      "recorded divergence, so its identity, not this entry's, is what ran")
            at = running_commit or "a commit the record does not name"
            print(f"         the run was at minfer {at}; the digest is NOT recorded — record the "
                  "identity and the digest together by hand")
            restore_fixture(out_path, backup, rejected)
            continue
        for i in matched:
            e = entries[i]
            new = [("sha256", actual), ("bytes", size), ("date", today)]
            diffs = [(k, e.get(k), v) for k, v in new if e.get(k) != v]
            if not diffs:
                print(f"OK       {path} [{e.get('box')}]: {actual} ({size} B) — the record already "
                      "holds this content; nothing written")
                continue
            for k, _old, v in diffs:
                e[k] = v
            changes += 1
            print(f"RECORD   {path} [{e.get('box')}]: " +
                  ", ".join(f"{k} {old!r} -> {new_v!r}" for k, old, new_v in diffs))
        if backup.exists():
            backup.unlink()
            print(f"         removed {backup} (byte-identical to the regenerated file)")

    if dry_run:
        print(f"check_f6_fixtures.py --regenerate --dry-run: {runs} runnable, {refusals} refused "
              "— nothing was run and nothing was written")
        return 1 if (strict_runnable and refusals) else 0

    if changes:
        manifest_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        reloaded, probs = load_manifest(manifest_path)
        if reloaded is not None:
            probs += check_manifest(manifest_path, reloaded)
        if probs:
            for p in probs:
                print(p, file=sys.stderr)
            print(f"check_f6_fixtures.py: the re-recorded {manifest_path} is not well formed",
                  file=sys.stderr)
            return 1
        print(f"check_f6_fixtures.py --regenerate: wrote {manifest_path} "
              f"({changes} content identity/identities re-recorded, S1-S7 still clean)")
    print(f"check_f6_fixtures.py --regenerate: {runs} producer(s) run, {changes} re-recorded, "
          f"{findings} finding(s), {refusals} refused")
    if findings:
        return 1
    if refusals:
        return 3
    return 0


# --------------------------------------------------------------------------- #
# --selftest
# --------------------------------------------------------------------------- #

def _tiny_manifest(root: Path, sha: str, **over) -> dict:
    e = {
        "path": "ref/tiny.gguf",
        "bytes": 4,
        "sha256": sha,
        # The recorded output path has to be the entry's own (S7), and the input
        # is outside the cache root so it is not a fixture S5 would ask about.
        "producer": "llama-quantize /tmp/f6-selftest-src.gguf "
                    "~/.cache/minfer/f6-src/ref/tiny.gguf q4_0",
        "producer_kind": "llama-quantize",
        "llamacpp_commit": "deadbeef",
        "compiler": "gcc 13.3.0",
        "ffp_contract": "fast",
        "cflags": "-O3 -DNDEBUG",
        "authoritative_reference": True,
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


def _regen_manifest(root: Path, fake: Path, content: bytes, **over) -> dict:
    """A one-entry manifest whose producer is a fake `minfer` in the temp tree."""
    e = {
        "path": "ref/tiny.gguf",
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
        "producer": f"{fake} convert {root}/in.gguf {root}/ref/tiny.gguf --outtype f32",
        "producer_kind": "minfer",
        "minfer_commit": "abc1234",
        "date": "2026-09-27",
        "box": "dgxspark (aarch64, GB10 sm_121)",
    }
    e.update(over)
    return {
        "schema": 1,
        "cache_root": str(root),
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
        bad["entries"][0]["producer"] = "minfer convert /tmp/in.gguf " \
                                        "~/.cache/minfer/f6-src/ref/tiny.gguf --outtype f16"
        # `authoritative_reference` is a llama-quantize field (S2): a minfer
        # producer has no compiler, so the mark goes with the kind.
        bad["entries"][0].pop("authoritative_reference")
        bad["entries"][0].pop("llamacpp_commit")
        bad["entries"][0].pop("compiler")
        bad["entries"][0].pop("ffp_contract")
        bad["entries"][0].pop("cflags")
        record("an unrecorded producer without a note is a structural problem",
               bool(check_manifest(manifest_path, bad)))
        bad["entries"][0]["producer_note"] = "not recorded at the time"
        record("... and is accepted once the note explains it",
               not check_manifest(manifest_path, bad))
        bad = _tiny_manifest(root, good)
        soft = dict(bad["entries"][0], authoritative_reference=False)
        del soft["sha256"]
        soft["sha256_prefix"] = good[:8]
        soft["digest_note"] = "recorded as a prefix only"
        bad["entries"].append(soft)
        record("a prefix that is a prefix of the full digest for the same path is rejected",
               any("not a divergence" in p for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good)
        bad["entries"].append(
            dict(bad["entries"][0], sha256="f" * 64, authoritative_reference=False)
        )
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
        # S6 (#349): the manifest must record which content the byte-parity claim
        # is asserted against, and it must be the contracting build.
        bad = _tiny_manifest(root, good)
        del bad["entries"][0]["authoritative_reference"]
        record("a llama-quantize path with no authoritative reference is rejected",
               any("no entry marked" in p for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, ffp_contract="off")
        record("... and a non-contracting authoritative build is rejected",
               any("asserted against the contracting build" in p
                   for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good)
        bad["entries"].append(dict(bad["entries"][0], sha256="f" * 64))
        bad["divergence_notes"] = [
            {"cause": "two builds", "paths": ["ref/tiny.gguf"], "explanation": "why"}
        ]
        record("two authoritative entries for one path are rejected",
               any("marks 2 entries" in p for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, producer_kind="minfer", minfer_commit="abc")
        bad["entries"][0].pop("compiler")
        record("an authoritative_reference on a non-llama entry is rejected",
               any("only meaningful on a llama-quantize" in p
                   for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, authoritative_reference="yes")
        record("a non-boolean authoritative_reference is rejected",
               any("must be true or false" in p for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good)
        bad["entries"][0]["minfercommit"] = "typo"
        record("an unknown field is rejected", bool(check_manifest(manifest_path, bad)))
        record("a missing manifest is a problem",
               bool(load_manifest(tmp / "nope.json")[1]))

        # The manifest override (#354): --manifest > MINFER_F6_MANIFEST > tracked.
        # An empty override is refused, never read as "unset".
        path, source, problems = resolve_manifest("/flag.json", "/env.json", tmp)
        record("--manifest wins over the environment override",
               path == Path("/flag.json") and source == "argv" and not problems, str(problems))
        path, source, problems = resolve_manifest(None, "/env.json", tmp)
        record("MINFER_F6_MANIFEST selects the record when no flag is given",
               path == Path("/env.json") and source == "env" and not problems, str(problems))
        path, source, problems = resolve_manifest(None, None, tmp)
        record("with no override the tracked manifest is the record",
               path == tmp / MANIFEST and source == "default" and not problems, str(problems))
        _, source, problems = resolve_manifest(None, "   ", tmp)
        record("an empty override is a usage problem, not a silent fall back",
               source == "env" and len(problems) == 1 and MANIFEST_ENV in problems[0]
               and "empty" in problems[0], str(problems))

        # S7 (#345): the recorded command runs the kind's program and writes the
        # entry. Without this, --regenerate would run a command that cannot
        # re-record the entry it is recorded on.
        bad = _tiny_manifest(root, good, producer_kind="minfer", minfer_commit="abc1234")
        bad["entries"][0].pop("compiler")
        bad["entries"][0].pop("ffp_contract")
        bad["entries"][0].pop("cflags")
        bad["entries"][0].pop("llamacpp_commit")
        bad["entries"][0].pop("authoritative_reference")
        record("a command that runs another producer than its kind names is rejected",
               any("runs 'llama-quantize'" in p for p in check_manifest(manifest_path, bad)),
               str(check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good)
        bad["entries"][0]["producer"] = bad["entries"][0]["producer"].replace(
            "ref/tiny.gguf", "ref/other.gguf")
        record("a command that writes another path than the entry is rejected",
               any("writes" in p and "other.gguf" in p
                   for p in check_manifest(manifest_path, bad)))
        bad = _tiny_manifest(root, good, producer="unrecorded (the producer was never written down)")
        record("a producer that is not a command needs a producer_note",
               any("not a runnable command" in p for p in check_manifest(manifest_path, bad)))
        bad["entries"][0]["producer_note"] = "the command was never recorded"
        record("... and is accepted once the note explains it",
               not check_manifest(manifest_path, bad))

        # --regenerate (#345): the producer runs, the content is verified against
        # the record, and only sha256/bytes/date are written.
        fake = tmp / "bin" / "minfer"
        fake.parent.mkdir()
        fake.write_text('#!/bin/sh\nprintf "%s" "$F6_SELFTEST_CONTENT" > "$3"\n',
                        encoding="utf-8")
        fake.chmod(0o755)
        content = b"tiny-regenerated"
        payload = b"a different producer version"
        fake_cache = tmp / "regen-cache"
        (fake_cache / "ref").mkdir(parents=True)
        (fake_cache / "ref" / "tiny.gguf").write_bytes(content)
        old_environ = os.environ.get("F6_SELFTEST_CONTENT")
        # The fake producer writes whatever the environment names, so the two
        # cases below differ only in what the record claims was produced.
        os.environ["F6_SELFTEST_CONTENT"] = content.decode()

        def _regen(doc: dict, **kw) -> tuple[int, str]:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = regenerate(doc, manifest_path, fake_cache, tmp, today="2026-10-07", **kw)
            return code, buf.getvalue()

        # A run that reproduces the record: the content fields stay, the date moves.
        doc = _regen_manifest(fake_cache, fake, content)
        manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        before = json.loads(manifest_path.read_text())
        code, out = _regen(doc, running_commit="abc1234")
        after = manifest_path.read_text()
        record("a reproducing run exits 0", code == 0, out)
        record("a reproducing run records the date of the run",
               json.loads(after)["entries"][0]["date"] == "2026-10-07", after[:200])
        record("a reproducing run leaves sha256 and bytes as the record has them",
               json.loads(after)["entries"][0]["sha256"] == before["entries"][0]["sha256"]
               and json.loads(after)["entries"][0]["bytes"] == before["entries"][0]["bytes"])
        record("a reproducing run invents no provenance",
               {k: v for k, v in json.loads(after)["entries"][0].items()
                if k not in ("date",)} == {k: v for k, v in before["entries"][0].items()
                                           if k not in ("date",)})
        # Idempotent: a second run the same day writes nothing.
        code, out = _regen(json.loads(manifest_path.read_text()), running_commit="abc1234")
        record("a second run the same day writes nothing (idempotent)",
               code == 0 and manifest_path.read_text() == after and "nothing written" in out, out)

        # A run that produces different bytes: a finding, the record untouched
        # and the fixture restored.
        (fake_cache / "ref" / "tiny.gguf").write_bytes(payload)
        doc = _regen_manifest(fake_cache, fake, payload)
        manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        code, out = _regen(doc, running_commit="abc1234")
        record("a run that no longer reproduces the record is a finding (exit 1)",
               code == 1 and "FINDING" in out, out)
        record("a finding names both digests",
               hashlib.sha256(content).hexdigest() in out
               and hashlib.sha256(payload).hexdigest() in out, out)
        record("a finding does not rewrite the manifest",
               json.loads(manifest_path.read_text()) == doc, manifest_path.read_text()[:200])
        record("a finding restores the fixture and keeps the rejected content",
               (fake_cache / "ref" / "tiny.gguf").read_bytes() == payload
               and (fake_cache / "ref" / "tiny.gguf.regen-rejected").read_bytes() == content)

        # A producer program that is not here: refused, naming the binary.
        missing = tmp / "no-such-dir" / "minfer"
        doc = _regen_manifest(fake_cache, fake, content,
                              producer=f"{missing} convert {fake_cache}/in.gguf "
                                       f"{fake_cache}/ref/tiny.gguf --outtype f32")
        code, out = _regen(doc, running_commit="abc1234")
        record("a producer program that is not here is refused by name (exit 3)",
               code == 3 and "REFUSED" in out and str(missing) in out, out)

        # A recorded producer identity that is not what runs here: refused.
        doc = _regen_manifest(fake_cache, fake, content, minfer_commit="deadbee")
        code, out = _regen(doc, running_commit="abc1234")
        record("a run at another producer identity is refused naming both commits",
               code == 3 and "deadbee" in out and "abc1234" in out, out)

        # A llama-quantize entry: out of scope, refused naming the build, and no
        # run even when the binary exists.
        llama = tmp / "bin" / "llama-quantize"
        llama.write_text("#!/bin/sh\nexit 1\n", encoding="utf-8")
        llama.chmod(0o755)
        runner = _tiny_manifest(root, good)
        runner["cache_root"] = str(fake_cache)
        runner["entries"][0]["path"] = "ref/tiny.gguf"
        runner["entries"][0]["producer"] = f"{llama} /tmp/in.gguf {fake_cache}/ref/tiny.gguf q4_0"
        runner["entries"][0]["llamacpp_binary"] = str(llama)
        code, out = _regen(runner)
        record("a llama-quantize entry is refused as out of scope, naming the build",
               code == 3 and "out of scope" in out and str(llama) in out, out)

        # --dry-run: classified, nothing run and nothing written.
        dry = tmp / "dry-cache"
        (dry / "ref").mkdir(parents=True)
        (dry / "ref" / "tiny.gguf").write_bytes(payload)
        doc = _regen_manifest(dry, fake, payload)
        manifest_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
        code, out = _regen(doc, dry_run=True, running_commit="abc1234")
        record("--dry-run lists the command and writes nothing",
               code == 0 and "RUN" in out and manifest_path.read_text() == json.dumps(doc, indent=2),
               out)
        if old_environ is None:
            del os.environ["F6_SELFTEST_CONTENT"]
        else:
            os.environ["F6_SELFTEST_CONTENT"] = old_environ
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
    parser.add_argument("--manifest", default=None,
                        help=f"the manifest (default ${MANIFEST_ENV}, else {MANIFEST})")
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
    parser.add_argument("--regenerate", action="store_true",
                        help="re-run the recorded producer(s) and re-record the content identity")
    parser.add_argument("--only", default=None,
                        help="with --regenerate: a manifest path, optionally PATH@BOX")
    parser.add_argument("--box", default=None,
                        help="with --regenerate: every entry recorded on this box label")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --regenerate: classify and print the commands, run nothing")
    parser.add_argument("--strict-runnable", action="store_true",
                        help="with --regenerate --dry-run: a refusal is a failure")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if (args.only or args.box or args.dry_run or args.strict_runnable) and not args.regenerate:
        print("check_f6_fixtures.py: --only/--box/--dry-run/--strict-runnable need --regenerate",
              file=sys.stderr)
        return 2

    root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    manifest_path, manifest_source, problems = resolve_manifest(
        args.manifest, os.environ.get(MANIFEST_ENV), root)
    if problems:
        for p in problems:
            print(p, file=sys.stderr)
        return 2
    if args.manifest and os.environ.get(MANIFEST_ENV) is not None:
        # The override is shadowed by the explicit flag. Say so: an override that
        # is quietly ignored is the same silent wrong-record hazard as one that is
        # quietly honoured.
        print(f"check_f6_fixtures.py: {MANIFEST_ENV} is set but --manifest was given; using "
              f"{manifest_path}", file=sys.stderr)
    if manifest_source == "env":
        if args.regenerate:
            print(f"check_f6_fixtures.py: --regenerate refuses {MANIFEST_ENV}: re-recording into "
                  f"a manifest chosen by the environment is the accident the override must not "
                  f"create (issue #354). Unset {MANIFEST_ENV}, or pass --manifest "
                  f"{manifest_path} explicitly — the flag is the auditable spelling.",
                  file=sys.stderr)
            return 2
        print(f"check_f6_fixtures.py: {MANIFEST_ENV} is set; auditing {manifest_path} instead of "
              f"{root / MANIFEST}", file=sys.stderr)
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
    if args.regenerate:
        return regenerate(doc, manifest_path, cache, root, only=args.only, box=args.box,
                          dry_run=args.dry_run, strict_runnable=args.strict_runnable)
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
