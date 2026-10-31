# 0064. A hash that does not resolve is history only where the document or the ledger says so

- Status: Accepted
- Date: 2026-10-31
- Issues: #684
- Amends: ADR-0027 (a citation is dated — a commit hash is a citation class it did not name)

## Context

[#443](https://github.com/yusiwen/minfer/issues/443) closed on a rule: no document cites a commit
`git cat-file -t` cannot resolve. [#683](https://github.com/yusiwen/minfer/issues/683) swept the
tree for violations and found four, and
[#685](https://github.com/yusiwen/minfer/pull/685) repaired them. Neither had a gate. A backticked
hash is neither a relative link (`scripts/check_docs_links.py` reads those) nor a `path:NNN` anchor
(`scripts/check_doc_line_anchors.py` reads those), so an unresolvable hash in a table cell or a
sentence is invisible to every checker in `check-docs`. It is the one locator class
[ADR-0027](0027-a-citation-is-dated-too.md) did not name.

The class is not small, and a bare probe is not usable. Measured on 2026-10-31 at `master`
`24626ad`, of 936 backticked hash-like tokens in the 260 documents the doc gates read, **129 are
unresolvable, in 58 distinct hashes** — and every one of them is legitimate:

| Class | Occurrences | Where |
|---|---|---|
| pre-rewrite labels of this repository, kept as history | 72, in 41 hashes | `docs/METAL_OPTIMIZATIONS.md`, under its caveat |
| llama.cpp hashes, a foreign repository | 41, in 6 hashes | 17 documents, from the CUDA port record to ADR-0060 |
| this repository's own labels that an amend or a rewrite removed, each already narrated as unresolvable | 16, in 11 hashes | the CUDA optimization records, the plan, the registry design |

The measurement itself carried the trap that decided decision 1. "Unresolvable" cannot mean "this
clone lacks the object": a checkout that has amended or rebased keeps the discarded commits, so
`git cat-file -t` resolves them locally while a fresh clone cannot. On this tree the difference was
**14 citations in 10 hashes** — exactly the amended-away twins the CUDA records already describe
as unresolvable. A gate on bare existence would be green on the author's box and red in CI, which
is the worst split a gate can have.

Two lessons decided the rest of the shape. A document-level *prose* caveat was added to the Metal
record in #495 and had to be narrowed by #685: that file also cites current-history commits that do
resolve, so "every hash here is history" was false within weeks. And a gate that cannot tell history
from staleness is a gate somebody switches off — [ADR-0027](0027-a-citation-is-dated-too.md) had
already learned both for paths, and this is the same problem one token class over.

## Decision

**1. A backticked hash is a citation, and resolution is the test.** A token of 7–40 lower-case hex
characters inside one code span, outside a fenced block, is a locator. It passes when
`git cat-file -t` resolves it **to a commit HEAD can reach** — an object of another type, or a
commit outside this revision's history, is a defect rather than a citation: the first is not a
pointer, the second is an amended-away twin or an unmerged branch. `scripts/check_status.py` already
holds its baseline commits to the same ancestry rule. Everything else must say what it is.

**2. Three classes, and only three.** *resolves* (a locator), *ledgered* (pinned in
`scripts/doc-hashes.toml` with a `kind`, a `source` and a reason), *marked* (the document declares
its unresolvable hashes are history). Anything else fails, naming the file, the line, the hash and
the sentence that carries it.

**3. The marker is a token, never a paragraph.** One standalone HTML comment in the document:

```html
<!-- doc-hashes: history -->
```

It applies to the whole document and is invisible in the rendered book. Its scope is exact: it
excuses an **unresolvable** hash. A hash that resolves is a locator, and no marker can exempt it —
which is what keeps the construct from rebuilding the blanket claim #685 had to narrow. A marker
that covers no unresolvable hash fails, so deleting the last pre-rewrite label deletes the marker
in the same commit.

**4. The ledger pins a hash, never a line.** `scripts/doc-hashes.toml` holds one `[[exempt]]` entry
per hash — `hash`, `kind` (`foreign` or `pre-rewrite`), `source` (whose history it is) and `reason`
(why it is kept rather than repaired). A line number drifts on any edit above it, and one hash can
be cited from ten documents; the hash is the stable key.

**5. Both exemptions ratchet.** An entry no document cites fails; an entry whose hash *resolves
now* fails, because it is not history any more; an unknown `kind` or an empty `source`/`reason` is a
usage error; a marker that covers nothing fails. That is the `scripts/adr-citations.toml` shape,
applied to the second citation class that needed it.

**6. A tree the gate cannot judge is a failure.** No `git` on `PATH`, not a work tree, or a
`.git/shallow` clone — where an absent object may simply not have been fetched — exits 2 with the
reason. A gate that silently passes in a shallow clone is worse than no gate.

**7. The gate lives in `check-docs`, unclassified by the path filter.** It reads documents and
history, needs no toolchain, and a rename or a deletion of its ledger is a red job rather than a
silently disabled check ([ADR-0023](0023-the-machine-ledgers-live-beside-their-checkers.md)).

## Alternatives considered

- **A bare probe with no exemptions.** Rejected: 129 failures on the tree that motivated the rule,
  and no way to tell history from a typo — the failure mode
  [ADR-0027](0027-a-citation-is-dated-too.md) rejected for paths, and the reason such a gate would
  be switched off within a week.
- **One document-level marker, no ledger.** Rejected twice over: 41 foreign occurrences live in 17
  documents that also cite this repository's own commits, so a document-wide "history" marker would
  excuse a mistyped current hash in every one of them; and the two classes are not the same fact —
  a foreign hash belongs to one repository, stated once, not repeated per document.
- **Qualifying each foreign citation in place (`llama.cpp@<hash>`).** Rejected: it reads well, but
  it rewrites 41 citations, and — decisive — a bare backticked hash is also a *count stamp*:
  `scripts/check_doc_counts.py` reads one as the revision a size claim was measured at, so the
  migration would silently un-stamp claims in the documents it touched.
- **An advisory report that never fails.** Rejected for [ADR-0027](0027-a-citation-is-dated-too.md)'s
  reason: a check whose failure is invisible is not read, and no documentation check in this
  repository is advisory.
- **Line-keyed ledger entries.** Rejected by decision 4: the same hash is cited from up to ten
  documents, and every edit above a line churns the entry.

## Consequences

- The class is gated, and its initial state is explicit and reviewed: 72 occurrences behind one
  marked document and 57 occurrences behind 17 ledger entries, each with a source and a reason.
- Cost accepted: a new legitimate historical citation fails `check-docs` until it is marked or
  pinned — deliberate friction on the PR that causes it.
- The known blind spot is the marker's granularity: it is document-wide, so a mistyped current hash
  inside the one marked document would pass. Scoping the marker per table is the tightening, and it
  is a follow-up issue rather than a rename of this decision.
- The checker proves **existence, not correctness**: it cannot tell whether a hash that resolves is
  the change the sentence describes — the same limit `scripts/check_doc_counts.py` states for
  numbers.
- A backticked decimal number is read as a hash. Every 7+-digit token in the tree today is a real
  commit, so the class is not noisy; if one ever appears, a ledger entry with a `not-a-citation`
  kind is the escape hatch, and that is an amendment to this decision rather than a habit.

## References

- `scripts/check_doc_hashes.py` — the gate; `scripts/doc-hashes.toml` — the ledger it reads; both
  run in `check-docs`.
- `docs/GATE-CONTRACT.md` §"Citations" — the rule this records, and the doc-gate list it joins.
- `docs/METAL_OPTIMIZATIONS.md` — the one marked document, and the caveat the marker makes
  machine-readable.
- [ADR-0027](0027-a-citation-is-dated-too.md) (amended here),
  [ADR-0023](0023-the-machine-ledgers-live-beside-their-checkers.md).
- [#443](https://github.com/yusiwen/minfer/issues/443),
  [#683](https://github.com/yusiwen/minfer/issues/683),
  [#684](https://github.com/yusiwen/minfer/issues/684) — this decision;
  [#685](https://github.com/yusiwen/minfer/pull/685) — the narrowing this construct must not undo.
