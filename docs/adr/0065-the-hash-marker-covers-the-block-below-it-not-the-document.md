# 0065. The hash marker covers the block below it, not the document

- Status: Accepted
- Date: 2026-10-31
- Issues: #703
- Amends: ADR-0064 decision 3 (the marker applied to the whole document)

## Context

[ADR-0064](0064-a-hash-that-does-not-resolve-is-history-only-where-the-document-says-so.md) landed
the hash gate with one marker form: a standalone HTML comment that declared **every** unresolvable
hash in the document to be history. The issue that became this ADR was filed in the same session,
because the mechanism has exactly the defect it was meant to fix, one level down.

The one marked document is `docs/METAL_OPTIMIZATIONS.md`. Measured on 2026-10-31 at `master`
`24626ad`: it holds **87 unresolvable hashes** — 54 of them inside eight tables, 33 inside twenty
prose blocks — interleaved with current-history commits that do resolve. #685 had to narrow the
document's *prose* caveat for that reason ("every hash here is history" was false within weeks); a
document-wide marker rebuilds the same claim in machine-readable form, and the cost is concrete: a
**mistyped current hash** in a new record in that file is unresolvable too, and the marker would
pass it.

## Decision

**1. A marker covers the block that follows it.** The block is the next run of non-blank lines after
the marker, once any further marker lines are skipped — one paragraph, one table, or one record
entry. A hash outside every marked block is judged as it is anywhere else. There is no
document-wide form, so two meanings cannot coexist.

**2. The four `Commit (pre-rewrite)` tables each carry one marker.** They are the blocks whose
whole purpose is a label column; 28 of the 54 table occurrences are hashes that appear nowhere else,
and the other 26 belong to hashes that also appear in prose (decision 3). A marker sits above the
table, where the reader sees it and where the ratchet can delete it with the table.

**3. A label inside a list item or a paragraph goes to the ledger.** `docs/METAL_OPTIMIZATIONS.md`
§3.1–§3.5 are bullet lists and continuation paragraphs; a standalone comment above one would split
the list it belongs to, which is a worse defect than the one being fixed. The 18 hashes that appear
in prose are pinned in `scripts/doc-hashes.toml` with the record they come from, so the document's
structure is untouched.

**4. Both ends of the marker ratchet.** A marker that covers no unresolvable hash fails (its block
no longer needs it), and a marker with no block behind it — at the end of the file, or above another
marker — fails as well. The four other rules of ADR-0064 stand unchanged: resolution means a commit
HEAD can reach, the ledger is keyed by hash with a `kind`/`source`/`reason`, an exemption that
outlives its reason fails, and a tree the gate cannot judge exits 2.

## Alternatives considered

- **Keep the document-wide marker.** Rejected: it is #703's defect, and it makes the gate blind
  exactly where the record is still growing.
- **Section scope (a marker above a heading covers the section).** Rejected: the sections that carry
  pre-rewrite labels also carry the current-history commits the record was extended with, so section
  scope reproduces the document-wide problem at a smaller radius, with a rule ("until the next
  heading of the same or shallower level") that is harder to see in the source.
- **An inline marker at the end of the citing line.** Rejected: it reads as decoration on the
  sentence, it has to be repeated at every wrap boundary, and it would be a second marker form for
  the same fact — one form was the point.
- **A marker per record entry, indented inside the list item.** Rejected: it splits the item's
  paragraph in the rendered page and reflows 20 records, for a fact the ledger can hold without
  touching the document.
- **Ledger everything, drop the marker.** Rejected: it would put 41 entries in the ledger where four
  lines say the same thing about four tables, and leave a mechanism with no user — the opposite of
  [ADR-0023](0023-the-machine-ledgers-live-beside-their-checkers.md)'s "a gate earns its keep".

## Consequences

- The one marked document now carries four marker lines instead of one, and the ledger grows from 17
  entries to 35 (18 prose labels added). Coverage: 28 occurrences behind the markers, 101 behind the
  ledger; the total stays 129 unresolvable, 0 unexcused.
- The marker's failure mode is now local: a mistyped hash in a table still hides behind that table's
  marker, but nothing outside it does. The remaining radius is one table, which is the smallest a
  line-based format allows without an inline form.
- A future record that keeps a whole table of labels needs one comment line; one that keeps a label
  in prose needs a ledger entry. The asymmetry is deliberate and is stated in
  `docs/GATE-CONTRACT.md` §"Citations" and in `scripts/doc-hashes.toml`'s header.

## References

- `scripts/check_doc_hashes.py` — the block scan and its `--selftest` cases;
  `scripts/doc-hashes.toml` — the 35 pinned hashes.
- `docs/GATE-CONTRACT.md` §"Citations" — the rule this narrows, and the ledger/block split.
- `docs/METAL_OPTIMIZATIONS.md` — the four table markers and the header paragraph that points at
  them.
- [ADR-0064](0064-a-hash-that-does-not-resolve-is-history-only-where-the-document-says-so.md)
  (amended here), [ADR-0027](0027-a-citation-is-dated-too.md),
  [ADR-0023](0023-the-machine-ledgers-live-beside-their-checkers.md).
- [#703](https://github.com/yusiwen/minfer/issues/703) — this decision.
