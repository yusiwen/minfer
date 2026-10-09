# 0026. The classifier carries no docs-path exception, and its list is ratcheted

- Status: Accepted
- Date: 2026-10-10
- Issues: #461
- Corrects: ADR-0025

## Context

ADR-0025 decided that a docs-only change skips the three code jobs, classified once in a `changes` job, and its Decision named three safety rules. The third was a standing exception:

> **a `docs/` path a code job reads is carried as an explicit exception**: today that is `docs/dead-code-baseline.toml`, the dead-code oracle's accepted-annotation baseline, compared against rustc's stripped output in `test-linux-cpu` and `build-linux-cuda`; moving it beside its checkers (ADR-0024) deletes the exception, and #461 tracks that.

That follow-through landed on 2026-10-10: #461 (PR #464) moved the record to `scripts/dead-code-baseline.toml`, byte-identical, beside the one program that loads it (`scripts/check_dead_code_oracle.py`'s `DEFAULT_BASELINE`), and emptied the classifier's exception list. ADR-0025's text is frozen, so it still names the old path and its Consequences still say "Only a baseline-only edit could be skipped, and it is exactly the exception above" — a present-tense statement about a list that no longer holds anything. Per ADR-0022, a defect in a frozen ADR's *text* is corrected by a new ADR, and the old text is never edited.

## Decision

There is no `docs/` path a code job reads, and the classifier carries no exception for one.

`scripts/classify_changes.py` keeps `RUST_DOC_FILES` as the audit's landing place, and it is **empty because the audit finds nothing** — a `grep` over the scripts the three code jobs run is the recipe, and it is what found the record in the first place. #459 emptied it of the F6 fixture manifest (moved to `tests/fixtures/`) and #461 of the dead-code baseline (moved to `scripts/`).

The emptiness is **ratcheted, not incidental**: the classifier's `--selftest` carries the case `no docs/ path is carried as a code-job exception`, so putting a path back fails the run until the case is changed and the justification is written into the module docstring. That is the both-ends shape the doc checkers use for a frozen set.

The baseline's home is `scripts/dead-code-baseline.toml`, and a change confined to it classifies as `rust` through the ordinary `scripts/` rule — no special case — so the two jobs that compare against it still run for a baseline-only edit.

## Alternatives considered

- **Leave ADR-0025's text alone.** Rejected: it names a path that no longer exists, which is the defect class ADR-0022 corrects, and its Consequences assert a present-tense exception the code no longer has. Its "today that is …" phrasing made the *Decision* honest at its date, but a reader looking for the record's home would still be sent to `docs/`.
- **Edit ADR-0025's body to the new path.** Rejected: the corpus's central rule — the old text is never edited; the record of what was believed, and why, is the useful part.
- **Keep a no-op entry in `RUST_DOC_FILES` so the old text stays literally true.** Rejected: an exception for a path that no longer exists is dead configuration, and it would disarm the ratchet that makes the next addition deliberate.
- **Correct ADR-0024 (the rule) instead of ADR-0025 (the exception).** Rejected: ADR-0024 states where a machine-read record lives and never named this record; there is nothing in it to correct.

## Consequences

- The record's behaviour is now the ordinary rule rather than a special case: `scripts/**` other than the two ledgers is `rust`, so the oracle's jobs run for a baseline-only edit.
- ADR-0025's index row names this ADR and this ADR carries `Corrects: ADR-0025`; `check_adr.py` enforces both ends. ADR-0025's body is untouched.
- The audit recipe stays where the next author needs it — the module docstring of `scripts/classify_changes.py` — with the empty list as the place a future finding lands.
- A cost is accepted knowingly: a whole ADR, two index edits and a PR to re-point one citation. That is what the corpus pays for never rewriting a record, and it is the same price ADR-0022 paid for three citations at once. The alternative — "the ADR already announced the move" — was considered and rejected, because that phrase lives in the *same* sentence that names the stale path.

## References

- `scripts/classify_changes.py` — `RUST_DOC_FILES`, the audit recipe in its module docstring, and the ratchet case.
- `.github/workflows/ci.yml` — the `changes` job and the three guarded jobs.
- [ADR-0022](./0022-a-defect-in-a-frozen-adr-is-corrected-by-a-new-adr.md) — the correction mechanism this ADR uses.
- [ADR-0024](./0024-a-machine-read-record-is-not-book-content.md) — where a machine-read record lives.
- [ADR-0025](./0025-a-docs-only-change-skips-the-compilers.md) — the corrected record.
