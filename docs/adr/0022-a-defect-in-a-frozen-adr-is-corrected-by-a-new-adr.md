# 0022. A defect in a frozen ADR is corrected by a new ADR, not by editing it

- Status: Accepted
- Date: 2026-10-09
- Issues: #450, #452
- Corrects: ADR-0001, ADR-0005, ADR-0010

## Context

`docs/adr/README.md` makes ADR text immutable: a changed decision is a *new* ADR plus `Superseded by` on
the old one, and the old text is never edited. That immutability is deliberate — the record of what was
believed, and why, is the useful part — but it leaves a case the corpus did not provide for: what happens
when the **text itself** is wrong?

Three cases arrived at once, all found by the primary-source audit of the two device design documents
([#446](https://github.com/yusiwen/minfer/issues/446); the audit is reported in
[#449](https://github.com/yusiwen/minfer/pull/449)):

1. **ADR-0001 cites a section that does not exist.** Its *Alternatives considered* supports the rejected
   whole-layer `layer_gpu()` fast path with "(`docs/COMPUTE-GRAPH-DESIGN.md`, `docs/METAL-BACKEND-DESIGN.md`
   §2 non-goals)". `METAL-BACKEND-DESIGN.md` has no §2 non-goals — its non-goals are `### 1.3`.
2. **ADR-0010 cites a table row with no section.** Its *References* name "the 'differ by design' row and
   the tolerance-class gates" in `docs/CUDA-BACKEND-DESIGN.md`. The row is the *Accepted* consequence the
   ADR's whole argument rests on — the §6 risk ledger's row 9, with a related clause in §1.2 — and the
   citation cannot be navigated.
3. **ADR-0005 states a capability in the present tense.** It says `BackendCaps::reads_packed_kv`
   "is **now true** for CPU, CUDA and Metal" — the "currently supports X" this corpus forbids in so many
   words. The rot is already visible: [#440](https://github.com/yusiwen/minfer/issues/440) records four
   source sites still asserting the opposite.

The same audit showed the rule's other half is wrong as written. The README says an ADR "never carries a
test count", but two ADRs need one to make their argument, and both are frozen *with* the decision:

- ADR-0006 contrasts the process-wide global's `5 passed / 7 failed` (parallel harness) with what
  replaced it, `12 passed / 0 failed`.
- ADR-0016 records the `#[ignore]`d serial CUDA set going `5 passed / 14 failed` → `20 passed / 2 failed`.

## Decision

**1. Frozen text is never edited; a correction is a new ADR.** The correcting ADR declares
`- Corrects: ADR-NNNN[, …]`, and states what was wrong in its *Context* and what to conclude instead in
its *Decision*. The corrected ADR's bytes never change.

**2. The correction is visible from the index.** `docs/adr/README.md` is mutable, so the corrected ADR's
index row names the correcting ADR. Visibility is two-way — `Corrects:` names the targets, the target's
row names the corrector — exactly as supersession is two-way. `scripts/check_adr.py` enforces that the
target exists, is lower-numbered, and is named on its index row.

**3. A capability is recorded as a dated consequence, never in the present tense.** "At the `Date:`
above, #310 enabled X" is allowed; "X is now true" is not. The authority for what is true *today* is the
design doc and `docs/SUPPORT-MATRIX.md`, both mutable.

**4. A number is allowed as evidence, never as a baseline.** A count that is part of an argument — the
rejected alternative's failures, a named boundary — is frozen with the ADR and belongs in it; a *current*
suite baseline belongs in `docs/TEST-BASELINES.md` and `docs/status.toml`, which are machine-checked.
This accepts ADR-0006 and ADR-0016 as written, and refuses a new ADR that quotes today's suite result.

**5. The three corrections.** ADR-0001's Metal non-goals are `METAL-BACKEND-DESIGN.md` §1.3, not §2;
ADR-0010's row is `CUDA-BACKEND-DESIGN.md` §6 (risk ledger, row 9), not section-less; ADR-0005's clause
is read as the dated 2026-10-06 consequence of [#310](https://github.com/yusiwen/minfer/issues/310), with
the current capability stated in `docs/SUPPORT-MATRIX.md`.

## Alternatives considered

- **Edit the frozen ADR in place.** Rejected. It is the one guarantee the corpus makes, and it is what
  makes a backfilled ADR usable at all: a reader can trust that what a 2026-06 ADR says is what was
  believed then. Editing in place would also erase the evidence that the corpus can contain a defect —
  this list exists only because an audit found three.
- **Leave the defect and rely on the reader.** Rejected: a wrong section number is not a stylistic wart,
  it makes the citation unusable, and ADR-0010's row is load-bearing for its entire argument. Silence
  also guarantees the next audit rediscovers the same three.
- **A document-side alias only** — the other option `#450` offered: leave the ADR frozen and add a
  sentence to the design document. Rejected for *this* class: it helps a reader who starts at the
  document, but a reader who starts at the ADR — the whole reason `Corrects:` exists — never learns. It
  stays the cheaper right answer for a navigation error in a document rather than in a record.
- **A machine file (`docs/adr/corrections.toml`) as the correction index.** Rejected: a second index for
  a fact the ADR index already carries, placed where a reader of the ADR is not.
- **A new `Status` value ("Corrected").** Rejected: `Status` answers "is this decision current?", and a
  correction changes text, not currency. Overloading the field would make `Superseded by` ambiguous.

## Consequences

- The corpus gains an errata channel that does not weaken immutability: all 21 earlier ADRs stay
  byte-stable, and the three defects become visible where the reader already is.
- Two rules do machine-checkable work (`check_adr.py`: a `Corrects:` target must exist, be
  lower-numbered, and be named on its index row; `--selftest` proves each failure), while two remain
  review rules — the tense rule and the counts rule cannot be parsed without false positives, and are
  documented as reviewer tests instead of pretending to be gates.
- A future defect has a prescribed route, so "this ADR is wrong" stops being an argument for editing it.
- Cost knowingly accepted: a corrected ADR still reads as authoritative while its text stays wrong, so
  the reader must follow the index note. That is the price of immutability, and it is smaller than the
  price of a corpus whose records silently change.

## References

- `docs/adr/README.md` — the boundary rule, its corrected wording, and the index notes on ADR-0001 /
  ADR-0005 / ADR-0010.
- `docs/TEST-BASELINES.md`, `docs/status.toml` — the machine-checked home for suite baselines.
- `docs/METAL-BACKEND-DESIGN.md` §1.3; `docs/CUDA-BACKEND-DESIGN.md` §6; `docs/SUPPORT-MATRIX.md`.
- [#446](https://github.com/yusiwen/minfer/issues/446) and PR [#449](https://github.com/yusiwen/minfer/pull/449)
  — the audit that found all three; [#450](https://github.com/yusiwen/minfer/issues/450) — the citation
  defects; [#452](https://github.com/yusiwen/minfer/issues/452) — this decision.
