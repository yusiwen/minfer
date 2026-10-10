# 0027. A citation is dated too — paths, symbols and counts, not only capabilities

- Status: Accepted
- Date: 2026-10-10
- Issues: #469
- Corrects: ADR-0022

## Context

ADR-0022 wrote the corpus rule as *"a capability is a dated consequence, never a present-tense claim"*.
That half works. It missed the rest of the sentence: **a path is a citation too**, and so is a symbol and
a count. The evidence is same-day, and it includes ADR-0022 itself:

- ADR-0022's two citations of the old status-ledger path died hours after it was accepted, when
  [ADR-0023](0023-the-machine-ledgers-live-beside-their-checkers.md) moved the ledgers to `scripts/`.
- ADR-0025's citation of the old annotation-baseline path died the same day, corrected by
  [ADR-0026](0026-the-classifier-carries-no-docs-path-exception.md).
- ADR-0022 states a count that rots: *"all **21** earlier ADRs stay byte-stable"*. There are 22. The ADR
  broke the rule it was writing, because the rule as written named only capabilities and, on the numbers
  side, only *suite* counts.

A probe over the corpus — every backticked repository-rooted path, tested with `os.path.exists` — finds
**22 unresolvable citations across 9 ADRs**. Most of them are legitimate, and that is the design problem
rather than a bug to fix:

| Kind | Occurrences | Example |
|---|---|---|
| `historical-before-state` | 4 | ADR-0012 cites the pre-split CUDA translation unit **with its 10,215-line count**, the problem the split replaced |
| `corrected-elsewhere` | 7 | ADR-0020's two citations of the fixture-manifest path, corrected by ADR-0024 |
| `named-in-the-move` | 10 | ADR-0023/0024/0026 necessarily name the location they moved away from |

The paths themselves, with a reason per entry, are in the ledger rather than here — this ADR names the
*kind* of home, which is the rule it is writing.
| `rejected-alternative` | 1 | ADR-0022's machine correction index — an option that was never created |

So a bare probe is not a gate: on today's tree it would raise 22 failures, and a rule that cannot tell
history from staleness would be switched off within a week. The false positives are a **one-time** cost
if they are pinned.

## Decision

**1. Capabilities, paths, symbols and counts follow one rule: a citation is dated.** An ADR may say *"at
the `Date:` above, #310 enabled X"* and *"the ledgers then lived under `docs/`"*. It may not say *"X is
now true"* or *"the ledgers live under `docs/`"* — the tense is the whole difference. The authority for
what is true today is the design doc, `docs/SUPPORT-MATRIX.md` and the ledgers — all mutable.

**2. The errata for ADR-0022.** Its self-count is replaced in effect by a count-free statement (a later
ADR never edits an earlier one's text, however many there are). Its two citations of the old
status-ledger path need **no second correction here**: ADR-0023 already carries them, and stating a
correction twice would create two sources for one fact.

**3. The machine-checkable half is a pinned ledger.** `scripts/adr-citations.toml` lists every
unresolvable citation as `(adr, path)` with a controlled `kind` (`historical-before-state`,
`corrected-elsewhere`, `named-in-the-move`, `rejected-alternative`) and a one-line reason. Keys are never
line numbers: a line number drifts on any edit above it, which would churn the ledger on every
unrelated change.

**4. `scripts/check_adr.py` enforces four rules**, in `check-docs` (which runs unconditionally, so a
`rust`-class PR that moves a file is checked too): a new unresolvable citation fails and names the ADR and
the path; a pinned citation passes; a pinned citation that no ADR cites any more fails (the ledger may
not rot); a missing or unknown `kind` fails. A token containing a wildcard or a placeholder, or ending in
`/`, is a pattern or a directory, not a citation.

**5. A ledger update belongs to the migration that invalidates the citation**, not to the later PR that
trips over it. This is the correct placement and it is already where the friction lands: moving a file
that an ADR cites must produce the errata and the ledger entry **in that PR**, exactly as ADR-0023 and
ADR-0024 did.

## Alternatives considered

- **A bare probe as a gate, with no ledger.** Rejected: 22 failures on the tree that motivated the rule,
  and no way to separate history from staleness — a rule that fires on ADR-0023 for *describing its own
  move* gets disabled. The ledger is what makes the gate truthful.
- **An advisory step that prints and never fails** (the first preference, and the reason it was
  considered: no ledger to maintain, and no chance of blocking a legitimate historical citation).
  Rejected: **a check whose failure is invisible is not read.** This corpus's rule rotted within hours
  with no gate at all, and no documentation check in this repository is advisory — a print-only step would
  be new machinery with no adopter. The cost it avoids is one-time; the cost it keeps is permanent.
- **Rewrite or delete the historical citations** so the probe finds nothing. Rejected: it falsifies the
  record, which is the one thing an ADR exists to preserve. ADR-0012's value is partly that it says how
  large the pre-split CUDA translation unit had grown.
- **Put the ledger in `docs/adr/`.** Rejected by ADR-0024: a machine-read record does not live in the
  book tree. It belongs beside its checker, `scripts/`.
- **Fix ADR-0022's paths here as well.** Rejected by decision 2 — the correction exists, in ADR-0023.

## Consequences

- The corpus's dead-path class is now gated rather than hoped for, and the gate's initial state is
  **explicit and reviewed**: 13 entries, 22 occurrences, each with a reason a reader can disagree with.
- Cost accepted: a **new** legitimate historical citation — a file that moves, a pre-split path — fails
  `check-docs` until it is pinned with a reason. That is deliberate friction on the PR that causes it.
- The ledger proves **knowledge, not correctness**. It records that a path was known not to resolve; it
  cannot tell whether the citation is the right one. A citation that resolves but no longer names the
  thing it cites — the ADR-0005 class that #440 tracks — is still a review rule, and this ADR does not
  pretend otherwise.
- The four rules are mechanical, and the fifth (the tense rule) is not; the ADR states the split so a
  reviewer knows which half they are holding.

## References

- `docs/adr/README.md` — the three review rules and the ledger's row in the boundary table.
- `scripts/adr-citations.toml` — the pinned citations; `scripts/check_adr.py` — the four rules and their
  `--selftest` cases.
- [ADR-0022](0022-a-defect-in-a-frozen-adr-is-corrected-by-a-new-adr.md) (corrected here),
  [ADR-0023](0023-the-machine-ledgers-live-beside-their-checkers.md),
  [ADR-0024](0024-a-machine-read-record-is-not-book-content.md).
- [#469](https://github.com/yusiwen/minfer/issues/469) — this decision;
  [#454](https://github.com/yusiwen/minfer/issues/454) — the ADR-map audit that also checks which of a
  document's citations are historical.
