# 0039. The corpus is entered through a question at PR time, and its maps are checked

- Status: Accepted
- Date: 2026-10-11
- Issues: #539

## Context

The corpus was self-consistent but not self-triggering. `scripts/check_adr.py` enforces the shape a
decision record must have — dense numbering, a two-way `Superseded by`, an index row, cited paths that
resolve or are pinned in `scripts/adr-citations.toml` (ADR-0022, ADR-0027) — and its own docstring states
the boundary: *"A checker can force a shape, not a habit."* Nothing asked for the record in the first
place, and the one index from a contract back to its decision was unchecked: on 2026-10-11 the tree had
**17** maps and **32 of 38** ADRs listed by one, so six decisions had no document claiming them and a
typo'd number, a link to a renamed ADR or a decision nobody carries all read exactly like a maintained
map.

Two facts about *who* reads this repository decided the shape of the answer. `AGENTS.md` mentioned the
corpus once, as a Docs-Index row — a pointer, not a rule — and a fresh agent session loads that file and
then reads CI failures. A rule that is not in one of those two places is, for practical purposes, not
enforced at all.

## Decision

**1. The question is asked at PR time, and the waiver is the record that it was asked.** The change
classifier carries `DECISION_PATHS`: the files whose *subject is an ADR's subject* — the op vocabulary,
backend ids, the `Backend` trait's contract, the KV format and session file, the on-disk block layout,
the GGUF writer's byte contract, the conversion contract, the quantizers' byte-identity promise. A range
touching one of them, or **adding** a `docs/*-DESIGN.md`, is a `decision_surface`, and the PR body must
answer on one line: `ADR: NNNN` (this implements that decision), `ADR: supersedes NNNN`, or
`ADR: none — <reason>`. Silence is not one of the three answers.

**2. The fail-safe asks it too.** A range the classifier cannot read (an empty base, the all-zero `before`
of a first push, an unresolvable revision) counts as a decision surface, exactly as it counts as `rust`. A
range nobody could read is not evidence that no decision was taken.

**3. The maps are checked, in both directions.** Every design document's map must list ADRs that exist,
link them at their real filenames and carry at least one entry; every ADR must be listed by some document's
map or recorded in `scripts/adr-map-exclusions.toml` with a reason. That ledger holds the six ADRs which
govern no document's contract — five process decisions about the machinery and one superseded — as a
*recorded* decision rather than silence, and it ratchets: an exclusion whose ADR *is* mapped fails, an
entry that is not an ADR fails.

**4. The trigger list is short, derived from the corpus, and ratcheted.** Each `DECISION_PATHS` entry names
the ADRs whose decision it carries, so the list is a statement about *this* corpus rather than a guess about
what matters; `classify_changes.py --selftest` fails when an entry stops existing, and a missed case is
added together with the ADR that made it a decision — never speculatively.

**5. Presence, not truth.** The gate forces the sentence to exist; it cannot force it to be right. The same
limit every prose gate here has (ADR-0025's `check_pr_body.py`), stated rather than hidden.

## Alternatives considered

- **No gate — state the habit in the corpus and trust reviewers.** Rejected: this is the status quo that
  produced six unmapped ADRs and an unchecked map, and it is strongest exactly where it is weakest, because
  an agent session has no memory of a habit and no reviewer on a solo change.
- **Require an ADR for every PR that touches `src/`.** Rejected: it inverts ADR-0023. A record per change is
  a changelog, not a decision corpus, and a rule that fires on routine work trains its readers to paste the
  waiver without reading it.
- **A semantic judge — a checker (or a model) that decides "is this a decision?".** Rejected: it is neither
  deterministic nor offline, and it would answer the hard question with a weaker oracle than the author.
- **Document the rule and add no gate.** Kept as *half* the change (`AGENTS.md`, `docs/adr/README.md`, the
  PR template) but rejected as the whole of it: the half a fresh agent sees is the CI message.
- **Extend `adr-citations.toml` instead of adding a second ledger.** Rejected on kind: that ledger answers
  "a cited path no longer exists" — a fact about an ADR's text. This one answers "which decision does this
  document's contract depend on" — a fact about a document. ADR-0023's rule is one ledger per prose target.

## Consequences

- **A legitimate edit to one of the nine paths costs one sentence** in the PR body. That sentence is
  visible, diffable and reviewable, which is the whole mechanism: the gate converts a silent omission into
  a claim someone can disagree with.
- **The residue is irreducible, and it includes this ADR.** A decision that touches none of the trigger
  paths is still invisible — measured on this change: the range that added this record touches
  `scripts/classify_changes.py` and `docs/GATE-CONTRACT.md`, neither of which is a `DECISION_PATHS` entry,
  so the classifier would have reported `decision_surface=false`. The machinery is kept out of the trigger
  list on purpose (every ledger tweak would ask the question, and a reflex is not a decision); those
  decisions have been recorded by their authors (ADR-0023, ADR-0025, ADR-0028, ADR-0029) and the maps check
  asks the corpus-side question instead. The list grows from evidence, never speculatively.
- **`DECISION_PATHS` becomes a maintenance surface.** Renaming one of its files fails the classifier's
  selftest until the table is updated — the same ratchet the other machine ledgers carry beside their
  checkers (ADR-0023).
- **ADR-0028 and ADR-0029 become mechanical.** A new ADR must be mapped or recorded as governing no
  contract; the question "which document's contract does this fix?" is now asked from the corpus side as
  well as the code side.
- **`AGENTS.md` grows by one paragraph**, in the file whose size is capped — the rule is stated there and
  the long form lives in `docs/adr/README.md`, which is where a reader who is writing an ADR is already
  looking.
