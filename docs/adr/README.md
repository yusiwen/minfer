# Architecture Decision Records — index

An ADR records **a decision, the alternatives it beat, and what it cost** — and then stops
changing. This directory answers "why is it this way?", a question the design docs cannot answer
because they describe the *current* contract, which is allowed to move.

## The boundary rule

This is what keeps the corpus from becoming a fourth copy of the documentation. Each kind of fact
has exactly one home:

| Content | One home | Mutable? |
|---|---|---|
| The decision, the alternatives considered, the accepted consequences | `docs/adr/NNNN-*.md` | **No** — a change is a *new* ADR plus `Superseded by` on the old one |
| The current contract: capabilities, file formats, semantics | the design doc (`docs/KV-CACHE-DESIGN.md`, `docs/COMPUTE-GRAPH-DESIGN.md`, …), which links `Decision: ADR-NNNN` | Yes |
| Measurements, suite counts, per-ticket history | `docs/status.toml`, `docs/TEST-BASELINES.md`, `docs/ARCHITECTURE-EXECUTION-PLAN.md` | Yes |

**An ADR never says "currently supports X" and never carries a test count.** Those rot, and the ADR's
entire value is that it does not. `docs/status.toml` is the machine source for counts.

Two consequences worth stating outright:

- **Superseding is additive.** Correcting a decision means writing a new ADR and setting the old
  one's `Status` to `Superseded by ADR-NNNN`. The old text is never edited — the record of what we
  believed, and why, is the useful part.
- **Numbers are citations.** They are dense from `0001`, never reused, never renumbered. A number
  is assigned once, in the order decisions are established when their ADR is written, and the
  `Date:` field carries the date the decision was actually *taken*. Because most of this corpus is
  backfilled from existing records, a decision discovered later keeps its true (possibly earlier)
  `Date:` and takes the next free number — so the sequence is *approximately* chronological, and
  `Date:` is always the authority. That is the cost of keeping numbers stable; stability is worth
  more here than a perfect chronology.

`scripts/check_adr.py` (CI job `check-docs`) enforces the mechanically checkable half: filename and
heading agree, dense numbering, one of four `Status` values, a `YYYY-MM-DD` date, the required
sections, every ADR listed below, and `Superseded by` written from *both* ends.

## Index

| # | Date | Decision | Status |
|---|---|---|---|
| [0001](./0001-inference-runs-through-one-declarative-compute-graph.md) | 2026-08-21 | Inference runs through one declarative compute graph | Accepted |
| [0002](./0002-topology-is-a-function-of-graph-params.md) | 2026-08-21 | Topology is a function of `GraphParams` alone, so `positions` cannot be structure | Accepted |
| [0003](./0003-metal-is-out-of-scope-for-this-round.md) | 2026-09-16 | Metal is out of scope for this round | Superseded by ADR-0005 |
| [0004](./0004-the-batching-default-follows-the-device.md) | 2026-09-19 | The batching default follows the device | Accepted |
| [0005](./0005-metal-becomes-a-first-class-backend.md) | 2026-09-20 | Metal becomes a first-class backend | Accepted (supersedes ADR-0003) |
| [0006](./0006-kv-format-is-a-per-engine-gate.md) | 2026-09-25 | The KV storage format is a per-engine gate, not a process-wide global | Accepted |

## The template

```markdown
# NNNN. One-line decision title

- Status: Accepted | Proposed | Rejected | Superseded by ADR-NNNN
- Date: YYYY-MM-DD
- Issues: #NN, #NN            (optional)
- Supersedes: ADR-NNNN        (required when this ADR replaces one)

## Context

What the situation was, and what made it a decision rather than a default. Cite the record.

## Decision

What was decided, in one or two sentences, then the concrete facts (symbols, files, flags).

## Alternatives considered

Each alternative and the reason it lost — a measurement, a constraint, a cost. If the record holds
no rejected alternative, say so explicitly rather than inventing one.

## Consequences

What this makes easy, what it makes expensive, and which costs were knowingly accepted.

## References

The design doc or plan section that holds the current contract, and the issues/PRs involved.
```
