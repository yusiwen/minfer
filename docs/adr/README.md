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
| Measurements, suite counts, per-ticket history | `scripts/test-baselines.toml`, `docs/TEST-BASELINES.md`, `docs/ARCHITECTURE-EXECUTION-PLAN.md` | Yes |
| The plan's phase counters, `next:` sentence and baseline commits | `scripts/status.toml`, `docs/ARCHITECTURE-EXECUTION-PLAN.md` | Yes |
| Unresolvable path citations in this corpus, each with a reason | `scripts/adr-citations.toml`, beside its checker | Yes |
| A machine-read record: fixture provenance, an accepted-annotation baseline | beside its readers, never in the book tree — `tests/fixtures/f6-fixtures.json`, `scripts/dead-code-baseline.toml` (ADR-0024) | Yes |

**What a map lists.** A design document ends with `## Decisions governing this document` — its
**map**. It lists the ADRs whose decision that document's current contract **depends on** (states
it, or requires it); naming another document's decision is a cross-reference, which belongs in the
prose and *not* in the map, so a document may legitimately mention an ADR its map omits
([ADR-0028](./0028-a-map-lists-what-the-contract-depends-on.md)).

Three rules keep it that way. Both are about **kind**, not about banning a form:

- **A capability is a dated consequence, never a present-tense claim.** "At the `Date:` above, #310
  enabled X" is allowed; "X is now true" is not — that is the sentence that rots. The authority for
  what is true *today* is the design doc and `docs/SUPPORT-MATRIX.md`, which are mutable.
- **A citation is dated — a path, a symbol and a count, not only a capability.** An ADR may say *"the
  ledgers then lived under `docs/`"*; it may not say *"the ledgers live under `docs/`"*. Cite the *kind*
  of home (the machine ledger, the design doc) or date the reference. An unresolvable path citation is
  either **pinned with a reason** in `scripts/adr-citations.toml` or it fails `check_adr.py`; telling
  history from staleness is that ledger's whole purpose
  ([ADR-0027](./0027-a-citation-is-dated-too.md)).
- **A number may be evidence, never a baseline.** A count that is part of an argument — the rejected
  alternative's failures, a named boundary — is frozen with the ADR and belongs in it. A *current*
  suite baseline belongs in `docs/TEST-BASELINES.md` and its ledger
  `scripts/test-baselines.toml`, which are
  machine-checked. An ADR that quotes today's suite result is a bug.

Three consequences worth stating outright:

- **Superseding is additive.** Correcting a decision means writing a new ADR and setting the old
  one's `Status` to `Superseded by ADR-NNNN`. The old text is never edited — the record of what we
  believed, and why, is the useful part.
- **Corrections are additive too.** A defect in a frozen ADR's *text* is corrected by a new ADR
  carrying `- Corrects: ADR-NNNN`, and the corrected ADR's row below names the corrector — both ends,
  enforced by `check_adr.py`. The old text is never edited. See
  [ADR-0022](./0022-a-defect-in-a-frozen-adr-is-corrected-by-a-new-adr.md) (three citations at once) and
  [ADR-0026](./0026-the-classifier-carries-no-docs-path-exception.md) (a path a follow-through had moved).
- **Numbers are citations.** They are dense from `0001`, never reused, never renumbered. A number
  is assigned once, in the order decisions are established when their ADR is written, and the
  `Date:` field carries the date the decision was actually *taken*. Because most of this corpus is
  backfilled from existing records, a decision discovered later keeps its true (possibly earlier)
  `Date:` and takes the next free number — so the sequence is *approximately* chronological, and
  `Date:` is always the authority. That is the cost of keeping numbers stable; stability is worth
  more here than a perfect chronology.

`scripts/check_adr.py` (CI job `check-docs`) enforces the mechanically checkable half: filename and
heading agree, dense numbering, one of four `Status` values, a `YYYY-MM-DD` date, the required
sections, every ADR listed below, `Superseded by` written from *both* ends, a `Corrects:` target
that exists, is lower-numbered and is named on its own index row, and every path citation either
resolving or pinned in `scripts/adr-citations.toml`.

## Index

**Ordered by `Date:`**, because a number is a citation and a date is the chronology. Backfilled
ADRs are numbered in the order they were written, so a lower number does not imply an earlier
decision — see the numbering rule above. Within one date, the order is by number.

| # | Date | Decision | Status |
|---|---|---|---|
| [0007](./0007-no-ml-frameworks-every-operator-is-hand-written.md) | 2026-06-24 | No ML frameworks: every operator is hand-written | Accepted (citation corrected by ADR-0024) |
| [0013](./0013-cpu-quantizes-activations-device-reads-f32.md) | 2026-06-24 | The CPU quantizes activations to `Q8_0`; a device reads f32 | Accepted |
| [0008](./0008-gpu-safety-bounded-waits-and-runtime-limits.md) | 2026-08-02 | GPU safety: bounded waits, no early return past a barrier, runtime device limits | Accepted |
| [0001](./0001-inference-runs-through-one-declarative-compute-graph.md) | 2026-08-21 | Inference runs through one declarative compute graph | Accepted (citation corrected by ADR-0022) |
| [0002](./0002-topology-is-a-function-of-graph-params.md) | 2026-08-21 | Topology is a function of `GraphParams` alone, so `positions` cannot be structure | Accepted |
| [0009](./0009-a-failure-is-an-error-never-a-silent-fallback.md) | 2026-08-21 | A failure is an error, never a silent fallback | Accepted |
| [0010](./0010-the-identity-gate-bitwise-by-default.md) | 2026-08-28 | The identity gate: bitwise by default, a named tolerance class otherwise | Accepted (citation corrected by ADR-0022) |
| [0003](./0003-metal-is-out-of-scope-for-this-round.md) | 2026-09-16 | Metal is out of scope for this round | Superseded by ADR-0005 |
| [0004](./0004-the-batching-default-follows-the-device.md) | 2026-09-19 | The batching default follows the device | Accepted |
| [0005](./0005-metal-becomes-a-first-class-backend.md) | 2026-09-20 | Metal becomes a first-class backend | Accepted (supersedes ADR-0003; corrected by ADR-0022) |
| [0014](./0014-a-kv-session-is-a-versioned-file.md) | 2026-09-22 | A KV session is a versioned, checksummed file — never a memory dump | Accepted |
| [0015](./0015-the-offload-auto-fit-takes-a-prefix.md) | 2026-09-23 | The offload `auto` fit takes a prefix, not a knapsack | Accepted |
| [0011](./0011-backend-ids-are-append-only.md) | 2026-09-24 | Backend ids are a file-format contract: appended, never renumbered | Accepted |
| [0016](./0016-a-failed-device-query-is-not-a-zero-budget.md) | 2026-09-24 | A failed device-memory query is not a zero budget | Accepted |
| [0017](./0017-speculative-decoding-refuses-what-it-cannot-carry.md) | 2026-09-24 | Speculative decoding refuses the features its identity contract cannot carry | Accepted |
| [0018](./0018-the-grammar-mask-is-one-pipeline-stage.md) | 2026-09-24 | The grammar mask is one stage inside the single sampler pipeline | Accepted |
| [0019](./0019-a-chat-template-that-cannot-render-refuses-the-load.md) | 2026-09-24 | A chat template that cannot be rendered refuses the load | Accepted |
| [0006](./0006-kv-format-is-a-per-engine-gate.md) | 2026-09-25 | The KV storage format is a per-engine gate, not a process-wide global | Accepted |
| [0020](./0020-a-quantized-file-is-byte-identical-or-wrong.md) | 2026-09-27 | A quantized file is byte-identical to `llama-quantize`, or it is wrong | Accepted (citation corrected by ADR-0024) |
| [0021](./0021-bf16-is-round-to-nearest-even-and-1d-stays-f32.md) | 2026-09-27 | bf16 is a round-to-nearest-even cast, and 1-D tensors stay f32 | Accepted |
| [0012](./0012-device-first-layering-and-no-premature-common.md) | 2026-10-04 | Device is the first axis, the layer the second — and no premature `common` | Accepted |
| [0022](./0022-a-defect-in-a-frozen-adr-is-corrected-by-a-new-adr.md) | 2026-10-09 | A defect in a frozen ADR is corrected by a new ADR, not by editing it | Accepted (corrected by ADR-0023 and ADR-0027) |
| [0023](./0023-the-machine-ledgers-live-beside-their-checkers.md) | 2026-10-09 | Each machine ledger lives beside its checker, one per prose target | Accepted |
| [0024](./0024-a-machine-read-record-is-not-book-content.md) | 2026-10-09 | A machine-read record is not book content | Accepted |
| [0025](./0025-a-docs-only-change-skips-the-compilers.md) | 2026-10-09 | A docs-only change runs the docs gate, not the compilers | Accepted (corrected by ADR-0026) |
| [0026](./0026-the-classifier-carries-no-docs-path-exception.md) | 2026-10-10 | The classifier carries no docs-path exception, and its list is ratcheted | Accepted |
| [0027](./0027-a-citation-is-dated-too.md) | 2026-10-10 | A citation is dated too — paths, symbols and counts, not only capabilities | Accepted (corrects ADR-0022) |
| [0028](./0028-a-map-lists-what-the-contract-depends-on.md) | 2026-10-10 | A document's ADR map lists what its contract depends on | Accepted (corrected by ADR-0029) |
| [0029](./0029-an-adrs-prose-count-is-dated-and-its-references-are-map-evidence.md) | 2026-10-10 | An ADR's prose count is dated too, and its References are map evidence | Accepted (corrects ADR-0028) |
| [0030](./0030-launch-severity-lives-in-the-helper.md) | 2026-10-10 | Launch severity lives in the helper, not in 120 call sites | Accepted |
| [0031](./0031-capture-runs-in-thread-local-mode.md) | 2026-10-10 | Capture runs in thread-local mode, because Global lets a foreign thread's call join the window | Accepted |

## The template

```markdown
# NNNN. One-line decision title

- Status: Accepted | Proposed | Rejected | Superseded by ADR-NNNN
- Date: YYYY-MM-DD
- Issues: #NN, #NN            (optional)
- Supersedes: ADR-NNNN        (required when this ADR replaces one)
- Corrects: ADR-NNNN          (when this ADR corrects an earlier one's text; see ADR-0022)

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
