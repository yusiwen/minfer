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

## When an ADR is required

The corpus cannot enforce its own use — `scripts/check_adr.py` can force a shape, not a
habit — so the trigger is written here, and the `changes`/`check-pr-body` jobs ask the
question at PR time. The test is **not** "is this change important?":

| Write an ADR when the change… | Do **not** when it… |
|---|---|
| picks or moves a shape that later work must obey: a file format, an id space, an op vocabulary, a byte-identity promise, a refusal rule, a new design contract | implements a decision already recorded — cite it, do not restate it |
| chooses between alternatives a reader would otherwise re-litigate | fixes a bug, a typo, a count or a broken citation |
| makes a mechanism's contract that a design document will then describe | measures something (that is a record, not a decision) |

`scripts/adr-governs.toml` carries the mechanical half — the binding from each ADR to the
surfaces it governs (a file, or an identifier's *definition sites*, so the binding survives a move) —
and a PR touching one of them, or adding a `docs/*-DESIGN.md`, must answer on one line, in the body:

```
ADR: 0011                                          # this implements that decision
ADR: supersedes 0011                               # this replaces it (new ADR + Status)
ADR: none — a comment-only edit to the gate table  # not a decision, and here is why
```

`check_pr_body.py` enforces the *presence* of that line and the shape of its answer, never
its truth — the same limit every other prose gate has. The failure quotes the binding, so it reads
*"`src/graph/registry.rs` is governed by ADR-0011 …"* rather than asking whether an ADR is involved
(ADR-0040). The trigger list is deliberately
short (ADR-0023: a gate earns its keep), because a question asked on routine work is
answered with a reflex, and a reflex is not a decision.

**What a map lists.** A design document ends with `## Decisions governing this document` — its
**map**. It lists the ADRs whose decision that document's current contract **depends on** (states
it, or requires it); naming another document's decision is a cross-reference, which belongs in the
prose and *not* in the map, so a document may legitimately mention an ADR its map omits
([ADR-0028](./0028-a-map-lists-what-the-contract-depends-on.md)).

**Three kinds of page carry no map, and that is not an omission** — each is a place a fact *about* a
decision lives, not a document whose contract depends on one:

- a **record** page: the plan and its per-ticket history (`docs/ARCHITECTURE-EXECUTION-PLAN.md`), the
  optimization records (`docs/CUDA_OPTIMIZATION.md`, `docs/METAL_OPTIMIZATIONS.md`), and a plan's own
  root-cause notes (`docs/QWEN3-SUPPORT-PLAN.md`);
- a **ledger** page: `docs/TEST-BASELINES.md`, whose numbers are machine-checked against
  `scripts/test-baselines.toml`;
- an **upstream comparison**: `docs/LLAMA-COMPUTE-GRAPH.md`, which is not this repository's contract.

**And a map may omit an ADR that names the page, in two cases** — both are references *about* a document
rather than a dependency of it:

- a **correction reference**: ADR-0022 names the documents whose citations it fixes (ADR-0029 records the
  class);
- a **contrast reference**: ADR-0037 names `docs/CUDA-BACKEND-DESIGN.md` §4.3 as a different mechanism, not
  as a dependency.

Three rules keep it that way. All three are about **kind**, not about banning a form:

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
| [0019](./0019-a-chat-template-that-cannot-render-refuses-the-load.md) | 2026-09-24 | A chat template that cannot be rendered refuses the load | Accepted (corrected by ADR-0042) |
| [0006](./0006-kv-format-is-a-per-engine-gate.md) | 2026-09-25 | The KV storage format is a per-engine gate, not a process-wide global | Accepted |
| [0020](./0020-a-quantized-file-is-byte-identical-or-wrong.md) | 2026-09-27 | A quantized file is byte-identical to `llama-quantize`, or it is wrong | Accepted (citation corrected by ADR-0024; corrected by ADR-0041) |
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
| [0032](./0032-the-packed-kv-staging-window-is-f32.md) | 2026-10-10 | The packed-KV staging window is f32, not f16 | Accepted |
| [0033](./0033-the-cuda-pool-recycles-exact-byte-lengths.md) | 2026-10-10 | The CUDA pool recycles exact byte lengths, never frees, and reports OOM as an error | Accepted |
| [0034](./0034-the-smem-opt-in-is-an-eager-prewarm-by-construction.md) | 2026-10-10 | The smem opt-in is an eager pre-warm by construction, with the lazy path as defence in depth | Accepted |
| [0035](./0035-the-q4k-dsc-plane-admission-is-two-gates.md) | 2026-10-10 | The q4_K dsc plane is admitted by two gates, and the payload test is equality | Accepted |
| [0036](./0036-bf16-gets-its-own-device-kernels.md) | 2026-10-10 | bf16 weights get their own device kernels, not a dtype flag on the f16 ones | Accepted |
| [0037](./0037-a-metal-weight-dtype-a-kernel-cannot-consume-is-refused.md) | 2026-10-10 | A Metal weight dtype a kernel cannot consume is refused, never run as a wrong kernel | Accepted |
| [0038](./0038-the-per-tensor-registration-dispatch-is-one-shared-rule.md) | 2026-10-10 | The per-tensor registration dispatch is one shared rule, not a copy per loader | Accepted |
| [0039](./0039-the-corpus-is-entered-through-a-question-at-pr-time.md) | 2026-10-11 | The corpus is entered through a question at PR time, and its maps are checked | Accepted (corrected by ADR-0040) |
| [0040](./0040-a-decision-is-bound-to-the-surfaces-it-governs.md) | 2026-10-11 | A decision is bound to the surfaces it governs, so the question names it | Accepted (corrects ADR-0039) |
| [0041](./0041-a-row-that-cannot-take-the-targets-block-is-demoted.md) | 2026-10-13 | A row that cannot take the target's block is demoted, the legacy targets included | Accepted (corrects ADR-0020) |
| [0042](./0042-the-tokenizer-applies-the-models-declared-nfc-normalizer.md) | 2026-10-15 | The tokenizer applies the model's declared NFC normalizer | Accepted (corrects ADR-0019; corrected by ADR-0044) |
| [0043](./0043-a-special-token-is-declared-by-the-file-never-inferred-from-its-name.md) | 2026-10-15 | A special token is declared by the file, never inferred from its name | Accepted |
| [0044](./0044-the-input-normalizer-follows-the-pre-tokenizer-family.md) | 2026-10-15 | The input normalizer follows the pre-tokenizer family's declaration | Accepted (corrects ADR-0042) |
| [0045](./0045-the-tokenizer-engine-is-a-dispatch-on-tokenizer-ggml-model.md) | 2026-10-15 | The tokenizer engine is a dispatch on `tokenizer.ggml.model` | Accepted |
| [0046](./0046-wordpiece-is-the-third-engine-and-follows-berts-normalizer.md) | 2026-10-15 | WordPiece is the third engine, and it follows BERT's normalizer | Accepted (amends ADR-0045) |
| [0047](./0047-the-unigram-engine-reads-the-files-charsmap-and-sentencepiece-is-its-reference.md) | 2026-10-15 | The unigram engine reads the file's charsmap, and SentencePiece is its reference | Accepted (amends ADR-0045) |
| [0048](./0048-gemma4-is-a-bpe-over-raw-utf8-and-its-rule-comes-from-the-model-value.md) | 2026-10-15 | gemma4 is a BPE over raw UTF-8, and its rule comes from the model value | Accepted (amends ADR-0045) |
| [0049](./0049-rwkv-is-a-trie-over-decoded-bytes-and-its-unknown-role-is-conditional.md) | 2026-10-16 | rwkv is a trie over decoded bytes, and its unknown role is conditional | Accepted (amends ADR-0045) |
| [0050](./0050-plamo2-is-a-backward-dp-over-the-files-own-scores-and-the-automaton-is-an-index.md) | 2026-10-16 | plamo2 is a backward DP over the file's own scores, and llama.cpp's automaton is an index | Accepted (amends ADR-0045) |
| [0051](./0051-hybriddna-is-the-gpt2-engine-plus-a-dna-kmer-block.md) | 2026-10-16 | hybriddna is the gpt2 engine plus a DNA k-mer block, and its vocabulary is escaped in place | Accepted (amends ADR-0045) |
| [0052](./0052-none-is-a-vocabulary-less-file-and-the-checked-entry-points-are-how-it-refuses.md) | 2026-10-16 | none is a vocabulary-less file, and the checked entry points are how it refuses | Accepted (amends ADR-0045) |
| [0053](./0053-whitespace-is-the-gpt2-engine-behind-a-whitespace-cut-and-has-no-artifact-gate.md) | 2026-10-17 | whitespace is the gpt2 engine behind a whitespace cut, and it has no artifact gate | Accepted (amends ADR-0045) |
| [0054](./0054-jina-v1-en-is-the-plain-gpt2-pre-rule.md) | 2026-10-17 | jina-v1-en is the plain GPT-2 pre rule, not a Qwen variant | Accepted |
| [0055](./0055-ignore-merges-is-a-whole-piece-shortcut-and-the-llama3-group-sets-it.md) | 2026-10-17 | ignore_merges is a whole-piece shortcut, and the llama3 pre group sets it | Accepted |
| [0056](./0056-a-missing-byte-token-is-accepted-only-when-no-token-can-reach-it.md) | 2026-10-17 | a missing byte token is accepted only when no token can reach it | Accepted (amends ADR-0045) |

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
