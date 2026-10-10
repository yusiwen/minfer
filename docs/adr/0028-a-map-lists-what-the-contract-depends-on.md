# 0028. A document's ADR map lists what its contract depends on

- Status: Accepted
- Date: 2026-10-10
- Issues: #454

## Context

Each design document ends with a section `## Decisions governing this document` — its **map** — listing
the ADRs whose decisions that document carries. [#439](https://github.com/yusiwen/minfer/issues/439) S3
built those maps by hand and never wrote down what "governing" means.

The audit of all 16 maps ([#454](https://github.com/yusiwen/minfer/issues/454), two independent read-only
passes against the 27-ADR corpus) found **no padded map** — every ADR currently listed is genuinely
carried — and three real defects, of which GATE-CONTRACT's missing three and one live citation are fixed
in [#471](https://github.com/yusiwen/minfer/pull/471). But roughly **22 candidate additions** turned on a
question the corpus had not answered, and the two readings differ by everything:

- **Reading A — realized.** List every ADR whose decision the document's content realizes.
- **Reading B — home.** List only the ADRs whose decision this document is the *contract home* of.

Two pieces of evidence bear on the choice, and neither settles it alone. The **existing 16 maps are all
A-shaped** (for example `KV-CACHE-DESIGN.md` lists ADR-0002 for the topology invariant its cell store
depends on, not because the KV document is that decision's home), and no map is over-listed — an
A-shaped corpus that never over-listed suggests A is what the authors meant. But **inverting the ADRs'
own `## References`** — the mechanical candidate for a definition — derives only **2** entries for
`CUDA-BACKEND-DESIGN.md` and `METAL-BACKEND-DESIGN.md`, whose maps correctly list **9** and **10**: a
References section names a decision's *primary* contract, not every document whose contract depends on
it. So References corroborate (they independently derive the additions for GPU_SAFETY→0009,
GATE-CONTRACT→0023, BACKEND-REGISTRY→0008/0009, ARCHITECTURE→0007, COMPUTE-GRAPH→0004 and others) but
cannot be the rule.

## Decision

**1. A map lists the ADRs whose decision this document's current contract depends on.** "Depends on"
means the document states, or its semantics require, the obligation that ADR froze. An embodied
consequence counts: a document that states Metal's KV layout depends on Metal being a first-class
backend, so that ADR belongs in its map.

**2. A bare cross-reference is not a dependency.** Naming another document's decision — "see ADR-NNNN"
— belongs in the prose, not the map. A document may therefore legitimately mention an ADR its map does
not list, and `CUDA-BACKEND-DESIGN.md` is the worked example: its pointer to ADR-0005 (added by
[#449](https://github.com/yusiwen/minfer/pull/449) when a Metal paragraph moved out) is a
cross-reference, and its map correctly omits that ADR.

**3. The "body mentions an ADR the map omits" probe is informational, not a criterion.** It flags
candidates for a human and *expects* one class of false positive — the cross-reference in rule 2. It is
not a gate, and it never was: the acceptance criterion it looked like is withdrawn.

**4. The map stays curated, and is not required to equal the ADR→document References.** The two answer
different questions ("what must I know to read this contract?" vs "where does this decision's contract
live?") and may legitimately differ in both directions.

**5. The map is updated in the same PR that makes a document's contract newly depend on an ADR.**

## Alternatives considered

- **Reading B, "the document is the decision's sole home".** Rejected: it would *remove* entries from
  maps the audit verified as carried — `CUDA-BACKEND-DESIGN.md` lists ADR-0001/0002/0009/0010 whose homes
  are the graph and gate documents — so it contradicts all 16 existing maps, and "home" is not decidable
  for a cross-cutting decision that several documents realize.
- **Derive the maps from the ADRs' `## References`.** Kept as corroboration, rejected as the rule: it
  derives 2 entries for documents whose maps correctly list 9 and 10, and because an ADR is immutable a
  References gap could only be repaired with an errata ADR each time — the rule would make the corpus
  harder to keep true than the maps it replaced.
- **Gate the probe (body mentions ⇒ map entry).** Rejected: rule 2 makes a cross-reference correct, so
  the gate would fire on correct documents. It stays informational, like the candidate matrix.
- **Leave the definition implicit.** Rejected: this audit produced two readings differing by ~20 entries
  across 12 documents, and the next audit would re-litigate it. Writing it down costs one ADR.
- **List nothing in a map and rely on the ADRs' References.** Rejected: the map answers the reader's
  question from inside the document they are reading, which is the whole point of the backlink section.

## Consequences

- The audit's 22 additions land under rule 1, across 10 documents; each was verified by reading the
  document, not by trusting the audit's quotation. Two borderline candidates are **not** added and are
  recorded instead: `SUPPORT-MATRIX.md`→ADR-0014 (it states a session *consequence*, not the session
  file's contract) and `CPU_OPTIMIZATIONS.md`→ADR-0007 (hand-written kernels are a result of that
  decision, not an obligation the document states).
- A legitimate cross-reference stops reading as a defect, which is why `CUDA-BACKEND-DESIGN.md` is left
  untouched by the sweep that this ADR authorizes.
- Cost accepted: the maps can drift from the ADR corpus, and the only mechanical help is the
  informational probe plus the References inversion. Both are cheap to run and neither is a gate, so a
  wrong map is caught by review, not by CI.
- A map remains a *navigation* aid: it says which decisions to read before trusting the page, and nothing
  more.

## References

- `docs/adr/README.md` — the boundary rule and the map's one-line definition.
- [#454](https://github.com/yusiwen/minfer/issues/454) — the audit, both passes, the per-document
  evidence, and the two candidate defects that did not survive checking.
- [#471](https://github.com/yusiwen/minfer/pull/471) — the unambiguous fixes (GATE-CONTRACT's map, one
  live citation, one policy line).
