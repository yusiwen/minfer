# 0029. An ADR's prose count is dated too, and its References are map evidence

- Status: Accepted
- Date: 2026-10-10
- Issues: #478
- Corrects: ADR-0028

## Context

The campaign review found two defects in ADR-0028, both of the class the corpus has already legislated
against:

1. **A prose count that was stale the moment it was written.** ADR-0028 says the audit ran "against the
   **27**-ADR corpus". There are **28** — ADR-0028 is the 28th. ADR-0027 corrected exactly this pattern in
   ADR-0022 ("all 21 earlier ADRs stay byte-stable"), and ADR-0028 was written after that correction.
2. **Three map candidates were judged "not carried" against the ADRs' own text.** ADR-0028's
   *Consequences* record two of them as deliberately not added. A two-way check — an ADR's `## References`
   naming a document as its contract, against that document's map — finds:

   | ADR | Document | ADR-0028's verdict |
   |---|---|---|
   | ADR-0007 (no ML frameworks) | `docs/CPU_OPTIMIZATIONS.md` | not added, "hand-written kernels result from that decision rather than stating an obligation" |
   | ADR-0007 (no ML frameworks) | `docs/GGUF-TOOLING.md` | not added, "an upstream tool mention" |
   | ADR-0011 (backend ids append-only) | `docs/KV-CACHE-DESIGN.md` | not added, "only the References inversion derived it" |

   In each case the *ADR itself* names that document as the contract for its decision while the
   document's map omits the ADR: the document says "not mine", the record says "theirs".

The same check flags ADR-0022 → `METAL-BACKEND-DESIGN.md`, `CUDA-BACKEND-DESIGN.md`, `SUPPORT-MATRIX.md`.
Those are **expected false positives**: ADR-0022 names them because it corrects citations *in* them, not
because they carry the errata mechanism. The distinction matters, which is why this ADR writes it down.

## Decision

**1. A count is dated, wherever it appears — including inside an ADR.** ADR-0028's "27-ADR corpus" is read
as of its `Date:`. This ADR records that the corpus has grown (it is 29 with this one) rather than editing
frozen text, which is what the errata channel is for.

**2. The three entries are added.** `docs/CPU_OPTIMIZATIONS.md` and `docs/GGUF-TOOLING.md` gain ADR-0007;
`docs/KV-CACHE-DESIGN.md` gains ADR-0011.

**3. ADR-0028's rule 4 is narrowed: an ADR's own `## References` naming a document is evidence that the
document's contract depends on the decision.** A map may still omit such an ADR, but only with a written
reason — the default is now to add it. Rule 4's "the two may differ" remains true, but it now names the
class it was written for: **a correction reference** (ADR-0022 → the three documents whose citations it
fixes) is not a dependency, and neither is a bare cross-reference (ADR-0005 → `CUDA-BACKEND-DESIGN.md`,
ADR-0028 decision 2).

**4. The expected false-positive class is recorded** so the next audit does not re-file it: a `Corrects:`
ADR naming the documents whose text it corrects.

## Alternatives considered

- **Leave the count and rely on the reader to notice.** Rejected: ADR-0027 already ruled that a prose
  count is exactly the thing that rots, and this one rotted within the same working day.
- **Accept the three mismatches under ADR-0028 rule 4.** Rejected, and this is the substantive call:
  rule 4 was written for *genuine* differences of question ("what must I know to read this contract?" vs
  "where does this decision's contract live?"), not to excuse three candidates the audit had raised and
  the ADR's own References had already answered. Leaving them would make the reverse navigation ADR-0028
  promises one-way for exactly the documents that need it most.
- **Add every two-way mismatch, including ADR-0022's three.** Rejected: it would put the errata mechanism
  in the maps of three documents that do not carry it, which is the "padded map" failure the audit found
  none of.
- **Edit ADR-0028's text.** Rejected by ADR-0022: the old text is the record of what was believed.

## Consequences

- Three maps gain an entry, and the two-way check drops from 6 mismatches to the **3 expected meta cases**,
  which decision 4 names.
- A prose count inside an ADR joins the present-tense capability as a named rot class; the boundary rule
  in `docs/adr/README.md` already covers it by kind ("a capability is a dated consequence" / "a number may
  be evidence, never a baseline") — this ADR adds the third instance, a *corpus self-count*.
- ADR-0028's text still reads "27-ADR corpus". That is the accepted cost of immutability, and the reader
  reaches this ADR through ADR-0028's index row.
- Cost accepted: the "References are evidence" rule makes a map's default slightly stronger, so a future
  ADR whose References name a document for a non-contract reason (a correction, a cross-reference) must
  say so — which is the written reason decision 3 asks for.

## References

- [ADR-0028](0028-a-map-lists-what-the-contract-depends-on.md) (corrected here),
  [ADR-0027](0027-a-citation-is-dated-too.md), [ADR-0022](0022-a-defect-in-a-frozen-adr-is-corrected-by-a-new-adr.md).
- `docs/adr/README.md` — the map definition and the index row for ADR-0028 naming this correction.
- [#478](https://github.com/yusiwen/minfer/issues/478) — this decision.
