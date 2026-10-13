# 0040. A decision is bound to the surfaces it governs, so the question names it

- Status: Accepted
- Date: 2026-10-11
- Issues: #541
- Corrects: ADR-0039

## Context

ADR-0039 decision 1 put the decision surfaces in a hard-coded table inside the classifier —
`DECISION_PATHS`, nine files whose subject is an ADR's subject. It answered *"is this an ADR
matter?"* and never *"which decision?"*: the question was generic, so the author of a change to
`src/graph/registry.rs` was told that an ADR was probably at stake, not that ADR-0011 fixes the id
space there. Two smaller problems came with it. A path cannot survive a move — the table would
silently stop matching after a module split, and only a selftest naming the paths would notice. And
the table was a *second* statement of something the corpus already holds: every ADR's `## References`
names its primary contract, which ADR-0029 already calls map evidence.

The measurement that shaped the fix came from the first draft of the ledger. Binding a surface by
**name occurrence** — "everywhere this token appears under `src/`" — matched **eight** bindings for a
single change to one Metal file, because `MINFER_CACHE_TYPE`, `MINFER_PACKED_KV_STAGE` and friends
appear in comments across the device layer. A question asked eight times is a question nobody reads,
and ADR-0023's rule (a gate earns its keep) applies to the *quality* of a trigger, not only to its
existence.

## Decision

**1. The binding lives in a ledger, and it is one of two kinds.** `scripts/adr-governs.toml` carries
`[[governs]]` entries: `adr`, `what`, and **exactly one** of

- `path = "src/…"` — the file itself is the surface, and the ratchet is that it still exists; or
- `symbol = "name"` — the surface is the **definition sites** of that identifier under `src/**/*.rs`
  (a `fn`/`struct`/`enum`/`trait`/`const`/`static`/`type`/`impl`/`mod` that defines it). A mere
  mention is not a surface.

A name this tree does not *define* — a `MINFER_*` knob, an FFI constant, a kernel symbol — is bound
by the file that implements the mechanism, which is the weaker form and is chosen deliberately over
the measured noisy one.

**2. The question names the decision.** The classifier reads the ledger, and for a range touching a
governed surface it reports `ADR-NNNN governs <surface> — <what>` (the `adr_question=` line, human
form on stderr). The PR-body checker quotes that line back into its failure, so the message is
*"`src/graph/registry.rs` is governed by ADR-0011 (backend ids are appended, never renumbered) — write
`ADR: 0011`, `ADR: supersedes 0011`, or `ADR: none — <reason>`"* rather than *"an ADR may be at
stake"*.

**3. The ledger is validated where the corpus is.** `scripts/check_adr.py` requires: the ADR exists;
`what` is non-empty; exactly one of `path`/`symbol`; a `path` exists and a `symbol` is defined
somewhere; and the ADR is **not** superseded — a superseded decision governs nothing, so the entry
must move to the successor, and the failure names it.

**4. Coverage is partial, and the checker says so.** An ADR may govern no code surface: a process
decision (the corpus, the gates), a scope decision (what a round includes), an upstream decision (the
identity gate). `check_adr.py` prints both numbers — bindings, and ADRs binding nothing — so partial
coverage is a stated fact rather than an implied one.

**5. `DECISION_PATHS` is gone.** One source of truth for "which surfaces are decision surfaces", and
it is the one the corpus can grow: a missed case is added as a binding with the ADR that made it one.

## Alternatives considered

- **Keep the hard-coded table (the text this corrects).** It worked, and it was the mechanism ADR-0039
  named — but it answered the weaker question, and a rename would have degraded it silently.
- **Bind by symbol name occurrence.** Rejected on measurement, recorded above: eight bindings for one
  file change is the reflex failure ADR-0023 warns about, and the reflex would have been pasting the
  waiver.
- **Bind only paths.** Rejected: it loses the property that motivated symbols — a module split or a
  file move keeps the binding, because the definition moved with it.
- **Derive the bindings automatically from each ADR's `## References`.** Rejected on measurement: only
  **9** of the 39 ADRs name a source path there, and the CUDA/Metal ADRs (0030–0038) name only their
  design document. A derived binding would miss precisely the code surfaces where a change can violate
  the decision, and it would look complete.
- **Require a binding for every ADR.** Rejected: 15 ADRs bind no code surface for legitimate reasons,
  and forcing one would create entries that fire on unrelated files — the reflex again, arrived at
  from the other side.
- **Put the binding in `scripts/adr-citations.toml`.** Rejected on kind, as ADR-0039 already argued:
  that ledger answers a fact about an ADR's *text* (a cited path no longer resolves); this answers a
  fact about a *surface* (whose shape is fixed by which decision). ADR-0023: one ledger per prose
  target.

## Consequences

- **A governed change gets a named question**, which is the point: the gate moves from *"think about
  whether this is a decision"* to *"ADR-0011 fixes this; are you continuing it or moving it?"*. The
  waiver stays one sentence, and now it names what it waives.
- **The ledger is a maintenance surface with its own ratchets.** A renamed file breaks the `path`
  entry; a moved definition breaks the `symbol` entry; a supersession breaks the entry outright. Each
  failure names the fix.
- **The symbol rule is itself a decision with a cost.** Binding definition sites means a change that
  *consumes* a shape without redefining it may not be asked (a new call site, a new branch in a
  dispatch that mentions the identifier). That is the trade the measurement bought: a trigger that
  fires on mentions is a trigger nobody reads.
- **The residue is still irreducible**, exactly as ADR-0039 recorded it: a decision whose surface
  nobody bound is not asked. The difference is that a missed case now has an obvious shape — add a
  binding — instead of a table to edit in a script.
