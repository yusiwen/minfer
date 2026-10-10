# 0035. The q4_K dsc plane is admitted by two gates, and the payload test is equality

- Status: Accepted
- Date: 2026-10-10
- Issues: #480, #165

## Context

`prefill_mmq` looks up a weight's derived planes in maps keyed by the weight's **device pointer**, and one
of those planes is the q4_K `W_dsc` pre-decoded f32-pair form. Only one type can be decoded this way: the
dsc template that `mmq_raw_nb_bt` dispatches is a q4_K template. A weight of any other type admitted to
that plane would either be decoded from bytes it never had or read past the tensor, so the admission test
is a correctness gate, not a convenience check.

`docs/CUDA-BACKEND-DESIGN.md` §2.3 states the rule and its reasoning
(`src/q4k_dsc.rs`, `q4k_dsc_plane_admitted`, [#165](https://github.com/yusiwen/minfer/issues/165)).

## Decision

**The q4_K `W_dsc` plane is admitted for q4_K and only q4_K, by two gates that both have to hold.**

- **The type gate.** `TensorType::Q4_K` is the one type `mmq_raw_nb_bt` dispatches the dsc template for,
  so the loader admits no other type.
- **The payload gate.** `raw.len()` must be **exactly** `od * (id / 256) * 144`, q4_K's own block layout —
  **equality, not a lower bound**.
- **Both are re-checked where a direct caller could skip them.** `register_weight_q4k_dsc` re-checks the
  payload before the budget query and before `expand_q4k_dsc`, and `expand_q4k_dsc` itself returns `None`
  for a payload it cannot index.

## Alternatives considered

**The record holds no rejected alternative for this decision.** §2.3 states the rule and, separately, why
**one gate is not enough** — which is a justification of the chosen design, not a comparison against a
design that lost. It is recorded here as such, because ADR-0022 forbids inventing the comparison:

- a **q4_0** payload has exactly q4_K's bytes-per-element ratio (**18/32 == 144/256**), so the size check
  **cannot refuse it** — only the type gate can;
- a **q8_0** payload (**34/32**) is *longer* and would be misread as 144-byte q4_K super-blocks;
- a future type with a *smaller* ratio (a 2-bit K-quant: **84/256**) is *shorter* than the row arithmetic
  needs, so the size check is what **refuses it instead of reading past the tensor**.

A reader should not infer that a single-gate design, a lower-bound size test, or a
check-at-the-call-site-only scheme was weighed. None is mentioned.

## Consequences

- **Neither gate substitutes for the other**, and the record says why in both directions: the size check
  cannot tell a q4_K payload from another type's bytes of the same length — that is the type gate's job —
  and the type gate cannot catch a q4_K-typed payload whose length does not match the row arithmetic.
- **A direct caller cannot bypass either gate**: the registration path re-checks before the budget query
  and before the expansion, and the expansion refuses a payload it cannot index.
- The equality form is what makes the future-small-ratio case safe: a *shorter* payload is refused rather
  than indexed past its end, which is the behaviour a lower bound would have removed.

## References

- `docs/CUDA-BACKEND-DESIGN.md` §2.3 — the plane map, the two gates and the three payload classes.
- `docs/CUDA-BACKEND-DESIGN.md` §4.9 rule 5's #38 bullet — the sibling `dsc`-adjacent guard on the KV
  store, for the same "refuse rather than read past" shape.
- [#165](https://github.com/yusiwen/minfer/issues/165) — the ticket;
  [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
