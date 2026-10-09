# 0018. The grammar mask is one stage inside the single sampler pipeline

- Status: Accepted
- Date: 2026-09-24
- Issues: #47

## Context

F3 (2026-09-24) had just consolidated sampling into **one** `SamplerConfig` pipeline, with the
property that every F3 knob defaults to a no-op so the default path stays bit-identical to the
pre-F3 chain. Issue #47 asked for constrained decoding and pinned both halves of the acceptance in
advance: *"The mask is applied at the same point as the other samplers, and greedy output with no
grammar is unchanged (bitwise)."*

The design question is where a mask belongs in a chain that both *shifts* logits (penalties, DRY,
temperature) and *removes* candidates (top-k, top-p, min-p, XTC).

## Decision

**The mask is one stage inside the single existing pipeline**, placed after every logit-shifting
stage and before every candidate-removing stage, and it touches no RNG state:

```
logit bias → penalties → DRY → [GRAMMAR MASK] → greedy shortcut
           → top-k → typical → top-p → min-p → XTC → temperature | mirostat
```

- `sample_with_config_grammar` is the implementation in `src/sampler.rs`, and `sample_with_config`
  **forwards to it** with no grammar — there is one pipeline, not two.
- Forbidden logits are set to `-inf`, exactly as `apply_top_k` / `apply_min_p` do, so the downstream
  filters' "survivors" logic (`v > -inf`) needs no special case.
- `SamplerConfig.grammar: Option<Arc<Grammar>>` is immutable per request; the mutable `GrammarState`
  is owned by the run, exactly like `MirostatState`.

## Alternatives considered

- **Mask earlier, before DRY and the penalties.** Recorded and rejected — and interestingly, not on
  correctness: *"Earlier would also be safe (a finite shift cannot lift `-inf`), but later is stated
  as the invariant, independent of how the penalty stages evolve: **nothing downstream of the mask
  may add**."* The later position is chosen because it is a rule that survives future stages, not
  because the earlier one is wrong today.
- **A mask that consumes RNG or writes shared state.** Rejected by the recorded property: the mask
  consumes no RNG and writes nothing the other samplers read, so mirostat's `mu` trajectory and DRY's
  deterministic penalties are unchanged for the same token sequence; the F2 bitwise gate pins the
  no-grammar path.
- **A separate sampling entry point for grammars.** The record states the one-pipeline invariant
  (*"One pipeline, in `src/sampler.rs`; there is no second sampler path"*) rather than labelling a
  second path as a considered-and-rejected option — so this ADR does not claim a rejection the
  record does not contain.
- **Masking before the greedy shortcut** is separately justified: so `--greedy` respects the grammar.

## Consequences

- `--greedy` respects the grammar; the no-grammar path is bit-identical; downstream filters need no
  special case.
- Cost: the mask is O(vocabulary) per **new** state — 5.4 ms on a 151k vocabulary, cached per state —
  so a long constrained generation pays it once per unseen state. The first version cost 71.3 ms per
  state (13× the memoized cost), which is why the cache exists.
- Accepted limitations, recorded rather than hidden: object properties are accepted in *declaration
  order*, `oneOf` is compiled as `anyOf` (not exclusive), and `pattern` / `minLength` / `maxLength` /
  `multipleOf` are **refused** — never approximated. Unsupported GBNF constructs are startup/`400`
  refusals.
- All acceptance is CPU-only: the mask is host-side, before any device work, so CUDA and Metal are
  untouched by this decision.
- Follow-ups recorded at the time: [#125](https://github.com/yusiwen/minfer/issues/125) (the refused
  GBNF/JSON-Schema constructs) and [#126](https://github.com/yusiwen/minfer/issues/126) (index the
  vocabulary by first codepoint).
- **Contradiction to note:** the pinned gate is named
  `test_default_pipeline_matches_the_pinned_pre_f2_sequence` in the F2 record but
  `test_default_pipeline_matches_the_pinned_pre_f3_sequence` in `AGENTS.md`. They are the same gate
  naming different baselines; a reader following one name will not find it.

## References

- `docs/GRAMMAR-DESIGN.md` — the accepted subset, the refusals and the mask position.
- `docs/USAGE.md` ("Sampler pipeline and invariants") — the pipeline and the CLI surface.
- `src/sampler.rs` — `sample_with_config_grammar`, `apply_top_k` / `apply_min_p`.
- Commit `0a183c9` (2026-09-24, "feat(grammar): F2 … (#47)"), with the record `d9efdb8`.
