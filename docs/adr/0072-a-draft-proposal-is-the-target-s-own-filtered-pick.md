# 0072. A draft proposal is the target's own filtered pick, computed from one shared prefix

- Status: Accepted
- Date: 2026-11-02
- Issues: #651

## Context

The speculative round's contract (ADR-0010, ADR-0017) is that every emitted token is the **target's
own decision** at that position: the drafter only *predicts*, the target arbitrates. That makes the
drafter's selection rule look like a free implementation detail — a bad guess costs acceptance, never
correctness. It is not free, and the two halves of minfer's round were not the same function:

- the **target** picks the greedy token *after* the deterministic filter prefix — logit bias →
  penalties → DRY (`src/sampler.rs`) — and `repeat_penalty` defaults to **1.1** over a 64-token
  window;
- the **drafter** took the raw argmax of its own logits, with a comment that argued a filtered
  proposal "would only fail to match" (`src/spec.rs`, removed by this decision).

The premise was inverted. A control settles it: run **one model as both draft and target**, so the
proposals are the same weights on the same row and every proposal *should* survive. Measured
2026-11-02 on `macbookpro` (14B q4_K_M, raw 21-token prompt, greedy):

| acceptance, one model on both sides | `d = 1` | `d = 2` | `d = 3` |
|---|---|---|---|
| `--repeat-penalty 1.0` (filter is a no-op) | 85.3 % | **93.2 %** | 71.7 % |
| `--repeat-penalty 1.1` (default), raw argmax | — | 51.6 % | — |
| `--repeat-penalty 1.1` (default), **this decision** | 70.3 % | **58.6 %** | 50.7 % |

Two facts, and the second one is the larger:

1. the asymmetry is real and worth **+7 points** at `d = 2` (51.6 % → 58.6 %) — the drafter proposes a
   token the target's own window demotes, and loses a row it would otherwise have won;
2. **a perfect drafter still loses 7-29 %** (the `1.0` row): with identical weights on both sides, the
   proposals disagree with the verify rows because the verify's per-row logits are *not bitwise equal*
   to decode's — they are the tolerance class ADR-0010 names (~0.01-0.05 logits, the prefill attention
   kernel against the decode one). `repeat_penalty` **amplifies** that: dividing a positive logit by
   1.1 is a 10 % relative demotion, which reorders a top-k whose neighbours sit tenths apart, so the
   same 0.05-logit difference costs 30-49 points instead of 7-29.

So this decision is a **prerequisite, not the cure**: it makes the rule right, and the acceptance
ceiling stays set by the verify's logits until those are bitwise.

## Decision

**The drafter predicts the target's pick**: the argmax of its logits **after the deterministic filter
prefix**, over the same lazy window the accept loop builds — `prev_tokens` extended with each
proposal already committed in the round.

The prefix has **one implementation**, `sampler::apply_deterministic_filters`, called by both
`sample_with_config_grammar` and `spec::draft_proposal`. A stage added to the prefix reaches both
callers or neither; there is no second copy of the pipeline to drift.

Bounded by what it is *not*:

- **not a sampler** — no temperature, no RNG, no top-k/top-p: everything after the greedy shortcut is
  stochastic and cannot be predicted, and the prefix is exactly the part that is not;
- **not a re-weighting of acceptance** — the target still arbitrates, so a better prediction can only
  raise the hit rate, never change the emitted stream (measured: the 4-prompt identity battery stays
  4/4 on 7B q4_K_M + 0.5B q4_K_M at `n = 200`);
- **not a claim that a perfect drafter is accepted everywhere** — it is not, and the `1.0` row above is
  the proof; that residual is a separate item (§Consequences);
- **not a grammar carrier** — a grammar stays refused (ADR-0017): its mask is per-row automaton state,
  and one round samples several rows from a single state.

## Alternatives considered

- **Keep the raw argmax and document it as "target-sampler semantics".** Rejected: the documented
  premise ("a penalized proposal would only fail to match") is false, and it costs 7 points on a
  control where the drafter cannot be at fault.
- **Apply only the repeat penalty instead of the whole prefix.** Rejected: that is a second copy of the
  pipeline. Logit bias and DRY are the same class of stage, and a partial prefix re-diverges the moment
  one is added — the failure mode this decision exists to close.
- **Use one window for the whole round** instead of extending it per proposal. Rejected: the target's
  row `i` filters with the tokens committed *before* it (the lazy rule the accept loop already
  implements), so a fixed window is wrong at `i > 0` exactly where the penalty bites.
- **Turn the penalty off so both sides are unfiltered, matching llama.cpp's default.** Not taken here:
  it changes the generated text and is a product default, not a spec knob. It is priced as the
  largest single acceptance lever available (the `1.0` row), and the decision belongs to the default,
  not to the round.
- **Wait for bitwise verify logits and skip the rule change.** Rejected: the rule is wrong on its own
  terms, the fix is three lines and identity-safe, and it is what makes the later logits work pay off
  instead of landing on a drafter that throws half of it away.

## Consequences

- The drafter now pays one penalty pass over the ≤64-token window per proposal (the target pays the
  same pass for every row it samples) plus one `prev_tokens.clone()` per round. Round cost, and
  therefore `round_cost / decode_step`, is unchanged: this buys yield, not throughput.
- **Measured yield, stated honestly**: +7 points on the same-model control at the default penalty;
  unchanged on the real pair at `n = 64` (76.0 % templated, 48.4 % raw prompt) and +2.5 on the raw
  prompt at `n = 192` (39.3 % → 41.8 %). The real pair's proposals rarely cross the penalty boundary
  the way the self-control's do, so the fix's value on a production pair is currently small.
- **The acceptance ceiling is now a named item with a number**: with identical weights, 85-93 % at
  penalty 1.0 (not 100 %). Closing it means making the verify's per-row logits bitwise-equal to decode's
  — a different kernel (the batched prefill attention vs the decode one), and a reversal of ADR-0010's
  tolerance class for this band. Until then, any penalty with a relative demotion multiplies the
  residual, which is why llama.cpp's `penalty_repeat = 1.00` default reads 76.9 % where minfer's 1.1
  reads 58.6 % on the same models.
- Every future drafter — an n-gram lookup, a trained head, a precomputed router — inherits the rule:
  **predict the filtered pick, do not sample it.**
- The drafter's KV now holds *filtered* proposals, so a rejection point differs from the pre-fix runs
  while the emitted stream does not (identity battery 4/4).

## References

- [#651](https://github.com/yusiwen/minfer/issues/651) — the Metal speculative-parity campaign; the
  measurement record is `docs/METAL_OPTIMIZATIONS.md` §0.10.
- `docs/SPECULATIVE-DECODING-PLAN.md` §6 — the Metal row's current state.
- ADR-0010 (identity by default), ADR-0017 (what a round refuses to carry).
