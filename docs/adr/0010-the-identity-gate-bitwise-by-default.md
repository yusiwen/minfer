# 0010. The identity gate: bitwise by default, a named tolerance class otherwise

- Status: Accepted
- Date: 2026-08-28

## Context

By mid-2026 the engine had several ways to compute the same thing — quantized and `f32`-weight
matmuls, fused and unfused decode paths, CPU and device backends — and a change to any one of them
had to be shown safe. Two circumstances made the bar a real decision rather than a formality.

First, a cross-device whole-forward **can never** be bitwise: the CPU quantizes activations to
`Q8_0` while a device reads `f32`, so the two differ by design, and
`docs/CUDA-BACKEND-DESIGN.md` records the consequence as **Accepted** — gates use greedy-text
equality plus tolerance classes rather than a raw logit comparison.

Second, a specific kernel boundary forced the issue. In the D5-R speculative campaign, batched
verify (`nt > 1`) had to be byte-identical to sequential decode. Multi-MMVQ is bitwise only up to
`nt ≤ 8`; at `nt = 9` the matmuls fall onto the padded GEMM, whose accumulator lifetime differs.
So the identity bar had a *structural* boundary in it, and the question was what to do at the
boundary rather than pretend it was not there.

The tolerance-class language enters the record in the CUDA implementation plan (`d7ea013`,
2026-08-28).

## Decision

**Any change to a kernel or an execution path is A/B'd against the existing path.** The default bar
is **bitwise-identical**. Where bitwise is impossible, the change must **name the tolerance class
and its cause** — the slack is a decision with a stated reason, not a constant.

The recorded example is the prefill boundary: **`nt ≤ 8` bitwise, `nt = 9` tolerance-class**, while
that path's KV stays bitwise (established by a 48-layer dump).

Two further consequences of the same bar are explicit elsewhere in the tree:

- **Fused vs unfused is bit-identical.** When that comparison runs, the unfused path must run the
  `FusionPass`, or the two sides differ by the fusion's own rounding (~1e-6) and the comparison
  measures the wrong thing.
- **A default path is pinned bitwise against its predecessor.** Every F3 sampler knob defaults to a
  no-op and the default pipeline is asserted bit-identical to the pre-F3 chain; the same holds for
  the pre-F2 chain through the grammar-aware entry point.

## Alternatives considered

- **Extend multi-MMVQ to `nt` 9–16 so the bitwise guarantee holds across the boundary.** This is the
  alternative that would have *deleted* the tolerance class, and it was implemented and **rejected by
  measurement**: token-groups-of-8 reached parity (85.0 vs 84.9 ms matmul) but re-streamed the
  weights from DRAM instead of L2, and a single-pass `acc[16]` regressed to 111 ms from register
  spill. The conclusion recorded in the campaign is that "the doc-82 GEMM boundary at `nt ≥ 9`
  stands", together with the lesson: extrapolating a per-row slope across a structural boundary (the
  accumulator lifetime) is how plans go wrong.
- **A single absolute tolerance as the default** ("logits within 1e-3"). Rejected: it cannot see a
  regression below the bar, and it makes the bar unattributable — the author need not say *why* a
  difference exists, so nobody can tell a legitimate architecture difference from a bug. The
  recorded practice is the opposite: the class is *named*, and its cause recorded.
- **Compare mode A against mode B** (the two implementations, rather than each against a reference).
  Rejected, and later stated as the gate contract's first rule — "assert the value, not a relation
  between two code paths" — because a gate that only compares the two modes is blind to a fault they
  share.
- **Trust review and spot-checks.** Rejected: numeric drift is invisible in review, and the A/B is
  cheap precisely when the bar is exact equality (no thresholds to tune, no flakiness).

## Consequences

- The speculative path's output is byte-identical to sequential decode, and the adaptive controller
  is **capped at `d = 7`** precisely to stay inside the bitwise region. The tolerance class is not a
  licence to wander; it is a mapped boundary.
- A test's name states its bar (`*_bitwise*`, `*matches_the_cpu_reference*`), so a reader can see
  which class a path claims without reading the assertion.
- The default being bitwise makes the A/B runnable in CI: exact equality needs no threshold and no
  statistical argument, which is why so many gates run without a device.
- Asking for a tolerance is a visible decision — the intended friction. It is what made C3's
  compaction tolerance get re-examined, and C6 later **tightened** it to bit-identical rather than
  leaving a class in place (ADR-0002).
- The bitwise multi-token gate paid for itself: it caught a variable-shadowing corruption before
  that change shipped (acceptance 50.8% → 38.4%).
- Cost accepted: whole-forward cross-device comparisons cannot be bitwise, so they are judged on
  greedy-text equality plus a named class; and where bitwise really is impossible the author must
  characterize the difference rather than round it away.

## References

- `docs/ARCHITECTURE-EXECUTION-PLAN.md` §1 standing rule 3 (the rule and the `nt ≤ 8` / `nt = 9`
  example) and the D5-R record (`aac86df`, doc 87) that measured the boundary.
- `docs/CUDA_OPTIMIZATION.md` (doc 95, `0fe132f`) — "verify `nt ≤ 8` bitwise, `nt = 9` tolerance-class
  while its KV stays bitwise (48-layer dump proof)".
- `docs/GATE-CONTRACT.md` §1 — "assert the value, not a relation between two code paths".
- `docs/CUDA-BACKEND-DESIGN.md` — the "differ by design" row and the tolerance-class gates.
- Commit `d7ea013` (2026-08-28), where the tolerance-class language enters the record.
