# 0003. Metal is out of scope for this round

- Status: Superseded by ADR-0005
- Date: 2026-09-16

## Context

`ARCHITECTURE-EXECUTION-PLAN.md` was written on a box that could not compile Metal at all: the
maintainer's machine was a DGX Spark (aarch64, GB10 SM 121) running Linux, and Metal is
macOS-only. The plan's verification matrix states it plainly — CPU and CUDA are buildable and
runnable there, Metal is neither, and the x86 AVX2/AVX-512 paths are also unavailable
(`ARCHITECTURE-EXECUTION-PLAN.md` §2). Any Metal change in that round would have been
compile-unverified and run-unverified.

## Decision

**Metal is out of scope for the round.** No ticket in the plan edits the Metal sources, and every
phase records what it defers into **Phase G (Metal alignment)**. The rule was given teeth: a
ticket whose cross-backend design changes Metal's behaviour must add a Phase G line *in the same
commit* ("Deferred-Metal marking", §1 standing rule 6).

## Alternatives considered

- **Carry Metal along in the same round.** Rejected on two grounds. First, the practical one: no
  macOS machine was available, so a Metal edit could not be compiled or run, and this repository's
  convention is that an unverifiable change is not a landed change (§2's verification matrix).
  Second, the ordering argument, which turned out to be the important one: the KV arena's shape
  was still moving (C7, C7b, C8 were all ahead), so porting Metal first would have meant porting
  it twice. Deferring it until the arena stopped changing meant the port could be written once.

## Consequences

- Every cross-backend design change carried an explicit "deferred to Phase G" note, so deferral
  was recorded rather than forgotten.
- The cost was real and was paid later: Metal drifted from the arena semantics for the whole
  round, and closing that gap became the Phase G work itself ([#44](https://github.com/yusiwen/minfer/issues/44)
  for the KV cell store, removal/shift and explicit attention span; [#54](https://github.com/yusiwen/minfer/issues/54)
  for the gap and parity measurements). The decision bought a stable target and cost a drift
  period; that is the trade this record preserves.
- Metal behaviour recorded as "not compilable here" had to be re-established on a Mac before the
  Metal claims in the docs could be trusted again.

## References

- `ARCHITECTURE-EXECUTION-PLAN.md` §0 (the decision row), §1 rule 6 (deferred-Metal marking) and
  §2 (the verification matrix).
- Superseded by **ADR-0005**, which records Metal becoming a first-class backend.
- Commit `0b2a380` (2026-09-16, "docs: add the architecture execution plan").
