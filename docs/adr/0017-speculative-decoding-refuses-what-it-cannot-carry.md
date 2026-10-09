# 0017. Speculative decoding refuses the features its identity contract cannot carry

- Status: Accepted
- Date: 2026-09-24
- Issues: #87, #47, #48

## Context

Speculative decoding's greedy contract is **not** "the output looks right" — it is that a batched
verify round is *bitwise equal* to sequential decode. A round is one `nt = d+1` forward through
`forward_graph_cached`, and the plan's own gate recorded that batched verify against `nt = 1` decode
kernels differ by ~0.01–0.05 logits — the same "differ by design" class as ADR-0013 — so the identity
has to be established per kernel band, not assumed:

- verify `nt ≤ 8` is **bitwise** (single + multi MMVQ);
- `nt = 9` is a **tolerance class** (the BT-MMQ `lm_head`), which is why the adaptive depth is capped
  at `d = 7` (ADR-0010's boundary).

Three features cannot be carried through that contract without either a state the round never commits
or a different reduction schedule.

## Decision

**Each incompatible feature is refused loudly at construction or startup**, rather than silently
sampled on a different schedule.

- **Mirostat** — refused. `mu` is per-decode-step state, while one round samples several rows from a
  single shared RNG, so carrying it faithfully would need a per-position state the round never
  commits.
- **A grammar or JSON schema** — refused. A verify round samples several rows from *one* automaton
  state, and the draft model would need the same mask.
- **A packed KV cache** — refused. `SpecEngine::new` returns: *"speculative decoding needs the batched
  verify kernel (1 < nt <= 16) to be bitwise-equal to sequential decode, and a {} KV cache routes that
  band to the general attention kernel instead; run the target with MINFER_CACHE_TYPE=f32 or f16, or
  without --spec-draft"*.

The escape hatches are named in the refusal: a non-packed cache, or no `--spec-draft`. A static
`--spec-draft-n 8` remains available for throughput and is documented as **not** identity-safe.

## Alternatives considered

- **Mirostat: sample without the carried state, or with a per-position state the round never
  commits.** Rejected: the CLI and the server refuse `mirostat + speculative` loudly, *never silently
  sampling without it* — a silent difference in the sampling schedule is invisible in the output.
- **Packed KV cache: run the round anyway on the general `gqa_attn_f32` kernel.** Rejected: the
  batched split kernel (`gqa_attn_split_partial_bt`) exists *for* spec-verify's bitwise identity with
  sequential decode, and rather than claim that contract without measuring it, a speculative session
  refuses a packed cache. The code comment states the sharper reason: *"silently verifying on a
  different reduction schedule would make the draft's acceptance sampling wrong in a way no output
  check would catch."*
- **Grammar: carry the same mask through the verify round** (or only through the target). Rejected:
  *"the draft model would need the same mask: refused rather than silently generating without the
  constraint."*

## Consequences

- The greedy identity guarantee holds **by construction**: an unsupported combination cannot be
  started, so there is no configuration in which the guarantee is quietly weaker.
- `--spec-draft` therefore cannot be combined with mirostat, a grammar/JSON schema, or a packed KV
  cache; the adaptive controller is capped at `d = 7`; and `--spec-draft-n 8` is available but
  documented as outside the identity guarantee.
- Other limits accepted: the draft model doubles the footprint (0.5 GB on top of 9 GB on the measured
  box), the flag is opt-in, and Metal parity was recorded as *"a follow-up, not a gate"*.

## References

- `src/spec.rs` — `SpecEngine::new` and the packed-cache refusal; `src/main.rs` — the mirostat and
  grammar refusals.
- `docs/SPECULATIVE-DECODING-PLAN.md` — the D5 plan and its measurements.
- `ARCHITECTURE-EXECUTION-PLAN.md` — the D5-R record (module landed 2026-09-12) and gate G1's
  batched-verify-vs-decode observation.
- Commits: `a6b7cf3` (2026-09-12, the decode loop); the three refusals land 2026-09-24 with
  `bc414f7` (mirostat), `0a183c9` (grammar) and `0b5823f` (packed cache).
