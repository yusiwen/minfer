# 0002. Topology is a function of `GraphParams` alone, so `positions` cannot be structure

- Status: Accepted
- Date: 2026-08-21

## Context

The graph is reused across decode steps (ADR-0001), so something has to decide whether a cached
graph may be reused or must be rebuilt. The obvious candidate for "what changed" is the token
position: every decode step advances it. If the graph's *shape* depended on how far along a
sequence is, reuse would be impossible and every step would rebuild. The Phase-1 IR already
recorded the intent — "`n_past` never influences allocation" — and the reuse cache made it a
comparison of `GraphParams` (`a163a07`, `8091b61`, both 2026-08-21).

The decision predates this repository's issue tracker, so the commits are the record.

## Decision

**Topology is a deterministic function of `GraphParams`**, and `GraphCache::try_reuse` compares
params only. `n_past` never enters the identity; neither does the number of sequences.

A consequence was made explicit later (C6, 2026-09-20): a token has a *position* and a *cell*, and
they are different things.

| Input | Meaning | Consumed by |
|---|---|---|
| `positions` | the token's index **within its sequence** | RoPE (q and k), the causal bound |
| `cells` | the arena row the allocator resolves for `(sequence, position)` | `KvcacheStore`, `Op::FusedQKV`, `Op::QkvBiasRopeStore` |
| `attn_span` | the `[lo, hi)` **cell** range, resolved from the per-sequence span list | attention |

So moving a run in the arena changes `cells` and nothing else; RoPE still rotates by `positions`.
The window is resolved from a span list (position base, first cell, length) — explicitly **not**
`start + position`.

## Alternatives considered

- **Keep `positions` as the KV cell index** (equivalently, compute the store row as
  `start + position`) — the original design, and the plan's own "fatal flaw". Rejected on measured
  evidence: a sequence's logits tail changed when its run moved, because one number served as both
  the RoPE angle and the arena index, so a cell move shifted every rotation. At cell 0 vs cell 8
  the max |Δ| over the vocabulary was **2.6% relative** (greedy token unchanged). The entry point
  was isolated by bisection to layer 0's attention output, and the decisive intervention was
  **RoPE injection**: feeding run A's 48 rope outputs into run B made the logits bitwise
  identical. A hand-built q/k/v → rope → store → attn graph was exact to ≤ 1.2e-7 at both 1-cell
  and 8-cell offsets, and a distributed ~1e-6 rope perturbation already saturated the tail
  (0.44 vs 0.43).
- **Put the position (or the sequence count) in the reuse identity.** Rejected because it would
  force a rebuild on every step. Two dead identity fields were deleted on the way: `CParams.n_batch`
  (A7) and then `GraphParams.n_seqs` (E2) — the latter after a measurement showed a 2-sequence and
  a 1-sequence batch with equal `n_tokens`/`n_out`/`gtype`/`explicit_span` are the same topology
  and were nonetheless rebuilt (uid 3 → 4). The plan records the choice as "Dead reuse-identity
  fields: option (a), then (c)".

## Consequences

- A **compaction** (C3) moves cells verbatim and is **bit-identical**: since C6 it no longer
  re-ropes, and `kv_defrag(need)` lost its rope parameter. C3's acceptance therefore *tightened* —
  from a named amplified-rounding tolerance class to bitwise equality — and the post-C6 offset
  probes assert bitwise, which is why the old perturbation probe was deleted rather than kept.
- A **physical move** is a different thing and keeps its cost: `kv_rm` / `kv_shift` (C2) changes
  `positions`, so it *does* re-ropes. A packed `q8_0` region survives that through
  `kvformat::map_q8_0_cells`; an **f16** region refuses it loudly, pinned by a gate
  ([#306](https://github.com/yusiwen/minfer/issues/306)).
- A read path **refuses a multi-span sequence loudly**; a set-valued window (shared prefix plus a
  private run) goes through the `kv_map` input instead.
- Every path that feeds a batch must pass **sequence-relative** positions, never `run.start + pos`.
  That trap was not hypothetical: porting C6 exposed a server bug where `submit_on` still built
  `start + i` and was rejected as "past sequence 2's reserved run".
- Accepted: the fused decode QKV family is gated off under `explicit_span` until C6 S3 (Metal keeps
  the pre-C6 gate). A four-slot throughput A/B of the fusion was within noise (92.6 vs 92.7 tok/s),
  so the port's value is correctness, not speed.

## References

- `docs/COMPUTE-GRAPH-DESIGN.md` §1.4 and §6 — the invariant and the reuse flow.
- `docs/KV-CACHE-DESIGN.md` §1–§2 — the current contract for positions, cells and the cell store.
- `ARCHITECTURE-EXECUTION-PLAN.md` — C6's design and evidence, and the "Dead reuse-identity fields"
  row in §0.
- Commits: `a163a07` and `8091b61` (2026-08-21, the IR and the reuse cache); C6 landed in `9fe27e6`
  and closed in `001b8cc` (both 2026-09-20).
