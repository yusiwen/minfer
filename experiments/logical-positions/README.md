# Logical positions — the cell-offset experiments (pre-C6 record)

Two `#[test]` probes that established **why** a sequence's logits changed when its KV
run moved. They were written against the tree *before* C6 and are kept here as the
record of that attribution, **not** as live gates: C6 (`positions` ≠ cells, merged as
`001b8cc`) removed exactly the coupling they diagnose, so their premise is gone and the
offset tests in `src/models/qwen2/graph.rs` now assert **bitwise** equality instead.

Context: before C6, `positions` was simultaneously the RoPE angle and the KV cell
index, so moving a run (what C3's compaction does) rotated every token differently and
the logits' tail moved by ~2.6% relative. The canonical record is
`docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 (C6) and §14 row 9; this directory keeps the
code and the numbers behind it.

## Files

| File | Purpose | Run | Result |
|---|---|---|---|
| `offset_experiments.rs` | EXP1 `offset_divergence_is_caused_by_the_rope_rounding_alone` + EXP2 `a_distributed_rope_perturbation_scales_like_the_offset` | see below | EXP1: logits **bitwise identical** after injection; EXP2: the tail delta saturates from ~1e-6 on |

## The two experiments

**EXP1 — causal injection.** Run the model's graph node by node at cell 0 (run A) and at
cell 8 (run C), capturing A's 48 RoPE outputs (2 per layer × 24 layers). Then run C again
and overwrite each RoPE node's output with A's captured values through
`Backend::write_host`, immediately after that node executes. If RoPE is the *only*
position-dependent entry, C's logits must come out **bitwise identical** to A's.

```
[inject] rope nodes injected: 48 | logits delta offset 0.4307506 -> injected 0
```

The `+0` control (the same code path with no perturbation) reads `0.0`, which is what
makes the zero above evidence rather than a silent no-op.

**EXP2 — distributed sweep.** The same runner without the capture: add the same delta to
**every element of every RoPE output** and sweep its magnitude. Printed, not asserted —
the shape of the response is the result:

| perturbation | max abs delta vs offset 0 |
|---|---|
| offset 8 vs 0 | 0.4307506 |
| rope +1e-6 | 0.4399 |
| rope +1e-5 | 0.3137 |
| rope +1e-4 | 0.4065 |
| rope +1e-3 | 0.4873 |
| rope +1e-2 | 0.3882 |

The greedy token is 12095 in every case: the tail saturates at a ~1e-6 **distributed**
perturbation without moving the argmax. The negative result that mattered is why the
sweep perturbs every element: a *single* element nudged by 1e-5 is not equivalent to the
offset's distributed 1.5e-5 (it can stay inside its quantisation bin), so a control
validates the path, not the equivalence of the perturbation. The earlier reading of a
single-element probe as a refutation of amplification was an over-read, corrected during
the campaign.

## How to run them

Both are test-module code: they use `crate::`-internal items, so they must live inside
`mod tests` of `src/models/qwen2/graph.rs`. They are **not** compiled from this directory
— nothing outside `src/` and `tests/` belongs to the crate. To re-run them, append the two
functions (verbatim, keeping their 4-space indentation) to that module and run:

```bash
cargo test --release offset_divergence_is_caused_by_the_rope_rounding_alone -- --nocapture
cargo test --release a_distributed_rope_perturbation_scales_like_the_offset -- --nocapture
```

Both need the cached Qwen2.5-0.5B Q4_0 GGUF (`cached_model_path()`) and skip with a message
when it is absent. Neither is `#[ignore]`d, so appending them makes them part of an
ordinary `cargo test` run.

**They do not compile against post-C6 `master` unchanged.** They were written when
`positions` *was* the KV row: the runner builds `positions = start..start + n` and relies
on the store writing at those rows, whereas C6 makes positions sequence-relative and has
the allocator resolve the `cells` input. Re-running them today means updating the runner to
cell 0 plus a `cells` input — the shape the current offset tests in
`src/models/qwen2/graph.rs` already use. One naming drift to expect: the plan text called
EXP2 `a_distributed_rope_perturbation_saturates_the_logits_tail`, while the code that was
stashed and is archived here is `a_distributed_rope_perturbation_scales_like_the_offset`.
