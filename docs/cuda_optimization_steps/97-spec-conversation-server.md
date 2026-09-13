# 97 · Speculative Decoding — Conversation & Server Integration

**Status**: ✅ done (2026-09-15). Speculative decoding (static d and the adaptive-d
controller, doc 95) is now available in **all three frontends** — the CLI gen loop,
the multi-turn conversation (`--cnv --spec-draft`), and the OpenAI-compatible server
(`serve --spec-draft`). All identity gates are byte-identical; the suite is
187 passed / 0 failed (baseline 185 + 2 new spec-loop tests).

## 1. Design

One integration seam per layer, no duplicated model logic:

- **`conversation::Engine` trait** gains two default methods:
  `has_spec() -> bool` (default `false`) and
  `spec_round(seed, pos, sparams, prev_tokens, rng) -> Option<Vec<u32>>`
  (default `None`). Mock engines and plain `GraphEngine` keep the defaults, so
  every existing L1 conversation test compiles and passes unchanged.
- **`conversation::SpecAwareEngine`** wraps a `GraphEngine` plus an
  `Option<SpecEngine>` and the turn's `SpecSampler`. `forward`/`reset_cache`
  delegate; `spec_round` drives the real `SpecEngine::round` against the *same*
  target model + KV cache the plain path uses (one KV state, doc 94's identity
  contract). `reset_cache` also drops the draft KV (`SpecEngine::reset_draft`,
  new) — draft positions are absolute and must rewind with the target on
  `/clear` and full re-renders.
- **`Conversation::generate_assistant_spec`** — a sibling of the plain decode
  loop (not a branch inside it), dispatched on `engine.has_spec()`. It mirrors
  the CLI gen loop's proven round structure: sample the seed from the entry
  logits, then per round commit a batch of 1..=d+1 pre-sampled tokens through
  the same per-token machinery (EOG / stop strings / penalty window / UTF-8
  holdback / streaming).
- **Server**: `chat::generate` dispatches to a spec-aware `generate_seq` (one
  function, `Option<&mut SpecEngine>`); `worker_loop` builds one draft engine
  per slot (the draft is small; each slot's draft KV is isolated exactly like
  its target KV) and calls `generate_spec` for spec slots. `serve --spec-draft
  <model> [--spec-draft-n N] [--spec-draft-adaptive]` wires the parsed CLI
  options into `server::run`. The viz path passes `None`.

## 2. Position & termination contract (the part that must be exact)

`current_pos` is the slot of the newest committed-but-unwritten token (the
round's seed). A round writes the seed's row plus every batch row except the
last, so:

- full batch committed → `current_pos += batch.len()`, batch's last token is
  the next seed;
- stop-string cut at index i → tokens up to i−1 stay committed,
  `current_pos += 1 + i` (the stop token is never committed; its spec-written
  row is stale and overwritten before it is ever read);
- EOG at index i → committed through i, `current_pos += 1 + i + 1`; if the EOG
  is the batch's last token its row is unwritten, so the conversation loop
  forwards it explicitly (§5.4 cross-turn KV consistency);
- loop-top cap/context exit in the conversation → the newest committed token
  (the seed) is flushed with one nt=1 forward before the turn ends.

The server has no cross-request KV continuity (rows 0.. are rewritten before
they are read), so its EOG paths skip the explicit write. Termination flags
(`finish_reason`, `stopped_by_string`, `stopped_by_eog`) are checked at every
level: a mid-batch break ends the *turn*, not just the batch (§3, bug ③).

## 3. Bugs found during integration (all fixed, each with a test or gate)

1. **Duplicate penalty-window pushes** — the round's accept loop already pushes
   every emitted token into `prev_tokens`; pushing again in the batch loop
   skewed the repeat penalty (conversation; caught by the identity battery at
   token ~15).
2. **Stale-logits re-sampling** (server) — the server loop sampled a fresh seed
   from the never-refreshed prefill logits every iteration → the first seed
   repeated forever ("George WashingtonGeorge Washington…"). Fixed with an
   explicit seed carry: sample once, then the previous batch's last token is
   the seed.
3. **Termination leak** (server) — a mid-batch stop/EOG break only left the
   batch loop; the outer loop then ran another round after the turn was
   already finished. Fixed: a mid-batch break ends the turn.
4. **`#[derive(Debug)]` swallowed** by inserting the `BreakKind` enum between
   the derive and `TurnOutcome` (caught by the test build).
5. **EOG counting mismatch** — the spec loop counted the EOG in
   `tokens_generated`; the plain loop breaks before its `n_gen += 1`. Mirrored
   exactly (the new unit tests pin this).

**Pre-existing server bug (fixed in passing)**: a slot's `GraphCache` was
reused across requests. The graph path's persistent KV regions are built for
append-only sessions (the conversation resets on any full re-render), so
re-prefilling a *different* prompt over the same regions left stale rows inside
the new attention window — the **plain** server path was contaminated too
(req1 → req2 produced wrong output; two identical requests in a row hid the bug
because the stale rows were byte-equal to the fresh ones). Fixed with a
per-request `slot.cache = GraphCache::new()` (~1 ms graph rebuild, negligible
next to a prefill) plus `spec.reset_draft()` per request.

## 4. Gates (all green)

- **Conversation identity** (Qwen2.5-14B q4_K_M + 0.5B q4_K_M draft, greedy,
  −n 128, single-turn): code & prose prompts × static `--spec-draft-n 4` and
  `--spec-draft-adaptive` — **4/4 byte-identical** to the sequential run.
  Multi-turn (two user turns via stdin): **2/2 byte-identical** (exercises the
  EOT-insert / cross-turn KV path).
- **Server identity** (three servers side by side: plain, `--spec-draft-n 4`,
  `--spec-draft-adaptive`): 3 rounds × 2 requests (no-stop + stop-string) × 2
  spec configs — **all byte-identical** to plain; `finish_reason: stop` and
  completion-token counts match; streaming (`stream: true`) emits the same
  text chunk sequence.
- **Unit tests**: `spec_rounds_commit_batches_and_stop_at_eog` and
  `spec_round_mid_batch_stop_string_truncates` pin the batch commit, EOG and
  stop-string semantics of the conversation spec loop (mock `spec_round`).
- **Suite**: 187 passed / 0 failed / 3 ignored.

## 5. Scope notes

- The draft never re-prefills the conversation prompt (its context below the
  seed is stale by design, doc 94 §4): proposals are verified, so identity
  holds regardless of draft quality; only the acceptance rate is lower on
  turn 1 of a context switch.
- The conversation/server paths do not print the `[spec] rounds/accepted`
  stats line yet (CLI gen loop only).
- The identity cap is unchanged: adaptive d ≤ 7 (verify nt ≤ 8 stays
  bitwise-identical, doc 95); static d=8 remains available and is *not*
  identity-class.

## 6. Verification recipe

```bash
# conversation identity (single-turn)
minfer --cnv --single-turn --greedy -n 128 <target> "<prompt>" > plain.txt
minfer --cnv --single-turn --greedy -n 128 \
  --spec-draft <0.5B q4_K_M> --spec-draft-adaptive <target> "<prompt>" > spec.txt
diff plain.txt spec.txt   # byte-equal after stripping perf lines

# server identity
minfer serve --port 8971 --n-ctx 4096 <target> &
minfer serve --port 8972 --n-ctx 4096 --spec-draft <0.5B> --spec-draft-n 4 <target> &
# same POST /v1/chat/completions to both ports → identical choices[0].message.content
```
