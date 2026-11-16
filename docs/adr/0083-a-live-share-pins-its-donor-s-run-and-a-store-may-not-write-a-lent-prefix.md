# 0083. A live share pins its donor's run, and a store may not write a lent prefix

- Status: Accepted
- Date: 2026-11-16
- Issues: #829, #831, #833, #837

## Context

[C8b S2](../ARCHITECTURE-EXECUTION-PLAN.md) made a sequence read another's prefix **in place**:
a `SharedPrefix { cell, rows }` in its span list names the donor's cells, and no byte is
copied. Occupancy, `free_runs` and the written count are derived from those span lists — a
deliberate departure from per-block refcounts, because a refcount would be a derived cache
that can drift.

The share is a **borrow**, and the donor is not told. That produced three defects:

- **[#831]** releasing a donor's run while a sharer read it left the rows covered by a span
  no run held — an "orphan" share. `compaction_plan` packs over **reservations**, so a later
  resize could place another run on the sharer's prefix; `apply_moves` renumbers a sharer only
  for a move that carries the donor's own rows, which no longer exist. The failure was silent.
- **[#833]** a donor slot reused for a prompt that does not reuse the lent rows prefills from
  position 0 and overwrites the sharer's rows. Measured on the checked-in toy GGUF: 192 of 256
  K elements changed.
- **[#829]** `free_cells` was `n_ctx - reserved_cells` while `free_runs` read spans, so a
  released donor's shared rows counted as free (56 cells reported, 22 writable) and the
  server's slot-growth loop budgeted with the wrong number.

[#833] is the write side, [#831] the release/placement side, [#829] the ledger. They are one
contract, so they are one decision.

## Decision

1. **A share always lives inside a reserved run.** `KvCache::release_seq` refuses a run a live
   share points into (0, state unchanged), and `set_cap(seq, 0)` refuses it by name. The pin is
   **derived from the span list** — `shared_from(seq) > 0` — so the no-refcount decision stands:
   nothing counts anything. The release succeeds once the last sharer releases.
2. **The donor's write path is gated at placement and backstopped at the store.** Admission
   places a prompt on a lending slot only when its own reuse point is at or beyond the lent
   rows (`slot_takes_prompt`), and `submit_on` refuses the write outright; when no slot is
   eligible, the refusal names the lending slots. `GraphAllocator::fill_batch_inputs` refuses a
   position below `shared_from(seq)` as well, so a caller that bypasses placement (speculative
   decoding, a new server path, a test) fails loudly instead of overwriting.
3. **One occupancy authority.** `free_cells` is `n_ctx - (cells a live span covers)`, the same
   `occupied()` map `free_runs` and `reserve_seq` read, so `free_cells` equals the sum of the
   free runs and the growth loop's budget is the writable arena. `reserved_cells` keeps its
   reservation meaning (the private runs).

## Alternatives considered

- **Walls in `compaction_plan`.** Treat cells an orphan share covers as immovable, first-fit
  around them. Refused: `compaction_plan` is a pure, infallible function today and would need a
  "cannot place" outcome, `set_cap` an error path for the `cap` it mutates before planning, and
  the walls are permanent holes that stop compaction from packing the runs above them — the
  packing quality C3 exists for. The pin keeps every share inside a moveable run.
- **Keep the observer-only rule** (a released donor's rows stay taken but unreserved). That is
  the defect: a span no run holds is a cell the packer cannot move and may reuse.
- **llama.cpp's `seq_rm` + fresh cells.** A donor resets by dropping its sequence bits and
  re-prefilling into freshly allocated cells, paying a copy of the prefix; the sharer's cells
  survive untouched. This is the right shape for an availability-first policy and is filed as
  [#839], not adopted here: materialising a share needs room in the sharer's run, which
  re-enters the resize/compaction path this decision closes.
- **Per-cell sequence sets and a position→cell map** (the same cache's addressing model). It
  would make the orphan state unrepresentable rather than refused, but it rewrites the
  addressing layer, the session format and a cross-backend topology constant; filed as
  [#840], out of scope here.

## Consequences

- **Availability has a price, and it is explicit.** A lending slot's run stays reserved and the
  slot refuses prompts that do not reuse the lent rows until the share ends, with a named
  `503` ("slot N is lending its first R row(s) ..."). The cells were never writable while
  shared, so no capacity is lost; the ledger now says so.
- **A growth that cannot fit soft-fails** through the existing "cannot grow" path and the slot
  re-prefills, rather than the arena being repartitioned over a live share.
- **`minfer_kv_free_cells` drops by the whole shared prefix** for existing dashboards. That is
  the fix; the metric's help text says what the number is.
- **A pre-#831 session file can still carry an orphan share** — `restore_session` accepts it.
  The span-based `free_cells` and `after_rm`'s share check remain as backstops, and both are
  tested through the restore path.
- The memory saving of sharing is unchanged: one copy of the bytes, read in place by every
  sharer.

## References

- The current contract: [`docs/KV-CACHE-DESIGN.md`](../KV-CACHE-DESIGN.md) §1 (positions and
  cells), §2 (the cell store), §4 (sessions).
- The history: [`ARCHITECTURE-EXECUTION-PLAN.md`](../ARCHITECTURE-EXECUTION-PLAN.md) §5's C8b
  section, its 2026-11-16 note.
- PRs: [#830], [#832], [#834], [#835], [#838]; issues [#829], [#831], [#833], [#837]; the
  deferred alternatives [#839], [#840].

[#829]: https://github.com/yusiwen/minfer/issues/829
[#831]: https://github.com/yusiwen/minfer/issues/831
[#833]: https://github.com/yusiwen/minfer/issues/833
[#837]: https://github.com/yusiwen/minfer/issues/837
[#839]: https://github.com/yusiwen/minfer/issues/839
[#840]: https://github.com/yusiwen/minfer/issues/840
[#830]: https://github.com/yusiwen/minfer/pull/830
[#832]: https://github.com/yusiwen/minfer/pull/832
[#834]: https://github.com/yusiwen/minfer/pull/834
[#835]: https://github.com/yusiwen/minfer/pull/835
[#838]: https://github.com/yusiwen/minfer/pull/838
