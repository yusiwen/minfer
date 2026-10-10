# 0033. The CUDA pool recycles exact byte lengths, never frees, and reports OOM as an error

- Status: Accepted
- Date: 2026-10-10
- Issues: #480

## Context

`CudaBackend`'s pool is the device-memory home for two kinds of buffer that behave differently: graph
nodes' intermediate buffers, which are short-lived and reused every step, and the **persistent KV
regions**, which must survive a graph rebuild. The same pool also serves the scheduler's split-boundary
staging. `docs/CUDA-BACKEND-DESIGN.md` §4.1 states four rules for it and says outright that they "carry
correctness weight" — which is a decision, and the corpus had no record of it.

## Decision

**The pool recycles exact byte lengths, only ever frees on `Drop`, offers `alloc_fresh` for staging that
must not be recycled, and treats OOM as an error rather than a panic.**

- **Exact byte-length reuse only.** `alloc_buffer` scans `free` for an id whose `pool[id].bytes == size * 4`.
- **`free_buffer` never frees.** It returns an id to the free list; only `Drop` returns memory to the
  driver.
- **`alloc_fresh` exists for split-boundary staging**, bypassing the free list entirely.
- **OOM is not a panic.** `cuda_malloc` logs and returns null; the null buffer fails cleanly at execute
  time through `ptr_of`. `pool_gen` bumps on every allocation (so a captured exec that embedded a scratch
  pointer is re-captured rather than replayed against a freed one).

## Alternatives considered

- **Panic on OOM.** Explicitly forbidden in the record: a panic "would poison the shared scratch maps and
  the device-entry token (the legacy path) for every other user" — the pool's memory is process-wide, so
  one engine's allocation failure is not one engine's problem to abort.
- **Free in `free_buffer`.** Rejected by the persistent-KV requirement: a region must survive a graph
  rebuild, and the pool keeps device memory for the next graph rather than returning it per step.
- **Recycle a free-list id for split-boundary staging.** Rejected: ids in the free list are still
  referenced by `node_to_buf` and are physically live during the execute that follows, so a staging write
  into one would clobber a node's buffer.
- **Reuse by size class rather than exact byte length** (the ladder `allocplan` uses for the *planner*).
  **No rejected alternative is recorded for this rule** — §4.1 states exact-byte reuse, and the record
  neither weighs nor mentions a class-based pool. A reader should not infer that one was considered; if
  the rule is ever revisited, that is new ground.

## Consequences

- **Memory is held, not returned.** The pool's high-water mark is accepted and documented as a risk
  (§6 row 5: "same policy as CPU/Metal; `Drop` frees; documented"), which is the cost of not re-allocating
  a KV region per rebuild.
- An allocation failure surfaces **where it can be reported** — at execute time with a node in hand —
  rather than as a process abort inside an allocator shared with other engines.
- `pool_gen` is load-bearing beyond bookkeeping: capture embeds scratch pointers, so growth must invalidate
  a captured exec rather than let it replay against a stale address.

## References

- `docs/CUDA-BACKEND-DESIGN.md` §4.1 (the pool table and the four rules) and §6 row 5 (the accepted
  high-water mark).
- `docs/CUDA-BACKEND-DESIGN.md` §4.5 — §4.1's rules as the allocator/scheduler integration sees them;
  [#480](https://github.com/yusiwen/minfer/issues/480) — this record.
