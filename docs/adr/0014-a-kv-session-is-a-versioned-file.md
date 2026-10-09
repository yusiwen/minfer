# 0014. A KV session is a versioned, checksummed file — never a memory dump

- Status: Accepted
- Date: 2026-09-22
- Issues: #43, #130

## Context

A session's KV rows, its ownership table and its run table lived only in memory, so every restart
re-prefilled the whole context. The obvious shape for a snapshot is a dump of the arena — and that
is exactly the shape this decision rejects, because the arena is not self-describing: a byte range
means nothing without the shape, the element type, the owner table and the span lists that produced
it.

Two later increments sharpened the contract. S2 (2026-09-24) recorded that *"the container described
the KV rows; the rows belong to a host state, and a restore that brought one back without the other
would be a different session"*. S3 (2026-09-25) fixed a real self-refusal: an f16 region wrote
`flags == 0` and read back as **f32**, so `kv_load`'s `header.format != live` check rejected the file
its own writer had just produced.

## Decision

`src/graph/kvsession.rs` writes a **self-describing, versioned, checksummed container**, and the whole
file is verified before a single byte is applied.

- An 8-byte magic (`MINFERKV`) and `VERSION = 2`.
- A flags word carrying the **element type**: `FLAG_PACKED` (Q8_0), `FLAG_F16`; `KNOWN_FLAGS` is their
  union, the two type bits are mutually exclusive, and an unknown bit is refused loudly.
  `flags_of` / `format_of_flags` are each other's exact inverse.
- An FNV-1a checksum over the payload — with truncation also caught *by construction, not only by the
  checksum*: every read is exact against a length the header fixes.
- `kvsession::verify(path)` makes a full pass over the header, every layer's declared length, the
  bookkeeping, the checksum and end-of-file **before** `ensure_kv` creates a single region, and
  `KvCache::restore_session` validates arena capacity, owner-table length, every reservation and span,
  and every live sequence's span list, before applying.
- The header must describe the run that loads it (backend, `n_ctx`, `n_embd`, element type). A mixed
  offload plan has KV on two backends, so `kv_load` **refuses a session** — one arena per file.

Version 2 adds an opaque, length-prefixed **host-state blob** inside the checksum: the KV rows belong
to a host state, so the container carries both or neither, and a version-1 file is refused loudly.

## Alternatives considered

- **A format *field* instead of a flag bit** (S3, when f16 had to be distinguished). Rejected
  explicitly: a field would have had to either renumber `FLAG_PACKED` — changing the meaning of every
  Q8_0 file already on disk — or move the type elsewhere in the header, shifting the byte layout a
  version-2 reader is already parsing. Both are the silent misread the container exists to prevent.
  The corollary is recorded too: **no version bump**, because a bump is for a layout change and this
  is an additive flag whose older-reader behaviour is a *loud refusal*.
- **Resume the in-flight server request** (`--slots-file`'s tempting extension). Rejected: a `Run`
  holds the response channel to a client a restart has already disconnected, plus its RNG and the
  row's logits; none of it survives a process boundary, and "resuming" a stream nobody is reading
  would be a fiction. The snapshot therefore carries the **context**, not the in-flight request.
- **Dump the arena raw, or verify after applying.** **No rejected alternative is recorded.** The
  written record describes the chosen contract; it never weighs a raw dump or an
  apply-then-verify ordering against it. A reader should not infer that those were considered.

## Consequences

- A restart resumes with **0** tokens prefilled and a **bitwise-identical** continuation: the
  real-model gate asserts `max |Δlogit| = 0`. Measured on the 0.5B: 24 layers / 256 cells / 5 written
  / 6 316 748 bytes; the CLI resumes in 0.47 s against 2.36 s re-seeded (**5.0×**).
- The version discipline means an older file is refused by name rather than misread, and a foreign or
  corrupted file is a no-op rather than a partial load.
- `--slots-file` is opt-in because it rewrites on completion — one arena write per completed request
  (~12 MiB for the 0.5B/512-row fixture, printed at startup). The serial path has no shared arena to
  snapshot and says so loudly.
- Honest gap from the record: at S2 the CUDA/Metal session companion was **not verified here** (the
  container is backend-tagged, the CPU path was the measured one).

## References

- `docs/KV-CACHE-DESIGN.md` §4 — the current contract.
- `ARCHITECTURE-EXECUTION-PLAN.md` — the C5 record and its S2/S3 sub-records.
- Commits: `7fd4765` (2026-09-22, the container); the S2 host-state blob and S3 f16 flag are dated
  2026-09-24/25 in the C5 record.
