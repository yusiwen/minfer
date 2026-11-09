# 0081. A tensor's bytes are a backing handle plus a range, and the backing may be an owned buffer

- Status: Accepted
- Date: 2026-11-09
- Issues: #797

## Context

[ADR-0076](./0076-a-tensor-s-bytes-are-a-mapping-handle-plus-a-range.md) fixed the shape that made
the engine's bytes safe: a tensor is a **handle plus a `(off, len)` range**, the range is validated
once at construction, and no `'static` view of a mapping exists. Its decision 1 enumerated the
production variant as `Mapped { map: Arc<MmapFile>, off, len }` and kept `Owned(Vec<u8>)` as a
`#[cfg(test)]` scratch case, because every production tensor was a range of a mapped GGUF file.

[#797](https://github.com/yusiwen/minfer/issues/797) makes loading path-or-bytes: a host that already
holds a model — received over the network, embedded, or read from a volume it decrypted itself — must
be able to load it without writing a temp file. That model's bytes are an owned buffer, not a file,
and its weights have to be ranges of it for the same reason a path load's are ranges of the mapping:
a per-tensor copy would double the resident size of every weight and make the load cost proportional
to the model twice.

So the enumerated variant no longer covers production: the handle can be a mapping **or** an owned
buffer, and ADR-0076's literal `Mapped { map: Arc<MmapFile>, … }` no longer exists.

## Decision

1. The backing is an enum: `PartBytes::{Mapped(Arc<MmapFile>), Owned(Arc<Vec<u8>>)}`. It answers
   `as_slice`, `len`, `mapping()` (the mapping handle when there is one) and `alive` (the #745 "still
   mapped" flag; an owned buffer has no unmap and hands back a flag that stays true).
2. `GgufPart` owns a `PartBytes` and exposes `data()` as a borrow tied to the part. There is still no
   `'static` view, and `MmapFile`'s unsafe escape still does not exist — ADR-0076's decision 2 and 3
   stand unchanged.
3. `TensorData` is `Borrowed { bytes: PartBytes, off, len }` (plus the `#[cfg(test)]` scratch case),
   and `Tensor::from_part` takes the backing. The range check moves from "inside the mapping" to
   "inside the backing", which is the same predicate with one more type.
4. A bytes load is **not** a copy: the part owns the `Arc<Vec<u8>>` the host handed over, and every
   tensor is a range of it. A split model is refused by name (the path loader resolves siblings beside
   the file, and a bytes load has no directory), and its refusal names `<memory>` as the source.
5. On macOS an owned part is copied **once** into one `MTLBuffer` (`newBufferWithBytesNoCopy` requires
   page-aligned pages whose lifetime the engine can guarantee; a `Vec`'s allocation is neither), and
   its weights are `(buffer, offset)` into that copy — the same one-buffer-per-part shape the mapped
   path uses. The mapped path keeps its zero-copy alias and its page-fault warm-up unchanged.

## Alternatives considered

- **Copy the buffer into an owned `Vec` per tensor.** Rejected: it doubles the resident weight bytes
  and makes load time linear in the model twice, for a host that already paid for the bytes once.
- **Require a `Path` and document "write a temp file".** Rejected: it is what the issue exists to
  remove, and the temp file is a full disk copy plus a lifetime problem.
- **A `'static` leak of the buffer** (the pre-#745 shape). Rejected: ADR-0076 exists because a safe
  function that can produce a use-after-free is not acceptable in a public API.
- **`newBufferWithBytesNoCopy` for the owned part.** Rejected: it needs page-aligned pages that
  outlive the buffer, and the allocation is neither aligned nor owned by the engine. Copying once per
  part is the honest cost, and it is named.
- **Accept `Vec<Vec<u8>>` for a split set.** Deferred, not rejected: it needs a rule for part order
  and for a missing sibling, and no consumer has asked. The refusal names the reason today.

## Consequences

- One handle type serves both loaders; `Engine::load(path)` and `Engine::load_from_bytes(bytes)` share
  the entire tail of the load (`Engine::from_gguf`), so a turn cannot behave differently per source.
- A zero-copy assertion is possible for both backings: the part's `data().as_ptr()` equals the
  supplied buffer's, and a weight's range lies inside it. `src/gguf/tests.rs` asserts both.
- The CUDA registry's model identity no longer has a path for a bytes load: it gets a content-derived
  fingerprint (length + the first and last 4 KiB), deliberately not a full hash.
- The test-only `Owned(Vec<u8>)` variant of ADR-0076 becomes the production case under a new name, so
  the old `Mapped` variant name is gone from the code; ADR-0076 remains the record of *why* the
  handle-plus-range shape was chosen, and this record extends its enumeration.
- A bytes load of a multi-part model is a named refusal, not a silent partial load.
