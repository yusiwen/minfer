# 0076. A tensor's bytes are a mapping handle plus a range, never a `'static` view of a mapping

- Status: Accepted
- Date: 2026-11-06
- Issues: #745
- Supersedes: [ADR-0075](./0075-a-model-owns-the-mapping-it-reads.md)

## Context

[#751](https://github.com/yusiwen/minfer/pull/751) removed the `Box::leak` and made `GgufPart` own an
`Arc<MmapFile>`, but it kept the *view* as a `'static` slice: `GgufPart.data: &'static [u8]` came from
a safe `MmapFile::as_static_slice`, and `Tensor` carried the handle in a second public field,
`owner: Option<Arc<MmapFile>>`, beside `Cow::Borrowed(&'static [u8])`. That is B1 — the mapping is
owned, but the lifetime promise is still a convention.

The convention is not a type. `GgufPart.data` is a public `'static` slice, so `let d = part.data;
drop(model); d[0]` is safe code with a use-after-free, and `Tensor.owner` is public too, so the
handle beside the slice can be removed by a caller. Inside the current binary every construction
site upholds the contract; #745's goal is a public engine API, where a safe function that can produce
UB is not acceptable.

ADR-0075 rejected the typed view with a cost that does not hold: *"`Tensor.data` is read as `&[u8]`
in ~200 call sites; a range-based variant would have to move every one of them or keep a cached slice
anyway — i.e. it re-derives the same self-referential pair, with more surface."* A
`Deref<Target = [u8]>` newtype keeps every read site's syntax, and a `Mapped { Arc, off, len }`
deref returns `&mapping[off..off + len]` tied to `&self` — no cached slice, no self-reference. The
constructors are three; the read sites do not move.

## Decision

1. `TensorData` replaces `Cow<'static, [u8]>` + `owner`. It is a private enum behind a public struct:
   `Mapped { map: Arc<MmapFile>, off: usize, len: usize }` in production, plus an `Owned(Vec<u8>)`
   scratch case that is `#[cfg(test)]` (the only production tensors are ranges of a GGUF part). It
   implements `Deref<Target = [u8]>` and `AsRef<[u8]>`, so `t.data.len()`, `t.data[..]`,
   `t.data.as_ptr()`, `t.data.to_vec()` and iteration work unchanged.
2. The range is validated once, in `TensorData::mapped`: the offset + length must not overflow and
   must end inside the mapping. `Deref` therefore needs no `unsafe`, and a bad range is a
   construction bug, not a kernel read.
3. `Tensor.data: TensorData`; `Tensor.owner` and `Tensor::with_owner` are deleted, and
   `Tensor::from_data_borrowed_with_strides` is replaced by `Tensor::from_mapped(…, map, off, len)`.
   A clone of a tensor bumps the same `Arc`; there is no parallel handle that can be dropped
   separately.
4. `GgufPart` holds `ctx` and `mmap` only. `GgufPart::data()` returns `&[u8]` borrowed from the part;
   `MmapFile::as_static_slice` is deleted. A `'static` view of a mapping cannot be produced by safe
   code any more.
5. Both loaders pass `(owner, off, len)` instead of a `raw: &'static [u8]` slice, and
   `MpsState::register_part` takes `&[u8]` — it stores the page-aligned base, length and buffer, not
   the slice, so the owner is what keeps the pages mapped.
6. The three #745 gates keep their observables (the `alive` flag, the descriptor's target, pointer
   containment) and now exercise `TensorData::Mapped`; no numerical path changes.

## Alternatives considered

- **Keep ADR-0075's B1.** Rejected: a safe `'static` view of a mapping is unsound as a public API,
  and the public `owner` field is a second way to break the same invariant. Inside the current binary
  the discipline holds; the issue exists to make the API safe for callers outside it.
- **`Tensor<'a>` and `ModelDef<'a>`.** Rejected: it infects every signature that mentions a tensor —
  the backend traits, the registries, the graph's node metadata — for an invariant that is "as long
  as the model", which one `Arc` inside the data already expresses.
- **A cached slice beside the range (self-referential).** Rejected: unnecessary. `Deref` returns a
  borrow tied to `&self`; caching it would only re-create the pair B1 had.
- **`load_from_owned_bytes` only for embedders (the ticket's option C).** Rejected: it is B2 with an
  extra API surface; the mapping is the zero-copy win, and an owned copy would lose it.
- **A droppable per-process model arena (the ticket's option D).** Still needs an owned handle; it
  adds a global with its own lifetime and does not make the view sound by itself.
- **A self-referential-struct crate (`ouroboros`, `yoke`).** Rejected: a new dependency for one
  struct that `Deref` already solves.

## Consequences

- No safe code can obtain a `'static` slice of a mapping: the only production `TensorData` variant
  is `Mapped`, and `GgufPart::data()` borrows from the part. The two soundness holes ADR-0075's
  shape left (`as_static_slice`, public `owner`) are gone.
- The zero-copy path is unchanged in cost: `Mapped` is `(Arc, off, len)`, `deref` slices the same
  pages, Metal's `newBufferWithBytesNoCopy` still points at them, and CUDA still copies H2D at
  registration. `Tensor` grows by the enum's tag and the range instead of a separate `Option<Arc>`
  field; no kernel or graph code changes.
- `TensorData::mapped` panics on an out-of-range view. The loaders pass ranges the parser already
  validated (`GgufTensorInfo::nbytes`, the data-section bound), so the panic is a construction bug,
  not a file error; the parser's named refusal still covers a corrupt file.
- The Metal registration still holds the pages until MPS is torn down, and CUDA's name+size registry
  still assumes one model per process. Neither is fixed here: the first is the #745 S2 follow-up,
  the second a CUDA identity follow-up. The host-side release this ADR governs is real on CPU and
  CUDA, and on macOS only where no Metal registration exists.
- The `'static` literal case is gone from `TensorData` because no caller needed it. A future fixture
  that does can use an owned buffer; adding a `Static` variant back is a new decision, not a default.

## References

- `docs/inference_e2e_walkthrough/02-gguf-load.md` §2.5 and §3.2–§3.4 — the current contract.
- `docs/inference_e2e_walkthrough/03-model-dispatch-weights.md` — what the loader hands to the graph.
- Issue [#745](https://github.com/yusiwen/minfer/issues/745); PR [#751](https://github.com/yusiwen/minfer/pull/751)
  landed ADR-0075 (B1); this decision is the typed view that follows it.
