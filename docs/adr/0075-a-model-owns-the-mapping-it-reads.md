# 0075. A model owns the mapping it reads, and a tensor carries a handle into it

- Status: Superseded by ADR-0076
- Date: 2026-11-05
- Issues: #745
- Amends: the leaked-mmap decision recorded in `docs/inference_e2e_walkthrough/02-gguf-load.md` §2.5 and §3.3

## Context

Since the zero-copy load (2026-08-21, `docs/METAL_OPTIMIZATIONS.md` row 36) every GGUF part is `mmap`'d
and `load_gguf_model` **leaks** the `MmapFile` with `Box::leak`, so weight tensors can hold
`&'static [u8]` slices into it (`gguf.rs`, two sites). `MmapFile::drop` — which does `munmap` — was
therefore unreachable from the load path, and the mapping's file descriptor stayed open for the life
of the process.

That was a deliberate trade: the process ran one model, so a mapping freed by `exit` "is not a leak
in any meaningful sense". The reasoning stops holding the moment a process loads more than one model
— the tooling (convert/quantize round-trips), the test suite (hundreds of loads), and a server that
reloads a model all accumulate address space and descriptors they never release, and macOS's
16 384-file default descriptor limit makes it observable rather than theoretical.

## Decision

**1. The model owns its mapping.** `GgufPart` gains `mmap: Arc<MmapFile>`, and `load_gguf_model`
builds an `Arc` instead of leaking one. `data: &'static [u8]` stays a `'static` view of those pages,
obtained from `MmapFile::as_static_slice`, whose contract is the sentence that replaces the leak:
*the caller keeps an owning handle for as long as the slice is reachable.*

**2. A tensor carries the handle, not just a slice.** `Tensor` gains
`owner: Option<Arc<MmapFile>>` (`Tensor::with_owner`). This is what makes the change sound rather
than merely tidier: the loader returns `Box<dyn ModelDef>` whose weight tensors point into the
mapping, and callers are free to drop the `GgufModel` first. Cloning a tensor copies the `Arc`, so a
clone is a handle in its own right; the last handle unmaps the file and closes the fd.

**3. The Metal zero-copy registration owns a handle too.** `MpsState::register_part` wraps the
mapped pages in an `MTLBuffer` with `newBufferWithBytesNoCopy`, so the GPU reads them directly and
the registration must keep them mapped after the model goes. The registry entry now carries the
`Arc`. That hold is bounded by the registration itself (today: process lifetime, because MPS is torn
down at exit); releasing registrations on model drop is a separate change, and the ADR records the
distinction rather than hiding it.

**4. `MmapFile` is `Send + Sync`, with the reason stated once.** A read-only mapping is immutable
and is unmapped only when the last `Arc` goes, so sharing it between threads is sound; without the
impls `Tensor` loses `Send` and `ModelDef`, which requires `Send + Sync`, stops being implementable.

**5. Two gates, both observable without a device and without `/proc`-only tricks.**
`a_dropped_model_releases_its_mapping_and_fd` asserts the mapping's `alive` flag flips and (on Linux)
its fd disappears from `/proc/self/fd`; `a_loaded_model_keeps_its_mapping_until_it_is_dropped` asserts
a `ModelDef` outliving its `GgufModel` keeps the pages readable, that a cloned weight is a handle of
its own, and that the last handle unmaps. On macOS a live MPS registration legitimately holds the
pages, and the gate says so instead of passing quietly.

## Alternatives considered

- **Keep the leak.** It was the recorded decision, and for `minfer <model> "prompt"` it is still
  harmless. Rejected because the process that reloads models is the one this engine is growing into
  (a server, the tooling, the suite), and because the leak's cost is invisible until it is a
  descriptor-limit failure.
- **Thread a lifetime through `Tensor` (`Tensor<'a>`), and `ModelDef<'a>` behind it.** The correct
  type-system answer, and the reason the leak was chosen. Rejected here because it infects every
  signature that mentions a tensor — the backend traits, the registries, the graph's node metadata —
  for an invariant that is "as long as the model", which the `Arc` already expresses at one point.
- **`TensorData { Borrowed(&'static [u8]), Owned(Vec<u8>), Mapped(Arc<MmapFile>, Range<usize>) }`**
  instead of a parallel `owner` field. Cleaner in the abstract, but `Tensor.data` is read as a
  `&[u8]` in ~200 call sites; a range-based variant would have to move every one of them or keep a
  cached slice anyway — i.e. it re-derives the same self-referential pair, with more surface.
- **A self-referential struct crate (`ouroboros`, `yoke`).** A new dependency for one struct that the
  two-field invariant already covers, in a crate whose whole premise is few dependencies.

## Consequences

- Dropping the last handle on a model unmaps every part and closes its fd. A process that loads and
  drops models (the suite, the tooling, a reloading server) no longer accumulates either.
- `Tensor` grows by one `Option<Arc<…>>` (16 bytes on 64-bit) and every weight constructor that reads
  from a mapping pays one atomic increment. `Tensor::with_owner` is the only new API.
- The `'static` in `data` no longer means "never freed"; it means "the owner keeps it alive". The one
  sentence that used to be a `Box::leak` is now a documented contract on `as_static_slice`, with
  `GgufPart` and `Tensor` as its two guards.
- On macOS the pages of a loaded model stay mapped until MPS is torn down, because a zero-copy
  `MTLBuffer` points at them. Freeing that on model drop needs an unregister path (and a decision
  about what a live engine does when its model disappears) — not this change.
