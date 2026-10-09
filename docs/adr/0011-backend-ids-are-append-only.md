# 0011. Backend ids are a file-format contract: appended, never renumbered

- Status: Accepted
- Date: 2026-09-24
- Issues: #57

## Context

`Backend` was a compile-time enum (`enum Backend { CPU, Metal, Cuda }`), and every consumer matched
on it: the allocator's twelve dispatch sites, the scheduler, the fusion wiring, the JSON and DOT
exporters, and the KV-session tag table. Adding a backend meant editing all of them, and a backend
could not be requested by name.

At the same time, the id was already *on disk*: the KV session wrote a `u32` backend tag, and
pre-F4 `Ord` was merely enum declaration order. So when F4 replaced the enum with a cheap handle,
the numbering it introduced was not an internal detail — it inherited a file-format obligation.

## Decision

`Backend` is an **opaque index into a fixed id space**, and that space is append-only:

```rust
pub const CPU:   Backend = Backend(0);
pub const METAL: Backend = Backend(1);
pub const CUDA:  Backend = Backend(2);
```

Three properties, all deliberate:

1. **The id space is compile-time and configuration-independent.** `cpu = 0`, `metal = 1`,
   `cuda = 2` on every build, whether or not the backend is compiled in. A CUDA-less build still
   answers `2` for CUDA, so a file written by one build means the same thing to another. The
   constant is annotated as such: `N_BACKENDS` — *"How many backend ids exist. Fixed: the id is a
   file-format contract."*
2. **Ids are appended, never renumbered.** The identity order is what the on-disk tag, the export
   index, `Ord` and the fusion list all read.
3. **The id and the name are different surfaces.** The id is for the file and the exports; the name
   is for humans (`--backend cpu`, `MINFER_BACKENDS`, error messages) via `NAMES = ["cpu", "metal",
   "cuda"]`. A named backend that cannot run is **refused by name**, never dropped — a compiled-out
   backend must not look absent.

The tag binding lives in one place: `tag_of` is `backend.index() as u32`, `backend_of_tag` is
`Backend::from_index(tag)`, and an out-of-range tag is refused loudly — *"unknown backend tag
{tag} (this file was written by a newer build?)"*.

## Alternatives considered

- **Keep the compile-time enum.** Rejected: adding a backend touched the scheduler, the allocator and
  every `match` on `Backend` — the design doc enumerates the twelve dispatch sites — and an enum
  variant has no stable numeric value to persist, so there was nothing to put in a session header or
  a graph export.
- **A dynamic `register(Box<dyn Backend>)` plugin registry** (the roadmap's original
  recommendation). Rejected: "Not a plugin/`dlopen` system: the set of backends is fixed at compile
  time." The backend set is a build-time property of the crate, and `dlopen` would add a failure mode
  with no user.
- **On the numbering itself, the record holds no rejected alternative.** The reason for the order is
  recorded (identity order is read by the on-disk tag, the export index, `Ord` and the fusion list),
  but the tree never weighs "reorder for readability" or "persist names instead of an id" against it.
  A reader should not infer that those were considered and dismissed.

## Consequences

- Adding a backend is additive: append an id, a `NAMES` entry and a `register()` call, and no
  consumer is edited and no existing artifact changes meaning.
- The registry is indexed by handle id and **never iterated as a set**, so nothing in it depends on
  how many backends are compiled in; a named backend that cannot run is refused by name.
- An accidental reordering would silently change where nodes land, so the registered set and the
  priorities (Metal 300, CUDA 200, CPU 100) are pinned by a gate —
  `the_registered_set_and_priority_order_are_pinned` — and two different mutations of it both fail.
- Accepted: the backend set stays fixed at compile time (not runtime-pluggable), and there are two
  id spaces — `models::Device` and `Backend` — bridged by `Device::backend()`.
- The capability surface shrank rather than grew: [#244](https://github.com/yusiwen/minfer/issues/244)
  later deleted three write-only capability fields.

## References

- `docs/BACKEND-REGISTRY-DESIGN.md` §2 ("The handle: `Backend(u16)`") and §9–§10 (the plugin
  boundary; the pinned set and order).
- `docs/KV-CACHE-DESIGN.md` §4 — the session container whose header carries the backend tag.
- `src/graph/registry.rs` — the id constants, `N_BACKENDS`, `NAMES`, and the refusal rule;
  `src/graph/kvsession.rs` — `tag_of` / `backend_of_tag` and the unknown-tag refusal.
- Commits: `a8506fd` (the contract, written before the code), `cdf41b2` (the implementation) and
  `d06f825` (the record), all 2026-09-24.
