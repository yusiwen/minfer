# F4 — Backend registry (design record)

Ticket: [#57](https://github.com/yusiwen/minfer/issues/57) ("[F4] Backend
registry instead of the compile-time enum"). This document is the
**design-first** artifact: it is committed before the implementation and states
the registry's shape, the name surface, the ordering rule and the exact
refusals. Gaps and the backlog item live in
`docs/ARCHITECTURE-ROADMAP.md`; the ticket record lives in
`docs/ARCHITECTURE-EXECUTION-PLAN.md`.

## 1. Why the enum was the problem

`Backend` was `enum Backend { CPU, Metal, Cuda }` in `src/graph/mod.rs`. Every
consumer matched on it:

- `GraphAllocator` (`src/graph/alloc.rs`) held one field per backend
  (`cpu: CpuBackend`, `metal: Option<MetalBackend>`, `cuda: Option<CudaBackend>`)
  and dispatched **twelve** operations to it through `match backend { … }`:
  `alloc_buffer`, `alloc_fresh`, `free_buffer`, `pool_len`, `weights_bytes`,
  `write_host`, `write_host_window`, `read_host`/`copy_to_host`, `synchronize`,
  `copy_cells`, the KV element format and the lazy `enable` of a session's pool.
- `BackendScheduler::execute` (`src/graph/scheduler.rs`) matched to pick the
  pool that executes a node, and again to read a node's output back for the
  `MINFER_TRACE`/viz capture path.
- The fusion-pass wiring in `models/qwen2/graph.rs`, `models/qwen3/graph.rs` and
  `graph/json.rs` built a `Vec<&dyn Backend>` **by hand** and mapped a node's
  backend to an index in that vector with a match whose CUDA arm was
  `cuda_idx = backends.iter().position(|b| b.name() == "cuda")` — a lookup that
  existed only because the `match` could not express "whichever device is
  present".

Adding a backend therefore meant editing the allocator, the scheduler, the
fusion wiring, the JSON/DOT exporters, the KV-session tag table and the op
matrix — the enum was a shape every consumer had to be taught. None of that
told a *user* anything either: there was no way to ask for a backend by name,
and no way to find out that a requested one did not exist.

## 2. The handle: `Backend(u16)`

`Backend` stays a cheap, `Copy`, `Hash`, `Eq`, `Ord` value; it stops being an
enum. It is an opaque index into a fixed id space:

```rust
pub struct Backend(u16);

impl Backend {
    pub const CPU:   Backend = Backend(0);
    pub const METAL: Backend = Backend(1);
    pub const CUDA:  Backend = Backend(2);
}
```

The id space is **compile-time and configuration-independent**: `cpu = 0`,
`metal = 1`, `cuda = 2` on every build, whether or not the backend is compiled
in. That is deliberate — the id is already on disk and in exports:

- the KV-session backend tag (`graph/kvsession.rs`, `tag_of` / `backend_of_tag`)
  is a `u32` in a versioned file, so the numbering is a file-format contract;
- `graph/json.rs` / `graph/dot.rs` name and index the backend in exported
  graph documents;
- `Ord` is derived from the id, and the pre-F4 derived `Ord` on the enum was
  declaration order — `CPU < Metal < Cuda`.

`Backend::index()`, `Backend::from_index()` and `Backend::name()` are the only
ways to leave and re-enter the id space; `kvsession`'s tag functions and the
fusion index are now one-line calls to them instead of three-arm matches.

`Debug` is implemented by hand so diagnostics print exactly what the enum
printed (`CPU`, `Metal`, `Cuda`) — no log line or error message changes shape.

### 2.1 Two orders, stated once

The registry has two orders and they are **not** the same; conflating them is
the failure this document exists to prevent.

| order | what it fixes | authority |
|---|---|---|
| **identity** (`Backend::index()`) | the on-disk KV-session tag, the exported graph's backend index, `Ord`, the fusion pass's backend list | `Backend::CPU/METAL/CUDA` |
| **priority** (assignment preference) | which backend `supports_for` offers first | `BackendEntry::priority`, descending |

Before F4 the priority order was implicit in the *statement order* of
`GraphAllocator::supports_for`: Metal, then CUDA, then CPU. It is preserved
exactly, and it is now a number on each entry — Metal 300, CUDA 200, CPU 100 —
so the ordering gate can pin it and a future accidental reordering fails a test
instead of silently changing where a graph's nodes land.

**Determinism rule.** The assignment pass fixes the graph's topology, which is
part of `GraphParams`' reuse identity (standing rule 3). Nothing in the registry
may depend on `HashMap` iteration order:

- the registry table is a fixed-size array indexed by the handle id — not a
  `HashMap`;
- `Registry::by_priority()` sorts by `(Reverse(priority), index)`, so two
  entries can never tie and the result is a total order derived from the
  pinned numbers;
- `BackendFilter` is a `[bool; N_BACKENDS]` indexed by id, never a set that is
  iterated;
- the fusion-pass backend list is built by walking the identity order, so the
  index it hands the pass is `Backend::index()` on every run.

## 3. The registry table

One `BackendEntry` per backend, built once at startup (`registry()`, a
`OnceLock`, is forced by `main` before anything else runs):

```rust
pub struct BackendCaps {
    pub supports_op: fn(&Op, DType) -> bool,
    pub supports_fused: fn(&FusedOp) -> bool,
    pub supports_attn_span: bool,
    pub reads_packed_kv: bool,          // the #87 seam, §8
}

pub struct BackendEntry {
    pub handle: Backend,
    pub name: &'static str,             // "cpu" | "metal" | "cuda"
    pub priority: u16,
    pub caps: BackendCaps,
    pub pool:     fn(&GraphAllocator) -> Option<&dyn BackendTrait>,
    pub pool_mut: fn(&mut GraphAllocator) -> Option<&mut dyn BackendTrait>,
    pub host_read: fn(&GraphAllocator, usize) -> Option<Vec<f32>>,
    pub kv_format: fn() -> KvFormat,
    pub enable:    fn(&mut GraphAllocator) -> bool,
    pub unavailable: fn() -> Option<&'static str>,
}
```

Each backend module owns its entry and its hooks (`cpu_backend::entry()`,
`metal_backend::entry()`, `cuda_backend::entry()`), and `Registry::build()`
calls their `register()` — that call site is the *only* place a backend is
introduced. The capability matrices (`supports_op`, `supports_fused`) and the
`supports_attn_span` / `reads_packed_kv` constants move to module-level items,
and the `impl Backend for X` methods become one-line forwards to them, so the
registry's answer and the trait's answer are **the same code** and cannot
diverge.

`pool` / `pool_mut` are why "sync" and "copy" are not a match any more: the
allocator's dispatch helpers ask the entry for the pool and then call the
`Backend` trait method (`synchronize`, `copy_cells`, `write_host`, …) on it.
The two backends that have a native host-read path different from the trait's
borrowed `read_host` (CUDA's stream-ordered `copy_to_host`) say so in
`host_read`, which is why `copy_to_cpu`, `read_pool` and the CUDA KV debug read
all collapse to one call without a `match`.

`enable` is the lazy "a session names this backend, so bring its pool up"
hook; `unavailable` is the runtime probe (device present, not disabled) used by
the availability refusal in §5.

## 4. What no longer matches on `Backend`

| site | before | after |
|---|---|---|
| `alloc.rs` pool ops (12) | `match backend { Backend::CPU => …, Metal => …, Cuda => … }` | `(entry.pool_mut)(self)` → trait method |
| `alloc.rs` `supports_for` | hardcoded Metal-then-CUDA-then-CPU `if let` chain | `registry().by_priority()` + `entry.caps` |
| `alloc.rs` `kv_load` enable | three-arm `match` with per-`cfg` fallbacks | `entry.enable` + `unavailable` |
| `alloc.rs` `kv_element_format` | `match` with per-`cfg` fallbacks | `entry.kv_format` |
| `alloc.rs` `copy_cells_in_pool` | `match` + per-`cfg` strings | `entry.pool_mut` + registry-aware refusal |
| `scheduler.rs` `execute` | `match split.backend { … }` | `alloc.pool_mut(split.backend)` |
| `scheduler.rs` `read_host_buffer` | `match` | `entry.host_read` |
| `json.rs` / `qwen2,3/graph.rs` fusion wiring | hand-built `Vec` + index `match` | `alloc.fusion_backends()` + `Backend::index()` |
| `json.rs` / `dot.rs` naming | `match` on the enum | `Backend::name()` / `entry` |
| `kvsession.rs` tags | `match` 0/1/2 | `Backend::index()` / `from_index()` |
| `op_matrix.rs` (test matrix) | `match tag { … }` | equality on the handle + registry caps |
| `ensure_kv` packed check, `KvFormat::supports` | `backend != Backend::CPU` / `matches!(device, Device::Cpu)` | `entry.caps.reads_packed_kv` |

Everything the enum used to decide is now either a registry field or a trait
call behind the entry's pool hook.

## 5. The name surface

Accepted names: **`cpu`, `metal`, `cuda`**. Comparison is case-insensitive and
whitespace is trimmed (`  Metal ` is `metal`); the *canonical* spelling is
lower-case and is what every message uses.

Two spellings request a set of backends:

- `--backend <name>` on the CLI (repeatable, and comma-separated values are
  accepted: `--backend cpu,metal`). It is extracted from `argv` before any
  subcommand dispatch, so `serve`, `viz`, `run` and `bench` all honour it.
- `MINFER_BACKENDS=<csv>` in the environment, for callers that cannot pass a
  flag. The flag wins when both are present.

The request is a **fence**: it removes backends from participation. It never
adds one, and it is not an offload policy (`--gpu-layers` / `MINFER_GPU_LAYERS`
stay the authority for how many blocks the device holds). `cpu` is always
allowed, whether or not it is named: it is the universal fallback and a graph
must always be assignable, so `--backend cuda` means "the device if it can take
the node, the CPU otherwise", and `--backend cpu` is the useful spelling —
force the CPU exactly as `MINFER_DISABLE_MPS=1` does, but for every device.

Unset (the default) means "every backend this build has and this machine can
use" — the pre-F4 behaviour, byte for byte. The pre-existing
`MINFER_DISABLE_MPS` and `MINFER_DISABLE_CUDA` fences keep working unchanged,
both presence-checked: they are read by the device layer (`MpsState`,
`CudaState`) and a fenced backend is simply never available, whatever the name
surface says.

Where the fence is enforced:

1. **Device participation** — `Qwen2Graph::device` / `Qwen3Graph::device` (the
   single authority `CParams.gpu` and the server's batching default read) return
   `Cpu` for a fenced device, so the builder never emits a device-only fused
   node (`FusedQKV`, `QkvBiasNorm`, `FusedFFN`, `QkvBiasRopeStore`) that the
   CPU cannot execute.
2. **Assignment** — `GraphAllocator::supports_for` skips a fenced backend, so a
   node is never placed on a pool whose weights were never registered there.

Both read one process-wide filter installed at startup
(`registry::active_filter()`), which is what makes them unable to disagree.

## 6. Refusals — three classes, three messages, always loud

A name is resolved in two stages, because the compile-time answer and the
runtime answer are different questions and a backend that is *compiled out* must
not be silently treated as absent.

**Stage 1 — names, before anything else in `main`.** Purely a function of the
name and the compile-time registry (no device is touched, so it is covered by
CI). Two failures:

```
Error: unknown backend 'gpu2'; known backends are: cpu, metal, cuda
Error: backend 'cuda' is known but not compiled into this build: the CUDA
       backend is compiled only with --features cuda
```

The first is "no such backend name" — always an error, on every build. The
second is "the name exists, this binary does not contain it", and it names which
of the two situations it is. A `metal` name on Linux and a `cuda` name on a
default build both land here, with the reason spelled out (`Metal is compiled
only on macOS (target_os = "macos")`).

**Stage 2 — availability, once the device layer is up.** Still startup (before
the model is loaded), but now the device layer can be asked. A compiled-in
backend that this machine cannot use is a *third*, different message:

```
Error: backend 'cuda' is compiled in but not available on this machine: no CUDA
       device is available, or CUDA is disabled (MINFER_DISABLE_CUDA)
```

A `cuda` name on a `--features cuda` build with `MINFER_DISABLE_CUDA=1`, or with
no device, lands here — not in the "not compiled" bucket, and never in a silent
fallback to the CPU.

Both stages exit non-zero and print nothing else about backends. The reverse
direction is also loud: no code path may *drop* a named backend and keep going.

## 7. Feature gates

| configuration | `metal` entry | `cuda` entry | `metal` name | `cuda` name |
|---|---|---|---|---|
| default Linux | absent | absent | not compiled | not compiled |
| `--features cuda` | absent | present | not compiled | resolves |
| macOS | present | absent | resolves | not compiled |
| macOS + `--features cuda` | present | present | resolves | resolves |

The registry's *registered set* is the compile-time one: `cuda_backend::register`
is `#[cfg(feature = "cuda")]`, `metal_backend::register` is
`#[cfg(target_os = "macos")]`. The *names* are not gated: `Backend::name()` and
the known-name list are unconditional, so an unregistered name still resolves to
a handle and still produces the accurate "not compiled into this build" refusal
instead of "unknown backend". The ordering gate pins the registered set and the
priority order per configuration (all four rows above), so a change to either is
visible in CI on every configuration.

## 8. The seam for a later per-format capability query (#87)

[#87](https://github.com/yusiwen/minfer/issues/87) wants the registry to answer
"can this backend read a packed `q8_0` KV region?" instead of a hardcoded
CPU-only test. The registry carries exactly that field —
`BackendCaps::reads_packed_kv` — and it is **used by this ticket's own code**,
not reserved for the future: `GraphAllocator::ensure_kv`'s packed-region
refusal and `KvFormat::supports` both read it, replacing
`backend != Backend::CPU` and `matches!(device, Device::Cpu)` respectively. That
is why it is a field and not dead abstraction: there is one authority for the
answer, and #87 is the work that flips CUDA's and Metal's value to `true` (and
adds their kernels). No other per-format query is added here.

## 9. What the registry is not

- Not a plugin/`dlopen` system: the set of backends is fixed at compile time.
  The registry makes the *set* data instead of control flow; it does not make it
  extensible at runtime.
- Not a device-selection policy: `--gpu`, `--gpu-layers` and `MINFER_GPU_LAYERS`
  keep their meanings.
- Not a weight registry: `GraphAllocator::register_weight` (CPU) and
  `CudaState::register_weight` (CUDA) are unchanged; the "all weights
  registered" gate is per architecture and stays where it is.
- Not a place to move `Device`: `models::Device` stays the coarse "the device
  participates" fact that the server's batching default reads, and gains a
  `Device::backend()` mapping so the two id spaces have one bridge.

## 10. Acceptance and the gates

- **Behaviour preservation.** The identity and priority orders, the
  `supports_op` / `supports_fused` / `supports_attn_span` answers, the
  sync/copy arms and the memory accounting are unchanged; the existing suites
  are the measurement, and the two order gates pin the parts a suite would not
  notice.
- **Gates**
  1. `registry::tests::names_resolve_and_unknown_names_are_refused` — the known
     names resolve to the pinned handles and an unknown name is a distinct,
     loud error (pure; CI covers it).
  2. `registry::tests::the_registered_set_and_priority_order_are_pinned` — the
     registered set and the priority order, per configuration.
  3. `tests/backend_registry_cli.rs` — the process level: an unknown name exits
     non-zero naming the accepted set, a compiled-out name gives the *other*
     message, and a known name passes the gate (proved by the failure moving on
     to the model).
- **Mutation evidence.** (a) making an unknown name fall back to the default
  backend fails gate 1; (b) swapping two priorities fails gate 2. Both are
  reverted; the observed failure output is recorded in the ticket record.
