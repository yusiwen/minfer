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
    // F5 (#58): the split boundary's two phases (§11).
    pub copy_cross:  fn(&mut GraphAllocator, u64, NodeId, Backend) -> Result<bool, String>,
    pub await_cross: fn(&mut GraphAllocator, u64, NodeId, Backend) -> Result<(), String>,
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

`copy_cross` / `await_cross` are F5's addition (§11): the split boundary's
cross-backend staging copy, split into *enqueue* and *wait* so a backend with
device memory can make the transfer asynchronous and name exactly where the
consumer waits. They follow the same rule as the rest of the entry — the
allocator never asks "is this the CPU?", it asks the entry.

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

**Only a *named* backend is checked at stage 2.** The filter therefore carries
two facts per backend — `allowed` (may this run use it?) and `requested` (did the
request name it?) — because the default request means "whatever this build can
use". Preserving that distinction is what keeps the pre-existing
`MINFER_DISABLE_CUDA` / `MINFER_DISABLE_MPS` flags meaning "run on the CPU"
rather than turning them into "refuse to start": with no `--backend` /
`MINFER_BACKENDS`, nothing was named, so stage 2 has nothing to check and the
run proceeds on the CPU exactly as it did before F4. Naming the fenced backend
(`--backend cuda` with `MINFER_DISABLE_CUDA=1`) *is* a refusal, because the user
asked for it.

Stage 2 runs before the model path is resolved on every path that will run a
model (`run`/`serve`/`viz` in `main`, and `bench`/`specverify` in their own
`run`), so an unrelated failure — a missing file, a bad GGUF — cannot preempt it.

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
  are the measurement, and the order/name gates pin the parts a suite would not
  notice.
- **Gates**
  1. `registry::tests::names_resolve_and_unknown_names_are_refused` — the known
     names resolve to the pinned handles, an unknown name and a compiled-out
     name are *distinct* loud errors, and diagnostics keep the pre-F4 spelling
     (pure; CI covers it).
  2. `registry::tests::the_registered_set_and_priority_order_are_pinned` — the
     registered set, the priority **order** and the priority **numbers**, per
     configuration, plus `Ord` and a fresh allocator's deterministic answer.
  3. `registry::tests::the_name_surface_fences_devices_and_keeps_cpu` — the
     fence surface (comma/repeat spellings, `cpu` always admitted, the flag
     winning over the environment, both refusals).
  4. `registry::tests::the_packed_kv_capability_is_the_registrys_answer` and
     `registry::tests::registry_caps_match_the_backend_trait` — the #87 seam is
     the field both C4 gates read, and the registry's capability matrix is the
     trait's (they are one authority).
  5. `alloc::tests::a_fresh_allocator_inherits_the_runs_backend_filter` — the
     fence reaches the assignment pass through the same active filter.
  6. `tests/backend_registry_cli.rs` — the process level: an unknown name exits
     non-zero naming the accepted set, a compiled-out name gives the *other*
     message, a known name passes the gate (the failure moves on to the model),
     `bench` honours the flag, `--help` documents it, and an *unnamed* device
     disabled by the pre-existing flags is still "run on the CPU".
  7. `alloc::tests::cuda_the_backend_fence_moves_assignment_off_a_usable_device`
     — `#[ignore]`d (needs a device): the fence moves assignment off an enabled,
     usable device without tearing it down.
- **Mutation evidence.** (a) making an unknown name fall back to the default
  backend fails gate 1; (b) swapping two priorities fails gate 2; (b2)
  perturbing one priority *value* without reordering also fails gate 2, which is
  why the expected numbers are literals. All three are reverted; the observed
  failure output is recorded in the ticket record.

## 11. F5 — the async staging copy and its synchronization points (#58)

Ticket: [#58](https://github.com/yusiwen/minfer/issues/58) ("[F5] Async
cross-backend copies and events"). The implementation record (measurements,
mutation evidence, honest scope) lives in
`docs/ARCHITECTURE-EXECUTION-PLAN.md` §F5; the scheduler-side contract is also
summarized in `docs/COMPUTE-GRAPH-DESIGN.md` §3.4. This section is the registry
contract those records refer to.

### 11.1 What the hot path is, exactly

`BackendScheduler::execute` partitions a graph into contiguous same-backend
splits (`assign_backends` → `split_graph`). When a value produced by one split is
consumed by a split on **another backend**, the boundary must move it; the
scheduler does that through `GraphAllocator::copy_across` once per entry of
`Split::inputs`. **Those copies are the hot path this ticket is about** — they
run on every forward, on the critical path of every decode step of a partially
offloaded model.

Everything else that reads device memory back to the host is **legitimately
host-visible and out of scope**, and is enumerated here so "zero blocking copies"
is not read as "zero device→host copies anywhere":

| site | why it blocks on purpose |
|---|---|
| `GraphAllocator::copy_to_cpu` on the logits / output path | the run's answer; the caller is about to read it |
| `GraphAllocator::copy_kv_to_cpu` (KV session save, `--session`) | a file write; no overlap to exploit |
| `Copy` / debug dumps, `MINFER_GRAPH_DUMP`, doc dumps | diagnostics |
| `MINFER_TRACE` / viz capture (`CudaBackend::copy_to_host` fallback, `CaptureStaging`) | already batched asynchronously; the per-node fallback is for tensors above the staging ceiling |
| weight / tokenizer loading, `fill_input`, `write_host` | host→device or host-only; no device leg to wait on |

Before F5 a *single* CUDA→host boundary input cost two host stalls: a full
`cudaStreamSynchronize` **plus** a blocking `cudaMemcpy` D2H inside
`CudaBackend::copy_to_host`. The counters below are how that became a number
(`graph::copystats`, per allocator, read by the gates).

### 11.2 The two phases

`copy_across` (phase A) resolves/allocates the destination staging buffer, marks
the entry **pending**, increments `copies`, and calls the source backend's
`copy_cross`. `await_cross` (phase B) calls the source backend's `await_cross`,
clears the pending flag and increments `waits`. One phase-B call per phase-A call
is the **contract**, and `GraphAllocator::cross_input` — the only accessor the
scheduler's consumer path uses — refuses a still-pending entry with a loud `Err`
naming the missing wait, so dropping a wait can never be bought with a silent
read of in-flight data.

Phase A returning `Ok(true)` means "this backend issued the transfer";
`Ok(false)` means "**declined** — use the synchronous host round trip". Declining
is not a silent CPU fallback: the allocator *does* perform the pair, just
synchronously, and the boundary counters record it as a blocking copy.

| backend | phase A (`copy_cross`) | phase B (`await_cross`) |
|---|---|---|
| **cpu** | **synchronous host round trip** — the CPU has no device memory, so there is no transfer to make asynchronous. The device leg of CPU→device is the destination pool's own stream-ordered `write_host` (pinned + `cudaMemcpyAsync`, 7e⑥), which never blocked the host either. | **documented no-op** — nothing was enqueued that needs waiting for; the destination device orders its own fill on its stream. Still counted, so the one-wait-per-copy contract is backend-independent. |
| **cuda** | device→host: `cudaMemcpyAsync` D2H into a pinned slab + `cudaEventRecord`, both stream-ordered after the producing kernels. Any other destination **declines**: a device→device staging copy (unreachable — `copy_across` early-returns when the source and destination backends match), CUDA→Metal on a macOS+CUDA build. | device→host: `cudaEventSynchronize` — **the** host block, and the only one the async path takes for that copy — then the bytes are published into the staging buffer. A device consumer uses `cudaStreamWaitEvent` (no host block). |
| **metal** | **declines** (`Ok(false)`): the async form is a `MTLBlitCommandEncoder` copy plus a completion handler or an `MTLEvent` the consumer waits on — a different mechanism from CUDA's, and this ticket was developed on a Linux box with no Metal device and no macOS toolchain, so it could not be compiled or verified. The port is a filed follow-up; until then a Metal source keeps the pre-F5 synchronous behaviour. | no-op (phase A declined). |

### 11.3 The synchronization points, enumerated

Every execution, in order:

1. **The boundary flush** (`BackendScheduler::execute` step 1) —
   `GraphAllocator::sync_backend(previous)`. Not a copy: it retires the previous
   backend's kernels (and, for CUDA, closes an open graph-capture window and
   clears the MMQ memoization). Unchanged by F5.
2. **Phase A, per staged input** — `copy_across`. A CUDA→host copy enqueues
   `cudaMemcpyAsync` + `cudaEventRecord`; a CUDA→device copy would use
   `cudaMemcpyAsync` D2D; the CPU does its host memcpy; Metal does the
   synchronous read. **No host wait here.**
3. **Phase B, per staged input** — `await_cross`, before any node of the
   consuming split runs:
   - **device→host**: `cudaEventSynchronize` on the event recorded in step 2.
     *Invariant that makes it necessary*: the D2H destination is host memory, and
     the host is about to read it; without the wait the consumer reads bytes the
     DMA may not have written yet.
   - **host→device**: no host wait; the fill is ordered on the consuming
     backend's own stream ahead of the kernels that read it.
     *Invariant*: stream order — the copy and the first consumer share one stream,
     so no host synchronization is needed and none is taken.
   - **device→device** (no backend implements it today): `cudaStreamWaitEvent`
     on the consuming stream would be the mechanism; the host never blocks.
   - The *contract* invariant, independent of backend: one phase-B wait per
     phase-A copy, enforced by `cross_input`'s refusal.
4. **The next boundary's flush**, or the final flush after the last split —
   `sync_backend`, which for CUDA is also where an open capture window closes.

The gated evidence for 2–3 is `copystats::CrossCopyStats` (per allocator:
`copies`, `waits`, `blocking_host_copies`, `async_host_copies`, `event_syncs`,
`stream_waits`), `CudaBackend::blocking_readback_count()` (a device-level count of
blocking `cudaMemcpy` D2H calls) and `cuda::stream_sync_count()` (host stalls).
`MINFER_SYNC_COPIES=1` restores the pre-F5 synchronous path as the bitwise
reference; `copystats::set_sync_for_test` is its programmatic form.

### 11.4 Overlap: what F5 does and does not deliver

F5 delivers the **async substrate + documented waits**: no blocking copy on the
boundary path, one explicit wait per staged input, and a measured reduction in
host stalls (the per-copy stream synchronizations are gone).

**True overlap is not claimed.** The scheduler's split loop is strictly
sequential — a split's nodes run, the next split's copies are enqueued and
immediately waited on — so there is no independent work for a copy to overlap
*with* on a single graph, and a host-side consumer must wait by definition. What
the substrate buys today is that the transfers are enqueued back to back and the
redundant per-copy stream syncs disappear; exploiting them further needs the
scheduler to defer a wait to the consumer's first use (and to know that the
consumer is independent), which is a scheduling change beyond this ticket and is
filed as a follow-up issue rather than claimed here.

