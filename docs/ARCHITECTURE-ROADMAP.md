# minfer Architecture Roadmap

**Status:** analysis + prioritized backlog (no code changed).
**Baseline:** minfer `HEAD = 293eb19` (2026-09-15, working tree clean).
**Method:** direct reading of `src/` (44,105 LOC) and `docs/` (~15.6k LOC). Every
claim about minfer carries a `file:line` anchor; claims taken from a document
rather than code are marked *"per docs"*.

**Scope.** Architecture-level work on the *system* layers: IR, scheduler,
allocator, KV/sequence state, batching, backend abstraction, and model/quant
coverage. Kernel micro-optimization is out of scope — `CUDA_OPTIMIZATION.md`
and `METAL_OPTIMIZATIONS.md` cover it, and both campaigns are formally
converged. Model-family selection is a separate document:
`docs/MODEL-SUPPORT-ROADMAP.md`.

---

## 0. Verdict

minfer's kernel work is complete and competitive for the architectures it
supports: eight quantized weight types on CPU, Metal and CUDA; int8 MMQ
(quantized matrix-multiply on tensor cores) prefill and dp4a MMVQ
(quantized matrix-vector multiply) decode on CUDA; simdgroup GEMM
(general matrix-multiply) and flash attention on Metal; speculative decoding
with adaptive draft depth. On the measured CUDA target, 7B Q4_K_M reaches
~3581 tok/s prefill and ~51.2 tok/s decode; on Metal, 7B decode is at parity and
prefill reaches ~0.82–0.85× of the achievable ceiling, with the residual
measured as not source-addressable.

The remaining work is at the **system layer**, and three items dominate it:

1. **No multi-sequence batching.** The IR, the attention kernels, the KV
   (Key/Value) allocator and the server are all single-sequence. E1/E1b made the
   *attention* side sequence-aware and E2 landed sequence-addressable batches
   and a batching server worker; the CPU payoff measured negative while the GPU
   payoff was 1.9x, so the default now follows the device (**E6**: batches iff the
   model runs on CUDA, off on CPU/Metal).
2. **KV cache is a fixed per-layer buffer**, not a sequence-addressable cell
   store: no sequence ids, no eviction/context shift, no defragmentation, no
   state save/restore, no quantized KV. (Phase C has since closed every one of
   those: C1/C2 the cell store and physical removal/shift, C3 the compaction — a
   pure planner, the counters, `Backend::copy_cells` on CPU and CUDA, the K re-rope
   a move needs while `positions` are cells, and a model-level continuation gate —
   C4 the packed Q8_0 cache (CPU and CUDA) and C5 the session container. What is
   still open is driving the compaction from the server's serving model, which reserves every
   slot once and packed, the remaining C4 tuning ([#87](https://github.com/yusiwen/minfer/issues/87):
   the packed fused epilogue, a dp4a packed dot and the packed FA prefill), and Metal's
   half (G5, [#44](https://github.com/yusiwen/minfer/issues/44)). See §2.4.)
3. ~~**The server has no persistent context.** Every request builds a fresh
   `GraphCache` (KV regions + device pool re-allocated, CUDA Graph capture
   re-warmed) and re-prefills the whole prompt.~~ **Fixed in B2/B3** for the
   prefix-matched case: a slot keeps its cache and reuses the rows its prompt
   already covers (measured ≈11× TTFT on a second turn).

Everything else — IR expressiveness (views, multi-output), memory placement
policy (VRAM budget, layer offload), backend pluggability, additional model
families, quantizer tooling — is real but secondary,
and mostly *enabled* by fixing (1) and (2) first.

### Ranked backlog (full list with numbering in §3)

| §3 item(s) | Item | Layer | Class | Effort |
|---|---|---|---|---|
| 1–3 | Multi-sequence batch + continuous batching | L4+L5+L2 | XL | 3–6 w |
| 1 | KV cache redesign (cells, seq ids, prefix reuse, shift/defrag) | L4 | XL | 2–5 w |
| 4 | Persistent server context (no per-request rebuild/realloc) | L5+L3 | M | 3–5 d |
| 7 | IR expressiveness: strided views/aliasing + multi-output nodes | L1 | L | 1–2 w |
| 8–9 | Memory placement policy: VRAM budget, layer offload, size-class allocator — **E4 complete (accounting/gate, size-class pools, reserve/assign + the multi-graph cache) and E5 complete (explicit per-block offload and the `auto` fit); only a Metal free-bytes query (so `auto` works there) and per-block tuning remain** | L3+L6 | L | 1–2 w |
| 12 | Backend registry (drop the hard-coded 3-way enum/match) — **DONE (F4, 2026-09-24)**: the enum is a `Copy`/`Hash`/`Ord` handle over a fixed id space and the name-keyed table lives in `graph/registry.rs`; the nine match sites are gone. The set is still closed at compile time — this makes it data, not pluggable at runtime | L6 | M | 4–7 d |
| 11 | CPU: AVX2/AVX-512 for the K-quant dots + weight repacking | L7 | L | 1–2 w |
| 10 | Chunked prefill (`n_batch` actually used) — **E3, landed 2026-09-22** | L2+L5 | M | 3–5 d |
| 15 | Grammar / JSON-schema constrained decoding — **DONE (F2, 2026-09-24)**; the mask is host-side and per state (5.4 ms on a 151k vocabulary) | L7 | M | 4–6 d |
| 23 | Op × dtype × backend correctness matrix in CI | L8 | M | 3–5 d |

---

## 1. Current state

### 1.1 Scale

| Metric | Value |
|---|---|
| `src/` Rust | 44,105 LOC across 48 files |
| Largest files | `graph/cuda_backend.rs` 6,664 · `src/cuda.rs` 6,340 · `metal.rs` 3,944 · `models/qwen2/graph.rs` 2,253 · `gguf.rs` 2,096 |
| Compute-graph core (`src/graph/`) | 13,052 LOC; 3,391 LOC excluding the three backends |
| CUDA kernels | `src/cuda_kernels.cu` (single TU, compiled by `build.rs` only under `--features cuda`) |
| Metal shaders | `src/metal.metal` → `minfer.metallib` at build time, source-compile fallback |
| Tests | 5 integration files (`tests/`, ~3.1k LOC, four of them Metal-only) + inline `#[cfg(test)]` |
| Docs | ~182 Markdown files; 41 in `docs/`, 106 numbered CUDA campaign records |

### 1.2 Landed surface

Compute graph (build → assign → fuse → alloc → execute) with params-only reuse
(`graph/cache.rs:47`, `graph/params.rs:51`); per-op backend assignment
(`graph/scheduler.rs:60`); liveness allocator with persistent per-layer KV
regions (`graph/alloc.rs:164`, `:384`); eight quantized weight types
(`Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q4_K/Q5_K/Q6_K`) on CPU + Metal + CUDA; two model
families (`qwen2`, `qwen3` dense); GGUF v3 with split parts and Metal zero-copy
mmap; self-contained BPE tokenizer; chat templates via minijinja;
greedy/penalty/top-k/top-p/temperature sampling; speculative decoding with
adaptive depth; multi-turn conversation CLI; OpenAI-compatible server
(chat completions, streaming, multi-slot); a runtime graph introspection stack
(`MINFER_TRACE` per-node capture, live SSE visualizer, pipeline view, DOT/JSON
export).

### 1.3 Documentation hygiene

The index-level documents (`README.md`, `ARCHITECTURE.md`,
`CPU_OPTIMIZATIONS.md`, `QWEN3-SUPPORT-PLAN.md`, `OPENAI-CHAT-API-PLAN.md`,
`MODEL-SUPPORT-ROADMAP.md`) are the entry points `AGENTS.md` routes readers
through, so a stale statement there costs more than one buried in a campaign
record.

A pass on 2026-09-15 fixed these stale statements:

- `README.md` and `ARCHITECTURE.md` — the removed `BiasRope` fusion in the
  two architecture diagrams, plus the `fusion.rs` row in the module table
  (the pass produces only `SwiGLU`; the decode fusions are built by the model
  code, not by `FusionPass`).
- `ARCHITECTURE.md` — the "`n_out` tail-row optimization is not yet ported"
  claim (it landed as the G3 work), and the `RopeStyle` "already covers
  Qwen2 and Llama" claim (only the non-interleaved form is wired up).
- `CPU_OPTIMIZATIONS.md` — the pre-compute-graph snapshot is now banner-marked
  as historical, with the still-open AVX2 gap pointed at §2.7.
- `QWEN3-SUPPORT-PLAN.md` — the self-contradictory status line.
- `OPENAI-CHAT-API-PLAN.md` — the "KV is f32" note (f16 is auto-selected for
  7B-class models).
- `MODEL-SUPPORT-ROADMAP.md` — the misleading status line.

Keep the invariant: when a change lands, update the index rows in the same
commit. No outstanding doc-truth work item remains.

---

## 2. Layer-by-layer analysis

The layer names follow the vocabulary of `docs/GLOSSARY.md`, which classifies
the campaign's terms into seven layers; §2.8 adds one cross-cutting layer for
ops, safety, testing and observability.

Legend for **Gap**: 🔴 structural (blocks a class of use cases) · 🟠 material
(measurable capability or performance loss) · 🟡 hygiene.

### 2.1 L1 — Graph IR and build

**Today.** `ComputeGraph` is a flat, topologically ordered vector of `CNode`
(`graph/mod.rs:83-115`). Each node has exactly one output, a static
`[usize; 4]` shape, and an `Op` carrying its full payload. `Op::View { offset,
shape }`, `Reshape`, `Permute` exist (`ops.rs:105-116`) but both backends
execute them as **identity copies** (`cpu_backend.rs:384-388`,
`cuda_backend.rs:433-438`, `metal_backend.rs` likewise) — the allocator has no
aliasing for views, so a view costs a full tensor copy. The builder appends
sources before consumers, so node-id order *is* the execution order
(`scheduler.rs:214`).

**Gap.** 🔴 Strided, zero-copy views and multi-output nodes were missing — **closed by D1 (2026-09-19)**: views are zero-copy with allocator-known aliasing, and `GraphBuilder::split_parts` exposes a node's single output as independent parts, so what remains under this header is model work (item 17), not IR work. Without
them the IR cannot express slicing, concatenation, or a copy between
overlapping regions as compositions of primitive ops — which is why minfer
carries four hand-written decode-specific fused ops instead: `FusedQKV`,
`QkvBiasRopeStore`, `FusedFFN`, `FusedQkvNorm` (`ops.rs:125-154`). Each one
costs changes in *five* places: the builder constructor (`builder.rs:193-281`),
the allocator's special cases (`alloc.rs:231-272`), the scheduler's `kv_pair`
resolution (`scheduler.rs:261-271`), every backend's `supports_op` +
`execute_node`, and both models' `build_graph`. `BatchMatMul` is deferred for
the same reason (`COMPUTE-GRAPH-DESIGN.md §5.4`).

**Consequence for the model work.** `MODEL-SUPPORT-ROADMAP.md` ranks new model
families by how much of the existing graph they reuse. That ranking is accurate
for parameter-isomorphic architectures (Llama/Mistral) but not for anything
needing new structure: MoE (Mixture-of-Experts) needs `mul_mat_id` (3-D expert
indexing), MLA (Multi-head Latent Attention) needs a different KV layout and a
view-based latent split, and sliding-window attention needs a mask parameter
threaded into attention. Each of those currently pays the five-site cost.

**Recommendation.** Add strided views with allocator-known aliasing, and let a
node carry a small output list. Fusions can then be produced by the fusion pass
from compositions rather than hand-written as IR variants, and the four decode
fusions become compiler output instead of bespoke ops.

---

### 2.2 L2 — Scheduler and execution

**Today.** `assign_backends` walks nodes in build order and assigns the
highest-priority backend whose `supports_op` returns true
(`scheduler.rs:60-69`). `split_graph` partitions into *contiguous* same-backend
runs derived from that positional assignment (`scheduler.rs:73-120`); split
inputs/outputs are the crossing edges. Execution then walks splits, calling
`sync_backend(prev)` and `copy_across` for each cross edge
(`scheduler.rs:176-189`), and finally `sync_backend`.

**Gap.** 🟠 Two distinct issues.

1. **No cost model.** Splits are positional. A single unsupported op in the
   middle of an otherwise GPU graph produces three splits with two host round
   trips; the assignment never considers the cost of the resulting movement, or
   whether a neighbouring op could be moved so the op becomes supportable. In
   practice this is masked today because the supported models build a *single*
   GPU split on both GPU backends (`CUDA-BACKEND-DESIGN.md:313-315`), but it is
   exactly what breaks the first time an op is unsupported — e.g. interleaved
   RoPE on CUDA (`cuda_backend.rs:1324`), or `FusedQkvNorm`, which CUDA does
   not advertise (`:1289-1321`) while Metal does (`metal_backend.rs:276`).
2. **Synchronous, host-mediated cross-backend movement.** — **partly closed by
   F5** ([#58](https://github.com/yusiwen/minfer/issues/58), 2026-09-24): the
   boundary is now two registered phases. Phase A (`copy_across` →
   `BackendEntry::copy_cross`) enqueues the transfer — for a CUDA source a
   device→host `cudaMemcpyAsync` into a pinned slab plus a `cudaEventRecord`,
   where the pre-F5 code did `copy_to_cpu` (a full stream sync + a blocking
   `cudaMemcpy`) — and phase B (`await_cross`) waits on the recorded event at the
   documented synchronization point, once per staged input. The CPU's hooks are a
   synchronous host round trip and a no-op (it has no device memory); **Metal
   declines and keeps the synchronous path** until its blit/event port is written
   and verified on a Mac. Counters (`graph/copystats.rs`), the per-backend table,
   the enumeration of the synchronization points and the measured before/after are
   in `docs/BACKEND-REGISTRY-DESIGN.md` §11 and the plan's F5 record.
   **What remains:** true *overlap* — the split loop is still strictly sequential
   (enqueue, then immediately wait), so there is no independent work for a copy to
   overlap with; exploiting the substrate needs the scheduler to defer a wait to
   the consumer's first use, and the Metal port. Both are filed follow-ups (see
   the plan's F5 record), and multi-device execution still needs both.

**Also:** ~~the cross-boundary staging map is keyed by node id alone
(`alloc.rs:34`), so a node consumed by two different foreign backends can only
have one staging buffer; the scheduler compensates by filtering on backend
(`scheduler.rs:252-255`), which is correct only while at most two backends are
enabled.~~ **Fixed in A5**: the map is keyed by `(node, destination backend)`,
one node can feed two foreign consumers, and the consumer-side filter is gone.

**Recommendation.** Add an assignment pass that (a) propagates support backwards
from unsupported ops and (b) scores a candidate assignment by crossing count; and
replace the host round trip with an async copy plus a recorded event when the
source and destination are different devices (**F5 landed the CUDA half of this;
the Metal half and the actual consumer-side waiting that would let a copy overlap
independent work are the remaining follow-ups**). Neither is urgent *today*; both
are prerequisites for multi-device execution and for any heterogeneous split.

---

### 2.3 L3 — Allocator and memory

**Today.** `GraphAllocator::alloc_graph` (`alloc.rs:164`) rebuilds the whole
node→buffer mapping on every graph rebuild: it frees every previously live
buffer back to the backend pool (`:165-177`), recomputes `last_use` over build
order, and re-allocates. Buffer pools are per backend, not unified
(`alloc.rs:314-334`), and allocation is by **exact element count** — both the
Metal (`metal_backend.rs:293-303`) and CUDA (`cuda_backend.rs:1333-1353`) pools
scan a free list for an exact byte-length match and otherwise allocate fresh.
`free_buffer` never returns memory to the device (`cuda_backend.rs:1355-1362`).

**Gap.** 🟠 Three consequences.

1. **Per-rebuild teardown.** Non-persistent buffers are freed and re-derived on
   every topology change. For a decode loop with a stable shape this happens
   once; for anything with a varying `n_tokens` (speculative decoding with
   adaptive depth, server requests of differing prompt lengths, future
   continuous batching) it happens every time, together with a full device
   re-allocation for any size not seen before.
2. **No size classes / no rounding.** Because matching is exact, a workload
   touching *k* distinct activation shapes ends up with *k* sets of live
   buffers. The pool is a per-`GraphCache` high-water mark that never shrinks
   (documented as accepted debt, `CUDA-BACKEND-DESIGN.md:495`) — but that was
   reasoned about for a fixed-shape CLI run, not for a server whose prompt
   lengths vary per request. **E4 S2 (2026-09-23)** rounds every pooled
   activation to its class, so shapes inside one class share a buffer across
   rebuilds; **E4 S3** then split reservation from assignment (a `slots` table
   per backend and class: a released buffer is re-assigned rather than returned
   to the pool, so a rebuild re-maps and CUDA's `pool_gen` — the capture
   generation — stops moving) and gave `GraphCache` one graph per `GraphParams`,
   so a switch is a re-map, not a build (§14 row 3, closed by E4 S3).
3. **No VRAM budget or feasibility check.** `cudaMemGetInfo` is queried once at
   init and only printed (`cuda.rs:1525-1546`); the sole consumer of free-memory
   information today is a valve guarding the optional f16 weight cache
   (`cuda.rs:3931-3936`) — the activation/KV allocator has no accounting at all.
   Out-of-memory surfaces as a null pointer that fails at execute time
   (`cuda_backend.rs:1345-1348`). There is no "this graph will not fit, offload
   the last *n* layers" fallback because there is no layer-offload concept at
   all (§2.6).

**Recommendation.** Split allocation into a *reserve* phase (size the graph for
the worst-case shape the session will use: `n_ctx`, `n_batch`) and an *assign*
phase that re-maps nodes into reserved regions without touching the device.
Round pool sizes to a size-class ladder (e.g. 16/64/256 KiB steps) so distinct
shapes share. Add real memory accounting (`bytes_reserved`, `bytes_live`,
`bytes_device_free`) and make the prefill path consult it before allocating,
with CPU execution as an explicit fallback *decided at build time* (consistent
with the "no silent fallback" rule — the fallback would be visible in the
backend assignment).

---

### 2.4 L4 — KV cache and sequence state 🔴

**Today.** Each layer owns two persistent contiguous regions, K and V, created
on first use by `ensure_kv(layer, backend, size)` (`alloc.rs:384-392`), sized
`n_kv_embd × n_ctx` f32 (or f16 when the type flag is set). They live in the
allocator inside `GraphCache` and survive graph rebuilds (`graph/cache.rs:69`).
Positions are *data*, injected per step through the `positions` input node, so
the topology never depends on `n_past` (`ops.rs:94-99`,
`COMPUTE-GRAPH-DESIGN.md §1.4`).

That design is a genuine strength — it is the precondition for graph reuse — but
it stops at the single-sequence append-only case. What is missing:

| Capability | Status |
|---|---|
| Sequence ids / per-sequence views of one cache | ✔ **C1** per-cell `owner` (`SeqId`); **E1/E2** made several views of one cache real (reservations + explicit `attn_span`, device-aware batching default via E6); **C7 increment 1** made the partition elastic. One cell still belongs to exactly one sequence — **C8** ([#41](https://github.com/yusiwen/minfer/issues/41), **closed 2026-09-22**) turns `owner` into a set/refcount so a prefix can be shared, and C8b's span list + `kv_map` ([#41](https://github.com/yusiwen/minfer/issues/41)) delivers it on CPU and CUDA |
| Partial removal / keep / copy between sequences | ◐ **C2**: physical removal of a row range; **compaction inside one arena is C3 (done)**; *sharing* one cell range across sequences no longer needs `owner[cell]` to become a set — **C8b S2 landed 2026-09-21** derives occupancy from per-sequence span lists and **S3 landed 2026-09-22** adds the copy-on-write store rule, both gated on byte-equality; CUDA's gather (S4) is next |
| Defragmentation | ✔ **C3 (2026-09-19)**: pure planner + `KvArenaStats` counters + `Backend::copy_cells` (CPU `copy_within`; CUDA `kv_move_rows`, one block walking rows ascending with a barrier, because overlapping device-to-device copies are undefined; Metal refuses it, G5) + host K re-rope while `positions` are cells + a mid-session compaction gate. The re-rope is **gone** (C6 landed): compaction moves rows verbatim and is bit-identical. **C7** is the scheduled consumer — the server's dynamic-run trigger uses the planner to make a slot's reservation elastic |
| Sliding-window eviction | ◐ **C2**: physical shift + re-rope (exact mechanism; the retained rows keep the context they were written in, see below) |
| Recurrent / hybrid memory (state-space models) | ✗ |
| Context shift (keep KV, shift positions) | ✔ **C2**: `kv_rm`/`kv_shift` plus the conversation's overflow shift — 185 → 14 prefilled tokens per overflowing turn on the 0.5B probe; `MINFER_NO_CONTEXT_SHIFT=1` restores the exact re-render |
| State save/restore (session persistence) | ✔ **C5 (2026-09-22)**: a versioned, checksummed container (`graph/kvsession.rs`) holds the shape, the backend, the KV element type, one K/V blob per layer and the arena's bookkeeping (owner table + run table + span lists); `GraphAllocator::kv_save`/`kv_load` stream it through the backends' host I/O. A load **verifies the whole file before it applies anything**, so a truncated/corrupted/foreign file is a no-op, and the real-model gate resumes a session that continues **bitwise** (max \|Δlogit\| = 0, CPU and CUDA). The element type rides in the header flags (`FLAG_PACKED` Q8_0 / `FLAG_F16` f16 / `0` f32; the two bits are mutually exclusive and an unknown bit is refused), so **C5 S3 ([#130](https://github.com/yusiwen/minfer/issues/130), 2026-09-25)** closed the last gap — a CUDA f16 auto-policy session (`--session` or `--slots-file`) now restores instead of refusing its own snapshot, and the Qwen3-0.6B configuration's real-model set is 31/0. Resuming the CLI's `--session` and an E2 slot snapshot: [#89](https://github.com/yusiwen/minfer/issues/89) · [#43](https://github.com/yusiwen/minfer/issues/43) |
| Prefix reuse across requests | ✔ **B2/B3** — ≈11× TTFT on the second turn |
| Quantized KV | ◐ **C4 S1 + S2a + S2b (2026-09-24, CPU and CUDA)**: `MINFER_CACHE_TYPE=q8_0` stores packed Q8_0 cells — 3.76× smaller regions, measured on both cached models. S2a reads the packed blocks directly on the CPU (Q8_0 × Q8_0 K dot, V out of the cell: 1.16×/1.31× over S1's dequantizing read at ctx 512/2048) and makes a physical `kv_rm`/`kv_shift` work on a packed region; S2b adds the CUDA kernels (a `KV_LAYOUT_F32/F16/Q8_0` tag with a byte-addressed `kv4<LAYOUT>` load, a packed store that uses the CPU's quantizer byte for byte) and flips the registry's `reads_packed_kv`. **The win is memory**: decode against f16 is 1.13× slower on Qwen3-0.6B and 1.48× on the 0.5B (1.25× of it the packed load, the rest the stated no-fused-epilogue cut); at hd 128 the packed **prefill is 15× slower** because the f16-typed FA path is not offered for a packed cell. Metal stays G5 ([#44](https://github.com/yusiwen/minfer/issues/44)); the remaining CUDA tuning is [#87](https://github.com/yusiwen/minfer/issues/87) · [#42](https://github.com/yusiwen/minfer/issues/42). **Format ownership is per engine since [#99](https://github.com/yusiwen/minfer/issues/99) (2026-09-25)**: the resolved format lives on the loaded model and reaches the graph through `CParams::kv_format` and the CPU kernels through `GraphAllocator::set_kv_format`; the CUDA *device* layout tag is still process-wide ([#153](https://github.com/yusiwen/minfer/issues/153)) |
| KV memory growth | fixed at first allocation, **never resized** |
| Position vs cell | ✔ **C6 (landed 2026-09-20, `001b8cc`)**: sequence-relative `positions` + an allocator-resolved `cells` input, so a cell move changes no rotation and compaction is bit-identical. The CUDA fused decode QKV family (`Op::FusedQKV`, `Op::QkvBiasRopeStore`) takes `cells` too, so the build-time gate that kept it on the unfused chain under `explicit_span` is now `(cuda_on \|\| !explicit_span)` — Metal keeps it (no explicit-span attention, G5). A request placed in a non-zero-start slot exposed and fixed a server position bug (`submit_on` still added the run start). The two user-visible limits this design leaves open are **C7** (one request may use the whole arena) and **C8** (cross-sequence sharing), both now scheduled |
| Elastic per-slot KV partition (a long request vs `n_slots`) | ✔ **C7 + C7b (2026-09-20)**: the engine sizes a slot from the request (`prompt + max_tokens + 1`, clamped to the arena); when that exceeds its share it reclaims **idle** runs for capacity and `KvCache::set_cap` returns a plan that moves whatever is in the way — **in either direction**, so a *busy* neighbour above the slot is no longer a wall (`order_moves`: upward top-down, downward bottom-up, upward first, with the owner table travelling in the same order). The HTTP bound moved from a slot's share to the whole arena, and CPU + CUDA `copy_cells` pin an overlapping upward move byte-for-byte. GB10: `--n-slots 4 --n-ctx 8192` serves a 2054-token prompt (2048 → 2071 cells, three idle slots released) with a continuation byte-identical to `--n-slots 1` |
| Multi-sequence attention masks | (Metal's port is **G5**, [#44](https://github.com/yusiwen/minfer/issues/44)) ✔ **E1 + E1b + E2** (device-aware default: see the batching row below): the allowed window is an explicit `attn_span` input resolved from the sequence's span list (C8b S1a/S1b, derived from cell ownership; multi-span layouts are refused until S2's `kv_map`), read by the CPU kernel and by CUDA's windowed kernel instantiations — **device-verified on GB10 since 2026-09-18**, and since 2026-09-19 swept over **both** KV dtypes (f16 and f32), which is what caught the f16 prefill mask fault (`ARCHITECTURE-EXECUTION-PLAN.md` §14 row 0). Metal still derives from `positions` and refuses a multi-sequence node — **G5 is scheduled** (after C8; CI's `build-macos` compile-checks it) |

**Gap.** 🔴 This is the single largest structural gap, because it blocks four
separate user-visible capabilities at once: multi-slot serving throughput,
cross-request prompt caching, long-conversation context handling without
re-prefill, and any state-space/hybrid model family.

Two concrete defects live here as well:

- **`ensure_kv` ignores the requested size after the first call**
  (`alloc.rs:384-392`: the early `if let Some(&pair) = self.kv.get(&layer)`
  returns without comparing `size`). `CParams.n_ctx` is part of the reuse
  identity, so a session that changes `n_ctx` on the same `GraphCache` silently
  keeps the old region. The CPU backend then errors on out-of-range positions
  (`cpu_backend.rs:170-172`); the GPU backends do not check at all (below).
  Today no caller changes `n_ctx` on a live cache, so it is latent rather than
  live — but it is a landmine directly under the "grow the context" feature
  that a server wants.
- **GPU `KvcacheStore` has no bounds validation.**
  `cpu_backend.rs:170` returns `Err` when `pos >= n_ctx`; the CUDA path
  (`cuda_backend.rs:1028-1061`) and the Metal path write unconditionally. A
  caller violating the documented contract produces an out-of-bounds device
  write instead of an error. `docs/GPU_SAFETY.md`'s rule — "guard failures abort
  with actual values" — is enforced on CPU and not on GPU here.

**What C1/C2 landed (2026-09-16).** `src/graph/kvcache.rs` now owns a
per-layer arena with an owner per cell, and `GraphAllocator::kv_rm(start, len,
&rope)` removes a row range and re-bases the rows after it — a *physical*
operation, so `cell == pos` survives and no backend needed a new kernel. The
conversation's overflow path uses it instead of dropping turns and re-prefilling
them. Its one approximation is inherent and recorded in
`ARCHITECTURE-EXECUTION-PLAN.md` §5 (C2 record): the retained rows hold the
values they were written with, so rows that attended to the dropped turns keep
that influence — no shift that avoids re-prefilling can avoid it (llama.cpp's
context shift behaves the same way). What is exact is the mechanism, and that is
what the tests pin bitwise: a tail removal leaves the retained head
byte-identical (so continuing from it matches a fresh prefill *exactly*), a
middle removal copies V verbatim and re-ropes only K, and the whole operation
keeps the identity cell mapping C1's scheduler gate requires.

**Recommendation.** Redesign the KV layer as a sequence-addressable cell store
*before* adding batched attention, because batching without per-sequence KV
addressing cannot be correct. Concretely: a `KvCache` owning a per-layer arena
of `n_ctx` cells, each carrying the set of sequence ids that own it; the store
node resolves `(layer, seq_id)` → cell index on the host and passes an index
array to the kernel; the attention kernel receives an explicit per-query
allowed-cell mask instead of deriving the bound from `positions`. **E1 landed
that last part on CPU**: `attn_span` carries each query's `[lo, hi)` cell range
(resolved from ownership + the query's position), and a representation as a range
is complete because a sequence's cells are contiguous — a per-cell mask is what a
hole-creating layout would need (C3/D1). Build **one**
such cache with an optional window parameter and an optional recurrent state,
rather than a family of per-variant caches. That single abstraction
simultaneously delivers prefix reuse (cells already owned by the matching
prefix), eviction (drop cell ownership rather than re-prefill), and
defragmentation (a cell-copy op that a strided-view IR makes expressible).

---

### 2.5 L5 — Batching and serving 🔴

**Today.** Single sequence by default on CPU — batching exists and its default
follows the device (E6: on for CUDA, off for CPU/Metal; `MINFER_BATCH=0/1` forces
it), slower on CPU (0.49x) and 1.9x faster on the GPU, where the E2 acceptance is
met. `GraphParams` carries no sequence
count at all: E2 deleted `n_seqs`, which A7 had kept as *reserved for item 3*,
once item 3 landed and showed the count is data rather than topology
(`CParams.explicit_span` carries the only topology decision it can force —
see `ARCHITECTURE-EXECUTION-PLAN.md` §8). The other dead identity field,
`CParams.n_batch`, was **deleted** in Phase A7; chunked prefill (item 10) will
reintroduce it with its real semantics. Attention derives its causal bound from
the per-token positions input (`cpu_backend.rs:411-416`, `cuda_backend.rs:1100-1102`), so two
sequences in one batch would attend to each other.

The prefill is one graph covering the entire prompt (`main.rs:862-890`):
`n_ctx = max(--n-ctx, prompt_len)`, one forward with `nt = prompt_len`. The
server rejects any prompt longer than the slot context (`chat.rs:118-124`) and
allocates `n_ctx_total / n_slots` per slot (`slot.rs:27-37`). `worker_loop`
drains the queue **serially, one slot at a time** (`chat.rs:452-518`), and every
request starts from a fresh `GraphCache` (`chat.rs:491`). Default `--n-slots` is
1 (`main.rs:211`), so the default server is strictly serial with a cold KV per
request.

**Gap.** 🔴 This is the largest capability difference between what minfer does
today and what a serving workload needs, and it costs throughput on exactly the
workload the server exists for. With `--n-slots N` minfer gets N independent
serial sessions, not N-way batching: aggregate decode throughput stays at
batch-1 tokens/s × (fraction of time the GPU is busy).

**Recommendation.** Work in dependency order: (1) KV cells → (2) IR `seq_id` and
mask inputs → (3) attention kernels with explicit masks → (4) batch composition
in the scheduler/worker → (5) `n_batch` chunking. Do not start with (4); it
cannot be made correct first.

Two smaller but immediate items sit in this layer:

- ~~**Persistent server context.** `chat.rs:491` discards the `GraphCache` per
  request. The comment justifies it as avoiding cross-request KV contamination
  (a real bug fixed in doc 97) — but the correct fix is sequence-aware KV
  invalidation (§2.4), not discarding the whole cache. As written, every request
  pays KV + pool re-allocation (≈235 MB for 7B at `n_ctx=4096`, f16) and
  invalidates the CUDA Graph capture warm-up (`pool_gen` bump,
  `cuda_backend.rs:1342`).~~ **Fixed in B2**: the slot keeps its cache and a
  record of the tokens its rows hold; a request reuses the KV only when its
  prompt starts with exactly that sequence. Measured end-to-end (0.5B Q4_0,
  219-token second turn): prefill 219 → 16 tokens, time-to-first-token
  ≈2.8 s → 0.25 s (≈11×), with the cold-slot turn unchanged.
- ~~**No panic isolation.**~~ **Fixed in A4.** The guard that existed
  (`guarded_forward`, `chat.rs:438-450`) only covered
  `forward_graph_cached`; the speculative path calls both models' forwards
  directly (`spec.rs`) and the tokenizer, sampler and stop-string paths were
  bare. A panic in any of them unwound `worker_loop`, dropping the bounded job
  channel (capacity 64, `server/mod.rs:66`) — later requests got 503 — and the
  queued jobs' `StreamEvent` senders, ending their SSE
  (Server-Sent Events) stream after an empty `[DONE]` rather than an error
  (`server/mod.rs:198-216`). The whole per-job body now runs under
  `run_job_isolated`, which turns a panic into a logged 500 for that request
  and keeps the worker draining the queue.

---

### 2.6 L6 — Backend abstraction and device reach

**Today.** `Backend` is a `Copy`/`Hash`/`Eq`/`Ord` **handle** over a fixed id space
(`graph/registry.rs`, F4) and a trait (`graph/backend.rs:21-95`). The ids are a
KV-session file-format contract, and the name-keyed registry carries each backend's
priority, capability matrix and pool hooks; consumers read it instead of matching.
Before F4 the enum was matched in `GraphAllocator::supports`
(`alloc.rs:140-157`), `alloc_in_pool`/`alloc_fresh_in`/`free_in_pool`
(`:314-380`), `sync_backend` (`:559-579`), `copy_across` (`:592-653`), and the
scheduler's execute match (`scheduler.rs:274-292`) — nine `#[cfg]`-laden match
sites, each of which had to be taught about a new backend.

GPU participation is decided by an all-or-nothing model-level gate: every weight
must be registered on the GPU or the model runs on CPU (`ARCHITECTURE.md §5.2`;
`CUDA-BACKEND-DESIGN.md:251-255`). The engine enumerates devices and honours
`--gpu N` but uses exactly one (`cuda.rs:1465-1501`); there is no
`tensor_split`, no peer copies, and no remote-device backend.

**Gap.** 🟠 Two axes.

*Pluggability* — **F4 (2026-09-24) landed the registry half**: the backend set is
data (a name-keyed table of entries registered at startup) rather than nine match
sites, and a name that is unknown, not compiled into this build, or not usable on
this machine is a loud startup refusal
(`BACKEND-REGISTRY-DESIGN.md`). What is **still** closed at compile time is the
*set itself*: there is no `dlopen`/plugin path, so a Vulkan backend (the only
route to Windows/Linux AMD and Intel GPUs) or a remote/RPC backend is added by
registering an entry — but it must still be compiled in. That is what remains of
this axis.

*Placement policy* — a model that does not fit in VRAM could not run at all. The
docs already record this as a real limitation on 8 GB devices
(`DEVICE-ADAPTATION-PLAN.md §9.2`: "7B Q8_0 will not fit (7.2 GB weights
alone)"). **E5 (2026-09-23) added the missing granularity and the fit**: `CNode.layer` (stamped
by the model builders), an `OffloadPlan` (`graph/offload.rs`) that the loader's
registration filter, the builder's device-only fused gates and the assignment pass
all read, `--gpu-layers N` / `MINFER_GPU_LAYERS=N`, the startup report, and a
verified mixed CPU+CUDA run on the 0.5B — then **S2**'s `auto`: per-block weight
bytes from the GGUF index, the pure `fit_blocks` prefix search against a weight
budget (`MINFER_GPU_MEM`, else three quarters of the device's free bytes) with a
quarter held back for the KV arenas and the activation pool. What remains on this
axis is a free-bytes query for Metal (so `auto` works there too); the registry/`BackendId`
half is **done** (F4).

**Recommendation.** (a) ~~Introduce a `BackendRegistry` with
`register(Box<dyn Backend>)`, `supports(op, dtype) -> Option<BackendId>` and
per-backend allocation/sync as trait methods; replace the enum with an opaque
`BackendId`.~~ **Done in F4 (2026-09-24)** — the shape that landed is a name-keyed
`Registry` of `BackendEntry` values (priority + caps + the pool hooks, registered at
startup) and a `Backend` handle over a fixed id space, with the assignment order
made an explicit pinned number (`docs/BACKEND-REGISTRY-DESIGN.md`). It keeps the
capability matrices in the trait's own modules (the trait methods forward to them),
so the registry and the trait cannot disagree. (b) Add a layer-offload budget policy that consumes the memory
accounting from §2.3.

---

### 2.7 L7 — Models, quantization, tokenizer, sampling

**Architectures.** Two (`qwen2`, `qwen3` dense), dispatched on
`general.architecture` (`models/mod.rs:105-121`). Every other family is absent:
MoE, MLA, sliding-window/hybrid, state-space (Mamba/RWKV), multimodal, and every
non-Qwen dense family (Llama, Mistral, Phi, GLM, InternLM). The per-family
decision, its prerequisites and its cost live in `MODEL-SUPPORT-ROADMAP.md`; the
caveat that matters here is §2.1 — its Tier 1 estimate holds only while a family
needs no new IR structure. 🟠

**Quantization coverage.** Eight types; no Q2_K/Q3_K/Q8_K, no I-quants, no
BF16/MXFP4/NVFP4 (`SUPPORT-MATRIX.md §Not Yet Supported`;
`CUDA_OPTIMIZATION.md §1.4` marks IQ/Q2/Q3 "not planned"). The notable ones for
reach are Q2_K/Q3_K (running large models on small machines) and BF16 (serving
unconverted checkpoints). 🟠 but legitimately deprioritized.

**Quantizer tooling — closed by F6 ([#49](https://github.com/yusiwen/minfer/issues/49), 2026-09-24), and
its f16-weight gap closed by [#141](https://github.com/yusiwen/minfer/issues/141) (2026-09-25).**
A GGUF v3 *writer* (`src/gguf_write.rs`), byte-verified weight encoders
(`src/quantize.rs`), a HuggingFace Qwen2 converter that satisfies the strict
loader (`src/convert.rs`), and the `convert` / `quantize` / `split` subcommands
(`src/tooling.rs`, `docs/GGUF-TOOLING.md`). A converted f16 model runs **on the
CPU and on CUDA** — the CPU graph path gained f16 weight dispatch in F6 and #141
vectorized that dot (AVX2 `F16C` / NEON `FCVTL`) and moved the multi-token row
loop onto the shared CPU pool (3.2 → 207 tok/s prefill on the 0.5B), while CUDA
registers the raw 2 B/element weights and converts in-register; **Metal refuses
f16 weights** until [#164](https://github.com/yusiwen/minfer/issues/164), and a
rewrite/split round-trip is bit-exact, and the HF conversion's f16 output is
byte-identical to llama.cpp's converter on the same checkpoint, per tensor.
**What remains** is scope F6 did not claim: the K-quant/I-quant encoders are
still refused by name ([#140](https://github.com/yusiwen/minfer/issues/140)) and
bf16 output is refused ([#142](https://github.com/yusiwen/minfer/issues/142)).
🟢 for conversion/quantization of the supported set.

**CPU SIMD.** Only `Q4_0` and `Q8_0` have AVX2 (Advanced Vector Extensions 2)
dot kernels (`quants.rs:42`, `:151`); `Q4_1`/`Q5_0`/`Q5_1`/`Q4_K`/`Q5_K`/`Q6_K`
are scalar on x86 and NEON-only on aarch64 (`quants.rs:891-1020` — the K-quant
dispatch has no `is_x86_feature_detected` branch at all). Since K-quants are the
most common GGUF format in circulation, x86 CPU inference is far below what the
hardware can deliver; the support matrix documents this (`SUPPORT-MATRIX.md`,
AVX2 column) and `CPU_OPTIMIZATIONS.md §P1` still lists Q4_K AVX2 as open. There
is also no AVX-512/VNNI/AMX path, no aarch64 `i8mm`, and **no weight
repacking** — the standard fix for this gap is to repack the quant blocks at
load time into SIMD-friendly interleaved layouts. 🟠 for anyone on x86; the fix
is well-understood and self-contained.

**Tokenizer.** Byte-level BPE only, loaded from GGUF metadata
(`tokenizer.rs`). F7 ([#50](https://github.com/yusiwen/minfer/issues/50),
2026-09-24) made `tokenizer.ggml.pre` authoritative with two hand-written rule
sets (`qwen2`, `qwen35`), replaced the silent `unwrap_or(0)` with a checked byte
fallback, and made every other pre-tokenizer value (including a missing one), a
non-`gpt2` model, an empty merge table and an incomplete byte vocabulary a loud
load refusal. What remains: **no SentencePiece/unigram or WordPiece path**, and
no `ignore_merges`/multi-regex pre-tokenizers (`llama3`, `default`,
`deepseek-*`, …) — those refuse by name, so the coverage gap is explicit rather
than a wrong split. Also no NFC normalization (llama.cpp does not apply one for
BPE either) and special-token *matching* is still a per-model hand-extension
(the DeepSeek-R1 fix noted in `AGENTS.md`), which does not scale. 🟡 for the
Qwen/Llama-3-shaped families this engine supports; 🟠 for anything
SentencePiece-based. Follow-up:
[#132](https://github.com/yusiwen/minfer/issues/132).

**Chat templates.** **Landed in F7** ([#50](https://github.com/yusiwen/minfer/issues/50),
2026-09-24). minijinja 2.21.0's `set_unknown_method_callback` is the extension
point, so Qwen3's `chat_template` renders (think-block extraction, tool-call
formatting, `enable_thinking`) with **no dependency change**; the reference
renderings are committed and byte-for-byte gated, and any template the engine
cannot render is a loud refusal naming the construct and line — validated at
load, so the CLI exits and the server refuses to start. What remains: a
`--chat-template <FILE>` override (a user-supplied template for a GGUF that has
none or a wrong one), `strftime_now`, and the Python `str` methods outside the
implemented set (each refused loudly). Design + accepted/refused sets:
`docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`. 🟢 for the supported models.

**Sampling.** One `SamplerConfig` pipeline in `sampler.rs` (#48, **landed
2026-09-24**): logit bias → penalties → DRY → greedy shortcut → top-k → typical
→ top-p → min-p → XTC → temperature **or** mirostat v1/v2, with every new knob
defaulting to a no-op so the pre-F3 chain is bit-identical. Landed from the old
gap list: min-p, typical, XTC, DRY, mirostat v1/v2, logit bias (and
frequency/presence penalties from the OpenAI plan). **Grammar / JSON-schema
constrained decoding landed in F2 (2026-09-24)**: `src/grammar.rs` compiles a
GBNF grammar (or a JSON Schema, via a generated GBNF) into a pushdown automaton
and masks the logits inside that one pipeline, between DRY and the greedy
shortcut; CLI `--grammar`/`--json-schema`, server `response_format`
(`json_object` / `json_schema`) plus a `grammar` field. Design and the exact
accepted subset: `docs/GRAMMAR-DESIGN.md`. What it deliberately does not do
(and refuses loudly rather than guessing): `pattern`/`minLength`/`maxLength`,
`multipleOf`, real-valued numeric bounds, object property permutations, and a
device-side mask — the mask is host work before any backend runs. Still
missing: top-n-sigma, adaptive-p, infill. 🟡

**No LoRA (Low-Rank Adaptation) / adapter support.** There is no adapter path
at all. minfer's `weights_version` field in `GraphParams` (`params.rs:62`) exists
precisely to break reuse on a weight swap, but nothing produces such a swap. 🟡

---

### 2.8 L8 — Ops coverage, safety, testing, observability

**Op coverage.** The `Op` vocabulary is deliberately small; six variants are
parity-only stubs that no architecture emits (`Scale`, `Softmax`, `View`,
`Reshape`, `Permute`, `AttnMode::Mha` — `COMPUTE-GRAPH-DESIGN.md §1.3`), and
`View`/`Reshape`/`Permute` execute as copies. CUDA additionally refuses
`transpose_b` matmul (`cuda_backend.rs:932-937`) and `FusedQkvNorm`
(`:1289-1321`), and Metal refuses `QkvBiasRopeStore` (`metal_backend.rs:282`) —
so the two GPU backends do not implement the same op set, and a model's decode
path differs by platform. ~~Neither is documented in `SUPPORT-MATRIX.md`.~~
**Fixed in A8**: `SUPPORT-MATRIX.md` now carries an "Operator Coverage by
Backend" table with the four asymmetric rows and their consequences.

**Guard asymmetry.** Metal uses `debug_assert!` where CUDA returns `Err` for the
same invariants (`metal_backend.rs:749`, `:800`, `:891`), so release builds pass
through invalid shapes; and Metal silently substitutes a weightless RMSNorm when
a weight name is missing (`:403-414`, `:457-468`) where CUDA errors
(`cuda_backend.rs:1246-1269`). Both violate the project's own stated rule that
"kernel-invariant violations return `Err`, never a silent fallback"
(`AGENTS.md §GPU Safety`). 🟠 — silent numerical degradation is the worst failure
class for an inference engine.

**Testing.** Five integration files, four of them `#![cfg(target_os = "macos")]`
Metal kernel isolation tests; CPU/CUDA correctness rests on inline unit tests
plus real-model tests that skip when a GGUF is not cached
(`tests/conversation_cli.rs:1-12`). ~~CI is `cargo build --release` on macOS only
— no test run, no CUDA build, no Linux build.~~ **Fixed in A2**: CI now runs
`cargo test --release` on Linux/CPU, compiles the CUDA backend in NVIDIA's devel
image, and keeps the macOS build. 🟠 The remaining highest-value addition is a
systematic op × dtype × backend correctness matrix (every op checked on every
backend against a CPU reference); it would have caught several items in §4
automatically — that is ticket A1.

**Observability.** **F8 landed 2026-09-24** ([#51](https://github.com/yusiwen/minfer/issues/51)):
`GET /metrics` (Prometheus text) exports live KV/arena occupancy, queue depth and
the running/in-flight counts; per-op timing is available under
`MINFER_OP_TIMING` (off by default, measured at the scheduler's per-node
dispatch); and SIGINT/SIGTERM drains gracefully, bounded by `MINFER_DRAIN_MS`.
Still missing: **structured logging and levels** (the server prints free-form
startup lines), and **worker-restart supervision** (a panicking job is isolated
per request by `run_job_isolated`, but a dead worker thread is not restarted).
The trace/viz stack remains a *development* instrument, not an operations one.
🟡 for a research engine, 🟠 the moment it is deployed.

---

## 3. Recommendation backlog

Effort: S ≤ 2 d · M ≤ 1 w · L ≤ 2 w · XL > 2 w. Backlog items tracked on GitHub link
their issue (the [open list](https://github.com/yusiwen/minfer/issues)); an item whose
work predates the tracker has no issue, and the plan is its record.

### P0 — structural, unblocks a class of use cases

| # | Item | Refs | Effort |
|---|---|---|---|
| 1 | **KV cache → sequence-addressable cell store** (cells + seq-id sets; host-resolved `(layer, seq)` → cell indices; explicit per-query mask passed to attention). Prerequisite for everything in P0. — **C1/C2/C3 landed** (cell store, physical removal/shift, compaction), **C6 landed 2026-09-20** (`001b8cc`: sequence-relative `positions` + an allocator-resolved `cells` input, so compaction is bit-identical) and **C7 + C7b landed 2026-09-20** (elastic partition: a long request reclaims idle capacity and the planner moves runs in both directions, so a busy neighbour cannot block it; the continuation stays byte-identical to a one-slot engine). Remaining: **C8** ([#41](https://github.com/yusiwen/minfer/issues/41), closed 2026-09-22), whose design splits it into **C8a ✔** (shared prefill with copied rows; measured 36.6× cheaper than the re-prefill it replaces) and **C8b** (true sharing: a per-sequence span list plus an *additive* `kv_map` the attention read path gathers from — S1a/S1b landed 2026-09-21 (write path and read path both resolve through the list; nothing observable changed), **S2 landed 2026-09-21** (admission shares a prefix in place — no bytes copied — and the CPU kernel gathers the map; block refcounts were **not** added, occupancy stays derived from the span lists), **S3 landed 2026-09-22** (copy-on-write: a store inside a shared prefix rebases the sequence's run and shifts its own rows up, so a shared block is never written through), **S4 landed 2026-09-22** (CUDA gathers the map in *every* attention path, FA prefill included, at 1.001x the span on decode and 1.1x on prefill; admission switches on `Device::gathers_attn_map`) and **S5 landed 2026-09-22** (Metal refuses both window layouts loudly at assignment and at execution, so C8b is closed on CPU and CUDA; Metal's share path is **G5**, where its cell store is ported) | §2.4 | XL |
| 2 | **IR `seq_id` + attention-mask inputs**; attention kernels take an allowed-cell mask instead of deriving the bound from `positions`. — **done** (E1: span input + resolver + CPU kernel + two-sequence test; E1b: CUDA windowed kernels, compile-verified) | §2.5 | L |
| 3 | **Batch composition + continuous batching** in the scheduler and server worker; make `n_seqs` real (or delete it). — **mechanism landed in E2**: sequence-addressable batches, per-sequence reservations, the server composes one decode batch and batched prefills; `n_seqs` **deleted** (it turned out to be data, closing A7); **opt-in** (`MINFER_BATCH=1`): the payoff is **device-dependent** — 0.49x on CPU (7B Q4_K_M) but **1.9x on the GB10 GPU** (measured 2026-09-18, after A0's "no device" verdict turned out to be an agent-sandbox artefact); the CPU refutation closed the ticket and the GPU result is the acceptance the closure predicted; the device-aware **default** landed in follow-up ticket **E6** (2026-09-19: unset batches iff the model runs on CUDA, with `MINFER_BATCH=0/1` to override — 1.97x on the 7B with no environment variable); see the plan's E2/E6 records | §2.5 | XL |
| 4 | **Persistent server context**: keep the `GraphCache` across requests, invalidate per sequence id; stop re-allocating KV and re-warming CUDA Graph capture per request. — **done in B2/B3** (prefix-matched reuse, ≈11× TTFT on turn 2) | §2.5 | M |
| 5 | **Fix `ensure_kv` size handling** and add the missing `pos < n_ctx` guard on both GPU backends. | §2.4 | S |
| 6 | **Worker panic isolation** (`catch_unwind` + supervision + an error event instead of a silent empty stream). — **done in A4** | §2.5 | S |

### P1 — material capability or performance

| # | Item | Refs | Effort |
|---|---|---|---|
| 7 | **IR expressiveness**: strided views with allocator-known aliasing, multi-output nodes; then re-express the four decode fusions as compositions. — **DONE (D1, 2026-09-19), increments 1–3**: exact views are zero-copy with allocator-known aliasing (`CNode.view` + liveness + no-op kernels), **offset/partial windows work on CPU and CUDA** (`BufRef` carries `offset`+`len` through the `Backend` trait; Metal is exact-only until G5), and the "multi-output node" is `GraphBuilder::split_parts` — one owning node plus one `Op::View` per part, so a producer's single output feeds several consumers with independently bindable tensors while the graph stays single-output. D2's windows are in production; the sketched `Op::SplitParts` was deliberately not added (a split op over one input is just views of that input). **Item 17 (MoE) therefore no longer waits on an IR blocker** — what it needs now is model work (expert weights, routing, the grouped GEMM), and item 16 (MLA) likewise | §2.1 | L |
| 8 | **Allocator reserve/assign split** + size-class rounding + real memory accounting + VRAM feasibility gate. — **S1 landed 2026-09-22**: the size-class ladder and a pure plan (`graph/allocplan.rs`), per-backend accounting (`memory_report`: weights/pool/live/peak + `headroom_bytes`), and a feasibility gate that refuses an over-budget graph *before* the pool is touched, naming its numbers. **S2 landed 2026-09-23**: the pools allocate at `class_size(size)` (two shapes in one class share a buffer across a rebuild), the node's logical length moved into `BufRef` (`write_host_window`, windowed capture reads, fills checked against `BufRef::len`), `pool_bytes` counts what the pool *holds*, all input buffers are placed before the walk (an input is filled before execution, so a released buffer would be clobbered by its previous owner's write), and a liveness extension now moves the buffer's `buf_alive` deadline — D1's view branch never ran at all. **S3 landed 2026-09-23**: reservation split from assignment (a `slots` table per backend/class; a released classed buffer is re-assigned, not returned to the backend), so a rebuild re-maps — CUDA's `pool_gen` stops moving and a captured graph survives it — and `GraphCache` holds one graph per `GraphParams` (a switch re-maps; a repeated chunked prefill builds nothing: 3 builds/5 reuses then 3/9) · [#55](https://github.com/yusiwen/minfer/issues/55) | §2.3 | L |
| 9 | **Layer offload policy** on top of (8); needs a layer-granular assignment pass. — **S1 landed 2026-09-23 (E5)**: `graph/offload.rs` (`OffloadPlan` + the pure resolver), `CNode.layer` + `GraphBuilder::set_layer`, `GraphAllocator::supports_for` refusing the device past the plan, per-block weight registration (`OffloadPlan::allows_weight`, the block parsed from the registry name), `--gpu-layers`/`MINFER_GPU_LAYERS`, `CParams.gpu_layers` in the reuse identity, the startup report, and the mixed-run gate (4/24 blocks on CUDA matching the all-CPU greedy tokens, one device split per offloaded block with cross-backend copies at the boundaries). **S2 landed 2026-09-23**: the `auto` request fits the largest block **prefix** into a weight budget (`MINFER_GPU_MEM`, else three quarters of the device's free bytes — the same default E4's gate uses) with a quarter held back for KV/activations; per-block bytes come from the GGUF index (measured before anything is registered, because the filter *is* the plan); the startup line explains the fit. Real-model gate: `MINFER_GPU_MEM=64` → 5 of 24 blocks, 40.0 MiB on the device, greedy tokens matching the all-CPU run; uncapped → all 24 · [#46](https://github.com/yusiwen/minfer/issues/46) | §2.6 | L |
| 10 | **Chunked prefill**: make `n_batch` real; cap activation memory and allow decode/prefill interleaving. — **landed in E3 (2026-09-22)**: `prefill_chunks` + `MINFER_N_BATCH` (default 2048, a no-op for prompts that fit), remainder-last so the final forward carries the tail row, and the other slots take their decode step *between* chunks. Measured: 5 forwards / max `nt` 24 vs 1 / 98 for a 98-token prompt; logits bitwise on CPU and ≤ 0.218 (class 1.0) on CUDA; the interleaving A/B is 3 decode steps vs 0; the split costs one forward's fixed overhead per chunk (514 tokens in 4 forwards: CUDA 1.11x, CPU 1.005x; 98 tokens in 5: CUDA 2.0x). Not mixed prefill+decode batches yet · [#45](https://github.com/yusiwen/minfer/issues/45) | §2.5 | M |
| 11 | **CPU AVX2 (and AVX-512/VNNI where available) for the K-quant dots**; then weight repacking. | §2.7 | L |
| 12 | **Backend registry** decoupling the enum from the nine match sites. — **landed in F4 (2026-09-24)**: [#57](https://github.com/yusiwen/minfer/issues/57); the handle + the name-keyed table live in `graph/registry.rs`, the priority order is a pinned number, and the name surface (`--backend` / `MINFER_BACKENDS`) has three distinct loud startup refusals (`docs/BACKEND-REGISTRY-DESIGN.md`). The registered set itself stays compile-time (no `dlopen`) | §2.6 | M |
| 13 | **Guard symmetry**: Metal `Err` instead of `debug_assert!`/weightless fallback; CUDA gains `FusedQkvNorm` or `SUPPORT-MATRIX.md` gains a per-backend op column. — **docs route done in A8**; the Metal half defers to Phase G | §2.8 | S |
| 14 | **Async cross-backend copy + events** (needed for any heterogeneous split and for multi-device execution). — **landed in F5 (2026-09-24)**: [#58](https://github.com/yusiwen/minfer/issues/58); the boundary's two registered phases (`BackendEntry::copy_cross`/`await_cross`), CUDA's `cudaMemcpyAsync` D2H + event + one `cudaEventSynchronize` at the documented point, the CPU's documented no-op, counter-gated (`graph/copystats.rs`: zero blocking boundary copies, one wait per copy) and bitwise-identical to the `MINFER_SYNC_COPIES` reference on a real split model. **Metal declines (unported, no Mac to verify) and true overlap is still open** — the deferred-wait scheduling change plus the Metal port are the follow-ups | §2.2 | M |

### P2 — coverage

| # | Item | Refs | Effort |
|---|---|---|---|
| 15 | **Constrained decoding**: GBNF-style grammar + JSON-schema → grammar. — **landed in F2 (2026-09-24)**: [#47](https://github.com/yusiwen/minfer/issues/47), `src/grammar.rs` + the mask in `src/sampler.rs`; the design record is `docs/GRAMMAR-DESIGN.md` | §2.7 | M |
| 16 | **Sampler set**: min-p, typical, XTC, DRY, mirostat, logit bias. — **landed in F3 (2026-09-24)**: [#48](https://github.com/yusiwen/minfer/issues/48) | §2.7 | M |
| 17 | **MoE support** (see `MODEL-SUPPORT-ROADMAP.md` Tier 2 #1) — depends on item 7 for a clean implementation. | §2.1, §2.7 | L |
| 18 | **Dense architecture port wave** (see `MODEL-SUPPORT-ROADMAP.md` Tier 1) — parameter mapping plus the per-port items listed there. | §2.7 | M–L |
| 19 | **Chat-template fidelity**: replace or extend minijinja so Qwen3's template actually renders. — **landed in F7 (2026-09-24)**: [#50](https://github.com/yusiwen/minfer/issues/50), the Python-`str`-method hook + the loud refusal; design record `docs/CHAT-TEMPLATE-AND-TOKENIZER-DESIGN.md`; remains: `--chat-template <FILE>` + `strftime_now` ([#133](https://github.com/yusiwen/minfer/issues/133)) | §2.7 | M |
| 20 | **Tokenizer generality**: SentencePiece/unigram, per-model pre-tokenizer variants, data-driven special tokens. — **partly landed in F7 (2026-09-24)**: [#50](https://github.com/yusiwen/minfer/issues/50) made `tokenizer.ggml.pre` authoritative (`qwen2`/`qwen35`, byte-for-byte gated, unknown values refused) and replaced the silent byte fallback; SentencePiece/unigram/WordPiece and the remaining pre-tokenizer rules are [#132](https://github.com/yusiwen/minfer/issues/132) | §2.7 | M |
| 21 | **Quantized KV** (Q8_0 first) — after item 1. — **landed in C4 S1 + S2a on the CPU and S2b on CUDA (fused read, quantize-aware shift, layout-tagged kernels); the remaining CUDA tuning (packed fused epilogue, dp4a packed dot, packed FA prefill) and Metal's half are [#87](https://github.com/yusiwen/minfer/issues/87) and [#44](https://github.com/yusiwen/minfer/issues/44)** | §2.4 | M |
| 22 | **Quantizer tooling**: `convert-hf-to-gguf` + `quantize` + `split`. | §2.7 | L |

### P3 — hygiene and operations

| # | Item | Refs | Effort |
|---|---|---|---|
| 23 | **Op × dtype × backend matrix test**. — **done in A1** | §2.8 | M |
| 24 | **CI**: run tests on macOS, add a Linux CPU job, add a CUDA build job. — **done in A2** | §2.8 | S |
| 25 | **Metrics/observability**: `/metrics`, KV occupancy, queue depth, per-op timing under a flag, graceful drain. — **done in F8** ([#51](https://github.com/yusiwen/minfer/issues/51), 2026-09-24; the only member of the A-era batch that had no ticket). Structured logging/levels and worker supervision were **not** part of the ticket and remain open in §2.8 | §2.8 | M |
| 26 | **Remove the dead identity fields**: delete `CParams.n_batch`; keep `GraphParams.n_seqs` marked *reserved for item 3* (decision recorded in `ARCHITECTURE-EXECUTION-PLAN.md` §8). — **done in A7; fully closed in E2**, which deleted `n_seqs` too after item 3 showed it was redundant, not just unread | §2.5 | S |
| 27 | **Re-key the cross-backend staging map** by `(node, dst_backend)`. — **done in A5** | §2.2 | S |
| 28 | **CPU per-op allocations**: `cpu_backend.rs:157-158` clones the K/V sources on every store node and `:195` allocates a `Vec<&[f32]>` per node. — **closed in A6 as not worth doing**: the allocation removal measured −1.2 % prefill / −1.8 % decode and was reverted (the loop is weight-streaming bound) | §2.3 | S |

---

## 4. Concrete defects found (verified in code)

Ordered by severity. Items 1–6 are behavioural; 7–12 are hygiene; 13–14 are
behavioural defects found while executing the plan, already fixed.

1. **GPU KV store has no bounds check.** CPU returns `Err` for `pos >= n_ctx`
   (`cpu_backend.rs:170-172`); CUDA (`cuda_backend.rs:1028-1061`) and Metal do
   not. A contract violation becomes an out-of-bounds device write. *Fix: one
   guard, mirrored from the CPU path.*
2. **`ensure_kv` ignores a changed size** (`alloc.rs:384-392`). The KV region is
   frozen at first allocation while `CParams.n_ctx` remains part of the reuse
   identity, so a size change is neither honoured nor detected.
3. **Metal weakens kernel guards.** `debug_assert!` where CUDA returns `Err`
   (`metal_backend.rs:749`, `:800`, `:891`) and a silent weightless RMSNorm when
   a weight is missing (`:403-414`, `:457-468`) — both contradict
   `docs/GPU_SAFETY.md` and `AGENTS.md`'s no-silent-fallback rule.
4. ~~**Server worker had no panic isolation outside the forward call**
   (`chat.rs:454-518`): a panic anywhere else unwound the worker, permanently
   degrading the server (503 for new jobs, empty 200/SSE for queued ones) with
   no log.~~ **Fixed in A4** — the whole per-job body is now guarded.
5. ~~**Backend op-set asymmetry drives silent path changes.** `FusedQkvNorm` is
   Metal-only (`metal_backend.rs:276`) but absent from CUDA's `supports_op`
   (`cuda_backend.rs:1289-1321`), so Qwen3 decode takes the fused path on Metal
   and the unfused path on CUDA. `QkvBiasRopeStore` is the mirror case
   (`metal_backend.rs:282`). Neither is documented in `SUPPORT-MATRIX.md`.~~
   **Documented in A8**: `SUPPORT-MATRIX.md` now has an "Operator Coverage by
   Backend" table; the asymmetry is visible rather than silent.
6. **CUDA RoPE is non-interleaved only** (`cuda_backend.rs:1324`), so any model
   needing the interleaved style splits every layer between CUDA and CPU,
   producing two host round trips per layer (§2.2). `ARCHITECTURE.md:391-393`
   advertises both styles as available.
7. **Stale `unreachable!("CUDA pool not implemented")`** in the non-CUDA arms
   (`alloc.rs:332`, `:356`) — misleading text in a live panic path.
8. ~~**Dead fields in the reuse identity**: `CParams.n_batch` and
   `GraphParams.n_seqs` are compared by `params_match` (`cache.rs:57-64`) but no
   builder reads them; every construction site hard-codes 1 / `n_tokens`.~~
   **Fixed in A7, closed in E2**: `n_batch` is deleted; `n_seqs` was kept for
   item 3 and then deleted by it (the sequence count is data — the topology
   decision lives in `CParams.explicit_span`), with
   `sequence_count_is_data_not_topology` pinning that a sequence-count change no
   longer rebuilds an otherwise identical graph.
9. ~~**Single-entry cross-backend staging** (`alloc.rs:34`, `:645`), mitigated by
   the consumer-side filter at `scheduler.rs:252-255`.~~ **Fixed in A5** — keyed
   by `(node, dst_backend)`; two foreign consumers can now be served.
10. **`read_host` returns `None` on CUDA** (`cuda_backend.rs:1403-1408`), so the
    trait's host-read contract is backend-dependent; the allocator compensates
    with `copy_to_host` (`alloc.rs:509`).
11. **CUDA pool never releases device memory** (`cuda_backend.rs:1355-1362`),
    documented as accepted debt but reasoned about for fixed-shape CLI runs; a
    varying-`n_tokens` workload accumulates one buffer set per distinct shape.
12. ~~**Tests**: no Linux/CUDA/CPU in CI, four of five integration files macOS-only
    (`.github/workflows/ci.yml`, `tests/*.rs`).~~ **Fixed in A2** for the CI half
    (Linux/CPU tests, CUDA compile, plus `--no-run` test compilation); the four
    macOS-only kernel-isolation files remain macOS-only by nature.
13. ~~**`Op::Softmax` returned unnormalised values on the CPU backend**
    (`cpu_backend.rs`): it called `vec_soft_max_f32`, which writes
    `exp(x - max)` and *returns* the sum, and discarded the return. No
    architecture emits a standalone `Softmax` node, so nothing exercised it.~~
    **Found by the A1 op matrix and fixed**: the arm now scales by `1/sum`, with
    an op-matrix case pinning it.
14. ~~**The CPU attention was not `nt`-invariant.** Each token's scores were
    padded to the batch-wide `nkv`, and the softmax, the normalisation and the
    weighted sum all ran over `nkv` instead of the token's own causal window
    `vl = pos+1`. The padded entries are `-inf → 0`, so the arithmetic reads as
    equivalent — but the reduction *length*, and with it the rounding, depended
    on how many tokens shared the batch. Measured on 0.5B: a one-token decode
    step differed from a single-shot prefill by `max|Δlogits| = 0.41`, and the
    same token's K rows differed from layer 3 on (`nt=6` vs `nt=13`).~~
    **Found while testing prefix reuse for B2 and fixed**: restricting the
    window to `vl` makes incremental prefill, the decode loop and a single-shot
    prefill bitwise identical, and removes the padding pass. It also matters
    beyond B2 — the speculative-decoding identity gates assume exactly this
    property.

---

## 5. Suggested execution order

The dependency structure is more informative than the priority table alone:

```
                       ┌──────────────────────────────────────────┐
                       │ 1. KV cell store (seq-addressable)       │
                       └───────────────┬──────────────────────────┘
                                       │
        ┌──────────────────────────────┼───────────────────────────────┐
        ▼                              ▼                               ▼
 2. IR seq_id + mask         4. persistent server context    21. quantized KV
        │                              │
        ▼                              │
 3. continuous batching  ◄──────────────┘   (4 alone already removes
        │                                    per-request realloc/re-warm)
        │
        ├──► 10. chunked prefill
        └──► 9. layer offload ◄── 8. allocator reserve + accounting

  7. IR views/multi-output ──► 17. MoE, 11. CPU SIMD (independent)
  12. backend registry      ──► 14. async copy ──► multi-device execution
```

Items 5, 6, 13, 26–28 are independent, small, and can land at any time — they
are the ones worth doing first simply because they are cheap and they remove
hazards that the larger work would otherwise have to work around.

---

## Appendix — evidence index

| Area | Primary sources read |
|---|---|
| IR / builder / ops | `src/graph/mod.rs`, `ops.rs`, `builder.rs` |
| Scheduler | `src/graph/scheduler.rs` |
| Allocator | `src/graph/alloc.rs` |
| Reuse / params | `src/graph/cache.rs`, `params.rs` |
| Backend trait | `src/graph/backend.rs` |
| Backend registry (F4) | `src/graph/registry.rs`, `docs/BACKEND-REGISTRY-DESIGN.md` |
| CPU execution | `src/graph/cpu_backend.rs`, `src/kernel.rs`, `src/quants.rs` |
| CUDA execution | `src/graph/cuda_backend.rs`, `src/cuda.rs`, `src/cuda_kernels.cu` |
| Metal execution | `src/graph/metal_backend.rs`, `src/metal.rs`, `src/metal.metal` |
| Models | `src/models/mod.rs`, `models/qwen2/graph.rs`, `models/qwen3/graph.rs` |
| Serving | `src/server/{mod,chat,slot}.rs`, `src/conversation.rs` |
| Sampling | `src/sampler.rs`, `src/tokenizer.rs`, `src/template.rs` |
| Speculative | `src/spec.rs` |
| Docs | `AGENTS.md`, `README.md`, `docs/{ARCHITECTURE,COMPUTE-GRAPH-DESIGN,MODEL-SUPPORT-ROADMAP,SUPPORT-MATRIX,METAL_OPTIMIZATIONS,CUDA_OPTIMIZATION,CUDA-BACKEND-DESIGN,DEVICE-ADAPTATION-PLAN,OPENAI-CHAT-API-PLAN,SPECULATIVE-DECODING-PLAN}.md` |
