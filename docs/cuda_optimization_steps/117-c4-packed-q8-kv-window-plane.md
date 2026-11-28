# 117 · #862: the staging plane is window-bounded, and a budget guard can refuse it (LANDED)

> **Result**: the packed prefill's f16 staging plane is sized by the **window**
> (`min(region rows, round_up(max(position)+1, 256))`) instead of the arena: **8 MiB instead of
> 16 MiB** at hd 128 and `n_ctx` 4 096, and **8 MiB instead of 128 MiB** for a 2 048-token prompt at
> `n_ctx` 32 768 (16x). Time is unchanged — hd-128 `pp2048` staged **8 663.2 tok/s** against the
> per-tile arm's 8 540.5 (**1.0144x**, the same win record 116 measured), hd 64 11 601.5 vs 11 614.8
> (0.9989x) — and the output stays bit-identical. A new **budget guard** refuses a plane that would
> take more than 1/8 of free device memory, falling back to the per-tile route rather than failing.
> **Bar**: [#862](https://github.com/yusiwen/minfer/issues/862)'s acceptance plus the pre-registered
> rule in the 2026-11-28 comment
> ([#862 comment](https://github.com/yusiwen/minfer/issues/862#issuecomment-7235279558)) — the
> sizing asserted on the device, no time regression, bit identity. **MET**.
> **Commit**: `<this PR>`. **Date**: 2026-11-28. **Box**: `dgxspark` (aarch64, GB10 sm_121,
> driver 580.178.04, CUDA 13.0).

## 1. Background — the plane was sized for the wrong bound

Record [116](./116-c4-packed-q8-kv-once-per-layer.md) left one kernel: `dequant_kv_q8_0_f16` costs
**35.68 µs per layer** at hd 128 (1.00 ms of a 235 ms prompt, 0.43%) and **2.50 µs** at hd 64, and it
is memory-bound (~350 GB/s effective). Its *time* is therefore nearly at the floor. Its *memory* was
not: the plane was sized from the KV region's row capacity — the arena — because the dispatch "did
not know the window", and the window is device data by the time attention runs.

That reasoning was half right. The window is device data **at attention time**; it is host data one
step earlier, in `GraphAllocator::fill_input_i32`, which is the single place positions become graph
data (it already scans them for `check_positions_bound`). A 32k context being asked a 2k prompt is the
normal case for a server, and it paid **16x** the window's memory for a transient scratch: 128 MiB
where 8 MiB does.

## 2. Principle — an upper bound, a bucket, and a high-water mark

**The bound.** `max(positions) + 1` is an upper bound on every row the causal staging reads: the
kernel's tile-wide `kv_end` is `bound[last_t] + 1` and the staging only walks forward from cell 0, so
a plane with that many rows is complete. It is exactly the device's own `nrows` when positions are
monotone (the normal case) and conservative when they are not. No sync is needed in either direction.

**The bucket.** The hint is rounded up to `PLANE_ROW_BUCKET = 256` rows (512 KiB per plane at hd 128)
and capped by the arena. Without it, every slightly different prompt would grow the scratch — and a
growth is not free: `get_or_grow` frees the old pointer, and a **captured prefill** (which is on by
default for repeated identical prefills) may have embedded it. So the growth bumps `pool_gen`,
exactly as `pos_scratch`'s growth does, and the captured graph re-captures. The bucket keeps that
rare; after a few prompts the plane sits at the largest window the engine has seen.

**The high-water mark.** It never shrinks. The win is "window, not arena", not "exactly this prompt";
shrinking would need a different caller knowing that no captured graph references the old size.

**The guard.** A plane is a *caching* scratch: the per-tile route needs none. So it is admitted only
when `need <= free / KV_STAGE_FREE_DIVISOR` (8), a pure rule with pass/fail cases
(`allocplan::kv_stage_admitted`). The refusal is the documented per-tile fallback, announced once on
stderr and counted at its chokepoint, never an error. A **failed** memory query is not a refusal —
#122's rule, in the other direction: unknown is not zero, and the allocation itself is the authority.

## 3. Implementation

- `graph/alloc.rs::fill_input_i32`: for `positions`, record `max + 1` on the CUDA backend
  (`note_plane_rows`), keyed by that input's buffer id. No entry means "no hint" and the arena.
- `graph/cuda_backend.rs`: the hint map, `plane_rows(region_rows, hint)` (pure: bucket, cap, and
  the no-hint arm), the plane-bytes high-water mark that bumps `pool_gen` on growth, and the
  `kv_rows` it hands to the dispatch.
- `graph/allocplan.rs`: `KV_STAGE_FREE_DIVISOR` + `kv_stage_admitted(need, mem)` (pure).
- `cuda/methods/attention.rs`: the guard in front of `get_or_grow`, the one-time warning, and the
  `cuda_kv_stage_refused` chokepoint counter. `cuda_kv_stage_budget` (the #147 seam) drives the
  refusal in a gate.
- Tests: `the_kv_stage_plane_needs_an_eighth_of_free_memory` (pure policy: the boundary is
  inclusive, a zero-free device admits nothing, a failed query admits — verified as an unknown, not
  as memory); and the packed parity gate grows two device arms — a 100-row window pointed at a
  1 024-row region must allocate `4 * 256 * nkt` bytes and stay bit-identical, and an armed refusal
  must leave the plane at 0 bytes, produce the per-tile output bit for bit, and move
  `cuda_kv_stage_refused`.

## 4. Verification

- `scripts/cuda_test.sh` (CUDA build, serial, `dgxspark`): **868 passed / 0 failed / 62 ignored**
  (867 before this step; the +1 is the pure policy test).
- The live-checked x86_64 baseline row moves **730 -> 731** (`scripts/test-baselines.toml` and
  `docs/TEST-BASELINES.md` in this commit): the new test is platform-neutral, and that row is compared
  live by `test-linux-cpu`.
- Device gates (in `cuda_q8_0_fa_prefill_attention_parity`): the window-sized plane asserted by byte
  count against a 10x larger region; the armed refusal asserted by zero allocation, zero staged
  launches and bit-identical output.
- No time regression and no numerics change: `pp2048` at hd 128 **8 663.2** (hint) vs **8 540.5**
  (per-tile) = 1.0144x, at hd 64 **11 601.5** vs **11 614.8** = 0.9989x (3 alternating rounds,
  medians); the `MINFER_NO_KV_STAGE=1` bit-identity arm is unchanged and green.

## 5. Results

| plane, hd 128 (`nk * hd * 2 B` per row, K + V) | before (arena) | after (window) |
|---|---|---|
| `n_ctx` 4 096, `pp2048` | 16 MiB | **8 MiB** |
| `n_ctx` 16 384, `pp2048` | 64 MiB | **8 MiB** |
| `n_ctx` 32 768, `pp2048` | 128 MiB | **8 MiB** |
| hd 64, `n_ctx` 4 096, `pp2048` | 2 MiB | 1 MiB |

The banked time is unchanged by design: the pass writes the same rows either way (`nrows` is the
device's `bound[nt-1] + 1`, and the plane only has to be at least that). The guard adds a free-memory
read and a policy branch on the *first* staging launch of a forward; nothing per layer.

## 6. Lessons

1. **"The host cannot know it" is a claim about a moment, not about the system.** The window is device
   data at attention time and host data at fill time; one map turned a 16x over-allocation into a
   one-line lookup.
2. **A growable pointer is a capture-invalidation event.** The scratch pattern (grow, free, realloc)
   is only safe because something bumps the identity a capture is keyed on; a plane whose size now
   varies with the prompt had to say so out loud (`pool_gen`), or a captured prefill would replay
   into freed memory.
3. **A scratch that can be refused needs its fallback exercised.** The guard is three lines of policy
   and one branch, and the gate drives it three ways: pure boundary cases, an armed refusal on the
   device, and the bit-identity that proves the fallback is the per-tile route and not a wrong answer.

<- [116 · #858: the packed prefill stages once per layer — the hd-128 cell at 1.021x](./116-c4-packed-q8-kv-once-per-layer.md) · [Index](./index.md)
