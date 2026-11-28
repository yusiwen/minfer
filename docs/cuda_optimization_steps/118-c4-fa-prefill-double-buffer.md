# 118 · #865: double-buffered FA staging — where it fits (hd 64, −4%), and the occupancy wall it hit (LANDED)

> **Result**: the f16 FA prefill double-buffers its K/V tile staging **at hd 64 only**, where a
> second tile pair still fits the SM's shared-memory budget: the launch drops **170.2-172.9 µs →
> 165.5-165.6 µs** (median, nsys; **−4.3%**) and the packed staged arm inherits it. At hd 128 the same
> change was measured **413.2-418.6 → 479.6 µs (+14.6%)** and is *not* shipped: a second pair takes a
> block from 34,816 to 52,224 B, which drops the SM from 2 resident blocks to 1 on a 102,400 B budget.
> Prompt-level effect at hd 64 is ~0.1% and inside noise (attention is 2.3% of that prompt).
> **Bar**: [#865](https://github.com/yusiwen/minfer/issues/865)'s own pre-registration — a per-launch
> `nsys` median on the f16 arm first, both head dims, bit identity, no regression. **MET for hd 64;
> the hd-128 arm is a recorded, unshipped negative.**
> **Commit**: `<this PR>`. **Date**: 2026-11-28. **Box**: `dgxspark` (aarch64, GB10 sm_121,
> driver 580.178.04, CUDA 13.0).

## 1. Background — where the time actually is

Record [117](./117-c4-packed-q8-kv-window-plane.md) left the packed prefill at one memory-bound pass
(36.9 µs/layer) over a f16 body that costs 410-419 µs per launch at hd 128. [#865](https://github.com/yusiwen/minfer/issues/865)
was filed with an ablation of that body (doubling a loop and reading the marginal cost): QK^T 12%,
P·V 16%, tile staging 8%, `expf` 7%, **remainder 56%** — and `ptxas` reporting 127-131 registers and
**0 spill**, i.e. the launch is latency-bound, not compute-bound. The first increment of the ticket
is the cheapest latency lever: make the tile staging genuinely asynchronous (the header comment
claimed it until [#860](https://github.com/yusiwen/minfer/issues/860) — the k-loop issues `cp.async`,
commits, `wait_group 0` and syncs, so every tile pays the full copy latency before the maths starts).

## 2. Principle — prefetch-ahead, and the budget that decides where it fits

With two K/V tile pairs, the loop issues tile `kt + FA_TKV` into the other buffer at the top of each
iteration and retires only the group *before* it (`cp.async.wait_group 1`), so the copy of the next
tile overlaps this tile's QK^T/softmax/P·V. The trailing `__syncthreads()` is what keeps the buffer
the next iteration writes from being the one this iteration still reads.

The cost is shared memory, and on this device shared memory decides the block count:

| shape | K/V tile bytes | block total (Q + K/V) | blocks/SM (smem) | blocks/SM (regs) | resident |
|---|---|---|---|---|---|
| hd 64, single | 9,216 | 18,432 | 5 | 3 | **3** |
| hd 64, double | 18,432 | 27,648 | 3 | 3 | **3** |
| hd 128, single | 17,408 | 34,816 | 2 | 3 | **2** |
| hd 128, double | 34,816 | 52,224 | 1 | 3 | **1** |

`cudaDeviceProp` on the GB10 answers `sharedMemPerMultiprocessor = 102,400`, `regsPerMultiprocessor =
65,536`, 48 SMs. So at hd 64 the doubled pair is free (registers already cap the block count at 3) and
at hd 128 it halves the resident blocks — which is exactly what the measurement shows: hd 64 −4.3%,
hd 128 **+14.6%**.

## 3. Implementation

- `attention_prefill.cu`: `constexpr int NBUF = (LAYOUT == KV_LAYOUT_F16 && HD == 64) ? 2 : 1;`, two
  tile rows, the prefetch-ahead loop (with `wait_group 1` when a next tile exists and `wait_group 0`
  on the last), and `bidx ^= (NBUF - 1)` at the end of each iteration. The packed layouts and the
  pre-sm80 path keep the stage-then-wait shape; hd 128's f16 instantiation compiles to the old code
  path byte for byte (the smem formula yields the same 34,816 B, so no `cudaFuncSetAttribute` opt-in
  is needed anywhere).
- No Rust change, no new knob: the shape decision is compile-time and the existing gates cover it.

## 4. Verification

- `scripts/cuda_test.sh`: **868 passed / 0 failed / 62 ignored** (unchanged count — no test added).
- Parity: `cuda_fa_prefill_attention_parity` and its packed twin read the same CPU-reference deltas
  as before (3.44e-4 / 2.78e-4 f16, 6.72e-4 / 4.80e-4 packed at hd 64/128), and the packed gate's
  `MINFER_NO_KV_STAGE=1` arm still requires **bit identity** — the double buffer changes *when* a copy
  lands, so the output must not move, and it does not.
- nsys per-launch medians, f16 arm, `pp2048`, two runs each side: hd 128 **418.6 / 413.2 → 418.4**
  (the unshipped hd-128 variant measured 479.6), hd 64 **172.9 / 170.2 → 165.6 / 165.5**.
- `pp2048` end-to-end (3 alternating rounds, two binaries): hd 64 f16 11 547.8 → 11 761.1 and
  11 868.9 → 11 741.1 across the two sessions — i.e. **inside the session noise**, which is the
  honest shape of a 0.1% prompt effect.

## 5. Results

| cell | before (µs/launch) | after | note |
|---|---|---|---|
| hd 64, f16 FA launch | 172.9 / 170.2 | **165.5 / 165.6** | −4.3%, shipped |
| hd 64, packed staged (launch + pass) | 175.4 / 172.7 | **168.0 / 168.1** | inherits the f16 body |
| hd 128, f16 FA launch | 418.6 / 413.2 | 418.4 (unchanged) | single-buffered by the budget |
| hd 128, *double-buffered variant* | — | **479.6** | +14.6%, measured and **not shipped** |

**Updated by [119](./119-c4-fa-tile-128.md)**: with the 128-row tile the block count becomes
register-bound at one per SM, so the second pair is affordable at **both** head dims and the hd-128
gate this record introduced is gone. The occupancy arithmetic below is what made that visible.

## 6. Lessons

1. **A latency lever can lose to an occupancy lever.** The copy is 8% of the launch and overlapping it
   is worth 4% — but only while the extra buffer does not cost a resident block. The budget arithmetic
   (102,400 B/SM) predicts the sign of the result before the benchmark, and the benchmark confirmed it
   at +14.6%.
2. **Ship the shape where the arithmetic holds.** `NBUF` is a compile-time function of
   (layout, head dim), documented with the block-count table; hd 128 keeps the old path exactly, so the
   change cannot regress the cell that pays most.
3. **The next lever must not cost shared memory.** At hd 128 the kernel is 2 blocks/SM because of its
   smem, not its registers; anything that adds a buffer, a wider tile or a staging plane *per block*
   trades the same currency. The directions left in #865 (two query tiles per block; fewer syncs per
   query row) have to be priced in that currency first.

<- [117 · #862: the staging plane is window-bounded, and a budget guard can refuse it](./117-c4-packed-q8-kv-window-plane.md) · [Index](./index.md)
