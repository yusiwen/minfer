# 116 · #858: the packed prefill stages once per layer — the hd-128 cell at 1.021x (LANDED)

> **Result**: hd-128 `pp2048` packed/f16 **1.0349x → 1.0206x** (8 346.0 → 8 463.5 tok/s against
> 8 637.8 f16) by replacing the per-tile dequantizing repeat with one staging pass per layer; the
> packed FA launch falls **552.74 → 446.62 µs** (410.94 µs of f16 body + 35.68 µs of pass, a 19.2%
> cut). The plane is **bit-identical** to the per-tile route. hd 64 gains the same 1.2% (staged
> 11 548.8 vs per-tile 11 410.2, and 0.9982x f16). Decode untouched.
> **Bar**: [#858](https://github.com/yusiwen/minfer/issues/858)'s acceptance (filed 2026-11-25:
> `pp2048` at hd 128 **<=1.03x** f16) with the pre-registered prediction and decision rule in the
> 2026-11-27 comment
> ([#858 comment](https://github.com/yusiwen/minfer/issues/858#issuecomment-7214808814)).
> **MET**: 1.0206x <= 1.03x.
> **Commit**: `<this PR>`. **Date**: 2026-11-27. **Box**: `dgxspark` (aarch64, GB10 sm_121,
> driver 580.178.04, CUDA 13.0, nsight-systems 2025.3.2).

## 1. Background — where things stood

A packed Q8_0 key-value (KV) cache cannot be read as `__half` tiles: a block's 32 quants must be
scaled by the block's f16 scale before they can be a tensor-core operand. The flash-attention (FA)
prefill therefore **stages** each tile — dequantize into shared memory, then run the same QK^T and
P·V — and it does that per *query tile*: one CUDA block owns 64 query rows and streams the whole
window through its 32-key staging tile, so every K/V row is dequantized `ceil(nt / 64)` times per
layer, **32** at `pp2048`.

[#858](https://github.com/yusiwen/minfer/issues/858) was filed on record
[114](./114-c4-packed-q8-kv-hd128-column.md)'s measurement of that: **+44.8%** per packed hd-128
launch (460.56 → 666.78 µs), 18.8 ms of a 251 ms prompt — 8%, against 1.88x fewer KV bytes. It
named three shapes and put "stage once per layer, not per tile" first because its cost was already
measured: the arms differ by exactly the conversion work.

Two things then moved under the ticket, and both are part of its story:

- [#845](https://github.com/yusiwen/minfer/issues/845) put hd 64 on the FA path and made the *head
  dimension a template parameter* (record [115](./115-c4-packed-q8-kv-hd64-prefill.md)). On the
  hd-128 instantiations that turned `hd` into a compile-time constant, which improved the staging
  loops' address arithmetic: the packed hd-128 launch reads **552.74 µs** on today's tree against
  record 114's 666.78 (17.1% faster) and the f16 one **410.94** against 460.56 (10.8%), so the
  *tax* had already fallen from 44.8% to **34.5%** before this step.
- That is why this step's own win is 1.4 pp of prompt time rather than 8: it removes a tax that had
  shrunk, and it is measured against the same-session control that shows what is left.

## 2. Principle — make the repeat a single pass, then hand the FA body an f16 region

**The plane.** The FA kernel's packed instantiation differs from its f16 one in exactly one loop —
the staging. If the layer's window is dequantized **once** into an f16 plane laid out like an f16
KV region (row stride `nk * hd`), the packed path can then run the *f16* instantiation unchanged:
same cells, same strides, same math. That is what this step does. `dequant_kv_q8_0_f16` walks the
window's rows (`bound[nt - 1] + 1`, device-side — positions are data) and writes the halves
`kv8_q8_0` would have written into the tile, so the plane's content is *the tile's content*, and
the two routes must be bit-identical, not merely close.

**Why one pass wins.** The per-tile route reads each packed row once per query tile and converts it
`ceil(nt/64)` times; the plane route reads and converts it once, then the tiles read f16. The
conversion is the expensive half (a block's scale load, eight int8 multiplies and a half
conversion per 8 elements, ~350 GB/s effective), and it is what the tile repeats. At `pp2048` the
pass costs **35.68 µs** per layer — 4.2 M elements × (1 B read + 2 B written) in 35.7 µs, i.e.
memory-bound — against the **141.8 µs** the repeat cost the launch.

**Why not every window mode.** A `kv_map` window resolves its cells through a run list, so a plane
indexed by the *window* position would not be the cell the FA kernel reads; a span window's union
is not worth a second device-side scan. Both keep the per-tile staging (documented, unchanged). And
a window of one query tile (`nt <= FA_TQ = 64`) stages once in the per-tile route too, so the plane
would only add a write and a re-read — the verify band (`2 <= nt <= 16`) and short prompts keep the
per-tile route by design.

**The memory it costs, named.** `arena_rows * nk * hd * 2` bytes per plane, two planes (K, V):
hd 128 at `n_ctx` 4 096 → **8 MiB each, 16 MiB transient**; at `n_ctx` 32 768 → 128 MiB. That is
exactly the size an f16 KV region would have been, it exists only while a prefill runs, and it is
allocated lazily (`get_or_grow`, the `buf_f16_w` precedent) — a failed allocation, or a capture
window, falls back to the per-tile route instead of erroring. Bounding it by the *window* instead of
the arena is the follow-up if memory ever matters more than the 1.4 pp.

## 3. Implementation

- **`dequant_kv_q8_0_f16`** (`attention_prefill.cu`): one 8-element group per thread, K and V
  together, grid sized by the arena and strided, blocks past the window return untouched. Launcher
  `launch_dequant_kv_q8_0_f16` is a documented-fallback site (`minfer_launch_ok_opt`, `-1` on
  failure), and the pre-warm covers the kernel.
- **`CudaState::buf_kv_stage`** — a per-stream `StreamScratch` grown on demand.
- **Dispatch** (`src/cuda/methods/attention.rs`): for a packed, causal, `nt > 64` prefill the
  method stages the window and then calls the launcher with `KV_LAYOUT_F16` over the plane; every
  other packed shape keeps the per-tile instantiation. `kv_rows` (the arena capacity) is passed from
  the backend, which reads it from the KV region's byte length and the cell width.
- **`MINFER_NO_KV_STAGE=1`** is the same-binary control (`EngineConfig::no_kv_stage`, registered in
  `scripts/env-vars.toml` with `docs/CUDA-BACKEND-DESIGN.md` as its home).
- **Gates**: the packed parity gate now asserts `cuda_fa_prefill_q8_0_staged` moved, and — because a
  work removal must not move numbers — re-runs the same node under `MINFER_NO_KV_STAGE=1` and
  requires **bit identity**, with the control arm proven to have taken the other route (the staged
  counter must not move). `#162`'s fixture/driver covers the new site (143 audited launches).

## 4. Verification

- `scripts/cuda_test.sh` (CUDA build, serial, `dgxspark`): **867 passed / 0 failed / 62 ignored**.
- `cuda_q8_0_fa_prefill_attention_parity` at hd 64 and 128: the CPU-reference delta is unchanged by
  the route (hd 64 6.72e-4, hd 128 4.80e-4 — the numbers #845's record reports for the per-tile
  route), and the staged-vs-per-tile comparison is **0 differing elements**.
- `cuda_kv_q8_0_roundtrip_attn` (`nt = 3`) still takes the per-tile route and passes — the `nt > 64`
  gate is why.
- `check_env_vars.py`: 98 production reads, 83 documented (the new knob among them), 15 internal.
- **Honest scope**: one earlier full-suite run on this branch failed five model-load budget checks
  (`out of Cuda memory: … exceeds the … budget`) while the box's 27B sglang server held device
  memory; the same five pass serially and the control tree (`cbd0cc0`, no staging kernel) reads
  867/0/62 on the same box. The planes the tests allocate before those loads are 64-131 KB, so this
  was the shared box, not a leak — but it is the second time this box's budget check has been the
  first casualty of a busy neighbour.

## 5. Results

### 5.1 The bench matrix

`minfer bench -p 2048 -n 128 -r 3 --n-ctx 4096`, three alternating rounds, medians of the per-round
`ts_mean`; f16 = the f16 cache, "staged" = packed with the plane (the default), "per-tile" = packed
with `MINFER_NO_KV_STAGE=1`:

| cell | f16 | packed staged | packed per-tile | staged/f16 | per-tile/f16 |
|---|---|---|---|---|---|
| **hd 128 `pp2048`** | **8 637.8** | **8 463.5** | 8 346.0 | **1.0206x** | 1.0349x |
| hd 128 `tg128` | 152.41 | 151.96 | 152.04 | 1.0030x | 1.0024x |
| **hd 64 `pp2048`** | **11 527.5** | **11 548.8** | 11 410.2 | **0.9982x** | 1.0103x |
| hd 64 `tg128` | 232.71 | 239.14 | 238.96 | 0.9732x | 0.9739x |

The hd-64 row is the control that this change is safe everywhere #845 put it: the plane is a 1.2%
win there too (its tax was only 0.7% of the prompt, so the gain is mostly the same "convert once"
arithmetic), and it does not touch decode.

### 5.2 The kernel trace

nsys medians per launch, hd 128, `pp2048`:

| kernel | per launch | x 28 layers | note |
|---|---|---|---|
| `fa_prefill_kv<1,0,2,128>` (per-tile) | **552.74 µs** | 15.48 ms | the pre-#858 route |
| `fa_prefill_kv<1,0,1,128>` (f16 body over the plane) | **410.94 µs** | 11.51 ms | the same body the f16 cache runs |
| `dequant_kv_q8_0_f16` | **35.68 µs** | 1.00 ms | the once-per-layer pass |
| **staged total** | **446.62 µs** | 12.51 ms | 19.2% under the per-tile launch |

Per prompt that is 3.0 ms of ~235 ms (1.3%), which reconciles with the bench's 1.4%. The residual
packed-vs-f16 gap is now **the pass itself** (35.68 µs, memory-bound at ~350 GB/s effective): the
"dequantize inside the tile maths" option #858 listed second would target exactly that, and it is
now the only packing cost left in the prefill.

## 6. Lessons

1. **A staged pass pays off only above one tile.** The per-tile route stages the window once when
   `nt <= 64`; a plane above that would add a write and a re-read for nothing. The gate is
   `nt > FA_TQ`, not "prefill".
2. **Measure the tax you are removing on the same tree.** Record 114's 44.8%/8% was true when it was
   written; #845's constexpr head dimension had already cut the packed launch by 17%, so the same
   ticket's bar was met by a smaller lever than its premise implied. The per-launch medians are what
   carried over; the prompt-level ratio moved with the tree and the session.
3. **A work removal owes a bitwise gate.** "Same numbers, less work" is a claim a tolerance cannot
   check — only re-running the other route and comparing bit patterns can, and the control arm has
   to prove it took the other route or the comparison is the plane against itself.
4. **Name the memory.** A 16 MiB-128 MiB transient is real even when it is "the f16 region you were
   not using"; naming the size, the lifetime and the fallback is what keeps a scratch from becoming
   an accounting surprise.

<- [115 · #845: the hd-64 prefill takes the FA body — 4.32x, and the packed arm reads 1.010x](./115-c4-packed-q8-kv-hd64-prefill.md) · [Index](./index.md)
