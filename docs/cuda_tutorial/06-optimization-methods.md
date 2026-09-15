# 06 · Optimization methods and how to verify them

> **Part**: Part 4 — from reading to changing. **Prereq**: chapters 03–05 (you can
> now read minfer's elementwise, dequant, matmul, attention kernels and the host
> layer that launches them).
> **Code**: `docs/cuda_optimization_steps/` (the evidence base — one document per
> historical optimization step), `src/cuda_kernels.cu` (the kernels each technique
> lives in — verified lines).

Chapters 03–05 taught you to *read* minfer's CUDA backend. This chapter teaches
you to *change* it safely. That skill is two-sided: knowing the small set of
techniques that actually move GPU kernels (a catalog, each anchored to a real
kernel and a real campaign record), and knowing how to *prove* that your change
helped — because on a GPU, "it compiles and prints plausible numbers" is very far
from "it is faster and still correct". Both halves come from minfer's own
history: 12 optimization sessions, ~60 levers tried, every landed and reverted
decision recorded in `docs/cuda_optimization_steps/`.

## 1. Background — where this sits

By now you know the machinery: kernels, warps, blocks, shared memory, the CUDA
Graph replay loop (chapter 05), and the matmul/attention kernel families
(chapter 04). What you do not yet have is judgment: when a kernel is slow,
*which lever do you reach for first*, and *how do you know it worked?*

minfer's answer to both questions is the same artifact: the campaign record in
`docs/cuda_optimization_steps/` (indexed by [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md),
the live-status hub — its §0 master table has one row per lever with the
measured delta and the one-line lesson). Every technique in this chapter's
catalog names the kernel it lives in, the step document(s) that used it, and the
measurement that adjudicated it. Nothing here is folklore; every entry is a link
into the record.

This chapter is the *get-started*. Three deeper documents do the heavy lifting:

- `docs/cuda_optimization_steps/77-verification-methodology.md` — the campaign's
  full evidence protocol (§2 and §4 of this chapter summarize it);
- [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) — the live status of every
  lever (landed / reverted / measured-only) plus the env-gate reference in its
  Appendix A;
- [`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) — the technique reference this
  tutorial is forbidden to duplicate (§10 covers profiling depth, §11 the
  env-gate inventory).

## 2. Principle — profiling: how to see a kernel before you touch it

You cannot optimize what you cannot see. Two tools ship with CUDA and both are
installed on this machine (GB10, CUDA 13.0). They answer different questions:

- **`ncu`** (Nsight Compute, `/usr/local/cuda-13.0/bin/ncu` — not on every
  shell's PATH) profiles *one kernel launch in depth*: throughput percentages,
  occupancy, stall reasons. Use it to answer "what is this kernel's problem?"
- **`nsys`** (Nsight Systems, `/usr/local/bin/nsys` — on PATH) records the
  *timeline of the whole process*: which kernels ran, in what order, how many
  times, and how much wall-clock time sits between them. Use it to answer
  "where does the step's time actually go, and how many launches am I paying
  for?"

**What "SOL %" means.** ncu's headline section is called *GPU Speed Of Light
Throughput* — the fraction of the hardware's theoretical peak the kernel
achieved. Three lines matter most. **Compute (SM) Throughput** is how busy the
arithmetic units (the SM's execution pipes) were as a share of their peak —
low means the math units are idle. **Memory Throughput** is how busy the
narrowest memory pipe (DRAM, L2, or L1 — the max of the three) was — low means
the memory system is idle. **Duration** is the kernel's wall time. A kernel
with *both* numbers low (say both under 30%) is **latency-bound**: threads are
mostly *waiting* (on loads, on barriers, on each other), not computing and not
streaming — and the fix is rarely "more FLOPs", it is more concurrency or
fewer dependencies. A kernel with Memory Throughput ~90%+ is **memory-bound**:
the only wins left are moving fewer bytes. This one-sentence triage — both low
→ latency; memory high → bandwidth; compute high → math — decided most of the
campaign's lever choices, and ncu prints the same advice as an `OPT` note when
it sees the latency case.

### 2.1 The minimal `ncu` command set

Profile a real minfer build (this runs on the actual GPU — commands and
numbers below are from this chapter's authoring session):

```bash
# Build first (chapter 01; details/pitfalls in docs/BUILD.md)
cargo build --release --features cuda

# Profile ONLY the q8_0 decode-MMVQ kernel, 3 launches, basic section set.
# --launch-count stops after N profiled launches; -k takes a kernel-name regex.
sudo -n env LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  /usr/local/cuda-13.0/bin/ncu --set basic --launch-count 3 \
  -k regex:q8_0_p32_q8_mmvq \
  ./target/release/minfer ~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf "hi" -n 4
```

Why `sudo -n env LD_LIBRARY_PATH=...` and not a bare `ncu`? Two GB10 facts the
campaign paid to learn (recorded in
[`77-verification-methodology.md`](../cuda_optimization_steps/77-verification-methodology.md) §2.5):
bare `ncu` fails with `ERR_NVGPUCTRPERM` — the GPU's performance counters are
permission-gated, and this is *expected*, not a broken install; and plain
`sudo` strips environment variables, which can silently make ncu profile a
different (legacy) code path than the one you think you are measuring — so the
env var must be re-injected *through* sudo. Both were verified live for this
chapter: the bare command printed `ERR_NVGPUCTRPERM`; the sudo form worked.
(One more local gotcha observed here: `sudo` ncu creates a root-owned
`/tmp/nvidia/`, after which a plain-user `nsys` fails with
`Failed to create directory "/tmp/nvidia/nsight_systems"` — run `nsys` with
`TMPDIR=/tmp/yourtmp` or fix the directory ownership.)

The kernel lines of the observed output (Qwen3-0.6B Q8_0, 9-token prompt,
decode `q8_0_p32_q8_mmvq` — the kernel at `src/cuda_kernels.cu:8292`):

```text
  q8_0_p32_q8_mmvq(...) (1024, 1, 1)x(256, 1, 1), Context 1, Stream 13, Device 0, CC 12.1
    Section: GPU Speed Of Light Throughput
    SM Frequency                    Ghz         2.14
    Elapsed Cycles                cycle       48,821
    Memory Throughput                 %         8.31
    Duration                         us        22.78
    L1/TEX Cache Throughput           %        11.90
    L2 Cache Throughput               %        14.93
    Compute (SM) Throughput           %         8.31
    Section: Launch Statistics
    Block Size                                     256
    Grid Size                                    1,024
    Registers Per Thread             register/thread  40
```

Read it like this: grid 1,024 blocks × 256 threads on a 48-SM GPU, and *both*
SOL percentages at ~8% — this decode-step kernel (profiled serialized and cold,
on a tiny model) is nowhere near either roofline; it is dominated by latency
and launch-size effects. That is exactly the kind of verdict ncu gives you in
one command. Do **not** quote ncu's Duration as a speed claim, though: ncu
serializes kernels and replays them several times (the output says
`9 passes`), so its per-kernel times are distorted — the campaign's rule is
that **ncu is for structural metrics** (occupancy, sectors, stall shares,
register counts) and **nsys is the wall-clock authority**.

Useful narrower invocations, all seen in the step records:

```bash
# Count only what you need (fast, huge runs stay usable):
ncu --metrics sm__warps_active.avg.per_cycle_active,sm__throughput.avg.pct_of_peak_sustained_elapsed ...
# Export a report file for the UI instead of stdout:
ncu -o /tmp/myreport ...
# Attribute stalls to source lines (PC sampling):
ncu --set full --page source ...
```

### 2.2 The minimal `nsys` round trip

`nsys` in three sentences. **Record**: `nsys profile -t cuda -o /tmp/trace
./target/release/minfer <model> "hi" -n 16` wraps the whole process and writes
a `.nsys-rep` timeline of every kernel launch, memcpy, and gap. **Summarize**:
`nsys stats --report cuda_gpu_kern_sum /tmp/trace.nsys-rep` prints the
per-kernel table — total time, instance count, min/med/max per kernel — which
is where launch-count claims come from. **Read**: sort by total time and the
top rows *are* your optimization priority list.

Observed on this machine (same 0.6B model, `nsys stats
--report cuda_gpu_kern_sum`, top rows quoted as printed):

```text
 Time (%)  Total Time (ns)  Instances  Avg (ns)  Med (ns)  ...  Name
 --------  ---------------  ---------  --------  --------       ----
     51.2       10,285,344        193  53,291.9  50,592.0       void mmq_nt_kernel<...>
     29.0        5,837,152        229  25,489.7  11,648.0       q8_0_f32_matmul(...)
      7.2        1,439,456        113  12,738.5  13,696.0       q8_0_p32_q8_mmvq(...)
      2.0          410,464        223   1,840.6   1,440.0       rms_norm_f32(...)
      1.7          351,232        167   2,103.2   2,048.0       quantize_q8_0_pad40(...)
      1.5          307,360        168   1,829.5   1,824.0       store_kv_f16(...)
```

This is the instrument behind some of the campaign's most-cited numbers: the
D3-8 fusion verdict "`total launches −310 per decode step`" and the D3-5
verdict "`standalone quantize_q8_0_pad40: 4448 → 964 launches per trace`" are
nsys instance counts, not beliefs (step docs
[73](../cuda_optimization_steps/73-d3-8-fusedqkv-port.md) and
[70](../cuda_optimization_steps/70-d3-5-fused-producer-a-quantize.md)). When
you propose a fusion, this table is how you measure what you deleted.

### 2.3 The evidence discipline — the 15-line version

The full protocol is
[`77-verification-methodology.md`](../cuda_optimization_steps/77-verification-methodology.md)
(read it before your first real A/B — §4 of this chapter explains *why* each
gate exists). The tool protocol in compressed form:

1. **ncu needs `sudo -n env LD_LIBRARY_PATH=...`** — plain sudo strips env vars
   and has silently profiled the wrong (legacy) path before (the r56 lesson,
   doc 56).
2. **GB10 has no `dram__*`, `launch__grid_size`, or shared-sector counters** —
   bandwidth rooflines are derived from `lts__t_sectors_aperture_device`
   (L2 sector count × 32 B per sector) plus analytic byte counts (r55).
3. **ncu serializes; nsys is the wall-clock authority** — use ncu for
   occupancy/sectors/stall structure only, never for headline durations.
4. **PC-sampling attributes a stall to the *consumer*** instruction that is
   waiting, not the producer that is slow (r20/r43) — do not read causality
   backwards.
5. **SASS first**: `cuobjdump -sass ./binary | grep LDGSTS` before writing a
   cp.async lever — verify the compiler actually emitted the instruction you
   are betting on (r45; ptxas `-Xptxas -v` gives the register/spill budget).
6. **A/B means same-binary, env-gated, interleaved** — flip a `MINFER_*` flag,
   never rebuild between sides (§4 of this chapter).
7. **Counters + SASS + a reductio must agree** before a mechanism claim is
   allowed to stand (the campaign's counter-forensics rule, doc 18).

Items 1–5 are [`CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) §10's territory at
reference depth; item 6 is §11's.

## 3. In minfer's code — the technique catalog

Ten techniques cover nearly everything the campaign did. Each entry follows the
same fixed shape: the principle in three sentences, where it lives in minfer's
kernels (file:line, verified on the current tree), which step record(s) used
it, and the one measurement you would run. Links go to the step documents;
[`CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) §0 is the cross-reference index if
you want the surrounding session.

### 3.1 Memory coalescing

A warp's 32 lanes issue one memory instruction together, and the hardware
coalesces (merges) those per-lane addresses into as few 32-byte sectors as the
pattern allows: *consecutive lanes touching consecutive addresses* is one
transaction per warp, while a strided or scattered pattern splits into one
transaction per lane. Uncoalesced access therefore multiplies memory
transactions without multiplying useful bytes — the classic silent tax. The
first question about any data-parallel kernel is thus "what byte does lane *i*
touch, as a function of i?"

- **Where minfer uses it**: the elementwise family is written warp-dense — e.g.
  `store_kv_f16` maps one lane to four *consecutive* floats (`src +
  t*nkt + j`, `src/cuda_kernels.cu:2550`, the `float4` at :2561), and the MMVQ
  decode kernels' shape gate explicitly protects against the uncoalesced case —
  doc 06 records that small shapes lose because "1–2 units per thread expose
  the uncoalesced q5/q6 byte loads" (`src/cuda_kernels.cu:1254` and the
  dispatch comment recorded at `src/cuda.rs`).
- **Step records**: [11-p5-gemm-tiles-fa-rewrite.md](../cuda_optimization_steps/11-p5-gemm-tiles-fa-rewrite.md)
  (P5·1: the 1-element-per-thread elementwise kernels "left 15/16 of every
  transaction unused"; vectorizing to 4/8 elements per lane: 1435 → 1493 tok/s,
  +4%) and, as the cautionary tale,
  [26-r21-coalesced-block-linear-a.md](../cuda_optimization_steps/26-r21-coalesced-block-linear-a.md)
  (making A-staging *perfectly* coalesced cut sectors −28.6% yet moved the wall
  −2.3% — sectors were not the binding constraint; the lesson "stall mass is
  conserved" lives there).
- **How you'd measure it**: ncu's memory tables — sectors per request
  (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` ÷ instruction count; 1.0-ish
  for dense float loads, 4–32 for strided ones) or simply the L1/TEX
  Throughput % in the SOL section; A/B via the P5·1 pattern (same kernel,
  scalar vs vectorized body).

### 3.2 Shared-memory tiling (chapter 04's GEMM)

DRAM is far too slow to feed tensor cores from directly, so a GEMM block first
stages the tile of A and B it needs into shared memory — the on-chip, per-block
scratch — and every warp then reads its fragments from there many times. The
tile size is the whole trade: bigger tiles cut how often each weight byte is
re-read from DRAM/L2 but eat shared memory, and shared memory per block
*limits how many blocks fit per SM* (occupancy, §3.4). Chapter 04 walked
`gemm_f16_nt_kernel_t` line by line; the design arithmetic is in step doc 02.

- **Where minfer uses it**: `gemm_f16_nt_kernel_t`
  (`src/cuda_kernels.cu:4787`) — TN=64 × TM tile, KS=32 k-step, dynamic smem
  (`extern __shared__` at :4796); the MMQ GEMM family tiles the same way with
  raw quantized bytes (`mmq_raw_nb_kernel`, `src/cuda_kernels.cu:6376`; its BT
  successor `mmq_raw_nb_bt_kernel` :6656; q6_K variant :6976).
- **Step records**: [02-wmma-f16-prefill-gemm-8m.md](../cuda_optimization_steps/02-wmma-f16-prefill-gemm-8m.md)
  (the 64×64×32 tile turned prefill from 30.7 → 1204 tok/s, 39×, by cutting
  weight re-reads from nt× to ~1×; also records the two *negative* tile
  experiments: KS=64 −38% and TM=256 −3%) and
  [11-p5-gemm-tiles-fa-rewrite.md](../cuda_optimization_steps/11-p5-gemm-tiles-fa-rewrite.md)
  (TM=128, +30% — "halves B-panel L2 re-reads and barriers per FLOP").
- **How you'd measure it**: whole-prefill tok/s A/B (the tile changes bytes
  moved per FLOP — `lts__t_sectors` per GMAC before/after) plus ncu occupancy
  (§3.4) because the smem budget is what occupancy pays with.

### 3.3 Vectorized loads (`float4` / `uint4`)

A 32-bit scalar load moves 4 bytes with a full instruction and a full
transaction slot; a 128-bit `float4`/`uint4` load moves 16 bytes in one
instruction. Vectorizing turns N small loads into N/4 wide ones — fewer
instructions, fewer sectors, and wider (LDG.128) transactions — *but only when
the address is 16-byte aligned and the data layout actually puts 16 useful
bytes together*. When it applies it is one of the cheapest levers there is;
when the layout does not cooperate it is a parity bug factory (nibble offsets,
alignment).

- **Where minfer uses it**: `store_kv_f16` (`float4` load + two `__half2`
  stores, `src/cuda_kernels.cu:2561`); the q6_K B-expand reads packed data as
  `uint4` groups; the q8_0 p32 decode planes are *designed around* the
  `uint4*` row pointer (`q8_0_p32_q8_mmvq`, `src/cuda_kernels.cu:8292`, row
  pointer :8302, the `__ldg` group loads :8308).
- **Step records**: [11-p5-gemm-tiles-fa-rewrite.md](../cuda_optimization_steps/11-p5-gemm-tiles-fa-rewrite.md)
  (P5·1, +4%); [44-r41-q6k-bexpand-uint4.md](../cuda_optimization_steps/44-r41-q6k-bexpand-uint4.md)
  (widening 32 per-byte loads to `uint4` groups: q6_K GEMM kernel −61.5%, the
  `long_scoreboard` stall share 85.5% → 33.6%, whole prefill +30.7% — the
  single biggest q6_K lever);
  [79-phase8-coverage-batch.md](../cuda_optimization_steps/79-phase8-coverage-batch.md)
  (8q: Q5_0's 22-byte block makes its `qh` word *not* 4-byte-aligned — two
  `u16` loads instead of one `u32`, which eliminated the CPU fallback);
  [76-d4-4-dpl-q6k-final.md](../cuda_optimization_steps/76-d4-4-dpl-q6k-final.md)
  and [104-q80-p32-split-plane.md](../cuda_optimization_steps/104-q80-p32-split-plane.md)
  (repacking planes so that `uint4` loads align — layout work *for*
  vectorization).
- **How you'd measure it**: ncu `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum`
  and the long_scoreboard stall share (`smsp__average_warps_issue_stalled_long_scoreboard...`)
  before/after — doc 44's headline is exactly those two counters — plus the
  same-binary A/B.

### 3.4 Occupancy & block-size tuning

Occupancy is how many warps are resident per SM, and resident warps are the
*latency cover*: when one warp stalls on a load, the SM schedules another.
Per-SM residency is capped by three budgets — threads, registers, and shared
memory — so "make the kernel more comfortable" (bigger smem, more registers)
can directly *cost* performance by evicting a resident block, and "spend a
little" (a spill, a smaller staging buffer) can *buy* a whole block back. The
campaign's occupancy ladder is the best-documented arc in the record: r28
bought a 2nd block by shrinking smem, r39/40 bought a 3rd by spending
registers.

- **Where minfer uses it**: `__launch_bounds__(256, 3)` on the q6_K BT GEMM
  (`mmq_raw_nb_bt_q6k_kernel`, `src/cuda_kernels.cu:6976`) — the compiler
  limit of 80 regs/thread to fit 3 blocks/SM (80 × 768 threads = 61,440 ≤ the
  65,536-register file, vs 87 regs → only 2 blocks); the q4_K NB kernel's
  45,056 B smem budget → 2 blocks/SM (`mmq_raw_nb_kernel`,
  `src/cuda_kernels.cu:6376`, the r28 kernel doc 31 measures); the
  MMVQ family's `__launch_bounds__(256)` everywhere.
- **Step records**: [31-r28-nb-kernel-2blocks.md](../cuda_optimization_steps/31-r28-nb-kernel-2blocks.md)
  (smem 45,056 B ⇒ 2 blocks/SM, +2.56%; ncu
  `sm__warps_active.avg.per_cycle_active` 16.17 confirms residency),
  [43-r40-third-resident-block.md](../cuda_optimization_steps/43-r40-third-resident-block.md)
  (the register-arithmetic table; +13.0% from one line; also falsified the
  "0 spill" dogma — 4 B of spill was immaterial),
  [42-r39-q6k-kdr2-double-buffer.md](../cuda_optimization_steps/42-r39-q6k-kdr2-double-buffer.md)
  (+13.3% at 2 blocks/SM), and the negatives:
  [11-p5-gemm-tiles-fa-rewrite.md](../cuda_optimization_steps/11-p5-gemm-tiles-fa-rewrite.md)
  (KS=64: depth traded for occupancy, −38%),
  [53-r50-fa-tkv-16.md](../cuda_optimization_steps/53-r50-fa-tkv-16.md)
  (occupancy ↑ cancelled by per-tile overhead — reverted),
  [67-d3-14b-attribution-bitwise-mmvq.md](../cuda_optimization_steps/67-d3-14b-attribution-bitwise-mmvq.md)
  (block-size right-sizing measured neutral at 14B shapes).
- **How you'd measure it**: ncu's Occupancy section (Theoretical vs Achieved
  Occupancy, Block Limit Registers/Shared Mem/Warps — which budget is the
  binding one) + `ptxas -Xptxas -v` for the register/spill numbers before you
  build; A/B on the whole-prefill bar (+1.5%).

### 3.5 Warp divergence in quantized kernels

A warp executes *one instruction at a time for all 32 lanes*; when lanes take
different sides of a data-dependent branch, the two sides run serially and the
warp pays for the union of both paths (divergence). In quantized kernels the
danger looks like a per-element `switch` on block type or a per-nibble edge
case; the structural cure is to make branch conditions *warp-uniform* (every
lane of the warp takes the same side) — or to move the choice out of the
kernel entirely. minfer does both: quantization types are dispatched as
*separate per-type kernels* (the type is a template/dispatch-level constant,
never a runtime branch inside the loop), and where a fused producer reduces
across a warp it explicitly documents the uniformity invariant.

- **Where minfer uses it**: the per-type kernel families
  (`q4_k_q8_mmvq`/`q5_k`/`q6_k…` at `src/cuda_kernels.cu:1254/:1391/:1339`)
  mean the hot loop never asks "which type am I?"; the fused rms+quantize
  producer quantizes "THIS warp's row (warp-uniform row ⇒ the shfl_xor
  reductions below never see divergence)" — the comment is at
  `src/cuda_kernels.cu:1125`; the dequant kernels
  (`dequant_q4_0_f16`, :4449) have a single uniform body with a bounds check
  only. Divergence still shows up in the accounting: doc 43 attributes part of
  the gap between achieved and theoretical occupancy to wave-tail divergence,
  and doc 62 rejects a chunk-distribution scheme precisely because "differing
  chunk distributions across a warp cause divergence".
- **Step records**: [79-phase8-coverage-batch.md](../cuda_optimization_steps/79-phase8-coverage-batch.md)
  (8e/8e② — per-type kernels and the launch table instead of in-kernel type
  switches), [55-r52-skip-write-mode2.md](../cuda_optimization_steps/55-r52-skip-write-mode2.md)
  (the skip-write quantizer's reduction is arranged so lanes never diverge),
  [62-r59-q4k-wdsc-plane.md](../cuda_optimization_steps/62-r59-q4k-wdsc-plane.md)
  (divergence as a design *veto*).
- **How you'd measure it**: ncu's Scheduler Statistics — the
  "Warp Cycles Per Issued Instruction" and branch-efficiency metrics
  (`smsp__sass_average_data_branch_divergence`...) — plus the SASS census
  approach of doc 30 when in doubt.

### 3.6 Kernel fusion (`attn_bias_rope_store_f32`, `Op::FusedQKV`/`FusedFFN`)

Every kernel launch has a fixed price — launch latency, the input/output
round trip through memory, the scheduler gap between kernels. Fusing merges a
chain of small kernels into one so intermediate values stay in registers or
shared memory and N launches become 1. On minfer's decode path the fusion
targets were chosen by counting launches with nsys: the per-layer chain
add_bias×3 + rope×2 + store_kv×2 (7 launches) became one kernel, and nsys
counted the difference.

- **Where minfer uses it**: `attn_bias_rope_store_f32`
  (`src/cuda_kernels.cu:2590`) — one launch replaces the 7-launch decode tail;
  the graph ops `Op::FusedQKV` (concat matmul + fused epilogue) and
  `Op::FusedFFN` (gate|up concat matmul + in-place swiglu) are declared in
  `src/graph/ops.rs:125/:144`, executed in `src/graph/cuda_backend.rs:843/:715`,
  and gated at build time in `src/models/qwen2/graph.rs:466–473`. The decode
  A-quantize fusion (`swiglu_quant_pad40`, `rms_norm_quant_pad40`,
  `src/cuda_kernels.cu:2455/:2330`) writes the quantized activation plane
  beside the f32 output so the following matmul skips a standalone quantize
  launch.
- **Step records**: [73-d3-8-fusedqkv-port.md](../cuda_optimization_steps/73-d3-8-fusedqkv-port.md)
  (D3-8: −310 launches/decode-step, +1.63% @14B — and note the fusion is
  **bit-identical** because each piece is verbatim the unfused math),
  [70-d3-5-fused-producer-a-quantize.md](../cuda_optimization_steps/70-d3-5-fused-producer-a-quantize.md)
  (D3-5: standalone quantize 4448 → 964 launches per trace),
  [72-d3-7-attnv-mmvq-rms.md](../cuda_optimization_steps/72-d3-7-attnv-mmvq-rms.md)
  (2c: `f32_bits_to_i32` 239.6 → 1.2 launches/step via one-execution-window
  memoization — fusion's sibling, *caching*).
- **How you'd measure it**: `nsys stats --report cuda_gpu_kern_sum` instance
  counts before/after (launches deleted are the point), then the A/B gate:
  `MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` flip the same binary to the
  unfused topology (they are part of the graph-reuse identity — the rebuild is
  forced for you; `src/graph/cache.rs:136`).

### 3.7 CUDA Graph launch amortization (`MINFER_NO_CUDA_GRAPH=1`)

A decode step is hundreds of tiny kernels; each launch costs microseconds of
host + driver work. CUDA Graph records the whole kernel sequence once
(capture) and replays it with a single call — the kernels are identical, only
the launch overhead is amortized. minfer captures the whole decode step
(Phase 7d) and repeatedly-identical prefills (R3-B, after a 3-run
"will this repeat?" protocol — one-shot CLI prefills never capture, and r55
measured that capture would be a pure loss for them plus a capture-illegal
mid-window malloc).

- **Where minfer uses it**: `src/graph/cuda_backend.rs:105` reads
  `MINFER_NO_CUDA_GRAPH` (replay off → per-kernel eager launches); the
  capture/replay machinery and its pool-generation invalidation live in that
  file's `CudaBackend` (chapter 05 walked it).
- **Step records**: [01-phase7-cuda-backend.md](../cuda_optimization_steps/01-phase7-cuda-backend.md)
  (7d decode capture/replay), [07-r3-small-model-overhead.md](../cuda_optimization_steps/07-r3-small-model-overhead.md)
  (R3-B prefill capture default-on; the 3-run protocol rationale; also A1/A2 —
  the non-graph overheads a timeline exposes),
  [58-r55-swiglu-roofline-prefill-graph.md](../cuda_optimization_steps/58-r55-swiglu-roofline-prefill-graph.md)
  (the measured *skip* for one-shot prefills).
- **How you'd measure it**: the env-gate A/B. Measured for this chapter on
  Qwen3-0.6B Q8_0, `bench -p 256 -n 32 -r 2`, two interleaved pairs
  (author's run, 2026-09-14, quiet box): graph on tg32 **224.45 / 223.29**
  tok/s vs graph off **183.31 / 181.00** — ≈ **+23%** decode, 2/2 pairs with
  clean separation; pp256 7510/7411 vs 7319/7305 (~+2%). The cited
  campaign-side number for the same gate at 14B scale is in doc 88
  ([88-d5-r-stage5-final-battery.md](../cuda_optimization_steps/88-d5-r-stage5-final-battery.md)):
  "48.08 captured vs 50.15 eager" for the prefill-capture cell.

### 3.8 Async copy / double buffering (`cp.async`, KDR=2)

`cp.async` (SASS: `LDGSTS`) copies global→shared *asynchronously*: the warp
issues the copy for the *next* tile and keeps computing the current one, with
`commit_group`/`wait_group N` as the pacing barrier. Double buffering gives
the pipeline two buffers per plane (fill one while computing the other), which
turns the staging latency from a wall into overlap — *if* the kernel is
actually staging-latency-bound, and *if* the extra smem does not cost
occupancy (§3.4's trap: r39 notes doubling every plane at KDR=4 is exactly the
1-block/SM mistake).

- **Where minfer uses it**: the q6_K BT GEMM stages every per-kt plane twice
  ("r39: DOUBLE-BUFFERED staging — two copies of every per-kt plane so kt+1's
  global→smem expansion overlaps kt's compute", comment at
  `src/cuda_kernels.cu:6986–6988`); the A/B/dsc staging planes ride `cp.async`
  (the r53/r56 bundles); `gemm_f16_nt_kernel_t` double-buffers its A/B panels
  (doc 02 §2.4).
- **Step records** (this technique has both spectacular wins and instructive
  nulls): [02-wmma-f16-prefill-gemm-8m.md](../cuda_optimization_steps/02-wmma-f16-prefill-gemm-8m.md)
  (8m② cp.async: 31 → 35 TFLOPS),
  [42-r39-q6k-kdr2-double-buffer.md](../cuda_optimization_steps/42-r39-q6k-kdr2-double-buffer.md)
  (+13.3%), [56-r53-q6k-wexp-cpasync-bundle.md](../cuda_optimization_steps/56-r53-q6k-wexp-cpasync-bundle.md)
  (+5.03% — and the "nearly landed silently" liveness lesson),
  [59-r56-q6k-a-cpasync-wdsc.md](../cuda_optimization_steps/59-r56-q6k-a-cpasync-wdsc.md)
  (+2.35%, LDGSTS verified in SASS); the negatives:
  [17-staging-shape-family.md](../cuda_optimization_steps/17-staging-shape-family.md)
  (cp.async-db neutral — the kernel was L2-throughput-bound, not
  MLP-starved), [61-r58-q4k-bt-cpasync-transplant.md](../cuda_optimization_steps/61-r58-q4k-bt-cpasync-transplant.md)
  (−12.6%: the mechanism's cost scales with what it replaces),
  [66-d2-kv-register-staging.md](../cuda_optimization_steps/66-d2-kv-register-staging.md)
  (three cp.async attention pipelines all slower than register staging), and
  [98-bt-cpasync-null.md](../cuda_optimization_steps/98-bt-cpasync-null.md)
  (the final null that closed the staging line on GB10).
- **How you'd measure it**: `cuobjdump -sass | grep LDGSTS` (is the async copy
  even emitted?), ncu long_scoreboard share before/after, kernel µs from nsys
  (not ncu), and the wall A/B with the +1.5% bar.

### 3.9 f16 KV storage

KV cache rows are read once per decode step per head, every step, for the
whole context; storing them as f16 instead of f32 halves those bytes while
attention math stays f32-accumulated. The win grows linearly with context, and
the cost is none at the byte level — f16 KV is a pure traffic cut — but it is
a *policy* decision (which tensors convert, who else reads the region), so it
is gated and load-time-decided.

- **Where minfer uses it**: `store_kv_f16` (`src/cuda_kernels.cu:2550`) and
  the f16-KV attention mirror `gqa_attn_f32_f16kv` (:2677); policy in
  `src/cuda.rs` (`kv_cache_is_f16`): auto-f16 when `n_layers × n_kv_embd ≥
  8192`, `MINFER_CACHE_TYPE=f16|f32` override.
- **Step records**: [79-phase8-coverage-batch.md](../cuda_optimization_steps/79-phase8-coverage-batch.md)
  (8b: 7B @2K decode +11%; the caveat on record — `MINFER_GRAPH_DUMP` reads KV
  as f32, so dump and f16-KV are incompatible on the debug path).
- **How you'd measure it**: decode tok/s A/B across context lengths (the delta
  widens with KV size — doc 79's framing), or nsys attention-kernel time at
  two context lengths; `MINFER_CACHE_TYPE=f32` is the A/B side of the same
  binary.

### 3.10 int8 MMQ prefill (weights quantized, activations quantized on the fly)

The f16 route (§3.2) dequantizes weights once and runs one f16 GEMM; the MMQ
route keeps weights in their quantized bytes end-to-end and quantizes the
activations to int8 (`q8_0`) in a prepass, so the GEMM streams ~4× fewer
weight bytes and multiplies on int8 tensor cores (`mma.m16n8k32`). That is
what made minfer's prefill reach llama.cpp parity (1.080× at r59b) — and it is
the single deepest line in the campaign: the design was reverse-engineered
from llama.cpp (the MMQ analysis doc), then re-derived kernel by kernel over
~30 rounds.

- **Where minfer uses it**: the A-plane prepass
  (`quantize_q8_0_pad40_t`, `src/cuda_kernels.cu:794` — the pre-transposed,
  64-token-blocked layout), the raw-byte NB/BT GEMM family
  (`mmq_raw_nb_bt_kernel` :6656, q6_K variant :6976), dispatched for
  `nt ≥ 16` under the `MINFER_MMQ` gate read through `CudaState::mmq_gate_on`
  (`cuda.rs:2992`).
- **Step records**: [08-r1-int8-mmq-prefill-gemm.md](../cuda_optimization_steps/08-r1-int8-mmq-prefill-gemm.md)
  (R1: parity-first strategy — "parity-clean but ~2.9 TMAC/s vs llama ~24: the
  8× gap was unprofiled"), the r9→r59 redesign ladder
  ([37-r34-quantize-transpose-prepass.md](../cuda_optimization_steps/37-r34-quantize-transpose-prepass.md)
  +9.72% layout-transform locality;
  [44-r41-q6k-bexpand-uint4.md](../cuda_optimization_steps/44-r41-q6k-bexpand-uint4.md)
  +30.7%; [62-r59-q4k-wdsc-plane.md](../cuda_optimization_steps/62-r59-q4k-wdsc-plane.md)
  +11.1% W_dsc plane; [63-r59b-clean-remeasure.md](../cuda_optimization_steps/63-r59b-clean-remeasure.md)
  the baseline-correction that produced the definitive 1.080×),
  [64-r60-promotion-default-on.md](../cuda_optimization_steps/64-r60-promotion-default-on.md)
  (the verified gate set flipped default-on). The llama.cpp side of the story —
  what their MMQ does, instruction for instruction — is
  [`docs/LLAMA-CPP-MMQ-ANALYSIS.md`](../LLAMA-CPP-MMQ-ANALYSIS.md) (§11 mirrors the
  redesign rounds one for one).
- **How you'd measure it**: whole-prefill tok/s interleaved A/B with
  `MINFER_MMQ=0` as the f16-escape side of the same binary (Appendix A of
  [`CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) carries the full gate list and the
  memory cost of each plane), plus `lts__t_sectors` per GMAC for the
  byte-stream claim.

### 3.11 Honesty footer — what this campaign did *not* use

- **Persistent kernels** (blocks that loop over tiles instead of one-tile-per-
  block): tried once — r24's scheduling ladder — measured −3.3% and reverted
  ([29-r24-scheduling-ladder.md](../cuda_optimization_steps/29-r24-scheduling-ladder.md));
  no wave-quantization tail existed to remove on these shapes.
- **TMA (Tensor Memory Accelerator) and cluster launch**: no step record
  mentions them; the campaign's staging levers were all cp.async/smem/L2, and
  nothing in the current kernels needs hardware-managed tensor tile movement.
  They are future-hardware levers, not GB10 campaign levers.
- **PDL (programmatic dependent launch)** — the closest relative of the above
  that *was* fully integrated: measured −2.6%/−1.8% in-situ and reverted, with
  a standing warning against co-residency tricks on this workload
  ([76-d4-4-dpl-q6k-final.md](../cuda_optimization_steps/76-d4-4-dpl-q6k-final.md);
  [`CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) §11's PDL note).

## 4. The verification discipline — why the gates exist

Everything in §3 came with a measurement, and the measurements all share one
skeleton — the campaign's gate chain, written up in full in
[`77-verification-methodology.md`](../cuda_optimization_steps/77-verification-methodology.md).
This section is the why: every gate can state in one sentence *which defect
class it defends* — doc 77's first lesson is that a gate which cannot is
ritual, not verification. The doc's three villain names are **phantom gain**,
**correctness erosion**, and **baseline drift**.

**Gate 1 — the parity trio (numeric correctness).** Three independent test
binaries run before any landing: `cuda_prefill_mmq` (1/0, 8 quant types × 8
shapes against the host reference — defends the quant kernels' numeric path),
`cuda_prefill` (7/0 — the prefill graph end to end), and
`cuda_fa_prefill_attention_parity` (1/0 — the attention kernel). The 1e-3
tolerance is *informative*, not lenient: a nibble-layout error (the classic
quantized-kernel bug) deviates at ~1e0 magnitude while legitimate f32 rounding
noise sits at ~1e-5, so the magnitude of a failure identifies its class.
Catches: wrong unpacking, wrong scale folds, misaligned loads — the "it
produces numbers" class of bug.

**Gate 2 — greedy byte-for-byte identity (end-to-end correctness).** `-n 32
--greedy --seed 42` on a long prompt, pre-change binary vs candidate: the
token stream must be byte-for-byte identical. This defends the class parity
fixtures structurally cannot see — graph-level and memory-level corruption
that happens to miss the fixture shapes: the doc's two real cases are r52's
rms-kernel out-of-bounds write and r58's smem cross-write, both invisible in
kernel tests and both fatal to a 2000-token generation. The known boundary:
*any* change to tile size necessarily regroups float accumulation order, so
for that class the gate is swapped for a calibrated tolerance package (kernel
≤1e-4 on outlier data, the argmax hard gate, the rp=1.0 greedy stream — the
D3a calibration, doc 68). Catches: correctness erosion — the sampling chain
amplifying a numerics change into visible divergence.

**Gate 3 — interleaved A/B medians (performance truth).** 3×/5× pairs, same
binary (env-gated), same time window, alternating which side runs first;
headline numbers require *distribution separation* (min-new > max-base), and
the whole-prefill landing bar is +1.5% against a *re-measured* baseline.
Alternating order is the design core: the GPU is shared, and back-to-back
runs disguise window drift as a trend while paired medians cancel it. Catches:
phantom gain (the fast path never actually taken — hence the liveness-label
rule, r53/r54: an intentional fallback prints `exp=off`, an accidental one
prints `fallback!`) and baseline drift (the r59/r59b story: a "baseline"
binary that was actually a stale deficient build inflated a +26.2% claim down
to its true +11.1%, doc 63).

**Gate 4 — the suite.** The device test suite (166 → 174+ over the campaign)
guards collateral damage; a test that flakes under a co-tenant window is
adjudicated with an isolated `--exact` rerun, never silently retried to green.
Catches: regression in *other* kernels — the change you did not mean to make.

**The rule for a new kernel variant.** A variant is not "done" when it is
faster; it is done when it has (1) the parity trio, (2) its greedy-identity or
calibrated-tolerance verdict, (3) an interleaved same-binary A/B through an
env gate with separation past the bar, and (4) a liveness/label check proving
the fast path actually ran. The env-gate A/B pattern is the load-bearing
habit: **one binary, two sides, flipped by a `MINFER_*` switch** — never
rebuild between A and B, because a rebuild changes the baseline too. The
campaign's full gate inventory (which switch reverts which lever) is
[`CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) §11 and
[`CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) Appendix A. One-line summary of
doc 77 §4's hard-won coda: when both sides of your A/B share the same bug, the
comparison is bit-identical — cross-binary comparisons must anchor on a `-n 1`
first-step dump (D4-2's blind-spot case).

## 5. Try it — three exercises, escalating

Each exercise states the task, what to measure, which step document recorded
the original result, and the expected difficulty. None of them requires
touching the repo (exercise (b) explicitly forbids it): work in `/tmp` copies.

### Exercise (a) — vectorize a toy and A/B it with cudaEvents *(easy)*

**Task.** Take a scalar elementwise kernel from your chapters 01–02 toys (the
vector add or SiLU toy) and write a second version where each thread handles
4 elements via one `float4` load. Time both with `cudaEvent` pairs
(device-side timestamps — the §3.7 A/B discipline in miniature: fresh input,
interleaved reps, report both). A complete, compilable example is below; the
A/B skeleton is the part to keep for your own kernels.

```cuda
// /tmp/vec4_ab.cu — scalar vs float4 SAXPY, timed with cudaEvents (CUDA 13.0, GB10)
#include <cstdio>
#include <cuda_runtime.h>

__global__ void saxpy_scalar(int n, float a, const float* x, float* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];              // 1 element per thread
}

__global__ void saxpy_float4(int n4, float a, const float4* x, float4* y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n4) {                                   // 4 elements per thread
        float4 xv = x[i], yv = y[i];
        yv.x = a * xv.x + yv.x; yv.y = a * xv.y + yv.y;
        yv.z = a * xv.z + yv.z; yv.w = a * xv.w + yv.w;
        y[i] = yv;
    }
}

int main() {
    const int n = 1 << 26;                          // 64 Mi floats = 256 MB
    const size_t bytes = (size_t)n * sizeof(float);
    float *x, *y, *hx = (float*)malloc(bytes), *hy = (float*)malloc(bytes);
    for (int i = 0; i < n; i++) { hx[i] = 1.0f; hy[i] = 2.0f; }
    cudaMalloc(&x, bytes); cudaMalloc(&y, bytes);
    cudaMemcpy(x, hx, bytes, cudaMemcpyHostToDevice);
    const float a = 2.0f, blocks = 1024;

    cudaEvent_t beg, end;                           // events: device-side timestamps
    cudaEventCreate(&beg); cudaEventCreate(&end);
    for (int rep = 0; rep < 2; rep++) {             // 2 timed reps each, interleaved
        float ms_s = 0.f, ms_v = 0.f;
        cudaMemcpy(y, hy, bytes, cudaMemcpyHostToDevice);   // fresh y per run
        cudaEventRecord(beg);
        saxpy_scalar<<<(int)((n + blocks - 1) / blocks), (int)blocks>>>(n, a, x, y);
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms_s, beg, end);
        cudaMemcpy(y, hy, bytes, cudaMemcpyHostToDevice);   // fresh y per run
        cudaEventRecord(beg);
        saxpy_float4<<<(int)((n / 4 + blocks - 1) / blocks), (int)blocks>>>(
            n / 4, a, reinterpret_cast<const float4*>(x), reinterpret_cast<float4*>(y));
        cudaEventRecord(end); cudaEventSynchronize(end);
        cudaEventElapsedTime(&ms_v, beg, end);
        // both kernels move 3 x bytes (x read, y read, y write): GB/s = 3*bytes/ms
        printf("rep %d: scalar %.3f ms (%.0f GB/s)  float4 %.3f ms (%.0f GB/s)\n",
               rep, ms_s, 3.0 * bytes / ms_s / 1e6, ms_v, 3.0 * bytes / ms_v / 1e6);
    }
    float maxerr = 0.f;
    cudaMemcpy(hy, y, bytes, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) maxerr = fmaxf(maxerr, fabsf(hy[i] - 4.0f));
    printf("max |y - 4| = %g\n", maxerr);
    cudaEventDestroy(beg); cudaEventDestroy(end);
    cudaFree(x); cudaFree(y); free(hx); free(hy);
    return 0;
}
```

```bash
/usr/local/cuda/bin/nvcc -O2 -arch=sm_121 /tmp/vec4_ab.cu -o /tmp/vec4_ab && /tmp/vec4_ab
```

Observed on the GB10 (author's run; both kernels stream 3×256 MB — read x,
read y, write y):

```text
rep 0: scalar 3.539 ms (228 GB/s)  float4 3.221 ms (250 GB/s)
rep 1: scalar 3.173 ms (254 GB/s)  float4 3.205 ms (251 GB/s)
max |y - 4| = 0
```

**What to measure, and the honest trap.** The expected lesson is *not* "float4
wins" — this access is already dense and coalesced, every byte of every sector
is useful, so vectorization mostly removes instructions and the time is a
wash inside noise. That is the point: §3.3 pays when the scalar version
*wastes* sectors (strided, or 1 useful byte per 32-byte sector). To see a real
win, make the scalar kernel touch memory with a stride (e.g. one element per
32) and watch both the A/B and ncu's sectors-per-request move. The original
record of this exact effect is
[11-p5-gemm-tiles-fa-rewrite.md](../cuda_optimization_steps/11-p5-gemm-tiles-fa-rewrite.md)
(P5·1: +4% whole-prefill from the 15/16-unused-transaction fix).

### Exercise (b) — change dequant thread granularity in a *copy* *(medium)*

**Task.** Copy `dequant_q4_0_f16` (`src/cuda_kernels.cu:4449`) into
`/tmp/dq_gran.cu` with a synthetic weight buffer — **do not modify the repo**.
The incumbent maps one thread to one 32-element block (`g = blockIdx.x *
blockDim.x + threadIdx.x` indexes `od*nb` blocks; each thread writes 32
halves). Write two variants: (1) one thread per *element pair* (a thread
unpacks one byte's two nibbles); (2) one thread per *four* blocks (64 halves
per thread). Host-verify all three against a scalar CPU dequant
(`max|Δ| == 0`), then profile each with ncu:

```bash
sudo -n env LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64 \
  /usr/local/cuda-13.0/bin/ncu --set basic -k regex:dq_ /tmp/dq_gran
```

**What to measure.** Duration is suggestive only (ncu serializes); compare
Memory Throughput %, L1/TEX Throughput %, and the Occupancy section's
Theoretical Occupancy and Block Limit lines across the three granularities.
The incumbent sits in a real sweet spot — element-parallel dequant is
write-bound and trivially coalesced — so your most likely finding is "the
incumbent was already right", which is itself a result (the campaign's
metadata: 8p's fused-dequant experiments live in
[05-persistent-f16-cache-8p.md](../cuda_optimization_steps/05-persistent-f16-cache-8p.md),
and the aligned/unaligned load subtleties of quant blocks — the Q5_0 22-byte
block — are doc 79's 8q item). Difficulty: medium — the CUDA is easy; the
discipline (bit-exact host check *before* trusting any timing) is the
exercise.

### Exercise (c) — re-run one historical A/B on the current tree *(hard)*

**Task.** Pick one clear-win step doc — good first choices: doc 43
(`__launch_bounds__(256,3)`, one-line code change) or doc 11's P5·2 (TM=128) —
and re-run *its* A/B on the current tree using *its* env gate and protocol:
same benchmark anchor, interleaved pairs, median-vs-median, distribution
separation. For doc 43 the modern equivalent knob is the q6_K kernel's
`__launch_bounds__` line itself (do not modify the repo — build a scratch
worktree copy in `/tmp` if you want to flip it); for P5·2 note the era
shift first: `MINFER_GEMM_TM` (64/128/256, read at
`src/cuda_kernels.cu:5079–5086`) retiles the *f16 wmma GEMM*, which is only
on the hot path when you run the escape side `MINFER_MMQ=0` — exactly the A/B
frame P5·2 was measured in. Then compare your numbers with the doc's recorded
ones.

**What to measure.** Whole-prefill tok/s (or `bench -p/-n` cells for decode),
3+ interleaved pairs, min-new > max-base required before quoting a delta. The
teaching payload is the *meta*-result: your absolute numbers will not match
the doc's (the anchors moved; machine state differs — that is r59b's
baseline-anchoring lesson, doc 63), but a healthy tree should reproduce the
*direction* of every landed lever and the flat/negative direction of every
reverted one. If a landed lever reads negative on your run, suspect your own
protocol first (wrong gate value, no warmup, co-tenant window) — that
suspicion reflex is the chapter's real deliverable. Difficulty: hard — not
because any step is hard, but because this is the first time *you* own the
whole gate chain; doc
[77](../cuda_optimization_steps/77-verification-methodology.md) is the rubric.

## 6. Cross-references

- [**07 — the map, cheat sheet, pitfalls**](07-where-next.md): the reading
  order for every reference doc, the command cheat sheet (including the ncu
  one-liners of §2), and the pitfall list — your next chapter.
- [`docs/cuda_optimization_steps/77-verification-methodology.md`](../cuda_optimization_steps/77-verification-methodology.md)
  — the full evidence protocol this chapter compresses: five gates, GB10 tool
  specifics, and the transferable rules (baseline anchoring, liveness labels,
  roofline-before-code, SASS-first…).
- [`docs/CUDA_OPTIMIZATION.md`](../CUDA_OPTIMIZATION.md) — the live status hub: §0's
  master table (one row per lever, Δ, verdict, lesson), §1 current state +
  wall decomposition, Appendix A env-gate reference.
- [`docs/CUDA-TECH-PRIMER.md`](../CUDA-TECH-PRIMER.md) — §10 for profiling/forensics
  at reference depth, §11 for the complete env-gate inventory; the technique
  reference behind every §3 entry.
- [`docs/LLAMA-CPP-MMQ-ANALYSIS.md`](../LLAMA-CPP-MMQ-ANALYSIS.md) — the llama.cpp
  MMQ dissection behind §3.10 (read it before touching the MMQ GEMM family).
- [`docs/CUDA-BACKEND-DESIGN.md`](../CUDA-BACKEND-DESIGN.md) — the backend design
  record (Phase 7a–7e) behind the launch/capture machinery of §3.7.
- [`docs/GPU_SAFETY.md`](../GPU_SAFETY.md) — read before *changing* any kernel:
  bounded waits, no early returns past barriers, errors not silent fallbacks.
- Chapters 03–05 ([03 · Reading minfer's kernels I](03-kernels-elementwise.md),
  `04-kernels-matmul.md`,
  [05 · Reading minfer's kernels III](05-kernels-attention-host.md)) — the
  kernels this chapter's catalog points into, taught line by line.

← [05 · Reading minfer's kernels III](05-kernels-attention-host.md) · [Index](./README.md) · [07 →](07-where-next.md)
