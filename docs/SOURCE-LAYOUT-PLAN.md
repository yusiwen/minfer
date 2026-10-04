# Source layout plan — the runtime, launch and kernel layers

> **Status: decided; Step −1 started 2026-10-04.** Baseline master `6b6d94f`
> (`dgxspark (aarch64, GB10 sm_121)`). This is the plan of record for splitting the four long backend
> files and for the naming convention the crate follows afterwards. Tickets:
> [#261](https://github.com/yusiwen/minfer/issues/261) (umbrella) +
> [#262](https://github.com/yusiwen/minfer/issues/262) `cuda.rs` ·
> [#263](https://github.com/yusiwen/minfer/issues/263) `cuda/kernels/` ·
> [#264](https://github.com/yusiwen/minfer/issues/264) CPU ·
> [#265](https://github.com/yusiwen/minfer/issues/265) Metal (Mac) ·
> [#266](https://github.com/yusiwen/minfer/issues/266) anchor checker ·
> [#267](https://github.com/yusiwen/minfer/issues/267) test files.
> Step −1 lands [#138](https://github.com/yusiwen/minfer/issues/138) and
> [#225](https://github.com/yusiwen/minfer/issues/225) first.

## Status

| step | ticket | state |
|---|---|---|
| −1 #138 + #225 | #138, #225 | **#225 landed** `5386a1c` (PR #268, 7/7 green; the cold first-run row is in §2.4); **#138 in flight** on `feat/138-defer-cross-wait` |
| 0 plan document + conventions | this file | **landed** `9174644` (PR #269, 7/7 green, zero code annotations): this document + `SUMMARY.md` + `AGENTS.md` + `ARCHITECTURE.md` + `BACKENDS.md`, plus 4 of #219's 6 stale claims |
| 1 `src/cuda.rs` → `src/cuda/*.rs` | #262 | blocked on Step −1; blueprint ready (all 126 `impl CudaState` fns mapped to their target file) |
| 2 `src/cuda_kernels.cu` → `src/cuda/kernels/` | #263 (after #266) | not started |
| 3 CPU files | #264 | not started |
| 4 Metal (Mac-local) | #265 | not started |
| 5 close the loop (#225 re-measure, #53 `DeviceMemory`) | — | not started |
| 6 long test files | #267 | **files 1–5 landed** (PRs #275, #277, #278, #279, #280): `graph/cuda_backend/tests.rs` 8,603 → a 106-line parent + 11 `tests/<topic>.rs` (61 tests); `models/qwen2/graph/tests.rs` 3,399 → a 111-line parent + 5 (21); `server/batch/tests.rs` 2,630 → a 356-line parent + 7 (21); `graph/alloc/tests.rs` 1,833 → a 72-line parent + 6 (42); `tooling/tests.rs` 1,669 → a 166-line parent + 7 (14); 4 files left |

Each step appends its dated record here when it lands (gates run, counts, box label).



1. **Device is the first axis, the layer is the second axis inside each device.** The crate keeps its
   per-device modules (`cuda.rs`, `metal.rs`, the CPU trio `kernel.rs`/`quants.rs`/`vec_ops.rs`) and
   each of them is split into its own inner axis. There is **no** top-level `L1/`, `L2/`, `L3/`
   directory tree: the layers only become directories *inside* a device.
2. **The cross-device interface stays flat and singular.** `src/graph/backend.rs` (`Backend`,
   `KvProvider`) plus `src/graph/registry.rs` remain the one device seam — exactly as llama.cpp keeps
   `ggml-backend.cpp` + `ggml-backend-impl.h` + `ggml-backend-reg.cpp` flat beside the per-device
   directories. **No new trait is introduced by this plan.**
3. **A `common` is only allowed to exist when it has a second real implementation.** Interface
   eligibility = at least two real implementations **and** at least two callers using it with the
   same semantics. Today only one candidate qualifies and it already exists in part:
   `allocplan::DeviceMemory` (CUDA answers it; Metal does not — the second implementation is
   [#53](https://github.com/yusiwen/minfer/issues/53)). This rule goes into
   `docs/ARCHITECTURE.md`.
4. **Each backend picks its own inner axis** (this is what llama.cpp actually does — it is *not*
   uniform): CUDA = **kernel family**, Metal = **layer** (`ggml-metal-device.*` → `ggml-metal-ops.cpp`
   → `kernels/`), CPU = **ISA** (`ggml-cpu/arch/{x86,arm,…}`).
5. **Existing file paths are preserved.** `src/cuda.rs` stays the module file and gains
   `src/cuda/<part>.rs` children (the layout already used by `src/cuda/tests.rs`); the same for
   `metal.rs`, `quants.rs`, `vec_ops.rs`, `kernel.rs`. This keeps all `crate::…` paths, the
   `check_dead_code_annotations.py` grandfather keys, the dead-code baseline `file =` fields and the
   documentation's file references valid.
6. **Policy is expressed as pure predicates next to the family they gate** (llama.cpp's
   `ggml_cuda_should_use_mmq` / `_mmvq` / `_mmf` convention), not as a separate policy module and not
   by relocating the decision point. The predicates become pure functions with unit tests that need
   no device.
7. **`src/cuda.rs` keeps its name and its type; the family `impl` blocks become descendants of one
   intermediate parent** (`src/cuda/methods.rs` — **not** `impl.rs`: `impl` is a Rust keyword, so
   `mod impl;` is a syntax error — `expected identifier, found keyword 'impl'`, verified with a
   two-file `rustc` probe on 2026-10-04), so privacy does the work: private fields of
   `CudaState` (defined in `cuda`) and private helper methods (defined in `cuda::impl`) are visible in
   every family file. **The split is therefore a pure move — 0 field-visibility edits, 0
   `pub(super)`** — with exactly one mechanical edit: the 86 `extern "C"` launch declarations get
   `pub(crate)` so the two launch test files keep resolving them through `use super::*;`.

Confirmed execution decisions (2026-10-04): land
[#138](https://github.com/yusiwen/minfer/issues/138) and
[#225](https://github.com/yusiwen/minfer/issues/225) **first** (Step −1); this document lands as a
docs-only PR (Step 0); the seven tickets in §7.4 are filed now.

Non-goals: no behaviour change, no renaming of public items, no new abstraction layer, no Metal work
on a non-Mac box.

## 2. The four layers today

| Layer | CUDA | Metal | CPU |
|---|---|---|---|
| **L1 device/runtime** — context, streams, memory, events, capture, resident weights, device query | `cuda.rs` 1147–1930 + `impl` families A–H | `metal.rs` (`MpsState`, `MetalDevice`, `MpsCommandBuffer`) | — (std threads; `kernel.rs`'s `Pool` is not a device layer) |
| **L2 launch/dispatch** — one thin host wrapper per op | `cuda.rs` families I–R (~3,000 lines) + 86 `extern "C"` declarations | `metal.rs` command-buffer encoding (~1,700 lines) | `kernel.rs` + `quants.rs` + `vec_ops.rs` |
| **L3 kernel sources** | `cuda_kernels.cu` | `metal.metal` | the `*_avx2` bodies and `mod neon_*` inside `quants.rs`/`vec_ops.rs` |
| **L4 graph executor** — Op → backend, buffers, capture replay | `graph/cuda_backend.rs` | `graph/metal_backend.rs` | `graph/cpu_backend.rs` |

L2 is **not** a layer that can be moved away from L1: the CUDA launchers are inherent methods of
`CudaState`, and the Metal ones are methods of `MpsCommandBuffer`. Splitting L1/L2 apart by directory
would be a type refactor, not a file move — that is why the layer axis stays *inside* each device.

## 3. What each backend's second axis is

| Backend | Second axis | Target shape |
|---|---|---|
| CUDA | kernel family | `src/cuda/{ffi_runtime,policy}.rs` + `src/cuda/methods.rs` + `src/cuda/methods/<family>.rs` (L2, Rust); **`src/cuda/kernels/*.cu` + `*.cuh`** (L3 + the C++ half of L2) |
| Metal (Mac round) | layer | `src/metal/{runtime,encode,ops,policy}.rs` (L1/L2) + **`src/metal/kernels/*.metal` + `*.h`** (L3) |
| CPU | ISA | `src/quants/*.rs`, `src/vec_ops/*.rs`, `src/kernel/*.rs` |

**The one rule both device backends share: kernel sources live in `<backend>/kernels/`.** llama.cpp is
*not* uniform here (its CUDA keeps `*.cu`/`*.cuh` flat in `ggml-cuda/`, with only `vendors/` and
`template-instances/` as subdirectories, while its Metal puts shaders in `ggml-metal/kernels/`); this
plan chooses the consistent form the request asked for, and states the rule once.

Two honest wrinkles:

- A CUDA `.cu` holds the kernels **and their host-side launchers** (the C++ half of L2), because a
  launcher must live in the TU that instantiates its kernel (§5). `src/cuda/kernels/` therefore means
  "the CUDA translation units", not "device code only"; the Rust half of L2 is `src/cuda/methods/`.
- **CPU is the exception**: `quants.rs` and `vec_ops.rs` are not device-private layers — they are the
  crate's numeric kernel library (`graph/kvformat.rs` uses
  `quants::quantize_row_q8_0_into`, `graph/cuda_backend.rs` uses `vec_ops::RopeStyle`), so they
  stay where they are and are split by ISA, not into a `kernels/` directory.

## 4. Steps

### Step −1 — land the two colliding tickets first (decided 2026-10-04)

- [#138](https://github.com/yusiwen/minfer/issues/138) (F5 late cross-backend wait). It edits
  `copy_to_host` and consumes `stream_wait_event` / `cudaStreamWaitEvent` — both in `cuda.rs` family G,
  and both are the only two grandfathered bare `allow(dead_code)` sites. Landing it first **removes
  code the split would otherwise move** and deletes two `GRANDFATHERED_BARE` keys plus one
  `docs/dead-code-baseline.toml` entry.
- [#225](https://github.com/yusiwen/minfer/issues/225) (pre-warm cost table). It corrects the same
  measurement the `.cu` split will change (the fatbin's per-module one-time load), so the corrected
  record must exist before Step 2 re-measures it.
- Acceptance: each ticket's own gates; master hard-synced afterwards.

### Step 0 — documentation and hygiene (docs-only PR)

- Land this document as `docs/SOURCE-LAYOUT-PLAN.md`; add it to the `AGENTS.md` docs index and update
  the `AGENTS.md` Layout block.
- Fold the stale facts found while measuring into
  [#219](https://github.com/yusiwen/minfer/issues/219) (it already owns two of them):
  `AGENTS.md:3` `~4400 LOC` (production code is 55,529 lines),
  `inference_e2e_walkthrough/15-cuda-backend.md:4/29/36` line counts,
  `CUDA-BACKEND-DESIGN.md` §"the gates" — `120 sites / 120 / 120` (the count at that revision;
  130 on `6b6d94f`),
  `cuda.rs:3471`'s banner naming the deleted `CudaCommandBuffer`. If #219 is not widened, they become
  ticket 7 in §7.4.
- Add the interface-eligibility rule (§1.3) and the layer definition (§2) to `docs/ARCHITECTURE.md`.
- Acceptance: `check-docs` green (`check_docs_links.py`, `check_status.py --check`, book build).
  No counter in `docs/status.toml` changes (the suite counts do not move in this step).

### Step 1 — `src/cuda.rs` → `src/cuda/*.rs`

- `src/cuda.rs` keeps: the module doc, `AttnWindow`, `pub struct CudaState` (1616–1745, **all fields
  stay private**), the free items (`cuda_error_name`, `cstr_owned`, `layout_of`/`format_of`,
  `concat_rows`, `bind_stream`, the KV-layout constants, …), the ten `#[cfg(test)] mod` declarations,
  and `pub(crate) use` lines.
- `src/cuda/methods.rs` — the 16 methods called across families (`get_or_grow` is called by seven) plus
  the `mod` declarations for the family files. Private here means visible in every family file, so
  **no `pub(super)` is needed anywhere**.
- 18 family files under `src/cuda/methods/`: each holds its `impl CudaState` block **and its own
  `pub(crate)` `extern "C"` launch declarations** (the 86 declarations move with their family; no
  symbol is used by two families).
- `src/cuda/ffi_runtime.rs` — the cudart/driver FFI block and the test-only extern block.
- `src/cuda/policy.rs` — `mmq_gate_on`, `mmq_enabled`, `mmq_active`, `mmq_a_fuse_mode`, the `no_*`
  knobs, `fused_b_on`, `no_w16cache`, `no_prefill_gemm`, `no_fa_prefill`, `plane_budget_ok`,
  `gemm_prewarm_disabled` as pure predicates with unit tests (no device).
- Acceptance: CUDA unit **567 / 0 / 42**; CPU **481 / 0 / 36** on `dgxspark (aarch64, GB10 sm_121)`
  (**479** on the CI runner) + integration **10 / 0 / 6** — the rows [#138](https://github.com/yusiwen/minfer/issues/138)
  moved when it landed (565 → 567, 480 → 481 / 478 → 479);
  `cargo fmt --all --check`; `check_source_layout.py`; `check_dead_code_annotations.py` (its two
  `src/cuda.rs:` keys still valid **because the file keeps its name**); `check_dead_code_oracle.py
  --config {cpu,cuda}` unchanged; real-model gates `FEATURES=cuda scripts/real_model_gates.sh` **42 / 0 ×2**.

### Step 2 — `src/cuda_kernels.cu` → `src/cuda/kernels/` (1 header + guard TU + 19 TUs)

Route (a): launchers move with the kernels they launch (llama.cpp's CUDA shape). No `-rdc=true`, no
new nvcc flag (route (b), `-static-global-template-stub=false`, is the recorded fallback; see §5).
**Prerequisite:** ticket 6 in §7.4 (the documentation-anchor checker) lands first or in parallel,
because this step moves 163 line anchors in 23 documents.

**Placement (decided 2026-10-04): `src/cuda/kernels/`**, so that both device backends obey one rule —
kernel sources live in `<backend>/kernels/` (Metal gets `src/metal/kernels/`). The path
`src/cuda_kernels.cu` disappears, so the 342 documentation mentions of it are swept in this step
(they are being swept for anchors anyway).

**Granularity: every file at or below ~800 lines, no kernel split across files.** The previous draft
stopped at "ten family TUs", which left `attention` (~1,600 lines) and `mmq_prefill` (~2,270) too
large. The section inventory measured on `6b6d94f` regroups into:

| file (new) | source sections (pre-split lines) | ≈ lines |
|---|---|---:|
| `kernels/common.cuh` | macros (`Q4B`…`WARP`), `warp_reduce_sum`, `h2f`, `get_scale_min_k4`, `Q8PB`/`MMQ_A_*`, the KV layout + load idiom (2655–2793), **declarations** of the `minfer_launch_*`/`minfer_smem_optin` family | ~450 |
| `kernels/guard.cu` | 5521–5922 — #147 gating + #162 sticky state + the `minfer_launch_*` **definitions** (one owner; external linkage) | 402 |
| `kernels/matmul_f32act.cu` | 63–693 (+ its launchers) | ~750 |
| `kernels/mmvq_aquant.cu` | 694–1094 — fused-producer A-quantize, transposed-A prepass, pad40 producer fusion | 401 |
| `kernels/mmvq_skipwrite.cu` | 1095–1651 — P6 r52 mode-2 skip-write variants | 557 |
| `kernels/mmvq_q6k.cu` | 1652–1942 — pipelined q6_K + dense split-plane | 291 |
| `kernels/ops_misc.cu` | 1943–2375 — padded Q6_K matmul, row gather/embed, f32×f32, f16×f32 | 433 |
| `kernels/ops_elementwise.cu` | 2376–2654 — f32→Q8_0 quantize, RMSNorm, bias, add/mul/SiLU/SwiGLU, i32 decode, RoPE | 279 |
| `kernels/kv_store.cu` | 2794–3084 + 10173–10215 — KV store, fused QKV epilogue (f16 + packed), arena row move | 334 |
| `kernels/attention_decode.cu` | 3085–3677 — GQA f32, E1 window, `kv_map`, split-K, batched split | 593 |
| `kernels/attention_hybrid.cu` | 3678–4047 — hybrid rpw (hd 128, f16 KV) | 370 |
| `kernels/attention_prefill.cu` | 5136–5520 — FA-style prefill (staged KV) | 385 |
| `kernels/gemm_wmma.cu` | 5923–6490 — dequant-to-f16 + wmma HGEMM | 568 |
| `kernels/gemm_smem.cu` | 6491–6859 — prefill-GEMM dynamic smem formula + checked opt-ins | 369 |
| `kernels/gemm_fused_dequant.cu` | 6860–7135 — 8p fused dequant-in-GEMM | 276 |
| `kernels/mmq_int8.cu` | 7136–7540 — R1 int8 MMQ prefill GEMM | 405 |
| `kernels/mmq_raw.cu` | 7541–8074 — P6 raw-byte MMQ | 534 |
| `kernels/mmq_nb.cu` | 8075–8667 — raw-nibble NB + its A-layout transform | 593 |
| `kernels/mmq_bt_q6k.cu` | 8668–9407 — r38 q6_K BT | 740 |
| `kernels/mmvq_multi.cu` | 9408–10172 — multi-token MMVQ + doc103/doc104 decode arms | 765 |

(the `extern "C"` launcher block 4052–5135, 1,086 lines, contributes ~50–150 lines to each file
above; that is why `matmul_f32act` and `attention_*` look slightly over their section size.)

- `minfer_prewarm_kernels` (9188–9237) is decomposed into one `extern "C"
  minfer_prewarm_<family>_kernels()` per file plus a dispatcher that **keeps the symbol name** the
  Rust side declares at `src/cuda.rs:618`.
- **Grouping into PRs (revised): 5–6 groups, not one family each** — the file count grew from 10 to 20,
  so the natural batches are: (1) `common.cuh` + `guard.cu` (the infrastructure), (2) attention (3
  files), (3) the MMQ prefill family (4 files), (4) the MMVQ decode family (4 files), (5) ops/KV/gemm
  (5 files), (6) `matmul_f32act` + leftovers. Each group is independently verifiable on the device.
- **Tooling in the same PRs:**
  - `check_cuda_launch_returns.py` discovers `src/cuda/kernels/*.cu` as a list and audits one file per
    `audit()` call (`_RESOLVE_LINES` is a module global); it must also **assert that no `<<<>>>` lives
    in a `.cuh`** (the invariant that keeps the audit complete).
  - `tests/fixtures/cuda_launch_sites.tsv` is regenerated with a **5th `file` column appended**, so the
    existing column indices stay valid; `--check-fixture`'s
    `[w[1:4] for w in want if len(w) == 4]` becomes a `>= 4` filter whose key includes the file; and
    `src/cuda/issue162_tests.rs:55`'s `assert_eq!(f.len(), 4)` becomes 5 (its only structural use of
    the fixture — lines 228 and 911 read fields 2 and 3 only).
  - `build.rs` compiles the list, emits one `.o` per file, keeps `libcuda_kernels.a`, and adds one
    `rerun-if-changed` **per `.cu` and per `.cuh`** (a header edit that does not trigger a rebuild is
    the silent-stale hazard of this step). It also gains a **new-file guard**: every `.cu`/`.cuh` found
    in `src/cuda/kernels/` must appear in the explicit list and every listed file must exist, so a new
    kernel file cannot silently not compile.
- Acceptance per group: 130-site audit + `--check-fixture`; `MINFER_TEST_ISSUE162=1` device gate;
  CUDA unit **565 / 0 / 42**; real-model gates **42 / 0 ×2** with bitwise-identical greedy output;
  cold-start timing recorded (the fatbin module count changes — see §7 and
  [#225](https://github.com/yusiwen/minfer/issues/225)).

### Step 3 — CPU files

In-place split along the ISA axis that is already there (`src/quants.rs:479 mod neon_kernels`,
`:966 mod neon_q8k`, `src/vec_ops.rs:923 mod neon_f16`, `:1169 mod neon_vec`, plus the inline
`#[cfg(target_arch = "x86_64")] *_avx2` bodies). No path changes. Acceptance: CPU **480 / 0 / 36** +
**10 / 0 / 6**, and the dead-code `(name, kind)` set identical on `aarch64` and `x86_64`.

### Step 4 — Metal (Mac round, after [#255](https://github.com/yusiwen/minfer/issues/255))

`src/metal.rs` and `src/metal.metal` are **not compiled on Linux** (`src/main.rs:31-32` gates the
module; a failed `.metal` compile only warns and writes an empty metallib marker), so this step is
part of [#260](https://github.com/yusiwen/minfer/issues/260) and is verified by the Mac-local gates.

**Shape: `src/metal/{runtime,encode,ops,policy}.rs` + `src/metal/kernels/*.metal` + `*.h`** — the same
rule as CUDA (`<backend>/kernels/` = the shader sources), which is also llama.cpp's Metal shape
(`ggml-metal-device` → `ggml-metal-ops` → `kernels/`, 22 `.metal` + `common.h`/`dequantize.h`/`quantize.h`).

**Granularity: every `.metal` file at or below ~800 lines.** The 5,151-line `metal.metal` regroups as:

| file (new) | source sections (pre-split lines) | ≈ lines |
|---|---|---:|
| `kernels/common.h` | shared macros/preamble | ~80 |
| `kernels/dequantize.h` + `quantize.h` | 595–890 dequant helpers (shared by every GEMM) + the quantize helpers | ~330 |
| `kernels/mul_q4_0_q8_0.metal` | 15–363 — Q4_0×Q8_0 + its prefill | 349 |
| `kernels/mul_f32act_q4q5.metal` | 364–567, 1530–1847 — Q5_1, Q4_0 prefill, Q4_1/Q5_K matmul + prefill | ~500 |
| `kernels/mul_f32act_kquant.metal` | 1848–2339 — Q4_K/Q6_K/Q8_0 matmul + prefill | ~490 |
| `kernels/mul_mm.metal` | 568–1144 — Q4_0/Q4_1/Q8_0 simdgroup GEMM | ~580 |
| `kernels/mul_mm_kq.metal` | 1145–1529 + 4897–5151 — Q5_0/Q5_1/Q6_K/Q4_K/Q5_K simdgroup GEMM | ~640 |
| `kernels/get_rows.metal` | 2340–2508 — embedding lookups, all types | 169 |
| `kernels/norm_elementwise.metal` | 2524–2742 — RMSNorm ×2, add, add-bias, mul, SiLU, SwiGLU | ~220 |
| `kernels/rope.metal` | 2743–2746 + the RoPE kernels | ~60 |
| `kernels/fa_parallel.metal` | 2747–2908 — P1 parallel prefill attention | 162 |
| `kernels/kv.metal` | 2909–3023 — KV store + fused bias/rope/store epilogue | 115 |
| `kernels/qkv_fused.metal` | 3024–3306 — fused decode QKV with per-head Q/K RMSNorm (Qwen3) | 283 |
| `kernels/fa_split.metal` | 3307–3568 — KV-parallel split attention (decode) | 262 |
| `kernels/fa_decode.metal` | 3569–4084 — flash attention decode | 516 |
| `kernels/fa_prefill.metal` | 4085–4896 — flash attention prefill (812 lines — **accepted as one unit**, decision 2 in §9) | 812 |

`build.rs` compiles the parts into one metallib, and the runtime `newLibraryWithSource` fallback
(`metal.rs:2002` `include_str!`) needs the parts joined (`concat!`) or a thin umbrella source; both
entry points must see the same set. Then [#53](https://github.com/yusiwen/minfer/issues/53)
(reserve/assign + the `DeviceMemory` report) lands on top of the new layout.

### Step 5 — close the loop

Re-measure the N-module cold start and update the §2.4 pre-warm table
([#225](https://github.com/yusiwen/minfer/issues/225)); implement `allocplan::DeviceMemory` for Metal
([#53](https://github.com/yusiwen/minfer/issues/53), CUDA already answers it) so that "the device
memory report" becomes the first interface with two real implementations; add the mechanical
documentation-anchor check (§6) if it is not already landed.

### Step 6 — the long test files (in scope, decided 2026-10-04)

The largest files in the crate are **tests**: `src/graph/cuda_backend/tests.rs` (8,384 lines, 143
tests), `src/models/qwen2/graph/tests.rs` (3,374), `src/server/batch/tests.rs` (2,629),
`src/cuda/issue162_tests.rs` (1,186), `src/graph/alloc/tests.rs` (1,773), `src/graph/kvcache/tests.rs`
(1,134), `src/conversation/tests.rs` (1,156), `src/tooling/tests.rs` (1,669), `src/sampler/tests.rs`
(1,232), `src/graph/cuda_backend/tests.rs` … — this step splits them by op family / topic into
`<module>/tests/<topic>.rs` (the same rule: every file named by a `mod` declaration, checked by
`scripts/check_source_layout.py`).

- **Why last**: `src/graph/cuda_backend/tests.rs` is the evidence base for Steps 1–2 (its 143 tests are
  what proves the moves), and every later PR's line references would churn if it moved first.
- **Order inside the step**: `graph/cuda_backend/tests.rs` first (the largest), then the other >1,000-line
  test files, one PR each.
- **Acceptance**: the test counts are **identical** (nothing added or removed — the same tests run from
  new files, which `check_source_layout.py`'s rule 2 is precisely there to guarantee), plus the
  step-appropriate gates (CUDA unit 565/0/42 for the executor tests, CPU 480/0/36 + 10/0/6 for the rest).
- Note: **no size ratchet is added** (decided 2026-10-04) — the ~800-line target in this document is
  guidance, enforced by review, not by a script.

## 5. Why no `-rdc=true`, and why route (a)

Measured on `dgxspark (aarch64, GB10 sm_121)`, CUDA 13.0, 2026-10-03 (two-file probe, four
cross-TU patterns, each compiled and run):

| cross-TU use | default nvcc | `-static-global-template-stub=false` |
|---|---|---|
| plain `__global__` launch | ✅ links and runs | ✅ |
| plain `__global__` address-taken + `cudaFuncSetAttribute` | ✅ | ✅ |
| **templated** `__global__` launch | ❌ link error (`hidden symbol … isn't defined`, nvcc warning #20280-D) | ✅ runs |
| **templated** instance address-taken (the prewarm idiom) | ❌ link error | ✅ `cudaSuccess` |

So the default toolchain forces "the launcher lives in the TU that instantiates the kernel" — which is
route (a) and is also llama.cpp's CUDA shape. The evidence base for the split's other costs:

- one nvcc invocation today, **13 `-gencode` targets** (12 SASS + `compute_121` PTX); serial compile
  **117.3 s / 114.1 s** (two runs), 675 MB RSS, 35.8 MB object; `nvcc --threads 0` **16.4 s / 15.6 s**;
- **the CUDA build is not bit-reproducible today** (two identical serial runs differ by 16 bytes in
  the `.text` of two cubins) — so the split's evidence is runtime gates, not binary identity;
- the recorded per-module fatbin load is **~2.2 ms, set-size independent**
  (`docs/CUDA-BACKEND-DESIGN.md` §2.4's cost table), so ten modules are extrapolated at ~13–22 ms one-time —
  **unverified**, and to be measured in Step 2's first increment.

## 6. Documentation plan

### 6.1 What the split invalidates

| measure | count |
|---|---|
| documents that mention one of the four paths (`src/cuda.rs`, `src/cuda_kernels.cu`, `src/metal.rs`, `src/metal.metal`), measured on `6b6d94f` — the campaign's own documents (this plan, `AGENTS.md`, `ARCHITECTURE.md`, `BACKENDS.md`) have since added mentions | **126** (869 mentions) |
| documents carrying a line anchor **into** one of them (`…:NNN`), measured on `6b6d94f` | **35** (401 anchors: cuda side 269, metal side 132) |
| the anchor hot spots | `docs/cuda_tutorial/*` 180 (6 files), `docs/LLAMA_METAL_E2E.md` 50, `docs/inference_e2e_walkthrough/14-metal-backend.md` 35, `docs/LLAMA-CPP-MMQ-ANALYSIS.md` 18, `docs/METAL-OBJC2-MIGRATION-PLAN.md` 15, `docs/inference_e2e_walkthrough/15-cuda-backend.md` 10, `docs/metal-inference-analysis.md` 10, `docs/ARCHITECTURE-EXECUTION-PLAN.md` 11 |
| documents that describe the **layout** and need rewriting, not sweeping | 18 (listed in §6.3) |
| machine-checked today | `check_docs_links.py` (relative link targets only — it **cannot** see `path:NNN`), `check_status.py --check` (AGENTS.md prose ↔ `docs/status.toml`), `build_book.sh` (mdBook chapters from `docs/SUMMARY.md`) |

### 6.2 Policy: live documents are edited, historical records are frozen

- **Live documents** (the ones a maintainer reads to find code): edited in the step that moves the code,
  with anchors **converted to symbol anchors** (`` `prefill_mmq` (`src/cuda/kernels/mmq_*`) ``) wherever
  the line number was only a locator.
- **Historical records** (`docs/cuda_optimization_steps/*.md`, `docs/QWEN2.5-*.md`,
  `docs/DEBUGGING-*.md`, `docs/KNOWN-CPU-ISSUES-*.md`, `docs/PARAMETER_AUDIT.md`'s older tables) keep
  their text — they record a measurement taken against a revision, and rewriting them would falsify the
  record. They are resolved through the **path mapping table** this document keeps (§6.4).
- **The checker must know about the freeze**: `scripts/check_doc_line_anchors.py` (ticket 6) carries a
  frozen-file set (the `GRANDFATHERED_BARE` pattern), so a frozen record does not fail CI, and the set
  can only shrink. This is the one design constraint the frozen policy puts on ticket 6.

### 6.3 Per-step update table

| Step | Live documents edited (content) | Mechanical sweep (paths + anchors) |
|---|---|---|
| 0 | `docs/SOURCE-LAYOUT-PLAN.md` (new) + `docs/SUMMARY.md` (chapter entry) + `AGENTS.md` (Layout block, docs index, the CUDA/Metal bullets, the `~4400 LOC` figure) + `docs/ARCHITECTURE.md` (module map + the layer/interface-eligibility convention) + `docs/BACKENDS.md` (the device-layer rows) + the stale-number list folded into [#219](https://github.com/yusiwen/minfer/issues/219) | none yet (no file has moved) |
| 1 `cuda.rs` | `docs/inference_e2e_walkthrough/15-cuda-backend.md`, `docs/cuda_tutorial/{02,04,05}.md` (the Rust-side excerpts), `docs/CUDA-BACKEND-DESIGN.md` (§device layer), `docs/DEVICE-ADAPTATION-PLAN.md`, `docs/COMPUTE-GRAPH-DESIGN.md` | the **269-anchor cuda-Rust half** and the 269 mentions of `cuda.rs` across the live set |
| 2 `.cu` | `docs/CUDA-BACKEND-DESIGN.md` (§kernels), `docs/cuda_tutorial/{03,04,05,06}.md`, `docs/LLAMA-CPP-MMQ-ANALYSIS.md`, `docs/CUDA-TECH-PRIMER.md`, `docs/CUDA_OPTIMIZATION.md`, `docs/GPU_SAFETY.md` (the `<<<>>>`/opt-in rules), `docs/BUILD.md` (the nvcc file list) | `cuda_kernels.cu` 342 mentions + its anchors, via the mapping table |
| 3 CPU | `docs/ARCHITECTURE.md`, `docs/inference_e2e_walkthrough/{10,11}.md`, `docs/CPU_OPTIMIZATIONS.md` | `quants.rs`/`vec_ops.rs`/`kernel.rs` mentions (few, all live) |
| 4 Metal | `docs/METAL-BACKEND-DESIGN.md`, `docs/METAL_OPTIMIZATIONS.md`, `docs/inference_e2e_walkthrough/14-metal-backend.md`, `docs/LLAMA_METAL_E2E.md`, `docs/METAL_OBJC2-MIGRATION-PLAN.md`, `docs/metal-inference-analysis.md`, `docs/multi-token-kernel-analysis.md` | the 132 metal anchors + 250 `metal.rs`/`metal.metal` mentions |
| 6 tests | `AGENTS.md` (the test-module convention paragraph), `docs/GATE-CONTRACT.md` if a gate's location is named | test-file paths named in docs |
| every step | a dated entry in `docs/ARCHITECTURE-EXECUTION-PLAN.md` §test-infrastructure (the repo's per-ticket record) + the `Status` table of this document | — |

`docs/status.toml` is **not** edited by Steps 0–6: the suite counts do not move (code moves, tests move,
no test is added or deleted). If a step ever changes a count, `scripts/check_status.py --check` must be
updated in the same PR, and this document says so in that step's record.

### 6.4 The path mapping table (lives here; grows per step)

The frozen records resolve old paths through this table, and the live sweeps are generated from it:

| old | new |
|---|---|
| `src/cuda_kernels.cu` 20–50, 2655–2793, 5521–5922 | `src/cuda/kernels/{common.cuh, guard.cu}` |
| `src/cuda_kernels.cu` 4050–5135 | distributed: each launcher to its kernel's file |
| `src/cuda_kernels.cu` *other ranges* | the §4 Step 2 table (one row per new file) |
| `src/cuda.rs` 1147–1930, 1934–6479 | `src/cuda/{ffi_runtime,policy}.rs` + `src/cuda/methods/*.rs` (the §4 Step 1 family table) |
| `src/metal.rs`, `src/metal.metal` | the §4 Step 4 table |
| `src/graph/cuda_backend/tests.rs` | `src/graph/cuda_backend/tests/{staging,pool,elementwise,matmul,mmvq,prefill,weights,kv,attention,attn_window,capture}.rs` |
| `src/models/qwen2/graph/tests.rs` | `src/models/qwen2/graph/tests/{cuda_kv,offload_copy,kv_reuse,batching,real_model}.rs` |
| `src/server/batch/tests.rs` | `src/server/batch/tests/{kv_sharing,slots,prefill,batching,stall,http,metrics}.rs` |
| `src/graph/alloc/tests.rs` | `src/graph/alloc/tests/{backend_fence,views,liveness,kv_arena,staging,budget}.rs` |
| `src/tooling/tests.rs` | `src/tooling/tests/{parse,f16_encode,f6_roundtrip,f141_device,f167_qwen3,quantize_bounds,bf16}.rs` |

### 6.5 The macOS hand-off (decided 2026-10-04: Step 4 is Mac-local)

Step 4 cannot be executed or verified on the Linux box, so this document must be **sufficient alone** for
a macOS agent: §3's rule, §4 Step 4's file table, §5's cross-TU constraint, §6.3's doc sweep, and §10's
verification row. The ticket (T5) and [#260](https://github.com/yusiwen/minfer/issues/260) both link
here, and Step 4's record names the Mac box explicitly (gate-contract rule 5: an absolute box label,
e.g. `macbook (macOS 15.x, Apple M4)`), never "this box".

## 7. Interaction with the open issues (as of 2026-10-03, 38 open)

Symbol-level scan of all 38 issue bodies against the identifiers defined in the files to be split
(601 distinctive symbols; plus a direct `src/<file>:NNN` path scan). **18 issues reference affected
code or files, or target code that this plan moves.** (#150's `worker_loop` hit is the *server*'s
`worker_loop_serial`, not `kernel.rs` — counted as unaffected.)

### 7.1 Must be sequenced against this plan

| Issue | Why it collides | Action |
|---|---|---|
| [#138](https://github.com/yusiwen/minfer/issues/138) F5 late cross-backend wait | edits `copy_to_host` and consumes `stream_wait_event` / `cudaStreamWaitEvent` — both in `cuda.rs` family G, and both are the two grandfathered bare `allow(dead_code)` sites | **land #138 first if it is next**: it removes code the split would otherwise move and deletes two grandfather keys; otherwise keep it out of flight during Step 1 |
| [#219](https://github.com/yusiwen/minfer/issues/219) two stale CUDA claims | owns `walkthrough/15-cuda-backend.md` §3.2.2 (`register_weight`) and `CUDA-BACKEND-DESIGN.md` — the same files Step 0 and Step 1 re-anchor | merge Step 0's stale-fact list into #219 (one docs PR), or land Step 0 first and reference #219 |
| [#225](https://github.com/yusiwen/minfer/issues/225) pre-warm cost table | the split changes the **fatbin module count**, i.e. exactly what #225 records (2.2 ms per module, cold-run 14.5 ms) | land #225's correction first (cheap), then re-measure in Step 2's first increment and cross-reference |
| [#200](https://github.com/yusiwen/minfer/issues/200) CUDA kernel for `Op::FusedQkvNorm` | **adds** a kernel to `cuda_kernels.cu` and a launcher to `cuda.rs`, in the `attn_bias_rope_store*` (family R) shape | land the split first, or rebase onto the new family files |
| [#208](https://github.com/yusiwen/minfer/issues/208) bf16 weights on device | adds device kernels to `cuda_kernels.cu` (+ `metal.metal`) and touches `vec_ops::mat_mul_bf16` | same as #200 |
| [#212](https://github.com/yusiwen/minfer/issues/212) packed Q8_0 residual attribution | profiles `gqa_attn_f32` in `cuda_kernels.cu` (attention family) with line-level references | finish or park it; if the split lands first, re-anchor its references to `cuda_kernels_attn.cu` |
| [#164](https://github.com/yusiwen/minfer/issues/164) Metal f16 matmul/embedding kernels | adds kernels to `metal.metal` | Mac round; do it after the Metal split (Step 4) |
| [#255](https://github.com/yusiwen/minfer/issues/255) two macOS-only dead-code annotations | its two targets are `metal.rs:971` / `metal.rs:2029` — line anchors the Metal split moves | judge them first (Mac), then split |
| [#260](https://github.com/yusiwen/minfer/issues/260) Mac round umbrella | the entry point for a Mac agent; it lists the Metal gaps and the order | update it with the Step 4 shape and the new #53 item |
| [#53](https://github.com/yusiwen/minfer/issues/53) Metal reserve/assign (+ the device-memory gap) | the only issue that already owns the one interface this plan promotes; its pool code moves in Step 4 | Step 4 first, then #53; #53 supplies the second `DeviceMemory` implementation |
| [#44](https://github.com/yusiwen/minfer/issues/44) Metal KV cell store / explicit span | adds Metal kernels + `copy_cells` work in `metal_backend.rs`/`metal.metal` | Mac round, after Step 4 |
| [#56](https://github.com/yusiwen/minfer/issues/56) AVX2/AVX-512 K-quant dots + repacking | adds kernels to `quants.rs` — Step 3's target file | land after Step 3, or rebase onto `src/quants/*.rs` |

### 7.2 Needs a body/anchor update only

[#137](https://github.com/yusiwen/minfer/issues/137) (async staging, `copy_cross`/`await_cross` +
Metal), [#135](https://github.com/yusiwen/minfer/issues/135) (walkthrough/architecture stale
`Backend` enum — same docs), [#54](https://github.com/yusiwen/minfer/issues/54) (re-run Metal gap
measurements), [#52](https://github.com/yusiwen/minfer/issues/52) (mixed-quant QKV epilogue),
[#39](https://github.com/yusiwen/minfer/issues/39) (`debug_assert!` in release on Metal),
[#231](https://github.com/yusiwen/minfer/issues/231) (five macOS-only attention call sites, 8
`src/…:NNN` references).

### 7.3 Unaffected

#215, #209, #205, #204, #203, #198, #195, #179, #157, #150 (server-side `worker_loop_serial`, not
`kernel.rs`), #133, #132, #126, #125, #118, #103, #62, #40, #38, and #229 (allocator dead-code
bookkeeping only).

### 7.4 Tickets this plan files (decided 2026-10-04: filed now, labelled)

| # | Title | Labels | Step |
|---|---|---|---|
| 1 | **Umbrella**: [#261](https://github.com/yusiwen/minfer/issues/261) `[layout] split the runtime/launch/kernel layers per device` | `enhancement` | all |
| 2 | [#262](https://github.com/yusiwen/minfer/issues/262) `[cuda] split src/cuda.rs into src/cuda/*.rs (pure move)` | `enhancement` | 1 |
| 3 | [#263](https://github.com/yusiwen/minfer/issues/263) `[cuda] split src/cuda_kernels.cu into src/cuda/kernels/ (header + guard + 19 TUs)` | `enhancement`,`test` | 2 |
| 4 | [#264](https://github.com/yusiwen/minfer/issues/264) `[cpu] split quants.rs / vec_ops.rs / kernel.rs along the ISA axis` | `enhancement` | 3 |
| 5 | [#265](https://github.com/yusiwen/minfer/issues/265) `[metal] split metal.rs / metal.metal into src/metal/{runtime,encode,ops,policy}.rs + src/metal/kernels/` | `enhancement` | 4 |
| 6 | [#266](https://github.com/yusiwen/minfer/issues/266) `[docs] mechanical check for src/<file>:NNN anchors + convert to symbol anchors` | `documentation`,`ci` | before 2 |
| 7 | [#267](https://github.com/yusiwen/minfer/issues/267) `[test] split the >1,000-line test files (cuda_backend/tests.rs first), counts identical` | `test` | 6 |

That is **7 tickets, all filed 2026-10-04**; every sub-ticket carries `Part of #261`, and #261 carries
the Step −1…6 checklist, the interaction table of §7.1, the target tree of §8 and the documentation plan
of §6. The former "stale size/claim sweep" ticket was folded into
[#219](https://github.com/yusiwen/minfer/issues/219) as a comment (decision 3, §9).

## 8. Resulting tree

`src/kernel/`, `src/quants/`, `src/vec_ops/`, `src/metal/` and `src/cuda/` **already exist** today —
they hold only `tests.rs` (plus `metal/mmap_align_test.rs` and `cuda/`'s ten issue probes). The plan
therefore does not create a new convention: the production parts simply join the directories that are
already there. `[S1]`…`[S4]` name the step that produces each entry; line ranges refer to the
pre-split file.

```text
src/
├── main.rs                                    (unchanged)
├── cuda.rs                          [S1]  module cuda: doc + AttnWindow + `pub struct CudaState`
│                                             (fields stay private) + free items + `pub(crate) use`
│                                             + the ten `#[cfg(test)] mod` declarations
├── cuda/                                    (exists: 10 test files today, unchanged)
│   ├── impl.rs                      [S1]  the 16 cross-family helpers + `mod` declarations
│   ├── impl/
│   │   ├── init.rs                  [S1]  A  1945–2251   device probe / tier / singleton
│   │   ├── accounting.rs            [S1]  B  2252–2276   weights_bytes / device_memory
│   │   ├── weights.rs               [S1]  C  2277–2901   register_weight + q6k/q4k expansion
│   │   ├── stream.rs                [S1]  D  2902–2964   bound/context stream, create/destroy
│   │   ├── buffers.rs               [S1]  E  2965–3016   get_or_grow, cuda_malloc/free
│   │   ├── copy.rs                  [S1]  F  3017–3188   H2D / async / D2H / pinned / D2D
│   │   ├── events.rs                [S1]  G  3189–3341   events, async staging, sync, latch
│   │   ├── capture.rs               [S1]  H  3342–3470   CUDA-graph capture / replay
│   │   ├── dispatch.rs              [S1]  I/J 3471–3918  matmul_f32_ptr*, MMQ gates
│   │   ├── mmq_quant.rs             [S1]  K  3919–4295   A-quantize + MmqCache
│   │   ├── prefill_mmq.rs           [S1]  L  4296–4667   auto_ksplit, prefill_mmq
│   │   ├── prefill_f16.rs           [S1]  M  4668–4919   f16 GEMM + w16 cache
│   │   ├── gpu_act.rs               [S1]  N  4920–5083   on-GPU quantize/gather/embed
│   │   ├── elementwise.rs           [S1]  O  5084–5223   norm/add/mul/silu/swiglu/rope
│   │   ├── attention.rs             [S1]  P  5224–5636   gqa / split / batched / prefill
│   │   ├── mmvq.rs                  [S1]  Q  5637–6274   decode MMVQ + q8_0 p32 planes
│   │   └── kvstore.rs               [S1]  R  6275–6479   KV store + fused QKV epilogue
│   ├── ffi_runtime.rs               [S1]  cudart/driver FFI + CudaPtr/CudaDevicePropBuf + test FFI
│   ├── policy.rs                    [S1]  pure predicates (env knobs / gates) + unit tests
│   └── kernels/                     [S2]  the CUDA translation units (kernels + their host
│       │                                   launchers; `<backend>/kernels/` is the one rule both
│       │                                   device backends share)
│       ├── common.cuh               [S2]  ~450: defines + device helpers + KV load idiom +
│       │                                   declarations of the #147/#162 helpers
│       ├── guard.cu                 [S2]  5521–5922  single owner of the #147/#162 state and of
│       │                                   the minfer_launch_* definitions (external linkage)
│       ├── matmul_f32act.cu         [S2]  63–693            ≈750 with launchers
│       ├── mmvq_aquant.cu           [S2]  694–1094          fused-producer A-quantize prepass
│       ├── mmvq_skipwrite.cu        [S2]  1095–1651         mode-2 skip-write variants
│       ├── mmvq_q6k.cu              [S2]  1652–1942         pipelined + dense split-plane q6_K
│       ├── ops_misc.cu              [S2]  1943–2375         padded Q6_K, gather/embed, f32×f32, f16×f32
│       ├── ops_elementwise.cu       [S2]  2376–2654         quantize f32→Q8_0, norm, bias, add/mul,
│       │                                                     SiLU/SwiGLU, i32 decode, RoPE
│       ├── kv_store.cu              [S2]  2794–3084 + 10173–10215  KV store, fused QKV epilogue,
│       │                                                     arena row move
│       ├── attention_decode.cu      [S2]  3085–3677         GQA, E1 window, kv_map, split-K, batched
│       ├── attention_hybrid.cu      [S2]  3678–4047         hybrid rpw (hd 128, f16 KV)
│       ├── attention_prefill.cu     [S2]  5136–5520         FA-style prefill (staged KV)
│       ├── gemm_wmma.cu             [S2]  5923–6490         dequant-to-f16 + wmma HGEMM
│       ├── gemm_smem.cu             [S2]  6491–6859         dynamic-smem formula + checked opt-ins
│       ├── gemm_fused_dequant.cu    [S2]  6860–7135         8p fused dequant-in-GEMM
│       ├── mmq_int8.cu              [S2]  7136–7540         R1 int8 MMQ prefill GEMM
│       ├── mmq_raw.cu               [S2]  7541–8074         P6 raw-byte MMQ
│       ├── mmq_nb.cu                [S2]  8075–8667         raw-nibble NB + A-layout transform
│       ├── mmq_bt_q6k.cu            [S2]  8668–9407         r38 q6_K BT
│       └── mmvq_multi.cu            [S2]  9408–10172        multi-token MMVQ + doc103/104
├── metal.rs                         [S4]  module metal: doc + free items + `pub use`
├── metal/                                   (exists: mmap_align_test.rs + tests.rs)
│   ├── runtime.rs                   [S4]  L1: MpsState / MetalDevice / library + pipeline cache
│   ├── encode.rs                    [S4]  L2: MpsCommandBuffer encoding
│   ├── ops.rs                       [S4]  L2: the op → encoding table
│   ├── policy.rs                    [S4]  pure predicates (MINFER_METAL_* / MINFER_* knobs)
│   └── kernels/                     [S4]  L3, ≤~800 lines each:
│       ├── common.h · dequantize.h · quantize.h
│       ├── mul_q4_0_q8_0.metal · mul_f32act_q4q5.metal · mul_f32act_kquant.metal
│       ├── mul_mm.metal · mul_mm_kq.metal · get_rows.metal · norm_elementwise.metal · rope.metal
│       ├── kv.metal · qkv_fused.metal
│       └── fa_parallel.metal · fa_split.metal · fa_decode.metal · fa_prefill.metal
│                                            (runtime fallback joins them with `concat!`)
├── kernel.rs                        [S3]  module kernel: `mod` + `pub use`
├── kernel/
│   ├── dispatch.rs                  [S3]  cpu_quant_matmul* (12–44, 322–390)
│   ├── pool.rs                      [S3]  Pool / par_for / set_cpu_threads (45–321)
│   ├── embed.rs                     [S3]  embed_tokens (391–)
│   └── tests.rs                            (exists)
├── quants.rs                        [S3]  module quants: `pub use`
├── quants/
│   ├── dot_q4_0.rs · dot_q4_1.rs · dot_q5.rs · dot_q8_0.rs    [S3]
│   ├── kquant.rs                    [S3]  Q4_K/Q5_K/Q6_K dots + Q8_K activation quantization
│   ├── quantize_q8_0.rs · quantize_q8_k.rs                    [S3]
│   ├── neon.rs                      [S3]  was `mod neon_kernels` / `mod neon_q8k`
│   ├── avx2.rs                      [S3]  was the inline `*_avx2` bodies
│   └── tests.rs                            (exists)
├── vec_ops.rs                       [S3]  module vec_ops: `pub use`
├── vec_ops/
│   ├── vec.rs · rms_norm.rs · rope.rs · softmax.rs · silu.rs   [S3]
│   ├── f16.rs · bf16.rs             [S3]
│   ├── neon.rs                      [S3]  was `mod neon_f16` / `mod neon_vec`
│   └── tests.rs                            (exists)
├── graph/                                   (unchanged: `backend.rs` + `registry.rs` stay the one
│                                             device seam; `*_backend.rs` stay the executors)
└── … (all other modules unchanged)
```

Non-`src/` changes that ride along: `build.rs` (CUDA/Metal file lists + one `rerun-if-changed` per
`.cu`/`.cuh`/.metal/.h + one `.o` per `.cu`, still one `libcuda_kernels.a`; Metal parts → one metallib;
plus the **new-file guard** that a file in `kernels/` cannot silently be absent from the list),
`scripts/check_cuda_launch_returns.py` (directory discovery + the "no `<<<>>>` in a `.cuh`" assertion) +
`tests/fixtures/cuda_launch_sites.tsv` (regenerated, still 130 rows, **5th column `file`**),
`scripts/check_doc_line_anchors.py` (new, ticket 6),
`docs/SOURCE-LAYOUT-PLAN.md` (this file) + `AGENTS.md` Layout block + `docs/ARCHITECTURE.md`
(layer definition + interface-eligibility rule) + the path/anchor sweeps in `docs/BACKENDS.md`,
`docs/CUDA-BACKEND-DESIGN.md`, `docs/inference_e2e_walkthrough/{07,15}*.md`, `docs/GPU_SAFETY.md`,
`docs/BUILD.md`, `docs/ARCHITECTURE-ROADMAP.md`, `docs/CUDA_OPTIMIZATION.md`,
`docs/LLAMA-CPP-MMQ-ANALYSIS.md`, `docs/ARCHITECTURE-EXECUTION-PLAN.md`.

## 9. Decisions (all taken — nothing open)

| # | Decision |
|---|---|
| 1 | **No file-size ratchet.** The ~800-line target is guidance enforced by review; no script. |
| 2 | **`kernels/fa_prefill.metal` (812 lines) is accepted** — the prefill flash-attention kernel plus its helpers is one unit; llama.cpp splits FA *instantiations* (`template-instances/`), not the body, so there is no better boundary here. |
| 3 | **The stale-fact sweep is folded into [#219](https://github.com/yusiwen/minfer/issues/219)**; no separate documentation ticket. |
| 4 | **Step 4 is Mac-local**, and this document must be sufficient alone for a macOS agent (§6.5): the file table, the two compile entry points, the verification commands, and an absolute Mac box label in the record. |
| 5 | **The long test files are in scope** as Step 6 (split by op family/topic, counts identical). |
| 6 | **The frozen set is approved** (§6.2): `docs/cuda_optimization_steps/*`, `docs/QWEN2.5-*.md`, `docs/DEBUGGING-*.md`, `docs/KNOWN-CPU-ISSUES-*.md`; it may only shrink, and ticket 6's checker carries it with a reason per file. |
| 7 | **`docs/cuda_tutorial/*` is live** (180 of the cuda anchors live there): each example is re-pointed in Steps 1–2, not frozen. |

## 10. Verification matrix

| Step | Commands | Expected |
|---|---|---|
| 0 docs | `scripts/check_docs_links.py`, `scripts/check_status.py --check`, `scripts/build_book.sh` | green |
| 1 `cuda.rs` | `cargo build --release --features cuda`, `scripts/cuda_test.sh`, `cargo test --release`, `cargo fmt --all --check`, `check_source_layout.py`, `check_dead_code_{annotations,oracle}.py`, `FEATURES=cuda scripts/real_model_gates.sh` ×2 | 565/0/42; 480/0/36 + 10/0/6; 42/0 ×2; all checkers green |
| 2 `.cu` | Step 1's list plus `check_cuda_launch_returns.py` (+`--selftest`, `--check-fixture`), `MINFER_TEST_ISSUE162=1` device gate, `MINFER_OP_TIMING=1` cold-start record | 130 sites; 42/0 ×2 bitwise; module-load cost recorded |
| 3 CPU | CPU suites + the two-arch dead-code set comparison | 480/0/36, 10/0/6, sets identical |
| 4 Metal | on a Mac: `cargo build --release` (non-empty metallib), real-model gates, #255's two judgments | recorded on the Mac box |
| 5 close | #225's table re-measured; #53's `DeviceMemory` for Metal | one interface, two implementations |
| 6 tests | the step-appropriate suite (CUDA 565/0/42 for the executor tests, CPU 480/0/36 + 10/0/6) + `check_source_layout.py` | counts **identical**; every new test file named by a `mod` |
| all | `scripts/check_docs_links.py`, `scripts/check_status.py --check`, `scripts/build_book.sh`, `scripts/check_doc_line_anchors.py` (once ticket 6 lands) | green; no stale anchor |

**Suite counts are read, not remembered.** The numbers in the table above are the pre-#138 rows; the
current rows are `docs/status.toml` (after #138: CUDA **567 / 0 / 42**, CPU **481 / 0 / 36** on
`dgxspark (aarch64, GB10 sm_121)` and **479** on the CI runner, integration **10 / 0 / 6**). Every step
must read the rows from that file rather than quote this document, and a step that moves a row updates
`AGENTS.md` and `docs/status.toml` in its own PR.

Standing rule for every step: a layout PR moves code and nothing else; anything else it notices is
filed, not fixed inside it.
