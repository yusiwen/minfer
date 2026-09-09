# 64 · r60 — the coronation: flipping the verified gate set to default-on (PROMOTION, LANDED)

> **Result**: the default path = the verified 1.080× path — 7B pp3314 default
> ≈3578–3599 tok/s (headline ~3581); `MINFER_MMQ=0` falls back to legacy f16
> (measured ~2226 in this window; the clean-class documented value is ~2353).
> Plane memory +3.27 GB (default 9484 MiB vs planes-off 6217 MiB). decode
> (nt==1) untouched. Along the way, a bisect caught and fixed a pre-existing
> mode-2 flaw on mixed-quant models (multiturn_reuse gate 168/1/3).
> **Commit**: `57edcf6` (+ `7029ee4` docs). **Date**: 2026-09-06 (Session F
> finale).

## 1. Background — where things stood

By r59b, every gate of P6's r34–r59 gate set had been individually verified
in opt-in state: r34's quantize-transpose prepass, r28/r29's NB kernel
family, r38–r41's q6_K line, r48/r49's FA and prepass dedup, r51/r52's fused
producers (mode 1/2), r53/r56's plane bundle, r59's q4_K W_dsc. r59b
delivered the final numbers: clean 3590.8 tok/s, 1.080× vs llama-bench
3323.29 @pp3314. Exactly one move of the campaign remained — **flip the
default path**.

Why the default was still f16 is written in this campaign's history: at R1
(2026-08-31), the then-untuned MMQ kernels managed only ~2.5–3 GMAC/s/matmul
under GPU contention while the f16 w16-cache path ran ~8–11, 7B @2K ~155 vs
~630–880 tok/s — **at that time defaulting to f16 was the only right call**.
Twenty-six rounds later the ranking inverted: the MMQ stack 3580.7 vs f16
~2353 (clean-class). Flipping the default is therefore not "flipping a
flag" but a full measurement task: six gates move from opt-in to opt-out at
once, every former A/B escape hatch must survive under the new semantics,
and every existing mechanism that interacts with the default path (CUDA
Graph capture, the decode path, mixed-quant models, the memory budget) must
be re-exercised.

This is the campaign's only step that "changes no kernel, only decisions" yet demands the most verification.

## 2. Principle — the GPU mechanism: promotion is a measurement problem

### 2.1 The r54-pattern gate flip

The six promoted gates uniformly flip to opt-out semantics (**unset or any
non-"0" value = ON**, i.e. the verified-best path; explicit `"0"` = pre-r60
behavior):

| Gate | What it controls | What "0" reverts to |
|---|---|---|
| `MINFER_MMQ` | prefill uses int8 MMQ GEMM | legacy f16 w16-cache path |
| `MINFER_MMQ_RAW` | raw-byte staging variant (q4_K whole super-block) | per-type legacy kernels |
| `MINFER_MMQ_RAW_NB` | NB (raw-nibble) kernel family | qb8 pre-expanded form |
| `MINFER_MMQ_A_TRANSPOSE` | r34 quantize-transpose prepass | in-kernel A layout transform |
| `MINFER_MMQ_Q6K_NB` | q6_K NB path | q6_K f16 fallback |
| `MINFER_MMQ_A_FUSE` | unset = mode 2 (skip-write fused producer); "1"/"2" keep the r51/r52 semantics; "0"/unknown = off | independent quantize prepass |

`MINFER_MMQ_Q6K_EXP` / `MINFER_MMQ_Q4K_DSC`, already default-on since
r54/r59, keep their semantics. All reads are single-sourced in
`CudaState::mmq_gate_on(name)` — unset maps to true (fail-open to the
verified path is deliberate: the release config = the verified config; no
stray value should pull the user off it).

### 2.2 The four classes of "things that must not move"

The risk of flipping the default is not the new path (verified by 26 rounds
of A/B) but **whether the default-path change touches mechanisms that never
interacted with these gates**:

1. **decode neutrality**: decode (nt==1) never ran MMQ; after promotion no
   `MINFER_MMQ*` read may appear on the nt==1 path — otherwise decode
   behavior drifts with env vars.
2. **reuse-identity neutrality**: graph topology is decided by
   `GraphParams`/`CParams`, and reuse compares params — these structures
   must stay env-free. The gates are allowed to act in exactly two places:
   **dispatch-time kernel choice** (nt>=16 matmul dispatch) and
   **registration-time plane building** (the loader); CUDA Graph capture
   bakes each process-constant choice into replay, which is inherently safe.
3. **per-gate liveness**: under the default environment every gate must
   actually be alive (r53's lesson: a fallback-correct fast path must carry
   a visible label — parity cannot see the dispatch path).
4. **mixed-quant degradation + two-way memory**: not every model is
   all-q4_K/q6_K — mode-2's skip-write producer is unsound under a weight
   mix that NB-BT cannot consume (the protagonist of §3.3); memory must be
   reported in both directions.

### 2.3 The gate × proof matrix

The six gates share one proof set but each has its own failure modes —
verification is booked pairwise as "gate × evidence":

| Proof | MINFER_MMQ | RAW | RAW_NB | A_TRANSPOSE | Q6K_NB | A_FUSE(mode 2) |
|---|---|---|---|---|---|---|
| parity ×3 default env | ✓ (entry arm) | ✓ | ✓ | ✓ (prepass layout) | ✓ | ✓ (skip-write semantics) |
| greedy-32 vs snapshot+gated | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| dispatch-label liveness | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ (166× hits) |
| opt-out "0" aliveness | ✓ (f16 path) | ✓ | ✓ | ✓ | ✓ | ✓ ("0"/unknown=off) |
| decode/Reuse neutrality | ✓ (shared by the family) | — | — | — | — | ✓ (rows/n ≥ 16 guard) |
| Memory both ways | ✓ (+3.27 GB / 20.5 GB) | — | — | — | — | — |

The "snapshot+gated" contrast means: the greedy stream produced by the
r59b-finalized binary plus the full opt-in gate env-var set (the
configuration those 26 A/B rounds verified), compared byte-for-byte against
the new default's stream with zero env vars — the two must be the same
numeric path for promotion to be merely a relocation of the switch.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Single-sourced semantics**: all six gate reads go through
  `mmq_gate_on`; gate semantics henceforth change in one place.
- **mode-2 becomes default with a degradation ladder**: in
  `mmq_a_fuse_mode()` unset maps to 2; any window reader
  (GRAPH_DUMP/DUMP_DIR/trace/viz) or fallback condition (NO_PREFILL_GEMM,
  non-NB-BT weight mix) present → degrade to mode 1 (fused but writes f32)
  — keeping r51's gain, giving up only the skip.
- **The weight-mix flag lives at registration**: `nb_bt_only` is an
  AtomicBool initialized true, cleared by the loader when it registers a
  non-qualifying weight. Registration precedes the first forward, so zero
  per-node cost.
- **The compute-capability gate stays**: `mmq_active()` still requires
  `cc >= 800` (mma.m16n8k32 s8 needs sm_80+; sm_75 only has k16) — promotion
  changes the environment semantics, not the hardware applicability. A side
  effect points the right way too: the loader uses `mmq_active()` to decide
  whether to skip the f16 cache warm pass (the MMQ path reads raw bytes; the
  w16 copy is dead weight) — with the default on, non-MMQ machines
  automatically fall back to the f16 path and rebuild their own cache, and
  both paths' memory accounts hold.

### 3.2 Key code

The gate semantics proper (`src/cuda.rs`):

```rust
// r60: the promoted MMQ gate semantics — unset / any non-"0" value =
// ON (the verified 1.080x path), explicit "0" = opt-out to the pre-r60
// disabled/f16 behavior (the r54 `MINFER_MMQ_Q6K_EXP` pattern).
// Single-sourced: every promoted MINFER_MMQ_* dispatch and
// plane-registration read goes through this.
pub fn mmq_gate_on(name: &str) -> bool {
    std::env::var(name).map_or(true, |v| v != "0")
}

// r60: the loaders call this when they register a quantized weight that
// is NOT NB-BT-consumable (not q4_K/q6_K), or a 2-D F32 matmul weight:
// mode-2 skip-write fused producers become unsound for such mixes ...
pub fn clear_mmq_nb_bt_only(&self) {
    self.nb_bt_only.store(false, std::sync::atomic::Ordering::Relaxed);
}
```

Mode selection (default 2 + triple degradation ladder):

```rust
pub fn mmq_a_fuse_mode(&self) -> u8 {
    if !(self.mmq_active()
        && Self::mmq_gate_on("MINFER_MMQ_RAW")
        && Self::mmq_gate_on("MINFER_MMQ_RAW_NB")
        && Self::mmq_gate_on("MINFER_MMQ_A_TRANSPOSE")
        && Self::mmq_gate_on("MINFER_MMQ_Q6K_NB"))
    { return 0; }
    // r60 promotion: unset = mode 2 (the verified-best skip-write fused
    // producers); "1"/"2" keep the r51/r52 override semantics; "0" (and
    // any other unrecognized value, as before r60) = off.
    let requested = match std::env::var("MINFER_MMQ_A_FUSE").as_deref() {
        Err(std::env::VarError::NotPresent) => 2,
        Ok("1") => 1,
        Ok("2") => 2,
        _ => 0,
    };
    // r60: mode 2 additionally requires the NB-BT-only weight mix —
    // a mixed-quant model degrades to mode 1 regardless of how mode 2
    // was requested (default or explicit "2").
    let mode2_possible = requested == 2
        && !Self::no_prefill_gemm()
        && self.nb_bt_only.load(std::sync::atomic::Ordering::Relaxed);
    match requested {
        1 => 1,
        2 if mode2_possible
            && std::env::var_os("MINFER_GRAPH_DUMP").is_none()
            && std::env::var_os("MINFER_DUMP_DIR").is_none()
            && !crate::trace::enabled()
            && !crate::live::enabled() =>
        { 2 }
        2 => 1,   // window reader/fallback condition active: keep the r51 fused semantics
        ...
    }
}
```

The guards at the MMQ dispatch entry (after promotion they guard exactly the
default path):

```rust
// src/cuda.rs — matmul dispatch header (excerpt)
// MINFER_MMQ-gated (r60 PROMOTION: default ON — the promoted 1.080x path;
// `MINFER_MMQ=0` = the f16 wmma path). ... id % 32 == 0 covers the block
// math of every type (q6_K runs as k32 chunks with dual 16-sub rescale).
if nt >= 16
    && id % 32 == 0
    && !Self::no_prefill_gemm()
    && matches!(ttype, TensorType::Q4_0 | TensorType::Q4_1 | ... )
```

The loader's registration arms (the two clearing points for mixed-quant +
2-D F32, `src/models/qwen2/loader.rs`):

```rust
cuda.register_weight(&ti.name, tensor.data());
// r60: a non-NB-BT-consumable quantized weight (not q4_K/q6_K) makes
// mode-2 skip-write fused producers unsound — see CudaState::clear_mmq_nb_bt_only.
if !matches!(ttype, TensorType::Q4_K | TensorType::Q6_K) {
    cuda.clear_mmq_nb_bt_only();
}
... // r59 W_dsc registration gate (RAW_NB/A_TRANSPOSE default-on after r60)
} else if ttype == TensorType::F32 {
    cuda.register_weight(&ti.name, tensor.data());
    // r60: a 2-D F32 weight is an f32 MATMUL weight (norms/biases are
    // 1-D) — its GEMM reads the f32 A directly, so a mode-2 skip-write
    // producer upstream would feed it a dead buffer.
    if tensor.shape.len() == 2 {
        cuda.clear_mmq_nb_bt_only();
    }
}
```

### 3.3 Pitfalls: the pre-existing mode-2 flaw the bisect caught

After the flip, suite gate #7 (multiturn_reuse, 0.5b q4_0 fixture) ran
**168/1/3** for the first time — one failure. The troubleshooting itself is
this doc's methodology sample:

1. **A triple stash bisect**: clean HEAD + default env (= simulating how
   promotion runs) **PASS**; clean HEAD + gated env (`MINFER_MMQ=1 ...`,
   i.e. the verified opt-in configuration) **also FAIL**; the r60 build
   **also FAIL**.
2. Conclusion: the failure is a **pre-existing flaw of the verified
   configuration itself**, exposed by promotion (0.5b's q4_0 weights are not
   an NB-BT-consumable quant type — the mode-2 fused producer skipped the
   f32 write, while a generic `mmq_nt` consumer later in the graph
   legitimately demanded a re-quantization of that A → read a dead buffer).
   Promotion introduced no new error; it merely let a long-buried landmine
   step into the default path.
3. **Fix, not revert** (r60's principle: fix the wart the default now
   exposes, loudly): mode-2 producers are conditioned on the registration-
   time `nb_bt_only` flag; mixed-quant models degrade mode 2 → mode 1
   (writes **both** the plane and f32, correct everywhere).
4. A subtler exposure fixed along the way: **2-D F32 weights**. Their GEMM
   reads the f32 A directly and bypasses MmqCache, so r52's dead-write
   backstop (a loud refusal on cache miss) cannot catch it at all — after
   the producer skip-writes, the f32 GEMM would silently read a dead buffer.
   The refusal path never had this exposure; this is also why the clearing
   point sits at registration rather than consumption.

## 4. Verification

- **decode neutrality**: grep proves zero `MINFER_MMQ*` reads on the nt==1
  path (plus a `rows/n >= 16` guard before every `mmq_a_fuse_mode` call);
  `decode -n 16 --greedy` byte-identical; tg128 45.2 = 45.2 flat — defends
  against "prefill gates leaking into decode".
- **reuse-identity neutrality**: `GraphParams`/`CParams` env-free,
  `supports_op`/`supports_fused` env-free (the graph side only mentions
  those env names in comments, zero reads) — defends against "gates affect
  graph topology, breaking reuse / drifting CUDA Graph replay".
- **parity ×3 default env 9/9** — defends against "some cross term of the
  default combination never individually verified".
- **greedy-32: default vs snapshot+gated env byte-identical (453 B stream)**
  — proves the default path and the verified opt-in path are the same
  numeric path; promotion only moved the switch.
- **opt-out aliveness**: `MINFER_MMQ=0` → zero mmq occurrences in dispatch
  labels; f16 spot ~2226 tok/s (this window; clean-class documented ~2353)
  — defends against "a rusted-shut escape hatch".
- **0.5b q4_k_m smoke**: post-fix byte-identical — after the fix it runs the
  mode-1 producer (the plane is still usable: **the plane is a pure function
  of A**), the mixed-quant degradation path really works.
- **suite 169/0/3**: one transient SIGSEGV did not reproduce
  (overcommitted-pool-hazard class, recorded).
- **memory both ways**: default-on 9484 MiB; planes-off 6217 MiB (**+3.27
  GB** plane cost); `MINFER_MMQ=0` ~20.5 GB — **the escape hatch is ~11 GB
  heavier than the default** (the f16 cache is dead weight).

## 5. Results

- **Prefill A/B** (baseline contrast before the fix): base 3592.1 vs new
  default 3578.0 (−0.39%, overlapping intervals) — no gate leaked; post-fix
  re-measure +0.44% (the other direction, noise; mode 2 confirmed active on
  7B).
- **Finalized default**: 7B pp3314 ≈3578–3599 (headline ~3581) = **1.080×**
  vs llama-bench 3323.29 — r59b's verified values become the release default
  unchanged.
- **Documentation rule (in force from here)**: **default = the verified
  1.080× path; `MINFER_MMQ=0` = legacy f16 path**. From R1's ~441 tok/s
  first parity-clean MMQ measurement to the finalized ~3581, the campaign
  arc is **8.1×**.
- **A glance at the campaign arc** (same-anchor series, master-table Perf
  column): R1 opt-in 441 → r34 prepass 1496.8 → r39 KDR=2 1777.5 → r40 third
  resident block 2015.6 → r41 uint4 B-expand 2605.2 → r48 register softmax
  2749.9 → r52 skip-write 3011.3 → r53 W_exp bundle 3176.9 → r56 pre-q4_K
  3212.5 → r59 W_dsc → **r59b finalized 3590.8 / headline ~3581 = the r60
  default**. Each hop's mechanism is in its numbered step doc.
- **Status**: LANDED. Six gates default-on, every gate with a "0" exit,
  decode and reuse identity proven unmoved, mixed-quant auto-degrades,
  memory reported both ways.

## 6. Lessons

1. **Promotion is a measurement problem, not a flag flip**: decode
   neutrality, reuse neutrality, per-gate liveness, mixed-quant degradation,
   two-way memory — only when all five are measured does the default deserve
   to flip.
2. **After flipping a default, bisect config vs code on any new failure**
   (clean HEAD default / clean HEAD gated / new build, three-way): all three
   failing = a pre-existing flaw of the verified configuration exposed —
   **fix it loudly, do not revert the promotion**.
3. **skip-write-type optimizations are sound only relative to a consumer
   set**: when betting that "this output will never be read", condition the
   bet on a registration-time-decidable predicate (`nb_bt_only`), not on
   hopes; mixed configs need automatic degradation, not crashes or silent
   corruption.
4. **Escape hatches must report memory too**: here the opt-out path (f16
   cache ~20.5 GB) is 11 GB heavier than the default (~9.5 GB) — the
   "conservative old path" may be the aggressive one on memory.

---
← 63 · [Index](./README.md) · 65 →
