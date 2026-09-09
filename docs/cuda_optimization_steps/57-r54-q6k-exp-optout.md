# 57 · r54 — `MINFER_MMQ_Q6K_EXP`: an exit valve for the 1.52 GB W_exp plane (LANDED)

> **Result**: the default path is byte-unchanged (3181.0 tok/s, same noise band as r53's landed 3176.9); `MINFER_MMQ_Q6K_EXP=0` trades −5.04% of whole-prefill speed (3020.7 tok/s) for 1.52 GB of device memory back (measured per PID: 7636 → 6182 MiB, Δ1454 MiB ≈ the plane's census value of 1,521,237,632 B). Both modes are parity/greedy all-green, and the liveness census of the 27 q6_K launches holds 27×/0 in both directions.
> **Commit**: `3252e96` (code) + `b860b7e` (record). **Date**: 2026-09-06.

## 1. Background — where things stood

r53 had just closed out the q6_K line: bundling r44's "pre-expanded dense B plane W_exp" (removing the recombination work) and r45's cp.async staging (removing the wait) into the NB-BT q6_K kernel took whole-prefill 3024.7 → 3176.9 tok/s (+5.03%), with the vs-llama multiple tightening from 1.09× to 1.05×. The price was written on the same line: **+1.52 GB of device memory**.

What is this 1.52 GB? W_exp is a "dense centered-int8 plane expanded per padded q6_K tensor at `od × id` bytes" — the q6_K ql/qh nibble stream expanded **once at registration** into a dense 1-byte-per-element int8, so the kernel's B staging degenerates into a pure cp.async copy. On 7B q4_k_m the tensors that go through q6_K are 14 attn_v + 14 ffn_down + output.weight, totaling 1,521,237,632 B ≈ 1.52 GB (r53's record also incidentally corrected the task's "+15 MB" underestimate — that estimate had taken one ffn_down's increment as the whole price). GB10's 127 GB unified memory doesn't feel it, but minfer is an engine shipped to users: on an 8–16 GB card, carrying an extra 1.5 GB for a default-on +5% turns "can it run at all" into "am I willing to buy another card."

And the shape at the time made this tax **impossible to decline**: the plane registration hung under the `MINFER_MMQ_Q6K_NB` gate, and that gate also controlled the kernel itself. To turn the plane off you had to turn the kernel off too, back to the f16 path — a retreat counted in tens of percent, not "5% less." r53's failure semantics covered only one case: on allocation failure the map stays empty, the kernel falls back, and it eprintlns loudly. A user "wanting a smaller memory footprint" was not among them — the allocation would succeed, and the memory would be eaten.

So r54's task was to fit this memory-for-speed switch with an **exit valve**. It sounds like a one-line env check; the real constraints were three:

1. **No second numeric path may be introduced.** With the plane opted out, the kernel must still be there, and its output must be bit-identical to the default path. Fortunately r53 had already templated the kernel as `<KDR, EXP>`: the `EXP=false` instantiation compiles the cp.async B branch out entirely, keeping the r41-era "in-kernel uint4 ql+qh expansion" — a path that was itself a parity-green, in-service implementation, compiled into the binary since r53. The opt-out merely means **not building the plane and letting the dispatch map-miss fall into this existing instantiation**. Both instantiations live in the same cubin; the env only changes whether the host side registers the plane.
2. **The plane gate must be decoupled from the kernel gate.** `MINFER_MMQ_Q6K_NB` controls the NB-BT kernel itself; a user may want that kernel (its in-kernel expansion is not slow either) but not the plane. So EXP is an independent switch, ANDed with NB — the user picks any of four combinations: kernel+plane (default), kernel without plane (saving 1.52 GB), all-f16, and the meaningless "kernel off, plane on" (the plane then isn't built either — the registration gate incidentally guarantees no reader-less plane is left behind).
3. **A deliberate fallback must be distinguishable from an accidental one.** r53's namesake lesson: parity/greedy cannot see a "fast path that silently never runs" — on r53's landing day, a wrong map key sent all 27 launches down the slow path while parity and greedy stayed all green; the liveness label was what counted it out. r54 widens the B-path label from two states to three: a user-initiated off (`exp=off`) and a registration-bug map-miss (`fallback!`) must look different in the log.

Once this step was done, the default path had not moved by a single bit — its whole value is that it gave r56 (the W_dsc plane) and r60 (the promotion flipping defaults on) a reusable "default-on + `0` opt-out" template. r60's promotion definition copies this shape verbatim: all six MMQ gates become "default-on + `0` opt-out."

Two execution-level notes: first, this is a **user-approved independent switch** (the record's own words: "Independent switch (user-approved)") — it is not on any bundled thesis's chain; it is a pure product decision. Second, it was deliberately done **before** r55's convergence verdict: making "the plane is optional" an accomplished fact first is what makes the convergence statement's 1.05× a 1.05× with an escape hatch.

## 2. Principle — the GPU mechanism

### 2.1 What the 1.52 GB buys while on

The NB-BT q6_K kernel must stage the B (weights) into shared memory in KDR=2 batches for every output tile. The two paths differ entirely in the staging phase:

- **EXP=true (r53 default)**: B staging is a pure cp.async 16 B-chunk copy from the W_exp plane. Dense index `W_exp + j*id + sb*256 + cbase*32 + cc*16` (r44's stride root-cause fix), 16 B alignment guaranteed by the `id % 256 == 0` gate; rows beyond od are zero-filled by `gemm_cp16`'s src-size 0. No ql+qh recombination ALU, no register round trip, no per-nibble ql/qh reads; the copy latency goes to the async units, hidden under the previous tile's compute by the group wait. r44 quantified "removing the work" at −10.9% of kernel time and r45 quantified "removing the wait" at −10.2% — r53 proved the two are near-additive (ffn_down kernel −20.5%: 16.06 → 12.76 ms; attn_v −15.9%).
- **EXP=false (the r41 path)**: the staging phase itself reads the raw ql/qh nibble stream and recombines the centered int8 on the mma loop's critical path with shift/mask ops (the per-16-element-group version of `v = ((ql[..]>>qsh)&0xF) | (((qh[..]>>qh_shift)&3)<<4) - 32`). This path has been in service since r41 — r41's uint4-ization merged 16 per-byte LDGs into one 16 B read, cutting the L1TEX scoreboard exposure (85.5% → 33.6%) — and r54 wrote zero new lines for it.

So the performance price of turning the plane off = giving back exactly what r44+r45+r53 bought. The whole-prefill arithmetic: 3020.7 / 3181.0 = 0.9496, i.e. **−5.04%** — almost symmetric with r53's +5.03%. The symmetry is no coincidence: r53's gain was precisely "the kernel stops doing these two things"; r54 puts them back in the kernel and the gain flows away down the same path.

| Instantiation | B staging form | origin | role in r54 |
|---|---|---|---|
| `<KDR, EXP=true>` | pure cp.async copy from the W_exp plane | r53 | default path (3181.0 tok/s) |
| `<KDR, EXP=false>` | in-kernel uint4 ql+qh expansion | r41 | the EXP=0 path + all map-miss fallbacks (3020.7 tok/s) |

Both instantiations are in the same cubin (template compilation artifacts); the env changes only the host-side "register the plane or not."

### 2.2 Why the byte counts reconcile

W_exp per tensor = `od × id` bytes (1 B centered int8 per element). The census value 1,521,237,632 B = 1450.9 MiB; the measured Δ is 1454 MiB, a gap within page granularity and allocator overhead — "≈ the census" holds. Conversely this also verifies that with EXP=0 **all** of the plane memory comes back: the registration early-return happens before any sibling allocation, so device memory sits at its pre-r53 level rather than "built but unused."

### 2.3 Why byte-identity is a structural guarantee, not a verification result

The q6_K expansion is a pure integer transform: ql's 4-bit nibble and qh's 2-bit high bits compose a 6-bit value, minus 32 to center — no rounding, no floating point, no ordering issues. Both EXP instantiations feed the same mma element-identical values; the only difference is whether the expansion happens on the host (at registration, `expand_q6k_dense`) or in-kernel (r41's uint4 branch). So as long as the dispatch really lands in `<KDR, EXP=false>`, bit-identity is constructed, not something a tolerance gate has to rescue. r54's verification budget therefore went almost entirely into "the dispatch really lands there" (the liveness census); parity is a routine confirmation.

### 2.4 The liveness label mechanism

Under `MINFER_MMQ_RAW_NB_DEBUG=1`, every q6_K launch eprintlns one line: "the path currently taken." Its necessity comes from r53's field test: parity/greedy cannot distinguish "the fast path is fast" from "the fast path never ran but the slow path isn't slow enough to give itself away." The label bins every launch into a countable bucket, and a census of `27×/0` (27 fast, 0 accidental) is the evidence that "the mechanism is actually running."

### 2.5 New concepts, first appearance

- **W_exp plane**: a registration-time pre-expanded dense int8 copy of the q6_K weights, `od × id` B per tensor, introduced in r53, opt-out-able via `MINFER_MMQ_Q6K_EXP=0` from r54 on.
- **in-kernel expand (the r41 path)**: the pre-existing implementation where, absent a plane, the kernel reads ql/qh as uint4 in the staging phase and recombines them with shifts.
- **liveness label**: the path-annotating eprintln under `MINFER_MMQ_RAW_NB_DEBUG=1`; used to count fast-path launches and guard against silent fallbacks.
- **per-PID memory measurement**: on GB10, `nvidia-smi`'s aggregate usage field reads `[N/A]`; per-process usage requires `--query-compute-apps`.

## 3. Implementation

### 3.1 Design choices (why this shape and not another)

- **Default-on, not default-off**: the campaign's default discipline is "default = the verified fastest path" — performance responsibility sits on the default, and the tradeoff is handed to an explicit opt-out. The reverse (default-off, opt-in for speed) would hide 5% behind an environment variable nobody reads. r60's promotion "default-on + `0` opt-out" inherits exactly this ordering.
- **Default-on, `"0"` to opt out**: the gate is written `std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() != Ok("0")`. Rust's `env::var` returns `Err` for an unset variable, and after `as_deref()` the miss is still not `Ok("0")`, so the inequality holds — **unset is equivalent to `"1"`, i.e. r53's behavior**. This "default is on" phrasing deliberately avoids boilerplate like `unwrap_or("1")`, makes an explicit "0" the only opt-out channel, and later became r60's promotion-standard pattern.
- **Early-return at the registration point, not a dispatch-time fork**: `EXP=0` makes the registration function return before the sibling is even built. "Build the plane but don't use it at dispatch" would cost the full 1.52 GB and leave the switch nominal; "check the env at dispatch and pick the instantiation" would leak the env from registration semantics into execution semantics, one extra lookup per launch. The early return means the env is read exactly once, at load.
- **AND semantics**: EXP is only meaningful while the NB kernel is alive, so plane registration hangs under `mmq_gate_on("MINFER_MMQ_Q6K_NB") && EXP != "0" && id % 256 == 0`; `Q6K_NB=0` turns off kernel and plane together, so "the plane exists but its consumer is gone" orphan memory cannot occur.
- **Failure semantics inherited from r53**: when plane alloc/upload fails, the map stays empty, the kernel falls back to in-kernel expand, and one loud eprintln fires — r54 doesn't touch that path, it only makes sure the path now has a name (see the labels in 3.2).

### 3.2 Key code

The registration gate (current tree `src/cuda.rs`, the EXP check introduced in r54; the `od % 2 == 0` dsc lines are a r56 addition, see doc 59):

```rust
// r54: MINFER_MMQ_Q6K_EXP decouples the plane from the kernel gate —
// unset/"1" keeps the r53 default (build it), explicit "0" skips the
// build entirely (registration early-returns; device memory stays at
// the pre-r53 level) so dispatch map-misses into the EXP=false r41
// in-kernel expand. ANDed with Q6K_NB: EXP only matters when the NB
// kernel is live.
if Self::mmq_gate_on("MINFER_MMQ_Q6K_NB")
    && std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() != Ok("0")
    && id % 256 == 0
{
    self.register_weight_q6k_exp(name, &padded, od, id);
    if od % 2 == 0 {                       // r56: the W_dsc plane rides the same gate
        self.register_weight_q6k_dsc(name, &padded, od, id);
    }
}
```

`register_weight_q6k_exp`'s map semantics (same file, current tree) — the key is the **padded weight's device pointer** (r53's liveness incident was exactly a mistake here), the value is the plane pointer:

```rust
pub fn register_weight_q6k_exp(&self, name: &str, padded: &[u8], od: usize, id: usize) {
    // geometry-encoded sibling name: a same-name different-shape
    // re-registration can never collide with (and silently reuse) a stale
    // plane of the same byte size but a different od/id layout.
    let exp_name = format!("{name}__exp{od}x{id}");
    let exp = Self::expand_q6k_dense(padded, od, id);
    self.register_weight(&exp_name, &exp);
    // the MAP is keyed by the PADDED weight's device pointer (what
    // prefill_mmq holds); the value is the W_exp plane's pointer
    if let Some(wp) = self.get_weight_ptr(name) {
        if let Some(ep) = self.get_weight_ptr(&exp_name) { /* insert(wp, ep) */ }
    }
}
```

The dispatch-side lookup (inside the current tree's `prefill_mmq`) — with `EXP=0` the map has no entry at all, `unwrap_or(null)` leaves `w_exp` a null pointer, and the launcher selects the `<KDR, EXP=false>` instantiation accordingly:

```rust
// r53: pre-expanded B plane lookup by the padded weight's
// device pointer (null on miss -> launcher selects the r41
// in-kernel-expand instantiation).
let w_exp = self
    .q6k_exp
    .lock()
    .unwrap()
    .get(&(wptr as usize))
    .map(|cp| cp.0)
    .unwrap_or(std::ptr::null_mut());
// r56 (Session E item 2b): the precomputed dsc f32-pair plane
// (null on miss -> the r41 scalar dsc path in-kernel).
let w_dsc = self
    .q6k_dsc
    .lock()
    .unwrap()
    .get(&(wptr as usize))
    .map(|cp| cp.0)
    .unwrap_or(std::ptr::null_mut());
```

The kernel's two instantiations (current tree `src/cuda_kernels.cu`; the `W_dsc` parameter in the signature is likewise a r56 addition; the `EXP=false` branch is r41's uint4 expansion preserved verbatim):

```cuda
template <int KDR, bool EXP>
__global__ void __launch_bounds__(256, 3) mmq_raw_nb_bt_q6k_kernel(
    const uint8_t* __restrict__ W, const uint8_t* __restrict__ W_exp,
    const uint8_t* __restrict__ W_dsc, ...
) {
    ...
    if (EXP) {
        /* r53 bundle: the ql+qh recomb + -32 centering ran ONCE at
         * registration ... so the staging is a pure cp.async bulk copy
         * (explicit PTX) from W_exp — no recomb ALU, no register
         * round-trip, no ql/qh reads ... Dense index:
         * W_exp + j*id + sb*256 + cbase*32 (16B-aligned: id is a
         * multiple of 256 on this path). Rows beyond od zero-fill via
         * the cp.async src-size qualifier (gemm_cp16 full=0). */
        ...
    } else if ((bstride & 15) == 0) {
        /* r41: 16-elem group expand via uint4 ql+qh global loads. ...
         * Element-for-element identical to expand_q6_elem. */
        ...
    }
```

The B path's three-state label (current tree `src/cuda.rs`, under `MINFER_MMQ_RAW_NB_DEBUG=1`):

```rust
// r54: name WHY the r41 in-kernel expand is running —
// "exp=off" is the intentional MINFER_MMQ_Q6K_EXP=0
// switch; "fallback!" means a W_exp build was expected
// (padded weight, EXP gate on) but the map missed
// (alloc/upload failure or a registration bug). Raw
// 210-B weights never get a plane -> kept unqualified.
let b = if !w_exp.is_null() {
    "W_exp-cp.async"
} else if !padded_q6k {
    "in-kernel-expand"
} else if std::env::var("MINFER_MMQ_Q6K_EXP").as_deref() == Ok("0") {
    "in-kernel-expand(exp=off)"
} else {
    "in-kernel-expand(fallback!)"
};
```

The four states' semantics: `W_exp-cp.async` = the fast path; `in-kernel-expand` (unqualified) = raw 210 B weights that never had a plane to build — not a fallback; `exp=off` = the user switched it off deliberately; `fallback!` = should have been built but wasn't — a bug. The label also reports r39's KDR=2 and A-transpose together, so one census reads the whole path.

### 3.3 Pitfalls

- **The label must distinguish "never eligible" from "eligible but not built"**: raw 210 B q6_K weights can never get a plane; if that class were also labeled `fallback!`, the liveness census would drown in legitimate in-kernel paths and the real bug would be invisible. The three-state design's core is separating exactly these two.
- **GB10's aggregate memory reading is `[N/A]`**: on a unified-memory architecture `nvidia-smi`'s total fields report nothing; only `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` reads per-process usage — this trap nearly demoted "measured memory" into "citing r53's census value." The measured 7636/6182 MiB sits 3 MiB from the census, confirming that the per-PID reading is the engine's true footprint.
- **Co-tenant suite noise**: this window's suite once showed 165/2 (a 46 GB sglang co-tenant was running); both cases re-ran individually with `--exact` and passed — judged a co-tenant flake, unrelated to this change, and finally recorded as 167/0/3.
- **r53's map-key lesson is an inherited constraint**: if the map were keyed on the plane's own pointer, dispatch's lookup by padded pointer would miss forever — r54's registration code and three-state label are both designed around "this class of error must be visible on the spot."

## 4. Verification

- **parity ×3, run once per mode**: guards against math errors; per §2.3 this should be a construction guarantee, and parity here is a routine confirmation.
- **greedy byte-identical, three-way comparison**: exp=1 vs exp=0, plus both against r53's recorded greedy stream. Guards against "the path switch changing float accumulation" — theoretically impossible, confirmed in measurement; it also covers the r41 branch's in-service health on the current tree in the exp=0 mode.
- **liveness census 27×/0, both directions**: with exp=1, all 27 q6_K launches are `W_exp-cp.async` with 0 accidental fallbacks; with exp=0, all `exp=off` with 0 `fallback!`. Guards against r53's "parity/greedy all green but the fast path never ran" (back then a wrong map key, caught by exactly this discipline).
- **memory measured per PID**: 7636 vs 6182 MiB. Guards against "assumed byte counts" — r53's "+15 MB" underestimate is what an assumption costs; the measurement matching the census's 1450.9 MiB confirms the opted-out memory truly came back.
- **suite 167/0/3**: guards against cross-feature regressions (including r53's new gate test); the 165/2 co-tenant false positives were excluded per the "isolated `--exact` re-run" procedure.

## 5. Results

| Mode | whole prefill | vs r53 default | device memory (per-PID) | B-path label |
|---|---|---|---|---|
| default / `=1` (exp=on) | 3181.0 tok/s | unchanged (r53 was 3176.9, same noise band) | 7636 MiB | 27× `W_exp-cp.async`, 0 fallbacks |
| `MINFER_MMQ_Q6K_EXP=0` | 3020.7 tok/s | **−5.04%** | 6182 MiB (−1454 MiB ≈ 1.52 GB) | 27× `exp=off`, 0 `fallback!` |

vs-llama (the 3324.4 anchor, campaign accounting) is recorded from r53's 1.05× to 1.04×. Each direction gets what it wants: the default is the verified fastest path; `EXP=0` is a legitimate downshift for memory-constrained cards, and the downshifted path is also a parity/greedy-verified in-service implementation, not a second-class citizen.

Put −1454 MiB in a user's terms: it is about 19% of an 8 GB card's memory budget and 9.5% of a 16 GB card's — for the former, that is the magnitude of "can I also load one more LoRA / one more KV copy," not a negligible rounding error. This is also why the exit valve had to be in place **before** r56/r60 kept adding planes to this chain: every plane added raises the valve's value by a notch.

Two follow-ons that show the switch's shape was right:

- **r56 rides it directly**: the W_dsc plane's registration hangs under the same `Q6K_EXP != "0"` gate, so `EXP=0` opts out of all plane memory in one move (another 363 MB). Had r54 not raised this gate first, stripping the plane after r56 would have meant re-verifying two stacked features.
- **r60 canonized the pattern**: the promotion's definition is exactly "default-on + `0` opt-out" (the r54 pattern) — all six MMQ gates flipped on it, with `MINFER_MMQ=0` kept as the legacy-f16 escape hatch; r54 supplied the first precedent.

## 6. Lessons

1. **A memory-for-speed knob needs the trio**: a byte-identical fallback compiled into the binary, a liveness label that distinguishes "deliberate fallback" from "accidental fallback," and a measured (not derived) memory delta.
2. **Fit the exit valve before the next feature rides the gate**: r56's dsc plane and r60's promotion both inherit this gate directly; stripping the plane afterwards means re-verifying two stacked features.
3. **On GB10, verify memory per PID**: the aggregate reading is `[N/A]`; `nvidia-smi --query-compute-apps` is the instrument that works.
4. **The exit valve's fallback path must be a first-class citizen**: EXP=false is not a degraded implementation but the in-service path since r41 preserved verbatim — which lets the verification budget go entirely into "the path really gets taken."

---
← 56 · [Index](./README.md) · 58 →
