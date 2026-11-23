# 0084. A backend declares whether it can store a packed cell from the fused epilogue

- Status: Accepted
- Date: 2026-11-22
- Issues: #844, #843, #310

## Context

C4 gave the KV region a second storage format and the registry a **capability** for it:
`BackendCaps::reads_packed_kv` is the single authority for "this backend's attention kernel can
read a `q8_0` cell", read by `GraphAllocator::ensure_kv` and `KvFormat::supports`
([#87], [#99], [#153]). That capability answers the *read* half of the question.

The **write** half has no capability, and the decode QKV fusion needs it. `Op::FusedQKV` (and
Qwen3's `Op::FusedQkvNorm`) compute q, k and v in one node and store K and V themselves — so a
fused node must be able to write a *packed* cell. Two model builders therefore ask a question
they cannot name, and answer it from the build. In `src/models/qwen2/graph.rs` the answer is
`qkv_epilogue_ok`, computed as "this binary compiled CUDA and a device is present", and the fusion
gate is `(!b.kv_is_packed() || qkv_epilogue_ok)`; in `src/models/qwen3/graph.rs` the same question
is `fused_norm_packed_ok = device(model) == Device::Cuda`, with the gate `(!b.kv_is_packed() ||
fused_norm_packed_ok)`.

"Was this binary built with CUDA?" and "is this the CUDA device?" stand in for "can **this
backend** store a packed cell from a fused epilogue?". On Metal the answer is *no by
construction*, so a packed cache always takes the **unfused** chain — 3 `MatMul` + 2 `RoPE` +
2 `KvcacheStore` per layer instead of one node.

Measured on `macbookpro` (Apple M4 Pro, macOS 27.2.1, tree `f679f5f`, `-p 8` — a tiny KV, so
everything left is per-token cost, `-r 3`, medians of 3 interleaved rounds):

| arm | nodes/step | µs/token |
|---|---|---|
| f16, fused | 243 | **3 033** |
| f16, `MINFER_NO_FUSE_QKV=1` | 387 | **3 589** |
| q8_0 (fusion refused by the gate) | 363 | **3 712** |

Giving the *f16* arm the unfused chain costs **556 µs/token** — that is the price of the extra
nodes alone, with no packed store in the run — and the packed arm's **679 µs** gap is **82%**
that plus 123 µs of packed-path work (the store, [#844]'s P1, recovered ~60–70 µs of it; the
read path measured ~1%). The residual is graph topology, not kernel time.

CUDA has the fused packed epilogue (`attn_bias_rope_store_q8_0`, [#843]/E1: 21.6 → 1.92 µs/layer,
byte-identical). Metal's `kernel_attn_rope_store` writes f32/f16 only. The CPU implements no
`Op::FusedQKV` at all — `cpu_backend.rs` refuses the op by name — so on a CPU layer the question
never arises (`layer_gpu` is false there too).

## Decision

1. **A packed cell's fused-epilogue store is a declared backend capability**, carried in
   `BackendCaps` beside `reads_packed_kv`: one authority, one place to read it, the `Backend`
   trait's methods forwarding to the same answer. Name it for what it allows, e.g.
   `writes_packed_kv_from_fused_epilogue`.
2. **It is conditional on the format, and distinct from `supports_op(Op::FusedQKV)`.** CUDA and
   Metal both claim that op; the question is whether the *store arm* accepts a packed cell. A
   backend may therefore claim the op and answer this capability `false`.
3. **The graph builders read the capability, never a `cfg`, a device name, or a global.**
   `qwen2`'s `qkv_epilogue_ok` and `qwen3`'s `fused_norm_packed_ok` are the same question and must
   be answered by the same call, in the same change.
4. **A backend answering `false` keeps the unfused chain** (3 matmul + 2 rope + 2 store). The
   chain is the fallback, not a degeneration: it is what every backend gets today when the KV is
   packed.
5. **Frozen answers, pinned by a pure-function matrix test** in the style of
   `kvformat::tests::the_auto_policy_is_a_pure_function_of_the_dims`:
   - **CUDA `true`** — it has the fused packed epilogue; a refactor must not silently take it away;
   - **CPU `false`, by design** — no `Op::FusedQKV` implementation exists to extend, and this ADR
     does not add one;
   - **f16/f32 unaffected** — the gate applies only when the KV is packed.
6. **The capability flips only in the change that lands the kernel.** A backend that answers
   `true` without a packed store arm would write f16 K/V into a region sized for `q8_0` — half the
   bytes, an out-of-region write rather than a wrong number. Metal's answer stays `false` until
   its fused epilogue has the arm.

## Alternatives considered

- **Flip the gate at graph level and measure the node-count win without the kernel.** Refused: the
  fused node would write f16 K/V into a packed region (34 B per 32 values, half the f16 width), so
  the run would corrupt memory outside the region and its timing would be meaningless. The number
  was obtained safely instead by giving the *f16* arm the unfused chain.
- **Route packed-on-Metal to `Op::QkvBiasRopeStore`** (3 matmuls + one epilogue node, the CUDA
  "class 2" path) as a smaller first step. Refused as the first step, kept as the fallback:
  Metal's `supports_op(QkvBiasRopeStore)` is `false` (only the fused path exists), so it is a
  second new kernel for a smaller win — 4 nodes/layer instead of 1.
- **Keep the CUDA test and add a Metal special case** (`device == Metal && …`). Refused: a list of
  device names in model code is the exact shape [#153] removed for the layout tag. A capability is
  what a device list is trying to be.
- **Teach the CPU the packed fused epilogue so the answer is `true` everywhere.** Refused on scope:
  the CPU has no `Op::FusedQKV` at all; giving it one (and a packed store inside it) is its own
  ticket, and this decision does not depend on it.

## Consequences

- **This ADR changes no behaviour by itself.** The behaviour delta is one row of the capability
  matrix (a packed cache on a backend that declares the store), and it is gated by [#844]'s
  pre-registered bar: `-p 8` q8_0 ≤3 300 µs/token (from 3 712), `-p 2048` gap ≤300 µs (ratio
  ≤1.09), hd 128 `-p 8` gap ≤350 µs, packed cells **byte-identical**, CUDA/f16/f32 frozen, serial
  suites green, and `MINFER_NO_FUSE_QKV=1` still reproducing 3 589 µs as the control that keeps the
  measurement honest.
- **Adding the field is internal.** `BackendCaps` is not re-exported (`advanced` exports
  `Backend`/`BackendFilter` only), so the crate's public contract and its tier rules do not move.
- **The contract becomes describable.** `docs/KV-CACHE-DESIGN.md` §3 states the format gate and
  names `reads_packed_kv` as *the* capability; it gains the second one in the same PR, so a reader
  learns both halves in the place the first was written down.
- **A future backend cannot forget it is unfused.** Today the answer is unreachable from the code
  (a `cfg` and a device comparison); after this it is a declared value with a test, and a new
  backend that omits it fails the matrix rather than silently running the slow chain.
- **Two similarly-named capabilities are accepted**, with the cost stated: `reads_packed_kv` (does
  attention read a packed cell) and `writes_packed_kv_from_fused_epilogue` (does the fusion store
  one). One sentence each in the design doc and the matrix test keep them apart.
- **The kernel work is not justified by this ADR.** If Metal's packed store arm cannot be written
  at a cost comparable to f16's, the capability stays `false`, the fallback chain stays, and the
  measured 123 µs of packed-path work is the only thing this ticket closes on.

## References

- The format gate and its capability: [`docs/KV-CACHE-DESIGN.md`](../KV-CACHE-DESIGN.md) §3, and
  [`docs/SUPPORT-MATRIX.md`](../SUPPORT-MATRIX.md)'s packed-cache rows.
- The measurement: record 112 (the decode residual and the read path), the [#844] attribution
  comments (the context sweep, the store, the per-node table, the f16 control).
- The CUDA twin this mirrors: [#843]/E1 and `attn_bias_rope_store_q8_0`; the Metal kernels:
  `src/metal/kernels/qkv_fused.metal` and `src/metal/kernels/kv.metal`.
- The capability precedent: [#87], [#99], [#153], [`ADR-0006`](./0006-kv-format-is-a-per-engine-gate.md).

[#87]: https://github.com/yusiwen/minfer/issues/87
[#99]: https://github.com/yusiwen/minfer/issues/99
[#153]: https://github.com/yusiwen/minfer/issues/153
[#310]: https://github.com/yusiwen/minfer/issues/310
[#843]: https://github.com/yusiwen/minfer/issues/843
[#844]: https://github.com/yusiwen/minfer/issues/844
