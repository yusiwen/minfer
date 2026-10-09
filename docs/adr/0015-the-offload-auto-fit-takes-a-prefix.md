# 0015. The offload `auto` fit takes a prefix, not a knapsack

- Status: Accepted
- Date: 2026-09-23
- Issues: #46

## Context

Layer offload first shipped as a **ceiling**: `--gpu-layers N` / `MINFER_GPU_LAYERS=N` told the
engine how many blocks to place on the device, and the user had to know the model's block count and
guess how many fit. Device participation was all-or-nothing before that (`Qwen2Model::device` asked
"is every weight registered on the GPU?"), so *"a model that does not fit in device memory could not
run at all"* — on a machine whose free memory is smaller than the model, the engine had nothing to
offer.

The missing half is to **compute** the split from the device's memory.

## Decision

Add an `auto` spelling (`--gpu-layers auto` / `MINFER_GPU_LAYERS=auto`) that fits the **largest
contiguous prefix** of blocks into a device weight budget measured from the GGUF index **before
anything is loaded**.

- Per-block bytes come from `GgufTensorInfo::nbytes` — the decision cannot wait for a measurement,
  because *the registration filter **is** the plan*.
- `fit_blocks(budget, per_block, reserve)` takes the largest `k` with
  `reserve + Σ per_block[0..k] ≤ budget`.
- The budget is `MINFER_GPU_MEM=<MiB>` if set, else **three quarters** of what the device reports
  free; a quarter is held back as `reserve` for the KV arenas and the activation pool.
- A spelling that is not a block count and not `auto` is a **refused load**, never a guess.
- `device_holds_unblocked` puts embedding / final norm / `lm_head` on the device only when **every**
  block is offloaded (llama.cpp's `n_gpu_layers > n_layer` convention).

## Alternatives considered

**A prefix, not a knapsack.** The recorded reason, from both the plan and the code comment in
`fit_blocks`: the plan is `0..gpu_layers` **by construction**, and a gap would put a CPU block between
two device blocks for nothing. The walk therefore stops at the first block that does not fit — a
deliberate, documented conservatism that accepts a smaller fit in exchange for a plan that stays
contiguous.

The rejected alternative is therefore a knapsack/bin-packing fit (or any non-contiguous block set):
it could place more bytes, at the cost of an offload plan with holes in it and a gap in execution
that buys nothing.

**Measured, with a mutation** (GB10, 0.5B q4_0):

- with `MINFER_GPU_MEM=64` the fit is a **strict prefix** — 5 of 24 blocks, 40.0 MiB of device
  weights, inside the 48 MiB the budget left for weights;
- the model's **four greedy steps match the all-CPU run**;
- with no cap the same request selects all 24 blocks;
- making `fit_blocks` ignore the budget fails the `auto` gate on its first assertion
  (`a 64 MiB budget must be a strict prefix, got gpu_layers: 24`).

## Consequences

- A model larger than device memory runs without the user knowing its block count, and the startup
  report (`auto_source`) names the numbers that produced the plan.
- The prefix conservatism can **under**-offload. The fit measures **raw tensor bytes** and omits the
  auxiliary device copies the loader builds (the fused `attn_qkv`/`ffn_gu` concats, the padded Q6_K
  layout, the q8_0 p32 split, the q4_K `dsc` pair), so an under-estimate ends as a **refusal**, never
  as a silent overcommit.
- The reserve is a fixed quarter rather than a function of `n_ctx`/`n_batch`, so a long-context
  request on a tight budget may be refused by the E4 gate even though `auto` accepted the weights.
- At the time of the decision Metal had no free-bytes query, so `auto` fit nothing there without
  `MINFER_GPU_MEM`; [#53](https://github.com/yusiwen/minfer/issues/53) later gave it one
  (`MTLDevice.recommendedMaxWorkingSetSize`).

## References

- `docs/MEMORY-POLICY-DESIGN.md` §2 — the current contract for the offload plan.
- `ARCHITECTURE-EXECUTION-PLAN.md` — "E5 record, S1 (2026-09-23) — a layer-granular offload plan" and
  "E5 record, S2 (2026-09-23) — the offload plan as a fit, not a ceiling".
- `src/graph/offload.rs` — `fit_blocks`, `OffloadRequest::plan`, `weight_budget`,
  `device_holds_unblocked`.
