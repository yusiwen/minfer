# 0074. The llama3 rope table is read from the GGUF, not re-derived from four numbers

- Status: Accepted
- Date: 2026-11-05
- Issues: #724

## Context

Llama 3.1/3.2 HF configs declare `rope_scaling.rope_type = "llama3"`: the frequency band is
piecewise by wavelength — the high-frequency pairs are untouched, the low-frequency ones are divided
by `factor`, and the pairs between ramp continuously. #723 taught the engine the rope **layout**
(consecutive pairs for `llama`, half-split for the Qwen family) but left every frequency at
`freq_scale / base^(2d/hd)`, which is right only for models that never declared the scaling.

What a GGUF actually carries was measured rather than assumed (2026-11-05, llama.cpp build b11829 /
`184f65360`):

- llama.cpp's converter writes **one global f32 tensor** `rope_freqs.weight` with `hd/2` entries
  (`conversion/llama.py`, `LlamaModel.generate_extra_tensors`), and its loader resolves that same
  global name — the arch table declares `LLM_TENSOR_ROPE_FREQS` without a block prefix.
- Both cached files carry it: Llama-3.2-1B `[32]` (factor 32, three ramp pairs) and Llama-3.1-8B
  `[64]` (factor 8, seven ramp pairs). Neither carries `rope.scaling.*` metadata, and gguf-py's
  `RopeScalingType` enum has no `llama3` value at all.
- The reference **applies** it: overwriting the 1B file's table with `1000.0` changes llama.cpp's
  greedy output completely, while overwriting it with `1.0` leaves the output byte-identical.

So the contract is "one table per file, f32, `hd/2` factors, each dividing its pair's angle", plus a
metadata spelling (`{arch}.rope.scaling.type = "llama3"` with `factor`, `original_context_length`
and the two band factors) that no llama.cpp file uses but a hand-written or third-party conversion
can.

## Decision

**1. Read the table at load; refuse what is not implemented.** `models::rope_scaling::resolve`
takes the global `rope_freqs.weight` first, then a per-block `blk.N.rope_freqs.weight` (every block
must agree — the engine's rope parameters are model-global, so a file whose blocks disagree is
refused rather than silently using block 0's), then the `llama3` metadata, synthesising the same
table with the converter's formula. An unknown `rope.scaling.type` is refused **by name**, and
`linear`/`yarn`/`longrope` count as unknown: this engine implements none of them, so accepting one
would rotate every position by the wrong angle and read as a numerics bug far from its cause.

**2. The table rides in the node metadata, not as a graph input.** `RoPEMeta`, `FusedQkvMeta`,
`QkvBiasRopeStoreMeta` and `FusedQkvNormMeta` carry `Option<Box<RopeFreqs>>`, and `KvRope` carries
`Option<RopeFreqs>` for the context-shift re-rope. The graph's reuse identity is `CParams`
(`gpu`, `gpu_layers`, `kv_format`); a model-fixed table in the meta cannot enter it, so params-only
reuse is untouched.

**3. The kernels divide, and the formula has one home per implementation.**
`freq(d) = freq_scale / base^(2d/hd) / ff[d]`, written once per implementation —
`vec_ops::rope_inv_freq` (the CPU kernel and the KV shift), `rope_freq` in `metal/kernels/common.h`,
`minfer_rope_freq` in `cuda/kernels/common.cuh` — and mirrored, not re-derived, per call site.

**4. `None` is today's expression, evaluated exactly as before.** The kernels branch on an empty
table rather than reading a precomputed table; an unscaled model is then bit-identical to the
pre-#724 binary, which is what keeps the pinned Qwen and Llama goldens — CPU, CUDA and Metal —
valid without re-recording any of them.

**5. The table crosses to the device as a kernel argument.** A `#[repr(C)]` struct of one `int` and
128 `float`s (515 bytes) goes through Metal's `setBytes` and a CUDA kernel parameter, so no
allocator, scheduler, registry or staging-copy change is needed, and the table cannot outlive the
launch that reads it. `ROPE_FREQS_MAX = 128` matches the bound the CPU kernel already assumed
(`hd ≤ 256`), and a wider table is refused by name rather than truncated.

## Alternatives considered

- **Four numbers in the metadata and the piecewise formula in every kernel.** The issue's shape (a).
  It cannot represent the file's table at all — and the table is the authoritative artifact that the
  reference reads — while duplicating the same 12-line expression in six places with nothing making
  them agree.
- **Precompute the effective inverse frequencies on the host and pass them as a graph input.**
  The issue's preferred shape (b). Rejected: it adds a second input to four op variants across three
  backends' `execute_node`, plus the allocator's liveness, the cross-backend staging copy, the
  fusion matcher, the JSON exporter and the model builders — for at most 64 model-constant floats.
- **Precompute the effective table and read it unconditionally.** Simpler kernels, but it replaces
  the *unscaled* path's arithmetic on both devices (their `pow` becomes our `powf`), which would
  invalidate the device goldens #725 recorded. The branch buys bit-identity for one comparison.
- **Converge on the metadata spelling only** — the shape the issue described. Wrong for every file
  in the wild: both cached Llama GGUFs carry the tensor and no metadata.

## Consequences

- **Long context is where it should be measurable — and the first attempt was not.** The
  `llama_scaled_long_context_matches_the_pinned_reference` gate (8192 prompt tokens, 16 greedy ids,
  on a pinned Llama-3.1-8B file) matches llama.cpp exactly, but it **does not discriminate the
  table**: stubbing the resolver to `Ok(None)` still reproduces the same 16 ids, because the pairs
  the table moves are the lowest-frequency ones and this trajectory's margins are wider than their
  effect at 8192. It is therefore recorded as a long-context *parity* gate; the table's evidence is
  the unit gates (its bytes, the division, the bit-identical unscaled path) and the 1B pin's arms,
  where applying it moves step 5 across a 0.037-logit tie on CPU and CUDA. A discriminating
  real-model gate is a follow-up (a longer context or a needle-style prompt).
- **The 32-token #725 pin loses its token-exact claim on CPU, and its device arms get stronger.**
  With the table applied — as llama.cpp always did — step 5 becomes a 0.037-logit tie in llama.cpp
  itself (10057 over 9822); the CPU arm (quantized activations) lands 0.064 further, on the other
  candidate, while the CUDA and Metal arms now reproduce llama.cpp's 32 ids token for token. The
  CPU arm records that tie (agreeing prefix plus the recorded pair and margins) instead of demanding
  exactness across it; the 8192-token gate carries the exact-id claim. This is a deliberate
  narrowing, not a tolerance widened to hide a failure — and it closes #737: the seven-token cycle
  the device arms repeated was this missing table, not the f32-activation class.
- A third-party file that spells the table per block below `ROPE_FREQS_MAX` is supported; one that
  varies by layer is refused.
- `models/rope_scaling.rs` is the single reader for both spellings, so a future llama.cpp renaming
  costs one constant there.
