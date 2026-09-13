# 102 · Draft-scale sweep: bigger drafts lose — and a namespaced-registration bug the sweep flushed out

**Status**: ✅ measured, default unchanged (2026-09-15). The one untested
speculative-quality axis — draft MODEL SIZE (doc 93 swept draft quants only) —
is now measured: throughput falls monotonically with draft size, because
acceptance is bounded by the target's predictability, not by draft capacity.
The sweep also flushed out a real latent bug: the Qwen3 loader's Q6_K
padded-registration used the raw GGUF tensor name instead of the namespaced
registry key, silently replacing the target's entry and dropping BOTH models
to CPU.

## 1. The sweep (steady-state method, doc 101; Qwen2.5-14B q4_k_m target, greedy)

| draft | vocab family | bytes | code d2 / adaptive (tok/s, −n 200) | prose d2 / adaptive | acceptance (code, −n 100 probe) |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B Q4_K_M (incumbent)** | same | 0.40 GB | 41.6 / **44.8** | **36.6** / 35.9 | **76.5%** |
| Qwen3-0.6B Q8_0 | cross | 0.64 GB | 37.6 / 39.7 | 33.9 / 33.7 | 66.7% |
| Qwen3-1.7B Q8_0 | cross | 1.83 GB | — (−n 100: 34.2 vs 0.6B's 47.5 same-protocol) | — | 63.4% |
| Qwen3-1.7B Q4_K_M (unsloth) | cross | 1.11 GB | **broken file — 0.3% acceptance** | | |

Verdict: **the incumbent stays**. Same-family small drafts win twice — higher
acceptance (76.5% vs 66.7%; the draft-target distribution gap costs more than
capacity buys) and cheaper draft steps. The 100-token probe that first showed
the 0.6B draft "ahead" (47.5) was a generation-length confound: at −n 200 the
ordering inverts. Cross-family identity was re-proven at the new gate (§3):
spec output is byte-identical to sequential 4/4 with the Qwen3 draft.

## 2. The broken file (kept out of the registry of usable drafts)

unsloth's Qwen3-1.7B-Q4_K_M produced **0.3% acceptance** (tokens/round 1.02,
12.4 tok/s — worse than sequential). Diagnosis chain: the file omits
`tokenizer.ggml.bos_token_id` (official Qwen conversions carry bos 151643;
the engine's `unwrap_or(0)` then failed the special-token gate). Adding the
KV with a byte-verified GGUF rewrite (tensor-data fidelity asserted) did not
fix acceptance — the 0.3% is the file itself. The official Qwen conversion of
the same model behaves normally (63.4%).

## 3. Gate change: post-EOS token-text divergence is a warning, not an error

The old gate hard-errorred on ANY token-text mismatch in the shared id range.
Cross-family drafts legitimately differ BEYOND EOS (Qwen3 fills the post-EOS
ids with tool tokens where Qwen2.5 keeps PAD placeholders). Since the spec
loop exchanges token IDs (the draft proposes an id, verify accepts on the
target's argmax id) and decoding always uses the target's vocabulary, text
disagreement there is not a correctness risk — `src/spec.rs` now warns once
with the diverged count and continues; the control region (ids ≤ EOS) stays a
hard error. Identity battery for the cross-family pair: 4/4 byte-identical.

## 4. The bug the sweep flushed out (fixed here)

`src/models/qwen3/loader.rs` registered Q6_K weights in the padded 224-byte
layout under **`ti.name`** (the raw GGUF tensor name) instead of `reg_name`
(the namespaced key). Under the namespaced draft load the draft's Q6_K
`token_embd.weight` therefore REPLACED the target's registry entry for the
same name; the next `build_graph` failed the target's `has_weight_of_size`
all-or-nothing gate and BOTH graphs dropped to CPU (0.8 tok/s, `CUDA GATE:`
messages naming both models' embeddings). The Qwen2 loader already used
`reg_name`; the Qwen3 loader now does too. Any future Qwen3-arch draft (or
any namespaced second model) with a Q6_K tensor would have hit this.

## 5. Method note

The standing battery (`pre_s8_*` baselines) compares build-vs-build with the
SAME draft and flags; a different draft must be identity-tested against
freshly generated sequential output (as in §3), not against the 0.5B-draft
baseline files — the baseline comparison conflates draft change with
identity.

## 6. Disposition

- Fixed: the qwen3 loader namespaced-registration bug (§4); gate relaxation
  (§3). Suite: 187 passed / 0 failed / 3 ignored.
- Default draft unchanged: Qwen2.5-0.5B Q4_K_M.
- Artifacts: `/tmp/d101/` (matrix outputs, GGUF probes, identity batteries,
  the official/unsloth 1.7B files).
