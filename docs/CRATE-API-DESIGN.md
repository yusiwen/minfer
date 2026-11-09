# Crate API design

**Status: current contract (Phase 1 landed, and every gap it named has since closed).** The decision
behind it is [ADR-0079](adr/0079-the-engine-is-a-crate-with-two-api-tiers.md); this page is what a host
can actually call today. The slices that built it are #763's S0–S10 (the epic's comments carry each
landing), and the end of the page names each gap that was left, with the ticket that closed it.

## The tiers

| tier | surface | consumers | stability |
| :--- | :--- | :--- | :--- |
| **L2 facade** | `engine::{Engine, Session, SessionConfig}`, `Session::chat` (delta callback), `Engine::generate` (single-shot), `EngineError`, `ConversationSpec`/`TurnParams` | embedders, the CLI | documented; the only semver-relevant promise before 1.0 |
| **L1 primitives** (`minfer::advanced`) | `ModelDef`, `GraphCache`, `GraphParams`, `Batch`, `KvFormat`, `Backend`/`BackendFilter`, `OffloadRequest`, `Tokenizer`, `SpecialTokens`, `SamplerConfig`/`MirostatMode`, `Grammar`, the conversation types, `SpecEngine` | the HTTP server, bench, tooling | public, **not** semver-stable before 1.0 |
| **L0 observation** (`observe`) | `live`, `trace`, `graph::json`, `optiming` — re-exported from `minfer::advanced` (`live` only with `observe`) | `viz`, `/metrics` | feature-gated; off in the default consumer build |

The rule that keeps the tiers honest: an embedder that needs L1 is telling us the facade is missing a
capability; the server is the only consumer allowed to be on L1 by design, and its imports are the
tier (`src/server/**` reaches these types only through `crate::advanced`).

## Ownership and lifetime

- `Engine` owns the loaded `GgufModel` (and therefore the mapping), the model (`Box<dyn ModelDef>`),
  the tokenizer and the **validated** chat template. `Engine::load` / `load_with` are the only
  constructors; a load failure is an `EngineError` naming the GGUF, model, tokenizer or template step.
- A `Session` borrows the engine and owns one conversation: its `Conversation` (messages, positions)
  and its `GraphEngine` (the KV cache). Two sessions on one engine share nothing.
- The model owns the mapping it reads (ADR-0076). Dropping an `Engine` releases the mapping, the
  device registrations (ADR-0077/0078) and the KV arena; a process may then load another model. The
  reload gate is `src/gguf/tests.rs`.
- Loading takes a **path or bytes**. `Engine::load_from_bytes(Vec<u8>)` (#797) serves a host that
  already holds the model — received over the network, embedded, or read from a volume it decrypted
  itself — under the same contract: the buffer is **not** copied (every weight is a range of it, as a
  path load's weights are ranges of the mapping), and a split model is refused by name because its
  siblings are resolved beside the file and a bytes load has no directory. On macOS an owned part is
  copied once into its `MTLBuffer` (`newBufferWithBytesNoCopy` needs page-aligned pages whose lifetime
  the engine can guarantee), which is the same one-buffer-per-part shape the mapped path uses.

## Configuration

`EngineConfig` (`src/config.rs`) is the process configuration: the CLI reads `MINFER_*` through
`EngineConfig::from_env()` and installs it once, before any device or graph exists. The engine reads
`config::get()`; a consumer that installs nothing gets the built-in defaults, never the host's
environment. Since [#769](https://github.com/yusiwen/minfer/issues/769) that covers the device
backends and the model graph builders too: CUDA's and Metal's kernel-policy knobs (`mmq_*`,
`flash_attn`, `gemm`, `mv_mma*`, `packed_kv_stage`, …) are `EngineConfig` fields, and no file under
`src/cuda/`, `src/metal/`, `src/cuda.rs`, `src/metal.rs`, `src/graph/metal_backend.rs` or
`src/models/*/graph.rs` reads `MINFER_*` outside `from_env`. The residual readers are the observation
modules (`trace`, `dump`, `optiming`) and the speculative-decoding seam (`spec`, `spec_verify`), which
still read their own variables — the honest edge of #769's scope. On the test side,
`config::with_config` and `registry::with_filter` scope a value to one thread so two engines can
differ.

| field | meaning | environment spelling |
| :--- | :--- | :--- |
| `backends` | the fence this run may use | `--backend` / `MINFER_BACKENDS` |
| `kv_cache_type` | KV storage format (`f32`/`f16`/`q8_0`) | `MINFER_CACHE_TYPE` |
| `gpu_layers`, `gpu_mem` | offload request and weight budget | `MINFER_GPU_LAYERS` / `MINFER_GPU_MEM` |
| `fuse_qkv`, `fuse_ffn`, `ffn_composition` | the fusion gates | `MINFER_NO_FUSE_QKV` / `MINFER_NO_FUSE_FFN` / `MINFER_FFN_COMPOSITION` |
| `avx2`, `avx512`, `neon`, `f16_rowb` | ISA overrides | `MINFER_NO_AVX2` / `MINFER_NO_AVX512` / `MINFER_NO_NEON` / `MINFER_NO_F16_ROWB` |
| `context_shift` | C2's overflow path | `MINFER_NO_CONTEXT_SHIFT` |

`n_ctx` and the seed are per session (`SessionConfig`), not process configuration: two conversations
may size their arenas differently. `threads` stays a process-wide pool (`kernel::set_cpu_threads`).

**The one-process contract.** The device context (Metal, CUDA), the CPU worker pool and the
observability sinks are process-wide; the first `config::install` wins and a later call is ignored.
Everything that can differ between engines — the backend fence, the KV format, the context size, the
offload plan — is per engine, and the fence can be scoped (`registry::with_filter`).

## Failures

- Every load returns a named error: `EngineError::{Gguf, Model, Tokenizer, Template}`, wrapping
  `GgufLoadError`, `ModelLoadError` (arch + reason), the tokenizer's refusal or `TemplateError`.
  `ConvError` implements `std::error::Error`, so a host can use `?` on a turn.
- The load and generation paths never call `process::exit`, never panic on malformed input and never
  write to the host's stderr; the CLI renders what the engine returns.
- A generation that hits a limit (tokens, EOG, stop string, grammar refusal) returns a
  `GenerateOutcome` that says so; it is not an `Err` unless the request itself was refused.
- A caller can stop a run between tokens: `GenerateParams.cancel` carries a `CancelToken`, checked
  once per decoded token and once per speculative round, and the outcome then reports `cancelled`
  (never `hit_n_predict`). The check **cannot** interrupt a forward pass — a single call into the
  model — so a cancel that arrives mid-pass is honoured at the next token. The CLI wires Ctrl-C to
  this seam and exits 130 (#796).

## Features

| feature | brings | default |
| :--- | :--- | :--- |
| `cli` | the `minfer` binary; implies `server`, `tooling`, `viz` | on |
| `server` | the OpenAI-compatible server (axum + tokio) | via `cli` |
| `tooling` | `convert` / `quantize` / `split` | via `cli` |
| `viz` | the `viz` subcommand and its assets (implies `server`) | via `cli` |
| `observe` | the `live`/`trace` hooks (tokio `sync`) | via `server` |
| `metal` | the macOS Metal backend, and its `objc2` dependencies | on (inert off macOS) |
| `cuda`, `cuda_static`, `debug_dump` | unchanged | off |

The binary declares `required-features = ["cli"]`, so `cargo build --release` produces it while
`--no-default-features` builds the core alone. The engine-only configuration is **testable** too: `cargo test
--no-default-features --lib` compiles and runs it — the bf16 encoder lives in `block` with the word
format it inverts, so no engine test reaches into a `cli` module
([#791](https://github.com/yusiwen/minfer/issues/791)). The `feature-matrix` CI job builds the core, each
application feature alone, and the `cargo package` tarball. `metal` is a feature since
[#777](https://github.com/yusiwen/minfer/issues/777): compiled only with
`cfg(all(target_os = "macos", feature = "metal"))`, default-on, and inert off macOS. Without it a
macOS build resolves no `objc2` crate and calls no `xcrun`; `--no-default-features --features cli`
still produces the binary, which then refuses `--backend metal` by name (ADR-0080).

## The consumer's view

```toml
[dependencies]
minfer = { version = "0.0.9", default-features = false }
```

```rust
use minfer::conversation::TurnParams;
use minfer::engine::{Engine, SessionConfig};
use minfer::sampler::SamplerConfig;

let engine = Engine::load(std::path::Path::new("model.gguf"))?;
let mut session = engine.session(SessionConfig::default());
let params = TurnParams {
    n_predict: 64,
    sampler: SamplerConfig::default(),
    stop_strings: Vec::new(),
};
let mut out = Vec::new();
session.chat("hello", &params, &mut |delta| out.extend_from_slice(delta))?;
```

## Sessions

`Session::chat` runs a **turn**: the input is rendered through the model's template, the delta is
prefilled after the rows the session already holds, and the reply is recorded in the history.
`Session::n_past()` reports the KV cursor, so a host can log or bound what a session holds.

`Session::complete` runs a **raw continuation**: the tokens are prefilled at that cursor (no template,
no message recorded) and decoded like a turn, so a long prompt can be fed in pieces — the second call
charges only its own tokens, and `TurnOutcome::prefill_tokens` says how many. The continuation is not
history: `messages` stays the authority, and the next `chat` rebuilds the KV from it through the full
re-render path, so a continuation never leaks into a later turn. A session carrying a speculative
draft refuses a continuation by name (`ConvError::CompleteWithSpec`): the draft's KV is indexed by the
same positions and half-driving the pair would desynchronise it.

`examples/embed.rs` is this program; `tests/library_api.rs` runs the same surface from outside the
crate (load → chat → drop → reload, two sessions independent, and a single-shot `Engine::generate`).

## What Phase 1 did not do

Every gap this list named has landed, each with the ticket that closed it:

- `metal` as a feature ([#777](https://github.com/yusiwen/minfer/issues/777)) — §Features.
- cancellation ([#796](https://github.com/yusiwen/minfer/issues/796)) — §Failures.
- the raw continuation `Session::complete` ([#795](https://github.com/yusiwen/minfer/issues/795)) —
  §Sessions.
- the bytes constructor `Engine::load_from_bytes` ([#797](https://github.com/yusiwen/minfer/issues/797))
  — §Ownership and lifetime.

The only deferred work left is Phase 2 below, behind the epic's three named triggers.

## Phase 2 — the workspace split (deferred)

Phase 1 keeps every path. Phase 2 moves the app layer (`main`, `server`, `tooling`, `convert`,
`quantize`, `download`, `bench`) into `crates/minfer-app`, keeps the engine package at the repository
root, and version-publishes `minfer` and `minfer-cli` separately. It is a move plus a scripted anchor
pass, entered only when a named trigger fires (see the epic).

## Non-goals

- No inference-semantics change; the identity gates are the guard.
- No whole-module `pub`; only the tiers are exported.
- The server is not rewritten onto the facade.
- `deny(warnings)` does not move into the library.

## Decisions governing this document

- [ADR-0079](adr/0079-the-engine-is-a-crate-with-two-api-tiers.md) — The engine is a crate with two API tiers, and the apps are its first consumers
- [ADR-0009](adr/0009-a-failure-is-an-error-never-a-silent-fallback.md) — A failure is an error, never a silent fallback
- [ADR-0006](adr/0006-kv-format-is-a-per-engine-gate.md) — The KV storage format is a per-engine gate, not a process-wide global
- [ADR-0076](adr/0076-a-tensor-s-bytes-are-a-mapping-handle-plus-a-range.md) — A tensor's bytes are a mapping handle plus a range
