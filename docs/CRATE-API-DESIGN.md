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

**The graph cache is per engine too** ([#799](https://github.com/yusiwen/minfer/issues/799)). A
`GraphCache` owns the persistent KV regions, sized once by `n_ctx` and the model's KV format, so two
owners that share one is the bug the allocator refuses at run time. The owner is therefore an
argument, never a global: `Engine::generate` keeps one per engine and rebuilds it when `n_ctx`
changes, a `Session` owns one through `GraphEngine`, a server slot owns one, and `ModelDef`'s single
forward entry (`forward_graph_cached`) takes it.

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
  this seam and exits 130 (#796). The conversation turn loop takes the same token as
  `TurnParams.cancel` and reports it as `TurnOutcome.cancelled`
  ([#800](https://github.com/yusiwen/minfer/issues/800)); there the CLI interrupts one turn and keeps
  the session.

## Features

| package | feature | brings | default |
| :--- | :--- | :--- | :--- |
| `minfer` | `metal` | the macOS Metal backend, and its `objc2` dependencies | on (inert off macOS) |
| `minfer` | `observe` | the `live`/`trace` hooks (tokio `sync`) | off |
| `minfer` | `cuda`, `cuda_static`, `debug_dump` | unchanged | off |
| `minfer` | `test-hooks` | the test-only observers another package's tests read | off |
| `minfer-cli` | `metal`, `cuda`, `cuda_static`, `debug_dump` | one line each, forwarded to `minfer` | `metal` |

Phase 2 ([#805](https://github.com/yusiwen/minfer/issues/805), ADR-0082) turned the fences into a package
boundary, so the four names that only selected which modules of the *same* package were compiled —
`cli`, `server`, `tooling`, `viz` — are gone; the package is the switch, and `crates/minfer-cli` always
contains the binary, the server, the tooling and the visualizer. What the feature matrix proves is now
per package: `cargo check -p minfer --no-default-features` (the core alone), `-p minfer --features
observe`, `-p minfer-cli --no-default-features` (the app's own sliver), and `cargo package -p minfer`.

The engine-only configuration is **testable**: `cargo test -p minfer --no-default-features --lib` compiles
and runs it — the bf16 encoder lives in `block` with the word format it inverts, so no engine test reaches
into an application module ([#791](https://github.com/yusiwen/minfer/issues/791)). `metal` is a feature
since [#777](https://github.com/yusiwen/minfer/issues/777): compiled only with
`cfg(all(target_os = "macos", feature = "metal"))`, default-on, and inert off macOS. Without it a macOS
build resolves no `objc2` crate and calls no `xcrun`; `-p minfer-cli --no-default-features` still produces
the binary, which then refuses `--backend metal` by name (ADR-0080).

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

The one piece Phase 1 deliberately deferred was the package boundary; it is the record below.

## Phase 2 — the split, as it landed

Phase 2 ([#805](https://github.com/yusiwen/minfer/issues/805), ADR-0082) moved the application layer
(`main`, `server`, `tooling`, `convert`, `download`, `bench`, `spec_verify`) into `crates/minfer-cli`,
kept the engine package at the repository root, and turned the four module-selecting features into a
package boundary. Two points changed from the original sketch, both because the measurement said so:
`quantize` stayed in the engine ([`src/lib.rs`](../src/lib.rs) exports it unconditionally — it is the
encoder, not a subcommand), and the package is `minfer-cli`, not `minfer-app`. Only the engine is
published; the CLI ships as a release artifact.

The mechanics, for the next reader: 564 cross-package `crate::` references rewritten to `minfer::`; nine
`pub(crate)` observers and five `cfg(test)`-only surfaces exposed through one `test-hooks` feature that
only the CLI's dev-dependencies enable; 51 ADR citations pinned, because ADRs are frozen; 75 anchors
re-pointed and seven corrected to the symbol's new line. The suite ledger stayed a single merged row
(ADR-0082 decision 4) and **did not move**: 708 / 0 / 48 unit and 44 / 0 / 6 integration, before and
after the move.

## Non-goals

- No inference-semantics change; the identity gates are the guard.
- No whole-module `pub`; only the tiers are exported. The one deliberate exception is the
  `test-hooks` feature, which exposes test-only observers to a dependent package's tests
  (ADR-0082) and is off in every build but that package's test build.
- The server is not rewritten onto the facade.
- `deny(warnings)` does not move into the library.

## Decisions governing this document

- [ADR-0079](adr/0079-the-engine-is-a-crate-with-two-api-tiers.md) — The engine is a crate with two API tiers, and the apps are its first consumers
- [ADR-0009](adr/0009-a-failure-is-an-error-never-a-silent-fallback.md) — A failure is an error, never a silent fallback
- [ADR-0006](adr/0006-kv-format-is-a-per-engine-gate.md) — The KV storage format is a per-engine gate, not a process-wide global
- [ADR-0076](adr/0076-a-tensor-s-bytes-are-a-mapping-handle-plus-a-range.md) — A tensor's bytes are a mapping handle plus a range
- [ADR-0081](adr/0081-a-tensor-s-bytes-are-a-backing-handle-plus-a-range.md) — A tensor's bytes are a backing handle plus a range, and the backing may be an owned buffer
- [ADR-0082](adr/0082-the-application-layer-is-its-own-package.md) — The application layer is its own package, and the engine's four application features retire
