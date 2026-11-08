# Crate API design

**Status: Phase 1 target.** This page is the contract epic
[#763](https://github.com/yusiwen/minfer/issues/763) is building towards. Each section names the
slice that lands it; nothing below is shipped behaviour until that slice merges, and the page is
updated in the same PR that changes a section.

The decision behind it is [ADR-0079](adr/0079-the-engine-is-a-crate-with-two-api-tiers.md). The
classic contract (what an embedder can call, what it may not) lives here; the ADR holds why the tiers
exist and what was rejected.

## The tiers

| tier | surface | consumers | stability |
| :--- | :--- | :--- | :--- |
| **L2 facade** | `Engine::load`, `Session`, `complete`, `chat`, the delta callback, cancellation | embedders, the CLI | documented; the only semver-relevant promise before 1.0 |
| **L1 primitives** (`minfer::advanced`) | `ModelDef`, `GraphCache`, `graph::Batch`, KV sessions, `Tokenizer`, `SamplerConfig`, `Grammar` | the HTTP server, bench, tooling | public, **not** semver-stable before 1.0 |
| **L0 observation** (`observe` feature) | `live`, `trace`, `graph::json`, `optiming` — re-exported from `minfer::advanced` (`live` only with `observe`) | `viz`, `/metrics` | feature-gated; off in the default consumer build |

The rule that keeps the tiers honest: an embedder that needs L1 is telling us the facade is missing a
capability; the server is the only consumer allowed to be on L1 by design.

## Ownership and lifetime

- `Engine` owns `Arc<dyn ModelDef>`, the tokenizer, the chat template and the special-token set. A
  `Session` borrows nothing from the caller: it owns its KV state (`GraphCache`) and its
  `Conversation`.
- The model owns the mapping it reads (ADR-0076). Dropping an `Engine` releases the mapping, the
  device registrations (ADR-0077/0078) and the KV arena; the process may then load another model.
  This is the property #745 established and the reload gate asserts.
- `Engine::load` accepts a path and, once S6 lands, an owned byte buffer. The buffer form is what a
  host that receives a model over the network needs; it must not require the caller to leak it.

## Configuration

`EngineConfig` replaces the environment inside the crate (S4). The CLI reads the environment and
maps it; `docs/USAGE.md` remains the environment's home.

| field | meaning | today's source |
| :--- | :--- | :--- |
| `n_ctx` | KV arena size | `--n-ctx` |
| `kv_format` | `f32` / `f16` / `q8_0` | `MINFER_CACHE_TYPE` |
| `backends` | the backend fence for this engine | `--backend` / `MINFER_BACKENDS` |
| `offload` | `gpu_layers` / `gpu_mem` | `--gpu-layers` / `MINFER_GPU_LAYERS` / `MINFER_GPU_MEM` |
| `threads` | CPU worker count | `--threads` |
| `fusion` | QKV/FFN fusion toggles | `MINFER_NO_FUSE_QKV` / `MINFER_NO_FUSE_FFN` |

**The one-process contract.** The device context (Metal, CUDA), the CPU worker pool and the
observability sinks are process-wide. Everything that can differ between engines — KV format, the
backend fence, the context size, offload — is per engine (S5 makes the fence per engine; the others
already are). Two engines that ask for conflicting process-wide state get an error naming the
conflict, never a silent second install.

## Failures

- Every load returns `Result<_, minfer::Error>` with a reason. The model layer stops answering
  `Option` plus `eprintln!` (S3); the GGUF layer already answers a named error.
- Library modules never call `process::exit`, never panic on malformed input, and never write to the
  host's stderr. Diagnostics go through `tracing`-style observation only when `observe` is on.
- A generation that hits a limit (tokens, context, cancellation) returns a value that says so; it is
  not an `Err` unless the request itself was refused.

## Features

| feature | brings | default |
| :--- | :--- | :--- |
| `cli` | the `minfer` binary | on |
| `server` | the OpenAI-compatible server | on |
| `tooling` | `convert` / `quantize` / `split` | on |
| `download` | HuggingFace / Ollama resolution | on |
| `viz` | the `viz` subcommand and its assets (implies `server`) | on |
| `observe` | `live` / `trace` hooks | on for the binary; off for consumers |
| `metal` | the Metal backend (macOS only) | on where it compiles |
| `cuda`, `cuda_static`, `debug_dump` | unchanged | off |

The binary declares `required-features = ["cli"]`, so `cargo build` at the repository root produces it
while `--no-default-features` builds the core alone.

## The consumer's view

```toml
[dependencies]
minfer = { version = "0.1", default-features = false }
```

```rust
let engine = minfer::Engine::load("model.gguf", minfer::EngineConfig::default())?;
let mut session = engine.session();
session.complete("hello", &minfer::GenParams::default(), &mut |delta| {
    print!("{}", String::from_utf8_lossy(delta));
})?;
```

This is the example S9 ships (`examples/embed.rs`) and the shape `tests/library_api.rs` exercises:
load → generate → drop → load a **different** model in one process.

## Phase 2 — the workspace split (deferred)

Phase 1 keeps every path. Phase 2 moves the app layer (`main`, `server`, `tooling`, `convert`,
`quantize`, `download`, `bench`) into `crates/minfer-app`, keeps the engine package at the repository
root, and version-publishes `minfer` and `minfer-cli` separately. It is a move plus a scripted anchor
pass, and it is entered only when a named trigger fires (see the epic).

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
