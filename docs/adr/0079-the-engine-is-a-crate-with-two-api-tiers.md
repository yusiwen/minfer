# 0079. The engine is a crate with two API tiers, and the apps are its first consumers

- Status: Accepted
- Date: 2026-11-07
- Issues: #763

## Context

The inference engine is already a separable layer: no module under `src/gguf.rs`, `src/tensor.rs`,
`src/models/`, `src/graph/`, `src/kernel/`, `src/quants/`, `src/vec_ops/`, `src/sampler/`,
`src/grammar/`, `src/template/`, `src/tokenizer/` or `src/conversation.rs` names `crate::server`,
`crate::tooling`, `crate::bench` or `crate::main`. What it is not is a *crate*: there is no library
target, the only "complete this prompt" path is inline in `src/main.rs`, and the model-loading layer
still answers `Option` plus `eprintln!` (`src/models/mod.rs`).

#745 removed the one structural blocker — a model owns the mapping it reads, so `load → forward →
drop → load another` works inside one process (`src/gguf/tests.rs`). Deciding the public contract is
therefore no longer blocked by a data-structure problem, but it still has to be decided before code
freezes the wrong shape:

- Publishing the whole project as one crate makes every consumer pay for `axum`, `tokio` and, on
  macOS, the Metal toolchain — and nothing proves the isolation.
- Exposing a single high-level facade cannot serve the HTTP server: continuous batching, slots and
  prefix sharing are driven through `ModelDef::forward_batch`, `graph::Batch` and `GraphCache`, which
  a "generate one prompt" API does not reach.
- Moving the code into a workspace first is the correct end state, but it relocates ~30,000 lines and
  breaks roughly 1,500 documentation anchors before the public API exists to justify the move.

## Decision

**1. Two API tiers, with different stability promises.**

- **L2 — the facade.** `Engine` (load), `Session` (a conversation with its own KV state),
  `complete`/`chat`, a token-delta callback and cancellation. This is the documented, semver-relevant
  surface; an embedder should never need anything below it.
- **L1 — the primitives.** `ModelDef`, `GraphCache`, `graph::Batch`, the KV session format, the
  tokenizer, the sampler and the grammar compiler, exported as `advanced`. Public because the HTTP
  server is a first-class consumer, and explicitly **not semver-stable before 1.0**.
- **L0 — observation.** `live`/`trace`/`graph::json`/`optiming` behind an `observe` feature, so the
  core crate does not require `tokio` (today `src/graph/scheduler.rs` calls `live`, and
  `src/live.rs` owns a `tokio::sync::broadcast`).

**2. One package, feature-fenced, before any directory move.** Phase 1 keeps every path in place:
core modules are always compiled, app modules (`cli`, `server`, `tooling`, `download`, `viz`) are
features, and the binary declares `required-features = ["cli"]`. A workspace split is Phase 2, and it
is entered only when a named trigger fires (an external consumer needing an independent version
cadence, a publish that must *prove* HTTP dependencies cannot enter a consumer's graph, or the app
layer needing its own release artifacts). Phase 2 moves the app layer to `crates/minfer-app` while the
engine stays at the repository root, which keeps the core anchors valid.

**3. One process holds one device and one backend configuration — expressed per engine.** `CudaState`,
the Metal context and the CPU worker pool remain process-wide; the backend *fence*, which is today a
once-only global (`src/graph/registry.rs`), becomes a field of the engine configuration so two engines
in one process can differ. A second, conflicting install must be an error, never a silent no-op.

**4. The CLI is the facade's first consumer; the server is the advanced tier's first consumer.** The
single-shot prefill/decode loop in `src/main.rs` moves into the facade, and the CLI calls it — the
public API is dogfooded by the project's own binary, not only by an example. The server is **not**
rewritten onto the facade: batching is an L1 capability.

**5. Failures are values.** A library must not print to the host's stderr, call `process::exit`, or
panic on recoverable input. Every load path returns a typed error with a reason; the environment is
mapped to configuration by the CLI, never read inside the engine (ADR-0009's rule, applied at the
crate boundary).

**6. Naming and versioning.** The engine crate is `minfer` (free on crates.io as of 2026-11-07); the
binary keeps the name `minfer`; a later CLI package is `minfer-cli`. The crate stays 0.x, and the
facade is not declared stable before the CLI and the example both exercise it.

## Alternatives considered

- **One published crate for the whole project.** Rejected: consumers compile the HTTP stack and the
  macOS Metal toolchain unconditionally, and a feature flag cannot *prove* isolation the way a
  package boundary can.
- **Workspace split first.** Rejected for Phase 1: it pays the anchor migration (`src/` paths are
  cited ~8,500 times across documentation) before the API being split out exists. Retained as Phase 2,
  which is a move plus a scripted anchor pass, not a redesign.
- **A facade only, with the server keeping private access.** Impossible across a package boundary —
  the server would have to be duplicated or moved into the engine. Rejected.
- **A C ABI as the first public surface.** Rejected as a first step: the project's own consumers are
  Rust, and an ABI freezes a less expressive shape first. The OpenAI-compatible server remains the
  cross-language answer.
- **Keeping the environment as the only configuration surface.** Rejected: an embedder cannot
  configure the engine from code, and two engines cannot differ. The environment stays as the CLI's
  input and is documented, but it is mapped into the engine configuration at the boundary.

## Consequences

- The app layer ships in the same package behind features. `default` keeps today's binary working;
  the consumer's contract is `default-features = false`, proven by a CI build and a runnable example
  rather than by prose.
- `advanced` is public and unstable; the server is its first consumer, so it cannot rot.
- The core must compile without `tokio`/`axum`, which forces the observability seam instead of a hard
  dependency (`observe`).
- Phase 1's no-move property is what keeps the documentation anchors valid; Phase 2 is a deliberate,
  scripted break of that property, gated by the anchor checker.
- The dead-code oracle's sets will move as the binary stops using internals; each slice records its
  set change instead of widening an allowance.
