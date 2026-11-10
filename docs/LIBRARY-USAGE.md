# Using the engine as a library

`minfer` is the engine crate: it turns a GGUF file into an answer. The binary, the HTTP server and the
visualizer are the separate `crates/minfer-cli` package, which depends on this crate the way any host
would (ADR-0082) — so the surface described here is the one the project's own tools use.

This page is the task-ordered guide: depend on the crate, load a model, run a turn, stop it, and go
lower only when the facade runs out. The [crate contract](./CRATE-API-DESIGN.md) is the reference
behind it — tiers, ownership, configuration, failure rules and the stability promise — and the
[CLI reference](./USAGE.md) covers the binary's own surface.

The program in §4 is the crate root's doctest: CI compiles and runs it, so an example that stops
working fails a build rather than misleading a reader.

## 1. Depend on it

The crate is **not published on crates.io yet**. Until it is, depend on a checkout:

```toml
[dependencies]
minfer = { path = "../minfer", default-features = false }
```

Once it is published the same line becomes `minfer = { version = "…", default-features = false }`.
The crate is edition 2021 and declares `rust-version = "1.97"`.

| feature | what it brings | default |
| :--- | :--- | :--- |
| `metal` | the Metal backend and its `objc2` dependencies | **on**; inert off macOS |
| `observe` | the `live` / `trace` hooks (adds tokio's `sync`) | off |
| `cuda` | the CUDA backend (needs `nvcc`) | off |
| `cuda_static` | link `cudart` statically | off |
| `debug_dump` | `MINFER_DUMP_DIR` per-layer dumps | off |

`default-features = false` now means **only** "no Metal backend": the server, the tooling and the
visualizer are in another package, and no feature of this crate can pull them in. On Linux that form
resolves no `objc2` crate and runs no `xcrun`.

Inside this repository, `cargo check -p minfer --no-default-features` is the sliver a consumer gets,
and `cargo test -p minfer --no-default-features --lib` proves that configuration still builds and
tests.

## 2. Load a model

| constructor | use it when |
| :--- | :--- |
| `Engine::load(&path)` | the normal case: process configuration, no template override |
| `Engine::load_with(&path, offload, cache_type, template_override)` | you need `--gpu-layers`-style offload, an explicit KV format (`"f32"`/`"f16"`/`"q8_0"`), or a chat-template file. `None` falls back to the process configuration |
| `Engine::load_from_bytes(bytes)` | the model is already in memory. The buffer is **not** copied — every weight is a range of it — and a split model is refused by name, because its siblings are resolved beside the file and bytes have no directory |

An `Engine` owns the mapping, the model, the tokenizer and the **validated** chat template; every
constructor fails with a named `EngineError`:

| variant | means |
| :--- | :--- |
| `EngineError::Gguf(GgufLoadError)` | the file did not parse as GGUF (the error names the path and the parser's reason) |
| `EngineError::Model(ModelLoadError)` | unsupported architecture, or a missing weight |
| `EngineError::Tokenizer(String)` | the tokenizer metadata is unsupported or incomplete |
| `EngineError::Template(TemplateError)` | the chat template cannot be rendered |

A load never calls `process::exit`, never panics on malformed input and never writes to your stderr —
rendering the error is the host's job (ADR-0009).

## 3. Configure the process

`EngineConfig` (`minfer::config`) is the **process** configuration. The CLI builds it from `MINFER_*`
via `EngineConfig::from_env()`; a library host should either accept the built-in defaults or install
its own once, before any device or graph exists:

```text
minfer::config::install(cfg)   // first call wins; later calls are ignored
minfer::config::get()          // the installed config, or the defaults
```

A consumer that installs nothing gets the defaults, never the host's environment. What can differ
between engines — the backend fence, the KV format, the offload plan — is resolved per engine; what
cannot (`n_ctx`, the seed) is per `SessionConfig`. The CPU worker pool is process-wide:
`minfer::kernel::set_cpu_threads(n)`.

## 4. One turn

```rust
use minfer::conversation::TurnParams;
use minfer::engine::{Engine, SessionConfig};
use minfer::sampler::SamplerConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // The toy model committed in this repository; a host passes its own path.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
    let engine = Engine::load(&path)?;

    let mut session = engine.session(SessionConfig {
        n_ctx: 64,
        ..Default::default()
    });
    let params = TurnParams {
        n_predict: 32,
        sampler: SamplerConfig::default(),
        stop_strings: Vec::new(),
        cancel: None,
    };

    let mut out = Vec::new();
    session.chat("hello", &params, &mut |delta| out.extend_from_slice(delta))?;
    assert!(!out.is_empty(), "a turn streams its answer as bytes");
    println!("{}", String::from_utf8_lossy(&out));
    Ok(())
}
```

`SessionConfig` carries what a conversation needs and the engine cannot answer: `n_ctx` (default
4096), `seed` (42), the initial mirostat `tau` (5.0, i.e. `mu = 2 * tau`, owned by the session across
turns) and an optional `system_prompt`.

`TurnParams` has **no `Default`** — every field is a decision:

| field | meaning |
| :--- | :--- |
| `n_predict` | the cap for this turn |
| `sampler` | the whole pipeline: penalties, DRY, grammar, top-k/typical/top-p/min-p/XTC, temperature **or** mirostat |
| `stop_strings` | truncate the answer at the first match (the text before it is kept) |
| `cancel` | a `CancelToken` checked between decoded tokens (§6) |

`Session::chat` renders the input through the model's template, prefills it after the rows the session
already holds, decodes, and records both messages. The callback receives **bytes**: a delta is not
guaranteed to end on a UTF-8 boundary, so concatenate before decoding. `TurnOutcome` says how the turn
ended — `hit_n_predict`, `stopped_by_eog`, `stopped_by_string`, `cancelled`, `tokens_generated`,
`prefill_tokens`, `dropped_turns` (old turns dropped on context overflow) and `text` (the assembled
answer, for a caller that does not stream).

`session.n_past()` is the KV cursor — the position the next prefill starts at — and
`session.conversation().messages` is the recorded history, `(role, content)` pairs.

## 5. Feed a prompt in pieces

`Session::complete(&tokens, &params, emit)` is a **raw continuation**: the tokens are prefilled at the
cursor, no template is rendered and no message is recorded, then the run decodes like a turn. Only the
new tokens are prefilled, which is what makes a long prompt feedable in chunks —
`TurnOutcome::prefill_tokens` counts them, and the second call charges only its own tokens.

A continuation is **not history**: the next `chat` rebuilds the KV from `messages`, so a continuation
never leaks into a later turn. A session carrying a speculative draft refuses a continuation by name
(`ConvError::CompleteWithSpec`) — the draft's KV is indexed by the same positions and half-driving the
pair would desynchronise it.

## 6. Stop a run or a turn

`CancelToken` (`minfer::cancel`; also re-exported from `minfer::engine` and `minfer::advanced`) is one
flag behind an `Arc`: cheap to clone, `Send + Sync`, and a signal handler or a UI thread can hold its
own copy.

- Set it on `TurnParams.cancel` for a conversation turn, or on `GenerateParams.cancel` for a
  single-shot run. Both loops check it once per decoded token — and once per speculative round — so a
  draft round cannot outrun it.
- It stops at a token boundary and the outcome reports `cancelled` (and `hit_n_predict` is false by
  construction). It **cannot** interrupt a forward pass: a cancel that arrives mid-pass is honoured at
  the next token.
- There is no reset. A cancelled token stays cancelled, so a host that wants to run again makes a new
  one. A cancelled turn stays usable: the partial reply is recorded and the next turn still runs.
- `TurnOutcome::cancelled` / `GenerateOutcome::cancelled` is a stop, not an error: the call returns
  `Ok`.

## 7. Single-shot generation

`Engine::generate` runs one prefill/decode with no session state. The caller owns tokenization and
prompt rendering — `GenerateParams` takes the tokens:

| field | meaning |
| :--- | :--- |
| `tokens` | the already-tokenized (and, for a chat model, already-rendered) prompt |
| `n_ctx`, `n_predict`, `seed`, `stop_strings`, `sampler` | as in `TurnParams`, plus the context size |
| `timing` | count sampling and forward time (`GenerateOutcome::timing`) |
| `spec_adaptive`, `spec_d_max` | the speculative-decoding knobs, when a `SpecEngine` is in play |
| `cancel` | the same token as §6 |

`GenerateOutcome` reports `generated`, `prefill_tokens`, `prefill_secs`, `decode_secs`, the three stop
reasons (`stopped_by_eog`, `stopped_by_string`, `hit_n_predict`), `cancelled`, `grammar_error`,
`timing` and the speculative counters. All of them are **stops**: the request is refused with an
`Err`/`ConvError` instead, and `grammar_error` is the one field that names a refusal inside an `Ok`
run (the emitted text is a valid prefix).

The engine owns the run's KV cache, so repeated runs on one engine build their graph once; a changed
`n_ctx` rebuilds it. Two engines never share one, and two concurrent runs on the *same* engine
serialise on it — sessions are the independent-concurrency path. The second callback,
`on_prefill(&[f32], f64)`, receives the prompt's logits and the prefill time.

## 8. Render the prompt yourself

`Session::chat` does this for you. A single-shot host has to:

1. render the prompt — `minfer::template::render_single_turn(embedded, override_text, prompt, bos_token)`,
   or `render_messages(template, messages, add_generation_prompt, bos_token)` for a full conversation,
   or `format_single(...)` when the host appends one message to an existing rendering;
2. tokenize it — `engine.tokenizer().encode(&text)`;
3. pass the ids as `GenerateParams.tokens`.

`engine.template()` is the validated template the engine loaded (the file's own, or the override);
`engine.chatml_notice()` is true when neither existed and the ChatML fallback is in force — announce it
once, as the CLI does. A template that cannot be rendered refuses the load, and a template that cannot
express `tools` refuses the request by name rather than dropping them (ADR-0058).

## 9. Concurrency and lifetime

- **`Engine` is `Send + Sync`.** Put it behind an `Arc` and create sessions from several threads; a
  turn never mutates the engine.
- **A `Session` borrows its engine** (`Session<'_>`) and owns one conversation: its messages and its
  KV cache. Two sessions on one engine share nothing.
- **`Engine::generate` serialises** on a per-engine cache — concurrent single-shot calls queue.
  Independent concurrency is one session per thread.
- **A changed `n_ctx`** on the single-shot path rebuilds that cache; two engines with different
  `n_ctx` are fine ([#799](https://github.com/yusiwen/minfer/issues/799)).
- **Dropping the engine** releases the mapping, the device registrations and the KV arena, so a
  process may load another model afterwards; the reload gate is `src/gguf/tests.rs`.
- **Process state is shared**: the device context, the worker pool and the installed `EngineConfig`. A
  second, conflicting install is ignored, not merged.

## 10. The advanced tier

`minfer::advanced` exposes what the HTTP server needs and the facade deliberately does not:
`ModelDef`, `GraphCache`/`GraphParams`, `Batch`, `KvFormat`, `Backend`/`BackendFilter`,
`OffloadRequest`, `Tokenizer`, `SpecialTokens`, `SamplerConfig`/`MirostatMode`, `Grammar`,
`CancelToken`, the conversation types, `SpecEngine`, and the observation hooks.

It is public because the server is a first-class consumer, and **not semver-stable before 1.0**: an
embedder that needs it is telling us the facade is missing a capability, so report it rather than
building on the shape.

The rule that bites a naive host: **a `GraphCache` is per engine.** It owns the persistent KV regions,
sized once by `n_ctx` and the model's KV format; two owners sharing one is an allocator refusal at run
time. `Engine::generate` keeps one per engine, a `Session` owns one through its `GraphEngine`, and a
server slot owns one.

## 11. Observation

`observe` turns on the sinks the visualizer and `/metrics` read: `live` (lazy; off until a client
subscribes), `trace`, `graph::json` and `optiming`. They are off in a default consumer build, so the
engine core does not need tokio. The CLI's `MINFER_TRACE` / `MINFER_GRAPH_DUMP` / `MINFER_OP_TIMING`
knobs are its own wiring of these hooks.

## 12. Limits and refusals

- A **split** model cannot be loaded from bytes (`load_from_bytes` refuses by name); load it by path.
- A **chat template that cannot render** refuses the load; there is no generic substitution.
- **Speculative decoding** refuses mirostat, and refuses a grammar; a session with a draft refuses
  `complete` (`ConvError::CompleteWithSpec`).
- A **failure is an error, never a silent CPU fallback** — a backend that cannot run a node fails the
  run (ADR-0009).
- A **cancel cannot interrupt a forward pass** (§6), and a single engine's `generate` calls queue.

## Where to read next

| you want | read |
| :--- | :--- |
| the contract behind this page: tiers, ownership, failures, features | [Crate API design](./CRATE-API-DESIGN.md) |
| the design of the graph every call runs through | [Compute graph](./COMPUTE-GRAPH-DESIGN.md) |
| the KV cache: cells, the format gate, sessions | [KV cache design](./KV-CACHE-DESIGN.md) |
| the binary, the server and their flags | [Usage](./USAGE.md) |
| building the workspace, the CUDA/Metal toolchains, the devShell | [Build](./BUILD.md) |
| supported quants and architectures | [Support matrix](./SUPPORT-MATRIX.md) |

## Decisions governing this document

- [ADR-0079](adr/0079-the-engine-is-a-crate-with-two-api-tiers.md) — The engine is a crate with two API tiers, and the apps are its first consumers
- [ADR-0082](adr/0082-the-application-layer-is-its-own-package.md) — The application layer is its own package, and the engine's four application features retire
- [ADR-0009](adr/0009-a-failure-is-an-error-never-a-silent-fallback.md) — A failure is an error, never a silent fallback
- [ADR-0081](adr/0081-a-tensor-s-bytes-are-a-backing-handle-plus-a-range.md) — A tensor's bytes are a backing handle plus a range, and the backing may be an owned buffer
