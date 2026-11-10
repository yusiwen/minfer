//! minfer — a pure-Rust inference engine for GGUF models.
//!
//! This crate is the engine: it loads a GGUF model, renders and tokenizes a prompt, and runs one
//! declarative compute graph on the CPU, Metal (macOS) or CUDA. The binary, the HTTP server and the
//! visualizer are the separate `crates/minfer-cli` package, which depends on this one (ADR-0082); a
//! host that only wants to load a model and answer a prompt needs nothing from it.
//!
//! # Tiers
//!
//! - [`engine`] — the facade: [`Engine`](engine::Engine), [`Session`](engine::Session),
//!   `chat`/`complete`, `generate`, and the error types. The only semver-relevant surface before 1.0.
//! - [`advanced`] — the primitives the server drives: the graph cache and its params, the batch, the
//!   KV format gate, the tokenizer, the sampler, the grammar, the speculative engine and
//!   [`CancelToken`](cancel::CancelToken). Public, **not** semver-stable before 1.0.
//! - `live`/`trace`/`graph::json`/`optiming` — the observation surface, behind the `observe` feature.
//!
//! The remaining modules are public because the project's own packages live in this workspace and
//! consume them directly; they are not part of the contract (ADR-0079).
//!
//! # Quick start
//!
//! One chat turn on the toy model committed under `tests/fixtures/`. The delta callback receives
//! bytes, not text: a delta may end inside a UTF-8 sequence, so concatenate before decoding.
//!
//! ```
//! use minfer::conversation::TurnParams;
//! use minfer::engine::{Engine, SessionConfig};
//! use minfer::sampler::SamplerConfig;
//!
//! fn main() -> Result<(), Box<dyn std::error::Error>> {
//!     // The toy model committed in this repository; a host passes its own path.
//!     let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
//!         .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
//!     let engine = Engine::load(&path)?;
//!
//!     let mut session = engine.session(SessionConfig {
//!         n_ctx: 64,
//!         ..Default::default()
//!     });
//!     let params = TurnParams {
//!         n_predict: 32,
//!         sampler: SamplerConfig::default(),
//!         stop_strings: Vec::new(),
//!         cancel: None,
//!     };
//!
//!     let mut out = Vec::new();
//!     session.chat("hello", &params, &mut |delta| out.extend_from_slice(delta))?;
//!     assert!(!out.is_empty(), "a turn streams its answer as bytes");
//!     println!("{}", String::from_utf8_lossy(&out));
//!     Ok(())
//! }
//! ```
//!
//! A **raw continuation** writes exactly the tokens it is given, at the session's current cursor —
//! no template render, no history entry — which is how a long prompt is fed in pieces:
//!
//! ```
//! # use minfer::conversation::TurnParams;
//! # use minfer::engine::{Engine, SessionConfig};
//! # use minfer::sampler::SamplerConfig;
//! # let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
//! #     .join("tests/fixtures/toy/toy-qwen2-f32.gguf");
//! # let engine = Engine::load(&path)?;
//! # let mut session = engine.session(SessionConfig { n_ctx: 64, ..Default::default() });
//! # let params = TurnParams { n_predict: 0, sampler: SamplerConfig::default(), stop_strings: Vec::new(), cancel: None };
//! let tokens = engine.tokenizer().encode("The capital of France is");
//! let outcome = session.complete(&tokens, &params, &mut |_delta| {})?;
//! assert_eq!(outcome.prefill_tokens, tokens.len());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The full guide — features, the three loaders, sessions, cancellation, concurrency, the
//! single-shot path and the advanced tier — is `docs/LIBRARY-USAGE.md`.

pub mod advanced;
pub mod block;
pub mod cancel;
pub mod config;
pub mod conversation;
// #185: the checked "one thread at a time in the CUDA device path" contract.
// Pure and feature-independent on purpose, so the CPU CI job runs its tests.
// Consumers live in the CUDA backend; the offline unit tests still run without
// the feature, so silence dead-code only in CPU-only builds.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub mod device_tier;
pub mod dump;
pub mod engine;
pub mod gguf;
pub mod gguf_write;
pub mod grammar;
pub mod graph;
pub mod kernel;
#[cfg(feature = "observe")]
pub mod live;
pub mod models;
pub mod optiming;
pub mod q4k_dsc;
pub mod quantize;
pub mod quants;
pub mod sampler;
pub mod spec;
pub mod template;
pub mod tensor;
pub mod testfail;
pub mod tokenizer;
pub mod trace;
pub mod vec_ops;

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(all(target_os = "macos", feature = "metal"))]
pub mod metal;
