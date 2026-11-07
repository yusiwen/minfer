//! minfer — a pure-Rust LLM inference engine.
//!
//! This is the engine as a library. Phase 1 ([#763]) keeps every module public
//! while the tiers settle: S6 narrows the surface to the facade and `advanced`,
//! and the application layer below is the binary's, not an embedder's.
//!
//! [#763]: https://github.com/yusiwen/minfer/issues/763
//!
//! The decision behind the split is ADR-0079 (`docs/adr/0079-the-engine-is-a-crate-with-two-api-tiers.md`).

pub mod block;
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
#[cfg(target_os = "macos")]
pub mod metal;

// The application layer: only the binary uses these. Phase 1 gates them behind
// `cli`; S2 makes their dependencies optional and S7 moves the server onto the
// `advanced` tier.
#[cfg(feature = "cli")]
pub mod bench;
#[cfg(feature = "cli")]
pub mod convert;
#[cfg(feature = "cli")]
pub mod download;
#[cfg(feature = "cli")]
pub mod server;
#[cfg(feature = "cli")]
pub mod spec_verify;
#[cfg(feature = "cli")]
pub mod tooling;
