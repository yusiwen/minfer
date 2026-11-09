//! The binary's version string, and nothing else.
//!
//! The engine's `build.rs` compiles the Metal library and sets `MINFER_VERSION`
//! for that package; a `cargo:rustc-env` value is per-package, so the binary
//! needs its own emitter for `--version`. The release workflow still passes
//! `MINFER_VERSION` as `vX.Y.Z(shortsha)`; a local build falls back to the
//! package version, exactly as the engine's does.

fn main() {
    println!("cargo:rerun-if-env-changed=MINFER_VERSION");
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
    let minfer_version =
        std::env::var("MINFER_VERSION").unwrap_or_else(|_| format!("v{pkg_version}"));
    println!("cargo:rustc-env=MINFER_VERSION={minfer_version}");
}
