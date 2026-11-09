//! What the engine's `build.rs` cannot give this package: the version string,
//! and the SDK framework search path the binary's link needs.
//!
//! A cargo build script's output is scoped to the package that emitted it.
//! `cargo:rustc-env` is per-package, so `--version` needs its own
//! `MINFER_VERSION`; and `cargo:rustc-link-arg` reaches only the binaries,
//! tests, examples and benches of the *same* package, so the engine's
//! `-F <Xcode SDK>` flag never arrives at this package's final link. Before the
//! application layer became its own package (#805) both lived in `minfer` and
//! the flag applied; afterwards it silently did not, and a build under the nix
//! devShell's clang died with `ld: framework not found Metal` (#809).
//!
//! The release workflow still passes `MINFER_VERSION` as `vX.Y.Z(shortsha)`; a
//! local build falls back to the package version, exactly as the engine's does.

fn main() {
    println!("cargo:rerun-if-env-changed=MINFER_VERSION");
    let pkg_version = std::env::var("CARGO_PKG_VERSION").unwrap_or_else(|_| "0.0.0".into());
    let minfer_version =
        std::env::var("MINFER_VERSION").unwrap_or_else(|_| format!("v{pkg_version}"));
    println!("cargo:rustc-env=MINFER_VERSION={minfer_version}");

    // ─── Real-SDK framework search path for this package's link ──────────
    // Same reasoning as the engine's copy of this block (see its
    // "Real-SDK framework search path" section): the nix devShell's link driver
    // is nixpkgs' clang, whose `-F` list covers nixpkgs' apple-sdk frameworks
    // but not Metal, so `-framework Metal` needs the Xcode SDK on the search
    // path. `/usr/bin/xcrun` with DEVELOPER_DIR/SDKROOT stripped is the path to
    // the real toolchain (docs/METAL_OBJC-ECOSYSTEM.md §2b: fix it here, not by
    // redirecting the devShell's SDK). On a system-clang build the flag is a
    // harmless no-op — the directory is already an implicit search path.
    //
    // Gated on the `metal` feature: without the backend nothing links Metal,
    // and a metal-less macOS build must not need the Apple toolchain (#777).
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var_os("CARGO_FEATURE_METAL").is_some()
    {
        println!("cargo:rerun-if-env-changed=MINFER_METAL_XCRUN");
        let xcrun = std::env::var("MINFER_METAL_XCRUN").unwrap_or_else(|_| "/usr/bin/xcrun".into());
        if let Ok(out) = std::process::Command::new(&xcrun)
            .env_remove("DEVELOPER_DIR")
            .env_remove("SDKROOT")
            .args(["--sdk", "macosx", "--show-sdk-path"])
            .output()
        {
            if out.status.success() {
                let sdk = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let frameworks = format!("{sdk}/System/Library/Frameworks");
                if std::path::Path::new(&frameworks).is_dir() {
                    println!("cargo:rustc-link-arg=-F{frameworks}");
                }
            }
        }
    }
}
