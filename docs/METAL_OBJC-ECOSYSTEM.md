# Metal Backend and the Objective-C Crate Ecosystem (objc 0.2 vs objc2)

**Context (ℹ️ superseded).** minfer's Metal backend (`src/metal.rs`,
`src/metal.metal`) was built on the
[`metal`](https://crates.io/crates/metal) crate, which was locked into the
**legacy `objc` 0.2 ecosystem**. This document records what that meant, why it
mattered for building/running minfer, and what the modern alternative (`objc2`)
offered. The migration described here **landed on 2026-08-25** (see
`docs/METAL-OBJC2-MIGRATION-PLAN.md`): the backend now uses `objc2-metal` +
`block2` + `objc2-foundation`, and `metal`/`block`/`vendor/block`/
`[patch.crates-io]` were removed. The rest of this document is kept as
historical context for why the old setup looked the way it did.

---

## 1. What `objc` is

macOS/iOS system APIs (Metal, Foundation, AppKit, …) are written in
Objective-C. Calling them from Rust means sending messages to Objective-C
objects (`objc_msgSend`). The `objc` crate is the **FFI bridge** that wraps
this: it lets Rust create objects, call methods, and pass block callbacks to
Objective-C. The `metal` crate is a Metal-framework binding built on top of
`objc`.

---

## 2. The legacy `objc` 0.2 ecosystem

- Author: Steven Sheldon (`SSheldon/rust-objc`), started 2012.
- **Frozen at `objc` 0.2.7 (last release 2019-10)** — no updates for 6+ years.
- The whole companion set is frozen in the same era and **mutually locked**:
  - `block` 0.1.6 (2016) — hard dependency of `metal` (completion handlers)
  - `metal` 0.28 (minfer's version) and 0.33 (current) — both still depend on
    `block ^0.1.6` and `objc ^0.2.4`
  - `cocoa` 0.26, etc.
- Design characteristics:
  - `msg_send!` macro straight to the runtime; **`unsafe` everywhere**
  - message names are strings — a typo fails at runtime, not compile time
  - reference counting (`retain`/`release`) is manual
  - pre-2024-edition `extern` blocks, weak typing
- Because it is the *foundation* of the old Apple-Rust ecosystem, when the
  foundation breaks, every crate above it breaks together. Two concrete
  instances of this in minfer:

  **a) `block` 0.1.6 uninhabited static (rust-lang/rust#74840).** — *historical,
  resolved by the 2026-08-25 objc2 migration: `metal`, `block` and `vendor/block`
  no longer exist in the dependency tree, so this issue is gone.*
  Historically, `block` declared `enum Class {}` +
  `extern { static _NSConcreteStackBlock: Class }`
  — a *static of uninhabited type*, which rustc was phasing out: a
  future-incompat warning at the time, a **hard error in a future rustc**
  (breaking every macOS build, since `metal` pulled `block` unconditionally).
  Upstream was unmaintained (master had the same code; crates.io had no newer
  release). The **fix** was a vendored one-line fix under `vendor/block/`, wired
  via `[patch.crates-io] block = { path = "vendor/block" }` (see
  `vendor/block/README.md`): it turned `Class` into an opaque
  `#[repr(C)]` ZST (same `isa`-pointer semantics, now inhabited and FFI-safe)
  and added explicit `extern "C"` ABIs.

  **b) nix devShells shadow the real Apple toolchain.**
  `flake.nix` (loaded via direnv) uses nixpkgs' darwin stdenv, which exports
  `DEVELOPER_DIR`/`SDKROOT` pointing at nixpkgs' open-source `apple-sdk`, and
  puts nixpkgs' **xcbuild `xcrun`** first on `PATH`. Neither has the `metal`
  compiler, so `build.rs`'s metallib precompile step failed and minfer fell
  back to compiling shaders from source at every process start.
  **Fix:** `build.rs` invokes `/usr/bin/xcrun` by absolute path and strips
  `DEVELOPER_DIR`/`SDKROOT` for those two child processes, so the real Xcode
  (`xcode-select -p`) is always used. rustc linking in the same shell is
  unaffected (it still uses the nix SDK/clang).

---

## 3. The `objc2` ecosystem (modern rewrite)

- Author: **Mads Marquart** (`madsmtm/objc2`), a complete rewrite started 2021
  — not a patch-level upgrade.
- **Actively maintained**: `objc2` 0.6.4 (2026-02), ~100M downloads and
  climbing (already ahead of `objc`'s ~39M).
- Design goal: **memory safety first**:
  - typed messages: `method!` macros generate real Rust function signatures —
    a wrong selector/argument is a **compile-time error**, not a runtime crash
  - RAII-managed reference counting (objects release themselves on drop)
  - exception handling, `Send`/`Sync` annotations, safe block wrappers
- Family: `block2` (0.6.2), `objc2-metal` (0.3.2), `objc2-foundation`,
  `icrate` (auto-generated bindings for the whole system framework set).
- New projects overwhelmingly choose this ecosystem.

---

## 4. "0.2" vs "2.0" — they are *not* a version upgrade

A common point of confusion: `objc 0.2` and `objc2` are **two different
projects**, not two versions of the same crate.

| | `objc` 0.2.x | `objc2` |
|---|---|---|
| What "2" means | a normal semver version (0.x = pre-1.0, never stabilized) | "second generation" codename |
| Version history | frozen at 0.2.7 (2019) | independent, started at 0.1, now 0.6.4 |
| Relationship | old project | rewrite, **API-incompatible** |

The best analogy is **Python 2 vs Python 3**: nominally the same name,
actually a hard break, coexisting in parallel, code not interchangeable.
`objc 0.2` will never "become" `objc2`; `block` 0.1.6 and `block2` have the
same relationship.

---

## 5. What this means for minfer (decisions and roadmap)

- **(Historical — no longer applies.)** While the `metal` crate was used,
  `block` 0.1.6 was compiled unconditionally (metal's public API references
  `block::ConcreteBlock`), so it could not be dropped without forking `metal`;
  the vendor (`vendor/block` + `[patch.crates-io]`) was the cheapest
  offline-reproducible fix. This is **moot since the 2026-08-25 objc2 migration**
  (see the DONE bullet below): `metal`/`block`/`vendor/block`/`[patch.crates-io]`
  are all gone.
- **Do not "fix" the nix/devShell SDK issue in `flake.nix`**: overriding
  `DEVELOPER_DIR`/`SDKROOT` there would also redirect rustc's linker
  environment; the `build.rs` scoped fix is the correct place.
- **✅ DONE (2026-08-25)**: the Metal layer was migrated to the `objc2`
  ecosystem (`objc2-metal` + `block2` + `objc2-foundation`), dropping the
  `metal` crate, `block` crate, `vendor/block` and the `[patch.crates-io]`
  entry (see `docs/METAL-OBJC2-MIGRATION-PLAN.md`). `src/metal.rs`,
  `src/graph/metal_backend.rs` and `tests/*_isolation.rs` now use objc2-metal;
  `metal`/`block` are no longer dependencies.

---

*Last updated: 2026-08-25 — migration to objc2-metal landed (see
`METAL-OBJC2-MIGRATION-PLAN.md`); the sections above are kept as historical
context. Facts (versions/downloads) were verified against crates.io at the
original date.*
