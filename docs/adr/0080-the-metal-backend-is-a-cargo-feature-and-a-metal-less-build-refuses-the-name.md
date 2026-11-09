# 0080. The Metal backend is a cargo feature, and a metal-less build refuses the name

- Status: Accepted
- Date: 2026-11-09
- Issues: #777

## Context

On macOS the Metal backend is compiled unconditionally: `src/metal/**`, the graph-side
`metal_backend` and every model graph's device branch are behind `cfg(target_os = "macos")`, and
`build.rs` spawns `/usr/bin/xcrun` to compile the shader sources into a `.metallib`. A consumer that
builds the engine with `default-features = false` still pays for all of it — and, because the shader
compile runs, still needs an Apple toolchain even when it will never touch Metal.

CUDA already has the shape a consumer expects: `feature = "cuda"` gates the backend, a build without
it is a legitimate configuration, and asking for the backend by name refuses with a message naming
the feature. Metal is the one device that cannot be turned off.

The mechanical problem is the surface: ~260 `cfg(target_os = "macos")` sites across `src/` and
`tests/`, half of them in files only a Mac compiles. The predicate they are rewritten to must be
decided once, or each site becomes its own answer.

## Decision

**1. `metal` is a cargo feature, and the predicate is one conjunction.**
`metal = []` (its `objc2` dependencies are optional and enabled by it), and every site that reaches
the backend is `cfg(all(target_os = "macos", feature = "metal"))` — including the negations, which
become `not(all(target_os = "macos", feature = "metal"))`, so a paired `cfg`/`cfg(not(...))` arm
stays exhaustive. The feature is inert off macOS (all its sites require the target), which is what
lets `default` name it without a target-conditional default (cargo has none).

**2. `default` lists it; `cli` does not imply it.** `default = ["cli", "metal"]` keeps
`cargo build --release` producing today's Metal-capable binary on macOS, while
`--no-default-features --features cli` still builds a binary **without** the backend. That second
configuration is the one that must refuse `--backend metal` by name, and the only one that can
exercise the refusal from the CLI.

**3. Without the feature, the build touches no Apple toolchain.** `build.rs` compiles no metallib and
emits no `-F <SDK>` framework search path when `metal` is off; instead it prints one
`cargo:warning` — the observable marker that `xcrun` was never spawned (`cargo check
--no-default-features` on macOS). The optional `objc2` dependencies are not even resolved into the
build.

**4. The name stays known, and the refusal names the feature.** The backend id space (the KV-session
tag) is unchanged; `resolve_name("metal")` on a metal-less build answers `NameError::NotCompiled`
with `the Metal backend is compiled only with --features metal` (macOS) or `… only on macOS with
--features metal` (elsewhere). An unknown name and a compiled-out name remain distinct errors, and
the CLI prints the message and exits 1 before it looks for the model.

## Alternatives considered

- **`cli = ["metal"]`** (the first shape suggested in the issue). It keeps the binary Metal-capable
  by definition, but it also makes a metal-less binary unbuildable — so `--backend metal` can never
  be refused *from the CLI*, and acceptance "a `--backend metal` run on a metal-less build refuses
  by name" becomes untestable.
- **`[[bin]] required-features = ["cli", "metal"]`.** The binary would be unbuildable without the
  backend: the opposite of what a consumer needs.
- **A custom `--cfg metal_backend` emitted by `build.rs`.** It hides the condition behind a build
  script, makes `cargo check`-only consumers read `build.rs` to know what compiles, and needs a
  `rustc-check-cfg` declaration to stay warning-free. The explicit `all(target_os, feature)`
  conjunction is greppable and explains itself at every site.
- **Leaving `default = ["cli"]` and documenting that macOS users should add `--features metal`.**
  Silent behaviour change for every existing build and script.

## Consequences

- ~260 cfg sites now carry the feature, and a **new** Metal code site must use the same conjunction;
  a plain `cfg(target_os = "macos")` around backend code is now a bug (it would compile references to
  a module that the feature has removed).
- A metal-less macOS build needs no Apple SDK at all: `cargo check --no-default-features` spawns no
  `xcrun`, resolves no `objc2` crate, and links no `-framework Metal`.
- The metal-less unit-test run (`--no-default-features --features cli`) loses the Metal-gated tests
  by construction; the default-feature suites keep their recorded counts. The engine-only test build
  (`--no-default-features`, no `cli`) has a separate, pre-existing gap ([#791](https://github.com/yusiwen/minfer/issues/791)).
- The feature is documented in `docs/CRATE-API-DESIGN.md` (the feature table), `docs/BUILD.md`
  ("macOS / Metal notes"), `docs/FEATURES.md` and `AGENTS.md`.
