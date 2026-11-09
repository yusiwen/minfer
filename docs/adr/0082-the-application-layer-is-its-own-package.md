# 0082. The application layer is its own package, and the engine's four application features retire

- Status: Accepted
- Date: 2026-11-09
- Issues: #805

## Context

[ADR-0079](./0079-the-engine-is-a-crate-with-two-api-tiers.md) made the engine a crate with two API
tiers and deferred the workspace split — its decision 5 — behind three named triggers, sketching it as
Phase 2 with a `crates/minfer-app` name. Phase 1 landed (S0–S10) and **none of the triggers fired**:
`crates.io` answers 404 for `minfer`, so no version has ever been published; there is no external
consumer; and the `feature-matrix` CI job already proves the consumer's sliver with `cargo tree` and
`cargo check --no-default-features`. The maintainer decided to do the split anyway, for one reason the
triggers do not model: **a feature fence can be forgotten, a package boundary cannot.**

The measurements that shaped the decision (2026-11-09, master `d4cff0c`):

- the application layer is 56 files / ~26,400 lines, and it is **already on the public surface** — only
  8 `pub(crate)` items (21 references) are touched, because S7 moved the server onto `advanced`;
- 564 `crate::` references become `minfer::`, and 113 stay (`server` / `convert` / `download` are
  same-package once moved);
- 75 gated `path:NNN` anchors and 44 ADR citations point into the moving paths;
- one workspace mechanic **fails silently**: a plain `--features cuda` at the workspace root is applied
  to every selected member, but a member that does not declare the feature is compiled *without* it and
  no error is printed.

## Decision

1. **One non-virtual workspace.** The engine stays the root package and `src/` does not move;
   the application layer becomes `crates/minfer-cli`; `default-members = [".", "crates/minfer-cli"]`
   keeps a plain `cargo build --release` producing `target/release/minfer`.
2. **The engine's feature table retires `cli`, `server`, `tooling` and `viz`**, and keeps `metal`,
   `cuda`, `cuda_static`, `debug_dump` and `observe` with `default = ["metal"]`. The CLI package forwards
   those five, one line each. The four retiring names are not re-created in the CLI package: its binary
   is its only target, nothing gates its subcommand dispatch, and turning them into real switches would
   be the redesign this phase excludes. **The package is the switch.**
3. **Only the engine is published.** `crates/minfer-cli` is `publish = false` and the CLI ships as a
   release artifact, so `viz/`, `tests/fixtures/` and `scripts/gen_toy_gguf.py` stay at the repository
   root, reached with `concat!(env!("CARGO_MANIFEST_DIR"), "/../../…")`.
4. **The suite ledger stays one merged row.** One `cargo test --release` at the root runs both packages,
   and `scripts/check_baselines.py` already sums every `test result:` block into its two buckets.
5. **Every command that needs one package names it.** The engine-only build and test are
   `-p minfer --no-default-features`; `cargo package` is `-p minfer`. A bare `--no-default-features` at
   the root applies to every selected member and therefore stops isolating the engine.
6. **ADRs stay frozen.** The 44 repository-rooted citations the move invalidates are pinned in
   `scripts/adr-citations.toml` with `corrected-elsewhere` and this ADR as the errata, and ADR-0060's
   single markdown link target is re-pointed while the backticked citation beside it stays as written.
   The hyperlink is navigation; the citation is the record.
7. **The three-way split is deferred behind a named trigger**: when `minfer-server` needs a dependency the
   CLI does not want, or when `minfer-server` / `minfer-viz` needs its own release artifact. The move
   preserves the two properties that make that a pure move: `viz` depends on the server one-way, and only
   two `pub(crate)` items would need promoting.

## Alternatives considered

- **Keep the three triggers as the gate and do nothing.** Rejected by the maintainer on 2026-11-09: the
  package boundary is exactly the proof the triggers cannot measure, and the feature fence has already
  needed manual repair (a test reaching into an application module, #791).
- **Move the engine into `crates/minfer/` too**, for a uniform layout. Rejected: it rewrites ~1,100
  `path:NNN` anchors for no packaging gain, and a non-virtual workspace gives the same four packages.
- **Keep `cli` as an empty feature for one release.** Rejected: no published version has ever offered it
  (crates.io 404), so there is no manifest to keep working, and an empty feature is a false row in the
  table.
- **Keep `server` / `tooling` / `viz` as CLI-package features.** Rejected: `main.rs` is the only target
  and nothing gates its dispatch, so they would need a `#[cfg]` refactor; and after the split they gate
  nothing.
- **Publish the CLI package as well.** Rejected for now: `release.yml` already produces the binary, and
  publishing would add the asset/fixture packaging constraints for no consumer.
- **Split the app three ways now** (`minfer-cli` / `minfer-server` / `minfer-viz`). Deferred, not
  rejected: it is cheap in code (one new manifest, two visibility promotions) but buys no dependency
  isolation — `viz` introduces no dependency the server lacks — so it waits for the named trigger.

## Consequences

- The engine package's dependency graph becomes the published contract instead of a hand-maintained
  promise; `cargo tree -p minfer --no-default-features` is the proof.
- `cargo build --release --no-default-features` no longer isolates the engine, so every engine-only
  command names `-p minfer`. `docs/BUILD.md`'s engine-only suite line is rewritten in the same change.
- A forgotten feature forward is invisible to the compiler. The CLI package declares all five forwards,
  and `-D warnings` on its feature check turns the resulting `unexpected_cfgs` into a failure.
- A dependent package cannot see `#[cfg(test)]` items, and the application's tests observe the engine
  through several of them. They are gated `cfg(any(test, feature = "test-hooks"))`, and `test-hooks` is a
  new engine feature that only the CLI's `[dev-dependencies]` enables: the observers stay out of a
  consumer's build, and the alternative — moving the server's tests into the engine — would have put
  application tests in the wrong package.
- Three gates must learn the workspace roots or they silently stop checking the moved tree:
  `check_env_vars.py` (it walks `<root>/src` only), `check_source_layout.py` (one root per package) and
  `check_doc_line_anchors.py` (`ROOT_DIRS` plus its symbol index).
- The suite ledger's merged row counts both packages, so a later change that shifts tests between them
  does not move the number.
- ADR-0079's Phase 2 sentence and its `crates/minfer-app` name are superseded in substance; 0079 keeps
  `Status: Accepted`, because its tier decision is unchanged and only its deferral is now executed
  (the same relationship ADR-0080 has with 0079's feature table).
