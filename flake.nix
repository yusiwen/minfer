{
  description = "minfer — pure Rust LLM inference engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
    git-hooks.url = "github:cachix/git-hooks.nix";
  };

  outputs = { nixpkgs, rust-overlay, flake-utils, git-hooks, ... }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs { inherit system overlays; };
      rust = pkgs.rust-bin.stable."1.97.1".default.override {
        extensions = [ "rust-src" "rust-analyzer" ];
      };
      # git-hooks.nix pre-commit checks: runnable as a derivation via
      # `nix flake check`, and installed as real .git/hooks by the devShell
      # shellHook below. `rustfmt --check` uses the same toolchain the project
      # pins (1.97.1), so format verdicts are identical for every contributor
      # regardless of what is on their shell PATH.
      pre-commit-check = git-hooks.lib.${system}.run {
        src = ./.;
        hooks.rustfmt = {
          enable = true;
          settings.check = true;
          packageOverrides = {
            cargo = rust;
            rustfmt = rust;
          };
        };
      };
    in {
      checks.pre-commit-check = pre-commit-check;

      devShells.default = pkgs.mkShell {
        # Provide a real locale archive so a host `LC_ALL=en_US.UTF-8` is
        # honored — otherwise the nix bash warns "cannot change locale
        # (en_US.UTF-8)" on every command.
        #
        # glibcLocales / locale-archive are Linux-only. On macOS (darwin)
        # pkgs.glibcLocales evaluates to `null`, so guard every reference
        # behind stdenv.isLinux — interpolating `null` into the string above
        # was the "cannot coerce null to a string" build failure.
        buildInputs = pkgs.lib.optionals pkgs.stdenv.isLinux [
          pkgs.glibcLocales
        ];
        LOCALE_ARCHIVE = pkgs.lib.optionalString pkgs.stdenv.isLinux
          "${pkgs.glibcLocales}/lib/locale/locale-archive";
        nativeBuildInputs = [
          rust
          pkgs.pkg-config
          pkgs.curl
          pkgs.uv
          pkgs.libiconv
          pkgs.mdbook
          pkgs.mdbook-mermaid
        ];
        # The rusty-hook pre-commit (cargo fmt + re-stage) was removed; the
        # git hook is now provided by git-hooks.nix as a rustfmt --check gate.
        # It is installed when entering this devShell (nix develop); running
        # `git commit` outside of it has no hook.
        shellHook = ''
          export RUST_BACKTRACE=1
        '' + pre-commit-check.shellHook;
      };
    });
}