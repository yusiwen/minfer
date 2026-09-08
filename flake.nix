{
  description = "minfer — pure Rust LLM inference engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { nixpkgs, rust-overlay, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system: let
      overlays = [ (import rust-overlay) ];
      pkgs = import nixpkgs { inherit system overlays; };
      rust = pkgs.rust-bin.stable."1.97.1".default.override {
        extensions = [ "rust-src" "rust-analyzer" ];
      };
    in {
      devShells.default = pkgs.mkShell {
        # Provide a real locale archive so a macos host `LC_ALL=en_US.UTF-8`
        # is honored — otherwise the nix bash warns "cannot change locale
        # (en_US.UTF-8)" on every command.
        buildInputs = [ pkgs.glibcLocales ];
        LOCALE_ARCHIVE = "${pkgs.glibcLocales}/lib/locale/locale-archive";
        nativeBuildInputs = [
          rust
          pkgs.pkg-config
          pkgs.curl
          pkgs.uv
          pkgs.libiconv
          pkgs.mdbook
        ];
        shellHook = ''
          export RUST_BACKTRACE=1
        '';
      };
    });
}
