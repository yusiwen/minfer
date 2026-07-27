{
  description = "minfer — pure Rust LLM inference engine";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
      rust = pkgs.rust-bin.stable.latest.default.override {
        extensions = [ "rust-src" "rust-analyzer" ];
      };
    in {
      devShells.default = pkgs.mkShell {
        nativeBuildInputs = [
          rust
          pkgs.pkg-config
          pkgs.curl
          pkgs.uv
          pkgs.libiconv
        ];
        shellHook = ''
          export RUST_BACKTRACE=1
        '';
      };
    });
}
