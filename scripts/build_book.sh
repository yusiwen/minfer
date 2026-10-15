#!/usr/bin/env bash
# Build the mdBook site (docs/ → book/).
#
# Usage:
#   scripts/build_book.sh                 # `mdbook build`
#   scripts/build_book.sh serve           # live-preview server
#   scripts/build_book.sh build --open    # extra arguments are forwarded
#
# This script used to be the one place that staged the Mermaid library — a 2.9 MB
# third-party JavaScript, downloaded and sha384-checked on every build — inside the
# book root, wrote the boot call beside it, ran mdbook and removed both again on
# exit. Every diagram is now a hand-authored SVG in docs/figures/, pulled into its
# page with {{#include}} and wrapped in <figure class="book-figure">, so there is no
# client-side renderer, no preprocessor and nothing to stage or clean up.
#
# It stays as the single entry point CI, the local devShell and the docs all call,
# so the pinned command cannot drift between them.
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v mdbook >/dev/null 2>&1; then
  echo "build_book.sh: mdbook not found on PATH." >&2
  echo "  install mdBook 0.4.37 (see .github/workflows/docs.yml), or enter the devShell: nix develop" >&2
  exit 127
fi

if [ "$#" -eq 0 ]; then
  set -- build
fi
mdbook "$@"
