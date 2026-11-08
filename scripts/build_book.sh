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

# `serve` (and any other subcommand) is an interactive server whose verdict is a
# page a human reads, not a build log this script can judge: pass it through.
if [ "$1" != "build" ]; then
  exec mdbook "$@"
fi

# [#780] mdBook 0.4.37 logs a failed `{{#include}}` / `{{#rustdoc_include}}` at
# ERROR and still exits 0, leaving the directive in the page — so the exit code
# is not this gate. Capture the log, print it (a CI log must stay complete) and
# fail on any `[ERROR]` line.
log=$(mktemp "${TMPDIR:-/tmp}/minfer-book-build.XXXXXX")
trap 'rm -f "$log"' EXIT

set +e
mdbook "$@" 2>&1 | tee "$log"
status=${PIPESTATUS[0]}
set -e

if [ "$status" -ne 0 ]; then
  echo "build_book.sh: mdbook build failed (exit $status)" >&2
  exit "$status"
fi

if grep -q '\[ERROR\]' "$log"; then
  echo "build_book.sh: mdbook exited 0 but logged ERROR lines; it does not fail on them (#780):" >&2
  grep '\[ERROR\]' "$log" | sed 's/^/  /' >&2
  echo "build_book.sh: an unresolvable include/rustdoc_include is the usual cause. To *show* a" >&2
  echo "build_book.sh: directive instead of running it, escape it as \\{{#directive}}." >&2
  exit 1
fi
