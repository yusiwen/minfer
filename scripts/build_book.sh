#!/usr/bin/env bash
# Build the mdBook site (docs/ → book/), staging the Mermaid assets it needs.
#
# Two files have to exist *inside the book root* before `mdbook build` runs:
#
#   docs/assets/mermaid.min.js   the library, downloaded and hash-checked here
#   docs/assets/mermaid-init.js  the one-line `mermaid.initialize` boot call
#
# Neither is committed: together they are 2.9 MB of third-party JavaScript
# (issue #110). mdBook cannot reference an asset by URL — `copy_additional_css_and_js`
# copies `root.join(entry)` to `destination.join(entry)`, with no URL branch — and
# an *absolute* entry is worse than unsupported, because `Path::join` replaces the
# prefix: input and output become the same file and the copy truncates the source
# (measured: exit 0, nothing emitted into book/). So the bytes must be staged
# inside the book root, and this script is the one place that does it: stage,
# verify, build, and remove — whether the build succeeds, fails or is interrupted.
#
# Usage:
#   scripts/build_book.sh                 # `mdbook build`
#   scripts/build_book.sh serve           # assets stay staged for the session
#   scripts/build_book.sh build --open    # extra arguments are forwarded
#
# Environment:
#   MINFER_MERMAID_URL  override the download URL (e.g. a mirror). The sha384 pin
#                       below still applies, so a mirror must serve identical bytes.
set -euo pipefail

cd "$(dirname "$0")/.."

MERMAID_VERSION="10.6.1"
MERMAID_URL="${MINFER_MERMAID_URL:-https://cdn.jsdelivr.net/npm/mermaid@${MERMAID_VERSION}/dist/mermaid.min.js}"
# sha384 of that exact artifact — the npm dist file, *before* the MIT notice below
# is prepended. mdbook-mermaid 0.13.0 vendors the same bytes with that notice in
# front; we take the upstream file so the version and the hash are both explicit.
MERMAID_SHA384="f8d19f8d4f0ace90cb5d11ddb84a96f99889af6ac883e722754564ec1e75479c4aedc1f030a41fadd17019daead41733"

ASSET_DIR="docs/assets"
LIB="$ASSET_DIR/mermaid.min.js"
INIT="$ASSET_DIR/mermaid-init.js"
DL=""

cleanup() {
  rm -f "$LIB" "$INIT" ${DL:+"$DL"}
  rmdir "$ASSET_DIR" 2>/dev/null || true # only when we left it empty
}
trap cleanup EXIT

for tool in mdbook mdbook-mermaid curl; do
  command -v "$tool" >/dev/null 2>&1 && continue
  echo "build_book.sh: $tool not found on PATH." >&2
  case "$tool" in
    mdbook) echo "  install mdBook 0.4.37 (see .github/workflows/docs.yml), or enter the devShell: nix develop" >&2 ;;
    mdbook-mermaid) echo "  cargo install mdbook-mermaid --locked --version 0.13.0" >&2 ;;
  esac
  exit 127
done

# sha384 as lowercase hex, using whichever tool the platform ships: GNU coreutils
# on Linux, `shasum` on macOS, `openssl` as the last resort.
sha384_hex() {
  if command -v sha384sum >/dev/null 2>&1; then
    sha384sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 384 "$1" | cut -d' ' -f1
  else
    openssl dgst -sha384 "$1" | awk '{print $NF}'
  fi
}

mkdir -p "$ASSET_DIR"
echo "build_book.sh: mermaid ${MERMAID_VERSION} from ${MERMAID_URL}"
# Download outside the book (the staged file must be composed below), verify the
# bytes, and only then write them where mdBook will pick them up.
DL="$(mktemp "${TMPDIR:-/tmp}/minfer-mermaid.XXXXXX")"
curl --fail --silent --show-error --location --retry 3 --retry-delay 2 \
  --output "$DL" "$MERMAID_URL"

actual="$(sha384_hex "$DL")"
if [ "$actual" != "$MERMAID_SHA384" ]; then
  echo "build_book.sh: sha384 mismatch — refusing to build with unverified bytes" >&2
  echo "  expected $MERMAID_SHA384" >&2
  echo "  actual   $actual" >&2
  exit 1
fi

# npm's dist file carries no header, and the MIT licence asks for the notice to
# travel with copies. mdbook-mermaid prepends exactly these two lines; reproducing
# them keeps the served bytes identical to the copy that was committed before #110.
{
  printf '/* MIT Licensed. Copyright (c) 2014 - 2022 Knut Sveidqvist */\n'
  printf '/* For license information please see https://github.com/mermaid-js/mermaid/blob/release/10.6.1/LICENSE */\n'
  cat "$DL"
} >"$LIB"

# The boot call is ours, not the CDN's: it is what renders every
# `<pre class="mermaid">` block the mdbook-mermaid preprocessor emits.
printf 'mermaid.initialize({startOnLoad:true});\n' >"$INIT"

if [ "$#" -eq 0 ]; then
  set -- build
fi
mdbook "$@"
