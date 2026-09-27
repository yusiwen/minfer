#!/usr/bin/env bash
# Lint the GitHub Actions workflow files with a pinned, checksum-verified actionlint.
#
# `.github/workflows/*.yml` is the one YAML surface no other gate reads. GitHub
# reports a workflow that does not *parse*, but not the mistakes actionlint finds
# (issue #178): an unknown `runs-on` label, a `types:` value that is not a real
# pull-request activity, an `if:` expression that cannot be evaluated, or a
# `needs:` naming a job that does not exist. Each of those is green on GitHub and
# red here.
#
# The binary is pinned the way `scripts/build_book.sh` pins mdBook and the Mermaid
# library: a release tarball plus a per-architecture sha256, so a mirrored or
# replaced artifact is refused instead of linted. Verified bytes are the whole
# verdict — which is why shellcheck integration is disabled explicitly
# (`-shellcheck=`). By default actionlint shells out to whatever `shellcheck` the
# machine happens to ship, so leaving it on would make the gate's *strictness*
# depend on the environment rather than on the pinned binary; `release.yml` also
# carries one pre-existing SC2086:info finding. Pinning shellcheck (and deciding
# that finding) is a follow-up issue, not part of this gate.
#
# Usage:
#   scripts/lint_workflows.sh                 # lint .github/workflows/*.yml
#   scripts/lint_workflows.sh FILE...         # lint the named files instead
#
# Environment:
#   MINFER_ACTIONLINT_URL  override the download URL (e.g. a mirror). The sha256
#                          pin below still applies, so a mirror must serve
#                          identical bytes.
set -euo pipefail

cd "$(dirname "$0")/.."

ACTIONLINT_VERSION="1.7.12"

# sha256 of each release artifact, taken from the release's own
# actionlint_<version>_checksums.txt.
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64 | Linux-amd64)
    ACTIONLINT_OS="linux"
    ACTIONLINT_ARCH="amd64"
    ACTIONLINT_SHA256="8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
    ;;
  Linux-aarch64 | Linux-arm64)
    ACTIONLINT_OS="linux"
    ACTIONLINT_ARCH="arm64"
    ACTIONLINT_SHA256="325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6"
    ;;
  Darwin-x86_64 | Darwin-amd64)
    ACTIONLINT_OS="darwin"
    ACTIONLINT_ARCH="amd64"
    ACTIONLINT_SHA256="5b44c3bc2255115c9b69e30efc0fecdf498fdb63c5d58e17084fd5f16324c644"
    ;;
  Darwin-arm64)
    ACTIONLINT_OS="darwin"
    ACTIONLINT_ARCH="arm64"
    ACTIONLINT_SHA256="aba9ced2dee8d27fecca3dc7feb1a7f9a52caefa1eb46f3271ea66b6e0e6953f"
    ;;
  *)
    echo "lint_workflows.sh: no pinned actionlint for $(uname -s)/$(uname -m)." >&2
    echo "  Add that platform's sha256 from the v${ACTIONLINT_VERSION} release, or run the linter in CI." >&2
    exit 127
    ;;
esac

TARBALL="actionlint_${ACTIONLINT_VERSION}_${ACTIONLINT_OS}_${ACTIONLINT_ARCH}.tar.gz"
URL="${MINFER_ACTIONLINT_URL:-https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/${TARBALL}}"

DL=""
cleanup() {
  rm -rf ${DL:+"$DL"}
}
trap cleanup EXIT

for tool in curl tar; do
  command -v "$tool" >/dev/null 2>&1 && continue
  echo "lint_workflows.sh: $tool not found on PATH." >&2
  exit 127
done

# sha256 as lowercase hex, using whichever tool the platform ships: GNU coreutils
# on Linux, `shasum` on macOS, `openssl` as the last resort.
sha256_hex() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | cut -d' ' -f1
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$1" | cut -d' ' -f1
  else
    openssl dgst -sha256 "$1" | awk '{print $NF}'
  fi
}

DL="$(mktemp -d "${TMPDIR:-/tmp}/minfer-actionlint.XXXXXX")"
echo "lint_workflows.sh: actionlint ${ACTIONLINT_VERSION} (${ACTIONLINT_OS}/${ACTIONLINT_ARCH}) from ${URL}"
curl --fail --silent --show-error --location --retry 3 --retry-delay 2 \
  --output "$DL/$TARBALL" "$URL"

actual="$(sha256_hex "$DL/$TARBALL")"
if [ "$actual" != "$ACTIONLINT_SHA256" ]; then
  echo "lint_workflows.sh: sha256 mismatch — refusing to lint with unverified bytes" >&2
  echo "  expected $ACTIONLINT_SHA256" >&2
  echo "  actual   $actual" >&2
  exit 1
fi

mkdir -p "$DL/x"
tar -xzf "$DL/$TARBALL" -C "$DL/x"

if [ "$#" -eq 0 ]; then
  set -- .github/workflows/*.yml
fi

# `-shellcheck=` disables actionlint's shellcheck integration; see the header.
"$DL/x/actionlint" -shellcheck= "$@"
