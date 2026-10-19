#!/usr/bin/env bash
# Lint the GitHub Actions workflow files with pinned, checksum-verified actionlint
# *and* shellcheck.
#
# `.github/workflows/*.yml` is the one YAML surface no other gate reads. GitHub
# reports a workflow that does not *parse*, but not the mistakes actionlint finds
# (issue #178): an unknown `runs-on` label, a `types:` value that is not a real
# pull-request activity, an `if:` expression that cannot be evaluated, or a
# `needs:` naming a job that does not exist. Each of those is green on GitHub and
# red here.
#
# actionlint's shell integration is enabled against a **pinned** shellcheck (#215).
# Left to itself, actionlint shells out to whatever `shellcheck` the machine ships,
# which would make the gate's *strictness* depend on the runner image (present here
# → more findings; absent there → silently fewer) instead of on the pinned binaries.
# Both tools are therefore fetched as release tarballs and verified against a
# per-architecture sha256; unverified bytes are refused, never linted.
#
# Findings are zero on `master`, and that is a decision, not an absence: turning the
# integration on exposed four (#215) — two were rewritten, one was grouped, and the
# one intentional literal keeps a disable directive *at its site* with its reason.
#
# Usage:
#   scripts/lint_workflows.sh                 # lint .github/workflows/*.yml
#   scripts/lint_workflows.sh FILE...         # lint the named files instead
#
# Environment:
#   MINFER_ACTIONLINT_URL  override the actionlint download URL (e.g. a mirror).
#                          The sha256 pin below still applies, so a mirror must
#                          serve identical bytes.
#   MINFER_SHELLCHECK_URL  the same for the shellcheck tarball.
set -euo pipefail

cd "$(dirname "$0")/.."

ACTIONLINT_VERSION="1.7.12"
SHELLCHECK_VERSION="0.10.0"

# sha256 of each release artifact. actionlint's are taken from the release's own
# actionlint_<version>_checksums.txt; shellcheck publishes no checksum file, so
# those are the sha256 of the release tarballs as GitHub serves them.
#
# Two tools, two architecture spellings — actionlint says amd64/arm64, shellcheck
# x86_64/aarch64 — so both are resolved for the same host in one table.
case "$(uname -s)-$(uname -m)" in
  Linux-x86_64 | Linux-amd64)
    ACTIONLINT_OS="linux"; ACTIONLINT_ARCH="amd64"; ACTIONLINT_SHA256="8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"
    SHELLCHECK_OS="linux"; SHELLCHECK_ARCH="x86_64"; SHELLCHECK_SHA256="6c881ab0698e4e6ea235245f22832860544f17ba386442fe7e9d629f8cbedf87"
    ;;
  Linux-aarch64 | Linux-arm64)
    ACTIONLINT_OS="linux"; ACTIONLINT_ARCH="arm64"; ACTIONLINT_SHA256="325e971b6ba9bfa504672e29be93c24981eeb1c07576d730e9f7c8805afff0c6"
    SHELLCHECK_OS="linux"; SHELLCHECK_ARCH="aarch64"; SHELLCHECK_SHA256="324a7e89de8fa2aed0d0c28f3dab59cf84c6d74264022c00c22af665ed1a09bb"
    ;;
  Darwin-x86_64 | Darwin-amd64)
    ACTIONLINT_OS="darwin"; ACTIONLINT_ARCH="amd64"; ACTIONLINT_SHA256="5b44c3bc2255115c9b69e30efc0fecdf498fdb63c5d58e17084fd5f16324c644"
    SHELLCHECK_OS="darwin"; SHELLCHECK_ARCH="x86_64"; SHELLCHECK_SHA256="ef27684f23279d112d8ad84e0823642e43f838993bbb8c0963db9b58a90464c2"
    ;;
  Darwin-arm64)
    ACTIONLINT_OS="darwin"; ACTIONLINT_ARCH="arm64"; ACTIONLINT_SHA256="aba9ced2dee8d27fecca3dc7feb1a7f9a52caefa1eb46f3271ea66b6e0e6953f"
    SHELLCHECK_OS="darwin"; SHELLCHECK_ARCH="aarch64"; SHELLCHECK_SHA256="bbd2f14826328eee7679da7221f2bc3afb011f6a928b848c80c321f6046ddf81"
    ;;
  *)
    echo "lint_workflows.sh: no pinned actionlint/shellcheck for $(uname -s)/$(uname -m)." >&2
    echo "  Add both tools' sha256 for that platform from the pinned releases, or run the linter in CI." >&2
    exit 127
    ;;
esac

ACTIONLINT_TARBALL="actionlint_${ACTIONLINT_VERSION}_${ACTIONLINT_OS}_${ACTIONLINT_ARCH}.tar.gz"
ACTIONLINT_URL="${MINFER_ACTIONLINT_URL:-https://github.com/rhysd/actionlint/releases/download/v${ACTIONLINT_VERSION}/${ACTIONLINT_TARBALL}}"
SHELLCHECK_TARBALL="shellcheck-v${SHELLCHECK_VERSION}.${SHELLCHECK_OS}.${SHELLCHECK_ARCH}.tar.xz"
SHELLCHECK_URL="${MINFER_SHELLCHECK_URL:-https://github.com/koalaman/shellcheck/releases/download/v${SHELLCHECK_VERSION}/${SHELLCHECK_TARBALL}}"

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
# GNU tar unpacks .xz by running the `xz` binary; BSD tar (macOS) does it in-process.
if [ "$(uname -s)" = "Linux" ] && ! command -v xz >/dev/null 2>&1; then
  echo "lint_workflows.sh: xz not found on PATH — GNU tar needs it for the shellcheck .tar.xz." >&2
  exit 127
fi

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

# fetch_verified URL OUT WANT NAME — download, then refuse any byte the pin does not
# expect. A mirror (the MINFER_*_URL overrides) must serve identical bytes.
fetch_verified() {
  local url="$1" out="$2" want="$3" name="$4" actual
  curl --fail --silent --show-error --location --retry 3 --retry-delay 2 \
    --output "$out" "$url"
  actual="$(sha256_hex "$out")"
  if [ "$actual" != "$want" ]; then
    echo "lint_workflows.sh: ${name} sha256 mismatch — refusing to lint with unverified bytes" >&2
    echo "  expected $want" >&2
    echo "  actual   $actual" >&2
    exit 1
  fi
}

DL="$(mktemp -d "${TMPDIR:-/tmp}/minfer-lint-workflows.XXXXXX")"

echo "lint_workflows.sh: actionlint ${ACTIONLINT_VERSION} (${ACTIONLINT_OS}/${ACTIONLINT_ARCH}) from ${ACTIONLINT_URL}"
echo "lint_workflows.sh: shellcheck ${SHELLCHECK_VERSION} (${SHELLCHECK_OS}/${SHELLCHECK_ARCH}) from ${SHELLCHECK_URL}"
fetch_verified "$ACTIONLINT_URL" "$DL/$ACTIONLINT_TARBALL" "$ACTIONLINT_SHA256" "actionlint ${ACTIONLINT_VERSION}"
fetch_verified "$SHELLCHECK_URL" "$DL/$SHELLCHECK_TARBALL" "$SHELLCHECK_SHA256" "shellcheck ${SHELLCHECK_VERSION}"

mkdir -p "$DL/actionlint" "$DL/shellcheck"
tar -xzf "$DL/$ACTIONLINT_TARBALL" -C "$DL/actionlint"
tar -xJf "$DL/$SHELLCHECK_TARBALL" -C "$DL/shellcheck"

ACTIONLINT_BIN="$DL/actionlint/actionlint"
SHELLCHECK_BIN="$DL/shellcheck/shellcheck-v${SHELLCHECK_VERSION}/shellcheck"

for bin in "$ACTIONLINT_BIN" "$SHELLCHECK_BIN"; do
  if [ ! -x "$bin" ]; then
    echo "lint_workflows.sh: $bin is missing from its verified tarball — refusing to lint." >&2
    exit 1
  fi
done

if [ "$#" -eq 0 ]; then
  set -- .github/workflows/*.yml
fi

# Both tools come from the pins above, never from PATH.
"$ACTIONLINT_BIN" -shellcheck="$SHELLCHECK_BIN" "$@"
