#!/usr/bin/env bash
# The MacBook gate (#678): what a change to the Metal path must run, and the record
# it must leave in the PR body.
#
# CI cannot run this. GitHub's hosted macOS runners have no Metal device at all —
# measured 2026-10-24, PR #675: 13 device tests failed with no `MPS:` line in the log,
# so `MTLCreateSystemDefaultDevice()` returned nil before anything was compiled
# (docs/GATE-CONTRACT.md §5, issue #674). Until a self-hosted runner exists, this
# script *is* the Metal gate, and `scripts/check_pr_body.py --metal true` refuses a
# Metal-path PR whose `Mac verification` section is `N/A` or lacks the record.
#
# Usage, on the Mac, on the branch tip (the commit goes into the record):
#
#     scripts/mac_gate.sh
#
# It prints the block to paste into the PR body, and exits non-zero if anything
# fails. `caffeinate` is not decoration: a machine that sleeps mid-run cuts the
# suite in half and the numbers are then meaningless.
#
# What it cannot prove: that the pasted record came from this run. The checker
# validates the shape, a reviewer validates the content.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 2

# The box label is part of the record's identity (gate contract rule 5): absolute, and
# ASCII so it survives a paste. `MINFER_GATE_BOX` overrides it for a machine whose
# hostname is not the label the records use.
BOX=${MINFER_GATE_BOX:-"(hostname -s)"}
BOX=$(printf '%s (macOS %s, %s)' "${MINFER_GATE_BOX:-$(hostname -s)}" \
    "$(sw_vers -productVersion)" "$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo 'Apple Silicon')")
COMMIT=$(git rev-parse --short HEAD)
DATE=$(date '+%Y-%m-%d %H:%M %Z')
LOAD=$(sysctl -n vm.loadavg | tr -d '{}' | awk '{print $1}')

# `scripts/identity_battery.sh` bounds each run with `timeout`, which macOS does not
# ship: without it every run fails and the battery reports "4 diverged", which reads
# like a Metal bug. Provide the bound when coreutils' `gtimeout` is installed, and a
# bare shim (the command, no bound) otherwise — the shim is honest about what it is,
# so a run that hangs is visible as a hang rather than as a wrong answer.
SHIM=$(mktemp -d)
if command -v timeout >/dev/null 2>&1; then
    :
elif command -v gtimeout >/dev/null 2>&1; then
    printf '#!/bin/sh\nshift\nexec gtimeout 600 "$@"\n' > "$SHIM/timeout"
    chmod +x "$SHIM/timeout"
else
    printf '#!/bin/sh\nshift\nexec "$@"\n' > "$SHIM/timeout"
    chmod +x "$SHIM/timeout"
    echo "mac_gate: no timeout/gtimeout — the identity battery runs unbounded" >&2
fi
export PATH="$SHIM:$PATH"

fail=0
run() {
    local label="$1"
    shift
    echo "=== $label: $*" >&2
    if ! caffeinate -dimsu "$@"; then
        echo "FAILED: $label" >&2
        fail=1
        return 1
    fi
}

run build cargo build --release
run suite bash -c 'set -o pipefail; cargo test --release 2>&1 | tee /tmp/mac-gate-suite.log'
COUNTS=$(grep -E '^test result' /tmp/mac-gate-suite.log 2>/dev/null |
    sed -E 's/^test result: (ok|FAILED)\. ([0-9]+) passed; ([0-9]+) failed; ([0-9]+) ignored.*/\1 \2 \3 \4/' |
    awk '{p+=$2; f+=$3; i+=$4; n++} END {printf "%d passed / %d failed / %d ignored", p, f, i}')
TARGET="/Volumes/WD_BLACK/models/hf/Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"
DRAFT="/Volumes/WD_BLACK/models/hf/Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
IDENTITY="not run (the 7B/0.5B GGUF pair is not on this box)"
if [ -f "$TARGET" ] && [ -f "$DRAFT" ]; then
    if run identity bash -c "bash scripts/identity_battery.sh '$TARGET' '$DRAFT' 2>&1 | tee /tmp/mac-gate-identity.log"; then
        IDENTITY=$(grep -oE '[0-9]+/[0-9]+ identical, [0-9]+ diverged' /tmp/mac-gate-identity.log | tail -1)
        IDENTITY=${IDENTITY:-"ran (see /tmp/mac-gate-identity.log)"}
    fi
fi

echo
if [ "$fail" -ne 0 ]; then
    echo "mac_gate: FAILED — do not record this run" >&2
    exit 1
fi
echo "mac_gate: paste this into the PR body's \`Mac verification\` section:" >&2
cat <<RECORD
$BOX / $DATE / $COMMIT / load $LOAD
cargo build --release -> 0 · cargo test --release -> $COUNTS · identity battery -> $IDENTITY
RECORD
