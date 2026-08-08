#!/usr/bin/env bash
# minfer GPU benchmark wrapper — asserts the MPS (Metal) path is actually
# active before trusting any timing. Guards the failure mode hit on 2026-08-06:
# a Metal shader compile error makes MpsState fall back to CPU SILENTLY, which
# reads as a huge "GPU throttling" slowdown (7 tok/s = CPU, not a hot GPU).
#
# Usage:
#   scripts/bench.sh <minfer args...>     # run + assert MPS active
#   scripts/bench.sh --health <model>     # 30-token prefill sanity (>= 200 tok/s)
set -u

MINFER_BIN="${MINFER_BIN:-./target/release/minfer}"

if [ "${1:-}" = "--health" ]; then
    model="${2:?usage: bench.sh --health <model>}"
    out=$("$MINFER_BIN" --greedy -n 1 "$model" "hi" 2>&1) || { echo "$out"; exit 1; }
    if echo "$out" | grep -q 'GPU acceleration enabled'; then
        prefill=$(echo "$out" | grep 'Prefill:' | grep -oE '[0-9.]+ tok/s' | head -1)
        tok=$(echo "$prefill" | grep -oE '^[0-9.]+')
        echo "GPU: MPS active, prefill ${prefill}"
        if [ -n "$tok" ] && awk -v t="$tok" 'BEGIN{exit !(t+0 < 200)}'; then
            echo "FATAL: prefill ${tok} tok/s (< 200) — GPU throttled or CPU fallback; benchmark data invalid" >&2
            exit 1
        fi
        exit 0
    fi
    echo "FATAL: MPS not active — GPU path unavailable (Metal shader compile error?)" >&2
    echo "$out" | grep -iE 'shader|not available|cpu fallback' | head >&2
    exit 1
fi

# Normal run: execute, echo output, then assert MPS was used.
tmp=$(mktemp)
trap 'rm -f "$tmp"' EXIT
"$MINFER_BIN" "$@" >"$tmp" 2>&1
cat "$tmp"

if [ "${MINFER_DISABLE_MPS:-}" = "1" ]; then
    exit 0
fi
if grep -q 'GPU acceleration enabled' "$tmp"; then
    exit 0
fi
echo "ERROR: MPS GPU path is NOT active — refusing the benchmark as invalid." >&2
grep -iE 'shader|not available|cpu fallback' "$tmp" | head >&2
exit 1
