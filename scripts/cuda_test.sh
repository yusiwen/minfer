#!/usr/bin/env bash
# Run the CUDA suite the way the device expects to be used.
#
# The CUDA device state is a process-wide singleton (`CudaState`): its MMQ memo,
# captured graph execs and stream/pool state are shared by every test in the
# process. The test harness runs test functions in parallel threads by default, so
# one test can clear or populate those caches while another is measuring a forward
# — which shows up as a *within-test* determinism failure (issue #64).
#
# Serial execution removes that interleaving. It is also what the real-model CPU
# tests in docs/KNOWN-CPU-ISSUES-2026-08-29.md already do, for the same reason.
#
# Usage: scripts/cuda_test.sh [extra cargo test args...]
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
cargo test --release --features cuda -- --test-threads=1 "$@"
