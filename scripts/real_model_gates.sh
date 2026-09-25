#!/usr/bin/env bash
# Run the real-model gate set — the `#[ignore]`d tests — the way the harness must.
#
# Why this exists (issue #99). These gates load the cached real models and measure
# the engine end to end. Until #99 they had to be driven serially by hand, because
# the C4 packed-cache gate
# (`models::qwen2::graph::tests::a_packed_kv_cache_answers_like_the_f32_one`) set the
# process-wide KV format for its measurement runs: while it ran, every other test
# building a graph in the same process sized its KV nodes for the *packed* format
# and `ensure_kv` refused the mismatch on the next rebuild. The parallel harness
# therefore reported a red set (measured at master `a756419`: 19 passed / 9 failed)
# while the serial run was green (28 passed / 0 failed).
#
# #99 made the format **per engine** (the loaded model resolves
# `MINFER_CACHE_TYPE` once, `CParams::kv_format` carries it into the graph, and
# `GraphAllocator::set_kv_format` gives the CPU kernels the same answer), so all nine
# KV-region failures are gone from the parallel run. One parallel failure remained:
# `server_batch_matches_serial_and_is_faster` asserted a wall-clock relation from two
# sequential whole-workload measurements, so a loaded harness let the first-measured
# phase absorb the start-up wave — measured **21.20s batched vs 9.95s serial (0.47x)**
# in parallel against **1.50x** for the same binary serially. #154 replaced that with
# interleaved matched rounds and the **median of the per-round `serial/batched`
# ratios** (the shape #123 gave the CUDA map-window gate); the correctness comparison
# stays a separate full-length pair. Measured on this box 2026-09-25 (CPU): serial
# **29 passed / 0 failed**; parallel **29 passed / 0 failed** in every run (3 plain
# harness runs plus 1 under 16 extra CPU spinners), the #154 gate's median
# 1.379-1.468x plain and 2.159x under the extra load.
#
# A **device** build is a different story: the CUDA state is a process-wide
# singleton (`CudaState`: MMQ memo, captured graph execs, stream state), so the
# device set must run with one thread (issue #64). The wrapper therefore defaults to
# **serial when `FEATURES` includes `cuda`, and to the parallel form otherwise** —
# on a CPU-only build the device reason does not exist, the KV-format reason is gone
# (#99), and the parallel harness is where the #154 timing gate's robustness is
# exercised, so the default CPU command runs it.
#
# Usage:
#   scripts/real_model_gates.sh [extra cargo test args...]
#
#   scripts/real_model_gates.sh                    # CPU-only: parallel, 29 passed / 0 failed
#   PARALLEL=0 scripts/real_model_gates.sh         # CPU-only: serial, 29 passed / 0 failed
#   FEATURES=cuda scripts/real_model_gates.sh      # device set, serial (correct: #64)
#   PARALLEL=1 scripts/real_model_gates.sh         # force parallel (CPU-only; see #64 on a device)
#
# The device set's second configuration is a second run of the same command:
#   MINFER_BATCH_TEST_MODEL=~/.cache/minfer/models/hf/Qwen/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q8_0.gguf \
#     FEATURES=cuda scripts/real_model_gates.sh
set -euo pipefail
cd "$(dirname "$0")/.."

# The CUDA toolkit path the CUDA suite already pins (scripts/cuda_test.sh). Harmless
# on a CPU-only run: no CUDA code is compiled without `--features cuda`.
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

cargo_args=(--release --bin minfer)
[ -n "${FEATURES:-}" ] && cargo_args+=(--features "$FEATURES")
cargo_args+=(-- --ignored)

# Serial on a device build (the process-wide `CudaState`, issue #64); parallel on a
# CPU-only build, where that reason does not exist.
parallel=1
case ",${FEATURES:-}," in
  *cuda*) parallel=0 ;;
esac
# An explicit PARALLEL=1/0 wins.
case "${PARALLEL:-}" in
  1) parallel=1 ;;
  0) parallel=0 ;;
esac
if [ "$parallel" = "1" ]; then
  case ",${FEATURES:-}," in
    *cuda*)
      echo "real_model_gates: warning: PARALLEL=1 on a CUDA build contradicts issue #64 — the" >&2
      echo "                  device state is a process-wide singleton, so a device measurement may" >&2
      echo "                  be perturbed by a concurrent test. Serial is the honest device form." >&2
      ;;
  esac
else
  cargo_args+=(--test-threads=1)
fi
cargo_args+=("$@")

echo "+ cargo test ${cargo_args[*]} (parallel=$parallel)"
cargo test "${cargo_args[@]}"
