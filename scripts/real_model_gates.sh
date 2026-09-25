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
# KV-region failures are gone from the parallel run. Measured on a CPU build
# 2026-09-25: serial **29 passed / 0 failed**, parallel **28 passed / 1 failed**. The
# one parallel failure is not a KV failure: `server_batch_matches_serial_and_is_faster`
# asserts a wall-clock relation from two sequential whole-workload measurements, so a
# loaded harness lets the first-measured phase absorb the start-up wave (15.60s
# batched vs 11.54s serial). That fragility is filed as
# https://github.com/yusiwen/minfer/issues/154; until it is robust, this wrapper's
# default (serial) is the documented entry point.
#
# A **device** build is a different story: the CUDA state is still a process-wide
# singleton (`CudaState`: MMQ memo, captured graph execs, stream state), so the
# device set must run with one thread (issue #64). This wrapper defaults to
# `--test-threads=1`, which is correct on every build and is the honest "one
# command" entry point for the gates.
#
# Usage:
#   scripts/real_model_gates.sh [extra cargo test args...]
#
#   PARALLEL=1 scripts/real_model_gates.sh        # CPU-only: 27 passed / 1 failed (see #154)
#   FEATURES=cuda scripts/real_model_gates.sh     # device set, serial (correct)
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
# Serial unless the caller explicitly asks for the CPU-only parallel form.
[ "${PARALLEL:-0}" = "1" ] || cargo_args+=(--test-threads=1)
cargo_args+=("$@")

echo "+ cargo test ${cargo_args[*]}"
cargo test "${cargo_args[@]}"
