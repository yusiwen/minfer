# Build

Requirements: **Rust** (edition 2021, no ML-framework dependencies — runtime
deps are minimal). A **Nix flake devShell** (`nix develop`) is available for a
batteries-included dev environment.

## Build commands

```bash
# CPU + Metal (macOS) — plain build, never touches nvcc
cargo build --release
./target/release/minfer <model.gguf> "hello"

# CUDA (NVIDIA GPU) — opt-in feature, requires the CUDA toolkit (nvcc)
cargo build --release --features cuda
# statically-linked cudart (no libcudart.so runtime dep)
cargo build --release --features cuda,cuda_static

# + per-node debug dumps (MINFER_DUMP_DIR)
cargo build --release --features debug_dump
```

## GGUF tooling (F6) — no extra dependencies

`minfer convert` / `quantize` / `split` need no Python, no PyTorch and no
`gguf-py`: safetensors is parsed as a length-prefixed JSON header plus raw
bytes and `tokenizer.json`/`config.json` are plain JSON, both through the
already-present `serde_json`. They are host-side file work — no GPU backend is
initialized — and they build in every configuration (CPU, Metal, CUDA). The
writer/encoders and their verification references are documented in
[`GGUF-TOOLING.md`](./GGUF-TOOLING.md).

## Tests

```bash
cargo test --release                         # unit + integration, no model files needed
scripts/real_model_gates.sh                  # the #[ignore]d real-model gate set, serial
PARALLEL=1 scripts/real_model_gates.sh       # CPU-only parallel form (see the count below)
FEATURES=cuda scripts/real_model_gates.sh    # device gate set (serial, one GPU)
scripts/cuda_test.sh                         # the whole CUDA suite on a real GPU, serial
```

The `#[ignore]`d set needs the cached real models (the 0.5B for the default
configuration, `MINFER_BATCH_TEST_MODEL=…/Qwen3-0.6B-Q8_0.gguf` for the f16-KV one)
and writes temporary session files. `scripts/real_model_gates.sh` is the
documented entry point: it defaults to `--test-threads=1`, which a **device**
build requires (the CUDA state is a process-wide singleton, issue
[#64](https://github.com/yusiwen/minfer/issues/64)), and accepts `PARALLEL=1`
for the CPU-only parallel form. Since the KV storage format became **per engine**
([#99](https://github.com/yusiwen/minfer/issues/99)) the parallel harness no
longer makes one gate size another gate's KV regions — measured 2026-09-25 on a
CPU build: **28 / 0** serial and **27 / 1** parallel, where the one parallel
failure is the unrelated, load-sensitive wall-clock assertion in
`server_batch_matches_serial_and_is_faster`
([#154](https://github.com/yusiwen/minfer/issues/154)); before #99 the parallel
run was 19 / 9, every failure a KV-region width mismatch. The serial form stays
the documented one, and on a CUDA build it is the only honest one.

## Git hooks (git-hooks.nix)

The formatting gate is provided by [git-hooks.nix](https://github.com/cachix/git-hooks.nix)
(flake input) instead of a Cargo dev-dependency. Pinned in `flake.lock`; the
`rustfmt --check` hook runs with the project's pinned toolchain (1.97.1).

- The hook is **installed when entering the devShell** (`nix develop` installs
  `.git/hooks/pre-commit` + the `.pre-commit-config.yaml` symlink into the nix
  store). Committing **outside** the shell has no hook — those contributors
  should run `cargo fmt --all -- --check` manually first.
- Format drift is rejected, not auto-fixed: run `cargo fmt --all`, re-stage the
  changed files, then commit again.
- The same check runs sandboxed as a derivation: `nix flake check`
  (`checks.pre-commit-check`), or `nix develop -c pre-commit run --all-files`.

## macOS / Metal notes

On macOS the Metal backend is built in automatically: `build.rs` compiles
`src/metal.metal` → a precompiled `.metallib` via `/usr/bin/xcrun` at build
time (no per-run shader compile). If Xcode tools are unavailable the build
still succeeds and the shader is compiled from source at first run.

## CUDA build details

- nvcc is located via `PATH`, `CUDA_HOME`/`CUDA_PATH` (e.g. `/usr/local/cuda`);
  with `--features cuda` a missing toolkit is a **hard error** (with a clear
  message), and plain builds never touch nvcc at all.
- The host compiler is auto-detected: nvcc's default works when compatible;
  otherwise the first accepted GCC is pinned via `-ccbin` (e.g. a nix devShell
  putting GCC 15 first while CUDA 13 accepts ≤ GCC 13). Force one with
  `MINFER_CUDA_CCBIN=/path/to/g++`.
- GPU architectures are auto-detected from what the toolkit accepts (SASS for
  `sm_70`…`sm_121` as available, plus PTX for the highest and a backward-JIT
  `compute_70`/`compute_72` PTX) — one binary covers older and newer GPUs.
  The candidate list includes **sm_87/sm_88 (Jetson Orin)** explicitly: the
  device-tier gate routes sm_87 to the int8 BT path, which is compiled out of
  any PTX below sm_80 — native SASS per target is mandatory because PTX JITs
  forward only (docs 105–106, plan §14 R9). The minimum is **sm_70 (Volta)**: the kernels in `cuda_kernels.cu` use WMMA tensor
  cores (`nvcuda::wmma`), which require sm_70+, so Pascal (sm_61) is not a
  target. The Volta V100/Titan-V (sm_70/72) PTX is only emitted when nvcc
  supports it: CUDA 12.x does, **CUDA 13 removed Volta**, so keep Volta coverage
  by building with CUDA 12.8 (the only version supporting Volta + the Blackwell
  RTX 50 sm_120/121, which needs ≥ 12.8).
- **cudart linking** (mirrors llama.cpp's `GGML_STATIC`): by default `-lcudart`
  is a shared link, so the binary NEEDEDs `libcudart.so.N` and needs the CUDA
  toolkit runtime present at runtime (an rpath to `<cuda_home>/lib64` is baked
  in). Adding `cuda_static` links `libcudart_static.a` instead: the binary has
  **no** `libcudart.so` NEEDED dependency and only needs the NVIDIA driver
  (`libcuda.so.1`, dlopen'd lazily at runtime) + libstdc++ — deployable without
  a CUDA toolkit. The driver is never a link-time dependency in either mode.
- In CUDA builds the int8 tensor-core MMQ prefill path is **default-on**
  (runtime gates `MINFER_MMQ*`, see the README's
  [Performance](../README.md#performance) section /
  [CUDA_OPTIMIZATION.md](CUDA_OPTIMIZATION.md)).
