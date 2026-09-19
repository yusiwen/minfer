# Usage

Examples use the built binary (`cargo build --release` first — see the
[Build section](README.md#build) of the project README); `cargo run --release
-- …` works identically.

```bash
./target/release/minfer <model> [prompt] [OPTIONS]
```

`<model>` can be a local path, a download URI, or a cached model name:

| Format | Example |
|--------|---------|
| Local file | `~/models/qwen2.gguf`, `./model.gguf`, `/abs/model.gguf` |
| Hugging Face | `hf:Qwen/Qwen2-0.5B-GGUF:qwen2-0.5b-q4_0.gguf` (auto-download) |
| Ollama | `ollama:qwen2.5:0.5b` (pull) |
| Cached model name | `qwen2.5-0.5b-instruct-q4_0` (resolved from `~/.cache/minfer/models`, see `list`) |

If `prompt` is omitted, reads from stdin. Run `minfer --help` for the full
option list; the subcommands are:

| Command | Purpose |
|---------|---------|
| `<model> [prompt] [OPTIONS]` | single-shot generation |
| `serve [--port N] [--n-ctx N] [--n-slots N] <model>` | OpenAI-compatible HTTP server |
| `info <model>` | print GGUF metadata + key tensors |
| `download hf <repo> [quant]` / `download ollama <model>[:tag]` | fetch models |
| `list` | list locally cached models |
| `viz [--port N] <model>` | self-contained viz server (default port 8081) |
| `bench [-p N] [-n N] [-r N] [-o md\|csv\|json] <model>` | perf test: `pp<P>` prefill / `tg<T>` decode, mean ± stddev over reps |
| `specverify [-p N] [-r N] [-o json\|md] <model>` | D5-1a gate bench: batched verify cost C_T(nt) + per-token amortization at deep KV |

Sampling and runtime options:

- `--temp <T>` — sampling temperature (default 0.8; `--greedy` = greedy decoding)
- `--top-k <K>` / `--top-p <P>` — top-K / nucleus sampling (defaults 40 / 0.95)
- `--repeat-penalty <N>` — repeat penalty (default 1.1; 1.0 = off), plus
  `--frequency-penalty` / `--presence-penalty`
- `--stop <STR>` — stop generation at this string (repeatable)
- `-n, --n-predict <N>` — max tokens to generate (default 512)
- `--seed <N>` — RNG seed for sampling
- `--n-ctx <N>` — sizes the KV cache (clamped to the model's max context)
- `-t, --threads <N>` — CPU worker threads

## Multi-turn conversation

`--cnv` (docs/CLI-CONVERSATION-PLAN.md): append-only KV + incremental template
rendering — each turn only prefills the new message delta, the whole
conversation accumulates in the KV cache:

```bash
./target/release/minfer --cnv qwen2.5-0.5b-instruct-q4_0           # interactive REPL
./target/release/minfer --cnv -st qwen2.5-0.5b-instruct-q4_0 "hi"  # single turn
```

In-conversation commands: `/exit` `/quit`, `/clear`, `/regen` (regenerate the
last reply), `/help`; EOF (Ctrl+D) exits.

Conversation options:

- `-st, --single-turn` — run one turn, then exit
- `--system <STR>` — system prompt
- `-mli, --multiline-input` — submit input on an empty line
- `--color on|off|auto` — color output (default auto = tty)
- `--session <FILE>` — save/load the conversation history as JSON; on overflow
  the oldest turns are dropped automatically and generation continues. The
  dropped turn's KV rows are removed in place and the tail is re-based (Phase C
  / C2), so only the new turn's delta is prefilled; `MINFER_NO_CONTEXT_SHIFT=1`
  forces the older, exact "drop the turns and re-prefill the rest" behaviour
  (an engine that cannot move rows — e.g. Metal, where it is Phase G — falls
  back to that path on its own and says so on stderr)

Qwen3-style `<think>…</think>` reasoning blocks are gray-highlighted
(single-shot mode too, when stdout is a terminal or `MINFER_COLOR=1`).

## OpenAI-compatible HTTP server

Continuous batching (the worker composes one decode batch across the active slots
instead of one forward per slot — Phase E / E2) is **on by default when the
model's forwards run on CUDA, and off on CPU/Metal** (E6). The reason is measured,
not assumed: on this project's reference CPU batching is *slower* than serving
requests one at a time (0.49x on 7B Q4_K_M, 0.88x on 0.5B Q4_0 with `--n-slots 4`
— the CPU decode kernels gain nothing from `nt > 1`, and concurrency forfeits the
cross-request prefix reuse each slot otherwise keeps), while on the GB10 it is
**1.97x faster** (7B Q4_K_M, four identical prompts, equal work, `--n-slots 4`,
default settings). The plan's E2 and E6 records have the tables.

- `MINFER_BATCH=1` forces batching on (this is how to batch on CPU, for
  experiments or for a machine where your own measurement says it wins).
- `MINFER_BATCH=0` forces it off.
- Any other value warns and uses the device default.
- Metal is deliberately never auto-enabled: the batched path needs an explicit
  attention span and Metal refuses that node, so it waits for Phase G.
- The server prints its choice at startup:
  `[server] batching: on (device cuda; MINFER_BATCH=1 forces it on, =0 forces it off)`.

Two fuse-related switches are easy to confuse (D3):

- `MINFER_NO_FUSE_QKV=1` / `MINFER_NO_FUSE_FFN=1` **disable** the corresponding
  decode fusion (the decoder builds the plain matmul/rope/store — or gate+up+
  silu+mul — path instead).
- `MINFER_FFN_COMPOSITION=1` keeps the FFN fusion but builds it as the proven
  **composition** (concat matmul + gate/up windows + in-place SwiGLU) instead of the
  hand-written fused node. It is the reference the A/B is run against, and it is
  ignored with a warning on a backend without offset views (Metal until G5).

Two environment switches around the GPU are easy to get wrong:

- `MINFER_DISABLE_CUDA` is checked for **presence**, not value: setting it to
  `0` *disables* CUDA (and therefore also turns the batching default off, since
  the model then runs on CPU). To force the CPU path deliberately use
  `MINFER_DISABLE_CUDA=1`; to use the GPU, leave it unset.
- On a device, batched *prefills* stay per request by construction (CUDA's
  `fa_prefill` tiles one query tile against one KV window — E1b), so `--n-slots`
  concurrency still pays one prefill per request, and prefix reuse across slots
  needs a cell copy (C3/D1). The batched-decode win is unaffected.


```bash
./target/release/minfer serve --n-ctx 4096 --n-slots 1 qwen2.5-0.5b-instruct-q4_0
# POST /v1/chat/completions  (stream + non-stream)
# GET  /v1/models, GET /health
```

## Performance testing (bench)

```bash
./target/release/minfer bench -r 3 <model>        # pp512 + tg128, markdown table
./target/release/minfer bench -p 3314 -n 128 <model>  # campaign-shape pp/tg
```

`pp<P>` ingests P prompt tokens (prefill-only, generate nothing); `tg<T>`
prefills the context then decodes T tokens. `-p 0` / `-n 0` skip a test, `-r`
sets the measured reps (1 untimed warmup each), `-o csv|json` emits the same
fields machine-readable, `--n-ctx` only ever grows the auto KV sizing
(`P+T+16`, clamped to the model's context length).

## Verify-step gate bench (specverify)

```bash
./target/release/minfer specverify -p 512 -r 40 -o json <model>
```

Measures the batched verify-step cost `C_T(nt)` (nt = 1, 3, 5 by default) at
a fixed deep KV depth and reports the per-token amortization
`nt·C_T(1)/C_T(nt)` — the D5 speculative-decoding gate instrument (step doc
81). `-p` sets the depth, `-r` the timed reps (3 untimed warmups each),
`MINFER_SPECVERIFY_NTS=1,3,16` overrides the phase list,
`MINFER_SPECVERIFY_NOUT=1` forces prefill-style `n_out=1`. Exit code is 0
whenever the measurement completes; the PASS/FAIL verdict is in the JSON.

## Examples

```bash
# Local model
./target/release/minfer ~/models/qwen2-0.5b-q4_0.gguf "What is the capital of France?"

# Cached model by name (no full path needed)
./target/release/minfer qwen2.5-0.5b-instruct-q4_0 "Hello"

# Auto-download from Hugging Face + run (quant auto-detected, splits included)
./target/release/minfer hf:Qwen/Qwen2.5-0.5B-Instruct-GGUF:qwen2.5-0.5b-instruct-q4_0.gguf "Hello"

# Inspect GGUF metadata + key tensors
./target/release/minfer info qwen2.5-0.5b-instruct-q4_0

# List available GGUF files in a HF repo (without downloading)
./target/release/minfer download hf Qwen/Qwen2.5-0.5B-Instruct-GGUF

# Pull from Ollama and create a symlink
./target/release/minfer download ollama qwen2.5:0.5b

# List locally cached models
./target/release/minfer list
```
