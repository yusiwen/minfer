# Usage

Examples use the built binary (`cargo build --release` first — see the
[Build section](../README.md#build) of the project README); `cargo run --release
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
- **F3 sampler set** ([#48](https://github.com/yusiwen/minfer/issues/48)) — every default below
  leaves the pre-F3 chain unchanged:
  - `--min-p <P>` — drop tokens whose probability is below `P * max` (0 = off; 1.0 = argmax only)
  - `--typical <P>` — locally typical sampling (1.0 = off; 0 keeps the single most typical token)
  - `--xtc-probability <P>` / `--xtc-threshold <T>` — XTC: with probability `P`, exclude the top
    choices whose probability is at least `T` (`T` <= 0.5; 0 = off)
  - `--dry-multiplier <N>` — DRY (Don't Repeat Yourself) penalty strength (0 = off), with
    `--dry-base <N>` (default 1.75), `--dry-allowed-length <N>` (default 2),
    `--dry-penalty-last-n <N>` (default 64) and `--dry-sequence-breakers <L>` — restart sequences
    as *token ids*, e.g. `--dry-sequence-breakers 198;13,2` (`;` between sequences, `,` between ids)
  - `--mirostat <0|1|2>` — mirostat off / v1 / v2, with `--mirostat-tau <N>` (target surprise in
    bits, default 5.0), `--mirostat-eta <N>` (learning rate, default 0.1) and `--mirostat-m <N>`
    (v1 estimator window, default 100). In mirostat mode the temperature is ignored (mirostat's
    `mu` truncation subsumes it); `--temp 0` still means greedy. Mirostat cannot be combined with
    `--spec-draft`.
  - `--logit-bias <L>` — add to raw logits: `ID:BIAS` pairs separated by `,`, repeatable, e.g.
    `--logit-bias 15043:-2.0,198:1.5`. A token id outside the vocabulary, or a bias outside
    `[-100, 100]`, is refused at startup.
  A nonsensical value for any of these is refused at startup (exit 1), never silently ignored.
- `--stop <STR>` — stop generation at this string (repeatable)
- `-n, --n-predict <N>` — max tokens to generate (default 512)
- `--seed <N>` — RNG seed for sampling
- `--n-ctx <N>` — sizes the KV cache (clamped to the model's max context)
- `-t, --threads <N>` — CPU worker threads
- `--gpu-layers <N|auto>` — E5: run the first `N` transformer blocks on the device and the rest
  on the CPU (`0` = CPU only; unset/`MINFER_GPU_LAYERS` = every block the device can hold, the
  pre-E5 behaviour; `auto` = as many as the budget allows). The placement is printed at load
  (`offload: 4 of 24 blocks on cuda, 20 on cpu; embed/output on cpu (32.0 MiB of device
  weights; --gpu-layers 4)`), and the tensors outside the blocks — `token_embd`, the final norm
  and `lm_head` — stay on the CPU unless every block is offloaded.
- `MINFER_GPU_MEM <MiB>` — the weight budget `auto` fits into; unset = three quarters of the
  device's free bytes (the same default the activation gate uses). `auto` also holds back a
  quarter of that budget for the KV arenas and the activation pool, and reports what it decided
  (`offload: … auto: 5 of 24 blocks fit — weights budget 64 MiB, 16 MiB reserved for
  KV/activations; MINFER_GPU_MEM=64 MiB`). Without a device that reports free memory (Metal
  today) `auto` needs `MINFER_GPU_MEM`.

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
- `--session <FILE>` (with `--cnv`) — save/load the conversation. `FILE` is the
  history as JSON; `FILE.kv` is a **KV session companion** written next to it (C5)
  that carries the rows those messages were rendered into, plus the host state they
  belong to. On start a matching companion is resumed and the history is **not**
  re-prefilled — the run prints `resumed N message(s) and M KV row(s) … — 0 tokens
  prefilled`; anything that does not match this run (another `--n-ctx`, another
  model's `n_kv_embd`, another `MINFER_CACHE_TYPE`, a history the user edited, an
  older file version) prints the reason and falls back to re-rendering the JSON,
  which is always correct if slower. On overflow the oldest turns are dropped
  automatically and generation continues: the dropped turn's KV rows are removed in
  place and the tail is re-based (Phase C / C2), so only the new turn's delta is
  prefilled; `MINFER_NO_CONTEXT_SHIFT=1` forces the older, exact "drop the turns and
  re-prefill the rest" behaviour (an engine that cannot move rows — e.g. Metal,
  where it is Phase G — falls back to that path on its own and says so on stderr)

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

- `--slots-file <PATH>` (batched engine only) — resume the server's slot contexts from
  `PATH` at startup and rewrite it after every completed request (C5 S2). A request whose
  prompt matches a restored slot's tokens prefills only its own delta; the file also carries
  the KV rows, so a restart no longer re-prefills the conversations that had finished. The
  startup line prints the size it will write per request (about 12 MiB for a 0.5B/512-row
  arena), because that cost is a decision. A snapshot from another `--n-slots` or `--n-ctx`
  (or another model / KV element type) is refused loudly and the server starts empty.
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

The KV cache type is its own switch (C4): `MINFER_CACHE_TYPE=f32|f16|q8_0`, strict — an
unknown value fails the load on every device, `f16` resolves to `f32` on the CPU (no f16
KV kernel there) and `q8_0` is refused on CUDA and Metal until their kernels land
([#87](https://github.com/yusiwen/minfer/issues/87)). A packed `q8_0` cache is 3.76×
smaller and, since C4 S2, is read by a fused `Q8_0 × Q8_0` K dot with V accumulated out of
the cell; `MINFER_NO_FUSED_Q8_KV=1` restores the older dequantize-into-a-scratch read for
the A/B. Details, numbers and the named tolerance class: `docs/BACKENDS.md`,
`docs/ARCHITECTURE-EXECUTION-PLAN.md` §5 (C4).

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
