# minfer Inference Graph Demo (viz/)

An interactive web visualizer for minfer's inference compute graph. Two views of the
**same** graph are available (toggle on the right of the toolbar):

- **Operators** (default) — the layered tensor grid: one node = one operator, one edge = a
  tensor data flow.
- **Pipeline** — a semantic reasoning pipeline: one box = one function/stage
  (`RMSNorm` / `Attention` / `FFN` / `Logits`…), with arrows for the reasoning sequence.

Click any node/box to see that step's data (shape, dtype, the weight it reads, quantization
type, etc.).

A third page, **Shape flow** (`e2e.html`, link in the toolbar), leaves the graph abstraction and
plays one end-to-end forward pass instead: what *shape* the data has at every node, how it changes,
and what text the model is reading and writing. Jump to [the end-to-end shape flow
section](#end-to-end-shape-flow-e2ehtml).

Three kinds of data are supported:
- **Structure graph** (`--dump-graph-json`): graph + metadata, plays an execution-order animation.
- **Real trace** (`MINFER_TRACE`, P2): per-tensor **real statistics** for each node output
  (min/max/mean/abs-mean) + a downsampled-value heatmap + per-decode-step **token and
  logits top-5 distribution**; nodes are colored by this step's data magnitude and play
  continuously across steps.
- **Live streaming** (P3, `serve`): the page connects to minfer's OpenAI server and watches
  inference live over SSE — nodes light up one by one (with real stats and coloring), tokens
  stream out one at a time.

Zero dependencies: plain HTML + CSS + vanilla JS, no build toolchain, works straight out of the box.

## End-to-end shape flow (`e2e.html`)

The operator grid answers *what is in the graph*; this page answers *what shape is the data at each
node, and how does it change*. It plays one animated forward pass — prompt in, next token out — and
turns every shape change into something you can see instead of a number you have to read.

It is the second view of the same export, so the two pages share their data and their stage names:
the toolbar link **Shape flow →** jumps here, and **Operators view →** jumps back.

```bash
# static, no build step
cd viz && python3 -m http.server 8080     # → http://127.0.0.1:8080/e2e.html

# or let the engine serve it (same page, plus the live endpoints)
./target/release/minfer viz model.gguf    # → http://127.0.0.1:8081/e2e.html
```

Under `file://` the sample dropdown is unavailable (the page cannot fetch `samples/manifest.json`),
but **Open File…** and the built-in Qwen2.5-0.5B profile still work.

### What the page is made of

| Region | What it shows |
|---|---|
| Header, left | **The tensor at the current node**: the graph's own shape (`[151936, 2]`), element count, bytes, and a proportional card — **width ∝ token count, height ∝ log₂(feature dim)** |
| Header, middle | The stage: label, op, weight quant type, the shape formula with real numbers, and one line of what it does |
| Header, right | Counters: phase, layer, decode step, KV cells written, timeline position |
| Text strip | The prompt, then one chip per sampled token, popping in as the animation reaches the sampler (see below) |
| Rail (canvas) | The stage nodes in execution order, each showing the shapes it produced; the layer block is bracketed as `× N layers` |
| Footer, left | The KV cache as a `[kv_dim, n_ctx]` bar: prefill writes `nt` cells, every decode step writes one more |
| Footer, right | One chip per timeline step — click to jump |

A travelling packet — a small fixed-size chip carrying the step's primary shape, coloured by tensor
kind, printed inside the chip so it never covers a node label, and faded out as it lands — is what
connects them: the rail is the map, the packet is the data.

### The rail, node by node

One pass is the prologue, **one layer played in full**, a tick for the remaining layers, then the
epilogue. `nt` is the token count, `n_embd` the residual width:

| Stage | Op | Shape change |
|---|---|---|
| Tokenize | BPE | text → `[nt]` token ids (+ positions `0..nt-1`) |
| Embedding | `get_rows` | `[nt]` → `[n_embd, nt]` (a row gather, not a matmul) |
| RMSNorm | `rms_norm` | `[n_embd, nt]` → `[n_embd, nt]` (normalize over the feature dim) |
| QKV projection | `matmul ×3` (or one fused `fused_qkv`) | `[n_embd, nt]` → q `[n_head·hd, nt]`, k/v `[n_kv_head·hd, nt]` |
| QK-Norm | `qk_norm` | Qwen3 only; per-head norm, shapes unchanged |
| RoPE | `rope` | shapes unchanged — position enters the values, not the layout |
| KV cache write | `kvcache_store` | k/v → the persistent `[kv_dim, n_ctx]` region, columns `[pos, pos+nt)` |
| Attention | `attn` | `[n_head·hd, nt]`; inside: heads split, scores `[n_head, nt, n_kv]`, softmax, `·v`, heads merged |
| Output projection | `matmul` | `[n_head·hd, nt]` → `[n_embd, nt]` |
| Residual add | `add` | `[n_embd, nt]` → `[n_embd, nt]` |
| RMSNorm | `rms_norm` | `[n_embd, nt]`, before the FFN |
| Gate & Up projection | `matmul ×2` (or one fused `fused_ffn`) | `[n_embd, nt]` → `[n_ff, nt]` each — most of the weights live here |
| SiLU · SwiGLU | `silu` + `swiglu` | elementwise gating, `[n_ff, nt]` unchanged |
| Down projection | `matmul` | `[n_ff, nt]` → `[n_embd, nt]` |
| Residual add | `add` | back to `[n_embd, nt]`, ready for the next layer |
| Final RMSNorm | `rms_norm` | `[n_embd, nt]` |
| LM head | `matmul` | `[n_embd, nt]` → `[vocab, nt]` — the biggest single jump |
| Sample one token | softmax → top-k → top-p | one logits column `[vocab]` → one id `[1]` |
| Next step | loop | the id is appended; the next pass has `nt = 1` and one more KV cell |

`Prefill` plays the table above; `Decode` plays the same graph with `nt = 1` and a KV window of
`prompt + step`. Switching between them is the point of the page: **same graph, different shapes** —
`[896, nt] → [896, 1]`, and the score matrix `[n_head, nt, n_kv]` that grows by one column per step.

### Prefill and decode are not the same story

- **Prefill** writes `nt` cells and computes every position; the attention score matrix is
  `[n_head, nt, nt]`.
- **Decode** writes one cell per step; every activation is `[…, 1]` and the score matrix is
  `[n_head, 1, n_kv]`, stretching as the cache fills.
- The epilogue runs at `nt = 1` **even during prefill**, because only the last position needs
  logits — one of the two graph quirks the animation labels at the node where it happens.
- The KV cache bar is why `n_ctx` matters: the prompt fills a couple of cells out of 4096, and the
  decode steps walk one cell at a time.

### The text strip

The strip under the header is the model's input and output text, driven by the same playhead as the
rail: the prompt is there from the start, and every time the animation reaches the sampler stage one
more token appears (with a short pop and a running caret). Stepping back or replaying takes the
tokens away again — the text is *derived from the playhead*, never accumulated, so it can never
drift out of sync with the animation.

Where the text comes from:

- `MINFER_TRACE=trace.json` records the real tokens: the prompt is `Hello!` and the strip fills with
  `" I"`, `"'m"`, `" a"`… (quoted, so a leading space is visible). The very last sample shows as
  `#48948`, because an id in `logits_top` is all the trace recorded for it.
- A plain `--dump-graph-json` export carries no token text at all: the strip shows `⟨tok⟩`
  placeholders and an amber note saying so. It still moves in step with the rail.

Because a trace records one token per forward pass, loading one also caps the decode steps played
(3 recorded steps → 3 decode steps), so the strip and the rail stay 1:1.

### Fusion changes the node count, not the shapes

A decode graph runs the FusionPass, and the page follows it: `fused_qkv` shows up as one node
(`[1152, 1]` instead of q/k/v), `fused_ffn` as one `[2·n_ff, nt]`, and the RoPE stage is marked as
already applied inside the fused node. Fusion is per layer (it depends on the quant types meeting),
and the page plays the layer the graph actually fused — `fused(layer 0): qkv=false ffn=true` in the
0.5B decode sample — with a badge when only some layers are fused.

### Where the numbers come from

Everything is read from the export, never hardcoded per model:

- dims: `n_embd`, `n_layer`, `n_head`, `hd`, `n_kv_head`, `kv_dim`, `n_ff`, `vocab`, `n_ctx` and the
  token count, from the nodes and their metadata;
- stage shapes: computed by the stage template and **checked against the export** by
  `scripts/check_viz_e2e.mjs` (see below);
- weights and quant types: the exact names the graph reads, including tied embeddings (Qwen3's
  `matmul_token_embd.weight` is labelled as tied) and mixed q/k/v quant types;
- node grouping: by op **and** weight name (`blk.7.attn_q.weight` → the QKV stage), with the block
  boundary taken from the contiguous slice a block owns. That is why a new architecture needs no
  per-model table here — only a new stage, if it has one, in `TEMPLATE` (`viz/e2e-model.js`).

Two behaviours it surfaces instead of smoothing over:

- the exported prefill graph finishes the **last layer at `nt = 1`** (only the last position needs
  logits), so the epilogue shows `[n_embd, 1] → [vocab, 1]` even during prefill;
- attention output width is `n_head × hd`, which equals `n_embd` for Qwen2.5 but is **2× for
  Qwen3-0.6B** (16 heads × 128 = 2048 against `n_embd` = 1024); the O projection is what maps it
  back. Both were found by the shape check below, not by reading the code.

### Controls

| Control | What it does |
|---|---|
| **▶ / ❚❚** | Play / pause. Nothing moves until you press it — loading a graph or switching the range never auto-plays |
| **⏮ / ⏭** | One step back / forward (pauses) |
| **↻** | Replay from step 0 and start playing |
| Speed slider | Milliseconds per step (`1140` slow → `150` fast); decode steps and sweeps carry their own multipliers |
| **Run / Prefill / Decode** | Which range to play: the full prefill→decode run, one phase only, or decode only |
| **×N** | Include the `×N layers` tick between layer 0 and the epilogue |
| **decode detail** | Expand every decode step into every node instead of one sweep per step |
| Step chips (footer) | Jump to any step |
| `Space` / `←` `→` / `R` | Play-pause / step / restart |

### Checks

```bash
node scripts/check_viz_e2e.mjs          # stage shapes vs every exported sample graph
node scripts/check_viz_e2e_render.mjs   # boot + play every step on a stubbed DOM/canvas
```

The first is the valuable one: it requires the template's computed shape at every stage to equal the
shape `--dump-graph-json` actually wrote, for every file in `samples/` — the layer played in detail,
the attention internals, the KV region, and that decode is the same graph with `nt = 1` and a window
one cell longer. Both engine behaviours above were caught by it, so a wrong assumption fails the
gate instead of shipping a plausible-looking animation. The second boots the real `viz/e2e.js`
against a stubbed DOM and canvas, plays every step of every mode, switches samples and checks the
panel is never empty — it catches crashes in the render path, not visual regressions.

Both run in the `check-viz` CI job and need no dependencies (plain Node, no browser).

### Files

| File | What it is |
|---|---|
| `viz/e2e.html` | The page: toolbar, canvas, side panel, text strip, footer |
| `viz/e2e.css` | Styles (palette shared with `style.css`) |
| `viz/e2e-model.js` | The data layer: export → stage script with shape math. No DOM, so it is testable in Node |
| `viz/e2e.js` | The canvas renderer, animation and UI wiring |
| `scripts/check_viz_e2e.mjs` | Shape agreement with every sample (the gate above) |
| `scripts/check_viz_e2e_render.mjs` | Headless boot + full-timeline playthrough |
| `src/server/viz.rs` | The `minfer viz` routes for the four assets above (`no-store`, so a fresh page never pairs with a stale script) |

## Quick start

```bash
# 1. Serve a static server (the dropdown is unavailable under file://, but the "Open File…" button still works)
cd viz && python3 -m http.server 8080

# 2. Open in your browser
open http://127.0.0.1:8080/index.html
```

The page auto-loads the pre-generated samples in `samples/manifest.json` (including a real 0.5B
trace), or you can click "Open File…" to load any exported graph.

## Generating your own samples

### Structure graph (--dump-graph-json)

```bash
cargo build --release
./target/release/minfer --dump-graph-json graph.json <model.gguf> "Hello there"
# MINFER_DISABLE_MPS=1 → CPU coloring; --no-template "!" → decode graph (fused nodes)
```

### Real trace (MINFER_TRACE) — P2

```bash
MINFER_TRACE=trace.json ./target/release/minfer <model.gguf> "Hello!" -n 5
```

- Records every node output for prefill + each decode step: full stats + ≤64 downsampled values
  (GPU uses a staging blit, close to native speed; KV nodes are skipped at both ends)
- Each decode step also carries: the input token (with its decoded text) + this step's logits
  softmax top-5
- Works in single-shot CLI mode (not under `--cnv` / `serve`)

Drop the JSON into `samples/` and register it in `manifest.json` to make it appear in the dropdown.

### Live streaming (P3) — the self-contained `minfer viz` demo server

```bash
# All in one process: page + live events + trigger endpoint (default port 8081)
./target/release/minfer viz <model.gguf>          # or viz --port 9000 <model.gguf>
# Open http://127.0.0.1:8081/ in your browser — the page auto-detects and connects to the stream
```

- **GPU capture is staged, not per-node**: Metal blits each split's node outputs into
  host-visible staging within its single submit; CUDA queues one async pinned-D2H per node
  right after its launch (stream-ordered, safe against intra-split pool reuse) and drains the
  staging with a single sync at the split boundary. Both replace per-node flushes. The
  remaining capture tax is the host-side stats scan + event serialization of every node
  (~1.6× slowdown on 7B decode, identical on both backends)
- **The `/viz/graph` preview mirrors the engine's CParams**: GPU participation is Metal OR
  CUDA, and both decode fusions (QKV concat/mixed-quant, FFN gate+up) run on either GPU backend
  when the engine enables them — node ids match the live per-node events on every backend
  (`json::preview_fuse_flags` is shared by the export/trace/live paths and the engine)
- **Lazy arming**: per-node data is only captured while an SSE client is connected — a `viz`
  server with nobody watching, and normal CLI / `serve` inference, all cost nothing
  (`serve` is a pure OpenAI API with no viz routes)
- KV cache nodes (`kvcache_store/load`) are skipped at both ends because their data is huge
  (n_embd×n_ctx per layer) — the panel shows "no data for this step", consistent with fused
  orphan nodes
- The page is embedded and served at compile time; `samples/` is optionally loaded from disk
  (`MINFER_VIZ_DIR`, default `viz`), so offline samples and the live stream work together
- Endpoints: `GET /viz/graph` (prefill/decode graphs), `GET /viz/events` (SSE),
  `POST /viz/run` (`{prompt, max_tokens?, temperature?}`, chat template rendered server-side)
- The page calls `/viz/run` directly, and the events drive node lighting, magnitude coloring,
  the token strip, and the panel's logits distribution in real time

## How to use (user guide)

### 1. Load a graph
- **Model dropdown** (`Select a sample model`): pick a pre-generated sample from
  `samples/manifest.json` (structure graph or trace). Selecting one loads it.
- **Open File…**: load any exported graph JSON from disk
  (`minfer --dump-graph-json graph.json <model> "Hello"`), works under `file://` too.
- **`?file=samples/xxx.json`** deep link: load a specific sample via the URL.

### 2. Toolbar controls

Left — loading:
| Control | What it does |
|---|---|
| Model dropdown | Choose a sample graph |
| **Open File…** | Load a graph JSON from disk |

Middle — playback (only applies to the structure-graph / trace animation; disabled in live mode):
| Control | What it does |
|---|---|
| **⏮** | Previous execution step |
| **▶ / ⏸** | Play / pause the execution-order animation |
| **⏭** | Next execution step |
| **Speed slider** (`60ms`) | Milliseconds per step (animation speed) |
| **`Ready (click ▶ to start)`** | Status text. Shows the current step / progress while playing |

Right — filters & view:
| Control | What it does |
|---|---|
| **KV edges** | Show/hide KV cache read/write edges (`kvcache_store`/`kvcache_load`) |
| **Input/Output** | Show/hide input/output nodes |
| **Attention path** | Only show nodes on the attention path (Q/K/V matmuls, rope, attn, residual adds) |
| **FFN path** | Only show nodes on the FFN path (gate/up/down matmuls, silu, swiglu) |
| **Operators \| Pipeline** | Switch render mode: the operator grid vs. the semantic reasoning pipeline (§6) |
| **Fit** | Fit the whole graph to the window (zoom & pan) |
| **Legend** | Context-aware legend: operator list (Operators view) or stage/flowing legend (Pipeline view) |
| **Live** | Toggle the live-streaming panel (top-right) |

### 3. Reading the graph
- **Phase tabs** (`prefill · Live` / `decode · Live`, top-left of the graph area): switch between
  the prefill and decode graphs. Each phase has its own coloring from the last run.
- **Layered layout**: `start` → `blk.0 … blk.N` → `end`, one row per transformer layer.
  Click a **row label** to collapse/expand that layer.
- **Node coloring**: each box is one operator node. The fill color is the backend
  (Metal blue / CPU yellow / CUDA green) **and** is tinted by that node's data magnitude
  (abs-mean, blue → red) for the current/last step. A **dashed border** means a fused operator
  (e.g. `QKV✚`, `FFN✚`, `swiglu`).
- **Edges**: tensor data flow. KV edges (to/from the KV cache) are drawn differently and can be
  filtered with the **KV edges** checkbox.

### 4. Node inspector (right panel)
- With nothing selected, the panel shows the current/last step summary: **Input token** and
  **THIS STEP OUTPUT LOGITS TOP-5** probability bars.
- **Click any node** → the panel shows that node's details: op, backend, tensor shape/dtype,
  execution order, the weight it reads + quantization type, bias, in×out dims, upstream/downstream
  links, and "what this step does". With trace data it also shows the **tensor stats table**
  (min/max/mean/abs-mean) and a **downsampled value heatmap**.
- Following the **Upstream / Downstream** links (or drilling from a Pipeline stage's `Contained ops`)
  pushes onto a **panel nav stack**; a **`← Back`** button appears at the top-left to step back.

### 5. Live streaming (`Live` panel)
- Open the **Live** panel (top-right), point it at `minfer viz` (default
  `http://127.0.0.1:8081`), click **Connect**, enter a prompt, click **Run**.
- Nodes light up one by one with live stats/coloring; the token strip and the panel's logits
  distribution update as tokens stream out.

### 6. Pipeline view — the semantic reasoning pipeline

The toolbar has a **`Operators | Pipeline`** switch. **Operators** is the default layered
tensor grid (one box = one operator). **Pipeline** re-renders the *same* graph as a semantic
reasoning pipeline, where **one box = one function/stage** and the arrows are the reasoning
sequence:

```
Input → Embedding → [Transformer Layers × N] → Final RMSNorm → Logits → [Sampler]
```

- **Layers are collapsible.** The `Transformer Layers` box is a summary by default (`× N`);
  click it to expand into one row per layer:
  `RMSNorm → Attention → +Residual → RMSNorm → FFN → +Residual`. Click the `(click to collapse)`
  label to fold back. Each layer is a **thin dashed group frame** with internal padding, so the
  stage boxes (and their chips) sit inside it and are fully visible.
- **Each stage box is a function.** Its label is the stage (`Attention`, `FFN`, …), its sublabel
  is the aggregate `in×out / h / hd / nf / vocab`, the backend it runs on, and the contained ops
  are drawn as small chips inside. Executed boxes use a pale magnitude tint (blue→red by
  abs-mean) with **dark text** for high contrast.
- **Drill in with Back.** Click a stage/layer box → the inspector shows the stage's aggregate
  (backend, op count, the ops it owns) + "How this stage works". Click an op chip (or a
  `Contained ops` link) → the per-op inspector, with a **`← Back`** button in the top-left that
  returns to the previous view (the nav stack also covers Upstream/Downstream links).
- **Same animation/live data.** The stage/layer boxes are driven by the same execution cursor and
  the same `viz` SSE stream: as an op runs, its box lights up and is tinted by that op's
  abs-mean. The `Legend` button is **context-aware** — it shows the operator list in the
  Operators view and the stage/flow explanation (stages, pipeline order, box markings, backend
  colors) in the Pipeline view.
- **Empty state.** With no graph loaded, switching to Pipeline just shows the load prompt (no
  empty stage boxes).
- **Where the stages come from (no new instrumentation).** The stage for each op is derived at
  render time from the exported graph — `op` + the weight name in `meta.weight`
  (`blk.{i}.attn_norm`→`RMSNorm`, `.attn_qkv/.attn_output`→`Attention`, `.ffn_*`→`FFN`,
  `output_norm`→`Final RMSNorm`, `output/token_embd`→`Logits`, `token_embd`→`Embedding`). The
  final norm / logits are recognised by weight name so the residual chain's `layerOf` propagation
  can't fold them into the last layer. This works identically for the structure graph, the
  `MINFER_TRACE` sample, and live.

## JSON format

### Structure graph (--dump-graph-json)

```json
{
  "format": "minfer-graph", "version": 1,
  "model": "…", "kind": "prefill | decode",
  "inputs": [0, …], "outputs": [n, …],
  "nodes": [{
    "id": 0, "name": "token_ids", "op": "input",
    "detail": { …op payload… },
    "shape": [2,1,1,1], "dtype": "i32",
    "backend": "metal | cpu | cuda | null",
    "src": [ … ], "meta": { …weight/dims… }
  }]
}
```

### trace (MINFER_TRACE)

```json
{
  "format": "minfer-trace", "version": 1,
  "model": "…", "prompt": "…",
  "phases": [{
    "kind": "prefill | decode",
    "graph": { …the structure graph above… },
    "steps": [{
      "token": 358, "text": " I",             // decode input token (null for prefill)
      "logits_top": [[358, 0.364], …],        // this step's logits softmax top-5
      "nodes": [{ "id": 0, "dtype": "i32",
                  "stats": {min,max,mean,absmean},
                  "values": [ …downsampled… ], "stride": 8, "n": 896 }]
    }]
  }]
}
```

Note: the exported graph has already gone through the FusionPass, so it matches the runtime
execution graph (`silu+mul` is fused into `swiglu`; the fused-away `silu` node remains but has no
output buffer — it has no data in the trace and the page will say so).

## Roadmap

- **P1 (done)**: graph structure + metadata + playback animation + interaction
- **P2 (done)**: `MINFER_TRACE` real-data trace — node stats + downsampled values +
  decode token / logits top-5, with the page's heatmap, magnitude coloring, and token strip
- **P3 (done)**: SSE live streaming — `serve` + `/viz/graph` + `/viz/events`, where the page's
  "Live" panel lights up nodes and tokens in real time
