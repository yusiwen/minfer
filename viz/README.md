# minfer Inference Graph Demo (viz/)

An interactive web visualizer for minfer's inference compute graph. **Each node = one step of
inference** (one operator), and an edge = a tensor data flow. Click a node to see that step's
data (shape, dtype, the weight it reads, quantization type, etc.).

Three kinds of data are supported:
- **Structure graph** (`--dump-graph-json`): graph + metadata, plays an execution-order animation.
- **Real trace** (`MINFER_TRACE`, P2): per-tensor **real statistics** for each node output
  (min/max/mean/abs-mean) + a downsampled-value heatmap + per-decode-step **token and
  logits top-5 distribution**; nodes are colored by this step's data magnitude and play
  continuously across steps.
- **Live streaming** (P3, `--server`): the page connects to minfer's OpenAI server and watches
  inference live over SSE — nodes light up one by one (with real stats and coloring), tokens
  stream out one at a time.

Zero dependencies: plain HTML + CSS + vanilla JS, no build toolchain, works straight out of the box.

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
- Works in single-shot CLI mode (not under `--cnv` / `--server`)

Drop the JSON into `samples/` and register it in `manifest.json` to make it appear in the dropdown.

### Live streaming (P3) — the self-contained `minfer --viz` demo server

```bash
# All in one process: page + live events + trigger endpoint (default port 8081)
./target/release/minfer --viz <model.gguf>          # or --viz 9000 to pick a port
# Open http://127.0.0.1:8081/ in your browser — the page auto-detects and connects to the stream
```

- **MPS (GPU) capture uses a staging blit**: within each split's single submit, node outputs are
  blitted into host-visible staging and read back after sync — no more per-node flush, so GPU
  live streaming runs close to native speed
- **Lazy arming**: per-node data is only captured while an SSE client is connected — a `--viz`
  server with nobody watching, and normal CLI / `--server` inference, all cost nothing
  (`--server` is a pure OpenAI API with no viz routes)
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
| **Fit** | Fit the whole graph to the window (zoom & pan) |
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

### 5. Live streaming (`Live` panel)
- Open the **Live** panel (top-right), point it at `minfer --viz` (default
  `http://127.0.0.1:8081`), click **Connect**, enter a prompt, click **Run**.
- Nodes light up one by one with live stats/coloring; the token strip and the panel's logits
  distribution update as tokens stream out.


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
- **P3 (done)**: SSE live streaming — `--server` + `/viz/graph` + `/viz/events`, where the page's
  "Live" panel lights up nodes and tokens in real time
