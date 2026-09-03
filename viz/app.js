/* minfer 推理流程图演示 — 零依赖 vanilla JS。
 *
 * 数据来源：minfer `--dump-graph-json <path>` 导出的图 JSON。
 * P1：结构 + 元数据；若节点带 stats/values（P2 trace），面板自动多渲染一截。
 */

"use strict";

/* ---------------- 算子展示名 / 说明 ---------------- */

const OP_LABEL = {
  input: "input", add: "add", mul: "mul", scale: "scale", silu: "silu",
  softmax: "softmax", rms_norm: "rms_norm", qk_norm: "qk_norm",
  matmul: "matmul", get_rows: "embed", rope: "rope", attn: "attn",
  kvcache_store: "KV store", kvcache_load: "KV load", view: "view",
  reshape: "reshape", permute: "permute", swiglu: "swiglu",
  fused_bias_rope: "bias+rope", batch_matmul: "bmm",
  fused_qkv: "QKV✚", fused_ffn: "FFN✚", fused_qkv_norm: "QKV✚N",
};

const OP_EXPLAIN = {
  input: "Leaf input: token id / positions / KV indices. Not computed; filled externally each step (KV position is data, not structure).",
  get_rows: "Embedding lookup: take the row of the weight matrix for each token id (token embedding), giving a token vector.",
  rms_norm: "RMSNorm: for the last dim, x / sqrt(mean(x²)+eps) × γ. The transformer pre-norm.",
  qk_norm: "Per-head RMSNorm (Qwen3 decoupled head dim): treat [nt·nh, hd] as a matrix and normalize each row of Q/K.",
  matmul: "Matrix multiply: activation × weight. Weights are GGUF-quantized (Q4_K etc.), dequantized per block.",
  add: "Element-wise add: residual connection (add a block input back to its output) or a bias.",
  mul: "Element-wise multiply (FFN gate × up).",
  silu: "SiLU activation: x·sigmoid(x). The activation of the FFN gate branch.",
  swiglu: "Fused SwiGLU: silu(gate)·up, both ops in one kernel.",
  rope: "Rotary position embedding: rotate Q/K by the token position, injecting position info.",
  attn: "Attention: Q·Kᵀ·V (GQA), with scale and softmax. Reads the historical K/V from this layer's KV cache.",
  kvcache_store: "Write the current token's K/V into this layer's persistent cache region. Positions come from the positions input, so the graph topology is independent of context length.",
  kvcache_load: "Read all historical K/V from this layer's KV cache — O(1) incremental autoregressive read, avoiding recompute.",
  fused_qkv: "Decode fusion: concatenated wq|wk|wv matmul + bias + RoPE + KV write, one kernel (llama's attn_bias_rope_store).",
  fused_qkv_norm: "Decode QKV fusion with per-head Q/K RMSNorm (Qwen3): concatenated wq|wk|wv matmul + per-head qk_norm + no-bias RoPE + KV write, one kernel.",
  fused_ffn: "Decode fusion: concatenated ffn_gate|ffn_up matmul + in-place swiglu, one kernel.",
  fused_bias_rope: "Fused bias + RoPE op.",
  view: "View node (not constructed in the current architecture).",
  reshape: "Shape transform (not constructed in the current architecture).",
  permute: "Dimension permute (not constructed in the current architecture).",
  scale: "Scale (in-vocabulary op, not currently constructed).",
  softmax: "Softmax (in-vocabulary op; fused inside the attention kernel, so no standalone node).",
  batch_matmul: "Batched matmul (planned fused variant).",
};

const BACKEND_NAME = { metal: "Metal (MPS)", cpu: "CPU", cuda: "CUDA", none: "unassigned" };

/* ---------------- 全局状态 ---------------- */

const state = {
  doc: null,
  nodes: [], byId: new Map(),
  edges: [], edgeById: new Map(), edgeKey: new Map(),
  order: [], // 执行顺序 = 节点 id 顺序（builder 保证拓扑序）
  rows: [],  // [{name, nodes:[id], y, collapsed}]
  rowOf: new Map(), depthOf: new Map(), consumers: [],
  cursor: -1,
  playing: false, timer: null,
  speed: 60,
  filters: { kv: true, io: true, attn: false, ffn: false },
  collapsed: new Set(),
  selected: null,
  view: { x: 0, y: 0, k: 1 },
  // P2 trace mode (minfer-trace docs)
  trace: null,        // the trace doc
  phases: [],         // [{kind, graph, steps}]
  phaseIdx: 0,
  stepIdx: 0,
  stepData: null,     // Map nodeId -> {stats, values, stride, n, dtype}
  live: false,        // P3 live (SSE) mode
  p2: null,
  // Flow view (semantic reasoning pipeline)
  viewMode: "graph",      // "graph" (op grid) | "flow"
  flow: null,             // built in buildFlow()
  flowBoxEls: new Map(),  // blockKey -> {g, rect, x, y, w, h, members}
  flowLayerExpanded: false,
  selectedBlock: null,    // Flow container key (when a stage/segment is selected)
  panelNav: [],           // right-panel navigation stack: [{kind:"node"|"block", id|key}]
};

const SVG_NS = "http://www.w3.org/2000/svg";
const NODE_H = 30, ROW_H = 108, X_PAD = 76, HGAP = 26;

/* ---------------- 工具 ---------------- */

function el(tag, attrs, parent) {
  const e = document.createElementNS(SVG_NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}
function htmlEscape(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  }[c]));
}
function fmtSize(n) {
  const bytes = n * 4; // f32
  if (bytes >= 1 << 30) return (bytes / (1 << 30)).toFixed(2) + " GiB";
  if (bytes >= 1 << 20) return (bytes / (1 << 20)).toFixed(2) + " MiB";
  if (bytes >= 1 << 10) return (bytes / (1 << 10)).toFixed(1) + " KiB";
  return bytes + " B";
}
function shapeStr(shape) {
  const s = shape.join("×");
  return "[" + s + "]";
}

/* ---------------- 数据准备 ---------------- */

function prepare(doc) {
  const nodes = doc.nodes.map(n => ({ ...n, layer: undefined }));
  const byId = new Map(nodes.map(n => [n.id, n]));
  const consumers = nodes.map(() => []);

  // 执行顺序 = id 顺序（导出即拓扑序）
  const order = nodes.map(n => n.id);

  // 边
  const edges = [];
  for (const n of nodes) {
    for (const s of n.src) {
      edges.push({ id: edges.length, src: s, dst: n.id });
      consumers[s].push(n.id);
    }
  }

  // 层分组
  const layerRe = /blk\.(\d+)\./;
  function layerOf(id, seen = new Set()) {
    const n = byId.get(id);
    if (n.layer !== undefined) return n.layer;
    if (seen.has(id)) return null;
    seen.add(id);
    const hay = n.name + " " + JSON.stringify(n.meta || {}) + " " + JSON.stringify(n.detail || {});
    const m = hay.match(layerRe);
    if (m) { n.layer = parseInt(m[1], 10); return n.layer; }
    if (n.meta && typeof n.meta.layer === "number") { n.layer = n.meta.layer; return n.layer; }
    if (n.detail && typeof n.detail.layer === "number") { n.layer = n.detail.layer; return n.layer; }
    // 残差 add 的 src 同时含上一层输出与当前层输出：取最大层（当前层）
    let best = null;
    for (const s of n.src) {
      const l = layerOf(s, seen);
      if (l !== null) best = best === null ? l : Math.max(best, l);
    }
    n.layer = best;
    return best;
  }
  for (const n of nodes) layerOf(n.id);

  // 深度（最长路径）
  const depth = new Map();
  function depthOf(id) {
    if (depth.has(id)) return depth.get(id);
    const n = byId.get(id);
    let d = 0;
    for (const s of n.src) d = Math.max(d, depthOf(s) + 1);
    depth.set(id, d);
    return d;
  }
  for (const n of nodes) depthOf(n.id);

  // 行：start → blk.0..N → end
  const blkMap = new Map();
  const start = [], end = [];
  for (const n of nodes) {
    if (n.layer === null) {
      if (n.op === "input" || n.op === "get_rows") start.push(n.id);
      else end.push(n.id);
    } else {
      if (!blkMap.has(n.layer)) blkMap.set(n.layer, []);
      blkMap.get(n.layer).push(n.id);
    }
  }
  const rows = [{ name: "start", nodes: start, collapsed: false }];
  for (const [layer, ids] of [...blkMap.entries()].sort((a, b) => a[0] - b[0])) {
    rows.push({ name: "blk." + layer, layer, nodes: ids, collapsed: false });
  }
  rows.push({ name: "end", nodes: end, collapsed: false });
  const rowOf = new Map();
  rows.forEach((r, i) => r.nodes.forEach(id => rowOf.set(id, i)));

  // 行内排序：(depth, id)
  for (const r of rows) {
    r.nodes.sort((a, b) => (depthOf(a) - depthOf(b)) || (a - b));
  }

  // 边索引 src>dst
  const edgeById = new Map();
  for (const e of edges) edgeById.set(e.src + ">" + e.dst, e);

  return { nodes, byId, consumers, edges, edgeById, order, rows, rowOf, depth };
}

function nodeIsInputOutput(n) {
  return n.op === "input" || state.doc.outputs.includes(n.id);
}
function nodeOnAttnPath(n) {
  const m = n.meta || {}, name = n.name;
  if (n.op === "input" || n.op === "get_rows") return true;
  if (["rope", "attn", "kvcache_store", "kvcache_load", "fused_qkv", "fused_bias_rope"].includes(n.op)) return true;
  if (n.op === "matmul" || n.op === "rms_norm") {
    const w = m.weight || "";
    if (String(w).includes("attn")) return true;
  }
  if (n.op === "add" && name.includes("attn")) return true;
  if (n.op === "matmul" && (m.weight || "").includes("qkv")) return true;
  return false;
}
function nodeOnFfnPath(n) {
  const m = n.meta || {}, name = n.name;
  if (n.op === "input" || n.op === "get_rows") return true;
  if (["silu", "mul", "swiglu", "fused_ffn"].includes(n.op)) return true;
  if (n.op === "matmul" || n.op === "rms_norm") {
    const w = m.weight || "";
    if (String(w).includes("ffn")) return true;
  }
  if (n.op === "add" && name.includes("ffn")) return true;
  return false;
}

/* ---------------- 布局 ---------------- */

function layout() {
  const pos = new Map();
  let y = 24;
  for (const r of state.rows) {
    r.y = y;
    let x = X_PAD;
    r.maxLabelW = 0;
    for (const id of r.nodes) {
      const n = state.byId.get(id);
      const label = OP_LABEL[n.op] || n.op;
      const w = Math.max(58, label.length * 6.6 + 16);
      n._w = w; n._label = label;
      n._x = x;
      pos.set(id, { x, y: y + NODE_H / 2, w, label });
      r.maxLabelW = Math.max(r.maxLabelW, w);
      x += w + HGAP;
    }
    r.w = x - HGAP - X_PAD;
    y += ROW_H;
  }
  return pos;
}

/* ---------------- 渲染 ---------------- */

const gNodes = document.getElementById("nodes");
const gEdges = document.getElementById("edges");
const viewport = document.getElementById("viewport");
const canvas = document.getElementById("canvas");

let nodeEls = [], edgeEls = [];

function renderGraph() {
  if (state.viewMode === "flow") { renderFlow(); return; }
  renderGraphGrid();
}

function renderGraphGrid() {
  gNodes.textContent = ""; gEdges.textContent = "";
  nodeEls = []; edgeEls = [];
  const pos = layout();
  const n = state.nodes.length;

  // 行标签
  for (const r of state.rows) {
    const t = el("text", {
      x: 6, y: r.y + NODE_H / 2 + 4, class: "row-label",
    }, gNodes);
    t.textContent = r.name + (r.collapsed ? " ▸" : " ▾");
    t.addEventListener("click", () => {
      r.collapsed = !r.collapsed;
      renderGraph(); applyFilters(); updateStepInfo(); fit();
    });
  }

  // 节点
  for (let i = 0; i < n; i++) {
    const node = state.nodes[i];
    const p = pos.get(node.id);
    if (!p) continue;
    const g = el("g", { class: "node-group" }, gNodes);
    g.setAttribute("data-id", node.id);
    const w = p.w, cx = p.x + w / 2;
    const rect = el("rect", { class: "node-rect", x: p.x, y: p.y - NODE_H / 2, width: w, height: NODE_H, rx: 6 }, g);
    el("text", { class: "node-label", x: cx, y: p.y + 3.5 }, g).textContent = p.label;
    el("text", { class: "node-sublabel", x: cx, y: p.y + NODE_H / 2 + 11 }, g).textContent = shortName(node.name);
    const title = el("title", {}, g);
    title.textContent = `${node.name}\n${p.label} · ${shapeStr(node.shape)} ${node.dtype}\nbackend: ${BACKEND_NAME[node.backend || "none"]}`;
    g.addEventListener("click", ev => { ev.stopPropagation(); selectNode(node.id); });
    nodeEls[node.id] = g;
  }

  // 边
  for (const e of state.edges) {
    const s = pos.get(e.src), d = pos.get(e.dst);
    if (!s || !d) continue;
    const path = el("path", { class: "edge", d: edgePath(s, d) }, gEdges);
    edgeEls[e.id] = path;
  }

  // 后端类别
  for (const n of state.nodes) {
    const g = nodeEls[n.id];
    g.classList.add("backend-" + (n.backend || "none"));
    if (n.op.startsWith("fused") || n.op === "swiglu") g.classList.add("fused");
  }
  // 边类别：KV / 跨行残差
  for (const e of state.edges) {
    const p = edgeEls[e.id];
    const srcOp = state.byId.get(e.src).op, dstOp = state.byId.get(e.dst).op;
    if (srcOp === "kvcache_store" || srcOp === "kvcache_load" || dstOp === "kvcache_store" || dstOp === "kvcache_load") {
      p.classList.add("kv");
    } else if (state.rowOf.get(e.src) !== state.rowOf.get(e.dst)) {
      p.classList.add("residual");
    }
  }
}

function shortName(name) {
  const s = String(name);
  return s.length > 26 ? s.slice(0, 24) + "…" : s;
}

function edgePath(s, d) {
  const sx = s.x + s.w, sy = s.y;
  const tx = d.x, ty = d.y;
  const dx = Math.min(Math.max(Math.abs(tx - sx) / 2, 26), 150);
  return `M ${sx} ${sy} C ${sx + dx} ${sy}, ${tx - dx} ${ty}, ${tx} ${ty}`;
}

/* ================================================================
 * Flow 视图：语义推理流水线
 * 每个框 = 一个函数 / 语义阶段（Input / Embedding / 每层 [norm→attn→ffn] /
 * Final Norm / Logits / Sampler）。从算子图 + 权重名派生，无需新仪器。
 * 复用现有播放 / 直播数据：执行某一算子时点亮其所属的阶段框 + 层框。
 * ================================================================ */

const STAGE_NAME = {
  norm1: "RMSNorm", norm2: "RMSNorm", attn: "Attention",
  add_attn: "+ Residual", add_ffn: "+ Residual", ffn: "FFN",
  add: "+ Residual", other: "Other",
};
const STAGE_ORDER = { norm1: 1, attn: 2, add_attn: 3, norm2: 4, ffn: 5, add_ffn: 6, other: 7, add: 3 };

const CHIP_W = 74, CHIP_H = 16, CHIP_GX = 6, CHIP_GY = 4, CHIPS_PER_ROW = 2;
const SEG_W = 200, SEG_HDR = 44, SEG_GAP = 64, STAGE_GAP = 24, BOX_PAD = 10, LAYER_GAP = 34, LABEL_H = 18;
const STAGE_MIN_W = 130, LAYER_PAD = 14;

/* 一个 op 属于哪个语义阶段（依赖 meta.weight + op）。 */
function stageOfNode(n) {
  const op = n.op;
  const w = String((n.meta && n.meta.weight) || "");
  if (op === "rms_norm") return /attn_norm|input_layernorm/.test(w) ? "norm1" : "norm2";
  if (op === "qk_norm") return "norm1";
  if (op === "rope" || op === "attn" || op === "kvcache_store" || op === "kvcache_load" || op === "fused_bias_rope" || op === "fused_qkv" || op === "fused_qkv_norm") return "attn";
  if (op === "matmul") {
    if (/ffn|_gate|_up|_down|ffn_gu/.test(w)) return "ffn";
    if (/attn|qkv|q_proj|k_proj|v_proj|attn_output|output/.test(w)) return "attn";
    return "other";
  }
  if (op === "silu" || op === "mul" || op === "swiglu" || op === "fused_ffn") return "ffn";
  if (op === "add") return "add"; // 由 srcs 再解析
  return "other";
}

/* 由算子图派生 flow 结构（phase / layer / sub-stage，以及 op→block 映射）。 */
function buildFlow() {
  const nodes = state.nodes, byId = state.byId;
  const phaseOf = new Map(), layerOf = new Map(), sub = new Map();
  let maxLayer = -1, firstLayer = Infinity, layerCount = 0;
  for (const n of nodes) {
    const op = n.op;
    let phase;
    if (op === "input") phase = "input";
    else if (op === "get_rows") phase = "embed";
    else {
      const w = String((n.meta && n.meta.weight) || "");
      // final norm / logits 按权重名单独识别 —— `prepare` 的 layerOf 会把残差
      // 链上的 output_norm / logits matmul 也传播进最后一层，不能靠 layer 判定。
      if (op === "rms_norm" && /output_norm|model\.norm|final_norm/.test(w)) phase = "final_norm";
      else if (op === "matmul" && !/^blk\./.test(w) && /output|lm_head|token_embd/.test(w)) phase = "logits";
      else if (op === "add") {
        // 位置嵌入 add：src 全部来自 input/embed → 并入 embed 段
        let allEmbed = n.src.length > 0;
        for (const s of n.src) { const p = phaseOf.get(s); if (p !== "input" && p !== "embed") { allEmbed = false; break; } }
        phase = allEmbed ? "embed" : (n.layer === null || n.layer === undefined ? "other" : "layer");
      }
      else if (n.layer === null || n.layer === undefined) phase = (op === "matmul" ? "logits" : (op === "rms_norm" ? "final_norm" : "other"));
      else phase = "layer";
    }
    phaseOf.set(n.id, phase);
    const L = n.layer;
    layerOf.set(n.id, L);
    let s = phase;
    if (phase === "layer") {
      s = stageOfNode(n);
      if (s === "add") {
        let attn = false, ffn = false;
        for (const src of n.src) {
          const ss = sub.get(src);
          if (ss === "ffn") { ffn = true; break; }
          if (ss === "attn" || ss === "add_attn") attn = true;
        }
        s = ffn ? "add_ffn" : (attn ? "add_attn" : "add");
      }
    }
    sub.set(n.id, s);
    if (phase === "layer") { if (L !== null && L !== undefined) { maxLayer = Math.max(maxLayer, L); firstLayer = Math.min(firstLayer, L); } }
  }

  // 收集各 bucket 的算子 id（按 id 升序 = 构建/执行序）
  const buckets = { input: [], embed: [], final_norm: [], logits: [], other: [] };
  const layerBuckets = new Map();  // layer -> { stage: Map<sub, [id]> , ids: [] }
  for (const n of nodes) {
    const ph = phaseOf.get(n.id);
    if (ph === "layer") {
      const L = layerOf.get(n.id);
      let lb = layerBuckets.get(L);
      if (!lb) { lb = { layer: L, stages: new Map(), ids: [] }; layerBuckets.set(L, lb); }
      lb.ids.push(n.id);
      const s = sub.get(n.id);
      if (!lb.stages.has(s)) lb.stages.set(s, []);
      lb.stages.get(s).push(n.id);
    } else {
      buckets[ph].push(n.id);
    }
  }

  // 有层数则按层编号排序
  const layers = [...layerBuckets.values()].sort((a, b) => (a.layer - b.layer));
  layerCount = layers.length;

  // 组装 segments
  const segments = [];
  segments.push({ key: "input", type: "seg", name: "Input", caption: "token_ids · positions", ids: buckets.input });
  segments.push({ key: "embed", type: "seg", name: "Embedding", caption: "token_embd lookup", ids: buckets.embed });
  const layersSeg = {
    key: "layers", type: "layers", name: "Transformer Layers",
    count: layerCount, allIds: layers.flatMap(l => l.ids),
    layers: layers.map(l => ({
      layer: l.layer, ids: l.ids,
      stages: [...l.stages.entries()].map(([k, ids]) => ({ key: k, name: STAGE_NAME[k] || k, ids }))
            .sort((a, b) => (STAGE_ORDER[a.key] || 9) - (STAGE_ORDER[b.key] || 9)),
    })),
  };
  segments.push(layersSeg);
  segments.push({ key: "final_norm", type: "seg", name: "Final RMSNorm", caption: "pre-logits norm", ids: buckets.final_norm });
  segments.push({ key: "logits", type: "seg", name: "Logits", caption: "lm_head matmul", ids: buckets.logits });
  if (buckets.other.length) segments.push({ key: "other", type: "seg", name: "Other", caption: "unclassified", ids: buckets.other });
  if (state.doc && state.doc.kind === "decode") segments.push({ key: "sample", type: "seg", name: "Sampler", caption: "top-k → next token", ids: [], synthetic: true });

  // op -> block 映射
  const blockKeyOf = new Map(), layerKeyOf = new Map();
  for (const n of nodes) {
    const ph = phaseOf.get(n.id);
    if (ph === "layer") {
      blockKeyOf.set(n.id, `stage:${layerOf.get(n.id)}:${sub.get(n.id)}`);
      layerKeyOf.set(n.id, `layer:${layerOf.get(n.id)}`);
    } else {
      blockKeyOf.set(n.id, `seg:${ph}`);
      layerKeyOf.set(n.id, null);
    }
  }

  state.flow = {
    segments, blockKeyOf, layerKeyOf, phaseOf, sub, layerOf,
    firstLayer, maxLayer, layerCount,
    stageName: (k) => STAGE_NAME[k] || k,
  };
}

/* 一组算子的聚合信息：主导后端 + 一句 sub（该阶段的体量/参数）。 */
function flowAggregate(ids, stageKey) {
  let ncpu = 0, nmetal = 0, ncuda = 0;
  let dim = "", nhead = "", hd = "", nf = "", vocab = "", kind = "";
  for (const id of (ids || [])) {
    const n = state.byId.get(id); if (!n) continue;
    const m = n.meta || {};
    if (n.backend === "metal") nmetal++; else if (n.backend === "cuda") ncuda++; else if (n.backend === "cpu") ncpu++;
    if (m.in_dim !== undefined) dim = `${m.in_dim}×${m.out_dim !== undefined ? m.out_dim : (m.nf !== undefined ? m.nf : "…")}`;
    if (m.n_head !== undefined) nhead = m.n_head;
    if (m.hd !== undefined) hd = m.hd;
    if (m.nf !== undefined) nf = m.nf;
    if (m.vocab_size !== undefined) vocab = m.vocab_size;
    if (m.mode) kind = `${m.mode}`;
  }
  const backend = (nmetal && nmetal >= ncpu && nmetal >= ncuda) ? "metal"
    : (ncuda && ncuda >= ncpu) ? "cuda"
    : (ncpu ? "cpu" : "none");
  let sub = `${(ids || []).length} op${(ids || []).length !== 1 ? "s" : ""}`;
  if (stageKey === "attn") sub += ` · h=${nhead} hd=${hd}`;
  if (stageKey === "ffn") sub += ` · nf=${nf}`;
  if (stageKey === "embed") sub += ` · vocab=${vocab}`;
  if (stageKey === "logits") sub += ` · in×out=${dim}`;
  if (!sub && kind) sub = kind;
  return { backend, sub, dim };
}

/* 每个算子在 flow 里的一个小框（可点选，复用现有 inspector）。 */
function flowOp(id, x, y, w, h) {
  const n = state.byId.get(id);
  const label = OP_LABEL[n.op] || n.op;
  const fused = n.op.startsWith("fused") || n.op === "swiglu";
  const g = el("g", { class: "flow-op backend-" + (n.backend || "none") + (fused ? " fused" : "") }, gNodes);
  g.setAttribute("data-id", id);
  el("rect", { x, y, width: w, height: h, rx: 4 }, g);
  el("text", { class: "flow-op-label", x: x + w / 2, y: y + h / 2 + 3 }, g).textContent = label;
  const title = el("title", {}, g);
  title.textContent = `${n.name}\n${label} · ${shapeStr(n.shape)} ${n.dtype}\nbackend: ${BACKEND_NAME[n.backend || "none"]}`;
  g.addEventListener("click", ev => { ev.stopPropagation(); selectNode(id); });
  nodeEls[id] = g;
  return g;
}

/* 一个可点选的 flow 容器框（阶段 / 段）。 */
function flowContainer(key, x, y, w, h, o) {
  const g = el("g", { class: "flow-box backend-" + (o.backend || "none") }, gNodes);
  g.setAttribute("data-block", key);
  const rect = el("rect", { class: "flow-box-rect", x, y, width: w, height: h, rx: 8 }, g);
  el("text", { class: "flow-box-label", x: x + BOX_PAD, y: y + 15 }, g).textContent = o.label;
  if (o.sub) el("text", { class: "flow-box-sub", x: x + BOX_PAD, y: y + 28 }, g).textContent = o.sub;
  if (o.dim) el("text", { class: "flow-dim", x: x + BOX_PAD, y: y + 38 }, g).textContent = o.dim;
  const title = el("title", {}, g);
  title.textContent = `${o.label}${(o.members && o.members.length) ? ` · ${o.members.length} ops` : ""}${o.sub ? `\n${o.sub}` : ""}`;
  if (o.onClick) g.addEventListener("click", ev => { ev.stopPropagation(); o.onClick(); });
  state.flowBoxEls.set(key, { g, rect, x, y, w, h, members: o.members || [], label: o.label });
  return g;
}

function flowLayerContainer(key, x, y, w, h, o) {
  // Layer 容器只是"分组框"：半透明底 + 细边框（不整块染色/不遮内容），stage 框画在其上。
  const g = el("g", { class: "flow-layer" }, gNodes);
  g.setAttribute("data-block", key);
  const rect = el("rect", { class: "flow-layer-bg", x, y, width: w, height: h, rx: 8 }, g);
  el("text", { class: "flow-layer-label", x: x + LAYER_PAD, y: y + 13 }, g).textContent = o.label;
  g.addEventListener("click", ev => { ev.stopPropagation(); o.onClick(); });
  state.flowBoxEls.set(key, { g, rect, x, y, w, h, members: o.members || [], label: o.label });
  return g;
}

function stageGeom(st, x, y) {
  const ids = st.ids;
  const cols = Math.min(CHIPS_PER_ROW, Math.max(1, ids.length));
  const rows = Math.ceil(ids.length / Math.max(1, cols));
  const chipH = rows ? rows * (CHIP_H + CHIP_GY) - CHIP_GY : 0;
  const h = SEG_HDR + chipH + BOX_PAD;
  const gridW = cols * (CHIP_W + CHIP_GX) - CHIP_GX + BOX_PAD * 2;
  const w = Math.max(STAGE_MIN_W, gridW);
  return { x, y, w, h };
}

function drawVArrow(a, b) {
  const ax = a.x + a.w / 2, ay = a.y + a.h, by = b.y;
  el("path", { class: "flow-arrow", d: `M ${ax} ${ay} L ${ax} ${by - 9} M ${ax} ${by - 9} L ${ax - 5} ${by - 15} M ${ax} ${by - 9} L ${ax + 5} ${by - 15}` }, gEdges);
}
function drawHArrow(a, b, y) {
  const ax = a.x + a.w, bx = b.x;
  el("path", { class: "flow-arrow", d: `M ${ax} ${y} L ${bx - 6} ${y} M ${bx - 6} ${y} L ${bx - 12} ${y - 5} M ${bx - 6} ${y} L ${bx - 12} ${y + 5}` }, gEdges);
}

function drawStageBox(st, x, y, layer) {
  const key = `stage:${layer}:${st.key}`;
  const ids = st.ids;
  const agg = flowAggregate(ids, st.key);
  const geom = stageGeom(st, x, y);
  const w = geom.w, h = geom.h;
  const g = flowContainer(key, x, y, w, h, {
    label: STAGE_NAME[st.key] || st.key, sub: agg.sub, dim: agg.dim ? `in×out ${agg.dim}` : "",
    backend: agg.backend, members: ids, onClick: () => flowSelectBlock(key),
  });
  let cx = x + BOX_PAD, cy = y + SEG_HDR;
  ids.forEach((id, i) => {
    if (i > 0 && i % CHIPS_PER_ROW === 0) { cx = x + BOX_PAD; cy += CHIP_H + CHIP_GY; }
    flowOp(id, cx, cy, CHIP_W, CHIP_H);
    cx += CHIP_W + CHIP_GX;
  });
  return { key, x, y, w, h };
}

function drawLayerRow(layer, x, y) {
  // pass 1: 计算各 stage 框位置/尺寸（不绘制），以便先把 layer 分组框垫底。
  const pad = LAYER_PAD;
  const geoms = [];
  let cx = x + pad, boxH = 0, innerTop = y + LABEL_H + pad;
  for (const st of layer.stages) {
    const g = stageGeom(st, cx, innerTop);
    geoms.push(g);
    boxH = Math.max(boxH, g.h);
    cx = g.x + g.w + STAGE_GAP;
  }
  const w = cx - STAGE_GAP - x + pad;       // 左右各留 pad 内边距
  const h = LABEL_H + pad + boxH + pad;      // 标签 + 上下 pad + 框体
  const key = "layer:" + layer.layer;
  // 先画 layer 分组框（垫底），再让 stage 框/算子画在其上，避免被盖住。
  flowLayerContainer(key, x, y, w, h, { label: `Layer ${layer.layer}`, members: layer.ids, onClick: () => flowSelectBlock(key) });
  for (let i = 0; i < layer.stages.length; i++) drawStageBox(layer.stages[i], geoms[i].x, geoms[i].y, layer.layer);
  for (let i = 1; i < layer.stages.length; i++) drawHArrow(geoms[i - 1], geoms[i], innerTop + boxH / 2);
  return { key, x, y, w, h };
}

function drawSegBox(seg, x, y) {
  const key = "seg:" + seg.key;
  const ids = seg.ids || [];
  const agg = flowAggregate(ids, seg.key);
  const rows = Math.ceil(ids.length / CHIPS_PER_ROW);
  const chipH = rows ? rows * (CHIP_H + CHIP_GY) - CHIP_GY : 0;
  const h = SEG_HDR + chipH + BOX_PAD;
  const w = Math.max(SEG_W, CHIPS_PER_ROW * (CHIP_W + CHIP_GX) - CHIP_GX + BOX_PAD * 2);
  const dimTxt = seg.synthetic ? "" : (agg.dim ? `in×out ${agg.dim}` : "");
  flowContainer(key, x, y, w, h, {
    label: seg.name, sub: seg.synthetic ? (seg.caption || "sample") : agg.sub,
    dim: dimTxt, backend: agg.backend, members: ids, onClick: () => flowSelectBlock(key),
  });
  if (!seg.synthetic) {
    let cx = x + BOX_PAD, cy = y + SEG_HDR;
    ids.forEach((id, i) => {
      if (i > 0 && i % CHIPS_PER_ROW === 0) { cx = x + BOX_PAD; cy += CHIP_H + CHIP_GY; }
      flowOp(id, cx, cy, CHIP_W, CHIP_H);
      cx += CHIP_W + CHIP_GX;
    });
  }
  return { key, x, y, w, h };
}

function drawLayersSegment(seg, x, y) {
  if (!state.flowLayerExpanded) {
    const n = seg.count, w = SEG_W, h = 58;
    const agg = flowAggregate(seg.allIds);
    const key = "seg:layers";
    flowContainer(key, x, y, w, h, {
      label: `${seg.name}  × ${n}`, sub: agg.sub, dim: "click to expand",
      backend: agg.backend, members: seg.allIds || [],
      onClick: () => toggleFlowLayers(),
    });
    return { key, x, y, w, h };
  }
  const label = el("text", { class: "flow-layer-label", x: x + BOX_PAD, y: y + 4 }, gNodes);
  label.textContent = `${seg.name}  × ${seg.count}  (click to collapse)`;
  label.classList.add("flow-clickable");
  label.addEventListener("click", () => toggleFlowLayers());
  let ly = y + 18;
  let prev = null, maxW = SEG_W;
  for (const layer of seg.layers) {
    const row = drawLayerRow(layer, x, ly);
    if (prev) drawVArrow(prev, row);
    prev = row;
    ly = row.y + row.h + LAYER_GAP;
    maxW = Math.max(maxW, row.w);
  }
  return { key: "seg:layers", x, y, w: maxW, h: ly - y - LAYER_GAP, members: seg.allIds || [] };
}

function toggleFlowLayers() {
  state.flowLayerExpanded = !state.flowLayerExpanded;
  renderFlow();
  restoreExec();
  fit();
}

function setFlowBounds() { /* bounds are stored on flowBoxEls; used by fit() */ }

function renderFlow() {
  // 没有加载任何图（空态）：清空画布，什么都不画，让 #empty-hint 干净地显示。
  if (!state.nodes.length) {
    gNodes.textContent = ""; gEdges.textContent = "";
    nodeEls = []; edgeEls = [];
    state.flowBoxEls = new Map();
    return;
  }
  if (!state.flow) buildFlow();
  const f = state.flow;
  gNodes.textContent = ""; gEdges.textContent = "";
  nodeEls = []; edgeEls = [];
  state.flowBoxEls = new Map();
  const X0 = 96;
  let y = 46;
  let prev = null;
  for (const seg of f.segments) {
    const box = seg.type === "layers" ? drawLayersSegment(seg, X0, y) : drawSegBox(seg, X0, y);
    if (prev) drawVArrow(prev, box);
    prev = box;
    y = Math.max(y, box.y + box.h) + SEG_GAP;
  }
}

/* 选一个 flow 框 → 面板显示聚合（成员 op 列表 + 体量）。画布点选 = 全新浏览。 */
function flowSelectBlock(key) {
  if (state.selectedBlock !== key) state.panelNav = [];
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("selected");
  const b = state.flowBoxEls.get(key);
  if (b) b.g.classList.add("selected");
  state.selected = null;
  state.selectedBlock = key;
  renderFlowPanel(key);
}

function renderFlowPanel(key) {
  const b = state.flowBoxEls.get(key);
  const body = document.getElementById("panel-body");
  if (!b) return;
  const members = b.members || [];
  let h = `<div class="node-header">${backBtnHtml()}<h3>${htmlEscape(b.label)}</h3>`;
  h += `<span class="badge flow">flow · ${members.length} ops</span></div>`;
  if (members.length) {
    const agg = flowAggregate(members);
    h += `<table class="kv-table">`;
    h += row("Backend", BACKEND_NAME[agg.backend]);
    h += row("Ops", members.length);
    h += row("Op params", htmlEscape(members.map(id => state.byId.get(id).op).join(" · ")));
    h += `</table>`;
    h += `<div class="section-title">Contained ops</div><div class="link-list">`;
    h += members.map(id => {
      const n = state.byId.get(id);
      return `<span class="link-node" data-id="${id}">#${id} ${htmlEscape(n.name)}</span>`;
    }).join(" ");
    h += `</div>`;
    h += `<div class="section-title">How this stage works</div>`;
    h += `<div class="explain">${OP_EXPLAIN[members[0] ? state.byId.get(members[0]).op : ""] || "(aggregate of ops above)"}</div>`;
  } else {
    h += `<div class="explain">Synthetic step box — the sampler picks the next token from the logits distribution.</div>`;
    if (state.trace) {
      const phase = state.phases[state.phaseIdx];
      const step = phase && phase.steps ? phase.steps[state.stepIdx] : null;
      if (step) h += stepSummaryHtml(step);
    }
  }
  body.innerHTML = h;
  body.querySelectorAll(".link-node").forEach(s => s.addEventListener("click", () => navigateNode(parseInt(s.dataset.id, 10))));
  wireBackBtn(body);
}

/* 点亮某算子所属的阶段框 + 层框（执行态）。 */
function flowLight(id) {
  if (state.viewMode !== "flow") return;
  const f = state.flow; if (!f) return;
  for (const k of [f.blockKeyOf.get(id), f.layerKeyOf.get(id)]) {
    if (!k) continue;
    const b = state.flowBoxEls.get(k);
    if (b) b.g.classList.add("executed");
  }
}
/* 按该算子数据的 abs-mean 给所属框染色。 */
function flowTint(id) {
  if (state.viewMode !== "flow") return;
  const f = state.flow; if (!f) return;
  const d = state.stepData && state.stepData.get(id);
  if (!d || !d.stats) return;
  const a = Math.abs(d.stats.absmean);
  const t = Math.min(Math.max((Math.log10(a + 1e-9) + 6) / 8, 0), 1);
  const hue = 240 - t * 240;
  // 浅填充：固定 lightness/saturation，只让 hue 表现量级（蓝→红），
  // 这样配深色文字在任何色相下都有强对比。
  const fill = `hsl(${hue}, 55%, 62%)`;
  // 只染阶段框（携带内容与颜色）；layer 分组框保持半透明细边框不整块染色。
  const k = f.blockKeyOf.get(id);
  const b = k ? state.flowBoxEls.get(k) : null;
  if (b) { b.rect.style.fill = fill; b.g.classList.add("executed"); }
}

/* 在 flow 渲染后恢复当前执行态（直播用 live 数据，否则从 cursor 重放）。 */
function restoreExec() {
  if (state.live) { applyLivePhaseState(); return; }
  if (state.cursor >= 0 && state.order.length) syncExecutionState();
}

function setViewMode(m) {
  if (state.viewMode === m) return;
  state.viewMode = m;
  const a = document.getElementById("view-dataflow"), b = document.getElementById("view-flow");
  a.classList.toggle("active", m === "graph");
  b.classList.toggle("active", m === "flow");
  if (m === "flow") { if (!state.flow) buildFlow(); }
  renderGraph();
  restoreExec();
  applyFilters();
  fit();
  updateStepInfo();
  buildLegend();
}

/* ---------------- 动画 ---------------- */

function setPlaying(on) {
  state.playing = on;
  clearInterval(state.timer);
  const btn = document.getElementById("btn-play");
  if (on) {
    btn.textContent = "⏸";
    state.timer = setInterval(() => {
      if (state.cursor >= state.order.length - 1) {
        // 当前步播完：trace 模式下自动进入下一步
        if (state.trace) {
          const phase = state.phases[state.phaseIdx];
          if (state.stepIdx < (phase.steps || []).length - 1) {
            loadStep(state.stepIdx + 1);
            return;
          }
        }
        setPlaying(false);
        return;
      }
      stepForward();
    }, state.speed);
  } else {
    btn.textContent = "▶";
  }
}

function stepForward() {
  if (state.cursor >= state.order.length - 1) return;
  state.cursor++;
  const id = state.order[state.cursor];
  // 已执行样式（flow 折叠态可能未绘制该 op，nodeEls[id] 可能缺失 → 仅点亮所属框）
  const g = nodeEls[id];
  if (g) { g.classList.remove("ready", "current"); g.classList.add("executed"); }
  applyTint(id);
  flowLight(id);
  // 入边点亮
  const node = state.byId.get(id);
  for (const s of node.src) {
    const key = s + ">" + id;
    const e = state.edgeById.get(key);
    if (e && edgeEls[e.id]) edgeEls[e.id].classList.add("executed");
  }
  // 更新 ready 集
  updateReady(id);
  updateStepInfo();
}

function stepBack() {
  if (state.cursor < 0) {
    // 回退到上一步的末尾
    if (state.trace && state.stepIdx > 0) {
      loadStep(state.stepIdx - 1);
      state.cursor = state.order.length - 1;
      syncExecutionState();
      updateStepInfo();
    }
    return;
  }
  state.cursor--;
  syncExecutionState();
  updateStepInfo();
}

function updateReady(executedId) {
  for (const c of state.consumers[executedId]) {
    if (!nodeEls[c] || nodeEls[c].classList.contains("executed")) continue;
    let all = true;
    for (const s of state.byId.get(c).src) {
      if (!nodeEls[s] || !nodeEls[s].classList.contains("executed")) { all = false; break; }
    }
    if (all && nodeEls[c]) nodeEls[c].classList.add("ready");
  }
}

function updateStepInfo() {
  const total = state.order.length;
  const info = document.getElementById("step-info");
  if (state.cursor < 0) { info.textContent = "Ready (click ▶ to start)"; return; }
  const n = state.byId.get(state.order[state.cursor]);
  const pct = ((state.cursor + 1) / total * 100).toFixed(1);
  info.textContent = `Step ${state.cursor + 1}/${total} (${pct}%) · ${n.name} (${n.op})`;
}

/* ---------------- 选中 / 面板 ---------------- */

function selectNode(id) {
  // 画布上直接点选任意节点 = 一次全新浏览，清空导航栈（无 back）。
  state.panelNav = [];
  state.selected = id;
  state.selectedBlock = null;
  for (const g of nodeEls) if (g) g.classList.remove("selected");
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("selected");
  nodeEls[id].classList.add("selected");
  try {
    renderPanel(id);
  } catch (e) {
    console.error("renderPanel:", e);
  }
}

// 面板内从"分组"钻到"单节点"（Contained ops / upstream-downstream 链接）：
// 把当前视图压栈，这样左上角的 Back 能回到之前的面板。
function navigateNode(id) {
  const cur = currentView();
  if (cur) state.panelNav.push(cur);
  state.selected = id;
  state.selectedBlock = null;
  for (const g of nodeEls) if (g) g.classList.remove("selected");
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("selected");
  if (nodeEls[id]) nodeEls[id].classList.add("selected");
  renderPanel(id);
}

function navigateBlock(key) {
  const cur = currentView();
  if (cur) state.panelNav.push(cur);
  state.selected = null;
  state.selectedBlock = key;
  for (const g of nodeEls) if (g) g.classList.remove("selected");
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("selected");
  const b = state.flowBoxEls.get(key);
  if (b) b.g.classList.add("selected");
  renderFlowPanel(key);
}

function currentView() {
  if (state.selectedBlock) return { kind: "block", key: state.selectedBlock };
  if (state.selected !== null) return { kind: "node", id: state.selected };
  return null;
}

// Back：弹出导航栈并渲染上一个视图。
function goBack() {
  const prev = state.panelNav.pop();
  if (!prev) return;
  for (const g of nodeEls) if (g) g.classList.remove("selected");
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("selected");
  if (prev.kind === "block") {
    state.selected = null; state.selectedBlock = prev.key;
    const b = state.flowBoxEls.get(prev.key);
    if (b) b.g.classList.add("selected");
    renderFlowPanel(prev.key);
  } else {
    state.selected = prev.id; state.selectedBlock = null;
    if (nodeEls[prev.id]) nodeEls[prev.id].classList.add("selected");
    renderPanel(prev.id);
  }
}

function backBtnHtml() {
  return state.panelNav.length
    ? '<button class="panel-back" title="Back to the previous panel">← Back</button>'
    : "";
}
function wireBackBtn(body) {
  const b = body.querySelector(".panel-back");
  if (b) b.addEventListener("click", () => goBack());
}

function deselect() {
  state.selected = null;
  state.selectedBlock = null;
  state.panelNav = [];
  for (const g of nodeEls) if (g) g.classList.remove("selected");
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("selected");
  document.getElementById("panel-body").innerHTML =
    '<div class="placeholder">Click any node to view that step\'s data</div>';
}

function renderPanel(id) {
  const n = state.byId.get(id);
  const body = document.getElementById("panel-body");
  const isInput = n.op === "input", isOutput = state.doc.outputs.includes(id);
  const fused = n.op.startsWith("fused") || n.op === "swiglu";

  let h = "";
  // 当前步的 token / logits 概要：trace 模式取预置步，直播模式取最近的事件
  let step = null;
  if (state.trace) {
    const phase = state.phases[state.phaseIdx];
    if (phase && phase.steps && phase.steps[state.stepIdx]) {
      step = phase.steps[state.stepIdx];
    } else if (state.live && phase) {
      const liveSteps = liveState.steps[phase.kind] || [];
      step = liveSteps.length ? liveSteps[liveSteps.length - 1] : null;
    }
  }

  h += `<div class="node-header">${backBtnHtml()}<h3>${htmlEscape(n.name)}</h3>`;
  h += `<span class="badge op">${htmlEscape(n.op)}</span>`;
  h += `<span class="badge ${n.backend || "none"}">${BACKEND_NAME[n.backend || "none"]}</span>`;
  if (fused) h += `<span class="badge fused">fused</span>`;
  if (isInput || isOutput) h += `<span class="badge inout">${isInput ? "In" : "Out"}</span>`;
  h += `</div>`;

  h += `<table class="kv-table">`;
  h += row("Tensor shape", shapeStr(n.shape));
  h += row("Elements", n.shape.reduce((a, b) => a * b, 1).toLocaleString());
  h += row("Type", n.dtype + " (activation)");
  h += row("Exec order", `Step ${state.order.indexOf(id) + 1}/${state.order.length}`);
  const d = n.detail || {};
  if (Object.keys(d).length) h += row("Op params", htmlEscape(JSON.stringify(d)));
  const m = n.meta || {};
  const metaRows = [];
  if (m.weight) metaRows.push(["Weight", m.weight + (m.wtype ? " · " + m.wtype : "")]);
  if (m.bias) metaRows.push(["Bias", m.bias]);
  if (m.in_dim !== undefined) metaRows.push(["in×out", `${m.in_dim}×${m.out_dim !== undefined ? m.out_dim : (m.nf !== undefined ? "nf=" + m.nf : "?")}`]);
  if (m.n_head !== undefined) metaRows.push(["Attention", `h=${m.n_head} kv=${m.n_head_kv !== undefined ? m.n_head_kv : "?"} hd=${m.hd}`]);
  if (m.nkt !== undefined) metaRows.push(["KV stride", m.nkt]);
  if (m.scale !== undefined) metaRows.push(["scale", m.scale]);
  if (m.eps !== undefined) metaRows.push(["eps", m.eps]);
  if (m.layer !== undefined) metaRows.push(["Layer", m.layer]);
  if (m.vocab_size !== undefined) metaRows.push(["Vocab", m.vocab_size]);
  for (const [k, v] of metaRows) h += row(k, htmlEscape(String(v)));

  // 输入/输出连接
  const srcs = n.src.map(s => `<span class="link-node" data-id="${s}">#${s} ${htmlEscape(state.byId.get(s).name)}</span>`).join(", ");
  const cons = (state.consumers[id] || []).map(c => `<span class="link-node" data-id="${c}">#${c} ${htmlEscape(state.byId.get(c).name)}</span>`).join(", ");
  h += row("Upstream", srcs || "—");
  h += row("Downstream", cons || "—");
  h += `</table>`;

  h += `<div class="section-title">What this step does</div>`;
  h += `<div class="explain">${OP_EXPLAIN[n.op] || "(no description yet)"}</div>`;

  h += renderExtraData(n);
  // Step overview (input token + logits top-5) goes below the node details so
  // clicking a node shows its details first instead of hiding them under the
  // (scrolling) step summary.
  if (step) h += stepSummaryHtml(step);
  body.innerHTML = h;
  body.querySelectorAll(".link-node").forEach(s => {
    s.addEventListener("click", () => navigateNode(parseInt(s.dataset.id, 10)));
  });
  wireBackBtn(body);
}

function row(k, v) { return `<tr><th>${k}</th><td>${v || "—"}</td></tr>`; }

/* 真实数据渲染：优先取当前步的 trace 数据，其次节点自带 stats/values */
function renderExtraData(n) {
  const d = (state.stepData && state.stepData.get(n.id)) || null;
  const stats = d ? d.stats : (n.stats || null);
  const values = d ? d.values : (Array.isArray(n.values) ? n.values : null);
  const stride = d ? d.stride : 1;
  const nTotal = d ? d.n : (n.shape || []).reduce((a, b) => a * b, 1);
  let h = "";
  if (stats) {
    h += `<div class="section-title">Tensor stats${d ? " (this step)" : ""}</div>`;
    h += `<table class="kv-table">`;
    h += row("min", stats.min !== undefined ? stats.min.toPrecision(4) : "—");
    h += row("max", stats.max !== undefined ? stats.max.toPrecision(4) : "—");
    h += row("mean", stats.mean !== undefined ? stats.mean.toPrecision(4) : "—");
    h += row("abs mean", stats.absmean !== undefined ? stats.absmean.toPrecision(4) : "—");
    if (d) {
      h += row("Sampled", `${values.length} / ${nTotal} (stride ${stride})`);
      if (d.dtype === "i32") h += row("Type", "i32 (token ids, etc.)");
    }
    h += `</table>`;
  } else if (state.trace) {
    h += `<div class="explain" style="color:var(--fg-faint)">No data for this node in this step (skipped at runtime / no output buffer)</div>`;
  }
  if (Array.isArray(values) && values.length) {
    h += `<div class="section-title">Value heatmap (downsampled)</div>`;
    h += `<div class="hm-wrap" id="hm"></div>`;
    h += `<div class="hm-caption">${values.length.toLocaleString()} samples · stride ${stride} · total ${nTotal.toLocaleString()}</div>`;
    setTimeout(() => renderHeatmap("hm", values, n.shape, stride), 0);
  }
  return h;
}

function renderHeatmap(containerId, values, shape, stride) {
  const wrap = document.getElementById(containerId);
  if (!wrap) return;
  // The samples are a flat, strided downsample of the whole tensor, so they
  // carry no reliable 2D layout (the server's per-phase `shape[1]` is a
  // canonical nt, not the real token count of this run). Layout them on a
  // near-square grid so a small nt (1 or 2) can't collapse the heatmap into a
  // thin strip — one cell per sample, roughly square.
  const cols = Math.max(2, Math.min(16, Math.ceil(Math.sqrt(values.length))));
  const rows = Math.ceil(values.length / cols);
  const CELL = Math.max(6, Math.min(12, Math.floor(200 / cols)));
  let min = Infinity, max = -Infinity;
  for (const v of values) { if (v < min) min = v; if (v > max) max = v; }
  const span = (max - min) || 1;
  const W = cols * CELL, H = rows * CELL;
  let cells = "";
  for (let i = 0; i < values.length; i++) {
    const t = (values[i] - min) / span;
    const hue = 240 - t * 240; // 蓝 → 红
    const x = (i % cols) * CELL, y = Math.floor(i / cols) * CELL;
    cells += `<rect x="${x}" y="${y}" width="${CELL}" height="${CELL}" fill="hsl(${hue},75%,55%)"/>`;
  }
  wrap.innerHTML = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" xmlns="${SVG_NS}" preserveAspectRatio="none">${cells}</svg>`;
}

/* ---------------- 过滤 ---------------- */

function applyFilters() {
  for (const n of state.nodes) {
    const g = nodeEls[n.id];
    if (!g) continue;
    const row = state.rows[state.rowOf.get(n.id)];
    let hide = row.collapsed;
    if (!hide && !state.filters.io && nodeIsInputOutput(n)) hide = true;
    if (!hide && state.filters.attn && !nodeOnAttnPath(n)) hide = true;
    if (!hide && state.filters.ffn && !nodeOnFfnPath(n)) hide = true;
    g.classList.toggle("hidden", hide);
  }
  for (const e of state.edges) {
    const p = edgeEls[e.id];
    if (!p) continue; // flow 视图不绘制逐算子边
    const srcN = state.byId.get(e.src), dstN = state.byId.get(e.dst);
    let hide = false;
    if (!state.filters.kv && p.classList.contains("kv")) hide = true;
    if (nodeEls[e.src].classList.contains("hidden") || nodeEls[e.dst].classList.contains("hidden")) hide = true;
    p.classList.toggle("hidden", hide);
  }
}

/* ---------------- 缩放 / 平移 ---------------- */

function setView() {
  viewport.setAttribute("transform",
    `translate(${state.view.x} ${state.view.y}) scale(${state.view.k})`);
}
function fit() {
  if (state.viewMode === "flow") { fitFlow(); return; }
  const wrap = document.getElementById("canvas-wrap");
  const W = wrap.clientWidth, H = wrap.clientHeight;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, any = false;
  for (const r of state.rows) {
    if (r.collapsed) continue;
    for (const id of r.nodes) {
      if (nodeEls[id].classList.contains("hidden")) continue;
      const n = state.byId.get(id);
      minX = Math.min(minX, n._x); maxX = Math.max(maxX, n._x + n._w);
      minY = Math.min(minY, r.y); maxY = Math.max(maxY, r.y + NODE_H);
      any = true;
    }
  }
  if (!any) { state.view = { x: 30, y: 30, k: 1 }; setView(); return; }
  // Guard against a degenerate (very short) viewport: keep the usable window
  // positive so k can never go negative (a negative scale mirrors the graph
  // into invisibility — the root cause of the "blank graph / nodes don't
  // respond" symptom). The min keeps a tiny layout from collapsing to a spec.
  const pad = 60;
  const winW = Math.max(W - pad * 2, 40);
  const winH = Math.max(H - pad * 2, 40);
  const k = Math.min(winW / (maxX - minX + 40), winH / (maxY - minY + 40), 1.6);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  state.view = { x: W / 2 - cx * k, y: H / 2 - cy * k, k };
  setView();
}

function fitFlow() {
  const wrap = document.getElementById("canvas-wrap");
  const W = wrap.clientWidth, H = wrap.clientHeight;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, any = false;
  for (const [, b] of state.flowBoxEls) {
    if (b.g.classList.contains("hidden")) continue;
    minX = Math.min(minX, b.x); minY = Math.min(minY, b.y);
    maxX = Math.max(maxX, b.x + b.w); maxY = Math.max(maxY, b.y + b.h);
    any = true;
  }
  if (!any) { state.view = { x: 30, y: 30, k: 1 }; setView(); return; }
  const pad = 60;
  const winW = Math.max(W - pad * 2, 40);
  const winH = Math.max(H - pad * 2, 40);
  const k = Math.min(winW / (maxX - minX + 40), winH / (maxY - minY + 40), 1.6);
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  state.view = { x: W / 2 - cx * k, y: H / 2 - cy * k, k };
  setView();
}

function initPanZoom() {
  let dragging = false, sx = 0, sy = 0, vx = 0, vy = 0;
  canvas.addEventListener("pointerdown", e => {
    // Let node/box clicks through: setPointerCapture here would retarget the pointer
    // (and the resulting click) to the canvas, so the node's click handler never
    // fires and selecting a node by a real mouse click silently fails. The grid
    // uses .node-group; the Pipeline view uses .flow-box / .flow-op / .flow-clickable.
    if (e.target.closest(".node-group, .flow-box, .flow-op, .flow-layer, .flow-clickable")) return;
    dragging = true; sx = e.clientX; sy = e.clientY; vx = state.view.x; vy = state.view.y;
    canvas.classList.add("panning");
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", e => {
    if (!dragging) return;
    state.view.x = vx + (e.clientX - sx);
    state.view.y = vy + (e.clientY - sy);
    setView();
  });
  canvas.addEventListener("pointerup", () => { dragging = false; canvas.classList.remove("panning"); });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const mx = e.clientX - rect.left, my = e.clientY - rect.top;
    const factor = Math.exp(-e.deltaY * 0.0015);
    const k = Math.min(Math.max(state.view.k * factor, 0.05), 8);
    // 以鼠标为中心缩放
    state.view.x = mx - (mx - state.view.x) * (k / state.view.k);
    state.view.y = my - (my - state.view.y) * (k / state.view.k);
    state.view.k = k;
    setView();
  }, { passive: false });
}

/* ---------------- 加载 ---------------- */

async function loadDoc(doc) {
  stopPlayback();
  if (doc.format === "minfer-trace") {
    // Trace mode: phases (prefill/decode) each carry a graph + per-step data.
    state.trace = doc;
    state.phases = doc.phases || [];
    if (!state.phases.length) { alert("trace has no phases"); return; }
    await loadPhase(0);
    return;
  }
  // Plain graph mode.
  state.trace = null;
  state.phases = [];
  state.stepData = null;
  document.getElementById("tracebar").hidden = true;
  state.doc = doc;
  const prep = prepare(doc);
  Object.assign(state, prep);
  state.cursor = -1;
  state.collapsed = new Set();
  state.selected = null;
  state.flow = null;
  state.flowLayerExpanded = false;
  state.selectedBlock = null;

  document.getElementById("doc-title").textContent =
    `${doc.model} · ${doc.kind} · ${doc.nodes.length} nodes`;
  document.getElementById("status-model").textContent = `Model: ${doc.model}`;
  document.getElementById("status-counts").textContent =
    `${state.nodes.length} nodes · ${state.edges.length} edges · ${state.rows.filter(r => r.name.startsWith("blk.")).length} layers`;
  document.getElementById("empty-hint").style.display = "none";

  renderGraph();
  applyFilters();
  fit();
  updateStepInfo();
  updatePlaybackEnabled();
  deselect();
}

/* ---------------- P2 trace 模式 ---------------- */

function loadPhase(pi) {
  stopPlayback();
  state.phaseIdx = pi;
  const phase = state.phases[pi];
  state.doc = phase.graph;
  const prep = prepare(phase.graph);
  Object.assign(state, prep);
  state.cursor = -1;
  state.collapsed = new Set();
  state.selected = null;
  state.stepIdx = 0;
  state.flow = null;
  state.flowLayerExpanded = false;
  state.selectedBlock = null;

  document.getElementById("tracebar").hidden = false;
  renderTracebar();
  document.getElementById("doc-title").textContent =
    `${phase.graph.model} · ${phase.kind} · ${phase.graph.nodes.length} nodes · ${(phase.steps || []).length} steps`;
  document.getElementById("status-model").textContent = `Model: ${phase.graph.model}`;
  document.getElementById("status-counts").textContent =
    `${state.nodes.length} nodes · ${state.edges.length} edges · data: ${(phase.steps || []).length} steps`;
  document.getElementById("empty-hint").style.display = "none";

  renderGraph();
  applyFilters();
  fit();
  loadStep(0);
  if (state.live) applyLivePhaseState();
  updatePlaybackEnabled();
}

function loadStep(si) {
  const phase = state.phases[state.phaseIdx];
  const steps = phase.steps || [];
  si = Math.max(0, Math.min(si, Math.max(steps.length - 1, 0)));
  state.stepIdx = si;
  const step = steps[si];
  if (!state.live) {
    state.stepData = new Map();
    if (step) for (const nd of step.nodes) state.stepData.set(nd.id, nd);
  }

  // 重置本步播放状态
  state.cursor = -1;
  if (!state.live && nodeEls.length) {
    clearTints();
    for (const n of state.nodes) if (nodeEls[n.id]) nodeEls[n.id].classList.remove("executed", "ready", "current");
    for (const e of state.edges) if (edgeEls[e.id]) edgeEls[e.id].classList.remove("executed", "current");
  }
  const slider = document.getElementById("step-slider");
  slider.max = Math.max(0, steps.length - 1);
  slider.value = si;
  document.getElementById("step-label").textContent =
    `${si + 1}/${Math.max(steps.length, 1)}`;
  renderTokenStrip();
  updateStepInfo();
  deselect();
  if (step) renderStepPanel(step);
  else if (state.live) {
    // Live mode has no preset steps; restore the latest live step summary so a
    // phase switch (or auto prefill→decode transition) doesn't leave the panel
    // blank.
    const liveSteps = liveState.steps[phase.kind] || [];
    if (liveSteps.length) renderStepPanel(liveSteps[liveSteps.length - 1]);
  }
}

function renderTracebar() {
  const tabs = document.getElementById("phase-tabs");
  tabs.innerHTML = "";
  state.phases.forEach((p, i) => {
    const b = document.createElement("button");
    b.className = "phase-tab" + (i === state.phaseIdx ? " active" : "");
    b.textContent = state.live
      ? `${p.kind} · Live`
      : `${p.kind} · ${(p.steps || []).length} steps`;
    b.title = `Switch phase (${p.graph.nodes.length} nodes)`;
    b.addEventListener("click", () => {
      // Re-loading the current phase would `deselect()` and wipe the panel
      // (and, in live mode, leave it blank) — ignore clicks on the active tab.
      if (i !== state.phaseIdx) loadPhase(i);
    });
    tabs.appendChild(b);
  });
  const slider = document.getElementById("step-slider");
  slider.max = Math.max(0, (state.phases[state.phaseIdx].steps || []).length - 1);
  // 直播模式没有预置步，隐藏步进滑杆
  document.querySelector(".step-ctl").style.display = state.live ? "none" : "";
}

function renderTokenStrip() {
  const strip = document.getElementById("token-strip");
  const phase = state.phases[state.phaseIdx];
  if (!state.trace) { strip.textContent = ""; return; }
  strip.innerHTML = "";
  if (state.live) {
    // 直播模式：token 条 = prompt + 本回合已到达的 decode token
    const prompt = document.createElement("span");
    prompt.className = "tok-chip tok-prompt";
    prompt.textContent = `prompt: ${state.trace.prompt || "—"}`;
    strip.appendChild(prompt);
    if (phase && phase.kind === "decode") {
      for (const s of liveState.steps.decode || []) {
        const chip = document.createElement("span");
        chip.className = "tok-chip active";
        chip.textContent = (s.text || "∅");
        chip.title = `token #${s.token}`;
        strip.appendChild(chip);
      }
    }
    return;
  }
  if (phase.kind !== "decode") {
    const span = document.createElement("span");
    span.className = "tok-chip tok-prompt";
    span.textContent = `prompt: ${state.trace.prompt || ""}`;
    strip.appendChild(span);
    return;
  }
  const prompt = document.createElement("span");
  prompt.className = "tok-chip tok-prompt";
  prompt.textContent = `prompt: ${state.trace.prompt || ""}`;
  strip.appendChild(prompt);
  phase.steps.forEach((s, i) => {
    const chip = document.createElement("span");
    chip.className = "tok-chip" + (i === state.stepIdx ? " active" : "");
    chip.textContent = (s.text || "∅");
    chip.title = `token #${s.token} · step ${i + 1}`;
    chip.addEventListener("click", () => { stopPlayback(); loadStep(i); });
    strip.appendChild(chip);
  });
}

function stepSummaryHtml(step) {
  let h = "";
  if (step.token !== undefined && step.token !== null) {
    h += `<div class="step-token">Input token <span class="tok-id">#${step.token}</span> <span class="tok-text">${htmlEscape(step.text)}</span></div>`;
  }
  if (step.logits_top && step.logits_top.length) {
    h += `<div class="section-title">This step output logits top-5</div>`;
    h += `<div class="logits">`;
    const maxw = step.logits_top[0] ? step.logits_top[0][1] : 1;
    for (const [id, p] of step.logits_top) {
      const rel = maxw > 0 ? (p / maxw) * 100 : 0;
      h += `<div class="logit-row"><span class="logit-token">#${id}</span>` +
        `<span class="logit-bar-wrap"><span class="logit-bar" style="width:${rel.toFixed(1)}%"></span></span>` +
        `<span class="logit-p">${(p * 100).toFixed(1)}%</span></div>`;
    }
    h += `</div>`;
  }
  return h;
}

function renderStepPanel(step) {
  const body = document.getElementById("panel-body");
  body.innerHTML = stepSummaryHtml(step) ||
    '<div class="placeholder">No data for this step</div>';
}

/* 执行状态全量重算（回退/切步时用） */
function syncExecutionState() {
  if (!nodeEls.length) return;
  clearTints();
  for (const n of state.nodes) if (nodeEls[n.id]) nodeEls[n.id].classList.remove("executed", "ready", "current");
  for (const e of state.edges) if (edgeEls[e.id]) edgeEls[e.id].classList.remove("executed", "current");
  const rem = state.nodes.map(n => n.src.length);
  for (let i = 0; i <= state.cursor; i++) {
    const id = state.order[i];
    if (nodeEls[id]) nodeEls[id].classList.add("executed");
    applyTint(id);
    flowLight(id);
    const node = state.byId.get(id);
    for (const s of node.src) {
      const e = state.edgeById.get(s + ">" + id);
      if (e && edgeEls[e.id]) edgeEls[e.id].classList.add("executed");
    }
    for (const c of state.consumers[id]) {
      rem[c]--;
      if (rem[c] === 0 && c > state.cursor && nodeEls[c]) nodeEls[c].classList.add("ready");
    }
  }
}

/* 按本步数据的 abs-mean 给已执行节点染色（log 尺度） */
function applyTint(id) {
  const g = nodeEls[id];
  if (!g) return;
  const rect = g.querySelector("rect");
  const d = state.stepData && state.stepData.get(id);
  if (d && d.stats && d.stats.absmean !== undefined) {
    const a = Math.abs(d.stats.absmean);
    const t = Math.min(Math.max((Math.log10(a + 1e-9) + 6) / 8, 0), 1); // -6..2 → 0..1
    const hue = 240 - t * 240; // 蓝 → 红
    rect.style.fill = `hsl(${hue}, 70%, 45%)`;
  } else {
    rect.style.fill = "";
  }
  flowTint(id);
}

function clearTints() {
  for (const n of state.nodes) {
    const rect = nodeEls[n.id] && nodeEls[n.id].querySelector("rect");
    if (rect) rect.style.fill = "";
  }
  for (const [, b] of state.flowBoxEls) {
    if (b.rect) b.rect.style.fill = "";
    b.g.classList.remove("executed", "current");
  }
}

// After a graph render, re-apply the current phase's live execution state from
// the per-phase store so switching phase tabs keeps the coloring from the run.
function applyLivePhaseState() {
  if (!state.live) return;
  const phase = state.phases[state.phaseIdx];
  if (!phase) return;
  const kind = phase.kind;
  const data = liveState.stepData[kind];
  if (!data) return;
  state.stepData = data;
  const exec = liveState.executed[kind];
  for (const n of state.nodes) {
    const g = nodeEls[n.id];
    if (!g) continue;
    const had = data.has(n.id) || (exec && exec.has(n.id));
    if (!had) continue;
    if (g.classList.contains("hidden")) continue;
    g.classList.add("executed");
    applyTint(n.id);
    flowLight(n.id);
  }
}

/* ---------------- P3 直播（SSE） ---------------- */

const liveState = {
  connected: false, es: null, base: "http://127.0.0.1:8080",
  steps: { prefill: [], decode: [] },
  // Per-phase execution state so switching phase tabs after a run preserves the
  // node coloring (executed classes + data tint) instead of wiping it.
  stepData: { prefill: null, decode: null },  // Map nodeId -> {stats, values, stride, n, dtype}
  executed: { prefill: null, decode: null },  // Set of executed node ids
  curKind: null,                              // phase currently being executed
};

function setLiveStatus(msg, dot) {
  document.getElementById("live-status").textContent = msg;
  const d = document.getElementById("live-dot");
  d.className = "live-dot" + (dot ? " " + dot : "");
}

// Reset the per-run live capture state (steps + per-phase coloring).
function resetLiveRun() {
  liveState.steps = { prefill: [], decode: [] };
  liveState.stepData = { prefill: new Map(), decode: new Map() };
  liveState.executed = { prefill: new Set(), decode: new Set() };
  liveState.curKind = null;
}

// Two groups of view-changing controls with different enable rules:
//  - playback controls require a loaded graph; in live mode they are disabled
//    because the view is driven by the SSE stream.
//  - load controls (sample dropdown + Open File) must stay usable whenever
//    there is no graph (that's how you load one) and are disabled only while
//    connected to live, where loading a static graph would clobber the view.
const PLAYBACK_CTRL_IDS = ["btn-play", "btn-step-back", "btn-step-fwd", "speed", "step-slider"];
const LOAD_CTRL_IDS = ["model-select", "btn-open"];

function setControlsDisabled(ids, on) {
  for (const id of ids) document.getElementById(id).disabled = on;
}

// Enable playback controls only when there is a graph to play AND we are not
// in live mode; disable the load controls only while live. Call whenever the
// "has a graph" / "live" state changes.
function updatePlaybackEnabled() {
  setControlsDisabled(PLAYBACK_CTRL_IDS, state.live || !state.nodes?.length);
  setControlsDisabled(LOAD_CTRL_IDS, state.live);
}

async function connectLive() {
  const base = document.getElementById("live-url").value.trim().replace(/\/+$/, "");
  if (!base) return;
  liveState.base = base;
  try {
    const res = await fetch(base + "/viz/graph");
    if (!res.ok) throw new Error("HTTP " + res.status);
    const doc = await res.json();
    state.live = true;
    state.trace = doc;
    state.phases = doc.phases || [];
    resetLiveRun();
    await loadPhase(0);

    if (liveState.es) liveState.es.close();
    liveState.es = new EventSource(base + "/viz/events");
    liveState.es.onmessage = ev => { try { handleLiveEvent(JSON.parse(ev.data)); } catch (e) { console.error("live event:", e); } };
    liveState.es.onerror = () => { if (liveState.connected) setLiveStatus("Event stream disconnected"); };
    liveState.connected = true;
    updatePlaybackEnabled();
    document.getElementById("btn-live").classList.add("live-on");
    document.getElementById("live-connect").textContent = "Disconnect";
    document.getElementById("live-run").disabled = false;
    setLiveStatus("Connected — click “Run” to start inference", "on");
  } catch (e) {
    setLiveStatus("Connection failed: " + e.message);
  }
}

function disconnectLive() {
  if (liveState.es) { liveState.es.close(); liveState.es = null; }
  liveState.connected = false;
  state.live = false;
  updatePlaybackEnabled();
  document.getElementById("btn-live").classList.remove("live-on");
  document.getElementById("live-connect").textContent = "Connect";
  document.getElementById("live-run").disabled = true;
  setLiveStatus("Not connected");
}

function handleLiveEvent(ev) {
  switch (ev.type) {
    case "hello":
      setLiveStatus(`Connected ${ev.model}, waiting for inference…`, "on");
      break;
    case "phase": {
      liveState.curKind = ev.kind;
      if (!liveState.stepData[ev.kind]) {
        liveState.stepData[ev.kind] = new Map();
        liveState.executed[ev.kind] = new Set();
      }
      const idx = state.phases.findIndex(p => p.kind === ev.kind);
      if (idx !== -1 && idx !== state.phaseIdx) loadPhase(idx);
      break;
    }
    case "node": {
      const rec = { stats: ev.stats, values: ev.values, stride: ev.stride, n: ev.n, dtype: ev.dtype };
      const kind = liveState.curKind;
      if (kind) {
        (liveState.stepData[kind] = liveState.stepData[kind] || new Map()).set(ev.id, rec);
        (liveState.executed[kind] = liveState.executed[kind] || new Set()).add(ev.id);
      }
      // If the node belongs to a phase the user isn't currently viewing, don't
      // tint the view — it is restored when that phase is reloaded.
      const viewKind = state.phases[state.phaseIdx] ? state.phases[state.phaseIdx].kind : null;
      if (kind !== null && viewKind !== null && kind !== viewKind) break;
      const g = nodeEls[ev.id];
      if (!g || g.classList.contains("hidden")) break;
      g.classList.add("executed");
      if (!state.stepData) state.stepData = new Map();
      state.stepData.set(ev.id, rec);
      applyTint(ev.id);
      flowLight(ev.id);
      break;
    }
    case "step": {
      (liveState.steps[ev.phase] = liveState.steps[ev.phase] || []).push(ev);
      renderTokenStrip();
      if (ev.logits_top && ev.logits_top.length) {
        // 面板展示本步 token + logits（无选中节点时）
        if (state.selected === null) renderStepPanel(ev);
      }
      break;
    }
    case "finish":
      setLiveStatus(`Done (${ev.reason} · ${ev.tokens} tokens): ${(ev.text || "").slice(0, 80)}`, "on");
      break;
    case "lag":
      setLiveStatus("Events too fast; some nodes not shown (increase max_tokens or speed up the server)", "on");
      break;
  }
}

async function runLive() {
  const prompt = document.getElementById("live-prompt").value;
  const maxTokens = parseInt(document.getElementById("live-max").value, 10) || 8;
  if (!prompt) return;
  // 重置本回合显示
  resetLiveRun();
  state.stepData = new Map();
  if (nodeEls.length) {
    clearTints();
    for (const n of state.nodes) if (nodeEls[n.id]) nodeEls[n.id].classList.remove("executed", "ready", "current");
    for (const e of state.edges) if (edgeEls[e.id]) edgeEls[e.id].classList.remove("executed", "current");
  }
  renderTokenStrip();
  deselect();
  setLiveStatus("Inference running… (nodes light up one by one)", "busy");
  document.getElementById("live-run").disabled = true;
  try {
    await fetch(liveState.base + "/viz/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_tokens: maxTokens,
      }),
    });
  } catch (e) {
    setLiveStatus("Run failed: " + e.message, "on");
  } finally {
    document.getElementById("live-run").disabled = false;
  }
}

function stopPlayback() { setPlaying(false); }

async function loadManifest() {
  try {
    const res = await fetch("samples/manifest.json");
    if (!res.ok) throw new Error("no manifest");
    const m = await res.json();
    const sel = document.getElementById("model-select");
    for (const s of m.samples) {
      const opt = document.createElement("option");
      opt.value = "samples/" + s.file;
      opt.textContent = s.label;
      sel.appendChild(opt);
    }
    sel.addEventListener("change", () => {
      const v = sel.value;
      if (v) fetch(v).then(r => r.json()).then(loadDoc).catch(err => console.error(err));
    });
  } catch (e) {
    // file:// 下无法 fetch：保留「打开文件」路径
    const sel = document.getElementById("model-select");
    const opt = document.createElement("option");
    opt.value = ""; opt.textContent = "(under file:// use “Open File…” above)";
    sel.appendChild(opt);
  }
}

/* ---------------- 初始化 ---------------- */

function init() {
  initPanZoom();

  document.getElementById("btn-play").addEventListener("click", () => {
    if (state.cursor >= state.order.length - 1) {
      // 播完：trace 模式从头再来，普通模式重播本图
      if (state.trace) {
        const phase = state.phases[state.phaseIdx];
        const atEnd = state.stepIdx >= (phase.steps || []).length - 1;
        if (atEnd) loadStep(0);
      } else {
        state.cursor = -1;
        resetPlayback();
      }
    }
    setPlaying(!state.playing);
  });
  document.getElementById("btn-step-fwd").addEventListener("click", () => { stopPlayback(); stepForward(); });
  document.getElementById("btn-step-back").addEventListener("click", () => { stopPlayback(); stepBack(); });
  document.getElementById("speed").addEventListener("input", e => {
    state.speed = parseInt(e.target.value, 10);
    document.getElementById("speed-label").textContent = state.speed + "ms";
    if (state.playing) {
      clearInterval(state.timer);
      state.timer = setInterval(() => {
        if (state.cursor >= state.order.length - 1) {
          if (state.trace) {
            const phase = state.phases[state.phaseIdx];
            if (state.stepIdx < (phase.steps || []).length - 1) {
              loadStep(state.stepIdx + 1);
              return;
            }
          }
          setPlaying(false);
          return;
        }
        stepForward();
      }, state.speed);
    }
  });
  document.getElementById("step-slider").addEventListener("input", e => {
    stopPlayback();
    loadStep(parseInt(e.target.value, 10));
  });
  document.getElementById("btn-fit").addEventListener("click", fit);
  document.getElementById("btn-open").addEventListener("click", () => document.getElementById("file-input").click());
  document.getElementById("file-input").addEventListener("change", e => {
    const f = e.target.files[0];
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => { try { loadDoc(JSON.parse(rd.result)); } catch (err) { alert("JSON parse failed: " + err.message); } };
    rd.readAsText(f);
    e.target.value = "";
  });
  for (const [id, key] of [["f-kv", "kv"], ["f-io", "io"], ["f-attn", "attn"], ["f-ffn", "ffn"]]) {
    document.getElementById(id).addEventListener("change", e => {
      state.filters[key] = e.target.checked;
      if (key === "attn" && e.target.checked) document.getElementById("f-ffn").checked = false;
      if (key === "ffn" && e.target.checked) document.getElementById("f-attn").checked = false;
      applyFilters(); fit();
    });
  }

  // 视图切换 Operators / Pipeline
  document.getElementById("view-dataflow").addEventListener("click", () => setViewMode("graph"));
  document.getElementById("view-flow").addEventListener("click", () => setViewMode("flow"));

  // 键盘
  window.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { e.preventDefault(); document.getElementById("btn-play").click(); }
    else if (e.key === "ArrowRight") { stopPlayback(); stepForward(); }
    else if (e.key === "ArrowLeft") { stopPlayback(); stepBack(); }
    else if (e.key === "Escape") deselect();
  });

  // 图例
  const legendPop = document.getElementById("legend-pop");
  document.getElementById("btn-legend").addEventListener("click", () => legendPop.classList.toggle("show"));
  document.getElementById("legend-close").addEventListener("click", () => legendPop.classList.remove("show"));
  buildLegend();

  // 直播（P3）
  const livePanel = document.getElementById("live-panel");
  document.getElementById("btn-live").addEventListener("click", () => {
    livePanel.hidden = !livePanel.hidden;
  });
  document.getElementById("live-close").addEventListener("click", () => { livePanel.hidden = true; });
  document.getElementById("live-connect").addEventListener("click", () => {
    if (liveState.connected) disconnectLive();
    else connectLive();
  });
  document.getElementById("live-run").addEventListener("click", runLive);
  document.getElementById("live-prompt").addEventListener("keydown", e => {
    if (e.key === "Enter") runLive();
  });

  // Live panel is draggable, clamped to the viewport.
  {
    const panel = document.getElementById("live-panel");
    const head = panel.querySelector(".live-head");
    let dragging = false, sx = 0, sy = 0, sl = 0, st = 0;
    head.addEventListener("mousedown", e => {
      if (e.target.closest("button")) return; // don't drag via the × close button
      dragging = true;
      const rect = panel.getBoundingClientRect();
      sx = e.clientX; sy = e.clientY; sl = rect.left; st = rect.top;
      e.preventDefault();
    });
    window.addEventListener("mousemove", e => {
      if (!dragging) return;
      const x = sl + (e.clientX - sx);
      const y = st + (e.clientY - sy);
      const left = Math.max(0, Math.min(x, window.innerWidth - panel.offsetWidth));
      const top = Math.max(0, Math.min(y, window.innerHeight - panel.offsetHeight));
      panel.style.left = left + "px";
      panel.style.top = top + "px";
      panel.style.right = "auto";
    });
    window.addEventListener("mouseup", () => { dragging = false; });
  }

  loadManifest();

  // 若本页由 minfer --viz 伺服（同源 /viz/graph 可达），自动填充直播地址并连接
  fetch("/viz/graph").then(r => {
    if (r.ok) {
      document.getElementById("live-url").value = location.origin;
      document.getElementById("live-panel").hidden = false;
      connectLive();
    }
  }).catch(() => {});

  // ?file= 直接加载
  const params = new URLSearchParams(location.search);
  const file = params.get("file");
  if (file) fetch(file).then(r => r.json()).then(loadDoc).catch(e => console.error("loadDoc:", e));

  // With no graph loaded yet, nothing is playable — gray out the playback and
  // load-graph controls until a graph/trace is actually loaded (or live connects).
  updatePlaybackEnabled();
}

function resetPlayback() {
  for (const n of state.nodes) if (nodeEls[n.id]) nodeEls[n.id].classList.remove("executed", "ready", "current");
  for (const e of state.edges) if (edgeEls[e.id]) edgeEls[e.id].classList.remove("executed", "current");
  for (const [, b] of state.flowBoxEls) b.g.classList.remove("executed", "current");
  updateStepInfo();
}

function buildLegend() {
  const title = document.getElementById("legend-title-text");
  if (state.viewMode === "flow") {
    if (title) title.textContent = "Pipeline legend";
    buildPipelineLegend();
  } else {
    if (title) title.textContent = "Operator legend";
    buildOperatorLegend();
  }
}

const BACKEND_SWATCHES = [
  ["metal", "Metal (MPS) backend"],
  ["cpu", "CPU backend"],
  ["cuda", "CUDA backend"],
  ["none", "unassigned"],
];
function backendColorRows() {
  let h = "<tr><th colspan=2 style='padding-top:10px'>Backend colors</th></tr>";
  for (const [b, desc] of BACKEND_SWATCHES) {
    h += `<tr><td><span class="legend-swatch" style="background:var(--${b === "none" ? "none" : b})"></span><span class="op-name">${b}</span></td><td>${desc}</td></tr>`;
  }
  return h;
}

function buildOperatorLegend() {
  const ops = [
    ["input", "Leaf input (token/positions/KV indices)"],
    ["get_rows", "Embedding lookup (token embedding)"],
    ["rms_norm", "RMSNorm（pre-norm）"],
    ["qk_norm", "Per-head Q/K normalization (Qwen3)"],
    ["matmul", "Matrix multiply (quantized weight)"],
    ["add", "Add (residual/bias)"],
    ["silu", "SiLU activation"],
    ["swiglu", "Fused SwiGLU (silu(gate)·up)"],
    ["rope", "Rotary position embedding"],
    ["attn", "Attention Q·Kᵀ·V"],
    ["kvcache_store", "K/V write to cache"],
    ["kvcache_load", "K/V read from cache"],
    ["fused_qkv", "Decode QKV fusion (concat matmul + bias + rope + store)"],
    ["fused_ffn", "Decode FFN fusion (concat matmul + swiglu)"],
  ];
  let h = "<tr><th>Op</th><th>Description</th></tr>";
  for (const [op, desc] of ops) h += `<tr><td class="op-name">${op}</td><td>${desc}</td></tr>`;
  h += backendColorRows();
  document.getElementById("legend-table").innerHTML = h;
}

function buildPipelineLegend() {
  const stages = [
    ["Input", "token_ids / positions (leaf input, filled each step)"],
    ["Embedding", "token_embd lookup → token vector"],
    ["RMSNorm", "per-token layer norm (pre-attn / pre-FFN / per-head Q·K)"],
    ["Attention", "Q·Kᵀ·V (GQA) + RoPE + KV cache read/write"],
    ["+ Residual", "add the block input back onto the sublayer output"],
    ["FFN", "gate·up (SwiGLU) → down projection"],
    ["Final RMSNorm", "pre-logits norm (output_norm)"],
    ["Logits", "lm_head matmul → vocab scores"],
    ["Sampler", "top-k / top-p → next token (decode only)"],
  ];
  let h = "<tr><th>Stage (one box = one function)</th><th>What it is</th></tr>";
  for (const [name, desc] of stages) h += `<tr><td class="op-name">${name}</td><td>${desc}</td></tr>`;
  h += "<tr><th colspan=2 style='padding-top:10px'>Pipeline order</th></tr>";
  h += `<tr><td colspan=2 class="legend-flow">Input → Embedding → [ RMSNorm → Attention → +Residual → RMSNorm → FFN → +Residual ] × N → Final RMSNorm → Logits → [ Sampler ]</td></tr>`;
  h += "<tr><th colspan=2 style='padding-top:10px'>Box markings</th></tr>";
  h += `<tr><td><span class="legend-swatch" style="background:hsl(48,55%,62%)"></span><span class="op-name">executed</span></td><td>pale fill by data magnitude (blue→red) + dark text</td></tr>`;
  h += `<tr><td><span class="legend-swatch" style="background:transparent;border:1.5px dashed var(--accent-2)"></span><span class="op-name">fused</span></td><td>one fused kernel (QKV✚ / FFN✚, dashed border)</td></tr>`;
  h += `<tr><td><span class="legend-swatch" style="background:transparent;border:1.5px dashed var(--fg-dim)"></span><span class="op-name">collapsed</span></td><td>the ×N layer summary — click to expand</td></tr>`;
  h += `<tr><td><span class="legend-swatch" style="background:transparent;border-top:1.5px solid var(--fg-dim)"></span><span class="op-name">arrow</span></td><td>next reasoning step (a layer also loops at the top)</td></tr>`;
  h += backendColorRows();
  document.getElementById("legend-table").innerHTML = h;
}

/* 边索引（由 prepare 构建并随 doc 一起装载） */

// 调试 / 测试句柄
window.minferViz = { getDoc: () => state.doc, getState: () => state };

window.addEventListener("DOMContentLoaded", init);
