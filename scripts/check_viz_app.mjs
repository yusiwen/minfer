#!/usr/bin/env node
/* Headless gate for the operator / pipeline view (viz/app.js, #549).
 *
 * `viz/e2e.js` has two gates: the stage template is replayed against every exported
 * sample (`check_viz_e2e.mjs`), and the render path is booted on a stubbed DOM
 * (`check_viz_e2e_render.mjs`). `viz/app.js` — the operator grid, the pipeline view,
 * the node inspector, the sample dropdown, the `?file=` deep link — had none, so the
 * two freshness rules that cover it (#103's stamped loader, #546's revalidating
 * fetch) could only be source audits.
 *
 * There is no browser here. The DOM and the SVG namespace are stubbed just far enough
 * for the real `viz/app.js` to boot, load a real exported sample through the
 * documented `?file=` deep link, and render both views. What it proves, against a
 * value computed from the **export** rather than from the page:
 *
 *   * the graph grid built one group per node and one path per edge — the counts come
 *     from the sample JSON's own `nodes[].src`, not from app.js's state;
 *   * the status line and the dropdown agree with that export (the model name, the
 *     node / edge / layer counts, the manifest's entries);
 *   * clicking a node group fills the inspector with that node's name and op, and
 *     the Operators ↔ Pipeline switch builds the other view without throwing;
 *   * nothing threw while any of that ran.
 *
 * What it cannot prove: anything about pixels, CSS or layout. It catches a render
 * path that throws or stops building the structure the export implies, not a visual
 * regression — the same limit `check_viz_e2e_render.mjs` states for its own path.
 *
 * Usage: node scripts/check_viz_app.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const require = createRequire(import.meta.url);

/** The sample the gate loads, and the manifest the dropdown is built from. */
const SAMPLE = 'samples/qwen2.5-0.5b-prefill-metal.json';

// ------------------------------------------------------------------ fake DOM
function makeClassList() {
  const set = new Set();
  return {
    add: (c) => set.add(c),
    remove: (c) => set.delete(c),
    toggle: (c, on) => {
      if (on === undefined) { set.has(c) ? set.delete(c) : set.add(c); }
      else if (on) set.add(c);
      else set.delete(c);
    },
    contains: (c) => set.has(c),
    _set: set,
  };
}

class El {
  constructor(id, tag) {
    this.id = id || '';
    this.tagName = (tag || 'div').toUpperCase();
    this.children = [];
    this.parentNode = null;
    this.dataset = {};
    this.style = {};
    this.classList = makeClassList();
    this.attributes = new Map();
    this.listeners = new Map();
    this._text = '';
    this._html = '';
    this.value = '';
    this.checked = true;
    this.hidden = false;
    this.options = [];
    this.selectedIndex = 0;
    this.offsetLeft = 0;
    this.offsetTop = 0;
    this.offsetWidth = 220;
    this.offsetHeight = 140;
    this.clientWidth = 1264;
    this.clientHeight = 846;
    this.files = [];
    this._q = new Map();
  }
  // The real DOM drops the children of an element whose text or markup is replaced;
  // app.js re-renders by clearing `gNodes.textContent` first, so this matters for
  // every count below.
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text; }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  get innerHTML() { return this._html; }
  addEventListener(type, fn) {
    if (!this.listeners.has(type)) this.listeners.set(type, []);
    this.listeners.get(type).push(fn);
  }
  /** Fire the handlers registered for `type` (the gate's only way to click). */
  dispatch(type, ev = {}) {
    const event = { type, target: this, stopPropagation() {}, preventDefault() {}, ...ev };
    for (const fn of this.listeners.get(type) || []) fn(event);
  }
  appendChild(c) {
    this.children.push(c);
    c.parentNode = this;
    if (this.tagName === 'SELECT' && c.tagName === 'OPTION') this.options.push(c);
    return c;
  }
  removeChild(c) {
    const i = this.children.indexOf(c);
    if (i >= 0) this.children.splice(i, 1);
    return c;
  }
  remove() { if (this.parentNode) this.parentNode.removeChild(this); }
  querySelector(sel) {
    if (!this._q.has(sel)) this._q.set(sel, new El(sel, 'div'));
    return this._q.get(sel);
  }
  querySelectorAll() { return []; }
  closest() { return null; }
  click() { this.dispatch('click'); }
  setAttribute(k, v) {
    this.attributes.set(k, String(v));
    // The SVG builders set `class` as an attribute, so the class list has to follow
    // it — `applyFilters` and the gate both ask `classList`.
    if (k === 'class') {
      this.classList = makeClassList();
      for (const token of String(v).split(/\s+/).filter(Boolean)) this.classList.add(token);
    }
  }
  getAttribute(k) { return this.attributes.has(k) ? this.attributes.get(k) : null; }
  setPointerCapture() {}
  releasePointerCapture() {}
  getBoundingClientRect() { return { width: 1264, height: 846, left: 0, top: 0 }; }
  getContext() { return null; }
}

const els = new Map();
/** The ids whose element type matters (`options`, `checked`, `files`). */
const TAGS = {
  'model-select': 'select',
  'file-input': 'input',
  canvas: 'svg',
  speed: 'input',
  'step-slider': 'input',
  'f-kv': 'input',
  'f-io': 'input',
  'f-attn': 'input',
  'f-ffn': 'input',
  'live-url': 'input',
  'live-prompt': 'input',
  'live-max': 'input',
};
const documentStub = {
  readyState: 'complete',
  documentElement: new El('html', 'html'),
  getElementById(id) {
    if (!els.has(id)) els.set(id, new El(id, TAGS[id] || 'div'));
    return els.get(id);
  },
  createElement: (tag) => new El('', tag),
  createElementNS: (_ns, tag) => new El('', tag),
  createDocumentFragment: () => new El('', 'fragment'),
  querySelector(sel) { return documentStub.getElementById('doc:' + sel); },
  addEventListener() {},
};

const windowListeners = new Map();
const windowStub = {
  devicePixelRatio: 1,
  innerWidth: 1280,
  innerHeight: 900,
  addEventListener(type, fn) {
    if (!windowListeners.has(type)) windowListeners.set(type, []);
    windowListeners.get(type).push(fn);
  },
  dispatch(type, ev = {}) {
    for (const fn of windowListeners.get(type) || []) fn({ type, target: new El('', 'body'), ...ev });
  },
};

// `app.js` has no canvas: it draws SVG, and its only timers are the playback
// interval. Stub them so a stray timer cannot keep the gate alive.
const notifiers = [];
globalThis.window = windowStub;
globalThis.document = documentStub;
globalThis.location = { search: '?file=' + SAMPLE, origin: 'http://127.0.0.1:8081' };
globalThis.alert = (msg) => notifiers.push('alert: ' + msg);
globalThis.confirm = () => true;
globalThis.FileReader = class { readAsText() {} };
globalThis.setInterval = () => 0;
globalThis.clearInterval = () => {};

/** Serve the repository's own `viz/` files; an absolute URL is the engine's API. */
const fetchStub = async (url) => {
  const rel = String(url).replace(/^\.\//, '');
  const missing = { ok: false, status: 404, json: async () => { throw new Error('404 ' + url); } };
  if (rel.startsWith('/')) return missing;
  const file = path.join(repo, 'viz', rel);
  if (!fs.existsSync(file)) return missing;
  return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(file, 'utf8')) };
};
globalThis.fetch = fetchStub;

// ------------------------------------------------------------- load the app
require(path.join(repo, 'viz/app.js'));
const viz = windowStub.minferViz;

let failures = 0, checks = 0;
function ok(cond, msg) {
  checks++;
  if (!cond) { failures++; console.error('  FAIL ' + msg); }
}
/** Drive one interaction; a throw anywhere inside it is a named failure, not a stack
 * trace — a broken render path is exactly what this gate exists to report. */
function step(label, fn) {
  checks++;
  try {
    fn();
  } catch (err) {
    failures++;
    console.error(`  FAIL ${label} threw: ${(err && err.message) || err}`);
  }
}
const settle = () => new Promise((r) => setTimeout(r, 25));

ok(!!viz && typeof viz.getState === 'function', 'app.js must expose window.minferViz');
if (!viz) {
  console.error('\nFAIL  viz/app.js exposed no test handle');
  process.exit(1);
}

const sample = JSON.parse(fs.readFileSync(path.join(repo, 'viz', SAMPLE), 'utf8'));
const manifest = JSON.parse(fs.readFileSync(path.join(repo, 'viz/samples/manifest.json'), 'utf8'));
const expectedEdges = sample.nodes.reduce((n, node) => n + (node.src || []).length, 0);

// Boot the way the browser does: app.js waits for DOMContentLoaded.
ok(windowListeners.has('DOMContentLoaded'), 'app.js registers a DOMContentLoaded boot');
step('booting on DOMContentLoaded', () => {
  for (const fn of windowListeners.get('DOMContentLoaded') || []) fn({ type: 'DOMContentLoaded' });
});
await settle();
await settle();

const state = viz.getState();
const doc = viz.getDoc();
ok(!!doc, 'the ?file= deep link must load the sample');
ok(doc && doc.nodes.length === sample.nodes.length,
  `the loaded export must be the sample (${doc ? doc.nodes.length : 'none'} vs ${sample.nodes.length} nodes)`);

// --- the structure the export implies, counted from the export
const nodeGroupCount = documentStub.getElementById('nodes').children.length;
ok(nodeGroupCount === state.rows.length + state.nodes.length,
  `one group per node plus one row label each (${nodeGroupCount} vs ${state.rows.length} + ${state.nodes.length})`);
const edgeCount = documentStub.getElementById('edges').children.length;
ok(edgeCount === expectedEdges, `one path per edge of the export (${edgeCount} vs ${expectedEdges})`);
ok(state.edges.length === expectedEdges, `state.edges matches the export (${state.edges.length} vs ${expectedEdges})`);

// --- the page's own summary lines agree with the export
const title = documentStub.getElementById('doc-title').textContent;
ok(title.includes(String(sample.nodes.length)) && title.includes(sample.model),
  `the doc title names the model and node count (${JSON.stringify(title)})`);
ok(documentStub.getElementById('status-model').textContent.includes(sample.model),
  'the status bar names the model');
ok(documentStub.getElementById('status-counts').textContent.includes(`${expectedEdges} edges`),
  `the status bar counts the export's edges (${JSON.stringify(documentStub.getElementById('status-counts').textContent)})`);

// --- the sample dropdown was built from the manifest
const options = documentStub.getElementById('model-select').options;
ok(options.length === manifest.samples.length,
  `one dropdown entry per manifest sample (${options.length} vs ${manifest.samples.length})`);

// --- the legend is the operator one, and it lists the ops
ok(documentStub.getElementById('legend-table').innerHTML.includes('rms_norm'),
  'the operator legend lists the ops');
ok(documentStub.getElementById('legend-title-text').textContent === 'Operator legend',
  'the Operators view shows the operator legend');

// --- clicking a node fills the inspector
const groups = documentStub.getElementById('nodes').children.filter((c) => c.classList.contains('node-group'));
ok(groups.length === state.nodes.length, `every node has a clickable group (${groups.length} vs ${state.nodes.length})`);
const clicked = state.nodes[state.nodes.length - 1];
if (groups.length) {
  step('clicking a node group', () => {
    groups[groups.length - 1].dispatch('click');
    ok(state.selected === clicked.id, `a group click selects its node (${state.selected} vs ${clicked.id})`);
    const panel = documentStub.getElementById('panel-body').innerHTML;
    ok(panel.includes(clicked.name) && panel.includes(clicked.op),
      `the inspector names the clicked node (${JSON.stringify(clicked.name)})`);
  });
}

// --- the Operators ↔ Pipeline switch builds the other view, both ways
step('switching to the Pipeline view', () => {
  documentStub.getElementById('view-flow').dispatch('click');
});
ok(state.viewMode === 'flow', 'the Pipeline button switches the view');
for (const key of ['seg:input', 'seg:embed', 'seg:layers', 'seg:final_norm', 'seg:logits']) {
  ok(state.flowBoxEls.has(key), `the pipeline draws ${key}`);
}
ok(documentStub.getElementById('nodes').children.length > 0, 'the pipeline view draws into the canvas');
ok(documentStub.getElementById('legend-title-text').textContent === 'Pipeline legend',
  'the Pipeline view shows the pipeline legend');

// --- the ×N layers box expands into one row per block the export has
const collapsedBoxes = state.flowBoxEls.size;
const blkRows = state.rows.filter((r) => r.name.startsWith('blk.')).length;
const layersBox = state.flowBoxEls.get('seg:layers');
if (layersBox) {
  step('expanding the ×N layers box', () => {
    layersBox.g.dispatch('click');
  });
  ok(state.flowLayerExpanded === true, 'clicking the ×N box expands it');
  const layerKeys = [...state.flowBoxEls.keys()].filter((k) => k.startsWith('layer:')).length;
  ok(layerKeys === blkRows, `one pipeline layer box per block row (${layerKeys} vs ${blkRows})`);
  ok(state.flowBoxEls.size > collapsedBoxes,
    `the expanded view draws more boxes (${state.flowBoxEls.size} vs ${collapsedBoxes})`);
  const collapse = documentStub.getElementById('nodes').children.find((c) => c.classList.contains('flow-clickable'));
  ok(!!collapse, 'the expanded view offers a click-to-collapse label');
  if (collapse) {
    step('collapsing the layers again', () => {
      collapse.dispatch('click');
    });
    ok(state.flowLayerExpanded === false, 'clicking the label collapses the layers again');
  }
}

step('switching back to the Operators view', () => {
  documentStub.getElementById('view-dataflow').dispatch('click');
});
ok(state.viewMode === 'graph', 'the Operators button switches back');
const backCount = documentStub.getElementById('nodes').children.length;
ok(backCount === state.rows.length + state.nodes.length,
  `the grid is rebuilt on the way back (${backCount} vs ${state.rows.length} + ${state.nodes.length})`);

// --- the filters and the fit button are wired to something that runs
step('the fit button and the KV filter', () => {
  documentStub.getElementById('btn-fit').dispatch('click');
  documentStub.getElementById('f-kv').dispatch('change', { target: { checked: false } });
  documentStub.getElementById('f-kv').dispatch('change', { target: { checked: true } });
});
ok(state.filters.kv === true, 'the KV filter round-trips through its handler');

ok(notifiers.length === 0, `no alert fired during the run (${notifiers.join('; ')})`);

console.log(`  ${state.nodes.length} nodes, ${state.edges.length} edges, ${state.rows.length} rows, ` +
  `${state.flowBoxEls.size} pipeline boxes`);
console.log(`\n${failures ? 'FAIL' : 'OK'}  ${checks - failures}/${checks} app.js checks passed`);
process.exit(failures ? 1 : 0);
