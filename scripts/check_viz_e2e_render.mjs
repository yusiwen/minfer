#!/usr/bin/env node
/* Headless smoke check for the end-to-end animation (viz/e2e.js).
 *
 * There is no browser here: DOM and canvas are stubbed just enough to boot the
 * page, play every step of the timeline, switch language and load a second
 * sample. What it proves is that the render path never throws, that drawing
 * actually happens, and that every step produces a non-empty panel -- the
 * visual judgement stays with a human (or a screenshot), but regressions that
 * crash a frame are caught here.
 *
 * Usage: node scripts/check_viz_e2e_render.mjs
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const require = createRequire(import.meta.url);

// ------------------------------------------------------------------ fake DOM
const drawCalls = [];
function makeClassList() {
  const set = new Set();
  return {
    add: (c) => set.add(c),
    remove: (c) => set.delete(c),
    toggle: (c, on) => { if (on === undefined) { set.has(c) ? set.delete(c) : set.add(c); } else if (on) set.add(c); else set.delete(c); },
    contains: (c) => set.has(c),
    _set: set
  };
}

class El {
  constructor(id, tag) {
    this.id = id || '';
    this.tagName = (tag || 'div').toUpperCase();
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.classList = makeClassList();
    this.textContent = '';
    this._html = '';
    this.value = '';
    this.checked = true;
    this.options = [];
    this.selectedIndex = 0;
    this.offsetLeft = 0;
    this.offsetWidth = 82;
    this.clientWidth = 1200;
    this.clientHeight = 60;
    this.files = [];
    this._q = new Map();
  }
  set innerHTML(v) { this._html = String(v); this.children = []; }
  get innerHTML() { return this._html; }
  addEventListener() {}
  appendChild(c) {
    this.children.push(c);
    if (this.tagName === 'SELECT' && c.tagName === 'OPTION') this.options.push(c);
    return c;
  }
  querySelector(sel) {
    if (!this._q.has(sel)) this._q.set(sel, new El(sel, 'div'));
    return this._q.get(sel);
  }
  querySelectorAll() { return []; }
  closest() { return null; }
  click() {}
  scrollTo() {}
  getBoundingClientRect() { return { width: 1264, height: 846, left: 0, top: 0 }; }
}

const ctxCalls = drawCalls;
const ctxImpl = {
  measureText: (t) => ({ width: String(t == null ? '' : t).length * 6 }),
  createLinearGradient: () => ({ addColorStop() {} }),
  createPattern: () => null,
  getImageData: () => ({ data: new Uint8ClampedArray(4) })
};
function makeCtx() {
  return new Proxy({}, {
    get(_t, k) {
      if (typeof k === 'symbol') return undefined;
      if (k in ctxImpl) return ctxImpl[k];
      if (k === 'canvas') return { width: 1264, height: 846 };
      return (...args) => { ctxCalls.push(k); return undefined; };
    },
    set() { return true; }
  });
}

const els = new Map();
const canvas = new El('cv', 'canvas');
canvas.getContext = () => makeCtx();
els.set('cv', canvas);

const documentStub = {
  readyState: 'complete',
  documentElement: new El('html', 'html'),
  getElementById: (id) => {
    if (!els.has(id)) els.set(id, new El(id, id === 'file' ? 'input' : 'div'));
    return els.get(id);
  },
  createElement: (tag) => new El('', tag),
  createDocumentFragment: () => new El('', 'fragment'),
  addEventListener() {}
};

const rafQueue = [];
const windowStub = {
  devicePixelRatio: 1,
  addEventListener() {},
  requestAnimationFrame: (cb) => { rafQueue.push(cb); return rafQueue.length; }
};

// fetch serves the repo's own samples so the page loads real data.
const fetchStub = async (url) => {
  const rel = String(url).replace(/^\.\//, '');
  const file = path.join(repo, 'viz', rel);
  if (!fs.existsSync(file)) return { ok: false, status: 404, json: async () => { throw new Error('404 ' + url); } };
  return { ok: true, status: 200, json: async () => JSON.parse(fs.readFileSync(file, 'utf8')) };
};

globalThis.window = windowStub;
globalThis.document = documentStub;
globalThis.requestAnimationFrame = windowStub.requestAnimationFrame;
globalThis.fetch = fetchStub;
windowStub.E2EModel = undefined;

// ------------------------------------------------------------- load the app
require(path.join(repo, 'viz/e2e-model.js'));
windowStub.E2EModel = globalThis.E2EModel;
require(path.join(repo, 'viz/e2e.js'));

let failures = 0, checks = 0;
function ok(cond, msg) {
  checks++;
  if (!cond) { failures++; console.error('  FAIL ' + msg); }
}

function pump(n) {
  let t = 0;
  for (let i = 0; i < n; i++) {
    const cbs = rafQueue.splice(0, rafQueue.length);
    if (!cbs.length) break;
    t += 16;
    for (const cb of cbs) cb(t);
  }
  return t;
}

const app = windowStub.E2EApp;
ok(!!app, 'e2e.js must expose E2EApp');

// boot() runs on load (readyState is not "loading"); let the manifest promise settle
await new Promise(r => setTimeout(r, 20));
pump(6);

ok(app.state.profile.dims.nEmbd > 0, 'a profile must be loaded (built-in or sample)');
ok(app.state.timeline && app.state.timeline.steps.length > 0, 'the timeline must be built');
ok(drawCalls.length > 100, 'the canvas must actually be drawn on (' + drawCalls.length + ' ops)');
for (const need of ['fillRect', 'beginPath', 'fill', 'stroke', 'fillText']) {
  ok(drawCalls.includes(need), 'the render must call ctx.' + need);
}

// every step: jump there, render, and require a panel that names the stage
const total = app.state.timeline.steps.length;
let badPanels = 0, badSteps = 0;
for (let i = 0; i < total; i++) {
  try {
    app.goto(i);
    app.state.t = 0.5;
    pump(2);
    const html = app.panelHtml();
    if (!html || html.length < 60 || !/shape/.test(html)) badPanels++;
  } catch (e) {
    badSteps++;
    if (badSteps <= 3) console.error('    step ' + i + ' threw: ' + (e && e.stack || e));
  }
}
ok(badSteps === 0, `${badSteps}/${total} steps threw while rendering`);
ok(badPanels === 0, `${badPanels}/${total} steps produced an empty/short panel`);
console.log(`  played ${total} steps, ${drawCalls.length} canvas ops so far`);

// both modes, both decode presentations, and a second (fused) sample
app.setMode('prefill');
pump(2);
ok(app.chipCount() > 0, 'prefill-only timeline must not be empty');
app.setMode('decode');
pump(2);
ok(app.chipCount() > 0, 'decode-only timeline must not be empty');
app.state.detailDecode = true;
app.rebuildTimeline(true);
pump(2);
ok(app.chipCount() > 0, 'decode-detail timeline must not be empty');
app.state.detailDecode = false;
app.setMode('run');
pump(2);

const decodeSample = path.join(repo, 'viz/samples/qwen2.5-0.5b-decode-metal.json');
const fused = JSON.parse(fs.readFileSync(decodeSample, 'utf8'));
app.applyJson(fused, 'fused decode sample');
await new Promise(r => setTimeout(r, 10));
pump(4);
ok(app.state.profile.fused.ffn === true, 'the fused decode sample must be detected as FFN-fused');
ok(app.state.profile.dims.nt === 1, 'the decode sample must report nt=1');
let fusedBad = 0;
for (let i = 0; i < Math.min(app.chipCount(), 40); i++) {
  try { app.goto(i); app.state.t = 1; pump(1); } catch { fusedBad++; }
}
ok(fusedBad === 0, `${fusedBad} steps threw on the fused sample`);
console.log(`  fused sample: ${app.chipCount()} steps played`);
ok(drawCalls.length > 2000, 'the fused sample must draw too (' + drawCalls.length + ' ops)');

console.log(`\n${failures ? 'FAIL' : 'OK'}  ${checks - failures}/${checks} render checks passed`);
process.exit(failures ? 1 : 0);
