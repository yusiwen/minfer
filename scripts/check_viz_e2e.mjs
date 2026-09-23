#!/usr/bin/env node
/* Headless checks for the end-to-end shape-flow model (viz/e2e-model.js).
 *
 * Two kinds of evidence, both cheap enough for CI:
 *   1. the stage template's computed shapes must equal the shapes that
 *      `minfer --dump-graph-json` actually exported for block 0 -- the layer
 *      the animation plays in detail;
 *   2. every playable step must carry positive integer dims, and decode must
 *      be the same graph with nt=1 and a KV window one cell longer per step.
 *
 * Usage: node scripts/check_viz_e2e.mjs [viz/samples]
 */
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const sampleDir = path.resolve(process.argv[2] || path.join(repo, 'viz/samples'));

const require = createRequire(import.meta.url);
const E2E = require(path.join(repo, 'viz/e2e-model.js'));
if (!E2E || !E2E.buildProfile) {
  console.error('FAIL: viz/e2e-model.js did not export E2EModel');
  process.exit(1);
}

let failures = 0;
let checks = 0;
function ok(cond, msg) {
  checks++;
  if (!cond) { failures++; console.error('  FAIL ' + msg); }
}
const eq = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const sorted = (xs) => (xs || []).map(x => JSON.stringify(x)).sort();

function validDims(dims, where) {
  ok(Array.isArray(dims) && dims.length > 0, `${where}: dims must be a non-empty array (${JSON.stringify(dims)})`);
  for (const d of dims || []) {
    ok(Number.isInteger(d) && d > 0, `${where}: dim ${d} must be a positive integer in ${JSON.stringify(dims)}`);
  }
}

function shapesOf(stage) { return stage.outputs.map(o => o.dims); }

// The template's stage -> the graph node roles it must reproduce.
const ROLE_MAP = {
  'pre.embed': ['@embed'],
  'layer.norm1': ['norm1'],
  'layer.qkv': ['qkv'],
  'layer.qk_norm': ['qk_norm'],
  'layer.attn': ['attn'],
  'layer.attn_out': ['attn_out'],
  'layer.residual1': ['add#0'],
  'layer.norm2': ['norm2'],
  'layer.ffn_gu': ['ffn_gu'],
  'layer.swiglu': ['swiglu'],
  'layer.ffn_down': ['ffn_down'],
  'layer.residual2': ['add#1'],
  'layer.kv_write': ['@kv'],
  'post.final_norm': ['norm_final'],
  'post.lm_head': ['@lmHead']
};

function blockRoles(profile) {
  const roles = {};
  for (const n of profile.graphShapes.block0) (roles[n.role] ||= []).push(n.shape);
  for (const n of profile.graphShapes.epilogue) (roles[n.role] ||= []).push(n.shape);
  const adds = roles.add || [];
  roles['add#0'] = adds.length > 0 ? [adds[0]] : [];
  roles['add#1'] = adds.length > 1 ? [adds[1]] : [];
  roles['@kv'] = profile.graphShapes.kv ? [profile.graphShapes.kv] : [];
  roles['@embed'] = profile.graphShapes.embed ? [profile.graphShapes.embed] : [];
  roles['@lmHead'] = profile.graphShapes.lmHead ? [profile.graphShapes.lmHead] : [];
  return roles;
}

const files = fs.readdirSync(sampleDir).filter(f => f.endsWith('.json') && f !== 'manifest.json').sort();
if (!files.length) {
  console.error('FAIL: no samples in ' + sampleDir);
  process.exit(1);
}

const summary = [];
for (const f of files) {
  const raw = JSON.parse(fs.readFileSync(path.join(sampleDir, f), 'utf8'));
  const profile = E2E.buildProfile(raw);
  const d = profile.dims;
  console.log(`\n== ${f}  (${profile.source.kind}, ${profile.source.model})`);
  console.log(`   dims: n_embd=${d.nEmbd} n_layer=${d.nLayer} n_head=${d.nHead} hd=${d.hd} ` +
    `n_kv_head=${d.nKvHead} kv_dim=${d.kvDim} n_ff=${d.nFf} vocab=${d.vocab} n_ctx=${d.nCtx} nt=${d.nt}`);
  console.log(`   fused(layer 0): qkv=${profile.fused.qkv} ffn=${profile.fused.ffn}` +
    `  (graph-wide: qkv ${profile.provenance.fusedLayers.qkv.length}/${d.nLayer},` +
    ` ffn ${profile.provenance.fusedLayers.ffn.length}/${d.nLayer})`);

  for (const [k, v] of Object.entries(d)) ok(Number.isInteger(v) && v > 0, `${f}: dim ${k}=${v} must be a positive integer`);

  // ---- 1. template shapes vs the exported graph (block 0 + prologue/epilogue)
  const pre = E2E.script(profile, { phase: 'prefill', layer: 0 });
  const byKey = Object.fromEntries(pre.map(s => [s.key, s]));
  const roles = blockRoles(profile);
  for (const [key, roleNames] of Object.entries(ROLE_MAP)) {
    const stage = byKey[key];
    if (!stage) continue;
    // The decode graph folds some stages into fused nodes; those stages are
    // derived, and the shape they derive is checked by the fused node instead.
    if (stage.merged) continue;
    if (key === 'layer.swiglu' && profile.fused.ffn) continue;
    if (key === 'layer.rope' && profile.fused.qkv) continue;
    for (const roleName of roleNames) {
      const want = roles[roleName] || [];
      if (!want.length) continue;
      const got = shapesOf(stage);
      ok(eq(sorted(want), sorted(got)),
        `${f}: ${key} shape ${JSON.stringify(got)} != graph ${roleName} ${JSON.stringify(want)}`);
    }
  }

  // ---- 2. attention internals
  const attn = byKey['layer.attn'];
  const internals = attn.internal || [];
  const scores = internals.find(x => x.hero);
  ok(!!scores, `${f}: attention must expose the score matrix`);
  ok(scores && eq(scores.dims, [d.nHead, d.nt, d.nt]),
    `${f}: scores ${JSON.stringify(scores && scores.dims)} != [${d.nHead},${d.nt},${d.nt}] (prefill nKv == nt)`);
  const nQt = d.nHead * d.hd;
  ok(internals[internals.length - 1] && eq(internals[internals.length - 1].dims, [nQt, d.nt]),
    `${f}: attention must merge heads back to [n_head*hd, nt] = [${nQt},${d.nt}]`);
  ok(eq(shapesOf(byKey['layer.attn'])[0], [nQt, d.nt]), `${f}: attn output width must be n_head*hd`);
  ok(eq(shapesOf(byKey['layer.attn_out'])[0], [d.nEmbd, d.nt]), `${f}: attn_out must map back to n_embd`);

  // ---- 3. decode = same graph, nt=1, KV window +1 per step
  const dec1 = E2E.script(profile, { phase: 'decode', layer: 0, step: 1 });
  const dec2 = E2E.script(profile, { phase: 'decode', layer: 0, step: 2 });
  const act = (st, key) => st.find(s => s.key === key).outputs[0].dims;
  ok(eq(act(dec1, 'pre.embed'), [d.nEmbd, 1]), `${f}: decode embed must be [n_embd, 1]`);
  const kv1 = act(dec1, 'layer.kv_write');
  const kv2 = act(dec2, 'layer.kv_write');
  ok(eq(kv1, [d.kvDim, d.nCtx]) && eq(kv2, kv1), `${f}: KV region shape must not change with step`);
  ok(dec1.find(s => s.key === 'layer.attn').internal[2].dims[2] === d.nt + 1,
    `${f}: decode step 1 scores must span nt+1 keys`);
  ok(dec2.find(s => s.key === 'layer.attn').internal[2].dims[2] === d.nt + 2,
    `${f}: decode step 2 scores must span nt+2 keys`);

  // ---- 4. the playable timeline
  const tl = E2E.buildTimeline(profile, { decodeSteps: 3 });
  ok(tl.steps.length > 0, `${f}: timeline must not be empty`);
  let bad = 0;
  for (const s of tl.steps) {
    if (s.kind === 'stage') {
      if (!s.stage || !s.stage.outputs.length) bad++;
      else for (const o of s.stage.outputs) validDims(o.dims, `${f}/${s.stage.key}`);
    } else if (s.kind === 'sweep') {
      for (const sub of s.subs) for (const o of sub.outputs) validDims(o.dims, `${f}/sweep/${sub.key}`);
    }
  }
  ok(bad === 0, `${f}: ${bad} timeline steps without output tensors`);
  const kinds = tl.steps.reduce((m, s) => (m[s.kind] = (m[s.kind] || 0) + 1, m), {});
  console.log(`   timeline: ${tl.steps.length} steps ${JSON.stringify(kinds)}`);
  summary.push({ file: f, steps: tl.steps.length, dims: d });
}

// ---- 5. the built-in profile must animate with no file loaded at all
{
  const p = E2E.BUILTIN;
  console.log(`\n== built-in profile (${p.source.model})`);
  const st = E2E.script(p, { phase: 'prefill', layer: 0 });
  ok(st.length >= 16, 'builtin: stage list looks too short');
  for (const s of st) for (const o of s.outputs) validDims(o.dims, 'builtin/' + s.key);
  const tl = E2E.buildTimeline(p, { decodeSteps: 3 });
  ok(tl.steps.length > 20, 'builtin: timeline looks too short');
  console.log(`   ${st.length} stages, ${tl.steps.length} timeline steps`);
}

console.log(`\n${failures ? 'FAIL' : 'OK'}  ${checks - failures}/${checks} checks passed over ${files.length} samples`);
process.exit(failures ? 1 : 0);
