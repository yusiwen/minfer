/* minfer end-to-end shape flow -- pure data layer.
 *
 * Turns a `minfer --dump-graph-json` export into the ordered *stage script*
 * that the animation plays, carrying the tensor shape of every step.
 *
 * No DOM and no canvas here: this file is loaded both by viz/e2e.html and by
 * scripts/check_viz_e2e.mjs (Node), so the shape math is testable headlessly.
 *
 * Shape conventions, straight from src/graph/json.rs (dim0 first):
 *   activation  [features, tokens, 1, 1]   e.g. [896, 7]
 *   token ids   [tokens, 1, 1, 1]
 *   KV region   [kv_dim, n_ctx, 1, 1]      e.g. [128, 4096]
 *   logits      [vocab, tokens, 1, 1]      e.g. [151936, 7]
 */
(function (root) {
  'use strict';

  var VERSION = 1;

  // ------------------------------------------------------- shape utilities
  // Trailing singleton dims are noise in the export; the animation only ever
  // reasons about the leading 1-3 real dims.
  function realDims(shape) {
    var d = (shape || []).slice();
    while (d.length > 1 && d[d.length - 1] === 1) d.pop();
    return d;
  }

  function prod(dims) {
    var p = 1;
    for (var i = 0; i < dims.length; i++) p *= dims[i];
    return p;
  }

  function fmtShape(dims) { return '[' + dims.join(', ') + ']'; }

  function fmtCount(n) {
    if (!isFinite(n)) return '?';
    if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e7 ? 0 : 1) + 'M';
    if (n >= 1e3) return (n / 1e3).toFixed(n >= 1e4 ? 0 : 1) + 'k';
    return String(n);
  }

  // A tensor as the animation sees it: graph-native dims plus a rendering kind.
  function T(dims, kind, name, detail) {
    return { dims: dims.slice(), kind: kind || 'act', name: name || '', detail: detail || '' };
  }

  // ------------------------------------------------------- graph inspection
  function weightName(n) { return (n && n.meta && n.meta.weight) || ''; }

  function blockOf(name) {
    var m = /(?:^|\.)blk\.(\d+)\./.exec(name || '');
    return m ? parseInt(m[1], 10) : -1;
  }

  // The exported graph has no explicit layer tag on `add` / `silu` / `swiglu`,
  // but it is emitted block by block, so a block is the contiguous node slice
  // between its first and last identified node.
  function layerSlice(nodes, layer) {
    var first = -1, last = -1;
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      var owns = blockOf(weightName(n)) === layer || blockOf(n.name) === layer ||
        (n.meta && n.meta.layer === layer) ||
        new RegExp('^kv_(store|load)\\.' + layer + '$').test(n.name || '');
      if (owns) { if (first < 0) first = i; last = i; }
    }
    if (first < 0) return [];
    return nodes.slice(first, last + 1);
  }

  var KNOWN_OPS = ['rms_norm', 'qk_norm', 'rope', 'kvcache_store', 'kvcache_load',
    'attn', 'fused_qkv', 'fused_ffn', 'silu', 'swiglu', 'add', 'matmul', 'get_rows'];

  function roleOf(n) {
    var w = weightName(n);
    switch (n.op) {
      case 'rms_norm':
        if (/attn_norm/.test(w)) return 'norm1';
        if (/ffn_norm/.test(w)) return 'norm2';
        return 'norm_final';
      case 'qk_norm': return 'qk_norm';
      case 'rope': return 'rope';
      case 'kvcache_store': return 'kv_write';
      case 'kvcache_load': return 'kv_read';
      case 'attn': return 'attn';
      case 'fused_qkv': return 'qkv';
      case 'fused_ffn': return 'ffn_gu';
      case 'silu': return 'silu';
      case 'swiglu': return 'swiglu';
      case 'add': return 'add';
      case 'get_rows': return 'embed';
      case 'matmul':
        if (/attn_[qkv]\.weight/.test(w)) return 'qkv';
        if (/attn_output\.weight/.test(w)) return 'attn_out';
        if (/ffn_(gate|up)\.weight/.test(w)) return 'ffn_gu';
        if (/ffn_down\.weight/.test(w)) return 'ffn_down';
        // `output.weight` normally; tied-embedding models (Qwen3) reuse
        // `token_embd.weight` as the LM head.
        if (/(^|\.)output\.weight/.test(w) || /(^|\.)token_embd\.weight/.test(w)) return 'lm_head';
        return 'matmul?';
      default: return n.op;
    }
  }

  function pick(nodes, role) {
    var out = [];
    for (var i = 0; i < nodes.length; i++) if (roleOf(nodes[i]) === role) out.push(nodes[i]);
    return out;
  }

  function firstOf(nodes, role) { var p = pick(nodes, role); return p.length ? p[0] : null; }

  // Node shapes are rank-2 in every case the animation compares: activations
  // and logits as [features, tokens], KV regions as [kv_dim, n_ctx]. Trailing
  // singleton dims are dropped, leading ones are NOT (decode has tokens = 1).
  function nodeShape(n) {
    if (!n || !n.shape) return null;
    return n.shape.length >= 2 ? [n.shape[0], n.shape[1]] : n.shape.slice();
  }

  // ------------------------------------------------------------- the profile
  // A profile is everything the stage template needs: the model's dims, the
  // fusion choices this graph made, and where every number came from.
  function profileFromDims(dims, extra) {
    var p = {
      version: VERSION,
      source: (extra && extra.source) || { kind: 'builtin', model: 'built-in profile' },
      dims: dims,
      fused: (extra && extra.fused) || { qkv: false, ffn: false },
      has: (extra && extra.has) || { qkNorm: false, kv: true, rope: true },
      wtype: (extra && extra.wtype) || {},
      backends: (extra && extra.backends) || [],
      prompt: (extra && extra.prompt) || { text: 'Hello!', nt: dims.nt || 4 },
      notes: (extra && extra.notes) || [],
      provenance: (extra && extra.provenance) || {}
    };
    return p;
  }

  // A `MINFER_TRACE` export wraps the graph per phase; accept both shapes.
  function normalizeExport(raw) {
    if (raw && raw.nodes && raw.nodes.length) return { graph: raw, trace: null, kind: raw.kind || 'graph' };
    if (raw && raw.phases && raw.phases.length) {
      var g = raw.phases[0].graph || raw.phases[0];
      return { graph: g, trace: raw, kind: g.kind || 'graph' };
    }
    return { graph: raw || {}, trace: null, kind: 'unknown' };
  }

  // Real generated tokens, when the file came from MINFER_TRACE:trace.json.
  //
  // The sampler runs once per forward pass, so there is exactly one token per
  // prefill + decode step: the trace records it as the *input* of the next
  // decode step (`phases[1].steps[i].token/.text`), and the very last one only
  // as an id in that step's `logits_top`. Length is therefore
  // `1 + decode_steps`, which is exactly the number of samples the animation
  // plays -- so the text strip stays in step with the rail.
  function traceTokens(trace) {
    if (!trace || !trace.phases) return null;
    var out = [];
    var decode = trace.phases[1];
    var steps = (decode && decode.steps) || [];
    for (var i = 0; i < steps.length; i++) {
      out.push({ token: steps[i].token, text: steps[i].text || null });
    }
    // The token the last recorded step sampled: id (and probability) only.
    if (steps.length) {
      var top = steps[steps.length - 1].logits_top;
      if (top && top.length) out.push({ token: top[0][0], text: null, prob: top[0][1] });
    }
    return out.length ? out : null;
  }

  function buildProfile(raw) {
    var norm = normalizeExport(raw);
    var graph = norm.graph;
    var nodes = (graph && graph.nodes) || [];
    var embed = firstOf(nodes, 'embed');
    var attn = firstOf(nodes, 'attn');
    var fusedQkv = firstOf(nodes, 'qkv');
    var fusedFfn = firstOf(nodes, 'ffn_gu');
    var kvNode = firstOf(nodes, 'kv_write') || firstOf(nodes, 'kv_read');
    var lmHead = firstOf(nodes, 'lm_head');
    var rope = firstOf(nodes, 'rope');
    var qkNorm = firstOf(nodes, 'qk_norm');
    var ffnDown = null, ffnGu = [];

    // Walk blocks once: layer count, fusion flags, per-weight quant types.
    // Fusion is per layer (it depends on the quant types meeting), so the
    // animation follows the layer it actually plays -- layer 0.
    var nLayer = 0, wtype = {}, backends = {}, fusedQkvLayers = {}, fusedFfnLayers = {};
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      var blk = blockOf(weightName(n));
      var layerIdx = blk >= 0 ? blk : (n.meta && typeof n.meta.layer === 'number' ? n.meta.layer : -1);
      if (layerIdx >= 0 && layerIdx + 1 > nLayer) nLayer = layerIdx + 1;
      if (n.op === 'fused_qkv') fusedQkvLayers[layerIdx < 0 ? 0 : layerIdx] = true;
      if (n.op === 'fused_ffn') fusedFfnLayers[layerIdx < 0 ? 0 : layerIdx] = true;
      if (n.backend) backends[n.backend] = true;
      var w = weightName(n);
      if (w && n.meta && n.meta.wtype) wtype[w] = n.meta.wtype;
    }
    var useFusedQkv = !!fusedQkvLayers[0];
    var useFusedFfn = !!fusedFfnLayers[0];

    var meta = (attn && attn.meta) || {};
    var hd = meta.hd || (rope && rope.meta && rope.meta.hd) || 0;
    var hdKv = meta.hd_kv || hd;
    var nHead = meta.n_head || (rope && rope.meta && rope.meta.n_head) || 0;
    var nKvHead = meta.n_head_kv || kvHeadsFrom(nodes) || nHead;

    var nEmbd = (embed && nodeShape(embed)[0]) ||
      (attn && nodeShape(attn)[0]) ||
      (firstOf(nodes, 'norm1') && nodeShape(firstOf(nodes, 'norm1'))[0]) || 0;
    // The token count is the raw second dim: for a decode graph it is 1, which
    // a "drop trailing singleton" pass would erase.
    var tokenIds = null;
    for (var t = 0; t < nodes.length; t++) {
      if (nodes[t].op === 'input' && /token_ids/.test(nodes[t].name || '')) { tokenIds = nodes[t]; break; }
    }
    var nt = (embed && embed.shape && embed.shape.length > 1) ? embed.shape[1]
      : (tokenIds ? tokenIds.shape[0] : 1);

    if (!fusedFfn) {
      var blk0 = layerSlice(nodes, 0);
      ffnGu = pick(blk0, 'ffn_gu');
      ffnDown = firstOf(blk0, 'ffn_down');
    }

    var nFf = (fusedFfn && fusedFfn.meta && fusedFfn.meta.nf) ||
      (ffnGu.length ? (ffnGu[0].meta && ffnGu[0].meta.out_dim) || nodeShape(ffnGu[0])[0] : 0) ||
      (function () { var s = firstOf(nodes, 'swiglu'); return s ? nodeShape(s)[0] : 0; })();

    var vocab = (embed && embed.meta && embed.meta.vocab_size) ||
      (lmHead && lmHead.meta && lmHead.meta.out_dim) ||
      (lmHead && nodeShape(lmHead)[0]) || 0;

    var nCtx = kvNode ? nodeShape(kvNode)[1] : 0;
    var kvDim = (kvNode && nodeShape(kvNode)[0]) || (nKvHead && hdKv ? nKvHead * hdKv : 0);

    var dims = {
      nEmbd: nEmbd, nt: nt, vocab: vocab, nCtx: nCtx, nLayer: nLayer,
      nHead: nHead, nKvHead: nKvHead, hd: hd, hdKv: hdKv, kvDim: kvDim, nFf: nFf
    };

    // The exported prefill graph finishes the last layer at nt=1: only the
    // last position is needed for logits. Surface it instead of pretending the
    // epilogue keeps the prompt length.
    var epilogueNt = (lmHead && lmHead.shape && lmHead.shape.length > 1) ? lmHead.shape[1] : nt;

    var promptText = (graph && graph.prompt) || (norm.trace && norm.trace.prompt) || 'Hello!';
    var p = profileFromDims(dims, {
      source: {
        kind: (graph && graph.kind) || 'graph',
        model: (graph && graph.model) || 'unknown model',
        backend: Object.keys(backends).join('+') || 'n/a',
        nodes: nodes.length
      },
      fused: { qkv: useFusedQkv, ffn: useFusedFfn },
      has: { qkNorm: !!qkNorm, kv: !!kvNode, rope: !!rope },
      wtype: wtype,
      backends: Object.keys(backends),
      prompt: { text: promptText, nt: nt },
      provenance: {
        dimsFromGraph: !!(embed && attn),
        graphNt: nt,
        graphNLayer: nLayer,
        // Fusion is per layer; a decode graph may fuse only some of them.
        fusedLayers: {
          qkv: Object.keys(fusedQkvLayers).map(Number).sort(function (a, b) { return a - b; }),
          ffn: Object.keys(fusedFfnLayers).map(Number).sort(function (a, b) { return a - b; })
        }
      }
    });
    p.traceTokens = traceTokens(norm.trace);
    p.source.kind = norm.kind;
    p.lastTokenOnly = epilogueNt < nt;
    p.epilogueNt = Math.max(1, epilogueNt);

    // Real weight names, so tied-embedding models print the truth.
    var finalNormNode = firstOf(nodes, 'norm_final');
    p.weights = {
      embed: (embed && weightName(embed)) || 'token_embd.weight',
      lmHead: (lmHead && weightName(lmHead)) || 'output.weight',
      finalNorm: (finalNormNode && weightName(finalNormNode)) || 'output_norm.weight'
    };
    p.weights.tied = p.weights.lmHead === p.weights.embed;

    // Real graph shapes for the prologue / epilogue / block 0, kept for the
    // headless check (scripts/check_viz_e2e.mjs cross-checks the template's
    // computed shapes against the export).
    p.graphShapes = {
      embed: nodeShape(embed),
      lmHead: nodeShape(lmHead),
      attn: nodeShape(attn),
      kv: nodeShape(kvNode),
      block0: layerSlice(nodes, 0).map(function (n) {
        return { id: n.id, op: n.op, role: roleOf(n), shape: nodeShape(n), weight: weightName(n) };
      }),
      prologue: [firstOf(nodes, 'embed')].map(function (n) {
        return { id: n.id, op: n.op, role: roleOf(n), shape: nodeShape(n), weight: weightName(n) };
      }),
      epilogue: nodes.filter(function (n) {
        var r = roleOf(n);
        return r === 'norm_final' || r === 'lm_head';
      }).map(function (n) {
        return { id: n.id, op: n.op, role: roleOf(n), shape: nodeShape(n), weight: weightName(n) };
      })
    };
    return p;
  }

  // The KV head count is not on every node; the K projection's out dim is the
  // most reliable second source.
  function kvHeadsFrom(nodes) {
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.op === 'fused_qkv' && n.meta && n.meta.nk) return n.meta.nk;
      if (n.op === 'matmul' && /attn_k\.weight/.test(weightName(n)) && n.meta && n.meta.out_dim) {
        var ropeHd = null;
        for (var j = 0; j < nodes.length; j++) {
          if (nodes[j].op === 'rope' && nodes[j].meta && nodes[j].meta.hd) { ropeHd = nodes[j].meta.hd; break; }
        }
        if (ropeHd) return Math.round(n.meta.out_dim / ropeHd);
      }
    }
    return 0;
  }

  // ------------------------------------------------------------------ phases
  // Prefill writes `nt` cells; decode writes one more cell per step on top of
  // the prompt. nKv is the causal window the query attends over.
  function contextFor(profile, phase, step) {
    var d = profile.dims;
    if (phase === 'decode') {
      var ctx = (d.nt || 1) + Math.max(1, step | 0);
      return { nt: 1, nKv: Math.min(ctx, d.nCtx || ctx), pos: (d.nt || 1) + Math.max(0, (step | 0) - 1) };
    }
    return { nt: d.nt || 1, nKv: d.nt || 1, pos: 0 };
  }

  function ctxOf(profile, opts) {
    var phase = opts.phase || 'prefill';
    var c = contextFor(profile, phase, opts.step || 0);
    var f = profile.fused;
    var d = profile.dims;
    var nQt = (d.nHead && d.hd) ? d.nHead * d.hd : d.nEmbd;
    return {
      phase: phase,
      step: opts.step || 0,
      layer: opts.layer == null ? 0 : opts.layer,
      nLayer: d.nLayer,
      nt: c.nt, nKv: c.nKv, pos: c.pos,
      // From the final norm on, the engine only carries the last position.
      ntLogits: (phase === 'decode') ? 1 : Math.max(1, Math.min(profile.epilogueNt || c.nt, c.nt)),
      lastTokenOnly: !!profile.lastTokenOnly && phase !== 'decode',
      nEmbd: d.nEmbd, vocab: d.vocab, nCtx: d.nCtx,
      nHead: d.nHead, nKvHead: d.nKvHead, hd: d.hd, hdKv: d.hdKv,
      kvDim: d.kvDim || (d.nKvHead * d.hdKv), nFf: d.nFf, nQt: nQt,
      fusedQkv: !!f.qkv, fusedFfn: !!f.ffn,
      hasQkNorm: !!profile.has.qkNorm,
      promptText: profile.prompt.text,
      weights: profile.weights || { embed: 'token_embd.weight', lmHead: 'output.weight', finalNorm: 'output_norm.weight' },
      lastIdx: c.nt - 1
    };
  }

  // ------------------------------------------------------------------ stages
  // One template drives both phases and both fusion choices: only the shapes
  // and the labels change. `mk(ctx)` returns the stage's output tensor(s).
  var TEMPLATE = [
    {
      key: 'pre.tokenize', group: 'pre', layerStage: false,
      label: 'Tokenize', op: 'BPE',
      note: 'The prompt text becomes token ids; positions are 0..nt-1.',
      math: function (c) { return 'prompt \u2192 ' + c.nt + ' ids, positions [0..' + (c.nt - 1) + ']'; },
      mk: function (c) { return [T([c.nt], 'ids', 'token_ids')]; },
      detail: function (c) { return 'i32 ids \u00b7 ' + c.nt + ' token(s)'; }
    },
    {
      key: 'pre.embed', group: 'pre', layerStage: false,
      label: 'Embedding', op: 'get_rows',
      weight: function (c) { return c.weights.embed; },
      note: 'Gather the embedding rows: a lookup, not a matmul.',
      math: function (c) { return 'E[' + (c.vocab || 'V') + ',' + c.nEmbd + '] rows \u2192 [' + c.nEmbd + ',' + c.nt + ']'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'x', 'n_embd \u00d7 nt')]; },
      detail: function (c) { return fmtCount(c.nEmbd * c.nt) + ' f32 values'; }
    },
    {
      key: 'layer.norm1', group: 'layer', layerStage: true,
      label: 'RMSNorm', op: 'rms_norm',
      weight: function (c) { return 'blk.' + c.layer + '.attn_norm.weight'; },
      note: 'Normalize each token across the feature dim (dim0). Shape is unchanged.',
      math: function (c) { return 'x / rms(x) \u00b7 w'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'x_norm')]; }
    },
    {
      key: 'layer.qkv', group: 'layer', layerStage: true,
      label: 'QKV projection', op: 'matmul \u00d73',
      weight: function (c) {
        return 'blk.' + c.layer + '.attn_{q,k,v}.weight';
      },
      weightKeys: function (c) {
        return ['blk.' + c.layer + '.attn_q.weight', 'blk.' + c.layer + '.attn_k.weight', 'blk.' + c.layer + '.attn_v.weight'];
      },
      note: 'Three matmuls against the weight: the feature dim changes, the token count does not.',
      math: function (c) {
        return 'q = x\u00b7Wq\u1d40 [' + c.nQt + ',' + c.nt + '] \u00b7 k = x\u00b7Wk\u1d40 [' + c.kvDim + ',' + c.nt + '] \u00b7 v = x\u00b7Wv\u1d40';
      },
      mk: function (c) {
        if (c.fusedQkv) {
          return [T([c.nQt + 2 * c.kvDim, c.nt], 'act', 'qkv', 'one fused matmul, split after')];
        }
        return [
          T([c.nQt, c.nt], 'act', 'q', 'n_head \u00d7 head_dim'),
          T([c.kvDim, c.nt], 'act', 'k', 'n_kv_head \u00d7 head_dim'),
          T([c.kvDim, c.nt], 'act', 'v', 'n_kv_head \u00d7 head_dim')
        ];
      },
      fusedLabel: 'QKV projection + RoPE (fused)',
      fusedOp: 'fused_qkv'
    },
    {
      key: 'layer.qk_norm', group: 'layer', layerStage: true, optional: true,
      when: function (c) { return c.hasQkNorm; },
      label: 'QK-Norm', op: 'qk_norm',
      weight: function (c) { return 'blk.' + c.layer + '.attn_{q,k}_norm.weight'; },
      weightKeys: function (c) {
        return ['blk.' + c.layer + '.attn_q_norm.weight', 'blk.' + c.layer + '.attn_k_norm.weight'];
      },
      note: 'Qwen3 normalizes each head of q and k before RoPE. Shape unchanged.',
      math: function (c) { return 'per-head RMSNorm over hd'; },
      mk: function (c) {
        return [
          T([c.nQt, c.nt], 'act', 'q'),
          T([c.kvDim, c.nt], 'act', 'k')
        ];
      }
    },
    {
      key: 'layer.rope', group: 'layer', layerStage: true,
      label: 'RoPE', op: 'rope',
      weight: function () { return null; },
      note: 'Rotate q and k by their position. Shape unchanged -- this is what puts position into attention.',
      math: function (c) { return 'rotate pairs of hd by angle(pos, ' + (c.hd || 64) + ')'; },
      mk: function (c) {
        if (c.fusedQkv) return [T([c.nQt + 2 * c.kvDim, c.nt], 'act', 'qkv', 'rotated inside the fused node')];
        return [
          T([c.nQt, c.nt], 'act', 'q_roped'),
          T([c.kvDim, c.nt], 'act', 'k_roped')
        ];
      },
      mergedNote: 'The fused QKV node already applied RoPE; only shapes matter here, and they are unchanged.'
    },
    {
      key: 'layer.kv_write', group: 'layer', layerStage: true,
      label: 'KV cache write', op: 'kvcache_store',
      weight: function () { return null; },
      note: 'K and V are appended to the persistent cache at positions [pos, pos+nt). The cache is huge and mostly empty.',
      math: function (c) {
        return 'cache[' + c.kvDim + ',' + c.nCtx + '][:, ' + c.pos + ':' + (c.pos + c.nt) + '] \u2190 k, v';
      },
      mk: function (c) { return [T([c.kvDim, c.nCtx], 'kv', 'kv_cache', 'kv_dim \u00d7 n_ctx')]; },
      detail: function (c) { return 'window [' + c.pos + ', ' + (c.pos + c.nt) + ') of ' + c.nCtx + ' cells'; }
    },
    {
      key: 'layer.attn', group: 'layer', layerStage: true,
      label: 'Scaled dot-product attention', op: 'attn',
      weight: function () { return null; },
      note: 'The shape story of the whole model: heads are split out, scores are nt\u00d7nKv, then the heads are merged back.',
      math: function (c) {
        return 'softmax(q\u00b7k\u1d40/\u221ahd)\u00b7v : [' + c.nHead + ',' + c.nt + ',' + c.nKv + '] \u2192 [' + c.nHead + ',' + c.nt + ',' + c.hd + ']';
      },
      mk: function (c) { return [T([c.nQt, c.nt], 'act', 'attn_ctx', 'merged heads: n_head \u00d7 hd')]; },
      internal: function (c) {
        return [
          { label: 'q, heads split', dims: [c.nHead, c.nt, c.hd] },
          { label: 'k, heads split', dims: [c.nKvHead, c.nKv, c.hdKv] },
          { label: 'scores = q\u00b7k\u1d40 \u00b7 scale', dims: [c.nHead, c.nt, c.nKv], hero: true },
          { label: 'softmax (causal)', dims: [c.nHead, c.nt, c.nKv] },
          { label: 'ctx = p \u00b7 v', dims: [c.nHead, c.nt, c.hd] },
          { label: 'merge heads', dims: [c.nQt, c.nt] }
        ];
      },
      detail: function (c) {
        return c.nHead + ' heads \u00b7 hd ' + c.hd + ' \u00b7 ' + c.nKvHead + ' kv heads (GQA ' + (c.nHead / (c.nKvHead || 1)) + '\u00d7)';
      }
    },
    {
      key: 'layer.attn_out', group: 'layer', layerStage: true,
      label: 'Output projection', op: 'matmul',
      weight: function (c) { return 'blk.' + c.layer + '.attn_output.weight'; },
      note: 'Project the merged heads back into the residual stream: n_head\u00d7hd \u2192 n_embd (the same width only when hd\u00b7n_head == n_embd).',
      math: function (c) { return 'y = attn_ctx\u00b7Wo\u1d40 : [' + c.nQt + ',' + c.nt + '] \u2192 [' + c.nEmbd + ',' + c.nt + ']'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'attn_out')]; }
    },
    {
      key: 'layer.residual1', group: 'layer', layerStage: true,
      label: 'Residual add', op: 'add',
      weight: function () { return null; },
      note: 'Add the block input back. Both operands have the same shape, so the shape never changes.',
      math: function (c) { return 'x + attn_out [' + c.nEmbd + ',' + c.nt + ']'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'h')]; }
    },
    {
      key: 'layer.norm2', group: 'layer', layerStage: true,
      label: 'RMSNorm', op: 'rms_norm',
      weight: function (c) { return 'blk.' + c.layer + '.ffn_norm.weight'; },
      note: 'Second norm, before the feed-forward network.',
      math: function (c) { return 'x / rms(x) \u00b7 w'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'h_norm')]; }
    },
    {
      key: 'layer.ffn_gu', group: 'layer', layerStage: true,
      label: 'Gate & Up projection', op: 'matmul \u00d72',
      weight: function (c) { return 'blk.' + c.layer + '.ffn_{gate,up}.weight'; },
      weightKeys: function (c) {
        return ['blk.' + c.layer + '.ffn_gate.weight', 'blk.' + c.layer + '.ffn_up.weight'];
      },
      note: 'The FFN widens first: n_embd \u2192 n_ff (usually about 4\u00d7). This is where most weights live.',
      math: function (c) { return 'gate, up = h\u00b7Wg\u1d40, h\u00b7Wu\u1d40 \u2192 [' + c.nFf + ',' + c.nt + '] each'; },
      mk: function (c) {
        if (c.fusedFfn) return [T([2 * c.nFf, c.nt], 'act', 'gate_up', 'one fused matmul')];
        return [
          T([c.nFf, c.nt], 'act', 'gate'),
          T([c.nFf, c.nt], 'act', 'up')
        ];
      },
      fusedLabel: 'Gate & Up projection (fused)',
      fusedOp: 'fused_ffn'
    },
    {
      key: 'layer.swiglu', group: 'layer', layerStage: true,
      label: 'SiLU \u00b7 SwiGLU', op: 'silu + swiglu',
      weight: function () { return null; },
      note: 'Elementwise gating: silu(gate) \u00d7 up. Shape unchanged.',
      math: function (c) { return 'silu(gate) \u2299 up [' + c.nFf + ',' + c.nt + ']'; },
      mk: function (c) { return [T([c.nFf, c.nt], 'act', 'ffn_hidden')]; }
    },
    {
      key: 'layer.ffn_down', group: 'layer', layerStage: true,
      label: 'Down projection', op: 'matmul',
      weight: function (c) { return 'blk.' + c.layer + '.ffn_down.weight'; },
      note: 'Narrow back to the residual width so it can be added to h.',
      math: function (c) { return 'y = ffn_hidden\u00b7Wd\u1d40 [' + c.nEmbd + ',' + c.nt + ']'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'ffn_out')]; }
    },
    {
      key: 'layer.residual2', group: 'layer', layerStage: true,
      label: 'Residual add', op: 'add',
      weight: function () { return null; },
      note: 'End of the block: the output is again [n_embd, nt], ready for the next layer.',
      math: function (c) { return 'h + ffn_out [' + c.nEmbd + ',' + c.nt + ']'; },
      mk: function (c) { return [T([c.nEmbd, c.nt], 'act', 'h_out')]; }
    },
    {
      key: 'post.final_norm', group: 'post', layerStage: false,
      label: 'Final RMSNorm', op: 'rms_norm',
      weight: function (c) { return c.weights.finalNorm; },
      note: 'One last norm over the token stream.',
      math: function (c) { return 'x / rms(x) \u00b7 w [' + c.nEmbd + ',' + c.ntLogits + ']'; },
      mk: function (c) { return [T([c.nEmbd, c.ntLogits], 'act', 'h_final')]; },
      detail: function (c) {
        return c.lastTokenOnly ? 'n_embd \u00d7 1 \u2014 only the last position needs logits' : '';
      }
    },
    {
      key: 'post.lm_head', group: 'post', layerStage: false,
      label: 'LM head (logits)', op: 'matmul',
      weight: function (c) { return c.weights.lmHead + (c.weights.tied ? ' (tied to the embedding)' : ''); },
      note: 'The biggest single shape jump: n_embd \u2192 vocab, for every token.',
      math: function (c) { return 'logits = h\u00b7Wo\u1d40 [' + c.vocab + ',' + c.ntLogits + ']'; },
      mk: function (c) { return [T([c.vocab, c.ntLogits], 'logits', 'logits', fmtCount(c.vocab) + ' \u00d7 ' + c.ntLogits)]; },
      detail: function (c) { return fmtCount(c.vocab * c.ntLogits) + ' logits = ' + fmtCount(c.vocab * c.ntLogits * 4 / 1048576) + ' MiB f32'; }
    },
    {
      key: 'post.sample', group: 'post', layerStage: false,
      label: 'Sample one token', op: 'softmax \u2192 top-k \u2192 top-p',
      weight: function () { return null; },
      note: 'Only the last position matters: slice one column of logits, then sample. nt columns collapse to 1 id.',
      math: function (c) { return 'logits[:,' + (c.ntLogits - 1) + '] [' + c.vocab + '] \u2192 p \u2192 1 id'; },
      mk: function (c) {
        return [
          T([c.vocab], 'probs', 'logits_last', 'one position'),
          T([1], 'ids', 'next_id')
        ];
      }
    },
    {
      key: 'post.next', group: 'post', layerStage: false,
      label: 'Next step (decode)', op: 'loop',
      weight: function () { return null; },
      note: 'The sampled id is appended to the sequence. The next forward pass has nt = 1 and a KV cache one cell longer -- same graph, different shapes.',
      math: function (c) { return 'nt 1, nKv ' + c.nKv + ' \u2192 ' + (c.nKv + 1) + ' next step'; },
      mk: function (c) { return [T([1], 'ids', 'token_ids')]; }
    }
  ];

  // Fill a template entry into a concrete stage for one context.
  function instantiate(tpl, ctx, profile) {
    var outs = tpl.mk(ctx);
    var fused = (tpl.key === 'layer.qkv' && ctx.fusedQkv) || (tpl.key === 'layer.ffn_gu' && ctx.fusedFfn);
    var w = tpl.weight ? tpl.weight(ctx) : null;
    // Quant types come from the export, keyed by exact weight name. A stage
    // that reads several weights (q/k/v, gate/up) lists them all.
    var keys = tpl.weightKeys ? tpl.weightKeys(ctx)
      : (w && w.indexOf('{') < 0 ? [w] : []);
    var seen = {}, types = [];
    keys.forEach(function (k) {
      var t = profile.wtype && profile.wtype[k];
      if (t && !seen[t]) { seen[t] = true; types.push(t); }
    });
    var wtype = types.length ? types.join(' / ') : null;
    var stage = {
      key: tpl.key,
      group: tpl.group,
      layerStage: tpl.layerStage,
      label: fused && tpl.fusedLabel ? tpl.fusedLabel : tpl.label,
      op: fused && tpl.fusedOp ? tpl.fusedOp : tpl.op,
      note: tpl.note,
      math: tpl.math ? tpl.math(ctx) : '',
      outputs: outs,
      weight: fused ? fusedWeight(w, ctx) : w,
      wtype: wtype,
      detail: tpl.detail ? tpl.detail(ctx) : '',
      internal: tpl.internal ? tpl.internal(ctx) : null,
      fused: !!fused,
      merged: !!(tpl.mergedNote && ctx.fusedQkv),
      phase: ctx.phase,
      step: ctx.step,
      layer: ctx.layer
    };
    if (stage.merged) stage.note = tpl.mergedNote;
    stage.headline = dimsSummary(outs);
    return stage;
  }

  function fusedWeight(w, ctx) {
    if (w && w.indexOf('{q,k,v}') >= 0) return 'blk.' + ctx.layer + '.attn_qkv';
    if (w && w.indexOf('{gate,up}') >= 0) return 'blk.' + ctx.layer + '.ffn_gu';
    return w;
  }

  function dimsSummary(outs) {
    if (!outs.length) return '';
    return outs.map(function (t) {
      return (outs.length > 1 ? (t.name + ' ') : '') + fmtShape(t.dims);
    }).join('  ');
  }

  // The stage list for one forward pass: prologue, one layer, epilogue.
  function script(profile, opts) {
    var c = ctxOf(profile, opts || {});
    var stages = [];
    for (var i = 0; i < TEMPLATE.length; i++) {
      var tpl = TEMPLATE[i];
      if (tpl.optional && tpl.when && !tpl.when(c)) continue;
      stages.push(instantiate(tpl, c, profile));
    }
    return stages;
  }

  function layerStageKeys(profile) {
    return TEMPLATE.filter(function (t) { return t.group === 'layer'; })
      .map(function (t) { return t.key; });
  }

  // ---------------------------------------------------------------- timeline
  // The playable script: prefill in full, an ×N layer tick, then decode steps
  // (compressed into one sweep per step unless `detailDecode` is set).
  function buildTimeline(profile, opts) {
    var o = opts || {};
    var modes = o.modes || ['prefill', 'decode'];
    var decodeSteps = o.decodeSteps == null ? 10 : o.decodeSteps;
    var detailDecode = !!o.detailDecode;
    var steps = [];
    var stageOf = {};

    function push(phase, layer, step, s) {
      stageOf[s.key] = s;
      steps.push({ kind: 'stage', key: s.key, phase: phase, layer: layer, step: step, stage: s });
    }

    if (modes.indexOf('prefill') >= 0) {
      var pre = script(profile, { phase: 'prefill', layer: 0 });
      // Play the prologue, then layer 0 in full; the remaining layers are the
      // `loop` tick below, and the epilogue closes the pass.
      pre.forEach(function (s) { if (s.group !== 'post') push('prefill', 0, 0, s); });
      if (profile.dims.nLayer > 1) {
        steps.push({
          kind: 'loop', key: 'loop', phase: 'prefill', layer: 0, step: 0,
          from: 1, to: profile.dims.nLayer - 1,
          label: '\u00d7' + (profile.dims.nLayer - 1) + ' more layers (same shapes, different weights)'
        });
      }
      var post = script(profile, { phase: 'prefill', layer: 0 });
      post.forEach(function (s) { if (s.group === 'post') push('prefill', 0, 0, s); });
    }

    if (modes.indexOf('decode') >= 0) {
      for (var st = 1; st <= decodeSteps; st++) {
        var sc = script(profile, { phase: 'decode', layer: 0, step: st });
        var byKey = {};
        sc.forEach(function (s) { byKey[s.key] = s; });
        push('decode', 0, st, byKey['pre.embed']);
        if (detailDecode) {
          sc.forEach(function (s) { if (s.group === 'layer') push('decode', 0, st, s); });
        } else {
          steps.push({
            kind: 'sweep', key: 'sweep', phase: 'decode', layer: 0, step: st,
            subs: sc.filter(function (s) { return s.group === 'layer'; }),
            label: '\u00d7' + profile.dims.nLayer + ' layers, nt=1'
          });
        }
        ['post.final_norm', 'post.lm_head', 'post.sample', 'post.next'].forEach(function (k) {
          if (byKey[k]) push('decode', 0, st, byKey[k]);
        });
      }
    }

    return { steps: steps, stageOf: stageOf, profile: profile };
  }

  // ---------------------------------------------------------------- built-in
  // Dims copied from viz/samples/qwen2.5-0.5b-prefill-metal.json so the page
  // animates before any file is loaded. Every number here is real; the page
  // labels it as the built-in profile until a graph is loaded.
  var BUILTIN = profileFromDims(
    { nEmbd: 896, nt: 7, vocab: 151936, nCtx: 4096, nLayer: 24, nHead: 14, nKvHead: 2, hd: 64, hdKv: 64, kvDim: 128, nFf: 4864 },
    {
      source: { kind: 'builtin', model: 'Qwen2.5-0.5B-Instruct Q4_K_M (built-in dims)', backend: 'metal/cpu', nodes: 440 },
      fused: { qkv: false, ffn: false },
      has: { qkNorm: false, kv: true, rope: true },
      wtype: {
        'token_embd.weight': 'q5_0', 'output.weight': 'q8_0',
        'blk.0.attn_q.weight': 'q5_0', 'blk.0.attn_k.weight': 'q5_0', 'blk.0.attn_v.weight': 'q8_0',
        'blk.0.attn_output.weight': 'q5_0', 'blk.0.ffn_gate.weight': 'q5_0',
        'blk.0.ffn_up.weight': 'q5_0', 'blk.0.ffn_down.weight': 'q6_K'
      },
      backends: ['metal', 'cpu'],
      prompt: { text: 'Hello!', nt: 7 },
      provenance: { dimsFromGraph: false, graphNt: null, graphNLayer: 24 }
    }
  );

  var API = {
    VERSION: VERSION,
    buildProfile: buildProfile,
    profileFromDims: profileFromDims,
    script: script,
    buildTimeline: buildTimeline,
    layerStageKeys: layerStageKeys,
    contextFor: contextFor,
    realDims: realDims,
    fmtShape: fmtShape,
    fmtCount: fmtCount,
    prod: prod,
    BUILTIN: BUILTIN,
    TEMPLATE: TEMPLATE,
    /** Visible for the headless renderer test. */
    _internal: { roleOf: roleOf, layerSlice: layerSlice, kvHeadsFrom: kvHeadsFrom }
  };

  root.E2EModel = API;
  if (typeof module !== 'undefined' && module.exports) module.exports = API;
})(typeof globalThis !== 'undefined' ? globalThis : this);
