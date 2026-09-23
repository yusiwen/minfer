/* minfer end-to-end shape flow -- canvas animation + UI.
 *
 * Plays the stage script produced by e2e-model.js: a tensor packet travels the
 * rail of nodes, morphing its shape at every step, while the side panel spells
 * out the shape math and the footer tracks the KV cache.
 *
 * The whole scene is one <canvas>; the panel, chips and KV bar are DOM.
 */
(function () {
  'use strict';

  var M = window.E2EModel;
  var MONO_PX = 'ui-monospace, Menlo, Consolas, monospace';
  var SANS_PX = '-apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC", sans-serif';

  // ---------------------------------------------------------------- strings

  // ------------------------------------------------------------------ utils
  function $(id) { return document.getElementById(id); }

  function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  function elide(ctx, text, maxW) {
    if (ctx.measureText(text).width <= maxW) return text;
    var s = text;
    while (s.length > 2 && ctx.measureText(s + '…').width > maxW) s = s.slice(0, -1);
    return s + '…';
  }

  function wrap(ctx, text, maxW, maxLines) {
    var words = String(text).split(/\s+/), lines = [], cur = '';
    for (var i = 0; i < words.length; i++) {
      var probe = cur ? cur + ' ' + words[i] : words[i];
      if (ctx.measureText(probe).width > maxW && cur) { lines.push(cur); cur = words[i]; }
      else cur = probe;
    }
    if (cur) lines.push(cur);
    if (lines.length > maxLines) {
      lines = lines.slice(0, maxLines);
      lines[maxLines - 1] = elide(ctx, lines[maxLines - 1] + ' …', maxW);
    }
    return lines;
  }

  function lerp(a, b, k) { return a + (b - a) * k; }
  function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
  function fmtShape(dims) { return '[' + dims.join(', ') + ']'; }
  function prod(dims) { var p = 1; for (var i = 0; i < dims.length; i++) p *= dims[i]; return p; }

  function pointOnPolyline(pts, k) {
    var total = 0, i, segLen = [];
    for (i = 0; i + 1 < pts.length; i++) {
      var d = Math.hypot(pts[i + 1].x - pts[i].x, pts[i + 1].y - pts[i].y);
      segLen.push(d); total += d;
    }
    if (total === 0) return { x: pts[0].x, y: pts[0].y };
    var want = clamp(k, 0, 1) * total;
    for (i = 0; i < segLen.length; i++) {
      if (want <= segLen[i] || i === segLen.length - 1) {
        var f = segLen[i] ? want / segLen[i] : 0;
        return {
          x: lerp(pts[i].x, pts[i + 1].x, f),
          y: lerp(pts[i].y, pts[i + 1].y, f)
        };
      }
      want -= segLen[i];
    }
    return pts[pts.length - 1];
  }

  var KIND_COLOR = {
    act: '#38bdf8', logits: '#f472b6', probs: '#c084fc', kv: '#22c55e',
    ids: '#fbbf24', scores: '#f472b6', vec: '#38bdf8', text: '#94a3b8'
  };
  function kindColor(k) { return KIND_COLOR[k] || '#38bdf8'; }

  // A tensor dims interpolator: shapes morph instead of snapping.
  function lerpTensor(a, b, k) {
    if (!a) return b;
    if (!b) return a;
    if (a.kind !== b.kind || a.dims.length !== b.dims.length) return k < 0.5 ? a : b;
    var dims = a.dims.map(function (d, i) { return Math.max(1, Math.round(lerp(d, b.dims[i], k))); });
    return { dims: dims, kind: k < 0.5 ? a.kind : b.kind, name: (k < 0.5 ? a : b).name, detail: b.detail };
  }

  // ------------------------------------------------------------------ state
  var S = {
    profile: M.BUILTIN,
    timeline: null,
    idx: 0,
    t: 0,
    playing: false,
    mode: 'run',
    withLayers: true,
    detailDecode: false,
    speed: 55,
    rail: [], railIndex: {},
    badges: {},
    prevKey: null,
    prevOuts: [],
    layout: null,
    dirty: true,
    decodeSteps: 10,
    emitted: 0,     // sampled tokens the playhead has passed
    emitT: 1        // 0..1 pop animation for the newest one
  };

  var cv, ctx, dpr = 1, W = 800, H = 600;

  // ------------------------------------------------------- profile / timeline
  function sampleLabel(p) {
    var kinds = { prefill: 'prefill', decode: 'decode', trace: 'trace', builtin: 'built-in' };
    var k = p.source.kind === 'graph' ? (p.source.nodes > 300 ? 'graph' : 'graph') : p.source.kind;
    return (p.source.model || '') + ' · ' + (kinds[k] || k) + ' · ' + p.source.nodes + ' nodes';
  }

  // Loading a graph never starts playback: the animation runs only after the
  // play button (or the restart button) is pressed.
  function setProfile(p, play) {
    S.profile = p;
    S.badges = {};
    S.prevKey = null;
    S.prevOuts = [];
    S.emitted = 0;
    S.emitT = 1;
    // A trace recorded one sample per forward pass: play exactly that many
    // decode steps, so the text strip lines up with the rail.
    S.decodeSteps = (p.traceTokens && p.traceTokens.length > 1)
      ? Math.min(p.traceTokens.length - 1, 12) : 10;
    rebuildRail();
    rebuildTimeline();
    S.idx = 0; S.t = 0;
    S.playing = play === true;
    onEnterStep(0);
    $('src').textContent = sampleLabel(p);
    $('src').title = sampleLabel(p);
    updateKV(S.timeline.steps[0]);
    S.dirty = true;
    syncPlayBtn();
  }

  function rebuildRail() {
    var stages = M.script(S.profile, { phase: 'prefill', layer: 0 });
    S.rail = stages.map(function (s) {
      return { key: s.key, label: s.label, op: s.op, group: s.group, layerStage: s.layerStage, outs: null };
    });
    S.railIndex = {};
    S.rail.forEach(function (n, i) { S.railIndex[n.key] = i; });
  }

  function rebuildTimeline(keepPos) {
    var modes = S.mode === 'run' ? ['prefill', 'decode'] : [S.mode];
    var tl = M.buildTimeline(S.profile, {
      modes: modes,
      decodeSteps: S.mode === 'prefill' ? 0 : S.decodeSteps,
      detailDecode: S.detailDecode
    });
    if (!S.withLayers) tl.steps = tl.steps.filter(function (s) { return s.kind !== 'loop'; });
    if (S.mode === 'decode') tl.steps = tl.steps.filter(function (s) { return s.phase === 'decode'; });
    S.timeline = tl;
    if (!keepPos) { S.idx = 0; S.t = 0; }
    else { S.idx = clamp(S.idx, 0, tl.steps.length - 1); S.t = 0; }
    buildChips();
    S.dirty = true;
  }

  // -------------------------------------------------------------- step logic
  function curStep() { return S.timeline.steps[S.idx]; }

  function durOf(step) {
    var base = 1150 - S.speed * 10;         // 1050 ms (slow) .. 150 ms (fast)
    if (step.kind === 'loop') return base * 3.2;
    if (step.kind === 'sweep') return base * 3.6;
    if (step.phase === 'decode') return base * 0.45;
    return base;
  }

  function prevStageOutputs(i) {
    for (var j = i - 1; j >= 0; j--) {
      var s = S.timeline.steps[j];
      if (s.kind === 'stage' && s.stage) return s.stage.outputs;
      if (s.kind === 'sweep' && s.subs && s.subs.length) return s.subs[s.subs.length - 1].outputs;
    }
    return [];
  }

  // The sampler stage is where a token leaves the model; counting the samples
  // the playhead has passed keeps the text strip and the rail in sync in both
  // directions (stepping back takes the token away again).
  function emittedCount(idx) {
    var n = 0;
    for (var i = 0; i <= idx && i < S.timeline.steps.length; i++) {
      var st = S.timeline.steps[i];
      if (st.kind === 'stage' && st.key === 'post.sample') n++;
    }
    return n;
  }

  function onEnterStep(i) {
    var step = S.timeline.steps[i];
    if (!step) return;
    var wasEmitted = S.emitted || 0;
    S.emitted = emittedCount(i);
    if (S.emitted > wasEmitted) S.emitT = 0;
    S.prevOuts = prevStageOutputs(i);
    if (step.kind === 'stage' && step.stage) {
      var prev = null;
      for (var j = i - 1; j >= 0; j--) {
        if (S.timeline.steps[j].kind === 'stage') { prev = S.timeline.steps[j]; break; }
      }
      S.prevKey = prev ? prev.key : null;
      S.badges[step.key] = { outs: step.stage.outputs, layer: step.layer, phase: step.phase };
      updateKV(step);
      updatePanel(step.stage, step, null);
    } else if (step.kind === 'sweep') {
      S.prevKey = null;
      updateKV(step);
      updatePanel(step.subs[0], step, 0);
    } else {
      updatePanel(loopPseudoStage(step), step, null);
    }
    setActiveChip(i);
  }

  function loopPseudoStage(step) {
    return {
      key: 'loop', label: '×N layers', op: 'repeat',
      note: 'Layers 2..' + S.profile.dims.nLayer + ' repeat this exact shape sequence; only the weights differ.',
      math: 'for layer in 2..' + S.profile.dims.nLayer + ':  [n_embd, nt] \u2192 [n_embd, nt]',
      outputs: [{ dims: [S.profile.dims.nEmbd, S.profile.dims.nt], kind: 'act' }],
      internal: null, detail: '', weight: null, wtype: null, phase: 'prefill', layer: step.from
    };
  }

  function advance(dt) {
    if (!S.playing || !S.timeline || !S.timeline.steps.length) return;
    var step = curStep();
    S.t += dt / durOf(step);
    var guard = 0;
    while (S.t >= 1 && guard++ < 8) {
      S.t -= 1;
      if (S.idx + 1 >= S.timeline.steps.length) {
        S.idx = S.timeline.steps.length - 1;
        S.t = 1;
        S.playing = false;
        syncPlayBtn();
        onEnterStep(S.idx);
        break;
      }
      S.idx++;
      onEnterStep(S.idx);
    }
  }

  function goto(i) {
    i = clamp(i, 0, S.timeline.steps.length - 1);
    S.idx = i; S.t = 0;
    onEnterStep(i);
  }

  function syncPlayBtn() {
    var b = $('play');
    b.textContent = S.playing ? '❚❚' : '▶';
    b.classList.toggle('on', S.playing);
    b.title = S.playing ? "pause" : "play";
  }

  // ---------------------------------------------------------------- KV / DOM
  function kvFor(step) {
    var prompt = S.profile.dims.nt || 1;
    var nCtx = S.profile.dims.nCtx || 1;
    if (step && step.phase === 'decode') {
      var written = prompt + Math.max(1, step.step);
      return { nKv: written, total: nCtx, pos: written - 1, nt: 1 };
    }
    return { nKv: prompt, total: nCtx, pos: 0, nt: prompt };
  }

  function updateKV(step) {
    var k = kvFor(step);
    S.kv = k;
    $('kvlabel').textContent = "KV cache" + '  ' + fmtShape([S.profile.dims.kvDim || 0, k.total]);
    $('kvnum').textContent = k.nKv + ' / ' + k.total + ' ' + "cells";
    $('kvfill').style.width = clamp(k.nKv / k.total * 100, 0, 100) + '%';
    var win = $('kvwin');
    var leftPct = clamp(k.pos / k.total * 100, 0, 100);
    win.style.left = leftPct + '%';
    win.style.width = Math.max(2, clamp(k.nt / k.total * 100, 0, 100 - leftPct)) + '%';
  }

  function buildChips() {
    var box = $('chips');
    box.innerHTML = '';
    var frag = document.createDocumentFragment();
    S.timeline.steps.forEach(function (s, i) {
      var chip = document.createElement('div');
      chip.className = 'chip';
      chip.dataset.i = String(i);
      var t0 = s.kind === 'stage' ? s.stage.outputs[0]
        : s.kind === 'sweep' ? s.subs[0].outputs[0]
          : { dims: [S.profile.dims.nEmbd, S.profile.dims.nt], kind: 'act' };
      var label = s.kind === 'loop' ? "×N layers"
        : s.kind === 'sweep' ? "decode sweep"
          : (s.stage.label);
      chip.innerHTML = '<span class="ck"></span><span class="cs"></span>';
      chip.querySelector('.ck').textContent = label;
      chip.querySelector('.cs').textContent = fmtShape(t0.dims);
      if (t0.kind === 'logits') chip.classList.add('logits');
      if (t0.kind === 'kv') chip.classList.add('kv');
      if (t0.kind === 'ids') chip.classList.add('ids');
      if (s.kind !== 'stage') chip.classList.add('special');
      if (s.phase === 'decode') chip.title = "decode step" + ' ' + s.step;
      frag.appendChild(chip);
    });
    box.appendChild(frag);
  }

  function setActiveChip(i) {
    var box = $('chips');
    var prev = box.querySelector('.chip.active');
    if (prev) prev.classList.remove('active');
    var el = box.querySelector('.chip[data-i="' + i + '"]');
    if (el) {
      el.classList.add('active');
      var left = el.offsetLeft - box.clientWidth / 2 + el.offsetWidth / 2;
      box.scrollTo({ left: Math.max(0, left), behavior: 'smooth' });
    }
  }

  // ------------------------------------------------------------ side panel
  function shapeChips(outs, cls) {
    return outs.map(function (o) {
      return '<span class="shape ' + cls + ' k-' + o.kind + '">' + fmtShape(o.dims) + '</span>';
    }).join('');
  }

  function updatePanel(stage, step, subIdx) {
    var d = S.profile.dims;
    var head = '<div class="p-head"><div class="p-title">' + (stage.label) +
      ' <span class="p-op">' + stage.op + '</span></div>' +
      '<div class="p-sub">' + phaseText(step, subIdx) + '</div></div>';

    var ins = S.prevOuts && S.prevOuts.length ? S.prevOuts : null;
    var shapes = '<div class="p-shapes">' +
      '<div class="col"><span class="dim">' + "in" + '</span>' +
      (ins ? shapeChips(ins, 'in') : '<span class="shape in">—</span>') + '</div>' +
      '<span class="arrow">→</span>' +
      '<div class="col"><span class="dim">' + "out" + '</span>' + shapeChips(stage.outputs, 'out') + '</div>' +
      '</div>';

    var math = stage.math ? '<div class="p-math">' + escapeHtml(stage.math) + '</div>' : '';
    var note = stage.note ? '<div class="p-note">' + escapeHtml((stage.note)) + '</div>' : '';

    var rows = '';
    function row(k, v) { if (v) rows += '<tr><td class="k">' + k + '</td><td class="v">' + escapeHtml(v) + '</td></tr>'; }
    var o0 = stage.outputs[0];
    row("output", stage.outputs.map(function (o) { return (o.name ? o.name + ' ' : '') + fmtShape(o.dims); }).join('  '));
    row("elements", M.fmtCount(prod(o0.dims)) + (stage.outputs.length > 1 ? ' (+ more)' : ''));
    row("weight", stage.weight || '—');
    row("quant", stage.wtype || (stage.weight ? "mixed" : '—'));
    row("backends", (S.profile.backends || []).join(' + ') || 'n/a');
    var tbl = '<table class="kv">' + rows + '</table>';

    var badges = '';
    if (stage.fused) badges += '<span class="badge fused">fused node</span>';
    if (stage.merged) badges += '<span class="badge">RoPE inside the fused node</span>';
    if (S.profile.lastTokenOnly && stage.key === 'post.final_norm') {
      badges += '<span class="badge warn">' + "the last layer runs at nt=1: only the last position needs logits" + '</span>';
    }
    if (S.profile.weights && S.profile.weights.tied && stage.key === 'post.lm_head') {
      badges += '<span class="badge">tied to the embedding</span>';
    }
    if ((stage.fused || stage.merged) && S.profile.provenance) {
      var fl = S.profile.provenance.fusedLayers || { qkv: [], ffn: [] };
      var n = Math.max(fl.qkv.length, fl.ffn.length);
      if (n && n < d.nLayer) badges += '<span class="badge">' + n + '/' + d.nLayer + ' layers fused this way</span>';
    }
    badges = badges ? '<div>' + badges + '</div>' : '';

    var internals = '';
    if (stage.internal && stage.internal.length) {
      internals = '<div class="p-sec">' + "inside this node" + '</div><table class="internal">' +
        stage.internal.map(function (x) {
          return '<tr' + (x.hero ? ' class="hero"' : '') + '><td class="k">' + (x.label) +
            '</td><td class="v">' + fmtShape(x.dims) + '</td></tr>';
        }).join('') + '</table>';
    }

    var src = S.profile.provenance && S.profile.provenance.dimsFromGraph
      ? "dims read from the loaded graph export" : "built-in Qwen2.5-0.5B dims (no graph loaded)";

    $('panel-body').innerHTML = head + shapes + math + note + badges + tbl + internals +
      '<div class="p-src">' + "source" + ': ' + escapeHtml(sampleLabel(S.profile)) + '<br>' + src + '</div>';
  }

  function phaseText(step, subIdx) {
    var d = S.profile.dims;
    var parts = [step.phase === 'decode' ? "decode" : "prefill"];
    if (step.phase === 'decode') {
      parts.push("decode step" + ' ' + step.step + '/' + S.decodeSteps);
      parts.push("KV cache" + ' ' + (d.nt + step.step) + '/' + d.nCtx);
    } else {
      parts.push("layer" + ' ' + (step.layer + 1) + '/' + d.nLayer);
    }
    if (subIdx != null && step.subs) {
      parts.push((step.subs[subIdx].label) + ' ' + fmtShape(step.subs[subIdx].outputs[0].dims));
    }
    parts.push("step" + ' ' + (S.idx + 1) + '/' + S.timeline.steps.length);
    return parts.join(' · ');
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>"]/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
    });
  }

  // ------------------------------------------------------------------ layout
  function layout() {
    var HEADER = 178;
    var keys = S.rail;
    if (!keys.length) return;
    // Reserved band for the text strip: prompt + the tokens sampled so far.
    var textRect = { x: 14, y: 170, w: W - 28, h: 54 };
    var railTop = textRect.y + textRect.h + 14, railBottom = H - 40;
    // One reserved line under the rail top holds the ×N-layers bracket label, so
    // it can never land on the text strip above.
    var gridTop = railTop + 22;
    var cols = clamp(Math.floor((W - 56) / 192), 3, 8);
    var rows = Math.ceil(keys.length / cols);
    var boxW = 158, boxH = 62, gapY = 76;
    // Spread the columns over the available width: a wider gap gives the
    // travelling packet room, so it does not have to sit on a node.
    var gapX = clamp(Math.floor((W - 40 - cols * boxW) / Math.max(1, cols - 1)), 30, 84);
    var gridW = cols * boxW + (cols - 1) * gapX;
    // Squeeze the vertical pitch when the grid would not fit: a short window
    // must not push the last row off the canvas.
    var availH = Math.max(120, railBottom - gridTop);
    var pitch = boxH + gapY;
    if (rows > 1 && rows * boxH + (rows - 1) * gapY > availH) {
      pitch = clamp((availH - boxH) / (rows - 1), boxH + 12, boxH + gapY);
    }
    var gapYEff = pitch - boxH;
    var gridH = rows * boxH + (rows - 1) * gapYEff;
    var x0 = Math.max(20, (W - gridW) / 2);
    var y0 = gridTop + Math.max(0, (railBottom - gridTop - gridH) / 2);

    var nodes = keys.map(function (n, i) {
      var row = Math.floor(i / cols);
      var col = row % 2 === 0 ? i % cols : cols - 1 - (i % cols);
      return {
        key: n.key, label: n.label, group: n.group, layerStage: n.layerStage,
        row: row, col: col,
        x: x0 + col * (boxW + gapX), y: y0 + row * pitch,
        w: boxW, h: boxH
      };
    });
    nodes.forEach(function (n) { n.cx = n.x + n.w / 2; n.cy = n.y + n.h / 2; });

    // Bracket around the repeated layer block.
    var bracket = null;
    var idxs = [];
    nodes.forEach(function (n, i) { if (n.group === 'layer') idxs.push(i); });
    if (idxs.length) {
      var minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
      idxs.forEach(function (i) {
        var n = nodes[i];
        minX = Math.min(minX, n.x); minY = Math.min(minY, n.y);
        maxX = Math.max(maxX, n.x + n.w); maxY = Math.max(maxY, n.y + n.h);
      });
      // Keep the bracket (and its label) clear of the header band.
      var bTop = Math.max(minY - 22, railTop + 6);
      bracket = { x: minX - 12, y: bTop, w: maxX - minX + 24, h: maxY - bTop + 34 };
    }

    S.layout = { nodes: nodes, cols: cols, rows: rows, bracket: bracket, header: HEADER, text: textRect };
    // Badge list per rail node keeps the drawing loop cheap.
    nodes.forEach(function (n) { n.outs = (S.badges[n.key] && S.badges[n.key].outs) || null; });
    S.dirty = false;
  }

  function syncBadges() {
    if (!S.layout) return;
    S.layout.nodes.forEach(function (n) { n.outs = (S.badges[n.key] && S.badges[n.key].outs) || null; });
  }

  // -------------------------------------------------------------- rail paths
  // `inset` keeps a travelling packet off the boxes themselves: it leaves the
  // source and arrives at the destination a little way out, so the card never
  // covers a node label.
  // The packet stops half a gap short of the next box, derived from the actual
  // gap (a fixed inset larger than the gap would fold the path back on itself).
  function segPoints(a, b, shrink) {
    if (a.row === b.row) {
      var gap = Math.abs(b.x - a.x) - a.w;
      var inset = shrink ? Math.min(insetFor(gap), gap / 2) : 0;
      if (b.x > a.x) return [{ x: a.x + a.w + inset, y: a.cy }, { x: b.x - inset, y: b.cy }];
      return [{ x: a.x - inset, y: a.cy }, { x: b.x + b.w + inset, y: b.cy }];
    }
    var vgap = b.y - (a.y + a.h);
    var vinset = shrink ? Math.min(insetFor(vgap), vgap / 2) : 0;
    var from = { x: a.cx, y: a.y + a.h + vinset }, to = { x: b.cx, y: b.y - vinset };
    var mid = (from.y + to.y) / 2;
    return [from, { x: from.x, y: mid }, { x: to.x, y: mid }, to];
  }

  function insetFor(gap) { return clamp(gap / 2 - 3, 1, 22); }

  function pathBetween(fromIdx, toIdx, shrink) {
    var nodes = S.layout.nodes;
    if (fromIdx == null || toIdx == null || fromIdx === toIdx) return null;
    var pts = [];
    if (fromIdx < toIdx) {
      for (var i = fromIdx; i < toIdx; i++) {
        var sp = segPoints(nodes[i], nodes[i + 1], shrink);
        if (!pts.length) pts.push(sp[0]);
        pts = pts.concat(sp.slice(1));
      }
      return pts;
    }
    // The autoregressive jump back to the embedding: a dashed arc below/above.
    var a = nodes[fromIdx], b = nodes[toIdx];
    return [
      { x: a.cx + 22, y: a.cy },
      { x: a.cx + 22, y: (a.cy + b.cy) / 2 },
      { x: b.cx, y: (a.cy + b.cy) / 2 },
      { x: b.cx, y: b.cy - 22 }
    ];
  }

  // ------------------------------------------------------------ tensor cards
  function cardHeight(ft, maxFt) {
    var lf = Math.log2(Math.max(2, ft)), lm = Math.log2(Math.max(2, maxFt));
    return 20 + 88 * (lf / lm);
  }

  function drawTensor(ctx2, cx, cy, tn, opts) {
    opts = opts || {};
    var col = kindColor(tn.kind);
    var scale = opts.scale == null ? 1 : opts.scale;
    var alpha = opts.alpha == null ? 1 : opts.alpha;
    var maxFt = opts.maxFt || S.profile.dims.vocab || 151936;
    ctx2.save();
    ctx2.globalAlpha = alpha;

    if (tn.kind === 'kv') {
      var w = 116 * scale, h = 15 * scale;
      ctx2.fillStyle = 'rgba(34,197,94,.18)';
      roundRect(ctx2, cx - w / 2, cy - h / 2, w, h, 3); ctx2.fill();
      ctx2.strokeStyle = col; ctx2.lineWidth = 1;
      roundRect(ctx2, cx - w / 2, cy - h / 2, w, h, 3); ctx2.stroke();
      var kv = S.kv || { nKv: 0, total: 1, pos: 0, nt: 1 };
      var fx = clamp(kv.nKv / kv.total, 0, 1);
      ctx2.fillStyle = 'rgba(34,197,94,.45)';
      ctx2.fillRect(cx - w / 2, cy - h / 2, Math.max(1.5, w * fx), h);
      var wx = cx - w / 2 + w * clamp(kv.pos / kv.total, 0, 1);
      ctx2.fillStyle = '#fbbf24';
      ctx2.fillRect(wx, cy - h / 2 - 2, Math.max(2, w * (kv.nt / kv.total)), h + 4);
      ctx2.restore();
      return { w: w, h: h };
    }

    if (tn.kind === 'ids') {
      var n = clamp(tn.dims[0], 1, 10);
      var s = 11 * scale;
      var total = n * s + (n - 1) * 3;
      for (var i = 0; i < n; i++) {
        ctx2.fillStyle = col;
        ctx2.globalAlpha = alpha * (0.35 + 0.65 * (i + 1) / n);
        roundRect(ctx2, cx - total / 2 + i * (s + 3), cy - s / 2, s, s, 2);
        ctx2.fill();
      }
      ctx2.globalAlpha = alpha;
      if (tn.dims[0] > n) {
        ctx2.fillStyle = col; ctx2.font = '11px ' + MONO_PX;
        ctx2.textAlign = 'center'; ctx2.textBaseline = 'middle';
        ctx2.fillText('…', cx + total / 2 + 7, cy);
      }
      ctx2.restore();
      return { w: total, h: s };
    }

    if (tn.kind === 'scores') {
      var heads = clamp(tn.dims[0], 1, 3);
      var nt = clamp(tn.dims[1], 1, 6), nk = clamp(tn.dims[2], 1, 20);
      var cell = 4 * scale, pw = nk * cell, ph = nt * cell, gap2 = 4;
      var tw = heads * pw + (heads - 1) * gap2;
      for (var hi = 0; hi < heads; hi++) {
        var hx = cx - tw / 2 + hi * (pw + gap2);
        for (var r = 0; r < nt; r++) {
          for (var c2 = 0; c2 <= r; c2++) {
            var lit = c2 <= r;
            ctx2.fillStyle = lit ? col : 'rgba(148,163,184,.18)';
            ctx2.globalAlpha = alpha * (lit ? 0.25 + 0.75 * (c2 + 1) / Math.max(1, nt) : 1);
            ctx2.fillRect(hx + (nk - nt + c2) * cell, cy - ph / 2 + r * cell, cell - 1, cell - 1);
          }
        }
      }
      ctx2.globalAlpha = alpha;
      ctx2.restore();
      return { w: tw, h: ph };
    }

    // act / logits / text: nt vertical bars whose height shows the feature dim
    var ft = tn.dims[0];
    var ntv = tn.dims.length > 1 ? tn.dims[1] : 1;
    var h2 = cardHeight(ft, maxFt) * scale;
    var nCols = clamp(ntv, 1, 6);
    var barW = 7 * scale, gap3 = 3 * scale;
    var wTot = nCols * barW + (nCols - 1) * gap3;
    for (var k = 0; k < nCols; k++) {
      ctx2.fillStyle = col;
      ctx2.globalAlpha = alpha * (0.4 + 0.6 * (k + 1) / nCols);
      roundRect(ctx2, cx - wTot / 2 + k * (barW + gap3), cy - h2 / 2, barW, h2, 2);
      ctx2.fill();
    }
    ctx2.globalAlpha = alpha;
    if (ntv > nCols) {
      ctx2.fillStyle = col; ctx2.font = '11px ' + MONO_PX;
      ctx2.textAlign = 'center'; ctx2.textBaseline = 'middle';
      ctx2.fillText('…', cx + wTot / 2 + 7, cy);
    }
    ctx2.restore();
    return { w: wTot, h: h2 };
  }

  // ------------------------------------------------------------ scene drawing
  function renderHeader() {
    var step = curStep();
    var stage = step.kind === 'stage' ? step.stage
      : step.kind === 'sweep' ? step.subs[sweepSubIdx(step)]
        : loopPseudoStage(step);
    var d = S.profile.dims;
    var maxFt = Math.max(d.vocab || 0, d.nFf || 0, d.nEmbd || 0, 2);

    // --- NOW card
    var nw = 320;
    ctx.fillStyle = '#1e293b';
    roundRect(ctx, 14, 14, nw, 148, 8); ctx.fill();
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    roundRect(ctx, 14, 14, nw, 148, 8); ctx.stroke();

    ctx.fillStyle = '#64748b';
    ctx.font = '10px ' + SANS_PX;
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText("tensor at this node".toUpperCase(), 28, 36);

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 22px ' + MONO_PX;
    var out0 = stage.outputs[0];
    ctx.fillText(fmtShape(out0.dims), 28, 66);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px ' + MONO_PX;
    ctx.fillText(M.fmtCount(prod(out0.dims)) + ' ' + "elements" + ' · ' +
      M.fmtCount(prod(out0.dims) * 4) + ' B', 28, 84);

    var outs = stage.outputs;
    if (outs.length > 1) {
      ctx.font = '10px ' + MONO_PX;
      outs.slice(0, 3).forEach(function (o, i3) {
        ctx.fillStyle = kindColor(o.kind);
        ctx.fillText((o.name ? o.name + ' ' : '') + fmtShape(o.dims), 28, 106 + i3 * 14);
      });
    } else if (out0.detail) {
      ctx.fillStyle = '#64748b';
      ctx.font = '10.5px ' + SANS_PX;
      ctx.fillText(elide(ctx, out0.detail, nw - 40), 28, 106);
    }

    // The card visual: one slot per output, morphing from the previous shape.
    var slots = Math.min(outs.length, 3);
    var prev = S.prevOuts || [];
    var morph = step.kind === 'stage' ? clamp(S.t / 0.5, 0, 1) : 1;
    var areaCx = 14 + nw - (slots > 1 ? 74 : 62);
    var slotStep = slots > 1 ? 50 : 0;
    for (var i = 0; i < slots; i++) {
      var from = prev.length ? prev[Math.min(i, prev.length - 1)] : null;
      var tn = lerpTensor(from, outs[i], morph);
      drawTensor(ctx, areaCx + (i - (slots - 1) / 2) * slotStep, 96, tn,
        { maxFt: maxFt, scale: slots > 1 ? 0.6 : 0.95 });
    }

    // --- stage info panel
    var sx = 14 + nw + 14;
    var sw = Math.max(240, W - sx - 232);
    ctx.fillStyle = 'rgba(30,41,59,.65)';
    roundRect(ctx, sx, 14, sw, 148, 8); ctx.fill();
    ctx.strokeStyle = '#334155'; roundRect(ctx, sx, 14, sw, 148, 8); ctx.stroke();

    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 15px ' + SANS_PX;
    ctx.textAlign = 'left';
    ctx.fillText(elide(ctx, (stage.label), sw - 24), sx + 14, 40);

    ctx.fillStyle = '#38bdf8';
    ctx.font = '11px ' + MONO_PX;
    ctx.fillText(elide(ctx, stage.op + (stage.wtype ? '  ·  ' + stage.wtype : ''), sw - 24), sx + 14, 58);

    if (stage.math) {
      ctx.fillStyle = '#a5f3fc';
      ctx.font = '12px ' + MONO_PX;
      var ml = wrap(ctx, stage.math, sw - 28, 2);
      ml.forEach(function (line, i2) { ctx.fillText(line, sx + 14, 78 + i2 * 15); });
    }
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11.5px ' + SANS_PX;
    var note = stage.note ? (stage.note) : '';
    var nl = wrap(ctx, note, sw - 28, 2);
    nl.forEach(function (line, i2) { ctx.fillText(line, sx + 14, 114 + i2 * 15); });

    // --- counters
    var cxx = W - 218, cw = 204;
    ctx.fillStyle = '#1e293b';
    roundRect(ctx, cxx, 14, cw, 148, 8); ctx.fill();
    ctx.strokeStyle = '#334155'; roundRect(ctx, cxx, 14, cw, 148, 8); ctx.stroke();

    var kv = S.kv || kvFor(step);
    var layerNow;
    if (step.kind === 'loop') layerNow = (1 + Math.round(S.t * (step.to - step.from))) + ' / ' + d.nLayer;
    else if (step.phase === 'decode') layerNow = '× ' + d.nLayer;
    else layerNow = (stage.layer != null ? stage.layer + 1 : 1) + ' / ' + d.nLayer;
    var rows = [
      ["phase", step.phase === 'decode' ? "decode" : "prefill"],
      ["layer", layerNow],
      ["decode step", step.phase === 'decode' ? String(step.step) : '—'],
      ["KV cache", kv.nKv + ' / ' + kv.total],
      ["step", (S.idx + 1) + ' / ' + S.timeline.steps.length]
    ];
    ctx.font = '11px ' + SANS_PX;
    rows.forEach(function (r, i2) {
      var y = 36 + i2 * 24;
      ctx.fillStyle = '#64748b'; ctx.textAlign = 'left';
      ctx.fillText(r[0], cxx + 12, y);
      ctx.fillStyle = i2 === 3 ? '#22c55e' : '#e2e8f0';
      ctx.font = '600 12px ' + MONO_PX;
      ctx.textAlign = 'right';
      ctx.fillText(r[1], cxx + cw - 12, y);
      ctx.textAlign = 'left';
      ctx.font = '11px ' + SANS_PX;
    });
  }

  // ---- the text strip: prompt in, one token out per sampler visit ---------
  function renderTextStrip() {
    var r = S.layout && S.layout.text;
    if (!r) return;
    var tt = S.profile.traceTokens;
    var n = S.emitted || 0;

    ctx.save();
    ctx.fillStyle = 'rgba(30,41,59,.5)';
    roundRect(ctx, r.x, r.y, r.w, r.h, 8); ctx.fill();
    ctx.strokeStyle = '#334155'; ctx.lineWidth = 1;
    roundRect(ctx, r.x, r.y, r.w, r.h, 8); ctx.stroke();

    ctx.font = '9.5px ' + SANS_PX;
    ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillStyle = '#64748b';
    ctx.fillText("prompt → generated text".toUpperCase(), r.x + 12, r.y + 15);

    ctx.textAlign = 'right';
    ctx.fillStyle = tt ? '#64748b' : '#fbbf24';
    ctx.fillText(tt ? (n + ' ' + "tokens") : "token text needs a MINFER_TRACE export", r.x + r.w - 12, r.y + 15);

    var cy = r.y + r.h - 21;
    ctx.font = '600 12px ' + MONO_PX;
    var prompt = S.profile.prompt.text || '';
    var promptW = Math.min(ctx.measureText(prompt).width + 18, 240);

    // One chip per sampler visit. A trace carries the text of every token but
    // the very last one (there it only recorded the id and its probability).
    var items = [];
    for (var k = 0; k < n; k++) {
      var tok = tt && tt[k] ? tt[k] : null;
      if (tok && tok.text != null) items.push({ label: '"' + tok.text + '"', known: true });
      else if (tok) items.push({ label: '#' + tok.token, known: 'id' });
      else items.push({ label: '\u27e8tok\u27e9', known: false });
    }
    var widths = items.map(function (it) { return ctx.measureText(it.label).width + 18; });

    // Keep the newest tokens that fit; older ones collapse into a leading "…".
    var budget = r.w - 24 - promptW - 34;
    var sum = 0, first = items.length;
    for (var i2 = items.length - 1; i2 >= 0; i2--) {
      var wI = widths[i2] + 6;
      if (i2 < items.length - 1 && sum + wI > budget) break;
      sum += wI;
      first = i2;
    }
    var dropped = first > 0;

    var x = r.x + 12;
    x += drawStripChip(x, cy, promptW, prompt,
      { fill: 'rgba(56,189,248,.10)', stroke: 'rgba(56,189,248,.4)', color: '#bae6fd' });
    x += 8;
    ctx.fillStyle = '#475569';
    ctx.font = '12px ' + SANS_PX;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText('\u2192', x + 6, cy + 0.5);
    x += 20;
    if (dropped) {
      ctx.fillStyle = '#64748b';
      ctx.font = '11px ' + MONO_PX;
      ctx.fillText('\u2026', x + 4, cy + 0.5);
      x += 16;
    }
    for (var j = first; j < items.length; j++) {
      var newest = j === items.length - 1;
      var pop = newest ? clamp(S.emitT == null ? 1 : S.emitT, 0, 1) : 1;
      var scale = 0.72 + 0.28 * (1 - Math.pow(1 - pop, 3));
      var it = items[j];
      var style = it.known === true
        ? { fill: 'rgba(244,114,182,.13)', stroke: 'rgba(244,114,182,.45)', color: '#fbcfe8' }
        : it.known === 'id'
          ? { fill: 'rgba(100,116,139,.14)', stroke: 'rgba(148,163,184,.45)', color: '#cbd5e1' }
          : { fill: 'rgba(100,116,139,.10)', stroke: 'rgba(148,163,184,.3)', color: '#94a3b8' };
      if (newest && pop < 1) { style.glow = style.stroke; style.blur = 14 * (1 - pop); }
      drawStripChip(x, cy, widths[j], it.label, style, scale);
      x += widths[j] + 6;
    }
    if (S.playing) {                       // caret: the engine is running
      ctx.globalAlpha = 0.25 + 0.6 * Math.abs(Math.sin(Date.now() / 380));
      ctx.fillStyle = '#f472b6';
      ctx.fillRect(x + 1, cy - 9, 2.5, 18);
      ctx.globalAlpha = 1;
    }
    ctx.restore();
  }

  function drawStripChip(left, cy, w, label, style, scale) {
    var h = 22, y = cy - h / 2;
    ctx.save();
    if (scale && scale !== 1) {
      ctx.translate(left + w / 2, cy);
      ctx.scale(scale, scale);
      ctx.translate(-(left + w / 2), -cy);
    }
    if (style.glow) { ctx.shadowColor = style.glow; ctx.shadowBlur = style.blur || 10; }
    roundRect(ctx, left, y, w, h, 6);
    ctx.fillStyle = style.fill; ctx.fill();
    ctx.strokeStyle = style.stroke; ctx.lineWidth = 1.2;
    roundRect(ctx, left, y, w, h, 6); ctx.stroke();
    ctx.shadowBlur = 0;
    ctx.fillStyle = style.color;
    ctx.font = '600 12px ' + MONO_PX;
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(elide(ctx, label, w - 10), left + w / 2, cy + 0.5);
    ctx.restore();
    return w;
  }

  function sweepSubIdx(step) {
    return clamp(Math.floor(S.t * step.subs.length), 0, step.subs.length - 1);
  }

  function renderRail() {
    var L = S.layout;
    if (!L) return;

    // bracket
    if (L.bracket) {
      var b = L.bracket;
      ctx.save();
      ctx.fillStyle = 'rgba(56,189,248,.045)';
      roundRect(ctx, b.x, b.y, b.w, b.h, 10); ctx.fill();
      ctx.setLineDash([6, 5]);
      ctx.strokeStyle = 'rgba(56,189,248,.35)'; ctx.lineWidth = 1;
      roundRect(ctx, b.x, b.y, b.w, b.h, 10); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = 'rgba(56,189,248,.85)';
      ctx.font = '600 11px ' + MONO_PX;
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      var fl = S.profile.provenance && S.profile.provenance.fusedLayers;
      var extra = '';
      if (fl) {
        var nq = fl.qkv.length, nf = fl.ffn.length, n = S.profile.dims.nLayer;
        if (nq && nq < n) extra = '  (qkv fused: ' + nq + '/' + n + ')';
      }
      ctx.fillText('× ' + S.profile.dims.nLayer + ' ' + "layer" + extra, b.x + 6, b.y - 6);
      ctx.restore();
    }

    // wire
    ctx.save();
    ctx.strokeStyle = 'rgba(51,65,85,.9)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (var i = 0; i + 1 < L.nodes.length; i++) {
      var pts = segPoints(L.nodes[i], L.nodes[i + 1]);
      for (var j = 0; j + 1 < pts.length; j++) {
        ctx.moveTo(pts[j].x, pts[j].y);
        ctx.lineTo(pts[j + 1].x, pts[j + 1].y);
      }
    }
    ctx.stroke();
    ctx.restore();

    var step = curStep();
    var activeKey = step.kind === 'stage' ? step.key
      : step.kind === 'sweep' ? step.subs[sweepSubIdx(step)].key : null;

    // nodes
    L.nodes.forEach(function (n) {
      var isActive = n.key === activeKey;
      var isLayerNode = n.group === 'layer';
      var base = n.group === 'pre' ? '#7dd3fc' : n.group === 'post' ? '#f9a8d4' : '#38bdf8';
      ctx.save();
      ctx.fillStyle = isActive ? 'rgba(56,189,248,.16)' : '#1e293b';
      roundRect(ctx, n.x, n.y, n.w, n.h, 7); ctx.fill();
      if (isActive) {
        ctx.shadowColor = 'rgba(56,189,248,.65)'; ctx.shadowBlur = 14;
      }
      ctx.strokeStyle = isActive ? base : (isLayerNode ? 'rgba(56,189,248,.3)' : 'rgba(148,163,184,.3)');
      ctx.lineWidth = isActive ? 2 : 1;
      roundRect(ctx, n.x, n.y, n.w, n.h, 7); ctx.stroke();
      ctx.restore();

      ctx.save();
      ctx.fillStyle = isActive ? '#f1f5f9' : '#cbd5e1';
      ctx.font = (isActive ? '600 ' : '') + '11px ' + SANS_PX;
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
      ctx.fillText(elide(ctx, (n.label), n.w - 16), n.x + 8, n.y + 16);

      ctx.fillStyle = '#64748b';
      ctx.font = '9px ' + MONO_PX;
      var opTxt = stageOpFor(n.key);
      ctx.fillText(elide(ctx, opTxt, n.w - 16), n.x + 8, n.y + 27);

      // shape badges: the trail of shapes the packets left behind
      if (n.outs) {
        var one = n.outs.length === 1;
        var first = (one ? '' : (n.outs[0].name ? n.outs[0].name + ' ' : '')) + fmtShape(n.outs[0].dims);
        ctx.fillStyle = kindColor(n.outs[0].kind);
        ctx.font = '10px ' + MONO_PX;
        ctx.textAlign = 'right';
        ctx.fillText(elide(ctx, first, n.w - 16), n.x + n.w - 8, one ? n.y + 46 : n.y + 44);
        if (!one) {
          var rest = n.outs.slice(1).map(function (o) {
            return (o.name ? o.name + ' ' : '') + fmtShape(o.dims);
          }).join(' ');
          ctx.fillStyle = 'rgba(148,163,184,.85)';
          ctx.font = '9.5px ' + MONO_PX;
          ctx.fillText(elide(ctx, rest, n.w - 16), n.x + n.w - 8, n.y + 57);
        }
      }
      ctx.restore();
    });

    // the repeated-layer sweep highlight
    if (step.kind === 'loop' || step.kind === 'sweep') {
      drawSweepBand(step);
    }
  }

  function stageOpFor(key) {
    for (var i = 0; i < S.rail.length; i++) if (S.rail[i].key === key) return S.rail[i].op || '';
    return '';
  }

  function layerNodeRects() {
    return S.layout.nodes.filter(function (n) { return n.group === 'layer'; });
  }

  function drawSweepBand(step) {
    var rects = layerNodeRects();
    if (!rects.length) return;
    var minX = Math.min.apply(null, rects.map(function (r) { return r.x; }));
    var maxX = Math.max.apply(null, rects.map(function (r) { return r.x + r.w; }));
    var minY = Math.min.apply(null, rects.map(function (r) { return r.y; }));
    var maxY = Math.max.apply(null, rects.map(function (r) { return r.y + r.h; }));
    var prog = clamp(S.t, 0, 1);
    var bandW = 70;
    var x = minX - bandW + prog * (maxX - minX + bandW * 2);
    ctx.save();
    ctx.beginPath();
    ctx.rect(minX - 8, minY - 8, maxX - minX + 16, maxY - minY + 16);
    ctx.clip();
    var grad = ctx.createLinearGradient(x - bandW, 0, x + bandW, 0);
    grad.addColorStop(0, 'rgba(56,189,248,0)');
    grad.addColorStop(0.5, 'rgba(56,189,248,.16)');
    grad.addColorStop(1, 'rgba(56,189,248,0)');
    ctx.fillStyle = grad;
    ctx.fillRect(x - bandW, minY - 8, bandW * 2, maxY - minY + 16);
    ctx.restore();

    // cycle the badges through the layer stages
    if (step.kind === 'sweep') {
      var sub = step.subs[sweepSubIdx(step)];
      S.layout.nodes.forEach(function (n) { if (n.key === sub.key) n.outs = sub.outputs; });
    }
    ctx.save();
    ctx.fillStyle = '#f472b6';
    ctx.font = '600 11px ' + MONO_PX;
    ctx.textAlign = 'right';
    var layerNum = 1 + Math.round(prog * (S.profile.dims.nLayer - 1));
    ctx.fillText(step.label + '   ' + "layer" + ' ' + layerNum + '/' + S.profile.dims.nLayer,
      maxX + 6, maxY + 14);
    ctx.restore();
  }

  // The travelling packet is one small fixed-height chip carrying the step's
  // primary shape; extra outputs collapse to "+n" (the panel has the detail).
  // Its width is clamped to the gap it travels through, so it never sits on a
  // node label, and it fades as it lands.
  function drawPacketChip(cx, cy, stage, alpha, maxW) {
    var outs = stage.outputs;
    // One shape only: it fits the hop gap, so it is never elided. Extra outputs
    // are on the node's own badge and in the side panel.
    var text = fmtShape(outs[0].dims);
    var col = kindColor(outs[0].kind);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.font = '600 10.5px ' + MONO_PX;
    var w = Math.min(ctx.measureText(text).width + 16, Math.max(40, maxW));
    var h = 22;
    roundRect(ctx, cx - w / 2, cy - h / 2, w, h, 6);
    ctx.fillStyle = 'rgba(15,23,42,.96)';
    ctx.fill();
    ctx.strokeStyle = col;
    ctx.lineWidth = 1.5;
    roundRect(ctx, cx - w / 2, cy - h / 2, w, h, 6);
    ctx.stroke();
    ctx.fillStyle = col;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(elide(ctx, text, w - 10), cx, cy + 0.5);
    ctx.restore();
  }

  function renderPacket() {
    var step = curStep();
    var L = S.layout;
    if (!L || step.kind !== 'stage' || !step.stage) return;
    var to = S.railIndex[step.key];
    var from = S.prevKey != null ? S.railIndex[S.prevKey] : null;
    var node = L.nodes[to];
    var p = clamp(S.t / 0.62, 0, 1);
    var pos, alpha = 1, maxW = 134;
    if (from == null || from === to) {
      // First step: the packet is already at its node, shown as a ghost.
      pos = { x: node.cx, y: node.y - 14 };
      alpha = 0.22;
    } else {
      var path = pathBetween(from, to, true);
      pos = pointOnPolyline(path, p);
      ctx.save();
      ctx.strokeStyle = from < to ? 'rgba(56,189,248,.5)' : 'rgba(244,114,182,.35)';
      ctx.lineWidth = 2;
      if (from > to) ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(path[0].x, path[0].y);
      for (var i = 1; i < path.length; i++) ctx.lineTo(path[i].x, path[i].y);
      ctx.stroke();
      ctx.restore();
      // Hand over to the node: the badge inside the box takes it from here.
      if (p > 0.72) alpha = clamp(1 - (p - 0.72) / 0.28, 0.16, 1);
      // Horizontal hops must fit inside the column gap; a row change has the
      // whole row gap to itself.
      var a = L.nodes[from], b = L.nodes[to];
      maxW = a.row === b.row
        ? Math.max(58, Math.abs(b.x - a.x) - a.w + 10)
        : 134;
    }
    drawPacketChip(pos.x, pos.y, step.stage, alpha, maxW);
  }

  function renderLegend() {
    var x = 16, y = H - 15;
    ctx.save();
    ctx.font = '10.5px ' + SANS_PX;
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    [['act', 'activation'], ['logits', 'logits'], ['probs', 'probs'],
      ['kv', 'KV cache'], ['ids', 'token id']].forEach(function (it) {
      ctx.fillStyle = kindColor(it[0]);
      roundRect(ctx, x, y - 4, 8, 8, 2); ctx.fill();
      ctx.fillStyle = '#64748b';
      var w = ctx.measureText(it[1]).width;
      ctx.fillText(it[1], x + 12, y);
      x += 12 + w + 13;
    });
    // Idle at the first step: say how to start instead of how to jump.
    var tail = (!S.playing && S.idx === 0) ? "press ▶ (or Space) to start" : "click a step to jump";
    ctx.fillStyle = '#475569';
    ctx.fillText('·  ' + "NOW card: width ∝ tokens · height ∝ log₂(features)" + '  ·  ' + tail, x + 2, y);
    ctx.restore();
  }

  function render() {
    if (S.dirty) layout();
    syncBadges();
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, W, H);
    renderHeader();
    renderTextStrip();
    renderRail();
    renderPacket();
    renderLegend();
  }

  // ------------------------------------------------------------------- loop
  var last = 0;
  function frame(ts) {
    var dt = last ? Math.min(120, ts - last) : 16;
    last = ts;
    advance(dt);
    S.emitT = Math.min(1, (S.emitT == null ? 1 : S.emitT) + dt / 300);
    var step = curStep();
    if (step && step.kind === 'sweep') {
      var si = sweepSubIdx(step);
      if (S._sweepSub !== si) {
        S._sweepSub = si;
        updatePanel(step.subs[si], step, si);
      }
    }
    render();
    requestAnimationFrame(frame);
  }

  function resize() {
    dpr = window.devicePixelRatio || 1;
    var r = cv.getBoundingClientRect();
    W = Math.max(320, r.width);
    H = Math.max(240, r.height);
    cv.width = Math.round(W * dpr);
    cv.height = Math.round(H * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    S.dirty = true;
  }



  // ------------------------------------------------------------------ wiring
  function wire() {
    $('play').addEventListener('click', function () {
      if (!S.playing && S.idx === S.timeline.steps.length - 1) { goto(0); }
      S.playing = !S.playing;
      syncPlayBtn();
    });
    $('prev').addEventListener('click', function () { S.playing = false; syncPlayBtn(); goto(S.idx - 1); });
    $('next').addEventListener('click', function () { S.playing = false; syncPlayBtn(); goto(S.idx + 1); });
    $('restart').addEventListener('click', function () { goto(0); S.playing = true; syncPlayBtn(); });

    $('speed').addEventListener('input', function () {
      S.speed = parseInt(this.value, 10);
      $('speedlbl').textContent = String(S.speed);
    });

    $('phase').addEventListener('click', function (e) {
      var b = e.target.closest('button[data-mode]');
      if (!b) return;
      S.mode = b.dataset.mode;
      Array.prototype.forEach.call(this.querySelectorAll('button'), function (x) {
        x.classList.toggle('on', x === b);
      });
      // Switching the played range stays paused if the user was paused.
      rebuildTimeline();
      onEnterStep(0);
      syncPlayBtn();
    });

    $('layers').addEventListener('change', function () {
      S.withLayers = this.checked;
      rebuildTimeline(true); onEnterStep(S.idx); S.dirty = true;
    });

    $('detail').addEventListener('change', function () {
      S.detailDecode = this.checked;
      rebuildTimeline(true); onEnterStep(S.idx); S.dirty = true;
    });

    // The "Operators view" jump is an <a class="nav-btn"> in e2e.html: no
    // listener here, so a stale script cannot break it.

    $('chips').addEventListener('click', function (e) {
      var chip = e.target.closest('.chip');
      if (!chip) return;
      S.playing = false; syncPlayBtn();
      goto(parseInt(chip.dataset.i, 10));
    });

    $('file').addEventListener('change', function (e) {
      var f = e.target.files && e.target.files[0];
      if (!f) return;
      var fr = new FileReader();
      fr.onload = function () { applyJson(JSON.parse(fr.result), f.name); };
      fr.readAsText(f);
    });
    $('open').addEventListener('click', function () { $('file').click(); });

    $('sample').addEventListener('change', function () {
      if (!this.value) return;
      var opt = this.options[this.selectedIndex];
      fetch('samples/' + this.value)
        .then(function (r) { return r.json(); })
        .then(function (j) { applyJson(j, opt.textContent); })
        .catch(function (err) { $('src').textContent = 'load failed: ' + err.message; });
    });

    window.addEventListener('resize', resize);
    window.addEventListener('keydown', function (e) {
      var tag = e.target && e.target.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'BUTTON' || tag === 'A') return;
      if (e.key === ' ') { e.preventDefault(); $('play').click(); }
      if (e.key === 'ArrowRight') $('next').click();
      if (e.key === 'ArrowLeft') $('prev').click();
      if (e.key === 'r') $('restart').click();
    });
  }

  function applyJson(json, label) {
    var p = M.buildProfile(json);
    setProfile(p, false);
    if (label) { $('src').textContent = label; $('src').title = label; }
  }

  function loadManifest() {
    var sel = $('sample');
    fetch('samples/manifest.json')
      .then(function (r) { return r.json(); })
      .then(function (m) {
        (m.samples || []).forEach(function (s) {
          var o = document.createElement('option');
          o.value = s.file; o.textContent = s.label;
          sel.appendChild(o);
        });
        // The flagship sample loads itself so the page is never empty.
        var first = (m.samples || [])[0];
        if (first) {
          sel.value = first.file;
          return fetch('samples/' + first.file).then(function (r) { return r.json(); })
            .then(function (j) { applyJson(j, first.label); });
        }
      })
      .catch(function () { /* file:// or no samples: the built-in profile stands */ });
  }

  function boot() {
    cv = $('cv');
    ctx = cv.getContext('2d');
    wire();
    resize();
    setProfile(S.profile, false);
    loadManifest();
    requestAnimationFrame(frame);
  }

  if (typeof document !== 'undefined' && document.readyState !== 'loading') boot();
  else if (typeof document !== 'undefined') document.addEventListener('DOMContentLoaded', boot);

  // Exposed for the headless renderer check (scripts/check_viz_e2e_render.mjs).
  window.E2EApp = {
    state: S,
    boot: boot,
    render: render,
    advance: advance,
    goto: goto,
    setProfile: setProfile,
    layout: layout,
    curStep: curStep,
    applyJson: applyJson,
    rebuildTimeline: rebuildTimeline,
    setSpeed: function (v) { S.speed = v; },
    setMode: function (m) { S.mode = m; rebuildTimeline(true); onEnterStep(S.idx); },
    panelHtml: function () { return $('panel-body').innerHTML; },
    chipCount: function () { return S.timeline.steps.length; }
  };
})();
