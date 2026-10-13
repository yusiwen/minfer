#!/usr/bin/env node
/* The viz pages' external inputs are fresh: the asset tags they write, and the data
 * they fetch at runtime (#103, #546).
 *
 * The pages are plain HTML with no build step. `viz/index.html` loads `style.css`
 * and `app.js`; `viz/e2e.html` loads `e2e.css`, `e2e-model.js` and `e2e.js`. They
 * used to name a hand-bumped `?v=N` so a caching static server would not pair a
 * fresh page with a stale asset, and nothing checked that the number had been
 * bumped; they now compose each URL at load time (`?t=<epoch ms>`) inside an inline
 * loader, so there is no number to forget.
 *
 * The same hazard applies to the *data* the pages fetch: `samples/manifest.json`
 * and `samples/<file>.json` are regenerated in place under stable names, and a
 * plain static server sends no `Cache-Control`, only `Last-Modified` — so a browser
 * may reuse a previous response for a heuristic freshness window. Every runtime
 * request therefore passes `cache: 'no-cache'`, which keeps the stored body and has
 * the server confirm it (RFC 9111 §4.3).
 *
 * What this checks, and against which independent value:
 *
 *   * the **file set** — every `viz/*.css` and `viz/*.js`, read from the
 *     filesystem — must be requested by exactly the pages, and nothing else may
 *     be requested. An asset nobody loads is a page that forgot it; a request for
 *     an asset that is not there is a renamed file and a dead page;
 *   * every requested URL must carry `?t=<digits>`: it is *composed* at load time,
 *     not written into the markup. The loader is evaluated in a `vm` with a
 *     stubbed `document.write`, so what is inspected is the URL the page would
 *     actually request, not a pattern match on the source;
 *   * no page may reach a `.css`/`.js` through a literal `href=`/`src=` attribute
 *     outside a `<script>` — that is the bypass that would silently reintroduce
 *     the hand-maintained version (and the stale-asset window with it);
 *   * every `fetch(...)` call site in `viz/*.js` must pass `cache: 'no-cache'`,
 *     unless it names a non-GET `method:` (HTTP does not cache those). The call
 *     sites are found by scanning the source with strings and comments skipped;
 *     this is an **audit, not a browser**, so it proves the option is written, not
 *     that a cache honoured it.
 *
 * What it cannot prove: that a browser honours either mechanism. It cannot see a
 * cache, a proxy or a service worker; the argument for `?t=` is that a URL no cache
 * has seen cannot be served from one, and the argument for `no-cache` is that the
 * request asks for exactly the confirmation that would otherwise be skipped. The
 * engine-side halves — `no-store` on every page and asset, and `no-cache` on
 * `samples/` — are pinned by `src/server/viz/tests.rs`.
 *
 * Usage:
 *
 *     node scripts/check_viz_assets.mjs
 *     node scripts/check_viz_assets.mjs --selftest
 *
 * Exit codes: 0 every page's requests match the asset set and revalidate, 1 a drift
 * was found, 2 usage or I/O error. Stdlib + node builtins only.
 */
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, '..');
const VIZ = path.join(repo, 'viz');

/** The pages that carry a loader. A new page must be added here *and* checked in. */
const PAGES = ['index.html', 'e2e.html'];

const IS_ASSET = /\.(?:css|js)$/;
/** The page scripts whose runtime requests are audited: every `viz/*.js`. */
const IS_SCRIPT = /\.js$/;
/** Any literal `href=`/`src=` value — the declarative form we no longer allow. */
const ATTRIBUTE = /\b(?:href|src)\s*=\s*"([^"]*)"/gi;
/** An attribute value that names a `.css`/`.js`, query string and all. */
const IS_ASSET_REF = /^[^?#]+\.(?:css|js)(?:[?#].*)?$/;
/** An attribute inside a string the loader writes (`document.write`). */
const WRITTEN_ATTRIBUTE = /\b(?:href|src)\s*=\s*"([^"]+)"/gi;
const SCRIPT_BLOCK = /<script\b[^>]*>([\s\S]*?)<\/script>/gi;
/** A script block, with its *opening tag* kept: the tag is markup, the body is not. */
const SCRIPT_BODY = /(<script\b[^>]*>)[\s\S]*?<\/script>/gi;
const HTML_COMMENT = /<!--[\s\S]*?-->/g;
/** The stamp: this is what makes the URL one no cache has seen. */
const STAMPED = /\?t=(\d+)$/;
/** The runtime-request mechanism: keep the stored body, confirm it with the server. */
const REVALIDATES = /cache\s*:\s*['"]no-cache['"]/;
/** A call site that names its method is not a cacheable GET (POST/PUT/… are exempt). */
const EXPLICIT_METHOD = /\bmethod\s*:/;

/** The inline loader bodies of one page: the `<script>` blocks that write a tag. */
function loadersOf(html) {
  return [...html.matchAll(SCRIPT_BLOCK)]
    .map((m) => m[1])
    .filter((body) => body.includes('document.write'));
}

/** The asset URLs one page composes at load time.
 *
 * The loader is executed, not pattern-matched: substitution, concatenation and
 * ordering are its business, and only running it shows the URL it produces. The
 * sandbox holds `document` and nothing else — no `require`, no `process`, no
 * filesystem — so evaluating the page's own inline script cannot reach outside.
 */
function requestsOf(html) {
  const written = [];
  const sandbox = { document: { write: (chunk) => written.push(String(chunk)) } };
  for (const body of loadersOf(html)) {
    vm.runInNewContext(body, sandbox, { timeout: 1000 });
  }
  const urls = [];
  for (const chunk of written) {
    for (const m of chunk.matchAll(WRITTEN_ATTRIBUTE)) {
      urls.push(m[1]);
    }
  }
  return urls;
}

/** `.css`/`.js` reached by a literal attribute in `html`, comments removed.
 *
 * Script *bodies* are dropped but their opening tags are kept, so a real
 * `<script src="app.js">` is still a tag to inspect, while the `<script src="…">`
 * text a loader writes into its own string is not (it lives in a body, and it is
 * the loader's business). A page whose script contains a literal `</script>`
 * inside a string would truncate that body — no page here does, and the loader
 * escapes it as `<\/script>`.
 */
function bareReferencesOf(html) {
  const stripped = html.replace(HTML_COMMENT, '').replace(SCRIPT_BODY, '$1');
  return [...stripped.matchAll(ATTRIBUTE)]
    .map((m) => m[1])
    .filter((value) => IS_ASSET_REF.test(value));
}

// --- runtime request audit ------------------------------------------------

const isIdentChar = (ch) => /[A-Za-z0-9_$]/.test(ch || '');

/** The index of the quote that closes the string opening at `start`. */
function endOfString(source, start) {
  const quote = source[start];
  for (let i = start + 1; i < source.length; i++) {
    if (source[i] === '\\') {
      i++;
    } else if (source[i] === quote) {
      return i;
    }
  }
  return source.length;
}

/** The index of the `)` matching the `(` at `open`, with strings and comments skipped. */
function matchParen(source, open) {
  let depth = 0;
  for (let i = open; i < source.length; i++) {
    const ch = source[i];
    if (ch === '"' || ch === "'" || ch === '`') {
      i = endOfString(source, i);
    } else if (ch === '/' && source[i + 1] === '/') {
      i = source.indexOf('\n', i);
      if (i < 0) return source.length;
    } else if (ch === '/' && source[i + 1] === '*') {
      const end = source.indexOf('*/', i + 2);
      i = end < 0 ? source.length : end + 1;
    } else if (ch === '(') {
      depth++;
    } else if (ch === ')' && --depth === 0) {
      return i;
    }
  }
  return source.length;
}

/** Every `fetch(...)` call site in one page script: `{line, args}`.
 *
 * A source scan with strings and comments skipped, so a `fetch(` inside a string or
 * a commented-out line is not a call site. Template-literal nesting and regex
 * literals are not modelled — no page here uses either around a `fetch(`.
 */
function fetchCallsOf(source) {
  const calls = [];
  let line = 1;
  let i = 0;
  while (i < source.length) {
    const ch = source[i];
    if (ch === '\n') {
      line++;
      i++;
      continue;
    }
    if (ch === '"' || ch === "'" || ch === '`') {
      const end = endOfString(source, i);
      line += (source.slice(i, end).match(/\n/g) || []).length;
      i = end + 1;
    } else if (ch === '/' && source[i + 1] === '/') {
      const end = source.indexOf('\n', i);
      if (end < 0) break;
      i = end;
    } else if (ch === '/' && source[i + 1] === '*') {
      const end = source.indexOf('*/', i + 2);
      const stop = end < 0 ? source.length : end + 1;
      line += (source.slice(i, stop).match(/\n/g) || []).length;
      i = stop + 1;
    } else if (source.startsWith('fetch', i) && !isIdentChar(source[i - 1]) && source[i + 5] === '(') {
      const open = i + 5;
      const close = matchParen(source, open);
      calls.push({ line, args: source.slice(open + 1, close) });
      line += (source.slice(i, close).match(/\n/g) || []).length;
      i = close + 1;
    } else {
      i++;
    }
  }
  return calls;
}

/** The failures one page script contributes: a call site that may be cache-served. */
function scriptFailures(name, source) {
  const failures = [];
  for (const call of fetchCallsOf(source)) {
    if (EXPLICIT_METHOD.test(call.args)) continue;
    if (!REVALIDATES.test(call.args)) {
      failures.push(
        `viz/${name}:${call.line}: fetch(...) may be served from a heuristic cache — pass { cache: 'no-cache' }`,
      );
    }
  }
  return failures;
}

/** Inspect the tree: `pages` (`[{name, html}]`) against `assetFiles` (`['app.js', …]`),
 * plus `scripts` (`[{name, source}]`).
 *
 * Pure: no filesystem, no clock. `--selftest` drives it with synthetic sources, so
 * the checker's own cases are the control arm for the real run below.
 */
function inspect({ assetFiles, pages, scripts }) {
  const failures = [];
  const requested = new Set();

  for (const page of pages) {
    const loaders = loadersOf(page.html);
    if (loaders.length === 0) {
      failures.push(`${page.name}: no inline loader writes an asset tag`);
    }
    let urls = null;
    try {
      urls = requestsOf(page.html);
    } catch (err) {
      // A loader that throws is a page that loads nothing: report it as the drift
      // it is, instead of letting the stack trace be the answer.
      failures.push(`${page.name}: the loader threw while composing its URLs (${err.message})`);
    }
    if (urls && urls.length === 0) {
      failures.push(`${page.name}: the loader requests no asset`);
    }
    for (const url of urls || []) {
      if (!IS_ASSET.test(url.split('?')[0])) {
        failures.push(`${page.name}: loader requested a non-asset '${url}'`);
        continue;
      }
      if (!STAMPED.test(url)) {
        failures.push(`${page.name}: '${url}' carries no ?t=<epoch ms> stamp`);
        continue;
      }
      const file = url.slice(0, url.indexOf('?'));
      if (file.includes('/') || file.includes('..')) {
        failures.push(`${page.name}: asset '${file}' must be a plain file in viz/`);
      }
      if (!assetFiles.includes(file)) {
        failures.push(`${page.name}: requests '${file}', which is not in viz/`);
      }
      requested.add(file);
    }
    for (const ref of bareReferencesOf(page.html)) {
      failures.push(`${page.name}: '${ref}' is a literal link/script attribute — route it through the loader`);
    }
  }

  for (const file of assetFiles) {
    if (!requested.has(file)) {
      failures.push(`viz/${file}: no page requests it (add it to a loader or delete it)`);
    }
  }

  let sites = 0;
  for (const script of scripts) {
    const calls = fetchCallsOf(script.source);
    sites += calls.length;
    failures.push(...scriptFailures(script.name, script.source));
  }
  if (sites === 0) {
    failures.push('viz/*.js: no runtime fetch call site found — if the pages no longer fetch, teach this checker');
  }
  return failures;
}

const GOOD_LOADER =
  "document.write('<link rel=\"stylesheet\" href=\"a.css?t=' + Date.now() + '\">');";
const page = (name, html) => ({ name, html });
const script = (name, source) => ({ name, source });
/** A page script that follows both rules: a revalidating GET and an uncacheable POST. */
const GOOD_SCRIPT = script(
  'good.js',
  [
    "function load() {",
    "  fetch('samples/manifest.json', { cache: 'no-cache' }).then(function (r) { return r.json(); });",
    "  fetch('/viz/run', { method: 'POST', body: '{}' });",
    "}",
  ].join('\n'),
);

/** The checker's own pass/fail cases: one good tree, one per drift shape. */
function selftest() {
  let failed = 0;
  const caseOk = (label, cond, detail) => {
    if (cond) return;
    failed++;
    console.error(`  FAIL selftest: ${label}${detail ? ` — ${detail}` : ''}`);
  };
  const expect = (label, failures, needle) =>
    caseOk(label, failures.some((f) => f.includes(needle)), `got ${JSON.stringify(failures)}`);
  /** The asset half of every case below, held good so only its own rule can fail. */
  const withScripts = (parts) => ({ scripts: [GOOD_SCRIPT], ...parts });

  const good = inspect(
    withScripts({
      assetFiles: ['a.css', 'a.js'],
      pages: [page('good.html', `<script>${GOOD_LOADER}</script>\n<script>document.write('<script src="a.js?t=' + Date.now() + '"><\\/script>');</script>`)],
    }),
  );
  caseOk('a stamped loader pair and a revalidating script pass', good.length === 0, JSON.stringify(good));

  const bare = inspect(
    withScripts({
      assetFiles: ['a.css', 'a.js'],
      pages: [page('bare.html', `<script>${GOOD_LOADER}</script>\n<script src="a.js"></script>`)],
    }),
  );
  expect('a literal <script src> is a bypass', bare, "'a.js' is a literal link/script attribute");
  expect('a bypassed asset is also an orphan', bare, 'viz/a.js: no page requests it');

  // The markup this gate replaced, verbatim: a hand-bumped version parameter.
  const handBumped = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('bumped.html', `<link rel="stylesheet" href="a.css?v=2">`)],
    }),
  );
  expect('a hand-bumped ?v= attribute is a bypass', handBumped, "'a.css?v=2' is a literal link/script attribute");

  const orphan = inspect(
    withScripts({
      assetFiles: ['a.css', 'b.js'],
      pages: [page('orphan.html', `<script>${GOOD_LOADER}</script>`)],
    }),
  );
  expect('an asset no page requests fails', orphan, 'viz/b.js: no page requests it');

  const phantom = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('phantom.html', `<script>document.write('<link rel="stylesheet" href="gone.css?t=' + Date.now() + '">');</script>`)],
    }),
  );
  expect('a request for a missing asset fails', phantom, "requests 'gone.css', which is not in viz/");

  const unstamped = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('unstamped.html', `<script>document.write('<link rel="stylesheet" href="a.css">');</script>`)],
    }),
  );
  expect('an unstamped URL fails', unstamped, 'carries no ?t=<epoch ms> stamp');

  const unloaded = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('nobody.html', '<p>no loader here</p>')],
    }),
  );
  expect('a page with no loader fails', unloaded, 'no inline loader writes an asset tag');

  const throwing = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('throwing.html', `<script>document.write(notDefined());</script>`)],
    }),
  );
  expect('a loader that throws is a named failure', throwing, 'the loader threw while composing its URLs');

  // --- the runtime-request half
  const cacheless = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('cacheless.html', `<script>${GOOD_LOADER}</script>`)],
      scripts: [script('cacheless.js', "fetch('samples/manifest.json');")],
    }),
  );
  expect('a fetch without cache: no-cache fails', cacheless, "fetch(...) may be served from a heuristic cache — pass { cache: 'no-cache' }");
  expect('the failing site carries its line number', cacheless, 'viz/cacheless.js:1:');

  const postOnly = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('post.html', `<script>${GOOD_LOADER}</script>`)],
      scripts: [script('post.js', "fetch('/viz/run', { method: 'POST' });")],
    }),
  );
  caseOk('a POST call site needs no cache option', postOnly.length === 0, JSON.stringify(postOnly));

  const commentsOnly = inspect(
    withScripts({
      assetFiles: ['a.css'],
      pages: [page('commented.html', `<script>${GOOD_LOADER}</script>`)],
      scripts: [script('commented.js', "// fetch('samples/manifest.json')\n/* fetch('x') */\nvar s = 'fetch(\"y\")';")],
    }),
  );
  expect('a commented or quoted fetch is not a call site', commentsOnly, 'no runtime fetch call site found');

  if (failed) {
    console.error(`\nFAIL  selftest: ${failed} case(s) wrong`);
    return 1;
  }
  console.log('OK  selftest: 13/13 checker cases behave as intended');
  return 0;
}

function main() {
  const files = fs.readdirSync(VIZ);
  const assetFiles = files.filter((f) => IS_ASSET.test(f)).sort();
  const pages = PAGES.map((name) => ({
    name,
    html: fs.readFileSync(path.join(VIZ, name), 'utf8'),
  }));
  const scripts = files
    .filter((f) => IS_SCRIPT.test(f))
    .sort()
    .map((name) => ({ name, source: fs.readFileSync(path.join(VIZ, name), 'utf8') }));

  const failures = inspect({ assetFiles, pages, scripts });
  const requests = pages
    .map((p) => {
      // A page whose loader throws is already a named failure; the summary must not
      // be the thing that crashes on it.
      try {
        return `${p.name}: ${requestsOf(p.html).length}`;
      } catch {
        return `${p.name}: threw`;
      }
    })
    .join(', ');
  const sites = scripts.map((s) => `${s.name}: ${fetchCallsOf(s.source).length}`).join(', ');
  console.log(
    `check_viz_assets: ${assetFiles.length} asset(s) in viz/ (${assetFiles.join(', ')}), ` +
      `${pages.length} page(s) (${requests} request(s)), ${scripts.length} script(s) (${sites} fetch site(s))`,
  );
  for (const failure of failures) {
    console.error(`  FAIL ${failure}`);
  }
  if (failures.length) {
    console.error(`\nFAIL  ${failures.length} freshness drift(s)`);
    return 1;
  }
  console.log(
    `OK  ${assetFiles.length}/${assetFiles.length} assets through the stamped loader, ` +
      `every runtime fetch revalidates`,
  );
  return 0;
}

if (process.argv.includes('--selftest')) {
  process.exit(selftest());
}
if (process.argv.length > 2) {
  console.error('usage: node scripts/check_viz_assets.mjs [--selftest]');
  process.exit(2);
}
process.exit(main());
