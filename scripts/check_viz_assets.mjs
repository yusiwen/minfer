#!/usr/bin/env node
/* Every asset a viz page loads must go through the page's stamped loader (#103).
 *
 * The pages are plain HTML with no build step: `viz/index.html` loads `style.css`
 * and `app.js`, `viz/e2e.html` loads `e2e.css`, `e2e-model.js` and `e2e.js`. They
 * used to name a hand-bumped `?v=N` so a caching static server would not pair a
 * fresh page with a stale asset, and nothing checked that the number had been
 * bumped. The pages now compose each URL at load time (`?t=<epoch ms>`) inside an
 * inline loader, so there is no number to forget.
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
 *     the hand-maintained version (and the stale-asset window with it).
 *
 * What it cannot prove: that the browser honours the stamp. It cannot see a cache,
 * a proxy or a service worker; the argument for `?t=` is that a URL no cache has
 * seen cannot be served from one, and the engine-side counterpart (`no-store` on
 * every page and asset) is pinned by `src/server/viz/tests.rs`.
 *
 * Usage:
 *
 *     node scripts/check_viz_assets.mjs
 *     node scripts/check_viz_assets.mjs --selftest
 *
 * Exit codes: 0 every page's requests match the asset set, 1 a drift was found,
 * 2 usage or I/O error. Stdlib + node builtins only.
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

/** Inspect `pages` (`[{name, html}]`) against `assetFiles` (`['app.js', …]`).
 *
 * Pure: no filesystem, no clock. `--selftest` drives it with synthetic sources, so
 * the checker's own cases are the control arm for the real run below.
 */
function inspect({ assetFiles, pages }) {
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
  return failures;
}

const GOOD_LOADER =
  "document.write('<link rel=\"stylesheet\" href=\"a.css?t=' + Date.now() + '\">');";
const page = (name, html) => ({ name, html });

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

  const good = inspect({
    assetFiles: ['a.css', 'a.js'],
    pages: [page('good.html', `<script>${GOOD_LOADER}</script>\n<script>document.write('<script src="a.js?t=' + Date.now() + '"><\\/script>');</script>`)],
  });
  caseOk('a stamped loader pair passes', good.length === 0, JSON.stringify(good));

  const bare = inspect({
    assetFiles: ['a.css', 'a.js'],
    pages: [page('bare.html', `<script>${GOOD_LOADER}</script>\n<script src="a.js"></script>`)],
  });
  expect('a literal <script src> is a bypass', bare, "'a.js' is a literal link/script attribute");
  expect('a bypassed asset is also an orphan', bare, 'viz/a.js: no page requests it');

  // The markup this gate replaced, verbatim: a hand-bumped version parameter.
  const handBumped = inspect({
    assetFiles: ['a.css'],
    pages: [page('bumped.html', `<link rel="stylesheet" href="a.css?v=2">`)],
  });
  expect('a hand-bumped ?v= attribute is a bypass', handBumped, "'a.css?v=2' is a literal link/script attribute");

  const orphan = inspect({
    assetFiles: ['a.css', 'b.js'],
    pages: [page('orphan.html', `<script>${GOOD_LOADER}</script>`)],
  });
  expect('an asset no page requests fails', orphan, 'viz/b.js: no page requests it');

  const phantom = inspect({
    assetFiles: ['a.css'],
    pages: [page('phantom.html', `<script>document.write('<link rel="stylesheet" href="gone.css?t=' + Date.now() + '">');</script>`)],
  });
  expect('a request for a missing asset fails', phantom, "requests 'gone.css', which is not in viz/");

  const unstamped = inspect({
    assetFiles: ['a.css'],
    pages: [page('unstamped.html', `<script>document.write('<link rel="stylesheet" href="a.css">');</script>`)],
  });
  expect('an unstamped URL fails', unstamped, 'carries no ?t=<epoch ms> stamp');

  const unloaded = inspect({
    assetFiles: ['a.css'],
    pages: [page('nobody.html', '<p>no loader here</p>')],
  });
  expect('a page with no loader fails', unloaded, 'no inline loader writes an asset tag');

  const throwing = inspect({
    assetFiles: ['a.css'],
    pages: [page('throwing.html', `<script>document.write(notDefined());</script>`)],
  });
  expect('a loader that throws is a named failure', throwing, 'the loader threw while composing its URLs');

  if (failed) {
    console.error(`\nFAIL  selftest: ${failed} case(s) wrong`);
    return 1;
  }
  console.log('OK  selftest: 8/8 checker cases behave as intended');
  return 0;
}

function main() {
  const assetFiles = fs
    .readdirSync(VIZ)
    .filter((f) => IS_ASSET.test(f))
    .sort();
  const pages = PAGES.map((name) => ({
    name,
    html: fs.readFileSync(path.join(VIZ, name), 'utf8'),
  }));

  const failures = inspect({ assetFiles, pages });
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
  console.log(
    `check_viz_assets: ${assetFiles.length} asset(s) in viz/ (${assetFiles.join(', ')}), ` +
      `${pages.length} page(s) (${requests} request(s))`,
  );
  for (const failure of failures) {
    console.error(`  FAIL ${failure}`);
  }
  if (failures.length) {
    console.error(`\nFAIL  ${failures.length} asset drift(s)`);
    return 1;
  }
  console.log(`OK  ${assetFiles.length}/${assetFiles.length} assets requested through the stamped loader`);
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
