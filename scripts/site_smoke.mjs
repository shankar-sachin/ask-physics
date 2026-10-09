// End-to-end check of the website in headless Chromium: serve build/site,
// boot the real Python engine, and ask questions through the actual page.
//
//   sh scripts/build_site.sh && node scripts/site_smoke.mjs
//
// Needs Playwright with Chromium. PYODIDE_LOCAL_DIR=<dir> serves Pyodide from a
// local copy of the full distribution instead of the CDN (for offline sandboxes).
import { createRequire } from "node:module";
import { createServer } from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const ROOT = "build/site";
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript",
  ".css": "text/css", ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
  ".json": "application/json", ".wasm": "application/wasm", ".gz": "application/gzip",
  ".zip": "application/zip", ".whl": "application/zip", ".sh": "text/plain", ".ps1": "text/plain",
};

async function serveFile(dir, urlPath) {
  const path = normalize(join(dir, decodeURIComponent(urlPath)));
  if (!path.startsWith(normalize(dir))) throw new Error("path escapes root");
  return { body: await readFile(path), type: TYPES[extname(path)] || "application/octet-stream" };
}

// Offline mode: Playwright can't intercept requests made inside a Web Worker, so
// the server hands out a worker.js pointed at a local Pyodide and local wheels.
const local = process.env.PYODIDE_LOCAL_DIR;
const wheels = new Map();
if (local) {
  for (const spec of ["pint==0.26.1", "flexcache==0.3", "flexparser==0.4"]) {
    const [name, version] = spec.split("==");
    const meta = await (await fetch(`https://pypi.org/pypi/${name}/${version}/json`)).json();
    const wheel = meta.urls.find((u) => u.filename.endsWith("-none-any.whl"));
    wheels.set(wheel.filename, Buffer.from(await (await fetch(wheel.url)).arrayBuffer()));
  }
}

const server = createServer(async (req, res) => {
  try {
    const url = new URL(req.url, "http://localhost");
    if (local && url.pathname.startsWith("/__pyodide/")) {
      const { body, type } = await serveFile(local, url.pathname.slice("/__pyodide/".length));
      res.writeHead(200, { "Content-Type": type });
      return res.end(body);
    }
    if (local && url.pathname.startsWith("/__wheels/")) {
      res.writeHead(200, { "Content-Type": "application/zip" });
      return res.end(wheels.get(url.pathname.slice("/__wheels/".length)));
    }
    if (local && url.pathname === "/worker.js") {
      let js = (await readFile(join(ROOT, "worker.js"), "utf8"))
        .replace(/const PYODIDE_URL = .*;/, `const PYODIDE_URL = "${base}__pyodide/";`)
        .replace(/const PYPI_PACKAGES = .*;/, `const PYPI_PACKAGES = ${JSON.stringify([...wheels.keys()].map((f) => `${base}__wheels/${f}`))};`);
      res.writeHead(200, { "Content-Type": "text/javascript" });
      return res.end(js);
    }
    const { body, type } = await serveFile(ROOT, url.pathname.endsWith("/") ? `${url.pathname}index.html` : url.pathname);
    res.writeHead(200, { "Content-Type": type });
    res.end(body);
  } catch {
    res.writeHead(404);
    res.end("not found");
  }
});
await new Promise((resolve) => server.listen(0, resolve));
const base = `http://localhost:${server.address().port}/`;

const browser = await chromium.launch();
const context = await browser.newContext({ viewport: { width: 1200, height: 1600 } });
if (local) {
  await context.route(/^https:\/\/fonts\.(googleapis|gstatic)\.com\//, (route) => route.abort());
  // Offline, the docs' KaTeX can't load either; its math then shows as plain LaTeX.
  await context.route(/^https:\/\/cdn\.jsdelivr\.net\/npm\/katex@/, (route) => route.abort());
}

const page = await context.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
page.on("requestfailed", (request) => {
  const url = request.url();
  // Offline mode blocks web fonts on purpose; anything else failing is a bug.
  const blocked = /fonts\.(googleapis|gstatic)\.com|cdn\.jsdelivr\.net\/npm\/katex@/;
  if (!(local && blocked.test(url))) errors.push(`request failed: ${url}`);
});
page.on("console", (msg) => {
  // Failed loads are reported (with their URL) by the requestfailed handler.
  if (msg.type() === "error" && !msg.text().startsWith("Failed to load resource")) {
    errors.push(`console: ${msg.text()}`);
  }
});

function check(condition, message) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    if (errors.length) console.error(errors.join("\n"));
    process.exitCode = 1;
    throw new Error(message);
  }
  console.log(`ok   ${message}`);
}

async function ask(question) {
  await page.fill("#question", question);
  await page.click("#ask-button");
  await page.waitForSelector("#result .card, #result .error", { timeout: 240_000 });
  const error = await page.$("#result .error");
  if (error) check(false, `asking "${question}" failed: ${await error.textContent()}`);
  return page.$("#result .card");
}

try {
  await page.goto(base);
  check((await page.title()) === "Ask Physics", "page loads with its title");

  let card = await ask("A ball is dropped from 45 m. How fast does it hit the ground?");
  check(await card.evaluate((n) => n.classList.contains("answered")), "falling-ball question is ANSWERED");
  const big = await card.$eval(".big", (n) => n.textContent);
  check(big.includes("29.7086") && big.includes("m/s"), `result shows 29.7086 m/s (got "${big}")`);
  const math = await card.$eval(".eq-math", (n) => n.textContent);
  check(math === "v² = v₀² + 2·a·d", `equation renders as one-line math (got "${math}")`);
  check((await card.$$(".input-row")).length === 3, "all three inputs are listed");
  const docsLink = await card.$eval("a.eq-id", (n) => n.getAttribute("href"));
  check(docsLink === "/docs/kin_v_squared/", `the equation id links to its docs page (${docsLink})`);
  const engine = await page.textContent("#engine-text");
  check(engine.includes("Engine ready"), `engine reports ready (${engine})`);
  await page.screenshot({ path: "build/site-smoke.png", fullPage: false });

  card = await ask("How much does the color blue weigh?");
  check(await card.evaluate((n) => n.classList.contains("refused")), "color question is refused");
  check(!!(await card.$(".redirect")), "refusal offers a 'try instead' question");

  card = await ask("How many rubber ducks would it take to stop a freight train?");
  check(await card.evaluate((n) => n.classList.contains("degraded")), "Fermi question degrades honestly");

  // Question text must render as text, never as markup.
  card = await ask("<img src=x onerror=alert(1)> dropped from 20 m, how fast?");
  check((await card.$$("img")).length === 0, "question HTML is escaped");

  // The docs: the home page, then the equation page the answer card linked to.
  await page.goto(`${base}docs/`);
  check((await page.title()).startsWith("Ask Physics Docs"), "the docs home page loads");
  await page.goto(`${base}docs/kin_v_squared/`);
  const heading = await page.textContent(".docs-page h1");
  check(heading.length > 0 && heading !== "kin_v_squared", `the equation page is titled by name (${heading})`);
  if (local) {
    check((await page.textContent(".math")).includes("v^2"), "offline, the formula shows as LaTeX");
  } else {
    await page.waitForSelector(".math .katex", { timeout: 30_000 });
    check(true, "KaTeX typesets the formula");
  }
  check((await page.$$(".docs-page table tbody tr")).length === 4, "the variables table lists all four");

  // On a phone: nothing scrolls sideways, and every nav link, the docs' included, is there.
  const phone = await browser.newContext({
    viewport: { width: 375, height: 740 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
  });
  if (local) {
    await phone.route(/^https:\/\/fonts\.(googleapis|gstatic)\.com\//, (route) => route.abort());
    await phone.route(/^https:\/\/cdn\.jsdelivr\.net\/npm\/katex@/, (route) => route.abort());
  }
  const small = await phone.newPage();
  small.on("pageerror", (error) => errors.push(`phone pageerror: ${error.message}`));
  const fits = () => small.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth);
  await small.goto(base);
  check(await fits(), "on a phone, the home page doesn't scroll sideways");
  const navLinks = await small.$$eval(".nav nav a", (links) =>
    links.filter((a) => a.offsetParent !== null).map((a) => a.textContent.trim()),
  );
  check(navLinks.length === 6 && navLinks.includes("Docs"), `on a phone, the nav shows every link (${navLinks})`);
  await small.click(".nav nav a[href='/docs/']");
  await small.waitForURL(/\/docs\/$/);
  check(await fits(), "on a phone, the docs home doesn't scroll sideways");
  await small.goto(`${base}docs/kin_v_squared/`);
  check(await fits(), "on a phone, an equation page doesn't scroll sideways");
  await phone.close();

  check(errors.length === 0, `no page or console errors${errors.length ? `:\n${errors.join("\n")}` : ""}`);
} finally {
  await browser.close();
  server.close();
}
