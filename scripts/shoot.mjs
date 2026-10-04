// Screenshot HTML pages with headless Chromium (Playwright).
// Usage: node scripts/shoot.mjs jobs.json
// jobs.json: [{"html": "page.html", "out": "x.png", "width": 1200, "height": 1200,
//              "scale": 1, "type": "png" | "jpeg", "quality": 90}]
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const jobs = JSON.parse(readFileSync(process.argv[2], "utf8"));
const browser = await chromium.launch();
for (const job of jobs) {
  const page = await browser.newPage({
    viewport: { width: job.width, height: job.height },
    deviceScaleFactor: job.scale ?? 1,
  });
  await page.goto(pathToFileURL(job.html).href, { waitUntil: "load" });
  await page.evaluate(() => document.fonts.ready);
  const options = { path: job.out, type: job.type ?? "png", omitBackground: job.transparent ?? false };
  if (options.type === "jpeg") options.quality = job.quality ?? 90;
  await page.screenshot(options);
  await page.close();
  console.log(job.out);
}
await browser.close();
