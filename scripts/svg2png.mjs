// Render Rich SVG exports to crisp PNGs with headless Chromium (Playwright).
// Usage: node scripts/svg2png.mjs out_dir file1.svg [file2.svg ...]
import { createRequire } from "node:module";
import { readFileSync } from "node:fs";
import { basename, join } from "node:path";

const require = createRequire(import.meta.url);
const { chromium } = require("playwright");

const [outDir, ...files] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ deviceScaleFactor: 2 });
for (const file of files) {
  const svg = readFileSync(file, "utf8");
  await page.setContent(
    `<html><body style="margin:0;background:transparent">${svg}</body></html>`,
  );
  const element = await page.$("svg");
  const out = join(outDir, basename(file).replace(/\.svg$/, ".png"));
  await element.screenshot({ path: out, omitBackground: true });
  console.log(out);
}
await browser.close();
