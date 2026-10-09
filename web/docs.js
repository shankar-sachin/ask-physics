// Typesets the wiki's ```math blocks with KaTeX. If KaTeX didn't load, the LaTeX
// source stays visible as plain text, which is still readable.
for (const block of document.querySelectorAll(".math")) {
  if (!window.katex) break;
  try {
    window.katex.render(block.textContent, block, { displayMode: true, throwOnError: false });
  } catch {
    // Leave the source text in place.
  }
}
