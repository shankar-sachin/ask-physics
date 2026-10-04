// The Ask Physics page: boots the Python engine in a worker when the visitor
// gets near the question box, sends questions, and renders the answer card.
// Every string from the question or the engine goes in through textContent,
// never innerHTML.

const form = document.getElementById("ask-form");
const input = document.getElementById("question");
const button = document.getElementById("ask-button");
const result = document.getElementById("result");
const engineLine = document.getElementById("engine");
const engineText = document.getElementById("engine-text");

const STATUS = {
  answered: "ANSWERED",
  degraded: "PARTIAL",
  refused: "CAN'T ANSWER",
};
const REDIRECT_MARKER = " A close question that can be answered:";

let worker = null;
let nextId = 0;
const pending = new Map();

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

function setEngine(state, text) {
  engineLine.className = `engine ${state}`;
  engineText.textContent = text;
}

function startEngine() {
  if (worker) return;
  worker = new Worker("worker.js", { type: "module" });
  setEngine("booting", "Starting the physics engine…");
  worker.onmessage = ({ data }) => {
    if (data.type === "status") {
      setEngine("booting", `Booting the physics engine (${data.step}/${data.total}): ${data.text}…`);
    } else if (data.type === "ready") {
      const d = data.info.data;
      setEngine(
        "ready",
        `Engine ready · askphysics ${data.info.version} · ${d.equations} equations · running on your device`,
      );
      document.getElementById("version").textContent = `v${data.info.version}`;
    } else if (data.type === "answer" || data.type === "error") {
      const settle = pending.get(data.id);
      if (data.type === "error") {
        setEngine("failed", `Engine error: ${data.message}`);
        if (settle) settle.reject(new Error(data.message));
      } else if (settle) {
        settle.resolve(data.payload);
      }
      pending.delete(data.id);
    }
  };
  worker.onerror = (event) => setEngine("failed", `Engine crashed: ${event.message || "unknown error"}`);
  worker.postMessage({ type: "boot" });
}

function askEngine(question) {
  startEngine();
  const id = ++nextId;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ type: "ask", id, question });
  });
}

function meter(score, kind) {
  const bar = el("span", `meter ${kind}`);
  const filled = Math.round(score * 20);
  for (let i = 0; i < 20; i++) bar.append(el("i", i < filled ? "on" : ""));
  return bar;
}

function row(rows, label, ...nodes) {
  rows.append(el("dt", "", label));
  const dd = el("dd");
  dd.append(...nodes);
  rows.append(dd);
}

function list(items, className) {
  const ul = el("ul", className);
  for (const item of items) ul.append(el("li", "", item));
  return ul;
}

function renderCard({ answer, display }) {
  const card = el("article", `card ${answer.status}`);

  const head = el("div", "card-head");
  head.append(el("span", "badge", STATUS[answer.status]), el("span", "category", answer.category.replace("_", " ")));
  card.append(head, el("p", "question", answer.question));

  if (display.result) {
    const big = el("div", "big", display.result.value);
    big.append(el("span", "unit", display.result.unit));
    card.append(big);
  }

  const rows = el("dl", "rows");
  const conf = answer.confidence;
  const kind = conf.score >= 0.75 ? "ok" : conf.score >= 0.45 ? "warn" : "bad";
  const confText = el("span", `conf-${conf.label}`, `${conf.label} · ${conf.score.toFixed(2)}`);
  row(rows, "confidence", meter(conf.score, kind), confText);

  if (answer.status === "refused" && answer.redirect) {
    row(rows, "why", el("span", "", display.why || answer.explanation.split(REDIRECT_MARKER)[0]));
    const redirect = el("button", "redirect", answer.redirect);
    redirect.type = "button";
    redirect.title = "Ask this instead";
    redirect.addEventListener("click", () => submit(answer.redirect));
    row(rows, "try instead", redirect);
  } else {
    row(rows, "explanation", el("span", "", answer.explanation));
  }

  if (display.equations.length) {
    const box = el("div");
    for (const eq of display.equations) {
      const line = el("div");
      line.append(el("span", "eq-id", eq.id), el("span", "eq-name", eq.name));
      if (eq.math) line.append(el("span", "eq-math", eq.math));
      if (eq.source) line.append(el("span", "eq-src", `source: ${eq.source}`));
      box.append(line);
    }
    row(rows, "equation", box);
  }

  if (display.inputs.length) {
    const box = el("div");
    for (const k of display.inputs) {
      const line = el("div", "input-row");
      line.append(
        el("span", "sym", `${k.symbol} = `),
        el("span", `origin-${k.origin}`, `${k.value} ${k.unit}`),
        el("span", "tag", k.origin),
      );
      box.append(line);
    }
    row(rows, "inputs", box);
  }

  if (answer.assumptions.length) row(rows, "assumes", list(answer.assumptions));
  if (answer.caveats.length) row(rows, "caveats", list(answer.caveats, "caveats"));

  card.append(rows);
  return card;
}

async function submit(question) {
  question = question.trim();
  if (!question) return;
  input.value = question;
  button.disabled = true;
  button.textContent = "Thinking…";
  result.replaceChildren(el("p", "engine booting", "Solving…"));
  try {
    const payload = await askEngine(question);
    result.replaceChildren(renderCard(payload));
  } catch (error) {
    result.replaceChildren(el("p", "error", `Something broke: ${error.message}. Reload the page to retry.`));
  } finally {
    button.disabled = false;
    button.textContent = "Ask";
  }
}

form.addEventListener("submit", (event) => {
  event.preventDefault();
  submit(input.value || input.placeholder);
});

for (const chip of document.querySelectorAll("#examples button")) {
  chip.addEventListener("click", () => submit(chip.textContent));
}

// Boot the ~15 MB engine only once the visitor shows interest.
input.addEventListener("focus", startEngine, { once: true });
new IntersectionObserver((entries, observer) => {
  if (entries.some((e) => e.isIntersecting)) {
    startEngine();
    observer.disconnect();
  }
}, { rootMargin: "200px" }).observe(document.getElementById("ask"));

for (const copy of document.querySelectorAll(".copy")) {
  copy.addEventListener("click", async () => {
    const text = copy.previousElementSibling.textContent;
    try {
      await navigator.clipboard.writeText(text);
      copy.textContent = "Copied";
    } catch {
      copy.textContent = "Select it";
    }
    setTimeout(() => (copy.textContent = "Copy"), 1500);
  });
}
