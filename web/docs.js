/*
 * Ask Physics docs pages: browser behavior for /docs/.
 * Renders .math blocks with KaTeX, adds the "Copy page" button, the mobile
 * sidebar toggle, the "On this page" scroll-spy, and search over
 * /docs/search.json. Each feature runs on its own so a failure in one does
 * not stop the others. DOM is built with createElement and textContent only.
 */
(function () {
  "use strict";

  function safe(name, fn) {
    try {
      fn();
    } catch (err) {
      console.error("docs.js: " + name + " failed", err);
    }
  }

  function renderMath() {
    if (!window.katex) return;
    document.querySelectorAll(".math").forEach((el) => {
      const source = el.textContent;
      try {
        window.katex.render(source, el, { displayMode: true, throwOnError: false });
      } catch (err) {
        el.textContent = source;
      }
    });
  }

  function setupCopy() {
    const LABEL = "Copy page";
    document.querySelectorAll(".copy-page").forEach((btn) => {
      let timer = null;

      function flash(text, ok) {
        btn.textContent = text;
        btn.classList.toggle("done", ok);
        clearTimeout(timer);
        timer = setTimeout(() => {
          btn.textContent = LABEL;
          btn.classList.remove("done");
        }, 1500);
      }

      btn.addEventListener("click", () => {
        fetch(btn.getAttribute("data-src"), { credentials: "same-origin" })
          .then((r) => {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.text();
          })
          .then((text) => navigator.clipboard.writeText(text))
          .then(
            () => flash("Copied", true),
            () => flash("Couldn't copy", false)
          );
      });
    });
  }

  function setupMenu() {
    const btn = document.querySelector(".docs-menu");
    const side = document.getElementById("docs-side");
    if (!btn || !side) return;
    btn.addEventListener("click", () => {
      const open = side.classList.toggle("open");
      btn.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  function setupScrollSpy() {
    const toc = document.querySelector(".docs-toc");
    if (!toc || !("IntersectionObserver" in window)) return;

    const targets = [];
    toc.querySelectorAll("a[href^='#']").forEach((link) => {
      const id = link.getAttribute("href").slice(1);
      const el = id ? document.getElementById(id) : null;
      if (el) targets.push({ el, link, hit: false });
    });
    if (!targets.length) return;

    let current = null;

    function update() {
      // Prefer the first heading currently in the observed band; otherwise
      // fall back to the last heading that has scrolled above it.
      let active = targets.find((t) => t.hit) || null;
      if (!active) {
        targets.forEach((t) => {
          if (t.el.getBoundingClientRect().top < 80) active = t;
        });
      }
      if (active === current) return;
      current = active;
      toc.querySelectorAll("a.here").forEach((a) => a.classList.remove("here"));
      if (active) active.link.classList.add("here");
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const t = targets.find((x) => x.el === entry.target);
          if (t) t.hit = entry.isIntersecting;
        });
        update();
      },
      { rootMargin: "-80px 0px -70% 0px" }
    );
    targets.forEach((t) => observer.observe(t.el));
    update();
  }

  function setupSearch() {
    const input = document.getElementById("docs-q");
    const list = document.getElementById("docs-results");
    if (!input || !list) return;
    const box = input.closest(".docs-search") || input.parentNode;

    let indexPromise = null;
    let items = [];
    let active = -1;
    let timer = null;
    let seq = 0;

    function lower(v) {
      return typeof v === "string" ? v.toLowerCase() : "";
    }

    function prepare(data) {
      if (!Array.isArray(data)) return [];
      const pages = [];
      data.forEach((p) => {
        if (!p || typeof p.t !== "string" || typeof p.u !== "string") return;
        const heads = Array.isArray(p.h) ? p.h.filter((h) => typeof h === "string") : [];
        pages.push({
          t: p.t,
          u: p.u,
          s: typeof p.s === "string" ? p.s : "",
          heads,
          lt: p.t.toLowerCase(),
          lh: heads.map((h) => h.toLowerCase()),
          lx: lower(p.x),
        });
      });
      return pages;
    }

    function loadIndex() {
      if (!indexPromise) {
        indexPromise = fetch("/docs/search.json", { credentials: "same-origin" })
          .then((r) => {
            if (!r.ok) throw new Error("HTTP " + r.status);
            return r.json();
          })
          // Only links into the docs, whatever the index says.
          .then((pages) => pages.filter((p) => typeof p.u === "string" && p.u.startsWith("/docs/")))
          .then(prepare);
        // Allow a retry after a failed load.
        indexPromise.catch(() => {
          indexPromise = null;
        });
      }
      return indexPromise;
    }

    function rank(pages, q) {
      const words = q.split(/\s+/);
      const scored = [];
      pages.forEach((p) => {
        let score = 0;
        let headIdx = -1;
        const ok = words.every((w) => {
          const inTitle = p.lt.includes(w);
          const headMatch = p.lh.findIndex((h) => h.includes(w));
          const inExcerpt = p.lx.includes(w);
          if (!inTitle && headMatch === -1 && !inExcerpt) return false;
          if (inTitle) score += 10;
          if (headMatch !== -1) score += 4;
          if (inExcerpt) score += 1;
          if (!inTitle && headMatch !== -1 && headIdx === -1) headIdx = headMatch;
          return true;
        });
        if (!ok) return;
        if (p.lt.startsWith(q)) score += 5;
        scored.push({ page: p, score, headIdx });
      });
      scored.sort((a, b) => b.score - a.score || a.page.t.length - b.page.t.length);
      return scored.slice(0, 8);
    }

    function clearList() {
      while (list.firstChild) list.removeChild(list.firstChild);
    }

    function hide() {
      list.hidden = true;
      clearList();
      items = [];
      active = -1;
    }

    function setActive(i) {
      if (!items.length) return;
      if (active >= 0) {
        items[active].li.classList.remove("active");
        items[active].li.setAttribute("aria-selected", "false");
      }
      const n = items.length;
      active = ((i % n) + n) % n;
      items[active].li.classList.add("active");
      items[active].li.setAttribute("aria-selected", "true");
    }

    function render(results, label) {
      clearList();
      items = [];
      active = -1;
      if (!results.length) {
        const empty = document.createElement("li");
        empty.className = "empty";
        empty.textContent = 'No pages match "' + label + '"';
        list.appendChild(empty);
      } else {
        results.forEach((r) => {
          const p = r.page;
          const li = document.createElement("li");
          li.setAttribute("role", "option");
          li.setAttribute("aria-selected", "false");

          const a = document.createElement("a");
          a.setAttribute("href", p.u);

          const title = document.createElement("span");
          title.className = "r-title";
          title.textContent = p.t;

          const meta = document.createElement("span");
          meta.className = "r-meta";
          const parts = [p.s];
          if (r.headIdx !== -1) parts.push(p.heads[r.headIdx]);
          meta.textContent = parts.filter(Boolean).join(" › ");

          a.append(title, meta);
          li.appendChild(a);
          list.appendChild(li);
          items.push({ li, href: p.u });
        });
      }
      list.hidden = false;
    }

    function run(raw) {
      const label = raw.trim();
      const q = label.toLowerCase();
      const mine = ++seq;
      if (!q) {
        hide();
        return;
      }
      loadIndex().then(
        (pages) => {
          if (mine === seq) render(rank(pages, q), label);
        },
        () => {
          if (mine === seq) hide();
        }
      );
    }

    input.addEventListener("focus", () => {
      loadIndex().catch(() => {});
      if (input.value.trim()) run(input.value);
    });

    input.addEventListener("input", () => {
      loadIndex().catch(() => {});
      clearTimeout(timer);
      timer = setTimeout(() => run(input.value), 80);
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") {
        if (!items.length) return;
        e.preventDefault();
        setActive(active + 1);
      } else if (e.key === "ArrowUp") {
        if (!items.length) return;
        e.preventDefault();
        setActive(active <= 0 ? items.length - 1 : active - 1);
      } else if (e.key === "Enter") {
        if (!items.length) return;
        e.preventDefault();
        window.location.href = items[active >= 0 ? active : 0].href;
      } else if (e.key === "Escape") {
        clearTimeout(timer);
        input.value = "";
        hide();
        input.blur();
      }
    });

    document.addEventListener("click", (e) => {
      if (!box.contains(e.target)) hide();
    });

    document.addEventListener("keydown", (e) => {
      const t = e.target;
      const typing =
        t instanceof HTMLElement &&
        (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      const slash = e.key === "/" && !typing && !e.ctrlKey && !e.metaKey && !e.altKey;
      const ctrlK =
        (e.ctrlKey || e.metaKey) && !e.altKey && (e.key === "k" || e.key === "K");
      if (slash || ctrlK) {
        e.preventDefault();
        input.focus();
      }
    });
  }

  safe("math", renderMath);
  safe("copy", setupCopy);
  safe("menu", setupMenu);
  safe("scroll-spy", setupScrollSpy);
  safe("search", setupSearch);
})();
