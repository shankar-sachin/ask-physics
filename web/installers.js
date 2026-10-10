// The install pages: Copy buttons on every command, and (on /more-installers) the table of
// releases, fetched from the GitHub API in the browser. Everything that comes from the API goes
// in through textContent or setAttribute, never innerHTML, and links are only made for URLs on
// github.com.

const REPO = "shankar-sachin/ask-physics";
const RELEASES_API = `https://api.github.com/repos/${REPO}/releases?per_page=100`;
const RELEASES_PAGE = `https://github.com/${REPO}/releases`;
const SITE = "https://askphysics.vercel.app";

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = text;
  return node;
}

// Copy buttons: they copy the text of the element just before them.
function wireCopy(root) {
  for (const copy of root.querySelectorAll(".copy")) {
    if (copy.dataset.wired) continue;
    copy.dataset.wired = "1";
    copy.addEventListener("click", async () => {
      const source = copy.previousElementSibling;
      const text = source.dataset.copy || source.textContent;
      try {
        await navigator.clipboard.writeText(text.trim());
        copy.textContent = "Copied";
      } catch {
        const range = document.createRange();
        range.selectNodeContents(source);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
        copy.textContent = "Select it";
      }
      setTimeout(() => (copy.textContent = "Copy"), 1500);
    });
  }
}
wireCopy(document);

function size(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1000) return `${bytes} B`;
  if (bytes < 1e6) return `${(bytes / 1e3).toFixed(0)} KB`;
  if (bytes < 1e9) return `${(bytes / 1e6).toFixed(1)} MB`;
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

function safeGithub(url) {
  return typeof url === "string" && url.startsWith("https://github.com/") ? url : null;
}

function link(href, text) {
  const a = el("a", "", text);
  a.href = href;
  a.rel = "noopener";
  return a;
}

function command(text) {
  const row = el("div", "cmd");
  row.append(el("code", "", text), el("button", "copy", "Copy"));
  row.lastChild.type = "button";
  return row;
}

// What an asset is, from its file name.
function kindOf(name) {
  if (/\.sha256$/i.test(name)) return "sha256";
  if (/^AskPhysicsSetup(-[\w.]+)?\.exe$/i.test(name)) {
    return /-/.test(name) ? "Windows installer (this version)" : "Windows installer (latest alias)";
  }
  if (/winget/i.test(name)) return "WinGet manifests";
  return "";
}

function isModels(release) {
  return release.tag_name.startsWith("models-");
}

function row(release, isLatest) {
  const tag = release.tag_name;
  const models = isModels(release);
  const tr = el("tr", isLatest ? "latest-row" : "");

  const ver = el("td", "ver ver-cell");
  const a = link(safeGithub(release.html_url) || `${RELEASES_PAGE}/tag/${encodeURIComponent(tag)}`, tag);
  ver.append(a);
  if (isLatest) ver.append(el("span", "tag latest", "latest"));
  if (models) ver.append(el("span", "tag models", "models"));
  if (release.prerelease) ver.append(el("span", "tag pre", "pre-release"));

  const date = el("td", "date date-cell", (release.published_at || release.created_at || "").slice(0, 10));

  const notes = el("td", "");
  notes.dataset.label = "Notes";
  notes.append(link(safeGithub(release.html_url) || RELEASES_PAGE, "Release notes"));

  const assets = el("td", "");
  assets.dataset.label = "Downloads";
  const list = el("ul", "assets");
  for (const asset of release.assets || []) {
    const href = safeGithub(asset.browser_download_url);
    if (!href) continue;
    const li = el("li");
    li.append(link(href, asset.name), el("span", "size", size(asset.size)));
    const kind = kindOf(asset.name);
    if (kind) li.append(el("span", "kind", kind));
    list.append(li);
  }
  if (!models) {
    const base = `https://github.com/${REPO}/archive/refs/tags/${encodeURIComponent(tag)}`;
    for (const [suffix, label] of [[".tar.gz", "Source (tar.gz)"], [".zip", "Source (zip)"]]) {
      const li = el("li");
      li.append(link(base + suffix, `${tag}${suffix}`), el("span", "kind", label));
      list.append(li);
    }
  }
  assets.append(list);

  const install = el("td", "");
  install.dataset.label = "Install";
  if (models) {
    install.append(el("span", "kind", "Model weights: askphysics model pull"));
  } else {
    const details = el("details");
    details.append(el("summary", "", `Install ${tag}`));
    details.append(
      command(`curl -fsSL ${SITE}/install.sh | ASKPHYSICS_REF=${tag} bash`),
      command(`$env:ASKPHYSICS_REF = "${tag}"; irm ${SITE}/install.ps1 | iex`),
    );
    install.append(details);
  }
  tr.append(ver, date, notes, assets, install);
  return tr;
}

function renderReleases(releases, mount) {
  const shown = releases.filter((r) => !r.draft);
  if (!shown.length) throw new Error("no releases");
  const latest = shown.find((r) => !isModels(r) && !r.prerelease);
  const code = shown.filter((r) => !isModels(r));
  const models = shown.filter(isModels);

  const table = (rows) => {
    const wrap = el("div", "tbl-wrap stack");
    const t = el("table", "rels");
    const head = el("thead");
    const hr = el("tr");
    for (const h of ["Version", "Date", "Notes", "Downloads", "Install"]) hr.append(el("th", "", h));
    head.append(hr);
    const body = el("tbody");
    for (const r of rows) body.append(row(r, r === latest));
    t.append(head, body);
    wrap.append(t);
    return wrap;
  };

  mount.replaceChildren(table(code));
  if (models.length) {
    mount.append(el("h3", "", "Model releases"), el("p", "note",
      "Weights for the Fermi models, published as release assets. The installers and " +
      "askphysics model pull fetch them for you."), table(models));
  }
  wireCopy(mount);
}

const mount = document.getElementById("versions");
if (mount) {
  const fail = (why) => {
    const box = el("div", "state error");
    box.append(el("b", "", "Couldn't load the release list. "), document.createTextNode(`${why} `));
    box.append(link(RELEASES_PAGE, "See every release on GitHub"));
    box.append(document.createTextNode("."));
    mount.replaceChildren(box);
  };
  try {
    const response = await fetch(RELEASES_API, { headers: { Accept: "application/vnd.github+json" } });
    if (!response.ok) {
      fail(response.status === 403 ? "GitHub's rate limit for this network is used up." : `GitHub answered ${response.status}.`);
    } else {
      renderReleases(await response.json(), mount);
    }
  } catch (error) {
    fail("The request failed.");
  }
}
