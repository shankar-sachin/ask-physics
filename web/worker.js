// Runs the real askphysics package in the browser with Pyodide (Python on
// WebAssembly), in a worker so the page never freezes while Python boots or
// solves. It's a module worker: Pyodide 314 no longer supports classic ones.
// Messages in: {type: "boot"} and {type: "ask", id, question}.
// Messages out: status, ready, answer, and error.

const PYODIDE_VERSION = "314.0.7";
const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
// Pure-Python dependencies that Pyodide doesn't bundle, pinned to what CI tests.
const PYPI_PACKAGES = ["pint==0.26.1", "flexcache==0.3", "flexparser==0.4"];
const PACKAGE_DIR = "/home/pyodide/askphysics-src";

let booting = null;

function status(step, total, text) {
  postMessage({ type: "status", step, total, text });
}

async function boot() {
  status(1, 4, "Starting Python");
  const { loadPyodide } = await import(`${PYODIDE_URL}pyodide.mjs`);
  const pyodide = await loadPyodide({ indexURL: PYODIDE_URL });

  status(2, 4, "Loading SymPy, pydantic, and numpy");
  await pyodide.loadPackage(["pydantic", "sympy", "numpy", "micropip"]);

  status(3, 4, "Installing Pint");
  const micropip = pyodide.pyimport("micropip");
  await micropip.install(PYPI_PACKAGES);

  status(4, 4, "Loading Ask Physics");
  const response = await fetch("py/askphysics.tar.gz");
  if (!response.ok) throw new Error(`could not fetch the askphysics package (${response.status})`);
  pyodide.FS.mkdirTree(PACKAGE_DIR);
  pyodide.unpackArchive(await response.arrayBuffer(), "gztar", { extractDir: PACKAGE_DIR });
  pyodide.runPython(`
import sys
sys.path.insert(0, ${JSON.stringify(PACKAGE_DIR)})
import askphysics.web as askphysics_web
`);
  const info = JSON.parse(pyodide.runPython("askphysics_web.info()"));
  postMessage({ type: "ready", info });
  return pyodide.globals.get("askphysics_web").ask;
}

async function engine() {
  if (!booting) booting = boot();
  try {
    return await booting;
  } catch (error) {
    booting = null; // the next message retries from scratch
    throw error;
  }
}

self.onmessage = async (event) => {
  const message = event.data;
  try {
    const ask = await engine();
    if (message.type === "ask") {
      const payload = JSON.parse(ask(message.question));
      postMessage({ type: "answer", id: message.id, payload });
    }
  } catch (error) {
    postMessage({ type: "error", id: message.id, message: String((error && error.message) || error) });
  }
};
