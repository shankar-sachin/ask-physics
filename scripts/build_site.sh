#!/bin/sh
# Build the askphysics.vercel.app static site into build/site. Vercel runs this
# (see vercel.json); run it locally to preview: sh scripts/build_site.sh [out-dir], then
# serve build/site with any static server.
set -eu

out=${1:-build/site}
rm -rf "$out"
mkdir -p "$out/installers" "$out/images" "$out/py"

# The page, the installers (copied as-is), and the artwork.
cp web/index.html web/styles.css web/app.js web/worker.js "$out/"
cp install.sh install.ps1 "$out/installers/"
cp docs/images/logo.svg docs/images/banner.png docs/images/fermi-*.jpg "$out/images/"

# The Python package the browser runs. Leave out the terminal and torch code:
# the pipeline never imports it, and Pyodide can't load torch anyway. Of the
# models package, only the torch-free modules the settings and router read are
# shipped; tests/test_site_bundle.py checks the pipeline needs nothing else.
lm_shipped="__init__.py config.py formats.py paths.py reading.py templates.py tokenizer.py"
tar -cf "$out/py/askphysics.tar" -C src \
  --exclude='__pycache__' --exclude='*.pyc' \
  --exclude='askphysics/lm' --exclude='askphysics/cli.py' \
  --exclude='askphysics/ui.py' --exclude='askphysics/__main__.py' \
  askphysics
for f in $lm_shipped; do
  tar -rf "$out/py/askphysics.tar" -C src "askphysics/lm/$f"
done
gzip -9 "$out/py/askphysics.tar"

# The docs, rendered from docs/pages/ and the equation database (scripts/docs_site.py).
# It needs Python 3.9+; without it the site still builds, just without /docs/.
if command -v python3 >/dev/null 2>&1 \
  && python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
  python3 scripts/docs_site.py "$out/docs"
else
  echo "no Python 3.9+: building the site without /docs/" >&2
fi

echo "built $out"
