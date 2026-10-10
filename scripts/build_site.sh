#!/bin/sh
# Build the askphysics.vercel.app static site into build/site. Vercel runs this
# (see vercel.json); run it locally to preview: sh scripts/build_site.sh [out-dir], then
# serve build/site with any static server.
#
# Three stages, each a spinner that settles into a tick with its count (files written, files
# in the bundle, pages rendered). It needs no askphysics install: without one the stages print
# as plain lines, which is what Vercel's log shows.
set -eu
. "$(dirname "$0")/lib.sh"

stage=""
case ${1:-} in
  --stage) stage=$2 out=${3:-build/site} ;;
  -h | --help) usage 0 ;;
  *) out=${1:-build/site} ;;
esac

# The page, the installers (copied as-is), and the artwork.
stage_page() {
  rm -rf "$out"
  mkdir -p "$out/installers" "$out/images" "$out/py"
  cp web/index.html web/404.html web/styles.css web/app.js web/worker.js "$out/"
  cp install.sh install.ps1 "$out/installers/"
  cp docs/images/logo.svg docs/images/banner.png docs/images/fermi-*.jpg docs/images/demo.mp4 docs/images/demo-poster.jpg "$out/images/"
  echo "$(find "$out" -type f | wc -l | tr -d ' ') files written"
}

# The Python package the browser runs. Leave out the terminal and torch code:
# the pipeline never imports it, and Pyodide can't load torch anyway. Of the
# models package, only the torch-free modules the settings and router read are
# shipped; tests/test_site_bundle.py checks the pipeline needs nothing else.
stage_bundle() {
  lm_shipped="__init__.py config.py formats.py paths.py reading.py templates.py tokenizer.py"
  tar -cf "$out/py/askphysics.tar" -C src \
    --exclude='__pycache__' --exclude='*.pyc' \
    --exclude='askphysics/lm' --exclude='askphysics/cli.py' \
    --exclude='askphysics/ui.py' --exclude='askphysics/__main__.py' \
    --exclude='askphysics/train_ui.py' --exclude='askphysics/train_view.py' \
    --exclude='askphysics/shell_ui.py' --exclude='askphysics/eval_view.py' \
    --exclude='askphysics/files_view.py' \
    askphysics
  for f in $lm_shipped; do
    tar -rf "$out/py/askphysics.tar" -C src "askphysics/lm/$f"
  done
  gzip -9 "$out/py/askphysics.tar"
  echo "$(tar -tzf "$out/py/askphysics.tar.gz" | grep -vc '/$') files in askphysics.tar.gz ($(($(wc -c <"$out/py/askphysics.tar.gz") / 1000)) KB)"
}

# The docs, rendered from docs/pages/ and the equation database (scripts/docs_site.py).
# It needs Python 3.9+; without it the site still builds, just without /docs/.
stage_docs() {
  if command -v python3 >/dev/null 2>&1 \
    && python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' 2>/dev/null; then
    python3 scripts/docs_site.py "$out/docs"
  else
    echo "no Python 3.9+: building the site without /docs/"
  fi
}

case $stage in
  page) stage_page ;;
  bundle) stage_bundle ;;
  docs) stage_docs ;;
  "")
    ui_begin "Build site" "The static site for askphysics.vercel.app, into $out" 3 18
    gate "Page and artwork" last sh "$0" --stage page "$out"
    gate "Python bundle" last sh "$0" --stage bundle "$out"
    gate "Docs pages" last sh "$0" --stage docs "$out"
    ui_end "Built $out" "Site build failed" "python3 -m http.server -d $out::preview it locally"
    ;;
  *) fail "unknown stage $stage" ;;
esac
