#!/bin/sh
# Package trained models for a weights release (ADR-012, docs/RELEASING.md).
#
#   sh scripts/release_weights.sh models-v0.4.0
#   sh scripts/release_weights.sh models-v0.4.0 fermi-tellus-1 fermi-solem-1
#
# For each model (default: every one installed of tellus, solem, celeste):
#   1. package it: bf16 weights, model card, pinned in src/askphysics/lm/weights.json,
#      with the corpus attribution when build/corpus/ATTRIBUTION.md exists
#   2. score the packaged copy, so the card's numbers are what actually ships
# Then it prints what to upload and commit. It never tags, uploads, or pushes.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help | "") usage ;;
  -*) usage 1 ;;
esac
release=$1
shift
to_repo_root
need_askphysics

if [ $# -gt 0 ]; then
  models=$*
else
  models=""
  for name in fermi-tellus-1 fermi-solem-1 fermi-celeste-1; do
    [ -d "$(models_dir)/$name" ] && models="$models $name"
  done
fi
[ -n "$models" ] || fail "no trained models installed to package"

attribution=build/corpus/ATTRIBUTION.md
for name in $models; do
  info "Packaging $name for $release"
  set -- askphysics model package --model "$name" --release "$release"
  [ -f "$attribution" ] && set -- "$@" --attribution "$attribution"
  awake "$@"
  info "Scoring the packaged $name"
  awake askphysics model eval --model "$name" --directory "build/release/$name" --examples 3000
done

say ""
say "Next, by hand:"
say "  1. On GitHub, draft a release tagged $release on main and attach every file in"
say "     build/release/assets/"
say "  2. Open a pull request with src/askphysics/lm/weights.json"
say "  3. After it merges: askphysics model pull --force (checks the published copies)"
