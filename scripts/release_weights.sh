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
#   3. list its release files with their sizes and sha256, checked against the pins
# Then it prints what to upload and commit. It never tags, uploads, or pushes.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help | "") usage 0 ;;
  -*) usage 1 ;;
esac
release=$1
shift
to_repo_root
need_askphysics_dev

if [ $# -gt 0 ]; then
  models=$*
else
  models=""
  for name in fermi-tellus-1 fermi-solem-1 fermi-celeste-1; do
    [ -d "$(models_dir)/$name" ] && models="$models $name"
  done
fi
[ -n "$models" ] || fail "no trained models installed to package"

count=0
for _ in $models; do count=$((count + 1)); done
ui_begin "Release $release" "Package, score, and pin the weights; nothing is uploaded or pushed" \
  $((count * 2)) 28

attribution=build/corpus/ATTRIBUTION.md
for name in $models; do
  set -- askphysics-dev model package --model "$name" --release "$release"
  [ -f "$attribution" ] && set -- "$@" --attribution "$attribution"
  phase "Package $name for $release" "$@"
  phase_live "Score the packaged $name" askphysics-dev model eval --model "$name" \
    --directory "build/release/$name" --examples 3000
  # The files this model ships, each with its size and whether its sha256 matches the pin.
  if [ "${DRY_RUN:-}" = "1" ]; then
    say "-- Release files of $name"
    run python -m askphysics.shell_ui assets --model "$name" --dir build/release/assets
  elif ui_has_python; then
    ui_call assets --model "$name" --dir build/release/assets
  fi
done

ui_end "Packaged $release" "Release failed" \
  "build/release/assets/::attach every file to a GitHub release tagged $release on main" \
  "src/askphysics/lm/weights.json::open a pull request with the new pins" \
  "askphysics-dev model pull --force::after it merges, checks the published copies"
