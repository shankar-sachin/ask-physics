#!/bin/sh
# Build the 20M-word prose corpus (ADR-017) with the Mac kept awake.
#
#   sh scripts/corpus.sh
#   sh scripts/corpus.sh --target-words 5000000
#
# Downloads the OpenStax repositories and public-domain books once (into
# build/corpus-cache) and writes build/corpus/. Safe to rerun after an interruption:
# finished downloads are reused. Extra options go to scripts/build_corpus.py.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help) usage 0 ;;
esac
to_repo_root
need_askphysics

info "Building the prose corpus (prints a line per book; quiet stretches are downloads)"
awake python scripts/build_corpus.py "$@"
