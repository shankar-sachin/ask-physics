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
# The OpenStax stage settles into a line of its own (books and words); the Gutenberg stage
# shows its words so far against the target. The full output is kept in a log file.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help) usage 0 ;;
esac
to_repo_root
need_askphysics

ui_begin "Corpus" "The prose corpus for solem and celeste (ADR-017); quiet stretches are downloads" 1
gate "Build the corpus" corpus python scripts/build_corpus.py "$@"
ui_end "Corpus built" "Corpus failed" "sh scripts/train.sh fermi-solem-1::train on it"
