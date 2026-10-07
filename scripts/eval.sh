#!/bin/sh
# Score a Fermi model on held-out questions, optionally against another copy of it.
#
#   sh scripts/eval.sh fermi-solem-1
#   sh scripts/eval.sh fermi-solem-1 --examples 200
#   sh scripts/eval.sh fermi-solem-1 --against ~/askphysics-backup/fermi-solem-1-20261007-0412
#
# Options:
#   --examples N     questions per task (default 3000)
#   --data DIR       default build/data
#   --directory DIR  score this model directory instead of the installed one
#   --against DIR    also score this directory on the same questions, then compare
#
# Each report is saved as eval.json in the model's directory.
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help | "") usage 0 ;;
  -*) usage 1 ;;
esac
model=$1
shift
examples=3000 data=build/data directory="" against=""
while [ $# -gt 0 ]; do
  case $1 in
    --examples) examples=$2 && shift ;;
    --data) data=$2 && shift ;;
    --directory) directory=$2 && shift ;;
    --against) against=$2 && shift ;;
    -h | --help) usage 0 ;;
    *) fail "unknown option $1 (see --help)" ;;
  esac
  shift
done
to_repo_root
need_askphysics
directory=${directory:-$(models_dir)/$model}

if [ -n "$against" ]; then
  info "Scoring $against"
  awake askphysics model eval --model "$model" --directory "$against" --data "$data" \
    --examples "$examples"
fi
info "Scoring $directory"
awake askphysics model eval --model "$model" --directory "$directory" --data "$data" \
  --examples "$examples"
if [ -n "$against" ]; then
  run python3 "$(dirname "$0")/compare_evals.py" "$against/eval.json" "$directory/eval.json"
fi
