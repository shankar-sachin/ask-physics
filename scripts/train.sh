#!/bin/sh
# Train a Fermi model end to end, safely, with the Mac kept awake.
#
#   sh scripts/train.sh fermi-solem-1
#   sh scripts/train.sh fermi-tellus-1 --steps 1500
#   sh scripts/train.sh fermi-solem-1 --resume        after a crash: pick up where it stopped
#
# Steps, each only after the one before it worked:
#   1. back up the installed model (scripts/models.sh), unless resuming
#   2. build the training data, if missing (or --fresh-data)
#   3. train the tokenizer, if missing (or --fresh-tokenizer), on the corpus when built
#   4. score the backed-up model, so there is a baseline on the same data
#   5. train (solem and celeste read the prose corpus first, when it is built)
#   6. score the new model and compare it with the baseline
#
# Options:
#   --steps N            total steps, prose included (tellus 1500; solem and celeste
#                        5000: 2000 prose-only, then 3000 on the tasks)
#   --batch-size N       default 32
#   --data DIR           default build/data
#   --examples N         examples when building data (default 1000000)
#   --workers N          processes when building data (default 10)
#   --tokenizer FILE     default build/tokenizer.json
#   --prose FILE         prose to read first (default build/corpus/prose.jsonl if built)
#   --prose-steps N      prose-only steps first (default 2000 with the corpus); must be
#                        fewer than --steps, or the model never trains on the tasks
#   --prose-share F      share of later batches on prose (default 0.1 with the corpus)
#   --no-prose           train on the tasks only
#   --fresh-data         rebuild the training data even if it exists
#   --fresh-tokenizer    retrain the tokenizer even if it exists
#   --resume             continue an interrupted run (no backup, no rebuilds)
#   --eval-examples N    questions per task when scoring (default 3000)
#   --no-eval            skip both evals
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help | "") usage 0 ;;
  -*) usage 1 ;;
esac
model=$1
shift

case $model in
  fermi-tellus-1) steps=1500 wants_prose=0 ;;
  fermi-solem-1 | fermi-celeste-1) steps=5000 wants_prose=1 ;;
  fermi-luna-1) steps=300 wants_prose=0 ;;
  *) fail "unknown model $model (fermi-tellus-1, fermi-solem-1, fermi-celeste-1, fermi-luna-1)" ;;
esac
batch=32 data=build/data examples=1000000 workers=10 tokenizer=build/tokenizer.json
prose="" prose_steps="" prose_share="" fresh_data=0 fresh_tokenizer=0 resume=0
eval_examples=3000 evaluate=1

while [ $# -gt 0 ]; do
  case $1 in
    --steps) steps=$2 && shift ;;
    --batch-size) batch=$2 && shift ;;
    --data) data=$2 && shift ;;
    --examples) examples=$2 && shift ;;
    --workers) workers=$2 && shift ;;
    --tokenizer) tokenizer=$2 && shift ;;
    --prose) prose=$2 wants_prose=1 && shift ;;
    --prose-steps) prose_steps=$2 && shift ;;
    --prose-share) prose_share=$2 && shift ;;
    --no-prose) wants_prose=0 ;;
    --fresh-data) fresh_data=1 ;;
    --fresh-tokenizer) fresh_tokenizer=1 ;;
    --resume) resume=1 ;;
    --eval-examples) eval_examples=$2 && shift ;;
    --no-eval) evaluate=0 ;;
    -h | --help) usage 0 ;;
    *) fail "unknown option $1 (see --help)" ;;
  esac
  shift
done

to_repo_root
need_askphysics

corpus=build/corpus/prose.jsonl
if [ "$wants_prose" = 1 ] && [ -z "$prose" ]; then
  if [ -f "$corpus" ]; then
    prose=$corpus
  else
    warn "no prose corpus at $corpus (build it with: sh scripts/corpus.sh); training on the tasks only"
    wants_prose=0
  fi
fi
if [ "$wants_prose" = 1 ]; then
  prose_steps=${prose_steps:-2000}
  prose_share=${prose_share:-0.1}
  # Prose-only steps come first: as many as the whole run would never train the tasks.
  if [ "$prose_steps" -ge "$steps" ]; then
    fail "--prose-steps ($prose_steps) must be fewer than --steps ($steps), or the model never trains on the tasks"
  fi
fi

installed="$(models_dir)/$model"
baseline=""
if [ "$resume" = 1 ]; then
  if [ "$fresh_data" = 1 ] || [ "$fresh_tokenizer" = 1 ]; then
    fail "--resume can't rebuild the data or tokenizer: the run would no longer match"
  fi
  info "Resuming $model"
else
  if [ -d "$installed" ]; then
    stamp="$model-$(date +%Y%m%d-%H%M%S)"
    sh "$(dirname "$0")/models.sh" backup "$model" "$stamp"
    baseline="$(backup_dir)/$stamp"
  fi
  if [ "$fresh_data" = 1 ] || [ ! -f "$data/manifest.json" ]; then
    info "Building $examples training examples in $data"
    run rm -rf "${data:?}"
    awake askphysics model build-data --out "$data" --examples "$examples" --workers "$workers"
  fi
  if [ "$fresh_tokenizer" = 1 ] || [ ! -f "$tokenizer" ]; then
    info "Training the tokenizer"
    if [ -f "$corpus" ]; then
      awake askphysics model train-tokenizer --data "$data" --out "$tokenizer" --vocab-size 8192 \
        --prose "$corpus"
    else
      awake askphysics model train-tokenizer --data "$data" --out "$tokenizer" --vocab-size 8192
    fi
  fi
  if [ "$evaluate" = 1 ] && [ -n "$baseline" ]; then
    info "Scoring the previous $model for a baseline"
    awake askphysics model eval --model "$model" --directory "$baseline" --data "$data" \
      --examples "$eval_examples"
  fi
fi

info "Training $model ($steps steps)"
set -- askphysics model train --model "$model" --data "$data" --tokenizer "$tokenizer" \
  --steps "$steps" --batch-size "$batch"
if [ "$wants_prose" = 1 ]; then
  set -- "$@" --prose "$prose" --prose-steps "$prose_steps" --prose-share "$prose_share"
fi
[ "$resume" = 1 ] && set -- "$@" --resume
awake "$@"

if [ "$evaluate" = 1 ]; then
  info "Scoring the new $model"
  awake askphysics model eval --model "$model" --data "$data" --examples "$eval_examples"
  if [ -n "$baseline" ]; then
    run python3 "$(dirname "$0")/compare_evals.py" "$baseline/eval.json" "$installed/eval.json"
    say "Keep the new one, or put the old one back: sh scripts/models.sh restore $model"
  fi
fi
info "Done: $model"
