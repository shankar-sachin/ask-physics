#!/bin/sh
# Everything that must pass before a commit, plus a merge-conflict check.
#
#   sh scripts/check.sh                   lint, typecheck, tests, data
#   sh scripts/check.sh --fast            skip the test suite
#   sh scripts/check.sh --conflicts       also test-merge against origin/main
#   sh scripts/check.sh --conflicts origin/feat/model-pull origin/feat/solver-chaining
#
# The conflict check fetches, tries each merge without committing, and aborts it, so
# the working tree is never changed. It needs a clean working tree.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

fast=0 conflicts=0
while [ $# -gt 0 ]; do
  case $1 in
    --fast) fast=1 ;;
    --conflicts) conflicts=1 && shift && break ;;
    -h | --help) usage 0 ;;
    *) fail "unknown option $1 (see --help)" ;;
  esac
  shift
done
to_repo_root
need_askphysics

info "Lint"
run python -m ruff check .
run python -m ruff format --check .
info "Typecheck"
run python -m mypy
info "Data"
run askphysics validate-data
if [ "$fast" = 0 ]; then
  info "Tests"
  run python -m pytest -q
fi

if [ "$conflicts" = 1 ]; then
  branches=${*:-origin/main}
  if [ "${DRY_RUN:-}" != "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    fail "commit or stash your changes before the conflict check"
  fi
  # Git's own messages are noise here; DRY_RUN still prints each command.
  quiet() {
    if [ "${DRY_RUN:-}" = "1" ]; then run "$@"; else "$@" >/dev/null 2>&1; fi
  }
  run git fetch -q origin
  clashes=0
  for branch in $branches; do
    if quiet git merge --no-commit --no-ff -q "$branch"; then
      say "  $branch: merges cleanly"
    else
      say "  $branch: CONFLICTS"
      clashes=1
    fi
    quiet git merge --abort || true
  done
  [ "$clashes" = 0 ] || fail "merge conflicts; resolve them before pushing"
fi
info "All clear"
