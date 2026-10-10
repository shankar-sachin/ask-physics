#!/bin/sh
# Everything that must pass before a commit, plus a merge-conflict check.
#
#   sh scripts/check.sh                   lint, typecheck, shell scripts, data, tests
#   sh scripts/check.sh --fast            skip the test suite
#   sh scripts/check.sh --conflicts       also test-merge against origin/main
#   sh scripts/check.sh --conflicts origin/feat/model-pull origin/feat/solver-chaining
#
# Each gate is one live line that turns into a tick or a cross, with pytest's counts. A failed
# gate shows its command and the end of its output (the full log is saved), and the rest still
# run. The conflict check fetches, tries each merge without committing, and aborts it, so
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

branches=${*:-origin/main}
steps=5
[ "$fast" = 1 ] || steps=$((steps + 1))
if [ "$conflicts" = 1 ]; then
  steps=$((steps + 1))
  for _branch in $branches; do steps=$((steps + 1)); done
fi
ui_begin "Check" "Everything that must pass before a commit" "$steps" 13

gate "ruff check" last python -m ruff check .
gate "ruff format" last python -m ruff format --check .
gate "mypy" last python -m mypy
if [ "${DRY_RUN:-}" = "1" ] || command -v shellcheck >/dev/null 2>&1; then
  ui_ok "no findings"
  gate "shellcheck" last shellcheck install.sh scripts/*.sh
else
  ui_result skip "shellcheck" "not installed"
fi
gate "validate-data" first askphysics validate-data
if [ "$fast" = 0 ]; then
  gate "pytest" pytest python -m pytest -q
fi

if [ "$conflicts" = 1 ]; then
  if [ "${DRY_RUN:-}" != "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    fail "commit or stash your changes before the conflict check"
  fi
  # Git's own messages are noise here; DRY_RUN still prints each command.
  quiet() {
    if [ "${DRY_RUN:-}" = "1" ]; then run "$@"; else "$@" >/dev/null 2>&1; fi
  }
  phase "git fetch" git fetch -q origin
  for branch in $branches; do
    if quiet git merge --no-commit --no-ff -q "$branch"; then
      ui_result ok "$branch" "merges cleanly"
    else
      ui_result fail "$branch" "conflicts"
    fi
    quiet git merge --abort || true
  done
fi
ui_end "All clear" "Not ready to commit" "git push -u origin HEAD::publish the branch"
