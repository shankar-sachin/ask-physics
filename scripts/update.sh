#!/bin/sh
# Bring a checkout up to date with main after a pull request merges.
#
#   sh scripts/update.sh
#
# Switches to main, pulls, reinstalls (new dependencies or entry points take effect), and
# prints the commit you are now on. Refuses if you have uncommitted changes.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help) usage ;;
  "") ;;
  *) usage 1 ;;
esac
to_repo_root
need_askphysics

if [ "${DRY_RUN:-}" != "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "you have uncommitted changes; commit or stash them first"
fi
info "Updating main"
run git checkout main
run git pull --ff-only origin main
info "Reinstalling"
run python -m pip install -q -e ".[dev]"
run git log --oneline -1
