#!/bin/sh
# Bring a checkout up to date with main after a pull request merges.
#
#   sh scripts/update.sh
#
# Switches to main, pulls, reinstalls (new dependencies or entry points take effect), and
# shows the commit you are now on. Refuses if you have uncommitted changes.
#
# DRY_RUN=1 prints every command instead of running it.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help) usage 0 ;;
  "") ;;
  *) usage 1 ;;
esac
to_repo_root
need_askphysics_dev

if [ "${DRY_RUN:-}" != "1" ] && [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  fail "you have uncommitted changes; commit or stash them first"
fi
ui_begin "Update" "Bring this checkout up to date with main" 3 20
phase "Switch to main" git checkout main
phase "Pull main" git pull --ff-only origin main
phase "Reinstall askphysics" python -m pip install -q -e ".[dev]"

if [ "${DRY_RUN:-}" = "1" ]; then
  run git log --oneline -1
else
  info "Now at $(git log --oneline -1)"
fi
ui_end "Up to date" "sh scripts/check.sh::run every gate"
