#!/bin/sh
# First-time setup for working on Ask Physics: a virtualenv with everything installed.
#
#   sh scripts/setup.sh
#
# Creates .venv (Python 3.11 or newer) if it's missing, installs the package with its
# development tools, and checks the data. Each step is a spinner that turns into a tick, with
# a panel of commands to try at the end; pip's output is folded into a few dimmed lines (the
# full log is kept if a step fails). Afterwards, activate it in each new terminal:
#   source .venv/bin/activate
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

python=${PYTHON:-python3}
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || fail "$python is older than 3.11; set PYTHON to a newer one"

steps=3
[ -f .venv/bin/activate ] || steps=4
ui_begin "Setup" "A virtualenv with everything installed for working on Ask Physics" "$steps" 20
if [ ! -f .venv/bin/activate ]; then
  phase "Create .venv" "$python" -m venv .venv
fi

# The display needs Rich, which the package depends on anyway: put it in the new environment
# first (quietly), and draw from the checkout's source until the package itself is installed.
if [ "${DRY_RUN:-}" != "1" ] && ! ui_has_python; then
  .venv/bin/python -m pip install -q rich >/dev/null 2>&1 || true
  ui_python .venv/bin/python "$root/src"
fi

phase "Update pip" .venv/bin/python -m pip install --upgrade pip
phase "Install askphysics" .venv/bin/python -m pip install -e ".[dev]"
phase "Check the data" .venv/bin/askphysics validate-data

ui_end "Setup complete"
ui_ready "Ask Physics is ready" \
  "source .venv/bin/activate::in each new terminal" \
  'askphysics ask "How fast does a ball dropped from 20 m hit the ground?"::ask a question' \
  "sh scripts/check.sh::run every gate before a commit"
