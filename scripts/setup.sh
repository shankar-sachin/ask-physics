#!/bin/sh
# First-time setup for working on Ask Physics: a virtualenv with everything installed.
#
#   sh scripts/setup.sh
#
# Creates .venv (Python 3.11 or newer) if it's missing, installs the package with its
# development tools, and checks the data. Afterwards, activate it in each new terminal:
#   source .venv/bin/activate
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

python=${PYTHON:-python3}
"$python" -c 'import sys; sys.exit(sys.version_info < (3, 11))' \
  || fail "$python is older than 3.11; set PYTHON to a newer one"
if [ ! -f .venv/bin/activate ]; then
  info "Creating .venv"
  run "$python" -m venv .venv
fi
info "Installing askphysics and its development tools"
run .venv/bin/python -m pip install --upgrade pip
run .venv/bin/python -m pip install -e ".[dev]"
info "Checking the data"
run .venv/bin/askphysics validate-data
say ""
say "Ready. In each new terminal: source .venv/bin/activate"
