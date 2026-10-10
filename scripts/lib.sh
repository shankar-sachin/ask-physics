# Shared helpers for the workflow scripts. Sourced, never run:  . "$(dirname "$0")/lib.sh"
#
# DRY_RUN=1      print each command instead of running it (the tests use this)
# NO_CAFFEINATE=1  don't wrap long steps in macOS caffeinate
# shellcheck shell=sh

say() { printf '%s\n' "$*"; }
info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
fail() {
  printf '\033[1;31mxx\033[0m %s\n' "$*" >&2
  exit 1
}

# Run a command, or print it under DRY_RUN=1. Arguments are printed shell-quoted enough
# to copy and paste.
run() {
  if [ "${DRY_RUN:-}" = "1" ]; then
    printf '+'
    for arg in "$@"; do
      case $arg in
        *[[:space:]]* | '') printf " '%s'" "$arg" ;;
        *) printf ' %s' "$arg" ;;
      esac
    done
    printf '\n'
    return 0
  fi
  "$@"
}

# Run a long command with the Mac kept awake (caffeinate), when available.
awake() {
  if [ "${NO_CAFFEINATE:-}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
    run caffeinate -dims "$@"
  else
    run "$@"
  fi
}

# Run a long step as a named phase: a spinner with the time so far while it runs, then a tick
# and how long it took (askphysics model phase). Plain lines when output is not a terminal.
# phase_live is for a command that draws its own live display, such as training.
#   phase "Building data" askphysics model build-data ...
phase() {
  _phase "" "$@"
}
phase_live() {
  _phase --inherit "$@"
}
_phase() {
  _mode=$1 _title=$2
  shift 2
  if [ "${DRY_RUN:-}" = "1" ]; then
    say "-- $_title"
    awake "$@"
    return
  fi
  if [ "${NO_CAFFEINATE:-}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
    set -- caffeinate -dims "$@"
  fi
  if [ -n "$_mode" ]; then
    askphysics model phase --title "$_title" "$_mode" -- "$@"
  else
    askphysics model phase --title "$_title" -- "$@"
  fi
}

# Go to the repository root, so every relative path (build/, .venv) means the same thing.
to_repo_root() {
  root=$(cd "$(dirname "$0")/.." && pwd)
  cd "$root" || fail "can't enter $root"
}

# Make sure askphysics is runnable, activating the repo's .venv when it isn't on PATH.
need_askphysics() {
  if ! command -v askphysics >/dev/null 2>&1 && [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
  fi
  if [ "${DRY_RUN:-}" != "1" ] && ! command -v askphysics >/dev/null 2>&1; then
    fail "askphysics isn't installed here; run: sh scripts/setup.sh"
  fi
}

# Where installed models live (same rule as askphysics: $ASKPHYSICS_MODEL_DIR wins).
models_dir() {
  printf '%s\n' "${ASKPHYSICS_MODEL_DIR:-$HOME/.cache/askphysics/models}"
}

# Where backups go: $ASKPHYSICS_BACKUP_DIR, else ~/askphysics-backup.
backup_dir() {
  printf '%s\n' "${ASKPHYSICS_BACKUP_DIR:-$HOME/askphysics-backup}"
}

# Print usage (the script's leading comment block, without the #) and exit with $1.
usage() {
  sed -n '2,/^[^#]/{/^#/s/^# \{0,1\}//p;}' "$0"
  exit "$1"
}
