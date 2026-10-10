#!/bin/sh
# Ask Physics installer for macOS and Linux.
#
#   curl -fsSL https://askphysics.vercel.app/install.sh | bash
#   (the script is plain POSIX sh, so `| sh` works too)
#
# Installs the `askphysics` command in its own isolated environment with uv
# (https://docs.astral.sh/uv/), installing uv first if it's missing. Nothing
# touches your system Python.
#
# Each step is a spinner that turns into a tick with the time it took; the full output of a
# step is kept in a log file whose path is printed if it fails. Without a terminal (CI,
# a pipe) or with NO_COLOR set, the same steps print as plain lines.
#
# Environment variables:
#   ASKPHYSICS_REF          git tag, branch, or commit to install (default: latest release)
#   ASKPHYSICS_SOURCE       install from this local path or URL instead of GitHub
#   ASKPHYSICS_SKIP_MODELS  set to 1 to skip downloading the Fermi models
#
# On Linux, torch comes from PyTorch's CPU-only index (pinned in pyproject.toml):
# the default PyPI wheel bundles about 2 GB of CUDA libraries the CLI doesn't need.
#
# Uninstall:  uv tool uninstall askphysics
set -eu

REPO="shankar-sachin/ask-physics"
PYTHON_VERSION="3.12"

# >>> renderer (the same text is in install.sh; tests/test_scripts_lib.py keeps them equal)
# The pure-sh look of the scripts, for a machine with no Python or no askphysics yet: the
# same theme, glyphs and columns as askphysics.shell_ui, without the folded output lines and
# bars. A terminal gets colour, a spinner and ticks; a pipe, a log, or NO_COLOR gets plain lines.

fb_init() {
  FB_PRETTY=0 FB_ERRTTY=0 FB_WIDTH=80 FB_UTF8=0
  if [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != dumb ]; then
    if [ -t 1 ]; then FB_PRETTY=1; fi
    if [ -t 2 ]; then FB_ERRTTY=1; fi
  fi
  case ${LC_ALL:-${LC_CTYPE:-${LANG:-}}} in
    *UTF-8* | *utf-8* | *UTF8* | *utf8*) FB_UTF8=1 ;;
  esac
  FB_COLS=${COLUMNS:-}
  [ -n "$FB_COLS" ] || FB_COLS=$(tput cols 2>/dev/null) || FB_COLS=80
  case $FB_COLS in '' | *[!0-9]*) FB_COLS=80 ;; esac
  FB_WIDTH=$FB_COLS
  [ "$FB_WIDTH" -le 100 ] || FB_WIDTH=100
  [ "$FB_WIDTH" -ge 48 ] || FB_WIDTH=48
  FB_BRAND="" FB_ACCENT="" FB_MUTED="" FB_LABEL="" FB_OK="" FB_WARN="" FB_BAD=""
  FB_VALUE="" FB_RST=""
  if [ "$FB_PRETTY" = 1 ] || [ "$FB_ERRTTY" = 1 ]; then
    esc=$(printf '\033')
    case ${COLORTERM:-} in
      truecolor | 24bit)
        FB_BRAND="${esc}[1;38;2;139;233;253m" FB_ACCENT="${esc}[38;2;189;147;249m"
        FB_MUTED="${esc}[38;2;139;147;167m"
        FB_OK="${esc}[1;38;2;80;250;123m" FB_WARN="${esc}[1;38;2;241;250;140m"
        FB_BAD="${esc}[1;38;2;255;110;110m" FB_VALUE="${esc}[1;38;2;248;248;242m"
        FB_LABEL="${esc}[1;38;2;166;172;205m"
        ;;
      *)
        case ${TERM:-} in
          *256color*)
            FB_BRAND="${esc}[1;38;5;117m" FB_ACCENT="${esc}[38;5;141m" FB_MUTED="${esc}[38;5;103m"
            FB_OK="${esc}[1;38;5;84m" FB_WARN="${esc}[1;38;5;228m"
            FB_BAD="${esc}[1;38;5;203m" FB_VALUE="${esc}[1;97m" FB_LABEL="${esc}[1;38;5;146m"
            ;;
          *)
            FB_BRAND="${esc}[1;36m" FB_ACCENT="${esc}[35m" FB_MUTED="${esc}[90m"
            FB_OK="${esc}[1;32m" FB_WARN="${esc}[1;33m" FB_BAD="${esc}[1;31m" FB_VALUE="${esc}[1;97m"
            FB_LABEL="${esc}[1;37m"
            ;;
        esac
        ;;
    esac
    FB_RST="${esc}[0m"
  fi
  if [ "$FB_UTF8" = 1 ]; then
    FB_TICK='✓' FB_CROSS='✗' FB_DASH='–' FB_BULLET='◉' FB_DOT='·' FB_ARROW='›' FB_RULE='─'
  else
    FB_TICK='+' FB_CROSS='x' FB_DASH='-' FB_BULLET='o' FB_DOT='-' FB_ARROW='>' FB_RULE='-'
  fi
}
fb_init

say() { printf '%s\n' "$*"; }
# A quiet line: an arrow on a terminal, the bare text in a log.
info() {
  if [ "$FB_PRETTY" = 1 ]; then
    printf '  %s%s%s %s\n' "$FB_ACCENT" "$FB_ARROW" "$FB_RST" "$*"
  else
    printf '%s\n' "$*"
  fi
}
warn() {
  if [ "$FB_ERRTTY" = 1 ]; then
    printf '  %s!%s %s\n' "$FB_WARN" "$FB_RST" "$*" >&2
  else
    printf 'warning: %s\n' "$*" >&2
  fi
}
fail() {
  if [ "$FB_ERRTTY" = 1 ]; then
    printf '  %s%s%s %s\n' "$FB_BAD" "$FB_CROSS" "$FB_RST" "$*" >&2
  else
    printf 'error: %s\n' "$*" >&2
  fi
  exit 1
}

# 45s, 2m 05s, 1h 03m 09s: the same wording as askphysics.train_ui.format_elapsed.
fb_fmt() {
  if [ "$1" -ge 3600 ]; then
    printf '%dh %02dm %02ds' $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60))
  elif [ "$1" -ge 60 ]; then
    printf '%dm %02ds' $(($1 / 60)) $(($1 % 60))
  else
    printf '%ds' "$1"
  fi
}

# fb_repeat CHAR COUNT: CHAR printed COUNT times.
fb_repeat() {
  fb_n=$2
  while [ "$fb_n" -gt 0 ]; do
    printf '%s' "$1"
    fb_n=$((fb_n - 1))
  done
}

# fb_row KIND MARK COUNTER TITLE DETAIL TIME: one aligned line, without the line ending.
# KIND is ok, bad, run, or skip; the time sits at the right edge, as in the Rich view.
fb_row() {
  case $1 in
    ok) fb_ms=$FB_OK fb_ts=$FB_VALUE fb_ds=$FB_MUTED ;;
    bad) fb_ms=$FB_BAD fb_ts=$FB_BAD fb_ds=$FB_BAD ;;
    skip) fb_ms=$FB_MUTED fb_ts=$FB_MUTED fb_ds=$FB_MUTED ;;
    *) fb_ms=$FB_ACCENT fb_ts=$FB_BRAND fb_ds=$FB_MUTED ;;
  esac
  printf '  %s%s%s' "$fb_ms" "$2" "$FB_RST"
  [ -z "$3" ] || printf ' %s%s%s' "$FB_MUTED" "$3" "$FB_RST"
  printf ' %s%s%s' "$fb_ts" "$4" "$FB_RST"
  if [ -n "$5" ]; then
    [ "${#4}" -ge "${FB_PAD:-0}" ] || fb_repeat ' ' $((FB_PAD - ${#4}))
    printf '  %s%s%s' "$fb_ds" "$5" "$FB_RST"
  fi
  if [ -n "$6" ]; then
    printf '\033[%dG%s%s%s' $((FB_WIDTH - ${#6} + 1)) "$FB_MUTED" "$6" "$FB_RST"
  fi
}

# fb_header TITLE ABOUT STEPS
fb_header() {
  fb_count=""
  if [ "${3:-0}" -gt 0 ]; then
    fb_count="$3 step"
    [ "$3" -eq 1 ] || fb_count="${fb_count}s"
  fi
  if [ "$FB_PRETTY" != 1 ]; then
    printf 'Ask Physics: %s%s\n' "$1" "${fb_count:+ ($fb_count)}"
    [ -z "${2:-}" ] || printf '  %s\n' "$2"
    return 0
  fi
  printf '\n  %s%s %sAsk Physics%s  %s%s%s  %s%s%s\n' "$FB_ACCENT" "$FB_BULLET" "$FB_BRAND" "$FB_RST" \
    "$FB_MUTED" "$FB_DOT" "$FB_RST" "$FB_VALUE" "$1" "$FB_RST"
  fb_sub=${2:-}
  if [ -n "$fb_count" ]; then
    fb_sub="${fb_sub}${fb_sub:+  $FB_DOT  }$fb_count"
  fi
  [ -z "$fb_sub" ] || printf '    %s%s%s\n' "$FB_MUTED" "$fb_sub" "$FB_RST"
  printf '\n'
}

# fb_step TITLE INDEX TOTAL LOG COMMAND...: run COMMAND as a named step; returns its exit code.
# INDEX and TOTAL may be empty. The output goes to LOG (a temporary file when LOG is empty)
# on a terminal, and straight through otherwise.
fb_step() {
  fb_title=$1 fb_i=$2 fb_n=$3 fb_log=$4
  shift 4
  fb_label=$fb_title fb_counter=""
  if [ -n "$fb_i" ] && [ -n "$fb_n" ]; then
    fb_label="[$fb_i/$fb_n] $fb_title"
    fb_counter="$fb_i/$fb_n"
  fi
  fb_t0=$(date +%s)
  fb_rc=0
  if [ "$FB_PRETTY" != 1 ]; then
    printf 'start: %s\n' "$fb_label"
    "$@" </dev/null || fb_rc=$?
    fb_el=$(fb_fmt $(($(date +%s) - fb_t0)))
    if [ "$fb_rc" = 0 ]; then
      printf 'done: %s (%s)\n' "$fb_label" "$fb_el"
    else
      printf 'failed: %s (exit %s, %s)\n' "$fb_label" "$fb_rc" "$fb_el"
      printf '  command: %s\n' "$*"
    fi
    return "$fb_rc"
  fi
  if [ -z "$fb_log" ]; then
    fb_log=$(mktemp "${TMPDIR:-/tmp}/askphysics-step.XXXXXX")
  fi
  mkdir -p "$(dirname "$fb_log")" 2>/dev/null || true
  "$@" >"$fb_log" 2>&1 </dev/null &
  fb_pid=$!
  # A background job ignores Ctrl-C in a script, so stop it by hand.
  trap 'kill "$fb_pid" 2>/dev/null; printf "\n"; exit 130' INT TERM
  fb_f=0
  while kill -0 "$fb_pid" 2>/dev/null; do
    if [ "$FB_UTF8" = 1 ]; then
      case $((fb_f % 10)) in
        0) fb_sp='⠋' ;; 1) fb_sp='⠙' ;; 2) fb_sp='⠹' ;; 3) fb_sp='⠸' ;; 4) fb_sp='⠼' ;;
        5) fb_sp='⠴' ;; 6) fb_sp='⠦' ;; 7) fb_sp='⠧' ;; 8) fb_sp='⠇' ;; *) fb_sp='⠏' ;;
      esac
    else
      case $((fb_f % 4)) in 0) fb_sp='-' ;; 1) fb_sp=\\ ;; 2) fb_sp='|' ;; *) fb_sp='/' ;; esac
    fi
    printf '\r\033[K'
    fb_row run "$fb_sp" "$fb_counter" "$fb_title" "" "$(fb_fmt $(($(date +%s) - fb_t0)))"
    fb_f=$((fb_f + 1))
    sleep 0.08 2>/dev/null || sleep 1
  done
  wait "$fb_pid" || fb_rc=$?
  trap - INT TERM
  fb_el=$(fb_fmt $(($(date +%s) - fb_t0)))
  printf '\r\033[K'
  if [ "$fb_rc" = 0 ]; then
    fb_row ok "$FB_TICK" "$fb_counter" "$fb_title" "" "$fb_el"
    printf '\n'
    [ "${fb_keep_log:-}" = 1 ] || rm -f "$fb_log" 2>/dev/null || true
  else
    fb_row bad "$FB_CROSS" "$fb_counter" "$fb_title" "exit $fb_rc" "$fb_el"
    printf '\n'
    printf '      %s$ %s%s\n\n' "$FB_MUTED" "$*" "$FB_RST"
    tail -n 12 "$fb_log" | while IFS= read -r fb_line; do
      printf '      %s\n' "$fb_line"
    done
    printf '\n      %sfull log%s  %s%s%s\n' "$FB_MUTED" "$FB_RST" "$FB_ACCENT" "$fb_log" "$FB_RST"
  fi
  return "$fb_rc"
}

# fb_live TITLE INDEX TOTAL COMMAND...: like fb_step, for a command that draws its own display
# (a download with its own bars): a heading, the command on the terminal, then the closing line.
fb_live() {
  fb_title=$1 fb_i=$2 fb_n=$3
  shift 3
  fb_counter=""
  [ -z "$fb_i" ] || [ -z "$fb_n" ] || fb_counter="$fb_i/$fb_n"
  fb_t0=$(date +%s)
  fb_rc=0
  if [ "$FB_PRETTY" != 1 ]; then
    printf 'start: %s\n' "${fb_counter:+[$fb_counter] }$fb_title"
    "$@" </dev/null || fb_rc=$?
    fb_el=$(fb_fmt $(($(date +%s) - fb_t0)))
    if [ "$fb_rc" = 0 ]; then
      printf 'done: %s (%s)\n' "${fb_counter:+[$fb_counter] }$fb_title" "$fb_el"
    else
      printf 'failed: %s (exit %s, %s)\n' "${fb_counter:+[$fb_counter] }$fb_title" "$fb_rc" "$fb_el"
    fi
    return "$fb_rc"
  fi
  printf '  %s%s %s%s%s\n' "$FB_ACCENT" "$FB_BULLET" "$FB_BRAND" "$fb_title" "$FB_RST"
  "$@" </dev/null || fb_rc=$?
  fb_el=$(fb_fmt $(($(date +%s) - fb_t0)))
  if [ "$fb_rc" = 0 ]; then
    fb_row ok "$FB_TICK" "$fb_counter" "$fb_title" "" "$fb_el"
  else
    fb_row bad "$FB_CROSS" "$fb_counter" "$fb_title" "exit $fb_rc" "$fb_el"
  fi
  printf '\n'
  return "$fb_rc"
}

# fb_result KIND TITLE DETAIL INDEX TOTAL: a finished line with no command behind it.
# KIND is ok, fail, or skip.
fb_result() {
  fb_label=$2
  fb_counter=""
  if [ -n "${4:-}" ] && [ -n "${5:-}" ]; then
    fb_label="[$4/$5] $2"
    fb_counter="$4/$5"
  fi
  if [ "$FB_PRETTY" != 1 ]; then
    case $1 in ok) fb_word='done' ;; fail) fb_word=failed ;; *) fb_word=skipped ;; esac
    printf '%s: %s%s\n' "$fb_word" "$fb_label" "${3:+ - $3}"
    return 0
  fi
  case $1 in
    ok) fb_row ok "$FB_TICK" "$fb_counter" "$2" "${3:-}" "" ;;
    fail) fb_row bad "$FB_CROSS" "$fb_counter" "$2" "${3:-}" "" ;;
    *) fb_row skip "$FB_DASH" "$fb_counter" "$2" "${3:-}" "" ;;
  esac
  printf '\n'
}

# fb_summary TITLE STEPS PASSED SECONDS FAILED NEXT...: how the script ended. FAILED is the
# names of failed steps joined with ", " (empty when all passed); each NEXT is "command::about".
fb_summary() {
  fb_title=$1 fb_steps=$2 fb_passed=$3 fb_secs=$4 fb_bad=$5
  shift 5
  fb_el=""
  [ -z "$fb_secs" ] || fb_el=$(fb_fmt "$fb_secs")
  fb_unit=steps
  [ "$fb_steps" -ne 1 ] || fb_unit=step
  if [ "$FB_PRETTY" != 1 ]; then
    if [ -z "$fb_bad" ]; then
      fb_bits=""
      [ "$fb_steps" -le 0 ] || fb_bits="$fb_passed of $fb_steps $fb_unit"
      [ -z "$fb_el" ] || fb_bits="${fb_bits}${fb_bits:+, }$fb_el"
      printf 'ok: %s%s\n' "$fb_title" "${fb_bits:+ ($fb_bits)}"
    else
      fb_count=$(printf '%s' "$fb_bad" | awk -F', ' '{print NF}')
      fb_bits="failed: $fb_bad"
      [ "$fb_steps" -le 0 ] || fb_bits="$fb_count of $fb_steps $fb_unit failed: $fb_bad"
      [ -z "$fb_el" ] || fb_bits="$fb_bits; $fb_el"
      printf 'failed: %s (%s)\n' "$fb_title" "$fb_bits"
    fi
    for fb_next in "$@"; do
      fb_cmd=${fb_next%%::*}
      fb_about=""
      case $fb_next in *::*) fb_about=${fb_next#*::} ;; esac
      printf '  next: %s%s\n' "$fb_cmd" "${fb_about:+  - $fb_about}"
    done
    return 0
  fi
  printf '\n  %s' "$FB_MUTED"
  fb_repeat "$FB_RULE" $((FB_WIDTH - 2))
  printf '%s\n' "$FB_RST"
  if [ -z "$fb_bad" ]; then
    fb_mark=$FB_TICK fb_style=$FB_OK
  else
    fb_mark=$FB_CROSS fb_style=$FB_BAD
  fi
  printf '  %s%s %s%s' "$fb_style" "$fb_mark" "$fb_title" "$FB_RST"
  fb_bits=""
  if [ "$fb_steps" -gt 0 ]; then
    if [ -z "$fb_bad" ]; then
      if [ "$fb_passed" = "$fb_steps" ]; then
        fb_bits="all $fb_steps $fb_unit"
      else
        fb_bits="$fb_passed of $fb_steps $fb_unit passed"
      fi
    else
      fb_count=$(printf '%s' "$fb_bad" | awk -F', ' '{print NF}')
      fb_bits="$fb_count of $fb_steps failed: $fb_bad"
    fi
  fi
  [ -z "$fb_el" ] || fb_bits="${fb_bits}${fb_bits:+  $FB_DOT  }$fb_el"
  [ -z "$fb_bits" ] || printf '   %s%s%s' "$FB_MUTED" "$fb_bits" "$FB_RST"
  printf '\n'
  fb_first=1
  for fb_next in "$@"; do
    [ "$fb_first" = 1 ] && printf '\n'
    fb_cmd=${fb_next%%::*}
    fb_about=""
    case $fb_next in *::*) fb_about=${fb_next#*::} ;; esac
    if [ "$fb_first" = 1 ]; then fb_lead="next"; else fb_lead="    "; fi
    printf '  %s%s%s  %s%s%s%s\n' "$FB_LABEL" "$fb_lead" "$FB_RST" "$FB_BRAND" "$fb_cmd" "$FB_RST" \
      "${fb_about:+  $FB_MUTED$fb_about$FB_RST}"
    fb_first=0
  done
  printf '\n'
}

# fb_ready TITLE ROW...: the panel a setup ends on; each ROW is "command::about".
fb_ready() {
  fb_title=$1
  shift
  if [ "$FB_PRETTY" != 1 ]; then
    printf 'ready: %s\n' "$fb_title"
    for fb_row_text in "$@"; do
      fb_cmd=${fb_row_text%%::*}
      fb_about=""
      case $fb_row_text in *::*) fb_about=${fb_row_text#*::} ;; esac
      printf '  %s%s\n' "$fb_cmd" "${fb_about:+  - $fb_about}"
    done
    return 0
  fi
  fb_inner=$((FB_WIDTH - 4)) # between the two border characters
  fb_cmdw=0
  for fb_row_text in "$@"; do
    fb_cmd=${fb_row_text%%::*}
    [ "${#fb_cmd}" -le "$fb_cmdw" ] || fb_cmdw=${#fb_cmd}
  done
  printf '  %s╭─ %s %s %s' "$FB_OK" "$FB_TICK" "$fb_title" "$FB_OK"
  fb_repeat '─' $((fb_inner - ${#fb_title} - 5))
  printf '╮%s\n' "$FB_RST"
  printf '  %s│%s' "$FB_OK" "$FB_RST"
  fb_repeat ' ' "$fb_inner"
  printf '%s│%s\n' "$FB_OK" "$FB_RST"
  for fb_row_text in "$@"; do
    fb_cmd=${fb_row_text%%::*}
    fb_about=""
    case $fb_row_text in *::*) fb_about=${fb_row_text#*::} ;; esac
    printf '  %s│%s   %s%s%s' "$FB_OK" "$FB_RST" "$FB_BRAND" "$fb_cmd" "$FB_RST"
    fb_repeat ' ' $((fb_cmdw - ${#fb_cmd} + 4))
    printf '%s%s%s' "$FB_MUTED" "$fb_about" "$FB_RST"
    fb_repeat ' ' $((fb_inner - 7 - fb_cmdw - ${#fb_about}))
    printf '%s│%s\n' "$FB_OK" "$FB_RST"
  done
  printf '  %s│%s' "$FB_OK" "$FB_RST"
  fb_repeat ' ' "$fb_inner"
  printf '%s│%s\n' "$FB_OK" "$FB_RST"
  printf '  %s╰' "$FB_OK"
  fb_repeat '─' "$fb_inner"
  printf '╯%s\n\n' "$FB_RST"
}
# <<< renderer

has() { command -v "$1" >/dev/null 2>&1; }

case "$(uname -s)" in
  Darwin) os="macOS" ;;
  Linux) os="Linux" ;;
  *) fail "unsupported OS $(uname -s); on Windows use install.ps1 (see the README)" ;;
esac
has curl || has wget || fail "need curl or wget"

# The newest release tag, or main when there is none (or no network).
latest_ref() {
  api="https://api.github.com/repos/$REPO/releases/latest"
  if has curl; then
    tag=$(curl -fsSL --max-time 10 "$api" 2>/dev/null | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1) || true
  else
    tag=$(wget -qO- -T 10 "$api" 2>/dev/null | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1) || true
  fi
  printf '%s\n' "${tag:-main}"
}

# What to install, and how many steps that makes.
ref="" spec=""
if [ -n "${ASKPHYSICS_SOURCE:-}" ]; then
  spec="$ASKPHYSICS_SOURCE"
  what="from $spec"
else
  ref="${ASKPHYSICS_REF:-}"
  [ -n "$ref" ] || ref=$(latest_ref)
  spec="askphysics @ git+https://github.com/$REPO@$ref"
  what="$ref"
fi
total=2
has uv || total=$((total + 1))
[ "${ASKPHYSICS_SKIP_MODELS:-}" = "1" ] || total=$((total + 1))
n=0
t0=$(date +%s)

fb_header "Install" "$os ($(uname -m)) · Ask Physics $what, in its own environment" "$total"

# step TITLE COMMAND...: one numbered step; the installer stops if it fails.
step() {
  n=$((n + 1))
  fb_title=$1
  shift
  fb_step "$fb_title" "$n" "$total" "" "$@" || exit 1
}

fetch() {
  if has curl; then curl -LsSf --max-time 600 "$1"; else wget -qO- "$1"; fi
}

install_uv() {
  fetch https://astral.sh/uv/install.sh | sh
}

# 1. uv
if ! has uv; then
  step "Install uv" install_uv
  # uv installs to ~/.local/bin by default; make it visible to the rest of this script.
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  export PATH
  has uv || fail "uv was installed but is not on PATH; open a new shell and rerun"
fi

# 3. Install
step "Install Ask Physics ${ref:-$spec}" uv tool install --force --python "$PYTHON_VERSION" "$spec"

# 4. PATH
bin_dir="$(uv tool dir --bin)"
newpath=0
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *)
    uv tool update-shell >/dev/null 2>&1 || true
    PATH="$bin_dir:$PATH"
    export PATH
    newpath=1
    ;;
esac

# The installed command starts, and says which version it is.
n=$((n + 1))
fb_title="Check that askphysics starts"
fb_version=$("$bin_dir/askphysics" version 2>&1) || {
  fb_result fail "$fb_title" "" "$n" "$total"
  fail "installed, but askphysics did not start"
}
fb_result ok "$fb_title" "$(printf '%s\n' "$fb_version" | head -n 1)" "$n" "$total"

# 5. Models (tellus and solem, checked against the manifest the package pins)
if [ "${ASKPHYSICS_SKIP_MODELS:-}" = "1" ]; then
  info "Skipping the Fermi models (ASKPHYSICS_SKIP_MODELS=1)"
else
  n=$((n + 1))
  fb_live "Download the Fermi models" "$n" "$total" "$bin_dir/askphysics" model pull --if-published \
    || info "Couldn't download the models just now. Run later: askphysics model pull"
fi

if [ "$newpath" = 1 ]; then
  info "$bin_dir is now on your PATH. Open a new terminal, or run: export PATH=\"$bin_dir:\$PATH\""
fi
fb_summary "Ask Physics is installed" "$total" "$n" "$(($(date +%s) - t0))" ""
fb_ready "Ask Physics is ready" \
  'askphysics ask "A 20 m drop: how fast does it land?"::ask a question' \
  "askphysics --help::see every command" \
  "uv tool uninstall askphysics::remove it again"
