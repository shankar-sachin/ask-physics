# Shared helpers for the workflow scripts. Sourced, never run:  . "$(dirname "$0")/lib.sh"
#
# What a script uses, and what it looks like (docs/SCRIPTS.md):
#   ui_begin TITLE ABOUT [STEPS [PAD]]   a header: what the script does and how many steps
#   phase TITLE COMMAND...               a step: a spinner and the time, then a tick; long
#                                        output folds into a few dimmed lines (full log kept)
#   phase_live TITLE COMMAND...          the same for a command that draws its own display
#   gate TITLE PARSE COMMAND...          a step that, on failure, lets the script go on
#   ui_result ok|fail|skip TITLE [DETAIL]   a finished line with no command behind it
#   ui_ready TITLE "command::about"...   a panel of commands to try
#   ui_end OK-TITLE [FAIL-TITLE] ["command::about"...]   a summary; exits 1 after a failed gate
#   info, warn, fail, say                one line each; fail exits 1
#
# DRY_RUN=1        print each command instead of running it (the tests use this)
# NO_CAFFEINATE=1  don't wrap long steps in macOS caffeinate
# NO_COLOR=1       plain lines, no colour or animation (also the default off a terminal)
# ASKPHYSICS_UI=sh use the pure-sh renderer even when Python is available
# ASKPHYSICS_LOG_DIR  where full step logs go (default $TMPDIR/askphysics-logs)
#
# The look comes from "python -m askphysics.shell_ui" (Rich) when askphysics is importable, and
# from the pure-sh renderer below when it isn't (before setup, on a machine with no Python).
# shellcheck shell=sh

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

# Which renderer draws: the Rich one when askphysics's display module can be imported (checked
# once), else the pure-sh one above. DRY_RUN always prints plain lines.
UI_T0="" UI_TOTAL=0 UI_INDEX=0 UI_PAD=0 UI_PASSED=0 UI_FAILED="" UI_LOGDIR=""
UI_SRC="" UI_OK=""
if [ "${DRY_RUN:-}" = "1" ]; then FB_PRETTY=0; fi

# ui_python [PYTHON [SRC]]: choose the interpreter that draws the display. With no argument,
# the first of python and python3 that can import askphysics. setup.sh passes the new
# virtualenv's python and the checkout's src/, because the package isn't installed yet.
ui_python() {
  UI_PYTHON=""
  UI_SRC=${2:-}
  if [ "${DRY_RUN:-}" = "1" ] || [ "${ASKPHYSICS_UI:-}" = "sh" ]; then
    return 0
  fi
  if [ -n "${1:-}" ]; then
    set -- "$1"
  else
    set -- python python3
  fi
  for candidate in "$@"; do
    if command -v "$candidate" >/dev/null 2>&1 \
      && PYTHONPATH="${UI_SRC:+$UI_SRC:}${PYTHONPATH:-}" \
        "$candidate" -c 'import askphysics.shell_ui' >/dev/null 2>&1; then
      UI_PYTHON=$candidate
      return 0
    fi
  done
}
ui_call() {
  PYTHONPATH="${UI_SRC:+$UI_SRC:}${PYTHONPATH:-}" "$UI_PYTHON" -m askphysics.shell_ui "$@"
}
ui_has_python() {
  if [ -z "${UI_PYTHON+x}" ]; then ui_python; fi
  [ -n "$UI_PYTHON" ]
}

ui_now() { date +%s; }

# ui_begin TITLE ABOUT [STEPS [PAD]]: the header, and the clock for the summary. STEPS is
# how many steps follow (shown as 3/6 beside each); PAD is the width of the title column.
ui_begin() {
  UI_T0=$(ui_now) UI_TOTAL=${3:-0} UI_PAD=${4:-0} UI_INDEX=0 UI_PASSED=0 UI_FAILED=""
  FB_PAD=$UI_PAD
  if ui_has_python; then
    ui_call header --title "$1" --about "${2:-}" --steps "$UI_TOTAL"
  else
    fb_header "$1" "${2:-}" "$UI_TOTAL"
  fi
}

# Where a step's full output goes: one directory per run, one file per step.
ui_log() {
  if [ -z "$UI_LOGDIR" ]; then
    UI_LOGDIR="${ASKPHYSICS_LOG_DIR:-${TMPDIR:-/tmp}/askphysics-logs}/$(basename "$0" .sh)-$(date +%Y%m%d-%H%M%S)"
  fi
  slug=$(printf '%s' "$1" | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-40)
  printf '%s/%02d-%s.log' "$UI_LOGDIR" "$UI_INDEX" "$slug"
}

# ui_ok TEXT: what the next step says beside its tick when it succeeds with no output.
ui_ok() { UI_OK=$1; }

# Run a long step as a named phase: a spinner with the time so far while it runs, then a tick
# and how long it took. Plain lines when output is not a terminal.
# phase_live is for a command that draws its own live display, such as training.
#   phase "Building data" askphysics model build-data ...
phase() {
  _phase "" "" "$@"
}
phase_live() {
  _phase --inherit "" "$@"
}
# gate TITLE PARSE COMMAND...: like phase, but a failure is recorded and the script goes on
# (ui_end then reports it). PARSE is pytest (counts), corpus (books and words), last or first (the closing or
# opening line), or - for none.
gate() {
  _gate_title=$1 _gate_parse=$2
  shift 2
  _phase "" "$_gate_parse" "$_gate_title" "$@" || {
    UI_FAILED="$UI_FAILED${UI_FAILED:+
}$_gate_title"
    return 0
  }
}
_phase() {
  _mode=$1 _parse=$2 _title=$3
  shift 3
  UI_INDEX=$((UI_INDEX + 1))
  if [ "${DRY_RUN:-}" = "1" ]; then
    say "-- $_title"
    awake "$@"
    UI_PASSED=$((UI_PASSED + 1))
    return 0
  fi
  if [ "${NO_CAFFEINATE:-}" != "1" ] && command -v caffeinate >/dev/null 2>&1; then
    set -- caffeinate -dims "$@"
  fi
  _rc=0
  _total=""
  [ "$UI_TOTAL" -le 0 ] || _total=$UI_TOTAL
  _ok=$UI_OK
  UI_OK=""
  if ui_has_python; then
    set -- -- "$@"
    [ -z "$_ok" ] || set -- --ok "$_ok" "$@"
    [ -z "$_mode" ] || set -- "$_mode" "$@"
    case $_parse in pytest | last | first | corpus) set -- --parse "$_parse" "$@" ;; esac
    if [ -n "$_total" ]; then set -- --index "$UI_INDEX" --total "$_total" "$@"; fi
    ui_call step --title "$_title" --pad "$UI_PAD" --log "$(ui_log "$_title")" "$@" || _rc=$?
  elif [ "$_mode" = "--inherit" ]; then
    fb_live "$_title" "${_total:+$UI_INDEX}" "$_total" "$@" || _rc=$?
  else
    fb_step "$_title" "${_total:+$UI_INDEX}" "$_total" "$(ui_log "$_title")" "$@" || _rc=$?
  fi
  if [ "$_rc" = 0 ]; then
    UI_PASSED=$((UI_PASSED + 1))
  fi
  return "$_rc"
}

# ui_result ok|fail|skip TITLE [DETAIL]: a finished line (a tick, a cross, or a dash).
ui_result() {
  UI_INDEX=$((UI_INDEX + 1))
  case $1 in
    ok) UI_PASSED=$((UI_PASSED + 1)) ;;
    fail) UI_FAILED="$UI_FAILED${UI_FAILED:+
}$2" ;;
  esac
  _total=""
  [ "$UI_TOTAL" -le 0 ] || _total=$UI_TOTAL
  if ui_has_python; then
    set -- "$1" --title "$2" --detail "${3:-}" --pad "$UI_PAD"
    if [ -n "$_total" ]; then set -- "$@" --index "$UI_INDEX" --total "$_total"; fi
    ui_call result "$@"
  else
    fb_result "$1" "$2" "${3:-}" "${_total:+$UI_INDEX}" "$_total"
  fi
}

# ui_ready TITLE "command::about"...: a panel of commands to try.
ui_ready() {
  _title=$1
  shift
  if ui_has_python; then
    _n=$#
    while [ "$_n" -gt 0 ]; do
      set -- "$@" --row "$1"
      shift
      _n=$((_n - 1))
    done
    ui_call ready --title "$_title" "$@"
  else
    fb_ready "$_title" "$@"
  fi
}

# ui_end OK-TITLE [FAIL-TITLE] ["command::about"...]: the summary. After a failed gate it
# names what failed and exits 1. Each "command::about" is a next step to offer.
ui_end() {
  _ok_title=$1
  _bad_title=Failed
  shift
  if [ $# -gt 0 ]; then
    case $1 in
      *::*) ;;
      *)
        _bad_title=$1
        shift
        ;;
    esac
  fi
  _secs=""
  [ -z "$UI_T0" ] || _secs=$(($(ui_now) - UI_T0))
  _names=""
  _old_ifs=$IFS
  IFS='
'
  for _f in $UI_FAILED; do _names="$_names${_names:+, }$_f"; done
  IFS=$_old_ifs
  if [ -n "$UI_FAILED" ]; then _title=$_bad_title; else _title=$_ok_title; fi
  if ui_has_python; then
    _n=$#
    while [ "$_n" -gt 0 ]; do
      set -- "$@" --next "$1"
      shift
      _n=$((_n - 1))
    done
    set -- --title "$_title" --steps "$UI_TOTAL" --passed "$UI_PASSED" "$@"
    [ -z "$_secs" ] || set -- --since "$UI_T0" "$@"
    _old_ifs=$IFS
    IFS='
'
    for _f in $UI_FAILED; do set -- --failed "$_f" "$@"; done
    IFS=$_old_ifs
    ui_call summary "$@"
  else
    fb_summary "$_title" "$UI_TOTAL" "$UI_PASSED" "$_secs" "$_names" "$@"
  fi
  [ -z "$UI_FAILED" ] || exit 1
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
