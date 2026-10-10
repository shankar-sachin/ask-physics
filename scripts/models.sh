#!/bin/sh
# Back up, restore, and list installed Fermi models.
#
#   sh scripts/models.sh list
#   sh scripts/models.sh backup fermi-solem-1     copy it to ~/askphysics-backup/<model>-<time>
#   sh scripts/models.sh backup fermi-solem-1 NAME   ...or to ~/askphysics-backup/NAME
#   sh scripts/models.sh restore fermi-solem-1    put the newest backup back
#   sh scripts/models.sh restore fermi-solem-1 fermi-solem-1-20261007-0412
#
# Backups go to $ASKPHYSICS_BACKUP_DIR (default ~/askphysics-backup). A restore first
# backs up what it replaces, so nothing is ever lost. Each copy is shown file by file, with
# sizes, a bar with speed and ETA while it runs, and the sha256 of every file checked against
# its source.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help | "") usage 0 ;;
esac
action=$1
models=$(models_dir)
backups=$(backup_dir)

newest_backup() {
  # Timestamps sort as text, so the last match is the newest.
  find "$backups" -maxdepth 1 -type d -name "$1-*" 2>/dev/null | sort | tail -n 1
}

# copy_dir SOURCE DEST TITLE: a verified copy with the file-by-file view, or plain cp -R when
# the display module isn't available (or under DRY_RUN).
copy_dir() {
  if [ "${DRY_RUN:-}" != "1" ] && ui_has_python; then
    ui_call copy --title "$3" "$1" "$2"
  else
    info "$3"
    run cp -R "$1" "$2"
  fi
}

backup() {
  name=$1
  [ -d "$models/$name" ] || {
    warn "no $name installed in $models; nothing to back up"
    return 0
  }
  dest="$backups/${2:-$name-$(date +%Y%m%d-%H%M%S)}"
  [ ! -e "$dest" ] || fail "$dest already exists; not overwriting a backup"
  run mkdir -p "$backups"
  copy_dir "$models/$name" "$dest" "Backing up $name"
}

list_dir() {
  if [ "${DRY_RUN:-}" != "1" ] && ui_has_python; then
    ui_call list --title "$1" "$2"
  else
    info "$1 ($2)"
    ls -1 "$2" 2>/dev/null || say "  (none)"
  fi
}

case $action in
  list)
    list_dir "Installed models" "$models"
    list_dir "Backups" "$backups"
    ;;
  backup)
    [ $# -ge 2 ] || usage 1
    backup "$2" "${3:-}"
    ;;
  restore)
    [ $# -ge 2 ] || usage 1
    name=$2
    if [ $# -ge 3 ]; then
      source_dir="$backups/$3"
    else
      source_dir=$(newest_backup "$name")
    fi
    if [ -z "$source_dir" ] || [ ! -d "$source_dir" ]; then
      fail "no backup of $name in $backups"
    fi
    backup "$name"
    run rm -rf "${models:?}/$name"
    run mkdir -p "$models"
    copy_dir "$source_dir" "$models/$name" "Restoring $name from $(basename "$source_dir")"
    ;;
  *) usage 1 ;;
esac
