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
# backs up what it replaces, so nothing is ever lost.
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help | "") usage ;;
esac
action=$1
models=$(models_dir)
backups=$(backup_dir)

newest_backup() {
  # Timestamps sort as text, so the last match is the newest.
  find "$backups" -maxdepth 1 -type d -name "$1-*" 2>/dev/null | sort | tail -n 1
}

backup() {
  name=$1
  [ -d "$models/$name" ] || {
    warn "no $name installed in $models; nothing to back up"
    return 0
  }
  dest="$backups/${2:-$name-$(date +%Y%m%d-%H%M%S)}"
  [ ! -e "$dest" ] || fail "$dest already exists; not overwriting a backup"
  info "Backing up $name to $dest"
  run mkdir -p "$backups"
  run cp -R "$models/$name" "$dest"
}

case $action in
  list)
    info "Installed in $models"
    ls -1 "$models" 2>/dev/null || say "  (none)"
    info "Backups in $backups"
    ls -1 "$backups" 2>/dev/null || say "  (none)"
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
    [ -n "$source_dir" ] && [ -d "$source_dir" ] || fail "no backup of $name in $backups"
    backup "$name"
    info "Restoring $name from $source_dir"
    run rm -rf "${models:?}/$name"
    run mkdir -p "$models"
    run cp -R "$source_dir" "$models/$name"
    ;;
  *) usage 1 ;;
esac
