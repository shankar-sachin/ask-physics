#!/bin/sh
# Render the WinGet manifests for one release.
#
#   sh scripts/winget.sh VERSION INSTALLER_SHA256 [OUTDIR]
#
# Fills packaging/winget/*.yaml.in for VERSION (for example 0.4.0) and the SHA-256 of
# AskPhysicsSetup-VERSION.exe, and writes the three manifests to
#   OUTDIR/manifests/s/shankars/askphysics/VERSION/
# (OUTDIR defaults to build/winget), the layout of microsoft/winget-pkgs, so the folder can
# be copied straight into a branch of a winget-pkgs fork. The release date is today in UTC;
# set WINGET_DATE=YYYY-MM-DD to override it.
#
# Check the result on Windows with:  winget validate --manifest <that folder>
set -eu
. "$(dirname "$0")/lib.sh"

case ${1:-} in
  -h | --help) usage 0 ;;
esac
if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  usage 1
fi

version=$1
sha=$2
outdir=${3:-build/winget}
root=$(cd "$(dirname "$0")/.." && pwd)
templates=$root/packaging/winget

if ! printf '%s\n' "$version" | grep -Eq '^[0-9]+\.[0-9]+\.[0-9]+$'; then
  fail "version must look like 0.4.0, not '$version'"
fi
if ! printf '%s\n' "$sha" | grep -Eq '^[0-9A-Fa-f]{64}$'; then
  fail "the installer SHA-256 must be 64 hex digits, not '$sha'"
fi
sha=$(printf '%s\n' "$sha" | tr 'a-f' 'A-F')
date=${WINGET_DATE:-$(date -u +%Y-%m-%d)}
if ! printf '%s\n' "$date" | grep -Eq '^[0-9]{4}-[0-9]{2}-[0-9]{2}$'; then
  fail "WINGET_DATE must look like 2026-10-10, not '$date'"
fi

target=$outdir/manifests/s/shankars/askphysics/$version
mkdir -p "$target"
for template in "$templates"/shankars.askphysics*.yaml.in; do
  name=$(basename "$template" .in)
  sed -e "s/@VERSION@/$version/g" -e "s/@DATE@/$date/g" -e "s/@SHA_X64@/$sha/g" \
    "$template" >"$target/$name"
  if grep -q '@[A-Z_0-9]*@' "$target/$name"; then
    fail "$name still has an unfilled placeholder"
  fi
  info "wrote $target/$name"
done
