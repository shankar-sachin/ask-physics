#!/bin/sh
# Ask Physics installer for macOS and Linux.
#
#   curl -LsSf https://raw.githubusercontent.com/shankar-sachin/ask-physics/main/install.sh | sh
#
# Installs the `askphysics` command in its own isolated environment with uv
# (https://docs.astral.sh/uv/), installing uv first if it's missing. Nothing
# touches your system Python.
#
# Environment variables:
#   ASKPHYSICS_REF          git tag, branch, or commit to install (default: latest release)
#   ASKPHYSICS_SOURCE       install from this local path or URL instead of GitHub
#
# On Linux, torch comes from PyTorch's CPU-only index (pinned in pyproject.toml):
# the default PyPI wheel bundles about 2 GB of CUDA libraries the CLI doesn't need.
#
# Uninstall:  uv tool uninstall askphysics
set -eu

REPO="shankar-sachin/ask-physics"
PYTHON_VERSION="3.12"

say() { printf '%s\n' "$*"; }
info() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
has() { command -v "$1" >/dev/null 2>&1; }

say ""
say "  ◉ Ask Physics installer"
say "    grounded answers, real math"
say ""

case "$(uname -s)" in
  Darwin) os="macos" ;;
  Linux) os="linux" ;;
  *) fail "unsupported OS $(uname -s); on Windows use install.ps1 (see the README)" ;;
esac
info "Detected $os ($(uname -m))"

has curl || has wget || fail "need curl or wget"

# 1. uv
if ! has uv; then
  info "Installing uv (Python package manager)"
  if has curl; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  else
    wget -qO- https://astral.sh/uv/install.sh | sh
  fi
  # uv installs to ~/.local/bin by default; make it visible to the rest of this script.
  PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  export PATH
  has uv || fail "uv was installed but is not on PATH; open a new shell and rerun"
fi
info "Using $(uv --version)"

# 2. What to install
if [ -n "${ASKPHYSICS_SOURCE:-}" ]; then
  spec="$ASKPHYSICS_SOURCE"
  info "Installing from $spec"
else
  ref="${ASKPHYSICS_REF:-}"
  if [ -z "$ref" ]; then
    api="https://api.github.com/repos/$REPO/releases/latest"
    if has curl; then
      ref=$(curl -fsSL "$api" 2>/dev/null | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1) || true
    else
      ref=$(wget -qO- "$api" 2>/dev/null | sed -n 's/.*"tag_name": *"\([^"]*\)".*/\1/p' | head -n 1) || true
    fi
    ref="${ref:-main}"
  fi
  spec="askphysics @ git+https://github.com/$REPO@$ref"
  info "Installing Ask Physics $ref"
fi

# 3. Install
uv tool install --force --python "$PYTHON_VERSION" "$spec"

# 4. PATH
bin_dir="$(uv tool dir --bin)"
case ":$PATH:" in
  *":$bin_dir:"*) ;;
  *)
    info "Adding $bin_dir to your PATH (uv tool update-shell)"
    uv tool update-shell >/dev/null 2>&1 || true
    PATH="$bin_dir:$PATH"
    export PATH
    say "    Open a new terminal, or run: export PATH=\"$bin_dir:\$PATH\""
    ;;
esac

say ""
"$bin_dir/askphysics" version || fail "installed, but askphysics did not start"
say ""
say "  Try it:"
say "    askphysics ask \"How fast does a ball dropped from 20 m hit the ground?\""
say ""
