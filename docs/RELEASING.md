# Releasing

Releases are tagged only with the maintainer's approval (`CLAUDE.md`).

1. **Prepare.** On a branch, bump `version` in `pyproject.toml` and
   `__version__` in `src/askphysics/__init__.py`, move `CHANGELOG.md`'s
   `Unreleased` entries under the new version with today's date, and update
   the version badge in `README.md`. Merge once CI is green.
2. **Tag.** On GitHub, open Releases, then "Draft a new release", create the
   tag `vX.Y.Z` on `main`, paste the CHANGELOG section as the notes, and
   publish.
3. **Bump the Homebrew tap.** In `shankar-sachin/homebrew-tap`, update
   `Formula/askphysics.rb`:

   ```bash
   curl -fsSL https://github.com/shankar-sachin/ask-physics/archive/refs/tags/vX.Y.Z.tar.gz \
     | shasum -a 256
   ```

   Set `url` to that tarball and `sha256` to the hash, then push. The tap's CI
   installs and tests the formula on macOS.
4. **Check the installers.** `install.sh` and `install.ps1` pick up the latest
   release automatically. Run the Installers workflow (`workflow_dispatch`) on
   `main` to confirm they still work everywhere.
