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

## Publishing model weights (ADR-012)

Weights are published separately from code, under their own release tag
(for example `models-v0.4.0`), so a code release can reuse them.

1. **Evaluate.** Run `askphysics model eval --model fermi-solem-1 --examples 3000`
   so the card carries the results (they are read from the model's `eval.json`).
2. **Package.** For each model:

   ```bash
   askphysics model package --model fermi-solem-1 --release models-v0.4.0 \
     --attribution build/corpus/ATTRIBUTION.md
   ```

   Pass `--attribution` for every model trained on the prose corpus or the
   OpenStax *Physics* text: CC BY needs the credit to travel with the weights.
   This writes `build/release/fermi-solem-1/` (the packaged model, bf16),
   `build/release/assets/` (the files to upload), and pins them in
   `src/askphysics/lm/weights.json`.
3. **Check the packaged copy.** `askphysics model eval --model fermi-solem-1
   --directory build/release/fermi-solem-1 --examples 3000` scores exactly what
   ships. bf16 should match the original within noise.
4. **Release the assets.** On GitHub, draft a release with the tag
   `models-v0.4.0` on `main`, attach every file in `build/release/assets/`,
   and publish. The URLs in the manifest point there.
5. **Commit the manifest.** Open a PR with `src/askphysics/lm/weights.json`
   and the model cards. Once it merges, `askphysics model pull` works from
   `main`, and from the next code release for everyone else.

### Homebrew

The formula can't write to the user's home directory, where models live, so
it asks the user to pull them. Add to `Formula/askphysics.rb` once weights
are published:

```ruby
def caveats
  <<~EOS
    Download the Fermi models (about 66 MB) before your first question:
      askphysics model pull
  EOS
end
```
