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

1. **Evaluate.** Run `askphysics-dev model eval --model fermi-solem-1 --examples 3000`
   so the card carries the results (they are read from the model's `eval.json`).
2. **Package.** For each model:

   ```bash
   askphysics-dev model package --model fermi-solem-1 --release models-v0.4.0 \
     --attribution build/corpus/ATTRIBUTION.md
   ```

   Pass `--attribution` for every model trained on the prose corpus or the
   OpenStax *Physics* text: CC BY needs the credit to travel with the weights.
   This writes `build/release/fermi-solem-1/` (the packaged model, bf16),
   `build/release/assets/` (the files to upload), and pins them in
   `src/askphysics/lm/weights.json`.
3. **Check the packaged copy.** `askphysics-dev model eval --model fermi-solem-1
   --directory build/release/fermi-solem-1 --examples 3000` scores exactly what
   ships. bf16 should match the original within noise.
4. **Release the assets.** On GitHub, draft a release with the tag
   `models-v0.4.0` on `main`, attach every file in `build/release/assets/`,
   and publish. The URLs in the manifest point there.
5. **Commit the manifest.** Open a PR with `src/askphysics/lm/weights.json`
   and the model cards. Once it merges, `askphysics-dev model pull` works from
   `main`; the installers, Homebrew, and the first question download the models
   from the next code release on (and from `main` for an install of `main`).

### Homebrew

A formula can't write to the user's home directory, where `askphysics` keeps
models by default, but it can write to `#{var}`. So the formula (ADR-022):

1. wraps `askphysics` in a small script that points `ASKPHYSICS_MODEL_DIR` at
   `#{var}/askphysics/models` (a value the user sets themselves still wins);
2. runs the hidden `askphysics install-models` in `post_install`, which
   downloads tellus and solem (about 66 MB, checked against the manifest the
   release pins) into that directory; and
3. falls back to the first question if that download fails: `ask` downloads
   into the same directory, so a failed `post_install` loses nothing.

Put this in `Formula/askphysics.rb` (merge it with the existing `install`; the
rest of the formula, its `url`, `sha256`, resources, and `depends_on`, stays as
it is):

```ruby
  def install
    virtualenv_install_with_resources # whatever builds libexec/bin/askphysics today

    # Replace the symlink with a wrapper: the models live in var, which brew can write.
    # ${...:-} keeps a caller's own ASKPHYSICS_MODEL_DIR.
    rm_f bin/"askphysics"
    (bin/"askphysics").write <<~SH
      #!/bin/bash
      export ASKPHYSICS_MODEL_DIR="${ASKPHYSICS_MODEL_DIR:-#{var}/askphysics/models}"
      exec "#{libexec}/bin/askphysics" "$@"
    SH
    chmod 0555, bin/"askphysics"

    # askphysics-dev only runs from a source checkout (ADR-022); keep it off users' PATH.
    rm_f bin/"askphysics-dev"
  end

  def post_install
    (var/"askphysics/models").mkpath
    system bin/"askphysics", "install-models"
  rescue BuildError
    opoo "Couldn't download the Fermi models now. They will download when you ask your first question."
  end

  def caveats
    <<~EOS
      The Fermi models are kept in #{var}/askphysics/models, which
      `brew uninstall askphysics` leaves alone. To remove them too:
        rm -rf #{var}/askphysics
    EOS
  end

  test do
    # --llm fake: the test must not download anything.
    assert_match "19.8", shell_output("#{bin}/askphysics ask --llm fake 'A ball is dropped from 20 m. How fast does it land?'")
    assert_match version.to_s, shell_output("#{bin}/askphysics version")
  end
```

**This snippet has not been run against Homebrew.** Before the first release
that uses it, run `brew install --build-from-source ./Formula/askphysics.rb`
(or `brew reinstall`) on a Mac and on Linux, and check that:

- `post_install` writes into `$(brew --prefix)/var/askphysics/models` (its
  sandbox must allow `var`, and the network);
- `$(brew --prefix)/bin/askphysics ask "..."` finds the models there with
  `HOME` unset or pointing somewhere empty;
- a failed download (try it offline) prints the `opoo` line and leaves the
  install in place;
- the `rescue BuildError` actually catches a non-zero exit from `system` in a
  `post_install` on the Homebrew version in use.

`brew upgrade` runs `post_install` again. It only fetches models that are
missing, so the models in `var` carry over from one version to the next.

**Why not a search path** (the user directory first, then a read-only system
directory)? The directory `post_install` fills must be the one the program
reads, and, when `post_install` fails, the one the first question fills. A
search path would add a second setting and change every place that joins a
models directory and a model name. `ASKPHYSICS_MODEL_DIR` exists, both
`ask` and `install-models` honour it, and a test pins that.
