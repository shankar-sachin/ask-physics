# Releasing

Releases are tagged only with the maintainer's approval (`CLAUDE.md`).

1. **Prepare.** On a branch, bump `version` in `pyproject.toml` and
   `__version__` in `src/askphysics/__init__.py`, move `CHANGELOG.md`'s
   `Unreleased` entries under a `## [X.Y.Z] - DATE` heading, and update the
   version badge in `README.md`. Merge once CI is green. The release notes
   are that CHANGELOG section, so the heading must exist before you tag.
2. **Tag.** Tag the merge commit on `main` and push the tag:

   ```bash
   git tag vX.Y.Z && git push origin vX.Y.Z
   ```

   That is all: pushing the tag runs the Release workflow
   (`.github/workflows/release.yml`), which creates the GitHub Release. It

   - checks the tag against `pyproject.toml` and `__init__.py` (a mismatch
     fails the run before anything is built),
   - builds `AskPhysicsSetup-X.Y.Z.exe` with Inno Setup on Windows and
     smoke-tests it silently (install from the checkout, `askphysics version`,
     `askphysics ask`, silent uninstall, and a deliberately failing install
     that must make Setup exit non-zero),
   - creates the Release `vX.Y.Z` with the CHANGELOG section as its notes and
     attaches the installer, `AskPhysicsSetup-X.Y.Z.exe`, its `.sha256`, an
     identical unversioned copy `AskPhysicsSetup.exe` (with its own `.sha256`),
     and `shankars.askphysics-X.Y.Z-winget-manifests.zip` (the rendered WinGet
     manifests, step 4). The unversioned copy gives the website a stable link,
     `https://github.com/shankar-sachin/ask-physics/releases/latest/download/AskPhysicsSetup.exe`,
     that always serves the newest release. The WinGet manifests point at the
     versioned file, whose URL never changes.

   Do not draft the Release by hand. If a step fails, fix the cause and re-run
   the workflow from the Actions tab (the release step updates an existing
   Release instead of failing), or delete the tag and push it again. Do not
   re-run it after the WinGet pull request is open: rebuilding the installer
   changes its SHA-256. To rehearse without releasing, run the Release
   workflow by hand (`workflow_dispatch`) with a `ref`; it builds and
   smoke-tests the installer and stops there. Pull requests that touch
   `packaging/`, `install.ps1` or the workflow do the same.
3. **Bump the Homebrew tap.** In `shankar-sachin/homebrew-tap`, update
   `Formula/askphysics.rb`:

   ```bash
   curl -fsSL https://github.com/shankar-sachin/ask-physics/archive/refs/tags/vX.Y.Z.tar.gz \
     | shasum -a 256
   ```

   Set `url` to that tarball and `sha256` to the hash, then push. The tap's CI
   installs and tests the formula on macOS.
4. **Submit to WinGet** (ADR-023). The package is `shankars.askphysics`; the
   Release carries its manifests, already filled in with the version, the
   installer URL and the installer's SHA-256. Nothing is submitted
   automatically. In a clone of the fork `shankar-sachin/winget-pkgs`, with the
   fork synced to upstream:

   ```bash
   git switch -c shankars.askphysics-X.Y.Z upstream/master
   gh release download vX.Y.Z -R shankar-sachin/ask-physics \
     -p 'shankars.askphysics-*-winget-manifests.zip' -D /tmp
   unzip /tmp/shankars.askphysics-X.Y.Z-winget-manifests.zip   # at the clone's root
   ```

   That creates `manifests/s/shankars/askphysics/X.Y.Z/` with three files. The
   publisher folder is lowercase `shankars`, the same as the maintainer's other
   package, so folders never collide on case-insensitive file systems. To
   render them by hand instead, run
   `sh scripts/winget.sh X.Y.Z <installer sha256> <output dir>`. On Windows,
   check them before committing:

   ```powershell
   winget validate --manifest manifests\s\shankars\askphysics\X.Y.Z
   ```

   Commit, push, and open a pull request against `microsoft/winget-pkgs`
   titled **`New package: shankars.askphysics version X.Y.Z`** for the first
   version and **`New version: shankars.askphysics version X.Y.Z`** for every
   later one. Microsoft's pipeline then validates the manifests, installs the
   package in a clean Windows machine, and a moderator reviews it. Expect hours
   to a few days for the first version, longer if they ask questions. The
   install runs `install.ps1`, which downloads Python, torch and the models, so
   it takes several minutes; if their pipeline times out, or a reviewer objects
   to an installer that downloads at install time, that is the likely place. If
   a reviewer asks for changes to the manifests, make them in
   `packaging/winget/*.yaml.in` as well, or the next version loses them.

   Once the pull request is merged and the Windows Package Manager source has
   refreshed, verify it on a clean Windows machine:

   ```powershell
   winget install shankars.askphysics     # ids are matched case-insensitively,
                                          # so ShankarS.AskPhysics works too
   askphysics version                     # in a new terminal
   askphysics ask "A ball is dropped from 20 m. How fast does it land?"
   winget list shankars.askphysics
   winget uninstall shankars.askphysics   # askphysics is gone again
   ```

   For a later version, `winget upgrade shankars.askphysics` runs the new
   installer over the old one (same Inno `AppId`, `UpgradeBehavior: install`).
5. **Check the installers.** `install.sh` and `install.ps1` pick up the latest
   release automatically (a Release created by the workflow becomes "latest").
   Run the Installers workflow (`workflow_dispatch`) on `main` to confirm they
   still work everywhere.

### How the Windows installer works

`AskPhysicsSetup-X.Y.Z.exe` (`packaging/windows/askphysics.iss`) is a
per-user installer: it never asks for administrator rights, and it supports
`/VERYSILENT /SUPPRESSMSGBOXES /NORESTART`. It does not contain the program.
Before it writes anything, it runs `install.ps1` with
`ASKPHYSICS_REF=vX.Y.Z` (from Setup's temporary folder, through a small runner,
`askphysics-setup.ps1`), so it installs exactly that release with uv and then
downloads the models. Its output goes to `%TEMP%\AskPhysics-setup.log`; if the
script fails, Setup stops with the end of that log and exit code 7, which
WinGet relies on, and leaves nothing installed. Otherwise it copies the two
scripts into `%LOCALAPPDATA%\Programs\Ask Physics` for the uninstaller. The
uninstaller runs `uv tool uninstall askphysics`. It does not remove uv, the
uv-managed Python, or the downloaded models in
`%USERPROFILE%\.cache\askphysics\models`: the models are the user's data, a
reinstall reuses them, and a pip or brew install of Ask Physics shares the
folder. The `AppId` in the `.iss` must never change; the WinGet `ProductCode`
is derived from it (`tests/test_scripts_winget.py` checks they agree). The
installer is not code-signed. The WinGet manifest lists `astral-sh.uv` as a
dependency, but `install.ps1` installs uv itself when it is not on `PATH`, so
setup works either way.

### PyPI

The same tag publishes `askphysics` to PyPI (`pip install askphysics`), in the
release workflow's `pypi` job. It uses trusted publishing: PyPI trusts this
repository's `release.yml` in the `pypi` environment, so no token is stored. The
job checks that the tag matches `pyproject.toml`, builds the sdist and wheel,
runs `twine check --strict`, and uploads both. It runs on its own, so a PyPI
problem never holds up the GitHub Release or the installer; if it fails, fix
the cause and re-run just that job. A version can be uploaded to PyPI only once,
so a broken upload needs a new version, not a re-tag.

The publisher was registered as a pending publisher for the project name
`askphysics` (owner `shankar-sachin`, repository `ask-physics`, workflow
`release.yml`, environment `pypi`); the first successful upload creates the
project.

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
   and publish. The URLs in the manifest point there. (This stays manual: the
   Release workflow only reacts to `vX.Y.Z` tags.)
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
