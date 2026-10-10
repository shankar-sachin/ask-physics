## Install

**macOS and Linux**

```bash
curl -LsSf https://askphysics.vercel.app/installers/install.sh | sh
```

**Windows** (PowerShell)

```powershell
irm https://askphysics.vercel.app/installers/install.ps1 | iex
```

**Homebrew** (macOS and Linux)

```bash
brew install shankar-sachin/tap/askphysics
```

The installers put `askphysics` in its own environment with [uv](https://docs.astral.sh/uv/)
and never touch your system Python. Uninstall with `uv tool uninstall askphysics` (or
`brew uninstall askphysics`).

Or skip installing and use the website, [askphysics.vercel.app](https://askphysics.vercel.app),
which runs the same package inside your browser (with the fake model for now; see the
[FAQ](FAQ#does-the-website-use-the-fermi-models)).

## Ask a question

```bash
askphysics ask "How fast does a falling object hit the ground if it is dropped from 20 m?"
```

Quotes are optional. Useful options:

| Option | What it does |
|---|---|
| `--json` | Print the full answer as JSON instead of a card. |
| `--model fermi-tellus-1` | Use one Fermi model for every stage. |
| `--llm fake` | Use the built-in fake model, which only knows "dropped from a height" questions. |

## Get the models

You don't have to do anything. The installers and Homebrew download tellus and solem when they
install (about 66 MB, once), and any install, pip included, downloads whichever of them is
missing on your first `askphysics ask`, with a progress display. They are kept in
`~/.cache/askphysics/models/` (or the folder `ASKPHYSICS_MODEL_DIR` names; Homebrew keeps them
in its own `var` folder), so it happens once.

Every file is checked against the sizes and hashes this version of Ask Physics pins, and
anything that doesn't match is refused. Set `ASKPHYSICS_AUTO_PULL=0` to stop the downloads
that happen while you ask. If a download fails, `ask` still answers (with a stand-in) and says why, and the next `ask`
tries again. Without models, `ask` uses a stand-in that only knows "dropped from a height"
questions, and says so.

## celeste asks first

tellus and solem are small and every question needs them. celeste (about 240 MB) is the last
resort when solem can't make a plan work, and most questions never get that far, so it never
downloads by itself. When a question does reach it, `askphysics ask` on a terminal asks:

```
? This question needs celeste-1 (about 240.1 MB, a one-time download) to try harder.
  [y] yes   [N] no   [a] always download   [never] don't ask again
  Download it?
```

`y` downloads it now (with a progress display) and carries on with it; `n` or Enter skips
celeste for this question and doesn't ask again during it. `a` and `never` are remembered in a
small file, `preferences.json`, in the models folder (`~/.cache/askphysics/models/`, or
`ASKPHYSICS_MODEL_DIR`); to reset the choice, delete that file. You can also set
`ASKPHYSICS_CELESTE_DOWNLOAD=ask`, `always`, or `never`, which wins over the file.

With `--json`, in a script (no terminal), and on the website, nothing is ever asked: celeste is
skipped unless it is already installed or your saved choice is `always`.

## Other commands

| Command | What it does |
|---|---|
| `askphysics version` | The installed version. |
| `askphysics --help` | Every command and option. |

Next: [Reading an answer](Reading-an-Answer).
