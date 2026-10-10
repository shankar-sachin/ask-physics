## Install

**macOS and Linux**

```bash
curl -fsSL https://askphysics.vercel.app/install.sh | bash
```

**Windows** (PowerShell)

```powershell
irm https://askphysics.vercel.app/install.ps1 | iex
```

**Homebrew** (macOS and Linux)

```bash
brew install shankar-sachin/tap/askphysics
```

The installers put `askphysics` in its own environment with [uv](https://docs.astral.sh/uv/)
and never touch your system Python. Uninstall with `uv tool uninstall askphysics` (or
`brew uninstall askphysics`). Every way to install is on [askphysics.vercel.app/installers](https://askphysics.vercel.app/installers).

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

You don't have to do anything: the first `askphysics ask` downloads tellus and solem if they
aren't installed yet (about 66 MB, once, with a progress display), and the installers do it
up front. To get them yourself, or after installing another way:

```bash
askphysics model pull          # tellus and solem, about 66 MB
askphysics model pull --all    # celeste too, about 240 MB more
```

Every file is checked against the sizes and hashes this version of Ask Physics pins, and
anything that doesn't match is refused. celeste downloads by itself the first time a
question needs it. Set `ASKPHYSICS_AUTO_PULL=0` to stop both downloads. If a download fails,
`ask` still answers (with a stand-in) and says why, and the next `ask` tries again. Without models, `ask` uses a
stand-in that only knows "dropped from a height" questions, and says so.

## Other commands

| Command | What it does |
|---|---|
| `askphysics version` | The installed version. |
| `askphysics validate-data` | Check every equation, constant, and assumption in the database. |
| `askphysics model info` | List the Fermi models installed, with sizes and validation loss. |
| `askphysics model eval` | Score a model on held-out questions. |

Next: [Reading an answer](Reading-an-Answer).
