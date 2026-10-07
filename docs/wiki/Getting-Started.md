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

Without trained models installed, `ask` uses the fake model. Trained weights ship with
v0.4 (`askphysics model pull`); until then you can [train your own](Training-the-Models).

## Other commands

| Command | What it does |
|---|---|
| `askphysics version` | The installed version. |
| `askphysics validate-data` | Check every equation, constant, and assumption in the database. |
| `askphysics model info` | List the Fermi models installed, with sizes and validation loss. |
| `askphysics model eval` | Score a model on held-out questions. |

Next: [Reading an answer](Reading-an-Answer).
