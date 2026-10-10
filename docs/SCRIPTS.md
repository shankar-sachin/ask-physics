# The workflow scripts

Every script in `scripts/` (and the installers) shares one look: a header, a step per line, a
summary. A step is a spinner that settles into a tick and the time it took. The colours and
symbols come from the theme in `src/askphysics/ui.py`, so the scripts and the CLI read as one
product.

```
  ◉ Ask Physics  ·  Check
    Everything that must pass before a commit  ·  6 steps

  ✓ 1/6 ruff check      All checks passed!                                        0s
  ✓ 2/6 ruff format     72 files already formatted                                0s
  ✓ 3/6 mypy            Success: no issues found in 57 source files               4s
  ✓ 4/6 shellcheck      no findings                                               1s
  ✓ 5/6 validate-data   seed data: all valid                                      1s
  ⠹ 6/6 pytest          412 passed · 1 failed  34%                            1m 12s
      │ tests/test_lm_train.py ..........                              [ 34%]
```

## What each script shows

| Script | Look |
|---|---|
| `check.sh` | One line per gate (ruff, format, mypy, shellcheck, validate-data, pytest) that turns into a tick or a cross. pytest shows live counts, then `1284 passed · 3 skipped`. A failed gate shows its command and the end of its output, the rest still run, and the summary names what failed (exit 1). |
| `eval.sh` | A bar over the examples with an ETA from the first example and a clock time to finish, the valid-plan rate and confidently-wrong count as they settle, and its own bar and ETA for the `--rescue-with` pass. Each pass settles into a line; the results close as a panel with a bar per check. |
| `setup.sh`, `update.sh` | A step list with ticks. `setup.sh` ends on a panel of commands to try. |
| `install.sh`, `install.ps1` | The same step list, drawn without Python (they run before anything is installed), and the same panel. |
| `models.sh` | Backups and restores copy file by file under a bar with size, speed and ETA, and check every copy's sha256 against its source. `list` is a table. |
| `release_weights.sh` | Package, score, and then each release file with its size and whether its sha256 matches the pin in `weights.json`. |
| `corpus.sh` | One step with live counts: the OpenStax stage settles into a line of books and words; Gutenberg shows words so far against the target. |
| `build_site.sh` | Three stages with counts: files written, files in the Python bundle, pages rendered. |
| `train.sh` | A header, a phase per step (see `TRAINING.md`), and a closing summary. |

## Long commands

A long child command (pip, pytest, a data build, a download) runs under its spinner with the last
three lines of its output underneath, dimmed. The full output is saved to
`$TMPDIR/askphysics-logs/<script>-<time>/<NN>-<step>.log` (`ASKPHYSICS_LOG_DIR` changes the
folder). When a step fails, the cross is followed by the command, the last lines of its output,
and the path of that log.

## Without a terminal

In CI, in `| tee`, with `NO_COLOR` set, or with `TERM=dumb`, there is no colour, no animation,
and no escape code. Every step prints plain lines, and the command's own output goes straight
through:

```
Ask Physics: Check (6 steps)
  Everything that must pass before a commit
start: [1/6] ruff check
All checks passed!
done: [1/6] ruff check (0s)
...
ok: All clear (6 of 6 steps, 1m 52s)
  next: git push -u origin HEAD  - publish the branch
```

`DRY_RUN=1` also prints plain lines: `-- <step title>` and then `+ <command>`.

## How it is built

- `scripts/lib.sh` is what a script sources. Its helpers: `ui_begin TITLE ABOUT [STEPS [PAD]]`,
  `phase TITLE COMMAND...` (fails the script), `gate TITLE PARSE COMMAND...` (records the failure and
  goes on), `phase_live` (the command draws its own display), `ui_result`, `ui_ready`, `ui_end`,
  and `info`, `warn`, `fail`.
- The drawing is `python -m askphysics.shell_ui` (`src/askphysics/shell_ui.py`), Rich only, with
  `train_ui.py` (the step), `eval_view.py` (the eval bars and results) and `files_view.py` (copies,
  downloads, release files) behind it. It imports no sympy, Pint, torch, or pydantic, so each call
  starts in a fraction of a second.
- When `askphysics` cannot be imported (before `setup.sh` has run, on a machine with no Python,
  on Vercel), `lib.sh` draws with a pure-sh renderer instead: the same glyphs, columns and
  colours, without the folded output lines and bars. `install.sh` embeds the same renderer, since
  it is piped from `curl` and cannot source `lib.sh`; `tests/test_scripts_lib.py` keeps the two
  copies identical. `install.ps1` has the matching look in PowerShell.
- `askphysics-dev model eval` and `askphysics-dev model pull` draw their own views (the eval bar, the
  download bars), so they look the same whether a script or a person runs them.

## Adding a script

Source `lib.sh`, call `ui_begin` with the number of steps, run each long command with `phase`
(or `gate`), and finish with `ui_end`. Keep it POSIX `sh` and shellcheck-clean, print
`DRY_RUN` commands through `run`/`awake`, and test the plain output in
`tests/test_scripts_*.py`.
