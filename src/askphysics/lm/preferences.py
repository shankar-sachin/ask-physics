"""The one choice a user can save: whether celeste may download (ADR-022).

tellus and solem download by themselves on the first question; celeste is 240 MB and rarely
needed, so the CLI asks before downloading it, and the answer ``always`` or ``never`` is kept
here so it isn't asked again. The file is ``preferences.json`` inside the models directory
(``$ASKPHYSICS_MODEL_DIR``, default ``~/.cache/askphysics/models``), so it follows the models
and a Homebrew install keeps it beside them. To reset the choice, delete the file.

Torch-free, and read only by the CLI: the website never reads it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast, get_args

from askphysics.config import CelesteDownload
from askphysics.lm.paths import default_model_dir

PREFERENCES_FILE = "preferences.json"
KEY = "celeste_download"


def preferences_path(root: Path | None = None) -> Path:
    """Where the saved choice lives."""
    return (root or default_model_dir()) / PREFERENCES_FILE


def read_celeste_choice(root: Path | None = None) -> CelesteDownload | None:
    """The saved choice, or None when nothing is saved (or the file is unreadable)."""
    try:
        data = json.loads(preferences_path(root).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = data.get(KEY) if isinstance(data, dict) else None
    return cast(CelesteDownload, value) if value in get_args(CelesteDownload) else None


def save_celeste_choice(choice: CelesteDownload, root: Path | None = None) -> Path:
    """Save ``choice`` (keeping any other keys), and return the file.

    Raises:
        OSError: the models directory isn't writable.
    """
    path = preferences_path(root)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        data = data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        data = {}
    data[KEY] = choice
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path
