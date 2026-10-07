"""Download the published Fermi model weights, checked against a pinned manifest (ADR-012).

The maintainer trains the models; everyone else downloads them. ``weights.json``, shipped
inside the package, pins every published file by URL, size, and sha256, so a download that
doesn't match byte for byte is refused. Each model installs into its own directory under
``default_model_dir()``, and only once every one of its files has arrived and checked out,
so a broken download never leaves a half-installed model behind.

No torch here: ``model pull`` and the installers run this before any model is loaded.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from askphysics.errors import ConfigError
from askphysics.lm.paths import default_model_dir, is_installed

MANIFEST_FORMAT = 1
USER_AGENT = "askphysics-model-pull (+https://github.com/shankar-sachin/ask-physics)"
RELEASE_URL = "https://github.com/shankar-sachin/ask-physics/releases/download/{release}/{asset}"
_CHUNK = 1 << 16

# Called with (file name, bytes so far, total bytes) while a file downloads.
Progress = Callable[[str, int, int], None]


@dataclass(frozen=True)
class PinnedFile:
    """One published file: where it is installed, where it downloads from, and its hash."""

    name: str  # the file name inside the model directory
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class PinnedModel:
    """Every file of one published model."""

    name: str
    release: str
    files: tuple[PinnedFile, ...]


def manifest_path() -> Path:
    """The manifest shipped with the package."""
    return Path(str(resources.files("askphysics.lm").joinpath("weights.json")))


def read_manifest(path: Path | None = None) -> dict[str, PinnedModel]:
    """Published models by name. Empty until the maintainer publishes weights.

    Raises:
        ConfigError: the manifest is malformed or from a newer askphysics.
    """
    data = json.loads((path or manifest_path()).read_text(encoding="utf-8"))
    if data.get("format_version") != MANIFEST_FORMAT:
        raise ConfigError(
            f"weights manifest format {data.get('format_version')} is not {MANIFEST_FORMAT}; "
            "upgrade askphysics"
        )
    models: dict[str, PinnedModel] = {}
    for name, entry in data.get("models", {}).items():
        try:
            files = tuple(
                PinnedFile(name=f, url=v["url"], size=int(v["size"]), sha256=v["sha256"])
                for f, v in entry["files"].items()
            )
            models[name] = PinnedModel(name=name, release=entry["release"], files=files)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"weights manifest entry for {name} is malformed: {exc}") from exc
    return models


def sha256_of(path: Path) -> str:
    """The sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(directory: Path, model: PinnedModel) -> bool:
    """Whether ``directory`` already holds exactly the published files."""
    return all(
        (directory / f.name).is_file()
        and (directory / f.name).stat().st_size == f.size
        and sha256_of(directory / f.name) == f.sha256
        for f in model.files
    )


def _open(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(request, timeout=60)


def download(
    pinned: PinnedFile,
    dest: Path,
    *,
    opener: Callable[[str], Any] | None = None,
    progress: Progress | None = None,
) -> None:
    """Download one file to ``dest``, refusing it unless its size and sha256 match.

    Raises:
        ConfigError: the download failed or doesn't match the manifest.
    """
    digest, size = hashlib.sha256(), 0
    try:
        with (opener or _open)(pinned.url) as response, dest.open("wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                size += len(chunk)
                if size > pinned.size:
                    raise ConfigError(f"{pinned.name} is larger than the manifest says")
                digest.update(chunk)
                out.write(chunk)
                if progress:
                    progress(pinned.name, size, pinned.size)
    except OSError as exc:
        raise ConfigError(f"couldn't download {pinned.name} from {pinned.url}: {exc}") from exc
    if size != pinned.size or digest.hexdigest() != pinned.sha256:
        dest.unlink(missing_ok=True)
        raise ConfigError(
            f"{pinned.name} doesn't match the manifest (got {size:,} bytes, sha256 "
            f"{digest.hexdigest()[:12]}...); refusing it"
        )


def install(
    model: PinnedModel,
    root: Path | None = None,
    *,
    force: bool = False,
    opener: Callable[[str], Any] | None = None,
    progress: Progress | None = None,
) -> bool:
    """Install ``model`` under ``root``; returns False when it was already up to date.

    Every file downloads into a staging directory first and the model directory is
    replaced only once all of them check out.

    Raises:
        ConfigError: a download failed or didn't match, or a model of the same name that
            isn't the published one is installed (a locally trained one) and ``force`` is
            off.
    """
    root = root or default_model_dir()
    target = root / model.name
    if target.is_dir() and matches(target, model):
        return False
    if is_installed(target) and not force:
        raise ConfigError(
            f"{target} holds a different {model.name} (trained locally?); "
            "pass --force to replace it with the published one"
        )
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{model.name}-", dir=root))
    try:
        for pinned in model.files:
            download(pinned, staging / pinned.name, opener=opener, progress=progress)
        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return True


def pull(
    names: Iterable[str],
    root: Path | None = None,
    *,
    manifest: dict[str, PinnedModel] | None = None,
    force: bool = False,
    opener: Callable[[str], Any] | None = None,
    progress: Progress | None = None,
) -> dict[str, bool]:
    """Install each named model; returns name -> whether it was downloaded.

    Raises:
        ConfigError: a name isn't published, or an install failed.
    """
    names = list(names)
    manifest = read_manifest() if manifest is None else manifest
    missing = [n for n in names if n not in manifest]
    if missing:
        published = ", ".join(sorted(manifest)) or "none yet"
        raise ConfigError(
            f"no published weights for {', '.join(missing)} in this askphysics "
            f"(published: {published})"
        )
    return {
        n: install(manifest[n], root, force=force, opener=opener, progress=progress) for n in names
    }


def pin(asset_dir: Path, name: str, release: str, files: dict[str, str]) -> dict[str, Any]:
    """A manifest entry for ``name``'s files in ``asset_dir``.

    ``files`` maps each installed file name to its release asset name, for example
    ``{"model.safetensors": "fermi-solem-1.safetensors"}``.
    """
    return {
        "release": release,
        "files": {
            installed: {
                "url": RELEASE_URL.format(release=release, asset=asset),
                "size": (asset_dir / asset).stat().st_size,
                "sha256": sha256_of(asset_dir / asset),
            }
            for installed, asset in files.items()
        },
    }


def write_manifest(entries: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    """Add or replace ``entries`` in the manifest, keeping the other models."""
    path = path or manifest_path()
    data = json.loads(path.read_text(encoding="utf-8"))
    data["models"] = dict(sorted({**data.get("models", {}), **entries}.items()))
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


__all__ = [
    "PinnedFile",
    "PinnedModel",
    "download",
    "install",
    "manifest_path",
    "matches",
    "pin",
    "pull",
    "read_manifest",
    "sha256_of",
    "write_manifest",
]
