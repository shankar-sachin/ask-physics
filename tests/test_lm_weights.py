"""``askphysics.lm.weights``: pinned downloads that refuse anything that doesn't match."""

import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from askphysics.errors import ConfigError
from askphysics.lm.paths import CONFIG_FILE, TOKENIZER_FILE, WEIGHTS_FILE, is_installed
from askphysics.lm.weights import (
    PinnedFile,
    PinnedModel,
    manifest_path,
    pin,
    pull,
    read_manifest,
    write_manifest,
)

FILES = {WEIGHTS_FILE: b"weights" * 100, CONFIG_FILE: b'{"name": "x"}', TOKENIZER_FILE: b"{}"}


def _published(name: str = "fermi-tellus-1", files: dict[str, bytes] = FILES) -> PinnedModel:
    return PinnedModel(
        name=name,
        release="models-test",
        files=tuple(
            PinnedFile(f, f"https://example.test/{name}/{f}", len(b), hashlib.sha256(b).hexdigest())
            for f, b in files.items()
        ),
    )


class Server:
    """Serves bytes by URL and counts requests; ``serve`` overrides what a URL returns."""

    def __init__(self, files: dict[str, bytes] = FILES, name: str = "fermi-tellus-1") -> None:
        self.serve = {f"https://example.test/{name}/{f}": b for f, b in files.items()}
        self.requests: list[str] = []

    def __call__(self, url: str) -> Any:
        self.requests.append(url)
        if url not in self.serve:
            raise OSError("404")
        return io.BytesIO(self.serve[url])


def test_the_shipped_manifest_reads(tmp_path: Path) -> None:
    assert isinstance(read_manifest(), dict)
    assert json.loads(manifest_path().read_text())["format_version"] == 1


def test_pull_installs_every_file(tmp_path: Path) -> None:
    model, server = _published(), Server()
    progress: list[tuple[str, int, int]] = []
    done = pull([model.name], tmp_path, manifest={model.name: model}, opener=server,
                progress=lambda *a: progress.append(a))  # fmt: skip
    assert done == {model.name: True}
    assert is_installed(tmp_path / model.name)
    assert (tmp_path / model.name / WEIGHTS_FILE).read_bytes() == FILES[WEIGHTS_FILE]
    assert progress[-1][1] == progress[-1][2]
    # Pulling again downloads nothing.
    assert pull([model.name], tmp_path, manifest={model.name: model}, opener=server) == {
        model.name: False
    }
    assert len(server.requests) == len(FILES)


@pytest.mark.parametrize("tampered", [b"weights" * 99 + b"evilevi", b"weights" * 100 + b"x"])
def test_a_file_that_does_not_match_is_refused(tmp_path: Path, tampered: bytes) -> None:
    model, server = _published(), Server()
    server.serve[f"https://example.test/{model.name}/{WEIGHTS_FILE}"] = tampered
    with pytest.raises(ConfigError, match="manifest"):
        pull([model.name], tmp_path, manifest={model.name: model}, opener=server)
    # Nothing half-installed, and no staging directory left behind.
    assert not (tmp_path / model.name).exists()
    assert list(tmp_path.iterdir()) == []


def test_a_failed_download_is_a_config_error(tmp_path: Path) -> None:
    model = _published()
    with pytest.raises(ConfigError, match="couldn't download"):
        pull([model.name], tmp_path, manifest={model.name: model}, opener=Server(files={}))


def test_a_locally_trained_model_is_kept_unless_forced(tmp_path: Path) -> None:
    model, server = _published(), Server()
    local = tmp_path / model.name
    local.mkdir()
    for f in FILES:
        (local / f).write_bytes(b"my own training run")
    with pytest.raises(ConfigError, match="--force"):
        pull([model.name], tmp_path, manifest={model.name: model}, opener=server)
    assert (local / WEIGHTS_FILE).read_bytes() == b"my own training run"
    pull([model.name], tmp_path, manifest={model.name: model}, opener=server, force=True)
    assert (local / WEIGHTS_FILE).read_bytes() == FILES[WEIGHTS_FILE]


def test_an_unpublished_model_says_what_is_published(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="published: none yet"):
        pull(["fermi-solem-1"], tmp_path, manifest={})


def test_pin_and_write_manifest(tmp_path: Path) -> None:
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "fermi-solem-1.safetensors").write_bytes(b"abc")
    entry = pin(assets, "fermi-solem-1", "models-v0.4.0",
                {WEIGHTS_FILE: "fermi-solem-1.safetensors"})  # fmt: skip
    pinned = entry["files"][WEIGHTS_FILE]
    assert pinned["size"] == 3 and pinned["sha256"] == hashlib.sha256(b"abc").hexdigest()
    assert pinned["url"].endswith("/releases/download/models-v0.4.0/fermi-solem-1.safetensors")
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"format_version": 1, "models": {"fermi-tellus-1": {
        "release": "old", "files": {}}}}))  # fmt: skip
    write_manifest({"fermi-solem-1": entry}, manifest)
    models = read_manifest(manifest)
    assert set(models) == {"fermi-solem-1", "fermi-tellus-1"}
    assert models["fermi-solem-1"].files[0].size == 3


def test_a_newer_manifest_format_is_refused(tmp_path: Path) -> None:
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"format_version": 99, "models": {}}))
    with pytest.raises(ConfigError, match="upgrade askphysics"):
        read_manifest(manifest)


def test_missing_published_lists_only_published_models_that_are_not_installed(
    tmp_path: Path,
) -> None:
    from askphysics.lm.weights import missing_published

    manifest = {m.name: m for m in (_published("fermi-tellus-1"), _published("fermi-solem-1"))}
    pull(["fermi-tellus-1"], tmp_path, manifest=manifest, opener=Server())
    names = ["fermi-tellus-1", "fermi-solem-1", "fermi-celeste-1"]
    assert missing_published(names, tmp_path, manifest=manifest) == ["fermi-solem-1"]
    assert missing_published(names, tmp_path, manifest={}) == []
