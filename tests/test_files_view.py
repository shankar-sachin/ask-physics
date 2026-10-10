"""Copying, downloading and listing model files, with sizes and sha256 verdicts."""

import hashlib
import io
import json
import re
from pathlib import Path

import pytest
from rich.console import Console

from askphysics import files_view
from askphysics.files_view import (
    FileFact,
    PullView,
    check_assets,
    copy_tree,
    human_size,
    list_models,
    print_files,
)
from askphysics.ui import make_console


def _console(*, terminal: bool, width: int = 100) -> tuple[Console, io.StringIO]:
    buffer = io.StringIO()
    return make_console(file=buffer, force_terminal=terminal, width=width), buffer


def _shown(buffer: io.StringIO) -> str:
    return re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", buffer.getvalue())


def _model(root: Path, **files: bytes) -> Path:
    root.mkdir(parents=True)
    for name, data in files.items():
        (root / name.replace("__", "/")).write_bytes(data)
    return root


def test_sizes_read_the_way_a_release_page_shows_them() -> None:
    assert human_size(812) == "812 B"
    assert human_size(12_400) == "12.4 KB"
    assert human_size(240_100_000) == "240.1 MB"
    assert human_size(3_200_000_000) == "3.2 GB"


def test_a_copy_checks_every_file_and_says_so(tmp_path: Path) -> None:
    source = _model(
        tmp_path / "models" / "m", **{"model.safetensors": b"w" * 5000, "config.json": b"{}"}
    )
    console, buffer = _console(terminal=True)
    assert copy_tree(console, source, tmp_path / "backup", "Backing up m") == 0
    assert (tmp_path / "backup" / "model.safetensors").read_bytes() == b"w" * 5000
    shown = _shown(buffer)
    digest = hashlib.sha256(b"w" * 5000).hexdigest()[:12]
    assert "◉ Backing up m   2 files · 5.0 KB" in shown
    assert re.search(rf"✓ model\.safetensors\s+5\.0 KB · sha256 {digest} · verified", shown)
    assert "✓ Backing up m" in shown and "2 files · 5.0 KB · sha256 verified" in shown


def test_a_copy_of_nested_files_keeps_the_layout_and_the_modification_time(
    tmp_path: Path,
) -> None:
    source = tmp_path / "m"
    (source / "checkpoints").mkdir(parents=True)
    (source / "checkpoints" / "step-100.json").write_text("{}")
    (source / "eval.json").write_text("{}")
    mtime = 1_700_000_000
    (source / "eval.json").touch()
    import os

    os.utime(source / "eval.json", (mtime, mtime))
    console, _ = _console(terminal=False)
    assert copy_tree(console, source, tmp_path / "copy", "Copy") == 0
    assert (tmp_path / "copy" / "checkpoints" / "step-100.json").read_text() == "{}"
    assert int((tmp_path / "copy" / "eval.json").stat().st_mtime) == mtime


def test_a_copy_that_differs_from_its_source_is_reported_not_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _model(tmp_path / "m", **{"model.safetensors": b"abc"})
    real = files_view.sha256_of
    calls: list[Path] = []

    def corrupt(path: Path, on_bytes: object = None) -> str:
        calls.append(path)
        return "0" * 64 if path.parent.name == "copy" else real(path)  # type: ignore[arg-type]

    monkeypatch.setattr(files_view, "sha256_of", corrupt)
    console, buffer = _console(terminal=True)
    assert copy_tree(console, source, tmp_path / "copy", "Copy") == 1
    shown = _shown(buffer)
    assert "✗ model.safetensors" in shown and "MISMATCH" in shown
    assert "differs: model.safetensors" in shown


def test_a_copy_never_overwrites_a_backup(tmp_path: Path) -> None:
    source = _model(tmp_path / "m", **{"a": b"1"})
    (tmp_path / "taken").mkdir()
    console, buffer = _console(terminal=False)
    assert copy_tree(console, source, tmp_path / "taken", "Copy") == 1
    assert "already exists; not overwriting it" in buffer.getvalue()


def test_a_plain_copy_prints_lines_with_no_escape_codes(tmp_path: Path) -> None:
    source = _model(tmp_path / "m", **{"model.safetensors": b"x" * 2000})
    console, buffer = _console(terminal=False)
    assert copy_tree(console, source, tmp_path / "copy", "Backing up m") == 0
    digest = hashlib.sha256(b"x" * 2000).hexdigest()[:12]
    assert buffer.getvalue().splitlines() == [
        "start: Backing up m (1 files, 2.0 KB)",
        f"  file: model.safetensors (2.0 KB) sha256 {digest} verified",
        "done: Backing up m (0s)",
    ]


def test_many_files_fold_the_small_ones_into_a_count() -> None:
    facts = [FileFact(f"f{n:02d}", 100 + n, "a" * 64, "verified", 0) for n in range(12)]
    console, buffer = _console(terminal=True)
    print_files(console, facts)
    shown = _shown(buffer)
    assert shown.count("✓ f") == 8 and "f11" in shown and "f03" not in shown
    assert "and 4 smaller files" in shown


def test_listing_models_shows_size_files_and_when_each_changed(tmp_path: Path) -> None:
    _model(tmp_path / "fermi-solem-1", **{"model.safetensors": b"x" * 4000, "config.json": b"{}"})
    _model(tmp_path / "fermi-tellus-1", **{"model.safetensors": b"x" * 100})
    console, buffer = _console(terminal=True)
    assert list_models(console, "Installed models", tmp_path) == 0
    shown = _shown(buffer)
    assert re.search(r"fermi-solem-1\s+4\.0 KB\s+2 files\s+\d{4}-\d\d-\d\d \d\d:\d\d", shown)
    assert shown.index("fermi-solem-1") < shown.index("fermi-tellus-1")
    empty, text = _console(terminal=True)
    list_models(empty, "Backups", tmp_path / "nothing")
    assert "(none)" in _shown(text)
    plain, lines = _console(terminal=False)
    list_models(plain, "Backups", tmp_path)
    assert lines.getvalue().splitlines() == [
        f"Backups: {tmp_path}",
        "  fermi-solem-1",
        "  fermi-tellus-1",
    ]


def _release(tmp_path: Path, tamper: bool = False) -> tuple[Path, Path]:
    assets = tmp_path / "assets"
    assets.mkdir()
    pins = {}
    for installed, asset, data in [
        ("model.safetensors", "m.safetensors", b"weights" * 100),
        ("config.json", "m.config.json", b"{}"),
    ]:
        (assets / asset).write_bytes(data)
        pins[installed] = {
            "url": f"https://example.org/releases/{asset}",
            "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    if tamper:
        (assets / "m.safetensors").write_bytes(b"changed")
    manifest = tmp_path / "weights.json"
    manifest.write_text(json.dumps({"format_version": 1, "models": {"m": {"files": pins}}}))
    return assets, manifest


def test_release_files_are_listed_with_their_sha256_checked_against_the_pins(
    tmp_path: Path,
) -> None:
    assets, manifest = _release(tmp_path)
    console, buffer = _console(terminal=True)
    assert check_assets(console, "m", assets, manifest) == 0
    shown = _shown(buffer)
    assert re.search(r"✓ m\.safetensors\s+700 B · sha256 [0-9a-f]{12} · matches the pin", shown)
    assert "m ready to upload" in shown and "all pinned in weights.json" in shown


def test_a_release_file_that_no_longer_matches_its_pin_fails_the_check(tmp_path: Path) -> None:
    assets, manifest = _release(tmp_path, tamper=True)
    (assets / "m.config.json").unlink()
    (assets / "m.extra.md").write_text("not pinned")
    console, buffer = _console(terminal=True)
    assert check_assets(console, "m", assets, manifest) == 1
    shown = _shown(buffer)
    assert "MISMATCH" in shown and "missing from the folder" in shown and "not pinned" in shown
    assert "✗ m ready to upload" in shown


def test_a_pull_shows_a_settled_line_per_file_and_one_for_the_model(tmp_path: Path) -> None:
    console, buffer = _console(terminal=True)
    sha = "ab" * 32
    with PullView(console) as view:
        view.start_model("fermi-tellus-1", [("model.safetensors", 1000), ("config.json", 10)])
        view.update("model.safetensors", 400, 1000)
        view.update("model.safetensors", 1000, 1000)
        view.update("config.json", 10, 10)
        view.finish_model(
            "fermi-tellus-1",
            [("model.safetensors", 1000, sha), ("config.json", 10, sha)],
            True,
            tmp_path / "fermi-tellus-1",
        )
    shown = _shown(buffer)
    assert "◉ fermi-tellus-1   downloading 2 files · 1.0 KB" in shown
    assert re.search(r"✓ model\.safetensors\s+1\.0 KB · sha256 abababababab · verified", shown)
    assert "✓ fermi-tellus-1 downloaded and verified" in shown


def test_a_pull_that_found_nothing_to_do_says_up_to_date_and_a_plain_one_has_no_escapes(
    tmp_path: Path,
) -> None:
    console, buffer = _console(terminal=False)
    with PullView(console) as view:
        view.start_model("m", [("a", 100)])
        view.update("a", 50, 100)
        view.finish_model("m", [("a", 100, "cd" * 32)], False, tmp_path)
    text = buffer.getvalue()
    assert "\x1b" not in text
    assert "start: m (downloading 1 files, 100 B)" in text
    assert "  a: 50% of 100 B" in text
    assert "done: m already up to date (0s)" in text
