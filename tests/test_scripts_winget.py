"""``scripts/winget.sh``: the WinGet manifests rendered from ``packaging/winget/``."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "winget.sh"
SHA = "ab" * 32
FOLDER = Path("manifests") / "s" / "shankars" / "askphysics" / "0.4.0"

pytestmark = pytest.mark.skipif(shutil.which("sh") is None, reason="needs a POSIX sh")


def render(out: Path, version: str = "0.4.0", sha: str = SHA) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin:/usr/local/bin", "WINGET_DATE": "2026-11-01", "NO_COLOR": "1"}
    return subprocess.run(
        ["sh", str(SCRIPT), version, sha, str(out)],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def load(path: Path) -> dict[str, object]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def test_renders_the_three_manifests_where_winget_pkgs_wants_them(tmp_path: Path) -> None:
    done = render(tmp_path)
    assert done.returncode == 0, done.stderr
    folder = tmp_path / FOLDER
    assert sorted(p.name for p in folder.iterdir()) == [
        "shankars.askphysics.installer.yaml",
        "shankars.askphysics.locale.en-US.yaml",
        "shankars.askphysics.yaml",
    ]


def test_no_placeholder_is_left_and_the_header_is_kept(tmp_path: Path) -> None:
    assert render(tmp_path).returncode == 0
    for path in (tmp_path / FOLDER).iterdir():
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"@[A-Z_0-9]+@", text), path.name
        lines = text.splitlines()
        assert lines[0].startswith(
            "# yaml-language-server: $schema=https://aka.ms/winget-manifest."
        )
        assert lines[0].endswith(".1.12.0.schema.json")
        assert (
            lines[1] == "# Rendered by scripts/winget.sh from packaging/winget/. Edit the template."
        )


def test_every_manifest_names_the_package_and_version(tmp_path: Path) -> None:
    assert render(tmp_path).returncode == 0
    types = set()
    for path in (tmp_path / FOLDER).iterdir():
        data = load(path)
        assert data["PackageIdentifier"] == "shankars.askphysics"
        assert data["PackageVersion"] == "0.4.0"
        assert data["ManifestVersion"] == "1.12.0"
        types.add(data["ManifestType"])
    assert types == {"version", "installer", "defaultLocale"}


def test_installer_manifest_points_at_the_release_asset(tmp_path: Path) -> None:
    assert render(tmp_path).returncode == 0
    data = load(tmp_path / FOLDER / "shankars.askphysics.installer.yaml")
    assert data["InstallerType"] == "inno"
    assert data["Scope"] == "user"
    assert data["Commands"] == ["askphysics"]
    assert data["Dependencies"] == {"PackageDependencies": [{"PackageIdentifier": "astral-sh.uv"}]}
    installers = data["Installers"]
    assert isinstance(installers, list) and len(installers) == 1
    installer = installers[0]
    assert installer["Architecture"] == "x64"
    assert installer["InstallerUrl"] == (
        "https://github.com/shankar-sachin/ask-physics/releases/download/v0.4.0/"
        "AskPhysicsSetup-0.4.0.exe"
    )
    assert installer["InstallerSha256"] == SHA.upper()


def test_product_code_matches_the_app_id_of_the_inno_script(tmp_path: Path) -> None:
    assert render(tmp_path).returncode == 0
    data = load(tmp_path / FOLDER / "shankars.askphysics.installer.yaml")
    installer = data["Installers"][0]
    iss = (ROOT / "packaging" / "windows" / "askphysics.iss").read_text(encoding="utf-8")
    # `AppId={{GUID}` in Inno Setup is the literal text `{GUID}` (the first brace escapes).
    app_id = re.search(r"^AppId=\{(\{[0-9A-F-]+\})", iss, re.MULTILINE)
    assert app_id is not None
    assert installer["ProductCode"] == app_id.group(1) + "_is1"


def test_locale_manifest_has_the_publisher_and_release_notes(tmp_path: Path) -> None:
    assert render(tmp_path).returncode == 0
    data = load(tmp_path / FOLDER / "shankars.askphysics.locale.en-US.yaml")
    assert data["Publisher"] == "Sachin Shankar"
    assert data["PackageName"] == "Ask Physics"
    assert data["License"] == "MIT"
    assert data["PackageUrl"] == "https://askphysics.vercel.app"
    assert data["Moniker"] == "askphysics"
    assert data["ReleaseNotesUrl"].endswith("/releases/tag/v0.4.0")
    assert len(str(data["ShortDescription"])) <= 256


@pytest.mark.parametrize(
    ("version", "sha", "message"),
    [
        ("0.4", SHA, "version must look like"),
        ("v0.4.0", SHA, "version must look like"),
        ("0.4.0", "abc", "64 hex digits"),
        ("0.4.0", "z" * 64, "64 hex digits"),
    ],
)
def test_bad_arguments_are_refused(tmp_path: Path, version: str, sha: str, message: str) -> None:
    done = render(tmp_path, version, sha)
    assert done.returncode == 1
    assert message in done.stderr
    assert not (tmp_path / "manifests").exists()
