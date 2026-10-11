"""The website's Python bundle (``scripts/build_site.sh``) has everything the pipeline imports.

The site runs the package under Pyodide without torch, from a tarball that leaves
the torch modules out. This rebuilds that tarball and answers a question from it
alone, with torch blocked, so a new import breaks here instead of in the browser.
"""

import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

PROBE = """
import json, sys
sys.modules["torch"] = None  # Pyodide has no torch
sys.path.insert(0, sys.argv[1])
import askphysics, askphysics.web as web
assert askphysics.__file__.startswith(sys.argv[1]), askphysics.__file__
answer = json.loads(web.ask("A ball is dropped from 45 m. How fast does it hit the ground?"))
print(json.dumps(answer)[:200])
"""


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("tar") is None, reason="needs sh")
def test_the_site_ships_a_404_page_with_root_relative_links(tmp_path: Path) -> None:
    # Vercel serves the top-level 404.html for missing paths at any depth, such as /docs/x/,
    # so its links must be root-relative or they resolve under the missing path.
    subprocess.run(["sh", "scripts/build_site.sh", str(tmp_path / "site")], cwd=ROOT, check=True)
    page = (tmp_path / "site" / "404.html").read_text(encoding="utf-8")
    assert "probability amplitude" in page
    refs = re.findall(r'(?:href|src)="([^"]*)"', page)
    assert refs
    relative = [r for r in refs if not r.startswith(("/", "#", "https://", "http://"))]
    assert relative == [], relative
    for ref in refs:
        if ref.startswith("/") and not ref.startswith("//"):
            assert (tmp_path / "site" / ref.lstrip("/").split("#")[0]).exists() or ref == "/", ref


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("tar") is None, reason="needs sh")
def test_the_site_bundle_answers_without_torch(tmp_path: Path) -> None:
    subprocess.run(["sh", "scripts/build_site.sh", str(tmp_path / "site")], cwd=ROOT, check=True)
    with tarfile.open(tmp_path / "site" / "py" / "askphysics.tar.gz") as bundle:
        bundle.extractall(tmp_path / "py", filter="data")
    result = subprocess.run(
        [sys.executable, "-c", PROBE, str(tmp_path / "py")],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "answered" in result.stdout


@pytest.mark.skipif(shutil.which("sh") is None or shutil.which("tar") is None, reason="needs sh")
def test_the_install_scripts_and_pages_ship_at_the_short_and_old_paths(tmp_path: Path) -> None:
    site = tmp_path / "site"
    subprocess.run(["sh", "scripts/build_site.sh", str(site)], cwd=ROOT, check=True)
    for script in ("install.sh", "install.ps1"):
        original = (ROOT / script).read_bytes()
        assert (site / script).read_bytes() == original
        assert (site / "installers" / script).read_bytes() == original
    for page in ("installers", "more-installers"):
        html = (site / page / "index.html").read_text(encoding="utf-8")
        assert "<h1>" in html
        refs = re.findall(r'(?:href|src)="([^"]*)"', html)
        assert refs
        relative = [r for r in refs if not r.startswith(("/", "#", "https://", "http://"))]
        assert relative == [], relative
        for ref in refs:
            if ref.startswith("/") and not ref.startswith("//"):
                path = ref.lstrip("/").split("#")[0]
                target = site / path
                if target.is_dir():
                    target = target / "index.html"
                assert target.exists() or path == "", ref
    installers = (site / "installers" / "index.html").read_text(encoding="utf-8")
    assert "curl -fsSL https://askphysics.vercel.app/install.sh | bash" in installers
    assert "irm https://askphysics.vercel.app/install.ps1 | iex" in installers
    assert "releases/latest/download/AskPhysicsSetup.exe" in installers
    assert 'href="/more-installers"' in installers


def test_the_scripts_and_docs_use_the_short_installer_urls() -> None:
    old = re.compile(r"askphysics\.vercel\.app/installers/install\.")
    for name in ("install.sh", "install.ps1", "README.md", "web/index.html"):
        assert not old.search((ROOT / name).read_text(encoding="utf-8")), name
