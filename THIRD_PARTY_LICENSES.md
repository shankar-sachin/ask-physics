# Third-party licenses

Ask Physics itself is [MIT licensed](LICENSE). It builds on the open-source
software, fonts, and data below, each under its own license. Nothing here is
relicensed; every component keeps its original terms, and the links point to
each project's full license text.

How each part reaches you:

- **Runtime libraries** are installed alongside Ask Physics (by pip, uv, or
  Homebrew) from their own projects; we don't copy their code into ours.
- **The website** loads Pyodide and its packages from the jsDelivr CDN and Pint
  from PyPI, in your browser; fonts come from Google Fonts.
- **Fonts and imagery** are embedded in our artwork (the logo, banner, and
  model portraits in `docs/images/`).
- **Development tools** are used to build and test the project and are not
  part of what users install.

Last reviewed for v0.3.0. When a dependency is added, add it here.

## Runtime libraries

What `askphysics` installs (`pyproject.toml` dependencies and their
dependencies).

| Package | License | Copyright | Project |
|---------|---------|-----------|---------|
| SymPy | BSD-3-Clause | SymPy Development Team | <https://github.com/sympy/sympy> |
| mpmath | BSD-3-Clause | Fredrik Johansson and mpmath contributors | <https://github.com/mpmath/mpmath> |
| Pint | BSD-3-Clause | Hernan E. Grecco and contributors | <https://github.com/hgrecco/pint> |
| flexcache | BSD-3-Clause | Hernan E. Grecco and contributors | <https://github.com/hgrecco/flexcache> |
| flexparser | BSD-3-Clause | Hernan E. Grecco and contributors | <https://github.com/hgrecco/flexparser> |
| pydantic | MIT | Pydantic Services Inc. and individual contributors | <https://github.com/pydantic/pydantic> |
| pydantic-core | MIT | Samuel Colvin | <https://github.com/pydantic/pydantic-core> |
| annotated-types | MIT | the annotated-types contributors | <https://github.com/annotated-types/annotated-types> |
| typing-inspection | MIT | Pydantic Services Inc. | <https://github.com/pydantic/typing-inspection> |
| typing-extensions | PSF-2.0 | Python Software Foundation | <https://github.com/python/typing_extensions> |
| NumPy | BSD-3-Clause (bundled parts: 0BSD, MIT, Zlib, CC0-1.0) | NumPy Developers | <https://github.com/numpy/numpy> |
| PyTorch (`torch`) | BSD-3-Clause (bundled parts: Apache-2.0, Apache-2.0 WITH LLVM-exception, BSD-2-Clause, BSL-1.0, MIT) | Facebook, Inc. (Meta) and contributors | <https://github.com/pytorch/pytorch> |
| filelock | MIT | Bernát Gábor and contributors | <https://github.com/tox-dev/filelock> |
| fsspec | BSD-3-Clause | Martin Durant and contributors | <https://github.com/fsspec/filesystem_spec> |
| Jinja2 | BSD-3-Clause | Pallets | <https://github.com/pallets/jinja> |
| MarkupSafe | BSD-3-Clause | Pallets | <https://github.com/pallets/markupsafe> |
| NetworkX | BSD-3-Clause | NetworkX Developers | <https://github.com/networkx/networkx> |
| setuptools | MIT | Jason R. Coombs and contributors | <https://github.com/pypa/setuptools> |
| safetensors | Apache-2.0 | Hugging Face | <https://github.com/huggingface/safetensors> |
| MLX (`mlx`, the `mlx` extra on arm64 Macs; `mlx-cpu`, the `dev` extra on Linux; ADR-019) | MIT | Apple Inc. and the MLX contributors | <https://github.com/ml-explore/mlx> |
| Typer | MIT | Sebastián Ramírez | <https://github.com/fastapi/typer> |
| annotated-doc | MIT | Sebastián Ramírez | <https://github.com/fastapi/annotated-doc> |
| shellingham | ISC | Tzu-ping Chung | <https://github.com/sarugaku/shellingham> |
| Rich | MIT | Will McGugan | <https://github.com/Textualize/rich> |
| markdown-it-py | MIT | Vitaly Puzrin, Alex Kocharin, and contributors | <https://github.com/executablebooks/markdown-it-py> |
| mdurl | MIT | Vitaly Puzrin, Alex Kocharin | <https://github.com/executablebooks/mdurl> |
| Pygments | BSD-2-Clause | the Pygments authors | <https://github.com/pygments/pygments> |
| platformdirs | MIT | the platformdirs developers | <https://github.com/tox-dev/platformdirs> |

PyTorch's own license file lists the many projects it bundles; see its
[`LICENSE`](https://github.com/pytorch/pytorch/blob/main/LICENSE) and
[`NOTICE`](https://github.com/pytorch/pytorch/blob/main/NOTICE). On Linux our
installers fetch the CPU-only PyTorch build. If you install a CUDA build
yourself, PyTorch also pulls in NVIDIA's CUDA libraries (for example
`nvidia-cublas`, `nvidia-cudnn`) and Triton (MIT); the NVIDIA libraries are
under the [NVIDIA Software License](https://docs.nvidia.com/cuda/eula/), not
an open-source license, and Ask Physics never ships them.

## The website

| Component | License | Copyright | Project |
|-----------|---------|-----------|---------|
| Pyodide (Python on WebAssembly, incl. CPython and its packaged libraries) | MPL-2.0 (CPython: PSF-2.0; packages under their own licenses) | the Pyodide developers | <https://github.com/pyodide/pyodide> |
| SymPy, pydantic, NumPy, mpmath, typing-extensions, platformdirs (as built by Pyodide) | as listed above | as listed above | via <https://pyodide.org> |
| Pint, flexcache, flexparser (from PyPI) | BSD-3-Clause | as listed above | via <https://pypi.org> |
| Space Grotesk (font) | OFL-1.1 | The Space Grotesk Project Authors | <https://github.com/floriankarsten/space-grotesk> |
| JetBrains Mono (font) | OFL-1.1 | The JetBrains Mono Project Authors | <https://github.com/JetBrains/JetBrainsMono> |
| Inter (font) | OFL-1.1 | The Inter Project Authors | <https://github.com/rsms/inter> |
| KaTeX 0.16.11 (typesets formulas on the docs pages; loaded from jsDelivr) | MIT | Khan Academy and other contributors | <https://github.com/KaTeX/KaTeX> |
| Octicons (GitHub mark in the docs page header, inlined) | MIT | GitHub Inc. | <https://github.com/primer/octicons> |
| micropip (installs packages inside Pyodide) | MPL-2.0 | the Pyodide developers | <https://github.com/pyodide/micropip> |

The site is static: these load in the visitor's browser from their own
servers (jsDelivr, PyPI, Google Fonts). Ask Physics's own code on the site is
MIT.

## Fonts and imagery in our artwork

| Asset | Where we use it | License | Credit |
|-------|-----------------|---------|--------|
| STIX Two Text | The π in the logo (`docs/images/logo.svg`) is the STIX Two Text glyph outline; equations on the model portraits are set in it | OFL-1.1 | The STIX Fonts Project Authors, <https://github.com/stipub/stixfonts> |
| Space Grotesk, JetBrains Mono | Text in the banner and model portraits | OFL-1.1 | as above |
| NASA Blue Marble Next Generation | Earth surface in the `fermi-tellus-1` portrait and banner | Public domain (NASA) | NASA Earth Observatory, <https://visibleearth.nasa.gov/collection/1484/blue-marble> |
| NOAA ETOPO1 | Terrain relief in the `fermi-tellus-1` portrait | Public domain (U.S. Government) | NOAA National Centers for Environmental Information |
| Solar System Scope Moon map | Lunar surface in the `fermi-luna-1` portrait and banner | CC BY 4.0 | [Solar System Scope](https://www.solarsystemscope.com/textures/), based on NASA LRO data |

`scripts/brand.py` downloads the Earth maps from the
[basemap-data](https://pypi.org/project/basemap-data/) package (its packaging is
LGPL-3.0-or-later; the maps themselves are public domain) and the Moon map
from the [PyVista data repository](https://github.com/pyvista/data). The Sun
and the black hole are computed from scratch, with no third-party imagery.

## Data

Every entry in `src/askphysics/data/` is written by the project and released
under MIT (each entry records its own `source` and `license`). Entries cite
published references: OpenStax University Physics for equations (cited as a
reference only; its text is CC BY-NC-SA 4.0 and none of it is copied),
CODATA 2022 and the SI (NIST) for constants, and public sources such as USDA
FoodData Central and UN World Population Prospects for Fermi assumptions. No
text is copied from these references; equations and physical constants are
facts. Training data is generated by our own data factory (ADR-009).

`third_party/openstax-physics/` is different: `prose.jsonl`, `questions.jsonl`,
and the questions in `real_eval.jsonl` are text from
OpenStax *Physics* (2020) by Fatih Gozuacik, Denise Pattison, and Catherine
Tabor, published by OpenStax (Rice University), licensed **CC BY 4.0**, and
extracted and reformatted by this project (ADR-016). It keeps that license
(`third_party/openstax-physics/LICENSE`) and is not covered by MIT; its
`ATTRIBUTION.md` lists the source, commit, and every change. The gold plans in
`real_eval.jsonl` were written by this project. Fermi weights trained with
`--prose`, or on data built with its questions, carry that attribution in their
model cards. It is used for training and evaluation only and is not shipped in
the Python package.

The larger prose corpus (ADR-017) is built locally by `scripts/build_corpus.py`
and never committed. It combines 52 OpenStax books under **CC BY 4.0**, read
at commits from before OpenStax relicensed them (listed with their authors in
the corpus's generated `ATTRIBUTION.md` and pinned in
`third_party/corpus/sources.json`), with public-domain books from Project
Gutenberg. Weights trained on it carry that attribution in their model cards.

## Installers and distribution

| Tool | License | Project |
|------|---------|---------|
| uv (installed by `install.sh` and `install.ps1` if missing, and used by the Homebrew formula) | MIT OR Apache-2.0 | <https://github.com/astral-sh/uv> |
| Homebrew (runs the `shankar-sachin/tap` formula) | BSD-2-Clause | <https://github.com/Homebrew/brew> |

## Development tools

Used to build, test, and render the project; not installed for users.

| Tool | License | Project |
|------|---------|---------|
| pytest | MIT | <https://github.com/pytest-dev/pytest> |
| pytest-cov | MIT | <https://github.com/pytest-dev/pytest-cov> |
| coverage.py | Apache-2.0 | <https://github.com/coveragepy/coveragepy> |
| Ruff | MIT | <https://github.com/astral-sh/ruff> |
| mypy | MIT | <https://github.com/python/mypy> |
| PyYAML | MIT | <https://github.com/yaml/pyyaml> |
| types-PyYAML (typeshed) | Apache-2.0 | <https://github.com/python/typeshed> |
| Hatchling (build backend) | MIT | <https://github.com/pypa/hatch> |
| Playwright (screenshots, artwork, website tests) | Apache-2.0 | <https://github.com/microsoft/playwright> |
| Chromium (headless, via Playwright) | BSD-3-Clause and others | <https://www.chromium.org> |
| Pillow (`scripts/brand.py`) | MIT-CMU | <https://github.com/python-pillow/Pillow> |
| fontTools (extracting the logo's π glyph) | MIT | <https://github.com/fonttools/fonttools> |
| GitHub Actions: checkout, setup-python, setup-node, upload-artifact | MIT | <https://github.com/actions> |
| Homebrew/actions (tap CI) | BSD-2-Clause | <https://github.com/Homebrew/actions> |

## License texts

The full text of each license named above:

- MIT: <https://opensource.org/license/mit>
- BSD-2-Clause: <https://opensource.org/license/bsd-2-clause>
- BSD-3-Clause: <https://opensource.org/license/bsd-3-clause>
- ISC: <https://opensource.org/license/isc-license-txt>
- Apache-2.0: <https://www.apache.org/licenses/LICENSE-2.0>
- PSF-2.0: <https://docs.python.org/3/license.html>
- MPL-2.0: <https://www.mozilla.org/en-US/MPL/2.0/>
- OFL-1.1 (SIL Open Font License): <https://openfontlicense.org>
- CC BY 4.0: <https://creativecommons.org/licenses/by/4.0/>
- MIT-CMU (Pillow): <https://github.com/python-pillow/Pillow/blob/main/LICENSE>
- BSL-1.0, 0BSD, Zlib, CC0-1.0 (bundled in NumPy and PyTorch): <https://spdx.org/licenses/>

Each installed package also carries its own license file; find it with
`pip show -f <package>` (look under `licenses/` in the `.dist-info` folder).
