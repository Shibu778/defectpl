# Developer Guide

This page covers everything you need to contribute to DefectPL, manage the documentation
website, publish new releases to PyPI, and submit to conda-forge.

---

## 1. Development environment

```bash
git clone https://github.com/Shibu778/defectpl.git
cd defectpl
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pip install pytest pytest-cov ruff
```

Run the test suite:

```bash
pytest -q                          # all tests
pytest -q tests/test_utils.py      # single module
pytest -q --cov=defectpl           # with coverage
```

Run the linter:

```bash
ruff check defectpl/
```

---

## 2. Project structure

```
defectpl/
├── defectpl/           # Python package
│   ├── __init__.py
│   ├── cli.py          # Click command-line entry points
│   ├── constants.py    # Physical constants
│   ├── data.py         # Atomic masses, isotope data
│   ├── defect_utils.py # defect_entry / defect_structure_info generators
│   ├── defectpl.py     # Core: Photoluminescence, VibrationalSpectra1D, CCD
│   ├── ks_analysis.py  # Kohn–Sham eigenvalue analysis
│   ├── participation_ratio.py
│   ├── phonon.py
│   ├── plot.py         # Plotter class
│   ├── utils.py        # Pure math
│   ├── vasp.py         # VASP file I/O (lazy pymatgen)
│   └── vasp_wrapper.py # High-level VASP wrappers
├── tests/              # pytest test suite
├── docs/               # MkDocs documentation source
├── development/        # Papers, benchmarks, scratch scripts
├── pyproject.toml      # Package metadata and dependencies
└── mkdocs.yml          # Documentation site configuration
```

---

## 3. Adding a new feature

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Write the code with **numpy-style docstrings** (see §6 below).
3. Add tests in `tests/test_<module>.py`.
4. If the feature is user-facing, add a CLI command in `cli.py` and document it in
   `docs/command_line_interface.md`.
5. Update `docs/api/<module>.md` to expose the new symbol via mkdocstrings.
6. Open a pull request against `main`.

### Docstring style (numpy)

```python
def calc_delQ(masses: np.ndarray, dR: np.ndarray) -> float:
    """
    Compute the mass-weighted configuration coordinate difference ΔQ.

    Parameters
    ----------
    masses : np.ndarray
        Atomic masses in AMU, shape ``(natoms,)``.
    dR : np.ndarray
        Cartesian displacement matrix in Å, shape ``(natoms, 3)``.

    Returns
    -------
    float
        ΔQ in units of ``sqrt(amu) · Å``.

    Notes
    -----
    The formula is :math:`\\Delta Q = \\sqrt{\\sum_a m_a |\\Delta\\mathbf{R}_a|^2}`.

    Examples
    --------
    >>> import numpy as np
    >>> masses = np.array([12.011, 15.999])
    >>> dR = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
    >>> calc_delQ(masses, dR)
    0.5744...
    """
```

---

## 4. Managing the documentation

### Install docs dependencies

```bash
pip install -r docs/requirements.txt
# mkdocs>=1.6, mkdocs-material>=9.5, mkdocstrings[python]>=0.25
```

### Preview locally

```bash
mkdocs serve
# Opens http://127.0.0.1:8000 with live reload
```

### Build static HTML

```bash
mkdocs build            # output in site/
```

### Deploy to GitHub Pages

#### One-time setup

1. Go to **GitHub → Settings → Pages**.
2. Set Source to **GitHub Actions**.

#### Automatic deployment (recommended)

Create `.github/workflows/docs.yml`:

```yaml
name: Deploy docs

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0          # needed for git-revision-date plugin

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          pip install ".[all]"
          pip install -r docs/requirements.txt

      - name: Deploy
        run: mkdocs gh-deploy --force
```

Push to `main` — GitHub Actions builds the site and pushes it to the `gh-pages` branch
automatically. The site will be live at `https://Shibu778.github.io/defectpl/`.

#### Manual deployment

```bash
mkdocs gh-deploy          # builds + pushes to gh-pages in one step
```

### Versioned docs (optional, using mike)

```bash
pip install mike
mike deploy 0.2 latest --update-aliases
mike set-default latest
mike serve                # preview versioned docs
```

Then replace `mkdocs gh-deploy` in the GitHub Action with:

```bash
git config user.name "github-actions[bot]"
git config user.email "github-actions[bot]@users.noreply.github.com"
mike deploy --push --update-aliases $(python -c "import defectpl; print(defectpl.__version__)") latest
```

---

## 5. Publishing to PyPI

### Prerequisites

- PyPI account at [pypi.org](https://pypi.org)
- API token: **Account settings → API tokens → Add API token** (scope: project `defectpl`)
- Store as GitHub secret `PYPI_API_TOKEN`

### Version bump

Edit `pyproject.toml`:

```toml
[tool.poetry]
version = "0.3.0"
```

Also update `defectpl/__init__.py` if it has a `__version__` string.

### Build and upload

```bash
pip install build twine
python -m build                    # creates dist/*.whl and dist/*.tar.gz
twine check dist/*                 # verify the package
twine upload dist/*                # upload (prompts for credentials)
```

Or with GitHub Actions (create `.github/workflows/publish.yml`):

```yaml
name: Publish to PyPI

on:
  push:
    tags:
      - "v*"

jobs:
  publish:
    runs-on: ubuntu-latest
    environment: release
    permissions:
      id-token: write             # OIDC trusted publishing (no token needed)
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"
      - run: pip install build
      - run: python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
```

Tag a release to trigger:

```bash
git tag v0.3.0
git push origin v0.3.0
```

!!! note "Trusted publishing"
    The workflow above uses PyPI's **OIDC trusted publishing** — no API token secret needed.
    Configure it at **PyPI → Project → Publishing → Add a new publisher → GitHub Actions**.

---

## 6. Publishing to conda-forge

conda-forge uses a **feedstock** repository that is separate from the main package.

### First submission

1. Ensure the package is already on PyPI.
2. Fork [staged-recipes](https://github.com/conda-forge/staged-recipes).
3. Create `recipes/defectpl/meta.yaml` following the
   [conda-forge documentation](https://conda-forge.org/docs/maintainer/adding_pkgs/).
4. Open a pull request — bots will run CI and merge when tests pass.
5. You will be added as a maintainer of the new `defectpl-feedstock` repository.

### Subsequent releases

After a new PyPI release, the `regro-cf-autotick-bot` usually opens a PR automatically in
`defectpl-feedstock` within a few hours. Review and merge it:

```bash
# Clone your feedstock
git clone https://github.com/<your-gh-username>/defectpl-feedstock
cd defectpl-feedstock

# The bot already made a branch; review meta.yaml, check the SHA256 hash
# matches the new PyPI tarball:
pip download defectpl==0.3.0 --no-deps --dest /tmp/dl/
sha256sum /tmp/dl/defectpl-0.3.0.tar.gz

# Approve and merge the bot's PR on GitHub
```

If the bot is slow, update manually:

```yaml
# recipes/meta.yaml — bump version and sha256
{% set version = "0.3.0" %}

source:
  url: https://pypi.io/packages/source/d/defectpl/defectpl-{{ version }}.tar.gz
  sha256: <new-hash>
```

---

## 7. GitHub Actions CI

Create `.github/workflows/ci.yml` to run tests on every push and pull request:

```yaml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest]
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install ".[all]" pytest pytest-cov
      - run: pytest -q --cov=defectpl --cov-report=xml
      - uses: codecov/codecov-action@v4
        with:
          token: ${{ secrets.CODECOV_TOKEN }}
```

---

## 8. Code style

- Formatter: [Ruff](https://docs.astral.sh/ruff/) (`ruff format defectpl/`)
- Linter: Ruff (`ruff check defectpl/`)
- Docstrings: numpy style
- Type hints: standard Python annotations; `from __future__ import annotations` where needed
- Max line length: 99 characters

Add to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 99
target-version = "py310"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```
