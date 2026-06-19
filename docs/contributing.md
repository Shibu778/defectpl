# Developer Guide

This page covers everything you need to contribute to DefectPL: setting up your
environment, running the quality-gate tools locally, managing the documentation
site, publishing releases to PyPI, and submitting to conda-forge.

---

## 1. Development environment

### Conda (recommended)

```bash
conda env create -f environment.yaml   # creates the 'defectpl-dev' environment
conda activate defectpl-dev
pip install -e ".[all]"                # editable install of defectpl itself
```

The `environment.yaml` at the repo root pins all runtime and tooling dependencies.

### Pip only

```bash
git clone https://github.com/Shibu778/defectpl.git
cd defectpl
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e ".[all]"
pip install pytest pytest-cov ruff pre-commit
pip install -r docs/requirements.txt
```

Run the test suite:

```bash
pytest -q                          # all tests
pytest -q tests/test_utils.py      # single module
pytest -q --cov=defectpl           # with coverage
```

---

## 2. Pre-commit hooks

The repo ships a `.pre-commit-config.yaml` that mirrors the CI pipeline so that
lint, format, and test failures are caught locally before you push.

### One-time setup

```bash
pip install pre-commit
pre-commit install                      # installs the pre-commit hook
pre-commit install --hook-type pre-push # installs the pre-push hook (runs pytest)
```

### What runs and when

| Stage | Hook | Trigger |
|---|---|---|
| `pre-commit` | `end-of-file-fixer` | every commit |
| `pre-commit` | `trailing-whitespace` | every commit |
| `pre-commit` | `check-yaml` (excl. `mkdocs.yml`) | every commit |
| `pre-commit` | `check-toml` | every commit |
| `pre-commit` | `check-merge-conflict` | every commit |
| `pre-commit` | `check-added-large-files` (>500 KB) | every commit |
| `pre-commit` | **ruff lint** — auto-fix then exit 1 | every commit |
| `pre-commit` | **ruff format** | every commit |
| `pre-push` | **pytest** `tests/ -q --tb=short` | every push |

> **Why `ruff lint` exits 1 after auto-fixing?**  
> When ruff rewrites a file it exits with a non-zero code so the commit is
> blocked. Re-stage the auto-fixed files (`git add -u`) and commit again.

### Manual dry run

```bash
pre-commit run --all-files
```

### Update hook versions

```bash
pre-commit autoupdate
```

---

## 3. Project structure

```
defectpl/
├── defectpl/           # Python package
│   ├── __init__.py
│   ├── cli.py          # Click command-line entry points
│   ├── constants.py    # Physical constants
│   ├── data.py         # Atomic masses, isotope data
│   ├── defect_utils.py # defect_entry / defect_structure_info generators
│   ├── defectpl.py     # Core: Photoluminescence, VibrationalSpectra1D, CCD
│   ├── ks_analysis.py  # Kohn-Sham eigenvalue analysis
│   ├── participation_ratio.py
│   ├── phonon.py
│   ├── plot.py         # Plotter class
│   ├── utils.py        # Pure math / physics utilities
│   ├── vasp.py         # VASP file I/O (lazy pymatgen)
│   └── vasp_wrapper.py # High-level VASP wrappers
├── tests/              # pytest test suite
├── docs/               # MkDocs documentation source
│   ├── assets/         # SVG logos and icons
│   ├── stylesheets/    # Custom CSS (extra.css)
│   └── javascripts/    # MathJax config (mathjax.js)
├── overrides/          # MkDocs Material theme overrides
│   └── partials/
│       └── logo.html   # Light/dark sidebar logo swap
├── environment.yaml    # Conda dev environment
├── pyproject.toml      # Package metadata and dependencies
├── .pre-commit-config.yaml
└── mkdocs.yml          # Documentation site configuration
```

---

## 4. Adding a new feature

1. Create a feature branch: `git checkout -b feat/my-feature`
2. Write the code with **numpy-style docstrings** (see §7 below).
3. Add tests in `tests/test_<module>.py`.
4. If the feature is user-facing, add a CLI command in `cli.py` and document
   it in `docs/command_line_interface.md`.
5. Update `docs/api/<module>.md` to expose the new symbol via mkdocstrings.
6. Open a pull request against `main`.

---

## 5. Managing the documentation

### Asset inventory

All brand assets live in `docs/assets/`:

| File | Purpose |
|---|---|
| `defectpl-logo-horizontal.svg` | Horizontal logo — light background (README hero, docs hero in light mode) |
| `defectpl-logo-horizontal-reverse.svg` | Horizontal logo — dark background (docs hero in dark mode, GitHub dark README) |
| `defectpl-logo-stacked.svg` | Stacked logo — centered layouts |
| `defectpl-mark-color.svg` | Color mark — sidebar icon in light mode |
| `defectpl-mark-mono-white.svg` | White mark — sidebar icon in dark mode |
| `defectpl-mark-mono-ink.svg` | Dark mark — monochrome use on light backgrounds |
| `defectpl-icon.svg` | Minimal 3-node icon mark |
| `defectpl-appicon.svg` | Rounded-rect app icon — browser favicon |

The sidebar logo is swapped between light and dark mode via
`overrides/partials/logo.html` (sets `logo-light-mode` / `logo-dark-mode` CSS classes)
and the corresponding rules in `docs/stylesheets/extra.css`.

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

A workflow at `.github/workflows/docs.yml` builds and deploys on every push to
`main` that touches `docs/**`, `mkdocs.yml`, `defectpl/**`, or `pyproject.toml`.
It also runs on `workflow_dispatch` (manual trigger from the GitHub UI).

Key design decisions that keep local and deployed builds identical:

- `pip install -r docs/requirements.txt` pins `mkdocs-material>=9.5.0` and
  `pymdown-extensions>=10.7` so CI always uses the same feature set as local.
- `overrides/partials/logo.html` and `docs/stylesheets/extra.css` are part of
  the mkdocs build; they are included automatically because `custom_dir: overrides`
  is set in `mkdocs.yml`.
- `mkdocs gh-deploy --force --clean --verbose` rebuilds from scratch on every
  deploy so stale assets never persist on the `gh-pages` branch.

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

---

## 6. Publishing to PyPI

### Prerequisites

- PyPI account at [pypi.org](https://pypi.org)
- API token: **Account settings → API tokens → Add API token** (scope: project `defectpl`)
- Store as GitHub secret `PYPI_API_TOKEN`

### Version bump

Edit `pyproject.toml`:

```toml
[tool.poetry]
version = "0.4.0"
```

Also update `defectpl/__init__.py` if it has a `__version__` string.

### Build and upload

```bash
pip install build twine
python -m build                    # creates dist/*.whl and dist/*.tar.gz
twine check dist/*                 # verify the package
twine upload dist/*                # upload (prompts for credentials)
```

Or via GitHub Actions (`.github/workflows/publish.yml`) — push a version tag:

```bash
git tag v0.4.0
git push origin v0.4.0
```

The workflow uses PyPI OIDC trusted publishing — no API token secret needed.
Configure it at **PyPI → Project → Publishing → Add a new publisher → GitHub Actions**.

---

## 7. Publishing to conda-forge

conda-forge uses a **feedstock** repository separate from the main package.

### First submission

1. Ensure the package is already on PyPI.
2. Fork [staged-recipes](https://github.com/conda-forge/staged-recipes).
3. Create `recipes/defectpl/meta.yaml` following the
   [conda-forge documentation](https://conda-forge.org/docs/maintainer/adding_pkgs/).
4. Open a pull request — bots will run CI and merge when tests pass.
5. You will be added as a maintainer of the new `defectpl-feedstock` repository.

### Subsequent releases

After a new PyPI release, the `regro-cf-autotick-bot` usually opens a PR in
`defectpl-feedstock` within a few hours. Review and merge it, or update manually:

```yaml
# meta.yaml — bump version and sha256
{% set version = "0.4.0" %}

source:
  url: https://pypi.io/packages/source/d/defectpl/defectpl-{{ version }}.tar.gz
  sha256: <new-hash>
```

---

## 8. Docstring style (numpy)

```python
def calc_delQ(masses: np.ndarray, dR: np.ndarray) -> float:
    """
    Compute the mass-weighted configuration coordinate difference Delta-Q.

    Parameters
    ----------
    masses : np.ndarray
        Atomic masses in AMU, shape ``(natoms,)``.
    dR : np.ndarray
        Cartesian displacement matrix in Ang, shape ``(natoms, 3)``.

    Returns
    -------
    float
        Delta-Q in units of ``sqrt(amu) * Ang``.

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

## 9. Code style

- **Formatter**: [Ruff](https://docs.astral.sh/ruff/) — `ruff format defectpl/ tests/`
- **Linter**: Ruff — `ruff check defectpl/ tests/`
- **Docstrings**: numpy style (see §8)
- **Type hints**: standard Python annotations; `from __future__ import annotations` where needed
- **Max line length**: 99 characters (set in `pyproject.toml`)
- **Pre-commit**: all of the above are enforced automatically; see §2

```toml
# pyproject.toml
[tool.ruff]
line-length = 99
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I"]
```
