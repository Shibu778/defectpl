# Changelog

All notable changes to DefectPL are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the project adheres to [Semantic Versioning](https://semver.org/).

---

## [0.3.0] — 2026-06-16

### Added
- `defectpl pr plot` — scatter plot of P-ratio / IPR vs energy **or** band index
  (`--xaxis energy|band`, `--vbm`, `--cbm`, `--kpt` options).
- `defectpl pr ksplot` — Kohn–Sham level diagram with continuous P-ratio colour code.
- `plot_pr_vs_energy()`, `plot_pr_vs_band_index()` in `participation_ratio.py`.
- `plot_ks_with_pr()` in `ks_analysis.py`.
- `flatten_pr_result()` helper for converting nested PR dict to flat rows.
- `defect_entry.json` schema versioning (`schema_version: "1.0"`); reader warns on mismatch.
- `calc_delta_Q()` in `utils.py` — PBC-safe ΔQ from two pymatgen `Structure` objects.
- `get_omega_from_pes()` in `utils.py` — parabolic fit of PES data to extract ℏω in eV.
- `_require_phonopy()` guard in `phonon.py`; all phonopy imports are now lazy.
- pymatgen and phonopy declared as optional extras (`vasp`, `phonon`, `all`) in `pyproject.toml`.
- All pymatgen imports in `vasp.py` made lazy; `OutcarParser` no longer inherits from `Outcar`.
- `docs/theory/` section with phonon, CCD, and P-ratio theory pages.
- `docs/tutorials/` section with four step-by-step tutorials.
- `docs/contributing.md` — developer guide covering docs, PyPI, and conda-forge.
- MkDocs Material theme replacing the old ReadTheDocs theme.
- GitHub Actions: CI matrix (Ubuntu/Windows × Python 3.10/3.11/3.12), docs auto-deploy, and
  PyPI OIDC trusted-publishing workflow.
- Full numpy-style docstrings for all public classes and functions.

### Fixed
- `calc_IPR()` formula corrected from `1 / Σp²` to `Σp² / (Σp)²` (true IPR);
  `localization_ratio` formula updated from `natoms / iprs` to `natoms * iprs` accordingly.
- `plotly` declared as an explicit base dependency (was imported at module level but undeclared).
- Python 3.12 deprecation warnings for `\ ` escape sequences in docstrings resolved.
- Wrong function name `plot_ks_levels` in `docs/api/ks_analysis.md` corrected to
  `plot_spin_resolved_levels`.
- Duplicate `HBAR_JS` import in `utils.py` removed.
- Phonon eigenvector shape documented as `(nmodes, natoms, 3)` throughout.
- `test_phonon.py::test_read_band_yaml` shape assertions corrected to `(6, 2, 3)`.

---

## [0.2.1] — 2025-??-??

### Added
- Participation ratio feature: `ParticipationRatioCalculator`, `read_procar`, PROCAR native parser.
- `defectpl pr` CLI group: `calc`, `batch`, `summary`, `top`, `make-entry`, `make-dsi`.
- `defect_utils.py`: `make_defect_entry()`, `make_defect_structure_info()`.
- `ks_analysis.py`: `extract_ksplot_data()`, `plot_ks_levels()`.
- `defectpl ksplot` CLI command.

---

## [0.2.0] — 2024-??-??

### Added
- `defectpl pl displacement` and `defectpl pl force` CLI commands.
- `VibrationalSpectra1D` class for 1D lineshape with $\omega_1 \neq \omega_2$.
- `ConfigurationCoordinateDiagram` class.
- `defectpl setup-ccd` / `analyze-ccd` CLI commands.
- `defectpl compare-json` / `compare-yaml` comparison commands.

---

## [0.1.0] — 2024-??-??

### Added
- Initial release.
- `Photoluminescence` dataclass with displacement mode and force mode.
- Phonon utilities: `read_band_yaml`, `create_force_constants_from_vasprun`.
- `Plotter` class with ten standard diagnostic plots.
- MSONable JSON serialization for all core objects.
