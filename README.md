<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)"
          srcset="docs/assets/defectpl-logo-horizontal-reverse.svg">
  <source media="(prefers-color-scheme: light)"
          srcset="docs/assets/defectpl-logo-horizontal.svg">
  <img alt="DefectPL"
       src="docs/assets/defectpl-logo-horizontal.svg"
       width="480">
</picture>

**A unified Python package for the optical properties of point defects in solids.**

[![PyPI](https://img.shields.io/pypi/v/defectpl.svg)](https://pypi.org/project/defectpl)
[![Conda](https://anaconda.org/conda-forge/defectpl/badges/version.svg)](https://anaconda.org/conda-forge/defectpl)
[![Downloads](https://static.pepy.tech/badge/defectpl)](https://pepy.tech/project/defectpl)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](https://python.org)
[![CI](https://github.com/Shibu778/defectpl/actions/workflows/ci.yml/badge.svg)](https://github.com/Shibu778/defectpl/actions/workflows/ci.yml)

</div>

---

DefectPL implements the generating-function formalism of Alkauskas *et al.* (2014) to compute
multi-phonon photoluminescence lineshapes from first-principles VASP + phonopy data.
It is designed for high-throughput workflows: calculations serialize to JSON, plots are
publication-ready out of the box, and the full pipeline is accessible via a `defectpl` CLI.

**[Documentation](https://Shibu778.github.io/defectpl/)** &nbsp;·&nbsp;
**[API Reference](https://Shibu778.github.io/defectpl/api/)** &nbsp;·&nbsp;
**[Tutorials](https://Shibu778.github.io/defectpl/tutorials/)** &nbsp;·&nbsp;
**[CLI Reference](https://Shibu778.github.io/defectpl/command_line_interface/)**

---

## Features

| Capability | Details |
|---|---|
| PL lineshapes | Multi-phonon generating-function spectra; displacement & force mode |
| Temperature | Finite-T spectra via Bose-Einstein occupation and thermal spectral density |
| Absorption | Absorption spectrum from the conjugate generating function |
| HR factors | Total and mode-resolved partial Huang–Rhys factors S_k |
| Spectral density | Continuous S(ω) and C(ω,T) from phonopy band.yaml |
| Phonon localization | IPR, Alkauskas IPR, localization ratio per phonon mode |
| Electronic localization | P-ratio and IPR from VASP PROCAR; KS level diagrams |
| Configuration coordinate | ΔQ, ΔR, parabolic PES fitting, Stokes shift |
| Serialization | Full MSONable JSON round-trip for all engine objects |
| CLI | `defectpl pl displacement`, `pl force`, `pl from-json`, `plot` |
| Plots | 17 standard diagnostic plots; publication-quality PDF/PNG |

---

## Installation

```bash
# Full install (VASP I/O + phonopy)
pip install "defectpl[all]"

# Core only (no pymatgen / phonopy)
pip install defectpl

# conda-forge
conda install -c conda-forge defectpl
```

For a reproducible development environment see [`environment.yaml`](environment.yaml):

```bash
conda env create -f environment.yaml
conda activate defectpl-dev
pip install -e ".[all]"
```

---

## Quick start

### Python API

```python
from pymatgen.core import Structure
from defectpl.phonon import read_band_yaml
from defectpl.io.vasp import calc_dR
from defectpl.defectpl import Photoluminescence

gs = Structure.from_file("CONTCAR_gs")
es = Structure.from_file("CONTCAR_es")
frequencies, eigenvectors, masses = read_band_yaml("band.yaml")

pl = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    dR=calc_dR(gs, es),
    EZPL=1.945,   # zero-phonon line in eV
    gamma=2.0,    # ZPL broadening in meV
)

print(f"Huang-Rhys factor  S = {pl.HR_factor:.3f}")
print(f"Debye-Waller factor  = {pl.DW_factor:.4f}")

pl.generate_plots(out_dir="./plots/", fig_format="png")
```

### CLI

```bash
# Displacement mode — structures + phonopy band.yaml
defectpl pl displacement \
  --band_yaml band.yaml \
  --contcar_gs CONTCAR_gs \
  --contcar_es CONTCAR_es \
  --ezpl 1.945 --gamma 2.0 \
  --temperature 300 \
  --plot_all --json_out pl.json

# Force mode — no excited-state relaxation needed
defectpl pl force \
  --band_yaml band.yaml \
  --outcar_gs OUTCAR_gs \
  --outcar_es OUTCAR_es \
  --ezpl 1.945

# Regenerate all plots from a saved JSON
defectpl pl from-json pl.json

# Standalone plot with type filter
defectpl plot pl.json --type absorption
```

---

## Calculation modes

**Displacement mode** requires fully relaxed ground-state (GS) and excited-state (ES)
geometries plus a phonopy supercell calculation on the GS structure.

**Force mode** only requires the GS relaxation.
The force-difference vector dF = F_ES − F_GS is extracted directly from VASP OUTCARs,
making it suitable for high-throughput screening where ES relaxation is expensive.

---

## Cite

See the full [citation page](https://Shibu778.github.io/defectpl/cite/) for BibTeX entries and method references.
If you use DefectPL in published work please cite:

> Shibu Meher *et al.*,
> **Carbon with Stone-Wales Defect as Quantum Emitter in h-BN**,
> *Phys. Rev. B* **111**, 104109 (2025).
> [doi:10.1103/PhysRevB.111.104109](https://doi.org/10.1103/PhysRevB.111.104109)

> Shibu Meher *et al.*,
> **High-throughput Computational Search for Group-IV-related Quantum Defects as
> Spin-photon Interfaces in 4H-SiC**,
> *Phys. Rev. B* **112**, 184112 (2025).
> [doi:10.1103/PhysRevB.112.184112](https://doi.org/10.1103/PhysRevB.112.184112)

---

## Acknowledgements

DefectPL is inspired by `pyphotonics`, integrating specialized functionalities from `pypl`, `pydefect`, and `nonrad` with the refined plotting aesthetics of `sumo`.

---

## Contributing

Contributions, bug reports, and feature requests are welcome.
See the [Developer Guide](https://Shibu778.github.io/defectpl/contributing/) for
environment setup, pre-commit hooks, coding conventions, and the release workflow.

**Maintainers:** Shibu Meher, Manoj Dey
