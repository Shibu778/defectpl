# DefectPL

A comprehensive toolkit for calculating and visualizing photoluminescence spectra of quantum defects. It also supports the analysis of other optical properties of point defects in insulators and semiconductors.

[![PyPI Version](https://img.shields.io/pypi/v/defectpl.svg)](https://pypi.org/pypi/defectpl)
[![Downloads](https://static.pepy.tech/badge/defectpl)](https://pepy.tech/project/defectpl)
[![Conda Recipe](https://img.shields.io/badge/recipe-defectpl-green.svg)](https://github.com/conda-forge/defectpl-feedstock)
[![Anaconda](https://anaconda.org/conda-forge/defectpl/badges/version.svg)](https://anaconda.org/conda-forge/defectpl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> ⚠️ **This package is currently under active development.**

---

## 📌 Purpose

**DefectPL** is designed to compute the photoluminescence (PL) lineshape and electronic-vibrational coupling profiles of point defects in solids using the formal 1D configuration coordinate model methodologies (*New J. Phys.* **16** 073026 (2014)). 

The package features an automated calculation pipeline to compute, serialize, and visualize:
- Full photoluminescence spectra lineshapes in high Huang-Rhys (HR) factor regimes
- Total & Partial Huang-Rhys factors ($S_k$) alongside Debye-Waller factors ($I_{\text{ZPL}}/I_{\text{tot}}$)
- Phonon localization metrics via Inverse Participation Ratios (IPR) and Localization Ratios
- Multi-mode Electron-Phonon Spectral Densities $S(\omega)$
- Isotope substitution effects on electronic-vibrational coupling pathways

If you use this package in your research, please consider citing:

> [**Carbon with Stone-Wales Defect as Quantum Emitter in h-BN**, *Phys. Rev. B* **111**, 104109 (2025)](https://doi.org/10.1103/PhysRevB.111.104109)

> [**High-throughput Computational Search for Group-IV-related Quantum Defects as Spin-photon Interfaces in 4H-SiC**, *Phys. Rev. B* **112**, 184112 (2025)](https://doi.org/10.1103/PhysRevB.112.184112)

---

## 📚 Documentation

Full documentation is available at: [https://Shibu778.github.io/defectpl/](https://Shibu778.github.io/defectpl/)

---

## 🚀 Installation

Install via **pip**:
```bash
pip install defectpl

```

Install via **conda**:

```bash
conda install conda-forge::defectpl

```

Install from **GitHub (Development Mode)**:

```bash
git clone [https://github.com/Shibu778/defectpl.git](https://github.com/Shibu778/defectpl.git)
cd defectpl
pip install -e .

```

---

## 🧑‍💻 Example Usage

DefectPL natively provides **two calculation paths**: **Displacement Mode** (for structural geometries) and **Force Mode** (for vertical electronic excitation structures). It also inherits from Monty's `MSONable` to safely serialize parameter states directly to lightweight JSON metadata payloads.

### 1. Displacement Mode (Structure Coordinates Vector Track)

```python
from pathlib import Path
from pymatgen.core import Structure
from monty.serialization import dumpfn

from defectpl.phonon import read_band_yaml
from defectpl.vasp_wrapper import calc_dR
from defectpl.defectpl import Photoluminescence

# 1. Parse your input geometries and Phonopy band coordinates
struct_gs = Structure.from_file("CONTCAR_GS")
struct_es = Structure.from_file("CONTCAR_ES")
frequencies, eigenvectors, masses = read_band_yaml("band.yaml")

# 2. Extract PBC-safe displacement vectors matrix (dR)
dR = calc_dR(struct_gs, struct_es)

# 3. Instantiate core engine
pl_engine = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    dR=dR,          # Pass dR for Displacement mode
    dF=None,
    EZPL=1.95,
    gamma=2.0
)

# 4. Generate graphics & Serialize safe properties to JSON via Monty
pl_engine.generate_plots(out_dir="./plots", fig_format="png")
dumpfn(pl_engine, "properties.json", indent=4)

```

### 2. Force Mode (Force Difference Vector Track)

```python
from defectpl.vasp_wrapper import prepare_dF_files
from defectpl.defectpl import Photoluminescence
from defectpl.phonon import read_band_yaml

frequencies, eigenvectors, masses = read_band_yaml("band.yaml")

# Extract vertical force differences (dF = F_excited - F_ground) from OUTCARs
dF = prepare_dF_files("OUTCAR_GS", "OUTCAR_ES")

pl_engine = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    dR=None,
    dF=dF,          # Pass dF for Force mode
    EZPL=1.95,
    gamma=2.0
)
pl_engine.generate_plots(out_dir="./plots", fig_format="png")

```

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome!

If you encounter any issues, please open an issue or submit a pull request.

---

## 👤 Author

**Main Maintainers:** Shibu Meher, Manoj Dey

### Acknowledgements

We gratefully acknowledge the use of several excellent open-source tools that have contributed to the development of this package. This work is inspired by the `PyPhotonics` package, which motivated the development of a more flexible framework for calculating defect-related optical properties using multiple first-principles codes. The `defectpl.mplstyle` file is adapted from the `base.mplstyle` provided in the `sumo` package. We appreciate the high-quality plotting aesthetics and design philosophy of `sumo`, which significantly influenced the visualization components of this project.
