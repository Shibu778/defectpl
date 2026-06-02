# DefectPL

A comprehensive toolkit for calculating and visualizing photoluminescence spectra of quantum defects. It also supports the analysis of other optical properties of point defects in insulators and semiconductors.

[![PyPI Version](https://img.shields.io/pypi/v/defectpl.svg)](https://pypi.python.org/pypi/defectpl)
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

## 🚀 Key Features

### 1. Dual Physics Calculation Tracks
The core `Photoluminescence` module accommodates two distinct operational modes for mapping electronic transition coupling matrices depending on your electronic structure data:
- **Displacement Mode ($\Delta R$):** Operates on real periodic-boundary-condition (PBC) safe atomic coordinate shifts ($\mathbf{R}_{\text{Excited}} - \mathbf{R}_{\text{Ground}}$) parsed directly from geometries.
- **Force Mode ($\Delta F$):** Evaluates vertical excitation limits directly by computing force difference matrices ($\mathbf{F}_{\text{Excited}} - \mathbf{F}_{\text{Ground}}$) mapped at identical geometries.

### 2. Seamless VASP Automation & Robust Parsing
- Built-in integrations via `defectpl.vasp_wrapper` cleanly parse `OUTCAR` and structural parameters while checking for configuration consistency.
- Features real-space Gamma-point phonon eigenvector parsing natively extracting shapes directly from Phonopy `band.yaml` outputs.

### 3. Native MSONable Data Serialization
The architecture inherits fully from Monty's `MSONable` design pattern. Call `dumpfn` or export `.as_dict()` properties directly to save your state records to high-fidelity, lightweight JSON payloads. Derived spectral calculations handle complex data pathways cleanly and rehydrate instantly on load.

### 4. Publication-Ready Visualizations
An internal unified `Plotter` engine leverages automated double-y axes frameworks (`twinx`) and multi-variable color mapping to plot calculations against IPR or Localization rules. Matplotlib configurations are backed by a built-in `defectpl.mplstyle` profile designed natively for single-column journals (APS, ACS, and Nature style layouts).

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome!  
If you encounter any issues, please open an issue or submit a pull request.

---

## 👤 Author

**Main Maintainers:** Shibu Meher, Manoj Dey

### Acknowledgements
We gratefully acknowledge the use of several excellent open-source tools that have contributed to the development of this package. This work is inspired by the `PyPhotonics` package, which motivated the development of a more flexible framework for calculating defect-related optical properties using multiple first-principles codes. The `defectpl.mplstyle` file is adapted from the `base.mplstyle` provided in the `sumo` package. We appreciate the high-quality plotting aesthetics and design philosophy of `sumo`, which significantly influenced the visualization components of this project.