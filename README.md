# DefectPL
A comprehensive toolkit for calculating and visualizing photoluminescence spectra of quantum defects. It also supports the analysis of other optical properties of point defects in insulators and semiconductors.

[![image](https://img.shields.io/pypi/v/defectpl.svg)](https://pypi.python.org/pypi/defectpl)
[![Downloads](https://static.pepy.tech/badge/defectpl)](https://pepy.tech/project/defectpl)
[![Conda Recipe](https://img.shields.io/badge/recipe-defectpl-green.svg)](https://github.com/conda-forge/defectpl-feedstock)
[![Anaconda](https://anaconda.org/conda-forge/defectpl/badges/version.svg)](https://anaconda.org/conda-forge/defectpl)
[![image](https://img.shields.io/conda/vn/conda-forge/defectpl.svg)](https://anaconda.org/conda-forge/defectpl)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/defectpl.svg)](https://anaconda.org/conda-forge/defectpl)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Shibu778/defectpl)

> ⚠️ **This package is currently under active development.**

---

## 📌 Purpose

**DefectPL** is designed to compute the photoluminescence intensity of point defects in solids using the methodology described in *New J. Phys.* **16** 073026 (2014). It also provides tools to calculate and plot related quantities such as:

- Partial Huang-Rhys factors
- Huang-Rhys factor
- Debye-Waller factor
- Inverse participation ratios (IPR)  
- Localization ratios  
- Vibrational displacements  
- Effect of Isotope substitution
- Photoluminescence Spectra in the High Huang-Rhys Factor Regime

If you use this package in your research, please consider citing:

> [**Carbon with Stone-Wales Defect as Quantum Emitter in h-BN**, *Phys. Rev. B* **111**, 104109 (2025)](https://doi.org/10.1103/PhysRevB.111.104109)

> [Read the article](https://doi.org/10.1103/PhysRevB.111.104109)

> [**High-throughput Computational Search for Group-IV-related Quantum Defects as Spin-photon Interfaces in 4H-SiC**, *Phys. Rev. B* **112**, 184112 (2025)](https://doi.org/10.1103/lsxj-nvhw)

> [Read the article](https://doi.org/10.1103/lsxj-nvhw)

---

## 📚 Documentation

Full documentation is available at: https://Shibu778.github.io/defectpl/
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

Install from **GitHub**:

```bash
git clone https://github.com/Shibu778/defectpl.git
cd defectpl/defectpl
pip install -e .
```

---

## 🧑‍💻 Example Usage

Here’s a minimal example using data for a negative NV center in diamond:

```python
from defectpl.defectpl import DefectPl

band_yaml = "../tests/data/band.yaml"
contcar_gs = "../tests/data/CONTCAR_gs"
contcar_es = "../tests/data/CONTCAR_es"
out_dir = "./plots"
EZPL = 1.95
gamma = 2
plot_all = True
iplot_xlim = [1000, 2000]

defctpl = DefectPl(
    band_yaml,
    contcar_gs,
    contcar_es,
    EZPL,
    gamma,
    iplot_xlim=iplot_xlim,
    plot_all=plot_all,
    out_dir=out_dir,
)
```

### Plots Gallery

| Intensity vs Phonon Energy | Spectral Function, Partial HR factor and Localization Ratio |
| :------------------------: | :----------------------------: |
| ![intensity-photon-energy] | ![somega-pHR-locrat-penergy]   |
| Vibrational Displacement | Phonon Energy |
| ![vibrational-displacement] | ![phonon-energy]   |
| Inverse Participation Ratio | Localization Ratio |
| ![ipr] | ![loc_ratio] |
| Partial HR factor (pHR) | Spectral Function, pHR |
| ![pHR] | ![S_pHR] |
| Spectral Function, Partial HR factor and IPR | One Dimensional Vibrational Spectra |
| ![S_ipr] | ![oned] |

[intensity-photon-energy]: docs/plots/intensity_vs_penergy.svg
[somega-pHR-locrat-penergy]: docs/plots/S_omega_HRf_loc_rat_vs_penergy.svg
[vibrational-displacement]: docs/plots/qk_vs_penergy.svg
[phonon-energy]: docs/plots/penergy_vs_pmode.svg
[ipr]: docs/plots/ipr_vs_penergy.svg
[loc_ratio]: docs/plots/loc_rat_vs_penergy.svg
[pHR]: docs/plots/HR_factor_vs_penergy.svg
[S_pHR]: docs/plots/S_omega_vs_penergy.svg
[S_ipr]: docs/plots/S_omega_HRf_ipr_vs_penergy.svg
[oned]: docs/plots/one_d_lineshape.svg

---

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome!  
If you encounter any issues, please open an issue or submit a pull request.

---

## 👤 Author

**Main Maintainer:** Shibu Meher, Manoj Dey

## Acknowledgement

We gratefully acknowledge the use of several excellent open-source tools that have contributed to the development of this package. This work is inspired by the `PyPhotonics` package, which motivated the development of a more flexible framework for calculating defect-related optical properties using multiple first-principles codes. The `defectpl.mplstyle` file is adapted from the `base.mplstyle` provided in the `sumo` package. We also appreciate the high-quality plotting aesthetics and design philosophy of `sumo`, which significantly influenced the visualization components of this project.
