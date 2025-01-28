# DefectPl
A unified package to calculate and plot optical properties of point defects in insulators and semiconductors.

[![image](https://img.shields.io/pypi/v/defectpl.svg)](https://pypi.python.org/pypi/defectpl)
[![Downloads](https://static.pepy.tech/badge/defectpl)](https://pepy.tech/project/defectpl)
[![Conda Recipe](https://img.shields.io/badge/recipe-defectpl-green.svg)](https://github.com/conda-forge/defectpl-feedstock)
[![Anaconda](https://anaconda.org/conda-forge/defectpl/badges/version.svg)](https://anaconda.org/conda-forge/defectpl)
[![image](https://img.shields.io/conda/vn/conda-forge/defectpl.svg)](https://anaconda.org/conda-forge/defectpl)
[![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/defectpl.svg)](https://anaconda.org/conda-forge/defectpl)
[![image](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**This Package is Currently Under Active Development.**

#### Purpose of the Package
The purpose of this package is to calculate the intensity of photoluminescence from point defects in solids with method described in New J. Phys. 16 (2014) 073026. It also calculates and plot other relevant quantities like partial Huang Rhys factor, inverse participation ratio etc.

### Documentation
For documentation check : [Documentations](https://Shibu778.github.io/defectpl/)

### Getting Started
The package can be found in pypi. You can install it using `pip`.

#### Installation

```bash
pip install defectpl
```

Using `conda`
```bash
conda install conda-forge::defectpl
```

Using the GitHub clone

```bash
git clone https://github.com/Shibu778/defectpl.git
cd defectpl/defectpl
pip install -e .
```

### Usage

Following is an example usage with the data stored in `tests/data` for NV center in diamond.
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

![Intensity vs Phonon Energy](./docs/plots/intensity_vs_photon_energy.png)

![S_omega_pHR_loc_rat_vs_penergy.png](./docs/plots/S_omega_pHR_loc_rat_vs_penergy.png)

![S_omega_pHR_ipr_vs_penergy.png](./docs/plots/S_omega_pHR_ipr_vs_penergy.png)

![S_omega_pHR_vs_penergy.png](./docs/plots/S_omega_pHR_vs_penergy.png)

![ipr_vs_phonon_energy.png](./docs/plots/ipr_vs_phonon_energy.png)

![loc_rat_vs_phonon_energy.png](./docs/plots/loc_rat_vs_phonon_energy.png)

![phonon_energy_vs_phonon_mode_index.png](./docs/plots/phonon_energy_vs_phonon_mode_index.png)

![pHR_vs_phonon_energy.png](./docs/plots/pHR_vs_phonon_energy.png)

![vibrational_dis_vs_phonon_energy.png](./docs/plots/vibrational_dis_vs_phonon_energy.png)


### Contribution
Contributions are welcome.
Notice a bug let us know. Thanks.

### Author
Main Maintainer: Shibu Meher
