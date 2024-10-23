# DefectPl
A unified package to calculate and plot optical properties of point defects in insulators and semiconductors.

#### Purpose of the Package
The purpose of this package is to calculate the intensity of photoluminescence from point defects in solids with method described in New J. Phys. 16 (2014) 073026. It also calculates and plot other relevant quantities like partial Huang Rhys factor, inverse participation ratio etc.

### Getting Started
The package can be found in pypi. You can install it using `pip`.

#### Installation

```bash
pip install defectpl
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


### API
 TODO

### Contribution
Contributions are welcome.
Notice a bug let us know. Thanks.

### Author
Main Maintainer: Shibu Meher
