# Tutorial: PL Spectrum — Displacement Mode

**Use this mode when you have** relaxed ground-state and excited-state geometries (`CONTCAR_gs`,
`CONTCAR_es`) **and** Gamma-point phonons (`band.yaml`) of the defect supercell.

---

## 1. DFT setup

Run three VASP calculations in the defect supercell:

| Job | Description | Key INCAR settings |
|-----|-------------|-------------------|
| `relax_gs/` | Ground-state geometry relaxation | `IBRION = 2; NSW = 200; ISPIN = 2` |
| `relax_es/` | Excited-state geometry relaxation | As above; promote one electron manually (constrained occupation or CDFT) |
| `phonon/` | Gamma-point phonons | `IBRION = 8; NSW = 1` (DFPT) |

!!! tip
    Use `EDIFF = 1e-8` for tight convergence in the phonon run.

---

## 2. Extract phonons

```bash
cd phonon/
defectpl phonon-fc vasprun.xml        # writes FORCE_CONSTANTS

defectpl phonon-band \
    --poscar  POSCAR \
    --fc      FORCE_CONSTANTS \
    --dim     "1 1 1" \             # use your actual supercell dimension
    --out     band.yaml
```

---

## 3. Compute the PL lineshape

### CLI (simplest)

```bash
defectpl pl displacement \
    --band_yaml   phonon/band.yaml \
    --contcar_gs  relax_gs/CONTCAR \
    --contcar_es  relax_es/CONTCAR \
    --ezpl        1.945 \
    --gamma       2.0 \
    --json_out    pl.json \
    --plot_all \
    --fig_format  png
```

### Python API

```python
from pathlib import Path
from pymatgen.core import Structure
from defectpl.phonon import read_band_yaml
from defectpl.io.vasp import calc_dR
from defectpl.defectpl import Photoluminescence
from monty.serialization import dumpfn

# Load structures and phonon data
gs = Structure.from_file("relax_gs/CONTCAR")
es = Structure.from_file("relax_es/CONTCAR")
frequencies, eigenvectors, masses = read_band_yaml("phonon/band.yaml")

# PBC-safe atomic displacement matrix
dR = calc_dR(gs, es)

# Build the PL engine
pl = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    EZPL=1.945,           # eV  — energy difference E_e(Q_e) - E_g(Q_g)
    dR=dR,
    gamma=2.0,            # meV — ZPL broadening
    resolution=1000,      # points per eV
    max_energy=5.0,       # eV  — spectral range
)

print(f"Huang–Rhys factor  S = {pl.HR_factor:.3f}")
print(f"Debye–Waller factor  = {pl.DW_factor:.4f}")
print(f"ΔQ = {pl.delQ:.4f} amu^0.5·Å")
print(f"ΔR = {pl.delR:.4f} Å")

# Save results and generate all plots
pl.generate_plots(out_dir="plots/", fig_format="png")
dumpfn(pl, "pl.json", indent=4)
```

---

## 4. Reload and re-plot

```bash
defectpl plot pl.json --type intensity --fmt pdf
defectpl plot pl.json --type all       --out_dir ./plots/
```

Or in Python:

```python
from monty.serialization import loadfn
pl = loadfn("pl.json")

from defectpl.plot import Plotter
p = Plotter()
p.plot_intensity_vs_penergy(
    A=pl.A_line,
    intensity=pl.intensity,
    out_dir="./plots/",
    fig_format="pdf",
)
```

---

## 5. Output files

| File | Content |
|------|---------|
| `pl.json` | Serialized `Photoluminescence` object (all scalars and arrays) |
| `plots/intensity.png` | Final PL lineshape $L(\hbar\omega)$ |
| `plots/HR_factor.png` | Mode-resolved partial HR factors $S_k$ vs $\omega_k$ |
| `plots/S_omega.png` | Spectral function $S(\omega)$ |
| `plots/ipr.png` | IPR vs phonon frequency |
| `plots/localization_ratio.png` | Localization ratio $\beta_k$ vs frequency |

---

## 6. Typical values for a deep defect

| Quantity | Typical range |
|----------|--------------|
| $S$ (total HR factor) | 1 – 10 |
| $w_\text{ZPL} = e^{-S}$ | 0.0001 – 0.37 |
| $\Delta Q$ | 0.5 – 5 $\sqrt{\text{amu}}\cdot\text{Å}$ |
| Dominant mode IPR | 0.01 – 1 |
