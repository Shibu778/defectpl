# Getting Started with DefectPL

This guide takes you from installation to your first photoluminescence (PL) spectrum in a few
minutes.  Each section builds on the previous one — work through them in order the first time.

---

## 1. Installation

### Minimal install (no DFT parser)

```bash
pip install defectpl
```

This gives you the core math (Huang-Rhys factors, lineshape generation, 1D spectra) and the
participation-ratio native PROCAR parser.  No pymatgen or phonopy required.

### VASP + phonopy workflow (recommended)

```bash
pip install "defectpl[vasp,phonon]"
```

This adds pymatgen (for reading VASP files) and phonopy (for phonon calculations).

### Everything

```bash
pip install "defectpl[all]"
```

### From source

```bash
git clone https://github.com/Shibu778/defectpl.git
cd defectpl
pip install -e ".[all]"
```

---

## 2. Required VASP settings

The table below shows which INCAR keywords each workflow needs.

| Workflow | Required INCAR keywords |
|----------|------------------------|
| PL spectrum (displacement mode) | Standard relaxation settings |
| PL spectrum (force mode) | `NSW = 1`, `IBRION = -1` (single-point on fixed geometry) |
| Phonon modes | `IBRION = 8` (or use phonopy finite-displacement supercells) |
| Participation ratio | `LORBIT = 11` (or `12`) |
| Kohn-Sham plot | None beyond a standard SCF |

---

## 3. Your first PL spectrum (displacement mode)

This is the most common workflow.  You need:

- `band.yaml` — Gamma-point phonons from phonopy
- `CONTCAR_gs` — ground-state relaxed geometry
- `CONTCAR_es` — excited-state relaxed geometry

### Step 3a — Generate phonons with phonopy

```bash
# Inside your phonon supercell directory
defectpl phonon-fc vasprun.xml          # writes FORCE_CONSTANTS
defectpl phonon-band \
    --poscar POSCAR \
    --fc    FORCE_CONSTANTS \
    --dim   "1 1 1" \
    --out   band.yaml
```

### Step 3b — Compute the PL lineshape

```bash
defectpl pl displacement \
    --band_yaml   band.yaml \
    --contcar_gs  CONTCAR_gs \
    --contcar_es  CONTCAR_es \
    --ezpl        1.945 \
    --gamma       2.0 \
    --json_out    pl.json \
    --plot_all \
    --fig_format  png
```

This writes `pl.json` (serialized results) and generates all standard diagnostic plots.

Key options:

| Option | What it controls |
|--------|-----------------|
| `--ezpl` | Zero-phonon line energy (eV) — from the energy difference between excited and ground state |
| `--gamma` | Lorentzian broadening of the ZPL (meV) |
| `--plot_all` | Generates all 10 standard plots automatically |
| `--fig_format` | `png`, `pdf`, or `svg` |

### Step 3c — Re-plot from saved JSON

```bash
defectpl plot pl.json --type intensity --fmt png
defectpl plot pl.json --type mode      --fmt pdf
defectpl plot pl.json --type all       --out_dir ./plots/
```

---

## 4. PL spectrum from forces (force mode)

Use this when you have single-point forces on the unrelaxed geometry rather than two relaxed
structures.

```bash
defectpl pl force \
    --band_yaml   band.yaml \
    --outcar_gs   OUTCAR_gs \
    --outcar_es   OUTCAR_es \
    --ezpl        1.945 \
    --gamma       2.0 \
    --json_out    pl_force.json \
    --plot_all
```

`OUTCAR_gs` and `OUTCAR_es` must be single-point calculations (`NSW = 1`, `IBRION = -1`) of
the opposite-state geometry — i.e., the excited-state geometry evaluated at the ground-state
electronic structure, and vice versa.

---

## 5. Participation ratio (electronic state localization)

The participation ratio (P-ratio) tells you how much of each Kohn-Sham wavefunction is
concentrated on the defect neighbourhood.  Range: 0 (bulk-like) → 1 (fully localized).

### Step 5a — Add to INCAR

```
LORBIT = 11
```

Re-run VASP.  This generates the `PROCAR` file with site-projected wavefunction data.

### Step 5b — Create prerequisite JSON files

```bash
# Option A: provide defect centre directly
defectpl pr make-entry --name Va_O1_2 --center 0.5,0.5,0.5

# Option B: auto-detect from perfect vs defect structure
defectpl pr make-entry \
    --name    Va_O1_2 \
    --perfect ../perfect/POSCAR \
    --defect  CONTCAR

# Build neighbour list
defectpl pr make-dsi \
    --poscar  CONTCAR \
    --center  0.5,0.5,0.5 \
    --cutoff  3.5
```

### Step 5c — Run the calculation

```bash
defectpl pr calc
```

All paths are auto-detected.  Output: `participation_ratio.json` + `participation_ratio_summary.csv`.

### Step 5d — Inspect and plot

```bash
# Top-10 most localised states
defectpl pr top participation_ratio.json --n 10

# P-ratio vs energy scatter plot
defectpl pr plot participation_ratio.json --vbm 5.20 --cbm 8.10

# P-ratio vs band index
defectpl pr plot participation_ratio.json --xaxis band

# Kohn-Sham level diagram coloured by P-ratio
defectpl pr ksplot \
    --eigenval EIGENVAL \
    --pr-json  participation_ratio.json \
    --vbm 5.20 --cbm 8.10
```

---

## 6. Kohn-Sham level plot

Visualise defect states in the bandgap without computing the participation ratio:

```bash
defectpl ksplot EIGENVAL \
    --vbm 9.6747 \
    --cbm 13.7934 \
    --espan 1.5 \
    --out_img defect_levels.png
```

---

## 7. Configuration Coordinate Diagram (CCD)

### Generate interpolated structures

```bash
defectpl setup-ccd \
    --gs      CONTCAR_GS \
    --es      CONTCAR_ES \
    --tmpl_gs template_gs/ \
    --tmpl_es template_es/ \
    --out_dir ccd_workspace/ \
    --steps   "-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2"
```

### Fit curvatures after running VASP on each point

```bash
defectpl analyze-ccd \
    --gs      CONTCAR_GS \
    --es      CONTCAR_ES \
    --gs_runs "run_0/vasprun.xml run_1/vasprun.xml" \
    --es_runs "run_0/vasprun.xml run_1/vasprun.xml" \
    --de      1.945 \
    --save_plot ccd_fit.pdf
```

---

## 8. 1D analytical lineshape

For a quick single-mode spectrum (no phonon calculation needed):

```bash
defectpl spectra1d \
    --ezpl    2.60 \
    --w1      35.75 \
    --w2      41.56 \
    --dq_val  1.5476 \
    --temp    300.0 \
    --plot
```

---

## 9. Python API quick reference

```python
from defectpl.phonon import read_band_yaml
from defectpl.defectpl import Photoluminescence

# Load phonon data
phonon_data = read_band_yaml("band.yaml")

# Build the PL object from displacement data
from pymatgen.core import Structure
gs = Structure.from_file("CONTCAR_gs")
es = Structure.from_file("CONTCAR_es")
import numpy as np
dR = np.array([site_es.coords - site_gs.coords
               for site_gs, site_es in zip(gs, es)])

pl = Photoluminescence(
    frequencies  = phonon_data.frequencies,
    eigenvectors = phonon_data.eigenvectors,
    masses       = phonon_data.masses,
    EZPL         = 1.945,
    dR           = dR,
    gamma        = 2.0,
)

print(f"HR factor S = {pl.HR_factor:.3f}")
print(f"ΔQ = {pl.delQ:.4f} amu^0.5·Å")

# Save / reload
import json
from monty.json import MontyEncoder, MontyDecoder
with open("pl.json", "w") as f:
    json.dump(pl.as_dict(), f, cls=MontyEncoder)

with open("pl.json") as f:
    pl2 = Photoluminescence.from_dict(json.load(f, cls=MontyDecoder))
```

---

## 10. Where to go next

| Task | Documentation |
|------|--------------|
| Full CLI reference | [command_line_interface.md](command_line_interface.md) |
| P-ratio theory and API | [participation_ratio.md](participation_ratio.md) |
| Physical background | [theory.md](theory.md) |
| Design roadmap | [design_review.md](design_review.md) |
