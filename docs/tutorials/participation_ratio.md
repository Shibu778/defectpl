# Tutorial: Participation Ratio

This tutorial walks through computing, inspecting, and plotting the **electronic-state P-ratio
and IPR** for a vacancy defect.

---

## 1. VASP setup

Add to your INCAR:

```
LORBIT = 11
```

This makes VASP write the PROCAR file with full site-projected wavefunction data.
Run a standard SCF calculation on the defect supercell.

---

## 2. Generate prerequisite JSON files

### Option A — provide the defect centre directly

If you already know the fractional coordinates of the defect site:

```bash
defectpl pr make-entry \
    --name   Va_O1_2 \
    --center 0.5,0.5,0.5

defectpl pr make-dsi \
    --poscar  CONTCAR \
    --center  0.5,0.5,0.5 \
    --cutoff  3.5
```

### Option B — auto-detect from structures

```bash
defectpl pr make-entry \
    --name    Va_O1_2 \
    --perfect ../perfect/POSCAR \
    --defect  CONTCAR

defectpl pr make-dsi \
    --poscar  CONTCAR \
    --center  0.5,0.5,0.5 \   # use the center printed by make-entry
    --cutoff  3.5
```

### Python equivalent

```python
from defectpl.defect_utils import make_defect_entry, make_defect_structure_info

make_defect_entry(name="Va_O1_2", center=[0.5, 0.5, 0.5])
make_defect_structure_info(
    poscar="CONTCAR",
    defect_center_frac=[0.5, 0.5, 0.5],
    cutoff_radius=3.5,
)
```

---

## 3. Run the calculation

```bash
defectpl pr calc               # auto-detects all files in current directory
```

Explicit paths:

```bash
defectpl pr calc \
    --procar  PROCAR \
    --entry   defect_entry.json \
    --dsi     defect_structure_info.json \
    --out     .
```

Output: `participation_ratio.json` and `participation_ratio_summary.csv`.

---

## 4. Inspect results

```bash
# Print top-10 most localized states (by P-ratio)
defectpl pr top participation_ratio.json --n 10

# Full summary table
defectpl pr summary participation_ratio.json --top 20
```

---

## 5. Plot

### P-ratio vs energy

```bash
defectpl pr plot participation_ratio.json \
    --metric p_ratio \
    --vbm 5.20 --cbm 8.10 \
    --threshold 0.2 \
    --out pr_energy.png
```

### P-ratio vs band index

```bash
defectpl pr plot participation_ratio.json --xaxis band
```

### Kohn–Sham level diagram with P-ratio colour code

```bash
defectpl pr ksplot \
    --eigenval EIGENVAL \
    --pr-json  participation_ratio.json \
    --vbm 5.20 --cbm 8.10 \
    --metric   p_ratio \
    --cmap     RdYlGn_r \
    --out      ks_pr.png
```

### Python API

```python
import json
from defectpl.participation_ratio import plot_pr_vs_energy, plot_pr_vs_band_index
from defectpl.io.vasp import read_eigenval_file
from defectpl.ks_analysis import extract_ksplot_data, plot_ks_with_pr

with open("participation_ratio.json") as f:
    result = json.load(f)

# P-ratio vs energy
ax = plot_pr_vs_energy(
    result, metric="p_ratio",
    vbm=5.20, cbm=8.10, threshold=0.2,
    out="pr_energy.pdf",
)

# P-ratio vs band index
ax = plot_pr_vs_band_index(result, metric="p_ratio", out="pr_band.pdf")

# KS plot coloured by P-ratio
ev_data = read_eigenval_file("EIGENVAL", k_idx=0)
ks_data = extract_ksplot_data(ev_data, vbm=5.20, cbm=8.10, espan=1.5)
plot_ks_with_pr(ks_data, result, metric="p_ratio", output_filename="ks_pr.png")
```

---

## 6. Batch processing

Process all charge states of a defect at once:

```bash
defectpl pr batch --dir defects/Va_O1/ --cutoff 3.5
```

This walks all immediate subdirectories of `defects/Va_O1/` that contain `PROCAR` +
`defect_entry.json`, runs `pr calc` in each, and writes a combined CSV to the parent directory.
