# Command Line Interface (CLI) Manual: `defectpl`

The `defectpl` suite features a unified command-line application layer managed via `click`. This utility handles tasks ranging from raw VASP force parsing and phonon symmetry reduction to full multi-mode photoluminescence lineshape synthesis and Configuration Coordinate Diagram (CCD) mapping.

---

## 1. Global Application entry Point

Once installed, the root command hook can be executed directly from your terminal:

```bash
defectpl --help

```

---

## 2. Multi-Mode Photoluminescence (`defectpl pl`)

The `pl` group executes high-level multi-phonon spectral sideband convolutions. It exposes two sub-commands corresponding to the available physical data-collection frameworks.

### A. Displacement Mode (`defectpl pl displacement`)

Evaluates vibronic profiles from converged ground-state and excited-state geometry coordinates (`CONTCAR` structures).

**Usage and Parameters:**

```bash
defectpl pl displacement \
  --band_yaml ./band.yaml \
  --contcar_gs ./CONTCAR_gs \
  --contcar_es ./CONTCAR_es \
  --ezpl 1.95 \
  --gamma 2.0 \
  --out_dir ./output_data/ \
  --json_out properties.json \
  --plot_all \
  --fig_format svg

```

* `--band_yaml`: Path to the Phonopy structural `band.yaml` file *(Default: `./band.yaml`)*.
* `--contcar_gs`: Ground-state equilibrium geometry structure file *(Default: `./CONTCAR_gs`)*.
* `--contcar_es`: Excited-state equilibrium geometry structure file *(Default: `./CONTCAR_es`)*.
* `--ezpl`: Zero-Phonon Line transition energy threshold in eV *(Default: `1.95`)*.
* `--gamma`: Homogeneous Lorentzian broadening damping metric in meV *(Default: `2.0`)*.
* `--json_out`: Optional path to dump the serialized `Photoluminescence` object state using Monty JSON hooks.
* `--plot_all`: Boolean flag. When invoked, it automatically generates all 10 standard diagnostic plots.
* `--fig_format`: Graphics layout vector standard extension (e.g., `pdf`, `png`, `svg`) *(Default: `pdf`)*.
* `--max_freq`: Frequency range limit override for spectral density charts (in meV).
* `--iylim`: Y-axis limits override passed as a comma-separated string (e.g., `--iylim 0,1.5`).

### B. Force Mode (`defectpl pl force`)

Evaluates vibronic coupling paths via vertical forces acting on atoms from unrelaxed, fixed-geometry electronic manifolds (`OUTCAR` inputs).

**Usage and Parameters:**

```bash
defectpl pl force \
  --band_yaml ./band.yaml \
  --outcar_gs ./OUTCAR_gs \
  --outcar_es ./OUTCAR_es \
  --ezpl 1.95 \
  --gamma 2.0 \
  --json_out force_properties.json \
  --plot_all \
  --fig_format png

```

* Parameters match the displacement mode command, substituting `--contcar_*` path options for VASP `--outcar_gs` and `--outcar_es` force vectors. Spatial metrics like $\Delta Q$ and $\Delta R$ are omitted automatically in the backend during generation sweeps to enforce force-mode consistency.

---

## 3. Standalone Plotting (`defectpl plot`)

Deserializes previously cached `properties.json` data files to regenerate or customize visual layouts without re-executing the underlying Fourier transforms.

**Usage:**

```bash
defectpl plot ./properties.json --type intensity --out_dir ./plots/ --fmt png

```

* **Argument:** Path to a serialized JSON file containing a valid MSONable `Photoluminescence` configuration.
* `--type` (`-t`): Specific graphic layout configuration target. Must be one of: `intensity`, `mode`, `partial_energy`, or `all`.
* `--fmt`: Graphic layout format selection (`pdf`, `png`, `svg`).

---

## 4. Generalized Displacements (`defectpl dq`)

Calculates the raw, mass-weighted configuration coordinate vector offset ($\Delta Q$) across any two matching structural geometry supercells.

**Usage:**

```bash
defectpl dq ./CONTCAR_GS ./CONTCAR_ES --format json --out delta_q.json

```

* **Arguments:** Paths to the initial (`structure1`) and final (`structure2`) VASP structure files.
* `--format` (`-f`): Output representation format. Select `plain` (prints raw float string to standard output stream) or `json`.
* `--out` (`-o`): Optional file path to dump the compiled structural displacement metadata records.

---

## 5. Analytical 1D Lineshapes (`defectpl spectra1d`)

Simulates a decoupled 1D displaced-distorted harmonic oscillator spectrum without assuming identical Hessians ($\omega_1 \neq \omega_2$).

**Usage:**

```bash
defectpl spectra1d \
  --ezpl 2.60 \
  --w1 35.75 \
  --w2 41.56 \
  --dq_val 1.5476 \
  --temp 300.0 \
  --points 5000 \
  --plot \
  --save_prefix tracking_1d

```

* `--ezpl`: ZPL transition energy center in eV *(Required)*.
* `--w1`: Ground-state effective vibrational frequency in meV *(Required)*.
* `--w2`: Excited-state effective vibrational frequency in meV *(Required)*.
* `--dq_val`: Configuration coordinate shift $\Delta Q$ in $\text{amu}^{1/2}\cdot\text{Å}$ *(Required)*.
* `--temp`: Physical target evaluation temperature in Kelvin *(Default: `300.0`)*.
* `--points`: Linear sampling integration rows count dimension for the grid array *(Default: `5000`)*.
* `--plot`: Flag enabling immediate rendering of the output PDF visual spectrum sideband chart.
* `--save_prefix`: Prefix appended to generated data output records (`*_overlap.json`, `*_lineshape.json`).

---

## 6. Configuration Coordinate Diagrams (`defectpl setup-ccd` / `analyze-ccd`)

Automates linear structure interpolation tasks across structural potential surfaces to map the classical harmonic potential energy curves.

### A. Initialization Script (`setup-ccd`)

Generates structural interpolation arrays scaling across specific configuration displacement grid targets.

```bash
defectpl setup-ccd \
  --gs ./CONTCAR_GS \
  --es ./CONTCAR_ES \
  --tmpl_gs ./template_GS_dir/ \
  --tmpl_es ./template_ES_dir/ \
  --out_dir ./ccd_workspace/ \
  --steps "-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2"

```

### B. Curvature Analysis Script (`analyze-ccd`)

Parses completed calculations across the interpolated task trees to fit potential curvatures and resolve corresponding frequencies.

```bash
defectpl analyze-ccd \
  --gs ./CONTCAR_GS \
  --es ./CONTCAR_ES \
  --gs_runs "./run_0/vasprun.xml ./run_1/vasprun.xml" \
  --es_runs "./run_0/vasprun.xml ./run_1/vasprun.xml" \
  --de 1.95 \
  --save_plot ccd_fit.pdf

```

---

## 7. Comparative Benchmark Suites (`defectpl compare-json` / `compare-yaml`)

Compiles and plots multiple calculated datasets side-by-side to benchmark modeling parameters.

### A. JSON Spectra Comparisons (`compare-json`)

```bash
defectpl compare-json \
  --files "run1/properties.json run2/properties.json" \
  --legends "Functional-A,Functional-B" \
  --xmin 1.2 --xmax 2.2 \
  --out_dir ./comparisons/

```

### B. Raw Phonopy Input Comparisons (`compare-yaml`)

Useful for evaluating lineshape variations against different phonon datasets without running separate pipeline loops manually.

```bash
defectpl compare-yaml \
  --yamls "phonon_90_atoms/band.yaml phonon_216_atoms/band.yaml" \
  --gs ./CONTCAR_GS --es ./CONTCAR_ES \
  --ezpl 1.945 --xmin 1.0 --xmax 2.5

```

---

## 8. Phonon and Lattice Utilities

Exposes administrative shortcuts for mapping Gamma-point lattice frequencies directly within your processing directories.

### A. Extract Force Constants (`phonon-fc`)

Parses VASP XML traces to generate standard Phonopy force tracking files.

```bash
defectpl phonon-fc ./vasprun.xml --hdf5

```

### B. Extract Phonon Irreducible Representations (`phonon-symm`)

Identifies the point-group irreducible representation labels for calculated Gamma-point modes.

```bash
defectpl phonon-symm --poscar ./POSCAR --fc ./FORCE_CONSTANTS --dim "1 1 1"

```

### C. Compile Band Target Structural Outputs (`phonon-band`)

Evaluates eigenvalues at the Gamma point using standard force constants matrices, converting outputs natively into eV energy metrics.

```bash
defectpl phonon-band --poscar ./POSCAR --fc ./FORCE_CONSTANTS --dim "1 1 1" --out band.yaml

```

### D. Re-serialize Band Properties (`phonon-parse`)

Converts standard Phonopy configurations to eV and writes them to an MSONable JSON data model file.

```bash
defectpl phonon-parse ./band.yaml --json_out parsed_phonons.json

```

---

## 9. Kohn-Sham Level Visualization (`defectpl ksplot`)

Parses a VASP `EIGENVAL` file to screen degeneracies, handle spin-polarization tracks, and plot single-particle electronic defect states within the bandgap.

```bash
defectpl ksplot ./EIGENVAL \
  --vbm 9.6747 \
  --cbm 13.7934 \
  --espan 1.5 \
  --kidx 0 \
  --out_img bandgap_levels.png \
  --out_json defect_states.json

```

* `--vbm`: Calculated energy value of the Valence Band Maximum in eV *(Required)*.
* `--cbm`: Calculated energy value of the Conduction Band Minimum in eV *(Required)*.
* `--espan`: Energy canvas buffer padding depth evaluated above and below the band edges in eV *(Default: `1.0`)*.
* `--kidx`: Sequence array index tracking target k-point coordinates *(Default: `0`)*.

---

## 10. `pr` — Participation Ratio

The `pr` group computes electronic-state **P-ratio** and **IPR** from VASP PROCAR data, and provides utilities to generate all prerequisite JSON files without pydefect.

```bash
defectpl pr --help
```

### A. Generate `defect_entry.json` (`defectpl pr make-entry`)

```bash
# Manual (provide centre directly)
defectpl pr make-entry --name Va_O1_2 --center 0.5,0.5,0.5

# Auto-detect vacancy from perfect vs defect structure
defectpl pr make-entry \
    --name    Va_O1_2 \
    --perfect ../perfect/POSCAR \
    --defect  CONTCAR
```

| Option | Default | Description |
|--------|---------|-------------|
| `--name`, `-n` | *(required)* | Defect label, e.g. `Va_O1_2`. |
| `--center`, `-c` | — | Fractional coordinates as `x,y,z`. |
| `--perfect`, `-P` | — | Perfect supercell POSCAR for auto-detection. |
| `--defect`, `-D` | — | Defect supercell POSCAR for auto-detection. |
| `--site-tol` | `0.5` | Site-matching tolerance in Å. |
| `--out`, `-o` | `defect_entry.json` | Output path. |

### B. Generate `defect_structure_info.json` (`defectpl pr make-dsi`)

```bash
defectpl pr make-dsi \
    --poscar CONTCAR \
    --center 0.5,0.5,0.5 \
    --cutoff 3.5
```

| Option | Default | Description |
|--------|---------|-------------|
| `--poscar`, `-p` | *(required)* | POSCAR/CONTCAR of the defect supercell. |
| `--center`, `-c` | *(required)* | Fractional coordinates of the defect centre. |
| `--cutoff`, `-r` | `3.5` | Neighbour search radius in Å. |
| `--out`, `-o` | `defect_structure_info.json` | Output path. |

### C. Run the P-ratio calculation (`defectpl pr calc`)

```bash
# Minimal (auto-detect all file paths in current directory)
defectpl pr calc

# Explicit paths
defectpl pr calc \
    --procar  Va_O1_2/PROCAR \
    --entry   Va_O1_2/defect_entry.json \
    --dsi     Va_O1_2/defect_structure_info.json \
    --out     Va_O1_2/
```

| Option | Default | Description |
|--------|---------|-------------|
| `--procar`, `-p` | `PROCAR` | VASP PROCAR file. Needs `LORBIT=11` or `12`. |
| `--entry`, `-e` | `defect_entry.json` | Path to defect_entry.json. |
| `--dsi`, `-s` | auto-detect | Path to defect_structure_info.json. |
| `--poscar` | auto-detect | POSCAR/CONTCAR for distance-based fallback. |
| `--cutoff`, `-c` | `3.5` | Neighbour cut-off radius in Å (fallback only). |
| `--out`, `-o` | `.` | Output directory. |
| `--top` | `15` | Number of most-localised states to print. |
| `--no-csv` | off | Skip writing the flat CSV file. |
| `--native-procar` | off | Use built-in parser instead of pymatgen's. |

### D. Batch processing (`defectpl pr batch`)

```bash
defectpl pr batch --dir defects/Va_O1/ --cutoff 4.0
```

Walks all immediate subdirectories of `--dir`, runs `pr calc` in each one
that contains `PROCAR` + `defect_entry.json`, and writes a combined CSV.

### E. Inspect results (`defectpl pr summary` / `defectpl pr top`)

```bash
# Pretty-print summary table
defectpl pr summary Va_O1_2/participation_ratio.json --top 20

# List top-5 most localised states by IPR
defectpl pr top Va_O1_2/participation_ratio.json --n 5 --metric ipr
```

### F. Scatter plot (`defectpl pr plot`)

Plot P-ratio or IPR against **energy** (default) or **band index**.

```bash
# P-ratio vs energy (default)
defectpl pr plot Va_O1_2/participation_ratio.json

# P-ratio vs band index
defectpl pr plot Va_O1_2/participation_ratio.json --xaxis band

# IPR vs energy with gap markers and energy window
defectpl pr plot Va_O1_2/participation_ratio.json \
    --metric ipr --xaxis energy \
    --vbm 5.20 --cbm 8.10 \
    --emin 4.0 --emax 9.5 \
    --threshold 0.05 --out ipr.pdf
```

| Option | Default | Description |
|--------|---------|-------------|
| `JSON_FILE` | `participation_ratio.json` | Input results file. |
| `--xaxis`, `-x` | `energy` | X-axis: `energy` (eV) or `band` (index). |
| `--metric`, `-m` | `p_ratio` | Y-axis quantity: `p_ratio` or `ipr`. |
| `--threshold`, `-t` | `0.2` | Dashed horizontal threshold line. |
| `--vbm` | — | VBM energy in eV — orange dotted vertical line (energy mode). |
| `--cbm` | — | CBM energy in eV — green dotted vertical line (energy mode). |
| `--emin` | — | Lower energy filter (eV). |
| `--emax` | — | Upper energy filter (eV). |
| `--kpt` | `0` | 0-based k-point index. |
| `--out`, `-o` | auto | Output image (default: `pr_energy.png` or `pr_band.png`). |
| `--title` | defect name | Plot title. |

Filled markers = occupied (occ ≥ 0.5); open = empty.
Spin channels: blue = spin ↑, red = spin ↓.

### G. KS level diagram with P-ratio colour code (`defectpl pr ksplot`)

Renders the standard Kohn-Sham level diagram where each horizontal bar
is coloured by the P-ratio or IPR of that state (instead of plain black).
A colorbar is added on the right.

```bash
defectpl pr ksplot \
    --eigenval EIGENVAL \
    --pr-json  Va_O1_2/participation_ratio.json \
    --vbm 5.20 --cbm 8.10 \
    --espan 1.5 \
    --metric p_ratio \
    --cmap RdYlGn_r \
    --out ks_pr_plot.png
```

| Option | Default | Description |
|--------|---------|-------------|
| `--eigenval`, `-e` | `EIGENVAL` | VASP EIGENVAL file. |
| `--pr-json` | `participation_ratio.json` | PR results from `pr calc`. |
| `--vbm` | *(required)* | Valence Band Maximum energy (eV). |
| `--cbm` | *(required)* | Conduction Band Minimum energy (eV). |
| `--espan` | `1.0` | Energy padding above/below VBM/CBM (eV). |
| `--metric`, `-m` | `p_ratio` | Colour-coding metric: `p_ratio` or `ipr`. |
| `--cmap` | `RdYlGn_r` | Matplotlib colormap (green = delocalized, red = localized). |
| `--vmin` | `0.0` | Colormap lower bound. |
| `--vmax` | `1.0` | Colormap upper bound. |
| `--kidx` | `0` | 0-based k-point index. |
| `--out`, `-o` | `ks_pr_plot.png` | Output image file. |
| `--title` | defect name | Plot title. |
