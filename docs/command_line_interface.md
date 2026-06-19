# Command Line Interface (CLI) Manual: `defectpl`

The `defectpl` suite features a unified command-line application layer managed via `click`. This utility handles tasks ranging from raw VASP force parsing and phonon symmetry reduction to full multi-mode photoluminescence lineshape synthesis, temperature-dependent PL/absorption spectra, and Configuration Coordinate Diagram (CCD) mapping.

---

## 1. Global Application Entry Point

Once installed, the root command hook can be executed directly from your terminal:

```bash
defectpl --help
```

The global `--verbose` / `-v` flag prints the full Python traceback on any error instead of a condensed message:

```bash
defectpl -v pl displacement ...
```

---

## 2. Multi-Mode Photoluminescence (`defectpl pl`)

The `pl` group executes high-level multi-phonon spectral sideband convolutions using the generating-function formalism (Alkauskas 2014; Jin 2021). It exposes three sub-commands.

### A. Displacement Mode (`defectpl pl displacement`)

Evaluates vibronic profiles from converged ground-state and excited-state geometry coordinates (`CONTCAR` structures).

```bash
defectpl pl displacement \
  --band_yaml     ./band.yaml \
  --contcar_gs    ./CONTCAR_gs \
  --contcar_es    ./CONTCAR_es \
  --ezpl          1.95 \
  --gamma         2.0 \
  --temperature   300 \
  --sigma         "3e-3,8e-3" \
  --resolution    1000 \
  --max_energy    5.0 \
  --out_dir       ./output_data/ \
  --json_out      properties.json \
  --plot_all \
  --fig_format    svg \
  --iylim         "0,1.2" \
  --max_freq      120.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `--band_yaml` | `./band.yaml` | Phonopy `band.yaml` file. |
| `--contcar_gs` | `./CONTCAR_gs` | Ground-state equilibrium geometry. |
| `--contcar_es` | `./CONTCAR_es` | Excited-state equilibrium geometry. |
| `--ezpl` | `1.95` | Zero-Phonon Line energy in eV. |
| `--gamma` | `2.0` | Lorentzian ZPL broadening in meV. |
| `--temperature` | `0.0` | Lattice temperature in K. `0` reproduces the T = 0 K limit exactly. |
| `--sigma` | `"6e-3"` | Gaussian broadening in eV. Scalar `"6e-3"` applies uniform broadening; two comma-separated values `"3e-3,8e-3"` interpolate linearly from the lowest to the highest phonon frequency (frequency-dependent broadening, Jin 2021). |
| `--resolution` | `1000` | Spectral grid density in points per eV. |
| `--max_energy` | `5.0` | Upper energy axis limit in eV. |
| `--json_out` | — | Path to serialize the `Photoluminescence` object as a Monty JSON file for later use. |
| `--plot_all` | off | Generate all 16 standard diagnostic plots and write them to `--out_dir`. |
| `--out_dir` | `./` | Output directory for plots. |
| `--fig_format` | `pdf` | Figure format: `pdf`, `png`, `svg`. |
| `--iylim` | — | Y-axis limits for the intensity plot, e.g. `"0,1.2"`. |
| `--max_freq` | — | Upper frequency cut-off for mode/S(ω) plots in meV. |

After a successful run the command prints a short summary:

```
Photoluminescence engine data properties calculated successfully.
  HR factor      : 3.4521
  DW factor      : 0.0318
  Temperature    : 300.0 K
  C_total        : 0.1842
```

### B. Force Mode (`defectpl pl force`)

Evaluates vibronic coupling via vertical forces on atoms from unrelaxed, fixed-geometry electronic manifolds (`OUTCAR` inputs). Accepts the same calculation parameters as displacement mode.

```bash
defectpl pl force \
  --band_yaml     ./band.yaml \
  --outcar_gs     ./OUTCAR_gs \
  --outcar_es     ./OUTCAR_es \
  --ezpl          1.95 \
  --gamma         2.0 \
  --temperature   300 \
  --sigma         "6e-3" \
  --json_out      force_properties.json \
  --plot_all \
  --fig_format    png
```

| Option | Default | Description |
|--------|---------|-------------|
| `--outcar_gs` | `./OUTCAR_gs` | VASP OUTCAR for the ground-state forces. |
| `--outcar_es` | `./OUTCAR_es` | VASP OUTCAR for the excited-state vertical forces. |

All other options (`--ezpl`, `--gamma`, `--temperature`, `--sigma`, `--resolution`, `--max_energy`, `--json_out`, `--plot_all`, `--out_dir`, `--fig_format`, `--iylim`, `--max_freq`) are identical to displacement mode.

### C. Restore from JSON (`defectpl pl from-json`)

Restores a previously saved `Photoluminescence` JSON and regenerates all 16 diagnostic plots without re-running any phonon calculation.

```bash
defectpl pl from-json properties.json \
  --out_dir  ./figs/ \
  --fig_format png \
  --max_freq   120.0
```

| Option | Default | Description |
|--------|---------|-------------|
| `JSON_FILE` | *(required)* | Path to a Monty-serialized `Photoluminescence` JSON. |
| `--out_dir` | `./` | Directory to write the regenerated plots. |
| `--fig_format` | `pdf` | Figure format. |
| `--iylim` | — | Y-axis limits for the intensity plot. |
| `--max_freq` | — | Upper frequency cut-off for mode plots in meV. |

---

## 3. Photoabsorption (`defectpl absorption`)

The `absorption` group computes multi-phonon photoabsorption lineshapes using the
generating-function formalism with **excited-state phonons**.  The `--band_yaml` passed
to every `absorption` sub-command must be obtained from a phonopy run on the
**excited-state geometry** — this is the key physics distinction from `defectpl pl`,
which uses ground-state phonons.

### A. Displacement Mode (`defectpl absorption displacement`)

```bash
defectpl absorption displacement \
  --band_yaml     ./band_es.yaml \
  --contcar_gs    ./CONTCAR_gs \
  --contcar_es    ./CONTCAR_es \
  --ezpl          1.95 \
  --gamma         2.0 \
  --temperature   300 \
  --sigma         "6e-3" \
  --resolution    1000 \
  --max_energy    5.0 \
  --out_dir       ./output_data/ \
  --json_out      abs_properties.json \
  --plot_all \
  --fig_format    pdf
```

| Option | Default | Description |
|--------|---------|-------------|
| `--band_yaml` | `./band.yaml` | **Excited-state** phonopy `band.yaml` file (phonopy run at ES geometry). |
| `--contcar_gs` | `./CONTCAR_gs` | Ground-state equilibrium geometry. |
| `--contcar_es` | `./CONTCAR_es` | Excited-state equilibrium geometry. |

All other options are identical to `defectpl pl displacement`.

### B. Force Mode (`defectpl absorption force`)

```bash
defectpl absorption force \
  --band_yaml     ./band_es.yaml \
  --outcar_gs     ./OUTCAR_gs_at_es_geom \
  --outcar_es     ./OUTCAR_es_at_es_geom \
  --ezpl          1.95 \
  --json_out      abs_force.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--band_yaml` | `./band.yaml` | **Excited-state** phonopy `band.yaml`. |
| `--outcar_gs` | `./OUTCAR_gs` | OUTCAR at ES geometry with GS charge state (for force difference). |
| `--outcar_es` | `./OUTCAR_es` | OUTCAR at ES geometry with ES charge state. |

### C. Restore from JSON (`defectpl absorption from-json`)

```bash
defectpl absorption from-json abs_properties.json \
  --out_dir  ./figs/ \
  --fig_format png
```

---

## 4. PL + Absorption Overlay (`defectpl overlay`)

Loads a `Photoluminescence` JSON (computed with GS phonons) and a `Photoabsorption`
JSON (computed with ES phonons) and renders both spectra on a shared energy axis.

```bash
defectpl overlay \
  --pl    pl.json \
  --abs   abs.json \
  --out_dir ./figs/ \
  --fmt   pdf
```

| Option | Default | Description |
|--------|---------|-------------|
| `--pl` | *(required)* | Path to a serialized `Photoluminescence` JSON. |
| `--abs` | *(required)* | Path to a serialized `Photoabsorption` JSON. |
| `--out_dir` | `./` | Output directory for the overlay figure. |
| `--fmt` | `pdf` | Figure format: `pdf`, `png`, `svg`. |
| `--iylim` | — | Y-axis limits for the overlay plot (e.g. `"0,1.2"`). |

---

## 5. Standalone Plotting (`defectpl plot`)

Deserializes a saved `properties.json` and renders any individual figure or the full set, without re-running the generating function.

```bash
defectpl plot ./properties.json --type intensity
defectpl plot ./properties.json --type all --out_dir figs/ --fmt png
defectpl plot ./properties.json --type absorption --fmt pdf
defectpl plot ./properties.json --type nk
```

**Argument:** Path to a Monty JSON file containing a valid `Photoluminescence` or
`Photoabsorption` object.

| Option | Default | Description |
|--------|---------|-------------|
| `--type`, `-t` | *(required)* | Plot to render — see table below. |
| `--out_dir` | `./` | Output directory for the generated figure. |
| `--fmt` | `pdf` | Figure format: `pdf`, `png`, `svg`. |
| `--iylim` | — | Y-axis limits for the intensity plot (e.g. `"0,1.2"`). |
| `--max_freq` | — | Upper phonon-frequency cut-off for mode/S(omega) plots in meV. |

### Available plot types

| `--type` | Required JSON type | Output file | Description |
|----------|--------------------|-------------|-------------|
| `mode` | either | `penergy_vs_pmode.*` | Phonon energy vs mode index scatter. |
| `ipr` | either | `ipr_vs_penergy.*` | Traditional IPR vs phonon energy. |
| `ipr_alkauskas` | either | `ipr_alkauskas_vs_penergy.*` | Alkauskas-convention IPR (range [1, N]) vs phonon energy. |
| `loc_ratio` | either | `loc_rat_vs_penergy.*` | Localization ratio β_k = N × IPR vs phonon energy. |
| `qk` | either | `qk_vs_penergy.*` | Mode displacement q_k vs phonon energy. |
| `hr_factor` | either | `HR_factor_vs_penergy.*` | Partial HR factor S_k vs phonon energy. |
| `s_omega` | either | `S_omega_vs_penergy.*` | Broadened spectral density S(ω). |
| `s_omega_sk` | either | `S_omega_Sks_vs_penergy.*` | S(ω) line + S_k scatter (dual-axis). |
| `s_omega_locrat` | either | `S_omega_HRf_loc_rat_vs_penergy.*` | S(ω) + S_k scatter coloured by localization ratio. |
| `s_omega_ipr` | either | `S_omega_HRf_ipr_vs_penergy.*` | S(ω) + S_k scatter coloured by traditional IPR. |
| `s_omega_ipr_alkauskas` | either | `S_omega_HRf_ipr_alkauskas_vs_penergy.*` | S(ω) + S_k scatter coloured by Alkauskas IPR. |
| `nk` | either | `nk_vs_penergy.*` | Bose-Einstein phonon occupation n̄_k(T) vs phonon energy. |
| `c_omega` | either | `C_omega_vs_penergy.*` | Thermal spectral density C(ω, T). Zero at T = 0. |
| `intensity` | `Photoluminescence` only | `intensity_vs_penergy.*` | Normalised PL emission spectrum L(ħω). Raises an error if a `Photoabsorption` JSON is provided. |
| `absorption` | `Photoabsorption` only | `absorption_vs_penergy.*` | Normalised absorption spectrum α(ħω). Raises an error if a `Photoluminescence` JSON is provided. |
| `all` | either | all of the above (applicable) | Generate every figure valid for the given JSON type. |

> **Note:** To overlay PL and absorption on a single plot, use `defectpl overlay` (Section 4)
> rather than `defectpl plot -t pl_absorption`.  The `pl_absorption` type has been removed
> because PL and absorption now come from separate JSON files (different phonon inputs).

---

## 6. Generalized Displacements (`defectpl dq`)

Calculates the mass-weighted configuration coordinate vector offset (ΔQ) across any two matching structural geometry supercells.

```bash
defectpl dq ./CONTCAR_GS ./CONTCAR_ES --format json --out delta_q.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `structure1` | *(required)* | Path to the first structure file. |
| `structure2` | *(required)* | Path to the second structure file. |
| `--format`, `-f` | `plain` | Output format: `plain` (raw float) or `json`. |
| `--out`, `-o` | — | Optional file path to write the displacement metadata. |

---

## 7. Analytical 1D Lineshapes (`defectpl spectra1d`)

Simulates a decoupled 1D displaced-distorted harmonic oscillator spectrum without assuming identical Hessians (ω₁ ≠ ω₂). Boltzmann-weighted Franck–Condon overlaps are summed explicitly over vibrational quantum numbers.

```bash
defectpl spectra1d \
  --ezpl       2.60 \
  --w1         35.75 \
  --w2         41.56 \
  --dq_val     1.5476 \
  --temp       300.0 \
  --e0         1.8 \
  --de         0.001 \
  --points     5000 \
  --nn1        22 \
  --nn2        52 \
  --plot \
  --save_prefix tracking_1d
```

| Option | Default | Description |
|--------|---------|-------------|
| `--ezpl` | *(required)* | ZPL transition energy in eV. |
| `--w1` | *(required)* | Ground-state effective vibrational frequency in meV. |
| `--w2` | *(required)* | Excited-state effective vibrational frequency in meV. |
| `--dq_val` | *(required)* | Configuration coordinate shift ΔQ in amu½·Å. |
| `--temp` | `300.0` | Temperature in K (sets Boltzmann weights). |
| `--e0` | `0.0` | Energy grid starting point in eV. |
| `--de` | `0.001` | Energy grid step in eV. |
| `--points` | `5000` | Number of energy grid points. |
| `--nn1` | `22` | Maximum ground-state vibrational quantum number. |
| `--nn2` | `52` | Maximum excited-state vibrational quantum number. |
| `--plot` | off | Save a PDF plot of the normalised lineshape. |
| `--save_prefix` | `vibrational_1d` | Prefix for `*_overlap.json` and `*_lineshape.json` output files. |

---

## 8. Configuration Coordinate Diagrams (`defectpl setup-ccd` / `analyze-ccd`)

Automates linear structure interpolation tasks across structural potential surfaces to map the classical harmonic potential energy curves.

### A. Initialization (`defectpl setup-ccd`)

Generates interpolated structural task directories scaled across a displacement grid.

```bash
defectpl setup-ccd \
  --gs      ./CONTCAR_GS \
  --es      ./CONTCAR_ES \
  --tmpl_gs ./template_GS_dir/ \
  --tmpl_es ./template_ES_dir/ \
  --out_dir ./ccd_workspace/ \
  --steps   "-0.2,0.0,0.2,0.4,0.6,0.8,1.0,1.2"
```

### B. Curvature Analysis (`defectpl analyze-ccd`)

Parses completed calculations to fit parabolic potential energy surfaces and report effective phonon frequencies.

```bash
defectpl analyze-ccd \
  --gs      ./CONTCAR_GS \
  --es      ./CONTCAR_ES \
  --gs_runs "./run_0/vasprun.xml ./run_1/vasprun.xml" \
  --es_runs "./run_0/vasprun.xml ./run_1/vasprun.xml" \
  --de      1.95 \
  --save_plot ccd_fit.pdf
```

---

## 9. Comparative Benchmark Suites (`defectpl compare-json` / `compare-yaml`)

Compiles and plots multiple calculated datasets side-by-side to benchmark modelling parameters.

### A. JSON Spectra Comparisons (`defectpl compare-json`)

```bash
defectpl compare-json \
  --files   "run1/properties.json run2/properties.json" \
  --legends "Functional-A,Functional-B" \
  --xmin 1.2 --xmax 2.2 \
  --out_dir ./comparisons/
```

### B. Raw Phonopy Input Comparisons (`defectpl compare-yaml`)

Evaluate lineshape variations against different phonon datasets without running separate pipeline loops.

```bash
defectpl compare-yaml \
  --yamls "phonon_90_atoms/band.yaml phonon_216_atoms/band.yaml" \
  --gs    ./CONTCAR_GS \
  --es    ./CONTCAR_ES \
  --ezpl  1.945 \
  --xmin  1.0 \
  --xmax  2.5
```

---

## 10. Phonon and Lattice Utilities

Exposes administrative shortcuts for mapping Gamma-point lattice frequencies within your processing directories.

### A. Extract Force Constants (`defectpl phonon-fc`)

Parses a `vasprun.xml` file and writes Phonopy force constants.

```bash
defectpl phonon-fc ./vasprun.xml --hdf5
```

### B. Phonon Irreducible Representations (`defectpl phonon-symm`)

Identifies point-group irreducible representation labels for Gamma-point modes.

```bash
defectpl phonon-symm --poscar ./POSCAR --fc ./FORCE_CONSTANTS --dim "1 1 1"
```

### C. Compute `band.yaml` (`defectpl phonon-band`)

Evaluates eigenvalues at the Gamma point from a `FORCE_CONSTANTS` file and writes `band.yaml`.

```bash
defectpl phonon-band --poscar ./POSCAR --fc ./FORCE_CONSTANTS --dim "1 1 1" --out band.yaml
```

### D. Re-serialize Phonon Data (`defectpl phonon-parse`)

Converts a `band.yaml` to eV and writes an MSONable JSON data model.

```bash
defectpl phonon-parse ./band.yaml --json_out parsed_phonons.json
```

---

## 11. Kohn-Sham Level Visualization (`defectpl ksplot`)

Parses a VASP `EIGENVAL` file to resolve degeneracies, handle spin polarization, and plot single-particle defect states within the bandgap.

```bash
defectpl ksplot ./EIGENVAL \
  --vbm     9.6747 \
  --cbm     13.7934 \
  --espan   1.5 \
  --kidx    0 \
  --out_img bandgap_levels.png \
  --out_json defect_states.json
```

| Option | Default | Description |
|--------|---------|-------------|
| `--vbm` | *(required)* | Valence Band Maximum energy in eV. |
| `--cbm` | *(required)* | Conduction Band Minimum energy in eV. |
| `--espan` | `1.0` | Energy padding above/below band edges in eV. |
| `--kidx` | `0` | 0-based k-point index. |
| `--out_img` | `ks_plot.png` | Output image path. |
| `--out_json` | — | Optional path to export the KS data as JSON. |

---

## 12. Participation Ratio (`defectpl pr`)

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
| `--entry`, `-e` | `defect_entry.json` | Path to `defect_entry.json`. |
| `--dsi`, `-s` | auto-detect | Path to `defect_structure_info.json`. |
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

Walks all immediate subdirectories of `--dir`, runs `pr calc` in each one that contains `PROCAR` + `defect_entry.json`, and writes a combined CSV.

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
| `--vbm` | — | VBM energy in eV (orange dotted vertical line, energy mode). |
| `--cbm` | — | CBM energy in eV (green dotted vertical line, energy mode). |
| `--emin` | — | Lower energy filter (eV). |
| `--emax` | — | Upper energy filter (eV). |
| `--kpt` | `0` | 0-based k-point index. |
| `--out`, `-o` | auto | Output image (`pr_energy.png` or `pr_band.png`). |
| `--title` | defect name | Plot title. |

Filled markers = occupied (occ ≥ 0.5); open = empty. Spin channels: blue = spin up, red = spin down.

### G. KS level diagram with P-ratio colour code (`defectpl pr ksplot`)

Renders the standard Kohn-Sham level diagram where each horizontal bar is coloured by the P-ratio or IPR of that state. A colorbar is added on the right.

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
