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
