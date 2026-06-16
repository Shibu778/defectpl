# Using `defectpl`: A Comprehensive Python Interface Tutorial

This tutorial provides step-by-step instructions on utilizing the Python interface of the `defectpl` package to execute photoluminescence (PL) lineshape calculations from first-principles data. It guides you through setting up raw structural parameters, executing calculations in both **Displacement Mode** and **Force Mode** using authentic script paths, performing 1D effective coordinate lineshape analysis, and exporting publication-ready visuals.

---

## 1. Prerequisites and Structural Setup

The core calculation module requires specific structural arrays extracted from density functional theory (DFT) packages (e.g., VASP, Quantum ESPRESSO). Before instantiating the engine, make sure you have compiled the following variables into `numpy` arrays:

* **`frequencies`**: An array of shape `(nmodes,)` containing the ground-state phonon energies in units of eV.
* **`eigenvectors`**: A displacement tensor of shape `(nmodes, natoms, 3)` describing the normalized atomic displacement matrix vectors.
* **`masses`**: An array of shape `(natoms,)` tracking individual atomic weights in atomic mass units (AMU).
* **`EZPL`**: A floating-point number defining the zero-phonon line (ZPL) electronic transition energy gap in eV.

---

## 2. Multi-Mode Analysis (Displacement Mode vs. Force Mode)

The `Photoluminescence` core engine accommodates two primary operational configurations depending on your defect cell parameters.

### A. Displacement Mode Execution
Use this mode when you possess full structural convergence for both the equilibrium ground-state (`CONTCAR_GS`) and excited-state (`CONTCAR_ES`) Cartesian coordinates. The script extracts periodic boundary condition (PBC) safe displacement vectors `dR` corresponding to the difference vector ($\mathbf{R}_{e} - \mathbf{R}_{g}$) in units of Å.

```python
# -*- coding: utf-8 -*-
"""
Execution script to parse structural displacements, load Gamma-point phonons,
and execute the core DefectPL engine to compute photoluminescence properties.
"""

from pathlib import Path
from pymatgen.core import Structure
from monty.serialization import dumpfn

from defectpl.phonon import read_band_yaml
from defectpl.io.vasp import calc_dR
from defectpl.defectpl import Photoluminescence

# =====================================================================
# Configuration Paths, Tolerances, and Variables Setup
# =====================================================================
band_yaml_path = Path("../../old_data/band.yaml")
contcar_gs_path = Path("../../old_data/CONTCAR_GS")
contcar_es_path = Path("../../old_data/CONTCAR_ES")
outdir = Path("./")

ezpl = 1.95        # Zero-phonon line energy threshold in eV
gamma = 2.0        # Homogeneous ZPL broadening scale factor
fig_format = "png" # Figure export format type string

# =====================================================================
# Data Processing Pipeline Execution
# =====================================================================
# Ensure output data directory wrapper boundaries exist prior to export actions
outdir.mkdir(parents=True, exist_ok=True)

# 1. Parse ground and excited configuration geometries into Pymatgen Structures
print("Parsing atomic structural files...")
struct_gs = Structure.from_file(str(contcar_gs_path))
struct_es = Structure.from_file(str(contcar_es_path))

# 2. Extract periodic boundary condition (PBC) safe displacement vectors matrix
dR = calc_dR(struct_gs, struct_es)

# 3. Read Gamma-point phonon values natively via safe yaml parser lines
frequencies, eigenvectors, masses = read_band_yaml(band_yaml_path)
print(f"DEBUG: Eigenvectors shape is {eigenvectors.shape}")

# 4. Instantiate core photoluminescence dynamics computation engine loops
print("Running multi-mode PL physics calculations...")
pl_engine = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    dR=dR,
    EZPL=ezpl,
    gamma=gamma,
    max_energy=5.0,
    sigma=6e-3
)
        
# 5. Dispatch processed arrays to the automated Plotter module to generate graphics
print(f"Generating and exporting diagnostic plots as .{fig_format} files...")
pl_engine.generate_plots(out_dir=outdir, fig_format=fig_format)

# =====================================================================
# Data Serialization Output Dump
# =====================================================================
# Export clean JSON tracking all state-space parameters using Monty serialization
output_json_path = outdir / "properties.json"
dumpfn(pl_engine, str(output_json_path), indent=4)

print(f"Data state records successfully exported to: {output_json_path}")

```

### B. Force Mode Execution

Use this mode if you are simulating dilute limits or large supercells where computing full coordinate relaxation vectors is computationally restrictive. It parses the raw VASP `OUTCAR` outputs to determine a force shift matrix `dF` ($\mathbf{F}_{e} - \mathbf{F}_{g}$ at a fixed reference geometry) in units of eV/Å.

```python
# -*- coding: utf-8 -*-
"""
Execution script to parse structural forces, load Gamma-point phonons,
and execute the core DefectPL engine in FORCE MODE to compute PL properties.
"""

from pathlib import Path
from monty.serialization import dumpfn

from defectpl.phonon import read_band_yaml
from defectpl.defectpl import Photoluminescence
from defectpl.io.vasp import prepare_dF_files

# =====================================================================
# Configuration Paths, Tolerances, and Variables Setup
# =====================================================================
band_yaml_path = Path("./band.yaml")
outcar_gs_path = Path("./OUTCAR_gs")
outcar_es_path = Path("./OUTCAR_es")
outdir = Path(".")

ezpl = 1.95        # Zero-phonon line energy threshold in eV
gamma = 2.0        # Homogeneous ZPL broadening scale factor
fig_format = "png" # Figure export format type string

# =====================================================================
# Data Processing Pipeline Execution
# =====================================================================
# Ensure output data directory wrapper boundaries exist prior to export actions
outdir.mkdir(parents=True, exist_ok=True)

# 1. Parse Gamma-point phonon values natively via safe yaml parser lines
print("Parsing phonon configuration parameters...")
frequencies, eigenvectors, masses = read_band_yaml(band_yaml_path)

# 2. Extract force shift matrix (dF = F_excited - F_ground) using wrapper utilities
print("Extracting vertical force differences from VASP OUTCAR files...")
dF = prepare_dF_files(str(outcar_gs_path), str(outcar_es_path))

# 3. Instantiate core photoluminescence dynamics computation engine loops in FORCE MODE
print("Running multi-mode PL physics calculations...")
pl_engine = Photoluminescence(
    frequencies=frequencies,
    eigenvectors=eigenvectors,
    masses=masses,
    dR=None,       # Explicitly set to None for Force Mode pipeline execution
    dF=dF,         # Force difference matrix vector payload passed dynamically
    EZPL=ezpl,
    gamma=gamma,
    max_energy=5.0,
    sigma=6e-3
)
        
# 4. Dispatch processed arrays to the automated Plotter module to generate graphics
print(f"Generating and exporting diagnostic plots as .{fig_format} files...")
pl_engine.generate_plots(out_dir=outdir, fig_format=fig_format)

# =====================================================================
# Data Serialization Output Dump
# =====================================================================
# Export clean JSON tracking all state-space parameters using Monty serialization
output_json_path = outdir / "properties.json"
dumpfn(pl_engine, str(output_json_path), indent=4)

print(f"Data state records successfully exported to: {output_json_path}")

```

---

## 3. High-Throughput Serialization and Properties Extraction

The `Photoluminescence` object inherits from `monty.json.MSONable`, enabling clean state saving and rehydration across persistent cache storage.

### A. Saving and Rehydrating Object State

To process previously calculated results (such as benchmarks located in your test suites), use `loadfn` combined with `MontyDecoder` to reconstruct the true class instance seamlessly:

```python
from monty.serialization import loadfn
from monty.json import MontyDecoder
from pathlib import Path

# Path matching the precalculated benchmarking outputs folder structure
json_input_path = Path("/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/tests/data/test_outputs/pl_from_abs_force/properties.json")

# Rehydrate the true Photoluminescence object instance instead of a fallback dict
pl_engine = loadfn(str(json_input_path), cls=MontyDecoder)

```

### B. Automated Text Properties Extraction

To output a summarized text configuration dashboard of your system's scalar properties without custom formatting loops, call the built-in utility. This tracks calculation run modes dynamically, omitting coordination-based fields (like $\Delta Q$ and $\Delta R$) when evaluating a Force Mode run:

```python
from defectpl.utils import extract_important_properties

# Automatically evaluates engine state flags and outputs structured parameters
prop_output_file = Path("./important_properties.txt")
extract_important_properties(pl_engine, filename=str(prop_output_file))
print(f"Extracted important scalar properties to {prop_output_file} for quick reference.")

```

---

## 4. 1D Harmonic Approximation Lineshape Analysis

When resolving simplified or highly localized systems where the optical transition is dominated by a single effective vibrational mode with distinct ground-state ($\omega_{1}$) and excited-state ($\omega_{2}$) frequencies, use the `VibrationalSpectra1D` model.

```python
from defectpl.defectpl import VibrationalSpectra1D
from pathlib import Path

out_dir = Path(".")

# Configure the 1D Displaced-Distorted Harmonic Oscillator
ligo = VibrationalSpectra1D(
    EZPL=2.6,        # Zero-phonon line transition energy in eV
    w1_meV=35.75,    # Ground state effective phonon energy in meV
    w2_meV=41.56,    # Excited state effective phonon energy in meV
    DQ=1.5476,       # Mass-weighted structural coordinate offset in amu^(1/2)*Å
    T=300,           # Target simulation temperature in Kelvin
    E0=1.2,          # Energy mesh grid baseline start coordinate in eV
    dE=0.001,        # Energy mesh sampling intervals (step resolution) in eV
    M=1800           # Total number of linear grid interpolation points
)

# Compute matrix transitions and convolute with appropriate Gaussian profiles
ligo.compute_lineshape()

# Save the final lineshape visual out to your target directory
ligo.plot_lineshape(save_file=str(out_dir / "one_d_lineshape.svg"), figsize=(4, 4))
print("1D effective coordinate verification lineshapes successfully updated.")

```

---

## 5. Automated Multi-Plot Visualization Generation

`defectpl` features a unified visualization suite capable of plotting lattice configurations and electronic-vibrational couplings simultaneously. You can generate publication-ready plots either by interacting with the `Plotter` class directly or via the parent engine interface.

### A. Comprehensive Automation Method

```python
from pathlib import Path

out_directory = Path("./gallery_plots")
out_directory.mkdir(exist_ok=True)

# Generates all 10 diagnostic tracks (IPR, Mode Energy, HR-Spread, 
# S_omega spectral density profiles, and convoluted final intensities) automatically.
pl_engine.generate_plots(
    out_dir=out_directory, 
    max_freq=None,       # Frequencies range limits (None defaults to automatic scaling)
    fig_format="svg"     # Target output format ("pdf", "png", "svg", etc.)
)

```

### B. Granular Plotting Method

If you require isolated plots or customized graph configurations, interact with the `Plotter` asset independently:

```python
from defectpl.plot import Plotter

plotter = Plotter()

# Isolate the final convoluted multi-phonon intensity line shape
plotter.plot_intensity_vs_penergy(
    frequencies=pl_engine.frequencies,
    I=pl_engine.intensity,
    resolution=pl_engine.resolution,
    xlim=(max(0.0, pl_engine.EZPL - 2.0), pl_engine.EZPL + 1.0),
    plot=False,  # Set to True to trigger immediate matplotlib visualization windows
    out_dir=Path("."),
    iylim=None,  # Intensity vertical limits override
    fig_format="svg"
)

```