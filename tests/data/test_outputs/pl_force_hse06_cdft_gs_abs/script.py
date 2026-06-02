# -*- coding: utf-8 -*-
"""
Execution script to parse structural forces, load Gamma-point phonons,
and execute the core DefectPL engine in FORCE MODE to compute PL properties.
"""

from pathlib import Path
from monty.serialization import dumpfn

from defectpl.phonon import read_band_yaml
from defectpl.defectpl import Photoluminescence
from defectpl.vasp_wrapper import prepare_dF_files
from defectpl.utils import extract_important_properties

# =====================================================================
# Configuration Paths, Tolerances, and Variables Setup
# =====================================================================
band_yaml_path = Path("../../cdft_calc/hse06/gs_dfpt/band.yaml")
outcar_gs_path = Path("../../cdft_calc/hse06/gs/OUTCAR")
outcar_es_path = Path("../../cdft_calc/hse06/abs/OUTCAR")
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

# Extract important properties
print("Extracting important properties for summary output...")
extract_important_properties(pl_engine)

# =====================================================================
# Data Serialization Output Dump
# =====================================================================
# Export clean JSON tracking all state-space parameters using Monty serialization
output_json_path = outdir / "properties.json"
dumpfn(pl_engine, str(output_json_path), indent=4)

print(f"Data state records successfully exported to: {output_json_path}")