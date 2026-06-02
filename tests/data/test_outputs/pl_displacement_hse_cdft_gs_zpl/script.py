# -*- coding: utf-8 -*-
"""
Execution script to parse structural displacements, load Gamma-point phonons,
and execute the core DefectPL engine to compute photoluminescence properties.
"""

from pathlib import Path
from pymatgen.core import Structure
from monty.serialization import dumpfn

from defectpl.phonon import read_band_yaml
# FIX: Import from vasp_wrapper instead of the legacy utils module
from defectpl.vasp_wrapper import calc_dR
from defectpl.defectpl import Photoluminescence
from defectpl.utils import extract_important_properties

# =====================================================================
# Configuration Paths, Tolerances, and Variables Setup
# =====================================================================
band_yaml_path = Path("../../cdft_calc/hse06/gs_dfpt/band.yaml")
contcar_gs_path = Path("../../cdft_calc/hse06/gs/CONTCAR")
contcar_es_path = Path("../../cdft_calc/hse06/zpl/CONTCAR")
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