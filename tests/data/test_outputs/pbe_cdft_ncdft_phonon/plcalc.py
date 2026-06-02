# -*- coding: utf-8 -*-
"""
Execution script to parse structural displacements, load Gamma-point phonons,
and execute the core DefectPL engine to compute photoluminescence properties.
"""

import json
from pathlib import Path
from pymatgen.core import Structure
from defectpl.phonon import read_band_yaml
from defectpl.utils import calc_dR
from defectpl.defectpl import Photoluminescence

# =====================================================================
# Configuration Paths, Tolerances, and Variables Setup
# =====================================================================
# Legacy path references kept for development tracking history:
band_yaml_path = Path("../../ground_dfpt/band.yaml")
contcar_gs_path = Path("../../pbe/gs/CONTCAR")
contcar_es_path = Path("../../pbe/zpl/CONTCAR")
outdir = Path(".")

ezpl = 1.95        # Zero-phonon line energy threshold in eV
gamma = 2.0        # Homogeneous ZPL broadening scale factor
fig_format = "png" # Figure export format type string

# =====================================================================
# Data Processing Pipeline Execution
# =====================================================================
# Ensure output data directory wrapper boundaries exist prior to export actions
outdir.mkdir(parents=True, exist_ok=True)

# 1. Parse ground and excited configuration geometries into Pymatgen Structures
struct_gs = Structure.from_file(str(contcar_gs_path))
struct_es = Structure.from_file(str(contcar_es_path))

# 2. Extract periodic boundary condition (PBC) safe displacement vectors matrix
dR = calc_dR(struct_gs, struct_es)

# 3. Read Gamma-point phonon values natively via safe yaml parser lines
frequencies, eigenvectors, masses = read_band_yaml(band_yaml_path)

# 4. Instantiate core photoluminescence dynamics computation engine loops
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
pl_engine.generate_plots(out_dir=outdir, fig_format=fig_format)

# =====================================================================
# Data Serialization Output Dump
# =====================================================================
# Extract standard serialization dictionary layout properties payload
properties = pl_engine.as_dict()

# Export clean JSON file tracking all state-space parameters
output_json_path = outdir / "properties.json"
with open(output_json_path, "w", encoding="utf-8") as f:
    json.dump(properties, f, indent=4)

print(f"Data state records successfully exported to: {output_json_path}")