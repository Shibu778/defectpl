# -*- coding: utf-8 -*-
"""
Documentation generation script: Rehydrates a precalculated DefectPL dataset 
from properties.json using Monty, summarizes scalar physical properties, 
and exports publication-ready diagnostic plots.
"""

from pathlib import Path
from monty.serialization import loadfn
from monty.json import MontyDecoder

# Import core model engines and standalone plotting pipeline utilities
from defectpl.defectpl import Photoluminescence, VibrationalSpectra1D
from defectpl.plot import Plotter
from defectpl.utils import extract_important_properties

# =====================================================================
# Configuration Paths, Tolerances, and Variables Setup
# =====================================================================
path = Path("/home/user/Project/ht_SiN/benchmark/NV_diamond_PL/codes/defectpl/tests/data/test_outputs/pl_force_pbe_ncdft_gs_abs")
json_input_path = path / "properties.json"
out_dir = Path(".")
fig_format = "svg"

# Verify precalculated data matrix state exists before initializing load sequences
if not json_input_path.exists():
    raise FileNotFoundError(
        f"Could not locate '{json_input_path}'. Run your calculation pipeline "
        f"script first to export the state metadata records via dumpfn."
    )

# =====================================================================
# 1. Rehydrate Core Engine Instance & Save Scalar Physical Properties
# =====================================================================
print(f"Loading cached state properties from {json_input_path}...")
# Rehydrate the true Photoluminescence object instance instead of a raw python dict
pl_engine = loadfn(str(json_input_path), cls=MontyDecoder)

# Run property extraction summary file out via imported utility function
prop_output_file = out_dir / "important_properties.txt"
extract_important_properties(pl_engine, filename=str(prop_output_file))
print(f"Extracted important scalar properties to {prop_output_file} for quick reference.")

# =====================================================================
# 2. Sequential Granular Generation of All Gallery Plots
# =====================================================================
print(f"Generating full sequential suite of diagnostic plots as .{fig_format} files...")

# Initialize standalone plotting instance 
plotter = Plotter()

# Fix configuration scope limits by fetching directly from the rehydrated engine parameters
iplot_xlim = (max(0.0, pl_engine.EZPL - 2.0), pl_engine.EZPL + 1.0)
max_freq = None  # Capping limits tracker for frequency ranges
freq_limit = (max_freq / 1000.0) if max_freq else None
iylim = None     # Intensity vertical y-limit limits override

# --- Track A: Lattice Dynamics & Structural Confinements ---

# 1. penergy_vs_pmode.svg
plotter.plot_penergy_vs_pmode(
    frequencies=pl_engine.frequencies, 
    plot=False, 
    out_dir=out_dir, 
    fig_format=fig_format
)

# 2. ipr_vs_penergy.svg
plotter.plot_ipr_vs_penergy(
    frequencies=pl_engine.frequencies, 
    iprs=pl_engine.iprs, 
    plot=False, 
    out_dir=out_dir, 
    fig_format=fig_format
)

# 3. loc_rat_vs_penergy.svg
plotter.plot_loc_rat_vs_penergy(
    frequencies=pl_engine.frequencies, 
    localization_ratio=pl_engine.localization_ratio, 
    plot=False, 
    out_dir=out_dir, 
    fig_format=fig_format
)

# 4. qk_vs_penergy.svg
plotter.plot_qk_vs_penergy(
    frequencies=pl_engine.frequencies, 
    qks=pl_engine.qks, 
    plot=False, 
    out_dir=out_dir, 
    fig_format=fig_format
)

# 5. HR_factor_vs_penergy.svg
plotter.plot_HR_factor_vs_penergy(
    frequencies=pl_engine.frequencies, 
    Sks=pl_engine.Sks, 
    plot=False, 
    out_dir=out_dir, 
    fig_format=fig_format
)

# --- Track B: Electronic-Vibrational Coupling Spectra Functionals ---

# 6. S_omega_vs_penergy.svg
plotter.plot_S_omega_vs_penergy(
    frequencies=pl_engine.frequencies,
    S_omega=pl_engine.S_omega,
    omega_range=pl_engine.omega_range,
    plot=False,
    out_dir=out_dir,
    max_freq=freq_limit,
    fig_format=fig_format,
    figsize=(4, 4)
)

# 7. S_omega_Sks_vs_penergy.svg
plotter.plot_S_omega_Sks_vs_penergy(
    frequencies=pl_engine.frequencies,
    S_omega=pl_engine.S_omega,
    omega_range=pl_engine.omega_range,
    Sks=pl_engine.Sks,
    plot=False,
    out_dir=out_dir,
    max_freq=freq_limit,
    fig_format=fig_format,
    figsize=(4, 4)
)

# 8. S_omega_HRf_loc_rat_vs_penergy.svg
plotter.plot_S_omega_Sks_Loc_rat_vs_penergy(
    frequencies=pl_engine.frequencies,
    S_omega=pl_engine.S_omega,
    omega_range=pl_engine.omega_range,
    Sks=pl_engine.Sks,
    localization_ratio=pl_engine.localization_ratio,
    plot=False,
    out_dir=out_dir,
    max_freq=freq_limit,
    fig_format=fig_format,
    figsize=(4, 4)
)

# 9. S_omega_HRf_ipr_vs_penergy.svg
plotter.plot_S_omega_Sks_ipr_vs_penergy(
    frequencies=pl_engine.frequencies,
    S_omega=pl_engine.S_omega,
    omega_range=pl_engine.omega_range,
    Sks=pl_engine.Sks,
    iprs=pl_engine.iprs,
    plot=False,
    out_dir=out_dir,
    max_freq=freq_limit,
    fig_format=fig_format,
    figsize=(4, 4)
)

# --- Track C: Final Convoluted Multi-Phonon Luminescence Profiles ---

# 10. intensity_vs_penergy.svg
plotter.plot_intensity_vs_penergy(
    frequencies=pl_engine.frequencies,
    I=pl_engine.intensity,
    resolution=pl_engine.resolution,
    xlim=iplot_xlim,
    plot=False,
    out_dir=out_dir,
    iylim=iylim,
    fig_format=fig_format
)

# =====================================================================
# 3. 1D Harmonic Approximation Lineshape Analysis
# =====================================================================
print("Generating 1D effective coordinate verification lineshapes...")

# 11. one_d_lineshape.svg
ligo = VibrationalSpectra1D(
    EZPL=2.6, 
    w1_meV=35.75, 
    w2_meV=41.56, 
    DQ=1.5476, 
    T=300, 
    E0=1.2, 
    dE=0.001, 
    M=1800
)
ligo.compute_lineshape()
ligo.plot_lineshape(save_file=str(out_dir / "one_d_lineshape.svg"), figsize=(4, 4))

print("All documentation visualization graphics successfully updated.")