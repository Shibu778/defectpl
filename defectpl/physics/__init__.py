# -*- coding: utf-8 -*-
"""
defectpl.physics — code-agnostic physics engines.

Sub-modules
-----------
lineshape            PL lineshape (Photoluminescence, VibrationalSpectra1D, CCD).
phonon               Phonon data helpers (read_band_yaml, force constants, etc.).
participation_ratio  Electronic participation ratio calculations.
tdm                  Transition Dipole Moment, IPR, and optical properties.
tdm_viz              TDM / IPR plots and wavefunction export.
"""

from defectpl.physics.lineshape import (
    ConfigurationCoordinateDiagram,
    Photoluminescence,
    VibrationalSpectra1D,
)
from defectpl.physics.phonon import read_band_yaml
from defectpl.physics.participation_ratio import (
    ParticipationRatioCalculator,
    compute_participation_ratios,
)
from defectpl.physics.tdm import (
    WavecarReader,
    VaspwaveH5Reader,
    select_bands,
    compute_ipr_band,
    compute_ipr_all,
    compute_ipr_weighted,
    save_ipr_json,
    save_ipr_csv,
    get_zpl,
    get_dQ,
    get_einstein_coefficient,
    get_radiative_lifetime,
    compute_optical_properties,
)
from defectpl.physics.tdm_viz import (
    plot_tdm_heatmap,
    plot_tdm_bubble,
    plot_tdm_components,
    plot_tdm_kpoint_strip,
    plot_tdm_absorption,
    plot_tdm_dashboard,
    plot_ipr_scatter,
    plot_ipr_bar,
    plot_ipr_kpoint_heatmap,
    save_wfc_vasp,
    save_wfc_vesta,
)

__all__ = [
    # Lineshape
    "ConfigurationCoordinateDiagram",
    "ParticipationRatioCalculator",
    "Photoluminescence",
    "VibrationalSpectra1D",
    "compute_participation_ratios",
    "read_band_yaml",
    # TDM — readers
    "WavecarReader",
    "VaspwaveH5Reader",
    # TDM — band selection
    "select_bands",
    # TDM — IPR
    "compute_ipr_band",
    "compute_ipr_all",
    "compute_ipr_weighted",
    "save_ipr_json",
    "save_ipr_csv",
    # TDM — optical properties
    "get_zpl",
    "get_dQ",
    "get_einstein_coefficient",
    "get_radiative_lifetime",
    "compute_optical_properties",
    # TDM — visualisation
    "plot_tdm_heatmap",
    "plot_tdm_bubble",
    "plot_tdm_components",
    "plot_tdm_kpoint_strip",
    "plot_tdm_absorption",
    "plot_tdm_dashboard",
    "plot_ipr_scatter",
    "plot_ipr_bar",
    "plot_ipr_kpoint_heatmap",
    "save_wfc_vasp",
    "save_wfc_vesta",
]
