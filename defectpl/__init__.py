# -*- coding: utf-8 -*-
"""
defectpl — A unified Python package for the optical properties of point defects in solids.

Public API
----------
Core physics engines::

    from defectpl import Photoluminescence, Photoabsorption, VibrationalSpectra1D
    from defectpl import ConfigurationCoordinateDiagram

``Photoluminescence`` uses **ground-state phonons** and computes the PL emission
spectrum.  ``Photoabsorption`` uses **excited-state phonons** (a phonopy run on
the ES geometry) and computes only the absorption spectrum.

Phonon helpers::

    from defectpl import read_band_yaml

I/O Protocols (for duck-typing custom readers)::

    from defectpl import PhononReader, ElectronicReader

Data containers::

    from defectpl import PhononData, EigenvalData

VASP reader::

    from defectpl.io.vasp import VaspReader

Full sub-package tree
---------------------
defectpl.io.*         — code-specific I/O (VASP implemented; QE/ABINIT/CP2K stubs)
defectpl.core.*       — code-agnostic data containers
defectpl.physics.*    — physics computation engines
"""

from defectpl.defectpl import (
    ConfigurationCoordinateDiagram,
    Photoabsorption,
    Photoluminescence,
    VibrationalSpectra1D,
)
from defectpl.phonon import read_band_yaml
from defectpl.io.base import ElectronicReader, PhononReader
from defectpl.core.structures import EigenvalData, PhononData
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

__version__ = "0.4.0"

__all__ = [
    # PL physics engines
    "ConfigurationCoordinateDiagram",
    "Photoabsorption",
    "Photoluminescence",
    "VibrationalSpectra1D",
    # Phonon helpers
    "read_band_yaml",
    # I/O protocols
    "ElectronicReader",
    "PhononReader",
    # Data containers
    "EigenvalData",
    "PhononData",
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
