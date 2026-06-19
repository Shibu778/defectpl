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

__version__ = "0.3.0"

__all__ = [
    # Physics engines
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
]
