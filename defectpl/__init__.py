# -*- coding: utf-8 -*-
"""
defectpl — unified package for optical properties of point defects.

Public API
----------
Core physics engines::

    from defectpl import Photoluminescence, VibrationalSpectra1D
    from defectpl import ConfigurationCoordinateDiagram

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
