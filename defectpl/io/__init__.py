# -*- coding: utf-8 -*-
"""
defectpl.io — code-specific I/O readers.

Sub-modules
-----------
base      Protocol definitions (PhononReader, ElectronicReader).
vasp      VASP reader implementation (requires pymatgen for most operations).
wavecar   Standalone VASP file readers (WAVECAR, IBZKPT, OSZICAR, OUTCAR, POSCAR).
qe        Quantum ESPRESSO stub (not yet implemented).
abinit    ABINIT stub (not yet implemented).
cp2k      CP2K stub (not yet implemented).
"""

from defectpl.io.base import ElectronicReader, PhononReader
from defectpl.io.wavecar import (
    open_text,
    open_wavecar,
    read_ibzkpt_weights,
    read_oszicar,
    read_outcar_energy,
    read_outcar_fermi,
    read_poscar,
    get_total_energy,
    get_fermi_level,
    get_structure,
)

__all__ = [
    "ElectronicReader",
    "PhononReader",
    "open_text",
    "open_wavecar",
    "read_ibzkpt_weights",
    "read_oszicar",
    "read_outcar_energy",
    "read_outcar_fermi",
    "read_poscar",
    "get_total_energy",
    "get_fermi_level",
    "get_structure",
]
