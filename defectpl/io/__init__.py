# -*- coding: utf-8 -*-
"""
defectpl.io — code-specific I/O readers.

Sub-modules
-----------
base      Protocol definitions (PhononReader, ElectronicReader).
vasp      VASP reader implementation (requires pymatgen for most operations).
qe        Quantum ESPRESSO stub (not yet implemented).
abinit    ABINIT stub (not yet implemented).
cp2k      CP2K stub (not yet implemented).
"""

from defectpl.io.base import ElectronicReader, PhononReader

__all__ = [
    "ElectronicReader",
    "PhononReader",
]
