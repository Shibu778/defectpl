# -*- coding: utf-8 -*-
"""
defectpl.vasp — backward-compatible re-export shim.

All VASP I/O code now lives in :mod:`defectpl.io.vasp`.
This module is kept for backward compatibility and re-exports everything
from there so that existing ``from defectpl.vasp import ...`` calls
continue to work unchanged.
"""

from defectpl.io.vasp import (
    OutcarParser,
    check_outcar_convergence,
    get_final_structure_and_forces_from_outcar,
    get_first_structure_and_forces_from_outcar,
    get_nions,
    get_species_and_index_map,
    get_spin_multiplicity,
    get_structures_and_forces,
    read_eigenval_file,
)

__all__ = [
    "OutcarParser",
    "check_outcar_convergence",
    "get_final_structure_and_forces_from_outcar",
    "get_first_structure_and_forces_from_outcar",
    "get_nions",
    "get_species_and_index_map",
    "get_spin_multiplicity",
    "get_structures_and_forces",
    "read_eigenval_file",
]
