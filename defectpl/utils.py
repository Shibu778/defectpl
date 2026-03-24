# -*- coding: utf-8-*-
# Utilities for the defectpl package
from pymatgen.core import Structure
from pymatgen.util.coord import pbc_shortest_vectors
import numpy as np


def calc_deltaQ(structure1, structure2):
    """Calculate the difference in atomic positions between two structures.

    Args:
        structure1: path to first structure file
        structure2: path to second structure file
        Note: structure1 and structure2 must have the same number and types of atoms
    Returns:
        Change in configuration coordinate delta Q calculated as follows
    """
    struct1 = Structure.from_file(structure1)
    struct2 = Structure.from_file(structure2)
    length = len(struct1.sites)
    lattice = struct1.lattice
    # Calculate the dR
    dR = np.vstack(
        [
            pbc_shortest_vectors(
                lattice, struct1.frac_coords[i], struct2.frac_coords[i]
            )
            for i in range(length)
        ]
    ).reshape(length, 3)
    mlist = []
    for site in struct1.sites:
        mlist.append(site.specie.atomic_mass)
    mlist = np.array(mlist)

    return np.sqrt(np.sum(mlist * np.sum(dR**2, axis=1)))
