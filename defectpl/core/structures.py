# -*- coding: utf-8 -*-
"""
Code-agnostic data containers for phonon and electronic-structure data.

These dataclasses are the exchange format between the code-specific I/O
layer (``defectpl.io.*``) and the physics engine (``defectpl.physics.*``).
They carry no DFT-code-specific logic and require no optional dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np


@dataclass
class PhononData:
    """
    Code-agnostic container for phonon data at the Γ-point.

    Attributes
    ----------
    frequencies : np.ndarray, shape (nmodes,)
        Phonon frequencies in eV.
    eigenvectors : np.ndarray, shape (nmodes, natoms, 3)
        Mass-normalised eigenvectors (dimensionless).
    masses : np.ndarray, shape (natoms,)
        Atomic masses in amu.
    natoms : int
        Number of atoms in the supercell.
    nmodes : int
        Number of phonon modes (usually 3 × natoms).
    meta : dict
        Arbitrary metadata (source file, code name, etc.).
    """

    frequencies: np.ndarray
    eigenvectors: np.ndarray
    masses: np.ndarray
    natoms: int
    nmodes: int
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frequencies = np.asarray(self.frequencies, dtype=float)
        self.eigenvectors = np.asarray(self.eigenvectors, dtype=float)
        self.masses = np.asarray(self.masses, dtype=float)

    @property
    def as_tuple(self) -> tuple:
        """Return ``(frequencies, eigenvectors, masses)`` for legacy callers."""
        return (self.frequencies, self.eigenvectors, self.masses)


@dataclass
class EigenvalData:
    """
    Code-agnostic container for spin-resolved eigenvalue data.

    Attributes
    ----------
    up, down : list of (energy, occupancy) pairs
        Spin-up and spin-down eigenvalues at a single k-point.
    homo_up_idx, homo_down_idx : int
        Zero-based band indices of the HOMO for each spin channel.
    lumo_up_idx, lumo_down_idx : int
        Zero-based band indices of the LUMO for each spin channel.
    homo_up, homo_down : float
        HOMO eigenvalue (eV) for each spin channel.
    lumo_up, lumo_down : float
        LUMO eigenvalue (eV) for each spin channel.
    hl_gap_up, hl_gap_down : float
        HOMO–LUMO gap (eV) for each spin channel.
    nelect : float
        Number of electrons.
    nbands : int
        Total number of bands.
    nkpt : int
        Total number of k-points in the calculation.
    selected_kpoint : list
        ``[k_idx, [kx, ky, kz]]`` of the selected k-point.
    spin_multiplicity : float
        2S+1 derived from the HOMO indices.
    """

    up: List
    down: List
    homo_up_idx: int
    homo_down_idx: int
    lumo_up_idx: int
    lumo_down_idx: int
    homo_up: float
    homo_down: float
    lumo_up: float
    lumo_down: float
    hl_gap_up: float
    hl_gap_down: float
    nelect: float
    nbands: int
    nkpt: int
    selected_kpoint: List
    spin_multiplicity: float
